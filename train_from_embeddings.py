import contextlib
import datetime
import functools
import json
import math
import pathlib
import queue
import random
import threading
import time
import traceback
from typing import Union, Dict, Optional, Any, List, Tuple, Callable

import diffusers
import jax
import numpy as np
import optax
import smart_open
import tqdm
import transformers
import typer
import wandb
from diffusers.models.attention_flax import FlaxAttention, FlaxGEGLU
from diffusers.models.resnet_flax import FlaxResnetBlock2D
from flax import linen as nn
from flax.jax_utils import replicate
from flax.linen import normalization
from flax.linen.linear import _Conv as Conv, Dense, canonicalize_padding
from flax.training.train_state import TrainState
from jax import lax, numpy as jnp
from jax.experimental.compilation_cache import compilation_cache as cc
from jaxhelper import promote, clip_norm, ema, to_host, dot, if_flag, remat, index, attention, nan_default, map_fn, \
    select, cast, cast_tree, shift, tree_dtype, set_flag
from namedtreemap import named_treemap
from optax import GradientTransformation
from smart_open import open

SCALE_DOWN_CONTEXT = 0.001
WEIGHT_DECAY_FOR_CONTEXT = 0  # .01

if pathlib.Path('/fsx/lucas').exists():
    cc.initialize_cache("/fsx/lucas/cache")
elif pathlib.Path('~').expanduser().exists():
    cc.initialize_cache(f"{str(pathlib.Path('~').expanduser())}/cache")

app = typer.Typer(pretty_exceptions_enable=False)
_UPLOAD_RETRIES = 8
_ACTIVATION = None
_MODEL_PARALLEL_DEVICES = 4
_COMPUTE_DTYPE = jnp.bfloat16
_AXIS = "batch"

if_shuffle = functools.partial(if_flag, "shuffle")


def device_id():
    return lax.axis_index(_AXIS)


def isctx():
    return ((device_id() - _MODEL_PARALLEL_DEVICES // 2) % _MODEL_PARALLEL_DEVICES) != 0


def _comm_fn(fn, permuted, postprocess):
    inputs = cast(permuted, _COMPUTE_DTYPE)
    # TODO: evaluate whether to use scan instead of for (drawback: less unroll)
    # TODO: sum locally rather than psum (bottleneck: currently unknown permutations)
    ys = [cast(fn(shift(inputs, _MODEL_PARALLEL_DEVICES // 2 - i, _AXIS)), jnp.float32)  #
          for i in range(_MODEL_PARALLEL_DEVICES)]
    return cast(postprocess(ys), tree_dtype(permuted))


def comm(fn):
    @if_shuffle(fn)
    @jax.custom_gradient
    def _fn(inp: jax.Array, *args):
        devices = [list(range(j, j + _MODEL_PARALLEL_DEVICES))  #
                   for j in range(0, jax.device_count(), _MODEL_PARALLEL_DEVICES)]

        def _grad(dy: jax.Array):  # double-check this grad is correct
            dys = lax.all_gather(dy, _AXIS, axis_index_groups=devices)
            idx = 0

            def _curried(inp):
                nonlocal idx
                local_dy = index(idx)(dys)
                out, grad_fn = jax.vjp(fn, inp, *args)
                d_inp, *d_args = grad_fn(cast_tree(local_dy, out))
                d_inp = shift(d_inp, idx - _MODEL_PARALLEL_DEVICES // 2, _AXIS)
                idx += 1
                return d_inp, *d_args

            return _comm_fn(_curried, inp, lambda x: jax.tree_util.tree_map(lambda *k: sum(k), *x))

        def _select_sum(ys):
            ys = lax.psum(ys, _AXIS, axis_index_groups=devices)
            return select(device_id() % _MODEL_PARALLEL_DEVICES, ys)

        return _comm_fn(lambda x: fn(x, *args), inp, _select_sum), _grad

    return _fn


_old_attention = FlaxAttention.__call__


def multi_dot(x, *ws):
    return [dot(x, w) for w in ws]


def dot_fn(*args):
    xs = list(set(a for a, _ in args))
    ws = [[w for x, w in args if x is outer] for outer in xs]
    out = {(x, p): o for x, w in zip(xs, ws) for o, p in zip(comm(multi_dot)(x, *w), w)}
    return [out[a] for a in args]


@if_shuffle(_old_attention)
def _new_attention(self: FlaxAttention, hidden_states: jax.Array, context: Optional[jax.Array] = None,
                   deterministic=True):
    context = hidden_states if context is None else context
    ks = [c.scope.param("kernel", c.kernel_init, (f, c.features), c.param_dtype).astype(_COMPUTE_DTYPE) for c, f in (
        (self.query, hidden_states.shape[-1]), (self.key, context.shape[-1]), (self.value, context.shape[-1]),
        (self.proj_attn, self.value.features))]
    pb = self.proj_attn.scope.param('bias', self.proj_attn.bias_init, (self.proj_attn.features,),
                                    self.proj_attn.param_dtype)
    return attention(hidden_states, context, *ks, pb, self.scale, self.heads, dot_fn=dot_fn)


FlaxAttention.__call__ = _new_attention


def _normalize(mdl, x, mean, var, reduction_axes, feature_axes, dtype, param_dtype, epsilon: float, use_bias: bool,
               use_scale: bool, bias_init, scale_init):
    reduction_axes = normalization._canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = normalization._canonicalize_axes(x.ndim, feature_axes)
    stats_shape = list(x.shape)
    for axis in reduction_axes:
        stats_shape[axis] = 1
    mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)
    feature_shape = [1] * x.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
        reduced_feature_shape.append(x.shape[ax])

    if use_scale:
        scale = mdl.param('scale', scale_init, reduced_feature_shape, param_dtype).reshape(feature_shape)
    else:
        scale = 1
    if use_bias:
        bias = mdl.param('bias', bias_init, reduced_feature_shape, param_dtype).reshape(feature_shape)
    else:
        bias = 0

    act = _ACTIVATION  # copy to ensure it's the same value during backward pass

    @remat
    def _fn(x, m, v, s, b):
        original_dtype = x.dtype
        x = x.astype(jnp.float32)
        y = (x - m) * s * lax.rsqrt(v + epsilon) + b
        if act is None:
            return y
        return act(y).astype(original_dtype)

    y = _fn(x, mean, var, scale, bias)
    return jnp.asarray(y, dtype)


normalization._normalize = _normalize


@contextlib.contextmanager
def set_activation(x):
    global _ACTIVATION
    try:
        _ACTIVATION = x
        yield
    finally:
        _ACTIVATION = None


def _new_resnet(self: FlaxResnetBlock2D, hidden_states: jax.Array, temb: jax.Array,
                deterministic: bool = True) -> jax.Array:
    residual = hidden_states
    with set_activation(nn.swish):
        hidden_states = self.norm1(hidden_states)
    hidden_states = self.conv1(hidden_states)

    temb = self.time_emb_proj(nn.swish(temb))
    temb = jnp.expand_dims(jnp.expand_dims(temb, 1), 1)
    hidden_states = hidden_states + temb

    with set_activation(nn.swish):
        hidden_states = self.norm2(hidden_states)
    hidden_states = self.dropout(hidden_states, deterministic)
    hidden_states = self.conv2(hidden_states)

    if self.conv_shortcut is not None:
        residual = self.conv_shortcut(residual)

    return hidden_states + residual


FlaxResnetBlock2D.__call__ = _new_resnet


@remat
def _geglu_fn(s):
    hidden_linear, hidden_gelu = jnp.split(s, 2, axis=2)
    return hidden_linear * nn.gelu(hidden_gelu)


def _new_geglu(self: FlaxGEGLU, hidden_states, deterministic=True):
    return _geglu_fn(self.proj(hidden_states))


FlaxGEGLU.__call__ = _new_geglu

_original_conv_call = Conv.__call__
_original_dense_call = Dense.__call__


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


@if_shuffle(_original_conv_call)
def _wrapped_conv_call(self: Conv, inputs: jax.Array) -> jax.Array:
    kernel_size = tuple(self.kernel_size)
    in_features = jnp.shape(inputs)[-1]
    kernel_shape = kernel_size + (in_features // self.feature_group_count, self.features)
    kernel = self.scope.param("kernel", self.kernel_init, kernel_shape, self.param_dtype)
    use_bias = self.use_bias
    if use_bias:
        bias = self.scope.param("bias", self.bias_init, (self.features,), self.param_dtype)
    else:
        bias = None
    kernel, bias = kernel.astype(self.dtype), bias.astype(self.dtype)

    def maybe_broadcast(x) -> Tuple[int, ...]:
        if x is None:
            # backward compatibility with using None as sentinel for
            # broadcast 1
            x = 1
        if isinstance(x, int):
            return (x,) * len(kernel_size)
        return tuple(x)

    # Combine all input batch dimensions into a single leading batch axis.
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
        input_batch_shape = inputs.shape[:num_batch_dimensions]
        total_batch_size = int(np.prod(input_batch_shape))
        flat_input_shape = (total_batch_size,) + inputs.shape[num_batch_dimensions:]
        inputs = jnp.reshape(inputs, flat_input_shape)

    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    dimension_numbers = _conv_dimension_numbers(inputs.shape)

    if self.mask is not None and self.mask.shape != kernel_shape:
        raise ValueError(f'Mask needs to have the same shape as weights. Shapes are: {self.mask.shape}, '
                         f'{kernel_shape}')

    if self.mask is not None:
        kernel *= self.mask
    padding_lax = canonicalize_padding(self.padding, len(kernel_size))
    inner_padding = "VALID" if padding_lax in ("CIRCULAR", "CAUSAL") else padding_lax

    @comm
    def conv(x, w):
        return self.conv_general_dilated(x, w, strides, inner_padding, lhs_dilation=input_dilation,
                                         rhs_dilation=kernel_dilation, dimension_numbers=dimension_numbers,
                                         feature_group_count=self.feature_group_count, precision=self.precision)

    @remat
    def _outer(inputs: jax.Array, kernel: jax.Array, bias: Optional[jax.Array]) -> jax.Array:
        if padding_lax == 'CIRCULAR':
            kernel_size_dilated = [(k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)]
            zero_pad: List[Tuple[int, int]] = [(0, 0)]
            pads = zero_pad + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] + [(0, 0)]
            inputs = jnp.pad(inputs, pads, mode='wrap')
        elif padding_lax == 'CAUSAL':
            if len(kernel_size) != 1:
                raise ValueError('Causal padding is only implemented for 1D convolutions.')
            left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
            pads = [(0, 0), (left_pad, 0), (0, 0)]
            inputs = jnp.pad(inputs, pads)

        y = conv(inputs, kernel)

        if bias is not None:
            y = y.astype(bias.dtype)
            bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
            y += bias

        if num_batch_dimensions != 1:
            output_shape = input_batch_shape + y.shape[1:]
            y = jnp.reshape(y, output_shape)
        return y

    return _outer(inputs, kernel.astype(_COMPUTE_DTYPE), bias).astype(inputs.dtype)


@if_shuffle(_original_dense_call)
def _wrapped_dense_call(self: Dense, inputs: jax.Array) -> jax.Array:
    kernel = self.scope.param('kernel', self.kernel_init, (jnp.shape(inputs)[-1], self.features), self.param_dtype)
    if self.use_bias:
        bias = self.scope.param('bias', self.bias_init, (self.features,), self.param_dtype)
    else:
        bias = None
    kernel, bias = kernel.astype(self.dtype), bias.astype(self.dtype)

    @remat
    def _outer(x: jax.Array, kernel: jax.Array, bias: Optional[jax.Array]) -> jax.Array:
        y = comm(dot)(x, kernel)
        if bias is not None:
            y = y.astype(bias.dtype)
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y

    return _outer(inputs, kernel.astype(_COMPUTE_DTYPE), bias).astype(inputs.dtype)


# synchronise everywhere to ensure we don't get into a pseudo-MoE situation where different sequence items see
# different parameters but instead always pretend it's a sliding window
Dense.__call__ = _wrapped_dense_call
Conv.__call__ = _wrapped_conv_call


def dict_to_array_dispatch(v):
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return dict_to_array_dispatch(v.item())
        if v.dtype == object:
            raise ValueError(str(v))
        return v
    elif isinstance(v, dict):
        return dict_to_array(v)
    elif isinstance(v, (list, tuple)):
        return list(zip(*sorted(dict_to_array(dict(enumerate(v))).items())))[1]
    else:
        return dict_to_array(v)


def dict_to_array(x):
    new_weights = {}
    for k, v in dict(x).items():
        new_weights[k] = dict_to_array_dispatch(v)
    return new_weights


USE_FSDP = jax.device_count() // _MODEL_PARALLEL_DEVICES >= 4


def scale_by_laprop(b1: float, b2: float, lb1: float, lb2: float, eps: float, lr: optax.Schedule,
                    clip: float = 1e-2) -> GradientTransformation:  # adam+lion
    def zero(x):
        return jnp.zeros_like(x, dtype=jnp.bfloat16)

    def init_fn(params):
        count = jnp.zeros((), dtype=jnp.int64)
        return {"mu": jax.tree_util.tree_map(zero, params), "nu": jax.tree_util.tree_map(zero, params), "count": count}

    def update_fn(updates, state, params=None):
        count = state["count"] + 1

        def get_update(name: str, grad: jax.Array, param: jax.Array, mu: jax.Array, nu: jax.Array):
            dtype = mu.dtype
            grad, param, mu, nu = promote(nan_default((grad, param, mu, nu), 0))
            g_norm = clip_norm(grad, 1e-8)
            p_norm = clip_norm(param, 1e-3)
            grad *= nan_default(lax.min(p_norm / g_norm * clip, 1.), 1.)

            nuc, nu = ema(lax.square(grad), nu, b2, count)
            grad /= nan_default(lax.max(lax.sqrt(nuc), eps), 1)
            muc, mu = ema(grad, mu, b1, count)

            name = '|'.join(map(str, name)).lower()
            update = lax.sign(muc)
            update *= jnp.linalg.norm(muc) / clip_norm(update, 1e-8)
            update += param * isctx() * (param.ndim > 1) * ("kernel" in name) * WEIGHT_DECAY_FOR_CONTEXT
            update *= -lr(count)
            return nan_default((update, mu.astype(dtype), nu.astype(dtype)), 0)

        updates, mus, nus = named_treemap(get_update, updates, params, state["mu"], state["nu"])
        return updates, {"count": count, "mu": mus, "nu": nus}

    return GradientTransformation(init_fn, update_fn)


def log(*args, **kwargs):
    print(f'{datetime.datetime.now()} | ', *args, **kwargs)


def distance(x: jax.Array, y: jax.Array) -> Tuple[jax.Array, jax.Array]:
    dist = x - y
    dist_sq = lax.square(dist).mean()
    dist_abs = lax.abs(dist).mean()
    return dist_sq / jax.device_count(), dist_abs / jax.device_count()


def to_img(x: jax.Array) -> wandb.Image:
    return wandb.Image(x.reshape(-1, *x.shape[-2:]))  # flatten context dim into height, keep width + channels const


def tile(x: jax.Array, mul: int):
    return lax.broadcast_in_dim(x, (mul, *x.shape), tuple(range(1, 1 + x.ndim))).reshape(mul * x.shape[0], *x.shape[1:])


def to_nchw(x: jax.Array):
    return x.transpose(0, x.ndim - 1, *range(1, x.ndim - 1))


def to_nhwc(x: jax.Array):
    return x.transpose(0, *range(2, x.ndim), 1)


def all_to_all(x, split=1):
    out = lax.all_to_all(x.reshape(1, -1, *x.shape[1:]), _AXIS, split, 0, tiled=True)
    return out.reshape(jax.device_count(), -1, *out.shape[3:])


def all_to_all_batch(batch: Dict[str, Union[np.ndarray, int]]) -> Dict[str, Union[np.ndarray, int]]:
    return {"mean": all_to_all(batch["mean"], 1), "std": all_to_all(batch["std"], 1),
            "encoded": all_to_all(batch["encoded"], 1), "frame_ids": all_to_all(batch["frame_ids"], 1)}


def if_usefsdp(fn):
    def _fn(self, x, *args, **kwargs):
        if USE_FSDP:
            return fn(self, x, *args, **kwargs)
        return x

    return _fn


class Sharding:
    def __init__(self, xs):
        arrays, self.tree = jax.tree_util.tree_flatten(xs)
        self.shapes = [x.shape for x in arrays]
        self.sizes = [x.size for x in arrays]
        self.cum_sizes = list(np.cumsum(self.sizes))
        self.total_size = self.cum_sizes[-1]

        self.partitions = [list(range(i, jax.device_count(), _MODEL_PARALLEL_DEVICES))  #
                           for i in range(_MODEL_PARALLEL_DEVICES)]
        self.partition_size = len(self.partitions[0])
        self.comm_args = {"axis_name": _AXIS, "axis_index_groups": self.partitions}

    @if_usefsdp
    def prepare_shard(self, xs: Any, fill_value: float = 0) -> jax.Array:
        arrays, _ = jax.tree_util.tree_flatten(xs)
        concat = jnp.concatenate([x.reshape(-1) for x in arrays])

        size = concat.shape[0]
        if size % self.partition_size:
            padding = jnp.full(((-size) % self.partition_size,), fill_value, dtype=concat.dtype)
            concat = jnp.concatenate([concat, padding])
        slice_size = size // self.partition_size
        # we want each device to store a small section of each tensor, so we can remat them one at a time
        return concat.reshape(slice_size, -1)

    @if_usefsdp
    def shard(self, xs: Any, fill_value: float = 0):
        return lax.dynamic_index_in_dim(self.prepare_shard(xs, fill_value), device_id(), axis=1, keepdims=False)

    @if_usefsdp
    def unshard(self, x: jax.Array):
        x = lax.all_gather(x, **self.comm_args, axis=1).reshape(-1)
        arrays = [x[start:end].reshape(shape)  #
                  for start, end, shape in zip([0] + self.cum_sizes, self.cum_sizes, self.shapes)]
        return self.tree.unflatten(arrays)

    def shard_via_psum_scatter(self, xs: Any, fill_value: float = 0):
        if not USE_FSDP:
            return lax.psum(xs, **self.comm_args)
        return lax.psum_scatter(self.prepare_shard(xs, fill_value), **self.comm_args, scatter_dimension=1).reshape(-1)


def _scale_down_dense(prefix, x):
    prefix = ('|'.join(map(str, prefix))).lower()
    if 'kernel' not in prefix or x.ndim < 2:
        return x
    return x * SCALE_DOWN_CONTEXT


def get_train_step(unet: diffusers.FlaxUNet2DConditionModel, resolution: int, unc: np.ndarray,
                   noise_scheduler: diffusers.FlaxDDIMScheduler, sched_state: Any, sampling_steps: int,
                   guidance: List[int], sharding: Sharding, weight_ema: float):
    def unet_fn(noise, encoded, timesteps, params, frame_ids):
        noise = lax.stop_gradient(noise) + params["input_embedding"][frame_ids].reshape(-1, 4, 1, 1)
        with set_flag("shuffle"):
            return unet.apply({"params": params}, noise, timesteps,
                              lax.stop_gradient(encoded)).sample

    def sample(params: jax.Array, p_ema: jax.Array, hidden_mode, encoded, ema_step: jax.Array, frame_ids: jax.Array):
        if USE_FSDP:
            params, p_ema = ema(p_ema, params, weight_ema, ema_step)
        params = sharding.unshard(params)
        latents = to_nchw(hidden_mode) * 0.18215

        encoded = jnp.concatenate([unc, encoded], 0).astype(jnp.float32)
        state = noise_scheduler.set_timesteps(sched_state, sampling_steps)

        def _outer(_, g):
            def _step(latents, i):
                pred = unet_fn(tile(latents, 2), encoded, i, params, frame_ids)
                pred = pred.reshape(2, -1, *pred.shape[1:])
                pred = jnp.einsum("cbhwf,c->bhwf", pred, g)
                return noise_scheduler.step(state, pred, i, latents).prev_sample, None

            out, _ = lax.scan(_step, noise, state.timesteps)
            return None, out

        structured = jax.random.normal(jax.random.PRNGKey(0), (1, *latents.shape[1:]), latents.dtype)
        local_prngkey = jax.random.PRNGKey(device_id())
        _, local_prngkey = jax.random.split(local_prngkey)
        noise = jax.random.normal(local_prngkey, structured.shape, latents.dtype)  # different perturbation for devices
        t0 = jnp.full((), len(noise_scheduler), jnp.int32)
        noise = noise_scheduler.add_noise(state, structured, noise, t0)  # keeps structure while adding perturbation
        noise = noise * state.init_noise_sigma

        _, out = lax.scan(_outer, None, jnp.array([[1 - g, g] for g in guidance], dtype=latents.dtype))
        out = to_nhwc(out[:, 0])
        return jnp.concatenate([hidden_mode, out / 0.18215]), p_ema

    def train_step(outer_state: TrainState, batch: Dict[str, jax.Array], p_ema: jax.Array, do_sample: jax.Array,
                   ema_step: jax.Array, seed: jax.Array):
        batch = all_to_all_batch(batch)

        params = outer_state.params
        cond = jnp.logical_and(isctx(), outer_state.step == 0).astype(jnp.int32)
        params = lax.switch(cond, [lambda: params, lambda: named_treemap(_scale_down_dense, params)])
        outer_state = outer_state.replace(params=params)

        new_img, p_ema = lax.switch(do_sample, [
            lambda: (jnp.zeros((len(guidance) + 1, resolution // 8, resolution // 8, 4), dtype=jnp.float32), p_ema),
            lambda: sample(outer_state.params, p_ema, batch["mean"][0], batch["encoded"][0], ema_step,
                           batch["frame_ids"][0])])

        def _loss(params, inp):
            noisy_latents, target, t0, encoded, frame_ids = inp
            unet_pred = unet_fn(noisy_latents, encoded[0].astype(jnp.float32), t0, params, frame_ids)
            return distance(unet_pred, target)

        def _outer(state: TrainState, x):
            params = sharding.unshard(state.params)
            scalars, grads = jax.value_and_grad(_loss, has_aux=True)(params, x)
            grads = sharding.shard_via_psum_scatter(grads)
            return state.apply_gradients(grads=grads), scalars

        seed, time_key, latent_key, noise_key = jax.random.split(seed, 4)
        latents = batch["mean"] + batch["std"] * jax.random.normal(latent_key, batch["std"].shape)
        latents = latents.transpose(0, 1, latents.ndim - 1, *range(2, latents.ndim - 1)) * 0.18215
        noise = jax.random.normal(noise_key, latents.shape)

        timesteps = jax.random.randint(time_key, (noise.shape[0],), 0, len(noise_scheduler))
        noisy_latents = noise_scheduler.add_noise(sched_state, latents, noise, timesteps)

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(sched_state, latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        outer_state, outer_scalars = lax.scan(_outer, outer_state, (noisy_latents, target, timesteps, batch["encoded"],
                                                                    batch["frame_ids"]))
        return outer_state, p_ema, lax.psum(map_fn(jnp.ravel)(outer_scalars), _AXIS), new_img, device_id(), seed

    return train_step


class DataLoader:
    def __init__(self, path: list, context: int, batch: int, prefetch: int = 4, seed: int = 0):
        """
        Example filenames:
        gs://video-us/data/0/FeGMU3fh-kA_98_image_embd.np
        gs://video-us/data/0/FeGMU3fh-kA_99_image_embd.np
        gs://video-us/data/0/Fhz6scjQK2Q_214_image_embd.np
        gs://video-us/data/0/Fhz6scjQK2Q_214_subtitles.txt
        gs://video-us/data/0/Fhz6scjQK2Q_214_text_embd.np
        gs://video-us/data/0/FuujXU_4JjE_130_image_embd.np
        gs://video-us/data/0/FuujXU_4JjE_130_subtitles.txt
        gs://video-us/data/0/FuujXU_4JjE_130_text_embd.np
        gs://video-us/data/0/FuujXU_4JjE_131_image_embd.np
        """
        self.batch = batch
        self.context = context

        path = path[len('gs://'):]
        bucket = path.split('/')[0]
        self.prefix = f'gs://{bucket}/'

        with open("list.json", 'r') as f:
            self.files = json.load(f)

        self.rng = random.Random(seed)
        self.rng.shuffle(self.files)

        k, v = self.files[0]
        self.shape = self._load(f'{k}_{v[0]}_image_embd.np', 'mean')[0].shape
        self.context_per_object = self.shape[0] * self.shape[1]
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.batch_queue = queue.Queue(prefetch)

    @contextlib.contextmanager
    def _open(self, path: str, mode: str):
        with open(f'{self.prefix}{path}', mode) as f:
            yield f

    def _load(self, path: str, *keys: str) -> List[np.ndarray]:
        values = []
        with self._open(path, "rb") as f:
            embed = np.load(f, allow_pickle=True)
            for k in keys:
                out = embed[k]
                if out.dtype == np.dtype('V2'):
                    out.dtype = jnp.bfloat16
                values.append(out.astype(jnp.float32))
        return values

    def _flatten(self, x: List[np.ndarray], offset: int = 0):
        x = np.concatenate(x)
        batch, ctx, *_ = x.shape
        x = x.reshape(batch * ctx, 1, *x.shape[2:])
        return x[offset:offset + self.context]

    def _worker(self):
        required_samples = int(math.ceil(self.context / self.context_per_object))
        batch = []
        for path, indices in self.files:
            if len(indices) < required_samples:
                continue

            first_index = indices[0]
            text_embd = self._load(f'{path}_{first_index}_text_embd.np', 'encoded')
            text_embd = self._flatten(text_embd)

            if len(indices) - required_samples == 0:
                start = 0
            else:
                start = 0
                # start = self.rng.randint(0, len(indices) - required_samples)
            embds = [self._load(f'{path}_{v}_image_embd.np', "std", "mean")  #
                     for v in indices[start:start + required_samples]]
            std, mean = zip(*embds)
            # offset = self.rng.randint(0, required_samples * self.context_per_object - self.context)
            offset = 0
            std = self._flatten(std, offset)
            mean = self._flatten(mean, offset)

            with self._open(f'{path}_{first_index}_subtitles.txt', 'r') as f:
                subtitles = f.read()

            start = start * self.context_per_object + offset
            batch.append((text_embd, std, mean, subtitles, np.arange(start, start + self.context)))

            if len(batch) == self.batch:
                text_embd, std, mean, subtitles, frame_ids = [np.stack(x) for x in zip(*batch)]
                self.batch_queue.put({"encoded": text_embd, "std": std, "mean": mean, "subs": subtitles,
                                      "frame_ids": frame_ids})
                batch.clear()

        self.running = False

    def _start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()
        return

    def __iter__(self):
        self._start()
        while self.running:
            try:
                yield self.batch_queue.get(timeout=60)
            except queue.Empty:
                continue


def load(path: str, prototype: Dict[str, jax.Array]):
    try:
        with smart_open.open(path + ".np", 'rb') as f:
            params = list(zip(*sorted([(int(i), v) for i, v in np.load(f).items()])))[1]
    except:
        with smart_open.open(path + ".np", 'rb') as f:
            params = \
                list(zip(*sorted([(int(i), v) for i, v in np.load(f, allow_pickle=True)["arr_0"].item().items()])))[1]

    _, tree = jax.tree_util.tree_flatten(prototype)
    return tree.unflatten(params)


def get_uncond_embds(base_model: str, clip_tokens: int):
    text_encoder = transformers.FlaxCLIPTextModel.from_pretrained(base_model, jnp.float32, subfolder="text_encoder")
    tokenizer = transformers.CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")

    unconditioned_tokens = tokenizer([""], padding="max_length", max_length=clip_tokens, return_tensors="np")
    return text_encoder(unconditioned_tokens["input_ids"])[0]


def log_loss(epoch: int, subs: np.ndarray, i: int, do_sample: bool, scalars, samples, run: Any, start_time: int,
             guidance: List[float], eval_path: str, lsteps: int, state):
    if state is None:
        step = 0
        loss_history = []
    else:
        step, loss_history = state
    step += 1
    step_id = step * lsteps
    i *= lsteps

    sclr = to_host(scalars)
    log("To host")

    if do_sample:
        samples = to_host(samples, lambda x: x)
        s_mode, *rec = np.split(samples, 1 + len(guidance), 1)
        for _ in range(_UPLOAD_RETRIES):
            try:
                with smart_open.open(eval_path + f"{step_id}.npz", "wb") as f:
                    np.savez(f, **{f"guidance{g}": r for g, r in zip(guidance, rec)}, mode=s_mode, subs=subs)
            except:
                log("failed to write guidance checkpoint")
                traceback.print_exc()
            else:
                break
        log("Finished post-processing samples")

    timediff = time.time() - start_time
    sclr = [[float(x) for x in val] for val in sclr]
    log("Losses:", sclr[0])
    for offset, (unet_sq, unet_abs) in enumerate(zip(*sclr)):
        vid_per_day = step_id / timediff * 24 * 3600 * jax.device_count()
        loss_history.append(unet_sq)
        loss_history = loss_history[-512:]
        vals = {"U-Net MSE/Total": unet_sq, "U-Net MAE/Total": unet_abs, "GlobalStep": step_id + offset - lsteps,
                "Step": i + offset - lsteps, "Epoch": epoch, "U-Net MSE/Median 32": np.median(loss_history[-32:]),
                "U-Net MSE/Median 128": np.median(loss_history[-128:]),
                "U-Net MSE/Median 512": np.median(loss_history[-512:]),
                "U-Net MSE/Mean 32": np.mean(loss_history[-32:]), "U-Net MSE/Mean 128": np.mean(loss_history[-128:]),
                "U-Net MSE/Mean 512": np.mean(loss_history[-512:])}
        if offset == lsteps - 1:
            vals.update({"Runtime": timediff, "Speed/Videos per Day": vid_per_day, "Speed/Frames per Day": vid_per_day})
        run.log(vals, step=(step - 1) * lsteps + offset)
    return step, loss_history


class Pipeline:
    def __init__(self, fn: Callable, length: int = 16, **kwargs):
        self.length = length
        self.cond = threading.Condition()
        self.pipe = []
        self.fn = fn
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.kwargs = kwargs

        self.start()

    def _loop(self):
        start_time = time.time()
        state = None
        while True:
            with self.cond:
                if len(self.pipe) < self.length and not any(p[1] for p in self.pipe):
                    self.cond.wait()
                if len(self.pipe) < self.length and not any(p[1] for p in self.pipe):
                    continue
                if len(self.pipe) > 2 * self.length:
                    log(f"\nWARNING: Pipeline has {len(self.pipe)} elements, but upload can't keep up\n")
                values, _ = self.pipe.pop(0)
                state = self.fn(*values, start_time=start_time, state=state, **self.kwargs)

    def start(self):
        if self.running:
            return

        self.thread = threading.Thread(target=self._loop)
        self.thread.start()

    def put(self, *item, flush=False):
        self.pipe.append((item, flush))
        with self.cond:
            self.cond.notify_all()


# TODO: Use EMA to improve sample quality
@app.command()
def main(lr: float = 1e-5, beta1: float = 0.9, beta2: float = 0.99, lion_beta1: float = 0.9, lion_beta2: float = 0.99,
         eps: float = 1e-16, lr_halving_every_n_steps: int = 2 ** 16, warmup_steps: int = 1024,  #
         data_path: str = "gs://video-us/data/0/", batch_prefetch: int = 4, resolution: int = 1024,
         clip_tokens: int = 77,  #
         save_interval: int = 4096, overwrite: bool = True, base_model: str = "flax/stable-diffusion-2-1",
         eval_path: str = f"gs://video-us/eval-embeddings/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/{jax.process_index()}/",
         base_path: str = "gs://video-us/checkpoint_2",  #
         guidance: List[int] = typer.Option([1, 4, 7, 10]), sample_interval: int = 8192, sampling_steps: int = 1000,
         weight_ema_beta: float = 0.999,  #
         seed: int = 0):
    data = DataLoader(data_path, jax.device_count(), jax.local_device_count(), batch_prefetch, seed)

    run = wandb.init(entity="ttv", project="ttv")

    unet, unet_params = diffusers.FlaxUNet2DConditionModel.from_pretrained(base_model, jnp.float32, subfolder="unet")
    noise_scheduler, sched_state = diffusers.FlaxDDIMScheduler.from_pretrained(base_model, subfolder="scheduler")

    unet: diffusers.FlaxUNet2DConditionModel = unet
    noise_scheduler: diffusers.FlaxDDIMScheduler = noise_scheduler

    if not overwrite:
        log("Loading..")
        weights = [load(f"/home/ubuntu/unet{i}", unet_params) for i in range(_MODEL_PARALLEL_DEVICES)]
        log("Finished")

    lr_sched = optax.warmup_exponential_decay_schedule(0, lr, warmup_steps, lr_halving_every_n_steps, 0.5)
    optimizer = scale_by_laprop(beta1, beta2, lion_beta1, lion_beta2, eps, lr_sched)

    sharding = Sharding(unet_params)

    train_step = get_train_step(unet, resolution, get_uncond_embds(base_model, clip_tokens), noise_scheduler,
                                sched_state, sampling_steps, guidance, sharding, weight_ema_beta)
    p_train_step = jax.pmap(train_step, _AXIS, donate_argnums=(0, 1, 2, 3, 4))
    if overwrite:
        device_ids = None
    else:
        log("Retrieving device ids")
        device_ids = jax.pmap(lambda x: device_id(), _AXIS, donate_argnums=(0,))(jnp.arange(jax.local_device_count()))
        device_ids = to_host(device_ids, lambda x: x)
        log("Got device ids")

        def _get_weight(*w):
            return jnp.stack([w[d % _MODEL_PARALLEL_DEVICES] for d in device_ids])

        unet_params = jax.tree_util.tree_map(_get_weight, *weights)

    def get_state(p):
        if "input_embedding" not in p:
            p["input_embedding"] = np.zeros((65536, 4), dtype=np.float32)
        return TrainState.create(apply_fn=unet.__call__, params=p, tx=optimizer)

    if USE_FSDP:
        if overwrite:
            unet_params = replicate(unet_params)

        unet_state = jax.pmap(lambda x: get_state(sharding.shard(x)), _AXIS)(unet_params)
    elif not overwrite:
        unet_state = [get_state(index(unet_params, i)) for i in range(jax.local_device_count())]
        unet_state = jax.device_put_sharded(unet_state, jax.local_devices())
    else:
        unet_state = replicate(get_state(unet_params))

    if USE_FSDP:
        unshard = jax.pmap(sharding.unshard, _AXIS)
    else:
        unshard = None

    global_step = 0
    lsteps = jax.device_count()

    sample_count = 0
    if USE_FSDP:
        p_ema = map_fn(jnp.zeros_like)(unet_state.params)
    else:
        p_ema = jnp.zeros((jax.local_device_count(),), dtype=jnp.int8)

    logging_pipeline = Pipeline(log_loss, run=run, guidance=guidance, eval_path=eval_path, lsteps=lsteps)
    uploaded_ids = None
    rng_seed = jax.vmap(lambda x: jax.random.split(jax.random.PRNGKey(x))[1])(
        jnp.array([random.Random(d.id).randint(0, 2 ** 32 - 1) for d in jax.local_devices()]))
    for epoch in range(10 ** 9):
        for i, batch in tqdm.tqdm(enumerate(data, 1)):
            global_step += 1
            subs = batch.pop("subs")

            if global_step <= 2:
                log(f"Step {global_step}")

            do_sample = int(global_step % (sample_interval // lsteps) == 1)
            sample_count += do_sample

            step_id = global_step * lsteps
            log(f"Before step {step_id}")
            do_sample_jax = jnp.full((jax.local_device_count(),), do_sample, dtype=jnp.int32)
            count_jax = jnp.full((jax.local_device_count(),), sample_count, dtype=jnp.int32)
            unet_state, p_ema, scalars, samples, new_device_ids, rng_seed = \
                p_train_step(unet_state, batch, p_ema, do_sample_jax, count_jax, rng_seed)
            save_now = step_id % save_interval == 0

            logging_pipeline.put(epoch, subs, i, do_sample, scalars, samples, flush=save_now)
            log("After")

            if device_ids is None:
                device_ids = new_device_ids
            if uploaded_ids is None:
                uploaded_ids = set([int(x) for x in device_ids]).union(set(list(range(_MODEL_PARALLEL_DEVICES))))

            if save_now:
                if USE_FSDP:
                    params = unshard(p_ema)  # has to be here so all devices participate
                else:
                    params = unet_state.params
                for uid in uploaded_ids:
                    p = to_host(params, lambda x: x[uid])
                    flattened, jax_structure = jax.tree_util.tree_flatten(p)
                    for _ in range(_UPLOAD_RETRIES):
                        try:
                            with smart_open.open(f"{base_path}unet{uid}.np", "wb") as f:
                                np.savez(f, **{str(i): v for i, v in enumerate(flattened)})
                        except:
                            log("failed to write unet checkpoint")
                            traceback.print_exc()
                        else:
                            break


if __name__ == "__main__":
    app()
