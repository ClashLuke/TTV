import json
import os
import pathlib
import threading

import requests
import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')

import datetime
import hashlib
import time
from typing import Union, Dict, Callable, Any, List

import diffusers
import jax
import numpy as np
import smart_open
import tqdm
import transformers
import typer
import wandb
from diffusers.models.vae_flax import FlaxDiagonalGaussianDistribution
from flax import jax_utils
from jax import lax, numpy as jnp
from jax.experimental.compilation_cache import compilation_cache as cc
from google.cloud.storage import Client
from data_av import DataLoader, load_hf
from db_api import API_KEY

if pathlib.Path('/fsx/lucas').exists():
    cc.initialize_cache("/fsx/lucas/cache")
elif pathlib.Path('~').expanduser().exists():
    cc.initialize_cache(f"{str(pathlib.Path('~').expanduser())}/cache")
app = typer.Typer(pretty_exceptions_enable=False)
_UPLOAD_RETRIES = 8
_SHUFFLE = False
_ACTIVATION = None


def _take_0th(x):
    return x[0]


def device_id():
    return lax.axis_index("batch")


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


def to_host(k, index_fn: Callable[[jax.Array], jax.Array] = _take_0th):
    return jax.device_get(jax.tree_util.tree_map(index_fn, k))


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


def log(*args, **kwargs):
    print(f'{datetime.datetime.now()} | ', *args, **kwargs)


def rng(idx: Union[jax.Array, int]):
    return jax.random.PRNGKey(idx * jax.device_count() + device_id() // 2)


def to_nchw(x: jax.Array):
    return x.transpose(0, x.ndim - 1, *range(1, x.ndim - 1))


def get_train_step(text_encoder: transformers.FlaxCLIPTextModel, vae: diffusers.FlaxAutoencoderKL, resolution: int,
                   vae_params: Any):
    def vae_apply(*args, method=vae.__call__, **kwargs):
        return vae.apply({"params": vae_params}, *args, method=method, **kwargs)

    def vae_encode(batch: Dict[str, jax.Array], deterministic: bool) -> FlaxDiagonalGaussianDistribution:
        gauss0, drop0 = jax.random.split(rng(batch["idx"] + 1), 2)
        rngs = {"gaussian": gauss0, "dropout": drop0}
        img = batch["pixel_values"].astype(jnp.float32) / 255
        img = img.reshape(-1, resolution, resolution, 3)
        return vae_apply(to_nchw(img), rngs=rngs, deterministic=deterministic, method=vae.encode).latent_dist

    def train_step(batch: Dict[str, jax.Array]):
        batch["pixel_values"] = batch["pixel_values"].reshape(-1, batch["pixel_values"].shape[-1])
        encoded = vae_encode(batch, False)
        return encoded.std, encoded.mean

    def encode(input_ids: jax.Array, p):
        return text_encoder(input_ids, params=p)[0]

    return train_step, encode


client = Client.from_service_account_json(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])


def upload(path, _text=None, **kwargs):
    for _ in range(_UPLOAD_RETRIES):
        try:
            if _text is None:
                with smart_open.open(path, "wb", transport_params={"client": client}) as f:
                    np.savez(f, **kwargs)
            else:
                with smart_open.open(path, "w", transport_params={"client": client}) as f:
                    f.write(_text)
        except:
            log(f"failed to write checkpoint")
        else:
            break


def upload_pipe(run: Any, dataset_path: str, pipe: List, cond: threading.Condition, pipeline_length: int):
    start = time.time()
    i = 0
    while True:
        with cond:
            if len(pipe) < pipeline_length:
                cond.wait()
            if len(pipe) < pipeline_length:
                continue
            if len(pipe) > 2 * pipeline_length:
                log(f"\nWARNING: Pipeline has {len(pipe)} elements, but upload can't keep up\n")
            url, outputs, text, encoded = pipe.pop(0)
            i += 1
            frames = i * jax.device_count() * jax.device_count()
            runtime = time.time() - start
            run.log({"Wall Time": time.time(), "Runtime": runtime, "Step": i, "Frames/Total": frames,
                     "Frames/Per Day": frames / runtime * 3600 * 24}, step=i)

            outputs = to_host(outputs, lambda x: x)
            upload(f"{dataset_path}/{jax.process_index()}/{url}_{i}_image_embd.np", std=outputs[0], mean=outputs[1])

            if text is not None:
                upload(f"{dataset_path}/{jax.process_index()}/{url}_{i}_subtitles.txt", text)
                encoded = to_host(encoded, lambda x: x)
                upload(f"{dataset_path}/{jax.process_index()}/{url}_{i}_text_embd.np", encoded=encoded)
            requests.post("https://limitless.sh/url", data=json.dumps({"data": API_KEY, "url": url}))  # remove url


@app.command()
def main(downloaders: int = 2, resolution: int = 1024, fps: int = 16, workers: int = 256, batch_prefetch: int = 4,
         base_model: str = "flax/stable-diffusion-2-1-base", parallel_videos: int = 256, clip_tokens: int = 77,
         dataset_path: str = "gs://video-us/data", pipeline_length: int = 256, local_batch: int = 128):
    data = DataLoader(workers, downloaders, resolution, fps, 1, parallel_videos,
                      jax.local_device_count() * local_batch, batch_prefetch,
                      {"pretrained_model_name_or_path": base_model, "subfolder": "tokenizer"})

    run = wandb.init(entity="ttv", project="encode")

    text_encoder = load_hf(transformers.FlaxCLIPTextModel, base_model, jnp.float32, subfolder="text_encoder")
    vae, vae_params = load_hf(diffusers.FlaxAutoencoderKL, base_model, jnp.float32, subfolder="vae")

    text_encoder: transformers.FlaxCLIPTextModel = text_encoder
    vae: diffusers.FlaxAutoencoderKL = vae

    train_step, encode = get_train_step(text_encoder, vae, resolution, vae_params)
    p_train_step = jax.pmap(train_step, "batch")
    p_encode = jax.pmap(encode, "batch")

    text_params = jax_utils.replicate(text_encoder.params)

    global_step = 0
    lsteps = jax.device_count() * 2

    pipe = []
    cond = threading.Condition()
    thread = threading.Thread(target=upload_pipe, args=(run, dataset_path, pipe, cond, pipeline_length), daemon=True)
    thread.start()

    for vid, text, text_tokens, url in tqdm.tqdm(data):
        global_step += 1

        if global_step <= 2:
            log(f"Step {global_step}")

        log(f"Before step {global_step * lsteps}")
        idx = jnp.full((jax.local_device_count(),), int(hashlib.blake2b(str(global_step).encode()).hexdigest()[:4], 16),
                       dtype=jnp.int_)
        batch = {"pixel_values": vid.astype(jnp.uint8).reshape(jax.local_device_count(), local_batch, -1), "idx": idx}
        outputs = p_train_step(batch)  # tokens get ignored here

        if text_tokens is None:
            encoded = None
        else:
            encoded = p_encode(text_tokens.reshape(jax.local_device_count(), -1, clip_tokens), text_params)

        pipe.append((url, outputs, text, encoded))
        with cond:
            cond.notify_all()


if __name__ == "__main__":
    app()
