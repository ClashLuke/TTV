import dataclasses
import datetime
import io
import json
import multiprocessing
import os
import queue
import shutil
import subprocess
import threading
import traceback
import uuid
from multiprocessing import managers
from multiprocessing import shared_memory
from queue import Empty
from typing import List, Callable, Optional, Tuple, Any, Dict

import ffmpeg
import jax
import numpy as np
import requests
import transformers
import yt_dlp as youtube_dl
from sharedutils import FiFoSemaphore

from db_api import API_KEY

_DEBUG = False
_CLIP_TOKENS = 77


def load_hf(cls, *args, **kwargs):
    while True:
        try:
            return cls.from_pretrained(*args, **kwargs)
        except:
            pass


@dataclasses.dataclass
class Share:
    dtype: np.dtype
    shape: List[int]
    name: str


def to_share(inp: np.array, smm: managers.SharedMemoryManager) -> Share:
    mem = smm.SharedMemory(inp.nbytes)
    np_mem = np.ndarray(inp.shape, dtype=inp.dtype, buffer=mem.buf)
    np_mem[:] = inp[:]
    return Share(dtype=inp.dtype, shape=inp.shape, name=mem.name)


def from_share(share: Share) -> np.ndarray:
    mem = shared_memory.SharedMemory(name=share.name, create=False)
    arr = np.copy(np.ndarray(share.shape, share.dtype, buffer=mem.buf))
    mem.unlink()
    return arr


def try_except(fn: Callable, default=None):
    def _fn(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # skipcq: PYL-W0703
            print(r"IGNORED EXCEPTION \/\/\/")
            print(fn, exc)
            traceback.print_exc()
            print("IGNORED EXCEPTION /\\/\\/\\")

        return default

    return _fn


@try_except
def get_urls(youtube_getter, url: str, lock: threading.Semaphore, target_image_size: int) -> List[dict]:
    with lock:
        # We have to lock this part because it can lead to errors if multiple thread try to scrape video Information at
        # the same time.
        info = youtube_getter.extract_info(url, download=False)
    if info is None or 'formats' not in info:
        return []
    video_urls = []
    audio_urls = []
    for f in info['formats']:
        if f.get('acodec') != 'none' and f.get('vcodec') == 'none':
            audio_urls.append({'ext': f['ext'], 'url': f['url'], 'tbr': f.get('tbr')})

        width = f.get('width')
        height = f.get('height')
        url = f.get('url')
        ext = f.get('ext')
        format_note = f.get('format_note')

        if any(x is None for x in (width, height, url, ext, format_note)):
            continue
        if any(not x for x in (width, height, url, ext)):
            continue
        if format_note == "tiny" or width <= target_image_size or height <= target_image_size:
            continue

        video_urls.append({'width': width, 'height': height, 'ext': f['ext'], 'url': f['url'], })
    video_urls = sorted(video_urls, key=lambda x: (x['ext'] != 'mp4', x['width'], x['height']))
    audio_urls = sorted(audio_urls, key=lambda x: x['tbr'])
    return video_urls, audio_urls


def get_video_frames(video_urls: Tuple[List[dict], List[dict]], target_image_size: int, target_fps: int,
                     device_steps: int) -> np.ndarray:
    filename = uuid.uuid4()
    path = str(filename)

    for a in video_urls[1]:
        try:
            audio_buffer = io.BytesIO()
            with requests.get(a["url"], stream=True) as r:
                shutil.copyfileobj(r.raw, audio_buffer)
        except:  # We know this will fail many times due to the YT API. No need to inspect errors
            continue

        try:
            audio_buffer.seek(0)
            file = {'file': audio_buffer}
            headers = {'Authorization': f'Bearer {os.environ["WHISPER_API_TOKEN"]}'}
            data = {"fileType": a["ext"], "diarization": "false"}
            response = requests.post("https://transcribe.whisperapi.com", headers=headers, files=file, data=data)
            response.raise_for_status()
            response = response.json()
            if "error" in response:
                continue
            yield response["text"]
            break
        except:  # Unfortunate if it fails, possibly our fault
            traceback.print_exc()
            continue
    else:
        return

    for vid in video_urls[0]:
        if os.path.exists(path):
            os.remove(path)

        url = vid["url"]
        path = f"{filename}.{vid['ext']}"

        aspect_ratio = vid["width"] / vid["height"]
        w = round(target_image_size * aspect_ratio) if aspect_ratio > 1 else target_image_size
        h = target_image_size if aspect_ratio > 1 else round(target_image_size / aspect_ratio)
        try:
            vid = ffmpeg.input("pipe:")
            vid = vid.filter("scale", w=w, h=h)
            vid = vid.filter("crop", w=target_image_size, h=target_image_size)
            vid = vid.filter("fps", target_fps)
            vid = vid.output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="error", preset="ultrafast",
                             threads=target_image_size // 40)

            proc_v: subprocess.Popen = ffmpeg.run_async(vid, pipe_stdout=True, pipe_stdin=True)

        except ffmpeg.Error:  # Broken Video, next might work
            continue

        should_stop = [False]

        def _copy_v():
            try:
                with requests.get(url, stream=True) as r:
                    shutil.copyfileobj(r.raw, proc_v.stdin)
            except Exception:  # skipcq: PYL-W0703
                should_stop[0] = True
                proc_v.kill()

        thread_v = threading.Thread(target=_copy_v)
        thread_v.start()

        v_size = device_steps * target_image_size * target_image_size * 3

        while not should_stop[0]:
            v_tile = proc_v.stdout.read(v_size)
            if not v_tile or len(v_tile) != v_size:
                break
            yield np.frombuffer(v_tile, np.uint8).reshape(-1, target_image_size, target_image_size, 3)

        proc_v.kill()

        if os.path.exists(path):
            os.remove(path)
        return


def frame_worker(worker_id: int, lock: threading.Semaphore, target_image_size: int, target_fps: int,
                 queue: multiprocessing.Queue, smm: managers.SharedMemoryManager, device_steps: int,
                 qlock: FiFoSemaphore, tokenizer_args: Dict[str, Any]):
    tokenizer = load_hf(transformers.CLIPTokenizer, **tokenizer_args)
    youtube_base = 'https://www.youtube.com/watch?v='
    youtube_getter = youtube_dl.YoutubeDL(
        {'writeautomaticsub': False, 'socket_timeout': 600, "quiet": True, "verbose": False, "no_warnings": True,
         "ignoreerrors": True})
    youtube_getter.add_default_info_extractors()

    started_at = datetime.datetime.now()
    while True:
        wor = requests.get("https://limitless.sh/url", data=json.dumps({"data": API_KEY}))  # read url (not removed yet)
        wor.raise_for_status()
        wor = wor.json()
        urls = get_urls(youtube_getter, youtube_base + wor, lock, target_image_size)

        if not urls or not urls[0]:
            continue

        with qlock(1, release_first=True):
            iterator = get_video_frames(urls, target_image_size, target_fps, device_steps)

            for text in iterator:
                break
            else:
                continue
            tokens = tokenizer(text, return_tensors="np", padding="longest", truncation=False,
                               pad_to_multiple_of=jax.local_device_count() * _CLIP_TOKENS)["input_ids"]
            tokens = to_share(tokens, smm)

            print(f"Worker {worker_id} acquired first sample at {datetime.datetime.now()}, which took "
                  f"{datetime.datetime.now() - started_at}.")

        for t in iterator:
            sample = (to_share(t, smm), text, tokens, wor)
            text = None
            tokens = None
            with qlock(1, release_first=True):
                queue.put(sample)


class DataLoader:
    def __init__(self, workers: int, video_downloaders: int, resolution: int, fps: int, batch_size: int,
                 parallel_videos: int, context: int, batch_prefetch: int, tokenizer_args: Dict[str, Any],
                 seed: int = 0):
        self.workers = workers
        self.video_downloaders = video_downloaders
        self.resolution = resolution
        self.fps = fps
        self.batch_size = batch_size
        self.seed = seed
        self.parallel_videos = parallel_videos
        self.device_steps = context
        self.tokenizer_args = tokenizer_args

        self.running = False
        self.batch_queue = queue.Queue(batch_prefetch)
        self.thread: Optional[threading.Thread] = None
        self.batch_thread: Optional[threading.Thread] = None
        self._start()

    def _start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()
        return

    def _worker(self):
        # self.rng.shuffle(self.ids)
        lock = multiprocessing.Semaphore(self.video_downloaders)
        workers = []
        queue = multiprocessing.Queue(2)
        cond = FiFoSemaphore(self.parallel_videos)

        with managers.SharedMemoryManager() as smm:
            for i in range(self.workers):
                args = (i, lock, self.resolution, self.fps, queue, smm, self.device_steps, cond, self.tokenizer_args)
                workers.append(multiprocessing.Process(target=frame_worker, args=args, daemon=True))
            for w in workers:
                w.start()

            while self.running:
                while True:
                    try:
                        out = queue.get(timeout=120)
                    except Empty:
                        print(f"Queue empty. Couldn't load a new sample within 120 seconds.")
                    else:
                        break
                vid, text, text_tokens, url = out
                try:
                    vid = from_share(vid)
                except:
                    print("failed to load video share")
                    continue

                if text_tokens is not None:
                    try:
                        text_tokens = from_share(text_tokens)
                    except:
                        print("failed to load text share")
                        continue

                self.batch_queue.put((vid, text, text_tokens, url))

            for w in workers:
                w.join()

    def __iter__(self):
        self._start()
        while self.running:
            try:
                yield self.batch_queue.get(timeout=60)
            except queue.Empty:
                continue
        raise StopIteration
