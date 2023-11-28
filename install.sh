python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir --force-reinstall --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo python3 -m pip uninstall tensorboard tbp-nightly tb-nightly tensorboard-plugin-profile -y
python3 -m pip install wandb smart-open[gcs] jsonpickle sharedutils git+https://github.com/ytdl-org/youtube-dl/ typer diffusers flax optax ffmpeg-python huggingface-hub transformers gdown torch torchvision opencv-python ftfy namedtreemap jaxhelper
python3 -m pip install --upgrade --force-reinstall tensorflow==2.8.0 protobuf==3.20.1 librosa redis fastapi uvicorn numpy
python3 -m pip install --no-cache-dir --force-reinstall --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m pip install --force-reinstall https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz
python3 -m pip uninstall torch -y

gsutil cp gs://video-us/list.json .