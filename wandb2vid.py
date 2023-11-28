import collections
import os

import cv2
import numpy as np
import tqdm
import typer
import wandb
from PIL import Image

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(api_key: str = "", project_name: str = 'homebrewnlp/stable-giffusion',
         data_dir: str = 'samples', download: bool = False, tpu: int = 256, fps: int = 16,
         resolution: int = 1024):
    if api_key == "":
        print("Please provide an api key")
        return

    targets = ['VAE Mode',
               'Reconstruction (U-Net, Guidance 1)',
               'Reconstruction (U-Net, Guidance 2)',
               'Reconstruction (U-Net, Guidance 4)',
               'Reconstruction (U-Net, Guidance 8)'
               ]

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(data_dir + '/video'):
        os.mkdir(data_dir + '/video')

    runs = []
    wandb.login(key=api_key)
    video_to_path = collections.defaultdict(dict)
    for n, run in enumerate(wandb.Api().runs(project_name)):
        if str(run)[-10:] == '(running)>':
            runs.append(str(run)[5:-11])
        if n > tpu:
            break

    if download:
        for run in tqdm.tqdm(runs, desc="download from runs"):
            samples = wandb.Api().run(run).files()
            for file in samples:
                file.download(replace=True, root=data_dir)

    for i, target in enumerate(targets):
        known_stepids = set()
        for file in tqdm.tqdm(os.listdir("samples/media/images/Samples"), desc="list dir"):
            step_id = int(file.split('_')[-2])
            known_stepids.add(step_id)

        for s in tqdm.tqdm(known_stepids, desc="building videos"):
            frames = []
            for file in os.listdir("samples/media/images/Samples"):
                if target in file and str(s) == file.split('_')[-2]:
                    frames.append(file)

            frames.sort(key=lambda x: int(x.split(" ")[-1].split(".")[0].split("-")[0]))
            if not frames:
                continue

            path = f'{data_dir}/video/{s}_{i}.mp4'
            video_to_path[s][target] = path
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (resolution, resolution))
            for j, im in enumerate(frames):
                im = Image.open("samples/media/images/Samples/" + im).convert("RGB")
                for image in np.array(im).reshape(-1, resolution, resolution, 3):
                    out.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            out.release()

    all_targets = sorted(list({k for st, sa in video_to_path.items() for k, _ in sa.items()}))
    all_targets = [t for t in targets if t in all_targets]
    paths = {}
    total_frames = collections.defaultdict(int)
    width = resolution * len(all_targets)
    for step, samples in tqdm.tqdm(video_to_path.items(), desc="merging targets"):
        caps = [cv2.VideoCapture(samples[tgt]) if tgt in samples else None for tgt in all_targets]

        out_path = f"{data_dir}/out_{step}.mp4"
        paths[step] = out_path
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, resolution))

        while True:
            fs = []
            for c in caps:
                if c is None:
                    fs.append(Image.fromarray(np.zeros((resolution, resolution, 3))))
                    continue
                ret, f = c.read()
                if not ret:
                    break
                fs.append(f)
            if len(fs) != len(caps):  # ret hit
                break
            out.write(cv2.hconcat(fs))
            total_frames[step] += 1

        for c in caps:
            c.release()
        out.release()
        for path in samples.values():
            os.remove(path)

    out = cv2.VideoWriter(f"{data_dir}/out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (width, resolution * len(video_to_path)))
    caps = [cv2.VideoCapture(out_path) for _, out_path in sorted(paths.items())]

    for _ in tqdm.tqdm(range(min(total_frames.values())), desc="merging steps"):
        fs = []
        for c in caps:
            ret, f = c.read()
            if not ret:
                break
            fs.append(f)
        if len(fs) != len(caps):  # ret hit
            break
        out.write(cv2.vconcat(fs))

    for c in caps:
        c.release()
    out.release()
    for path in paths.values():
        os.remove(path)
    os.system(f"ffmpeg -hide_banner -y -i {data_dir}/out.mp4 -an -c:v libvpx-vp9 -movflags +faststart -fflags +genpts {data_dir}/out.webm ")
    os.remove(f"{data_dir}/out.mp4")


if __name__ == "__main__":
    app()
