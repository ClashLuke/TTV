import hashlib
import json
import multiprocessing.queues
import os

import numpy as np
import tqdm
import typer


def worker(queue: multiprocessing.Queue, list_json: list):
    while True:
        try:
            name, vals = queue.get(60)
        except:
            break
        idx = 0
        name_id = hashlib.blake2b(name.encode()).hexdigest()[:16]
        internal = f'.internal_{name_id}.np'
        for i, v in enumerate(sorted(map(int, vals))):
            folder_name = str(i)
            try:
                os.mkdir(folder_name)
            except:
                pass
            try:
                os.remove(internal)
            except FileNotFoundError:
                pass
            os.system(f"gsutil -q cp gs://video-us/{name}_{v}_image_embd.np {internal}")
            with open(internal, 'rb') as f:
                k = np.load(f)
                std = k["std"]
                mean = k["mean"]
            for s, m in zip(std.reshape(-1, *std.shape[2:]), mean.reshape(-1, *mean.shape[2:])):
                with open(f"{folder_name}/{name_id}_{idx}_image_embd.np", 'wb') as f:
                    np.savez(f, std=s, mean=m)
                idx += 1
            os.system(f"gsutil -m -q mv {folder_name}/{name_id}* 'gs://video-us/1/' &")
        list_json.append((name, idx))


def main(procs: int, prefetch: int = 16):
    with open("list.json", 'r') as f:
        items = json.load(f)
    queue = multiprocessing.Queue(prefetch)
    mgr = multiprocessing.Manager()
    list_json = mgr.list()
    workers = [multiprocessing.Process(target=worker, args=[queue, list_json], daemon=True) for _ in range(procs)]
    for w in workers:
        w.start()
    for i in range(prefetch + procs):
        queue.put(items[i])
    for dat in tqdm.tqdm(items[prefetch + procs:], miniters=0):
        queue.put(dat)
    for w in tqdm.tqdm(workers, desc="Finishing up"):
        w.join()
    with open("new_list.json", "w") as f:
        json.dump(f, list_json)
    os.system("gsutil cp new_list.json gs://video-us/new_list.json")


if __name__ == '__main__':
    typer.run(main)
