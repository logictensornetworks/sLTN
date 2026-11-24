"""Digital Hub job: cut the CLEVR crops on the cluster and save them as the project artifact `clevr-crops`
(train_crops.npy, val_crops.npy), as prep_crops.py does locally. Launched by `e3_dh_launch.py prep`.

The images are read straight from CLEVR's full zip (18 GB), downloaded to /tmp and not extracted; the
cropping is prep_crops.crops_of, unchanged. The job writes about 24 GB under /tmp, which counts against
its memory: use a profile with enough RAM (on cpu-shared it was killed with OOMKilled; 1xa40-shared,
123 GiB, worked in 13 min). Progress goes to the artifact `prep-log`.
"""

import io
import json
import sys
import threading
import time
import urllib.request
import zipfile
from multiprocessing import Pool
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
CLEVR = Path(__file__).resolve().parent


def main(project, workers: int = 12, log_every: int = 120):
    sys.path.insert(0, str(CLEVR))
    work = Path("/tmp/clevr_prep")
    (work / "crops").mkdir(parents=True, exist_ok=True)
    log_path = work / "prep.log"
    log = open(log_path, "w")

    def say(m):
        print(m, flush=True)
        log.write(m + "\n")
        log.flush()

    stop = threading.Event()

    def upload_log():
        try:
            project.log_artifact(name="prep-log", kind="artifact", source=str(log_path))
        except Exception as e:
            print(f"log upload failed: {e}", flush=True)

    def uploader():
        while not stop.wait(log_every):
            upload_log()

    threading.Thread(target=uploader, daemon=True).start()
    try:
        say(f"[job] python {sys.version.split()[0]}, code in {CLEVR}")
        import dh_deps
        dh_deps.install(["numpy", "pillow"], say)
        import numpy as np
        from prep_crops import O, R, crops_of

        zpath, t0 = work / "CLEVR_v1.0.zip", time.time()
        with urllib.request.urlopen(URL) as r, open(zpath, "wb") as f:
            total, done, last = int(r.headers.get("Content-Length", 0)), 0, 0
            while chunk := r.read(1 << 24):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 60:
                    say(f"download {done / 2**30:.1f} / {total / 2**30:.1f} GiB, "
                        f"{done / 2**20 / (time.time() - t0):.0f} MiB/s")
                    last = time.time()
        say(f"downloaded {done / 2**30:.1f} GiB in {time.time() - t0:.0f}s")
        with zipfile.ZipFile(zpath) as z:
            for split in ("train", "val"):
                scenes = json.loads(z.read(f"CLEVR_v1.0/scenes/CLEVR_{split}_scenes.json"))["scenes"]
                arr = np.lib.format.open_memmap(work / "crops" / f"{split}_crops.npy", mode="w+", dtype=np.uint8,
                                                shape=(len(scenes), O, R, R, 3))
                jobs = ((io.BytesIO(z.read(f"CLEVR_v1.0/images/{split}/{s['image_filename']}")), s["objects"])
                        for s in scenes)
                t1 = time.time()
                with Pool(workers) as pool:
                    for i, c in enumerate(pool.imap(crops_of, jobs, chunksize=64)):
                        arr[i] = c
                        if i % 10000 == 0:
                            say(f"{split}: {i}/{len(scenes)} ({time.time() - t1:.0f}s)")
                arr.flush()
                say(f"{split}: {arr.shape} in {time.time() - t1:.0f}s")
        zpath.unlink()
        t2 = time.time()
        project.log_artifact(name="clevr-crops", kind="artifact", source=str(work / "crops"))
        say(f"uploaded clevr-crops in {time.time() - t2:.0f}s; total {time.time() - t0:.0f}s")
    except Exception as e:
        import traceback
        say(f"FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise
    finally:
        stop.set()
        upload_log()
