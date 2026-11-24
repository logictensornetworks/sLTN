"""Digital Hub job: train a slot model on the cluster. Launched by `e3_dh_launch.py slots`.
mode="joint": clevr_qa.train_slots, the slot model trained by the knowledge base (question rules and the
reconstruction axiom), one process per GPU of the profile (torchrun) when there are several;
mode="pretrain": e2_slots_pretrain.py, reconstruction alone (one GPU).

1. installs torch, numpy, pillow (dh_deps.py);
2. gets the 128 x 192 images from the project artifact `clevr-images128`; the first time, it builds them
   from CLEVR's full zip (18 GB, read without extracting; the resizing is prep_images.load) and uploads them;
3. if the artifact `slots-<name>` exists (an earlier run of the same name, e.g. preempted), downloads it and
   resumes from its ckpt.pt; otherwise, with init_from, starts from the ckpt.pt of `slots-<init_from>`
   (train_slots --init; add --fresh_relation / --fresh_present to the args to reset those parts);
4. runs the training in a fresh Python process, and every `upload_every` seconds uploads the run
   folder (ckpt.pt, history.json, config.json, ...) as `slots-<name>` and the log as `slots-log-<name>`.

Shared profiles can be preempted at any time: relaunch with the same name and it continues. Everything goes
under /tmp: the folder the code is unpacked in (/shared) has no free space.
"""

import io
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from multiprocessing import Pool
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
CLEVR = Path(__file__).resolve().parent
ROOT = CLEVR.parents[2]


def build_images(dest, say, workers):
    """CLEVR's full zip -> dest/train.npy, dest/val.npy (uint8 [scenes, 128, 192, 3], scene-file order)."""
    import numpy as np
    from prep_images import H, W, load_bytes
    work = Path("/tmp/clevr_zip")
    work.mkdir(parents=True, exist_ok=True)
    zpath, t0 = work / "CLEVR_v1.0.zip", time.time()
    with urllib.request.urlopen(URL) as r, open(zpath, "wb") as f:
        total, done, last = int(r.headers.get("Content-Length", 0)), 0, 0
        while chunk := r.read(1 << 24):
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 60:
                say(f"[images] download {done / 2**30:.1f} / {total / 2**30:.1f} GiB")
                last = time.time()
    say(f"[images] downloaded {done / 2**30:.1f} GiB in {time.time() - t0:.0f}s")
    with zipfile.ZipFile(zpath) as z:
        for split in ("train", "val"):
            scenes = json.loads(z.read(f"CLEVR_v1.0/scenes/CLEVR_{split}_scenes.json"))["scenes"]
            arr = np.lib.format.open_memmap(dest / f"{split}.npy", mode="w+", dtype=np.uint8,
                                            shape=(len(scenes), H, W, 3))
            jobs = (z.read(f"CLEVR_v1.0/images/{split}/{s['image_filename']}") for s in scenes)
            with Pool(workers) as pool:
                for i, im in enumerate(pool.imap(load_bytes, jobs, chunksize=64)):
                    arr[i] = im
                    if i % 10000 == 0:
                        say(f"[images] {split}: {i}/{len(scenes)}")
            arr.flush()
            say(f"[images] {split}: {arr.shape}")
    zpath.unlink()


def main(project, name: str = "k11", args: str = "", upload_every: int = 900, workers: int = 12,
         mode: str = "joint", init_from: str = ""):
    out = Path("/tmp/slots_out")
    (out / name).mkdir(parents=True, exist_ok=True)
    log_path = out / f"{name}.log"
    log = open(log_path, "a")

    def say(m):
        print(m, flush=True)
        log.write(m + "\n")
        log.flush()

    def upload():
        for art, src in ((f"slots-log-{name}", log_path), (f"slots-{name}", out / name)):
            try:
                if Path(src).exists() and (Path(src).is_file() or any(Path(src).iterdir())):
                    project.log_artifact(name=art, kind="artifact", source=str(src))
            except Exception as e:                                   # never kill the run over an upload
                print(f"[job] upload of {art} failed: {e}", flush=True)

    stop = threading.Event()
    threading.Thread(target=lambda: [upload() for _ in iter(lambda: stop.wait(upload_every), True)],
                     daemon=True).start()
    try:
        say(f"[job] root {ROOT}, name {name!r}, args {args!r}")
        sys.path.insert(0, str(CLEVR))
        import dh_deps
        dh_deps.install(["torch==2.10.*", "numpy", "pillow", "scipy"], say)

        images = Path("/tmp/datasets/CLEVR_images128")
        images.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            project.get_artifact("clevr-images128").download(destination=str(images), overwrite=True)
            say(f"[job] images downloaded in {time.time() - t0:.0f}s")
        except Exception as e:
            say(f"[job] no clevr-images128 artifact ({type(e).__name__}): building it")
            build_images(images, say, workers)
            project.log_artifact(name="clevr-images128", kind="artifact", source=str(images))
            say(f"[job] images built and uploaded in {time.time() - t0:.0f}s")
        images = next(p.parent for p in images.rglob("train.npy"))

        resume = []
        try:
            project.get_artifact(f"slots-{name}").download(destination=str(out / name), overwrite=True)
            ck = next((out / name).rglob("ckpt.pt"))
            if ck.parent != out / name:
                ck.replace(out / name / "ckpt.pt")
            resume = ["--resume"]
            say("[job] resuming from the artifact's ckpt.pt")
        except Exception as e:
            say(f"[job] no artifact slots-{name} ({type(e).__name__})")
            if init_from:                        # a new run started from another run's checkpoint
                src = Path("/tmp/slots_init")
                project.get_artifact(f"slots-{init_from}").download(destination=str(src), overwrite=True)
                resume = ["--init", str(next(src.rglob("ckpt.pt")))]
                say(f"[job] initialising from slots-{init_from}")
            else:
                say("[job] starting from scratch")

        if mode == "pretrain":
            cmd = [sys.executable, "-u", str(CLEVR / "e2_slots_pretrain.py"), "--name", name, *resume,
                   *shlex.split(args), "--images", str(images), "--results", str(out)]
        else:                                    # train_slots resumes by itself when the folder holds ckpt.pt
            r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            gpus = sum(line.startswith("GPU ") for line in r.stdout.splitlines())
            say(f"[job] {gpus} GPU(s):\n{r.stdout.strip()}")
            launcher = ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"] if gpus > 1 else []
            init = resume if resume[:1] == ["--init"] else []
            cmd = [sys.executable, "-u", *launcher, "-m", "clevr_qa.train_slots", *shlex.split(args), *init,
                   "--images", str(images), "--run_dir", str(out / name)]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([dh_deps.TARGET, str(ROOT), str(CLEVR)]),
                   CLEVR_DATA="/tmp/datasets/CLEVR_v1.0",
                   # NCCL's GPU-to-GPU (P2P) and shared-memory transports hung in the cluster's containers (the
                   # first broadcast never arrived): use sockets; the gradients are ~4 MB per step
                   NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="1", NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET",
                   TORCH_HOME="/tmp/torch_home")
        say(f"[job] {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                cwd=str(CLEVR))
        for line in proc.stdout:
            say(line.rstrip("\n"))
        if proc.wait() != 0:
            raise RuntimeError(f"training exited with code {proc.returncode}")
        say("[job] done")
    except Exception:
        import traceback
        say(f"[job] FAILED\n{traceback.format_exc()}")
        raise
    finally:
        stop.set()
        upload()
