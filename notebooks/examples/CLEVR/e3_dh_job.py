"""Digital Hub job: train E3 (e3_images.py) on the cluster. Launched by e3_dh_launch.py.

The job receives a copy of the repo (sltn/ and this folder) and runs from it. What it does:
1. installs torch and numpy (dh_deps.py: a plain job doesn't install `requirements`);
2. downloads the crops from the project artifact `clevr-crops` (made by prep_crops_dh_job.py);
3. runs e3_images.py (e3_dh_run.py) or, with module="clevr_qa", `python -m clevr_qa.train`, in a fresh Python
   process: the platform's own process has an old `typing_extensions` loaded that breaks `import torch`;
4. saves the run folder as the artifact `e3-results-<tag>`.

Printed output doesn't reach the console's Logs tab, so everything goes to a file saved as the artifact
`e3-log-<tag>` every few minutes and at the end. All data goes under `data_root` (/tmp): the folder the
code is unpacked in (/shared) has no free space.
"""

import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

CLEVR = Path(__file__).resolve().parent                # notebooks/examples/CLEVR
ROOT = CLEVR.parents[2]                                # the repo (or its copy on the cluster)


def main(project, args: str = "", tag: str = "run", log_every: int = 300, data_root: str = "/tmp/datasets",
         module: str = "e3"):
    out = Path("/tmp/e3_out")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"
    log = open(log_path, "w")

    def say(m):
        print(m, flush=True)
        log.write(m + "\n")
        log.flush()

    stop = threading.Event()

    def upload_log():
        try:
            log.flush()
            project.log_artifact(name=f"e3-log-{tag}", kind="artifact", source=str(log_path))
        except Exception as e:                                  # never kill the run over a log upload
            print(f"[job] log upload failed: {e}", flush=True)

    def uploader():
        while not stop.wait(log_every):
            upload_log()

    threading.Thread(target=uploader, daemon=True).start()
    try:
        say(f"[job] root {ROOT}, tag {tag!r}, args {args!r}")
        sys.path.insert(0, str(CLEVR))
        import dh_deps
        dh_deps.install(["torch>=2.10", "numpy"], say)

        crops = Path(data_root) / "CLEVR_crops"
        t0 = time.time()
        project.get_artifact("clevr-crops").download(destination=str(crops), overwrite=True)
        found = sorted(str(p.relative_to(crops)) for p in crops.rglob("*.npy"))
        say(f"[job] crops downloaded in {time.time() - t0:.0f}s: {found}")
        crops_dir = next(p.parent for p in crops.rglob("train_crops.npy"))

        entry = [str(CLEVR / "e3_dh_run.py")] if module == "e3" else ["-m", "clevr_qa.train"]
        cmd = [sys.executable, "-u", *entry, *shlex.split(args), "--crops", str(crops_dir),
               "--results", str(out / "results"), "--no_wandb"]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([dh_deps.TARGET, str(ROOT), str(CLEVR)]),
                   CLEVR_DATA=str(Path(data_root) / "CLEVR_v1.0"))
        say(f"[job] {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                cwd=str(CLEVR))
        for line in proc.stdout:
            say(line.rstrip("\n"))
        if proc.wait() != 0:
            raise RuntimeError(f"training exited with code {proc.returncode}")
        project.log_artifact(name=f"e3-results-{tag}", kind="artifact", source=str(out / "results"))
        say("[job] results uploaded")
    except Exception:
        import traceback
        say(f"[job] FAILED\n{traceback.format_exc()}")
        raise
    finally:
        stop.set()
        upload_log()
