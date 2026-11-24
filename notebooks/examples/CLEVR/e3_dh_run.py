"""Started by e3_dh_job.py in a fresh Python process: runs e3_images.main() with this command line.

Two things differ from running e3_images.py directly: the job image has no git, so the code version
comes from VERSION.json (written by e3_dh_launch.py at the root of the copy); and the CLEVR scene and
question files go to $CLEVR_DATA instead of notebooks/datasets (no free space next to the code).
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
version = json.loads((ROOT / "VERSION.json").read_text())

import torch  # noqa: E402

print(f"[run] torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}", flush=True)

import clevr_sltn  # noqa: E402

clevr_sltn.sltn_version = lambda: version
if os.environ.get("CLEVR_DATA"):
    clevr_sltn.DATA = Path(os.environ["CLEVR_DATA"])
import e3_images  # noqa: E402

e3_images.sltn_version = lambda: version
e3_images.DATA = clevr_sltn.DATA
print(f"[run] CLEVR files in {clevr_sltn.DATA}", flush=True)
sys.argv[0] = "e3_images.py"
e3_images.main()
