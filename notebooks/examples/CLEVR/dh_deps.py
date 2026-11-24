"""Install a Digital Hub job's Python packages at start-up.

A plain `job` run doesn't install the function's `requirements` (those are for the `build` action), so the
handler installs them itself, into a private folder put first on sys.path. Import the packages after this.
"""

import subprocess
import sys
from pathlib import Path

TARGET = "/tmp/pydeps"


def install(packages, say=print, target=TARGET, extra=()):
    Path(target).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--target", target, *extra, *packages]
    say(f"[deps] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    say(f"[deps] exit {r.returncode}" + (f"\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}" if r.returncode else ""))
    if r.returncode:
        raise RuntimeError("pip install failed")
    sys.path.insert(0, target)
