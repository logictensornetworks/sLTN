"""Run E3 on FBK's Digital Hub from the terminal: stage the code, launch, follow, fetch.

Needs the Digital Hub Python library (`pip install "digitalhub~=0.15.11" "digitalhub-runtime-python~=0.15.2"`,
Python <= 3.12, a separate env from sltn's) and ~/.dhcore.ini set up; on the FBK network or VPN. The
project is --project or $DH_PROJECT.

  python e3_dh_launch.py prep  --profile 1xa40-shared            # once: the crops, as artifact clevr-crops
  python e3_dh_launch.py train --profile 1xa40-shared --tag given -- --steps 4500 --seed 0
  python e3_dh_launch.py train --profile 1xa40-shared --tag learned -- --steps 4500 --seed 0 --learn_relations
  python e3_dh_launch.py train --module clevr_qa --tag qa-pixel -- --steps 4500 --seed 0 --learn_relations
  python e3_dh_launch.py slots --profile 1xa40 --name joint-w0.5 -- --steps 100000 --amp   # slots + sLTN (resumes)
  python e3_dh_launch.py slots --profile 4xa40-shared --name joint-4gpu -- --amp   # one process per GPU
  python e3_dh_launch.py slots --profile 2xa40-shared --name antiR --init_from joint-w05 -- --steps 50000 --amp --fresh_relation
  python e3_dh_launch.py slots --mode pretrain --name k11 -- --steps 500000 --amp          # reconstruction only
  python e3_dh_launch.py follow <run id> e3-log-given              # state + last log line, until it ends
  python e3_dh_launch.py fetch e3-results-given results/e3          # the run folder, as saved by e3_images.py
  python e3_dh_launch.py why <run id>                              # the platform's reason for a failed run

`train` sends a copy of sltn/ and the files of this folder that the job imports, with the repo's layout, and
the code version (VERSION.json, from git here: the job image has no git). Each `train` or `prep` makes a
new version of its function. Give every run its own --tag: its log and results are saved under that name.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CLEVR = Path(__file__).resolve().parent
ROOT = CLEVR.parents[2]
SHIP = ["clevr_sltn.py", "clevr_diag.py", "e1_images.py", "e3_images.py", "e3_dh_job.py", "e3_dh_run.py",
        "dh_deps.py", "prep_crops.py", "prep_crops_dh_job.py", "slots.py", "prep_images.py", "e2_slots_pretrain.py",
        "slots_dh_job.py"]
HANDLERS = {"train": "notebooks.examples.CLEVR.e3_dh_job:main", "prep": "notebooks.examples.CLEVR.prep_crops_dh_job:main",
            "slots": "notebooks.examples.CLEVR.slots_dh_job:main"}
NAMES = {"train": "clevr-e3-train", "prep": "clevr-prep-crops", "slots": "clevr-slots-pretrain"}


def stage():
    """A copy of the code the job needs, with the repo's layout, in a temporary folder."""
    dst = Path(tempfile.mkdtemp(prefix="e3_dh_"))
    shutil.copytree(ROOT / "sltn", dst / "sltn", ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "notebooks" / "examples" / "CLEVR").mkdir(parents=True)
    for f in SHIP:
        shutil.copy(CLEVR / f, dst / "notebooks" / "examples" / "CLEVR" / f)
    shutil.copytree(CLEVR / "clevr_qa", dst / "notebooks" / "examples" / "CLEVR" / "clevr_qa",
                    ignore=shutil.ignore_patterns("__pycache__"))
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    (dst / "VERSION.json").write_text(json.dumps(dict(
        commit=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
        untracked=[f for f in SHIP if git("ls-files", f"notebooks/examples/CLEVR/{f}") == ""])))
    return dst


def launch(dh, a, kind, parameters):
    src = stage()
    try:
        p = dh.get_project(a.project)
        fn = p.new_function(name=NAMES[kind], kind="python", code_src=str(src), handler=HANDLERS[kind],
                            python_version="PYTHON3_12")
        run = fn.run(action="job", profile=a.profile, parameters=parameters)
        print(f"run {run.id} ({run.status.state}) | function {NAMES[kind]} version {fn.id} | {parameters}")
    finally:
        shutil.rmtree(src)


def follow(dh, a):
    p = dh.get_project(a.project)
    while True:
        try:
            state = p.get_run(a.run).status.state
        except Exception as e:                                  # the VPN dropped or the laptop slept
            print(time.strftime("%H:%M"), "unreachable:", type(e).__name__, flush=True)
            time.sleep(60)
            continue
        try:
            with tempfile.TemporaryDirectory() as d:
                path = Path(p.get_artifact(a.log).download(destination=d, overwrite=True))
                last = [l for l in path.read_text().splitlines() if l.strip()][-1]
        except Exception as e:
            last = f"(no log yet: {type(e).__name__})"
        print(time.strftime("%H:%M"), state, "|", last[:200], flush=True)
        if state in ("COMPLETED", "ERROR", "STOPPED"):
            return
        time.sleep(a.every)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=os.environ.get("DH_PROJECT"), help="Digital Hub project (or $DH_PROJECT)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="launch e3_images.py or clevr_qa.train; arguments after -- go to it")
    t.add_argument("--module", choices=["e3", "clevr_qa"], default="e3", help="e3_images.py or python -m clevr_qa.train")
    t.add_argument("--profile", default="1xa40-shared")
    t.add_argument("--tag", required=True, help="names the run's artifacts: e3-log-<tag>, e3-results-<tag>")
    t.add_argument("e3_args", nargs=argparse.REMAINDER)
    pr = sub.add_parser("prep", help="cut the crops into the artifact clevr-crops (once)")
    pr.add_argument("--profile", default="1xa40-shared")
    pr.add_argument("--workers", type=int, default=12)
    sl = sub.add_parser("slots", help="pretrain the slot model (slots_dh_job.py); arguments after -- go to it")
    sl.add_argument("--profile", default="1xa40-shared")
    sl.add_argument("--name", required=True, help="the run's name: relaunching the same name resumes it")
    sl.add_argument("--mode", choices=["joint", "pretrain"], default="joint")
    sl.add_argument("--init_from", default="", help="a new run starts from slots-<init_from>'s ckpt (train_slots --init)")
    sl.add_argument("slot_args", nargs=argparse.REMAINDER)
    f = sub.add_parser("follow", help="print the run's state and its log's last line until it ends")
    f.add_argument("run")
    f.add_argument("log", help="the log artifact, e.g. e3-log-given or prep-log")
    f.add_argument("--every", type=int, default=120, help="seconds between checks")
    w = sub.add_parser("why", help="the platform's reason for a run's state")
    w.add_argument("run")
    g = sub.add_parser("fetch", help="download the newest version of an artifact")
    g.add_argument("artifact")
    g.add_argument("dest")
    a = ap.parse_args()
    if not a.project:
        sys.exit("give --project or set DH_PROJECT")
    import digitalhub as dh                                     # the Digital Hub library, not needed by sltn

    if a.cmd == "train":
        args = [x for x in a.e3_args if x != "--"]
        launch(dh, a, "train", {"args": " ".join(args), "tag": a.tag, "module": a.module})
    elif a.cmd == "slots":
        launch(dh, a, "slots", {"name": a.name, "mode": a.mode, "init_from": a.init_from,
                                "args": " ".join(x for x in a.slot_args if x != "--")})
    elif a.cmd == "prep":
        launch(dh, a, "prep", {"workers": a.workers})
    elif a.cmd == "follow":
        follow(dh, a)
    elif a.cmd == "why":
        s = dh.get_project(a.project).get_run(a.run).status.to_dict()
        print("state:", s.get("state"), "|", s.get("message", ""))
        for pod in s.get("pods", []):
            for c in pod.get("containers", []):
                print("container:", c.get("state"))
    elif a.cmd == "fetch":
        print(dh.get_project(a.project).get_artifact(a.artifact).download(destination=a.dest, overwrite=True))


if __name__ == "__main__":
    main()
