"""Every run of the notebook's comparison, for 5 seeds: both models, on 1,000 and on 100 training sequences.

    python sweep.py            # 20 runs (stored results: a T4 GPU, sLTN a0513f2, about 2 hours in all)
    python sweep.py --table    # only the table, from the stored results

Each run is `run.run` (the notebook's procedure as a script) and writes results/<model>-n<size>-seed<seed>/.
The table gives the mean and standard deviation over the seeds.
"""

import json
import sys
from pathlib import Path

import numpy as np

MODELS = ["sltn", "gru"]
SIZES = [1_000, 100]
SEEDS = [0, 1, 2, 3, 4]
HERE = Path(__file__).resolve().parent


def table(records):
    # mean ± std over seeds, per (model, size)
    cols = ["frame_acc", "frame_acc_probe", "before_f1_T6", "before_f1_T12",
            "contains_f1_T6", "contains_f1_T12", "contains_from_before_f1_T6", "contains_from_before_f1_T12"]
    lines = ["| model | n_train | seeds | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 3)]
    groups = {}
    for r in records:
        groups.setdefault((r["config"]["model"], r["config"]["n_train"]), []).append(r["metrics"])
    for (model, n), ms in sorted(groups.items(), key=lambda kv: (-kv[0][1], kv[0][0] != "sltn")):
        cells = []
        for c in cols:
            v = [m[c] for m in ms if c in m]
            cells.append(f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "—")
        lines.append(f"| {model} | {n} | {len(ms)} | " + " | ".join(cells) + " |")
    floor = [m["frame_acc_probe_untrained"] for ms in groups.values() for m in ms]
    lines.append(f"\nuntrained CNN, probe: {np.mean(floor):.3f} ± {np.std(floor):.3f}")
    return "\n".join(lines)


def stored():
    return [dict(config=json.loads((d / "config.json").read_text()), metrics=json.loads((d / "metrics.json").read_text()))
            for d in sorted((HERE / "results").iterdir()) if (d / "metrics.json").exists()]


def main():
    import torch
    import run

    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
    records = []
    for seed in SEEDS:
        for n in SIZES:
            for model in MODELS:
                records.append(run.run(model, n, seed))
    print(table(records))


if __name__ == "__main__":
    if "--table" in sys.argv:
        print(table(stored()))
    else:
        main()
