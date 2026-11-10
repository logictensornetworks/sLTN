"""Digit relations over time: sLTN and the CNN + GRU baseline over relations, dataset sizes and seeds, in one process.

    python sweep.py              # the grid of RUNS below for 5 seeds (stored: results/sweep-0d-seeds-*, T4 GPU)
    python sweep.py --table results/<dir>    # the table of a stored sweep

Each training run is prototype.run. The other stored sweeps (sweep-0d-*, one seed each) were the exploration
that led to this grid, with other models and settings; each of their rows records its model and settings.
Results: results/<name>-<time>/results.json.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

MODELS = ["untrained", "sltn-prod", "sltn-godel", "gru"]
SEEDS = [0, 1, 2, 3, 4]
CONFIG = dict(n_train=800, n_test=1000, steps=2000, batch=32, lr=1e-3, relation="sum")
# (settings that override CONFIG, models). Run 1 (2026-11-03): sum, 2,000 steps, lr 1e-3: nothing learned.
RUNS = [
    # Run 7 (2026-11-03, seed 0): labels only 0.81 image acc; with the rule 0.96 at 800 sequences; GRU near chance
    # up to 2,000 sequences. Run 8: the whole grid, 5 seeds.
]
for _rel, _steps in (("less", 4000), ("sum", 6000)):
    for _n in (800, 2000, 20000):
        RUNS.append((dict(relation=_rel, n_train=_n, steps=_steps, gru_lr=1e-4),
                     ["sltn-prod-lab", "gru"] + (["sltn-prod-labonly"] if _n == 20000 else [])))
HERE = Path(__file__).resolve().parent


def table(rows):
    import numpy as np
    keys = ["seq_acc", "seq_acc_long", "img_acc", "probe_acc"]
    lines = ["| relation | steps | lr | model | seeds | " + " | ".join(keys) + " |", "|---|---|---|---|---|" + "---|" * len(keys)]
    groups = {}
    for r in rows:
        groups.setdefault((r.get("relation", "sum"), r.get("steps", 2000), r.get("lr", 1e-3), r["model"]), []).append(r)
    for (rel, steps, lr, m), sel in groups.items():
        cells = []
        for k in keys:
            v = [r[k] for r in sel if r.get(k) is not None]
            cells.append(f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "—")
        lines.append(f"| {rel} | {steps} | {lr:g} | {m} | {len(sel)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    import torch
    import prototype as P

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("device:", device, flush=True)
    out = HERE / "results" / f"sweep-0d-seeds-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(config=CONFIG, sltn=P.sltn_version(), device=device)
    pools = {"train": P.load_pool(True), "test": P.load_pool(False)}
    rows = []
    for settings, models in RUNS:
        cfg = {**CONFIG, **settings}
        a = type("Args", (), cfg)()
        for seed in SEEDS:
            sets = P.make_sets(a, seed, pools)
            for model in models:
                r = P.run(model, a, seed, pools, sets)
                rows.append(dict(seed=seed, model=model, relation=cfg["relation"], steps=cfg["steps"],
                                 lr=cfg.get("gru_lr", cfg["lr"]) if model == "gru" else cfg["lr"],
                                 n_train=cfg["n_train"], **r))
                (out / "results.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1, default=str))
    print(table(rows))
    print(f"saved {out / 'results.json'}")


if __name__ == "__main__":
    if "--table" in sys.argv:
        print(table(json.loads((Path(sys.argv[sys.argv.index("--table") + 1]) / "results.json").read_text())["rows"]))
    else:
        main()
