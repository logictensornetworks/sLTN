"""Graph colouring: CNN, CNN + GraphSAGE and sLTN over seeds, label fractions and wrong edges, in one process.

Each training run is prototype.py's. The three stored sweeps (a T4 GPU, about 40 s per sLTN run) were:

    python sweep.py --name sweep-0c --fracs 0.01 0.02 0.05 --wrongs 0.0                  # sLTN 857c7ac
    python sweep.py --name sweep-0c-wrong --fracs 0.02 --wrongs 0.0 0.1 0.2 0.3           # sLTN 124c1f8
    python sweep.py --name sweep-0c-soft --fracs 0.02 --wrongs 0.0 0.1 0.2 0.3 \\
        --models sltn-p1 sltn-w0.25                                                       # sLTN 124c1f8

`--wrongs`: the fraction of the edges that join two nodes of the same colour (wrong knowledge).
Results: results/<name>-<time>/results.json, and a table of mean ± std over the seeds.
`python sweep.py --table results/<dir>` prints the table of a stored sweep.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# sLTN variants: name -> settings that override CONFIG (a softer rule for the wrong-knowledge runs)
VARIANTS = {"sltn": {}, "sltn-p1": {"p_agg": 1.0}, "sltn-w0.25": {"rule_weight": 0.25}}
CONFIG = dict(n_train=300, n_test=1000, steps=1500, batch=32, p_agg=2.0)
HERE = Path(__file__).resolve().parent


def table(rows):
    models = list(dict.fromkeys(r["model"] for r in rows))
    lines = ["| labels | wrong edges | model | seeds | node acc | same colour on right edges | graphs fully right "
             "| acc, nodes on a wrong edge | acc, other nodes |", "|---|---|---|---|---|---|---|---|---|"]
    for frac, wrong in sorted({(r["frac"], r.get("wrong", 0.0)) for r in rows}):
        key = lambda r: (r["frac"], r.get("wrong", 0.0)) == (frac, wrong)
        for m in models:
            sel = [r for r in rows if key(r) and r["model"] == m]
            if not sel:
                continue

            def cell(k):
                v = [r[k] for r in sel if k in r and r[k] == r[k]]          # drop NaN (no wrong edges)
                return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "—"
            lines.append(f"| {frac:.0%} | {wrong:.0%} | {m} | {len(sel)} | {cell('acc')} | {cell('edge_viol')} | "
                         f"{cell('graph_ok')} | {cell('acc_wrong_nodes')} | {cell('acc_other_nodes')} |")
        acc = {m: {r["seed"]: r["acc"] for r in rows if key(r) and r["model"] == m} for m in models}
        for m in ("cnn", "gnn"):
            if "sltn" not in acc or m not in acc:
                continue
            common = sorted(set(acc["sltn"]) & set(acc[m]))
            if common:
                gains = [acc["sltn"][s] - acc[m][s] for s in common]
                lines.append(f"\n{frac:.0%}, wrong {wrong:.0%}: sLTN beats {m} on {sum(g > 0 for g in gains)}/{len(gains)} seeds, "
                             f"smallest gain {100 * min(gains):+.1f} points")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sweep")
    ap.add_argument("--models", nargs="+", default=["cnn", "gnn", "sltn"])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    ap.add_argument("--wrongs", type=float, nargs="+", default=[0.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--table", default=None, help="print the table of a stored sweep and exit")
    a = ap.parse_args()
    if a.table:
        print(table(json.loads((Path(a.table) / "results.json").read_text())["rows"]))
        return

    import torch
    import prototype as P

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("device:", device, flush=True)
    out = HERE / "results" / f"{a.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(config=CONFIG, sltn=P.sltn_version(), device=device)
    args = type("Args", (), CONFIG)()
    pools = {"train": P.load_pool(True), "test": P.load_pool(False)}
    rows = []
    for frac, wrong in [(f, w) for f in a.fracs for w in a.wrongs]:
        for seed in a.seeds:
            train = P.to_device(P.make_graphs(args.n_train, pools["train"], frac,
                                              torch.Generator().manual_seed(1000 + seed), wrong=wrong))
            test = P.to_device(P.make_graphs(args.n_test, pools["test"], 0.0,
                                             torch.Generator().manual_seed(5000 + seed), wrong=wrong))
            n_lab = int(train["labelled"].sum())
            for model in a.models:
                t0 = time.perf_counter()
                if model in VARIANTS:
                    predict = P.train_sltn(train, type("Args", (), {**CONFIG, **VARIANTS[model]})(), seed)
                else:
                    predict = {"cnn": P.train_cnn, "gnn": P.train_gnn}[model](train, args, seed)
                r = P.scores(predict, test)
                rows.append(dict(frac=frac, wrong=wrong, seed=seed, model=model, n_labelled=n_lab, **r,
                                 seconds=round(time.perf_counter() - t0, 1)))
                (out / "results.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))
                print(f"labels {frac}, wrong {wrong}, seed {seed}, {model}: node acc {r['acc']:.3f} "
                      f"({rows[-1]['seconds']:.0f}s)", flush=True)
    print(table(rows))
    print(f"saved {out / 'results.json'}")


if __name__ == "__main__":
    main()
