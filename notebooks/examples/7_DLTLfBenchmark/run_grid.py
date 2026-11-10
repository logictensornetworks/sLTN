"""Run repro.run over sets, class counts and seeds; save every result with the sLTN version.

    conda run -n sltn python run_grid.py --sets set1 set2 --classes 8 --seeds 0 1 2 3 4 5 6 7 8 9

Results: results/<time>/results.json, and a table against the published numbers.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repro  # noqa: E402
from repro import sltn_version  # noqa: E402


def table(rows):
    lines = ["| set | |C| | runs | sLTN image acc | ∂LTLf image acc | sLTN seq acc | ∂LTLf seq acc |",
             "|---|---|---|---|---|---|---|"]
    for key in sorted({(r["set"], r["classes"]) for r in rows}):
        sel = [r for r in rows if (r["set"], r["classes"]) == key]
        img, seq = [r["img_acc"] for r in sel], [r["seq_acc"] for r in sel]
        pub = repro.PUBLISHED.get(key, (float("nan"), float("nan")))
        lines.append(f"| {key[0]} | {key[1]} | {len(sel)} | {np.mean(img):.1f} ± {np.std(img):.1f} | {pub[0]} | "
                     f"{np.mean(seq):.1f} ± {np.std(seq):.1f} | {pub[1]} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["set1", "set2"])
    ap.add_argument("--classes", type=int, nargs="+", default=[8])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--mixed", action="store_true", help="their batching: mixed lengths, padded")
    a = ap.parse_args()
    out = HERE / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(config=vars(a), sltn=sltn_version(), device=str(repro.DEVICE))
    rows = []
    for fset in a.sets:
        for C in a.classes:
            for seed in a.seeds:
                r = repro.run(fset, C // 2, seed, epochs=a.epochs, mixed=a.mixed)
                rows.append(dict(set=fset, classes=C, seed=seed, **r))
                (out / "results.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))
                print(f"{fset} |C|={C} seed {seed}: image {r['img_acc']:.1f}, sequence {r['seq_acc']:.1f} "
                      f"({r['seconds']:.0f}s)", flush=True)
    print(table(rows))
    print(f"saved {out / 'results.json'}")


if __name__ == "__main__":
    main()
