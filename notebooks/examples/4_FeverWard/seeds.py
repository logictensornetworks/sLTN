"""More seeds for the fever ward (4_FeverWard.ipynb grew out of this script): a BCE baseline, one rule removed at a time, satisfaction.

Uses the data, network and clauses of prototype.py. For every setting and seed it trains:
  bce       the same FeverNet with binary cross-entropy on the labelled days
  labels    the two label clauses
  rules     the label clauses and the three rules
  -<rule>   the label clauses and all the rules but one
and records test day accuracy, rule violations, and the satisfaction of every clause on
the training ward. Results: results/<run-id>/results.json, with the settings of the run.

Run: conda run -n sltn python seeds.py [--seeds 0 1 2 3 4] [--noise 1.0] [--n_train 200]
"""

import argparse
import itertools
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

import prototype as P
import sltn
from sltn import KB

HERE = Path(__file__).resolve().parent


def sltn_version():
    """Commit, branch and dirty flag of the sLTN checkout in use, saved with every result."""
    root = Path(sltn.__file__).resolve().parent.parent
    git = lambda *args: subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(git("status", "--porcelain", "--untracked-files=no")))

ALL = {**P.LABELS, **P.RULES}


def train_bce(w, seed, epochs):
    torch.manual_seed(seed)
    it = P.make_interp()
    P.ground(it, w)
    opt = torch.optim.Adam(it.parameters(), lr=1e-2)
    lab = w["labelled"]
    target = w["fever"][lab].float()
    for _ in range(epochs):
        opt.zero_grad()
        r = it("Fever(obs[p, d])[d]")
        assert r.axis_names[:2] == ["p", "d"]
        f = r.value.squeeze(-1)[lab].clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(f, target)
        loss.backward()
        opt.step()
    return it


def train_kb(w, clauses, seed, epochs):
    it, _ = P.train(w, clauses, seed, epochs)
    return it


@torch.no_grad()
def satisfaction(it, w):
    P.ground(it, w)
    return {k: it(f).value.item() for k, f in ALL.items()}


def models():
    out = {"bce": None, "labels": P.LABELS, "rules": ALL}
    for r in P.RULES:
        out[f"-{r}"] = {k: v for k, v in ALL.items() if k != r}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.02, 0.01])
    ap.add_argument("--relapses", type=float, nargs="+", default=[0.0, 0.1, 0.3])
    a = ap.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = HERE / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(config=vars(a), sltn=sltn_version(), length="variable", n_test=2000, p_agg=4, lr=1e-2)
    rows = []
    print(f"run {run_id}: {meta['config']}, sLTN {meta['sltn']['commit'][:7]} dirty={meta['sltn']['dirty']}")

    for frac, rel in itertools.product(a.fracs, a.relapses):
        t0 = time.perf_counter()
        for s in a.seeds:
            tr = P.make_ward(a.n_train, frac, rel, None, torch.Generator().manual_seed(1000 + s), a.noise)
            te = P.make_ward(2000, frac, rel, None, torch.Generator().manual_seed(5000 + s), a.noise)
            for name, cl in models().items():
                it = train_bce(tr, s, a.epochs) if cl is None else train_kb(tr, cl, s, a.epochs)
                rows.append(dict(frac=frac, relapse=rel, seed=s, model=name,
                                 **P.scores(it, te), sat=satisfaction(it, tr)))
        print(f"labels {frac}, relapse {rel}: {time.perf_counter() - t0:.0f}s", flush=True)
        (out_dir / "results.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))

    report(rows, a.seeds)
    print(f"\nsaved {out_dir / 'results.json'}")


def report(rows, seeds):
    names = list(models())
    get = lambda f, r, m, k: torch.tensor([x[k] for x in rows if (x["frac"], x["relapse"], x["model"]) == (f, r, m)])
    sat = lambda f, r, m, c: torch.tensor([x["sat"][c] for x in rows if (x["frac"], x["relapse"], x["model"]) == (f, r, m)])
    ms = lambda t: f"{t.mean():.3f}±{t.std():.3f}"
    settings = sorted({(x["frac"], x["relapse"]) for x in rows}, key=lambda t: (-t[0], t[1]))

    print(f"\nTest day accuracy, mean ± std over {len(seeds)} seeds (variable lengths)")
    print(f"{'labels':6s} {'relapse':7s} | " + " | ".join(f"{n:>24s}" for n in names))
    for f, r in settings:
        print(f"{f:<6} {r:<7} | " + " | ".join(f"{ms(get(f, r, n, 'acc')):>24s}" for n in names))

    print("\nPer seed: seeds where `rules` beats labels / bce, and the smallest gain over each (points)")
    for f, r in settings:
        ru, la, bc = get(f, r, "rules", "acc"), get(f, r, "labels", "acc"), get(f, r, "bce", "acc")
        print(f"{f:<6} {r:<7} | vs labels {int((ru > la).sum())}/{len(ru)}, min {100 * (ru - la).min():+.1f} | "
              f"vs bce {int((ru > bc).sum())}/{len(ru)}, min {100 * (ru - bc).min():+.1f}")

    print("\nAccuracy on the days each rule speaks about: day 0 (admitted) / two days after a treatment "
          "(treatment) / after the first recovery (no relapse)")
    for f, r in settings:
        print(f"{f:<6} {r:<7} | " + " | ".join(
            f"{n} " + " / ".join(f"{get(f, r, n, k).mean():.3f}" for k in ("acc_day0", "acc_treat_day", "acc_post_recovery"))
            for n in names))

    print("\nRule violations on test (fraction of patients: relapse / admit / treatment)")
    for f, r in settings:
        cells = []
        for n in ("bce", "labels", "rules"):
            cells.append(f"{n} " + " / ".join(f"{get(f, r, n, v).mean():.2f}"
                                              for v in ("viol_relapse", "viol_admit", "viol_treat")))
        print(f"{f:<6} {r:<7} | " + " | ".join(cells))

    print("\nSatisfaction on the training ward, mean over seeds (model `rules` / model `labels`)")
    print(f"{'labels':6s} {'relapse':7s} | " + " | ".join(f"{c[:14]:>14s}" for c in ALL))
    for f, r in settings:
        print(f"{f:<6} {r:<7} | " + " | ".join(
            f"{sat(f, r, 'rules', c).mean():.2f} / {sat(f, r, 'labels', c).mean():.2f}".rjust(14) for c in ALL))


if __name__ == "__main__":
    main()
