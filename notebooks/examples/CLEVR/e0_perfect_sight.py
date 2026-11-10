"""E0, perfect sight: every in-scope CLEVR question answered by its family's sLTN formula, with the
objects' attributes read from the scene file (one-hot, nothing learned). The formula must reproduce
the dataset's answer on every question; this checks the program -> formula translation.

Run: conda run -n sltn python e0_perfect_sight.py [--split val] [--limit N]
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch

import sltn
from sltn import Interpretation, Logic
from sltn.interpretation.operators.aggregator import AggregMax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clevr_sltn import (ANS, ATTRS, FEAT, IS, KVAR, OFFSET, PARAM, SORT, VALUES, O, batch_tensors,  # noqa: E402
                        DATA, RESULTS, compile_question, in_scope, load, make_signature, scene_tensors, sltn_version)


def make_interp(sig):
    logic = Logic().with_defaults()
    logic["exists,Obj"] = AggregMax()
    logic["forall,Question"] = sltn.interpretation.operators.aggregator.AggregPMeanError(p=2)
    it = Interpretation(sig, logic=logic)
    it["Feat"] = sltn.Type("Feat", shape=(FEAT,))
    for a in ATTRS:
        n = len(VALUES[a])
        lo, hi = OFFSET[a]
        it[SORT[a]] = sltn.Type(SORT[a], shape=(n,))
        it[a] = (lambda lo, hi: lambda x: x[..., lo:hi])(lo, hi)     # perfect sight: read the one-hot
        it[IS[a]] = lambda p, q: (p * q).sum(-1, keepdim=True)
        it[KVAR[a]] = torch.eye(n)
    it["Neq"] = 1 - torch.eye(O)
    return it


def ground(it, t, n_slots, n_rels):
    it["x"], it["Present"], it["Yes"] = t["x"], t["present"], t["yes"]
    for k in range(n_rels):
        it[f"R{k}"] = t["rel"][k]
    for a in ATTRS:
        it[ANS[a]] = t["ans"][a]
        for k in range(n_slots):
            it[f"{PARAM[a]}{k}"] = t["slots"][a][k]


def values(r, order):
    names = [a.name for a in r.axes][:-1]
    return r.value.squeeze(-1).permute(*[names.index(n) for n in order])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None, help="questions per family (default: all)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    scenes, questions = load(DATA, a.split)
    cache = {s["image_index"]: scene_tensors(s) for s in scenes}
    qs = [q for q in questions if in_scope(q)]
    print(f"{a.split}: {len(questions)} questions, {len(qs)} in scope; loaded in {time.perf_counter() - t0:.0f}s")

    groups = collections.defaultdict(list)
    for q in qs:
        groups[compile_question(q).key].append(q)
    sig = make_signature()
    it = make_interp(sig)
    rows, wrong_examples = [], []
    for key, group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        group = group[: a.limit] if a.limit else group
        c = compile_question(group[0])
        assert all(compile_question(q).body == c.body for q in group[:50]), key
        body, rule = sig.parse(c.body), sig.parse(c.rule)
        t = batch_tensors(group, cache, c.n_slots, c.n_rels)
        ground(it, t, c.n_slots, c.n_rels)
        with torch.no_grad():
            r = it(body)
            sat = it(rule).value.item()
        if c.kind == "yesno":
            v = values(r, ["q"])
            pred = v > 0.5
            ok = pred == t["yes"].bool()
            crisp = ((v < 1e-3) | (v > 1 - 1e-3)).float().mean().item()
        else:
            v = values(r, ["q", KVAR[c.kind]])
            ok = v.argmax(-1) == t["ans"][c.kind].argmax(-1)
            crisp = ((v > 0.5).sum(-1) == 1).float().mean().item()      # exactly one value true
        fams = sorted({q["question_family_index"] for q in group})
        rows.append(dict(key=key, families=fams, kind=c.kind, n=len(group), acc=ok.float().mean().item(),
                         crisp=crisp, sat=sat))
        for i in torch.nonzero(~ok).flatten()[:3].tolist():
            wrong_examples.append(dict(key=key, question=group[i]["question"], answer=group[i]["answer"],
                                       image=group[i]["image_index"]))
        print(f"{len(group):5d} acc {rows[-1]['acc']:.4f} crisp {crisp:.3f} sat {sat:.4f} | {c.kind:8s} | {key}",
              flush=True)

    n = sum(r["n"] for r in rows)
    acc = sum(r["acc"] * r["n"] for r in rows) / n
    by_kind = collections.defaultdict(lambda: [0, 0.0])
    for r in rows:
        k = "query" if r["kind"] != "yesno" else r["key"].split("(")[0]
        k = "compare" if k.startswith("equal_") else k
        by_kind[k][0] += r["n"]
        by_kind[k][1] += r["acc"] * r["n"]
    print(f"\n{len(rows)} formulas, {n} questions: agreement with the dataset {acc:.5f}")
    for k, (m, s) in by_kind.items():
        print(f"  {k:8s} {m:6d} questions, agreement {s / m:.5f}")
    print(f"min rule satisfaction {min(r['sat'] for r in rows):.4f}; total {time.perf_counter() - t0:.0f}s")
    for w in wrong_examples[:10]:
        print("  wrong:", w)
    out = RESULTS / "e0" / (a.split if a.limit is None else f"{a.split}_limit{a.limit}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(dict(config=vars(a), sltn=sltn_version(), agreement=acc, n=n,
                                                      rows=rows, wrong=wrong_examples), indent=1))
    print(f"saved {out / 'results.json'}")


if __name__ == "__main__":
    main()
