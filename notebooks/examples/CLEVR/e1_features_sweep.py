"""Sweep for step 2: noise x training questions, sLTN (answers only) vs supervised (attribute labels).

Run: conda run -n sltn python e1_features_sweep.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_features import (ATTRS, DATA, RESULTS, EMB, VALUES, AttrNet, concept_acc, family_tensors, load,  # noqa: E402
                         make_features, make_interp, make_signature, qa_acc, scene_arrays, ground)

NOISES = [1.0, 2.0, 3.0]
N_QS = [3000, 30000]          # first questions of the training file (about 2/3 are in scope)
STEPS, FAMS, BATCH, SEED = 1000, 8, 64, 0


def train_sltn(tr, fams, D, steps, g):
    torch.manual_seed(SEED)
    sig = make_signature()
    nets = nn.ModuleDict({a: AttrNet(D, len(VALUES[a])) for a in ATTRS})
    it = make_interp(sig, nets, D)
    opt = torch.optim.Adam(nets.parameters(), lr=1e-3)
    keys = list(fams)
    rules = {k: sig.parse(fams[k][0].rule) for k in keys}
    sizes = torch.tensor([len(fams[k][1]["scene"]) for k in keys], dtype=torch.float)
    for _ in range(steps):
        sats = []
        for fi in torch.multinomial(sizes, min(FAMS, len(keys)), replacement=False, generator=g).tolist():
            c, t = fams[keys[fi]]
            idx = torch.randint(0, len(t["scene"]), (BATCH,), generator=g)
            ground(it, c, t, idx, tr["feats"], tr["present"], tr["rel"])
            sats.append(it(rules[keys[fi]]).value.squeeze())
        loss = 1 - torch.stack(sats).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return nets, it, sig


def train_supervised(tr, scene_ids, D, steps, g):
    # the same networks, cross-entropy on the true attributes of the objects of the same training scenes
    torch.manual_seed(SEED)
    nets = nn.ModuleDict({a: AttrNet(D, len(VALUES[a])) for a in ATTRS})
    opt = torch.optim.Adam(nets.parameters(), lr=1e-3)
    m = tr["present"][scene_ids]
    X, Y = tr["feats"][scene_ids][m], tr["attr"][scene_ids][m]
    for _ in range(steps):
        idx = torch.randint(0, len(X), (FAMS * BATCH,), generator=g)
        loss = sum(F.nll_loss(torch.log(nets[a](X[idx]) + 1e-9), Y[idx, j]) for j, a in enumerate(ATTRS))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return nets, len(X)


def main():
    t0 = time.perf_counter()
    g = torch.Generator().manual_seed(1000 + SEED)
    emb = {a: torch.randn(len(VALUES[a]), EMB, generator=g) for a in ATTRS}
    D = EMB * len(ATTRS)
    mix = torch.linalg.qr(torch.randn(D, D, generator=g))[0]
    raw = {}
    for split in ("train", "val"):
        scenes, questions = load(DATA, split)
        attr, present, rel = scene_arrays(scenes)
        raw[split] = dict(attr=attr, present=present, rel=rel, questions=questions)
    fams_tr = {n: family_tensors(raw["train"]["questions"][:n]) for n in N_QS}
    fams_va = family_tensors(raw["val"]["questions"])
    print(f"loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    rows = []
    for noise in NOISES:
        d = {s: dict(raw[s], feats=make_features(raw[s]["attr"], raw[s]["present"], emb, mix, noise,
                                                 torch.Generator().manual_seed(7 + (s == "val"))))
             for s in raw}
        for n in N_QS:
            fams = fams_tr[n]
            n_inscope = sum(len(t["scene"]) for _, t in fams.values())
            scene_ids = torch.unique(torch.cat([t["scene"] for _, t in fams.values()]))
            t1 = time.perf_counter()
            nets, it, sig = train_sltn(d["train"], fams, D, STEPS, torch.Generator().manual_seed(SEED))
            acc = concept_acc(nets, d["val"]["feats"], d["val"]["attr"], d["val"]["present"])
            qa, per = qa_acc(it, sig, fams_va, d["val"]["feats"], d["val"]["present"], d["val"]["rel"])
            rows.append(dict(noise=noise, n_questions=n_inscope, model="sltn", **acc, qa=qa, **{f"qa_{k}": v for k, v in per.items()}))
            nets_s, n_obj = train_supervised(d["train"], scene_ids, D, STEPS, torch.Generator().manual_seed(SEED))
            acc_s = concept_acc(nets_s, d["val"]["feats"], d["val"]["attr"], d["val"]["present"])
            qa_s, per_s = qa_acc(make_interp(sig, nets_s, D), sig, fams_va, d["val"]["feats"], d["val"]["present"], d["val"]["rel"])
            rows.append(dict(noise=noise, n_questions=n_inscope, model="supervised", n_labelled_objects=n_obj, **acc_s, qa=qa_s,
                             **{f"qa_{k}": v for k, v in per_s.items()}))
            for r in rows[-2:]:
                print(f"noise {noise} | {n_inscope:6d} questions ({len(scene_ids)} scenes) | {r['model']:10s} | "
                      + " ".join(f"{a} {r[a]:.3f}" for a in ATTRS)
                      + f" | QA {r['qa']:.4f} (exist {r['qa_exist']:.3f}, query {r['qa_query']:.3f}, compare {r['qa_compare']:.3f})",
                      flush=True)
            print(f"   ({time.perf_counter() - t1:.0f}s)", flush=True)
    out = RESULTS / "e1_features" / "sweep_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(steps=STEPS, fams=FAMS, batch=BATCH, seed=SEED, rows=rows), indent=1))
    print(f"saved {out}; total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
