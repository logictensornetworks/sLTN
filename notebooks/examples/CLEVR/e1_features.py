"""Step 2 : learn the four attributes from the answers only, with invented per-object features.

Each attribute value has a fixed random embedding (EMB dims); an object's feature is the concatenation of
its four embeddings, mixed by a fixed random matrix, plus Gaussian noise (fixed per object). Four MLPs
(feature -> softmax over the values) replace perfect sight; the formulas are those of E0. Training uses
only the rules (answers); the scene files' attributes are used only to score.

Run: conda run -n sltn python e1_features.py [--steps 3000] [--noise 0.5]
"""

import argparse
import collections
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

import sltn
from sltn import Interpretation, Logic
from sltn.interpretation.operators.aggregator import AggregMax, AggregPMeanError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clevr_sltn import (ANS, ATTRS, DIRS, IS, KVAR, PARAM, SORT, VALUES, O, compile_question,  # noqa: E402
                        DATA, RESULTS, in_scope, load, make_signature, question_params)
EMB = 16


def scene_arrays(scenes):
    """attributes [S, O, 4] (value index, -1 = empty), present [S, O], relations [S, 4 dirs, O, O]."""
    S = len(scenes)
    attr = torch.full((S, O, len(ATTRS)), -1, dtype=torch.long)
    rel = torch.zeros(S, len(DIRS), O, O)
    for s, sc in enumerate(scenes):
        for i, ob in enumerate(sc["objects"]):
            for j, a in enumerate(ATTRS):
                attr[s, i, j] = VALUES[a].index(ob[a])
        for d, name in enumerate(DIRS):
            for u, objs in enumerate(sc["relationships"][name]):
                rel[s, d, u, objs] = 1
    return attr, attr[..., 0] >= 0, rel


def make_features(attr, present, emb, mix, noise, g):
    parts = [emb[a][attr[..., j].clamp(min=0)] for j, a in enumerate(ATTRS)]
    f = torch.cat(parts, -1) @ mix
    f = f + noise * torch.randn(f.shape, generator=g)
    return f * present[..., None]


def family_tensors(questions):
    """Group in-scope questions by formula; per family, the groundings as tensors (scene index, relation
    direction indices, filter values, ground truth)."""
    groups = collections.defaultdict(list)
    comp = {}
    for q in questions:
        if not in_scope(q):
            continue
        c = compile_question(q)
        groups[c.key].append(q)
        comp.setdefault(c.key, c)
    fams = {}
    for key, qs in groups.items():
        c = comp[key]
        N = len(qs)
        t = dict(scene=torch.tensor([q["image_index"] for q in qs]),
                 rel=torch.zeros(N, c.n_rels, dtype=torch.long),
                 slots={a: torch.ones(c.n_slots, N, len(VALUES[a])) for a in ATTRS},
                 yes=torch.zeros(N), ans={a: torch.zeros(N, len(VALUES[a])) for a in ATTRS})
        for b, q in enumerate(qs):
            fil, rels = question_params(q)
            for k, d in enumerate(rels):
                t["rel"][b, k] = DIRS.index(d)
            for k, vals in enumerate(fil):
                for a, v in vals.items():
                    t["slots"][a][k, b] = 0
                    t["slots"][a][k, b, VALUES[a].index(v)] = 1
            if q["answer"] in ("yes", "no"):
                t["yes"][b] = q["answer"] == "yes"
            else:
                t["ans"][c.kind][b, VALUES[c.kind].index(q["answer"])] = 1
        fams[key] = (c, t)
    return fams


class AttrNet(nn.Module):
    def __init__(self, d_in, n_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        return torch.softmax(self.net(x), -1)


def dot(p, q):
    # clamped: with an all-ones filter value, the sum of a softmax can be 1 + 1e-7, which isn't a truth value
    return (p * q).sum(-1, keepdim=True).clamp(0, 1)


def make_interp(sig, nets, d_in):
    logic = Logic().with_defaults()
    logic["exists,Obj"] = AggregMax()
    logic["forall,Question"] = AggregPMeanError(p=2)
    it = Interpretation(sig, logic=logic)
    it["Feat"] = sltn.Type("Feat", shape=(d_in,))
    for a in ATTRS:
        n = len(VALUES[a])
        it[SORT[a]] = sltn.Type(SORT[a], shape=(n,))
        it[a] = nets[a]
        it[IS[a]] = dot
        it[KVAR[a]] = torch.eye(n)
    it["Neq"] = 1 - torch.eye(O)
    return it


def ground(it, c, t, idx, feats, present, rel):
    s = t["scene"][idx]
    it["x"], it["Present"] = feats[s], present[s].float()
    it["Yes"] = t["yes"][idx]
    for k in range(c.n_rels):
        it[f"R{k}"] = rel[s, t["rel"][idx, k]]
    for a in ATTRS:
        it[ANS[a]] = t["ans"][a][idx]
        for k in range(c.n_slots):
            it[f"{PARAM[a]}{k}"] = t["slots"][a][k, idx]


def by_name(r, order):
    names = [ax.name for ax in r.axes][:-1]
    return r.value.squeeze(-1).permute(*[names.index(n) for n in order])


@torch.no_grad()
def concept_acc(nets, feats, attr, present):
    m = present
    out = {}
    for j, a in enumerate(ATTRS):
        pred = nets[a](feats[m]).argmax(-1)
        out[a] = (pred == attr[m][:, j]).float().mean().item()
    return out


@torch.no_grad()
def qa_acc(it, sig, fams, feats, present, rel, chunk=4096):
    by_type = collections.defaultdict(lambda: [0, 0])
    for key, (c, t) in fams.items():
        phi = sig.parse(c.body)
        N = len(t["scene"])
        typ = "query" if c.kind != "yesno" else ("compare" if key.startswith("equal_") else "exist")
        for s0 in range(0, N, chunk):
            idx = torch.arange(s0, min(N, s0 + chunk))
            ground(it, c, t, idx, feats, present, rel)
            r = it(phi)
            if c.kind == "yesno":
                ok = (by_name(r, ["q"]) > 0.5) == t["yes"][idx].bool()
            else:
                ok = by_name(r, ["q", KVAR[c.kind]]).argmax(-1) == t["ans"][c.kind][idx].argmax(-1)
            by_type[typ][0] += int(ok.sum())
            by_type[typ][1] += len(idx)
    tot = sum(v[0] for v in by_type.values()) / sum(v[1] for v in by_type.values())
    return tot, {k: v[0] / v[1] for k, v in by_type.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--fams_per_step", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train_q", type=int, default=None, help="use only this many training questions")
    ap.add_argument("--log_every", type=int, default=250)
    a = ap.parse_args()
    print(vars(a), flush=True)
    t0 = time.perf_counter()
    g = torch.Generator().manual_seed(1000 + a.seed)
    emb = {at: torch.randn(len(VALUES[at]), EMB, generator=g) for at in ATTRS}
    D = EMB * len(ATTRS)
    mix = torch.linalg.qr(torch.randn(D, D, generator=g))[0]          # a random rotation of the 64 dims

    data = {}
    for split in ("train", "val"):
        scenes, questions = load(DATA, split)
        if split == "train" and a.n_train_q:
            questions = questions[: a.n_train_q]
        attr, present, rel = scene_arrays(scenes)
        feats = make_features(attr, present, emb, mix, a.noise, g)
        data[split] = dict(attr=attr, present=present, rel=rel, feats=feats, fams=family_tensors(questions))
        print(f"{split}: {len(scenes)} scenes, {sum(len(t['scene']) for _, t in data[split]['fams'].values())} "
              f"in-scope questions, {len(data[split]['fams'])} formulas ({time.perf_counter() - t0:.0f}s)", flush=True)

    torch.manual_seed(a.seed)
    sig = make_signature()
    nets = nn.ModuleDict({at: AttrNet(D, len(VALUES[at])) for at in ATTRS})
    it = make_interp(sig, nets, D)
    opt = torch.optim.Adam(nets.parameters(), lr=a.lr)
    tr = data["train"]
    keys = list(tr["fams"])
    rules = {k: sig.parse(tr["fams"][k][0].rule) for k in keys}
    sizes = torch.tensor([len(tr["fams"][k][1]["scene"]) for k in keys], dtype=torch.float)
    va = data["val"]
    acc0 = concept_acc(nets, va["feats"], va["attr"], va["present"])
    print("step 0 concept acc (val objects): " + ", ".join(f"{k} {v:.3f}" for k, v in acc0.items()), flush=True)
    run_loss, t1 = 0.0, time.perf_counter()
    for step in range(1, a.steps + 1):
        fam_idx = torch.multinomial(sizes, a.fams_per_step, replacement=False, generator=g)
        sats = []
        for fi in fam_idx.tolist():
            c, t = tr["fams"][keys[fi]]
            idx = torch.randint(0, len(t["scene"]), (a.batch,), generator=g)
            ground(it, c, t, idx, tr["feats"], tr["present"], tr["rel"])
            sats.append(it(rules[keys[fi]]).value.squeeze())
        loss = 1 - torch.stack(sats).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        run_loss += loss.item()
        if step % a.log_every == 0:
            acc = concept_acc(nets, va["feats"], va["attr"], va["present"])
            print(f"step {step:5d} loss {run_loss / a.log_every:.4f} | concept acc (val): "
                  + ", ".join(f"{k} {v:.3f}" for k, v in acc.items())
                  + f" | {(time.perf_counter() - t1) / step * 1000:.0f} ms/step", flush=True)
            run_loss = 0.0
    tot, per = qa_acc(it, sig, va["fams"], va["feats"], va["present"], va["rel"])
    acc = concept_acc(nets, va["feats"], va["attr"], va["present"])
    print(f"\nfinal concept acc (val objects): " + ", ".join(f"{k} {v:.3f}" for k, v in acc.items()))
    print(f"final QA acc (val, in scope): {tot:.4f} | " + ", ".join(f"{k} {v:.4f}" for k, v in per.items()))
    # the same QA with perfect concepts would be 1.0 (E0); chance per attribute: 1/8, 1/2, 1/2, 1/3
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
