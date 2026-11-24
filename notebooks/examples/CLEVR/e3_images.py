"""E3: learn colour, size, material and shape from CLEVR images, from the answers only, with E3's formulas.

The same networks, crops, data and training loop as e1_images.py; only the formulas change: every
question is one individual of the diagonal quantifier `forall (x, c0, ..., ans_c)` (clevr_diag.py,
e3_named_variables.ipynb), and an object's features are one vector

    x[o] = [present, 3D coordinates (3), the CNN embedding of the object's crop (256)]

C, Z, M, S read the embedding through the network's heads (linear + softmax) and compare with a value;
SameC & co. compare two objects' heads. R reads the coordinates: by default CLEVR's rule (exact, as in the
notebook), with --learn_relations a small network on the two objects' 3D coordinates.

No attribute label is used; the scene files' attributes only score the result (concept accuracy on the val
objects, QA accuracy on the val questions answered by the formulas with the learned concepts).

Needs the crops of prep_crops.py. Run (GPU): python e3_images.py [--steps 4500] [--learn_relations] [--seed 0]
"""

import argparse
import collections
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clevr_diag as cd  # noqa: E402
from clevr_sltn import CROPS, DATA, RESULTS, load, scene_arrays, sltn_version  # noqa: E402
from e1_images import ConceptNet, subset  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB = slice(4, None)                                  # the embedding part of x[o]


def dot(p, q):
    # clamped: with an all-ones filter value, the sum of a softmax can be 1 + 1e-7, which isn't a truth value
    return (p * q).sum(-1, keepdim=True).clamp(0, 1)


class Has(nn.Module):
    """C(x[o], v), ...: the head's distribution for the object, dotted with the value v."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x, v):
        return dot(torch.softmax(self.linear(x[..., EMB]), -1), v)


class Same(nn.Module):
    """SameC(x[o], x[o1]), ...: the two objects' distributions, dotted."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x, x1):
        return dot(torch.softmax(self.linear(x[..., EMB]), -1), torch.softmax(self.linear(x1[..., EMB]), -1))


class Relation(nn.Module):
    """R(x[o], x[o1], r): o1 is in direction r of o. Rule: CLEVR's (exact); learned: an MLP on the 3D
    coordinates of both objects and their difference, one sigmoid per direction."""

    def __init__(self, directions, learn, hidden=64):
        super().__init__()
        self.register_buffer("directions", directions)
        self.mlp = nn.Sequential(nn.Linear(9, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, len(cd.DIRS))) if learn else None

    def probs(self, xyz, xyz1):
        if self.mlp is None:
            return ((xyz1 - xyz) @ self.directions.T > 0).float()
        return torch.sigmoid(self.mlp(torch.cat([xyz, xyz1, xyz1 - xyz], -1)))

    def forward(self, x, x1, r):
        return (self.probs(x[..., cd.XYZ], x1[..., cd.XYZ]) * r).sum(-1, keepdim=True)


def make_interp(sig, net, relation):
    logic = cd.Logic().with_defaults()
    logic["exists,Obj"] = cd.AggregMax()
    logic["forall"] = cd.AggregPMeanError(p=2)
    it = cd.Interpretation(sig, logic=logic)
    D = net.heads[cd.ATTRS[0]].in_features
    it["Obj"] = range(cd.O)
    it["Object"] = cd.sltn.Type("Object", shape=(4 + D,))
    it["Dir"] = cd.sltn.Type("Dir", shape=(len(cd.DIRS),))
    it["Neq"] = 1 - torch.eye(cd.O, device=DEVICE)
    it["Present"] = lambda x: x[..., cd.PRESENT:cd.PRESENT + 1]
    it["R"] = relation
    for a in cd.ATTRS:
        it[cd.SORT[a]] = cd.sltn.Type(cd.SORT[a], shape=(len(cd.VALUES[a]),))
        it[cd.PRED[a]] = Has(net.heads[a])
        it[cd.SAME[a]] = Same(net.heads[a])
        it[cd.KVAR[a]] = torch.eye(len(cd.VALUES[a]), device=DEVICE)
    return it


def family_tensors(questions):
    """{skeleton: (Compiled, {variable: tensor [N, ...] on DEVICE, plus `scene` [N]})} for the in-scope questions."""
    groups = collections.defaultdict(list)
    for q in questions:
        if cd.in_scope(q):
            groups[cd.canonical(q)[0]].append(q)
    out = {}
    for key, qs in groups.items():
        c = cd.compile_question(qs[0])
        out[key] = (c, {k: v.to(DEVICE) for k, v in cd.family_params(qs).items()})
    return out


def embed_scenes(net, crops, present):
    """[B, O, ...] crops, [B, O] present -> [B, O, D] embeddings, zero for an empty position."""
    emb = torch.zeros(crops.shape[0], cd.O, net.heads[cd.ATTRS[0]].in_features, device=DEVICE)
    emb[present] = net.embed(crops[present])
    return emb


def ground(it, c, t, idx, emb, sd):
    """Ground the family's variables for questions idx; emb [len(idx), O, D] are their objects' embeddings."""
    s = t["scene"][idx]
    it["x"] = torch.cat([sd["present"][s].float()[..., None], sd["xyz"][s], emb], -1)
    for name in c.variables[1:] + (["yes"] if c.kind == "yesno" else [cd.ANS[c.kind]]):
        it[name] = t[name][idx]


@torch.no_grad()
def concept_acc(net, sd, max_scenes=None, chunk=256):
    net.eval()
    S = len(sd["crops"]) if max_scenes is None else min(max_scenes, len(sd["crops"]))
    hit, n = {a: 0 for a in cd.ATTRS}, 0
    for s0 in range(0, S, chunk):
        sl = slice(s0, min(S, s0 + chunk))
        m = sd["present"][sl]
        probs = net(sd["crops"][sl][m])
        for j, a in enumerate(cd.ATTRS):
            hit[a] += int((probs[a].argmax(-1) == sd["attr"][sl][m][:, j]).sum())
        n += int(m.sum())
    net.train()
    return {a: hit[a] / n for a in cd.ATTRS}


@torch.no_grad()
def relation_acc(relation, sd, max_scenes=None):
    """Accuracy per direction over ordered pairs of distinct real objects, against CLEVR's relations."""
    S = len(sd["xyz"]) if max_scenes is None else min(max_scenes, len(sd["xyz"]))
    xyz, pres, true = sd["xyz"][:S], sd["present"][:S], sd["rel"][:S].bool()          # true [S, 4, O, O]
    pred = relation.probs(xyz[:, :, None, :].expand(-1, -1, cd.O, -1), xyz[:, None, :, :].expand(-1, cd.O, -1, -1)) > 0.5
    pair = pres[:, :, None] & pres[:, None, :] & ~torch.eye(cd.O, dtype=torch.bool, device=DEVICE)
    return {d: (pred[..., i] == true[:, i])[pair].float().mean().item() for i, d in enumerate(cd.DIRS)}


@torch.no_grad()
def qa_acc(net, it, sig, fams, sd, chunk=256):
    """Each question's reply (cd.per_question) against its answer, by question type."""
    net.eval()
    res = {}
    for key, (c, t) in fams.items():
        phi = sig.parse(c.predict)
        N, ok_all = len(t["scene"]), 0
        for s0 in range(0, N, chunk):
            idx = torch.arange(s0, min(N, s0 + chunk), device=DEVICE)
            s = t["scene"][idx]
            ground(it, c, t, idx, embed_scenes(net, sd["crops"][s], sd["present"][s]), sd)
            v = cd.per_question(it, phi)
            if c.kind == "yesno":
                ok = (v > 0.5) == (t["yes"][idx].squeeze(-1) > 0.5)
            else:
                ok = v.argmax(-1) == t[cd.ANS[c.kind]][idx].argmax(-1)
            ok_all += int(ok.sum())
        typ = ("exist" if key.startswith("exist") else "compare" if key.startswith("equal_") else "query") \
            + ("+relate" if c.n_rels else "")
        h, n = res.get(typ, (0, 0))
        res[typ] = (h + ok_all, n + N)
    net.train()
    total = sum(h for h, _ in res.values()) / sum(n for _, n in res.values())
    return total, {k: h / n for k, (h, n) in res.items()}


def start_tracking(config, seed, enabled):
    """W&B through sltn-applications' common.tracking (GPU runs only); None if disabled or unavailable."""
    if not enabled:
        return None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "sltn-applications"))
        from common.tracking import start_run
    except ImportError as e:
        print(f"no W&B tracking ({e})", flush=True)
        return None
    return start_run("clevr-e3", config, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--fams_per_step", type=int, default=8)
    ap.add_argument("--batch", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train_q", type=int, default=None, help="train on N in-scope questions (default: all)")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_scenes", type=int, default=2000, help="val scenes for the periodic concept accuracy")
    ap.add_argument("--learn_relations", action="store_true", help="learn R from the 3D coordinates")
    ap.add_argument("--crops", type=Path, default=CROPS, help="folder with <split>_crops.npy")
    ap.add_argument("--max_scenes", type=int, default=None, help="smoke tests: keep only the first N scenes per split")
    ap.add_argument("--results", type=Path, default=RESULTS / "e3", help="where the run folder goes")
    ap.add_argument("--no_wandb", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    sd, directions = {}, None
    for split in ("train", "val"):
        scenes, questions = load(DATA, split)
        directions = cd.camera_directions(scenes) if directions is None else directions
        if a.max_scenes:
            scenes = scenes[:a.max_scenes]
            questions = [q for q in questions if q["image_index"] < a.max_scenes]
        if split == "train" and a.n_train_q:
            questions = subset(questions, a.n_train_q, a.seed)
        attr, present, rel = scene_arrays(scenes)
        xyz = torch.zeros(len(scenes), cd.O, 3)
        for s, sc in enumerate(scenes):
            for i, ob in enumerate(sc["objects"]):
                xyz[s, i] = torch.tensor(ob["3d_coords"])
        crops = torch.from_numpy(np.load(a.crops / f"{split}_crops.npy", mmap_mode="r")[:len(scenes)].copy())
        sd[split] = dict(attr=attr.to(DEVICE), present=present.to(DEVICE), rel=rel.to(DEVICE), xyz=xyz.to(DEVICE),
                         crops=crops.to(DEVICE), fams=family_tensors(questions))
        n_q = sum(len(t["scene"]) for _, t in sd[split]["fams"].values())
        print(f"{split}: {len(scenes)} scenes, {n_q} in-scope questions ({time.perf_counter() - t0:.0f}s)", flush=True)

    sig = cd.make_signature()
    net = ConceptNet().to(DEVICE)
    relation = Relation(directions, a.learn_relations).to(DEVICE)
    it = make_interp(sig, net, relation)
    opt = torch.optim.Adam(list(net.parameters()) + list(relation.parameters()), lr=a.lr)
    tr, va = sd["train"], sd["val"]
    keys = list(tr["fams"])
    rules = {k: sig.parse(tr["fams"][k][0].rule) for k in keys}
    sizes = torch.tensor([len(tr["fams"][k][1]["scene"]) for k in keys], dtype=torch.float)
    config = dict({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                  n_train_inscope=int(sizes.sum()), crop=48, crop_k=500, device=str(DEVICE), formulas="e3")
    tag = ("-relations" if a.learn_relations else "") + (f"-n{a.n_train_q}" if a.n_train_q else "") + f"-seed{a.seed}"
    out = a.results / (datetime.now().strftime("%Y%m%d-%H%M%S") + tag)
    out.mkdir(parents=True, exist_ok=True)
    print(f"sltn {sltn_version()['commit'][:7]} | {config}", flush=True)
    run = start_tracking(config, a.seed, DEVICE.type == "cuda" and not a.no_wandb)

    run_loss, t1 = 0.0, time.perf_counter()
    for step in range(1, a.steps + 1):
        fam_idx = torch.multinomial(sizes, min(a.fams_per_step, len(keys)), replacement=False, generator=g).tolist()
        picks = []
        for fi in fam_idx:
            t = tr["fams"][keys[fi]][1]
            picks.append((fi, torch.randint(0, len(t["scene"]), (a.batch,), generator=g).to(DEVICE)))
        scene_ids = torch.cat([tr["fams"][keys[fi]][1]["scene"][idx] for fi, idx in picks])
        emb = embed_scenes(net, tr["crops"][scene_ids], tr["present"][scene_ids])   # every object, once
        sats = []
        for j, (fi, idx) in enumerate(picks):
            c, t = tr["fams"][keys[fi]]
            ground(it, c, t, idx, emb[j * a.batch:(j + 1) * a.batch], tr)
            sats.append(it(rules[keys[fi]]).value.squeeze())
        loss = 1 - torch.stack(sats).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        run_loss += loss.item()
        if step % a.eval_every == 0:
            acc = concept_acc(net, va, a.eval_scenes)
            racc = relation_acc(relation, va, a.eval_scenes) if a.learn_relations else {}
            ms = (time.perf_counter() - t1) / step * 1000
            print(f"step {step:5d} loss {run_loss / a.eval_every:.4f} | concept acc (val, {a.eval_scenes} scenes): "
                  + ", ".join(f"{k} {v:.3f}" for k, v in acc.items())
                  + "".join(f", {k} {v:.3f}" for k, v in racc.items()) + f" | {ms:.0f} ms/step", flush=True)
            if run is not None:
                run.log({"loss": run_loss / a.eval_every, **{f"concept_acc/{k}": v for k, v in acc.items()},
                         **{f"relation_acc/{k}": v for k, v in racc.items()}}, step=step)
            run_loss = 0.0

    acc = concept_acc(net, va)
    qa, per = qa_acc(net, it, sig, va["fams"], va)
    racc = relation_acc(relation, va) if a.learn_relations else {}
    summary = {**{f"concept_acc/{k}": v for k, v in acc.items()}, **{f"relation_acc/{k}": v for k, v in racc.items()},
               "qa_acc": qa, **{f"qa_acc/{k}": v for k, v in per.items()}}
    print("\nfinal concept acc (all val objects): " + ", ".join(f"{k} {v:.4f}" for k, v in acc.items()))
    if racc:
        print("final relation acc (all val pairs): " + ", ".join(f"{k} {v:.4f}" for k, v in racc.items()))
    print(f"final QA acc (val, in scope): {qa:.4f} | " + ", ".join(f"{k} {v:.4f}" for k, v in per.items()))
    (out / "results.json").write_text(json.dumps(dict(config=config, sltn=sltn_version(), **summary), indent=1))
    torch.save(net.state_dict(), out / "concept_net.pt")
    if a.learn_relations:
        torch.save(relation.state_dict(), out / "relation_net.pt")
    if run is not None:
        run.summary.update(summary)
        run.finish()
    print(f"saved {out}; total {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
