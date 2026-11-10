"""E1 (proof of concept): learn colour, size, material and shape from CLEVR images, from the answers only.

One crop per object (prep_crops.py). A small CNN with four heads (one softmax per attribute) is trained so
that the rule of each question family (E0's formulas) is satisfied; no attribute label is used. The scene
files' attributes only score the result: concept accuracy on the val objects, and QA accuracy on the val
questions answered by the formulas with the learned concepts.

Grounding: `x[q, o]` is the CNN trunk's embedding of the object's crop, computed once per step (gradients
flow through a constant grounded with a tensor), and `color`, `size`, `material`, `shape` are the heads
(linear layer + softmax), grounded as modules. The trunk runs only on real objects; an empty position gets
a zero embedding, and `Present` keeps it out of every formula.

Needs the crops of prep_crops.py. Run (GPU): python e1_images.py [--steps 4500] [--learn_relations] [--seed 0]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sltn
from sltn import Interpretation, Logic
from sltn.interpretation.operators.aggregator import AggregMax, AggregPMeanError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clevr_sltn import (ANS, ATTRS, DIRS, IS, KVAR, PARAM, SORT, VALUES, O, by_name, family_tensors,  # noqa: E402
                        CROPS, DATA, RESULTS, load, make_signature, question_type, scene_arrays, sltn_version)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConceptNet(nn.Module):
    """Trunk: [N, 48, 48, 3] uint8 crops -> [N, 4w] embeddings; heads: embedding -> softmax per attribute."""

    def __init__(self, width=64):
        super().__init__()
        w = width
        block = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                           nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                           nn.MaxPool2d(2))
        self.trunk = nn.Sequential(block(3, w), block(w, 2 * w), block(2 * w, 2 * w), block(2 * w, 4 * w),
                                   nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.heads = nn.ModuleDict({a: nn.Linear(4 * w, len(VALUES[a])) for a in ATTRS})

    def embed(self, crops):
        return self.trunk(crops.permute(0, 3, 1, 2).float().div(255).sub(0.5))

    def forward(self, crops):
        h = self.embed(crops)
        return {a: torch.softmax(head(h), -1) for a, head in self.heads.items()}


class Head(nn.Module):
    """The attribute function: embedding -> distribution over the attribute's values."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, h):
        return torch.softmax(self.linear(h), -1)


class RelationNet(nn.Module):
    """--learn_relations: positions [B, O, 3] -> [B, 4 directions, O, O], R[b, d, u, o] = P(o is d of u)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(9, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, len(DIRS)))

    def forward(self, pos):
        B = pos.shape[0]
        u = pos[:, :, None, :].expand(B, O, O, 3)
        o = pos[:, None, :, :].expand(B, O, O, 3)
        return torch.sigmoid(self.mlp(torch.cat([u, o, o - u], -1))).permute(0, 3, 1, 2)


def positions(scenes):
    """[S, O, 3]: pixel x / 480, pixel y / 320, depth / 15 (zeros for empty positions)."""
    pos = torch.zeros(len(scenes), O, 3)
    for s, sc in enumerate(scenes):
        for i, ob in enumerate(sc["objects"]):
            x, y, d = ob["pixel_coords"]
            pos[s, i] = torch.tensor([x / 480, y / 320, d / 15])
    return pos


@torch.no_grad()
def relation_acc(relnet, sd, max_scenes=None):
    """Accuracy per direction over ordered pairs of distinct real objects, against CLEVR's adjacency."""
    S = len(sd["pos"]) if max_scenes is None else min(max_scenes, len(sd["pos"]))
    hit, n = torch.zeros(len(DIRS)), 0
    for s0 in range(0, S, 2048):
        sl = slice(s0, min(S, s0 + 2048))
        pred, true, pres = relnet(sd["pos"][sl]) > 0.5, sd["rel"][sl].bool(), sd["present"][sl]
        pair = pres[:, :, None] & pres[:, None, :] & ~torch.eye(O, dtype=torch.bool, device=DEVICE)
        hit += torch.stack([(pred[:, i] == true[:, i])[pair].sum() for i in range(len(DIRS))]).cpu()
        n += int(pair.sum())
    return {d: (hit[i] / n).item() for i, d in enumerate(DIRS)}


def dot(p, q):
    # clamped: with an all-ones filter value, the sum of a softmax can be 1 + 1e-7, which isn't a truth value
    return (p * q).sum(-1, keepdim=True).clamp(0, 1)


def make_interp(sig, net):
    logic = Logic().with_defaults()
    logic["exists,Obj"] = AggregMax()
    logic["forall,Question"] = AggregPMeanError(p=2)
    it = Interpretation(sig, logic=logic)
    it["Feat"] = sltn.Type("Feat", shape=(net.heads[ATTRS[0]].in_features,))
    for a in ATTRS:
        n = len(VALUES[a])
        it[SORT[a]] = sltn.Type(SORT[a], shape=(n,))
        it[a] = Head(net.heads[a])
        it[IS[a]] = dot
        it[KVAR[a]] = torch.eye(n, device=DEVICE)
    it["Neq"] = 1 - torch.eye(O, device=DEVICE)
    return it


def ground(it, c, t, idx, emb, sd, relnet=None):
    """Ground the family's symbols for questions idx; emb [len(idx), O, D] are their objects' embeddings.
    Relations: CLEVR's adjacency, or with relnet, the network's output for the direction each relate names."""
    s = t["scene"][idx]
    it["x"] = emb
    it["Present"] = sd["present"][s].float()
    it["Yes"] = t["yes"][idx]
    if c.n_rels:
        rel = relnet(sd["pos"][s]) if relnet is not None else sd["rel"][s]
        b = torch.arange(len(idx), device=DEVICE)
        for k in range(c.n_rels):
            it[f"R{k}"] = rel[b, t["rel"][idx, k]]
    for a in ATTRS:
        it[ANS[a]] = t["ans"][a][idx]
        for k in range(c.n_slots):
            it[f"{PARAM[a]}{k}"] = t["slots"][a][k, idx]


def to_dev(fams):
    out = {}
    for key, (c, t) in fams.items():
        out[key] = (c, dict(scene=t["scene"].to(DEVICE), rel=t["rel"].to(DEVICE), yes=t["yes"].to(DEVICE),
                            slots={a: v.to(DEVICE) for a, v in t["slots"].items()},
                            ans={a: v.to(DEVICE) for a, v in t["ans"].items()}))
    return out


def embed_scenes(net, crops, present):
    """[B, O, ...] crops, [B, O] present -> [B, O, D] embeddings, zero for an empty position."""
    B = crops.shape[0]
    emb = torch.zeros(B, O, net.heads[ATTRS[0]].in_features, device=DEVICE)
    emb[present] = net.embed(crops[present])
    return emb


@torch.no_grad()
def concept_acc(net, sd, max_scenes=None, chunk=256):
    net.eval()
    S = len(sd["crops"]) if max_scenes is None else min(max_scenes, len(sd["crops"]))
    hit = {a: 0 for a in ATTRS}
    n = 0
    for s0 in range(0, S, chunk):
        sl = slice(s0, min(S, s0 + chunk))
        m = sd["present"][sl]
        probs = net(sd["crops"][sl][m])
        for j, a in enumerate(ATTRS):
            hit[a] += int((probs[a].argmax(-1) == sd["attr"][sl][m][:, j]).sum())
        n += int(m.sum())
    net.train()
    return {a: hit[a] / n for a in ATTRS}


@torch.no_grad()
def qa_acc(net, it, sig, fams, sd, relnet=None, chunk=256):
    net.eval()
    res = {}
    for key, (c, t) in fams.items():
        phi = sig.parse(c.body)
        N = len(t["scene"])
        ok_all = 0
        for s0 in range(0, N, chunk):
            idx = torch.arange(s0, min(N, s0 + chunk), device=DEVICE)
            s = t["scene"][idx]
            ground(it, c, t, idx, embed_scenes(net, sd["crops"][s], sd["present"][s]), sd, relnet)
            r = it(phi)
            if c.kind == "yesno":
                ok = (by_name(r, ["q"]) > 0.5) == t["yes"][idx].bool()
            else:
                ok = by_name(r, ["q", KVAR[c.kind]]).argmax(-1) == t["ans"][c.kind][idx].argmax(-1)
            ok_all += int(ok.sum())
        typ = question_type(key) + ("+relate" if c.n_rels else "")
        h, n = res.get(typ, (0, 0))
        res[typ] = (h + ok_all, n + N)
    net.train()
    total = sum(h for h, _ in res.values()) / sum(n for _, n in res.values())
    return total, {k: h / n for k, (h, n) in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--fams_per_step", type=int, default=8)
    ap.add_argument("--batch", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train_q", type=int, default=None, help="only the first N training questions")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_scenes", type=int, default=2000, help="val scenes for the periodic concept accuracy")
    ap.add_argument("--learn_relations", action="store_true", help="learn left/right/front/behind from positions")
    ap.add_argument("--only_relation_questions", action="store_true",
                    help="train only on the families with a relate step (QA is still scored on all families)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    sd = {}
    for split in ("train", "val"):
        scenes, questions = load(DATA, split)
        if split == "train" and a.n_train_q:
            questions = questions[: a.n_train_q]
        attr, present, rel = scene_arrays(scenes)
        crops = torch.from_numpy(np.load(CROPS / f"{split}_crops.npy"))
        sd[split] = dict(attr=attr.to(DEVICE), present=present.to(DEVICE), rel=rel.to(DEVICE), pos=positions(scenes).to(DEVICE),
                         crops=crops.to(DEVICE), fams=to_dev(family_tensors(questions)))
        n_q = sum(len(t["scene"]) for _, t in sd[split]["fams"].values())
        print(f"{split}: {len(scenes)} scenes, {n_q} in-scope questions ({time.perf_counter() - t0:.0f}s)", flush=True)

    sig = make_signature()
    net = ConceptNet().to(DEVICE)
    it = make_interp(sig, net)
    relnet = RelationNet().to(DEVICE) if a.learn_relations else None
    params = list(net.parameters()) + (list(relnet.parameters()) if relnet is not None else [])
    opt = torch.optim.Adam(params, lr=a.lr)
    tr, va = sd["train"], sd["val"]
    keys = [k for k, (c, _) in tr["fams"].items() if c.n_rels > 0 or not a.only_relation_questions]
    rules = {k: sig.parse(tr["fams"][k][0].rule) for k in keys}
    sizes = torch.tensor([len(tr["fams"][k][1]["scene"]) for k in keys], dtype=torch.float)
    config = dict(vars(a), n_train_inscope=int(sizes.sum()), crop=48, crop_k=500, device=str(DEVICE))
    out = RESULTS / "e1" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"sltn {sltn_version()['commit'][:7]} | {config}", flush=True)

    run_loss, t1 = 0.0, time.perf_counter()
    for step in range(1, a.steps + 1):
        fam_idx = torch.multinomial(sizes, a.fams_per_step, replacement=False, generator=g).tolist()
        picks = []
        for fi in fam_idx:
            c, t = tr["fams"][keys[fi]]
            picks.append((fi, torch.randint(0, len(t["scene"]), (a.batch,), generator=g).to(DEVICE)))
        scene_ids = torch.cat([tr["fams"][keys[fi]][1]["scene"][idx] for fi, idx in picks])
        emb = embed_scenes(net, tr["crops"][scene_ids], tr["present"][scene_ids])   # every object, once
        sats = []
        for j, (fi, idx) in enumerate(picks):
            c, t = tr["fams"][keys[fi]]
            ground(it, c, t, idx, emb[j * a.batch:(j + 1) * a.batch], tr, relnet)
            sats.append(it(rules[keys[fi]]).value.squeeze())
        loss = 1 - torch.stack(sats).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        run_loss += loss.item()
        if step % a.eval_every == 0:
            acc = concept_acc(net, va, a.eval_scenes)
            racc = relation_acc(relnet, va, a.eval_scenes) if relnet is not None else {}
            ms = (time.perf_counter() - t1) / step * 1000
            print(f"step {step:5d} loss {run_loss / a.eval_every:.4f} | concept acc (val, {a.eval_scenes} scenes): "
                  + ", ".join(f"{k} {v:.3f}" for k, v in acc.items())
                  + "".join(f", {k} {v:.3f}" for k, v in racc.items()) + f" | {ms:.0f} ms/step", flush=True)
            run_loss = 0.0

    acc = concept_acc(net, va)
    qa, per = qa_acc(net, it, sig, va["fams"], va, relnet)
    racc = relation_acc(relnet, va) if relnet is not None else {}
    summary = {**{f"concept_acc/{k}": v for k, v in acc.items()}, **{f"relation_acc/{k}": v for k, v in racc.items()},
               "qa_acc": qa, **{f"qa_acc/{k}": v for k, v in per.items()}}
    print("\nfinal concept acc (all val objects): " + ", ".join(f"{k} {v:.4f}" for k, v in acc.items()))
    if racc:
        print("final relation acc (all val pairs): " + ", ".join(f"{k} {v:.4f}" for k, v in racc.items()))
    print(f"final QA acc (val, in scope): {qa:.4f} | " + ", ".join(f"{k} {v:.4f}" for k, v in per.items()))
    (out / "results.json").write_text(json.dumps(dict(config=config, sltn=sltn_version(), **summary), indent=1))
    torch.save(net.state_dict(), out / "concept_net.pt")
    if relnet is not None:
        torch.save(relnet.state_dict(), out / "relation_net.pt")
    print(f"saved {out}; total {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
