"""Learn the spatial relations from the answers only (CPU). Attributes: perfect sight (as in E0).

A relation network reads the positions of two objects (pixel x, y and depth from the scene file) and gives
four probabilities: is the second object left of / right of / in front of / behind the first. Each relate
of a question grounds its relation Rk with the network's output for the direction it names, instead of
CLEVR's adjacency. The formulas are E0's. Scored on val pairs against CLEVR's adjacency.

Run: conda run -n sltn python e1_relations_perfect_sight.py [--steps 1500] [--n_train_q 30000]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

import sltn
from sltn import Interpretation, Logic
from sltn.interpretation.operators.aggregator import AggregMax, AggregPMeanError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clevr_sltn import (ANS, ATTRS, DIRS, FEAT, IS, KVAR, OFFSET, PARAM, SORT, VALUES, O, by_name,  # noqa: E402
                        DATA, RESULTS, family_tensors, load, make_signature, question_type, scene_arrays)



def positions(scenes):
    """[S, O, 3]: pixel x / 480, pixel y / 320, depth / 15 (zeros for empty positions)."""
    pos = torch.zeros(len(scenes), O, 3)
    for s, sc in enumerate(scenes):
        for i, ob in enumerate(sc["objects"]):
            x, y, d = ob["pixel_coords"]
            pos[s, i] = torch.tensor([x / 480, y / 320, d / 15])
    return pos


def one_hot_attrs(attr, present):
    """Perfect sight: [S, O, FEAT] one-hot attributes."""
    x = torch.zeros(*attr.shape[:2], FEAT)
    for j, a in enumerate(ATTRS):
        lo, hi = OFFSET[a]
        x[..., lo:hi] = nn.functional.one_hot(attr[..., j].clamp(min=0), hi - lo).float()
    return x * present[..., None]


class RelationNet(nn.Module):
    """positions [B, O, 3] -> [B, 4 directions, O, O]: R[b, d, u, o] = P(o is d of u)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(9, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, len(DIRS)))

    def forward(self, pos):
        B = pos.shape[0]
        u = pos[:, :, None, :].expand(B, O, O, 3)
        o = pos[:, None, :, :].expand(B, O, O, 3)
        logits = self.mlp(torch.cat([u, o, o - u], -1))              # [B, O(u), O(o), 4]
        return torch.sigmoid(logits).permute(0, 3, 1, 2)


def perfect_sight(a):
    lo, hi = OFFSET[a]
    return lambda x: x[..., lo:hi]


def dot(p, q):
    return (p * q).sum(-1, keepdim=True).clamp(0, 1)


def make_interp(sig):
    logic = Logic().with_defaults()
    logic["exists,Obj"] = AggregMax()
    logic["forall,Question"] = AggregPMeanError(p=2)
    it = Interpretation(sig, logic=logic)
    it["Feat"] = sltn.Type("Feat", shape=(FEAT,))
    for a in ATTRS:
        n = len(VALUES[a])
        it[SORT[a]] = sltn.Type(SORT[a], shape=(n,))
        it[a] = perfect_sight(a)
        it[IS[a]] = dot
        it[KVAR[a]] = torch.eye(n)
    it["Neq"] = 1 - torch.eye(O)
    return it


def ground(it, c, t, idx, sd, relnet):
    s = t["scene"][idx]
    it["x"], it["Present"], it["Yes"] = sd["x"][s], sd["present"][s].float(), t["yes"][idx]
    if c.n_rels:
        rel = relnet(sd["pos"][s])                                    # [B, 4, O, O], learned
        b = torch.arange(len(idx))
        for k in range(c.n_rels):
            it[f"R{k}"] = rel[b, t["rel"][idx, k]]
    for a in ATTRS:
        it[ANS[a]] = t["ans"][a][idx]
        for k in range(c.n_slots):
            it[f"{PARAM[a]}{k}"] = t["slots"][a][k, idx]


@torch.no_grad()
def relation_acc(relnet, sd, max_scenes=3000):
    """Accuracy per direction over ordered pairs of distinct real objects, against CLEVR's adjacency."""
    S = min(max_scenes, len(sd["pos"]))
    pred = relnet(sd["pos"][:S]) > 0.5
    true = sd["rel"][:S].bool()
    pres = sd["present"][:S]
    pair = pres[:, :, None] & pres[:, None, :] & ~torch.eye(O, dtype=torch.bool)
    return {d: (pred[:, i] == true[:, i])[pair].float().mean().item() for i, d in enumerate(DIRS)}


@torch.no_grad()
def qa_acc(it, sig, fams, sd, relnet, chunk=4096):
    res = {}
    for key, (c, t) in fams.items():
        phi = sig.parse(c.body)
        N = len(t["scene"])
        hits = 0
        for s0 in range(0, N, chunk):
            idx = torch.arange(s0, min(N, s0 + chunk))
            ground(it, c, t, idx, sd, relnet)
            r = it(phi)
            if c.kind == "yesno":
                ok = (by_name(r, ["q"]) > 0.5) == t["yes"][idx].bool()
            else:
                ok = by_name(r, ["q", KVAR[c.kind]]).argmax(-1) == t["ans"][c.kind][idx].argmax(-1)
            hits += int(ok.sum())
        typ = question_type(key) + ("+relate" if c.n_rels else "")
        h, n = res.get(typ, (0, 0))
        res[typ] = (h + hits, n + N)
    return sum(h for h, _ in res.values()) / sum(n for _, n in res.values()), {k: h / n for k, (h, n) in sorted(res.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--fams_per_step", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_train_q", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    a = ap.parse_args()
    print(vars(a), flush=True)
    t0 = time.perf_counter()
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    sd = {}
    for split in ("train", "val"):
        scenes, questions = load(DATA, split)
        if split == "train":
            questions = questions[: a.n_train_q]
        attr, present, rel = scene_arrays(scenes)
        sd[split] = dict(x=one_hot_attrs(attr, present), present=present, rel=rel, pos=positions(scenes),
                         fams=family_tensors(questions))
    print(f"loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    tr, va = sd["train"], sd["val"]
    sig = make_signature()
    it = make_interp(sig)
    relnet = RelationNet()
    opt = torch.optim.Adam(relnet.parameters(), lr=1e-3)
    keys = [k for k, (c, _) in tr["fams"].items() if c.n_rels > 0]   # only the families that use relations
    rules = {k: sig.parse(tr["fams"][k][0].rule) for k in keys}
    sizes = torch.tensor([len(tr["fams"][k][1]["scene"]) for k in keys], dtype=torch.float)
    print(f"training on {len(keys)} families with relations, {int(sizes.sum())} questions", flush=True)
    print("step     0 relation acc (val pairs): " + ", ".join(f"{d} {v:.3f}" for d, v in relation_acc(relnet, va).items()), flush=True)
    run_loss, history = 0.0, [dict(step=0, **relation_acc(relnet, va))]
    for step in range(1, a.steps + 1):
        sats = []
        for fi in torch.multinomial(sizes, a.fams_per_step, replacement=False, generator=g).tolist():
            c, t = tr["fams"][keys[fi]]
            idx = torch.randint(0, len(t["scene"]), (a.batch,), generator=g)
            ground(it, c, t, idx, tr, relnet)
            sats.append(it(rules[keys[fi]]).value.squeeze())
        loss = 1 - torch.stack(sats).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        run_loss += loss.item()
        if step % a.log_every == 0:
            acc = relation_acc(relnet, va)
            history.append(dict(step=step, loss=run_loss / a.log_every, **acc))
            print(f"step {step:5d} loss {run_loss / a.log_every:.4f} | relation acc (val pairs): "
                  + ", ".join(f"{d} {v:.3f}" for d, v in acc.items()), flush=True)
            run_loss = 0.0
    qa, per = qa_acc(it, sig, va["fams"], va, relnet)
    print("\nfinal relation acc (val pairs): " + ", ".join(f"{d} {v:.4f}" for d, v in relation_acc(relnet, va, 15000).items()))
    print(f"final QA acc (val, in scope, perfect attributes + learned relations): {qa:.4f}")
    for k, v in per.items():
        print(f"  {k:16s} {v:.4f}")
    import json
    out = RESULTS / "e1_relations_perfect_sight"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(dict(config=vars(a), history=history,
                                                      relation_acc=relation_acc(relnet, va, 15000), qa=qa, qa_by_type=per), indent=1))
    print(f"saved {out / 'results.json'}; total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
