"""The ground-truth interpretation, and the check that the formulas are satisfied by the training data.

Ground truth: scene[o]'s appearance is the object's one-hot attributes from the scene file, C, Z, M, S and
SameC & co. read them (dot products), R is CLEVR's rule on the 3D coordinates (networks.Relation with
learn=False). If the formulas say what the questions mean, every question's rule is true (1) under it.

Run: python -m clevr_qa.ground_truth [--split train] [--limit N] [--counting]    (from notebooks/examples/CLEVR)
"""

import argparse
import time

import torch

from .batching import family_tensors, question_type
from .counting import ground_counts
from .data import camera_directions, load, scene_arrays
from .formulas import make_signature
from .interpretation import base_interpretation, ground, per_question
from .networks import Relation
from .vocabulary import ATTRS, DIRS, N_SLOTS, PRED, SAME, VALUES

OFFSET, _o = {}, 0                                 # where each attribute's one-hot is in the appearance
for _a in ATTRS:
    OFFSET[_a] = slice(_o, _o + len(VALUES[_a]))
    _o += len(VALUES[_a])
APPEARANCE_DIM = _o                                # 15


def appearance(scenes):
    """[S, N_SLOTS, 15]: the one-hot attributes of every object, zero for an empty object slot."""
    arr = scene_arrays(scenes)["attr"]
    out = torch.zeros(len(scenes), N_SLOTS, APPEARANCE_DIM)
    for j, a in enumerate(ATTRS):
        m = arr[..., j] >= 0
        out[..., OFFSET[a]][m] = torch.nn.functional.one_hot(arr[..., j][m], len(VALUES[a])).float()
    return out


def ground_truth_interpretation(sig, directions):
    it = base_interpretation(sig, APPEARANCE_DIM)
    it["R"] = Relation(directions, learn=False)
    for a in ATTRS:
        s = slice(4 + OFFSET[a].start, 4 + OFFSET[a].stop)
        it[PRED[a]] = (lambda s: lambda x, v: (x[..., s] * v).sum(-1, keepdim=True))(s)
        it[SAME[a]] = (lambda s: lambda x, x1: (x[..., s] * x1[..., s]).sum(-1, keepdim=True))(s)
    return it


def check_relations(scenes, directions):
    """Fraction of ordered pairs of distinct objects where CLEVR's rule agrees with the scene file's
    relationships, per direction (1.0 = the rule is exact)."""
    arr = scene_arrays(scenes)
    xyz, pres, true = arr["xyz"], arr["present"], arr["rel"].bool()
    rule = Relation(directions, learn=False)
    pred = rule.probs(xyz[:, :, None, :].expand(-1, -1, N_SLOTS, -1), xyz[:, None, :, :].expand(-1, N_SLOTS, -1, -1)) > 0.5
    pair = pres[:, :, None] & pres[:, None, :] & ~torch.eye(N_SLOTS, dtype=torch.bool)
    return {d: (pred[..., i] == true[:, i])[pair].float().mean().item() for i, d in enumerate(DIRS)}


def check_formulas(scenes, questions, tol=1e-4, chunk=4096, counting=False):
    """Each in-scope question's rule under the ground truth. Returns {question type: (questions, how many
    have truth value < 1 - tol, the minimum truth value)}, and the families' aggregated truth values."""
    sig = make_signature()
    it = ground_truth_interpretation(sig, camera_directions(scenes))
    arr = scene_arrays(scenes)
    arr["pos"] = arr["xyz"]                        # CLEVR's rule reads the 3D coordinates
    app = appearance(scenes)
    res, agg = {}, {}
    for key, (c, t) in family_tensors(questions, counting=counting).items():
        rule = sig.parse(c.rule)
        counts = [sig.parse(x.formula) for x in c.counts]
        N, vals = len(t["scene_idx"]), []
        for s0 in range(0, N, chunk):
            idx = torch.arange(s0, min(N, s0 + chunk))
            ground(it, c, t, idx, app[t["scene_idx"][idx]], arr)
            ground_counts(it, c, counts)
            vals.append(per_question(it, rule))
        v = torch.cat(vals)
        agg[key] = it(rule).value.item() if N <= chunk and not counts else None
        typ = question_type(key, c)
        n, bad, lo = res.get(typ, (0, 0, 1.0))
        res[typ] = (n + N, bad + int((v < 1 - tol).sum()), min(lo, v.min().item()))
    return res, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None, help="only the first N scenes")
    ap.add_argument("--counting", action="store_true", help="also the counting questions")
    a = ap.parse_args()
    t0 = time.perf_counter()
    scenes, questions = load(a.split)
    if a.limit:
        scenes = scenes[:a.limit]
        questions = [q for q in questions if q["image_index"] < a.limit]
    print(f"{a.split}: {len(scenes)} scenes, {len(questions)} questions ({time.perf_counter() - t0:.0f}s)")
    rel = check_relations(scenes, camera_directions(scenes))
    print("R (CLEVR's rule) vs the scene files' relationships: " + ", ".join(f"{d} {v:.6f}" for d, v in rel.items()))
    res, _ = check_formulas(scenes, questions, counting=a.counting)
    n_all = sum(n for n, _, _ in res.values())
    print(f"rules under the ground truth, {n_all} in-scope questions:")
    for typ, (n, bad, lo) in sorted(res.items()):
        print(f"  {typ:15s} {n:7d} questions, {bad} not satisfied, min truth value {lo:.6f}")
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
