"""Scores on held-out scenes: concept accuracy (attributes), relation accuracy, and QA accuracy.

The scene files' attributes and relationships are used here only, never in training.
`split` is the dict train.load_split builds: data.scene_arrays' tensors, `crops`, and `fams`.
"""

import torch

from .batching import question_type
from .interpretation import embed_scenes, ground, per_question
from .vocabulary import ANS, ATTRS, DIRS, N_SLOTS


@torch.no_grad()
def concept_acc(net, split, max_scenes=None, chunk=256):
    """Accuracy of each attribute head's argmax over the real objects of the first max_scenes scenes."""
    net.eval()
    S = len(split["crops"]) if max_scenes is None else min(max_scenes, len(split["crops"]))
    hit, n = {a: 0 for a in ATTRS}, 0
    for s0 in range(0, S, chunk):
        sl = slice(s0, min(S, s0 + chunk))
        m = split["present"][sl]
        probs = net(split["crops"][sl][m])
        for j, a in enumerate(ATTRS):
            hit[a] += int((probs[a].argmax(-1) == split["attr"][sl][m][:, j]).sum())
        n += int(m.sum())
    net.train()
    return {a: hit[a] / n for a in ATTRS}


@torch.no_grad()
def relation_acc(relation, split, max_scenes=None):
    """Accuracy per direction over ordered pairs of distinct real objects, against CLEVR's relations."""
    S = len(split["present"]) if max_scenes is None else min(max_scenes, len(split["present"]))
    pos, pres, true = split[relation.reads][:S], split["present"][:S], split["rel"][:S].bool()   # true [S, 4, N_SLOTS, N_SLOTS]
    pred = relation.probs(pos[:, :, None, :].expand(-1, -1, N_SLOTS, -1), pos[:, None, :, :].expand(-1, N_SLOTS, -1, -1)) > 0.5
    pair = pres[:, :, None] & pres[:, None, :] & ~torch.eye(N_SLOTS, dtype=torch.bool, device=pres.device)
    return {d: (pred[..., i] == true[:, i])[pair].float().mean().item() for i, d in enumerate(DIRS)}


@torch.no_grad()
def qa_acc(net, it, sig, split, chunk=256):
    """Each question's reply (the `predict` formula, read per question) against its answer, by question type.
    A yes/no reply is its truth value > 0.5; a query reply is the value with the highest truth value."""
    net.eval()
    res = {}
    for key, (c, t) in split["fams"].items():
        phi = sig.parse(c.predict)
        N, ok_all = len(t["scene_idx"]), 0
        for s0 in range(0, N, chunk):
            idx = torch.arange(s0, min(N, s0 + chunk), device=t["scene_idx"].device)
            s = t["scene_idx"][idx]
            ground(it, c, t, idx, embed_scenes(net, split["crops"][s], split["present"][s]), split)
            v = per_question(it, phi)
            if c.kind == "yesno":
                ok = (v > 0.5) == (t["yes"][idx].squeeze(-1) > 0.5)
            else:
                ok = v.argmax(-1) == t[ANS[c.kind]][idx].argmax(-1)
            ok_all += int(ok.sum())
        typ = question_type(key, c)
        h, n = res.get(typ, (0, 0))
        res[typ] = (h + ok_all, n + N)
    net.train()
    total = sum(h for h, _ in res.values()) / sum(n for _, n in res.values())
    return total, {k: h / n for k, (h, n) in res.items()}
