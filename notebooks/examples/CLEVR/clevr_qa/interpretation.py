"""Grounding the signature: the shared setup, the learned interpretation, and reading each question's value.

The learned interpretation and the ground truth (ground_truth.py) differ only in the predicates C, Z, M, S,
SameC & co., R and in what scene[o]'s appearance is; `base_interpretation` holds the rest.
"""

import torch

import sltn
from sltn import Interpretation, Logic
from sltn.interpretation.operators.aggregator import AggregMax, AggregPMeanError

from .counting import add_counting
from .networks import Has, Same
from .vocabulary import ATTRS, DIRS, KVAR, N_SLOTS, PRED, PRESENT, SAME, SORT, VALUES


def base_interpretation(sig, appearance_dim, device="cpu", n_slots=N_SLOTS):
    """Logic, types and the predicates that read scene directly (Present, Neq); scene[o] has 4 + appearance_dim
    features. Exists over objects is the max, the outer forall over questions a p-mean error (p = 2), and so
    is the forall over pixels of the reconstruction axiom. n_slots: the size of Obj (11 for the slot model)."""
    logic = Logic().with_defaults()
    logic["exists,Obj"] = AggregMax()
    logic["forall"] = AggregPMeanError(p=2)
    logic["forall,Pix"] = AggregPMeanError(p=2)
    it = Interpretation(sig, logic=logic)
    it["Obj"] = range(n_slots)
    it["Object"] = sltn.Type("Object", shape=(4 + appearance_dim,))
    it["Dir"] = sltn.Type("Dir", shape=(len(DIRS),))
    it["Neq"] = 1 - torch.eye(n_slots, device=device)
    it["Present"] = lambda x: x[..., PRESENT:PRESENT + 1]
    for a in ATTRS:
        it[SORT[a]] = sltn.Type(SORT[a], shape=(len(VALUES[a]),))
        it[KVAR[a]] = torch.eye(len(VALUES[a]), device=device)
    add_counting(it, device)
    return it


def learned_interpretation(sig, net, relation, device="cpu"):
    """C, Z, M, S and SameC & co. from net's heads (networks.Has, Same); R is `relation`."""
    it = base_interpretation(sig, net.dim, device)
    it["R"] = relation
    for a in ATTRS:
        it[PRED[a]] = Has(net.heads[a])
        it[SAME[a]] = Same(net.heads[a])
    return it


def embed_scenes(net, crops, present):
    """[B, N_SLOTS, ...] crops, [B, N_SLOTS] present -> [B, N_SLOTS, D] embeddings, zero for an empty object slot."""
    emb = torch.zeros(crops.shape[0], N_SLOTS, net.dim, device=crops.device)
    emb[present] = net.embed(crops[present])
    return emb


def ground(it, c, t, idx, appearance, scenes):
    """Ground family c's variables for its questions idx. t: batching.family_params of the family;
    appearance [len(idx), N_SLOTS, D]: their objects' appearance; scenes: data.scene_arrays' present, and
    "pos": the position R reads (pixel or xyz, see networks.Relation.reads)."""
    s = t["scene_idx"][idx]
    it["scene"] = torch.cat([scenes["present"][s].float()[..., None], scenes["pos"][s], appearance], -1)
    for name in c.variables[1:] + [c.answer]:
        it[name] = t[name][idx]


class _Recorder:
    """A Forall aggregator that also keeps what it aggregates. sLTN returns only the aggregate of a
    quantifier; to score each question we need the values inside the outer `forall`, one per question.
    The formulas have a single ordinary `forall` (the object quantifiers are structural), so the values
    recorded are exactly the per-question ones."""

    def __init__(self, inner):
        self.inner, self.values = inner, None

    def __call__(self, xs, dim=None, keepdim=False, mask=None, **kw):
        assert self.values is None, "more than one ordinary forall in the formula"
        self.values = xs.detach().movedim(dim[0], 0)              # question first
        return self.inner(xs, dim=dim, keepdim=keepdim, mask=mask, **kw)

    def __repr__(self):
        return f"_Recorder({self.inner!r})"


def per_question(it, formula):
    """Truth value of the formula's body for each question: [N] for a rule or a yes/no `predict`,
    [N, values] for a query `predict` (its answer left open, k<attr>)."""
    inner = it.logic.Forall
    rec = it.logic.Forall = _Recorder(inner)
    try:
        with torch.no_grad():
            it(formula)
    finally:
        it.logic.Forall = inner
    return rec.values.squeeze(-1)
