"""Counts: the term `count o: phi(o)` of the counting questions (formulas.py), computed outside sLTN.

For one question, sLTN gives the set's truth value on each slot, p_k = phi(o_k) (Count.formula). Read as
independent probabilities, as the product t-norm does, the truth of "exactly n slots are in the set" is the
probability that exactly n of the K events happen. Going through the slots one at a time, each slot is in the
set (probability p_k: the count so far goes up by one) or not (1 - p_k: it stays):

    P_0 = [1, 0, 0, ...]        P_k[n] = P_{k-1}[n] * (1 - p_k) + P_{k-1}[n - 1] * p_k        count = P_K

e.g. p = [0.9, 0.8, 0.1] gives count = [0.018, 0.236, 0.674, 0.072] over n = 0, 1, 2, 3. The entries add up to 1,
and each is a truth value in [0, 1]. "At least one" is 1 - count[0] = 1 - prod(1 - p_k), the probabilistic sum.

The two predicates on counts (vocabulary.py), both in [0, 1]:
    EqCount(a, b)  = sum_n a[n] b[n]                 b one-hot (an answer n): a[n]; two counts: they are equal
    MoreThan(a, b) = sum_n a[n] (b[0] + ... + b[n-1])  the first count is larger
"""

import torch

import sltn

from .vocabulary import COUNT_VARS, MAX_COUNT

SIZE = MAX_COUNT + 1


def count(p):
    """p [N, K]: the set's truth value per question and slot -> [N, SIZE], the truth of "exactly n" for each n."""
    P = torch.zeros(p.shape[0], SIZE, dtype=p.dtype, device=p.device)
    P[:, 0] = 1
    for k in range(p.shape[1]):
        pk = p[:, k:k + 1]
        P = P * (1 - pk) + torch.nn.functional.pad(P[:, :-1], (1, 0)) * pk     # the second term: count n - 1, shifted
    return P


def eq_count(a, b):
    return (a * b).sum(-1, keepdim=True).clamp(0, 1)                # clamp: float rounding can pass 1


def more_than(a, b):
    below = torch.cumsum(b, -1) - b                                  # below[n] = b[0] + ... + b[n-1]
    return (a * below).sum(-1, keepdim=True).clamp(0, 1)


def add_counting(it, device="cpu"):
    """The symbols of the counting questions, in Interpretation it."""
    it["Num"] = sltn.Type("Num", shape=(SIZE,))
    it["kn"] = torch.eye(SIZE, device=device)
    it["EqCount"] = eq_count
    it["MoreThan"] = more_than


class _Values:
    """A Forall aggregator that keeps what it aggregates, with its gradient: the set's value per question and
    slot (Count.formula has one ordinary forall, over the questions; the slot variable is free)."""

    def __init__(self, inner):
        self.inner, self.values = inner, None

    def __call__(self, xs, dim=None, keepdim=False, mask=None, **kw):
        assert self.values is None, "more than one ordinary forall in a count's formula"
        self.values = xs.movedim(dim[0], 0)
        return self.inner(xs, dim=dim, keepdim=keepdim, mask=mask, **kw)


def ground_counts(it, c, formulas):
    """Ground cnt0 (cnt1) of counting family c, whose question variables are already grounded. formulas: the
    parsed Count.formula of each count."""
    for name, phi in zip(COUNT_VARS, formulas):
        inner = it.logic.Forall
        rec = it.logic.Forall = _Values(inner)
        try:
            out = it(phi)
        finally:
            it.logic.Forall = inner
        p = rec.values.squeeze(-1)                                   # [N, K]: question, slot
        assert p.dim() == 2 and out.value.shape[0] == p.shape[1], (p.shape, out)
        it[name] = count(p)
