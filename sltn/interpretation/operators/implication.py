from __future__ import annotations

import torch

from .base import BinaryConnectiveOperator, UnaryConnectiveOperator
from .conjunction import AndLuk, AndMin, AndProd
from .negation import NotStandard
from .utils import DEFAULT_EPS, DEFAULT_STABLE, pi_0


class ImpliesDual(BinaryConnectiveOperator):
    """
    Given a conjunction and a negation, return the implication A -> B = not (A and not B)
    """
    def __init__(self, and_op: BinaryConnectiveOperator, neg_op: UnaryConnectiveOperator):
        
        self.and_op = and_op
        self.neg_op = neg_op

    def __repr__(self) -> str:
        return f"ImpliesDual(and_op={self.and_op}, neg_op={self.neg_op})"

    def __call__(self, x, y) -> torch.Tensor:

        return self.neg_op(self.and_op(x, self.neg_op(y)))


class ImpliesKleeneDienes(ImpliesDual):
    """
    Kleene Dienes fuzzy implication operator: A -> B = torch.maximum(1. - A, B)
    """
    def __init__(self):
        super().__init__(and_op=AndMin(), neg_op=NotStandard())

    def __repr__(self) -> str:
        return "ImpliesKleeneDienes()"


class ImpliesReichenbach(ImpliesDual):
    """
    Reichenbach fuzzy implication operator: A -> B = 1 - A + A*B
    """
    def __init__(self, stable=DEFAULT_STABLE, eps=DEFAULT_EPS):
        super().__init__(and_op=AndProd(stable=stable, eps=eps), neg_op=NotStandard())
        self.stable = stable
        self.eps = eps

    def __repr__(self) -> str:
        return f"ImpliesReichenbach(stable={self.stable}, eps={self.eps})"


class ImpliesLuk(ImpliesDual):
    """
    Lukasiewicz fuzzy implication operator.
    """
    def __init__(self, stable=DEFAULT_STABLE, eps=DEFAULT_EPS):
        super().__init__(and_op=AndLuk(stable=stable, eps=eps), neg_op=NotStandard())
        self.stable = stable
        self.eps = eps

    def __repr__(self) -> str:
        return f"ImpliesLuk(stable={self.stable}, eps={self.eps})"


class ImpliesGodel(BinaryConnectiveOperator):
    """
    Godel fuzzy implication operator: 1 if x <= y, else y.
    """
    def __repr__(self) -> str:
        return "ImpliesGodel()"

    def __call__(self, x, y) -> torch.Tensor:
        return torch.where(torch.le(x, y), torch.ones_like(x), y)


class ImpliesGoguen(BinaryConnectiveOperator):
    """
    Goguen fuzzy implication operator: 1 if x <= y, else y / x.
    """
    def __init__(self, stable=DEFAULT_STABLE, eps=DEFAULT_EPS):
        self.stable = stable
        self.eps = eps

    def __repr__(self) -> str:
        return f"ImpliesGoguen(stable={self.stable}, eps={self.eps})"

    def __call__(self, x, y, stable=None) -> torch.Tensor:
        stable = self.stable if stable is None else stable
        x_raw = x
        if stable:
            x = pi_0(x, self.eps)
        # The clamp keeps the unselected branch finite, so x = 0 gives no NaN value or gradient.
        return torch.where(torch.le(x_raw, y), torch.ones_like(y), y / x.clamp_min(self.eps))


# Equivalence operators
class Equiv(BinaryConnectiveOperator):
    """
    Equivalence (<->) fuzzy operator.
    """
    def __init__(self, and_op, implies_op):

        self.and_op = and_op
        self.implies_op = implies_op

    def __repr__(self) -> str:
        return f"Equiv(and_op={self.and_op}, implies_op={self.implies_op})"

    def __call__(self, x, y) -> torch.Tensor:
        op1 = self.implies_op(x, y)
        op2 = self.implies_op(y, x)
        return self.and_op(op1, op2)


class EquivSimilarity(BinaryConnectiveOperator):
    """
    Equivalence based on absolute difference: 1 - |x - y|.
    This operator has the benefit of non-vanishing gradients, which can
    be helpful during training.
    """
    def __init__(self, p=1, stable=DEFAULT_STABLE, eps=DEFAULT_EPS):
        self.p = p
        self.stable = stable
        self.eps = eps

    def __repr__(self) -> str:
        return f"EquivSimilarity(p={self.p}, stable={self.stable}, eps={self.eps})"

    def __call__(self, x, y, stable=None) -> torch.Tensor:
        stable = self.stable if stable is None else stable
        diff = torch.abs(x - y)
        if stable:
            diff = pi_0(diff, self.eps)
        return 1.0 - torch.pow(diff, self.p)
