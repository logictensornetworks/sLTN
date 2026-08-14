from .base import BinaryConnectiveOperator
from .utils import DEFAULT_EPS, DEFAULT_BETA
import torch

from .conjunction import AndProd, AndMin, AndLuk
from .negation import NotStandard


class OrDual(BinaryConnectiveOperator):
    """
    Classical fuzzy disjunction operator: A ∨ B = ¬ (¬A & ¬B)
    """
    def __init__(self, and_op: BinaryConnectiveOperator, neg_op: BinaryConnectiveOperator, stable=False, eps=DEFAULT_EPS):

        self.and_op = and_op
        self.neg_op = neg_op
        self.stable = stable
        self.eps = eps

        if stable:
            if hasattr(self.and_op, 'stable'):
                self.and_op.stable = stable
                self.and_op.eps = eps 
            if hasattr(self.neg_op, 'stable'):
                self.neg_op.stable = stable
                self.neg_op.eps = eps 

    def __repr__(self) -> str:
        return f"OrDual(and_op={self.and_op}, neg_op={self.neg_op}, stable={self.stable}, eps={self.eps})"

    def __call__(self, x, y) -> torch.Tensor:
        return self.neg_op(self.and_op(self.neg_op(x), self.neg_op(y)))


class OrMax(OrDual):
    """
    Godel fuzzy disjunction operator (max operator): A ∨ B = max(A, B)
    """
    def __init__(self, stable=False, eps=DEFAULT_EPS):
        super().__init__(and_op=AndMin(), neg_op=NotStandard(), stable=stable, eps=eps)

    def __repr__(self) -> str:
        return "OrMax()"


class OrProbSum(OrDual):
    """
    Goguen fuzzy disjunction operator (probabilistic sum).
    """
    def __init__(self, stable=False, eps=DEFAULT_EPS):
        super().__init__(and_op=AndProd(), neg_op=NotStandard(), stable=stable, eps=eps)

    def __repr__(self) -> str:
        return f"OrProbSum(stable={self.stable}, eps={self.eps})"


class OrLuk(OrDual):
    """
    Lukasiewicz fuzzy disjunction operator.
    """
    def __init__(self, stable=False, eps=DEFAULT_EPS, beta=DEFAULT_BETA, smoothing_method="squareplus"):
        super().__init__(and_op=AndLuk(stable=stable, eps=eps, beta=beta, smoothing_method=smoothing_method), neg_op=NotStandard(), stable=stable, eps=eps)
        self.beta = beta
        self.smoothing_method = smoothing_method

    def __repr__(self) -> str:
        return f"OrLuk(stable={self.stable}, eps={self.eps}, beta={self.beta}, smoothing_method={self.smoothing_method})"


class XOrClassical(BinaryConnectiveOperator):
    """
    XOR fuzzy disjunction operator: A ⊕ B = ¬(¬A & ¬B) & ¬(A & B)
    """
    def __init__(self, and_op: BinaryConnectiveOperator, neg_op: BinaryConnectiveOperator, stable=False, eps=DEFAULT_EPS):

        self.and_op = and_op
        self.neg_op = neg_op
        self.stable = stable
        self.eps = eps

        if stable:
            if hasattr(self.and_op, 'stable'):
                self.and_op.stable = stable
                self.and_op.eps = eps 
            if hasattr(self.neg_op, 'stable'):
                self.neg_op.stable = stable
                self.neg_op.eps = eps 

    def __repr__(self) -> str:
        return f"XOrClassical(and_op={self.and_op}, neg_op={self.neg_op}, stable={self.stable}, eps={self.eps})"

    def __call__(self, x, y) -> torch.Tensor:
        return self.and_op(self.neg_op(self.and_op(self.neg_op(x), self.neg_op(y))), self.neg_op(self.and_op(x, y)))
