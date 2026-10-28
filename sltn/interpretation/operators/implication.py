import torch
from .base import BinaryConnectiveOperator, UnaryConnectiveOperator
from .negation import NotStandard
from .conjunction import AndProd, AndMin, AndLuk
from .utils import DEFAULT_EPS, DEFAULT_BETA, pi_0, smooth_clamp


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
    def __init__(self, stable=True, eps=DEFAULT_EPS):
        super().__init__(and_op=AndProd(stable=stable, eps=eps), neg_op=NotStandard())
        self.stable = stable
        self.eps = eps

    def __repr__(self) -> str:
        return f"ImpliesReichenbach(stable={self.stable}, eps={self.eps})"


class ImpliesLuk(ImpliesDual):
    """
    Lukasiewicz fuzzy implication operator.
    """
    def __init__(self, stable=True, eps=DEFAULT_EPS, beta=DEFAULT_BETA, smoothing_method="squareplus"):
        super().__init__(and_op=AndLuk(stable=stable, eps=eps, beta=beta, smoothing_method=smoothing_method), neg_op=NotStandard())
        self.stable = stable
        self.eps = eps
        self.beta = beta
        self.smoothing_method = smoothing_method

    def __repr__(self) -> str:
        return f"ImpliesLuk(stable={self.stable}, eps={self.eps}, beta={self.beta}, smoothing_method={self.smoothing_method})"


class ImpliesGodel(BinaryConnectiveOperator):
    """
    Godel fuzzy implication operand.
    
    Use smooth conditional logic to maintain non-zero gradients everywhere.
    """
    def __init__(self, smooth=False, beta=DEFAULT_BETA, smoothing_method="softplus"):
        """
        Args:
            smooth: If True, use smooth conditional (non-zero gradients everywhere).
                   If False, use hard torch.where (standard behavior, zero gradients at boundary).
            beta: Controls how steeply the smooth condition transitions (higher = closer to hard).
            smoothing_method: Method used for smoothing ("softplus" or "squareplus").
        """
        self.smooth = smooth
        self.beta = beta
        self.smoothing_method = smoothing_method

    def __repr__(self) -> str:
        return f"ImpliesGodel(smooth={self.smooth}, beta={self.beta}, smoothing_method={self.smoothing_method})"

    def __call__(self, x, y) -> torch.Tensor:
        if self.smooth:
            res = y + (1.0 - y) * smooth_clamp(0.5 + self.beta * (y - x), 
                                                max=1.0, min=0.0,
                                                beta=self.beta, 
                                                smoothing_method=self.smoothing_method)
            return torch.clamp(res, 0.0, 1.0)
        else:
            return torch.where(torch.le(x, y), torch.ones_like(x), y)


class ImpliesGoguen(BinaryConnectiveOperator):
    """
    Goguen fuzzy implication operator.
    """
    def __init__(self, stable=True, eps=DEFAULT_EPS, 
                 smooth=False, beta=10., smoothing_method="softplus"):
        """
        Args:
            stable: If True with smooth=False, uses pi_0 for x before division.
            eps: Epsilon for pi_0 boundary shift.
            smooth: If True, use smooth conditional (non-zero gradients everywhere).
                   If False, use hard torch.where (standard behavior, zero gradients at boundary).
            beta: Controls how steeply the smooth condition transitions.
        """
        self.stable = stable
        self.eps = eps
        self.smooth = smooth
        self.beta = beta
        self.smoothing_method = smoothing_method

    def __repr__(self) -> str:
        return f"ImpliesGoguen(stable={self.stable}, eps={self.eps}, smooth={self.smooth}, beta={self.beta}, smoothing_method={self.smoothing_method})"

    def __call__(self, x, y, stable=None) -> torch.Tensor:
        stable = self.stable if stable is None else stable
        x_raw = x

        # Smooth mode always needs a safe (non-zero) denominator; hard mode
        # only clamps when explicitly requested via `stable`.
        if stable or self.smooth:
            x = pi_0(x, self.eps)

        if self.smooth:
            clamped_ratio = smooth_clamp(torch.div(y, x), 
                                         beta=self.beta, 
                                         max=1.0, min=0.0,
                                         smoothing_method=self.smoothing_method)
            gate = smooth_clamp(0.5 + self.beta * (y - x),
                                max=1.0, min=0.0,
                                beta=self.beta, 
                                smoothing_method=self.smoothing_method)
            return clamped_ratio + (1.0 - clamped_ratio) * gate
        else:
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
    def __init__(self, p=1, stable=True, eps=DEFAULT_EPS):
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
