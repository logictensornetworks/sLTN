import torch
from .base import BinaryConnectiveOperator
from .utils import DEFAULT_EPS,DEFAULT_BETA, pi_0, pi_1, smooth_clamp


class AndMin(BinaryConnectiveOperator):
    """Gödel fuzzy conjunction operator (minimum T-norm)."""

    def __repr__(self) -> str:
        return "AndMin()"

    def __call__(self, x, y) -> torch.Tensor:
        return torch.minimum(x, y)


class AndProd(BinaryConnectiveOperator):
    """
    Goguen fuzzy conjunction operator (product operator).
    """
    def __init__(self, stable=False, eps=DEFAULT_EPS):
        self.stable = stable
        self.eps = eps

    def __repr__(self) -> str:
        return f"AndProd(stable={self.stable}, eps={self.eps})"

    def __call__(self, x, y, stable=None) -> torch.Tensor:

        stable = self.stable if stable is None else stable

        if stable:
            x, y = pi_0(x, self.eps), pi_0(y, self.eps)

        return torch.mul(x, y)


class AndPMean(BinaryConnectiveOperator):
    """
    Power mean fuzzy conjunction operator.
    """
    def __init__(self, stable=False, p=1., eps=DEFAULT_EPS):
        """
        :param stable: if True, clamps inputs away from 0 for gradient stability.
        :param p: exponent for the power mean (p=1 gives arithmetic mean).
        """
        self.stable = stable
        self.p = p
        self.eps = eps

    def __repr__(self) -> str:
        return f"AndPMean(stable={self.stable}, p={self.p}, eps={self.eps})"

    def __call__(self, x, y, p=None, stable=None) -> torch.Tensor:

        stable = self.stable if stable is None else stable
        p = self.p if p is None else p

        if stable:
            x, y = pi_0(x, self.eps), pi_0(y, self.eps)
        return torch.pow((x**p + y**p) / 2., 1./p)


class AndCos(BinaryConnectiveOperator):
    """
    Tcos T-norm from B. Moser (https://doi.org/10.1016/j.fss.2006.01.007).
    """
    def __init__(self, stable=False, eps=DEFAULT_EPS):

        self.stable = stable
        self.eps = eps

    def __repr__(self) -> str:
        return f"AndCos(stable={self.stable}, eps={self.eps})"

    def __call__(self, x, y, stable=None) -> torch.Tensor:

        stable = self.stable if stable is None else stable
        if stable:
            x, y = pi_1(x, self.eps), pi_1(y, self.eps)
        return torch.clamp(torch.mul(x, y) - torch.sqrt(1-x**2) * torch.sqrt(1-y**2), min=0.)


class AndLuk(BinaryConnectiveOperator):
    """
    Lukasiewicz fuzzy conjunction operator.
    """

    def __init__(self, stable=False, eps=DEFAULT_EPS, beta=DEFAULT_BETA, smoothing_method="squareplus"):
        
        self.stable = stable
        self.eps = eps

        if self.stable:
            self.smoothing_method = smoothing_method
            self.beta = beta
        else:
            self.smoothing_method = None
            self.beta = None

    def __repr__(self) -> str:
        return f"AndLuk(stable={self.stable}, beta={self.beta}, smoothing_method={self.smoothing_method})"

    def __call__(self, x, y, stable=None) -> torch.Tensor:
        stable = self.stable if stable is None else stable

        if stable:
            beta = self.beta if self.beta is not None else DEFAULT_BETA
            smoothing_method = self.smoothing_method or "squareplus"
            x, y = pi_0(x, self.eps), pi_0(y, self.eps)
            res = smooth_clamp(x + y - 1., min=0., beta=beta, smoothing_method=smoothing_method)
            return torch.clamp(res, max=1.)
        else:
            return torch.clamp(x + y - 1., min=0.)

