import torch
from .base import AggregationOperator
from .utils import DEFAULT_EPS, pi_0, pi_1


class _AggregExtremum(AggregationOperator):
    """Base for min/max fuzzy aggregation over the top-k or bottom-k values.

    Computes the mean of the *k* extreme values along each requested dimension.
    When *k* is 1 (default) this is the true min or max.

    Parameters
    ----------
    k : int
        Number of extreme values to average.  Default: 1.
    largest : bool
        ``True`` for max (top-k), ``False`` for min (bottom-k).
    _mask_fill : float
        Value used to neutralise masked-out positions (1.0 for min, 0.0 for max).
    """

    def __init__(self, k: int = 1, *, largest: bool, _mask_fill: float) -> None:
        self.k          = k
        self._largest   = largest
        self._mask_fill = _mask_fill

    def __call__(self, xs, dim=None, keepdim=False, mask=None, k=None):
        k = self.k if k is None else k

        if mask is not None:
            w  = mask.float() if mask.dtype == torch.bool else mask
            xs = torch.where(w > 0, xs, torch.full_like(xs, self._mask_fill))

        dims   = [dim] if isinstance(dim, int) else (dim if dim is not None else [])
        result = xs
        for d in dims:
            topk_res, _ = torch.topk(result, k=k, dim=d, largest=self._largest, sorted=False)
            result      = torch.mean(topk_res, dim=d, keepdim=True)

        if not keepdim:
            result = torch.squeeze(result, dim=dims)
        return result


class AggregMin(_AggregExtremum):
    """Min fuzzy aggregation: mean of the bottom-k values (default k=1 is true min).

    Parameters
    ----------
    bottom_k : int
        Number of smallest values to average.  Default: 1 (true min).
    """

    def __init__(self, bottom_k: int = 1) -> None:
        super().__init__(k=bottom_k, largest=False, _mask_fill=1.0)
        self.bottom_k = bottom_k

    def __repr__(self) -> str:
        return f"AggregMin(bottom_k={self.bottom_k})"

    def __call__(self, xs, dim=None, keepdim=False, mask=None, bottom_k=None):
        return super().__call__(xs, dim=dim, keepdim=keepdim, mask=mask, k=bottom_k)


class AggregMax(_AggregExtremum):
    """Max fuzzy aggregation: mean of the top-k values (default k=1 is true max).

    Parameters
    ----------
    top_k : int
        Number of largest values to average.  Default: 1 (true max).
    """

    def __init__(self, top_k: int = 1) -> None:
        super().__init__(k=top_k, largest=True, _mask_fill=0.0)
        self.top_k = top_k

    def __repr__(self) -> str:
        return f"AggregMax(top_k={self.top_k})"

    def __call__(self, xs, dim=None, keepdim=False, mask=None, top_k=None):
        return super().__call__(xs, dim=dim, keepdim=keepdim, mask=mask, k=top_k)

class AggregGeometricMean(AggregationOperator):
    """Geometric mean fuzzy aggregation operator.

    Parameters
    ----------
    stable : bool
        If ``True``, clamps inputs away from 0 before taking the log to
        prevent gradient blow-up at the boundary.
    eps : float
        Clamping epsilon used when *stable* is ``True``.

    Notes
    -----
    Accepts an optional ``mask``,
    (:class:`AggregPMean`, :class:`AggregPMeanError`): a hard boolean mask
    excludes entries from the aggregation domain, while a soft mask in
    ``[0, 1]`` is used as a per-element weight, turning the operator into a
    *weighted* geometric mean,
    ``exp(sum(w_i * log(x_i)) / sum(w_i))``.
    """

    def __init__(self, stable=False, eps=DEFAULT_EPS):
        self.stable = stable
        self.eps = eps

    def __repr__(self):
        return "AggregGeometricMean(stable=" + str(self.stable) + ")"

    def __call__(self, xs, dim=None, mask=None, keepdim=False, stable=None):
        stable = self.stable if stable is None else stable

        if stable:
            xs = pi_0(xs, self.eps)

        if mask is not None:
            # Accept both bool masks (hard guard) and float tensors in [0,1];
            # a soft mask turns this into a weighted geometric mean.
            # Zero-weight positions are replaced by the product's neutral
            # element (1.0, i.e. log 0) so they never contribute, regardless
            # of their raw truth value, and so log(0) never triggers a NaN
            # via 0 * -inf.
            w = mask.float() if mask.dtype == torch.bool else mask
            safe_xs     = torch.where(w > 0, xs, torch.ones_like(xs))
            log_xs      = torch.log(safe_xs)
            numerator   = torch.sum(w * log_xs, dim=dim, keepdim=keepdim)
            denominator = torch.sum(w,          dim=dim, keepdim=keepdim)
            mean_log    = torch.div(numerator, denominator + 1e-6)
        else:
            mean_log = torch.mean(torch.log(xs), dim=dim, keepdim=keepdim)

        return torch.exp(mean_log)

class AggregPMean(AggregationOperator):
    """
    pMean fuzzy aggregation operator.
    """
    def __init__(self, p=2, stable=False, eps=DEFAULT_EPS):

        self.p = p
        self.stable = stable
        self.eps = eps

    def __repr__(self):
        return "AggregPMean(p=" + str(self.p) + ", stable=" + str(self.stable) + ")"

    def __call__(self, xs, dim=None, keepdim=False, mask=None, p=None, stable=None):

        p = self.p if p is None else p
        stable = self.stable if stable is None else stable

        if stable:
            # Protects inner pow for p < 1
            xs = pi_0(xs, self.eps)

        xs_pow = torch.pow(xs, p)

        if mask is not None:
            # Accept both bool masks (hard guard) and float tensors in [0,1]
            w = mask.float() if mask.dtype == torch.bool else mask
            numerator   = torch.sum(w * xs_pow, dim=dim, keepdim=keepdim)
            denominator = torch.sum(w,          dim=dim, keepdim=keepdim)
            mean_val    = torch.div(numerator, denominator + 1e-6)
            if stable:
                mean_val = pi_0(mean_val, self.eps)
            return torch.pow(mean_val, 1 / p)
        else:
            mean_val = torch.mean(xs_pow, dim=dim, keepdim=keepdim)
            if stable:
                mean_val = pi_0(mean_val, self.eps)
            return torch.pow(mean_val, 1 / p)

class AggregPMeanError(AggregationOperator):
    """pMeanError fuzzy aggregation operator.

    Defined as ``1 - pMean(1 - xs)``.
    """
    def __init__(self, p=2, stable=False, eps=DEFAULT_EPS):
        self.p = p
        self.stable = stable
        self.eps = eps
        self._pmean_op = AggregPMean(p=p, stable=stable, eps=eps)

    def __repr__(self):
        return "AggregPMeanError(p=" + str(self.p) + ", stable=" + str(self.stable) + ")"

    def __call__(self, xs, dim=None, keepdim=False, mask=None, p=None, stable=None):
        p_call      = self.p      if p      is None else p
        stable_call = self.stable if stable is None else stable
        return 1. - self._pmean_op(1. - xs, dim=dim, keepdim=keepdim, mask=mask, p=p_call, stable=stable_call)
