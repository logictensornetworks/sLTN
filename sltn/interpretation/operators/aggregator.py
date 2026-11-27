from __future__ import annotations

import torch

from .base import AggregationOperator
from .utils import DEFAULT_EPS, DEFAULT_STABLE, pi_0, safe_div


def _mask_weights(xs, mask):
    """Return *xs* and *mask* broadcast to one shape, the mask as float weights.

    Broadcasting lets the weighted means count every entry a partial mask spans.
    """
    return torch.broadcast_tensors(xs, mask.to(xs.dtype))


class _AggregExtremum(AggregationOperator):
    """Base for min/max fuzzy aggregation over the top-k or bottom-k values.

    Computes the mean of the *k* extreme values along each requested dimension.
    When *k* is 1 (default) this is the true min or max.

    A mask only decides whether an entry takes part (weight above 0): soft
    weights are not used, and the mask receives no gradient.

    Parameters
    ----------
    k : int
        Number of extreme values to average, at least 1.  Default: 1.
    largest : bool
        ``True`` for max (top-k), ``False`` for min (bottom-k).
    """

    def __init__(self, k: int = 1, *, largest: bool) -> None:
        self.k        = self._check_k(k)
        self._largest = largest

    @staticmethod
    def _check_k(k: int) -> int:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}.")
        return k

    def __call__(self, xs, dim=None, keepdim=False, mask=None, k=None):
        k = self.k if k is None else self._check_k(k)

        # Normalise dims; None reduces over every dim, as torch.mean does.
        if dim is None:
            dims = list(range(xs.ndim))
        else:
            dims = [dim] if isinstance(dim, int) else list(dim)
        dims = sorted(d % xs.ndim for d in dims)
        kept = [d for d in range(xs.ndim) if d not in dims]

        # Flatten the reduced dims into one trailing dim, so that the k extreme
        # values are taken over all of them jointly, not one dim at a time.
        def flatten(t):
            return t.permute(kept + dims).reshape(*[t.shape[d] for d in kept], -1)

        flat = flatten(xs)
        k    = min(k, flat.shape[-1])

        if mask is None:
            vals, _ = torch.topk(flat, k=k, dim=-1, largest=self._largest, sorted=False)
            result  = torch.mean(vals, dim=-1)
        else:
            # Masked-out entries are filled with -inf / +inf so they are only
            # selected when fewer than k entries are valid; the mean is then
            # taken over the valid selected entries only.
            xs, w = _mask_weights(xs, mask)
            flat, valid = flatten(xs), flatten(w > 0)
            fill = float("-inf") if self._largest else float("inf")
            vals, idx = torch.topk(torch.where(valid, flat, fill), k=k, dim=-1,
                                   largest=self._largest, sorted=False)
            sel   = torch.gather(valid, -1, idx)
            count = sel.sum(dim=-1)
            total = torch.where(sel, vals, torch.zeros_like(vals)).sum(dim=-1)
            result = total / count.clamp(min=1)

        if keepdim:
            for d in dims:
                result = result.unsqueeze(d)
        return result


class AggregMin(_AggregExtremum):
    """Min fuzzy aggregation: mean of the bottom-k values (default k=1 is true min).

    Parameters
    ----------
    bottom_k : int
        Number of smallest values to average.  Default: 1 (true min).
    """

    def __init__(self, bottom_k: int = 1) -> None:
        super().__init__(k=bottom_k, largest=False)
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
        super().__init__(k=top_k, largest=True)
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
        If ``True``, maps inputs to ``[eps, 1]`` with :func:`pi_0` before
        taking the log, so the value and gradient stay finite at 0.
    eps : float
        Shift used by :func:`pi_0` when *stable* is ``True``.

    Notes
    -----
    The geometric mean is the limit of :class:`AggregPMean` as ``p → 0`` and
    behaves like a normalised product, i.e. a soft ∀.  With a mask it is the
    weighted geometric mean ``exp(sum(w_i * log(x_i)) / sum(w_i))``.
    """

    def __init__(self, stable=DEFAULT_STABLE, eps=DEFAULT_EPS):
        self.stable = stable
        self.eps = eps

    def __repr__(self):
        return f"AggregGeometricMean(stable={self.stable}, eps={self.eps})"

    def __call__(self, xs, dim=None, mask=None, keepdim=False, stable=None):
        stable = self.stable if stable is None else stable

        if stable:
            xs = pi_0(xs, self.eps)

        if mask is not None:
            # Zero-weight entries are replaced by 1 (log 1 = 0) so they never
            # contribute and log(0) never gives 0 * -inf = NaN.
            xs, w = _mask_weights(xs, mask)
            safe_xs     = torch.where(w > 0, xs, torch.ones_like(xs))
            log_xs      = torch.log(safe_xs)
            numerator   = torch.sum(w * log_xs, dim=dim, keepdim=keepdim)
            denominator = torch.sum(w,          dim=dim, keepdim=keepdim)
            mean_log    = safe_div(numerator, denominator)
        else:
            mean_log = torch.mean(torch.log(xs), dim=dim, keepdim=keepdim)

        return torch.exp(mean_log)


class AggregPMean(AggregationOperator):
    """pMean fuzzy aggregation operator, ``(mean(x_i ** p)) ** (1 / p)``.

    For ``p >= 1`` it lies between the arithmetic mean and the max and suits
    ∃; for ``p < 1`` it moves towards the min and suits ∀.  ``p = 0`` is the
    geometric mean.  With a mask it is the weighted mean
    ``(sum(w_i * x_i ** p) / sum(w_i)) ** (1 / p)``.
    """
    def __init__(self, p=2, stable=DEFAULT_STABLE, eps=DEFAULT_EPS):

        self.p = p
        self.stable = stable
        self.eps = eps

    def __repr__(self):
        return f"AggregPMean(p={self.p}, stable={self.stable}, eps={self.eps})"

    def __call__(self, xs, dim=None, keepdim=False, mask=None, p=None, stable=None):

        p = self.p if p is None else p
        stable = self.stable if stable is None else stable

        if p == 0:
            # The p-mean tends to the geometric mean as p -> 0.
            return AggregGeometricMean(stable=stable, eps=self.eps)(xs, dim=dim, keepdim=keepdim, mask=mask)

        if stable:
            # Protects inner pow for p < 1
            xs = pi_0(xs, self.eps)

        if mask is not None:
            xs, w = _mask_weights(xs, mask)
            # Masked-out entries get a finite placeholder: 0 ** p is inf for p < 0.
            xs_pow      = torch.pow(torch.where(w > 0, xs, torch.ones_like(xs)), p)
            numerator   = torch.sum(w * xs_pow, dim=dim, keepdim=keepdim)
            denominator = torch.sum(w,          dim=dim, keepdim=keepdim)
            empty       = denominator == 0
            mean_val    = safe_div(numerator, denominator)
            if stable:
                mean_val = pi_0(mean_val, self.eps)
            # An empty mask gives a finite value with finite gradients, also for p < 0.
            return torch.pow(torch.where(empty, torch.ones_like(mean_val), mean_val), 1 / p)
        else:
            mean_val = torch.mean(torch.pow(xs, p), dim=dim, keepdim=keepdim)
            if stable:
                mean_val = pi_0(mean_val, self.eps)
            return torch.pow(mean_val, 1 / p)


class AggregPMeanError(AggregationOperator):
    """pMeanError fuzzy aggregation operator.

    Defined as ``1 - pMean(1 - xs)``.
    """
    def __init__(self, p=2, stable=DEFAULT_STABLE, eps=DEFAULT_EPS):
        self.p = p
        self.stable = stable
        self.eps = eps
        self._pmean_op = AggregPMean(p=p, stable=stable, eps=eps)

    def __repr__(self):
        return f"AggregPMeanError(p={self.p}, stable={self.stable}, eps={self.eps})"

    def __call__(self, xs, dim=None, keepdim=False, mask=None, p=None, stable=None):
        p_call      = self.p      if p      is None else p
        stable_call = self.stable if stable is None else stable
        return 1. - self._pmean_op(1. - xs, dim=dim, keepdim=keepdim, mask=mask, p=p_call, stable=stable_call)
