import torch

# Default shift of the stable operators: pi_0 maps [0, 1] onto [eps, 1], pi_1 onto [0, 1 - eps].
# Values below about 1e-7 vanish in float32 arithmetic (1 - 1e-8 == 1).
DEFAULT_EPS: float = 1e-6
# Default for the `stable` flag of every connective and aggregator.
DEFAULT_STABLE: bool = True
# Default threshold used by Logic.classical() to binarise atomic truth values.
DEFAULT_CLASSICAL_THRESHOLD: float = 0.5
# Experimental, not used by any operator: smooth clamping helpers kept for future work.
DEFAULT_BETA: float = 10.

def pi_0(x, eps=DEFAULT_EPS):
    """
    Affine map of [0, 1] onto [eps, 1]: ``(1 - eps) * x + eps``.

    Used by stable operators whose value or gradient is singular at 0
    (log, division, powers with exponent below 1).
    """
    return (1 - eps) * x + eps


def pi_1(x, eps=DEFAULT_EPS):
    """
    Affine map of [0, 1] onto [0, 1 - eps]: ``(1 - eps) * x``.

    Used by stable operators whose gradient is singular at 1
    (``sqrt(1 - x**2)`` in :class:`AndCos`).
    """
    return (1 - eps) * x


def safe_div(numerator, denominator):
    """
    numerator / denominator for weighted means, where both are sums over the same non-negative weights. An empty
    guard (all weights 0) has numerator 0, so dividing it by 1 instead of 0 gives 0 rather than NaN, with no NaN in
    the gradient either. Any positive denominator, however small, is used exactly.
    """
    return torch.div(numerator, torch.where(denominator > 0, denominator, torch.ones_like(denominator)))


def squareplus(x, beta = DEFAULT_BETA):
    return 0.5*(x+torch.sqrt(x**2+1/beta))


def softplus(x: torch.Tensor, beta: float = DEFAULT_BETA) -> torch.Tensor:
    return torch.nn.Softplus(beta)(x)


def smooth_clamp(x, min=None, max=None, beta = DEFAULT_BETA, smoothing_method="softplus"):
    """
    Smooth approximation of torch.clamp(x, min=min, max=max).

    This implementation composes smooth max and smooth min via softplus.
    As sharpness -> +inf, the function converges pointwise to hard clamp.
    
    Args:
        x: Input tensor.
        beta: Higher values approach hard clamp; lower values are smoother.
        min: Lower clamp bound (possibly None).
        max: Upper clamp bound (possibly None).
    
    Returns:
        Tensor smoothly clamped in [min, max].
    """
    if not beta > 0:
        raise ValueError("beta must be > 0")
    
    if smoothing_method == "softplus":
        smoother = softplus
    elif smoothing_method == "squareplus":
        smoother = squareplus
    else:
        raise ValueError(f"Unknown smoothing_method: {smoothing_method}")
    
    if min is not None and max is not None:
        if not min < max:
            raise ValueError("min must be < max")
        res = max - smoother(max - x, beta) + smoother(min - x, beta)
        return torch.clamp(res, min, max)
    elif min is not None:
        res = min + smoother(x - min, beta)
        return torch.clamp(res, min=min)
    elif max is not None:
        res = max - smoother(max - x, beta)
        return torch.clamp(res, max=max)
    else:
        raise ValueError("At least one of min or max must be specified.")