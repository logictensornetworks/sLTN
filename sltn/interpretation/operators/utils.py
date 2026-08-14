import torch

# Default epsilon for numerical stability (used to clamp truth values away from 0 or 1).
DEFAULT_EPS: float = 1e-12
DEFAULT_BETA: float = 10.

def pi_0(x, eps=DEFAULT_EPS):
    """
    Function that has to be used when we need to assure that the truth value in input to a fuzzy operator is never equal
    to zero, in such a way to avoid gradient problems. It maps the interval [0, 1] in the interval ]eps, 1].
    """
    return (1 - eps) * x + eps

def pi_1(x, eps=DEFAULT_EPS):
    """
    Function that has to be used when we need to assure that the truth value in input to a fuzzy operator is never equal
    to one, in such a way to avoid gradient problems. It maps the interval [0, 1] in the interval [0, 1-eps[.
    """
    return (1 - eps) * x


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
        res = max - smoother(max - x, beta) - (min - smoother(min - x, beta))
        return torch.clamp(res, min, max)
    elif min is not None:
        res = min + smoother(x - min, beta)
        return torch.clamp(res, min=min)
    elif max is not None:
        res = max - smoother(max - x, beta)
        return torch.clamp(res, max=max)
    else:
        raise ValueError("At least one of min or max must be specified.")