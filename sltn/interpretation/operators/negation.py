import torch
from .base import UnaryConnectiveOperator

class NotStandard(UnaryConnectiveOperator):
    """Standard fuzzy negation: ¬A = 1 - A."""

    def __repr__(self) -> str:
        return "NotStandard()"

    def __call__(self, x) -> torch.Tensor:
        return 1. - x

class NotGodel(UnaryConnectiveOperator):
    """Gödel (crisp) negation: ¬A = 1 if A == 0 else 0."""

    def __repr__(self) -> str:
        return "NotGodel()"

    def __call__(self, x) -> torch.Tensor:
        return torch.eq(x, 0.).to(x.dtype)
