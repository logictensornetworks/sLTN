"""
sltn.interpretation.core.axis — Named, role-tagged axis descriptor.

An :class:`Axis` labels one dimension of an :class:`~sltn.interpretation.core.tensor.Tensor`
tensor with a name and a semantic role (``'variable'``, ``'structural'``, or ``'domain'``).
"""
from __future__ import annotations

__all__ = ["Axis"]

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Axis — named tensor-dimension tag
# --------------------------------------------------------------------------- 

@dataclass(frozen=True)
class Axis:
    """A named, role-tagged axis of a tensor inside a Tensor.

    Roles:
      - ``'variable'``   — a batch axis over individuals (one per bound FOL variable)
      - ``'structural'`` — a structural (world/time/event) axis
      - ``'domain'``     — a feature/domain axis (part of the element type)
    """
    name: str
    role: str

    def __post_init__(self) -> None:
        allowed = {'variable', 'structural', 'domain'}
        if self.role not in allowed:
            raise ValueError(
                f"Axis.role must be one of {sorted(allowed)}; got '{self.role}'"
            )

    def __repr__(self) -> str:
        return f"Axis(name={self.name}, role={self.role})"