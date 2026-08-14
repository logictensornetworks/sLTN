"""
sltn.interpretation — Neural-symbolic interpretation layer.

Provides :class:`Interpretation` (symbol→tensor bindings and formula evaluation),
:class:`Logic` (fuzzy operator bundle), and the public ``apply_*`` functions.

The :mod:`sltn.interpretation.operators` sub-package contains all concrete
fuzzy logic operator implementations.
"""

from __future__ import annotations

from .interpretation import Interpretation, Assignment
from .apply import apply_connective, apply_quantifier, apply_sat_agg, TensorCallable
from .logic import Logic

__all__ = [
    "Assignment",
    "TensorCallable",
    "Interpretation",
    "Logic",
    "apply_connective",
    "apply_quantifier",
    "apply_sat_agg",
]
