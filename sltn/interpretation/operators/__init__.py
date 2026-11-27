"""
sltn.interpretation.operators — Fuzzy logic operator library.

Provides base classes and concrete implementations for connectives and
aggregation operators used by :class:`~sltn.interpretation.Logic` and
:class:`~sltn.interpretation.Interpretation`.

Connectives:
  - :mod:`.conjunction`  — And: AndMin, AndProd, AndPMean, AndCos, AndLuk
  - :mod:`.disjunction`  — Or: OrMax, OrProbSum, OrLuk, OrDual (¬(¬A ∧ ¬B) from any And/Not),
                           and XOrClassical
  - :mod:`.negation`     — Not: NotStandard, NotGodel
  - :mod:`.implication`  — Implies: ImpliesKleeneDienes, ImpliesGodel, ImpliesReichenbach,
                           ImpliesGoguen, ImpliesLuk, ImpliesDual (¬(A ∧ ¬B) from any And/Not);
                           equivalence: Equiv, EquivSimilarity

Aggregation:
  - :mod:`.aggregator`   — AggregMin, AggregMax, AggregGeometricMean, AggregPMean, AggregPMeanError
"""
from __future__ import annotations

from .aggregator import (
    AggregGeometricMean,
    AggregMax,
    AggregMin,
    AggregPMean,
    AggregPMeanError,
)
from .base import (
    AggregationOperator,
    BinaryConnectiveOperator,
    ConnectiveOperator,
    UnaryConnectiveOperator,
)
from .conjunction import (
    AndCos,
    AndLuk,
    AndMin,
    AndPMean,
    AndProd,
)
from .disjunction import (
    OrDual,
    OrLuk,
    OrMax,
    OrProbSum,
    XOrClassical,
)
from .implication import (
    Equiv,
    EquivSimilarity,
    ImpliesDual,
    ImpliesGodel,
    ImpliesGoguen,
    ImpliesKleeneDienes,
    ImpliesLuk,
    ImpliesReichenbach,
)
from .negation import (
    NotGodel,
    NotStandard,
)
from .utils import DEFAULT_CLASSICAL_THRESHOLD, DEFAULT_EPS, DEFAULT_STABLE, pi_0, pi_1

__all__ = [
    "DEFAULT_CLASSICAL_THRESHOLD",
    "DEFAULT_EPS",
    "DEFAULT_STABLE",
    "AggregGeometricMean",
    "AggregMax",
    "AggregMin",
    "AggregPMean",
    "AggregPMeanError",
    "AggregationOperator",
    "AndCos",
    "AndLuk",
    "AndMin",
    "AndPMean",
    "AndProd",
    "BinaryConnectiveOperator",
    "ConnectiveOperator",
    "Equiv",
    "EquivSimilarity",
    "ImpliesDual",
    "ImpliesGodel",
    "ImpliesGoguen",
    "ImpliesKleeneDienes",
    "ImpliesLuk",
    "ImpliesReichenbach",
    "NotGodel",
    "NotStandard",
    "OrDual",
    "OrLuk",
    "OrMax",
    "OrProbSum",
    "UnaryConnectiveOperator",
    "XOrClassical",
    "pi_0",
    "pi_1",
]
