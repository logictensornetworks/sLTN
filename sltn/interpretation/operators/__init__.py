"""
sltn.interpretation.operators — Fuzzy logic operator library.

Provides base classes and concrete implementations for connectives and
aggregation operators used by :class:`~sltn.interpretation.Logic` and
:class:`~sltn.interpretation.Interpretation`.

Connectives:
  - :mod:`.conjunction`  — And (min, product, pMean, cosine, Łukasiewicz)
  - :mod:`.disjunction`  — Or (max, probabilistic sum, classical, Łukasiewicz)
  - :mod:`.negation`     — Not (standard, Gödel)
  - :mod:`.implication`  — Implies (Kleene-Dienes, Gödel, Reichenbach, Goguen, Łukasiewicz, classical)

Aggregation:
  - :mod:`.aggregator`   — AggregMin, AggregMax, AggregGeometricMean, AggregPMean, AggregPMeanError
"""
from .base import (
    ConnectiveOperator,
    UnaryConnectiveOperator,
    BinaryConnectiveOperator,
    AggregationOperator,
)
from .utils import DEFAULT_EPS, smooth_clamp, pi_0, pi_1
from .conjunction import (
    AndMin,
    AndProd,
    AndPMean,
    AndCos,
    AndLuk,
)
from .disjunction import (
    OrMax,
    OrProbSum,
    OrDual,
    OrLuk,
    XOrClassical,
)
from .negation import (
    NotStandard,
    NotGodel,
)
from .implication import (
    ImpliesKleeneDienes,
    ImpliesGodel,
    ImpliesReichenbach,
    ImpliesGoguen,
    ImpliesLuk,
    ImpliesDual,
    Equiv,
    EquivSimilarity,
)
from .aggregator import (
    AggregMin,
    AggregMax,
    AggregGeometricMean,
    AggregPMean,
    AggregPMeanError,
)

__all__ = [
    "DEFAULT_EPS",
    "smooth_clamp",
    "pi_0",
    "pi_1",
    "ConnectiveOperator",
    "UnaryConnectiveOperator",
    "BinaryConnectiveOperator",
    "AggregationOperator",
    "AndMin",
    "AndProd",
    "AndPMean",
    "AndCos",
    "AndLuk",
    "OrMax",
    "OrProbSum",
    "OrDual",
    "OrLuk",
    "XOrClassical",
    "NotStandard",
    "NotGodel",
    "ImpliesDual",
    "ImpliesKleeneDienes",
    "ImpliesGodel",
    "ImpliesReichenbach",
    "ImpliesGoguen",
    "ImpliesLuk",
    "Equiv",
    "EquivSimilarity",
    "AggregMin",
    "AggregMax",
    "AggregGeometricMean",
    "AggregPMean",
    "AggregPMeanError",
]
