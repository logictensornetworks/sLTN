from __future__ import annotations

__all__ = ["Logic"]

from typing import Dict, List, Optional, TYPE_CHECKING
from sltn.interpretation.operators.base import AggregationOperator, UnaryConnectiveOperator, BinaryConnectiveOperator
from sltn.interpretation.operators import aggregator, conjunction, disjunction, implication, negation
from sltn.interpretation.operators.utils import DEFAULT_EPS

if TYPE_CHECKING:
    from sltn.signature.signature import Signature

# Alternate spellings accepted for the inner ``{"forall": op, "exists": op}``
# StructAgg entry keys, so both keyword and symbol forms work interchangeably.
_KIND_ALIASES: Dict[str, tuple] = {
    "forall": ("forall", "∀"),
    "exists": ("exists", "∃"),
}


def _lookup_kind(entry: Dict[str, AggregationOperator], kind: str):
    """Return entry[kind], tolerating the '∀'/'∃' symbol spellings too."""
    for alt in _KIND_ALIASES.get(kind, (kind,)):
        if alt in entry:
            return entry[alt]
    return None


def _parse_kind(q_sym: str, key: str) -> str:
    """Map the quantifier part of a ``"∀,t"`` key to ``"forall"``/``"exists"``."""
    q_sym = q_sym.strip()
    for kind, spellings in _KIND_ALIASES.items():
        if q_sym in spellings:
            return kind
    raise KeyError(
        f"Unknown quantifier {q_sym!r} in logic key {key!r}. "
        f"Use one of 'forall', '∀', 'exists', '∃'."
    )

# ---------------------------------------------------------------------------
# Symbol → Logic field mapping
# ---------------------------------------------------------------------------
_LOGIC_OP_FIELDS = ("And", "Or", "Not", "Implies", "Equiv", "Forall", "Exists", "SatAgg", "StructAgg")

_DEFAULT_SYMBOLS_DICT = {
    "And": "∧",
    "Or": "∨",
    "Not": "¬",
    "Implies": "→",
    "Equiv": "↔",
    "Forall": "∀",
    "Exists": "∃",
    "SatAgg": "SatAgg",
}

# All textual aliases that map to a Logic attribute name.
# Used by Logic.__getitem__ / __setitem__.
_SYMBOL_TO_FIELD: dict[str, str] = {
    # Conjunction
    "∧": "And", "&": "And", "and": "And",
    # Disjunction
    "∨": "Or", "or": "Or",
    # Negation
    "¬": "Not", "~": "Not", "not": "Not",
    # Implication
    "→": "Implies", "->": "Implies", "=>": "Implies", "implies": "Implies",
    # Equivalence
    "↔": "Equiv", "<->": "Equiv", "<=>": "Equiv", "iff": "Equiv",
    # Universal quantifier
    "∀": "Forall", "forall": "Forall",
    # Existential quantifier
    "∃": "Exists", "exists": "Exists",
    # KB-level satisfaction aggregation
    "SatAgg": "SatAgg",
}

class Logic:
    """Bundle of fuzzy logic operators for an Interpretation.
    """

    def __init__(
        self,
        And:        Optional[BinaryConnectiveOperator]                      = None,
        Or:         Optional[BinaryConnectiveOperator]                      = None,
        Not:        Optional[UnaryConnectiveOperator]                       = None,
        Implies:    Optional[BinaryConnectiveOperator]                      = None,
        Equiv:      Optional[BinaryConnectiveOperator]                      = None,
        Forall:     Optional[AggregationOperator]                          = None,
        Exists:     Optional[AggregationOperator]                          = None,
        SatAgg:     Optional[AggregationOperator]                          = None,
        StructAgg:  Optional[Dict[str, Dict[str, AggregationOperator]]]    = None,
    ) -> None:
        """Create a fuzzy logic operator bundle.

        All parameters are optional; unset fields are filled in with sensible
        defaults by :meth:`with_defaults`.  If both ``And`` and ``Not`` are
        provided, unset ``Or``, ``Implies``, and ``Equiv`` are derived from
        them automatically.

        Parameters
        ----------
        And, Or, Implies, Equiv : BinaryConnectiveOperator, optional
            Binary connective operators.
        Not : UnaryConnectiveOperator, optional
            Unary negation operator.
        Forall, Exists : AggregationOperator, optional
            Aggregation operators for first-order quantifiers.
        SatAgg : AggregationOperator, optional
            Aggregation operator for KB-level clause aggregation.  This must
            be a plain :class:`~sltn.interpretation.operators.AggregationOperator` instance
            (e.g. ``AggregPMeanError(p=2)``)
        StructAgg : dict, optional
            Per-axis structural aggregation operators, keyed as
            ``{axis_name: {"forall": op, "exists": op}}``.
        """
        self.And       = And
        self.Or        = Or
        self.Not       = Not
        self.Implies   = Implies
        self.Equiv     = Equiv
        self.Forall    = Forall
        self.Exists    = Exists
        self.SatAgg    = SatAgg
        self.StructAgg = StructAgg

        # Optional back-reference to the owning Signature, set by Interpretation.
        # When present, it lets composite StructAgg keys ("∀,t" / "∀,T") resolve
        # any structural variable (base, alias, or independently declared) to
        # the canonical Dimension it ranges over. Logic remains usable
        # standalone (e.g. Logic.classical(...)) when no signature is attached.
        self._signature: Optional["Signature"] = None


    def with_defaults(self, stable = True, eps = DEFAULT_EPS) -> Logic:
        """Return a copy with any None field replaced by a sensible default.

        When both ``And`` and ``Not`` are set explicitly, any unset connectives
        among ``Or`` and ``Implies`` are derived from them via the classical
        de-Morgan definitions (``OrDual``, ``ImpliesDual``).  Otherwise those
        fields keep their independent defaults.  An unset ``Equiv`` is always
        ``Equiv(And, Implies)``, built from the final ``And`` and ``Implies``.

        Note: ``StructAgg`` has no built-in default.  Every structural axis
        used in a formula must be declared explicitly.
        """
        and_op = self.And or conjunction.AndProd(stable, eps)
        not_op = self.Not or negation.NotStandard()
        derive_connectives = self.And is not None and self.Not is not None

        if self.Or is None:
            or_op = (
                disjunction.OrDual(and_op, not_op)
                if derive_connectives
                else disjunction.OrProbSum(stable, eps)
            )
        else:
            or_op = self.Or

        if self.Implies is None:
            implies_op = (
                implication.ImpliesDual(and_op, not_op)
                if derive_connectives
                else implication.ImpliesGoguen(stable, eps)
            )
        else:
            implies_op = self.Implies

        if self.Equiv is None:
            # Built from the final And and Implies, so a user-set Implies is used by <-> too.
            equiv_op = implication.Equiv(and_op, implies_op)
        else:
            equiv_op = self.Equiv

        result = Logic(
            And=and_op,
            Or=or_op,
            Not=not_op,
            Implies=implies_op,
            Equiv=equiv_op,
            Forall=self.Forall or aggregator.AggregPMeanError(stable=stable, eps=eps),
            Exists=self.Exists or aggregator.AggregPMean(stable=stable, eps=eps),
            SatAgg=self.SatAgg or aggregator.AggregPMeanError(stable=stable, eps=eps),
            # Copy two levels deep, so that setting a structural aggregator on the
            # result does not change this Logic or other Logics built from it.
            StructAgg=(
                {axis: dict(ops) for axis, ops in self.StructAgg.items()}
                if self.StructAgg is not None
                else None
            ),
        )
        result._signature = self._signature
        return result

    @classmethod
    def classical(cls, struct_agg_axes: Optional[List[str]] = None) -> Logic:
        """Return a hard Boolean logic bundle.

        All operators are closed over ``{0, 1}``: conjunction is min,
        disjunction is max, negation is ``1 - x``, implication is Gödel
        (``1 if x ≤ y else y``), quantifiers and aggregation use min/max.

        Parameters
        ----------
        struct_agg_axes : list of str, optional
        """
        
        struct_agg = None
        if struct_agg_axes:
            struct_agg = {
                ax: {"forall": aggregator.AggregMin(), "exists": aggregator.AggregMax()}
                for ax in struct_agg_axes
            }
        return cls(
            And=conjunction.AndMin(),
            Or=disjunction.OrMax(),
            Not=negation.NotStandard(),
            Implies=implication.ImpliesGodel(smooth=False),
            Equiv=implication.Equiv(conjunction.AndMin(), implication.ImpliesGodel(smooth=False)),
            Forall=aggregator.AggregMin(),
            Exists=aggregator.AggregMax(),
            SatAgg=aggregator.AggregMin(),
            StructAgg=struct_agg,
        )

    def _canonical_axis(self, axis: str) -> str:
        """Resolve *axis* to its canonical Dimension name, if a signature is attached."""
        axis = axis.strip()
        if self._signature is not None:
            return self._signature.resolve_structural_axis(axis)
        return axis

    def _struct_entry(self, axis: str) -> dict | None:
        """The StructAgg entry for *axis*, matching keys by the dimension they resolve to.

        Keys set before a signature was attached (e.g. ``"t"``) stay unresolved, so
        they are compared through the signature here.
        """
        if not self.StructAgg:
            return None
        canonical = self._canonical_axis(axis)
        if canonical in self.StructAgg:
            return self.StructAgg[canonical]
        for key, entry in self.StructAgg.items():
            if self._canonical_axis(key) == canonical:
                return entry
        return None

    def __getitem__(self, key: str):
        """Read a logic operator by symbol or keyword.

        Simple keys (e.g. ``"∧"``, ``"and"``, ``"&"``) return the corresponding
        operator attribute.  Composite keys of the form ``"∀,t"`` or ``"∃,t"``
        return the structural aggregation operator for axis ``t``.  The axis
        is canonicalised to its ``Dimension`` name when a signature is
        attached (via an owning :class:`Interpretation`), so a variable
        (``"t"``), one of its aliases (``"t1"``), or the dimension itself
        (``"T"``) all resolve to the same entry.

        Examples::

            logic["∧"]      # → logic.And
            logic["forall"] # → logic.Forall
            logic["∀,t"]    # → logic.StructAgg["T"]["forall"]  (t ranges over dimension T)
        """
        if "," in key:
            q_sym, axis = key.split(",", 1)
            q = _parse_kind(q_sym, key)
            if self.StructAgg is None:
                raise KeyError(f"No structural aggregation operators registered (key={key!r}).")
            entry = self._struct_entry(axis)
            if entry is None:
                raise KeyError(f"No structural aggregation operators registered for axis '{axis.strip()}'.")
            value = _lookup_kind(entry, q)
            if value is None:
                raise KeyError(f"No structural '{q}' operator for axis '{axis.strip()}'.")
            return value
        if key not in _SYMBOL_TO_FIELD:
            raise KeyError(f"Unknown logic operator key {key!r}. Valid keys: {sorted(_SYMBOL_TO_FIELD)}")
        return getattr(self, _SYMBOL_TO_FIELD[key])

    def __setitem__(self, key: str, value) -> None:
        """Set a logic operator by symbol or keyword.

        Simple keys assign the corresponding attribute.  Composite keys of the
        form ``"∀,t"`` or ``"∃,t"`` set the structural aggregation operator
        for axis ``t``.  When a signature is attached, ``t`` is canonicalised
        to the ``Dimension`` it ranges over, so setting via a variable name
        binds the whole dimension: every other variable (aliases or
        independently declared) ranging over that dimension inherits it too.

        Examples::

            logic["∧"] = AndProd(stable=True)
            logic["∀,t"] = AggregPMean(p=2)   # binds dimension T (t ranges over T)
            logic["∃,t"] = AggregPMean(p=1)
        """
        if "," in key:
            q_sym, axis = key.split(",", 1)
            q = _parse_kind(q_sym, key)
            if self.StructAgg is None:
                self.StructAgg = {}
            canonical = self._canonical_axis(axis)
            self.StructAgg.setdefault(canonical, {})[q] = value
        else:
            if key not in _SYMBOL_TO_FIELD:
                raise KeyError(f"Unknown logic operator key {key!r}. Valid keys: {sorted(_SYMBOL_TO_FIELD)}")
            setattr(self, _SYMBOL_TO_FIELD[key], value)

    def __repr__(self) -> str:
        lines = []
        for f in _LOGIC_OP_FIELDS:
            val = getattr(self, f)
            if val is None:
                continue
            if f == 'StructAgg':
                for ax, ops in val.items():
                    for q, op in ops.items():
                        if q in _KIND_ALIASES["forall"]:
                            q = "∀"
                        elif q in _KIND_ALIASES["exists"]:
                            q = "∃"
                        lines.append(f"    ({q}, {ax}) : {op}")
            else:
                lines.append(f"    {_DEFAULT_SYMBOLS_DICT[f]}: {val}")
        return "Logic(\n" + "\n".join(lines) + "\n  )"
