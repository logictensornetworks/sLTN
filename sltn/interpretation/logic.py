from __future__ import annotations

__all__ = ["Logic"]

from typing import TYPE_CHECKING

from sltn.fol import formula
from sltn.interpretation.operators import (
    aggregator,
    conjunction,
    disjunction,
    implication,
    negation,
)
from sltn.interpretation.operators.base import (
    AggregationOperator,
    BinaryConnectiveOperator,
    UnaryConnectiveOperator,
)
from sltn.interpretation.operators.utils import (
    DEFAULT_CLASSICAL_THRESHOLD,
    DEFAULT_EPS,
    DEFAULT_STABLE,
)
from sltn.parse.tokenizer import BUILTIN_TOKENS

if TYPE_CHECKING:
    from sltn.signature.signature import Signature

# Symbol → Logic field mapping

_LOGIC_OP_FIELDS = ("And", "Or", "Not", "Implies", "Equiv", "Forall", "Exists", "SatAgg", "StructAgg")

# Printed symbol of each Logic field, taken from the formula classes so that
# print(logic) and printed formulas agree.
_DEFAULT_SYMBOLS_DICT = {
    **{cls.op_attr: cls.symbol for cls in (formula.And, formula.Or, formula.Implies, formula.Equivalent)},
    "Not": formula.Not.symbol,
    "Forall": formula.Forall.symbol,
    "Exists": formula.Exists.symbol,
    "SatAgg": "SatAgg",
}

# Every spelling the parser accepts for a connective or quantifier (e.g. "∧", "&",
# "and") maps to the Logic attribute that interprets it. Derived from the lexer
# table so that Logic keys and formula syntax cannot drift apart.
_TOKEN_KIND_TO_FIELD = {
    "and": "And", "or": "Or", "not": "Not", "implies": "Implies", "iff": "Equiv",
    "forall": "Forall", "exists": "Exists",
}
_SYMBOL_TO_FIELD: dict[str, str] = {
    spelling: _TOKEN_KIND_TO_FIELD[kind]
    for spelling, kind in BUILTIN_TOKENS.items()
    if kind in _TOKEN_KIND_TO_FIELD
}
_SYMBOL_TO_FIELD["SatAgg"] = "SatAgg"


def _parse_kind(q_sym: str, key: str) -> str:
    """Map a quantifier spelling (``"∀"``, ``"forall"``, ``"∃"``, …) to ``"forall"``/``"exists"``."""
    field = _SYMBOL_TO_FIELD.get(q_sym.strip())
    if field not in ("Forall", "Exists"):
        raise KeyError(
            f"Unknown quantifier {q_sym.strip()!r} in logic key {key!r}. "
            f"Use one of 'forall', '∀', 'exists', '∃'."
        )
    return field.lower()


def _normalize_struct_agg(
    struct_agg: dict[str, dict[str, AggregationOperator]] | None,
) -> dict[str, dict[str, AggregationOperator]] | None:
    """Copy *struct_agg* two levels deep, with quantifier keys spelled ``"forall"``/``"exists"``."""
    if struct_agg is None:
        return None
    result: dict[str, dict[str, AggregationOperator]] = {}
    for axis, ops in struct_agg.items():
        entry: dict[str, AggregationOperator] = {}
        for q_sym, op in ops.items():
            kind = _parse_kind(q_sym, f"StructAgg[{axis!r}][{q_sym!r}]")
            if kind in entry:
                raise ValueError(f"StructAgg[{axis!r}] sets '{kind}' twice.")
            entry[kind] = op
        result[axis] = entry
    return result


class Logic:
    """Bundle of fuzzy logic operators for an Interpretation.
    """

    def __init__(
        self,
        And:        BinaryConnectiveOperator | None                      = None,
        Or:         BinaryConnectiveOperator | None                      = None,
        Not:        UnaryConnectiveOperator | None                       = None,
        Implies:    BinaryConnectiveOperator | None                      = None,
        Equiv:      BinaryConnectiveOperator | None                      = None,
        Forall:     AggregationOperator | None                          = None,
        Exists:     AggregationOperator | None                          = None,
        SatAgg:     AggregationOperator | None                          = None,
        StructAgg:  dict[str, dict[str, AggregationOperator]] | None    = None,
        *,
        stable:     bool                                                    = DEFAULT_STABLE,
        eps:        float                                                   = DEFAULT_EPS,
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
        stable, eps : optional
            ``stable`` and ``eps`` of the operators that :meth:`with_defaults`
            fills in.  Operators passed explicitly keep their own settings.
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
        self.stable    = stable
        self.eps       = eps
        # Binarisation threshold for atomic truth values; set only by classical().
        self._threshold: float | None = None

        # Optional back-reference to the owning Signature, set by Interpretation.
        # When present, it lets composite StructAgg keys ("∀,t" / "∀,T") resolve
        # any structural variable (base, alias, or independently declared) to
        # the canonical Dimension it ranges over. Logic remains usable
        # standalone (e.g. Logic.classical(...)) when no signature is attached.
        self._signature: Signature | None = None

    @property
    def StructAgg(self) -> dict[str, dict[str, AggregationOperator]] | None:
        """Per-dimension structural aggregators, ``{dim: {"forall": op, "exists": op}}``.

        Assigning a dict stores a copy whose quantifier keys are normalised, so
        ``"∀"``/``"∃"`` and ``"forall"``/``"exists"`` are interchangeable.
        """
        return self._struct_agg

    @StructAgg.setter
    def StructAgg(self, value: dict[str, dict[str, AggregationOperator]] | None) -> None:
        self._struct_agg = _normalize_struct_agg(value)

    def with_defaults(self, stable: bool | None = None, eps: float | None = None) -> Logic:
        """Return a copy with any None field replaced by a sensible default.

        Defaults are built with ``stable`` and ``eps`` if given, else with this
        Logic's own; the copy records the values used.

        When both ``And`` and ``Not`` are set explicitly, any unset connectives
        among ``Or`` and ``Implies`` are derived from them via the classical
        de-Morgan definitions (``OrDual``, ``ImpliesDual``).  Otherwise those
        fields keep their independent defaults.  An unset ``Equiv`` is always
        ``Equiv(And, Implies)``, built from the final ``And`` and ``Implies``.

        ``StructAgg`` holds only explicit per-dimension overrides; a structural
        quantifier without one uses the current ``Forall`` / ``Exists``
        (see :meth:`struct_op`).
        """
        stable = self.stable if stable is None else stable
        eps = self.eps if eps is None else eps
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
                else implication.ImpliesReichenbach(stable, eps)
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
            # The StructAgg setter copies two levels deep, so setting a structural
            # aggregator on the result does not change this Logic.
            StructAgg=self.StructAgg,
            stable=stable,
            eps=eps,
        )
        result._threshold = self._threshold
        result._signature = self._signature
        return result

    @classmethod
    def classical(
        cls,
        *,
        threshold: float | None = DEFAULT_CLASSICAL_THRESHOLD,
    ) -> Logic:
        """Return a classical Boolean logic bundle.

        All operators are closed over ``{0, 1}``: conjunction is min,
        disjunction is max, negation is ``1 - x``, implication is Gödel
        (``1 if x ≤ y else y``), quantifiers and aggregation use min/max
        (structural quantifiers too, unless overridden per dimension).
        By default, atomic truth values are binarised at ``threshold``
        (``DEFAULT_CLASSICAL_THRESHOLD = 0.5``); pass ``threshold=None`` for
        unthresholded Zadeh/Gödel operators over ``[0, 1]``.

        Parameters
        ----------
        threshold : float or None, optional
            Truth values are binarised where they enter the logic (atomic
            formulas, structural relations and Bool terms used as formulas) via
            ``(x >= threshold).float()`` before connectives and quantifiers are
            applied (default ``0.5``).  Pass ``None`` to skip thresholding.
        """
        logic = cls(
            And=conjunction.AndMin(),
            Or=disjunction.OrMax(),
            Not=negation.NotStandard(),
            Implies=implication.ImpliesGodel(),
            Equiv=implication.Equiv(conjunction.AndMin(), implication.ImpliesGodel()),
            Forall=aggregator.AggregMin(),
            Exists=aggregator.AggregMax(),
            SatAgg=aggregator.AggregMin(),
        )
        logic._threshold = threshold
        return logic

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

    def struct_op(self, kind: str, axis: str) -> AggregationOperator:
        """Return the aggregation operator for quantifier *kind* over the structural axis *axis*.

        *kind* is ``"forall"`` or ``"exists"`` (or another accepted spelling, such as ``"∀"``).
        *axis* is a dimension or a structural variable; a variable is resolved to its dimension.

        If ``StructAgg`` defines an operator for this dimension and *kind*, that operator is
        returned. Otherwise ``self.Forall`` (for ``"forall"``) or ``self.Exists`` (for
        ``"exists"``) is returned by reference. The lookup is done on every
        call, so assigning a new ``Forall`` / ``Exists`` changes the result for every
        dimension that has no ``StructAgg`` entry for *kind*.

        Raises
        ------
        KeyError
            If a signature is attached and *axis* is neither a declared dimension nor a
            structural variable, or if the required ``Forall`` / ``Exists`` is ``None``.
        """
        kind = _parse_kind(kind, kind)
        entry = self._struct_entry(axis)
        if entry is not None and kind in entry:
            return entry[kind]
        if self._signature is not None and self._canonical_axis(axis) not in self._signature.dimensions:
            raise KeyError(
                f"Unknown structural axis {axis.strip()!r}: "
                f"expected a dimension or a structural variable of the signature."
            )
        op = self.Forall if kind == "forall" else self.Exists
        if op is None:
            raise KeyError(
                f"No '{kind}' operator for structural axis {axis.strip()!r}: "
                f"Logic.{kind.capitalize()} is not set. Use Logic.with_defaults() or set it explicitly."
            )
        return op

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
            logic["∀,t"]    # → logic.StructAgg["T"]["forall"] if set, else logic.Forall
        """
        if "," in key:
            q_sym, axis = key.split(",", 1)
            return self.struct_op(_parse_kind(q_sym, key), axis)
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
                        lines.append(f"    ({_DEFAULT_SYMBOLS_DICT[q.capitalize()]}, {ax}) : {op}")
            else:
                lines.append(f"    {_DEFAULT_SYMBOLS_DICT[f]}: {val}")
        lines.append(f"    stable={self.stable}, eps={self.eps}")
        if self._threshold is not None:
            lines.append(f"    threshold={self._threshold}")
        return "Logic(\n" + "\n".join(lines) + "\n  )"
