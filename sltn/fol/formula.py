"""
sltn.fol.formula — Syntactic AST nodes for FOL formulas.

Formulas are pure data structures describing the syntax of a logical statement.
Evaluation is handled exclusively by :class:`~sltn.interpretation.Interpretation`.

Formulas are:
  - :class:`AtomicFormula`   — P(t1, …, tn)
  - :class:`And`, :class:`Or`, :class:`Not`, :class:`Implies`, :class:`Equivalent`
  - :class:`Forall`, :class:`Exists`
  - :class:`KB`              — knowledge base (flat set of axioms aggregated via SatAgg)
"""

from __future__ import annotations

__all__ = [
    "Formula",
    "TermFormula",
    "FormulaAxisSelect",
    "AtomicFormula",
    "And",
    "Or",
    "Not",
    "Implies",
    "Equivalent",
    "Forall",
    "Exists",
    "KB",
    "StructuralRelation",
]

from abc import ABC
from typing import Sequence

from sltn.fol.term import Term, _call_axis_dims
from sltn.signature.symbols import PredicateSymbol, FunctionSymbol, StructuralRelationSymbol


class Formula(ABC):
    """Abstract syntax node for a FOL formula."""

    # free variables
    _free_variables: set[str] | None = None

    @property
    def free_variables(self) -> set[str]:
        """The set of free variable names in this formula."""
        if self._free_variables is not None:
            return self._free_variables
        return self._compute_free_variables()

    def _compute_free_variables(self) -> set[str]:
        return set()

    # free structural variables
    _free_structural_variables: set[str] | None = None

    @property
    def free_structural_variables(self) -> set[str]:
        """The set of free structural variables in this formula."""
        if self._free_structural_variables is not None:
            return self._free_structural_variables
        return self._compute_free_structural_variables()

    def _compute_free_structural_variables(self) -> set[str]:
        return set()

    def _axis_dimensions(self) -> dict:
        """Free structural axis names, mapped to their Dimension (``None`` if not tracked)."""
        return dict.fromkeys(self.free_structural_variables)

    def is_clause(self) -> bool:
        """Return True iff the formula has no free variables (normal or structural)."""
        return not self.free_variables and not self.free_structural_variables


class TermFormula(Formula):
    """A bool-valued term used in formula position.

    This is used during definition expansion when a derived predicate body is
    provided as a term string.
    """

    def __init__(self, term: Term) -> None:
        self.term = term

    def _compute_free_variables(self) -> set[str]:
        return self.term.free_variables

    def _compute_free_structural_variables(self) -> set[str]:
        return self.term.free_structural_variables

    def _axis_dimensions(self) -> dict:
        return self.term._axis_dimensions()

    def __repr__(self) -> str:
        return str(self.term)


class FormulaAxisSelect(Formula):
    """A formula projected onto a single structural index.

    Syntax: ``(formula)[axis=index]``
    """

    def __init__(self, formula: Formula, axis: str, index: int) -> None:
        self.base_formula = formula
        self.axis = axis
        self.index = index

    def _compute_free_variables(self) -> set[str]:
        return self.base_formula.free_variables

    def _compute_free_structural_variables(self) -> set[str]:
        return self.base_formula.free_structural_variables - {self.axis}

    def __repr__(self) -> str:
        return f"({self.base_formula})[{self.axis}={self.index}]"


# ---------------------------------------------------------------------------
# Atomic formula
# ---------------------------------------------------------------------------

class AtomicFormula(Formula):
    """P(t1, …, tn) — a predicate applied to a sequence of terms.

    ``symbol`` is a string name before signature resolution, and a
    :class:`~sltn.signature.symbols.PredicateSymbol` object after.
    """

    def __init__(
        self,
        symbol: str | PredicateSymbol,
        *args: Term,
        structural_axes: Sequence[str] | None = None,
    ) -> None:
        self.symbol          = symbol
        self.args: tuple[Term, ...] = args
        self.structural_axes: list[str] | None = list(structural_axes) if structural_axes is not None else None

    @property
    def pred_name(self) -> str:
        return self.symbol if isinstance(self.symbol, str) else self.symbol.name

    def _compute_free_variables(self) -> set[str]:
        fvs: set[str] = set()
        for arg in self.args:
            fvs.update(arg.free_variables)
        return fvs

    def _compute_free_structural_variables(self) -> set[str]:
        return set(self._axis_dimensions())

    def _axis_dimensions(self) -> dict:
        if isinstance(self.symbol, (PredicateSymbol, FunctionSymbol)):
            return _call_axis_dims(self.symbol, self.args, self.structural_axes)
        axes: dict = {}
        for arg in self.args:
            axes.update(arg._axis_dimensions())
        return axes

    def __repr__(self) -> str:
        args_str = ", ".join(map(str, self.args))
        if self.structural_axes is not None:
            output_dims = self.structural_axes
        elif isinstance(self.symbol, PredicateSymbol) and self.symbol.output_dims:
            output_dims = [d.name for d in self.symbol.output_dims]
        else:
            output_dims = []
        if output_dims:
            return f"{self.pred_name}({args_str})[{', '.join(output_dims)}]"
        return f"{self.pred_name}({args_str})"



# ---------------------------------------------------------------------------
# Structural relation
# ---------------------------------------------------------------------------

class StructuralRelation(Formula):
    """A structural predicate applied to structural variable names.

    Syntax: ``next(t, t1)`` where ``next`` is a
    :class:`~sltn.signature.StructuralRelationSymbol`.  Produces an
    :class:`~sltn.interpretation.core.tensor.Tensor` with only structural
    axes and a boolean domain value — no regular variable axes.

    ``symbol`` is a string name before signature resolution, and a
    :class:`~sltn.signature.StructuralRelationSymbol` object after.
    """

    def __init__(self, symbol: str | StructuralRelationSymbol, *var_names: str) -> None:
        self.symbol = symbol
        self.var_names: tuple[str, ...] = var_names

    @property
    def pred_name(self) -> str:
        return self.symbol if isinstance(self.symbol, str) else self.symbol.name

    def _compute_free_variables(self) -> set[str]:
        return set()  # structural relations have no regular (individual) variable axes

    def _compute_free_structural_variables(self) -> set[str]:
        return set(self.var_names)

    def __repr__(self) -> str:
        return f"{self.pred_name}({', '.join(self.var_names)})"


# ---------------------------------------------------------------------------
# Connectives
# ---------------------------------------------------------------------------

def _binary_connective(class_name: str, op_attr: str, repr_fmt: str, doc: str = "") -> type:
    """Return a concrete binary Formula class for a given connective."""
    def __init__(self, left: Formula, right: Formula) -> None:
        self.left  = left
        self.right = right

    def _compute_free_variables(self) -> set[str]:
        return self.left.free_variables | self.right.free_variables

    def _compute_free_structural_variables(self) -> set[str]:
        return self.left.free_structural_variables | self.right.free_structural_variables

    def __repr__(self) -> str:
        def _fmt(f):
            return f"({f})" if isinstance(f, _BoundFormula) else str(f)
        return repr_fmt.format(_fmt(self.left), _fmt(self.right))

    cls = type(class_name, (Formula,), {
        "__doc__":                             doc,
        "op_attr":                            op_attr,
        "__init__":                           __init__,
        "_compute_free_variables":            _compute_free_variables,
        "_compute_free_structural_variables": _compute_free_structural_variables,
        "__repr__":                           __repr__,
    })
    return cls


And = _binary_connective(
    "And", "And", "({} & {})",
    doc="Conjunction (A ∧ B) of two formulas.",
)
Or = _binary_connective(
    "Or", "Or", "({} ∨ {})",
    doc="Disjunction (A ∨ B) of two formulas.",
)
Implies = _binary_connective(
    "Implies", "Implies", "({} → {})",
    doc="Implication (A → B) of two formulas.",
)
Equivalent = _binary_connective(
    "Equivalent", "Equiv", "({} ↔ {})",
    doc="Equivalence (A ↔ B) of two formulas.",
)


class Not(Formula):
    def __init__(self, operand: Formula) -> None:
        self.operand = operand

    def _compute_free_variables(self) -> set[str]:
        return self.operand.free_variables

    def _compute_free_structural_variables(self) -> set[str]:
        return self.operand.free_structural_variables

    def __repr__(self) -> str:
        return f"¬{self.operand}"


# ---------------------------------------------------------------------------
# Quantifiers
# ---------------------------------------------------------------------------

class _BoundFormula(Formula, ABC):
    """Base for quantified formulas. Subclasses set ``quantifier`` and ``symbol``."""

    quantifier: str   # "forall" or "exists"
    symbol: str       # "∀" or "∃"

    def __init__(
        self,
        var_groups: list[str | list[str]],
        formula: Formula,
        guard: Formula | None = None,
    ) -> None:
        self._var_groups: list[list[str]] = [
            [g] if isinstance(g, str) else list(g) for g in var_groups
        ]
        self.formula = formula
        self.guard: Formula | None = guard

    @property
    def var_names(self) -> list[str]:
        return [name for group in self._var_groups for name in group]

    def _compute_free_variables(self) -> set[str]:
        guard_fvs = self.guard.free_variables if self.guard else set()
        return (self.formula.free_variables | guard_fvs) - set(self.var_names)

    def _compute_free_structural_variables(self) -> set[str]:
        guard_fvs = self.guard.free_structural_variables if self.guard else set()
        return (self.formula.free_structural_variables | guard_fvs) - set(self.var_names)

    def _render_vars(self) -> str:
        parts = []
        for group in self._var_groups:
            parts.append(
                group[0] if len(group) == 1 else f"({', '.join(group)})"
            )
        return ", ".join(parts)

    def __repr__(self) -> str:
        guard_str = f" | {self.guard}" if self.guard else ""
        return f"{self.symbol} {self._render_vars()}{guard_str}: {self.formula}"


class Forall(_BoundFormula):
    """∀ x1, …, (xi, xj), …, xn [axis, …] | guard: φ"""
    quantifier = "forall"
    symbol     = "∀"


class Exists(_BoundFormula):
    """∃ x1, …, (xi, xj), …, xn [axis, …] | guard: φ"""
    quantifier = "exists"
    symbol     = "∃"


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------

class KB(Formula):
    """{ φ1, …, φn } — flat set of axiom clauses.

    Every clause must be a closed :class:`Formula` (see :meth:`check_clauses`).
    ``validate=False`` skips the check; the parser uses it because it builds the
    KB before names are resolved, and the parse pipeline validates it afterwards.
    """

    def __init__(self, clauses: Sequence[Formula | KB], *, validate: bool = True) -> None:
        flat: list[Formula] = []
        for c in clauses:
            if isinstance(c, KB):
                flat.extend(c.clauses)
            elif isinstance(c, Formula):
                flat.append(c)
            else:
                raise TypeError(f"KB clauses must be Formula objects, got {type(c).__name__}: {c!r}")
        self.clauses: list[Formula] = flat
        if validate:
            self.check_clauses()

    def check_clauses(self) -> None:
        """Raise ``ValueError`` if a clause is not closed (has free variables).

        An open clause would otherwise be silently averaged over its free axes by SatAgg.
        """
        for clause in self.clauses:
            if not clause.is_clause():
                free = clause.free_variables | clause.free_structural_variables
                raise ValueError(
                    f"KB requires closed formulas (no free variables), "
                    f"but found free {free!r} in: {clause!r}"
                )

    def _compute_free_variables(self) -> set[str]:
        fvs: set[str] = set()
        for clause in self.clauses:
            fvs.update(clause.free_variables)
        return fvs

    def _compute_free_structural_variables(self) -> set[str]:
        fvs: set[str] = set()
        for clause in self.clauses:
            fvs.update(clause.free_structural_variables)
        return fvs

    def __add__(self, other: Formula | KB) -> KB:
        if not isinstance(other, Formula):
            return NotImplemented
        if isinstance(other, KB):
            return KB(self.clauses + other.clauses)
        return KB(self.clauses + [other])

    def __radd__(self, other: Formula | KB) -> KB:
        if not isinstance(other, Formula):
            return NotImplemented
        if isinstance(other, KB):
            return KB(other.clauses + self.clauses)
        return KB([other] + self.clauses)

    def __repr__(self) -> str:
        return "{ " + ", ".join(map(str, self.clauses)) + " }"
