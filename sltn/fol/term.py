"""
sltn.fol.term — Syntactic AST nodes for FOL terms.

Terms are pure data structures describing the syntax of a formula.
Evaluation is handled exclusively by :class:`~sltn.interpretation.Interpretation`.

Terms are:
  - :class:`Variable`, :class:`Constant`, :class:`StructuralVariable` — resolved references
  - :class:`_CallTerm`               — unresolved ``f(t₁,…,tₙ)`` produced by the parser; promoted to
                                      :class:`FunctionApplication` or :class:`~sltn.fol.formula.AtomicFormula`
                                      during symbol resolution.
  - :class:`FunctionApplication` — a resolved function application (post-resolution only)
  - :class:`AxisSelect` — a term projected onto a single structural index (``t[axis=k]``)

String parsing for terms lives in :mod:`sltn.parse.parser`.
"""
from __future__ import annotations

__all__ = [
    "Term",
    "Variable",
    "Constant",
    "StructuralVariable",
    "_CallTerm",
    "FunctionApplication",
    "TermAxisSelect",
]

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sltn.signature.symbols import ConstantSymbol, VariableSymbol, StructuralVariableSymbol, FunctionSymbol


# ---------------------------------------------------------------------------
# Term Base Class
# ---------------------------------------------------------------------------

class Term(ABC):
    """Abstract syntax node for a FOL term."""

    # free variables
    _free_variables: set[str] | None = None

    @property
    def free_variables(self) -> set[str]:
        """The set of free variable names in this term."""
        if self._free_variables is not None:
            return self._free_variables
        return self._compute_free_variables()

    def _compute_free_variables(self) -> set[str]:
        return set()

    # free structural variables
    _free_structural_variables: set[str] | None = None

    @property
    def free_structural_variables(self) -> set[str]:
        """The set of free structural variables in this term."""
        if self._free_structural_variables is not None:
            return self._free_structural_variables
        return self._compute_free_structural_variables()

    def _compute_free_structural_variables(self) -> set[str]:
        return set()


# ---------------------------------------------------------------------------
# Name Term (unresolved identifier from the parser)
# ---------------------------------------------------------------------------

class _NameTerm(Term):
    """Unresolved identifier from the grammar (``str`` only).

    :meth:`~sltn.signature.Signature.parse` rewrites instances in-place into
    :class:`Variable`, :class:`Constant`, or :class:`StructuralVariable`.
    """

    symbol: str
    axis_annotations: list[str]

    def __init__(self, symbol: str, axis_annotations: list[str] | None = None) -> None:
        if not isinstance(symbol, str):
            raise TypeError(f"_NameTerm expects a str before resolution; got {type(symbol).__name__}.")
        self.symbol = symbol
        self.axis_annotations = list(axis_annotations or [])

    @property
    def name(self) -> str:
        return self.symbol

    def _compute_free_variables(self) -> set[str]:
        return {self.name}

    def _compute_free_structural_variables(self) -> set[str]:
        return set()

    def __repr__(self) -> str:
        if self.axis_annotations:
            return f"{self.symbol}[{', '.join(self.axis_annotations)}]"
        return self.symbol


# ---------------------------------------------------------------------------
# Variable Term
# ---------------------------------------------------------------------------

class Variable(Term):
    """Reference to a :class:`~sltn.signature.symbols.VariableSymbol` (after resolution)."""

    symbol: VariableSymbol
    axis_annotations: list[str]

    def __init__(self, symbol: VariableSymbol, axis_annotations: list[str] | None = None) -> None:
        self.symbol = symbol
        self.axis_annotations = list(axis_annotations or [])

    @property
    def name(self) -> str:
        return self.symbol.name

    def _compute_free_variables(self) -> set[str]:
        return {self.name}

    def _compute_free_structural_variables(self) -> set[str]:
        if self.axis_annotations:
            return set(self.axis_annotations)
        return {d.name for d in self.symbol.structural_variables}

    def __repr__(self) -> str:
        if self.axis_annotations:
            return f"{self.symbol.name}[{', '.join(self.axis_annotations)}]"
        if self.symbol.structural_variables:
            dims_str = ", ".join(d.name for d in self.symbol.structural_variables)
            return f"{self.symbol.name}[{dims_str}]"
        return self.name


# ---------------------------------------------------------------------------
# Constant Term
# ---------------------------------------------------------------------------

class Constant(Term):
    """Reference to a :class:`~sltn.signature.symbols.ConstantSymbol` (after resolution)."""

    symbol: ConstantSymbol
    axis_annotations: list[str]

    def __init__(self, symbol: ConstantSymbol, axis_annotations: list[str] | None = None) -> None:
        self.symbol = symbol
        self.axis_annotations = list(axis_annotations or [])

    @property
    def name(self) -> str:
        return self.symbol.name

    def _compute_free_variables(self) -> set[str]:
        return set()

    def _compute_free_structural_variables(self) -> set[str]:
        return {d.name for d in self.symbol.structural_variables}

    def __repr__(self) -> str:
        if self.axis_annotations:
            return f"{self.symbol.name}[{', '.join(self.axis_annotations)}]"
        return self.name

# ---------------------------------------------------------------------------
# Structural Variable Term
# ---------------------------------------------------------------------------

class StructuralVariable(Term):
    """Reference to a :class:`~sltn.signature.symbols.StructuralVariableSymbol` (after resolution)."""

    symbol: StructuralVariableSymbol
    axis_annotations: list[str]

    def __init__(self, symbol: StructuralVariableSymbol, axis_annotations: list[str] | None = None) -> None:
        self.symbol = symbol
        self.axis_annotations = list(axis_annotations or [])

    @property
    def name(self) -> str:
        return self.symbol.name

    def _compute_free_variables(self) -> set[str]:
        return set()

    def _compute_free_structural_variables(self) -> set[str]:
        return {self.symbol.name}

    def __repr__(self) -> str:
        if self.axis_annotations:
            return f"{self.symbol.name}[{', '.join(self.axis_annotations)}]"
        return self.name


# ---------------------------------------------------------------------------
# Function Application Term
# ---------------------------------------------------------------------------

class _CallTerm(Term):
    """Unresolved call ``f(t₁, …, tₙ)[out_axes]`` — produced by the parser.

    ``symbol`` is always a plain string name.  The parse pipeline's
    :func:`~sltn.parse.pipeline._resolve_symbols` step promotes every
    ``_CallTerm`` to either a :class:`FunctionApplication` (when *symbol*
    is a ``FunctionSymbol``) or an :class:`~sltn.fol.formula.AtomicFormula`
    (when it names a ``PredicateSymbol``).

    ``output_axes`` — optional list of explicit output structural axis names from
    the ``f(args)[t, s]`` annotation syntax.
    """

    def __init__(
        self,
        symbol: str,
        args: list,
        output_axes: list[str] | None = None,
    ) -> None:
        self.symbol      = symbol
        self.args        = args
        self.output_axes = output_axes

    @property
    def name(self) -> str:
        return self.symbol

    def _compute_free_variables(self) -> set[str]:
        fvs: set[str] = set()
        for arg in self.args:
            fvs.update(arg.free_variables)
        return fvs

    def _compute_free_structural_variables(self) -> set[str]:
        dims: set[str] = set()
        for arg in self.args:
            dims.update(arg.free_structural_variables)
        return dims

    def __repr__(self) -> str:
        args_str = ', '.join(map(str, self.args))
        if self.output_axes is not None:
            return f"{self.symbol}({args_str})[{', '.join(self.output_axes)}]"
        return f"{self.symbol}({args_str})"


# ---------------------------------------------------------------------------
# FunctionApplication (post-resolution)
# ---------------------------------------------------------------------------

class FunctionApplication(Term):
    """A resolved function application in term position: f(t₁, …, tₙ)[out_axes].

    ``symbol`` is always a :class:`~sltn.signature.symbols.FunctionSymbol` object (set
    during symbol resolution — see :func:`~sltn.parse.pipeline._resolve_symbols`).

    ``output_axes`` — optional list of explicit output structural axis names from
    the ``f(args)[t, s]`` annotation syntax.  ``None`` means use the declared
    ``output_dims`` from the signature; an explicit list (possibly empty)
    overrides them.
    """

    def __init__(
        self,
        symbol: FunctionSymbol,
        args: list,
        output_axes: list[str] | None = None,
    ) -> None:
        self.symbol      = symbol
        self.args        = args
        self.output_axes = output_axes

    @property
    def name(self) -> str:
        return self.symbol.name

    def _compute_free_variables(self) -> set[str]:
        fvs: set[str] = set()
        for arg in self.args:
            fvs.update(arg.free_variables)
        return fvs

    def _compute_free_structural_variables(self) -> set[str]:
        arg_dims: set[str] = set()
        for arg in self.args:
            arg_dims.update(arg.free_structural_variables)
        in_dim_names = {d.name for d in self.symbol.input_dims}
        consumed = {alias for alias in arg_dims if alias in in_dim_names}
        out_names = set(self.output_axes) if self.output_axes is not None else {d.name for d in self.symbol.output_dims}
        return (arg_dims - consumed) | out_names

    def __repr__(self) -> str:
        args_str = ', '.join(map(str, self.args))
        if self.output_axes is not None:
            return f"{self.name}({args_str})[{', '.join(self.output_axes)}]"
        return f"{self.name}({args_str})"


# ---------------------------------------------------------------------------
# TermAxisSelect
# ---------------------------------------------------------------------------


class TermAxisSelect(Term):
    """A term that selects a single index along a named structural axis.

    Syntax: ``base_term[axis=index]``
    """

    def __init__(self, base_term: Term, index: int, axis: str) -> None:
        self.base_term = base_term
        self.index     = index
        self.axis      = axis

    def _compute_free_variables(self) -> set[str]:
        return self.base_term.free_variables

    def _compute_free_structural_variables(self) -> set[str]:
        return self.base_term.free_structural_variables - {self.axis}

    def __repr__(self) -> str:
        return f"{self.base_term}[{self.axis}={self.index}]"
