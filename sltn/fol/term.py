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


def _deduplicate_axis_names(dims: list) -> list[str]:
    """Return unique axis names for *dims*, appending ``0``, ``1``, … when
    the same :class:`Dimension` appears more than once (``[T, T]`` → ``['T0', 'T1']``).
    """
    seen: dict[str, int] = {}
    for d in dims:
        seen[d.name] = seen.get(d.name, 0) + 1
    counts: dict[str, int] = {}
    names: list[str] = []
    for d in dims:
        n = d.name
        if seen[n] > 1:
            idx = counts.get(n, 0)
            counts[n] = idx + 1
            names.append(f"{n}{idx}")
        else:
            names.append(n)
    return names


def _annotated_axis_dims(dims, annotations) -> dict:
    """Structural axes of ``x[a1, …]`` declared over *dims*, mapped to their Dimension.

    Mirrors the evaluator: repeated dimensions are named ``T0, T1, …`` (#26), then
    the first ``len(annotations)`` axes are renamed position by position.
    """
    names = _deduplicate_axis_names(dims)
    names[:len(annotations)] = annotations[:len(names)]
    return dict(zip(names, dims))


def _call_axis_dims(symbol, args, output_axes) -> dict:
    """Structural axes of ``f(args)[a1, …]``, mapped to their Dimension.

    The dimension of every axis comes from the symbols in the tree, so no
    signature is needed (#71). Argument axes over one of ``input_dims`` are
    consumed; ``a1, …`` name the ``output_dims`` in order (a name already on an
    argument passes that axis through).
    """
    arg_axes: dict = {}
    for arg in args:
        arg_axes.update(arg._axis_dimensions())
    in_names = {d.name for d in symbol.input_dims}
    kept = {n: d for n, d in arg_axes.items() if d is None or d.name not in in_names}
    names = output_axes if output_axes is not None else [d.name for d in symbol.output_dims]
    return kept | dict(zip(names, symbol.output_dims))


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
        return set(self._axis_dimensions())

    def _axis_dimensions(self) -> dict:
        """Free structural axis names of this term, mapped to their Dimension."""
        return {}


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

    def _axis_dimensions(self) -> dict:
        return _annotated_axis_dims(self.symbol.structural_variables, self.axis_annotations)

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

    def _axis_dimensions(self) -> dict:
        return _annotated_axis_dims(self.symbol.structural_variables, self.axis_annotations)

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

    def _axis_dimensions(self) -> dict:
        return {self.symbol.name: self.symbol.dimension}

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

    def _axis_dimensions(self) -> dict:
        axes: dict = {}
        for arg in self.args:
            axes.update(arg._axis_dimensions())
        return axes

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

    def _axis_dimensions(self) -> dict:
        return _call_axis_dims(self.symbol, self.args, self.output_axes)

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

    def _axis_dimensions(self) -> dict:
        axes = dict(self.base_term._axis_dimensions())
        axes.pop(self.axis, None)
        return axes

    def __repr__(self) -> str:
        return f"{self.base_term}[{self.axis}={self.index}]"
