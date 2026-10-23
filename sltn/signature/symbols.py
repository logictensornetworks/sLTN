"""
sltn.signature.symbols — Symbol declarations for first-order logic signatures.

A :class:`Symbol` is a purely syntactic entity — it carries name and type
metadata but no concrete tensor values.  Concrete values are bound at
evaluation time through an :class:`~sltn.interpretation.Interpretation`.

The corresponding AST *term* nodes used inside parsed formulas live in
:mod:`sltn.fol.term` (:class:`~sltn.fol.term.Constant`,
:class:`~sltn.fol.term.Variable`, …).

Vocabulary
----------
- :class:`Symbol`                    — abstract base for all declared entities
- :class:`Sort`                      — a domain type (e.g. 'Animal', 'Image')
- :class:`Dimension`                 — a structural/world axis (e.g. 'time', 'nodes')
- :class:`ConstantSymbol`            — declaration of an individual constant
- :class:`VariableSymbol`            — declaration of a first-order variable
- :class:`FunctionSymbol`            — declaration of a function symbol
- :class:`PredicateSymbol`           — declaration of a predicate symbol
- :class:`StructuralVariableSymbol`  — declaration of an index variable over a Dimension
- :class:`StructuralRelationSymbol` — declaration of a boolean/fuzzy relation over Dimensions
"""
from __future__ import annotations

__all__ = [
    "Symbol",
    "Definition",
    "Sort",
    "Dimension",
    "ConstantSymbol",
    "VariableSymbol",
    "FunctionSymbol",
    "PredicateSymbol",
    "StructuralVariableSymbol",
    "StructuralRelationSymbol",
]

from dataclasses import dataclass, field
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Base symbol
# ---------------------------------------------------------------------------

class Symbol:
    """Abstract base for a symbol, which is every declared entity in a Signature."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


# ---------------------------------------------------------------------------
# Definition
# ---------------------------------------------------------------------------

@dataclass
class Definition:
    """Used to define functions and predicates by means of a string formula/term."""
    name: str
    params: list[str]
    body: str
    infix: bool = False
    input_sorts: list[Sort] = field(default_factory=list)
    output_sort: Optional[Sort] = None
    input_dims: list[Dimension] = field(default_factory=list)
    output_dims: list[Dimension] = field(default_factory=list)

    def sig_line(self) -> str:
        return f"{self.name}: {self.body} with arguments {self.params}"


# ---------------------------------------------------------------------------
# Sort
# ---------------------------------------------------------------------------

class Sort(Symbol):
    """A domain (sort) declaration.

    A ``Sort`` is a pure logical annotation.  The associated tensor type
    (shape of individuals) is assigned via :class:`~sltn.interpretation.Interpretation`
    at evaluation time (e.g. ``interp[S] = Real``).
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def sig_line(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Sort({self.name!r})"


# ---------------------------------------------------------------------------
# Dimension
# ---------------------------------------------------------------------------

class Dimension(Symbol):
    """A structural/world dimension symbol (e.g. 'time', 'event').

    A ``Dimension`` is a pure logical label; its concrete size and values are
    determined at runtime by the tensors bound through an
    :class:`~sltn.interpretation.Interpretation`.

    The optional *labels* list associates a human-readable label with each
    index along the dimension (e.g. ``["EventA", "EventB"]`` or
    ``[4, 5, 6, ..., 54]`` for time instants). Labels are only used for display.

    Note: the *tensor-level* axis tag is :class:`~sltn.interpretation.core.axis.Axis`.
    """

    def __init__(
        self,
        name: str,
        sort: Optional[Sort] = None,
        labels: Optional[list] = None,
    ) -> None:
        super().__init__(name)
        self.sort = sort
        self.labels: list | None = list(labels) if labels is not None else None

    def sig_line(self) -> str:
        if self.labels is not None:
            return f"{self.name}  [{', '.join(str(l) for l in self.labels)}]"
        return self.name

    def __repr__(self) -> str:
        parts = [f"{self.name!r}"]
        if self.sort is not None:
            parts.append(f"sort={self.sort.name!r}")
        if self.labels is not None:
            parts.append(f"labels={self.labels!r}")
        return f"Dimension({', '.join(parts)})"


# ---------------------------------------------------------------------------
# ConstantSymbol
# ---------------------------------------------------------------------------

class ConstantSymbol(Symbol):
    """An individual constant declaration.

    Parameters
    ----------
    name:   unique identifier in the signature
    sort:   the Sort this constant belongs to
    dims:   structural Dimension symbols used by this constant (can be empty)
    """

    def __init__(
        self,
        name: str,
        sort: Sort,
        dims: Optional[Sequence[Dimension]] = None,
    ) -> None:
        super().__init__(name)
        self.sort = sort
        self.structural_variables: list[Dimension] = list(dims or [])

    def sig_line(self) -> str:
        dims = f" [{', '.join(d.name for d in self.structural_variables)}]" if self.structural_variables else ""
        return f"{self.name}: {self.sort.name}{dims}"

    def __repr__(self) -> str:
        dims_str = (
            (", dims=[" + ", ".join(d.name for d in self.structural_variables) + "]")
            if self.structural_variables else ""
        )
        return f"ConstantSymbol({self.name!r}, sort={self.sort.name!r}{dims_str})"


# ---------------------------------------------------------------------------
# VariableSymbol
# ---------------------------------------------------------------------------

class VariableSymbol(Symbol):
    """A variable symbol declaration.

    A ``VariableSymbol`` is a *declaration* — it describes what sort and structural
    dimensions a variable ranges over.  Actual tensor data is bound at
    evaluation time via ``interp[name] = tensor``.
    """

    def __init__(
        self,
        name: str,
        sort: Sort,
        dims: Optional[Sequence[Dimension]] = None,
    ) -> None:
        super().__init__(name)
        self.sort = sort
        self.structural_variables: list[Dimension] = list(dims or [])

    def sig_line(self) -> str:
        dims = f" [{', '.join(d.name for d in self.structural_variables)}]" if self.structural_variables else ""
        return f"{self.name}: {self.sort.name}{dims}"

    def __repr__(self) -> str:
        dims_str = (
            (", dims=[" + ", ".join(d.name for d in self.structural_variables) + "]")
            if self.structural_variables else ""
        )
        return f"VariableSymbol({self.name!r}, sort={self.sort.name!r}{dims_str})"


# ---------------------------------------------------------------------------
# FunctionSymbol
# ---------------------------------------------------------------------------

class FunctionSymbol(Symbol):
    """A function symbol.

    Logical metadata
    ~~~~~~~~~~~~~~~~
    - ``input_sorts``  — ordered list of input Sort symbols
    - ``output_sort``  — output Sort symbol
    - ``input_dims``   — structural Dimension symbols consumed/preserved (d_in(f))
    - ``output_dims``  — structural Dimension symbols produced (d_out(f))

    Computational attributes
    ~~~~~~~~~~~~~~~~~~~~~~~~
    - ``infix``        — format name as ``a name b`` instead of ``name(a, b)``

    Bind a callable via :class:`~sltn.interpretation.Interpretation`:
    ``interp[f] = nn.Module_or_lambda``.
    """

    def __init__(
        self,
        name: str,
        input_sorts: Optional[list[Sort]] = None,
        output_sort: Optional[Sort] = None,
        input_dims: Optional[Sequence[Dimension]] = None,
        output_dims: Optional[Sequence[Dimension]] = None,
        infix: bool = False,
    ) -> None:
        super().__init__(name)
        self.input_sorts:  list[Sort]      = list(input_sorts or [])
        self.output_sort:  Optional[Sort]  = output_sort
        self.input_dims:   list[Dimension] = list(input_dims or [])
        self.output_dims:  list[Dimension] = list(output_dims or [])
        self.infix: bool = infix

    def sig_line(self) -> str:
        lhs = " x ".join(s.name for s in self.input_sorts) or "ε"
        rhs = self.output_sort.name if self.output_sort else "?"
        return f"{self.name}: {lhs} ⟶ {rhs}"

    def __repr__(self) -> str:
        in_sorts = " x ".join(s.name for s in self.input_sorts) or "ε"
        out_sort = self.output_sort.name if self.output_sort else "?"
        return f"FunctionSymbol({self.name!r}: {in_sorts} → {out_sort})"


# ---------------------------------------------------------------------------
# PredicateSymbol
# ---------------------------------------------------------------------------

class PredicateSymbol(Symbol):
    """A predicate symbol mapping to truth values in [0, 1].

    Logical metadata
    ~~~~~~~~~~~~~~~~
    - ``input_sorts`` — ordered list of input Sort symbols
    - ``input_dims``  — structural Dimension symbols consumed (d_in(p))
    - ``output_dims`` — structural Dimension symbols produced (d_out(p))

    Bind a callable via :class:`~sltn.interpretation.Interpretation`:
    ``interp[P] = nn.Module_or_lambda``.  The callable must return values in ``[0, 1]``.
    """

    def __init__(
        self,
        name: str,
        input_sorts: Optional[list[Sort]] = None,
        input_dims: Optional[Sequence[Dimension]] = None,
        output_dims: Optional[Sequence[Dimension]] = None,
        infix: bool = False,
    ) -> None:
        super().__init__(name)
        self.input_sorts: list[Sort]      = list(input_sorts or [])
        self.input_dims:  list[Dimension] = list(input_dims or [])
        self.output_dims: list[Dimension] = list(output_dims or [])
        self.infix: bool = infix

    def sig_line(self) -> str:
        lhs = " x ".join(s.name for s in self.input_sorts) or "ε"
        return f"{self.name}: {lhs} ⟶ Bool"

    def __repr__(self) -> str:
        in_sorts = " x ".join(s.name for s in self.input_sorts) or "ε"
        return f"PredicateSymbol({self.name!r}: {in_sorts} → Bool)"


# ---------------------------------------------------------------------------
# StructuralVariableSymbol
# ---------------------------------------------------------------------------

class StructuralVariableSymbol(Symbol):
    """A dimensional variable symbol (e.g. τ, α of dimension 'time').

    A ``StructuralVariableSymbol`` ranges over the indices of its associated :class:`Dimension`.
    In formulas ``∀ τ: φ`` quantifies over all time-points via the structural
    aggregation operator; ``∃ τ: φ`` likewise.
    """

    def __init__(self, name: str, dimension: Dimension) -> None:
        super().__init__(name)
        self.dimension = dimension

    def sig_line(self) -> str:
        return f"{self.name}: {self.dimension.name}"

    def __repr__(self) -> str:
        return f"StructuralVariableSymbol({self.name!r}, dimension={self.dimension.name!r})"


# ---------------------------------------------------------------------------
# StructuralRelationSymbol
# ---------------------------------------------------------------------------

class StructuralRelationSymbol(Symbol):
    """A structural relation symbol (e.g. 'next(time, time)').

    Encodes relations between world-indices such as temporal adjacency or
    causal ordering.  Bound to a concrete ``[0, 1]``-valued tensor via
    ``interp[sp] = tensor`` on an :class:`~sltn.interpretation.Interpretation`.
    """

    def __init__(self, name: str, dimensions: Sequence[Dimension]) -> None:
        super().__init__(name)
        self.structural_variables: list[Dimension] = list(dimensions)

    def sig_line(self) -> str:
        lhs = " x ".join(d.name for d in self.structural_variables) or "ε"
        return f"{self.name}: {lhs} ⟶ Bool"

    def __repr__(self) -> str:
        dims_str = " x ".join(d.name for d in self.structural_variables)
        return f"StructuralRelationSymbol({self.name!r}: {dims_str})"
