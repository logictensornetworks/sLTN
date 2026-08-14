"""sltn.signature.signature — Signature registry.

The :class:`Signature` registers all symbols (sorts, dimensions, constants,
variables, functions, predicates) and provides ``sig.parse(str)`` for formula
parsing with symbol-identity resolution.

Symbol class definitions live in :mod:`sltn.signature.symbols`.
"""
from __future__ import annotations

__all__ = [
    "Signature",
]

from typing import Optional, Sequence, TYPE_CHECKING

from sltn.signature.symbols import (
    Symbol, Definition, Sort, Dimension, ConstantSymbol, VariableSymbol,
    FunctionSymbol, PredicateSymbol,
    StructuralVariableSymbol, StructuralRelationSymbol,
)

if TYPE_CHECKING:
    from sltn.fol.formula import Formula
    from sltn.fol.term import Term

# ---------------------------------------------------------------------------
# Reserved names — cannot be used as symbol identifiers
# ---------------------------------------------------------------------------

#: Names that the parser tokenises as keywords or structural punctuation.
#: Declaring a symbol with one of these names would silently break parsing.
#: Note: infix operators that merely *start* with "=" (e.g. "=f", "==") are
#: fine — the check uses exact equality.
_RESERVED_NAMES: frozenset[str] = frozenset({
    # ASCII keyword spellings
    "forall", "exists",
    "struct_forall", "struct_exists",
    "not", "and", "or", "implies", "iff",
    # Unicode single-char alternatives
    "∀", "∃", "¬", "∧", "&", "∨", "→", "↔",
    # axis-select operator
    "=",
})

# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

class Signature:
    """
    A signature Σ = (S, D, F, P, C, X) consisting of declared:

      - **S** — sorts (``Sort``)
      - **D** — structural dimensions (``Dimension``)
      - **C** — constants (``ConstantSymbol``)
      - **X** — variables (``VariableSymbol``)
      - **F** — function symbols (``FunctionSymbol``)
      - **P** — predicate symbols (``PredicateSymbol``)

    The Signature defines the *vocabulary* of the logic.  To assign concrete
    values (tensors, nn.Modules), ground them through an
    :class:`~sltn.interpretation.Interpretation` object.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._sorts:       dict[str, Sort]      = {}
        self._dimensions:  dict[str, Dimension] = {}
        self._constants:   dict[str, ConstantSymbol]  = {}
        self._variables:   dict[str, VariableSymbol]  = {}
        self._functions:   dict[str, FunctionSymbol]  = {}
        self._predicates:  dict[str, PredicateSymbol] = {}
        self._structural_variables:  dict[str, StructuralVariableSymbol]  = {}
        self._structural_relations: dict[str, StructuralRelationSymbol] = {}
        self._variables_aliases:        dict[str, str] = {}
        self._structural_variables_aliases: dict[str, str] = {}
        self._definitions: dict[str, Definition] = {}

        # Pre-declare standard sorts; their tensor-level Type is bound by Interpretation.
        self.sort("Bool")
        self.sort("Real")

    # ------------------------------------------------------------------
    # Private name-resolution helpers
    # ------------------------------------------------------------------

    def _resolve_sort(self, sort: str | Sort) -> Sort:
        if isinstance(sort, Sort):
            return sort
        if sort not in self._sorts:
            raise KeyError(f"Sort '{sort}' not declared in '{self.name}'.")
        return self._sorts[sort]

    def _resolve_sorts(self, sorts: Optional[Sequence[str | Sort]]) -> list[Sort]:
        return [self._resolve_sort(s) for s in sorts] if sorts else []

    def _resolve_dim(self, dim: str | Dimension) -> Dimension:
        if isinstance(dim, Dimension):
            return dim
        if dim not in self._dimensions:
            raise KeyError(f"Dimension '{dim}' not declared in '{self.name}'.")
        return self._dimensions[dim]

    def _resolve_dims(self, dims: Optional[Sequence[str | Dimension]]) -> list[Dimension]:
        return [self._resolve_dim(d) for d in dims] if dims else []

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def _check_name(self, name: str) -> None:
        """Raise if *name* is a parser-reserved keyword or operator."""
        if name in _RESERVED_NAMES:
            raise ValueError(
                f"'{name}' is a reserved name in the sLTN grammar and cannot "
                f"be used as a symbol identifier. Reserved: {sorted(_RESERVED_NAMES)}"
            )

    def sort(self, name: str) -> Sort:
        self._check_name(name)
        if name in self._sorts:
            raise ValueError(f"Sort '{name}' already declared.")
        s = Sort(name=name)
        self._sorts[name] = s
        return s

    def dimension(
        self,
        name: str,
        sort: Optional[str | Sort] = None,
        labels: Optional[list] = None,
    ) -> Dimension:
        self._check_name(name)
        if name in self._dimensions:
            raise ValueError(f"Dimension '{name}' already declared.")
        resolved_sort = self._resolve_sort(sort) if sort is not None else None
        d = Dimension(name=name, sort=resolved_sort, labels=labels)
        self._dimensions[name] = d
        return d

    def constant(
        self,
        name: str,
        sort: str | Sort,
        dims: Optional[Sequence[str | Dimension]] = None,
    ) -> ConstantSymbol:
        self._check_name(name)
        if name in self._constants:
            raise ValueError(f"ConstantSymbol '{name}' already declared.")
        c = ConstantSymbol(name=name, sort=self._resolve_sort(sort), dims=self._resolve_dims(dims))
        self._constants[name] = c
        return c

    def variable(
        self,
        name: str,
        sort: str | Sort,
        dims: Optional[Sequence[str | Dimension]] = None,
    ) -> VariableSymbol:
        self._check_name(name)
        if name in self._variables:
            raise ValueError(f"VariableSymbol '{name}' already declared.")
        v = VariableSymbol(name=name, sort=self._resolve_sort(sort), dims=self._resolve_dims(dims))
        self._variables[name] = v
        return v

    def function(
        self,
        name: str,
        input_sorts: list[str | Sort],
        output_sort: str | Sort,
        input_dims: Optional[Sequence[str | Dimension]] = None,
        output_dims: Optional[Sequence[str | Dimension]] = None,
        **kwargs,
    ) -> FunctionSymbol:
        self._check_name(name)
        if name in self._functions:
            raise ValueError(f"FunctionSymbol '{name}' already declared.")
        f = FunctionSymbol(
            name=name,
            input_sorts=self._resolve_sorts(input_sorts),
            output_sort=self._resolve_sort(output_sort),
            input_dims=self._resolve_dims(input_dims),
            output_dims=self._resolve_dims(output_dims),
            **kwargs,
        )
        self._functions[name] = f
        return f

    def predicate(
        self,
        name: str,
        input_sorts: list[str | Sort],
        input_dims: Optional[Sequence[str | Dimension]] = None,
        output_dims: Optional[Sequence[str | Dimension]] = None,
        **kwargs,
    ) -> PredicateSymbol:
        self._check_name(name)
        if name in self._predicates:
            raise ValueError(f"PredicateSymbol '{name}' already declared.")
        p = PredicateSymbol(
            name=name,
            input_sorts=self._resolve_sorts(input_sorts),
            input_dims=self._resolve_dims(input_dims),
            output_dims=self._resolve_dims(output_dims),
            **kwargs,
        )
        self._predicates[name] = p
        return p

    def structural_variable(
        self,
        name_or_dim: str | Dimension,
        dimension: Optional[str | Dimension] = None,
    ) -> StructuralVariableSymbol:
        """Declare a structural variable ranging over *dimension*.

        *name_or_dim* can be a plain string (name), a :class:`Dimension` object,
        or a dimension name string used as both name-source and dimension lookup
        when *dimension* is also supplied.
        """
        if isinstance(name_or_dim, Dimension):
            dim = name_or_dim
            n   = sum(1 for sv in self._structural_variables.values() if sv.dimension is dim)
            name = f"{dim.name}_{n}"
        else:
            name = name_or_dim
            self._check_name(name)
            if dimension is None:
                raise ValueError(
                    "dimension must be provided when name_or_dim is a string."
                )
            dim = self._resolve_dim(dimension)
        if name in self._structural_variables:
            raise ValueError(f"StructuralVariableSymbol '{name}' already declared.")
        sv = StructuralVariableSymbol(name=name, dimension=dim)
        self._structural_variables[name] = sv
        return sv

    def structural_relation(
        self,
        name: str,
        dimensions: Sequence[str | Dimension],
    ) -> StructuralRelationSymbol:
        self._check_name(name)
        if name in self._structural_relations:
            raise ValueError(f"StructuralRelationSymbol '{name}' already declared.")
        sp = StructuralRelationSymbol(name=name, dimensions=[self._resolve_dim(d) for d in dimensions])
        self._structural_relations[name] = sp
        return sp

    def define(
        self,
        name: str,
        arg_names: list[str],
        formula_body: str,
        infix: bool = False,
        input_sorts: Optional[Sequence[str | Sort]] = None,
        output_sort: Optional[str | Sort] = None,
        input_dims: Optional[Sequence[str | Dimension]] = None,
        output_dims: Optional[Sequence[str | Dimension]] = None,
    ) -> None:
        """Register a derived predicate as a syntactic abbreviation (rewriting rule).

        When :meth:`parse` encounters ``name(t1, …, tn)`` in a formula string,
        it replaces it with *formula_body* with *arg_names[i]* substituted by
        *ti* — at parse time, before symbol resolution or evaluation.
        """
        
        if input_sorts is not None and len(input_sorts) != len(arg_names):
            raise ValueError(
                f"Definition '{name}' declares {len(input_sorts)} input_sorts "
                f"but has {len(arg_names)} parameter(s)."
            )

        self._definitions[name] = Definition(
            name=name,
            params=list(arg_names),
            body=formula_body,
            infix=infix,
            input_sorts=self._resolve_sorts(input_sorts),
            output_sort=self._resolve_sort(output_sort) if output_sort is not None else None,
            input_dims=self._resolve_dims(input_dims),
            output_dims=self._resolve_dims(output_dims),
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _get_from(self, store: dict, name: str, kind: str):
        if name not in store:
            raise KeyError(f"{kind} '{name}' not declared in '{self.name}'.")
        return store[name]

    def get_sort(self, name: str) -> Sort:
        return self._get_from(self._sorts, name, "Sort")

    def get_dimension(self, name: str) -> Dimension:
        return self._get_from(self._dimensions, name, "Dimension")

    def get_constant(self, name: str) -> ConstantSymbol:
        return self._get_from(self._constants, name, "ConstantSymbol")

    def get_variable(self, name: str) -> VariableSymbol:
        return self._get_from(self._variables, name, "VariableSymbol")

    def get_function(self, name: str) -> FunctionSymbol:
        return self._get_from(self._functions, name, "FunctionSymbol")

    def get_predicate(self, name: str) -> PredicateSymbol:
        return self._get_from(self._predicates, name, "PredicateSymbol")

    def get_structural_variable(self, name: str) -> StructuralVariableSymbol:
        return self._get_from(self._structural_variables, name, "StructuralVariableSymbol")

    def get_structural_relation(self, name: str) -> StructuralRelationSymbol:
        return self._get_from(self._structural_relations, name, "StructuralRelationSymbol")

    def resolve_structural_axis(self, name: str) -> str:
        """Return the canonical ``Dimension`` name for a structural axis identifier.

        *name* can be a dimension name (returned as-is), the name of a
        :class:`StructuralVariableSymbol` (declared explicitly or auto-created
        by the parser, base or alias — e.g. ``t``, ``t1``, ``t'``), which is
        resolved to the name of the ``Dimension`` it ranges over.

        This lets structural-aggregator groundings be declared once per
        dimension and inherited by every variable that ranges over it,
        regardless of the variable's own name. Unknown names are returned
        unchanged.
        """
        if name in self._dimensions:
            return name
        sv = self._structural_variables.get(name)
        if sv is None:
            base = self._structural_variables_aliases.get(name)
            if base is not None:
                sv = self._structural_variables.get(base)
        return sv.dimension.name if sv is not None else name

    def get(self, name: str) -> Symbol:
        """Look up a symbol by name across all namespaces (first match wins)."""
        for store in (
            self._sorts, self._dimensions, self._constants,
            self._variables, self._functions, self._predicates,
            self._structural_variables, self._structural_relations,
        ):
            if name in store:
                return store[name]
        raise KeyError(f"Symbol '{name}' not found in signature '{self.name}'.")

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(self, formula_string: str) -> Formula | Term:
        """Parse *formula_string* as a formula or term against this signature."""
        from sltn.parse.pipeline import parse
        return parse(formula_string, self)

    def parse_term(self, term_string: str) -> Term:
        """Parse *term_string* and resolve symbol names against this signature."""
        from sltn.parse.pipeline import parse_term
        return parse_term(term_string, self)
    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines: list[str] = []

        def _section(title: str, items, key=lambda x: x.sig_line()):
            if items:
                lines.append(f"{title}:")
                lines.extend(f"  {key(item)}" for item in items)

        _section("Sorts",                 self._sorts.values())
        _section("Dimensions",            self._dimensions.values())
        _section("Structural variables",  self._structural_variables.values())
        _section("Structural predicates", self._structural_relations.values())
        _section("Constants",             self._constants.values())
        _section("Variables",             self._variables.values())
        _section("Functions",             self._functions.values())
        _section("Predicates",            self._predicates.values())
        _section("Definitions",           self._definitions.values())

        body = "\n  ".join(lines) if lines else "(empty)"
        return f"Signature({self.name!r}):\n  {body}"
