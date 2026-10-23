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

import warnings
from contextlib import contextmanager
from types import MappingProxyType
from typing import Iterator, Optional, Sequence, TYPE_CHECKING

from sltn.signature.symbols import (
    Symbol, Definition, Sort, Dimension, ConstantSymbol, VariableSymbol,
    FunctionSymbol, PredicateSymbol,
    StructuralVariableSymbol, StructuralRelationSymbol,
)

from sltn.parse.tokenizer import RESERVED_NAMES as _RESERVED_NAMES, is_identifier, _base_name_of

if TYPE_CHECKING:
    from sltn.fol.formula import Formula
    from sltn.fol.term import Term

_GRAMMAR_CHARS = frozenset("()[]{},:|")

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
            if self._sorts.get(sort.name) is not sort:
                raise KeyError(f"Sort object '{sort.name}' is not registered in '{self.name}'.")
            return sort
        if sort not in self._sorts:
            raise KeyError(f"Sort '{sort}' not declared in '{self.name}'.")
        return self._sorts[sort]

    def _resolve_sorts(self, sorts: Optional[Sequence[str | Sort]]) -> list[Sort]:
        return [self._resolve_sort(s) for s in sorts] if sorts else []

    def _resolve_dim(self, dim: str | Dimension) -> Dimension:
        if isinstance(dim, Dimension):
            if self._dimensions.get(dim.name) is not dim:
                raise KeyError(f"Dimension object '{dim.name}' is not registered in '{self.name}'.")
            return dim
        if dim not in self._dimensions:
            raise KeyError(f"Dimension '{dim}' not declared in '{self.name}'.")
        return self._dimensions[dim]

    def _resolve_dims(self, dims: Optional[Sequence[str | Dimension]]) -> list[Dimension]:
        return [self._resolve_dim(d) for d in dims] if dims else []

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def _check_name(self, name: str, infix: bool = False) -> None:
        """Raise if *name* is reserved or can't be written in a formula."""
        if name in _RESERVED_NAMES:
            raise ValueError(
                f"'{name}' is a reserved name in the sLTN grammar and cannot "
                f"be used as a symbol identifier. Reserved: {sorted(_RESERVED_NAMES)}"
            )
        if not infix and not is_identifier(name):
            raise ValueError(
                f"Symbol name {name!r} is not an identifier (a letter or '_' "
                f"followed by letters, digits, '_' or \"'\")."
            )
        if infix and not is_identifier(name) and (
            not name or name[0].isalnum() or any(c.isspace() or c in _GRAMMAR_CHARS for c in name)
        ):
            raise ValueError(
                f"Infix symbol {name!r} must be an identifier or a run of symbols that "
                f"starts with a symbol and contains no spaces or {''.join(sorted(_GRAMMAR_CHARS))}."
            )

    @staticmethod
    def _warn_if_redeclared(store: dict, name: str, kind: str) -> None:
        if name in store:
            warnings.warn(
                f"{kind} '{name}' is redeclared: the new declaration replaces the old one. "
                f"Formulas parsed before keep the old symbol.",
                UserWarning, stacklevel=3,
            )

    def sort(self, name: str) -> Sort:
        self._check_name(name)
        self._warn_if_redeclared(self._sorts, name, "Sort")
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
        if name in self._structural_variables:
            raise ValueError(f"'{name}' is already a structural variable; a dimension needs a different name.")
        self._warn_if_redeclared(self._dimensions, name, "Dimension")
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
        self._warn_if_redeclared(self._constants, name, "ConstantSymbol")
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
        self._warn_if_redeclared(self._variables, name, "VariableSymbol")
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
        self._check_name(name, infix=kwargs.get("infix", False))
        self._warn_if_redeclared(self._functions, name, "FunctionSymbol")
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
        self._check_name(name, infix=kwargs.get("infix", False))
        self._warn_if_redeclared(self._predicates, name, "PredicateSymbol")
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
        name: str,
        dimension: str | Dimension,
    ) -> StructuralVariableSymbol:
        """Declare a structural variable *name* ranging over *dimension*."""
        if not isinstance(name, str):
            raise TypeError(
                f"structural_variable(name, dimension) expects a name string, got {type(name).__name__}."
            )
        self._check_name(name)
        if name in self._dimensions:
            raise ValueError(
                f"'{name}' is already a dimension, and dimension names can't name axes. "
                f"Use another name for the structural variable."
            )
        dim = self._resolve_dim(dimension)
        self._warn_if_redeclared(self._structural_variables, name, "StructuralVariableSymbol")
        sv = StructuralVariableSymbol(name=name, dimension=dim)
        self._structural_variables[name] = sv
        return sv

    def structural_relation(
        self,
        name: str,
        dimensions: Sequence[str | Dimension],
    ) -> StructuralRelationSymbol:
        self._check_name(name)
        self._warn_if_redeclared(self._structural_relations, name, "StructuralRelationSymbol")
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
        self._check_name(name, infix=infix)
        for p in arg_names:
            self._check_name(p)
        if len(set(arg_names)) != len(arg_names):
            raise ValueError(f"Definition '{name}' has duplicate parameter names: {list(arg_names)}.")
        self._warn_if_redeclared(self._definitions, name, "Definition")
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
        :class:`StructuralVariableSymbol` or of an alias of one (``t``, ``t1``,
        ``t'``), which is resolved to the name of the ``Dimension`` it ranges over.

        This lets structural-aggregator groundings be declared once per
        dimension and inherited by every variable that ranges over it,
        regardless of the variable's own name. Unknown names are returned
        unchanged.
        """
        if name in self._dimensions:
            return name
        dim = self.dimension_of(name)
        return dim.name if dim is not None else name

    def dimension_of(self, name: str) -> Optional[Dimension]:
        """The dimension of the structural variable *name*, or of the one it's an alias of."""
        sv = self._structural_variables.get(name) or self._structural_variables.get(_base_name_of(name))
        return sv.dimension if sv is not None else None

    def is_dimension_name(self, name: str) -> bool:
        """Whether *name* is a dimension or looks like an alias of one (``T1``)."""
        return name in self._dimensions or _base_name_of(name) in self._dimensions

    def alias_base(self, name: str) -> Optional[str]:
        """The base variable of the registered alias *name* (``x1`` → ``x``), or ``None``."""
        return self._variables_aliases.get(name) or self._structural_variables_aliases.get(name)

    def alias(self, name: str) -> VariableSymbol | StructuralVariableSymbol | None:
        """Return the alias *name* (``t1``, ``x'``) of a declared variable, registering it if new.

        Returns ``None`` if *name* is not an alias of a (structural) variable.
        """
        base = _base_name_of(name)
        if base == name:
            return None
        if base in self._structural_variables:
            alias = StructuralVariableSymbol(name=name, dimension=self._structural_variables[base].dimension)
            self._structural_variables[name] = alias
            self._structural_variables_aliases[name] = self._structural_variables_aliases.get(base, base)
            return alias
        if base in self._variables:
            base_var = self._variables[base]
            alias = VariableSymbol(name=name, sort=base_var.sort, dims=base_var.structural_variables or None)
            self._variables[name] = alias
            self._variables_aliases[name] = self._variables_aliases.get(base, base)
            return alias
        return None

    @contextmanager
    def rollback_aliases_on_error(self) -> Iterator[None]:
        """Undo the aliases registered inside the block if it raises."""
        registries = (self._variables, self._variables_aliases,
                      self._structural_variables, self._structural_variables_aliases)
        snapshot = [dict(r) for r in registries]
        try:
            yield
        except BaseException:
            for registry, saved in zip(registries, snapshot):
                registry.clear()
                registry.update(saved)
            raise

    # Read-only views of the declared symbols, by name; declare through the methods above.
    sorts                = property(lambda self: MappingProxyType(self._sorts))
    dimensions           = property(lambda self: MappingProxyType(self._dimensions))
    constants            = property(lambda self: MappingProxyType(self._constants))
    variables            = property(lambda self: MappingProxyType(self._variables))
    functions            = property(lambda self: MappingProxyType(self._functions))
    predicates           = property(lambda self: MappingProxyType(self._predicates))
    structural_variables = property(lambda self: MappingProxyType(self._structural_variables))
    structural_relations = property(lambda self: MappingProxyType(self._structural_relations))
    definitions          = property(lambda self: MappingProxyType(self._definitions))

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
        """Parse *formula_string* as a formula (or as a term, if the whole string is one)."""
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
        _section("Structural relations", self._structural_relations.values())
        _section("Constants",             self._constants.values())
        _section("Variables",             self._variables.values())
        _section("Functions",             self._functions.values())
        _section("Predicates",            self._predicates.values())
        _section("Definitions",           self._definitions.values())

        body = "\n  ".join(lines) if lines else "(empty)"
        return f"Signature({self.name!r}):\n  {body}"
