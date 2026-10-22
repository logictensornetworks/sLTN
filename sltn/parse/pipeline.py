"""
sltn.parse.pipeline — Parse-time AST pipeline for :class:`~sltn.signature.Signature`.

These functions are invoked by :meth:`Signature.parse` and :meth:`Signature.parse_term`
after the grammar produces a raw AST.  The pipeline runs in this order:

1. :func:`_expand_definitions`               — expand syntactic abbreviations
2. :func:`_resolve_symbols`                  — replace string names with Symbol objects
3. :func:`_annotate_free_structural_variables` — compute d(node) for each AST node
4. :func:`_annotate_free_variables`          — compute free FOL variables per node
5. :func:`_validate_quantifiers`            — check bound names are distinct and used
6. :func:`_validate_kb_clauses`             — check every clause of a parsed KB is closed
"""
from __future__ import annotations

import warnings
from typing import Any, Literal, TYPE_CHECKING

from sltn.signature.symbols import (
    VariableSymbol, ConstantSymbol, StructuralVariableSymbol, FunctionSymbol, PredicateSymbol, Dimension
)

from sltn.fol.term import (
    _NameTerm, Variable, Constant, StructuralVariable,
    FunctionApplication, _CallTerm, TermAxisSelect, Term, _annotated_axis_names
)
from sltn.fol.formula import (
    AtomicFormula, StructuralRelation, And, Or,
    Not, Implies, Equivalent,
    Forall, Exists, _BoundFormula,
    KB, StructuralForall, StructuralExists,
    TermFormula, FormulaAxisSelect, Formula
)

from sltn.parse.parser import parse_string, signature_tables

from sltn.fol.definitions import _expand_definitions, _base_name_of, _child_nodes

if TYPE_CHECKING:
    from sltn.signature.signature import Signature


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------

def _annotation_dimension(name: str, sig: Signature) -> str | None:
    """Return the dimension named by the axis annotation *name*, or ``None`` if unknown.

    *name* can be a structural variable, an alias of one (``t1``), or a dimension name.
    Nothing is registered in *sig*.
    """
    if name in sig._structural_variables:
        return sig._structural_variables[name].dimension.name
    if name in sig._dimensions:
        return name
    base = _base_name_of(name)
    if base != name:
        real_base = sig._structural_variables_aliases.get(base, base)
        if real_base in sig._structural_variables:
            return sig._structural_variables[real_base].dimension.name
        if base in sig._dimensions:
            return base
    return None


def _check_annotation_dimensions(name: str, dims: list, annotations: list, sig: Signature) -> None:
    """Raise if the annotations in ``name[...]`` don't match the symbol's dims.

    Annotations rename the symbol's structural axes position by position, so there can't be
    more annotations than dims, and the i-th one must range over the i-th declared dimension.
    """
    written = f"{name}[{', '.join(annotations)}]"
    if len(annotations) > len(dims):
        raise ValueError(
            f"'{written}' has {len(annotations)} axis annotation(s), but '{name}' has "
            f"{len(dims)} dimension(s): {[d.name for d in dims]}."
        )
    for dim, ann in zip(dims, annotations):
        ann_dim = _annotation_dimension(ann, sig)
        if ann_dim is None:
            raise NameError(
                f"Axis annotation '{ann}' in '{written}' is not a structural variable, "
                f"an alias of one, or a dimension."
            )
        if ann_dim != dim.name:
            raise ValueError(
                f"Axis annotation '{ann}' in '{written}' ranges over "
                f"dimension '{ann_dim}', but the axis it renames has dimension '{dim.name}'."
            )


def _relation_argument_names(sym, args: list, sig: Signature) -> tuple[str, ...]:
    """Return the axis names passed to the structural relation *sym*, checking each one."""
    dims = sym.structural_variables
    if len(args) != len(dims):
        raise ValueError(
            f"Structural relation '{sym.name}' takes {len(dims)} argument(s) "
            f"({[d.name for d in dims]}), got {len(args)}."
        )
    names = []
    for arg, dim in zip(args, dims):
        if not isinstance(arg, _NameTerm) or arg.axis_annotations:
            raise ValueError(
                f"Arguments of structural relation '{sym.name}' must be structural variable "
                f"names, got '{arg}'."
            )
        arg_dim = _annotation_dimension(arg.symbol, sig)
        if arg_dim is None:
            raise NameError(
                f"Argument '{arg.symbol}' of structural relation '{sym.name}' is not a structural "
                f"variable, an alias of one, or a dimension."
            )
        if arg_dim != dim.name:
            raise ValueError(
                f"Argument '{arg.symbol}' of structural relation '{sym.name}' ranges over "
                f"dimension '{arg_dim}', but the relation expects '{dim.name}' there."
            )
        names.append(arg.symbol)
    return tuple(names)


def _check_selection(axis: str, base: Any, sig: Signature) -> None:
    """Raise unless ``base[axis=k]`` selects a structural axis that is named explicitly in *base*.

    ``x[t][t=0]`` is accepted; ``x[T=0]``, which relies on the default axis name, is not.
    """
    available = base._free_structural_variables or set()
    if axis not in available:
        raise ValueError(
            f"Selection '[{axis}=...]' on '{base}': '{axis}' is not a structural axis of it. "
            f"Its structural axes are {sorted(available) or '(none)'}."
        )
    if axis not in _collect_explicit_axis_names(base):
        raise ValueError(
            f"Selection '[{axis}=...]' on '{base}' relies on the default axis name '{axis}'. "
            f"Name the axis first, e.g. 'x[t][t=0]' or 'P(x)[t][t=0]'."
        )


def _resolve_symbols(node: Formula | Term, sig: Signature) -> None:
    """Walk the AST and replace string names with Symbol objects from *sig*.

    Auto-registers alias ``StructuralVariableSymbol`` / ``VariableSymbol`` entries in *sig* for
    numbered / prime suffixes (e.g. ``t1``, ``x'``).
    """
    match node:
        case _NameTerm():
            name = node.symbol
            # Resolution order: StructuralVariableSymbol > VariableSymbol > ConstantSymbol > alias
            if name in sig._structural_variables:
                resolved = sig._structural_variables[name]
            elif name in sig._variables:
                resolved = sig._variables[name]
            elif name in sig._constants:
                resolved = sig._constants[name]
            else:
                base = _base_name_of(name)
                if base != name:
                    if base in sig._structural_variables or base in sig._structural_variables_aliases:
                        real_base = sig._structural_variables_aliases.get(base, base)
                        base_sv = sig._structural_variables.get(real_base)
                        if base_sv is None:
                            raise NameError(
                                f"Symbol '{name}' is not declared in signature '{sig.name}'."
                            )
                        resolved = StructuralVariableSymbol(name=name, dimension=base_sv.dimension)
                        sig._structural_variables[name] = resolved
                        sig._structural_variables_aliases[name] = real_base
                    elif base in sig._variables or base in sig._variables_aliases:
                        real_base = sig._variables_aliases.get(base, base)
                        base_var = sig._variables.get(real_base)
                        if base_var is None:
                            raise NameError(
                                f"Symbol '{name}' is not declared in signature '{sig.name}'."
                            )
                        resolved = VariableSymbol(name=name, sort=base_var.sort, dims=base_var.structural_variables or None)
                        sig._variables[name] = resolved
                        sig._variables_aliases[name] = real_base
                    elif base in sig._dimensions:
                        dim = sig._dimensions[base]
                        if base not in sig._structural_variables:
                            sig._structural_variables[base] = StructuralVariableSymbol(name=base, dimension=dim)
                        resolved = StructuralVariableSymbol(name=name, dimension=dim)
                        sig._structural_variables[name] = resolved
                        sig._structural_variables_aliases[name] = base
                    else:
                        raise NameError(
                            f"Symbol '{name}' is not declared in signature '{sig.name}'. "
                            f"Available variables: {list(sig._variables)}, "
                            f"constants: {list(sig._constants)}, "
                            f"structural_variables: {list(sig._structural_variables)}."
                        )
                elif name in sig._dimensions:
                    dim = sig._dimensions[name]
                    resolved = StructuralVariableSymbol(name=name, dimension=dim)
                    sig._structural_variables[name] = resolved
                else:
                    raise NameError(
                        f"Symbol '{name}' is not declared in signature '{sig.name}'. "
                        f"Available variables: {list(sig._variables)}, "
                        f"constants: {list(sig._constants)}, "
                        f"structural_variables: {list(sig._structural_variables)}."
                    )
            # Promote _NameTerm in-place to typed term node.
            ann = list(node.axis_annotations)
            if isinstance(resolved, (VariableSymbol, ConstantSymbol)):
                _check_annotation_dimensions(name, resolved.structural_variables, ann, sig)
            match resolved:
                case StructuralVariableSymbol():
                    node.__class__ = StructuralVariable
                    StructuralVariable.__init__(node, resolved, ann)
                case VariableSymbol():
                    node.__class__ = Variable
                    Variable.__init__(node, resolved, ann)
                case ConstantSymbol():
                    node.__class__ = Constant
                    Constant.__init__(node, resolved, ann)
                case _:
                    raise TypeError(f"Unexpected resolved symbol type: {type(resolved)!r}")

        case _CallTerm():
            sym_name = node.symbol
            if sym_name in sig._functions:
                args = list(node.args)
                out_axes = node.output_axes
                node.__class__ = FunctionApplication
                FunctionApplication.__init__(node, sig._functions[sym_name], args, output_axes=out_axes)
                for arg in node.args:
                    _resolve_symbols(arg, sig)
            elif sym_name in sig._predicates:
                pred = sig._predicates[sym_name]
                args = list(node.args)
                out_axes = node.output_axes
                node.__class__ = AtomicFormula
                AtomicFormula.__init__(node, pred, *args, structural_axes=out_axes)
                for arg in args:
                    _resolve_symbols(arg, sig)
            else:
                raise NameError(
                    f"FunctionSymbol/predicate '{sym_name}' is not declared in signature '{sig.name}'. "
                    f"Available predicates: {list(sig._predicates)}, "
                    f"functions: {list(sig._functions)}."
                )

        case TermAxisSelect():
            _resolve_symbols(node.base_term, sig)

        case AtomicFormula():
            sym_name = node.symbol if isinstance(node.symbol, str) else node.symbol.name
            if sym_name in sig._predicates:
                node.symbol = sig._predicates[sym_name]
                for arg in node.args:
                    _resolve_symbols(arg, sig)
            elif sym_name in sig._functions:
                node.symbol = sig._functions[sym_name]
                for arg in node.args:
                    _resolve_symbols(arg, sig)
            elif sym_name in sig._structural_relations:
                # Promote to StructuralRelation — pure structural node with no Term args.
                sym_obj = sig._structural_relations[sym_name]
                var_names = _relation_argument_names(sym_obj, list(node.args), sig)
                node.__class__ = StructuralRelation
                node.__dict__ = {}
                StructuralRelation.__init__(node, sym_obj, *var_names)
            else:
                raise NameError(
                    f"PredicateSymbol, function, or structural predicate '{sym_name}' is not declared "
                    f"in signature '{sig.name}'. "
                    f"Available predicates: {list(sig._predicates)}, "
                    f"functions: {list(sig._functions)}, "
                    f"structural_relations: {list(sig._structural_relations)}."
                )

        case TermFormula() if isinstance(node.term, _NameTerm) and node.term.symbol in sig._predicates:
            # A nullary predicate written without parentheses: Rain.
            pred = sig._predicates[node.term.symbol]
            axes = node.term.axis_annotations or None
            node.__class__ = AtomicFormula
            node.__dict__ = {}
            AtomicFormula.__init__(node, pred, structural_axes=axes)

        case TermFormula():
            _resolve_symbols(node.term, sig)

        case FormulaAxisSelect():
            _resolve_symbols(node.base_formula, sig)

        case And() | Or() | Implies() | Equivalent():
            _resolve_symbols(node.left, sig)
            _resolve_symbols(node.right, sig)

        case Not():
            _resolve_symbols(node.operand, sig)

        case _BoundFormula():
            # Auto-register StructuralVariables for quantified names that reference a Dimension.
            # var_groups hold raw strings (not _NameTerm nodes) so they bypass the _NameTerm branch.
            for vname in node.var_names:
                if vname in sig._structural_variables or vname in sig._variables:
                    continue
                if vname in sig._dimensions:
                    dim = sig._dimensions[vname]
                    sig._structural_variables[vname] = StructuralVariableSymbol(name=vname, dimension=dim)
                    continue
                base = _base_name_of(vname)
                resolved = False
                if base != vname:
                    if base in sig._structural_variables or base in sig._structural_variables_aliases:
                        real_base = sig._structural_variables_aliases.get(base, base)
                        base_sv = sig._structural_variables.get(real_base)
                        if base_sv is not None:
                            alias_sv = StructuralVariableSymbol(name=vname, dimension=base_sv.dimension)
                            sig._structural_variables[vname] = alias_sv
                            sig._structural_variables_aliases[vname] = real_base
                            resolved = True
                    elif base in sig._variables or base in sig._variables_aliases:
                        real_base = sig._variables_aliases.get(base, base)
                        base_var = sig._variables.get(real_base)
                        if base_var is not None:
                            sig._variables[vname] = VariableSymbol(name=vname, sort=base_var.sort, dims=base_var.structural_variables or None)
                            sig._variables_aliases[vname] = real_base
                            resolved = True
                    elif base in sig._dimensions:
                        dim = sig._dimensions[base]
                        if base not in sig._structural_variables:
                            sig._structural_variables[base] = StructuralVariableSymbol(name=base, dimension=dim)
                        sig._structural_variables[vname] = StructuralVariableSymbol(name=vname, dimension=dim)
                        sig._structural_variables_aliases[vname] = base
                        resolved = True
                if not resolved:
                    raise NameError(
                        f"Quantified name '{vname}' in signature '{sig.name}' is not declared "
                        f"as a variable (sig.variable(...)), a structural variable "
                        f"(sig.structural_variable(...)), or a dimension (sig.dimension(...)), "
                        f"and is not a recognized numeric/primed alias of one (e.g. 't1' for 't')."
                    )
            if node.guard is not None:
                _resolve_symbols(node.guard, sig)
            _resolve_symbols(node.formula, sig)

        case KB():
            for clause in node.clauses:
                _resolve_symbols(clause, sig)


# ---------------------------------------------------------------------------
# Explicit-axis collection (used by _validate_quantifiers)
# ---------------------------------------------------------------------------

def _collect_explicit_axis_names(node: Formula | Term) -> set:
    """Return all axis names that appear explicitly in structural-axes annotations
    or as StructuralVariable arguments to predicates/functions.

    Walks the AST and collects strings from:
    - every ``AtomicFormula.structural_axes`` list and ``FunctionApplication.output_axes`` list
    - every ``StructuralVariable`` used as a predicate/function argument
      (including inside guards), so that a StructuralVariableSymbol used only via guard
      predicates is still considered "mentioned".
    """
    names: set = set()
    match node:
        case StructuralVariable():
            names.add(node.symbol.name)
            names.update(node.axis_annotations)
        case Variable() | Constant():
            names.update(node.axis_annotations)
        case StructuralRelation():
            names.update(node.var_names)
        case AtomicFormula():
            if node.structural_axes is not None:
                names.update(node.structural_axes)
            for arg in node.args:
                names |= _collect_explicit_axis_names(arg)
        case TermFormula():
            names |= _collect_explicit_axis_names(node.term)
        case FormulaAxisSelect():
            names.add(node.axis)
            names |= _collect_explicit_axis_names(node.base_formula)
        case And() | Or() | Implies() | Equivalent():
            names |= _collect_explicit_axis_names(node.left)
            names |= _collect_explicit_axis_names(node.right)
        case Not():
            names |= _collect_explicit_axis_names(node.operand)
        case _BoundFormula():
            names |= _collect_explicit_axis_names(node.formula)
            if node.guard is not None:
                names |= _collect_explicit_axis_names(node.guard)
        case KB():
            for c in node.clauses:
                names |= _collect_explicit_axis_names(c)
        case TermAxisSelect():
            if node.base_term.output_axes if hasattr(node.base_term, 'output_axes') else False:
                names.update(node.base_term.output_axes)
            names |= _collect_explicit_axis_names(node.base_term)
        case FunctionApplication():
            if node.output_axes:
                names.update(node.output_axes)
            for arg in node.args:
                names |= _collect_explicit_axis_names(arg)
    return names


# ---------------------------------------------------------------------------
# Quantifier validation
# ---------------------------------------------------------------------------

def _validate_quantifiers(node: Formula | Term, sig: Signature) -> None:
    """Validate the names bound by every quantifier.

    - A quantifier must not bind the same name twice (``forall y, y``).
    - Every bound individual variable must occur free in the body or the guard;
      otherwise the quantifier is vacuous (``forall y: C(x)``).
    - Every bound *alias* StructuralVariableSymbol (one whose name differs from its
      base Dimension name, e.g. ``t1``, ``t'``) must appear as an explicit axis
      annotation in the body or the guard; otherwise the aggregation would reduce
      nothing, because the default axis name is the dimension name.

    Nested shadowing is allowed when the outer variable is used
    (``forall y: (C(y) & forall y: C(y))``). In ``forall y: forall y: C(y)``
    the outer ``y`` is vacuous and is rejected.

    Raises :class:`ValueError` if a violation is found.
    """
    match node:
        case _BoundFormula():
            duplicates = sorted({v for v in node.var_names if list(node.var_names).count(v) > 1})
            if duplicates:
                raise ValueError(
                    f"Quantifier binds {', '.join(map(repr, duplicates))} more than once: "
                    f"{list(node.var_names)}."
                )
            structural = [v for v in node.var_names if v in sig._structural_variables]
            individual = [v for v in node.var_names if v not in sig._structural_variables]
            if structural and individual:
                q = "forall" if node.quantifier == "forall" else "exists"
                raise ValueError(
                    f"Quantifier mixes individual variable(s) {individual} and structural "
                    f"variable(s) {structural}. Nest them instead, e.g. "
                    f"'{q} {', '.join(individual)}: {q} {', '.join(structural)}: ...'."
                )
            dims = {sig._structural_variables[v].dimension.name for v in structural}
            if len(dims) > 1:
                raise ValueError(
                    f"Quantifier binds structural variables {structural} over different "
                    f"dimensions {sorted(dims)}. Use one quantifier per dimension."
                )
            free = set(node.formula.free_variables)
            if node.guard is not None:
                free |= node.guard.free_variables
            for vname in node.var_names:
                if vname in sig._variables and vname not in free:
                    raise ValueError(
                        f"Variable '{vname}' is declared in the quantifier but never used "
                        f"in the formula body or guard, so the quantifier has no effect. "
                        f"Remove '{vname}' from the quantifier or use it in the body."
                    )
            mentioned = _collect_explicit_axis_names(node.formula)
            if node.guard is not None:
                mentioned |= _collect_explicit_axis_names(node.guard)
            for vname in node.var_names:
                sv = sig._structural_variables.get(vname)
                if sv is not None and sv.name != sv.dimension.name:
                    if vname not in mentioned:
                        raise ValueError(
                            f"StructuralVariableSymbol '{vname}' is declared in the quantifier but never used "
                            f"as an explicit axis annotation in the formula body. "
                            f"Add e.g. `A(x)[{vname}]` or remove '{vname}' from the quantifier. "
                            f"Axis names found in body: {sorted(mentioned) or '(none)'}."
                        )
            _validate_quantifiers(node.formula, sig)
            if node.guard is not None:
                _validate_quantifiers(node.guard, sig)
        case And() | Or() | Implies() | Equivalent():
            _validate_quantifiers(node.left, sig)
            _validate_quantifiers(node.right, sig)
        case Not():
            _validate_quantifiers(node.operand, sig)
        case KB():
            for c in node.clauses:
                _validate_quantifiers(c, sig)
        case TermFormula():
            _validate_quantifiers(node.term, sig)
        case FormulaAxisSelect():
            _validate_quantifiers(node.base_formula, sig)


# ---------------------------------------------------------------------------
# Structural-variable annotation (dimensional calculus)
# ---------------------------------------------------------------------------

def _alias_to_dim(name: str, sig: Signature) -> Dimension | None:
    """Resolve an axis alias name to its underlying Dimension, or None.

    First looks up the name directly in structural variables and dimensions.
    If not found and the name has a trailing digit suffix (e.g. ``"t1"``),
    strips the suffix via :func:`_base_name_of` and retries, so that
    dynamically-generated aliases resolve to the same Dimension as their base.
    """
    sv = sig._structural_variables.get(name)
    if sv is not None:
        return sv.dimension
    base = _base_name_of(name)
    if base != name:
        sv = sig._structural_variables.get(base)
        if sv is not None:
            return sv.dimension
    return sig._dimensions.get(name)


def _annotate_free_structural_variables(node: Formula | Term, sig: Signature) -> set[str]:
    """Recursively annotate every AST node with its structural-axis set.

    Stores the result as ``node._free_structural_variables`` and returns it.
    Called by :meth:`Signature.parse` after symbol resolution.

    Rules (structural domain calculus, alias-name edition):

    - d(StructuralVariableSymbol sv)                = {sv.name}
    - d(VariableSymbol x) / d(ConstantSymbol c), with annotations a₁…aₖ over dims D₁…Dₙ
                                     = {a₁, …, a_min(k,n)} ∪ {D_k+1, …, Dₙ}
      (the evaluator renames positionally; with no annotation this is {D₁, …, Dₙ})
    - d(Call/FunctionCall f)         = (∪ d(tᵢ) − consumed(f)) ∪ out_names(f)
    - d(AtomicFormula p)             = (∪ d(tᵢ) − consumed(p)) ∪ out_names(p)
    - d(∘(φ, ψ))                     = d(φ) ∪ d(ψ)
    - d(¬φ)                          = d(φ)
    - d(∀x.φ) / d(∃x.φ)  [x symbol ref]  = d(φ)
    - d(∀α.φ) / d(∃α.φ)  [α SVar]    = d(φ) − {α.name}
    - d(StructAgg(axis, φ))          = d(φ) − {axis}
    - d(KB)                          = ∪ d(clause)
    """
    dims: set[str]

    match node:
        case StructuralVariable():
            dims = {node.symbol.name}

        case Variable() | Constant():
            dims = _annotated_axis_names(node.symbol.structural_variables, node.axis_annotations)

        case _NameTerm():
            # Unresolved _NameTerm — structural set unknown here.
            dims = set()

        case TermAxisSelect():
            _annotate_free_structural_variables(node.base_term, sig)
            _check_selection(node.axis, node.base_term, sig)
            base_dims: set[str] = node.base_term._free_structural_variables or set()
            dims = base_dims - {node.axis}

        case FunctionApplication():
            arg_dims: set[str] = set()
            for arg in node.args:
                _annotate_free_structural_variables(arg, sig)
                arg_dims |= (arg._free_structural_variables or set())
            sym = node.symbol
            if isinstance(sym, (FunctionSymbol, PredicateSymbol)):
                in_dim_set = set(sym.input_dims)
                consumed: set[str] = {
                    alias for alias in arg_dims
                    if _alias_to_dim(alias, sig) in in_dim_set
                }
                out_names: set[str] = set(node.output_axes) if node.output_axes is not None else {d.name for d in sym.output_dims}
                dims = (arg_dims - consumed) | out_names
            else:
                dims = arg_dims

        case StructuralRelation():
            dims = set(node.var_names)

        case AtomicFormula():
            arg_dims_a: set[str] = set()
            for arg in node.args:
                _annotate_free_structural_variables(arg, sig)
                arg_dims_a |= (arg._free_structural_variables or set())
            sym = node.symbol
            if isinstance(sym, (PredicateSymbol, FunctionSymbol)):
                in_dim_set = set(sym.input_dims)
                consumed_a: set[str] = {
                    alias for alias in arg_dims_a
                    if _alias_to_dim(alias, sig) in in_dim_set
                }
                out_names_a: set[str] = set(node.structural_axes) if node.structural_axes is not None else {d.name for d in sym.output_dims}
                dims = (arg_dims_a - consumed_a) | out_names_a
            else:
                dims = arg_dims_a

        case TermFormula():
            dims = _annotate_free_structural_variables(node.term, sig)

        case FormulaAxisSelect():
            inner_dims = _annotate_free_structural_variables(node.base_formula, sig)
            _check_selection(node.axis, node.base_formula, sig)
            dims = inner_dims - {node.axis}

        case And() | Or() | Implies() | Equivalent():
            dl = _annotate_free_structural_variables(node.left, sig)
            dr = _annotate_free_structural_variables(node.right, sig)
            dims = dl | dr

        case Not():
            dims = _annotate_free_structural_variables(node.operand, sig)

        case _BoundFormula():
            guard_dims: set[str] = set()
            if node.guard is not None:
                guard_dims = _annotate_free_structural_variables(node.guard, sig)
            inner_dims = _annotate_free_structural_variables(node.formula, sig)
            dims = inner_dims | guard_dims
            for vname in node.var_names:
                if vname in sig._structural_variables:
                    dims -= {vname}

        case KB():
            dims = set()
            for clause in node.clauses:
                dims |= _annotate_free_structural_variables(clause, sig)

        case _:
            dims = set()

    node._free_structural_variables = dims
    return dims


# ---------------------------------------------------------------------------
# Free-variable annotation
# ---------------------------------------------------------------------------

def _annotate_free_variables(node: Formula | Term, sig: Signature) -> set[str]:
    """Recursively annotate every AST node with its set of free FOL variables.

    Unlike :func:`_annotate_free_structural_variables`, this pass needs no
    signature lookups (no alias resolution) — it just eagerly caches, bottom
    up, the exact same result each node's own ``_compute_free_variables()``
    would produce lazily. Stores the result as ``node._free_variables`` and
    returns it. Called by :meth:`Signature.parse` after symbol resolution.
    """
    for child in _child_nodes(node):
        _annotate_free_variables(child, sig)
    node._free_variables = node._compute_free_variables()
    return node._free_variables

# ---------------------------------------------------------------------------
# KB clause validation (post-annotation)
# ---------------------------------------------------------------------------

def _validate_kb_clauses(node: Any) -> None:
    """Walk the AST and raise ValueError for any KB whose clauses are not closed."""
    match node:
        case KB():
            node.check_clauses()
        case And() | Or() | Implies() | Equivalent():
            _validate_kb_clauses(node.left)
            _validate_kb_clauses(node.right)
        case Not():
            _validate_kb_clauses(node.operand)
        case Forall() | Exists():
            _validate_kb_clauses(node.formula)
            if node.guard is not None:
                _validate_kb_clauses(node.guard)
        case StructuralForall() | StructuralExists():
            _validate_kb_clauses(node.formula)
        case FormulaAxisSelect():
            _validate_kb_clauses(node.base_formula)


# ---------------------------------------------------------------------------
# Top-level parse / pipeline entry points
# ---------------------------------------------------------------------------

def _warn_on_escaped_variables(ast: Formula | Term) -> None:
    """Warn when a quantified name also occurs free: usually a missing pair of parentheses."""
    if not isinstance(ast, Formula):
        return
    bound: set[str] = set()
    stack = [ast]
    while stack:
        n = stack.pop()
        if isinstance(n, _BoundFormula):
            bound.update(n.var_names)
        stack.extend(_child_nodes(n))
    escaped = (ast.free_variables | ast.free_structural_variables) & bound
    if escaped:
        warnings.warn(
            f"{sorted(escaped)} occur both quantified and free in {ast}. Quantifiers bind only "
            f"their immediate subformula; write e.g. 'forall x: (A(x) -> B(x))'.",
            stacklevel=4,
        )


def _run_pipeline(ast, sig: "Signature"):
    """Expand definitions and run all annotation passes on *ast* in place.

    Symbols auto-registered while resolving names (aliases such as ``x1`` or
    ``t'``) are rolled back if any pass fails, so a failed parse leaves *sig*
    unchanged.
    """
    registries = (sig._variables, sig._variables_aliases,
                  sig._structural_variables, sig._structural_variables_aliases)
    snapshot = [dict(r) for r in registries]
    try:
        if sig._definitions:
            ast = _expand_definitions(ast, sig._definitions, sig)
        _resolve_symbols(ast, sig)
        _annotate_free_structural_variables(ast, sig)
        _annotate_free_variables(ast, sig)
        _validate_quantifiers(ast, sig)
        _validate_kb_clauses(ast)
    except BaseException:
        for registry, saved in zip(registries, snapshot):
            registry.clear()
            registry.update(saved)
        raise
    return ast


def parse_raw(source: str, sig: "Signature", expect: Literal["formula", "term", "any"]) -> "Formula | Term":
    """Parse *source* with *sig*'s infix symbols, without resolving names.

    ``"any"`` parses a formula, but returns a bare term (``digit(x)``, ``y``) as a term.
    """
    infix, functions = signature_tables(sig)
    ast = parse_string(source, infix, functions, "term" if expect == "term" else "formula")
    if (expect == "any" and isinstance(ast, TermFormula)
            and not (isinstance(ast.term, _NameTerm) and ast.term.symbol in sig._predicates)):
        return ast.term
    return ast


def parse(formula_string: str, sig: "Signature") -> "Formula | Term":
    """Parse *formula_string* against *sig* and run the full pipeline (``sig.parse``).

    The result is a formula, or a term when the whole string is a term (``digit(x)``).
    """
    ast = _run_pipeline(parse_raw(formula_string, sig, "any"), sig)
    _warn_on_escaped_variables(ast)
    return ast


def parse_term(term_string: str, sig: "Signature") -> "Term":
    """Parse *term_string* against *sig* and run the resolution pipeline (``sig.parse_term``)."""
    return _run_pipeline(parse_raw(term_string, sig, "term"), sig)
