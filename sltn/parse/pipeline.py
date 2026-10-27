"""
sltn.parse.pipeline — Parse-time AST pipeline for :class:`~sltn.signature.Signature`.

These functions are invoked by :meth:`Signature.parse` and :meth:`Signature.parse_term`
after the grammar produces a raw AST.  The pipeline runs in this order:

1. :func:`_expand_definitions`               — expand syntactic abbreviations
2. :func:`_resolve_symbols`                  — replace string names with Symbol objects
3. :func:`_check_arguments`                  — check the number and sorts of arguments
4. :func:`_annotate_free_structural_variables` — compute d(node) for each AST node
5. :func:`_annotate_free_variables`          — compute free FOL variables per node
6. :func:`_validate_quantifiers`            — check bound names are distinct and used
7. :func:`_validate_kb_clauses`             — check every clause of a parsed KB is closed
"""
from __future__ import annotations

import warnings
from typing import Any, Literal, TYPE_CHECKING

from sltn.signature.symbols import (
    VariableSymbol, ConstantSymbol, StructuralVariableSymbol, FunctionSymbol, PredicateSymbol, Dimension
)

from sltn.fol.term import (
    _NameTerm, Variable, Constant, StructuralVariable,
    FunctionApplication, _CallTerm, TermAxisSelect, Term
)
from sltn.fol.formula import (
    AtomicFormula, StructuralRelation, And, Or,
    Not, Implies, Equivalent,
    Forall, Exists, _BoundFormula,
    KB,
    TermFormula, FormulaAxisSelect, Formula
)

from sltn.parse.parser import parse_string, signature_tables

from sltn.fol.definitions import _expand_definitions, _child_nodes

if TYPE_CHECKING:
    from sltn.signature.signature import Signature


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------

def _dimension_name_error(name: str, sig: Signature) -> NameError:
    return NameError(
        f"'{name}' is a dimension name (or an alias of one), which can't name an axis. "
        f"Declare a structural variable over the dimension with "
        f"sig.structural_variable(name, dimension) and use that name instead."
    )


def _check_annotation_dimensions(name: str, dims: list, annotations: list, sig: Signature) -> None:
    """Raise unless ``name[...]`` names every structural axis of the symbol.

    The i-th annotation names the i-th declared dimension, so there must be one
    distinct structural variable (or alias) per dimension.
    """
    written = f"{name}[{', '.join(annotations)}]" if annotations else name
    if len(annotations) != len(dims):
        raise ValueError(
            f"'{written}' names {len(annotations)} structural axis/axes, but '{name}' has "
            f"{len(dims)} dimension(s) {[d.name for d in dims]}. Name each axis with a "
            f"structural variable over its dimension."
            if dims else
            f"'{written}' has axis annotations, but '{name}' has no structural dimensions."
        )
    if len(set(annotations)) != len(annotations):
        raise ValueError(f"'{written}' uses the same axis name more than once.")
    for dim, ann in zip(dims, annotations):
        ann_dim = sig.dimension_of(ann)
        if ann_dim is None:
            if sig.is_dimension_name(ann):
                raise _dimension_name_error(ann, sig)
            raise NameError(
                f"Axis annotation '{ann}' in '{written}' is not a structural variable "
                f"or an alias of one."
            )
        if ann_dim.name != dim.name:
            raise ValueError(
                f"Axis annotation '{ann}' in '{written}' ranges over "
                f"dimension '{ann_dim.name}', but the axis it renames has dimension '{dim.name}'."
            )


def _check_output_annotation(sym, output_axes: list | None, sig: Signature) -> None:
    """Raise unless ``f(...)[a1, …]`` names exactly the output axes of *sym*, in order."""
    if not sym.output_dims and output_axes:
        raise ValueError(
            f"'{sym.name}(...)[{', '.join(output_axes)}]': '{sym.name}' declares no output "
            f"dimensions, so its result keeps the argument axes and takes no annotation. "
            f"To rename an argument axis, annotate the argument instead (e.g. x[t1])."
        )
    _check_annotation_dimensions(f"{sym.name}(...)", sym.output_dims, list(output_axes or []), sig)


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
        arg_dim = sig.dimension_of(arg.symbol)
        if arg_dim is None:
            if sig.is_dimension_name(arg.symbol):
                raise _dimension_name_error(arg.symbol, sig)
            raise NameError(
                f"Argument '{arg.symbol}' of structural relation '{sym.name}' is not a structural "
                f"variable or an alias of one."
            )
        if arg_dim.name != dim.name:
            raise ValueError(
                f"Argument '{arg.symbol}' of structural relation '{sym.name}' ranges over "
                f"dimension '{arg_dim.name}', but the relation expects '{dim.name}' there."
            )
        names.append(arg.symbol)
    return tuple(names)


def _check_selection(axis: str, base: Any, sig: Signature) -> None:
    """Raise unless ``base[axis=k]`` selects a structural axis of *base*."""
    available = base._free_structural_variables or set()
    if axis not in available:
        raise ValueError(
            f"Selection '[{axis}=...]' on '{base}': '{axis}' is not a structural axis of it. "
            f"Its structural axes are {sorted(available) or '(none)'}."
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
            if name in sig.structural_variables:
                resolved = sig.structural_variables[name]
            elif name in sig.variables:
                resolved = sig.variables[name]
            elif name in sig.constants:
                resolved = sig.constants[name]
            else:
                resolved = sig.alias(name)
                if resolved is None:
                    if sig.is_dimension_name(name):
                        raise _dimension_name_error(name, sig)
                    raise NameError(
                        f"Symbol '{name}' is not declared in signature '{sig.name}'. "
                        f"Available variables: {list(sig.variables)}, "
                        f"constants: {list(sig.constants)}, "
                        f"structural_variables: {list(sig.structural_variables)}."
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
            if sym_name in sig.functions:
                args = list(node.args)
                out_axes = node.output_axes
                _check_output_annotation(sig.functions[sym_name], out_axes, sig)
                node.__class__ = FunctionApplication
                FunctionApplication.__init__(node, sig.functions[sym_name], args, output_axes=out_axes)
                for arg in node.args:
                    _resolve_symbols(arg, sig)
            elif sym_name in sig.predicates:
                pred = sig.predicates[sym_name]
                args = list(node.args)
                out_axes = node.output_axes
                _check_output_annotation(pred, out_axes, sig)
                node.__class__ = AtomicFormula
                AtomicFormula.__init__(node, pred, *args, structural_axes=out_axes)
                for arg in args:
                    _resolve_symbols(arg, sig)
            else:
                raise NameError(
                    f"FunctionSymbol/predicate '{sym_name}' is not declared in signature '{sig.name}'. "
                    f"Available predicates: {list(sig.predicates)}, "
                    f"functions: {list(sig.functions)}."
                )

        case TermAxisSelect():
            _resolve_symbols(node.base_term, sig)

        case AtomicFormula():
            sym_name = node.symbol if isinstance(node.symbol, str) else node.symbol.name
            if sym_name in sig.predicates:
                node.symbol = sig.predicates[sym_name]
                _check_output_annotation(node.symbol, node.structural_axes, sig)
                for arg in node.args:
                    _resolve_symbols(arg, sig)
            elif sym_name in sig.functions:
                node.symbol = sig.functions[sym_name]
                _check_output_annotation(node.symbol, node.structural_axes, sig)
                for arg in node.args:
                    _resolve_symbols(arg, sig)
            elif sym_name in sig.structural_relations:
                # Promote to StructuralRelation — pure structural node with no Term args.
                sym_obj = sig.structural_relations[sym_name]
                var_names = _relation_argument_names(sym_obj, list(node.args), sig)
                node.__class__ = StructuralRelation
                node.__dict__ = {}
                StructuralRelation.__init__(node, sym_obj, *var_names)
            else:
                raise NameError(
                    f"PredicateSymbol, function, or structural predicate '{sym_name}' is not declared "
                    f"in signature '{sig.name}'. "
                    f"Available predicates: {list(sig.predicates)}, "
                    f"functions: {list(sig.functions)}, "
                    f"structural_relations: {list(sig.structural_relations)}."
                )

        case TermFormula() if isinstance(node.term, _NameTerm) and node.term.symbol in sig.predicates:
            # A nullary predicate written without parentheses: Rain.
            pred = sig.predicates[node.term.symbol]
            axes = node.term.axis_annotations or None
            _check_output_annotation(pred, axes, sig)
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
            # var_groups hold raw strings (not _NameTerm nodes) so they bypass the _NameTerm branch.
            for vname in node.var_names:
                if vname in sig.structural_variables or vname in sig.variables:
                    continue
                if vname in sig.dimensions:
                    raise _dimension_name_error(vname, sig)
                if sig.alias(vname) is None:
                    if sig.is_dimension_name(vname):
                        raise _dimension_name_error(vname, sig)
                    raise NameError(
                        f"Quantified name '{vname}' in signature '{sig.name}' is not declared "
                        f"as a variable (sig.variable(...)) or a structural variable "
                        f"(sig.structural_variable(...)), and is not a recognized numeric/primed "
                        f"alias of one (e.g. 't1' for 't')."
                    )
            if node.guard is not None:
                _resolve_symbols(node.guard, sig)
            _resolve_symbols(node.formula, sig)

        case KB():
            for clause in node.clauses:
                _resolve_symbols(clause, sig)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def _sort_name(arg: Formula | Term) -> str | None:
    """Sort of a resolved argument; ``None`` when it has none (a structural variable
    over a dimension without a sort)."""
    match arg:
        case Variable() | Constant():
            return arg.symbol.sort.name
        case FunctionApplication():
            return arg.symbol.output_sort.name
        case TermAxisSelect():
            return _sort_name(arg.base_term)
        case StructuralVariable():
            sort = arg.symbol.dimension.sort
            return sort.name if sort is not None else None
        case Formula():
            return "Bool"
        case _:
            return None


def _check_arguments(node: Formula | Term) -> None:
    """Check the number and the sorts of the arguments of every predicate and function.

    Sorts are compared by name, so a sort redeclared after a symbol was declared
    still matches. Arguments are checked before the call that uses them, so the
    innermost mistake is reported: ``Near(embed(x, x))`` reports ``embed``'s arity.
    """
    for child in _child_nodes(node):
        _check_arguments(child)
    if isinstance(node, (AtomicFormula, FunctionApplication)):
        sym = node.symbol
        kind = "Predicate" if isinstance(sym, PredicateSymbol) else "Function"
        expected = sym.input_sorts
        if len(node.args) != len(expected):
            raise TypeError(
                f"{kind} '{sym.name}' expects {len(expected)} argument(s), got {len(node.args)}, in {node}."
            )
        for i, (arg, sort) in enumerate(zip(node.args, expected)):
            got = _sort_name(arg)
            if got is not None and got != sort.name:
                raise TypeError(
                    f"{kind} '{sym.name}': argument {i} has sort '{got}', expected '{sort.name}', in {node}."
                )


# ---------------------------------------------------------------------------
# Quantifier validation
# ---------------------------------------------------------------------------

def _validate_quantifiers(node: Formula | Term, sig: Signature) -> None:
    """Validate the names bound by every quantifier.

    - A quantifier must not bind the same name twice (``forall y, y``).
    - Every bound variable must occur free in the body or the guard (as a free
      variable, or as a free structural axis); otherwise the quantifier is vacuous
      (``forall y: C(x)``).

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
            structural = [v for v in node.var_names if v in sig.structural_variables]
            individual = [v for v in node.var_names if v not in sig.structural_variables]
            if structural and individual:
                q = "forall" if node.quantifier == "forall" else "exists"
                raise ValueError(
                    f"Quantifier mixes individual variable(s) {individual} and structural "
                    f"variable(s) {structural}. Nest them instead, e.g. "
                    f"'{q} {', '.join(individual)}: {q} {', '.join(structural)}: ...'."
                )
            dims = {sig.structural_variables[v].dimension.name for v in structural}
            if len(dims) > 1:
                raise ValueError(
                    f"Quantifier binds structural variables {structural} over different "
                    f"dimensions {sorted(dims)}. Use one quantifier per dimension."
                )
            free = set(node.formula.free_variables)
            used_axes = set(node.formula.free_structural_variables)
            if node.guard is not None:
                free |= node.guard.free_variables
                used_axes |= node.guard.free_structural_variables
            for vname in node.var_names:
                if vname in sig.variables and vname not in free:
                    raise ValueError(
                        f"Variable '{vname}' is declared in the quantifier but never used "
                        f"in the formula body or guard, so the quantifier has no effect. "
                        f"Remove '{vname}' from the quantifier or use it in the body."
                    )
                if vname in sig.structural_variables and vname not in used_axes:
                    raise ValueError(
                        f"Structural variable '{vname}' is declared in the quantifier but names no "
                        f"free axis of the formula body or guard, so the quantifier has no effect. "
                        f"Free structural axes: {sorted(used_axes) or '(none)'}."
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

def _annotate_free_structural_variables(node: Formula | Term, sig: Signature) -> set[str]:
    """Recursively annotate every AST node with its structural-axis set.

    Stores the result as ``node._free_structural_variables`` and returns it.
    Called by :meth:`Signature.parse` after symbol resolution.

    Rules (structural domain calculus, alias-name edition):

    - d(StructuralVariableSymbol sv)                = {sv.name}
    - d(VariableSymbol x) / d(ConstantSymbol c), with annotations a₁…aₖ over dims D₁…Dₙ
                                     = {a₁, …, a_min(k,n)} ∪ {D_k+1, …, Dₙ}
      (the evaluator renames positionally; a repeated D is named D0, D1, …)
    - d(Call/FunctionCall f)         = (∪ d(tᵢ) − consumed(f)) ∪ out_names(f)
    - d(AtomicFormula p)             = (∪ d(tᵢ) − consumed(p)) ∪ out_names(p)
      (the dimension of each axis comes from the symbols in the tree, see
      :func:`~sltn.fol.term._call_axis_dims`)
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
            dims = set(node._axis_dimensions())

        case _NameTerm():
            # Unresolved _NameTerm — structural set unknown here.
            dims = set()

        case TermAxisSelect():
            _annotate_free_structural_variables(node.base_term, sig)
            _check_selection(node.axis, node.base_term, sig)
            base_dims: set[str] = node.base_term._free_structural_variables or set()
            dims = base_dims - {node.axis}

        case FunctionApplication():
            for arg in node.args:
                _annotate_free_structural_variables(arg, sig)
            dims = set(node._axis_dimensions())

        case StructuralRelation():
            dims = set(node.var_names)

        case AtomicFormula():
            for arg in node.args:
                _annotate_free_structural_variables(arg, sig)
            dims = set(node._axis_dimensions())

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
                if vname in sig.structural_variables:
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
    with sig.rollback_aliases_on_error():
        if sig.definitions:
            ast = _expand_definitions(ast, sig)
        _resolve_symbols(ast, sig)
        _check_arguments(ast)
        _annotate_free_structural_variables(ast, sig)
        _annotate_free_variables(ast, sig)
        _validate_quantifiers(ast, sig)
        _validate_kb_clauses(ast)
    return ast


def parse_raw(source: str, sig: "Signature", expect: Literal["formula", "term", "any"]) -> "Formula | Term":
    """Parse *source* with *sig*'s infix symbols, without resolving names.

    ``"any"`` parses a formula, but returns a bare term (``digit(x)``, ``y``) as a term.
    """
    infix, functions = signature_tables(sig)
    ast = parse_string(source, infix, functions, "term" if expect == "term" else "formula")
    if (expect == "any" and isinstance(ast, TermFormula)
            and not (isinstance(ast.term, _NameTerm) and ast.term.symbol in sig.predicates)):
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
