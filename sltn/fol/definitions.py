"""
sltn.fol.definitions — Syntactic definition expansion for the parse pipeline.

A :class:`~sltn.signature.Definition` is a named macro declared via
:meth:`Signature.define`.  This module provides :func:`_expand_definitions`,
which walks the raw AST produced by the parser and substitutes every
definition call with the corresponding body — performing capture-avoiding
renaming of structural-axis aliases as needed.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Callable

from sltn.fol.term import (
    _NameTerm, Variable, Constant, StructuralVariable, 
    _CallTerm, FunctionApplication, TermAxisSelect, Term
)
from sltn.fol.formula import (
    AtomicFormula, StructuralRelation,
    And, Or, Not, Implies, Equivalent,
    Forall, Exists,
    KB, StructuralForall, StructuralExists, 
    TermFormula, FormulaAxisSelect,
    Formula
)



def _base_name_of(name: str) -> str:
    """Strip trailing primes and/or digits from *name* to get the alias base.

    Examples: ``'x\\''`` → ``'x'``, ``'x1'`` → ``'x'``, ``'t1'`` → ``'t'``.
    Returns *name* unchanged when no suffix is found.
    """
    m = re.match(r"^(.+?)([0-9']+)$", name)
    return m.group(1) if m else name

# ---------------------------------------------------------------------------
# Definition expansion
# ---------------------------------------------------------------------------

def _build_numbered_alias_subst(
    param_names: list[str],
    args: list[Any],
) -> dict[str, Any]:
    """Build a substitution dict that also maps numbered aliases.

    For each formal parameter ``P`` bound to argument ``a``:

    * If ``a`` is a ``_NameTerm("X")``, then ``"P1"`` → ``_NameTerm("X1")``,
      ``"P2"`` → ``_NameTerm("X2")``, etc.
    * Otherwise (constant, complex term), ``"P1"`` … ``"P9"`` → ``a``.

    The numbered aliases produce *different* variable names so that
    ``U * U1`` in a definition body creates a cross-product rather than
    an element-wise product.
    """
    subst: dict[str, Any] = dict(zip(param_names, args))
    aliases: dict[str, Any] = {}
    for param, arg in subst.items():
        for n in range(1, 10):
            alias = f"{param}{n}"
            if alias not in subst:
                if isinstance(arg, _NameTerm):
                    arg_name = arg.symbol if isinstance(arg.symbol, str) else arg.symbol.name
                    aliases[alias] = _NameTerm(arg_name + str(n))
                else:
                    aliases[alias] = arg
    subst.update(aliases)
    return subst


def _child_nodes(n: Any) -> list[Any]:
    """Return the direct AST children of *n*, for generic recursive tree walks."""
    match n:
        case _CallTerm() | FunctionApplication():
            return list(n.args)
        case TermAxisSelect():
            return [n.base_term]
        case AtomicFormula():
            return list(n.args)
        case And() | Or() | Implies() | Equivalent():
            return [n.left, n.right]
        case Not():
            return [n.operand]
        case Forall() | Exists():
            children = [n.formula]
            if getattr(n, "guard", None) is not None:
                children.append(n.guard)
            return children
        case TermFormula():
            return [n.term]
        case FormulaAxisSelect():
            return [n.base_formula]
        case KB():
            return list(n.clauses)
        case StructuralForall() | StructuralExists():
            return [n.formula]
        case _:
            return []


def _walk_ast(node: Any, visit: Callable[[Any], None]) -> None:
    """Recursively invoke *visit* on *node* and every descendant AST node."""
    visit(node)
    for child in _child_nodes(node):
        _walk_ast(child, visit)


def _term_axis_annotations_for_subst(arg: Any) -> list[str] | None:
    """Structural axis names supplied on a call-site parameter (``x[s1]`` syntax)."""
    match arg:
        case _NameTerm() if arg.axis_annotations:
            return list(arg.axis_annotations)
        case Variable() if arg.axis_annotations:
            return list(arg.axis_annotations)
        case _:
            return None


def _collect_formal_param_axis_annotations(node: Any, pname: str) -> list[str] | None:
    """Return the axis-annotation list for ``_NameTerm(pname)[…]`` in a definition body (must be unique)."""
    found: list[str] | None = None

    def on_node(n: Any) -> None:
        nonlocal found
        if isinstance(n, _NameTerm) and n.symbol == pname and n.axis_annotations:
            ann = list(n.axis_annotations)
            if found is None:
                found = ann
            elif found != ann:
                raise ValueError(
                    f"Definition body uses {pname!r} with inconsistent axis templates "
                    f"{found!r} vs {ann!r}."
                )

    _walk_ast(node, on_node)
    return found


def _collect_param_axis_annotation_variants(node: Any, pname: str) -> list[tuple[str, ...]]:
    """All axis-annotation tuples used for definition parameter *pname* in *node*."""
    found: list[tuple[str, ...]] = []

    def on_node(n: Any) -> None:
        if isinstance(n, (_NameTerm, Variable, Constant)) and n.name == pname and n.axis_annotations:
            found.append(tuple(n.axis_annotations))

    _walk_ast(node, on_node)
    return found


def _param_axis_canonical_for_definition(defn: Any, body_ast: Any) -> dict[str, tuple[str, ...]]:
    """Pick a canonical axis template per definition parameter (lexicographic min of variants).

    When a body uses ``X[s, d]`` and ``X[s1, d]``, expansion must instantiate the call
    argument separately for each occurrence — see :func:`_substitute` with
    *param_axis_canonical*.
    """
    out: dict[str, tuple[str, ...]] = {}
    for pname in defn.params:
        variants = _collect_param_axis_annotation_variants(body_ast, pname)
        if not variants:
            continue
        lens = {len(v) for v in variants}
        if len(lens) != 1:
            continue
        out[pname] = min(variants)
    return out


def _collect_all_structural_axis_strings(node: Any) -> set[str]:
    """Collect every structural axis name string appearing in *node*."""
    found: set[str] = set()

    def on_node(n: Any) -> None:
        match n:
            case _NameTerm() | Variable() | Constant() | StructuralVariable():
                found.update(n.axis_annotations)
            case _CallTerm():
                if n.output_axes is not None:
                    found.update(n.output_axes)
            case TermAxisSelect():
                found.add(n.axis)
            case AtomicFormula():
                if n.structural_axes is not None:
                    found.update(n.structural_axes)
            case FormulaAxisSelect():
                found.add(n.axis)
            case Forall() | Exists():
                for group in n._var_groups:
                    found.update(group)
            case StructuralForall() | StructuralExists():
                found.add(n.structural_axis)

    _walk_ast(node, on_node)
    return found


def _fresh_structural_axis(used: set[str]) -> str:
    for i in range(2, 10_000):
        cand = f"s{i}"
        if cand not in used:
            return cand
    raise RuntimeError("Could not allocate a fresh structural axis name.")


def _resolve_dst_clash_with_struct_quantifiers(repl: Any, dst: str) -> None:
    """If *dst* is bound by ``Forall``/``Exists`` in *repl*, rename that binder (capture-avoid).

    Needed when a definition maps ``s → s1`` but *repl* already contains ``∀ s1 | …`` from a
    nested macro expansion: blind global renames would conflate the two roles.
    """
    def visit(n: Any) -> None:
        match n:
            case Forall() | Exists():
                visit(n.formula)
                if n.guard is not None:
                    visit(n.guard)
                bound = {v for group in n._var_groups for v in group}
                if dst in bound:
                    used = _collect_all_structural_axis_strings(repl)
                    fresh = _fresh_structural_axis(used)
                    for gi, group in enumerate(n._var_groups):
                        n._var_groups[gi] = [fresh if v == dst else v for v in group]
                    _rename_structural_axes_inplace(n.formula, {dst: fresh})
                    if n.guard is not None:
                        _rename_structural_axes_inplace(n.guard, {dst: fresh})
            case _NameTerm() | Variable() | Constant() | StructuralVariable():
                pass
            case _CallTerm():
                for a in n.args:
                    visit(a)
            case TermAxisSelect():
                visit(n.base_term)
            case AtomicFormula():
                for a in n.args:
                    visit(a)
            case TermFormula():
                visit(n.term)
            case FormulaAxisSelect():
                visit(n.base_formula)
            case And() | Or() | Implies() | Equivalent():
                visit(n.left)
                visit(n.right)
            case Not():
                visit(n.operand)
            case KB():
                for c in n.clauses:
                    visit(c)
            case StructuralForall() | StructuralExists():
                visit(n.formula)

    visit(repl)


def _rename_structural_axes_inplace(node: Any, mapping: dict[str, str]) -> None:
    """Rename structural axis name strings inside *node* (mutate in place)."""
    if not mapping:
        return

    def m(ax: str) -> str:
        return mapping.get(ax, ax)

    match node:
        case _NameTerm() | Variable() | Constant() | StructuralVariable():
            if node.axis_annotations:
                node.axis_annotations[:] = [m(a) for a in node.axis_annotations]
        case _CallTerm():
            if node.output_axes is not None:
                node.output_axes = [m(a) for a in node.output_axes]
            for a in node.args:
                _rename_structural_axes_inplace(a, mapping)
        case TermAxisSelect():
            node.axis = m(node.axis)
            _rename_structural_axes_inplace(node.base_term, mapping)
        case AtomicFormula():
            if node.structural_axes is not None:
                node.structural_axes[:] = [m(a) for a in node.structural_axes]
            for a in node.args:
                _rename_structural_axes_inplace(a, mapping)
        case TermFormula():
            _rename_structural_axes_inplace(node.term, mapping)
        case FormulaAxisSelect():
            node.axis = m(node.axis)
            _rename_structural_axes_inplace(node.base_formula, mapping)
        case And() | Or() | Implies() | Equivalent():
            _rename_structural_axes_inplace(node.left, mapping)
            _rename_structural_axes_inplace(node.right, mapping)
        case Not():
            _rename_structural_axes_inplace(node.operand, mapping)
        case Forall() | Exists():
            _rename_structural_axes_inplace(node.formula, mapping)
            if getattr(node, "guard", None) is not None:
                _rename_structural_axes_inplace(node.guard, mapping)
        case KB():
            for c in node.clauses:
                _rename_structural_axes_inplace(c, mapping)
        case StructuralForall() | StructuralExists():
            node.structural_axis = m(node.structural_axis)
            _rename_structural_axes_inplace(node.formula, mapping)


def _definition_subst(defn: Any, body_ast: Any, args: list[Any]) -> dict[str, Any]:
    """Build substitution for a definition call, including structural-axis renames.

    When the template body uses ``X[s, …]`` but the call site supplies ``x[s1, …]``,
    output-axis strings in the template (``…[s, v]``) must be rewritten to ``s1`` so
    tensor axis bookkeeping stays consistent after substitution.
    """
    subst = _build_numbered_alias_subst(defn.params, args)

    for pname, arg in zip(defn.params, args):
        actual_axes = _term_axis_annotations_for_subst(arg)
        if not actual_axes:
            continue
        formal_axes = _collect_formal_param_axis_annotations(body_ast, pname)
        if formal_axes is None or len(formal_axes) != len(actual_axes):
            continue
        for fa, aa in zip(formal_axes, actual_axes):
            if fa != aa:
                subst[fa] = _NameTerm(aa)
    return subst


def _substitute_axis_name(name: str, subst: dict[str, Any]) -> str:
    """Substitute a structural axis name via the term substitution map.

    If *name* maps to a named symbol term in *subst*, returns that node's
    logical ``name``.  Otherwise returns *name* unchanged.
    """
    if name in subst:
        val = subst[name]
        match val:
            case _NameTerm() | Variable() | Constant() | StructuralVariable():
                return val.name
    return name


def _substitute(
    node: Any,
    subst: dict[str, Any],
    *,
    param_axis_canonical: dict[str, tuple[str, ...]] | None = None,
) -> Any:
    """Syntactically substitute symbol names from *subst* in a formula/term AST.

    Replaces every symbol reference whose logical name is a key in *subst* with
    ``subst[name]``.  Structural axis name strings in annotations are also
    substituted via :func:`_substitute_axis_name`.  Used to instantiate a
    definition body with actual call-site arguments.

    *param_axis_canonical* maps definition parameter names to a canonical axis
    tuple (lexicographic minimum of all occurrences in the template body).  When
    a reference carries ``axis_annotations`` that differ from that canonical
    template, the replacement term is deep-copied and structural axis names are
    renamed so e.g. ``X[s1, d]`` does not incorrectly reuse the ``s`` slots from
    ``X[s, d]`` after substitution.
    """
    match node:
        case _NameTerm() | Variable() | Constant() | StructuralVariable():
            name = node.name
            if name in subst:
                repl = subst[name]
                if (
                    param_axis_canonical is not None
                    and isinstance(node, (_NameTerm, Variable, Constant))
                    and node.axis_annotations
                ):
                    canon = param_axis_canonical.get(name)
                    ann = tuple(node.axis_annotations)
                    if canon is not None and len(canon) == len(ann):
                        mapping = {c: a for c, a in zip(canon, ann) if c != a}
                        is_leaf_like = isinstance(
                            repl, (_NameTerm, Variable, Constant, StructuralVariable)
                        )
                        if is_leaf_like and not getattr(repl, "axis_annotations", None):
                            # A bare leaf replacement carries no axis annotation of its
                            # own, so re-apply the body's axis tag (renamed via `mapping`)
                            # onto it directly. Compound replacements (e.g. a `CallTerm`)
                            # are handled below via structural in-place renaming instead.
                            repl = copy.deepcopy(repl)
                            repl.axis_annotations = [mapping.get(a, a) for a in ann]
                        elif mapping:
                            repl = copy.deepcopy(repl)
                            for dst in set(mapping.values()):
                                _resolve_dst_clash_with_struct_quantifiers(repl, dst)
                            _rename_structural_axes_inplace(repl, mapping)
                return repl
            return node

        case _CallTerm():
            new_output_axes = (
                [_substitute_axis_name(ax, subst) for ax in node.output_axes]
                if node.output_axes is not None else None
            )
            return _CallTerm(
                node.symbol,
                [_substitute(a, subst, param_axis_canonical=param_axis_canonical) for a in node.args],
                output_axes=new_output_axes,
            )

        case TermAxisSelect():
            return TermAxisSelect(
                _substitute(node.base_term, subst, param_axis_canonical=param_axis_canonical),
                node.index,
                node.axis,
            )

        case AtomicFormula():
            new_structural_axes = (
                [_substitute_axis_name(ax, subst) for ax in node.structural_axes]
                if node.structural_axes is not None else None
            )
            return AtomicFormula(
                node.symbol,
                *[_substitute(a, subst, param_axis_canonical=param_axis_canonical) for a in node.args],
                structural_axes=new_structural_axes,
            )

        case TermFormula():
            return TermFormula(_substitute(node.term, subst, param_axis_canonical=param_axis_canonical))

        case FormulaAxisSelect():
            return FormulaAxisSelect(
                _substitute(node.base_formula, subst, param_axis_canonical=param_axis_canonical),
                axis=node.axis,
                index=node.index,
            )

        case And() | Or():
            return type(node)(
                _substitute(node.left, subst, param_axis_canonical=param_axis_canonical),
                _substitute(node.right, subst, param_axis_canonical=param_axis_canonical),
            )

        case Implies():
            return Implies(
                _substitute(node.left, subst, param_axis_canonical=param_axis_canonical),
                _substitute(node.right, subst, param_axis_canonical=param_axis_canonical),
            )

        case Equivalent():
            return Equivalent(
                _substitute(node.left, subst, param_axis_canonical=param_axis_canonical),
                _substitute(node.right, subst, param_axis_canonical=param_axis_canonical),
            )

        case Not():
            return Not(_substitute(node.operand, subst, param_axis_canonical=param_axis_canonical))

        case Forall() | Exists():
            rename: dict[str, str] = {}
            for v in node.var_names:
                if v in subst and isinstance(subst[v], (_NameTerm, Variable, Constant, StructuralVariable)):
                    rename[v] = subst[v].name

            new_var_groups = [
                [rename.get(v, v) for v in group]
                for group in node._var_groups
            ]
            non_renamed_bound = {v for v in node.var_names if v not in rename}
            inner = {k: v for k, v in subst.items() if k not in non_renamed_bound}
            new_formula = _substitute(node.formula, inner, param_axis_canonical=param_axis_canonical)
            new_guard = (
                _substitute(node.guard, inner, param_axis_canonical=param_axis_canonical)
                if node.guard is not None
                else None
            )
            return type(node)(new_var_groups, new_formula, guard=new_guard)

        case KB():
            return KB(
                [_substitute(c, subst, param_axis_canonical=param_axis_canonical) for c in node.clauses],
                validate=False,   # names are not resolved yet; the pipeline validates the KB
            )

        case StructuralForall():
            return StructuralForall(
                _substitute(node.formula, subst, param_axis_canonical=param_axis_canonical),
                structural_axis=_substitute_axis_name(node.structural_axis, subst),
            )

        case StructuralExists():
            return StructuralExists(
                _substitute(node.formula, subst, param_axis_canonical=param_axis_canonical),
                structural_axis=_substitute_axis_name(node.structural_axis, subst),
            )

        case _:
            return node


def _expand_definitions(
    node: Any,
    definitions: dict[str, Any],
    sig: Any,
) -> Any:
    """Recursively expand syntactic abbreviations in a formula/term AST.

    - **Formula level**: an :class:`~sltn.fol.formula.AtomicFormula` whose predicate
      name is in *definitions* is replaced by the instantiated body.
    - **Term level**: a :class:`~sltn.fol.term._CallTerm` whose function name is in
      *definitions* is replaced by the instantiated body (parsed as a formula).

    Expansion is recursive so definitions may reference other definitions.
    """
    def _term_sort_name_for_definition(term: Any) -> str | None:
        match term:
            case TermFormula():
                return _term_sort_name_for_definition(term.term)
            case AtomicFormula():
                sname = term.pred_name
                if sname in sig._predicates:
                    return "Bool"
                if sname in definitions:
                    out = definitions[sname].output_sort
                    return out.name if out is not None else None
                return None
            case Constant():
                return term.symbol.sort.name
            case Variable():
                return term.symbol.sort.name
            case StructuralVariable():
                if term.symbol.dimension.sort is not None:
                    return term.symbol.dimension.sort.name
                return None
            case _NameTerm():
                name = term.name
                if name in sig._constants:
                    return sig._constants[name].sort.name
                if name in sig._variables:
                    return sig._variables[name].sort.name
                return None
            case TermAxisSelect():
                return _term_sort_name_for_definition(term.base_term)
            case _CallTerm():
                sname = term.symbol
                if sname in sig._functions:
                    out = sig._functions[sname].output_sort
                    return out.name if out is not None else None
                if sname in sig._predicates:
                    return "Bool"
                if sname in definitions:
                    out = definitions[sname].output_sort
                    return out.name if out is not None else None
                return None
            case _:
                return None

    def _term_dim_names_for_definition(term: Any) -> set[str]:
        match term:
            case TermFormula():
                return _term_dim_names_for_definition(term.term)
            case And() | Or() | Implies() | Equivalent():
                return _term_dim_names_for_definition(term.left) | _term_dim_names_for_definition(term.right)
            case Not():
                return _term_dim_names_for_definition(term.operand)
            case Forall() | Exists():
                d = _term_dim_names_for_definition(term.formula)
                if term.guard is not None:
                    d |= _term_dim_names_for_definition(term.guard)
                return d
            case FormulaAxisSelect():
                return _term_dim_names_for_definition(term.base_formula)
            case StructuralVariable():
                return {term.name}
            case Variable():
                if term.axis_annotations:
                    return set(term.axis_annotations)
                return {d.name for d in term.symbol.structural_variables}
            case Constant():
                if term.axis_annotations:
                    return set(term.axis_annotations)
                return {d.name for d in term.symbol.structural_variables}
            case _NameTerm():
                if term.axis_annotations:
                    return set(term.axis_annotations)
                name = term.name
                if name in sig._variables:
                    return {d.name for d in sig._variables[name].structural_variables}
                if name in sig._constants:
                    return {d.name for d in sig._constants[name].structural_variables}
                return set()
            case TermAxisSelect():
                return _term_dim_names_for_definition(term.base_term) - {term.axis}
            case AtomicFormula():
                arg_dims: set[str] = set()
                for a in term.args:
                    arg_dims |= _term_dim_names_for_definition(a)
                sname = term.pred_name
                axes = set(term.structural_axes) if term.structural_axes else None
                if sname in sig._predicates:
                    pr = sig._predicates[sname]
                    consumed = {d.name for d in pr.input_dims}
                    out_names = axes if axes is not None else {d.name for d in pr.output_dims}
                    return (arg_dims - consumed) | out_names
                if sname in definitions:
                    defn = definitions[sname]
                    consumed = {d.name for d in defn.input_dims}
                    out_names = axes if axes is not None else {d.name for d in defn.output_dims}
                    return (arg_dims - consumed) | out_names
                return arg_dims if axes is None else arg_dims | axes
            case _CallTerm():
                arg_dims: set[str] = set()
                for a in term.args:
                    arg_dims |= _term_dim_names_for_definition(a)
                sname = term.symbol
                if sname in sig._functions:
                    fn = sig._functions[sname]
                    consumed = {d.name for d in fn.input_dims}
                    out_names = set(term.output_axes) if term.output_axes is not None else {d.name for d in fn.output_dims}
                    return (arg_dims - consumed) | out_names
                if sname in sig._predicates:
                    pr = sig._predicates[sname]
                    consumed = {d.name for d in pr.input_dims}
                    out_names = set(term.output_axes) if term.output_axes is not None else {d.name for d in pr.output_dims}
                    return (arg_dims - consumed) | out_names
                if sname in definitions:
                    defn = definitions[sname]
                    consumed = {d.name for d in defn.input_dims}
                    out_names = set(term.output_axes) if term.output_axes is not None else {d.name for d in defn.output_dims}
                    return (arg_dims - consumed) | out_names
                return arg_dims
            case _:
                return set()

    def _validate_definition_call(sym_name: str, defn: Any, args: list[Any]) -> None:
        if len(args) != len(defn.params):
            raise TypeError(
                f"Definition '{sym_name}' expects {len(defn.params)} argument(s), got {len(args)}."
            )

        if defn.input_sorts:
            for i, (arg, expected_sort) in enumerate(zip(args, defn.input_sorts)):
                actual_sort = _term_sort_name_for_definition(arg)
                if actual_sort is None:
                    raise TypeError(
                        f"Definition '{sym_name}': cannot infer sort of argument {i} "
                        f"to validate expected sort '{expected_sort.name}'."
                    )
                if actual_sort != expected_sort.name:
                    raise TypeError(
                        f"Definition '{sym_name}': argument {i} has sort '{actual_sort}', "
                        f"expected '{expected_sort.name}'."
                    )

        if defn.input_dims:
            present_axis_names: set[str] = set()
            for arg in args:
                present_axis_names |= _term_dim_names_for_definition(arg)
            present_dim_names: set[str] = set()
            for ax in present_axis_names:
                sv = sig._structural_variables.get(ax)
                if sv is not None:
                    present_dim_names.add(sv.dimension.name)
                    continue
                if ax in sig._dimensions:
                    present_dim_names.add(ax)
                    continue
                # Axis name not registered yet (rare): strip numeric/prime suffix.
                base = _base_name_of(ax)
                present_dim_names.add(base if base in sig._dimensions else ax)
            missing = [d.name for d in defn.input_dims if d.name not in present_dim_names]
            if missing:
                raise ValueError(
                    f"Definition '{sym_name}': missing required input structural dim(s) {missing}. "
                    f"Present axis names: {sorted(present_axis_names)} "
                    f"(resolved to dimensions: {sorted(present_dim_names)})."
                )

    def _validate_definition_output(sym_name: str, defn: Any, expanded: Any) -> None:
        if defn.output_sort is None:
            return
        match expanded:
            case TermFormula():
                actual_sort = _term_sort_name_for_definition(expanded.term)
            case Formula():
                actual_sort = "Bool"
            case Term():
                actual_sort = _term_sort_name_for_definition(expanded)
            case _:
                actual_sort = None

        if actual_sort is None:
            raise TypeError(
                f"Definition '{sym_name}': cannot infer output sort to validate expected "
                f"'{defn.output_sort.name}'."
            )
        if actual_sort != defn.output_sort.name:
            raise TypeError(
                f"Definition '{sym_name}': output sort is '{actual_sort}', "
                f"expected '{defn.output_sort.name}'."
            )

    def _parse_definition_body(defn: Any) -> Formula | Term:
        from sltn.parse.pipeline import parse_raw
        is_term = defn.output_sort is not None and defn.output_sort.name != "Bool"
        return parse_raw(defn.body, sig, "term" if is_term else "any")

    def expand(n: Any) -> Any:
        return _expand_definitions(n, definitions, sig)

    match node:
        case _NameTerm() | Variable() | Constant() | StructuralVariable():
            return node

        case TermAxisSelect():
            node.base_term = expand(node.base_term)
            return node

        case _CallTerm():
            node.args = [expand(a) for a in node.args]
            sym_name = node.symbol
            if sym_name in definitions:
                defn = definitions[sym_name]
                _validate_definition_call(sym_name, defn, node.args)
                body_ast = _parse_definition_body(defn)
                subst    = _definition_subst(defn, body_ast, node.args)
                pac      = _param_axis_canonical_for_definition(defn, body_ast)
                expanded = expand(_substitute(body_ast, subst, param_axis_canonical=pac))
                if not isinstance(expanded, (Formula, Term)):
                    raise TypeError(
                        f"Definition '{sym_name}' is used in term position, "
                        f"but its body expanded to {type(expanded).__name__}, "
                        f"which is neither a term nor a formula."
                    )
                _validate_definition_output(sym_name, defn, expanded)
                return expanded
            return node

        case AtomicFormula():
            new_args = [expand(a) for a in node.args]
            sym_name = node.symbol if isinstance(node.symbol, str) else node.symbol.name
            if sym_name in definitions:
                defn = definitions[sym_name]
                _validate_definition_call(sym_name, defn, new_args)
                body_ast = _parse_definition_body(defn)
                subst    = _definition_subst(defn, body_ast, new_args)
                pac      = _param_axis_canonical_for_definition(defn, body_ast)
                expanded = _substitute(body_ast, subst, param_axis_canonical=pac)
                if isinstance(expanded, Term):
                    expanded = TermFormula(expanded)
                _validate_definition_output(sym_name, defn, expanded)
                return expand(expanded)
            if new_args != list(node.args):
                node = AtomicFormula(node.symbol, *new_args, structural_axes=node.structural_axes)
            return node

        case TermFormula():
            node.term = expand(node.term)
            return node

        case FormulaAxisSelect():
            node.base_formula = expand(node.base_formula)
            return node

        case And() | Or() | Implies() | Equivalent():
            node.left  = expand(node.left)
            node.right = expand(node.right)
            return node

        case Not():
            node.operand = expand(node.operand)
            return node

        case Forall() | Exists():
            node.formula = expand(node.formula)
            if node.guard is not None:
                node.guard = expand(node.guard)
            return node

        case KB():
            node.clauses = [expand(c) for c in node.clauses]
            return node

        case _:
            return node
