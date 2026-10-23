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
from typing import Any, Callable

from sltn.fol.term import (
    _NameTerm, Variable, Constant, StructuralVariable, 
    _CallTerm, FunctionApplication, TermAxisSelect, Term
)
from sltn.fol.formula import (
    AtomicFormula,
    And, Or, Not, Implies, Equivalent,
    Forall, Exists,
    KB,
    TermFormula, FormulaAxisSelect,
    Formula
)
from sltn.parse.tokenizer import _base_name_of


# ---------------------------------------------------------------------------
# Definition expansion
# ---------------------------------------------------------------------------

def _build_numbered_alias_subst(
    param_names: list[str],
    args: list[Any],
    declared: set[str] = frozenset(),
) -> dict[str, Any]:
    """Build a substitution dict that also maps numbered aliases.

    For each formal parameter ``P`` bound to argument ``a``:

    * If ``a`` is a ``_NameTerm("X")``, then ``"P1"`` → ``_NameTerm("X1")``,
      ``"P2"`` → ``_NameTerm("X2")``, etc.
    * Otherwise (constant, complex term), ``"P1"`` … ``"P9"`` → ``a``.

    Names in *declared* are never treated as aliases.

    The numbered aliases produce *different* variable names so that
    ``U * U1`` in a definition body creates a cross-product rather than
    an element-wise product.
    """
    subst: dict[str, Any] = dict(zip(param_names, args))
    aliases: dict[str, Any] = {}
    for param, arg in subst.items():
        for n in range(1, 10):
            alias = f"{param}{n}"
            if alias not in subst and alias not in declared:
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
        # Children first: an inner binder of dst must be renamed before its outer one.
        for child in _child_nodes(n):
            visit(child)
        if isinstance(n, (Forall, Exists)) and dst in n.var_names:
            fresh = _fresh_structural_axis(_collect_all_structural_axis_strings(repl))
            n._var_groups = [[fresh if v == dst else v for v in group] for group in n._var_groups]
            _rename_structural_axes_inplace(n.formula, {dst: fresh})
            if n.guard is not None:
                _rename_structural_axes_inplace(n.guard, {dst: fresh})

    visit(repl)


def _rename_structural_axes_inplace(node: Any, mapping: dict[str, str]) -> None:
    """Rename structural axis name strings inside *node* (mutate in place)."""
    if not mapping:
        return

    def m(ax: str) -> str:
        return mapping.get(ax, ax)

    def visit(n: Any) -> None:
        match n:
            case _NameTerm() | Variable() | Constant() | StructuralVariable():
                if n.axis_annotations:
                    n.axis_annotations[:] = [m(a) for a in n.axis_annotations]
            case _CallTerm() if n.output_axes is not None:
                n.output_axes = [m(a) for a in n.output_axes]
            case AtomicFormula() if n.structural_axes is not None:
                n.structural_axes[:] = [m(a) for a in n.structural_axes]
            case TermAxisSelect() | FormulaAxisSelect():
                n.axis = m(n.axis)

    _walk_ast(node, visit)


def _declared_names(sig: Any) -> set[str]:
    """Names declared explicitly in *sig* (auto-registered aliases excluded)."""
    names = {n for n in (*sig.variables, *sig.structural_variables) if sig.alias_base(n) is None}
    return names | set(sig.constants) | set(sig.functions) | set(sig.predicates) \
        | set(sig.dimensions) | set(sig.structural_relations)


def _definition_subst(
    defn: Any, args: list[Any], canonical: dict[str, tuple[str, ...]], sig: Any,
) -> dict[str, Any]:
    """Build substitution for a definition call, including structural-axis renames.

    When the template body uses ``X[s, …]`` but the call site supplies ``x[s1, …]``,
    the axis names of the canonical template (``s``) are mapped to the call-site ones
    (``s1``), so annotations and selections in the body follow the call.
    """
    subst = _build_numbered_alias_subst(defn.params, args, _declared_names(sig))

    for pname, arg in zip(defn.params, args):
        actual_axes = _term_axis_annotations_for_subst(arg)
        formal_axes = canonical.get(pname)
        if not actual_axes or formal_axes is None or len(formal_axes) != len(actual_axes):
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


def _names_in(node: Any) -> set[str]:
    """Every symbol name and bound name occurring in *node*."""
    found: set[str] = set()

    def on_node(n: Any) -> None:
        match n:
            case _NameTerm() | Variable() | Constant() | StructuralVariable():
                found.add(n.name)
            case Forall() | Exists():
                found.update(n.var_names)

    _walk_ast(node, on_node)
    return found


def _fresh_name(base: str, used: set[str]) -> str:
    cand = base + "'"
    while cand in used:
        cand += "'"
    return cand


def _substitute(
    node: Any,
    subst: dict[str, Any],
    *,
    param_axis_canonical: dict[str, tuple[str, ...]] | None = None,
    avoid: frozenset[str] = frozenset(),
) -> Any:
    """Syntactically substitute symbol names from *subst* in a formula/term AST.

    Replaces every symbol reference whose logical name is a key in *subst* with
    ``subst[name]``.  Structural axis name strings in annotations and selections are
    also substituted via :func:`_substitute_axis_name`.  Used to instantiate a
    definition body with actual call-site arguments.

    A leaf replacement (``x`` or ``x[t]``) takes the body's annotations, substituted,
    so ``X[s1]`` with ``s → t`` gives ``x[s1]`` and ``X[s]`` gives ``x[t]``.
    *param_axis_canonical* maps parameter names to their canonical axis tuple; it is
    used to rename the axes of compound replacements (e.g. ``g(x)[s]``).

    *avoid* holds individual variable names free in the arguments: a quantifier in the
    body binding one of them is renamed (``y`` → ``y'``) so the argument isn't captured.
    """
    def sub(n: Any, s: dict[str, Any] = subst) -> Any:
        return _substitute(n, s, param_axis_canonical=param_axis_canonical, avoid=avoid)

    match node:
        case _NameTerm() | Variable() | Constant() | StructuralVariable():
            name = node.name
            if name not in subst:
                return node
            repl = subst[name]
            if not (isinstance(node, (_NameTerm, Variable, Constant)) and node.axis_annotations):
                return repl
            ann = tuple(node.axis_annotations)
            if isinstance(repl, (_NameTerm, Variable, Constant, StructuralVariable)):
                own = getattr(repl, "axis_annotations", None)
                if not own or len(own) == len(ann):
                    repl = copy.deepcopy(repl)
                    repl.axis_annotations = [_substitute_axis_name(a, subst) for a in ann]
                return repl
            canon = (param_axis_canonical or {}).get(name)
            if canon is not None and len(canon) == len(ann):
                mapping = {c: a for c, a in zip(canon, ann) if c != a}
                if mapping:
                    repl = copy.deepcopy(repl)
                    for dst in set(mapping.values()):
                        _resolve_dst_clash_with_struct_quantifiers(repl, dst)
                    _rename_structural_axes_inplace(repl, mapping)
            return repl

        case _CallTerm():
            new_output_axes = (
                [_substitute_axis_name(ax, subst) for ax in node.output_axes]
                if node.output_axes is not None else None
            )
            return _CallTerm(node.symbol, [sub(a) for a in node.args], output_axes=new_output_axes)

        case TermAxisSelect():
            return TermAxisSelect(sub(node.base_term), node.index, _substitute_axis_name(node.axis, subst))

        case AtomicFormula():
            new_structural_axes = (
                [_substitute_axis_name(ax, subst) for ax in node.structural_axes]
                if node.structural_axes is not None else None
            )
            return AtomicFormula(node.symbol, *[sub(a) for a in node.args],
                                 structural_axes=new_structural_axes)

        case TermFormula():
            return TermFormula(sub(node.term))

        case FormulaAxisSelect():
            return FormulaAxisSelect(sub(node.base_formula),
                                     axis=_substitute_axis_name(node.axis, subst), index=node.index)

        case And() | Or() | Implies() | Equivalent():
            return type(node)(sub(node.left), sub(node.right))

        case Not():
            return Not(sub(node.operand))

        case Forall() | Exists():
            rename: dict[str, str] = {}
            for v in node.var_names:
                if v in subst and isinstance(subst[v], (_NameTerm, Variable, Constant, StructuralVariable)):
                    rename[v] = subst[v].name
            non_renamed_bound = {v for v in node.var_names if v not in rename}
            inner = {k: v for k, v in subst.items() if k not in non_renamed_bound}
            captured = non_renamed_bound & avoid
            if captured:
                used = _names_in(node) | set(avoid)
                for v in sorted(captured):
                    fresh = _fresh_name(v, used)
                    used.add(fresh)
                    rename[v] = fresh
                    inner[v] = _NameTerm(fresh)

            new_var_groups = [[rename.get(v, v) for v in group] for group in node._var_groups]
            new_guard = sub(node.guard, inner) if node.guard is not None else None
            return type(node)(new_var_groups, sub(node.formula, inner), guard=new_guard)

        case KB():
            return KB([sub(c) for c in node.clauses],
                      validate=False)   # names are not resolved yet; the pipeline validates the KB

        case _:
            return node


def _apply_call_axes(root: Any, axes: list[str], sym_name: str) -> None:
    """Put the call-site output annotation ``D(…)[axes]`` on the expanded body's root."""
    match root:
        case TermFormula():
            return _apply_call_axes(root.term, axes, sym_name)
        case AtomicFormula():
            current, attr = root.structural_axes, "structural_axes"
        case _CallTerm():
            current, attr = root.output_axes, "output_axes"
        case _:
            raise ValueError(
                f"Definition '{sym_name}' is called with output axes {axes}, but its body "
                f"is a {type(root).__name__}, not a single predicate/function application."
            )
    if current is not None and len(current) != len(axes):
        raise ValueError(
            f"Definition '{sym_name}' is called with output axes {axes}, but its body "
            f"already names {len(current)} output axis/axes {list(current)}."
        )
    setattr(root, attr, axes)


# ---------------------------------------------------------------------------
# Sort and axis inference on the unresolved AST (to validate definition calls)
# ---------------------------------------------------------------------------

def _call_target(name: str, sig: Any) -> Any:
    """The function, predicate or definition called *name*, or ``None``."""
    return sig.functions.get(name) or sig.predicates.get(name) or sig.definitions.get(name)


def _sort_name(term: Any, sig: Any) -> str | None:
    """The sort of *term*, or ``None`` if it can't be inferred."""
    match term:
        case TermFormula():
            return _sort_name(term.term, sig)
        case TermAxisSelect():
            return _sort_name(term.base_term, sig)
        case AtomicFormula() | _CallTerm():
            name = term.pred_name if isinstance(term, AtomicFormula) else term.symbol
            if name in sig.predicates:
                return "Bool"
            target = _call_target(name, sig)
            out = target.output_sort if target is not None else None
            return out.name if out is not None else None
        case Constant() | Variable():
            return term.symbol.sort.name
        case StructuralVariable():
            sort = term.symbol.dimension.sort
            return sort.name if sort is not None else None
        case _NameTerm():
            sym = sig.constants.get(term.name) or sig.variables.get(term.name)
            return sym.sort.name if sym is not None else None
    return None


def _axis_names(term: Any, sig: Any) -> set[str]:
    """The structural axis names of *term*."""
    match term:
        case TermAxisSelect():
            return _axis_names(term.base_term, sig) - {term.axis}
        case AtomicFormula() | _CallTerm():
            if isinstance(term, AtomicFormula):
                name, axes = term.pred_name, term.structural_axes
            else:
                name, axes = term.symbol, term.output_axes
            arg_axes = set().union(*(_axis_names(a, sig) for a in term.args))
            target = _call_target(name, sig)
            if target is None:
                return arg_axes | set(axes or ())
            consumed = {d.name for d in target.input_dims}
            kept = {a for a in arg_axes if _axis_dimension_name(a, sig) not in consumed}
            return kept | (set(axes) if axes else {d.name for d in target.output_dims})
        case StructuralVariable():
            return {term.name}
        case Variable() | Constant() | _NameTerm():
            if term.axis_annotations:
                return set(term.axis_annotations)
            sym = term.symbol if isinstance(term, (Variable, Constant)) else (
                sig.variables.get(term.name) or sig.constants.get(term.name))
            return {d.name for d in sym.structural_variables} if sym is not None else set()
    return set().union(*(_axis_names(c, sig) for c in _child_nodes(term)))


def _axis_dimension_name(axis: str, sig: Any) -> str:
    """The dimension of *axis* (a structural variable or alias); *axis* itself if unknown."""
    dim = sig.dimension_of(axis)
    return dim.name if dim is not None else axis


# ---------------------------------------------------------------------------
# Definition instantiation
# ---------------------------------------------------------------------------

def _validate_definition_call(sym_name: str, defn: Any, args: list[Any], sig: Any) -> None:
    if len(args) != len(defn.params):
        raise TypeError(
            f"Definition '{sym_name}' expects {len(defn.params)} argument(s), got {len(args)}."
        )

    for i, (arg, expected_sort) in enumerate(zip(args, defn.input_sorts)):
        actual_sort = _sort_name(arg, sig)
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
        present_axis_names = set().union(*(_axis_names(a, sig) for a in args))
        present_dim_names = {_axis_dimension_name(ax, sig) for ax in present_axis_names}
        missing = [d.name for d in defn.input_dims if d.name not in present_dim_names]
        if missing:
            raise ValueError(
                f"Definition '{sym_name}': missing required input structural dim(s) {missing}. "
                f"Present axis names: {sorted(present_axis_names)} "
                f"(resolved to dimensions: {sorted(present_dim_names)})."
            )


def _validate_definition_output(sym_name: str, defn: Any, expanded: Any, sig: Any) -> None:
    if defn.output_sort is None:
        return
    is_formula = isinstance(expanded, Formula) and not isinstance(expanded, TermFormula)
    actual_sort = "Bool" if is_formula else _sort_name(expanded, sig)
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


def _is_individual(name: str, sig: Any) -> bool:
    if name in sig.structural_variables or name in sig.dimensions:
        return False
    return name in sig.variables or _base_name_of(name) in sig.variables


def _instantiate(sym_name: str, defn: Any, args: list[Any], axes: list[str] | None, sig: Any) -> Any:
    """The body of *defn* with *args* substituted for its parameters."""
    from sltn.parse.pipeline import parse_raw
    _validate_definition_call(sym_name, defn, args, sig)
    is_term = defn.output_sort is not None and defn.output_sort.name != "Bool"
    body_ast = parse_raw(defn.body, sig, "term" if is_term else "any")
    pac = _param_axis_canonical_for_definition(defn, body_ast)
    subst = _definition_subst(defn, args, pac, sig)
    avoid = frozenset(n for a in args for n in _names_in(a) if _is_individual(n, sig))
    expanded = _substitute(body_ast, subst, param_axis_canonical=pac, avoid=avoid)
    if axes is not None:
        _apply_call_axes(expanded, list(axes), sym_name)
    return expanded


def _expand_definitions(node: Any, sig: Any) -> Any:
    """Recursively expand the definitions of *sig* (syntactic abbreviations) in a formula/term AST.

    - **Formula level**: an :class:`~sltn.fol.formula.AtomicFormula` whose predicate
      name is a definition is replaced by the instantiated body.
    - **Term level**: a :class:`~sltn.fol.term._CallTerm` whose function name is a
      definition is replaced by the instantiated body.

    Expansion is recursive so definitions may reference other definitions.
    """
    definitions = sig.definitions

    def expand(n: Any) -> Any:
        return _expand_definitions(n, sig)

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
                expanded = expand(_instantiate(sym_name, defn, node.args, node.output_axes, sig))
                _validate_definition_output(sym_name, defn, expanded, sig)
                return expanded
            return node

        case AtomicFormula():
            new_args = [expand(a) for a in node.args]
            sym_name = node.symbol if isinstance(node.symbol, str) else node.symbol.name
            if sym_name in definitions:
                defn = definitions[sym_name]
                expanded = _instantiate(sym_name, defn, new_args, node.structural_axes, sig)
                if isinstance(expanded, Term):
                    expanded = TermFormula(expanded)
                _validate_definition_output(sym_name, defn, expanded, sig)
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
