"""
sltn.interpretation — Interpretation context for evaluating formulas.

An :class:`Interpretation` is the authoritative source of all symbol→value
bindings for a :class:`~sltn.signature.Signature`.  It also bundles logic
operators and provides :meth:`Interpretation.__call__` to numerically
evaluate any formula or term.
"""
from __future__ import annotations

__all__ = [
    "TensorCallable",
    "Interpretation",
    "Assignment",
]

import copy
from typing import Any, Dict, Optional

import torch

# ---------------------------------------------------------------------------
# SLTN Imports
# ---------------------------------------------------------------------------
from sltn.interpretation.core.axis import Axis
from sltn.interpretation.core.type import Type, Bool, Real
from sltn.interpretation.core.tensor import Tensor

from sltn.fol.term import (
    _NameTerm, Variable, Constant, StructuralVariable,
    FunctionApplication, TermAxisSelect, Term,
)
from sltn.fol.formula import (
    Formula, TermFormula, FormulaAxisSelect, AtomicFormula,
    StructuralRelation,
    And, Or, Implies, Equivalent, Not, _BoundFormula, KB,
)

from sltn.parse.pipeline import _base_name_of
from sltn.signature import (
    Signature, Symbol, FunctionSymbol, PredicateSymbol, StructuralRelationSymbol
)

from sltn.interpretation.apply import (
    apply_connective, apply_quantifier, apply_sat_agg,
    _var_axis_names, _deduplicate_axis_names,
    TensorCallable, _apply_callable
)

from sltn.interpretation.operators.base import AggregationOperator
from sltn.interpretation.logic import Logic, _lookup_kind

# A mapping of free variable names → tensors or annotated sltn.Tensors.
# Plain tensors are auto-wrapped into annotated sltn.Tensors inside _normalize_assignment.
Assignment = Dict[str, torch.Tensor | Tensor]

# ---------------------------------------------------------------------------
# Symbol → sltn.Tensor wrapping helpers (formerly ground() on each symbol class)
# ---------------------------------------------------------------------------

def _ground_constant(sym, val: torch.Tensor, sort_type: Optional[Type]) -> Tensor:
    """Wrap a tensor as an annotated sltn.Tensor for a ConstantSymbol symbol."""
    if sort_type is None:
        raise ValueError(f"Sort '{sym.sort.name}' is not grounded with a type.")
    struct_names = _deduplicate_axis_names(sym.structural_variables)
    axes: list[Axis] = [Axis(n, 'structural') for n in struct_names]
    axes += [Axis(n, 'domain') for n in sort_type.axis_names]
    return Tensor(value=val, axes=axes, domain_type=sort_type)


def _ground_variable(sym, val: torch.Tensor, sort_type: Optional[Type]) -> Tensor:
    """Wrap a tensor as an annotated sltn.Tensor for a VariableSymbol symbol."""
    if sort_type is None:
        raise ValueError(f"Sort '{sym.sort.name}' is not grounded with a type.")
    struct_names = _deduplicate_axis_names(sym.structural_variables)
    axes: list[Axis] = [Axis(sym.name, 'variable')]
    axes += [Axis(n, 'structural') for n in struct_names]
    axes += [Axis(n, 'domain') for n in sort_type.axis_names]
    return Tensor(value=val, axes=axes, domain_type=sort_type)


def _ground_structural_variable(sym, val: torch.Tensor, sort_type: Optional[Type]) -> Tensor:
    """Wrap a tensor as an annotated sltn.Tensor for a StructuralVariableSymbol symbol."""
    axes: list[Axis] = [Axis(sym.name, 'structural')]
    if sort_type is not None:
        axes.extend(sort_type.domain_axes)
        return Tensor(value=val, axes=axes, domain_type=sort_type)
    for i in range(1, val.ndim):
        axes.append(Axis(f'{sym.dimension.name}_{i - 1}', 'domain'))
    return Tensor(value=val, axes=axes, domain_type=None)


def _ground_structural_relation(sym, relation: torch.Tensor) -> Tensor:
    """Wrap the relation tensor as an annotated sltn.Tensor for a StructuralRelationSymbol symbol."""
    unique_names = _deduplicate_axis_names(sym.structural_variables)
    axes = [Axis(n, 'structural') for n in unique_names] + [Axis('bool', 'domain')]
    return Tensor(value=relation.unsqueeze(-1), axes=axes, domain_type=Bool)


def _resolve_sort_type(sort, interp) -> Optional[Type]:
    """Look up the tensor Type for *sort* from the interpretation's overrides."""
    grounding = interp._get_grounding(sort.name)
    if isinstance(grounding, Type):
        return grounding
    return {"Bool": Bool, "Real": Real}.get(sort.name)


def _value_to_expression(
    val: torch.Tensor | Tensor,
    name: str,
    sig: Signature,
    device: Optional[torch.device],
    interp=None,
) -> Tensor:
    """Convert a Tensor binding into a properly annotated sltn.Tensor."""
    if isinstance(val, Tensor):
        return val
    if isinstance(val, torch.Tensor):
        if device is not None:
            val = val.to(device)
        sv_obj = sig._structural_variables.get(name)
        if sv_obj is not None:
            sort_type = (
                _resolve_sort_type(sv_obj.dimension.sort, interp)
                if sv_obj.dimension.sort is not None and interp is not None
                else None
            )
            return _ground_structural_variable(sv_obj, val, sort_type)
        sym_obj = sig._variables.get(name)
        if sym_obj is not None:
            sort_type = _resolve_sort_type(sym_obj.sort, interp) if interp is not None else None
            return _ground_variable(sym_obj, val, sort_type)
        const_obj = sig._constants.get(name)
        if const_obj is not None:
            sort_type = _resolve_sort_type(const_obj.sort, interp) if interp is not None else None
            return _ground_constant(const_obj, val, sort_type)
        return Tensor.as_variable(name, val)
    raise TypeError(
        f"Binding for '{name}' must be a Tensor or annotated sltn.Tensor, "
        f"got {type(val).__name__}."
    )


def _apply_axis_renames(axis_annotations: list[str], val: Tensor) -> Tensor:
    """Rename structural axes when ``x[t1, …]`` syntax supplies explicit names."""
    if not axis_annotations:
        return val
    struct_axes = val.structural_axes
    renames = {
        d.name: ann
        for d, ann in zip(struct_axes, axis_annotations)
        if d.name != ann
    }
    return val.rename_axes(renames) if renames else val



# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

class Interpretation:
    """
    A logical interpretation of a Signature.

    Binds concrete values to symbols (constants, variables, functions,
    predicates) and holds a fuzzy logic operator bundle.

    Parameters
    ----------
    signature : Signature
        The vocabulary this interpretation is for.
    overrides : dict, optional
        Initial groundings: ``{name_or_symbol: value}``.  Keys can be strings
        or Symbol objects.
    **kwargs
        Short-hand initial grounding entries (name → value).

    The logic operators are set via the :attr:`logic` property after
    construction.
    """

    def __init__(
        self,
        signature: Signature,
        overrides: Optional[Dict[str | Symbol, Any]] = None,
        parent: Optional["Interpretation"] = None,
        logic: Optional["Logic"] = None,
        **kwargs: Any,
    ) -> None:
        self.signature = signature
        self._device: Optional[torch.device] = None
        self._parent: Optional["Interpretation"] = parent
        self._overrides: Dict[str, Any] = {}

        self._logic: Logic = (logic.with_defaults() if logic is not None else Logic().with_defaults())
        self._logic._signature = self.signature

        # Per-call evaluation cache; active only during Interpretation.__call__
        self._eval_cache: dict | None = None

        # Toggle memoisation on/off (default: on).  Set to False to disable
        self.use_cache: bool = True

        # When True, a snapshot of the per-call cache is saved to _last_cache
        # after each interp(formula) call so it can be inspected via
        # print_cache().  Defaults to False — no persistent tensor references
        # between calls.
        self.save_cache_snapshot: bool = False

        # Snapshot of the cache from the most recent evaluation call.
        # Populated only when save_cache_snapshot=True.
        self._last_cache: dict | None = None

        # Classical-mode state: threshold flag and saved fuzzy logic for restore.
        self._classical: bool = False
        self.classical_threshold: float = 0.5
        self._pre_classical_logic: Optional[Logic] = None

        self._ground_all({**(overrides or {}), **kwargs})

    def _ground_all(self, items: Dict[str | Symbol, Any]) -> None:
        """Ground every entry of *items*, sorts first (tensor shape checks need their Type)."""
        def is_sort(k: str | Symbol) -> bool:
            return (k.name if isinstance(k, Symbol) else k) in self.signature._sorts
        for k, v in sorted(items.items(), key=lambda kv: not is_sort(kv[0])):
            self.ground(k, v)

    def _derive(self, overrides: Dict[str, Any], parent: Optional["Interpretation"],
                source: Optional["Interpretation"] = None) -> "Interpretation":
        """New interpretation grounded with *overrides*, with logic, classical mode and
        device copied from *source* (default: self). The Logic is copied, not shared."""
        source = source or self
        new = Interpretation(self.signature, parent=parent)
        new._device = source._device
        new.logic = source._logic
        new._classical = source._classical
        new.classical_threshold = source.classical_threshold
        if source._pre_classical_logic is not None:
            new._pre_classical_logic = source._pre_classical_logic.with_defaults()
            new._pre_classical_logic._signature = self.signature
        new._ground_all(overrides)
        return new

    # ------------------------------------------------------------------
    # Logic property
    # ------------------------------------------------------------------

    @property
    def logic(self) -> Logic:
        """The fuzzy operator bundle for this Interpretation.

        Assigning a :class:`Logic` object fills any unset fields with
        sensible defaults; individual operators can also be mutated directly.
        """
        return self._logic

    @logic.setter
    def logic(self, value: Logic) -> None:
        if self._classical:
            # Keep classical semantics active; *value* is used once classical mode is off.
            self._pre_classical_logic = value.with_defaults()
            self._pre_classical_logic._signature = self.signature
            struct_axes = list(value.StructAgg) if value.StructAgg else None
            self._logic = Logic.classical(struct_agg_axes=struct_axes)
        else:
            self._logic = value.with_defaults()
        self._logic._signature = self.signature


    # ------------------------------------------------------------------
    # Classical mode
    # ------------------------------------------------------------------

    @property
    def classical(self) -> bool:
        """When ``True``, the interpretation uses strictly Boolean semantics.
        Setting it back to ``False`` restores the previously saved logic.
        """
        return self._classical

    @classical.setter
    def classical(self, value: bool) -> None:
        if value and not self._classical:
            self._pre_classical_logic = self._logic
            struct_axes = list(self._logic.StructAgg) if self._logic.StructAgg else None
            self._logic = Logic.classical(struct_agg_axes=struct_axes)
            self._logic._signature = self.signature
        elif not value and self._classical and self._pre_classical_logic is not None:
            self._logic = self._pre_classical_logic
            self._pre_classical_logic = None
        self._classical = value

    # ------------------------------------------------------------------
    # Binding interface
    # ------------------------------------------------------------------

    def _has_grounding(self, name: str) -> bool:
        """Return True if *name* is bound in this Interpretation or any parent."""
        if name in self._overrides:
            return True
        # Check alias: a variable/StructuralVariableSymbol alias is satisfied if its base is bound
        base = (self.signature._variables_aliases.get(name)
                or self.signature._structural_variables_aliases.get(name))
        if base is not None and base in self._overrides:
            return True
        if self._parent is not None:
            return self._parent._has_grounding(name)
        return False

    def _get_grounding(self, name: str) -> Any:
        """Return the grounding for *name*, falling back to parent chain. Returns None if not found."""
        if name in self._overrides:
            return self._overrides[name]
        if self._parent is not None:
            return self._parent._get_grounding(name)
        return None

    def ground(self, sym: str | Symbol, value: Any) -> Interpretation:
        """Bind *value* to *sym* in this Interpretation.

        Parameters
        ----------
        sym : str | Symbol
            Symbol name or object.
        value :
            - ``torch.nn.Module`` or callable for FunctionSymbol/PredicateSymbol symbols
            - ``torch.Tensor`` or annotated ``sltn.Tensor`` for Constants/Variables
            - ``torch.Tensor`` for StructuralRelationSymbol relation matrices

        Returns
        -------
        self
            Fluent: allows chaining, e.g. ``interp.ground(P, m).ground(x, t)``.
        """
        key = sym.name if isinstance(sym, Symbol) else sym
        try:
            self.signature.get(key)
        except KeyError:
            raise KeyError(
                f"Cannot ground '{key}': no symbol with this name is declared in "
                f"signature '{self.signature.name}'."
            ) from None
        base = (self.signature._variables_aliases.get(key)
                or self.signature._structural_variables_aliases.get(key))
        if base is not None:
            raise KeyError(
                f"Cannot ground '{key}': it is an alias of '{base}' and shares its "
                f"grounding. Ground '{base}' instead."
            )
        if self._device is not None and isinstance(value, (torch.nn.Module, torch.Tensor)):
            value = value.to(self._device)
        if isinstance(value, torch.Tensor):
            self._check_tensor_shape(key, value)
        self._overrides[key] = value
        return self

    def _check_tensor_shape(self, key: str, tensor: torch.Tensor) -> None:
        """Raise ``ValueError`` if *tensor*'s feature shape does not match the
        sort's declared ``Type`` shape.

        For a ``VariableSymbol`` the tensor layout is ``[batch, *feature_dims]``;
        for a ``ConstantSymbol`` it is ``[*feature_dims]`` (or ``[1, *feature_dims]``
        for a single individual in batch form).  In both cases only the
        trailing ``len(type.shape)`` dimensions are checked.
        """
        sig = self.signature
        sym_obj = sig._constants.get(key) or sig._variables.get(key)
        struct_var_obj = sig._structural_variables.get(key)
        if sym_obj is None and struct_var_obj is None:
            return
        if sym_obj is not None:
            sort = sym_obj.sort
        else:
            sort = struct_var_obj.dimension.sort if struct_var_obj is not None else None
            if sort is None:
                return

        sort_type = self._get_grounding(sort.name)
        if not isinstance(sort_type, Type):
            sort_type = None
        if sort_type is None:
            return
        expected = tuple(sort_type.shape)
        n = len(expected)
        if tensor.ndim < n:
            raise ValueError(
                f"Shape mismatch for '{key}': sort '{sort.name}' expects feature "
                f"shape {expected}, but tensor has only {tensor.ndim} dimension(s) "
                f"(full shape: {tuple(tensor.shape)})."
            )
        actual = tuple(tensor.shape[-n:])
        if actual != expected:
            raise ValueError(
                f"Shape mismatch for '{key}': sort '{sort.name}' expects feature "
                f"shape {expected}, but tensor has trailing shape {actual} "
                f"(full shape: {tuple(tensor.shape)})."
            )

    def __setitem__(self, sym: str | Symbol, value: Any) -> None:
        """Shorthand for :meth:`ground`: ``interp[P] = Net()``."""
        self.ground(sym, value)

    def __getitem__(self, sym: str | Symbol) -> Any:
        """Return the raw grounding for *sym* (tensor, Module, or callable).

        Raises :class:`KeyError` if the symbol has no grounding yet.
        """
        key = sym.name if isinstance(sym, Symbol) else sym
        if not self._has_grounding(key):
            raise KeyError(
                f"Symbol '{key}' has no grounding in this Interpretation. "
                f"Use 'interp[{key}] = value' to ground it."
            )
        val = self._get_grounding(key)
        if val is None:   # an alias: return its base grounding
            base = (self.signature._variables_aliases.get(key)
                    or self.signature._structural_variables_aliases.get(key))
            val = self._get_grounding(base)
        return val

    def _normalize_assignment(self, raw: dict) -> Assignment:
        return {
            var_name: _value_to_expression(val, var_name, self.signature, self._device, self)
            for var_name, val in raw.items()
        }

    # ------------------------------------------------------------------
    # Central evaluator
    # ------------------------------------------------------------------

    def _eval(self, node: Any, asgn: Assignment) -> Tensor:
        """Evaluate *node* (Term or Formula) under this interpretation.

        Caching (keyed on node repr + assignment identity) is applied here
        when ``self._eval_cache`` is set by ``__call__``.
        """
        cache = self._eval_cache
        if cache is not None:
            node_repr = node.__dict__.get('_cache_repr')
            if node_repr is None:
                node_repr = repr(node)
                try:
                    node.__dict__['_cache_repr'] = node_repr
                except (AttributeError, TypeError):
                    pass
            asgn_key = frozenset((k, id(v)) for k, v in asgn.items())
            key = (node_repr, asgn_key)
            if key in cache:
                return cache[key]
            result = self._eval_node(node, asgn)
            cache[key] = result
            return result
        return self._eval_node(node, asgn)

    def _eval_node(self, node: Any, asgn: Assignment) -> Tensor:
        """Dispatch evaluation of a single AST node."""
        match node:
            case _NameTerm():
                raise RuntimeError(
                    f"_NameTerm({node.symbol!r}) was not resolved — "
                    "call Signature.parse() before evaluation."
                )
            case Variable() | StructuralVariable():
                n = node.name
                val = asgn[n] if n in asgn else self.get_variable_value(n)
                return _apply_axis_renames(node.axis_annotations, val)
            case Constant():
                n = node.name
                val = asgn[n] if n in asgn else self.get_constant_value(n)
                return _apply_axis_renames(node.axis_annotations, val)
            case FunctionApplication():
                evaluated = [self._eval(arg, asgn) for arg in node.args]
                fn = self.resolve(node.symbol, structural_axes_override=node.output_axes)
                return fn(*evaluated)
            case TermAxisSelect():
                return self._eval(node.base_term, asgn).select(node.axis, node.index)
            case TermFormula():
                return self._eval(node.term, asgn)
            case FormulaAxisSelect():
                return self._eval(node.base_formula, asgn).select(node.axis, node.index)
            case StructuralRelation():
                rel = self.get_structural_relation(node.pred_name)
                current = rel.axis_names
                renames = {old: new for old, new in zip(current, node.var_names) if old != new}
                return rel.rename_axes(renames) if renames else rel
            case AtomicFormula():
                return self._eval_atomic(node, asgn)
            case Not():
                return apply_connective(self.logic.Not, self._eval(node.operand, asgn))
            case And() | Or() | Implies() | Equivalent():
                return apply_connective(
                    getattr(self.logic, node.op_attr),
                    self._eval(node.left, asgn),
                    self._eval(node.right, asgn),
                )
            case KB():
                results = [self._eval(clause, asgn) for clause in node.clauses]
                return apply_sat_agg(self.logic.SatAgg, *results)
            case _BoundFormula():
                return self._eval_bound(node, asgn)
            case _:
                raise TypeError(
                    f"Interpretation._eval received an unknown node type: {type(node).__name__}"
                )

    def _eval_atomic(self, node: Any, asgn: Assignment) -> Tensor:
        sym = node.symbol
        if isinstance(sym, StructuralRelationSymbol):
            relTensor = self.get_structural_relation(sym.name)
            sv_names = [arg.name for arg in node.args]
            current  = relTensor.axis_names
            renames  = {old: new for old, new in zip(current, sv_names) if old != new}
            return relTensor.rename_axes(renames) if renames else relTensor
        fn = self.resolve(sym, structural_axes_override=node.structural_axes)
        evaluated = [self._eval(arg, asgn) for arg in node.args]
        return fn(*evaluated)

    def _eval_bound(self, node: Any, asgn: Assignment) -> Tensor:
        """Evaluate a Forall/Exists quantifier node."""
        assignment = dict(asgn)
        sig = self.signature

        # Pre-populate assignment for quantified variables bound in the interpretation.
        for vname in node.var_names:
            if vname not in assignment and self._has_grounding(vname):
                assignment[vname] = self.get_variable_value(vname)

        # Diagonalise joint variable groups.
        for group in node._var_groups:
            if len(group) > 1:
                present = [v for v in group if v in assignment]
                if len(present) > 1:
                    diag_vars = Tensor.diag(*[assignment[v] for v in present])
                    for v_name, var in zip(present, diag_vars):
                        assignment[v_name] = var

        # Individual and structural variables are never mixed in one quantifier
        # (rejected at parse time), and structural ones share one dimension.
        struct_var_names = [v for v in node.var_names if v in sig._structural_variables]
        quantifier = node.quantifier  # "forall" or "exists"
        symbol     = node.symbol      # "∀" or "∃"

        formula_result = self._eval(node.formula, assignment)
        guard_result   = self._eval(node.guard, assignment) if node.guard else None

        if struct_var_names:
            agg_op = self.get_struct_agg_op(struct_var_names[0], quantifier)
            axes: list[str] = []
            for sv_axis in struct_var_names:
                actual = sv_axis
                if sv_axis not in formula_result.axis_map:
                    dim_name = sig._structural_variables[sv_axis].dimension.name
                    if dim_name in formula_result.axis_map:
                        actual = dim_name
                axes.append(actual)
        else:
            agg_op = self.logic.Forall if quantifier == "forall" else self.logic.Exists
            axes = _var_axis_names([assignment[v] for v in node.var_names if v in assignment])
        return apply_quantifier(agg_op, formula_result, axes, mask=guard_result, symbol=symbol)

    def __call__(
        self,
        target: Any,
        assignment: Optional[Assignment] = None,
        at: Optional[Dict[str, int]] = None,
    ) -> Tensor | Any:
        """Evaluate *target* under this interpretation.

        *target* may be a string formula, a :class:`~sltn.fol.formula.Formula`,
        a :class:`~sltn.fol.term.Term`, a :class:`~sltn.signature.symbols.Symbol`,
        or an :class:`~sltn.interpretation.core.tensor.Tensor` (pass-through).

        *at* is an optional dict ``{axis_name: index}`` applied after
        evaluation: each entry calls ``result.select(axis_name, index)``.
        """

        # String: parse via the signature, then re-enter.
        if isinstance(target, str):
            target = self.signature.parse(target)
            return self(target, assignment=assignment, at=at)

        # annotated sltn.Tensor: pass-through.
        if isinstance(target, Tensor):
            return target

        # Symbol: return raw grounding.
        if isinstance(target, Symbol):
            return self[target]

        if not isinstance(target, (Term, Formula)):
            raise TypeError(
                f"interp() expects a Formula, Term, Symbol, or annotated sltn.Tensor; "
                f"got {type(target).__name__}."
            )

        asgn = self._normalize_assignment(assignment or {})

        # For Formulas, check for unbound free variables before evaluating.
        if isinstance(target, Formula):
            missing = {
                v for v in target.free_variables - set(asgn)
                if not self._has_grounding(v)
            }
            if missing:
                raise ValueError(
                    f"Formula has free variables {sorted(missing)} not bound in this "
                    f"Interpretation and not in the assignment dict. "
                    f"Use 'interp[var] = tensor' to ground them. "
                    f"Assignment keys: {sorted(asgn)}"
                )

        self._eval_cache = {} if self.use_cache else None
        try:
            result = self._eval(target, asgn)
        finally:
            if self._eval_cache is not None and self.save_cache_snapshot:
                self._last_cache = dict(self._eval_cache)
            self._eval_cache = None

        if at:
            for axis_name, index in at.items():
                result = result.select(axis_name, index)
        return result

    # ------------------------------------------------------------------
    # Context-manager support (optional scope marker; no hidden global state)
    # ------------------------------------------------------------------

    def __enter__(self) -> Interpretation:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    # ------------------------------------------------------------------
    # Device placement
    # ------------------------------------------------------------------

    def to(self, device: str | torch.device) -> Interpretation:
        """Move all bound tensors and modules to *device* and remember it
        for future :meth:`ground` / ``interp[sym] = value`` calls.

        Returns *self* for chaining.
        """
        device = torch.device(device)
        self._device = device
        for k, v in list(self._overrides.items()):
            if isinstance(v, (torch.nn.Module, torch.Tensor)):
                self._overrides[k] = v.to(device)
        return self

    @property
    def device(self) -> Optional[torch.device]:
        """The device this interpretation runs on."""
        return self._device

    # ------------------------------------------------------------------
    # Structural axis operator lookup
    # ------------------------------------------------------------------

    def _resolve_struct_axis(self, axis: str) -> str:
        """Return the canonical ``Dimension`` name for *axis*.

        Structural aggregators are grounded per-*dimension*: any structural
        variable ranging over that dimension — the base variable, an
        auto-created alias (``t1``, ``t2``, ...), or an independently
        declared variable — resolves to the same dimension name, so users
        only need one ``StructAgg`` entry per dimension.
        """
        sig = self.signature
        if sig is not None and hasattr(sig, "resolve_structural_axis"):
            return sig.resolve_structural_axis(axis)
        return axis

    def get_struct_agg_op(self, axis: str, kind: str) -> AggregationOperator:
        """Return the structural aggregation operator for *axis* and *kind* ('forall'/'exists')."""
        struct_agg = self.logic.StructAgg
        if struct_agg is None:
            raise ValueError(
                f"No structural aggregation operator registered for axis '{axis}'. "
                f"Set Logic(StructAgg={{'{self._resolve_struct_axis(axis)}': {{'{kind}': <AggregOp>, ...}}}})"
            )
        canonical = self._resolve_struct_axis(axis)
        # Dimension-level entries take precedence; also accept a literal entry
        # keyed by the variable name itself, for callers that key StructAgg by
        # a variable whose name happens to equal its dimension's.
        entry = struct_agg.get(canonical, struct_agg.get(axis))
        if entry is None:
            raise ValueError(
                f"No structural aggregation operator registered for axis '{axis}'. "
                f"Set Logic(StructAgg={{'{canonical}': {{'{kind}': <AggregOp>, ...}}}})"
            )
        value = _lookup_kind(entry, kind)
        if value is None:
            raise ValueError(
                f"Missing '{kind}' key in Logic.StructAgg['{canonical}']. "
                f"Add: Logic(StructAgg={{'{canonical}': {{'{kind}': <AggregOp>}}}})"
            )
        return value

    def get_structural_relation(self, name: str) -> Tensor:
        """Return the annotated sltn.Tensor for a **structural relation** by name.

        A *structural relation* is a truth-valued tensor indexed purely by
        structural (index-like) axes and carrying no variable axes.  It
        expresses relationships between structural indices — e.g. ``next(t, t1)``
        maps each time-step ``t`` to a truth value for each candidate successor
        ``t1``.  Structural relations are the grounding of
        :class:`~sltn.signature.StructuralRelationSymbol` symbols, bound the
        same way as any other symbol via ``interp[sp] = tensor``.
        """
        val = self._get_grounding(name)
        if val is None:
            raise KeyError(
                f"Structural relation '{name}' not found. "
                f"Bind via 'interp[{name}] = tensor'."
            )
        if isinstance(val, torch.Tensor):
            sp = self.signature._structural_relations.get(name)
            result = _ground_structural_relation(sp, val) if sp is not None else val
        else:
            result = val
        if isinstance(result, Tensor) and result.variable_axes:
            raise ValueError(
                f"Structural relation '{name}' must have no variable axes; "
                f"got {[a.name for a in result.variable_axes]}."
            )
        return result

    # ------------------------------------------------------------------
    # Symbol resolution
    # ------------------------------------------------------------------

    def resolve(self, sym: str | Symbol, structural_axes_override: list[str] | None = None) -> Any:
        """Return the effective callable for a FunctionSymbol or PredicateSymbol symbol.

        Parameters
        ----------
        sym : str | Symbol
            The symbol name or object.
        structural_axes_override : list[str] | None
            When given (from ``AtomicFormula.structural_axes``), these axis names
            replace the predicate's declared ``output_dims`` names for this call.
            An empty list means "produce no structural output axis".
            ``None`` means "use the predicate's declared output_dims".
        """
        name = sym.name if isinstance(sym, Symbol) else sym
        override_val = self._get_grounding(name)

        try:
            sym_obj = self.signature.get(name)
        except KeyError:
            raise KeyError(
                f"Symbol '{name}' not found in signature '{self.signature.name}'."
            )

        if isinstance(sym_obj, (FunctionSymbol, PredicateSymbol)):
            if override_val is None:
                raise RuntimeError(
                    f"Symbol '{name}' has no grounding in this Interpretation. "
                    f"Use 'interp[{name}] = callable' to ground it."
                )
            return _SymbolProxy(sym_obj, override_val, self,
                                structural_axes_override=structural_axes_override)

        raise TypeError(
            f"resolve() is for FunctionSymbol/PredicateSymbol symbols; "
            f"got {type(sym_obj).__name__} ('{name}'). "
            "Use get_constant_value() or get_variable_value() for non-callable symbols."
        )

    def get_constant_value(self, name: str) -> Tensor:
        """Return the annotated sltn.Tensor value for a ConstantSymbol symbol."""
        val = self._get_grounding(name)
        if val is None:
            raise RuntimeError(
                f"ConstantSymbol '{name}' has no grounding in this Interpretation. "
                f"Use 'interp[{name}] = tensor' to ground it."
            )
        try:
            sym_obj = self.signature.get_constant(name)
        except KeyError:
            raise KeyError(f"ConstantSymbol '{name}' not in signature '{self.signature.name}'.")
        sort_type = _resolve_sort_type(sym_obj.sort, self)
        return _ground_constant(sym_obj, val, sort_type)

    def get_variable_value(self, name: str) -> Tensor:
        """Return the annotated sltn.Tensor for a VariableSymbol from this Interpretation's groundings."""
        val = self._get_grounding(name)
        if val is None:
            # Try alias: if 'x\'' -> 'x', use base grounding and rename axis
            base_name = (self.signature._variables_aliases.get(name)
                         or self.signature._structural_variables_aliases.get(name))
            if base_name is not None:
                val = self._get_grounding(base_name)
                if val is None:
                    raise RuntimeError(
                        f"VariableSymbol '{name}' (alias of '{base_name}') has no grounding. "
                        f"Use 'interp[{base_name!r}] = tensor' to ground it."
                    )
                if isinstance(val, Tensor):
                    return val.rename_axes({base_name: name})
                return _value_to_expression(val, name, self.signature, self._device, self)
            raise RuntimeError(
                f"VariableSymbol '{name}' has no grounding in this Interpretation. "
                f"Pass it in the assignment dict or use 'interp[{name}] = tensor'."
            )
        return _value_to_expression(val, name, self.signature, self._device, self)

    # ------------------------------------------------------------------
    # Binding management
    # ------------------------------------------------------------------
    def with_updates(self, **overrides) -> Interpretation:
        """Return a shallow copy with additional/overridden groundings."""
        return self._derive({**self._overrides, **overrides}, self._parent)

    def copy(self) -> Interpretation:
        """Return a shallow copy sharing all tensor and module references."""
        return self.with_updates()

    def deepcopy(self) -> Interpretation:
        """Return a deep copy where all tensors and modules are independently cloned."""

        new_overrides: Dict[str, Any] = {}
        for k, v in self._overrides.items():
            if isinstance(v, torch.nn.Module):
                new_overrides[k] = copy.deepcopy(v)
            elif isinstance(v, torch.Tensor):
                new_overrides[k] = v.clone()
            else:
                new_overrides[k] = v
        return self._derive(new_overrides, self._parent)

    def __copy__(self) -> Interpretation:
        return self.copy()

    def __deepcopy__(self, memo: dict) -> Interpretation:
        result = self.deepcopy()
        memo[id(self)] = result
        return result

    # ------------------------------------------------------------------
    # Trainable parameters
    # ------------------------------------------------------------------

    def parameters(self):
        """Yield all trainable parameters from this Interpretation.

        Covers:
        - ``torch.nn.Module`` groundings (e.g. a neural predicate).
        - ``torch.nn.Parameter`` groundings (e.g. a learnable embedding constant
          or structural relation).
        """
        for val in self._overrides.values():
            if isinstance(val, torch.nn.Module):
                yield from val.parameters()
            elif isinstance(val, torch.nn.Parameter):
                yield val

    def named_parameters(self):
        """Yield ``(name, param)`` pairs from this Interpretation.

        Covers the same sources as :meth:`parameters`, with qualified names.
        """
        for key, val in self._overrides.items():
            if isinstance(val, torch.nn.Module):
                for param_name, param in val.named_parameters():
                    yield f"{key}.{param_name}", param
            elif isinstance(val, torch.nn.Parameter):
                yield key, val

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    @classmethod
    def merge(cls, a: Interpretation, b: Interpretation) -> Interpretation:
        """Merge two interpretations; *b* overrides *a*. Logic, classical mode and
        device come from *b* (device falls back to *a*'s)."""
        merged = b._derive({**a._overrides, **b._overrides}, b._parent or a._parent)
        if merged._device is None and a._device is not None:
            merged.to(a._device)
        return merged

    # ------------------------------------------------------------------
    # Cache inspection / management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Discard the snapshot tensors saved by the last evaluation.

        The per-call memoisation cache (``_eval_cache``) is torn down
        automatically at the end of every ``interp(formula)`` call.
        ``_last_cache`` is only populated when ``save_cache_snapshot=True``;
        call this method to release those tensors explicitly.
        """
        self._last_cache = None

    def print_cache(self) -> None:
        """Print the sub-formula cache captured during the last evaluation.

        Each line shows the formula string and the shape / value of the
        cached :class:`~sltn.interpretation.core.tensor.Tensor`.  Call this
        immediately after ``interp(formula)`` to inspect what was memoised.
        """
        cache = self._last_cache
        if cache is None:
            print("No cache snapshot available.  "
                  "Either use_cache=False or no formula has been evaluated yet.")
            return
        if not cache:
            print("Cache is empty (no sub-formula was memoised).")
            return

        print(f"Cache snapshot — {len(cache)} entr{'y' if len(cache) == 1 else 'ies'}:")
        for (formula_repr, asgn_key), expr in cache.items():
            # Render assignment: show variable name and object id
            if asgn_key:
                asgn_parts = ", ".join(
                    f"{k}=<id {v_id}>" for k, v_id in sorted(asgn_key)
                )
                asgn_str = f"  [{asgn_parts}]"
            else:
                asgn_str = ""
            val = getattr(expr, "value", expr)
            if isinstance(val, torch.Tensor):
                shape_str = f"Tensor{list(val.shape)}"
            else:
                shape_str = repr(val)
            print(f"  {formula_repr}{asgn_str}  →  {shape_str}")
            

    def __repr__(self) -> str:
        sig = self.signature
        lines = [f"Interpretation(signature={sig.name!r})"]

        def _val_label(val: Any) -> str:
            if val is None:
                return "(not grounded)"
            if isinstance(val, torch.nn.Module):
                n_params = sum(p.numel() for p in val.parameters())
                return f"{val.__class__.__name__}({n_params:,} params)"
            if isinstance(val, torch.Tensor):
                return f"Tensor{list(val.shape)}"
            if callable(val):
                n = getattr(val, "__name__", None) or val.__class__.__name__
                return f"λ {n}"
            return repr(val)

        # ── Sorts ─────────────────────────────────────────────────────
        if sig._sorts:
            lines.append("  Sorts:")
            for s_name in sig._sorts:
                typ = self._overrides.get(s_name)
                suffix = f" : {typ}" if typ is not None else ""
                lines.append(f"    {s_name}{suffix}")

        # ── Constants ─────────────────────────────────────────────────
        if sig._constants:
            lines.append("  Constants:")
            for c_name in sig._constants:
                val = self._overrides.get(c_name)
                lines.append(f"    {c_name}: {_val_label(val)}")

        # ── Variables ─────────────────────────────────────────────────
        if sig._variables:
            lines.append("  Variables:")
            for v_name in sig._variables:
                val = self._overrides.get(v_name)
                lines.append(f"    {v_name}: {_val_label(val)}")

        # ── Predicates ────────────────────────────────────────────────
        if sig._predicates:
            lines.append("  Predicates:")
            for p_name in sig._predicates:
                val = self._overrides.get(p_name)
                lines.append(f"    {p_name}: {_val_label(val)}")

        # ── Functions ─────────────────────────────────────────────────
        if sig._functions:
            lines.append("  Functions:")
            for f_name in sig._functions:
                val = self._overrides.get(f_name)
                lines.append(f"    {f_name}: {_val_label(val)}")

        # ── Logic ─────────────────────────────────────────────────────
        lines.append(f"  {self.logic}")

        return "\n".join(lines)




# ---------------------------------------------------------------------------
# Internal proxy for overriding the callable of a symbol
# ---------------------------------------------------------------------------

class _SymbolProxy:
    """Wraps a FunctionSymbol/PredicateSymbol with its grounding callable for one evaluation call.

    ``axes_override``:
      - ``None``  — use the symbol's declared output_dims
      - ``[]``    — strip all structural output (A(x) with no annotation)
      - ``[...]`` — rename output axes to annotation names (A(x)[t1])
    """

    def __init__(
        self,
        sym: FunctionSymbol | PredicateSymbol,
        override_callable: Any,
        interp: Interpretation,
        structural_axes_override: list[str] | None = None,
    ) -> None:
        self._sym      = sym
        self._fn       = override_callable
        self._is_pred  = isinstance(sym, PredicateSymbol)
        self._interp   = interp
        self._axes_override: list[str] | None = structural_axes_override

    def __call__(self, *inputs: Tensor) -> Tensor:
        preserved = self._preserved_axes(inputs)

        effective_out = (
            [d.name for d in self._sym.output_dims]
            if self._axes_override is None
            else list(self._axes_override)
        )

        # When annotation names are given, ensure the matching input axes are
        # not batched away by _apply_callable.
        if self._axes_override:
            sig = self._interp.signature
            dim_to_input: dict[str, str] = {}
            for inp in inputs:
                for axis in inp.structural_axes:
                    sv = sig._structural_variables.get(axis.name)
                    dname = sv.dimension.name if sv else axis.name
                    dim_to_input.setdefault(dname, axis.name)
            for decl_dim, _ in zip(self._sym.output_dims, effective_out):
                in_ax = dim_to_input.get(decl_dim.name)
                if in_ax is not None:
                    preserved.add(in_ax)

        # If every annotation axis already lives in the inputs (passthrough),
        # pass them directly so _apply_callable preserves the existing axes
        # rather than creating new ones and renaming later.
        all_input_struct = {
            ax for inp in inputs for ax in inp.free_structural_variables
        }
        if (effective_out
                and len(effective_out) == len(self._sym.output_dims)
                and all(n in all_input_struct for n in effective_out)):
            ax_names = list(effective_out)
            preserved.update(effective_out)
        else:
            ax_names = [d.name for d in self._sym.output_dims]

        result = self._call_symbol(inputs, preserved, ax_names)

        # Rename declared dim names to annotation aliases where they differ.
        if self._axes_override:
            renames = {
                decl.name: ann
                for decl, ann in zip(self._sym.output_dims, effective_out)
                if decl.name != ann
                and decl.name in result._axis_map
                and ann not in result._axis_map
            }
            if renames:
                result = result.rename_axes(renames)

        # Empty annotation [] → collapse all structural axes.
        if self._axes_override is not None and not self._axes_override:
            keep = result.variable_axes + result.domain_axes
            if result.structural_axes:
                struct_indices = [result.axis_map[ax.name] for ax in result.structural_axes]
                val = result.value
                for idx in sorted(struct_indices, reverse=True):
                    val = val.mean(dim=idx)
                result = Tensor(value=val, axes=keep, domain_type=result.domain_type)

        return result

    def _call_symbol(self, inputs, preserved, ax_names) -> Tensor:
        if self._is_pred:
            result = _apply_callable(
                func=self._fn, inputs=inputs, preserved_axis_names=preserved,
                output_axis_names=ax_names, output_type=Bool,
                name=self._sym.name,
            )
            if self._interp.classical:
                thr = self._interp.classical_threshold
                result = Tensor(
                    value=(result.value >= thr).float(),
                    axes=result.axes, domain_type=Bool,
                )
            return result

        # FunctionSymbol: resolve output type from sort grounding.
        output_type = None
        if self._sym.output_sort is not None:
            osort = self._sym.output_sort
            sort_grounding = self._interp._get_grounding(osort.name)
            if isinstance(sort_grounding, Type):
                output_type = sort_grounding
            else:
                output_type = {"Bool": Bool, "Real": Real}.get(osort.name)

        if output_type is None:
            sname = self._sym.output_sort.name if self._sym.output_sort is not None else "?"
            raise ValueError(
                f"FunctionSymbol '{self._sym.name}' has no output type. "
                f"Ground sort '{sname}' to a Type via interp[sort] = Type."
            )
        return _apply_callable(
            func=self._fn, inputs=inputs, preserved_axis_names=preserved,
            output_axis_names=ax_names, output_type=output_type,
            name=self._sym.name,
        )

    def _preserved_axes(self, inputs) -> set:
        """Structural axes from inputs that correspond to declared input_dims."""
        sig = self._interp.signature
        decl_dims = {d.name for d in self._sym.input_dims}
        if not decl_dims:
            return set()
        preserved: set = set()
        for inp in inputs:
            for axis in inp.structural_axes:
                sv = sig._structural_variables.get(axis.name)
                base = sv.dimension.name if sv else _base_name_of(axis.name)
                if axis.name in decl_dims or base in decl_dims:
                    preserved.add(axis.name)
        return preserved
