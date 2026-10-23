"""
sltn.interpretation — Interpretation context for evaluating formulas.

An :class:`Interpretation` is the authoritative source of all symbol→value
bindings for a :class:`~sltn.signature.Signature`.  It also bundles logic
operators and provides ``interp(formula)`` (:meth:`Interpretation.forward`)
to numerically evaluate any formula or term.
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
    FunctionApplication, TermAxisSelect, Term, _deduplicate_axis_names,
)
from sltn.fol.formula import (
    Formula, TermFormula, FormulaAxisSelect, AtomicFormula,
    StructuralRelation,
    And, Or, Implies, Equivalent, Not, _BoundFormula, KB,
)

from sltn.parse.tokenizer import _base_name_of
from sltn.signature import (
    Signature, Symbol, FunctionSymbol, PredicateSymbol
)

from sltn.interpretation.apply import (
    apply_connective, apply_quantifier, apply_sat_agg,
    _var_axis_names,
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
    reserved = set(sort_type.axis_names) | set(Bool.axis_names)
    if sym.name in reserved:
        raise ValueError(
            f"Variable '{sym.name}' has the same name as a domain axis "
            f"({sorted(reserved)}) of its sort '{sym.sort.name}' or of predicate outputs. "
            f"Rename the variable, or set the axis names with Type(..., axis_names=[...])."
        )
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


def _value_to_expression(val: torch.Tensor | Tensor, name: str, interp: "Interpretation") -> Tensor:
    """Convert a Tensor binding into a properly annotated sltn.Tensor."""
    if isinstance(val, Tensor):
        return val
    if isinstance(val, torch.Tensor):
        sig = interp.signature
        if interp._device is not None:
            val = val.to(interp._device)
        sv_obj = sig.structural_variables.get(name)
        if sv_obj is not None:
            sort = sv_obj.dimension.sort
            sort_type = _resolve_sort_type(sort, interp) if sort is not None else None
            return _ground_structural_variable(sv_obj, val, sort_type)
        sym_obj = sig.variables.get(name)
        if sym_obj is not None:
            return _ground_variable(sym_obj, val, _resolve_sort_type(sym_obj.sort, interp))
        const_obj = sig.constants.get(name)
        if const_obj is not None:
            return _ground_constant(const_obj, val, _resolve_sort_type(const_obj.sort, interp))
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


# torch groundings live in the `_grounded` submodule under escaped keys:
# symbol names like 'train' or 'keys' would clash with nn.Module attributes.
_REG_PREFIX = "g_"
_REG_PATH = "_grounded." + _REG_PREFIX


def _reg_key(name: str) -> str:
    return _REG_PREFIX + name.replace(".", "·")


def _sym_name(reg: str) -> str:
    return reg[len(_REG_PREFIX):].replace("·", ".")


def _symbol_param_name(qualified: str) -> str:
    """'_grounded.g_P.0.weight' -> 'P.0.weight'."""
    if not qualified.startswith(_REG_PATH):
        return qualified
    reg, _, rest = qualified[len("_grounded."):].partition(".")
    return f"{_sym_name(reg)}.{rest}" if rest else _sym_name(reg)


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

class Interpretation(torch.nn.Module):
    """
    A logical interpretation of a Signature.

    Binds concrete values to symbols (constants, variables, functions,
    predicates) and holds a fuzzy logic operator bundle.

    An Interpretation is an ``nn.Module``: ``nn.Module`` / ``nn.Parameter``
    groundings are registered as submodules / parameters and data tensors as
    non-persistent buffers, so ``parameters()``, ``state_dict()`` (models only),
    ``train()`` / ``eval()`` and ``.to()`` cover everything.

    Networks used inside a plain callable (lambda, function) are not registered:
    trainable groundings must be ``nn.Module`` or ``nn.Parameter``.

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
        logic: Optional["Logic"] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.signature = signature
        self._device: Optional[torch.device] = None
        self._overrides: Dict[str, Any] = {}   # non-torch groundings
        self._grounded = torch.nn.Module()     # modules, parameters, tensors (buffers)

        self._logic: Logic = (logic.with_defaults() if logic is not None else Logic().with_defaults())
        self._logic._signature = self.signature

        # Per-call sub-formula cache and variable groundings; set only during forward().
        self._eval_cache: dict | None = None
        self._eval_vars: dict | None = None

        # Toggle memoisation on/off (default: on).
        self.use_cache: bool = True

        # Classical-mode state: threshold flag and saved fuzzy logic for restore.
        self._classical: bool = False
        self.classical_threshold: float = 0.5
        self._pre_classical_logic: Optional[Logic] = None

        self._ground_all({**(overrides or {}), **kwargs})

    def _ground_all(self, items: Dict[str | Symbol, Any]) -> None:
        """Ground every entry of *items*, sorts first (tensor shape checks need their Type)."""
        def is_sort(k: str | Symbol) -> bool:
            return (k.name if isinstance(k, Symbol) else k) in self.signature.sorts
        for k, v in sorted(items.items(), key=lambda kv: not is_sort(kv[0])):
            self.ground(k, v)

    def _derive(self, overrides: Dict[str, Any]) -> "Interpretation":
        """New interpretation grounded with *overrides*, with logic, classical mode,
        device and train/eval mode copied from self. The Logic is copied, not shared."""
        source = self
        new = Interpretation(self.signature)
        new.training = source.training
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
        """Return True if *name* (or the base of an alias) is bound."""
        if self._get_grounding(name) is not None:
            return True
        base = self.signature.alias_base(name)
        return base is not None and self._get_grounding(base) is not None

    def _get_grounding(self, name: str) -> Any:
        """Return the grounding for *name*, or None if not bound."""
        reg = _reg_key(name)
        g = self._grounded
        for store in (g._modules, g._parameters, g._buffers):
            if reg in store:
                return store[reg]
        return self._overrides.get(name)

    def _groundings(self) -> Dict[str, Any]:
        """All groundings by symbol name."""
        g = self._grounded
        out = dict(self._overrides)
        for store in (g._modules, g._parameters, g._buffers):
            out.update((_sym_name(reg), v) for reg, v in store.items())
        return out

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
        base = self.signature.alias_base(key)
        if base is not None:
            raise KeyError(
                f"Cannot ground '{key}': it is an alias of '{base}' and shares its "
                f"grounding. Ground '{base}' instead."
            )
        if self._device is not None:
            if isinstance(value, torch.nn.Parameter):
                value.data = value.data.to(self._device)   # keep the Parameter object
            elif isinstance(value, (torch.nn.Module, torch.Tensor)):
                value = value.to(self._device)
        if isinstance(value, torch.Tensor):
            self._check_tensor_shape(key, value)
        reg = _reg_key(key)
        if hasattr(self._grounded, reg):
            delattr(self._grounded, reg)
        self._overrides.pop(key, None)
        if isinstance(value, torch.nn.Parameter):
            self._grounded.register_parameter(reg, value)
        elif isinstance(value, torch.nn.Module):
            self._grounded.add_module(reg, value)
        elif isinstance(value, torch.Tensor):
            self._grounded.register_buffer(reg, value, persistent=False)
        else:
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
        sym_obj = sig.constants.get(key) or sig.variables.get(key)
        if sym_obj is not None:
            sort = sym_obj.sort
        elif key in sig.structural_variables:
            sort = sig.structural_variables[key].dimension.sort
        else:
            sort = None
        sort_type = self._get_grounding(sort.name) if sort is not None else None
        if not isinstance(sort_type, Type):
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
            val = self._get_grounding(self.signature.alias_base(key))
        return val

    def _normalize_assignment(self, raw: dict) -> Assignment:
        return {name: _value_to_expression(val, name, self) for name, val in raw.items()}

    # ------------------------------------------------------------------
    # Central evaluator
    # ------------------------------------------------------------------

    def _eval(self, node: Any, asgn: Assignment) -> Tensor:
        """Evaluate *node* (Term or Formula) under this interpretation.

        Caching (keyed on node repr + assignment identity) is applied here
        when ``self._eval_cache`` is set by ``__call__``.
        """
        cache = self._eval_cache
        if cache is None:
            return self._eval_node(node, asgn)
        node_repr = node.__dict__.get('_cache_repr')
        if node_repr is None:
            node_repr = node.__dict__['_cache_repr'] = repr(node)
        key = (node_repr, frozenset((k, id(v)) for k, v in asgn.items()))
        if key not in cache:
            # Storing asgn keeps its tensors alive, so their ids can't be reused in this call.
            cache[key] = (self._eval_node(node, asgn), asgn)
        return cache[key][0]

    def _variable(self, name: str) -> Tensor:
        """``get_variable_value``, grounded once per call so cached sub-formulas are shared."""
        memo = self._eval_vars
        if memo is None:
            return self.get_variable_value(name)
        if name not in memo:
            memo[name] = self.get_variable_value(name)
        return memo[name]

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
                val = asgn[n] if n in asgn else self._variable(n)
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
        fn = self.resolve(node.symbol, structural_axes_override=node.structural_axes)
        evaluated = [self._eval(arg, asgn) for arg in node.args]
        return fn(*evaluated)

    def _eval_bound(self, node: Any, asgn: Assignment) -> Tensor:
        """Evaluate a Forall/Exists quantifier node."""
        assignment = dict(asgn)
        sig = self.signature

        # Pre-populate assignment for quantified variables bound in the interpretation.
        for vname in node.var_names:
            if vname not in assignment and self._has_grounding(vname):
                assignment[vname] = self._variable(vname)

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
        struct_var_names = [v for v in node.var_names if v in sig.structural_variables]
        quantifier = node.quantifier  # "forall" or "exists"
        symbol     = node.symbol      # "∀" or "∃"

        formula_result = self._eval(node.formula, assignment)
        guard_result   = self._eval(node.guard, assignment) if node.guard else None

        if struct_var_names:
            agg_op = self.get_struct_agg_op(struct_var_names[0], quantifier)
            axes = struct_var_names
        else:
            agg_op = self.logic.Forall if quantifier == "forall" else self.logic.Exists
            axes = _var_axis_names([assignment[v] for v in node.var_names if v in assignment])
        return apply_quantifier(agg_op, formula_result, axes, mask=guard_result, symbol=symbol)

    def forward(
        self,
        target: Any,
        assignment: Optional[Assignment] = None,
        at: Optional[Dict[str, int]] = None,
    ) -> Tensor | Any:
        """Evaluate *target* under this interpretation (called as ``interp(target)``).

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

        if self.use_cache:
            self._eval_cache, self._eval_vars = {}, {}
        try:
            result = self._eval(target, asgn)
        finally:
            self._eval_cache = self._eval_vars = None

        if at:
            for axis_name, index in at.items():
                result = result.select(axis_name, index)
        return result

    # ------------------------------------------------------------------
    # Device placement
    # ------------------------------------------------------------------

    def to(self, *args: Any, **kwargs: Any) -> Interpretation:
        """``nn.Module.to``; the device is also remembered for future
        :meth:`ground` / ``interp[sym] = value`` calls. Returns *self*.
        """
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        for a in args:
            if isinstance(a, torch.Tensor):
                device = a.device
            elif isinstance(a, (str, int, torch.device)):
                device = a
        if device is not None:
            self._device = torch.device(device)
        return self

    @property
    def device(self) -> Optional[torch.device]:
        """The device this interpretation runs on."""
        return self._device

    # ------------------------------------------------------------------
    # Structural axis operator lookup
    # ------------------------------------------------------------------

    def get_struct_agg_op(self, axis: str, kind: str) -> AggregationOperator:
        """Return the structural aggregation operator for *axis* and *kind* ('forall'/'exists').

        Operators are set per dimension, so every structural variable over it shares them.
        """
        canonical = self.signature.resolve_structural_axis(axis)
        entry = self.logic._struct_entry(axis)
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
            sp = self.signature.structural_relations.get(name)
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
            # An alias (x1) reads its base grounding, renamed to the alias.
            base_name = self.signature.alias_base(name)
            if base_name is not None:
                val = self._get_grounding(base_name)
                if val is None:
                    raise RuntimeError(
                        f"VariableSymbol '{name}' (alias of '{base_name}') has no grounding. "
                        f"Use 'interp[{base_name!r}] = tensor' to ground it."
                    )
                if isinstance(val, Tensor):
                    return val.rename_axes({base_name: name})
                return _value_to_expression(val, name, self)
            raise RuntimeError(
                f"VariableSymbol '{name}' has no grounding in this Interpretation. "
                f"Pass it in the assignment dict or use 'interp[{name}] = tensor'."
            )
        return _value_to_expression(val, name, self)

    # ------------------------------------------------------------------
    # Binding management
    # ------------------------------------------------------------------
    def with_updates(self, **overrides) -> Interpretation:
        """Return a shallow copy with additional/overridden groundings."""
        return self._derive({**self._groundings(), **overrides})

    def copy(self) -> Interpretation:
        """Return a shallow copy sharing all tensor and module references."""
        return self.with_updates()

    def deepcopy(self) -> Interpretation:
        """Return a deep copy where all tensors and modules are independently cloned.
        A module bound to several symbols stays shared in the copy."""
        memo: dict = {}
        new_overrides: Dict[str, Any] = {}
        for k, v in self._groundings().items():
            if isinstance(v, (torch.nn.Module, torch.nn.Parameter)):
                new_overrides[k] = copy.deepcopy(v, memo)
            elif isinstance(v, torch.Tensor):
                new_overrides[k] = v.clone()
            else:
                new_overrides[k] = v
        return self._derive(new_overrides)

    def __copy__(self) -> Interpretation:
        return self.copy()

    def __deepcopy__(self, memo: dict) -> Interpretation:
        result = self.deepcopy()
        memo[id(self)] = result
        return result

    # ------------------------------------------------------------------
    # Trainable parameters
    # ------------------------------------------------------------------

    def named_parameters(self, prefix: str = "", recurse: bool = True,
                         remove_duplicate: bool = True):
        """``nn.Module.named_parameters`` with symbol-based names (``"P.0.weight"``).
        A module bound to several symbols is yielded once."""
        for name, p in super().named_parameters(recurse=recurse,
                                                remove_duplicate=remove_duplicate):
            yield prefix + _symbol_param_name(name), p

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    @classmethod
    def merge(cls, a: Interpretation, b: Interpretation) -> Interpretation:
        """Merge two interpretations; *b* overrides *a*. Logic, classical mode and
        device come from *b* (device falls back to *a*'s)."""
        merged = b._derive({**a._groundings(), **b._groundings()})
        if merged._device is None and a._device is not None:
            merged.to(a._device)
        return merged

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
        if sig.sorts:
            lines.append("  Sorts:")
            for s_name in sig.sorts:
                typ = self._get_grounding(s_name)
                suffix = f" : {typ}" if typ is not None else ""
                lines.append(f"    {s_name}{suffix}")

        for title, names in (("Constants", sig.constants), ("Variables", sig.variables),
                             ("Predicates", sig.predicates), ("Functions", sig.functions)):
            if names:
                lines.append(f"  {title}:")
                lines.extend(f"    {n}: {_val_label(self._get_grounding(n))}" for n in names)

        # ── Logic ─────────────────────────────────────────────────────
        lines.append(f"  {self.logic}")

        return "\n".join(lines)




# ---------------------------------------------------------------------------
# Internal proxy for overriding the callable of a symbol
# ---------------------------------------------------------------------------

class _SymbolProxy:
    """Wraps a FunctionSymbol/PredicateSymbol with its grounding callable for one evaluation call.

    ``axes_override`` names the symbol's ``output_dims`` in order (``A(x[t])[t1]``).
    A name that is already an argument axis passes that axis through.
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
        out_dims = self._sym.output_dims
        if self._axes_override is None:
            names = [d.name for d in out_dims]
        else:
            names = list(self._axes_override)
            if len(names) != len(out_dims):
                raise ValueError(
                    f"'{self._sym.name}' has {len(out_dims)} output dimension(s), "
                    f"got the output annotation {names}."
                )
        input_axes = {ax for inp in inputs for ax in inp.free_structural_variables}
        preserved |= (input_axes - preserved) & set(names)
        return self._call_symbol(inputs, preserved, names)

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
        osort = self._sym.output_sort
        output_type = _resolve_sort_type(osort, self._interp) if osort is not None else None
        if output_type is None:
            sname = osort.name if osort is not None else "?"
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
                dim = sig.dimension_of(axis.name)
                base = dim.name if dim is not None else _base_name_of(axis.name)
                if axis.name in decl_dims or base in decl_dims:
                    preserved.add(axis.name)
        return preserved
