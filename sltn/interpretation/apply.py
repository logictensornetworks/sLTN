"""
sltn.interpretation.apply — Public apply functions for connectives, quantifiers, and KB aggregation.

These are the building blocks for evaluating logical formulas numerically.
They are stateless and operate purely on :class:`~sltn.interpretation.core.tensor.Tensor` objects.
"""
from __future__ import annotations

__all__ = [
    "apply_connective",
    "apply_quantifier",
    "apply_sat_agg",
]

import copy
from math import prod
from typing import Any, Literal

import torch

from sltn.interpretation.core import Axis, Bool, Tensor
from sltn.interpretation.operators.base import AggregationOperator, ConnectiveOperator


# Internal helpers

def _var_axis_names(vars: list) -> list[str]:
    """Return unique variable-role axis names from a list of variable Tensors."""
    seen: set = set()
    for var in vars:
        seen.update(var.free_variables)
    return list(seen)


def _exact(op: Any) -> Any:
    """Return *op* without its stable approximations, for evaluating guards.

    The stable operators keep truth values away from 0 (the stable product
    maps 0 & 1 to about 1e-6), and the aggregators count every entry with a
    weight above 0 as inside the guard. In a guard, the exact operator gives
    a real 0 where an atom is false. Values strictly between 0 and 1 change
    by about eps only, so soft guards are unaffected.

    The copy has ``stable=False``, and so have the operators it is built
    from (the conjunction inside ``OrProbSum`` or ``ImpliesReichenbach``).
    An operator with no ``stable`` setting is returned as it is.
    """
    attrs = getattr(op, "__dict__", {})
    inner = {k: v for k, v in attrs.items() if isinstance(v, ConnectiveOperator)}
    if attrs.get("stable") is not True and not inner:
        return op
    exact = copy.copy(op)
    if attrs.get("stable") is True:
        exact.stable = False
    for k, v in inner.items():
        setattr(exact, k, _exact(v))
    return exact


# Public apply functions

def apply_connective(op: ConnectiveOperator, *exprs: Tensor, exact: bool = False) -> Tensor:
    """Apply a connective operator to one or more :class:`sltn.Tensor` objects.

    Handles axis alignment and wraps the result back in a Tensor with
    domain type :class:`~sltn.Bool`. With ``exact=True`` (inside a guard),
    the operator is applied without its stable approximations (see
    :func:`_exact`).
    """
    if exact:
        op = _exact(op)
    aligned = Tensor.align(*exprs)
    out = op(*[e.value for e in aligned])
    return Tensor(value=out, axes=aligned[0].axes, domain_type=Bool)


def apply_quantifier(
    agg_op: AggregationOperator,
    formula: Tensor,
    axis_names: list,
    *,
    mask: Tensor | None = None,
    quantifier: Literal["forall", "exists"] = "forall",
) -> Tensor:
    """Aggregate *formula* over *axis_names* using *agg_op*.

    Parameters
    ----------
    agg_op:
        Raw aggregation operator.
    formula:
        Boolean annotated sltn.Tensor to aggregate over.
    axis_names:
        Names of the axes to reduce.  Axes not present in *formula* are
        silently skipped.
    mask:
        Optional guard: a Bool Tensor of weights in ``[0, 1]``, aligned with
        *formula* by axis name.  Entries of weight 0 are left out and only
        the relative weights matter.
    quantifier:
        ``"forall"`` or ``"exists"``.  Where there is nothing to aggregate
        (an empty axis, or a guard that is 0 everywhere), the result is the
        quantifier's neutral value: 1 for ∀, 0 for ∃, whatever *agg_op* is.
    """
    ax_map = formula.axis_map
    dims = [ax_map[ax] for ax in axis_names if ax in ax_map]
    if not dims:
        return formula

    weights = None
    if mask is not None:
        formula, mask = Tensor.align(formula, mask)
        # The guard may span axes the body lacks, and vice versa.
        value, weights = torch.broadcast_tensors(formula.value, mask.value)
    else:
        value = formula.value

    neutral = 1.0 if quantifier == "forall" else 0.0
    if any(value.shape[i] == 0 for i in dims):
        rest_shape = [n for i, n in enumerate(value.shape) if i not in dims]
        output = torch.full(rest_shape, neutral, dtype=value.dtype, device=value.device)
    else:
        output = agg_op(value, dim=dims, mask=weights)
        if weights is not None:
            empty = weights.sum(dim=dims) == 0
            output = torch.where(empty, neutral, output)

    new_axes = [d for d in formula.axes if d.name not in axis_names]
    return Tensor(value=output, axes=new_axes, domain_type=Bool)


def apply_sat_agg(agg_op: AggregationOperator, *formulas: Tensor) -> Tensor:
    """Aggregate KB clause results into a single scalar loss value."""
    if not formulas:
        raise ValueError("apply_sat_agg requires at least one formula.")
    stacked = torch.stack([f.value.mean() for f in formulas])
    result  = agg_op(stacked, dim=0)
    return Tensor(
        value=result.unsqueeze(0),
        axes=[Axis('bool', 'domain')],
        domain_type=Bool,
    )


# Callable-dispatch helpers (used by Interpretation._SymbolProxy)

def _copied_axes_note(inputs: list, preserved_structural_axes: list) -> str:
    """Describe the preserved axes that an argument received only as copies."""
    copied = [
        (arg, pos, axis.name)
        for arg, expr in enumerate(inputs, start=1)
        for pos, axis in enumerate(preserved_structural_axes, start=1)
        if axis.name not in {d.name for d in expr.structural_axes}
    ]
    if not copied:
        return ""
    described = ", ".join(
        f"argument {arg} has no axis '{axis}' (position {pos} of its shape)"
        for arg, pos, axis in copied
    )
    return (
        f"\nEvery argument receives all the consumed axes of the call, so "
        f"{described} and gets copies along it. If your grounding removes "
        f"'{copied[0][2]}' (e.g. a GRU over the frames) before combining it with "
        f"this argument, drop the copies first, e.g. arg.select({copied[0][1]}, 0)."
    )


def _batch_axis_note(batch_axes: list, batch_shape: tuple, batch_size: int) -> str:
    """Say where the size of the batch axis comes from, for a grounding that lost it."""
    if not batch_axes:
        where = "the arguments have no free variables, so it has a single row"
    elif len(batch_axes) == 1:
        where = f"one row per individual of {batch_axes[0].name} ({batch_size})"
    else:
        sizes = " × ".join(f"{d.name} ({n})" for d, n in zip(batch_axes, batch_shape, strict=True))
        where = f"one row per combination of {sizes} = {batch_size}, flattened"
    return (
        f"\nThe first axis the grounding receives is the batch: {where}. "
        f"The grounding must keep it and reduce only over the other axes "
        f"(e.g. dim=-1, not over all axes)."
    )


def _apply_batch_callable(
    func,
    inputs: list,
    preserved_axis_names,
    output_axis_names,
    output_type,
    name: str,
) -> Tensor:
    """Shared implementation for plain callables and :class:`torch.nn.Module`.

    Variable axes and non-preserved structural axes are flattened into a single
    batch dimension before the call.  Preserved structural axes are kept as
    leading dimensions throughout, on every argument: an argument without one
    receives copies along it, and a grounding that doesn't want them drops them.
    """
    if not inputs:
        out_tensor = func()
        if not isinstance(out_tensor, torch.Tensor):
            raise TypeError(f"Callable for '{name}' must return a Tensor, got {type(out_tensor)}.")
        return Tensor(value=out_tensor, axes=output_type.domain_axes, domain_type=output_type)

    aligned = Tensor.align(*inputs)

    # preserved_axis_names is ordered: it fixes the layout of the consumed axes.
    struct_by_name = {d.name: d for d in aligned[0].structural_axes}
    preserved_structural_axes = [struct_by_name[n] for n in preserved_axis_names if n in struct_by_name]
    batched_structural_axes   = [d for d in aligned[0].structural_axes if d.name not in preserved_axis_names]
    batch_axes = aligned[0].variable_axes + batched_structural_axes

    def _canonical_layout(expr: Tensor) -> torch.Tensor:
        order = batch_axes + preserved_structural_axes + list(expr.domain_axes)
        perm = tuple(expr.axis_map[d.name] for d in order)
        return expr.value.permute(*perm)

    n_broad = len(batch_axes) + len(preserved_structural_axes)
    broad_shapes = [v.shape[:n_broad] for v in (_canonical_layout(e) for e in aligned)]
    target_broad = torch.broadcast_shapes(*broad_shapes)
    batch_shape, pres_shape = target_broad[:len(batch_axes)], target_broad[len(batch_axes):n_broad]
    batch_size = prod(batch_shape)

    reshaped = []
    for expr in aligned:
        v        = _canonical_layout(expr)
        domain   = v.shape[n_broad:]
        expanded = v.expand(target_broad + domain)
        reshaped.append(expanded.reshape(batch_size, *pres_shape, *domain))

    try:
        if batch_size == 0:
            probe_inputs = [r.new_zeros((1, *r.shape[1:])) for r in reshaped]
            # Only the output shape is needed: eval mode keeps BatchNorm/Dropout state untouched.
            modes = [(m, m.training) for m in func.modules()] if isinstance(func, torch.nn.Module) else []
            try:
                for m, _ in modes:
                    m.training = False
                with torch.no_grad():
                    probe_out = func(*probe_inputs)
            finally:
                for m, mode in modes:
                    m.training = mode
            if not isinstance(probe_out, torch.Tensor):
                raise TypeError(f"Callable for '{name}' must return a Tensor, got {type(probe_out)}.")
            output_flat = reshaped[0].new_zeros((0, *probe_out.shape[1:]))
        else:
            output_flat = func(*reshaped)
            if not isinstance(output_flat, torch.Tensor):
                raise TypeError(f"Callable for '{name}' must return a Tensor, got {type(output_flat)}.")
    except RuntimeError as err:
        note = _copied_axes_note(inputs, preserved_structural_axes)
        # Subclasses such as out-of-memory errors are not about shapes.
        if type(err) is not RuntimeError or not note:
            raise
        shapes = ", ".join(str(tuple(r.shape)) for r in reshaped)
        raise RuntimeError(f"Grounding of '{name}' failed on arguments of shapes {shapes}: {err}{note}") from err

    output_axis_names_set = set(output_axis_names) if output_axis_names else set()
    output_preserved  = [d for d in preserved_structural_axes if d.name in output_axis_names_set]
    output_pres_shape = tuple(
        pres_shape[i] for i, d in enumerate(preserved_structural_axes)
        if d.name in output_axis_names_set
    )
    n_out_pres     = len(output_pres_shape)
    out_data_shape = output_flat.shape[1 + n_out_pres:]

    pres_out_names        = {d.name for d in output_preserved}
    new_output_axis_names = [n for n in output_axis_names if n not in pres_out_names]

    n_struct_new = len(new_output_axis_names)
    if n_struct_new > 0 and len(output_type.shape) > 0 and len(out_data_shape) == n_struct_new:
        for _ in output_type.shape:
            output_flat = output_flat.unsqueeze(-1)
        out_data_shape = output_flat.shape[1 + n_out_pres:]

    new_struct_out = [Axis(n, 'structural') for n in new_output_axis_names]
    out_axes       = new_struct_out + output_type.domain_axes
    final_dims     = batch_axes + output_preserved + out_axes

    leading = (batch_size, *output_pres_shape)
    n_data  = n_struct_new + len(output_type.domain_axes)
    if output_flat.shape[:len(leading)] != leading or len(out_data_shape) != n_data:
        expected = [*leading, *"?" * n_struct_new, *output_type.shape]
        lost_batch = output_flat.dim() == 0 or output_flat.shape[0] != batch_size
        raise ValueError(
            f"The grounding of '{name}' returned shape {tuple(output_flat.shape)}, "
            f"expected ({', '.join(map(str, expected))})."
            + (_batch_axis_note(batch_axes, batch_shape, batch_size) if lost_batch else "")
            + _copied_axes_note(inputs, preserved_structural_axes)
        )

    output = output_flat.reshape(*batch_shape, *output_pres_shape, *out_data_shape)
    return Tensor(value=output, axes=final_dims, domain_type=output_type)

