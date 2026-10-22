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

from math import prod
from typing import Any, Optional

import torch

from sltn.interpretation.core import Tensor, Axis, Bool
from sltn.interpretation.operators.base import AggregationOperator, ConnectiveOperator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _var_axis_names(vars: list) -> list[str]:
    """Return unique variable-role axis names from a list of variable Tensors."""
    seen: set = set()
    for var in vars:
        seen.update(var.free_variables)
    return list(seen)


def _deduplicate_axis_names(dims: list[str]) -> list[str]:
    """Return unique axis names for *dims*, appending ``_0``, ``_1``, … when
    the same :class:`Dimension` appears more than once (e.g. ``[T, T]`` →
    ``['T_0', 'T_1']``).
    """
    seen: dict[str, int] = {}
    for d in dims:
        seen[d.name] = seen.get(d.name, 0) + 1
    counts: dict[str, int] = {}
    names: list[str] = []
    for d in dims:
        n = d.name
        if seen[n] > 1:
            idx = counts.get(n, 0)
            counts[n] = idx + 1
            names.append(f"{n}{idx}")
        else:
            names.append(n)
    return names

# ---------------------------------------------------------------------------
# Public apply functions
# ---------------------------------------------------------------------------

def apply_connective(op: ConnectiveOperator, *exprs: Tensor) -> Tensor:
    """Apply a connective operator to one or more :class:`sltn.Tensor` objects.

    Handles axis alignment and wraps the result back in a Tensor with
    domain type :class:`~sltn.Bool`.
    """
    aligned = Tensor.align(*exprs)
    out = op(*[e.value for e in aligned])
    return Tensor(value=out, axes=aligned[0].axes, domain_type=Bool)


def apply_quantifier(
    agg_op: AggregationOperator,
    formula: Tensor,
    axis_names: list,
    *,
    mask: Optional[Tensor] = None,
    symbol: str = "∀",
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
        Optional Tensor of soft weights or a hard boolean mask.

        * **float in [0, 1]** (soft / learnable guard) — passed to the
          aggregation operator as weights; with pmean-based operators this
          produces a weighted power mean.
        * **bool** — hard filter; masked-out positions are excluded.

        Missing data must be excluded with a mask (a guard). NaN values in
        *formula* are not treated specially: they propagate, and the result
        then fails the ``Bool`` check.
    symbol:
        Unicode quantifier symbol used in the output Tensor name.
    """
    if mask is not None:
        formula, aligned_mask = Tensor.align(formula, mask)
        m = aligned_mask.value
        mask_domain_names = {d.name for d in aligned_mask.domain_axes}
        n_extra = sum(1 for d in formula.domain_axes if d.name not in mask_domain_names)
        for _ in range(n_extra):
            m = m.unsqueeze(-1)
        explicit_mask = m
    else:
        explicit_mask = None

    value = formula.value

    if explicit_mask is not None:
        effective = explicit_mask.float() if explicit_mask.dtype == torch.bool else explicit_mask
        # Broadcast both to the full shape, so weighted aggregators count every
        # element along axes the guard doesn't span.
        value, effective = torch.broadcast_tensors(value, effective)
    else:
        effective = None

    agg_indices: list = []
    ax_map = formula.axis_map
    for ax in axis_names:
        if ax in ax_map:
            agg_indices.append(ax_map[ax])

    if not agg_indices:
        return formula

    vacuous_axes = any(value.shape[i] == 0 for i in agg_indices)
    if vacuous_axes:
        identity = 1.0 if symbol == "∀" else 0.0
        rest_shape = tuple(
            value.shape[j] for j in range(value.ndim) if j not in set(agg_indices)
        )
        output = torch.full(rest_shape, identity, dtype=value.dtype, device=value.device)
    else:
        output = agg_op(value, dim=agg_indices, mask=effective)

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


# ---------------------------------------------------------------------------
# TensorCallable — user-facing wrapper type for sltn.Tensor-level callables
# ---------------------------------------------------------------------------

class TensorCallable:
    """Wrap a callable so it receives aligned :class:`sltn.Tensor` objects rather
    than raw tensors.  The wrapped function must return a Tensor.

    The resulting Tensor is used directly — no further axis bookkeeping
    is applied by the framework. This is intended as a shortcut, but does not
    fully comply with sLTN principles.
    """

    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def __call__(self, *expressions: Tensor) -> Tensor:
        return self.fn(*expressions)

    def __repr__(self) -> str:
        return f"TensorCallable({self.fn!r})"


def _apply_expression_callable(func: "TensorCallable", inputs: list, name: str) -> Tensor:
    aligned = Tensor.align(*inputs) if inputs else []
    result = func(*aligned)
    if not isinstance(result, Tensor):
        raise TypeError(
            f"TensorCallable for '{name}' must return a Tensor, "
            f"got {type(result).__name__}."
        )
    return result


# ---------------------------------------------------------------------------
# Callable-dispatch helpers (used by Interpretation._SymbolProxy)
# ---------------------------------------------------------------------------

def _apply_batch_callable(
    func,
    inputs: list,
    preserved_axis_names,
    output_axis_names,
    output_type,
    name: str,
    no_grad_probe: bool = False,
) -> Tensor:
    """Shared implementation for plain callables and :class:`torch.nn.Module`.

    Variable axes and non-preserved structural axes are flattened into a single
    batch dimension before the call.  Preserved structural axes are kept as
    leading dimensions throughout.
    """
    if not inputs:
        out_tensor = func()
        if not isinstance(out_tensor, torch.Tensor):
            raise TypeError(f"Callable for '{name}' must return a Tensor, got {type(out_tensor)}.")
        return Tensor(value=out_tensor, axes=output_type.domain_axes, domain_type=output_type)

    aligned = Tensor.align(*inputs)

    batched_structural_axes   = [d for d in aligned[0].structural_axes if d.name not in preserved_axis_names]
    preserved_structural_axes = [d for d in aligned[0].structural_axes if d.name in preserved_axis_names]
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

    if batch_size == 0:
        probe_inputs = [r.new_zeros((1,) + r.shape[1:]) for r in reshaped]
        if no_grad_probe:
            with torch.no_grad():
                probe_out = func(*probe_inputs)
        else:
            probe_out = func(*probe_inputs)
        if not isinstance(probe_out, torch.Tensor):
            raise TypeError(f"Callable for '{name}' must return a Tensor, got {type(probe_out)}.")
        output_flat = reshaped[0].new_zeros((0,) + probe_out.shape[1:])
    else:
        output_flat = func(*reshaped)
        if not isinstance(output_flat, torch.Tensor):
            raise TypeError(f"Callable for '{name}' must return a Tensor, got {type(output_flat)}.")

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
    if n_struct_new > 0 and len(output_type.shape) > 0:
        if len(out_data_shape) == n_struct_new:
            for _ in output_type.shape:
                output_flat = output_flat.unsqueeze(-1)
            out_data_shape = output_flat.shape[1 + n_out_pres:]

    new_struct_out = [Axis(n, 'structural') for n in new_output_axis_names]
    out_axes       = new_struct_out + output_type.domain_axes
    output         = output_flat.reshape(*batch_shape, *output_pres_shape, *out_data_shape)
    final_dims     = batch_axes + output_preserved + out_axes

    return Tensor(value=output, axes=final_dims, domain_type=output_type)


def _apply_callable(
    func,
    inputs: list,
    preserved_axis_names,
    output_axis_names,
    output_type,
    name: str,
) -> Tensor:
    """Dispatch to the appropriate execution path based on *func* type."""
    if isinstance(func, TensorCallable):
        return _apply_expression_callable(func, inputs, name)
    return _apply_batch_callable(
        func, inputs, preserved_axis_names, output_axis_names, output_type,
        name, no_grad_probe=isinstance(func, torch.nn.Module),
    )
