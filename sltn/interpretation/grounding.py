"""Wrapping of raw groundings as annotated :class:`Tensor` values, and the
callable proxy that applies function and predicate groundings."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sltn.fol.term import _deduplicate_axis_names
from sltn.interpretation.apply import _apply_batch_callable
from sltn.interpretation.core.axis import Axis
from sltn.interpretation.core.tensor import Tensor
from sltn.interpretation.core.type import Bool, Real, Type
from sltn.parse.tokenizer import _base_name_of
from sltn.signature import FunctionSymbol, PredicateSymbol

if TYPE_CHECKING:
    from sltn.interpretation.interpretation import Interpretation


# Symbol → sltn.Tensor wrapping helpers

def _ground_constant(sym, val: torch.Tensor, sort_type: Type | None) -> Tensor:
    """Wrap a tensor as an annotated sltn.Tensor for a constant."""
    if sort_type is None:
        raise ValueError(f"Sort '{sym.sort.name}' is not grounded with a type.")
    struct_names = _deduplicate_axis_names(sym.dims)
    axes: list[Axis] = [Axis(n, 'structural') for n in struct_names]
    axes += [Axis(n, 'domain') for n in sort_type.axis_names]
    return Tensor(value=val, axes=axes, domain_type=sort_type)


def _ground_variable(sym, val: torch.Tensor, sort_type: Type | None) -> Tensor:
    """Wrap a tensor as an annotated sltn.Tensor for a variable."""
    if sort_type is None:
        raise ValueError(f"Sort '{sym.sort.name}' is not grounded with a type.")
    reserved = set(sort_type.axis_names) | set(Bool.axis_names)
    if sym.name in reserved:
        raise ValueError(
            f"Variable '{sym.name}' has the same name as a domain axis "
            f"({sorted(reserved)}) of its sort '{sym.sort.name}' or of predicate outputs. "
            f"Rename the variable, or set the axis names with Type(..., axis_names=[...])."
        )
    struct_names = _deduplicate_axis_names(sym.dims)
    axes: list[Axis] = [Axis(sym.name, 'variable')]
    axes += [Axis(n, 'structural') for n in struct_names]
    axes += [Axis(n, 'domain') for n in sort_type.axis_names]
    return Tensor(value=val, axes=axes, domain_type=sort_type)


def _ground_structural_variable(sym, val: torch.Tensor, sort_type: Type | None) -> Tensor:
    """Wrap a tensor as an annotated sltn.Tensor for a structural variable over a sorted dimension."""
    if sort_type is None:
        raise RuntimeError(
            f"Structural variable '{sym.name}': sort '{sym.dimension.sort.name}' of dimension "
            f"'{sym.dimension.name}' has no Type grounding. Use 'interp[{sym.dimension.sort.name!r}] = Type(...)'."
        )
    axes = [Axis(sym.name, 'structural'), *sort_type.domain_axes]
    return Tensor(value=val, axes=axes, domain_type=sort_type)


def _ground_structural_relation(sym, relation: torch.Tensor) -> Tensor:
    """Wrap the relation tensor as an annotated sltn.Tensor for a structural relation."""
    unique_names = _deduplicate_axis_names(sym.dims)
    axes = [Axis(n, 'structural') for n in unique_names] + [Axis('bool', 'domain')]
    return Tensor(value=relation.unsqueeze(-1), axes=axes, domain_type=Bool)


def _resolve_sort_type(sort, interp) -> Type | None:
    """Look up the tensor Type for *sort* from the interpretation's overrides."""
    grounding = interp._get_grounding(sort.name)
    if isinstance(grounding, Type):
        return grounding
    return {"Bool": Bool, "Real": Real}.get(sort.name)


def _value_to_tensor(val: torch.Tensor | Tensor, name: str, interp: Interpretation) -> Tensor:
    """Convert a Tensor binding into a properly annotated sltn.Tensor."""
    if isinstance(val, Tensor):
        return val
    if isinstance(val, torch.Tensor):
        sig = interp.signature
        if interp._device is not None:
            val = val.to(interp._device)
        sv_obj = sig.structural_variables.get(name)
        if sv_obj is not None:
            sort_type = _resolve_sort_type(sv_obj.dimension.sort, interp)
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
        for d, ann in zip(struct_axes, axis_annotations, strict=True)
        if d.name != ann
    }
    return val.rename_axes(renames) if renames else val


# torch groundings live in the `_grounded` submodule under escaped keys:
# symbol names like 'train' or 'keys' would clash with nn.Module attributes.

# Proxy applying the grounding of a function or predicate symbol

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
        preserved += [n for n in names if n in input_axes and n not in preserved]
        return self._call_symbol(inputs, preserved, names)

    def _call_symbol(self, inputs, preserved, ax_names) -> Tensor:
        if self._is_pred:
            output_type = Bool
        else:
            # FunctionSymbol: resolve output type from sort grounding.
            osort = self._sym.output_sort
            output_type = _resolve_sort_type(osort, self._interp) if osort is not None else None
            if output_type is None:
                sname = osort.name if osort is not None else "?"
                raise ValueError(
                    f"FunctionSymbol '{self._sym.name}' has no output type. "
                    f"Ground sort '{sname}' to a Type via interp[sort] = Type."
                )
        return _apply_batch_callable(
            func=self._fn, inputs=inputs, preserved_axis_names=preserved,
            output_axis_names=ax_names, output_type=output_type,
            name=self._sym.name,
        )

    def _preserved_axes(self, inputs) -> list[str]:
        """Argument structural axes of the declared input_dims, in input_dims order (ties by name)."""
        sig = self._interp.signature
        rank: dict[str, int] = {}
        for i, d in enumerate(self._sym.input_dims):
            rank.setdefault(d.name, i)
        if not rank:
            return []
        found: dict[str, int] = {}
        for inp in inputs:
            for axis in inp.structural_axes:
                dim = sig.dimension_of(axis.name)
                base = dim.name if dim is not None else _base_name_of(axis.name)
                r = rank.get(axis.name, rank.get(base))
                if r is not None:
                    found[axis.name] = r
        return sorted(found, key=lambda n: (found[n], n))
