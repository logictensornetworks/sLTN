"""
sltn.interpretation.core.tensor — Annotated tensor wrapper.
"""
from __future__ import annotations

__all__ = ["Tensor"]

from typing import Optional, Sequence

import torch

from sltn.interpretation.core.type import Type
from sltn.interpretation.core.axis import Axis

# ---------------------------------------------------------------------------
# Annotated Tensor
# ---------------------------------------------------------------------------

class Tensor:
    """
    A wrapper around a ``torch.Tensor`` with named and role-tagged axes.

    Every axis is described by an :class:`Axis` object (name + role).
    Roles:
      - *variable*   — individuals batch over a FOL variable
      - *structural* — structural (world/time/…) axis
      - *domain*     — feature / type dimension

    The ``domain_type`` encodes the expected shape and constraints of the
    domain axes collectively.
    """

    def __init__(
        self,
        value: torch.Tensor,
        axes: Sequence[Axis],
        domain_type: Optional[Type] = None,
    ) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected 'value' to be a torch.Tensor, got {type(value).__name__}")

        if value.ndim != len(axes):
            raise ValueError(
                f"Number of tensor dims ({value.ndim}) does not match "
                f"number of Axis objects ({len(axes)})."
            )

        self.value = value
        self.axes  = tuple(axes)

        self._axis_names            = [d.name for d in self.axes]
        self._axis_map              = {d.name: i for i, d in enumerate(self.axes)}
        self._variable_axes         = [d for d in self.axes if d.role == 'variable']
        self._structural_axes       = [d for d in self.axes if d.role == 'structural']
        self._domain_axes           = [d for d in self.axes if d.role == 'domain']
        self._free_variables        = [d.name for d in self._variable_axes]
        self._free_structural_vars  = [d.name for d in self._structural_axes]
        self._domain_names          = [d.name for d in self._domain_axes]
        self._domain_shape          = tuple(self.value.shape[self._axis_map[d.name]]
                                            for d in self._domain_axes)

        if len(set(self._axis_names)) != len(self._axis_names):
            raise ValueError(f"Axis names must be unique. Got: {self._axis_names}")

        if domain_type is not None:
            if self._domain_shape != domain_type.shape:
                raise ValueError(
                    f"Domain axes shape {self._domain_shape} does not match "
                    f"expected shape {domain_type.shape} for type '{domain_type.name}'."
                )
            if self._domain_names != domain_type.axis_names:
                raise ValueError(
                    f"Domain axis names {self._domain_names} do not match "
                    f"expected axis names {domain_type.axis_names} for type '{domain_type.name}'."
                )
            self.domain_type = domain_type
            domain_type.validate(self.value)
        else:
            self.domain_type = Type(shape=self._domain_shape, axis_names=self._domain_names)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> torch.Size:
        return self.value.shape

    @property
    def axis_names(self) -> list[str]:
        return self._axis_names

    @property
    def axis_map(self) -> dict[str, int]:
        return self._axis_map

    @property
    def named_shape(self) -> dict[str, int]:
        return {d.name: self.value.shape[i] for i, d in enumerate(self.axes)}

    @property
    def ndim(self) -> int:
        return self.value.ndim

    @property
    def free_variables(self) -> list[str]:
        return self._free_variables

    @property
    def variable_axes(self) -> list[Axis]:
        return self._variable_axes

    @property
    def free_structural_variables(self) -> list[str]:
        return self._free_structural_vars

    @property
    def structural_axes(self) -> list[Axis]:
        return self._structural_axes

    @property
    def domain_axes(self) -> list[Axis]:
        return self._domain_axes

    @property
    def domain_shape(self) -> tuple[int, ...]:
        return self._domain_shape

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        axes_repr = ", ".join(
            f"{d.name}({d.role}): {self.value.shape[i]}"
            for i, d in enumerate(self.axes)
        )
        return (
            f"Tensor(shape=({axes_repr}), "
            f"domain_type={self.domain_type.name})"
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, idx) -> Tensor:
        slicing         = [slice(None)] * self.ndim
        new_axes        = list(self.axes)
        axes_to_remove  = []

        if isinstance(idx, dict):
            for axis_name, index_val in idx.items():
                if axis_name not in self._axis_map:
                    raise ValueError(f"Axis '{axis_name}' not in annotated Tensor.")
                axis = self._axis_map[axis_name]
                slicing[axis] = index_val
                if isinstance(index_val, int):
                    axes_to_remove.append(axis_name)
        else:
            variable_axes = [
                (i, d) for i, d in enumerate(self.axes) if d.role == 'variable'
            ]
            if len(variable_axes) != 1:
                raise ValueError(
                    "Default indexing requires exactly one 'variable' axis."
                )
            axis, dim_obj = variable_axes[0]
            slicing[axis] = idx
            if isinstance(idx, int):
                axes_to_remove.append(dim_obj.name)

        new_value  = self.value[tuple(slicing)]
        final_axes = [d for d in new_axes if d.name not in axes_to_remove]

        if not final_axes:
            new_value  = new_value.unsqueeze(0)
            final_axes = [Axis('domain_0', 'domain')]

        return Tensor(
            value=new_value,
            axes=final_axes,
            domain_type=self.domain_type,
        )

    # ------------------------------------------------------------------
    # Permute / rename / flatten
    # ------------------------------------------------------------------

    def permute(self, *axis_names: str) -> Tensor:
        if len(axis_names) != self.ndim or set(axis_names) != set(self._axis_map):
            raise ValueError("Permutation axes must be a permutation of existing axes.")
        axis_order = [self._axis_map[name] for name in axis_names]
        new_axes   = [self.axes[idx] for idx in axis_order]
        return Tensor(
            value=self.value.permute(axis_order),
            axes=new_axes,
            domain_type=self.domain_type,
        )

    def rename_axes(self, axis_renames: dict[str, str]) -> Tensor:
        new_axes = []
        for d in self.axes:
            if d.name in axis_renames:
                new_axes.append(Axis(name=axis_renames[d.name], role=d.role))
            else:
                new_axes.append(d)
        return Tensor(
            value=self.value, axes=new_axes, domain_type=self.domain_type
        )

    def select(self, axis_name: str, index: int) -> Tensor:
        """Select a single index along the named axis, removing that axis.

        Works on any axis role: structural, variable, or domain.

        Uses :func:`torch.select` which returns a **view** of the tensor —
        no data is copied and gradients flow unimpeded.

        Parameters
        ----------
        axis_name : str
            Name of the axis to select along.
        index : int
            Index to select.  Supports negative indexing.

        Returns
        -------
        Tensor
            Same annotated tensor with *axis_name* removed and the value reduced
            to the selected slice.
        """
        if axis_name not in self._axis_map:
            available = list(self._axis_map)
            raise ValueError(
                f"Axis '{axis_name}' not found. Available axes: {available}"
            )
        axis_pos  = self._axis_map[axis_name]
        new_value = torch.select(self.value, dim=axis_pos, index=index)
        new_axes  = [d for d in self.axes if d.name != axis_name]
        return Tensor(
            value=new_value,
            axes=new_axes,
            domain_type=self.domain_type,
        )

    # ------------------------------------------------------------------
    # Broadcast alignment
    # ------------------------------------------------------------------

    @classmethod
    def align(cls, *expressions: Tensor) -> list[Tensor]:
        """Broadcast-align a set of sltn.Tensors over all variable and structural axes."""
        if not expressions:
            return []

        all_variable_axes:   dict[str, Axis] = {}
        all_structural_axes: dict[str, Axis] = {}

        for expr in expressions:
            for d in expr.variable_axes:
                if d.name in all_variable_axes and all_variable_axes[d.name] != d:
                    raise ValueError(
                        f"Axis '{d.name}' has conflicting definitions: "
                        f"{all_variable_axes[d.name]} vs {d}"
                    )
                all_variable_axes[d.name] = d
            for d in expr.structural_axes:
                if d.name in all_structural_axes and all_structural_axes[d.name] != d:
                    raise ValueError(
                        f"Axis '{d.name}' has conflicting definitions: "
                        f"{all_structural_axes[d.name]} vs {d}"
                    )
                all_structural_axes[d.name] = d

        both = all_variable_axes.keys() & all_structural_axes.keys()
        if both:
            raise ValueError(
                f"Axis name(s) {sorted(both)} used both as a variable axis and as a "
                f"structural axis; an axis name must have a single role."
            )
        sorted_var_names    = sorted(all_variable_axes)
        sorted_struct_names = sorted(all_structural_axes)
        sorted_broadcast    = sorted_var_names + sorted_struct_names
        all_broadcast_axes  = {**all_variable_axes, **all_structural_axes}

        aligned = []
        for expr in expressions:
            expr_var    = sorted(expr.variable_axes,   key=lambda d: d.name)
            expr_struct = sorted(expr.structural_axes, key=lambda d: d.name)
            expr_domain = expr.domain_axes
            current     = expr_var + expr_struct + expr_domain

            permute_idx    = [expr.axis_map[d.name] for d in current]
            permuted_value = expr.value.permute(*permute_idx)

            final_axes     = list(current)
            expr_names_set = set(expr.axis_names)

            for i, gname in enumerate(sorted_broadcast):
                if gname not in expr_names_set:
                    permuted_value = permuted_value.unsqueeze(i)
                    final_axes.insert(i, all_broadcast_axes[gname])

            aligned.append(
                cls(value=permuted_value, axes=final_axes, domain_type=expr.domain_type)
            )

        return aligned

    # ------------------------------------------------------------------
    # Diagonal
    # ------------------------------------------------------------------

    @classmethod
    def diag(cls, *expressions: Tensor) -> list[Tensor]:
        """Share a single variable axis across multiple expressions (for diagonal quantification)."""
        for e in expressions:
            n_var = len(e.variable_axes)
            if n_var != 1:
                raise TypeError(
                    f"All arguments to diag must have exactly one variable axis, "
                    f"but an expression with axes {e.axis_names} has {n_var}."
                )
        if len(expressions) < 2:
            raise ValueError("diag requires at least two expressions.")

        var_axes  = [e.variable_axes[0] for e in expressions]
        var_sizes = [e.shape[e.axis_map[d.name]] for e, d in zip(expressions, var_axes)]
        if len(set(var_sizes)) != 1:
            raise ValueError(
                f"All expressions in diag must have the same number of individuals. "
                f"Got sizes: {var_sizes}"
            )

        diag_axis_name = f"({','.join(sorted(d.name for d in var_axes))})"
        new_var_axis   = Axis(name=diag_axis_name, role='variable')

        diag_exprs = []
        for expr, old_var_axis in zip(expressions, var_axes):
            new_axes = [new_var_axis if d == old_var_axis else d for d in expr.axes]
            diag_exprs.append(cls(value=expr.value, axes=new_axes, domain_type=expr.domain_type))
        return diag_exprs

    # ------------------------------------------------------------------
    # Factory: variable-shaped sltn.Tensor
    # ------------------------------------------------------------------

    @staticmethod
    def as_variable(
        name: str,
        value: torch.Tensor,
        domain_type: Optional[Type] = None,
    ) -> Tensor:
        """
        Create an sltn.Tensor suitable for use as a FOL variable in assignments.
        """
        if domain_type is None:
            domain_shape = value.shape[1:]
            axis_names = [f"domain_{i}" for i in range(len(domain_shape))]
            domain_type = Type(shape=domain_shape, axis_names=axis_names)

        axes = [Axis(name, 'variable')] + [
            Axis(n, 'domain') for n in domain_type.axis_names
        ]

        return Tensor(value=value, axes=axes, domain_type=domain_type)
