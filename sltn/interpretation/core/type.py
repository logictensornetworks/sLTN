"""
sltn.interpretation.core.type — Type system for tensor domain shapes and constraints.
"""
from __future__ import annotations

__all__ = ["Type", "Real", "Bool", "Int", "PositiveReal", "Complex", "product_type"]

import torch
from typing import Callable, Tuple
from sltn.interpretation.core.axis import Axis

class Type:
    def __init__(
        self,
        name: str = "Tensor",
        shape: Tuple[int, ...] = (1,),
        constraint: Callable[[torch.Tensor], bool] | None = None,
        axis_names: list[str] | None = None,
    ) -> None:

        self.name = name
        self.shape = shape
        self.constraint = constraint

        if axis_names is None:
            if len(shape) > 1:
                self.axis_names = [f"{name.lower()}_{i}" for i in range(len(shape))]
            elif len(shape) == 1:
                self.axis_names = [name.lower()]
            else:
                self.axis_names = []
        else:
            if len(axis_names) != len(shape):
                raise ValueError(
                    f"axis_names length ({len(axis_names)}) must match "
                    f"shape length ({len(shape)}); got axis_names={axis_names}, shape={shape}"
                )
            self.axis_names = list(axis_names)
        
        self.domain_axes = [Axis(name=name, role='domain') for name in self.axis_names]

    def __eq__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return self.shape == other.shape and self.axis_names == other.axis_names

    def validate(self, tensor: torch.Tensor) -> None:
        if self.constraint and not self.constraint(tensor):
            raise ValueError(f"Tensor does not satisfy the constraints of type '{self.name}'.")

    def __repr__(self) -> str:
        return f"Type(name={self.name}, shape={self.shape}, axis_names={self.axis_names})"


def _is_bool_like(x: torch.Tensor) -> bool:
    return (x >= 0).all() and (x <= 1).all()

def _is_int64(x: torch.Tensor) -> bool:
    return x.dtype == torch.int64

def _is_float(x: torch.Tensor) -> bool:
    return x.dtype == torch.float

def _is_positive_float(x: torch.Tensor) -> bool:
    return (x > 0).all() and x.dtype == torch.float

def _is_complex(x: torch.Tensor) -> bool:
    return torch.is_complex(x)

# Define some common types
Bool = Type(name="Bool", 
            shape=(1,), 
            constraint=_is_bool_like, 
            axis_names=["bool"])

Int = Type(name="Int", 
           shape=(1,), 
           constraint=_is_int64, 
           axis_names=["int"])

Real = Type(name="Real", 
            shape=(1,), 
            constraint=_is_float, 
            axis_names=["real"])

PositiveReal = Type(name="PositiveReal", 
                    shape=(1,), 
                    constraint=_is_positive_float, 
                    axis_names=["positive_real"])

Complex = Type(name="Complex", 
               shape=(1,), 
               constraint=_is_complex, 
               axis_names=["complex"])


def product_type(
    base_type: Type,
    shape: Tuple[int, ...],
    axis_names: list[str] | None = None,
) -> Type:
    """Create a product type: N copies of *base_type* arranged along new prefix axes.

    Parameters
    ----------
    base_type : Type
        The element type.  Its ``shape`` forms the *trailing* axes of the result.
    shape : tuple of int
        Sizes of the new *prefix* (product) axes.  The full shape of the
        returned type is ``shape + base_type.shape``.
    axis_names : list[str] | None
        Names for the prefix axes.
        If *base_type* has ``axis_names`` but ``axis_names`` is not given,
        generic ``"axis_i"`` names are used for the prefix.
    """
    if not shape:
        return base_type
    full_shape = shape + base_type.shape
    pnames = axis_names if axis_names is not None else [
        f'axis_{i}' for i in range(len(shape))
    ]
    axis_names = pnames + base_type.axis_names
    shape_str = 'x'.join(map(str, full_shape))
    return Type(
        name=f"{base_type.name}^{shape_str}",
        shape=full_shape,
        constraint=base_type.constraint,
        axis_names=axis_names,
    )
