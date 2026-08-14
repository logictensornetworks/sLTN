"""Tests for sltn.Tensor: construction, alignment, diag/undiag, select, flatten."""
import pytest
import torch
from sltn import Axis, Tensor, Type, Real, Bool, product_type


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_basic():
    t = torch.rand(3, 4)
    dims = [Axis("x", "variable"), Axis("f", "domain")]
    e = Tensor(value=t, axes=dims)
    assert e.value is t
    assert e.ndim == 2
    assert e.named_shape == {"x": 3, "f": 4}


def test_construction_dim_count_mismatch_raises():
    t = torch.rand(3, 4)
    dims = [Axis("x", "variable")]  # only 1 dim for 2-D tensor
    with pytest.raises(ValueError, match="Number of tensor dims"):
        Tensor(value=t, axes=dims)


def test_construction_duplicate_axis_name_raises():
    t = torch.rand(3, 4)
    dims = [Axis("x", "variable"), Axis("x", "domain")]
    with pytest.raises(ValueError, match="unique"):
        Tensor(value=t, axes=dims)


def test_construction_invalid_role_raises():
    with pytest.raises(ValueError, match="role"):
        Axis("x", "invalid_role")


def test_as_variable_1d():
    t = torch.rand(5)
    e = Tensor.as_variable("x", t)
    assert e.free_variables == ["x"]
    assert e.value.shape == (5,)  # No longer auto-unsqueezes domain; represents 1D batch of scalar individuals

    # If domain_type is specified (e.g., Real with shape (1,)), passing 1D tensor should raise ValueError from sltn.Tensor
    with pytest.raises(ValueError, match="Number of tensor dims"):
        Tensor.as_variable("x", t, domain_type=Real)


def test_as_variable_2d():
    t = torch.rand(5, 3)
    e = Tensor.as_variable("x", t)
    assert e.free_variables == ["x"]
    assert e.named_shape == {"x": 5, "domain_0": 3}


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_free_variables_and_free_structural_variables():
    dims = [Axis("x", "variable"), Axis("t", "structural"), Axis("f", "domain")]
    e = Tensor(value=torch.rand(2, 3, 4), axes=dims)
    assert e.free_variables == ["x"]
    assert e.free_structural_variables == ["t"]


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def test_getitem_integer_removes_axis():
    dims = [Axis("x", "variable"), Axis("f", "domain")]
    e = Tensor(value=torch.rand(4, 2), axes=dims)
    e2 = e[1]
    assert "x" not in e2.axis_names


def test_getitem_dict():
    dims = [Axis("x", "variable"), Axis("y", "variable"), Axis("f", "domain")]
    e = Tensor(value=torch.rand(3, 4, 2), axes=dims)
    e2 = e[{"x": 0}]
    assert e2.value.shape == (4, 2)


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def test_select_structural():
    dims = [Axis("x", "variable"), Axis("t", "structural"), Axis("f", "domain")]
    e = Tensor(value=torch.rand(4, 5, 2), axes=dims)
    e2 = e.select("t", 2)
    assert "t" not in e2.axis_names
    assert e2.value.shape == (4, 2)


def test_select_invalid_axis_raises():
    e = Tensor.as_variable("x", torch.rand(4, 2))
    with pytest.raises(ValueError, match="not found"):
        e.select("nonexistent", 0)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def test_align_simple_variable_broadcasting():
    expr_a = Tensor(
        value=torch.randn(3, 4, 1),
        axes=[Axis("x", "variable"), Axis("y", "variable"), Axis("type", "domain")],
        domain_type=Type(shape=(1,), axis_names=["type"]),
    )
    expr_b = Tensor(
        value=torch.randn(4, 5, 1),
        axes=[Axis("y", "variable"), Axis("z", "variable"), Axis("type", "domain")],
        domain_type=Type(shape=(1,), axis_names=["type"]),
    )

    aligned_a, aligned_b = Tensor.align(expr_a, expr_b)

    assert aligned_a.axis_names == ["x", "y", "z", "type"]
    assert aligned_a.value.shape == (3, 4, 1, 1)

    assert aligned_b.axis_names == ["x", "y", "z", "type"]
    assert aligned_b.value.shape == (1, 4, 5, 1)


def test_align_adds_missing_variable_axis():
    e1 = Tensor.as_variable("x", torch.rand(3, 2))
    e2 = Tensor.as_variable("y", torch.rand(4, 2))
    a1, a2 = Tensor.align(e1, e2)
    # Alignment inserts a size-1 broadcast axis for the missing dim
    assert a1.value.shape == (3, 1, 2)
    assert a2.value.shape == (1, 4, 2)
    # After torch broadcasting both can be combined to (3, 4, 2)
    combined = a1.value * a2.value
    assert combined.shape == (3, 4, 2)


def test_align_single_expression_passthrough():
    e = Tensor.as_variable("x", torch.rand(3, 2))
    (aligned,) = Tensor.align(e)
    assert aligned.value.shape == e.value.shape


def test_align_three_expressions():
    e1 = Tensor.as_variable("a", torch.rand(2, 1))
    e2 = Tensor.as_variable("b", torch.rand(3, 1))
    e3 = Tensor.as_variable("c", torch.rand(4, 1))
    a1, a2, a3 = Tensor.align(e1, e2, e3)
    # Each gets broadcast size-1 axes for missing dims
    assert a1.value.shape == (2, 1, 1, 1)
    assert a2.value.shape == (1, 3, 1, 1)
    assert a3.value.shape == (1, 1, 4, 1)
    # After broadcasting, all combine to (2, 3, 4, 1)
    combined = a1.value * a2.value * a3.value
    assert combined.shape == (2, 3, 4, 1)


def test_align_single_expression_is_idempotent():
    """Aligning a single expression with itself returns unchanged values."""
    e1 = Tensor.as_variable("x", torch.rand(3, 2))
    a1, = Tensor.align(e1)
    assert torch.allclose(a1.value, e1.value)


def test_align_overlapping_variable_axis():
    """Two expressions share one variable axis ('y'); canonical order: x, y, z."""
    expr_a = Tensor(
        value=torch.randn(3, 4),
        axes=[Axis("x", "variable"), Axis("y", "variable")],
    )
    expr_b = Tensor(
        value=torch.randn(4, 5),
        axes=[Axis("y", "variable"), Axis("z", "variable")],
    )
    aligned_a, aligned_b = Tensor.align(expr_a, expr_b)
    # Variables sorted alphabetically: x, y, z
    assert aligned_a.axis_names == ["x", "y", "z"]
    assert aligned_a.value.shape == (3, 4, 1)   # missing z → broadcast
    assert aligned_b.axis_names == ["x", "y", "z"]
    assert aligned_b.value.shape == (1, 4, 5)   # missing x → broadcast


def test_align_variable_and_structural_mix():
    """Variables sorted first, then structural; missing dims get broadcast size 1."""
    expr_c = Tensor(
        value=torch.randn(8, 10),
        axes=[Axis("batch", "structural"), Axis("time", "variable")],
    )
    expr_d = Tensor(
        value=torch.randn(8, 6),
        axes=[Axis("batch", "structural"), Axis("other_var", "variable")],
    )
    aligned_c, aligned_d = Tensor.align(expr_c, expr_d)
    # canonical: variables sorted (other_var, time) + structural sorted (batch)
    assert aligned_c.axis_names == ["other_var", "time", "batch"]
    assert aligned_c.value.shape == (1, 10, 8)
    assert aligned_d.axis_names == ["other_var", "time", "batch"]
    assert aligned_d.value.shape == (6, 1, 8)


def test_align_preserves_domain_dims():
    """Domain dims are per-expression and stay at the end after alignment."""
    expr_e = Tensor(
        value=torch.randn(3, 2),
        axes=[Axis("x", "variable"), Axis("d1", "domain")],
    )
    expr_f = Tensor(
        value=torch.randn(4, 7),
        axes=[Axis("y", "variable"), Axis("d2", "domain")],
    )
    aligned_e, aligned_f = Tensor.align(expr_e, expr_f)
    assert aligned_e.axis_names == ["x", "y", "d1"]
    assert aligned_e.value.shape == (3, 1, 2)
    assert aligned_f.axis_names == ["x", "y", "d2"]
    assert aligned_f.value.shape == (1, 4, 7)


def test_align_complex_variable_structural_domain():
    """Variables + structural + domain: each expression keeps its own domain last."""
    expr_g = Tensor(
        value=torch.randn(3, 8, 2),
        axes=[Axis("x", "variable"), Axis("batch", "structural"), Axis("d1", "domain")],
    )
    expr_h = Tensor(
        value=torch.randn(8, 4, 7),
        axes=[Axis("batch", "structural"), Axis("y", "variable"), Axis("d2", "domain")],
    )
    aligned_g, aligned_h = Tensor.align(expr_g, expr_h)
    assert aligned_g.axis_names == ["x", "y", "batch", "d1"]
    assert aligned_g.value.shape == (3, 1, 8, 2)
    assert aligned_h.axis_names == ["x", "y", "batch", "d2"]
    assert aligned_h.value.shape == (1, 4, 8, 7)


# ---------------------------------------------------------------------------
# diag / undiag
# ---------------------------------------------------------------------------

def test_diag_basic():
    e1 = Tensor.as_variable("x", torch.rand(4, 1))
    e2 = Tensor.as_variable("y", torch.rand(4, 1))
    d1, d2 = Tensor.diag(e1, e2)
    # Both should share the same combined variable axis name
    assert d1.free_variables == d2.free_variables
    assert d1.is_diag and d2.is_diag


def test_diag_requires_same_size():
    e1 = Tensor.as_variable("x", torch.rand(4, 1))
    e2 = Tensor.as_variable("y", torch.rand(3, 1))
    with pytest.raises(ValueError, match="same number"):
        Tensor.diag(e1, e2)


def test_diag_requires_exactly_one_variable_axis():
    dims = [Axis("x", "variable"), Axis("y", "variable"), Axis("f", "domain")]
    e = Tensor(value=torch.rand(2, 3, 1), axes=dims)
    e2 = Tensor.as_variable("z", torch.rand(2, 1))
    with pytest.raises(TypeError, match="exactly one variable"):
        Tensor.diag(e, e2)


def test_undiag_restores_original():
    e1 = Tensor.as_variable("x", torch.rand(4, 1))
    e2 = Tensor.as_variable("y", torch.rand(4, 1))
    d1, d2 = Tensor.diag(e1, e2)
    r1, r2 = Tensor.undiag(d1, d2)
    assert r1.free_variables == ["x"]
    assert r2.free_variables == ["y"]


# ---------------------------------------------------------------------------
# flatten_variable_axes
# ---------------------------------------------------------------------------

def test_flatten_variable_axes():
    dims = [Axis("x", "variable"), Axis("y", "variable"), Axis("f", "domain")]
    e = Tensor(value=torch.rand(3, 4, 2), axes=dims)
    flat = e.flatten_variable_axes("flat")
    assert flat.value.shape == (12, 2)
    assert flat.free_variables == ["flat"]


def test_flatten_single_variable_axis_renames():
    e = Tensor.as_variable("x", torch.rand(5, 2))
    flat = e.flatten_variable_axes("individuals")
    assert flat.free_variables == ["individuals"]
    assert flat.value.shape == (5, 2)


# ---------------------------------------------------------------------------
# rename_axes / permute
# ---------------------------------------------------------------------------

def test_rename_axes():
    e = Tensor.as_variable("x", torch.rand(4, 2))
    e2 = e.rename_axes({"x": "z"})
    assert e2.free_variables == ["z"]


def test_permute():
    dims = [Axis("a", "variable"), Axis("b", "structural"), Axis("c", "domain")]
    e = Tensor(value=torch.rand(2, 3, 4), axes=dims)
    e2 = e.permute("b", "a", "c")
    assert e2.axis_names == ["b", "a", "c"]
    assert e2.value.shape == (3, 2, 4)


def test_permute_wrong_names_raises():
    e = Tensor.as_variable("x", torch.rand(4, 2))
    with pytest.raises(ValueError):
        e.permute("z", "x")


# ---------------------------------------------------------------------------
# from_labeled_output
# ---------------------------------------------------------------------------

def test_from_labeled_output_basic():
    t = torch.rand(8, 5, 3)
    e = Tensor.from_labeled_output(t, variable_axis="x", structural_axes={"class": 5})
    assert e.free_variables == ["x"]
    assert e.free_structural_variables == ["class"]
    assert e.axis_names == ["x", "class", "domain_0"]
    assert e.shape == (8, 5, 3)


def test_from_labeled_output_no_domain():
    t = torch.rand(6, 4)
    e = Tensor.from_labeled_output(t, variable_axis="batch", structural_axes={"time": 4})
    assert e.free_variables == ["batch"]
    assert e.free_structural_variables == ["time"]
    assert len(e.axes) == 2


def test_from_labeled_output_size_mismatch_raises():
    t = torch.rand(6, 4)
    with pytest.raises(ValueError, match="structural axis 'time'"):
        Tensor.from_labeled_output(t, variable_axis="x", structural_axes={"time": 99})


# ---------------------------------------------------------------------------
# Type Axis Names Initialization
# ---------------------------------------------------------------------------

def test_type_axis_names_initialization():
    
    # 1D shape with len(shape) == 1
    t1 = Type(name="Feature", shape=(5,))
    assert t1.axis_names == ["feature"]
    
    # ND shape with len(shape) > 1
    t2 = Type(name="Matrix", shape=(3, 4))
    assert t2.axis_names == ["matrix_0", "matrix_1"]
    
    # Scalar shape with len(shape) == 0
    t0 = Type(name="Scalar", shape=())
    assert t0.axis_names == []

    # Standard types
    assert Real.axis_names == ["real"]
    assert Bool.axis_names == ["bool"]

    # Product type axis names propagation
    p_type = product_type(Real, (2, 3), axis_names=["x", "y"])
    assert p_type.axis_names == ["x", "y", "real"]


def test_expression_domain_names_compatibility():
    
    # Matching domain axis name should succeed
    t = torch.randn(3, 2)
    custom_type = Type(shape=(2,), axis_names=["feat"])
    e1 = Tensor(
        value=t,
        axes=[Axis("x", "variable"), Axis("feat", "domain")],
        domain_type=custom_type
    )
    assert e1.domain_type.axis_names == ["feat"]

    # Mismatching domain axis name should raise ValueError
    with pytest.raises(ValueError, match="Domain axis names .* do not match"):
        Tensor(
            value=t,
            axes=[Axis("x", "variable"), Axis("wrong_feat", "domain")],
            domain_type=custom_type
        )
    
    # When domain_type is None, the auto-generated Type should adopt the given domain names
    e2 = Tensor(
        value=t,
        axes=[Axis("x", "variable"), Axis("custom_name", "domain")],
        domain_type=None
    )
    assert e2.domain_type.axis_names == ["custom_name"]

