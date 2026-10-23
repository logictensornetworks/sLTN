"""
Unit tests for smooth fuzzy operators and gradient-safe clamping functions.

Tests cover:
1. Smooth clamping output bounds
2. Gradient flow at boundaries
3. Smooth vs. hard conditional comparison
4. ImpliesGodel and ImpliesGoguen with smooth=True/False
"""

import torch
from sltn.interpretation.operators import (
    smooth_clamp,
    ImpliesGodel,
    ImpliesGoguen,
)


# ---------------------------------------------------------------------------
# smooth_clamp
# ---------------------------------------------------------------------------

def test_smooth_clamp_output_bounds():
    """smooth_clamp keeps output in [0, 1]."""
    x = torch.tensor([-10.0, -1.0, 0.0, 0.5, 1.0, 10.0])
    y = smooth_clamp(x, min=0.0, max=1.0)

    assert torch.all(y >= 0.0), "smooth_clamp output should be >= 0"
    assert torch.all(y <= 1.0), "smooth_clamp output should be <= 1"


def test_smooth_clamp_gradient_nonzero_everywhere():
    """Gradients are non-zero in the active region (key property)."""
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 1.0, 1.5], requires_grad=True)
    y = smooth_clamp(x, min=0.0, max=1.0)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should exist"
    assert x.grad[2] != 0.0, "Gradient at x=0 should be non-zero"
    assert x.grad[3] != 0.0, "Gradient at x=0.5 should be non-zero"
    assert x.grad[4] != 0.0, "Gradient at x=1 should be non-zero"
    assert x.grad[5] != 0.0, "Gradient at x=1.5 should be non-zero"


def test_smooth_clamp_monotonicity():
    """smooth_clamp is monotonically increasing."""
    x = torch.linspace(-10, 10, 100)
    y = smooth_clamp(x, min=0.0, max=1.0)

    diffs = y[1:] - y[:-1]
    assert torch.all(diffs >= -1e-6), "smooth_clamp should be monotonically increasing"


def test_smooth_clamp_approaches_hard_clamp_as_beta_increases():
    """smooth_clamp converges to torch.clamp for large beta."""
    x = torch.tensor([-2.0, -0.1, 0.2, 0.8, 1.3, 4.0])
    hard = torch.clamp(x, min=0.0, max=1.0)
    smooth = smooth_clamp(x, min=0.0, max=1.0, beta=200.0)
    assert torch.allclose(smooth, hard, atol=5e-3), \
        "smooth_clamp should converge to hard clamp for large beta"


def test_smooth_clamp_beta_parameter():
    """The beta parameter affects the transition steepness."""
    x = torch.tensor([-0.2, 0.2, 0.8, 1.2])

    y_sharp = smooth_clamp(x, min=0.0, max=1.0, beta=50.0)
    y_smooth = smooth_clamp(x, min=0.0, max=1.0, beta=1.0)

    assert not torch.allclose(y_sharp, y_smooth), \
        "Different beta values should produce different values"


def test_smooth_clamp_no_nan_inf():
    """smooth_clamp doesn't produce NaN or Inf for extreme inputs."""
    extreme_values = torch.tensor([-1e6, -1e3, 0.0, 1e3, 1e6], dtype=torch.float32)
    result = smooth_clamp(extreme_values, min=0.0, max=1.0)

    assert not torch.isnan(result).any(), "smooth_clamp should not produce NaN"
    assert not torch.isinf(result).any(), "smooth_clamp should not produce Inf"


# ---------------------------------------------------------------------------
# ImpliesGodel(smooth=...)
# ---------------------------------------------------------------------------

def test_implies_godel_smooth_output_bounds():
    """ImpliesGodel(smooth=True) produces outputs in [0, 1]."""
    x = torch.tensor([0.0, 0.3, 0.7, 1.0])
    y = torch.tensor([0.5, 0.5, 0.5, 0.5])

    op_smooth = ImpliesGodel(smooth=True)
    result = op_smooth(x, y)

    assert torch.all(result >= 0.0) and torch.all(result <= 1.0), \
        "ImpliesGodel(smooth=True) output should be in [0, 1]"


def test_implies_godel_smooth_vs_hard_semantics():
    """Smooth version approximates hard version semantics away from the boundary."""
    x = torch.tensor([0.1, 0.2, 0.9])
    y = torch.tensor([0.9, 0.8, 0.1])  # Clear ordering: y > x, y > x, x > y

    op_smooth = ImpliesGodel(smooth=True, beta=100.0)  # High beta -> closer to hard
    op_hard = ImpliesGodel(smooth=False)

    result_smooth = op_smooth(x, y)
    result_hard = op_hard(x, y)

    diff = torch.abs(result_smooth - result_hard)
    assert torch.all(diff < 0.15), "High beta should approximate hard version away from boundary"


def test_implies_godel_smooth_gradients_at_boundary():
    """Gradients are non-zero at the x=y boundary for the smooth version."""
    x = torch.tensor([0.5], requires_grad=True)
    y = torch.tensor([0.5], requires_grad=True)

    op = ImpliesGodel(smooth=True)
    result = op(x, y)
    loss = result.sum()
    loss.backward()

    assert x.grad is not None and abs(x.grad) > 1e-6, \
        "Smooth ImpliesGodel should have non-zero gradient w.r.t. x at boundary"
    assert y.grad is not None and abs(y.grad) > 1e-6, \
        "Smooth ImpliesGodel should have non-zero gradient w.r.t. y at boundary"


def test_implies_godel_hard_gradients_at_boundary():
    """Hard version has zero gradient at the x=y boundary (baseline for the smooth test above)."""
    x = torch.tensor([0.5], requires_grad=True)
    y = torch.tensor([0.5], requires_grad=True)

    op = ImpliesGodel(smooth=False)
    result = op(x, y)
    loss = result.sum()
    loss.backward()

    # At x == y, torch.where(x <= y, 1, y) takes the constant branch (1), so the
    # result does not depend on x at all (grad stays None, never accumulated)
    # and depends on y only through the discarded branch (grad is exactly 0).
    assert x.grad is None, \
        "Hard ImpliesGodel should have no gradient w.r.t. x at the boundary (unused branch)"
    assert y.grad is not None and y.grad.item() == 0.0, \
        "Hard ImpliesGodel should have zero gradient w.r.t. y at the boundary"


def test_implies_godel_hard_mode_matches_torch_where():
    """ImpliesGodel(smooth=False) matches the hard (non-smooth) semantics."""
    x = torch.tensor([0.1, 0.5, 0.9])
    y = torch.tensor([0.2, 0.5, 0.8])

    op_hard = ImpliesGodel(smooth=False)
    result = op_hard(x, y)

    expected = torch.where(torch.le(x, y), torch.ones_like(x), y)

    assert torch.allclose(result, expected), \
        "Hard mode should match torch.where behavior"


def test_implies_godel_defaults_to_hard():
    """ImpliesGodel defaults to smooth=False."""
    x = torch.tensor([0.5], requires_grad=True)
    y = torch.tensor([0.5], requires_grad=True)

    op = ImpliesGodel()  # No arguments
    result = op(x, y)
    assert result is not None
    assert op.smooth is False


def test_implies_godel_beta_affects_smooth_transition():
    """The beta parameter affects ImpliesGodel(smooth=True) behavior."""
    x = torch.tensor([0.4, 0.5, 0.6])
    y = torch.tensor([0.5, 0.5, 0.5])

    op_smooth = ImpliesGodel(smooth=True, beta=5.0)
    op_sharp = ImpliesGodel(smooth=True, beta=50.0)

    result_smooth = op_smooth(x, y)
    result_sharp = op_sharp(x, y)

    assert not torch.allclose(result_smooth, result_sharp), \
        "Different beta values should produce different results"


def test_implies_godel_smooth_no_nan_inf():
    """Numerical stability of smooth ImpliesGodel for extreme inputs."""
    x = torch.tensor([-1e6, -1e3, 0.0, 1e3, 1e6], dtype=torch.float32)
    y = torch.tensor([1e-6, 1e-3, 0.5, 1e3, 1e6], dtype=torch.float32)

    op = ImpliesGodel(smooth=True)
    result = op(x, y)

    assert not torch.isnan(result).any(), "Result should not contain NaN"
    assert not torch.isinf(result).any(), "Result should not contain Inf"


# ---------------------------------------------------------------------------
# ImpliesGoguen(smooth=...)
# ---------------------------------------------------------------------------

def test_implies_goguen_smooth_output_bounds():
    """ImpliesGoguen(smooth=True) with clamped division produces outputs in [0, 1]."""
    x = torch.tensor([0.3, 0.5, 0.7, 1.0])
    y = torch.tensor([0.5, 0.5, 0.5, 0.5])

    op_smooth = ImpliesGoguen(smooth=True)
    result = op_smooth(x, y)

    assert torch.all(result >= 0.0) and torch.all(result <= 1.0), \
        "ImpliesGoguen(smooth=True) output should be in [0, 1] (clamped division)"


def test_implies_goguen_smooth_protects_from_division_by_zero():
    """smooth=True uses pi_0 to prevent division singularities near x=0."""
    x_near_zero = torch.tensor([1e-10, 1e-8, 1e-6])
    y = torch.tensor([0.5, 0.5, 0.5])

    op = ImpliesGoguen(smooth=True)
    result = op(x_near_zero, y)

    assert not torch.isnan(result).any(), "Result should not contain NaN"
    assert not torch.isinf(result).any(), "Result should not contain Inf"
    assert torch.all(result >= 0.0) and torch.all(result <= 1.0), \
        "Result should be in [0, 1] due to clamping"


def test_implies_goguen_smooth_vs_stable_hard():
    """Smooth version approximates the stable hard version for high beta."""
    x = torch.tensor([0.2, 0.5, 0.9])
    y = torch.tensor([0.3, 0.5, 0.8])

    op_smooth = ImpliesGoguen(smooth=True, beta=100.0)
    op_stable_hard = ImpliesGoguen(smooth=False, stable=True)

    result_smooth = op_smooth(x, y)
    result_hard = op_stable_hard(x, y)

    diff = torch.abs(result_smooth - result_hard)
    assert torch.all(diff < 0.15), "High beta should approximate stable hard version"


def test_implies_goguen_smooth_gradients_at_boundary():
    """Gradients are non-zero at the x=y boundary for the smooth version."""
    x = torch.tensor([0.5], requires_grad=True)
    y = torch.tensor([0.5], requires_grad=True)

    op = ImpliesGoguen(smooth=True)
    result = op(x, y)
    loss = result.sum()
    loss.backward()

    assert x.grad is not None and abs(x.grad) > 1e-6, \
        "Smooth ImpliesGoguen should have non-zero gradient w.r.t. x at boundary"
    assert y.grad is not None and abs(y.grad) > 1e-6, \
        "Smooth ImpliesGoguen should have non-zero gradient w.r.t. y at boundary"


def test_implies_goguen_defaults_to_hard():
    """ImpliesGoguen defaults to smooth=False."""
    x = torch.tensor([0.5], requires_grad=True)
    y = torch.tensor([0.5], requires_grad=True)

    op = ImpliesGoguen()  # No arguments, should still support the stable parameter
    result = op(x, y)
    assert result is not None
    assert op.smooth is False


def test_implies_goguen_smooth_no_nan_inf():
    """Numerical stability of smooth ImpliesGoguen for extreme inputs."""
    x = torch.tensor([1e-10, 1e-6, 0.1, 0.5, 1.0], dtype=torch.float32)
    y = torch.tensor([0.1, 0.2, 0.5, 0.9, 1.0], dtype=torch.float32)

    op = ImpliesGoguen(smooth=True)
    result = op(x, y)

    assert not torch.isnan(result).any(), "Result should not contain NaN"
    assert not torch.isinf(result).any(), "Result should not contain Inf"


def test_smooth_clamp_with_nonzero_min():
    """Bug #49: the two-sided branch shifted the output down by min."""
    x = torch.tensor([-5.0, 0.5, 0.7, 5.0])
    y = smooth_clamp(x, min=0.2, max=1.0, beta=1000.0)
    assert torch.allclose(y, torch.tensor([0.2, 0.5, 0.7, 1.0]), atol=1e-3)


# ---------------------------------------------------------------------------
# Stable Łukasiewicz operators
# ---------------------------------------------------------------------------

def test_stable_lukasiewicz_operators_stay_in_unit_interval():
    """Bug #50: AndLuk(stable=True)(1, 1) was 1.024, so OrLuk / ImpliesLuk went negative."""
    from sltn.interpretation.operators import AndLuk, OrLuk, ImpliesLuk
    v = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    x, y = torch.meshgrid(v, v, indexing="ij")
    for op in (AndLuk(stable=True), OrLuk(stable=True), ImpliesLuk(stable=True)):
        r = op(x, y)
        assert torch.all((r >= 0.0) & (r <= 1.0)), op
    one, zero = torch.tensor(1.0), torch.tensor(0.0)
    assert AndLuk(stable=True)(one, one) == 1.0
    assert OrLuk(stable=True)(zero, zero) == 0.0
    assert ImpliesLuk(stable=True)(one, zero) == 0.0


def test_stable_lukasiewicz_and_in_a_formula_with_crisp_values():
    """Bug #50: a crisp A(x) & B(x) failed the Bool check."""
    from sltn import Signature, Interpretation, Type
    from sltn.interpretation import Logic
    from sltn.interpretation.operators import AndLuk, OrLuk, ImpliesLuk
    sig = Signature("luk")
    sig.sort("S"); sig.variable("x", "S")
    sig.predicate("A", ["S"]); sig.predicate("B", ["S"])
    i = Interpretation(sig)
    i.logic = Logic(And=AndLuk(stable=True), Or=OrLuk(stable=True), Implies=ImpliesLuk(stable=True))
    i["S"] = Type("S", shape=(1,), axis_names=["feat"])
    i["x"] = torch.rand(4, 1)
    i["A"] = lambda a: torch.ones_like(a)
    i["B"] = lambda a: torch.ones_like(a)
    assert torch.all(i("A(x) & B(x)").value == 1.0)
    assert torch.all(i("A(x) -> B(x)").value <= 1.0)


# ---------------------------------------------------------------------------
# Small connective issues (bug #52)
# ---------------------------------------------------------------------------

def test_implies_godel_smooth_default_beta_matches_goguen():
    assert ImpliesGodel(smooth=True).beta == ImpliesGoguen(smooth=True).beta
    assert ImpliesGodel(smooth=True)(torch.tensor(1.0), torch.tensor(0.0)) < 0.01


def test_not_godel_preserves_dtype():
    from sltn.interpretation.operators import NotGodel
    x = torch.tensor([0.0, 0.5], dtype=torch.float64)
    assert NotGodel()(x).dtype == torch.float64


def test_lukasiewicz_reprs_are_complete():
    from sltn.interpretation.operators import AndLuk, ImpliesLuk
    assert "eps=" in repr(AndLuk())
    assert "smoothing_method=" in repr(ImpliesLuk())
