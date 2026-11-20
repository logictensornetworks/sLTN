"""
Unit tests for the exact (non-smoothed) fuzzy operators and for the experimental,
unexported ``smooth_clamp`` helper kept in ``operators/utils.py``.
"""

import pytest
import torch
from sltn.interpretation.operators import ImpliesGodel, ImpliesGoguen, AndLuk, OrLuk, ImpliesLuk
from sltn.interpretation.operators.utils import smooth_clamp

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
# Operators carry no smoothing options
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [ImpliesGodel, ImpliesGoguen, AndLuk, OrLuk, ImpliesLuk])
@pytest.mark.parametrize("kw", ["smooth", "beta", "smoothing_method"])
def test_operators_reject_smoothing_options(cls, kw):
    with pytest.raises(TypeError):
        cls(**{kw: 1})


def test_smooth_clamp_not_exported():
    import sltn.interpretation.operators as ops
    assert not hasattr(ops, "smooth_clamp") or "smooth_clamp" not in ops.__all__


def test_godel_and_goguen_are_exact():
    v = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    x, y = torch.meshgrid(v, v, indexing="ij")
    godel = torch.where(x <= y, torch.ones_like(x), y)
    assert torch.equal(ImpliesGodel()(x, y), godel)
    r = ImpliesGoguen()(x, y)
    assert torch.all(r[x <= y] == 1.0)
    assert torch.allclose(r[x > y], (y / x)[x > y], atol=1e-5)


def test_stable_lukasiewicz_is_exact_up_to_pi0():
    zero, half, one = torch.tensor(0.0), torch.tensor(0.5), torch.tensor(1.0)
    assert AndLuk()(zero, zero) == 0.0
    assert AndLuk()(half, half) < 1e-5
    assert torch.isclose(AndLuk()(one, torch.tensor(0.75)), torch.tensor(0.75), atol=1e-5)
    assert torch.isclose(ImpliesLuk()(zero, zero), one, atol=1e-5)


def test_stable_lukasiewicz_gradient_is_finite():
    x = torch.rand(50, requires_grad=True)
    y = torch.rand(50, requires_grad=True)
    AndLuk()(x, y).sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()



# ---------------------------------------------------------------------------
# Stable Łukasiewicz operators
# ---------------------------------------------------------------------------

def test_stable_lukasiewicz_operators_stay_in_unit_interval():
    """AndLuk(stable=True)(1, 1) was 1.024, so OrLuk / ImpliesLuk went negative."""
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
    """A crisp A(x) & B(x) failed the Bool check."""
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
# Small connective issues
# ---------------------------------------------------------------------------

def test_not_godel_preserves_dtype():
    from sltn.interpretation.operators import NotGodel
    x = torch.tensor([0.0, 0.5], dtype=torch.float64)
    assert NotGodel()(x).dtype == torch.float64


def test_lukasiewicz_reprs_are_complete():
    assert repr(AndLuk()) == "AndLuk(stable=True, eps=1e-06)"
    assert repr(ImpliesLuk()) == "ImpliesLuk(stable=True, eps=1e-06)"
