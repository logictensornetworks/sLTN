import pytest
import torch

import sltn.interpretation.operators as fuzzy_ops


def test_andprod_matches_tensor_product() -> None:
    op = fuzzy_ops.AndProd(stable=False)
    x = torch.tensor([0.2, 0.5, 1.0])
    y = torch.tensor([0.3, 0.1, 0.7])

    out = op(x, y)
    assert torch.allclose(out, x * y)


def test_andprod_stable_avoids_exact_zero() -> None:
    op = fuzzy_ops.AndProd(stable=True, eps=1e-6)
    x = torch.tensor([0.0])
    y = torch.tensor([0.0])

    out = op(x, y)
    assert out.item() > 0.0


def test_implies_goguen_is_one_when_x_leq_y() -> None:
    op = fuzzy_ops.ImpliesGoguen(stable=False)
    x = torch.tensor([0.1, 0.3, 0.6])
    y = torch.tensor([0.2, 0.3, 0.9])

    out = op(x, y)
    assert torch.allclose(out, torch.ones_like(x))


@pytest.mark.parametrize("stable", [False, True])
def test_implies_goguen_at_zero_antecedent(stable) -> None:
    op = fuzzy_ops.ImpliesGoguen(stable=stable)
    x = torch.tensor([0.0, 0.0, 0.5], requires_grad=True)
    y = torch.tensor([0.0, 0.3, 0.25], requires_grad=True)
    out = op(x, y)
    assert out[0].item() == 1.0 and out[1].item() == 1.0
    assert out[2].item() == pytest.approx(0.25 / (0.5 if not stable else 0.5 * (1 - op.eps) + op.eps), rel=1e-5)
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()


def test_aggreg_pmeanerror_definition() -> None:
    x = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    p = 2.0

    op = fuzzy_ops.AggregPMeanError(p=p, stable=False)
    pmean = fuzzy_ops.AggregPMean(p=p, stable=False)

    lhs = op(x, dim=0)
    rhs = 1.0 - pmean(1.0 - x, dim=0)

    assert torch.allclose(lhs, rhs)


# ---------------------------------------------------------------------------
# Conjunction
# ---------------------------------------------------------------------------

def test_andmin_matches_elementwise_minimum() -> None:
    op = fuzzy_ops.AndMin()
    x = torch.tensor([0.2, 0.9, 0.5])
    y = torch.tensor([0.7, 0.1, 0.5])

    assert torch.equal(op(x, y), torch.minimum(x, y))


def test_andpmean_p1_is_arithmetic_mean() -> None:
    op = fuzzy_ops.AndPMean(p=1.0, stable=False)
    x = torch.tensor([0.2, 0.6])
    y = torch.tensor([0.4, 0.8])

    assert torch.allclose(op(x, y), (x + y) / 2.0)


def test_andpmean_gradient_flows() -> None:
    op = fuzzy_ops.AndPMean(p=2.0, stable=True)
    x = torch.tensor([0.3, 0.5], requires_grad=True)
    y = torch.tensor([0.4, 0.6], requires_grad=True)

    op(x, y).sum().backward()
    assert x.grad is not None and torch.all(x.grad != 0.0)


def test_andcos_matches_definition() -> None:
    op = fuzzy_ops.AndCos()
    x = torch.tensor([0.3, 0.6])
    y = torch.tensor([0.4, 0.7])

    expected = torch.clamp(x * y - torch.sqrt(1 - x**2) * torch.sqrt(1 - y**2), min=0.0)
    assert torch.allclose(op(x, y), expected)
    assert torch.all(op(x, y) >= 0.0)


def test_andluk_hard_matches_definition() -> None:
    op = fuzzy_ops.AndLuk(stable=False)
    x = torch.tensor([0.3, 0.8, 1.0])
    y = torch.tensor([0.5, 0.5, 1.0])

    expected = torch.clamp(x + y - 1.0, min=0.0)
    assert torch.allclose(op(x, y), expected)


def test_andluk_call_stable_override() -> None:
    """Per-call `stable` override behaves like the constructor flag."""
    op = fuzzy_ops.AndLuk(stable=False)
    x = torch.tensor([0.0])
    y = torch.tensor([0.0])

    hard = op(x, y)
    smooth = op(x, y, stable=True)
    assert torch.allclose(hard, torch.zeros_like(x))
    assert smooth.item() != hard.item() or smooth.item() >= 0.0


# ---------------------------------------------------------------------------
# Disjunction
# ---------------------------------------------------------------------------

def test_ormax_matches_elementwise_maximum() -> None:
    op = fuzzy_ops.OrMax()
    x = torch.tensor([0.2, 0.9, 0.5])
    y = torch.tensor([0.7, 0.1, 0.5])

    assert torch.allclose(op(x, y), torch.maximum(x, y))


def test_orprobsum_matches_probabilistic_sum() -> None:
    op = fuzzy_ops.OrProbSum()
    x = torch.tensor([0.2, 0.5])
    y = torch.tensor([0.3, 0.4])

    expected = x + y - x * y
    assert torch.allclose(op(x, y), expected)


def test_orluk_is_dual_of_andluk() -> None:
    or_op = fuzzy_ops.OrLuk()
    and_op = fuzzy_ops.AndLuk()
    x = torch.tensor([0.3, 0.6])
    y = torch.tensor([0.5, 0.7])

    expected = 1.0 - and_op(1.0 - x, 1.0 - y)
    assert torch.allclose(or_op(x, y), expected)


def test_xorclassical_matches_definition() -> None:
    and_op = fuzzy_ops.AndMin()
    neg_op = fuzzy_ops.NotStandard()
    op = fuzzy_ops.XOrClassical(and_op=and_op, neg_op=neg_op)
    x = torch.tensor([0.2, 0.9])
    y = torch.tensor([0.9, 0.9])

    expected = and_op(neg_op(and_op(neg_op(x), neg_op(y))), neg_op(and_op(x, y)))
    assert torch.allclose(op(x, y), expected)


def test_xorclassical_crisp_truth_table() -> None:
    """At crisp (0/1) truth values, XorClassical reduces to the classical XOR table."""
    op = fuzzy_ops.XOrClassical(and_op=fuzzy_ops.AndMin(), neg_op=fuzzy_ops.NotStandard())
    x = torch.tensor([0.0, 0.0, 1.0, 1.0])
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])

    expected = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert torch.allclose(op(x, y), expected)


# ---------------------------------------------------------------------------
# Implication / equivalence
# ---------------------------------------------------------------------------

def test_implies_kleene_dienes_matches_definition() -> None:
    op = fuzzy_ops.ImpliesKleeneDienes()
    x = torch.tensor([0.2, 0.8])
    y = torch.tensor([0.5, 0.3])

    expected = torch.maximum(1.0 - x, y)
    assert torch.allclose(op(x, y), expected)


def test_implies_reichenbach_matches_definition() -> None:
    op = fuzzy_ops.ImpliesReichenbach(stable=False)
    x = torch.tensor([0.2, 0.6])
    y = torch.tensor([0.5, 0.3])

    expected = 1.0 - x + x * y
    assert torch.allclose(op(x, y), expected)


def test_implies_luk_matches_definition() -> None:
    op = fuzzy_ops.ImpliesLuk(stable=False)
    x = torch.tensor([0.2, 0.8, 0.5])
    y = torch.tensor([0.5, 0.3, 0.5])

    expected = torch.clamp(1.0 - x + y, max=1.0)
    assert torch.allclose(op(x, y), expected)


def test_equiv_similarity_matches_definition() -> None:
    op = fuzzy_ops.EquivSimilarity(p=1, stable=False)
    x = torch.tensor([0.2, 0.8, 0.5])
    y = torch.tensor([0.5, 0.8, 0.1])

    expected = 1.0 - torch.abs(x - y)
    assert torch.allclose(op(x, y), expected)


def test_equiv_similarity_applies_pi0_only_when_stable() -> None:
    # float64, so that the eps shift of pi_0 is visible
    x = torch.tensor([0.4], dtype=torch.float64)

    assert torch.equal(fuzzy_ops.EquivSimilarity(stable=False)(x, x), torch.ones_like(x))
    assert torch.all(fuzzy_ops.EquivSimilarity(stable=True)(x, x) < 1.0)
    assert torch.equal(fuzzy_ops.EquivSimilarity(stable=True)(x, x, stable=False), torch.ones_like(x))


def test_equiv_similarity_is_symmetric() -> None:
    op = fuzzy_ops.EquivSimilarity(p=2, stable=True)
    x = torch.tensor([0.1, 0.9])
    y = torch.tensor([0.7, 0.2])

    assert torch.allclose(op(x, y), op(y, x))


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

def test_notgodel_matches_definition() -> None:
    op = fuzzy_ops.NotGodel()
    x = torch.tensor([0.0, 0.1, 0.5, 1.0])

    expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.equal(op(x), expected)


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------

def test_aggregmin_default_is_true_minimum() -> None:
    op = fuzzy_ops.AggregMin()
    x = torch.tensor([[0.2, 0.8, 0.5]])

    assert torch.allclose(op(x, dim=1), torch.tensor([0.2]))


def test_aggregmin_bottom_k_averages_smallest() -> None:
    op = fuzzy_ops.AggregMin(bottom_k=2)
    x = torch.tensor([0.9, 0.1, 0.5, 0.3])

    assert torch.allclose(op(x, dim=0), torch.tensor((0.1 + 0.3) / 2))


def test_aggregmax_default_is_true_maximum() -> None:
    op = fuzzy_ops.AggregMax()
    x = torch.tensor([[0.2, 0.8, 0.5]])

    assert torch.allclose(op(x, dim=1), torch.tensor([0.8]))


def test_aggregmax_top_k_averages_largest() -> None:
    op = fuzzy_ops.AggregMax(top_k=2)
    x = torch.tensor([0.9, 0.1, 0.5, 0.3])

    assert torch.allclose(op(x, dim=0), torch.tensor((0.9 + 0.5) / 2))


def test_agreggeometricmean_matches_definition() -> None:
    op = fuzzy_ops.AggregGeometricMean(stable=False)
    x = torch.tensor([0.2, 0.5, 0.8])

    expected = torch.exp(torch.mean(torch.log(x)))
    assert torch.allclose(op(x, dim=0), expected)


def test_agreggeometricmean_masked_excludes_zero_weight_entries() -> None:
    op = fuzzy_ops.AggregGeometricMean(stable=False)
    x = torch.tensor([0.2, 0.9, 0.5])
    mask = torch.tensor([True, False, True])

    masked_result = op(x, dim=0, mask=mask)
    unmasked_result = op(x[mask], dim=0)
    assert torch.allclose(masked_result, unmasked_result, atol=1e-4)


def test_default_logic_uses_default_eps():
    # Logic.with_defaults hardcoded eps=1e-12, a no-op in float32.
    from sltn.interpretation import Logic
    from sltn.interpretation.operators import DEFAULT_EPS
    L = Logic().with_defaults()
    for name in ("And", "Or", "Implies", "Forall", "Exists", "SatAgg"):
        assert getattr(L, name).eps == DEFAULT_EPS, name


def test_and_cos_stable_gradient_is_finite_at_one():
    # sqrt(1 - x^2) has an infinite derivative at 1.
    x = torch.tensor(1.0, requires_grad=True)
    fuzzy_ops.AndCos(stable=True)(x, torch.tensor(0.5)).backward()
    assert torch.isfinite(x.grad)


@pytest.mark.parametrize("op", [fuzzy_ops.AggregPMean(), fuzzy_ops.AggregGeometricMean()])
def test_masked_mean_is_exact_and_empty_guard_is_finite(op):
    # The masked means divided by (sum of weights + 1e-6), so a full guard over ones gave < 1.
    xs = torch.ones(5)
    assert op(xs, dim=0, mask=torch.ones(5, dtype=torch.bool)) == 1.0
    assert torch.isfinite(op(xs, dim=0, mask=torch.zeros(5, dtype=torch.bool)))


# AggregMin(bottom_k) / AggregMax(top_k) with k > 1

def test_extremum_k_larger_than_reduced_size_averages_everything():
    x = torch.tensor([0.9, 0.2, 0.6])
    assert torch.isclose(fuzzy_ops.AggregMin(bottom_k=5)(x, dim=0), x.mean())


def test_extremum_ignores_masked_entries_when_fewer_than_k_are_valid():
    x = torch.tensor([0.9, 0.2, 0.6])
    assert torch.isclose(fuzzy_ops.AggregMin(bottom_k=2)(x, dim=0, mask=torch.tensor([False, True, False])), torch.tensor(0.2))
    # a valid 1.0 must not lose a tie against the masked-out entry
    y = torch.tensor([0.2, 1.0, 0.5])
    assert torch.isclose(fuzzy_ops.AggregMin(bottom_k=2)(y, dim=0, mask=torch.tensor([True, True, False])), torch.tensor(0.6))
    empty = torch.zeros(3, dtype=torch.bool)
    assert fuzzy_ops.AggregMin(bottom_k=2)(x, dim=0, mask=empty) == 1.0
    assert fuzzy_ops.AggregMax(top_k=2)(x, dim=0, mask=empty) == 0.0


def test_extremum_over_several_dims_takes_k_extremes_jointly():
    x = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    assert torch.isclose(fuzzy_ops.AggregMin(bottom_k=2)(x, dim=[0, 1]), torch.tensor(0.15))
    X = torch.rand(4, 5, 6)
    assert torch.allclose(fuzzy_ops.AggregMin()(X, dim=[1, 2]), X.amin(dim=(1, 2)))
    assert fuzzy_ops.AggregMax()(X, dim=[1, 2], keepdim=True).shape == (4, 1, 1)


# Weighted p-means

def test_weighted_pmean_is_exact_with_tiny_weights():
    x = torch.tensor([0.9, 0.2, 0.6])
    w = torch.full((3,), 1e-9)
    assert torch.isclose(fuzzy_ops.AggregPMean(p=1)(x, dim=0, mask=w), x.mean())
    assert torch.isclose(fuzzy_ops.AggregPMeanError(p=1)(x, dim=0, mask=w), x.mean())


def test_pmean_with_p_zero_is_the_geometric_mean():
    x = torch.tensor([0.9, 0.2, 0.6])
    assert torch.isclose(fuzzy_ops.AggregPMean(p=0)(x, dim=0), fuzzy_ops.AggregGeometricMean()(x, dim=0))


# An aggregator built by hand (logic["∀"] = AggregPMeanError()) must be as safe
# as the default one: no NaN gradient when every value is at the boundary.
@pytest.mark.parametrize("op, boundary", [
    (fuzzy_ops.AggregPMeanError(), 1.0),     # every individual satisfies the clause
    (fuzzy_ops.AggregPMean(), 0.0),
    (fuzzy_ops.AggregPMean(p=0), 0.0),
], ids=["pMeanError at 1", "pMean at 0", "geometric mean at 0"])
def test_default_aggregator_gradient_is_finite_at_the_boundary(op, boundary):
    xs = torch.full((4,), boundary, requires_grad=True)
    op(xs, dim=[0]).backward()
    assert torch.isfinite(xs.grad).all()
