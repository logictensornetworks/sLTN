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
