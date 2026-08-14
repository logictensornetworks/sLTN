"""Focused unit tests for sltn.interpretation.apply: apply_sat_agg and the
mask / NaN-handling branches of apply_quantifier.

Formula tensors passed through `apply_quantifier`/`apply_sat_agg` always carry
a trailing Bool domain axis (as produced by `apply_connective`), so the
low-level fixtures below include one explicitly.
"""
import pytest
import torch

from sltn import Signature, Interpretation, Type
from sltn.interpretation.core import Axis, Tensor, Bool
from sltn.interpretation.apply import apply_sat_agg, apply_quantifier
import sltn.interpretation.operators as fuzzy_ops


def _bool_tensor(values: list[float] | list[bool], var_name: str = "x") -> Tensor:
    """Build a 1-variable Bool-domain Tensor, e.g. as produced by apply_connective."""
    data = torch.tensor(values).unsqueeze(-1)
    return Tensor(
        value=data,
        axes=[Axis(var_name, "variable"), Axis("bool", "domain")],
        domain_type=Bool,
    )


# ---------------------------------------------------------------------------
# apply_sat_agg
# ---------------------------------------------------------------------------

def test_apply_sat_agg_requires_at_least_one_formula():
    with pytest.raises(ValueError, match="at least one formula"):
        apply_sat_agg(fuzzy_ops.AggregPMean(p=2.0))


def test_apply_sat_agg_aggregates_clause_means():
    f1 = _bool_tensor([0.2, 0.8])
    f2 = _bool_tensor([0.6], var_name="y")

    agg_op = fuzzy_ops.AggregPMean(p=1.0, stable=False)
    result = apply_sat_agg(agg_op, f1, f2)

    expected = agg_op(torch.stack([f1.value.mean(), f2.value.mean()]), dim=0)
    assert result.value.shape == (1,)
    assert torch.allclose(result.value, expected.unsqueeze(0))
    assert result.domain_type is Bool


def test_apply_sat_agg_single_formula_matches_its_own_mean():
    f1 = _bool_tensor([0.3, 0.7, 0.5])
    agg_op = fuzzy_ops.AggregMin()

    result = apply_sat_agg(agg_op, f1)
    assert torch.allclose(result.value, f1.value.mean().unsqueeze(0))


# ---------------------------------------------------------------------------
# apply_quantifier — mask handling
# ---------------------------------------------------------------------------

def test_apply_quantifier_bool_mask_excludes_positions():
    formula = _bool_tensor([0.2, 0.8, 0.0])
    mask = Tensor(
        value=torch.tensor([True, True, False]).unsqueeze(-1),
        axes=[Axis("x", "variable"), Axis("bool", "domain")],
        domain_type=Bool,
    )
    agg_op = fuzzy_ops.AggregMin()

    result = apply_quantifier(agg_op, formula, ["x"], mask=mask, symbol="∀")
    # The masked-out 0.0 entry must not affect the min: min(0.2, 0.8) = 0.2.
    assert torch.allclose(result.value, torch.tensor([0.2]))


def test_apply_quantifier_soft_mask_weights_pmean():
    formula = _bool_tensor([0.2, 0.8])
    soft_mask = _bool_tensor([1.0, 0.0])
    agg_op = fuzzy_ops.AggregPMean(p=1.0, stable=False)

    result = apply_quantifier(agg_op, formula, ["x"], mask=soft_mask, symbol="∀")
    # Weight-1 entry dominates entirely: weighted mean collapses to it.
    assert torch.allclose(result.value, torch.tensor([0.2]), atol=1e-4)


def test_apply_quantifier_guard_via_interpretation():
    """End-to-end: a `forall x | Guard(x): P(x)` guard is passed through as `mask`."""
    sig = Signature("guarded")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.predicate("Guard", ["S"])
    x = sig.variable("x", S)

    interp = Interpretation(sig)
    interp[S] = Type("S", shape=(1,))
    interp[x] = torch.tensor([[1.0], [2.0], [3.0]])
    interp[sig.get_predicate("P")] = lambda v: torch.sigmoid(-v)  # decreasing in v
    interp[sig.get_predicate("Guard")] = lambda v: (v > 1.5).float()

    guarded = interp(sig.parse("forall x | Guard(x): P(x)"))
    unguarded = interp(sig.parse("forall x: P(x)"))
    # Excluding the smallest (most-satisfying) element via the guard must not
    # increase the aggregated satisfaction relative to the unguarded case.
    assert guarded.value.item() <= unguarded.value.item() + 1e-6


# ---------------------------------------------------------------------------
# apply_quantifier — NaN handling
# ---------------------------------------------------------------------------

def test_apply_quantifier_nan_values_are_masked_out():
    # Bool's constraint rejects NaN, so build this tensor without domain_type
    # validation (mirroring a formula that produced NaN mid-computation).
    formula = Tensor(
        value=torch.tensor([0.2, float("nan"), 0.8]).unsqueeze(-1),
        axes=[Axis("x", "variable"), Axis("bool", "domain")],
    )
    agg_op = fuzzy_ops.AggregMin()

    result = apply_quantifier(agg_op, formula, ["x"], symbol="∀")
    assert not torch.isnan(result.value).any()
    assert torch.allclose(result.value, torch.tensor([0.2]))


def test_apply_quantifier_all_nan_falls_back_to_identity():
    formula = Tensor(
        value=torch.tensor([float("nan"), float("nan")]).unsqueeze(-1),
        axes=[Axis("x", "variable"), Axis("bool", "domain")],
    )
    agg_op = fuzzy_ops.AggregPMean(p=2.0, stable=True)

    result_forall = apply_quantifier(agg_op, formula, ["x"], symbol="∀")
    assert not torch.isnan(result_forall.value).any()

    result_exists = apply_quantifier(agg_op, formula, ["x"], symbol="∃")
    assert not torch.isnan(result_exists.value).any()


def test_apply_quantifier_vacuous_axis_uses_identity_value():
    """Aggregating over a size-0 axis returns the quantifier's identity element."""
    formula = Tensor(
        value=torch.empty(0, 1),
        axes=[Axis("x", "variable"), Axis("bool", "domain")],
        domain_type=Bool,
    )
    agg_op = fuzzy_ops.AggregMin()

    forall_result = apply_quantifier(agg_op, formula, ["x"], symbol="∀")
    assert torch.allclose(forall_result.value, torch.ones(1))

    exists_result = apply_quantifier(agg_op, formula, ["x"], symbol="∃")
    assert torch.allclose(exists_result.value, torch.zeros(1))
