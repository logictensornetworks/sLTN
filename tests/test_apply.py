"""Focused unit tests for sltn.interpretation.apply: apply_sat_agg and the
mask branches of apply_quantifier.

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
# apply_quantifier — NaN is not treated as missing data
# ---------------------------------------------------------------------------

def test_apply_quantifier_does_not_mask_nan():
    # The "NaN = missing data" branch and the fallback that
    # replaced a non-finite result with the identity were removed. NaN now
    # propagates and fails the Bool check of the result instead of silently
    # becoming 1 (forall) or 0 (exists); missing data must be excluded with a guard.
    formula = Tensor(
        value=torch.tensor([0.2, float("nan"), 0.8]).unsqueeze(-1),
        axes=[Axis("x", "variable"), Axis("bool", "domain")],
    )
    for symbol in ("∀", "∃"):
        with pytest.raises(ValueError, match="constraints of type 'Bool'"):
            apply_quantifier(fuzzy_ops.AggregPMean(p=2.0), formula, ["x"], symbol=symbol)


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


# ---------------------------------------------------------------------------
# Consumed axes: every argument receives them, and failures say so
# ---------------------------------------------------------------------------

def _consuming_predicate(n_videos: int, n_labels: int):
    """`P` consumes `T`: `x` ranges over `T`, the label `y` and the constant `c` don't."""
    sig = Signature("consumed")
    sig.sort("S")
    sig.sort("D")
    sig.dimension("T")
    sig.structural_variable("t", "T")
    sig.variable("x", "S", dims=["T"])
    sig.variable("y", "D")
    sig.constant("c", "D")
    sig.predicate("P", ["S", "D"], input_dims=["T"], output_dims=[])
    i = Interpretation(sig)
    i["S"] = Type("R1", shape=(1,))
    i["D"] = Type("R2", shape=(2,))
    i["x"] = torch.rand(n_videos, 5, 1)
    i["y"] = torch.rand(n_labels, 2)
    i["c"] = torch.rand(2)
    return i


def _reduce_first(a, b):
    """Reduces `T` before using the label, so it fails on the label's copy of `T`."""
    return torch.sigmoid((a.mean(1) * b).sum(-1, keepdim=True))


def test_argument_without_a_consumed_axis_receives_a_copy_of_it():
    i = _consuming_predicate(3, 3)
    seen = []

    def grounding(a, b):
        seen.extend([a.shape, b.shape])
        return torch.sigmoid((a * b).sum(-1).mean(1, keepdim=True))

    i["P"] = grounding
    assert i("P(x[t], y)").value.shape == (3, 3, 1)
    assert seen == [(9, 5, 1), (9, 5, 2)]


def test_grounding_can_drop_the_copy_of_a_consumed_axis():
    i = _consuming_predicate(3, 3)
    i["P"] = lambda a, b: _reduce_first(a, b.select(1, 0))
    assert i("P(x[t], y)").value.shape == (3, 3, 1)


def test_failing_grounding_is_told_about_the_copied_axis():
    i = _consuming_predicate(3, 3)
    i["P"] = _reduce_first
    with pytest.raises(RuntimeError) as err:
        i("P(x[t], y)")
    message = str(err.value)
    assert "'P'" in message
    assert "(9, 5, 2)" in message
    assert "argument 2" in message and "'t'" in message
    assert "select(1, 0)" in message
    assert "must match the size" in message  # the original error is kept


def test_copied_axis_note_covers_constants():
    i = _consuming_predicate(3, 1)
    i["P"] = _reduce_first
    with pytest.raises(RuntimeError, match="argument 2"):
        i("P(x[t], c)")


def test_grounding_failing_without_a_copied_axis_keeps_its_error():
    i = _consuming_predicate(3, 3)
    i.signature.variable("z", "D", dims=["T"])  # has the consumed axis: nothing is copied
    i["z"] = torch.rand(3, 5, 2)

    def grounding(a, b):
        raise RuntimeError("broken network")

    i["P"] = grounding
    with pytest.raises(RuntimeError, match="^broken network$"):
        i("P(x[t], z[t])")


def test_wrong_output_shape_when_batch_equals_the_consumed_size():
    # The reduced `a` broadcasts against the label's copy of `T`, so the
    # grounding itself succeeds and only its output shape is wrong.
    i = _consuming_predicate(1, 5)
    i["P"] = _reduce_first
    with pytest.raises(ValueError) as err:
        i("P(x[t], y)")
    message = str(err.value)
    assert "'P'" in message
    assert "(5, 5, 1)" in message and "(5, 1)" in message
    assert "argument 2" in message


def test_wrong_output_shape_names_the_symbol():
    i = _consuming_predicate(3, 3)
    i["P"] = lambda a, b: torch.sigmoid(a.mean(1)).sum(-1)  # no keepdim
    with pytest.raises(ValueError, match=r"'P' returned shape \(9,\), expected \(9, 1\)"):
        i("P(x[t], y)")


def test_wrong_number_of_rows_names_the_symbol():
    i = _consuming_predicate(3, 3)
    i["P"] = lambda a, b: torch.full((3, 1), 0.5)
    with pytest.raises(ValueError, match=r"'P' returned shape \(3, 1\), expected \(9, 1\)"):
        i("P(x[t], y)")


def test_swapped_batch_and_kept_axis_are_not_merged_by_the_reshape():
    sig = Signature("swapped")
    sig.sort("S")
    sig.dimension("T")
    sig.structural_variable("t", "T")
    sig.variable("x", "S", dims=["T"])
    sig.predicate("Q", ["S"], input_dims=["T"], output_dims=["T"])
    i = Interpretation(sig)
    i["S"] = Type("R1", shape=(1,))
    i["x"] = torch.rand(4, 3, 1)
    i["Q"] = lambda v: torch.sigmoid(v).permute(1, 0, 2)  # (3, 4, 1): same size as (4, 3, 1)
    with pytest.raises(ValueError, match=r"'Q' returned shape \(3, 4, 1\), expected \(4, 3, 1\)"):
        i("Q(x[t])[t]")
