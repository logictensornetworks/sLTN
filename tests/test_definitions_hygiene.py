"""Regression tests for definition (macro) expansion: capture, aliases, selections, output axes."""
import pytest
import torch

from sltn import Signature, Interpretation, Real


def _names(phi) -> str:
    return repr(phi)


def test_body_quantifier_does_not_capture_argument():
    sig = Signature("s")
    sig.sort("E")
    sig.variable("y", "E")
    sig.predicate("R2", ["E", "E"])
    sig.define("Rel", ["a"], "exists y: R2(a, y)")
    phi = sig.parse("forall y: Rel(y)")
    text = _names(phi)
    assert "R2(y, y)" not in text
    assert "y'" in text


def test_declared_name_is_not_a_numbered_alias():
    sig = Signature("s")
    sig.sort("E")
    sig.variable("y", "E")
    sig.constant("c1", "E")
    sig.predicate("R2", ["E", "E"])
    sig.define("D", ["c"], "R2(c, c1)")
    assert "R2(y, c1)" in _names(sig.parse("D(y)"))


def test_selection_axis_follows_call_site_axis():
    sig = Signature("s")
    sig.sort("E")
    sig.dimension("S")
    sig.structural_variable("s", "S")
    sig.structural_variable("t", "S")
    sig.variable("X", "E", dims=["S"])
    sig.variable("x", "E", dims=["S"])
    sig.predicate("C", ["E"])
    sig.define("First", ["X"], "C(X[s][s=0])")
    phi = sig.parse("forall x: First(x[t])")
    interp = Interpretation(sig)
    interp[sig.get_sort("E")] = Real
    interp[sig.get_variable("x")] = torch.rand(3, 4, 1)
    interp[sig.get_predicate("C")] = lambda e: torch.sigmoid(e)
    out = interp(phi)
    assert out.value.numel() == 1


def test_call_site_output_axes_apply_to_body_root():
    sig = Signature("s")
    sig.sort("E")
    sig.dimension("T")
    sig.structural_variable("t", "T")
    sig.variable("x", "E")
    sig.predicate("PerStep", ["E"], output_dims=["T"])
    sig.define("PS", ["a"], "PerStep(a)")
    sig.parse("forall x: forall t: PS(x)[t]")


def test_call_site_output_axes_on_compound_body_rejected():
    sig = Signature("s")
    sig.sort("E")
    sig.dimension("T")
    sig.structural_variable("t", "T")
    sig.variable("x", "E")
    sig.predicate("PerStep", ["E"], output_dims=["T"])
    sig.define("Both", ["a"], "PerStep(a) & PerStep(a)")
    with pytest.raises(ValueError, match="output axes"):
        sig.parse("forall x: forall t: Both(x)[t]")


def test_parameter_with_different_axis_annotations():
    sig = Signature("s")
    sig.sort("E")
    sig.dimension("S")
    sig.structural_variable("s", "S")
    sig.structural_variable("t", "S")
    sig.variable("X", "E", dims=["S"])
    sig.variable("x", "E", dims=["S"])
    sig.predicate("C", ["E"])
    sig.structural_relation("next", ["S", "S"])
    sig.define("Inc", ["X"], "forall s1 | next(s, s1): (C(X[s]) -> C(X[s1]))")
    text = _names(sig.parse("forall x: forall t: Inc(x[t])"))
    assert "x[t]" in text and "x[s1]" in text
