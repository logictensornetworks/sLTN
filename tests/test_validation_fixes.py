"""Regression tests for signature validation, axis-name checks, quantifiers and Interpretation fixes."""
import pytest
import torch

from sltn import Signature, Interpretation, Real
from sltn.interpretation import Logic
from sltn.interpretation.operators import AggregMin, AggregPMeanError


# ---------------------------------------------------------------------------
# Signature (redeclaration, #17, #19, #21)
# ---------------------------------------------------------------------------

def test_redeclaration_warns_and_replaces():
    sig = Signature("s")
    sig.sort("E")
    sig.predicate("P", ["E"])
    with pytest.warns(UserWarning, match="redeclared"):
        new = sig.predicate("P", ["E", "E"])
    assert sig.get_predicate("P") is new


def test_foreign_sort_object_rejected():
    sig, other = Signature("a"), Signature("b")
    sig.sort("E")
    foreign = other.sort("E")
    with pytest.raises(KeyError, match="not registered"):
        sig.variable("x", foreign)


def test_non_identifier_name_rejected():
    sig = Signature("s")
    sig.sort("E")
    with pytest.raises(ValueError):
        sig.predicate("my pred", ["E"])


def test_structural_variable_requires_a_name():
    sig = Signature("s")
    T = sig.dimension("T")
    with pytest.raises(TypeError):
        sig.structural_variable(T)


def test_define_rejects_duplicate_parameters():
    sig = Signature("s")
    with pytest.raises(ValueError, match="duplicate"):
        sig.define("D", ["a", "a"], "a")


# ---------------------------------------------------------------------------
# Axis names (#25, #41, #43, #62)
# ---------------------------------------------------------------------------

@pytest.fixture
def tsig():
    sig = Signature("t")
    sig.sort("E")
    T = sig.dimension("T")
    sig.structural_variable("t", T)
    sig.variable("x", "E", dims=[T])
    sig.variable("y", "E")
    sig.predicate("C", ["E"])
    sig.predicate("G", ["E"])
    sig.structural_relation("next", [T, T])
    return sig


def test_too_many_annotations_rejected(tsig):
    with pytest.raises(ValueError, match="axis annotation"):
        tsig.parse("forall x: forall t, t1: C(x[t, t1])[t, t1]")


def test_unknown_annotation_rejected(tsig):
    with pytest.raises(NameError, match="not a structural variable"):
        tsig.parse("forall x: forall t: C(x[q])[t]")


def test_relation_argument_count_checked(tsig):
    with pytest.raises(ValueError, match="takes 2 argument"):
        tsig.parse("forall t: next(t)")


def test_relation_argument_must_be_a_bare_name(tsig):
    with pytest.raises(ValueError, match="must be structural variable names"):
        tsig.parse("forall t, t1: next(t, t1[t])")


def test_selection_on_default_axis_name_rejected(tsig):
    with pytest.raises(ValueError, match="default axis name"):
        tsig.parse("forall x: C(x[T=0])")
    tsig.parse("forall x: C(x[t][t=0])")


# ---------------------------------------------------------------------------
# Quantifiers (#1, #14, #58)
# ---------------------------------------------------------------------------

def _interp(sig):
    i = Interpretation(sig)
    i["E"] = Real
    i["x"] = torch.rand(3, 5, 1)
    i["y"] = torch.rand(4, 1)
    i["C"] = lambda e: torch.sigmoid(e)
    i["G"] = lambda e: torch.sigmoid(e)
    i.logic = Logic(StructAgg={"T": {"forall": AggregPMeanError(p=2), "exists": AggregMin()}})
    return i


def test_mixed_quantifier_rejected(tsig):
    with pytest.raises(ValueError, match="mixes individual"):
        tsig.parse("forall x, t: C(x[t])[t]")


def test_guard_not_spanning_quantified_axis(tsig):
    i = _interp(tsig)
    guarded = i("forall x | G(y): C(x)").value
    plain = i("forall x: C(x)").value
    # G(y) is constant along x: per individual y, the weights cancel.
    assert torch.allclose(guarded, plain.unsqueeze(0).expand_as(guarded), atol=1e-6)


# ---------------------------------------------------------------------------
# Logic keys (#55)
# ---------------------------------------------------------------------------

def test_logic_struct_key_typo_rejected():
    L = Logic()
    with pytest.raises(KeyError, match="Unknown quantifier"):
        L["froall,T"] = AggregMin()
    L["∀,T"] = AggregMin()
    assert isinstance(L["forall,T"], AggregMin)


# ---------------------------------------------------------------------------
# Interpretation (#64)
# ---------------------------------------------------------------------------

def test_ground_unknown_name_rejected(tsig):
    i = Interpretation(tsig)
    with pytest.raises(KeyError, match="Cannot ground 'c'"):
        i["c"] = lambda e: e
    with pytest.raises(KeyError, match="Cannot ground"):
        Interpretation(tsig, overrides={"nope": 1})


def test_constructor_checks_tensor_shape(tsig):
    with pytest.raises(ValueError, match="Shape mismatch"):
        Interpretation(tsig, overrides={"E": Real, "y": torch.rand(4, 3)})


def test_alias_lookup_returns_base_grounding(tsig):
    tsig.parse("forall y1: C(y1)")
    i = _interp(tsig)
    assert i["y1"] is i["y"]
    with pytest.raises(KeyError, match="alias"):
        i["y1"] = torch.rand(4, 1)


def test_logic_set_in_classical_mode_is_kept(tsig):
    i = _interp(tsig)
    i.classical = True
    L = Logic(StructAgg={"T": {"forall": AggregMin(), "exists": AggregMin()}})
    i.logic = L
    i.classical = False
    assert isinstance(i.logic.StructAgg["T"]["forall"], AggregMin)


def test_copies_do_not_share_logic(tsig):
    i = _interp(tsig)
    for other in (i.copy(), i.deepcopy(), i.with_updates(), Interpretation.merge(i, i)):
        assert other.logic is not i.logic
        other.logic["forall,T"] = AggregMin()
        assert isinstance(i.logic["forall,T"], AggregPMeanError)
