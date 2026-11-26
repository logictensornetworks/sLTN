"""Tests for signature validation, axis-name checks, quantifiers, and Interpretation groundings."""
import sltn
import pytest
import torch

from sltn import Signature, Interpretation, Real
from sltn.interpretation import Logic
from sltn.interpretation.operators import AggregMax, AggregMin, AggregPMean, AggregPMeanError


# ---------------------------------------------------------------------------
# Signature (redeclaration, foreign sorts, names, structural variables, definitions)
# ---------------------------------------------------------------------------

def test_redeclaration_warns_and_replaces():
    sig = Signature("s")
    sig.sort("E")
    sig.predicate("P", ["E"])
    with pytest.warns(
        UserWarning, match=r"^Predicate 'P' is already declared; replacing its declaration\.$"
    ):
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
# Axis names (annotations, structural-relation arguments, selections)
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
    with pytest.raises(ValueError, match="names 2 structural"):
        tsig.parse("forall x: forall t, t1: C(x[t, t1])")


def test_unknown_annotation_rejected(tsig):
    with pytest.raises(NameError, match="not a structural variable"):
        tsig.parse("forall x: forall t: C(x[q])")


def test_relation_argument_count_checked(tsig):
    with pytest.raises(ValueError, match="takes 2 argument"):
        tsig.parse("forall t: next(t)")


def test_relation_argument_must_be_a_bare_name(tsig):
    with pytest.raises(ValueError, match="must be structural variable names"):
        tsig.parse("forall t, t1: next(t, t1[t])")


def test_selection_on_dimension_name_rejected(tsig):
    with pytest.raises(ValueError, match="not a structural axis"):
        tsig.parse("forall x: C(x[t][T=0])")
    tsig.parse("forall x: C(x[t][t=0])")


# ---------------------------------------------------------------------------
# Quantifiers (mixed quantifiers, guards)
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
        tsig.parse("forall x, t: C(x[t])")


def test_guard_not_spanning_quantified_axis(tsig):
    i = _interp(tsig)
    guarded = i("forall x | G(y): C(x[t])").value
    plain = i("forall x: C(x[t])").value
    # G(y) is constant along x: per individual y, the weights cancel.
    assert torch.allclose(guarded, plain.unsqueeze(0).expand_as(guarded), atol=1e-6)


# ---------------------------------------------------------------------------
# Logic keys
# ---------------------------------------------------------------------------

def test_logic_struct_key_typo_rejected():
    L = Logic()
    with pytest.raises(KeyError, match="Unknown quantifier"):
        L["froall,T"] = AggregMin()
    L["∀,T"] = AggregMin()
    assert isinstance(L["forall,T"], AggregMin)


# ---------------------------------------------------------------------------
# Interpretation
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


@pytest.mark.parametrize("value", [
    lambda: torch.eye(5),
    torch.eye(5).tolist(),
    torch.nn.Linear(5, 5),
], ids=["function", "list", "module"])
def test_structural_relation_must_be_grounded_with_a_tensor(tsig, value):
    # Structural relations accept only tensors (or Parameters) and reject callables, lists, or modules at grounding time.
    i = _interp(tsig)
    i["next"] = torch.eye(5)
    with pytest.raises(TypeError, match=r"Structural relation 'next' must be grounded with a tensor .*\(T x T\), got \w+"):
        i["next"] = value
    assert torch.equal(i["next"], torch.eye(5))         # the old grounding is kept
    with pytest.raises(TypeError, match="Structural relation 'next'"):
        Interpretation(tsig, overrides={"next": value})


def test_structural_relation_accepts_tensors_and_parameters(tsig):
    i = _interp(tsig)
    i["next"] = torch.eye(5)
    i["next"] = torch.nn.Parameter(torch.eye(5))
    assert i("next(t, t1)").value.shape == (5, 5, 1)


def test_structural_default_follows_current_forall_exists(tsig):
    i = Interpretation(tsig)
    i.logic["forall"] = AggregMin()
    assert i.logic["forall,T"] is i.logic.Forall
    assert i.logic["exists,t"] is i.logic.Exists
    assert i.logic.StructAgg is None


def test_structural_override_wins_over_default(tsig):
    i = Interpretation(tsig)
    override = AggregPMeanError(p=4)
    i.logic["forall,t"] = override
    i.logic["forall"] = AggregMin()
    assert i.logic["forall,T"] is override
    assert i.logic["exists,T"] is i.logic.Exists


def test_dimension_declared_after_interpretation_uses_default(tsig):
    i = Interpretation(tsig)
    U = tsig.dimension("U")
    tsig.structural_variable("u", U)
    i.logic["exists"] = AggregMax()
    assert i.logic["exists,u"] is i.logic.Exists


def test_structural_op_rejects_undeclared_axis(tsig):
    with pytest.raises(KeyError, match="Unknown structural axis"):
        Interpretation(tsig).logic["forall,nope"]


def test_with_updates_accepts_logic_and_classical_struct_quantifiers(tsig):
    # Without overrides, structural quantifiers use the first-order Forall/Exists.
    fresh = Interpretation(tsig)
    assert isinstance(fresh.logic["forall,T"], AggregPMeanError)
    assert isinstance(fresh.logic["exists,T"], AggregPMean)

    i = _interp(tsig)
    i["C"] = lambda z: torch.full((z.shape[0], 1), 0.7)
    crisp = i.with_updates(logic=Logic.classical())
    assert isinstance(crisp.logic["forall,T"], AggregMin)
    assert isinstance(crisp.logic["exists,T"], AggregMax)
    assert crisp("forall x: forall t: C(x[t])").value.item() == 1.0
    low = i.with_updates(logic=Logic.classical(threshold=0.8))
    assert low("forall x: forall t: C(x[t])").value.item() == 0.0
    godel = i.with_updates(logic=Logic.classical(threshold=None))
    assert godel("forall x: forall t: C(x[t])").value.item() == pytest.approx(0.7)
    # Original interpretation keeps its fuzzy logic untouched.
    assert i.logic._threshold is None
    assert 0.0 < i("forall x: forall t: C(x[t])").value.item() < 1.0


def test_classical_threshold_binarises_structural_relations(tsig):
    i = _interp(tsig)
    i["next"] = torch.full((5, 5), 0.7)
    assert torch.all(i("next(t, t1)").value == 0.7)
    crisp = i.with_updates(logic=Logic.classical())
    assert torch.all(crisp("next(t, t1)").value == 1.0)
    strict = i.with_updates(logic=Logic.classical(threshold=0.8))
    assert torch.all(strict("next(t, t1)").value == 0.0)


def test_copies_do_not_share_logic(tsig):
    i = _interp(tsig)
    for other in (i.copy(), i.deepcopy(), i.with_updates(), Interpretation.merge(i, i)):
        assert other.logic is not i.logic
        other.logic["forall,T"] = AggregMin()
        assert isinstance(i.logic["forall,T"], AggregPMeanError)


# ---------------------------------------------------------------------------
# Arguments of predicates and functions: number and sorts
# ---------------------------------------------------------------------------

@pytest.fixture
def asig():
    sig = Signature("args")
    sig.sort("Animal")
    sig.sort("Person")
    sig.sort("Embedding")
    sig.variable("x", "Animal")
    sig.variable("z", "Animal")
    sig.variable("p", "Person")
    sig.constant("Fido", "Animal")
    sig.constant("Bob", "Person")
    sig.predicate("Near", ["Animal"])
    sig.predicate("Owns", ["Person", "Animal"])
    sig.predicate("Pb", ["Bool"])
    sig.predicate("Rain", [])
    sig.predicate("%%", ["Animal", "Animal"], infix=True)
    sig.function("embed", ["Animal"], "Embedding")
    sig.function("diff", ["Animal", "Animal"], "Animal")
    return sig


@pytest.mark.parametrize("formula, symbol, expected, got", [
    ("Near(x, z)", "Near", 1, 2),
    ("Owns(p)", "Owns", 2, 1),
    ("Near()", "Near", 1, 0),
    ("Rain(x)", "Rain", 0, 1),
    ("Near(diff(x))", "diff", 2, 1),
    ("diff(x, z, x) %% x", "diff", 2, 3),
    ("forall x: Near(embed(x, x))", "embed", 1, 2),
])
def test_wrong_number_of_arguments_rejected(asig, formula, symbol, expected, got):
    with pytest.raises(TypeError, match=rf"'{symbol}' expects {expected} argument\(s\), got {got}"):
        asig.parse(formula)


@pytest.mark.parametrize("formula, symbol, position, got, expected", [
    ("Near(p)", "Near", 0, "Person", "Animal"),            # variable of the wrong sort
    ("Near(Bob)", "Near", 0, "Person", "Animal"),          # constant of the wrong sort
    ("Owns(x, p)", "Owns", 0, "Animal", "Person"),         # arguments swapped
    ("Near(embed(x))", "Near", 0, "Embedding", "Animal"),  # function output of the wrong sort
    ("embed(p)", "embed", 0, "Person", "Animal"),          # inside a term
    ("forall x: (Near(x) -> Near(embed(x)))", "Near", 0, "Embedding", "Animal"),
    ("forall p1: Near(p1)", "Near", 0, "Person", "Animal"),   # alias of p
    ("Near(Near(x))", "Near", 0, "Bool", "Animal"),        # a formula is a Bool argument
    ("Pb(x)", "Pb", 0, "Animal", "Bool"),
    ("x %% p", "%%", 1, "Person", "Animal"),                 # infix predicate
])
def test_argument_of_wrong_sort_rejected(asig, formula, symbol, position, got, expected):
    with pytest.raises(TypeError, match=rf"'{symbol}': argument {position} has sort '{got}', expected '{expected}'"):
        asig.parse(formula)


@pytest.mark.parametrize("formula", [
    "Owns(p, x)", "Owns(Bob, Fido)", "forall x1: Near(x1)", "Pb(Near(x))", "Pb(forall x: Near(x))",
    "Near(diff(x, Fido))", "x %% Fido", "Rain", "Rain()", "forall x: exists z: (x %% z)",
])
def test_arguments_of_the_right_sort_accepted(asig, formula):
    asig.parse(formula)


def test_macro_body_is_checked_after_expansion(asig):
    # A macro without input_sorts doesn't check its arguments; its expanded body is checked.
    asig.define("Both", ["a", "b"], "Near(a) & Near(b)")
    asig.parse("Both(x, Fido)")
    with pytest.raises(TypeError, match=r"'Near': argument 0 has sort 'Person'"):
        asig.parse("Both(x, p)")


def test_failed_argument_check_leaves_no_alias(asig):
    with pytest.raises(TypeError):
        asig.parse("forall p1: Near(p1)")
    assert "p1" not in asig.variables


@pytest.mark.parametrize("declare", [
    lambda s: s.predicate("<<", ["Animal"], infix=True),
    lambda s: s.predicate("<<", ["Animal", "Animal", "Animal"], infix=True),
    lambda s: s.function("++", ["Animal"], "Animal", infix=True),
    lambda s: s.define("+++", ["a", "b", "c"], "Near(a)", infix=True),
])
def test_infix_symbol_needs_two_arguments(asig, declare):
    with pytest.raises(ValueError, match="must take exactly 2 arguments"):
        declare(asig)
    assert "<<" not in asig.predicates and "++" not in asig.functions and "+++" not in asig.definitions


def test_definition_cannot_be_grounded(asig):
    asig.define("NearToo", ["a"], "Near(a)")
    with pytest.raises(KeyError, match="is a definition"):
        Interpretation(asig)["NearToo"] = lambda v: v
    with pytest.raises(KeyError, match="is a definition"):
        Interpretation(asig, NearToo=lambda v: v)


def test_classical_threshold_binarises_bool_variables_and_constants():
    sig = Signature("b")
    sig.sort("B")
    sig.variable("v", "B")
    sig.constant("c", "B")
    i = Interpretation(sig)
    i["B"] = sltn.Bool
    i["v"] = torch.tensor([[0.7], [0.9]])
    i["c"] = torch.tensor([0.6])
    assert i("forall v: v").value.item() < 1.0
    crisp = i.with_updates(logic=Logic.classical())
    assert crisp("forall v: v").value.item() == 1.0
    assert crisp("c & c").value.item() == 1.0


def test_classical_threshold_leaves_function_arguments_unchanged():
    sig = Signature("b")
    sig.sort("B")
    sig.variable("v", "B")
    sig.function("f", ["B"], "B")
    seen = []
    i = Interpretation(sig)
    i["B"] = sltn.Bool
    i["v"] = torch.tensor([[0.7]])
    i["f"] = lambda z: (seen.append(z.clone()), z)[1]
    i.with_updates(logic=Logic.classical())("forall v: f(v)")
    assert torch.allclose(seen[-1], torch.tensor([[0.7]]))


@pytest.mark.parametrize("quantifier, last", [("forall", 1.0), ("exists", 0.0)])
def test_guard_without_successor_gives_quantifier_neutral(quantifier, last):
    # The last instant has no successor: weak next under ∀, strong next under ∃,
    # also when the ∀ aggregator alone would give 0 (pMean with p < 1).
    sig = Signature("next")
    sig.sort("E")
    T = sig.dimension("T")
    sig.structural_variable("t", T)
    sig.constant("c", "E", dims=[T])
    sig.predicate("P", ["E", "E"])
    sig.structural_relation("next", [T, T])
    i = Interpretation(sig, overrides={"E": Real, "c": torch.rand(4, 1)})
    i.logic = Logic(Forall=AggregPMean(p=0.5))
    i["P"] = lambda a, b: torch.sigmoid(a - b)
    i["next"] = torch.diag(torch.ones(3), 1)
    out = i(f"{quantifier} t1 | next(t, t1): P(c[t], c[t1])")
    assert out.value[-1].item() == last
