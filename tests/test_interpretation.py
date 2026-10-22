"""Tests for Interpretation: grounding, evaluation, logic, copy, device."""
import pytest
import torch
import torch.nn as nn
from sltn import Signature, Interpretation, Tensor, Type, Real, Bool
from sltn.interpretation import Logic
import sltn.interpretation.operators as fuzzy_ops


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_sig():
    sig = Signature("demo")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.predicate("Q", ["S", "S"])
    sig.variable("x", S)
    sig.variable("y", S)
    sig.constant("c", S)
    return sig


@pytest.fixture
def simple_interp(simple_sig):
    sig = simple_sig
    interp = Interpretation(sig)
    interp[sig.get_sort("S")] = Real
    interp[sig.get_variable("x")] = torch.rand(5, 1)
    interp[sig.get_variable("y")] = torch.rand(4, 1)
    interp[sig.get_constant("c")] = torch.rand(1)
    interp[sig.get_predicate("P")] = lambda v: torch.sigmoid(v)
    interp[sig.get_predicate("Q")] = lambda a, b: torch.sigmoid(a * b)
    return interp


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------

def test_ground_and_getitem(simple_sig, simple_interp):
    sig = simple_sig
    val = simple_interp[sig.get_predicate("P")]
    assert callable(val)


def test_setitem_shorthand(simple_sig):
    sig = simple_sig
    interp = Interpretation(sig)
    fn = lambda v: torch.sigmoid(v)
    interp[sig.get_predicate("P")] = fn
    assert interp[sig.get_predicate("P")] is fn


def test_getitem_missing_raises(simple_sig):
    sig = simple_sig
    interp = Interpretation(sig)
    with pytest.raises(KeyError):
        _ = interp[sig.get_predicate("P")]


def test_ground_returns_self(simple_sig):
    sig = simple_sig
    interp = Interpretation(sig)
    result = interp.ground(sig.get_predicate("P"), lambda v: v)
    assert result is interp


# ---------------------------------------------------------------------------
# Formula evaluation
# ---------------------------------------------------------------------------

def test_eval_atomic(simple_sig, simple_interp):
    sig = simple_sig
    phi = sig.parse("P(x)")
    result = simple_interp(phi)
    assert isinstance(result, Tensor)
    assert result.value.shape[0] == 5  # batch dim matches x


def test_eval_forall(simple_sig, simple_interp):
    sig = simple_sig
    phi = sig.parse("forall x: P(x)")
    result = simple_interp(phi)
    assert result.value.shape == torch.Size([1])


def test_eval_exists(simple_sig, simple_interp):
    sig = simple_sig
    phi = sig.parse("exists x: P(x)")
    result = simple_interp(phi)
    assert result.value.shape == torch.Size([1])


def test_eval_and(simple_sig, simple_interp):
    sig = simple_sig
    phi = sig.parse("P(x) & P(x)")
    result = simple_interp(phi)
    assert result.value.shape[0] == 5


def test_eval_implies(simple_sig, simple_interp):
    sig = simple_sig
    phi = sig.parse("P(x) -> P(x)")
    result = simple_interp(phi)
    # implication of same thing should be 1
    assert (result.value > 0.99).all()


def test_eval_not(simple_sig, simple_interp):
    sig = simple_sig
    phi = sig.parse("not P(x)")
    result = simple_interp(phi)
    assert result.value.shape[0] == 5


def test_eval_string_formula(simple_sig, simple_interp):
    result = simple_interp("forall x: P(x)")
    assert isinstance(result, Tensor)


def test_eval_missing_variable_raises(simple_sig):
    sig = simple_sig
    interp = Interpretation(sig)
    # P is not bound
    phi = sig.parse("P(x)")
    with pytest.raises((RuntimeError, ValueError)):
        interp(phi)


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

def test_logic_default_operators(simple_interp):
    logic = simple_interp.logic
    assert logic.And is not None
    assert logic.Forall is not None
    assert logic.SatAgg is not None


def test_logic_setter_fills_defaults():
    sig = Signature("s")
    interp = Interpretation(sig)
    interp.logic = Logic(And=fuzzy_ops.AndMin())
    assert isinstance(interp.logic.And, fuzzy_ops.AndMin)
    assert interp.logic.Forall is not None  # defaults filled in


def test_logic_derives_connectives_from_and_not():
    """When And and Not are both set, Or/Implies/Equiv are derived classically."""
    and_op = fuzzy_ops.AndProd(stable=True, eps=1e-10)
    not_op = fuzzy_ops.NotStandard()
    logic = Logic(And=and_op, Not=not_op).with_defaults()

    assert isinstance(logic.Or, fuzzy_ops.OrDual)
    assert logic.Or.and_op is and_op
    assert logic.Or.neg_op is not_op

    assert isinstance(logic.Implies, fuzzy_ops.ImpliesDual)
    assert logic.Implies.and_op is and_op
    assert logic.Implies.neg_op is not_op

    assert isinstance(logic.Equiv, fuzzy_ops.Equiv)
    assert logic.Equiv.and_op is and_op
    assert logic.Equiv.implies_op is logic.Implies


def test_logic_field_mutation(simple_interp):
    simple_interp.logic.Forall = fuzzy_ops.AggregPMean(p=4)
    assert isinstance(simple_interp.logic.Forall, fuzzy_ops.AggregPMean)


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

def test_copy_shares_tensors(simple_interp, simple_sig):
    copy = simple_interp.copy()
    sig = simple_sig
    orig_x = simple_interp[sig.get_variable("x")]
    copy_x = copy[sig.get_variable("x")]
    assert isinstance(orig_x, torch.Tensor) and isinstance(copy_x, torch.Tensor)
    # A shallow copy() must share the underlying tensor storage (same identity).
    assert orig_x.data_ptr() == copy_x.data_ptr()


def test_deepcopy_clones_tensors(simple_interp, simple_sig):
    deep = simple_interp.deepcopy()
    sig = simple_sig
    orig = simple_interp[sig.get_variable("x")]
    copy = deep[sig.get_variable("x")]
    assert isinstance(orig, torch.Tensor) and isinstance(copy, torch.Tensor)
    # A deepcopy() must clone the underlying tensor storage (different identity).
    assert orig.data_ptr() != copy.data_ptr()


# ---------------------------------------------------------------------------
# with_updates
# ---------------------------------------------------------------------------

def test_with_updates_overrides_grounding(simple_sig, simple_interp):
    sig = simple_sig
    new_x = torch.rand(3, 1)
    updated = simple_interp.with_updates(**{sig.get_variable("x").name: new_x})
    # The updated interpretation has the new x
    val = updated[sig.get_variable("x")]
    assert isinstance(val, torch.Tensor) and val.shape[0] == 3


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge_b_overrides_a(simple_sig):
    sig = simple_sig
    fn1 = lambda v: torch.zeros_like(v)
    fn2 = lambda v: torch.ones_like(v)
    a = Interpretation(sig)
    a[sig.get_predicate("P")] = fn1
    b = Interpretation(sig)
    b[sig.get_predicate("P")] = fn2
    merged = Interpretation.merge(a, b)
    assert merged[sig.get_predicate("P")] is fn2


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

def test_parameters_yields_module_params(simple_sig):
    sig = simple_sig
    interp = Interpretation(sig)
    net = nn.Linear(1, 1)
    interp[sig.get_predicate("P")] = net
    params = list(interp.parameters())
    assert len(params) == 2  # weight + bias


def test_parameters_empty_when_no_modules(simple_sig):
    sig = simple_sig
    interp = Interpretation(sig)
    interp[sig.get_predicate("P")] = lambda v: torch.sigmoid(v)
    params = list(interp.parameters())
    assert params == []


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_cache_produces_same_result(simple_sig, simple_interp):
    sig = simple_sig
    phi = sig.parse("forall x: P(x)")
    r1 = simple_interp(phi)
    r2 = simple_interp(phi)
    assert torch.allclose(r1.value, r2.value)


def test_cache_toggle(simple_sig, simple_interp):
    sig = simple_sig
    phi = sig.parse("P(x)")
    simple_interp.use_cache = False
    r1 = simple_interp(phi)
    simple_interp.use_cache = True
    r2 = simple_interp(phi)
    assert torch.allclose(r1.value, r2.value)


def test_function_call_uses_bool_sort_fallback_type():
    sig = Signature("demo_fn")
    image = sig.sort("Image")
    if "Bool" not in sig._sorts:
        sig.sort("Bool")
    digit = sig.dimension("digit")
    sig.variable("w", image)
    sig.function("mnist", ["Image"], "Bool", output_dims=[digit])

    interp = Interpretation(sig)
    interp["Image"] = Real
    interp["w"] = torch.rand(4, 1)
    interp["mnist"] = nn.Sequential(nn.Linear(1, 10), nn.Sigmoid())

    result = interp("mnist(w)")
    assert isinstance(result, Tensor)
    assert result.domain_type == Bool
    assert "digit" in result.axis_names


def test_function_call_without_output_type_raises_clear_error():
    sig = Signature("demo_fn_missing_type")
    image = sig.sort("Image")
    score = sig.sort("Score")
    sig.variable("w", image)
    sig.function("f", ["Image"], "Score")

    interp = Interpretation(sig)
    interp["Image"] = Real
    interp["w"] = torch.rand(3, 1)
    interp["f"] = nn.Linear(1, 1)

    with pytest.raises(ValueError, match="has no output type"):
        interp("f(w)")


def test_formula_axis_select_matches_interpretation_at_projection():
    sig = Signature("demo_axis_select")
    S = sig.sort("S")
    T = sig.dimension("t")
    sig.predicate("P", ["S"], input_dims=[T], output_dims=[T])
    sig.variable("x", S, dims=[T])

    interp = Interpretation(sig)
    interp["S"] = Type("S", shape=(1,), axis_names=["feat"])
    interp["x"] = torch.rand(5, 3, 1)
    interp["P"] = lambda v: v

    selected_inline = interp("(P(x)[t] & P(x)[t])[t=0]")
    selected_at = interp("P(x) & P(x)", at={"t": 0})

    assert torch.allclose(selected_inline.value, selected_at.value)


def test_cache_skips_redundant_evaluations():
    """Cache must evaluate a shared sub-formula exactly once per interp() call.
    Formula:  (P(x) & P(x)) or (P(x) & P(x))
    P(x) appears four times in the AST, but all four nodes have the same
    repr and the same assignment.  With caching, P(x) should be evaluated
    once; without caching, four times.
    """
    sig = Signature("cache_test")
    S   = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.variable("x", S)
    call_count = [0]
    def counting_P(v):
        call_count[0] += 1
        return torch.sigmoid(v)
    interp = Interpretation(sig)
    interp["S"]   = Real
    interp["x"]   = torch.rand(5, 1)
    interp["P"]   = counting_P
    phi = sig.parse("(P(x) & P(x)) or (P(x) & P(x))")
    # With cache
    call_count[0] = 0
    interp.use_cache = True
    interp(phi)
    calls_with_cache = call_count[0]
    # Without cache
    call_count[0] = 0
    interp.use_cache = False
    interp(phi)
    calls_without_cache = call_count[0]
    assert calls_with_cache < calls_without_cache, (
        f"Cache did not reduce evaluations: {calls_with_cache} vs {calls_without_cache}"
    )

def test_float64_data_through_a_real_sorted_function():
    # found-bugs.md #47: float64 data failed every Real-sorted term.
    sig = Signature("f64")
    sig.sort("Img"); sig.sort("Scalar")
    sig.variable("x", "Img")
    sig.function("f", ["Img"], "Scalar")
    sig.predicate("P", ["Scalar"])
    interp = Interpretation(sig)
    interp["Img"] = Type("Img", shape=(2,), axis_names=["feat"])
    interp["Scalar"] = Real
    interp["x"] = torch.rand(4, 2, dtype=torch.float64)
    interp["f"] = lambda a: a.sum(-1, keepdim=True)
    interp["P"] = lambda a: torch.sigmoid(a)
    out = interp("P(f(x))")
    assert out.value.dtype == torch.float64
    assert out.value.shape == (4, 1)


def test_logic_reused_across_interpretations_does_not_share_struct_agg():
    # found-bugs.md #56: with_defaults passed the StructAgg dict through, so it was shared.
    sig = Signature("shared_logic")
    sig.dimension("T")
    sig.structural_variable("t", "T")
    L = Logic(StructAgg={"T": {"forall": fuzzy_ops.AggregPMeanError(p=2)}})
    i1 = Interpretation(sig, logic=L)
    i2 = Interpretation(sig)
    i2.logic = L
    i1.logic["forall,t"] = fuzzy_ops.AggregMin()
    assert isinstance(i1.logic["forall,T"], fuzzy_ops.AggregMin)
    assert isinstance(i2.logic["forall,T"], fuzzy_ops.AggregPMeanError)
    assert isinstance(L.StructAgg["T"]["forall"], fuzzy_ops.AggregPMeanError)
    # an axis added on one interpretation does not appear on the other
    i1.logic["exists,T"] = fuzzy_ops.AggregMax()
    assert "exists" not in i2.logic.StructAgg["T"]


def test_logic_repr_prints_symbol_struct_agg_keys_correctly():
    # found-bugs.md #57: every kind other than "forall" was printed as ∃.
    L = Logic(StructAgg={"T": {"∀": fuzzy_ops.AggregMin(), "exists": fuzzy_ops.AggregMax()}})
    r = repr(L)
    assert "(∀, T) : AggregMin" in r
    assert "(∃, T) : AggregMax" in r


def test_equiv_default_uses_user_set_implies():
    # found-bugs.md #23: Logic(Implies=...) left <-> on Goguen.
    L = Logic(Implies=fuzzy_ops.ImpliesReichenbach()).with_defaults()
    assert isinstance(L.Equiv.implies_op, fuzzy_ops.ImpliesReichenbach)
    a, b = torch.tensor([0.9, 0.2, 0.6]), torch.tensor([0.3, 0.8, 0.6])
    expected = L.And(L.Implies(a, b), L.Implies(b, a))
    assert torch.allclose(L.Equiv(a, b), expected)
    assert torch.allclose(L.Equiv(a, b), torch.tensor([0.359, 0.346, 0.578]), atol=1e-3)
    # with nothing set, <-> still uses the default implication (Goguen)
    assert isinstance(Logic().with_defaults().Equiv.implies_op, fuzzy_ops.ImpliesGoguen)
