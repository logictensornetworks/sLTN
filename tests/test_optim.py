"""Tests for kb_backward, kb_eval, and kb_describe."""
import pytest
import torch
import torch.nn as nn
from sltn import Signature, Interpretation, Type, Logic
from sltn.tools.optim import kb_backward, kb_eval, kb_describe, CrispnessScheduler
import sltn.interpretation.operators as fuzzy_ops


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_kb():
    sig = Signature("simple")
    S = sig.sort("S")
    P = sig.predicate("P", ["S"])
    Q = sig.predicate("Q", ["S"])
    x = sig.variable("x", S)

    interp = Interpretation(sig)
    interp[S]   = Type("S", shape = (1,))
    interp[x] = torch.rand(6, 1)
    interp[P] = lambda v: torch.sigmoid(v)
    interp[Q] = lambda v: torch.sigmoid(-v)

    return sig, interp


@pytest.fixture
def single_pred_kb():
    """A one-clause KB ``forall x: P(x)`` with a small trainable predicate net.

    Returns a factory so tests needing independent `Interpretation` instances
    (e.g. comparing two aggregator modes) can build several sharing the same
    signature, data, and network.
    """
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.variable("x", S)
    x_data = torch.rand(4, 1)
    net = nn.Linear(1, 1)

    def make_interp():
        interp = Interpretation(sig)
        interp[S] = Type("S", shape=(1,))
        interp[sig.get_variable("x")] = x_data
        interp[sig.get_predicate("P")] = lambda v: torch.sigmoid(net(v))
        return interp

    phi = sig.parse("forall x: P(x)")
    kb_dict = kb_describe([phi])
    return sig, net, make_interp, kb_dict


@pytest.fixture
def single_pred_kb_module():
    """Like `single_pred_kb`, but the predicate is a registered `nn.Module`
    (rather than a lambda closing over a bare net), so `interp.parameters()`
    picks it up directly.
    """
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.variable("x", S)
    interp = Interpretation(sig)
    interp[S] = Type("S", shape=(1,))
    interp[sig.get_variable("x")] = torch.rand(4, 1)
    interp[sig.get_predicate("P")] = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

    phi = sig.parse("forall x: P(x)")
    kb_dict = kb_describe([phi])
    return interp, kb_dict


# ---------------------------------------------------------------------------
# kb_describe
# ---------------------------------------------------------------------------

def test_kb_describe_returns_ordered_dict(simple_kb):
    sig, interp = simple_kb
    phi_kb = sig.parse("{forall x: P(x), forall x: Q(x)}")
    kb_dict = kb_describe(phi_kb)
    assert list(kb_dict.keys()) == ["φ_0", "φ_1"]


def test_kb_describe_single_clause(simple_kb):
    sig, interp = simple_kb
    phi = sig.parse("forall x: P(x)")
    kb_dict = kb_describe([phi])
    assert "φ_0" in kb_dict


def test_kb_describe_shows_canonical_clause_form(capsys):
    sig = Signature("surface")
    s = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.variable("x", s)

    phi_kb = sig.parse("{forall x: P(x)}")
    kb_describe(phi_kb)
    out = capsys.readouterr().out

    assert "P(x)" in out


# ---------------------------------------------------------------------------
# kb_eval
# ---------------------------------------------------------------------------

def test_kb_eval_returns_floats(simple_kb):
    sig, interp = simple_kb
    phi_kb = sig.parse("{forall x: P(x), forall x: Q(x)}")
    kb_dict = kb_describe(phi_kb)
    stats = kb_eval(interp, kb_dict)
    for v in stats.values():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


def test_kb_eval_under_no_grad(simple_kb):
    sig, interp = simple_kb
    phi = sig.parse("forall x: P(x)")
    kb_dict = kb_describe([phi])
    with torch.no_grad():
        stats = kb_eval(interp, kb_dict)
    assert "φ_0" in stats


# ---------------------------------------------------------------------------
# kb_backward — standard (satagg) mode
# ---------------------------------------------------------------------------

def test_kb_backward_satagg_returns_expected_keys(single_pred_kb_module):
    interp, kb_dict = single_pred_kb_module
    optimizer = torch.optim.SGD(list(interp.parameters()), lr=0.01)

    optimizer.zero_grad()
    stats = kb_backward(optimizer, interp, kb_dict)

    assert "losses" in stats
    assert "sats" in stats
    assert "agg_loss" in stats
    assert "agg_sat" in stats
    assert "weights" in stats
    assert "aggregator" in stats


def test_kb_backward_satagg_sets_gradients(single_pred_kb_module):
    interp, kb_dict = single_pred_kb_module
    optimizer = torch.optim.SGD(list(interp.parameters()), lr=0.01)
    optimizer.zero_grad()
    kb_backward(optimizer, interp, kb_dict)

    # At least one gradient should be set after backward
    has_grad = any(p.grad is not None for p in interp.parameters())
    assert has_grad


def test_kb_backward_agg_loss_in_range(single_pred_kb):
    _, net, make_interp, kb_dict = single_pred_kb
    interp = make_interp()
    optimizer = torch.optim.SGD(list(net.parameters()), lr=0.01)
    optimizer.zero_grad()
    stats = kb_backward(optimizer, interp, kb_dict)

    assert 0.0 <= stats["agg_loss"] <= 1.0
    assert abs(stats["agg_loss"] + stats["agg_sat"] - 1.0) < 1e-5


def test_kb_backward_none_default_same_as_satagg(single_pred_kb):
    """aggregator=None and aggregator='satagg' must produce the same loss."""
    _, net, make_interp, kb_dict = single_pred_kb
    opt_none   = torch.optim.SGD(list(net.parameters()), lr=0.01)
    opt_satagg = torch.optim.SGD(list(net.parameters()), lr=0.01)

    i1 = make_interp()
    i2 = make_interp()
    opt_none.zero_grad()
    stats_none   = kb_backward(opt_none,   i1, kb_dict, aggregator=None)
    opt_satagg.zero_grad()
    stats_satagg = kb_backward(opt_satagg, i2, kb_dict, aggregator="satagg")

    assert abs(stats_none["agg_loss"] - stats_satagg["agg_loss"]) < 1e-5


def test_kb_backward_invalid_aggregator_raises(single_pred_kb):
    _, net, make_interp, kb_dict = single_pred_kb
    interp = make_interp()
    optimizer = torch.optim.SGD(list(net.parameters()), lr=0.01)
    optimizer.zero_grad()
    pytest.importorskip("torchjd", reason="invalid string aggregator is validated only when torchjd is installed")
    with pytest.raises(ValueError, match="Unknown aggregator"):
        kb_backward(optimizer, interp, kb_dict, aggregator="not_an_aggregator")


# ---------------------------------------------------------------------------
# CrispnessScheduler (experimental, not yet implemented)
# ---------------------------------------------------------------------------

def test_crispness_scheduler_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        CrispnessScheduler()


@pytest.mark.parametrize("make_logic", [
    lambda: Logic(SatAgg=fuzzy_ops.AggregMin()),
    lambda: Logic(SatAgg=fuzzy_ops.AggregMax()),
    lambda: Logic.classical(struct_agg_axes=["T"]),
], ids=["AggregMin", "AggregMax", "classical"])
def test_kb_backward_with_min_max_satagg_on_several_clauses(make_logic):
    # found-bugs.md #2: AggregMin/AggregMax reduced nothing with dim=None, so the
    # SatAgg loss of a multi-clause KB was a vector and .backward() failed.
    sig = Signature("minmax_satagg")
    I = sig.sort("Img")
    T = sig.dimension("T")
    sig.variable("x", I, dims=[T])
    sig.structural_variable("t", T)
    sig.predicate("C", ["Img"])
    kb = {"a": sig.parse("forall x: forall t: C(x[t])[t]"),
          "b": sig.parse("exists x: forall t: C(x[t])[t]")}

    mlp = nn.Linear(2, 1)
    interp = Interpretation(sig)
    interp["Img"] = Type("Img", shape=(2,), axis_names=["f"])
    interp["x"] = torch.rand(3, 5, 2)
    interp["C"] = nn.Sequential(mlp, nn.Sigmoid())
    interp.logic = make_logic()
    if interp.logic.StructAgg is None:
        interp.logic["forall,t"] = fuzzy_ops.AggregPMeanError(p=2)

    stats = kb_backward(torch.optim.SGD(mlp.parameters(), lr=0.1), interp, kb)
    assert 0.0 <= stats["agg_loss"] <= 1.0
    assert mlp.weight.grad is not None
