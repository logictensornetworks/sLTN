"""Interpretation as an nn.Module: parameters, state_dict, devices, copies and multi-objective learning."""
import pytest
import torch
import torch.nn as nn

from sltn import Signature, Interpretation, Type, kb_backward


@pytest.fixture
def sig():
    s = Signature("m")
    s.sort("S")
    s.variable("x", "S")
    for name in ("A", "B", "train"):
        s.predicate(name, ["S"])
    return s


def _interp(sig):
    i = Interpretation(sig)
    i["S"] = Type("S", shape=(2,))
    i["x"] = torch.rand(4, 2)
    return i


def _net():
    return nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())


def test_shared_module_counted_once(sig):
    i = _interp(sig)
    net = _net()
    i["A"] = net
    i["B"] = net
    assert len(list(i.parameters())) == 2
    assert [n for n, _ in i.named_parameters()] == ["A.0.weight", "A.0.bias"]


def test_state_dict_round_trip_without_data(sig):
    i = _interp(sig)
    i["A"] = _net()
    sd = i.state_dict()
    assert all("x" not in k.split(".") for k in sd)
    j = _interp(sig)
    j["A"] = _net()
    j.load_state_dict(sd)
    assert torch.equal(j["A"][0].weight, i["A"][0].weight)


def test_train_eval_reach_groundings(sig):
    i = _interp(sig)
    i["A"] = _net()
    i.eval()
    assert not i["A"].training
    i.train()
    assert i["A"].training


def test_symbol_named_like_module_attribute(sig):
    i = _interp(sig)
    i["train"] = _net()
    assert len(list(i.parameters())) == 2
    assert i("forall x: train(x)").value.numel() == 1


def test_data_tensors_follow_module_conversions(sig):
    i = _interp(sig)
    i["A"] = _net()
    i.double()
    assert i["x"].dtype == torch.float64
    assert i("forall x: A(x)").value.dtype == torch.float64
    i["x"] = torch.rand(3, 2)
    assert i["x"].shape == (3, 2)
    assert "x" not in [n for n, _ in i.named_parameters()]


def test_regrounding_removes_parameters(sig):
    i = _interp(sig)
    i["A"] = _net()
    i["A"] = lambda v: v[..., :1].sigmoid()
    assert list(i.parameters()) == []


def test_allowed_lambdas(sig):
    i = _interp(sig)
    frozen = nn.Linear(2, 1).requires_grad_(False)
    i["A"] = _net()
    i["B"] = lambda v: torch.sigmoid(frozen(v))
    i("forall x: B(x)")
    i["B"] = lambda v: 1 - i["A"](v)
    i("forall x: B(x)")


def test_parameter_grounding_registered():
    s = Signature("p")
    s.sort("S")
    s.constant("c", "S")
    i = Interpretation(s)
    i["S"] = Type("S", shape=(2,))
    w = nn.Parameter(torch.rand(2))
    i["c"] = w
    assert list(i.parameters()) == [w]
    assert [n for n, _ in i.named_parameters()] == ["c"]


def test_deepcopy_keeps_sharing(sig):
    i = _interp(sig)
    net = _net()
    i["A"] = net
    i["B"] = net
    j = i.deepcopy()
    assert j["A"] is j["B"] and j["A"] is not net


def test_multiobjective_shared_module_does_not_crash(sig):
    # A module bound to two symbols crashed jac_to_grad.
    pytest.importorskip("torchjd")
    i = _interp(sig)
    net = _net()
    i["A"] = net
    i["B"] = net
    kb = {"a": sig.parse("forall x: A(x)"), "b": sig.parse("forall x: not B(x)")}
    opt = torch.optim.SGD(i.parameters(), lr=0.1)
    kb_backward(opt, i, kb, aggregator="upgrad")
    assert all(p.grad is not None for p in net.parameters())


def test_multiobjective_uses_optimizer_parameters(sig):
    # Parameters came from interp.parameters(), not from the optimizer.
    pytest.importorskip("torchjd")
    i = _interp(sig)
    i["A"] = _net()
    i["B"] = _net()
    kb = {"a": sig.parse("forall x: A(x)"), "b": sig.parse("forall x: not B(x)")}
    opt = torch.optim.SGD(i["B"].parameters(), lr=0.1)
    kb_backward(opt, i, kb, aggregator="upgrad")
    assert all(p.grad is not None for p in i["B"].parameters())
    assert all(p.grad is None for p in i["A"].parameters())
