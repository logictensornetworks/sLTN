"""Regressions for #22, #26, #27, #44, #45, #60, #61, #70, #71 and the axis-naming rules."""
import pytest
import torch
import torch.nn as nn

from sltn import Signature, Interpretation, Type
from sltn.interpretation.core.axis import Axis
from sltn.interpretation.core.tensor import Tensor
from sltn.interpretation.operators import AggregPMean


def test_empty_batch_probe_keeps_batchnorm_state():  # #60
    sig = Signature("bn")
    sig.sort("S")
    sig.variable("x", "S")
    sig.predicate("P", ["S"])
    i = Interpretation(sig)
    i["S"] = Type("S", shape=(2,))
    i["x"] = torch.rand(0, 2)
    bn = nn.BatchNorm1d(2)
    i["P"] = nn.Sequential(bn, nn.Linear(2, 1), nn.Sigmoid())
    before = bn.running_mean.clone()
    i.train()
    out = i(sig.parse("P(x)"))
    assert out.value.shape[0] == 0
    assert bn.training
    assert torch.equal(bn.running_mean, before)
    assert bn.num_batches_tracked.item() == 0


@pytest.mark.parametrize("stable", [True, False])
def test_pmean_negative_p_with_mask(stable):  # #70
    agg = AggregPMean(p=-2, stable=stable)
    xs = torch.tensor([[0.0, 0.5, 0.8], [0.3, 0.0, 0.9]])
    mask = torch.tensor([[False, True, True], [False, False, False]])
    out = agg(xs, dim=1, mask=mask)
    assert torch.isfinite(out).all()
    assert out[1].item() == 0.0


def test_align_rejects_axis_with_two_roles():  # #44
    a = Tensor(torch.rand(3, 1), [Axis("t", "variable"), Axis("d", "domain")])
    b = Tensor(torch.rand(3, 1), [Axis("t", "structural"), Axis("d", "domain")])
    with pytest.raises(ValueError, match="single role"):
        Tensor.align(a, b)


@pytest.fixture
def gen_sig():
    sig = Signature("gen")
    sig.sort("S")
    T = sig.dimension("T")
    sig.structural_variable("t", T)
    sig.variable("x", "S", dims=[T])
    sig.predicate("Gen", ["S"], output_dims=[T])
    i = Interpretation(sig)
    i["S"] = Type("S", shape=(1,))
    i["x"] = torch.rand(4, 5, 1)
    return sig, i


def test_unnamed_output_axis_is_rejected(gen_sig):  # #61
    sig, _ = gen_sig
    with pytest.raises(ValueError, match="Gen"):
        sig.parse("Gen(x[t])")


def test_explicit_passthrough(gen_sig):  # #61
    sig, i = gen_sig
    i["Gen"] = lambda v: torch.sigmoid(v)
    out = i(sig.parse("Gen(x[t])[t]"))
    assert out.value.shape == (4, 5, 1)


def test_new_output_axis(gen_sig):  # #61
    sig, i = gen_sig
    i["Gen"] = lambda v: torch.sigmoid(v).unsqueeze(1).expand(-1, 5, 1)
    out = i(sig.parse("Gen(x[t])[t1]"))
    assert set(out.free_structural_variables) == {"t", "t1"}
    assert out.value.shape == (4, 5, 5, 1)


@pytest.mark.parametrize("src", ["Gen(x[T])[t]", "Gen(x[t])[T]", "Gen(x[t])[T1]",
                                 "forall T: Gen(x[t])[t]"])
def test_dimension_names_never_name_axes(gen_sig, src):
    sig, _ = gen_sig
    with pytest.raises(NameError, match="dimension name"):
        sig.parse(src)


def test_structural_variable_cannot_reuse_dimension_name():
    sig = Signature("n")
    sig.dimension("T")
    with pytest.raises(ValueError, match="already a dimension"):
        sig.structural_variable("T", "T")
    sig.structural_variable("t", "T")
    with pytest.raises(ValueError):
        sig.dimension("t")


def test_struct_agg_keyed_by_variable_before_attach(gen_sig):
    from sltn.interpretation import Logic
    from sltn.interpretation.operators import AggregMin
    sig, i = gen_sig
    L = Logic()
    L["forall,t"] = AggregMin()
    i.logic = L
    i["Gen"] = lambda v: torch.sigmoid(v)
    out = i(sig.parse("forall t1: Gen(x[t1])[t1]"))
    assert torch.allclose(out.value, torch.sigmoid(i["x"]).amin(dim=1))
    assert isinstance(i.logic["forall,T"], AggregMin)


def test_vacuous_structural_quantifier_rejected(gen_sig):
    sig, _ = gen_sig
    with pytest.raises(ValueError, match="names no free axis"):
        sig.parse("forall t1: Gen(x[t])[t]")


def test_removed_symbol_options():  # #22
    sig = Signature("r")
    sig.sort("S")
    with pytest.raises(TypeError):
        sig.predicate("P", ["S"], consumes_args=True)


@pytest.fixture
def seq_sig():
    sig = Signature("seq")
    sig.sort("S")
    T = sig.dimension("T")
    sig.structural_variable("t", T)
    sig.variable("x", "S", dims=[T])
    sig.variable("m", "S", dims=[T, T])
    sig.predicate("C", ["S"])
    sig.predicate("Seq", ["S"], input_dims=[T])
    return sig


def test_uncached_fsv_is_alias_aware(seq_sig):  # #71
    from sltn.fol.definitions import _child_nodes
    phi = seq_sig.parse("forall x: Seq(x[t1])")
    stack = [phi]
    while stack:
        node = stack.pop()
        node._free_structural_variables = None
        stack.extend(_child_nodes(node))
    assert phi.is_clause()


def test_repeated_dims_need_distinct_names(seq_sig):  # #26
    for src in ("C(m)", "C(m[t])", "C(m[t, t])"):
        with pytest.raises(ValueError):
            seq_sig.parse(src)
    assert seq_sig.parse("C(m[t, t1])").free_structural_variables == {"t", "t1"}


def test_pointwise_output_annotation_rejected(seq_sig):  # #27
    for src in ("C(x[t])[t1]", "C(x[t])[t]"):
        with pytest.raises(ValueError, match="annotate the argument"):
            seq_sig.parse(src)
    assert seq_sig.parse("C(x[t])").free_structural_variables == {"t"}


@pytest.mark.parametrize("name", ["img", "bool"])
def test_variable_named_like_domain_axis(name):  # #45
    sig = Signature("ax")
    sig.sort("Img")
    sig.variable(name, "Img")
    sig.predicate("C", ["Img"])
    i = Interpretation(sig)
    i["Img"] = Type("Img", shape=(2,))
    i[name] = torch.rand(4, 2)
    i["C"] = lambda v: torch.sigmoid(v.sum(-1, keepdim=True))
    with pytest.raises(ValueError, match=f"Variable '{name}'"):
        i(sig.parse(f"C({name})"))
