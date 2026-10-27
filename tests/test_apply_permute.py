"""Regression: canonical axis order (batched vs preserved structural) is applied
consistently whether the predicate grounding is an `nn.Module` or a plain callable.

Lexicographic align order can place a *preserved* structural axis before a
*batched* one; both apply paths must permute to [vars | batched | preserved |
domain] before reshape so the grounding receives the correct layout.
"""

import pytest
import torch
import torch.nn as nn

from sltn import Signature, Interpretation, Type


@pytest.fixture
def perm_interp():
    """A signature/interpretation pair whose variable `x` has structural
    dims [z, a] (z=2, a=3), with predicate `P` preserving only dim `a`.

    After alignment, structural axes are sorted by name: a then z ->
    [batch, a, z, domain] = [1, 3, 2, 1]; the apply layer must then permute
    to [batched_z | preserved_a | domain] = [2, 3, 1] before calling the
    grounding.
    """
    sig = Signature("perm")
    S = sig.sort("S")
    Da = sig.dimension("A")
    Dz = sig.dimension("Z")
    sig.structural_variable("a", Da)
    sig.structural_variable("z", Dz)
    x = sig.variable("x", S, dims=[Dz, Da])
    sig.predicate("P", ["S"], input_dims=[Da], output_dims=[Da])

    interp = Interpretation(sig)
    interp[S] = Type("R1", shape=(1,))
    interp[x] = torch.arange(1 * 2 * 3 * 1, dtype=torch.float32).reshape(1, 2, 3, 1)
    return sig, interp


def test_nn_module_permutes_batched_before_preserved_structural(perm_interp):
    """The nn.Module apply path must permute to [vars | batched | preserved | domain]."""
    sig, interp = perm_interp

    class ShapeEcho(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_shape: tuple[int, ...] | None = None

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            self.last_shape = tuple(t.shape)
            # Predicate output must satisfy Bool constraints.
            return torch.sigmoid(t)

    echo = ShapeEcho()
    interp[sig.get_predicate("P")] = echo
    interp(sig.parse("P(x[z, a])[a]"))

    # Flattened batch = batch_z = 1 * 2 = 2; preserved axis a has size 3; domain 1.
    assert echo.last_shape == (2, 3, 1), echo.last_shape


def test_plain_callable_permutes_batched_before_preserved_structural(perm_interp):
    """The plain-callable apply path (e.g. a lambda) must match the nn.Module path.

    This guards the path used by notebook callbacks where users bind lambdas
    that internally call a model; structural dims must be flattened consistently.
    """
    sig, interp = perm_interp

    last_shape: dict[str, tuple[int, ...] | None] = {"shape": None}

    def recorder(t: torch.Tensor) -> torch.Tensor:
        last_shape["shape"] = tuple(t.shape)
        return torch.sigmoid(t)

    interp[sig.get_predicate("P")] = recorder
    interp(sig.parse("P(x[z, a])[a]"))

    # Flattened batch = batch_z = 1 * 2 = 2; preserved axis a has size 3; domain 1.
    assert last_shape["shape"] == (2, 3, 1), last_shape["shape"]


def test_consumed_axes_follow_input_dims_order():
    """Consumed axes reach the grounding in input_dims order, not alphabetical order."""
    sig = Signature("order")
    S = sig.sort("S")
    Da = sig.dimension("A")
    Dz = sig.dimension("Z")
    sig.structural_variable("a", Da)
    sig.structural_variable("z", Dz)
    x = sig.variable("x", S, dims=[Dz, Da])
    sig.predicate("P", ["S"], input_dims=[Dz, Da], output_dims=[Dz, Da])

    interp = Interpretation(sig)
    interp[S] = Type("R1", shape=(1,))
    val = torch.rand(1, 2, 3, 1)
    interp[x] = val

    seen = {}

    def recorder(t: torch.Tensor) -> torch.Tensor:
        seen["shape"] = tuple(t.shape)
        return t

    interp[sig.get_predicate("P")] = recorder
    out = interp(sig.parse("P(x[z, a])[z, a]"))

    assert seen["shape"] == (1, 2, 3, 1)
    assert torch.allclose(out.value[0, ..., 0], val[0, ..., 0])


def _alias_interp():
    """`t` declared over T; `t1`, `t2` are undeclared aliases of it."""
    sig = Signature("alias_consume")
    S = sig.sort("S")
    T = sig.dimension("T")
    sig.structural_variable("t", T)
    x = sig.variable("x", S, dims=[T])
    sig.predicate("Seq", ["S"], input_dims=[T])
    sig.predicate("Step", ["S"], input_dims=[T], output_dims=[T])

    interp = Interpretation(sig)
    interp[S] = Type("R1", shape=(1,))
    interp[x] = torch.rand(2, 3, 1)
    seen = {}

    def seq(v: torch.Tensor) -> torch.Tensor:
        seen["Seq"] = tuple(v.shape)
        return torch.sigmoid(v.mean(dim=1))

    def step(v: torch.Tensor) -> torch.Tensor:
        seen["Step"] = tuple(v.shape)
        return torch.sigmoid(v.cumsum(dim=1))

    interp[sig.get_predicate("Seq")] = seq
    interp[sig.get_predicate("Step")] = step
    return sig, interp, seen


def test_alias_used_only_in_annotations_is_consumed():
    # `t1` appears only inside an annotation, so it is never registered;
    # Seq must still consume it as an axis over T.
    sig, interp, seen = _alias_interp()
    out = interp(sig.parse("Seq(x[t1])"))
    assert seen["Seq"] == (2, 3, 1)
    assert [a.name for a in out.axes] == ["x", "bool"]
    assert torch.allclose(out.value, interp(sig.parse("Seq(x[t])")).value)


def test_alias_used_only_in_annotations_is_renamed_by_output():
    # Step consumes t1 and produces t2; t1 must not survive as an extra axis.
    sig, interp, seen = _alias_interp()
    out = interp(sig.parse("Step(x[t1])[t2]"))
    assert seen["Step"] == (2, 3, 1)
    assert [a.name for a in out.axes] == ["x", "t2", "bool"]
    assert tuple(out.value.shape) == (2, 3, 1)
    selected = interp(sig.parse("(Step(x[t1])[t2])[t2=1]"))
    assert torch.allclose(selected.value, out.value[:, 1])
