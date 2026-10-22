"""Tests for sltn.tools.diagnostics.cache_benchmark."""
import pytest
import torch

from sltn import Signature, Interpretation, Type
from sltn.tools.diagnostics import cache_benchmark


@pytest.fixture
def kb_interp():
    sig = Signature("diag")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.predicate("Q", ["S"])
    x = sig.variable("x", S)

    interp = Interpretation(sig)
    interp[S] = Type("S", shape=(1,))
    interp[x] = torch.rand(4, 1)
    interp[sig.get_predicate("P")] = lambda v: torch.sigmoid(v)
    interp[sig.get_predicate("Q")] = lambda v: torch.sigmoid(-v)

    return sig, interp


def test_cache_benchmark_returns_expected_keys(kb_interp):
    sig, interp = kb_interp
    phi = sig.parse("forall x: (P(x) & Q(x))")

    result = cache_benchmark(interp, phi, n_runs=2)

    assert set(result.keys()) == {
        "without_cache_ms_per_call",
        "with_cache_ms_per_call",
        "speedup",
    }
    assert result["without_cache_ms_per_call"] >= 0.0
    assert result["with_cache_ms_per_call"] >= 0.0
    assert result["speedup"] > 0.0


def test_cache_benchmark_restores_use_cache_flag(kb_interp):
    sig, interp = kb_interp
    phi = sig.parse("forall x: P(x)")

    interp.use_cache = False
    cache_benchmark(interp, phi, n_runs=2)
    assert interp.use_cache is False

    interp.use_cache = True
    cache_benchmark(interp, phi, n_runs=2)
    assert interp.use_cache is True


def test_cache_benchmark_restores_use_cache_flag_on_error(kb_interp):
    """If evaluation raises, use_cache must still be restored (finally block)."""
    sig, interp = kb_interp
    interp.use_cache = True

    with pytest.raises(Exception):
        cache_benchmark(interp, "not a valid formula object", n_runs=1)

    assert interp.use_cache is True


def test_cache_benchmark_accepts_string_formula(kb_interp):
    sig, interp = kb_interp

    result = cache_benchmark(interp, "forall x: P(x)", n_runs=2)
    assert result["speedup"] > 0.0


def test_cache_benchmark_with_backward_compute():
    sig = Signature("diag_grad")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    x = sig.variable("x", S)

    interp = Interpretation(sig)
    interp[S] = Type("S", shape=(1,))
    interp[x] = torch.rand(4, 1, requires_grad=True)
    interp[sig.get_predicate("P")] = lambda v: torch.sigmoid(v)

    phi = sig.parse("forall x: P(x)")
    result = cache_benchmark(interp, phi, n_runs=2, backward_compute=True)
    assert result["speedup"] > 0.0


def test_cache_benchmark_backward_leaves_parameter_grads_untouched():
    # found-bugs.md #67: every run's gradients accumulated into the model.
    sig = Signature("diag_net")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.predicate("Q", ["S"])
    x = sig.variable("x", S)
    net = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Linear(4, 1), torch.nn.Sigmoid())

    interp = Interpretation(sig)
    interp[S] = Type("S", shape=(1,))
    interp[x] = torch.rand(4, 1)
    interp[sig.get_predicate("P")] = net
    interp[sig.get_predicate("Q")] = lambda v: 1 - interp["P"](v)   # reuses a grounded module

    cache_benchmark(interp, "forall x: (P(x) & Q(x))", n_runs=3, backward_compute=True)
    for p in net.parameters():
        assert p.grad is None


def test_cache_benchmark_warms_up_and_alternates_modes(kb_interp, monkeypatch):
    # found-bugs.md #67: no warm-up, and the no-cache mode always ran first.
    sig, interp = kb_interp
    modes = []
    original_call = type(interp).__call__

    def recording_call(self, *args, **kwargs):
        modes.append(self.use_cache)
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(type(interp), "__call__", recording_call)
    cache_benchmark(interp, "forall x: P(x)", n_runs=4, n_warmup=2)

    assert len(modes) == 2 * (2 + 4)
    timed = modes[4:]
    assert timed == [False, True, True, False, False, True, True, False]


def test_cache_benchmark_backward_without_grad_raises(kb_interp):
    sig, interp = kb_interp   # lambdas and data with no parameters
    with pytest.raises(ValueError, match="requires grad"):
        cache_benchmark(interp, "forall x: P(x)", n_runs=1, backward_compute=True)
