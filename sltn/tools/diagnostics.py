"""Utility functions for the sLTN framework."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sltn.interpretation import Interpretation
    from sltn.fol.term import Term
    from sltn.fol.formula import Formula


# ------------------------------------------------------------------
# Cache benchmarking
# ------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    """Wait for pending GPU work on *device*, so that the clock measures it and not just kernel launches."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _grad_leaves(output: torch.Tensor) -> list:
    """Return the leaf tensors that require grad and were used to compute *output*."""
    seen, leaves = set(), {}
    stack = [output.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        if hasattr(fn, "variable"):  # AccumulateGrad node: a leaf
            leaves[id(fn.variable)] = fn.variable
        stack.extend(f for f, _ in fn.next_functions)
    return list(leaves.values())


def cache_benchmark(
    interp: Interpretation,
    formula_or_term: str | Formula | Term,
    n_runs: int = 50,
    backward_compute: bool = False,
    n_warmup: int = 3,
) -> dict:
    """Time formula evaluation with and without the sub-formula cache.

    Runs *formula_or_term* ``n_runs`` times under each mode and returns a
    summary dict so you can verify whether caching pays off.  Each mode is
    first run ``n_warmup`` times untimed, and the two modes are then
    alternated run by run, so that one-time warm-up costs and drift are not
    charged to one side.  A string formula is parsed once, before timing.

    Parameters
    ----------
    interp : Interpretation
        The evaluation context to benchmark.
    formula_or_term : str, Formula, or Term
        The formula (or term) to evaluate — the same object you would pass
        to ``interp(formula_or_term)``.
    n_runs : int
        Number of timed repetitions per mode.  Default: 50.
    backward_compute : bool
        If ``True``, also computes the gradients of the result (summed, if
        it is not a scalar) after each forward pass.  They are computed with
        ``torch.autograd.grad`` and discarded, so the ``.grad`` of the
        model's parameters is left untouched.  Default: ``False``.
    n_warmup : int
        Number of untimed runs per mode before timing.  Default: 3.

    Returns
    -------
    dict
        ``"without_cache_ms_per_call"`` — average wall time (ms) without cache.
        ``"with_cache_ms_per_call"``    — average wall time (ms) with cache.
        ``"speedup"``                   — ratio; >1 means cache is faster.
    """
    old_use_cache = interp.use_cache
    leaves: list | None = None

    def _run(use: bool) -> float:
        nonlocal leaves
        interp.use_cache = use
        # The previous run ended with a sync, so no GPU work is pending here.
        t0 = time.perf_counter()
        sat = interp(formula_or_term)
        if backward_compute:
            out = sat.value.sum()
            if leaves is None:
                leaves = _grad_leaves(out)
                if not leaves:
                    raise ValueError("backward_compute=True, but the result does not depend on any tensor that requires grad.")
            torch.autograd.grad(out, leaves, allow_unused=True)
        _sync(sat.value.device)
        return time.perf_counter() - t0

    try:
        if isinstance(formula_or_term, str):
            formula_or_term = interp.signature.parse(formula_or_term)
        for _ in range(n_warmup):
            _run(False)
            _run(True)
        t_no_cache = t_with_cache = 0.0
        for i in range(n_runs):
            # Alternate which mode goes first.
            if i % 2 == 0:
                t_no_cache   += _run(False)
                t_with_cache += _run(True)
            else:
                t_with_cache += _run(True)
                t_no_cache   += _run(False)
    finally:
        interp.use_cache = old_use_cache

    t_no_cache   = t_no_cache   / n_runs * 1000
    t_with_cache = t_with_cache / n_runs * 1000
    speedup = round(t_no_cache / t_with_cache, 2) if t_with_cache > 0 else float("inf")
    return {
        "without_cache_ms_per_call": round(t_no_cache,   4),
        "with_cache_ms_per_call":    round(t_with_cache, 4),
        "speedup":                   speedup,
    }
