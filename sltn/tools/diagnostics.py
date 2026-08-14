"""Utility functions for the sLTN framework."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sltn.interpretation import Interpretation
    from sltn.fol.term import Term
    from sltn.fol.formula import Formula


# ------------------------------------------------------------------
# Cache benchmarking
# ------------------------------------------------------------------

def cache_benchmark(
    interp: Interpretation,
    formula_or_term: str | Formula | Term,
    n_runs: int = 50,
    backward_compute: bool = False,
) -> dict:
    """Time formula evaluation with and without the sub-formula cache.

    Runs *formula_or_term* ``n_runs`` times under each mode and returns a
    summary dict so you can verify whether caching pays off.

    Parameters
    ----------
    interp : Interpretation
        The evaluation context to benchmark.
    formula_or_term : str, Formula, or Term
        The formula (or term) to evaluate — the same object you would pass
        to ``interp(formula_or_term)``.
    n_runs : int
        Number of repetitions per mode.  Default: 50.
    backward_compute : bool
        If ``True``, also calls ``.backward()`` on the result after each
        forward pass.  Default: ``False``.

    Returns
    -------
    dict
        ``"without_cache_ms_per_call"`` — average wall time (ms) without cache.
        ``"with_cache_ms_per_call"``    — average wall time (ms) with cache.
        ``"speedup"``                   — ratio; >1 means cache is faster.
    """
    old_use_cache = interp.use_cache

    def _run(use: bool) -> float:
        interp.use_cache = use
        t0 = time.perf_counter()
        for _ in range(n_runs):
            sat = interp(formula_or_term)
            if backward_compute:
                sat.value.backward()
                del sat.value.grad
            interp.clear_cache()
        return (time.perf_counter() - t0) / n_runs * 1000

    try:
        t_no_cache   = _run(False)
        t_with_cache = _run(True)
    finally:
        interp.use_cache = old_use_cache

    speedup = round(t_no_cache / t_with_cache, 2) if t_with_cache > 0 else float("inf")
    return {
        "without_cache_ms_per_call": round(t_no_cache,   4),
        "with_cache_ms_per_call":    round(t_with_cache, 4),
        "speedup":                   speedup,
    }