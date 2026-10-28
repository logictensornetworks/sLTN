"""
sltn.tools.optim — Optimisation utilities for knowledge bases.

The core function is :func:`kb_backward`, which evaluates all KB clauses and
backpropagates the gradient.  Two modes are supported:

**Standard mode** (default, ``aggregator=None``):
    Applies the interpretation's ``SatAgg`` to aggregate all clauses into a
    single scalar, then calls ``.backward()`` on ``1 - sat``.  No extra
    dependencies required.

**Multi-objective mode** (``aggregator=<string or instance>``):
    Backed by `torchjd <https://github.com/TorchJD/torchjd>`_.  Evaluates
    all KB clauses in a single forward pass, computes the full per-clause
    Jacobian, then applies an aggregator to produce a combined gradient.
    The ``aggregator`` parameter accepts any
    ``torchjd.aggregation.Aggregator`` instance, or one of the following
    convenience strings:

    ``"mgda"``
        *Multiple Gradient Descent Algorithm.*  Solves
        :math:`\\min_{\\alpha\\in\\Delta_n}\\|J^\\top\\alpha\\|^2` (Frank-Wolfe).
        Guaranteed simultaneous descent for all clauses; returns MGDA weights.

    ``"pcgrad"``
        *Projected Conflicting Gradients* (Yu et al. 2020).  Fast heuristic
        that projects out conflicting gradient components between pairs of
        clauses.

    ``"upgrad"``
        *Uncertainty-weighted Projected Gradient.*  More conservative than
        MGDA when objectives strongly conflict.

    ``"config"``
        *Conflict-Free Inverse Gradient* (ConFIG).  Scales combined gradient
        to match the average gradient magnitude.

    ``"mean"``
        Simple mean of per-clause gradients.  Equivalent to optimising the
        average satisfaction loss.

**Crispness scheduling:** :class:`CrispnessScheduler` is reserved for a future
release; it is not implemented yet and currently raises ``NotImplementedError``.
"""
from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch

from sltn.fol.formula import KB

if TYPE_CHECKING:
    from sltn.interpretation import Interpretation
    from sltn.fol.formula import Formula

# Emitted at most once per process if a torchjd aggregator is requested but
# torchjd is not installed and the caller is silently downgraded to satagg.
_TORCHJD_MISSING_WARNED: bool = False

# ---------------------------------------------------------------------------
# String → torchjd aggregator mapping
# ---------------------------------------------------------------------------

_AGGREGATOR_ALIASES: dict[str, str] = {
    "mgda":    "MGDA",
    "pcgrad":  "PCGrad",
    "upgrad":  "UPGrad",
    "config":  "ConFIG",
    "mean":    "Mean",
}


def _resolve_aggregator(aggregator):
    """Accept a torchjd Aggregator instance or a convenience string."""
    try:
        from torchjd.aggregation import Aggregator
    except ImportError as e:
        raise ImportError(
            "torchjd is required for kb_backward. "
            "Install it with: pip install torchjd"
        ) from e

    if isinstance(aggregator, Aggregator):
        return aggregator

    if isinstance(aggregator, str):
        key = aggregator.lower()
        if key not in _AGGREGATOR_ALIASES:
            raise ValueError(
                f"Unknown aggregator {aggregator!r}. "
                f"Choose from {list(_AGGREGATOR_ALIASES)} "
                "or pass a torchjd.aggregation.Aggregator instance."
            )
        import torchjd.aggregation as _agg
        return getattr(_agg, _AGGREGATOR_ALIASES[key])()

    raise TypeError(
        f"aggregator must be a string or torchjd.aggregation.Aggregator, "
        f"got {type(aggregator)}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _unpack_clauses(formulas) -> list:
    if isinstance(formulas, KB):
        return list(formulas.clauses)
    return list(formulas)


def _graph_leaves(output: torch.Tensor) -> list[torch.Tensor]:
    """The trainable leaf tensors *output* depends on, found by walking its autograd graph."""
    found, seen, stack = {}, set(), [output.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        if hasattr(fn, "variable"):              # AccumulateGrad: a leaf that requires grad
            found[id(fn.variable)] = fn.variable
        stack.extend(f for f, _ in fn.next_functions)
    return list(found.values())


def _check_untrained_parameters(optimizer: torch.optim.Optimizer, interp: Interpretation,
                                output: torch.Tensor) -> None:
    """Raise if *output* depends on trainable parameters that nothing will update.

    A network used inside a lambda is not registered in the interpretation, so
    an optimizer built from ``interp.parameters()`` would silently never train it.
    Registered parameters left out of the optimizer are a visible choice (training
    part of the KB) and are allowed.
    """
    known = {id(p) for group in optimizer.param_groups for p in group["params"]}
    known |= {id(p) for p in interp.parameters()}
    missing = [p for p in _graph_leaves(output) if id(p) not in known]
    if missing:
        shapes = [list(p.shape) for p in missing]
        raise ValueError(
            f"kb_backward: the KB depends on {len(missing)} trainable parameter(s) that are "
            f"neither in the optimizer nor in the interpretation, of shapes {shapes}. "
            "A network used inside a lambda "
            "is not registered in the interpretation: wrap it in an nn.Module. Otherwise, add "
            "these parameters to the optimizer, or freeze them with requires_grad_(False)."
        )


def kb_describe(formulas) -> OrderedDict[str, Formula]:
    """Create and display a labelled KB mapping short names to formulas.

    Prints a human-readable description (for logging), then returns the
    ``φ_0``, ``φ_1``, … mapping as an OrderedDict. Pass this dict to
    :func:`kb_backward` and :func:`kb_eval`.
    """
    clauses = _unpack_clauses(formulas)
    n = len(clauses)
    lines = [f"KB ({n} clause{'s' if n != 1 else ''}):"]

    kb_dict = OrderedDict()
    for i, c in enumerate(clauses):
        lines.append(f"  φ_{i}: {str(c)}")
        kb_dict[f"φ_{i}"] = c
    print("\n".join(lines))
    return kb_dict


def kb_eval(
    interp: Interpretation,
    formulas: dict[str, Formula] | None = None,
) -> OrderedDict[str, float]:
    """Evaluate KB clauses under ``torch.no_grad()`` and return sat values.

    Parameters
    ----------
    interp : Interpretation
        Evaluation context (with variables already bound for this step).
    formulas : dict[str, Formula]
        The KB dict from :func:`kb_describe`.

    Returns
    -------
    OrderedDict[str, float]
        Satisfaction values in [0, 1] — higher is better.
    """
    if formulas is None:
        raise TypeError(
            "kb_eval() requires 'formulas' to be passed explicitly. "
            "Pass the dict returned by kb_describe()."
        )
    with torch.no_grad():
        return OrderedDict(
            (name, interp(phi).value.item())
            for name, phi in formulas.items()
        )



def kb_backward(
    optimizer: torch.optim.Optimizer,
    interp: Interpretation,
    formulas: dict[str, Formula],
    *,
    aggregator: str | object | None = None,
    parallel_chunk_size: int | None = None,
) -> dict:
    """Multi-objective backward pass for a knowledge base.

    Evaluates all KB clauses in a **single forward pass**, then backpropagates
    a combined gradient into each parameter's ``.grad`` field.  Call
    ``optimizer.step()`` immediately after.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer.  Its parameter groups determine which tensors receive
        the combined gradient.
    interp : Interpretation
        Evaluation context.
    formulas : dict[str, Formula]
        The KB dict from :func:`kb_describe` (keys are ``φ_0``, ``φ_1``, …).
    aggregator : None, str, or torchjd.aggregation.Aggregator
        How to combine the per-clause gradients.

        ``None`` (default): **standard single-loss backward** — applies the
        ``SatAgg`` operator from the interpretation's logic to aggregate all
        clause satisfactions, then calls ``.backward()`` on ``1 - sat``.
        Does **not** require torchjd.

        ``"satagg"`` is accepted as an explicit alias for ``None``.

        Strings ``"mgda"``, ``"pcgrad"``, ``"upgrad"``, ``"config"``, ``"mean"``
        select a torchjd multi-objective aggregator (requires torchjd).
        Any :class:`torchjd.aggregation.Aggregator` instance can be passed
        directly.
    parallel_chunk_size : int or None
        Relevant only for torchjd paths.  Passed to
        :func:`torchjd.autojac.backward`.  ``None`` (default) differentiates
        all clause losses in parallel (fastest, most memory); ``1``
        differentiates them sequentially (least memory).

    Returns
    -------
    dict
        ``"losses"``     — ``OrderedDict[str, float]``, per-clause loss values
                           (1 - sat_i) keyed by formula names (``φ_0``, ``φ_1``, …).
        ``"sats"``       — ``OrderedDict[str, float]``, per-clause satisfaction
                           values (sat_i) keyed by formula names.
        ``"agg_loss"``   — ``float``, aggregated scalar loss used for the
                           backward pass (``1 - SatAgg(sat_vals)`` in standard
                           mode, mean of clause losses in torchjd modes).
        ``"agg_sat"``    — ``float``, ``1 - agg_loss``.  Convenient for
                           monitoring training progress across steps.
        ``"weights"``    — ``list[float] | None``, combination weights returned
                           by the torchjd aggregator (e.g. MGDA weights), or
                           ``None`` in standard / satagg mode.
        ``"aggregator"`` — the aggregator string or instance that was used, or
                           ``None`` for the standard mode.

    Raises
    ------
    ValueError
        If the KB depends on trainable parameters that are neither in the
        optimizer nor in the interpretation (e.g. a network used inside a
        lambda), since nothing would train them.  Raised before any backward.
    """
    clause_list = list(formulas.values())
    num_clauses = len(clause_list)
    sat_vals = torch.stack([interp(phi).value for phi in clause_list])
    _check_untrained_parameters(optimizer, interp, sat_vals)

    clause_sats = OrderedDict(
        (name, sat_val.item())
        for name, sat_val in zip(formulas.keys(), sat_vals)
    )
    clause_losses = OrderedDict(
        (name, 1.0 - sat_val.item())
        for name, sat_val in zip(formulas.keys(), sat_vals)
    )

    # ------------------------------------------------------------------
    # Standard path: single SatAgg loss (no torchjd required)
    # ------------------------------------------------------------------
    if aggregator is None or aggregator == "satagg":
        sat = interp.logic.SatAgg(sat_vals)
        loss = 1.0 - sat
        loss.backward()
        agg_loss_val = loss.item()
        return {
            "losses":     clause_losses,
            "sats":       clause_sats,
            "agg_loss":   agg_loss_val,
            "agg_sat":    1.0 - agg_loss_val,
            "weights":    None,
            "aggregator": None,
        }

    # ------------------------------------------------------------------
    # Multi-objective path: torchjd Jacobian aggregation
    # ------------------------------------------------------------------
    try:
        from torchjd.autojac import backward, jac_to_grad
    except ImportError:
        global _TORCHJD_MISSING_WARNED
        if not _TORCHJD_MISSING_WARNED:
            warnings.warn(
                f"torchjd is not installed — aggregator={aggregator!r} is not available. "
                "Falling back to the standard SatAgg backward pass. "
                "To use multi-objective aggregators install torchjd: "
                "pip install torchjd  (or: pip install sltn[multiobjective])",
                UserWarning,
                stacklevel=2,
            )
            _TORCHJD_MISSING_WARNED = True
        sat = interp.logic.SatAgg(sat_vals)
        loss = 1.0 - sat
        loss.backward()
        agg_loss_val = loss.item()
        return {
            "losses":     clause_losses,
            "sats":       clause_sats,
            "agg_loss":   agg_loss_val,
            "agg_sat":    1.0 - agg_loss_val,
            "weights":    None,
            "aggregator": None,
        }

    agg = _resolve_aggregator(aggregator)
    params = list({
        id(p): p
        for group in optimizer.param_groups for p in group["params"] if p.requires_grad
    }.values())
    if not params:
        raise ValueError("kb_backward: the optimizer has no parameters that require grad.")
    losses_t = 1.0 - sat_vals
    backward(losses_t, inputs=params, parallel_chunk_size=parallel_chunk_size)

    raw_weights = jac_to_grad(params, agg)
    weights: list[float] | None = None
    if raw_weights is not None:
        weights = raw_weights.tolist()

    agg_loss_val = sum(clause_losses.values()) / num_clauses
    return {
        "losses":     clause_losses,
        "sats":       clause_sats,
        "agg_loss":   agg_loss_val,
        "agg_sat":    1.0 - agg_loss_val,
        "weights":    weights,
        "aggregator": aggregator,
    }


# ---------------------------------------------------------------------------
# Crispness scheduler (p-parameter for pMean aggregators)
# ---------------------------------------------------------------------------

class CrispnessScheduler:
    """Experimental, not yet implemented.

    Reserved for a future utility that anneals the ``p`` parameter of
    ``AggregPMean`` / ``AggregPMeanError`` toward crisper logic over the
    course of training. Instantiating this class currently always raises
    ``NotImplementedError``.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CrispnessScheduler is not implemented yet")