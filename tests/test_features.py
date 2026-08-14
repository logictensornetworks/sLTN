"""Regression + feature tests for structural sLTN features."""
import torch
import pytest
import sltn
from sltn import Signature, Interpretation


# ---------------------------------------------------------------------------
# Shared fixture — build signature and interpretation once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sig_interp():
    sig = Signature("test")
    S   = sig.sort("S")
    T   = sig.dimension("t")
    A   = sig.predicate("A", ["S"], input_dims=[T], output_dims=[T])
    B   = sig.predicate("B", ["S", "S"])
    x   = sig.variable("x", S, dims=[T])
    u   = sig.variable("u", S)
    y   = sig.variable("y", S)
    nxt = sig.structural_relation("next", [T, T])

    interp = Interpretation(sig)
    interp.logic.StructAgg = {
        "t": {
            "forall": sltn.interpretation.operators.AggregPMeanError(p=2.0, stable=True, eps=1e-12),
            "exists": sltn.interpretation.operators.AggregPMean(p=2.0, stable=True, eps=1e-12),
        }
    }

    interp[S] = sltn.Type("R1", shape=(1,))
    interp[x] = torch.rand(4, 5, 1)   # [batch, t, feature]
    interp[u] = torch.rand(4, 1)
    interp[y] = torch.rand(4, 1)
    interp[A] = lambda v: torch.sigmoid(v)
    interp[B] = lambda a, b: torch.sigmoid(a * b)

    nxt_mat = torch.zeros(5, 5)
    for i in range(4):
        nxt_mat[i, i + 1] = 1.0
    interp[nxt] = nxt_mat

    return sig, interp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forall_exists_over_structural_axis(sig_interp):
    """forall/exists over structural axis t produce correct shapes."""
    sig, interp = sig_interp
    r1 = interp(sig.parse("forall t: A(x)[t]"))
    r2 = interp(sig.parse("exists t: A(x)[t]"))
    assert r1.value.shape == torch.Size([4, 1])
    assert r2.value.shape == torch.Size([4, 1])
    assert not torch.allclose(r1.value, r2.value)


def test_variable_aliases(sig_interp):
    """u1 is a copy of u; forall u, u1: B(u, u1) collapses to scalar."""
    sig, interp = sig_interp
    phi_free = sig.parse("B(u, u1)")
    r_free = interp(phi_free)
    assert r_free.value.shape == torch.Size([4, 4, 1])

    r3 = interp(sig.parse("forall u, u1: B(u, u1)"))
    assert r3.value.shape == torch.Size([1])


def test_guarded_quantification_with_next(sig_interp):
    """next(t, t1) guard produces correct shape."""
    sig, interp = sig_interp
    phi = sig.parse("forall t, t1 | next(t, t1): (A(x[t])[t] -> A(x[t1])[t1])")
    r5 = interp(phi)
    assert r5.value.shape == torch.Size([4, 1])


def test_repr_shows_axis_annotations(sig_interp):
    """repr includes [t] / [t1] axis annotations."""
    sig, interp = sig_interp
    phi_fa = sig.parse("forall t: A(x)[t]")
    assert "[t]" in repr(phi_fa)

    phi_guard = sig.parse("forall t, t1 | next(t, t1): (A(x[t])[t] -> A(x[t1])[t1])")
    r4 = repr(phi_guard)
    assert "A(x[t])[t]" in r4 and "A(x[t1])[t1]" in r4


def test_postfix_axis_syntax(sig_interp):
    """P(args)[axes] postfix parse sets structural_axes."""
    sig, interp = sig_interp
    phi_suf = sig.parse("A(x)[t1]")
    assert phi_suf.structural_axes == ["t1"]


def test_prefix_axis_syntax_rejected(sig_interp):
    """P[axes](args) prefix syntax is not supported and raises an error."""
    sig, interp = sig_interp
    with pytest.raises((ValueError, Exception)):
        sig.parse("A[t](x)")


def test_axis_renaming_t_vs_t1(sig_interp):
    """A(x)[t] and A(x)[t1] produce differently-labelled output axes."""
    sig, interp = sig_interp
    r_t  = interp(sig.parse("A(x[t])[t]"))
    r_t1 = interp(sig.parse("A(x[t1])[t1]"))
    assert r_t.free_structural_variables  == ["t"]
    assert r_t1.free_structural_variables == ["t1"]


def test_forall_two_struct_axes_no_guard(sig_interp):
    """forall t, t1 (no guard) aggregates both axes."""
    sig, interp = sig_interp
    phi_2t = sig.parse("forall t, t1: (A(x[t])[t] -> A(x[t1])[t1])")
    r_2t = interp(phi_2t)
    assert r_2t.value.shape == torch.Size([4, 1])


def test_bare_predicate_uses_declared_output_dims(sig_interp):
    """A(x) without annotation uses the declared output_dims [t]."""
    sig, interp = sig_interp
    r_no_annot = interp(sig.parse("A(x)"))
    struct_out = r_no_annot.structural_axes
    assert len(struct_out) == 1 and struct_out[0].name == "t"


def test_forall_t_leaves_t2_free(sig_interp):
    """forall t: ... A(x[t2])[t2] leaves the t2 axis free in the result."""
    sig, interp = sig_interp
    phi_free_t2 = sig.parse("forall t: (A(x[t])[t] -> A(x[t2])[t2])")
    r_free_t2 = interp(phi_free_t2)
    assert "t2" in r_free_t2.free_structural_variables


def test_guard_can_introduce_free_structural_axis(sig_interp):
    """A guard may carry a free structural axis that remains after quantification."""
    sig, interp = sig_interp
    phi = sig.parse("forall t | next(t, t1): A(x[t])[t]")
    r = interp(phi)
    assert "t1" in r.free_structural_variables
    assert r.value.shape == torch.Size([4, 5, 1])


# ---------------------------------------------------------------------------
# Dimension-level StructAgg grounding
#
# Mirrors notebooks/tutorials_sltn/5-StructuralLTN.ipynb, where the dimension
# ("T") is named differently from the structural variable that ranges over
# it ("t"). StructAgg groundings must be resolved through the dimension, so
# any variable of that dimension (base, alias, or independently declared)
# inherits the same aggregator without a separate entry.
# ---------------------------------------------------------------------------

@pytest.fixture
def dim_var_sig_interp():
    sig = Signature("dim_var_test")
    S = sig.sort("S")
    T = sig.dimension("T")
    sig.structural_variable("t", T)
    s_var = sig.structural_variable("s", T)  # independently declared, same dimension
    A = sig.predicate("A", ["S"], input_dims=[T], output_dims=[T])
    x = sig.variable("x", S, dims=[T])

    interp = Interpretation(sig)
    interp[S] = sltn.Type("R1", shape=(1,))
    interp[x] = torch.rand(4, 5, 1)
    interp[A] = lambda v: torch.sigmoid(v)
    return sig, interp, s_var


def test_struct_agg_set_by_dimension_used_by_variable(dim_var_sig_interp):
    """Grounding set via the dimension name is used by a variable ranging over it."""
    sig, interp, _ = dim_var_sig_interp
    interp.logic.StructAgg = {
        "T": {
            "forall": sltn.interpretation.operators.AggregPMeanError(p=2.0, stable=True, eps=1e-12),
            "exists": sltn.interpretation.operators.AggregPMean(p=2.0, stable=True, eps=1e-12),
        }
    }
    r1 = interp(sig.parse("forall t: A(x)[t]"))
    r2 = interp(sig.parse("exists t: A(x)[t]"))
    assert r1.value.shape == torch.Size([4, 1])
    assert not torch.allclose(r1.value, r2.value)


def test_struct_agg_shared_across_independent_variables_of_same_dimension(dim_var_sig_interp):
    """A second, independently declared variable over the same dimension inherits the grounding."""
    sig, interp, _ = dim_var_sig_interp
    op_fa = sltn.interpretation.operators.AggregPMeanError(p=2.0, stable=True, eps=1e-12)
    op_ex = sltn.interpretation.operators.AggregPMean(p=2.0, stable=True, eps=1e-12)
    interp.logic.StructAgg = {"T": {"forall": op_fa, "exists": op_ex}}

    r_t = interp(sig.parse("forall t: A(x)[t]"))
    r_s = interp(sig.parse("forall s: A(x)[s]"))
    assert torch.allclose(r_t.value, r_s.value)


def test_struct_agg_set_via_variable_binds_dimension(dim_var_sig_interp):
    """Setting via a structural variable name binds the dimension: readable/usable by other variables too."""
    sig, interp, _ = dim_var_sig_interp
    op_fa = sltn.interpretation.operators.AggregPMeanError(p=2.0, stable=True, eps=1e-12)
    op_ex = sltn.interpretation.operators.AggregPMean(p=2.0, stable=True, eps=1e-12)
    interp.logic["∀,t"] = op_fa
    interp.logic["∃,t"] = op_ex

    # Readable back through the dimension name.
    assert interp.logic["∀,T"] is op_fa
    assert interp.logic["∃,T"] is op_ex

    # And usable by an unrelated variable ("s") ranging over the same dimension.
    r_s = interp(sig.parse("forall s: A(x)[s]"))
    assert r_s.value.shape == torch.Size([4, 1])


def test_struct_agg_tolerates_symbol_inner_keys(dim_var_sig_interp):
    """{'∀': op, 'exists': op} (mixed symbol/keyword inner keys) is accepted, not just 'forall'/'exists'."""
    sig, interp, _ = dim_var_sig_interp
    interp.logic.StructAgg = {
        "T": {
            "∀": sltn.interpretation.operators.AggregPMeanError(p=2.0, stable=True, eps=1e-12),
            "exists": sltn.interpretation.operators.AggregPMean(p=2.0, stable=True, eps=1e-12),
        }
    }
    r1 = interp(sig.parse("forall t: A(x)[t]"))
    r2 = interp(sig.parse("exists t: A(x)[t]"))
    assert r1.value.shape == torch.Size([4, 1])
    assert not torch.allclose(r1.value, r2.value)


# ---------------------------------------------------------------------------
# Repeated-dimension groundings for VariableSymbol / ConstantSymbol
#
# sig.variable("x", "S", dims=["T", "T"]) grounds two structural axes that
# both derive from the same dimension "T". Grounding deduplicates them the
# same way StructuralRelationSymbol does ("T" -> "T0", "T1"), and formulas
# can rename them positionally via postfix annotation (e.g. x[t, t1]).
# ---------------------------------------------------------------------------

def test_variable_with_repeated_dimension_grounds_with_deduplicated_axes():
    sig = Signature("repeated_dim_var")
    S = sig.sort("S")
    T = sig.dimension("T")
    x = sig.variable("x", S, dims=[T, T])

    interp = Interpretation(sig)
    interp[S] = sltn.Type("R1", shape=(1,))
    interp[x] = torch.rand(4, 5, 5, 1)

    r = interp(sig.parse("x"))
    assert set(r.axis_map) == {"x", "T0", "T1", "r1"}


def test_constant_with_repeated_dimension_grounds_with_deduplicated_axes():
    sig = Signature("repeated_dim_const")
    S = sig.sort("S")
    T = sig.dimension("T")
    c = sig.constant("c", S, dims=[T, T])

    interp = Interpretation(sig)
    interp[S] = sltn.Type("R1", shape=(1,))
    interp[c] = torch.rand(5, 5, 1)

    r = interp(sig.parse("c"))
    assert set(r.axis_map) == {"T0", "T1", "r1"}


def test_variable_with_repeated_dimension_can_be_renamed_and_quantified():
    """Deduplicated axes can be renamed via postfix annotation and quantified independently."""
    sig = Signature("repeated_dim_quant")
    S = sig.sort("S")
    T = sig.dimension("T")
    t = sig.structural_variable("t", T)
    x = sig.variable("x", S, dims=[T, T])
    P = sig.predicate("P", ["S"], input_dims=[T, T], output_dims=[T, T])

    interp = Interpretation(sig)
    interp[S] = sltn.Type("R1", shape=(1,))
    interp[x] = torch.rand(4, 5, 5, 1)
    interp[P] = lambda v: torch.sigmoid(v)
    interp.logic.StructAgg = {
        "T": {
            "forall": sltn.interpretation.operators.AggregPMeanError(p=2.0, stable=True, eps=1e-12),
            "exists": sltn.interpretation.operators.AggregPMean(p=2.0, stable=True, eps=1e-12),
        }
    }

    r_named = interp(sig.parse("x[t, t1]"))
    assert r_named.free_structural_variables == ["t", "t1"]

    r_quant = interp(sig.parse("forall t, t1: P(x[t, t1])[t, t1]"))
    assert r_quant.value.shape == torch.Size([4, 1])