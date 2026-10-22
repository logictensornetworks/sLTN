"""Focused tests for sltn.fol.definitions: definition expansion edge cases not
already exercised indirectly via test_signature.py's `sig.define(...)` calls.
"""
import pytest

from sltn import Signature
from sltn.fol.definitions import _base_name_of
from sltn.fol.formula import And, Or, KB, AtomicFormula


# ---------------------------------------------------------------------------
# _base_name_of
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("x", "x"),
        ("x1", "x"),
        ("x12", "x"),
        ("x'", "x"),
        ("x''", "x"),
        ("t1", "t"),
        ("abc", "abc"),
    ],
)
def test_base_name_of(name, expected):
    assert _base_name_of(name) == expected


# ---------------------------------------------------------------------------
# Numbered-alias substitution (P1, P2, … for definition parameters)
# ---------------------------------------------------------------------------

def test_define_numbered_alias_resolves_to_call_site_variable_alias():
    """``Def(a)`` with body referencing the formal param's ``P1`` alias should
    substitute in the call-site variable's own ``a1`` alias, not a fresh name.
    """
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("Sim", ["S", "S"])
    sig.variable("a", S)
    sig.define("Cross", ["U"], "Sim(U, U1)")

    phi = sig.parse("Cross(a)")
    assert isinstance(phi, AtomicFormula)
    assert phi.pred_name == "Sim"
    arg_names = [a.name for a in phi.args]
    assert arg_names == ["a", "a1"]


# ---------------------------------------------------------------------------
# Definitions used inside KB clause lists
# ---------------------------------------------------------------------------

def test_define_expands_inside_kb_clause_list():
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.predicate("Q", ["S"])
    sig.variable("x", S)
    sig.define("Both", ["a"], "P(a) & Q(a)")

    kb = sig.parse("{ forall x: Both(x), forall x: P(x) }")
    assert isinstance(kb, KB)
    assert len(kb.clauses) == 2
    assert isinstance(kb.clauses[0].formula, And)


# ---------------------------------------------------------------------------
# Recursive / multi-level definition composition
# ---------------------------------------------------------------------------

def test_define_three_level_recursive_composition():
    """A definition can call another definition, which calls a third."""
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.variable("x", S)

    sig.define("L0", ["a"], "P(a)")
    sig.define("L1", ["a"], "L0(a) & L0(a)")
    sig.define("L2", ["a"], "L1(a) or L1(a)")

    phi = sig.parse("L2(x)")
    assert isinstance(phi, Or)
    assert isinstance(phi.left, And)
    assert isinstance(phi.left.left, AtomicFormula)
    assert phi.left.left.pred_name == "P"


def test_define_reused_twice_with_different_arguments_is_independent():
    """Expanding the same definition twice with different call-site arguments
    must not let one expansion's substitutions leak into the other.
    """
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.variable("x", S)
    sig.variable("y", S)
    sig.define("Id", ["a"], "P(a)")

    phi = sig.parse("Id(x) & Id(y)")
    assert isinstance(phi, And)
    assert phi.left.args[0].name == "x"
    assert phi.right.args[0].name == "y"


# ---------------------------------------------------------------------------
# Multiple numbered aliases (P1, P2, …) in a single definition body
# ---------------------------------------------------------------------------

def test_define_multiple_numbered_aliases_chain():
    """A body may reference several numbered aliases (`a1`, `a2`, …) of the
    same formal parameter; each must resolve to the corresponding numbered
    alias of the call-site variable.
    """
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("Sim", ["S", "S"])
    sig.variable("x", S)
    sig.define("Chain", ["a"], "Sim(a, a1) & Sim(a1, a2)")

    phi = sig.parse("Chain(x)")
    assert isinstance(phi, And)
    assert [a.name for a in phi.left.args] == ["x", "x1"]
    assert [a.name for a in phi.right.args] == ["x1", "x2"]


# ---------------------------------------------------------------------------
# Structural-axis capture avoidance
# ---------------------------------------------------------------------------

def test_define_call_site_axis_alias_does_not_corrupt_definition_body():
    """Calling with a renamed structural axis (`x[s1]` instead of `x[s]`) must
    rewrite the definition's internal axis tags consistently rather than
    leaving the original ``s`` tag dangling.
    """
    sig = Signature("s")
    Tok = sig.sort("Tok")
    s = sig.dimension("S")
    d = sig.dimension("D")
    sig.structural_variable("s", s)
    sig.structural_variable("d", d)
    sig.predicate("Emb", [Tok], output_dims=[d])
    sig.variable("x", Tok, dims=[s])
    sig.define("Wrap", ["X"], "Emb(X[s])[d]")

    phi = sig.parse("Wrap(x[s1])")
    assert isinstance(phi, AtomicFormula)
    assert phi.structural_axes == ["d"]
    assert phi.free_structural_variables == {"s1", "d"}
