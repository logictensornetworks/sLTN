"""Tests for parser operator precedence and associativity.

Precedence (highest to lowest binding):
    ¬ / quantifier (non-greedy)  >  ∧  >  ∨  >  →  >  ↔

Associativity:
    ∧, ∨, ↔  — left-associative
    →         — right-associative
    ¬         — right-associative (prefix)
    quantifiers — non-greedy (bind to immediate sub-expression only)

OR uses keyword ``or`` or ``∨``; ``|`` is NOT a disjunction operator in sLTN.
"""
import pytest

from sltn import Signature
from sltn.fol.formula import (
    And, Or, Implies, Equivalent, Not,
    AtomicFormula, Forall, Exists,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sig():
    """Minimal signature with predicates A, B, C (unary) and variable x."""
    s = Signature("prec_test")
    S = s.sort("S")
    s.predicate("A", ["S"])
    s.predicate("B", ["S"])
    s.predicate("C", ["S"])
    s.variable("x", S)
    return s


def _parse(sig, formula: str):
    return sig.parse(formula)


# ---------------------------------------------------------------------------
# Conjunction binds tighter than disjunction: A & B or C → (A & B) or C
# ---------------------------------------------------------------------------

def test_and_tighter_than_or(sig):
    phi = _parse(sig, "A(x) & B(x) or C(x)")
    assert isinstance(phi, Or), f"Expected Or at root, got {type(phi).__name__}"
    assert isinstance(phi.left, And)


def test_and_tighter_than_or_unicode(sig):
    phi = _parse(sig, "A(x) ∧ B(x) ∨ C(x)")
    assert isinstance(phi, Or)
    assert isinstance(phi.left, And)


# ---------------------------------------------------------------------------
# Disjunction binds tighter than implication: A or B -> C → (A or B) -> C
# ---------------------------------------------------------------------------

def test_or_tighter_than_implies(sig):
    phi = _parse(sig, "A(x) or B(x) -> C(x)")
    assert isinstance(phi, Implies), f"Expected Implies at root, got {type(phi).__name__}"
    assert isinstance(phi.left, Or)


def test_or_tighter_than_implies_unicode(sig):
    phi = _parse(sig, "A(x) ∨ B(x) → C(x)")
    assert isinstance(phi, Implies)
    assert isinstance(phi.left, Or)


# ---------------------------------------------------------------------------
# Implication binds tighter than iff: A -> B iff C → (A -> B) iff C
# ---------------------------------------------------------------------------

def test_implies_tighter_than_iff(sig):
    phi = _parse(sig, "A(x) -> B(x) iff C(x)")
    assert isinstance(phi, Equivalent), f"Expected Equivalent at root, got {type(phi).__name__}"
    assert isinstance(phi.left, Implies)


# ---------------------------------------------------------------------------
# Negation binds tightest among connectives
# ---------------------------------------------------------------------------

def test_not_tighter_than_and(sig):
    phi = _parse(sig, "not A(x) & B(x)")
    assert isinstance(phi, And), f"Expected And at root, got {type(phi).__name__}"
    assert isinstance(phi.left, Not)


def test_not_tighter_than_or(sig):
    phi = _parse(sig, "not A(x) or B(x)")
    assert isinstance(phi, Or)
    assert isinstance(phi.left, Not)


def test_not_tighter_than_implies(sig):
    phi = _parse(sig, "not A(x) -> B(x)")
    assert isinstance(phi, Implies)
    assert isinstance(phi.left, Not)


def test_not_unicode(sig):
    phi = _parse(sig, "¬A(x) & B(x)")
    assert isinstance(phi, And)
    assert isinstance(phi.left, Not)


def test_not_tilde(sig):
    phi = _parse(sig, "~A(x) & B(x)")
    assert isinstance(phi, And)
    assert isinstance(phi.left, Not)


# ---------------------------------------------------------------------------
# Associativity
# ---------------------------------------------------------------------------

def test_and_is_left_associative(sig):
    phi = _parse(sig, "A(x) & B(x) & C(x)")
    assert isinstance(phi, And)
    assert isinstance(phi.left, And), "Expected left-assoc: (A & B) & C"
    assert isinstance(phi.right, AtomicFormula)


def test_or_is_left_associative(sig):
    phi = _parse(sig, "A(x) or B(x) or C(x)")
    assert isinstance(phi, Or)
    assert isinstance(phi.left, Or), "Expected left-assoc: (A or B) or C"


def test_iff_is_left_associative(sig):
    phi = _parse(sig, "A(x) iff B(x) iff C(x)")
    assert isinstance(phi, Equivalent)
    assert isinstance(phi.left, Equivalent), "Expected left-assoc: (A iff B) iff C"


def test_implies_is_right_associative(sig):
    phi = _parse(sig, "A(x) -> B(x) -> C(x)")
    assert isinstance(phi, Implies)
    assert isinstance(phi.right, Implies), "Expected right-assoc: A -> (B -> C)"
    assert isinstance(phi.left, AtomicFormula)


# ---------------------------------------------------------------------------
# Non-greedy quantifiers
# ---------------------------------------------------------------------------

def test_forall_is_nongreedy_with_and(sig):
    # forall x: A(x) & B(x)  →  (forall x: A(x)) & B(x)
    phi = _parse(sig, "forall x: A(x) & B(x)")
    assert isinstance(phi, And), f"Expected And at root (non-greedy forall), got {type(phi).__name__}"
    assert isinstance(phi.left, Forall)
    assert isinstance(phi.right, AtomicFormula)


def test_forall_is_nongreedy_with_or(sig):
    phi = _parse(sig, "forall x: A(x) or B(x)")
    assert isinstance(phi, Or)
    assert isinstance(phi.left, Forall)


def test_forall_is_nongreedy_with_implies(sig):
    phi = _parse(sig, "forall x: A(x) -> B(x)")
    assert isinstance(phi, Implies)
    assert isinstance(phi.left, Forall)


def test_exists_is_nongreedy_with_and(sig):
    phi = _parse(sig, "exists x: A(x) & B(x)")
    assert isinstance(phi, And)
    assert isinstance(phi.left, Exists)


def test_quantifier_as_rhs_of_and(sig):
    # A(x) & forall x: B(x)  →  A(x) & (forall x: B(x))
    phi = _parse(sig, "A(x) & forall x: B(x)")
    assert isinstance(phi, And)
    assert isinstance(phi.left, AtomicFormula)
    assert isinstance(phi.right, Forall)


def test_not_applied_to_quantifier(sig):
    # not forall x: A(x)  →  Not(Forall(x, A(x)))
    phi = _parse(sig, "not forall x: A(x)")
    assert isinstance(phi, Not)
    assert isinstance(phi.operand, Forall)


def test_forall_unicode(sig):
    phi = _parse(sig, "∀ x: A(x) & B(x)")
    assert isinstance(phi, And)
    assert isinstance(phi.left, Forall)


def test_forall_explicit_scope_with_parens(sig):
    # Parentheses extend quantifier scope
    phi = _parse(sig, "forall x: (A(x) & B(x))")
    assert isinstance(phi, Forall)
    assert isinstance(phi.formula, And)


# ---------------------------------------------------------------------------
# Keyword aliases
# ---------------------------------------------------------------------------

def test_keyword_and(sig):
    phi = _parse(sig, "A(x) and B(x)")
    assert isinstance(phi, And)


def test_keyword_or(sig):
    phi = _parse(sig, "A(x) or B(x)")
    assert isinstance(phi, Or)


def test_keyword_not(sig):
    phi = _parse(sig, "not A(x)")
    assert isinstance(phi, Not)


def test_keyword_implies(sig):
    phi = _parse(sig, "A(x) implies B(x)")
    assert isinstance(phi, Implies)


def test_keyword_iff(sig):
    phi = _parse(sig, "A(x) iff B(x)")
    assert isinstance(phi, Equivalent)
