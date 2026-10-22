"""Regression tests for the Pratt parser / lexer rewrite."""
import warnings

import pytest

from sltn import Signature
from sltn.fol.formula import AtomicFormula, Forall, Implies, TermFormula
from sltn.fol.term import FunctionApplication
from sltn.parse.tokenizer import tokenize


@pytest.fixture
def sig():
    s = Signature("p")
    S = s.sort("S")
    s.dimension("t")
    s.predicate("A", [S])
    s.predicate("B", [S])
    s.predicate("Rain", [])
    s.predicate("eq", [S, S], infix=True)
    s.function("+", [S, S], S, infix=True)
    s.function("*", [S, S], S, infix=True)
    s.variable("x", S)
    s.variable("y", S, dims=["t"])
    return s


def test_negative_selection_index(sig):
    assert sig.parse("A(y[t=-1])") is not None


def test_non_ascii_digit_is_not_an_index():
    with pytest.raises(ValueError):
        tokenize("x[t=\u0663]", {})


def test_unicode_aliases(sig):
    assert isinstance(sig.parse("A(x) \u21d2 B(x)"), Implies)
    with pytest.raises(ValueError, match="nsupported"):
        sig.parse("A(x) -> \u22a4")


def test_nullary_predicate_without_parentheses(sig):
    phi = sig.parse("Rain -> Rain()")
    assert isinstance(phi.left, AtomicFormula) and isinstance(phi.right, AtomicFormula)


def test_infix_function_is_left_associative(sig):
    phi = sig.parse("x + x * x eq x")
    assert isinstance(phi, AtomicFormula)
    lhs = phi.args[0]
    assert isinstance(lhs, FunctionApplication) and lhs.name == "*"
    assert isinstance(lhs.args[0], FunctionApplication) and lhs.args[0].name == "+"


def test_parenthesised_infix_term(sig):
    assert isinstance(sig.parse("(x + x) eq x"), AtomicFormula)


def test_infix_predicates_cannot_chain(sig):
    with pytest.raises(ValueError, match="can't be chained"):
        sig.parse("x eq x eq x")


def test_bare_term_parses_as_term(sig):
    assert isinstance(sig.parse("x + x"), FunctionApplication)
    with pytest.raises(ValueError, match="builds a formula"):
        sig.parse_term("A(x) & B(x)")


@pytest.mark.parametrize("name", ["|", "a b", "(", "1x"])
def test_invalid_infix_names_rejected(name):
    s = Signature("p")
    s.sort("S")
    with pytest.raises(ValueError):
        s.predicate(name, ["S", "S"], infix=True)


@pytest.mark.parametrize("name", ["=>", "&", "forall"])
def test_reserved_names_rejected(name):
    s = Signature("p")
    s.sort("S")
    with pytest.raises(ValueError, match="reserved"):
        s.predicate(name, ["S", "S"], infix=True)


def test_longest_match_for_user_symbols():
    kinds = [t.text for t in tokenize("a && b ~= c", {"&&": "pred", "~=": "pred"})]
    assert "&&" in kinds and "~=" in kinds


def test_selection_then_infix(sig):
    sig.predicate("=d", ["S", "S"], infix=True)
    assert isinstance(sig.parse("y[t][t=0] =d x"), AtomicFormula)


def test_term_macro_infix_and_prefix(sig):
    sig.define("double", ["a"], "a + a", output_sort="S")
    assert isinstance(sig.parse("double(x) eq x"), AtomicFormula)


def test_narrow_quantifier_scope_warns(sig):
    with pytest.warns(UserWarning, match="quantified and free"):
        phi = sig.parse("forall x: A(x) -> B(x)")
    assert isinstance(phi, Implies) and isinstance(phi.left, Forall)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(sig.parse("forall x: (A(x) -> B(x))"), Forall)


def test_empty_annotation_is_an_error(sig):
    with pytest.raises(ValueError):
        sig.parse("A(y[])")
