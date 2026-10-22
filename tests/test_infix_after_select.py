"""Regression tests: infix predicates/functions following a `[axis=idx]` select.

Verifies that ``name(args)[out_axes]`` followed by one or more
``[axis=idx]`` selects and *then* an infix predicate (e.g.
``digit(x1[p])[p=0] =d d0``) is parsed correctly: the parser must check for
a trailing infix operator *after* any axis selects, not just right after
``(args)``, and infix predicates must be tokenised consistently in both the
formula parser and the term-fallback path so that error messages remain
accurate.
"""
import pytest

from sltn import Signature
from sltn.fol.formula import AtomicFormula
from sltn.fol.term import TermAxisSelect, FunctionApplication
from sltn.parse import parse_formula_string, parse_term_string


@pytest.fixture
def sig():
    s = Signature("infix_select_test")
    Digit = s.sort("Digit")
    Img = s.sort("Img")
    p = s.dimension("P")
    s.structural_variable("p", p)
    s.variable("x1", Img, dims=[p])
    s.variable("x2", Img, dims=[p])
    s.constant("d0", Digit)
    s.function("digit", [Img], Digit)
    s.function("add", [Digit, Digit], Digit)
    s.predicate("=d", [Digit, Digit], infix=True)
    return s


def test_infix_pred_after_single_select(sig):
    phi = sig.parse("digit(x1[p])[p=0] =d d0")
    assert isinstance(phi, AtomicFormula)
    assert phi.pred_name == "=d"
    lhs = phi.args[0]
    assert isinstance(lhs, TermAxisSelect)
    assert lhs.axis == "p" and lhs.index == 0


def test_infix_pred_after_select_both_sides(sig):
    phi = sig.parse("digit(x1[p])[p=0] =d digit(x2[p])[p=0]")
    assert isinstance(phi, AtomicFormula)
    assert phi.pred_name == "=d"
    assert isinstance(phi.args[0], TermAxisSelect)
    assert isinstance(phi.args[1], TermAxisSelect)


def test_infix_fn_after_select_still_chains(sig):
    # add(...) is not infix here, but the LHS select-then-infix-predicate
    # path must still work when the RHS itself contains a plain call.
    phi = sig.parse("digit(x1[p])[p=0] =d add(d0, d0)")
    assert isinstance(phi, AtomicFormula)
    rhs = phi.args[1]
    assert isinstance(rhs, FunctionApplication)
    assert rhs.name == "add"


def test_plain_predicate_call_with_select_unaffected(sig):
    """A predicate call followed by a select and *no* trailing infix operator
    must still parse as a plain (non-infix) atomic formula chain, i.e. the
    lookahead must not force every ``[axis=idx]`` onto the term path."""
    from sltn.fol.formula import FormulaAxisSelect

    s = Signature("plain_select_test")
    S = s.sort("S")
    p = s.dimension("Pd")
    s.structural_variable("p", p)
    s.variable("x", S, dims=[p])
    s.predicate("P", [S])
    phi = s.parse("P(x[p])[p=0]")
    assert isinstance(phi, FormulaAxisSelect)
    assert phi.axis == "p" and phi.index == 0


def test_infix_pred_after_two_selects():
    """Two chained `[axis=idx]` selects before the infix predicate."""
    s = Signature("two_select_test")
    T = s.sort("T")
    I = s.sort("I")
    p = s.dimension("P")
    q = s.dimension("Q")
    s.structural_variable("p", p)
    s.structural_variable("q", q)
    s.variable("x", I, dims=[p, q])
    s.constant("c0", T)
    s.function("f", [I], T, input_dims=[p, q], output_dims=[p, q])
    s.predicate("=t", [T, T], infix=True)
    phi = s.parse("f(x[p, q])[p, q][p=0][q=1] =t c0")
    assert isinstance(phi, AtomicFormula)
    assert phi.pred_name == "=t"
    lhs = phi.args[0]
    assert isinstance(lhs, TermAxisSelect) and lhs.axis == "q" and lhs.index == 1
    assert isinstance(lhs.base_term, TermAxisSelect) and lhs.base_term.axis == "p"


def test_parse_formula_string_low_level_repro():
    """Same case, exercised directly at the tokenizer/parser level (no Signature)."""
    ast = parse_formula_string(
        "digit(x1[p])[p=0] =d d0", infix_preds=["=d"], infix_fns=None
    )
    assert isinstance(ast, AtomicFormula)
    assert ast.pred_name == "=d"


def test_parse_term_string_tokenizes_infix_pred_for_accurate_errors():
    """Even though a term can never *contain* a top-level infix predicate,
    passing `infix_preds` to `parse_term_string` must still tokenize it as one
    unit, so a fallback failure reports the operator itself rather than a
    mis-split '=' token."""
    with pytest.raises(ValueError, match=r"'=d'"):
        parse_term_string("digit(x1[p])[p=0] =d d0", infix_preds=["=d"])


def test_parse_surfaces_original_formula_error_not_confusing_term_error(sig):
    """When a string is genuinely a broken formula (not a valid term either),
    `Signature.parse`'s term-fallback must not mask the real formula error."""
    with pytest.raises(ValueError) as excinfo:
        sig.parse("digit(x1[p])[p=0] =d =d d0")
    # The surfaced error must reference the formula-level failure, not the
    # unrelated "Failed to parse term" fallback message.
    assert "Failed to parse formula" in str(excinfo.value)
