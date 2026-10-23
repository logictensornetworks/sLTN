"""Tests for Signature: declaration, parsing, symbol resolution, aliases, define."""
import pytest
import torch
from sltn import Signature, Interpretation, Real
from sltn.signature import Sort, Dimension, VariableSymbol, PredicateSymbol, FunctionSymbol
from sltn.fol.formula import AtomicFormula, Forall, And, Implies, FormulaAxisSelect, Or
from sltn.fol.formula import TermFormula
from sltn.fol.term import FunctionApplication


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------

def test_sort_declaration():
    sig = Signature("test")
    A = sig.sort("Animal")
    assert isinstance(A, Sort)
    assert A.name == "Animal"


def test_duplicate_sort_warns_and_replaces():
    sig = Signature("test")
    old = sig.sort("A")
    with pytest.warns(UserWarning, match="redeclared"):
        new = sig.sort("A")
    assert new is not old and sig.get_sort("A") is new


def test_dimension_declaration():
    sig = Signature("test")
    T = sig.dimension("time")
    assert isinstance(T, Dimension)
    assert T.name == "time"


def test_predicate_declaration():
    sig = Signature("test")
    S = sig.sort("S")
    P = sig.predicate("P", ["S"])
    assert isinstance(P, PredicateSymbol)
    assert P.name == "P"
    assert P.input_sorts[0] is S


def test_function_declaration():
    sig = Signature("test")
    S = sig.sort("S")
    f = sig.function("f", ["S"], "S")
    assert isinstance(f, FunctionSymbol)
    assert f.name == "f"


def test_variable_declaration():
    sig = Signature("test")
    S = sig.sort("S")
    x = sig.variable("x", S)
    assert isinstance(x, VariableSymbol)
    assert x.sort is S


def test_variable_with_dim():
    sig = Signature("test")
    S = sig.sort("S")
    T = sig.dimension("t")
    x = sig.variable("x", S, dims=[T])
    assert len(x.structural_variables) == 1 and x.structural_variables[0] is T


def test_structural_relation_declaration():
    sig = Signature("test")
    T = sig.dimension("t")
    sp = sig.structural_relation("next", [T])
    assert sp.name == "next"


def test_get_lookup():
    sig = Signature("test")
    S = sig.sort("S")
    found = sig.get("S")
    assert found is S


def test_get_missing_raises():
    sig = Signature("test")
    with pytest.raises(KeyError):
        sig.get("NotDeclared")


# ---------------------------------------------------------------------------
# parse — basic formulas
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_sig():
    sig = Signature("basic")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.predicate("Q", ["S", "S"])
    sig.variable("x", S)
    sig.variable("y", S)
    return sig


def test_parse_atomic(basic_sig):
    phi = basic_sig.parse("P(x)")
    assert isinstance(phi, AtomicFormula)
    assert phi.pred_name == "P"


def test_parse_and(basic_sig):
    phi = basic_sig.parse("P(x) & Q(x, y)")
    assert isinstance(phi, And)


def test_parse_implies(basic_sig):
    phi = basic_sig.parse("P(x) -> Q(x, y)")
    assert isinstance(phi, Implies)


def test_parse_forall(basic_sig):
    phi = basic_sig.parse("forall x: P(x)")
    assert isinstance(phi, Forall)
    assert "x" in phi.var_names


def test_parse_unicode_quantifier(basic_sig):
    phi = basic_sig.parse("∀ x: P(x)")
    assert isinstance(phi, Forall)


def test_parse_invalid_formula_raises(basic_sig):
    with pytest.raises((ValueError, Exception)):
        basic_sig.parse("&&&")


def test_parse_resolves_symbols(basic_sig):
    phi = basic_sig.parse("P(x)")
    assert isinstance(phi.symbol, PredicateSymbol)


# ---------------------------------------------------------------------------
# parse — structural axes
# ---------------------------------------------------------------------------

def test_parse_structural_axis_annotation():
    sig = Signature("s")
    S = sig.sort("S")
    T = sig.dimension("T")
    sig.structural_variable("t", T)
    sig.predicate("A", ["S"], input_dims=[T], output_dims=[T])
    sig.variable("x", S, dims=[T])
    phi = sig.parse("A(x[t])[t]")
    assert isinstance(phi, AtomicFormula)
    assert phi.structural_axes == ["t"]


def test_parse_struct_guard():
    sig = Signature("s")
    S = sig.sort("S")
    T = sig.dimension("T")
    sig.structural_variable("t", T)
    A = sig.predicate("A", ["S"], input_dims=[T], output_dims=[T])
    x = sig.variable("x", S, dims=[T])
    nxt = sig.structural_relation("next", [T, T])
    phi = sig.parse("forall t, t1 | next(t, t1): (A(x[t])[t] -> A(x[t1])[t1])")
    assert isinstance(phi, Forall)
    assert phi.guard is not None


def test_parse_formula_level_axis_select_on_compound_formula():
    sig = Signature("s")
    S = sig.sort("S")
    T = sig.dimension("T")
    sig.structural_variable("t", T)
    sig.predicate("A", ["S"], input_dims=[T], output_dims=[T])
    sig.variable("x", S, dims=[T])

    phi = sig.parse("(A(x[t])[t] & A(x[t])[t])[t=0]")
    assert isinstance(phi, FormulaAxisSelect)
    assert phi.axis == "t"
    assert phi.index == 0


def test_parse_formula_level_axis_select_on_atomic_formula():
    sig = Signature("s")
    S = sig.sort("S")
    T = sig.dimension("T")
    sig.structural_variable("t", T)
    sig.predicate("A", ["S"], input_dims=[T], output_dims=[T])
    sig.variable("x", S, dims=[T])

    phi = sig.parse("A(x[t])[t][t=0]")
    assert isinstance(phi, FormulaAxisSelect)
    assert phi.axis == "t"
    assert phi.index == 0


def test_formula_level_axis_select_updates_free_structural_variables():
    sig = Signature("s")
    S = sig.sort("S")
    T = sig.dimension("T")
    sig.structural_variable("t", T)
    sig.predicate("A", ["S"], input_dims=[T], output_dims=[T])
    sig.variable("x", S, dims=[T])

    phi = sig.parse("(A(x[t])[t] & A(x[t])[t])[t=0]")
    assert phi.free_structural_variables == set()


# ---------------------------------------------------------------------------
# define — syntactic abbreviations
# ---------------------------------------------------------------------------

def test_define_expands_at_parse():
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.predicate("Q", ["S"])
    sig.variable("x", S)
    sig.define("Conj", ["a", "b"], "P(a) & Q(b)")
    phi = sig.parse("Conj(x, x)")
    assert isinstance(phi, And)


def test_define_infix():
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("Likes", ["S", "S"])
    sig.variable("x", S)
    sig.variable("y", S)
    sig.define("likes", ["a", "b"], "Likes(a, b)", infix=True)
    phi = sig.parse("x likes y")
    assert isinstance(phi, AtomicFormula)


def test_define_with_metadata_prints_arity_and_dims():
    sig = Signature("s")
    image = sig.sort("Image")
    sig.dimension("t")
    digit = sig.dimension("digit")
    sig.function("mnist", ["Image"], "Image", output_dims=[digit])
    sig.define(
        "is_zero",
        ["x"],
        "mnist(x)[digit][digit=0]",
        input_sorts=["Image"],
        output_sort="Image",
        input_dims=["t"],
        output_dims=["digit"],
    )

    r = repr(sig)
    assert "Definitions:" in r
    assert "is_zero" in r


def test_define_metadata_length_is_validated():
    sig = Signature("s")
    sig.sort("S")

    with pytest.raises(ValueError, match="input_sorts"):
        sig.define("Bad", ["x", "y"], "P(x)", input_sorts=["S"])


def test_define_metadata_input_sort_checked_during_parse():
    sig = Signature("s")
    s1 = sig.sort("S1")
    s2 = sig.sort("S2")
    sig.predicate("P", ["S1"])
    sig.variable("x", s1)
    sig.define("Alias", ["a"], "P(a)", input_sorts=["S2"])

    with pytest.raises(TypeError, match="argument 0 has sort 'S1', expected 'S2'"):
        sig.parse("Alias(x)")


def test_define_metadata_input_dims_checked_during_parse():
    sig = Signature("s")
    s = sig.sort("S")
    sig.dimension("t")
    sig.predicate("P", ["S"])
    sig.variable("x", s)
    sig.define("Alias", ["a"], "P(a)", input_dims=["t"])

    with pytest.raises(ValueError, match="missing required input structural dim"):
        sig.parse("Alias(x)")


def test_define_input_dims_accepts_structural_axis_alias():
    """``input_dims`` lists *dimensions*; call sites may use ``s1``, an alias of ``s`` over ``S``."""
    sig = Signature("s")
    Tok = sig.sort("Tok")
    sig.dimension("S")
    sig.dimension("D")
    sig.structural_variable("s", "S")
    sig.structural_variable("d", "D")
    sig.predicate("P", [Tok], input_dims=[sig.get_dimension("S")], output_dims=[sig.get_dimension("D")])
    sig.variable("x", Tok, dims=["S"])
    sig.define("Emb", ["X"], "P(X[s])[d]", input_dims=["S"], output_dims=["D"])
    sig.parse("Emb(x[s1])")


def test_define_rewrites_template_output_axes_when_parameter_uses_struct_axis_alias():
    """Call ``x[s1]`` must rename template output tags ``[s, …]`` to ``[s1, …]`` (not leave ``s``)."""
    from sltn.fol.formula import TermFormula
    from sltn.fol.term import FunctionApplication

    sig = Signature("s")
    Tok = sig.sort("Tok")
    sdim = sig.dimension("S")
    vdim = sig.dimension("V")
    sig.structural_variable("s", sdim)
    sig.structural_variable("v", vdim)
    sig.function("G", [Tok], "Tok", input_dims=[sdim], output_dims=[sdim, vdim])
    sig.variable("x", Tok, dims=["S"])
    sig.define("F", ["X"], "G(X[s])[s,v]")
    term = sig.parse("F(x[s1])")
    assert isinstance(term, TermFormula)
    inner = term.term
    assert isinstance(inner, FunctionApplication)
    assert inner.output_axes == ["s1", "v"]


def test_define_distinct_param_axis_indices_duplicate_subtree():
    """When a body uses ``X[s, …]`` and ``X[s1, …]``, each occurrence must get the right axes.

    Regression: naive substitution reused the same AST for both, so the second
    slot kept ``s`` instead of ``s1`` (broken ``nn.Module`` batching / attention).
    """
    from sltn.fol.formula import AtomicFormula, Forall, Or
    from sltn.fol.term import FunctionApplication

    sig = Signature("t")
    Tok = sig.sort("Tok")
    for dim, sv in (("S", "s"), ("D", "d"), ("V", "v")):
        sig.dimension(dim)
        sig.structural_variable(sv, dim)
    sig.variable("x", Tok, dims=["S"])
    sig.constant("Table", "Bool", dims=["V", "D"])
    sig.function("OneHotEncoding", [Tok], "Bool", output_dims=["V"])
    sig.predicate("=f", ["Bool", "Bool"], input_dims=["V", "V"], infix=True)
    sig.predicate("Head", ["Bool", "Bool"], input_dims=["S", "S", "D"], output_dims=["S", "S"])
    sig.function("Val", ["Bool"], "Bool", input_dims=["D"], output_dims=["D"])
    sig.define(
        "Embed",
        ["X"],
        "OneHotEncoding(X[s])[v] =f Table[v,d]",
        input_dims=[],
        output_dims=["D"],
    )
    sig.define(
        "L0",
        ["X"],
        "(∀ s1 | Head(X[s,d], X[s1,d])[s,s1] : Val(X[s1,d])[d]) ∨ X[s,d]",
    )
    phi = sig.parse("L0(Embed(x[s]))")
    assert isinstance(phi, Or)
    forall = phi.left
    assert isinstance(forall, Forall)
    guard = forall.guard
    assert isinstance(guard, AtomicFormula)
    fn0 = guard.args[0].args[0]
    fn1 = guard.args[1].args[0]
    assert isinstance(fn0, FunctionApplication) and isinstance(fn1, FunctionApplication)
    assert fn0.args[0].axis_annotations == ["s"]
    assert fn1.args[0].axis_annotations == ["s1"]


def test_define_input_dims_sees_structural_axes_on_defined_symbol_atomic_formula():
    """A defined symbol's result axes are seen by the ``input_dims`` check of an enclosing definition."""
    sig = Signature("t")
    sig.sort("Token")
    for dim, sv in (("S", "s"), ("V", "v"), ("D", "d")):
        sig.dimension(dim)
        sig.structural_variable(sv, dim)
    sig.variable("x", "Token", dims=["S"])
    sig.constant("Table", "Bool", dims=["V", "D"])
    sig.function("OneHotEncoding", ["Token"], "Bool", output_dims=["V"])
    sig.predicate("SimilarityV", ["Bool", "Bool"], input_dims=["V", "V"])
    sig.predicate("SimilarityD", ["Bool", "Bool"], input_dims=["D", "D"])
    sig.define(
        "Embed",
        ["X"],
        "SimilarityV(OneHotEncoding(X[s])[v], Table[v,d])",
        input_dims=["S"],
        output_dims=["D"],
    )
    sig.define(
        "DeEmbed",
        ["X"],
        "SimilarityD(X[s,d], Table[v,d])",
        input_dims=["D"],
        output_dims=["V"],
    )
    inner = sig.parse("Embed(x[s])")
    assert inner.free_structural_variables == {"s", "d"}
    assert sig.parse("DeEmbed(Embed(x[s]))").free_structural_variables == {"s", "v"}
    with pytest.raises(ValueError, match="missing required input structural dim"):
        sig.parse("DeEmbed(OneHotEncoding(x[s])[v])")


def test_define_metadata_output_sort_checked_during_parse():
    sig = Signature("s")
    s = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.variable("x", s)
    sig.define("Alias", ["a"], "P(a)", output_sort="S")

    with pytest.raises(TypeError, match="output sort is 'Bool', expected 'S'"):
        sig.parse("Alias(x)")


# ---------------------------------------------------------------------------
# VariableSymbol aliases (x1, x2 auto-created from x)
# ---------------------------------------------------------------------------

def test_variable_alias_resolution():
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("B", ["S", "S"])
    sig.variable("u", S)
    phi = sig.parse("B(u, u1)")
    assert isinstance(phi, AtomicFormula)


@pytest.mark.parametrize("formula", ["forall u1: C(u1)", "forall u': C(u')", "forall u, u1: B(u, u1)"])
def test_variable_alias_as_quantified_name(formula):
    """A variable alias can be quantified without first appearing free."""
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("B", ["S", "S"])
    sig.predicate("C", ["S"])
    sig.variable("u", S)
    phi = sig.parse(formula)
    assert isinstance(phi, Forall)


def test_quantified_variable_alias_shares_base_grounding():
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("B", ["S", "S"])
    sig.variable("u", S)
    interp = Interpretation(sig)
    interp[S] = Real
    interp["u"] = torch.tensor([[0.0], [1.0], [2.0]])
    interp["B"] = lambda a, b: torch.sigmoid(a - b)
    # u1 ranges over the same 3 individuals as u: the result aggregates the full 3x3 grid
    grid = torch.sigmoid(torch.tensor([0.0, 1.0, 2.0])[:, None] - torch.tensor([0.0, 1.0, 2.0])[None, :])
    expected = interp.logic.with_defaults().Forall(grid.reshape(-1), dim=0)
    assert torch.allclose(interp(sig.parse("forall u, u1: B(u, u1)")).value.reshape(()), expected)


def test_failed_parse_does_not_register_aliases():
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("C", ["S"])
    sig.variable("u", S)
    sig.dimension("T")
    before = (dict(sig._variables), dict(sig._variables_aliases),
              dict(sig._structural_variables), dict(sig._structural_variables_aliases))
    with pytest.raises(NameError):
        sig.parse("forall T: C(u2) and Nope(u)")
    after = (sig._variables, sig._variables_aliases,
             sig._structural_variables, sig._structural_variables_aliases)
    assert after == before


def test_define_formula_expanded_in_nested_fnapp():
    """A definition with a formula body may appear as an argument to another defined call."""
    sig = Signature("s")
    Tok = sig.sort("Tok")
    sig.predicate("P", [Tok])
    sig.variable("x", Tok)
    sig.define("L0", ["X"], "P(X) or P(X)")
    B = sig.get_sort("Bool")
    sig.function("Wrap", [B], B)
    sig.define("L1", ["Y"], "Wrap(Y)")
    phi = sig.parse("L1(L0(x))")
    assert isinstance(phi, TermFormula)
    inner = phi.term
    assert isinstance(inner, FunctionApplication)
    assert inner.name == "Wrap"
    from sltn.fol.formula import Or

    assert isinstance(inner.args[0], Or)


def test_define_formula_body_quantifier_or_struct_symref_parses():
    """Bodies like ``(... forall ...) ∨ X[s,d]`` need ``X[s,d]`` as a formula primary."""
    sig = Signature("s")
    Tok = sig.sort("Tok")
    s = sig.dimension("S")
    d = sig.dimension("D")
    sig.structural_variable("s", s)
    sig.structural_variable("d", d)
    sig.predicate("Head_L2", [Tok, Tok], output_dims=[s, s])
    sig.predicate("Value_L2", [Tok])
    sig.variable("X", Tok, dims=[s, d])
    body = "(forall s1 | Head_L2(X[s,d], X[s1,d])[s,s1] : Value_L2(X[s,d])) ∨ X[s,d]"
    sig.define("L2", ["X"], body)
    phi = sig.parse("L2(X)")
    assert isinstance(phi, Or)
    assert isinstance(phi.left, Forall)
    assert isinstance(phi.right, TermFormula)


def test_infix_predicate_parses_when_lhs_is_symref_with_axis_list():
    from sltn.fol.formula import AtomicFormula
    from sltn.parse import parse_formula_string

    ast = parse_formula_string("X[s,d] =b Table[v,d]", infix_preds=["=b", "=f"], infix_fns=None)
    assert isinstance(ast, AtomicFormula)
    assert ast.pred_name == "=b"


def test_parse_term_with_sig_expands_sig_define():
    """``parse_term_with_sig`` must run definition expansion (same as the formula ``parse`` path)."""
    from sltn.fol.term import FunctionApplication
    from sltn.parse import parse_term

    sig = Signature("t")
    sig.sort("Tok")
    for dim, sv in (("S", "s"), ("V", "v"), ("D", "d")):
        sig.dimension(dim)
        sig.structural_variable(sv, dim)
    sig.variable("x", "Tok", dims=["S"])
    sig.constant("Table", "Bool", dims=["V", "D"])
    sig.function("OneHotEncoding", ["Tok"], "Bool", output_dims=["V"])
    sig.predicate("=f", ["Bool", "Bool"], input_dims=["V", "V"], infix=True)
    sig.define("Embed", ["X"], "OneHotEncoding(X[s])[v] =f Table[v,d]", input_dims=[], output_dims=["D"])

    ast = parse_term("Embed(x[s])", sig)
    assert not (
        isinstance(ast, FunctionApplication)
        and (ast.symbol if isinstance(ast.symbol, str) else ast.symbol.name) == "Embed"
    )


def test_define_term_body_in_formula_position_parses():
    sig = Signature("s")
    S = sig.sort("Image")
    L = sig.sort("Logit")
    D = sig.dimension("Digit")
    sig.structural_variable("d", D)
    sig.function("mnist", ["Image"], "Logit", output_dims=[D])
    sig.variable("x", S)

    # A derived predicate body may be provided as a bool-valued term.
    sig.define("is_zero", ["a"], "mnist(a)[d][d=0]")
    phi = sig.parse("is_zero(x)")

    assert isinstance(phi, TermFormula)


def test_parse_keeps_surface_after_definition_expansion():
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.variable("x", S)
    sig.define("Alias", ["a"], "P(a)")

    phi = sig.parse("Alias(x)")
    assert str(phi) == "P(x)"
    assert isinstance(phi, AtomicFormula)


# ---------------------------------------------------------------------------
# Signature repr
# ---------------------------------------------------------------------------

def test_repr_contains_sort_names(basic_sig):
    r = repr(basic_sig)
    assert "S" in r
    assert "Predicates:" in r
    assert "P: S ⟶ Bool" in r


def test_repr_shows_defined_formulas_section():
    sig = Signature("s")
    S = sig.sort("S")
    sig.predicate("P", ["S"])
    sig.variable("x", S)
    sig.define("Alias", ["a"], "P(a)")

    r = repr(sig)
    assert "Definitions:" in r
    assert "Alias" in r
    assert "P(a)" in r


def test_repr_shows_defined_terms_section():
    sig = Signature("s")
    S = sig.sort("Image")
    L = sig.sort("Logit")
    D = sig.dimension("digit")
    sig.function("mnist", ["Image"], "Logit", output_dims=[D])
    sig.variable("x", S)
    sig.define("is_zero", ["a"], "mnist(a)[digit=0]")

    r = repr(sig)
    assert "Definitions:" in r
    assert "is_zero" in r
    assert "mnist(a)[digit=0]" in r


def test_struct_var_can_be_typed_and_interpreted_as_expression():
    sig = Signature("s")
    sig.sort("Scalar")
    sig.dimension("T", sort="Scalar")
    sig.structural_variable("t", "T")

    interp = Interpretation(sig)
    interp["Scalar"] = Real
    interp["t"] = torch.linspace(0.0, 1.0, steps=5).unsqueeze(-1)

    t_expr = interp("t")
    assert t_expr.domain_type == Real
    assert [a.name for a in t_expr.structural_axes] == ["t"]
    assert [a.name for a in t_expr.domain_axes] == ["real"]


def test_typed_struct_var_validates_trailing_feature_shape():
    sig = Signature("s")
    sig.sort("Scalar")
    sig.dimension("T", sort="Scalar")
    sig.structural_variable("t", "T")

    interp = Interpretation(sig)
    interp["Scalar"] = Real

    with pytest.raises(ValueError, match="expects feature shape"):
        interp["t"] = torch.linspace(0.0, 1.0, steps=5)


def test_axis_annotation_must_match_the_dimension_it_renames():
    # Bug #43: x[u] with x over T and u over U was accepted, and the
    # T axis was then aggregated with U's operator.
    sig = Signature("ann_dims")
    I = sig.sort("Img")
    T, U = sig.dimension("T"), sig.dimension("U")
    sig.variable("x", I, dims=[T])
    sig.variable("z", I, dims=[T, U])
    sig.constant("c", I, dims=[T])
    sig.structural_variable("t", T)
    sig.structural_variable("u", U)
    sig.predicate("C", ["Img"])

    for ok in ("forall x: forall t: C(x[t])",
               "forall x: forall t1: C(x[t1])",
               "forall z: forall t: forall u: C(z[t, u])",
               "forall t: C(c[t])"):
        sig.parse(ok)

    for bad in ("forall x: forall u: C(x[u])",
                "forall x: forall u1: C(x[u1])",
                "forall z: forall u: forall t: C(z[u, t])",
                "forall u: C(c[u])"):
        before = set(sig._structural_variables)
        with pytest.raises(ValueError, match="ranges over dimension"):
            sig.parse(bad)
        assert set(sig._structural_variables) == before   # failed parse rolled back


def test_quantifier_rejects_duplicate_and_unused_variables():
    # Bug #29: duplicate bound names and unused individual variables
    # were accepted, while an unused structural variable raised.
    sig = Signature("bound_vars")
    I = sig.sort("Img")
    T = sig.dimension("T")
    U = sig.dimension("U")
    sig.variable("x", I, dims=[T])
    sig.variable("y", I)
    sig.variable("z", I)
    sig.variable("w", I, dims=[T, U])
    sig.structural_variable("t", T)
    sig.structural_variable("u", U)
    sig.predicate("C", ["Img"])
    sig.predicate("R", ["Img", "Img"])

    for bad, msg in (("forall y, y: C(y)", "more than once"),
                     ("forall t, t: C(x[t])", "more than once"),
                     ("forall y: C(x[t])", "never used"),
                     ("exists y, z: C(y)", "never used"),
                     ("forall y: forall y: C(y)", "never used"),
                     ("forall x, t: C(x[t])", "mixes individual"),
                     ("forall w: forall t, u: C(w[t, u])", "different")):
        with pytest.raises(ValueError, match=msg):
            sig.parse(bad)

    for ok in ("forall y, z: R(y, z)",
               "forall y1: C(y1)",
               "forall y | C(y): C(z)",
               "forall y: (C(y) & forall y: C(y))",
               "forall x: forall t, t1: (C(x[t]) & C(x[t1]))",
               "{ forall y: C(y), forall z: C(z) }"):
        sig.parse(ok)


def test_kb_built_in_python_is_validated():
    # Bug #28: KB([...]) and `+` never checked their clauses.
    from sltn.fol.formula import KB
    sig = Signature("kb_python")
    I = sig.sort("Img")
    sig.variable("y", I)
    sig.variable("z", I)
    sig.constant("c", I)
    sig.predicate("C", ["Img"])
    sig.predicate("G", ["Img"])
    closed = sig.parse("forall z: G(z)")

    with pytest.raises(ValueError, match="closed formulas"):
        KB([sig.parse("C(y)"), closed])
    kb = KB([sig.parse("forall y: C(y)"), sig.parse("C(c)")])
    with pytest.raises(ValueError, match="closed formulas"):
        kb + sig.parse("G(z)")
    with pytest.raises(TypeError):
        kb + "garbage"
    with pytest.raises(TypeError):
        "garbage" + kb
    with pytest.raises(TypeError, match="must be Formula"):
        KB(["garbage"])

    assert len((kb + closed).clauses) == 3
    assert len((closed + kb).clauses) == 3
    assert len((kb + kb).clauses) == 4
    # a parsed KB is still validated by the pipeline
    with pytest.raises(ValueError, match="closed formulas"):
        sig.parse("{ C(y), forall z: G(z) }")


@pytest.mark.parametrize("text, expected", [
    ("C(video[t])", {"t"}),
    ("forall t: C(video[t])", set()),
    ("C(grid[t, u])", {"t", "u"}),
    ("C(z[t, u])", {"t", "u"}),
    ("C(z[t1, u])", {"t1", "u"}),
])
def test_fsv_of_annotated_variables_and_constants_matches_evaluation(text, expected):
    # Bug #24: a constant's annotations were ignored, and a variable
    # with fewer annotations than dims dropped the unannotated axes.
    from sltn import Type
    from sltn.fol.definitions import _child_nodes
    from sltn.interpretation.operators import AggregPMeanError
    sig = Signature("fsv_ann")
    I = sig.sort("Img")
    T, U = sig.dimension("T"), sig.dimension("U")
    sig.structural_variable("t", T)
    sig.structural_variable("u", U)
    sig.variable("z", I, dims=[T, U])
    sig.constant("video", I, dims=[T])
    sig.constant("grid", I, dims=[T, U])
    sig.predicate("C", ["Img"])
    i = Interpretation(sig)
    i["Img"] = Type("Img", shape=(1,), axis_names=["f"])
    i["z"] = torch.rand(2, 3, 4, 1)
    i["video"] = torch.rand(3, 1)
    i["grid"] = torch.rand(3, 4, 1)
    i["C"] = lambda a: torch.sigmoid(a)
    i.logic["forall,T"] = AggregPMeanError()

    phi = sig.parse(text)
    assert phi.free_structural_variables == expected
    assert {a.name for a in i(phi).structural_axes} == expected

    def wipe(node):   # the uncached path, as for a tree built in Python
        node._free_structural_variables = None
        for child in _child_nodes(node):
            wipe(child)
    wipe(phi)
    assert phi.free_structural_variables == expected
