"""Recursive-descent formula parser for sLTN.

Grammar semantics
-----------------
Operator precedence (highest first):

  not  (prefix, right-assoc)
  &  / and / ∧   (left)
  or  / ∨        (left)
  ->  / →        (right)
  <-> / ↔        (left)

"""

from __future__ import annotations

from sltn.fol.formula import (
    And, Or, Not, Implies, Equivalent, # Connectives
    Forall, Exists, StructuralForall, StructuralExists, # Quantifiers
    AtomicFormula, Formula, KB, FormulaAxisSelect, TermFormula, # Formula
)

from sltn.fol.term import TermAxisSelect, _CallTerm, _NameTerm, Term
from sltn.parse.tokenizer import TK, Token, tokenize


# ---------------------------------------------------------------------------
# Parser class
# ---------------------------------------------------------------------------


class _Parser:
    """Stateful parser for one formula / term string.

    Both formula-level and term-level productions live here so they can share
    the same token stream without re-tokenising sub-strings.
    """

    def __init__(
        self,
        tokens: list[Token],
        original: str,
        infix_preds: list[str],
        infix_fns: list[str],
    ) -> None:
        self._tokens = tokens
        self._pos    = 0
        self._original = original
        self._infix_preds: frozenset[str] = frozenset(infix_preds)
        self._infix_fns:   frozenset[str] = frozenset(infix_fns)

    # ── Token-stream helpers ─────────────────────────────────────────────────

    def _peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx >= len(self._tokens):
            return self._tokens[-1]  # EOF sentinel
        return self._tokens[idx]

    def _consume(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.kind is not TK.EOF:
            self._pos += 1
        return tok

    def _expect(self, kind: TK) -> Token:
        tok = self._peek()
        if tok.kind is not kind:
            raise ValueError(
                f"Expected {kind.name} but found {tok.text!r} at position {tok.pos}"
                f" in: {self._original!r}"
            )
        return self._consume()

    # ── Formula entry points ─────────────────────────────────────────────────

    def parse_formula(self) -> Formula:
        """
        Parse a complete formula from the current position.
        """
        return self._parse_iff()

    def parse_formula_all(self) -> Formula:
        """Parse a formula, then assert the entire string was consumed."""
        f = self.parse_formula()
        if self._peek().kind is not TK.EOF:
            tok = self._peek()
            raise ValueError(
                f"Unexpected token {tok.text!r} at position {tok.pos}"
                f" in: {self._original!r}"
            )
        return f

    # ── Connective levels (iterative loops — no recursion per operator) ──────

    def _parse_iff(self) -> Formula:
        left = self._parse_impl()
        while self._peek().kind is TK.IFF:
            self._consume()
            left = Equivalent(left, self._parse_impl())
        return left

    def _parse_impl(self) -> Formula:
        left = self._parse_or()
        if self._peek().kind is TK.IMPLIES:
            self._consume()
            return Implies(left, self._parse_impl())   # right-assoc
        return left

    def _parse_or(self) -> Formula:
        left = self._parse_and()
        while self._peek().kind is TK.OR:
            self._consume()
            left = Or(left, self._parse_and())
        return left

    def _parse_and(self) -> Formula:
        left = self._parse_not()
        while self._peek().kind is TK.AND:
            self._consume()
            left = And(left, self._parse_not())
        return left

    def _parse_not(self) -> Formula:
        if self._peek().kind in (TK.FORALL, TK.EXISTS):
            return self._parse_quantifier()
        if self._peek().kind is TK.NOT:
            self._consume()
            return Not(self._parse_not())
        return self._parse_primary()

    # ── Quantifiers ──────────────────────────────────────────────────────────

    def _parse_quantifier(self) -> Formula:
        kw_tok   = self._consume()           # FORALL or EXISTS
        is_forall = kw_tok.kind is TK.FORALL

        qvars = self._parse_qvars()

        guard: Formula | None = None
        if self._peek().kind is TK.PIPE:
            self._consume()
            guard = self.parse_formula()     # greedy; stops at COLON naturally

        self._expect(TK.COLON)
        body = self._parse_not()             # non-greedy: stops before connectives

        cls = Forall if is_forall else Exists
        return cls(qvars, body, guard=guard)

    def _parse_qvars(self) -> list[list[str]]:
        """Parse the variable list of a quantifier.

        Handles both single variables (``x``) and diagonal groups
        (``(x1, x2)``), separated by commas.  Stops when the next token
        after a comma is not a name or ``(``, which means the comma belongs
        to something else (e.g. the guard formula).
        """
        qvars: list[list[str]] = []
        while True:
            if self._peek().kind is TK.LPAR:
                # Diagonal group: (x1, x2, ...)
                self._consume()
                names = [self._expect(TK.NAME).text]
                while self._peek().kind is TK.COMMA:
                    self._consume()
                    names.append(self._expect(TK.NAME).text)
                self._expect(TK.RPAR)
                qvars.append(names)
            else:
                qvars.append([self._expect(TK.NAME).text])

            if self._peek().kind is not TK.COMMA:
                break
            # A comma here could be: another qvar, or a separator inside the
            # guard formula.  Look one step ahead: if the token after the
            # comma is NAME or LPAR, it is another qvar; otherwise stop.
            if self._peek(1).kind not in (TK.NAME, TK.LPAR):
                break
            self._consume()  # consume the comma
        return qvars

    # ── Primary formula ───────────────────────────────────────────────────────

    def _parse_primary(self) -> Formula:
        tok = self._peek()

        # ── Parenthesised formula ─────────────────────────────────────────────
        if tok.kind is TK.LPAR:
            self._consume()
            inner = self.parse_formula()
            self._expect(TK.RPAR)
            return self._apply_formula_selects(inner)

        # ── KB  { clause, … } ────────────────────────────────────────────────
        if tok.kind is TK.LBRACE:
            return self._parse_kb()

        # ── struct_forall / struct_exists ───────────────────────────────────────
        if tok.kind is TK.STRUCT_FORALL:
            self._consume()
            self._expect(TK.LPAR)
            axis = self._expect(TK.NAME).text
            self._expect(TK.COMMA)
            body = self.parse_formula()
            self._expect(TK.RPAR)
            return StructuralForall(body, structural_axis=axis)

        if tok.kind is TK.STRUCT_EXISTS:
            self._consume()
            self._expect(TK.LPAR)
            axis = self._expect(TK.NAME).text
            self._expect(TK.COMMA)
            body = self.parse_formula()
            self._expect(TK.RPAR)
            return StructuralExists(body, structural_axis=axis)

        # ── NAME-headed formulas ──────────────────────────────────────────────
        if tok.kind is TK.NAME:
            return self._parse_name_primary()

        raise ValueError(
            f"Expected formula primary, found {tok.text!r} at position {tok.pos}"
            f" in: {self._original!r}"
        )

    def _parse_name_primary(self) -> Formula:
        name_tok = self._consume()
        name     = name_tok.text

        # name '(' … ')' ('[' axes ']')? ('[' axis '=' int ']')*
        if self._peek().kind is TK.LPAR:
            self._consume()
            args     = self._parse_term_list()
            self._expect(TK.RPAR)
            out_axes = self._parse_optional_axes()
            # Decide term-vs-formula by looking *past* any trailing `[axis=idx]`
            # selects: an infix predicate/function may follow those selects too,
            # e.g. ``digit(x1[p])[p=0] =d d0``. Without this lookahead the
            # selects get greedily consumed as FormulaAxisSelect first and the
            # infix operator is never recognised.
            if self._is_infix_after_selects():
                lhs: Term = _CallTerm(name, args, output_axes=out_axes)
                lhs = self._parse_term_selects(lhs)
                return self._maybe_infix_atomic(lhs)
            # Otherwise it is a regular predicate call → AtomicFormula.
            atom = AtomicFormula(name, *args, structural_axes=out_axes)
            return self._apply_formula_selects(atom)

        # name '[' … ']' → _NameTerm with annotations (TermFormula or infix LHS)
        if self._peek().kind is TK.LBRACK:
            lhs = self._parse_name_bracket_term(name)
            return self._maybe_infix_atomic(lhs)

        # Bare name → _NameTerm (possibly an alphanumeric infix predicate LHS)
        return self._maybe_infix_atomic(_NameTerm(name))

    def _is_token_in(self, names: frozenset[str]) -> bool:
        """Return True if the current token is a NAME or INFIX_OP matching one of *names*."""
        tok = self._peek()
        return (
            (tok.kind is TK.INFIX_OP and tok.text in names)
            or (tok.kind is TK.NAME   and tok.text in names)
        )

    def _is_infix_pred_next(self) -> bool:
        """Return True if the current token is a declared infix predicate."""
        return self._is_token_in(self._infix_preds)

    def _is_infix_fn_next(self) -> bool:
        """Return True if the current token is a declared infix function."""
        return self._is_token_in(self._infix_fns)

    def _is_select_group_at(self, offset: int) -> bool:
        """Return True if a ``[NAME=...]`` select group starts at token *offset*."""
        return (
            self._peek(offset).kind is TK.LBRACK
            and self._peek(offset + 1).kind is TK.NAME
            and self._peek(offset + 2).kind is TK.EQ
        )

    def _skip_select_group(self, offset: int) -> int | None:
        """Return the offset just past one well-formed ``[NAME=int]`` group starting
        at *offset*, or ``None`` if the group is malformed (let the real parser
        raise the error in that case)."""
        offset += 3  # '[' NAME '='
        if self._peek(offset).kind is TK.NOT and self._peek(offset).text == "-":
            offset += 1
        if self._peek(offset).kind is not TK.INT:
            return None
        offset += 1
        if self._peek(offset).kind is not TK.RBRACK:
            return None
        return offset + 1

    def _is_infix_after_selects(self) -> bool:
        """Look past zero or more ``[axis=idx]`` select groups (without consuming
        them) and report whether an infix predicate/function token follows.

        Used to decide whether ``name(args)[sel1][sel2]...`` should be parsed as
        the LHS of an infix atomic formula (a *term*) rather than as a plain
        predicate call (a *formula*).
        """
        offset = 0
        while self._is_select_group_at(offset):
            next_offset = self._skip_select_group(offset)
            if next_offset is None:
                break
            offset = next_offset
        tok = self._peek(offset)
        return (
            (tok.kind is TK.INFIX_OP and (tok.text in self._infix_preds or tok.text in self._infix_fns))
            or (tok.kind is TK.NAME and (tok.text in self._infix_preds or tok.text in self._infix_fns))
        )

    def _parse_term_selects(self, atom: Term) -> Term:
        """Consume zero or more ``[axis=idx]`` selects, wrapping *atom* in
        :class:`TermAxisSelect` for each. Shared by :meth:`_parse_selectable_atom`
        and the infix-LHS path of :meth:`_parse_name_primary`."""
        while self._is_select_group_at(0):
            self._consume()                              # [
            axis = self._expect(TK.NAME).text
            self._expect(TK.EQ)
            idx  = self._parse_signed_int()
            self._expect(TK.RBRACK)
            atom = TermAxisSelect(atom, axis=axis, index=idx)
        return atom

    def _chain_infix_fns(self, lhs: Term) -> Term:
        """Chain zero or more declared infix function operators onto *lhs*."""
        while self._is_infix_fn_next():
            op_tok = self._consume()
            rhs    = self._parse_selectable_atom()
            lhs    = _CallTerm(op_tok.text, [lhs, rhs])
        return lhs

    def _maybe_infix_atomic(self, lhs: Term) -> Formula:
        """If the current token is a declared infix predicate, consume it and
        parse the RHS term to build an AtomicFormula; otherwise wrap lhs in
        a TermFormula.

        Infix function operators (e.g. ``+``) are chained first so that
        ``digit(x1) + digit(x2) = s`` is parsed as
        ``=(+(digit(x1), digit(x2)), s)`` rather than failing with an
        unexpected ``+``.
        """
        lhs = self._chain_infix_fns(lhs)
        if self._is_infix_pred_next():
            op_tok = self._consume()
            rhs    = self._parse_term()
            result = AtomicFormula(op_tok.text, lhs, rhs)
            return self._apply_formula_selects(result)
        return TermFormula(lhs)

    def _apply_formula_selects(self, formula: Formula) -> Formula:
        """Apply zero or more ``[axis=int]`` suffix selects."""
        while (self._peek().kind is TK.LBRACK
               and self._peek(1).kind is TK.NAME
               and self._peek(2).kind is TK.EQ):
            self._consume()                              # [
            axis = self._expect(TK.NAME).text
            self._expect(TK.EQ)
            idx  = self._parse_signed_int()
            self._expect(TK.RBRACK)
            formula = FormulaAxisSelect(formula, axis=axis, index=idx)
        return formula

    def _parse_kb(self) -> KB:
        self._expect(TK.LBRACE)
        clauses: list[Formula] = []
        while True:
            clause = self.parse_formula()
            clauses.append(clause)
            if self._peek().kind is not TK.COMMA:
                break
            self._consume()
        self._expect(TK.RBRACE)
        return KB(clauses, validate=False)   # validated by the pipeline after resolution

    # ── Term productions ──────────────────────────────────────────────────────

    def _parse_term(self) -> Term:
        """Parse a full term (including infix function operators)."""
        atom = self._parse_selectable_atom()
        return self._chain_infix_fns(atom)

    def _parse_selectable_atom(self) -> Term:
        """Parse a term atom then apply any ``[axis=int]`` postfix selects."""
        atom = self._parse_term_atom()
        return self._parse_term_selects(atom)

    def _parse_term_atom(self) -> Term:
        tok = self._peek()

        # '(' formula ')' in term position → TermFormula
        if tok.kind is TK.LPAR:
            self._consume()
            inner = self.parse_formula()
            self._expect(TK.RPAR)
            return TermFormula(inner)

        if tok.kind is TK.NAME:
            name_tok = self._consume()
            name     = name_tok.text

            # func_term:  name '(' args ')' ('[' axes ']')?
            if self._peek().kind is TK.LPAR:
                self._consume()
                args     = self._parse_term_list()
                self._expect(TK.RPAR)
                out_axes = self._parse_optional_axes()
                return _CallTerm(name, args, output_axes=out_axes)

            # var_with_dim / axis_select:  name '[' … ']'
            if self._peek().kind is TK.LBRACK:
                return self._parse_name_bracket_term(name)

            # leaf term
            return _NameTerm(name)

        raise ValueError(
            f"Expected term, found {tok.text!r} at position {tok.pos}"
            f" in: {self._original!r}"
        )

    def _parse_name_bracket_term(self, name: str) -> Term:
        """Parse the ``[…]`` suffix that follows *name*, returning the appropriate term."""
        self._consume()   # consume [
        # [axis=int] → AxisSelect  vs  [axes] → _NameTerm with annotations
        if self._peek().kind is TK.NAME and self._peek(1).kind is TK.EQ:
            axis = self._expect(TK.NAME).text
            self._expect(TK.EQ)
            idx  = self._parse_signed_int()
            self._expect(TK.RBRACK)
            return TermAxisSelect(_NameTerm(name), axis=axis, index=idx)

        # [axes] — one or more axis names separated by commas
        axes = [self._expect(TK.NAME).text]
        while self._peek().kind is TK.COMMA:
            self._consume()
            axes.append(self._expect(TK.NAME).text)
        self._expect(TK.RBRACK)
        return _NameTerm(name, axis_annotations=axes)

    def _parse_term_list(self) -> list[Term]:
        """Parse ``term (',' term)*`` or empty (when next token is RPAR)."""
        if self._peek().kind is TK.RPAR:
            return []
        args = [self._parse_term()]
        while self._peek().kind is TK.COMMA:
            self._consume()
            args.append(self._parse_term())
        return args

    def _parse_optional_axes(self) -> list[str] | None:
        """Parse ``'[' name (',' name)* ']'`` for output-axis annotations.

        Returns ``None`` if the next bracket starts a ``[axis=int]`` select
        (which is handled separately by :meth:`_apply_formula_selects` or
        :meth:`_parse_selectable_atom`).
        """
        if self._peek().kind is not TK.LBRACK:
            return None
        # Distinguish  [name, ...]  (axis list)  from  [name=int]  (select).
        if self._peek(1).kind is TK.NAME and self._peek(2).kind is TK.EQ:
            return None  # leave for the select handler
        self._consume()   # consume [
        axes = [self._expect(TK.NAME).text]
        while self._peek().kind is TK.COMMA:
            self._consume()
            axes.append(self._expect(TK.NAME).text)
        self._expect(TK.RBRACK)
        return axes

    def _parse_signed_int(self) -> int:
        """Parse an optional ``-`` then digits."""
        if self._peek().kind is TK.NOT and self._peek().text == "-":
            self._consume()
            return -int(self._expect(TK.INT).text)
        return int(self._expect(TK.INT).text)

    # ── Term-only entry point ─────────────────────────────────────────────────

    def parse_term_all(self) -> Term:
        """Parse a term, then assert the entire string was consumed."""
        t = self._parse_term()
        if self._peek().kind is not TK.EOF:
            tok = self._peek()
            raise ValueError(
                f"Unexpected token {tok.text!r} at position {tok.pos}"
                f" in: {self._original!r}"
            )
        return t


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------


def parse_formula_string(
    formula_string: str,
    infix_preds: list[str] | None = None,
    infix_fns:   list[str] | None = None,
) -> Formula:
    """Parse *formula_string* into a :class:`~sltn.fol.formula.Formula` AST.

    Grammar
    -------
    - All symbol names: any letter start, mixed case / primes allowed;
      also non-ASCII mathematical symbols (e.g. ``⊩``, ``≤``).
    - Variables/constants are string names until :meth:`~sltn.signature.Signature.parse`
      resolves them.
    - Quantifiers: ∀/forall, ∃/exists
    - Connectives: not, &, or/∨, ->, <->
    - Structural axes: ``P(args)[axes]``
    - struct_forall / struct_exists: ``struct_forall(axis, formula)``
    - KB: ``{ f1, f2, ... }``
    - Infix predicates: when *infix_preds* is provided those names are also
      accepted in infix position, e.g. ``x ⊩ U`` as well as ``⊩(x, U)``.
    """
    all_infix = frozenset(infix_preds or []) | frozenset(infix_fns or [])
    tokens = tokenize(formula_string, infix_ops=all_infix)
    parser = _Parser(
        tokens,
        original=formula_string,
        infix_preds=infix_preds or [],
        infix_fns=infix_fns or [],
    )
    try:
        return parser.parse_formula_all()
    except ValueError as exc:
        raise ValueError(
            f"Failed to parse formula: {formula_string!r}.\n{exc}"
        ) from exc



def parse_term_string(
    term_string: str,
    infix_fns: list[str] | None = None,
    infix_preds: list[str] | None = None,
) -> Term:
    """Parse a term string into a :class:`~sltn.fol.term.Term` AST node.

    *infix_preds* never contributes to the term grammar itself (a term cannot
    contain a top-level Bool-valued infix predicate) but is still fed to the
    tokenizer so that, e.g., a declared infix predicate like ``=d`` is
    tokenised as one ``INFIX_OP`` rather than mis-split into ``=`` and ``d``.
    This only affects error-message accuracy when this function is reached
    as a fallback after formula-parsing failed.
    """
    all_infix = frozenset(infix_fns or []) | frozenset(infix_preds or [])
    tokens = tokenize(term_string, infix_ops=all_infix)
    parser = _Parser(
        tokens,
        original=term_string,
        infix_preds=[],
        infix_fns=infix_fns or [],
    )
    try:
        return parser.parse_term_all()
    except ValueError as exc:
        raise ValueError(
            f"Failed to parse term: {term_string!r}.\n{exc}"
        ) from exc
