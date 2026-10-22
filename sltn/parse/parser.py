"""Parser for sLTN formula and term strings.

Parsing has two steps:

1. A precedence-climbing parser builds an untyped expression tree. It never asks
   whether it is reading a term or a formula, and parentheses only group.
2. A classifier turns the tree into the formula/term AST, knowing for every node
   whether it sits in formula position or in term position.

Binding, loosest first:

  ==================  ======================  ===================================
  quantifier prefix   ``∀ x | guard: body``   body parsed at the ``not`` level
  ``<->`` iff ↔ ⇔     non-associative         ``a <-> b <-> c`` is an error
  ``->`` implies → ⇒  right
  ``or`` ∨            right
  ``&`` and ∧         right
  ``not`` ~ ¬         prefix
  infix predicates    non-associative         declared with ``infix=True``
  infix functions     left (one level)        declared with ``infix=True``
  postfix             ``f(args)``, annotation ``[t, u]``, selection ``[t=k]``
  ==================  ======================  ===================================

Quantifiers bind only their immediate subformula: ``forall x: A(x) -> B(x)`` is
``(forall x: A(x)) -> B(x)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sltn.fol.formula import (
    And, Or, Not, Implies, Equivalent, Forall, Exists,
    AtomicFormula, Formula, KB, FormulaAxisSelect, TermFormula,
)
from sltn.fol.term import TermAxisSelect, _CallTerm, _NameTerm, Term
from sltn.parse.tokenizer import Token, tokenize

InfixTable = dict[str, Literal["pred", "fn"]]


# ---------------------------------------------------------------------------
# Untyped expression tree
# ---------------------------------------------------------------------------

@dataclass
class _Name:
    name: str
    axes: list[str] | None = None


@dataclass
class _Call:
    name: str
    args: list[Any]
    axes: list[str] | None = None


@dataclass
class _Select:
    base: Any
    axis: str
    index: int


@dataclass
class _Op:
    op: str          # "not", "and", "or", "implies", "iff", or a user infix symbol
    args: list[Any]


@dataclass
class _Quant:
    kind: str        # "forall" or "exists"
    groups: list[list[str]]
    guard: Any
    body: Any


@dataclass
class _KB:
    items: list[Any]


_IFF, _IMPLIES, _OR, _AND, _NOT, _PRED, _FN = 10, 20, 30, 40, 50, 60, 70
_CONNECTIVE_BP = {"iff": _IFF, "implies": _IMPLIES, "or": _OR, "and": _AND}
_RIGHT_ASSOC = {_IMPLIES, _OR, _AND}
_NON_ASSOC = {_IFF: "'<->' can't be chained; add parentheses",
              _PRED: "infix predicates can't be chained; add parentheses"}


class _Parser:
    def __init__(self, source: str, infix: InfixTable) -> None:
        self._src = source
        self._infix = infix
        self._toks = tokenize(source, frozenset(infix))
        self._i = 0

    # ── token helpers ────────────────────────────────────────────────────────

    def _peek(self, k: int = 0) -> Token:
        return self._toks[min(self._i + k, len(self._toks) - 1)]

    def _at(self, kind: str, k: int = 0) -> bool:
        return self._peek(k).kind == kind

    def _next(self) -> Token:
        tok = self._peek()
        self._i += tok.kind != "EOF"
        return tok

    def _error(self, msg: str, tok: Token | None = None) -> ValueError:
        tok = tok or self._peek()
        return ValueError(f"{msg} at position {tok.pos} in: {self._src!r}")

    def _expect(self, kind: str) -> Token:
        if not self._at(kind):
            tok = self._peek()
            found = repr(tok.text) if tok.kind != "EOF" else "end of input"
            raise self._error(f"Expected {kind!r} but found {found}")
        return self._next()

    # ── expressions ──────────────────────────────────────────────────────────

    def parse_all(self) -> Any:
        tree = self._expr(0)
        if not self._at("EOF"):
            raise self._error(f"Unexpected token {self._peek().text!r}")
        return tree

    def _bp(self, tok: Token) -> int:
        if tok.kind == "OP":
            return _PRED if self._infix[tok.text] == "pred" else _FN
        return _CONNECTIVE_BP.get(tok.kind, 0)

    def _expr(self, min_bp: int) -> Any:
        left = self._prefix()
        while (bp := self._bp(self._peek())) > min_bp:
            tok = self._next()
            right = self._expr(bp - 1 if bp in _RIGHT_ASSOC else bp)
            left = _Op(tok.text if tok.kind == "OP" else tok.kind, [left, right])
            if bp in _NON_ASSOC and self._bp(self._peek()) == bp:
                raise self._error(_NON_ASSOC[bp])
        return left

    def _prefix(self) -> Any:
        tok = self._peek()
        if tok.kind == "not":
            self._next()
            return _Op("not", [self._expr(_NOT)])
        if tok.kind in ("forall", "exists"):
            return self._quantifier()
        if tok.kind == "(":
            self._next()
            inner = self._expr(0)
            self._expect(")")
            return self._selections(inner)
        if tok.kind == "{":
            self._next()
            items = [self._expr(0)]
            while self._at(","):
                self._next()
                items.append(self._expr(0))
            self._expect("}")
            return _KB(items)
        if tok.kind == "NAME" or (tok.kind == "OP" and self._at("(", 1)):
            return self._name()
        found = repr(tok.text) if tok.kind != "EOF" else "end of input"
        raise self._error(f"Expected a formula or term, found {found}")

    def _quantifier(self) -> _Quant:
        kind = self._next().kind
        groups = [self._var_group()]
        while self._at(","):
            self._next()
            groups.append(self._var_group())
        guard = None
        if self._at("|"):
            self._next()
            guard = self._expr(0)
        self._expect(":")
        return _Quant(kind, groups, guard, self._expr(_NOT))

    def _var_group(self) -> list[str]:
        if not self._at("("):
            return [self._expect("NAME").text]
        self._next()
        names = self._names()
        self._expect(")")
        return names

    def _names(self) -> list[str]:
        names = [self._expect("NAME").text]
        while self._at(","):
            self._next()
            names.append(self._expect("NAME").text)
        return names

    def _name(self) -> Any:
        """``name``, ``name(args)``, then an optional annotation ``[t, u]`` and selections."""
        name = self._next().text
        args = None
        if self._at("("):
            self._next()
            args = []
            if not self._at(")"):
                args.append(self._expr(0))
                while self._at(","):
                    self._next()
                    args.append(self._expr(0))
            self._expect(")")
        axes = None
        if self._at("[") and not self._at("=", 2):
            self._next()
            axes = self._names()
            self._expect("]")
        node = _Name(name, axes) if args is None else _Call(name, args, axes)
        return self._selections(node)

    def _selections(self, node: Any) -> Any:
        while self._at("[") and self._at("=", 2):
            self._next()
            axis = self._expect("NAME").text
            self._next()
            index = int(self._expect("INT").text)
            self._expect("]")
            node = _Select(node, axis, index)
        return node


# ---------------------------------------------------------------------------
# Classification: untyped tree -> Formula / Term AST
# ---------------------------------------------------------------------------

_CONNECTIVES = {"and": And, "or": Or, "implies": Implies, "iff": Equivalent}


class _Classifier:
    """*functions* are the names whose application is a term (declared functions
    and term-valued definitions); every other application is an atomic formula."""

    def __init__(self, infix: InfixTable, functions: set[str]) -> None:
        self._infix = infix
        self._functions = functions

    def formula(self, n: Any) -> Formula:
        match n:
            case _Name():
                return TermFormula(_NameTerm(n.name, n.axes))
            case _Call() if n.name in self._functions:
                return TermFormula(self.term(n))
            case _Call():
                return AtomicFormula(n.name, *map(self.term, n.args), structural_axes=n.axes)
            case _Select():
                base = self.formula(n.base)
                if isinstance(base, TermFormula):
                    return TermFormula(TermAxisSelect(base.term, axis=n.axis, index=n.index))
                return FormulaAxisSelect(base, axis=n.axis, index=n.index)
            case _Op(op="not"):
                return Not(self.formula(n.args[0]))
            case _Op() if n.op in _CONNECTIVES:
                return _CONNECTIVES[n.op](self.formula(n.args[0]), self.formula(n.args[1]))
            case _Op() if self._infix[n.op] == "pred":
                return AtomicFormula(n.op, *map(self.term, n.args))
            case _Op():
                return TermFormula(self.term(n))
            case _Quant():
                cls = Forall if n.kind == "forall" else Exists
                guard = self.formula(n.guard) if n.guard is not None else None
                return cls(n.groups, self.formula(n.body), guard=guard)
            case _KB():
                return KB([self.formula(i) for i in n.items], validate=False)
        raise TypeError(f"Unexpected parse node {n!r}")

    def term(self, n: Any) -> Term | Formula:
        """A formula in term position is kept as a Bool-valued argument (``TermFormula``)."""
        match n:
            case _Name():
                return _NameTerm(n.name, n.axes)
            case _Call():
                return _CallTerm(n.name, [self.term(a) for a in n.args], output_axes=n.axes)
            case _Select():
                return TermAxisSelect(self.term(n.base), axis=n.axis, index=n.index)
            case _Op() if n.op in self._infix and self._infix[n.op] == "fn":
                return _CallTerm(n.op, [self.term(a) for a in n.args])
        return TermFormula(self.formula(n))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def signature_tables(sig: Any) -> tuple[InfixTable, set[str]]:
    """Return the infix table and the term-valued names of *sig*.

    A definition is term-valued when its ``output_sort`` is set and is not Bool.
    """
    infix: InfixTable = {n: "pred" for n, s in sig._predicates.items() if s.infix}
    infix |= {n: "fn" for n, s in sig._functions.items() if s.infix}
    functions = set(sig._functions)
    for name, d in sig._definitions.items():
        is_fn = d.output_sort is not None and d.output_sort.name != "Bool"
        if is_fn:
            functions.add(name)
        if d.infix:
            infix[name] = "fn" if is_fn else "pred"
    return infix, functions


def parse_string(
    source: str,
    infix: InfixTable | None = None,
    functions: set[str] | None = None,
    expect: Literal["formula", "term"] = "formula",
) -> Formula | Term:
    """Parse *source* as a formula or a term (names are left unresolved)."""
    infix = infix or {}
    try:
        tree = _Parser(source, infix).parse_all()
        classifier = _Classifier(infix, functions or set())
        if expect == "formula":
            return classifier.formula(tree)
        result = classifier.term(tree)
        if isinstance(result, Formula):
            what = f"'{tree.op}'" if isinstance(tree, _Op) else "this expression"
            raise ValueError(f"Expected a term, but {what} builds a formula")
        return result
    except ValueError as exc:
        raise ValueError(f"Failed to parse {expect}: {source!r}.\n{exc}") from exc


def _infix_table(infix_preds: list[str] | None, infix_fns: list[str] | None) -> InfixTable:
    return {**{op: "pred" for op in infix_preds or ()}, **{op: "fn" for op in infix_fns or ()}}


def parse_formula_string(
    formula_string: str,
    infix_preds: list[str] | None = None,
    infix_fns: list[str] | None = None,
) -> Formula:
    """Parse *formula_string* into an unresolved Formula AST.

    Without a Signature, only the declared infix symbols are known; use
    :meth:`Signature.parse` to also resolve names.
    """
    return parse_string(formula_string, _infix_table(infix_preds, infix_fns),
                        set(infix_fns or ()), "formula")


def parse_term_string(
    term_string: str,
    infix_fns: list[str] | None = None,
    infix_preds: list[str] | None = None,
) -> Term:
    """Parse *term_string* into an unresolved Term AST."""
    return parse_string(term_string, _infix_table(infix_preds, infix_fns),
                        set(infix_fns or ()), "term")
