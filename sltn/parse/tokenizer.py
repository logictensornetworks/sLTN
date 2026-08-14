"""Single-pass lexer for formula and term strings.

:func:`tokenize` converts a raw string into a flat ``list[Token]`` in O(n) time.
The token kinds cover all constructs of the sLTN grammar plus a slot for
user-declared infix predicates and functions (``TK.INFIX_OP``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class TK(Enum):
    NAME          = auto()   # identifier
    INT           = auto()   # unsigned digit sequence
    LPAR          = auto()   # (
    RPAR          = auto()   # )
    LBRACK        = auto()   # [
    RBRACK        = auto()   # ]
    LBRACE        = auto()   # {
    RBRACE        = auto()   # }
    COMMA         = auto()   # ,
    COLON         = auto()   # :
    PIPE          = auto()   # |
    EQ            = auto()   # = (axis selector)
    NOT           = auto()   # not / ~ / ¬ / - (unary minus)
    AND           = auto()   # & / and / ∧
    OR            = auto()   # or / ∨
    IMPLIES       = auto()   # -> / → / implies / =>
    IFF           = auto()   # <-> / ↔ / iff / <=>
    FORALL        = auto()   # forall / ∀
    EXISTS        = auto()   # exists / ∃
    STRUCT_FORALL = auto()   # struct_forall
    STRUCT_EXISTS = auto()   # struct_exists
    INFIX_OP      = auto()   # user-declared symbolic infix operator
    EOF           = auto()


@dataclass(frozen=True)
class Token:
    kind: TK
    text: str
    pos:  int


# ---------------------------------------------------------------------------
# Keyword table  (word-bounded identifiers)
# ---------------------------------------------------------------------------

_WORD_KEYWORDS: dict[str, TK] = {
    "forall":       TK.FORALL,
    "exists":       TK.EXISTS,
    "struct_forall": TK.STRUCT_FORALL,
    "struct_exists": TK.STRUCT_EXISTS,
    "not":          TK.NOT,
    "and":          TK.AND,
    "or":           TK.OR,
    "implies":      TK.IMPLIES,
    "iff":          TK.IFF,
}

# Unicode chars used as dedicated logic operators — excluded from NAME sequences.
_LOGIC_UNICODE: frozenset[str] = frozenset("¬→↔∀∃∨∧")

_ID_START = re.compile(r"[^\W\d]", re.UNICODE)
_ID_CONT  = re.compile(r"[\w']",   re.UNICODE)


def tokenize(
    s: str,
    infix_ops: frozenset[str] = frozenset(),
) -> list[Token]:
    """Tokenise *s* into a ``list[Token]``.
    *infix_ops* contains user-declared infix predicate and function names.
    """
    tokens: list[Token] = []
    i = 0
    n = len(s)

    # Symbolic infix ops sorted longest-first for maximal munch
    symbolic_infix: list[str] = sorted(
        (op for op in infix_ops if op and not _ID_START.match(op[0])),
        key=len,
        reverse=True,
    )

    while i < n:
        c = s[i]

        # ── Whitespace ───────────────────────────────────────────────────────
        if c.isspace():
            i += 1
            continue

        # ── Multi-character operators (checked before single-char) ───────────

        # IFF <=> <-> ↔
        if s.startswith("<->", i):
            tokens.append(Token(TK.IFF,     "<->", i)); i += 3; continue
        if s.startswith("<=>", i):
            tokens.append(Token(TK.IFF,     "<=>", i)); i += 3; continue
        if c == "↔": tokens.append(Token(TK.IFF,     c, i)); i += 1; continue
        
        # IMPLIES -> => →
        if s.startswith("->",  i):
            tokens.append(Token(TK.IMPLIES, "->",  i)); i += 2; continue
        if s.startswith("=>",  i):
            tokens.append(Token(TK.IMPLIES, "=>",  i)); i += 2; continue
        if c == "→": tokens.append(Token(TK.IMPLIES, c, i)); i += 1; continue

        # AND & ∧
        if c == "&":  tokens.append(Token(TK.AND,    c, i)); i += 1; continue
        if c == "∧": tokens.append(Token(TK.AND,     c, i)); i += 1; continue

        # OR ∨
        if c == "∨": tokens.append(Token(TK.OR,      c, i)); i += 1; continue

        # NOT ~ ¬
        if c == "~": tokens.append(Token(TK.NOT,     c, i)); i += 1; continue
        if c == "¬": tokens.append(Token(TK.NOT,     c, i)); i += 1; continue
        
        # Quantifiers ∀ ∃
        if c == "∀": tokens.append(Token(TK.FORALL,  c, i)); i += 1; continue
        if c == "∃": tokens.append(Token(TK.EXISTS,  c, i)); i += 1; continue
        
        
        # ── Symbolic user-declared infix operators ────────────────────────────
        # Checked before single-char "=" so that "=f" beats a bare "=".
        infix_matched = False
        for op in symbolic_infix:
            if s.startswith(op, i):
                tokens.append(Token(TK.INFIX_OP, op, i))
                i += len(op)
                infix_matched = True
                break
        if infix_matched:
            continue

        # ── Single-char punctuation ───────────────────────────────────────────
        if c == "(":  tokens.append(Token(TK.LPAR,   c, i)); i += 1; continue
        if c == ")":  tokens.append(Token(TK.RPAR,   c, i)); i += 1; continue
        if c == "[":  tokens.append(Token(TK.LBRACK, c, i)); i += 1; continue
        if c == "]":  tokens.append(Token(TK.RBRACK, c, i)); i += 1; continue
        if c == "{":  tokens.append(Token(TK.LBRACE, c, i)); i += 1; continue
        if c == "}":  tokens.append(Token(TK.RBRACE, c, i)); i += 1; continue
        if c == ",":  tokens.append(Token(TK.COMMA,  c, i)); i += 1; continue
        if c == ":":  tokens.append(Token(TK.COLON,  c, i)); i += 1; continue
        if c == "|":  tokens.append(Token(TK.PIPE,   c, i)); i += 1; continue
        if c == "=":  tokens.append(Token(TK.EQ,     c, i)); i += 1; continue

        # ── Digit sequence → integer literal ─────────────────────────────────
        if c.isdigit():
            j = i + 1
            while j < n and s[j].isdigit():
                j += 1
            tokens.append(Token(TK.INT, s[i:j], i))
            i = j
            continue

        # ── Standard Unicode identifier or keyword ────────────────────────────
        if _ID_START.match(c):
            j = i + 1
            while j < n and _ID_CONT.match(s[j]):
                j += 1
            word = s[i:j]
            kw = _WORD_KEYWORDS.get(word)
            tokens.append(Token(kw if kw is not None else TK.NAME, word, i))
            i = j
            continue

        # ── Non-ASCII symbolic name  (e.g. ⊩, ≤, ⊆ used as identifiers) ─────
        if ord(c) > 127 and c not in _LOGIC_UNICODE:
            j = i + 1
            while (j < n
                   and ord(s[j]) > 127
                   and s[j] not in _LOGIC_UNICODE
                   and not s[j].isspace()):
                j += 1
            tokens.append(Token(TK.NAME, s[i:j], i))
            i = j
            continue

        raise ValueError(
            f"Unexpected character {c!r} at position {i} in: {s!r}"
        )

    tokens.append(Token(TK.EOF, "", n))
    return tokens
