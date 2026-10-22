"""Lexer for sLTN formula and term strings.

Built-in operators, punctuation and the Signature's symbolic infix symbols live in
one table matched longest-first, so a user symbol can neither shadow nor be
shadowed by a grammar token (``&&`` is one token, never ``&`` ``&``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Every built-in spelling and the canonical token kind it produces.
BUILTIN_TOKENS: dict[str, str] = {
    "(": "(", ")": ")", "[": "[", "]": "]", "{": "{", "}": "}",
    ",": ",", ":": ":", "|": "|", "=": "=",
    "not": "not", "~": "not", "¬": "not",
    "and": "and", "&": "and", "∧": "and",
    "or": "or", "∨": "or",
    "implies": "implies", "->": "implies", "=>": "implies", "→": "implies", "⇒": "implies",
    "iff": "iff", "<->": "iff", "<=>": "iff", "↔": "iff", "⇔": "iff",
    "forall": "forall", "∀": "forall",
    "exists": "exists", "∃": "exists",
}
UNSUPPORTED_TOKENS = frozenset("⊤⊥")
#: Names a symbol can't take, derived from the lexer table so the two can't drift.
RESERVED_NAMES: frozenset[str] = frozenset(BUILTIN_TOKENS) | UNSUPPORTED_TOKENS

_IDENT = re.compile(r"[^\W\d][\w']*")
_INT = re.compile(r"-?[0-9]+")
_BUILTIN_SYMBOLS = [t for t in BUILTIN_TOKENS if not _IDENT.fullmatch(t)]


def is_identifier(name: str) -> bool:
    return _IDENT.fullmatch(name) is not None


@dataclass(frozen=True)
class Token:
    kind: str   # a BUILTIN_TOKENS value, or NAME / INT / OP (user infix) / EOF
    text: str
    pos: int


def tokenize(s: str, infix_ops: frozenset[str] | set[str] = frozenset()) -> list[Token]:
    """Split *s* into tokens. *infix_ops* are the Signature's infix symbol names."""
    word_ops = {op for op in infix_ops if is_identifier(op)}
    symbols = sorted(_BUILTIN_SYMBOLS + [op for op in infix_ops if op not in word_ops],
                     key=len, reverse=True)
    tokens: list[Token] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        # Signed integers only appear as a selection index: x[t=-1].
        m = _INT.match(s, i)
        if m and (c != "-" or (tokens and tokens[-1].kind == "=")):
            tokens.append(Token("INT", m.group(), i))
            i = m.end()
            continue
        m = _IDENT.match(s, i)
        if m:
            word = m.group()
            kind = BUILTIN_TOKENS.get(word) or ("OP" if word in word_ops else "NAME")
            tokens.append(Token(kind, word, i))
            i = m.end()
            continue
        sym = next((t for t in symbols if s.startswith(t, i)), None)
        if sym is not None:
            tokens.append(Token(BUILTIN_TOKENS.get(sym, "OP"), sym, i))
            i += len(sym)
            continue
        if c in UNSUPPORTED_TOKENS:
            raise ValueError(f"Unsupported operator {c!r} at position {i} in: {s!r}")
        if ord(c) > 127 and not c.isalnum():
            # Undeclared non-ASCII symbol name (e.g. ⊩); the run stops at a letter, digit or operator.
            j = i + 1
            while (j < n and ord(s[j]) > 127 and not s[j].isalnum() and not s[j].isspace()
                   and s[j] not in BUILTIN_TOKENS and s[j] not in UNSUPPORTED_TOKENS):
                j += 1
            tokens.append(Token("NAME", s[i:j], i))
            i = j
            continue
        raise ValueError(f"Unexpected character {c!r} at position {i} in: {s!r}")
    tokens.append(Token("EOF", "", n))
    return tokens
