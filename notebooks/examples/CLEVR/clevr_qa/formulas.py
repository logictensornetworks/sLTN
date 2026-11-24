"""The signature, and the compiler from a CLEVR program to its family's sLTN formulas.

Every per-question value is an ordinary variable with one value per question (vocabulary.py lists them),
and a family's rule quantifies them together, `forall (scene, c0, z0, ..., r0, yes): ...`, so question i's
scene meets only question i's values. Nothing indexes a question.

  yes/no question:  forall (scene, c0, ..., r0, yes): (yes <-> body)
  query question:   forall (scene, c0, ..., r0, ans_c): exists o | Present(scene[o]): (... & C(scene[o], ans_c))

With present="conjunct" (the slot model, train_slots.py) every object quantifier is written
`exists o: (Present(scene[o]) & ...)` instead: a guard is a hard mask (an entry counts iff its guard is
above 0), so a learned Present in a guard would let every slot through and get no gradient.

Counting questions (compile_question(counting=True)): a count is a term, the vector over n = 0..MAX_COUNT of
the truth of "exactly n slots are in the set" (counting.py). The question's formula, as it reads:
  how many:       forall (scene, c0, ..., n): EqCount(count o1: (Present(scene[o1]) & ...), n)
  more A than B:  forall (scene, c0, ..., yes): (yes <-> MoreThan(count o1: A(o1), count o2: B(o2)))
sLTN has no `count` term, so the formula is evaluated in two steps: each count's set, an ordinary formula
open in its slot variable (`Count.formula`: one truth value per question and slot), then counting.py turns
those into the count, grounded as the variable cnt0 (cnt1), and the rule is over the counts:
  forall (cnt0, n): EqCount(cnt0, n)        forall (cnt0, cnt1, yes): (yes <-> MoreThan(cnt0, cnt1))
`Compiled.text` gives the formula as it reads, with the counts written in.

The reconstruction axiom (make_signature(recon=True)): the slots explain the image,
  forall (img, rec): forall p: Eq(rec[p], img[p])          p ranges over the pixels (dimension Pix)

Syntax only: nothing here grounds a symbol.
"""

from dataclasses import dataclass

from sltn import Signature

from .programs import canonical
from .vocabulary import (ANS, ATTRS, COUNT_VARS, KVAR, MAX_COUNT, MAX_OBJVARS, MAX_RELS, MAX_FILTERS, PARAM, PRED,
                         SAME, SORT)


@dataclass
class Count:
    """One count of a counting question: the set it counts, open in the slot variable `var`."""
    var: str                  # e.g. "o1"
    body: str                 # the set, free in var: (Present(scene[o1]) & C(scene[o1], c0) & ...)
    formula: str              # what sLTN evaluates: forall (scene, c0, ...): body, one value per question and slot


@dataclass
class Compiled:
    """A family's formulas. `rule` is the knowledge (what training maximises); `predict` is the same
    body with the answer left open, to read off what the formula answers."""
    key: str                  # the canonical skeleton
    kind: str                 # "yesno" or the attribute a query asks for
    variables: list           # the per-question variables, in the order of the quantifier
    rule: str                 # forall (variables): (yes <-> body)   or   forall (variables): body(ans_<a>)
    predict: str              # forall (variables without the answer): body   or   ... body(k<a>)
    n_filters: int
    n_rels: int
    counts: tuple = ()        # counting questions: the Count of cnt0 (and of cnt1)
    text: str = ""            # the rule as it reads (for a counting question, with the counts written in)

    @property
    def answer(self):
        """The variable that holds the question's answer."""
        return "yes" if self.kind == "yesno" else "n" if self.kind == "count" else ANS[self.kind]


RECON_AXIOM = "forall (img, rec): forall p: Eq(rec[p], img[p])"
# two different slots that are both present are not at the same place; spot[o] = where slot o's attention
# peaks (train_slots.perceive), read by Near only
DUP_AXIOM = ("forall (scene, spot): forall o1: forall o2: "
             "(Neq(o1, o2) -> not (Present(scene[o1]) & Present(scene[o2]) & Near(spot[o1], spot[o2])))")


def make_signature(recon=False):
    sig = Signature("clevr_qa")
    sig.sort("Object")
    sig.sort("Dir")
    for a in ATTRS:
        sig.sort(SORT[a])
    sig.dimension("Obj")
    for i in range(1, MAX_OBJVARS + 1):
        sig.structural_variable(f"o{i}", "Obj")
    sig.structural_relation("Neq", ["Obj", "Obj"])
    sig.variable("scene", "Object", dims=["Obj"])
    sig.variable("yes", "Bool")
    for k in range(MAX_RELS):
        sig.variable(f"r{k}", "Dir")
    sig.predicate("Present", ["Object"])
    sig.predicate("R", ["Object", "Object", "Dir"])
    for a in ATTRS:
        sig.predicate(PRED[a], ["Object", SORT[a]])
        sig.predicate(SAME[a], ["Object", "Object"])
        sig.variable(ANS[a], SORT[a])
        sig.variable(KVAR[a], SORT[a])
        for k in range(MAX_FILTERS):
            sig.variable(f"{PARAM[a]}{k}", SORT[a])
    if recon:
        sig.sort("Rgb")
        sig.dimension("Pix")
        sig.structural_variable("p", "Pix")
        sig.variable("img", "Rgb", dims=["Pix"])            # the image, one colour per pixel
        sig.variable("rec", "Rgb", dims=["Pix"])            # the slot model's reconstruction of it
        sig.predicate("Eq", ["Rgb", "Rgb"])
    if recon:
        sig.sort("Place")
        sig.variable("spot", "Place", dims=["Obj"])         # each slot's attention peak (x, y)
        sig.predicate("Near", ["Place", "Place"])
    sig.sort("Num")                                         # a count (counting.py), a vector over 0..MAX_COUNT
    for name in COUNT_VARS + ["n", "kn"]:
        sig.variable(name, "Num")
    sig.predicate("EqCount", ["Num", "Num"])
    sig.predicate("MoreThan", ["Num", "Num"])
    return sig


def compile_question(q, present="guard"):
    """present: "guard" (Present is known, a 0/1 bit) or "conjunct" (Present is learned)."""
    key, fil, rels = canonical(q)                              # the family's key; fil, rels only to check counts
    b = _Builder(q["program"], present)
    kind, body = b.top(lambda a: "n" if a == "count" else ANS[a])   # the program as a formula over scene, c0, ..., r0;
                                                               #   kind: "yesno" or the asked attribute ("color")
    assert b.fil == len(fil) and b.rel == len(rels), (key, b.fil, len(fil), b.rel, len(rels))   # same numbering
    params = (["scene"] + [f"{PARAM[a]}{k}" for k in range(b.fil) for a in ATTRS]   # scene, c0, z0, m0, s0, c1,
              + [f"r{k}" for k in range(b.rel)])                                    #   ..., r0, r1, ...
    if b.counts:                                               # a counting question: the rule is over the counts
        cnts = COUNT_VARS[:len(b.counts)]
        counts = tuple(Count(v, text, f"forall ({', '.join(['scene'] + vs)}): {text}") for v, text, vs in b.counts)
        if kind == "count":
            rule = f"forall ({', '.join(cnts)}, n): {body}"               # body = EqCount(cnt0, n)
            predict = f"forall ({', '.join(cnts)}): {body.replace(', n)', ', kn)')}"   # one value per count
            reads = body
        else:
            rule = f"forall ({', '.join(cnts)}, yes): (yes <-> {body})"
            predict = f"forall ({', '.join(cnts)}): {body}"
            reads = f"(yes <-> {body})"
        for name, c in zip(cnts, counts):
            reads = reads.replace(name, f"(count {c.var}: {c.body})")
        text = f"forall ({', '.join(params + ['n' if kind == 'count' else 'yes'])}): {reads}"
        return Compiled(key, kind, params, rule, predict, b.fil, b.rel, counts, text)
    if kind == "yesno":
        rule = f"forall ({', '.join(params + ['yes'])}): (yes <-> {body})"   # body is true iff the answer is yes
        predict = f"forall ({', '.join(params)}): {body}"                    # reply: the body's truth value
    else:
        rule = f"forall ({', '.join(params + [ANS[kind]])}): {body}"        # the asked object has the answer's
                                                                             #   value: ... & C(scene[o], ans_c)
        _, open_body = _Builder(q["program"], present).top(lambda a: KVAR[a])        # the same body, with kc (every
                                                                             #   colour) in place of ans_c
        predict = f"forall ({', '.join(params)}): {open_body}"              # reply: one truth value per colour,
                                                                             #   the highest is the answer
    return Compiled(key, kind, params, rule, predict, b.fil, b.rel)


class _Builder:
    """Formula text from a CLEVR program. Filter slots and relates are numbered in the order `canonical` collects
    the question's parameters (post-order), so filter slot k of the formula is filter slot k of every
    question."""

    def __init__(self, program, present="guard"):
        assert present in ("guard", "conjunct"), present
        self.p = program
        self.present = present
        self.fil = 0
        self.rel = 0
        self.counts = []                                              # (var, set, its variables) per count
        uniques = [i for i, n in enumerate(program) if n["function"] == "unique"]
        self.obj = {i: f"o{k}" for k, i in enumerate(uniques, 1)}     # the k-th `unique` is object o<k>
        self.next = len(uniques) + 1                                  # for the object `exist` looks for

    def fresh(self):
        self.next += 1
        return f"o{self.next - 1}"

    def filter_atoms(self, v, k):
        return " & ".join(f"{PRED[a]}(scene[{v}], {PARAM[a]}{k})" for a in ATTRS)

    def set_(self, i, v):
        """Formula, free in object variable v, true iff object v is in the set computed by node i."""
        j = i
        while self.p[j]["function"].startswith("filter_"):         # a chain of filters is one filter slot
            j = self.p[j]["inputs"][0]
        inner = self.core(j, v)
        k = self.fil
        self.fil += 1
        atoms = self.filter_atoms(v, k)
        return f"({inner} & {atoms})" if inner else f"({atoms})"

    def exists(self, v, body):
        if self.present == "conjunct":
            return f"(exists {v}: (Present(scene[{v}]) & {body}))"
        return f"(exists {v} | Present(scene[{v}]): {body})"

    def core(self, i, v):
        n = self.p[i]
        f = n["function"]
        if f == "scene":
            return None                                             # every object: the guard already says it
        if f in ("intersect", "union"):
            a = self.set_(n["inputs"][0], v)
            b = self.set_(n["inputs"][1], v)
            return f"({a} {'&' if f == 'intersect' else 'or'} {b})"
        if f == "relate":
            ref = self.obj[n["inputs"][0]]                         # the object the set is defined from
            ref_set = self.unique(n["inputs"][0])
            k = self.rel
            self.rel += 1
            return self.exists(ref, f"({ref_set} & R(scene[{ref}], scene[{v}], r{k}))")
        if f.startswith("same_"):
            a = f[5:]
            ref = self.obj[n["inputs"][0]]
            ref_set = self.unique(n["inputs"][0])
            return self.exists(ref, f"({ref_set} & Neq({ref}, {v}) & {SAME[a]}(scene[{ref}], scene[{v}]))")
        raise ValueError(f"unexpected set function {f}")

    def count(self, i):
        """Count node i: the set it counts, as a Count's (var, body, variables), stored in self.counts. The term
        it becomes in the rule is cnt0, cnt1, ... in order."""
        n = self.p[i]
        assert n["function"] == "count", n["function"]
        v = self.fresh()
        f0, r0 = self.fil, self.rel
        s = self.set_(n["inputs"][0], v)
        vs = [f"{PARAM[a]}{k}" for k in range(f0, self.fil) for a in ATTRS] + [f"r{k}" for k in range(r0, self.rel)]
        self.counts.append((v, f"(Present(scene[{v}]) & {s})", vs))
        return COUNT_VARS[len(self.counts) - 1]

    def unique(self, i):
        n = self.p[i]
        assert n["function"] == "unique", n["function"]
        return self.set_(n["inputs"][0], self.obj[i])

    def top(self, answer):
        """(kind, body). For a query, `answer` is the term the asked attribute is compared with."""
        n = self.p[-1]
        f = n["function"]
        if f == "exist":
            v = self.fresh()
            return "yesno", self.exists(v, self.set_(n["inputs"][0], v))
        if f == "count":
            return "count", f"EqCount({self.count(len(self.p) - 1)}, {answer('count')})"
        if f in ("equal_integer", "less_than", "greater_than"):
            a, b = (self.count(j) for j in n["inputs"])
            body = {"equal_integer": f"EqCount({a}, {b})", "greater_than": f"MoreThan({a}, {b})",
                    "less_than": f"MoreThan({b}, {a})"}[f]
            return "yesno", body
        if f.startswith("query_"):
            a = f[6:]
            v = self.obj[n["inputs"][0]]
            s = self.unique(n["inputs"][0])
            return a, self.exists(v, f"({s} & {PRED[a]}(scene[{v}], {answer(a)}))")
        if f.startswith("equal_"):
            a = f[6:]
            q1, q2 = (self.p[j] for j in n["inputs"])
            assert q1["function"] == q2["function"] == f"query_{a}"
            v1, v2 = self.obj[q1["inputs"][0]], self.obj[q2["inputs"][0]]
            s1, s2 = self.unique(q1["inputs"][0]), self.unique(q2["inputs"][0])
            return "yesno", self.exists(v1, f"({s1} & " + self.exists(v2, f"({s2} & {SAME[a]}(scene[{v1}], scene[{v2}]))") + ")")
        raise ValueError(f"unexpected top function {f}")
