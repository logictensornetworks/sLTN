"""CLEVR as sLTN with named variables: every question is one individual of a diagonal quantifier.

Layout of a batch (all the questions of one family, i.e. one formula):
  x            the question's scene: one row per object position, padded to O = 10
               x[o] = [present, 3D coordinates (3), colour (8), size (2), material (2), shape (3)]
  c<k>, z<k>, m<k>, s<k>   the k-th filter slot's colour, size, material and shape: one-hot, or the all-ones
               vector when the question doesn't name that attribute (the atom is then exactly 1)
  r<k>         the direction of the k-th relate: one-hot over left, right, front, behind
  yes          the answer of a yes/no question (sort Bool, 1 or 0)
  ans_c, ans_z, ans_m, ans_s   the answer of a query question, one-hot

Every one of them is an ordinary variable with one value per question, and a family's rule quantifies them
together, `forall (x, c0, z0, ..., r0, yes): ...`, so question i's scene meets only question i's values.
Nothing indexes a question: there is no Question axis and no `[q]`.

Predicates (perfect sight: they read the slices of x; learning replaces C, Z, M, S with networks)
  Present(x[o])          the presence bit: 0 for an empty position. Every object quantifier is guarded by it
  C, Z, M, S(x[o], v)    the object has colour / size / material / shape v (a dot product with v)
  SameC, SameZ, SameM, SameS(x[o], x[o1])   the two objects have the same colour / size / ...
  R(x[o], x[o1], r)      o1 is in direction r of o, from the 3D coordinates and CLEVR's camera directions
  Neq(o, o1)             structural relation: o and o1 are different positions ("another object")

The programs are canonicalised by `clevr_sltn.canonical`, unchanged: questions with the same skeleton share
one formula, and their filter values and directions become the variables above. Counting questions are out
of scope.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

import sltn
from sltn import Interpretation, Logic, Signature
from sltn.interpretation.operators.aggregator import AggregMax, AggregPMeanError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clevr_sltn import (ATTRS, DIRS, O, VALUES, canonical, in_scope,  # noqa: E402,F401
                        question_params)

PRED = {"color": "C", "size": "Z", "material": "M", "shape": "S"}       # C(x[o], c): o has colour c
SAME = {a: f"Same{p}" for a, p in PRED.items()}                         # SameC(x[o], x[o1])
SORT = {"color": "Col", "size": "Siz", "material": "Mat", "shape": "Shp"}
PARAM = {"color": "c", "size": "z", "material": "m", "shape": "s"}      # filter slot k: c<k>, z<k>, m<k>, s<k>
ANS = {a: f"ans_{p}" for a, p in PARAM.items()}                         # ans_c, ans_z, ans_m, ans_s
KVAR = {a: f"k{p}" for a, p in PARAM.items()}                           # kc: every colour (for predictions)
MAX_SLOTS, MAX_RELS, MAX_OBJVARS = 12, 6, 12

# x[o]: presence bit, 3D coordinates, then the attributes one-hot in ATTRS order
PRESENT = 0
XYZ = slice(1, 4)
OFFSET = {}
_o = 4
for _a in ATTRS:
    OFFSET[_a] = slice(_o, _o + len(VALUES[_a]))
    _o += len(VALUES[_a])
FEAT = _o                                                   # 19


# ----------------------------------------------------------------------------- data

def camera_directions(scenes):
    """[4, 3]: CLEVR's unit vectors for left, right, front, behind (the same camera in every scene)."""
    d = torch.tensor([scenes[0]["directions"][k] for k in DIRS])
    for s in scenes:
        assert torch.allclose(torch.tensor([s["directions"][k] for k in DIRS]), d), s["image_index"]
    return d


def scene_rows(scene):
    """x for one scene, [O, FEAT]: presence, 3D coordinates and one-hot attributes; zeros past the last object."""
    x = torch.zeros(O, FEAT)
    for i, ob in enumerate(scene["objects"]):
        x[i, PRESENT] = 1
        x[i, XYZ] = torch.tensor(ob["3d_coords"])
        for a in ATTRS:
            x[i, OFFSET[a].start + VALUES[a].index(ob[a])] = 1
    return x


# ----------------------------------------------------------------------------- compiler

@dataclass
class Compiled:
    """A family's formulas. `rule` is the knowledge (what training would maximise); `predict` is the same
    body with the answer left open, to read off what the formula answers."""
    key: str                  # the canonical skeleton
    kind: str                 # "yesno" or the attribute a query asks for
    variables: list           # the per-question variables, in the order of the quantifier
    rule: str                 # forall (variables): (yes <-> body)   or   forall (variables): body(ans_<a>)
    predict: str              # forall (variables without the answer): body   or   ... body(k<a>)
    n_slots: int
    n_rels: int


class _Builder:
    """Formula text from a CLEVR program. Slots and relates are numbered in the order `canonical` collects
    the question's parameters (post-order), so slot k of the formula is slot k of every question."""

    def __init__(self, program):
        self.p = program
        self.slot = 0
        self.rel = 0
        uniques = [i for i, n in enumerate(program) if n["function"] == "unique"]
        self.obj = {i: f"o{k}" for k, i in enumerate(uniques, 1)}     # the k-th `unique` is object o<k>
        self.next = len(uniques) + 1                                  # for the object `exist` looks for

    def fresh(self):
        self.next += 1
        return f"o{self.next - 1}"

    def filter_atoms(self, v, k):
        return " & ".join(f"{PRED[a]}(x[{v}], {PARAM[a]}{k})" for a in ATTRS)

    def set_(self, i, v):
        """Formula, free in object variable v, true iff object v is in the set computed by node i."""
        j = i
        while self.p[j]["function"].startswith("filter_"):         # a chain of filters is one slot
            j = self.p[j]["inputs"][0]
        inner = self.core(j, v)
        k = self.slot
        self.slot += 1
        atoms = self.filter_atoms(v, k)
        return f"({inner} & {atoms})" if inner else f"({atoms})"

    def exists(self, v, body):
        return f"(exists {v} | Present(x[{v}]): {body})"

    def core(self, i, v):
        n = self.p[i]
        f = n["function"]
        if f == "scene":
            return None                                             # every object: the guard already says it
        if f in ("intersect", "union"):
            a = self.set_(n["inputs"][0], v)
            b = self.set_(n["inputs"][1], v)
            return f"({a} {'&' if f == 'intersect' else '|'} {b})"
        if f == "relate":
            ref = self.obj[n["inputs"][0]]                         # the object the set is defined from
            ref_set = self.unique(n["inputs"][0])
            k = self.rel
            self.rel += 1
            return self.exists(ref, f"({ref_set} & R(x[{ref}], x[{v}], r{k}))")
        if f.startswith("same_"):
            a = f[5:]
            ref = self.obj[n["inputs"][0]]
            ref_set = self.unique(n["inputs"][0])
            return self.exists(ref, f"({ref_set} & Neq({ref}, {v}) & {SAME[a]}(x[{ref}], x[{v}]))")
        raise ValueError(f"unexpected set function {f}")

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
        if f.startswith("query_"):
            a = f[6:]
            v = self.obj[n["inputs"][0]]
            s = self.unique(n["inputs"][0])
            return a, self.exists(v, f"({s} & {PRED[a]}(x[{v}], {answer(a)}))")
        if f.startswith("equal_"):
            a = f[6:]
            q1, q2 = (self.p[j] for j in n["inputs"])
            assert q1["function"] == q2["function"] == f"query_{a}"
            v1, v2 = self.obj[q1["inputs"][0]], self.obj[q2["inputs"][0]]
            s1, s2 = self.unique(q1["inputs"][0]), self.unique(q2["inputs"][0])
            return "yesno", self.exists(v1, f"({s1} & " + self.exists(v2, f"({s2} & {SAME[a]}(x[{v1}], x[{v2}]))") + ")")
        raise ValueError(f"unexpected top function {f}")


def compile_question(q):
    key, fil, rels = canonical(q)
    b = _Builder(q["program"])
    kind, body = b.top(lambda a: ANS[a])
    assert b.slot == len(fil) and b.rel == len(rels), (key, b.slot, len(fil), b.rel, len(rels))
    params = ["x"] + [f"{PARAM[a]}{k}" for k in range(b.slot) for a in ATTRS] + [f"r{k}" for k in range(b.rel)]
    if kind == "yesno":
        rule = f"forall ({', '.join(params + ['yes'])}): (yes <-> {body})"
        predict = f"forall ({', '.join(params)}): {body}"
    else:
        rule = f"forall ({', '.join(params + [ANS[kind]])}): {body}"
        _, open_body = _Builder(q["program"]).top(lambda a: KVAR[a])
        predict = f"forall ({', '.join(params)}): {open_body}"
    return Compiled(key, kind, params, rule, predict, b.slot, b.rel)


# ----------------------------------------------------------------------------- signature and grounding

def make_signature():
    sig = Signature("clevr_named")
    sig.sort("Object")
    sig.sort("Dir")
    for a in ATTRS:
        sig.sort(SORT[a])
    sig.dimension("Obj")
    for i in range(1, MAX_OBJVARS + 1):
        sig.structural_variable(f"o{i}", "Obj")
    sig.structural_relation("Neq", ["Obj", "Obj"])
    sig.variable("x", "Object", dims=["Obj"])
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
        for k in range(MAX_SLOTS):
            sig.variable(f"{PARAM[a]}{k}", SORT[a])
    return sig


def make_interp(sig, directions, logic=None):
    """Perfect sight: every predicate reads x. `directions` is camera_directions(scenes)."""
    if logic is None:
        logic = Logic().with_defaults()
        logic["exists,Obj"] = AggregMax()
        logic["forall"] = AggregPMeanError(p=2)
    it = Interpretation(sig, logic=logic)
    it["Obj"] = range(O)
    it["Object"] = sltn.Type("Object", shape=(FEAT,))
    it["Dir"] = sltn.Type("Dir", shape=(len(DIRS),))
    it["Neq"] = 1 - torch.eye(O)
    it["Present"] = lambda x: x[..., PRESENT:PRESENT + 1]
    # o1 is in direction d of o iff (o1 - o) points along d: exact on CLEVR (checked in the notebook)
    it["R"] = lambda x, x1, r: (((x1[..., XYZ] - x[..., XYZ]) @ directions.T > 0).float() * r).sum(-1, keepdim=True)
    for a in ATTRS:
        n = len(VALUES[a])
        it[SORT[a]] = sltn.Type(SORT[a], shape=(n,))
        it[PRED[a]] = (lambda s: lambda x, v: (x[..., s] * v).sum(-1, keepdim=True))(OFFSET[a])
        it[SAME[a]] = (lambda s: lambda x, x1: (x[..., s] * x1[..., s]).sum(-1, keepdim=True))(OFFSET[a])
        it[KVAR[a]] = torch.eye(n)
    return it


def family_batch(questions, scene_x):
    """The groundings of one family's variables: {name: tensor with one row per question}.
    scene_x: {image_index: scene_rows(scene)}."""
    g = family_params(questions)
    g["x"] = torch.stack([scene_x[i] for i in g.pop("scene").tolist()])
    return g


def family_params(questions):
    """Like family_batch, without x: the question's values and answer, plus `scene`, the scene index of
    each question (to build x from the scenes, e.g. from learned embeddings)."""
    N = len(questions)
    fil0, rels0 = question_params(questions[0])
    g = {"scene": torch.tensor([q["image_index"] for q in questions])}
    for k in range(len(fil0)):
        for a in ATTRS:
            g[f"{PARAM[a]}{k}"] = torch.ones(N, len(VALUES[a]))
    for k in range(len(rels0)):
        g[f"r{k}"] = torch.zeros(N, len(DIRS))
    g["yes"] = torch.zeros(N, 1)
    for a in ATTRS:
        g[ANS[a]] = torch.zeros(N, len(VALUES[a]))
    for b, q in enumerate(questions):
        fil, rels = question_params(q)
        for k, vals in enumerate(fil):
            for a, v in vals.items():
                g[f"{PARAM[a]}{k}"][b] = 0
                g[f"{PARAM[a]}{k}"][b, VALUES[a].index(v)] = 1
        for k, d in enumerate(rels):
            g[f"r{k}"][b, DIRS.index(d)] = 1
        if q["answer"] in ("yes", "no"):
            g["yes"][b] = float(q["answer"] == "yes")
        else:
            for a in ATTRS:
                if q["answer"] in VALUES[a]:
                    g[ANS[a]][b, VALUES[a].index(q["answer"])] = 1
    return g


def ground(it, g):
    for name, value in g.items():
        it[name] = value


# ----------------------------------------------------------------------------- reading per-question values

class _Recorder:
    """A Forall aggregator that also keeps what it aggregates. sLTN returns only the aggregate of a
    quantifier; to score each question we need the values inside the outer diagonal `forall`, one per
    question. The rules have a single ordinary `forall` (the object quantifiers are structural), so the
    values recorded are exactly the per-question ones."""

    def __init__(self, inner):
        self.inner, self.values = inner, None

    def __call__(self, xs, dim=None, keepdim=False, mask=None, **kw):
        assert self.values is None, "more than one ordinary forall in the formula"
        self.values = xs.detach().movedim(dim[0], 0)              # question first
        return self.inner(xs, dim=dim, keepdim=keepdim, mask=mask, **kw)

    def __repr__(self):
        return f"_Recorder({self.inner!r})"


def per_question(it, formula):
    """Truth value of the formula's body for each question: [N] for a yes/no body, [N, values] for a query
    body with its answer left open (k<attr>)."""
    inner = it.logic.Forall
    rec = it.logic.Forall = _Recorder(inner)
    try:
        with torch.no_grad():
            it(formula)
    finally:
        it.logic.Forall = inner
    return rec.values.squeeze(-1)
