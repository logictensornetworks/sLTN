"""CLEVR as sLTN: data loading, the signature, and a compiler from CLEVR programs to formulas.

Layout of a batch of questions (one question per scene slot, so a scene appears once per question):
  Question axis q   one entry per question (the question's scene, its parameters and its answer)
  Obj axis o     the objects of that scene, padded to O = 10

Symbols
  x[q, o]              the object's appearance: here one-hot attributes (E0) or features; later a crop
  color, size, material, shape : functions x -> distribution over the attribute's values
  IsC, IsZ, IsM, IsS   dot product of two distributions of the same attribute (value match, "same")
  Present(q, o)        1 for a real object, 0 for an empty position (scenes are padded to O = 10)
  Neq(o, o1)           1 - identity: "another object"
  R<k>(q, o, o1)       the k-th relate of the program, grounded per question with the adjacency of the
                       direction it names: R<k>[q, o', o] = 1 iff o is <direction> of o'
  c<k>, z<k>, m<k>, s<k> [q]   the k-th filter slot's values; an attribute the filter doesn't name is the
                       all-ones vector, so its atom is exactly 1 (and exactly 1 for any distribution)
  Yes(q)               the answer of a yes/no question
  ac, az, am, as_ [q]  the answer of a query question, one-hot
  kc, kz, km, ks       variables over the values of each attribute (for query questions)

Programs are canonicalised so that all the questions of a family share one formula: every set node is
followed by one filter slot (empty if the program has none), and each relate's direction is a parameter.
Counting questions (count, equal_integer, less_than, greater_than) are out of scope.
"""

import json
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import torch

import sltn
from sltn import Signature

HERE = Path(__file__).resolve().parent                 # notebooks/examples/CLEVR
DATA = HERE.parents[1] / "datasets" / "CLEVR_v1.0"      # scene and question files: downloaded, not in the repo
CROPS = HERE.parents[1] / "datasets" / "CLEVR_crops"   # one crop per object, cut by prep_crops.py
IMAGES = HERE / "images"                               # the 12 validation images the notebooks show
RESULTS = HERE / "results"
URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip"

VALUES = {
    "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    "size": ["small", "large"],
    "material": ["rubber", "metal"],
    "shape": ["cube", "sphere", "cylinder"],
}
ATTRS = list(VALUES)
SORT = {"color": "Col", "size": "Siz", "material": "Mat", "shape": "Shp"}
IS = {"color": "IsC", "size": "IsZ", "material": "IsM", "shape": "IsS"}
PARAM = {"color": "c", "size": "z", "material": "m", "shape": "s"}      # filter slot k: c<k>, z<k>, ...
ANS = {"color": "ac", "size": "az", "material": "am", "shape": "as_"}
KVAR = {"color": "kc", "size": "kz", "material": "km", "shape": "ks"}
OFFSET = {}
_o = 0
for _a in ATTRS:
    OFFSET[_a] = (_o, _o + len(VALUES[_a]))
    _o += len(VALUES[_a])
FEAT = _o                                    # 15: the one-hot attributes concatenated
DIRS = ["left", "right", "front", "behind"]
O = 10
OUT_OF_SCOPE = {"count", "equal_integer", "less_than", "greater_than"}
MAX_SLOTS, MAX_RELS, MAX_OBJVARS = 12, 6, 12


# ----------------------------------------------------------------------------- data

def download():
    """CLEVR's scene and question files (CLEVR_v1.0_no_images.zip, 85 MB) into notebooks/datasets, once.
    The images are only in the full CLEVR zip (18 GB, https://cs.stanford.edu/people/jcjohns/clevr/), needed
    only to cut the crops for training (prep_crops.py)."""
    if (DATA / "scenes").is_dir() and (DATA / "questions").is_dir():
        return
    archive = DATA.parent / "CLEVR_v1.0_no_images.zip"
    if not archive.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL} (85 MB) to {archive}", flush=True)
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(DATA.parent)


def sltn_version():
    """Commit, branch and dirty flag of the sLTN checkout in use, saved with every result."""
    root = Path(sltn.__file__).resolve().parent.parent
    git = lambda *args: subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(git("status", "--porcelain", "--untracked-files=no")))


def load(root, split):
    root = Path(root)
    if root == DATA:
        download()
    scenes = json.load(open(root / "scenes" / f"CLEVR_{split}_scenes.json"))["scenes"]
    questions = json.load(open(root / "questions" / f"CLEVR_{split}_questions.json"))["questions"]
    return scenes, questions


def in_scope(q):
    return not any(n["function"] in OUT_OF_SCOPE for n in q["program"])


def scene_tensors(scene):
    """One-hot attributes [O, FEAT], present [O], relations {dir: [O, O]} with R[u, o] = 1 iff o is dir of u."""
    x = torch.zeros(O, FEAT)
    present = torch.zeros(O)
    for i, ob in enumerate(scene["objects"]):
        present[i] = 1
        for a in ATTRS:
            x[i, OFFSET[a][0] + VALUES[a].index(ob[a])] = 1
    rel = {}
    for d in DIRS:
        r = torch.zeros(O, O)
        for u, objs in enumerate(scene["relationships"][d]):
            r[u, objs] = 1
        rel[d] = r
    return x, present, rel


# ----------------------------------------------------------------------------- compiler

@dataclass
class Compiled:
    """A family's formula and how to ground its parameters from one question."""
    key: str                                  # the canonical skeleton
    kind: str                                 # "yesno" or an attribute name (query)
    body: str                                 # the answer as a formula, free in q (and k<attr> for queries)
    rule: str                                 # forall q: answer <-> body
    n_slots: int
    n_rels: int
    params: callable = field(repr=False, default=None)   # question -> (filters list, relate dirs list)


def _canon(p, i):
    """Canonical skeleton of node i, and the question's parameters in the same order (filters, relates)."""
    n = p[i]
    f = n["function"]
    if f.startswith("filter_"):                            # a chain of filters = one slot
        vals, j = {}, i
        while p[j]["function"].startswith("filter_"):
            vals[p[j]["function"][7:]] = p[j]["value_inputs"][0]
            j = p[j]["inputs"][0]
        sk, fil, rels = _canon_set_core(p, j)
        return f"F({sk})", fil + [vals], rels
    if f in ("scene", "relate", "intersect", "union") or f.startswith("same_"):   # set without a filter
        sk, fil, rels = _canon_set_core(p, i)
        return f"F({sk})", fil + [{}], rels
    sks, fil, rels = [], [], []
    for j in n["inputs"]:
        s, fl, rl = _canon(p, j)
        sks.append(s), fil.extend(fl), rels.extend(rl)
    return f"{f}({','.join(sks)})", fil, rels


def _canon_set_core(p, i):
    n = p[i]
    f = n["function"]
    sks, fil, rels = [], [], []
    for j in n["inputs"]:
        s, fl, rl = _canon(p, j)
        sks.append(s), fil.extend(fl), rels.extend(rl)
    if f == "relate":
        rels.append(n["value_inputs"][0])
    return f"{f}({','.join(sks)})", fil, rels


def canonical(q):
    p = q["program"]
    return _canon(p, len(p) - 1)


class _Builder:
    """Turns a canonical skeleton into formula text. Slots and relates are numbered in the same order
    as `_canon` collects the question's parameters (post-order)."""

    def __init__(self, program):
        self.p = program
        self.slot = 0
        self.rel = 0
        # object variables in program order: the k-th `unique` of the program is o<k> (k = 1, 2, ...)
        uniques = [i for i, n in enumerate(program) if n["function"] == "unique"]
        self.obj = {i: f"o{k}" for k, i in enumerate(uniques, 1)}
        self.next = len(uniques) + 1                      # for the objects `exist` looks for

    def fresh(self):
        self.next += 1
        return f"o{self.next - 1}"

    def filter_atoms(self, v, k):
        return " & ".join(f"{IS[a]}({a}(x[q, {v}]), {PARAM[a]}{k}[q])" for a in ATTRS)

    def set_(self, i, v):
        """Formula, free in object variable v, true iff object v is in the set computed by node i."""
        n = self.p[i]
        f = n["function"]
        if f.startswith("filter_"):
            j = i
            while self.p[j]["function"].startswith("filter_"):
                j = self.p[j]["inputs"][0]
            inner = self.core(j, v)
        else:
            inner = self.core(i, v)
        k = self.slot
        self.slot += 1
        return f"({inner} & {self.filter_atoms(v, k)})"

    def core(self, i, v):
        n = self.p[i]
        f = n["function"]
        if f == "scene":
            return f"Present(q, {v})"
        if f == "intersect":
            a = self.set_(n["inputs"][0], v)
            b = self.set_(n["inputs"][1], v)
            return f"({a} & {b})"
        if f == "union":
            a = self.set_(n["inputs"][0], v)
            b = self.set_(n["inputs"][1], v)
            return f"({a} | {b})"
        if f == "relate":
            ref = self.obj[n["inputs"][0]]                 # the object the set is defined from
            ref_set = self.unique(n["inputs"][0])
            k = self.rel
            self.rel += 1
            return f"(Present(q, {v}) & exists {ref}: ({ref_set} & R{k}(q, {ref}, {v})))"
        if f.startswith("same_"):
            a = f[5:]
            ref = self.obj[n["inputs"][0]]
            ref_set = self.unique(n["inputs"][0])
            return (f"(Present(q, {v}) & exists {ref}: ({ref_set} & Neq({ref}, {v}) & "
                    f"{IS[a]}({a}(x[q, {ref}]), {a}(x[q, {v}]))))")
        raise ValueError(f"unexpected set function {f}")

    def unique(self, i):
        """Formula free in this unique's object variable: it is in the set the unique picks from."""
        n = self.p[i]
        assert n["function"] == "unique", n["function"]
        return self.set_(n["inputs"][0], self.obj[i])

    def top(self):
        i = len(self.p) - 1
        n = self.p[i]
        f = n["function"]
        if f == "exist":
            v = self.fresh()
            return "yesno", f"exists {v}: {self.set_(n['inputs'][0], v)}"
        if f.startswith("query_"):
            a = f[6:]
            v = self.obj[n["inputs"][0]]
            s = self.unique(n["inputs"][0])
            return a, f"exists {v}: ({s} & {IS[a]}({a}(x[q, {v}]), {KVAR[a]}))"
        if f.startswith("equal_"):
            a = f[6:]
            q1, q2 = (self.p[j] for j in n["inputs"])
            assert q1["function"] == q2["function"] == f"query_{a}"
            v1, v2 = self.obj[q1["inputs"][0]], self.obj[q2["inputs"][0]]
            s1, s2 = self.unique(q1["inputs"][0]), self.unique(q2["inputs"][0])
            return "yesno", (f"exists {v1}: ({s1} & exists {v2}: ({s2} & "
                             f"{IS[a]}({a}(x[q, {v1}]), {a}(x[q, {v2}]))))")
        raise ValueError(f"unexpected top function {f}")


def compile_question(q):
    key, fil, rels = canonical(q)
    b = _Builder(q["program"])
    kind, body = b.top()
    assert b.slot == len(fil) and b.rel == len(rels), (key, b.slot, len(fil), b.rel, len(rels))
    if kind == "yesno":
        rule = f"forall q: (Yes(q) <-> {body})"
    else:
        rule = f"forall q: forall {KVAR[kind]}: ({IS[kind]}({KVAR[kind]}, {ANS[kind]}[q]) <-> {body})"
    return Compiled(key, kind, body, rule, b.slot, b.rel)


def question_params(q):
    _, fil, rels = canonical(q)
    return fil, rels


# ----------------------------------------------------------------------------- signature

def make_signature():
    sig = Signature("clevr")
    sig.sort("Feat")
    for a in ATTRS:
        sig.sort(SORT[a])
    sig.dimension("Question")
    sig.dimension("Obj")
    sig.structural_variable("q", "Question")
    for i in range(1, MAX_OBJVARS + 1):
        sig.structural_variable(f"o{i}", "Obj")
    sig.constant("x", "Feat", dims=["Question", "Obj"])
    for a in ATTRS:
        sig.function(a, ["Feat"], SORT[a])
        sig.predicate(IS[a], [SORT[a], SORT[a]])
        sig.variable(KVAR[a], SORT[a])
        sig.constant(ANS[a], SORT[a], dims=["Question"])
        for k in range(MAX_SLOTS):
            sig.constant(f"{PARAM[a]}{k}", SORT[a], dims=["Question"])
    sig.structural_relation("Present", ["Question", "Obj"])
    sig.structural_relation("Neq", ["Obj", "Obj"])
    sig.structural_relation("Yes", ["Question"])
    for k in range(MAX_RELS):
        sig.structural_relation(f"R{k}", ["Question", "Obj", "Obj"])
    return sig


def batch_tensors(questions, scene_cache, n_slots, n_rels):
    """Groundings for a batch of questions of one family (same canonical skeleton)."""
    B = len(questions)
    x = torch.zeros(B, O, FEAT)
    present = torch.zeros(B, O)
    rel = torch.zeros(n_rels, B, O, O)
    slots = {a: torch.ones(n_slots, B, len(VALUES[a])) for a in ATTRS}
    yes = torch.zeros(B)
    ans = {a: torch.zeros(B, len(VALUES[a])) for a in ATTRS}
    for b, q in enumerate(questions):
        sx, sp, sr = scene_cache[q["image_index"]]
        x[b], present[b] = sx, sp
        fil, rels = question_params(q)
        for k, d in enumerate(rels):
            rel[k, b] = sr[d]
        for k, vals in enumerate(fil):
            for a, v in vals.items():
                slots[a][k, b] = 0
                slots[a][k, b, VALUES[a].index(v)] = 1
        if q["answer"] in ("yes", "no"):
            yes[b] = q["answer"] == "yes"
        else:
            for a in ATTRS:
                if q["answer"] in VALUES[a]:
                    ans[a][b, VALUES[a].index(q["answer"])] = 1
    return dict(x=x, present=present, rel=rel, slots=slots, yes=yes, ans=ans)


# ----------------------------------------------------------------------------- training data

def scene_arrays(scenes):
    """Attribute value indices [S, O, 4] (-1 for an empty position, attributes in ATTRS order),
    present [S, O] (bool), relations [S, 4 directions (DIRS), O, O] with R[s, d, u, o] = 1 iff o is d of u."""
    S = len(scenes)
    attr = torch.full((S, O, len(ATTRS)), -1, dtype=torch.long)
    rel = torch.zeros(S, len(DIRS), O, O)
    for s, sc in enumerate(scenes):
        for i, ob in enumerate(sc["objects"]):
            for j, a in enumerate(ATTRS):
                attr[s, i, j] = VALUES[a].index(ob[a])
        for d, name in enumerate(DIRS):
            for u, objs in enumerate(sc["relationships"][name]):
                rel[s, d, u, objs] = 1
    return attr, attr[..., 0] >= 0, rel


def family_tensors(questions):
    """In-scope questions grouped by formula: {skeleton: (Compiled, tensors)}. The tensors hold, per
    question, its scene index, the direction index of each relate, the filter values (one-hot, all-ones
    when not named) and the ground truth, so a batch is grounded by indexing."""
    groups, comp = {}, {}
    for q in questions:
        if not in_scope(q):
            continue
        c = compile_question(q)
        groups.setdefault(c.key, []).append(q)
        comp.setdefault(c.key, c)
    fams = {}
    for key, qs in groups.items():
        c = comp[key]
        N = len(qs)
        t = dict(scene=torch.tensor([q["image_index"] for q in qs]),
                 rel=torch.zeros(N, c.n_rels, dtype=torch.long),
                 slots={a: torch.ones(c.n_slots, N, len(VALUES[a])) for a in ATTRS},
                 yes=torch.zeros(N), ans={a: torch.zeros(N, len(VALUES[a])) for a in ATTRS})
        for b, q in enumerate(qs):
            fil, rels = question_params(q)
            for k, d in enumerate(rels):
                t["rel"][b, k] = DIRS.index(d)
            for k, vals in enumerate(fil):
                for a, v in vals.items():
                    t["slots"][a][k, b] = 0
                    t["slots"][a][k, b, VALUES[a].index(v)] = 1
            if q["answer"] in ("yes", "no"):
                t["yes"][b] = q["answer"] == "yes"
            else:
                t["ans"][c.kind][b, VALUES[c.kind].index(q["answer"])] = 1
        fams[key] = (c, t)
    return fams


def question_type(key):
    """exist / compare / query, from a skeleton."""
    return "exist" if key.startswith("exist") else "compare" if key.startswith("equal_") else "query"


def by_name(r, order):
    """The value of an sLTN result with its axes in the given order (the axes are named)."""
    names = [ax.name for ax in r.axes][:-1]
    return r.value.squeeze(-1).permute(*[names.index(n) for n in order])
