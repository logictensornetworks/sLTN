"""Questions -> families -> the groundings of a family's variables.

A family is the questions with the same canonical skeleton: they share one Compiled formula, and their
values (filter slots, directions, answer) are tensors with one row per question. `scene` is
grounded separately (interpretation.ground, ground_truth.ground), from `scene_idx`, each question's
scene index.
"""

import collections

import torch

from .formulas import compile_question
from .programs import canonical, in_scope, question_params
from .vocabulary import ANS, ATTRS, DIRS, MAX_COUNT, PARAM, VALUES


def family_params(questions):
    """{variable: tensor [N, ...]} for one family's questions, plus `scene_idx` [N], each question's scene index.
    An attribute a filter slot doesn't filter on is the all-ones vector.
    For data.py's example question: c0 = gray, z0 = small, m0 = all ones, s0 = cube (filter slot 0, the
    reference object), c1 = z1 = m1 = all ones, s1 = sphere (filter slot 1), r0 = left, yes = 1, ans_* = zeros."""
    N = len(questions)
    fil0, rels0 = question_params(questions[0])
    g = {"scene_idx": torch.tensor([q["image_index"] for q in questions])}
    for k in range(len(fil0)):
        for a in ATTRS:
            g[f"{PARAM[a]}{k}"] = torch.ones(N, len(VALUES[a]))
    for k in range(len(rels0)):
        g[f"r{k}"] = torch.zeros(N, len(DIRS))
    g["yes"] = torch.zeros(N, 1)
    g["n"] = torch.zeros(N, MAX_COUNT + 1)                  # the answer of a "how many" question
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
        elif q["answer"].isdigit():
            g["n"][b, int(q["answer"])] = 1
        else:
            for a in ATTRS:
                if q["answer"] in VALUES[a]:
                    g[ANS[a]][b, VALUES[a].index(q["answer"])] = 1
    return g


def family_tensors(questions, device="cpu", present="guard", counting=False):
    """{skeleton: (Compiled, family_params)} for the in-scope questions, tensors on `device`.
    present: how the formulas use Present (formulas.compile_question); counting: also the counting questions."""
    groups = collections.defaultdict(list)
    for q in questions:
        if in_scope(q, counting):
            groups[canonical(q)[0]].append(q)
    out = {}
    for key, qs in groups.items():
        c = compile_question(qs[0], present)
        out[key] = (c, {k: v.to(device) for k, v in family_params(qs).items()})
    return out


def question_type(key, c):
    """exist / query / compare / count / compare_count, + "+relate" if the program has a relate: the groups
    results are reported by."""
    if key.startswith("count"):
        typ = "count"
    elif key.startswith(("equal_integer", "less_than", "greater_than")):
        typ = "compare_count"
    else:
        typ = "exist" if key.startswith("exist") else "compare" if key.startswith("equal_") else "query"
    return typ + ("+relate" if c.n_rels else "")
