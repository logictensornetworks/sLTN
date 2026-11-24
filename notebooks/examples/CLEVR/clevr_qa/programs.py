"""CLEVR's functional programs: which questions are in scope, and a program's canonical skeleton.

`canonical` turns a question's json program into a skeleton string, e.g.
"exist(F(relate(unique(F(scene())))))", plus its parameters: one {attribute: value} per filter slot F and one
direction per relate. Counting questions (COUNTING) are in scope only when asked for (`in_scope`).

The skeleton is canonical: it hides everything that varies between the questions of one CLEVR template.
- A chain of filters is one F, whatever attributes it names and however many: "large cube" and "red metal
  sphere" give the same F. An attribute a filter doesn't name becomes all ones, so its atom is exactly true.
- Every set node gets an F, even with no filter (an empty {}): "the cube left of the sphere" and "the thing
  left of the sphere" share a skeleton.
- The relate's direction is a parameter, so "left" and "front" questions share a skeleton too.
So the 101,028 in-scope validation questions fall into only 56 families (56 in train too).

The skeleton is the key that groups questions into families, not the formula: formulas.compile_question
writes the formula by walking the json program again (_Builder), in the same post-order as `canonical`, so
filter slot k and relate k mean the same in both (it checks that the counts agree).

The parameters ground the formula's variables; they are not pasted into its text. Every question of a
family has the same formula over c0, z0, m0, s0, ..., r0, and batching.family_params turns each question's
parameters into their values ({"shape": "cube"} -> s0 = one-hot cube, c0 = z0 = m0 = all ones). The diagonal
forall pairs question i's values with question i's scene.
"""

COUNTING = {"count", "equal_integer", "less_than", "greater_than"}
OUT_OF_SCOPE = COUNTING                            # without counting=True


def in_scope(q, counting=False):
    """q: a raw CLEVR question (data.py's docstring). counting: also the counting questions (formulas.py)."""
    return counting or not any(n["function"] in OUT_OF_SCOPE for n in q["program"])


def canonical(q):
    """(skeleton, filters, relates): filters is one {attribute: value} per filter slot, relates one direction
    per relate, both in post-order (the order formulas._Builder numbers them).
    The example question of data.py gives
      skeleton "exist(F(relate(unique(F(scene())))))"   (F = a filter slot)
      filters  [{"shape": "cube", "color": "gray", "size": "small"}, {"shape": "sphere"}]
      relates  ["left"]"""
    p = q["program"]
    return _canon(p, len(p) - 1)


def question_params(q):
    _, fil, rels = canonical(q)
    return fil, rels


def _canon(p, i):
    # p: the program, a list of nodes; i: a node's index. A node is {"function": str, "inputs": [indices of
    # earlier nodes], "value_inputs": [its argument, e.g. "small" for filter_size, "left" for relate]}
    # (an example program is in data.py's docstring).
    n = p[i]
    f = n["function"]
    if f.startswith("filter_"):                            # a chain of filters = one filter slot
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
