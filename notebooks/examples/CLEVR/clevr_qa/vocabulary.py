"""The names shared by every module: CLEVR's attribute values and directions, the sLTN symbol names, and
the layout of an object's feature vector scene[o].

A scene has 3 to 10 objects. To put every scene in one tensor, each scene gets N_SLOTS = 10 object slots
(the structural dimension Obj, indexed by o, o1, ...): slot o holds the feature vector scene[o] of the o-th
object, and the slots past the scene's last object are padding, all zeros (so their presence bit is 0).

Symbols (declared in formulas.make_signature)
  scene                  the question's scene: one row scene[o] per object slot
  c<k>, z<k>, m<k>, s<k> the k-th filter slot's colour, size, material and shape: one-hot, or the all-ones
                         vector when the question doesn't name that attribute (the atom is then exactly 1)
  r<k>                   the direction of the k-th relate step (left, right, front, behind), one-hot over DIRS:
                         o1 is in direction r of o when the step from o to o1 points that way (see R)
  yes                    the answer of a yes/no question (sort Bool)
  ans_c, ans_z, ans_m, ans_s   the answer of a query question, one-hot
  kc, kz, km, ks         every value of an attribute (to read off what a query formula answers)
  Present(scene[o])      the presence bit: 0 for a padding slot; every object quantifier is guarded by it
  C, Z, M, S(scene[o], v)  the object has colour / size / material / shape v
  SameC, ..., SameS(scene[o], scene[o1])   the two objects have the same colour / ...
  R(scene[o], scene[o1], r)  o1 is in direction r of o
  Neq(o, o1)             structural relation: o and o1 are different slots ("another object")

Counting questions (programs.COUNTING; only with counting=True):
  cnt0, cnt1             a count: for n = 0..MAX_COUNT, the truth of "exactly n slots are in the set", one
                         vector per question (counting.py computes it from the set formula's value per slot)
  n                      the answer of a "how many" question, one-hot over 0..MAX_COUNT
  kn                     every count (to read off what a "how many" formula answers)
  EqCount(cnt, n)        the count is n; EqCount(cnt0, cnt1): the two counts are equal
  MoreThan(cnt0, cnt1)   the first count is larger than the second

Neq is structural because it depends only on the slot indices: it is the same 10 x 10 matrix for every
question. Present and R depend on the question's scene (which slots are filled, where the objects are),
so they are predicates that read scene[o]: there is no question axis to index a fixed tensor by.
"""

VALUES = {
    "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    "size": ["small", "large"],
    "material": ["rubber", "metal"],
    "shape": ["cube", "sphere", "cylinder"],
}
ATTRS = list(VALUES)
DIRS = ["left", "right", "front", "behind"]
N_SLOTS = 10                                                            # object slots per scene

PRED = {"color": "C", "size": "Z", "material": "M", "shape": "S"}       # C(scene[o], c): o has colour c
SAME = {a: f"Same{p}" for a, p in PRED.items()}                         # SameC(scene[o], scene[o1])
SORT = {"color": "Col", "size": "Siz", "material": "Mat", "shape": "Shp"}
PARAM = {"color": "c", "size": "z", "material": "m", "shape": "s"}      # filter slot k: c<k>, z<k>, m<k>, s<k>
ANS = {a: f"ans_{p}" for a, p in PARAM.items()}                         # ans_c, ans_z, ans_m, ans_s
KVAR = {a: f"k{p}" for a, p in PARAM.items()}                           # kc: every colour
MAX_FILTERS, MAX_RELS, MAX_OBJVARS = 12, 6, 12                          # enough for every CLEVR program
MAX_COUNT = 11                                     # a count is a vector over 0..MAX_COUNT (11 slots at most)
COUNT_VARS = ["cnt0", "cnt1"]                      # the counts a question compares (one for "how many")

# scene[o] = [present, position (3), appearance]. The position is what R reads: the object's pixel coordinates
# (column / 480, row / 320, depth / 15) when R is learned; the 3D coordinates when R is CLEVR's own rule (the
# relations given, and the ground truth). The appearance is the CNN embedding of the object's crop when
# learning, the one-hot attributes for the ground truth (ground_truth.py).
# The three constants below index the parts of scene[o], e.g. scene[o][POS]. APPEARANCE is open-ended because its
# length varies: 15 for the ground truth (8 + 2 + 2 + 3 one-hot), the embedding size D for the CNN.
PRESENT = 0
POS = slice(1, 4)
APPEARANCE = slice(4, None)
