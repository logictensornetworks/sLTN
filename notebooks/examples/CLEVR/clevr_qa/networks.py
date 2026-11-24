"""The learned groundings: a CNN for the attributes, the predicates that read it, and the relation R.

ConceptNet's trunk embeds each object's crop (the appearance part of scene[o]); one linear head per attribute
gives a distribution over the attribute's values. C, Z, M, S compare that distribution with a value,
SameC & co. compare two objects' distributions. R reads the position part of scene[o]: CLEVR's rule on the 3D
coordinates (exact; the relations given), or a small network on the pixel coordinates (learned).
"""

import torch
import torch.nn as nn

from .vocabulary import APPEARANCE, ATTRS, DIRS, POS, VALUES


class ConceptNet(nn.Module):
    """Trunk: [N, 48, 48, 3] uint8 crops -> [N, 4w] embeddings; heads: embedding -> softmax per attribute."""

    def __init__(self, width=64):
        super().__init__()
        w = width
        block = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                           nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                           nn.MaxPool2d(2))
        self.trunk = nn.Sequential(block(3, w), block(w, 2 * w), block(2 * w, 2 * w), block(2 * w, 4 * w),
                                   nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.heads = nn.ModuleDict({a: nn.Linear(4 * w, len(VALUES[a])) for a in ATTRS})

    @property
    def dim(self):
        return self.heads[ATTRS[0]].in_features

    def embed(self, crops):
        return self.trunk(crops.permute(0, 3, 1, 2).float().div(255).sub(0.5))

    def forward(self, crops):
        h = self.embed(crops)
        return {a: torch.softmax(head(h), -1) for a, head in self.heads.items()}


def dot(p, q):
    # clamped: with an all-ones filter value, the sum of a softmax can be 1 + 1e-7, which isn't a truth value
    return (p * q).sum(-1, keepdim=True).clamp(0, 1)


class Has(nn.Module):
    """C(scene[o], v), ...: the head's distribution for the object, dotted with the value v."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x, v):
        return dot(torch.softmax(self.linear(x[..., APPEARANCE]), -1), v)


class Same(nn.Module):
    """SameC(scene[o], scene[o1]), ...: the two objects' distributions, dotted."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x, x1):
        return dot(torch.softmax(self.linear(x[..., APPEARANCE]), -1),
                   torch.softmax(self.linear(x1[..., APPEARANCE]), -1))


class Relation(nn.Module):
    """R(scene[o], scene[o1], r): o1 is in direction r of o. learn=False: CLEVR's rule on the 3D coordinates,
    (o1 - o) points along the direction (exact on CLEVR, checked by ground_truth.check_relations): the relations
    given. learn=True: an MLP on the pixel coordinates (column, row, depth) of both objects and their
    difference, one sigmoid per direction; it never sees the 3D coordinates.
    `reads` names the position scene[o][POS] must hold (a key of data.scene_arrays): "xyz" or "pixel".
    `directions` is data.camera_directions(scenes), used only by the rule.

    antisymmetric=True (learn only): the MLP g gives one score per axis (left/right, front/behind) and
        R(o, o1, left)  = sigmoid(g(o, o1) - g(o1, o))      R(o, o1, right)  = R(o1, o, left)
        R(o, o1, front) = sigmoid(h(o, o1) - h(o1, o))      R(o, o1, behind) = R(o1, o, front)
    Swapping o and o1 flips the score's sign, so R(o, o1, left) + R(o1, o, left) = 1 and right = 1 - left:
    "everything in every direction" can't be represented (a free MLP fell into it when trained from the answers
    alone, its sigmoids saturated at 1 and R stopped learning). R(o, o, d) = 0.5."""

    def __init__(self, directions, learn, hidden=64, antisymmetric=False):
        super().__init__()
        assert learn or not antisymmetric
        self.reads = "pixel" if learn else "xyz"
        self.antisymmetric = antisymmetric
        self.register_buffer("directions", directions)
        out = 2 if antisymmetric else len(DIRS)
        self.mlp = nn.Sequential(nn.Linear(9, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out)) if learn else None

    def probs(self, p, p1):
        """p, p1 [..., 3]: the positions of o and o1 -> [..., 4], one truth value per direction."""
        if self.mlp is None:
            return ((p1 - p) @ self.directions.T > 0).float()
        if not self.antisymmetric:
            return torch.sigmoid(self.mlp(torch.cat([p, p1, p1 - p], -1)))
        s = self.mlp(torch.cat([p, p1, p1 - p], -1)) - self.mlp(torch.cat([p1, p, p - p1], -1))   # [..., 2]
        assert DIRS == ["left", "right", "front", "behind"]
        return torch.sigmoid(torch.stack([s[..., 0], -s[..., 0], s[..., 1], -s[..., 1]], -1))

    def forward(self, x, x1, r):
        return (self.probs(x[..., POS], x1[..., POS]) * r).sum(-1, keepdim=True)
