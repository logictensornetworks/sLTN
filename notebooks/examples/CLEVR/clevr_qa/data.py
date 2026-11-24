"""CLEVR's files: where they are, loading them, and the per-scene tensors the other modules read.

The scene and question files (CLEVR_v1.0_no_images.zip, 85 MB) are downloaded once into notebooks/datasets.
The images are only in the full CLEVR zip (18 GB, https://cs.stanford.edu/people/jcjohns/clevr/), needed
only to cut the crops (prepare_crops.py).

CLEVR's raw records (json; the example is val scene 0 and one of its questions)

scene = {
  "image_index": 0,                    # int: the scene's position in the file; questions point to it, and
                                       #   the rows of every per-scene tensor here (and of the crops) follow it
  "image_filename": "CLEVR_val_000000.png",
  "split": "val",
  "objects": [                         # list of 3 to 10 objects, in no particular order
    {"color": "brown", "size": "large", "material": "rubber", "shape": "cylinder",   # values: vocabulary.VALUES
     "3d_coords": [-1.49, -1.99, 0.70],     # [x, y, z] of the centre in the 3D scene (world units); z is the
                                            #   height: 0.7 for every large object, 0.35 for every small one
     "pixel_coords": [119, 131, 10.80],     # [column, row, depth]: the object's centre in the 480x320 image,
                                            #   and its distance from the camera (prepare_crops uses all three;
                                            #   a learned R reads them, scaled to about [0, 1])
     "rotation": 178.9},                    # degrees around the vertical axis; not used
    ...],
  "relationships": {                   # for each direction d: one list per object u, the indices of the
    "left": [[], [0, 2], [0], ...],    #   objects that are d of u. Here: object 0 is left of object 1.
    "right": ..., "front": ..., "behind": ...},
  "directions": {                      # one unit vector per direction, in 3D coordinates; the same camera
    "left": [-0.66, -0.75, 0.0], ...}  #   in every scene. Six are given; only left, right, front, behind are
                                       #   used ("above", "below" are not).
                                       #   The x, y axes are fixed to the floor, but the camera looks at it at
                                       #   an angle, so "left in the image" is a diagonal on the floor: moving
                                       #   left goes -0.66 in x and -0.75 in y. o1 is d of o iff o1 is further
                                       #   along d: dot(xyz(o1) - xyz(o), directions[d]) > 0 (comparing x alone
                                       #   is wrong for 27% of pairs). Checked to give exactly "relationships".
                                       #   Used only by R's fixed rule (the relations given); a learned R
                                       #   reads only the pixel coordinates, never the 3D ones.
}

question = {
  "image_index": 0,                    # the scene the question is about (scene["image_index"])
  "question": "There is a small gray block; are there any spheres to the left of it?",   # text: not used
  "answer": "yes",                     # "yes" / "no", an attribute value, or a number (counting: out of scope)
  "question_family_index": 73,         # CLEVR's template id: not used (our families come from the program)
  "question_index": 3, "image_filename": ..., "split": ...,
  "program": [                         # the functional program, a list of nodes in execution order
    {"function": "scene",        "inputs": [],  "value_inputs": []},        # node 0: all the objects
    {"function": "filter_size",  "inputs": [0], "value_inputs": ["small"]}, # inputs: indices of earlier nodes
    {"function": "filter_color", "inputs": [1], "value_inputs": ["gray"]},  # value_inputs: the node's argument
    {"function": "filter_shape", "inputs": [2], "value_inputs": ["cube"]},  #   ("block" in the text = "cube")
    {"function": "unique",       "inputs": [3], "value_inputs": []},        # the set has exactly one object
    {"function": "relate",       "inputs": [4], "value_inputs": ["left"]},  # objects left of it
    {"function": "filter_shape", "inputs": [5], "value_inputs": ["sphere"]},
    {"function": "exist",        "inputs": [6], "value_inputs": []}]        # the last node gives the answer
}
"""

import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

from .programs import in_scope
from .vocabulary import ATTRS, DIRS, N_SLOTS, VALUES

HERE = Path(__file__).resolve().parent                    # notebooks/examples/CLEVR/clevr_qa
DATA = Path(os.environ.get("CLEVR_DATA") or HERE.parents[2] / "datasets" / "CLEVR_v1.0")   # scene and question
                                                          #   files (and the images, if any); $CLEVR_DATA on the cluster
CROPS = HERE.parents[2] / "datasets" / "CLEVR_crops"      # one crop per object, cut by prepare_crops.py
IMAGES = HERE.parents[2] / "datasets" / "CLEVR_images128"  # whole images, 128 x 192 (../prep_images.py)
RESULTS = HERE.parent / "results" / "clevr_qa"            # one folder per training run
URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip"


def download():
    if (DATA / "scenes").is_dir() and (DATA / "questions").is_dir():
        return
    archive = DATA.parent / "CLEVR_v1.0_no_images.zip"
    if not archive.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL} (85 MB) to {archive}", flush=True)
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(DATA.parent)


def load(split, root=DATA):
    """(scenes, questions) of a split, as in CLEVR's json files."""
    root = Path(root)
    if root == DATA:
        download()
    scenes = json.load(open(root / "scenes" / f"CLEVR_{split}_scenes.json"))["scenes"]
    questions = json.load(open(root / "questions" / f"CLEVR_{split}_questions.json"))["questions"]
    return scenes, questions


def subset(questions, n, seed):
    """n in-scope questions drawn at random with the run's seed (10,000 + seed), kept in the file's order."""
    pool = [q for q in questions if in_scope(q)]
    idx = torch.randperm(len(pool), generator=torch.Generator().manual_seed(10_000 + seed))[:n]
    return [pool[i] for i in sorted(idx.tolist())]


def camera_directions(scenes):
    """[4, 3]: CLEVR's unit vectors for left, right, front, behind (the same camera in every scene)."""
    d = torch.tensor([scenes[0]["directions"][k] for k in DIRS])
    for s in scenes:
        assert torch.allclose(torch.tensor([s["directions"][k] for k in DIRS]), d), s["image_index"]
    return d


def scene_arrays(scenes):
    """The annotations as tensors, one row per scene, padded to N_SLOTS object slots:

      attr     [S, N_SLOTS, 4]           indices: each object's colour, size, material and shape as its
                                         position in VALUES (in ATTRS order; cyan -> 6); -1 for an empty slot
      present  [S, N_SLOTS]              bool: the object slot holds an object (attr >= 0)
      rel      [S, 4, N_SLOTS, N_SLOTS]  a 0/1 table, not indices: rel[s, d, u, o] = 1 iff object o is in
                                         direction d (DIRS order) of object u; built from the file's index lists
      pixel    [S, N_SLOTS, 3]           pixel coordinates: column / 480, row / 320, depth / 15 (all in about
                                         [0, 1]); zeros for an empty slot
      xyz      [S, N_SLOTS, 3]           3D coordinates; zeros for an empty slot

    present and one of pixel / xyz (the position, see vocabulary.POS) go into scene[o] (interpretation.ground):
    a learned R reads pixel, CLEVR's rule reads xyz. attr gives the ground-truth appearance and the concept
    accuracy; rel never enters a formula: it is only what R is checked against."""
    S = len(scenes)
    attr = torch.full((S, N_SLOTS, len(ATTRS)), -1, dtype=torch.long)
    rel = torch.zeros(S, len(DIRS), N_SLOTS, N_SLOTS)
    pixel = torch.zeros(S, N_SLOTS, 3)
    xyz = torch.zeros(S, N_SLOTS, 3)
    for s, sc in enumerate(scenes):
        for i, ob in enumerate(sc["objects"]):
            x, y, d = ob["pixel_coords"]
            pixel[s, i] = torch.tensor([x / 480, y / 320, d / 15])
            xyz[s, i] = torch.tensor(ob["3d_coords"])
            for j, a in enumerate(ATTRS):
                attr[s, i, j] = VALUES[a].index(ob[a])
        for d, name in enumerate(DIRS):
            for u, objs in enumerate(sc["relationships"][name]):
                rel[s, d, u, objs] = 1
    return dict(attr=attr, present=attr[..., 0] >= 0, rel=rel, pixel=pixel, xyz=xyz)


def load_crops(split, n_scenes, folder=CROPS):
    """uint8 [n_scenes, N_SLOTS, 48, 48, 3]: the first n_scenes of the split's crops."""
    return torch.from_numpy(np.load(Path(folder) / f"{split}_crops.npy", mmap_mode="r")[:n_scenes].copy())


def load_images(split, n_scenes, folder=IMAGES):
    """uint8 [n_scenes, 128, 192, 3]: the first n_scenes of the split's resized images (the slot model's input)."""
    return torch.from_numpy(np.load(Path(folder) / f"{split}.npy", mmap_mode="r")[:n_scenes].copy())
