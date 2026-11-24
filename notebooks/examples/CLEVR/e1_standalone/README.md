# E1, standalone

E1: learn colour, size, material and shape from CLEVR object crops, from the answers only. Uses the
original formulation (structural question axis `q`, the compiler in `clevr_sltn.py`).

Copied from `notebooks/examples/CLEVR` on 2026-11-13 (working-tree versions). Imports only `sltn` and these files.

| File | Role |
|---|---|
| `clevr_sltn.py` | data loading, signature, compiler from CLEVR programs to formulas, batch tensors |
| `prep_crops.py` | cuts one 48x48 crop per object from the CLEVR images (run once) |
| `e1_images.py` | the CNN (`ConceptNet`), the grounding, the training loop, the evaluation |

Changes from the originals: only the paths (one folder deeper). Data still read from
`notebooks/datasets/CLEVR_v1.0` and `notebooks/datasets/CLEVR_crops`; results go to `./results/e1/`.

Run: `python prep_crops.py` (needs the full CLEVR images), then `python e1_images.py [--steps 4500] [--learn_relations] [--seed 0]`.
