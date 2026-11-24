# clevr_qa

Learn CLEVR's attributes (and optionally the spatial relations) from question answers only, with sLTN.
Every per-question value is an ordinary variable, and a family's rule quantifies them together:
`forall (scene, c0, ..., r0, yes): (yes <-> body)`.

Run from `notebooks/examples/CLEVR` (relative imports, so always with `-m`):

```
python -m clevr_qa.prepare_crops                    # once; needs the full CLEVR images
python -m clevr_qa.ground_truth --split train       # the formulas are satisfied by the ground truth
python -m clevr_qa.train --steps 4500 [--learn_relations] [--seed 0] [--no_wandb]
python -m clevr_qa.train_slots --counting --amp [--recon_weight 0.5] [--dup_weight 0.5] [--match paint]   # slots; needs ../prep_images.py
```

| Module | Holds |
|---|---|
| `vocabulary.py` | attribute values, directions, symbol names, layout of `scene[o]` |
| `data.py` | paths, download, loading, per-scene tensors (`scene_arrays`), crops, `subset` |
| `programs.py` | CLEVR programs: `in_scope`, canonical skeleton and parameters |
| `formulas.py` | signature and compiler: program → family formulas (`rule`, `predict`) |
| `batching.py` | questions → families → variable groundings |
| `networks.py` | `ConceptNet` (CNN + heads), `Has`, `Same`, `Relation` |
| `interpretation.py` | shared setup, the learned interpretation, `ground`, `per_question` |
| `counting.py` | the counts of the counting questions (`count`, `EqCount`, `MoreThan`), computed outside sLTN |
| `ground_truth.py` | the interpretation from the annotations, `check_formulas`, `check_relations` |
| `evaluation.py` | concept, relation and QA accuracy |
| `train.py` | command line, training loop, results |
| `prepare_crops.py` | cuts one 48x48 crop per object |
| `slot_model.py` | Slot Attention (a copy of `../slots.py`): image -> 11 slots, their masks and the reconstruction |
| `train_slots.py` | no crops: the slot model trained by the knowledge base (the question rules, the reconstruction axiom, the duplicate axiom), from scratch |

## Reading order

From raw CLEVR to training. About 2.5 h for a careful first read, 1 h to skim (estimate); steps 3–4 take
about half of it.

| # | File | Lines | Time | What to focus on |
|---|---|---|---|---|
| 1 | `vocabulary.py` | 52 | 5 min | the symbol table in the docstring; the layout of `scene[o]` |
| 2 | `data.py` | 139 | 15 min | the annotated raw scene and question at the top; `scene_arrays` |
| 3 | `programs.py` | 81 | 20 min | `canonical`, with the worked example in its docstring; trace the recursive `_canon` once on it |
| 4 | `formulas.py` | 161 | 40 min | `make_signature`, `compile_question`, then `_Builder` (the core); print `compile_question(q).rule` for a few questions alongside |
| 5 | `batching.py` | 66 | 10 min | how a family's values become tensors (example in `family_params`) |
| 6 | `interpretation.py` | 89 | 20 min | `base_interpretation`, `ground`, `per_question` (why `_Recorder` exists) |
| 7 | `ground_truth.py` | 105 | 15 min | the same interpretation from the annotations: what "the formulas are satisfied" means |
| 8 | `networks.py` | 86 | 10 min | `ConceptNet`, `Has` / `Same`, `Relation` (rule vs MLP) |
| 9 | `evaluation.py` | 65 | 10 min | how QA replies are read from `predict` |
| 10 | `train.py` | 164 | 20 min | `load_split`, then the step loop in `main` |
| 11 | `prepare_crops.py` | 60 | 5 min | optional: how the input crops are cut |

`ground_truth.py` comes before `networks.py` so that the formulas are seen satisfied with the true values
before the networks replace them.

Syntax (`vocabulary`, `programs`, `formulas`) doesn't use torch. Data is read from `notebooks/datasets/`;
runs are saved to `notebooks/examples/CLEVR/results/clevr_qa/`.
