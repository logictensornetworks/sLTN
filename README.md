# sLTN — Structural Logic Tensor Networks

[![arXiv](https://img.shields.io/badge/arXiv-2608.11136-b31b1b.svg)](https://arxiv.org/abs/2608.11136)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

**sLTN** (*Structural Logic Tensor Networks*) is a PyTorch framework for differentiable neuro-symbolic learning and reasoning with first-order logic. Building on Real Logic and [LTNtorch](https://github.com/logictensornetworks/LTNtorch), sLTN introduces a clean separation between logical syntax and neural grounding, a compact formula language, named tensor axes, and **structural dimensions** for reasoning natively over sequences, time series, grids, videos, and graphs.

For the formal foundations, dimensional calculus, and experimental evaluation, see the companion paper:
**[sLTN: Structural Logic Tensor Networks](https://arxiv.org/abs/2608.11136)** (*Davide Rinaldi & Luciano Serafini*).

---

## Highlights

- **Strict separation of syntax and semantics.** Declare sorts, dimensions, symbols, and syntactic macros in a pure-Python `Signature` (no PyTorch dependency). Bind symbols to tensors, `nn.Module`s, or callables in an `Interpretation`.
- **Readable formula DSL.** Write formulas and knowledge bases directly as strings (`forall x, y: (isDog(x) & likes(x, y)) -> happy(y)`), parsed into reusable, inspectable ASTs with static sort and axis validation.
- **Named axes & automatic alignment.** Every intermediate `sltn.Tensor` tracks `variable`, `structural`, and `domain` axes by name, aligning and broadcasting operands automatically without manual `unsqueeze` or `permute` bookkeeping.
- **First-class structural dimensions.** Model structured index spaces—such as time steps, token positions, image patches, or graph nodes—independently from batch individuals, complete with structural relations (`next(t, t1)`), axis slicing (`x[t][t=0]`), and per-dimension fuzzy aggregators.
- **PyTorch-native `Interpretation`.** `Interpretation` subclasses `torch.nn.Module`: grounded networks and `nn.Parameter`s are registered automatically, so `.parameters()`, `.state_dict()`, `.train()` / `.eval()`, `.to(device)`, and PyTorch Lightning work out of the box.
- **Differentiable Real Logic & multi-objective training.** Full suite of exact and numerically stable fuzzy connectives and aggregators (`AndProd`, `ImpliesReichenbach`, `AggregPMeanError`, `Logic.classical()`, …), plus `kb_describe`, `kb_eval`, and `kb_backward` with optional Jacobian-descent multi-objective optimization via [`torchjd`](https://torchjd.org/).

---

## Installation

Requires **Python 3.10+** and **PyTorch**.

Using [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip` from the package root:

```bash
uv venv
source .venv/bin/activate

# Core library
uv pip install -e .
```

Optional extras can be installed according to your workflow:

```bash
uv pip install -e ".[multiobjective]"  # Jacobian descent (torchjd) for multi-objective KB training
uv pip install -e ".[lightning]"       # PyTorch Lightning integration
uv pip install -e ".[notebooks]"       # JupyterLab, Matplotlib, torchvision, scikit-learn, etc.
uv pip install -e ".[examples]"        # Extra dependencies for paper examples (ltlf2dfa, python-sat, torchmetrics)
uv pip install -e ".[test]"            # pytest suite

# Full development environment
uv pip install -e ".[multiobjective,lightning,notebooks,examples,test]"
```

---

## Quickstart

### 1. First-order logic & training a Knowledge Base

A `Signature` defines *what* symbols exist; an `Interpretation` defines *how* they are grounded in PyTorch. Formulas can be parsed once and evaluated across minibatches or multiple interpretations.

```python
import torch
import torch.nn as nn
import sltn
from sltn import Signature, Interpretation, kb_describe, kb_backward, kb_eval

# 1. Declare the symbolic vocabulary (pure syntax)
sig = Signature("animals")
sig.sort("Animal")
sig.variable("x", "Animal")
sig.variable("y", "Animal")
sig.predicate("isDog", ["Animal"])
sig.predicate("likes", ["Animal", "Animal"])

kb = sig.parse("""
{
    forall x: isDog(x),
    forall x, y: (isDog(x) -> likes(x, y))
}
""")

# 2. Ground symbols in PyTorch (Interpretation is an nn.Module)
class LikesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, a, b):
        return self.net(torch.cat([a, b], dim=-1))

interp = Interpretation(sig)
interp["Animal"] = sltn.Type("Animal", shape=(4,))          # 4-D feature vectors
interp["isDog"]  = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
interp["likes"]  = LikesNet()

# 3. Train grounded neural predicates to satisfy the Knowledge Base
kb_dict = kb_describe(kb)
optimizer = torch.optim.Adam(interp.parameters(), lr=1e-2)

for step in range(50):
    interp["x"] = torch.randn(32, 4)                        # rebind batch data per step
    interp["y"] = torch.randn(32, 4)

    optimizer.zero_grad()
    stats = kb_backward(optimizer, interp, kb_dict)         # or aggregator="upgrad" with torchjd
    optimizer.step()

print(kb_eval(interp, kb_dict))
```

### 2. Reasoning over structural dimensions

When individuals carry internal structure—such as sequences of length `T`—declare a `Dimension` and a `StructuralVariable` ranging over it. Numbered and primed aliases (`t1`, `t2`, `t'`) are resolved automatically over the same dimension.

```python
import torch
import sltn
from sltn import Signature, Interpretation

sig = Signature("temporal")
sig.sort("Frame")
sig.dimension("T")
sig.structural_variable("t", "T")

sig.variable("video", "Frame", dims=["T"])                  # batch of sequences: [B, T, *Frame]
sig.predicate("Active", ["Frame"])                          # pointwise over T
sig.predicate("Detect", ["Frame"], input_dims=["T"], output_dims=["T"])  # sequence-aware over T
sig.structural_relation("next", ["T", "T"])                 # temporal adjacency mask

# "Active holds at step 0, and Detect is preserved across consecutive steps"
phi = sig.parse(
    "forall video: (Active(video[t][t=0]) & "
    "forall t, t1 | next(t, t1): (Detect(video[t])[t] -> Detect(video[t1])[t1]))"
)

T_len = 6
interp = Interpretation(sig)
interp["Frame"]  = sltn.Type("Frame", shape=(16,))
interp["video"]  = torch.randn(8, T_len, 16)
interp["Active"] = torch.nn.Sequential(torch.nn.Linear(16, 1), torch.nn.Sigmoid())
interp["Detect"] = lambda x: torch.sigmoid(x.mean(dim=-1, keepdim=True))
interp["next"]   = torch.diag(torch.ones(T_len - 1), diagonal=1)

satisfaction = interp(phi)                                  # sltn.Tensor of shape (1,)
```

Structural quantifiers over `T` use copies of the first-order `Forall`/`Exists` by default; override them per dimension with `interp.logic["forall,T"] = ...`.

---

## Package Architecture

`sltn` is organized into five subpackages with a strict unidirectional boundary—syntax modules (`signature`, `fol`, `parse`) never import `interpretation`, `tools`, or `torch`:

| Subpackage | Role |
| :--- | :--- |
| `sltn.signature` | Symbolic declarations (`Signature`, `Sort`, `Dimension`, constants, variables, structural variables/relations, functions, predicates, macros). |
| `sltn.fol` | Pure syntactic AST nodes for terms, formulas, quantifiers, knowledge bases (`KB`), and capture-avoiding macro expansion. |
| `sltn.parse` | Tokenizer, recursive-descent/infix parser, and static validation pipeline (sort checking, axis calculus, quantifier & clause checks). |
| `sltn.interpretation` | `Interpretation` (`nn.Module`), annotated `Tensor` & `Axis` alignment, `Type` descriptors, `Logic`, and fuzzy operators. |
| `sltn.tools` | Training and inspection utilities (`kb_describe`, `kb_eval`, `kb_backward`, `cache_benchmark`). |

---

## Tutorials & Examples

| Directory | Contents |
| :--- | :--- |
| `notebooks/tutorials_sltn/` | Progressive 5-part tutorial series: **(1)** Signatures & Interpretations, **(2)** FOL Formulas, Connectives & Quantifiers, **(3)** Fuzzy Operators & Gradients, **(4)** Knowledge Bases & Learning, **(5)** Structural LTN. |
| `notebooks/examples/` | Companion-paper and structured reasoning tasks |
| `notebooks/ltn_examples/` | Classic LTN benchmarks reimplemented in sLTN: binary, multi-class, and multi-label classification, semi-supervised pattern recognition, regression, clustering, and embedding learning. |

To download the datasets required by `notebooks/ltn_examples/`:

```bash
python notebooks/datasets/datasets_generation/download_original_datasets.py
```

---

## Running Tests

```bash
pytest -q tests/
```

---

## Citation

If you use **sLTN** in your research or software, please cite:

```bibtex
@misc{rinaldi2026sltn,
  title         = {sLTN: Structural Logic Tensor Networks},
  author        = {Rinaldi, Davide and Serafini, Luciano},
  year          = {2026},
  eprint        = {2608.11136},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2608.11136}
}
```

---

## Acknowledgements & License

**sLTN** was conceived and developed by **Davide Rinaldi** (Nokia Bell Labs) and **Luciano Serafini** (Fondazione Bruno Kessler). It builds on the original [LTNtorch](https://github.com/logictensornetworks/LTNtorch) implementation by **Tommaso Carraro**, who also contributed to refining and finalising this library. Large language models (LLMs) were used to assist with selected development tasks, including testing, tutorials, and bug fixing.

Licensed under the [MIT License](./LICENSE).