# sLTN — Structural Logic Tensor Networks

**sLTN** is a PyTorch framework for neural-symbolic learning based on differentiable first-order logic. It extends [LTNtorch](https://github.com/logictensornetworks/LTNtorch) with a typed symbolic interface, string-parsed first-order logic formulas, named tensor axes, and structural dimensions for reasoning over temporal, sequential, graph-based, and other structured data.

A detailed description is available in the companion paper: [arXiv:268.11136](https://arxiv.org/abs/2608.11136).

## Key Features

- **Formula domain-specific language.** Write first-order logic formulas and knowledge bases as strings using compact notation close to standard logic.
- **Typed symbolic interface.** Define a typed vocabulary through a `Signature`, including sorts, dimensions, predicates, functions, variables, and structural relations.
- **Separation of syntax and grounding.** Declare symbolic objects independently of tensors, then bind them through an `Interpretation` to PyTorch tensors, modules, or callables.
- **Named and typed tensor axes.** Track each tensor axis by name and role, distinguishing variable, structural, and domain axes.
- **Structural dimensions.** Represent axes such as time, sequence position, video frames, or graph nodes, and quantify over them independently from first-order variables.
- **Differentiable Real Logic semantics.** Preserve the fuzzy-logic semantics of LTN/LTNtorch, including fuzzy connectives, product logic, p-mean aggregation, guarded quantification, and diagonal quantification.
- **Training utilities.** optional `torchjd` support for multi-objective satisfaction of knowledge bases, via `kb_describe`, `kb_eval`, and `kb_backward` API.


## Citation

If you use sLTN in your research, please cite the accompanying preprint:

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

sLTN has been developed, starting from the original implentation of LTNtorch, by Davide Rinaldi (Nokia Bell Labs) and Luciano Serafini (FBK).


## Installation

Requires Python 3.10+.

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install sLTN in editable mode from the repository root:
```
uv pip install -e .                   # core sLTN (torch)
```

Optional extras can be installed from the same directory:

```
uv pip install -e ".[multiobjective]" # + torchjd for multi-objective training
uv pip install -e ".[notebooks]"      # + tutorial notebook dependencies (JupyterLab, etc.)
uv pip install -e ".[test]"           # + pytest
uv pip install -e ".[examples]"       # + examples
uv pip install -e ".[multiobjective,notebooks,examples,test]" # complete installation
```

## Quickstart

The following example illustrates the separation between symbolic declaration and tensor-level grounding.

```python
import torch
import torch.nn as nn
import sltn
from sltn import Signature, Interpretation
from sltn import kb_backward, kb_describe

# --- 1. Declare the domain and symbols --------------------------------
sig = Signature("animals")
sig.sort("Animal")
sig.predicate("isDog", ["Animal"])
sig.predicate("likes", ["Animal", "Animal"])
sig.variable("x", Animal), sig.variable("y", Animal)

# --- 2. Bind values ---------------------------------------------------
class Likes(nn.Module):
    """A binary predicate net must take one argument per predicate slot."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))

interp = Interpretation(sig)
interp["Animal"] = sltn.Type("Animal", shape=(4,))  # 4-dim feature vectors
interp["x"]      = torch.randn(50, 4)
interp["y"]      = torch.randn(50, 4)
interp["isDog"]  = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
interp["likes"]  = Likes()
```

In this design, the Signature specifies what symbolic objects exist, while the Interpretation specifies how they are grounded computationally. The signature is independent of tensors and neural modules, which allows formulas to be parsed into reusable syntax trees and re-evaluated across different batches or interpretations.

### Structural dimensions

Some domains require reasoning over structured axes rather than only over independent individuals. Examples include temporal signals, ordered sequences, video data, and graph-structured representations. `sLTN` exposes such axes as structural dimensions declared in the signature.

```python
sig = Signature("seq")
sig.sort("S")
sig.dimension("T")
sig.predicate("A", ["S"], input_dims=["T"], output_dims=["T"])
sig.structural_relation("next", ["T", "T"])
sig.structural_variable("t","T")

# "A is preserved across consecutive steps"
phi = sig.parse("forall t, t1 | next(t, t1): (A(x[t])[t] -> A(x[t1])[t1])")
```

Here, `"T"` denotes a structural dimension, while next defines a structural relation over that dimension. The parsed formula states that if A holds at one structural position, then it should also hold at the next related position.

## Running tests

```bash
source .venv/bin/activate
pytest -v tests/
```

## Generating datasets

The datasets used in `notebooks/ltn_examples/` can be downloaded from the original LTN repository with:

```bash
python notebooks/datasets/datasets_generation/download_original_datasets.py
```

## Notebooks


| Folder                      | Contents                                                                                                                                                                                                       |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `notebooks/tutorials_sltn/` | Step-by-step tutorials, in order: (1) signatures and interpretations, (2) FOL formulas — connectives and quantifiers, (3) fuzzy operators and gradients, (4) knowledge bases and learning, (5) structural LTN. |
| `notebooks/ltn_examples/`   | Classic LTN benchmark tasks reimplemented in sLTN: classification, regression, clustering, embedding learning.                                                                                                 |
| `notebooks/examples/`       | Worked examples: the running example (`0_Running_example.ipynb`) used in companion paper, combining the full tutorial series, multi-digit addition, and LTLf-style temporal reasoning in sLTN.                 |


## Note on AI-Assisted Development

Large language models (LLMs) were used to assist with selected development tasks, including tests, tutorials, and bug fixing.

