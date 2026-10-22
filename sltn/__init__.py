"""
sltn — Structural Logic Tensor Networks
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("sltn")   # single source of truth: pyproject.toml
except PackageNotFoundError:        # running from a source tree that is not installed
    __version__ = "0+unknown"

from sltn.interpretation.core.type import Type, Real, Bool, Int, PositiveReal, Complex, product_type
from sltn.interpretation.core.tensor import Tensor, Axis
from sltn.signature import (
    Symbol, Sort, Dimension, ConstantSymbol, VariableSymbol, FunctionSymbol, PredicateSymbol,
    StructuralVariableSymbol, StructuralRelationSymbol, Signature,
)
from sltn.fol.formula import (
    Formula, TermFormula, FormulaAxisSelect, AtomicFormula,
    And, Or, Not, Implies, Equivalent,
    Forall, Exists, KB, StructuralForall, StructuralExists, StructuralRelation,
)
from sltn.fol.term import Term, Variable, Constant, StructuralVariable, FunctionApplication, TermAxisSelect
from sltn.interpretation import Interpretation, Logic, Assignment
from sltn.interpretation.interpretation import TensorCallable
from sltn.tools.optim import kb_backward, kb_describe, kb_eval
from sltn.parse import parse_term_string, parse_formula_string

__all__ = [
    "__version__",
    # Core
    "Tensor",
    "Axis",
    "Type",
    "Real",
    "Bool",
    "Int",
    "PositiveReal",
    "Complex",
    "product_type",
    # Signature
    "Symbol",
    "Sort",
    "Dimension",
    "ConstantSymbol",
    "VariableSymbol",
    "FunctionSymbol",
    "PredicateSymbol",
    "StructuralVariableSymbol",
    "StructuralRelationSymbol",
    "Signature",
    # Interpretation
    "Interpretation",
    "Logic",
    "Assignment",
    "TensorCallable",
    # Formulas
    "Formula",
    "TermFormula",
    "FormulaAxisSelect",
    "AtomicFormula",
    "And",
    "Or",
    "Not",
    "Implies",
    "Equivalent",
    "Forall",
    "Exists",
    "KB",
    "StructuralForall",
    "StructuralExists",
    "StructuralRelation",
    # Terms
    "Term",
    "Variable",
    "Constant",
    "StructuralVariable",
    "FunctionApplication",
    "TermAxisSelect",
    # Parse
    "parse_term_string",
    "parse_formula_string",
    # Training utilities
    "kb_backward",
    "kb_describe",
    "kb_eval",
    # "CrispnessScheduler",
]
