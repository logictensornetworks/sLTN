"""
sltn — Structural Logic Tensor Networks
"""
from __future__ import annotations

from sltn.interpretation.core.type import Type, Real, Bool, Int, PositiveReal, Complex, product_type
from sltn.interpretation.core.tensor import Tensor, Axis
from sltn.signature import (
    Symbol, Sort, Dimension, ConstantSymbol, VariableSymbol, FunctionSymbol, PredicateSymbol,
    StructuralVariableSymbol, StructuralRelationSymbol, Signature,
)
from sltn.fol.formula import (
    Formula, TermFormula, FormulaAxisSelect, AtomicFormula,
    And, Or, Not, Implies, Equivalent,
    Forall, Exists, KB, StructuralForall, StructuralExists,
)
from sltn.fol.term import Term, Variable, Constant, StructuralVariable, FunctionApplication, TermAxisSelect
from sltn.interpretation import Interpretation, Logic, Assignment
from sltn.interpretation.interpretation import TensorCallable
from sltn.tools.optim import kb_backward, kb_describe, kb_eval
from sltn.parse import parse_term_string, parse_formula_string

__all__ = [
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
