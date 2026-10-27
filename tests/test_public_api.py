"""Tests for the public API exported by the package __init__ files."""
import importlib

import pytest

import sltn

PACKAGES = [
    "sltn",
    "sltn.fol",
    "sltn.tools",
    "sltn.interpretation",
    "sltn.interpretation.operators",
    "sltn.signature",
    "sltn.parse",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_star_import_works_and_all_names_exist(name):
    # "CrispnessScheduler" was in sltn.tools.__all__ but not imported.
    module = importlib.import_module(name)
    assert [n for n in module.__all__ if not hasattr(module, n)] == []
    exec(f"from {name} import *", {})


def test_version_matches_package_metadata():
    # sltn had no __version__.
    from importlib.metadata import version
    assert sltn.__version__ == version("sltn")


def test_structural_relation_and_term_formula_are_exported():
    # StructuralRelation was not exported, and sltn.fol omitted TermFormula.
    from sltn.fol.formula import StructuralRelation, TermFormula
    assert sltn.StructuralRelation is StructuralRelation
    assert sltn.fol.StructuralRelation is StructuralRelation
    assert sltn.fol.TermFormula is TermFormula
