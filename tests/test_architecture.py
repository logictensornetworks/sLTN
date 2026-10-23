"""Architecture constraints of sltn.

1. The package layout is fixed: changing it is a design decision, not a refactor.
2. Syntax (signature, fol, parse) never depends on semantics (interpretation,
   tools) or on torch. Semantics may import syntax, never the reverse.
"""
import ast
from pathlib import Path

import pytest

SLTN = Path(__file__).resolve().parent.parent / "sltn"

PACKAGES = {"signature", "fol", "parse", "interpretation", "tools"}
SYNTAX = ("signature", "fol", "parse")
FORBIDDEN_IN_SYNTAX = ("sltn.interpretation", "sltn.tools", "torch")


def _imported_modules(path: Path) -> set[str]:
    """Absolute names of every module imported by *path*, relative imports resolved."""
    package = ".".join(path.relative_to(SLTN.parent).with_suffix("").parts[:-1])
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                names.add(f"{base}.{node.module}" if node.module else base)
            else:
                names.add(node.module)
    return names


def test_top_level_packages_are_fixed():
    found = {p.name for p in SLTN.iterdir() if (p / "__init__.py").is_file()}
    assert found == PACKAGES


@pytest.mark.parametrize("package", SYNTAX)
def test_syntax_does_not_import_semantics_or_torch(package):
    violations = [
        f"{path.relative_to(SLTN.parent)} imports {name}"
        for path in sorted((SLTN / package).rglob("*.py"))
        for name in _imported_modules(path)
        if any(name == f or name.startswith(f + ".") for f in FORBIDDEN_IN_SYNTAX)
    ]
    assert not violations, "\n".join(violations)
