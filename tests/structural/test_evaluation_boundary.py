"""The reusable evaluation framework depends only on its forms and general libraries."""

import ast
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY_MODULE, ENTRY_NAME = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"][
    "scripts"
]["alphalattice"].split(":")
ALLOWED = {ENTRY_MODULE: {ENTRY_NAME}}


def forbidden_imports(root):
    failures = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level != 1:
                        failures.append((path.name, node.lineno, "outside package"))
                    continue
                if node.module in ALLOWED and all(
                    alias.name in ALLOWED[node.module] for alias in node.names
                ):
                    continue
                names = [node.module or ""]
            for name in names:
                if not (
                    name.split(".")[0] in sys.stdlib_module_names | {"yaml"}
                    or name.startswith("devtools.evaluation.")
                ):
                    failures.append((path.name, node.lineno, name))
            if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id in {"__import__", "eval", "exec"})
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
            ):
                failures.append((path.name, node.lineno, "dynamic import"))
    return failures


def test_evaluation_imports_no_product_or_private_implementation():
    root = ROOT / "src/devtools/evaluation"
    assert any(root.glob("*.py"))
    assert forbidden_imports(root) == []


@pytest.mark.parametrize(
    "source,refused",
    [
        ("import json\nimport yaml\nfrom .forms import Scenario\n", False),
        (f"from {ENTRY_MODULE} import {ENTRY_NAME}\n", False),
        (f"from {ENTRY_MODULE} import private_helper\n", True),
        (f"import {ENTRY_MODULE}\n", True),
        ("import alphalattice.control.product_host\n", True),
        ("from private_dataset import runner\n", True),
        ("from ..architecture import identity_closures\n", True),
        ("__import__('owner')\n", True),
    ],
)
def test_evaluation_import_boundary_refuses_internal_and_dynamic_readers(tmp_path, source, refused):
    (tmp_path / "module.py").write_text(source, encoding="utf-8")
    assert bool(forbidden_imports(tmp_path)) == refused
