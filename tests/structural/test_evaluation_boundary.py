"""The reusable evaluation framework depends only on its forms and general libraries."""

import tomllib
from pathlib import Path

import pytest

from devtools.architecture.structural import discover_python_modules, evaluation_imports

ROOT = Path(__file__).resolve().parents[2]
ENTRY_MODULE, ENTRY_NAME = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"][
    "scripts"
]["alphalattice"].split(":")


def forbidden_imports(root):
    sources = {path.name: path.read_text("utf-8") for path in root.rglob("*.py")}
    return evaluation_imports(sources, f"{ENTRY_MODULE}:{ENTRY_NAME}")


def test_evaluation_imports_no_product_or_private_implementation(tmp_path):
    root = ROOT / "src/devtools/evaluation"
    assert any(root.glob("*.py"))
    assert forbidden_imports(root) == []
    r = tmp_path
    m = r / "devtools/evaluation/forms.py"
    assert discover_python_modules((m,), source_root=r)[0][1] == "evaluation.forms"
    m = r / "devtools/evaluation_extra/forms.py"
    assert discover_python_modules((m,), source_root=r)[0][1] == "devtools.evaluation_extra.forms"


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
