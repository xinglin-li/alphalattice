from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from alphalattice.kernel.shared_kernel import spawn

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

HELPER_MODULES = frozenset({"broad_ensemble_research_closure.py", "model_sandbox.py"})
"""Modules under `scripts/` that other scripts import instead of starting."""
PARSER_BUILT_ELSEWHERE = frozenset(
    {
        "run_alphalattice.py",
        "run_local_portfolio_web.py",
        "create_retrieval_environment.py",
        "install_retrieval_pack.py",
        "materialize_evidence_cro_authority.py",
        "native_research.py",
    }
)
"""Entry points whose parser is the product's own (`cli.py`), so `--help` is their probe."""

ENTRY_POINTS_WITHOUT_A_PARSER = frozenset(
    {
        "configure_playpen_dev.py",
        "run_playpen_precommit.py",
    }
)
"""Entry points that take no flags, so `--help` is not a startup probe for them.

They are imported instead. Membership is pinned because moving a command in or
out of this set is a judgement about what its supported invocation is, and the
probe that covers it changes with the answer.
"""


@dataclass(frozen=True)
class ScriptModule:
    """One `scripts/*.py`, classified from its own source."""

    name: str
    executable: bool
    parser: bool
    imports_product: bool
    inserts_source_root: bool


def _is_main_guard(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


def _builds_a_parser(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "ArgumentParser")
            or (isinstance(node.func, ast.Name) and node.func.id == "ArgumentParser")
        )
        for node in ast.walk(tree)
    )


def _imports_the_product(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("alphalattice"):
            return True
        if isinstance(node, ast.Import) and any(
            alias.name.startswith("alphalattice") for alias in node.names
        ):
            return True
    return False


def _classify(path: Path) -> ScriptModule:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    return ScriptModule(
        name=path.name,
        executable=any(_is_main_guard(node) for node in tree.body),
        parser=_builds_a_parser(tree) or path.name in PARSER_BUILT_ELSEWHERE,
        imports_product=_imports_the_product(tree),
        inserts_source_root="sys.path.insert" in source and '"src"' in source,
    )


INVENTORY: tuple[ScriptModule, ...] = tuple(
    _classify(path) for path in sorted(SCRIPTS.glob("*.py"))
)


def _startup_command(module: ScriptModule) -> list[str]:
    path = SCRIPTS / module.name
    if module.executable and module.parser:
        return [sys.executable, str(path), "--help"]
    # `sys.path[0]` is the scripts directory, exactly as it is when the file is
    # started directly; the module's own bootstrap is what has to find `src`.
    return [
        sys.executable,
        "-c",
        f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); import {path.stem}",
    ]


def _startup_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment["ALPHALATTICE_NETWORK_DISABLED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


@pytest.mark.parametrize("module", INVENTORY, ids=[module.name for module in INVENTORY])
def test_every_script_starts_without_an_inherited_import_path(module: ScriptModule) -> None:
    """requirement: a supported command runs under this workspace and nothing else."""

    with TemporaryDirectory(prefix="entry-point-startup-") as directory:
        completed = spawn.run(
            _startup_command(module),
            cwd=directory,
            env=_startup_environment(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    assert completed.returncode == 0, f"{module.name}: {completed.stderr[-2000:]}"
    if module.executable and module.parser:
        assert "usage:" in completed.stdout, module.name


def test_every_product_importing_script_binds_this_tree_as_its_import_root() -> None:
    """Every product importing script binds this tree as its import root."""

    unbound = sorted(
        module.name
        for module in INVENTORY
        if module.imports_product and not module.inserts_source_root
    )
    assert unbound == []


@pytest.mark.parametrize(
    "case", ["inventory", "configure-child", "review-child", "authority-child"]
)
def test_the_entry_point_inventory_separates_commands_from_helpers(
    case, monkeypatch, tmp_path
) -> None:
    """requirement: every script is covered, by the probe its invocation admits."""

    if case == "authority-child":
        from types import SimpleNamespace

        from scripts import materialize_post_governance_strategy_authority as authority

        calls = []

        def fake(command, **kwargs):
            calls.append((command, kwargs))
            value = str(tmp_path) if len(calls) == 1 else authority.POST_OBSERVED_RESEARCH_SHA
            return SimpleNamespace(returncode=0, stdout=value)

        monkeypatch.setattr(authority.spawn, "run", fake)
        assert authority._repository_root(tmp_path) == tmp_path.resolve()
        assert len(calls) == 2 and all(call[0][0] == "git" for call in calls)
        return

    if case == "review-child":
        from types import SimpleNamespace

        from scripts import playpen_review_context as context

        calls = []
        monkeypatch.setattr(
            context.spawn,
            "run",
            lambda *a, **kw: calls.append((a, kw)) or SimpleNamespace(stdout="answer\n"),
        )
        assert context._git("status") == "answer"
        assert calls[0][0][0][0] == "git" and calls[0][1]["check"]
        return

    if case == "configure-child":
        from types import SimpleNamespace

        from scripts import configure_playpen_dev as configure

        (tmp_path / ".githooks").mkdir()
        (tmp_path / ".githooks/pre-commit").touch()
        calls = []
        monkeypatch.setattr(configure, "PLAYPEN_ROOT", tmp_path)
        monkeypatch.setattr(
            configure.spawn,
            "run",
            lambda *a, **kw: calls.append((a, kw)) or SimpleNamespace(stdout=".githooks\n"),
        )
        assert configure.main() == 0 and len(calls) == 2
        assert all(call[0][0][0] == "git" and call[1]["check"] for call in calls)
        return

    assert INVENTORY
    helpers = frozenset(module.name for module in INVENTORY if not module.executable)
    assert helpers == HELPER_MODULES
    without_a_parser = frozenset(
        module.name for module in INVENTORY if module.executable and not module.parser
    )
    assert without_a_parser == ENTRY_POINTS_WITHOUT_A_PARSER
