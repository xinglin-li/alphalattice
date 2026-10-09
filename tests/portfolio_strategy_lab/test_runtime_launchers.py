"""The installed entry composes product owners without checkout script imports."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from alphalattice.control.product_host.composition import saved_object_readback
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    publish_research_workspace_manifest,
)


def test_runtime_launchers_do_not_import_checkout_scripts_or_test_harnesses() -> None:
    """Requirement: runtime launch and saved-object readback need no checkout harness."""
    folder = Path(saved_object_readback.__file__).parent
    for name in (
        "entry",
        "web_launcher",
        "model_sandbox",
        "saved_object_readback",
        "retrieval_pack_setup",
        "evidence_authority_setup",
    ):
        tree = ast.parse((folder / f"{name}.py").read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        assert not any(module.split(".")[0] in {"scripts", "tests"} for module in modules), name


def test_product_probe_reopens_a_workspace_and_reads_saved_object_indexes(tmp_path: Path) -> None:
    """Requirement: sandbox readback starts actual product sessions with no harness."""
    harvest = tmp_path / "harvest"
    workspace = harvest / "workspace"
    publish_research_workspace_manifest(workspace, ResearchWorkspaceManifest.research_only("probe"))
    out = tmp_path / "readback.json"
    work = tmp_path / "copies"
    saved_object_readback.probe(harvest, work, out)
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["workspaces"] == 1
    by_op = {row["op"]: row for row in receipt["rows"]}
    assert set(by_op) == {"OPEN", "TASKS", "RESULTS", "FEATURE_TRIALS"}
    assert {op: row["verdict"] for op, row in by_op.items()} == {
        "OPEN": "OPENS",
        "TASKS": "OPENS",
        "FEATURE_TRIALS": "OPENS",
        "RESULTS": "REFUSES research_workspace.strategy_not_installed",
    }
    assert not work.exists()
    assert (workspace / "research-workspace.json").is_file()
