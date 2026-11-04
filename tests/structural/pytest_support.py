"""Shared external pytest checkout for routing and real-root layer contract controls."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def pytest_checkout(tmp_path: Path, source: Path) -> tuple[Path, dict[str, str]]:
    """The real suite support, reached by pytest in a disposable external checkout."""
    from datetime import UTC, datetime
    from uuid import UUID

    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceDecisionUpdate,
        ResearchWorkspaceManifest,
        publish_research_workspace_manifest,
    )
    from alphalattice.control.product_host.publication.goals import GoalStore
    from alphalattice.interface.local_application.goals import (
        Goal,
        GoalCompletion,
        GoalDeclaration,
        GoalSubmission,
    )
    from alphalattice.interface.local_application.native_setup import declare_project

    checkout = tmp_path / "checkout"
    tests = checkout / "tests"
    tests.mkdir(parents=True)
    shutil.copyfile(source / "tests" / "conftest.py", tests / "conftest.py")
    scripts = checkout / "scripts"
    scripts.mkdir()
    (scripts / "build_local_web_ui.py").write_text(
        "# This collection-only fixture has no Local Web consumers or source.\n"
        "# Complete the controller's prerequisite with a current-output probe.\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
        newline="\n",
    )
    (checkout / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = '-p no:cacheprovider'\n"
        "testpaths = ['tests']\njunit_family = 'xunit1'\n",
        encoding="utf-8",
        newline="\n",
    )
    (tests / "unit").mkdir()
    (tests / "unit" / "test_collection_probe.py").write_text(
        "import json\nimport os\nfrom pathlib import Path\n"
        "CHECKOUT = Path(__file__).resolve().parents[2]\n"
        "def test_temp(tmp_path, request, record_property):\n"
        "    assert 'golden_cache' not in request.fixturenames\n"
        "    assert Path.cwd() == CHECKOUT\n"
        "    record_property('temporary_paths', json.dumps({\n"
        "        'test': str(tmp_path.resolve()), 'discovery': str(Path.cwd()),\n"
        "        'backup': os.environ['ALPHALATTICE_BACKUP_ROOT'],\n"
        "    }))\n",
        encoding="utf-8",
        newline="\n",
    )
    (tests / "integration").mkdir()
    (tests / "integration" / "test_discovery_probe.py").write_text(
        "from pathlib import Path\nimport pytest\n"
        "CHECKOUT = Path(__file__).resolve().parents[2]\n"
        "(CHECKOUT / 'collected').write_text('collected', encoding='utf-8')\n"
        "def test_discovery(tmp_path):\n"
        "    from alphalattice.control.product_host.composition.first_use_road "
        "import completed_first_use\n"
        "    assert completed_first_use(tmp_path / 'empty') is None\n"
        "    with pytest.raises(AssertionError, match='project_outside_test_root'):\n"
        "        completed_first_use(CHECKOUT / 'workspaces' / 'completed')\n",
        encoding="utf-8",
        newline="\n",
    )
    declare_project(checkout, "codex")
    workspace = checkout / "workspaces" / "completed"
    manifest = ResearchWorkspaceManifest.create(
        workspace_id="caller-completed",
        default_strategy_package_id="caller-package",
        default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
        strategy_artifacts=(),
        decision_updates=(
            ResearchWorkspaceDecisionUpdate(
                strategy_package_id="caller-package",
                strategy_package_hash="a" * 64,
                checkpoint_hash="b" * 64,
            ),
        ),
    )
    publish_research_workspace_manifest(workspace, manifest)
    now = datetime(2026, 10, 19, tzinfo=UTC)
    goal = Goal.seal(
        goal_id=UUID(int=1),
        workspace_id=manifest.workspace_id,
        revision=1,
        recorded_at=now,
        intent_registered_at=now,
        submitted_by="fixture",
        change_reason="Completed caller fixture",
        state="COMPLETE",
        declaration=GoalDeclaration(
            title="First use",
            objective="Caller book",
            kind="FIRST_USE",
            criteria=({"criterion_id": "book", "text": "Caller result"},),
        ),
        submission=GoalSubmission(
            outcome="NOT_ACHIEVED",
            summary="Caller completed",
            criteria=({"criterion_id": "book", "answer": "NOT_ASSESSED", "note": "Fixture"},),
        ),
        completion=GoalCompletion(checked_at=now, request_count=0),
    )
    GoalStore(workspace / "artifacts", manifest.workspace_id).publish(goal, None)
    external = tmp_path / "system-temp"
    external.mkdir()
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTEST_") and key != "ALPHALATTICE_PIPELINE_REQUIRED_TESTS"
        if key != "ALPHALATTICE_REACH_FLOOR"
        if key != "ALPHALATTICE_CHANGED_PATHS" or not os.environ.get("ALPHALATTICE_REACH_FLOOR")
    }
    environment.update(
        PYTHONPATH=os.pathsep.join((str(source), str(source / "src"), str(source / "scripts"))),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        TMPDIR=str(external),
        TEMP=str(external),
        TMP=str(external),
    )
    return checkout, environment


AUDIT_HOOK_SAMPLE = """
from devtools.testing import audit as observe
seen=[]
with observe.subscribe(lambda event,args: seen.append(('a',event))):
    with observe.subscribe(lambda event,args: seen.append(('b',event))):
        __import__('sys').audit('test.contract.first')
with observe.subscribe(lambda event,args: seen.append(('c',event))):
    __import__('sys').audit('test.contract.second')
assert seen.count(('a','test.contract.first')) == 1
assert seen.count(('b','test.contract.first')) == 1
assert seen.count(('c','test.contract.second')) == 1
assert ('a','test.contract.second') not in seen
import pytest
calls=[]
def stopped(event,args):
    if event == 'test.contract.failure':
        calls.append(event)
        raise ValueError('stopped')
with pytest.raises(ValueError):
    with observe.subscribe(stopped):
        __import__('sys').audit('test.contract.failure')
__import__('sys').audit('test.contract.failure')
assert calls == ['test.contract.failure']
"""


def descriptor_provenance(tmp_path: Path) -> None:
    import pytest

    from devtools.testing.contracts import LayerContract

    entry = tmp_path / "owned"
    entry.write_text("data")
    tree = tmp_path / "tree"
    (tree / "child").mkdir(parents=True)
    (tree / "child" / "file").write_text("data")
    guard = LayerContract("unit", tmp_path, ({"kind": "module", "name": "sqlite3"},))
    unknown = os.open(entry, os.O_RDONLY)
    raw_dup2 = os.dup2
    try:
        with guard.active():
            with pytest.raises(AssertionError, match="unproved file descriptor"):
                guard.path(unknown)
            if os.name == "posix":
                with pytest.raises(AssertionError, match="unproved file descriptor"):
                    os.stat(unknown)
                with pytest.raises(AssertionError, match="unproved file descriptor"):
                    os.open("file", os.O_RDONLY, dir_fd=unknown)
            known = os.open(entry, os.O_RDONLY)
            try:
                if os.name == "posix":
                    guard.path(known)
                    assert os.stat(known).st_size == 4
                    os.dup2(unknown, known)
                    with pytest.raises(AssertionError, match="unproved file descriptor"):
                        guard.path(known)
                else:
                    with pytest.raises(AssertionError, match="unproved file descriptor"):
                        guard.path(known)
            finally:
                os.close(known)
            with pytest.raises(AssertionError, match="unproved file descriptor"):
                guard.path(known)
            if os.name == "posix":
                stale = os.open(entry, os.O_RDONLY)
                os.fdopen(stale, "rb").close()
                with entry.open("rb") as reopened:
                    if reopened.fileno() != stale:
                        raw_dup2(reopened.fileno(), stale)
                    try:
                        with pytest.raises(AssertionError, match="unproved file descriptor"):
                            guard.path(stale)
                    finally:
                        if reopened.fileno() != stale:
                            os.close(stale)
                root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    child_fd = os.open("tree/child", os.O_RDONLY | os.O_DIRECTORY, dir_fd=root_fd)
                    try:
                        assert os.stat("file", dir_fd=child_fd).st_size == 4
                        with os.scandir(child_fd) as entries:
                            assert [item.name for item in entries] == ["file"]
                        os.unlink("file", dir_fd=child_fd)
                    finally:
                        os.close(child_fd)
                    with pytest.raises(AssertionError, match="outside temporary root"):
                        os.open("../escape", os.O_WRONLY | os.O_CREAT, dir_fd=root_fd)
                finally:
                    os.close(root_fd)
                import shutil

                from devtools.testing.audit import ResourceObservation, subscribe

                observation = ResourceObservation()
                with subscribe(observation.note):
                    shutil.rmtree(tree)
                assert observation.files and not tree.exists()
    finally:
        os.close(unknown)
