"""A workspace's Panel closure and the log: the upgrade moves its closure folder onto the log byte
for byte, the closure store then reads and writes there, and one switch reads the folder again."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.data_layer import history
from alphalattice.foundation.data_layer.upgrade import Family, UpgradeRunner
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_files import (
    CLOSURE,
    FolderFiles,
    KeptFiles,
    closure_upgrade_family,
)

TEST_SOURCE = "src/alphalattice/foundation/feature_engine/panels/closure_files.py"
HEAD = "feature-ledger/operational/heads/" + "a" * 64 + ".json"
RUN = "0.2.8"
PENDING = "feature-ledger/operational/pending/" + "b" * 64 + ".json"


def _old_workspace(workspace: Path) -> tuple[PanelClosureArtifactStore, str, object]:
    """A 0.2.7 workspace's closure folder, registered on the log: a JSON child, a Parquet child
    and a ledger head pointer, written by the store before the move."""
    (workspace / "artifacts").mkdir(parents=True)
    (workspace / "research-workspace.json").write_text("{}", encoding="utf-8")
    history.held().record(
        "workspace.created", ("workspace", "w1"), {"folder": str(workspace.resolve()), "name": "w"}
    )
    store = PanelClosureArtifactStore(ArtifactResolver(workspace / "artifacts"))
    store._files = FolderFiles(store.root)  # the 0.2.7 folder, written before the move
    content = hashlib.sha256(b"map").hexdigest()
    store.publish_json(category="sector-maps", content_hash=content, payload={"map": 1})
    table = pa.table({"value": [1.0, 2.0]})
    reference = store.publish_parquet(
        category="feature-ledger/patch-rows", content_hash=content, kind="rows", table=table
    )
    store.files.write(HEAD, b'{"head_hash":"h1"}')
    return store, content, reference


def _runner(workspace: Path, subject: str = RUN) -> SimpleNamespace:
    """An upgrade run over `workspace`, as the runner hands its families: its root, log and run."""
    root, log = history.data_root(), history.held()
    return SimpleNamespace(root=root, log=log, workspaces=(workspace,), subject=subject)


def _run(workspace: Path, family: Family, subject: str) -> bool:
    """One real upgrade run of `family` over `workspace` under `subject`: True when it completed."""
    backups = {"backup": "verified"}
    root, log = history.data_root(), history.held()
    return UpgradeRunner(
        root, log, (workspace,), (family,), lambda *_: backups, subject=subject
    ).run()


def _moved(workspace: Path) -> SimpleNamespace:
    runner = _runner(workspace)
    family = closure_upgrade_family(lambda _workspace: "w1")
    family.step(runner)
    assert family.verify(runner)
    return runner


@pytest.mark.proves("feature_engine.closure_moves_onto_the_log_all_equal")
def test_the_upgrade_moves_a_workspaces_closure_onto_the_log_all_equal(tmp_path: Path) -> None:
    """The family moves every closure file equal with its workspace's hold, and a store then reads
    them on the log, as a new workspace's starts there; a linked closure file or root is refused
    by name before any body is kept."""
    workspace = tmp_path / "workspace"
    _store, content, reference = _old_workspace(workspace)
    family = closure_upgrade_family(lambda _workspace: "w1")
    assert family.holds(workspace)
    assert _run(workspace, family, RUN)  # its start recorded before its move
    assert not family.holds(workspace)
    holds = history.held().read("SELECT DISTINCT root_kind, root_id FROM live_ref")
    assert [tuple(row) for row in holds] == [("workspace", "w1")]
    store = PanelClosureArtifactStore(ArtifactResolver(workspace / "artifacts"))
    assert isinstance(store.files, KeptFiles)
    assert store.load_json(category="sector-maps", content_hash=content) == {"map": 1}
    table = store.load_parquet(category="feature-ledger/patch-rows", reference=reference)
    assert table.column("value").to_pylist() == [1.0, 2.0]
    assert store.files.read(HEAD) == b'{"head_hash":"h1"}'
    store.files.write(HEAD, b'{"head_hash":"h2"}')
    assert not family.verify(_runner(workspace))  # its own run, resumed, still checks the bytes
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    escaped = workspace / "artifacts" / "feature-panel" / "closure" / "sector-maps" / "escaped.json"
    os.symlink(outside, escaped)
    with pytest.raises(ValueError, match="closure file is an alias"):
        family.holds(workspace)
    fresh = tmp_path / "fresh"
    (fresh / "artifacts").mkdir(parents=True)
    (fresh / "research-workspace.json").write_text("{}", encoding="utf-8")
    place = {"folder": str(fresh.resolve()), "name": "fresh"}
    history.held().record("workspace.created", ("workspace", "w2"), place)
    new = PanelClosureArtifactStore(ArtifactResolver(fresh / "artifacts"))
    assert isinstance(new.files, KeptFiles)
    new.publish_json(category="sector-maps", content_hash=content, payload={"map": 3})
    assert not (fresh / "artifacts" / "feature-panel").exists() and not family.holds(fresh)
    linked = tmp_path / "linked"
    (linked / "artifacts" / "feature-panel").mkdir(parents=True)
    (linked / "research-workspace.json").write_text("{}", encoding="utf-8")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "file.json").write_bytes(b"{}")
    os.symlink(tmp_path / "elsewhere", linked / CLOSURE, target_is_directory=True)
    with pytest.raises(ValueError, match="closure root is an alias"):
        family.holds(linked)


@pytest.mark.proves("feature_engine.closure_writes_go_to_the_log_and_rollback_reads_the_folder")
def test_after_the_move_writes_go_to_the_log_across_a_restart(
    tmp_path: Path,
) -> None:
    """After the move, writes and removals stay on the log across a restart and a later run that
    includes the workspace, the folder unchanged."""
    workspace = tmp_path / "workspace"
    _old_workspace(workspace)
    FolderFiles(workspace / "artifacts" / "feature-panel" / "closure").write(PENDING, b"{}")
    _moved(workspace)
    store = PanelClosureArtifactStore(ArtifactResolver(workspace / "artifacts"))
    store.files.write(HEAD, b'{"head_hash":"h2"}')
    store.files.remove(PENDING)
    restarted = PanelClosureArtifactStore(ArtifactResolver(workspace / "artifacts")).files
    assert isinstance(restarted, KeptFiles) and restarted.read(PENDING) is None
    family = closure_upgrade_family(lambda _workspace: "w1")
    assert not family.holds(workspace)
    # the move came first, as a copy's carried cutover does: runs on this log that owe another
    # family, under the same first-start subject or a later one, keep it and complete
    for subject in (RUN, f"{RUN}/another"):
        assert _run(workspace, family, subject)
    assert restarted.read(HEAD) == b'{"head_hash":"h2"}' and not restarted.exists(PENDING)
    added = hashlib.sha256(b"added").hexdigest()
    store.publish_json(category="sector-maps", content_hash=added, payload={"map": 2})
    folder = workspace / "artifacts" / "feature-panel" / "closure"
    assert (folder / HEAD).read_bytes() == b'{"head_hash":"h1"}'
    assert not (folder / "sector-maps" / f"{added}.json").exists()
    assert store.holds("sector-maps", added) and store.files.read(HEAD) == b'{"head_hash":"h2"}'


@pytest.mark.proves("feature_engine.kept_closure_names_follow_their_writes")
def test_kept_closure_files_are_listed_by_name_and_by_their_last_write(tmp_path: Path) -> None:
    """On the log, a folder's names are sorted, by write the file rewritten last is last, and a
    removed file is gone from both; the move keeps the folder's write order, not its names'."""
    workspace = tmp_path / "workspace"
    _old_workspace(workspace)
    markers = workspace / "artifacts" / "feature-panel" / "closure" / "sector-activation-markers"
    markers.mkdir()
    for name, written in (("z.json", 1_000_000_000), ("y.json", 2_000_000_000)):
        (markers / name).write_bytes(name.encode())
        os.utime(markers / name, ns=(written, written))
    assert FolderFiles(markers.parent).written(markers.name) == ["z.json", "y.json"]
    _moved(workspace)
    files = PanelClosureArtifactStore(ArtifactResolver(workspace / "artifacts")).files
    assert files.written(markers.name) == ["z.json", "y.json"]
    assert files.names(markers.name) == ["y.json", "z.json"]
    for name, data in (("b.json", b"1"), ("a.json", b"2"), ("b.json", b"3"), ("c.json", b"4")):
        files.write(f"logical/operational/snapshot-mappings/{name}", data)
    files.remove("logical/operational/snapshot-mappings/c.json")
    folder = "logical/operational/snapshot-mappings"
    assert files.names(folder) == ["a.json", "b.json"]
    assert files.written(folder) == ["a.json", "b.json"]
    files.write(f"{folder}/a.json", b"5")
    assert files.written(folder) == ["b.json", "a.json"]
    assert files.read(f"{folder}/c.json") is None and not files.exists(f"{folder}/c.json")
