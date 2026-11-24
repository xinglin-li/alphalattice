"""The research-inputs upgrade on a real data root and log: a 0.2.7 workspace's binding copies are
kept once behind the door and read back equal, and each copy is removed only after a machine
backup holds its bytes and a replay of that backup reads them equal."""

from __future__ import annotations

import os
import shutil
from datetime import date
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    AuthoringError,
    FactorInputBundle,
    factor_input_paths,
    legacy_bindings,
    read_factor_bundle,
    research_input_copies_family,
    research_input_upgrade_family,
)
from alphalattice.foundation.data_layer import history, inputs
from alphalattice.foundation.data_layer.backup import restore_data_root

TEST_SOURCE = "src/alphalattice/control/product_host/research_authoring/factor_inputs.py"


def _copy(workspace: Path, panel: bytes, database: bytes) -> str:
    """A 0.2.7 binding copy: its tree, the pooled database it links, and a publication record."""
    inputs = workspace / "research-inputs"
    files = {"artifacts/panel.parquet": panel, "market-data.duckdb": database}
    bundle = FactorInputBundle.create(
        panel_snapshot_hash="a" * 64,
        outcome_snapshot_hash="b" * 64,
        sessions=(date(2026, 1, 2),),
        files=tuple(sorted((name, sha256(body).hexdigest()) for name, body in files.items())),
        database_snapshot_hash=sha256(database).hexdigest(),
    )
    pooled = inputs / "databases" / bundle.database_snapshot_hash / "market-data.duckdb"
    if not pooled.is_file():  # bindings of one database share its pool, as 0.2.7 kept them
        pooled.parent.mkdir(parents=True)
        pooled.write_bytes(database)
    source = inputs / bundle.binding_hash / "source"
    (source / "artifacts").mkdir(parents=True)
    (source / "artifacts/panel.parquet").write_bytes(panel)
    os.link(pooled, source / "market-data.duckdb")
    (inputs / bundle.binding_hash / "manifest.json").write_text(bundle.model_dump_json())
    (inputs / "publications").mkdir(exist_ok=True)
    (inputs / "publications" / "receipt.json").write_text("{}")
    return bundle.binding_hash


@pytest.mark.proves("product_host.research_input_copies_removed_after_backup_replay")
def test_copies_are_kept_read_back_and_removed_only_after_the_backup_replays_them(
    tmp_path: Path,
) -> None:
    """A workspace's 0.2.7 copies upgrade ALL EQUAL, read the same bundles through a rollback and
    forward, and are removed only once their backup restored reads every binding equal to the
    copies, byte for byte."""
    original = tmp_path / "held"
    _copy(original, b"panel one", b"db")
    _copy(original, b"panel two", b"db")
    _kept_read_back_and_removed(tmp_path, original)


@pytest.mark.proves("product_host.research_input_copies_removed_after_backup_replay")
@pytest.mark.real_evidence
def test_qa_copies_are_kept_read_back_and_removed_only_after_the_backup_replays_them(
    tmp_path: Path, evidence_roots: Any
) -> None:
    """The same upgrade over a QA copy of a 0.2.7 workspace's research inputs: the upgrade replay
    that keeps original data, in the nightly."""
    qa = evidence_roots.require("alpha_development_outcomes")
    _kept_read_back_and_removed(tmp_path, qa / "alpha-base")  # read only


def _kept_read_back_and_removed(tmp_path: Path, original: Path) -> None:
    """Upgrade a copy of `original`'s research inputs, remove the copies
    after a replayed backup, and read every binding back equal."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copytree(original / "research-inputs", workspace / "research-inputs")
    expected = dict(legacy_bindings(original))
    assert expected
    runner = SimpleNamespace(root=history.data_root(), log=history.held(), workspaces=(workspace,))
    family = research_input_upgrade_family(lambda _workspace: "w")
    copies = research_input_copies_family(tmp_path / "machine-backups")
    assert family.holds(workspace) and copies.holds(workspace)
    family.step(runner)
    family.step(runner)  # a resumed step keeps what it kept and changes nothing
    assert family.verify(runner) is True  # ALL EQUAL
    assert not family.holds(workspace)
    reads: dict[str, FactorInputBundle] = {}
    for _round in range(2):  # each round reads every binding equal to its first read
        for binding_hash in expected:
            source, _artifacts = factor_input_paths(workspace, binding_hash)
            assert "views" in source.parts  # the data root's, never a copy
            bundle = read_factor_bundle(workspace, binding_hash)
            assert reads.setdefault(binding_hash, bundle) == bundle
    copies.step(runner)
    assert copies.verify(runner) is True
    assert not copies.holds(workspace)
    assert sorted(p.name for p in (workspace / "research-inputs").iterdir()) == ["publications"]
    for binding_hash in expected:
        assert read_factor_bundle(workspace, binding_hash) == reads[binding_hash]
    (generation,) = (tmp_path / "machine-backups" / "generations").iterdir()
    restored = restore_data_root(generation, tmp_path / "restored")
    with history.own_root(restored):
        log = history.held()
        for binding_hash, files in expected.items():
            kept = dict(inputs.members(restored, log.read, binding_hash))
            assert sorted(kept) == sorted(relative for relative, _path in files)
            for relative, path in files:
                assert inputs.file_digest(kept[relative]) == inputs.file_digest(path)


@pytest.mark.proves("product_host.research_input_copy_replay_unequal_keeps_every_copy")
def test_a_copy_that_differs_from_its_kept_bytes_keeps_every_copy(tmp_path: Path) -> None:
    """Copies of a workspace with no id on the log are refused before anything is kept, so none
    is kept unheld; a copy changed after it was kept replays unequal: the upgrade stops by name
    and no copy or pool is removed."""
    workspace = tmp_path / "workspace"
    binding_hash = _copy(workspace, b"panel", b"db")
    runner = SimpleNamespace(root=history.data_root(), log=history.held(), workspaces=(workspace,))
    with pytest.raises(DataError) as unheld:
        research_input_upgrade_family(lambda _workspace: None).step(runner)
    assert unheld.value.failure.code == "upgrade.refused"
    assert not history.research_input_kept(binding_hash)
    research_input_upgrade_family(lambda _workspace: "w").step(runner)
    panel = workspace / "research-inputs" / binding_hash / "source/artifacts/panel.parquet"
    panel.unlink()  # a new file, not the kept body a link may share
    panel.write_bytes(b"other")
    before = sorted(p.name for p in (workspace / "research-inputs").iterdir())
    with pytest.raises(DataError) as refused:
        research_input_copies_family(tmp_path / "machine-backups").step(runner)
    assert refused.value.failure.code == "upgrade.refused"
    assert sorted(p.name for p in (workspace / "research-inputs").iterdir()) == before


def _tree(inputs: Path, name: str, bundle: FactorInputBundle, database: bool) -> None:
    source = inputs / name / "source"
    (source / "artifacts").mkdir(parents=True)
    (source / "artifacts/panel.parquet").write_bytes(b"panel")
    if database:
        (source / "market-data.duckdb").write_bytes(b"db")
    (inputs / name / "manifest.json").write_text(bundle.model_dump_json())


@pytest.mark.proves("product_host.research_input_copies_listed_as_kept")
def test_each_copy_is_listed_as_the_door_keeps_it(tmp_path: Path) -> None:
    """A copy's files are listed under its binding, its database also at the pooled path the door
    reads; a copy a cleanup released lists only what it holds; publication records and pools are
    no binding, a workspace with no copies lists none, and a copy under another name is refused."""
    files = (
        ("artifacts/panel.parquet", "1" * 64),
        ("market-data.duckdb", sha256(b"db").hexdigest()),
    )
    bundle = FactorInputBundle.create(
        panel_snapshot_hash="a" * 64,
        outcome_snapshot_hash="b" * 64,
        sessions=(date(2026, 1, 2),),
        files=files,
        database_snapshot_hash=files[1][1],
    )
    inputs = tmp_path / "research-inputs"
    _tree(inputs, bundle.binding_hash, bundle, database=True)
    (inputs / "publications").mkdir()
    (inputs / "databases" / files[1][1]).mkdir(parents=True)
    [(name, listed)] = legacy_bindings(tmp_path)
    assert name == bundle.binding_hash
    assert sorted(relative for relative, _ in listed) == sorted(
        [
            f"{name}/manifest.json",
            f"{name}/source/artifacts/panel.parquet",
            f"{name}/source/market-data.duckdb",
            f"databases/{files[1][1]}/market-data.duckdb",
        ]
    )
    (inputs / name / "source/market-data.duckdb").unlink()
    [(_, released)] = legacy_bindings(tmp_path)
    assert len(released) == 2
    assert legacy_bindings(tmp_path / "new-workspace") == []
    other = tmp_path / "other"  # a copy is kept only under the binding its own manifest names
    _tree(other / "research-inputs", "c" * 64, bundle, database=True)
    with pytest.raises(AuthoringError, match="input_manifest_identity_mismatch"):
        legacy_bindings(other)


@pytest.mark.proves("product_host.research_input_copy_removal_resumes_after_a_stop")
@pytest.mark.parametrize("stop", ["moving", "deleting"])
def test_a_removal_stopped_part_way_resumes_to_the_same_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    """A stop while the verified copies move aside or are deleted leaves each tree whole or
    retired, and the next step finishes the recorded set, pools included, to publications alone."""
    from alphalattice.control.product_host.research_authoring import factor_inputs
    from alphalattice.control.task_control import upgrade

    workspace = tmp_path / "workspace"
    bindings = [_copy(workspace, b"panel one", b"db"), _copy(workspace, b"panel two", b"db")]
    runner = SimpleNamespace(root=history.data_root(), log=history.held(), workspaces=(workspace,))
    research_input_upgrade_family(lambda _workspace: "w").step(runner)
    copies = research_input_copies_family(tmp_path / "machine-backups")
    module, name = (upgrade.os, "replace") if stop == "moving" else (factor_inputs.shutil, "rmtree")
    real, calls = getattr(module, name), []

    def stopped(*args, **kwargs):
        calls.append(args)
        if len(calls) == (3 if stop == "moving" else 1):
            raise OSError("stopped")
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, stopped)
    with pytest.raises(OSError, match="stopped"):
        copies.step(runner)
    monkeypatch.setattr(module, name, real)
    for binding_hash, files in legacy_bindings(workspace):  # each tree left is whole
        assert (workspace / "research-inputs" / binding_hash / "manifest.json").is_file()
        assert all(path.is_file() for _relative, path in files)
    assert copies.holds(workspace) and copies.verify(runner) is False
    copies.step(runner)
    assert copies.verify(runner) is True and not copies.holds(workspace)
    assert sorted(p.name for p in (workspace / "research-inputs").iterdir()) == ["publications"]
    for binding_hash in bindings:
        assert read_factor_bundle(workspace, binding_hash).binding_hash == binding_hash
