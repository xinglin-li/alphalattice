"""The research-input door on a real data root and log: a binding's files are kept once under
blobs/, held by one pin per binding, read back as a folder, and collected only when no binding
holds them."""

from __future__ import annotations

import errno
import json
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer import governance, history, inputs
from alphalattice.foundation.data_layer.store import StorePort

TEST_SOURCE = "src/alphalattice/foundation/data_layer/inputs.py"


def _binding(folder: Path, files: dict[str, bytes]) -> Path:
    for relative, payload in files.items():
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return folder


def _laid_out(binding_hash: str, folder: Path) -> list[tuple[str, Path]]:
    """A binding's files at their paths under `research-inputs/`."""
    return [(f"{binding_hash}/{relative}", path) for relative, path in inputs.folder_files(folder)]


def _tree(folder: Path) -> dict[str, bytes]:
    files = (path for path in folder.rglob("*") if path.is_file())
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in files}


def _holders(log: Any, digest: str) -> list[tuple[str, str]]:
    rows = log.read(
        "SELECT r.root_kind, r.root_id FROM live_ref r JOIN artifact a"
        " ON a.artifact_id = r.artifact_id WHERE a.content_digest = ? ORDER BY 1, 2",
        (digest,),
    )
    return [(str(kind), str(root)) for kind, root in rows]


@pytest.mark.proves("data_layer.research_inputs_kept_once_and_held_per_workspace")
def test_a_binding_is_kept_once_held_by_its_workspace_and_read_back_as_its_folder(
    tmp_path: Path,
) -> None:
    """Two bindings sharing a file keep its bytes once, the workspace that binds them holding the
    shared body, each manifest and relation once, and no pin; each reads back as the folder it
    was bound from, byte for byte; a binding kept again records nothing twice."""
    log, root = history.held(), history.data_root()
    shared = b"duckdb file bytes"
    files = {"market-data.duckdb": shared, "artifacts/panel.parquet": b"a", "binding.json": b"{}"}
    first = _binding(tmp_path / "a", files)
    second = _binding(tmp_path / "b", {"market-data.duckdb": shared, "binding.json": b"[]"})
    bindings = (("a" * 64, first), ("b" * 64, second))
    for binding_hash, folder in bindings:
        inputs.keep_binding(root, log, binding_hash, _laid_out(binding_hash, folder), workspace="w")
    events = log.read("SELECT count(*) FROM event")[0][0]
    inputs.keep_binding(root, log, "a" * 64, _laid_out("a" * 64, first), workspace="w")
    assert log.read("SELECT count(*) FROM event")[0][0] == events  # kept again: nothing new
    assert _holders(log, "a" * 64) == [("workspace", "w")]
    for binding_hash, folder in bindings:
        view = inputs.view(root, log.read, binding_hash, "w")
        assert view == root / "views" / "w" / "research-inputs"
        assert _tree(view / binding_hash) == _tree(folder)
    duckdb = inputs.file_digest(first / "market-data.duckdb")
    assert _holders(log, duckdb) == [("workspace", "w")]  # one body, held once
    bodies = [p for p in (root / "blobs").iterdir() if not p.name.startswith(".")]
    assert len(bodies) == 6  # four distinct files and two manifests
    with pytest.raises(DataError) as refused:
        inputs.view(root, log.read, "c" * 64, "w")
    assert refused.value.failure.code == "data_layer.input_missing"


@pytest.mark.proves("data_layer.research_input_held_by_each_workspace_binding_it")
def test_a_second_workspace_binding_a_kept_input_holds_it_too(tmp_path: Path) -> None:
    """A binding kept for one workspace is held again by the next workspace that binds it, so
    either one's copy carries it and neither one's removal releases it from the other."""
    folder = _binding(tmp_path / "a", {"market-data.duckdb": b"db", "binding.json": b"{}"})
    binding = "a" * 64
    history.open_history("w1", "first", tmp_path / "w1")
    try:
        history.keep_research_input(binding, _laid_out(binding, folder))
        history.hold_research_input(binding)  # already held: nothing more
        history.open_history("w2", "second", tmp_path / "w2")
        history.hold_research_input(binding)
        holders = history.held().read(
            "SELECT r.root_id FROM live_ref r JOIN artifact a ON a.artifact_id = r.artifact_id"
            " WHERE a.content_digest = ? AND r.root_kind = 'workspace' ORDER BY r.root_id",
            (binding,),
        )
    finally:
        history.close_history()
    assert [row[0] for row in holders] == ["w1", "w2"]


@pytest.mark.proves("data_layer.research_input_released_with_its_last_workspace")
def test_a_binding_goes_with_its_last_workspace_and_its_views_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While a workspace binds it, the collector keeps every body of a binding; once the last
    workspace that binds it is deleted, its members and manifest are collected and the views
    that linked them are removed, so their bytes leave the disk; another binding's view stays."""
    monkeypatch.setattr(governance, "UNHELD", timedelta(0))
    with history.own_root(tmp_path / "root") as root:
        log = history.held()
        for workspace in ("w1", "w2"):
            log.record("workspace.created", ("workspace", workspace), {"name": workspace})
        gone, stays = "a" * 64, "b" * 64
        for binding_hash, workspace, payload in ((gone, "w1", b"one"), (stays, "w2", b"two")):
            folder = _binding(tmp_path / binding_hash[:1], {"panel.parquet": payload})
            laid_out = _laid_out(binding_hash, folder)
            inputs.keep_binding(root, log, binding_hash, laid_out, workspace=workspace)
            inputs.view(root, log.read, binding_hash, "-")
        port = StorePort.open(root, log)
        try:
            assert port.collect_unheld() == []  # held while a workspace binds it
            log.record("workspace.status_changed", ("workspace", "w1"), {"status": "deleted"})
            assert len(port.collect_unheld()) == 2  # its member and its manifest
        finally:
            port.close()
        view = root / "views" / "-" / "research-inputs"
        assert sorted(p.name for p in view.iterdir()) == [stays]
        assert not inputs.kept(root, log.read, gone) and inputs.kept(root, log.read, stays)
        assert _tree(view / stays) == {"panel.parquet": b"two"}


@pytest.mark.proves("data_layer.research_inputs_carried_into_a_copy")
def test_a_copy_carries_its_workspace_bindings_onto_its_own_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy carries only its workspace's bindings, which its own root keeps and reads back byte
    for byte, refusing before any keep a body, path, digest or manifest name that is not sound."""
    log, root = history.held(), history.data_root()
    folder = _binding(tmp_path / "a", {"market-data.duckdb": b"db", "binding.json": b"{}"})
    held, alone = "a" * 64, "b" * 64
    inputs.keep_binding(root, log, held, _laid_out(held, folder), workspace="w")
    inputs.keep_binding(root, log, alone, _laid_out(alone, folder))
    copy, tampered = tmp_path / "copy", tmp_path / "tampered"
    carried = copy / inputs.CARRIED
    open_path = Path.open

    @contextmanager
    def short_manifest(stream):
        with stream:

            def write(payload):
                stream.write(payload[:4])
                raise OSError("carried manifest write interrupted")

            yield SimpleNamespace(write=write)

    def interrupted_open(path, mode="r", *args, **kwargs):
        stream = open_path(path, mode, *args, **kwargs)
        if path.parent == carried and mode in {"wb", "xb"}:
            return short_manifest(stream)
        return stream

    with monkeypatch.context() as interrupted:
        interrupted.setattr(Path, "open", interrupted_open)
        with pytest.raises(OSError, match="carried manifest write interrupted"):
            inputs.carry(log.read, root, "w", copy)
    assert not (carried / f"{held}.json").exists()
    assert inputs.carry(log.read, root, "w", copy) == 1

    def no_links(_source: Path, _target: Path) -> None:
        raise OSError(errno.ENOTSUP, "hard links are unavailable")

    with monkeypatch.context() as linkless:
        linkless.setattr(inputs.os, "link", no_links)
        assert inputs.carry(log.read, root, "w", tampered) == 1
    manifest = copy / inputs.CARRIED / f"{held}.json"
    original = manifest.stat()
    manifest.chmod(0o444)
    try:
        assert inputs.carry(log.read, root, "w", copy) == 1
        assert manifest.stat().st_mtime_ns == original.st_mtime_ns
    finally:
        manifest.chmod(original.st_mode)

    conflicting = tampered / inputs.CARRIED / f"{held}.json"
    previous = conflicting.read_bytes()
    conflicting.write_bytes(b"[]")
    with pytest.raises(DataError, match="carried research input manifest differs") as refused:
        inputs.carry(log.read, root, "w", tampered)
    assert refused.value.failure.code == "data_layer.input_invalid"
    assert conflicting.read_bytes() == b"[]"
    conflicting.write_bytes(previous)
    body = next(p for p in (tampered / inputs.CARRIED).iterdir() if p.suffix != ".json")
    body.unlink()  # a new file, not the kept body a link may share
    body.write_bytes(b"other")
    escaping, unsealed, misnamed = (
        tmp_path / "escaping",
        tmp_path / "unsealed",
        tmp_path / "misnamed",
    )
    for forged in (escaping, unsealed, misnamed):
        assert inputs.carry(log.read, root, "w", forged) == 1
    manifest = escaping / inputs.CARRIED / f"{held}.json"
    listed = json.loads(manifest.read_bytes())
    manifest.write_text(json.dumps([[r"..\..\escape.duckdb", listed[0][1]], *listed[1:]]))
    manifest = unsealed / inputs.CARRIED / f"{held}.json"
    manifest.write_text(json.dumps([[listed[0][0], "../" + listed[0][1][7:]], *listed[1:]]))
    (misnamed / inputs.CARRIED / f"{held}.json").rename(misnamed / inputs.CARRIED / "..json")
    with history.own_root(tmp_path / "other") as other:
        copied = history.held()
        for forged in (tampered, escaping, unsealed, misnamed):  # refused before anything is kept
            with pytest.raises(DataError) as refused:
                inputs.import_carried(other, copied, forged, workspace="w2")
            assert refused.value.failure.code == "data_layer.input_invalid"
        assert not [p for p in (other / "blobs").glob("*") if not p.name.startswith(".")]
        assert inputs.import_carried(other, copied, copy, workspace="w2") == 1
        assert not (copy / inputs.CARRIED).exists()
        view = inputs.view(other, copied.read, held, "w2")
        assert _tree(view / held) == _tree(folder)
        with pytest.raises(DataError):
            inputs.view(other, copied.read, alone, "w2")  # held by no workspace: not carried
