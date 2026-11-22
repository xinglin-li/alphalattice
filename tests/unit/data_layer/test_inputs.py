"""The research-input door's file seal, listing and view, with no log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from blake3 import blake3

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import inputs

TEST_SOURCE = "src/alphalattice/kernel/data_layer/inputs.py"


@pytest.mark.proves("data_layer.research_input_seal_is_blake3_of_its_bytes")
@pytest.mark.parametrize(
    "size",
    [
        pytest.param(0, id="empty"),
        pytest.param(1, id="one-byte"),
        pytest.param((1 << 20) + 3, id="over-a-mebibyte"),
    ],
)
def test_a_files_seal_is_the_blake3_of_its_bytes_and_a_binding_lists_relative_paths(
    tmp_path: Path, size: int
) -> None:
    """The seal reads a file of any size, the empty one included, as BLAKE3 over its bytes; a
    binding's files are listed by their relative POSIX paths, nested folders included."""
    payload = bytes(range(256)) * (size // 256) + bytes(size % 256)
    nested = tmp_path / "source" / "artifacts" / "panel.parquet"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(payload)
    (tmp_path / "source" / "market-data.duckdb").write_bytes(b"db")
    assert inputs.file_digest(nested) == "blake3:" + blake3(payload).hexdigest()
    listed = sorted(relative for relative, _ in inputs.folder_files(tmp_path / "source"))
    assert listed == ["artifacts/panel.parquet", "market-data.duckdb"]


@pytest.mark.proves("data_layer.research_input_path_stays_in_its_binding")
@pytest.mark.parametrize(
    "relative",
    [
        pytest.param("../escape.duckdb", id="parent"),
        pytest.param("/abs/escape.duckdb", id="absolute"),
        pytest.param("binding/../../escape.duckdb", id="nested-parent"),
        pytest.param(r"..\escape.duckdb", id="windows-parent"),
        pytest.param(r"binding\..\..\escape.duckdb", id="windows-nested-parent"),
        pytest.param(r"C:\escape.duckdb", id="windows-drive"),
        pytest.param(r"C:escape.duckdb", id="windows-drive-relative"),
        pytest.param(r"\\host\share\escape.duckdb", id="windows-share"),
        pytest.param("", id="empty"),
    ],
)
def test_a_path_that_leaves_its_binding_is_refused_before_anything_is_kept(
    tmp_path: Path, relative: str
) -> None:
    """A file laid out outside `research-inputs/` is refused by name before any log or body is
    touched, so a carried or upgraded binding never writes beside its folder."""
    source = tmp_path / "file"
    source.write_bytes(b"bytes")
    with pytest.raises(DataError) as refused:
        inputs.keep_binding(tmp_path, cast(Any, None), "a" * 64, [(relative, source)])
    assert refused.value.failure.code == "data_layer.input_invalid"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["file"]


def _store(root: Path, bindings: dict[str, dict[str, bytes]]) -> Any:
    """Blobs under `root` for each binding's members and manifest, and a read that answers the
    door's two lookups (a binding's manifest digest, a body's artifact) as a log would."""
    known: dict[str, str] = {}
    relations: dict[str, str] = {}

    def body(payload: bytes) -> str:
        digest = "blake3:" + blake3(payload).hexdigest()
        if digest not in known:
            known[digest] = f"artifact-{len(known)}"
            (root / "blobs").mkdir(parents=True, exist_ok=True)
            (root / "blobs" / known[digest]).write_bytes(payload)
        return digest

    for binding_hash, files in bindings.items():
        listed = [[relative, body(payload)] for relative, payload in sorted(files.items())]
        relations[binding_hash] = body(json.dumps(listed).encode())

    def read(sql: str, args: tuple[str, ...]) -> list[tuple[str, ...]]:
        if "json_extract" in sql:
            return [(relations[args[0]],)] if args[0] in relations else []
        return [(known[args[0]], "present")] if args[0] in known else []

    return read


@pytest.mark.proves("data_layer.research_input_view_lays_out_each_binding")
def test_a_view_lays_out_each_binding_and_places_a_shared_database_once(tmp_path: Path) -> None:
    """Each binding reads as its folder under the holder's view, its pooled database placed once
    beside it and kept for the next binding that shares it, each file the kept body's bytes."""
    a, b = "a" * 64, "b" * 64
    database = "databases/d/market-data.duckdb"
    read = _store(
        tmp_path,
        {
            a: {f"{a}/source/panel.parquet": b"panel-a", database: b"db"},
            b: {f"{b}/source/panel.parquet": b"panel-b", database: b"db"},
        },
    )
    folder = inputs.view(tmp_path, read, a, "w")
    assert folder == tmp_path / "views" / "w" / "research-inputs"
    assert inputs.view(tmp_path, read, b, "w") == folder
    files = {relative: path.read_bytes() for relative, path in inputs.folder_files(folder)}
    assert files == {
        f"{a}/source/panel.parquet": b"panel-a",
        f"{b}/source/panel.parquet": b"panel-b",
        database: b"db",
    }
    assert not [p for p in folder.iterdir() if p.name.startswith(".")]  # no staged folder left
    with pytest.raises(DataError) as refused:
        inputs.view(tmp_path, read, "c" * 64, "w")
    assert refused.value.failure.code == "data_layer.input_missing"


@pytest.mark.proves("data_layer.research_input_views_go_with_their_bodies")
def test_a_collected_bindings_view_and_its_unlisted_database_are_dropped(tmp_path: Path) -> None:
    """A view of a binding whose bodies were collected goes, with each pooled database no kept
    binding in that view lists, while a kept binding's view and its database stay."""
    kept, collected = "a" * 64, "b" * 64
    read = _store(
        tmp_path,
        {
            kept: {
                f"{kept}/source/panel.parquet": b"panel-a",
                "databases/d/market-data.duckdb": b"d",
            },
            collected: {f"{collected}/x.parquet": b"x", "databases/e/market-data.duckdb": b"e"},
        },
    )
    folder = inputs.view(tmp_path, read, kept, "w")
    inputs.view(tmp_path, read, collected, "w")

    def after_collection(sql: str, args: tuple[str, ...]) -> list[tuple[str, ...]]:
        return [] if "json_extract" in sql and args[0] == collected else read(sql, args)

    assert inputs.drop_views(tmp_path, after_collection) == 2
    assert sorted(p.name for p in folder.iterdir()) == sorted([kept, "databases"])
    assert sorted(p.name for p in (folder / "databases").iterdir()) == ["d"]
    assert inputs.drop_views(tmp_path, after_collection) == 0
