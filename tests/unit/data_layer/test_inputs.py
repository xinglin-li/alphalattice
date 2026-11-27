"""The research-input door's file seal, listing and view, with no log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from blake3 import blake3

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer import chain, inputs
from alphalattice.kernel.identity import blake3_digest, blake3_filename, canonical_hash

TEST_SOURCE = "src/alphalattice/foundation/data_layer/inputs.py"


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
    binding = canonical_hash("binding")
    with pytest.raises(DataError) as refused:
        inputs.keep_binding(tmp_path, cast(Any, None), binding, [(relative, source)])
    assert refused.value.failure.code == "data_layer.input_invalid"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["file"]


def _store(root: Path, bindings: dict[str, dict[str, bytes]]) -> Any:
    """Blobs under `root` for each binding's members and manifest, and a read that answers the
    door's two lookups (a binding's manifest digest, a body's artifact) as a log would."""
    known: dict[str, str] = {}
    relations: dict[str, str] = {}

    def body(payload: bytes) -> str:
        digest = blake3_digest(payload)
        if digest not in known:
            known[digest] = f"artifact-{len(known)}"
            (root / "blobs").mkdir(parents=True, exist_ok=True)
            (root / "blobs" / known[digest]).write_bytes(payload)
        return digest

    for binding_hash, files in bindings.items():
        listed = [[relative, body(payload)] for relative, payload in sorted(files.items())]
        relations[binding_hash] = body(chain.canonical(listed))

    def read(sql: str, args: tuple[str, ...]) -> list[tuple[str, ...]]:
        if "json_extract" in sql:
            return [(relations[args[0]],)] if args[0] in relations else []
        return [(known[args[0]], "present")] if args[0] in known else []

    return read


@pytest.mark.proves("data_layer.research_input_view_lays_out_each_binding")
def test_a_view_lays_out_each_binding_and_places_a_shared_database_once(tmp_path: Path) -> None:
    """Each binding reads as its folder under the holder's view, its pooled database placed once
    beside it and kept for the next binding that shares it, each file the kept body's bytes."""
    a, b = canonical_hash("binding-a"), canonical_hash("binding-b")
    panel_a = f"{blake3_filename(a)}/source/panel.parquet"
    panel_b = f"{blake3_filename(b)}/source/panel.parquet"
    database = "databases/d/market-data.duckdb"
    read = _store(
        tmp_path,
        {
            a: {panel_a: b"panel-a", database: b"db"},
            b: {panel_b: b"panel-b", database: b"db"},
        },
    )
    folder = inputs.view(tmp_path, read, a, "w")
    assert folder == tmp_path / "views" / "w" / "research-inputs"
    assert inputs.view(tmp_path, read, b, "w") == folder
    files = {relative: path.read_bytes() for relative, path in inputs.folder_files(folder)}
    assert files == {
        panel_a: b"panel-a",
        panel_b: b"panel-b",
        database: b"db",
    }
    assert not [p for p in folder.iterdir() if p.name.startswith(".")]  # no staged folder left
    with pytest.raises(DataError) as refused:
        inputs.view(tmp_path, read, canonical_hash("missing"), "w")
    assert refused.value.failure.code == "data_layer.input_missing"


@pytest.mark.proves("data_layer.research_input_views_go_with_their_bodies")
def test_a_collected_bindings_view_and_its_unlisted_database_are_dropped(tmp_path: Path) -> None:
    """A view of a binding whose bodies were collected goes, with each pooled database no kept
    binding in that view lists, while a kept binding's view and its database stay."""
    kept, collected = canonical_hash("kept"), canonical_hash("collected")
    kept_name, collected_name = blake3_filename(kept), blake3_filename(collected)
    read = _store(
        tmp_path,
        {
            kept: {
                f"{kept_name}/source/panel.parquet": b"panel-a",
                "databases/d/market-data.duckdb": b"d",
            },
            collected: {
                f"{collected_name}/x.parquet": b"x",
                "databases/e/market-data.duckdb": b"e",
            },
        },
    )
    folder = inputs.view(tmp_path, read, kept, "w")
    inputs.view(tmp_path, read, collected, "w")

    def after_collection(sql: str, args: tuple[str, ...]) -> list[tuple[str, ...]]:
        return [] if "json_extract" in sql and args[0] == collected else read(sql, args)

    assert inputs.drop_views(tmp_path, after_collection) == 2
    assert sorted(p.name for p in folder.iterdir()) == sorted([kept_name, "databases"])
    assert sorted(p.name for p in (folder / "databases").iterdir()) == ["d"]
    assert inputs.drop_views(tmp_path, after_collection) == 0


@pytest.mark.proves("data_layer.a_carried_binding_names_its_market_selection")
def test_a_carried_binding_takes_its_selection_only_from_its_own_files(tmp_path: Path) -> None:
    """A carried binding whose document names no market selection records none. One whose
    document is another binding's, or whose files lack the sealed index or the panel manifest its
    window is derived from, is refused by name before any database is read."""
    held, other = canonical_hash("held"), canonical_hash("other")
    document = tmp_path / "manifest.json"
    files = [(f"{blake3_filename(held)}/manifest.json", document)]
    named = {
        "binding_hash": held,
        "market_selection_id": "s-1",
        "panel_snapshot_hash": canonical_hash("panel"),
    }
    for body, refusal in (
        ({"binding_hash": held}, None),
        (named | {"binding_hash": other}, "holds the document of another binding"),
        (named, "names a selection its files do not hold"),
    ):
        document.write_text(json.dumps(body), encoding="utf-8")
        if refusal is None:
            assert inputs._sealed_selection(held, files) is None
            continue
        with pytest.raises(DataError, match=refusal) as refused:
            inputs._sealed_selection(held, files)
        assert refused.value.failure.code == "data_layer.input_invalid"
