"""A Panel chunk read under ALPHALATTICE_PANEL_READS: its stored file, or its derived body, which
follows the workspace's Panels: rooted, evicted, refilled and collected."""

import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphalattice.control.workspace_runtime import panel_reads
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels import reader  # noqa: F401  # registers the proof
from alphalattice.foundation.feature_engine.panels.identity import (
    panel_chunk_hash,
    panel_schema_hash,
)
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import history
from tests.feature_engine.panel_workspace import identity_panel_table

TEST_SOURCE = "src/alphalattice/control/workspace_runtime/panel_reads.py"
CHUNK = "c" * 64
METADATA = {"panel_binding_hash": "b" * 64, "calendar_year": "2025", "calendar_part": "2"}
BINDING = "5" * 64
YEARS = {"a": 2024, "b": 2025, "c": 2026}  # three Panel chunks, by name
FIRST, NEXT = "1" * 64, "2" * 64  # two Panel snapshots: FIRST holds a and b, NEXT holds a and c
ACTIVE = {"lifecycle": "ACTIVE", "physical_availability": "AVAILABLE"}
SUPERSEDED = {"lifecycle": "SUPERSEDED", "physical_availability": "AVAILABLE"}

CASES = {
    # case: (the projection published, the chunks read in this Host before it, the stored
    #        chunks gone, the stored chunks whose values moved under their row hashes, N in
    #        hours, the roots dropped, bodies evicted and collected, the bodies held after it).
    #        a and b were last read two hours before it.
    "a new date replaces a chunk": ({NEXT: ACTIVE}, "", "", "", 720, (1, 0, 1), "a"),
    "a superseded Panel keeps its chunks": (
        {FIRST: SUPERSEDED, NEXT: ACTIVE},
        "",
        "",
        "",
        720,
        (0, 0, 0),
        "ab",
    ),
    "unread past N and refillable": ({FIRST: ACTIVE}, "", "", "", 1, (0, 2, 0), ""),
    "the only copy is never evicted": ({FIRST: ACTIVE}, "", "b", "", 1, (0, 1, 0), "b"),
    "a damaged stored chunk keeps its body": ({FIRST: ACTIVE}, "", "", "b", 1, (0, 1, 0), "b"),
    "a read since its mark keeps it": ({FIRST: ACTIVE}, "a", "", "", 1, (0, 1, 0), "a"),
    "neither body nor stored chunk": ({NEXT: ACTIVE}, "", "b", "", 720, (1, 0, 1), "a"),
}


@pytest.mark.proves("workspace_runtime.panel_chunks_read_from_their_derived_bodies")
@pytest.mark.parametrize(
    "case",
    [
        "stored",
        "derived in the Host",
        "kept at its first read",
        "a body gone is kept again",
        "unkept outside the Host",
    ],
)
def test_a_panel_chunk_is_read_from_its_derived_body_once_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Under derived the Host keeps each chunk once, off its first read, byte for byte, keyed
    and rooted in the log, and later reads take that body; stored reads the workspace's file;
    outside the Host an unkept chunk is refused by name."""
    root, workspace = tmp_path / "data-root", tmp_path / "workspace"
    reads = "derived" if case == "derived in the Host" else "stored"
    monkeypatch.setenv(panel_reads.SETTING, reads)
    history.open_history("panel", "panel", workspace, root)
    try:
        resolver = ArtifactResolver(workspace / "artifacts")
        table = pa.table({"session_date": [1, 2], "row_hash": ["x", "y"]})
        chunk = resolver.publish_feature_panel_chunk(
            table=table, content_hash=CHUNK, metadata=METADATA
        )
        stored = workspace / "artifacts" / "feature-panel" / "chunks" / f"{CHUNK}.parquet"
        body = root / "blobs" / "feature_panel" / f"{CHUNK}.parquet"
        assert not body.exists()  # publishing writes only the workspace's chunk, as before
        monkeypatch.setenv(panel_reads.SETTING, "stored" if case == "stored" else "derived")
        if case == "unkept outside the Host":
            history.close_history()
            monkeypatch.setenv(history.DATA_ROOT_ENV, str(root))
            with pytest.raises(DataError, match="only the Host keeps feature_panel"):
                resolver.resolve_feature_panel_chunk(chunk)
            return
        assert (
            resolver.resolve_feature_panel_chunk(chunk) == stored
        )  # the first read waits on no keep
        history.close_history()  # finishes the keeps handed to StorePort's worker
        if case == "stored":
            assert not body.exists()
            return
        assert body.read_bytes() == stored.read_bytes()
        history.open_history("panel", "panel", workspace, root)
        if case == "a body gone is kept again":
            body.unlink()
            assert resolver.resolve_feature_panel_chunk(chunk) == stored
            history.close_history()
            history.open_history("panel", "panel", workspace, root)
        assert resolver.resolve_feature_panel_chunk(chunk) == body
        with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con:
            entries = con.execute(
                "SELECT recipe_key, part, input_digest, row_count, body FROM cache_entry"
            ).fetchall()
            roots = con.execute("SELECT derived_table, part FROM cache_root").fetchall()
        assert entries == [("b" * 64, "2025/2", CHUNK, 2, f"blobs/feature_panel/{CHUNK}.parquet")]
        assert roots == [("feature_panel", "2025/2")]
        port = history.store()
        assert port is not None
        with pytest.raises(DataError, match="not a table StorePort keeps as files"):
            port.keep_file(
                "feature_set", recipe_key="r", part="p", digest=CHUNK, source=body, rooted_by=None
            )
    finally:
        history.close_history()


def _panels(resolver: ArtifactResolver, panel: Path) -> dict:
    """Publish YEARS' chunks under their real identity and the manifests of FIRST and NEXT."""
    chunks, entries, schema = {}, {}, ""
    for name, year in YEARS.items():
        table = identity_panel_table(
            pa.table(
                {
                    "session_date": pa.array([date(year, 1, 2)] * 2, pa.date32()),
                    "listing_id": ["listing-a", "listing-b"],
                    "materialization_receipt_hash": ["a" * 64, "b" * 64],
                    "factor_a": pa.array([float(year), None], pa.float64()),
                    "factor_b": pa.array([0.0, -2.5], pa.float64()),
                }
            )
        )
        schema = panel_schema_hash(table.schema)
        digest = panel_chunk_hash(table, panel_binding_hash=BINDING, year=year)
        chunks[name] = resolver.publish_feature_panel_chunk(
            table=table,
            content_hash=digest,
            metadata={"panel_binding_hash": BINDING, "calendar_year": str(year)},
        )
        entries[name] = {"chunk_hash": digest, "year": year, "row_count": table.num_rows}
    (panel / "manifests").mkdir(parents=True, exist_ok=True)
    for snapshot, names in ((FIRST, "ab"), (NEXT, "ac")):
        (panel / "manifests" / f"{snapshot}.json").write_text(
            json.dumps(
                {
                    "snapshot_hash": snapshot,
                    "panel_binding_hash": BINDING,
                    "schema_hash": schema,
                    "chunks": [entries[name] for name in names],
                }
            )
        )
    return chunks


@pytest.mark.proves("workspace_runtime.panel_bodies_follow_the_workspaces_panels")
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_a_lifecycle_publication_roots_evicts_and_collects_the_hosts_panel_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """A lifecycle publication roots the available Panels' chunks, collects the rest, and evicts
    an unread body only while its stored chunk passes the reader's proof, which a read refills."""
    projection, read, gone, damaged, hours, counts, held = CASES[case]
    root, workspace = tmp_path / "data-root", tmp_path / "workspace"
    resolver = ArtifactResolver(workspace / "artifacts")
    monkeypatch.setenv(panel_reads.SETTING, "derived")
    chunks = _panels(resolver, workspace / "artifacts" / "feature-panel")
    stored = workspace / "artifacts" / "feature-panel"

    def path(name: str, folder: Path) -> Path:
        return folder / f"{chunks[name].content_hash}.parquet"

    bodies = root / "blobs" / "feature_panel"
    history.open_history("panel", "panel", workspace, root)
    try:
        for name in "ab":
            resolver.resolve_feature_panel_chunk(chunks[name])  # their first reads keep them
        history.close_history()
        with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con, con:
            con.execute(
                "UPDATE cache_entry SET last_read_us = ?", (int((time.time() - 7200) * 1e6),)
            )
        history.open_history("panel", "panel", workspace, root)
        for name in read:
            resolver.resolve_feature_panel_chunk(chunks[name])  # its body, marked as read
        for name in gone:
            path(name, stored / "chunks").unlink()
        for name in damaged:  # a value moved under its kept row hash, in a file of its own
            moved = pq.read_table(path(name, stored / "chunks"))
            at = moved.schema.get_field_index("factor_b")
            moved = moved.set_column(at, "factor_b", pa.array([0.0, 99.0], pa.float64()))
            pq.write_table(moved, tmp_path / "moved.parquet")
            os.replace(tmp_path / "moved.parquet", path(name, stored / "chunks"))
        monkeypatch.setattr(panel_reads, "UNREAD", timedelta(hours=hours))
        found: list[dict[str, int] | None] = []
        rooting = panel_reads.root
        monkeypatch.setattr(panel_reads, "root", lambda *args: found.append(rooting(*args)))
        resolver.publish_feature_panel_lifecycle_projection(
            snapshots=[{"snapshot_hash": h, **state} for h, state in projection.items()]
        )
        assert found[0] is not None
        assert (found[0]["unrooted"], found[0]["evicted"], found[0]["collected"]) == counts
        assert "".join(name for name in "ab" if path(name, bodies).exists()) == held
        for name in "ab":
            if path(name, stored / "chunks").exists() and name not in damaged:
                resolver.resolve_feature_panel_chunk(chunks[name])  # refilled, or its body
            elif not path(name, bodies).exists():
                with pytest.raises(FileNotFoundError, match=r"feature_panel\.chunk_unserved"):
                    resolver.resolve_feature_panel_chunk(chunks[name])
        history.close_history()  # finishes the refills handed to StorePort's worker
        assert all(
            path(name, bodies).exists() for name in "ab" if path(name, stored / "chunks").exists()
        )  # every chunk whose stored file is there is held again after its read
    finally:
        history.close_history()
