"""A Panel chunk read under ALPHALATTICE_PANEL_READS: its stored file, or its derived body."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pyarrow as pa
import pytest

from alphalattice.control.workspace_runtime import panel_reads
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import history

TEST_SOURCE = "src/alphalattice/control/workspace_runtime/panel_reads.py"
CHUNK = "c" * 64
METADATA = {"panel_binding_hash": "b" * 64, "calendar_year": "2025", "calendar_part": "2"}


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
