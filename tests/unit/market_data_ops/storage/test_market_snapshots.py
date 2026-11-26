"""A workspace's committed selection, copied into each database instance from one read of it."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from alphalattice.foundation.market_data_ops.storage import market_snapshots

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/storage/market_snapshots.py"


@pytest.mark.proves("market_data_ops.a_host_reads_its_committed_selection_once")
def test_a_fresh_instance_takes_the_process_read_only_of_its_file_root_and_selection(
    tmp_path: Path,
) -> None:
    """A fresh instance of the same file, catalog root and selection (anchor and version-index
    rows) takes this process's read of it; a moved anchor or index, another file or another
    root reads the catalog."""
    database, root = tmp_path / "market-data.duckdb", tmp_path / "data"
    selected = ((7, 1_000), (2, 41))
    bodies = {"committed_raw_bars": pa.table({"listing_id": ["L"], "close": [101.0]})}
    market_snapshots._READ[(str(database), str(root))] = (selected, bodies)
    assert market_snapshots._read_of(database, root, selected) is bodies
    for other in (
        (database, root, ((8, 1_000), (2, 41))),  # a publication moved the anchor
        (database, root, ((7, 1_000), (2, 42))),  # the version index changed under it
        (tmp_path / "other.duckdb", root, selected),
        (database, tmp_path / "other-root", selected),
    ):
        assert market_snapshots._read_of(*other) is None
