"""A workspace's committed selection, copied into each database instance from one read of it."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from alphalattice.foundation.market_data_ops.storage import market_snapshots

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/storage/market_snapshots.py"


class _Connection:
    """A connection whose reads answer the anchor and the version index's digest, recording every
    statement it is given."""

    def __init__(self, anchor: tuple[int, int], index: tuple[int, int]) -> None:
        self.anchor, self.index = anchor, index
        self.statements: list[str] = []

    def execute(self, sql: str) -> _Connection:
        self.statements.append(sql)
        self.row = self.anchor if market_snapshots.ANCHOR in sql else self.index
        return self

    def fetchall(self) -> list[tuple[int, int]]:
        return [self.row]

    def register(self, name: str, table: pa.Table) -> None:
        self.statements.append(f"register {name}")

    def unregister(self, name: str) -> None:
        self.statements.append(f"unregister {name}")


@pytest.mark.proves("market_data_ops.a_host_reads_its_committed_selection_once")
def test_a_fresh_instance_takes_the_process_read_of_its_root_and_selection_whatever_its_file(
    tmp_path: Path,
) -> None:
    """A fresh instance of any file holding a selection this process read from the same catalog
    root (its anchor and version-index rows) fills its copy from that read; a moved anchor or
    index, or another root, reads the catalog."""
    root = tmp_path / "data"
    selected = ((7, 1_000), (2, 41))
    bodies = {"committed_raw_bars": pa.table({"listing_id": ["L"], "close": [101.0]})}
    market_snapshots._READ[(str(root), selected)] = bodies
    assert market_snapshots._read_of(root, selected) is bodies
    for other in (
        (root, ((8, 1_000), (2, 41))),  # a publication moved the anchor
        (root, ((7, 1_000), (2, 42))),  # the version index changed under it
        (tmp_path / "other-root", selected),
    ):
        assert market_snapshots._read_of(*other) is None
    research_copy = _Connection(*selected)  # another file, the same selection
    assert market_snapshots._copy_selection(research_copy, False, root)
    assert any("SELECT * FROM selection_body" in sql for sql in research_copy.statements)


@pytest.mark.proves("market_data_ops.an_instance_attaches_the_catalog_only_to_read_it")
def test_a_selection_no_read_holds_asks_for_the_catalog_and_writes_nothing(tmp_path: Path) -> None:
    """A copy this process has not read, on an instance with no catalog attached, is refused
    with nothing written, so its owner attaches the catalog and fills it again."""
    moved = _Connection((9, 1_000), (3, 50))
    assert not market_snapshots._copy_selection(moved, False, tmp_path / "data")
    assert all(sql.lstrip().upper().startswith("SELECT") for sql in moved.statements)
    assert not any("workspace_market" in sql for sql in moved.statements)
