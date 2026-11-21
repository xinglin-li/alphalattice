"""What the readiness tests read of a workspace's storage."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    MarketDataRepository,
    record_delta,
)

WRITTEN_AT = {
    "raw_daily_bar_current": "observed_at",
    "provider_adjusted_close_current": "updated_at",
}
"""Each market table's column holding the time its writer wrote the row."""


@contextmanager
def future_sample(
    market: MarketDataRepository,
    gate: WorkspaceMutationGate,
    table: str,
    listing_id: str,
    session: date,
) -> Iterator[None]:
    """Write one later raw/adjusted sample as a market data update writes it, then take it out of
    the workspace.

    A later adjusted close comes with its raw bar, as the action audit requires of every stored
    close, and each row carries this write's time, as a writer stamps it. The rows and their
    journal entries go in one transaction inside the repository's update
    (`MarketDataRepository.writing`), so the Host publishes the bar into world when the update
    ends, where every reader outside an update reads. World keeps what it was told, so the
    published bar stands after the workspace rows are removed.
    """
    tables = dict.fromkeys(("raw_daily_bar_current", table))
    now = datetime.now(UTC)
    with gate.hold(), market.writing(), market.database.transaction() as conn:
        for name in tables:
            conn.execute(
                "CREATE TEMP TABLE later_sample AS SELECT * REPLACE (CAST(? AS DATE) AS"
                f" session_date, CAST(? AS TIMESTAMP) AS {WRITTEN_AT[name]}) FROM {name}"
                " WHERE listing_id = ? ORDER BY session_date DESC LIMIT 1",
                [session, now.replace(tzinfo=None), listing_id],
            )
            record_delta(conn, "bars", "FROM later_sample", "session_date", now)
            conn.execute(f"INSERT INTO {name} SELECT * FROM later_sample")
            conn.execute("DROP TABLE later_sample")
    try:
        yield
    finally:
        with gate.hold(), market.database.connect(read_only=False) as conn:
            for name in tables:
                conn.execute(
                    f"DELETE FROM {name} WHERE session_date = ? AND listing_id = ?",
                    [session, listing_id],
                )


def published_panel_chunks(root: Path, panel_hashes: list[str]) -> set[str]:
    """Chunk paths for the given published generations, as the storage plan names them."""

    manifests = root / "artifacts/feature-panel/manifests"
    return {
        f"artifacts/feature-panel/chunks/{chunk['chunk_hash']}.parquet"
        for panel_hash in panel_hashes
        for chunk in json.loads((manifests / f"{panel_hash}.json").read_text(encoding="utf-8"))[
            "chunks"
        ]
    }
