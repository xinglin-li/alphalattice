"""What the readiness tests read of a workspace's storage."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository


@contextmanager
def future_sample(
    market: MarketDataRepository,
    gate: WorkspaceMutationGate,
    table: str,
    listing_id: str,
    session: date,
) -> Iterator[None]:
    """Inject and restore one later raw/adjusted sample under the fixture's writer."""
    with gate.hold(), market.database.connect(read_only=False) as conn:
        conn.execute(
            f"INSERT INTO {table} SELECT * REPLACE (? AS session_date) FROM {table} "
            "WHERE listing_id = ? ORDER BY session_date DESC LIMIT 1",
            [session, listing_id],
        )
    try:
        yield
    finally:
        with gate.hold(), market.database.connect(read_only=False) as conn:
            conn.execute(
                f"DELETE FROM {table} WHERE session_date = ? AND listing_id = ?",
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
