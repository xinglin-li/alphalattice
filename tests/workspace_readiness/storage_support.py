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


def published_panel_chunks(root: Path) -> set[str]:
    """Every Panel chunk a published generation's manifest names, as the storage plan names it."""

    return {
        f"artifacts/feature-panel/chunks/{chunk['chunk_hash']}.parquet"
        for manifest in (root / "artifacts/feature-panel/manifests").glob("*.json")
        for chunk in json.loads(manifest.read_text(encoding="utf-8"))["chunks"]
    }
