"""An outcome chunk's identity is derived from its rows, and a carried chunk is proved by it."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pytest

from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.foundation.causal_outcomes.execution.artifacts import (
    carried_chunk_check,
    chunk_identity,
)

TEST_SOURCE = "src/alphalattice/foundation/causal_outcomes/execution/artifacts.py"


def _chunk(returns: tuple[float, ...]) -> pa.Table:
    """A chunk of one listing's rows, each row hashed over its values as the compiler hashes it."""
    at = datetime(2024, 1, 2, 21, tzinfo=UTC)
    rows = []
    for day, value in enumerate(returns, start=2):
        row = {
            "listing_id": "listing-aapl",
            "symbol": "AAPL",
            "sequence": day,
            "formation_session": date(2024, 1, day),
            "formation_close_at": at,
            "entry_session": date(2024, 1, day + 1),
            "entry_open_at": at,
            "holding_end_session": date(2024, 1, day + 2),
            "holding_end_open_at": at,
            "actual_session_span": 1,
            "entry_status": "OPEN",
            "holding_end_status": "OPEN",
            "entry_open_split_adjusted": 100.0,
            "holding_end_open_split_adjusted": 100.0 * (1 + value),
            "period_dividend_split_adjusted": 0.0,
            "simple_return": value,
            "entry_source_row_hash": "1" * 64,
            "holding_end_source_row_hash": "2" * 64,
        }
        rows.append(row | {"row_hash": canonical_hash(row)})
    return pa.Table.from_pylist(rows)


@pytest.mark.proves("causal_outcomes.carried_chunk_is_proved_by_its_identity")
def test_a_carried_chunk_is_admitted_only_with_the_rows_its_identity_names(tmp_path: Path):
    """A carried chunk passes when each row's hash is over its values and the chunk's identity
    over its rows; a changed value, the rows under another chunk's identity, or a chunk carried
    with a profile, which an outcome never has, is refused."""
    chunk = _chunk((0.01, -0.02))
    digest = chunk_identity(chunk)
    carried_chunk_check(tmp_path, digest, None, chunk)

    changed = chunk.set_column(
        chunk.schema.get_field_index("simple_return"), "simple_return", [[0.01, 0.5]]
    )
    with pytest.raises(ValueError, match="chunk_identity_invalid"):
        carried_chunk_check(tmp_path, digest, None, changed)
    with pytest.raises(ValueError, match="chunk_identity_invalid"):
        carried_chunk_check(tmp_path, chunk_identity(_chunk((0.03,))), None, chunk)
    with pytest.raises(ValueError, match="chunk_identity_invalid"):
        carried_chunk_check(tmp_path, digest, {"descriptor": digest}, chunk)
