"""The synthetic outcome manifest and source table of the walk-forward cases.

One development chunk and one sealed-holdout chunk with fixed hashes, and a
source table of formation rows, which the execution-target cases and the
deterministic program case both read.
Test support beside its owner; nothing here is product authority or evidence.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import pyarrow as pa

from alphalattice.foundation.causal_outcomes.execution.contracts import (
    CausalExecutionOutcomeChunk,
    CausalExecutionOutcomeManifest,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _manifest(listing_ids: tuple[str, ...]) -> CausalExecutionOutcomeManifest:
    development = CausalExecutionOutcomeChunk(
        split="DEVELOPMENT",
        year=2026,
        row_count=1,
        first_formation_session=date(2026, 1, 2),
        last_formation_session=date(2026, 12, 30),
        content_hash="a" * 64,
        metadata_hash="b" * 64,
        uri="playpen://outcomes/development.parquet",
    )
    holdout = CausalExecutionOutcomeChunk(
        split="SEALED_HOLDOUT",
        year=2027,
        row_count=1,
        first_formation_session=date(2027, 1, 2),
        last_formation_session=date(2027, 12, 30),
        content_hash="c" * 64,
        metadata_hash="d" * 64,
        uri="playpen://outcomes/holdout.parquet",
    )
    values = CausalExecutionOutcomeManifest.model_construct(
        research_cadence="DAILY",
        market_as_of=date(2027, 12, 31),
        listing_ids=listing_ids,
        listing_set_hash=canonical_hash(listing_ids),
        schedule_hash="e" * 64,
        ordered_session_triples_hash="f" * 64,
        source_rows_semantic_hash="1" * 64,
        price_basis="open_split_adjusted",
        return_formula_identity=(
            "formation-close-next-common-open-following-common-open-simple-return"
        ),
        corporate_action_identity="provider-split-adjusted-open-and-period-dividend",
        data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
        development_chunks=(development,),
        sealed_holdout_chunks=(holdout,),
        development_formation_count=1,
        sealed_holdout_formation_count=1,
        limitations=("fixture",),
        snapshot_hash="0" * 64,
    )
    return CausalExecutionOutcomeManifest(
        **values.model_dump(exclude={"snapshot_hash"}),
        snapshot_hash=canonical_hash(values.model_dump(mode="json", exclude={"snapshot_hash"})),
    )


def _open_timestamp(session: date, hour: int) -> datetime:
    return datetime.combine(session, time(hour=hour), tzinfo=UTC)


def _source_table(
    formations: tuple[date, ...],
    listing_ids: tuple[str, ...],
    *,
    missing_listing: str | None = None,
) -> pa.Table:
    rows: list[dict[str, object]] = []
    for sequence, formation in enumerate(formations, 1):
        entry_session = formation + timedelta(days=1)
        holding_end = formation + timedelta(days=2)
        for listing_index, listing_id in enumerate(listing_ids):
            entry_open = 100.0 + listing_index
            holding_open = entry_open + sequence
            dividend = 0.25
            simple_return = (holding_open + dividend) / entry_open - 1.0
            missing = listing_id == missing_listing
            row = {
                "listing_id": listing_id,
                "formation_session": formation,
                "formation_close_at": _open_timestamp(formation, 20),
                "entry_session": entry_session,
                "entry_open_at": _open_timestamp(entry_session, 13),
                "holding_end_session": holding_end,
                "holding_end_open_at": _open_timestamp(holding_end, 13),
                "actual_session_span": 2,
                "entry_status": (
                    "NO_OFFICIAL_OPEN" if missing else "ASSUMED_ELIGIBLE_FROM_DAILY_BAR"
                ),
                "holding_end_status": "ASSUMED_ELIGIBLE_FROM_DAILY_BAR",
                "entry_open_split_adjusted": None if missing else entry_open,
                "holding_end_open_split_adjusted": holding_open,
                "period_dividend_split_adjusted": dividend,
                "simple_return": None if missing else simple_return,
                "row_hash": canonical_hash((listing_id, formation, simple_return)),
            }
            rows.append(row)
    return pa.Table.from_pylist(rows)
