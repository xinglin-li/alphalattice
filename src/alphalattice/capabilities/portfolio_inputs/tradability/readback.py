"""Bounded numerical readback for an admitted tradability lineage."""

from __future__ import annotations

from datetime import date

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from alphalattice.kernel.data.enums import ExecutionSessionStatus

from .contracts import (
    DecisionTradabilityStatus,
    HistoricalDecisionTradabilitySurface,
    HistoricalExecutionAvailabilitySurface,
    TradabilitySurfaceChunk,
)
from .surface import TradabilityArtifactStore

type BoolArray = npt.NDArray[np.bool_]
type FloatArray = npt.NDArray[np.float64]

_ELIGIBLE_EXECUTION = frozenset(
    {
        ExecutionSessionStatus.VERIFIED_ELIGIBLE.value,
        ExecutionSessionStatus.ASSUMED_ELIGIBLE_FROM_DAILY_BAR.value,
    }
)


def _ordered_table(
    table: pa.Table,
    *,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
    kind: str,
) -> pa.Table:
    ordered = table.combine_chunks().sort_by(
        [("formation_session", "ascending"), ("listing_id", "ascending")]
    )
    # The exact rectangular axis, compared column against column: every row
    # at its position carries the session and listing that position names.
    expected_sessions = pa.array(
        np.repeat(np.asarray(sessions, dtype="datetime64[D]"), len(listings)), pa.date32()
    )
    expected_listings = pa.array(np.tile(np.asarray(listings, dtype=object), len(sessions)))
    if (
        ordered.num_rows != len(expected_sessions)
        or not pc.all(pc.equal(ordered["formation_session"], expected_sessions)).as_py()
        or not pc.all(pc.equal(ordered["listing_id"], expected_listings)).as_py()
    ):
        raise ValueError(f"data_tradability.{kind}_readback_axis_invalid")
    return ordered


def read_tradability_matrices(
    *,
    store: TradabilityArtifactStore,
    decision: HistoricalDecisionTradabilitySurface,
    execution: HistoricalExecutionAvailabilitySurface,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
) -> tuple[BoolArray, BoolArray, FloatArray]:
    """Read an exact rectangular slice without opening a workspace database."""
    if (
        not sessions
        or not listings
        or sessions != tuple(sorted(set(sessions)))
        or listings != tuple(sorted(set(listings)))
        or any(value not in decision.formation_sessions for value in sessions)
        or any(value not in decision.ordered_listing_ids for value in listings)
        or decision.formation_sessions != execution.formation_sessions
        or decision.ordered_listing_ids != execution.ordered_listing_ids
    ):
        raise ValueError("data_tradability.public_readback_axis_invalid")
    requested_sessions = set(sessions)

    def tables_for(
        chunks: tuple[TradabilitySurfaceChunk, ...],
    ) -> list[pa.Table]:
        selected = [
            chunk for chunk in chunks if requested_sessions.intersection(chunk.formation_sessions)
        ]
        return [pq.read_table(store.resolve_chunk(chunk)) for chunk in selected]

    decision_table = pa.concat_tables(tables_for(decision.chunks)).filter(
        pc.is_in(
            pc.field("formation_session"),
            value_set=pa.array(sessions, type=pa.date32()),
        )
        & pc.is_in(pc.field("listing_id"), value_set=pa.array(listings, type=pa.string()))
    )
    execution_table = pa.concat_tables(tables_for(execution.chunks)).filter(
        pc.is_in(
            pc.field("formation_session"),
            value_set=pa.array(sessions, type=pa.date32()),
        )
        & pc.is_in(pc.field("listing_id"), value_set=pa.array(listings, type=pa.string()))
    )
    decision_table = _ordered_table(
        decision_table,
        sessions=sessions,
        listings=listings,
        kind="decision",
    )
    execution_table = _ordered_table(
        execution_table,
        sessions=sessions,
        listings=listings,
        kind="execution",
    )
    shape = (len(sessions), len(listings))
    decision_values: BoolArray = np.asarray(
        pc.equal(
            decision_table["decision_status"],
            DecisionTradabilityStatus.PLANNED_ORDER_ELIGIBLE.value,
        ).to_numpy(zero_copy_only=False),
        dtype=np.bool_,
    ).reshape(shape)
    execution_values: BoolArray = np.asarray(
        pc.is_in(
            execution_table["execution_status"],
            value_set=pa.array(sorted(_ELIGIBLE_EXECUTION), pa.string()),
        ).to_numpy(zero_copy_only=False),
        dtype=np.bool_,
    ).reshape(shape)
    adv20: FloatArray = np.asarray(
        decision_table["causal_adv20"].to_numpy(zero_copy_only=False),
        dtype=np.float64,
    ).reshape(shape)
    decision_values.setflags(write=False)
    execution_values.setflags(write=False)
    adv20.setflags(write=False)
    return decision_values, execution_values, adv20


__all__ = ["read_tradability_matrices"]
