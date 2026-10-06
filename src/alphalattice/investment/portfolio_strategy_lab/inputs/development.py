"""One-shot verified readback into a shared Portfolio development workspace."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from alphalattice.capabilities.causal_inputs.schedules import (
    entry_session_by_formation,
    session_clocks_from_execution_rows,
)
from alphalattice.capabilities.portfolio_inputs.signed_score.contracts import (
    PortfolioExecutionEvents,
)
from alphalattice.capabilities.portfolio_inputs.signed_score.execution_events import (
    execution_events_from_schedule,
)
from alphalattice.capabilities.portfolio_inputs.tradability.contracts import (
    DecisionTradabilityStatus,
    HistoricalDecisionTradabilitySurface,
    HistoricalExecutionAvailabilitySurface,
)
from alphalattice.capabilities.portfolio_inputs.tradability.surface import (
    TradabilityArtifactStore,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
    CausalOutcomeDevelopmentRows,
)
from alphalattice.kernel.data.enums import ExecutionSessionStatus

if TYPE_CHECKING:
    pass

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int64]

_ELIGIBLE_EXECUTION = frozenset(
    {
        ExecutionSessionStatus.VERIFIED_ELIGIBLE.value,
        ExecutionSessionStatus.ASSUMED_ELIGIBLE_FROM_DAILY_BAR.value,
    }
)


class PortfolioInputError(ValueError):
    """Stable fail-closed Strategy Lab input boundary."""


class RiskInputAuthority(Protocol):
    """Exact Risk input facts needed to admit an operational Tradability lane."""

    input_binding_hash: str
    return_epoch_hash: str
    universe_revision_sha256: str
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]


def _readonly[ArrayT: np.ndarray](value: ArrayT) -> ArrayT:
    value.setflags(write=False)
    return value


def _ordered_table(
    table: pa.Table,
    *,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
    kind: str,
) -> pa.Table:
    table = table.combine_chunks().sort_by(
        [("formation_session", "ascending"), ("listing_id", "ascending")]
    )
    expected_rows = len(sessions) * len(listings)
    if table.num_rows != expected_rows:
        raise PortfolioInputError(f"portfolio_strategy_lab.{kind}_row_count_invalid")
    observed_sessions = tuple(cast(list[date], table["formation_session"].to_pylist()))
    observed_listings = tuple(str(value) for value in table["listing_id"].to_pylist())
    expected_sessions = tuple(session for session in sessions for _ in listings)
    expected_listings = listings * len(sessions)
    if observed_sessions != expected_sessions or observed_listings != expected_listings:
        raise PortfolioInputError(f"portfolio_strategy_lab.{kind}_axis_invalid")
    return table


def _tradability_matrices(
    *,
    artifacts: TradabilityArtifactStore,
    decision: HistoricalDecisionTradabilitySurface,
    execution: HistoricalExecutionAvailabilitySurface,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
) -> tuple[BoolArray, BoolArray, FloatArray, tuple[date, ...]]:
    session_set = set(sessions)
    decision_chunks = [
        chunk for chunk in decision.chunks if session_set.intersection(chunk.formation_sessions)
    ]
    execution_chunks = [
        chunk for chunk in execution.chunks if session_set.intersection(chunk.formation_sessions)
    ]
    decision_tables = [pq.read_table(artifacts.resolve_chunk(chunk)) for chunk in decision_chunks]
    decision_table = pa.concat_tables(decision_tables)
    decision_table = decision_table.filter(
        pc.and_(
            pc.is_in(
                decision_table["formation_session"],
                value_set=pa.array(sessions, type=pa.date32()),
            ),
            pc.is_in(decision_table["listing_id"], value_set=pa.array(listings, type=pa.string())),
        )
    )
    execution_tables = [pq.read_table(artifacts.resolve_chunk(chunk)) for chunk in execution_chunks]
    execution_table = pa.concat_tables(execution_tables)
    execution_table = execution_table.filter(
        pc.and_(
            pc.is_in(
                execution_table["formation_session"],
                value_set=pa.array(sessions, type=pa.date32()),
            ),
            pc.is_in(execution_table["listing_id"], value_set=pa.array(listings, type=pa.string())),
        )
    )
    decision_table = _ordered_table(
        decision_table, sessions=sessions, listings=listings, kind="decision_tradability"
    )
    execution_table = _ordered_table(
        execution_table, sessions=sessions, listings=listings, kind="execution_availability"
    )
    shape = (len(sessions), len(listings))
    decision_values: BoolArray = np.asarray(
        [
            str(value) == DecisionTradabilityStatus.PLANNED_ORDER_ELIGIBLE.value
            for value in decision_table["decision_status"].to_pylist()
        ],
        dtype=np.bool_,
    ).reshape(shape)
    execution_values: BoolArray = np.asarray(
        [
            str(value) in _ELIGIBLE_EXECUTION
            for value in execution_table["execution_status"].to_pylist()
        ],
        dtype=np.bool_,
    ).reshape(shape)
    adv20: FloatArray = np.asarray(
        decision_table["causal_adv20"].to_numpy(zero_copy_only=False), dtype=np.float64
    ).reshape(shape)
    intended = tuple(
        cast(list[date], decision_table["intended_execution_session"].to_pylist())[
            index * len(listings)
        ]
        for index in range(len(sessions))
    )
    return (
        _readonly(decision_values),
        _readonly(execution_values),
        _readonly(adv20),
        intended,
    )


class ExecutionClockContext(NamedTuple):
    """One execution read, serving every temporal question a campaign asks.

    ``session_clocks`` is derived from the same rows as ``events`` and is not
    sealed: it is re-derivable at its owner, and a second copy on disk could only
    ever disagree with the snapshot it came from. Carrying it here is what lets
    admission resolve an open or a close at any offset without a second calendar
    being materialised beside the durable one.
    """

    events: PortfolioExecutionEvents
    session_clocks: dict[date, dict[str, object]]
    entry_session_by_formation: dict[date, date]
    """Which session each formation filled in, from the snapshot's own rows.

    Carried so a consumer re-marking a book looks the session up by label. An
    index alongside the formation axis would be one day early.
    """

    table: pa.Table
    """The rows themselves, so a caller needing returns does not read them again."""


def resolve_execution_clock_context(
    *,
    reader: CausalOutcomeDevelopmentRows,
    manifest_ref: str,
    sessions: tuple[date, ...],
    history_margin_sessions: int = 0,
) -> ExecutionClockContext:
    """Read the execution snapshot once and answer every clock question from it.

    The single opening of the execution owner on the strong path. Both the
    campaign events and the session instants admission compares against come out
    of one table, so there is no window in which a locally regenerated calendar
    could disagree with the rows the realised returns were measured on.
    """
    if history_margin_sessions < 0:
        raise PortfolioInputError("portfolio_strategy_lab.execution_clock_margin_invalid")
    owner_sessions = execution_axis_for(
        reader=reader,
        manifest_ref=manifest_ref,
        formation_sessions=sessions,
        margin_sessions=history_margin_sessions,
    )
    table = reader.read_development_sessions(manifest_ref, owner_sessions)
    events = _events_from_table(
        table=table, reader=reader, manifest_ref=manifest_ref, sessions=sessions
    )
    return ExecutionClockContext(
        events=events,
        session_clocks=session_clocks_from_execution_rows(table),
        entry_session_by_formation=entry_session_by_formation(table),
        table=table,
    )


def resolve_execution_clock_metadata(
    *,
    reader: CausalExecutionOutcomeDevelopmentReader,
    manifest_ref: str,
    sessions: tuple[date, ...],
    history_margin_sessions: int = 0,
) -> ExecutionClockContext:
    """Resolve temporal authority from one row per formation, without returns.

    The execution owner validates that every listing carried the same schedule
    before returning this projection.  Preflight therefore reuses the exact
    snapshot and method seal while avoiding the listing-granularity numerical
    surface that only execution consumes.
    """
    if history_margin_sessions < 0:
        raise PortfolioInputError("portfolio_strategy_lab.execution_clock_margin_invalid")
    schedule = reader.read_development_schedule(manifest_ref)
    formations = cast(list[date], schedule["formation_session"].to_pylist())
    holding_ends = cast(list[date], schedule["holding_end_session"].to_pylist())
    available = tuple(
        formation
        for formation, holding_end in zip(formations, holding_ends, strict=True)
        if holding_end <= max(sessions)
    )
    try:
        first = available.index(sessions[0])
    except ValueError as error:
        raise PortfolioInputError("portfolio_strategy_lab.exposure_margin_unavailable") from error
    if first < history_margin_sessions:
        raise PortfolioInputError("portfolio_strategy_lab.exposure_margin_unavailable")
    owner_sessions = (
        available[first - history_margin_sessions : first] + sessions
        if history_margin_sessions
        else sessions
    )
    table = schedule.filter(
        pc.is_in(
            schedule["formation_session"],
            value_set=pa.array(owner_sessions, type=pa.date32()),
        )
    )
    if table.num_rows != len(owner_sessions):
        raise PortfolioInputError("portfolio_strategy_lab.execution_events_session_absent")
    events = _events_from_table(
        table=table,
        reader=reader,
        manifest_ref=manifest_ref,
        sessions=sessions,
    )
    return ExecutionClockContext(
        events=events,
        session_clocks=session_clocks_from_execution_rows(table),
        entry_session_by_formation=entry_session_by_formation(table),
        table=table,
    )


def _events_from_table(
    *,
    table: pa.Table,
    reader: CausalOutcomeDevelopmentRows,
    manifest_ref: str,
    sessions: tuple[date, ...],
) -> PortfolioExecutionEvents:
    """The events, built from a table its caller already read.

    Separate from the read so the same table serves the events, the session
    clocks and the formation-to-entry projection. Its one caller is
    ``resolve_execution_clock_context``, which is what makes "one read per
    replay" a property of the code rather than of the call order.
    """

    try:
        return execution_events_from_schedule(
            table=table,
            authority=reader,
            manifest_ref=manifest_ref,
            sessions=sessions,
        )
    except ValueError as error:
        raise PortfolioInputError(str(error)) from error


def execution_axis_for(
    *,
    reader: CausalOutcomeDevelopmentRows,
    manifest_ref: str,
    formation_sessions: tuple[date, ...],
    margin_sessions: int,
) -> tuple[date, ...]:
    """The campaign axis with its history margin, taken from the owner's axis.

    Derived rather than stored, so a replay reproduces it from the graph's own
    formation axis and the Host's declared margin instead of trusting an axis
    written beside the events it is supposed to check.
    """
    if not margin_sessions:
        return formation_sessions
    available = reader.available_development_sessions(
        manifest_ref, holding_end_through=max(formation_sessions)
    )
    try:
        first = available.index(formation_sessions[0])
    except ValueError as error:
        raise PortfolioInputError("portfolio_strategy_lab.exposure_margin_unavailable") from error
    if first < margin_sessions:
        raise PortfolioInputError("portfolio_strategy_lab.exposure_margin_unavailable")
    return tuple(available[first - margin_sessions : first]) + formation_sessions


__all__ = ["PortfolioInputError", "resolve_execution_clock_metadata"]
