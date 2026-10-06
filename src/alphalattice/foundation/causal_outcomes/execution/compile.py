"""Pure causal execution schedule and row compilation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pyarrow as pa

from alphalattice.foundation.market_data_ops.returns.execution import open_to_open_simple_return
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)
from alphalattice.kernel.data.enums import ExecutionSessionStatus, ExecutionVenueStatus
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import CausalExecutionSchedulePoint
from .methods import (
    ExecutionOutcomeMethodRecipe,
    ExecutionOutcomeSessionAxis,
    period_dividend_for_point,
)

_SCHEMA_ID = "desktop-causal-execution-outcome"
_ELIGIBLE = {
    ExecutionSessionStatus.VERIFIED_ELIGIBLE,
    ExecutionSessionStatus.ASSUMED_ELIGIBLE_FROM_DAILY_BAR,
}


def _daily_session_axis(sessions: tuple[date, ...], *, minimum_sessions: int) -> tuple[date, ...]:
    """Admit one ordered session axis long enough for the requested method.

    The minimum used to be the literal three a two-point span needs. It is now
    supplied by the caller from its resolved recipe, because the axis length a
    schedule requires is a property of the method, not of this module.
    """
    ordered = tuple(sorted(set(sessions)))
    if minimum_sessions < 3:
        raise ValueError("causal execution session minimum is invalid")
    if ordered != sessions or len(ordered) < minimum_sessions:
        raise ValueError(
            "causal execution sessions must be sorted, unique, and cover the method span"
        )
    return ordered


def _schema() -> pa.Schema:
    timestamp = pa.timestamp("us", tz="UTC")
    return pa.schema(
        [
            pa.field("listing_id", pa.string(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("sequence", pa.int32(), nullable=False),
            pa.field("formation_session", pa.date32(), nullable=False),
            pa.field("formation_close_at", timestamp, nullable=False),
            pa.field("entry_session", pa.date32(), nullable=False),
            pa.field("entry_open_at", timestamp, nullable=False),
            pa.field("holding_end_session", pa.date32(), nullable=False),
            pa.field("holding_end_open_at", timestamp, nullable=False),
            pa.field("actual_session_span", pa.int16(), nullable=False),
            pa.field("entry_status", pa.string(), nullable=False),
            pa.field("holding_end_status", pa.string(), nullable=False),
            pa.field("entry_open_split_adjusted", pa.float64(), nullable=True),
            pa.field("holding_end_open_split_adjusted", pa.float64(), nullable=True),
            pa.field("period_dividend_split_adjusted", pa.float64(), nullable=True),
            pa.field("simple_return", pa.float64(), nullable=True),
            pa.field("entry_source_row_hash", pa.string(), nullable=False),
            pa.field("holding_end_source_row_hash", pa.string(), nullable=False),
            pa.field("row_hash", pa.string(), nullable=False),
        ]
    )


def execution_session_status(
    bar: RawDailyBar | None,
    *,
    venue_status: ExecutionVenueStatus | None = None,
) -> ExecutionSessionStatus:
    """Classify execution eligibility from an official bar and venue status.

    Args:
        bar: Daily observation, if one exists for the session.
        venue_status: Explicit venue eligibility or restriction, if known.

    Returns:
        The verified, assumed, missing, halted, or unknown session status.

    """
    if venue_status is ExecutionVenueStatus.HALTED_OR_MARKET_RESTRICTED:
        return ExecutionSessionStatus.HALTED_OR_MARKET_RESTRICTED
    if bar is None:
        return ExecutionSessionStatus.NO_OFFICIAL_OPEN
    if (
        math.isfinite(float(bar.open))
        and float(bar.open) > 0.0
        and bar.volume is not None
        and int(bar.volume) > 0
    ):
        return (
            ExecutionSessionStatus.VERIFIED_ELIGIBLE
            if venue_status is ExecutionVenueStatus.VERIFIED_ELIGIBLE
            else ExecutionSessionStatus.ASSUMED_ELIGIBLE_FROM_DAILY_BAR
        )
    return ExecutionSessionStatus.ELIGIBILITY_UNKNOWN


def _action_dividends(
    actions: tuple[CorporateActionEvent, ...], *, through: date
) -> dict[date, float]:
    """Validate actions and return provider split-adjusted cash amounts.

    The yfinance daily contract already expresses OHLCV and action amounts on
    the current share basis. Reapplying later split multipliers here would
    double-adjust the execution path and recreate split-sized returns.
    """
    dividends: dict[date, float] = {}
    for event in actions:
        if event.effective_date > through:
            continue
        if event.action_kind == "CASH_DIVIDEND":
            value = event.cash_amount
            if value is None or not math.isfinite(value) or value < 0.0:
                raise ValueError("causal execution dividend is invalid")
            dividends[event.effective_date] = dividends.get(event.effective_date, 0.0) + value
        elif event.action_kind == "SPLIT":
            value = event.new_shares_per_old_share
            if value is None or not math.isfinite(value) or value <= 0.0:
                raise ValueError("causal execution split is invalid")
        elif event.action_kind in {"CAPITAL_GAIN", "SPIN_OFF"}:
            raise ValueError("causal execution encountered an unsupported corporate action")
    return dividends


def derive_causal_execution_row(
    *,
    listing_id: str,
    symbol: str,
    point: CausalExecutionSchedulePoint,
    entry_bar: RawDailyBar | None,
    holding_bar: RawDailyBar | None,
    period_dividend_split_adjusted: float,
    entry_venue_status: ExecutionVenueStatus | None = None,
    holding_venue_status: ExecutionVenueStatus | None = None,
) -> dict[str, object]:
    """Apply the tracked execution-status and open-return semantics to one triple."""
    entry_status = execution_session_status(entry_bar, venue_status=entry_venue_status)
    holding_status = execution_session_status(holding_bar, venue_status=holding_venue_status)
    if not math.isfinite(period_dividend_split_adjusted) or period_dividend_split_adjusted < 0.0:
        raise ValueError("causal execution dividend is invalid")
    adjusted_dividend = period_dividend_split_adjusted
    entry_open = float(entry_bar.open) if entry_bar is not None else None
    holding_open = float(holding_bar.open) if holding_bar is not None else None
    simple_return = None
    if (
        entry_status in _ELIGIBLE
        and holding_status in _ELIGIBLE
        and entry_open is not None
        and holding_open is not None
    ):
        simple_return = open_to_open_simple_return(
            entry_open=entry_open,
            exit_open=holding_open,
            period_dividend=adjusted_dividend,
        )
    # The identities below are canonical JSON of these payloads. A session or
    # instant is encoded by ``canonical_hash`` as ``str(value)`` (its
    # ``default``); handing it the same string up front produces the same
    # bytes without the encoder leaving C for every temporal member, which is
    # most of what a million-row publication used to spend its clock on.
    entry_session = str(point.entry_session)
    holding_end_session = str(point.holding_end_session)
    entry_source = {
        "listing_id": listing_id,
        "session_date": entry_session,
        "open_split_adjusted": entry_open,
        "volume_raw": entry_bar.volume if entry_bar is not None else None,
        "status": entry_status,
    }
    holding_source = {
        "listing_id": listing_id,
        "session_date": holding_end_session,
        "open_split_adjusted": holding_open,
        "volume_raw": holding_bar.volume if holding_bar is not None else None,
        "dividend_split_adjusted": adjusted_dividend,
        "status": holding_status,
    }
    identity: dict[str, object] = {
        "listing_id": listing_id,
        "symbol": symbol,
        "sequence": point.sequence,
        "formation_session": str(point.formation_session),
        "formation_close_at": str(point.formation_close_at),
        "entry_session": entry_session,
        "entry_open_at": str(point.entry_open_at),
        "holding_end_session": holding_end_session,
        "holding_end_open_at": str(point.holding_end_open_at),
        "actual_session_span": point.actual_session_span,
        "entry_status": entry_status.value,
        "holding_end_status": holding_status.value,
        "entry_open_split_adjusted": entry_open,
        "holding_end_open_split_adjusted": holding_open,
        "period_dividend_split_adjusted": adjusted_dividend,
        "simple_return": simple_return,
        "entry_source_row_hash": canonical_hash(entry_source),
        "holding_end_source_row_hash": canonical_hash(holding_source),
    }
    values: dict[str, object] = {
        **identity,
        "formation_session": point.formation_session,
        "formation_close_at": point.formation_close_at,
        "entry_session": point.entry_session,
        "entry_open_at": point.entry_open_at,
        "holding_end_session": point.holding_end_session,
        "holding_end_open_at": point.holding_end_open_at,
    }
    values["row_hash"] = canonical_hash(identity)
    return values


@dataclass(frozen=True)
class ListingExecutionRows:
    """One listing's derived rows, grouped for publication, plus their source bindings.

    ``tables`` holds the rows of each ``(split, formation year)`` chunk in
    point order; ``source_row_bindings`` is ``(listing_id, formation_session,
    entry_source_row_hash, holding_end_source_row_hash)`` per point, the
    formation session already in its canonical string form.
    """

    listing_id: str
    tables: dict[tuple[str, int], pa.Table]
    source_row_bindings: tuple[tuple[str, str, str, str], ...]


@dataclass(frozen=True)
class ListingRowDerivation:
    """The per-listing unit of a daily publication: pure, order-preserving, pickle-safe.

    Everything a listing's rows depend on besides its own bars and actions is
    fixed for the publication -- the schedule points, the recipe, the session
    axis and the sealed formations -- so one listing is one unit of work that
    a worker process can derive from the listing's bars and actions alone.
    Derived in this process or in another, the rows, their hashes and their
    order are the same: the row function is the one every caller uses.
    """

    points: tuple[CausalExecutionSchedulePoint, ...]
    recipe: ExecutionOutcomeMethodRecipe
    ordered_sessions: tuple[date, ...]
    sealed_formations: frozenset[date]

    def derive(
        self,
        *,
        listing_id: str,
        symbol: str,
        bars: Sequence[RawDailyBar],
        actions: Sequence[CorporateActionEvent],
    ) -> ListingExecutionRows:
        """Derive one listing's ordered execution rows and source bindings.

        Args:
            listing_id: Listing whose rows are being derived.
            symbol: Display symbol retained in each row.
            bars: Raw daily observations for the listing.
            actions: Corporate actions effective during the axis.

        Returns:
            Split and year tables with their source-row bindings.

        """
        axis = ExecutionOutcomeSessionAxis(self.ordered_sessions)
        through = self.ordered_sessions[-1]
        bar_by_session = {value.session_date: value for value in bars}
        dividends = _action_dividends(tuple(actions), through=through)
        rows_by_split_year: dict[tuple[str, int], list[dict[str, object]]] = {}
        bindings: list[tuple[str, str, str, str]] = []
        for point in self.points:
            row = derive_causal_execution_row(
                listing_id=listing_id,
                symbol=symbol,
                point=point,
                entry_bar=bar_by_session.get(point.entry_session),
                holding_bar=bar_by_session.get(point.holding_end_session),
                period_dividend_split_adjusted=period_dividend_for_point(
                    recipe=self.recipe,
                    dividends=dividends,
                    ordered_sessions=axis,
                    point=point,
                ),
            )
            split = (
                "SEALED_HOLDOUT"
                if point.formation_session in self.sealed_formations
                else "DEVELOPMENT"
            )
            rows_by_split_year.setdefault((split, point.formation_session.year), []).append(row)
            bindings.append(
                (
                    listing_id,
                    str(point.formation_session),
                    str(row["entry_source_row_hash"]),
                    str(row["holding_end_source_row_hash"]),
                )
            )
        schema = _schema()
        return ListingExecutionRows(
            listing_id=listing_id,
            tables={
                key: pa.Table.from_pylist(rows, schema=schema)
                for key, rows in rows_by_split_year.items()
            },
            source_row_bindings=tuple(bindings),
        )


_WORKER_DERIVATION: ListingRowDerivation | None = None


def _initialize_derivation_worker(derivation: ListingRowDerivation) -> None:
    global _WORKER_DERIVATION
    _WORKER_DERIVATION = derivation


def _derive_in_worker(
    payload: tuple[str, str, tuple[RawDailyBar, ...], tuple[CorporateActionEvent, ...]],
) -> ListingExecutionRows:
    if _WORKER_DERIVATION is None:
        raise RuntimeError("causal execution derivation worker is not initialized")
    listing_id, symbol, bars, actions = payload
    return _WORKER_DERIVATION.derive(
        listing_id=listing_id, symbol=symbol, bars=bars, actions=actions
    )


__all__ = [
    "ListingExecutionRows",
    "ListingRowDerivation",
    "derive_causal_execution_row",
    "execution_session_status",
]
