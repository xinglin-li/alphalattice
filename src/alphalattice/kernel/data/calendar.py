"""Deterministic XNYS/XNAS schedule materialization and hashing."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime
from importlib import import_module
from typing import Any, Final, Protocol, cast

import pyarrow as pa

from alphalattice.kernel.data.errors import DataQualityError

CALENDAR_SCHEMA_VERSION: Final = "calendar-schedule-v1"
SUPPORTED_CALENDARS: Final = frozenset({"XNYS", "XNAS"})


class _ExchangeCalendar(Protocol):
    schedule: Any

    def sessions_in_range(self, start: date, end: date) -> Iterable[Any]: ...


def calendar_schedule_schema() -> pa.Schema:
    """Return the frozen Arrow schema for qualified exchange sessions."""
    timestamp = pa.timestamp("us", tz="UTC")
    return pa.schema(
        [
            pa.field("calendar_id", pa.string(), nullable=False),
            pa.field("session_date", pa.date32(), nullable=False),
            pa.field("session_open_timestamp", timestamp, nullable=False),
            pa.field("session_close_timestamp", timestamp, nullable=False),
        ]
    )


def _python_datetime(value: Any, field: str) -> datetime:
    conversion = getattr(value, "to_pydatetime", None)
    result = conversion() if callable(conversion) else value
    if not isinstance(result, datetime) or result.tzinfo is None or result.utcoffset() is None:
        raise DataQualityError(
            f"{field} is not timezone-aware",
            code="data.provider_schema_drift",
        )
    return result.astimezone(UTC)


def _python_date(value: Any) -> date:
    conversion = getattr(value, "date", None)
    result = conversion() if callable(conversion) else value
    if isinstance(result, datetime):
        return result.date()
    if not isinstance(result, date):
        raise DataQualityError(
            "calendar session label is invalid", code="data.provider_schema_drift"
        )
    return result


def materialize_calendar_schedule(
    calendar_ids: Iterable[str],
    *,
    start: date,
    end: date,
    as_of_timestamp: datetime,
) -> pa.Table:
    """Materialize closed XNYS/XNAS sessions through an explicit as-of clock."""
    if as_of_timestamp.tzinfo is None or as_of_timestamp.utcoffset() is None:
        raise DataQualityError(
            "as_of_timestamp must be timezone-aware", code="data.provider_schema_drift"
        )
    if end < start:
        raise DataQualityError("calendar end precedes start", code="data.provider_schema_drift")
    normalized = tuple(sorted({item.strip().upper() for item in calendar_ids}))
    if not normalized or not set(normalized).issubset(SUPPORTED_CALENDARS):
        raise DataQualityError(
            "unknown or empty calendar mapping", code="data.provider_schema_drift"
        )

    module = vars(import_module("exchange_calendars"))
    get_calendar = cast(Any, module["get_calendar"])
    rows: list[dict[str, object]] = []
    as_of_utc = as_of_timestamp.astimezone(UTC)
    for calendar_id in normalized:
        calendar = cast(_ExchangeCalendar, get_calendar(calendar_id))
        sessions = calendar.sessions_in_range(start, end)
        # The installed calendar's scalar open/close APIs read these same
        # columns. Select them once, keeping its range and our value checks.
        clocks = calendar.schedule.loc[sessions, ["open", "close"]]
        for session, opened, closed in clocks.itertuples(index=True, name=None):
            session_date = _python_date(session)
            session_open = _python_datetime(opened, "session_open")
            session_close = _python_datetime(closed, "session_close")
            if session_close > as_of_utc:
                continue
            rows.append(
                {
                    "calendar_id": calendar_id,
                    "session_date": session_date,
                    "session_open_timestamp": session_open,
                    "session_close_timestamp": session_close,
                }
            )
    if not rows:
        raise DataQualityError("calendar schedule is empty", code="data.snapshot_incomplete")
    table = pa.Table.from_pylist(rows, schema=calendar_schedule_schema())
    return canonicalize_calendar_schedule(table)


def canonicalize_calendar_schedule(table: pa.Table) -> pa.Table:
    """Check schema and unique sessions, then sort the schedule canonically."""
    if table.schema != calendar_schedule_schema():
        raise DataQualityError(
            "calendar Arrow schema differs from frozen schema",
            code="data.provider_schema_drift",
        )
    canonical = table.combine_chunks().sort_by(
        [("calendar_id", "ascending"), ("session_date", "ascending")]
    )
    keys = [
        (row["calendar_id"], row["session_date"])
        for row in canonical.select(["calendar_id", "session_date"]).to_pylist()
    ]
    if len(keys) != len(set(keys)):
        raise DataQualityError(
            "calendar schedule contains duplicates", code="data.provider_schema_drift"
        )
    return canonical
