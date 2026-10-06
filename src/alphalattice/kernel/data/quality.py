"""Calendar-aware normalization, corporate actions, and deterministic quality gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from itertools import pairwise
from typing import Any, Literal, cast
from uuid import uuid4

import pyarrow as pa
import pyarrow.compute as pc

from alphalattice.kernel.data.calendar import canonicalize_calendar_schedule
from alphalattice.kernel.data.contracts import DataQualityReport, SymbolQualityReport
from alphalattice.kernel.data.errors import DataQualityError

MISSING_RATIO_LIMIT = 0.02
MAXIMUM_CONSECUTIVE_GAP = 20
EXTREME_MOVE_THRESHOLD = 0.50


def _anniversary(value: date, years: int = 10) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _maximum_missing_run(expected: Sequence[date], observed: set[date]) -> int:
    longest = 0
    current = 0
    for session in expected:
        if session in observed:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


_REPORT_COLUMNS = (
    "session_date",
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "volume_raw",
    "cash_dividend",
    "split_ratio",
    "availability_timestamp",
)
"""The daily table's columns a symbol's report reads, one value per row in table order."""


def _symbol_report(
    *,
    symbol: str,
    calendar_id: str,
    columns: Mapping[str, Sequence[Any]],
    expected: tuple[date, ...],
    as_of_timestamp: datetime,
) -> SymbolQualityReport:
    row_dates = list(columns["session_date"])
    unique_dates = set(row_dates)
    expected_set = set(expected)
    observed_expected = unique_dates & expected_set
    missing = tuple(item for item in expected if item not in observed_expected)
    expected_count = len(expected)
    missing_ratio = len(missing) / expected_count if expected_count else 1.0
    duplicate_rows = len(row_dates) - len(unique_dates)
    non_monotonic_rows = sum(current <= previous for previous, current in pairwise(row_dates))
    invalid_ohlc_rows = 0
    invalid_volume_rows = 0
    invalid_corporate_action_rows = 0
    extreme_move_rows = 0
    previous_close: float | None = None
    availability_after_as_of = 0
    for (
        open_raw,
        high_raw,
        low_raw,
        close_raw,
        volume,
        cash_dividend,
        split_ratio,
        available,
    ) in zip(
        columns["open_raw"],
        columns["high_raw"],
        columns["low_raw"],
        columns["close_raw"],
        columns["volume_raw"],
        columns["cash_dividend"],
        columns["split_ratio"],
        columns["availability_timestamp"],
        strict=True,
    ):
        open_value = float(open_raw)
        high_value = float(high_raw)
        low_value = float(low_raw)
        close_value = float(close_raw)
        if (
            min(open_value, high_value, low_value, close_value) <= 0.0
            or high_value < max(open_value, close_value)
            or low_value > min(open_value, close_value)
            or high_value < low_value
        ):
            invalid_ohlc_rows += 1
        if volume is not None and int(volume) < 0:
            invalid_volume_rows += 1
        dividend = float(cash_dividend)
        split = float(split_ratio)
        if dividend < 0.0 or split <= 0.0:
            invalid_corporate_action_rows += 1
        if (
            previous_close
            and abs(close_value / previous_close - 1.0) > EXTREME_MOVE_THRESHOLD
            and split == 1.0
            and dividend == 0.0
        ):
            extreme_move_rows += 1
        previous_close = close_value
        if available > as_of_timestamp:
            availability_after_as_of += 1

    reasons: list[str] = []
    first_session = min(unique_dates) if unique_dates else None
    last_session = max(unique_dates) if unique_dates else None
    if not expected or first_session is None or first_session > expected[0]:
        reasons.append("history_shorter_than_ten_calendar_years")
    if missing_ratio > MISSING_RATIO_LIMIT:
        reasons.append(f"missing_ratio_exceeds_0.02:{len(missing)}/{expected_count}")
    maximum_gap = _maximum_missing_run(expected, observed_expected)
    if maximum_gap > MAXIMUM_CONSECUTIVE_GAP:
        reasons.append(f"maximum_consecutive_gap_exceeds_20:{maximum_gap}")
    for label, count in (
        ("duplicate_rows", duplicate_rows),
        ("non_monotonic_rows", non_monotonic_rows),
        ("invalid_ohlc_rows", invalid_ohlc_rows),
        ("invalid_volume_rows", invalid_volume_rows),
        ("invalid_corporate_action_rows", invalid_corporate_action_rows),
        ("availability_after_as_of", availability_after_as_of),
    ):
        if count:
            reasons.append(f"{label}:{count}")

    if calendar_id not in {"XNYS", "XNAS"}:
        raise DataQualityError("unknown calendar mapping", code="data.provider_schema_drift")
    calendar_literal = cast(Literal["XNYS", "XNAS"], calendar_id)
    return SymbolQualityReport(
        symbol=symbol,
        calendar_id=calendar_literal,
        expected_sessions=expected_count,
        observed_sessions=len(observed_expected),
        missing_sessions=len(missing),
        missing_ratio=missing_ratio,
        maximum_consecutive_gap=maximum_gap,
        duplicate_rows=duplicate_rows,
        non_monotonic_rows=non_monotonic_rows,
        invalid_ohlc_rows=invalid_ohlc_rows,
        invalid_volume_rows=invalid_volume_rows,
        invalid_corporate_action_rows=invalid_corporate_action_rows,
        extreme_move_rows=extreme_move_rows,
        first_session=first_session,
        last_session=last_session,
        eligible=not reasons,
        reasons=tuple(reasons),
    )


def diagnose_daily_table(
    table: pa.Table,
    *,
    symbol_calendars: Mapping[str, str],
    schedule: pa.Table,
    as_of_timestamp: datetime,
    created_at: datetime,
) -> DataQualityReport:
    """Compare daily rows with the qualified ten-year session schedule.

    Read by column: a symbol's rows are the table's values at its row positions, in table
    order, and its expected sessions the schedule's sessions of its calendar from the
    anniversary on, in the schedule's canonical order -- the same rows and sessions a row
    mapping per table row and per schedule row gave, which cost a first use's onboarding
    about 20 s over its ~500 listings.
    """
    canonical_schedule = canonicalize_calendar_schedule(schedule)
    positions_by_symbol: dict[str, list[int]] = {
        symbol.strip().upper(): [] for symbol in symbol_calendars
    }
    normalized: dict[object, str] = {}
    for position, raw_symbol in enumerate(table.column("symbol").to_pylist()):
        symbol = normalized.get(raw_symbol)
        if symbol is None:
            symbol = normalized[raw_symbol] = str(raw_symbol).strip().upper()
        positions = positions_by_symbol.get(symbol)
        if positions is None:
            raise DataQualityError(
                f"table contains unexpected symbol {symbol}",
                code="data.symbol_mismatch",
            )
        positions.append(position)
    columns = {name: table.column(name).to_pylist() for name in _REPORT_COLUMNS}

    anniversary = _anniversary(as_of_timestamp.date())
    reports: list[SymbolQualityReport] = []
    for symbol in sorted(positions_by_symbol):
        calendar_id = symbol_calendars[symbol].strip().upper()
        expected = tuple(
            canonical_schedule.filter(
                pc.and_(
                    pc.equal(canonical_schedule.column("calendar_id"), calendar_id),
                    pc.greater_equal(canonical_schedule.column("session_date"), anniversary),
                )
            )
            .column("session_date")
            .to_pylist()
        )
        positions = positions_by_symbol[symbol]
        reports.append(
            _symbol_report(
                symbol=symbol,
                calendar_id=calendar_id,
                columns=(
                    columns
                    if len(positions) == table.num_rows
                    else {
                        name: [values[position] for position in positions]
                        for name, values in columns.items()
                    }
                ),
                expected=expected,
                as_of_timestamp=as_of_timestamp,
            )
        )
    caveats = tuple(
        f"{item.symbol}:extreme_move_rows={item.extreme_move_rows}"
        for item in reports
        if item.extreme_move_rows
    )
    return DataQualityReport(
        report_id=uuid4(),
        created_at=created_at,
        as_of_timestamp=as_of_timestamp,
        symbol_reports=tuple(reports),
        all_eligible=all(item.eligible for item in reports),
        caveats=caveats,
    )
