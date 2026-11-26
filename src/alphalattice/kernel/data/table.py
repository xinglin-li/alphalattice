"""Frozen Arrow schema, canonical table hashing, and Parquet bytes."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from typing import Any, Final

import pyarrow as pa

from alphalattice.kernel.data.errors import DataQualityError

WRITER_PROFILE: Final = "parquet-2.6-zstd3-rg65536-nodict-us-v1"
DAILY_SCHEMA_VERSION: Final = "daily-price-v1"
DAILY_PRICE_COLUMNS: Final = (
    "symbol",
    "calendar_id",
    "session_date",
    "observation_timestamp",
    "availability_timestamp",
    "availability_method",
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "volume_raw",
    "open_split_adjusted",
    "high_split_adjusted",
    "low_split_adjusted",
    "close_split_adjusted",
    "close_total_return_adjusted",
    "cash_dividend",
    "split_ratio",
    "provider",
    "attempt_id",
)


def daily_price_schema() -> pa.Schema:
    """Return the frozen Arrow schema for normalized daily prices."""
    timestamp = pa.timestamp("us", tz="UTC")
    return pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("calendar_id", pa.string(), nullable=False),
            pa.field("session_date", pa.date32(), nullable=False),
            pa.field("observation_timestamp", timestamp, nullable=False),
            pa.field("availability_timestamp", timestamp, nullable=False),
            pa.field("availability_method", pa.string(), nullable=False),
            pa.field("open_raw", pa.float64(), nullable=False),
            pa.field("high_raw", pa.float64(), nullable=False),
            pa.field("low_raw", pa.float64(), nullable=False),
            pa.field("close_raw", pa.float64(), nullable=False),
            pa.field("volume_raw", pa.int64(), nullable=True),
            pa.field("open_split_adjusted", pa.float64(), nullable=False),
            pa.field("high_split_adjusted", pa.float64(), nullable=False),
            pa.field("low_split_adjusted", pa.float64(), nullable=False),
            pa.field("close_split_adjusted", pa.float64(), nullable=False),
            pa.field("close_total_return_adjusted", pa.float64(), nullable=True),
            pa.field("cash_dividend", pa.float64(), nullable=False),
            pa.field("split_ratio", pa.float64(), nullable=False),
            pa.field("provider", pa.string(), nullable=False),
            pa.field("attempt_id", pa.string(), nullable=False),
        ]
    )


def _plain(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except ValueError:
            return value
    to_python = getattr(value, "to_pydatetime", None)
    return to_python() if callable(to_python) else value


def _utc(value: Any, field: str) -> datetime:
    plain = _plain(value)
    if not isinstance(plain, datetime) or plain.tzinfo is None or plain.utcoffset() is None:
        raise DataQualityError(f"{field} must be timezone-aware", code="data.provider_schema_drift")
    return plain.astimezone(UTC)


def _date(value: Any) -> date:
    plain = _plain(value)
    if isinstance(plain, datetime):
        return plain.date()
    if isinstance(plain, date):
        return plain
    if isinstance(plain, str):
        try:
            return date.fromisoformat(plain)
        except ValueError as exc:
            raise DataQualityError(
                "session_date is invalid", code="data.provider_schema_drift"
            ) from exc
    raise DataQualityError("session_date is invalid", code="data.provider_schema_drift")


def _float(value: Any, field: str, *, nullable: bool = False) -> float | None:
    plain = _plain(value)
    if plain is None and nullable:
        return None
    if plain is None:
        raise DataQualityError(f"{field} is required", code="data.provider_schema_drift")
    try:
        result = float(plain)
    except (TypeError, ValueError) as exc:
        raise DataQualityError(
            f"{field} is not numeric", code="data.provider_schema_drift"
        ) from exc
    if not math.isfinite(result):
        raise DataQualityError(f"{field} must be finite", code="data.provider_schema_drift")
    return 0.0 if result == 0.0 else result


def _int(value: Any, field: str) -> int | None:
    plain = _plain(value)
    if plain is None:
        return None
    if isinstance(plain, bool):
        raise DataQualityError(f"{field} is not an integer", code="data.provider_schema_drift")
    try:
        result = int(plain)
    except (TypeError, ValueError) as exc:
        raise DataQualityError(
            f"{field} is not an integer", code="data.provider_schema_drift"
        ) from exc
    if result != plain:
        raise DataQualityError(f"{field} is not an integer", code="data.provider_schema_drift")
    return result


def coerce_daily_records(records: Iterable[Mapping[str, Any]]) -> pa.Table:
    """Validate records without changing their order."""
    expected = set(DAILY_PRICE_COLUMNS)
    normalized: list[dict[str, Any]] = []
    for source in records:
        unknown = set(source) - expected
        missing = expected - set(source)
        if unknown or missing:
            raise DataQualityError(
                f"daily schema mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}",
                code="data.provider_schema_drift",
            )
        symbol = str(source["symbol"]).strip().upper()
        provider = str(source["provider"]).strip().lower()
        calendar_id = str(source["calendar_id"]).strip().upper()
        if not symbol or not provider or calendar_id not in {"XNYS", "XNAS"}:
            raise DataQualityError(
                "invalid symbol/provider/calendar", code="data.provider_schema_drift"
            )
        normalized.append(
            {
                "symbol": symbol,
                "calendar_id": calendar_id,
                "session_date": _date(source["session_date"]),
                "observation_timestamp": _utc(
                    source["observation_timestamp"], "observation_timestamp"
                ),
                "availability_timestamp": _utc(
                    source["availability_timestamp"], "availability_timestamp"
                ),
                "availability_method": str(source["availability_method"]),
                "open_raw": _float(source["open_raw"], "open_raw"),
                "high_raw": _float(source["high_raw"], "high_raw"),
                "low_raw": _float(source["low_raw"], "low_raw"),
                "close_raw": _float(source["close_raw"], "close_raw"),
                "volume_raw": _int(source["volume_raw"], "volume_raw"),
                "open_split_adjusted": _float(source["open_split_adjusted"], "open_split_adjusted"),
                "high_split_adjusted": _float(source["high_split_adjusted"], "high_split_adjusted"),
                "low_split_adjusted": _float(source["low_split_adjusted"], "low_split_adjusted"),
                "close_split_adjusted": _float(
                    source["close_split_adjusted"], "close_split_adjusted"
                ),
                "close_total_return_adjusted": _float(
                    source["close_total_return_adjusted"],
                    "close_total_return_adjusted",
                    nullable=True,
                ),
                "cash_dividend": _float(source["cash_dividend"], "cash_dividend"),
                "split_ratio": _float(source["split_ratio"], "split_ratio"),
                "provider": provider,
                "attempt_id": str(source["attempt_id"]),
            }
        )
    if not normalized:
        raise DataQualityError("daily table cannot be empty", code="data.snapshot_incomplete")
    return pa.Table.from_pylist(normalized, schema=daily_price_schema())
