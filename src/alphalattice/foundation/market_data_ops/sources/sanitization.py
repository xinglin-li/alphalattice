"""Provider payload validation before any local database write."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.sources.price_integrity import (
    SplitAdjustedPriceIntegrityError,
    validate_split_adjusted_history,
)


class CorruptedPayload(ValueError):
    """A provider payload that cannot become trusted market observations."""

    def __init__(self, code: str, detail: str) -> None:
        """Create an error with a stable failure code and detail.

        Args:
            code: Machine-readable payload failure code.
            detail: Human-readable explanation of the rejected payload.

        """
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class SanitizedBatch:
    """Validated raw bars and corporate actions from one provider batch."""

    provider: str
    bars: tuple[RawDailyBar, ...]
    actions: tuple[CorporateActionEvent, ...]


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise CorruptedPayload("CORRUPTED_PAYLOAD", f"{field} is boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CorruptedPayload("CORRUPTED_PAYLOAD", f"{field} is not numeric") from exc
    if not math.isfinite(number):
        raise CorruptedPayload("CORRUPTED_PAYLOAD", f"{field} is not finite")
    return number


def _session(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise CorruptedPayload("CORRUPTED_PAYLOAD", "session_date is invalid")


def sanitize_payload(
    manifest: UniverseManifest,
    provider: str,
    payload: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_symbols: Sequence[str],
) -> SanitizedBatch:
    """Convert a provider-shaped payload to typed raw observations.

    An HTTP success with an empty, partial, duplicated, non-finite, or
    internally inconsistent payload is still a hard failure.
    """
    normalized_expected = tuple(sorted({symbol.strip().upper() for symbol in expected_symbols}))
    if not normalized_expected:
        raise ValueError("expected_symbols must not be empty")
    actual = {symbol.strip().upper() for symbol in payload}
    missing = sorted(set(normalized_expected) - actual)
    unexpected = sorted(actual - set(normalized_expected))
    if missing or unexpected:
        raise CorruptedPayload("SYMBOL_MISMATCH", f"missing={missing}, unexpected={unexpected}")
    bars: list[RawDailyBar] = []
    actions: list[CorporateActionEvent] = []
    for symbol in normalized_expected:
        listing = manifest.listing_for_symbol(symbol)
        if listing is None:
            raise ValueError(f"{symbol} is not in the frozen manifest")
        rows = payload[symbol]
        if not rows:
            raise CorruptedPayload("CORRUPTED_PAYLOAD", f"{symbol} returned no rows")
        try:
            validate_split_adjusted_history(rows, symbol=symbol)
        except SplitAdjustedPriceIntegrityError as exc:
            raise CorruptedPayload(exc.code, str(exc)) from exc
        seen_sessions: set[date] = set()
        for row in rows:
            session = _session(row.get("session_date"))
            if session in seen_sessions:
                raise CorruptedPayload("DUPLICATE_SESSION", f"{symbol} {session.isoformat()}")
            seen_sessions.add(session)
            open_value = _number(row.get("open"), "open")
            high = _number(row.get("high"), "high")
            low = _number(row.get("low"), "low")
            close = _number(row.get("close"), "close")
            volume_value = _number(row.get("volume"), "volume")
            if (
                min(open_value, high, low, close) <= 0
                or low > min(open_value, close)
                or high < max(open_value, close)
                or high < low
            ):
                raise CorruptedPayload("INVALID_OHLC", f"{symbol} {session.isoformat()}")
            if volume_value < 0 or not volume_value.is_integer():
                raise CorruptedPayload("INVALID_VOLUME", f"{symbol} {session.isoformat()}")
            bars.append(
                RawDailyBar(
                    listing_id=listing.listing_id,
                    provider=provider,
                    session_date=session,
                    open=open_value,
                    high=high,
                    low=low,
                    close=close,
                    volume=int(volume_value),
                )
            )
            # Provider payloads use ``split_ratio`` as their wire vocabulary;
            # the canonical record explicitly names its new/old share meaning.
            split = row.get("split_ratio")
            # yfinance represents an ordinary no-split session as 0.0.
            if split not in (None, 0, 0.0, 1, 1.0):
                ratio = _number(split, "split_ratio")
                if ratio <= 0:
                    raise CorruptedPayload("INVALID_ACTION", f"{symbol} split_ratio")
                actions.append(
                    CorporateActionEvent(
                        listing.listing_id,
                        provider,
                        session,
                        "SPLIT",
                        new_shares_per_old_share=ratio,
                    )
                )
            dividend = row.get("cash_dividend")
            if dividend not in (None, 0, 0.0):
                amount = _number(dividend, "cash_dividend")
                if amount < 0:
                    raise CorruptedPayload("INVALID_ACTION", f"{symbol} cash_dividend")
                actions.append(
                    CorporateActionEvent(
                        listing.listing_id, provider, session, "CASH_DIVIDEND", cash_amount=amount
                    )
                )
            # These observations are retained as evidence, but the projection
            # refuses to infer their share-unit semantics.  They therefore
            # fail closed at feature admission rather than disappearing.
            for field, action_kind in (
                ("capital_gain", "CAPITAL_GAIN"),
                ("spin_off", "SPIN_OFF"),
            ):
                value = row.get(field)
                if value not in (None, 0, 0.0):
                    amount = _number(value, field)
                    if amount < 0:
                        raise CorruptedPayload("INVALID_ACTION", f"{symbol} {field}")
                    actions.append(
                        CorporateActionEvent(
                            listing.listing_id,
                            provider,
                            session,
                            action_kind,
                            cash_amount=amount,
                        )
                    )
    return SanitizedBatch(
        provider=provider,
        bars=tuple(sorted(bars, key=lambda item: (item.listing_id, item.session_date))),
        actions=tuple(
            sorted(
                actions, key=lambda item: (item.listing_id, item.effective_date, item.action_kind)
            )
        ),
    )
