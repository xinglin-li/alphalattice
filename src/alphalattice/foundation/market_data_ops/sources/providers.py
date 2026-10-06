"""Provider boundary for the local pre-Factor Research playpen workflow.

The interface returns provider-shaped observations only.  Sanitization,
identity scope, persistence, action reconciliation, and snapshot admission stay
with deterministic AlphaLattice-owned code.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, runtime_checkable

from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
)
from alphalattice.foundation.market_data_ops.sources.price_integrity import (
    SplitAdjustedPriceIntegrityError,
    validate_split_adjusted_history,
)

YFINANCE_PRICE_POLICY: Mapping[str, object] = {
    "actions": True,
    "auto_adjust": False,
    "repair": True,
    "split_integrity": "action-aligned-open-ratio-and-unexplained-fivefold-discontinuity",
}
YFINANCE_PRICE_POLICY_HASH = hashlib.sha256(
    json.dumps(YFINANCE_PRICE_POLICY, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


class ProviderFetchError(RuntimeError):
    """A safe provider observation that can be grouped or remediated by the host."""

    def __init__(self, code: str, detail: str, *, retryable: bool) -> None:
        """Record a stable failure code and whether the host may retry.

        Args:
            code: Machine-readable provider failure code.
            detail: Safe explanation of the provider observation.
            retryable: Whether another attempt may resolve the failure.

        """
        super().__init__(detail)
        self.code = code
        self.retryable = retryable


class MarketDataProvider(Protocol):
    """Minimal provider capability needed by the deterministic data operation."""

    name: str

    def fetch_daily(
        self, symbols: Sequence[str], *, start: date, end: date
    ) -> Mapping[str, Sequence[Mapping[str, object]]]:
        """Fetch provider-shaped daily rows for the requested symbols and dates."""
        ...

    def fetch_action_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ) -> Sequence[CorporateActionEvent]:
        """Fetch the provider's corporate actions over a bounded date range."""
        ...

    def fetch_adjusted_close_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ) -> Sequence[ProviderAdjustedClosePoint]:
        """Fetch adjusted closes for diagnostics outside the raw-bar identity."""
        ...


@dataclass(frozen=True)
class HydrationEvidence:
    """One bounded history response split into governed evidence surfaces.

    Adjusted closes remain ephemeral diagnostic input.  They are never part of
    the canonical raw-bar payload or its revision identity.
    """

    daily_rows: tuple[Mapping[str, object], ...]
    actions: tuple[CorporateActionEvent, ...]
    adjusted_closes: tuple[ProviderAdjustedClosePoint, ...]
    repaired_sessions: tuple[date, ...] = ()
    provider_policy_hash: str | None = None


@runtime_checkable
class HistoricalHydrationProvider(Protocol):
    """Optional single-response history capability used by full onboarding."""

    name: str

    def fetch_hydration(
        self,
        *,
        listing_id: str,
        provider_symbol: str,
        start: date,
        end: date,
    ) -> HydrationEvidence:
        """Fetch one bounded response with bars, actions, and diagnostics."""
        ...


@dataclass(frozen=True)
class SectorObservation:
    """A current-provider classification, deliberately not a historical taxonomy."""

    provider: str
    provider_symbol: str
    sector_name: str
    sector_key: str | None
    payload_hash: str


class SectorReferenceProvider(Protocol):
    """Optional, rate-governed sector capability separate from price retrieval."""

    name: str

    def fetch_current_sector(self, *, provider_symbol: str) -> SectorObservation:
        """Fetch the provider's current sector label for one symbol."""
        ...


def _plain_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _error(exc: Exception) -> ProviderFetchError:
    detail = str(exc).strip()
    normalized = detail.casefold()
    if isinstance(exc, TimeoutError) or "timeout" in normalized:
        return ProviderFetchError(
            "data.provider_timeout", detail or type(exc).__name__, retryable=True
        )
    if "429" in normalized or "rate limit" in normalized:
        return ProviderFetchError("data.rate_limited", detail or type(exc).__name__, retryable=True)
    if "curl" in normalized and any(
        marker in normalized
        for marker in ("already in use", "bad handle", "multi handle", "session")
    ):
        return ProviderFetchError(
            "data.provider_session_unstable",
            detail or type(exc).__name__,
            retryable=True,
        )
    if "database is locked" in normalized:
        # yfinance 1.5.1 lazily creates process-global Peewee cache tables.
        # A first-use race is transport infrastructure instability, not a
        # permanent per-listing data failure.
        return ProviderFetchError(
            "data.provider_session_unstable",
            detail or type(exc).__name__,
            retryable=True,
        )
    return ProviderFetchError(
        "data.provider_fetch_failed", detail or type(exc).__name__, retryable=False
    )


@dataclass
class YFinanceMarketDataProvider:
    """Explicit yfinance adapter with per-symbol evidence and host-owned concurrency."""

    cache_root: Path
    name: str = "yfinance"
    _runtime_module: Any = field(default=None, init=False, repr=False)
    _initialization_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def execution_identity(self) -> Mapping[str, object]:
        """Return the exact provider behavior that can change stored observations."""
        try:
            package_version = version("yfinance")
        except PackageNotFoundError:  # pragma: no cover - diagnosed before provider use
            package_version = "unavailable"
        return {
            "provider": self.name,
            "adapter": f"{type(self).__module__}.{type(self).__qualname__}",
            "package": "yfinance",
            "package_version": package_version,
            "price_policy_hash": YFINANCE_PRICE_POLICY_HASH,
        }

    def _module(self) -> Any:
        if self._runtime_module is not None:
            return self._runtime_module
        with self._initialization_lock:
            if self._runtime_module is not None:
                return self._runtime_module
            try:
                import yfinance as yf
            except ImportError as exc:  # pragma: no cover - environment diagnosis
                raise ProviderFetchError(
                    "data.provider_unavailable",
                    "yfinance is not installed in the selected Python environment",
                    retryable=False,
                ) from exc
            self.cache_root.mkdir(parents=True, exist_ok=True)
            try:
                yf.cache.set_cache_location(str(self.cache_root))
                # Exact-version yfinance source shows that timezone, cookie,
                # and ISIN caches each lazily run SQLite DDL.  Initialize them
                # once under the provider lock before host workers fan out.
                for getter_name in (
                    "get_tz_cache",
                    "get_cookie_cache",
                    "get_isin_cache",
                ):
                    getattr(yf.cache, getter_name)().initialise()
            except Exception as exc:
                raise _error(exc) from exc
            self._runtime_module = yf
            return yf

    def fetch_daily(
        self, symbols: Sequence[str], *, start: date, end: date
    ) -> Mapping[str, Sequence[Mapping[str, object]]]:
        """Fetch split-repaired daily history for each requested symbol.

        Raises:
            ValueError: The requested date range is reversed.
            ProviderFetchError: A provider response is missing or invalid.

        """
        if start > end:
            raise ValueError("daily provider range start is after end")
        yf = self._module()
        payload: dict[str, Sequence[Mapping[str, object]]] = {}
        for symbol in symbols:
            try:
                frame = yf.Ticker(symbol).history(
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    auto_adjust=False,
                    actions=True,
                    repair=True,
                    raise_errors=True,
                )
            except ProviderFetchError:
                raise
            except Exception as exc:
                raise _error(exc) from exc
            if frame.empty:
                raise ProviderFetchError(
                    "data.empty_payload",
                    f"yfinance returned no daily rows for {symbol}",
                    retryable=False,
                )
            rows = self._daily_rows(frame, start=start, end=end)
            self._validate_price_history(rows, symbol=symbol)
            payload[symbol] = rows
        return payload

    def fetch_hydration(
        self,
        *,
        listing_id: str,
        provider_symbol: str,
        start: date,
        end: date,
    ) -> HydrationEvidence:
        """Fetch raw, actions, and adjusted-close diagnostics from one chart response."""
        if start > end:
            raise ValueError("hydration provider range start is after end")
        yf = self._module()
        try:
            frame = yf.Ticker(provider_symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                actions=True,
                repair=True,
                raise_errors=True,
            )
        except Exception as exc:
            raise _error(exc) from exc
        if frame.empty:
            raise ProviderFetchError(
                "data.empty_payload",
                f"yfinance returned no hydration rows for {provider_symbol}",
                retryable=False,
            )
        if "Adj Close" not in frame:
            raise ProviderFetchError(
                "data.adjusted_close_unavailable",
                f"yfinance returned no adjusted-close history for {provider_symbol}",
                retryable=False,
            )
        daily_rows = self._daily_rows(frame, start=start, end=end)
        self._validate_price_history(daily_rows, symbol=provider_symbol)
        return HydrationEvidence(
            daily_rows=daily_rows,
            actions=self._actions_from_frame(
                frame,
                listing_id=listing_id,
                provider_symbol=provider_symbol,
                start=start,
                end=end,
            ),
            adjusted_closes=self._adjusted_closes_from_frame(
                frame,
                listing_id=listing_id,
                provider_symbol=provider_symbol,
                start=start,
                end=end,
            ),
            repaired_sessions=self._repaired_sessions_from_frame(
                frame,
                start=start,
                end=end,
            ),
            provider_policy_hash=YFINANCE_PRICE_POLICY_HASH,
        )

    def fetch_adjusted_close_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ) -> Sequence[ProviderAdjustedClosePoint]:
        """Fetch the provider's full adjusted-close view for one audit only.

        The returned series is intentionally not staged into canonical bars:
        Yahoo can recalculate its entire historical adjustment view after a
        later dividend or correction.
        """
        if start > end:
            raise ValueError("adjusted-close provider range start is after end")
        yf = self._module()
        try:
            frame = yf.Ticker(provider_symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                actions=True,
                repair=True,
                raise_errors=True,
            )
        except Exception as exc:
            raise _error(exc) from exc
        if frame.empty or "Adj Close" not in frame:
            raise ProviderFetchError(
                "data.adjusted_close_unavailable",
                f"yfinance returned no adjusted-close history for {provider_symbol}",
                retryable=False,
            )
        return self._adjusted_closes_from_frame(
            frame,
            listing_id=listing_id,
            provider_symbol=provider_symbol,
            start=start,
            end=end,
        )

    def _adjusted_closes_from_frame(
        self,
        frame: Any,
        *,
        listing_id: str,
        provider_symbol: str,
        start: date,
        end: date,
    ) -> tuple[ProviderAdjustedClosePoint, ...]:
        points: list[ProviderAdjustedClosePoint] = []
        for index, row in frame.iterrows():
            session_date = index.date()
            if not start <= session_date <= end:
                continue
            adjusted_close = _plain_number(row.get("Adj Close"))
            if adjusted_close is None or adjusted_close <= 0:
                raise ProviderFetchError(
                    "data.invalid_adjusted_close_payload",
                    "yfinance returned an invalid adjusted close for "
                    f"{provider_symbol} on {session_date.isoformat()}",
                    retryable=False,
                )
            points.append(
                ProviderAdjustedClosePoint(
                    listing_id=listing_id,
                    provider=self.name,
                    session_date=session_date,
                    adjusted_close=adjusted_close,
                )
            )
        return tuple(points)

    def fetch_action_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ) -> Sequence[CorporateActionEvent]:
        """Fetch corporate actions for one listing in the bounded date range.

        Raises:
            ValueError: The requested date range is reversed.
            ProviderFetchError: The provider request fails.

        """
        if start > end:
            raise ValueError("action provider range start is after end")
        yf = self._module()
        try:
            frame = yf.Ticker(provider_symbol).actions
        except Exception as exc:
            raise _error(exc) from exc
        if frame is None or frame.empty:
            return ()
        return self._actions_from_frame(
            frame,
            listing_id=listing_id,
            provider_symbol=provider_symbol,
            start=start,
            end=end,
        )

    def _daily_rows(
        self,
        frame: Any,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        return tuple(
            {
                "session_date": index.date().isoformat(),
                "open": _plain_number(row.get("Open")),
                "high": _plain_number(row.get("High")),
                "low": _plain_number(row.get("Low")),
                "close": _plain_number(row.get("Close")),
                "volume": _plain_number(row.get("Volume")),
                "split_ratio": _plain_number(row.get("Stock Splits")),
                "cash_dividend": _plain_number(row.get("Dividends")),
                "capital_gain": _plain_number(row.get("Capital Gains")),
            }
            for index, row in frame.iterrows()
            if (start is None or index.date() >= start) and (end is None or index.date() <= end)
        )

    @staticmethod
    def _validate_price_history(rows: Sequence[Mapping[str, object]], *, symbol: str) -> None:
        try:
            validate_split_adjusted_history(rows, symbol=symbol)
        except SplitAdjustedPriceIntegrityError as exc:
            raise ProviderFetchError(exc.code, str(exc), retryable=False) from exc

    @staticmethod
    def _repaired_sessions_from_frame(
        frame: Any,
        *,
        start: date,
        end: date,
    ) -> tuple[date, ...]:
        if "Repaired?" not in frame:
            return ()
        return tuple(
            index.date()
            for index, value in frame["Repaired?"].items()
            if start <= index.date() <= end and bool(value)
        )

    def _actions_from_frame(
        self,
        frame: Any,
        *,
        listing_id: str,
        provider_symbol: str,
        start: date,
        end: date,
    ) -> tuple[CorporateActionEvent, ...]:
        events: list[CorporateActionEvent] = []
        for index, row in frame.iterrows():
            effective_date = index.date()
            if not start <= effective_date <= end:
                continue
            split = _plain_number(row.get("Stock Splits"))
            dividend = _plain_number(row.get("Dividends"))
            capital_gain = _plain_number(row.get("Capital Gains"))
            if split not in (None, 0.0, 1.0):
                if split <= 0:
                    raise ProviderFetchError(
                        "data.invalid_action_payload",
                        f"yfinance returned an invalid split for {provider_symbol}",
                        retryable=False,
                    )
                events.append(
                    CorporateActionEvent(
                        listing_id,
                        self.name,
                        effective_date,
                        "SPLIT",
                        new_shares_per_old_share=split,
                    )
                )
            if dividend not in (None, 0.0):
                if dividend < 0:
                    raise ProviderFetchError(
                        "data.invalid_action_payload",
                        f"yfinance returned a negative dividend for {provider_symbol}",
                        retryable=False,
                    )
                events.append(
                    CorporateActionEvent(
                        listing_id,
                        self.name,
                        effective_date,
                        "CASH_DIVIDEND",
                        cash_amount=dividend,
                    )
                )
            if capital_gain not in (None, 0.0):
                if capital_gain < 0:
                    raise ProviderFetchError(
                        "data.invalid_action_payload",
                        f"yfinance returned a negative capital gain for {provider_symbol}",
                        retryable=False,
                    )
                events.append(
                    CorporateActionEvent(
                        listing_id,
                        self.name,
                        effective_date,
                        "CAPITAL_GAIN",
                        cash_amount=capital_gain,
                    )
                )
        return tuple(events)

    def fetch_current_sector(self, *, provider_symbol: str) -> SectorObservation:
        """Read Yahoo's current sector label without calling it GICS or PIT truth."""
        yf = self._module()
        try:
            payload = yf.Ticker(provider_symbol).get_info()
        except Exception as exc:
            raise _error(exc) from exc
        if not isinstance(payload, Mapping):
            raise ProviderFetchError(
                "sector.invalid_payload",
                f"yfinance returned no info object for {provider_symbol}",
                retryable=False,
            )
        sector = payload.get("sector")
        if not isinstance(sector, str) or not sector.strip():
            raise ProviderFetchError(
                "sector.missing_current_sector",
                f"yfinance returned no current sector for {provider_symbol}",
                retryable=False,
            )
        sector_key = payload.get("sectorKey")
        normalized = {
            "sector": sector.strip(),
            "sectorKey": sector_key.strip() if isinstance(sector_key, str) else None,
        }
        import hashlib
        import json

        return SectorObservation(
            provider=self.name,
            provider_symbol=provider_symbol,
            sector_name=normalized["sector"],
            sector_key=normalized["sectorKey"],
            payload_hash=hashlib.sha256(
                json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        )
