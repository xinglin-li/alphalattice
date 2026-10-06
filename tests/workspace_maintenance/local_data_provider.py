"""The offline Data provider the update suites install on a Local Web session.

One recording provider over the seeded walk, with its as-of and its universe as
explicit arguments. Four suites used to drive it through module state: they
assigned ``NOW`` and ``SYMBOLS`` on the test module that defined ``_provider()``
-- through ``monkeypatch`` in the parent process, by plain assignment in the
crash children -- and the provider read both at call time. A caller now says
what the provider serves, and a reader of the call site sees it.

Test support beside its owner (the Data update suite); nothing here is product
authority, and the default arguments are exactly the values the module state
held, so every caller that did not touch that state is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from hashlib import sha256
from pathlib import Path

from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
)
from alphalattice.foundation.market_data_ops.sources.providers import (
    ProviderFetchError,
    SectorObservation,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from tests.researcher_methodology_surface.real_workspace import (
    HISTORY_START,
    SYMBOLS,
    SeededWalkProvider,
    _bootstrap,
)

NOW = datetime(2026, 8, 3, 23, tzinfo=UTC)
"""The as-of the Data update suite plans against: sessions through 2026-08-03."""

PRODUCT_QA_SYMBOLS = tuple(f"QA{index:03d}" for index in range(80))
"""The eighty-name universe of the synthetic Data/Feature product QA workspace."""


class RecordingProvider(SeededWalkProvider):
    """The seeded walk, recording every fetch and able to fail on demand."""

    def __init__(
        self, symbols: Sequence[str], sessions: Sequence[date], *, sector_size: int = 5
    ) -> None:
        super().__init__(symbols, sessions, sector_size=sector_size)
        self._session_ordinals: dict[date, int] = {}
        for index, session in enumerate(self.sessions):
            self._session_ordinals.setdefault(session, index)
        self.calls: list[tuple[tuple[str, ...], date, date]] = []
        self.unavailable = False
        self.unavailable_code = "data.fixture_unavailable"

    def fetch_daily(
        self, symbols: Sequence[str], *, start: date, end: date
    ) -> Mapping[str, list[dict[str, object]]]:
        self.calls.append((tuple(symbols), start, end))
        if self.unavailable:
            raise ProviderFetchError(
                self.unavailable_code, "Offline source not ready", retryable=True
            )
        values = super().fetch_daily(symbols, start=start, end=end)
        for symbol, rows in values.items():
            seed = self._seed(symbol)
            for row in rows:
                ordinal = self._session_ordinals[date.fromisoformat(row["session_date"])]
                row["volume"] = 1_000_000 + seed + ((ordinal * (seed % 97 + 3)) % 100_000)
        return values


def recording_provider(
    *, now: datetime = NOW, symbols: Sequence[str] = SYMBOLS, sector_size: int = 5
) -> RecordingProvider:
    """A provider serving ``symbols`` on the XNAS calendar through ``now``."""

    schedule = materialize_calendar_schedule(
        ("XNAS",), start=HISTORY_START, end=now.date(), as_of_timestamp=now
    )
    return RecordingProvider(
        tuple(symbols),
        tuple(row["session_date"] for row in schedule.to_pylist()),
        sector_size=sector_size,
    )


class HeldDataProvider:
    """A copy's own data served back as its provider would, and quiet sessions after it.

    For the sessions the copy holds: each held member's stored bars, actions and adjusted
    closes, so an update's overlap refetch restates nothing. For each later session through
    ``through``: a bar a few basis points from the last, its adjusted close moving with it, so
    an update past the copy's data publishes offline (V604). Read before the Host opens the
    copy; it records every fetch and, like ``RecordingProvider``, refuses them all while
    ``unavailable``.
    """

    name = "yfinance"

    def __init__(self, workspace: Path, *, through: date, window_days: int = 70) -> None:
        market = MarketDataRepository(workspace)
        # The admitted source root, every held member beside the research subset, as V599's held
        # Friday serves it: a candidate's raw retry reads its own held history.
        manifest = market.source_admission_manifest(market_profile_id="us-current-index-research")
        covered = None if manifest is None else market.manifest_raw_range(manifest)
        if manifest is None or covered is None:
            raise AssertionError("the copy holds no research manifest with bars")
        held = covered[1]
        listings = {listing.provider_symbol: listing.listing_id for listing in manifest.listings}
        features = FeatureStateRepository(workspace)
        reference = features.market_reference("SPY")
        if reference is not None:
            listings[str(reference["symbol"])] = str(reference["listing_id"])
        self.listings = listings
        # Each member's held current Sector, as V599's held Friday serves it; a member the copy
        # holds none for takes a populated held group's, a synthetic observation, not a claim
        # about its real Sector.
        self.sectors = {
            str(row["provider_symbol"]): SectorObservation(
                provider=str(row["provider"]),
                provider_symbol=str(row["provider_symbol"]),
                sector_name=str(row["sector_name"]),
                sector_key=None if row["sector_key"] is None else str(row["sector_key"]),
                payload_hash=str(row["payload_hash"]),
            )
            for row in features.reusable_current_sector_observations(
                manifest, observed_at=datetime.combine(held, time(23, 5), tzinfo=UTC)
            )
        }
        if self.sectors:
            example = next(iter(self.sectors.values()))
            for listing in manifest.listings:
                self.sectors.setdefault(
                    listing.provider_symbol,
                    SectorObservation(
                        provider=self.name,
                        provider_symbol=listing.provider_symbol,
                        sector_name=example.sector_name,
                        sector_key=example.sector_key,
                        payload_hash=sha256(
                            ("fixture-sector:" + listing.listing_id).encode()
                        ).hexdigest(),
                    ),
                )
        # Each listing's bars as (session, open, high, low, close, volume), its actions and its
        # adjusted closes, from the window an update's overlap refetch reads.
        self.bars: dict[str, list[tuple[date, float, float, float, float, float]]] = {}
        self.actions: dict[str, tuple[CorporateActionEvent, ...]] = {}
        self.adjusted: dict[str, dict[date, float]] = {}
        start = held - timedelta(days=window_days)
        with market.database.retain(read_only=True):
            for listing_id in listings.values():
                self.actions[listing_id] = market.actions(listing_id)
                self.bars[listing_id] = [
                    (bar.session_date, bar.open, bar.high, bar.low, bar.close, bar.volume)
                    for bar in market.raw_bars(listing_id, start=start)
                ]
                self.adjusted[listing_id] = {
                    point.session_date: point.adjusted_close
                    for point in market.provider_adjusted_closes(
                        listing_id, through=held, start=start
                    )
                }
        schedule = materialize_calendar_schedule(
            ("XNAS",), start=held, end=through, as_of_timestamp=datetime.now(UTC)
        )
        later = [
            session
            for session in (row["session_date"] for row in schedule.to_pylist())
            if held < session <= through
        ]
        for listing_id, rows in self.bars.items():
            seed = sum(ord(item) for item in listing_id)
            for session in later:
                if not rows or rows[-1][0] < held:
                    # A member the copy holds short of its last session gets no invented bars.
                    break
                last_session, *_, last_close, volume = rows[-1]
                move = ((seed + session.toordinal()) % 7 - 3) / 1000.0
                close = last_close * (1.0 + move)
                high, low = max(last_close, close) * 1.004, min(last_close, close) * 0.997
                rows.append((session, last_close, high, low, close, volume))
                points = self.adjusted[listing_id]
                if last_session in points:
                    points[session] = points[last_session] * (1.0 + move)
        self.calls: list[tuple[str, str, date, date]] = []
        self.unavailable = False
        self.unavailable_code = "data.fixture_unavailable"

    def _asked(self, call: str, symbol: str, start: date, end: date) -> None:
        self.calls.append((call, symbol, start, end))
        if self.unavailable:
            raise ProviderFetchError(
                self.unavailable_code, "Offline source not ready", retryable=True
            )

    def _amount(self, listing_id: str, session: date, kind: str) -> float:
        for action in self.actions[listing_id]:
            if action.effective_date == session and action.action_kind == kind:
                value = action.new_shares_per_old_share if kind == "SPLIT" else action.cash_amount
                return float(value or 0.0)
        return 0.0

    def fetch_daily(
        self, symbols: Sequence[str], *, start: date, end: date
    ) -> Mapping[str, list[dict[str, object]]]:
        result: dict[str, list[dict[str, object]]] = {}
        for symbol in symbols:
            self._asked("daily", symbol, start, end)
            listing_id = self.listings.get(symbol)
            if listing_id is None or not self.bars[listing_id]:
                # A candidate the copy holds no history for: its retry records it unavailable,
                # as V599's held Friday does; no data is invented for it.
                raise ProviderFetchError(
                    "data.empty_payload", "No fixture history", retryable=False
                )
            result[symbol] = [
                {
                    "session_date": session.isoformat(),
                    "open": opening,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "split_ratio": self._amount(listing_id, session, "SPLIT"),
                    "cash_dividend": self._amount(listing_id, session, "CASH_DIVIDEND"),
                    "capital_gain": self._amount(listing_id, session, "CAPITAL_GAIN"),
                }
                for session, opening, high, low, close, volume in self.bars[listing_id]
                if start <= session <= end
            ]
        return result

    def fetch_action_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ) -> tuple[CorporateActionEvent, ...]:
        self._asked("actions", provider_symbol, start, end)
        return tuple(
            action
            for action in self.actions.get(listing_id, ())
            if start <= action.effective_date <= end
        )

    def fetch_adjusted_close_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ) -> tuple[ProviderAdjustedClosePoint, ...]:
        self._asked("adjusted", provider_symbol, start, end)
        return tuple(
            ProviderAdjustedClosePoint(
                listing_id=listing_id,
                provider=self.name,
                session_date=session,
                adjusted_close=value,
            )
            for session, value in sorted(self.adjusted.get(listing_id, {}).items())
            if start <= session <= end
        )

    def fetch_current_sector(self, *, provider_symbol: str) -> SectorObservation:
        self._asked("sector", provider_symbol, date.min, date.min)
        if provider_symbol not in self.sectors:
            raise ProviderFetchError(
                "sector.missing_current_sector", "No fixture sector", retryable=False
            )
        return self.sectors[provider_symbol]


def unchanged_membership_source(symbols: Sequence[str] = SYMBOLS) -> Callable[..., object]:
    """A membership source that reports the workspace's own names, unchanged.

    The product re-observes the candidate sources once per requested trading
    session, so every update session needs a source; a suite that does not
    test a membership change observes the same names it already holds.
    """

    def loader(**_kwargs: object) -> object:
        return _bootstrap(tuple(symbols))

    return loader


__all__ = [
    "NOW",
    "PRODUCT_QA_SYMBOLS",
    "HeldDataProvider",
    "RecordingProvider",
    "recording_provider",
    "unchanged_membership_source",
]
