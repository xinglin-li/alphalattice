"""The development universe: seventeen listings over a two-year standard, each case the full data
holds planted by name, served offline as the data provider and the universe source.

Development and verification run here; the full universe runs only in the nightly. Every case
is a `Listing.cases` name or a dated change below, so a test asserting one reads its name.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from typing import Literal

import numpy as np

from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
)
from alphalattice.foundation.market_data_ops.sources.providers import (
    ProviderFetchError,
    SectorObservation,
)
from alphalattice.foundation.market_data_ops.sources.universe import (
    ORIGINAL_RESEARCH_WHITELIST_STANDARD,
    CandidateMembershipEvidence,
    CurrentUniverseBootstrap,
    CurrentUniverseCandidate,
    CurrentUniverseCandidateManifest,
    candidate_manifest_document,
)
from alphalattice.kernel.data.calendar import materialize_calendar_schedule

AS_OF = date(2026, 7, 31)
OBSERVED_AT = datetime(2026, 8, 2, tzinfo=UTC)
THROUGH = date(2026, 9, 4)
"""The last session served, for the updates after the first use."""
CHANGED_AT = datetime(2026, 8, 12, tzinfo=UTC)
"""From here the source and the provider answer with the later changes: the ticker, venue and
Sector changes, the correction and the delisting's departure."""
_SCALE = {"minimum_history_calendar_years": 2, "minimum_factor_listings": 10}
"""The development standard: two years of history, and a Factor floor of ten listings, which
the eleven past features clear, so a first use goes on through the Factor experiment."""
STANDARD = replace(
    ORIGINAL_RESEARCH_WHITELIST_STANDARD,
    **{
        k: v
        for k, v in _SCALE.items()
        if k in ORIGINAL_RESEARCH_WHITELIST_STANDARD.__dataclass_fields__
    },
)
REFERENCE = "SPY"
ROW_CAP = 10_000
"""At most this many rows in each family this provider writes; the bars, the widest, are checked at
each build. A family the first use derives from them is measured and named in the set's
registration, never cut by dropping a named case: panel_cross_section_availability, one row per
listing, session and availability group, holds 26,606 here."""


def _offsets(seed: int, ordinal: int) -> tuple[float, float, float]:
    """A session's open, high and low offsets from its close, asymmetric so no money-flow or
    range factor is constant across the universe."""
    rng = np.random.default_rng((seed, ordinal))
    above, below = float(rng.uniform(0.002, 0.020)), float(rng.uniform(0.002, 0.020))
    return float(rng.uniform(-below * 0.8, above * 0.8)), above, below


def _ordered(start: date, end: date, kind: str) -> None:
    """A reversed window is the caller's error, refused as the provider adapter refuses it."""
    if start > end:
        raise ValueError(f"{kind} provider range start is after end")


@dataclass(frozen=True)
class Listing:
    """One candidate: its venue's index, its Sector and the cases it carries."""

    symbol: str
    index: str
    sector: str
    cases: tuple[str, ...] = ()
    provider_symbol: str | None = None
    price: float = 100.0


LISTINGS = (
    Listing("DVA", "NASDAQ100", "Technology", ("ticker-change",)),
    Listing("DVB", "NASDAQ100", "Technology", ("split", "fractional-split")),
    Listing("DVC", "NASDAQ100", "Technology", ("dividends", "venue-change")),
    Listing("DVD", "SP500", "Technology", ("zero-volume",)),
    Listing("DVE", "SP500", "Technology", ("four-digit-price",), price=2400.0),
    Listing("DVF", "NASDAQ100", "Technology", ("sector-change",)),
    Listing("DVG", "SP500", "Healthcare", ("reverse-split",)),
    Listing("DVH", "SP500", "Healthcare", ("halt",)),
    Listing("DVI", "NASDAQ100", "Healthcare", ("missing-days",)),
    Listing("DVJ", "SP500", "Healthcare", ("flat-bars",)),
    Listing("DVK", "NASDAQ100", "Healthcare", ("correction-after-cutoff",)),
    Listing(
        "DVL.B",
        "SP500",
        "Healthcare",
        ("provider-symbol", "tiny-and-large-dividends"),
        provider_symbol="DVL-B",
        price=150.0,
    ),
    Listing("DVN", "SP500", "Healthcare", ("sub-dollar-price",), price=0.8),
    Listing("DVM", "SP500", "Industrials", ("same-day-split-and-dividend",)),
    Listing("DVQ", "SP500", "Industrials", ("delisting",)),
    Listing("DVR", "NASDAQ100", "Industrials", ("mid-period-start", "adjusted-closes-missing")),
    Listing("DVS", "SP500", "Industrials", ("long-halt",)),
)
"""Thirteen eligible members in two Sectors (Technology six, so the Sector change leaves both
at five or more), and four excluded by name, each as the full data excludes its kind: a split
and a dividend on one day (its action audit fails), a delisting, a mid-period start whose
first adjusted closes are missing, and a halt longer than the standard's longest gap."""
MOVED = ("DVE", "DVJ", "DVK")
"""The eligible listings a scenario's unexplained moves take, in this order."""
RENAMED = "DVW"
"""DVA's symbol after its ticker change; the provider serves its history under both."""
SECTOR_CHANGE = ("DVF", "Healthcare")


def calendar_sessions(start: date | None = None, end: date = THROUGH) -> tuple[date, ...]:
    """The common US sessions from the standard's floor before `AS_OF` through `end`."""
    first = start or AS_OF.replace(year=AS_OF.year - STANDARD.minimum_history_calendar_years)
    schedule = materialize_calendar_schedule(
        ("XNYS",),
        start=first,
        end=end,
        as_of_timestamp=datetime(end.year, end.month, end.day, 23, tzinfo=UTC),
    )
    return tuple(row["session_date"] for row in schedule.to_pylist())


_FLOOR = {
    k: v for k, v in _SCALE.items() if k in CurrentUniverseCandidateManifest.__dataclass_fields__
}
"""The candidate set names its floors where the running product reads them; a release before
that (0.2.7) keeps the original field set; its floor comes from the QA control that builds it."""


def bootstrap(observed_at: datetime = OBSERVED_AT) -> CurrentUniverseBootstrap:
    """The candidate set the universe source answers at `observed_at`, under the two-year standard.

    From `CHANGED_AT` the delisting has left, DVA is listed as DVW, and DVC is an SP500 member
    only, so its venue is XNYS."""
    later = observed_at >= CHANGED_AT
    candidates = []
    for item in LISTINGS:
        if later and "delisting" in item.cases:
            continue
        symbol = RENAMED if later and "ticker-change" in item.cases else item.symbol
        index = "SP500" if later and "venue-change" in item.cases else item.index
        candidates.append(
            CurrentUniverseCandidate(
                symbol=symbol,
                provider_symbol=item.provider_symbol or symbol,
                source_memberships=(CandidateMembershipEvidence(index, symbol),),
            )
        )
    provisional = CurrentUniverseCandidateManifest(
        created_at=observed_at,
        as_of_timestamp=observed_at,
        construction_rule=STANDARD.construction_rule,
        data_validity_class=STANDARD.data_validity_class,
        sources=(),
        candidates=tuple(sorted(candidates, key=lambda item: item.symbol)),
        content_hash="",
        **_FLOOR,
    )
    document = candidate_manifest_document(provisional)
    document.pop("content_hash")
    sealed = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return CurrentUniverseBootstrap(
        source_manifest=replace(
            provisional, content_hash=hashlib.sha256(sealed.encode()).hexdigest()
        ),
        standard=STANDARD,
    )


def source_loader(**kwargs: object) -> CurrentUniverseBootstrap:
    """The universe source: the candidate set as observed at the Host's `observed_at`."""
    observed_at = kwargs.get("observed_at")
    return bootstrap(observed_at if isinstance(observed_at, datetime) else OBSERVED_AT)


@dataclass(frozen=True)
class _Bar:
    session: date
    open: float
    high: float
    low: float
    close: float
    volume: float


class DevelopmentProvider:
    """The development universe's provider, deterministic and offline.

    `clock` is the Host's: from `CHANGED_AT` it serves the later Sector and the corrected bar.
    `moved` names listings whose close doubles with no event, the unexplained raw move a first
    use stops on for review: on `moved_sessions` sessions a week apart from the middle one.
    `sessions` are the US calendar's (`development_provider`), or a bound on them where no
    calendar is read. It refuses every fetch while `unavailable`.
    """

    name = "yfinance"

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: OBSERVED_AT,
        moved: Sequence[str] = (),
        moved_sessions: int = 1,
        sessions: Sequence[date],
    ) -> None:
        self.clock = clock
        self.moved = tuple(moved)
        self.moved_at = {len(sessions) // 2 + 5 * n for n in range(moved_sessions)}
        self.sessions = tuple(sessions)
        self.by_symbol = {item.provider_symbol or item.symbol: item for item in LISTINGS}
        self.by_symbol[RENAMED] = self.by_symbol["DVA"]
        self.calls: list[tuple[str, str, date, date]] = []
        self.unavailable = False
        self.unavailable_code = "data.fixture_unavailable"
        self._bars = {REFERENCE: self._walk(REFERENCE, Listing(REFERENCE, "SP500", "Reference"))}
        for item in LISTINGS:
            self._bars[item.symbol] = self._walk(item.symbol, item)
        if (served := sum(len(bars) for bars in self._bars.values())) > ROW_CAP:
            raise ValueError(f"development_universe.row_cap: {served} bars > {ROW_CAP}")

    @staticmethod
    def _seed(symbol: str) -> int:
        return sum(ord(letter) * (index + 1) for index, letter in enumerate(symbol))

    def _walk(self, symbol: str, item: Listing) -> list[_Bar]:
        seed = self._seed(symbol)
        rng = np.random.default_rng(seed)
        closes = item.price * np.cumprod(1.0 + rng.normal(0.0003, 0.012, len(self.sessions)))
        bars = []
        for ordinal, (session, close) in enumerate(zip(self.sessions, closes, strict=True)):
            opening, above, below = _offsets(seed, session.toordinal())
            volume = 1_000_000.0 + seed + (ordinal * (seed % 97 + 3)) % 100_000
            bar = _Bar(
                session,
                close * (1 + opening),
                close * (1 + above),
                close * (1 - below),
                close,
                volume,
            )
            if "zero-volume" in item.cases and ordinal % 97 == 13:
                bar = replace(bar, volume=0.0)
            if "flat-bars" in item.cases and ordinal % 89 == 7:
                bar = replace(bar, open=close, high=close, low=close)
            if symbol in self.moved and ordinal in self.moved_at:
                bar = replace(bar, open=close * 2, high=close * 2, low=close * 2, close=close * 2)
            bars.append(bar)
        return [bar for index, bar in enumerate(bars) if self._served(item, index)]

    def _served(self, item: Listing, index: int) -> bool:
        """Whether the provider holds the listing's bar at session `index`."""
        last = self.sessions.index(AS_OF) if AS_OF in self.sessions else len(self.sessions) - 1
        return not (
            ("delisting" in item.cases and index > last - 126)
            or ("mid-period-start" in item.cases and index < 400)
            or ("long-halt" in item.cases and 300 <= index < 325)
            or ("halt" in item.cases and 200 <= index < 205)
            or ("missing-days" in item.cases and index in (111, 333))
        )

    def actions(self, item: Listing, listing_id: str) -> tuple[CorporateActionEvent, ...]:
        """The listing's provider-observed actions, each kind its cases name."""
        when = self.sessions.__getitem__
        events: list[tuple[date, Literal["SPLIT", "CASH_DIVIDEND"], float]] = []
        if "split" in item.cases:
            events.append((when(240), "SPLIT", 2.0))
        if "reverse-split" in item.cases:
            events.append((when(260), "SPLIT", 0.125))
        if "fractional-split" in item.cases:
            events.append((when(280), "SPLIT", 1.067))
        if "dividends" in item.cases:
            events.extend((when(index), "CASH_DIVIDEND", 0.24) for index in range(30, 500, 63))
        if "tiny-and-large-dividends" in item.cases:
            events.extend([(when(150), "CASH_DIVIDEND", 0.003), (when(350), "CASH_DIVIDEND", 25.0)])
        if "same-day-split-and-dividend" in item.cases:
            events.extend([(when(300), "SPLIT", 3.0), (when(300), "CASH_DIVIDEND", 0.5)])
        return tuple(
            CorporateActionEvent(
                listing_id=listing_id,
                provider=self.name,
                effective_date=effective,
                action_kind=kind,
                new_shares_per_old_share=value if kind == "SPLIT" else None,
                cash_amount=value if kind == "CASH_DIVIDEND" else None,
            )
            for effective, kind, value in events
        )

    def _item(self, symbol: str) -> Listing:
        if symbol == REFERENCE:
            return Listing(REFERENCE, "SP500", "Reference")
        if symbol not in self.by_symbol:
            raise ProviderFetchError("data.empty_payload", "No fixture history", retryable=False)
        return self.by_symbol[symbol]

    def _bars_of(self, symbol: str) -> list[_Bar]:
        item = self._item(symbol)
        bars = self._bars[item.symbol]
        if "correction-after-cutoff" in item.cases and self.clock() >= CHANGED_AT:
            index = len(bars) - 40
            bars = [
                *bars[:index],
                replace(bars[index], close=bars[index].close * 1.01),
                *bars[index + 1 :],
            ]
        return bars

    def _asked(self, call: str, symbol: str, start: date, end: date) -> None:
        if call != "sector":
            _ordered(start, end, call)
        self.calls.append((call, symbol, start, end))
        if self.unavailable:
            raise ProviderFetchError(
                self.unavailable_code, "Offline source not ready", retryable=True
            )

    def fetch_daily(
        self, symbols: Sequence[str], *, start: date, end: date
    ) -> Mapping[str, list[dict[str, object]]]:
        result = {}
        for symbol in symbols:
            self._asked("daily", symbol, start, end)
            amounts: dict[tuple[date, str], float] = {}
            for event in self.actions(self._item(symbol), symbol):
                value = event.new_shares_per_old_share or event.cash_amount or 0.0
                amounts[(event.effective_date, event.action_kind)] = value
            result[symbol] = [
                {
                    "session_date": bar.session.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "split_ratio": amounts.get((bar.session, "SPLIT"), 0.0),
                    "cash_dividend": amounts.get((bar.session, "CASH_DIVIDEND"), 0.0),
                    "capital_gain": 0.0,
                }
                for bar in self._bars_of(symbol)
                if start <= bar.session <= end
            ]
        return result

    def fetch_action_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ) -> tuple[CorporateActionEvent, ...]:
        self._asked("action", provider_symbol, start, end)
        return tuple(
            event
            for event in self.actions(self._item(provider_symbol), listing_id)
            if start <= event.effective_date <= end
        )

    def fetch_adjusted_close_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ) -> tuple[ProviderAdjustedClosePoint, ...]:
        """Closes back-adjusted for each cash dividend, as the provider adjusts them; a listing
        whose case says so has none for its first twenty sessions."""
        self._asked("adjusted-close", provider_symbol, start, end)
        item = self._item(provider_symbol)
        bars = self._bars_of(provider_symbol)
        factor, adjusted = 1.0, {}
        dividends = {
            event.effective_date: event.cash_amount or 0.0
            for event in self.actions(item, listing_id)
            if event.action_kind == "CASH_DIVIDEND"
        }
        for previous, bar in zip(reversed(bars[:-1]), reversed(bars[1:]), strict=True):
            adjusted[bar.session] = bar.close * factor
            if bar.session in dividends:
                factor *= 1.0 - dividends[bar.session] / previous.close
        adjusted[bars[0].session] = bars[0].close * factor
        skipped = (
            {bar.session for bar in bars[:20]} if "adjusted-closes-missing" in item.cases else set()
        )
        return tuple(
            ProviderAdjustedClosePoint(
                listing_id=listing_id,
                provider=self.name,
                session_date=session,
                adjusted_close=value,
            )
            for session, value in sorted(adjusted.items())
            if start <= session <= end and session not in skipped
        )

    def fetch_current_sector(self, *, provider_symbol: str) -> SectorObservation:
        self._asked("sector", provider_symbol, date.min, date.min)
        item = self._item(provider_symbol)
        sector = item.sector
        if item.symbol == SECTOR_CHANGE[0] and self.clock() >= CHANGED_AT:
            sector = SECTOR_CHANGE[1]
        return SectorObservation(
            provider=self.name,
            provider_symbol=provider_symbol,
            sector_name=sector,
            sector_key=sector.casefold().replace(" ", "-"),
            payload_hash=hashlib.sha256(f"{provider_symbol}:{sector}".encode()).hexdigest(),
        )


def development_provider(
    *,
    clock: Callable[[], datetime] = lambda: OBSERVED_AT,
    moved: Sequence[str] = (),
    moved_sessions: int = 1,
) -> DevelopmentProvider:
    """The development universe's provider over the US sessions from the floor to `THROUGH`."""
    return DevelopmentProvider(
        clock=clock, moved=moved, moved_sessions=moved_sessions, sessions=calendar_sessions()
    )
