"""The market-data provider ports' conformance suites: each implementation keeps the port's shape.

One suite per port, in the form of dbt's adapter tests: each test below runs once per registered
implementation, the yfinance adapter (real yfinance parsing replayed chart bytes, offline) and the
shared fakes other suites stand on, so a fake cannot drift from what the adapter guarantees. A new
provider adds one row to its port's registry and inherits the whole suite.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from yfinance.data import YfData

from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
)
from alphalattice.foundation.market_data_ops.sources.price_integrity import (
    validate_split_adjusted_history,
)
from alphalattice.foundation.market_data_ops.sources.providers import (
    HydrationEvidence,
    ProviderFetchError,
    SectorObservation,
    YFinanceMarketDataProvider,
)
from tests.feature_engine.panel_workspace import FixtureProvider
from tests.researcher_methodology_surface.real_workspace import SeededWalkProvider
from tests.workspace_maintenance.bounded_provider_support import RecordingBoundedProvider
from tests.workspace_maintenance.local_data_provider import (
    HistoricalChartReplay,
    recording_provider,
)

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/sources/providers.py"

FIELDS = (
    "session_date",
    *("open", "high", "low", "close", "volume"),
    *("split_ratio", "cash_dividend", "capital_gain"),
)
"""The daily row the port promises: the adapter's own row, field for field."""
HEX64 = re.compile(r"[0-9a-f]{64}")
REPLAY_START, REPLAY_THROUGH, REPLAY_PAID = date(2025, 10, 1), date(2026, 8, 3), date(2026, 1, 15)
SESSIONS = tuple(pd.bdate_range("2026-01-02", "2026-02-27").date)


@dataclass(frozen=True)
class Scenario:
    """What an implementation can answer: its symbols, one listing, and a window it holds."""

    symbols: tuple[str, ...]
    listing_id: str
    start: date
    end: date


def _replayed_yahoo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real yfinance, its HTTP answered from synthetic chart bytes: no network, a paid dividend."""
    source = recording_provider(
        now=datetime(2026, 8, 4, 23, tzinfo=UTC), symbols=("AAPL",), start=REPLAY_START
    )
    replay = HistoricalChartReplay(
        source, start=REPLAY_START, through=REPLAY_THROUGH, paid=REPLAY_PAID
    )
    monkeypatch.setattr(YfData, "get", replay.get)
    YfData().cache_get.cache_clear()


def _yahoo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Scenario]:
    _replayed_yahoo(tmp_path, monkeypatch)
    window = Scenario(("AAPL",), "listing-aapl", date(2026, 1, 5), date(2026, 1, 30))
    return YFinanceMarketDataProvider(tmp_path / "yahoo-cache"), window


def _bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Scenario]:
    _replayed_yahoo(tmp_path, monkeypatch)
    provider = RecordingBoundedProvider(
        tmp_path / "yahoo-cache", earliest_allowed=REPLAY_START, target=REPLAY_THROUGH
    )
    return provider, Scenario(("AAPL",), "listing-aapl", date(2026, 1, 5), date(2026, 1, 30))


def _seeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Scenario]:
    del tmp_path, monkeypatch
    window = Scenario(("AAA", "BBB"), "listing-aaa", date(2026, 1, 12), date(2026, 2, 6))
    return SeededWalkProvider(("AAA", "BBB"), SESSIONS), window


def _recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Scenario]:
    del tmp_path, monkeypatch
    provider = recording_provider(
        now=datetime(2026, 2, 27, 23, tzinfo=UTC), symbols=("AAA", "BBB"), start=SESSIONS[0]
    )
    return provider, Scenario(("AAA", "BBB"), "listing-aaa", date(2026, 1, 12), date(2026, 2, 6))


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Scenario]:
    del tmp_path, monkeypatch
    window = Scenario(("AAA", "BBB"), "listing-aaa", date(2018, 3, 5), date(2018, 3, 30))
    return FixtureProvider(("AAA", "BBB")), window


def _yahoo_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Scenario]:
    """The adapter's normalisation of Yahoo's info object, its runtime module stood in for."""
    del monkeypatch
    provider = YFinanceMarketDataProvider(tmp_path / "yahoo-cache")
    info = {"AAPL": {"sector": " Technology ", "sectorKey": "technology"}}
    provider._runtime_module = SimpleNamespace(
        Ticker=lambda symbol: SimpleNamespace(get_info=lambda: dict(info.get(symbol, {})))
    )
    return provider, Scenario(("AAPL",), "listing-aapl", date(2026, 1, 5), date(2026, 1, 30))


Factory = Callable[[Path, pytest.MonkeyPatch], tuple[Any, Scenario]]
DAILY: dict[str, Factory] = {
    "yfinance": _yahoo,
    "bounded-yfinance": _bounded,
    "seeded-walk": _seeded,
    "recording": _recording,
    "feature-fixture": _fixture,
}
"""The daily-bar port (``MarketDataProvider``): the adapter and every shared fake. The held-copy
fake (``HeldDataProvider``) reads a whole workspace's database, so it is held by its own suites."""
HYDRATION: dict[str, Factory] = {"yfinance": _yahoo, "bounded-yfinance": _bounded}
"""The one-response history port (``HistoricalHydrationProvider``)."""
SECTOR: dict[str, Factory] = {
    "yfinance": _yahoo_info,
    "seeded-walk": _seeded,
    "recording": _recording,
    "feature-fixture": _fixture,
}
"""The current-sector port (``SectorReferenceProvider``)."""


def _number(value: object) -> bool:
    return value is None or (
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
    )


def _rows(rows: Sequence[Mapping[str, object]], window: Scenario, symbol: str) -> list[date]:
    """The port's rows: every field, finite numbers, inside the window, ascending, split-sound."""
    assert isinstance(rows, Sequence) and rows, f"{symbol}: no rows for a window it holds"
    sessions = []
    for row in rows:
        assert set(FIELDS) <= set(row), f"{symbol}: a row lacks {set(FIELDS) - set(row)}"
        sessions.append(date.fromisoformat(str(row["session_date"])))
        assert all(_number(row[name]) for name in FIELDS[1:]), f"{symbol}: {row}"
    assert sessions == sorted(set(sessions)), f"{symbol}: sessions repeat or run backwards"
    assert window.start <= sessions[0] and sessions[-1] <= window.end, f"{symbol}: outside"
    validate_split_adjusted_history(rows, symbol=symbol)  # the adapter's own integrity check
    return sessions


def _actions(events: Sequence[object], provider: Any, window: Scenario) -> None:
    for event in events:
        assert isinstance(event, CorporateActionEvent)
        assert (event.listing_id, event.provider) == (window.listing_id, provider.name)
        assert window.start <= event.effective_date <= window.end
        split = event.action_kind == "SPLIT"
        amount = event.new_shares_per_old_share if split else event.cash_amount
        assert event.action_kind in {"SPLIT", "CASH_DIVIDEND", "CAPITAL_GAIN"} and amount
        assert _number(amount) and amount > 0, event


def _closes(points: Sequence[object], provider: Any, window: Scenario) -> list[date]:
    sessions = []
    for point in points:
        assert isinstance(point, ProviderAdjustedClosePoint)
        assert (point.listing_id, point.provider) == (window.listing_id, provider.name)
        assert _number(point.adjusted_close) and point.adjusted_close > 0, point
        sessions.append(point.session_date)
    assert sessions == sorted(set(sessions))
    assert not sessions or (window.start <= sessions[0] and sessions[-1] <= window.end)
    return sessions


@pytest.mark.proves("market_data_ops.daily_provider_conformance")
@pytest.mark.parametrize("implementation", DAILY)
def test_every_daily_provider_answers_in_the_ports_shape(tmp_path, monkeypatch, implementation):
    """Daily bars, actions and adjusted closes keep the port's shape, whichever provider answers."""
    provider, window = DAILY[implementation](tmp_path, monkeypatch)
    assert isinstance(provider.name, str) and provider.name
    answer = provider.fetch_daily(window.symbols, start=window.start, end=window.end)
    assert set(answer) == set(window.symbols)
    sessions = {symbol: _rows(rows, window, symbol) for symbol, rows in answer.items()}
    first = window.symbols[0]
    request = dict(
        listing_id=window.listing_id, provider_symbol=first, start=window.start, end=window.end
    )
    _actions(provider.fetch_action_history(**request), provider, window)
    closes = _closes(provider.fetch_adjusted_close_history(**request), provider, window)
    assert set(closes) <= set(sessions[first]), "an adjusted close for a session with no bar"
    backwards = dict(start=window.end, end=window.start)  # the adapter's ValueError is the port's
    with pytest.raises(ValueError, match="start is after end"):
        provider.fetch_daily(window.symbols, **backwards)
    for read in (provider.fetch_action_history, provider.fetch_adjusted_close_history):
        with pytest.raises(ValueError, match="start is after end"):
            read(listing_id=window.listing_id, provider_symbol=first, **backwards)


@pytest.mark.proves("market_data_ops.hydration_provider_conformance")
@pytest.mark.parametrize("implementation", HYDRATION)
def test_every_hydration_provider_answers_one_consistent_history(
    tmp_path, monkeypatch, implementation
):
    """One hydration response holds the daily port's own rows, its actions and its closes."""
    provider, window = HYDRATION[implementation](tmp_path, monkeypatch)
    symbol = window.symbols[0]
    evidence = provider.fetch_hydration(
        listing_id=window.listing_id, provider_symbol=symbol, start=window.start, end=window.end
    )
    assert isinstance(evidence, HydrationEvidence)
    sessions = _rows(evidence.daily_rows, window, symbol)
    daily = provider.fetch_daily((symbol,), start=window.start, end=window.end)[symbol]
    assert list(evidence.daily_rows) == list(daily), "hydration and daily disagree on the rows"
    _actions(evidence.actions, provider, window)
    assert set(_closes(evidence.adjusted_closes, provider, window)) <= set(sessions)
    assert set(evidence.repaired_sessions) <= set(sessions)
    hashed = evidence.provider_policy_hash
    assert hashed is None or HEX64.fullmatch(hashed)
    with pytest.raises(ValueError, match="start is after end"):
        provider.fetch_hydration(
            listing_id=window.listing_id, provider_symbol=symbol, start=window.end, end=window.start
        )


@pytest.mark.proves("market_data_ops.sector_provider_conformance")
@pytest.mark.parametrize("implementation", SECTOR)
def test_every_sector_provider_answers_one_named_current_sector(
    tmp_path, monkeypatch, implementation
):
    """A current sector names its provider and symbol, and an unknown symbol is a typed failure."""
    provider, window = SECTOR[implementation](tmp_path, monkeypatch)
    symbol = window.symbols[0]
    observed = provider.fetch_current_sector(provider_symbol=symbol)
    assert isinstance(observed, SectorObservation)
    assert (observed.provider, observed.provider_symbol) == (provider.name, symbol)
    assert observed.sector_name and observed.sector_name == observed.sector_name.strip()
    assert observed.sector_key is None or isinstance(observed.sector_key, str)
    assert HEX64.fullmatch(observed.payload_hash)
    assert provider.fetch_current_sector(provider_symbol=symbol) == observed
    with pytest.raises(ProviderFetchError) as refused:
        provider.fetch_current_sector(provider_symbol="NO-SUCH-SYMBOL")
    assert refused.value.code and isinstance(refused.value.retryable, bool)
