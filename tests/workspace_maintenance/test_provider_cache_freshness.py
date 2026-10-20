"""A provider correction must survive the historical HTTP cache on the next daily."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from yfinance.data import YfData

from alphalattice.foundation.market_data_ops.runtime.refresh import normal_refresh_plan
from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
    CurrentUniverseMaintenance,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    build_quality_filtered_research_manifest,
)
from alphalattice.foundation.market_data_ops.sources.providers import YFinanceMarketDataProvider
from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from tests.workspace_maintenance.acquisition_manifest import acquisition_manifest
from tests.workspace_maintenance.audit_workspace import audit_one
from tests.workspace_maintenance.local_data_provider import RecordingProvider, recording_provider

START = date(2025, 10, 1)
PAID = date(2026, 1, 15)
FIRST = date(2026, 8, 3)
NEXT = date(2026, 8, 4)


class _HistoricalReplay:
    """Replay synthetic chart bytes without replacing the native response cache."""

    def __init__(self, source: RecordingProvider) -> None:
        self.source, self.through, self.cash = source, FIRST, 0.5
        self.paid = PAID
        self.requests: list[dict[str, Any]] = []

    def rows(self, start: date, end: date) -> list[dict[str, Any]]:
        return [
            dict(row, open=100.0, high=101.0, low=99.0, close=100.0)
            for row in self.source.fetch_daily(("AAPL",), start=start, end=end)["AAPL"]
        ]

    def get(self, url: str, params: Mapping[str, Any] | None = None, timeout: int = 30) -> Any:
        assert "/finance/chart/AAPL" in url
        params = dict(params or {})
        assert params.get("interval") == "1d"
        self.requests.append(dict(params=params, cash=self.cash, timeout=timeout))
        start = (
            datetime.fromtimestamp(params["period1"], UTC).date() if "period1" in params else START
        )
        end = (
            datetime.fromtimestamp(params["period2"], UTC).date() - timedelta(days=1)
            if "period2" in params
            else self.through
        )
        rows = self.rows(start, min(end, self.through))
        stamps = [
            int(pd.Timestamp(row["session_date"], tz="America/New_York").timestamp()) + 34200
            for row in rows
        ]
        quotes = {
            name: [row[name] for row in rows] for name in ("open", "high", "low", "close", "volume")
        }
        adjusted = [
            100.0 / (1.0 + self.cash / 100.0)
            if date.fromisoformat(row["session_date"]) < self.paid
            else 100.0
            for row in rows
        ]
        paid = int(pd.Timestamp(self.paid.isoformat(), tz="America/New_York").timestamp()) + 34200
        events = (
            {"dividends": {str(paid): {"amount": self.cash, "date": paid}}}
            if start <= self.paid <= end
            else {}
        )
        periods = [
            [dict(start=stamp, end=stamp + 23400, gmtoffset=-14400, timezone="EDT")]
            for stamp in stamps
        ]
        chart = dict(
            timestamp=stamps,
            events=events,
            meta=dict(
                currency="USD",
                symbol="AAPL",
                instrumentType="EQUITY",
                exchangeName="NMS",
                exchangeTimezoneName="America/New_York",
                priceHint=2,
                validRanges=["1d", "max"],
                regularMarketTime=stamps[-1],
                regularMarketPrice=100.0,
                tradingPeriods=periods,
            ),
            indicators=dict(quote=[quotes], adjclose=[dict(adjclose=adjusted)]),
        )
        payload = {"chart": {"result": [chart], "error": None}}
        return SimpleNamespace(
            url=url, status_code=200, text=json.dumps(payload), json=lambda: deepcopy(payload)
        )


@pytest.mark.parametrize("held", (date(2026, 1, 30), FIRST), ids=("catchup", "daily"))
def test_a_daily_catchup_sees_a_dividend_restatement_behind_a_cached_chunk(
    tmp_path, monkeypatch, record_property, held
) -> None:
    """A catch-up daily sees a restated dividend in a previously fetched history chunk.
    Real yfinance parsing and DuckDB audits can exceed ten seconds.
    """
    source = recording_provider(
        now=datetime(2026, 8, 4, 23, tzinfo=UTC), symbols=("AAPL",), start=START
    )
    data = YfData()
    calls = {}
    identity = {"listing_id": "listing-aapl", "provider_symbol": "AAPL"}
    acquisition = acquisition_manifest()
    manifest = build_quality_filtered_research_manifest(
        replace(acquisition, profile=replace(acquisition.profile, provider="yfinance")),
        eligible_listing_ids=("listing-aapl",),
    )
    window = normal_refresh_plan(held, FIRST).windows[0]
    try:
        for fresh_control in (False, True):
            replay = _HistoricalReplay(source)
            if held == FIRST:
                replay.paid = FIRST - timedelta(days=7)
            monkeypatch.setattr(YfData, "get", replay.get)
            data.cache_get.cache_clear()
            market = MarketDataRepository(tmp_path / f"workspace-{fresh_control}")
            market.bootstrap(manifest)
            provider = YFinanceMarketDataProvider(tmp_path / f"provider-cache-{fresh_control}")
            initial = provider.fetch_hydration(**identity, start=START, end=held)
            market.apply_validated_batch(
                manifest,
                sanitize_payload(manifest, "yfinance", {"AAPL": initial.daily_rows}, ("AAPL",)),
                ingestion_id="initial",
                observed_at=datetime(2026, 8, 3, 23, tzinfo=UTC),
            )
            audit_one(
                market,
                manifest,
                listing_id="listing-aapl",
                provider="yfinance",
                actions=initial.actions,
                adjusted_closes=initial.adjusted_closes,
                history_start=START,
                history_end=held,
                requested_as_of=held,
                observed_at=datetime(2026, 8, 3, 23, tzinfo=UTC),
            )
            warm = provider.fetch_hydration(**identity, start=window.start, end=window.end)
            assert warm.actions[0].cash_amount == 0.5
            assert market.actions("listing-aapl")[0].cash_amount == 0.5
            replay.cash, replay.through = 1.0, NEXT
            if fresh_control:
                data.cache_get.cache_clear()
            daily_start = len(replay.requests)
            runner = CurrentUniverseMaintenance(
                store=market,
                manifest=manifest,
                provider=provider,
                as_of_session=NEXT,
                max_workers=1,
            )
            second = runner.run(observed_at=datetime(2026, 8, 4, 23, tzinfo=UTC))
            calls[fresh_control] = len(replay.requests) - daily_start
            state = market.current_universe_maintenance_listings(runner.maintenance_id)[0]
            assert (second.updated, second.failed, state.failure_code) == (
                0,
                1,
                "data.full_history_audit_approval_required",
            )
            assert market.raw_bars("listing-aapl")[-1].session_date == NEXT
            for method in ("daily", "hydration", "adjusted_close_history", "action_history"):
                fields = {"symbols": ("AAPL",)} if method == "daily" else identity
                fresh = getattr(provider, f"fetch_{method}")(
                    **fields, start=window.start + timedelta(days=1), end=window.end
                )
                info = data.cache_get.cache_info()
                assert (info.currsize, info.hits) == (0, 0)
                if method == "hydration":
                    assert fresh.actions[0].cash_amount == 1.0
            record_property(f"replay_requests:{fresh_control}", json.dumps(replay.requests))
            assert market.actions("listing-aapl")[0].cash_amount == 1.0
        record_property("daily_http_calls", json.dumps(calls))
        if held == FIRST:
            assert calls[False] == calls[True]
    finally:
        data.cache_get.cache_clear()
