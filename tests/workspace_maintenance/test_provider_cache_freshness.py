"""A provider correction must survive the historical HTTP cache on the next daily."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

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
from tests.workspace_maintenance.local_data_provider import (
    HistoricalChartReplay,
    recording_provider,
)

START = date(2025, 10, 1)
PAID = date(2026, 1, 15)
FIRST = date(2026, 8, 3)
NEXT = date(2026, 8, 4)


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
            replay = HistoricalChartReplay(source, start=START, through=FIRST, paid=PAID)
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
