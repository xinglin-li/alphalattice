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
from alphalattice.foundation.market_data_ops.sources.providers import (
    ProviderFetchError,
    YFinanceMarketDataProvider,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from tests.workspace_maintenance.acquisition_manifest import acquisition_manifest
from tests.workspace_maintenance.audit_workspace import audit_one
from tests.workspace_maintenance.wire_replay import WireReplay, install

START = date(2025, 10, 1)
FIRST = date(2026, 8, 3)
NEXT = date(2026, 8, 4)
CHART = "/v8/finance/chart/"


@pytest.mark.parametrize("case", ("catchup", "daily"))
def test_a_daily_catchup_sees_a_dividend_restatement_behind_a_cached_chunk(
    tmp_path, monkeypatch, request, record_property, case
) -> None:
    """A catch-up daily sees a restated dividend in a previously fetched history chunk.
    Real yfinance parsing and DuckDB audits can exceed ten seconds.
    """
    held = date(2026, 1, 30) if case == "catchup" else FIRST
    data = YfData()
    calls = {}
    provider_failures = []
    identity = {"listing_id": "listing-aapl", "provider_symbol": "AAPL"}
    acquisition = acquisition_manifest()
    manifest = build_quality_filtered_research_manifest(
        replace(acquisition, profile=replace(acquisition.profile, provider="yfinance")),
        eligible_listing_ids=("listing-aapl",),
    )
    window = normal_refresh_plan(held, FIRST).windows[0]
    try:
        for fresh_control in (False, True):
            replay = WireReplay.load(case, scenario="restatement")
            replay.today = FIRST
            root = tmp_path / f"run-{fresh_control}"
            install(request, replay, root)
            market = MarketDataRepository(root / "workspace")
            market.bootstrap(manifest)
            provider = YFinanceMarketDataProvider(root / "provider-cache")
            read_hydration = provider.fetch_hydration

            def record_hydration(_read=read_hydration, **kwargs):
                try:
                    return _read(**kwargs)
                except ProviderFetchError as exc:
                    cause = exc.__cause__ or exc
                    provider_failures.append({"type": type(cause).__name__, "detail": str(cause)})
                    raise

            monkeypatch.setattr(provider, "fetch_hydration", record_hydration)
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
            replay.today = NEXT  # the provider restates the dividend to 1.0
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
            calls[fresh_control] = sum(
                path.startswith(CHART) for _host, path, _query in replay.requests[daily_start:]
            )
            state = market.current_universe_maintenance_listings(runner.maintenance_id)[0]
            record_property("provider_failures", json.dumps(provider_failures))
            assert (second.updated, second.failed, state.failure_code) == (
                0,
                1,
                "data.full_history_audit_approval_required",
            ), f"provider failures: {provider_failures}"
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
