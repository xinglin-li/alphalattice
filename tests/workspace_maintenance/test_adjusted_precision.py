"""The data rule's precision: a re-sent stored adjusted close within it stands, past it restates."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
    CurrentUniverseMaintenance,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    build_quality_filtered_research_manifest,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from tests.workspace_maintenance.acquisition_manifest import NOW, acquisition_manifest
from tests.workspace_maintenance.audit_workspace import audit_one

HISTORY = (date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30))
NEW = date(2026, 7, 31)


def _close(day: date) -> float:
    return 100.0 + day.day / 7.0


@pytest.mark.parametrize(
    ("scale", "dividend", "stands"),
    [(1 + 5e-7, False, True), (1 + 2e-6, False, False), (1 + 5e-7, True, False)],
)
def test_a_resent_stored_close_within_the_precision_stands_and_past_it_restates(
    tmp_path: Path, scale: float, dividend: bool, stands: bool
) -> None:
    """requirement: within the rule's precision a re-sent stored close stands; past it, or once
    the audit inserts or corrects an action for the listing, the series restates."""
    manifest = build_quality_filtered_research_manifest(
        acquisition_manifest(), eligible_listing_ids=("listing-aapl",)
    )
    store = MarketDataRepository(tmp_path / "workspace")
    store.bootstrap(manifest)
    bar = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000}

    def day(sessions: tuple[date, ...], adjusted: dict[date, float], offset: int, actions=()):  # type: ignore[no-untyped-def]
        store.apply_validated_batch(
            manifest,
            sanitize_payload(
                manifest,
                "fixture",
                {"AAPL": tuple({"session_date": item.isoformat()} | bar for item in sessions)},
                ("AAPL",),
            ),
            ingestion_id=f"day-{offset}",
            observed_at=NOW + timedelta(days=offset),
        )
        return audit_one(
            store,
            manifest,
            listing_id="listing-aapl",
            provider="fixture",
            actions=actions,
            adjusted_closes=tuple(
                ProviderAdjustedClosePoint("listing-aapl", "fixture", item, value)
                for item, value in adjusted.items()
            ),
            history_start=min(adjusted),
            history_end=max(adjusted),
            requested_as_of=max(adjusted),
            observed_at=NOW + timedelta(days=offset),
            adjusted_tolerance=1e-6,
        )

    day(HISTORY, {item: _close(item) for item in HISTORY}, 0)
    # The next day re-sends the window re-rounded, as a provider re-adjusting does, and the new
    # session; the raw bars are bit-equal.
    resent = {item: _close(item) * scale for item in HISTORY[1:]} | {NEW: _close(NEW)}
    paid = CorporateActionEvent("listing-aapl", "fixture", HISTORY[2], "CASH_DIVIDEND", None, 0.5)
    receipt, counts = day((NEW,), resent, 1, (paid,) if dividend else ())
    revision = store.provider_adjusted_revision(receipt.receipt_hash)
    assert revision is not None
    stored = {
        p.session_date: p.adjusted_close
        for p in store.provider_adjusted_closes("listing-aapl", through=NEW)
    }
    if stands:
        assert counts["precision_kept"] == pytest.approx(scale - 1, rel=1e-3)
        assert revision.changed_return_sessions == (NEW,)
        assert stored == {item: _close(item) for item in (*HISTORY, NEW)}
    else:
        assert "precision_kept" not in counts
        assert HISTORY[1] in revision.changed_return_sessions
        assert stored[HISTORY[2]] == _close(HISTORY[2]) * scale


@pytest.mark.parametrize(
    ("written", "tolerance"),
    [
        ({}, 1e-6),
        ({"corrected": 1}, 0.0),
        ({"action_inserted": 1}, 0.0),
        ({"action_corrected": 2}, 0.0),
    ],
)
def test_a_run_that_moved_a_bar_or_an_action_keeps_no_stored_close(written, tolerance) -> None:
    """requirement: a fresh dividend or a corrected bar written this run is never kept."""
    runner = SimpleNamespace(provider=SimpleNamespace(name="fixture"), as_of_session=NEW)
    runner.adjusted_tolerance = 1e-6
    hydration = SimpleNamespace(actions=(), adjusted_closes=())
    item = SimpleNamespace(listing_id="listing-aapl")
    observed = CurrentUniverseMaintenance._observation(runner, item, hydration, NEW, NOW, written)  # type: ignore[arg-type]
    assert observed.adjusted_tolerance == tolerance
