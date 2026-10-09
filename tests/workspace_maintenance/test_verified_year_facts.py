"""A store's year facts: what a closed year holds, kept until a write or an edit ends it."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import duckdb

from alphalattice.control.workspace_runtime.verified_facts import (
    forget_year_facts,
    record_year_facts,
    year_facts,
)
from alphalattice.foundation.market_data_ops.sources.contracts import ProviderAdjustedClosePoint
from alphalattice.foundation.market_data_ops.sources.manifest import (
    build_quality_filtered_research_manifest,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from tests.workspace_maintenance.acquisition_manifest import NOW, acquisition_manifest

Day = Callable[[list[date], date, int], tuple[str | None, str]]


def _audits(tmp_path: Path) -> tuple[MarketDataRepository, Day, str]:
    """A one-listing store across a year end, its first full audit taken: the store, a day of
    new bars with a rolling audit from a start (its prior and next series hashes), and the
    full audit's series hash."""
    manifest = build_quality_filtered_research_manifest(
        acquisition_manifest(), eligible_listing_ids=("listing-aapl",)
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    market_data.bootstrap(manifest)
    sessions: list[date] = []
    bar = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000}

    def day(new: list[date], start: date, offset: int) -> tuple[str | None, str]:
        sessions.extend(new)
        market_data.apply_validated_batch(
            manifest,
            sanitize_payload(
                manifest,
                "fixture",
                {"AAPL": tuple({"session_date": item.isoformat()} | bar for item in new)},
                ("AAPL",),
            ),
            ingestion_id=f"day-{offset}",
            observed_at=NOW + timedelta(days=offset),
        )
        session = sessions[-1]
        receipt, _ = market_data.complete_action_audit(
            manifest,
            listing_id="listing-aapl",
            provider="fixture",
            observed_actions=(),
            observed_adjusted_closes=tuple(
                ProviderAdjustedClosePoint("listing-aapl", "fixture", item, 100.0 + item.day)
                for item in sessions
                if item >= start
            ),
            history_start=start,
            history_end=session,
            requested_as_of=session,
            observed_at=NOW + timedelta(days=offset),
        )
        revision = market_data.provider_adjusted_revision(receipt.receipt_hash)
        assert revision is not None
        return revision.prior_series_hash, revision.next_series_hash

    history = [date(2025, 12, 30), date(2025, 12, 31), date(2026, 7, 28), date(2026, 7, 29)]
    _, whole = day(history, history[0], 0)
    return market_data, day, whole


def test_a_rolling_audit_holds_closed_years_by_facts_equal_to_the_whole_series(
    tmp_path: Path,
) -> None:
    """requirement: a rolling audit holds each closed year by its fact, and its prior series
    hash equals the hash the whole series gave the day before."""
    _market_data, day, whole = _audits(tmp_path)
    held, _ = day([date(2026, 7, 30)], date(2026, 7, 29), 1)
    assert held == whole


def test_an_edit_while_the_store_was_closed_ends_the_facts_and_is_seen(tmp_path: Path) -> None:
    """tamper: another program's edit to a closed year while the store was closed ends its
    facts, so the next rolling audit reads the year and its series hash moves."""
    market_data, day, _whole = _audits(tmp_path)
    _, after = day([date(2026, 7, 30)], date(2026, 7, 29), 1)
    outside = duckdb.connect(str(market_data.database.path))
    try:
        outside.execute(
            "UPDATE provider_adjusted_close_current SET adjusted_close = 1.0 "
            "WHERE session_date = ?",
            [date(2025, 12, 30)],
        )
    finally:
        outside.close()
    edited, _ = day([date(2026, 7, 31)], date(2026, 7, 29), 2)
    assert edited != after


def test_a_store_that_predates_the_table_holds_no_facts_until_its_first_record() -> None:
    """recovery: a store without the year-fact table answers no facts and ends none, inside a
    writer's transaction, and its first record creates the table."""
    connection = duckdb.connect()
    fact = {"kind": "k", "scope": "s", "basis": "b", "epoch": "e"}
    connection.execute("BEGIN TRANSACTION")
    assert year_facts(connection, **fact, years=(2025,)) == {}
    forget_year_facts(connection, "k", years=(2025,))
    record_year_facts(connection, **fact, values={2025: "v"})
    connection.execute("COMMIT")
    assert year_facts(connection, **fact, years=(2025,)) == {2025: "v"}
