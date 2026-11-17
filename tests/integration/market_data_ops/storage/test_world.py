"""A workspace's market tables in world: admitted at its open, equal to its tables, checked by
marks alone while unchanged, and a later correction known only from its admission."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import pytest
from tests.workspace_maintenance.acquisition_manifest import NOW, acquisition_manifest
from tests.workspace_maintenance.audit_workspace import audit_one

from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
from alphalattice.foundation.market_data_ops.storage import world
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.project import definition_id

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/storage/world.py"
DAYS = (date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30))
NEW = date(2026, 7, 31)
SYMBOLS = ("AAPL", "MSFT")
CURRENT = """(SELECT * FROM lake.world.{0} x WHERE NOT x.retracted AND NOT EXISTS (SELECT 1
    FROM lake.world.superseded s WHERE s.value_digest = x.value_digest
    AND s.admission_id = x.admission_id))"""
FIXTURE, OLD_ID, DISPLAY = (
    definition_id("provider", name)
    for name in ("fixture", world.LISTING_ALIAS, world.DISPLAY_ALIAS)
)
SAME = {  # each world read beside the 0.2.7 rows it must equal, by 0.2.7 listing id
    "raw_bar": (
        f"SELECT i.old, r.session_date, r.open, r.close, r.volume, r.adjusted_close"
        f" FROM {CURRENT.format('raw_bar')} r JOIN ids i ON i.new = r.listing_id",
        "SELECT r.listing_id, r.session_date, r.open, r.close, CAST(r.volume AS DOUBLE),"
        " a.adjusted_close FROM old.raw_daily_bar_current r LEFT JOIN"
        " old.provider_adjusted_close_current a USING (listing_id, provider, session_date)",
    ),
    "corporate_action": (
        f"SELECT i.old, c.effective_date, c.action_id, c.amount, c.provisional, c.provenance"
        f" FROM {CURRENT.format('corporate_action')} c JOIN ids i ON i.new = c.listing_id",
        "SELECT listing_id, effective_date, action_kind, cash_amount, provisional, provenance"
        " FROM old.corporate_action_current WHERE status = 'ACTIVE'",
    ),
    "listing": (
        f"SELECT i.old, l.market, l.universe_class, d.symbol, a.symbol"
        f" FROM {CURRENT.format('listing')} l"
        " JOIN ids i ON i.new = l.listing_id"
        f" JOIN {CURRENT.format('provider_symbol')} d ON d.listing_id = l.listing_id"
        f" AND d.provider_id = '{DISPLAY}'"
        f" JOIN {CURRENT.format('provider_symbol')} a ON a.listing_id = l.listing_id"
        f" AND a.provider_id = '{OLD_ID}'",
        "SELECT listing_id, 'XNAS', mic, display_symbol, listing_id FROM old.listing",
    ),
    "provider_symbol": (
        f"SELECT i.old, p.symbol, p.valid_from, p.valid_to"
        f" FROM {CURRENT.format('provider_symbol')} p JOIN ids i ON i.new = p.listing_id"
        f" WHERE p.provider_id = '{FIXTURE}'",
        "SELECT listing_id, provider_symbol, effective_from, effective_to"
        " FROM old.provider_symbol_mapping",
    ),
}


def bars(
    store: MarketDataRepository,
    manifest: UniverseManifest,
    days: tuple[date, ...],
    close: float,
    offset: int,
) -> None:
    """Both listings' bars on `days`, written `offset` days after the suites' clock."""
    rows = tuple(
        {"session_date": d.isoformat(), "open": 100.0, "high": 110.0, "low": 99.0}
        | {"close": close, "volume": 1_000_000}
        for d in days
    )
    store.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", dict.fromkeys(SYMBOLS, rows), SYMBOLS),
        ingestion_id=f"day-{offset}",
        observed_at=NOW + timedelta(days=offset),
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[tuple[MarketDataRepository, UniverseManifest, Path]]:
    """A 0.2.7 workspace its own owner wrote: two listings' bars, AAPL's adjusted closes and a
    dividend, and a Host history open on a data root of its own."""
    manifest = acquisition_manifest()
    store = MarketDataRepository(tmp_path / "workspace")
    store.bootstrap(manifest)
    bars(store, manifest, DAYS, 101.0, 0)
    dividend = CorporateActionEvent("listing-aapl", "fixture", DAYS[2], "CASH_DIVIDEND", None, 0.5)
    audit_one(
        store,
        manifest,
        listing_id="listing-aapl",
        provider="fixture",
        actions=(dividend,),
        adjusted_closes=tuple(
            ProviderAdjustedClosePoint("listing-aapl", "fixture", d, 100.5) for d in DAYS
        ),
        history_start=DAYS[0],
        history_end=DAYS[-1],
        observed_at=NOW,
    )
    history.open_history("w1", "workspace", tmp_path / "workspace", root=tmp_path / "data")
    try:
        yield store, manifest, tmp_path / "workspace" / "market-data.duckdb"
    finally:
        history.release_history()


@pytest.mark.proves("market_data_ops.world_holds_the_workspace_market_tables")
def test_a_workspace_open_admits_its_market_tables_once_and_a_correction_from_its_admission(
    workspace: tuple[MarketDataRepository, UniverseManifest, Path],
) -> None:
    """A workspace open admits its market tables into world once, equal row for row, and a
    later correction is seen only from its admission."""
    store, manifest, database = workspace
    first = world.admit(database, "w1")
    again = world.admit(database, "w1")
    lake = history.store().lake
    lake.execute(f"ATTACH '{database.as_posix()}' AS old (READ_ONLY)")
    lake.execute("CREATE TEMP TABLE ids (old VARCHAR, new UUID)")
    for old in ("listing-aapl", "listing-msft"):
        lake.execute("INSERT INTO ids VALUES (?, ?)", (old, world.world_listing(old)))
    differ = {
        name: lake.execute(
            f"SELECT (SELECT count(*) FROM (({new}) EXCEPT ALL ({old}))),"
            f" (SELECT count(*) FROM (({old}) EXCEPT ALL ({new}))), (SELECT count(*) FROM ({new}))"
        ).fetchone()
        for name, (new, old) in SAME.items()
    }
    lake.execute("DETACH old")
    bars(store, manifest, (DAYS[-1], NEW), 103.0, 1)  # the last day corrected, and a new one
    corrected = world.admit(database, "w1")
    known = history.store().log.read(
        "SELECT known_us FROM admission WHERE dataset = 'raw_bar' ORDER BY admission_seq DESC"
    )[0][0]
    read = "SELECT list(close ORDER BY session_date) FROM raw_bar_asof(?) WHERE listing_id = ?"
    aapl = world.world_listing("listing-aapl")
    seen = [lake.execute(read, (cutoff, aapl)).fetchone()[0] for cutoff in (known - 1, known)]
    assert first == {
        "rows": {"session": 14, "listing": 2, "provider_symbol": 6, "raw_bar": 8}
        | {"corporate_action": 1}
    }
    assert again is None
    assert differ == {"raw_bar": (0, 0, 8), "corporate_action": (0, 0, 1)} | {
        "listing": (0, 0, 2),
        "provider_symbol": (0, 0, 2),
    }
    assert corrected == {"rows": {"raw_bar": 4}}
    assert seen == [[101.0] * 4 + [103.0], [101.0] * 3 + [103.0] * 2]
