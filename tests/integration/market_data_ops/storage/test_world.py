"""A workspace's market tables in world: admitted at its open, equal to its tables, checked by
marks alone while unchanged, and a later correction known only from its admission."""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    FactorInputBundle,
    file_digest,
    select_research_inputs,
)
from alphalattice.control.workspace_runtime.database import release_workspace_database
from alphalattice.foundation.data_layer import history
from alphalattice.foundation.data_layer.chain import digest as text_digest
from alphalattice.foundation.data_layer.store import KNOWN
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
from alphalattice.foundation.market_data_ops.storage import duckdb as market_storage
from alphalattice.foundation.market_data_ops.storage import market_snapshots, world, world_reads
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.foundation.market_data_ops.storage.world_upgrade import market_upgrade_family
from alphalattice.kernel import storage
from alphalattice.kernel.clock import task_observation
from tests.workspace_maintenance.acquisition_manifest import NOW, acquisition_manifest
from tests.workspace_maintenance.audit_workspace import audit_one

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/storage/world.py"
DAYS = (date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30))
NEW = date(2026, 7, 31)
SYMBOLS = ("AAPL", "MSFT")
CURRENT = """(SELECT * FROM lake.world.{0} x WHERE NOT x.retracted AND NOT EXISTS (SELECT 1
    FROM lake.world.superseded s WHERE s.version_id = x.version_id))"""
SAME = {  # each world read beside the 0.2.7 rows it must equal, by 0.2.7 listing id
    "raw_bar": (
        "SELECT i.old, r.session_date, r.open, r.close, r.volume, r.adjusted_close,"
        " r.source_payload_digest"
        f" FROM {CURRENT.format('raw_bar')} r JOIN ids i ON i.new = r.listing_id",
        "SELECT r.listing_id, r.session_date, r.open, r.close, CAST(r.volume AS DOUBLE),"
        " a.adjusted_close, r.payload_hash FROM old.raw_daily_bar_current r LEFT JOIN"
        " old.provider_adjusted_close_current a USING (listing_id, provider, session_date)",
    ),
    "corporate_action": (
        "SELECT i.old, c.effective_date, c.action_id, c.amount, c.provisional, c.provenance,"
        " c.source_payload_digest"
        f" FROM {CURRENT.format('corporate_action')} c JOIN ids i ON i.new = c.listing_id",
        "SELECT listing_id, effective_date, action_kind, cash_amount, provisional, provenance,"
        " payload_hash"
        " FROM old.corporate_action_current WHERE status = 'ACTIVE'",
    ),
    "listing": (
        f"SELECT i.old, m.mic, l.universe_class, d.symbol, a.symbol"
        f" FROM {CURRENT.format('listing')} l"
        f" JOIN {CURRENT.format('market')} m USING (market_id)"
        " JOIN ids i ON i.new = l.listing_id"
        f" JOIN {CURRENT.format('provider_symbol')} d ON d.listing_id = l.listing_id"
        " AND d.provider_id = '{display}'"
        f" JOIN {CURRENT.format('provider_symbol')} a ON a.listing_id = l.listing_id"
        " AND a.provider_id = '{old_id}'",
        "SELECT listing_id, 'XNAS', mic, display_symbol, listing_id FROM old.listing",
    ),
    "provider_symbol": (
        f"SELECT i.old, p.symbol, p.valid_from, p.valid_to"
        f" FROM {CURRENT.format('provider_symbol')} p JOIN ids i ON i.new = p.listing_id"
        " WHERE p.provider_id = '{fixture}'",
        "SELECT listing_id, provider_symbol, effective_from, effective_to"
        " FROM old.provider_symbol_mapping",
    ),
}


def bars(
    store: MarketDataRepository,
    manifest: UniverseManifest,
    days: tuple[date, ...],
    close: float,
    offset: float,
    symbols: Sequence[str] = SYMBOLS,
) -> None:
    """The listings' bars on `days`, both by default, written `offset` days after the suites'
    clock."""
    rows = tuple(
        {"session_date": d.isoformat(), "open": 100.0, "high": 110.0, "low": 99.0}
        | {"close": close, "volume": 1_000_000}
        for d in days
    )
    store.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", dict.fromkeys(symbols, rows), tuple(symbols)),
        ingestion_id=f"day-{offset}",
        observed_at=NOW + timedelta(days=offset),
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[tuple[MarketDataRepository, UniverseManifest, Path]]:
    """A workspace its own owner wrote: two listings' bars, AAPL's adjusted closes and a
    dividend, and a Host history open on a data root of its own."""
    manifest = acquisition_manifest()
    store = MarketDataRepository(tmp_path / "workspace", universe_reader="workspace")
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


def _provider_summary(dataset: str, providers: list[str]) -> dict[str, Any]:
    """Read the one bounded summary and check it keeps the complete provider list."""
    [(attrs, body, named_digest)] = history.store().log.read(
        "SELECT attrs, text, text_digest FROM event LEFT JOIN event_text USING (seq)"
        " WHERE type = 'admission' AND subject_id = 'w1'"
        " AND json_extract(attrs, '$.dataset') = ?",
        (dataset,),
    )
    assert body is not None and json.loads(body) == providers
    assert text_digest(body.encode()) == named_digest
    summary = json.loads(attrs)
    assert summary.pop("provider") == {"count": len(providers), "digest": named_digest}
    return summary


def _universe_records_follow_their_rows(
    store: MarketDataRepository,
    manifest: UniverseManifest,
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A universe's records never run ahead of the rows they name: an admission interrupted
    inside its snapshot leaves no manifest version, bootstrap or journal event; a universe's first
    admission takes two snapshots, its intervals after the version that mints its lineage, and a
    later one holds its market batches and intervals in one."""
    from contextlib import contextmanager

    from alphalattice.foundation.market_data_ops.sources.membership import (
        MembershipEvent,
        UniverseBootstrapRecord,
    )

    profile = manifest.profile.market_profile_id
    port = history.store()
    publish = port.publish
    journal = (
        "SELECT count(*) FROM event WHERE type IN"
        " ('universe.bootstrapped', 'universe.member_added', 'universe.member_removed')"
    )

    def interrupted_at(member_sets: int) -> None:
        """Admit, interrupted at the `member_sets`-th universe member batch."""
        seen = []

        def counted(table: str, *args: Any) -> Any:
            if table == "universe_member":
                seen.append(table)
                if len(seen) == member_sets:
                    raise KeyboardInterrupt
            return publish(table, *args)

        monkeypatch.setattr(port, "publish", counted)
        with pytest.raises(KeyboardInterrupt):
            world.admit(database, "w1")
        monkeypatch.setattr(port, "publish", publish)

    def interrupted_at_its_end(update: Callable[[], None]) -> None:
        """Run `update`, the admission it ends with interrupted once its snapshot's writes are
        all made, before the snapshot commits."""
        admitting, depth = port.admitting, [0]

        @contextmanager
        def ending() -> Iterator[None]:
            depth[0] += 1
            try:
                with admitting():
                    yield
                    if depth[0] == 1:
                        raise KeyboardInterrupt
            finally:
                depth[0] -= 1

        monkeypatch.setattr(port, "admitting", ending)
        with pytest.raises(KeyboardInterrupt):
            update()
        monkeypatch.setattr(port, "admitting", admitting)

    def snapshots(admit: bool) -> int:
        """The catalog snapshots one whole publish takes."""
        (before,) = port.lake.execute("SELECT max(snapshot_id) FROM lake.snapshots()").fetchone()
        assert world.admit(database, "w1") is not None if admit else True
        (count,) = port.lake.execute(
            "SELECT count(*) FROM lake.snapshots() WHERE snapshot_id > ?", [before]
        ).fetchone()
        return count

    record = UniverseBootstrapRecord(
        market_profile_id=profile,
        t0_session=DAYS[1],
        history_start=DAYS[0],
        cohort_listing_ids=("listing-aapl", "listing-msft"),
        cohort_hash="",
        manifest_revision=manifest.revision_sha256,
        candidate_manifest_hash="1" * 64,
        qualification_policy_hash="2" * 64,
        feature_input_policy_hash="3" * 64,
        source_observed_at=NOW - timedelta(days=1),
        admitted_at=NOW,
        panel_snapshot_hash="4" * 64,
        derivation="FIRST_QUALIFIED_PUBLICATION",
    )
    # its writer's publish, interrupted inside its snapshot, records nothing it names
    interrupted_at_its_end(lambda: store.record_universe_bootstrap(record))
    interrupted_at(2)  # the manifest's member set is staged, the cohort's is not yet committed
    held = (world.universe_lineage("w1", profile), port.log.read(journal)[0][0])
    first = snapshots(admit=True)
    cutoff = int(datetime(2026, 8, 4, tzinfo=UTC).timestamp() * 1_000_000)
    manifest_read = world.load_universe_manifest(manifest.manifest_id, cutoff, workspace_id="w1")
    bootstrap_read = world.universe_bootstrap(profile, cutoff, workspace_id="w1")
    recorded = port.log.read(journal)[0][0]
    event = (
        MembershipEvent(
            market_profile_id=profile,
            sequence=1,
            listing_id="listing-msft",
            kind="EXIT",
            effective_session=DAYS[2],
            observed_at=datetime(2026, 7, 28, 15, tzinfo=UTC),
            decided_at=datetime(2026, 7, 28, 16, tzinfo=UTC),
            authority="fixture",
            reference_hash="5" * 64,
            manifest_revision=manifest.revision_sha256,
        ),
    )
    writer = MarketDataRepository(store.workspace)

    def update() -> None:  # new bars and a member's exit, admitted together at the update's end
        with writer.writing():
            bars(writer, manifest, (NEW,), 102.0, 1)
            writer.append_membership_events(event)

    # the universe is known: its intervals are written inside the update's one snapshot
    interrupted_at_its_end(update)
    after_interruption = port.log.read(journal)[0][0]
    later = snapshots(admit=True)
    assert held == (None, 0)
    assert (first, later) == (2, 1)
    assert manifest_read == store.load_universe_manifest(manifest.manifest_id)
    assert bootstrap_read == store.universe_bootstrap(profile)
    assert after_interruption == recorded
    assert port.log.read(journal)[0][0] == recorded + 1


@pytest.mark.proves("market_data_ops.a_reverted_mapping_is_admitted_again")
def test_a_mapping_reverted_after_an_admission_that_took_none_of_it_is_admitted_again(
    workspace: tuple[MarketDataRepository, UniverseManifest, Path],
) -> None:
    """An admission whose rebuilt mappings are unchanged takes none of them; a mapping's change and
    its reversal after it are each admitted, so world's newest version is the reverted one."""
    store, manifest, database = workspace
    world.admit(database, "w1")
    bars(store, manifest, (NEW,), 102.0, 1)  # the bars move, the mappings do not
    world.admit(database, "w1")
    port = history.store()
    fixture = port.log.require_lineage("provider", "fixture")
    states = []
    for state in ("DELISTED", "ACTIVE"):
        with store._connect() as connection:
            connection.execute(
                "UPDATE provider_symbol_mapping SET lifecycle_state = ?"
                " WHERE provider = 'fixture' AND listing_id = 'listing-aapl'",
                [state],
            )
        world.admit(database, "w1")
        states.append(
            port.lake.execute(
                f"SELECT p.lifecycle_state FROM {CURRENT.format('provider_symbol')} p"
                " WHERE p.provider_id = ? AND p.listing_id = ?",
                [fixture, world.world_listing("listing-aapl")],
            ).fetchall()
        )
    assert states == [[("DELISTED",)], [("ACTIVE",)]]


@pytest.mark.proves("market_data_ops.world_holds_the_workspace_market_tables")
@pytest.mark.parametrize(
    "focus",
    (
        pytest.param(
            "identity",
            id="identity",
            marks=pytest.mark.proves("market_data_ops.market_upgrade_is_all_equal"),
        ),
        "workspace",
        "lineage",
        "providers",
        "publication",
        "absent-file",
        "resent",
        "delta",
    ),
)
def test_a_workspace_open_admits_its_market_tables_once_and_a_correction_from_its_admission(
    workspace: tuple[MarketDataRepository, UniverseManifest, Path],
    monkeypatch: pytest.MonkeyPatch,
    focus: str,
    tmp_path: Path,
) -> None:
    """A workspace open admits its market tables into world once, equal row for row, and a
    later correction is seen only from its admission."""
    store, manifest, database = workspace
    if focus == "providers":
        providers = [f"development-alias-{index:02d}-" + "a" * 225 for index in range(17)]
        with store.database.transaction() as connection:
            listing, symbol, start, end, state = connection.execute(
                "SELECT listing_id, provider_symbol, effective_from, effective_to, lifecycle_state"
                " FROM provider_symbol_mapping WHERE provider = 'fixture'"
                " ORDER BY listing_id LIMIT 1"
            ).fetchone()
            connection.executemany(
                "INSERT INTO provider_symbol_mapping VALUES (?, ?, ?, ?, ?, ?)",
                [(listing, provider, symbol, start, end, state) for provider in providers],
            )
        try:
            admitted = world.admit(database, "w1")
        except sqlite3.IntegrityError as error:
            if "length(attrs) <= 4096" not in str(error):
                raise
            pytest.fail(f"the market summary exceeded the attrs limit: {error}")
        assert admitted is not None
        summary = _provider_summary("market", sorted([*providers, "fixture"]))
        assert summary["journal"] is not None  # the journal it took through, which it left
        results = {key: summary[key] for key in ("rows", "dropped", "rewritten") if key in summary}
        assert results == admitted
        assert world.admit(database, "w1") == {"rows": {}}
        return
    if focus == "resent":
        _a_series_sent_again_unchanged_is_no_delta(store, manifest)
        return
    if focus == "delta":
        _a_publish_takes_its_exact_keys_each_known_at_its_own_time(store, manifest, monkeypatch)
        return
    if focus == "publication":
        _universe_records_follow_their_rows(store, manifest, database, monkeypatch)
        return
    if focus == "absent-file":  # an update that ends before its file exists published nothing
        absent = tmp_path / "unmade" / "market-data.duckdb"
        with world_reads.writing(
            absent, lambda _: pytest.fail("a file that does not exist was published")
        ):
            pass
        assert not absent.exists()
        return
    if focus == "identity":
        from functools import partial

        from alphalattice.control.product_host.composition import first_start
        from alphalattice.control.product_host.composition.application_session import (
            verify_upgrade_backups,
        )
        from alphalattice.control.product_host.storage.backup import WorkspaceBackups
        from alphalattice.foundation.data_layer.backup import backup_data_root
        from alphalattice.foundation.data_layer.upgrade import UpgradeRunner

        assert "market-data" in [make().name for make in first_start.FAMILIES]
        family = market_upgrade_family(lambda _workspace: "w1")
        assert not family.holds(store.workspace)
        with store.database.transaction() as connection:
            connection.execute("COMMENT ON TABLE market_profile IS NULL")
        assert family.holds(store.workspace)
        (store.workspace / "research-workspace.json").write_text('{"workspace_id":"w1"}')
        history.flush()
        root = tmp_path / "data"
        log = history.held(root)
        generation = backup_data_root(root, tmp_path / "machine-backups")
        backups = WorkspaceBackups(store.workspace, workspace_id="w1", root=tmp_path / "backups")
        backups.create(reason="REQUEST")
        runner = UpgradeRunner(
            root,
            log,
            (store.workspace,),
            (family,),
            partial(
                verify_upgrade_backups, machine_generation=generation, workspace_backups=(backups,)
            ),
        )
        assert runner.run() is True
        assert family.holds(store.workspace) is False
        assert runner.run() is False
        assert runner.id_map["market-data", "XNAS"] == str(
            history.store()
            .lake.execute(
                "SELECT market_id FROM market_asof(?) WHERE mic = 'XNAS'", [world_reads.LATEST]
            )
            .fetchone()[0]
        )
        attrs = json.loads(
            log.read(
                "SELECT attrs FROM event WHERE type = 'admission' AND subject_id = 'w1'"
                " AND json_extract(attrs, '$.dataset') = 'market' ORDER BY seq DESC LIMIT 1"
            )[0][0]
        )
        first = {key: attrs[key] for key in ("rows", "dropped", "rewritten") if key in attrs}
    else:
        first = world.admit(database, "w1")
    again = world.admit(database, "w1")
    lake = history.store().lake
    assert lake.execute(
        "SELECT count(*) FROM lake.world.session WHERE known_asof <= make_timestamptz(0)"
    ).fetchall() == [(first["rows"]["session"],)]
    assert lake.execute(
        "SELECT calendar_origin, calendar_version, count(*) FROM lake.world.session"
        " WHERE known_at IS NULL AND source_observed_at IS NULL AND isinf(known_asof)"
        " GROUP BY ALL"
    ).fetchall() == [
        ("exchange_calendars", version("exchange-calendars"), first["rows"]["session"])
    ]
    parts = history.store().log.read(
        "SELECT parts FROM admission WHERE dataset = 'session' ORDER BY admission_seq DESC LIMIT 1"
    )[0][0]
    assert json.loads(parts) == [
        {"part": "all", "min_known_asof_us": -(1 << 63), "max_known_asof_us": -(1 << 63)}
    ]
    if focus == "lineage":
        # The same correction has the same original key in two workspaces, even when
        # each workspace observed it on a different day. Its log lineage must survive.
        second = database.parent.parent / "second"
        second.mkdir()
        shutil.copy2(database, second / database.name)
        other = MarketDataRepository(second)
        history.workspace_on_log(second, "second", workspace_id="w2", root=history.store().root)
        originals = {}
        for workspace_id, repository, offset in (("w1", store, 1), ("w2", other, 2)):
            audit_one(
                repository,
                manifest,
                listing_id="listing-aapl",
                provider="fixture",
                actions=(
                    CorporateActionEvent(
                        "listing-aapl", "fixture", DAYS[2], "CASH_DIVIDEND", None, 0.75
                    ),
                ),
                adjusted_closes=tuple(
                    ProviderAdjustedClosePoint("listing-aapl", "fixture", day, 100.5)
                    for day in DAYS
                ),
                history_start=DAYS[0],
                history_end=DAYS[-1],
                observed_at=NOW + timedelta(days=offset),
            )
            world.admit(repository.path, workspace_id)
            with repository._connect(read_only=True) as connection:
                originals[workspace_id] = connection.execute(
                    "SELECT revision_id, epoch_us(observed_at) FROM corporate_action_revision"
                    " ORDER BY revision_id"
                ).fetchall()
        assert [row[0] for row in originals["w1"]] == [row[0] for row in originals["w2"]]
        logged = history.store().log.read(
            "SELECT workspace_id, subject_id, occurred_us FROM event"
            " WHERE type = 'market.action_revision' ORDER BY workspace_id, subject_id"
        )
        assert [tuple(row) for row in logged] == [
            (workspace_id, key, observed)
            for workspace_id, rows in originals.items()
            for key, observed in rows
        ]
        for repository in (store, other):
            for instant in (NOW + timedelta(days=1, hours=12), NOW + timedelta(days=2)):
                cutoff = world_reads.cutoff(instant)
                with repository._connect(read_only=True) as connection:
                    expected = connection.execute(
                        "SELECT revision_id, epoch_us(observed_at) FROM corporate_action_revision"
                        " WHERE epoch_us(observed_at) <= ? ORDER BY revision_id",
                        [cutoff],
                    ).fetchall()
                with repository.market_reads(cutoff_us=cutoff) as (cursor, tables):
                    assert tables != world_reads.WORKSPACE
                    assert (
                        cursor.execute(
                            "SELECT revision_id, epoch_us(observed_at)"
                            f" FROM {tables['action_revisions']} ORDER BY revision_id"
                        ).fetchall()
                        == expected
                    )
            # Once both observations are known, each current audit uses its own journal.
            # A past world read must not be compared with a future corrected current row.
            with repository._connect(read_only=True) as connection:
                expected_hash = repository._action_evidence_hash(
                    connection, listing_id="listing-aapl", provider="fixture"
                )
            with repository.market_reads() as (cursor, tables):
                assert (
                    repository._action_evidence_hash(
                        cursor, listing_id="listing-aapl", provider="fixture", tables=tables
                    )
                    == expected_hash
                )
        assert world.admit(other.path, "w2") == {"rows": {}}
        return
    set_id = world.admit_member_set(("listing-aapl", "listing-msft"), DAYS[1], "fixture")
    assert world.admit_member_set(("listing-msft", "listing-aapl"), DAYS[1], "fixture") == set_id
    start = int(datetime(2026, 7, 28, tzinfo=UTC).timestamp() * 1_000_000)  # its valid date
    members = (
        "SELECT listing_id::VARCHAR, known_at FROM universe_member_asof(?)"
        " WHERE universe_id = ? ORDER BY 1"
    )
    assert lake.execute(members, (start - 1, set_id)).fetchall() == []
    assert lake.execute(members, (start, set_id)).fetchall() == [
        (item, None)
        for item in sorted(world.world_listing(old) for old in ("listing-aapl", "listing-msft"))
    ]
    world.admit_universe_manifest(manifest, "w1")
    manifest_cutoff = int(datetime(2026, 8, 4, tzinfo=UTC).timestamp() * 1_000_000)
    assert world.load_universe_manifest(
        manifest.manifest_id, manifest_cutoff, workspace_id="w1"
    ) == store.load_universe_manifest(manifest.manifest_id)
    from alphalattice.foundation.market_data_ops.sources.membership import UniverseBootstrapRecord

    bootstrap = UniverseBootstrapRecord(
        market_profile_id=manifest.profile.market_profile_id,
        t0_session=DAYS[1],
        history_start=DAYS[0],
        cohort_listing_ids=("listing-aapl", "listing-msft"),
        cohort_hash="",
        manifest_revision=manifest.revision_sha256,
        candidate_manifest_hash="1" * 64,
        qualification_policy_hash="2" * 64,
        feature_input_policy_hash="3" * 64,
        source_observed_at=NOW - timedelta(days=1),
        admitted_at=NOW,
        panel_snapshot_hash="4" * 64,
        derivation="FIRST_QUALIFIED_PUBLICATION",
    )
    store.record_universe_bootstrap(bootstrap)
    world.admit_universe_bootstrap(bootstrap, "fixture", "w1")
    assert world.universe_bootstrap(
        manifest.profile.market_profile_id, manifest_cutoff, workspace_id="w1"
    ) == store.universe_bootstrap(manifest.profile.market_profile_id)
    from alphalattice.foundation.market_data_ops.sources.membership import MembershipEvent

    event = MembershipEvent(
        market_profile_id=manifest.profile.market_profile_id,
        sequence=1,
        listing_id="listing-msft",
        kind="EXIT",
        effective_session=DAYS[2],
        observed_at=datetime(2026, 7, 28, 15, tzinfo=UTC),
        decided_at=datetime(2026, 7, 28, 16, tzinfo=UTC),
        authority="fixture",
        reference_hash="5" * 64,
        manifest_revision=manifest.revision_sha256,
    )
    MarketDataRepository(store.workspace).append_membership_events((event,))
    assert world.membership_events(
        manifest.profile.market_profile_id, workspace_id="w1"
    ) == store.membership_events(manifest.profile.market_profile_id)
    live = world.universe_lineage("w1", manifest.profile.market_profile_id)
    exit_start = int(datetime(2026, 7, 29, tzinfo=UTC).timestamp() * 1_000_000)
    interval = (
        "SELECT valid_to FROM universe_member_asof(?) WHERE universe_id = ? AND listing_id = ?"
    )
    assert lake.execute(
        interval, (exit_start - 1, live, world.world_listing("listing-msft"))
    ).fetchone() == (None,)
    assert lake.execute(
        interval, (exit_start, live, world.world_listing("listing-msft"))
    ).fetchone() == (DAYS[2],)
    old_schedule = store.membership_schedule(
        manifest.profile.market_profile_id, sessions=DAYS, fallback_listing_ids=()
    )
    new_schedule = world.membership_schedule(
        manifest.profile.market_profile_id, DAYS, (), manifest_cutoff, workspace_id="w1"
    )
    assert new_schedule == old_schedule
    selected = MarketDataRepository(store.workspace, cutoff_us=manifest_cutoff)
    with selected._universe_reads() as connection:
        assert connection is not None
    assert selected.load_universe_manifest(manifest.manifest_id) == store.load_universe_manifest(
        manifest.manifest_id
    )
    assert selected.load_universe_manifest_revision(
        manifest.revision_sha256
    ) == store.load_universe_manifest_revision(manifest.revision_sha256)
    assert selected.universe_bootstrap(manifest.profile.market_profile_id) == bootstrap
    assert selected.membership_events(manifest.profile.market_profile_id) == (event,)
    assert (
        selected.membership_schedule(
            manifest.profile.market_profile_id, sessions=DAYS, fallback_listing_ids=()
        )
        == old_schedule
    )
    lake.execute(f"ATTACH '{database.as_posix()}' AS old (READ_ONLY)")
    lake.execute("CREATE TEMP TABLE ids (old VARCHAR, new UUID)")
    for old in ("listing-aapl", "listing-msft"):
        lake.execute("INSERT INTO ids VALUES (?, ?)", (old, world.world_listing(old)))
    log = history.store().log
    names = {"fixture": "fixture", "old_id": world.LISTING_ALIAS, "display": world.DISPLAY_ALIAS}
    providers = {key: log.require_lineage("provider", name) for key, name in names.items()}
    differ = {
        name: lake.execute(
            f"SELECT (SELECT count(*) FROM (({new}) EXCEPT ALL ({old}))),"
            f" (SELECT count(*) FROM (({old}) EXCEPT ALL ({new}))), (SELECT count(*) FROM ({new}))"
        ).fetchone()
        for name, (sql, old) in SAME.items()
        for new in (sql.format_map(providers),)
    }
    lake.execute("DETACH old")
    # the last day corrected, and a new one: its writer's update publishes them at its end
    bars(store, manifest, (DAYS[-1], NEW), 103.0, 1)
    corrected = json.loads(
        history.store().log.read(
            "SELECT json_extract(attrs, '$.rows') FROM event WHERE type = 'admission'"
            " AND json_extract(attrs, '$.dataset') = 'market' ORDER BY seq DESC LIMIT 1"
        )[0][0]
    )
    known = world_reads.cutoff(NOW + timedelta(days=1))  # when its writer observed the correction
    read = "SELECT list(close ORDER BY session_date) FROM raw_bar_asof(?) WHERE listing_id = ?"
    aapl = world.world_listing("listing-aapl")
    seen = [lake.execute(read, (cutoff, aapl)).fetchone()[0] for cutoff in (known - 1, known)]
    assert first == {
        "rows": {"session": 14, "listing": 2, "provider_symbol": 6, "raw_bar": 8}
        | {"corporate_action": 1}
    }
    assert again == {"rows": {}}
    assert differ == {"raw_bar": (0, 0, 8), "corporate_action": (0, 0, 1)} | {
        "listing": (0, 0, 2),
        "provider_symbol": (0, 0, 2),
    }
    assert corrected == {"raw_bar": 4}
    assert seen == [[101.0] * 4 + [103.0], [101.0] * 3 + [103.0] * 2]
    entry_day = date(2026, 8, 4)
    entry = replace(
        event,
        sequence=2,
        kind="ENTRY",
        effective_session=entry_day,
        observed_at=NOW + timedelta(minutes=30),
        decided_at=NOW + timedelta(hours=1),
        reference_hash="6" * 64,
        event_hash="",
    )
    MarketDataRepository(store.workspace).append_membership_events((entry,))
    entry_close = int(datetime(2026, 8, 4, 20, tzinfo=UTC).timestamp() * 1_000_000)
    early = MarketDataRepository(store.workspace, cutoff_us=entry_close - 1)
    late = MarketDataRepository(store.workspace, cutoff_us=entry_close)
    for reader, expected in ((early, ("listing-aapl",)), (late, ("listing-aapl", "listing-msft"))):
        assert (
            reader.membership_schedule(
                manifest.profile.market_profile_id, sessions=(entry_day,), fallback_listing_ids=()
            ).members(entry_day)
            == expected
        )
    old_schedule = store.membership_schedule(
        manifest.profile.market_profile_id, sessions=DAYS, fallback_listing_ids=()
    )
    # A different workspace owns a different bootstrap even when both use the same profile.
    other = MarketDataRepository(tmp_path / "other-universe", universe_reader="workspace")
    other.bootstrap(manifest)
    bars(other, manifest, (*DAYS, NEW), 101.0, 0)
    audit_one(
        other,
        manifest,
        listing_id="listing-aapl",
        provider="fixture",
        actions=(
            CorporateActionEvent("listing-aapl", "fixture", DAYS[2], "CASH_DIVIDEND", None, 0.5),
        ),
        adjusted_closes=tuple(
            ProviderAdjustedClosePoint("listing-aapl", "fixture", d, 100.5) for d in DAYS
        ),
        history_start=DAYS[0],
        history_end=DAYS[-1],
        observed_at=NOW,
    )
    other_bootstrap = replace(
        bootstrap,
        cohort_listing_ids=("listing-aapl",),
        cohort_hash="",
        admitted_at=NOW + timedelta(seconds=1),
        record_hash="",
    )
    other.record_universe_bootstrap(other_bootstrap)
    if focus == "workspace":
        late_exit = replace(
            event,
            listing_id="listing-aapl",
            observed_at=datetime(2026, 7, 29, 19, tzinfo=UTC),
            decided_at=datetime(2026, 7, 29, 21, tzinfo=UTC),
            event_hash="",
        )
        late_entry = replace(
            late_exit,
            sequence=2,
            kind="ENTRY",
            effective_session=DAYS[3],
            observed_at=datetime(2026, 7, 30, 19, tzinfo=UTC),
            decided_at=datetime(2026, 7, 30, 21, tzinfo=UTC),
            event_hash="",
        )
        other.append_membership_events((late_exit, late_entry))
    history.open_history("w-universe2", "other", other.workspace, root=tmp_path / "data")
    try:
        world.admit(other.path, "w-universe2")
        other_world = MarketDataRepository(other.workspace, cutoff_us=manifest_cutoff)
        with other_world._universe_reads() as source:
            assert source is not None and source[1] == "w-universe2"
        with selected._universe_reads() as source:
            assert source is not None and source[1] == "w1"
        assert other_world.universe_bootstrap(manifest.profile.market_profile_id) == other_bootstrap
        if focus == "identity":
            assert other_world.membership_schedule(
                manifest.profile.market_profile_id, sessions=DAYS, fallback_listing_ids=()
            ) == other.membership_schedule(
                manifest.profile.market_profile_id, sessions=DAYS, fallback_listing_ids=()
            )
        assert selected.universe_bootstrap(manifest.profile.market_profile_id) == bootstrap
        assert (
            selected.membership_schedule(
                manifest.profile.market_profile_id, sessions=DAYS, fallback_listing_ids=()
            )
            == old_schedule
        )
        assert world.universe_lineage(
            "w1", manifest.profile.market_profile_id
        ) != world.universe_lineage("w-universe2", manifest.profile.market_profile_id)
        if focus == "workspace":
            # A decision after the effective close cannot change that session's historical read.
            schedule = other_world.membership_schedule(
                manifest.profile.market_profile_id,
                sessions=(DAYS[2], DAYS[3], NEW),
                fallback_listing_ids=(),
            )
            assert schedule.members(DAYS[2]) == ("listing-aapl",)
            assert schedule.members(DAYS[3]) == ()
            assert schedule.members(NEW) == ("listing-aapl",)
    finally:
        history.release_history()
    lake.execute(f"ATTACH '{database.as_posix()}' AS old (READ_ONLY)")
    revisions = lake.sql("SELECT * FROM old.bar_revision ORDER BY revision_id").to_arrow_table()
    cutoff = lake.execute("SELECT min(epoch_us(observed_at)) FROM old.bar_revision").fetchone()[0]
    lake.execute("DETACH old")
    original = {
        row["revision_id"]: {
            name: value.isoformat() if isinstance(value, date) else value
            for name, value in row.items()
        }
        for row in revisions.to_pylist()
    }
    logged = history.store().log.read(
        "SELECT subject_id, attrs, occurred_us FROM event WHERE type = 'market.bar_revision'"
    )
    assert {key: json.loads(attrs) for key, attrs, _at in logged} == original
    assert [at for _key, _attrs, at in logged] == [cutoff, cutoff]
    history.flush()
    before = world_reads.revisions("bar_revision", cutoff - 1, workspace_id="w1")
    at = world_reads.revisions("bar_revision", cutoff, workspace_id="w1")
    with store.market_reads(cutoff_us=cutoff) as (cursor, _tables):
        assert cursor.execute(f"SELECT count(*) FROM {before}").fetchone()[0] == 0
        assert cursor.execute(f"SELECT count(*) FROM {at}").fetchone()[0] == 2
    assert world.admit(database, "w1") == {"rows": {}}
    assert (
        history.store().log.read("SELECT count(*) FROM event WHERE type = 'market.bar_revision'")[
            0
        ][0]
        == 2
    )

    # Legacy sector classifications use the same listing ids and lag rule. A changed
    # classification is visible only after its own admission, and a repeat adds nothing.
    connection = store._connect()
    try:
        connection.execute(
            "INSERT INTO sector_classification_current VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "listing-aapl",
                "fixture",
                "AAPL",
                "Technology",
                None,
                "a" * 64,
                "b" * 64,
                "c" * 64,
                NOW,
            ],
        )
        connection.execute(
            "INSERT INTO sector_classification_current VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "listing-msft",
                "fixture",
                "MSFT",
                "Technology",
                None,
                "a" * 64,
                "b" * 64,
                "c" * 64,
                NOW,
            ],
        )
    finally:
        connection.close()
    sector = world.admit_sectors(database, "w1")
    summary = _provider_summary("sector", ["fixture"])
    assert summary["rows"] == sector["rows"]
    with store._connect(read_only=True) as connection:
        marks = {
            name: connection.execute(
                f"SELECT [count(*), bit_xor(hash(t))] FROM {table} t"
            ).fetchone()[0]
            for name, table in (
                ("current", "sector_classification_current"),
                ("revision", "sector_classification_revision"),
            )
        }
    assert summary["marks"] == marks
    assert world.admit_sectors(database, "w1") is None
    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
        PanelClosureArtifactStore,
    )
    from alphalattice.foundation.feature_engine.panels.closure_contracts import durable_payload
    from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
        sector_history_as_of,
    )
    from tests.integration.feature_engine.test_sector_forward_rule import _map

    artifacts = PanelClosureArtifactStore(ArtifactResolver(database.parent / "artifacts"))
    old_map = _map("1" * 64, {"listing-aapl": "Technology"})
    artifacts.publish_json(
        category="sector-maps", content_hash=old_map.map_hash, payload=durable_payload(old_map)
    )
    connection = store._connect()
    try:
        connection.execute(
            "UPDATE sector_classification_current SET sector_revision = ?", ["d" * 64]
        )
        connection.execute(
            "UPDATE sector_classification_current SET sector_name = ? WHERE listing_id = ?",
            ["Energy", "listing-aapl"],
        )
    finally:
        connection.close()
    world.admit_sectors(database, "w1")
    known = history.store().log.read(
        "SELECT known_us FROM admission WHERE dataset = 'sector' ORDER BY admission_seq DESC"
    )[0][0]
    read = "SELECT sector_code FROM sector_asof(?) WHERE listing_id = ?"
    assert [lake.execute(read, (cutoff, aapl)).fetchone()[0] for cutoff in (known - 1, known)] == [
        "Technology",
        "Energy",
    ]
    assert world.sector_map(("listing-aapl",), session=DAYS[1], cutoff_us=known - 1) == {
        "listing-aapl": "Technology"
    }
    assert world.sector_map(("listing-aapl",), session=DAYS[1], cutoff_us=known) == {
        "listing-aapl": "Energy"
    }
    from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository

    feature = FeatureStateRepository(store.database, market_data=store)
    assert [
        feature.sector_classifications(("listing-aapl",), cutoff_us=cutoff)
        for cutoff in (known - 1, known)
    ] == [{"listing-aapl": "Technology"}, {"listing-aapl": "Energy"}]
    state = feature.current_sector_state(manifest, cutoff_us=known - 1)
    assert state is not None and state.sector_revision == "d" * 64
    if focus == "identity":
        assert state.sector_by_listing_id["listing-aapl"] == "Energy"
    from alphalattice.foundation.feature_engine.panels.closure_source import (
        PanelClosureSourceRepository,
    )

    source = PanelClosureSourceRepository(
        database_path=database, resolver=ArtifactResolver(database.parent / "artifacts")
    )
    rows = source.read_sector_rows(listing_ids=("listing-aapl",), cutoff_us=known - 1)
    if focus == "identity":
        assert rows[0]["sector_name"] == "Energy" and rows[0]["sector_revision"] == "d" * 64
    assert dict(sector_history_as_of(artifacts, old_map.sector_revision)) == {
        "listing-aapl": "Technology"
    }
    connection = store._connect()
    try:
        connection.execute(
            "INSERT INTO sector_classification_revision (revision_id, listing_id, provider,"
            " prior_payload_hash, next_payload_hash, sector_revision, revision_kind, retrieved_at,"
            " prior_sector_name, sector_name, effective_session)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "d" * 64,
                "listing-aapl",
                "fixture",
                "a" * 64,
                "e" * 64,
                "f" * 64,
                "CORRECTED",
                NOW + timedelta(days=1),
                "Energy",
                "Health Care",
                DAYS[3],
            ],
        )
        connection.execute(
            "UPDATE sector_classification_current SET sector_name = ?", ["Health Care"]
        )
    finally:
        connection.close()
    world.admit_sectors(database, "w1")
    cutoff = history.store().log.clock()
    assert [world.sector_map(("listing-aapl",), session=day, cutoff_us=cutoff) for day in DAYS] == [
        {"listing-aapl": "Energy"},
        {"listing-aapl": "Energy"},
        {"listing-aapl": "Energy"},
        {"listing-aapl": "Health Care"},
    ]
    seen_world = []
    real_map = world.sector_map

    def tracked(*args, **kwargs):
        seen_world.append(True)
        return real_map(*args, **kwargs)

    monkeypatch.setattr(world, "sector_map", tracked)
    assert feature.sector_classifications(("listing-aapl",)) == {"listing-aapl": "Health Care"}
    assert seen_world
    connection = store._connect()
    try:
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    other = database.parent.parent / "other"
    other.mkdir()
    shutil.copy2(database, other / database.name)
    other_store = MarketDataRepository(other)
    connection = other_store._connect()
    try:
        connection.execute(
            "UPDATE sector_classification_current SET sector_name = ? WHERE listing_id = ?",
            ["Energy", "listing-aapl"],
        )
        connection.execute(
            "UPDATE sector_classification_revision SET sector_name = ? WHERE listing_id = ?",
            ["Energy", "listing-aapl"],
        )
    finally:
        connection.close()
    history.workspace_on_log(other, "other", "w2", root=history.store().root)
    world.admit_sectors(other_store.path, "w2")
    assert real_map(("listing-aapl",), session=date.max, cutoff_us=world_reads.LATEST) == {
        "listing-aapl": "Energy"
    }
    seen_world.clear()
    if focus == "workspace":
        # Another workspace's later classification is world's, which every read outside a
        # writing stage takes.
        assert feature.sector_classifications(("listing-aapl",)) == real_map(
            ("listing-aapl",), session=date.max, cutoff_us=world_reads.LATEST
        )
        assert seen_world
        with world_reads.reading(database) as served:
            assert served is not None
    connection = store._connect()
    try:
        connection.execute(
            "UPDATE sector_classification_current SET sector_name = ?", ["Utilities"]
        )
    finally:
        connection.close()
    port = history.store()
    commit = port.commit
    with monkeypatch.context() as dropped:
        dropped.setattr(
            port, "commit", lambda batch: commit(batch)._replace(dropped={"equal_time": 1})
        )
        world.admit_sectors(database, "w1")
    seen_world.clear()
    # every read outside a writing stage is world's, whatever its last sector admission dropped
    assert feature.sector_classifications(("listing-aapl",)) == real_map(
        ("listing-aapl",), session=date.max, cutoff_us=world_reads.LATEST
    )
    assert seen_world
    if focus == "identity":
        # Later world aliases cannot alter the old manifest's frozen listing facts.
        for provider, symbol in ((world.DISPLAY_ALIAS, "AAPL-NEW"), ("fixture", "AAPL.NEW")):
            with world.stage(
                port.root,
                "provider_symbol",
                port.log.require_lineage("provider", provider),
                world._arrow([{"listing_id": aapl, "symbol": symbol, "valid_from": DAYS[3]}]),
            ) as staged:
                port.commit(staged)
        assert (
            world.load_universe_manifest(
                manifest.manifest_id, world_reads.LATEST, workspace_id="w1"
            )
            == manifest
        )
    if focus == "workspace":
        import pyarrow as pa

        from alphalattice.foundation.data_layer.log import uuid7
        from alphalattice.foundation.data_layer.store import stage

        calendar = port.log.require_lineage("provider", world.CALENDARS)
        market = lake.execute(
            "SELECT market_id FROM lake.world.market WHERE mic = 'XNAS'"
        ).fetchall()[0][0]
        recorded = datetime(2026, 8, 30, 12, tzinfo=UTC)
        batch = pa.table(
            {
                "market_id": [str(market)],
                "session_date": [date(2026, 8, 31)],
                "close_utc": [datetime(2026, 8, 31, 20, tzinfo=UTC)],
                "half_day": [False],
                "calendar_origin": ["exchange_calendars"],
                "calendar_version": [version("exchange-calendars")],
                "source_observed_at": [recorded],
            }
        )
        with stage(port.root, "session", calendar, batch) as staged:
            port.commit(staged)
        assert lake.execute(
            "SELECT known_at, known_asof FROM lake.world.session"
            " WHERE market_id = ? AND session_date = ?",
            [market, date(2026, 8, 31)],
        ).fetchall() == [(recorded, recorded)]
        for hour, expected in ((23, ()), (19, ("listing-aapl",))):
            batch = lake.sql(
                "SELECT market_id, session_date, open_utc, ?::TIMESTAMPTZ AS close_utc, half_day"
                " FROM lake.world.session WHERE market_id = ? AND session_date = ?"
                " ORDER BY known_asof DESC LIMIT 1",
                params=[datetime(2026, 7, 29, hour, tzinfo=UTC), market, DAYS[2]],
            ).to_arrow_table()
            with stage(port.root, "session", calendar, batch) as staged:
                port.commit(staged)
            assert other_world.membership_schedule(
                manifest.profile.market_profile_id, sessions=(DAYS[2],), fallback_listing_ids=()
            ).members(DAYS[2]) == ("listing-aapl",)
            current = MarketDataRepository(other_world.workspace, cutoff_us=port.log.clock())
            assert (
                current.membership_schedule(
                    manifest.profile.market_profile_id, sessions=(DAYS[2],), fallback_listing_ids=()
                ).members(DAYS[2])
                == expected
            )
        future_market = uuid7()
        for table, batch in (
            ("market", pa.table({"market_id": [future_market], "mic": ["XNAS"]})),
            (
                "session",
                pa.table(
                    {
                        "market_id": [future_market] * 3,
                        "session_date": [DAYS[2], DAYS[3], NEW],
                        "close_utc": [
                            datetime.combine(day, datetime.min.time(), UTC).replace(hour=23)
                            for day in (DAYS[2], DAYS[3], NEW)
                        ],
                        "half_day": [False] * 3,
                    }
                ),
            ),
            (
                "listing",
                lake.sql(
                    "SELECT listing_id, ?::UUID AS market_id, universe_class, currency,"
                    " issuer_name, first_session, last_session FROM lake.world.listing"
                    " WHERE listing_id = ? ORDER BY known_asof DESC LIMIT 1",
                    params=[future_market, world.world_listing("listing-aapl")],
                ).to_arrow_table(),
            ),
        ):
            with stage(port.root, table, calendar, batch) as staged:
                port.commit(staged)
        schedule = other_world.membership_schedule(
            manifest.profile.market_profile_id,
            sessions=(DAYS[2], DAYS[3], NEW),
            fallback_listing_ids=(),
        )
        assert [schedule.members(day) for day in (DAYS[2], DAYS[3], NEW)] == [
            ("listing-aapl",),
            (),
            ("listing-aapl",),
        ]


@pytest.mark.proves("market_data_ops.world_serves_the_workspace_market_reads")
def test_world_serves_every_read_outside_a_writing_stage_which_reads_its_buffer(
    workspace: tuple[MarketDataRepository, UniverseManifest, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """World serves every read once the Host's open has published; a write publishes at its end,
    a stage's own reads take its buffer while other threads read world, it publishes once, and a
    write no Host has published is read by no one outside its stage."""
    store, manifest, database = workspace
    ids = ["listing-aapl", "listing-msft"]
    uses = {"world": 0}
    relations = world_reads.relations

    def counted(
        cutoff_us: int = world_reads.LATEST,
        held_us: int | None = None,
        listings: Sequence[str] | None = None,
        *,
        workspace_id: str | None = None,
    ) -> dict[str, str]:
        uses["world"] += 1
        return relations(cutoff_us, held_us, listings, workspace_id=workspace_id)

    def read() -> tuple[object, ...]:
        before = uses["world"]
        values = (
            store.raw_bars_by_listing(ids),
            store.actions_by_listing(ids),
            store.provider_adjusted_closes("listing-aapl", through=NEW),
            store.listing_raw_ranges(ids),
        )
        return values, uses["world"] - before

    def admissions() -> int:
        return history.store().log.read(
            "SELECT count(*) FROM event WHERE type = 'admission'"
            " AND json_extract(attrs, '$.dataset') = 'market'"
        )[0][0]

    monkeypatch.setattr(world_reads, "relations", counted)
    with store.database.connect(read_only=False) as connection:
        market_snapshots.install(connection)
    world.publish_pending(store.path, "w1")  # the Host's open publishes what was written before it
    with store._connect(read_only=True) as connection:
        pending = market_snapshots.pending_table(database, "provider_symbol_mapping")
        assert connection.execute(f"SELECT 1 FROM {pending} LIMIT 1").fetchone() is None
        assert connection.execute(
            "SELECT provider_symbol FROM provider_symbol_mapping ORDER BY provider_symbol"
        ).fetchall() == [("AAPL",), ("MSFT",)]
    host = [read()]
    assert store.raw_bars("listing-aapl", cutoff_us=0) == ()
    receipt = store.latest_action_audit_receipt(listing_id="listing-aapl", provider="fixture")
    checked = []
    for _ in range(3):
        before = uses["world"]
        checked.append(
            (
                store.verified_action_audit_receipt(
                    manifest,
                    receipt_hash=receipt.receipt_hash,
                    listing_id="listing-aapl",
                    provider="fixture",
                    requested_as_of=DAYS[-1],
                    now=datetime(1970, 1, 1) + timedelta(microseconds=history.store().log.clock()),
                    allow_historical=True,
                ),
                uses["world"] - before,
            )
        )
    assert [value for value, _calls in checked] == [receipt] * 3
    assert all(calls > 0 for _value, calls in checked)
    bars(store, manifest, (DAYS[-1], NEW), 103.0, 1)  # a write publishes at its update's end
    written = [read()]
    before = admissions()
    # a stage writing and reading listing by listing, its commits in a retained instance's WAL:
    # it reads its buffer, and publishes once at its end, through the retained instance
    with store.database.retain(read_only=False):
        with store.writing():
            bars(store, manifest, (NEW,), 104.0, 2)
            inside = read()
            with ThreadPoolExecutor(1) as request:  # a Host request meanwhile reads world
                elsewhere = request.submit(read).result()
            bars(store, manifest, (NEW,), 105.0, 3)
        held = (admissions() - before, read())
    with store._connect(read_only=True) as connection:
        versions = connection.execute(
            f"SELECT CAST(version_id AS VARCHAR) FROM {market_snapshots.VERSIONS}"
            " WHERE world_table = 'raw_bar' AND day = ? ORDER BY listing_id",
            [NEW],
        ).fetchall()
        anchor = connection.execute(f"SELECT snapshot_id FROM {market_snapshots.ANCHOR}").fetchone()
        scoped = "WHERE session_date = ? ORDER BY listing_id"
        bodies = market_snapshots.version_relations(market_snapshots.VERSIONS, "workspace_market")
        committed = bodies["committed_raw_bars"]
        pending = market_snapshots.pending_table(database, "raw_daily_bar_current")
        assert connection.execute(f"SELECT * FROM {pending} {scoped}", [NEW]).fetchall() == []
        assert (
            connection.execute(f"SELECT * FROM ({committed}) {scoped}", [NEW]).fetchall()
            == connection.execute(f"SELECT * FROM raw_daily_bar_current {scoped}", [NEW]).fetchall()
        )
    assert len(versions) == 2 and anchor is not None
    assert history.store().lake.execute(
        "SELECT close FROM lake.world.raw_bar WHERE version_id IN (?, ?) ORDER BY close",
        [row[0] for row in versions],
    ).fetchall() == [(105.0,), (105.0,)]
    # Import one source's live, delisted and re-listed intervals together. The real
    # StorePort/catalog boundary must retain their values and historical opening state.
    mappings = market_snapshots.pending_table(database, "provider_symbol_mapping")
    with store._connect() as connection:
        connection.executemany(
            f"INSERT INTO {mappings} VALUES (?, 'lifecycle-fixture', ?, ?, ?, ?)",
            [
                ("listing-msft", "LIVE", DAYS[0], None, "ACTIVE"),
                ("listing-aapl", "OLD", DAYS[0], DAYS[2], "DELISTED"),
                ("listing-aapl", "NEW", DAYS[3], None, "ACTIVE"),
            ],
        )
        # A recorded future end is known from its admission, never at that future date.
        connection.execute(
            f"INSERT INTO {mappings} VALUES"
            " ('listing-msft', 'scheduled-fixture', 'LATER', ?, '2099-12-31', 'DELISTED')",
            [DAYS[0]],
        )
    world.admit(database, "w1")
    scheduled = (
        history.store()
        .lake.execute(
            "SELECT valid_to, epoch_us(known_asof) FROM lake.world.provider_symbol"
            " WHERE symbol = 'LATER' ORDER BY ALL"
        )
        .fetchall()
    )
    assert [end for end, _known in scheduled] == [date(2099, 12, 31), None]
    assert max(known for _end, known in scheduled) <= history.store().log.clock()
    delisting = datetime.combine(DAYS[2], datetime.min.time(), UTC)  # its valid date's start
    for instant, expected in (
        (
            delisting - timedelta(microseconds=1),
            [("LIVE", None, "ACTIVE"), ("OLD", None, "ACTIVE")],
        ),
        (delisting, [("LIVE", None, "ACTIVE"), ("OLD", DAYS[2], "DELISTED")]),
        (
            datetime.combine(DAYS[3], datetime.max.time(), UTC),
            [("LIVE", None, "ACTIVE"), ("NEW", None, "ACTIVE"), ("OLD", DAYS[2], "DELISTED")],
        ),
    ):
        cutoff = world_reads.cutoff(instant)
        with store.market_reads(cutoff_us=cutoff) as (cursor, tables):
            assert tables != world_reads.WORKSPACE
            assert (
                cursor.execute(
                    "SELECT provider_symbol, effective_to, lifecycle_state"
                    f" FROM {tables['mappings']} WHERE provider = 'lifecycle-fixture'"
                    " ORDER BY provider_symbol"
                ).fetchall()
                == expected
            )
    for state, equal in (("ACTIVE", False), ("DELISTED", True)):
        with store._connect() as connection:
            connection.execute(
                f"UPDATE {mappings} SET lifecycle_state = ?"
                " WHERE provider = 'lifecycle-fixture' AND provider_symbol = 'OLD'",
                [state],
            )
        assert world_reads.holds(database, world_reads.LATEST) == equal
    history.release_history()
    monkeypatch.setenv(history.DATA_ROOT_ENV, str(database.parent.parent / "data"))
    child = [read()]
    bars(store, manifest, (NEW,), 106.0, 4)  # a write no Host has published
    unseen = [read()]
    assert [read_[1] for read_ in (host[0], written[0], child[0])] == [4, 4, 4]
    assert written[0][0][0]["listing-aapl"][-1].close == 103.0
    assert (inside[0][0]["listing-aapl"][-1].close, inside[1]) == (104.0, 0)
    assert (elsewhere[0][0]["listing-aapl"][-1].close, elsewhere[1]) == (103.0, 4)
    assert (held[0], held[1][0][0]["listing-aapl"][-1].close, held[1][1]) == (1, 105.0, 4)
    # world serves it, without the write no Host published
    assert (unseen[0][0][0]["listing-aapl"][-1].close, unseen[0][1]) == (105.0, 4)


@pytest.mark.proves("market_data_ops.a_moved_selection_is_carried_to_its_new_root")
def test_a_selection_copied_to_another_root_reads_there_once_carried(
    workspace: tuple[MarketDataRepository, UniverseManifest, Path], tmp_path: Path
) -> None:
    """A sealed selection and a published workspace copied to another data root refuse their
    market reads until carried, then read there as they read on their own root; a transaction
    that reads no market row does not wait for the carry."""
    store, _manifest, database = workspace
    with store.database.connect(read_only=False) as connection:
        market_snapshots.install(connection)
    with storage.memory() as raw:  # one-row version-index reads leave no result open on a raw
        market_snapshots.install(raw)  # connection, which DuckDB would refuse its next relation
        assert market_snapshots.enabled(raw) and market_snapshots.enabled(raw)
        assert raw.sql("SELECT 1").fetchall() == [(1,)]
    world.publish_pending(store.path, "w1")
    sealed = MarketDataRepository(tmp_path / "sealed", universe_reader="workspace")
    with store.database.connect(read_only=False) as connection:
        market_snapshots.write_selection_index(
            connection, store.path, sealed.path, omit=frozenset()
        )

    def reads(market: MarketDataRepository) -> dict[str, list[tuple[object, ...]]]:
        with market._connect(read_only=True) as connection:
            return {
                table: connection.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()
                for table in market_snapshots.OVERLAYS
            }

    there, published = reads(sealed), reads(store)
    with sealed._connect(read_only=True) as connection:
        named = connection.execute(
            f"SELECT world_table, version_id FROM {market_snapshots.VERSIONS} ORDER BY ALL"
        ).fetchall()
    history.release_history()
    for held in (database, sealed.path):
        release_workspace_database(held)
    moved = tmp_path / "moved"
    shutil.copytree(database.parent, moved)
    other = tmp_path / "other"
    history.open_history("w2", "moved", moved, root=other)
    unavailable = r"^market_data_ops\.catalog_selection_unavailable$"
    lake = history.store().lake
    copied = MarketDataRepository(moved, universe_reader="workspace")
    for market in (sealed, copied):
        with pytest.raises(ValueError, match=unavailable):
            reads(market)
    with copied.database.read_transaction() as connection:
        assert connection.execute("SELECT count(*) FROM market_profile").fetchone() is not None
    assert world.carry_selection(tmp_path / "data", sealed.path) == len(named)
    assert reads(sealed) == there
    held_ids = sorted(
        (table, version)
        for table in ("raw_bar", "corporate_action", "provider_symbol")
        for (version,) in lake.execute(f"SELECT version_id FROM lake.world.{table}").fetchall()
    )
    assert held_ids == sorted(named)
    assert lake.execute(
        "SELECT count(*) FROM lake.world.superseded WHERE superseded_asof = '-infinity'"
    ).fetchone() == (len(named),)
    snapshot = lake.execute("SELECT max(snapshot_id) FROM lake.snapshots()").fetchone()
    assert world.carry_selection(tmp_path / "data", sealed.path) == 0
    assert lake.execute("SELECT max(snapshot_id) FROM lake.snapshots()").fetchone() == snapshot
    (first,) = lake.execute("SELECT min(version_id) FROM lake.world.raw_bar").fetchone()
    lake.execute("UPDATE lake.world.raw_bar SET close = -1.0 WHERE version_id = ?", [first])
    with pytest.raises(DataError) as conflict:
        world.carry_selection(tmp_path / "data", sealed.path)
    assert conflict.value.failure.code == "data_layer.carried_version_conflict"
    assert lake.execute(
        "SELECT close FROM lake.world.raw_bar WHERE version_id = ?", [first]
    ).fetchone() == (-1.0,)
    assert world.carry_workspace(tmp_path / "data", copied.path) > 0
    assert world.carry_workspace(tmp_path / "data", copied.path) == 0
    assert reads(copied) == published
    world.publish_pending(copied.path, "w2")  # the new root's open selects them under its own ids
    with copied._connect(read_only=True) as connection:
        assert market_snapshots.anchored(connection)
        held = market_snapshots.pending_table(copied.path, "raw_daily_bar_current")
        assert connection.execute(f"SELECT count(*) FROM {held}").fetchone() == (0,)
    assert reads(copied) == published


@pytest.mark.proves("market_data_ops.world_serves_the_workspace_market_reads")
def test_a_first_use_reads_at_its_own_moment_the_mappings_and_members_it_wrote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Sunday first use reads as of its own moment every mapping and member it wrote that day,
    and its receipt verifies at once."""
    now = datetime(2026, 8, 2, tzinfo=UTC)
    fixture = acquisition_manifest()
    manifest = replace(fixture, profile=replace(fixture.profile, manifest_as_of=now.date()))
    store = MarketDataRepository(tmp_path / "workspace", universe_reader="workspace")
    monkeypatch.setattr(market_storage, "utc_now", lambda: now)
    with task_observation(now - timedelta(minutes=1)):
        store.bootstrap(manifest)
    with store._connect(read_only=True) as connection:
        journal_times = connection.execute(
            "SELECT DISTINCT observed_at FROM market_delta WHERE kind = 'reference'"
        ).fetchall()
    assert journal_times == [(now.replace(tzinfo=None),)]
    bars(store, manifest, DAYS, 101.0, 0)
    audit_one(
        store,
        manifest,
        listing_id="listing-aapl",
        provider="fixture",
        actions=(),
        adjusted_closes=tuple(
            ProviderAdjustedClosePoint("listing-aapl", "fixture", d, 100.5) for d in DAYS
        ),
        history_start=DAYS[0],
        history_end=DAYS[-1],
        observed_at=now,
    )
    receipt = store.latest_action_audit_receipt(listing_id="listing-aapl", provider="fixture")
    history.open_history("w1", "workspace", tmp_path / "workspace", root=tmp_path / "data")
    try:
        world.admit(store.workspace / "market-data.duckdb", "w1")
        with store.market_reads(cutoff_us=world_reads.cutoff(now)) as (cursor, tables):
            assert tables != world_reads.WORKSPACE
            known = cursor.execute(f"SELECT count(*) FROM {tables['mappings']}").fetchone()[0]
        with store.market_reads(cutoff_us=world_reads.LATEST) as (cursor, tables):
            held = cursor.execute(f"SELECT count(*) FROM {tables['mappings']}").fetchone()[0]
        verified = store.verified_action_audit_receipt(
            manifest,
            receipt_hash=receipt.receipt_hash,
            listing_id="listing-aapl",
            provider="fixture",
            requested_as_of=DAYS[-1],
            now=now,
        )
        selected = MarketDataRepository(store.workspace, cutoff_us=world_reads.cutoff(now))
        members = selected.load_universe_manifest_revision(manifest.revision_sha256).listings
    finally:
        history.release_history()
    assert (known, verified, members) == (held, receipt, manifest.listings)
    assert known > 0


@pytest.mark.proves("market_data_ops.a_research_copy_reads_world_at_its_selection")
def test_a_research_copy_reads_world_as_of_its_selection_and_its_own_tables_without_one(
    workspace: tuple[MarketDataRepository, UniverseManifest, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy bound in the Host reads world as of its selection, equal to its own tables and
    unmoved by a later correction; a copy bound once world holds another value of one of its keys
    gets no selection, and it and every copy under the workspace setting read their own tables."""
    store, manifest, database = workspace
    world.publish_pending(store.path, "w1")  # the Host's open publishes what was written before it
    root = database.parent
    digest = file_digest(database)
    bundles = [
        FactorInputBundle.create(
            panel_snapshot_hash=panel * 64,
            outcome_snapshot_hash="d" * 64,
            sessions=DAYS,
            files=(("market-data.duckdb", digest),),
            database_snapshot_hash=digest,
        )
        for panel in ("c", "e")
    ]
    copies = []
    for bundle in bundles:  # both copied before the correction
        copy = root / "research-inputs" / bundle.binding_hash / "source" / "market-data.duckdb"
        copy.parent.mkdir(parents=True)
        shutil.copy2(database, copy)
        copies.append(MarketDataRepository(copy.parent))
    select_research_inputs(copies[0].path.parents[1], root, bundles[0], "manifest")
    select_research_inputs(copies[0].path.parents[1], root, bundles[0], "manifest")  # once
    bars(store, manifest, (DAYS[-1], NEW), 103.0, 1)  # a correction its writer publishes
    # bound after it: its marks are an admission's, but world now holds 103 for its 07-30 bar
    select_research_inputs(copies[1].path.parents[1], root, bundles[1], "manifest")
    uses = {"world": 0}
    relations = world_reads.relations

    def counted(
        cutoff_us: int = world_reads.LATEST,
        held_us: int | None = None,
        listings: Sequence[str] | None = None,
        *,
        workspace_id: str | None = None,
    ) -> dict[str, str]:
        uses["world"] += 1
        return relations(cutoff_us, held_us, listings, workspace_id=workspace_id)

    def read(copy: MarketDataRepository) -> tuple[object, ...]:
        before = uses["world"]
        values = (
            copy.raw_bars_by_listing(["listing-aapl", "listing-msft"]),
            copy.actions_by_listing(["listing-aapl"]),
            copy.provider_adjusted_closes("listing-aapl", through=NEW),
        )
        return values, uses["world"] - before

    monkeypatch.setattr(world_reads, "relations", counted)
    selected, unselected = (read(copy) for copy in copies)
    made = history.store().log.read("SELECT count(*) FROM selection")[0][0]
    assert (made, selected[1], unselected[1]) == (1, 3, 0)
    assert selected[0] == unselected[0]
    assert [bar.close for bar in selected[0][0]["listing-aapl"]] == [101.0] * 4


@pytest.mark.proves("market_data_ops.world_serves_the_workspace_market_reads")
def test_a_selection_before_a_first_use_reads_its_members_though_world_saw_them_before_bars(
    tmp_path: Path,
) -> None:
    """A selection of the session before a first use reads its manifest's members, though the
    first use's listings reached world before their bars."""
    from alphalattice.foundation.market_data_ops.sources.membership import UniverseBootstrapRecord

    now = datetime(2026, 8, 2, tzinfo=UTC)
    fixture = acquisition_manifest()
    manifest = replace(fixture, profile=replace(fixture.profile, manifest_as_of=now.date()))
    store = MarketDataRepository(tmp_path / "workspace", universe_reader="workspace")
    store.bootstrap(manifest)
    history.open_history("w1", "workspace", tmp_path / "workspace", root=tmp_path / "data")
    try:
        world.admit(store.workspace / "market-data.duckdb", "w1")
        bars(store, manifest, (*DAYS, NEW), 101.0, 0)
        listings = tuple(item.listing_id for item in manifest.listings)
        store.record_universe_bootstrap(
            UniverseBootstrapRecord(
                market_profile_id=manifest.profile.market_profile_id,
                t0_session=NEW,
                history_start=DAYS[0],
                cohort_listing_ids=listings,
                cohort_hash="",
                manifest_revision=manifest.revision_sha256,
                candidate_manifest_hash="1" * 64,
                qualification_policy_hash="2" * 64,
                feature_input_policy_hash="3" * 64,
                source_observed_at=now,
                admitted_at=now,
                panel_snapshot_hash="4" * 64,
                derivation="FIRST_QUALIFIED_PUBLICATION",
            )
        )
        world.admit(store.workspace / "market-data.duckdb", "w1")
        selection = datetime.combine(NEW + timedelta(days=1), datetime.min.time(), UTC)
        selected = MarketDataRepository(store.workspace, cutoff_us=world_reads.cutoff(selection))
        members = selected.load_universe_manifest_revision(manifest.revision_sha256).listings
    finally:
        history.release_history()
    assert members == manifest.listings


def _a_series_sent_again_unchanged_is_no_delta(
    store: MarketDataRepository, manifest: UniverseManifest
) -> None:
    """A provider sends its adjusted series again each day. The closes it sends unchanged are not
    the stage's delta: its publish records no rewrite of them in world. One changed close is the
    delta's one bar, a new version."""
    world.publish_pending(store.path, "w1")
    port = history.store()
    dividend = CorporateActionEvent("listing-aapl", "fixture", DAYS[2], "CASH_DIVIDEND", None, 0.5)
    count = "SELECT count(*) FROM lake.world.{} WHERE table_id = world_table_id('raw_bar')"

    def resend(day: int, last: float) -> tuple[int, int, int]:
        audit_one(
            store,
            manifest,
            listing_id="listing-aapl",
            provider="fixture",
            actions=(dividend,),
            adjusted_closes=(
                *(
                    ProviderAdjustedClosePoint("listing-aapl", "fixture", d, 100.5)
                    for d in DAYS[:-1]
                ),
                ProviderAdjustedClosePoint("listing-aapl", "fixture", DAYS[-1], last),
            ),
            history_start=DAYS[0],
            history_end=DAYS[-1],
            observed_at=NOW + timedelta(days=day),
        )
        with store._connect(read_only=True) as connection:
            journal = connection.execute("SELECT count(*) FROM market_delta").fetchone()[0]
        return (
            port.lake.execute(count.format("rewritten")).fetchone()[0],
            port.lake.execute(count.format("superseded")).fetchone()[0],
            journal,
        )

    assert resend(1, 100.5) == (0, 0, 0)
    assert resend(2, 100.75) == (0, 1, 0)


def _a_publish_takes_its_exact_keys_each_known_at_its_own_time(
    store: MarketDataRepository, manifest: UniverseManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage's publish copies only what its writers appended, for each write sequence the journal
    meets: a correction, the same correction sent again, its revert, and the correction repeated
    after its revert, of a bar and of an action, copy their changed rows and only the revisions
    their inserts added; corrections days apart leave the day between unread; and two reference
    writes before one publish keep each row at its own write's time, never a later write's or
    bar's, a new session at its own day's bars'. A workspace holding its bodies with no selection,
    as an upgraded one does, selects them all at its open and keeps none itself, its reads
    unchanged on any owner's connection. A Host's open with nothing pending opens the file
    read-only."""
    from alphalattice.foundation.market_data_ops.sources.manifest import ManifestListing

    bars_read = "SELECT * FROM raw_daily_bar_current ORDER BY ALL"
    with store._connect(read_only=True) as connection:
        before = connection.execute(bars_read).fetchall()
    world.publish_pending(store.path, "w1")
    held = market_snapshots.pending_table(store.path, "raw_daily_bar_current")
    with store._connect(read_only=True) as connection:
        assert market_snapshots.anchored(connection)
        assert connection.execute(f"SELECT count(*) FROM {held}").fetchone() == (0,)
        assert connection.execute(bars_read).fetchall() == before != []
    with store.database.read_transaction() as connection:  # another owner's door reads them too
        assert connection.execute(bars_read).fetchall() == before
    opened: list[bool] = []
    open_database = world.open_workspace_database

    def opening(path: Path, *, read_only: bool) -> Any:
        opened.append(read_only)
        return open_database(path, read_only=read_only)

    monkeypatch.setattr(world, "open_workspace_database", opening)
    assert world.publish_pending(store.path, "w1") is None  # the next open: nothing pending,
    assert opened == [True]  # found read-only, the file never opened writable
    read: dict[str, int] = {}
    port, staged = history.store(), {}
    publish = port.publish

    def staging(table: str, provider: str, batch: Any, lag: Any, **named: Any) -> Any:
        if not read:  # the delta the publish read, which its first batch is built from
            counts = " UNION ALL ".join(
                f"SELECT '{name}', count(*) FROM {world.WORKSPACE}.{name}"
                for name in (*world.DELTA, *world.REVISED)
            )
            read.update(port.lake.execute(counts).fetchall())
        staged.setdefault(table, []).extend(batch.to_pylist())
        return publish(table, provider, batch, lag, **named)

    monkeypatch.setattr(port, "publish", staging)

    def appended(table: str) -> int:
        with store._connect(read_only=True) as connection:
            return connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    dividend = CorporateActionEvent("listing-aapl", "fixture", DAYS[2], "CASH_DIVIDEND", None, 0.5)
    closes = tuple(ProviderAdjustedClosePoint("listing-aapl", "fixture", d, 100.5) for d in DAYS)

    def action(amount: float, offset: float) -> None:
        audit_one(
            store,
            manifest,
            listing_id="listing-aapl",
            provider="fixture",
            actions=(replace(dividend, cash_amount=amount),),
            adjusted_closes=closes,
            history_start=DAYS[0],
            history_end=DAYS[-1],
            observed_at=NOW + timedelta(days=offset),
        )

    bar, act = (
        ("raw_daily_bar_current", "bar_revision"),
        (
            "corporate_action_current",
            "corporate_action_revision",
        ),
    )
    sequences = (  # each write sequence: its tables, its write, the rows changed, revisions added
        ("a bar correction", bar, lambda: bars(store, manifest, DAYS[1:2], 102.0, 1.1), 2, 2),
        ("sent again", bar, lambda: bars(store, manifest, DAYS[1:2], 102.0, 1.2), 0, 0),
        ("its revert", bar, lambda: bars(store, manifest, DAYS[1:2], 101.0, 1.3), 2, 2),
        (
            "repeated after its revert",
            bar,
            lambda: bars(store, manifest, DAYS[1:2], 102.0, 1.4),
            2,
            0,
        ),
        ("an action correction", act, lambda: action(0.6, 1.5), 1, 1),
        ("sent again", act, lambda: action(0.6, 1.6), 0, 0),
        ("its revert", act, lambda: action(0.5, 1.7), 1, 1),
        ("repeated after its revert", act, lambda: action(0.6, 1.8), 1, 0),
    )
    for name, (current, revisions), write, changed, added in sequences:
        before = appended(revisions)
        read.clear()
        write()
        took = (read.get(current, 0), read.get(revisions, 0), appended(revisions) - before)
        assert (name, current, *took) == (name, current, changed, added, added)  # copied: appended
    read.clear()
    nvda = ManifestListing(
        listing_id="listing-nvda", symbol="NVDA", mic="XNAS", provider_symbol="NVDA"
    )
    added = replace(manifest, listings=(*manifest.listings, nvda), revision_sha256="b" * 64)
    later = NOW.date() + timedelta(days=1)  # a second write's mappings, effective a day later
    again = replace(
        added, profile=replace(added.profile, manifest_as_of=later), revision_sha256="c" * 64
    )
    with store.writing():
        first = datetime.now(UTC)
        store.bootstrap(added)
        second = datetime.now(UTC)
        store.bootstrap(again)
        third = datetime.now(UTC)
        bars(store, manifest, (NEW,), 104.0, 2)
        bars(store, manifest, (DAYS[0], DAYS[-1]), 103.0, 3)
        bars(store, added, DAYS[:1], 105.0, 10_000, ("NVDA",))
    assert read == {  # two days apart, NEW's session and NVDA's first bar; day 2 unread
        "raw_daily_bar_current": 7,
        "provider_adjusted_close_current": 2,  # AAPL's closes of its two corrected bars
        "corporate_action_current": 0,
        "bar_revision": 4,
        "corporate_action_revision": 0,
    }
    nvda_id, us = world.world_listing("listing-nvda"), world._us
    known = {
        (table, row.get("valid_from")): row[KNOWN]
        for table in ("listing", "provider_symbol")
        for row in staged[table]
        if row["listing_id"] == nvda_id
    }
    written = {key: us(first) <= at <= us(second) for key, at in known.items()}
    assert written == {  # the first write's rows at its time; the second's mapping at the second's
        **dict.fromkeys(known, True),
        ("provider_symbol", later.isoformat()): False,
    }
    assert us(second) <= known["provider_symbol", later.isoformat()] <= us(third)
    sessions = {row["session_date"]: row[KNOWN] for row in staged["session"]}
    assert {day: sessions.get(day.isoformat()) for day in (DAYS[0], DAYS[-1], NEW)} == {
        DAYS[0]: us(NOW + timedelta(days=3)),  # its earliest observation, AAPL's and MSFT's
        DAYS[-1]: us(NOW + timedelta(days=3)),
        NEW: us(NOW + timedelta(days=2)),  # its own bars', never the later corrections'
    }


@pytest.mark.proves("market_data_ops.a_sealed_index_hashes_its_values_never_its_file")
def test_a_sealed_index_hashes_the_values_it_holds_never_its_file(tmp_path: Path) -> None:
    """A sealed index's content hash reads one table, nulls and NaN among its values, to one hash
    each time, and the same values written under two writer settings to that hash; a null where
    a zero was moves it."""
    rows = [
        (
            i if i % 3 else None,
            i % 2 == 0 if i % 5 else None,
            date(2026, 1, 1 + i % 28) if i % 7 else None,
            (float("nan") if i % 11 == 0 else i / 7) if i % 4 else None,
            i if i % 6 else None,
            datetime(2026, 1, 1, i % 24) if i % 8 else None,
            f"0000000{i % 10}-0000-0000-0000-000000000000" if i % 9 else None,
            "x" * (i % 13) if i % 10 else None,
        )
        for i in range(3000)
    ]

    def written(name: str, options: str, setting: str | None, parts: list[list[tuple]]) -> Path:
        path = tmp_path / f"{name}.duckdb"
        with duckdb.connect() as writer:
            writer.execute(f"ATTACH '{path.as_posix()}' AS w {options}")
            if setting is not None:
                writer.execute(setting)
            writer.execute(
                "CREATE TABLE w.t (a BIGINT, b BOOLEAN, c DATE, d DOUBLE, e INTEGER,"
                " f TIMESTAMP, g UUID, h VARCHAR)"
            )
            for part in parts:
                writer.register(
                    "part", pa.table(list(zip(*part, strict=True)), names=list("abcdefgh"))
                )
                writer.execute("INSERT INTO w.t BY NAME SELECT * REPLACE (g::UUID AS g) FROM part")
                writer.unregister("part")
        return path

    def hashed(path: Path) -> str:
        with duckdb.connect() as reader:
            reader.execute(f"ATTACH '{path.as_posix()}' AS sealed (READ_ONLY)")
            return market_snapshots.content_hash(reader, "sealed")

    one = written("one", "", None, [rows])
    two = written(
        "two",
        "(STORAGE_VERSION 'v1.0.0')",
        "SET force_compression = 'uncompressed'",
        [rows[1500:][::-1], rows[:1500]],
    )
    zero = written("zero", "", None, [[(0, *rows[0][1:]), *rows[1:]]])

    assert one.read_bytes() != two.read_bytes()
    assert hashed(one) == hashed(one) == hashed(two)
    assert rows[0][0] is None and hashed(zero) != hashed(one)
