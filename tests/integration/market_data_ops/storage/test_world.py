"""A workspace's market tables in world: admitted at its open, equal to its tables, checked by
marks alone while unchanged, and a later correction known only from its admission."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from alphalattice.control.product_host.research_authoring.factor_inputs import (
    FactorInputBundle,
    file_digest,
    select_research_inputs,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
from alphalattice.foundation.market_data_ops.storage import world, world_reads
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.project import definition_id
from tests.workspace_maintenance.acquisition_manifest import NOW, acquisition_manifest
from tests.workspace_maintenance.audit_workspace import audit_one

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/storage/world.py"
DAYS = (date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30))
NEW = date(2026, 7, 31)
SYMBOLS = ("AAPL", "MSFT")
CURRENT = """(SELECT * FROM lake.world.{0} x WHERE NOT x.retracted AND NOT EXISTS (SELECT 1
    FROM lake.world.superseded s WHERE s.version_id = x.version_id))"""
FIXTURE, OLD_ID, DISPLAY = (
    definition_id("provider", name)
    for name in ("fixture", world.LISTING_ALIAS, world.DISPLAY_ALIAS)
)
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


@pytest.mark.proves("market_data_ops.world_holds_the_workspace_market_tables")
@pytest.mark.parametrize("focus", ("identity", "workspace", "lineage"))
def test_a_workspace_open_admits_its_market_tables_once_and_a_correction_from_its_admission(
    workspace: tuple[MarketDataRepository, UniverseManifest, Path],
    monkeypatch: pytest.MonkeyPatch,
    focus: str,
    tmp_path: Path,
) -> None:
    """A workspace open admits its market tables into world once, equal row for row, and a
    later correction is seen only from its admission."""
    store, manifest, database = workspace
    first = world.admit(database, "w1")
    again = world.admit(database, "w1")
    lake = history.store().lake
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
        assert world.admit(other.path, "w2") is None
        return
    set_id = world.admit_member_set(("listing-aapl", "listing-msft"), DAYS[1], "fixture")
    assert world.admit_member_set(("listing-msft", "listing-aapl"), DAYS[1], "fixture") == set_id
    close = int(datetime(2026, 7, 28, 20, tzinfo=UTC).timestamp() * 1_000_000)
    members = (
        "SELECT listing_id::VARCHAR, known_at FROM universe_member_asof(?)"
        " WHERE universe_id = ? ORDER BY 1"
    )
    assert lake.execute(members, (close - 1, set_id)).fetchall() == []
    assert lake.execute(members, (close, set_id)).fetchall() == [
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
    exit_close = int(datetime(2026, 7, 29, 20, tzinfo=UTC).timestamp() * 1_000_000)
    interval = (
        "SELECT valid_to FROM universe_member_asof(?) WHERE universe_id = ? AND listing_id = ?"
    )
    assert lake.execute(
        interval, (exit_close - 1, live, world.world_listing("listing-msft"))
    ).fetchone() == (None,)
    assert lake.execute(
        interval, (exit_close, live, world.world_listing("listing-msft"))
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
    assert world.admit(database, "w1") is None
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
    assert callable(getattr(world, "admit_sectors", None))
    world.admit_sectors(database, "w1")
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
    monkeypatch.setenv("ALPHALATTICE_MARKET_READS", "world")
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
    monkeypatch.setenv("ALPHALATTICE_MARKET_READS", "world")
    assert feature.sector_classifications(("listing-aapl",)) == {"listing-aapl": "Health Care"}
    assert seen_world
    monkeypatch.setenv("ALPHALATTICE_MARKET_READS", "workspace")
    seen_world.clear()
    assert feature.sector_classifications(("listing-aapl",)) == {"listing-aapl": "Health Care"}
    assert not seen_world
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
    monkeypatch.setenv("ALPHALATTICE_MARKET_READS", "world")
    seen_world.clear()
    if focus == "workspace":
        assert feature.sector_classifications(("listing-aapl",)) == {"listing-aapl": "Health Care"}
        assert (
            feature.current_sector_state(manifest).sector_by_listing_id["listing-aapl"]
            == "Health Care"
        )
        assert (
            source.read_sector_rows(listing_ids=("listing-aapl",))[0]["sector_name"]
            == "Health Care"
        )
        with world_reads.reading(
            database,
            lambda: world.sector_marks(database),
            dataset="sector",
            admit=world.admit_sectors,
        ) as served:
            assert served is None  # A's older marks cannot authorize B's later classification.
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
    monkeypatch.setenv("ALPHALATTICE_MARKET_READS", "world")
    seen_world.clear()
    assert feature.sector_classifications(("listing-aapl",)) == {"listing-aapl": "Utilities"}
    assert not seen_world  # the marks of an equal-time drop cannot make world serve the workspace
    if focus == "identity":
        # Later world aliases cannot alter the old manifest's frozen listing facts.
        for provider, symbol in ((world.DISPLAY_ALIAS, "AAPL-NEW"), ("fixture", "AAPL.NEW")):
            with world.stage(
                port.root,
                "provider_symbol",
                definition_id("provider", provider),
                world._arrow([{"listing_id": aapl, "symbol": symbol, "valid_from": DAYS[3]}]),
            ) as staged:
                port.commit(staged)
        assert (
            world.load_universe_manifest(
                manifest.manifest_id, world_reads.LATEST, workspace_id="w1"
            )
            == manifest
        )


@pytest.mark.proves("market_data_ops.world_serves_the_workspace_market_reads")
def test_world_serves_the_reads_while_it_holds_the_tables_and_the_host_admits_a_write_first(
    workspace: tuple[MarketDataRepository, UniverseManifest, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reads are equal under both settings: world serves them while it holds the tables, the Host
    admitting a write first and an update once, and the workspace answers an unadmitted write."""
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

    def read(setting: str) -> tuple[object, ...]:
        monkeypatch.setenv(world_reads.SETTING, setting)
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
    host = [read("workspace"), read("world")]  # the Host admits the unadmitted tables first
    assert store.raw_bars("listing-aapl", cutoff_us=0) == ()
    receipt = store.latest_action_audit_receipt(listing_id="listing-aapl", provider="fixture")
    checked = []
    for setting in ("workspace", "world", "workspace"):
        monkeypatch.setenv(world_reads.SETTING, setting)
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
    assert [calls for _value, calls in checked][::2] == [0, 0]
    assert checked[1][1] > 0
    bars(store, manifest, (DAYS[-1], NEW), 103.0, 1)
    written = [read("workspace"), read("world")]
    before = admissions()
    # an update writing and reading listing by listing, its commits in a retained instance's
    # WAL; the held file defers the admission to the first read after it closes
    with store.database.retain(read_only=False):
        with store.writing():
            bars(store, manifest, (NEW,), 104.0, 2)
            inside = read("world")
            bars(store, manifest, (NEW,), 105.0, 3)
        held = (admissions() - before, read("world"))
    after = read("world")  # the first read after the file closes admits
    updated = (admissions() - before, after)
    history.release_history()
    monkeypatch.setenv(history.DATA_ROOT_ENV, str(database.parent.parent / "data"))
    child = [read("workspace"), read("world")]
    bars(store, manifest, (NEW,), 106.0, 4)  # a write no Host has admitted
    unseen = [read("workspace"), read("world")]
    assert [(own[0] == seen[0], own[1], seen[1]) for own, seen in (host, written, child)] == [
        (True, 0, 4)
    ] * 3
    assert written[1][0][0]["listing-aapl"][-1].close == 103.0
    assert (inside[0][0]["listing-aapl"][-1].close, inside[1]) == (104.0, 0)
    assert (held[0], held[1][0][0]["listing-aapl"][-1].close, held[1][1]) == (0, 105.0, 0)
    assert (updated[0], updated[1][0][0]["listing-aapl"][-1].close, updated[1][1]) == (1, 105.0, 4)
    assert (unseen[0][0] == unseen[1][0], unseen[1][1]) == (True, 0)


@pytest.mark.proves("market_data_ops.a_research_copy_reads_world_at_its_selection")
def test_a_research_copy_reads_world_as_of_its_selection_and_its_own_tables_without_one(
    workspace: tuple[MarketDataRepository, UniverseManifest, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy bound in the Host reads world as of its selection, equal to its own tables and
    unmoved by a later correction; a copy bound once world holds another value of one of its keys
    gets no selection, and it and every copy under the workspace setting read their own tables."""
    store, manifest, database = workspace
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
    bars(store, manifest, (DAYS[-1], NEW), 103.0, 1)  # a correction world admits later
    world.admit(database, "w1")
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

    def read(copy: MarketDataRepository, setting: str) -> tuple[object, ...]:
        monkeypatch.setenv(world_reads.SETTING, setting)
        before = uses["world"]
        values = (
            copy.raw_bars_by_listing(["listing-aapl", "listing-msft"]),
            copy.actions_by_listing(["listing-aapl"]),
            copy.provider_adjusted_closes("listing-aapl", through=NEW),
        )
        return values, uses["world"] - before

    monkeypatch.setattr(world_reads, "relations", counted)
    selected, unselected = (read(copy, "world") for copy in copies)
    own = read(copies[0], "workspace")
    made = history.store().log.read("SELECT count(*) FROM selection")[0][0]
    assert (made, selected[1], unselected[1], own[1]) == (1, 3, 0, 0)
    assert selected[0] == own[0] == unselected[0]
    assert [bar.close for bar in selected[0][0]["listing-aapl"]] == [101.0] * 4
