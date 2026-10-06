"""The forward membership journal: one cohort, dated events, stable earlier identities."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
    build_quality_filtered_research_manifest,
    default_qualification_policy_hash,
    history_quality_obligation,
    qualification_obligation,
    qualification_policy_identity,
)
from alphalattice.foundation.market_data_ops.sources.membership import (
    FORWARD_AS_OBSERVED,
    INITIAL_COHORT_BACKFILL,
    MembershipEvent,
    UniverseBootstrapRecord,
    UniverseSourceObservation,
    membership_effective_session,
    membership_events_for_transition,
    membership_identity,
    resolve_membership_schedule,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository

NOW = datetime(2026, 1, 8, 22, tzinfo=UTC)
SESSIONS = tuple(
    date(2025, 12, 29) + timedelta(days=offset)
    for offset in range(21)
    if (date(2025, 12, 29) + timedelta(days=offset)).weekday() < 5
    and (date(2025, 12, 29) + timedelta(days=offset)) != date(2026, 1, 1)
)
T0 = date(2026, 1, 5)


def _bootstrap(cohort: tuple[str, ...] = ("A", "B", "C")) -> UniverseBootstrapRecord:
    return UniverseBootstrapRecord(
        market_profile_id="us-current-index-research",
        t0_session=T0,
        history_start=SESSIONS[0],
        cohort_listing_ids=cohort,
        cohort_hash="",
        manifest_revision="1" * 64,
        candidate_manifest_hash="2" * 64,
        qualification_policy_hash="3" * 64,
        feature_input_policy_hash="4" * 64,
        source_observed_at=NOW - timedelta(days=3),
        admitted_at=NOW - timedelta(days=3),
        panel_snapshot_hash="5" * 64,
        derivation="FIRST_QUALIFIED_PUBLICATION",
    )


def _event(
    sequence: int, listing_id: str, kind: str, effective: date, *, reference: str = "e"
) -> MembershipEvent:
    return MembershipEvent(
        market_profile_id="us-current-index-research",
        sequence=sequence,
        listing_id=listing_id,
        kind=kind,  # type: ignore[arg-type]
        effective_session=effective,
        observed_at=datetime.combine(effective, datetime.min.time(), tzinfo=UTC),
        decided_at=datetime.combine(effective, datetime.min.time(), tzinfo=UTC),
        authority="test",
        reference_hash=reference * 64,
        manifest_revision="6" * 64,
    )


def test_journal_replay_keeps_earlier_intervals_and_their_identities() -> None:
    """requirement: entry, exit and re-entry never rewrite what came before them.

    D joins after T0 and is absent from the backfill and from the sessions
    before its entry; B leaves and keeps its membership before the exit; a
    re-entry is a new interval with the gap intact; the identity of every
    session before an event is the same with or without that event.
    """

    entry = SESSIONS[SESSIONS.index(T0) + 1]
    exit_session = SESSIONS[SESSIONS.index(T0) + 3]
    re_entry = SESSIONS[SESSIONS.index(T0) + 5]
    events = (
        _event(1, "D", "ENTRY", entry),
        _event(2, "B", "EXIT", exit_session),
        _event(3, "B", "ENTRY", re_entry),
    )
    schedule = resolve_membership_schedule(
        sessions=SESSIONS, bootstrap=_bootstrap(), events=events, fallback_listing_ids=()
    )
    before_t0 = [session for session in SESSIONS if session < T0]
    for session in before_t0:
        assert schedule.members(session) == ("A", "B", "C")
        assert schedule.basis(session) == INITIAL_COHORT_BACKFILL
    assert schedule.basis(T0) == FORWARD_AS_OBSERVED
    assert schedule.members(T0) == ("A", "B", "C")
    assert schedule.members(entry) == ("A", "B", "C", "D")
    assert schedule.members(exit_session) == ("A", "C", "D")
    assert schedule.members(re_entry) == ("A", "B", "C", "D")
    assert [epoch.listing_ids for epoch in schedule.epochs] == [
        ("A", "B", "C"),
        ("A", "B", "C", "D"),
        ("A", "C", "D"),
        ("A", "B", "C", "D"),
    ]
    assert schedule.union == ("A", "B", "C", "D")
    assert schedule.row_count() == sum(len(schedule.members(s)) for s in SESSIONS)

    # A future event changes no earlier session's computational identity.
    shorter = resolve_membership_schedule(
        sessions=SESSIONS, bootstrap=_bootstrap(), events=events[:1], fallback_listing_ids=()
    )
    for session in SESSIONS:
        if session < exit_session:
            assert schedule.epoch(session).membership_hash == shorter.epoch(session).membership_hash
    assert schedule.epochs[0].membership_hash == membership_identity(("A", "B", "C"))

    # Membership needs the bootstrap first; a workspace still building its
    # cohort holds the manifest under construction on every session.
    building = resolve_membership_schedule(
        sessions=SESSIONS, bootstrap=None, events=(), fallback_listing_ids=("C", "A")
    )
    assert building.epochs[0].listing_ids == ("A", "C")
    assert {item.basis for item in building.basis_ranges} == {INITIAL_COHORT_BACKFILL}
    with pytest.raises(ValueError, match="before the bootstrap"):
        resolve_membership_schedule(
            sessions=SESSIONS,
            bootstrap=_bootstrap(),
            events=(_event(1, "D", "ENTRY", SESSIONS[0]),),
            fallback_listing_ids=(),
        )
    with pytest.raises(ValueError, match="already present"):
        resolve_membership_schedule(
            sessions=SESSIONS,
            bootstrap=_bootstrap(),
            events=(_event(1, "A", "ENTRY", entry),),
            fallback_listing_ids=(),
        )


def test_transition_events_are_the_exits_then_entries_between_two_sets() -> None:
    events = membership_events_for_transition(
        market_profile_id="p",
        current_listing_ids=("A", "B", "C"),
        next_listing_ids=("A", "C", "D", "E"),
        effective_session=T0,
        observed_at=NOW,
        decided_at=NOW,
        authority="feature_input_gateway",
        reference_hash="7" * 64,
        manifest_revision="8" * 64,
        first_sequence=4,
    )
    assert [(item.sequence, item.listing_id, item.kind) for item in events] == [
        (4, "B", "EXIT"),
        (5, "D", "ENTRY"),
        (6, "E", "ENTRY"),
    ]


def test_store_appends_only_actual_changes_and_refuses_gaps_and_rewrites(
    tmp_path: Path,
) -> None:
    """The journal is incremental: the cohort once, then events in sequence."""

    market = MarketDataRepository(tmp_path / "workspace")
    market.bootstrap(_manifest(("A", "B", "C")))
    assert market.universe_bootstrap("us-current-index-research") is None
    with pytest.raises(ValueError, match="require a recorded universe bootstrap"):
        market.append_membership_events((_event(1, "D", "ENTRY", T0),))
    record = _bootstrap()
    market.record_universe_bootstrap(record)
    market.record_universe_bootstrap(record)  # idempotent
    with pytest.raises(ValueError, match="already recorded"):
        market.record_universe_bootstrap(_bootstrap(cohort=("A", "B")))
    stored = market.universe_bootstrap("us-current-index-research")
    assert stored == record
    entry = SESSIONS[SESSIONS.index(T0) + 1]
    market.append_membership_events((_event(1, "D", "ENTRY", entry),))
    with pytest.raises(ValueError, match="not the journal's next"):
        market.append_membership_events((_event(3, "B", "EXIT", entry),))
    with pytest.raises(ValueError, match="before the bootstrap"):
        market.append_membership_events((_event(2, "B", "EXIT", SESSIONS[0]),))
    market.append_membership_events((_event(2, "B", "EXIT", entry),))
    assert [
        (item.sequence, item.listing_id, item.kind)
        for item in market.membership_events("us-current-index-research")
    ] == [(1, "D", "ENTRY"), (2, "B", "EXIT")]
    schedule = market.membership_schedule(
        "us-current-index-research", sessions=SESSIONS, fallback_listing_ids=()
    )
    assert schedule.members(SESSIONS[0]) == ("A", "B", "C")
    assert schedule.members(entry) == ("A", "C", "D")
    assert schedule.journal_sequence == 2


@pytest.mark.parametrize(
    ("observed", "decided", "expected"),
    [
        ("2026-09-04T19:00:00+00:00", "2026-09-05T12:00:00+00:00", "2026-09-04"),
        ("2026-09-04T21:00:00+00:00", "2026-09-04T22:00:00+00:00", "2026-09-08"),
        ("2026-11-27T17:59:00+00:00", "2026-11-27T20:00:00+00:00", "2026-11-27"),
        ("2026-11-27T18:01:00+00:00", "2026-11-27T20:00:00+00:00", "2026-11-30"),
    ],
)
def test_membership_clock_uses_observation_close_and_decision_deadline(
    observed: str, decided: str, expected: str
) -> None:
    assert membership_effective_session(
        observed_at=datetime.fromisoformat(observed),
        decided_at=datetime.fromisoformat(decided),
        not_before=date(2026, 1, 1),  # A historical catch-up target grants no earlier knowledge.
    ) == date.fromisoformat(expected)


def test_journal_refuses_a_current_observation_backdated_inside_an_unobserved_gap(
    tmp_path: Path,
) -> None:
    market = MarketDataRepository(tmp_path / "workspace")
    market.bootstrap(_manifest(("A", "B", "C")))
    market.record_universe_bootstrap(_bootstrap())
    early = replace(
        _event(1, "D", "ENTRY", date(2026, 1, 6)),
        observed_at=datetime.fromisoformat("2026-01-08T17:00:00-05:00"),
        decided_at=datetime.fromisoformat("2026-01-08T17:00:00-05:00"),
        event_hash="",
    )
    with pytest.raises(ValueError, match="effective_session_precedes_knowledge"):
        market.append_membership_events((early,))
    assert market.membership_events(early.market_profile_id) == ()
    admitted = replace(early, effective_session=date(2026, 1, 9), event_hash="")
    market.append_membership_events((admitted,))
    assert market.membership_events(early.market_profile_id) == (admitted,)
    assert admitted.observed_at == NOW
    history = market.membership_schedule(
        early.market_profile_id, sessions=SESSIONS, fallback_listing_ids=()
    )
    assert history.members(date(2026, 1, 6)) == ("A", "B", "C")
    assert history.members(date(2026, 1, 9)) == ("A", "B", "C", "D")


def test_unchanged_source_checks_record_the_gap_without_membership_copies(tmp_path: Path) -> None:
    market = MarketDataRepository(tmp_path / "workspace")
    market.bootstrap(_manifest(("A", "B", "C")))
    first = UniverseSourceObservation(
        market_profile_id="us-current-index-research",
        observed_at=NOW,
        candidate_membership_hash="a" * 64,
        source_identity_hash="b" * 64,
        first_eligible_session=date(2026, 1, 9),
    )
    second = UniverseSourceObservation(
        market_profile_id=first.market_profile_id,
        observed_at=NOW + timedelta(days=7),
        previous_observed_at=NOW,
        candidate_membership_hash=first.candidate_membership_hash,
        source_identity_hash="c" * 64,
        first_eligible_session=date(2026, 1, 16),
    )
    market.record_universe_source_observation(first)
    market.record_universe_source_observation(second)
    market.record_universe_source_observation(second)
    assert market.universe_source_observations(first.market_profile_id) == (second, first)
    assert market.membership_events(first.market_profile_id) == ()
    with market._connect() as connection:
        connection.execute(
            "UPDATE universe_source_observation SET source_identity_hash = ? "
            "WHERE observation_hash = ?",
            ["d" * 64, second.observation_hash],
        )
    with pytest.raises(ValueError, match="observation_identity_mismatch"):
        market.universe_source_observations(first.market_profile_id)
    with pytest.raises(ValueError, match="observation_conflict"):
        market.record_universe_source_observation(second)


def test_historical_listing_coverage_does_not_expand_the_current_manifest(tmp_path: Path) -> None:
    market = MarketDataRepository(tmp_path / "workspace")
    parent = _manifest(("A", "B", "C"))
    market.bootstrap(parent)
    current = build_quality_filtered_research_manifest(parent, eligible_listing_ids=("A", "C"))
    market.bootstrap(current)
    assert tuple(
        item.listing_id for item in market.listing_scope(current, listing_ids=("A", "B", "C"))
    ) == ("A", "B", "C")
    assert tuple(
        item.listing_id for item in market.load_universe_manifest(current.manifest_id).listings
    ) == ("A", "C")
    original = market.execution_source_watermark(current, through=NOW.date())
    assert (
        market.execution_source_watermark(current, through=NOW.date(), listing_ids=("A", "C"))
        == original
    )
    history = market.execution_source_watermark(
        current, through=NOW.date(), listing_ids=("A", "B", "C")
    )
    assert history["listing_count"] == 3
    assert history["watermark_hash"] != original["watermark_hash"]
    with pytest.raises(ValueError, match="covered_listing_identity_unavailable"):
        market.listing_scope(current, listing_ids=("A", "UNKNOWN"))
    # No journal means legacy current-only coverage. After bootstrap, exiting
    # B keeps its source but never adds it back to the current roster.
    assert market.research_listing_sources(current) == dict.fromkeys(
        ("A", "C"), current.revision_sha256
    )
    market.record_universe_bootstrap(
        replace(_bootstrap(), manifest_revision=parent.revision_sha256, record_hash="")
    )
    market.append_membership_events((_event(1, "B", "EXIT", SESSIONS[-2]),))
    sources = market.research_listing_sources(current)
    assert sources == {
        "A": current.revision_sha256,
        "B": parent.revision_sha256,
        "C": current.revision_sha256,
    }
    old = market.maintenance_revision_fingerprint(current)
    assert old == market.maintenance_revision_fingerprint(current, listing_ids=("A", "C"))
    covered_before = market.maintenance_revision_fingerprint(current, listing_ids=tuple(sources))
    # A newly acquired valuation bar outside today's roster is still an input
    # revision for a historical/holding consumer, without touching membership.
    with market._connect() as connection:
        connection.execute(
            "INSERT INTO raw_daily_bar_current VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "B",
                current.profile.provider,
                SESSIONS[-1],
                100.0,
                101.0,
                99.0,
                100.0,
                1000.0,
                "a" * 64,
                NOW.replace(tzinfo=None),
            ],
        )
    assert market.maintenance_revision_fingerprint(current) == old
    assert (
        market.maintenance_revision_fingerprint(current, listing_ids=tuple(sources))
        != covered_before
    )
    assert tuple(v.listing_id for v in current.listings) == ("A", "C")


def test_qualification_obligations_have_one_identity_and_children_keep_theirs() -> None:
    """One spelling for every derivation owner, idempotent under re-satisfaction."""

    parent = build_quality_filtered_research_manifest(
        _manifest(("A", "B", "C")),
        eligible_listing_ids=("A", "B", "C"),
        qualification_obligations=(history_quality_obligation(),),
    )
    gateway = qualification_obligation("feature_input_policy", "9" * 64)
    sector = qualification_obligation("sector", "YAHOO_CURRENT_SECTOR:require_nonempty")
    child = build_quality_filtered_research_manifest(
        parent, eligible_listing_ids=("A", "B"), qualification_obligations=(gateway,)
    )
    again = build_quality_filtered_research_manifest(
        child, eligible_listing_ids=("A", "B"), qualification_obligations=(gateway,)
    )
    with_sector = build_quality_filtered_research_manifest(
        child, eligible_listing_ids=("A", "B"), qualification_obligations=(sector,)
    )
    assert child.qualification_obligations == tuple(sorted((history_quality_obligation(), gateway)))
    assert child.qualification_policy_hash == qualification_policy_identity(
        child.qualification_obligations
    )
    # Re-satisfying the same obligation over the same listings is the same manifest.
    assert again.revision_sha256 == child.revision_sha256
    # A new obligation is a new identity that still carries the old ones.
    assert set(with_sector.qualification_obligations) == {
        history_quality_obligation(),
        gateway,
        sector,
    }
    assert with_sector.revision_sha256 != child.revision_sha256
    # The effective membership identity is the child's own, not the candidate set's.
    assert child.membership_fingerprint == parent.membership_fingerprint
    assert child.effective_membership_hash != parent.effective_membership_hash
    assert child.effective_membership_hash == membership_identity(("A", "B"))
    # A manifest recorded before obligations were named derives through its hash.
    legacy = build_quality_filtered_research_manifest(
        _manifest(("A", "B", "C")),
        eligible_listing_ids=("A", "B", "C"),
        qualification_policy_hash="c" * 64,
    )
    assert legacy.qualification_obligations == ()
    derived = build_quality_filtered_research_manifest(
        legacy, eligible_listing_ids=("A",), qualification_obligations=(gateway,)
    )
    assert derived.qualification_obligations == tuple(
        sorted((qualification_obligation("legacy_policy", "c" * 64), gateway))
    )


def _manifest(symbols: tuple[str, ...]) -> UniverseManifest:
    profile = MarketProfile(
        market_profile_id="us-current-index-research",
        display_name="US current index research",
        market="US",
        currency="USD",
        calendar_id="XNAS+XNYS",
        provider="fixture",
        daily_price_basis="split_adjusted",
        manifest_as_of=NOW.date(),
        data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
    )
    return UniverseManifest(
        manifest_id="us-current-index-research:acquisition:fixture",
        profile=profile,
        listings=tuple(
            ManifestListing(listing_id=symbol, symbol=symbol, mic="XNAS", provider_symbol=symbol)
            for symbol in symbols
        ),
        revision_sha256="a" * 64,
        membership_fingerprint="b" * 64,
        qualification_policy_hash=default_qualification_policy_hash(),
    )


def test_a_workspace_from_before_this_stage_reads_its_manifest_without_a_writer(
    tmp_path: Path,
) -> None:
    """requirement: an existing workspace opens under the new code before any upgrade.

    The obligations column and the journal tables are created by the store's
    writer at the next manifest bootstrap. The page reads manifests, the
    bootstrap record and the journal on read-only connections first, so every
    one of those readers must answer over the older schema: no obligations,
    no bootstrap, no events.
    """

    market = MarketDataRepository(tmp_path / "workspace")
    manifest = _manifest(("A", "B", "C"))
    market.bootstrap(manifest)
    connection = market._connect()
    try:
        connection.execute(
            "ALTER TABLE universe_manifest DROP COLUMN qualification_obligations_json"
        )
        connection.execute("DROP TABLE universe_bootstrap")
        connection.execute("DROP TABLE universe_membership_event")
    finally:
        connection.close()
    restored = market.load_universe_manifest_revision(manifest.revision_sha256)
    assert restored.revision_sha256 == manifest.revision_sha256
    assert restored.qualification_obligations == ()
    assert market.universe_bootstrap("us-current-index-research") is None
    assert market.membership_events("us-current-index-research") == ()
    # The next writer upgrade restores the column and the tables, and a
    # manifest bound afterwards carries its obligations again.
    child = build_quality_filtered_research_manifest(
        manifest,
        eligible_listing_ids=("A", "B"),
        qualification_obligations=(history_quality_obligation(),),
    )
    market.bootstrap(child)
    assert market.load_universe_manifest_revision(
        child.revision_sha256
    ).qualification_obligations == (history_quality_obligation(),)


def test_a_manifest_read_proves_the_members_it_assembled(tmp_path: Path) -> None:
    """regression (V269, EV2): a manifest whose membership row was removed read back, by its
    old revision, with fewer members. The read proves its members against the digest sealed at
    registration, and a manifest written before the digest against the member count it stored."""

    market = MarketDataRepository(tmp_path / "workspace")
    manifest = _manifest(("A", "B", "C"))
    market.bootstrap(manifest)
    assert len(market.load_universe_manifest(manifest.manifest_id).listings) == 3

    def execute(statement: str, *parameters: object) -> None:
        connection = market._connect()
        try:
            connection.execute(statement, list(parameters))
        finally:
            connection.close()

    removed = "DELETE FROM universe_manifest_listing WHERE manifest_id = ? AND listing_id = 'B'"
    restored = "INSERT INTO universe_manifest_listing VALUES (?, 'B')"
    execute(removed, manifest.manifest_id)
    with pytest.raises(ValueError, match="universe_manifest_membership_changed"):
        market.load_universe_manifest(manifest.manifest_id)
    with pytest.raises(ValueError, match="universe_manifest_membership_changed"):
        market.load_universe_manifest_revision(manifest.revision_sha256)
    execute(restored, manifest.manifest_id)
    assert len(market.load_universe_manifest(manifest.manifest_id).listings) == 3
    execute("UPDATE universe_manifest SET membership_sha256 = NULL")
    execute(removed, manifest.manifest_id)
    with pytest.raises(ValueError, match="universe_manifest_membership_changed"):
        market.load_universe_manifest(manifest.manifest_id)
    execute(restored, manifest.manifest_id)
    assert len(market.load_universe_manifest(manifest.manifest_id).listings) == 3
