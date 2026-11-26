"""Entry, exit, governance-only and recovery of a Panel whose sessions hold their own members.

The Feature service over the Universe journal, end to end: the bootstrap
cohort, a member joining at one session, a member leaving at a later one, a
manifest that changed for governance alone, exact reuse of a rebuild, and
artifact-only recovery of the ragged snapshot.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow.parquet as pq

from alphalattice.foundation.feature_engine.contracts import (
    FeatureInvalidation,
    PanelCrossSectionRange,
)
from alphalattice.foundation.feature_engine.panels.closure import PanelClosurePublisher
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_source import (
    PanelClosureSourceRepository,
)
from alphalattice.foundation.feature_engine.panels.rematerialization import (
    ArtifactOnlyPanelRematerializer,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    build_quality_filtered_research_manifest,
    qualification_obligation,
)
from alphalattice.foundation.market_data_ops.sources.membership import (
    FORWARD_AS_OBSERVED,
    INITIAL_COHORT_BACKFILL,
    MembershipEvent,
    UniverseBootstrapRecord,
)
from tests.feature_engine.panel_workspace import (
    NOW,
    SESSIONS,
    build_and_publish,
    fixture_manifest,
    new_session_invalidations,
    panel_workspace,
)

PROFILE = "fixture-feature-market"


def _rows_by_session(resolver, manifest_payload, listing_id: str) -> set[str]:
    sessions: set[str] = set()
    for chunk in manifest_payload["chunks"]:
        path = resolver.resolve_feature_panel_chunk_ref(
            uri=str(chunk["uri"]),
            content_hash=str(chunk["chunk_hash"]),
            metadata_hash=str(chunk["metadata_hash"]),
        )
        table = pq.read_table(path, columns=["session_date", "listing_id"])
        for session, listing in zip(
            table.column("session_date").to_pylist(),
            table.column("listing_id").to_pylist(),
            strict=True,
        ):
            if str(listing) == listing_id:
                sessions.add(session.isoformat())
    return sessions


def _event(sequence: int, listing_id: str, kind: str, effective, manifest_revision: str):
    return MembershipEvent(
        market_profile_id=PROFILE,
        sequence=sequence,
        listing_id=listing_id,
        kind=kind,  # type: ignore[arg-type]
        effective_session=effective,
        observed_at=datetime.combine(effective, datetime.min.time(), tzinfo=UTC),
        decided_at=datetime.combine(effective, datetime.min.time(), tzinfo=UTC),
        authority="feature_input_gateway",
        reference_hash="5" * 64,
        manifest_revision=manifest_revision,
    )


def test_members_join_and_leave_from_their_effective_session_and_the_rest_is_reused(
    tmp_path,
) -> None:
    """Members join and leave from their effective session and the rest is reused."""

    full = fixture_manifest()
    fourteen = build_quality_filtered_research_manifest(
        full, eligible_listing_ids=tuple(item.listing_id for item in full.listings[:14])
    )
    fifteen = build_quality_filtered_research_manifest(
        full, eligible_listing_ids=tuple(item.listing_id for item in full.listings)
    )
    entrant = full.listings[14].listing_id
    leaver = full.listings[0].listing_id
    # The helper's dense business-day axis ends on Presidents' Day; member
    # decisions use actual exchange sessions, unlike that numerical fixture.
    t0, t1, t2 = SESSIONS[-4], SESSIONS[-3], SESSIONS[-2]
    workspace = panel_workspace(tmp_path / "journey", full, end=t2, spy="a" * 64)
    # Two sectors of eight and seven: every membership below keeps both at or
    # above the five-member floor, so the Panel stays admissible throughout.
    workspace.provider.sectors = {
        item.symbol: "Sector-0" if index < 8 else "Sector-1"
        for index, item in enumerate(full.listings)
    }
    workspace.provider.sectors["SPY"] = "Reference"
    workspace.market_data.bootstrap(fourteen)
    workspace.market_data.bootstrap(fifteen)
    resolver = workspace.resolver

    # Day one: the cohort, then the bootstrap boundary the coordinator records
    # at the first qualified publication.
    workspace.service.manifest = fourteen
    _one, day_one = build_and_publish(
        workspace, fourteen, spy="a" * 64, as_of=t0, observed_at=NOW, refresh_sector=True
    )
    membership_one = day_one["safe_summary"]["membership"]
    assert membership_one["bootstrap_record_hash"] is None
    assert [item["basis"] for item in membership_one["basis_ranges"]] == [INITIAL_COHORT_BACKFILL]
    assert day_one["active_listing_count"] == 14
    workspace.market_data.record_universe_bootstrap(
        UniverseBootstrapRecord(
            market_profile_id=PROFILE,
            t0_session=t0,
            history_start=SESSIONS[0],
            cohort_listing_ids=tuple(item.listing_id for item in fourteen.listings),
            cohort_hash="",
            manifest_revision=fourteen.revision_sha256,
            candidate_manifest_hash=fourteen.membership_fingerprint or "1" * 64,
            qualification_policy_hash=fourteen.qualification_policy_hash or "2" * 64,
            feature_input_policy_hash="3" * 64,
            source_observed_at=NOW,
            admitted_at=NOW,
            panel_snapshot_hash=str(day_one["snapshot_hash"]),
            derivation="FIRST_QUALIFIED_PUBLICATION",
        )
    )

    # Day two: the fifteenth name enters at T1.
    workspace.market_data.append_membership_events(
        (_event(1, entrant, "ENTRY", t1, fifteen.revision_sha256),)
    )
    workspace.service.manifest = fifteen
    pending = workspace.service._panel_membership(
        calendar=tuple(session for session in SESSIONS if session <= t0),
        axis_listing_ids=tuple(item.listing_id for item in full.listings),
    )
    assert entrant not in pending.listing_ids  # Prepared history is not yet a Panel member.
    _two, day_two = build_and_publish(
        workspace,
        fifteen,
        spy="a" * 64,
        as_of=t1,
        observed_at=NOW + timedelta(days=1),
        invalidations=(
            *new_session_invalidations(fifteen, t1),
            FeatureInvalidation("manifest_addition", listing_id=entrant, earliest_session=t1),
        ),
        refresh_sector=True,
    )
    years = [int(chunk["year"]) for chunk in day_two["chunks"]]
    assert day_two["safe_summary"]["partition_reuse"] == {
        "reused_partition_count": len(years) - 1,
        "composed_partition_count": 1,
    }
    assert day_two["chunks"][:-1] == day_one["chunks"][:-1]
    assert day_two["active_listing_count"] == 15
    membership_two = day_two["safe_summary"]["membership"]
    assert membership_two["bootstrap_t0_session"] == t0.isoformat()
    assert [item["basis"] for item in membership_two["basis_ranges"]] == [
        INITIAL_COHORT_BACKFILL,
        FORWARD_AS_OBSERVED,
    ]
    assert [item["member_count"] for item in membership_two["epochs"]] == [14, 15]
    assert membership_two["epochs"][1]["first_session"] == t1.isoformat()
    assert membership_two["as_of_member_count"] == 15
    assert _rows_by_session(resolver, day_two, entrant) == {t1.isoformat()}
    # The entrant's base Formula values cover its history, not only T1.
    base_rows = workspace.feature_state.feature_rows(
        listing_ids=(entrant,),
        catalog_hash=workspace.service.catalog.binding.catalog_hash,
        start=SESSIONS[0],
        end=t1,
        include_lineage=False,
    )
    assert len(base_rows) > 100
    # Availability counts members, and a finite value for a non-member is not a row.
    availability = workspace.panel_state.panel_availability_rows(
        cross_sections=tuple(
            PanelCrossSectionRange.from_payload(item)
            for chunk in day_two["chunks"]
            for item in chunk["cross_sections"]
        ),
        catalog_hash=workspace.service.catalog.binding.catalog_hash,
        policy_hash=workspace.service.panel.policy_hash,
        start=t0,
        end=t1,
    )
    sizes = {str(item["session_date"]): int(item["universe_size"]) for item in availability}
    assert sizes == {t0.isoformat(): 14, t1.isoformat(): 15}

    # Day three: the first name leaves at T2 and keeps every earlier row.
    fourteen_again = build_quality_filtered_research_manifest(
        fifteen,
        eligible_listing_ids=tuple(
            item.listing_id for item in fifteen.listings if item.listing_id != leaver
        ),
    )
    workspace.market_data.bootstrap(fourteen_again)
    workspace.market_data.append_membership_events(
        (_event(2, leaver, "EXIT", t2, fourteen_again.revision_sha256),)
    )
    workspace.service.manifest = fourteen_again
    _three, day_three = build_and_publish(
        workspace,
        fourteen_again,
        spy="a" * 64,
        as_of=t2,
        observed_at=NOW + timedelta(days=2),
        invalidations=(
            *new_session_invalidations(fourteen_again, t2),
            FeatureInvalidation("manifest_removal", listing_id=leaver, earliest_session=t2),
        ),
        refresh_sector=False,
    )
    assert day_three["chunks"][:-1] == day_two["chunks"][:-1]
    assert day_three["active_listing_count"] == 15  # the axis still holds the leaver
    assert day_three["safe_summary"]["membership"]["as_of_member_count"] == 14
    assert [item["member_count"] for item in day_three["safe_summary"]["membership"]["epochs"]] == [
        14,
        15,
        14,
    ]
    leaver_sessions = _rows_by_session(resolver, day_three, leaver)
    assert t2.isoformat() not in leaver_sessions
    assert {t0.isoformat(), t1.isoformat()} <= leaver_sessions

    # Governance only: the same fourteen under a new revision reuse everything.
    regoverned = build_quality_filtered_research_manifest(
        fourteen_again,
        eligible_listing_ids=tuple(item.listing_id for item in fourteen_again.listings),
        qualification_obligations=(qualification_obligation("review", "6" * 64),),
    )
    assert regoverned.revision_sha256 != fourteen_again.revision_sha256
    workspace.market_data.bootstrap(regoverned)
    workspace.service.manifest = regoverned
    _four, day_four = build_and_publish(
        workspace,
        regoverned,
        spy="a" * 64,
        as_of=t2,
        observed_at=NOW + timedelta(days=3),
        invalidations=(FeatureInvalidation("panel_binding_change", earliest_session=SESSIONS[0]),),
        refresh_sector=False,
    )
    assert day_four["safe_summary"]["partition_reuse"] == {
        "reused_partition_count": len(years),
        "composed_partition_count": 0,
    }
    assert [chunk["chunk_hash"] for chunk in day_four["chunks"]] == [
        chunk["chunk_hash"] for chunk in day_three["chunks"]
    ]
    assert day_four["safe_summary"]["lineage"]["manifest_revision"] == regoverned.revision_sha256
    assert workspace.market_data.membership_events(PROFILE)[-1].sequence == 2

    # Recovery: the ragged snapshot freezes and rematerializes byte-exact.
    closure = PanelClosurePublisher(
        resolver=resolver,
        source=PanelClosureSourceRepository(
            database_path=workspace.market_data.path, resolver=resolver
        ),
        store=PanelClosureArtifactStore(resolver),
    ).publish()
    assert not closure.blocked_snapshots
    recipe = closure.recipes[str(day_three["snapshot_hash"])]
    assert recipe.row_identity_basis == "SESSION_CROSS_SECTION"
    assert recipe.membership is not None
    assert [len(epoch.absent_listing_ids) for epoch in recipe.membership.epochs] == [1, 0, 1]
    assert recipe.expected_row_count() == day_three["safe_summary"]["row_count"]
    result = ArtifactOnlyPanelRematerializer(
        resolver=resolver, store=PanelClosureArtifactStore(resolver)
    ).rematerialize(recipe.recipe_hash)
    assert result.logical_parity and result.physical_parity
    assert [item.chunk_hash for item in result.chunks] == [
        chunk["chunk_hash"] for chunk in day_three["chunks"]
    ]
