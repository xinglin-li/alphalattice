"""The coordinator writes the Universe journal: the cohort once, then only real changes."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphalattice.control.data_platform.maintenance.contracts import MarketDataChangeSet
from alphalattice.control.data_platform.maintenance.coordinator import (
    WorkspaceMaintenanceCoordinator,
)
from alphalattice.control.data_platform.maintenance.invalidation import (
    FeatureInvalidationTopology,
)
from alphalattice.control.data_platform.maintenance.registry import (
    DuckDbWorkspaceMaintenanceRegistry,
)
from alphalattice.control.data_platform.readiness import (
    WorkspaceReadinessDecision,
    WorkspaceReadinessStatus,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.contracts import FeatureBuildRequest
from alphalattice.foundation.feature_engine.runtime.factor_invalidation import (
    compile_listing_plan,
    compile_panel_plan,
    targets_by_session,
)
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    build_quality_filtered_research_manifest,
    qualification_obligation,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from tests.workspace_maintenance.acquisition_manifest import NOW, acquisition_manifest

PROFILE = "us-current-index-research"
T0 = date(2026, 7, 29)
T1 = date(2026, 7, 30)
T2 = date(2026, 7, 31)
GATEWAY = qualification_obligation("feature_input_policy", "9" * 64)
SECTOR = qualification_obligation("sector", "YAHOO_CURRENT_SECTOR:require_nonempty")


def _coordinator(market_data: MarketDataRepository, manifest) -> WorkspaceMaintenanceCoordinator:
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    gate = WorkspaceMutationGate()
    return WorkspaceMaintenanceCoordinator(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        manifest=manifest,
        provider=SimpleNamespace(name="fixture"),
        mutation_gate=gate,
        readiness_gate=SimpleNamespace(
            refresh_sources_if_due=lambda **_kwargs: WorkspaceReadinessDecision(
                WorkspaceReadinessStatus.FEATURE_BUILDING, manifest=manifest
            )
        ),
        feature_foundation=SimpleNamespace(catalog=FeatureCatalog.load(), manifest=manifest),
        registry=DuckDbWorkspaceMaintenanceRegistry(market_data.path, gate=gate),
    )


def _published(*, as_of: date, qualified: bool) -> SimpleNamespace:
    return SimpleNamespace(
        manifest=SimpleNamespace(
            safe_summary={
                "quality_governance": {
                    "gateway_qualified": qualified,
                    "quality_admission_hash": "7" * 64,
                }
            },
            as_of_session=as_of,
            history_start=date(2016, 7, 29),
            snapshot_hash="8" * 64,
        ),
        materialized_at=datetime(2026, 7, 29, 22, tzinfo=UTC),
    )


def test_population_refusal_does_not_become_an_empty_diagnoser_retry(tmp_path):
    from alphalattice.control.data_platform.maintenance.contracts import MaintenanceStatus
    from alphalattice.foundation.feature_engine.inputs.gateway import (
        FeatureInputAssessment,
        FeatureInputGatewayResult,
    )

    owner = _coordinator(MarketDataRepository(tmp_path), acquisition_manifest())
    result = FeatureInputGatewayResult(
        assessment=FeatureInputAssessment.DEFERRED,
        decisions=(),
        failure_reasons=("PANEL_SECTOR_BELOW_MINIMUM",),
    )
    assert owner._handle_governance_result(
        SimpleNamespace(result=result),
        maintenance_id=None,
        request=None,
        observed_at=NOW,
        observed_workers=4,
    ) == (MaintenanceStatus.BLOCKED, None, "feature_input.panel_sector_below_minimum")


def test_bound_manifests_journal_only_real_membership_changes_after_the_bootstrap(
    tmp_path: Path,
) -> None:
    """requirement: initialization once, then only actual changes, never backdated.

    Before the bootstrap is recorded the manifest under construction may
    change freely and the journal stays empty. The first Gateway-qualified
    publication freezes the cohort and T0. From then on a manifest that
    admits the same listings under a new revision writes nothing, an entry
    is one event effective at the session it was decided for, and a change
    effective before the journal's latest session is refused.
    """

    candidate = build_quality_filtered_research_manifest(
        acquisition_manifest(),
        eligible_listing_ids=("listing-aapl", "listing-msft"),
        qualification_obligations=(GATEWAY,),
    )
    reduced = build_quality_filtered_research_manifest(
        candidate, eligible_listing_ids=("listing-aapl",), qualification_obligations=(SECTOR,)
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    market_data.bootstrap(candidate)
    coordinator = _coordinator(market_data, candidate)

    # Still building the cohort: nothing is journaled, whatever changes.
    coordinator._bind_manifest(
        reduced, observed_at=NOW, effective_session=T0, authority="sector_partial_exclusion"
    )
    assert market_data.universe_bootstrap(PROFILE) is None
    assert market_data.membership_events(PROFILE) == ()

    # An unqualified publication records no boundary; the first qualified one does.
    coordinator._record_bootstrap_if_first(_published(as_of=T0, qualified=False))
    assert market_data.universe_bootstrap(PROFILE) is None
    coordinator._record_bootstrap_if_first(_published(as_of=T0, qualified=True))
    record = market_data.universe_bootstrap(PROFILE)
    assert record is not None
    assert record.t0_session == T0
    assert record.cohort_listing_ids == ("listing-aapl",)
    assert record.manifest_revision == reduced.revision_sha256
    assert record.derivation == "FIRST_QUALIFIED_PUBLICATION"
    assert record.initialization_assumption == "INITIAL_COHORT_BACKFILL_NOT_POINT_IN_TIME"
    coordinator._record_bootstrap_if_first(_published(as_of=T1, qualified=True))
    assert market_data.universe_bootstrap(PROFILE) == record

    # Governance only: the same listing under a new revision is not an event.
    regoverned = build_quality_filtered_research_manifest(
        reduced, eligible_listing_ids=("listing-aapl",), qualification_obligations=(GATEWAY,)
    )
    assert regoverned.revision_sha256 == reduced.revision_sha256  # same obligations, same manifest
    relabelled = build_quality_filtered_research_manifest(
        reduced,
        eligible_listing_ids=("listing-aapl",),
        qualification_obligations=(qualification_obligation("review", "1" * 64),),
    )
    assert relabelled.revision_sha256 != reduced.revision_sha256
    coordinator._bind_manifest(
        relabelled, observed_at=NOW, effective_session=T1, authority="feature_input_gateway"
    )
    assert market_data.membership_events(PROFILE) == ()

    # An explicit source transition at T2 remains one nominal event.
    readmitted = build_quality_filtered_research_manifest(
        candidate,
        eligible_listing_ids=("listing-aapl", "listing-msft"),
        qualification_obligations=(GATEWAY,),
    )
    coordinator._bind_manifest(
        readmitted,
        observed_at=datetime(2026, 7, 31, 18, tzinfo=UTC),
        effective_session=T2,
        authority="workspace_readiness",
        reference_hash="5" * 64,
    )
    assert market_data.membership_events(PROFILE) == ()  # Raw qualification is not ENTRY.
    with pytest.raises(ValueError, match="feature_candidate_admission_missing"):
        coordinator._bind_manifest(
            readmitted,
            observed_at=datetime(2026, 7, 31, 18, tzinfo=UTC),
            effective_session=T2,
            authority="qualified_source_membership",
            reference_hash="5" * 64,
        )
    from alphalattice.foundation.feature_engine.contracts import TemporalKnowledgeBoundary
    from alphalattice.foundation.feature_engine.inputs.gateway import (
        FeatureInputAdmission,
        FeatureInputAssessment,
        FeatureInputGatewayResult,
    )

    boundary = TemporalKnowledgeBoundary(
        market_as_of_session=T2,
        knowledge_cutoff_at=datetime(2026, 7, 31, 18, tzinfo=UTC),
        materialized_at=datetime(2026, 7, 31, 18, tzinfo=UTC),
        universe_source_observed_at=datetime(2026, 7, 31, 18, tzinfo=UTC),
        sector_source_observed_at=datetime(2026, 7, 31, 18, tzinfo=UTC),
    )
    # The upstream numerical qualification has its own real-workspace case;
    # this journal unit requires its exact, persisted typed admission.
    admission = FeatureInputAdmission.create(
        candidate_manifest_revision=readmitted.revision_sha256,
        admitted_listing_ids=tuple(item.listing_id for item in readmitted.listings),
        quarantines=(),
        quality_policy_hash="4" * 64,
        temporal_boundary=boundary,
        evidence_scope="BASE_FEATURE_CANDIDATES",
        feature_qualification_hash="5" * 64,
    )
    coordinator.panel_state.record_feature_input_gateway_result(
        candidate_manifest=readmitted,
        result=FeatureInputGatewayResult(
            FeatureInputAssessment.ADMITTED, (), admission=admission, research_manifest=readmitted
        ),
        temporal_boundary=boundary,
        observed_at=boundary.knowledge_cutoff_at,
    )
    coordinator._bind_manifest(
        readmitted,
        observed_at=boundary.knowledge_cutoff_at,
        effective_session=T2,
        authority="qualified_source_membership",
        reference_hash="5" * 64,
    )
    events = market_data.membership_events(PROFILE)
    assert [
        (e.sequence, e.listing_id, e.kind, e.effective_session, e.authority) for e in events
    ] == [(1, "listing-msft", "ENTRY", T2, "qualified_source_membership")]
    assert events[0].reference_hash == "5" * 64
    assert events[0].manifest_revision == readmitted.revision_sha256

    # A current observation while catching up T1 is applied prospectively, not
    # relabelled as information that existed on the old target date.
    coordinator._bind_manifest(
        reduced, observed_at=NOW, effective_session=T1, authority="workspace_readiness"
    )
    events = market_data.membership_events(PROFILE)
    assert events[-1].effective_session == date(2026, 8, 4)
    assert market_data.membership_schedule(
        PROFILE, sessions=(T1, T2), fallback_listing_ids=()
    ).members(T2) == ("listing-aapl", "listing-msft")
    with pytest.raises(ValueError, match="without_effective_session"):
        coordinator._bind_manifest(readmitted, observed_at=NOW)
    assert len(market_data.membership_events(PROFILE)) == 2


def test_membership_invalidations_take_effect_at_the_session_not_history_start() -> None:
    """A member joining or leaving reaches the Panel from the effective session on.

    The entrant's base Formula values are planned over its whole history (its
    windows need them) while the Panel sees them from the effective session
    only -- also for a listing that was a member before and rejoins, whose
    earlier rows stand.
    """

    catalog = FeatureCatalog.load()
    plan = FeatureInvalidationTopology(catalog).plan(
        MarketDataChangeSet(
            listing_changes=(),
            membership_additions=("listing-nvda",),
            membership_removals=("listing-old",),
            receipt_hashes=(),
        ),
        history_start=date(2016, 1, 4),
        as_of_session=T2,
    )
    by_kind = {item.kind: item for item in plan.invalidations}
    assert by_kind["manifest_addition"].earliest_session == T2
    assert by_kind["manifest_removal"].earliest_session == T2
    assert by_kind["manifest_removal"].listing_id == "listing-old"
    sessions = (date(2026, 7, 27), date(2026, 7, 28), T0, T1, T2)
    request = FeatureBuildRequest.create(
        manifest_revision="1" * 64,
        catalog=catalog.binding,
        spy_revision="2" * 64,
        history_start=sessions[0],
        as_of_session=T2,
        invalidations=plan.invalidations,
    )
    base_reach = compile_listing_plan(
        catalog=catalog,
        request=request,
        invalidations=plan.invalidations,
        listing_id="listing-nvda",
        sessions=sessions,
    )
    assert set(targets_by_session(base_reach, sessions)) == set(sessions)
    unrelated = compile_listing_plan(
        catalog=catalog,
        request=request,
        invalidations=plan.invalidations,
        listing_id="not_changed",
        sessions=sessions,
    )
    assert not unrelated.items
    panel_reach = compile_listing_plan(
        catalog=catalog,
        request=request,
        invalidations=plan.invalidations,
        listing_id="listing-nvda",
        sessions=sessions,
        reach="panel",
    )
    assert set(targets_by_session(panel_reach, sessions)) == {T2}
    panel_plan = compile_panel_plan(
        catalog=catalog,
        request=request,
        invalidations=plan.invalidations,
        listing_plans=(panel_reach,),
        sessions=sessions,
        base_reusable=True,
    )
    assert set(targets_by_session(panel_plan, sessions)) == {T2}
