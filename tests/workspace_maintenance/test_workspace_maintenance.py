"""Offline acceptance for stable daily maintenance contracts."""

# The ignored case adds playpen/src explicitly.

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Barrier, Event, get_ident
from types import SimpleNamespace

import pytest

from alphalattice.control.data_platform.maintenance.contracts import (
    ActionAuditChainReceipt,
    ActionAuditScope,
    ListingMarketDataChange,
    MaintenancePhase,
    MaintenanceStatus,
    MaintenanceTrigger,
    MarketDataChangeSet,
    WorkspaceMaintenanceRequest,
)
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
from alphalattice.control.product_host.maintenance import (
    MaintenanceBackgroundHost,
    MaintenanceWakeController,
)
from alphalattice.control.workspace_runtime.database import open_workspace_database
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.catalog.contracts import (
    FeatureCatalog,
    desktop_core_feature_bundle,
)
from alphalattice.foundation.feature_engine.inputs.contracts import (
    FeatureCandidateRecheck,
    MaterializedFeatureQualification,
)
from alphalattice.foundation.feature_engine.inputs.gateway import (
    FeatureInputGateway,
    FeatureInputGovernanceService,
)
from alphalattice.foundation.feature_engine.publication.current_storage import (
    canonical_cutoff_set,
)
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.runtime.diagnostics import (
    audit_bounded_restatements,
)
from alphalattice.foundation.market_data_ops.runtime.refresh import (
    RefreshGapDisposition,
    normal_refresh_plan,
)
from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
    CurrentUniverseMaintenance,
    CurrentUniverseMaintenanceStatus,
    current_universe_maintenance_id,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CandidateDataRecheck,
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    UniverseManifest,
    build_quality_filtered_research_manifest,
)
from alphalattice.foundation.market_data_ops.sources.providers import (
    HydrationEvidence,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import (
    sanitize_payload,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    ActionAuditScopeInsufficient,
    FeatureSourceDelta,
    MarketDataRepository,
    ProviderAdjustedSeriesRevision,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.workspace_maintenance.acquisition_manifest import NOW, acquisition_manifest
from tests.workspace_maintenance.bounded_provider_support import RecordingBoundedProvider

CASE_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = CASE_ROOT.parents[1]


def _request(*, cutoff: datetime = NOW) -> WorkspaceMaintenanceRequest:
    return WorkspaceMaintenanceRequest.create(
        market_profile_id="us-current-index-research",
        target_market_session=date(2026, 7, 31),
        knowledge_cutoff_at=cutoff,
        trigger=MaintenanceTrigger.STARTUP,
        membership_revision="b" * 64,
        data_policy_hash="d" * 64,
        feature_policy_hash="e" * 64,
    )


def test_a_sealed_request_changes_with_its_candidate_scopes_kept() -> None:
    """regression (V466): a research update rebuilt the data update's request through
    `asdict`, which turned its recheck scopes into dicts its seal could not read, so no
    update planned while a recheck was due; the request's owner changes it, scopes kept."""

    feature = FeatureCandidateRecheck(parent_manifest_revision="a" * 64, listing_ids=("L1", "L2"))
    data = CandidateDataRecheck(
        prior_onboarding_id="1" * 64,
        candidate_manifest_hash="2" * 64,
        qualified_parent_revision="3" * 64,
        listing_ids=("L3",),
        history_start=date(2016, 9, 12),
    )
    request = _request().with_changes(candidate_recheck=feature, candidate_data_recheck=data)
    assert request.with_changes() == request
    moved = request.with_changes(target_market_session=date(2026, 7, 30))
    assert (moved.candidate_recheck, moved.candidate_data_recheck) == (feature, data)
    assert moved.request_hash != request.request_hash
    assert moved == WorkspaceMaintenanceRequest.create(
        market_profile_id="us-current-index-research",
        target_market_session=date(2026, 7, 30),
        knowledge_cutoff_at=NOW,
        trigger=MaintenanceTrigger.STARTUP,
        membership_revision="b" * 64,
        data_policy_hash="d" * 64,
        feature_policy_hash="e" * 64,
        candidate_recheck=feature,
        candidate_data_recheck=data,
    )


def _maintenance_workspace(
    tmp_path: Path,
    *,
    eligible_listing_ids: tuple[str, ...] = ("listing-aapl",),
) -> tuple[
    UniverseManifest,
    MarketDataRepository,
    FeatureStateRepository,
    PanelStateRepository,
]:
    manifest = build_quality_filtered_research_manifest(
        acquisition_manifest(), eligible_listing_ids=eligible_listing_ids
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(manifest)
    return manifest, market_data, feature_state, panel_state


def test_dogfood_provider_allows_only_explicit_listing_scoped_history(tmp_path: Path) -> None:
    provider = RecordingBoundedProvider(
        tmp_path / "cache",
        earliest_allowed=date(2026, 6, 19),
        target=date(2026, 8, 3),
        bounded_subject_starts={"SPY": date(2026, 6, 16)},
        authorized_full_history_starts={"listing-aapl": date(2016, 8, 1)},
    )

    _started, scope = provider._record(
        kind="hydration",
        subject="AAPL",
        listing_id="listing-aapl",
        start=date(2016, 8, 1),
        end=date(2017, 7, 31),
    )
    assert scope == "LISTING_FULL_HISTORY"
    _started, scope = provider._record(
        kind="hydration",
        subject="SPY",
        listing_id="spy-reference",
        start=date(2026, 6, 16),
        end=date(2026, 8, 3),
    )
    assert scope == "ROLLING_BOUND"

    with pytest.raises(RuntimeError, match="BOUNDED_PROVIDER_PLAN_VIOLATION"):
        provider._record(
            kind="hydration",
            subject="MSFT",
            listing_id="listing-msft",
            start=date(2016, 8, 1),
            end=date(2017, 7, 31),
        )
    with pytest.raises(RuntimeError, match="BOUNDED_PROVIDER_PLAN_VIOLATION"):
        provider._record(
            kind="hydration",
            subject="AAPL",
            listing_id="listing-aapl",
            start=date(2016, 7, 31),
            end=date(2017, 7, 31),
        )
    with pytest.raises(RuntimeError, match="BOUNDED_PROVIDER_PLAN_VIOLATION"):
        provider._record(
            kind="daily",
            subject="AAPL",
            start=date(2016, 8, 1),
            end=date(2017, 7, 31),
        )


def test_membership_identity_is_stable_while_temporal_admission_changes() -> None:
    acquisition = acquisition_manifest()
    listing_ids = tuple(item.listing_id for item in acquisition.listings)
    first = build_quality_filtered_research_manifest(
        acquisition,
        eligible_listing_ids=listing_ids,
        quality_admission_hash="1" * 64,
    )
    second = build_quality_filtered_research_manifest(
        acquisition,
        eligible_listing_ids=listing_ids,
        quality_admission_hash="2" * 64,
    )
    reduced = build_quality_filtered_research_manifest(
        acquisition,
        eligible_listing_ids=("listing-aapl",),
        quality_admission_hash="3" * 64,
    )
    assert first.revision_sha256 == second.revision_sha256
    assert first.membership_fingerprint == acquisition.membership_fingerprint
    assert reduced.revision_sha256 != first.revision_sha256
    assert _request().request_hash != _request(cutoff=NOW + timedelta(minutes=1)).request_hash


def test_three_month_absence_is_incremental_but_historical_gap_needs_approval() -> None:
    three_months = normal_refresh_plan(date(2026, 5, 1), date(2026, 8, 3))
    assert three_months.disposition is RefreshGapDisposition.NORMAL_INCREMENT
    assert three_months.missing_calendar_days == 94
    assert three_months.fetch_start == date(2026, 3, 17)
    assert len(three_months.windows) == 1

    extended = normal_refresh_plan(date(2025, 12, 31), date(2026, 8, 3))
    assert extended.disposition is RefreshGapDisposition.EXTENDED_CATCH_UP
    assert len(extended.windows) == 2
    assert extended.windows[0].end + timedelta(days=1) == extended.windows[1].start

    historical = normal_refresh_plan(date(2025, 1, 1), date(2026, 8, 3))
    assert historical.disposition is RefreshGapDisposition.EXPLICIT_APPROVAL_REQUIRED
    assert historical.windows == ()


def test_action_audit_receipts_chain_rolling_windows_to_full_anchor() -> None:
    full = ActionAuditChainReceipt.create(
        listing_id="listing-aapl",
        provider="fixture",
        audit_scope=ActionAuditScope.FULL,
        previous_receipt_hash=None,
        full_anchor_receipt_hash=None,
        history_start=date(2016, 1, 4),
        history_end=date(2026, 7, 31),
        covered_through_session=date(2026, 7, 31),
        window_action_hash="a" * 64,
        action_set_hash="a" * 64,
        raw_evidence_hash="b" * 64,
        mapping_revision="c" * 64,
        data_policy_hash="d" * 64,
        provider_receipt_hash="e" * 64,
        observed_at=NOW,
    )
    rolling = ActionAuditChainReceipt.create(
        listing_id=full.listing_id,
        provider=full.provider,
        audit_scope=ActionAuditScope.ROLLING,
        previous_receipt_hash=full.receipt_hash,
        full_anchor_receipt_hash=full.full_anchor_receipt_hash,
        history_start=date(2026, 6, 17),
        history_end=date(2026, 8, 3),
        covered_through_session=date(2026, 8, 3),
        window_action_hash="f" * 64,
        action_set_hash="a" * 64,
        raw_evidence_hash="1" * 64,
        mapping_revision="c" * 64,
        data_policy_hash="d" * 64,
        provider_receipt_hash="2" * 64,
        observed_at=NOW + timedelta(days=1),
    )
    assert rolling.previous_receipt_hash == full.receipt_hash
    assert rolling.full_anchor_receipt_hash == full.full_anchor_receipt_hash
    with pytest.raises(ValueError, match="requires a full-history anchor"):
        ActionAuditChainReceipt.create(
            listing_id=full.listing_id,
            provider=full.provider,
            audit_scope=ActionAuditScope.ROLLING,
            previous_receipt_hash=full.receipt_hash,
            full_anchor_receipt_hash=None,
            history_start=rolling.history_start,
            history_end=rolling.history_end,
            covered_through_session=rolling.covered_through_session,
            window_action_hash=rolling.window_action_hash,
            action_set_hash=rolling.action_set_hash,
            raw_evidence_hash=rolling.raw_evidence_hash,
            mapping_revision=rolling.mapping_revision,
            data_policy_hash=rolling.data_policy_hash,
            provider_receipt_hash=rolling.provider_receipt_hash,
            observed_at=rolling.observed_at,
        )


def test_an_action_audit_receipt_changed_in_place_is_refused_when_read(tmp_path: Path) -> None:
    """regression (V258, SC4, EV2): the registry rebuilt a receipt from its stored JSON without
    its hash, so one whose coverage moved later, its old hash kept, read back accepted and the
    coordinator reused the chain by that date. A receipt checks its hash when built or read."""

    import duckdb

    registry = DuckDbWorkspaceMaintenanceRegistry(
        tmp_path / "market-data.duckdb", gate=WorkspaceMutationGate()
    )
    full = ActionAuditChainReceipt.create(
        listing_id="listing-aapl",
        provider="fixture",
        audit_scope=ActionAuditScope.FULL,
        previous_receipt_hash=None,
        full_anchor_receipt_hash=None,
        history_start=date(2016, 1, 4),
        history_end=date(2026, 7, 31),
        covered_through_session=date(2026, 7, 31),
        window_action_hash="a" * 64,
        action_set_hash="a" * 64,
        raw_evidence_hash="b" * 64,
        mapping_revision="c" * 64,
        data_policy_hash="d" * 64,
        provider_receipt_hash="e" * 64,
        observed_at=NOW,
    )
    assert registry.record_action_audit(full)
    assert registry.latest_action_audit("listing-aapl", "fixture") == full
    with pytest.raises(ValueError, match="receipt hash is invalid"):
        replace(full, covered_through_session=date(2026, 9, 30))
    connection = duckdb.connect(str(tmp_path / "market-data.duckdb"))
    try:
        stored = json.loads(
            connection.execute("SELECT receipt_json FROM action_audit_chain_receipt").fetchone()[0]
        )
        stored["covered_through_session"] = "2026-09-30"
        connection.execute(
            "UPDATE action_audit_chain_receipt SET receipt_json = ?", [json.dumps(stored)]
        )
    finally:
        connection.close()
    with pytest.raises(ValueError, match="receipt hash is invalid"):
        registry.latest_action_audit("listing-aapl", "fixture")


def test_domain_topology_coalesces_changes_without_a_general_dag() -> None:
    change_set = MarketDataChangeSet(
        listing_changes=(
            ListingMarketDataChange(
                "listing-aapl",
                new_session_start=date(2026, 8, 3),
                raw_correction_start=date(2026, 7, 1),
                source_receipt_hashes=("a" * 64,),
            ),
            ListingMarketDataChange(
                "listing-msft",
                action_correction_start=date(2020, 1, 2),
                source_receipt_hashes=("b" * 64,),
            ),
        ),
        spy_correction_start=date(2026, 7, 15),
        sector_revision_changed=True,
        membership_additions=("listing-nvda", "listing-nvda"),
        membership_removals=("listing-old",),
        receipt_hashes=("a" * 64, "b" * 64),
    )
    plan = FeatureInvalidationTopology(FeatureCatalog.load()).plan(
        change_set,
        history_start=date(2016, 1, 4),
        as_of_session=date(2026, 8, 3),
    )
    assert {item.kind for item in plan.invalidations} == {
        "normal_new_session",
        "ohlc_correction",
        "spy_correction",
        "sector_revision_change",
        "manifest_addition",
        "manifest_removal",
    }
    assert plan.ordered_stages[-1] == "active_panel_binding"
    assert plan.market_dependent_factor_ids
    assert len([item for item in plan.invalidations if item.kind == "manifest_addition"]) == 1


def test_registry_is_idempotent_and_persists_transport_degradation(tmp_path: Path) -> None:
    registry = DuckDbWorkspaceMaintenanceRegistry(
        tmp_path / "market-data.duckdb", gate=WorkspaceMutationGate()
    )
    request = _request()
    first = registry.admit(request, observed_at=NOW)
    assert registry.admit(request, observed_at=NOW) == first
    updated = registry.update(
        first.cycle_id,
        phase=MaintenancePhase.MARKET_DATA,
        status=MaintenanceStatus.DEFERRED,
        observed_at=NOW,
        retry_after_at=NOW + timedelta(minutes=5),
        failure_code="data.rate_limited",
        transport_workers=2,
    )
    assert updated.transport_workers == 2
    assert registry.latest_cycle(request.market_profile_id) == updated


def test_registry_records_the_maintenance_scope_a_cycle_admitted_once(tmp_path: Path) -> None:
    """requirement: a cycle's listing units are resolved by the run it admitted.

    The coordinator records the maintenance run (its id, the manifest revision it
    bound, the session) in the cycle's own event log when it admits the runner;
    the same scope recorded again (every bounded run of the cycle) appends
    nothing, a different one appends and is the one read back, and a cycle that
    recorded none reads back None -- the readback then falls back to the cycle's
    request, verified, never to today's manifest.
    """

    registry = DuckDbWorkspaceMaintenanceRegistry(
        tmp_path / "market-data.duckdb", gate=WorkspaceMutationGate()
    )
    request = _request()
    cycle = registry.admit(request, observed_at=NOW)
    assert registry.market_data_scope(cycle.cycle_id) is None
    scope = {
        "maintenance_id": "m" * 64,
        "manifest_revision": "b" * 64,
        "as_of_session": "2026-07-31",
    }
    for _ in range(3):
        registry.record_market_data_scope(
            cycle.cycle_id,
            maintenance_id=scope["maintenance_id"],
            manifest_revision=scope["manifest_revision"],
            as_of_session=date(2026, 7, 31),
            observed_at=NOW,
        )
    assert registry.market_data_scope(cycle.cycle_id) == scope
    registry.record_market_data_scope(
        cycle.cycle_id,
        maintenance_id="n" * 64,
        manifest_revision="c" * 64,
        as_of_session=date(2026, 7, 31),
        observed_at=NOW + timedelta(seconds=1),
    )
    assert registry.market_data_scope(cycle.cycle_id)["maintenance_id"] == "n" * 64
    connection = open_workspace_database(registry.database_path, read_only=True)
    try:
        kinds = [
            row[0]
            for row in connection.execute(
                "SELECT kind FROM workspace_maintenance_event WHERE cycle_id = ? ORDER BY sequence",
                [cycle.cycle_id],
            ).fetchall()
        ]
    finally:
        connection.close()
    assert kinds == [
        "workspace_maintenance.admitted",
        "workspace_maintenance.market_data_scope",
        "workspace_maintenance.market_data_scope",
    ]


def test_registry_reconciles_only_stale_orphaned_running_cycles(tmp_path: Path) -> None:
    registry = DuckDbWorkspaceMaintenanceRegistry(
        tmp_path / "market-data.duckdb", gate=WorkspaceMutationGate()
    )
    request = _request()
    cycle = registry.admit(request, observed_at=NOW)

    assert (
        registry.reconcile_stale_running_cycles(
            request.market_profile_id,
            observed_at=NOW + timedelta(seconds=29),
        )
        == ()
    )
    assert registry.cycle(cycle.cycle_id).status is MaintenanceStatus.RUNNING

    assert registry.reconcile_stale_running_cycles(
        request.market_profile_id,
        observed_at=NOW + timedelta(seconds=31),
    ) == (cycle.cycle_id,)
    reconciled = registry.cycle(cycle.cycle_id)
    assert reconciled.status is MaintenanceStatus.BLOCKED
    assert reconciled.failure_code == "workspace_maintenance.worker_lost"
    assert reconciled.updated_at == cycle.updated_at


def test_early_resume_is_rejected_before_any_readiness_or_provider_work(
    tmp_path: Path,
) -> None:
    registry = DuckDbWorkspaceMaintenanceRegistry(
        tmp_path / "market-data.duckdb", gate=WorkspaceMutationGate()
    )
    request = _request()
    cycle = registry.admit(request, observed_at=NOW)
    registry.update(
        cycle.cycle_id,
        phase=MaintenancePhase.MARKET_DATA,
        status=MaintenanceStatus.DEFERRED,
        observed_at=NOW,
        retry_after_at=NOW + timedelta(minutes=5),
        failure_code="data.rate_limited",
        transport_workers=2,
    )

    class ExplodingReadiness:
        def refresh_sources_if_due(self, **_kwargs):
            raise AssertionError("early resume reached durable readiness")

    manifest = build_quality_filtered_research_manifest(
        acquisition_manifest(),
        eligible_listing_ids=("listing-aapl", "listing-msft"),
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(manifest)
    coordinator = WorkspaceMaintenanceCoordinator(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        manifest=manifest,
        provider=SimpleNamespace(name="fixture"),
        mutation_gate=WorkspaceMutationGate(),
        readiness_gate=ExplodingReadiness(),
        feature_foundation=SimpleNamespace(catalog=FeatureCatalog.load()),
        registry=registry,
    )
    outcome = coordinator.run(request, observed_at=NOW + timedelta(minutes=1))
    assert outcome.status is MaintenanceStatus.DEFERRED
    assert outcome.retry_after_at == NOW + timedelta(minutes=5)


def test_candidate_binding_survives_a_reduced_research_manifest(
    tmp_path: Path,
) -> None:
    candidate = build_quality_filtered_research_manifest(
        acquisition_manifest(),
        eligible_listing_ids=("listing-aapl", "listing-msft"),
    )
    reduced = build_quality_filtered_research_manifest(
        candidate,
        eligible_listing_ids=("listing-aapl",),
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(candidate)
    feature_state.bind_feature_input_candidate(candidate, observed_at=NOW)
    market_data.bootstrap(reduced)

    restored = feature_state.feature_input_candidate(candidate.profile.market_profile_id)
    assert restored is not None
    assert restored.revision_sha256 == candidate.revision_sha256
    assert {item.listing_id for item in restored.listings} == {
        "listing-aapl",
        "listing-msft",
    }


def test_stale_candidate_membership_cannot_bypass_active_quarantine_child(
    tmp_path: Path,
) -> None:
    candidate = build_quality_filtered_research_manifest(
        acquisition_manifest(),
        eligible_listing_ids=("listing-aapl", "listing-msft"),
    )
    active = build_quality_filtered_research_manifest(
        candidate,
        eligible_listing_ids=("listing-aapl",),
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(candidate)
    feature_state.bind_feature_input_candidate(candidate, observed_at=NOW)
    market_data.bootstrap(active)
    mutation_gate = WorkspaceMutationGate()

    def readiness(**kwargs):
        assert kwargs["operation_started_at"] == NOW
        return WorkspaceReadinessDecision(
            WorkspaceReadinessStatus.FEATURE_BUILDING, manifest=active
        )

    coordinator = WorkspaceMaintenanceCoordinator(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        manifest=active,
        provider=SimpleNamespace(name="fixture"),
        mutation_gate=mutation_gate,
        readiness_gate=SimpleNamespace(refresh_sources_if_due=readiness),
        feature_foundation=SimpleNamespace(catalog=FeatureCatalog.load(), manifest=active),
        registry=DuckDbWorkspaceMaintenanceRegistry(market_data.path, gate=mutation_gate),
    )
    stale_request = WorkspaceMaintenanceRequest.create(
        market_profile_id=active.profile.market_profile_id,
        target_market_session=date(2026, 7, 31),
        knowledge_cutoff_at=None,
        requested_at=NOW,
        trigger=MaintenanceTrigger.STARTUP,
        membership_revision=candidate.revision_sha256,
        data_policy_hash="d" * 64,
        feature_policy_hash="e" * 64,
    )

    outcome = coordinator.run(stale_request, observed_at=NOW)

    assert outcome.status is MaintenanceStatus.BLOCKED
    assert outcome.failure_code == "workspace_maintenance.membership_revision_mismatch"


def test_cycle_blocked_inside_quality_governance_resumes_it_when_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression: a quality block was resumed as an ordinary cycle.

    A cycle that stopped inside quality governance (waiting for a decision,
    pending review, or refused while binding the child the decision derived)
    must govern again when it is run again. Market data is current by then,
    and an ordinary cycle would take an older admission for the same revision
    as clearance and build on the parent -- the decision silently dropped.
    """

    manifest = build_quality_filtered_research_manifest(
        acquisition_manifest(),
        eligible_listing_ids=("listing-aapl", "listing-msft"),
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(manifest)
    mutation_gate = WorkspaceMutationGate()
    registry = DuckDbWorkspaceMaintenanceRegistry(market_data.path, gate=mutation_gate)
    coordinator = WorkspaceMaintenanceCoordinator(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        manifest=manifest,
        provider=SimpleNamespace(name="fixture"),
        mutation_gate=mutation_gate,
        readiness_gate=SimpleNamespace(
            refresh_sources_if_due=lambda **_kwargs: WorkspaceReadinessDecision(
                WorkspaceReadinessStatus.RESEARCH_READY, manifest=None
            )
        ),
        feature_foundation=SimpleNamespace(catalog=FeatureCatalog.load(), manifest=manifest),
        registry=registry,
    )
    request = WorkspaceMaintenanceRequest.create(
        market_profile_id=manifest.profile.market_profile_id,
        target_market_session=date(2026, 7, 31),
        knowledge_cutoff_at=NOW,
        trigger=MaintenanceTrigger.STARTUP,
        membership_revision=manifest.revision_sha256,
        data_policy_hash="d" * 64,
        feature_policy_hash="e" * 64,
    )
    seen: list[tuple[MaintenanceStatus, MaintenancePhase, bool]] = []

    def observe_cycle(cycle, request_, now, *, resuming_quality_deferred, **_kwargs):
        seen.append((cycle.status, cycle.phase, resuming_quality_deferred))
        return coordinator._finish(
            cycle.cycle_id,
            phase=cycle.phase,
            status=cycle.status,
            observed_at=now,
            failure_code=cycle.failure_code,
        )

    monkeypatch.setattr(coordinator, "_run_cycle", observe_cycle)
    cycle = registry.admit(request, observed_at=NOW)
    for status, phase, expected in (
        (MaintenanceStatus.BLOCKED, MaintenancePhase.QUALITY, True),
        (MaintenanceStatus.REVIEW_PENDING, MaintenancePhase.QUALITY, True),
        (MaintenanceStatus.BLOCKED, MaintenancePhase.MARKET_DATA, False),
        (MaintenanceStatus.BLOCKED, MaintenancePhase.FEATURE, False),
        (MaintenanceStatus.RUNNING, MaintenancePhase.QUALITY, False),
    ):
        registry.update(
            cycle.cycle_id,
            phase=phase,
            status=status,
            observed_at=NOW,
            failure_code="workspace_maintenance.derived_manifest_evidence_incomplete"
            if status is MaintenanceStatus.BLOCKED
            else None,
        )
        coordinator.run(request, observed_at=NOW + timedelta(minutes=1))
        assert seen[-1] == (status, phase, expected), seen[-1]


def test_snapshot_publication_gets_one_idempotent_transient_retry(tmp_path: Path) -> None:
    manifest = build_quality_filtered_research_manifest(
        acquisition_manifest(),
        eligible_listing_ids=("listing-aapl", "listing-msft"),
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(manifest)

    class Publisher:
        def __init__(self) -> None:
            self.calls = 0

        def publish(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise OSError("fixture transient publication failure")
            return "published"

    publisher = Publisher()
    mutation_gate = WorkspaceMutationGate()
    coordinator = WorkspaceMaintenanceCoordinator(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        manifest=manifest,
        provider=SimpleNamespace(name="fixture"),
        mutation_gate=mutation_gate,
        readiness_gate=SimpleNamespace(),
        feature_foundation=SimpleNamespace(catalog=FeatureCatalog.load()),
        registry=DuckDbWorkspaceMaintenanceRegistry(market_data.path, gate=mutation_gate),
        snapshot_publisher=publisher,
    )
    result = coordinator._publish_snapshot_with_bounded_retry(
        history_start=date(2016, 8, 1),
        as_of_session=date(2026, 7, 31),
        observed_at=NOW,
    )
    assert result == "published"
    assert publisher.calls == 2


def test_snapshot_publication_clock_cannot_precede_active_panel_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_quality_filtered_research_manifest(
        acquisition_manifest(),
        eligible_listing_ids=("listing-aapl", "listing-msft"),
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(manifest)
    active_cutoff = NOW + timedelta(minutes=5)
    monkeypatch.setattr(
        panel_state,
        "active_feature_panel",
        lambda _market_profile_id: {"knowledge_cutoff_at": active_cutoff.replace(tzinfo=None)},
    )

    class Publisher:
        observed_at: datetime | None = None

        def publish(self, **kwargs):
            self.observed_at = kwargs["observed_at"]
            return "published"

    publisher = Publisher()
    mutation_gate = WorkspaceMutationGate()
    coordinator = WorkspaceMaintenanceCoordinator(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        manifest=manifest,
        provider=SimpleNamespace(name="fixture"),
        mutation_gate=mutation_gate,
        readiness_gate=SimpleNamespace(),
        feature_foundation=SimpleNamespace(catalog=FeatureCatalog.load()),
        registry=DuckDbWorkspaceMaintenanceRegistry(market_data.path, gate=mutation_gate),
        snapshot_publisher=publisher,
    )

    result = coordinator._publish_snapshot_with_bounded_retry(
        history_start=date(2016, 8, 1),
        as_of_session=date(2026, 7, 31),
        observed_at=NOW,
    )

    assert result == "published"
    assert publisher.observed_at == active_cutoff


def test_blocked_partial_maintenance_does_not_publish_adjusted_revision(
    tmp_path: Path,
) -> None:
    manifest = build_quality_filtered_research_manifest(
        acquisition_manifest(),
        eligible_listing_ids=("listing-aapl", "listing-msft"),
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(manifest)
    mutation_gate = WorkspaceMutationGate()
    registry = DuckDbWorkspaceMaintenanceRegistry(market_data.path, gate=mutation_gate)
    request = _request()
    cycle = registry.admit(request, observed_at=NOW)
    publish_calls = 0

    def publish_adjusted_return_revision() -> None:
        nonlocal publish_calls
        publish_calls += 1

    coordinator = WorkspaceMaintenanceCoordinator(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        manifest=manifest,
        provider=SimpleNamespace(name="fixture"),
        mutation_gate=mutation_gate,
        readiness_gate=SimpleNamespace(),
        feature_foundation=SimpleNamespace(catalog=FeatureCatalog.load()),
        registry=registry,
        publish_adjusted_return_revision=publish_adjusted_return_revision,
    )
    change_set = MarketDataChangeSet(
        listing_changes=(
            ListingMarketDataChange(
                listing_id="listing-aapl",
                adjusted_return_change_sessions=(date(2026, 7, 31),),
            ),
        )
    )

    outcome = coordinator._finish(
        cycle.cycle_id,
        phase=MaintenancePhase.MARKET_DATA,
        status=MaintenanceStatus.BLOCKED,
        observed_at=NOW,
        change_set=change_set,
        failure_code="data.full_history_audit_approval_required",
    )

    assert outcome.status is MaintenanceStatus.BLOCKED
    assert publish_calls == 0
    persisted = registry.cycle(cycle.cycle_id)
    assert persisted.market_data_change_set == change_set
    assert persisted.failure_code == "data.full_history_audit_approval_required"


def test_deferred_market_change_prefix_is_merged_without_losing_sessions() -> None:
    prior = MarketDataChangeSet(
        listing_changes=(
            ListingMarketDataChange(
                listing_id="listing-aapl",
                new_session_start=date(2026, 7, 31),
                new_sessions=(date(2026, 7, 31),),
                adjusted_return_change_sessions=(date(2026, 7, 31),),
                source_receipt_hashes=("a" * 64,),
            ),
        ),
        membership_additions=("listing-msft",),
        receipt_hashes=("1" * 64,),
    )
    current = MarketDataChangeSet(
        listing_changes=(
            ListingMarketDataChange(
                listing_id="listing-aapl",
                new_session_start=date(2026, 8, 3),
                new_sessions=(date(2026, 8, 3),),
                adjusted_return_change_sessions=(date(2026, 8, 3),),
                source_receipt_hashes=("b" * 64,),
            ),
        ),
        membership_removals=("listing-msft",),
        receipt_hashes=("2" * 64,),
    )

    merged = WorkspaceMaintenanceCoordinator._merge_change_sets(prior, current)

    assert merged.listing_changes[0].new_session_start == date(2026, 7, 31)
    assert merged.listing_changes[0].new_sessions == (
        date(2026, 7, 31),
        date(2026, 8, 3),
    )
    assert merged.listing_changes[0].adjusted_return_change_sessions == (
        date(2026, 7, 31),
        date(2026, 8, 3),
    )
    assert merged.listing_changes[0].source_receipt_hashes == ("a" * 64, "b" * 64)
    assert merged.membership_additions == ()
    assert merged.membership_removals == ("listing-msft",)
    assert merged.receipt_hashes == ("1" * 64, "2" * 64)


def test_unconsumed_feature_source_change_set_survives_cycle_and_manifest_change() -> None:
    prior_manifest = acquisition_manifest()
    active_manifest = build_quality_filtered_research_manifest(
        prior_manifest,
        eligible_listing_ids=("listing-aapl",),
    )
    append_session = date(2026, 8, 3)
    raw_correction = date(2026, 7, 15)
    adjusted_correction = date(2026, 7, 16)
    adjusted_revisions = (
        ProviderAdjustedSeriesRevision(
            receipt_hash="1" * 64,
            listing_id="listing-aapl",
            provider="fixture",
            prior_series_hash="a" * 64,
            next_series_hash="b" * 64,
            scope_start=date(2026, 6, 19),
            scope_end=append_session,
            full_history=False,
            changed_value_count=1,
            changed_return_sessions=(append_session,),
            session_set_changed=True,
            uniform_rescale=False,
            source_receipt_hash="2" * 64,
            observed_at=NOW,
        ),
        ProviderAdjustedSeriesRevision(
            receipt_hash="3" * 64,
            listing_id="listing-aapl",
            provider="fixture",
            prior_series_hash="b" * 64,
            next_series_hash="c" * 64,
            scope_start=date(2016, 8, 3),
            scope_end=append_session,
            full_history=True,
            changed_value_count=100,
            changed_return_sessions=(adjusted_correction,),
            session_set_changed=False,
            uniform_rescale=False,
            source_receipt_hash="4" * 64,
            observed_at=NOW,
        ),
        ProviderAdjustedSeriesRevision(
            receipt_hash="5" * 64,
            listing_id="listing-aapl",
            provider="fixture",
            prior_series_hash="c" * 64,
            next_series_hash="d" * 64,
            scope_start=date(2016, 8, 3),
            scope_end=append_session,
            full_history=True,
            changed_value_count=100,
            changed_return_sessions=(),
            session_set_changed=False,
            uniform_rescale=True,
            source_receipt_hash="6" * 64,
            observed_at=NOW,
        ),
    )

    class Store:
        def load_universe_manifest_revision(self, _revision: str):
            return prior_manifest

        def feature_source_appends(self, *_args, **_kwargs):
            assert _args[0] == prior_manifest  # Still owed pre-exit source coverage.
            return (
                FeatureSourceDelta(
                    "listing-aapl",
                    (append_session,),
                    "7" * 64,
                ),
            )

        def raw_bar_semantic_deltas_since(self, *_args, **_kwargs):
            assert _args[0] == prior_manifest
            return (
                FeatureSourceDelta(
                    "listing-aapl",
                    (raw_correction,),
                    "8" * 64,
                ),
            )

        def provider_adjusted_semantic_deltas_since(self, *_args, **_kwargs):
            assert _args[0] == prior_manifest
            return adjusted_revisions

        def verified_feature_source_session_sets(self, *_args, **_kwargs):
            assert _args[0] == prior_manifest
            return frozenset({"listing-aapl"})

        def current_sector_state(self, _manifest):
            return SimpleNamespace(sector_revision="new-sector")

    coordinator = object.__new__(WorkspaceMaintenanceCoordinator)
    coordinator.market_data = Store()
    coordinator.feature_state = coordinator.market_data
    coordinator.manifest = active_manifest
    coordinator.feature_foundation = SimpleNamespace(
        catalog=FeatureCatalog.load(), panel=SimpleNamespace(policy_hash="legacy")
    )

    change_set = coordinator._unconsumed_feature_source_change_set(
        {
            "manifest_revision": prior_manifest.revision_sha256,
            "sector_revision": "old-sector",
            "as_of_session": date(2026, 7, 31),
            "activated_at": NOW - timedelta(hours=1),
        },
        through=append_session,
    )

    assert change_set.membership_removals == ("listing-msft",)
    assert change_set.sector_revision_changed
    assert len(change_set.listing_changes) == 1
    change = change_set.listing_changes[0]
    assert change.new_sessions == (append_session,)
    assert change.raw_correction_sessions == (raw_correction,)
    assert change.adjusted_return_change_sessions == (
        adjusted_correction,
        append_session,
    )
    assert "5" * 64 not in change.source_receipt_hashes


@pytest.mark.parametrize(
    ("source_verified", "historical_member"), [(False, False), (True, True), (True, False)]
)
def test_unconsumed_feature_source_change_set_rejects_unknown_session_change(
    source_verified: bool, historical_member: bool
) -> None:
    manifest = acquisition_manifest()
    revision = ProviderAdjustedSeriesRevision(
        receipt_hash="1" * 64,
        listing_id="listing-aapl",
        provider="fixture",
        prior_series_hash="a" * 64,
        next_series_hash="b" * 64,
        scope_start=date(2026, 7, 1),
        scope_end=date(2026, 8, 3),
        full_history=False,
        changed_value_count=1,
        changed_return_sessions=(date(2026, 7, 15),),
        session_set_changed=True,
        uniform_rescale=False,
        source_receipt_hash="2" * 64,
        observed_at=NOW,
    )
    store = SimpleNamespace(
        load_universe_manifest_revision=lambda _revision: manifest,
        feature_source_appends=lambda *_args, **_kwargs: (),
        raw_bar_semantic_deltas_since=lambda *_args, **_kwargs: (),
        provider_adjusted_semantic_deltas_since=(lambda *_args, **_kwargs: (revision,)),
        verified_feature_source_session_sets=lambda *_args, **_kwargs: (
            frozenset({"listing-aapl"}) if source_verified else frozenset()
        ),
        membership_schedule=lambda *_args, **_kwargs: SimpleNamespace(
            members=lambda _day: ("listing-aapl",) if historical_member else ("listing-msft",)
        ),
        current_sector_state=lambda _manifest: None,
    )
    coordinator = object.__new__(WorkspaceMaintenanceCoordinator)
    coordinator.market_data = store
    coordinator.feature_state = store
    coordinator.manifest = manifest
    coordinator.feature_foundation = SimpleNamespace(
        catalog=FeatureCatalog.load(), panel=SimpleNamespace(policy_hash="legacy")
    )

    def reconcile():
        return coordinator._unconsumed_feature_source_change_set(
            {
                "manifest_revision": manifest.revision_sha256,
                "sector_revision": "old-sector",
                "as_of_session": date(2026, 7, 31),
                "activated_at": NOW - timedelta(hours=1),
            },
            through=date(2026, 8, 3),
        )

    if source_verified and not historical_member:
        changes = reconcile()
        assert changes.listing_changes[0].adjusted_return_change_sessions == (date(2026, 7, 15),)
    else:
        with pytest.raises(ValueError, match="session-set change"):
            reconcile()


def test_feature_source_appends_require_raw_and_adjusted_authority(tmp_path: Path) -> None:
    manifest = build_quality_filtered_research_manifest(
        acquisition_manifest(),
        eligible_listing_ids=("listing-aapl",),
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(manifest)
    feature_state.ensure_current_storage()
    catalog = FeatureCatalog.load()
    catalog_hash = catalog.binding.catalog_hash
    cutoff_payload, cutoff_hash = canonical_cutoff_set(
        {factor_id: None for factor_id in catalog.factor_ids},
        factor_ids=catalog.factor_ids,
    )
    connection = market_data._connect()
    try:
        for session, value in (
            (date(2026, 7, 31), 100.0),
            (date(2026, 8, 3), 101.0),
        ):
            connection.execute(
                """
                INSERT INTO raw_daily_bar_current
                VALUES (?, 'fixture', ?, ?, ?, ?, ?, 1000000, ?, ?)
                """,
                [
                    "listing-aapl",
                    session,
                    value,
                    value,
                    value,
                    value,
                    canonical_hash(["raw", session]),
                    NOW,
                ],
            )
            connection.execute(
                """
                INSERT INTO provider_adjusted_close_current
                VALUES (?, 'fixture', ?, ?, ?, ?)
                """,
                [
                    "listing-aapl",
                    session,
                    value,
                    canonical_hash(["adjusted", session]),
                    NOW,
                ],
            )
        connection.execute(
            """
            INSERT INTO feature_input_cutoff_set VALUES (?, ?, ?)
            """,
            [cutoff_hash, cutoff_payload, len(catalog.factor_ids)],
        )
        connection.execute(
            # A Feature row names the catalog that computed it. The column is
            # NOT NULL on purpose: a row whose producing catalog is unknown is a
            # row a catalog rotation could relabel for free.
            """
            INSERT INTO feature_daily_current (
                listing_id, session_date, catalog_hash, raw_input_hash,
                action_set_hash, market_reference_revision,
                cutoff_set_hash, row_hash, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "listing-aapl",
                date(2026, 7, 31),
                catalog_hash,
                "a" * 64,
                "b" * 64,
                "c" * 64,
                cutoff_hash,
                "d" * 64,
                NOW,
            ],
        )
    finally:
        connection.close()

    appends = market_data.feature_source_appends(
        manifest,
        catalog_hash=catalog_hash,
        through=date(2026, 8, 3),
    )
    assert len(appends) == 1
    assert appends[0].sessions == (date(2026, 8, 3),)
    assert (
        market_data.feature_source_appends(
            manifest, catalog_hash=catalog_hash, through=date(2026, 8, 3), include_incomplete=True
        )
        == appends
    )
    assert market_data.verified_feature_source_session_sets(
        manifest,
        catalog_hash=catalog_hash,
        through=date(2026, 8, 3),
    ) == frozenset({"listing-aapl"})

    connection = market_data._connect()
    try:
        connection.execute(
            """
            DELETE FROM provider_adjusted_close_current
            WHERE listing_id = 'listing-aapl' AND session_date = DATE '2026-08-03'
            """
        )
    finally:
        connection.close()
    assert not market_data.verified_feature_source_session_sets(
        manifest,
        catalog_hash=catalog_hash,
        through=date(2026, 8, 3),
    )
    with pytest.raises(ValueError, match="missing provider adjusted close"):
        market_data.feature_source_appends(
            manifest,
            catalog_hash=catalog_hash,
            through=date(2026, 8, 3),
        )
    incomplete = market_data.feature_source_appends(
        manifest, catalog_hash=catalog_hash, through=date(2026, 8, 3), include_incomplete=True
    )
    assert incomplete[0].sessions == appends[0].sessions
    assert incomplete[0].evidence_hash != appends[0].evidence_hash
    with pytest.raises(ValueError, match="provider adjusted series does not cover"):
        feature_state.projected_feature_frame(
            manifest, listing_id="listing-aapl", through=date(2026, 8, 3)
        )
    projected, _, _ = feature_state.projected_feature_frame(
        manifest,
        listing_id="listing-aapl",
        through=date(2026, 8, 3),
        allow_missing_adjusted=True,
    )
    assert projected[-1]["close_raw"] == 101.0
    assert math.isnan(projected[-1]["provider_adjusted_close"])


def test_child_sector_binding_preserves_evidence_time_and_is_idempotent(
    tmp_path: Path,
) -> None:
    parent = acquisition_manifest()
    child = build_quality_filtered_research_manifest(
        parent,
        eligible_listing_ids=("listing-aapl",),
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(parent)
    market_data.bootstrap(child)
    source_observed_at = NOW - timedelta(days=10)
    feature_state.activate_sector_revision(
        manifest=parent,
        observations=tuple(
            {
                "listing_id": listing.listing_id,
                "provider": "fixture",
                "provider_symbol": listing.provider_symbol,
                "sector_name": "Technology",
                "sector_key": None,
                "payload_hash": canonical_hash(["payload", listing.listing_id]),
                "evidence_hash": canonical_hash(["evidence", listing.listing_id]),
            }
            for listing in parent.listings
        ),
        observed_at=source_observed_at,
    )
    connection = feature_state._connect(read_only=True)
    try:
        revision_count_before = connection.execute(
            "SELECT count(*) FROM sector_classification_revision"
        ).fetchone()[0]
    finally:
        connection.close()

    first = feature_state.bind_current_sector_evidence_to_manifest(child)
    second = feature_state.bind_current_sector_evidence_to_manifest(child)

    assert first is not None
    assert second == first
    state = feature_state.current_sector_state(child)
    assert state is not None
    assert state.sector_observed_at == source_observed_at
    assert not feature_state.sector_revision_refresh_due(child, observed_at=NOW, every_days=30)
    connection = feature_state._connect(read_only=True)
    try:
        receipt = connection.execute(
            """
            SELECT observed_at FROM sector_reference_receipt
            WHERE manifest_revision = ? AND receipt_hash = ?
            """,
            [child.revision_sha256, first[1]],
        ).fetchone()
        revision_count_after = connection.execute(
            "SELECT count(*) FROM sector_classification_revision"
        ).fetchone()[0]
    finally:
        connection.close()
    assert receipt[0].replace(tzinfo=UTC) == source_observed_at
    assert revision_count_after == revision_count_before


def test_partial_sector_result_quarantines_missing_listing_and_activates_safe_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = acquisition_manifest()
    listings = tuple(
        ManifestListing(
            listing_id=f"listing-{index:02d}",
            symbol=f"S{index:02d}",
            mic="XNAS",
            provider_symbol=f"S{index:02d}",
        )
        for index in range(50)
    )
    manifest = UniverseManifest(
        manifest_id="sector-partial-candidate",
        profile=source.profile,
        listings=listings,
        revision_sha256="7" * 64,
        membership_fingerprint="8" * 64,
        qualification_policy_hash="9" * 64,
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(manifest)
    feature_state.bind_feature_input_candidate(manifest, observed_at=NOW)
    observations = tuple(
        {
            "listing_id": item.listing_id,
            "provider": "fixture",
            "provider_symbol": item.provider_symbol,
            "sector_name": "Sector A" if index < 25 else "Sector B",
            "sector_key": None,
            "payload_hash": f"{index + 1:064x}",
            "evidence_hash": f"{index + 101:064x}",
        }
        for index, item in enumerate(listings[:-1])
    )
    feature_state.stage_sector_observations(
        manifest_revision=manifest.revision_sha256,
        observations=observations,
        observed_at=NOW,
    )
    monkeypatch.setattr(
        market_data,
        "bind_action_audit_receipts_to_manifest",
        lambda *_args, **_kwargs: None,
    )
    mutation_gate = WorkspaceMutationGate()
    governance = FeatureInputGovernanceService(
        market_data=market_data,
        panel_state=panel_state,
        mutation_gate=mutation_gate,
        gateway=FeatureInputGateway(),
    )
    # The partial exclusion activates through the Foundation's coordinator (V346).
    feature_foundation = SimpleNamespace(
        catalog=FeatureCatalog.load(),
        manifest=manifest,
        sector_activation=SimpleNamespace(activate=feature_state.activate_sector_revision),
    )
    coordinator = WorkspaceMaintenanceCoordinator(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        manifest=manifest,
        provider=SimpleNamespace(name="fixture"),
        mutation_gate=mutation_gate,
        readiness_gate=SimpleNamespace(),
        feature_foundation=feature_foundation,
        registry=DuckDbWorkspaceMaintenanceRegistry(market_data.path, gate=mutation_gate),
        feature_input=governance,
    )
    request = WorkspaceMaintenanceRequest.create(
        market_profile_id=manifest.profile.market_profile_id,
        target_market_session=date(2026, 7, 31),
        knowledge_cutoff_at=NOW,
        trigger=MaintenanceTrigger.ONBOARDING,
        membership_revision=manifest.revision_sha256,
        data_policy_hash="d" * 64,
        feature_policy_hash="e" * 64,
    )

    # A preview timestamp cannot authorize later source observations. Fixed
    # cutoffs still refuse; an explicit acquisition request captures afterward.
    coordinator.clock = lambda: NOW + timedelta(minutes=5)
    with pytest.raises(ValueError, match="sector observation is after"):
        coordinator._apply_sector_partial_exclusion(request=request, observed_at=NOW)
    request = WorkspaceMaintenanceRequest.create(
        market_profile_id=request.market_profile_id,
        target_market_session=request.target_market_session,
        knowledge_cutoff_at=None,
        requested_at=NOW,
        trigger=request.trigger,
        membership_revision=request.membership_revision,
        data_policy_hash=request.data_policy_hash,
        feature_policy_hash=request.feature_policy_hash,
    )
    assert request.request_clock == NOW

    # A member of the reduced child without a reusable audit receipt for the
    # session refuses the exclusion instead of escaping and interrupting the
    # Task, and the cycle stops by a cause the user can act on: the evidence
    # shortfall by name, or the day's failed refresh when members of the
    # reduction failed it (a stale payload seals no audit).
    def refuse_binding(*_args, **_kwargs):
        raise ValueError(
            "source manifest has no reusable action-audit receipt: listing-07 as of 2026-07-31"
        )

    binding = market_data.bind_action_audit_receipts_to_manifest
    monkeypatch.setattr(market_data, "bind_action_audit_receipts_to_manifest", refuse_binding)
    assert (
        coordinator._apply_sector_partial_exclusion(request=request, observed_at=NOW)
        == "workspace_maintenance.derived_manifest_evidence_incomplete"
    )
    assert coordinator.manifest is manifest and len(coordinator.manifest.listings) == 50
    from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
        current_universe_maintenance_id,
    )

    maintenance_id = current_universe_maintenance_id(
        manifest,
        as_of_session=request.target_market_session,
        authorized_full_history_listing_ids=(),
    )
    market_data.admit_current_universe_maintenance(
        manifest,
        maintenance_id=maintenance_id,
        as_of_session=request.target_market_session,
        observed_at=NOW,
    )
    market_data.update_current_universe_maintenance_listing(
        maintenance_id=maintenance_id,
        listing_id=listings[0].listing_id,
        state="FAILED",
        failure_code="data.maintenance_stale_payload",
        observed_at=NOW,
    )
    assert (
        coordinator._apply_sector_partial_exclusion(request=request, observed_at=NOW)
        == "data.listing_updates_incomplete"
    )
    monkeypatch.setattr(market_data, "bind_action_audit_receipts_to_manifest", binding)
    assert (
        coordinator._apply_sector_partial_exclusion(
            request=request,
            observed_at=NOW,
        )
        is None
    )
    assert len(coordinator.manifest.listings) == 49
    assert feature_state.current_sector_state(coordinator.manifest) is not None
    quarantines = panel_state.active_listing_quarantines(manifest.revision_sha256)
    assert len(quarantines) == 1
    assert quarantines[0].listing_id == listings[-1].listing_id
    disclosure = panel_state.feature_input_quality_disclosure(
        result_manifest_revision=coordinator.manifest.revision_sha256
    )
    assert disclosure["gateway_qualified"] is False  # Sector evidence is not raw-quality clearance.
    assert disclosure["quality_exclusion_count"] == 1
    admission_scope = dict(
        candidate_revision=manifest.revision_sha256,
        result_revision=coordinator.manifest.revision_sha256,
        as_of_session=request.target_market_session,
    )
    assert not panel_state.admits_feature_candidate_subset(**admission_scope)
    assert panel_state.admits_feature_candidate_subset(
        **admission_scope, allow_sector_preparation=True
    )
    assert not panel_state.admits_feature_candidate_subset(
        **admission_scope, allow_sector_preparation=True, qualification_hash="a" * 64
    )

    # Once these names are U0 members, the same partial Sector evidence must
    # not derive a smaller active manifest that a later bind could journal.
    from alphalattice.foundation.market_data_ops.sources.membership import UniverseBootstrapRecord

    market_data.record_universe_bootstrap(
        UniverseBootstrapRecord(
            market_profile_id=manifest.profile.market_profile_id,
            t0_session=request.target_market_session,
            history_start=date(2016, 8, 1),
            cohort_listing_ids=tuple(item.listing_id for item in manifest.listings),
            cohort_hash="",
            manifest_revision=manifest.revision_sha256,
            candidate_manifest_hash="1" * 64,
            qualification_policy_hash="2" * 64,
            feature_input_policy_hash="3" * 64,
            source_observed_at=NOW,
            admitted_at=NOW,
            panel_snapshot_hash="4" * 64,
            derivation="FIRST_QUALIFIED_PUBLICATION",
        )
    )
    coordinator.manifest = manifest
    feature_state.stage_sector_observations(
        manifest_revision=manifest.revision_sha256, observations=observations, observed_at=NOW
    )
    assert (
        coordinator._apply_sector_partial_exclusion(request=request, observed_at=NOW)
        == "sector.partial_current_sector"
    )
    assert coordinator.manifest == manifest
    assert market_data.membership_events(manifest.profile.market_profile_id) == ()


def _baseline_qualification_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, sector_observed_at: datetime
):
    """A coordinator over a real repository whose Sector was activated at one instant,
    with the qualification owner and the admission gateway replaced by recorders, so the
    boundary sealed for the baseline judgement can be read back exactly."""

    source = acquisition_manifest()
    listings = tuple(
        ManifestListing(
            listing_id=f"listing-{index:02d}",
            symbol=f"S{index:02d}",
            mic="XNAS",
            provider_symbol=f"S{index:02d}",
        )
        for index in range(12)
    )
    manifest = UniverseManifest(
        manifest_id="baseline-candidate",
        profile=source.profile,
        listings=listings,
        revision_sha256="7" * 64,
        membership_fingerprint="8" * 64,
        qualification_policy_hash="9" * 64,
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(manifest)
    feature_state.bind_feature_input_candidate(manifest, observed_at=NOW)
    feature_state.activate_sector_revision(
        manifest=manifest,
        observations=tuple(
            {
                "listing_id": item.listing_id,
                "provider": "fixture",
                "provider_symbol": item.provider_symbol,
                "sector_name": "Sector A" if index < 6 else "Sector B",
                "sector_key": None,
                "payload_hash": f"{index + 1:064x}",
                "evidence_hash": f"{index + 101:064x}",
            }
            for index, item in enumerate(listings)
        ),
        observed_at=sector_observed_at,
    )
    mutation_gate = WorkspaceMutationGate()
    bundle = desktop_core_feature_bundle()
    seen: dict[str, object] = {}

    def qualify_features(**kwargs):
        seen["qualify_observed_at"] = kwargs["observed_at"]
        return MaterializedFeatureQualification(
            session=kwargs["as_of_session"],
            catalog_hash="c" * 64,
            factor_ids=tuple(sorted(bundle.factor_ids)),
            eligible_listing_ids=tuple(kwargs["listing_ids"]),
            pending_listing_ids=(),
            exclusions=(),
            evidence_hash="e" * 64,
        )

    def admit_feature_baseline(**kwargs):
        seen["boundary"] = kwargs["temporal_boundary"]
        seen["admit_observed_at"] = kwargs["observed_at"]
        return SimpleNamespace(research_manifest=manifest)

    coordinator = WorkspaceMaintenanceCoordinator(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        manifest=manifest,
        provider=SimpleNamespace(name="fixture"),
        mutation_gate=mutation_gate,
        readiness_gate=SimpleNamespace(),
        feature_foundation=SimpleNamespace(
            catalog=FeatureCatalog.load(), manifest=manifest, qualify_features=qualify_features
        ),
        registry=DuckDbWorkspaceMaintenanceRegistry(market_data.path, gate=mutation_gate),
        feature_input=SimpleNamespace(admit_feature_baseline=admit_feature_baseline),
    )
    monkeypatch.setattr(
        coordinator, "_bind_derived_manifest_evidence", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(coordinator, "_bind_manifest", lambda *_args, **_kwargs: None)
    return coordinator, manifest, seen


def _baseline_request(manifest: UniverseManifest, *, knowledge_cutoff_at: datetime | None):
    return WorkspaceMaintenanceRequest.create(
        market_profile_id=manifest.profile.market_profile_id,
        target_market_session=date(2026, 7, 31),
        knowledge_cutoff_at=knowledge_cutoff_at,
        requested_at=None if knowledge_cutoff_at is not None else NOW,
        trigger=MaintenanceTrigger.ONBOARDING,
        membership_revision=manifest.revision_sha256,
        data_policy_hash="d" * 64,
        feature_policy_hash="e" * 64,
    )


def test_baseline_qualification_is_sealed_at_the_assessment_instant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first-use cycle enters at t, its Feature build stamps the Sector at t+1, the
    judgement runs at t+2: the boundary is sealed at the assessment instant (t+2), so
    the observation is admitted; sealed at the cycle's entry time it was refused as if
    from the future (2026-09-15, run 1 of the first-initialization batch)."""

    coordinator, manifest, seen = _baseline_qualification_fixture(
        tmp_path, monkeypatch, sector_observed_at=NOW + timedelta(minutes=1)
    )
    coordinator.clock = lambda: NOW + timedelta(minutes=2)
    request = _baseline_request(manifest, knowledge_cutoff_at=None)
    outcome = coordinator._qualify_features_for_membership(request=request, observed_at=NOW)
    assert outcome == (MaintenanceStatus.RUNNING, "feature.baseline_qualification_applied", None)
    boundary = seen["boundary"]
    assert boundary.knowledge_cutoff_at == NOW + timedelta(minutes=2)
    assert boundary.materialized_at == NOW + timedelta(minutes=2)
    assert boundary.sector_source_observed_at == NOW + timedelta(minutes=1)
    assert boundary.market_as_of_session == date(2026, 7, 31)
    assert seen["qualify_observed_at"] == NOW + timedelta(minutes=2)
    assert seen["admit_observed_at"] == NOW + timedelta(minutes=2)
    # Without a coordinator clock the caller's instant is the assessment instant, as before.
    coordinator.clock = None
    seen.clear()
    with pytest.raises(ValueError, match="sector observation is after the knowledge cutoff"):
        coordinator._qualify_features_for_membership(request=request, observed_at=NOW)
    assert "boundary" not in seen


def test_baseline_qualification_keeps_an_explicit_cutoff_and_refuses_future_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refreshed instant never loosens the guards: an explicit request cutoff still
    refuses a Sector observed after it, and a Sector observed after the assessment
    instant itself is still from the future."""

    coordinator, manifest, seen = _baseline_qualification_fixture(
        tmp_path, monkeypatch, sector_observed_at=NOW + timedelta(minutes=1)
    )
    coordinator.clock = lambda: NOW + timedelta(minutes=2)
    fixed = _baseline_request(manifest, knowledge_cutoff_at=NOW)
    with pytest.raises(ValueError, match="sector observation is after the knowledge cutoff"):
        coordinator._qualify_features_for_membership(request=fixed, observed_at=NOW)
    assert "boundary" not in seen
    # An explicit cutoff after the observation binds exactly, at the assessment instant.
    later = _baseline_request(manifest, knowledge_cutoff_at=NOW + timedelta(minutes=1))
    outcome = coordinator._qualify_features_for_membership(request=later, observed_at=NOW)
    assert outcome == (MaintenanceStatus.RUNNING, "feature.baseline_qualification_applied", None)
    assert seen["boundary"].knowledge_cutoff_at == NOW + timedelta(minutes=1)
    assert seen["boundary"].materialized_at == NOW + timedelta(minutes=2)
    # A source observed after the assessment instant is refused.
    coordinator, manifest, seen = _baseline_qualification_fixture(
        tmp_path / "future", monkeypatch, sector_observed_at=NOW + timedelta(minutes=3)
    )
    coordinator.clock = lambda: NOW + timedelta(minutes=2)
    with pytest.raises(ValueError, match="sector observation is after the knowledge cutoff"):
        coordinator._qualify_features_for_membership(
            request=_baseline_request(manifest, knowledge_cutoff_at=None), observed_at=NOW
        )
    assert "boundary" not in seen


def test_daily_transport_workers_fetch_only_and_main_thread_persists(
    tmp_path: Path,
) -> None:
    manifest, market_data, _, _ = _maintenance_workspace(
        tmp_path, eligible_listing_ids=("listing-aapl", "listing-msft")
    )
    initial = {
        symbol: tuple(
            {
                "session_date": session.isoformat(),
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000_000,
            }
            for session, close in (
                (date(2026, 7, 29), 100.0),
                (date(2026, 7, 30), 101.0),
            )
        )
        for symbol in ("AAPL", "MSFT")
    }
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", initial, ("AAPL", "MSFT")),
        ingestion_id="initial",
        observed_at=NOW,
    )
    connection = market_data._connect()
    try:
        connection.execute(
            """
            INSERT INTO provider_adjusted_close_current
            SELECT listing_id, provider, session_date, close,
                   sha256(concat_ws('|', listing_id, CAST(session_date AS VARCHAR), close)), ?
            FROM raw_daily_bar_current
            """,
            [NOW.replace(tzinfo=None)],
        )
    finally:
        connection.close()

    class ConcurrentFixtureProvider:
        name = "fixture"

        def __init__(self) -> None:
            self.barrier = Barrier(2)
            self.worker_ids: set[int] = set()

        def fetch_hydration(self, *, listing_id, provider_symbol, start, end):
            self.worker_ids.add(get_ident())
            self.barrier.wait(timeout=2.0)
            rows = tuple(
                {
                    "session_date": session.isoformat(),
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000,
                }
                for session, close in (
                    (date(2026, 7, 29), 100.0),
                    (date(2026, 7, 30), 101.0),
                    (date(2026, 7, 31), 102.0),
                )
                if start <= session <= end
            )
            return HydrationEvidence(
                daily_rows=rows,
                actions=(),
                adjusted_closes=tuple(
                    ProviderAdjustedClosePoint(
                        listing_id=listing_id,
                        provider="fixture",
                        session_date=date.fromisoformat(str(row["session_date"])),
                        adjusted_close=float(row["close"]),
                    )
                    for row in rows
                ),
            )

    provider = ConcurrentFixtureProvider()
    main_thread = get_ident()
    outcome = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=provider,
        as_of_session=date(2026, 7, 31),
        max_workers=2,
        mutation_gate=WorkspaceMutationGate(),
    ).run(observed_at=NOW)
    assert outcome.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert len(provider.worker_ids) == 2
    assert main_thread not in provider.worker_ids
    assert all(
        market_data.raw_bars(item.listing_id)[-1].session_date == date(2026, 7, 31)
        for item in manifest.listings
    )
    # The session-only projection the Feature build plans on is the same
    # ordered axis as the bars, with and without the through bound.
    for item in manifest.listings:
        bars = market_data.raw_bars(item.listing_id)
        assert market_data.raw_bar_sessions(item.listing_id) == tuple(
            bar.session_date for bar in bars
        )
        assert market_data.raw_bar_sessions(item.listing_id, through=date(2026, 7, 30)) == tuple(
            bar.session_date for bar in bars if bar.session_date <= date(2026, 7, 30)
        )
    assert market_data.manifest_raw_through(manifest) == date(2026, 7, 31)
    assert market_data.manifest_provider_adjusted_through(manifest) == date(2026, 7, 31)


def test_run_releases_its_hold_around_the_provider_fetch_and_tells_the_caller(
    tmp_path: Path,
) -> None:
    """requirement: nothing is retained while the Provider is awaited.

    The runner retains the workspace instance for its bookkeeping before and
    after the fetch, never across it: a reader on another thread opens the
    file during the fetch without waiting, ``before_fetch`` runs once before
    the first fetch and ``after_fetch`` once after the last, so a caller can
    release its own hold for exactly that window and take it back afterwards.
    """

    manifest, market_data, _, _ = _maintenance_workspace(
        tmp_path, eligible_listing_ids=("listing-aapl", "listing-msft")
    )
    initial = {
        symbol: tuple(
            {
                "session_date": session.isoformat(),
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000_000,
            }
            for session, close in ((date(2026, 7, 29), 100.0), (date(2026, 7, 30), 101.0))
        )
        for symbol in ("AAPL", "MSFT")
    }
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", initial, ("AAPL", "MSFT")),
        ingestion_id="initial",
        observed_at=NOW,
    )
    edges: list[str] = []

    class EdgeProvider:
        name = "fixture"

        def fetch_hydration(self, *, listing_id, provider_symbol, start, end):
            # A reader on another thread is not kept waiting by the run.
            open_workspace_database(market_data.path, read_only=True, wait_seconds=0.0).close()
            edges.append(f"fetch:{provider_symbol}")
            rows = tuple(
                {
                    "session_date": session.isoformat(),
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000,
                }
                for session, close in (
                    (date(2026, 7, 29), 100.0),
                    (date(2026, 7, 30), 101.0),
                    (date(2026, 7, 31), 102.0),
                )
                if start <= session <= end
            )
            return HydrationEvidence(
                daily_rows=rows,
                actions=(),
                adjusted_closes=tuple(
                    ProviderAdjustedClosePoint(
                        listing_id=listing_id,
                        provider="fixture",
                        session_date=date.fromisoformat(str(row["session_date"])),
                        adjusted_close=float(row["close"]),
                    )
                    for row in rows
                ),
            )

    outcome = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=EdgeProvider(),
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        mutation_gate=WorkspaceMutationGate(),
    ).run(
        observed_at=NOW,
        before_fetch=lambda: edges.append("before_fetch"),
        after_fetch=lambda: edges.append("after_fetch"),
    )

    assert outcome.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert (outcome.updated, outcome.failed) == (2, 0)
    assert edges[0] == "before_fetch"
    assert sorted(edges[1:3]) == ["fetch:AAPL", "fetch:MSFT"]
    assert edges[3:] == ["after_fetch"]


def test_raw_coverage_cannot_hide_stale_provider_adjusted_authority(tmp_path: Path) -> None:
    manifest, market_data, _, _ = _maintenance_workspace(
        tmp_path, eligible_listing_ids=("listing-aapl", "listing-msft")
    )
    rows = {
        symbol: tuple(
            {
                "session_date": session.isoformat(),
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000_000,
            }
            for session, close in (
                (date(2026, 7, 30), 100.0),
                (date(2026, 7, 31), 101.0),
            )
        )
        for symbol in ("AAPL", "MSFT")
    }
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", rows, ("AAPL", "MSFT")),
        ingestion_id="raw-current",
        observed_at=NOW,
    )
    connection = market_data._connect()
    try:
        connection.execute(
            """
            INSERT INTO provider_adjusted_close_current
            SELECT listing_id, provider, session_date, close,
                   sha256(concat_ws('|', listing_id, CAST(session_date AS VARCHAR), close)), ?
            FROM raw_daily_bar_current WHERE session_date = '2026-07-30'
            """,
            [NOW.replace(tzinfo=None)],
        )
    finally:
        connection.close()

    assert market_data.manifest_raw_through(manifest) == date(2026, 7, 31)
    assert market_data.manifest_provider_adjusted_through(manifest) == date(2026, 7, 30)


def test_missing_adjusted_target_is_an_append_even_when_raw_target_already_exists(
    tmp_path: Path,
) -> None:
    manifest, market_data, _, _ = _maintenance_workspace(tmp_path)
    rows = tuple(
        {
            "session_date": session.isoformat(),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000,
        }
        for session, close in (
            (date(2026, 7, 29), 100.0),
            (date(2026, 7, 30), 101.0),
            (date(2026, 7, 31), 102.0),
        )
    )
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", {"AAPL": rows}, ("AAPL",)),
        ingestion_id="raw-current-adjusted-stale",
        observed_at=NOW,
    )
    connection = market_data._connect()
    try:
        connection.execute(
            """
            INSERT INTO provider_adjusted_close_current
            SELECT listing_id, provider, session_date, close,
                   sha256(concat_ws('|', listing_id, CAST(session_date AS VARCHAR), close)), ?
            FROM raw_daily_bar_current WHERE session_date < '2026-07-31'
            """,
            [NOW.replace(tzinfo=None)],
        )
    finally:
        connection.close()

    class FixtureProvider:
        name = "fixture"

        def fetch_hydration(self, *, listing_id, provider_symbol, start, end):
            del provider_symbol
            observed = tuple(
                row for row in rows if start <= date.fromisoformat(str(row["session_date"])) <= end
            )
            return HydrationEvidence(
                daily_rows=observed,
                actions=(),
                adjusted_closes=tuple(
                    ProviderAdjustedClosePoint(
                        listing_id=listing_id,
                        provider=self.name,
                        session_date=date.fromisoformat(str(row["session_date"])),
                        adjusted_close=float(row["close"]),
                    )
                    for row in observed
                ),
            )

    outcome = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=FixtureProvider(),
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        mutation_gate=WorkspaceMutationGate(),
    ).run(observed_at=NOW)

    listing = market_data.current_universe_maintenance_listings(outcome.maintenance_id)[0]
    assert (outcome.updated, outcome.failed) == (1, 0)
    assert listing.state == "UPDATED"
    assert market_data.manifest_provider_adjusted_through(manifest) == date(2026, 7, 31)


def test_failed_listing_gets_one_new_bounded_attempt_then_exactly_reuses(
    tmp_path: Path,
) -> None:
    manifest, market_data, _, _ = _maintenance_workspace(tmp_path)
    initial_rows = tuple(
        {
            "session_date": session.isoformat(),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000,
        }
        for session, close in (
            (date(2026, 7, 29), 100.0),
            (date(2026, 7, 30), 101.0),
        )
    )
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", {"AAPL": initial_rows}, ("AAPL",)),
        ingestion_id="initial",
        observed_at=NOW,
    )

    class RecoveringProvider:
        name = "fixture"

        def __init__(self) -> None:
            self.calls = 0

        def fetch_hydration(self, *, listing_id, provider_symbol, start, end):
            del provider_symbol
            self.calls += 1
            rows = tuple(
                {
                    "session_date": session.isoformat(),
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": (1_000_000 if self.calls > 1 else 1_000_000.5),
                }
                for session, close in (
                    (date(2026, 7, 29), 100.0),
                    (date(2026, 7, 30), 101.0),
                    (date(2026, 7, 31), 102.0),
                )
                if start <= session <= end
            )
            return HydrationEvidence(
                daily_rows=rows,
                actions=(),
                adjusted_closes=tuple(
                    ProviderAdjustedClosePoint(
                        listing_id=listing_id,
                        provider=self.name,
                        session_date=date.fromisoformat(str(row["session_date"])),
                        adjusted_close=float(row["close"]),
                    )
                    for row in rows
                ),
            )

    provider = RecoveringProvider()
    progress = []
    maintenance = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=provider,
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        mutation_gate=WorkspaceMutationGate(),
        progress_sink=progress.append,
    )

    first = maintenance.run(observed_at=NOW)
    assert first.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert first.failed == 1
    failed = market_data.current_universe_maintenance_listings(first.maintenance_id)[0]
    assert (failed.state, failed.attempt_count) == ("FAILED", 1)

    second = maintenance.run(observed_at=NOW + timedelta(minutes=5))
    assert second.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert (second.updated, second.failed) == (1, 0)
    recovered = market_data.current_universe_maintenance_listings(second.maintenance_id)[0]
    assert (recovered.state, recovered.attempt_count) == ("UPDATED", 2)

    third = maintenance.run(observed_at=NOW + timedelta(minutes=10))
    assert third.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert provider.calls == 2
    assert progress[-1].status == "REUSED_EXACT"
    assert progress[-1].completed_units == progress[-1].total_units == 1


def test_failed_listing_stops_after_two_bounded_attempts(tmp_path: Path) -> None:
    manifest, market_data, _, _ = _maintenance_workspace(tmp_path)
    rows = tuple(
        {
            "session_date": session.isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
        }
        for session in (date(2026, 7, 29), date(2026, 7, 30))
    )
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", {"AAPL": rows}, ("AAPL",)),
        ingestion_id="initial",
        observed_at=NOW,
    )

    class InvalidProvider:
        name = "fixture"

        def __init__(self) -> None:
            self.calls = 0

        def fetch_hydration(self, *, listing_id, provider_symbol, start, end):
            del provider_symbol
            self.calls += 1
            invalid = (
                {
                    "session_date": date(2026, 7, 31).isoformat(),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1_000_000.5,
                },
            )
            return HydrationEvidence(
                daily_rows=invalid,
                actions=(),
                adjusted_closes=(
                    ProviderAdjustedClosePoint(
                        listing_id=listing_id,
                        provider=self.name,
                        session_date=date(2026, 7, 31),
                        adjusted_close=100.0,
                    ),
                ),
            )

    provider = InvalidProvider()
    maintenance = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=provider,
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        mutation_gate=WorkspaceMutationGate(),
    )
    for minutes in (0, 5, 10):
        outcome = maintenance.run(observed_at=NOW + timedelta(minutes=minutes))
    listing = market_data.current_universe_maintenance_listings(outcome.maintenance_id)[0]
    assert (listing.state, listing.attempt_count) == ("FAILED", 2)
    assert provider.calls == 2


def test_interrupted_pending_listing_at_attempt_limit_is_not_fetched(
    tmp_path: Path,
) -> None:
    manifest, market_data, _, _ = _maintenance_workspace(tmp_path)

    class ExplodingProvider:
        name = "fixture"

        def fetch_hydration(self, **_kwargs):
            raise AssertionError("an exhausted interrupted listing reached the Provider")

    maintenance = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=ExplodingProvider(),
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        mutation_gate=WorkspaceMutationGate(),
    )
    maintenance.admit(observed_at=NOW)
    for minute in (0, 1):
        market_data.begin_current_universe_maintenance_listing_attempt(
            maintenance_id=maintenance.maintenance_id,
            listing_id="listing-aapl",
            observed_at=NOW + timedelta(minutes=minute),
        )

    outcome = maintenance.run(observed_at=NOW + timedelta(minutes=2))

    assert outcome.status is CurrentUniverseMaintenanceStatus.COMPLETED
    listing = market_data.current_universe_maintenance_listings(maintenance.maintenance_id)[0]
    assert listing.state == "FAILED"
    assert listing.attempt_count == 2
    assert listing.failure_code == ("data.listing_attempt_budget_exhausted_after_interruption")


def _two_exhausted_listings(tmp_path: Path):
    """Both names past the attempt budget, failed by a stale payload, with raw history."""

    manifest, market_data, _, _ = _maintenance_workspace(
        tmp_path, eligible_listing_ids=("listing-aapl", "listing-msft")
    )
    rows = tuple(
        {
            "session_date": session.isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
        }
        for session in (date(2026, 7, 29), date(2026, 7, 30))
    )
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", {"AAPL": rows, "MSFT": rows}, ("AAPL", "MSFT")),
        ingestion_id="initial",
        observed_at=NOW,
    )

    class CountingProvider:
        name = "fixture"

        def __init__(self) -> None:
            self.calls: list[str] = []
            self.stale: set[str] = set()

        def fetch_hydration(self, *, listing_id, provider_symbol, start, end):
            del provider_symbol
            self.calls.append(listing_id)
            sessions = (date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31))
            if listing_id in self.stale:
                sessions = sessions[:2]  # a stale payload: nothing for the session
            rows = tuple(
                {
                    "session_date": session.isoformat(),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1_000_000,
                }
                for session in sessions
                if start <= session <= end
            )
            return HydrationEvidence(
                daily_rows=rows,
                actions=(),
                adjusted_closes=tuple(
                    ProviderAdjustedClosePoint(
                        listing_id=listing_id,
                        provider=self.name,
                        session_date=date.fromisoformat(str(row["session_date"])),
                        adjusted_close=float(row["close"]),
                    )
                    for row in rows
                ),
            )

    provider = CountingProvider()

    def runner(**grants):
        return CurrentUniverseMaintenance(
            store=market_data,
            manifest=manifest,
            provider=provider,
            as_of_session=date(2026, 7, 31),
            max_workers=1,
            mutation_gate=WorkspaceMutationGate(),
            retry_grants=grants,
        )

    seed = runner()
    seed.admit(observed_at=NOW)
    for listing_id in ("listing-aapl", "listing-msft"):
        for minute in (0, 1):
            market_data.begin_current_universe_maintenance_listing_attempt(
                maintenance_id=seed.maintenance_id,
                listing_id=listing_id,
                observed_at=NOW + timedelta(minutes=minute),
            )
        market_data.update_current_universe_maintenance_listing(
            maintenance_id=seed.maintenance_id,
            listing_id=listing_id,
            state="FAILED",
            failure_code="data.maintenance_stale_payload",
            observed_at=NOW + timedelta(minutes=2),
        )
    market_data.set_current_universe_maintenance_lifecycle(
        seed.maintenance_id, lifecycle="COMPLETED", observed_at=NOW + timedelta(minutes=2)
    )
    return market_data, provider, runner, seed.maintenance_id


def _listing_states(market_data: MarketDataRepository, maintenance_id: str):
    return {
        item.listing_id: (
            item.state,
            item.attempt_count,
            item.failure_code,
            None if item.retry_grant is None else ("held" if item.retry_grant.held else "consumed"),
        )
        for item in market_data.current_universe_maintenance_listings(maintenance_id)
    }


def test_granted_retry_survives_the_batch_boundary_and_a_restart(tmp_path: Path) -> None:
    """counterexample: an unconsumed grant must not be lost once the unit is enqueued.

    Two names have spent the operation's two attempts on a stale payload;
    each carries one confirmed, elapsed wait. With a work budget of one
    unit, the first batch runs the first name. The second name is then
    pending past the budget: its grant is held on its row, so the next
    batch -- under a fresh runner, the shape of a restart between enqueue
    and attempt -- still runs it instead of marking it exhausted. Nothing
    resets ``attempt_count``; the attempt itself consumes the grant.
    """

    market_data, provider, runner, maintenance_id = _two_exhausted_listings(tmp_path)
    wait = NOW + timedelta(minutes=7)
    grants = {
        "listing-aapl": ("1" * 64, wait),
        "listing-msft": ("2" * 64, wait),
    }
    first = runner(**grants).run(observed_at=NOW + timedelta(minutes=8), work_budget=1)
    assert first.status is CurrentUniverseMaintenanceStatus.RUNNING
    assert provider.calls == ["listing-aapl"]
    assert _listing_states(market_data, maintenance_id) == {
        "listing-aapl": ("UPDATED", 3, None, "consumed"),
        "listing-msft": ("PENDING", 2, None, "held"),
    }
    # A fresh owner, and a cycle whose own view names no grant at all: the
    # row's held grant is the authority.
    second = runner().run(observed_at=NOW + timedelta(minutes=9), work_budget=1)
    assert second.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert provider.calls == ["listing-aapl", "listing-msft"]
    assert _listing_states(market_data, maintenance_id) == {
        "listing-aapl": ("UPDATED", 3, None, "consumed"),
        "listing-msft": ("UPDATED", 3, None, "consumed"),
    }
    # The same waits grant nothing again: exact reuse, no fetch.
    third = runner(**grants).run(observed_at=NOW + timedelta(minutes=10))
    assert third.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert provider.calls == ["listing-aapl", "listing-msft"]


def test_a_consumed_grant_is_not_spent_again_and_a_new_wait_grants_once_more(
    tmp_path: Path,
) -> None:
    """recovery: after the granted attempt began, no restart or continuation re-grants it.

    The granted attempt fails again (still stale): the row is failed with
    three attempts and the wait's grant consumed. The same wait, offered
    again by a later cycle or a restart, reopens nothing; a second confirmed
    wait (another receipt and instant) grants exactly one more attempt. An
    attempt interrupted after it began is exhausted by the existing rule,
    and a non-retryable failure is never reopened, granted or not.
    """

    market_data, provider, runner, maintenance_id = _two_exhausted_listings(tmp_path)
    provider.stale = {"listing-msft"}
    wait = NOW + timedelta(minutes=7)
    grants = {"listing-msft": ("2" * 64, wait)}
    outcome = runner(**grants).run(observed_at=NOW + timedelta(minutes=8))
    assert outcome.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert provider.calls == ["listing-msft"]
    assert _listing_states(market_data, maintenance_id)["listing-msft"] == (
        "FAILED",
        3,
        "data.maintenance_stale_payload",
        "consumed",
    )
    for minute in (9, 10):  # the same wait, a later cycle and a fresh owner
        again = runner(**grants).run(observed_at=NOW + timedelta(minutes=minute))
        assert again.status is CurrentUniverseMaintenanceStatus.COMPLETED
        assert provider.calls == ["listing-msft"]
    assert _listing_states(market_data, maintenance_id)["listing-msft"][1] == 3
    # A second confirmed wait: one more attempt, then the Provider recovers.
    provider.stale = set()
    second_wait = {"listing-msft": ("2" * 64, NOW + timedelta(minutes=15))}
    recovered = runner(**second_wait).run(observed_at=NOW + timedelta(minutes=16))
    assert recovered.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert provider.calls == ["listing-msft", "listing-msft"]
    assert _listing_states(market_data, maintenance_id)["listing-msft"] == (
        "UPDATED",
        4,
        None,
        "consumed",
    )
    # Interrupted after the granted attempt began: the grant is spent, the
    # existing exhaustion rule applies, and the Provider is not reached.
    third_wait = {"listing-aapl": ("1" * 64, NOW + timedelta(minutes=20))}
    market_data.requeue_failed_current_universe_maintenance_listings(
        maintenance_id=maintenance_id,
        maximum_attempts=2,
        retry_grants=third_wait,
        observed_at=NOW + timedelta(minutes=21),
    )
    market_data.begin_current_universe_maintenance_listing_attempt(
        maintenance_id=maintenance_id,
        listing_id="listing-aapl",
        observed_at=NOW + timedelta(minutes=21),
    )
    interrupted = runner(**third_wait).run(observed_at=NOW + timedelta(minutes=22))
    assert interrupted.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert provider.calls == ["listing-msft", "listing-msft"]
    assert _listing_states(market_data, maintenance_id)["listing-aapl"] == (
        "FAILED",
        3,
        "data.listing_attempt_budget_exhausted_after_interruption",
        "consumed",
    )
    # Non-retryable now: a fresh wait cannot reopen it.
    fourth_wait = {"listing-aapl": ("1" * 64, NOW + timedelta(minutes=30))}
    assert (
        market_data.requeue_failed_current_universe_maintenance_listings(
            maintenance_id=maintenance_id,
            maximum_attempts=2,
            excluded_failure_codes=("data.listing_attempt_budget_exhausted_after_interruption",),
            retry_grants=fourth_wait,
            observed_at=NOW + timedelta(minutes=31),
        )
        == ()
    )


def test_a_table_written_before_grants_reads_without_one_and_is_not_migrated_by_reading(
    tmp_path: Path,
) -> None:
    """compatibility: the plan and the pages read old maintenance tables without writing.

    A workspace whose listing table predates the grant columns (the shape
    bootstrap wrote before them) reads back with its attempts and no
    grant; reading adds no column. A row carrying only half a grant --
    its receipt without its wait -- is no grant either: it cannot exempt
    the unit from the attempt budget. Once the runner's own admission has
    upgraded the table, a granted wait is held and spent as on a new one.
    """

    market_data, provider, runner, maintenance_id = _two_exhausted_listings(tmp_path)
    connection = market_data.database.connect(read_only=False)
    try:
        for column in (
            "retry_grant_consumed_at",
            "retry_grant_wait_until",
            "retry_grant_receipt_hash",
        ):
            connection.execute(
                f"ALTER TABLE current_universe_maintenance_listing DROP COLUMN {column}"
            )
    finally:
        connection.close()

    def columns() -> set[str]:
        handle = market_data.database.connect(read_only=True)
        try:
            return {
                str(row[1])
                for row in handle.execute(
                    "PRAGMA table_info('current_universe_maintenance_listing')"
                ).fetchall()
            }
        finally:
            handle.close()

    before = columns()
    assert "retry_grant_receipt_hash" not in before and "attempt_count" in before
    assert _listing_states(market_data, maintenance_id) == {
        "listing-aapl": ("FAILED", 2, "data.maintenance_stale_payload", None),
        "listing-msft": ("FAILED", 2, "data.maintenance_stale_payload", None),
    }
    assert columns() == before
    # Over the old table, the budget still governs: nothing is granted,
    # nothing is fetched, both units are exhausted.
    exhausted = runner().run(observed_at=NOW + timedelta(minutes=8))
    assert exhausted.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert provider.calls == []
    # The runner's admission upgraded the table; a granted wait is now held
    # on the row and spent by the attempt, exactly as on a new workspace.
    assert "retry_grant_receipt_hash" in columns()
    wait = NOW + timedelta(minutes=9)
    granted = runner(**{"listing-aapl": ("1" * 64, wait)}).run(
        observed_at=NOW + timedelta(minutes=10)
    )
    assert granted.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert provider.calls == ["listing-aapl"]
    assert _listing_states(market_data, maintenance_id)["listing-aapl"] == (
        "UPDATED",
        3,
        None,
        "consumed",
    )
    # Half a grant is no grant: a receipt whose wait is missing exempts nothing.
    connection = market_data.database.connect(read_only=False)
    try:
        connection.execute(
            """
            UPDATE current_universe_maintenance_listing
            SET state = 'FAILED', failure_code = 'data.maintenance_stale_payload',
                retry_grant_receipt_hash = ?, retry_grant_wait_until = NULL,
                retry_grant_consumed_at = NULL
            WHERE maintenance_id = ? AND listing_id = 'listing-msft'
            """,
            ["2" * 64, maintenance_id],
        )
    finally:
        connection.close()
    assert _listing_states(market_data, maintenance_id)["listing-msft"] == (
        "FAILED",
        2,
        "data.maintenance_stale_payload",
        None,
    )
    half = runner().run(observed_at=NOW + timedelta(minutes=11))
    assert half.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert provider.calls == ["listing-aapl"]


def test_full_history_requirement_never_becomes_fetch_authority(tmp_path: Path) -> None:
    manifest, market_data, _, _ = _maintenance_workspace(tmp_path)
    rows = tuple(
        {
            "session_date": session.isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
        }
        for session in (date(2026, 7, 29), date(2026, 7, 30))
    )
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", {"AAPL": rows}, ("AAPL",)),
        ingestion_id="initial",
        observed_at=NOW,
    )

    class ExplodingProvider:
        name = "fixture"

        def fetch_hydration(self, **_kwargs):
            raise AssertionError("a full-history requirement reached the Provider")

    maintenance = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=ExplodingProvider(),
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        full_history_required_listing_ids=frozenset({"listing-aapl"}),
        mutation_gate=WorkspaceMutationGate(),
    )
    first = maintenance.run(observed_at=NOW)
    second = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=ExplodingProvider(),
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        full_history_required_listing_ids=frozenset({"listing-aapl"}),
        mutation_gate=WorkspaceMutationGate(),
    ).run(observed_at=NOW + timedelta(minutes=5))

    assert first.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert second.status is CurrentUniverseMaintenanceStatus.COMPLETED
    listing = market_data.current_universe_maintenance_listings(first.maintenance_id)[0]
    assert listing.state == "FAILED"
    assert listing.failure_code == "data.full_history_audit_approval_required"
    assert listing.attempt_count == 0

    authorized = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=ExplodingProvider(),
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        full_audit_listing_ids=frozenset({"listing-aapl"}),
        full_history_escalation_listing_ids=frozenset({"listing-aapl"}),
        mutation_gate=WorkspaceMutationGate(),
    )
    assert authorized.maintenance_id != maintenance.maintenance_id
    fulfilled_authorization = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=ExplodingProvider(),
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        authorization_identity_listing_ids=frozenset({"listing-aapl"}),
        mutation_gate=WorkspaceMutationGate(),
    )
    assert fulfilled_authorization.maintenance_id == authorized.maintenance_id


def test_listing_authorization_reuses_verified_unaffected_prefix(tmp_path: Path) -> None:
    manifest, market_data, _, _ = _maintenance_workspace(
        tmp_path, eligible_listing_ids=("listing-aapl", "listing-msft")
    )
    base_id = current_universe_maintenance_id(
        manifest,
        as_of_session=date(2026, 7, 31),
    )
    market_data.admit_current_universe_maintenance(
        manifest,
        maintenance_id=base_id,
        as_of_session=date(2026, 7, 31),
        observed_at=NOW,
    )
    documents = {}
    for listing_id in ("listing-aapl", "listing-msft"):
        document = {
            "listing_id": listing_id,
            "audit_scope": "ROLLING",
            "history_start": "2026-07-01",
            "history_end": "2026-07-31",
            "provider_receipt_hash": canonical_hash([listing_id, "provider"]),
        }
        documents[listing_id] = document
        market_data.update_current_universe_maintenance_listing(
            maintenance_id=base_id,
            listing_id=listing_id,
            state="UPDATED",
            raw_through=date(2026, 7, 31),
            change_document=document,
            observed_at=NOW,
        )

    authorized = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=SimpleNamespace(name="fixture"),
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        full_audit_listing_ids=frozenset({"listing-msft"}),
        full_history_escalation_listing_ids=frozenset({"listing-msft"}),
        mutation_gate=WorkspaceMutationGate(),
    )
    authorized.admit(observed_at=NOW + timedelta(minutes=1))
    authorized.admit(observed_at=NOW + timedelta(minutes=2))

    by_listing = {
        item.listing_id: item
        for item in market_data.current_universe_maintenance_listings(authorized.maintenance_id)
    }
    assert by_listing["listing-aapl"].state == "UPDATED"
    assert by_listing["listing-aapl"].attempt_count == 0
    assert by_listing["listing-aapl"].change_document == documents["listing-aapl"]
    assert by_listing["listing-msft"].state == "PENDING"
    assert by_listing["listing-msft"].change_document is None


def test_reused_change_document_does_not_advance_action_audit_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, market_data, feature_state, panel_state = _maintenance_workspace(tmp_path)
    mutation_gate = WorkspaceMutationGate()
    registry = DuckDbWorkspaceMaintenanceRegistry(market_data.path, gate=mutation_gate)
    coordinator = WorkspaceMaintenanceCoordinator(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        manifest=manifest,
        provider=SimpleNamespace(name="fixture"),
        mutation_gate=mutation_gate,
        readiness_gate=SimpleNamespace(),
        feature_foundation=SimpleNamespace(catalog=FeatureCatalog.load()),
        registry=registry,
    )
    document = {
        "listing_id": "listing-aapl",
        "audit_scope": "FULL",
        "history_start": "2026-07-01",
        "history_end": "2026-07-31",
        "new_session_start": "2026-07-31",
        "new_sessions": ["2026-07-31"],
        "raw_correction_start": None,
        "raw_correction_sessions": [],
        "action_correction_start": None,
        "adjusted_return_change_sessions": ["2026-07-31"],
        "window_action_hash": "1" * 64,
        "action_set_hash": "2" * 64,
        "raw_evidence_hash": "3" * 64,
        "mapping_revision": "4" * 64,
        "provider_receipt_hash": "5" * 64,
    }
    request = _request()

    first = coordinator._record_change_set((document,), request=request, observed_at=NOW)
    first_receipt = registry.latest_action_audit("listing-aapl", "fixture")
    second = coordinator._record_change_set(
        (document,), request=request, observed_at=NOW + timedelta(minutes=1)
    )
    second_receipt = registry.latest_action_audit("listing-aapl", "fixture")

    assert first == second
    assert first_receipt is not None
    assert second_receipt is not None
    assert second_receipt.receipt_hash == first_receipt.receipt_hash
    monkeypatch.setattr(
        market_data,
        "full_action_audit_receipt_hashes",
        lambda hashes: frozenset(hashes),
    )
    assert (
        coordinator._pending_authorized_full_audit_listing_ids(
            request,
            ("listing-aapl",),
        )
        == ()
    )

    changed_policy = WorkspaceMaintenanceRequest.create(
        market_profile_id=request.market_profile_id,
        target_market_session=request.target_market_session,
        knowledge_cutoff_at=request.knowledge_cutoff_at,
        trigger=request.trigger,
        membership_revision=request.membership_revision,
        data_policy_hash="9" * 64,
        feature_policy_hash=request.feature_policy_hash,
        full_history_listing_ids=("listing-aapl",),
    )
    assert coordinator._pending_authorized_full_audit_listing_ids(
        changed_policy,
        ("listing-aapl",),
    ) == ("listing-aapl",)


def test_rolling_provider_receipt_cannot_be_laundered_into_full_audit_anchor(
    tmp_path: Path,
) -> None:
    manifest, market_data, feature_state, panel_state = _maintenance_workspace(tmp_path)
    sessions = (
        date(2026, 7, 1),
        date(2026, 7, 2),
        date(2026, 7, 29),
        date(2026, 7, 30),
    )
    rows = tuple(
        {
            "session_date": session.isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
        }
        for session in sessions
    )
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", {"AAPL": rows}, ("AAPL",)),
        ingestion_id="initial",
        observed_at=NOW,
    )
    rolling_receipt, _counts = market_data.complete_action_audit(
        manifest,
        listing_id="listing-aapl",
        provider="fixture",
        observed_actions=(),
        observed_adjusted_closes=tuple(
            ProviderAdjustedClosePoint(
                listing_id="listing-aapl",
                provider="fixture",
                session_date=session,
                adjusted_close=100.0,
            )
            for session in sessions[-2:]
        ),
        history_start=sessions[-2],
        history_end=sessions[-1],
        requested_as_of=sessions[-1],
        observed_at=NOW,
    )
    verification = dict(
        listing_id="listing-aapl", provider="fixture", requested_as_of=sessions[-1], now=NOW
    )
    assert market_data.reusable_action_audit_receipt(manifest, **verification) is None
    assert (
        market_data.verified_action_audit_receipt(
            manifest, receipt_hash=rolling_receipt.receipt_hash, **verification
        )
        == rolling_receipt
    )
    assert (
        market_data.verified_action_audit_receipt(manifest, receipt_hash="f" * 64, **verification)
        is None
    )
    # The exact receipt identity matters as well as matching the current bytes.
    with market_data._connect() as connection:
        connection.execute(
            "UPDATE action_audit_receipt SET max_adjusted_close_difference_bps = ? "
            "WHERE receipt_hash = ?",
            [rolling_receipt.max_adjusted_close_difference_bps + 1, rolling_receipt.receipt_hash],
        )
    try:
        assert (
            market_data.verified_action_audit_receipt(
                manifest, receipt_hash=rolling_receipt.receipt_hash, **verification
            )
            is None
        )
    finally:
        with market_data._connect() as connection:
            connection.execute(
                "UPDATE action_audit_receipt SET max_adjusted_close_difference_bps = ? "
                "WHERE receipt_hash = ?",
                [rolling_receipt.max_adjusted_close_difference_bps, rolling_receipt.receipt_hash],
            )
    mutation_gate = WorkspaceMutationGate()
    registry = DuckDbWorkspaceMaintenanceRegistry(market_data.path, gate=mutation_gate)
    polluted = ActionAuditChainReceipt.create(
        listing_id="listing-aapl",
        provider="fixture",
        audit_scope=ActionAuditScope.FULL,
        previous_receipt_hash=None,
        full_anchor_receipt_hash=None,
        history_start=rolling_receipt.history_start,
        history_end=rolling_receipt.history_end,
        covered_through_session=rolling_receipt.requested_as_of,
        window_action_hash=rolling_receipt.action_set_hash,
        action_set_hash=rolling_receipt.action_set_hash,
        raw_evidence_hash=rolling_receipt.raw_evidence_hash,
        mapping_revision=rolling_receipt.mapping_revision,
        data_policy_hash="incorrect-full-anchor-fixture",
        provider_receipt_hash=rolling_receipt.receipt_hash,
        observed_at=NOW,
    )
    registry.record_action_audit(polluted)
    coordinator = WorkspaceMaintenanceCoordinator(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        manifest=manifest,
        provider=SimpleNamespace(name="fixture"),
        mutation_gate=mutation_gate,
        readiness_gate=SimpleNamespace(),
        feature_foundation=SimpleNamespace(catalog=FeatureCatalog.load(), progress_sink=None),
        registry=registry,
    )

    required = coordinator._required_full_audit_listing_ids(
        NOW,
        as_of_session=date(2026, 7, 31),
        progress_operation_id="full-audit-preflight-fixture",
    )

    assert required == ("listing-aapl",)


def test_a_batched_action_audit_keeps_every_per_event_outcome(tmp_path: Path) -> None:
    """A first audit inserts as one relation; the other outcomes are still judged per event."""

    manifest, market_data, _, _ = _maintenance_workspace(tmp_path)
    sessions = tuple(date(2026, 7, 20) + timedelta(days=offset) for offset in range(5))
    rows = tuple(
        {
            "session_date": session.isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
        }
        for session in sessions
    )
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", {"AAPL": rows}, ("AAPL",)),
        ingestion_id="initial",
        observed_at=NOW,
    )
    adjusted = tuple(
        ProviderAdjustedClosePoint(
            listing_id="listing-aapl", provider="fixture", session_date=s, adjusted_close=100.0
        )
        for s in sessions
    )

    def event(session: date, kind: str, **values) -> CorporateActionEvent:
        return CorporateActionEvent(
            listing_id="listing-aapl",
            provider="fixture",
            effective_date=session,
            action_kind=kind,
            **values,
        )

    def audit(actions, *, seconds: int):
        return market_data.complete_action_audit(
            manifest,
            listing_id="listing-aapl",
            provider="fixture",
            observed_actions=actions,
            observed_adjusted_closes=adjusted,
            history_start=sessions[0],
            history_end=sessions[-1],
            requested_as_of=sessions[-1],
            observed_at=NOW + timedelta(seconds=seconds),
        )

    first = (
        event(sessions[1], "CASH_DIVIDEND", cash_amount=1.0),
        event(sessions[2], "SPLIT", new_shares_per_old_share=2.0),
        event(sessions[3], "CASH_DIVIDEND", cash_amount=0.5),
    )
    _receipt, counts = audit(first, seconds=0)
    assert counts == {"inserted": 3, "corrected": 0, "retracted": 0}
    assert market_data.actions("listing-aapl") == first
    # The same observation again: nothing moves.
    _receipt, counts = audit(first, seconds=1)
    assert counts == {"inserted": 0, "corrected": 0, "retracted": 0}
    # One corrected, one absent (retracted), one new.
    second = (
        event(sessions[1], "CASH_DIVIDEND", cash_amount=1.25),
        event(sessions[2], "SPLIT", new_shares_per_old_share=2.0),
        event(sessions[4], "CASH_DIVIDEND", cash_amount=0.75),
    )
    _receipt, counts = audit(second, seconds=2)
    assert counts == {"inserted": 1, "corrected": 1, "retracted": 1}
    assert market_data.actions("listing-aapl") == second
    # The retracted event observed again: restored, not inserted.
    _receipt, counts = audit((*second, first[2]), seconds=3)
    assert counts == {"inserted": 0, "corrected": 1, "retracted": 0}
    assert set(market_data.actions("listing-aapl")) == {*second, first[2]}


def test_rolling_action_audit_ignores_actions_outside_declared_scope(tmp_path: Path) -> None:
    manifest, market_data, _, _ = _maintenance_workspace(tmp_path)
    sessions = (
        date(2026, 7, 1),
        date(2026, 7, 2),
        date(2026, 7, 29),
        date(2026, 7, 30),
    )
    rows = tuple(
        {
            "session_date": session.isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
        }
        for session in sessions
    )
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", {"AAPL": rows}, ("AAPL",)),
        ingestion_id="initial",
        observed_at=NOW,
    )
    historical_dividend = CorporateActionEvent(
        listing_id="listing-aapl",
        provider="fixture",
        effective_date=sessions[1],
        action_kind="CASH_DIVIDEND",
        cash_amount=1.0,
    )
    adjusted = tuple(
        ProviderAdjustedClosePoint(
            listing_id="listing-aapl",
            provider="fixture",
            session_date=session,
            adjusted_close=100.0,
        )
        for session in sessions
    )
    market_data.complete_action_audit(
        manifest,
        listing_id="listing-aapl",
        provider="fixture",
        observed_actions=(historical_dividend,),
        observed_adjusted_closes=adjusted,
        history_start=sessions[0],
        history_end=sessions[-1],
        requested_as_of=sessions[-1],
        observed_at=NOW,
    )

    receipt, counts = market_data.complete_action_audit(
        manifest,
        listing_id="listing-aapl",
        provider="fixture",
        observed_actions=(),
        observed_adjusted_closes=adjusted[-2:],
        history_start=sessions[-2],
        history_end=sessions[-1],
        requested_as_of=sessions[-1],
        observed_at=NOW + timedelta(seconds=1),
    )

    assert receipt.history_start == sessions[-2]
    assert counts == {"inserted": 0, "corrected": 0, "retracted": 0}
    assert market_data.actions("listing-aapl") == (historical_dividend,)


def test_rolling_action_ambiguity_uses_one_full_history_hydration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, market_data, _, _ = _maintenance_workspace(tmp_path)
    sessions = tuple(date(2026, 5, 1) + timedelta(days=offset) for offset in range(92))

    def rows_between(start: date, end: date) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "session_date": session.isoformat(),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000,
            }
            for session in sessions
            if start <= session <= end
        )

    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(
            manifest,
            "fixture",
            {"AAPL": rows_between(sessions[0], date(2026, 7, 30))},
            ("AAPL",),
        ),
        ingestion_id="initial",
        observed_at=NOW,
    )

    class RecordingProvider:
        name = "fixture"

        def __init__(self) -> None:
            self.starts: list[date] = []

        def fetch_hydration(self, *, listing_id, provider_symbol, start, end):
            self.starts.append(start)
            rows = rows_between(start, end)
            return HydrationEvidence(
                daily_rows=rows,
                actions=(),
                adjusted_closes=tuple(
                    ProviderAdjustedClosePoint(
                        listing_id=listing_id,
                        provider=self.name,
                        session_date=date.fromisoformat(str(row["session_date"])),
                        adjusted_close=float(row["close"]),
                    )
                    for row in rows
                ),
            )

    original_audit = market_data.complete_action_audit
    audits: list[date] = []

    def ambiguous_when_rolling(*args, **kwargs):
        audits.append(kwargs["history_start"])
        if kwargs["history_start"] != sessions[0]:
            raise ActionAuditScopeInsufficient("rolling session-set ambiguity")
        return original_audit(*args, **kwargs)

    monkeypatch.setattr(market_data, "complete_action_audit", ambiguous_when_rolling)
    provider = RecordingProvider()
    outcome = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=provider,
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        mutation_gate=WorkspaceMutationGate(),
        full_history_escalation_listing_ids=frozenset({"listing-aapl"}),
    ).run(observed_at=NOW)

    assert outcome.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert len(provider.starts) == 2
    assert provider.starts[0] > sessions[0]
    assert provider.starts[1] == sessions[0]
    # The rolling audit is ambiguous in the listing's transaction, which rolls back, and again on
    # separate commits; only then does the one full-history audit run.
    assert audits[-1] == sessions[0]
    assert len(audits) == 3 and all(start > sessions[0] for start in audits[:-1])


def test_rolling_ambiguity_never_expands_to_full_history_without_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, market_data, _, _ = _maintenance_workspace(tmp_path)
    sessions = tuple(date(2026, 5, 1) + timedelta(days=offset) for offset in range(92))
    rows = tuple(
        {
            "session_date": session.isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
        }
        for session in sessions
        if session <= date(2026, 7, 30)
    )
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", {"AAPL": rows}, ("AAPL",)),
        ingestion_id="initial",
        observed_at=NOW,
    )

    class RecordingProvider:
        name = "fixture"

        def __init__(self) -> None:
            self.starts: list[date] = []

        def fetch_hydration(self, *, listing_id, provider_symbol, start, end):
            del provider_symbol
            self.starts.append(start)
            observed = tuple(
                {
                    "session_date": session.isoformat(),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1_000_000,
                }
                for session in sessions
                if start <= session <= end
            )
            return HydrationEvidence(
                daily_rows=observed,
                actions=(),
                adjusted_closes=tuple(
                    ProviderAdjustedClosePoint(
                        listing_id=listing_id,
                        provider=self.name,
                        session_date=date.fromisoformat(str(row["session_date"])),
                        adjusted_close=100.0,
                    )
                    for row in observed
                ),
            )

    monkeypatch.setattr(
        market_data,
        "complete_action_audit",
        lambda *args, **kwargs: (_ for _ in ()).throw(ActionAuditScopeInsufficient("ambiguous")),
    )
    provider = RecordingProvider()
    outcome = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=provider,
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        mutation_gate=WorkspaceMutationGate(),
    ).run(observed_at=NOW)

    listing = market_data.current_universe_maintenance_listings(outcome.maintenance_id)[0]
    assert outcome.failed == 1
    assert listing.failure_code == "data.full_history_audit_approval_required"
    assert provider.starts == [date(2026, 6, 15)]


def test_turn_signal_is_memory_only_and_background_remains_wakeable() -> None:
    wake = MaintenanceWakeController()
    called = Event()
    calls: list[int] = []

    def run_once() -> None:
        calls.append(1)
        called.set()

    host = MaintenanceBackgroundHost(wake=wake, run_once=run_once)
    host.start()
    try:
        wake.set_next_due(NOW)
        assert not wake.signal_due_work(NOW - timedelta(seconds=1))
        assert wake.signal_due_work(NOW)
        assert called.wait(1.0)
        assert calls == [1]
    finally:
        host.close()


def test_timed_wake_fires_without_a_front_desk_turn_and_stops() -> None:
    wake = MaintenanceWakeController()
    called = Event()
    calls = []

    def once():
        calls.append(1)
        wake.set_next_due(None)
        called.set()

    wake.set_next_due(datetime.now(UTC) + timedelta(milliseconds=50))
    host = MaintenanceBackgroundHost(wake=wake, run_once=once, clock=lambda: datetime.now(UTC))
    host.start()
    try:
        assert called.wait(2)
        assert calls == [1]
    finally:
        assert host.close()
    called.clear()
    wake.set_next_due(datetime.now(UTC))
    wake.event.set()
    assert not called.wait(0.05)


def test_set_scoped_quality_evidence_reads_match_the_per_listing_readers(tmp_path: Path) -> None:
    """requirement: one governance pass reads the manifest once, listing evidence unchanged.

    The coordinator judges every listing of the manifest from its newest
    qualification, its raw close history through the target session and its
    active actions under its current provider. The set-scoped readers must
    return, per listing, exactly what the per-listing readers return -- the
    newest of several admissions, the bars up to and not beyond the target,
    the actions in their recorded order -- and omit a listing the per-listing
    action reader refuses for lack of a provider mapping.
    """

    manifest, market_data, _, _ = _maintenance_workspace(
        tmp_path, eligible_listing_ids=("listing-aapl", "listing-msft")
    )
    sessions = (date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31))

    def rows(selected: tuple[date, ...], *, close: float) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "session_date": session.isoformat(),
                "open": close + index,
                "high": close + index + 1.0,
                "low": close + index - 1.0,
                "close": close + index,
                "volume": 1_000_000,
            }
            for index, session in enumerate(selected)
        )

    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(
            manifest,
            "fixture",
            {"AAPL": rows(sessions, close=100.0), "MSFT": rows(sessions[1:], close=50.0)},
            ("AAPL", "MSFT"),
        ),
        ingestion_id="set-scoped-reads",
        observed_at=NOW - timedelta(days=1),
    )
    for onboarding_id, evaluated_at, eligible in (
        ("onboarding-older", NOW - timedelta(days=2), False),
        ("onboarding-newer", NOW - timedelta(days=1), True),
    ):
        for listing_id in ("listing-aapl", "listing-msft"):
            market_data.record_current_universe_quality_admission(
                onboarding_id=onboarding_id,
                listing_id=listing_id,
                eligible=eligible,
                expected_sessions=4,
                observed_sessions=4,
                missing_sessions=0,
                missing_ratio=0.0,
                maximum_consecutive_gap=0,
                reasons=() if eligible else ("QUALITY_FIXTURE",),
                observed_at=evaluated_at,
            )
    market_data.complete_action_audit(
        manifest,
        listing_id="listing-aapl",
        provider="fixture",
        observed_actions=(
            CorporateActionEvent(
                listing_id="listing-aapl",
                provider="fixture",
                effective_date=sessions[2],
                action_kind="SPLIT",
                new_shares_per_old_share=2.0,
            ),
            CorporateActionEvent(
                listing_id="listing-aapl",
                provider="fixture",
                effective_date=sessions[1],
                action_kind="CASH_DIVIDEND",
                cash_amount=1.0,
            ),
        ),
        observed_adjusted_closes=tuple(
            ProviderAdjustedClosePoint(
                listing_id="listing-aapl",
                provider="fixture",
                session_date=session,
                adjusted_close=100.0,
            )
            for session in sessions
        ),
        history_start=sessions[0],
        history_end=sessions[-1],
        requested_as_of=sessions[-1],
        observed_at=NOW,
    )

    known = ("listing-msft", "listing-aapl")
    through = sessions[-2]
    inputs = market_data.quality_governance_inputs((*known, "listing-none"), through=through)
    assert inputs.admissions == {
        listing_id: market_data.latest_quality_admission(listing_id) for listing_id in known
    }
    assert all(
        admission.onboarding_id == "onboarding-newer" for admission in inputs.admissions.values()
    )

    series = inputs.raw_close_series
    assert series.column_names == ["listing_id", "session_date", "close"]
    observed = list(
        zip(
            series.column("listing_id").to_pylist(),
            series.column("session_date").to_pylist(),
            series.column("close").to_pylist(),
            strict=True,
        )
    )
    assert observed == [
        (listing_id, bar.session_date, bar.close)
        for listing_id in sorted(known)
        for bar in market_data.raw_bars(listing_id, through=through)
    ]
    assert {session for _listing, session, _close in observed} == set(sessions[:-1])

    assert inputs.actions == {listing_id: market_data.actions(listing_id) for listing_id in known}
    assert [item.action_kind for item in inputs.actions["listing-aapl"]] == [
        "CASH_DIVIDEND",
        "SPLIT",
    ]
    with pytest.raises(ValueError, match="no provider mapping"):
        market_data.actions("listing-none")


def _restatement_bar(session: date, *, close: float) -> RawDailyBar:
    return RawDailyBar(
        listing_id="listing-aapl",
        provider="fixture",
        session_date=session,
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1_000_000,
    )


def test_bounded_restatement_observation_is_pure_and_content_addressed() -> None:
    """Same content -> exact reuse; different content -> observed correction.

    The observation claims nothing beyond what it saw: no mutation, no new
    qualified identity, no downstream surface names. The null check and the
    injected correction prove both directions, and an unbounded or duplicated
    scope is refused rather than widened.
    """

    qualified = (
        _restatement_bar(date(2026, 7, 29), close=100.0),
        _restatement_bar(date(2026, 7, 30), close=101.0),
    )
    unchanged = audit_bounded_restatements(
        candidate_bars=qualified,
        qualified_bars=qualified,
        provider="fixture",
        listing_scope=("listing-aapl",),
        session_scope_start=date(2026, 7, 29),
        session_scope_end=date(2026, 7, 30),
    )
    assert unchanged.corrections == ()
    assert unchanged.exact_reuse_count == 2
    assert unchanged.qualified_scope_identity == unchanged.candidate_scope_identity

    corrected = audit_bounded_restatements(
        candidate_bars=(
            _restatement_bar(date(2026, 7, 29), close=100.5),
            qualified[1],
            _restatement_bar(date(2026, 7, 31), close=102.0),
        ),
        qualified_bars=qualified,
        provider="fixture",
        listing_scope=("listing-aapl",),
        session_scope_start=date(2026, 7, 29),
        session_scope_end=date(2026, 7, 31),
    )
    assert len(corrected.corrections) == 1
    observation = corrected.corrections[0]
    assert observation.session_date == date(2026, 7, 29)
    assert observation.classification == "PROVIDER_CORRECTION_OBSERVED"
    assert observation.prior_identity != observation.candidate_identity
    assert corrected.exact_reuse_count == 1
    assert corrected.new_session_count == 1
    assert corrected.qualified_scope_identity != corrected.candidate_scope_identity

    with pytest.raises(ValueError, match="explicit ordered listing scope"):
        audit_bounded_restatements(
            candidate_bars=qualified,
            qualified_bars=qualified,
            provider="fixture",
            listing_scope=(),
            session_scope_start=date(2026, 7, 29),
            session_scope_end=date(2026, 7, 30),
        )
    with pytest.raises(ValueError, match="session scope is reversed"):
        audit_bounded_restatements(
            candidate_bars=qualified,
            qualified_bars=qualified,
            provider="fixture",
            listing_scope=("listing-aapl",),
            session_scope_start=date(2026, 7, 30),
            session_scope_end=date(2026, 7, 29),
        )
    with pytest.raises(ValueError, match="duplicate rows"):
        audit_bounded_restatements(
            candidate_bars=(qualified[0], qualified[0]),
            qualified_bars=qualified,
            provider="fixture",
            listing_scope=("listing-aapl",),
            session_scope_start=date(2026, 7, 29),
            session_scope_end=date(2026, 7, 30),
        )


def _restatement_workspace(tmp_path: Path):
    """A qualified two-session history plus a provider that restates one row."""

    manifest, market_data, _, _ = _maintenance_workspace(tmp_path)
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(
            manifest,
            "fixture",
            {
                "AAPL": tuple(
                    {
                        "session_date": session.isoformat(),
                        "open": close,
                        "high": close + 1.0,
                        "low": close - 1.0,
                        "close": close,
                        "volume": 1_000_000,
                    }
                    for session, close in (
                        (date(2026, 7, 29), 100.0),
                        (date(2026, 7, 30), 101.0),
                    )
                )
            },
            ("AAPL",),
        ),
        ingestion_id="restatement-initial",
        observed_at=NOW - timedelta(days=1),
    )

    corrected_rows = tuple(
        {
            "session_date": session.isoformat(),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000,
        }
        for session, close in (
            (date(2026, 7, 29), 100.5),
            (date(2026, 7, 30), 101.0),
            (date(2026, 7, 31), 102.0),
        )
    )

    class CorrectingProvider:
        name = "fixture"

        def fetch_hydration(self, *, listing_id, provider_symbol, start, end):
            del provider_symbol
            observed = tuple(
                row
                for row in corrected_rows
                if start <= date.fromisoformat(str(row["session_date"])) <= end
            )
            return HydrationEvidence(
                daily_rows=observed,
                actions=(),
                adjusted_closes=tuple(
                    ProviderAdjustedClosePoint(
                        listing_id=listing_id,
                        provider=self.name,
                        session_date=date.fromisoformat(str(row["session_date"])),
                        adjusted_close=float(row["close"]),
                    )
                    for row in observed
                ),
            )

    return manifest, market_data, CorrectingProvider()


def _qualified_state(market_data: MarketDataRepository) -> tuple[float, int, int]:
    connection = market_data._connect(read_only=True)
    try:
        close = connection.execute(
            """
            SELECT close FROM raw_daily_bar_current
            WHERE listing_id = 'listing-aapl' AND session_date = DATE '2026-07-29'
            """
        ).fetchone()
        sessions = connection.execute(
            "SELECT count(*) FROM raw_daily_bar_current WHERE listing_id = 'listing-aapl'"
        ).fetchone()
        revisions = connection.execute(
            "SELECT count(*) FROM bar_revision WHERE listing_id = 'listing-aapl'"
        ).fetchone()
    finally:
        connection.close()
    assert close is not None and sessions is not None and revisions is not None
    return float(close[0]), int(revisions[0]), int(sessions[0])


def test_unauthorized_restatement_never_reaches_qualification(tmp_path: Path) -> None:
    """Observing a correction is not permission to apply it.

    The whole point of observing before the write is that the write can be
    refused. Without escalation authority the qualified row keeps its content,
    no ``bar_revision`` appears, the appended session is not admitted either --
    the batch is refused whole -- and only failure facts are recorded, so no
    downstream identity can have moved. Authorizing the listing is what turns
    the observed correction into an explicit qualification with its own
    revision trail.
    """

    manifest, market_data, provider = _restatement_workspace(tmp_path)
    before_close, before_revisions, before_sessions = _qualified_state(market_data)
    assert (before_close, before_revisions, before_sessions) == (100.0, 0, 2)

    refused = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=provider,
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        mutation_gate=WorkspaceMutationGate(),
    ).run(observed_at=NOW)

    assert refused.failed == 1
    listing = market_data.current_universe_maintenance_listings(refused.maintenance_id)[0]
    assert listing.state == "FAILED"
    assert listing.failure_code == "data.full_history_audit_approval_required"
    assert listing.change_document == {
        "failure_cause": {
            "exception_type": "UNKNOWN",
            "detail": "The source failure cause was not recorded.",
            "step": "Provider price history",
            "unit": "AAPL",
            "row_count": "UNKNOWN",
            "sanitizer_code": "UNKNOWN",
        }
    }
    assert _qualified_state(market_data) == (100.0, 0, 2)

    authorized = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=provider,
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        full_history_escalation_listing_ids=frozenset({"listing-aapl"}),
        mutation_gate=WorkspaceMutationGate(),
    ).run(observed_at=NOW + timedelta(minutes=5))

    assert (authorized.updated, authorized.failed) == (1, 0)
    qualified = market_data.current_universe_maintenance_listings(authorized.maintenance_id)[0]
    assert qualified.state == "UPDATED"
    assert qualified.change_document is not None
    after_close, after_revisions, after_sessions = _qualified_state(market_data)
    assert (after_close, after_revisions, after_sessions) == (100.5, 1, 3)


def test_maintenance_observes_restatement_before_explicit_qualification(
    tmp_path: Path,
) -> None:
    """One changed historical row: observed pre-admission, revised on qualification.

    The refresh carries the pre-admission observation in its change document --
    the correction was known before any write -- while ``bar_revision`` and the
    updated current row are the storage owner's explicit qualification acts.
    The prior identity stays readable in the revision trail, the scoped content
    identities differ, and nothing moved except through that explicit act.
    """

    manifest, market_data, provider = _restatement_workspace(tmp_path)

    outcome = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=provider,
        as_of_session=date(2026, 7, 31),
        max_workers=1,
        full_audit_listing_ids=frozenset({"listing-aapl"}),
        mutation_gate=WorkspaceMutationGate(),
    ).run(observed_at=NOW)
    assert (outcome.updated, outcome.failed) == (1, 0)

    listing = market_data.current_universe_maintenance_listings(outcome.maintenance_id)[0]
    document = listing.change_document
    assert document is not None
    observation = document["restatement_observation"]
    assert observation["exact_reuse_count"] == 1
    assert observation["new_session_count"] == 1
    assert observation["qualified_scope_identity"] != observation["candidate_scope_identity"]
    (correction,) = observation["corrections"]
    assert correction["session_date"] == "2026-07-29"
    assert correction["classification"] == "PROVIDER_CORRECTION_OBSERVED"
    assert correction["prior_identity"] != correction["candidate_identity"]
    sentinel = document["price_action_sentinel"]
    assert sentinel["disposition"] == "ANCHORED"
    assert sentinel["finding_codes"] == []

    connection = market_data._connect(read_only=True)
    try:
        revisions = connection.execute(
            """
            SELECT session_date, prior_payload_hash, next_payload_hash, reason
            FROM bar_revision WHERE listing_id = 'listing-aapl'
            """
        ).fetchall()
        current_close = connection.execute(
            """
            SELECT close FROM raw_daily_bar_current
            WHERE listing_id = 'listing-aapl' AND session_date = DATE '2026-07-29'
            """
        ).fetchone()
    finally:
        connection.close()
    assert len(revisions) == 1
    session_value, prior_hash, next_hash, reason = revisions[0]
    assert str(session_value) == "2026-07-29"
    assert prior_hash != next_hash
    assert reason == "provider_fact_correction"
    assert current_close is not None and float(current_close[0]) == 100.5


def _two_listings_one_session_due(
    tmp_path: Path, *, dividend: bool = False
) -> tuple[UniverseManifest, MarketDataRepository, object, date]:
    """Two listings holding two sessions, and a provider answering the third.

    With ``dividend``, the provider also observes a cash dividend on the third session.
    """
    listing_ids = ("listing-aapl", "listing-msft")
    manifest, market_data, _, _ = _maintenance_workspace(tmp_path, eligible_listing_ids=listing_ids)
    sessions = (date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31))

    def rows(through: date, start: date = sessions[0]) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "session_date": session.isoformat(),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.0 + index,
                "volume": 1_000_000,
            }
            for index, session in enumerate(sessions)
            if start <= session <= through
        )

    symbols = ("AAPL", "MSFT")
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", dict.fromkeys(symbols, rows(sessions[1])), symbols),
        ingestion_id="initial",
        observed_at=NOW,
    )

    class Provider:
        name = "fixture"

        def fetch_hydration(self, *, listing_id, provider_symbol, start, end):
            fetched = rows(end, start)
            return HydrationEvidence(
                daily_rows=fetched,
                actions=(
                    CorporateActionEvent(
                        listing_id=listing_id,
                        provider=self.name,
                        effective_date=sessions[2],
                        action_kind="CASH_DIVIDEND",
                        cash_amount=0.25,
                    ),
                )
                if dividend and start <= sessions[2] <= end
                else (),
                adjusted_closes=tuple(
                    ProviderAdjustedClosePoint(
                        listing_id=listing_id,
                        provider=self.name,
                        session_date=date.fromisoformat(str(row["session_date"])),
                        adjusted_close=float(str(row["close"])),
                    )
                    for row in fetched
                ),
            )

    return manifest, market_data, Provider(), sessions[2]


def test_progress_counts_listings_without_reading_every_change_document(
    tmp_path: Path, monkeypatch
) -> None:
    """regression: each listing's progress read every unit's change document.

    After every listing, the runner built the whole outcome to report three counts: all 473 rows
    read and every change document parsed, 473 times a day. Progress now counts in the engine,
    with the same numbers, and the units are read whole only where the outcome is returned.
    """
    manifest, market_data, provider, as_of = _two_listings_one_session_due(tmp_path)
    read_whole = market_data.current_universe_maintenance_listings
    reads: list[str] = []

    def counted(maintenance_id: str):  # type: ignore[no-untyped-def]
        reads.append(maintenance_id)
        return read_whole(maintenance_id)

    monkeypatch.setattr(market_data, "current_universe_maintenance_listings", counted)
    progress = []
    outcome = CurrentUniverseMaintenance(
        store=market_data,
        manifest=manifest,
        provider=provider,
        as_of_session=as_of,
        max_workers=1,
        mutation_gate=WorkspaceMutationGate(),
        progress_sink=progress.append,
    ).run(observed_at=NOW)
    assert (outcome.status, outcome.updated, outcome.failed) == (
        CurrentUniverseMaintenanceStatus.COMPLETED,
        2,
        0,
    )
    per_listing = [
        (u.completed_units, u.total_units, u.counters)
        for u in progress
        if u.status == "RUNNING" and u.current_item is not None
    ]
    assert per_listing == [
        (1, 2, {"updated": 1, "failed": 0, "pending": 1}),
        (2, 2, {"updated": 2, "failed": 0, "pending": 0}),
    ]
    # Whole reads only where an outcome is built (the run's opening report, its last chunk and
    # its close), not one more per listing.
    assert len(reads) == 3


_MAINTENANCE_TABLES = (
    "raw_daily_bar_current",
    "bar_revision",
    "corporate_action_current",
    "provider_attempt",
    "action_audit_receipt",
    "provider_adjusted_close_current",
    "provider_adjusted_series_revision",
    "data_quality",
    "current_universe_maintenance_listing",
)


def _maintenance_rows(market_data: MarketDataRepository) -> dict[str, list[tuple[object, ...]]]:
    connection = market_data.database.connect(read_only=True)
    try:
        return {
            table: sorted(connection.execute(f"SELECT * FROM {table}").fetchall(), key=repr)
            for table in _MAINTENANCE_TABLES
        }
    finally:
        connection.close()


def test_a_listing_that_fails_inside_its_transaction_is_rolled_back_and_applied_again(
    tmp_path: Path, monkeypatch
) -> None:
    """recovery: a failure inside a listing's one transaction rolls its batch and
    audit back whole; it then applies on separate commits, its attempt not counted again, and
    the store holds what an unfaulted run writes (a dividend left behind would read as held).
    """

    def run(root: Path, *, fault: bool) -> dict[str, list[tuple[object, ...]]]:
        manifest, market_data, provider, as_of = _two_listings_one_session_due(root, dividend=True)
        update = market_data.update_current_universe_maintenance_listing
        faults: list[str] = []

        def failing(**kwargs):  # type: ignore[no-untyped-def]
            if (
                fault
                and not faults
                and kwargs.get("_connection") is not None
                and kwargs["state"] == "UPDATED"
            ):
                faults.append(kwargs["listing_id"])
                raise RuntimeError("fault after the listing's batch and audit")
            return update(**kwargs)

        monkeypatch.setattr(market_data, "update_current_universe_maintenance_listing", failing)
        outcome = CurrentUniverseMaintenance(
            store=market_data,
            manifest=manifest,
            provider=provider,
            as_of_session=as_of,
            max_workers=1,
            mutation_gate=WorkspaceMutationGate(),
        ).run(observed_at=NOW)
        assert (outcome.status, outcome.updated, outcome.failed) == (
            CurrentUniverseMaintenanceStatus.COMPLETED,
            2,
            0,
        )
        assert faults == (["listing-aapl"] if fault else [])
        units = market_data.current_universe_maintenance_listings(outcome.maintenance_id)
        assert [(u.state, u.attempt_count) for u in units] == [("UPDATED", 1)] * 2
        return _maintenance_rows(market_data)

    assert run(tmp_path / "faulted", fault=True) == run(tmp_path / "clean", fault=False)
