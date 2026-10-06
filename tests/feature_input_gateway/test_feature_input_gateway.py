from __future__ import annotations

import ast
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest

from alphalattice.control.data_platform.maintenance.contracts import (
    DataRemediationFailureReceipt,
)
from alphalattice.control.data_platform.maintenance.registry import (
    DuckDbWorkspaceMaintenanceRegistry,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.contracts import (
    TemporalKnowledgeBoundary,
    canonical_hash,
)
from alphalattice.foundation.feature_engine.inputs.gateway import (
    FeatureInputAssessment,
    FeatureInputExecutionStatus,
    FeatureInputGateway,
    FeatureInputGovernanceService,
    FeatureInputPolicy,
    FeatureInputRemediationExecutor,
    ListingQualityEvidence,
    PanelImpactProjection,
    QuarantineContinuation,
    QuarantineContinuationRefusal,
    UnexplainedRawMove,
    matching_raw_retention_proof,
)
from alphalattice.foundation.feature_engine.inputs.quality import (
    FeatureInputQualityEvaluator,
    FeatureInputQualityPolicy,
    RawCloseSeries,
)
from alphalattice.foundation.feature_engine.panels.reader import (
    FeaturePanelReader,
    FeaturePanelReadRequest,
)
from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
    build_quality_filtered_research_manifest,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    CurrentUniverseQualityAdmission,
    MarketDataRepository,
)

NOW = datetime(2026, 8, 3, 15, tzinfo=UTC)
AS_OF = date(2026, 7, 31)


def _manifest(count: int, *, revision: str = "a" * 64) -> UniverseManifest:
    profile = MarketProfile(
        market_profile_id=f"feature-input-{count}",
        display_name="Feature Input fixture",
        market="US",
        currency="USD",
        calendar_id="XNYS",
        provider="fixture",
        daily_price_basis="unadjusted",
        manifest_as_of=AS_OF,
        data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
    )
    listings = tuple(
        ManifestListing(
            listing_id=f"listing-{index:03d}",
            symbol=f"S{index:03d}",
            mic="XNYS",
            provider_symbol=f"S{index:03d}",
        )
        for index in range(count)
    )
    return UniverseManifest(
        manifest_id=f"feature-input-{count}:{revision[:8]}",
        profile=profile,
        listings=listings,
        revision_sha256=revision,
    )


def _temporal(*, cutoff: datetime = NOW) -> TemporalKnowledgeBoundary:
    return TemporalKnowledgeBoundary(
        market_as_of_session=AS_OF,
        knowledge_cutoff_at=cutoff,
        materialized_at=cutoff + timedelta(minutes=1),
        universe_source_observed_at=cutoff - timedelta(days=1),
        sector_source_observed_at=cutoff - timedelta(hours=1),
    )


def _evidence(
    manifest: UniverseManifest,
    *,
    failed: set[int] | None = None,
    failure_code: str = "data.provider_timeout",
    extreme: bool = False,
) -> tuple[ListingQualityEvidence, ...]:
    failed = failed or set()
    return tuple(
        ListingQualityEvidence(
            listing_id=listing.listing_id,
            provider="fixture",
            range_start=date(2016, 8, 1),
            range_end=AS_OF,
            evidence_hash=canonical_hash([listing.listing_id, index in failed, failure_code]),
            failure_code=failure_code if index in failed else None,
            reason_codes=(("UNEXPLAINED_RAW_MOVE",) if extreme else ("PROVIDER_TIMEOUT",))
            if index in failed
            else (),
            retry_exhausted=index in failed,
            extreme_move_unexplained=extreme and index in failed,
        )
        for index, listing in enumerate(manifest.listings)
    )


def _sectors(manifest: UniverseManifest, *, size: int = 10) -> dict[str, str]:
    return {
        listing.listing_id: f"sector-{index // size:02d}"
        for index, listing in enumerate(manifest.listings)
    }


def _panel_impact(manifest: UniverseManifest, sectors: dict[str, str]) -> PanelImpactProjection:
    distribution: dict[str, int] = {}
    for sector in sectors.values():
        distribution[sector] = distribution.get(sector, 0) + 1
    return PanelImpactProjection(distribution, 1.0, 60, True)


def test_clean_path_is_zero_agent_and_durable(tmp_path) -> None:
    manifest = _manifest(20)
    sectors = _sectors(manifest, size=5)
    market_data = MarketDataRepository(tmp_path / "workspace")
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    service = FeatureInputGovernanceService(
        market_data=market_data,
        panel_state=panel_state,
        mutation_gate=WorkspaceMutationGate(),
        gateway=FeatureInputGateway(),
    )
    result = service.assess_and_record(
        candidate_manifest=manifest,
        evidence=_evidence(manifest),
        sector_by_listing_id=sectors,
        temporal_boundary=_temporal(),
        panel_impact=_panel_impact(manifest, sectors),
        observed_at=NOW,
    )
    assert result.assessment is FeatureInputAssessment.ADMITTED
    assert not result.agent_cases and result.admission is not None
    assert result.research_manifest is not None
    disclosure = panel_state.feature_input_quality_disclosure(
        result_manifest_revision=result.research_manifest.revision_sha256
    )
    assert disclosure["gateway_qualified"] is True
    assert disclosure["candidate_listing_count"] == 20


def test_forward_quarantine_is_dated_usability_not_a_smaller_nominal_universe(tmp_path):
    from alphalattice.foundation.market_data_ops.sources.membership import UniverseBootstrapRecord

    candidate = _manifest(60)
    market = MarketDataRepository(tmp_path / "workspace")
    panel = PanelStateRepository(market.database, market_data=market)
    service = FeatureInputGovernanceService(
        market, panel, WorkspaceMutationGate(), FeatureInputGateway()
    )
    sectors = _sectors(candidate, size=20)
    initial = service.assess_and_record(
        candidate_manifest=candidate,
        evidence=_evidence(candidate),
        sector_by_listing_id=sectors,
        temporal_boundary=_temporal(),
        observed_at=NOW,
    )
    nominal = initial.research_manifest
    assert nominal is not None
    market.record_universe_bootstrap(
        UniverseBootstrapRecord(
            market_profile_id=nominal.profile.market_profile_id,
            t0_session=AS_OF - timedelta(days=1),
            history_start=date(2016, 8, 1),
            cohort_listing_ids=tuple(item.listing_id for item in nominal.listings),
            cohort_hash="",
            manifest_revision=nominal.revision_sha256,
            candidate_manifest_hash="1" * 64,
            qualification_policy_hash="2" * 64,
            feature_input_policy_hash=service.gateway.policy.policy_hash,
            source_observed_at=NOW,
            admitted_at=NOW,
            panel_snapshot_hash="3" * 64,
            derivation="RECONSTRUCTED_FROM_DURABLE_EVIDENCE",
        )
    )
    bad = _evidence(nominal, failed={0})
    quarantine = service.gateway.quarantine(
        evidence=bad[0],
        recheck_after_at=NOW + timedelta(days=1),
        execution_receipt_hash="4" * 64,
    )
    panel.record_listing_quarantines(
        candidate_manifest_revision=nominal.revision_sha256,
        quarantines=(quarantine,),
        observed_at=NOW,
        effective_session=AS_OF,
    )
    held = service.assess_and_record(
        candidate_manifest=nominal,
        evidence=bad,
        sector_by_listing_id=sectors,
        temporal_boundary=_temporal(),
        observed_at=NOW,
    )
    assert held.research_manifest == nominal
    assert held.admission is not None and len(held.admission.admitted_listing_ids) == 59
    assert held.admission.nominal_membership_hash
    assert market.membership_events(nominal.profile.market_profile_id) == ()
    axis = (AS_OF - timedelta(days=1), AS_OF, date(2026, 8, 3))
    restrictions = panel.panel_source_exclusions(
        market_profile_id=nominal.profile.market_profile_id,
        sessions=axis,
        listing_ids=tuple(item.listing_id for item in nominal.listings),
    )
    assert len(restrictions) == 1 and restrictions[0].first_session == AS_OF
    good = tuple(
        replace(item, range_end=axis[-1], qualification_receipt_hash="5" * 64)
        for item in _evidence(nominal)
    )
    restored = service.assess_and_record(
        candidate_manifest=nominal,
        evidence=good,
        sector_by_listing_id=sectors,
        temporal_boundary=replace(
            _temporal(cutoff=NOW + timedelta(days=2)), market_as_of_session=axis[-1]
        ),
        observed_at=NOW + timedelta(days=2),
    )
    assert restored.research_manifest == nominal
    assert len(restored.admission.admitted_listing_ids) == 60
    after = panel.panel_source_exclusions(
        market_profile_id=nominal.profile.market_profile_id,
        sessions=axis,
        listing_ids=tuple(item.listing_id for item in nominal.listings),
    )
    assert after[0].first_session == AS_OF
    assert tuple(
        session for session in axis if after[0].first_session <= session <= after[0].last_session
    ) == (AS_OF,)
    assert market.membership_events(nominal.profile.market_profile_id) == ()
    with market._connect(read_only=True) as connection:
        payloads = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT admission_json FROM feature_input_admission"
            ).fetchall()
        ]
    assert any(
        row.get("nominal_membership_hash") == held.admission.nominal_membership_hash
        for row in payloads
    )


def test_pending_quality_case_reopens_exactly_and_changed_evidence_supersedes_it(tmp_path):
    manifest = _manifest(20)
    sectors = _sectors(manifest)
    market = MarketDataRepository(tmp_path / "workspace")
    panel = PanelStateRepository(market.database, market_data=market)
    service = FeatureInputGovernanceService(
        market, panel, WorkspaceMutationGate(), FeatureInputGateway()
    )
    evidence = _evidence(
        manifest, failed={0}, failure_code="data.unexplained_raw_move", extreme=True
    )
    facts = dict(
        candidate_manifest=manifest,
        evidence=evidence,
        sector_by_listing_id=sectors,
        temporal_boundary=_temporal(),
        panel_impact=_panel_impact(manifest, sectors),
    )
    first = service.assess_and_record(**facts, observed_at=NOW).agent_cases[0]
    assert panel.has_unresolved_feature_input(manifest.revision_sha256)
    assert first.evidence == (evidence[0],)
    # A new owner instance has no memory of the first assessment.
    reopened = FeatureInputGovernanceService(
        market, panel, WorkspaceMutationGate(), FeatureInputGateway()
    )
    second = reopened.assess_and_record(
        **facts, observed_at=NOW + timedelta(minutes=10)
    ).agent_cases[0]
    assert second == first
    assert len(panel.feature_input_case_documents(manifest.revision_sha256)) == 1
    changed = replace(evidence[0], evidence_hash="e" * 64)
    successor = reopened.assess_and_record(
        **{**facts, "evidence": (changed, *evidence[1:])},
        observed_at=NOW + timedelta(minutes=20),
    ).agent_cases[0]
    assert successor.issue_hash != first.issue_hash
    assert successor.case_token != first.case_token
    assert len(panel.feature_input_case_documents(manifest.revision_sha256)) == 1
    with pytest.raises(ValueError, match="stale Data Engineer"):
        service.gateway.validate_agent_selection(
            first,
            run_id=first.run_id,
            case_token=first.case_token,
            evidence_hash=first.evidence_hash,
            option_id=first.options[0].option_id,
            current_evidence_hash=successor.evidence_hash,
            rediagnosis_count=0,
        )
    with duckdb.connect(str(market.path)) as connection:
        connection.execute(
            "UPDATE feature_input_agent_case SET case_json = '{}' WHERE case_token = ?",
            [successor.case_token],
        )
    with pytest.raises(ValueError, match="case_record_tampered"):
        panel.feature_input_case_documents(manifest.revision_sha256)


def test_data_issue_readback_refuses_one_missing_case_manifest_without_hiding_healthy_case(
    tmp_path,
):
    from alphalattice.control.product_host.composition.application_session import (
        WorkspaceApplicationSession,
    )
    from alphalattice.control.product_host.data_preparation.remediation import (
        WorkspaceDataIssueApplication,
    )

    manifest = _manifest(20)
    manifest = replace(
        manifest, profile=replace(manifest.profile, market_profile_id="us-current-index-research")
    )
    with WorkspaceApplicationSession.acquire(tmp_path / "workspace") as session:
        market = MarketDataRepository(session.workspace)
        panel = PanelStateRepository(market.database, market_data=market)
        service = FeatureInputGovernanceService(
            market, panel, session.mutation_gate, FeatureInputGateway()
        )
        sectors = _sectors(manifest)
        result = service.assess_and_record(
            candidate_manifest=manifest,
            evidence=_evidence(
                manifest, failed={0}, failure_code="data.unexplained_raw_move", extreme=True
            ),
            sector_by_listing_id=sectors,
            temporal_boundary=_temporal(),
            panel_impact=_panel_impact(manifest, sectors),
            observed_at=NOW,
        )
        (healthy_case,) = result.agent_cases
        market.readiness.save(
            market_profile_id=manifest.profile.market_profile_id,
            status="FEATURE_BUILDING",
            active_manifest_id=manifest.manifest_id,
            active_manifest_revision=manifest.revision_sha256,
            active_membership_fingerprint=None,
            active_candidate_manifest_document=None,
            pending_membership_fingerprint=None,
            pending_candidate_manifest_document=None,
            last_checked_at=NOW,
            last_changed_at=NOW,
            failure_code=None,
            observed_at=NOW,
        )

        # This second row is a self-consistent case whose index places it in this catalog page,
        # but its own immutable manifest reference is unavailable.
        missing_revision = "f" * 64
        refused_token = canonical_hash(
            [
                missing_revision,
                healthy_case.case_kind,
                healthy_case.failure_code,
                healthy_case.evidence_hash,
                tuple(option.option_hash for option in healthy_case.options),
                healthy_case.policy_hash,
            ]
        )
        refused_case = replace(
            healthy_case,
            run_id=f"feature-input:{refused_token[:24]}",
            case_token=refused_token,
            manifest_revision=missing_revision,
        )
        document = refused_case.document()
        with duckdb.connect(str(market.path)) as connection:
            connection.execute(
                """INSERT INTO feature_input_agent_case (
                       case_token, manifest_revision, case_kind, failure_code, evidence_hash,
                       listing_ids_json, option_catalog_hash, rediagnosis_count, lifecycle,
                       created_at, case_json, case_record_hash
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)""",
                [
                    refused_case.case_token,
                    healthy_case.manifest_revision,
                    refused_case.case_kind,
                    refused_case.failure_code,
                    refused_case.evidence_hash,
                    json.dumps(refused_case.listing_ids, separators=(",", ":")),
                    canonical_hash(
                        [(option.option_id, option.option_hash) for option in refused_case.options]
                    ),
                    refused_case.rediagnosis_count,
                    NOW.replace(tzinfo=None),
                    json.dumps(document, sort_keys=True, separators=(",", ":")),
                    canonical_hash(document),
                ],
            )

        answer = WorkspaceDataIssueApplication(session, lambda: NOW).readback()
        assert answer["status"] == "ISSUES_PENDING"
        assert [item["case"]["case_token"] for item in answer["issues"]] == [
            healthy_case.case_token
        ]
        assert answer["case_count"] == 2
        assert {item["case_token"] for item in answer["case_index"]} == {
            healthy_case.case_token,
            refused_case.case_token,
        }
        (refusal,) = answer["refused_cases"]
        assert (refusal["status"], refusal["case_token"], refusal["failure_code"]) == (
            "REFUSED",
            refused_case.case_token,
            "feature_input.case_manifest_unavailable",
        )
        assert "manifest" in refusal["detail"].lower()
        assert refusal["next_requests"]["issues"] == {
            "operation": "DATA_ISSUES",
            "history_limit": 25,
        }
        assert refusal["next_requests"]["workspace"]["operation"] == "WORKSPACE_SHOW"
        assert refusal["next_requests"]["backups"]["operation"] == "WORKSPACE_BACKUPS"
        assert answer["next_requests"]["pending"] == {"operation": "PENDING_DECISIONS"}


def test_human_raw_move_choice_is_consumed_as_a_caveat_not_source_truth(tmp_path):
    from types import SimpleNamespace

    from alphalattice.control.data_platform.maintenance.coordinator import (
        WorkspaceMaintenanceCoordinator,
    )
    from alphalattice.control.product_host.composition.application_session import (
        WorkspaceApplicationSession,
    )
    from alphalattice.control.product_host.data_preparation.remediation import (
        WorkspaceDataIssueApplication,
    )

    manifest = _manifest(20)
    manifest = replace(
        manifest, profile=replace(manifest.profile, market_profile_id="us-current-index-research")
    )
    with WorkspaceApplicationSession.acquire(tmp_path / "workspace") as session:
        market = MarketDataRepository(session.workspace)
        panel = PanelStateRepository(market.database, market_data=market)
        service = FeatureInputGovernanceService(
            market, panel, session.mutation_gate, FeatureInputGateway()
        )
        evidence = _evidence(
            manifest, failed={0}, failure_code="data.unexplained_raw_move", extreme=True
        )
        sectors = _sectors(manifest)
        impact = _panel_impact(manifest, sectors)
        facts = dict(
            candidate_manifest=manifest,
            evidence=evidence,
            sector_by_listing_id=sectors,
            temporal_boundary=_temporal(),
            panel_impact=impact,
            observed_at=NOW,
        )
        result = service.assess_and_record(**facts)
        market.readiness.save(
            market_profile_id=manifest.profile.market_profile_id,
            status="FEATURE_BUILDING",
            active_manifest_id=manifest.manifest_id,
            active_manifest_revision=manifest.revision_sha256,
            active_membership_fingerprint=None,
            active_candidate_manifest_document=None,
            pending_membership_fingerprint=None,
            pending_candidate_manifest_document=None,
            last_checked_at=NOW,
            last_changed_at=NOW,
            failure_code=None,
            observed_at=NOW,
        )
        application = WorkspaceDataIssueApplication(session, lambda: NOW)
        issue = application.readback()["issues"][0]
        case = result.agent_cases[0]
        option = next(
            v for v in case.options if v.option_id == "retain_isolated_raw_move_with_caveat"
        )
        choice = dict(
            case_token=case.case_token,
            evidence_hash=case.evidence_hash,
            option_id=option.option_id,
            option_hash=option.option_hash,
        )
        assert issue["status"] == "AWAITING_CHOICE"
        with pytest.raises(ValueError, match="human_confirmation_required"):
            application.confirm(**choice, caller="EXTERNAL_AUTOMATION")
        assert panel.feature_input_resolution(case.case_token) is None
        with pytest.raises(ValueError, match="option_changed"):
            application.confirm(**{**choice, "option_hash": "f" * 64}, caller="HUMAN")

        def fresh(*_args, **_kwargs):
            return SimpleNamespace(
                manifest=manifest,
                result=service.assess_and_record(**facts),
                evidence=evidence,
                panel_impact=impact,
                sector_source=None,
            )

        coordinator = object.__new__(WorkspaceMaintenanceCoordinator)
        coordinator.feature_input = service
        coordinator.panel_state = panel
        coordinator.mutation_gate = session.mutation_gate
        coordinator.diagnose = None
        coordinator._govern_quality = fresh
        coordinator._bind_derived_manifest_evidence = lambda *_args, **_kwargs: True
        coordinator._bind_manifest = lambda *_args, **_kwargs: None
        coordinator.feature_state = SimpleNamespace(
            current_sector_state=lambda _: SimpleNamespace(
                sector_distribution=impact.sector_distribution,
                sector_by_listing_id=sectors,
            )
        )
        quarantine = next(v for v in case.options if v.option_id == "recoverable_quarantine")
        rejected_choice = {
            **choice,
            "option_id": quarantine.option_id,
            "option_hash": quarantine.option_hash,
        }
        application.confirm(**rejected_choice, caller="HUMAN")
        refusal = coordinator._handle_governance_result(
            fresh(),
            maintenance_id=None,
            request=SimpleNamespace(target_market_session=AS_OF),
            observed_at=NOW,
            observed_workers=1,
        )
        assert refusal[2] == "data.truth_review_required"
        assert panel.has_unresolved_feature_input(manifest.revision_sha256)
        assert application.readback()["issues"][0]["status"] == "OPTION_REFUSED"
        assert application.confirm(**rejected_choice, caller="HUMAN")["status"] == "REFUSED"
        confirmed = application.confirm(**choice, caller="HUMAN")
        assert confirmed["status"] == "CONFIRMED_PENDING_REVALIDATION"
        assert application.confirm(**choice, caller="HUMAN") == confirmed
        outcome = coordinator._handle_governance_result(
            fresh(),
            maintenance_id=None,
            request=SimpleNamespace(target_market_session=AS_OF),
            observed_at=NOW,
            observed_workers=1,
        )
        assert outcome is None  # Continue the same maintenance command, not a second retry.
        repeated = application.confirm(**choice, caller="HUMAN")
        assert repeated["status"] == "ALREADY_APPLIED"
        assert repeated["receipt_hash"] == confirmed["receipt_hash"]
        assert len(panel.feature_input_resolution(case.case_token)["prior_decisions"]) == 1
        cleared = service.assess_and_record(**facts)
        assert cleared.assessment is FeatureInputAssessment.ADMITTED
        assert cleared.admission.caveat_receipts[0][0] == evidence[0].listing_id
        disclosure = panel.feature_input_quality_disclosure(
            result_manifest_revision=cleared.research_manifest.revision_sha256
        )
        assert disclosure["caveated_listing_count"] == 1
        assert not panel.has_unresolved_feature_input(manifest.revision_sha256)
        assert evidence[0].reason_codes == (
            "UNEXPLAINED_RAW_MOVE",
        )  # source evidence not rewritten
        changed = replace(evidence[0], evidence_hash="c" * 64)
        disputed = service.assess_and_record(**{**facts, "evidence": (changed, *evidence[1:])})
        assert disputed.assessment is FeatureInputAssessment.DEFERRED
        assert disputed.agent_cases[0].evidence_hash != case.evidence_hash
        # Waiting is a consumed timer, not quality clearance or an endlessly
        # renewed decision. A new cycle must still see this unresolved issue.
        wait_case = disputed.agent_cases[0]
        wait_option = next(
            v for v in wait_case.options if v.option_id == "wait_for_provider_recovery"
        )
        application.confirm(
            case_token=wait_case.case_token,
            evidence_hash=wait_case.evidence_hash,
            option_id=wait_option.option_id,
            option_hash=wait_option.option_hash,
            caller="HUMAN",
        )
        wait_effect = FeatureInputRemediationExecutor(service.gateway).execute(
            case=wait_case,
            option=wait_option,
            evidence_by_listing_id={v.listing_id: v for v in (changed, *evidence[1:])},
            observed_at=NOW,
        )
        from pydantic import TypeAdapter

        panel.record_feature_input_effect(
            wait_case.case_token,
            TypeAdapter(type(wait_effect)).dump_python(wait_effect, mode="json"),
        )
        assert panel.has_unresolved_feature_input(manifest.revision_sha256)
        assert application.readback()["issues"][0]["status"] == "WAITING_FOR_RETRY"
        # Once the wait has elapsed it is a decision that ran its course: the
        # case is a choice again, the wait reads back among the prior
        # decisions, and the same option may be chosen once more.
        elapsed = WorkspaceDataIssueApplication(
            session, lambda: NOW + timedelta(minutes=6)
        ).readback()["issues"][0]
        assert elapsed["status"] == "AWAITING_CHOICE" and elapsed["resolution"] is None
        assert [
            item["receipt"]["policy_decision"]["option_id"] for item in elapsed["prior_decisions"]
        ] == ["wait_for_provider_recovery"]
        assert (
            f"preview:{wait_case.case_token}:wait_for_provider_recovery"
            in (
                WorkspaceDataIssueApplication(
                    session, lambda: NOW + timedelta(minutes=6)
                ).readback()["next_requests"]
            )
        )
        after_wait = service.assess_and_record(
            **{
                **facts,
                "evidence": (changed, *evidence[1:]),
                "observed_at": NOW + timedelta(minutes=6),
            }
        )
        assert after_wait.agent_cases[0].case_token != wait_case.case_token
        assert after_wait.agent_cases[0].issue_hash == wait_case.issue_hash
        assert len(panel.feature_input_case_documents(manifest.revision_sha256)) == 1


def test_raw_retention_choice_continues_only_for_the_same_case_on_a_derived_manifest(
    tmp_path,
) -> None:
    from pydantic import TypeAdapter

    from alphalattice.control.data_platform.remediation_case import (
        raw_retention_decision_proofs,
    )
    from alphalattice.control.product_host.composition.application_session import (
        WorkspaceApplicationSession,
    )
    from alphalattice.control.product_host.data_preparation.remediation import (
        WorkspaceDataIssueApplication,
    )

    manifest = replace(
        _manifest(40),
        profile=replace(_manifest(40).profile, market_profile_id="us-current-index-research"),
        membership_fingerprint=canonical_hash("first-use-membership"),
    )
    evidence = _evidence(
        manifest,
        failed={0, 1, 10, 11, 20, 21},
        failure_code="data.unexplained_raw_move",
        extreme=True,
    )
    with WorkspaceApplicationSession.acquire(tmp_path / "workspace") as session:
        market = MarketDataRepository(session.workspace)
        panel = PanelStateRepository(market.database, market_data=market)
        service = FeatureInputGovernanceService(
            market, panel, session.mutation_gate, FeatureInputGateway()
        )
        sectors = _sectors(manifest)
        first_result = service.assess_and_record(
            candidate_manifest=manifest,
            evidence=evidence,
            sector_by_listing_id=sectors,
            temporal_boundary=_temporal(),
            panel_impact=_panel_impact(manifest, sectors),
            observed_at=NOW,
        )
        old_cases = first_result.agent_cases
        assert len(old_cases) == 6
        assert all(case.case_kind == "LISTING" and len(case.listing_ids) == 1 for case in old_cases)
        market.readiness.save(
            market_profile_id=manifest.profile.market_profile_id,
            status="FEATURE_BUILDING",
            active_manifest_id=manifest.manifest_id,
            active_manifest_revision=manifest.revision_sha256,
            active_membership_fingerprint=manifest.membership_fingerprint,
            active_candidate_manifest_document=None,
            pending_membership_fingerprint=None,
            pending_candidate_manifest_document=None,
            last_checked_at=NOW,
            last_changed_at=NOW,
            failure_code=None,
            observed_at=NOW,
        )
        application = WorkspaceDataIssueApplication(session, lambda: NOW)
        original_resolutions = {}
        for case in old_cases:
            option = next(
                value
                for value in case.options
                if value.option_id == "retain_isolated_raw_move_with_caveat"
            )
            confirmation = application.confirm(
                case_token=case.case_token,
                evidence_hash=case.evidence_hash,
                option_id=option.option_id,
                option_hash=option.option_hash,
                caller="HUMAN",
            )
            effect = FeatureInputRemediationExecutor(service.gateway).execute(
                case=case,
                option=option,
                evidence_by_listing_id={item.listing_id: item for item in case.evidence},
                observed_at=NOW,
                proposal_hash=confirmation["receipt_hash"],
                human_confirmed=True,
            )
            panel.record_feature_input_effect(
                case.case_token,
                TypeAdapter(type(effect)).dump_python(effect, mode="json"),
            )
            original_resolutions[case.case_token] = panel.feature_input_resolution(case.case_token)

        child = build_quality_filtered_research_manifest(
            manifest,
            eligible_listing_ids=tuple(item.listing_id for item in manifest.listings[:-1]),
        )
        child_evidence = _evidence(
            child,
            failed={0, 1, 10, 11, 20, 21},
            failure_code="data.unexplained_raw_move",
            extreme=True,
        )
        child_sectors = _sectors(child)
        child_facts = dict(
            candidate_manifest=child,
            evidence=child_evidence,
            sector_by_listing_id=child_sectors,
            temporal_boundary=_temporal(),
            panel_impact=_panel_impact(child, child_sectors),
            observed_at=NOW + timedelta(minutes=1),
        )
        pending = service.assess_and_record(**child_facts)
        assert len(pending.agent_cases) == 6
        market.readiness.save(
            market_profile_id=child.profile.market_profile_id,
            status="FEATURE_BUILDING",
            active_manifest_id=child.manifest_id,
            active_manifest_revision=child.revision_sha256,
            active_membership_fingerprint=child.membership_fingerprint,
            active_candidate_manifest_document=None,
            pending_membership_fingerprint=None,
            pending_candidate_manifest_document=None,
            last_checked_at=NOW + timedelta(minutes=1),
            last_changed_at=NOW + timedelta(minutes=1),
            failure_code=None,
            observed_at=NOW + timedelta(minutes=1),
        )
        proofs = raw_retention_decision_proofs(panel, market)
        assert len(proofs) == 6
        assert application.current_decisions_ready()
        issue_readback = application.readback()
        assert all(issue["continued_decision"] for issue in issue_readback["issues"])
        assert all(
            issue["status"] == "CONFIRMED_PENDING_REVALIDATION"
            for issue in issue_readback["issues"]
        )
        assert all(len(issue["prior_decisions"]) == 1 for issue in issue_readback["issues"])
        assert not any(
            key.startswith(("preview:", "delegate:")) for key in issue_readback["next_requests"]
        )
        current_by_listing = {case.listing_ids[0]: case for case in pending.agent_cases}
        old_by_listing = {case.listing_ids[0]: case for case in old_cases}
        proof_by_listing = {proof.case.listing_ids[0]: proof for proof in proofs}
        for listing_id, current_case in current_by_listing.items():
            matched = matching_raw_retention_proof(
                case=current_case,
                candidate_manifest=child,
                proofs=proofs,
                policy_hash=service.gateway.policy.policy_hash,
            )
            assert (
                matched is not None
                and matched.case.case_token == old_by_listing[listing_id].case_token
            )
            prior = issue_readback["issues"][
                next(
                    index
                    for index, item in enumerate(issue_readback["issues"])
                    if item["case"]["case_token"] == current_case.case_token
                )
            ]["prior_decisions"][0]
            old_resolution = original_resolutions[old_by_listing[listing_id].case_token]
            assert prior["receipt"]["receipt_hash"] == old_resolution["receipt"]["receipt_hash"]
            assert prior["continuation_receipt_hash"] == old_resolution["receipt"]["receipt_hash"]

        first_case = current_by_listing[old_cases[0].listing_ids[0]]
        first_proof = proof_by_listing[old_cases[0].listing_ids[0]]
        changed_evidence = replace(child_evidence[0], evidence_hash="e" * 64)
        changed_result = service.gateway.assess(
            candidate_manifest=child,
            evidence=(changed_evidence, *child_evidence[1:]),
            sector_by_listing_id=child_sectors,
            temporal_boundary=_temporal(),
            observed_at=NOW,
            panel_impact=_panel_impact(child, child_sectors),
        )
        changed_case = next(
            case
            for case in changed_result.agent_cases
            if case.listing_ids == first_case.listing_ids
        )
        assert (
            matching_raw_retention_proof(
                case=changed_case,
                candidate_manifest=child,
                proofs=proofs,
                policy_hash=service.gateway.policy.policy_hash,
            )
            is None
        )

        broader_evidence = _evidence(
            child,
            failed={0, 1, 2, 3, 10, 11, 12, 20, 21, 22},
            failure_code="data.unexplained_raw_move",
            extreme=True,
        )
        broader_result = service.gateway.assess(
            candidate_manifest=child,
            evidence=broader_evidence,
            sector_by_listing_id=child_sectors,
            temporal_boundary=_temporal(),
            observed_at=NOW,
            panel_impact=_panel_impact(child, child_sectors),
        )
        aggregate_case = broader_result.agent_cases[0]
        assert aggregate_case.case_kind == "AGGREGATED"
        assert (
            matching_raw_retention_proof(
                case=aggregate_case,
                candidate_manifest=child,
                proofs=proofs,
                policy_hash=service.gateway.policy.policy_hash,
            )
            is None
        )

        alternate_policy = FeatureInputPolicy(provider_retry_delay_seconds=301)
        alternate_gateway = FeatureInputGateway(alternate_policy)
        policy_result = alternate_gateway.assess(
            candidate_manifest=child,
            evidence=child_evidence,
            sector_by_listing_id=child_sectors,
            temporal_boundary=_temporal(),
            observed_at=NOW,
            panel_impact=_panel_impact(child, child_sectors),
        )
        policy_case = next(
            case for case in policy_result.agent_cases if case.listing_ids == first_case.listing_ids
        )
        assert (
            matching_raw_retention_proof(
                case=policy_case,
                candidate_manifest=child,
                proofs=proofs,
                policy_hash=alternate_gateway.policy.policy_hash,
            )
            is None
        )
        assert (
            matching_raw_retention_proof(
                case=first_case,
                candidate_manifest=replace(child, membership_fingerprint="f" * 64),
                proofs=proofs,
                policy_hash=service.gateway.policy.policy_hash,
            )
            is None
        )
        assert (
            matching_raw_retention_proof(
                case=first_case,
                candidate_manifest=child,
                proofs=(replace(first_proof, actor_kind="UNKNOWN"),),
                policy_hash=service.gateway.policy.policy_hash,
            )
            is None
        )
        assert (
            matching_raw_retention_proof(
                case=first_case,
                candidate_manifest=child,
                proofs=(replace(first_proof, option_hash="f" * 64),),
                policy_hash=service.gateway.policy.policy_hash,
            )
            is None
        )
        assert (
            matching_raw_retention_proof(
                case=first_case,
                candidate_manifest=child,
                proofs=(
                    first_proof,
                    replace(first_proof, receipt_hash=canonical_hash("different receipt")),
                ),
                policy_hash=service.gateway.policy.policy_hash,
            )
            is None
        )

        continued = service.assess_and_record(
            **child_facts,
            raw_retention_proofs=proofs,
        )
        assert continued.assessment is FeatureInputAssessment.ADMITTED
        assert len(continued.admission.caveat_receipts) == 6
        expected_execution_receipts = {
            resolution["effect"]["execution_receipt_hash"]
            for resolution in original_resolutions.values()
        }
        assert {receipt for _listing, receipt in continued.admission.caveat_receipts} == (
            expected_execution_receipts
        )
        assert all(
            panel.feature_input_resolution(case.case_token) == original_resolutions[case.case_token]
            for case in old_cases
        )
        assert not panel.has_unresolved_feature_input(child.revision_sha256)

        # A second independently sealed receipt for the same per-listing choice is ambiguous.
        second_child = build_quality_filtered_research_manifest(
            manifest,
            eligible_listing_ids=tuple(item.listing_id for item in manifest.listings[:-2]),
        )
        second_evidence = _evidence(
            second_child,
            failed={0, 1, 10, 11, 20, 21},
            failure_code="data.unexplained_raw_move",
            extreme=True,
        )
        second_sectors = _sectors(second_child)
        second_result = service.assess_and_record(
            candidate_manifest=second_child,
            evidence=second_evidence,
            sector_by_listing_id=second_sectors,
            temporal_boundary=_temporal(),
            panel_impact=_panel_impact(second_child, second_sectors),
            observed_at=NOW + timedelta(minutes=2),
        )
        second_case = next(
            case
            for case in second_result.agent_cases
            if case.listing_ids == old_cases[0].listing_ids
        )
        market.readiness.save(
            market_profile_id=second_child.profile.market_profile_id,
            status="FEATURE_BUILDING",
            active_manifest_id=second_child.manifest_id,
            active_manifest_revision=second_child.revision_sha256,
            active_membership_fingerprint=second_child.membership_fingerprint,
            active_candidate_manifest_document=None,
            pending_membership_fingerprint=None,
            pending_candidate_manifest_document=None,
            last_checked_at=NOW + timedelta(minutes=2),
            last_changed_at=NOW + timedelta(minutes=2),
            failure_code=None,
            observed_at=NOW + timedelta(minutes=2),
        )
        second_application = WorkspaceDataIssueApplication(session, lambda: NOW)
        second_option = next(
            value
            for value in second_case.options
            if value.option_id == "retain_isolated_raw_move_with_caveat"
        )
        second_confirmation = second_application.confirm(
            case_token=second_case.case_token,
            evidence_hash=second_case.evidence_hash,
            option_id=second_option.option_id,
            option_hash=second_option.option_hash,
            caller="HUMAN",
        )
        second_effect = FeatureInputRemediationExecutor(service.gateway).execute(
            case=second_case,
            option=second_option,
            evidence_by_listing_id={item.listing_id: item for item in second_case.evidence},
            observed_at=NOW + timedelta(minutes=2),
            proposal_hash=second_confirmation["receipt_hash"],
            human_confirmed=True,
        )
        panel.record_feature_input_effect(
            second_case.case_token,
            TypeAdapter(type(second_effect)).dump_python(second_effect, mode="json"),
        )
        second_proofs = raw_retention_decision_proofs(panel, market)
        same_listing_proofs = tuple(
            proof for proof in second_proofs if proof.case.listing_ids == first_case.listing_ids
        )
        assert len({proof.receipt_hash for proof in same_listing_proofs}) == 2
        assert (
            matching_raw_retention_proof(
                case=first_case,
                candidate_manifest=child,
                proofs=same_listing_proofs,
                policy_hash=service.gateway.policy.policy_hash,
            )
            is None
        )


def test_undated_quarantine_readback_does_not_migrate_or_invent_past_scope(tmp_path) -> None:
    manifest = _manifest(5)
    market = MarketDataRepository(tmp_path / "legacy")
    market.bootstrap(manifest)
    panel = PanelStateRepository(market.database, market_data=market)
    quarantine = FeatureInputGateway().quarantine(
        evidence=_evidence(manifest, failed={0})[0],
        recheck_after_at=NOW + timedelta(days=1),
        execution_receipt_hash="6" * 64,
    )
    panel.record_listing_quarantines(
        candidate_manifest_revision=manifest.revision_sha256,
        quarantines=(quarantine,),
        observed_at=NOW,
    )
    with market._connect() as connection:
        connection.execute("ALTER TABLE listing_quarantine DROP COLUMN effective_session")
        connection.execute("ALTER TABLE listing_quarantine DROP COLUMN cleared_effective_session")
    assert panel.active_listing_quarantines(manifest.revision_sha256) == (quarantine,)
    scope = panel.panel_source_exclusions(
        market_profile_id=manifest.profile.market_profile_id,
        sessions=(AS_OF - timedelta(days=10), AS_OF),
        listing_ids=tuple(item.listing_id for item in manifest.listings),
    )
    assert scope[0].first_session == scope[0].last_session == AS_OF
    assert "LEGACY_UNDATED_QUALITY_SCOPE" in scope[0].reason_codes
    with market._connect(read_only=True) as connection:
        assert "effective_session" not in {
            row[1]
            for row in connection.execute("PRAGMA table_info('listing_quarantine')").fetchall()
        }


def test_due_quarantine_requalifies_from_fresh_deterministic_evidence(tmp_path) -> None:
    manifest = _manifest(5)
    sectors = _sectors(manifest, size=5)
    market_data = MarketDataRepository(tmp_path / "workspace")
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    gate = WorkspaceMutationGate()
    gateway = FeatureInputGateway()
    service = FeatureInputGovernanceService(
        market_data=market_data,
        panel_state=panel_state,
        mutation_gate=gate,
        gateway=gateway,
    )
    failed = _evidence(manifest, failed={0})[0]
    quarantine = gateway.quarantine(
        evidence=failed,
        recheck_after_at=NOW,
        execution_receipt_hash="e" * 64,
    )
    gate.run(market_data.bootstrap, manifest)
    gate.run(
        panel_state.record_listing_quarantines,
        candidate_manifest_revision=manifest.revision_sha256,
        quarantines=(quarantine,),
        observed_at=NOW - timedelta(days=1),
    )
    passing = tuple(
        replace(
            item,
            qualification_receipt_hash=("q" * 64 if index == 0 else None),
        )
        for index, item in enumerate(_evidence(manifest))
    )

    result = service.assess_and_record(
        candidate_manifest=manifest,
        evidence=passing,
        sector_by_listing_id=sectors,
        temporal_boundary=_temporal(cutoff=NOW + timedelta(minutes=1)),
        panel_impact=_panel_impact(manifest, sectors),
        observed_at=NOW + timedelta(minutes=1),
    )

    assert result.assessment is FeatureInputAssessment.ADMITTED
    assert result.admission is not None
    assert len(result.admission.admitted_listing_ids) == 5
    assert panel_state.active_listing_quarantines(manifest.revision_sha256) == ()


@pytest.mark.parametrize("first", ("MARKET_DATA", "BASE_FEATURES", "SECTOR_REFERENCE"))
def test_quarantine_clearance_cannot_satisfy_the_other_qualification_layer(tmp_path, first):
    from alphalattice.foundation.feature_engine.inputs.contracts import ListingQuarantine

    manifest = _manifest(5)
    market = MarketDataRepository(tmp_path)
    market.bootstrap(manifest)
    store = PanelStateRepository(market.database, market_data=market)
    records = {
        domain: ListingQuarantine.create(
            listing_id=manifest.listings[0].listing_id,
            reason_codes=(reason,),
            evidence_hash="a" * 64,
            execution_receipt_hash="b" * 64,
            recheck_after_at=NOW,
        )
        for domain, reason in (
            ("MARKET_DATA", "PROVIDER_SOURCE_MISSING"),
            ("BASE_FEATURES", "BASE_FEATURE_UNAVAILABLE:cmf_21:zero_denominator"),
            ("SECTOR_REFERENCE", "SECTOR_CLASSIFICATION_UNAVAILABLE"),
        )
    }
    for index, item in enumerate(records.values()):
        store.record_listing_quarantines(
            candidate_manifest_revision=manifest.revision_sha256,
            quarantines=(item,),
            observed_at=NOW - timedelta(minutes=2 - index),
        )
    for domain, item in records.items():
        assert store.active_listing_quarantines(
            manifest.revision_sha256, qualification_domain=domain
        ) == (item,)
    other = next(domain for domain in records if domain != first)
    with pytest.raises(ValueError, match="clearance_domain_mismatch"):
        store.clear_listing_quarantine(
            quarantine_hash=records[first].quarantine_hash,
            qualification_receipt_hash="c" * 64,
            cleared_at=NOW,
            qualification_domain=other,
        )
    store.clear_listing_quarantine(
        quarantine_hash=records[first].quarantine_hash,
        qualification_receipt_hash="c" * 64,
        cleared_at=NOW,
        qualification_domain=first,
    )
    assert (
        store.active_listing_quarantines(manifest.revision_sha256, qualification_domain=first) == ()
    )
    for other in records.keys() - {first}:
        assert store.active_listing_quarantines(
            manifest.revision_sha256, qualification_domain=other
        ) == (records[other],)


def test_provider_and_sector_macro_circuit_breakers_defer_before_agent() -> None:
    manifest = _manifest(100)
    sectors = _sectors(manifest)
    gateway = FeatureInputGateway()
    provider_macro = gateway.assess(
        candidate_manifest=manifest,
        evidence=_evidence(manifest, failed=set(range(25))),
        sector_by_listing_id=sectors,
        temporal_boundary=_temporal(),
        observed_at=NOW,
        observed_workers=4,
    )
    assert provider_macro.assessment is FeatureInputAssessment.PROVIDER_COHORT_DEFERRED
    assert provider_macro.deferred is not None
    assert provider_macro.deferred.next_workers == 2
    assert provider_macro.agent_cases == ()

    sector_macro = gateway.assess(
        candidate_manifest=manifest,
        evidence=_evidence(manifest, failed=set(range(5))),
        sector_by_listing_id=sectors,
        temporal_boundary=_temporal(),
        observed_at=NOW,
    )
    assert sector_macro.assessment is FeatureInputAssessment.PROVIDER_COHORT_DEFERRED
    assert len(sector_macro.deferred.affected_listing_ids) == 5  # type: ignore[union-attr]


def test_macro_retry_exhaustion_creates_one_marketwide_case() -> None:
    manifest = _manifest(100)
    result = FeatureInputGateway().assess(
        candidate_manifest=manifest,
        evidence=_evidence(manifest, failed=set(range(25))),
        sector_by_listing_id=_sectors(manifest),
        temporal_boundary=_temporal(),
        observed_at=NOW,
        macro_retry_exhausted=True,
        last_known_good_snapshot_ref=(f"playpen://feature-panel/manifests/{'b' * 64}"),
    )
    assert len(result.agent_cases) == 1
    case = result.agent_cases[0]
    assert case.case_kind == "MARKETWIDE"
    assert {item.option_id for item in case.options} >= {
        "wait_for_provider_recovery",
        "retain_last_known_good_frozen_snapshot",
    }
    assert "recoverable_quarantine" not in {item.option_id for item in case.options}


def test_ordinary_failures_aggregate_once_without_triggering_macro() -> None:
    manifest = _manifest(200)
    failed = {index * 10 for index in range(20)}
    result = FeatureInputGateway().assess(
        candidate_manifest=manifest,
        evidence=_evidence(manifest, failed=failed),
        sector_by_listing_id=_sectors(manifest),
        temporal_boundary=_temporal(),
        observed_at=NOW,
    )
    assert len(result.agent_cases) == 1
    assert result.agent_cases[0].case_kind == "AGGREGATED"
    assert len(result.agent_cases[0].listing_ids) == 20


def test_temporal_cutoff_changes_admission_identity_but_not_quality_evidence() -> None:
    manifest = _manifest(20)
    evidence = _evidence(manifest)
    gateway = FeatureInputGateway()
    first = gateway.assess(
        candidate_manifest=manifest,
        evidence=evidence,
        sector_by_listing_id=_sectors(manifest, size=5),
        temporal_boundary=_temporal(cutoff=NOW),
        observed_at=NOW,
    )
    second = gateway.assess(
        candidate_manifest=manifest,
        evidence=evidence,
        sector_by_listing_id=_sectors(manifest, size=5),
        temporal_boundary=_temporal(cutoff=NOW + timedelta(minutes=1)),
        observed_at=NOW + timedelta(minutes=1),
    )
    assert first.admission is not None and second.admission is not None
    assert first.admission.admission_hash != second.admission.admission_hash
    assert first.admission.temporal_identity_hash != second.admission.temporal_identity_hash
    assert tuple(item.evidence_hash for item in evidence) == tuple(
        item.evidence_hash for item in evidence
    )


def test_forged_stale_and_repeatedly_churning_agent_evidence_fails_closed() -> None:
    manifest = _manifest(20)
    gateway = FeatureInputGateway()
    original = _evidence(manifest, failed={0})[0]
    result = gateway.assess(
        candidate_manifest=manifest,
        evidence=tuple(
            original if index == 0 else item for index, item in enumerate(_evidence(manifest))
        ),
        sector_by_listing_id=_sectors(manifest, size=5),
        temporal_boundary=_temporal(),
        observed_at=NOW,
    )
    case = result.agent_cases[0]
    with pytest.raises(ValueError, match="forged"):
        gateway.validate_agent_selection(
            case,
            run_id=case.run_id,
            case_token="f" * 64,
            evidence_hash=case.evidence_hash,
            option_id=case.options[0].option_id,
            current_evidence_hash=case.evidence_hash,
            rediagnosis_count=0,
        )
    changed_once = replace(original, evidence_hash="1" * 64)
    rediagnosed = gateway.reconcile_stale_case(
        candidate_manifest=manifest,
        prior_case=case,
        current_evidence=(changed_once,),
        observed_at=NOW,
    )
    assert rediagnosed.rediagnosis_count == 1  # type: ignore[union-attr]
    changed_twice = replace(original, evidence_hash="2" * 64)
    deferred = gateway.reconcile_stale_case(
        candidate_manifest=manifest,
        prior_case=rediagnosed,  # type: ignore[arg-type]
        current_evidence=(changed_twice,),
        observed_at=NOW,
    )
    assert deferred.failure_code == "data.evidence_churn"  # type: ignore[union-attr]
    assert deferred.retry_after_at == NOW + timedelta(minutes=5)  # type: ignore[union-attr]


def test_quality_evaluator_flags_only_unexplained_raw_move_and_5bps_mismatch() -> None:
    admission = CurrentUniverseQualityAdmission(
        onboarding_id="onboarding-a",
        listing_id="listing-000",
        eligible=True,
        expected_sessions=3,
        observed_sessions=3,
        missing_sessions=0,
        missing_ratio=0.0,
        maximum_consecutive_gap=0,
        reasons=(),
        evaluated_at=NOW,
    )
    raw = RawCloseSeries.of(
        (
            (date(2026, 7, 29), 100.0),
            (date(2026, 7, 30), 40.0),
            (date(2026, 7, 31), 41.0),
        )
    )
    evaluator = FeatureInputQualityEvaluator()
    evaluation = dict(
        admission=admission,
        provider="fixture",
        range_start=raw.session(0),
        range_end=raw.session(-1),
        raw_closes=raw,
        corporate_action_sessions=frozenset({raw.session(1)}),
        action_audit_completed=True,
        adjusted_diagnostic_max_bps=4.9,
        identity_verified=True,
        retry_exhausted=True,
    )
    explained = evaluator.evaluate(**evaluation)
    assert explained.failure_code is None
    assert explained.action_explained is True
    action_before = evaluator.evaluate(**evaluation, action_evidence_hash="a" * 64)
    action_after = evaluator.evaluate(**evaluation, action_evidence_hash="b" * 64)
    assert action_before.evidence_hash != action_after.evidence_hash
    assert action_before.reason_codes == action_after.reason_codes == explained.reason_codes

    unexplained = evaluator.evaluate(
        admission=admission,
        provider="fixture",
        range_start=raw.session(0),
        range_end=raw.session(-1),
        raw_closes=raw,
        corporate_action_sessions=frozenset(),
        action_audit_completed=True,
        adjusted_diagnostic_max_bps=5.1,
        identity_verified=True,
        retry_exhausted=True,
    )
    assert unexplained.failure_code == "data.adjusted_close_mismatch"
    assert set(unexplained.reason_codes) == {
        "ADJUSTED_CLOSE_DIAGNOSTIC_MISMATCH",
        "UNEXPLAINED_RAW_MOVE",
    }
    corrected = evaluator.evaluate(
        admission=admission,
        provider="fixture",
        range_start=raw.session(0),
        range_end=raw.session(-1),
        raw_closes=RawCloseSeries.of(
            (
                (date(2026, 7, 29), 100.0),
                (date(2026, 7, 30), 101.0),
                (date(2026, 7, 31), 102.0),
            )
        ),
        corporate_action_sessions=frozenset(),
        action_audit_completed=True,
        adjusted_diagnostic_max_bps=4.9,
        identity_verified=True,
        retry_exhausted=True,
        provider_correction_observed=True,
    )
    assert corrected.failure_code is None


def test_quarantine_is_recoverable_and_panel_preflight_blocks_bad_universe() -> None:
    manifest = _manifest(20)
    sectors = _sectors(manifest, size=5)
    gateway = FeatureInputGateway()
    failing = _evidence(manifest, failed={0})[0]
    quarantine = gateway.quarantine(
        evidence=failing,
        recheck_after_at=NOW + timedelta(days=1),
        execution_receipt_hash="e" * 64,
    )
    bad_distribution = {"sector-00": 4, "sector-01": 5, "sector-02": 5, "sector-03": 5}
    blocked = gateway.assess(
        candidate_manifest=manifest,
        evidence=_evidence(manifest),
        sector_by_listing_id=sectors,
        temporal_boundary=_temporal(),
        observed_at=NOW,
        panel_impact=PanelImpactProjection(bad_distribution, 19 / 20, 60, True),
        existing_quarantines=(quarantine,),
    )
    assert blocked.assessment is FeatureInputAssessment.DEFERRED
    assert "PANEL_SECTOR_BELOW_MINIMUM" in blocked.failure_reasons
    passing = ListingQualityEvidence(
        listing_id=failing.listing_id,
        provider="fixture",
        range_start=failing.range_start,
        range_end=failing.range_end,
        evidence_hash="f" * 64,
        qualification_receipt_hash="1" * 64,
    )
    decision = gateway.requalify(quarantine, passing, observed_at=NOW + timedelta(days=1))
    assert decision.assessment is FeatureInputAssessment.ADMITTED


def _anomaly_evidence(
    *,
    closes: tuple[tuple[date, float], ...],
    actions: frozenset[date] = frozenset(),
    audit_completed: bool = True,
    identity_verified: bool = True,
    correction: bool = False,
    policy: FeatureInputQualityPolicy | None = None,
) -> ListingQualityEvidence:
    """The evaluator's own evidence over one raw history, as the coordinator builds it."""
    admission = CurrentUniverseQualityAdmission(
        onboarding_id="onboarding-a",
        listing_id="listing-000",
        eligible=True,
        expected_sessions=len(closes),
        observed_sessions=len(closes),
        missing_sessions=0,
        missing_ratio=0.0,
        maximum_consecutive_gap=0,
        reasons=(),
        evaluated_at=NOW,
    )
    return FeatureInputQualityEvaluator(policy).evaluate(
        admission=admission,
        provider="fixture",
        range_start=closes[0][0],
        range_end=closes[-1][0],
        raw_closes=RawCloseSeries.of(closes),
        corporate_action_sessions=actions,
        action_audit_completed=audit_completed,
        adjusted_diagnostic_max_bps=1.0,
        identity_verified=identity_verified,
        retry_exhausted=True,
        provider_correction_observed=correction,
    )


DAY_ONE = (
    (date(2026, 7, 28), 100.0),
    (date(2026, 7, 29), 101.0),
    (date(2026, 7, 30), 102.0),
    (date(2026, 7, 31), 204.0),  # the jump, kept from here on
)
DAY_TWO = (*DAY_ONE, (date(2026, 8, 3), 205.0))


def _quarantine_decision(
    gateway: FeatureInputGateway, evidence: ListingQualityEvidence, *, option_id: str
):
    """Day one: the case the Gateway raises and the executed quarantine for it."""
    manifest = _manifest(20)
    sectors = _sectors(manifest)
    assert evidence.listing_id == manifest.listings[0].listing_id
    result = gateway.assess(
        candidate_manifest=manifest,
        evidence=(evidence, *_evidence(manifest)[1:]),
        sector_by_listing_id=sectors,
        temporal_boundary=_temporal(),
        observed_at=NOW,
    )
    (case,) = result.agent_cases
    option = next(item for item in case.options if item.option_id == option_id)
    execution = FeatureInputRemediationExecutor(gateway).execute(
        case=case,
        option=option,
        evidence_by_listing_id={evidence.listing_id: evidence},
        observed_at=NOW,
        panel_impact_after_exclusion=_panel_impact(manifest, sectors),
    )
    assert execution.status is FeatureInputExecutionStatus.QUARANTINE_READY
    (quarantine,) = execution.quarantines
    return case, execution, quarantine


def test_evaluator_signs_the_anomaly_independently_of_the_window() -> None:
    """requirement: an unchanged anomaly reads the same on a later day; a revised one does not."""
    day_one = _anomaly_evidence(closes=DAY_ONE)
    day_two = _anomaly_evidence(closes=DAY_TWO)
    assert day_one.failure_code == day_two.failure_code == "data.unexplained_raw_move"
    assert day_one.evidence_hash != day_two.evidence_hash  # the window moved
    assert (
        day_one.unexplained_moves
        == day_two.unexplained_moves
        == (UnexplainedRawMove(date(2026, 7, 30), 102.0, date(2026, 7, 31), 204.0),)
    )
    assert day_one.anomaly_signature_hash == day_two.anomaly_signature_hash
    revised = _anomaly_evidence(closes=(*DAY_ONE[:3], (date(2026, 7, 31), 203.0)))
    assert revised.anomaly_signature_hash != day_one.anomaly_signature_hash
    explained = _anomaly_evidence(closes=DAY_TWO, actions=frozenset({date(2026, 7, 31)}))
    assert explained.admitted and explained.unexplained_moves == ()
    assert explained.anomaly_signature_hash != day_one.anomaly_signature_hash
    stricter = _anomaly_evidence(
        closes=DAY_TWO, policy=FeatureInputQualityPolicy(unexplained_raw_move_fraction=0.4)
    )
    assert stricter.unexplained_moves == day_one.unexplained_moves
    assert stricter.anomaly_signature_hash != day_one.anomaly_signature_hash
    with pytest.raises(ValueError, match="unexplained moves must name"):
        replace(day_one, unexplained_sessions=(date(2026, 7, 30),))


def test_a_due_quarantine_is_continued_only_over_the_same_anomaly() -> None:
    """requirement: recheck due is not automatic expiry, and not automatic renewal either.

    The original decision is real and executed, the listing is the same
    source, the anomaly is the same two observations with the same
    explanation, nothing new was observed, the policy is the one decided
    under, and the listing still fails: the quarantine is carried to the
    next recheck under the original receipt, without a new decision. Every
    other reading names why it is not, and the listing re-enters the
    existing path.
    """
    gateway = FeatureInputGateway()
    day_one = _anomaly_evidence(closes=DAY_ONE)
    case, execution, quarantine = _quarantine_decision(
        gateway, day_one, option_id="recoverable_quarantine"
    )
    day_two_at = NOW + timedelta(days=1)
    assert quarantine.recheck_after_at <= day_two_at

    def judge(current, *, at=day_two_at, original_case=case, option="recoverable_quarantine"):
        return gateway.continue_quarantine(
            quarantine=quarantine,
            original_case=original_case,
            original_option_id=option,
            original_execution_policy_hash=gateway.policy.policy_hash,
            original_execution=execution,
            current=current,
            observed_at=at,
        )

    continued = judge(_anomaly_evidence(closes=DAY_TWO))
    assert isinstance(continued, QuarantineContinuation)
    assert continued.rule == "recoverable_quarantine.unchanged_unexplained_move.v1"
    assert continued.original_case_token == case.case_token
    assert continued.original_execution_receipt_hash == execution.execution_receipt_hash
    assert continued.continued_from_quarantine_hash == quarantine.quarantine_hash
    row = continued.quarantine
    assert row.listing_id == quarantine.listing_id
    assert row.reason_codes == quarantine.reason_codes == ("UNEXPLAINED_RAW_MOVE",)
    assert row.execution_receipt_hash == quarantine.execution_receipt_hash
    assert row.continued_from_quarantine_hash == quarantine.quarantine_hash
    assert row.quarantine_hash != quarantine.quarantine_hash
    assert row.evidence_hash == continued.evidence.evidence_hash != quarantine.evidence_hash
    assert row.recheck_after_at == continued.recheck_after_at == NOW + timedelta(days=2)
    # Durable and self-verifying: the document reads back to the same record.
    document = json.dumps(continued.document(), sort_keys=True)
    assert QuarantineContinuation.read_document(document) == continued
    tampered = json.loads(document)
    tampered["evidence"]["evidence_hash"] = "0" * 64
    with pytest.raises(ValueError, match="continuation_identity_mismatch"):
        QuarantineContinuation.read_document(json.dumps(tampered))
    # The reader never seals a document that lacks its identity: modified
    # evidence with the identity removed, emptied or malformed is refused
    # before any value is read, and so is an untouched document without it.
    for value in (None, "", "abc"):  # absent, empty, malformed
        for payload in (tampered, json.loads(document)):
            unsealed = dict(payload)
            if value is None:
                del unsealed["continuation_hash"]
            else:
                unsealed["continuation_hash"] = value
            with pytest.raises(ValueError, match="continuation_identity_absent"):
                QuarantineContinuation.read_document(json.dumps(unsealed))
    # A forger who re-seals modified evidence is refused on the record's own
    # consistency: the row must carry exactly the evidence the recheck
    # examined, and the row's identity must recompute from its fields.
    resealed = dict(tampered)
    resealed["continuation_hash"] = canonical_hash(
        {key: value for key, value in resealed.items() if key != "continuation_hash"}
    )
    with pytest.raises(ValueError, match="continuation_identity_mismatch"):
        QuarantineContinuation.read_document(json.dumps(resealed))
    moved_row = json.loads(document)
    moved_row["quarantine"]["recheck_after_at"] = (
        continued.recheck_after_at + timedelta(days=1)
    ).isoformat()
    moved_row["recheck_after_at"] = moved_row["quarantine"]["recheck_after_at"]
    moved_row["continuation_hash"] = canonical_hash(
        {key: value for key, value in moved_row.items() if key != "continuation_hash"}
    )
    with pytest.raises(ValueError, match="continuation_identity_mismatch"):
        QuarantineContinuation.read_document(json.dumps(moved_row))
    # A continued row is continued again on the same basis (the chain keeps
    # the original receipt), and a row not yet due is not.
    third = gateway.continue_quarantine(
        quarantine=row,
        original_case=case,
        original_option_id="recoverable_quarantine",
        original_execution_policy_hash=gateway.policy.policy_hash,
        original_execution=execution,
        current=_anomaly_evidence(closes=(*DAY_TWO, (date(2026, 8, 4), 206.0))),
        observed_at=NOW + timedelta(days=2),
    )
    assert isinstance(third, QuarantineContinuation)
    assert third.continued_from_quarantine_hash == row.quarantine_hash
    early = judge(_anomaly_evidence(closes=DAY_TWO), at=NOW + timedelta(hours=1))
    assert isinstance(early, QuarantineContinuationRefusal)
    assert early.reasons == ("QUARANTINE_NOT_DUE",)

    counterexamples = {
        "revised_value": (
            _anomaly_evidence(closes=(*DAY_ONE[:3], (date(2026, 7, 31), 203.0))),
            {"ANOMALY_CHANGED"},
        ),
        "revised_adjacent_observation": (
            _anomaly_evidence(closes=(*DAY_ONE[:2], (date(2026, 7, 30), 103.0), DAY_ONE[3])),
            {"ANOMALY_CHANGED"},
        ),
        "new_anomaly": (
            _anomaly_evidence(closes=(*DAY_ONE, (date(2026, 8, 3), 410.0))),
            {"ANOMALY_CHANGED"},
        ),
        "now_explained": (
            _anomaly_evidence(closes=DAY_TWO, actions=frozenset({date(2026, 7, 31)})),
            {"EVIDENCE_ADMITTED"},
        ),
        "new_reason": (
            _anomaly_evidence(closes=DAY_TWO, audit_completed=False),
            {"FAILURE_CODE_CHANGED", "REASON_CODES_CHANGED"},
        ),
        "identity": (
            _anomaly_evidence(closes=DAY_TWO, identity_verified=False),
            {"LISTING_IDENTITY_CHANGED"},
        ),
        "correction": (
            _anomaly_evidence(closes=DAY_TWO, correction=True),
            {"PROVIDER_CORRECTION_OBSERVED"},
        ),
        "history_rebuilt": (
            _anomaly_evidence(closes=DAY_TWO[1:]),
            {"HISTORY_RANGE_START_CHANGED"},
        ),
        "evaluator_policy": (
            _anomaly_evidence(
                closes=DAY_TWO, policy=FeatureInputQualityPolicy(unexplained_raw_move_fraction=0.4)
            ),
            {"POLICY_CHANGED"},
        ),
    }
    for name, (current, expected) in counterexamples.items():
        refused = judge(current)
        assert isinstance(refused, QuarantineContinuationRefusal), name
        assert expected <= set(refused.reasons), (name, refused.reasons)
        assert "ANOMALY_CHANGED" in refused.reasons or name not in {
            "revised_value",
            "revised_adjacent_observation",
            "new_anomaly",
        }
    # The decision itself must be the one continued: another option, another
    # policy, a refused execution, or a decision recorded before evidence
    # carried the signature (re-confirmed once on the existing path).
    other_option = judge(_anomaly_evidence(closes=DAY_TWO), option="exclude_from_next_manifest")
    assert isinstance(other_option, QuarantineContinuationRefusal)
    assert {"ORIGINAL_OPTION_NOT_CONTINUABLE", "ORIGINAL_DECISION_NOT_EXECUTED"} <= set(
        other_option.reasons
    )
    stricter_gateway = FeatureInputGateway(
        replace(gateway.policy, policy_hash="", provider_retry_delay_seconds=600)
    )
    other_policy = stricter_gateway.continue_quarantine(
        quarantine=quarantine,
        original_case=case,
        original_option_id="recoverable_quarantine",
        original_execution_policy_hash=gateway.policy.policy_hash,
        original_execution=execution,
        current=_anomaly_evidence(closes=DAY_TWO),
        observed_at=day_two_at,
    )
    assert isinstance(other_policy, QuarantineContinuationRefusal)
    assert "POLICY_CHANGED" in other_policy.reasons
    refused_execution = replace(execution, failure_reasons=("PANEL_SECTOR_BELOW_MINIMUM",))
    not_executed = gateway.continue_quarantine(
        quarantine=quarantine,
        original_case=case,
        original_option_id="recoverable_quarantine",
        original_execution_policy_hash=gateway.policy.policy_hash,
        original_execution=refused_execution,
        current=_anomaly_evidence(closes=DAY_TWO),
        observed_at=day_two_at,
    )
    assert isinstance(not_executed, QuarantineContinuationRefusal)
    assert "ORIGINAL_DECISION_NOT_EXECUTED" in not_executed.reasons
    legacy_evidence = replace(day_one, unexplained_moves=(), anomaly_signature_hash=None)
    legacy_case = replace(case, evidence=(legacy_evidence,))
    legacy = judge(_anomaly_evidence(closes=DAY_TWO), original_case=legacy_case)
    assert isinstance(legacy, QuarantineContinuationRefusal)
    assert legacy.reasons == ("ORIGINAL_ANOMALY_SIGNATURE_ABSENT",)


def test_persisted_continuation_reads_back_only_with_its_identity(tmp_path) -> None:
    """regression: the persisted readback refuses a record without or against its identity.

    The store keeps the continued row and its record; the data-issues
    readback chains them. A stored record whose identity was removed or
    emptied, one re-sealed over modified evidence, and a stored row that no
    longer matches its record are each refused by name -- the reader never
    signs what it reads. A quarantine row without a record (every row
    written before continuation existed) still reads back.
    """

    from alphalattice.control.product_host.composition.application_session import (
        WorkspaceApplicationSession,
    )
    from alphalattice.control.product_host.data_preparation.remediation import (
        WorkspaceDataIssueApplication,
    )

    manifest = replace(
        _manifest(20),
        profile=replace(_manifest(20).profile, market_profile_id="us-current-index-research"),
    )
    gateway = FeatureInputGateway()
    day_one = _anomaly_evidence(closes=DAY_ONE)
    case, execution, quarantine = _quarantine_decision(
        gateway, day_one, option_id="recoverable_quarantine"
    )
    continued = gateway.continue_quarantine(
        quarantine=quarantine,
        original_case=case,
        original_option_id="recoverable_quarantine",
        original_execution_policy_hash=gateway.policy.policy_hash,
        original_execution=execution,
        current=_anomaly_evidence(closes=DAY_TWO),
        observed_at=NOW + timedelta(days=1),
    )
    assert isinstance(continued, QuarantineContinuation)
    with WorkspaceApplicationSession.acquire(tmp_path / "workspace") as session:
        market = MarketDataRepository(session.workspace)
        market.bootstrap(manifest)
        panel = PanelStateRepository(market.database, market_data=market)
        market.readiness.save(
            market_profile_id=manifest.profile.market_profile_id,
            status="FEATURE_BUILDING",
            active_manifest_id=manifest.manifest_id,
            active_manifest_revision=manifest.revision_sha256,
            active_membership_fingerprint=None,
            active_candidate_manifest_document=None,
            pending_membership_fingerprint=None,
            pending_candidate_manifest_document=None,
            last_checked_at=NOW,
            last_changed_at=NOW,
            failure_code=None,
            observed_at=NOW,
        )
        panel.record_listing_quarantines(
            candidate_manifest_revision=manifest.revision_sha256,
            quarantines=(quarantine,),
            observed_at=NOW,
            effective_session=AS_OF,
        )
        application = WorkspaceDataIssueApplication(session, lambda: NOW + timedelta(days=1))
        # A row without a record reads back as before continuation existed.
        assert application.readback()["continued_dispositions"] == []
        assert panel.active_listing_quarantines(manifest.revision_sha256) == (quarantine,)
        panel.record_quarantine_continuation(
            candidate_manifest_revision=manifest.revision_sha256,
            continuation=continued,
            observed_at=NOW + timedelta(days=1),
            effective_session=AS_OF + timedelta(days=1),
        )
        (chain,) = application.readback()["continued_dispositions"]
        assert chain["continuations"][0]["continuation_hash"] == continued.continuation_hash
        stored = json.dumps(continued.document(), sort_keys=True, separators=(",", ":"))

        def store(payload: dict[str, object] | None) -> None:
            with market._connect() as connection:
                connection.execute(
                    "UPDATE listing_quarantine SET continuation_json = ? WHERE quarantine_hash = ?",
                    [
                        json.dumps(payload, sort_keys=True, separators=(",", ":"))
                        if payload is not None
                        else stored,
                        continued.quarantine.quarantine_hash,
                    ],
                )

        forged = json.loads(stored)
        forged["evidence"]["evidence_hash"] = "0" * 64
        for value in (None, ""):  # absent, empty
            unsealed = dict(forged)
            if value is None:
                del unsealed["continuation_hash"]
            else:
                unsealed["continuation_hash"] = value
            store(unsealed)
            with pytest.raises(ValueError, match="continuation_identity_absent"):
                application.readback()
        resealed = dict(forged)
        resealed["continuation_hash"] = canonical_hash(
            {key: value for key, value in resealed.items() if key != "continuation_hash"}
        )
        store(resealed)
        with pytest.raises(ValueError, match="continuation_identity_mismatch"):
            application.readback()
        # The stored row is the authority: a record that no longer describes
        # it is refused even when the record is consistent in itself.
        store(None)
        with market._connect() as connection:
            connection.execute(
                "UPDATE listing_quarantine SET evidence_hash = ? WHERE quarantine_hash = ?",
                ["1" * 64, continued.quarantine.quarantine_hash],
            )
        with pytest.raises(ValueError, match="continuation_identity_mismatch"):
            application.readback()
        with market._connect() as connection:
            connection.execute(
                "UPDATE listing_quarantine SET evidence_hash = ? WHERE quarantine_hash = ?",
                [continued.quarantine.evidence_hash, continued.quarantine.quarantine_hash],
            )
        assert (
            application.readback()["continued_dispositions"][0]["continuations"][0][
                "quarantine_hash"
            ]
            == continued.quarantine.quarantine_hash
        )


def test_partial_sector_evidence_governs_failures_only_and_invents_nothing() -> None:
    """requirement: a manifest without complete Sector evidence is not admitted through it.

    Listings declared without Sector evidence take no part in the sector
    judgement, no case offers an exclusion whose Panel impact cannot be
    projected, and with nothing to govern the Gateway refuses to admit
    rather than admit blind.
    """
    manifest = _manifest(20)
    known = _sectors(manifest, size=5)
    unknown = frozenset(listing.listing_id for listing in manifest.listings[15:])
    partial = {key: value for key, value in known.items() if key not in unknown}
    gateway = FeatureInputGateway()
    failing = gateway.assess(
        candidate_manifest=manifest,
        evidence=_evidence(manifest, failed={0, 16}),
        sector_by_listing_id=partial,
        temporal_boundary=_temporal(),
        observed_at=NOW,
        sector_unknown_listing_ids=unknown,
    )
    assert failing.assessment is FeatureInputAssessment.DEFERRED
    assert failing.admission is None and failing.research_manifest is None
    assert [case.listing_ids for case in failing.agent_cases] == [
        ("listing-000",),
        ("listing-016",),
    ]
    for case in failing.agent_cases:
        offered = {option.option_id for option in case.options}
        assert {"bounded_full_history_retry", "wait_for_provider_recovery"} <= offered
        assert not offered & {"recoverable_quarantine", "exclude_from_next_manifest"}
    # Four of the five unknown-sector listings fail: a sector-macro cohort by
    # count, but no sector is known for them, so no cohort is invented.
    clustered = gateway.assess(
        candidate_manifest=manifest,
        evidence=_evidence(manifest, failed={15, 16, 17, 18}),
        sector_by_listing_id=partial,
        temporal_boundary=_temporal(),
        observed_at=NOW,
        sector_unknown_listing_ids=unknown,
    )
    assert clustered.deferred is None and len(clustered.agent_cases) == 4
    clean = gateway.assess(
        candidate_manifest=manifest,
        evidence=_evidence(manifest),
        sector_by_listing_id=partial,
        temporal_boundary=_temporal(),
        observed_at=NOW,
        sector_unknown_listing_ids=unknown,
    )
    assert clean.assessment is FeatureInputAssessment.DEFERRED
    assert clean.failure_reasons == ("SECTOR_EVIDENCE_INCOMPLETE",)
    assert clean.admission is None and not clean.agent_cases
    with pytest.raises(ValueError, match="panel impact cannot be projected"):
        gateway.assess(
            candidate_manifest=manifest,
            evidence=_evidence(manifest),
            sector_by_listing_id=partial,
            temporal_boundary=_temporal(),
            observed_at=NOW,
            panel_impact=_panel_impact(manifest, known),
            sector_unknown_listing_ids=unknown,
        )
    with pytest.raises(ValueError, match="sector evidence must cover"):
        gateway.assess(
            candidate_manifest=manifest,
            evidence=_evidence(manifest),
            sector_by_listing_id=known,
            temporal_boundary=_temporal(),
            observed_at=NOW,
            sector_unknown_listing_ids=unknown,
        )


def test_host_executes_agent_option_without_accepting_agent_parameters(tmp_path) -> None:
    manifest = _manifest(24)
    evidence = _evidence(manifest, failed={0})
    gateway = FeatureInputGateway()
    result = gateway.assess(
        candidate_manifest=manifest,
        evidence=evidence,
        sector_by_listing_id=_sectors(manifest, size=6),
        temporal_boundary=_temporal(),
        observed_at=NOW,
    )
    case = result.agent_cases[0]
    option = next(item for item in case.options if item.option_id == "recoverable_quarantine")
    execution = FeatureInputRemediationExecutor(gateway).execute(
        case=case,
        option=option,
        evidence_by_listing_id={item.listing_id: item for item in evidence},
        observed_at=NOW,
        panel_impact_after_exclusion=PanelImpactProjection(
            {"sector-00": 5, "sector-01": 6, "sector-02": 6, "sector-03": 6},
            1.0,
            60,
            True,
        ),
        proposal_hash="f" * 64,
    )
    assert execution.status is FeatureInputExecutionStatus.QUARANTINE_READY
    assert execution.quarantines[0].listing_id == manifest.listings[0].listing_id
    market_data = MarketDataRepository(tmp_path / "policy-store")
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(manifest)
    service = FeatureInputGovernanceService(
        market_data,
        panel_state,
        WorkspaceMutationGate(),
        gateway,
    )
    assert service.record_policy_execution(
        candidate_manifest_revision=manifest.revision_sha256,
        execution=execution,
        observed_at=NOW,
    )
    assert not service.record_policy_execution(
        candidate_manifest_revision=manifest.revision_sha256,
        execution=execution,
        observed_at=NOW,
    )
    assert len(panel_state.active_listing_quarantines(manifest.revision_sha256)) == 1


def test_unique_verified_alias_and_last_known_good_are_host_bound() -> None:
    manifest = _manifest(100)
    gateway = FeatureInputGateway()
    alias_failure = replace(
        _evidence(manifest, failed={0})[0],
        verified_alias_candidate_id="fixture-alias-a",
        verified_alias_provider_symbol="S000.A",
        verified_alias_evidence_hash="a" * 64,
    )
    alias_case = gateway.assess(
        candidate_manifest=manifest,
        evidence=tuple(
            alias_failure if index == 0 else item for index, item in enumerate(_evidence(manifest))
        ),
        sector_by_listing_id=_sectors(manifest),
        temporal_boundary=_temporal(),
        observed_at=NOW,
    ).agent_cases[0]
    alias_option = next(
        item
        for item in alias_case.options
        if item.option_id == "use_unique_verified_alias_for_this_run"
    )
    assert alias_option.policy_args.provider_symbol == "S000.A"  # type: ignore[union-attr]

    marketwide = gateway.assess(
        candidate_manifest=manifest,
        evidence=_evidence(manifest, failed=set(range(25))),
        sector_by_listing_id=_sectors(manifest),
        temporal_boundary=_temporal(),
        observed_at=NOW,
        macro_retry_exhausted=True,
        last_known_good_snapshot_ref=f"playpen://feature-panel/manifests/{'b' * 64}",
    ).agent_cases[0]
    old_snapshot = next(
        item
        for item in marketwide.options
        if item.option_id == "retain_last_known_good_frozen_snapshot"
    )
    execution = FeatureInputRemediationExecutor(gateway).execute(
        case=marketwide,
        option=old_snapshot,
        evidence_by_listing_id={
            item.listing_id: item
            for item in _evidence(manifest, failed=set(range(25)))
            if item.listing_id in marketwide.listing_ids
        },
        observed_at=NOW,
    )
    assert execution.status is FeatureInputExecutionStatus.LAST_KNOWN_GOOD_BOUND
    assert execution.bound_market_as_of_session == AS_OF


def _publish_reader_fixture(root: Path) -> tuple[ArtifactResolver, str]:
    resolver = ArtifactResolver(root)
    reader_temporal = TemporalKnowledgeBoundary(
        market_as_of_session=date(2026, 1, 2),
        knowledge_cutoff_at=NOW,
        materialized_at=NOW + timedelta(minutes=1),
        universe_source_observed_at=NOW - timedelta(days=1),
        sector_source_observed_at=NOW - timedelta(hours=1),
    )
    candidate_manifest = _manifest(2)
    admission_result = FeatureInputGateway().assess(
        candidate_manifest=candidate_manifest,
        evidence=_evidence(candidate_manifest),
        sector_by_listing_id=_sectors(candidate_manifest, size=2),
        temporal_boundary=reader_temporal,
        observed_at=NOW,
    )
    assert admission_result.admission is not None
    from alphalattice.foundation.feature_engine.panels.identity import hash_panel_rows

    rows = [
        {
            "session_date": date(2025 + index // 2, 1, 2),
            "listing_id": f"listing-{index % 2:03d}",
            "materialization_receipt_hash": "4" * 64,
            "factor_a": float(index),
        }
        for index in range(4)
    ]
    chunks: list[dict[str, object]] = []
    schema_hash: str | None = None
    for year in (2025, 2026):
        # Written as a Panel writer writes a partition: its row identity's columns and the row
        # hashes its values give, which the reader checks (V270).
        table = hash_panel_rows(
            pa.Table.from_pylist([row for row in rows if row["session_date"].year == year]),
            manifest_revision="1" * 64,
            sector_revision="2" * 64,
            catalog_hash="e" * 64,
            policy_hash="3" * 64,
            factor_ids=("factor_a",),
        )
        current_schema_hash = (
            __import__("hashlib")
            .sha256(str(table.schema.remove_metadata()).encode("utf-8"))
            .hexdigest()
        )
        schema_hash = schema_hash or current_schema_hash
        chunk_hash = canonical_hash(
            {
                "panel_binding_hash": "b" * 64,
                "year": year,
                "row_count": table.num_rows,
                "row_hashes": table.column("row_hash").to_pylist(),
                "schema_hash": current_schema_hash,
            }
        )
        artifact = resolver.publish_feature_panel_chunk(
            table=table,
            content_hash=chunk_hash,
            metadata={
                "panel_binding_hash": "b" * 64,
                "calendar_year": str(year),
                "schema_hash": current_schema_hash,
            },
        )
        sessions = table.column("session_date").to_pylist()
        chunks.append(
            {
                "year": year,
                "first_session": min(sessions).isoformat(),
                "last_session": max(sessions).isoformat(),
                "row_count": table.num_rows,
                "chunk_hash": chunk_hash,
                "metadata_hash": artifact.metadata_hash,
                "uri": artifact.uri,
            }
        )
    temporal = reader_temporal.identity_payload()
    temporal["temporal_identity_hash"] = reader_temporal.identity_hash()
    payload = {
        "kind": "FeaturePanelSnapshotManifest",
        "panel_binding_hash": "b" * 64,
        "panel_content_hash": "c" * 64,
        "history_start": "2025-01-02",
        "as_of_session": "2026-01-02",
        "knowledge_cutoff_at": NOW.isoformat(),
        "temporal_identity_hash": temporal["temporal_identity_hash"],
        "active_listing_count": 2,
        "listing_set_hash": "d" * 64,
        "schema_hash": schema_hash,
        "chunks": chunks,
        "safe_summary": {
            "factor_catalog_summary": {"factor_a": {"available_session_count": 2}},
            "lineage": {
                "catalog_hash": "e" * 64,
                "panel_content_hash": "c" * 64,
            },
            "temporal_risk": {
                "universe_temporal_scope": "current-active-survivors",
                "universe_point_in_time_qualified": False,
                "sector_source": "fixture-current-sector",
                "sector_observed_at": NOW.isoformat(),
                "sector_point_in_time_qualified": False,
                "sector_history_treatment": "current-sector-backfilled-through-history",
                "research_use_class": "CURRENT_UNIVERSE_RESEARCH_ONLY",
            },
            "temporal_boundary": temporal,
            "quality_governance": {
                "gateway_qualified": True,
                "quality_admission_hash": admission_result.admission.admission_hash,
                "candidate_listing_count": 2,
                "admitted_listing_count": 2,
                "quality_exclusion_count": 0,
                "quarantine_reason_counts": {},
            },
            "universe_policy": {
                "type": "CURRENT_SURVIVOR_COMPOSITE",
                "components": ["SP500", "NASDAQ100", "DJIA"],
                "survivorship_bias_warning": True,
                "research_use_class": "CURRENT_UNIVERSE_RESEARCH_ONLY",
            },
        },
    }
    snapshot_hash = canonical_hash(payload)
    artifact = resolver.publish_feature_panel_manifest(
        payload={**payload, "snapshot_hash": snapshot_hash}, snapshot_hash=snapshot_hash
    )
    resolver.publish_feature_panel_lifecycle_projection(
        snapshots=(
            {
                "snapshot_hash": snapshot_hash,
                "lifecycle": "ACTIVE",
                "reason": None,
            },
        )
    )
    return resolver, artifact.uri


def test_feature_panel_reader_streams_projected_batches_and_supports_concurrent_reads(
    tmp_path,
) -> None:
    resolver, manifest_ref = _publish_reader_fixture(tmp_path / "artifacts")
    request = FeaturePanelReadRequest(
        manifest_ref=manifest_ref,
        start_session=date(2025, 1, 2),
        end_session=date(2026, 1, 2),
        factor_columns=("factor_a",),
        batch_size=1,
    )

    def read() -> tuple[int, tuple[str, ...]]:
        batches = list(FeaturePanelReader(resolver).batches(request))
        return sum(batch.num_rows for batch in batches), tuple(batches[0].schema.names)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: read(), range(4)))
    assert results == [(4, ("session_date", "listing_id", "factor_a"))] * 4
    forged = request.__class__(
        manifest_ref=f"playpen://feature-panel/manifests/{'0' * 64}",
        start_session=request.start_session,
        end_session=request.end_session,
        factor_columns=request.factor_columns,
    )
    with pytest.raises(ValueError, match="no lifecycle admission"):
        list(FeaturePanelReader(resolver).batches(forged))
    retired = request.__class__(
        manifest_ref=manifest_ref,
        start_session=request.start_session,
        end_session=request.end_session,
        factor_columns=("retired_factor",),
    )
    with pytest.raises(ValueError, match="outside the admitted catalog"):
        list(FeaturePanelReader(resolver).batches(retired))
    snapshot_hash = manifest_ref.rsplit("/", 1)[-1]
    resolver.publish_feature_panel_lifecycle_projection(
        snapshots=(
            {
                "snapshot_hash": snapshot_hash,
                "lifecycle": "QUARANTINED",
                "reason": "degenerate residual_mom_252_21 column",
            },
        )
    )
    with pytest.raises(ValueError, match="requires an ACTIVE Feature Panel snapshot"):
        list(FeaturePanelReader(resolver).batches(request))
    reader_tree = ast.parse(inspect.getsource(FeaturePanelReader))
    assert not any(
        isinstance(node, ast.Name) and node.id == "duckdb" for node in ast.walk(reader_tree)
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "to_table"
        for node in ast.walk(reader_tree)
    )


def test_data_remediation_terminal_failure_receipt_is_durable_and_idempotent(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "task-runtime.duckdb"
    registry = DuckDbWorkspaceMaintenanceRegistry(
        registry_path,
        gate=WorkspaceMutationGate(),
    )
    values = {
        "maintenance_id": "maintenance-001",
        "case_token": "case-token-001",
        "evidence_hash": "e" * 64,
        "failure_code": "cognition.budget_exhausted",
        "retryable": True,
        "execution_attempt_count": 2,
        "prior_failure_codes": (
            "cognition.budget_exhausted",
            "cognition.budget_exhausted",
        ),
        "agent_execution_hashes": ("a" * 64, "b" * 64),
        "model_context_audit_hashes": ("c" * 64, "d" * 64),
        "observed_at": NOW,
    }
    receipt = DataRemediationFailureReceipt(
        **values,
        receipt_hash=canonical_hash(values),
    )

    assert registry.record_data_remediation_failure(receipt) is True
    assert registry.record_data_remediation_failure(receipt) is False

    with duckdb.connect(str(registry_path), read_only=True) as connection:
        rows = connection.execute(
            "SELECT failure_code, receipt_json "
            "FROM data_remediation_failure_receipt WHERE maintenance_id = ?",
            [receipt.maintenance_id],
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == receipt.failure_code
    document = json.loads(rows[0][1])
    assert document["execution_attempt_count"] == 2
    assert document["prior_failure_codes"] == list(receipt.prior_failure_codes)


def test_admission_only_failure_is_bound_to_the_existing_workspace_cycle(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from alphalattice.control.data_platform.maintenance.contracts import (
        MaintenanceTrigger,
        WorkspaceMaintenanceRequest,
    )
    from alphalattice.control.data_platform.maintenance.coordinator import (
        WorkspaceMaintenanceCoordinator,
    )
    from alphalattice.control.data_platform.maintenance.registry import (
        DuckDbWorkspaceMaintenanceRegistry,
    )

    manifest = _manifest(20)
    market = MarketDataRepository(tmp_path)
    market.bootstrap(manifest)
    gate = WorkspaceMutationGate()
    registry = DuckDbWorkspaceMaintenanceRegistry(market.path, gate=gate)
    panel = PanelStateRepository(market.database, market_data=market)
    service = FeatureInputGovernanceService(market, panel, gate, FeatureInputGateway())
    evidence = _evidence(manifest, failed={0})
    sectors = _sectors(manifest)
    result = service.assess_and_record(
        candidate_manifest=manifest,
        evidence=evidence,
        sector_by_listing_id=sectors,
        temporal_boundary=_temporal(),
        observed_at=NOW,
    )
    request = WorkspaceMaintenanceRequest.create(
        market_profile_id=manifest.profile.market_profile_id,
        target_market_session=AS_OF,
        knowledge_cutoff_at=NOW,
        trigger=MaintenanceTrigger.USER_REQUEST,
        membership_revision=manifest.revision_sha256,
        data_policy_hash="d" * 64,
        feature_policy_hash="e" * 64,
    )
    cycle = registry.admit(request, observed_at=NOW)
    coordinator = object.__new__(WorkspaceMaintenanceCoordinator)
    coordinator.feature_input, coordinator.panel_state = service, panel
    coordinator.registry, coordinator.diagnose = registry, None

    def damaged(_):
        raise ValueError("feature_input.resolution_tampered")

    monkeypatch.setattr(panel, "feature_input_resolution", damaged)
    outcome = coordinator._handle_governance_result(
        SimpleNamespace(
            manifest=manifest,
            result=result,
            evidence=evidence,
            panel_impact=_panel_impact(manifest, sectors),
            sector_source=None,
        ),
        maintenance_id=None,
        request=request,
        observed_at=NOW,
        observed_workers=1,
    )
    # The reason the remediation stopped is the code the owner raised, not
    # the generic class of its exception; the page reads it back by name.
    assert outcome[2] == "feature_input.resolution_tampered"
    with duckdb.connect(str(market.path), read_only=True) as connection:
        row = connection.execute(
            "SELECT maintenance_id, receipt_json FROM data_remediation_failure_receipt"
        ).fetchone()
    assert row[0] == "workspace-cycle:" + cycle.cycle_id
    document = json.loads(row[1])
    assert document["execution_attempt_count"] == 0
    assert document["failure_code"] == "feature_input.resolution_tampered"
    assert document["prior_failure_codes"] == ["VALUEERROR"]
    assert document["explanation"] is None
    recorded = DataRemediationFailureReceipt.read_document(document)
    assert registry.latest_data_remediation_failure() == recorded


def test_remediation_failure_reason_is_stable_and_never_raw_text() -> None:
    """A stable code and bounded safe text per exception kind; nothing raw."""

    from alphalattice.control.data_platform.maintenance.contracts import (
        remediation_failure_reason,
    )
    from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError

    assert remediation_failure_reason(ValueError("feature_input.resolution_policy_changed")) == (
        "feature_input.resolution_policy_changed",
        None,
    )
    assert remediation_failure_reason(
        ValueError("feature_input.option_requires_data_owner_action:quarantine_listing")
    ) == ("feature_input.option_requires_data_owner_action:quarantine_listing", None)
    assert remediation_failure_reason(
        WorkspaceConflictError("replacement of x is blocked", code="catalog.replace_blocked")
    ) == ("catalog.replace_blocked", "replacement of x is blocked")
    for raw in (
        ValueError("Provider returned {'close': 12.3} for AAPL; retry"),
        KeyError("secret"),
        RuntimeError("model said: quarantine everything"),
    ):
        assert remediation_failure_reason(raw) == (
            "DATA_REMEDIATION_BUSINESS_VALIDATION_FAILED",
            None,
        )


def test_failure_receipt_without_explanation_keeps_its_identity_and_reads_back(tmp_path) -> None:
    """Receipts sealed before ``explanation`` existed verify and read back unchanged."""

    values = {
        "maintenance_id": "m" * 64,
        "case_token": "c" * 64,
        "evidence_hash": "e" * 64,
        "failure_code": "cognition.budget_exhausted",
        "retryable": True,
        "execution_attempt_count": 1,
        "prior_failure_codes": ("cognition.budget_exhausted",),
        "agent_execution_hashes": (),
        "model_context_audit_hashes": (),
        "observed_at": NOW,
    }
    legacy = DataRemediationFailureReceipt(**values, receipt_hash=canonical_hash(values))
    assert legacy.explanation is None
    document = json.loads(json.dumps(asdict(legacy), default=str))
    document.pop("explanation")  # as a row written before the field existed
    assert DataRemediationFailureReceipt.read_document(document) == legacy
    explained = DataRemediationFailureReceipt.seal(
        **{**values, "explanation": "replacement of pending.json is blocked by an open handle"}
    )
    assert explained.receipt_hash != legacy.receipt_hash
    assert (
        DataRemediationFailureReceipt.read_document(
            json.loads(json.dumps(asdict(explained), default=str))
        )
        == explained
    )


@pytest.mark.parametrize(
    ("tamper", "reason"),
    [
        (lambda d: d.__setitem__("agent_execution_hashes", "a" * 64), "sequence became text"),
        (lambda d: d.pop("model_context_audit_hashes"), "required sequence removed"),
        (lambda d: d.__setitem__("prior_failure_codes", ["ok", 7]), "non-string element"),
        (lambda d: d.__setitem__("prior_failure_codes", {"a": 1}), "sequence became a mapping"),
        (lambda d: d.pop("retryable"), "required field removed"),
        (lambda d: d.__setitem__("observed_at", 12), "clock is not text"),
    ],
)
def test_failure_receipt_reader_refuses_a_document_that_lost_its_shape(tamper, reason) -> None:
    """regression (supervisor): a malformed sequence field read back as empty and verified.

    ``read_document`` turned a missing or malformed sequence into ``()``
    before the identity check, so a document tampered from ``[]`` to a
    string, or stripped of a required list, still matched its original
    hash. The reader now requires every sealed field with its recorded
    shape and refuses anything else instead of repairing it.
    """

    values = {
        "maintenance_id": "m" * 64,
        "case_token": None,
        "evidence_hash": None,
        "failure_code": "DATA_REMEDIATION_BUSINESS_VALIDATION_FAILED",
        "retryable": True,
        "execution_attempt_count": 0,
        "prior_failure_codes": ("VALUEERROR",),
        "agent_execution_hashes": (),
        "model_context_audit_hashes": (),
        "observed_at": NOW,
    }
    sealed = DataRemediationFailureReceipt.seal(**values)
    document = json.loads(json.dumps(asdict(sealed), default=str))
    assert DataRemediationFailureReceipt.read_document(document) == sealed
    tamper(document)
    with pytest.raises(ValueError, match="Data remediation failure receipt"):
        DataRemediationFailureReceipt.read_document(document)


def test_materialized_qualification_is_scoped_and_requires_real_unavailability_evidence() -> None:
    from dataclasses import replace

    from alphalattice.foundation.feature_engine.inputs.quality import qualify_materialized_features
    from alphalattice.foundation.feature_engine.storage.contracts import FeatureIneligibilityRun

    session = NOW.date()
    catalog = "a" * 64
    rows = [
        {
            "listing_id": "A",
            "session_date": session,
            "catalog_hash": catalog,
            "row_hash": "b" * 64,
            "price": 1.0,
            "volume_formula": 0.0,
        },
        {
            "listing_id": "B",
            "session_date": session,
            "catalog_hash": catalog,
            "row_hash": "c" * 64,
            "price": 2.0,
            "volume_formula": None,
        },
    ]
    reason = FeatureIneligibilityRun(
        run_id="fixture-run",
        listing_id="B",
        catalog_hash=catalog,
        factor_id="volume_formula",
        reason="zero_denominator",
        first_session=session,
        last_session=session,
        first_observation_count=21,
        observation_cap=21,
        materialization_receipt_hash="d" * 64,
        updated_at=NOW,
    )
    inputs = dict(session=session, catalog_hash=catalog, listing_ids=("A", "B", "C"), rows=rows)
    prices = qualify_materialized_features(**inputs, factor_ids=("price",), ineligibility=())
    assert prices.eligible_listing_ids == ("A", "B")
    assert prices.pending_listing_ids == ("C",) and not prices.exclusions
    mixed = qualify_materialized_features(
        **inputs,
        factor_ids=("price", "volume_formula"),
        ineligibility=(reason,),
    )
    assert mixed.eligible_listing_ids == ("A",)  # A finite zero is not an undefined formula.
    assert mixed.pending_listing_ids == ("C",)
    assert mixed.exclusions == (("B", (("volume_formula", "zero_denominator"),)),)
    assert mixed.summary()["nominal_count"] == 3 and mixed.evidence_hash != prices.evidence_hash
    with pytest.raises(ValueError, match="qualification_unavailability_undocumented"):
        qualify_materialized_features(
            **inputs,
            factor_ids=("volume_formula",),
            ineligibility=(replace(reason, catalog_hash="f" * 64),),
        )
