from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from uuid import UUID

import pytest

from alphalattice.foundation.feature_engine.contracts import (
    canonical_hash,
)
from alphalattice.foundation.feature_engine.inputs.gateway import (
    FeatureInputAssessment,
    FeatureInputGateway,
    FeatureInputGovernanceService,
    FeatureInputPolicy,
    FeatureInputRemediationExecutor,
    matching_raw_retention_proof,
)
from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.foundation.market_data_ops.sources.manifest import (
    build_quality_filtered_research_manifest,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    MarketDataRepository,
)
from tests.feature_input_gateway.gateway_support import (
    NOW,
    _evidence,
    _manifest,
    _panel_impact,
    _sectors,
    _temporal,
    mark_building,
    rule_retention_confirmations,
)


@pytest.mark.parametrize("actor", ["human", "rule"])
def test_raw_retention_choice_continues_only_for_the_same_case_on_a_derived_manifest(
    tmp_path,
    actor,
    monkeypatch,
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
        unexplained_sessions=(NOW.date(),) if actor == "rule" else (),
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
        mark_building(market, manifest, NOW)
        application = WorkspaceDataIssueApplication(session, lambda: NOW)
        authority, confirm = None, application.confirm
        if actor == "rule":
            goals, confirm = rule_retention_confirmations(
                session, application, manifest, evidence, old_cases
            )
            authority = goals.recorded_data_confirmation
            application = WorkspaceDataIssueApplication(
                session, lambda: NOW, recorded_data_confirmation=authority
            )
        original_resolutions = {}
        for case in old_cases:
            option = next(
                value
                for value in case.options
                if value.option_id == "retain_isolated_raw_move_with_caveat"
            )
            confirmation = confirm(
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
            unexplained_sessions=(NOW.date(),) if actor == "rule" else (),
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
        mark_building(market, child, NOW + timedelta(minutes=1))
        proofs = raw_retention_decision_proofs(panel, market, historical_authority=authority)
        assert len(proofs) == 6
        if actor == "rule":
            head = goals.store.head(UUID(int=7))
            entries = goals.store.attributed(head.goal_id)
            for fault in ("missing", "foreign", "late", "rule_hash"):
                with monkeypatch.context() as invalid:
                    if fault == "foreign":
                        invalid.setattr(
                            goals.store,
                            "head",
                            lambda _: head.model_copy(update={"goal_id": UUID(int=8)}),
                        )
                    if fault == "rule_hash":
                        invalid.setattr(
                            goals.store,
                            "attributed",
                            lambda _: tuple({**e, "rule_hash": "f" * 64} for e in entries),
                        )
                    if fault == "late":
                        invalid.setattr(
                            goals.store,
                            "attributed",
                            lambda _: tuple(
                                {**e, "recorded_at": goals.delegation_ends(head).isoformat()}
                                for e in entries
                            ),
                        )
                    rejected = raw_retention_decision_proofs(
                        panel,
                        market,
                        historical_authority=None if fault == "missing" else authority,
                    )
                    assert rejected == (), fault
                    assert all(
                        matching_raw_retention_proof(
                            case=c,
                            candidate_manifest=child,
                            proofs=rejected,
                            policy_hash=service.gateway.policy.policy_hash,
                        )
                        is None
                        for c in pending.agent_cases
                    )
            from alphalattice.control.data_platform.contracts import DataRemediationExecutionReceipt
            from alphalattice.control.data_platform.remediation_case import (
                seal_validated_data_remediation_execution,
            )
            from alphalattice.protocols.actor_execution import ActorKind

            original = DataRemediationExecutionReceipt.model_validate(
                next(iter(original_resolutions.values()))["receipt"]
            )
            wrong_actor = seal_validated_data_remediation_execution(
                **original.model_dump(exclude={"kind", "actor_submission", "receipt_hash"}),
                actor_kind=ActorKind.HOST_FALLBACK,
                actor_id="host-rule:foreign",
            )
            assert not authority(wrong_actor)
            assert all(
                matching_raw_retention_proof(
                    case=c,
                    candidate_manifest=child,
                    proofs=tuple(replace(p, rule_authority_verified=False) for p in proofs),
                    policy_hash=service.gateway.policy.policy_hash,
                )
                is None
                for c in pending.agent_cases
            )
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
        for rejected_case, policy in (
            (changed_case, service.gateway.policy.policy_hash),
            (aggregate_case, service.gateway.policy.policy_hash),
            (policy_case, alternate_gateway.policy.policy_hash),
        ):
            assert (
                matching_raw_retention_proof(
                    case=rejected_case, candidate_manifest=child, proofs=proofs, policy_hash=policy
                )
                is None
            )
        for candidate, retained in (
            (replace(child, membership_fingerprint="f" * 64), proofs),
            (child, (replace(first_proof, actor_kind="UNKNOWN"),)),
            (child, (replace(first_proof, option_hash="f" * 64),)),
            (
                child,
                (
                    first_proof,
                    replace(first_proof, receipt_hash=canonical_hash("different receipt")),
                ),
            ),
        ):
            assert (
                matching_raw_retention_proof(
                    case=first_case,
                    candidate_manifest=candidate,
                    proofs=retained,
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
        mark_building(market, second_child, NOW + timedelta(minutes=2))
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
        second_proofs = raw_retention_decision_proofs(panel, market, historical_authority=authority)
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
