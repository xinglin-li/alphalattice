"""Shared Feature-input fixtures: a manifest, its temporal boundary, evidence and sectors."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from uuid import UUID

from alphalattice.foundation.feature_engine.contracts import (
    TemporalKnowledgeBoundary,
    canonical_hash,
)
from alphalattice.foundation.feature_engine.inputs.gateway import (
    ListingQualityEvidence,
    PanelImpactProjection,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
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
    unexplained_sessions: tuple[date, ...] = (),
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
            unexplained_sessions=unexplained_sessions if index in failed else (),
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


def rule_retention_confirmations(session, application, manifest, evidence, cases):
    """Build an original delegated rule record and its exact sealed choices."""
    from alphalattice.control.data_platform.data_decision_rule import DataDecisionRule
    from alphalattice.control.data_platform.remediation_case import (
        seal_validated_data_remediation_execution,
    )
    from alphalattice.control.product_host.composition.goals import GoalApplication
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceManifest,
        publish_research_workspace_manifest,
    )
    from alphalattice.control.product_host.publication.goals import GoalStore
    from alphalattice.foundation.market_data_ops.runtime.remediation import (
        PolicyDecision,
    )
    from alphalattice.foundation.market_data_ops.runtime.remediation import (
        canonical_hash as remediation_hash,
    )
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    configured = ResearchWorkspaceManifest.research_only("retention").with_bindings(
        data_decision_rule=DataDecisionRule(max_population_share=0.2)
    )
    publish_research_workspace_manifest(session.workspace, configured)
    goals = GoalApplication(
        GoalStore(session.workspace / "artifacts", configured.workspace_id),
        lambda: NOW,
        lambda *_: {},
        lambda _: NOW,
        lambda _: None,
        workspace=session.workspace,
    )
    goal_id, task_id = UUID(int=7), UUID(int=9)
    goals.operate(
        PortfolioResearchOperationRequest(
            operation="GOAL_OPEN",
            goal_id=goal_id,
            goal_declaration={
                "title": "First use",
                "objective": "Keep verified raw findings.",
                "kind": "FIRST_USE",
                "criteria": [{"criterion_id": "data", "text": "Verified data."}],
            },
            change_reason="The original delegated data preparation.",
        ),
        "EXTERNAL_AUTOMATION",
    )
    goals.store.attribute(goal_id, {"task_id": str(task_id)})
    decide = goals.data_rule(task_id)
    assert decide is not None
    by_token = {case.case_token: case for case in cases}

    def confirm(**selection):
        case = by_token[selection["case_token"]]
        actor = decide(case, evidence, None, None)
        option = next(o for o in case.options if o.option_id == actor.submission.proposal.option_id)
        assert option.option_id == selection["option_id"]
        receipt = seal_validated_data_remediation_execution(
            submission=actor.submission,
            actor_kind=actor.actor_kind,
            actor_id=actor.actor_id,
            execution_policy_hash=case.policy_hash,
            policy_decision=PolicyDecision(
                case_token=case.case_token,
                evidence_hash=case.evidence_hash,
                option_id=option.option_id,
                option_hash=option.option_hash,
                policy_hash=remediation_hash(option.policy_args),
                disposition=option.disposition,
            ),
        )
        application.panel.record_feature_input_resolution(
            case.case_token, receipt.model_dump(mode="json")
        )
        return {"receipt_hash": receipt.receipt_hash}

    return goals, confirm


def mark_building(market, manifest, now):
    """Publish the same readiness binding while the authority fixture advances revisions."""
    market.readiness.save(
        market_profile_id=manifest.profile.market_profile_id,
        status="FEATURE_BUILDING",
        active_manifest_id=manifest.manifest_id,
        active_manifest_revision=manifest.revision_sha256,
        active_membership_fingerprint=manifest.membership_fingerprint,
        active_candidate_manifest_document=None,
        pending_membership_fingerprint=None,
        pending_candidate_manifest_document=None,
        last_checked_at=now,
        last_changed_at=now,
        failure_code=None,
        observed_at=now,
    )
