"""Declared data choices and submissions, independent of local grant mechanics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from typing import Any, Literal, cast

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from alphalattice.control.data_platform.contracts import (
    DataRemediationActorSubmission,
    DataRemediationExecutionSubmission,
)
from alphalattice.foundation.feature_engine.inputs.gateway import (
    FeatureInputAgentCase,
    ListingQualityEvidence,
)
from alphalattice.foundation.market_data_ops.runtime.remediation import canonical_hash
from alphalattice.foundation.market_data_ops.sources.contracts import DataRemediationProposal
from alphalattice.protocols.actor_execution.contracts import ActorKind


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class DataDecisionRule:
    """The Host's declared rule for a first use's own data issues, never its results.

    An isolated verified move (one listing, one unexplained session, identity verified, no
    provider correction, retries spent) is retained with its caveat; any other case is
    quarantined, recoverably. Past the cap the rule decides none: many unexplained moves at
    once may be a source problem, a judgment it leaves to the agent. A changed rule is a new
    `rule_hash`, which every receipt it signs carries.
    """

    max_cases: int = Field(default=10, ge=0, strict=True)
    max_population_share: float = Field(default=0.02, ge=0, le=1, allow_inf_nan=False, strict=True)
    version: Literal["data-decision-rule/1"] = "data-decision-rule/1"

    @property
    def rule_hash(self) -> str:
        """The rule's identity: its declared fields."""
        return str(canonical_hash(asdict(cast(Any, self))))

    def capped(self, cases: int, population: int, *, affected_listings: int) -> bool:
        """Count cases and distinct affected listings against the declared bounds."""
        return cases > self.max_cases or affected_listings > self.max_population_share * population

    def scope(
        self,
        cases: Sequence[FeatureInputAgentCase],
        population: int,
        records: Iterable[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """An escalated set keeps its original bound after partial choices or exclusion."""
        prior = next(
            (
                row
                for row in records
                if row.get("rule_hash") == self.rule_hash
                and row.get("escalation_reason") == "RULE_BOUND_EXCEEDED"
            ),
            None,
        )
        counts = (
            {
                key: prior[key]
                for key in ("case_count", "affected_listing_count", "population_count")
            }
            if prior
            else {
                "case_count": len(cases),
                "affected_listing_count": len(
                    {listing for case in cases for listing in case.listing_ids}
                ),
                "population_count": population,
            }
        )
        return {
            **counts,
            "case_tokens": frozenset(case.case_token for case in cases),
            "capped": bool(prior)
            or self.capped(
                counts["case_count"],
                counts["population_count"],
                affected_listings=counts["affected_listing_count"],
            ),
        }

    def decide(
        self,
        case: FeatureInputAgentCase,
        evidence: tuple[ListingQualityEvidence, ...],
        scope: Mapping[str, Any],
    ) -> str:
        """Only a current case inside the original declared bound may be decided by rule."""
        return (
            ESCALATE
            if scope["capped"] or case.case_token not in scope["case_tokens"]
            else self.choose(case, evidence)
        )

    def choose(
        self, case: FeatureInputAgentCase, evidence: tuple[ListingQualityEvidence, ...]
    ) -> str:
        """The offered option this rule takes for `case`, over every listing's evidence."""
        own = [item for item in evidence if item.listing_id in case.listing_ids]
        isolated = (
            case.case_kind == "LISTING"
            and case.failure_code == "data.unexplained_raw_move"
            and len(own) == 1
            and all(
                item.reason_codes == ("UNEXPLAINED_RAW_MOVE",)
                and item.identity_verified
                and not item.provider_correction_observed
                and item.retry_exhausted
                and len(item.unexplained_sessions) == 1
                for item in own
            )
        )
        offered = {option.option_id for option in case.options}
        preferred = (RETAIN,) * isolated + (QUARANTINE, ESCALATE)
        return next((option for option in preferred if option in offered), ESCALATE)

    def submission(
        self, case: FeatureInputAgentCase, option_id: str, goal_id: str
    ) -> DataRemediationActorSubmission:
        """The Host's signed choice: HOST_FALLBACK, the goal's delegation and this rule."""
        proposal = DataRemediationProposal(
            run_id=case.run_id,
            case_token=case.case_token,
            evidence_hash=case.evidence_hash,
            option_id=option_id,
            rationale=f"first-use-goal:{goal_id}; rule:{self.rule_hash}",
        )
        payload = {
            "kind": "DataRemediationExecutionSubmission",
            "proposal": proposal.model_dump(mode="json"),
        }
        return DataRemediationActorSubmission(
            submission=DataRemediationExecutionSubmission(
                **payload, submission_hash=canonical_hash(payload)
            ),
            actor_kind=ActorKind.HOST_FALLBACK,
            actor_id=f"host-rule:{self.rule_hash[:16]}",
        )


RETAIN = "retain_isolated_raw_move_with_caveat"
QUARANTINE = "recoverable_quarantine"
ESCALATE = "escalate_data_truth_conflict"
