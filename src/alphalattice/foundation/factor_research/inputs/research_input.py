"""Host validation and publication contract for the 1D Factor Research input."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.factor_research.evaluation.oos_evidence import (
    FactorEvidenceClassification,
    FactorOosEvidenceReport,
)
from alphalattice.foundation.factor_research.evaluation.redundancy import FactorRedundancyStructure
from alphalattice.foundation.factor_research.programs.sealed import seal_contract
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class FactorResearchInputBoundaryError(ValueError):
    """Stable failure raised when an Agent proposal exceeds evidence authority."""


class FactorResearchCandidateRole(StrEnum):
    """Declare the evidence-qualified role proposed for a Factor candidate.

    Attributes:
        CORE: Choice requiring positive out-of-sample evidence.
        CONDITIONAL: Choice requiring mixed out-of-sample evidence and a rationale.
    """

    CORE = "CORE"
    CONDITIONAL = "CONDITIONAL"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class FactorResearchProposalChoice(_Contract):
    """Reference the verified evidence and redundancy cluster behind one candidate choice.

    Attributes:
        factor_id: Factor selected for downstream joint modeling.
        role: Proposed core or conditional evidence role.
        evidence_hash: Exact item of verified evidence interpreted by the choice.
        cluster_id: Measured redundancy cluster containing the factor.
        rationale: Bounded explanation of the proposed selection.
    """

    factor_id: str = Field(min_length=1)
    role: FactorResearchCandidateRole
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cluster_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    rationale: str = Field(min_length=1, max_length=500)


class FactorResearchProposal(_Contract):
    """Bind an actor's ordered choices and acknowledged limits to verified research evidence.

    Attributes:
        kind: Proposal discriminator.
        evidence_report_hash: Complete evidence report interpreted by the proposal.
        redundancy_structure_hash: Measured cluster structure used for selection.
        choices: Sorted unique choices by factor identity.
        limitations_acknowledged: Research limitations explicitly acknowledged by the actor.
        proposal_hash: Canonical identity of the proposal contents.
    """

    kind: Literal["FactorResearchProposal"] = "FactorResearchProposal"
    evidence_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    redundancy_structure_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    choices: tuple[FactorResearchProposalChoice, ...]
    limitations_acknowledged: tuple[str, ...]
    proposal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_proposal(self) -> FactorResearchProposal:
        """Verify unique ordered choices and canonical proposal identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Factor choices are not canonical or the proposal hash is invalid.
        """
        factor_ids = tuple(choice.factor_id for choice in self.choices)
        if factor_ids != tuple(sorted(set(factor_ids))):
            raise ValueError("Factor proposal choices are not canonical")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"proposal_hash"}))
        if self.proposal_hash != expected:
            raise ValueError("Factor proposal hash is invalid")
        return self


class FactorResearchExclusion(_Contract):
    """Retain evidence for a Factor omitted from the proposed modeling slate.

    Attributes:
        factor_id: Excluded factor identity.
        classification: Host-computed out-of-sample evidence classification.
        reason_codes: Recorded reasons behind the classification.
        evidence_hash: Exact complete-report evidence item for the omitted factor.
    """

    factor_id: str = Field(min_length=1)
    classification: FactorEvidenceClassification
    reason_codes: tuple[str, ...]
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class FactorEvidenceReference(_Contract):
    """Link one Factor to its complete-report evidence item.

    Attributes:
        factor_id: Factor whose evidence is retained.
        evidence_hash: Content identity of that evidence item.
    """

    factor_id: str = Field(min_length=1)
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class FactorRedundancyClusterReference(_Contract):
    """Retain the measured identity and member axis of a redundancy cluster.

    Attributes:
        cluster_id: Cluster handle used by proposed choices.
        cluster_hash: Measured cluster evidence identity.
        member_factor_ids: Nonempty factor axis belonging to the cluster.
    """

    cluster_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    cluster_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    member_factor_ids: tuple[str, ...] = Field(min_length=1)


class FactorHorizonResearchInput(_Contract):
    """Seal complete one-session evidence dispositions for downstream Alpha modeling.

    Attributes:
        kind: Research-input discriminator.
        horizon_sessions: One-session prediction horizon.
        formation_frequency: Daily formation schedule.
        execution_timing: Next-common-session open entry.
        evidence_report_hash: Qualified complete evidence report.
        redundancy_structure_hash: Qualified measured redundancy structure.
        proposal_hash: Actor proposal validated by the Host.
        core_factor_ids: Sorted unique factors supported by positive evidence.
        conditional_factor_ids: Sorted unique factors supported by mixed evidence.
        excluded_factors: All unselected factors and their measured evidence classifications.
        complete_factor_evidence_refs: Complete evidence axis, including exclusions.
        redundancy_cluster_refs: All measured clusters and their members.
        limitations: Caller-supplied research limits retained with the input.
        input_hash: Canonical identity of these dispositions and references.
    """

    kind: Literal["FactorHorizonResearchInput"] = "FactorHorizonResearchInput"
    horizon_sessions: Literal[1] = 1
    formation_frequency: Literal["DAILY"] = "DAILY"
    execution_timing: Literal["NEXT_COMMON_SESSION_OPEN"] = "NEXT_COMMON_SESSION_OPEN"
    evidence_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    redundancy_structure_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    core_factor_ids: tuple[str, ...]
    conditional_factor_ids: tuple[str, ...]
    excluded_factors: tuple[FactorResearchExclusion, ...]
    complete_factor_evidence_refs: tuple[FactorEvidenceReference, ...] = Field(min_length=1)
    redundancy_cluster_refs: tuple[FactorRedundancyClusterReference, ...] = Field(min_length=1)
    limitations: tuple[str, ...]
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_input(self) -> FactorHorizonResearchInput:
        """Verify disjoint canonical selections and complete evidence coverage.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Selection axes overlap or are noncanonical, dispositions omit evidence, or
                identity is invalid.
        """
        if self.core_factor_ids != tuple(sorted(set(self.core_factor_ids))):
            raise ValueError("Factor core axis is not canonical")
        if self.conditional_factor_ids != tuple(sorted(set(self.conditional_factor_ids))):
            raise ValueError("Factor conditional axis is not canonical")
        if set(self.core_factor_ids).intersection(self.conditional_factor_ids):
            raise ValueError("Factor core and conditional axes overlap")
        evidence_ids = tuple(item.factor_id for item in self.complete_factor_evidence_refs)
        excluded_ids = tuple(item.factor_id for item in self.excluded_factors)
        selected_ids = tuple(sorted((*self.core_factor_ids, *self.conditional_factor_ids)))
        if tuple(sorted((*excluded_ids, *selected_ids))) != evidence_ids:
            raise ValueError("Factor input dispositions do not cover the evidence axis")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"input_hash"}))
        if self.input_hash != expected:
            raise ValueError("Factor Research input hash is invalid")
        return self


def build_factor_research_proposal(
    *,
    evidence_report_hash: str,
    redundancy_structure_hash: str,
    choices: tuple[FactorResearchProposalChoice, ...],
    limitations_acknowledged: tuple[str, ...],
) -> FactorResearchProposal:
    """Sort candidate choices and seal their proposal against the referenced evidence.

    Args:
        evidence_report_hash: Complete report interpreted by the actor.
        redundancy_structure_hash: Cluster structure used in that interpretation.
        choices: Candidate selections to order by factor identity.
        limitations_acknowledged: Limits the actor acknowledges for this research.

    Returns:
        Validated proposal with its canonical content identity.

    Raises:
        ValueError: Choice identities are repeated or a proposal field is invalid.
    """
    ordered = tuple(sorted(choices, key=lambda choice: choice.factor_id))
    return seal_contract(
        FactorResearchProposal,
        "proposal_hash",
        evidence_report_hash=evidence_report_hash,
        redundancy_structure_hash=redundancy_structure_hash,
        choices=ordered,
        limitations_acknowledged=limitations_acknowledged,
    )


def compile_factor_horizon_research_input(
    *,
    evidence: FactorOosEvidenceReport,
    redundancy: FactorRedundancyStructure,
    proposal: FactorResearchProposal,
    limitations: tuple[str, ...],
) -> FactorHorizonResearchInput:
    """Validate an Agent proposal without granting it statistical authority.

    Args:
        evidence: Complete Host-computed report for the registered factor axis.
        redundancy: Measured redundancy structure on the same axis.
        proposal: Actor selections referring to those exact evidence items and clusters.
        limitations: Research limits to retain with the sealed input.

    Returns:
        Canonical core, conditional, and excluded dispositions covering the full evidence
        axis, with complete evidence and cluster references.

    Raises:
        FactorResearchInputBoundaryError: References or axes disagree, an eligible slate
            is empty, a selected role lacks its required evidence, or a multi-factor slate
            occupies one cluster when eligible alternatives span several.
        ValueError: Evidence, redundancy, proposal, or output contracts are invalid.
    """
    evidence = FactorOosEvidenceReport.model_validate(evidence)
    redundancy = FactorRedundancyStructure.model_validate(redundancy)
    proposal = FactorResearchProposal.model_validate(proposal)
    if proposal.evidence_report_hash != evidence.report_hash:
        raise FactorResearchInputBoundaryError("factor_research.proposal_evidence_stale")
    if proposal.redundancy_structure_hash != redundancy.structure_hash:
        raise FactorResearchInputBoundaryError("factor_research.proposal_redundancy_stale")
    if evidence.factor_ids != redundancy.factor_ids:
        raise FactorResearchInputBoundaryError("factor_research.input_factor_axis_mismatch")
    evidence_by_factor = {item.factor_id: item for item in evidence.items}
    cluster_by_factor = {
        factor_id: cluster
        for cluster in redundancy.clusters
        for factor_id in cluster.member_factor_ids
    }
    eligible = {
        item.factor_id
        for item in evidence.items
        if item.classification
        in {
            FactorEvidenceClassification.POSITIVE_OOS_EVIDENCE,
            FactorEvidenceClassification.MIXED_OOS_EVIDENCE,
        }
    }
    if eligible and not proposal.choices:
        raise FactorResearchInputBoundaryError(
            "factor_research.proposal_eligible_candidates_required"
        )
    selected_clusters: set[str] = set()
    for choice in proposal.choices:
        item = evidence_by_factor.get(choice.factor_id)
        cluster = cluster_by_factor.get(choice.factor_id)
        if item is None or cluster is None:
            raise FactorResearchInputBoundaryError("factor_research.proposal_factor_unknown")
        if choice.evidence_hash != item.evidence_hash or choice.cluster_id != cluster.cluster_id:
            raise FactorResearchInputBoundaryError("factor_research.proposal_reference_invalid")
        selected_clusters.add(cluster.cluster_id)
        if choice.role is FactorResearchCandidateRole.CORE:
            if item.classification is not FactorEvidenceClassification.POSITIVE_OOS_EVIDENCE:
                raise FactorResearchInputBoundaryError(
                    "factor_research.proposal_core_evidence_not_positive"
                )
        elif item.classification is not FactorEvidenceClassification.MIXED_OOS_EVIDENCE:
            raise FactorResearchInputBoundaryError(
                "factor_research.proposal_conditional_evidence_not_mixed"
            )
    eligible_clusters = {cluster_by_factor[factor_id].cluster_id for factor_id in eligible}
    if len(proposal.choices) > 1 and len(eligible_clusters) > 1 and len(selected_clusters) == 1:
        raise FactorResearchInputBoundaryError("factor_research.proposal_cluster_concentration")
    core = tuple(
        choice.factor_id
        for choice in proposal.choices
        if choice.role is FactorResearchCandidateRole.CORE
    )
    conditional = tuple(
        choice.factor_id
        for choice in proposal.choices
        if choice.role is FactorResearchCandidateRole.CONDITIONAL
    )
    selected = set((*core, *conditional))
    exclusions = tuple(
        FactorResearchExclusion(
            factor_id=item.factor_id,
            classification=item.classification,
            reason_codes=item.reason_codes,
            evidence_hash=item.evidence_hash,
        )
        for item in evidence.items
        if item.factor_id not in selected
    )
    evidence_refs = tuple(
        FactorEvidenceReference(factor_id=item.factor_id, evidence_hash=item.evidence_hash)
        for item in evidence.items
    )
    cluster_refs = tuple(
        FactorRedundancyClusterReference(
            cluster_id=cluster.cluster_id,
            cluster_hash=cluster.cluster_hash,
            member_factor_ids=cluster.member_factor_ids,
        )
        for cluster in redundancy.clusters
    )
    return seal_contract(
        FactorHorizonResearchInput,
        "input_hash",
        evidence_report_hash=evidence.report_hash,
        redundancy_structure_hash=redundancy.structure_hash,
        proposal_hash=proposal.proposal_hash,
        core_factor_ids=core,
        conditional_factor_ids=conditional,
        excluded_factors=exclusions,
        complete_factor_evidence_refs=evidence_refs,
        redundancy_cluster_refs=cluster_refs,
        limitations=limitations,
    )


__all__ = [
    "FactorEvidenceReference",
    "FactorHorizonResearchInput",
    "FactorRedundancyClusterReference",
    "FactorResearchCandidateRole",
    "FactorResearchExclusion",
    "FactorResearchInputBoundaryError",
    "FactorResearchProposal",
    "FactorResearchProposalChoice",
    "build_factor_research_proposal",
    "compile_factor_horizon_research_input",
]
