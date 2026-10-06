"""Actor-neutral review result and durable legacy receipt contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.factor_research.inputs.research_input import (
    FactorHorizonResearchInput,
    FactorResearchProposal,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution import (
    ActorSubmissionBinding,
)


class FactorResearchReviewSubmission(BaseModel):
    """Actor-neutral Factor curation proposed against already verified evidence.

    The submission deliberately contains only domain facts.  How those facts
    were produced is sealed separately by :class:`ActorSubmissionBinding`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["FactorResearchReviewSubmission"] = "FactorResearchReviewSubmission"
    dossier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal: FactorResearchProposal
    research_input: FactorHorizonResearchInput
    submission_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_submission(self) -> Self:
        """Verify the proposed research input belongs to this submission's proposal.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Proposal lineage or canonical submission identity differs.
        """
        if self.research_input.proposal_hash != self.proposal.proposal_hash:
            raise ValueError("Factor review submission research input differs from proposal")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"submission_hash"}))
        if self.submission_hash != expected:
            raise ValueError("Factor review submission hash is invalid")
        return self


class FactorResearchReviewDecisionReceipt(BaseModel):
    """Host-sealed decision evidence for one actor-neutral Factor submission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["FactorResearchReviewDecisionReceipt"] = "FactorResearchReviewDecisionReceipt"
    submission: FactorResearchReviewSubmission
    actor_submission: ActorSubmissionBinding
    decision_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_receipt(self) -> Self:
        """Verify the actor binding names the sealed domain submission.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Actor lineage or canonical decision-receipt identity differs.
        """
        if self.actor_submission.submission_hash != self.submission.submission_hash:
            raise ValueError("Factor review receipt actor binding differs from submission")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"receipt_hash"}))
        if self.receipt_hash != expected:
            raise ValueError("Factor review decision receipt hash is invalid")
        return self


__all__ = [
    "FactorResearchReviewDecisionReceipt",
    "FactorResearchReviewSubmission",
]
