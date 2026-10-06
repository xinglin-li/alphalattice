"""Actor-neutral contracts for Host-owned Data Preparation remediation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.market_data_ops.runtime.remediation import PolicyDecision
from alphalattice.foundation.market_data_ops.sources.contracts import DataRemediationProposal
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.contracts import (
    ActorKind,
    ActorSubmissionBinding,
    AgentExecutionBinding,
    ModelAttemptContextAudit,
)


@dataclass(frozen=True, slots=True)
class DataRemediationActorSubmission:
    """One actor's option selection before deterministic Data validation."""

    submission: DataRemediationExecutionSubmission
    actor_kind: ActorKind
    actor_id: str
    agent_execution: AgentExecutionBinding | None = None


class DataRemediationActorError(RuntimeError):
    """Observable terminal actor failure after bounded recovery."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str,
        execution_attempt_count: int,
        prior_failure_codes: tuple[str, ...],
        domain_tool_call_sequence: tuple[str, ...],
        agent_executions: tuple[AgentExecutionBinding, ...],
        model_context_audits: tuple[ModelAttemptContextAudit, ...],
    ) -> None:
        """Retain a remediation failure with exact attempt and actor provenance.

        Args:
            message: Human-readable error detail.
            failure_code: Stable failure code retained for diagnosis.
            execution_attempt_count: Number of attempted remediation executions.
            prior_failure_codes: Stable causes retained from earlier attempts.
            domain_tool_call_sequence: Ordered domain tool calls observed during execution.
            agent_executions: Actor execution bindings retained as provenance.
            model_context_audits: Model-attempt context audits retained by the execution owner.
        """
        super().__init__(message)
        self.failure_code = failure_code
        self.execution_attempt_count = execution_attempt_count
        self.prior_failure_codes = prior_failure_codes
        self.domain_tool_call_sequence = domain_tool_call_sequence
        self.agent_executions = agent_executions
        self.model_context_audits = model_context_audits


class DataRemediationExecutionSubmission(BaseModel):  # type: ignore[misc]
    """One actor-neutral selection from a Host-admitted remediation catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["DataRemediationExecutionSubmission"] = "DataRemediationExecutionSubmission"
    proposal: DataRemediationProposal
    submission_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_submission(self) -> Self:
        """Verify canonical identity of the submitted remediation record.

        Returns:
            This validated contract.

        Raises:
            ValueError: The submission hash differs from its canonical fields.
        """
        expected = canonical_hash(self.model_dump(mode="json", exclude={"submission_hash"}))
        if self.submission_hash != expected:
            raise ValueError("Data remediation submission hash is invalid")
        return self


class DataRemediationExecutionReceipt(BaseModel):  # type: ignore[misc]
    """Host-sealed remediation selection; mutation remains a separate policy owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["DataRemediationExecutionReceipt"] = "DataRemediationExecutionReceipt"
    submission: DataRemediationExecutionSubmission
    actor_submission: ActorSubmissionBinding
    policy_decision: PolicyDecision
    execution_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_receipt(self) -> Self:
        """Verify submission, policy-decision scope and canonical execution receipt identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Submission/policy decision bindings disagree or canonical receipt identity
                differs.
        """
        proposal = self.submission.proposal
        if (
            self.actor_submission.submission_hash != self.submission.submission_hash
            or proposal.case_token != self.policy_decision.case_token
            or proposal.evidence_hash != self.policy_decision.evidence_hash
            or proposal.option_id != self.policy_decision.option_id
        ):
            raise ValueError("Data remediation receipt differs from submitted policy decision")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"receipt_hash"}))
        if self.receipt_hash != expected:
            raise ValueError("Data remediation execution receipt hash is invalid")
        return self


__all__ = [
    "DataRemediationActorError",
    "DataRemediationActorSubmission",
    "DataRemediationExecutionReceipt",
    "DataRemediationExecutionSubmission",
]
