from __future__ import annotations

import pytest

from alphalattice.control.data_platform.contracts import (
    DataRemediationExecutionReceipt,
    DataRemediationExecutionSubmission,
)
from alphalattice.control.data_platform.remediation_case import (
    seal_validated_data_remediation_execution,
)
from alphalattice.foundation.market_data_ops.runtime.remediation import (
    EscalateForHumanArgs,
    PolicyDecision,
    PolicyDisposition,
    RemediationAction,
    RemediationOption,
    canonical_hash,
)
from alphalattice.foundation.market_data_ops.sources.contracts import DataRemediationProposal
from alphalattice.protocols.actor_execution import (
    ActorKind,
    AgentExecutionBinding,
)
from tests._support.identity_fixtures import fixture_digest


def _option() -> RemediationOption:
    return RemediationOption.create(
        option_id="escalate_feature_admission_conflict",
        target_listing_ids=("listing-001",),
        disposition=PolicyDisposition.HUMAN_REVIEW,
        policy_args=EscalateForHumanArgs(
            action=RemediationAction.ESCALATE_FOR_HUMAN,
            review_kind="feature_admission_data_truth_conflict",
        ),
    )


def _decision(
    option: RemediationOption, *, case_token: str = fixture_digest("a")
) -> PolicyDecision:
    """What the owning gateway returns once it has validated the selected option."""

    return PolicyDecision(
        case_token=case_token,
        evidence_hash=fixture_digest("b"),
        option_id=option.option_id,
        option_hash=option.option_hash,
        policy_hash=canonical_hash(option.policy_args),
        disposition=option.disposition,
    )


def _submission(option: RemediationOption) -> DataRemediationExecutionSubmission:
    proposal = DataRemediationProposal(
        run_id="run-001",
        case_token=fixture_digest("a"),
        evidence_hash=fixture_digest("b"),
        option_id=option.option_id,
        rationale="Use the one Host-admitted human review option.",
    )
    values = {"proposal": proposal}
    draft = DataRemediationExecutionSubmission.model_construct(
        **values, submission_hash=fixture_digest("0")
    )
    return DataRemediationExecutionSubmission(
        **values,
        submission_hash=canonical_hash(draft.model_dump(mode="json", exclude={"submission_hash"})),
    )


def _agent_execution() -> AgentExecutionBinding:
    return AgentExecutionBinding(
        profile_id="data-operations.remediation-advisor",
        mode="diagnose_remediation",
        profile_hash=fixture_digest("c"),
        document_hash=fixture_digest("c"),
        response_protocol="data_remediation_selection",
        response_protocol_hash=fixture_digest("c"),
        concrete_schema_hash=fixture_digest("c"),
    )


def test_a_validated_remediation_is_sealed_actor_neutral_before_mutation() -> None:
    """requirement: who selected the option changes only the actor binding, never the
    submission or the policy decision a Data mutation consumes; a tampered receipt is refused."""

    option = _option()
    common = {
        "submission": _submission(option),
        "policy_decision": _decision(option),
        "execution_policy_hash": fixture_digest("d"),
    }
    human = seal_validated_data_remediation_execution(
        **common, actor_kind=ActorKind.HUMAN, actor_id="researcher@example.test"
    )
    external = seal_validated_data_remediation_execution(
        **common, actor_kind=ActorKind.EXTERNAL_AUTOMATION, actor_id="external-codex"
    )
    installed = seal_validated_data_remediation_execution(
        **common,
        actor_kind=ActorKind.INSTALLED_AGENT,
        actor_id="data-operations.remediation-advisor",
        agent_execution=_agent_execution(),
    )
    assert human.submission.submission_hash == external.submission.submission_hash
    assert human.submission.submission_hash == installed.submission.submission_hash
    assert human.actor_submission.binding_hash != external.actor_submission.binding_hash
    assert external.actor_submission.binding_hash != installed.actor_submission.binding_hash
    assert human.policy_decision == external.policy_decision == installed.policy_decision
    assert human.actor_submission.agent_execution is None
    assert installed.actor_submission.agent_execution == _agent_execution()

    payload = human.model_dump(mode="json")
    payload["actor_submission"]["actor_id"] = "tampered"
    with pytest.raises(ValueError, match="actor submission binding identity"):
        DataRemediationExecutionReceipt.model_validate(payload)


def test_a_policy_decision_for_another_case_is_refused() -> None:
    """tamper: the sealer binds the submission to the gateway's decision for the same case."""

    option = _option()
    with pytest.raises(ValueError, match="policy decision differs from submission"):
        seal_validated_data_remediation_execution(
            submission=_submission(option),
            policy_decision=_decision(option, case_token=fixture_digest("e")),
            actor_kind=ActorKind.HUMAN,
            actor_id="researcher@example.test",
            execution_policy_hash=fixture_digest("d"),
        )
