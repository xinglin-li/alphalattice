"""The seal of one validated data remediation selection.

Once the owning deterministic gateway has validated a submitted option, the selection is sealed as
the actor-neutral receipt the deterministic Data policy may consume, before any Data mutation is
considered.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from alphalattice.control.data_platform.contracts import (
    DataRemediationExecutionReceipt,
    DataRemediationExecutionSubmission,
)
from alphalattice.foundation.feature_engine.inputs.gateway import (
    FeatureInputAgentCase,
    FeatureInputExecutionStatus,
    FeatureInputPolicyExecution,
    RawRetentionDecisionProof,
)
from alphalattice.foundation.feature_engine.inputs.gateway import (
    matching_raw_retention_proof as match_verified_raw_retention_proof,
)
from alphalattice.foundation.market_data_ops.runtime.remediation import (
    PolicyDecision,
    canonical_hash,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.protocols.actor_execution import (
    ActorKind,
    AgentExecutionBinding,
    seal_actor_submission,
)

if TYPE_CHECKING:
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository


def raw_retention_decision_proofs(
    panel_state: PanelStateRepository,
    market_data: MarketDataRepository,
) -> tuple[RawRetentionDecisionProof, ...]:
    """Verify original Host receipts before reusing a raw-retention caveat."""
    proofs = []
    for case_document, resolution in panel_state.feature_input_raw_retention_decisions():
        case = FeatureInputAgentCase.read_document(case_document)
        receipt_value = resolution.get("receipt")
        effect_value = resolution.get("effect")
        if not isinstance(receipt_value, dict) or not isinstance(effect_value, dict):
            continue
        try:
            receipt = DataRemediationExecutionReceipt.model_validate(receipt_value)
            effect = TypeAdapter(FeatureInputPolicyExecution).validate_python(effect_value)
        except (TypeError, ValueError):
            continue
        if effect.status is not FeatureInputExecutionStatus.RAW_VALUE_RETAINED:
            continue
        option = next(
            (
                value
                for value in case.options
                if value.option_id == "retain_isolated_raw_move_with_caveat"
            ),
            None,
        )
        decision = receipt.policy_decision
        policy_args_hash = (
            canonical_hash(option.policy_args.model_dump(mode="json")) if option else ""
        )
        effect_base = (
            canonical_hash(
                [case.case_token, case.evidence_hash, option.option_hash, policy_args_hash]
            )
            if option
            else ""
        )
        if (
            option is None
            or decision.case_token != case.case_token
            or decision.evidence_hash != case.evidence_hash
            or decision.option_id != option.option_id
            or decision.option_hash != option.option_hash
            or decision.policy_hash != canonical_hash(option.policy_args)
            or decision.disposition is not option.disposition
            or receipt.execution_policy_hash != case.policy_hash
            or receipt.actor_submission.actor_kind
            not in {
                ActorKind.HUMAN,
                ActorKind.EXTERNAL_AUTOMATION,
            }
            or effect.option_id != option.option_id
            or effect.option_hash != option.option_hash
            or effect.policy_args_hash != policy_args_hash
            or effect.execution_receipt_hash
            != canonical_hash([effect_base, "HUMAN_CONFIRMED", receipt.receipt_hash])
            or tuple(option.target_listing_ids) != case.listing_ids
            or tuple(sorted(effect.retained_evidence))
            != tuple(sorted((item.listing_id, item.evidence_hash) for item in case.evidence))
        ):
            continue
        try:
            source_manifest = market_data.load_universe_manifest_revision(case.manifest_revision)
        except ValueError:
            continue
        proofs.append(
            RawRetentionDecisionProof(
                case=case,
                receipt_hash=receipt.receipt_hash,
                execution_receipt_hash=effect.execution_receipt_hash,
                execution_policy_hash=receipt.execution_policy_hash,
                actor_kind=receipt.actor_submission.actor_kind.value,
                actor_id=receipt.actor_submission.actor_id,
                option_hash=option.option_hash,
                retained_evidence=effect.retained_evidence,
                receipt_document=json.dumps(
                    receipt.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
                ),
                effect_document=json.dumps(
                    TypeAdapter(FeatureInputPolicyExecution).dump_python(effect, mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                market_profile_id=source_manifest.profile.market_profile_id,
                membership_fingerprint=source_manifest.membership_fingerprint,
                universe_membership_basis=source_manifest.universe_membership_basis,
                universe_policy_type=source_manifest.universe_policy_type,
                universe_components=source_manifest.universe_components,
                research_use_class=source_manifest.research_use_class,
                source_listing_ids=tuple(item.listing_id for item in source_manifest.listings),
            )
        )
    return tuple(proofs)


def matching_raw_retention_proof(
    *,
    panel_state: PanelStateRepository,
    market_data: MarketDataRepository,
    candidate_manifest: UniverseManifest,
    case: FeatureInputAgentCase,
    proofs: tuple[RawRetentionDecisionProof, ...] | None = None,
) -> RawRetentionDecisionProof | None:
    """Find a prior sealed choice for the same case evidence and membership origin."""
    proof_values = (
        proofs if proofs is not None else raw_retention_decision_proofs(panel_state, market_data)
    )
    return match_verified_raw_retention_proof(
        case=case,
        candidate_manifest=candidate_manifest,
        proofs=proof_values,
        policy_hash=case.policy_hash,
    )


def seal_validated_data_remediation_execution(
    *,
    submission: DataRemediationExecutionSubmission,
    policy_decision: PolicyDecision,
    actor_kind: ActorKind,
    actor_id: str,
    execution_policy_hash: str,
    agent_execution: AgentExecutionBinding | None = None,
) -> DataRemediationExecutionReceipt:
    """Seal a selection after its owning deterministic gateway validates it."""
    submission = DataRemediationExecutionSubmission.model_validate(submission)
    policy_decision = PolicyDecision.model_validate(policy_decision)
    proposal = submission.proposal
    if (
        proposal.case_token != policy_decision.case_token
        or proposal.evidence_hash != policy_decision.evidence_hash
        or proposal.option_id != policy_decision.option_id
    ):
        raise ValueError("Data remediation policy decision differs from submission")
    actor_submission = seal_actor_submission(
        actor_kind=actor_kind,
        actor_id=actor_id,
        submission_hash=submission.submission_hash,
        agent_execution=agent_execution,
    )
    values = {
        "submission": submission,
        "actor_submission": actor_submission,
        "policy_decision": policy_decision,
        "execution_policy_hash": execution_policy_hash,
    }
    draft = DataRemediationExecutionReceipt.model_construct(**values, receipt_hash="0" * 64)
    return DataRemediationExecutionReceipt(
        **values,
        receipt_hash=canonical_hash(draft.model_dump(mode="json", exclude={"receipt_hash"})),
    )
