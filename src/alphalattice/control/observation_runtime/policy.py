"""Code-owned observation schemas and redaction-before-append policy."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import (
    ObservationAuthority,
    ObservationAuthorityReader,
    ObservationAuthorityReference,
    ObservationDraft,
    ObservationRetentionClass,
    ObservationSensitivity,
)

_DURABLE_AUTHORITY_COMPATIBILITY = {
    "ObservationBaselineSnapshot": (
        ObservationAuthority.HOST_DECISION,
        "ALPHA_SPARSE_LOOP_ARTIFACT_STORE",
        "ObservationBaselineSnapshot",
        "baseline_hash",
    ),
    "DecisionTraceObserved": (
        ObservationAuthority.HOST_DECISION,
        "ALPHA_SPARSE_LOOP_ARTIFACT_STORE",
        "DecisionTrace",
        "decision_trace_hash",
    ),
    "SparseLoopPublicationObserved": (
        ObservationAuthority.ARTIFACT_ASSERTION,
        "ALPHA_SPARSE_LOOP_ARTIFACT_STORE",
        "SparseLoopRunMarker",
        "marker_hash",
    ),
    "SparseLoopSampleFailureObserved": (
        ObservationAuthority.HOST_DECISION,
        "ALPHA_SPARSE_LOOP_ARTIFACT_STORE",
        "SparseLoopFailureFallback",
        "failure_fallback_hash",
    ),
    "AlphaGoalPublicationObserved": (
        ObservationAuthority.ARTIFACT_ASSERTION,
        "ALPHA_GOAL_RESEARCH_ARTIFACT_STORE",
        "AlphaGoalResearchMarker",
        "marker_hash",
    ),
}

_PROHIBITED_KEY_PARTS = frozenset(
    {
        "api_key",
        "credential",
        "password",
        "secret",
        "authorization",
        "chain_of_thought",
        "reasoning_content",
        "todo_text",
        "raw_prompt",
        "raw_tool_argument",
        "raw_tool_result",
    }
)


@dataclass(frozen=True, slots=True)
class ObservationSchemaPolicy:
    """The fields, size and sensitivity admitted for one observation schema."""

    schema_kind: str
    schema_version: int
    allowed_fields: frozenset[str]
    max_inline_bytes: int = 8192
    admitted_sensitivities: frozenset[ObservationSensitivity] = frozenset(
        {ObservationSensitivity.PUBLIC_SAFE}
    )
    rejected_retention_classes: frozenset[ObservationRetentionClass] = frozenset(
        {
            ObservationRetentionClass.INTERNAL_COGNITION_ELIGIBLE,
            ObservationRetentionClass.NEVER_RECORD,
        }
    )

    @property
    def policy_hash(self) -> str:
        """Return the content hash of the installed redaction policy."""
        return cast(
            str,
            canonical_hash(
                {
                    "schema_kind": self.schema_kind,
                    "schema_version": self.schema_version,
                    "allowed_fields": tuple(sorted(self.allowed_fields)),
                    "max_inline_bytes": self.max_inline_bytes,
                    "prohibited_key_parts": tuple(sorted(_PROHIBITED_KEY_PARTS)),
                    "admitted_sensitivities": tuple(
                        sorted(value.value for value in self.admitted_sensitivities)
                    ),
                    "rejected_retention_classes": tuple(
                        sorted(value.value for value in self.rejected_retention_classes)
                    ),
                }
            ),
        )


class ObservationPolicyRegistry:
    """Admit only named, bounded safe observation payloads."""

    def __init__(
        self,
        policies: Iterable[ObservationSchemaPolicy],
        *,
        authority_reader: ObservationAuthorityReader | None = None,
    ) -> None:
        """Index unique schema policies and the optional owner readback.

        Args:
            policies: Installed policies keyed by schema kind and version.
            authority_reader: Owner readback for durable decision references.

        Raises:
            ValueError: Two policies claim the same schema key.

        """
        admitted = tuple(policies)
        keyed = {(item.schema_kind, item.schema_version): item for item in admitted}
        if len(keyed) != len(admitted):
            raise ValueError("duplicate observation schema policy")
        self._policies = keyed
        self._authority_reader = authority_reader

    def policy(self, schema_kind: str, schema_version: int) -> ObservationSchemaPolicy:
        """Return the policy for one installed schema, refusing unknown keys."""
        try:
            return self._policies[(schema_kind, schema_version)]
        except KeyError as exc:
            raise ValueError("observation.payload_not_admitted") from exc

    @property
    def authority_reader(self) -> ObservationAuthorityReader | None:
        """Return the owner reader used to verify durable authority."""
        return self._authority_reader

    def validate_and_redact(self, draft: ObservationDraft) -> ObservationDraft:
        """Admit a bounded, public-safe payload after policy and authority checks.

        Args:
            draft: Proposed observation before redaction and append.

        Returns:
            A copy with sanitized inline fields.

        Raises:
            ValueError: The schema, sensitivity, fields or authority is refused.

        """
        policy = self.policy(draft.schema_kind, draft.schema_version)
        if draft.policy_hash != policy.policy_hash:
            raise ValueError("observation.payload_not_admitted")
        if draft.sensitivity not in policy.admitted_sensitivities:
            raise ValueError("observation.payload_not_admitted")
        if draft.retention_class in policy.rejected_retention_classes:
            raise ValueError("observation.payload_not_admitted")
        if draft.inline_safe_payload is None:
            if draft.retention_class is ObservationRetentionClass.DURABLE_DECISION:
                raise ValueError("observation.required_decision_missing")
            return draft
        payload = _sanitize_mapping(draft.inline_safe_payload)
        unexpected = set(payload).difference(policy.allowed_fields)
        if unexpected:
            raise ValueError("observation.payload_not_admitted")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > policy.max_inline_bytes:
            raise ValueError("observation.payload_not_admitted")
        if draft.retention_class is ObservationRetentionClass.DURABLE_DECISION:
            self.verify_durable_authority(draft=draft, payload=payload)
        return cast(
            ObservationDraft,
            draft.model_copy(update={"inline_safe_payload": payload}),
        )

    def verify_durable_authority(
        self, *, draft: ObservationDraft, payload: Mapping[str, Any]
    ) -> ObservationAuthorityReference:
        """Verify that a durable fact refers to the matching owner record.

        Args:
            draft: Durable observation claiming the authority.
            payload: Sanitized fields including the external reference.

        Returns:
            The validated external authority reference.

        Raises:
            ValueError: The reference or owner readback is absent or mismatched.

        """
        expected = _DURABLE_AUTHORITY_COMPATIBILITY.get(draft.schema_kind)
        raw_reference = payload.get("external_authority")
        if expected is None or not isinstance(raw_reference, Mapping):
            raise ValueError("observation.required_decision_missing")
        try:
            reference = cast(
                ObservationAuthorityReference,
                ObservationAuthorityReference.model_validate(raw_reference),
            )
        except ValueError as exc:
            raise ValueError("observation.required_decision_missing") from exc
        expected_authority, expected_owner, expected_record, payload_hash_field = expected
        if (
            draft.authority is not expected_authority
            or reference.owner_kind != expected_owner
            or reference.record_kind != expected_record
            or payload.get(payload_hash_field) != reference.record_hash
        ):
            raise ValueError("observation.required_decision_missing")
        if draft.schema_kind in {
            "DecisionTraceObserved",
            "SparseLoopPublicationObserved",
            "AlphaGoalPublicationObserved",
        } and (
            reference.task_id is None
            or reference.run_id is None
            or reference.subject_id is None
            or reference.task_id != draft.task_id
            or reference.run_id != draft.run_id
        ):
            raise ValueError("observation.required_decision_missing")
        if draft.schema_kind in {
            "ObservationBaselineSnapshot",
            "DecisionTraceObserved",
            "SparseLoopPublicationObserved",
        } and (
            payload.get("scientific_stop_preserved") is not True
            or payload.get("risk_admitted") is not False
        ):
            raise ValueError("observation.required_decision_missing")
        if (
            draft.schema_kind == "AlphaGoalPublicationObserved"
            and payload.get("risk_admitted") is not False
        ):
            raise ValueError("observation.required_decision_missing")
        if self._authority_reader is None:
            raise ValueError("observation.required_decision_missing")
        try:
            readback = self._authority_reader.read_observation_authority(reference)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("observation.required_decision_missing") from exc
        if (
            readback.reference_hash != reference.reference_hash
            or readback.owner_kind != reference.owner_kind
            or readback.record_kind != reference.record_kind
            or readback.record_hash != reference.record_hash
            or readback.task_id != reference.task_id
            or readback.run_id != reference.run_id
            or readback.subject_id != reference.subject_id
        ):
            raise ValueError("observation.required_decision_missing")
        if draft.schema_kind == "AlphaGoalPublicationObserved":
            safe_claims = dict(readback.safe_claims)
            required_claims = {
                "attempted_spec_count",
                "candidate_registry_hash",
                "completed_batch_count",
                "current_qualified_count",
                "goal_target",
                "marker_hash",
                "qualification_hash",
                "remaining_batch_count",
                "remaining_spec_count",
                "result_ref_hash",
                "risk_admitted",
                "scientific_stop_preserved",
                "status",
            }
            if set(safe_claims) != required_claims or any(
                payload.get(key) != safe_claims[key] for key in required_claims
            ):
                raise ValueError("observation.required_decision_missing")
        return reference


def _sanitize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        lowered = key.lower()
        if any(part in lowered for part in _PROHIBITED_KEY_PARTS):
            raise ValueError("observation.redaction_failed")
        sanitized[key] = _sanitize_value(item)
    return sanitized


def _sanitize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return _sanitize_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    raise ValueError("observation.redaction_failed")


PRODUCT_OPERATION_SCHEMA = "ProductOperationObserved"
"""One shared application operation as its entry saw it: requested, returned or failed.

Recorded by the Host composition around the actor-neutral operation owner, so the
caller class is the entry's own statement and never a request field. The payload
carries references (hashes, ids, operation names), never documents, specs, YAML,
briefs, assessments or any other user-authored content.
"""

PRODUCT_OPERATION_FIELDS = frozenset(
    {
        "operation",
        "caller",
        "phase",
        "operation_ref",
        "status",
        "failure_code",
        "failure_type",
        "task_id",
        "task_kind",
        "task_lifecycle",
        "subject",
        "next_read",
        "latency_milliseconds",
        "request_fields",
    }
)

EXTERNAL_ACTIVITY_SCHEMA = "ExternalActivityObserved"
"""One event declared by an admitted external client (for example a native host
adapter), recorded with the trust level the admitting boundary assigned. A
bounded summary and references only; never a transcript, tool payload or file."""

EXTERNAL_ACTIVITY_MAX_INLINE_BYTES = 48 * 1024
"""Cover the complete admitted payload, including JSON's astral Unicode escapes.

The kept 500-character summary and sixteen 200-character subject values take at
most 44,400 escaped bytes. Bounded ASCII keys, producer tokens, a signed SQLite
sequence and payload framing fit in the remaining 4,752 bytes. Other observation
schemas keep their own bounds.
"""

EXTERNAL_ACTIVITY_FIELDS = frozenset(
    {
        "event_kind",
        "producer_id",
        "producer_session",
        "producer_sequence",
        "summary",
        "summary_truncated",
        "subject",
        "task_verified",
    }
)


def default_observation_policies(
    *, authority_reader: ObservationAuthorityReader | None = None
) -> ObservationPolicyRegistry:
    """Install the bounded public-safe schemas used by observation producers."""
    common = {
        "status",
        "failure_code",
        "failure_type",
        "task_lifecycle",
        "stage_id",
        "task_class",
        "task_status",
        "task_generation",
        "runtime_policy_hash",
        "thread_id_hash",
        "checkpoint_id_hash",
        "event_kind",
        "disposition",
        "attempt",
        "heartbeat_freshness",
        "completed_units",
        "total_units",
        "unit_name",
        "model_calls",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "latency_milliseconds",
        "peak_rss_bytes",
        "tool_name",
        "tool_schema_hash",
        "result_ref_hash",
        "feedback_code",
        "preserved_result_count",
        "legal_next_actions",
        "artifact_kind",
        "artifact_hash",
        "availability",
        "projection_hash",
        "verified_prefix_count",
        "recovery_option_ids",
        "decision_trace_hash",
        "marker_hash",
        "scientific_stop_preserved",
        "risk_admitted",
        "source_state_hash",
        "baseline_hash",
        "root_reference_hashes",
        "root_kind",
        "byte_count",
        "compacted_count",
        "evicted_count",
        "current_obligation_id",
        "waiting_job_ref",
        "pending_review_ref",
        "last_verified_result_ref",
        "failure_fallback_hash",
        "incident_code",
        "user_safe_detail",
        "model_call_limit",
        "tool_call_limit",
        "last_heartbeat_at",
        "execution_id",
        "heartbeat_sequence",
        "liveness_status",
        "external_authority",
        "goal_target",
        "current_qualified_count",
        "attempted_spec_count",
        "completed_batch_count",
        "remaining_batch_count",
        "remaining_spec_count",
        "candidate_registry_hash",
        "qualification_hash",
    }
    kinds = (
        "ObservationBaselineSnapshot",
        "TaskControlTransition",
        "TaskCommandObserved",
        "TaskStageVerified",
        "WorkProgressObserved",
        "GraphNodeObserved",
        "AgentTurnObserved",
        "ModelCallObserved",
        "ToolLifecycleObserved",
        "WorkerLifecycleObserved",
        "RuntimeFeedbackObserved",
        "ArtifactVerificationObserved",
        "GuanyinIncidentObserved",
        "GuanyinProjectionObserved",
        "DecisionTraceObserved",
        "ObservationRetentionObserved",
        "SparseLoopPublicationObserved",
        "SparseLoopSampleFailureObserved",
        "AlphaGoalProgressObserved",
        "AlphaGoalPublicationObserved",
    )
    return ObservationPolicyRegistry(
        (
            *(ObservationSchemaPolicy(kind, 1, frozenset(common)) for kind in kinds),
            ObservationSchemaPolicy(
                PRODUCT_OPERATION_SCHEMA, 1, PRODUCT_OPERATION_FIELDS, max_inline_bytes=4096
            ),
            ObservationSchemaPolicy(
                EXTERNAL_ACTIVITY_SCHEMA, 1, EXTERNAL_ACTIVITY_FIELDS, max_inline_bytes=4096
            ),
            ObservationSchemaPolicy(
                EXTERNAL_ACTIVITY_SCHEMA,
                2,
                EXTERNAL_ACTIVITY_FIELDS,
                max_inline_bytes=EXTERNAL_ACTIVITY_MAX_INLINE_BYTES,
            ),
        ),
        authority_reader=authority_reader,
    )


__all__ = [
    "EXTERNAL_ACTIVITY_FIELDS",
    "EXTERNAL_ACTIVITY_MAX_INLINE_BYTES",
    "EXTERNAL_ACTIVITY_SCHEMA",
    "PRODUCT_OPERATION_FIELDS",
    "PRODUCT_OPERATION_SCHEMA",
    "ObservationPolicyRegistry",
    "ObservationSchemaPolicy",
    "default_observation_policies",
]
