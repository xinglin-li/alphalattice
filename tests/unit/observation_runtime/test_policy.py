"""The recording policy refuses, before anything is kept, what the history must never hold."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from alphalattice.control.observation_runtime.contracts import (
    ObservationAuthority,
    ObservationAuthorityReadback,
    ObservationAuthorityReference,
    ObservationDraft,
    ObservationRetentionClass,
    ObservationSensitivity,
    build_observation_authority_reference,
)
from alphalattice.control.observation_runtime.policy import default_observation_policies

TEST_SOURCE = "src/alphalattice/control/observation_runtime/policy.py"
NOW = datetime(2026, 8, 8, 12, tzinfo=UTC)


class _Unreadable:
    """An authority store that holds no record, so every reference to it is unbacked."""

    def read_observation_authority(
        self, reference: ObservationAuthorityReference
    ) -> ObservationAuthorityReadback:
        raise FileNotFoundError(reference.record_hash)


def _draft(**changes: object) -> ObservationDraft:
    policy = default_observation_policies().policy("TaskControlTransition", 1)
    values: dict[str, object] = {
        "schema_kind": "TaskControlTransition",
        "schema_version": 1,
        "occurred_at": NOW,
        "source_kind": "TASK_CONTROL",
        "source_id": "task-1",
        "source_sequence": 1,
        "task_id": "task-1",
        "run_id": "run-1",
        "stage_id": "alpha",
        "authority": ObservationAuthority.TASK_CONTROL_ASSERTION,
        "sensitivity": ObservationSensitivity.PUBLIC_SAFE,
        "retention_class": ObservationRetentionClass.RUN_OPERATIONAL,
        "inline_safe_payload": {"status": "RUNNING", "stage_id": "alpha"},
        "policy_hash": policy.policy_hash,
    }
    return ObservationDraft(**(values | changes))


def _decision(reference: ObservationAuthorityReference) -> ObservationDraft:
    policy = default_observation_policies().policy("DecisionTraceObserved", 1)
    return _draft(
        schema_kind="DecisionTraceObserved",
        source_kind="TASK_CONTROLLER",
        source_id="decision-1",
        source_sequence=0,
        stage_id=None,
        authority=ObservationAuthority.HOST_DECISION,
        retention_class=ObservationRetentionClass.DURABLE_DECISION,
        inline_safe_payload={
            "decision_trace_hash": reference.record_hash,
            "external_authority": reference.model_dump(mode="json"),
            "scientific_stop_preserved": True,
            "risk_admitted": False,
        },
        policy_hash=policy.policy_hash,
    )


@pytest.mark.proves("observation_runtime.records_refuse_what_they_must_never_hold")
@pytest.mark.parametrize(
    ("changes", "code"),
    [
        pytest.param(
            {"inline_safe_payload": {"status": "RUNNING", "api_key": "not-recorded"}},
            "redaction_failed",
            id="secret-like-field",
        ),
        pytest.param(
            {"inline_safe_payload": {"status": "RUNNING", "unknown": "no"}},
            "payload_not_admitted",
            id="unknown-field",
        ),
        pytest.param(
            {"sensitivity": ObservationSensitivity.LOCAL_SENSITIVE},
            "payload_not_admitted",
            id="local-sensitive",
        ),
        pytest.param(
            {"sensitivity": ObservationSensitivity.PRIVATE_RUNTIME},
            "payload_not_admitted",
            id="private-runtime",
        ),
        pytest.param(
            {"retention_class": ObservationRetentionClass.INTERNAL_COGNITION_ELIGIBLE},
            "payload_not_admitted",
            id="internal-cognition",
        ),
        pytest.param(
            {"retention_class": ObservationRetentionClass.DURABLE_DECISION},
            "required_decision_missing",
            id="decision-without-authority",
        ),
    ],
)
def test_the_policy_refuses_secrets_private_cognition_and_unbacked_decisions(
    changes: dict[str, object], code: str
) -> None:
    """A secret-like or unknown field, a private or cognition payload and a decision with no
    authority are each refused by name before anything is kept."""
    with pytest.raises(ValueError, match=rf"^observation\.{code}$"):
        default_observation_policies().validate_and_redact(_draft(**changes))


@pytest.mark.proves("observation_runtime.records_refuse_what_they_must_never_hold")
def test_a_decision_naming_an_unreadable_authority_is_refused() -> None:
    """A hash-shaped authority its owner cannot read back backs no decision."""
    reference = build_observation_authority_reference(
        owner_kind="ALPHA_SPARSE_LOOP_ARTIFACT_STORE",
        record_kind="DecisionTrace",
        record_hash="c" * 64,
    )
    unreadable = default_observation_policies(authority_reader=_Unreadable())
    with pytest.raises(ValueError, match=r"^observation\.required_decision_missing$"):
        unreadable.validate_and_redact(_decision(reference))
