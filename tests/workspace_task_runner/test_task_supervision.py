"""Guanyin's Supervisor and Recovery Center over Task Control (GY2)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from alphalattice.control.guanyin.tasks.supervision import (
    STAGE_STALLED_AFTER,
    Finding,
    IncidentRecord,
    IncidentStore,
    OfferedRemedy,
    TaskFacts,
    classify,
    first_detection,
    incident_key,
)
from alphalattice.control.product_host.composition.task_recovery import (
    TaskAttentionFact,
    task_attention,
)
from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord, TaskRecoveryLink
from tests.workspace_task_runner.task_control_support import task_contract

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def _facts(**values: object) -> TaskFacts:
    base: dict[str, object] = {
        "task_id": "t-1",
        "task_kind": "research_experiment",
        "execution_id": "e-1",
        "lifecycle": "RUNNING",
        "operation_running": True,
        "liveness": "OBSERVED",
        "telemetry": "OPERATIONAL",
        "current_stage": "execute_sealed_experiment",
        "stage_started_at": NOW - timedelta(minutes=5),
    }
    return TaskFacts(**{**base, **values})  # type: ignore[arg-type]


def _task_record(
    *,
    lifecycle: TaskLifecycle = TaskLifecycle.BLOCKED,
    task_id: UUID | None = None,
    execution_id: UUID | None = None,
    version: int = 1,
    salt: str = "current",
) -> TaskRecord:
    """Build a canonical Task contract for the public attention projection."""
    envelope, goal, plan = task_contract(salt=salt)
    if execution_id is None and lifecycle is not TaskLifecycle.QUEUED:
        execution_id = uuid4()
    return TaskRecord.from_identity(
        task_id=task_id or uuid4(),
        task_kind=envelope.task_kind,
        input=envelope,
        goal=goal,
        plan=plan,
        lifecycle=lifecycle,
        active_work_item_id=None,
        latest_execution_id=execution_id,
        admitted_at=NOW,
        started_at=None if lifecycle is TaskLifecycle.QUEUED else NOW,
        updated_at=NOW + timedelta(seconds=version),
        failure_code=(
            "TASK_EXECUTION_INTERRUPTED"
            if lifecycle in {TaskLifecycle.BLOCKED, TaskLifecycle.RECOVERY_REQUIRED}
            else None
        ),
        version=version,
    )


def _incident(
    task: TaskRecord,
    *,
    state: str,
    key: str,
    resolved_task_record_hash: str | None = None,
) -> IncidentRecord:
    """Create a sealed incident whose execution belongs to this canonical Task."""
    facts = _facts(
        task_id=str(task.task_id),
        task_kind=task.task_kind,
        execution_id=(None if task.latest_execution_id is None else str(task.latest_execution_id)),
        lifecycle="RECOVERY_REQUIRED",
        operation_running=False,
    )
    return IncidentRecord.create(
        key=key,
        task_id=str(task.task_id),
        task_kind=task.task_kind,
        execution_id=facts.execution_id,
        code="task_runtime.recovery_required",
        incident=first_detection(facts, Finding("task_runtime.recovery_required", None), NOW),
        state=state,  # type: ignore[arg-type]
        last_seen_at=NOW,
        resolved_at=NOW + timedelta(minutes=1) if state == "RESOLVED" else None,
        resolved_task_record_hash=resolved_task_record_hash,
        remedies=(),
    )


def test_the_rules_tell_running_work_from_each_incident_and_a_valid_stop() -> None:
    """requirement (GY2): running, stale liveness, stalled work, missing host visibility, a
    Task owed a recovery and a parked one are told apart by rule; a review, a deferral or a
    cancellation in progress is the Task's own state, never an incident."""

    assert classify(_facts(), NOW) is None
    codes = {
        "task_runtime.liveness_stale": _facts(liveness="NOT_RECENT"),
        "task_runtime.host_visibility_missing": _facts(telemetry="UNREADABLE"),
        "task_runtime.work_stalled": _facts(stage_started_at=NOW - STAGE_STALLED_AFTER),
        "task_runtime.recovery_required": _facts(
            lifecycle="RECOVERY_REQUIRED", operation_running=False
        ),
        "task_runtime.parked": _facts(lifecycle="QUEUED", operation_running=False),
    }
    for code, facts in codes.items():
        finding = classify(facts, NOW)
        assert finding is not None and finding.code == code, (code, finding)
    for lifecycle in ("REVIEW_PENDING", "DEFERRED", "CANCEL_REQUESTED"):
        assert classify(_facts(lifecycle=lifecycle, liveness="NOT_RECENT"), NOW) is None
    assert classify(_facts(lifecycle="QUEUED"), NOW) is None  # a command still drives it
    # Regression: a recovery the dispatcher holds waits its turn, as a queued Task does
    # an incident on it woke a resume's follower while its recovery was starting.
    assert classify(_facts(lifecycle="RECOVERY_REQUIRED"), NOW) is None


def test_an_incident_record_keeps_its_key_seals_its_lifecycle_and_refuses_a_changed_file(
    tmp_path: Path,
) -> None:
    """requirement (GY2): an incident's key stays while its Task, execution, rule and stage
    do, so it is recorded (and wakes) once; its record is sealed, a changed file is refused by
    name, and the open ones read first."""

    facts = _facts(liveness="NOT_RECENT")
    finding = classify(facts, NOW)
    assert finding is not None
    key = incident_key(facts, finding)
    assert key == incident_key(facts, finding)
    assert key != incident_key(_facts(execution_id="e-2", liveness="NOT_RECENT"), finding)
    store = IncidentStore(tmp_path)
    opened = IncidentRecord.create(
        key=key,
        task_id=facts.task_id,
        task_kind=facts.task_kind,
        execution_id=facts.execution_id,
        code=finding.code,
        incident=first_detection(facts, finding, NOW),
        state="OPEN",
        last_seen_at=NOW,
        remedies=(OfferedRemedy(action="CANCEL", operation="CANCEL", available=True, reason="r"),),
    )
    store.put(opened)
    assert store.get(key) == opened
    resolved = opened.changed(state="RESOLVED", resolved_at=NOW + timedelta(minutes=1))
    other = _facts(task_id="t-2", lifecycle="RECOVERY_REQUIRED", operation_running=False)
    other_finding = classify(other, NOW)
    assert other_finding is not None
    store.put(resolved)
    still_open = IncidentRecord.create(
        **{
            **resolved.model_dump(mode="python", exclude={"record_hash"}),
            "key": incident_key(other, other_finding),
            "task_id": "t-2",
            "code": other_finding.code,
            "state": "OPEN",
            "resolved_at": None,
            "incident": first_detection(other, other_finding, NOW),
        }
    )
    store.put(still_open)
    assert [record.state for record in store.records()] == ["OPEN", "RESOLVED"]
    path = tmp_path / "runtime" / "guanyin" / "incidents" / f"{key}.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"RESOLVED"', '"OPEN"', 1), "utf-8")
    with pytest.raises(ValueError, match="incident_record_unreadable"):
        store.get(key)


def test_attention_resolves_only_the_exact_stopped_task_version() -> None:
    """A resolved incident clears its observed blocked version but cannot clear a later one."""
    task_id = uuid4()
    execution_id = uuid4()
    stopped = _task_record(task_id=task_id, execution_id=execution_id, version=1)
    resolved = _incident(
        stopped,
        state="RESOLVED",
        key="a" * 64,
        resolved_task_record_hash=stopped.record_hash,
    )

    fact = task_attention(stopped, incidents=(resolved,))
    assert not fact.unresolved and fact.resolution == "INCIDENT_RESOLVED"
    assert fact.task_record_hash == stopped.record_hash and fact.incident_key == resolved.key

    later_failure = _task_record(
        task_id=task_id, execution_id=execution_id, version=2, salt="later-failure"
    )
    assert later_failure.record_hash != stopped.record_hash
    later_fact = task_attention(later_failure, incidents=(resolved,))
    assert later_fact.unresolved and later_fact.resolution == "STOPPED"
    assert later_fact.task_record_hash == later_failure.record_hash

    open_incident = _incident(later_failure, state="OPEN", key="b" * 64)
    with_open = task_attention(later_failure, incidents=(resolved, open_incident))
    assert with_open.unresolved and with_open.resolution == "STOPPED"
    assert with_open.incident_key == open_incident.key

    same_version_open = _incident(stopped, state="OPEN", key="e" * 64)
    resolved_and_open = task_attention(stopped, incidents=(resolved, same_version_open))
    assert resolved_and_open.unresolved and resolved_and_open.resolution == "STOPPED"
    assert resolved_and_open.incident_key == same_version_open.key


def test_legacy_resolved_incident_keeps_its_seal_but_does_not_clear_attention() -> None:
    """A pre-binding resolved incident remains readable, with no invented Task-version proof."""
    stopped = _task_record()
    legacy = _incident(stopped, state="RESOLVED", key="c" * 64)
    legacy_json = legacy.model_dump(mode="json")
    assert "resolved_task_record_hash" not in legacy_json
    restored = IncidentRecord.model_validate_json(json.dumps(legacy_json))
    assert restored == legacy and restored.record_hash == legacy.record_hash

    fact = task_attention(stopped, incidents=(restored,))
    assert fact.unresolved and fact.resolution == "STOPPED"


def test_attention_requires_a_current_explicit_link_and_a_succeeded_child() -> None:
    """Same-kind success is unrelated without the exact confirmed source link."""
    stopped = _task_record()
    unrelated = _task_record(lifecycle=TaskLifecycle.SUCCEEDED, salt="unrelated")
    stale_link = TaskRecoveryLink.create(
        source_task_id=stopped.task_id,
        source_record_hash="f" * 64,
        admission_request={"operation": "FACTOR_PLAN"},
        successor_task_id=unrelated.task_id,
        recorded_at=NOW,
    )
    for links in ((), (stale_link,)):
        fact = task_attention(
            stopped,
            successors={unrelated.task_id: unrelated},
            links=links,
        )
        assert fact.unresolved and fact.resolution == "STOPPED"

    confirmed_link = TaskRecoveryLink.create(
        source_task_id=stopped.task_id,
        source_record_hash=stopped.record_hash,
        admission_request={"operation": "FACTOR_PLAN"},
        successor_task_id=unrelated.task_id,
        recorded_at=NOW,
    )
    succeeded = task_attention(
        stopped,
        successors={unrelated.task_id: unrelated},
        links=(confirmed_link,),
    )
    assert not succeeded.unresolved and succeeded.resolution == "SUCCESSOR_SUCCEEDED"
    assert succeeded.successor_task_id == str(unrelated.task_id)
    assert succeeded.successor_task_hash == unrelated.record_hash

    pending = _task_record(lifecycle=TaskLifecycle.QUEUED, salt="pending-child")
    pending_link = TaskRecoveryLink.create(
        source_task_id=stopped.task_id,
        source_record_hash=stopped.record_hash,
        admission_request={"operation": "FACTOR_PLAN"},
        successor_task_id=pending.task_id,
        recorded_at=NOW,
    )
    waiting = task_attention(
        stopped,
        successors={pending.task_id: pending},
        links=(pending_link,),
    )
    assert waiting.unresolved and waiting.resolution == "STOPPED"
    assert waiting.successor_lifecycle == "QUEUED"
    resolved_by_incident = task_attention(
        stopped,
        successors={pending.task_id: pending},
        links=(pending_link,),
        incidents=(
            _incident(
                stopped,
                state="RESOLVED",
                key="d" * 64,
                resolved_task_record_hash=stopped.record_hash,
            ),
        ),
    )
    assert not resolved_by_incident.unresolved
    assert resolved_by_incident.resolution == "INCIDENT_RESOLVED"


def test_a_successful_current_task_resolves_as_not_stopped() -> None:
    """Task success resolves its own state without being attributed to a successor."""
    succeeded = _task_record(lifecycle=TaskLifecycle.SUCCEEDED)
    fact = task_attention(succeeded)
    assert not fact.unresolved and fact.resolution == "NOT_STOPPED"
    assert fact.task_record_hash == succeeded.record_hash


def test_attention_contract_requires_resolution_and_proof_fields() -> None:
    """A typed attention answer cannot contradict its resolution or omit its proof."""
    with pytest.raises(ValueError, match="unresolved state"):
        TaskAttentionFact(
            unresolved=False,
            resolution="STOPPED",
            task_record_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="canonical child facts"):
        TaskAttentionFact(
            unresolved=False,
            resolution="SUCCESSOR_SUCCEEDED",
            task_record_hash="a" * 64,
        )
