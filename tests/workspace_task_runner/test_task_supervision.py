"""Guanyin's Supervisor and Recovery Center over Task Control (GY2, V83)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from alphalattice.control.guanyin.tasks.supervision import (
    STAGE_STALLED_AFTER,
    IncidentRecord,
    IncidentStore,
    OfferedRemedy,
    TaskFacts,
    classify,
    first_detection,
    incident_key,
)

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
    # regression (V607): a recovery the dispatcher holds waits its turn, as a queued Task does;
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
