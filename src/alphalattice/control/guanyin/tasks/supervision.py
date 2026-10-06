"""Guanyin's Supervisor rules and incident records (GY2, V83).

The Supervisor tells a running Task from one whose liveness went stale, whose work stalled,
whose runner the Host cannot see, that waits for its recovery or that no command drives, by
rule and never by a model's guess. A valid stop (a scientific or an owner's refusal, a review a
person decides, a provider's deferral) is not an incident: it is the Task's own state.

Each finding is kept as an incident record: the first detection (a `RuntimeIncident`), when it
was last seen, the remedies the Host offered for it, every recovery attempt made through them,
and when it resolved. A record is evidence plus appended decisions; its key is stable across
detections, so a wake is sent once for it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.observation_runtime.guardian import RuntimeIncident
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model_from_dump

INCIDENTS_DIRECTORY = Path("runtime") / "guanyin" / "incidents"
"""Under a workspace: one record per incident, named by its key."""

STAGE_STALLED_AFTER = timedelta(hours=6)
"""A stage running this long while its heartbeat is fresh is stalled work: a code-owned policy
until incidents recorded in use or by AX measure the stages (the design's thresholds rule)."""

IncidentCode = Literal[
    "task_runtime.liveness_stale",
    "task_runtime.host_visibility_missing",
    "task_runtime.work_stalled",
    "task_runtime.recovery_required",
    "task_runtime.parked",
]

_DETAIL: dict[str, str] = {
    "task_runtime.liveness_stale": (
        "The Task is running but its heartbeat has not been seen recently; it may have stopped "
        "without the Host knowing. Its verified stages are kept."
    ),
    "task_runtime.host_visibility_missing": (
        "The Task is running but the Host cannot read its runner's heartbeat, so whether it is "
        "alive is not observed. Its verified stages are kept."
    ),
    "task_runtime.work_stalled": (
        "The Task's current stage has run longer than the Host's policy allows while its "
        "heartbeat stays fresh: its work may be stuck. Its verified stages are kept."
    ),
    "task_runtime.recovery_required": (
        "The Task stopped before it ended and waits for a recovery; its verified stages are "
        "kept and only unverified work would run again."
    ),
    "task_runtime.parked": (
        "The Task is queued but no command of this Host drives it any more; a recovery hands "
        "it back to the worker."
    ),
}


@dataclass(frozen=True, slots=True)
class TaskFacts:
    """What the Host observed of one unfinished Task, as the Supervisor's rules read it."""

    task_id: str
    task_kind: str
    execution_id: str | None
    lifecycle: str
    operation_running: bool
    liveness: str
    """`OBSERVED`, `NOT_RECENT`, `NOT_OBSERVED` or `NOT_APPLICABLE` (the recovery view's)."""
    telemetry: str
    current_stage: str | None
    stage_started_at: datetime | None


@dataclass(frozen=True, slots=True)
class Finding:
    """One rule's verdict on one Task."""

    code: IncidentCode
    stage_id: str | None


def classify(facts: TaskFacts, now: datetime) -> Finding | None:
    """The one finding the rules make of a Task's facts, or none for a running or valid stop."""
    if facts.lifecycle == "RECOVERY_REQUIRED":
        # A recovery the Host's dispatcher holds waits its turn, as a queued Task does, and is
        # owed no one's decision: an incident on it woke a follower while it was starting (V607).
        if facts.operation_running:
            return None
        return Finding("task_runtime.recovery_required", facts.current_stage)
    if facts.lifecycle == "QUEUED":
        return None if facts.operation_running else Finding("task_runtime.parked", None)
    if facts.lifecycle != "RUNNING":
        return None  # a deferral, a review or a cancellation in progress is the Task's own state
    if facts.liveness == "NOT_RECENT":
        return Finding("task_runtime.liveness_stale", facts.current_stage)
    if facts.telemetry in {"UNREADABLE", "UNBOUND"}:
        return Finding("task_runtime.host_visibility_missing", facts.current_stage)
    if facts.stage_started_at is not None and now - facts.stage_started_at >= STAGE_STALLED_AFTER:
        return Finding("task_runtime.work_stalled", facts.current_stage)
    return None


def incident_key(facts: TaskFacts, finding: Finding) -> str:
    """The same incident while its Task, execution, rule and stage stay the same."""
    return str(
        canonical_hash(
            {
                "task_id": facts.task_id,
                "execution_id": facts.execution_id,
                "code": finding.code,
                "stage_id": finding.stage_id,
            }
        )
    )


class _Record(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class OfferedRemedy(_Record):
    """One remedy the Host offered for an incident: an existing operation, never a new power."""

    action: Literal["CANCEL", "RECOVER", "REPLAN"]
    operation: str = Field(min_length=1, max_length=64)
    available: bool
    reason: str = Field(min_length=1, max_length=300)


class RecoveryAttempt(_Record):
    """One remedy performed through the Recovery Center, with what it found afterwards."""

    ordinal: int = Field(ge=1)
    action: Literal["CANCEL", "RECOVER", "REPLAN"]
    selected_by: Literal["USER_COMMAND", "AGENT_PROPOSAL"]
    expected_task_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_at: datetime
    disposition: str = Field(min_length=1, max_length=120)
    lifecycle_after: str | None = None
    verified_stages_after: int | None = Field(default=None, ge=0)
    failure_code: str | None = Field(default=None, max_length=240)


class IncidentRecord(_Record):
    """An incident: its first detection, when it was last seen, its remedies and attempts."""

    schema_version: Literal[1] = 1
    key: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str
    task_kind: str
    execution_id: str | None
    code: IncidentCode
    incident: RuntimeIncident
    state: Literal["OPEN", "RESOLVED"]
    last_seen_at: datetime
    resolved_at: datetime | None = None
    remedies: tuple[OfferedRemedy, ...]
    attempts: tuple[RecoveryAttempt, ...] = ()
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal a record's values under their hash."""
        return seal_model_from_dump(cls, values, field="record_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _sealed(self) -> Self:
        if self.record_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"record_hash"})
        ):
            raise ValueError("guanyin.incident_record_tampered")
        return self

    def changed(self, **values: object) -> IncidentRecord:
        """The same incident with its lifecycle moved on, sealed again."""
        kept = {name: getattr(self, name) for name in type(self).model_fields}
        kept.pop("record_hash")
        return IncidentRecord.create(**{**kept, **values})


def first_detection(facts: TaskFacts, finding: Finding, now: datetime) -> RuntimeIncident:
    """The incident as the rule first saw it."""
    values = {
        "incident_code": finding.code,
        "detected_at": now,
        "stage_id": finding.stage_id,
        "user_safe_detail": _DETAIL[finding.code],
    }
    identity = RuntimeIncident.model_construct(**values, shadow_only=True).model_dump(
        mode="json", exclude={"incident_id"}
    )
    return RuntimeIncident(incident_id=canonical_hash(identity), **values)


class IncidentStore:
    """A workspace's incident records, one file each under its key, written whole."""

    def __init__(self, workspace: Path) -> None:
        """Locate the records under the workspace's `runtime/`."""
        self.root = workspace / INCIDENTS_DIRECTORY

    def get(self, key: str) -> IncidentRecord | None:
        """One record, or none; a record that does not read as sealed is refused by name."""
        path = self.root / f"{key}.json"
        if not path.is_file():
            return None
        try:
            return cast(IncidentRecord, IncidentRecord.model_validate_json(path.read_bytes()))
        except (OSError, ValueError) as error:
            raise ValueError("guanyin.incident_record_unreadable") from error

    def records(self) -> tuple[IncidentRecord, ...]:
        """Every record that reads, open ones first, newest first within each."""
        if not self.root.is_dir():
            return ()
        found = []
        for path in self.root.glob("*.json"):
            try:
                found.append(IncidentRecord.model_validate_json(path.read_bytes()))
            except (OSError, ValueError):
                continue
        return tuple(
            sorted(
                found,
                key=lambda record: (
                    record.state != "OPEN",
                    -record.incident.detected_at.timestamp(),
                ),
            )
        )

    def put(self, record: IncidentRecord) -> None:
        """Write a record whole, atomically."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{record.key}.json"
        staged = path.with_name(f"{path.name}.partial")
        staged.write_text(json.dumps(record.model_dump(mode="json"), sort_keys=True), "utf-8")
        os.replace(staged, path)
