"""Each admitted Task's frozen request: what it was admitted to do (LAWS DA2).

Since 0.2.8 a Task's frozen request is the text of its admission on the log (`task.requested`),
kept forever with the record it was admitted under. A 0.2.7 workspace kept them here instead, as
content-addressed files beside its Task Control store; they are read once, when the workspace's
Tasks move onto the log, and set aside with that store after the log is verified to hold them.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import ResearchGoal, ResearchPlan, TaskInputEnvelope

REQUESTS_DIRECTORY = "task-requests"
"""Under a 0.2.7 workspace's `runtime/`, beside its Task Control store."""


class SubmittingAgent(BaseModel):  # type: ignore[misc]
    """The agent session that submitted a Task, as its client declared it: provenance only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vendor: str | None = Field(default=None, max_length=64)
    session: str | None = Field(default=None, max_length=200)
    goal_id: str | None = Field(default=None, max_length=64)


class TaskAdmissionRequest(BaseModel):  # type: ignore[misc]
    """What one Task was admitted to do, frozen at its admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    task_id: UUID
    task_kind: str
    input: TaskInputEnvelope
    goal: ResearchGoal
    plan: ResearchPlan
    admitted_at: datetime
    admission_sequence: int = Field(ge=1)
    submitted_by: SubmittingAgent | None = None


def requests_root(runtime_root: Path) -> Path:
    """Where a workspace's `runtime/` keeps its Tasks' frozen requests."""
    return runtime_root / REQUESTS_DIRECTORY


def read_frozen(runtime_root: Path) -> tuple[tuple[TaskAdmissionRequest, str], ...]:
    """Every frozen request with its exact text, in admission order; one that does not read is
    refused by name."""
    root = requests_root(runtime_root)
    if not root.is_dir():
        return ()
    frozen: list[tuple[TaskAdmissionRequest, str]] = []
    for path in sorted(root.glob("*.json")):
        try:
            text = path.read_bytes().decode("utf-8")
            request = cast(TaskAdmissionRequest, TaskAdmissionRequest.model_validate_json(text))
        except (OSError, ValueError) as error:
            raise ValueError(f"task_control.admission_request_unreadable:{path.name}") from error
        if path.stem != canonical_hash(request.model_dump(mode="json")):
            raise ValueError(f"task_control.admission_request_unreadable:{path.name}")
        frozen.append((request, text))
    return tuple(sorted(frozen, key=lambda pair: pair[0].admission_sequence))
