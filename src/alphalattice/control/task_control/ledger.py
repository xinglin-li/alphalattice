"""Each admitted Task's frozen request, kept beside the Task Control store (V181, LAWS DA2).

The store's rows are the Task's lifecycle; what the Task was admitted to do -- its input, goal,
plan, queue place and who submitted it -- is also written here, content-addressed, before its
row. A store that is lost or replaced lists its Tasks again from these files: what each was
admitted to do stays readable, and its progress is not claimed.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import ResearchGoal, ResearchPlan, TaskInputEnvelope

REQUESTS_DIRECTORY = "task-requests"
"""Under the workspace's `runtime/`, beside the Task Control store."""


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


def write_request(runtime_root: Path, request: TaskAdmissionRequest) -> Path:
    """Write one frozen request under its content hash; the same request is the same file."""
    document = request.model_dump(mode="json")
    root = requests_root(runtime_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{canonical_hash(document)}.json"
    if not path.is_file():
        staged = path.with_name(f"{path.name}.partial")
        staged.write_text(request.model_dump_json(), encoding="utf-8")
        os.replace(staged, path)
    return path


def read_requests(runtime_root: Path) -> tuple[TaskAdmissionRequest, ...]:
    """Every frozen request, in admission order; one that does not read is refused by name."""
    root = requests_root(runtime_root)
    if not root.is_dir():
        return ()
    requests: list[TaskAdmissionRequest] = []
    for path in sorted(root.glob("*.json")):
        try:
            request = cast(
                TaskAdmissionRequest, TaskAdmissionRequest.model_validate_json(path.read_bytes())
            )
        except (OSError, ValueError) as error:
            raise ValueError(f"task_control.admission_request_unreadable:{path.name}") from error
        if path.stem != canonical_hash(request.model_dump(mode="json")):
            raise ValueError(f"task_control.admission_request_unreadable:{path.name}")
        requests.append(request)
    return tuple(sorted(requests, key=lambda request: request.admission_sequence))
