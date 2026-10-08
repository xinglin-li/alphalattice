"""How long a Task queued, ran, and spent in each stage, from Task Control's own records (CLI-4).

A Task's record carries when it was admitted, when a worker started it and when it last
changed; each work item carries when its stage started and last changed. The seconds are read
from those timestamps, so a Task's timing is the same fact for the Web, the CLI and an Agent.

Beside them, the runner keeps what each stage's execution and verification spent its time on:
the spans its owners marked, added up by category, and the process's CPU and read/write bytes
(A4, `shared_kernel.spans`). They are kept per Task in the runtime folder's
`execution/spans/`, one JSON line a phase, and read back with their stage. They are execution
metadata: no verification, identity or decision reads them, and a Task without them (older,
or its line unreadable) reads exactly as before.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from alphalattice.control.task_control.contracts import (
    TaskLifecycle,
    TaskRecord,
    WorkItemLifecycle,
    WorkItemState,
)

_LOG = logging.getLogger(__name__)

_ENDED = frozenset(
    {
        TaskLifecycle.SUCCEEDED,
        TaskLifecycle.BLOCKED,
        TaskLifecycle.CANCELLED,
        TaskLifecycle.RECOVERY_REQUIRED,
        TaskLifecycle.REVIEW_PENDING,
        TaskLifecycle.DEFERRED,
    }
)
"""The states whose clock stops at their last change: ended, or waiting on someone or on a
retry time (V520: a deferred Task counted its wait for the provider as running)."""
_STAGE_ENDED = frozenset(
    {WorkItemLifecycle.VERIFIED, WorkItemLifecycle.BLOCKED, WorkItemLifecycle.CANCELLED}
)


def stage_spans_path(runtime_path: Path, task_id: UUID) -> Path:
    """Where a Task's stage spans are kept: `execution/spans/` in its runner's runtime folder."""
    return runtime_path.parent / "execution" / "spans" / f"{task_id}.jsonl"


def record_stage_spans(
    runtime_path: Path,
    *,
    task_id: UUID,
    execution_id: UUID,
    stage_id: str,
    phase: str,
    readout: Mapping[str, Any],
) -> None:
    """Append one phase's span readout; a write the filesystem refuses is logged, never raised.

    The runner is the Task's one writer while it runs, so one Task's lines never interleave.
    """
    line = {
        "task_id": str(task_id),
        "execution_id": str(execution_id),
        "stage_id": stage_id,
        "phase": phase,
        **readout,
    }
    path = stage_spans_path(runtime_path, task_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
    except OSError:
        _LOG.warning("stage spans for %s %s were not kept", task_id, stage_id, exc_info=True)


def read_stage_spans(runtime_path: Path, task_id: UUID) -> tuple[dict[str, Any], ...]:
    """A Task's kept phase readouts in the order they were written; none when absent.

    A line that is not a readout (a torn last write) is skipped: spans explain time and
    decide nothing, so a missing one is only less explanation.
    """
    path = stage_spans_path(runtime_path, task_id)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ()
    kept: list[dict[str, Any]] = []
    for raw in text.splitlines():
        try:
            value = json.loads(raw)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("task_id") == str(task_id):
            kept.append(value)
    return tuple(kept)


def _seconds(start: datetime, end: datetime) -> float:
    return round(max(0.0, (end - start).total_seconds()), 3)


def task_timing(
    record: TaskRecord,
    items: Sequence[WorkItemState],
    *,
    now: datetime,
    spans: Sequence[Mapping[str, Any]] = (),
) -> dict[str, object]:
    """Queued and running seconds, and each stage's, as of ``now`` for a Task still running.

    ``spans`` are the Task's kept phase readouts (`read_stage_spans`); each stage lists its own,
    oldest first, under ``phases``, with the execution each belongs to.
    """
    end = record.updated_at if record.lifecycle in _ENDED else now
    started = record.started_at
    phases: dict[str, list[dict[str, Any]]] = {}
    for value in spans:
        stage = value.get("stage_id")
        if isinstance(stage, str):
            phases.setdefault(stage, []).append(
                {key: item for key, item in value.items() if key not in {"task_id", "stage_id"}}
            )
    return {
        "admitted_at": record.admitted_at.isoformat(),
        "started_at": None if started is None else started.isoformat(),
        "queued_seconds": _seconds(record.admitted_at, started or end),
        "running_seconds": None if started is None else _seconds(started, end),
        "stages": [
            {
                "stage_id": item.stage_id,
                "lifecycle": item.lifecycle.value,
                "started_at": None if item.started_at is None else item.started_at.isoformat(),
                "seconds": None
                if item.started_at is None
                else _seconds(
                    item.started_at, item.updated_at if item.lifecycle in _STAGE_ENDED else now
                ),
                "phases": phases.get(item.stage_id, []),
            }
            for item in items
        ],
    }


__all__ = ["read_stage_spans", "record_stage_spans", "stage_spans_path", "task_timing"]
