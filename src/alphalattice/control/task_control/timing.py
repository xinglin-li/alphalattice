"""How long a Task queued, ran, and spent in each stage, from Task Control's own records (CLI-4).

A Task's record carries when it was admitted, when a worker started it and when it last
changed; each work item carries when its stage started and last changed. The spans are read
from those timestamps and nothing else, so a Task's timing is the same fact for the Web, the
CLI and an Agent, and costs no new record.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from alphalattice.control.task_control.contracts import (
    TaskLifecycle,
    TaskRecord,
    WorkItemLifecycle,
    WorkItemState,
)

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


def _seconds(start: datetime, end: datetime) -> float:
    return round(max(0.0, (end - start).total_seconds()), 3)


def task_timing(
    record: TaskRecord, items: Sequence[WorkItemState], *, now: datetime
) -> dict[str, object]:
    """Queued and running seconds, and each stage's, as of ``now`` for a Task still running."""
    end = record.updated_at if record.lifecycle in _ENDED else now
    started = record.started_at
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
            }
            for item in items
        ],
    }


__all__ = ["task_timing"]
