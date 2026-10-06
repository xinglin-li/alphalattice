"""Deterministic model-facing rendering for safe task projections."""

from __future__ import annotations

from .contracts import TaskSafeProjection


def render_task_projection(projection: TaskSafeProjection) -> str:
    """Render only safe projection fields; no machine artifact is decoded here."""
    values = [
        "# Research task status",
        "",
        f"- Task: `{projection.task_id}`",
        f"- Kind: `{projection.task_kind}`",
        f"- Status: `{projection.lifecycle.value}`",
        f"- Goal: {projection.goal_summary}",
        f"- Current stage: `{projection.current_stage or 'NONE'}`",
        (f"- Verified stages: {projection.verified_stage_count}/{projection.total_stage_count}"),
        f"- Last activity: `{projection.last_activity_at.isoformat()}`",
        f"- Cancellation available: {'YES' if projection.cancel_available else 'NO'}",
        f"- Cancellation pending: {'YES' if projection.cancel_pending else 'NO'}",
    ]
    if projection.latest_failure_code is not None:
        values.append(f"- Failure: `{projection.latest_failure_code}`")
    if projection.queued_next_task_id is not None:
        values.append(f"- Next queued task: `{projection.queued_next_task_id}`")
    return "\n".join(line.rstrip() for line in values).strip() + "\n"


__all__ = ["render_task_projection"]
