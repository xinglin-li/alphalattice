"""Keep the catalog's advisory health results beside the Daily that measured them."""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from alphalattice.control.task_control.contracts import TaskRecord
from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from alphalattice.kernel.data_layer.market_health import market_health, unavailable_market_health


def record_daily_health(
    registry: DuckDbTaskControlRegistry,
    task: TaskRecord,
    session: date,
    observed_at: datetime,
) -> None:
    """Measure the current catalog once for a Daily that verified its data inputs."""
    if registry.daily_health(task.task_id) is None:
        registry.record_daily_health(
            task,
            {
                "task_id": str(task.task_id),
                "target_session": session.isoformat(),
                "checks": market_health(session),
            },
            observed_at=observed_at,
        )


def daily_health_body(
    registry: DuckDbTaskControlRegistry, task_id: UUID | None = None
) -> dict[str, object]:
    """Show the saved observation, or show that its five results are unavailable."""
    return registry.daily_health(task_id) or {
        "task_id": str(task_id) if task_id is not None else None,
        "target_session": None,
        "checks": unavailable_market_health(),
    }
