"""What a domain adapter carries from one stage of a Task's execution to its later stages.

An adapter may keep, in its own process, what one execution of a Task already verified, so the
execution's later stages do not verify it again. A new execution of the Task, as a recovery
starts after an interruption, carries nothing from the previous one: it verifies its stored
stages again.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from alphalattice.control.task_control.contracts import TaskExecution


def carried_by[T](
    previous: dict[UUID, tuple[UUID, T]], execution: TaskExecution, fresh: Callable[[], T]
) -> T:
    """This execution's carried value, keyed by its Task; another execution's is replaced."""
    held = previous.get(execution.task_id)
    if held is None or held[0] != execution.execution_id:
        held = previous[execution.task_id] = (execution.execution_id, fresh())
    return held[1]


__all__ = ["carried_by"]
