"""A strategy's Tasks, by the package their sealed plans name (V595).

Work planned for one strategy -- its scores, its calibration, its Portfolio and research
updates -- is read back by its Task, or by "the latest". With two strategies in a workspace the
workspace's latest Task is one of them by guess, so every such read takes a strategy's own
latest here, keyed by its package and never another's, and an unselected read is refused where
two strategies hold Tasks, offering each one's own. The owners read the Task they are handed;
their own modules, which their implementations' identities hash, are left as they are.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from uuid import UUID

from alphalattice.control.task_control.contracts import TaskRecord


def planned(task: TaskRecord, path: str) -> str | None:
    """The text a Task's sealed plan holds at `path`, read from its input.

    The envelope's identity binds its payload, so a reader picking among Tasks reads the
    package (or the session a plan advances to) without validating every plan; the Task it picks
    is read whole by its owner.

    Args:
        task: A Task whose input carries its plan under `plan`.
        path: The place in the plan, dotted (`binding.strategy_package_id`).

    Returns:
        The text there, or None where the plan holds none.
    """
    value: object = task.input.payload.get("plan")
    for part in path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value if isinstance(value, str) else None


def latest_by_strategy(tasks: Iterable[TaskRecord], path: str) -> dict[str, TaskRecord]:
    """Each strategy's latest Task among these, by the package its plan names.

    The registry lists Tasks in their admission order, so a later one replaces an earlier one;
    a Task whose plan names no package is passed over.

    Args:
        tasks: Tasks of one kind, as the registry lists them.
        path: The package's place in the kind's plan.

    Returns:
        Each package's latest Task.
    """
    latest: dict[str, TaskRecord] = {}
    for task in tasks:
        package = planned(task, path)
        if package is not None:
            latest[package] = task
    return latest


@dataclass(frozen=True, slots=True)
class StrategyRead:
    """One kind of work planned per strategy, as its readback reads it.

    Attributes:
        owner: The owner whose codes a refusal of the read carries (`research_update`).
        path: The package's place in the kind's plan.
        none: The owner's answer status when there is no Task to read.
        kind: Whether a Task is one of this kind.
    """

    owner: str
    path: str
    none: str
    kind: Callable[[TaskRecord], bool]


def strategy_task(
    read: StrategyRead,
    operation: str,
    tasks: Iterable[TaskRecord],
    lookup: Callable[[UUID], TaskRecord],
    task_id: UUID | None,
    package_id: str | None,
) -> UUID | dict[str, object]:
    """The Task a read names, never another strategy's, or the answer it gives without one.

    A named Task is its own, refused when the strategy also named is not the one its plan names.
    Without a Task, the named strategy's latest; with neither, the workspace's latest while its
    Tasks are one strategy's, as before any strategy was named. Once two strategies hold Tasks
    that read would pick one for the reader, so it is refused, offering each strategy's own by
    its package.

    Args:
        read: The kind of work read.
        operation: The readback operation, which each offered read sends.
        tasks: The registry's Tasks, in their admission order.
        lookup: The registry's read of one Task by its id.
        task_id: The Task named, if any.
        package_id: The strategy named, if any.

    Returns:
        The Task's id; or the owner's no-Task answer, or the refusal, as the read's answer.

    Raises:
        ValueError: The Task named is another strategy's; the code names it.
    """
    if task_id is not None:
        if package_id is not None and planned(lookup(task_id), read.path) != package_id:
            raise ValueError(f"{read.owner}.task_of_another_strategy:{task_id}")
        return task_id
    latest = latest_by_strategy((task for task in tasks if read.kind(task)), read.path)
    if package_id is None and len(latest) > 1:
        packages = sorted(latest)
        return {
            "status": "REFUSED",
            "failure_code": f"{read.owner}.strategy_package_required",
            "strategy_package_ids": packages,
            "next_requests": {
                f"readback:{package}": {"operation": operation, "strategy_package_id": package}
                for package in packages
            },
        }
    found = latest.get(package_id) if package_id is not None else next(iter(latest.values()), None)
    return {"status": read.none, "task_id": None} if found is None else found.task_id


__all__ = ["StrategyRead", "latest_by_strategy", "planned", "strategy_task"]
