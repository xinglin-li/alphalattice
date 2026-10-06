"""How many Tasks may wait behind the running one: the Task queue's places (V100, LAWS PA2).

An execution setting, the operator's: it paces work and decides no number, so it enters no
identity. A person sets it in the Local Web's settings and an agent by the CLI, beside the CPU
budget in the workspace's execution settings (`runtime/execution/`); `auto` computes it from
the machine. Task Control reads it each time it admits a Task.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

QUEUE_SETTING_DIRECTORY = "execution"
"""Under the workspace's `runtime/`, with the CPU budget: a setting, not evidence."""
QUEUE_SETTING_FILE = "task-queue.json"
MAXIMUM_TASKS_WAITING = 64
PROCESSORS_PER_WAITING_PLACE = 4
"""`auto` gives one waiting place per four processors, at least one: a machine that drains a
backlog faster holds a longer one for the same wait."""

Chooser = Literal["DEFAULT", "HUMAN", "INSTALLED_AGENT", "EXTERNAL_AUTOMATION"]
TasksWaiting = Literal["auto"] | Annotated[int, Field(ge=1, le=MAXIMUM_TASKS_WAITING, strict=True)]


class TaskQueueSetting(BaseModel):  # type: ignore[misc]
    """The operator's Task queue: `auto`, or how many Tasks may wait behind the running one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    tasks_waiting: TasksWaiting = "auto"
    chosen_by: Chooser = "DEFAULT"
    chosen_at: datetime | None = None


def queue_setting_path(runtime_root: Path) -> Path:
    """Where a workspace's `runtime/` keeps its Task queue setting."""
    return runtime_root / QUEUE_SETTING_DIRECTORY / QUEUE_SETTING_FILE


def read_queue_setting(runtime_root: Path) -> TaskQueueSetting:
    """The setting as chosen, `auto` before any choice; an unreadable file is refused by name."""
    path = queue_setting_path(runtime_root)
    if not path.is_file():
        return TaskQueueSetting()
    try:
        return cast(TaskQueueSetting, TaskQueueSetting.model_validate_json(path.read_bytes()))
    except (OSError, ValueError) as error:
        raise ValueError("task_control.queue_setting_unreadable") from error


def parse_tasks_waiting(value: object) -> Literal["auto"] | int:
    """`auto` or a whole number of Tasks from 1; anything else refused by name."""
    if value == "auto":
        return "auto"
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if type(value) is not int or not 1 <= value <= MAXIMUM_TASKS_WAITING:
        raise ValueError("task_control.tasks_waiting_invalid")
    return value


def write_queue_setting(
    runtime_root: Path, value: object, *, chosen_by: Chooser, chosen_at: datetime
) -> TaskQueueSetting:
    """Validate and atomically write the operator's Task queue setting."""
    setting = TaskQueueSetting(
        tasks_waiting=parse_tasks_waiting(value), chosen_by=chosen_by, chosen_at=chosen_at
    )
    path = queue_setting_path(runtime_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".partial")
    staged.write_text(setting.model_dump_json(), encoding="utf-8")
    os.replace(staged, path)
    return setting


def waiting_places(setting: TaskQueueSetting) -> tuple[int, str]:
    """How many Tasks may wait now, and why."""
    if setting.tasks_waiting != "auto":
        return setting.tasks_waiting, f"set to {setting.tasks_waiting}"
    affinity = getattr(os, "sched_getaffinity", None)
    processors = len(affinity(0)) if affinity is not None else os.cpu_count() or 1
    places = max(1, processors // PROCESSORS_PER_WAITING_PLACE)
    return places, (
        f"auto: one waiting place per {PROCESSORS_PER_WAITING_PLACE} of the {processors} processors"
    )
