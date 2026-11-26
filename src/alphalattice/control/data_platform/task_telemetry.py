"""Task-bound telemetry sidecars kept beside a Task's durable records by its owner.

Two owners run long Tasks whose Feature and market-data steps report work progress
through the workspace projection (`WorkspaceProgressPublisher`): first-use preparation
and the daily data update. Both keep the latest report beside the Task, bound to the
Task, its execution and the stage that produced it, and read it back for that Task only
-- nothing is inferred from the workspace-wide file, which is the latest report of
whatever ran last. This module is that one responsibility: the sidecar path, the
bounded replace, the per-delivery accounting this process observed, the last valid
observation kept when the file is later unreadable, and the typed readback. It is
telemetry, never authority: a lost write is a staler age, never a wrong stage or a
completion; the durable stage and evidence records keep their own fail-closed writes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import UUID

from alphalattice.control.observation_runtime.telemetry.progress import (
    WorkProgressProjection,
    WorkProgressUpdate,
    WorkspaceProgressPublisher,
)
from alphalattice.control.task_control.contracts import TaskExecution, TaskLifecycle, TaskRecord
from alphalattice.kernel.shared_kernel.domain.errors import AlphaLatticeError
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.persistence import DURABLE_REPLACE_DELAYS, replace_with_retry

TELEMETRY_REPLACE_DELAYS: tuple[float, ...] = (0.01, 0.02)
"""Best-effort telemetry sidecars (the Task-bound work progress and listing activity): a
reader holding the file for an instant is waited for twice, about 30 ms in all, then the
delivery is counted as failed and the work goes on. The durable stage and evidence records
keep the durable policy; this one is for files whose loss is a staler snapshot."""

WORK_PROGRESS_FIELDS = (
    "stage_id",
    "status",
    "completed_units",
    "total_units",
    "unit_name",
    "current_item",
    "counters",
    "failure_code",
    "started_at",
    "updated_at",
    "heartbeat_sequence",
)

EXECUTING = frozenset({TaskLifecycle.RUNNING, TaskLifecycle.CANCEL_REQUESTED})


class TaskTelemetry:
    """The sidecars of one owner's Tasks under `<workspace>/<root>/<task_id>/<name>.json`.

    `stages` are the owner's Task Control stage ids: a sidecar naming another stage is
    unreadable, never bound. `clock` is the product clock every age is measured by.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        root: str,
        stages: tuple[str, ...],
        clock: Callable[[], datetime],
    ) -> None:
        """Bind optional task telemetry to its workspace, stage axis and product clock.

        Args:
            workspace: Workspace root owning the local records.
            root: Relative workspace directory containing task telemetry records.
            stages: Allowed work-stage identifiers for bound telemetry.
            clock: Product clock used for optional delivery observations and age.
        """
        self.workspace, self.root, self.stages, self.clock = workspace, root, stages, clock
        # Optional deliveries, per Task and channel, as this process observed them: attempts,
        # deliveries, failures with their typed cause; and the last observation that was
        # written or read validly, kept when the file is later unreadable.
        self.deliveries: dict[tuple[str, str], dict[str, Any]] = {}
        self.last_valid: dict[tuple[str, str], dict[str, Any]] = {}

    def path(self, task_id: UUID, name: str) -> Path:
        """Resolve a named task sidecar path beneath the configured telemetry root.

        Args:
            task_id: Task UUID whose telemetry path is resolved.
            name: Sidecar record name beneath the task directory.

        Returns:
            The task-scoped JSON sidecar path.
        """
        return self.workspace / self.root / str(task_id) / f"{name}.json"

    @staticmethod
    def replace_json(
        path: Path, data: dict[str, Any], *, delays: tuple[float, ...] = DURABLE_REPLACE_DELAYS
    ) -> None:
        """Write/fsync temporary JSON and replace the destination within the durable retry bound.

        Args:
            path: Destination of the durable JSON replacement.
            data: JSON-serializable observation document.
            delays: Bounded durable-replacement retry delays.

        Raises:
            OSError: The bounded durable write or replacement fails.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            handle.write(json.dumps(data, sort_keys=True).encode())
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        # A reader holding the stage file for an instant (the page polling
        # progress.json) stopped a first-use Task on the 2026-09-11 QA build;
        # the replace outlives that within the durable bound, and this
        # writer's temporary never survives a refusal.
        try:
            replace_with_retry(temporary, path, delays=delays)
        finally:
            temporary.unlink(missing_ok=True)

    def delivery(self, task_id: UUID, channel: str = "work-progress") -> dict[str, Any]:
        """Resolve the process-local delivery counters for a task and channel.

        Args:
            task_id: Task UUID whose delivery counters are resolved.
            channel: Optional telemetry delivery channel.

        Returns:
            The mutable counter record, initialized on first observation.
        """
        return self.deliveries.setdefault(
            (str(task_id), channel),
            {
                "attempts": 0,
                "delivered": 0,
                "failures": 0,
                "last_failure": None,
                "last_delivered_at": None,
            },
        )

    def delivery_failed(self, delivery: dict[str, Any], target: str, error: Exception) -> None:
        """Record one optional delivery failure by stable type/code and observation clock.

        Args:
            delivery: Mutable process-local delivery counters to update.
            target: Stable delivery destination label.
            error: Observed exception whose type/code is retained.
        """
        delivery["failures"] += 1
        delivery["last_failure"] = {
            "target": target,
            # A typed envelope's stable code, or none: never the message or a path.
            "code": error.failure.code if isinstance(error, AlphaLatticeError) else None,
            "type": type(error).__name__[:80],
            "at": self.clock().isoformat(),
        }

    @staticmethod
    def instant(value: object) -> datetime:
        """Normalize an owner clock to aware UTC.

        An owner's instant as an aware UTC datetime: a naive instant is the stores'
        naive UTC (the maintenance rows), never local time. The readback emits every
        instant through this so a page reads one clock, whichever store wrote it.
        """
        instant = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        return instant.replace(tzinfo=UTC) if instant.tzinfo is None else instant

    def age(self, updated_at: object) -> float:
        """Seconds from `updated_at` to the product clock."""
        return max(0.0, (self.clock() - self.instant(updated_at)).total_seconds())

    @staticmethod
    def observation(binding: dict[str, Any], projection: WorkProgressProjection) -> dict[str, Any]:
        """Project approved work-progress fields with the execution and stage binding.

        Args:
            binding: Task, execution and stage binding for the projection.
            projection: Typed progress projection already validated by its owner.

        Returns:
            The bounded task progress observation without raw exception or provider payloads.
        """
        values = projection.model_dump(mode="json")
        return {
            "stage": binding["stage"],
            "execution_id": binding["execution_id"],
            **{key: values.get(key) for key in WORK_PROGRESS_FIELDS},
        }

    def bound_progress_sink(
        self,
        task: TaskRecord,
        execution: TaskExecution,
        stage: str,
        publisher: WorkspaceProgressPublisher,
    ) -> Callable[[WorkProgressUpdate], object]:
        """Create an optional progress sink bound to the task, execution and stage.

        The work owners' progress, published to the workspace projection as before and
        also kept beside this Task's stage records, bound to the Task, its execution and
        this stage. `work_progress` shows the kept update for the selected Task only under
        that binding.
        """
        path = self.path(task.task_id, "work-progress")
        binding = {
            "task_id": str(task.task_id),
            "input_hash": task.input.input_hash,
            "execution_id": str(execution.execution_id),
            "stage": stage,
        }
        delivery = self.delivery(task.task_id)
        key = (str(task.task_id), "work-progress")

        def sink(update: WorkProgressUpdate) -> object:
            # Optional telemetry: neither the workspace projection nor the Task-bound
            # copy may stop the work it describes. A failed delivery is counted with
            # its typed cause (never the message) and the stage result is untouched.
            delivery["attempts"] += 1
            projection: WorkProgressProjection | None = None
            try:
                projection = publisher.publish(update)
            except (OSError, ValueError, RuntimeError) as error:
                self.delivery_failed(delivery, "WORKSPACE_PROJECTION", error)
                return None
            data = {**binding, "projection": projection.model_dump(mode="json")}
            data["content_hash"] = canonical_hash(data)
            try:
                self.replace_json(path, data, delays=TELEMETRY_REPLACE_DELAYS)
            except (OSError, ValueError, RuntimeError) as error:
                self.delivery_failed(delivery, "TASK_SIDECAR", error)
            else:
                delivery["delivered"] += 1
                delivery["last_delivered_at"] = self.clock().isoformat()
                self.last_valid[key] = self.observation(binding, projection)
            return projection

        return sink

    def work_progress(self, task: TaskRecord) -> dict[str, Any] | None:
        """Read validated task progress with binding state and a retained valid fallback.

        The kept work progress of one Task: BOUND while it names the Task's current
        execution and stage and the Task is executing, otherwise NOT_CURRENT (an earlier
        stage or execution, retained as the last observation). The age is measured by
        the product clock at this read. A file that cannot be found out about, read,
        parsed or validated as the typed projection (the update owner's count invariants,
        timezone-aware instants) is UNREADABLE, another Task's is UNBOUND; neither is
        promoted, and the last valid observation this process wrote or read is kept
        beside an UNREADABLE answer -- a malformed file never replaces it. `delivery`
        is what this process observed of the optional deliveries (absent after a
        restart).
        """
        key = (str(task.task_id), "work-progress")
        delivery = self.deliveries.get(key)
        report: dict[str, Any] = {} if delivery is None else {"delivery": dict(delivery)}

        def unreadable(cause: str) -> dict[str, Any]:
            retained = self.last_valid.get(key)
            values = {"availability": "UNREADABLE", "cause": cause, **report}
            if retained is not None:
                values["last_valid"] = {**retained, "age_seconds": self.age(retained["updated_at"])}
            return values

        path = self.path(task.task_id, "work-progress")
        try:
            present = path.exists()
        except OSError as error:
            return unreadable(type(error).__name__)
        if not present:
            return {"availability": "NONE", **report} if report else None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            return unreadable(type(error).__name__)
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("projection"), dict)
            or data.get("content_hash")
            != canonical_hash({k: v for k, v in data.items() if k != "content_hash"})
        ):
            return unreadable("content_hash")
        try:
            projection = WorkProgressProjection.model_validate(data["projection"])
        except ValueError:
            return unreadable("projection_invalid")
        binding = {key: data.get(key) for key in ("task_id", "input_hash", "execution_id", "stage")}
        if not all(isinstance(value, str) and value for value in binding.values()):
            return unreadable("binding_invalid")
        if binding["task_id"] != str(task.task_id):
            return {"availability": "UNBOUND", **report}
        if binding["input_hash"] != task.input.input_hash:
            return {"availability": "UNBOUND", **report}
        if binding["stage"] not in self.stages:
            return unreadable("stage_unknown")
        observation = self.observation(binding, projection)
        # The age is measured before the observation becomes the retained fallback: only a
        # projection whose counts and instant the typed boundary accepted, and whose age the
        # product clock could measure, is ever kept as "last valid".
        age = self.age(projection.updated_at)
        self.last_valid[key] = observation
        current = (
            binding["execution_id"] == str(task.latest_execution_id)
            and binding["stage"] == task.active_work_item_id
            and task.lifecycle in EXECUTING
        )
        return {
            "availability": "BOUND" if current else "NOT_CURRENT",
            **observation,
            "age_seconds": age,
            **report,
        }


__all__ = ["EXECUTING", "TELEMETRY_REPLACE_DELAYS", "WORK_PROGRESS_FIELDS", "TaskTelemetry"]
