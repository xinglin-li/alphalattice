"""Lightweight safe telemetry projections for long-running workspace progress."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Literal, cast

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.workspace_runtime.content_store import replace_shared_file
from alphalattice.kernel.shared_kernel.identity import canonical_hash

WorkProgressStatus = Literal["RUNNING", "SUCCEEDED", "FAILED", "REUSED_EXACT"]


def validate_work_counts(
    *,
    status: WorkProgressStatus,
    completed_units: int,
    total_units: int,
    failure_code: str | None,
    counters: dict[str, int],
) -> None:
    """Validate the count invariants shared by updates and readback projections.

    Args:
        status: Current operation state.
        completed_units: Finished units reported by the owner.
        total_units: Positive total for the operation.
        failure_code: Required code when the operation failed.
        counters: Additional nonnegative progress counts.

    Raises:
        ValueError: A count, terminal state or failure code is inconsistent.

    """
    if completed_units < 0 or total_units < 1:
        raise ValueError("work units are a non-negative count of a positive total")
    if completed_units > total_units:
        raise ValueError("completed work units exceed total work units")
    if status in {"SUCCEEDED", "REUSED_EXACT"} and completed_units != total_units:
        raise ValueError("terminal success progress must be complete")
    if status == "FAILED" and failure_code is None:
        raise ValueError("failed progress requires a failure code")
    if any(value < 0 for value in counters.values()):
        raise ValueError("progress counters cannot be negative")


class WorkProgressUpdate(BaseModel):  # type: ignore[misc]
    """One Host-owned progress observation; never completion authority."""

    model_config = ConfigDict(frozen=True)

    operation_id: str = Field(min_length=1, max_length=128)
    stage_id: str = Field(min_length=1, max_length=120)
    status: WorkProgressStatus
    completed_units: int = Field(ge=0)
    total_units: int = Field(ge=1)
    unit_name: str = Field(min_length=1, max_length=40)
    current_item: str | None = Field(default=None, max_length=160)
    counters: dict[str, int] = Field(default_factory=dict)
    failure_code: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_progress(self) -> WorkProgressUpdate:
        """Require this update to satisfy the owner's count invariants."""
        validate_work_counts(
            status=self.status,
            completed_units=self.completed_units,
            total_units=self.total_units,
            failure_code=self.failure_code,
            counters=self.counters,
        )
        return self


class WorkProgressProjection(BaseModel):  # type: ignore[misc]
    """Model/UI-safe latest progress, atomically replaced on every heartbeat.

    A projection read back from a file is validated as strictly as the update it was
    published from: the counts keep the update owner's invariants before any ratio is
    derived from them, and both instants are timezone-aware so an age can be measured
    against the product clock. The identity (the hash over the JSON form) is unchanged
    by these checks: every projection a publisher wrote still validates as itself.
    """

    model_config = ConfigDict(frozen=True)

    operation_id: str
    stage_id: str
    status: WorkProgressStatus
    completed_units: int = Field(ge=0)
    total_units: int = Field(ge=1)
    unit_name: str
    percent_complete: float = Field(ge=0.0, le=100.0)
    current_item: str | None = None
    counters: dict[str, int] = Field(default_factory=dict)
    failure_code: str | None = None
    started_at: AwareDatetime
    updated_at: AwareDatetime
    heartbeat_sequence: int = Field(ge=1)
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> WorkProgressProjection:
        """Verify counts, percentage and the readback projection hash."""
        validate_work_counts(
            status=self.status,
            completed_units=self.completed_units,
            total_units=self.total_units,
            failure_code=self.failure_code,
            counters=self.counters,
        )
        expected_percent = round(100.0 * self.completed_units / self.total_units, 3)
        if self.percent_complete != expected_percent:
            raise ValueError("work progress percentage is not derived from work units")
        identity = self.model_dump(mode="json", exclude={"projection_hash"})
        if self.projection_hash != canonical_hash(identity):
            raise ValueError("work progress projection hash is invalid")
        return self


class WorkspaceProgressPublisher:
    """Atomically publish the latest safe progress without touching DuckDB."""

    def __init__(self, artifact_root: Path) -> None:
        """Choose the workspace's atomically replaced progress artifact path."""
        self.path = artifact_root.resolve() / "run-monitor" / "work-progress.json"
        self._lock = Lock()
        self._sequence = 0
        self._operation_id: str | None = None
        self._stage_id: str | None = None
        self._started_at: datetime | None = None

    def publish(self, update: WorkProgressUpdate) -> WorkProgressProjection:
        """Publish an incremented, content-hashed progress heartbeat.

        Args:
            update: Host-owned safe progress observation.

        Returns:
            The projection written atomically to the workspace.

        """
        now = datetime.now(UTC)
        with self._lock:
            if update.operation_id != self._operation_id or update.stage_id != self._stage_id:
                self._operation_id = update.operation_id
                self._stage_id = update.stage_id
                self._started_at = now
                self._sequence = 0
            self._sequence += 1
            assert self._started_at is not None
            values = {
                **update.model_dump(mode="python"),
                "percent_complete": round(100.0 * update.completed_units / update.total_units, 3),
                "started_at": self._started_at,
                "updated_at": now,
                "heartbeat_sequence": self._sequence,
            }
            identity = WorkProgressProjection.model_construct(
                **values, projection_hash=""
            ).model_dump(mode="json", exclude={"projection_hash"})
            projection = WorkProgressProjection(**values, projection_hash=canonical_hash(identity))
            content = json.dumps(
                projection.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            staged = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            staged.write_bytes(content)
            replace_shared_file(staged, self.path)
            return projection

    def read(self) -> WorkProgressProjection | None:
        """Read and validate the latest progress artifact, if present."""
        with self._lock:
            if not self.path.exists():
                return None
            return cast(
                WorkProgressProjection,
                WorkProgressProjection.model_validate_json(self.path.read_bytes()),
            )


__all__ = [
    "WorkProgressProjection",
    "WorkProgressStatus",
    "WorkProgressUpdate",
    "WorkspaceProgressPublisher",
    "validate_work_counts",
]
