"""What a book's preparation reports while its units run: telemetry, never authority.

The stage receipts say what is done; this says how far a running stage is, in the owner's
own unit: the filings a unit's acquisition fetched of those its plan counts, the documents
canonicalized of those acquired, the chunks embedded of those the index cut, and, for the
book, the units whose selection finished. It is kept in the Host process for the reads that
ask while the run continues (`coverage_progress`) and handed to the Host's progress
publisher as each count moves; a restarted Host holds none and reads the stage receipts, as
it always did. Nothing here decides a unit's state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import Lock
from uuid import UUID

from alphalattice.control.observation_runtime.telemetry.progress import (
    WorkProgressUpdate,
    WorkspaceProgressPublisher,
)

ACQUIRE = "acquire_source_evidence"
CANONICALIZE = "canonicalize_documents"
BUILD = "build_retrieval_generation"
SELECT = "select_evidence_spans"
STAGE_UNIT_NAMES: Mapping[str, str] = {
    ACQUIRE: "filings",
    CANONICALIZE: "documents",
    BUILD: "chunks",
    SELECT: "units",
}
"""The stages that report a count, each in the owner's own unit."""

PUBLISH_INTERVAL_SECONDS = 1.0
"""A running stage is handed to the publisher at most once a second, and at its start and end."""


@dataclass(frozen=True, slots=True)
class StageWork:
    """One unit's stage as it last reported: `completed` of `total` `unit_name`."""

    task_id: UUID
    unit_id: str | None
    stage: str
    unit_name: str
    completed: int
    total: int
    running: bool
    updated_at: datetime


class StageCounter:
    """The count of one running stage, moved by the stage's own callbacks."""

    def __init__(self, owner: PreparationProgress | None, work: StageWork) -> None:
        """Track one stage and send updates to its progress owner."""
        self._owner = owner
        self._work = work

    def expect(self, total: int) -> None:
        """Record the stage's expected work count."""
        self._move(total=max(int(total), self._work.completed))

    def advance(self, count: int = 1) -> None:
        """Add completed work to the current stage count."""
        completed = self._work.completed + int(count)
        self._move(completed=completed, total=max(self._work.total, completed))

    def finish(self, count: int) -> None:
        """Finish the stage with ``count`` completed units."""
        self._move(completed=int(count), total=int(count), running=False, publish=True)

    def _move(
        self,
        *,
        completed: int | None = None,
        total: int | None = None,
        running: bool = True,
        publish: bool = False,
    ) -> None:
        if self._owner is None:
            return
        self._work = replace(
            self._work,
            completed=self._work.completed if completed is None else completed,
            total=self._work.total if total is None else total,
            running=running,
            updated_at=self._owner.clock(),
        )
        self._owner.note(self._work, publish=publish)


class PreparationProgress:
    """Each unit's latest stage count, per coverage Task, in this process."""

    def __init__(self, clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        """Store stage counts using the supplied clock for update times."""
        self.clock = clock
        self.sink: Callable[[StageWork], object] | None = None
        """The Host's progress publisher, bound when the Host composes the adapter; unbound,
        nothing is published and the counts are still read."""
        self._lock = Lock()
        self._latest: dict[tuple[UUID, str | None], StageWork] = {}
        self._ended: dict[tuple[UUID, str | None, str], StageWork] = {}
        self._published: dict[tuple[UUID, str | None, str], datetime] = {}

    def begin(
        self, task_id: UUID, unit_id: str | None, stage: str, *, total: int = 0
    ) -> StageCounter:
        """Start counting one unit's stage.

        Return a counter that records nothing for a stage without count telemetry.
        """
        work = StageWork(
            task_id=task_id,
            unit_id=unit_id,
            stage=stage,
            unit_name=STAGE_UNIT_NAMES.get(stage, "units"),
            completed=0,
            total=max(0, int(total)),
            running=True,
            updated_at=self.clock(),
        )
        if stage not in STAGE_UNIT_NAMES:
            return StageCounter(None, work)
        self.note(work, publish=True)
        return StageCounter(self, work)

    def drop(self, task_id: UUID, unit_id: str | None) -> None:
        """Discard an unfinished stage's in-memory progress."""
        with self._lock:
            work = self._latest.get((task_id, unit_id))
            if work is not None and work.running:
                del self._latest[(task_id, unit_id)]

    def note(self, work: StageWork, *, publish: bool = False) -> None:
        """Store a stage update and publish it when due."""
        key = (work.task_id, work.unit_id, work.stage)
        with self._lock:
            self._latest[(work.task_id, work.unit_id)] = work
            if not work.running:
                self._ended[key] = work
            last = self._published.get(key)
            due = (
                publish
                or last is None
                or (work.updated_at - last).total_seconds() >= PUBLISH_INTERVAL_SECONDS
            )
            if due:
                self._published[key] = work.updated_at
        sink = self.sink
        if due and sink is not None:
            try:
                sink(work)
            except Exception:  # telemetry never stops the work it describes
                return

    def units(self, task_id: UUID) -> dict[str | None, StageWork]:
        """Return each unit's latest running or completed stage."""
        with self._lock:
            return {unit: work for (task, unit), work in self._latest.items() if task == task_id}

    def book(
        self, task_id: UUID, *, planned_filings: int, units_total: int, units_selected: int
    ) -> tuple[tuple[str, str, int, int], ...]:
        """Return the book's four stage counts.

        Each tuple contains ``(stage, unit_name, completed, total)``. Counts
        cover filings against the unit plans, documents against acquired units,
        chunks against built indexes, and selected units against the run.
        """
        with self._lock:
            works = [
                *(work for key, work in self._ended.items() if key[0] == task_id),
                *(work for key, work in self._latest.items() if key[0] == task_id and work.running),
            ]

        def counted(stage: str) -> tuple[int, int]:
            chosen = [work for work in works if work.stage == stage]
            return sum(work.completed for work in chosen), sum(work.total for work in chosen)

        fetched, _ = counted(ACQUIRE)
        documents, expected = counted(CANONICALIZE)
        acquired = sum(
            work.completed for work in works if work.stage == ACQUIRE and not work.running
        )
        embedded, cut = counted(BUILD)
        return (
            (ACQUIRE, "filings", fetched, max(planned_filings, fetched)),
            (CANONICALIZE, "documents", documents, max(acquired, expected)),
            (BUILD, "chunks", embedded, cut),
            (SELECT, "units", units_selected, units_total),
        )


def publish_evidence_work(publisher: WorkspaceProgressPublisher, work: StageWork) -> None:
    """Publish one stage's count into the workspace's progress projection (section 10.10).

    As a data preparation's step is published; a stage with nothing to count yet is not.

    Args:
        publisher: The workspace's progress publisher.
        work: The stage's work, as the preparation progress counts it.
    """
    if work.total < 1:
        return
    publisher.publish(
        WorkProgressUpdate(
            operation_id=str(work.task_id),
            stage_id=work.stage if work.unit_id is None else f"{work.unit_id}_{work.stage}",
            status="RUNNING" if work.running else "SUCCEEDED",
            completed_units=min(work.completed, work.total),
            total_units=work.total,
            unit_name=work.unit_name,
            current_item=work.unit_id,
        )
    )


__all__ = [
    "ACQUIRE",
    "BUILD",
    "CANONICALIZE",
    "PUBLISH_INTERVAL_SECONDS",
    "SELECT",
    "STAGE_UNIT_NAMES",
    "PreparationProgress",
    "StageCounter",
    "StageWork",
    "publish_evidence_work",
]
