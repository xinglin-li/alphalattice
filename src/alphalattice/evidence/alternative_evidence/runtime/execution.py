"""Control CPU use for preparation, Task reads, and Alpha model fits.

An execution parameter is the operator's, never identity (the final close-out, F1). A budget
changes how fast a preparation runs and never what it prepares: each model session proves
the retrieval canary when it loads (`knowledge_hybrid_contracts.EMBEDDING_CANARIES`), so no
number depends on how the cores are split. The operator sets `auto` or a number of cores in
the workspace's typed configuration (`runtime/execution/cpu-budget.json`), by the CLI
(`cpu-budget set --cores`) or in the Workbench's settings; `auto` takes the processors not
in use when a preparation starts, never fewer than one session's safe threads
(`safe_intra_op_threads`). A preparation splits its cores between units at once and each
session's threads: the book's first unit runs alone on up to `MAXIMUM_SESSION_THREADS`
(S2 measured 1.6-1.8x at 8-16), then the later units share the cores at each safe width.
What each preparation used, and why, is appended to
`runtime/execution/preparations.jsonl`: a receipt, never identity. An Alpha study fits
one model at a time on the budget's cores, behind LightGBM's sealed canary
(`alpha_modeling.runtime.lightgbm_threads`); what its fits used is appended to
`runtime/execution/model-fits.jsonl` (binding plan, B3). Each Task the Host starts reads
DuckDB and Arrow on the budget's cores (`workspace_runtime.reader_threads`); what it started
with is appended to `runtime/execution/tasks.jsonl` (binding plan, B7). A Feature build in the
Task spreads its listings' computation over the Host's kept workers on those cores, one of them
left to the build's writer (`feature_engine.runtime.service`); the rows are the same at any count.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal, TypeVar, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from alphalattice.control.observation_runtime.telemetry.process_metrics import (
    busy_logical_processors,
    runtime_machine_capacity,
)
from alphalattice.kernel.knowledge._embeddings import VERIFIED_PACKS
from alphalattice.kernel.knowledge.hybrid_contracts import (
    MAXIMUM_SESSION_THREADS,
    safe_intra_op_threads,
)

EXECUTION_DIRECTORY = "execution"
"""Under the workspace's `runtime/`, beside the Host's own state and outside the managed
evidence roots: a setting and a log, not evidence."""
CPU_BUDGET_FILE = "cpu-budget.json"
PREPARATIONS_FILE = "preparations.jsonl"
MODEL_FITS_FILE = "model-fits.jsonl"
TASKS_FILE = "tasks.jsonl"
MAXIMUM_CPU_BUDGET = 1024

Cores = Annotated[int, Field(ge=1, le=MAXIMUM_CPU_BUDGET, strict=True)]
Chooser = Literal["DEFAULT", "HUMAN", "INSTALLED_AGENT", "EXTERNAL_AUTOMATION"]


class _Record(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class CpuBudget(_Record):
    """The operator's CPU budget: `auto`, or the cores a preparation may use."""

    schema_version: Literal[1] = 1
    cpu_budget: Literal["auto"] | Cores = "auto"
    chosen_by: Chooser = "DEFAULT"
    chosen_at: datetime | None = None


class MachineLoad(_Record):
    """Describe machine capacity and load at preparation time.

    An operator or agent reads these values to choose a budget.
    """

    processors: int = Field(ge=1)
    """The logical processors this process may run on."""
    busy_processors: float | None = Field(ge=0)
    """Processors in use over a quarter second, every process's work; None when unread."""
    total_memory_bytes: int = Field(ge=0)
    available_memory_bytes: int = Field(ge=0)
    preparations_running: int = Field(ge=0)
    """Book preparations of this Host running when read."""


class PreparationExecution(_Record):
    """What one preparation ran with, and why."""

    task_id: UUID | None
    units: int = Field(ge=1)
    cpu_budget: Literal["auto"] | Cores
    cores: int = Field(ge=1)
    units_at_once: int = Field(ge=1)
    threads_first_unit: int = Field(ge=1)
    threads_per_session: int = Field(ge=1)
    reason: str
    machine: MachineLoad
    planned_at: datetime


class PlatformIdentity(_Record):
    """Describe the machine and libraries used by a Task.

    This belongs to the realization record, not the Task identity (binding plan, N4).
    """

    system: str
    release: str
    architecture: str
    processor: str
    python: str
    libraries: dict[str, str]


def platform_identity() -> PlatformIdentity:
    import importlib.metadata
    import platform

    libraries = {}
    for name in ("numpy", "pandas", "pyarrow", "duckdb", "lightgbm", "scikit-learn"):
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return PlatformIdentity(
        system=platform.system(),
        release=platform.release(),
        architecture=platform.machine(),
        processor=platform.processor() or platform.machine(),
        python=platform.python_version(),
        libraries=libraries,
    )


class TaskReaderExecution(_Record):
    """What one Task's DuckDB and Arrow reads ran with, and why."""

    task_id: UUID
    task_kind: str
    cpu_budget: Literal["auto"] | Cores
    cores: int = Field(ge=1)
    reader_threads: int = Field(ge=1)
    reason: str
    machine: MachineLoad
    started_at: datetime
    platform: PlatformIdentity | None = None
    """Absent from what a Task recorded before the realization record (N4)."""


class ModelFitExecution(_Record):
    """What one Alpha study's LightGBM fits ran with, and why."""

    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cpu_budget: Literal["auto"] | Cores
    cores: int = Field(ge=1)
    lightgbm_threads: int = Field(ge=1)
    reason: str
    machine: MachineLoad
    planned_at: datetime
    numerical_threads: tuple[tuple[str, int], ...] = ()
    """Each numerical library's thread count (BLAS, OpenMP) as the child inherits it: the fold
    metrics' sums split by it, so a result claims no equality with a run under another (PA3).
    Empty on records written before it was kept."""


_Logged = TypeVar("_Logged", PreparationExecution, ModelFitExecution, TaskReaderExecution)


def parse_cpu_budget(value: object) -> Literal["auto"] | int:
    """`auto` or a whole number of cores from 1; anything else refused by name."""
    if value == "auto":
        return "auto"
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if type(value) is not int or not 1 <= value <= MAXIMUM_CPU_BUDGET:
        raise ValueError("execution.cpu_budget_invalid")
    return value


def machine_load(*, preparations_running: int = 0) -> MachineLoad:
    """Read machine capacity and current load for budget planning."""
    capacity = runtime_machine_capacity()
    return MachineLoad(
        processors=len(capacity.allowed_logical_processor_ids),
        # The work on this process's own processors, not the machine's.
        busy_processors=busy_logical_processors(
            processor_ids=capacity.allowed_logical_processor_ids
        ),
        total_memory_bytes=capacity.total_memory_bytes,
        available_memory_bytes=capacity.available_memory_bytes,
        preparations_running=preparations_running,
    )


def budget_cores(
    budget: CpuBudget, machine: MachineLoad, *, floor: int = 1, floor_reason: str = "one thread"
) -> tuple[int, str]:
    """Return the cores available to one piece of work and the reason.

    `auto` is the processors not in use (other preparations' work among what is in use),
    never fewer than `floor`; a number of cores is held to the processors this process may
    use and shared with the preparations already running.
    """
    processors = machine.processors
    others = machine.preparations_running
    if budget.cpu_budget == "auto":
        if machine.busy_processors is None:
            idle = processors
            reason = f"auto: the {processors} processors (the machine's load was not read)"
        else:
            idle = max(0, int(processors - machine.busy_processors))
            reason = f"auto: {idle} of {processors} processors not in use"
            if others:
                reason += f", {others} other preparations running among the rest"
        cores = max(floor, min(processors, idle))
        if cores > idle:
            reason += f"; never fewer than {floor_reason}"
        return cores, reason
    cores = min(budget.cpu_budget, processors)
    reason = f"set to {budget.cpu_budget} cores"
    if budget.cpu_budget > processors:
        reason += f", held to the {processors} processors this process may use"
    if others:
        cores = max(1, cores // (others + 1))
        reason += f"; shared with {others} other preparations running"
    return cores, reason


def plan_execution(
    budget: CpuBudget,
    machine: MachineLoad,
    *,
    units: int,
    task_id: UUID | None,
    planned_at: datetime,
) -> PreparationExecution:
    """Split the budget between units at once and each session's threads.

    `auto` is the processors not in use (other preparations' work among what is in use); a
    number of cores is held to the processors this process may use and shared with the
    preparations already running. At least one session of the safe width runs.
    """
    safe = safe_intra_op_threads(machine.processors)
    cores, reason = budget_cores(
        budget, machine, floor=safe, floor_reason=f"one session's {safe} threads"
    )
    units = max(1, units)
    at_once = max(1, min(units, cores // safe))
    threads = max(1, min(MAXIMUM_SESSION_THREADS, cores // at_once))
    first = max(1, min(MAXIMUM_SESSION_THREADS, cores))
    reason += (
        f"; the first unit alone on {first} threads"
        if units == 1
        else f"; the first unit alone on {first} threads, then {at_once} units at once "
        f"on {threads} threads each"
    )
    return PreparationExecution(
        task_id=task_id,
        units=units,
        cpu_budget=budget.cpu_budget,
        cores=cores,
        units_at_once=at_once,
        threads_first_unit=first,
        threads_per_session=threads,
        reason=reason,
        machine=machine,
        planned_at=planned_at,
    )


class CpuBudgetStore:
    """The workspace's CPU budget and the log of what preparations ran with."""

    def __init__(self, runtime_root: Path) -> None:
        """Locate the budget setting and execution logs under the runtime root."""
        self.root = runtime_root / EXECUTION_DIRECTORY
        self._lock = Lock()

    def read(self) -> CpuBudget:
        """Read the selected budget, defaulting to ``auto``.

        Refuse an unreadable file by name so the operator can set it again.
        """
        path = self.root / CPU_BUDGET_FILE
        if not path.is_file():
            return CpuBudget()
        try:
            return cast(CpuBudget, CpuBudget.model_validate_json(path.read_bytes()))
        except (OSError, ValueError) as error:
            raise ValueError("execution.cpu_budget_unreadable") from error

    def write(self, value: object, *, chosen_by: Chooser, chosen_at: datetime) -> CpuBudget:
        """Validate and atomically write the operator's CPU budget."""
        budget = CpuBudget(
            cpu_budget=parse_cpu_budget(value), chosen_by=chosen_by, chosen_at=chosen_at
        )
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / CPU_BUDGET_FILE
            staged = path.with_name(path.name + ".partial")
            staged.write_text(budget.model_dump_json(), encoding="utf-8")
            os.replace(staged, path)
        return budget

    def record(
        self, execution: PreparationExecution | ModelFitExecution | TaskReaderExecution
    ) -> None:
        """Append an execution receipt to its corresponding log."""
        name = (
            MODEL_FITS_FILE
            if isinstance(execution, ModelFitExecution)
            else TASKS_FILE
            if isinstance(execution, TaskReaderExecution)
            else PREPARATIONS_FILE
        )
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / name).open("a", encoding="utf-8") as log:
                log.write(execution.model_dump_json() + "\n")

    def last(self) -> PreparationExecution | None:
        """Return the last readable preparation receipt.

        Return ``None`` before the first receipt or for a partial last line.
        """
        return self._last(PREPARATIONS_FILE, PreparationExecution)

    def last_task(self) -> TaskReaderExecution | None:
        """Return the last Task reader execution, if recorded."""
        return self._last(TASKS_FILE, TaskReaderExecution)

    def task_execution(self, task_id: UUID) -> TaskReaderExecution | None:
        """Return the latest reader execution for a Task, if recorded."""
        found = self._each(TASKS_FILE, TaskReaderExecution, lambda v: v.task_id == task_id)
        return found[-1] if found else None

    def model_fit_execution(self, program_hash: str) -> ModelFitExecution | None:
        """Return the latest LightGBM execution for a Program, if recorded."""
        found = self._each(
            MODEL_FITS_FILE, ModelFitExecution, lambda v: v.program_hash == program_hash
        )
        return found[-1] if found else None

    def _each(
        self, name: str, kind: type[_Logged], wanted: Callable[[_Logged], bool]
    ) -> list[_Logged]:
        try:
            lines = (self.root / name).read_bytes().splitlines()
        except OSError:
            return []
        found = []
        for line in lines:
            try:
                value = kind.model_validate_json(line)
            except ValueError:
                continue  # a partial line
            if wanted(value):
                found.append(value)
        return found

    def last_model_fits(self) -> ModelFitExecution | None:
        """Return the last Alpha study's LightGBM execution, if recorded."""
        return self._last(MODEL_FITS_FILE, ModelFitExecution)

    def _last(self, name: str, kind: type[_Logged]) -> _Logged | None:
        path = self.root / name
        try:
            with path.open("rb") as log:
                log.seek(max(0, path.stat().st_size - 16384))
                lines = log.read().splitlines()
        except OSError:
            return None
        try:
            return kind.model_validate_json(lines[-1]) if lines else None
        except ValueError:
            return None


class RunningPreparations:
    """The book preparations of this process, and the threads their sessions load with.

    A unit's own stages lease sessions of its share (`threads`: the first unit its plan's
    `threads_first_unit`, a later unit `threads_per_session`), side by side (F2). Sessions
    leased elsewhere -- a selection's reader -- load with the fewest threads any running
    preparation allows (`VERIFIED_PACKS.use_threads`): the first unit's while it runs alone,
    the later units' share once they begin their model work (`open`).
    """

    def __init__(self) -> None:
        """Track concurrent preparations and their model session widths."""
        self._lock = Lock()
        self._threads: dict[UUID, int] = {}
        self._plans: dict[UUID, PreparationExecution] = {}

    def begin(self, task_id: UUID, plan: PreparationExecution) -> None:
        """Register a preparation and apply its first-unit session width."""
        with self._lock:
            self._plans[task_id] = plan
            self._threads[task_id] = plan.threads_first_unit
            VERIFIED_PACKS.hold_up_to(plan.units_at_once)
            self._apply()

    def open(self, task_id: UUID) -> None:
        """Apply the session width for later units in a preparation."""
        with self._lock:
            plan = self._plans.get(task_id)
            if plan is None or self._threads.get(task_id) == plan.threads_per_session:
                return
            self._threads[task_id] = plan.threads_per_session
            self._apply()

    def threads(self, task_id: UUID, *, first: bool) -> int | None:
        """Return a unit's session width, if its Task was planned."""
        with self._lock:
            plan = self._plans.get(task_id)
        if plan is None:
            return None
        return plan.threads_first_unit if first else plan.threads_per_session

    def running(self, alive: Callable[[UUID], bool]) -> int:
        """How many run, forgetting those `alive` says have ended."""
        with self._lock:
            for task_id in [value for value in self._plans if not alive(value)]:
                del self._plans[task_id]
                self._threads.pop(task_id, None)
            self._apply()
            return len(self._plans)

    def _apply(self) -> None:
        VERIFIED_PACKS.use_threads(min(self._threads.values()) if self._threads else None)


PREPARATIONS = RunningPreparations()
"""This process's preparations (model sessions are the process's; `VERIFIED_PACKS`)."""


__all__ = [
    "CPU_BUDGET_FILE",
    "EXECUTION_DIRECTORY",
    "MAXIMUM_CPU_BUDGET",
    "PREPARATIONS",
    "PREPARATIONS_FILE",
    "CpuBudget",
    "CpuBudgetStore",
    "MachineLoad",
    "PreparationExecution",
    "RunningPreparations",
    "machine_load",
    "parse_cpu_budget",
    "plan_execution",
]
