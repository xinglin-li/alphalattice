"""What a heavy plan is estimated to take, and the memory check that refuses it early.

A plan answer for a heavy Task (workspace preparation, model-training inputs, a lifecycle Alpha
study, a research update) carries `resource_estimate`: wall seconds and peak memory at the
workspace's current CPU budget, labelled as an estimate, with its basis. Where a lower budget
really lowers the peak, the estimate at that budget is given too, so an agent can choose
without a second call.

The memory check compares the machine's available memory with the estimated peak, without
padding: an estimate never refuses work that fits. It runs before admission (the run or confirm
request is refused and no Task is made) and again before each stage of a Task this service
admitted (the stage blocks; RECOVER reopens it once memory is free). Available memory is the
commit Windows can still give, or Linux's MemAvailable plus SwapFree; elsewhere it is unknown
and the check is skipped.

These are execution facts: no sealed plan, identity or decision reads them. Each calibration
retains its measured size and CPU budget. Shared-machine receipts are indicative, scaled by
plan size and capped CPU scaling; another machine's speed differs.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import psutil  # type: ignore[import-untyped]

from alphalattice.control.observation_runtime.telemetry.process_metrics import (
    available_work_memory_bytes,
)
from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord
from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from alphalattice.evidence.alternative_evidence.runtime.execution import (
    TASK_LEASES,
    CpuBudgetStore,
    budget_cores,
    machine_load,
)
from alphalattice.interface.local_application.cli_contract import refusal_words

MEMORY_INSUFFICIENT = "execution.memory_insufficient"
_HOLDING_MEMORY = frozenset({TaskLifecycle.RUNNING, TaskLifecycle.CANCEL_REQUESTED})
"""Tasks still executing: they hold the memory a refused run waits for."""
_GIB = 2**30


@dataclass(frozen=True)
class _Calibration:
    """One heavy kind's measured cost at a reference size and CPU budget."""

    unit: str
    reference_units: float
    reference_cores: int
    reference_wall_seconds: float
    fixed_wall_seconds: float
    parallelism: float
    peak_bytes_per_unit: float
    peak_fixed_bytes: float
    peak_bytes_per_core: float


_CALIBRATIONS: Mapping[str, _Calibration] = {
    # One component's inputs, 472 listings x 2,511 sessions: 114 s at 4 cores, 12.7 GiB.
    "MODEL_TRAINING_INPUT_PREPARE": _Calibration(
        "listing-sessions", 472 * 2511, 4, 114.0, 0.0, 1.35, 12.7 * _GIB / (472 * 2511), 0.0, 0.0
    ),
    # Forward (3 sessions) 437 s and 14.4 GiB at 4 cores, 16.5 GiB at 16; daily (1 session)
    # about 400 s and 12.7 GiB at 16: most of an update is fixed (its data and inputs stages).
    "RESEARCH_UPDATE_RUN": _Calibration(
        "decision sessions", 3, 4, 437.0, 380.0, 1.5, 1.9 * _GIB, 8.0 * _GIB, 0.175 * _GIB
    ),
    # G2: 8,832 predictions and 45 fits, about 185 s at 4 cores; at most 3.3 GiB.
    "EXPERIMENT_RUN": _Calibration(
        "prediction calls", 8832, 4, 185.0, 0.0, 1.5, 0.0, 3.3 * _GIB, 0.0
    ),
    # About 520 candidates over ten years: 390 s and 5.2 GiB at 16 cores.
    "WORKSPACE_PREPARE_CONFIRM": _Calibration(
        "candidate listings", 520, 16, 390.0, 0.0, 2.3, 5.2 * _GIB / 520, 0.0, 0.0
    ),
}
_LIFECYCLE_CALIBRATIONS: Mapping[str, _Calibration] = {
    # Retained lifecycle receipts on 471 listings: 45/60 fits, 18/20 features.
    # Shared-machine observations; only elapsed cost changes, never memory admission.
    "G2_R0_TREND": _Calibration(
        "prediction calls", 8856, 26, 96.152624, 0.0, 1.5, 0.0, 3.3 * _GIB, 0.0
    ),
    "G6_R0_FAST_REBOUND": _Calibration(
        "prediction calls", 12624, 25, 398.845444, 0.0, 1.5, 0.0, 3.3 * _GIB, 0.0
    ),
}
_PLAN_RUNS = {
    "MODEL_TRAINING_INPUT_PLAN": ("MODEL_TRAINING_INPUT_PREPARE", "experiment_plan_hash"),
    "RESEARCH_UPDATE_PLAN": ("RESEARCH_UPDATE_RUN", "update_plan_hash"),
    "EXPERIMENT_PLAN": ("EXPERIMENT_RUN", "experiment_plan_hash"),
    "WORKSPACE_PREPARE_PLAN": ("WORKSPACE_PREPARE_CONFIRM", "preparation_plan_hash"),
}


def _units(plan_operation: str, body: Mapping[str, Any]) -> float | None:
    """The plan's size in its kind's unit, read from its own answer."""
    if plan_operation == "MODEL_TRAINING_INPUT_PLAN":
        listings, sessions = body.get("source_listing_count"), body.get("source_session_count")
        return float(listings * sessions) if listings and sessions else None
    if plan_operation == "RESEARCH_UPDATE_PLAN":
        sessions = body.get("decision_sessions")
        return float(len(sessions)) if isinstance(sessions, list) and sessions else None
    if plan_operation == "EXPERIMENT_PLAN":
        preview = body.get("execution_preview")
        calls = preview.get("prediction_call_upper_bound") if isinstance(preview, dict) else None
        return float(calls) if calls else None
    if plan_operation == "WORKSPACE_PREPARE_PLAN":
        count = body.get("candidate_count")
        return float(count) if count else None
    return None


def _estimate(calibration: _Calibration, units: float, cores: int) -> dict[str, Any]:
    variable = calibration.reference_wall_seconds - calibration.fixed_wall_seconds
    wall = calibration.fixed_wall_seconds + variable * units / calibration.reference_units
    speedup = min(calibration.parallelism, calibration.reference_cores) / min(
        calibration.parallelism, max(1, cores)
    )
    peak = (
        calibration.peak_fixed_bytes
        + calibration.peak_bytes_per_unit * units
        + calibration.peak_bytes_per_core * cores
    )
    return {
        "cpu_cores": cores,
        "wall_seconds": round(wall * speedup),
        "peak_memory_bytes": int(peak),
    }


class ResourceGate:
    """Estimates heavy plans, remembers them by plan and Task, and refuses work that won't fit."""

    def __init__(self, workspace: Path) -> None:
        """Bind the workspace whose CPU budget the estimates use."""
        self.workspace = workspace
        self._by_plan: dict[str, tuple[str, float | None, str | None, bool]] = {}
        self._by_task: dict[UUID, tuple[str, float | None, str | None, bool]] = {}
        self._lock = threading.Lock()

    def estimate(
        self,
        run_operation: str,
        units: float | None,
        lifecycle_component: str | None = None,
        remaining_work: bool = False,
    ) -> dict[str, Any]:
        """The estimate at the current budget, the lighter one where it lowers the peak."""
        calibration = (
            _LIFECYCLE_CALIBRATIONS.get(lifecycle_component or "") or _CALIBRATIONS[run_operation]
        )
        estimated_units = units if units is not None else calibration.reference_units
        budget = CpuBudgetStore(self.workspace / "runtime").read()
        load = machine_load()
        cores, reason = budget_cores(budget, load)
        value = {
            "label": "ESTIMATE",
            "cpu_budget": budget.cpu_budget,
            "cpu_sampled_at": datetime.now(UTC).isoformat(),
            "cpu_basis": reason,
            "cpu_load": {"processors": load.processors, "busy_processors": load.busy_processors},
            "cpu_selection": "RUN_START",
            "cpu_scope": "EXECUTION_ONLY",
            "cpu_detail": (
                "The CPU budget affects scheduling; resources are chosen when the Task starts."
            ),
            "elapsed_scope": "REMAINING_WORK" if remaining_work else "FULL_TASK",
            "wall_status": "UNKNOWN" if remaining_work else "ESTIMATED",
            "memory_scope": "FULL_TASK_CONSERVATIVE" if remaining_work else "FULL_TASK",
            "basis": (
                "Measured shared-machine calibration scales workload by the plan size when "
                "known, otherwise by the reference size, with capped CPU scaling; machine "
                "speed, source access and retries can change elapsed time."
            ),
            "basis_details": {
                "kind": "SHARED_MACHINE_CALIBRATION",
                "unit": calibration.unit,
                "reference_units": calibration.reference_units,
                "reference_cores": calibration.reference_cores,
                "reference_wall_seconds": calibration.reference_wall_seconds,
                "plan_units": units,
                "size_basis": "PLAN_SIZE" if units is not None else "CALIBRATION_REFERENCE_SIZE",
                "cpu_scaling_cap": calibration.parallelism,
            },
            **_estimate(calibration, estimated_units, cores),
            "available_memory_bytes": available_work_memory_bytes(),
        }
        if calibration.peak_bytes_per_core and cores > 1:
            value["at_one_core"] = _estimate(calibration, estimated_units, 1)
        if remaining_work:
            value["wall_seconds"] = None
            if "at_one_core" in value:
                value["at_one_core"]["wall_seconds"] = None
            value["basis"] = (
                "Remaining time is unknown: retained local units must be revalidated, and "
                "missing units may access their sources again. Memory uses the conservative "
                "full-task calibration."
            )
        return value

    def plan_answered(self, plan_operation: str, body: dict[str, Any]) -> None:
        """Attach the estimate to a heavy plan answer and remember it by its plan."""
        if plan_operation not in _PLAN_RUNS or body.get("status") not in {
            "PLANNED",
            "CONFIRMATION_REQUIRED",
        }:
            return
        plan_hash = body.get("update_plan_hash") or body.get("plan_hash")
        units = _units(plan_operation, body)
        if not isinstance(plan_hash, str) or (
            units is None and plan_operation != "WORKSPACE_PREPARE_PLAN"
        ):
            return
        run_operation = _PLAN_RUNS[plan_operation][0]
        preview = body.get("execution_preview", {})
        component = (
            preview.get("component_id")
            if plan_operation == "EXPERIMENT_PLAN"
            and preview.get("methodology_id") == "MODEL_LIFECYCLE_REPLAY"
            and preview.get("model_adapter_id") == "dynamic_panel_lightgbm"
            else None
        )
        remaining = plan_operation == "WORKSPACE_PREPARE_PLAN" and bool(
            body.get("predecessor_task_id")
        )
        with self._lock:
            self._by_plan[plan_hash] = (run_operation, units, component, remaining)
        body["resource_estimate"] = self.estimate(run_operation, units, component, remaining)

    def run_refusal(
        self,
        run_operation: str,
        plan_hash: str | None,
        registry: DuckDbTaskControlRegistry,
    ) -> dict[str, Any] | None:
        """A refusal answer when the plan's estimated peak exceeds available memory; else None."""
        remembered = self._by_plan.get(plan_hash or "")
        if remembered is None or remembered[0] != run_operation:
            return None
        estimate = self.estimate(*remembered)
        available = estimate["available_memory_bytes"]
        if available is None or available >= estimate["peak_memory_bytes"]:
            return None
        plan_key = next(key for op, key in _PLAN_RUNS.values() if op == run_operation)
        again = {"operation": run_operation, plan_key: plan_hash}
        # Read only when refusing, as a display collection: an unreadable row names itself
        # elsewhere and is not a Task to wait on here.
        running = [
            t for t in registry.record_collection().records if t.lifecycle in _HOLDING_MEMORY
        ]
        lighter = estimate.get("at_one_core")
        if running:
            next_action = "WAIT_FOR_THE_RUNNING_TASKS_THEN_RUN_AGAIN"
            next_requests: dict[str, Any] = {
                f"wait_{i}": {"operation": "STATUS", "task_id": str(t.task_id)}
                for i, t in enumerate(running)
            }
        elif lighter is not None and available >= lighter["peak_memory_bytes"]:
            next_action = "SET_THE_CPU_BUDGET_TO_ONE_CORE_THEN_RUN_AGAIN"
            next_requests = {"cpu_budget": {"operation": "CPU_BUDGET_SET", "cpu_budget": 1}}
        else:
            next_action = "ASK_THE_PERSON_TO_FREE_MEMORY_THEN_RUN_AGAIN"
            next_requests = {}
        return {
            "status": "REFUSED",
            "failure_code": MEMORY_INSUFFICIENT,
            "refused": MEMORY_INSUFFICIENT,
            "detail": refusal_words(MEMORY_INSUFFICIENT)["detail"],
            "next_action": next_action,
            "running_tasks": [
                {"task_id": str(t.task_id), "task_kind": t.task_kind} for t in running
            ],
            "resource_estimate": estimate,
            "next_requests": {**next_requests, "again": again},
        }

    def task_admitted(self, run_operation: str, plan_hash: str | None, task_id: str) -> None:
        """Remember an admitted Task's estimate for the check before each of its stages."""
        remembered = self._by_plan.get(plan_hash or "")
        if remembered is not None and remembered[0] == run_operation:
            with self._lock:
                self._by_task[UUID(task_id)] = remembered

    def stage_refusal(self, task: TaskRecord) -> str | None:
        """The refusal code when an admitted Task's estimated peak no longer fits; else None.

        Beside the Tasks that started before it, it also needs what their peaks have yet to
        take: their estimated peaks past what the process holds now. A Task counts only the
        older leases, so two starting together do not each count the other.
        """
        remembered = self._by_task.get(task.task_id)
        if remembered is None:
            return None
        estimate = self.estimate(*remembered)
        available = estimate["available_memory_bytes"]
        holders = TASK_LEASES.holders()
        older = holders[: holders.index(task.task_id)] if task.task_id in holders else holders
        others = [self._by_task[held] for held in older if held in self._by_task]
        unreached = sum(self.estimate(*other)["peak_memory_bytes"] for other in others)
        if unreached:
            unreached = max(0, unreached - psutil.Process().memory_info().rss)
        if available is None or available >= estimate["peak_memory_bytes"] + unreached:
            return None
        return MEMORY_INSUFFICIENT


_GATES: dict[Path, ResourceGate] = {}
_GATES_LOCK = threading.Lock()


def gate_for(workspace: Path) -> ResourceGate:
    """The one gate of a workspace in this process, shared by its requests and its runners."""
    key = Path(workspace).resolve()
    with _GATES_LOCK:
        return _GATES.setdefault(key, ResourceGate(key))


__all__ = ["MEMORY_INSUFFICIENT", "ResourceGate", "gate_for"]
