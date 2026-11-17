"""A Task contract and its compatibility for the Task Control tests: one helper, two readers."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import duckdb
import pytest

from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    TaskEvidence,
    TaskExecution,
    TaskInputEnvelope,
    TaskRecord,
    TaskStageReceipt,
    WorkItemDefinition,
)
from alphalattice.control.task_control.contracts import (
    TaskExecutionCompatibility as Compatibility,
)
from alphalattice.control.task_control.ledger import REQUESTS_DIRECTORY
from alphalattice.control.task_control.queue import write_queue_setting
from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from alphalattice.control.task_control.runner import StageDisposition, StageExecutionResult
from alphalattice.control.task_control.upgrade import carry_tasks
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def set_tasks_waiting(workspace: Path, count: int, at: datetime) -> None:
    """Declare the waiting places used by this synthetic task fixture."""
    write_queue_setting(
        workspace / "runtime", count, chosen_by="HUMAN", chosen_at=at, field="tasks_waiting"
    )


def task_execution(task: TaskRecord, compatibility: Compatibility, at: datetime) -> TaskExecution:
    """Allocate a worker/execution identity and bind its thread and clock to this Task."""
    execution_id = uuid4()
    return TaskExecution.from_identity(
        execution_id=execution_id,
        task_id=task.task_id,
        graph_thread_id=f"workspace-task:{task.task_id}:{execution_id}",
        worker_instance_id=uuid4(),
        compatibility=compatibility,
        started_at=at,
        last_heartbeat_at=at,
        checkpoint_disposition="active",
    )


def digest(label: str) -> str:
    """A stable hash standing for the named artifact."""
    return str(canonical_hash({"label": label}))


def work_item(
    stage_id: str,
    *,
    dependencies: tuple[str, ...] = (),
    required: tuple[str, ...] = (),
) -> WorkItemDefinition:
    """One work item of a plan, verified by its stage's own verifier."""
    return WorkItemDefinition.create(
        stage_id=stage_id,
        dependency_ids=dependencies,
        verifier_id=f"task-control.{stage_id}",
        required_evidence_kinds=required,
    )


def task_contract(*, salt: str = "first") -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    """A two-stage Factor research Task's input, goal and plan."""
    envelope = TaskInputEnvelope.create(
        task_kind="factor_research",
        input_schema_id="factor-research.confirmed-mandate",
        payload={"cadence": "DAILY", "salt": salt},
    )
    goal = ResearchGoal.create(
        goal_kind="RUN_FACTOR_RESEARCH",
        input_hash=envelope.input_hash,
        deliverable_kind="ResearchDeskFactorInput",
        summary="Publish one governed DAILY research input.",
    )
    definitions = (
        work_item("resolve_inputs", required=("input_binding",)),
        work_item(
            "publish_result",
            dependencies=("resolve_inputs",),
            required=("screening_report",),
        ),
    )
    plan = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=digest("factor-workflow"),
        verifier_catalog_hash=digest("factor-verifiers"),
        work_items=definitions,
    )
    return envelope, goal, plan


def compatibility(plan: ResearchPlan) -> Compatibility:
    """The execution compatibility a Task of this plan is admitted under."""
    return Compatibility.create(
        task_contract_hash=digest("task-contract"),
        workflow_definition_hash=plan.workflow_definition_hash,
        input_schema_id="factor-research.confirmed-mandate",
        domain_policy_hash=digest("domain-policy"),
        framework_identity_hash=digest("framework"),
    )


class FixtureTaskAdapter:
    """A two-stage Task's stages: ready with their evidence, or failing or deferring once."""

    task_kind = "factor_research"

    def __init__(self, *, fail_once: bool = False, defer_once: bool = False) -> None:
        self.fail_once = fail_once
        self.defer_once = defer_once
        self.executed: list[str] = []

    def compatibility(self, task):
        return compatibility(task.plan)

    def execute_stage(self, *, task, execution, work_item):
        del task, execution
        self.executed.append(work_item.stage_id)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("simulated process interruption")
        if self.defer_once:
            self.defer_once = False
            return StageExecutionResult(
                disposition=StageDisposition.DEFERRED,
                failure_code="AGENT_TEMPORARILY_UNAVAILABLE",
            )
        kind = {
            "resolve_inputs": "input_binding",
            "publish_result": "screening_report",
        }[work_item.stage_id]
        return StageExecutionResult(
            disposition=StageDisposition.READY,
            evidence=(
                TaskEvidence(
                    evidence_kind=kind,
                    reference=f"playpen://runner/{work_item.stage_id}",
                    content_hash=digest(work_item.stage_id),
                ),
            ),
        )

    def verify_stage(self, *, task, execution, work_item, evidence):
        del task, execution, work_item
        return evidence


@contextmanager
def task_log() -> Iterator[sqlite3.Connection]:
    """The held log's connection in one unit: a test damages a Task's rows on it as a broken
    write would, the one writer's guard held."""
    with history.held().unit() as connection:
        yield connection


def lose_the_log_keeping_frozen_requests(
    registry: DuckDbTaskControlRegistry,
    gate: WorkspaceMutationGate,
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> DuckDbTaskControlRegistry:
    """A 0.2.7 workspace whose Task store was lost but whose frozen requests stayed: its requests
    written out as 0.2.7 kept them, and the process moved to the empty data root `root`, where
    opening the registry again lists each Task again from its request (`ledger_rebuilt`)."""
    with history.held().reading() as connection:
        rows = connection.execute(
            "SELECT t.text FROM event e JOIN event_text t ON t.seq = e.seq"
            " WHERE e.type = 'task.requested' AND e.workspace_id = ?",
            [registry.workspace_id],
        ).fetchall()
    requests = registry.database_path.parent / REQUESTS_DIRECTORY
    requests.mkdir(parents=True, exist_ok=True)
    for (text,) in rows:
        (requests / f"{canonical_hash(json.loads(text))}.json").write_text(text, encoding="utf-8")
    history.close_history()
    monkeypatch.setenv(history.DATA_ROOT_ENV, str(root))
    folder = registry.database_path.parent.parent
    history.workspace_on_log(folder, folder.name, registry.workspace_id)
    return DuckDbTaskControlRegistry(registry.database_path, gate=gate)


def write_0_2_7_store(registry: DuckDbTaskControlRegistry, *, ordered: bool = True) -> None:
    """Write a workspace's Tasks out as the store a 0.2.7 workspace kept; `ordered=False` is a
    store from before admissions were ordered."""
    folder = registry.database_path.parent.parent
    carry_tasks(folder, folder)
    if not ordered:
        with duckdb.connect(str(registry.database_path)) as store:
            store.execute("ALTER TABLE workspace_task DROP COLUMN admission_sequence")


def complete_stage(
    registry: DuckDbTaskControlRegistry,
    *,
    task_id,
    execution_id,
    stage_id: str,
    evidence_kind: str,
    now: datetime,
):
    """Begin, ready and verify one stage of a running Task with one piece of evidence."""
    item = registry.begin_work_item(
        task_id=task_id,
        execution_id=execution_id,
        stage_id=stage_id,
        observed_at=now,
    )
    evidence = (
        TaskEvidence(
            evidence_kind=evidence_kind,
            reference=f"playpen://task-evidence/{stage_id}",
            content_hash=digest(stage_id),
        ),
    )
    registry.mark_ready(
        task_id=task_id,
        execution_id=execution_id,
        stage_id=stage_id,
        evidence=evidence,
        observed_at=now + timedelta(seconds=1),
    )
    definition = next(
        value for value in registry.task(task_id).plan.work_items if value.stage_id == stage_id
    )
    return registry.verify_work_item(
        TaskStageReceipt.from_identity(
            receipt_id=uuid4(),
            task_id=task_id,
            execution_id=execution_id,
            stage_id=stage_id,
            work_item_definition_hash=item.definition_hash,
            verifier_id=definition.verifier_id,
            evidence=evidence,
            status="VERIFIED",
            failure_code=None,
            observed_at=now + timedelta(seconds=2),
        )
    )
