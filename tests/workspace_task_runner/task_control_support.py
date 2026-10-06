"""A Task contract and its compatibility for the Task Control tests: one helper, two readers."""

from __future__ import annotations

from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    TaskExecutionCompatibility,
    TaskInputEnvelope,
    WorkItemDefinition,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


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


def compatibility(plan: ResearchPlan) -> TaskExecutionCompatibility:
    """The execution compatibility a Task of this plan is admitted under."""
    return TaskExecutionCompatibility.create(
        task_contract_hash=digest("task-contract"),
        workflow_definition_hash=plan.workflow_definition_hash,
        input_schema_id="factor-research.confirmed-mandate",
        domain_policy_hash=digest("domain-policy"),
        framework_identity_hash=digest("framework"),
    )
