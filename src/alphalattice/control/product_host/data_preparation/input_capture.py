"""Capture an explicitly selected research input revision without updating Data or config."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    manifest_fields_hash,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    factor_input_paths,
    publish_prepared_factor_inputs,
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.input_revisions import (
    CAPTURE_TASK_KIND,
    ResearchInputRevision,
    ResearchInputRevisions,
    ResearchInputSource,
    describe_input_source,
)
from alphalattice.control.product_host.storage.plan_previews import (
    PLAN_PREVIEWS_DIRECTORY,
    PreviewRegistry,
)
from alphalattice.control.product_host.storage.retention import require_no_pending_cleanup
from alphalattice.control.task_control.child import ChildStartFailed
from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    TaskEvidence,
    TaskExecution,
    TaskExecutionCompatibility,
    TaskInputEnvelope,
    TaskLifecycle,
    TaskRecord,
    TaskReplan,
    WorkItemDefinition,
)
from alphalattice.control.task_control.runner import (
    StageDisposition,
    StageExecutionResult,
)
from alphalattice.interface.local_application.dispatcher import (
    CommandAdmission,
    LocalBackgroundDispatcher,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.sealing import seal_model
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash

STAGE = "publish_input_version"


IMPLEMENTATION_ROLE = "product_host.research_input_capture"
PLAN_FIELDS = ("data_update", "experiment_inputs")
"""The manifest fields a capture plan reads and writes: the data it captures and the
inputs it adds a version to. A publication of other fields (a model's training inputs,
a strategy installation) leaves the plan applicable (V180)."""
"""The role a move of this implementation is recorded under in `config/identity-successors.json`,
so a plan sealed under its predecessor stays current (binding plan R1, LAWS.md ID1)."""


def _implementation() -> str:
    return source_rule_closure_hash(
        root=resolve_playpen_root(Path(__file__)),
        semantic_owner="product_host",
        numerical_role="RESEARCH_INPUT_CAPTURE_TASK",
        tracked_paths=("src/alphalattice/control/product_host/data_preparation/input_capture.py",),
    )


class ResearchInputCapturePlan(BaseModel):  # type: ignore[misc]
    """Seal an explicit local input revision against its anchor, parent and source identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    workspace_id: str
    workspace_manifest_hash: str
    input_id: str
    anchor_binding_hash: str
    prior_binding_hash: str
    previous_publication_hash: str | None
    source: ResearchInputSource
    implementation_hash: str
    plan_hash: str

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal an exact research input capture plan.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical plan_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_model(cls, values, field="plan_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def verify(self) -> Self:
        """Require exact canonical research input capture plan identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Serialized fields differ from plan_hash.
        """
        if self.plan_hash != canonical_hash(self.model_dump(mode="json", exclude={"plan_hash"})):
            raise ValueError("research_input.plan_invalid")
        return self


def _task_contract(
    plan: ResearchInputCapturePlan,
) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    envelope = TaskInputEnvelope.create(
        task_kind=CAPTURE_TASK_KIND,
        input_schema_id="research-input-capture",
        payload={"plan": plan.model_dump(mode="json"), "caller": "HUMAN"},
    )
    goal = ResearchGoal.create(
        goal_kind="CAPTURE_RESEARCH_INPUT",
        input_hash=envelope.input_hash,
        deliverable_kind="ResearchInputRevision",
        summary="Seal verified local inputs; no data update or strategy change.",
    )
    workflow = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash(STAGE),
        verifier_catalog_hash=canonical_hash(STAGE),
        work_items=(
            WorkItemDefinition.create(
                stage_id=STAGE, dependency_ids=(), verifier_id="research_input.publication"
            ),
        ),
    )
    return envelope, goal, workflow


class ResearchInputCaptureApplication:
    """Own explicitly confirmed local input revisions without changing default research bindings."""

    task_kind = CAPTURE_TASK_KIND
    replans = (
        TaskReplan(
            task_kind=CAPTURE_TASK_KIND,
            preview="RESEARCH_INPUT_PLAN",
            admitting="RESEARCH_INPUT_CONFIRM",
        ),
    )
    """The re-plan of the Task kind this owner admits, which the recovery view offers
    (V188)."""

    def __init__(self, session: WorkspaceApplicationSession, *, clock: Callable[[], datetime]):
        """Wire retained capture tasks and deterministic input revision lineage.

        Args:
            session: Retained workspace writer/task session.
            clock: Explicit observed-time source.
        """
        self.session, self.clock = session, clock
        self.revisions = ResearchInputRevisions(session)
        self.last_plan: ResearchInputCapturePlan | None = None
        # Every plan an answer named, by its hash, sealed on disk until it expires: a run
        # from any of them, after a restart too, reopens it and checks it again (V493, V525).
        self._plans: PreviewRegistry[ResearchInputCapturePlan] = PreviewRegistry(
            model=ResearchInputCapturePlan,
            clock=self.clock,
            root=self.session.workspace / "runtime" / PLAN_PREVIEWS_DIRECTORY / "research-input",
        )

    def _source(self, root: Path | None = None) -> ResearchInputSource:
        manifest = read_research_workspace_manifest(self.session.workspace)
        if manifest.data_update is None:
            raise ValueError("research_input.data_binding_required")
        return describe_input_source(root or self.session.workspace, manifest.data_update)

    def plan(self, input_id: str) -> dict[str, object]:
        """Compare exact local sources and reuse or preview an explicit input revision.

        Args:
            input_id: Declared research input selection.

        Returns:
            Exact reused revision, retained in-flight task or confirmation preview; no network,
            numerical or model-training work is performed.
        """
        require_no_pending_cleanup(self.session.workspace)
        manifest = read_research_workspace_manifest(self.session.workspace)
        anchor = self.revisions.anchor(input_id)
        source = self._source()
        pending = next(
            (
                t
                for t in self.session.task_control_registry.tasks()
                if t.task_kind == CAPTURE_TASK_KIND
                and t.lifecycle not in {TaskLifecycle.SUCCEEDED, TaskLifecycle.CANCELLED}
                and self._of(t).input_id == input_id
                and self._of(t).source == source
            ),
            None,
        )
        if pending is not None:
            if pending.lifecycle is TaskLifecycle.BLOCKED:
                self._require(self._of(pending))
                read = self.readback(pending.task_id)
                return {
                    **read,
                    "status": "CONFIRMATION_REQUIRED",
                    "plan_hash": self._of(pending).plan_hash,
                    "resume_task_id": str(pending.task_id),
                    "next_action": "REPAIR_FAILURE_THEN_CONFIRM_RETRY",
                    "next_requests": {
                        **cast(dict[str, object], read.get("next_requests") or {}),
                        "confirm": {
                            "operation": "RESEARCH_INPUT_CONFIRM",
                            "research_input_plan_hash": self._of(pending).plan_hash,
                        },
                    },
                }
            return self.readback(pending.task_id)
        lineage = self.revisions.lineage(input_id)
        parent = lineage[-1] if lineage else None
        prior = parent.binding_hash if parent else anchor.binding_hash
        # A legacy base has no capture-key record. Derive the same source
        # projection from its verified sealed DB, rather than guessing by date.
        if parent is None:
            read_factor_bundle(self.session.workspace, prior)
            old_source = self._source(factor_input_paths(self.session.workspace, prior)[0])
        else:
            old_source = parent.source
        if old_source == source:
            self.revisions.select(input_id, prior)
            return {
                "status": "REUSED_EXACT",
                "task_id": None,
                "input_id": input_id,
                "binding_hash": prior,
                "next_action": "SELECT_INPUT_VERSION",
            }
        # The answer names this plan, never the owner's last one, which a concurrent plan
        # may have replaced in between (V534).
        planned = ResearchInputCapturePlan.create(
            workspace_id=manifest.workspace_id,
            workspace_manifest_hash=manifest_fields_hash(manifest, PLAN_FIELDS),
            input_id=input_id,
            anchor_binding_hash=anchor.binding_hash,
            prior_binding_hash=prior,
            previous_publication_hash=parent.receipt_hash if parent else None,
            source=source,
            implementation_hash=_implementation(),
        )
        self.last_plan = planned
        self._plans.remember(planned)
        return {
            "status": "CONFIRMATION_REQUIRED",
            "plan_hash": planned.plan_hash,
            "input_id": input_id,
            "prior_binding_hash": prior,
            "before": old_source.model_dump(mode="json"),
            "after": source.model_dump(mode="json"),
            "network_calls": 0,
            "numerical_calls": 0,
            "limits": ["EXPLICIT_VERSION_ONLY", "DEFAULTS_AND_CU_UNCHANGED", "NO_MODEL_TRAINING"],
            "next_requests": {
                "confirm": {
                    "operation": "RESEARCH_INPUT_CONFIRM",
                    "research_input_plan_hash": planned.plan_hash,
                }
            },
        }

    def _of(self, task: TaskRecord) -> ResearchInputCapturePlan:
        if task.task_kind != CAPTURE_TASK_KIND:
            raise ValueError("research_input.task_kind_invalid")
        plan: ResearchInputCapturePlan = ResearchInputCapturePlan.model_validate(
            task.input.payload["plan"]
        )
        if (task.input, task.goal, task.plan) != _task_contract(plan):
            raise ValueError("research_input.task_contract_invalid")
        return plan

    def replan_request(self, task: TaskRecord) -> dict[str, object]:
        """Replan the validated Task's research input without depending on a preview.

        Args:
            task: Durable Task whose declaration supplies the next request.

        Returns:
            The filled research-input planning request.

        Raises:
            ValueError: The Task's kind, sealed plan or workflow does not validate.
        """
        plan = self._of(task)
        return {"operation": "RESEARCH_INPUT_PLAN", "research_input_id": plan.input_id}

    def _require(self, plan: ResearchInputCapturePlan) -> None:
        manifest = read_research_workspace_manifest(self.session.workspace)
        if (
            plan.workspace_id != manifest.workspace_id
            or plan.workspace_manifest_hash != manifest_fields_hash(manifest, PLAN_FIELDS)
            or not is_current(IMPLEMENTATION_ROLE, plan.implementation_hash, _implementation())
            or self.revisions.anchor(plan.input_id).binding_hash != plan.anchor_binding_hash
        ):
            raise ValueError("research_input.configuration_changed")

    def _source_and_parent(self, plan: ResearchInputCapturePlan) -> None:
        self._require(plan)
        if self._source() != plan.source:
            raise ValueError("research_input.source_changed_preview_again")
        lineage = self.revisions.lineage(plan.input_id)
        if (lineage[-1].receipt_hash if lineage else None) != plan.previous_publication_hash:
            raise ValueError("research_input.publication_plan_stale")

    def replan_requests(self, plan_hash: str) -> dict[str, object]:
        """Name the input of a verified retained plan, even when its run has expired.

        Args:
            plan_hash: The refused plan's exact hash.

        Returns:
            A bound re-plan, or no offer when the plan cannot be verified.
        """
        kept = self._plans.get(plan_hash)
        if kept is None:
            return {}
        return {
            "replan": {
                "operation": "RESEARCH_INPUT_PLAN",
                "research_input_id": kept.plan.input_id,
            }
        }

    def confirm(
        self, plan_hash: str, *, caller: str, dispatcher: LocalBackgroundDispatcher
    ) -> dict[str, object]:
        """Require human confirmation before reusing, recovering or dispatching exact capture.

        Args:
            plan_hash: Exact preview or retained capture plan.
            caller: Explicit caller required to be HUMAN.
            dispatcher: Bounded local dispatcher.

        Returns:
            Verified reused publication, exact in-flight recovery or new submission metadata.

        Raises:
            ValueError: Caller is not human, preview is absent or retained publication/source is
                invalid.
        """
        if caller != "HUMAN":
            raise ValueError("research_input.human_confirmation_required")
        existing = next(
            (
                t
                for t in self.session.task_control_registry.tasks()
                if t.task_kind == CAPTURE_TASK_KIND
                and self._of(t).plan_hash == plan_hash
                and t.lifecycle is not TaskLifecycle.CANCELLED
            ),
            None,
        )
        if existing is not None:
            if existing.lifecycle is TaskLifecycle.SUCCEEDED:
                value = self.revisions.for_task(existing.task_id)
                if value is None:
                    raise ValueError("research_input.publication_missing")
                self.revisions.select(value.input_id, value.binding_hash)
                return {
                    "status": "REUSED_EXACT",
                    "task_id": None,
                    "publication_task_id": str(existing.task_id),
                    "binding_hash": value.binding_hash,
                }
            if existing.lifecycle is TaskLifecycle.BLOCKED:
                plan = self._of(existing)
                self._require(plan)
                value = self.revisions.for_task(existing.task_id)
                if value is None:
                    self._source_and_parent(plan)
                else:
                    read_factor_bundle(self.session.workspace, value.binding_hash)
                self.session.task_control_registry.mark_recovery_required(
                    task_id=existing.task_id,
                    failure_code=existing.failure_code or "research_input.retry_requested",
                    observed_at=self.clock(),
                    allow_blocked=True,
                )
            elif existing.lifecycle is not TaskLifecycle.RECOVERY_REQUIRED:
                return {"status": existing.lifecycle.value, "task_id": str(existing.task_id)}
            # This plan's Task only, never another of its kind (V529).
            dispatcher.resume(
                {CAPTURE_TASK_KIND: ResearchInputCaptureCommand(self)},
                only_task_id=existing.task_id,
            )
            return {"status": "REUSED_IN_FLIGHT", "task_id": str(existing.task_id)}
        kept = self._plans.runnable(plan_hash)
        if kept is None:
            raise ValueError("research_input.preview_required")
        sent = dispatcher.submit(ResearchInputCaptureCommand(self, kept))
        return {
            "status": sent.disposition,
            "task_id": str(sent.task_id) if sent.task_id else None,
            "lifecycle": sent.lifecycle,
            "failure_code": sent.refusal_detail,
        }

    def admit(self, plan: ResearchInputCapturePlan) -> CommandAdmission:
        """Validate source and parent under mutation ownership before exact capture admission.

        Args:
            plan: Explicit sealed capture plan.

        Returns:
            Exact retained or newly admitted task.

        Raises:
            ValueError: Cleanup is pending, source/parent changed or another task must finish or
                recover.
        """
        with self.session.mutation_gate.hold():
            require_no_pending_cleanup(self.session.workspace)
            self._source_and_parent(plan)
            envelope, goal, workflow = _task_contract(plan)
            registry = self.session.task_control_registry
            same = next(
                (
                    t
                    for t in registry.tasks()
                    if t.input == envelope and t.lifecycle is not TaskLifecycle.CANCELLED
                ),
                None,
            )
            if same is not None:
                return CommandAdmission(task_id=same.task_id, lifecycle=same.lifecycle.value)
            if any(
                t.lifecycle
                not in {TaskLifecycle.SUCCEEDED, TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED}
                for t in registry.tasks()
            ):
                raise ValueError("research_input.finish_or_recover_existing_task")
            task = registry.admit(
                input_envelope=envelope, goal=goal, plan=workflow, observed_at=self.clock()
            ).record
            return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Bind capture schema, workflow, exact source key and installed execution identity.

        Args:
            task: Exact retained capture task.

        Returns:
            Deterministic execution compatibility declaration.
        """
        plan = self._of(task)
        self._require(plan)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash(schema_structure(ResearchInputCapturePlan)),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=plan.source.source_key,
            framework_identity_hash=self.session.execution_identity(plan.implementation_hash),
        )

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one exact admitted local input capture task.

        Args:
            task_id: Exact retained task.
            expected_task_hash: Optional optimistic task identity.
        """
        task = self.session.task_control_registry.task(task_id)
        self.session.execute_admitted(task, self, self.clock, expected_task_hash)

    @staticmethod
    def _evidence(value: ResearchInputRevision) -> tuple[TaskEvidence, ...]:
        return (
            TaskEvidence(
                evidence_kind="research_input.publication",
                reference=f"playpen://research-inputs/publications/{value.receipt_hash}",
                content_hash=value.receipt_hash,
            ),
        )

    def execute_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
    ) -> StageExecutionResult:
        """Publish an exact verified local input bundle and only then its revision receipt.

        Recovery reopens and verifies the already published bundle. Publication leaves workspace
        defaults and current-universe bindings unchanged.

        Args:
            task: Exact retained capture task.
            execution: Current execution declaration.
            work_item: Exact installed capture stage.

        Returns:
            Exact ready receipt evidence, cancellation before publication or bounded named refusal.
        """
        del execution
        if work_item.stage_id != STAGE:
            raise ValueError("research_input.stage_unknown")
        plan = self._of(task)
        self._require(plan)
        try:
            revision = self.revisions.for_task(task.task_id)
            if revision is None:
                require_no_pending_cleanup(self.session.workspace)
                self._source_and_parent(plan)
                if (
                    self.session.task_control_registry.task(task.task_id).lifecycle
                    is TaskLifecycle.CANCEL_REQUESTED
                ):
                    return StageExecutionResult(StageDisposition.CANCELLED)
                binding = publish_prepared_factor_inputs(
                    self.session,
                    task.task_id,
                    input_id=plan.input_id,
                    panel_snapshot_hash=plan.source.panel_snapshot_hash,
                    completed_at=self.clock(),
                    bind_configuration=False,
                )
                self._source_and_parent(plan)
                revision = ResearchInputRevision.create(
                    input_id=plan.input_id,
                    anchor_binding_hash=plan.anchor_binding_hash,
                    prior_binding_hash=plan.prior_binding_hash,
                    binding_hash=binding.binding_hash,
                    previous_publication_hash=plan.previous_publication_hash,
                    source=plan.source,
                    task_id=task.task_id,
                    task_input_hash=task.input.input_hash,
                    published_at=self.clock(),
                )
                # The receipt is written only after the publisher has read
                # the bound bundle whole; a bundle damaged between the
                # binding and here leaves no receipt and blocks the Task.
                self.revisions.publish(revision)
            else:
                # Recovering a Task whose revision already exists: its bundle
                # is read whole again before the stage is ready.
                read_factor_bundle(self.session.workspace, revision.binding_hash)
            return StageExecutionResult(StageDisposition.READY, evidence=self._evidence(revision))
        except ChildStartFailed:
            raise
        except (ValueError, OSError, RuntimeError) as error:
            return StageExecutionResult(
                StageDisposition.BLOCKED,
                failure_code=str(getattr(error, "failure_code", str(error)))[:120],
            )

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Require exact task revision evidence and read its bound bundle whole.

        Args:
            task: Exact retained capture task.
            execution: Current execution declaration.
            work_item: Exact capture stage.
            evidence: Declared revision receipt evidence.

        Returns:
            Unchanged verified evidence.

        Raises:
            ValueError: Stage/publication/evidence or plan authority differs.
        """
        del execution
        self._require(self._of(task))
        value = self.revisions.for_task(task.task_id)
        if work_item.stage_id != STAGE or value is None or self._evidence(value) != evidence:
            raise ValueError("research_input.publication_evidence_mismatch")
        read_factor_bundle(self.session.workspace, value.binding_hash)
        return evidence

    def readback(self, task_id: UUID) -> dict[str, object]:
        """Read retained capture task state and its exact associated revision receipt.

        Args:
            task_id: Exact retained task.

        Returns:
            Lifecycle, failure code and revision publication when present.
        """
        task = self.session.task_control_registry.task(task_id)
        self._of(task)
        value = self.revisions.for_task(task_id)
        return {
            "status": task.lifecycle.value,
            "task_id": str(task_id),
            "failure_code": task.failure_code,
            "research_input_id": value.input_id if value else None,
            "input_binding_hash": value.binding_hash if value else None,
            "publication": value.model_dump(mode="json") if value else None,
            "next_requests": {
                "controls": {
                    "operation": "EXPERIMENT_CONTROLS",
                    "research_input_id": value.input_id,
                    "input_binding_hash": value.binding_hash,
                }
            }
            if value
            else {},
        }


@dataclass
class ResearchInputCaptureCommand:
    """Dispatch one explicit local input capture plan through its deterministic owner."""

    application: ResearchInputCaptureApplication
    plan: ResearchInputCapturePlan | None = None
    command_kind: str = CAPTURE_TASK_KIND

    def admit(self) -> CommandAdmission:
        """Require the exact local input capture plan before task admission.

        Returns:
            Deterministic task admission.

        Raises:
            ValueError: Required plan is absent or its declared admission checks refuse.
        """
        if self.plan is None:
            raise ValueError("research_input.preview_required")
        return self.application.admit(self.plan)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one exact task admitted for this local input capture plan.

        Args:
            task_id: Exact retained task.
            expected_task_hash: Optional optimistic task identity.
        """
        self.application.execute(task_id, expected_task_hash=expected_task_hash)
