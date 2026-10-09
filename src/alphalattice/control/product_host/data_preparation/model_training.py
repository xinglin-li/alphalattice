"""Task-controlled preparation of research-only frozen-component training inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    manifest_fields_hash,
    read_research_workspace_manifest,
    resolve_workspace_model_lifecycle,
    update_research_workspace_manifest,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    factor_input_paths,
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.input_revisions import (
    ResearchInputRevisions,
)
from alphalattice.control.product_host.research_authoring.model_training import (
    FROZEN_TRAINING_FACTOR_AXIS_HASH,
    ComponentTrainingPreparationReceipt,
    model_lifecycle_disclosure,
    prepare_component_training_inputs,
)
from alphalattice.control.product_host.storage.inventory import require_storage_capacity
from alphalattice.control.product_host.storage.plan_previews import (
    PLAN_PREVIEWS_DIRECTORY,
    PreviewRegistry,
)
from alphalattice.control.product_host.storage.retention import require_no_pending_cleanup
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
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.interface.local_application.dispatcher import (
    CommandAdmission,
    LocalBackgroundDispatcher,
)
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
)
from alphalattice.investment.alpha_research.scores.model_renewal import read_lifecycle_admission
from alphalattice.investment.alpha_research.scores.product_lifecycle import (
    DEFAULT_MODEL_LIFECYCLE,
    AlphaModelLifecycleRecipe,
    ModelLifecycle,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash

PLAN_FIELDS = ("experiment_inputs",)
"""The manifest fields a plan reads: the research input it selects its binding from; its own
publication writes `model_training_inputs`, merged into the manifest as it stands. A plan
binds them, never the whole manifest, so a publication of fields it does not read leaves it
applicable (OW10)."""

TASK_KIND = "model_training_input_preparation"
STAGES = ("prepare_component_training_inputs", "bind_research_training_sources")
RECEIPTS = "lifecycle-input-preparations"
PUBLICATIONS = "lifecycle-input-publications"


IMPLEMENTATION_ROLE = "product_host.research_component_training_input"
"""The role a move of this implementation is recorded under in `config/identity-successors.json`,
so a plan sealed under its predecessor stays current (binding plan R1, LAWS.md ID1)."""


def _implementation() -> str:
    return source_rule_closure_hash(
        root=resolve_playpen_root(Path(__file__)),
        semantic_owner="product_host",
        numerical_role="RESEARCH_COMPONENT_TRAINING_INPUT",
        tracked_paths=tuple(
            "src/alphalattice/" + path
            for path in (
                "control/product_host/data_preparation/model_training.py",
                "control/product_host/research_authoring/model_training.py",
                "control/product_host/composition/strategy_score_inputs.py",
                "investment/alpha_research/scores/model_renewal.py",
                "investment/alpha_research/experiments/development_artifacts.py",
                "investment/alpha_research/inputs/frozen_price_volume.py",
            )
        ),
    )


class ModelTrainingInputPlan(BaseModel):  # type: ignore[misc]
    """Seal model input preparation scope and source axes."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    workspace_id: str
    workspace_manifest_hash: str
    input_id: str
    input_binding_hash: str
    component_ids: tuple[str, ...]
    listing_count: int
    source_session_count: int
    implementation_hash: str
    model_lifecycle: ModelLifecycle = Field(
        default="FULL", exclude_if=lambda value: value == "FULL"
    )
    """The lifecycle the admission binds; a FULL plan keeps the hash it had before LIGHT."""
    plan_hash: str

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal explicit model input preparation scope and source axes.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical plan_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        sealed = {k: v for k, v in values.items() if (k, v) != ("model_lifecycle", "FULL")}
        return cls(**values, plan_hash=canonical_hash(sealed))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def verify(self) -> Self:
        """Require exact canonical plan_hash over the declared fields.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Serialized fields differ from plan_hash.
        """
        if self.plan_hash != canonical_hash(self.model_dump(mode="json", exclude={"plan_hash"})):
            raise ValueError("model_training.plan_invalid")
        return self


class ModelTrainingInputPublication(BaseModel):  # type: ignore[misc]
    """Seal prepared support receipt and exact manifest transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    plan_hash: str
    preparation_receipt_hash: str
    previous_manifest_hash: str
    result_manifest_hash: str
    publication_hash: str

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal explicit prepared support receipt and exact manifest transition.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical publication_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return cls(**values, publication_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def verify(self) -> Self:
        """Require exact canonical publication_hash over the declared fields.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Serialized fields differ from publication_hash.
        """
        if self.publication_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"publication_hash"})
        ):
            raise ValueError("model_training.publication_invalid")
        return self


def _task_contract(plan: ModelTrainingInputPlan, caller: str):  # type: ignore[no-untyped-def]
    envelope = TaskInputEnvelope.create(
        task_kind=TASK_KIND,
        input_schema_id="model-training-input",
        payload={"plan": plan.model_dump(mode="json"), "caller": caller},
    )
    goal = ResearchGoal.create(
        goal_kind="PREPARE_MODEL_TRAINING_INPUT",
        input_hash=envelope.input_hash,
        deliverable_kind="ComponentTrainingPreparationReceipt",
        summary="Prepare frozen-component Features, targets and supported quarters; "
        "no model fits or strategy activation.",
    )
    workflow = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash(STAGES),
        verifier_catalog_hash=canonical_hash(STAGES),
        work_items=tuple(
            WorkItemDefinition.create(
                stage_id=stage, dependency_ids=STAGES[:index], verifier_id=f"model_training.{stage}"
            )
            for index, stage in enumerate(STAGES)
        ),
    )
    return envelope, goal, workflow


class ModelTrainingInputApplication:
    """Own recorded component input preparation and manifest binding without model fits."""

    task_kind = TASK_KIND
    replans = (
        TaskReplan(
            task_kind=TASK_KIND,
            preview="MODEL_TRAINING_INPUT_PLAN",
            admitting="MODEL_TRAINING_INPUT_PREPARE",
        ),
    )
    """The re-plan of the Task kind this owner admits, which the recovery view offers."""

    def __init__(self, session: WorkspaceApplicationSession, *, clock: Callable[[], datetime]):
        """Wire retained task session, exact Alpha store and bounded input-plan previews.

        Args:
            session: Retained workspace writer/task session.
            clock: Explicit observed-time source.
        """
        self.session, self.clock = session, clock
        self.store = AlphaCurrentArtifactStore(session.workspace / "artifacts")
        # Every plan an answer named, by its hash, sealed on disk until it expires: a run
        # from any of them, after a restart too, reopens it and checks it again.
        self._plans: PreviewRegistry[ModelTrainingInputPlan] = PreviewRegistry(
            model=ModelTrainingInputPlan,
            clock=self.clock,
            capacity=16,
            root=self.session.workspace / "runtime" / PLAN_PREVIEWS_DIRECTORY / "model-training",
        )

    def plan(
        self,
        input_id: str,
        binding_hash: str | None,
        component_id: str,
        model_lifecycle: ModelLifecycle = DEFAULT_MODEL_LIFECYCLE,
    ) -> dict[str, object]:
        """Select an admitted component/input and retain its exact preparation preview.

        Args:
            input_id: Explicit research input.
            binding_hash: Optional exact input revision.
            component_id: Installed component selection.
            model_lifecycle: The lifecycle to prepare: the light default, or FULL by name.

        Returns:
            Exact plan, source axes/counts and declared preparation request; model_fit_calls is
            zero.

        Raises:
            ValueError: Component selection, cleanup state or admitted input is invalid.
        """
        if component_id not in {"G2_R0_TREND", "G6_R0_FAST_REBOUND"}:
            raise ValueError("model_training.component_selection_invalid")
        require_no_pending_cleanup(self.session.workspace)
        binding = ResearchInputRevisions(self.session).select(input_id, binding_hash)
        manifest = read_research_workspace_manifest(self.session.workspace)
        bundle = read_factor_bundle(self.session.workspace, binding.binding_hash)
        _, artifacts = factor_input_paths(self.session.workspace, binding.binding_hash)
        resolver = ArtifactResolver(artifacts)
        listing_count = len(
            FeaturePanelReader(resolver).listing_ids(
                resolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
            )
        )
        plan = ModelTrainingInputPlan.create(
            workspace_id=manifest.workspace_id,
            workspace_manifest_hash=manifest_fields_hash(manifest, PLAN_FIELDS),
            input_id=input_id,
            input_binding_hash=binding.binding_hash,
            component_ids=(component_id,),
            listing_count=listing_count,
            source_session_count=len(bundle.sessions),
            implementation_hash=_implementation(),
            model_lifecycle=model_lifecycle,
        )
        self._plans.remember(plan)
        rule = AlphaModelLifecycleRecipe.named(
            INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component_id), model_lifecycle
        )
        return {
            "status": "PLANNED",
            "plan_hash": plan.plan_hash,
            "input_binding_hash": binding.binding_hash,
            "component_ids": (component_id,),
            "training_factor_axis_hash": FROZEN_TRAINING_FACTOR_AXIS_HASH,
            "source_start": str(bundle.sessions[0]),
            "source_end": str(bundle.sessions[-1]),
            "source_session_count": len(bundle.sessions),
            "source_listing_count": listing_count,
            "model_fit_calls": 0,
            "model_lifecycle": model_lifecycle_disclosure(model_lifecycle, rule),
            "claim": "INPUT_PREPARATION_ONLY; complete model support is resolved before any fit",
            "next_requests": {
                "prepare": {
                    "operation": "MODEL_TRAINING_INPUT_PREPARE",
                    "experiment_plan_hash": plan.plan_hash,
                }
            },
        }

    def _of(self, task: TaskRecord) -> ModelTrainingInputPlan:
        plan: ModelTrainingInputPlan = ModelTrainingInputPlan.model_validate(
            task.input.payload["plan"]
        )
        if task.task_kind != TASK_KIND or (task.input, task.goal, task.plan) != _task_contract(
            plan, str(task.input.payload["caller"])
        ):
            raise ValueError("model_training.task_contract_invalid")
        return plan

    def replan_request(self, task: TaskRecord) -> dict[str, object]:
        """Replan the validated Task's exact training component and input revision.

        Args:
            task: Durable Task whose declaration supplies the next request.

        Returns:
            The filled single-component training-input planning request.

        Raises:
            ValueError: The Task does not validate or selects more than one component.
        """
        plan = self._of(task)
        if len(plan.component_ids) != 1:
            raise ValueError("model_training.component_selection_invalid")
        return {
            "operation": "MODEL_TRAINING_INPUT_PLAN",
            "component_id": plan.component_ids[0],
            "research_input_id": plan.input_id,
            "input_binding_hash": plan.input_binding_hash,
            # Named always: an omitted lifecycle plans the light default.
            "model_lifecycle": plan.model_lifecycle,
        }

    def _require(self, plan: ModelTrainingInputPlan) -> None:
        current = read_research_workspace_manifest(self.session.workspace)
        if plan.workspace_id != current.workspace_id or not is_current(
            IMPLEMENTATION_ROLE, plan.implementation_hash, _implementation()
        ):
            raise ValueError("model_training.execution_changed")
        # Its own publication writes `model_training_inputs`, which it does not read, so a
        # plan it published under still applies.
        if manifest_fields_hash(current, PLAN_FIELDS) != plan.workspace_manifest_hash:
            raise ValueError("model_training.configuration_changed")

    def _publication(self, plan_hash: str) -> ModelTrainingInputPublication | None:
        matches = []
        for path in (self.store.root / "current" / PUBLICATIONS).glob("*.json"):
            value = self.store._load(
                PUBLICATIONS, path.stem, "publication_hash", ModelTrainingInputPublication
            )
            if value.plan_hash == plan_hash:
                matches.append(value)
        if len(matches) > 1:
            raise ValueError("model_training.publication_ambiguous")
        return matches[0] if matches else None

    def owns_published_manifest(self, identity: str) -> bool:
        """Check validated preparation publications for one exact result manifest.

        Args:
            identity: Exact manifest identity.

        Returns:
            Whether a stored model input publication binds this result manifest.
        """
        return any(
            self.store._load(
                PUBLICATIONS, path.stem, "publication_hash", ModelTrainingInputPublication
            ).result_manifest_hash
            == identity
            for path in (self.store.root / "current" / PUBLICATIONS).glob("*.json")
        )

    def replan_requests(self, plan_hash: str) -> dict[str, object]:
        """Bind each training component to the input of a verified retained plan.

        Args:
            plan_hash: The refused plan's exact hash.

        Returns:
            Bound re-plans, or no offer when the source plan cannot be verified.
        """
        kept = self._plans.get(plan_hash)
        if kept is None:
            return {}
        return {
            f"replan:{component}": {
                "operation": "MODEL_TRAINING_INPUT_PLAN",
                "component_id": component,
                "research_input_id": kept.plan.input_id,
                "input_binding_hash": kept.plan.input_binding_hash,
                "model_lifecycle": kept.plan.model_lifecycle,
            }
            for component in kept.plan.component_ids
        }

    def prepare(
        self, plan_hash: str, *, caller: str, dispatcher: LocalBackgroundDispatcher
    ) -> dict[str, object]:
        """Reuse verified prepared support, recover its exact task or dispatch its retained preview.

        Args:
            plan_hash: Exact preparation preview or task plan.
            caller: Declared operation caller.
            dispatcher: Bounded local dispatcher.

        Returns:
            Verified exact reuse, in-flight recovery or explicit submission metadata.

        Raises:
            ValueError: Preview is absent or retained authority/preparation cannot be reopened.
        """
        existing = next(
            (
                task
                for task in self.session.task_control_registry.tasks()
                if task.task_kind == TASK_KIND
                and self._of(task).plan_hash == plan_hash
                and task.lifecycle is not TaskLifecycle.CANCELLED
            ),
            None,
        )
        if existing is not None:
            if existing.lifecycle is TaskLifecycle.SUCCEEDED:
                for binding in self._preparation(existing).bindings:
                    resolve_workspace_model_lifecycle(
                        self.session.workspace,
                        component_id=binding.component_id,
                        training_authority_hash=binding.authority_hash,
                    )
                return {
                    **self.readback(existing.task_id),
                    "status": "REUSED_EXACT",
                    "task_id": None,
                    "publication_task_id": str(existing.task_id),
                }
            if existing.lifecycle is TaskLifecycle.BLOCKED:
                self._require(self._of(existing))
                self.session.task_control_registry.mark_recovery_required(
                    task_id=existing.task_id,
                    failure_code=existing.failure_code or "model_training.retry_requested",
                    observed_at=self.clock(),
                    allow_blocked=True,
                )
            elif existing.lifecycle is not TaskLifecycle.RECOVERY_REQUIRED:
                return self.readback(existing.task_id)
            # This plan's Task only, never another of its kind.
            dispatcher.resume(
                {TASK_KIND: ModelTrainingInputCommand(self)}, only_task_id=existing.task_id
            )
            return {"status": "REUSED_IN_FLIGHT", "task_id": str(existing.task_id)}
        plan = self._plans.runnable(plan_hash)
        if plan is None:
            raise ValueError("model_training.preview_required")
        sent = dispatcher.submit(ModelTrainingInputCommand(self, plan, caller))
        return sent.answer(
            next_requests={
                "readback": {
                    "operation": "MODEL_TRAINING_INPUT_READBACK",
                    "task_id": str(sent.task_id),
                }
            }
            if sent.task_id
            else {},
        )

    def admit(self, plan: ModelTrainingInputPlan, caller: str) -> CommandAdmission:
        """Require exact source and exclusive preparation ownership under the mutation gate.

        Args:
            plan: Explicit sealed model input plan.
            caller: Declared submitting caller.

        Returns:
            Exact reused or newly admitted preparation task.

        Raises:
            ValueError: Plan/input/cleanup is invalid or another task must finish or recover.
        """
        with self.session.mutation_gate.hold():
            self._require(plan)
            require_no_pending_cleanup(self.session.workspace)
            read_factor_bundle(self.session.workspace, plan.input_binding_hash)
            envelope, goal, workflow = _task_contract(plan, caller)
            registry = self.session.task_control_registry
            for existing in registry.tasks():
                if (
                    existing.task_kind == TASK_KIND
                    and self._of(existing).plan_hash == plan.plan_hash
                    and existing.lifecycle is not TaskLifecycle.CANCELLED
                ):
                    return CommandAdmission(
                        task_id=existing.task_id, lifecycle=existing.lifecycle.value
                    )
                if existing.lifecycle not in {
                    TaskLifecycle.SUCCEEDED,
                    TaskLifecycle.BLOCKED,
                    TaskLifecycle.CANCELLED,
                }:
                    raise ValueError(
                        f"model_training.finish_or_recover_existing_task:{existing.task_id}"
                    )
            task = registry.admit(
                input_envelope=envelope, goal=goal, plan=workflow, observed_at=self.clock()
            ).record
            return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Bind exact preparation schema, workflow, training factor axis and execution identity.

        Args:
            task: Exact retained preparation task.

        Returns:
            Deterministic execution compatibility declaration.
        """
        plan = self._of(task)
        self._require(plan)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash(schema_structure(ModelTrainingInputPlan)),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=FROZEN_TRAINING_FACTOR_AXIS_HASH,
            framework_identity_hash=self.session.execution_identity(plan.implementation_hash),
        )

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one exact admitted model input preparation task.

        Args:
            task_id: Exact retained task.
            expected_task_hash: Optional optimistic task identity.
        """
        task = self.session.task_control_registry.task(task_id)
        self.session.execute_admitted(task, self, self.clock, expected_task_hash)

    def _preparation(self, task: TaskRecord) -> ComponentTrainingPreparationReceipt:
        item = next(
            v
            for v in self.session.task_control_registry.work_items(task.task_id)
            if v.stage_id == STAGES[0]
        )
        if len(item.evidence) != 1:
            raise ValueError("model_training.preparation_evidence_missing")
        return self.store._load(
            RECEIPTS,
            item.evidence[0].content_hash,
            "receipt_hash",
            ComponentTrainingPreparationReceipt,
        )

    def execute_stage(
        self, *, task: TaskRecord, execution: TaskExecution, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        """Prepare bounded component support or publish its exact manifest transition.

        Capacity is charged by the storage owner. Manifest publication preserves concurrent strategy
        installation and requires the exact recorded predecessor or result head; preparation
        performs no model fits.

        Args:
            task: Exact retained preparation task.
            execution: Current execution declaration.
            work_item: Exact support/publication stage.

        Returns:
            Ready exact receipt/publication evidence, safe-checkpoint cancellation or bounded named
            refusal.
        """
        del execution
        plan = self._of(task)
        try:
            self._require(plan)
            if work_item.stage_id == STAGES[0]:
                receipt = prepare_component_training_inputs(
                    workspace=self.session.workspace,
                    input_binding_hash=plan.input_binding_hash,
                    component_ids=plan.component_ids,
                    observed_at=self.clock(),
                    cancelled=lambda: (
                        self.session.task_control_registry.task(task.task_id).lifecycle
                        is TaskLifecycle.CANCEL_REQUESTED
                    ),
                    capacity=lambda amount: require_storage_capacity(
                        self.session.workspace,
                        additional_bytes=amount,
                    ),
                    lifecycle=plan.model_lifecycle,
                )
                category, identity = RECEIPTS, receipt.receipt_hash
            elif work_item.stage_id == STAGES[1]:
                receipt = self._preparation(task)
                recorded = [self._publication(plan.plan_hash)]

                # Apply to the manifest as it stands, keeping a concurrent strategy installation.
                def publish(current: ResearchWorkspaceManifest) -> ResearchWorkspaceManifest:
                    known = {v.authority_hash: v for v in current.model_training_inputs or ()}
                    known.update({v.authority_hash: v for v in receipt.bindings})
                    updated = current.with_bindings(model_training_inputs=tuple(known.values()))
                    publication = recorded[0]
                    if publication is None:
                        publication = ModelTrainingInputPublication.create(
                            plan_hash=plan.plan_hash,
                            preparation_receipt_hash=receipt.receipt_hash,
                            previous_manifest_hash=current.manifest_hash,
                            result_manifest_hash=updated.manifest_hash,
                        )
                        self.store._publish(PUBLICATIONS, publication, "publication_hash")
                        recorded[0] = publication
                    if current.manifest_hash == publication.previous_manifest_hash:
                        if updated.manifest_hash != publication.result_manifest_hash:
                            raise ValueError("model_training.publication_binding_invalid")
                        return updated
                    if current.manifest_hash != publication.result_manifest_hash:
                        raise ValueError("model_training.publication_head_changed")
                    return current

                update_research_workspace_manifest(
                    self.session.workspace, publish, gate=self.session.mutation_gate
                )
                published = recorded[0]
                assert published is not None
                category, identity = PUBLICATIONS, published.publication_hash
            else:
                raise ValueError("model_training.stage_unknown")
            return StageExecutionResult(
                StageDisposition.READY,
                evidence=(
                    TaskEvidence(
                        evidence_kind=f"model_training.{work_item.stage_id}",
                        reference=self.store.uri("current/" + category, identity),
                        content_hash=identity,
                    ),
                ),
            )
        except (ValueError, OSError, RuntimeError) as error:
            if str(error) == "model_training.cancelled_at_safe_checkpoint":
                return StageExecutionResult(StageDisposition.CANCELLED)
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
        """Require exact task-bound support and the published manifest/lifecycle authorities.

        Args:
            task: Exact retained task.
            execution: Current execution declaration.
            work_item: Exact preparation/publication stage.
            evidence: Single-content declared stage evidence.

        Returns:
            Unchanged verified evidence.

        Raises:
            ValueError: Stage, source/component receipt, publication identity or current manifest
                binding differs.
        """
        del execution
        plan = self._of(task)
        if len(evidence) != 1:
            raise ValueError("model_training.stage_evidence_invalid")
        category = RECEIPTS if work_item.stage_id == STAGES[0] else PUBLICATIONS
        if evidence != (
            TaskEvidence(
                evidence_kind=f"model_training.{work_item.stage_id}",
                reference=self.store.uri("current/" + category, evidence[0].content_hash),
                content_hash=evidence[0].content_hash,
            ),
        ):
            raise ValueError("model_training.stage_evidence_invalid")
        if work_item.stage_id == STAGES[0]:
            receipt = self.store._load(
                RECEIPTS,
                evidence[0].content_hash,
                "receipt_hash",
                ComponentTrainingPreparationReceipt,
            )
            if (
                receipt.input_binding_hash != plan.input_binding_hash
                or tuple(v.component_id for v in receipt.bindings) != plan.component_ids
            ):
                raise ValueError("model_training.preparation_binding_invalid")
        elif work_item.stage_id == STAGES[1]:
            publication = self._publication(plan.plan_hash)
            if (
                publication is None
                or publication.publication_hash != evidence[0].content_hash
                or read_research_workspace_manifest(self.session.workspace).manifest_hash
                != publication.result_manifest_hash
            ):
                raise ValueError("model_training.publication_readback_failed")
            for binding in self._preparation(task).bindings:
                resolve_workspace_model_lifecycle(
                    self.session.workspace,
                    component_id=binding.component_id,
                    training_authority_hash=binding.authority_hash,
                )
        else:
            raise ValueError("model_training.stage_unknown")
        return evidence

    def readback(self, task_id: UUID) -> dict[str, object]:
        """Read exact prepared component sources and their declared lifecycle support.

        Args:
            task_id: Exact retained preparation task.

        Returns:
            Task state, receipt, source handles/support/counts and zero fits during preparation.

        Raises:
            ValueError: A succeeded task lacks its exact publication.
        """
        task = self.session.task_control_registry.task(task_id)
        plan = self._of(task)
        publication = self._publication(plan.plan_hash)
        if task.lifecycle is TaskLifecycle.SUCCEEDED and publication is None:
            raise ValueError("model_training.publication_missing")
        receipt = (
            self.store._load(
                RECEIPTS,
                publication.preparation_receipt_hash,
                "receipt_hash",
                ComponentTrainingPreparationReceipt,
            )
            if publication
            else None
        )
        sources = []
        for binding in receipt.bindings if receipt else ():
            admission = read_lifecycle_admission(
                self.session.workspace / binding.authority_relative_path,
                expected_hash=binding.authority_hash,
            )
            sources.append(
                {
                    "source_handle": binding.source_handle,
                    "component_id": binding.component_id,
                    "formation_start": str(admission.formation_start),
                    "formation_end": str(admission.formation_end),
                    "required_vintages": admission.fit_vintages,
                    "fit_upper_bound": admission.maximum_fit_attempts,
                    "model_feature_ids": admission.component.ordered_feature_ids,
                    "lifecycle": admission.lifecycle.model_dump(mode="json"),
                    "model_fit_calls_in_preparation": 0,
                }
            )
        return {
            "status": task.lifecycle.value,
            "task_id": str(task_id),
            "failure_code": task.failure_code,
            "receipt": receipt.model_dump(mode="json") if receipt else None,
            "source_handles": [v.source_handle for v in receipt.bindings] if receipt else [],
            "sources": sources,
            "source_session_count": plan.source_session_count,
            "source_listing_count": plan.listing_count,
            "next_requests": {
                "readback": {"operation": "MODEL_TRAINING_INPUT_READBACK", "task_id": str(task_id)},
                # Prepared inputs go on to each component's lifecycle study, bound to the input
                # they were prepared on, as the strategy's controls offer it.
                **(
                    {
                        f"controls:{component}": {
                            "operation": "EXPERIMENT_CONTROLS",
                            "research_input_id": plan.input_id,
                            "input_binding_hash": plan.input_binding_hash,
                            "experiment_kind": "alpha.model-development",
                            "component_id": component,
                        }
                        for component in plan.component_ids
                    }
                    if task.lifecycle is TaskLifecycle.SUCCEEDED
                    else {}
                ),
            },
            "claim": "RECORDED_INPUT_PREPARATION_ONLY_NO_MODEL_FITS",
        }


@dataclass
class ModelTrainingInputCommand:
    """Dispatch one explicit model input preparation plan and declared caller."""

    application: ModelTrainingInputApplication
    plan: ModelTrainingInputPlan | None = None
    caller: str = "EXTERNAL_AUTOMATION"
    command_kind: str = TASK_KIND

    def admit(self) -> CommandAdmission:
        """Require the exact model input preparation preview before deterministic admission.

        Returns:
            Task admission from the owning application.

        Raises:
            ValueError: Exact preview is absent or declared admission checks refuse.
        """
        if self.plan is None:
            raise ValueError("model_training.preview_required")
        return self.application.admit(self.plan, self.caller)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one exact task admitted for model input preparation.

        Args:
            task_id: Exact retained task.
            expected_task_hash: Optional optimistic task identity.
        """
        self.application.execute(task_id, expected_task_hash=expected_task_hash)
