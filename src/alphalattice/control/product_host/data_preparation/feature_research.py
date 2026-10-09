"""Task-controlled local Formula materialization, distinct from downstream admission."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    factor_input_paths,
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.feature_materialization import (
    SOURCE_PROJECTION_ROLE,
    feature_source_projection_hash,
    feature_value_sources,
    materialize_feature_columns,
)
from alphalattice.control.product_host.research_authoring.feature_preprocessing import (
    prepare_feature_overlay,
    resolve_prepared_feature_input,
    verify_feature_preparation,
)
from alphalattice.control.product_host.research_authoring.feature_research import (
    ResearchFeatureDefinitions,
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
from alphalattice.foundation.feature_engine.catalog.research import research_feature_execution_spec
from alphalattice.foundation.feature_engine.catalog.research_values import (
    PREPARATION_CATEGORY,
    RECEIPT_CATEGORY,
    ResearchFeatureMaterialization,
    ResearchFeaturePreparation,
    ResearchFormulaColumn,
    ResearchFormulaValues,
)
from alphalattice.foundation.feature_engine.panels.development_input import (
    ResolvedDevelopmentFeatureInput,
)
from alphalattice.foundation.feature_engine.producers.factors.specifications import (
    FeatureFormulaSpecificationError,
    build_research_formula_specification,
)
from alphalattice.interface.local_application.dispatcher import (
    CommandAdmission,
    LocalBackgroundDispatcher,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash

TASK_KIND = "research_feature_materialization"
STAGE = "materialize_local_formula_columns"
PREPARATION_STAGE = "prepare_local_formula_overlay"


def _preparing(payload: dict[str, Any]) -> bool:
    if payload.get("purpose") not in {None, "PREPROCESS_VALUES"}:
        raise ValueError("feature_research.task_purpose_invalid")
    return payload.get("purpose") == "PREPROCESS_VALUES"


def implementation_role(*, preparation: bool = False) -> str:
    """Name the identity role governing exact feature build or preparation closure changes.

    The role a move of this closure is recorded under in `config/identity-successors.json`,
    so a build sealed under its predecessor stays current (binding plan, B14).
    """
    return "product_host.research_feature_materialization" + (".prepared" if preparation else "")


def implementation_hash(*, preparation: bool = False) -> str:
    """Read the installed semantic rule closure for raw feature builds or preparation.

    Args:
        preparation: Whether to include the admitted preprocessing preparation owners.

    Returns:
        Exact feature materialization implementation identity; no closure list is changed.
    """
    return source_rule_closure_hash(
        root=resolve_playpen_root(Path(__file__)),
        semantic_owner="product_host",
        numerical_role="RESEARCH_FEATURE_MATERIALIZATION",
        tracked_paths=tuple(
            "src/alphalattice/" + p
            for p in (
                "control/product_host/data_preparation/feature_research.py",
                "control/product_host/research_authoring/feature_materialization.py",
                "control/product_host/research_authoring/feature_research.py",
                "foundation/feature_engine/catalog/research_values.py",
                "foundation/feature_engine/panels/closure_artifacts.py",
            )
        )
        + (
            tuple(
                "src/alphalattice/" + p
                for p in (
                    "control/product_host/research_authoring/feature_preprocessing.py",
                    "control/product_host/research_authoring/authority.py",
                    "foundation/feature_engine/panels/development_overlay.py",
                    "foundation/feature_engine/producers/factors/specifications.py",
                    "foundation/feature_engine/producers/preprocessing/catalog.py",
                    "foundation/feature_engine/producers/preprocessing/adapters.py",
                    "foundation/feature_engine/producers/preprocessing/robust_cross_section.py",
                    "foundation/feature_engine/producers/preprocessing/development.py",
                )
            )
            if preparation
            else ()
        ),
    )


def _contract(payload: dict[str, Any]) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    preparing = _preparing(payload)
    stages = (STAGE, PREPARATION_STAGE) if preparing else (STAGE,)
    envelope = TaskInputEnvelope.create(
        task_kind=TASK_KIND, input_schema_id="research-feature-materialization", payload=payload
    )
    goal = ResearchGoal.create(
        goal_kind="PREPARE_LOCAL_FORMULA_OVERLAY" if preparing else "MATERIALIZE_LOCAL_FORMULAS",
        input_hash=envelope.input_hash,
        deliverable_kind="ResearchFeaturePreparation"
        if preparing
        else "ResearchFeatureMaterialization",
        summary=(
            "Build or reuse local columns and their development preprocessing; "
            "no global catalog or Factor/Alpha admission."
            if preparing
            else "Build or reuse source-bound local Formula columns; "
            "no global catalog or Factor/Alpha admission."
        ),
    )
    workflow = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash(stages),
        verifier_catalog_hash=canonical_hash(stages),
        work_items=(
            WorkItemDefinition.create(
                stage_id=STAGE,
                dependency_ids=(),
                verifier_id="feature_research.verified_values",
            ),
        )
        + (
            (
                WorkItemDefinition.create(
                    stage_id=PREPARATION_STAGE,
                    dependency_ids=(STAGE,),
                    verifier_id="feature_research.verified_overlay",
                ),
            )
            if preparing
            else ()
        ),
    )
    return envelope, goal, workflow


class ResearchFeatureBuildApplication:
    """Own admitted feature definition materialization and optional prepared research overlays."""

    task_kind = TASK_KIND
    replans = (
        TaskReplan(
            task_kind=TASK_KIND, preview="FEATURE_CATALOG_PLAN", admitting="FEATURE_CATALOG_BUILD"
        ),
    )
    """The re-plan of the Task kind this owner admits, which the recovery view offers."""

    def __init__(self, session: WorkspaceApplicationSession, *, clock: Callable[[], datetime]):
        """Wire feature build ownership to the retained workspace task session.

        Args:
            session: Retained workspace writer/task session.
            clock: Explicit observed-time source.
        """
        self.session, self.clock = session, clock

    def definitions(self, *, for_write: bool = False) -> ResearchFeatureDefinitions:
        """Open feature definitions with an explicit storage capacity bound for writes.

        Args:
            for_write: Whether to enforce the admitted current storage capacity.

        Returns:
            Workspace feature definition owner.
        """
        del for_write  # Both paths read the live workspace cap at each write.
        return ResearchFeatureDefinitions(self.session.workspace)

    def _of(self, task: TaskRecord) -> dict[str, Any]:
        payload = dict(task.input.payload)
        if task.task_kind != TASK_KIND or (task.input, task.goal, task.plan) != _contract(payload):
            raise ValueError("feature_research.task_contract_invalid")
        return payload

    def replan_request(self, task: TaskRecord) -> dict[str, object]:
        """Replan the validated Task's retained feature declaration without readmission.

        Args:
            task: Durable Task whose declaration supplies the next request.

        Returns:
            The filled feature planning request, retaining the authored edits and input.

        Raises:
            ValueError: The Task, retained definition plan or their input binding differs.
        """
        payload = self._of(task)
        plan = self.definitions().read(payload["definition_plan_hash"])
        if plan.request.input_binding_hash != payload["input_binding_hash"]:
            raise ValueError("feature_research.build_input_mismatch")
        return {
            "operation": "FEATURE_CATALOG_PLAN",
            "feature_document": plan.request.model_dump(mode="json"),
        }

    def _require(self, payload: dict[str, Any]) -> None:
        if not is_current(
            implementation_role(preparation=_preparing(payload)),
            payload["implementation_hash"],
            implementation_hash(preparation=_preparing(payload)),
        ) or not is_current(
            SOURCE_PROJECTION_ROLE,
            payload["source_projection_hash"],
            feature_source_projection_hash(),
        ):
            raise ValueError("feature_research.execution_changed_rebuild")
        plan = self.definitions().read(payload["definition_plan_hash"])
        if plan.request.input_binding_hash != payload["input_binding_hash"]:
            raise ValueError("feature_research.build_input_mismatch")

    def build(
        self,
        plan_hash: str,
        *,
        caller: str,
        dispatcher: LocalBackgroundDispatcher,
        preprocess: bool = False,
    ) -> dict[str, object]:
        """Bind exact feature definitions and input axes before bounded build dispatch.

        Args:
            plan_hash: Exact retained feature definition plan.
            caller: Declared operation caller.
            dispatcher: Bounded local task dispatcher.
            preprocess: Whether to prepare the raw values for research.

        Returns:
            Explicit dispatcher admission/refusal metadata.
        """
        plan = self.definitions().read(plan_hash)
        bundle = read_factor_bundle(
            self.session.workspace, plan.request.input_binding_hash, verify=False
        )
        _, artifacts = factor_input_paths(self.session.workspace, bundle.binding_hash)
        resolver = ArtifactResolver(artifacts)
        source = resolver.load_feature_panel_manifest(
            resolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
        )
        payload = {
            "sessions_hash": canonical_hash(bundle.sessions),
            "listing_ids_hash": source["listing_set_hash"],
            "definition_plan_hash": plan.plan_hash,
            "input_binding_hash": plan.request.input_binding_hash,
            "implementation_hash": implementation_hash(preparation=preprocess),
            "source_projection_hash": feature_source_projection_hash(),
            "caller": caller,
            **({"purpose": "PREPROCESS_VALUES"} if preprocess else {}),
        }
        return self._submit(payload, dispatcher)

    def _submit(
        self, payload: dict[str, Any], dispatcher: LocalBackgroundDispatcher
    ) -> dict[str, object]:
        definitions = self.definitions()
        executed = research_feature_execution_spec(
            definitions.read(payload["definition_plan_hash"])
        )
        for task in self.session.task_control_registry.tasks():
            if task.task_kind != TASK_KIND:
                continue
            old = self._of(task)
            if (
                old.get("purpose") != payload.get("purpose")
                or any(
                    old[field] != payload[field]
                    for field in ("input_binding_hash", "sessions_hash", "listing_ids_hash")
                )
                or research_feature_execution_spec(definitions.read(old["definition_plan_hash"]))
                != executed
                or not is_current(
                    SOURCE_PROJECTION_ROLE,
                    old["source_projection_hash"],
                    payload["source_projection_hash"],
                )
                or not is_current(
                    implementation_role(preparation=_preparing(payload)),
                    old["implementation_hash"],
                    payload["implementation_hash"],
                )
                or task.lifecycle is TaskLifecycle.CANCELLED
            ):
                continue
            if task.lifecycle is TaskLifecycle.SUCCEEDED:
                return {
                    **self.readback(task.task_id),
                    "status": "REUSED_EXACT",
                    "task_id": None,
                    "publication_task_id": str(task.task_id),
                    "kernel_calls_this_request": 0,
                    **({"preprocessing_calls_this_request": 0} if _preparing(payload) else {}),
                }
            if task.lifecycle is TaskLifecycle.BLOCKED:
                self.session.task_control_registry.mark_recovery_required(
                    task_id=task.task_id,
                    failure_code=task.failure_code or "feature_research.retry_requested",
                    observed_at=self.clock(),
                    allow_blocked=True,
                )
            if task.lifecycle in {TaskLifecycle.BLOCKED, TaskLifecycle.RECOVERY_REQUIRED}:
                # This build's Task only, never another of its kind.
                dispatcher.resume(
                    {TASK_KIND: ResearchFeatureBuildCommand(self)}, only_task_id=task.task_id
                )
            return self.readback(task.task_id)
        sent = dispatcher.submit(ResearchFeatureBuildCommand(self, payload))
        return sent.answer(
            next_requests={
                "readback": {
                    "operation": "FEATURE_CATALOG_BUILD_READBACK",
                    "task_id": str(sent.task_id),
                }
            }
            if sent.task_id
            else {},
        )

    def admit(self, payload: dict[str, Any]) -> CommandAdmission:
        """Require current build scope and exclusive task ownership under the mutation gate.

        Args:
            payload: Explicit sealed source/definition/implementation bindings.

        Returns:
            Exact newly admitted build task.

        Raises:
            ValueError: Scope is invalid, cleanup is pending or another task must finish or recover.
        """
        with self.session.mutation_gate.hold():
            self._require(payload)
            require_no_pending_cleanup(self.session.workspace)
            registry = self.session.task_control_registry
            for t in registry.tasks():
                if t.lifecycle not in {
                    TaskLifecycle.SUCCEEDED,
                    TaskLifecycle.CANCELLED,
                    TaskLifecycle.BLOCKED,
                }:
                    raise ValueError(
                        f"feature_research.finish_or_recover_existing_task:{t.task_id}"
                    )
            envelope, goal, workflow = _contract(payload)
            task = registry.admit(
                input_envelope=envelope, goal=goal, plan=workflow, observed_at=self.clock()
            ).record
            return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Bind raw/preparation workflow, exact definition plan and execution identity.

        Args:
            task: Exact admitted feature build task.

        Returns:
            Deterministic execution compatibility contract.
        """
        payload = self._of(task)
        self._require(payload)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash(
                (TASK_KIND, STAGE, PREPARATION_STAGE) if _preparing(payload) else (TASK_KIND, STAGE)
            ),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=payload["definition_plan_hash"],
            framework_identity_hash=self.session.execution_identity(payload["implementation_hash"]),
        )

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one exact admitted feature build task.

        Args:
            task_id: Exact retained task.
            expected_task_hash: Optional optimistic task identity.
        """
        task = self.session.task_control_registry.task(task_id)
        self.session.execute_admitted(task, self, self.clock, expected_task_hash)

    def execute_stage(
        self, *, task: TaskRecord, execution: TaskExecution, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        """Materialize raw columns or prepare their exact overlay under mutation ownership.

        Args:
            task: Exact admitted feature build task.
            execution: Current execution declaration.
            work_item: Exact raw/preparation stage.

        Returns:
            Ready exact receipt evidence, safe-checkpoint cancellation or a bounded named refusal.
        """
        del execution
        payload = self._of(task)
        try:
            self._require(payload)
            if _preparing(payload) and work_item.stage_id == PREPARATION_STAGE:
                raw, _ = self._receipt(task)
                with self.session.mutation_gate.hold():
                    prepared = prepare_feature_overlay(
                        definitions=self.definitions(for_write=True),
                        raw=raw,
                        implementation_hash=payload["implementation_hash"],
                        cancelled=lambda: (
                            self.session.task_control_registry.task(task.task_id).lifecycle
                            is TaskLifecycle.CANCEL_REQUESTED
                        ),
                    )
                return StageExecutionResult(
                    StageDisposition.READY,
                    evidence=(
                        TaskEvidence(
                            evidence_kind="feature_research.preprocessed_overlay",
                            content_hash=prepared.content_hash,
                            reference=f"playpen://feature-panel/closure/{PREPARATION_CATEGORY}/{prepared.content_hash}",
                        ),
                    ),
                )
            if work_item.stage_id != STAGE:
                raise ValueError("feature_research.stage_unknown")
            definitions = self.definitions(for_write=True)
            with self.session.mutation_gate.hold():
                receipt = materialize_feature_columns(
                    definitions=definitions,
                    plan=definitions.read(payload["definition_plan_hash"]),
                    implementation_hash=payload["implementation_hash"],
                    source_projection_hash=payload["source_projection_hash"],
                    cancelled=lambda: (
                        self.session.task_control_registry.task(task.task_id).lifecycle
                        is TaskLifecycle.CANCEL_REQUESTED
                    ),
                )
            return StageExecutionResult(
                StageDisposition.READY,
                evidence=(
                    TaskEvidence(
                        evidence_kind="feature_research.raw_values",
                        content_hash=receipt.content_hash,
                        reference=f"playpen://feature-panel/closure/{RECEIPT_CATEGORY}/{receipt.content_hash}",
                    ),
                ),
            )
        except (ValueError, OSError, RuntimeError) as error:
            if str(error) == "feature_research.cancelled_at_safe_checkpoint":
                return StageExecutionResult(StageDisposition.CANCELLED)
            return StageExecutionResult(
                StageDisposition.BLOCKED,
                failure_code=str(getattr(error, "failure_code", str(error)))[:240],
            )

    def _receipt(
        self, task: TaskRecord, evidence: tuple[TaskEvidence, ...] | None = None
    ) -> tuple[ResearchFeatureMaterialization, tuple[tuple[str, ResearchFormulaColumn], ...]]:
        payload = self._of(task)
        if evidence is None:
            evidence = next(
                item.evidence
                for item in self.session.task_control_registry.work_items(task.task_id)
                if item.stage_id == STAGE
            )
        if (
            len(evidence) != 1
            or evidence[0].reference
            != f"playpen://feature-panel/closure/{RECEIPT_CATEGORY}/{evidence[0].content_hash}"
        ):
            raise ValueError("feature_research.task_evidence_invalid")
        store = self.definitions().store
        receipt = store.load_model(
            category=RECEIPT_CATEGORY,
            content_hash=evidence[0].content_hash,
            model=ResearchFeatureMaterialization,
        )
        if (
            receipt.content_hash != evidence[0].content_hash
            or receipt.definition_plan_hash != payload["definition_plan_hash"]
            or receipt.implementation_hash != payload["implementation_hash"]
        ):
            raise ValueError("feature_research.receipt_binding_mismatch")
        plan = self.definitions().read(receipt.definition_plan_hash)
        inherited, owned = feature_value_sources(self.definitions(), plan)
        expected = {v.factor_id: v for v in plan.candidate.features}
        if (
            receipt.input_binding_hash != payload["input_binding_hash"]
            or receipt.inherited_input_factor_ids != inherited
            or {k for k, _ in receipt.columns} != {v.factor_id for v in owned}
        ):
            raise ValueError("feature_research.receipt_catalog_mismatch")
        from alphalattice.foundation.feature_engine.producers.factors.core_bundle import (
            numerical_spec_hash,
        )

        reader = ResearchFormulaValues(store)
        verified = []
        for name, identity in receipt.columns:
            column = reader.read(identity)
            if (
                column.input_binding_hash != receipt.input_binding_hash
                or column.numerical_spec_hash != numerical_spec_hash(expected[name].specification)
                or column.implementation_hash != expected[name].implementation_hash
                or not is_current(
                    SOURCE_PROJECTION_ROLE,
                    column.source_projection_hash,
                    payload["source_projection_hash"],
                )
                or canonical_hash(column.sessions) != payload["sessions_hash"]
                or canonical_hash(column.listing_ids) != payload["listing_ids_hash"]
            ):
                raise ValueError("feature_research.receipt_column_mismatch")
            verified.append((name, column))
        return receipt, tuple(verified)

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Verify the exact raw or preparation receipt for this task and installed stage.

        Args:
            task: Exact retained feature build task.
            execution: Current execution declaration.
            work_item: Exact raw/preparation stage.
            evidence: Declared exact receipt evidence.

        Returns:
            Unchanged verified evidence.

        Raises:
            ValueError: Stage, receipt or current preparation policy is invalid.
        """
        del execution
        if _preparing(self._of(task)) and work_item.stage_id == PREPARATION_STAGE:
            self._preparation_receipt(task, evidence, current_policy=True)
            return evidence
        if work_item.stage_id != STAGE:
            raise ValueError("feature_research.stage_unknown")
        self._receipt(task, evidence)
        return evidence

    def _preparation_receipt(
        self,
        task: TaskRecord,
        evidence: tuple[TaskEvidence, ...] | None = None,
        *,
        current_policy: bool = False,
        raw: ResearchFeatureMaterialization | None = None,
    ) -> ResearchFeaturePreparation:
        receipt = self._load_preparation(task, evidence)
        verify_feature_preparation(
            definitions=self.definitions(),
            raw=raw if raw is not None else self._receipt(task)[0],
            receipt=receipt,
            implementation_hash=self._of(task)["implementation_hash"],
            current_policy=current_policy,
        )
        return receipt

    def _load_preparation(
        self, task: TaskRecord, evidence: tuple[TaskEvidence, ...] | None = None
    ) -> ResearchFeaturePreparation:
        self._of(task)
        if evidence is None:
            evidence = next(
                item.evidence
                for item in self.session.task_control_registry.work_items(task.task_id)
                if item.stage_id == PREPARATION_STAGE
            )
        if (
            len(evidence) != 1
            or evidence[0].reference
            != f"playpen://feature-panel/closure/{PREPARATION_CATEGORY}/{evidence[0].content_hash}"
        ):
            raise ValueError("feature_research.preprocessing_evidence_invalid")
        receipt = self.definitions().store.load_model(
            category=PREPARATION_CATEGORY,
            content_hash=evidence[0].content_hash,
            model=ResearchFeaturePreparation,
        )
        if receipt.content_hash != evidence[0].content_hash:
            raise ValueError("feature_research.preprocessing_evidence_invalid")
        return receipt

    def prepared_source(
        self, preparation_hash: str, *, input_binding_hash: str, current_policy: bool = True
    ) -> ResolvedDevelopmentFeatureInput:
        """Only a completed owner Task admits its prepared source to research."""
        for task in self.session.task_control_registry.tasks():
            if task.task_kind != TASK_KIND or task.lifecycle is not TaskLifecycle.SUCCEEDED:
                continue
            items = self.session.task_control_registry.work_items(task.task_id)
            if not any(
                item.stage_id == PREPARATION_STAGE
                and any(e.content_hash == preparation_hash for e in item.evidence)
                for item in items
            ):
                continue
            payload = self._of(task)
            if not _preparing(payload) or payload["input_binding_hash"] != input_binding_hash:
                raise ValueError("feature_research.prepared_input_mismatch")
            raw, _ = self._receipt(task)
            return resolve_prepared_feature_input(
                definitions=self.definitions(),
                raw=raw,
                receipt=self._load_preparation(task),
                implementation_hash=payload["implementation_hash"],
                current_policy=current_policy,
            )
        raise ValueError("feature_research.completed_preparation_required")

    def readback(self, task_id: UUID) -> dict[str, Any]:
        """Read verified build receipts, saved column counts and downstream readiness.

        Args:
            task_id: Exact retained feature build task.

        Returns:
            Task state, verified raw columns and optional prepared source; raw formulas alone grant
            no Factor or Alpha admission.
        """
        task = self.session.task_control_registry.task(task_id)
        payload = self._of(task)
        receipt, verified = (
            self._receipt(task) if task.lifecycle is TaskLifecycle.SUCCEEDED else (None, ())
        )
        preparation = (
            self._preparation_receipt(task, raw=receipt)
            if receipt is not None and _preparing(payload)
            else None
        )
        columns = []
        downstream = []
        if receipt is not None:
            definition = self.definitions().read(receipt.definition_plan_hash)
            specifications = {v.factor_id: v.specification for v in definition.candidate.features}
            for name, col in verified:
                columns.append(
                    {
                        "factor_id": name,
                        "column_hash": col.content_hash,
                        "values_hash": col.values.content_hash,
                        "sessions": len(col.sessions),
                        "start": str(col.sessions[0]),
                        "end": str(col.sessions[-1]),
                        "listings": len(col.listing_ids),
                        "available": col.available_count,
                        "cells": col.values.row_count,
                        "physical_bytes": col.values.byte_count,
                    }
                )
                try:
                    specification = build_research_formula_specification(
                        specifications[name],
                        source_session_count=len(col.sessions),
                        preprocessing_recipe=definition.preprocessing_recipes.get(name),
                    )
                    if specification.implementation_hash != col.implementation_hash:
                        raise FeatureFormulaSpecificationError(
                            "feature_research.formula_implementation_changed_replan"
                        )
                except ValueError as error:
                    downstream.append(
                        {
                            "factor_id": name,
                            "status": "SPECIFICATION_REQUIRED",
                            "reason": str(error),
                        }
                    )
                else:
                    downstream.append(
                        {
                            "factor_id": name,
                            "status": "PREPARED_RESEARCH_SOURCE"
                            if preparation is not None
                            else "SPECIFICATION_AVAILABLE_PREPROCESSING_NOT_PREPARED",
                            "specification_hash": specification.specification_hash,
                            "preprocessing_role": specification.preprocessing_role,
                            "executable_formula": specification.executable_definition
                            or specification.formula,
                            "observation_clock": specification.clock.model_dump(mode="json"),
                        }
                    )
        return {
            "status": task.lifecycle.value,
            "task_id": str(task.task_id),
            "failure_code": task.failure_code,
            "receipt": None if receipt is None else receipt.model_dump(mode="json"),
            "columns": columns,
            "downstream_readiness": downstream,
            "claim": "SAVED_PREPARED_FEATURE_SOURCE_CURRENT_ADMISSION_CHECKED_AT_PLAN"
            if preparation is not None
            else "RAW_FORMULAS_NOT_PANEL_OR_FACTOR_ALPHA_ADMISSION",
            **({"preparation": preparation.model_dump(mode="json")} if preparation else {}),
            "next_requests": {
                "readback": {
                    "operation": "FEATURE_CATALOG_BUILD_READBACK",
                    "task_id": str(task_id),
                },
                **(
                    {
                        "research": {
                            "operation": "EXPERIMENT_CONTROLS",
                            "input_binding_hash": preparation.input_binding_hash,
                            "feature_preparation_hash": preparation.content_hash,
                            "experiment_kind": "factor.screening-development",
                        }
                    }
                    if preparation is not None
                    else {}
                ),
            },
        }


@dataclass
class ResearchFeatureBuildCommand:
    """Dispatch one explicit feature build request through its deterministic owner."""

    application: ResearchFeatureBuildApplication
    payload: dict[str, Any] | None = None
    command_kind: str = TASK_KIND

    def admit(self) -> CommandAdmission:
        """Require the exact feature build request before task admission.

        Returns:
            Deterministic task admission.

        Raises:
            ValueError: Required payload is absent or its declared admission checks refuse.
        """
        if self.payload is None:
            raise ValueError("feature_research.build_request_required")
        return self.application.admit(self.payload)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one exact task admitted for this feature build request.

        Args:
            task_id: Exact retained task.
            expected_task_hash: Optional optimistic task identity.
        """
        self.application.execute(task_id, expected_task_hash=expected_task_hash)
