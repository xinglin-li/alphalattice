"""One captured component-score preparation through the existing Task runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Self, cast
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from alphalattice.control.data_platform.readiness import _latest_common_us_session
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    ResearchWorkspaceManifestHolder,
    ResearchWorkspaceScoreInput,
    held,
    manifest_fields_hash,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.strategy_score_inputs import (
    PreparedWorkspaceComponentInputs,
    build_workspace_score_inputs,
    prepare_workspace_component_inputs,
    read_workspace_component_inputs,
    workspace_observation_history_columns,
    workspace_score_source_identity,
)
from alphalattice.control.product_host.maintenance.data_update import (
    installed_data_update_binding,
    read_workspace_inputs,
)
from alphalattice.control.product_host.storage.inventory import require_storage_capacity
from alphalattice.control.product_host.storage.plan_previews import (
    PLAN_PREVIEWS_DIRECTORY,
    PreviewRegistry,
)
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
from alphalattice.interface.local_application.dispatcher import CommandAdmission
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from alphalattice.investment.alpha_research.publication.contracts import (
    FrozenComponentScoreSnapshot,
    WorkspaceObservationHistoryHead,
    seal_current_contract,
)
from alphalattice.investment.alpha_research.scores.frozen_inference import (
    AdmittedFrozenInference,
)
from alphalattice.investment.alpha_research.scores.model_renewal import (
    AdmittedRenewingInference,
    AlphaModelSetPublication,
    AlphaTrainingObservations,
    admit_component_inference,
    publish_component_training_observations,
    renew_lifecycle_admission,
    verified_lifecycle_admissions,
    verify_lifecycle_successor,
    verify_model_set,
)
from alphalattice.investment.alpha_research.scores.product_lifecycle import (
    AlphaModelLifecycleRecipe,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    FrozenStrategyPackage,
    recipe_hashes_match,
)
from alphalattice.kernel.shared_kernel.identity import (
    canonical_hash,
    schema_structure,
)
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.sealing import seal_model_from_dump
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash

PLAN_FIELDS = ("score_inputs",)
"""The manifest fields a plan reads: the score inputs it prepares scores for. A plan binds them,
never the whole manifest, so a publication of fields it does not read leaves it applicable (V223,
OW10)."""

TASK_KIND = "strategy_score_preparation"
STAGES = ("verify_frozen_inputs", "materialize_features", "score_models", "publish_score_receipt")


IMPLEMENTATION_ROLE = "product_host.strategy_scoring"
"""The role a move of this implementation is recorded under in `config/identity-successors.json`,
so a plan sealed under its predecessor stays current (binding plan R1, LAWS.md ID1)."""


def _implementation_hash() -> str:
    root = resolve_playpen_root(Path(__file__))
    return source_rule_closure_hash(
        root=root,
        semantic_owner="product_host",
        numerical_role="FROZEN_COMPONENT_INFERENCE",
        tracked_paths=(
            "src/alphalattice/control/product_host/composition/strategy_scoring.py",
            "src/alphalattice/control/product_host/composition/strategy_score_inputs.py",
            "src/alphalattice/investment/alpha_research/inputs/frozen_price_volume.py",
            "src/alphalattice/investment/alpha_research/inputs/panel_feature_materialization.py",
            "src/alphalattice/investment/alpha_research/inputs/panel_feature_views.py",
            "src/alphalattice/investment/alpha_research/scores/frozen_inference.py",
            "src/alphalattice/investment/alpha_research/scores/product_lifecycle.py",
            "src/alphalattice/investment/alpha_research/scores/model_renewal.py",
            "src/alphalattice/investment/alpha_research/targets/component_training.py",
            "src/alphalattice/investment/alpha_research/targets/total_return.py",
            "src/alphalattice/investment/alpha_research/experiments/development_contracts.py",
            "src/alphalattice/investment/alpha_research/publication/contracts.py",
            "src/alphalattice/investment/alpha_research/publication/artifacts.py",
            "src/alphalattice/investment/alpha_research/experiments/development_artifacts.py",
            "src/alphalattice/investment/alpha_research/scores/product_replay.py",
            "src/alphalattice/investment/alpha_research/scores/heterogeneous_replay.py",
            "src/alphalattice/investment/alpha_research/inputs/preprocessing/sector_context.py",
            "src/alphalattice/investment/sector_research/inputs/surface.py",
            "src/alphalattice/foundation/feature_engine/producers/factors/session_observation.py",
            "src/alphalattice/foundation/feature_engine/producers/factors/session_liquidity.py",
            "src/alphalattice/foundation/feature_engine/producers/factors/interactions.py",
            "src/alphalattice/kernel/quant/cross_section.py",
            "src/alphalattice/foundation/feature_engine/producers/factors/registry.py",
            "src/alphalattice/foundation/feature_engine/producers/factors/catalog.py",
            "src/alphalattice/foundation/feature_engine/storage/repositories.py",
            "src/alphalattice/foundation/market_data_ops/storage/duckdb.py",
            "src/alphalattice/kernel/data/calendar.py",
            "src/alphalattice/control/data_platform/candidate_requalification.py",
            "src/alphalattice/foundation/causal_outcomes/execution/compile.py",
            "src/alphalattice/foundation/causal_outcomes/execution/methods.py",
            "src/alphalattice/foundation/market_data_ops/returns/execution.py",
        ),
    )


def workspace_observation_history_scope(binding: ResearchWorkspaceScoreInput) -> str:
    """Name the preparation slot of an installed component, apart from its models.

    A model or rule rotation replaces this slot's current and previous heads;
    it does not retain an ever-growing map of old model selectors.
    """
    return str(
        canonical_hash(
            {
                "kind": "WorkspaceObservationHistoryScope.v1",
                "package": binding.strategy_package_id,
                "component": binding.component_id,
                "source": binding.source_kind,
            }
        )
    )


def workspace_observation_history_selection(
    binding: ResearchWorkspaceScoreInput, *, implementation: str
) -> str:
    """Name what a kept history's values depend on: its slot, package and implementation.

    Not the authority, its Features or its training factors: they choose which columns a
    capture needs, and a capture proves the kept columns' source prefix apart. So a renewal,
    which changes the authority and adds training factors, keeps the history the next day reads.
    """
    return str(
        canonical_hash(
            {
                "kind": "WorkspaceObservationHistorySelection.v2",
                "scope": workspace_observation_history_scope(binding),
                "package": binding.strategy_package_hash,
                "implementation": implementation,
            }
        )
    )


def workspace_observation_history_reuse(
    head: WorkspaceObservationHistoryHead | None,
    prepared: PreparedWorkspaceComponentInputs,
    columns: Mapping[str, object],
) -> WorkspaceObservationHistoryHead | None:
    """The kept head whose stored parts a publication shares, or None to write its own.

    Parts are shared only by the head a capture reused and only with the same columns: the day
    after a renewal keeps fewer columns than the renewal's history and writes them afresh.
    """
    if (
        head is None
        or prepared.reused_history_hash != head.head_hash
        or head.column_names != tuple(sorted(columns))
    ):
        return None
    return head


class StrategyScorePlan(BaseModel):  # type: ignore[misc]
    """Seal the exact declared component scoring request, source and installed implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    workspace_id: str
    workspace_manifest_hash: str
    binding: ResearchWorkspaceScoreInput
    formation_session: date
    source_identity_hash: str
    implementation_hash: str
    plan_hash: str

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal an explicit declared component scoring plan.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical plan_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_model_from_dump(cls, values, field="plan_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact declared component scoring plan identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Serialized fields differ from plan_hash.
        """
        if self.plan_hash != canonical_hash(self.model_dump(mode="json", exclude={"plan_hash"})):
            raise ValueError("strategy_score.plan_identity_invalid")
        return self


def _task_contract(plan: StrategyScorePlan) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    envelope = TaskInputEnvelope.create(
        task_kind=TASK_KIND,
        input_schema_id="strategy-score-preparation",
        payload={"plan": plan.model_dump(mode="json")},
    )
    goal = ResearchGoal.create(
        goal_kind="PREPARE_FROZEN_COMPONENT_SCORE",
        input_hash=envelope.input_hash,
        deliverable_kind="FrozenComponentScoreSnapshot",
        summary="Prepare a component score; do not publish positions.",
    )
    workflow = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash(STAGES),
        verifier_catalog_hash=canonical_hash(tuple(f"strategy_score.{stage}" for stage in STAGES)),
        work_items=tuple(
            WorkItemDefinition.create(
                stage_id=stage, dependency_ids=STAGES[:index], verifier_id=f"strategy_score.{stage}"
            )
            for index, stage in enumerate(STAGES)
        ),
    )
    return envelope, goal, workflow


def _replan(plan: StrategyScorePlan) -> dict[str, object]:
    """The scoring request a plan was made from: its package, its day and its component."""
    binding = plan.binding
    return {
        "operation": "STRATEGY_SCORE_PLAN",
        "strategy_package_id": binding.strategy_package_id,
        "formation_session": plan.formation_session.isoformat(),
        **({"component_id": binding.component_id} if binding.component_id is not None else {}),
    }


class StrategyScoringApplication:
    """Own admitted declared component scoring tasks and their exact durable publications."""

    task_kind = TASK_KIND
    replans = (
        TaskReplan(
            task_kind=TASK_KIND, preview="STRATEGY_SCORE_PLAN", admitting="STRATEGY_SCORE_RUN"
        ),
    )
    """The re-plan of the Task kind this owner admits, which the recovery view offers
    (V188)."""
    additional_publications: Callable[[], tuple[FrozenComponentScoreSnapshot, ...]] | None = None

    @property
    def manifest(self) -> ResearchWorkspaceManifest:
        """The workspace manifest, read from the one holder the Host refreshes (V182)."""
        return self._manifests.current

    @property
    def packages(self) -> Mapping[str, FrozenStrategyPackage]:
        """The installed packages, read from their one holder each time."""
        return self._packages()

    def __init__(
        self,
        *,
        session: WorkspaceApplicationSession,
        manifest: ResearchWorkspaceManifest | ResearchWorkspaceManifestHolder,
        packages: Callable[[], Mapping[str, FrozenStrategyPackage]],
        clock: Callable[[], datetime],
    ):
        """Wire declared packages, retained workspace session and exact score storage.

        Args:
            session: Retained workspace writer/task session.
            manifest: Held workspace declaration.
            packages: Reads the installed frozen strategy packages from their one holder, the
                Host's operations, so an installation reaches this owner too.
            clock: Explicit observed-time source.
        """
        self.session, self._packages, self.clock = session, packages, clock
        self._manifests = held(manifest)
        self.store = AlphaCurrentArtifactStore(session.workspace / "artifacts")
        self.last_plan: StrategyScorePlan | None = None
        # Every plan an answer named, by its hash, sealed on disk until it expires: a run
        # from any of them, after a restart too, reopens it and checks it again (V493, V525).
        self._plans: PreviewRegistry[StrategyScorePlan] = PreviewRegistry(
            model=StrategyScorePlan,
            clock=self.clock,
            root=self.session.workspace / "runtime" / PLAN_PREVIEWS_DIRECTORY / "strategy-score",
        )

    def replan_requests(self, plan_hash: str) -> dict[str, object]:
        """Bind re-planning to a verified plan's package, component and session.

        Args:
            plan_hash: The refused plan's exact hash.

        Returns:
            Its bound scoring request, or no offer when the source cannot be verified.
        """
        kept = self._plans.get(plan_hash)
        return {} if kept is None else {"replan": _replan(kept.plan)}

    def _binding(
        self, package_id: str, component_id: str | None = None
    ) -> ResearchWorkspaceScoreInput:
        if not self.store.root.resolve().is_relative_to(self.session.workspace.resolve()):
            raise ValueError("strategy_score.artifact_root_outside_workspace")
        choices = [
            item
            for item in self.manifest.score_inputs or ()
            if item.strategy_package_id == package_id
            and (component_id is None or item.component_id == component_id)
        ]
        if len(choices) > 1:
            # A package of several components scores them one at a time (V332).
            raise ValueError(
                "strategy_score.component_required:"
                + ",".join(sorted(str(value.component_id) for value in choices))
            )
        if not choices:
            raise ValueError("strategy_score.input_preparation_not_installed")
        binding = choices[0]
        package = self.packages.get(package_id)
        if package is None or binding.strategy_package_hash != package.package_hash:
            raise ValueError("strategy_score.package_binding_mismatch")
        if manifest_fields_hash(
            read_research_workspace_manifest(self.session.workspace), PLAN_FIELDS
        ) != manifest_fields_hash(self.manifest, PLAN_FIELDS):
            raise ValueError("strategy_score.workspace_manifest_changed")
        return binding

    def _admitted_source(
        self, binding: ResearchWorkspaceScoreInput
    ) -> AdmittedFrozenInference | AdmittedRenewingInference:
        root = (self.session.workspace / binding.authority_relative_path).resolve()
        if not root.is_relative_to(self.session.workspace.resolve()):
            raise ValueError("strategy_score.authority_path_outside_workspace")
        return admit_component_inference(
            root, store=self.store, expected_hash=binding.authority_hash
        )

    def _authority(
        self, binding: ResearchWorkspaceScoreInput
    ) -> AdmittedFrozenInference | AdmittedRenewingInference:
        admitted = self._admitted_source(binding)
        if admitted.authority.model_set.component_id not in self.packages[
            binding.strategy_package_id
        ].component_ids or (
            binding.component_id is not None
            and binding.component_id != admitted.authority.model_set.component_id
        ):
            raise ValueError("strategy_score.package_component_mismatch")
        if isinstance(admitted, AdmittedRenewingInference):
            component = admitted.authority.component
            declared = next(
                item
                for item in self.packages[binding.strategy_package_id].component_plan
                if item.component_id == component.component_id
            )
            lifecycle_hash = (
                declared.model_lifecycle_hash
                or AlphaModelLifecycleRecipe.from_component(component).content_hash
            )
            if (
                not recipe_hashes_match(declared.recipe_hash, component.research_recipe_hash)
                or lifecycle_hash != admitted.authority.lifecycle.content_hash
            ):
                raise ValueError("strategy_score.package_lifecycle_mismatch")
        return admitted

    def _captured_authority(
        self, binding: ResearchWorkspaceScoreInput, identity: str
    ) -> AdmittedFrozenInference | AdmittedRenewingInference:
        ancestor = self._admitted_source(binding)
        if identity == ancestor.authority.authority_hash:
            return ancestor
        if not isinstance(ancestor, AdmittedRenewingInference):
            raise ValueError("strategy_score.renewal_not_installed")
        descendant = admit_component_inference(
            self.store.root / "current/lifecycle-admissions" / f"{identity}.json",
            store=self.store,
            expected_hash=identity,
        )
        if not isinstance(descendant, AdmittedRenewingInference):
            raise ValueError("strategy_score.renewal_not_installed")
        verify_lifecycle_successor(
            self.store, ancestor=ancestor.authority, successor=descendant.authority
        )
        return descendant

    def authority_for_formation(
        self, binding: ResearchWorkspaceScoreInput, formation: date
    ) -> AdmittedFrozenInference | AdmittedRenewingInference:
        """Resolve admitted authority or a verified published successor supporting formation.

        Args:
            binding: Exact declared score input binding.
            formation: Explicit formation session.

        Returns:
            Admitted inference authority; arbitrary artifact files cannot supply an effective
            successor.
        """
        admitted = self._authority(binding)
        if admitted.authority.supports(formation):
            return admitted
        # Only verified Task publications can supply an effective successor;
        # arbitrary files under the artifact root are not an active-model index.
        for score in reversed(self.published_scores()):
            if score.strategy_package_hash == binding.strategy_package_hash and recipe_hashes_match(
                score.component_recipe_hash, admitted.authority.model_set.recipe_hash
            ):
                candidate = self._captured_authority(binding, score.inference_authority_hash)
                if candidate.authority.supports(formation):
                    return candidate
        return admitted

    def _source(self, binding: ResearchWorkspaceScoreInput) -> tuple[str, date]:
        if binding.source_kind == "RECORDED_INPUT_SNAPSHOT":
            assert binding.observation_snapshot_hash is not None
            snapshot = self.store.load_frozen_observation_snapshot(
                binding.observation_snapshot_hash
            )
            self.store.load_frozen_observations(snapshot.snapshot_hash)
            return snapshot.snapshot_hash, snapshot.formation_sessions[-1]
        identity = workspace_score_source_identity(self.session.workspace)
        state = read_workspace_inputs(
            self.session.workspace, installed_data_update_binding(self.session.workspace)
        )
        return identity, min(
            state.panel_through,
            _latest_common_us_session(on_or_before=self.clock().date(), observed_at=self.clock()),
        )

    def plan(
        self, package_id: str, formation: date | None = None, *, component_id: str | None = None
    ) -> dict[str, object]:
        """Resolve formation and admitted lifecycle before sealing an exact scoring plan.

        Args:
            package_id: Declared installed strategy package.
            formation: Optional formation; defaults to admitted source coverage end.
            component_id: Optional declared component selection.

        Returns:
            Sealed plan metadata, authority counts, source kind, cache status and declared work.

        Raises:
            ValueError: Formation exceeds source coverage or admitted model epoch/preparation
                authority.
        """
        binding = self._binding(package_id, component_id)
        source_hash, latest = self._source(binding)
        formation = formation or latest
        authority = self.authority_for_formation(binding, formation)
        if formation > latest or not (
            authority.authority.supports(formation)
            or (
                isinstance(authority, AdmittedRenewingInference)
                and authority.authority.can_prepare(formation)
            )
        ):
            # The refusal names the formation it chose and the epoch the lifecycle admits (V335).
            admitted = authority.authority
            start = getattr(admitted, "formation_start", None)
            end = getattr(admitted, "formation_end", None)
            raise ValueError(
                f"strategy_score.formation_or_model_epoch_unavailable:formation={formation}"
                + (f",epoch={start}..{end}" if start and end else "")
            )
        if isinstance(authority, AdmittedRenewingInference):
            authority.authority.planned_refits(self.store, (formation,), allow_preparation=True)
        # The answer names this plan, never the owner's last one, which a concurrent plan
        # may have replaced in between (V534).
        planned = StrategyScorePlan.create(
            workspace_id=self.manifest.workspace_id,
            workspace_manifest_hash=manifest_fields_hash(self.manifest, PLAN_FIELDS),
            binding=binding,
            formation_session=formation,
            source_identity_hash=source_hash,
            implementation_hash=_implementation_hash(),
        )
        self.last_plan = planned
        return {
            "status": "PLANNED",
            "score_plan_hash": self._kept(planned).plan_hash,
            "strategy_package_id": package_id,
            "formation_session": formation.isoformat(),
            "source_kind": binding.source_kind,
            "model_count": len(authority.models),
            "feature_count": authority.authority.model_set.feature_count,
            "cached": self._published(planned) is not None,
            "work": "Resolve the declared lifecycle, renew missing models, predict and publish."
            if isinstance(authority, AdmittedRenewingInference)
            else "Construct inputs, predict with admitted models, publish score. Zero fits.",
            "limitation": "INPUT_TO_SCORE_QA_NOT_PORTFOLIO_RECOMMENDATION",
            # The same plan again from this answer, its day and component kept; a plan with no
            # day still takes the newest (V562).
            "next_requests": {
                "run": {"operation": "STRATEGY_SCORE_RUN", "score_plan_hash": planned.plan_hash},
                "replan": _replan(planned),
            },
        }

    def prepare(self, plan_hash: str) -> StrategyScorePlan:
        """Reopen the exact score preview or retained admitted task plan.

        Args:
            plan_hash: Exact plan identity.

        Returns:
            Retained scoring plan.

        Raises:
            ValueError: No exact preview or admitted task plan exists.
        """
        if (kept := self._plans.runnable(plan_hash)) is not None:
            return kept
        for task in self.session.task_control_registry.tasks():
            if task.task_kind == TASK_KIND:
                plan = self._plan_of(task, require_current=False)
                if plan.plan_hash == plan_hash:
                    return plan
        raise ValueError("strategy_score.plan_required")

    def _kept(self, plan: StrategyScorePlan) -> StrategyScorePlan:
        self._plans.remember(plan)
        return plan

    def _require_plan(self, plan: StrategyScorePlan) -> None:
        if (
            plan.workspace_id != self.manifest.workspace_id
            or plan.workspace_manifest_hash != manifest_fields_hash(self.manifest, PLAN_FIELDS)
            or plan.binding
            != self._binding(plan.binding.strategy_package_id, plan.binding.component_id)
            or not is_current(IMPLEMENTATION_ROLE, plan.implementation_hash, _implementation_hash())
        ):
            raise ValueError("strategy_score.task_binding_mismatch")

    def _plan_of(self, task: TaskRecord, *, require_current: bool = True) -> StrategyScorePlan:
        if task.task_kind != TASK_KIND:
            raise ValueError("strategy_score.task_kind_invalid")
        plan = StrategyScorePlan.model_validate(task.input.payload["plan"])
        envelope, goal, workflow = _task_contract(plan)
        if (task.input, task.goal, task.plan) != (envelope, goal, workflow):
            raise ValueError("strategy_score.task_contract_invalid")
        if require_current:
            self._require_plan(plan)
        return cast(StrategyScorePlan, plan)

    def replan_request(self, task: TaskRecord) -> dict[str, object]:
        """Replan the validated Task's exact package, component and formation session.

        Args:
            task: Durable Task whose declaration supplies the next request.

        Returns:
            The filled scoring planning request.

        Raises:
            ValueError: The Task's kind, sealed plan or workflow does not validate.
        """
        return _replan(self._plan_of(task, require_current=False))

    def _published(self, plan: StrategyScorePlan) -> FrozenComponentScoreSnapshot | None:
        # Immutable files, not a second index or an in-memory numerical cache.
        directory = self.store._path("current/frozen-component-scores", "0" * 64, "json").parent
        matches = [
            value
            for path in sorted(directory.glob("*.json"))
            if (value := self.store.load_frozen_component_score(path.stem)).request_hash
            == plan.plan_hash
        ]
        if len(matches) > 1:
            raise ValueError("strategy_score.publication_ambiguous")
        return matches[0] if matches else None

    def reusable(self, plan: StrategyScorePlan) -> FrozenComponentScoreSnapshot | None:
        """Require current source, authority and exact publication evidence before score reuse.

        Args:
            plan: Explicit sealed scoring plan.

        Returns:
            Verified score from admitted additional publications or a succeeded exact task,
            otherwise None.

        Raises:
            ValueError: Plan/source/authority is stale or task publication evidence differs.
        """
        self._require_plan(plan)
        if self._source(plan.binding)[0] != plan.source_identity_hash:
            raise ValueError("strategy_score.stale_plan")
        self._authority(plan.binding)
        value = self._published(plan)
        if (
            value is not None
            and self.additional_publications is not None
            and value in self.additional_publications()
        ):
            return value
        for task in self.session.task_control_registry.tasks():
            if (
                task.task_kind == TASK_KIND
                and task.lifecycle is TaskLifecycle.SUCCEEDED
                and self._plan_of(task, require_current=False) == plan
            ):
                if value is not None and self._stage_hash(task, STAGES[3]) != value.snapshot_hash:
                    raise ValueError("strategy_score.publication_evidence_mismatch")
                return value
        return None

    def admit(self, plan: StrategyScorePlan) -> CommandAdmission:
        """Reuse or admit the exact current component scoring plan.

        Reuse an exact uncancelled admission or admit the current declared component scoring plan.

        Args:
            plan: Explicit sealed plan.

        Returns:
            Exact retained or newly admitted task identity/lifecycle.

        Raises:
            ValueError: Plan authority or current source is invalid or stale.
        """
        self._require_plan(plan)
        self._authority(plan.binding)
        envelope, goal, workflow = _task_contract(plan)
        registry = self.session.task_control_registry
        for task in registry.tasks():
            if task.input == envelope and task.lifecycle is not TaskLifecycle.CANCELLED:
                return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)
        if self._source(plan.binding)[0] != plan.source_identity_hash:
            raise ValueError("strategy_score.stale_plan")
        task = registry.admit(
            input_envelope=envelope, goal=goal, plan=workflow, observed_at=self.clock()
        ).record
        return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute the exact admitted declared component scoring task through its writer session.

        Args:
            task_id: Exact admitted task.
            expected_task_hash: Optional optimistic task identity.
        """
        task = self.session.task_control_registry.task(task_id)
        self._plan_of(task)
        self.session.execute_admitted(task, self, self.clock, expected_task_hash)

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Bind the exact task schema, workflow, exact inference authority and execution identity.

        Args:
            task: Exact admitted task whose retained plan is verified.

        Returns:
            Deterministic execution compatibility declaration.
        """
        plan = self._plan_of(task)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash(schema_structure(StrategyScorePlan)),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=plan.binding.authority_hash,
            framework_identity_hash=self.session.execution_identity(plan.implementation_hash),
        )

    def _stage_hash(self, task: TaskRecord, stage: str) -> str:
        state = next(
            value
            for value in self.session.task_control_registry.work_items(task.task_id)
            if value.stage_id == stage
        )
        if len(state.evidence) != 1:
            raise ValueError("strategy_score.stage_evidence_absent")
        return str(state.evidence[0].content_hash)

    @staticmethod
    def _evidence(stage: str, content_hash: str) -> tuple[TaskEvidence, ...]:
        return (
            TaskEvidence(
                evidence_kind="strategy_score.stage",
                reference=f"playpen://strategy-score/{stage}/{content_hash}",
                content_hash=content_hash,
            ),
        )

    def execute_stage(
        self, *, task: TaskRecord, execution: TaskExecution, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        """Execute one admitted declared component scoring stage under verified lifecycle admission.

        Args:
            task: Exact admitted task.
            execution: Current execution declaration.
            work_item: Exact stage whose prior evidence is read from this task.

        Returns:
            Stage outcome with exact content evidence.
        """
        with verified_lifecycle_admissions():
            return self.execute_step(
                self._plan_of(task),
                work_item.stage_id,
                lambda stage: self._stage_hash(task, stage),
            )

    def prepare_inputs(
        self,
        binding: ResearchWorkspaceScoreInput,
        *,
        through: date,
        expected_source_hash: str,
        captured_authority_hash: str | None = None,
        observed_at: datetime | None = None,
    ) -> PreparedWorkspaceComponentInputs | None:
        """Hold verified raw/Feature history for this Task's later cutoff projections.

        Args:
            binding: Exact installed component input binding.
            through: Maximum requested formation not already prepared by this Task.
            expected_source_hash: Exact admitted workspace source identity.
            captured_authority_hash: Optional preceding formation's effective authority.
            observed_at: Explicit shared calendar observation; defaults to the owner clock.

        Returns:
            Independently held immutable history, or no prepared value for a recorded
            source or when a later invalid action requires each formation's full reader.

        Raises:
            ValueError: Source proof, qualification or authority checks fail.
        """
        if binding.source_kind != "WORKSPACE_DATA_FEATURE":
            return None
        authority = (
            self._authority(binding)
            if captured_authority_hash is None
            else self._captured_authority(binding, captured_authority_hash)
        )
        training_factor_ids = (
            authority.authority.training_factor_ids
            if isinstance(authority, AdmittedRenewingInference)
            and not authority.authority.supports(through)
            else ()
        )
        scope_hash = workspace_observation_history_scope(binding)
        selection_hash = workspace_observation_history_selection(
            binding, implementation=_implementation_hash()
        )
        with self.session.reads():
            candidate = self.store.load_workspace_observation_history(scope_hash)
            history = (
                candidate
                if candidate is not None and candidate[0].selection_hash == selection_hash
                else None
            )
            try:
                prepared = prepare_workspace_component_inputs(
                    self.session.workspace,
                    through=through,
                    observed_at=self.clock() if observed_at is None else observed_at,
                    expected_source_hash=expected_source_hash,
                    ordered_feature_ids=authority.authority.model_set.ordered_feature_ids,
                    training_factor_ids=training_factor_ids,
                    history=history,
                )
            except ValueError as error:
                if str(error) not in {
                    "causal execution dividend is invalid",
                    "causal execution split is invalid",
                    "causal execution encountered an unsupported corporate action",
                }:
                    raise
                # A max-cutoff capture must not move an invalid later action
                # into an earlier legal formation. The original per-cutoff
                # reader will accept or refuse at the action's own effective date.
                return None
        assert prepared.dependency_prefix_hash is not None
        columns = workspace_observation_history_columns(prepared)
        with self.session.mutation_gate.hold():
            self.store.publish_workspace_observation_history(
                scope_hash=scope_hash,
                selection_hash=selection_hash,
                dependency_prefix_hash=prepared.dependency_prefix_hash,
                formation_sessions=prepared.formation_sessions,
                ordered_listing_ids=prepared.ordered_listing_ids,
                stable_session_count=prepared.stable_session_count,
                columns=columns,
                reuse=workspace_observation_history_reuse(
                    history[0] if history is not None else None, prepared, columns
                ),
                capacity=lambda size: require_storage_capacity(
                    self.session.workspace, additional_bytes=size
                ),
            )
        return prepared

    def execute_step(
        self,
        plan: StrategyScorePlan,
        stage: str,
        prior: Callable[[str], str],
        *,
        captured_authority_hash: str | None = None,
        prepared_inputs: PreparedWorkspaceComponentInputs | None = None,
    ) -> StageExecutionResult:
        """Execute domain work; the caller owns Task stages and verifies the prefix."""
        self._require_plan(plan)
        if stage == STAGES[0]:
            if self._source(plan.binding)[0] != plan.source_identity_hash:
                return StageExecutionResult(
                    StageDisposition.BLOCKED, failure_code="strategy_score.stale_plan"
                )
            self._authority(plan.binding)
            content = plan.plan_hash
        elif stage == STAGES[1]:
            authority = (
                self.authority_for_formation(plan.binding, plan.formation_session)
                if captured_authority_hash is None
                else self._captured_authority(plan.binding, captured_authority_hash)
            )
            if not authority.authority.supports(plan.formation_session):
                if not isinstance(
                    authority, AdmittedRenewingInference
                ) or not authority.authority.can_prepare(plan.formation_session):
                    raise ValueError("strategy_score.training_source_not_admitted")
                if plan.binding.source_kind == "RECORDED_INPUT_SNAPSHOT":
                    observed_training = self.store._load(
                        "lifecycle-training-observations",
                        authority.authority.observations_hash,
                        "content_hash",
                        AlphaTrainingObservations,
                    )
                    if observed_training.observation_hash != plan.source_identity_hash:
                        raise ValueError("strategy_score.recorded_training_source_mismatch")
                else:
                    with self.session.reads():
                        source, training = read_workspace_component_inputs(
                            self.session.workspace,
                            formation=plan.formation_session,
                            observed_at=(
                                self.clock()
                                if prepared_inputs is None
                                else prepared_inputs.observed_at
                            ),
                            expected_source_hash=plan.source_identity_hash,
                            ordered_feature_ids=authority.authority.component.ordered_feature_ids,
                            training_factor_ids=authority.authority.training_factor_ids,
                            prepared=prepared_inputs,
                        )
                    assert training is not None
                    observed_training = publish_component_training_observations(
                        self.store,
                        component=authority.authority.component,
                        source=source,
                        training=training,
                    )
                renewed = renew_lifecycle_admission(
                    self.store,
                    previous=authority.authority,
                    observations=observed_training,
                    through=plan.formation_session,
                    source_policy=(
                        "REVISED_INPUTS_NEW_VINTAGES_ONLY"
                        if plan.binding.source_kind == "WORKSPACE_DATA_FEATURE"
                        else "APPEND_ONLY"
                    ),
                )
                authority = AdmittedRenewingInference(self.store.root.parent, renewed)
            if plan.binding.source_kind == "RECORDED_INPUT_SNAPSHOT":
                source = self.store.load_frozen_observations(plan.source_identity_hash)
                observation = self.store.load_frozen_observation_snapshot(plan.source_identity_hash)
            else:
                with self.session.reads():
                    source = build_workspace_score_inputs(
                        self.session.workspace,
                        formation=plan.formation_session,
                        observed_at=(
                            self.clock() if prepared_inputs is None else prepared_inputs.observed_at
                        ),
                        expected_source_hash=plan.source_identity_hash,
                        ordered_feature_ids=authority.authority.model_set.ordered_feature_ids,
                        prepared=prepared_inputs,
                    )
                observation = self.store.publish_frozen_observations(
                    source, disposition="RECORDED_INPUT_QA"
                )
                source = self.store.load_frozen_observations(observation.snapshot_hash)
            surfaces = authority.features(source, plan.formation_session)
            prepared = self.store.publish_frozen_feature_preparation(
                observation=observation,
                authority_hash=authority.authority.authority_hash,
                formation=plan.formation_session,
                surfaces=surfaces,
            )
            content = prepared.preparation_hash
        elif stage == STAGES[2]:
            score = self._published(plan)
            if score is None:
                prepared, surfaces = self.store.load_frozen_feature_preparation(prior(STAGES[1]))
                authority = self._captured_authority(
                    plan.binding, prepared.inference_authority_hash
                )
                observed = self.store.load_frozen_observations(prepared.observation_snapshot_hash)
                momentum = observed.formula_values.get("mom_252_21")
                projection = authority.score(
                    formation=plan.formation_session,
                    listing_ids=prepared.ordered_listing_ids,
                    eligible=(
                        np.ones(len(prepared.ordered_listing_ids), dtype=np.bool_)
                        if observed.reference_eligible is None
                        else observed.reference_eligible[
                            observed.formation_sessions.index(plan.formation_session)
                        ]
                    ),
                    surfaces=surfaces,
                    raw_12_1_momentum=(
                        None
                        if momentum is None
                        else momentum[observed.formation_sessions.index(plan.formation_session)]
                    ),
                )
                score = seal_current_contract(
                    FrozenComponentScoreSnapshot,
                    {
                        "request_hash": plan.plan_hash,
                        "strategy_package_hash": plan.binding.strategy_package_hash,
                        "component_recipe_hash": authority.authority.model_set.recipe_hash,
                        "inference_authority_hash": authority.authority.authority_hash,
                        "observation_snapshot_hash": prepared.observation_snapshot_hash,
                        "formation_session": plan.formation_session,
                        "ordered_listing_ids": prepared.ordered_listing_ids,
                        "feature_values_hashes": prepared.feature_values_hashes,
                        "model_identity_hashes": projection.model_identity_hashes,
                        "projection_hash": projection.projection_hash,
                        "scores": tuple(
                            float(value) if np.isfinite(value) else None
                            for value in projection.scores
                        ),
                        "live": tuple(bool(value) for value in projection.live),
                        "prediction_calls": authority.prediction_owner.predictions,
                        "fit_calls": authority.fit_calls
                        if isinstance(authority, AdmittedRenewingInference)
                        else 0,
                        "model_set_publication_hash": authority.model_set_publication_hash
                        if isinstance(authority, AdmittedRenewingInference)
                        else None,
                    },
                    "snapshot_hash",
                )
                self.store.publish_frozen_component_score(score)
            content = score.snapshot_hash
        elif stage == STAGES[3]:
            content = prior(STAGES[2])
            self.store.load_frozen_component_score(content)
        else:
            raise ValueError("strategy_score.stage_unknown")
        return StageExecutionResult(StageDisposition.READY, evidence=self._evidence(stage, content))

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Verify exact scoring-stage evidence under lifecycle admission.

        Args:
            task: Exact admitted scoring task.
            execution: Current execution declaration.
            work_item: Exact workflow stage.
            evidence: Declared stage evidence tuple.

        Returns:
            Unchanged stage evidence after deterministic verification.
        """
        with verified_lifecycle_admissions():
            return self.verify_step(self._plan_of(task), work_item.stage_id, evidence)

    def verify_step(
        self, plan: StrategyScorePlan, stage: str, evidence: tuple[TaskEvidence, ...]
    ) -> tuple[TaskEvidence, ...]:
        """Require exact plan, feature/observation or score/model publication bindings.

        Args:
            plan: Exact admitted scoring plan.
            stage: Installed stage identifier.
            evidence: Single-content exact stage evidence tuple.

        Returns:
            Unchanged evidence after authority, source, formation, listing and request checks.

        Raises:
            ValueError: Stage, evidence, feature source or published score/model binding differs.
        """
        self._verify_step(plan, stage, evidence)
        return evidence

    def verify_score_step(
        self, plan: StrategyScorePlan, stage: str, evidence: tuple[TaskEvidence, ...]
    ) -> FrozenComponentScoreSnapshot:
        """Verify a score stage and return the exact snapshot checked by that verification.

        Args:
            plan: Exact admitted scoring plan.
            stage: Installed scoring result stage.
            evidence: Single-content exact stage evidence tuple.

        Returns:
            Validated frozen score snapshot whose publication and request bindings were checked.

        Raises:
            ValueError: Stage, evidence, score source or published model binding differs.
        """
        if stage not in STAGES[2:]:
            raise ValueError("strategy_score.stage_unknown")
        score = self._verify_step(plan, stage, evidence)
        assert score is not None
        return score

    def _verify_step(
        self, plan: StrategyScorePlan, stage: str, evidence: tuple[TaskEvidence, ...]
    ) -> FrozenComponentScoreSnapshot | None:
        if len(evidence) != 1:
            raise ValueError("strategy_score.evidence_invalid")
        content = evidence[0].content_hash
        score: FrozenComponentScoreSnapshot | None = None
        if stage == STAGES[0]:
            self._authority(plan.binding)
            if content != plan.plan_hash:
                raise ValueError("strategy_score.plan_evidence_invalid")
        elif stage == STAGES[1]:
            prepared, _ = self.store.load_frozen_feature_preparation(content)
            self._captured_authority(plan.binding, prepared.inference_authority_hash)
            self.store.load_frozen_observations(prepared.observation_snapshot_hash)
            observation = self.store.load_frozen_observation_snapshot(
                prepared.observation_snapshot_hash
            )
            source_identity = (
                observation.snapshot_hash
                if plan.binding.source_kind == "RECORDED_INPUT_SNAPSHOT"
                else observation.source_binding_hash
            )
            if (
                prepared.formation_session != plan.formation_session
                or source_identity != plan.source_identity_hash
                or prepared.ordered_listing_ids != observation.ordered_listing_ids
            ):
                raise ValueError("strategy_score.feature_evidence_invalid")
        elif stage in STAGES[2:]:
            score = self.store.load_frozen_component_score(content)
            self._verify_model_publication(score, plan.binding)
            if (
                score.request_hash != plan.plan_hash
                or score.strategy_package_hash != plan.binding.strategy_package_hash
            ):
                raise ValueError("strategy_score.result_evidence_invalid")
        else:
            raise ValueError("strategy_score.stage_unknown")
        if evidence != self._evidence(stage, content):
            raise ValueError("strategy_score.evidence_invalid")
        return score

    def _verify_model_publication(
        self, score: FrozenComponentScoreSnapshot, binding: ResearchWorkspaceScoreInput
    ) -> None:
        if score.model_set_publication_hash is None:
            if score.inference_authority_hash != binding.authority_hash:
                raise ValueError("strategy_score.publication_evidence_mismatch")
            return
        effective = self._captured_authority(binding, score.inference_authority_hash)
        value = self.store._load(
            "lifecycle-model-sets",
            score.model_set_publication_hash,
            "content_hash",
            AlphaModelSetPublication,
        )
        if (
            value.lifecycle.vintages(value.formation)
            != value.lifecycle.vintages(score.formation_session)
            or not recipe_hashes_match(value.component.recipe_hash, score.component_recipe_hash)
            or len(value.children) != len(score.model_identity_hashes)
            or tuple(c.estimator_hash for c in value.children) != score.model_identity_hashes
        ):
            raise ValueError("strategy_score.model_publication_binding_invalid")
        if (
            not isinstance(effective, AdmittedRenewingInference)
            or value.lifecycle != effective.authority.lifecycle
        ):
            raise ValueError("strategy_score.model_publication_lifecycle_invalid")
        verify_model_set(self.store, value)

    def published_score(self, task: TaskRecord) -> FrozenComponentScoreSnapshot | None:
        """Read the exact Task publication once, including for downstream consumers.

        Walking every score file for every historical Task makes calibration's
        history read quadratic. The terminal evidence already names the artifact;
        verify that direct reference against the captured request instead.
        """
        plan = self._plan_of(task, require_current=False)
        if task.lifecycle is not TaskLifecycle.SUCCEEDED:
            return None
        score = self.store.load_frozen_component_score(self._stage_hash(task, STAGES[3]))
        self._verify_model_publication(score, plan.binding)
        if (
            score.request_hash != plan.plan_hash
            or score.strategy_package_hash != plan.binding.strategy_package_hash
        ):
            raise ValueError("strategy_score.publication_evidence_mismatch")
        return score

    def published_scores(self) -> tuple[FrozenComponentScoreSnapshot, ...]:
        """Verified standalone publications plus the composed advancement's terminal receipts."""
        values = tuple(
            score
            for task in self.session.task_control_registry.tasks()
            if task.task_kind == TASK_KIND and (score := self.published_score(task)) is not None
        )
        return values + (
            () if self.additional_publications is None else self.additional_publications()
        )

    def readback(self, task_id: UUID | None = None) -> dict[str, object]:
        """Read one retained scoring task and its exact verified published score.

        Args:
            task_id: Optional exact task; defaults to the latest scoring task.

        Returns:
            Task state or exact score snapshot and unit-bearing readout.
        """
        registry = self.session.task_control_registry
        task = (
            registry.task(task_id)
            if task_id
            else next(
                (task for task in reversed(registry.tasks()) if task.task_kind == TASK_KIND), None
            )
        )
        if task is None:
            return {"status": "NO_SCORE_PUBLICATION", "task_id": None}
        plan = self._plan_of(task, require_current=False)
        result = self.published_score(task)
        return {
            "status": "SCORE_PUBLISHED" if result else task.lifecycle.value,
            "task_id": str(task.task_id),
            "strategy_package_id": plan.binding.strategy_package_id,
            "score_snapshot_hash": result.snapshot_hash if result else None,
            "score": result.model_dump(mode="json") if result else None,
            "readout": result.readout().model_dump(mode="json") if result else None,
            "next_requests": {
                "calibration": {
                    "operation": "STRATEGY_CALIBRATION_PLAN",
                    "strategy_package_id": plan.binding.strategy_package_id,
                    "score_snapshot_hash": result.snapshot_hash,
                }
            }
            if result
            else {},
        }


@dataclass
class StrategyScoreCommand:
    """Dispatch one explicitly prepared declared component scoring plan."""

    application: StrategyScoringApplication
    plan: StrategyScorePlan | None = None
    command_kind: str = TASK_KIND

    def admit(self) -> CommandAdmission:
        """Require an explicit declared component scoring plan before deterministic admission.

        Returns:
            Task admission from the owning application.

        Raises:
            ValueError: No exact plan is supplied.
        """
        if self.plan is None:
            raise ValueError("strategy_score.plan_required")
        return self.application.admit(self.plan)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute the exact task admitted for this declared component scoring command.

        Args:
            task_id: Exact admitted task.
            expected_task_hash: Optional optimistic task identity.
        """
        self.application.execute(task_id, expected_task_hash=expected_task_hash)
