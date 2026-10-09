"""Task-owned preparation and explicit non-default installation of research strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceArtifact,
    ResearchWorkspaceManifest,
    admit_research_workspace,
    manifest_fields_hash,
    read_research_workspace_manifest,
    update_research_workspace_manifest,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    confined,
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.frozen_portfolio import (
    RISK_HISTORY_SESSIONS,
    FrozenPortfolioPreparationRequest,
    local_research_economic_schedule,
    prepare_frozen_portfolio_authority,
    risk_history_shortfall,
    risk_return_surface,
)
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
from alphalattice.control.task_control.registry import LEDGER_REBUILT_DETAIL, LEDGER_REBUILT_NEXT
from alphalattice.control.task_control.runner import (
    StageDisposition,
    StageExecutionResult,
)
from alphalattice.interface.local_application.dispatcher import (
    CommandAdmission,
    LocalBackgroundDispatcher,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    FROZEN_RESEARCH_BOOK_RECIPES,
    install_frozen_strategies,
)
from alphalattice.investment.portfolio_strategy_lab.policies.lifecycle_research import (
    ARTIFACT_KEY,
    CATEGORY,
    LifecyclePortfolioAuthority,
    LifecycleResearchScoreSource,
)
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioResearchArtifactStore,
)
from alphalattice.investment.risk_research.experiments.window import REQUIRED_LOOKBACK_SESSIONS
from alphalattice.investment.risk_research.surfaces.returns import CausalRiskReturnReader
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.sealing import seal_model
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash

TASK_KIND = "research_strategy_preparation"
IMPLEMENTATION_ROLE = "product_host.research_strategy_preparation"
PLAN_FIELDS = (
    "calibration_inputs",
    "decision_updates",
    "default_score_source_mode",
    "default_strategy_package_id",
    "experiment_inputs",
    "score_inputs",
    "strategy_artifacts",
    "strategy_installation",
)
"""The manifest fields a strategy plan reads and its installation writes, and no others:
the installed strategy, its defaults and the bindings an installation must find empty,
and the inputs it was prepared from. A publication of other fields (a model's training
inputs, a data update) leaves the plan applicable (V180)."""
"""A move of `implementation_hash` recorded in `config/identity-successors.json` keeps a
plan sealed under its predecessor preparable (binding plan, B14)."""
STAGE = "materialize_verified_lifecycle_portfolio_inputs"


def implementation_hash() -> str:
    """Read the installed semantic closure for local frozen-recipe input preparation.

    Returns:
        Exact installed implementation identity; no closure record is changed.
    """
    return source_rule_closure_hash(
        root=resolve_playpen_root(Path(__file__)),
        semantic_owner="product_host",
        numerical_role="LOCAL_FROZEN_RECIPE_INPUT_PREPARATION",
        tracked_paths=tuple(
            "src/alphalattice/" + p
            for p in (
                "control/product_host/data_preparation/research_strategy.py",
                "control/product_host/research_authoring/frozen_portfolio.py",
                "control/product_host/research_authoring/portfolio_handoff.py",
                "investment/alpha_research/experiments/lifecycle_authoring.py",
                "investment/alpha_research/scores/model_renewal.py",
                "investment/portfolio_strategy_lab/policies/lifecycle_research.py",
                "investment/portfolio_strategy_lab/policies/buffered_rank_return.py",
                "investment/portfolio_strategy_lab/policies/installed_strategies.py",
                "investment/portfolio_strategy_lab/application/research_experiment.py",
                "capabilities/portfolio_inputs/tradability/surface.py",
                "foundation/market_data_ops/publication/session_marks.py",
                "foundation/causal_outcomes/execution/readers.py",
                "foundation/causal_outcomes/execution/methods.py",
                "foundation/causal_outcomes/execution/compile.py",
                "foundation/causal_outcomes/execution/contracts.py",
            )
        ),
    )


def _require_promoted(row: dict[str, Any]) -> None:
    """A strategy is prepared and installed only from studies on the whole universe: an
    exploration study is refused until it is promoted (binding plan, B17)."""

    if row.get("research_lane") == "EXPLORATION":
        raise ValueError(f"research_lane.exploration_not_promoted:{row['task_id']}")


class ResearchStrategyPlan(BaseModel):  # type: ignore[misc]
    """Seal local frozen-recipe input preparation and exact output home."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    workspace_manifest_hash: str
    request: FrozenPortfolioPreparationRequest
    implementation_hash: str
    artifact_root_relative: str | None = Field(
        default=None,
        pattern=r"^artifacts/research-strategy-inputs/[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    plan_hash: str

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal explicit local frozen-recipe input preparation and exact output home.

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
        """Require exact canonical plan_hash over the declared fields.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Serialized fields differ from plan_hash.
        """
        if self.plan_hash != canonical_hash(self.model_dump(mode="json", exclude={"plan_hash"})):
            raise ValueError("research_strategy.plan_invalid")
        return self


def _contract(plan: ResearchStrategyPlan, caller: str):  # type: ignore[no-untyped-def]
    envelope = TaskInputEnvelope.create(
        task_kind=TASK_KIND,
        input_schema_id="research-strategy-input",
        payload={"plan": plan.model_dump(mode="json"), "caller": caller},
    )
    goal = ResearchGoal.create(
        goal_kind="PREPARE_RESEARCH_STRATEGY",
        input_hash=envelope.input_hash,
        deliverable_kind="LocalLifecyclePortfolioAuthority",
        summary=(
            "Verify local lifecycle results and prepare frozen Portfolio inputs; "
            "no fit, book or current activation."
        ),
    )
    workflow = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash((STAGE,)),
        verifier_catalog_hash=canonical_hash((STAGE,)),
        work_items=(
            WorkItemDefinition.create(
                stage_id=STAGE,
                dependency_ids=(),
                verifier_id="research_strategy.materialized_inputs",
            ),
        ),
    )
    return envelope, goal, workflow


class ResearchStrategyPreparation:
    """Own local research authority preparation and explicit non-default strategy installation."""

    task_kind = TASK_KIND
    replans = (
        TaskReplan(
            task_kind=TASK_KIND,
            preview="RESEARCH_STRATEGY_PLAN",
            admitting="RESEARCH_STRATEGY_PREPARE",
        ),
    )
    """The re-plan of the Task kind this owner admits, which the recovery view offers
    (V188)."""

    def __init__(
        self,
        session: WorkspaceApplicationSession,
        *,
        clock: Callable[[], datetime],
        read_experiment: Callable[[UUID], dict[str, Any]],
        list_experiments: Callable[[], dict[str, Any]],
    ):
        """Wire retained task session and exact completed-study readers.

        Args:
            session: Retained workspace writer/task session.
            clock: Explicit observed-time source.
            read_experiment: Deterministic exact study readback callback.
            list_experiments: Deterministic retained study metadata callback.
        """
        self.session, self.clock, self.read_experiment = session, clock, read_experiment
        self.list_experiments = list_experiments
        self.last_plan: ResearchStrategyPlan | None = None
        # Every plan an answer named, by its hash, sealed on disk until it expires: a run
        # from any of them, after a restart too, reopens it and checks it again (V493, V525).
        self._plans: PreviewRegistry[ResearchStrategyPlan] = PreviewRegistry(
            model=ResearchStrategyPlan,
            clock=self.clock,
            root=self.session.workspace / "runtime" / PLAN_PREVIEWS_DIRECTORY / "research-strategy",
        )

    def replan_requests(self, plan_hash: str) -> dict[str, object]:
        """Bind re-planning to the declaration of a verified retained plan.

        Args:
            plan_hash: The refused plan's exact hash.

        Returns:
            Its bound declaration, or no offer when the source cannot be verified.
        """
        kept = self._plans.get(plan_hash)
        if kept is None:
            return {}
        return {
            "replan": {
                "operation": "RESEARCH_STRATEGY_PLAN",
                "experiment_document": kept.plan.request.model_dump(mode="json"),
            }
        }

    def replan_request(self, task: TaskRecord) -> dict[str, object]:
        """Keep the preparation's verified declaration when its ledger is rebuilt."""
        return {
            "operation": "RESEARCH_STRATEGY_PLAN",
            "experiment_document": self._of(task).request.model_dump(mode="json"),
        }

    def controls(self, input_binding_hash: str | None = None) -> dict[str, object]:
        """Project installed preparation declarations and exact completed parent selections.

        Returns:
            Required component set, completed Alpha/Risk task choices and an explicit authored PLAN
            request; no Portfolio is run.
        """
        rows = self.list_experiments()["experiments"]
        completed = [
            row
            for row in rows
            if row["lifecycle"] == "SUCCEEDED"
            and (input_binding_hash is None or row.get("input_binding_hash") == input_binding_hash)
        ]
        recipes = FROZEN_RESEARCH_BOOK_RECIPES
        required = sorted({v.component_id for recipe in recipes for v in recipe.components})
        alphas = [row for row in completed if row.get("component_recipe_id")]
        held = {row["component_recipe_id"] for row in alphas}
        missing = [component for component in required if component not in held]
        risks = [row for row in completed if row["kind"] == "risk.covariance-development"]
        windows, formations = self._risk_windows(alphas)
        # A completed Risk study covers until a calibrated Alpha study names its window.
        covering = []
        for risk in risks:
            if not all(
                str(risk["sessions"]["start"]) <= window["start"]
                and str(risk["sessions"]["end"]) >= window["end"]
                for window in windows
            ):
                continue
            if windows:
                try:
                    root, surface = risk_return_surface(
                        self.session.workspace, self.read_experiment(UUID(str(risk["task_id"])))
                    )
                    if (
                        risk_history_shortfall(
                            CausalRiskReturnReader(root).available_sessions(surface), formations
                        )
                        is not None
                    ):
                        continue
                except (KeyError, OSError, ValueError):
                    continue
            covering.append(risk)
        return {
            "status": "AVAILABLE",
            "declaration_schema": FrozenPortfolioPreparationRequest.model_json_schema(),
            "required_components": required,
            "alpha_tasks": alphas,
            "risk_tasks": risks,
            # The components no completed lifecycle study holds yet, each with its first step.
            "missing_components": missing,
            # The window the Risk study's `experiment.sessions` must cover, per calibrated Alpha
            # study; empty until one completes (V533, FLOW-3).
            "risk_windows": windows,
            # The declaration is the reader's to write (V136).
            "next_requests": {
                **self._component_routes(missing, input_binding_hash),
                **({} if covering else {"risk": self._risk_route(input_binding_hash)}),
                "plan": {"operation": "RESEARCH_STRATEGY_PLAN", "experiment_document": None},
            },
            "claim": (
                "Select exact completed parents on the same input; "
                "preparation does not run a Portfolio."
            ),
        }

    def _selector(self, input_binding_hash: str | None = None) -> dict[str, object]:
        """Bind the declaration's input, or the workspace's sole input when unselected."""
        inputs = read_research_workspace_manifest(self.session.workspace).experiment_inputs or ()
        if input_binding_hash is not None:
            return {
                "research_input_id": next(
                    (v.input_id for v in inputs if v.binding_hash == input_binding_hash), None
                ),
                "input_binding_hash": input_binding_hash,
            }
        return (
            {"research_input_id": inputs[0].input_id, "input_binding_hash": inputs[0].binding_hash}
            if len(inputs) == 1
            else {"research_input_id": None}
        )

    def _risk_route(self, input_binding_hash: str | None = None) -> dict[str, object]:
        """The Risk study's controls on the input, its window from `risk_windows`."""
        return {
            "operation": "EXPERIMENT_CONTROLS",
            **self._selector(input_binding_hash),
            "experiment_kind": "risk.covariance-development",
        }

    def _economic_schedule(self, alpha: Mapping[str, Any]) -> tuple[Any, ...]:
        """The calibrated Alpha study's matured economic formations, as preparation reads them."""
        bundle = read_factor_bundle(self.session.workspace, str(alpha["input_binding_hash"]))
        return tuple(
            local_research_economic_schedule(
                first=date.fromisoformat(str(alpha["sessions"]["start"])),
                last=date.fromisoformat(str(alpha["sessions"]["end"])),
                through=bundle.sessions[-1],
            )
        )

    def _risk_windows(
        self, alphas: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], tuple[date, ...]]:
        """Each completed calibrated Alpha study's window a Risk study must cover (V533)."""
        calibrated = {
            v.component_id
            for recipe in FROZEN_RESEARCH_BOOK_RECIPES
            for v in recipe.components
            if v.weight_rule == "mu.iv0"
        }
        windows = []
        formations: list[date] = []
        for alpha in alphas:
            if alpha.get("component_recipe_id") not in calibrated:
                continue
            try:
                schedule = self._economic_schedule(alpha)
            except (KeyError, OSError, ValueError):
                continue
            if schedule:
                calendar = read_factor_bundle(
                    self.session.workspace, str(alpha["input_binding_hash"])
                ).sessions
                if schedule[0].formation_session not in calendar:
                    continue
                first = calendar.index(schedule[0].formation_session)
                formations.extend(point.formation_session for point in schedule)
                history_start = first - RISK_HISTORY_SESSIONS + REQUIRED_LOOKBACK_SESSIONS
                windows.append(
                    {
                        "alpha_task_id": str(alpha["task_id"]),
                        "start": schedule[0].formation_session.isoformat(),
                        "end": schedule[-1].formation_session.isoformat(),
                        "formation_history": {
                            "start": calendar[history_start].isoformat()
                            if history_start >= 0
                            else None,
                            "end": calendar[first - 1].isoformat() if first else None,
                            "required_return_sessions": RISK_HISTORY_SESSIONS,
                        },
                    }
                )
        return windows, tuple(formations)

    def _component_routes(
        self, missing: list[str], input_binding_hash: str | None = None
    ) -> dict[str, dict[str, object]]:
        """Each missing component's first step toward its lifecycle study (V505, RR5).

        Its study's controls once its training inputs are prepared on the input, else the
        plan of those inputs. The input is bound when the workspace has one, and left to
        choose otherwise.
        """
        manifest = read_research_workspace_manifest(self.session.workspace)
        selector = self._selector(input_binding_hash)
        trained = {
            v.component_id
            for v in manifest.model_training_inputs or ()
            if v.input_binding_hash == selector.get("input_binding_hash")
        }
        return {
            f"component:{component}": (
                {
                    "operation": "EXPERIMENT_CONTROLS",
                    **selector,
                    "experiment_kind": "alpha.model-development",
                    "component_id": component,
                }
                if component in trained
                else {
                    "operation": "MODEL_TRAINING_INPUT_PLAN",
                    **selector,
                    "component_id": component,
                }
            )
            for component in missing
        }

    def _root(self, plan: ResearchStrategyPlan) -> Path:
        return confined(
            self.session.workspace,
            plan.artifact_root_relative or f"artifacts/research-strategy-inputs/{plan.plan_hash}",
        )

    def plan(self, document: dict[str, Any]) -> dict[str, object]:
        """Require exact promoted parents and matured support before preparation.

        Require promoted completed parents and matured economic support before sealing preparation.

        Args:
            document: Explicit frozen portfolio preparation declaration.

        Returns:
            Exact retained plan, source support/counts and legal preparation request; no fit or
            Portfolio call.

        Raises:
            ValueError: Parent uniqueness, input/kind/component/promotion or matured Alpha/Risk
                support is invalid.
        """
        require_no_pending_cleanup(self.session.workspace)
        request = FrozenPortfolioPreparationRequest.model_validate(document)
        rows = {row["task_id"]: row for row in self.list_experiments()["experiments"]}
        if len(set(request.alpha_task_ids)) != len(request.alpha_task_ids):
            raise ValueError("research_strategy.duplicate_parent")
        for task_id in (*request.alpha_task_ids, request.risk_task_id):
            task = self.session.task_control_registry.task(task_id)
            if task.lifecycle is not TaskLifecycle.SUCCEEDED:
                raise ValueError("research_strategy.completed_parent_required")
            row = rows.get(str(task_id), {})
            if row.get("input_binding_hash") != request.input_binding_hash:
                raise ValueError("research_strategy.parent_input_mismatch")
            _require_promoted(row)
        expected = {
            v.component_id for recipe in FROZEN_RESEARCH_BOOK_RECIPES for v in recipe.components
        }
        selected = [rows[str(task)].get("component_recipe_id") for task in request.alpha_task_ids]
        if (
            set(selected) != expected
            or len(selected) != len(expected)
            or rows[str(request.risk_task_id)]["kind"] != "risk.covariance-development"
        ):
            raise ValueError("research_strategy.parent_kind_or_component_mismatch")
        bundle = read_factor_bundle(self.session.workspace, request.input_binding_hash)
        calibrated = {
            v.component_id
            for recipe in FROZEN_RESEARCH_BOOK_RECIPES
            for v in recipe.components
            if v.weight_rule == "mu.iv0"
        }
        primary = next(
            row
            for row in rows.values()
            if row.get("component_recipe_id") in calibrated
            and row["task_id"] in {str(v) for v in request.alpha_task_ids}
        )
        schedule = self._economic_schedule(
            {**primary, "input_binding_hash": request.input_binding_hash}
        )
        if not schedule:
            raise ValueError("research_strategy.matured_economic_support_absent")
        risk_period = rows[str(request.risk_task_id)]["sessions"]
        risk_start = date.fromisoformat(str(risk_period["start"]))
        risk_end = date.fromisoformat(str(risk_period["end"]))
        needed = (schedule[0].formation_session, schedule[-1].formation_session)
        if risk_start > needed[0] or risk_end < needed[1]:
            # The window the strategy needs and the one its Risk study covers, so the way on
            # is a Risk study over that window (V533, RR5d).
            raise ValueError(
                "research_strategy.risk_parent_support_incomplete:"
                f"needed {needed[0]}..{needed[1]}, Risk study {risk_start}..{risk_end}"
            )
        # The Risk history every formation needs before it, judged by the one owner the
        # materialization reads, so a plan admits only what its preparation runs (V596).
        surface_root, surface = risk_return_surface(
            self.session.workspace, self.read_experiment(request.risk_task_id)
        )
        short = risk_history_shortfall(
            CausalRiskReturnReader(surface_root).available_sessions(surface),
            (point.formation_session for point in schedule),
        )
        if short is not None:
            first, held = short
            calendar = list(bundle.sessions)
            earlier = calendar.index(risk_start) - (RISK_HISTORY_SESSIONS - held)
            start = (
                f"a Risk study starting on or before {calendar[earlier]}"
                if risk_start in calendar and earlier >= 0
                else "an input with more history before its first formation"
            )
            raise ValueError(
                "research_strategy.risk_history_insufficient:"
                f"formation {first} has {held} Risk return sessions before it, "
                f"{RISK_HISTORY_SESSIONS} needed, {start}"
            )
        # Same declared inputs retain their content-addressed output home even
        # when a code correction requires a new plan. Existing files are still
        # independently validated by their publishers; no old Task is relabelled.
        previous = sorted(
            (
                task
                for task in self.session.task_control_registry.tasks()
                if task.task_kind == TASK_KIND and self._of(task).request == request
            ),
            key=lambda task: task.admitted_at,
        )
        request_hash = canonical_hash(request.model_dump(mode="json"))
        output_root = (
            self._root(self._of(previous[0])).relative_to(self.session.workspace).as_posix()
            if previous
            else f"artifacts/research-strategy-inputs/{request_hash}"
        )
        # The answer names this plan, never the owner's last one, which a concurrent plan
        # may have replaced in between (V534).
        planned = ResearchStrategyPlan.create(
            workspace_manifest_hash=manifest_fields_hash(
                read_research_workspace_manifest(self.session.workspace), PLAN_FIELDS
            ),
            request=request,
            implementation_hash=implementation_hash(),
            artifact_root_relative=output_root,
        )
        self.last_plan = planned
        self._plans.remember(planned)
        return {
            "status": "PLANNED",
            "plan_hash": planned.plan_hash,
            "declaration": request.model_dump(mode="json"),
            "source_start": str(bundle.sessions[0]),
            "source_end": str(bundle.sessions[-1]),
            "fit_calls": 0,
            "portfolio_calls": 0,
            "economic_scope": "POST_OBSERVED_LOCAL_QA_NOT_HOLDOUT_RELEASE",
            "expected_economic_support": {
                "start": str(schedule[0].formation_session),
                "end": str(schedule[-1].formation_session),
                "count": len(schedule),
            },
            "claim": (
                "PREPARE_LOCAL_RESEARCH_AUTHORITY; install and Portfolio PLAN/RUN remain separate"
            ),
            "next_requests": {
                "prepare": {
                    "operation": "RESEARCH_STRATEGY_PREPARE",
                    "experiment_plan_hash": planned.plan_hash,
                }
            },
        }

    def _of(self, task: TaskRecord) -> ResearchStrategyPlan:
        if task.task_kind != TASK_KIND:
            raise ValueError("research_strategy.task_kind_mismatch")
        plan: ResearchStrategyPlan = ResearchStrategyPlan.model_validate(task.input.payload["plan"])
        if task.task_kind != TASK_KIND or (task.input, task.goal, task.plan) != _contract(
            plan, str(task.input.payload["caller"])
        ):
            raise ValueError("research_strategy.task_contract_invalid")
        return plan

    def _require(self, plan: ResearchStrategyPlan) -> None:
        if not is_current(
            IMPLEMENTATION_ROLE, plan.implementation_hash, implementation_hash()
        ) or plan.workspace_manifest_hash != manifest_fields_hash(
            read_research_workspace_manifest(self.session.workspace), PLAN_FIELDS
        ):
            raise ValueError("research_strategy.execution_changed_replan")

    def prepare(
        self, plan_hash: str, *, caller: str, dispatcher: LocalBackgroundDispatcher
    ) -> dict[str, object]:
        """Reuse verified local authority, recover its exact task or dispatch its retained preview.

        Args:
            plan_hash: Exact retained preparation plan.
            caller: Declared operation caller.
            dispatcher: Bounded local dispatcher.

        Returns:
            Verified reuse, explicit missing-outcome refusal, retained recovery or new submission
            metadata.

        Raises:
            ValueError: Exact preview or current retained plan authority is absent.
        """
        existing = next(
            (
                t
                for t in self.session.task_control_registry.tasks()
                if t.task_kind == TASK_KIND
                and self._of(t).plan_hash == plan_hash
                and t.lifecycle is not TaskLifecycle.CANCELLED
            ),
            None,
        )
        if existing is not None:
            if existing.lifecycle is TaskLifecycle.SUCCEEDED:
                prior = self.readback(existing.task_id)
                if prior.get("gap"):
                    return {**prior, "status": "REFUSED", "failure_code": prior["gap"]}
                return {
                    **prior,
                    "status": "REUSED_EXACT",
                    "task_id": None,
                    "publication_task_id": str(existing.task_id),
                }
            if existing.lifecycle is TaskLifecycle.BLOCKED:
                self._require(self._of(existing))
                self.session.task_control_registry.mark_recovery_required(
                    task_id=existing.task_id,
                    failure_code=existing.failure_code or "research_strategy.retry_requested",
                    observed_at=self.clock(),
                    allow_blocked=True,
                )
            if existing.lifecycle in {TaskLifecycle.BLOCKED, TaskLifecycle.RECOVERY_REQUIRED}:
                # This plan's Task only, never another of its kind (V529).
                dispatcher.resume(
                    {TASK_KIND: ResearchStrategyCommand(self)}, only_task_id=existing.task_id
                )
            return self.readback(existing.task_id)
        plan = self._plans.runnable(plan_hash)
        if plan is None:
            raise ValueError("research_strategy.preview_required")
        sent = dispatcher.submit(ResearchStrategyCommand(self, plan, caller))
        return {
            "status": sent.disposition,
            "task_id": str(sent.task_id) if sent.task_id else None,
            "failure_code": sent.refusal_detail,
            "lifecycle": sent.lifecycle,
            "next_requests": {
                "readback": {
                    "operation": "RESEARCH_STRATEGY_READBACK",
                    "task_id": str(sent.task_id),
                }
            }
            if sent.task_id
            else {},
        }

    def admit(self, plan: ResearchStrategyPlan, caller: str) -> CommandAdmission:
        """Require current preparation scope and exclusive task ownership under the mutation gate.

        Args:
            plan: Explicit sealed research strategy plan.
            caller: Declared submitting caller.

        Returns:
            Exact reused or newly admitted preparation task.

        Raises:
            ValueError: Plan/cleanup is invalid or another task must finish or recover.
        """
        with self.session.mutation_gate.hold():
            self._require(plan)
            require_no_pending_cleanup(self.session.workspace)
            registry = self.session.task_control_registry
            for task in registry.tasks():
                if (
                    task.task_kind == TASK_KIND
                    and self._of(task).plan_hash == plan.plan_hash
                    and task.lifecycle is not TaskLifecycle.CANCELLED
                ):
                    return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)
                if task.lifecycle not in {
                    TaskLifecycle.SUCCEEDED,
                    TaskLifecycle.CANCELLED,
                    TaskLifecycle.BLOCKED,
                }:
                    raise ValueError("research_strategy.finish_or_recover_existing_task")
            envelope, goal, workflow = _contract(plan, caller)
            task = registry.admit(
                input_envelope=envelope, goal=goal, plan=workflow, observed_at=self.clock()
            ).record
            return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Bind exact plan schema, workflow, authored request and installed execution identity.

        Args:
            task: Exact admitted preparation task.

        Returns:
            Deterministic execution compatibility declaration.
        """
        plan = self._of(task)
        self._require(plan)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash(schema_structure(ResearchStrategyPlan)),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=canonical_hash(plan.request.model_dump(mode="json")),
            framework_identity_hash=self.session.execution_identity(plan.implementation_hash),
        )

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one exact admitted research strategy preparation task.

        Args:
            task_id: Exact retained task.
            expected_task_hash: Optional optimistic task identity.
        """
        task = self.session.task_control_registry.task(task_id)
        self.session.execute_admitted(task, self, self.clock, expected_task_hash)

    def execute_stage(
        self, *, task: TaskRecord, execution: TaskExecution, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        """Prepare exact frozen research authority with owner-supplied capacity and cancellation.

        Args:
            task: Exact admitted preparation task.
            execution: Current execution declaration.
            work_item: Exact installed preparation stage.

        Returns:
            Ready materialized-input authority evidence, safe-checkpoint cancellation or bounded
            named refusal.
        """
        del execution
        plan = self._of(task)
        try:
            self._require(plan)
            if work_item.stage_id != STAGE:
                raise ValueError("research_strategy.stage_unknown")
            read_factor_bundle(self.session.workspace, plan.request.input_binding_hash)
            # The existing budget owner remains authoritative. The materializer
            # charges actual new packed lanes; shared immutable Parquet stays shared.
            authority = prepare_frozen_portfolio_authority(
                workspace=self.session.workspace,
                request=plan.request,
                output_root=self._root(plan),
                read_experiment=self.read_experiment,
                implementation_hash=plan.implementation_hash,
                observed_at=self.clock(),
                cancelled=lambda: (
                    self.session.task_control_registry.task(task.task_id).lifecycle
                    is TaskLifecycle.CANCEL_REQUESTED
                ),
            )
            return StageExecutionResult(
                StageDisposition.READY,
                evidence=(
                    TaskEvidence(
                        evidence_kind="research_strategy.materialized_inputs",
                        reference=f"playpen://portfolio-strategy-lab/{CATEGORY}/{authority.authority_hash}",
                        content_hash=authority.authority_hash,
                    ),
                ),
            )
        except (ValueError, OSError, RuntimeError) as error:
            if str(error) == "frozen_portfolio.cancelled_at_safe_checkpoint":
                return StageExecutionResult(StageDisposition.CANCELLED)
            return StageExecutionResult(
                StageDisposition.BLOCKED,
                failure_code=str(getattr(error, "failure_code", str(error)))[:240],
            )

    def _authority(
        self, task: TaskRecord, evidence: tuple[TaskEvidence, ...] | None = None
    ) -> LifecyclePortfolioAuthority:
        plan = self._of(task)
        if evidence is None:
            evidence = self.session.task_control_registry.work_items(task.task_id)[0].evidence
        if (
            len(evidence) != 1
            or evidence[0].reference
            != f"playpen://portfolio-strategy-lab/{CATEGORY}/{evidence[0].content_hash}"
        ):
            raise ValueError("research_strategy.evidence_invalid")
        value = PortfolioResearchArtifactStore(self._root(plan)).load(
            category=CATEGORY,
            content_hash=evidence[0].content_hash,
            model=LifecyclePortfolioAuthority,
            identity_field="authority_hash",
        )
        if (
            value.preparation_request_hash != canonical_hash(plan.request.model_dump(mode="json"))
            or value.preparation_implementation_hash != plan.implementation_hash
        ):
            raise ValueError("research_strategy.evidence_binding_mismatch")
        path = (
            PortfolioResearchArtifactStore(self._root(plan)).root
            / CATEGORY
            / f"{value.authority_hash}.json"
        )
        for recipe in FROZEN_RESEARCH_BOOK_RECIPES:
            LifecycleResearchScoreSource(manifest_path=path, recipe=recipe).verify()
        return value

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Require the installed preparation stage and exact retained authority evidence.

        Args:
            task: Exact retained task.
            execution: Current execution declaration.
            work_item: Exact installed preparation stage.
            evidence: Declared materialized-input authority evidence.

        Returns:
            Unchanged verified evidence.

        Raises:
            ValueError: Stage, task plan or exact authority evidence is invalid.
        """
        del execution
        if work_item.stage_id != STAGE:
            raise ValueError("research_strategy.stage_unknown")
        self._authority(task, evidence)
        return evidence

    def readback(self, task_id: UUID) -> dict[str, object]:
        """Read exact prepared authority and any missing complete local QA outcome support.

        Args:
            task_id: Exact retained preparation task.

        Returns:
            Task state, verified authority and explicit replan/install request; preparation grants
            no current activation.
        """
        task = self.session.task_control_registry.task(task_id)
        self._of(task)
        value = self._authority(task) if task.lifecycle is TaskLifecycle.SUCCEEDED else None
        gap = (
            "research_strategy.full_local_qa_outcomes_required"
            if value is not None and value.qa_outcome is None
            else None
        )
        return {
            "status": task.lifecycle.value,
            "task_id": str(task_id),
            "failure_code": task.failure_code,
            **(
                {"detail": LEDGER_REBUILT_DETAIL, "stop_next": LEDGER_REBUILT_NEXT}
                if task.failure_code == "task_control.ledger_rebuilt"
                else {}
            ),
            "authority": value.model_dump(mode="json") if value else None,
            "gap": gap,
            "claim": "INPUTS_READY_NOT_PORTFOLIO_EXECUTED_NO_CURRENT_ACTIVATION",
            "next_requests": {
                "replan": {
                    "operation": "RESEARCH_STRATEGY_PLAN",
                    "experiment_document": self._of(task).request.model_dump(mode="json"),
                    **(
                        {
                            "recovery_task_id": str(task.task_id),
                            "recovery_task_hash": task.record_hash,
                        }
                        if task.lifecycle in {TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED}
                        else {}
                    ),
                }
            }
            if gap or task.lifecycle in {TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED}
            else {
                "install_non_default": {
                    "operation": "RESEARCH_STRATEGY_INSTALL",
                    "task_id": str(task_id),
                }
            }
            if value
            else {},
        }

    def install(self, task_id: UUID) -> dict[str, object]:
        """Install exact completed authority as non-default research under mutation ownership.

        Complete local QA outcomes and promoted parents are required. Existing immutable authority
        stays readable; only the research installation pointer changes, and no default strategy or
        current scoring is installed.

        Args:
            task_id: Exact succeeded preparation task.

        Returns:
            Exact reused or newly installed authority, package identities and required service
            restart.

        Raises:
            ValueError: Cleanup, completion, full outcomes, parent promotion or exact
                configuration/head binding is invalid.
        """
        with self.session.mutation_gate.hold():
            require_no_pending_cleanup(self.session.workspace)
            task = self.session.task_control_registry.task(task_id)
            if task.lifecycle is not TaskLifecycle.SUCCEEDED:
                raise ValueError("research_strategy.completed_preparation_required")
            authority = self._authority(task)
            if authority.qa_outcome is None:
                raise ValueError("research_strategy.full_local_qa_outcomes_required")
            plan = self._of(task)
            rows = {row["task_id"]: row for row in self.list_experiments()["experiments"]}
            for parent in (*plan.request.alpha_task_ids, plan.request.risk_task_id):
                _require_promoted(rows.get(str(parent), {}))
            store = PortfolioResearchArtifactStore(self._root(plan))
            relative = (
                (store.root / CATEGORY / f"{authority.authority_hash}.json")
                .relative_to(self.session.workspace)
                .as_posix()
            )
            binding = ResearchWorkspaceArtifact(artifact_key=ARTIFACT_KEY, relative_path=relative)
            current = read_research_workspace_manifest(self.session.workspace)
            if (
                current.strategy_installation == "NON_DEFAULT_RESEARCH"
                and current.strategy_artifacts == (binding,)
            ):
                admit_research_workspace(self.session.workspace)
                return {
                    "status": "REUSED_EXACT",
                    "authority_hash": authority.authority_hash,
                }
            if (
                current.strategy_installation not in {"NOT_INSTALLED", "NON_DEFAULT_RESEARCH"}
                or manifest_fields_hash(current, PLAN_FIELDS) != plan.workspace_manifest_hash
            ):
                raise ValueError("research_strategy.existing_installation_or_configuration_changed")
            previous_non_default = None
            if current.strategy_installation == "NON_DEFAULT_RESEARCH":
                # Keep the previous immutable authority readable; validate its binding
                # before atomically replacing only the research installation pointer.
                admit_research_workspace(self.session.workspace)
                previous_non_default = Path(current.strategy_artifacts[0].relative_path).stem
            values = {
                name: getattr(current, name)
                for name in type(current).model_fields
                if name not in {"kind", "manifest_schema", "manifest_hash"}
            }
            values.update(
                strategy_artifacts=(binding,), strategy_installation="NON_DEFAULT_RESEARCH"
            )
            updated = ResearchWorkspaceManifest.create(**values)
            # Validate installation before the atomic manifest publication.
            packages = install_frozen_strategies(
                artifacts={ARTIFACT_KEY: self.session.workspace / relative}
            )

            def install(now: ResearchWorkspaceManifest) -> ResearchWorkspaceManifest:
                # The gate is held since `current` was read, so it is the manifest the
                # one write changes; were it not, nothing is installed.
                if now != current:
                    raise ValueError(
                        "research_strategy.existing_installation_or_configuration_changed"
                    )
                return updated

            update_research_workspace_manifest(
                self.session.workspace, install, gate=self.session.mutation_gate
            )
            return {
                "status": "INSTALLED_NON_DEFAULT_RESEARCH",
                "authority_hash": authority.authority_hash,
                "workspace_manifest_hash": updated.manifest_hash,
                "strategy_package_ids": [v.package.strategy_id for v in packages],
                "default_strategy_package_id": None,
                "current_scoring_installed": False,
                "previous_non_default_authority_hash": previous_non_default,
                "next_action": "READ_THE_PACKAGE_CONTROLS",
                # The running Host serves the package at once: each package's whole-support
                # book, the book whose review its activation reads.
                "next_requests": {
                    f"books:{v.package.strategy_id}": {
                        "operation": "CONTROLS",
                        "strategy_package_id": v.package.strategy_id,
                    }
                    for v in packages
                },
            }


@dataclass
class ResearchStrategyCommand:
    """Dispatch one explicit local research authority preparation plan and declared caller."""

    application: ResearchStrategyPreparation
    plan: ResearchStrategyPlan | None = None
    caller: str = "EXTERNAL_AUTOMATION"
    command_kind: str = TASK_KIND

    def admit(self) -> CommandAdmission:
        """Require the exact local authority preparation preview before admission.

        Require the exact local research authority preparation preview before deterministic
        admission.

        Returns:
            Task admission from the owning application.

        Raises:
            ValueError: Exact preview is absent or declared admission checks refuse.
        """
        if self.plan is None:
            raise ValueError("research_strategy.preview_required")
        return self.application.admit(self.plan, self.caller)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one exact task admitted for local research authority preparation.

        Args:
            task_id: Exact retained task.
            expected_task_hash: Optional optimistic task identity.
        """
        self.application.execute(task_id, expected_task_hash=expected_task_hash)
