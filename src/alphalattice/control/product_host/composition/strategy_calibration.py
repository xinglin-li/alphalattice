"""Compose Portfolio input preparation under the existing dispatcher and Task owner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Self, cast
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from alphalattice.capabilities.portfolio_inputs.tradability.surface import (
    decision_eligible_at_closes,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceCalibrationInput,
    ResearchWorkspaceManifest,
    manifest_fields_hash,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.strategy_score_inputs import (
    workspace_score_source_identity,
)
from alphalattice.control.product_host.composition.strategy_scoring import (
    StrategyScoringApplication,
)
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
from alphalattice.foundation.causal_outcomes.execution.contracts import LocalQAMarketSnapshot
from alphalattice.foundation.causal_outcomes.execution.readers import (
    PreparedLocalQASnapshotRows,
    local_qa_snapshot_rows,
    read_local_qa_execution_rows,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.interface.local_application.dispatcher import CommandAdmission
from alphalattice.investment.alpha_research.publication.contracts import (
    FrozenComponentScoreSnapshot,
)
from alphalattice.investment.alpha_research.scores.model_renewal import (
    verified_lifecycle_admissions,
)
from alphalattice.investment.portfolio_strategy_lab.application.calibration import (
    BoolArray,
    CalibrationObservations,
    PreparedPortfolioBookInput,
    PreparedPortfolioComponentInput,
    compile_portfolio_calibration,
    load_observations,
    publish_consumed_observations,
    select_calibration_scores,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    recipe_hashes_match,
)
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioResearchArtifactStore,
)
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from alphalattice.kernel.shared_kernel.identity import (
    canonical_hash,
    schema_structure,
)
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.sealing import seal_model_from_dump
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash

PLAN_FIELDS = ("calibration_inputs",)
"""The manifest fields a plan reads: the calibration inputs it prepares. A plan binds them, never
the whole manifest, so a publication of fields it does not read leaves it applicable (V223,
OW10)."""

TASK_KIND = "portfolio_calibration_preparation"
STAGES = (
    "verify_inputs",
    "seal_observations",
    "compile_portfolio_input",
    "publish_portfolio_input",
)
RESULT_CATEGORY = "prepared-component-inputs"


IMPLEMENTATION_ROLE = "product_host.strategy_calibration"
"""The role a move of this implementation is recorded under in `config/identity-successors.json`,
so a plan sealed under its predecessor stays current (binding plan R1, LAWS.md ID1)."""


def _implementation_hash() -> str:
    from pathlib import Path

    return source_rule_closure_hash(
        root=resolve_playpen_root(Path(__file__)),
        semantic_owner="product_host",
        numerical_role="FROZEN_CAUSAL_CALIBRATION_QA",
        tracked_paths=tuple(
            "src/alphalattice/" + path
            for path in (
                "control/product_host/composition/strategy_calibration.py",
                "control/product_host/composition/strategy_score_inputs.py",
                "control/product_host/composition/strategy_scoring.py",
                "foundation/causal_outcomes/execution/readers.py",
                "foundation/causal_outcomes/execution/compile.py",
                "foundation/causal_outcomes/execution/methods.py",
                "foundation/market_data_ops/returns/execution.py",
                "foundation/market_data_ops/storage/duckdb.py",
                "kernel/data/calendar.py",
                "capabilities/portfolio_inputs/tradability/surface.py",
                "investment/alpha_research/experiments/development_contracts.py",
                "investment/alpha_research/publication/contracts.py",
                "investment/alpha_research/publication/artifacts.py",
                "investment/portfolio_strategy_lab/application/calibration.py",
                "investment/portfolio_strategy_lab/policies/buffered_rank_return.py",
                "investment/portfolio_strategy_lab/publication/artifacts.py",
                "control/workspace_runtime/content_store.py",
            )
        ),
    )


class CalibrationPlan(BaseModel):  # type: ignore[misc]
    """Seal the exact causal portfolio calibration request, source and installed implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    workspace_manifest_hash: str
    binding: ResearchWorkspaceCalibrationInput
    score_snapshot_hash: str
    history_score_hashes: tuple[str, ...]
    source_hash: str
    implementation_hash: str
    plan_hash: str

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal an explicit causal portfolio calibration plan.

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
        """Require exact causal portfolio calibration plan identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Serialized fields differ from plan_hash.
        """
        if self.plan_hash != canonical_hash(self.model_dump(mode="json", exclude={"plan_hash"})):
            raise ValueError("portfolio_calibration.plan_identity_invalid")
        return self


def _task_contract(plan: CalibrationPlan) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    envelope = TaskInputEnvelope.create(
        task_kind=TASK_KIND,
        input_schema_id="portfolio-calibration-input",
        payload={"plan": plan.model_dump(mode="json")},
    )
    goal = ResearchGoal.create(
        goal_kind="PREPARE_PORTFOLIO_CALIBRATION",
        input_hash=envelope.input_hash,
        deliverable_kind="PreparedPortfolioComponentInput",
        summary="Prepare causal sizing input, not positions.",
    )
    workflow = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash(STAGES),
        verifier_catalog_hash=canonical_hash(STAGES),
        work_items=tuple(
            WorkItemDefinition.create(
                stage_id=stage,
                dependency_ids=STAGES[:i],
                verifier_id=f"portfolio_calibration.{stage}",
            )
            for i, stage in enumerate(STAGES)
        ),
    )
    return envelope, goal, workflow


class StrategyCalibrationApplication:
    """Own admitted causal portfolio calibration tasks and their exact durable publications."""

    task_kind = TASK_KIND
    replans = (
        TaskReplan(
            task_kind=TASK_KIND,
            preview="STRATEGY_CALIBRATION_PLAN",
            admitting="STRATEGY_CALIBRATION_RUN",
        ),
    )
    """The re-plan of the Task kind this owner admits, which the recovery view offers
    (V188)."""
    issued_score_hashes: Callable[[str, str], dict[date, str]] | None = None
    additional_input: (
        Callable[
            [str], tuple[str, PreparedPortfolioComponentInput | PreparedPortfolioBookInput] | None
        ]
        | None
    ) = None

    def __init__(self, *, scoring: StrategyScoringApplication, clock: Callable[[], datetime]):
        """Wire the scoring owner, shared task session and calibration artifact store.

        Args:
            scoring: Deterministic score publication owner.
            clock: Explicit observed-time source.
        """
        self.scoring, self.session, self.clock = scoring, scoring.session, clock
        self.store = PortfolioResearchArtifactStore(self.session.workspace / "artifacts")
        self.last_plan: CalibrationPlan | None = None
        # Every plan an answer named, by its hash, sealed on disk until it expires: a run
        # from any of them, after a restart too, reopens it and checks it again (V493, V525).
        self._plans: PreviewRegistry[CalibrationPlan] = PreviewRegistry(
            model=CalibrationPlan,
            clock=self.clock,
            root=self.session.workspace
            / "runtime"
            / PLAN_PREVIEWS_DIRECTORY
            / "strategy-calibration",
        )

    def replan_requests(self, plan_hash: str) -> dict[str, object]:
        """Bind re-planning to a verified plan's package and score publication.

        Args:
            plan_hash: The refused plan's exact hash.

        Returns:
            Its bound calibration request, or no offer when the source cannot be verified.
        """
        kept = self._plans.get(plan_hash)
        if kept is None:
            return {}
        return {
            "replan": {
                "operation": "STRATEGY_CALIBRATION_PLAN",
                "strategy_package_id": kept.plan.binding.strategy_package_id,
                "score_snapshot_hash": kept.plan.score_snapshot_hash,
            }
        }

    @property
    def manifest(self) -> ResearchWorkspaceManifest:
        """The workspace manifest scoring reads, the one holder the Host refreshes (V182)."""
        return self.scoring.manifest

    def _published_scores(self) -> tuple[FrozenComponentScoreSnapshot, ...]:
        return self.scoring.published_scores()

    def _binding(self, package_id: str) -> ResearchWorkspaceCalibrationInput:
        self._require_local_store()
        if manifest_fields_hash(
            read_research_workspace_manifest(self.session.workspace), PLAN_FIELDS
        ) != manifest_fields_hash(self.manifest, PLAN_FIELDS):
            raise ValueError("portfolio_calibration.workspace_manifest_changed")
        found = [
            v for v in self.manifest.calibration_inputs or () if v.strategy_package_id == package_id
        ]
        if len(found) != 1:
            raise ValueError("portfolio_calibration.not_installed")
        value = found[0]
        if value.strategy_package_hash != self.scoring.packages[package_id].package_hash:
            raise ValueError("portfolio_calibration.package_binding_mismatch")
        return value

    def _require_local_store(self) -> None:
        if not self.store.root.resolve().is_relative_to(self.session.workspace.resolve()):
            raise ValueError("portfolio_calibration.artifact_root_outside_workspace")

    def _source(self, binding: ResearchWorkspaceCalibrationInput) -> str:
        return (
            workspace_score_source_identity(self.session.workspace)
            if binding.source_kind == "WORKSPACE_DATA_FEATURE"
            else binding.seed_hash
        )

    def plan(self, package_id: str, score_hash: str) -> dict[str, object]:
        """Require exact published scores and causal support before sealing calibration.

        Args:
            package_id: Declared installed calibration package.
            score_hash: Exact published score snapshot.

        Returns:
            Sealed plan metadata, formation, activation, lookback and cache status; no predictions
            or fits.

        Raises:
            ValueError: Published score/recipe/source/support or intervening score coverage is
                unavailable.
        """
        binding = self._binding(package_id)
        seed, _, _, _ = load_observations(self.store, binding.seed_hash)
        scores = {v.snapshot_hash: v for v in self._published_scores()}
        if score_hash not in scores:
            raise ValueError("portfolio_calibration.published_score_required")
        score = scores[score_hash]
        if score.strategy_package_hash != binding.strategy_package_hash or not recipe_hashes_match(
            score.component_recipe_hash, seed.component_recipe_hash
        ):
            raise ValueError("portfolio_calibration.score_binding_mismatch")
        history = select_calibration_scores(
            tuple(
                v
                for v in scores.values()
                if v.formation_session > seed.formation_sessions[-1] or v == score
            ),
            package_hash=score.strategy_package_hash,
            component_recipe_hash=score.component_recipe_hash,
            issued={}
            if self.issued_score_hashes is None
            else self.issued_score_hashes(package_id, score.component_recipe_hash),
            current=score,
        )
        if (
            binding.source_kind == "RECORDED_INPUT_SNAPSHOT"
            and score.formation_session not in seed.formation_sessions
        ):
            raise ValueError("portfolio_calibration.recorded_outcome_support_absent")
        source_hash = self._source(binding)
        if binding.source_kind == "WORKSPACE_DATA_FEATURE":
            unscored = self._unscored(seed.formation_sessions, score.formation_session, history)
            if unscored:
                raise ValueError(
                    "portfolio_calibration.intervening_score_absent:"
                    f"{len(unscored)},{unscored[0].isoformat()}..{unscored[-1].isoformat()}"
                )
            observation = self.scoring.store.load_frozen_observation_snapshot(
                score.observation_snapshot_hash
            )
            if observation.source_binding_hash != source_hash:
                raise ValueError("portfolio_calibration.current_score_source_changed")
        # The answer names this plan, never the owner's last one, which a concurrent plan
        # may have replaced in between (V534).
        planned = CalibrationPlan.create(
            workspace_manifest_hash=manifest_fields_hash(self.manifest, PLAN_FIELDS),
            binding=binding,
            score_snapshot_hash=score_hash,
            history_score_hashes=tuple(v.snapshot_hash for v in history),
            source_hash=source_hash,
            implementation_hash=_implementation_hash(),
        )
        self.last_plan = planned
        return {
            "status": "PLANNED",
            "calibration_plan_hash": self._kept(planned).plan_hash,
            "strategy_package_id": package_id,
            "formation_session": score.formation_session.isoformat(),
            "activation_session": seed.rule.activation_session.isoformat(),
            "lookback": seed.rule.lookback,
            "cached": self._published(planned) is not None,
            "work": "Prepare causal Portfolio input. Zero predictions or fits.",
            "limitation": "PORTFOLIO_INPUT_QA_NOT_RECOMMENDATION",
            "next_requests": {
                "run": {
                    "operation": "STRATEGY_CALIBRATION_RUN",
                    "calibration_plan_hash": planned.plan_hash,
                }
            },
        }

    def _unscored(
        self,
        seed_sessions: tuple[date, ...],
        formation: date,
        history: tuple[FrozenComponentScoreSnapshot, ...],
    ) -> tuple[date, ...]:
        """The sessions after the seed through `formation` that hold no published score.

        The run's observations need a score for each (`_observations`, on the same XNYS and
        XNAS calendar), so the plan refuses, naming them, instead of admitting a run that
        stops in `seal_observations` and waits for recovery (V334).

        Args:
            seed_sessions: The calibration seed's formation sessions.
            formation: The planned score's formation session.
            history: The published scores the plan selected.

        Returns:
            The sessions without a score, in order; empty when none is missing.
        """
        if formation <= seed_sessions[-1]:
            return ()
        calendar = materialize_calendar_schedule(
            ("XNYS", "XNAS"), start=seed_sessions[-1], end=formation, as_of_timestamp=self.clock()
        )
        venues: dict[date, set[str]] = {}
        for row in calendar.to_pylist():
            venues.setdefault(row["session_date"], set()).add(row["calendar_id"])
        held = {d for d in seed_sessions if d <= formation} | {v.formation_session for v in history}
        return tuple(
            day
            for day in sorted(venues)
            if venues[day] == {"XNAS", "XNYS"}
            and seed_sessions[-1] < day <= formation
            and day not in held
        )

    def prepare(self, plan_hash: str) -> CalibrationPlan:
        """Reopen the exact calibration preview or retained task plan.

        Args:
            plan_hash: Exact plan identity.

        Returns:
            Retained calibration plan.

        Raises:
            ValueError: No exact preview or admitted plan exists.
        """
        if (kept := self._plans.runnable(plan_hash)) is not None:
            return kept
        for task in self.session.task_control_registry.tasks():
            if (
                task.task_kind == TASK_KIND
                and (value := self._plan_of(task, current=False)).plan_hash == plan_hash
            ):
                return value
        raise ValueError("portfolio_calibration.plan_required")

    def _kept(self, plan: CalibrationPlan) -> CalibrationPlan:
        self._plans.remember(plan)
        return plan

    def _require(self, plan: CalibrationPlan) -> None:
        if (
            plan.workspace_manifest_hash != manifest_fields_hash(self.manifest, PLAN_FIELDS)
            or plan.binding != self._binding(plan.binding.strategy_package_id)
            or not is_current(IMPLEMENTATION_ROLE, plan.implementation_hash, _implementation_hash())
        ):
            raise ValueError("portfolio_calibration.task_binding_mismatch")
        seed, _, _, _ = load_observations(self.store, plan.binding.seed_hash)
        if seed.strategy_package_hash != plan.binding.strategy_package_hash:
            raise ValueError("portfolio_calibration.seed_package_mismatch")
        published = {v.snapshot_hash for v in self._published_scores()}
        if (
            not set(plan.history_score_hashes) <= published
            or plan.score_snapshot_hash not in published
        ):
            raise ValueError("portfolio_calibration.published_score_required")

    def _plan_of(self, task: TaskRecord, *, current: bool = True) -> CalibrationPlan:
        if task.task_kind != TASK_KIND:
            raise ValueError("portfolio_calibration.task_kind_invalid")
        plan = CalibrationPlan.model_validate(task.input.payload["plan"])
        if (task.input, task.goal, task.plan) != _task_contract(plan):
            raise ValueError("portfolio_calibration.task_contract_invalid")
        if current:
            self._require(plan)
        return cast(CalibrationPlan, plan)

    def replan_request(self, task: TaskRecord) -> dict[str, object]:
        """Replan the validated Task's exact package and score publication.

        Args:
            task: Durable Task whose declaration supplies the next request.

        Returns:
            The filled calibration planning request.

        Raises:
            ValueError: The Task's kind, sealed plan or workflow does not validate.
        """
        plan = self._plan_of(task, current=False)
        return {
            "operation": "STRATEGY_CALIBRATION_PLAN",
            "strategy_package_id": plan.binding.strategy_package_id,
            "score_snapshot_hash": plan.score_snapshot_hash,
        }

    def _published(self, plan: CalibrationPlan) -> PreparedPortfolioComponentInput | None:
        found = [
            self._load(path.stem)
            for path in sorted((self.store.root / RESULT_CATEGORY).glob("*.json"))
        ]
        found = [value for value in found if value.request_hash == plan.plan_hash]
        if len(found) > 1:
            raise ValueError("portfolio_calibration.publication_ambiguous")
        return found[0] if found else None

    def _load(self, content_hash: str) -> PreparedPortfolioComponentInput:
        self._require_local_store()
        return self.store.load(
            category=RESULT_CATEGORY,
            content_hash=content_hash,
            model=PreparedPortfolioComponentInput,
            identity_field="content_hash",
        )

    def published_input(
        self, content_hash: str, *, expected_source_hash: str | None = None
    ) -> PreparedPortfolioComponentInput | PreparedPortfolioBookInput:
        """Resolve an admitted terminal publication for the Portfolio consumer."""
        for task in self.session.task_control_registry.tasks():
            if (
                task.task_kind == TASK_KIND
                and task.lifecycle is TaskLifecycle.SUCCEEDED
                and self._stage_hash(task, STAGES[-1]) == content_hash
            ):
                plan = self._plan_of(task, current=False)
                if expected_source_hash is not None and plan.source_hash != expected_source_hash:
                    raise ValueError("portfolio_calibration.published_input_source_mismatch")
                return self._bound_input(task, plan, content_hash)
        additional = None if self.additional_input is None else self.additional_input(content_hash)
        if additional is not None:
            source_hash, value = additional
            if value.content_hash != content_hash or (
                expected_source_hash is not None and source_hash != expected_source_hash
            ):
                raise ValueError("portfolio_calibration.published_input_source_mismatch")
            return value
        raise ValueError("portfolio_calibration.published_input_required")

    def _bound_input(
        self, task: TaskRecord, plan: CalibrationPlan, content_hash: str
    ) -> PreparedPortfolioComponentInput:
        return self.read_input(plan, self._stage_hash(task, STAGES[1]), content_hash)

    def read_input(
        self, plan: CalibrationPlan, observation_hash: str, content_hash: str
    ) -> PreparedPortfolioComponentInput:
        """Verify the product against its exact domain request and observations."""
        value = self._load(content_hash)
        if (
            value.request_hash != plan.plan_hash
            or value.score_snapshot_hash != plan.score_snapshot_hash
            or value.observation_hash != observation_hash
            or value.strategy_package_hash != plan.binding.strategy_package_hash
        ):
            raise ValueError("portfolio_calibration.result_evidence_invalid")
        return value

    def reusable(self, plan: CalibrationPlan) -> PreparedPortfolioComponentInput | None:
        """Require current source and succeeded exact-task evidence for calibration reuse.

        Args:
            plan: Explicit sealed calibration plan.

        Returns:
            Verified prepared component input, or None without a succeeded matching task.

        Raises:
            ValueError: Plan/source or result evidence is stale or invalid.
        """
        self._require(plan)
        if self._source(plan.binding) != plan.source_hash:
            raise ValueError("portfolio_calibration.stale_plan")
        for task in self.session.task_control_registry.tasks():
            if (
                task.task_kind == TASK_KIND
                and task.lifecycle is TaskLifecycle.SUCCEEDED
                and self._plan_of(task, current=False) == plan
            ):
                return self._bound_input(task, plan, self._stage_hash(task, STAGES[-1]))
        return None

    def admit(self, plan: CalibrationPlan) -> CommandAdmission:
        """Reuse or admit the exact current causal calibration plan.

        Reuse an exact uncancelled admission or admit the current causal portfolio calibration plan.

        Args:
            plan: Explicit sealed plan.

        Returns:
            Exact retained or newly admitted task identity/lifecycle.

        Raises:
            ValueError: Plan authority or current source is invalid or stale.
        """
        self._require(plan)
        envelope, goal, workflow = _task_contract(plan)
        registry = self.session.task_control_registry
        for task in registry.tasks():
            if task.input == envelope and task.lifecycle is not TaskLifecycle.CANCELLED:
                return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)
        if self._source(plan.binding) != plan.source_hash:
            raise ValueError("portfolio_calibration.stale_plan")
        task = registry.admit(
            input_envelope=envelope, goal=goal, plan=workflow, observed_at=self.clock()
        ).record
        return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute the exact admitted causal portfolio calibration task through its writer session.

        Args:
            task_id: Exact admitted task.
            expected_task_hash: Optional optimistic task identity.
        """
        task = self.session.task_control_registry.task(task_id)
        self._plan_of(task)
        self.session.execute_admitted(task, self, self.clock, expected_task_hash)

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Bind the exact task schema, workflow, exact calibration seed and execution identity.

        Args:
            task: Exact admitted task whose retained plan is verified.

        Returns:
            Deterministic execution compatibility declaration.
        """
        plan = self._plan_of(task)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash(schema_structure(CalibrationPlan)),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=plan.binding.seed_hash,
            framework_identity_hash=self.session.execution_identity(plan.implementation_hash),
        )

    def _stage_hash(self, task: TaskRecord, stage: str) -> str:
        item = next(
            v
            for v in self.session.task_control_registry.work_items(task.task_id)
            if v.stage_id == stage
        )
        if len(item.evidence) != 1:
            raise ValueError("portfolio_calibration.stage_evidence_absent")
        return str(item.evidence[0].content_hash)

    @staticmethod
    def _evidence(stage: str, content_hash: str) -> tuple[TaskEvidence, ...]:
        return (
            TaskEvidence(
                evidence_kind="portfolio_calibration.stage",
                reference=f"playpen://portfolio-calibration/{stage}/{content_hash}",
                content_hash=content_hash,
            ),
        )

    def _observations(
        self,
        plan: CalibrationPlan,
        captured: LocalQAMarketSnapshot | PreparedLocalQASnapshotRows | None = None,
    ) -> CalibrationObservations:
        if captured is None and self._source(plan.binding) != plan.source_hash:
            raise ValueError("portfolio_calibration.stale_plan")
        seed, values, returns, eligible = load_observations(self.store, plan.binding.seed_hash)
        current = self.scoring.store.load_frozen_component_score(plan.score_snapshot_hash)
        history = tuple(
            self.scoring.store.load_frozen_component_score(h) for h in plan.history_score_hashes
        )
        sessions = tuple(
            sorted(
                set(d for d in seed.formation_sessions if d <= current.formation_session)
                | {s.formation_session for s in history}
            )
        )
        positions = {d: i for i, d in enumerate(sessions)}
        listing_axis = tuple(
            sorted(
                set(seed.ordered_listing_ids).union(
                    *(set(v.ordered_listing_ids) for v in (*history, current))
                )
            )
        )
        columns = {v: i for i, v in enumerate(listing_axis)}
        seed_columns = [columns[v] for v in seed.ordered_listing_ids]
        membership: dict[date, set[str]] = {
            d: set(seed.ordered_listing_ids) for d in seed.formation_sessions
        }
        shape = (len(sessions), len(listing_axis))
        scores = np.full(shape, np.nan)
        realized = np.full(shape, np.nan)
        live: BoolArray = np.zeros(shape, dtype=np.bool_)
        ends = dict(zip(seed.formation_sessions, seed.holding_end_sessions, strict=True))
        for i, day in enumerate(seed.formation_sessions):
            if day in positions:
                row = positions[day]
                scores[row, seed_columns] = values[i]
                realized[row, seed_columns] = returns[i]
                live[row, seed_columns] = eligible[i]
        for value in history:
            if (
                value.strategy_package_hash != seed.strategy_package_hash
                and value.strategy_package_hash != seed.strategy_package_hash
            ) or not recipe_hashes_match(value.component_recipe_hash, seed.component_recipe_hash):
                raise ValueError("portfolio_calibration.history_score_binding_mismatch")
            row = positions[value.formation_session]
            selected_columns = [columns[v] for v in value.ordered_listing_ids]
            membership[value.formation_session] = set(value.ordered_listing_ids)
            scores[row] = np.nan
            scores[row, selected_columns] = [np.nan if v is None else v for v in value.scores]
            live[row] = False
            live[row, selected_columns] = value.live
        source_hashes: tuple[str, ...] = (seed.content_hash, plan.source_hash)
        if plan.binding.source_kind == "WORKSPACE_DATA_FEATURE":
            table, bars, axis = (
                local_qa_snapshot_rows(
                    captured, sessions=sessions, through=current.formation_session
                )
                if captured is not None
                else read_local_qa_execution_rows(
                    store=MarketDataRepository(self.session.workspace),
                    sessions=sessions,
                    listing_ids=listing_axis,
                    through=current.formation_session,
                    observed_at=self.clock(),
                    retained_listing_ids=listing_axis,
                )
            )
            if any(
                day > seed.formation_sessions[-1] and day not in positions
                for day in axis
                if day <= current.formation_session
            ):
                raise ValueError("portfolio_calibration.intervening_score_absent")
            realized[:] = np.nan
            ends = {}
            for item in table.select(
                ["listing_id", "formation_session", "simple_return", "holding_end_session"]
            ).to_pylist():
                if item["listing_id"] not in columns:
                    continue
                row, column = positions[item["formation_session"]], columns[item["listing_id"]]
                realized[row, column] = (
                    np.nan if item["simple_return"] is None else item["simple_return"]
                )
                ends[item["formation_session"]] = item["holding_end_session"]
            qualified = decision_eligible_at_closes(
                formation_sessions=sessions,
                listing_ids=listing_axis,
                sessions=axis,
                bars=bars,
            )
            for day, row in positions.items():
                selected = membership.get(day, ())
                member: BoolArray = np.fromiter(
                    (listing in selected for listing in listing_axis),
                    dtype=np.bool_,
                    count=len(listing_axis),
                )
                live[row] &= member & qualified[row]
            source_hashes += (canonical_hash(table["row_hash"].to_pylist()),)
        if captured is None and self._source(plan.binding) != plan.source_hash:
            raise ValueError("portfolio_calibration.source_changed_during_read")
        return publish_consumed_observations(
            self.store,
            decision_session=current.formation_session,
            scores=scores,
            returns=realized,
            eligible=live,
            origin="LOCAL_QA_OBSERVATIONS"
            if plan.binding.source_kind == "WORKSPACE_DATA_FEATURE"
            else seed.origin,
            strategy_package_hash=seed.strategy_package_hash,
            component_recipe_hash=seed.component_recipe_hash,
            rule=seed.rule,
            formation_sessions=sessions,
            holding_end_sessions=tuple(ends.get(d) for d in sessions),
            ordered_listing_ids=listing_axis,
            source_hashes=source_hashes,
            score_snapshot_hashes=plan.history_score_hashes,
        )

    def execute_stage(
        self, *, task: TaskRecord, execution: TaskExecution, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        """Execute an admitted calibration stage under verified lifecycle admission.

        Execute one admitted causal portfolio calibration stage under verified lifecycle admission.

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

    def execute_step(
        self,
        plan: CalibrationPlan,
        stage: str,
        prior: Callable[[str], str],
        *,
        captured: LocalQAMarketSnapshot | PreparedLocalQASnapshotRows | None = None,
    ) -> StageExecutionResult:
        """Validate, collect observations, compile or reopen one exact calibration step.

        Args:
            plan: Exact admitted calibration plan.
            stage: Installed stage identifier.
            prior: Reader of this task's previous stage content identities.
            captured: Optional explicit captured market snapshot.

        Returns:
            Ready stage result with exact plan, observations or prepared-input evidence.

        Raises:
            ValueError: Current source is stale, stage is unknown or retained calibration data is
                invalid.
        """
        if stage == STAGES[0]:
            if self._source(plan.binding) != plan.source_hash:
                raise ValueError("portfolio_calibration.stale_plan")
            content = plan.plan_hash
        elif stage == STAGES[1]:
            content = self._observations(plan, captured).content_hash
        elif stage == STAGES[2]:
            value = self._published(plan)
            if value is None:
                observed, scores, returns, eligible = load_observations(
                    self.store, prior(STAGES[1])
                )
                score = self.scoring.store.load_frozen_component_score(plan.score_snapshot_hash)
                value = compile_portfolio_calibration(
                    request_hash=plan.plan_hash,
                    score=score,
                    observations=observed,
                    scores=scores,
                    returns=returns,
                    eligible=eligible,
                    current_eligible=eligible[
                        observed.formation_sessions.index(score.formation_session),
                        [observed.ordered_listing_ids.index(v) for v in score.ordered_listing_ids],
                    ],
                )
                self.store.publish(
                    category=RESULT_CATEGORY, value=value, identity_field="content_hash"
                )
            content = value.content_hash
        elif stage == STAGES[3]:
            content = prior(STAGES[2])
            self._load(content)
        else:
            raise ValueError("portfolio_calibration.stage_unknown")
        return StageExecutionResult(StageDisposition.READY, evidence=self._evidence(stage, content))

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Require exact plan, observation or task-bound prepared-input evidence.

        Args:
            task: Exact admitted calibration task.
            execution: Current execution declaration.
            work_item: Exact workflow stage.
            evidence: Declared stage evidence tuple.

        Returns:
            Unchanged evidence after stage-specific source, score and input checks.

        Raises:
            ValueError: Stage, evidence tuple or exact plan/observation/result binding differs.
        """
        plan, stage = self._plan_of(task), work_item.stage_id
        if len(evidence) != 1 or evidence != self._evidence(stage, evidence[0].content_hash):
            raise ValueError("portfolio_calibration.evidence_invalid")
        content = evidence[0].content_hash
        if stage == STAGES[0]:
            if content != plan.plan_hash:
                raise ValueError("portfolio_calibration.plan_evidence_invalid")
        elif stage == STAGES[1]:
            value, _, _, _ = load_observations(self.store, content)
            if (
                value.source_hashes[:2] != (plan.binding.seed_hash, plan.source_hash)
                or value.score_snapshot_hashes != plan.history_score_hashes
            ):
                raise ValueError("portfolio_calibration.observation_evidence_invalid")
        elif stage in STAGES[2:]:
            with verified_lifecycle_admissions():
                self._bound_input(task, plan, content)
        else:
            raise ValueError("portfolio_calibration.stage_unknown")
        return evidence

    def readback(self, task_id: UUID | None = None) -> dict[str, object]:
        """Read retained calibration state and its succeeded task-bound input publication.

        Args:
            task_id: Optional exact task; defaults to the latest calibration task.

        Returns:
            Task state or verified prepared input/readout, explicitly QA-only without forward
            availability authority.
        """
        registry = self.session.task_control_registry
        task = (
            registry.task(task_id)
            if task_id
            else next((t for t in reversed(registry.tasks()) if t.task_kind == TASK_KIND), None)
        )
        if task is None:
            return {"status": "NO_CALIBRATION_PUBLICATION", "task_id": None}
        plan = self._plan_of(task, current=False)
        result = (
            self._bound_input(task, plan, self._stage_hash(task, STAGES[-1]))
            if task.lifecycle is TaskLifecycle.SUCCEEDED
            else None
        )
        return {
            "status": "PORTFOLIO_INPUT_PUBLISHED" if result else task.lifecycle.value,
            "task_id": str(task.task_id),
            "task_admitted_at": task.admitted_at.isoformat(),
            "availability_disposition": "QA_ONLY_NOT_FORWARD_AVAILABILITY_AUTHORITY",
            "strategy_package_id": plan.binding.strategy_package_id,
            "input": result.model_dump(mode="json") if result else None,
            "readout": result.readout() if result else None,
        }


@dataclass
class StrategyCalibrationCommand:
    """Dispatch one explicitly prepared causal portfolio calibration plan."""

    application: StrategyCalibrationApplication
    plan: CalibrationPlan | None = None
    command_kind: str = TASK_KIND

    def admit(self) -> CommandAdmission:
        """Require an explicit causal portfolio calibration plan before deterministic admission.

        Returns:
            Task admission from the owning application.

        Raises:
            ValueError: No exact plan is supplied.
        """
        if self.plan is None:
            raise ValueError("portfolio_calibration.plan_required")
        return self.application.admit(self.plan)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute the exact task admitted for this causal portfolio calibration command.

        Args:
            task_id: Exact admitted task.
            expected_task_hash: Optional optimistic task identity.
        """
        self.application.execute(task_id, expected_task_hash=expected_task_hash)
