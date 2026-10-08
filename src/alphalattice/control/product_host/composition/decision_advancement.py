"""Conditional-decision QA composition of the existing advancement Task.

The common-watermark contract remains distinct: a decision can be ready while
its future outcome is absent. This composition calls the same standalone domain
steps, never admits child Tasks, and keeps timestamped intermediate results in
the existing advancement store until the Portfolio branch is ready to expose.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Self, cast
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_inputs.tradability.surface import (
    decision_eligible_at_close,
)
from alphalattice.control.data_platform.maintenance.contracts import WorkspaceDataUpdatePlan
from alphalattice.control.data_platform.readiness import _latest_common_us_session
from alphalattice.control.product_host.composition.portfolio_updates import (
    PLAN_FIELDS as DECISION_PLAN_FIELDS,
)
from alphalattice.control.product_host.composition.portfolio_updates import (
    PortfolioUpdateApplication,
    PortfolioUpdatePlan,
)
from alphalattice.control.product_host.composition.portfolio_updates import (
    _implementation_hash as decision_identity,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceScoreInput,
    manifest_fields_hash,
)
from alphalattice.control.product_host.composition.strategy_calibration import (
    PLAN_FIELDS as CALIBRATION_PLAN_FIELDS,
)
from alphalattice.control.product_host.composition.strategy_calibration import (
    STAGES as CALIBRATION_STAGES,
)
from alphalattice.control.product_host.composition.strategy_calibration import (
    CalibrationPlan,
)
from alphalattice.control.product_host.composition.strategy_calibration import (
    _implementation_hash as calibration_identity,
)
from alphalattice.control.product_host.composition.strategy_score_inputs import (
    PreparedWorkspaceComponentInputs,
    score_source_identity,
    workspace_score_source_identity,
)
from alphalattice.control.product_host.composition.strategy_scoring import (
    PLAN_FIELDS as SCORE_PLAN_FIELDS,
)
from alphalattice.control.product_host.composition.strategy_scoring import (
    STAGES as SCORE_STAGES,
)
from alphalattice.control.product_host.composition.strategy_scoring import (
    StrategyScorePlan,
)
from alphalattice.control.product_host.composition.strategy_scoring import (
    _implementation_hash as scoring_identity,
)
from alphalattice.control.product_host.maintenance.data_update import (
    _STAGES as DATA_STAGES,
)
from alphalattice.control.product_host.maintenance.data_update import (
    WorkspaceDataUpdateApplication,
    read_workspace_inputs,
)
from alphalattice.control.product_host.storage.inventory import require_storage_capacity
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
    failure_code_from,
)
from alphalattice.control.task_control.runner import (
    StageDisposition,
    StageExecutionResult,
)
from alphalattice.control.workspace_runtime.content_store import (
    CommittedIndex,
    CommittedKind,
    verified_model_read_scope,
)
from alphalattice.foundation.causal_outcomes.execution.contracts import LocalQAMarketSnapshot
from alphalattice.foundation.causal_outcomes.execution.readers import (
    PreparedLocalQASnapshotRows,
    local_qa_prefix,
    planned_local_qa_schedule,
    read_local_qa_market_snapshot,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.interface.local_application.dispatcher import CommandAdmission
from alphalattice.investment.alpha_research.publication.contracts import (
    FrozenComponentScoreSnapshot,
    FrozenFeaturePreparation,
)
from alphalattice.investment.alpha_research.scores.model_renewal import (
    AdmittedRenewingInference,
    AlphaModelSetPublication,
    verified_lifecycle_admissions,
)
from alphalattice.investment.portfolio_strategy_lab.application.advancement_task import (
    ADVANCEMENT_TASK_KIND,
)
from alphalattice.investment.portfolio_strategy_lab.application.calibration import (
    BoolArray,
    PreparedPortfolioBookInput,
    PreparedPortfolioComponentInput,
    compile_ew_component_input,
    load_observations,
    select_calibration_scores,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioDecisionCheckpoint,
    PortfolioUpdatePublication,
    advance_decision_state,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    recipe_hashes_match,
)
from alphalattice.investment.portfolio_strategy_lab.publication.advancement_ledger import (
    AdvancementLedgerStore,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.static import render_decision_update
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.spans import span

SCHEMA = "portfolio-conditional-decision-advancement"
STAGES = (
    "verify_request",
    "update_inputs",
    "seal_inputs",
    "prepare_scores",
    "prepare_calibration",
    "advance_book",
    "publish_readback",
)


_IMPLEMENTATION_ROLE = "product_host.conditional_decision_advancement"
"""The role this advancement's implementation is recorded under in the identity successors."""


class _Sealed(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)
    content_hash: str

    @classmethod
    def create(cls, **values: object) -> Self:
        payload = cls.model_construct(**values).model_dump(mode="json", exclude={"content_hash"})
        return cls(**payload, content_hash=canonical_hash(payload))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def identity(self) -> Self:
        if self.content_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"content_hash"})
        ):
            raise ValueError("research_update.identity_invalid")
        return self


class DecisionAdvancementPlan(_Sealed):
    """Bind exact QA checkpoint, component score requests, data plan and numerical owners."""

    purpose: Literal["CONDITIONAL_DECISION_QA"] = "CONDITIONAL_DECISION_QA"
    workspace_manifest_hash: str
    catalog_hash: str
    package_id: str
    score_binding: ResearchWorkspaceScoreInput | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    score_bindings: tuple[ResearchWorkspaceScoreInput, ...] = Field(
        default=(), exclude_if=lambda v: not v
    )
    checkpoint_hash: str
    parent_hash: str | None
    target: date
    decision_sessions: tuple[date, ...]
    score_sessions: tuple[date, ...]
    score_requests: tuple[tuple[int, date], ...] = Field(default=(), exclude_if=lambda v: not v)
    history_score_hashes: tuple[str, ...]
    data_plan: WorkspaceDataUpdatePlan
    implementation_hash: str
    score_implementation_hash: str
    calibration_implementation_hash: str
    decision_implementation_hash: str

    @property
    def bindings(self) -> tuple[ResearchWorkspaceScoreInput, ...]:
        """Read explicit component bindings or the supported single-binding representation.

        Returns:
            Ordered component bindings, or the retained singleton.
        """
        return self.score_bindings or (() if self.score_binding is None else (self.score_binding,))

    @property
    def requests(self) -> tuple[tuple[int, date], ...]:
        """Read explicit component/session requests or project singleton score sessions.

        Returns:
            Ordered component-index/session pairs; single-binding compatibility uses index zero.
        """
        return (
            self.score_requests
            if self.score_bindings
            else tuple((0, day) for day in self.score_sessions)
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def binding_axis(self) -> Self:
        """Require one coherent component binding representation and unique admitted requests.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Single/multiple bindings are ambiguous, package/component axes differ or
                request indices/sessions are invalid.
        """
        if (
            (self.score_binding is None) == (not self.score_bindings)
            or any(b.strategy_package_id != self.package_id for b in self.bindings)
            or len({b.component_id for b in self.bindings}) != len(self.bindings)
        ):
            raise ValueError("research_update.component_binding_axis_invalid")
        if len(set(self.requests)) != len(self.requests) or any(
            not 0 <= i < len(self.bindings) or d not in self.score_sessions
            for i, d in self.requests
        ):
            raise ValueError("research_update.score_request_axis_invalid")
        return self


class DecisionAdvancementStep(_Sealed):
    """Retain exact advancement plan/stage products and source identity for durable readback."""

    plan_hash: str
    step: str
    products: tuple[str, ...]
    source_hash: str


def task_contract(
    plan: DecisionAdvancementPlan,
) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    """Construct the exact task envelope, goal and ordered advancement workflow.

    Args:
        plan: Sealed admitted decision advancement plan.

    Returns:
        Task input, research goal and workflow bound to the declared stages/schema.
    """
    envelope = TaskInputEnvelope.create(
        task_kind=ADVANCEMENT_TASK_KIND,
        input_schema_id=SCHEMA,
        payload={"plan": plan.model_dump(mode="json")},
    )
    goal = ResearchGoal.create(
        goal_kind="ADVANCE_CONDITIONAL_DECISION_QA",
        input_hash=envelope.input_hash,
        deliverable_kind="PortfolioUpdatePublication",
        summary="Update captured inputs, catch up in order, publish one complete QA branch.",
    )
    workflow = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash(STAGES),
        verifier_catalog_hash=canonical_hash([SCHEMA, STAGES]),
        work_items=tuple(
            WorkItemDefinition.create(
                stage_id=s, dependency_ids=STAGES[:i], verifier_id=SCHEMA + "." + s
            )
            for i, s in enumerate(STAGES)
        ),
    )
    return envelope, goal, workflow


class _Cancelled(Exception):
    pass


class _Deferred(Exception):
    """The provider asked the data stage to wait: the update waits with it (V601)."""

    def __init__(self, result: StageExecutionResult) -> None:
        super().__init__(result.failure_code)
        self.result = result


_DECISION_STAGES = CommittedKind(
    "decision-stages", "decision-stages", DecisionAdvancementStep, "content_hash"
)
_DECISION_CANDIDATES = CommittedKind(
    "decision-candidates", "decision-candidates", PortfolioUpdatePublication, "content_hash"
)


class DecisionAdvancementApplication:
    """One advancement schema, composed into the existing dispatcher and ledger."""

    task_kind = ADVANCEMENT_TASK_KIND
    replans = (
        TaskReplan(
            task_kind=ADVANCEMENT_TASK_KIND,
            preview="RESEARCH_UPDATE_PLAN",
            admitting="RESEARCH_UPDATE_RUN",
        ),
    )
    """The re-plan of the Task kind this owner admits, which the recovery view offers
    (V188)."""

    def __init__(self, updates: PortfolioUpdateApplication, data: WorkspaceDataUpdateApplication):
        """Compose decision, data, scoring and calibration owners for durable QA updates.

        The application registers its completed score/input readbacks with the retained
        scoring/calibration owners.

        Args:
            updates: Portfolio update owner retaining checkpoint, session and clock.
            data: Workspace data transition owner.
        """
        self.updates, self.data = updates, data
        self.calibration, self.scoring = updates.calibration, updates.calibration.scoring
        self.session, self.clock = updates.session, updates.clock
        self.ledger = AdvancementLedgerStore(self.session.workspace / "runtime" / "artifacts")
        self.index = CommittedIndex(self.ledger.root, self.ledger.content)
        self.scoring.additional_publications = self._completed_scores
        self.calibration.additional_input = self._completed_input

    def implementation(self) -> str:
        """Bind advancement rule syntax and its scoring, calibration and decision owner identities.

        Returns:
            Exact combined implementation identity; operational task runner state is outside
            numerical meaning.
        """
        from pathlib import Path

        from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash

        # This file decides which publications feed scoring and calibration; Task
        # Control's runner and registry, tracked here until the binding plan's B5,
        # decide lifecycles and no number.
        own = source_rule_closure_hash(
            root=resolve_playpen_root(Path(__file__)),
            semantic_owner="product_host",
            numerical_role="CONDITIONAL_DECISION_ADVANCEMENT",
            tracked_paths=(
                "src/alphalattice/control/product_host/composition/decision_advancement.py",
            ),
        )
        return str(
            canonical_hash([own, scoring_identity(), calibration_identity(), decision_identity()])
        )

    def _load[M: BaseModel](
        self, category: str, identity: str, model: type[M], field: str = "content_hash"
    ) -> M:
        return self.ledger.content.load_model(
            category=category, content_hash=identity, model=model, identity_field=field
        )

    def _save(self, category: str, value: BaseModel, field: str = "content_hash") -> str:
        return self.ledger.content.publish_model(
            category=category, value=value, identity_field=field
        )

    def _step(self, plan: DecisionAdvancementPlan, step: str) -> DecisionAdvancementStep | None:
        value = self.index.open(_DECISION_STAGES, canonical_hash([plan.content_hash, step]))
        if value is not None and (value.plan_hash != plan.content_hash or value.step != step):
            raise ValueError("research_update.stage_binding_invalid")
        return value

    def _need(self, plan: DecisionAdvancementPlan, step: str) -> DecisionAdvancementStep:
        value = self._step(plan, step)
        if value is None:
            raise ValueError("research_update.stage_missing:" + step)
        return value

    def _commit(
        self, plan: DecisionAdvancementPlan, step: str, products: tuple[str, ...], source: str
    ) -> DecisionAdvancementStep:
        value = DecisionAdvancementStep.create(
            plan_hash=plan.content_hash, step=step, products=products, source_hash=source
        )
        self.index.commit(_DECISION_STAGES, canonical_hash([plan.content_hash, step]), value)
        return value

    def prepare(self, identity: str) -> DecisionAdvancementPlan:
        """Reopen one exact sealed decision advancement program.

        Args:
            identity: Content identity in decision-programs.

        Returns:
            Validated DecisionAdvancementPlan from durable storage.
        """
        return self._load("decision-programs", identity, DecisionAdvancementPlan)

    def _require(self, plan: DecisionAdvancementPlan) -> None:
        captured = self._step(plan, STAGES[2]) is not None
        checkpoint = (
            self.updates.store.load_decision_checkpoint(plan.checkpoint_hash)
            if captured
            else self.updates._checkpoint(plan.package_id)
        )
        root = self.updates._checkpoint(plan.package_id, current=False)
        if (
            plan.workspace_manifest_hash != self.updates.manifest.manifest_hash
            or plan.catalog_hash != self.updates.application.resolver.strategy_catalog_hash
            or plan.checkpoint_hash != checkpoint.content_hash
            or checkpoint.history_hash != root.content_hash
            or checkpoint.package != root.package
            or checkpoint.model_authority_hashes != root.model_authority_hashes
            or any(
                b != self.scoring._binding(plan.package_id, b.component_id) for b in plan.bindings
            )
            or not is_current(_IMPLEMENTATION_ROLE, plan.implementation_hash, self.implementation())
        ):
            raise ValueError("research_update.binding_changed")

    def _parent(self, plan: DecisionAdvancementPlan) -> PortfolioUpdatePublication | None:
        if plan.parent_hash is None:
            return None
        return self.updates.store.content.load_model(
            category="decision-updates",
            content_hash=plan.parent_hash,
            model=PortfolioUpdatePublication,
            identity_field="content_hash",
        )

    def _fresh(self, plan: DecisionAdvancementPlan) -> None:
        self._require(plan)
        history = self.updates.history(plan.checkpoint_hash)
        if (history[-1].content_hash if history else None) != plan.parent_hash:
            raise ValueError("research_update.parent_advanced")

    def plan(self, package_id: str, target: date | None = None) -> dict[str, object]:
        """Prepare or reuse bounded QA advancement through the latest completed market session.

        Args:
            package_id: Installed strategy package with retained checkpoint.
            target: Optional completed target session; defaults to the latest common completed
                session.

        Returns:
            Plan/task reuse view, or OBSERVATIONS_PENDING when no executable change is due.

        Raises:
            ValueError: Target/session/epoch admission fails, required score authority is
                unavailable or workspace transition is not prepared.
        """
        checkpoint = self.updates._checkpoint(package_id)
        now = self.clock()
        latest = _latest_common_us_session(on_or_before=now.date(), observed_at=now)
        target = target or latest
        if target > latest:
            raise ValueError("research_update.target_not_completed")
        if target not in {p.formation_session for p in planned_local_qa_schedule(target, target)}:
            raise ValueError("research_update.target_not_session")
        # Pending requests and exact results keep their original identity despite
        # changed preparation timestamps or the branch they themselves published.
        tasks = [
            task
            for task in reversed(self.session.task_control_registry.tasks())
            if task.input.input_schema_id == SCHEMA
            and task.lifecycle is not TaskLifecycle.CANCELLED
        ]
        # An update of this strategy that has not ended answers its own plan, whatever target was
        # asked: its run follows it, or resumes a deferred one once due (V601), and a later
        # session is planned once it ends. Planned anew, the later update queued behind the
        # deferral, or was refused while the deferral held the workspace's inputs (V604).
        for task in tasks:
            if task.lifecycle in (
                TaskLifecycle.QUEUED,
                TaskLifecycle.RUNNING,
                TaskLifecycle.RECOVERY_REQUIRED,
                TaskLifecycle.DEFERRED,
            ):
                prior = self._plan_of(task, current=False)
                if prior.package_id == package_id:
                    self._require(prior)
                    return self._plan_body(prior, False)
        for task in tasks:
            prior = self._plan_of(task, current=False)
            if prior.package_id == package_id and prior.target == target:
                if (
                    not is_current(
                        _IMPLEMENTATION_ROLE, prior.implementation_hash, self.implementation()
                    )
                    or prior.checkpoint_hash != checkpoint.content_hash
                    or prior.workspace_manifest_hash != self.updates.manifest.manifest_hash
                ):
                    continue
                sealed = self._step(prior, STAGES[2])
                if sealed is not None and sealed.source_hash != workspace_score_source_identity(
                    self.session.workspace
                ):
                    continue
                if self.reusable(prior):
                    return self._plan_body(prior, True)
        history = self.updates.history(checkpoint.content_hash)
        previous = history[-1] if history else None
        book = checkpoint.initial_book if previous is None else previous.book
        pending = None if previous is None else previous.pending_proposal
        first = book.schedule.entry_session if pending is None else pending.schedule.entry_session
        if previous is not None and target < previous.observed_through:
            raise ValueError("research_update.target_before_publication")
        settlement_end = max(
            [checkpoint.epoch_end]
            + ([] if pending is None else [pending.schedule.holding_end_session])
            + (
                []
                if previous is None or previous.active_entry is None
                else [previous.active_entry.entry.schedule.holding_end_session]
            )
        )
        if target < checkpoint.epoch_start or target > settlement_end:
            raise ValueError("research_update.model_epoch_unavailable")
        decisions = tuple(
            p.formation_session
            for p in planned_local_qa_schedule(first, target)
            if first <= p.formation_session <= min(target, checkpoint.epoch_end)
        )
        source_changed = previous is not None and (
            self.updates._load_plan(previous.plan_hash).source_hash
            != workspace_score_source_identity(self.session.workspace)
        )
        if not decisions and source_changed:
            source_changed = (
                self.updates.revision_impact(self.updates._market(checkpoint, target), previous)
                is not None
            )
        if (
            not decisions
            and not source_changed
            and (
                previous is None
                or not (
                    (pending is not None and pending.schedule.entry_session <= target)
                    or (
                        previous.active_entry is not None
                        and previous.active_entry.entry.schedule.holding_end_session <= target
                    )
                )
            )
        ):
            return {
                "status": "OBSERVATIONS_PENDING",
                "task_id": None,
                "next_action": "WAIT_FOR_NEXT_COMPLETED_SESSION",
            }
        bindings = (
            (self.scoring._binding(package_id),)
            if len(checkpoint.recipe.components) == 1
            else tuple(
                self.scoring._binding(package_id, c.component_id)
                for c in checkpoint.recipe.components
            )
        )
        requests: list[tuple[int, date]] = []
        historical: list[str] = []
        for index, (binding, component) in enumerate(
            zip(bindings, checkpoint.recipe.components, strict=True)
        ):
            authority = self.scoring.authority_for_formation(
                binding, min(target, checkpoint.epoch_end)
            )
            existing = self._historical_scores(
                package_id, authority.authority.model_set.recipe_hash
            )
            needed = set(decisions)
            if component.weight_rule == "mu.iv0":
                seed, _, _, _ = load_observations(
                    self.calibration.store, self.calibration._binding(package_id).seed_hash
                )
                needed.update(
                    p.formation_session
                    for p in planned_local_qa_schedule(
                        seed.formation_sessions[-1], min(target, checkpoint.epoch_end)
                    )
                    if seed.formation_sessions[-1]
                    < p.formation_session
                    <= min(target, checkpoint.epoch_end)
                )
            if any(
                not (
                    authority.authority.supports(d)
                    or (
                        isinstance(authority, AdmittedRenewingInference)
                        and authority.authority.can_prepare(d)
                    )
                )
                for d in needed
            ):
                raise ValueError("research_update.model_epoch_unavailable")
            current_source = workspace_score_source_identity(self.session.workspace)
            requests.extend(
                (index, d)
                for d in sorted(needed)
                if d not in existing
                or (
                    d in decisions
                    and self.scoring.store.load_frozen_observation_snapshot(
                        existing[d].observation_snapshot_hash
                    ).source_binding_hash
                    != current_source
                )
            )
            historical.extend(existing[d].snapshot_hash for d in sorted(existing) if d <= target)
        requests.sort(key=lambda item: (item[1], item[0]))
        score_days = tuple(sorted({d for _, d in requests}))
        # Its own data plan, sealed into its own Task, never a data update's that waits (V604).
        state = self.data.plan(fresh=True)
        if state["status"] != "PLANNED":
            raise ValueError("research_update.workspace_transition_required")
        # The plan this call made, by the hash its answer names, never the data owner's last
        # one, which a concurrent plan may have replaced (V534).
        data_plan = self.data.prepare(str(state["plan_hash"]))
        # A bounded QA target never authorizes acquisition through today's date. The candidate
        # rechecks due were planned for today's target, so a bounded request leaves them due
        # for the update that reaches it (V467).
        request = data_plan.request.with_changes(
            target_market_session=target,
            **(
                {"candidate_recheck": None, "candidate_data_recheck": None}
                if target < data_plan.request.target_market_session
                else {}
            ),
        )
        data_plan = WorkspaceDataUpdatePlan.seal(
            workspace_id=data_plan.workspace_id,
            workspace_manifest_hash=data_plan.workspace_manifest_hash,
            binding=data_plan.binding,
            before=data_plan.before,
            request=request,
            valuation_grants=data_plan.valuation_grants,
        )
        plan = DecisionAdvancementPlan.create(
            workspace_manifest_hash=self.updates.manifest.manifest_hash,
            catalog_hash=self.updates.application.resolver.strategy_catalog_hash,
            package_id=package_id,
            score_binding=bindings[0] if len(bindings) == 1 else None,
            score_bindings=bindings if len(bindings) != 1 else (),
            checkpoint_hash=checkpoint.content_hash,
            parent_hash=None if previous is None else previous.content_hash,
            target=target,
            decision_sessions=decisions,
            score_sessions=score_days,
            score_requests=tuple(requests) if len(bindings) != 1 else (),
            history_score_hashes=tuple(historical),
            data_plan=data_plan,
            implementation_hash=self.implementation(),
            score_implementation_hash=scoring_identity(),
            calibration_implementation_hash=calibration_identity(),
            decision_implementation_hash=decision_identity(),
        )
        self._save("decision-programs", plan)
        return self._plan_body(plan, False)

    def _plan_body(self, plan: DecisionAdvancementPlan, cached: bool) -> dict[str, object]:
        checkpoint = self.updates.store.load_decision_checkpoint(plan.checkpoint_hash)
        lifecycles = {}
        for index, binding in enumerate(plan.bindings):
            inference = self.scoring.authority_for_formation(
                binding, min(plan.target, checkpoint.epoch_end)
            )
            if isinstance(inference, AdmittedRenewingInference):
                jobs = (
                    ()
                    if cached
                    else inference.authority.planned_refits(
                        self.scoring.store,
                        tuple(d for i, d in plan.requests if i == index),
                        allow_preparation=True,
                    )
                )
                lifecycles[inference.authority.component.component_id] = {
                    "rules": inference.authority.lifecycle.model_dump(mode="json"),
                    "fit_count": len(jobs),
                    "jobs": [{"vintage": v, "seed": s} for v, s in jobs],
                }
        lifecycle = (
            next(iter(lifecycles.values())) if len(plan.bindings) == 1 and lifecycles else None
        )
        held_inputs = self.data.historical_inputs_cover(plan.data_plan)
        provider_work = self.data.network_work(plan.data_plan)
        inputs_work = (
            f"Use admitted Data/Feature through session {plan.target.isoformat()}; "
            "no provider request. "
            if held_inputs
            else "Provider work: " + "; ".join(provider_work.values()) + ". "
            if provider_work
            else f"Complete local Feature/Panel through session {plan.target.isoformat()}; "
            "no provider request. "
        )
        return {
            "status": "PLANNED",
            "update_plan_hash": plan.content_hash,
            "strategy_package_id": plan.package_id,
            "target_session": plan.target.isoformat(),
            "decision_sessions": [d.isoformat() for d in plan.decision_sessions],
            "score_sessions": [d.isoformat() for d in plan.score_sessions],
            "cached": cached,
            "model_lifecycle": lifecycle,
            "component_lifecycles": lifecycles,
            "model_epoch_end": checkpoint.epoch_end.isoformat(),
            "source_access": "NOT_REQUIRED_EXACT_REUSE"
            if cached
            else self.data.source_access(plan.data_plan),
            "calibration": "CAUSAL_MATURED_HISTORY"
            if any(c.weight_rule == "mu.iv0" for c in checkpoint.recipe.components)
            else "NOT_CONSUMED_EW_COMPONENTS",
            "continuation_basis": "AS_ISSUED",
            "work": "Reopen verified artifacts; no source request, predictions or fits."
            if cached
            else inputs_work
            + (
                "Declared model renewal → scores → declared component sizing → book → report."
                if lifecycles
                else "Frozen scores → component sizing → book → report. Zero fits."
            ),
            "limitation": "POST_OBSERVED_QA_NOT_TIMELY_ADVICE",
            # The plan's way on: its run, which reuses an identical update already made or in
            # flight (V470).
            "next_requests": {
                "run": {"operation": "RESEARCH_UPDATE_RUN", "update_plan_hash": plan.content_hash}
            },
        }

    def _plan_of(self, task: TaskRecord, *, current: bool = True) -> DecisionAdvancementPlan:
        if task.task_kind != ADVANCEMENT_TASK_KIND or task.input.input_schema_id != SCHEMA:
            raise ValueError("research_update.task_kind_invalid")
        plan = DecisionAdvancementPlan.model_validate(task.input.payload["plan"])
        if (task.input, task.goal, task.plan) != task_contract(plan):
            raise ValueError("research_update.task_contract_invalid")
        if current:
            self._require(plan)
        return cast(DecisionAdvancementPlan, plan)

    def replan_request(self, task: TaskRecord) -> dict[str, object]:
        """Replan the validated Task's original package and target without readmitting it.

        Args:
            task: Durable Task whose declaration supplies the next request.

        Returns:
            The filled research-update planning request.

        Raises:
            ValueError: The Task's kind, sealed plan or workflow does not validate.
        """
        plan = self._plan_of(task, current=False)
        return {
            "operation": "RESEARCH_UPDATE_PLAN",
            "strategy_package_id": plan.package_id,
            "observed_through": plan.target.isoformat(),
        }

    def admit(self, plan: DecisionAdvancementPlan) -> CommandAdmission:
        """Admit exact planned advancement after verifying current source and plan freshness.

        Args:
            plan: Exact sealed prepared plan.

        Returns:
            Existing uncancelled matching task admission, a stopped one a rerun resumes reopened,
            or a newly admitted durable task.

        Raises:
            ValueError: Plan authority/freshness or captured workspace inputs changed before
                admission, or the network a stopped Task's stop names is still closed.
        """
        self._require(plan)
        envelope, goal, workflow = task_contract(plan)
        for task in self.session.task_control_registry.tasks():
            if task.input == envelope and task.lifecycle is not TaskLifecycle.CANCELLED:
                # Its data stage's stop a rerun resumes reopens it, as the data update's own does,
                # and a rerun while its network stays closed is refused by that stop (V600).
                task = self.data.resume_stopped(task, plan.data_plan)
                return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)
        self._fresh(plan)
        if (
            read_workspace_inputs(self.session.workspace, plan.data_plan.binding)
            != plan.data_plan.before
        ):
            raise ValueError("research_update.source_changed_before_admission")
        task = self.session.task_control_registry.admit(
            input_envelope=envelope, goal=goal, plan=workflow, observed_at=self.clock()
        ).record
        return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)

    def in_flight(self, plan: DecisionAdvancementPlan) -> CommandAdmission | None:
        """Find an admitted nonterminal task for this exact advancement envelope.

        A deferred one is not in flight: its plan's rerun resumes it once due, and is refused
        before then with its retry time (V601).

        Args:
            plan: Sealed plan used to derive exact task input.

        Returns:
            Matching queued/running/recovery/cancellation-pending admission, or None.
        """
        envelope, _, _ = task_contract(plan)
        for task in self.session.task_control_registry.tasks():
            if task.input == envelope and task.lifecycle in (
                TaskLifecycle.QUEUED,
                TaskLifecycle.RUNNING,
                TaskLifecycle.RECOVERY_REQUIRED,
                TaskLifecycle.CANCEL_REQUESTED,
            ):
                return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)
        return None

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one admitted advancement through retained session task ownership.

        Args:
            task_id: Exact task to reopen and validate.
            expected_task_hash: Optional optimistic task identity required by execution.
        """
        task = self.session.task_control_registry.task(task_id)
        self._plan_of(task)
        self.session.execute_admitted(task, self, self.clock, expected_task_hash)

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Describe structural plan/schema, checkpoint and framework execution compatibility.

        Args:
            task: Exact admitted task retaining its sealed plan.

        Returns:
            Compatibility contract bound to checkpoint, workflow and session execution identity.
        """
        plan = self._plan_of(task)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash(schema_structure(DecisionAdvancementPlan)),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=SCHEMA,
            domain_policy_hash=plan.checkpoint_hash,
            framework_identity_hash=self.session.execution_identity(plan.implementation_hash),
        )

    @staticmethod
    def _evidence(value: DecisionAdvancementStep) -> tuple[TaskEvidence, ...]:
        return (
            TaskEvidence(
                evidence_kind="portfolio-public.decision-advancement",
                content_hash=value.content_hash,
                reference="playpen://advancement/decision-stages/" + value.content_hash,
            ),
        )

    def _cancel(self, task: TaskRecord) -> None:
        if (
            self.session.task_control_registry.task(task.task_id).lifecycle
            is TaskLifecycle.CANCEL_REQUESTED
        ):
            raise _Cancelled

    @verified_model_read_scope(reuse_verified=True)
    def execute_stage(
        self, *, task: TaskRecord, execution: TaskExecution, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        """Execute or reopen one exact advancement stage under verified lifecycle admission.

        Args:
            task: Exact admitted advancement task.
            execution: Execution record supplied by task control.
            work_item: Declared advancement stage.

        Returns:
            READY with exact durable products, CANCELLED on cancellation or BLOCKED with typed
            refusal code.
        """
        plan, stage = self._plan_of(task), work_item.stage_id
        try:
            with verified_lifecycle_admissions():
                self._cancel(task)
                value = self._step(plan, stage)
                if value is None:
                    value = self._execute(plan, stage, task)
            return StageExecutionResult(StageDisposition.READY, evidence=self._evidence(value))
        except _Cancelled:
            return StageExecutionResult(StageDisposition.CANCELLED)
        except _Deferred as deferred:
            return deferred.result
        except ValueError as error:
            return StageExecutionResult(
                StageDisposition.BLOCKED, failure_code=failure_code_from(error)
            )

    def _execute(
        self, plan: DecisionAdvancementPlan, stage: str, task: TaskRecord
    ) -> DecisionAdvancementStep:
        if stage == STAGES[0]:
            self._fresh(plan)
            result = self.data.execute_step(plan.data_plan, DATA_STAGES[0], cancelled=lambda: False)
            self.data.verify_step(plan.data_plan, DATA_STAGES[0], result.evidence)
            return self._commit(plan, stage, (), plan.data_plan.content_hash)
        if stage == STAGES[1]:
            if self.data.historical_inputs_cover(plan.data_plan):
                if (
                    read_workspace_inputs(self.session.workspace, plan.data_plan.binding)
                    != plan.data_plan.before
                ):
                    raise ValueError("research_update.source_changed_before_readback")
                return self._commit(
                    plan,
                    stage,
                    (plan.data_plan.before.content_hash,),
                    score_source_identity(plan.data_plan.before),
                )
            for s in DATA_STAGES[1:]:
                result = self.data.execute_step(
                    plan.data_plan,
                    s,
                    cancelled=lambda: (
                        self.session.task_control_registry.task(task.task_id).lifecycle
                        is TaskLifecycle.CANCEL_REQUESTED
                    ),
                )
                if result.disposition is StageDisposition.CANCELLED:
                    raise _Cancelled
                if result.disposition is StageDisposition.DEFERRED:
                    # The provider asked to wait: the update defers with the data stage's own
                    # retry time, and its plan run again once due resumes it (V601).
                    raise _Deferred(result)
                if result.disposition is not StageDisposition.READY:
                    if result.failure_code == "workspace_data_update.source_access_not_admitted":
                        raise ValueError(
                            "research_update.input_source_access_not_admitted:"
                            + plan.target.isoformat()
                            + ","
                            + ",".join(self.data.network_work(plan.data_plan))
                        )
                    raise ValueError(result.failure_code or "research_update.inputs_incomplete")
                self.data.verify_step(plan.data_plan, s, result.evidence)
            return self._commit(
                plan,
                stage,
                (str(result.evidence[0].content_hash),),
                workspace_score_source_identity(self.session.workspace),
            )
        source = self._need(plan, STAGES[1]).source_hash
        if stage == STAGES[2]:
            self._fresh(plan)
            if workspace_score_source_identity(self.session.workspace) != source:
                raise ValueError("research_update.source_changed_before_seal")
            checkpoint = self.updates._checkpoint(plan.package_id)
            market = MarketDataRepository(self.session.workspace)
            manifest = market.current_quality_filtered_research_manifest(
                market_profile_id="us-current-index-research"
            )
            assert manifest is not None
            covered = market.manifest_raw_range(manifest)
            assert covered is not None
            first_input = checkpoint.initial_book.schedule.formation_session
            if any(c.weight_rule == "mu.iv0" for c in checkpoint.recipe.components):
                seed, _, _, _ = load_observations(
                    self.calibration.store, self.calibration._binding(plan.package_id).seed_hash
                )
                first_input = seed.formation_sessions[0]
            calendar = tuple(
                p.formation_session for p in planned_local_qa_schedule(covered[0], first_input)
            )
            if first_input not in calendar:
                raise ValueError("research_update.calibration_seed_history_unavailable")
            warm_start = calendar[max(0, calendar.index(first_input) - 20)]
            # Tradability needs twenty prior sessions; model inputs have their
            # own captured preparation. Do not duplicate unused decades here.
            market_start = min(warm_start, checkpoint.initial_book.schedule.formation_session)
            with span("materialize", "market_snapshot"):
                snapshot = read_local_qa_market_snapshot(
                    store=market,
                    start=market_start,
                    through=plan.target,
                    listing_ids=checkpoint.ordered_listing_ids,
                    observed_at=self.clock(),
                    retained_listing_ids=checkpoint.ordered_listing_ids,
                )
            if not set(plan.decision_sessions) <= {v.session_date for v in snapshot.bars}:
                raise ValueError("research_update.session_observations_missing")
            previous = self._parent(plan)
            # Verify the as-issued valuation bridge before prediction. Revised
            # inputs still feed new research; old proposals consume their own seal.
            with span("verify", "valuation_bridge"):
                self.updates.continued_market(snapshot, previous)
            with span("write", "market_inputs"):
                self._save("market-inputs", snapshot)
            products = [snapshot.content_hash]
            effective_authorities = {}
            for index, binding in enumerate(plan.bindings):
                current_authority = self.scoring._authority(binding)
                with span("read", "historical_scores"):
                    historical = [
                        self.scoring.store.load_frozen_component_score(h)
                        for h in plan.history_score_hashes
                    ]
                historical = [
                    s
                    for s in historical
                    if recipe_hashes_match(
                        s.component_recipe_hash, current_authority.authority.model_set.recipe_hash
                    )
                ]
                if not historical:
                    continue
                prior_score = max(historical, key=lambda s: s.formation_session)
                effective_authorities[index] = prior_score.inference_authority_hash
            pending = {
                index: tuple(
                    day
                    for component, day in plan.requests
                    if component == index
                    and self._step(
                        plan,
                        "features_"
                        + (f"{index}_" if plan.score_bindings else "")
                        + day.isoformat(),
                    )
                    is None
                )
                for index in range(len(plan.bindings))
            }
            prepared_inputs: dict[int, PreparedWorkspaceComponentInputs | None] = {}
            for index, day in plan.requests:
                self._cancel(task)
                score_plan = self._score_plan(plan, day, source, index)
                self._save("score-programs", score_plan, "plan_hash")
                key = "features_" + (f"{index}_" if plan.score_bindings else "") + day.isoformat()
                held = self._step(plan, key)
                if held is None:
                    binding = plan.bindings[index]
                    if (
                        index not in prepared_inputs
                        and binding.source_kind == "WORKSPACE_DATA_FEATURE"
                    ):
                        with span("materialize", "score_inputs"):
                            prepared_inputs[index] = self.scoring.prepare_inputs(
                                binding,
                                through=max(pending[index]),
                                expected_source_hash=source,
                                captured_authority_hash=effective_authorities.get(index),
                                observed_at=self.clock(),
                            )
                    with span("features", "score_features"):
                        result = self.scoring.execute_step(
                            score_plan,
                            SCORE_STAGES[1],
                            lambda _: "",
                            captured_authority_hash=effective_authorities.get(index),
                            prepared_inputs=prepared_inputs.get(index),
                        )
                    held = self._commit(
                        plan,
                        key,
                        (score_plan.plan_hash, str(result.evidence[0].content_hash)),
                        source,
                    )
                with span("verify", "score_features"):
                    self._verify_features(held)
                feature_preparation, _ = self.scoring.store.load_frozen_feature_preparation(
                    held.products[1]
                )
                effective_authorities[index] = feature_preparation.inference_authority_hash
                products.append(held.content_hash)
            if workspace_score_source_identity(self.session.workspace) != source:
                raise ValueError("research_update.source_changed_during_seal")
            return self._commit(plan, stage, tuple(products), source)
        captured = self._need(plan, STAGES[2])
        if stage == STAGES[3]:
            products = []
            for identity in captured.products[1:]:
                self._cancel(task)
                features = self._load("decision-stages", identity, DecisionAdvancementStep)
                with span("verify", "score_features"):
                    sp = self._verify_features(features)
                with span("predict", "scores"):
                    result = self.scoring.execute_step(
                        sp, SCORE_STAGES[2], {SCORE_STAGES[1]: features.products[1]}.__getitem__
                    )
                with span("verify", "scores"):
                    self.scoring.verify_step(sp, SCORE_STAGES[2], result.evidence)
                products.append(str(result.evidence[0].content_hash))
            return self._commit(plan, stage, tuple(products), source)
        market_snapshot = self._load("market-inputs", captured.products[0], LocalQAMarketSnapshot)
        if stage == STAGES[4]:
            market_rows: PreparedLocalQASnapshotRows | None = None
            checkpoint = self.updates.store.load_decision_checkpoint(plan.checkpoint_hash)
            scores = {}
            for h in (*plan.history_score_hashes, *self._need(plan, STAGES[3]).products):
                value = self.scoring.store.load_frozen_component_score(h)
                scores[value.component_recipe_hash, value.formation_session] = value
            products = []
            for day in plan.decision_sessions:
                children = []
                for index, (component, recipe_hash) in enumerate(
                    zip(checkpoint.recipe.components, checkpoint.model_recipe_hashes, strict=True)
                ):
                    self._cancel(task)
                    score = scores[recipe_hash, day]
                    key = (
                        ("calibration_" if component.weight_rule == "mu.iv0" else "ew_")
                        + (f"{index}_" if plan.score_bindings else "")
                        + day.isoformat()
                    )
                    held = self._step(plan, key)
                    if held is None and component.weight_rule == "mu.iv0":
                        if market_rows is None:
                            with span("materialize", "calibration_rows"):
                                market_rows = PreparedLocalQASnapshotRows.from_artifact(
                                    source_root=self.ledger.root,
                                    source_category="market-inputs",
                                    content_hash=captured.products[0],
                                    artifact_root=self.session.workspace / "artifacts",
                                    capacity=lambda size: require_storage_capacity(
                                        self.session.workspace, additional_bytes=size
                                    ),
                                )
                        cp = CalibrationPlan.create(
                            workspace_manifest_hash=self._owner_fields(CALIBRATION_PLAN_FIELDS),
                            binding=self.calibration._binding(plan.package_id),
                            score_snapshot_hash=score.snapshot_hash,
                            history_score_hashes=tuple(
                                v.snapshot_hash
                                for (r, d), v in sorted(scores.items())
                                if r == recipe_hash and d <= day
                            ),
                            source_hash=source,
                            implementation_hash=plan.calibration_implementation_hash,
                        )
                        self._save("calibration-programs", cp, "plan_hash")
                        with (
                            self.session.mutation_gate.hold(),
                            span("compute", "calibration_observations"),
                        ):
                            obs = self.calibration.execute_step(
                                cp, CALIBRATION_STAGES[1], lambda _: "", captured=market_rows
                            )
                        obs_hash = str(obs.evidence[0].content_hash)
                        with span("compute", "calibration"):
                            result = self.calibration.execute_step(
                                cp,
                                CALIBRATION_STAGES[2],
                                {CALIBRATION_STAGES[1]: obs_hash}.__getitem__,
                            )
                        held = self._commit(
                            plan,
                            key,
                            (cp.plan_hash, obs_hash, str(result.evidence[0].content_hash)),
                            source,
                        )
                    elif held is None:
                        with span("compute", "ew_input"):
                            value = self._ew_input(checkpoint, score, market_snapshot, source)
                        self.calibration.store.publish(
                            category="prepared-component-inputs",
                            value=value,
                            identity_field="content_hash",
                        )
                        held = self._commit(
                            plan,
                            key,
                            (
                                score.snapshot_hash,
                                value.content_hash,
                                market_snapshot.content_hash,
                                checkpoint.content_hash,
                            ),
                            source,
                        )
                    with span("verify", "component_input"):
                        self._verify_input(held)
                    children.append(held.content_hash)
                if len(children) == 1:
                    products.append(children[0])
                else:
                    parts = tuple(
                        self._verify_input(
                            self._load("decision-stages", h, DecisionAdvancementStep)
                        )
                        for h in children
                    )
                    if any(not isinstance(part, PreparedPortfolioComponentInput) for part in parts):
                        raise ValueError("research_update.nested_book_input_invalid")
                    value = PreparedPortfolioBookInput.create(
                        component_ids=checkpoint.package.component_ids, components=parts
                    )
                    self.calibration.store.publish(
                        category="prepared-book-inputs", value=value, identity_field="content_hash"
                    )
                    step = self._commit(
                        plan,
                        "book_input_" + day.isoformat(),
                        (*children, value.content_hash),
                        source,
                    )
                    products.append(step.content_hash)
            return self._commit(plan, stage, tuple(products), source)
        if stage == STAGES[5]:
            checkpoint = self.updates.store.load_decision_checkpoint(plan.checkpoint_hash)
            previous = self._parent(plan)
            history = self.updates.history(plan.checkpoint_hash)
            products = []
            inputs = [
                self._verify_input(self._load("decision-stages", h, DecisionAdvancementStep))
                for h in self._need(plan, STAGES[4]).products
            ]
            days = list(plan.decision_sessions)
            if not days or days[-1] < plan.target:
                days.append(plan.target)
            by_day = {v.formation_session: v for v in inputs}
            for day in days:
                self._cancel(task)
                key = "book_" + day.isoformat()
                held = self._step(plan, key)
                if held is None:
                    orphan = self.index.open(
                        _DECISION_CANDIDATES, canonical_hash([plan.content_hash, day])
                    )
                    if orphan is not None:
                        held = self._commit(plan, key, (orphan.content_hash,), source)
                if held is None:
                    raw = local_qa_prefix(
                        market_snapshot,
                        through=day,
                        start=checkpoint.initial_book.schedule.formation_session,
                    )
                    self.updates.store.content.publish_model(
                        category="decision-observations", value=raw, identity_field="content_hash"
                    )
                    prefix = self.updates.continued_market(raw, previous)
                    self.updates.store.content.publish_model(
                        category="decision-observations",
                        value=prefix,
                        identity_field="content_hash",
                    )
                    prepared = by_day.get(day)
                    dp = PortfolioUpdatePlan.create(
                        strategy_package_id=plan.package_id,
                        workspace_manifest_hash=self._owner_fields(DECISION_PLAN_FIELDS),
                        catalog_hash=plan.catalog_hash,
                        checkpoint_hash=plan.checkpoint_hash,
                        parent_hash=None if previous is None else previous.content_hash,
                        prepared_input_hash=None if prepared is None else prepared.content_hash,
                        requested_input_hash=None if prepared is None else prepared.content_hash,
                        observed_through=day,
                        source_hash=source,
                        market_hash=prefix.content_hash,
                        raw_market_hash=raw.content_hash if raw != prefix else None,
                        prior_market_hash=None
                        if previous is None
                        else previous.source_snapshot_hash,
                        prior_raw_market_hash=None
                        if previous is None
                        else self.updates.raw_basis(previous).content_hash,
                        implementation_hash=plan.decision_implementation_hash,
                    )
                    self.updates.store.content.publish_model(
                        category="decision-plans", value=dp, identity_field="plan_hash"
                    )
                    with span("compute", "advance_state"):
                        value = advance_decision_state(
                            checkpoint=checkpoint,
                            previous=previous,
                            prepared=prepared,
                            observed=prefix,
                            plan_hash=dp.plan_hash,
                            published_at=self.clock(),
                            previous_checkpoint=self.updates.prior_checkpoint(previous, checkpoint),
                            revised_source_hash=raw.content_hash if raw != prefix else None,
                            source_revision=self.updates.revision_impact(raw, previous),
                        )
                    with span("write", "decision_html"):
                        html, _ = self.updates.store.publish_html(
                            render_decision_update(checkpoint, (value,))
                        )
                    value = PortfolioUpdatePublication.create(
                        **{
                            **{
                                k: getattr(value, k)
                                for k in type(value).model_fields
                                if k != "content_hash"
                            },
                            "html_hash": html,
                        }
                    )
                    # Timestamped value is committed under the request before it
                    # becomes reachable from the public Portfolio parent.
                    self.index.commit(
                        _DECISION_CANDIDATES, canonical_hash([plan.content_hash, day]), value
                    )
                    held = self._commit(plan, key, (value.content_hash,), source)
                value = self._load(
                    "decision-candidates", held.products[0], PortfolioUpdatePublication
                )
                if value.parent_hash != (None if previous is None else previous.content_hash):
                    raise ValueError("research_update.branch_parent_invalid")
                previous = value
                history = (*history, value)
                products.append(value.content_hash)
            return self._commit(plan, stage, tuple(products), source)
        if stage == STAGES[6]:
            branch = self._branch(plan)
            # Tail links are unreachable until the original parent's first link
            # commits; interruption cannot expose half of the catch-up branch.
            for value in reversed(branch):
                self.updates.store.publish_decision_update(value)
            from alphalattice.control.product_host.composition.rolling_portfolio_report import (
                publish_rolling_report_heads,
            )

            checkpoint = self.updates.store.load_decision_checkpoint(plan.checkpoint_hash)
            with span("write", "rolling_report"):
                publish_rolling_report_heads(
                    application=self.updates.application,
                    checkpoint=checkpoint,
                    publications=self.updates.history(plan.checkpoint_hash),
                )
            return self._commit(plan, stage, tuple(v.content_hash for v in branch), source)
        raise ValueError("research_update.stage_unknown")

    def _owner_fields(self, fields: tuple[str, ...]) -> str:
        """The manifest fields a stage's owner binds, as that owner binds them (V468, V223).

        Each owner seals its plan with the fields it reads (OW10), not the whole manifest; the
        update verified the whole manifest against its own plan before the stage (`_require`),
        so these are the planned fields.
        """
        return manifest_fields_hash(self.updates.manifest, fields)

    def _score_plan(
        self, plan: DecisionAdvancementPlan, day: date, source: str, binding_index: int = 0
    ) -> StrategyScorePlan:
        return StrategyScorePlan.create(
            workspace_id=self.updates.manifest.workspace_id,
            workspace_manifest_hash=self._owner_fields(SCORE_PLAN_FIELDS),
            binding=plan.bindings[binding_index],
            formation_session=day,
            source_identity_hash=source,
            implementation_hash=plan.score_implementation_hash,
        )

    def _verify_features(
        self,
        step: DecisionAdvancementStep,
        *,
        verified_feature_plans: dict[tuple[str, str, tuple[str, ...]], StrategyScorePlan]
        | None = None,
        verified_feature_preparations: dict[str, FrozenFeaturePreparation] | None = None,
    ) -> StrategyScorePlan:
        key = (step.plan_hash, step.content_hash, step.products)
        if verified_feature_plans is not None and key in verified_feature_plans:
            return verified_feature_plans[key]
        plan = self._load("score-programs", step.products[0], StrategyScorePlan, "plan_hash")
        self.scoring.verify_step(
            plan, SCORE_STAGES[1], self.scoring._evidence(SCORE_STAGES[1], step.products[1])
        )
        if verified_feature_plans is not None:
            verified_feature_plans[key] = plan
        if verified_feature_preparations is not None:
            verified_feature_preparations[step.products[1]] = self.scoring.store._load(
                "frozen-feature-preparations",
                step.products[1],
                "preparation_hash",
                FrozenFeaturePreparation,
            )
        return plan

    def _ew_input(
        self,
        checkpoint: PortfolioDecisionCheckpoint,
        score: FrozenComponentScoreSnapshot,
        snapshot: LocalQAMarketSnapshot,
        source_hash: str,
    ) -> PreparedPortfolioComponentInput:
        matching = [
            index
            for index, value in enumerate(checkpoint.model_recipe_hashes)
            if recipe_hashes_match(score.component_recipe_hash, value)
        ]
        if len(matching) != 1 or score.strategy_package_hash != checkpoint.package.package_hash:
            raise ValueError("research_update.score_component_invalid")
        component = checkpoint.recipe.components[matching[0]]
        if component.weight_rule != "ew":
            raise ValueError("research_update.calibration_required")
        day = score.formation_session
        history = tuple(
            p.formation_session for p in snapshot.schedule if p.formation_session <= day
        )[-20:]
        bars = {
            (bar.session_date, bar.listing_id): bar
            for bar in snapshot.bars
            if bar.session_date <= day
        }
        eligible: BoolArray = np.asarray(
            [
                decision_eligible_at_close(
                    listing_id=name, formation_session=day, history=history, bars=bars
                )
                for name in score.ordered_listing_ids
            ],
            dtype=np.bool_,
        )
        return compile_ew_component_input(
            request_hash=canonical_hash(
                [source_hash, score.snapshot_hash, checkpoint.content_hash]
            ),
            score=score,
            current_eligible=eligible,
            outsider_sentinel=component.outsider_sentinel,
        )

    def _verify_input(
        self, step: DecisionAdvancementStep
    ) -> PreparedPortfolioComponentInput | PreparedPortfolioBookInput:
        if step.step.startswith("ew_"):
            if len(step.products) != 4:
                raise ValueError("research_update.ew_evidence_invalid")
            score = self.scoring.store.load_frozen_component_score(step.products[0])
            value = self.calibration.store.load(
                category="prepared-component-inputs",
                content_hash=step.products[1],
                model=PreparedPortfolioComponentInput,
                identity_field="content_hash",
            )
            snapshot = self._load("market-inputs", step.products[2], LocalQAMarketSnapshot)
            checkpoint = self.updates.store.load_decision_checkpoint(step.products[3])
            if value != self._ew_input(checkpoint, score, snapshot, step.source_hash):
                raise ValueError("research_update.ew_evidence_invalid")
            return value
        if step.step.startswith("book_input_"):
            value = self.calibration.store.load(
                category="prepared-book-inputs",
                content_hash=step.products[-1],
                model=PreparedPortfolioBookInput,
                identity_field="content_hash",
            )
            parts = tuple(
                self._verify_input(self._load("decision-stages", h, DecisionAdvancementStep))
                for h in step.products[:-1]
            )
            if value.components != parts:
                raise ValueError("research_update.book_input_evidence_invalid")
            return value
        plan = self._load("calibration-programs", step.products[0], CalibrationPlan, "plan_hash")
        observation, _, _, _ = load_observations(self.calibration.store, step.products[1])
        if (
            observation.source_hashes[:2] != (plan.binding.seed_hash, plan.source_hash)
            or observation.score_snapshot_hashes != plan.history_score_hashes
        ):
            raise ValueError("research_update.calibration_evidence_invalid")
        return self.calibration.read_input(plan, step.products[1], step.products[2])

    def _completed_scores(self) -> tuple[FrozenComponentScoreSnapshot, ...]:
        values: list[FrozenComponentScoreSnapshot] = []
        for task in self.session.task_control_registry.tasks():
            if task.input.input_schema_id == SCHEMA and task.lifecycle is TaskLifecycle.SUCCEEDED:
                plan = self._plan_of(task, current=False)
                self._verify_task_evidence(task, plan)
                self._verify_products(plan, STAGES[3])
                values.extend(
                    self.scoring.store.load_frozen_component_score(h)
                    for h in self._need(plan, STAGES[3]).products
                )
        return tuple(values)

    def _completed_input(
        self, content_hash: str
    ) -> tuple[str, PreparedPortfolioComponentInput | PreparedPortfolioBookInput] | None:
        for task in self.session.task_control_registry.tasks():
            if task.input.input_schema_id == SCHEMA and task.lifecycle is TaskLifecycle.SUCCEEDED:
                plan = self._plan_of(task, current=False)
                for h in self._need(plan, STAGES[4]).products:
                    step = self._load("decision-stages", h, DecisionAdvancementStep)
                    value = self._verify_input(step)
                    if value.content_hash == content_hash:
                        self._verify_task_evidence(task, plan)
                        return step.source_hash, value
        return None

    def _historical_scores(
        self, package_id: str, component_recipe_hash: str
    ) -> dict[date, FrozenComponentScoreSnapshot]:
        return {
            v.formation_session: v
            for v in select_calibration_scores(
                self.scoring.published_scores(),
                package_hash=self.scoring.packages[package_id].package_hash,
                component_recipe_hash=component_recipe_hash,
                issued=self.updates.issued_score_hashes(package_id, component_recipe_hash),
            )
        }

    def _branch(self, plan: DecisionAdvancementPlan) -> tuple[PortfolioUpdatePublication, ...]:
        branch = tuple(
            self._load("decision-candidates", h, PortfolioUpdatePublication)
            for h in self._need(plan, STAGES[5]).products
        )
        parent = plan.parent_hash
        prior = self._parent(plan)
        for value in branch:
            if (
                value.parent_hash != parent
                or value.checkpoint_hash
                != self.updates.store.load_decision_checkpoint(plan.checkpoint_hash).history_hash
                or value.html_hash is None
            ):
                raise ValueError("research_update.branch_binding_invalid")
            dp = self.updates._load_plan(value.plan_hash)
            if (
                dp.parent_hash != value.parent_hash
                or dp.checkpoint_hash != plan.checkpoint_hash
                or dp.market_hash != value.source_snapshot_hash
                or (
                    dp.raw_market_hash is not None
                    and dp.prior_market_hash
                    != (None if prior is None else prior.source_snapshot_hash)
                )
                or (
                    dp.prior_raw_market_hash is not None
                    and (
                        prior is None
                        or dp.prior_raw_market_hash != self.updates.raw_basis(prior).content_hash
                    )
                )
            ):
                raise ValueError("research_update.branch_source_lineage_invalid")
            self.updates._observed(dp)
            self.updates.store.load_html(value.html_hash)
            parent = value.content_hash
            prior = value
        if not branch or branch[-1].observed_through != plan.target:
            raise ValueError("research_update.branch_incomplete")
        return branch

    def _verify_products(
        self,
        plan: DecisionAdvancementPlan,
        stage: str,
        *,
        verified_branch: tuple[PortfolioUpdatePublication, ...] | None = None,
        verified_feature_plans: dict[tuple[str, str, tuple[str, ...]], StrategyScorePlan]
        | None = None,
        verified_feature_preparations: dict[str, FrozenFeaturePreparation] | None = None,
    ) -> tuple[PortfolioUpdatePublication, ...] | None:
        step = self._need(plan, stage)
        if stage == STAGES[0]:
            if step.products or step.source_hash != plan.data_plan.content_hash:
                raise ValueError("research_update.request_evidence_invalid")
        elif stage == STAGES[1]:
            if step.products == (plan.data_plan.before.content_hash,):
                if not self.data.historical_inputs_cover(
                    plan.data_plan
                ) or step.source_hash != score_source_identity(plan.data_plan.before):
                    raise ValueError("research_update.input_readback_evidence_invalid")
                return None
            evidence = (
                TaskEvidence(
                    evidence_kind="workspace_data_update.stage",
                    reference=f"playpen://workspace-data-update/{DATA_STAGES[-1]}/{step.products[0]}",
                    content_hash=step.products[0],
                ),
            )
            self.data.verify_step(plan.data_plan, DATA_STAGES[-1], evidence, require_current=False)
        elif stage == STAGES[2]:
            snapshot = self._load("market-inputs", step.products[0], LocalQAMarketSnapshot)
            if snapshot.through != plan.target or len(step.products) != len(plan.requests) + 1:
                raise ValueError("research_update.input_axis_invalid")
            for (index, day), h in zip(plan.requests, step.products[1:], strict=True):
                v = self._load("decision-stages", h, DecisionAdvancementStep)
                sp = self._verify_features(
                    v,
                    verified_feature_plans=verified_feature_plans,
                    verified_feature_preparations=verified_feature_preparations,
                )
                if (
                    sp != self._score_plan(plan, day, step.source_hash, index)
                    or v.plan_hash != plan.content_hash
                ):
                    raise ValueError("research_update.features_binding_invalid")
        elif stage == STAGES[3]:
            features = self._need(plan, STAGES[2]).products[1:]
            for h, f in zip(step.products, features, strict=True):
                v = self._load("decision-stages", f, DecisionAdvancementStep)
                sp = self._verify_features(
                    v,
                    verified_feature_plans=verified_feature_plans,
                    verified_feature_preparations=verified_feature_preparations,
                )
                score = self.scoring.verify_score_step(
                    sp, SCORE_STAGES[2], self.scoring._evidence(SCORE_STAGES[2], h)
                )
                preparation = (
                    None
                    if verified_feature_preparations is None
                    else verified_feature_preparations.get(v.products[1])
                )
                if preparation is None:
                    preparation, _ = self.scoring.store.load_frozen_feature_preparation(
                        v.products[1]
                    )
                if (
                    score.observation_snapshot_hash != preparation.observation_snapshot_hash
                    or score.feature_values_hashes != preparation.feature_values_hashes
                ):
                    raise ValueError("research_update.score_input_mismatch")
        elif stage == STAGES[4]:
            for day, h in zip(plan.decision_sessions, step.products, strict=True):
                v = self._load("decision-stages", h, DecisionAdvancementStep)
                value = self._verify_input(v)
                if v.plan_hash != plan.content_hash or value.formation_session != day:
                    raise ValueError("research_update.input_binding_invalid")
        elif stage in STAGES[5:]:
            branch = verified_branch if verified_branch is not None else self._branch(plan)
            if stage == STAGES[6]:
                if step.products != tuple(v.content_hash for v in branch):
                    raise ValueError("research_update.publication_evidence_invalid")
                for value in branch:
                    if (
                        self.updates.store.decision_successor(
                            value.checkpoint_hash, value.parent_hash
                        )
                        != value
                    ):
                        raise ValueError("research_update.publication_missing")
            elif stage == STAGES[5]:
                return branch
        else:
            raise ValueError("research_update.stage_unknown")
        return None

    @verified_model_read_scope(reuse_verified=True)
    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Verify durable stage products and require exact task evidence equality.

        Args:
            task: Exact admitted advancement task.
            execution: Execution record supplied by task control.
            work_item: Declared advancement stage.
            evidence: Proposed durable stage evidence.

        Returns:
            Original evidence after product verification.

        Raises:
            ValueError: Reopened products or exact declared evidence differ.
        """
        plan = self._plan_of(task)
        with verified_lifecycle_admissions():
            self._verify_products(plan, work_item.stage_id)
        if evidence != self._evidence(self._need(plan, work_item.stage_id)):
            raise ValueError("research_update.task_evidence_invalid")
        return evidence

    def _verify_task_evidence(self, task: TaskRecord, plan: DecisionAdvancementPlan) -> None:
        """Self-consistent replacement artifacts cannot replace what this Task verified."""
        states = {
            v.stage_id: v for v in self.session.task_control_registry.work_items(task.task_id)
        }
        for stage in STAGES:
            state = states.get(stage)
            if (
                state is None
                or state.lifecycle.value != "VERIFIED"
                or state.evidence != self._evidence(self._need(plan, stage))
            ):
                raise ValueError("research_update.task_evidence_invalid")

    def reusable(self, plan: DecisionAdvancementPlan) -> UUID | None:
        """Verify succeeded exact-plan task evidence, all stage products and current score source.

        Args:
            plan: Exact prepared plan whose authority is required.

        Returns:
            Reusable succeeded task identity, or None.

        Raises:
            ValueError: Existing task/product evidence differs or changed source requires a new
                plan.
        """
        self._require(plan)
        for task in self.session.task_control_registry.tasks():
            if (
                task.input.input_schema_id == SCHEMA
                and task.lifecycle is TaskLifecycle.SUCCEEDED
                and self._plan_of(task, current=False) == plan
            ):
                self._verify_task_evidence(task, plan)
                for stage in STAGES:
                    self._verify_products(plan, stage)
                if (
                    workspace_score_source_identity(self.session.workspace)
                    != self._need(plan, STAGES[2]).source_hash
                ):
                    raise ValueError("research_update.source_changed_requires_new_plan")
                return cast(UUID, task.task_id)
        return None

    def publication_task(self, publication_hash: str) -> UUID | None:
        """Find one succeeded advancement task that produced an exact publication.

        Args:
            publication_hash: Exact retained update publication identity.

        Returns:
            Unique matching task identity, or None.

        Raises:
            ValueError: More than one succeeded task binds this publication.
        """
        matches = [
            t.task_id
            for t in self.session.task_control_registry.tasks()
            if t.input.input_schema_id == SCHEMA
            and t.lifecycle is TaskLifecycle.SUCCEEDED
            and publication_hash in self._need(self._plan_of(t, current=False), STAGES[5]).products
        ]
        if len(matches) > 1:
            raise ValueError("research_update.publication_task_ambiguous")
        return matches[0] if matches else None

    def readback(
        self, task_id: UUID | None = None, *, publication_hash: str | None = None
    ) -> dict[str, object]:
        """Read exact task/publication lineage and bounded QA update facts.

        Args:
            task_id: Optional exact task; otherwise select latest advancement task.
            publication_hash: Optional exact publication bound to the task or its parent.

        Returns:
            No-update/task state or verified publication view with exact plan, target, counts and
            completed score/model evidence.

        Raises:
            ValueError: Requested publication is not task-bound or completed task/product
                verification fails.
        """
        task = (
            self.session.task_control_registry.task(task_id)
            if task_id
            else next(
                (
                    t
                    for t in reversed(self.session.task_control_registry.tasks())
                    if t.input.input_schema_id == SCHEMA
                ),
                None,
            )
        )
        if task is None:
            return {"status": "NO_RESEARCH_UPDATE", "task_id": None}
        plan = self._plan_of(task, current=False)
        completed = task.lifecycle is TaskLifecycle.SUCCEEDED
        value: PortfolioUpdatePublication | None
        if completed:
            self._verify_task_evidence(task, plan)
            branch: tuple[PortfolioUpdatePublication, ...] | None = None
            verified_feature_plans: dict[tuple[str, str, tuple[str, ...]], StrategyScorePlan] = {}
            verified_feature_preparations: dict[str, FrozenFeaturePreparation] = {}
            for stage in STAGES:
                stage_branch = self._verify_products(
                    plan,
                    stage,
                    verified_branch=branch,
                    verified_feature_plans=verified_feature_plans,
                    verified_feature_preparations=verified_feature_preparations,
                )
                if stage_branch is not None:
                    branch = stage_branch
            assert branch is not None
            value = (
                branch[-1]
                if publication_hash is None
                else next((v for v in branch if v.content_hash == publication_hash), None)
            )
            if publication_hash == plan.parent_hash and publication_hash is not None:
                value = self._parent(plan)
        else:
            value = self._parent(plan)
        if publication_hash is not None and (
            value is None or value.content_hash != publication_hash
        ):
            raise ValueError("research_update.publication_not_bound_to_task")
        if value is None:
            return {
                "status": task.lifecycle.value,
                "task_id": str(task.task_id),
                "failure_code": task.failure_code,
            }
        checkpoint = self.updates.store.load_decision_checkpoint(
            value.input_checkpoint_hash or value.checkpoint_hash
        )
        history = self.updates.history(plan.checkpoint_hash)
        prefix = history[
            : next(i for i, v in enumerate(history) if v.content_hash == value.content_hash) + 1
        ]
        body = self.updates.publication_body(
            checkpoint,
            value,
            prefix,
            task_id=task.task_id,
            lifecycle=None if completed else task.lifecycle.value,
        )
        scores = (
            tuple(
                self.scoring.store.load_frozen_component_score(h)
                for h in self._need(plan, STAGES[3]).products
            )
            if completed
            else None
        )
        body["update"] = {
            "plan_hash": plan.content_hash,
            "target_session": plan.target.isoformat(),
            "decision_count": len(plan.decision_sessions),
            "requested_score_sessions": len(plan.score_sessions),
            "score_artifact_prediction_calls": (
                sum(v.prediction_calls for v in scores) if scores is not None else None
            ),
            "claim": "POST_OBSERVED_QA_NOT_TIMELY_ADVICE",
        }
        if completed:
            assert scores is not None
            update = cast(dict[str, object], body["update"])
            update["score_artifact_fit_calls"] = sum(v.fit_calls for v in scores)
            update["model_set_publications"] = tuple(
                dict.fromkeys(
                    v.model_set_publication_hash
                    for v in scores
                    if v.model_set_publication_hash is not None
                )
            )
            update["model_vintages"] = [
                value.lifecycle.vintages(value.formation)
                for identity in cast(tuple[str, ...], update["model_set_publications"])
                if (
                    value := self.scoring.store._load(
                        "lifecycle-model-sets", identity, "content_hash", AlphaModelSetPublication
                    )
                )
            ]
        if not completed:
            body["failure_code"] = task.failure_code
        return body


@dataclass
class DecisionAdvancementCommand:
    """Carry a prepared plan through the deterministic task admission/execution owner."""

    application: DecisionAdvancementApplication
    plan: DecisionAdvancementPlan | None = None
    command_kind: str = ADVANCEMENT_TASK_KIND

    def admit(self) -> CommandAdmission:
        """Require a prepared plan and delegate exact advancement admission.

        Returns:
            Durable task admission from the retained application.

        Raises:
            ValueError: No plan is retained or application admission refuses it.
        """
        if self.plan is None:
            raise ValueError("research_update.plan_required")
        return self.application.admit(self.plan)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Delegate admitted advancement execution with optional optimistic task identity.

        Args:
            task_id: Exact admitted task.
            expected_task_hash: Optional expected task identity.
        """
        self.application.execute(task_id, expected_task_hash=expected_task_hash)
