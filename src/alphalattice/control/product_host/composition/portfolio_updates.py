"""Compose one bounded Portfolio update; the Desk owns intent and settlement."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Self, cast
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import PortfolioPerSideCostAssumption
from alphalattice.capabilities.portfolio_backtesting.metrics import evaluate_net_simple_return_path
from alphalattice.control.product_host.composition.portfolio_application import (
    PortfolioResearchApplication,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    ResearchWorkspaceManifestHolder,
    held,
    manifest_fields_hash,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.strategy_calibration import (
    StrategyCalibrationApplication,
)
from alphalattice.control.product_host.composition.strategy_score_inputs import (
    workspace_score_source_identity,
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
from alphalattice.foundation.causal_outcomes.execution.readers import read_local_qa_market_snapshot
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.interface.local_application.dispatcher import CommandAdmission
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioDecisionCheckpoint,
    PortfolioSourceRevision,
    PortfolioUpdatePublication,
    advance_decision_state,
    continue_issued_observations,
    require_proposal_position,
    source_revision_impact,
    transition_decision_checkpoint,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    recipe_hashes_match,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.static import render_decision_update
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash

PLAN_FIELDS = ("decision_updates", "score_inputs")
"""The manifest fields a plan reads: the decision checkpoints installed and the score inputs the
checkpoint's models bind. A plan binds them, never the whole manifest, so a publication of fields it
does not read leaves it applicable (V223, OW10)."""

TASK_KIND = "portfolio_decision_update"
STAGES = ("verify_update", "seal_market_observations", "publish_decision_settlement")


IMPLEMENTATION_ROLE = "product_host.portfolio_updates"
"""The role a move of this implementation is recorded under in `config/identity-successors.json`,
so a plan sealed under its predecessor stays current (binding plan R1, LAWS.md ID1)."""


def _implementation_hash() -> str:
    return source_rule_closure_hash(
        root=resolve_playpen_root(Path(__file__)),
        semantic_owner="product_host",
        numerical_role="CONDITIONAL_PORTFOLIO_QA",
        tracked_paths=tuple(
            "src/alphalattice/" + p
            for p in (
                "control/product_host/composition/portfolio_updates.py",
                "control/product_host/composition/strategy_score_inputs.py",
                "investment/portfolio_strategy_lab/application/decision_updates.py",
                "investment/portfolio_strategy_lab/application/tranche_book_execution.py",
                "investment/portfolio_strategy_lab/application/calibration.py",
                "investment/portfolio_strategy_lab/policies/tranche_book.py",
                "investment/portfolio_strategy_lab/policies/buffered_equal_weight.py",
                "investment/portfolio_strategy_lab/policies/post_observed_authority.py",
                "investment/portfolio_strategy_lab/publication/portfolio_ledger.py",
                "control/workspace_runtime/content_store.py",
                "investment/portfolio_strategy_lab/reporting/static.py",
                "capabilities/portfolio_backtesting/execution.py",
                "capabilities/portfolio_backtesting/contracts.py",
                "capabilities/portfolio_backtesting/state.py",
                "foundation/causal_outcomes/execution/readers.py",
                "foundation/causal_outcomes/execution/contracts.py",
                "foundation/causal_outcomes/execution/compile.py",
                "foundation/causal_outcomes/execution/methods.py",
                "foundation/market_data_ops/returns/execution.py",
                "kernel/data/calendar.py",
            )
        ),
    )


class PortfolioUpdatePlan(BaseModel):  # type: ignore[misc]
    """Seal observed decision inputs, parent lineage and current implementation binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    strategy_package_id: str
    workspace_manifest_hash: str
    catalog_hash: str
    checkpoint_hash: str
    parent_hash: str | None
    prepared_input_hash: str | None
    requested_input_hash: str | None
    observed_through: date
    source_hash: str
    market_hash: str
    implementation_hash: str
    plan_hash: str
    raw_market_hash: str | None = Field(default=None, exclude_if=lambda v: v is None)
    prior_market_hash: str | None = Field(default=None, exclude_if=lambda v: v is None)
    prior_raw_market_hash: str | None = Field(default=None, exclude_if=lambda v: v is None)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal an explicit portfolio decision update plan.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical plan_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        payload = cls.model_construct(**values).model_dump(mode="json", exclude={"plan_hash"})
        return cls(**payload, plan_hash=canonical_hash(payload))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def identity(self) -> Self:
        """Require the exact canonical portfolio update plan identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Serialized plan fields differ from plan_hash.
        """
        if self.plan_hash != canonical_hash(self.model_dump(mode="json", exclude={"plan_hash"})):
            raise ValueError("portfolio_update.plan_identity_invalid")
        return self


def _task_contract(
    plan: PortfolioUpdatePlan,
) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    envelope = TaskInputEnvelope.create(
        task_kind=TASK_KIND,
        input_schema_id="portfolio-decision-update",
        payload={"plan": plan.model_dump(mode="json")},
    )
    goal = ResearchGoal.create(
        goal_kind="PORTFOLIO_DECISION_SETTLEMENT_QA",
        input_hash=envelope.input_hash,
        deliverable_kind="PortfolioUpdatePublication",
        summary="Publish a conditional QA proposal and observed settlement.",
    )
    workflow = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash(STAGES),
        verifier_catalog_hash=canonical_hash(STAGES),
        work_items=tuple(
            WorkItemDefinition.create(
                stage_id=s, dependency_ids=STAGES[:i], verifier_id=f"portfolio_update.{s}"
            )
            for i, s in enumerate(STAGES)
        ),
    )
    return envelope, goal, workflow


class PortfolioUpdateApplication:
    """Own admitted decision updates and durable issued-observation readback."""

    task_kind = TASK_KIND
    replans = (
        TaskReplan(
            task_kind=TASK_KIND, preview="PORTFOLIO_UPDATE_PLAN", admitting="PORTFOLIO_UPDATE_RUN"
        ),
    )
    """The re-plan of the Task kind this owner admits, which the recovery view offers
    (V188)."""

    @property
    def manifest(self) -> ResearchWorkspaceManifest:
        """The workspace manifest, read from the one holder the Host refreshes (V182)."""
        return self._manifests.current

    def __init__(
        self,
        *,
        application: PortfolioResearchApplication,
        calibration: StrategyCalibrationApplication,
        manifest: ResearchWorkspaceManifest | ResearchWorkspaceManifestHolder,
        clock: Callable[[], datetime],
    ):
        """Wire retained portfolio, calibration, manifest and decision storage owners.

        Args:
            application: Deterministic portfolio application.
            calibration: Current input publication owner.
            manifest: Held workspace declaration.
            clock: Explicit observed-time source.
        """
        self.application, self.calibration, self.clock = application, calibration, clock
        self._manifests = held(manifest)
        self.session, self.store = application.session, application.ledger
        self.calibration.issued_score_hashes = self.issued_score_hashes

    def issued_score_hashes(self, package_id: str, component_recipe_hash: str) -> dict[date, str]:
        """Read issued score identities for one declared component recipe.

        Args:
            package_id: Declared decision-update strategy package.
            component_recipe_hash: Exact component recipe identity.

        Returns:
            Formation-session to issued score identity mapping, or an empty mapping without a
            declared update binding.
        """
        bindings = [
            v for v in self.manifest.decision_updates or () if v.strategy_package_id == package_id
        ]
        if not bindings:
            return {}
        result: dict[date, str] = {}
        for publication in self.history(bindings[0].checkpoint_hash):
            proposal = publication.pending_proposal
            if proposal is None:
                continue
            from alphalattice.investment.portfolio_strategy_lab.application.calibration import (
                PreparedPortfolioBookInput,
            )

            items = (
                proposal.input.components
                if isinstance(proposal.input, PreparedPortfolioBookInput)
                else (proposal.input,)
            )
            for item in items:
                if recipe_hashes_match(item.component_recipe_hash, component_recipe_hash):
                    result[item.formation_session] = item.score_snapshot_hash
        return result

    def valuation_obligations(self) -> dict[str, tuple[str, tuple[str, ...]]]:
        """A sealed book, its sleeves and unexpired intents own quote obligations."""
        result = {}
        for binding in self.manifest.decision_updates or ():
            history = self.history(binding.checkpoint_hash)
            latest = history[-1] if history else None
            checkpoint = self.prior_checkpoint(
                latest, self._checkpoint(binding.strategy_package_id, current=False)
            )
            book = checkpoint.initial_book if latest is None else latest.book
            weights = [book.weights, *book.sleeves]
            if latest is not None and latest.pending_proposal is not None:
                weights.extend(
                    (
                        latest.pending_proposal.estimated_weights,
                        latest.pending_proposal.reference_weights,
                    )
                )
            ids = tuple(
                v
                for i, v in enumerate(checkpoint.ordered_listing_ids)
                if any(row[i] > 0 for row in weights)
            )
            result[checkpoint.history_hash] = (
                checkpoint.content_hash if latest is None else latest.content_hash,
                ids,
            )
        return result

    def _checkpoint(self, package_id: str, *, current: bool = True) -> PortfolioDecisionCheckpoint:
        bindings = [
            v for v in self.manifest.decision_updates or () if v.strategy_package_id == package_id
        ]
        if len(bindings) != 1:
            raise ValueError("portfolio_update.not_installed")
        binding = bindings[0]
        value = self.store.load_decision_checkpoint(binding.checkpoint_hash)
        if (
            value.package.strategy_id != package_id
            or value.package.package_hash != binding.strategy_package_hash
            or not self.store.root.resolve().is_relative_to(self.session.workspace.resolve())
        ):
            raise ValueError("portfolio_update.checkpoint_binding_mismatch")
        if current:
            if manifest_fields_hash(
                read_research_workspace_manifest(self.session.workspace), PLAN_FIELDS
            ) != manifest_fields_hash(self.manifest, PLAN_FIELDS):
                raise ValueError("portfolio_update.workspace_manifest_changed")
            packages = self.application.resolver.installed_packages()
            selected = next((p for p in packages.values() if p.strategy_id == package_id), None)
            if (
                selected is None
                or value.package.package_hash != selected.package_hash
                or not self.model_bindings_match(value)
            ):
                raise ValueError("portfolio_update.package_or_model_epoch_mismatch")
            history = self.store.decision_history(value.content_hash)
            if history and history[-1].input_checkpoint_hash is not None:
                value = self.store.load_decision_checkpoint(history[-1].input_checkpoint_hash)
                if value.history_hash != binding.checkpoint_hash:
                    raise ValueError("portfolio_update.checkpoint_lineage_mismatch")
            market = MarketDataRepository(self.session.workspace)
            universe = market.current_quality_filtered_research_manifest(
                market_profile_id="us-current-index-research"
            )
            if universe is None:
                raise ValueError("portfolio_update.universe_not_admitted")
            value = transition_decision_checkpoint(
                value,
                candidate_labels={v.listing_id: v.symbol for v in universe.listings},
                source_hash=universe.revision_sha256,
            )
            self.store.publish_decision_checkpoint(value)
        return value

    def model_bindings_match(self, value: PortfolioDecisionCheckpoint) -> bool:
        """Configuration disclosure and execution require the same complete binding set."""
        scores = [
            s
            for s in self.manifest.score_inputs or ()
            if s.strategy_package_id == value.package.strategy_id
        ]
        return len(scores) == len(value.package.component_ids) and all(
            len(found := [s for s in scores if s.component_id in {None, component_id}]) == 1
            and (
                found[0].strategy_package_hash == value.package.package_hash
                or found[0].strategy_package_hash == value.package.package_hash
            )
            and found[0].authority_hash == authority_hash
            and found[0].source_kind == "WORKSPACE_DATA_FEATURE"
            for component_id, authority_hash in zip(
                value.package.component_ids, value.model_authority_hashes, strict=True
            )
        )

    def history(self, checkpoint_hash: str) -> tuple[PortfolioUpdatePublication, ...]:
        """Open the exact decision history bound to one checkpoint.

        Args:
            checkpoint_hash: Exact retained checkpoint identity.

        Returns:
            Ordered retained decision publications.
        """
        return self.store.decision_history(
            self.store.load_decision_checkpoint(checkpoint_hash).history_hash
        )

    def prior_checkpoint(
        self, previous: PortfolioUpdatePublication | None, checkpoint: PortfolioDecisionCheckpoint
    ) -> PortfolioDecisionCheckpoint:
        """Resolve the checkpoint governing the previous issued publication.

        Args:
            previous: Optional previous decision publication.
            checkpoint: Initial checkpoint when no publication exists.

        Returns:
            Exact previous input/checkpoint binding or the initial checkpoint.
        """
        return (
            checkpoint
            if previous is None
            else self.store.load_decision_checkpoint(
                previous.input_checkpoint_hash or previous.checkpoint_hash
            )
        )

    def continued_market(
        self, current: LocalQAMarketSnapshot, previous: PortfolioUpdatePublication | None
    ) -> LocalQAMarketSnapshot:
        """Continue issued observations while retaining their exact previous raw basis.

        Args:
            current: Current deterministic market snapshot.
            previous: Optional previous issued publication.

        Returns:
            Current snapshot for an initial update, otherwise owner-computed continued observations.
        """
        if previous is None:
            return current
        old = self.store.content.load_model(
            category="decision-observations",
            content_hash=previous.source_snapshot_hash,
            model=LocalQAMarketSnapshot,
            identity_field="content_hash",
        )
        raw = self.raw_basis(previous)
        return continue_issued_observations(old, current, raw_previous=raw)

    def raw_basis(self, previous: PortfolioUpdatePublication) -> LocalQAMarketSnapshot:
        """Open the exact raw market basis used by a previous publication.

        Args:
            previous: Exact retained publication.

        Returns:
            Identity-validated raw market snapshot.

        Raises:
            ValueError: A revised publication lacks its original raw basis.
        """
        plan = self._load_plan(previous.plan_hash)
        if plan.raw_market_hash is None and previous.source_revision is not None:
            raise ValueError("portfolio_update.previous_raw_basis_missing")
        return self.store.content.load_model(
            category="decision-observations",
            content_hash=plan.raw_market_hash or plan.market_hash,
            model=LocalQAMarketSnapshot,
            identity_field="content_hash",
        )

    def revision_impact(
        self, current: LocalQAMarketSnapshot, previous: PortfolioUpdatePublication | None
    ) -> PortfolioSourceRevision | None:
        """Ask the deterministic owner for source revision impact against issued raw data.

        Args:
            current: Current raw market snapshot.
            previous: Optional previous publication.

        Returns:
            Declared revision impact or None when no revision is present.
        """
        old = None if previous is None else self.raw_basis(previous)
        return source_revision_impact(old, current)

    def _market(
        self, checkpoint: PortfolioDecisionCheckpoint, through: date
    ) -> LocalQAMarketSnapshot:
        return read_local_qa_market_snapshot(
            store=MarketDataRepository(self.session.workspace),
            start=checkpoint.initial_book.schedule.formation_session,
            through=through,
            listing_ids=checkpoint.ordered_listing_ids,
            observed_at=self.clock(),
            retained_listing_ids=checkpoint.ordered_listing_ids,
        )

    def _load_plan(self, plan_hash: str) -> PortfolioUpdatePlan:
        return self.store.content.load_model(
            category="decision-plans",
            content_hash=plan_hash,
            model=PortfolioUpdatePlan,
            identity_field="plan_hash",
        )

    def plan(
        self, package_id: str, prepared_hash: str | None, through: date | None
    ) -> dict[str, object]:
        """Validate current observations and seal or reopen an exact decision update plan.

        Args:
            package_id: Declared strategy package with a decision checkpoint.
            prepared_hash: Optional exact current portfolio input.
            through: Optional observation close; defaults to admitted raw coverage end.

        Returns:
            Retained/reused plan metadata or an observations-pending action without a task.

        Raises:
            ValueError: Input/source/position/observation epoch is not admitted or required
                observations are unavailable.
        """
        checkpoint = self._checkpoint(package_id)
        source = workspace_score_source_identity(self.session.workspace)
        market = MarketDataRepository(self.session.workspace)
        manifest = market.current_quality_filtered_research_manifest(
            market_profile_id="us-current-index-research"
        )
        assert manifest is not None
        covered = market.manifest_raw_range(manifest)
        if covered is None:
            raise ValueError("portfolio_update.observations_unavailable")
        through = through or covered[1]
        history = self.history(checkpoint.content_hash)
        previous = history[-1] if history else None
        # Same admitted request/source reopens its original plan, not a new child.
        for item in reversed(history):
            prior = self._load_plan(item.plan_hash)
            if (
                prior.requested_input_hash == prepared_hash
                and prior.observed_through == through
                and prior.source_hash == source
                and prior.implementation_hash == _implementation_hash()
            ):
                return self._plan_body(prior, cached=True)
        requested = prepared_hash
        if prepared_hash is not None:
            prepared = self.calibration.published_input(prepared_hash)
            if prepared.strategy_package_hash != checkpoint.package.package_hash:
                raise ValueError("portfolio_update.input_package_mismatch")
            if any(
                v.pending_proposal is not None
                and v.pending_proposal.input.content_hash == prepared_hash
                for v in history
            ):
                prepared_hash = None
            elif prepared.formation_session != through:
                raise ValueError("portfolio_update.input_not_at_observed_close")
            else:
                self.calibration.published_input(prepared_hash, expected_source_hash=source)
                book = checkpoint.initial_book if previous is None else previous.book
                pending = None if previous is None else previous.pending_proposal
                if pending is not None and pending.schedule.entry_session > through:
                    raise ValueError("portfolio_update.previous_entry_pending")
                require_proposal_position(
                    checkpoint=checkpoint,
                    prepared=prepared,
                    position=book.next_position
                    if pending is None
                    else pending.book.next_position + 1,
                    expected_session=book.schedule.entry_session
                    if pending is None
                    else pending.schedule.entry_session,
                )
        settlement_end = max(
            [checkpoint.epoch_end]
            + (
                []
                if previous is None or previous.pending_proposal is None
                else [previous.pending_proposal.schedule.holding_end_session]
            )
            + (
                []
                if previous is None or previous.active_entry is None
                else [previous.active_entry.entry.schedule.holding_end_session]
            )
        )
        if (
            (previous is not None and through < previous.observed_through)
            or through < checkpoint.initial_book.schedule.entry_session
            or through > settlement_end
        ):
            raise ValueError("portfolio_update.observation_epoch_invalid")
        raw = self._market(checkpoint, through)
        observed = self.continued_market(raw, previous)
        if (
            prepared_hash is None
            and self.revision_impact(raw, previous) is None
            and (
                previous is None
                or not (
                    (
                        previous.pending_proposal is not None
                        and previous.pending_proposal.schedule.entry_session <= through
                    )
                    or (
                        previous.active_entry is not None
                        and previous.active_entry.entry.schedule.holding_end_session <= through
                    )
                )
            )
        ):
            return {
                "status": "OBSERVATIONS_PENDING",
                "task_id": None,
                "next_action": "PREPARE_CURRENT_INPUT_OR_UPDATE_DAILY_OBSERVATIONS",
            }
        plan = PortfolioUpdatePlan.create(
            strategy_package_id=package_id,
            workspace_manifest_hash=manifest_fields_hash(self.manifest, PLAN_FIELDS),
            catalog_hash=self.application.resolver.strategy_catalog_hash,
            checkpoint_hash=checkpoint.content_hash,
            parent_hash=None if previous is None else previous.content_hash,
            prepared_input_hash=prepared_hash,
            requested_input_hash=requested,
            observed_through=through,
            source_hash=source,
            market_hash=observed.content_hash,
            raw_market_hash=raw.content_hash if raw != observed else None,
            prior_market_hash=None if previous is None else previous.source_snapshot_hash,
            prior_raw_market_hash=None
            if previous is None
            else self.raw_basis(previous).content_hash,
            implementation_hash=_implementation_hash(),
        )
        self.store.content.publish_model(
            category="decision-plans", value=plan, identity_field="plan_hash"
        )
        return self._plan_body(plan, cached=False)

    def _plan_body(self, plan: PortfolioUpdatePlan, *, cached: bool) -> dict[str, object]:
        return {
            "status": "PLANNED",
            "update_plan_hash": plan.plan_hash,
            "strategy_package_id": plan.strategy_package_id,
            "observed_through": str(plan.observed_through),
            "cached": cached,
            "will_propose": plan.prepared_input_hash is not None,
            "work": (
                "Settle available observations; publish the sealed-input QA proposal. "
                "No fit or prediction."
            ),
            "limitation": "POST_OBSERVED_QA_NOT_TIMELY_ADVICE",
            "next_action": "PORTFOLIO_UPDATE_RUN",
            "next_requests": {
                "run": {"operation": "PORTFOLIO_UPDATE_RUN", "update_plan_hash": plan.plan_hash}
            },
        }

    def prepare(self, plan_hash: str) -> PortfolioUpdatePlan:
        """Reopen and require the current bindings of one sealed update plan.

        Args:
            plan_hash: Exact retained plan identity.

        Returns:
            Validated update plan; no task is admitted.
        """
        plan = self._load_plan(plan_hash)
        self._require(plan)
        return plan

    def _require(self, plan: PortfolioUpdatePlan) -> PortfolioDecisionCheckpoint:
        checkpoint = self._checkpoint(plan.strategy_package_id)
        if (
            plan.workspace_manifest_hash != manifest_fields_hash(self.manifest, PLAN_FIELDS)
            or plan.catalog_hash != self.application.resolver.strategy_catalog_hash
            or plan.checkpoint_hash != checkpoint.content_hash
            or not is_current(IMPLEMENTATION_ROLE, plan.implementation_hash, _implementation_hash())
        ):
            raise ValueError("portfolio_update.plan_binding_mismatch")
        if plan.prepared_input_hash is not None:
            self.calibration.published_input(
                plan.prepared_input_hash, expected_source_hash=plan.source_hash
            )
        return checkpoint

    def _publication(self, plan: PortfolioUpdatePlan) -> PortfolioUpdatePublication | None:
        value = self.store.decision_successor(
            self.store.load_decision_checkpoint(plan.checkpoint_hash).history_hash, plan.parent_hash
        )
        if value is not None and (
            value.plan_hash != plan.plan_hash
            or value.observed_through != plan.observed_through
            or value.source_snapshot_hash != plan.market_hash
            or (
                plan.prepared_input_hash is not None
                and (
                    value.pending_proposal is None
                    or value.pending_proposal.input.content_hash != plan.prepared_input_hash
                )
            )
        ):
            raise ValueError("portfolio_update.publication_binding_mismatch")
        return value

    def reusable(self, plan: PortfolioUpdatePlan) -> PortfolioUpdatePublication | None:
        """Require a succeeded exact update task and complete publication before reuse.

        Args:
            plan: Exact sealed update plan.

        Returns:
            Verified retained publication or None without a succeeded matching task.

        Raises:
            ValueError: Exact observations or the published report are missing or invalid.
        """
        self._require(plan)
        value = self._publication(plan)
        if value is None:
            return None
        for task in self.session.task_control_registry.tasks():
            if (
                task.task_kind == TASK_KIND
                and task.lifecycle is TaskLifecycle.SUCCEEDED
                and self._plan_of(task, current=False) == plan
            ):
                self._observed(plan)
                if value.html_hash is None:
                    raise ValueError("portfolio_update.report_missing")
                self.store.load_html(value.html_hash)
                return value
        return None

    def admit(self, plan: PortfolioUpdatePlan) -> CommandAdmission:
        """Reuse an exact admission or admit a fresh update with exclusive update ownership.

        Args:
            plan: Explicit sealed and current update plan.

        Returns:
            Retained or newly admitted task identity/lifecycle.

        Raises:
            ValueError: Another update is in progress or the plan is stale or inadmissible.
        """
        self._require(plan)
        envelope, goal, workflow = _task_contract(plan)
        registry = self.session.task_control_registry
        for task in registry.tasks():
            if task.input == envelope and task.lifecycle is not TaskLifecycle.CANCELLED:
                return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)
            if task.task_kind == TASK_KIND and task.lifecycle in (
                TaskLifecycle.QUEUED,
                TaskLifecycle.RUNNING,
                TaskLifecycle.RECOVERY_REQUIRED,
            ):
                raise ValueError("portfolio_update.update_in_progress")
        self._require_fresh(plan)
        task = registry.admit(
            input_envelope=envelope, goal=goal, plan=workflow, observed_at=self.clock()
        ).record
        return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)

    def _require_fresh(self, plan: PortfolioUpdatePlan) -> None:
        history = self.history(plan.checkpoint_hash)
        if (history[-1].content_hash if history else None) != plan.parent_hash:
            raise ValueError("portfolio_update.parent_advanced")
        if workspace_score_source_identity(self.session.workspace) != plan.source_hash:
            raise ValueError("portfolio_update.source_changed_before_seal")

    def _plan_of(self, task: TaskRecord, *, current: bool = True) -> PortfolioUpdatePlan:
        if task.task_kind != TASK_KIND:
            raise ValueError("portfolio_update.task_kind_invalid")
        value = PortfolioUpdatePlan.model_validate(task.input.payload["plan"])
        if (task.input, task.goal, task.plan) != _task_contract(value):
            raise ValueError("portfolio_update.task_contract_invalid")
        if current:
            self._require(value)
        return cast(PortfolioUpdatePlan, value)

    def replan_request(self, task: TaskRecord) -> dict[str, object]:
        """Replan the validated Task's selected package, input and observation boundary.

        Args:
            task: Durable Task whose declaration supplies the next request.

        Returns:
            The filled Portfolio-update planning request, retaining the requested input.

        Raises:
            ValueError: The Task's kind, sealed plan or workflow does not validate.
        """
        plan = self._plan_of(task, current=False)
        return {
            "operation": "PORTFOLIO_UPDATE_PLAN",
            "strategy_package_id": plan.strategy_package_id,
            "prepared_input_hash": plan.requested_input_hash,
            "observed_through": plan.observed_through.isoformat(),
        }

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute the exact retained update task through its writer session.

        Args:
            task_id: Exact admitted task identity.
            expected_task_hash: Optional optimistic task identity.
        """
        task = self.session.task_control_registry.task(task_id)
        self._plan_of(task)
        self.session.execute_admitted(task, self, self.clock, expected_task_hash)

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Bind an update task to its schema, workflow, checkpoint and execution identity.

        Args:
            task: Exact task whose update plan is validated.

        Returns:
            Deterministic execution compatibility contract.
        """
        plan = self._plan_of(task)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash(schema_structure(PortfolioUpdatePlan)),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=plan.checkpoint_hash,
            framework_identity_hash=self.session.execution_identity(plan.implementation_hash),
        )

    @staticmethod
    def _evidence(stage: str, content: str) -> tuple[TaskEvidence, ...]:
        return (
            TaskEvidence(
                evidence_kind="portfolio_update.stage",
                reference=f"playpen://portfolio-update/{stage}/{content}",
                content_hash=content,
            ),
        )

    def execute_stage(
        self, *, task: TaskRecord, execution: TaskExecution, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        """Validate, publish observations or advance and publish one admitted decision stage.

        Args:
            task: Exact admitted update task.
            execution: Current task execution declaration.
            work_item: Exact workflow stage.

        Returns:
            Ready stage result with exact plan, observation or publication evidence.

        Raises:
            ValueError: Stage is unknown, parent advanced, input is stale or exact snapshot binding
                differs.
        """
        plan, stage = self._plan_of(task), work_item.stage_id
        if stage == STAGES[0]:
            self._require_fresh(plan)
            content = plan.plan_hash
        elif stage == STAGES[1]:
            self._require_fresh(plan)
            raw = self._market(self._checkpoint(plan.strategy_package_id), plan.observed_through)
            history = self.history(plan.checkpoint_hash)
            previous = history[-1] if history else None
            observed = self.continued_market(raw, previous)
            if plan.raw_market_hash is not None:
                if raw.content_hash != plan.raw_market_hash:
                    raise ValueError("portfolio_update.market_snapshot_changed")
                self.store.content.publish_model(
                    category="decision-observations", value=raw, identity_field="content_hash"
                )
            if observed.content_hash != plan.market_hash:
                raise ValueError("portfolio_update.market_snapshot_changed")
            self.store.content.publish_model(
                category="decision-observations", value=observed, identity_field="content_hash"
            )
            content = observed.content_hash
        elif stage == STAGES[2]:
            value = self._publication(plan)
            if value is None:
                history = self.history(plan.checkpoint_hash)
                previous = history[-1] if history else None
                if (None if previous is None else previous.content_hash) != plan.parent_hash:
                    raise ValueError("portfolio_update.parent_advanced")
                observed = self._observed(plan)
                prepared = (
                    None
                    if plan.prepared_input_hash is None
                    else self.calibration.published_input(plan.prepared_input_hash)
                )
                value = advance_decision_state(
                    checkpoint=self._checkpoint(plan.strategy_package_id),
                    previous=previous,
                    prepared=prepared,
                    observed=observed,
                    plan_hash=plan.plan_hash,
                    published_at=self.clock(),
                    previous_checkpoint=self.prior_checkpoint(
                        previous, self._checkpoint(plan.strategy_package_id)
                    ),
                    revised_source_hash=plan.raw_market_hash,
                    source_revision=self.revision_impact(
                        self.store.content.load_model(
                            category="decision-observations",
                            content_hash=plan.raw_market_hash or plan.market_hash,
                            model=LocalQAMarketSnapshot,
                            identity_field="content_hash",
                        ),
                        previous,
                    ),
                )
                html_hash, _ = self.store.publish_html(
                    render_decision_update(self._checkpoint(plan.strategy_package_id), (value,))
                )
                values = {
                    name: getattr(value, name)
                    for name in type(value).model_fields
                    if name != "content_hash"
                }
                values["html_hash"] = html_hash
                value = PortfolioUpdatePublication.create(**values)
                self.store.publish_decision_update(value)
            from alphalattice.control.product_host.composition.rolling_portfolio_report import (
                publish_rolling_report_heads,
            )

            checkpoint = self._checkpoint(plan.strategy_package_id)
            publish_rolling_report_heads(
                application=self.application,
                checkpoint=checkpoint,
                publications=self.history(checkpoint.history_hash),
            )
            content = value.content_hash
        else:
            raise ValueError("portfolio_update.stage_unknown")
        return StageExecutionResult(StageDisposition.READY, evidence=self._evidence(stage, content))

    def _observed(self, plan: PortfolioUpdatePlan) -> LocalQAMarketSnapshot:
        value = self.store.content.load_model(
            category="decision-observations",
            content_hash=plan.market_hash,
            model=LocalQAMarketSnapshot,
            identity_field="content_hash",
        )
        if value.through != plan.observed_through:
            raise ValueError("portfolio_update.observation_binding_invalid")
        if plan.raw_market_hash is not None:
            raw = self.store.content.load_model(
                category="decision-observations",
                content_hash=plan.raw_market_hash,
                model=LocalQAMarketSnapshot,
                identity_field="content_hash",
            )
            if plan.prior_market_hash is None:
                raise ValueError("portfolio_update.continuation_parent_missing")
            old = self.store.content.load_model(
                category="decision-observations",
                content_hash=plan.prior_market_hash,
                model=LocalQAMarketSnapshot,
                identity_field="content_hash",
            )
            factual = (
                None
                if plan.prior_raw_market_hash is None
                else self.store.content.load_model(
                    category="decision-observations",
                    content_hash=plan.prior_raw_market_hash,
                    model=LocalQAMarketSnapshot,
                    identity_field="content_hash",
                )
            )
            if continue_issued_observations(old, raw, raw_previous=factual) != value:
                raise ValueError("portfolio_update.continuation_evidence_invalid")
        return value

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Reopen exact stage artifacts and require the declared evidence tuple.

        Args:
            task: Exact update task.
            execution: Current execution declaration.
            work_item: Exact workflow stage.
            evidence: Evidence tuple supplied by Task Control.

        Returns:
            Unchanged evidence after identity, report and stage checks.

        Raises:
            ValueError: Stage, observation, publication/report or evidence is invalid.
        """
        plan, stage = self._plan_of(task), work_item.stage_id
        if stage == STAGES[0]:
            content = plan.plan_hash
        elif stage == STAGES[1]:
            content = self._observed(plan).content_hash
        elif stage == STAGES[2]:
            self._observed(plan)
            publication = self._publication(plan)
            if publication is None:
                raise ValueError("portfolio_update.publication_missing")
            if publication.html_hash is None:
                raise ValueError("portfolio_update.report_missing")
            self.store.load_html(publication.html_hash)
            content = publication.content_hash
        else:
            raise ValueError("portfolio_update.stage_unknown")
        if evidence != self._evidence(stage, content):
            raise ValueError("portfolio_update.evidence_invalid")
        return evidence

    def readback(
        self, task_id: UUID | None = None, *, publication_hash: str | None = None
    ) -> dict[str, object]:
        """Read an update task and its exact current or previous issued publication.

        Args:
            task_id: Optional exact task; otherwise read the latest retained update task.
            publication_hash: Optional publication required to be bound to that task.

        Returns:
            Task state or verified publication/history/report projection.

        Raises:
            ValueError: Requested publication is not task-bound or required publication/observations
                are absent.
        """
        registry = self.session.task_control_registry
        task = (
            registry.task(task_id)
            if task_id
            else next((v for v in reversed(registry.tasks()) if v.task_kind == TASK_KIND), None)
        )
        if task is None:
            return {"status": "NO_PORTFOLIO_UPDATE", "task_id": None}
        plan = self._plan_of(task, current=False)
        completed = task.lifecycle is TaskLifecycle.SUCCEEDED
        if not completed and plan.parent_hash is None and publication_hash is None:
            return {"status": task.lifecycle.value, "task_id": str(task.task_id)}
        history = self.history(plan.checkpoint_hash)
        value = (
            self._publication(plan)
            if completed and (publication_hash is None or publication_hash != plan.parent_hash)
            else next((v for v in history if v.content_hash == plan.parent_hash), None)
        )
        if publication_hash is not None and (
            value is None or value.content_hash != publication_hash
        ):
            raise ValueError("portfolio_update.publication_not_bound_to_task")
        if value is None:
            raise ValueError("portfolio_update.publication_missing")
        self._observed(
            plan if value.plan_hash == plan.plan_hash else self._load_plan(value.plan_hash)
        )
        checkpoint = self.store.load_decision_checkpoint(
            value.input_checkpoint_hash or value.checkpoint_hash
        )
        prefix = history[
            : next(i for i, v in enumerate(history) if v.content_hash == value.content_hash) + 1
        ]
        return self.publication_body(
            checkpoint,
            value,
            prefix,
            task_id=task.task_id,
            lifecycle=None if completed else task.lifecycle.value,
        )

    def publication_task(self, value: PortfolioUpdatePublication) -> UUID | None:
        """Resolve the unique succeeded task that issued a publication plan.

        Args:
            value: Exact retained decision publication.

        Returns:
            Succeeded task identity or None without a match.

        Raises:
            ValueError: Multiple succeeded tasks claim the same publication plan.
        """
        matches = [
            t.task_id
            for t in self.session.task_control_registry.tasks()
            if t.task_kind == TASK_KIND
            and t.lifecycle is TaskLifecycle.SUCCEEDED
            and self._plan_of(t, current=False).plan_hash == value.plan_hash
        ]
        if len(matches) > 1:
            raise ValueError("portfolio_update.publication_task_ambiguous")
        return matches[0] if matches else None

    def publication_body(
        self,
        checkpoint: PortfolioDecisionCheckpoint,
        value: PortfolioUpdatePublication,
        prefix: tuple[PortfolioUpdatePublication, ...],
        *,
        task_id: UUID,
        lifecycle: str | None = None,
    ) -> dict[str, object]:
        """Project an issued publication with its verified history and optional report.

        Args:
            checkpoint: Exact publication checkpoint.
            value: Publication selected for readback.
            prefix: Ordered history through this publication.
            task_id: Exact associated task identity.
            lifecycle: Optional unfinished task state when reading its previous publication.

        Returns:
            Publication status, task association, listing labels, history and retained HTML.
        """
        body = value.model_dump(mode="json")
        return {
            "status": lifecycle
            if lifecycle is not None
            else (
                "PROPOSAL_PUBLISHED"
                if value.pending_proposal
                else (value.events[-1].phase if value.events else "ENTRY_SETTLED")
            ),
            "task_id": str(task_id),
            "publication_is_previous": lifecycle is not None,
            "strategy_package_id": checkpoint.package.strategy_id,
            "publication": body,
            "listing_labels": dict(
                zip(checkpoint.ordered_listing_ids, checkpoint.listing_labels, strict=True)
            ),
            "history": [v.model_dump(mode="json") for v in prefix],
            "realized_performance": _realized_performance(checkpoint, value, prefix),
            "html": self.store.load_html(value.html_hash) if value.html_hash else None,
        }


def _realized_performance(
    checkpoint: PortfolioDecisionCheckpoint,
    value: PortfolioUpdatePublication,
    prefix: tuple[PortfolioUpdatePublication, ...],
) -> dict[str, object]:
    """Project only sealed outcomes on one contiguous one-session return axis."""
    if (
        not prefix
        or prefix[-1].content_hash != value.content_hash
        or any(v.checkpoint_hash != checkpoint.history_hash for v in prefix)
        or any(
            v.parent_hash != (None if i == 0 else prefix[i - 1].content_hash)
            for i, v in enumerate(prefix)
        )
    ):
        raise ValueError("portfolio_update.performance_history_binding_invalid")
    outcomes = [
        (publication, event)
        for publication in prefix
        for event in publication.events
        if event.phase == "OUTCOME_SETTLED"
    ]
    positions = {day: i for i, day in enumerate(checkpoint.formation_sessions)}
    daily = all(
        event.entry.schedule.actual_session_span == 2
        and positions.get(event.entry.schedule.entry_session)
        == positions.get(event.formation_session, -3) + 1
        and positions.get(event.entry.schedule.holding_end_session)
        == positions.get(event.formation_session, -3) + 2
        and event.formation_session == event.entry.schedule.formation_session
        and event.entry.schedule.holding_end_session <= publication.observed_through
        for publication, event in outcomes
    ) and all(
        previous.entry.schedule.holding_end_session == following.entry.schedule.entry_session
        for (_, previous), (_, following) in pairwise(outcomes)
    )
    window = {
        "selected_start": outcomes[0][1].entry.schedule.entry_session.isoformat()
        if outcomes
        else None,
        "selected_end": outcomes[-1][1].entry.schedule.holding_end_session.isoformat()
        if outcomes
        else None,
        "observation_count": len(outcomes),
        "return_axis": "CONTIGUOUS_ONE_SESSION_ENTRY_OPEN_TO_HOLDING_END_OPEN"
        if daily
        else "NOT_A_CONTIGUOUS_ONE_SESSION_AXIS",
    }
    source: dict[str, object] = {
        **window,
        "strategy_package_id": checkpoint.package.strategy_id,
        "strategy_package_hash": checkpoint.package.package_hash,
        "checkpoint_hash": checkpoint.history_hash,
        "source_book_task_id": str(checkpoint.book_task_id) if checkpoint.book_task_id else None,
        "publication_hash": value.content_hash,
        "history_prefix_hash": canonical_hash(tuple(v.content_hash for v in prefix)),
        "settlement_path_hash": canonical_hash(tuple(event.content_hash for _, event in outcomes)),
        "source_snapshot_hash": value.source_snapshot_hash,
        "observed_through": value.observed_through.isoformat(),
        "published_at": value.published_at.isoformat(),
        "claim": value.claim,
        "execution_basis": "DAILY_BAR_QA_NOT_VERIFIED_VENUE_EXECUTION",
        "return_unit": "FRACTION",
        "annualization_sessions_per_year": 252 if daily and outcomes else None,
        "volatility_degrees_of_freedom": 1,
        "sharpe_cash_return_per_session": 0.0,
        "sortino_downside_threshold": 0.0,
        "window_hash": canonical_hash(window),
    }
    fields = (
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
        "sharpe",
        "sortino",
    )
    absence_details = {
        "NONCONTIGUOUS_OR_NONDAILY_REALIZED_RETURN_AXIS": "The published outcomes do not "
        "form consecutive one-session returns, so daily annualized metrics are unavailable.",
        "INVALID_SEALED_NET_RETURN_PATH": "The sealed outcome path contains an invalid return; "
        "no forward performance value is shown.",
        "INSUFFICIENT_REALIZED_OBSERVATIONS": "At least two settled one-session outcomes are "
        "needed for these metrics. Published proposals are not realized returns.",
        "NONFINITE_DERIVED_METRIC": "The shared return owner could not produce a finite value "
        "for this realized window.",
        "ZERO_DOWNSIDE_DEVIATION": "This realized window has no downside deviation against "
        "the owner's zero threshold, so Sortino is unavailable.",
    }
    lanes: dict[str, dict[str, object]] = {}
    for quote, field in (("5", "net_return_5bps"), ("10", "net_return_10bps")):
        cost = PortfolioPerSideCostAssumption.from_bps_per_side(quote)
        returns = np.asarray([getattr(event, field) for _, event in outcomes], dtype=np.float64)
        valid = bool(np.isfinite(returns).all() and np.all(returns > -1.0))
        reason = (
            "NONCONTIGUOUS_OR_NONDAILY_REALIZED_RETURN_AXIS"
            if not daily
            else "INVALID_SEALED_NET_RETURN_PATH"
            if not valid
            else "INSUFFICIENT_REALIZED_OBSERVATIONS"
            if len(outcomes) < 2
            else None
        )
        metrics: dict[str, object] = {"cost_bps": cost.platform_cost_bps}
        absences: dict[str, object] = (
            {name: {"reason": reason} for name in fields} if reason else {}
        )
        if reason is None:
            try:
                measured = evaluate_net_simple_return_path(net_simple_returns=returns)
            except (ValueError, OverflowError):
                reason = "NONFINITE_DERIVED_METRIC"
                absences = {name: {"reason": reason} for name in fields}
            else:
                for name in fields:
                    metric = getattr(measured, name)
                    if math.isfinite(metric):
                        metrics[name] = metric
                    else:
                        absences[name] = {
                            "reason": "ZERO_DOWNSIDE_DEVIATION"
                            if name == "sortino" and bool(np.all(returns >= 0))
                            else "NONFINITE_DERIVED_METRIC"
                        }
        elif daily and valid and len(outcomes) == 1:
            metrics["cumulative_return"] = float(returns[0])
            del absences["cumulative_return"]
        lanes[quote] = {
            "cost_bps_per_side": quote,
            "selected_window_metrics": metrics,
            "selected_window_metric_absences": {
                name: {
                    **cast(dict[str, object], absent),
                    "detail": absence_details[str(cast(dict[str, object], absent)["reason"])],
                }
                for name, absent in absences.items()
            },
            "selected_window_metric_provenance": {
                **source,
                "status": reason or "DERIVED_FROM_SEALED_REALIZED_NET_RETURN_PATH",
                "net_return_field": field,
                "cost_assumption_hash": cost.assumption_hash,
                "cost_bps_per_side": quote,
                "platform_one_way_cost_bps": cost.platform_cost_bps,
            },
            "series": [
                {
                    "session": event.entry.schedule.holding_end_session.isoformat(),
                    "formation_session": event.formation_session.isoformat(),
                    "entry_session": event.entry.schedule.entry_session.isoformat(),
                    "holding_end_session": event.entry.schedule.holding_end_session.isoformat(),
                    "net_simple_return": getattr(event, field),
                }
                for _, event in outcomes
                if daily and valid
            ],
        }
    return {**source, "cost_lanes": lanes}


@dataclass
class PortfolioUpdateCommand:
    """Dispatch one explicit decision update plan through its deterministic application."""

    application: PortfolioUpdateApplication
    plan: PortfolioUpdatePlan | None = None
    command_kind: str = TASK_KIND

    def admit(self) -> CommandAdmission:
        """Require an explicit update plan before admitting its command.

        Returns:
            Deterministic task admission.

        Raises:
            ValueError: No update plan is supplied.
        """
        if self.plan is None:
            raise ValueError("portfolio_update.plan_required")
        return self.application.admit(self.plan)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one exact admitted update command.

        Args:
            task_id: Exact task to execute.
            expected_task_hash: Optional optimistic task identity.
        """
        self.application.execute(task_id, expected_task_hash=expected_task_hash)
