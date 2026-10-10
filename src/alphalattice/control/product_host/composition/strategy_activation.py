"""A person's activation of an installed research strategy's reviewed book (LS1, V459; OW12).

The daily chain, a research update bringing data, scores, calibration and the book's next
positions to the latest completed session, reads three bindings for a package: its score
inputs, its calibration seed and its decision checkpoint. An installed research strategy
(`NON_DEFAULT_RESEARCH`) has none until a person activates one of its books, so its books
replay history and never give their next positions (V459).

An activation builds the three from the strategy's own research and binds them in one
manifest write, with no fit, no replay and no book built from final weights:

- each component's models: a lifecycle grant made from the training admission its Alpha study
  read, with the same observations and prepared refits, so the study's fitted children move
  into the workspace by their keys; it renews on the workspace's data through the horizon;
- the calibration seed: the installed authority's own lanes (the calibrated component's
  scores, the realized returns and the decision eligibility on the book's formations),
  anchored at the book's activation formation;
- the book's opening state: the reviewed run's sealed final boundary, as a checkpoint whose
  purpose says a person activated it forward, its formations the book's and then the
  installed calendar's through the horizon.

A book whose package moved since it ran is refused: running it again and activating that run
is the re-bind (V458). A deactivation removes the three bindings; the history stays readable.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final
from uuid import UUID

import numpy as np
import numpy.typing as npt

from alphalattice.control.data_platform.readiness import _latest_common_us_session
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceCalibrationInput,
    ResearchWorkspaceDecisionUpdate,
    ResearchWorkspaceManifest,
    ResearchWorkspaceScoreInput,
    component_training_selection,
    update_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.result_standing import Activation, ResultStanding
from alphalattice.control.product_host.research_authoring.factor_inputs import confined
from alphalattice.control.product_host.storage.retention import require_no_pending_cleanup
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.task_control.registry import TaskNotFoundError
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.causal_outcomes.execution.contracts import CausalExecutionSchedulePoint
from alphalattice.foundation.causal_outcomes.execution.readers import planned_local_qa_schedule
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from alphalattice.investment.alpha_research.scores.model_renewal import (
    AdmittedRenewingInference,
    AlphaModelLifecycleAdmission,
    AlphaModelSetPublication,
    AlphaTrainingObservations,
    admit_component_inference,
    read_lifecycle_admission,
    reuse_refit_child,
    verified_lifecycle_admissions,
)
from alphalattice.investment.portfolio_strategy_lab.application.calibration import (
    CalibrationObservations,
    FrozenRankCalibrationRule,
    publish_observations,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioExecutionLedger,
    SealedPortfolioBoundaryState,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioDecisionCheckpoint,
    PortfolioEntryBook,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    FrozenStrategyPackage,
)
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    PORTFOLIO_PUBLIC_TASK_KIND,
    portfolio_research_task_input,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    FROZEN_RESEARCH_BOOK_RECIPES,
)
from alphalattice.investment.portfolio_strategy_lab.policies.lifecycle_research import (
    ARTIFACT_KEY,
    LifecyclePortfolioAuthority,
    LifecycleResearchScoreSource,
)
from alphalattice.investment.portfolio_strategy_lab.policies.post_observed_authority import (
    POST_OBSERVED_AUTHORITY_MANIFEST_NAME,
    FrozenHistoricalBookRecipe,
    PostObservedStrategyAuthorityManifest,
)
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioResearchArtifactStore,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PortfolioLedgerStore,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.static import format_book_weight
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from alphalattice.kernel.data.errors import DataQualityError
from alphalattice.kernel.shared_kernel.identity import canonical_hash

FORWARD_HORIZON: Final = timedelta(days=330)
"""How far past the latest completed session an activation runs its strategy: the installed
calendar plans a year past the clock, and a schedule reads two weeks past its last formation.
Past it the research update refuses the epoch, and a person activates a newer book."""
REVIEW_HOLDINGS_WORDS: Final = (
    "These are the book's last sealed holdings and the sessions they were decided and entered. "
    "Its review standing is stated separately. Activation is reversible, and "
    "deactivating keeps its history. The first forward update after activation publishes the "
    "positions for the first actionable session."
)
"""What a person reads before activating, in place of a preview the product does not compute."""

_ADMISSIONS: Final = "artifacts/alpha-research/current/lifecycle-admissions"


@dataclass(frozen=True, slots=True)
class _Book:
    """The reviewed book an activation continues, with what it was run from."""

    task_id: UUID
    package: FrozenStrategyPackage
    recipe: FrozenHistoricalBookRecipe
    source: LifecycleResearchScoreSource
    execution: PortfolioExecutionLedger
    boundary: SealedPortfolioBoundaryState


def _bound(manifest: ResearchWorkspaceManifest, package_id: str) -> bool:
    return any(
        value.strategy_package_id == package_id
        for values in (
            manifest.score_inputs,
            manifest.calibration_inputs,
            manifest.decision_updates,
        )
        for value in values or ()
    )


def _without(
    manifest: ResearchWorkspaceManifest, package_id: str
) -> dict[str, tuple[object, ...] | None]:
    """The manifest's live bindings with one package's left out (None where none remain)."""
    return {
        name: tuple(v for v in getattr(manifest, name) or () if v.strategy_package_id != package_id)
        or None
        for name in ("score_inputs", "calibration_inputs", "decision_updates")
    }


def admit_decision_checkpoint(
    workspace: Path,
    checkpoint: PortfolioDecisionCheckpoint,
    score_inputs: tuple[ResearchWorkspaceScoreInput, ...],
) -> None:
    """Prove a decision checkpoint against the score inputs and the listings it will be bound with.

    One rule for every binder (OW10): a person's activation and the operator's QA admission
    (`scripts/bind_portfolio_decision_checkpoint.py`).

    Args:
        workspace: The workspace whose manifest will bind the checkpoint.
        checkpoint: The checkpoint to bind.
        score_inputs: The package's score inputs it will be bound beside.

    Raises:
        ValueError: `portfolio_update.workspace_scoring_required` when a component has no one
            workspace-data score input; `portfolio_update.checkpoint_model_epoch_mismatch` when
            its models do not admit the checkpoint's epoch;
            `portfolio_update.checkpoint_listing_authority_mismatch` when its listings are not
            the research universe's.
    """
    root = workspace.resolve()
    package_id = checkpoint.package.strategy_id
    store = AlphaCurrentArtifactStore(root / "artifacts")
    for component_id, source_hash, recipe_hash in zip(
        checkpoint.package.component_ids,
        checkpoint.model_authority_hashes,
        checkpoint.model_recipe_hashes,
        strict=True,
    ):
        choices = [
            v
            for v in score_inputs
            if v.strategy_package_id == package_id and v.component_id in {None, component_id}
        ]
        if len(choices) != 1 or choices[0].source_kind != "WORKSPACE_DATA_FEATURE":
            raise ValueError("portfolio_update.workspace_scoring_required")
        binding = choices[0]
        with verified_lifecycle_admissions():
            authority = admit_component_inference(
                root / binding.authority_relative_path,
                expected_hash=binding.authority_hash,
                store=store,
            )
        if (
            source_hash != binding.authority_hash
            or recipe_hash != authority.authority.model_set.recipe_hash
            or any(
                not (
                    authority.authority.supports(day)
                    or (
                        isinstance(authority, AdmittedRenewingInference)
                        and authority.authority.can_prepare(day)
                    )
                )
                for day in (checkpoint.epoch_start, checkpoint.epoch_end)
            )
        ):
            raise ValueError("portfolio_update.checkpoint_model_epoch_mismatch")
    market = MarketDataRepository(root)
    universe = market.current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    if universe is None:
        raise ValueError("portfolio_update.checkpoint_listing_authority_mismatch")
    history = market.membership_schedule(
        universe.profile.market_profile_id,
        sessions=checkpoint.formation_sessions[: checkpoint.initial_book.next_position],
        fallback_listing_ids=tuple(v.listing_id for v in universe.listings),
    )
    axis = set(checkpoint.ordered_listing_ids)
    # A sealed prior book, not the current candidate roster: it covers its own formation's
    # members; former holdings stay legal, a future-only entrant holds no historical position.
    held = {
        listing
        for i, listing in enumerate(checkpoint.ordered_listing_ids)
        if any(
            row[i] > 0
            for row in (checkpoint.initial_book.weights, *checkpoint.initial_book.sleeves)
        )
    }
    if (
        not set(history.members(checkpoint.initial_book.schedule.formation_session)) <= axis
        or not axis <= set(market.research_listing_sources(universe))
        or not held <= set(history.union)
    ):
        raise ValueError("portfolio_update.checkpoint_listing_authority_mismatch")
    names = market.listing_scope(universe, listing_ids=checkpoint.ordered_listing_ids)
    if (
        tuple(v.listing_id for v in names) != checkpoint.ordered_listing_ids
        or tuple(v.symbol for v in names) != checkpoint.listing_labels
    ):
        raise ValueError("portfolio_update.checkpoint_listing_authority_mismatch")


RUN_FORWARD_WORDS: Final = {
    False: "A book that runs forward to next positions comes from the research strategy: its "
    "controls name the required Alpha and Risk studies over their whole support, which need no "
    "Factor study; then prepare and install it, run its whole-support book and review that "
    "book. A Lab book is research only and is never activated.",
    True: "A book that runs forward to next positions comes from the research strategy: its "
    "controls name the required Alpha and Risk studies over their whole support, which need no "
    "Factor study; then prepare and install it and run its whole-support book, the first use's "
    "numerical check. Activation then makes the first use's date's update, and those positions "
    "are reviewed. A Lab book is research only and is never activated.",
}
"""The first use's shortest way, ahead of the inputs' Lab flows, by whether a first
use is open: its book is then the numerical check, and its date's positions are reviewed."""
INSTALLED_BOOK_WORDS: Final = {
    False: "The installed strategy runs its whole-support historical book from its controls; "
    "review that book, then read its exact activation offer.",
    True: "The installed strategy runs its whole-support historical book from its controls, "
    "the first use's numerical check; then read its activation offer, which makes the first "
    "use's date's update in the same act.",
}


class StrategyActivation:
    """Own a person's activation and deactivation of an installed research strategy's book."""

    def __init__(
        self,
        *,
        session: WorkspaceApplicationSession,
        manifest: Callable[[], ResearchWorkspaceManifest],
        packages: Callable[[], dict[str, FrozenStrategyPackage]],
        ledger: PortfolioLedgerStore,
        read_experiment: Callable[[UUID], dict[str, object]],
        clock: Callable[[], datetime],
        hold: Callable[[ResearchWorkspaceManifest], None],
        read_information: Callable[[UUID], dict[str, object]] | None = None,
        read_review: Callable[[UUID], dict[str, object]] | None = None,
    ) -> None:
        """Wire the workspace session, the installed packages, the Portfolio ledger and the Host.

        Args:
            session: The Host's workspace session (its workspace, Task registry and gate).
            manifest: The one manifest the Host's applications read (V182); the write reads
                the file under the gate and refuses one that differs from it.
            packages: The installed packages by strategy id.
            ledger: The Portfolio ledger the books and the decision checkpoints live in.
            read_experiment: A completed study's readback, its document included.
            clock: The observed time.
            hold: The Host's refresh of the one manifest every application reads (V182).
            read_information: Metadata-only summary locating the sealed study's date sources.
            read_review: The review owner's standing for the exact book Task's published result.
        """
        self.session, self.manifest = session, manifest
        self.packages, self.ledger, self.read_experiment = packages, ledger, read_experiment
        self.clock, self.hold = clock, hold
        self.read_information = read_information or read_experiment
        self.read_review = read_review
        self._in_flight: dict[str, tuple[str, float]] = {}
        """Each activation running now: its package's phase and when that phase began."""

    @property
    def workspace(self) -> Path:
        """The workspace the activation writes."""
        return Path(self.session.workspace)

    @property
    def gate(self) -> WorkspaceMutationGate:
        """The workspace's single-writer gate."""
        return self.session.mutation_gate

    def book_result_hash(self, task_id: UUID) -> str | None:
        """Read the book Task's exact ledger result, including a retained pre-pipeline book."""
        try:
            task = self.session.task_control_registry.task(task_id)
        except TaskNotFoundError:
            return None
        if task.task_kind != PORTFOLIO_PUBLIC_TASK_KIND:
            return None
        durable = portfolio_research_task_input(task)
        result = self.ledger.find_result_for(
            program_hash=durable.program.program_hash, spec_hash=durable.spec.spec_hash
        )
        return None if result is None else result.result_hash

    def _book(self, task_id: UUID, manifest: ResearchWorkspaceManifest) -> _Book:
        try:
            task = self.session.task_control_registry.task(task_id)
        except TaskNotFoundError as error:
            raise ValueError("strategy_activation.book_task_absent") from error
        if task.task_kind != PORTFOLIO_PUBLIC_TASK_KIND:
            raise ValueError("strategy_activation.book_task_required")
        if task.lifecycle is not TaskLifecycle.SUCCEEDED:
            raise ValueError("strategy_activation.completed_book_required")
        durable = portfolio_research_task_input(task)
        if manifest.strategy_installation != "NON_DEFAULT_RESEARCH":
            raise ValueError("strategy_activation.research_strategy_required")
        package = self.packages().get(durable.selected_strategy_package_id)
        recipe = next(
            (
                value
                for value in FROZEN_RESEARCH_BOOK_RECIPES
                if value.strategy_id == durable.selected_strategy_package_id
            ),
            None,
        )
        if package is None or recipe is None:
            raise ValueError("strategy_activation.research_book_required")
        if (
            package.package_hash != durable.selected_strategy_package_hash
            or durable.selected_score_source_mode != "HISTORICAL_ARRAY_REPLAY"
        ):
            raise ValueError("strategy_activation.book_package_moved")
        (artifact,) = manifest.strategy_artifacts
        source = LifecycleResearchScoreSource(
            manifest_path=confined(self.workspace, artifact.relative_path), recipe=recipe
        )
        result_hash = self.book_result_hash(task_id)
        if result_hash is None:
            raise ValueError("strategy_activation.book_result_absent")
        result = self.ledger.load_result(result_hash)
        execution = self.ledger.load_execution(result.execution_ledger_hash)
        boundary = execution.final_boundary
        if boundary is None or boundary.sleeve_weights_hash is None:
            raise ValueError("strategy_activation.book_state_unsealed")
        sessions = source.formation_sessions
        if (
            execution.formation_sessions != sessions
            or boundary.decided_formation_count != len(sessions)
            or execution.ordered_listing_ids != source.ordered_listing_ids
        ):
            # The book continues from its strategy's last formation, decided from its first.
            raise ValueError(f"strategy_activation.book_not_whole:{sessions[0]}..{sessions[-1]}")
        if len(sessions) <= recipe.sizing_activation_formation:
            raise ValueError("strategy_activation.book_shorter_than_its_activation")
        return _Book(task_id, package, recipe, source, execution, boundary)

    def _activation_book(self, task_id: UUID, manifest: ResearchWorkspaceManifest) -> _Book:
        """The existing book and already-active admission, shared by every reader (OW10)."""
        book = self._book(task_id, manifest)
        if any(
            value.strategy_package_id == book.package.strategy_id
            and value.strategy_package_hash == book.package.package_hash
            for value in manifest.decision_updates or ()
        ):
            raise ValueError("strategy_activation.already_active")
        return book

    def standing(self, task_id: UUID | None, *, result_hash: str | None = None) -> ResultStanding:
        """A verified installed result's marks, with activation from its existing owner rule."""
        if task_id is None and result_hash is not None:
            # Retained pre-pipeline books still have their durable Task and exact ledger result.
            task_id = next(
                (
                    task.task_id
                    for task in sorted(
                        self.session.task_control_registry.tasks(),
                        key=lambda value: value.admitted_at,
                        reverse=True,
                    )
                    if task.task_kind == PORTFOLIO_PUBLIC_TASK_KIND
                    and task.lifecycle.value == "SUCCEEDED"
                    and self.book_result_hash(task.task_id) == result_hash
                ),
                None,
            )
        activation: Activation = "NOT_ACTIVATABLE"
        reasons: dict[str, str] = {}
        if task_id is not None:
            try:
                task = self.session.task_control_registry.task(task_id)
            except TaskNotFoundError:
                activation = "HELD"
                reasons = {"activation": "strategy_activation.book_task_absent"}
            else:
                if task.task_kind == PORTFOLIO_PUBLIC_TASK_KIND:
                    durable = portfolio_research_task_input(task)
                    state = self._state(durable.selected_strategy_package_id)
                    if state.get("book_task_id") == str(task_id) and state["status"] == "ACTIVE":
                        activation = "ACTIVE"
                    elif self.manifest().strategy_installation == "NON_DEFAULT_RESEARCH" and any(
                        recipe.strategy_id == durable.selected_strategy_package_id
                        for recipe in FROZEN_RESEARCH_BOOK_RECIPES
                    ):
                        try:
                            self._activation_book(task_id, self.manifest())
                        except ValueError as error:
                            activation = "HELD"
                            reasons = {
                                "activation": public_failure(
                                    error, "strategy_activation.book_refused"
                                )
                            }
                        else:
                            activation = "A_PERSON_MAY_ACTIVATE"
        return ResultStanding.of(
            comparison="NOT_APPLICABLE",
            execution="SUCCEEDED",
            contract="PASSED",
            evidence="DEVELOPMENT",
            activation=activation,
            reasons=reasons,
        )

    def _schedule(
        self, book: _Book, now: datetime
    ) -> tuple[CausalExecutionSchedulePoint, tuple[CausalExecutionSchedulePoint, ...]]:
        """The book's last formation's point, and the formations after it through the horizon."""
        return self._schedule_after(book.source.formation_sessions[-1], now)

    def _schedule_after(
        self, last: date, now: datetime
    ) -> tuple[CausalExecutionSchedulePoint, tuple[CausalExecutionSchedulePoint, ...]]:
        """The same activation calendar for its admission and its conditional date read."""
        through = _latest_common_us_session(on_or_before=now.date(), observed_at=now)
        through += FORWARD_HORIZON
        points = planned_local_qa_schedule(last, through)
        if not points or points[0].formation_session != last:
            raise ValueError("strategy_activation.schedule_unavailable")
        future = tuple(p for p in points[1:] if p.formation_session <= through)
        if not future or future[0].formation_session != points[0].entry_session:
            raise ValueError("strategy_activation.schedule_unavailable")
        return points[0], future

    def dates(
        self, package_id: str, *, book_sessions: tuple[date, ...] | None = None
    ) -> dict[str, object]:
        """Read information dates and activation's start from the strategy's sealed sources.

        A missing source is explicit, never a book window standing in for training.
        This projection reads metadata only; it neither prepares nor fits a model.
        """
        package = self.packages().get(package_id)
        current = self.manifest()
        rows: list[dict[str, object]] = []
        known: list[date] = []
        missing: list[str] = []
        formations: tuple[date, ...] = ()
        state = self._state(package_id)
        renewals: list[dict[str, object]] = []
        checkpoint = None
        if state["status"] == "ACTIVE":
            decision = next(
                v for v in current.decision_updates or () if v.strategy_package_id == package_id
            )
            checkpoint = self.ledger.load_decision_checkpoint(decision.checkpoint_hash)

        def source(
            kind: str,
            days: tuple[date, ...],
            path: str | None,
            record: str | None,
            field: str,
            component: str | None = None,
        ) -> None:
            latest = max(days, default=None)
            detail = None if latest else f"The {kind} record does not provide its information date."
            rows.append(
                {
                    "kind": kind,
                    "component_id": component,
                    "through": None if latest is None else latest.isoformat(),
                    "relative_path": path,
                    "record_id": record,
                    "field": field,
                    "status": "KNOWN" if latest else "UNAVAILABLE",
                    "detail": detail,
                }
            )
            if latest is None:
                missing.append(str(detail))
            else:
                known.append(latest)

        portfolio = PortfolioResearchArtifactStore(self.workspace / "artifacts")
        alpha = AlphaCurrentArtifactStore(self.workspace / "artifacts")
        recipe = next(
            (v for v in FROZEN_RESEARCH_BOOK_RECIPES if v.strategy_id == package_id), None
        )
        artifact = next(
            (v for v in current.strategy_artifacts if v.artifact_key == ARTIFACT_KEY), None
        )
        authority: LifecyclePortfolioAuthority | None = None
        if artifact is not None and recipe is not None:
            try:
                authority_path = confined(self.workspace, artifact.relative_path)
                score_source = LifecycleResearchScoreSource(
                    manifest_path=authority_path, recipe=recipe
                )
                authority = score_source.manifest
                if (
                    package is None
                    or package.score_source("HISTORICAL_ARRAY_REPLAY").evidence_identity_hash
                    != authority.authority_hash
                ):
                    raise ValueError("portfolio_application.lifecycle_authority_invalid")
                components = {v.component_id: v for v in authority.components}
                selected = [components[c.component_id] for c in recipe.components]
                formations = score_source.formation_sessions
                for evidence_component in selected:
                    source(
                        "FROZEN_RESEARCH_WINDOW",
                        evidence_component.formation_sessions,
                        artifact.relative_path,
                        authority.authority_hash,
                        "components[].formation_sessions",
                        evidence_component.component_id,
                    )
                source(
                    "POST_OBSERVED_SESSION_AXIS",
                    authority.formation_sessions,
                    artifact.relative_path,
                    authority.authority_hash,
                    "formation_sessions",
                )
            except (OSError, ValueError, KeyError):
                source(
                    "FROZEN_RESEARCH_WINDOW", (), artifact.relative_path, None, "formation_sessions"
                )
        else:
            original = next(
                (
                    v
                    for v in current.strategy_artifacts
                    if v.artifact_key == "POST_OBSERVED_STRATEGY_AUTHORITY_ROOT"
                ),
                None,
            )
            if original is not None and recipe is not None:
                relative = f"{original.relative_path}/{POST_OBSERVED_AUTHORITY_MANIFEST_NAME}"
                try:
                    old = PostObservedStrategyAuthorityManifest.model_validate_json(
                        confined(self.workspace, relative).read_bytes()
                    )
                    if (
                        package is None
                        or package.score_source("HISTORICAL_ARRAY_REPLAY").evidence_identity_hash
                        != old.authority_hash
                    ):
                        raise ValueError("portfolio_application.lifecycle_authority_invalid")
                    formations = old.formation_sessions
                    source(
                        "POST_OBSERVED_SESSION_AXIS",
                        formations,
                        relative,
                        old.authority_hash,
                        "formation_sessions",
                    )
                except (OSError, ValueError):
                    source("POST_OBSERVED_SESSION_AXIS", (), relative, None, "formation_sessions")
            else:
                source("FROZEN_RESEARCH_WINDOW", (), None, None, "formation_sessions")

        for planned_component in () if package is None else package.component_plan:
            component_id = planned_component.component_id
            evidence = (
                None
                if authority is None
                else next((v for v in authority.components if v.component_id == component_id), None)
            )
            experiment: dict[str, object] | None = None
            if evidence is not None:
                try:
                    study = self.read_information(evidence.task_id)
                    document = study.get("document")
                    saved = document.get("experiment") if isinstance(document, dict) else None
                    if isinstance(saved, dict):
                        experiment = saved
                except (ValueError, OSError, KeyError):
                    pass
            binding = next(
                (
                    v
                    for v in current.score_inputs or ()
                    if v.strategy_package_id == package_id and v.component_id == component_id
                ),
                None,
            )
            path = None if binding is None else binding.authority_relative_path
            expected = None if binding is None else binding.authority_hash
            if binding is None:
                try:
                    if experiment is None:
                        raise ValueError("strategy_activation.training_admission_absent")
                    selected_id, expected = component_training_selection(
                        str(experiment.get("data_snapshot_handle", ""))
                    )
                    training = next(
                        (
                            v
                            for v in current.model_training_inputs or ()
                            if v.component_id == selected_id == component_id
                            and v.authority_hash == expected
                        ),
                        None,
                    )
                    if training is None:
                        raise ValueError("strategy_activation.training_admission_absent")
                    path = training.authority_relative_path
                except (ValueError, OSError):
                    path, expected = None, None
            try:
                if path is None or expected is None:
                    raise ValueError("strategy_activation.training_admission_absent")
                admission = read_lifecycle_admission(
                    confined(self.workspace, path), expected_hash=expected
                )
                if admission.component.component_id != component_id:
                    raise ValueError("strategy_activation.training_admission_mismatch")
                observations = alpha._load(
                    "lifecycle-training-observations",
                    admission.observations_hash,
                    "content_hash",
                    AlphaTrainingObservations,
                )
                snapshot = alpha.load_frozen_observation_snapshot(observations.observation_hash)
                source(
                    "TRAINING_OBSERVATIONS",
                    tuple(v for v in observations.label_available_sessions if v is not None),
                    f"artifacts/alpha-research/current/lifecycle-training-observations/{observations.content_hash}.json",
                    observations.content_hash,
                    "label_available_sessions",
                    component_id,
                )
                source(
                    "TRAINING_OBSERVATION_AXIS",
                    snapshot.formation_sessions,
                    f"artifacts/alpha-research/current/frozen-observation-snapshots/{snapshot.snapshot_hash}.json",
                    snapshot.snapshot_hash,
                    "formation_sessions",
                    component_id,
                )
                prepared = tuple(d for p in admission.prepared for d in p.plan.training_sessions)
                source(
                    "PREPARED_REFIT_WINDOWS",
                    prepared,
                    path,
                    admission.content_hash,
                    "prepared[].plan.training_sessions (vintages named by prepared[].plan.vintage)",
                    component_id,
                )
                rows[-1]["vintages"] = sorted({p.plan.vintage for p in admission.prepared})
                fitted = {c.prepared_hash for c in admission.initial_children}
                fitted_vintages = {p.plan.vintage for p in admission.prepared}
                roots = [alpha.root]
                if experiment is not None and experiment.get("output_workspace"):
                    roots.append(
                        AlphaDevelopmentArtifactStore(
                            confined(self.workspace, str(experiment["output_workspace"]))
                            / "alpha-lifecycle"
                        ).root
                    )
                seen_windows: set[tuple[str, ...]] = set()
                for model_file in sorted(
                    {
                        p
                        for root in roots
                        for p in (root / "current/lifecycle-model-sets").glob("*.json")
                    }
                ):
                    publication = AlphaModelSetPublication.model_validate_json(
                        model_file.read_bytes()
                    )
                    if (
                        publication.component.recipe_hash == admission.component.recipe_hash
                        and publication.lifecycle == admission.lifecycle
                    ):
                        if publication.formation <= (
                            admission.renewal_through or admission.formation_end
                        ):
                            fitted_vintages.update(c.vintage for c in publication.children)
                        if publication.formation > admission.formation_end:
                            continue
                        used = {c.prepared_hash for c in publication.children}
                        plans = tuple(p for p in admission.prepared if p.content_hash in used)
                        windows = tuple(p.content_hash for p in plans)
                        if plans and windows not in seen_windows:
                            seen_windows.add(windows)
                            fitted.update(used)
                            source(
                                "FIT_VINTAGES",
                                tuple(d for p in plans for d in p.plan.training_sessions),
                                model_file.relative_to(self.workspace).as_posix(),
                                publication.content_hash,
                                "children[].prepared_hash -> "
                                "admission.prepared[].plan.training_sessions",
                                component_id,
                            )
                            rows[-1]["vintages"] = sorted({p.plan.vintage for p in plans})
                initial = tuple(
                    p
                    for p in admission.prepared
                    if p.content_hash in {c.prepared_hash for c in admission.initial_children}
                )
                if initial or not fitted:
                    source(
                        "FIT_VINTAGES",
                        tuple(d for p in initial for d in p.plan.training_sessions),
                        path,
                        admission.content_hash,
                        "initial_children[].prepared_hash -> prepared[].plan.training_sessions",
                        component_id,
                    )
                    rows[-1]["vintages"] = sorted({p.plan.vintage for p in initial})
                if checkpoint is not None:
                    needed = {
                        vintage
                        for day in checkpoint.formation_sessions[
                            checkpoint.initial_book.next_position :
                        ]
                        if day <= (admission.renewal_through or admission.formation_end)
                        for vintage in admission.lifecycle.vintages(day)
                    }
                    remaining = sorted((set(admission.fit_vintages) & needed) - fitted_vintages)
                    renewals.append(
                        {
                            "component_id": component_id,
                            "status": "UNAVAILABLE" if binding is None else "KNOWN",
                            "remaining_fit_vintages": None if binding is None else remaining,
                            "next_fit_vintage": None
                            if binding is None
                            else next(iter(remaining), None),
                            "renewal_through": None
                            if binding is None or admission.renewal_through is None
                            else admission.renewal_through.isoformat(),
                            "grant_file": None if binding is None else path,
                            "grant_hash": None if binding is None else admission.content_hash,
                            "detail": "No active grant is bound for this component; "
                            "its next fit vintage is unknown."
                            if binding is None
                            else "Remaining admitted fit vintages for this component's "
                            "forward sessions through its grant's horizon."
                            if remaining
                            else "No admitted fit vintage remains for this component's "
                            "forward sessions through its grant's horizon.",
                        }
                    )
            except (OSError, ValueError):
                if state["status"] == "ACTIVE":
                    renewals.append(
                        {
                            "component_id": component_id,
                            "status": "UNAVAILABLE",
                            "remaining_fit_vintages": None,
                            "next_fit_vintage": None,
                            "renewal_through": None,
                            "grant_file": path,
                            "grant_hash": expected,
                            "detail": "The component's sealed grant or model metadata cannot be "
                            "read; its next fit vintage is unknown.",
                        }
                    )
                source(
                    "TRAINING_OBSERVATIONS", (), path, expected, "observations_hash", component_id
                )
                source(
                    "FIT_VINTAGES",
                    (),
                    path,
                    expected,
                    "prepared[].plan.training_sessions",
                    component_id,
                )

        calibrated = (
            ()
            if recipe is None
            else tuple(c for c in recipe.components if c.weight_rule == "mu.iv0")
        )
        for calibrated_component in calibrated:
            calibration_binding = next(
                (
                    v
                    for v in current.calibration_inputs or ()
                    if v.strategy_package_id == package_id
                ),
                None,
            )
            if calibration_binding is not None:
                try:
                    seed = portfolio.load(
                        category="calibration-observations",
                        content_hash=calibration_binding.seed_hash,
                        model=CalibrationObservations,
                        identity_field="content_hash",
                    )
                    source(
                        "CALIBRATION_SEED",
                        (
                            *seed.formation_sessions,
                            *(d for d in seed.holding_end_sessions if d is not None),
                        ),
                        f"artifacts/portfolio-strategy-lab/calibration-observations/{seed.content_hash}.json",
                        seed.content_hash,
                        "formation_sessions + holding_end_sessions",
                        calibrated_component.component_id,
                    )
                except (OSError, ValueError):
                    source(
                        "CALIBRATION_SEED",
                        (),
                        None,
                        calibration_binding.seed_hash,
                        "holding_end_sessions",
                        calibrated_component.component_id,
                    )
            elif authority is not None:
                ends = dict(
                    zip(authority.formation_sessions, authority.holding_end_sessions, strict=True)
                )
                source(
                    "CALIBRATION_SEED_LANES_IF_ACTIVATED",
                    tuple(ends[d] for d in formations),
                    artifact.relative_path if artifact else None,
                    authority.authority_hash,
                    "holding_end_sessions on activation's formation axis",
                    calibrated_component.component_id,
                )
            else:
                source(
                    "CALIBRATION_SEED",
                    (),
                    None,
                    None,
                    "holding_end_sessions",
                    calibrated_component.component_id,
                )

        cutoff = max(known, default=None) if not missing else None
        start: str | None = None
        basis = "UNAVAILABLE"
        start_source: dict[str, object] | None = None
        forward_sessions: tuple[date, ...] = ()
        activation_time: datetime | None = None
        if checkpoint is not None:
            start = str(state["first_forward_session"])
            basis = "ACTIVE"
            start_source = {
                "record": "PortfolioDecisionCheckpoint",
                "record_id": decision.checkpoint_hash,
                "relative_path": self.ledger.root.joinpath(
                    "decision-checkpoints", f"{decision.checkpoint_hash}.json"
                )
                .relative_to(self.workspace)
                .as_posix(),
                "field": "epoch_start",
                "book_task_id": state["book_task_id"],
            }
            forward_sessions = tuple(
                d for d in checkpoint.formation_sessions if d >= checkpoint.epoch_start
            )
            activation_time = checkpoint.activated_at
        elif formations:
            try:
                _, future = self._schedule_after(formations[-1], self.clock())
                start = future[0].formation_session.isoformat()
                basis = "IF_ACTIVATED"
                forward_sessions = tuple(p.formation_session for p in future)
                start_source = {
                    "record": "sealed research formation axis",
                    "sources": [
                        {key: v[key] for key in ("relative_path", "record_id", "field")}
                        for v in rows
                        if v["kind"] == "POST_OBSERVED_SESSION_AXIS"
                    ],
                    "formation_end": formations[-1].isoformat(),
                    "rule": "StrategyActivation._schedule_after (the activation rule)",
                }
            except ValueError:
                pass
        actionable: str | None = None
        actionable_basis = "UNAVAILABLE"
        actionable_source: dict[str, object] | None = None
        observed = activation_time if state["status"] == "ACTIVE" else self.clock()
        if observed is not None:
            try:
                completed, first_actionable = self._first_actionable(observed)
                actionable = first_actionable.entry_session.isoformat()
                actionable_basis = "ACTIVE" if state["status"] == "ACTIVE" else "IF_ACTIVATED"
                actionable_source = {
                    "reference": "ACTIVATION" if activation_time else "CURRENT_CLOCK",
                    "latest_completed_session": completed.isoformat(),
                    "formation_session": first_actionable.formation_session.isoformat(),
                    "entry_at": first_actionable.entry_open_at.isoformat(),
                    "rule": "first planned XNAS/XNYS entry_open_at strictly after the "
                    "activation or current clock",
                    **(
                        {"activated_at": activation_time.isoformat(), "checkpoint": start_source}
                        if activation_time
                        else {
                            "observed_on": observed.date().isoformat(),
                        }
                    ),
                }
            except (ValueError, DataQualityError):
                pass
        in_sample = (
            None
            if cutoff is None or actionable is None or start is None
            else tuple(
                d for d in forward_sessions if d <= cutoff and d < date.fromisoformat(actionable)
            )
        )
        sessions = book_sessions
        after = (
            None if sessions is None or cutoff is None else tuple(d for d in sessions if d > cutoff)
        )
        return {
            "model_renewals": {
                "status": "INACTIVE"
                if state["status"] != "ACTIVE"
                else "KNOWN"
                if all(v["status"] == "KNOWN" for v in renewals)
                else "UNAVAILABLE",
                "components": renewals,
                "detail": "This installed strategy is inactive; no fit vintage is scheduled."
                if state["status"] != "ACTIVE"
                else "Each component names its remaining admitted fit vintages through its "
                "grant's horizon; the first remaining vintage is next.",
            },
            "information_cutoff": None if cutoff is None else cutoff.isoformat(),
            "information_cutoff_status": "KNOWN" if cutoff is not None else "UNAVAILABLE",
            "latest_dated_information": None if not known else max(known).isoformat(),
            "information_sources": rows,
            "information_cutoff_detail": (
                "The information cutoff is the latest date in the strategy's sealed "
                "component records."
            )
            if cutoff
            else "The information cutoff cannot be determined from every required sealed record. "
            + " ".join(missing),
            "forward_book_first_decided_session": start,
            "forward_book_start_basis": basis,
            "forward_book_start_source": start_source,
            "forward_book_start_detail": "The forward book's first decided session continues "
            "from its sealed last formation; it can precede the first actionable session."
            if basis == "ACTIVE"
            else (
                (
                    "If a whole book over this sealed support is activated now, its first "
                    "forward decided session continues that book; this is conditional, "
                    "not an activation."
                )
                if basis == "IF_ACTIVATED"
                else (
                    "A forward book start cannot be determined without a sealed formation end and "
                    "the activation calendar. Run and read the strategy's whole book before "
                    "requesting activation."
                )
            ),
            "first_actionable_session": actionable,
            "first_actionable_basis": actionable_basis,
            "first_actionable_source": actionable_source,
            "first_actionable_detail": (
                "Hold positions only from the first planned entry strictly after activation. "
                "Earlier forward decisions are a causal replay."
                if actionable_basis == "ACTIVE"
                else "If activated now, hold positions only from the first planned entry "
                "strictly after the current clock. Earlier forward decisions would be a "
                "causal replay."
                if actionable_basis == "IF_ACTIVATED"
                else "The first actionable session is unknown without the activation time or "
                "current clock and the installed planned-entry calendar."
            ),
            "replayed_in_sample_forward_sessions": {
                "count": None if in_sample is None else len(in_sample),
                "first_session": None if not in_sample else in_sample[0].isoformat(),
                "last_session": None if not in_sample else in_sample[-1].isoformat(),
                "detail": "Forward sessions before the first actionable session and through "
                "the information cutoff are a causal replay in-sample, never out-of-sample."
                if in_sample is not None
                else "The in-sample replay range is unknown until the forward book start, "
                "information cutoff and first actionable session are known.",
            },
            "book_sessions_after_cutoff": None
            if sessions is None
            else {
                "count": None if after is None else len(after),
                "first_session": None if not after else after[0].isoformat(),
                "detail": "Sessions strictly after the information cutoff."
                if after is not None
                else (
                    "The book's sessions cannot be compared until every required information "
                    "date is known."
                ),
            },
        }

    def _first_actionable(self, observed_at: datetime) -> tuple[date, CausalExecutionSchedulePoint]:
        """The first planned entry still ahead of the person, using the execution schedule."""
        table = materialize_calendar_schedule(
            ("XNAS", "XNYS"),
            start=observed_at.date() - timedelta(days=14),
            end=observed_at.date(),
            as_of_timestamp=observed_at,
        )
        venues: dict[date, set[str]] = {}
        for row in table.select(["session_date", "calendar_id"]).to_pylist():
            venues.setdefault(row["session_date"], set()).add(row["calendar_id"])
        completed = max(d for d, v in venues.items() if v == {"XNAS", "XNYS"})
        point = next(
            (
                p
                for p in planned_local_qa_schedule(completed, observed_at.date())
                if p.entry_open_at > observed_at
            ),
            None,
        )
        if point is None:
            raise ValueError("strategy_activation.schedule_unavailable")
        return completed, point

    def _grant(
        self,
        book: _Book,
        manifest: ResearchWorkspaceManifest,
        component_id: str,
        future: tuple[CausalExecutionSchedulePoint, ...],
    ) -> tuple[AlphaModelLifecycleAdmission, int]:
        """One component's models: the study's training admission, renewing through the horizon.

        Returns:
            The published grant and how many of the study's fitted children it reuses.
        """
        evidence = next(
            v for v in book.source.manifest.components if v.component_id == component_id
        )
        study = self.read_experiment(evidence.task_id)
        document = study.get("document")
        experiment = document.get("experiment") if isinstance(document, dict) else None
        if not isinstance(experiment, dict):
            raise ValueError(f"strategy_activation.study_unreadable:{component_id}")
        selected, training_hash = component_training_selection(
            str(experiment.get("data_snapshot_handle", ""))
        )
        binding = next(
            (
                v
                for v in manifest.model_training_inputs or ()
                if v.component_id == component_id and v.authority_hash == training_hash
            ),
            None,
        )
        if selected != component_id or binding is None:
            raise ValueError(f"strategy_activation.training_admission_absent:{component_id}")
        store = AlphaCurrentArtifactStore(self.workspace / "artifacts")
        training = read_lifecycle_admission(
            confined(self.workspace, binding.authority_relative_path),
            expected_hash=binding.authority_hash,
        )
        first, horizon = future[0].formation_session, future[-1].formation_session
        if (
            evidence.lifecycle_hash != training.lifecycle.content_hash
            or evidence.component_recipe_hash != training.component.recipe_hash
            or training.formation_end < first
        ):
            raise ValueError(f"strategy_activation.training_admission_mismatch:{component_id}")
        live = sorted({v for p in future for v in training.lifecycle.vintages(p.formation_session)})
        prepared = {p.plan.vintage: p for p in training.prepared}
        study_store = AlphaDevelopmentArtifactStore(
            confined(self.workspace, str(experiment["output_workspace"])) / "alpha-lifecycle"
        )
        reused = sum(
            reuse_refit_child(
                study_store, store, prepared=prepared[v], component=training.component, seed=seed
            )
            for v in live
            if v in prepared
            for seed in training.lifecycle.seeds
        )
        observations = store._load(
            "lifecycle-training-observations",
            training.observations_hash,
            "content_hash",
            AlphaTrainingObservations,
        )
        source = store.load_frozen_observations(observations.observation_hash)
        grant = AlphaModelLifecycleAdmission.create(
            component=training.component,
            lifecycle=training.lifecycle,
            observations_hash=training.observations_hash,
            prepared=training.prepared,
            initial_children=(),
            formation_start=first,
            formation_end=training.formation_end,
            maximum_fit_attempts=len(live) * len(training.lifecycle.seeds),
            environment_hash=training.environment_hash,
            fit_vintages=tuple(sorted(set(training.fit_vintages) | set(live))),
            training_factor_ids=tuple(sorted(source.formula_values)),
            renewal_through=horizon,
        )
        store._publish("lifecycle-admissions", grant, "content_hash")
        with verified_lifecycle_admissions():
            admitted = admit_component_inference(
                self.workspace / f"{_ADMISSIONS}/{grant.content_hash}.json",
                store=store,
                expected_hash=grant.content_hash,
            )
        if not isinstance(admitted, AdmittedRenewingInference) or not (
            admitted.authority.supports(first) or admitted.authority.can_prepare(first)
        ):
            raise ValueError(f"strategy_activation.models_unavailable:{component_id}")
        return grant, reused

    def _seed(self, book: _Book, recipe_hash: str, component_id: str) -> str:
        """The calibration seed from the authority's own lanes on the book's formations."""
        source, manifest = book.source, book.source.manifest
        evidence = next(v for v in manifest.components if v.component_id == component_id)
        sessions = source.formation_sessions
        scores = source._lane(
            evidence.scores_hash,
            (len(evidence.formation_sessions), len(source.ordered_listing_ids)),
        )[source._rows(evidence.formation_sessions)]
        shared = source.resolve()
        holding = dict(zip(manifest.formation_sessions, manifest.holding_end_sessions, strict=True))
        seed = publish_observations(
            PortfolioResearchArtifactStore(self.workspace / "artifacts"),
            scores=np.ascontiguousarray(scores, dtype=np.float64),
            returns=np.ascontiguousarray(shared.realized_simple_returns, dtype=np.float64),
            eligible=np.ascontiguousarray(shared.decision_eligible, dtype=np.bool_),
            origin="FROZEN_RESEARCH_INPUTS",
            strategy_package_hash=book.package.package_hash,
            component_recipe_hash=recipe_hash,
            rule=FrozenRankCalibrationRule.create(
                activation_session=sessions[book.recipe.sizing_activation_formation]
            ),
            formation_sessions=sessions,
            holding_end_sessions=tuple(holding[day] for day in sessions),
            ordered_listing_ids=source.ordered_listing_ids,
            source_hashes=(manifest.authority_hash, book.execution.ledger_hash),
        )
        return str(seed.content_hash)

    def _checkpoint(
        self,
        book: _Book,
        grants: tuple[AlphaModelLifecycleAdmission, ...],
        last: CausalExecutionSchedulePoint,
        future: tuple[CausalExecutionSchedulePoint, ...],
        now: datetime,
    ) -> PortfolioDecisionCheckpoint:
        """The book's opening state, read from its run's sealed final boundary."""
        boundary, axis = book.boundary, book.execution.ordered_listing_ids
        assert boundary.sleeve_weights_hash is not None
        weights: npt.NDArray[np.float64] = np.frombuffer(
            self.ledger.load_lane(
                category="boundary-weights", content_hash=boundary.optimizer_reference_hash
            ),
            dtype="<f8",
        )
        sleeves: npt.NDArray[np.float64] = np.frombuffer(
            self.ledger.load_lane(
                category="boundary-sleeves", content_hash=boundary.sleeve_weights_hash
            ),
            dtype="<f8",
        ).reshape(boundary.sleeve_count, len(axis))
        count = len(book.recipe.components)
        market = MarketDataRepository(self.workspace)
        universe = market.current_quality_filtered_research_manifest(
            market_profile_id="us-current-index-research"
        )
        if universe is None:
            raise ValueError("portfolio_update.checkpoint_listing_authority_mismatch")
        labels = tuple(v.symbol for v in market.listing_scope(universe, listing_ids=axis))
        authorities = tuple(grant.content_hash for grant in grants)
        recipes = tuple(grant.component.recipe_hash for grant in grants)
        ids = book.package.component_ids
        return PortfolioDecisionCheckpoint.create(
            purpose="PERSON_ACTIVATED_FORWARD_RESEARCH",
            activated_at=now,
            book_task_id=book.task_id,
            package=book.package,
            recipe=book.recipe,
            ordered_listing_ids=axis,
            listing_labels=labels,
            model_authority_hash=authorities[0]
            if count == 1
            else canonical_hash(tuple(zip(ids, authorities, strict=True))),
            inference_recipe_hash=recipes[0]
            if count == 1
            else canonical_hash(tuple(zip(ids, recipes, strict=True))),
            component_authority_hashes=() if count == 1 else authorities,
            component_recipe_hashes=() if count == 1 else recipes,
            epoch_start=future[0].formation_session,
            epoch_end=future[-1].formation_session,
            formation_sessions=(
                *book.execution.formation_sessions,
                *(p.formation_session for p in future),
            ),
            initial_book=PortfolioEntryBook.create(
                schedule=last,
                next_position=boundary.decided_formation_count,
                weights=tuple(float(v) for v in weights),
                cash=boundary.optimizer_reference_cash,
                sleeves=tuple(tuple(float(v) for v in row) for row in sleeves),
                component_sleeve_counts=(book.recipe.tranches,) * count if count > 1 else (),
            ),
            source_hashes=(
                book.execution.ledger_hash,
                boundary.boundary_hash,
                book.source.evidence_identity_hash,
            ),
        )

    def activate(self, task_id: UUID) -> dict[str, object]:
        """Run a reviewed book's strategy forward: bind its models, seed and opening state.

        Args:
            task_id: The completed book run of an installed research strategy, as reviewed.

        Returns:
            What was bound: the models (each grant and the fitted children it reuses), the seed,
            the checkpoint, its first forward formation and its horizon; no fit ran.

        Raises:
            ValueError: The book is not a whole completed run of an installed research package,
                its package moved since it ran, the package is already active, or its models,
                schedule or listings refuse.
        """
        seconds: dict[str, float] = {}
        with self.gate.hold():
            with self._stage(seconds, None, "book"):
                require_no_pending_cleanup(self.workspace)
                current = self.manifest()
                book = self._activation_book(task_id, current)
                package_id = book.package.strategy_id
                review = None if self.read_review is None else self.read_review(task_id)
                now = self.clock()
                last, future = self._schedule(book, now)
            try:
                with self._stage(seconds, package_id, "models"):
                    made = tuple(
                        self._grant(book, current, c.component_id, future)
                        for c in book.recipe.components
                    )
                    grants = tuple(grant for grant, _ in made)
                    score_inputs = tuple(
                        ResearchWorkspaceScoreInput(
                            strategy_package_id=package_id,
                            strategy_package_hash=book.package.package_hash,
                            component_id=component.component_id,
                            authority_relative_path=f"{_ADMISSIONS}/{grant.content_hash}.json",
                            authority_hash=grant.content_hash,
                            source_kind="WORKSPACE_DATA_FEATURE",
                        )
                        for component, grant in zip(book.recipe.components, grants, strict=True)
                    )
                    calibrated = [
                        (c.component_id, g)
                        for c, g in zip(book.recipe.components, grants, strict=True)
                        if c.weight_rule == "mu.iv0"
                    ]
                with self._stage(seconds, package_id, "seed"):
                    seed = (
                        None
                        if not calibrated
                        else self._seed(
                            book, calibrated[0][1].component.recipe_hash, calibrated[0][0]
                        )
                    )
                with self._stage(seconds, package_id, "checkpoint"):
                    checkpoint = self._checkpoint(book, grants, last, future, now)
                    admit_decision_checkpoint(self.workspace, checkpoint, score_inputs)
                    self.ledger.publish_decision_checkpoint(checkpoint)

                def bind(now_manifest: ResearchWorkspaceManifest) -> ResearchWorkspaceManifest:
                    # The gate is held since `current` was read, so it is the manifest changed.
                    if now_manifest != current:
                        raise ValueError("strategy_activation.configuration_changed")
                    kept = _without(now_manifest, package_id)
                    return now_manifest.with_bindings(
                        score_inputs=(*(kept["score_inputs"] or ()), *score_inputs),
                        calibration_inputs=(
                            *(kept["calibration_inputs"] or ()),
                            *(
                                ()
                                if seed is None
                                else (
                                    ResearchWorkspaceCalibrationInput(
                                        strategy_package_id=package_id,
                                        strategy_package_hash=book.package.package_hash,
                                        seed_hash=seed,
                                        source_kind="WORKSPACE_DATA_FEATURE",
                                    ),
                                )
                            ),
                        )
                        or None,
                        decision_updates=(
                            *(kept["decision_updates"] or ()),
                            ResearchWorkspaceDecisionUpdate(
                                strategy_package_id=package_id,
                                strategy_package_hash=book.package.package_hash,
                                checkpoint_hash=checkpoint.content_hash,
                            ),
                        ),
                    )

                with self._stage(seconds, package_id, "bind"):
                    _, updated = update_research_workspace_manifest(
                        self.workspace, bind, gate=self.gate
                    )
                    self.hold(updated)
            finally:
                self._in_flight.pop(package_id, None)
        return {
            "status": "ACTIVATED",
            "review_standing": review,
            "strategy_dates": self.dates(package_id),
            "strategy_package_id": package_id,
            "book_task_id": str(task_id),
            "next_decision_session": future[0].formation_session.isoformat(),
            "horizon": future[-1].formation_session.isoformat(),
            "models": [
                {
                    "component_id": grant.component.component_id,
                    "authority_hash": grant.content_hash,
                    "reused_fits": reused,
                    "fit_vintages_through_horizon": sorted(
                        set(grant.fit_vintages) - {p.plan.vintage for p in grant.prepared}
                    ),
                }
                for grant, reused in made
            ],
            "calibration_seed_hash": seed,
            "checkpoint_hash": checkpoint.content_hash,
            "workspace_manifest_hash": updated.manifest_hash,
            "fit_calls": 0,
            "stage_seconds": seconds,
            "claim": "FORWARD_RESEARCH_NOT_TRADING_ADVICE",
            "next_requests": {
                "update": {"operation": "RESEARCH_UPDATE_PLAN", "strategy_package_id": package_id},
                # The person's one click back, which the agent names when it tells them (STOPS-1).
                "deactivate": {
                    "operation": "STRATEGY_DEACTIVATE",
                    "strategy_package_id": package_id,
                },
            },
        }

    @contextmanager
    def _stage(
        self, seconds: dict[str, float], package_id: str | None, name: str
    ) -> Iterator[None]:
        """One phase of an activation, timed; its package's summary names it while it runs."""
        started = time.monotonic()
        if package_id is not None:
            self._in_flight[package_id] = (name, started)
        try:
            yield
        finally:
            seconds[name] = round(time.monotonic() - started, 3)

    def deactivate(self, package_id: str) -> dict[str, object]:
        """Stop a strategy running forward: remove its three bindings; its history stays.

        Args:
            package_id: The active installed package.

        Returns:
            The package and the new manifest's identity.

        Raises:
            ValueError: `strategy_activation.not_active` when nothing binds the package.
        """
        with self.gate.hold():
            current = self.manifest()
            if not _bound(current, package_id):
                raise ValueError("strategy_activation.not_active")

            def unbind(now_manifest: ResearchWorkspaceManifest) -> ResearchWorkspaceManifest:
                if now_manifest != current:
                    raise ValueError("strategy_activation.configuration_changed")
                return now_manifest.with_bindings(**_without(now_manifest, package_id))

            _, updated = update_research_workspace_manifest(self.workspace, unbind, gate=self.gate)
            self.hold(updated)
        return {
            "status": "DEACTIVATED",
            "strategy_package_id": package_id,
            "workspace_manifest_hash": updated.manifest_hash,
        }

    def _state(self, package_id: str) -> dict[str, object]:
        """Whether a package runs forward, from which book and through when.

        Args:
            package_id: An installed package.

        Returns:
            `ACTIVE` with its book, first forward formation and horizon; `MOVED` when its
            bindings name a package that has since changed (activate a new run of its book);
            `INACTIVE` otherwise.
        """
        binding = next(
            (
                v
                for v in self.manifest().decision_updates or ()
                if v.strategy_package_id == package_id
            ),
            None,
        )
        package = self.packages().get(package_id)
        if binding is None or package is None:
            return {"status": "INACTIVE"}
        if binding.strategy_package_hash != package.package_hash:
            return {"status": "MOVED"}
        checkpoint = self.ledger.load_decision_checkpoint(binding.checkpoint_hash)
        return {
            "status": "ACTIVE",
            "book_task_id": None
            if checkpoint.book_task_id is None
            else str(checkpoint.book_task_id),
            "activated_at": None
            if checkpoint.activated_at is None
            else checkpoint.activated_at.isoformat(),
            "first_forward_session": checkpoint.epoch_start.isoformat(),
            "horizon": checkpoint.epoch_end.isoformat(),
        }

    def summary(self, package_id: str) -> dict[str, object]:
        """Read activation metadata and dates without opening its book's full review.

        Args:
            package_id: The installed strategy to discover.

        Returns:
            Activation state and owner dates. The selected state/offer verifies review standing.
        """
        running = self._in_flight.get(package_id)
        activating = (
            {}
            if running is None
            else {
                "activating": {
                    "stage": running[0],
                    "elapsed_seconds": round(time.monotonic() - running[1], 1),
                }
            }
        )
        return {
            **self._state(package_id),
            "strategy_dates": self.dates(package_id),
            **activating,
        }

    def state(self, package_id: str) -> dict[str, object]:
        """The activation state with the one owner's information and start-date readout."""
        state = self.summary(package_id)
        task = state.get("book_task_id")
        return {
            **state,
            "review_standing": None
            if task is None or self.read_review is None
            else self.read_review(UUID(str(task))),
        }

    def offer(self, package_id: str) -> dict[str, object]:
        """A package's state and the request a person can take on it (U73).

        A package its bindings name is offered its stop. One that does not run forward from
        its installed package is offered the activation of its newest completed book when
        `activate` would admit that book, checked as `activate` checks it; when it would not,
        the book is named as held with the refusal's code. A default installation has no book
        to activate.

        Args:
            package_id: An installed package.

        Returns:
            `state`'s answer, with `next_requests` (`activate`, `deactivate`) and `held`.
        """
        state = self.state(package_id)
        requests: dict[str, object] = {}
        if state["status"] != "INACTIVE":
            requests["deactivate"] = {
                "operation": "STRATEGY_DEACTIVATE",
                "strategy_package_id": package_id,
            }
        manifest = self.manifest()
        if state["status"] != "ACTIVE" and manifest.strategy_installation == "NON_DEFAULT_RESEARCH":
            newest = max(
                (
                    task
                    for task in self.session.task_control_registry.tasks()
                    if task.task_kind == PORTFOLIO_PUBLIC_TASK_KIND
                    and task.lifecycle is TaskLifecycle.SUCCEEDED
                    and portfolio_research_task_input(task).selected_strategy_package_id
                    == package_id
                ),
                key=lambda task: task.admitted_at,
                default=None,
            )
            if newest is not None:
                try:
                    book = self._activation_book(newest.task_id, manifest)
                except ValueError as error:
                    state["held"] = {
                        "task_id": str(newest.task_id),
                        "failure_code": public_failure(error, "strategy_activation.book_refused"),
                    }
                else:
                    if self.read_review is not None:
                        state["review_standing"] = self.read_review(newest.task_id)
                    state["review_holdings"] = self._review_holdings(book)
                    requests["activate"] = {
                        "operation": "STRATEGY_ACTIVATE",
                        "task_id": str(newest.task_id),
                    }
        if requests:
            state["next_requests"] = requests
        return state

    def _review_holdings(self, book: _Book) -> dict[str, object]:
        """The book's last sealed holdings, read where activation reads them (A2).

        Its run's sealed final weights and cash, by listing, with the formation that decided
        them and the session they were entered. Nothing is computed: the next positions come
        from the first forward update after a person activates the book.
        """
        boundary, axis = book.boundary, book.execution.ordered_listing_ids
        weights: npt.NDArray[np.float64] = np.frombuffer(
            self.ledger.load_lane(
                category="boundary-weights", content_hash=boundary.optimizer_reference_hash
            ),
            dtype="<f8",
        )
        last = book.execution.formation_sessions[-1]
        points = planned_local_qa_schedule(last, last)
        held = sorted(
            ((axis[i], float(w)) for i, w in enumerate(weights) if w != 0.0),
            key=lambda row: (-abs(row[1]), row[0]),
        )
        return {
            "claim": "BOOK_LAST_HOLDINGS_NOT_NEXT_POSITIONS",
            "detail": REVIEW_HOLDINGS_WORDS,
            "book_task_id": str(book.task_id),
            "formation_session": last.isoformat(),
            "entry_session": points[0].entry_session.isoformat()
            if points and points[0].formation_session == last
            else None,
            "held_count": len(held),
            "cash": format_book_weight(boundary.optimizer_reference_cash),
            "positions": [
                {"listing_id": listing, "weight": format_book_weight(weight)}
                for listing, weight in held
            ],
        }


__all__ = ["FORWARD_HORIZON", "StrategyActivation", "admit_decision_checkpoint"]
