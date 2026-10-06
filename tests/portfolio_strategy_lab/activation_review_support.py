"""A sealed synthetic public book for activation and review-standing tests.

The book is admitted and executed by the public Portfolio application. Its score
authority, training admission, and Market universe are synthetic QA inputs; the
activation owner still reopens those sealed artifacts and the completed book's
real execution ledger.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import numpy as np

from alphalattice.capabilities.portfolio_backtesting.clocks import EveryFormationClock
from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioStateTransitionBinding,
)
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceArtifact,
    ResearchWorkspaceManifest,
    ResearchWorkspaceModelTrainingInput,
    publish_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.strategy_activation import StrategyActivation
from alphalattice.control.product_host.maintenance.data_update import (
    installed_data_update_binding,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    planned_local_qa_schedule,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
    build_quality_filtered_research_manifest,
    qualification_obligation,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
    FrozenPriceVolumeInputs,
)
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
)
from alphalattice.investment.alpha_research.scores.model_renewal import (
    AlphaModelLifecycleAdmission,
    publish_training_observations,
)
from alphalattice.investment.alpha_research.scores.product_lifecycle import (
    AlphaModelLifecycleRecipe,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    OwnerCoverage,
    PortfolioDeclaredPathReport,
    PortfolioResearchSpec,
    PortfolioSupportCoverage,
)
from alphalattice.investment.portfolio_strategy_lab.application.executor import (
    ResolvedPortfolioExecution,
)
from alphalattice.investment.portfolio_strategy_lab.application.resolution import (
    PublicPortfolioReplayWorkspace,
    ResolvedPortfolioAuthorities,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    FrozenStrategyPackage,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    REBOUND_RETURN_BOOK_RECIPE,
    frozen_book_package,
)
from alphalattice.investment.portfolio_strategy_lab.policies.lifecycle_research import (
    ARTIFACT_KEY,
    CATEGORY,
    LANES,
    LifecyclePortfolioAuthority,
    LifecycleResearchScoreSource,
    LifecycleScoreEvidence,
)
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioResearchArtifactStore,
)
from alphalattice.investment.risk_research.surfaces.decomposition import (
    INSTALLED_RISK_DECOMPOSITION_RECIPE,
    FactorIdiosyncraticRiskSurface,
    RiskAttributionProjection,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

PACKAGE = REBOUND_RETURN_BOOK_RECIPE.strategy_id
_HASH = str(canonical_hash("activation review synthetic QA support"))
_NOW = datetime(2026, 8, 12, 16, tzinfo=UTC)
_HISTORY = 522
_COMPONENT = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component("G6_R0_FAST_REBOUND")
_LIFECYCLE = AlphaModelLifecycleRecipe.from_component(_COMPONENT)


@dataclass(frozen=True, slots=True)
class ActivationReviewHost:
    """The live product and exact identities of its synthetic completed book."""

    live: LocalPortfolioWebSession
    activation: StrategyActivation
    task_id: UUID
    result_hash: str
    last_session: date
    report: PortfolioDeclaredPathReport

    @property
    def book_task_id(self) -> UUID:
        """The exact Task whose published result is the activation book."""
        return self.task_id


class _ActivationReviewResolver:
    """The public Portfolio resolver seam over one sealed lifecycle score source."""

    def __init__(
        self,
        *,
        workspace: Path,
        authority: LifecyclePortfolioAuthority,
        package: FrozenStrategyPackage,
        sessions: tuple[date, ...],
        listings: tuple[str, ...],
        decision: np.ndarray,
        execution: np.ndarray,
        returns: np.ndarray,
        adv: np.ndarray,
        attributions: tuple[RiskAttributionProjection, ...],
    ) -> None:
        self.workspace = workspace
        self.package = package
        artifact = (
            workspace
            / "artifacts/research-strategy-inputs/synthetic/portfolio-strategy-lab"
            / CATEGORY
            / f"{authority.authority_hash}.json"
        )
        self.source = LifecycleResearchScoreSource(
            manifest_path=artifact, recipe=REBOUND_RETURN_BOOK_RECIPE
        )
        self.sessions, self.listings = sessions, listings
        self.decision, self.execution, self.returns, self.adv = (
            decision,
            execution,
            returns,
            adv,
        )
        self.attributions = attributions

    @property
    def strategy_catalog_hash(self) -> str:
        return str(
            canonical_hash(
                {
                    "kind": "ActivationReviewInstalledCatalog",
                    "packages": (self.package.model_dump(mode="json"),),
                }
            )
        )

    def installed_packages(self) -> dict[str, FrozenStrategyPackage]:
        return {self.package.package_hash: self.package}

    def resolve_authorities(
        self, *, workspace: Path, spec: PortfolioResearchSpec
    ) -> ResolvedPortfolioAuthorities:
        del workspace, spec
        source_hash = self.source.evidence_identity_hash
        coverage = PortfolioSupportCoverage.create(
            owners=(
                OwnerCoverage.of(
                    owner_id="sealed_lifecycle_score_source",
                    lane="VERIFIED_LOCAL_LIFECYCLE_SCORES",
                    sessions=self.sessions,
                    identity_hash=source_hash,
                ),
                OwnerCoverage.of(
                    owner_id="synthetic_shared_market_inputs",
                    lane="SYNTHETIC_SEALED_MARKET_LANES",
                    sessions=self.sessions,
                    identity_hash=_HASH,
                ),
            ),
            common_watermark_start=self.sessions[0],
            common_watermark_end=self.sessions[-1],
            common_session_count=len(self.sessions),
        )
        assert self.package.frozen_policy is not None
        return ResolvedPortfolioAuthorities(
            coverage=coverage,
            candidate_sessions=self.sessions,
            ordered_listing_ids=self.listings,
            eligible_count=len(self.listings),
            strategy_package_id=self.package.strategy_id,
            strategy_package_hash=self.package.package_hash,
            score_source_mode="HISTORICAL_ARRAY_REPLAY",
            policy_identity=self.package.frozen_policy,
            alpha_recipe_hash=self.package.alpha_recipe_hash,
            alpha_evidence_manifest_hash=source_hash,
            risk_recipe_hash=INSTALLED_RISK_DECOMPOSITION_RECIPE.recipe_hash,
            risk_return_surface_hash=_HASH,
            sector_map_hash=_HASH,
            tradability_decision_hash=_HASH,
            execution_outcome_manifest_hash=_HASH,
        )

    def resolve(
        self, *, workspace: Path, spec: PortfolioResearchSpec
    ) -> ResolvedPortfolioExecution:
        del workspace
        frozen = self.source.resolve()
        shared = SimpleNamespace(
            formation_sessions=self.sessions,
            ordered_listing_ids=self.listings,
            decision_eligible=frozen.decision_eligible,
        )
        components = self.source.resolve_components(shared=shared, spec=spec)
        replay = PublicPortfolioReplayWorkspace(
            formation_sessions=self.sessions,
            ordered_listing_ids=self.listings,
            execution_available=self.execution,
            realized_simple_returns=self.returns,
            causal_adv20=self.adv,
            sector_exposure_matrix=np.ones((1, len(self.listings)), dtype=np.float64),
            equal_weight_sector_exposure=np.ones(1, dtype=np.float64),
            passive_returns_by_session={
                day: self.returns[index] for index, day in enumerate(self.sessions)
            },
            capacity_evidence_available=False,
        )
        assert self.package.frozen_policy is not None
        return ResolvedPortfolioExecution(
            workspace=replay,
            components=components.components,
            score_receipt_hashes=components.score_receipt_hashes,
            report_risk_attributions=self.attributions,
            strategy_package_hash=self.package.package_hash,
            score_source_mode="HISTORICAL_ARRAY_REPLAY",
            policy_recipe_hash=self.package.frozen_policy.policy_recipe_hash,
            policy_catalog_hash=self.package.frozen_policy.policy_catalog_hash,
            policy_adapter_binding_hash=self.package.frozen_policy.policy_adapter_binding_hash,
            alpha_recipe_hash=self.package.alpha_recipe_hash,
            alpha_evidence_manifest_hash=self.source.evidence_identity_hash,
            risk_recipe_hash=INSTALLED_RISK_DECOMPOSITION_RECIPE.recipe_hash,
            risk_return_surface_hash=_HASH,
            sector_map_hash=_HASH,
            tradability_decision_hash=_HASH,
            execution_outcome_manifest_hash=_HASH,
            reference_mark=frozen.reference_mark,
        )


def _write_contract(workspace: Path, relative: str, value: object) -> Path:
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _sealed_score_source(workspace: Path):
    """Publish real packed lifecycle lanes and their synthetic Alpha training source."""
    cutoff = date(2026, 8, 11)
    all_points = tuple(
        point
        for point in planned_local_qa_schedule(date(2023, 1, 1), cutoff)
        if point.formation_session <= cutoff
    )
    points = tuple(all_points[-_HISTORY:])
    if len(points) != _HISTORY:
        raise ValueError("activation_review.synthetic_schedule_too_short")
    sessions = tuple(value.formation_session for value in points)
    entries = tuple(value.entry_session for value in points)
    endings = tuple(value.holding_end_session for value in points)
    listings = tuple(f"listing-{index:03d}" for index in range(80))
    symbols = tuple(f"QA{index:03d}" for index in range(len(listings)))
    rows = np.arange(len(sessions), dtype=np.float64)[:, None]
    columns = np.arange(len(listings), dtype=np.float64)[None, :]
    scores = np.ascontiguousarray(np.sin(rows / 7.0 + columns / 5.0), dtype=np.float64)
    decision = np.ones(scores.shape, dtype=np.bool_)
    execution = np.ones(scores.shape, dtype=np.bool_)
    returns = np.ascontiguousarray(0.004 * np.sin(rows / 11.0 + columns / 13.0), dtype=np.float64)
    adv = np.full(scores.shape, 1_000_000.0, dtype=np.float64)
    marks = np.ascontiguousarray(100.0 * np.exp(rows / 100_000.0 + columns / 10_000.0))
    curves = np.tile(np.linspace(0.002, 0.0001, 20), (len(sessions), 1)).astype(
        np.float64, copy=False
    )
    latest = np.maximum(np.arange(len(sessions), dtype=np.int64) - 2, 0)
    transition = PortfolioStateTransitionBinding.create(
        transition_implementation_hash=_HASH,
        rebalance_clock=EveryFormationClock().binding,
        reference_mark_method="MARKED_TO_MARKET_AT_CLOSE_T",
        mark_manifest_ref="synthetic-activation-review",
        mark_surface_hash=_HASH,
        mark_epoch_hash=_HASH,
        mark_price_basis="synthetic-close",
        mark_source_watermark_hash=_HASH,
        mark_availability_policy_hash=_HASH,
        mark_availability_policy_id="synthetic-activation-review",
    )
    artifact_store = PortfolioResearchArtifactStore(
        workspace / "artifacts/research-strategy-inputs/synthetic"
    )
    scores_hash = artifact_store.publish_array(category=LANES, values=scores)
    decision_hash = artifact_store.publish_array(category=LANES, values=decision)
    execution_hash = artifact_store.publish_array(category=LANES, values=execution)
    returns_hash = artifact_store.publish_array(category=LANES, values=returns)
    adv_hash = artifact_store.publish_array(category=LANES, values=adv)
    marks_hash = artifact_store.publish_array(category=LANES, values=marks)
    curve_hash = artifact_store.publish_array(category=LANES, values=curves)
    latest_hash = artifact_store.publish_array(category=LANES, values=latest)
    component_evidence = LifecycleScoreEvidence(
        component_id=_COMPONENT.component_id,
        task_id=uuid4(),
        program_hash=_HASH,
        receipt_hash=_HASH,
        component_recipe_hash=_COMPONENT.recipe_hash,
        lifecycle_hash=_LIFECYCLE.content_hash,
        formation_sessions=sessions,
        scores_hash=scores_hash,
    )
    authority = LifecyclePortfolioAuthority.create(
        input_binding_hash=_HASH,
        preparation_request_hash=_HASH,
        preparation_implementation_hash=_HASH,
        components=(component_evidence,),
        formation_sessions=sessions,
        ordered_listing_ids=listings,
        decision_hash=decision_hash,
        execution_hash=execution_hash,
        returns_hash=returns_hash,
        adv_hash=adv_hash,
        marks_hash=marks_hash,
        entry_sessions=entries,
        holding_end_sessions=endings,
        transition=transition,
        tradability_decision_hash=_HASH,
        outcome_snapshot_hash=_HASH,
        risk_return_surface_hash=_HASH,
        sector_map_hash=_HASH,
        curve_hash=curve_hash,
        latest_calibration_hash=latest_hash,
        unavailable_return_policy="require_complete",
    )
    artifact_store.publish(category=CATEGORY, value=authority, identity_field="authority_hash")

    # The activation grant uses only the source's public readback to seal future fit authority.
    formula_values = {
        feature_id: np.ascontiguousarray(
            np.cos(rows / (7.0 + index) + columns / (11.0 + index)), dtype=np.float64
        )
        for index, feature_id in enumerate(_COMPONENT.ordered_feature_ids)
    }
    close = np.ascontiguousarray(100.0 * np.exp(rows / 10_000.0 + columns / 20_000.0))
    alpha_source = FrozenPriceVolumeInputs(
        formation_sessions=sessions,
        ordered_listing_ids=listings,
        sector_by_listing_id={listing: "synthetic-sector" for listing in listings},
        open=np.ascontiguousarray(close * 0.995),
        high=np.ascontiguousarray(close * 1.01),
        low=np.ascontiguousarray(close * 0.99),
        close=close,
        volume=np.full(close.shape, 1_000_000.0, dtype=np.float64),
        market_context_values=np.zeros((len(sessions), 3), dtype=np.float64),
        sector_trend_values=np.zeros((len(sessions), 1), dtype=np.float64),
        source_binding_hash=_HASH,
        formula_values=formula_values,
    )
    alpha_store = AlphaCurrentArtifactStore(workspace / "artifacts")
    snapshot = alpha_store.publish_frozen_observations(
        alpha_source, disposition="SYNTHETIC_INPUT_QA"
    )
    training = publish_training_observations(
        alpha_store,
        observation_hash=snapshot.snapshot_hash,
        target_method_id=_COMPONENT.target_recipe,
        target_authority_hash=_HASH,
        label_available_sessions=tuple(day + timedelta(days=1) for day in sessions),
        targets=np.ascontiguousarray(np.log(alpha_source.close / alpha_source.open)),
        eligible=np.ones(alpha_source.close.shape, dtype=np.bool_),
    )
    horizon = sessions[-1] + timedelta(days=2)
    admission = AlphaModelLifecycleAdmission.create(
        component=_COMPONENT,
        lifecycle=_LIFECYCLE,
        observations_hash=training.content_hash,
        prepared=(),
        initial_children=(),
        formation_start=sessions[0],
        formation_end=horizon,
        maximum_fit_attempts=0,
        environment_hash=_HASH,
        training_factor_ids=tuple(sorted(formula_values)),
        renewal_through=horizon,
    )
    admission_path = _write_contract(
        workspace,
        f"artifacts/alpha-research/current/lifecycle-admissions/{admission.content_hash}.json",
        admission,
    )
    training_binding = ResearchWorkspaceModelTrainingInput(
        component_id=_COMPONENT.component_id,
        input_binding_hash=_HASH,
        source_identity_hash=snapshot.source_binding_hash,
        authority_relative_path=admission_path.relative_to(workspace).as_posix(),
        authority_hash=admission.content_hash,
    )

    attributions = []
    exposures = np.ones((len(listings), 1), dtype=np.float64)
    for session in sessions:
        surface = FactorIdiosyncraticRiskSurface.create(
            recipe=INSTALLED_RISK_DECOMPOSITION_RECIPE,
            formation_session=session,
            ordered_listing_ids=listings,
            ordered_factor_ids=("synthetic-sector",),
            exposures=exposures,
            factor_covariance=np.array([[0.01]], dtype=np.float64),
            idiosyncratic_variance=np.full(len(listings), 0.01, dtype=np.float64),
            conditional_volatility=np.full(len(listings), 0.20, dtype=np.float64),
            producer_identity=f"synthetic-activation-review-{session.isoformat()}",
        )
        attributions.append(
            RiskAttributionProjection.of(
                surface, classification_authority="SYNTHETIC_ACTIVATION_REVIEW"
            )
        )
    package = frozen_book_package(
        recipe=REBOUND_RETURN_BOOK_RECIPE,
        evidence_identity_hash=authority.authority_hash,
    )
    return (
        authority,
        package,
        sessions,
        listings,
        symbols,
        decision,
        execution,
        returns,
        adv,
        tuple(attributions),
        training_binding,
        component_evidence.task_id,
        {
            "document": {
                "experiment": {
                    "data_snapshot_handle": training_binding.source_handle,
                    "output_workspace": f"studies/{_COMPONENT.component_id}",
                }
            }
        },
    )


def _bootstrap_market(
    workspace: Path, *, listings: tuple[str, ...], symbols: tuple[str, ...]
) -> None:
    profile = MarketProfile(
        market_profile_id="us-current-index-research",
        display_name="Synthetic activation review universe",
        market="US",
        currency="USD",
        calendar_id="XNAS+XNYS",
        provider="fixture",
        daily_price_basis="split_adjusted",
        manifest_as_of=_NOW.date(),
        data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
    )
    manifest = UniverseManifest(
        manifest_id="us-current-index-research:acquisition:synthetic",
        profile=profile,
        listings=tuple(
            ManifestListing(
                listing_id=listing,
                symbol=symbol,
                mic="XNAS",
                provider_symbol=symbol,
            )
            for listing, symbol in zip(listings, symbols, strict=True)
        ),
        revision_sha256=_HASH,
        membership_fingerprint=str(canonical_hash(listings)),
        qualification_policy_hash=str(canonical_hash("synthetic activation QA qualification")),
    )
    market = MarketDataRepository(workspace)
    candidate_document = {
        "kind": "synthetic-activation-review-candidates",
        "listing_ids": listings,
        "symbols": symbols,
    }
    candidate_hash = str(canonical_hash(candidate_document))
    market.admit_current_universe_onboarding(
        manifest,
        onboarding_id="activation-review-synthetic-onboarding",
        candidate_manifest_hash=candidate_hash,
        candidate_manifest_document=candidate_document,
        history_start=date(2023, 1, 1),
        as_of_session=_NOW.date(),
        calendar_by_listing_id={listing: "XNAS" for listing in listings},
        observed_at=_NOW,
    )
    qualified = build_quality_filtered_research_manifest(
        manifest,
        eligible_listing_ids=listings,
        quality_admission_hash=_HASH,
        qualification_obligations=(qualification_obligation("fixture", _HASH),),
    )
    market.complete_current_universe_onboarding(
        onboarding_id="activation-review-synthetic-onboarding",
        research_manifest=qualified,
        quality_admission_hash=_HASH,
        observed_at=_NOW,
    )


@contextmanager
def activation_review_host(
    workspace: Path,
    *,
    review_authority_for_report: Callable[[PortfolioDeclaredPathReport], object] | None = None,
) -> Iterator[ActivationReviewHost]:
    """Yield a live Host and a completed 522-formation RETURN book.

    ``review_authority_for_report`` receives the actual, completed Portfolio report before the
    Host starts, so it can seal the exact review input and be supplied at construction time.
    Synthetic study metadata is connected through StrategyActivation's public read ports; its
    book, grant, seed, checkpoint, and manifest-write logic remains the installed owner path.
    """
    root = workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)
    prior_network = os.environ.get("ALPHALATTICE_NETWORK_DISABLED")
    os.environ["ALPHALATTICE_NETWORK_DISABLED"] = "1"
    live: LocalPortfolioWebSession | None = None
    try:
        (
            authority,
            package,
            sessions,
            listings,
            symbols,
            decision,
            execution,
            returns,
            adv,
            attributions,
            training_binding,
            study_task_id,
            study_document,
        ) = _sealed_score_source(root)
        artifact_path = (
            f"artifacts/research-strategy-inputs/synthetic/portfolio-strategy-lab/"
            f"{CATEGORY}/{authority.authority_hash}.json"
        )
        manifest = ResearchWorkspaceManifest.create(
            workspace_id="synthetic-activation-review",
            default_strategy_package_id=None,
            default_score_source_mode=None,
            strategy_artifacts=(
                ResearchWorkspaceArtifact(artifact_key=ARTIFACT_KEY, relative_path=artifact_path),
            ),
            model_training_inputs=(training_binding,),
            data_update=installed_data_update_binding(root),
            strategy_installation="NON_DEFAULT_RESEARCH",
        )
        publish_research_workspace_manifest(root, manifest)
        _bootstrap_market(root, listings=listings, symbols=symbols)
        resolver = _ActivationReviewResolver(
            workspace=root,
            authority=authority,
            package=package,
            sessions=sessions,
            listings=listings,
            decision=decision,
            execution=execution,
            returns=returns,
            adv=adv,
            attributions=attributions,
        )
        spec = PortfolioResearchSpec.create(
            strategy_package_id=PACKAGE,
            score_source_mode="HISTORICAL_ARRAY_REPLAY",
            top_k=REBOUND_RETURN_BOOK_RECIPE.top_k,
            tranches=REBOUND_RETURN_BOOK_RECIPE.tranches,
            exit_rank=REBOUND_RETURN_BOOK_RECIPE.exit_rank,
            weight_rule="mu.iv0",
        )
        from alphalattice.control.product_host.composition.portfolio_application import (
            PLAN_FIELDS,
            PortfolioResearchApplication,
        )
        from alphalattice.control.product_host.composition.research_workspace import (
            manifest_fields_hash,
        )

        with WorkspaceApplicationSession.acquire(root) as application_session:
            application = PortfolioResearchApplication(
                workspace_id=manifest.workspace_id,
                workspace=root,
                manifest_binding=lambda: manifest_fields_hash(manifest, PLAN_FIELDS),
                session=application_session,
                resolver=resolver,
                clock=lambda: _NOW,
            )
            planned = application.plan(spec)
            completed = application.run(spec=spec, planned=planned)
            task_id = completed.pipeline_manifest.task_id
            result = completed.result
            report = application.ledger.load_report(result.report_hash)
        result_hash = result.result_hash
        review_authority = (
            None if review_authority_for_report is None else review_authority_for_report(report)
        )
        live = LocalPortfolioWebSession(
            workspace=root,
            workspace_manifest=manifest,
            resolver=resolver,
            review_authority=review_authority,  # type: ignore[arg-type]
            clock=lambda: _NOW,
        )
        live.start()
        assert (
            live.session is not None
            and live.operations is not None
            and live.application is not None
        )
        original_owner = live.operations.activations
        assert original_owner is not None

        def reader(requested_task_id: UUID) -> dict[str, object]:
            if requested_task_id != study_task_id:
                raise ValueError("activation_review.synthetic_study_task_absent")
            return study_document

        activation = StrategyActivation(
            session=live.session,
            manifest=lambda: live.operations.manifests.current,
            packages=lambda: {
                value.strategy_id: value
                for value in live.application.resolver.installed_packages().values()
            },
            ledger=live.application.ledger,
            read_experiment=reader,
            read_information=reader,
            read_review=original_owner.read_review,
            clock=live.clock,
            hold=original_owner.hold,
        )
        live.operations.activations = activation
        # The public result callback also sees the same real Task and report identities.
        assert live.session.task_control_registry.task(task_id).task_id == task_id
        yield ActivationReviewHost(
            live=live,
            activation=activation,
            task_id=task_id,
            result_hash=result_hash,
            last_session=sessions[-1],
            report=report,
        )
    finally:
        if live is not None:
            live.stop()
        if prior_network is None:
            os.environ.pop("ALPHALATTICE_NETWORK_DISABLED", None)
        else:
            os.environ["ALPHALATTICE_NETWORK_DISABLED"] = prior_network


__all__ = ["PACKAGE", "ActivationReviewHost", "activation_review_host"]
