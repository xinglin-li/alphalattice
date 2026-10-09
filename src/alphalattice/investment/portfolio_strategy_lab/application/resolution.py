"""The public Portfolio path's Program compiler and the resolvers of its shared lanes.

Moved from the Product Host (O3): what a Program binds and how the lanes every installed
package shares are read from the owners' published artifacts are the lab's; the Host
constructs the resolvers for a workspace and routes their answers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

from alphalattice.capabilities.portfolio_backtesting.benchmark import (
    PortfolioBenchmarkBoundaryError,
)
from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioBacktestWorkspace,
)
from alphalattice.capabilities.portfolio_inputs.benchmark import (
    build_portfolio_benchmark_surface,
)
from alphalattice.capabilities.portfolio_inputs.signed_score.execution_events import (
    execution_events_from_schedule,
)
from alphalattice.capabilities.portfolio_inputs.tradability.publication import (
    CurrentTradabilityDataService,
)
from alphalattice.capabilities.portfolio_inputs.tradability.readback import (
    read_tradability_matrices,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.contracts import (
    CausalExecutionOutcomeManifest,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import (
    SectorRevisionMap,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    sector_history_as_of,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID,
    OwnerCoverage,
    PortfolioExecutionProgram,
    PortfolioResearchSpec,
    PortfolioSupportCoverage,
)
from alphalattice.investment.portfolio_strategy_lab.application.executor import (
    ResolvedPortfolioExecution,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    FrozenSharedMarketInputs,
    FrozenSharedMarketSource,
    FrozenStrategyPackage,
    InstalledFrozenStrategyCatalog,
    PackagePolicyIdentity,
    ScoreSourceMode,
    ScoreSupport,
    SelectedStrategy,
    SharedInputOwner,
    SharedPortfolioAuthorities,
    SharedPortfolioInputs,
    StrategyPackageError,
    StrategyScoreSource,
)
from alphalattice.investment.portfolio_strategy_lab.inputs.shared_lanes import (
    PortfolioResearchCompositionError,
    outcome_returns,
    sector_exposure_lanes,
)
from alphalattice.investment.portfolio_strategy_lab.policies.buffered_rank_return import (
    CAUSAL_RANK_MU_LOOKBACK,
    causal_rank_mu_maturity_support,
)
from alphalattice.investment.risk_research.surfaces.decomposition import (
    INSTALLED_RISK_DECOMPOSITION_RECIPE,
)
from alphalattice.investment.risk_research.surfaces.producer import (
    RiskDecompositionInputs,
    RiskSurfaceProducer,
)
from alphalattice.investment.risk_research.surfaces.returns import (
    CausalRiskReturnReader,
    RiskReturnArtifactStore,
)
from alphalattice.kernel.quant.sector_history import SectorHistory
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import recorded_origin
from alphalattice.kernel.shared_kernel.source_identity import (
    switched_source_identity,
)

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _ReplayMandate:
    bootstrap_resamples: int = 0


@dataclass(frozen=True, slots=True)
class PublicPortfolioReplayWorkspace:
    """Only the lanes shared Backtesting consumes; no Data/Feature runtime."""

    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    execution_available: BoolArray
    realized_simple_returns: FloatArray
    causal_adv20: FloatArray
    sector_exposure_matrix: FloatArray
    equal_weight_sector_exposure: FloatArray
    passive_returns_by_session: dict[date, FloatArray]
    capacity_evidence_available: bool = True
    mandate: _ReplayMandate = field(default_factory=_ReplayMandate)


@dataclass(frozen=True, slots=True)
class ResolvedPortfolioAuthorities:
    """What PLAN may know without running anything.

    Every field here comes from reading a manifest, an axis or a receipt. No
    score is replayed, no Risk surface is built, no policy path is walked. That
    is the whole point: PLAN has to answer "what will this cost and is it already
    done" without doing the thing it is estimating.
    """

    coverage: PortfolioSupportCoverage
    candidate_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    eligible_count: int
    strategy_package_id: str
    strategy_package_hash: str
    score_source_mode: ScoreSourceMode
    policy_identity: PackagePolicyIdentity
    """What this request compiles to under the selected package.

    Resolved here rather than in the compiler because only the package knows
    whether its holdings controls are frozen facts or admitted request values.
    It is deliberately *not* part of `authorities_hash`: the policy identities are
    Program fields in their own right and are already covered by the Program's
    numerical input assembly, so hashing them twice would make one change move
    two identities.
    """

    alpha_recipe_hash: str
    alpha_evidence_manifest_hash: str
    risk_recipe_hash: str
    risk_return_surface_hash: str
    sector_map_hash: str
    tradability_decision_hash: str
    execution_outcome_manifest_hash: str
    public_path_source_hash: str = field(default_factory=lambda: _public_path_source_hash())
    """The implementation bytes this Program's numbers depend on.

    The generic path's own files by default. A resolver composing an installed
    package binds the package's source identity beside it, so an edit to either
    half rotates the Program and neither half has to name the other's files.
    """
    sector_history_hash: str | None = None
    """The Sector history's identity while a reclassification is in force over the book's
    listings; None otherwise, so every Program before one is what it was."""

    def __post_init__(self) -> None:
        """Require exact common support and nonempty admitted listing/source authority.

        Raises:
            PortfolioResearchCompositionError: Session order/count/endpoints contradict common
                coverage, listing/eligible population is invalid or source hash syntax differs.
        """
        sessions = self.candidate_sessions
        listings = self.ordered_listing_ids
        if (
            not sessions
            or tuple(sorted(sessions)) != sessions
            or len(set(sessions)) != len(sessions)
            or self.coverage.common_watermark_start != sessions[0]
            or self.coverage.common_watermark_end != sessions[-1]
            or self.coverage.common_session_count != len(sessions)
        ):
            raise PortfolioResearchCompositionError(
                "portfolio_application.authority_session_axis_invalid"
            )
        if (
            not listings
            or len(set(listings)) != len(listings)
            or not 0 < self.eligible_count <= len(listings)
        ):
            raise PortfolioResearchCompositionError(
                "portfolio_application.authority_listing_axis_invalid"
            )
        if len(self.public_path_source_hash) != 64 or any(
            value not in "0123456789abcdef" for value in self.public_path_source_hash
        ):
            raise PortfolioResearchCompositionError(
                "portfolio_application.authority_source_identity_invalid"
            )

    @property
    def authorities_hash(self) -> str:
        """One identity over every authority a RUN must resolve to the same values."""
        return str(
            canonical_hash(
                {
                    "coverage_hash": self.coverage.coverage_hash,
                    "candidate_sessions_hash": canonical_hash(
                        tuple(value.isoformat() for value in self.candidate_sessions)
                    ),
                    "candidate_session_count": len(self.candidate_sessions),
                    "candidate_first": self.candidate_sessions[0].isoformat(),
                    "candidate_last": self.candidate_sessions[-1].isoformat(),
                    "ordered_listing_ids_hash": canonical_hash(self.ordered_listing_ids),
                    "eligible_count": self.eligible_count,
                    "strategy_package_hash": self.strategy_package_hash,
                    "score_source_mode": self.score_source_mode,
                    "alpha_recipe_hash": self.alpha_recipe_hash,
                    "alpha_evidence_manifest_hash": self.alpha_evidence_manifest_hash,
                    "risk_recipe_hash": self.risk_recipe_hash,
                    "risk_return_surface_hash": self.risk_return_surface_hash,
                    "sector_map_hash": self.sector_map_hash,
                    "tradability_decision_hash": self.tradability_decision_hash,
                    "execution_outcome_manifest_hash": self.execution_outcome_manifest_hash,
                    "public_path_source_hash": self.public_path_source_hash,
                    **(
                        {"sector_history_hash": self.sector_history_hash}
                        if self.sector_history_hash is not None
                        else {}
                    ),
                }
            )
        )

    def require_resolution(self, resolved: ResolvedPortfolioExecution) -> None:
        """Fail closed when the numerical read differs from the admitted read.

        `resolve_authorities()` and `resolve()` deliberately have different cost,
        but they are not different authorities. Current pointers or corrected
        manifests moving between the two reads must stop before a Program or an
        index is published under the stale PLAN receipt.
        """
        expected = (
            self.alpha_recipe_hash,
            self.alpha_evidence_manifest_hash,
            self.risk_recipe_hash,
            self.risk_return_surface_hash,
            self.sector_map_hash,
            self.tradability_decision_hash,
            self.execution_outcome_manifest_hash,
            self.sector_history_hash,
        )
        actual = (
            resolved.alpha_recipe_hash,
            resolved.alpha_evidence_manifest_hash,
            resolved.risk_recipe_hash,
            resolved.risk_return_surface_hash,
            resolved.sector_map_hash,
            resolved.tradability_decision_hash,
            resolved.execution_outcome_manifest_hash,
            resolved.sector_history_hash,
        )
        resolved_sessions = tuple(resolved.workspace.formation_sessions)
        if (
            actual != expected
            or tuple(resolved.workspace.ordered_listing_ids) != self.ordered_listing_ids
            or resolved_sessions != self.candidate_sessions
        ):
            raise PortfolioResearchCompositionError(
                "portfolio_application.numerical_resolution_authority_mismatch"
            )


class PortfolioExecutionResolver(Protocol):
    """Expose installed strategy identity and deterministic authority/execution resolution."""

    @property
    def strategy_catalog_hash(self) -> str:
        """Read the installed strategy declaration catalog identity.

        Returns:
            Exact installed strategy catalog hash.
        """
        ...

    def resolve_authorities(
        self, *, workspace: Path, spec: PortfolioResearchSpec
    ) -> ResolvedPortfolioAuthorities:
        """Resolve selected strategy and shared input coverage before execution.

        Args:
            workspace: Caller-owned workspace root.
            spec: Admitted strategy/mode and research controls.

        Returns:
            Resolved common support, listing population and exact strategy/input/source bindings.
        """
        ...

    def resolve(
        self, *, workspace: Path, spec: PortfolioResearchSpec
    ) -> ResolvedPortfolioExecution:
        """Resolve exact selected component inputs and shared execution/Risk/report authority.

        Args:
            workspace: Caller-owned workspace root.
            spec: Admitted strategy/mode and research controls.

        Returns:
            Resolved execution inputs with ordered components, formation score receipts and declared
            source/reference/report bindings.

        Raises:
            PortfolioResearchCompositionError: Package selection/required lanes, declared component
                set or formation score-receipt coverage cannot be reconciled.
        """
        ...

    def installed_packages(self) -> Mapping[str, FrozenStrategyPackage]:
        """The package declarations any Program this resolver seals may name.

        Asked rather than injected. A resolver that can compile a Program under a
        package can also say which package that was, and routing the declaration
        any other way would let a report be rebuilt from a stored ledger with no
        idea which strategy produced it.
        """
        ...


class PortfolioResearchCompiler:
    """Seal one exact Program from resolved owners; performs no estimation."""

    def compile(
        self,
        *,
        spec: PortfolioResearchSpec,
        authorities: ResolvedPortfolioAuthorities,
        resolved: ResolvedPortfolioExecution,
        continued_from_state_hash: str | None = None,
    ) -> PortfolioExecutionProgram:
        """Seal one Program, optionally over a book the walk is to continue.

        `continued_from_state_hash` is passed straight through to the Program
        rather than interpreted here: the compiler's job is to record what the
        run is bound to, and an opening book is one of those bindings. It is what
        keeps a continuation from reusing a flat-start ledger.
        """
        authorities.require_resolution(resolved)
        return self._seal(
            spec=spec,
            authorities=authorities,
            sessions=tuple(resolved.workspace.formation_sessions),
            listings=tuple(resolved.workspace.ordered_listing_ids),
            continued_from_state_hash=continued_from_state_hash,
        )

    def compile_from_authorities(
        self,
        *,
        spec: PortfolioResearchSpec,
        authorities: ResolvedPortfolioAuthorities,
        continued_from_state_hash: str | None = None,
    ) -> PortfolioExecutionProgram:
        """Seal the same Program without resolving a single score or surface.

        Every binding a Program carries is already an *authority* fact: the axes,
        the recipe identities, the tradability and outcome manifests. Resolving
        the numerical workspace to rediscover them is what made a first-time
        admission cost a full resolution, and admission is the one call that has
        to be cheap.

        Nothing is skipped by taking this route. The executor still verifies the
        Program against the real resolution before it walks anything -- axis,
        recipe hashes, policy -- so a Program sealed here that did not describe
        the resolution would be refused there rather than run.
        """
        return self._seal(
            spec=spec,
            authorities=authorities,
            sessions=tuple(authorities.candidate_sessions),
            listings=tuple(authorities.ordered_listing_ids),
            continued_from_state_hash=continued_from_state_hash,
        )

    def _seal(
        self,
        *,
        spec: PortfolioResearchSpec,
        authorities: ResolvedPortfolioAuthorities,
        sessions: tuple[date, ...],
        listings: tuple[str, ...],
        continued_from_state_hash: str | None,
    ) -> PortfolioExecutionProgram:
        eligible_count = len(listings)
        ceiling = min(spec.exit_rank_band()[1], eligible_count)
        if spec.exit_rank > ceiling:
            raise PortfolioResearchCompositionError(
                f"portfolio_application.exit_rank_exceeds_eligible_count:{spec.exit_rank}>{ceiling}"
            )
        policy = authorities.policy_identity
        return PortfolioExecutionProgram.create(
            holdings_spec_hash=spec.holdings_spec_hash,
            authorities_hash=authorities.authorities_hash,
            strategy_package_hash=authorities.strategy_package_hash,
            score_source_mode=authorities.score_source_mode,
            policy_recipe_hash=policy.policy_recipe_hash,
            policy_catalog_hash=policy.policy_catalog_hash,
            policy_adapter_binding_hash=policy.policy_adapter_binding_hash,
            alpha_recipe_hash=authorities.alpha_recipe_hash,
            alpha_evidence_manifest_hash=authorities.alpha_evidence_manifest_hash,
            risk_recipe_hash=authorities.risk_recipe_hash,
            risk_return_surface_hash=authorities.risk_return_surface_hash,
            sector_map_hash=authorities.sector_map_hash,
            tradability_decision_hash=authorities.tradability_decision_hash,
            execution_outcome_manifest_hash=authorities.execution_outcome_manifest_hash,
            public_path_source_hash=authorities.public_path_source_hash,
            formation_sessions_hash=canonical_hash(tuple(value.isoformat() for value in sessions)),
            ordered_listing_ids_hash=canonical_hash(listings),
            formation_start=sessions[0],
            formation_end=sessions[-1],
            formation_count=len(sessions),
            listing_count=len(listings),
            execution_clock="EVERY_FORMATION",
            sector_forecast_disposition="SECTOR_FORECAST_NOT_CONSUMED",
            continued_from_state_hash=continued_from_state_hash,
        )


PUBLIC_PATH_ROLE: Final = "portfolio_strategy_lab.public_path"
"""The role whose recorded moves the public path's source identity follows."""


def public_path_rule_value() -> str:
    """The generic public path's source identity as its files hash now.

    The implementation bytes that can change the public result, the generic path only: which
    strategy modules also matter is the installed strategy owner's to say, and it says so
    through the binding's own source identity, composed in by `_public_path_source_hash`.

    Returns:
        The identity; the readout's value for the public path's role.
    """
    root = Path(__file__).resolve().parents[4]
    relative = (
        "capabilities/portfolio_backtesting/active_metrics.py",
        "capabilities/portfolio_backtesting/contracts.py",
        "capabilities/portfolio_backtesting/segments.py",
        "capabilities/portfolio_inputs/signed_score/execution_events.py",
        "capabilities/portfolio_inputs/tradability/readback.py",
        "investment/alpha_research/scores/product_replay.py",
        "investment/portfolio_strategy_lab/application/contracts.py",
        "investment/portfolio_strategy_lab/application/controls.py",
        "investment/portfolio_strategy_lab/application/executor.py",
        "investment/portfolio_strategy_lab/application/resolution.py",
        "investment/portfolio_strategy_lab/application/strategy_package.py",
        "investment/portfolio_strategy_lab/application/tranche_book_execution.py",
        "investment/portfolio_strategy_lab/policies/buffered_rank_return.py",
        "investment/portfolio_strategy_lab/policies/tranche_book.py",
        "investment/portfolio_strategy_lab/reporting/risk.py",
        "investment/portfolio_strategy_lab/reporting/static.py",
        "investment/portfolio_strategy_lab/reporting/units.py",
        "investment/risk_research/surfaces/decomposition.py",
        "investment/risk_research/surfaces/producer.py",
    )
    return str(
        switched_source_identity(
            {
                value.removesuffix(".py").replace("/", "."): root / "alphalattice" / value
                for value in relative
            },
            semantic_owner="public_portfolio_declared_path",
            numerical_role="program_execution_and_report",
        )
    )


def _public_path_source_hash(*, installed_source_hash: str | None = None) -> str:
    """Bind the implementation bytes that can change the public result.

    The generic path's identity is held at the value its recorded moves lead from, so a
    move recorded as keeping the numbers leaves every Program sealed before it current, and the
    installed strategy owner's source identity is composed in when a selection is resolved.
    """
    generic = recorded_origin(PUBLIC_PATH_ROLE, public_path_rule_value())
    if installed_source_hash is None:
        return generic
    return str(
        canonical_hash(
            {
                "generic_path_source_hash": generic,
                "installed_strategy_source_hash": installed_source_hash,
            }
        )
    )


def _risk_surface(artifact_root: Path):  # type: ignore[no-untyped-def]
    store = RiskReturnArtifactStore(artifact_root)
    manifests = tuple(sorted((store.root / "manifests").glob("*.json")))
    if not manifests:
        # None where the shared lanes are read is said apart from several.
        raise PortfolioResearchCompositionError("portfolio_application.risk_return_surface_absent")
    if len(manifests) != 1:
        raise PortfolioResearchCompositionError(
            "portfolio_application.risk_return_surface_not_unique"
        )
    return store.load_manifest(manifests[0].stem)


def _sector_map(artifact_root: Path, *, manifest_revision: str) -> SectorRevisionMap:
    store = PanelClosureArtifactStore(ArtifactResolver(artifact_root))
    matches: list[SectorRevisionMap] = []
    for path in sorted((store.root / "sector-maps").glob("*.json")):
        value = store.load_model(
            category="sector-maps",
            content_hash=path.stem,
            model=SectorRevisionMap,
        )
        if value.manifest_revision == manifest_revision:
            matches.append(value)
    if len(matches) != 1:
        raise PortfolioResearchCompositionError(
            "portfolio_application.sector_revision_map_not_unique"
        )
    return matches[0]


def _sector_history(
    artifact_root: Path, classification: SectorRevisionMap, listings: tuple[str, ...]
) -> SectorHistory:
    """The Sector each formation reads over the book's listings, as of its revision."""
    try:
        history = sector_history_as_of(
            PanelClosureArtifactStore(ArtifactResolver(artifact_root)),
            classification.sector_revision,
        )
    except ValueError as error:
        raise PortfolioResearchCompositionError(
            "portfolio_application.sector_history_unavailable"
        ) from error
    return history.subset(listings)


def _sector_history_hash(history: SectorHistory) -> str | None:
    """A history's identity while it holds a reclassification; None otherwise."""
    return history.identity if history.reclassifications else None


def _execution_authority(
    artifact_root: Path,
    *,
    listings: tuple[str, ...],
) -> tuple[CausalExecutionOutcomeDevelopmentReader, str, CausalExecutionOutcomeManifest]:
    reader = CausalExecutionOutcomeDevelopmentReader(artifact_root)
    manifests: list[CausalExecutionOutcomeManifest] = []
    root = artifact_root / "data-operations" / "execution-outcomes" / "manifests"
    for path in sorted(root.glob("*.json")):
        value = reader.load_manifest(path.stem)
        if set(listings).issubset(value.listing_ids):
            manifests.append(value)
    if not manifests:
        raise PortfolioResearchCompositionError(
            "portfolio_application.execution_outcome_authority_absent"
        )
    latest_date = max(value.market_as_of for value in manifests)
    latest = tuple(value for value in manifests if value.market_as_of == latest_date)
    if len(latest) != 1:
        raise PortfolioResearchCompositionError(
            "portfolio_application.execution_outcome_authority_ambiguous"
        )
    manifest = latest[0]
    return reader, reader.manifest_uri(manifest.snapshot_hash), manifest


def _sector_lanes(
    classification: SectorRevisionMap,
    *,
    listings: tuple[str, ...],
    history: SectorHistory,
    sessions: tuple[date, ...],
) -> tuple[FloatArray, FloatArray]:
    # A history with a reclassification is read per formation, by Sector name.
    if history.reclassifications:
        return sector_exposure_lanes(history, listings=listings, sessions=sessions)
    entries = {value.listing_id: value for value in classification.entries}
    return sector_exposure_lanes(
        {value: entries[value].sector_key or entries[value].sector_name for value in listings},
        listings=listings,
    )


@dataclass(frozen=True, slots=True)
class _TradabilityAuthority:
    service: CurrentTradabilityDataService
    decision: object
    lineage: object


def _common_axis(
    *,
    support: ScoreSupport,
    risk_sessions: tuple[date, ...],
    risk_listings: tuple[str, ...],
    classification: SectorRevisionMap,
    tradability: _TradabilityAuthority,
) -> tuple[tuple[str, ...], tuple[date, ...]]:
    """The one intersection both resolution halves use.

    Written once because `PLAN` and `RUN` must agree about support exactly; two
    copies of this filter would be two chances for a preview to describe an axis
    the run does not produce.

    The selected package's own support enters as one more constraint rather than
    as the axis itself. A package that declares none is answered wherever the
    shared lanes agree, which is what stops one strategy's evidence window from
    silently bounding another's.
    """
    risk_session_set = set(risk_sessions)
    risk_listing_set = set(risk_listings)
    classified = {entry.listing_id for entry in classification.entries}
    decision = tradability.decision
    candidate_listings = support.ordered_listing_ids or tuple(risk_listings)
    listings = tuple(
        value
        for value in candidate_listings
        if value in risk_listing_set
        and value in decision.ordered_listing_ids  # type: ignore[attr-defined]
        and value in classified
    )
    tradability_sessions = set(decision.formation_sessions)  # type: ignore[attr-defined]
    candidate_sessions = support.formation_sessions or tuple(risk_sessions)
    common_sessions = tuple(
        value
        for value in candidate_sessions
        if value in risk_session_set and value in tradability_sessions
    )
    if support.requires_causal_rank_mu_maturity and len(common_sessions) <= CAUSAL_RANK_MU_LOOKBACK:
        # The bound belongs to the rank-`mu` policy, which averages over that
        # many matured prior formations, so it is read from that owner rather
        # than restated -- and applied only to a package that reads the curve.
        # Applying it to every package made a policy parameter look like a
        # property of the workspace axis, and refused a package that reads no
        # curve at all whenever its lane was shorter than one historical year.
        raise PortfolioResearchCompositionError("portfolio_application.full_support_insufficient")
    if not common_sessions:
        raise PortfolioResearchCompositionError("portfolio_application.common_session_axis_empty")
    if not listings:
        raise PortfolioResearchCompositionError("portfolio_application.common_listing_axis_empty")
    return listings, common_sessions


@dataclass(frozen=True, slots=True)
class _ResolvedAxis:
    """One read of every shared authority, before anything numerical is built."""

    artifact_root: Path
    risk_surface: Any
    risk_sessions: tuple[date, ...]
    classification: SectorRevisionMap
    sectors: SectorHistory
    """The Sector each formation reads: the classification's history over the listings."""
    tradability: Any | None
    """The `_TradabilityAuthority` record, untyped here because its lineage and
    decision views come from an owner that publishes no static contract."""

    listings: tuple[str, ...]
    common_sessions: tuple[date, ...]
    outcome_reader: Any | None
    outcome_ref: Any | None
    outcome_manifest: Any | None
    execution_events: Any | None
    frozen_market_source: FrozenSharedMarketSource | None
    executable_sessions: tuple[date, ...]
    maturity: Any


SHARED_INPUT_LANES: Final = frozenset(
    {
        "MARKET_REALIZED_RETURN",
        "TRADABILITY_DECISION_ELIGIBILITY",
        "CAUSAL_EXECUTION_OUTCOME",
        "SECTOR_CLASSIFICATION",
        "ELIGIBLE_UNIVERSE_EQUAL_WEIGHT_BENCHMARK",
        "RISK_ALLOCATION_AND_ATTRIBUTION",
        "RISK_ATTRIBUTION_REPORT_ONLY",
        "CAUSAL_RANK_RETURN_CURVE",
    }
)
"""Exactly what this resolver puts in `SharedPortfolioInputs`.

Named once here so a package that declares a lane nobody resolves is refused
rather than discovering the absence as a missing key mid-walk."""


class SharedPortfolioInputResolver:
    """Resolve the lanes every installed package shares, and nothing else.

    Market, Tradability, Outcome, classification, benchmark and the report-only
    Risk decomposition, from immutable artifacts rather than a workspace DB. It
    replays no scores, knows no strategy and holds no package: the selected
    package supplies its own score lane over what this returns, which is why an
    unselected strategy's models are never opened.
    """

    def _axis(self, *, workspace: Path, support: ScoreSupport) -> _ResolvedAxis:
        """Everything both halves of resolution must agree about, read once.

        `PLAN` and `RUN` disagreeing about the axis is the one failure this
        owner cannot report, because each would be internally consistent. So
        there is one read, and both halves call it.
        """
        binding = support.shared_artifacts
        artifact_root = (
            self._artifact_root(workspace) if binding is None else binding.artifact_root.resolve()
        )
        if not artifact_root.is_relative_to(workspace.resolve()):
            raise PortfolioResearchCompositionError(
                "portfolio_application.shared_artifact_root_escape"
            )
        risk_surface = (
            _risk_surface(artifact_root)
            if binding is None
            else RiskReturnArtifactStore(artifact_root).load_manifest(
                binding.risk_return_surface_hash
            )
        )
        risk_sessions = CausalRiskReturnReader(artifact_root).available_sessions(risk_surface)
        classification = (
            _sector_map(
                artifact_root, manifest_revision=risk_surface.epoch.universe_manifest_revision
            )
            if binding is None
            else PanelClosureArtifactStore(ArtifactResolver(artifact_root)).load_model(
                category="sector-maps",
                content_hash=binding.sector_map_hash,
                model=SectorRevisionMap,
            )
        )
        if classification.manifest_revision != risk_surface.epoch.universe_manifest_revision:
            raise PortfolioResearchCompositionError(
                "portfolio_application.shared_sector_epoch_mismatch"
            )
        frozen = support.frozen_shared_market_source
        if frozen is not None:
            candidate_listings = support.ordered_listing_ids or frozen.ordered_listing_ids
            risk_listing_set = set(risk_surface.epoch.ordered_listing_ids)
            classified = {entry.listing_id for entry in classification.entries}
            listings = tuple(
                value
                for value in candidate_listings
                if value in risk_listing_set
                and value in frozen.ordered_listing_ids
                and value in classified
            )
            risk_session_set = set(risk_sessions)
            candidate_sessions = support.formation_sessions or frozen.formation_sessions
            common_sessions = tuple(
                value
                for value in candidate_sessions
                if value in risk_session_set and value in frozen.formation_sessions
            )
            if binding is not None and (
                common_sessions != support.formation_sessions
                or listings != support.ordered_listing_ids
            ):
                raise PortfolioResearchCompositionError(
                    "portfolio_application.bound_shared_axis_incomplete"
                )
            if not common_sessions or not listings:
                # Which side is empty, in counts: the package's frozen axis against the
                # Risk surface and the sector map.
                raise PortfolioResearchCompositionError(
                    "portfolio_application.frozen_shared_axis_empty:"
                    f"sessions={len(common_sessions)},listings={len(listings)},"
                    f"package_listings={len(candidate_listings)},"
                    f"risk_listings={len(risk_listing_set)}"
                )
            return _ResolvedAxis(
                artifact_root=artifact_root,
                risk_surface=risk_surface,
                risk_sessions=tuple(risk_sessions),
                classification=classification,
                sectors=_sector_history(artifact_root, classification, listings),
                tradability=None,
                listings=listings,
                common_sessions=common_sessions,
                outcome_reader=None,
                outcome_ref=None,
                outcome_manifest=None,
                execution_events=None,
                executable_sessions=common_sessions,
                maturity=None,
                frozen_market_source=frozen,
            )
        tradability = self._tradability(artifact_root)
        listings, common_sessions = _common_axis(
            support=support,
            risk_sessions=risk_sessions,
            risk_listings=risk_surface.epoch.ordered_listing_ids,
            classification=classification,
            tradability=tradability,
        )
        outcome_reader, outcome_ref, outcome_manifest = _execution_authority(
            artifact_root, listings=listings
        )
        execution_events = execution_events_from_schedule(
            table=outcome_reader.read_development_schedule(outcome_ref),
            authority=outcome_reader,
            manifest_ref=outcome_ref,
            sessions=common_sessions,
        )
        maturity = causal_rank_mu_maturity_support(execution_events)
        return _ResolvedAxis(
            artifact_root=artifact_root,
            risk_surface=risk_surface,
            risk_sessions=tuple(risk_sessions),
            classification=classification,
            sectors=_sector_history(artifact_root, classification, listings),
            tradability=tradability,
            listings=listings,
            common_sessions=common_sessions,
            outcome_reader=outcome_reader,
            outcome_ref=outcome_ref,
            outcome_manifest=outcome_manifest,
            execution_events=execution_events,
            executable_sessions=(
                maturity.formation_sessions
                if support.requires_causal_rank_mu_maturity
                else common_sessions
            ),
            maturity=maturity,
            frozen_market_source=None,
        )

    def authorities(self, *, workspace: Path, support: ScoreSupport) -> SharedPortfolioAuthorities:
        """Resolve support and identities only. Nothing numerical happens here.

        The expensive half -- a Risk surface per formation and whatever scoring
        the selected package does -- is deliberately not reached. What is left is
        manifest and axis reads, which is what lets ``PLAN`` answer honestly
        before ``RUN`` is authorised.
        """
        axis = self._axis(workspace=workspace, support=support)
        tradability = axis.tradability
        common_sessions = axis.common_sessions
        maturity = axis.maturity
        executable_sessions = axis.executable_sessions
        owners = [
            OwnerCoverage.of(
                owner_id=support.owner_id,
                lane=support.lane,
                sessions=support.formation_sessions or common_sessions,
                identity_hash=support.identity_hash,
            ),
            OwnerCoverage.of(
                owner_id="risk_return_surface",
                lane="CAUSAL_OPEN_TO_OPEN_RETURN",
                sessions=axis.risk_sessions,
                identity_hash=axis.risk_surface.surface_hash,
            ),
        ]
        if axis.frozen_market_source is None:
            if tradability is None or axis.outcome_manifest is None:
                raise PortfolioResearchCompositionError(
                    "portfolio_application.shared_authority_incomplete"
                )
            owners.extend(
                (
                    OwnerCoverage.of(
                        owner_id="tradability_decision",
                        lane="DECISION_ELIGIBILITY",
                        sessions=tuple(tradability.decision.formation_sessions),
                        identity_hash=tradability.decision.surface_hash,
                    ),
                    OwnerCoverage.of(
                        owner_id="execution_outcome",
                        lane="CAUSAL_EXECUTION_OUTCOME",
                        sessions=common_sessions,
                        identity_hash=axis.outcome_manifest.snapshot_hash,
                    ),
                )
            )
            tradability_hash = tradability.decision.surface_hash
            outcome_hash = axis.outcome_manifest.snapshot_hash
        else:
            frozen = axis.frozen_market_source
            owners.extend(
                (
                    OwnerCoverage.of(
                        owner_id="frozen_tradability_decision",
                        lane="DECISION_ELIGIBILITY",
                        sessions=frozen.formation_sessions,
                        identity_hash=frozen.tradability_decision_hash,
                    ),
                    OwnerCoverage.of(
                        owner_id="frozen_execution_outcome",
                        lane="CAUSAL_EXECUTION_OUTCOME",
                        sessions=frozen.formation_sessions,
                        identity_hash=frozen.execution_outcome_manifest_hash,
                    ),
                )
            )
            tradability_hash = frozen.tradability_decision_hash
            outcome_hash = frozen.execution_outcome_manifest_hash
        if support.requires_causal_rank_mu_maturity:
            if maturity is None:
                raise PortfolioResearchCompositionError(
                    "portfolio_application.causal_rank_mu_authority_absent"
                )
            # A coverage row only the packages that read the curve carry. The
            # successor reads no curve, and inheriting this bound is exactly how
            # it ended up truncated by the predecessor's lookback.
            owners.append(
                OwnerCoverage.of(
                    owner_id="causal_rank_mu",
                    lane=(f"HOLDING_END_READY_{maturity.lookback}_FORMATION_MATURITY"),
                    sessions=maturity.formation_sessions,
                    identity_hash=maturity.support_hash,
                )
            )
        coverage = PortfolioSupportCoverage.create(
            owners=tuple(owners),
            common_watermark_start=executable_sessions[0],
            common_watermark_end=executable_sessions[-1],
            common_session_count=len(executable_sessions),
        )
        return SharedPortfolioAuthorities(
            coverage=coverage,
            candidate_sessions=executable_sessions,
            ordered_listing_ids=axis.listings,
            risk_recipe_hash=INSTALLED_RISK_DECOMPOSITION_RECIPE.recipe_hash,
            risk_return_surface_hash=axis.risk_surface.surface_hash,
            sector_map_hash=axis.classification.map_hash,
            tradability_decision_hash=tradability_hash,
            execution_outcome_manifest_hash=outcome_hash,
            sector_history_hash=_sector_history_hash(axis.sectors),
        )

    def resolve(
        self, *, workspace: Path, spec: PortfolioResearchSpec, support: ScoreSupport
    ) -> SharedPortfolioInputs:
        """Build every shared lane once, over the axis PLAN admitted."""
        workspace_root = workspace.resolve()
        axis = self._axis(workspace=workspace_root, support=support)
        artifact_root = axis.artifact_root
        risk_surface = axis.risk_surface
        risk_sessions = axis.risk_sessions
        classification = axis.classification
        authority = axis.tradability
        listings = axis.listings
        common_sessions = axis.common_sessions
        sessions = axis.executable_sessions
        reference_mark = None
        if axis.frozen_market_source is None:
            if authority is None or axis.outcome_reader is None or axis.outcome_ref is None:
                raise PortfolioResearchCompositionError(
                    "portfolio_application.shared_authority_incomplete"
                )
            tradability = authority.lineage
            decision, execution, adv20 = read_tradability_matrices(
                store=authority.service.store,
                decision=tradability.decision,
                execution=tradability.execution,
                sessions=common_sessions,
                listings=listings,
            )
            outcome_table = axis.outcome_reader.read_development_sessions(
                axis.outcome_ref, common_sessions
            )
            realized = outcome_returns(
                outcome_table,
                sessions=common_sessions,
                listings=listings,
            )
            capacity_evidence_available = True
            tradability_hash = authority.decision.surface_hash
            if axis.outcome_manifest is None:
                raise PortfolioResearchCompositionError(
                    "portfolio_application.execution_outcome_authority_absent"
                )
            outcome_hash = axis.outcome_manifest.snapshot_hash
        else:
            frozen_inputs: FrozenSharedMarketInputs = axis.frozen_market_source.resolve()
            reference_mark = frozen_inputs.reference_mark
            if (
                frozen_inputs.formation_sessions != common_sessions
                or frozen_inputs.ordered_listing_ids != listings
            ):
                raise PortfolioResearchCompositionError(
                    "portfolio_application.frozen_shared_axis_mismatch"
                )
            decision = np.asarray(frozen_inputs.decision_eligible, dtype=np.bool_)
            execution = np.asarray(frozen_inputs.execution_available, dtype=np.bool_)
            realized = np.asarray(frozen_inputs.realized_simple_returns, dtype=np.float64)
            if frozen_inputs.causal_adv20 is None:
                adv20 = np.full(realized.shape, np.nan, dtype=np.float64)
                capacity_evidence_available = False
            else:
                adv20 = np.asarray(frozen_inputs.causal_adv20, dtype=np.float64)
                capacity_evidence_available = True
            tradability_hash = axis.frozen_market_source.tradability_decision_hash
            outcome_hash = axis.frozen_market_source.execution_outcome_manifest_hash
        positions = {value: index for index, value in enumerate(common_sessions)}
        if any(value not in positions for value in sessions):
            # PLAN projects the executable axis from schedule maturity. RUN
            # proves the values fulfil that projection; a partial axis is a
            # moved authority, never a later silent clamp.
            raise PortfolioResearchCompositionError(
                "portfolio_application.causal_rank_mu_support_mismatch"
            )
        active_rows: IntArray = np.asarray([positions[value] for value in sessions], dtype=np.int64)
        risk_reader = CausalRiskReturnReader(artifact_root)
        all_risk_returns = risk_reader.read_sessions(risk_surface, risk_sessions)
        risk_positions = {value: index for index, value in enumerate(risk_sessions)}
        listing_positions = {
            value: index for index, value in enumerate(risk_surface.epoch.ordered_listing_ids)
        }
        columns: IntArray = np.asarray(
            [listing_positions[value] for value in listings], dtype=np.int64
        )
        producer = RiskSurfaceProducer()
        allocations = {}
        attributions = {}
        required = (
            producer.recipe.conditional_volatility_initialization_sessions
            + producer.recipe.factor_fit_sessions
        )
        history = axis.sectors
        for session in sessions:
            stop = risk_positions[session]
            if stop < required:
                raise PortfolioResearchCompositionError(
                    "portfolio_application.risk_history_insufficient"
                )
            rows = np.arange(stop - required, stop, dtype=np.int64)
            # The listings whose Sector at this formation is not their current one.
            in_force = history.at(session) if history.reclassifications else history.current
            moved = {
                listing: in_force[listing]
                for listing in listings
                if in_force[listing] != history.current[listing]
            }
            produced = producer.produce(
                RiskDecompositionInputs(
                    formation_session=session,
                    history_sessions=risk_sessions[stop - required : stop],
                    ordered_listing_ids=listings,
                    open_to_open_log_returns=np.ascontiguousarray(
                        all_risk_returns[np.ix_(rows, columns)],
                        dtype=np.float64,
                    ),
                    classification=classification,
                    return_surface_hash=risk_surface.surface_hash,
                    formation_sectors=moved or None,
                )
            )
            allocations[session] = produced.allocation
            attributions[session] = produced.attribution
        sector_matrix, sector_anchor = _sector_lanes(
            classification, listings=listings, history=history, sessions=sessions
        )
        active_realized = np.ascontiguousarray(realized[active_rows], dtype=np.float64)
        passive_returns = {
            session: (
                active_realized[index][np.asarray(decision[active_rows[index]])]
                if axis.frozen_market_source is not None
                else active_realized[index]
            )
            for index, session in enumerate(sessions)
        }
        replay_workspace: PortfolioBacktestWorkspace = PublicPortfolioReplayWorkspace(
            formation_sessions=sessions,
            ordered_listing_ids=listings,
            execution_available=np.ascontiguousarray(execution[active_rows]),
            realized_simple_returns=active_realized,
            causal_adv20=np.ascontiguousarray(adv20[active_rows], dtype=np.float64),
            sector_exposure_matrix=sector_matrix,
            equal_weight_sector_exposure=sector_anchor,
            passive_returns_by_session=passive_returns,
            capacity_evidence_available=capacity_evidence_available,
        )
        secondary_id, secondary_returns, secondary_disposition = self._secondary_benchmark(
            workspace=workspace_root, spec=spec, sessions=sessions
        )
        return SharedPortfolioInputs(
            workspace=replay_workspace,
            formation_sessions=sessions,
            ordered_listing_ids=listings,
            decision_eligible=np.ascontiguousarray(decision[active_rows]),
            common_sessions=common_sessions,
            active_rows=active_rows,
            common_decision_eligible=np.ascontiguousarray(decision),
            common_realized_simple_returns=np.ascontiguousarray(realized, dtype=np.float64),
            execution_events=axis.execution_events,
            risk_allocation_by_session=allocations,
            risk_attribution_by_session=attributions,
            report_risk_attributions=tuple(attributions[value] for value in sessions),
            risk_recipe_hash=INSTALLED_RISK_DECOMPOSITION_RECIPE.recipe_hash,
            risk_return_surface_hash=risk_surface.surface_hash,
            sector_map_hash=classification.map_hash,
            tradability_decision_hash=tradability_hash,
            execution_outcome_manifest_hash=outcome_hash,
            sector_history_hash=_sector_history_hash(history),
            secondary_benchmark_id=secondary_id,
            secondary_benchmark_returns=secondary_returns,
            secondary_benchmark_disposition=secondary_disposition,
            provided_lanes=SHARED_INPUT_LANES,
            reference_mark=reference_mark,
        )

    def _artifact_root(self, workspace: Path) -> Path:
        artifact_root = workspace.resolve() / "runtime" / "artifacts"
        if not artifact_root.is_dir():
            raise PortfolioResearchCompositionError(
                "portfolio_application.workspace_artifact_root_absent"
            )
        return artifact_root

    def _tradability(self, artifact_root: Path):  # type: ignore[no-untyped-def]
        service = CurrentTradabilityDataService(artifact_root)
        current = service.read_current()
        if current is None or current.projection.status != "DATA_TRADABILITY_READY":
            raise PortfolioResearchCompositionError(
                "portfolio_application.tradability_authority_absent"
            )
        lineage = service.read_lineage(current)
        return _TradabilityAuthority(service=service, decision=lineage.decision, lineage=lineage)

    def _secondary_benchmark(
        self,
        *,
        workspace: Path,
        spec: PortfolioResearchSpec,
        sessions: tuple[date, ...],
    ) -> tuple[str | None, tuple[float, ...] | None, str]:
        """Resolve the SPY comparator only when it was asked for.

        `anchor_only` resolves nothing, so the default path reads no market
        repository at all. When the comparator *is* requested and its lane is not
        available, the request is refused by name rather than silently downgraded
        to `anchor_only` -- a benchmark that quietly disappears is worse than one
        that refuses, because the report would still be labelled as comparing.
        """
        if spec.secondary_benchmark_view != "anchor_plus_spy":
            return None, None, "SECONDARY_BENCHMARK_NOT_REQUESTED"
        try:
            surface = build_portfolio_benchmark_surface(
                workspace=workspace,
                formation_sessions=sessions,
            )
        except PortfolioBenchmarkBoundaryError as error:
            raise PortfolioResearchCompositionError(
                "portfolio_application.secondary_benchmark_lane_unavailable:SPY_TOTAL_RETURN"
            ) from error
        return (
            "SPY_TOTAL_RETURN",
            tuple(float(value) for value in surface.simple_returns),
            "VISIBLY_CONTAMINATED_SECONDARY_COMPARATOR_NEVER_A_REPLACEMENT",
        )


class StrategyPortfolioResolver:
    """One shared resolver plus one installed catalog, behind the generic protocol.

    This is the whole strategy seam at runtime: the application asks for
    authorities and a resolution and never learns which package answered. Every
    refusal the package raises crosses here as a composition refusal with the
    package's own code intact.
    """

    def __init__(
        self, *, shared: SharedInputOwner, catalog: InstalledFrozenStrategyCatalog | None
    ) -> None:
        """Bind portfolio resolution to shared input ownership and installed strategy declarations.

        Args:
            shared: Deterministic shared-input authority and resolution owner.
            catalog: Finite installed strategy package catalog, or None while nothing is
                installed; an installation sets it on the running Host.
        """
        self.shared = shared
        self.catalog = catalog

    def _installed(self) -> InstalledFrozenStrategyCatalog:
        if self.catalog is None:
            raise PortfolioResearchCompositionError("portfolio_application.strategy_catalog_empty")
        return self.catalog

    @property
    def strategy_catalog_hash(self) -> str:
        """Read the installed strategy package declaration identity.

        Returns:
            The retained catalog catalog_hash.
        """
        return self._installed().catalog_hash

    def resolve_authorities(
        self, *, workspace: Path, spec: PortfolioResearchSpec
    ) -> ResolvedPortfolioAuthorities:
        """Resolve selected strategy and shared input coverage before execution.

        Args:
            workspace: Caller-owned workspace root.
            spec: Admitted strategy/mode and research controls.

        Returns:
            Resolved common support, listing population and exact strategy/input/source bindings.
        """
        selected = self._selected(spec)
        source = self._source(selected=selected, spec=spec)
        support = self._package(lambda: source.support(spec=spec))
        shared = self.shared.authorities(workspace=workspace, support=support)
        return ResolvedPortfolioAuthorities(
            coverage=shared.coverage,
            candidate_sessions=shared.candidate_sessions,
            ordered_listing_ids=shared.ordered_listing_ids,
            eligible_count=len(shared.ordered_listing_ids),
            strategy_package_id=selected.package.strategy_id,
            strategy_package_hash=selected.package.package_hash,
            score_source_mode=selected.mode,
            policy_identity=self._package(lambda: selected.policy_identity(spec=spec)),
            alpha_recipe_hash=selected.alpha_recipe_hash,
            alpha_evidence_manifest_hash=selected.alpha_evidence_manifest_hash,
            risk_recipe_hash=shared.risk_recipe_hash,
            risk_return_surface_hash=shared.risk_return_surface_hash,
            sector_map_hash=shared.sector_map_hash,
            tradability_decision_hash=shared.tradability_decision_hash,
            execution_outcome_manifest_hash=shared.execution_outcome_manifest_hash,
            public_path_source_hash=_public_path_source_hash(
                installed_source_hash=selected.source_identity_hash
            ),
            sector_history_hash=shared.sector_history_hash,
        )

    def resolve(
        self, *, workspace: Path, spec: PortfolioResearchSpec
    ) -> ResolvedPortfolioExecution:
        """Resolve exact selected component inputs and shared execution/Risk/report authority.

        Args:
            workspace: Caller-owned workspace root.
            spec: Admitted strategy/mode and research controls.

        Returns:
            Resolved execution inputs with ordered components, formation score receipts and declared
            source/reference/report bindings.

        Raises:
            PortfolioResearchCompositionError: Package selection/required lanes, declared component
                set or formation score-receipt coverage cannot be reconciled.
        """
        selected = self._selected(spec)
        source = self._source(selected=selected, spec=spec)
        policy = self._package(lambda: selected.policy_identity(spec=spec))
        support = self._package(lambda: source.support(spec=spec))
        shared = self.shared.resolve(workspace=workspace, spec=spec, support=support)
        self._package(lambda: shared.require_lanes(selected.package.required_shared_input_lanes))
        resolution = self._package(lambda: source.resolve_components(shared=shared, spec=spec))
        declared = selected.package.component_ids
        if tuple(value.component_id for value in resolution.components) != declared:
            raise PortfolioResearchCompositionError(
                "portfolio_application.package_component_set_invalid"
            )
        if len(resolution.score_receipt_hashes) != len(shared.formation_sessions):
            raise PortfolioResearchCompositionError(
                "portfolio_application.package_score_receipt_axis_invalid"
            )
        return ResolvedPortfolioExecution(
            workspace=shared.workspace,
            components=resolution.components,
            score_receipt_hashes=resolution.score_receipt_hashes,
            report_risk_attributions=shared.report_risk_attributions,
            strategy_package_hash=selected.package.package_hash,
            score_source_mode=selected.mode,
            policy_recipe_hash=policy.policy_recipe_hash,
            policy_catalog_hash=policy.policy_catalog_hash,
            policy_adapter_binding_hash=policy.policy_adapter_binding_hash,
            alpha_recipe_hash=selected.alpha_recipe_hash,
            alpha_evidence_manifest_hash=selected.alpha_evidence_manifest_hash,
            risk_recipe_hash=shared.risk_recipe_hash,
            risk_return_surface_hash=shared.risk_return_surface_hash,
            sector_map_hash=shared.sector_map_hash,
            tradability_decision_hash=shared.tradability_decision_hash,
            execution_outcome_manifest_hash=shared.execution_outcome_manifest_hash,
            sector_history_hash=shared.sector_history_hash,
            secondary_benchmark_id=shared.secondary_benchmark_id,
            secondary_benchmark_returns=shared.secondary_benchmark_returns,
            secondary_benchmark_disposition=shared.secondary_benchmark_disposition,
            reference_mark=shared.reference_mark,
        )

    def installed_packages(self) -> Mapping[str, FrozenStrategyPackage]:
        """Read installed frozen package declarations through the catalog owner.

        Returns:
            Mapping of installed package declarations supplied by the retained catalog.
        """
        return {} if self.catalog is None else self.catalog.packages()

    def _selected(self, spec: PortfolioResearchSpec) -> SelectedStrategy:
        strategy_id = (
            None
            if spec.strategy_package_id == WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID
            else spec.strategy_package_id
        )
        catalog = self._installed()
        return self._package(
            lambda: catalog.select(
                strategy_id=strategy_id, score_source_mode=spec.score_source_mode
            )
        )

    def _source(
        self, *, selected: SelectedStrategy, spec: PortfolioResearchSpec
    ) -> StrategyScoreSource:
        def _admitted() -> StrategyScoreSource:
            selected.admit_request(spec=spec)
            return selected.score_source()

        return self._package(_admitted)

    @staticmethod
    def _package[T](call: Callable[[], T]) -> T:
        try:
            return call()
        except StrategyPackageError as error:
            raise PortfolioResearchCompositionError(str(error)) from error


__all__ = [
    "SHARED_INPUT_LANES",
    "PortfolioExecutionResolver",
    "PortfolioResearchCompiler",
    "PublicPortfolioReplayWorkspace",
    "ResolvedPortfolioAuthorities",
    "SharedPortfolioInputResolver",
    "StrategyPortfolioResolver",
]
