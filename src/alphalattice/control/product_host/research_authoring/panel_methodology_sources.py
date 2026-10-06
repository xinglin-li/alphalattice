"""Host resolution of immutable inputs for installed Panel methodologies.

The current remediation consumes the failed experiment's stable Stage-2 input
children, so a small historical decoder has a real consumer.  It reads the old
content-addressed graph and resolves the Formula, Target and Panel identities;
it never reads a current pointer and never writes into the baseline workspace.
The returned numerical lanes are assembled only by the Alpha input owner.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.clocks import EveryFormationClock
from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioStateTransitionBinding,
)
from alphalattice.capabilities.portfolio_inputs.benchmark import (
    PortfolioBenchmarkStore,
    build_portfolio_benchmark_surface,
)
from alphalattice.capabilities.portfolio_inputs.session_marks import (
    resolve_session_mark_availability,
)
from alphalattice.capabilities.portfolio_inputs.tradability.contracts import (
    HistoricalDecisionTradabilitySurface,
    HistoricalExecutionAvailabilitySurface,
    HistoricalTradabilityBundle,
)
from alphalattice.capabilities.portfolio_inputs.tradability.surface import (
    TradabilityArtifactStore,
)
from alphalattice.control.product_host.research_authoring.feature_activations import (
    feature_catalog_for,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.methods import (
    build_installed_execution_outcome_method_catalog,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.foundation.factor_research.experiments.campaign import (
    CuratedFactorCheckpoint,
)
from alphalattice.foundation.factor_research.publication.artifacts import (
    FactorResearchArtifactStore,
)
from alphalattice.foundation.feature_engine.catalog.contracts import (
    desktop_core_feature_bundle,
    source_availability_binding,
)
from alphalattice.foundation.feature_engine.contracts import FeaturePanelBinding
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import (
    PanelDerivationRecipe,
    SectorRevisionMap,
)
from alphalattice.foundation.feature_engine.panels.development_input import (
    DevelopmentFeatureOverlayManifest,
)
from alphalattice.foundation.feature_engine.panels.development_overlay import (
    DevelopmentFeatureOverlayService,
    load_development_feature_overlay_manifest,
    load_development_methodology_surface_manifest,
    read_raw_development_feature_overlay,
)
from alphalattice.foundation.feature_engine.panels.rematerialization import (
    ArtifactOnlyPanelRematerializer,
    PanelRematerializationMismatch,
    recipe_sector_history,
)
from alphalattice.foundation.feature_engine.producers.factors.interactions import (
    interaction_market_state_children,
)
from alphalattice.foundation.feature_engine.producers.factors.session_liquidity import (
    raw_session_dollar_volume,
)
from alphalattice.foundation.feature_engine.producers.factors.specifications import (
    admit_factor_development_capabilities,
)
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.publication.session_marks import (
    SessionMarkArtifactStore,
    SessionMarkError,
    SessionMarkSurface,
    publish_market_session_marks,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.experiments.dynamic_panel_artifacts import (
    DynamicPanelArtifactStore,
)
from alphalattice.investment.alpha_research.experiments.dynamic_panel_development import (
    DynamicPanelAlphaDevelopmentService,
    DynamicPanelDevelopmentRequest,
    DynamicPanelSourceResolution,
)
from alphalattice.investment.alpha_research.inputs.dynamic_panel import (
    DynamicPanelSourceArrays,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
    FORMATION_OBSERVATION_DOLLAR_VOLUME_SOURCE_LANE_ID,
    SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS,
    PanelFeatureSourceArrays,
    assemble_panel_context_arrays,
)
from alphalattice.investment.alpha_research.scores.temporal_aggregation import (
    AlphaPanelSourceIdentity,
    resolve_alpha_panel_source_identity,
)
from alphalattice.investment.portfolio_strategy_lab.campaign.authority import (
    ResolvedReferenceMark,
    ResolvedStageSixCovariance,
    load_stage_six_covariance,
    resolve_portfolio_state_transition_binding,
    resolve_reference_mark_lane,
)
from alphalattice.investment.portfolio_strategy_lab.inputs.development import (
    ExecutionClockContext,
    _tradability_matrices,
    resolve_execution_clock_context,
    resolve_execution_clock_metadata,
)
from alphalattice.investment.portfolio_strategy_lab.research_loop.paired_alpha_portfolio import (
    ImmutablePortfolioBenchmark,
    ImmutablePortfolioMarketInputs,
    sector_exposures,
)
from alphalattice.investment.risk_research.contracts import CausalRiskReturnSurface
from alphalattice.investment.risk_research.experiments.development_artifacts import (
    DEVELOPMENT_SURFACE_CATEGORY,
    RiskDevelopmentCovarianceSurface,
)
from alphalattice.investment.risk_research.experiments.window import (
    REQUIRED_LOOKBACK_SESSIONS,
    RISK_INPUT_BINDING_CATEGORY,
    RiskDevelopmentInputBinding,
)
from alphalattice.investment.risk_research.surfaces.artifacts import RiskArtifactStore
from alphalattice.investment.risk_research.surfaces.returns import (
    CausalRiskReturnReader,
    RiskReturnArtifactStore,
)
from alphalattice.investment.sector_research.inputs.contracts import (
    SectorContextManifest,
)
from alphalattice.investment.sector_research.inputs.storage import SectorContextStore
from alphalattice.investment.sector_research.inputs.surface import (
    compile_sector_context_arrays,
)
from alphalattice.kernel.quant.sector_history import (
    SectorHistory,
    reclassification_payload,
    sector_ids,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResolvedResearchAuthority,
)

type FloatArray = npt.NDArray[np.float64]

_HASH = r"^[0-9a-f]{64}$"
_BASELINE_REPORT = Path("reports/dynamic-panel-score-aggregation.json")
_CORRECTED_INPUT_METHOD_ID = "FEATURE_T_PANEL_TOTAL_RETURN_INPUTS"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class PanelMethodologySourceResolution(_Contract):
    """Seal exact methodology/Panel/outcome axes with zero forbidden source access counts."""

    kind: str = "PanelMethodologySourceResolution"
    input_method_id: str
    baseline_report_hash: str = Field(pattern=_HASH)
    panel_snapshot_hash: str = Field(pattern=_HASH)
    development_overlay_hash: str | None = Field(default=None, pattern=_HASH)
    methodology_surface_hash: str | None = Field(default=None, pattern=_HASH)
    raw_formula_surface_hash: str | None = Field(default=None, pattern=_HASH)
    curated_factor_checkpoint_hash: str | None = Field(default=None, pattern=_HASH)
    panel_derivation_recipe_hash: str | None = Field(default=None, pattern=_HASH)
    panel_base_formula_closure_hash: str | None = Field(default=None, pattern=_HASH)
    factor_axis_admission_hash: str | None = Field(default=None, pattern=_HASH)
    formula_observation_policy_hash: str | None = Field(default=None, pattern=_HASH)
    source_availability_policy_hash: str | None = Field(default=None, pattern=_HASH)
    source_authority_binding_hash: str | None = Field(default=None, pattern=_HASH)
    total_return_target_evidence_hash: str = Field(pattern=_HASH)
    sector_revision: str = Field(pattern=_HASH)
    execution_outcome_recipe_id: str
    outcome_method_binding_hash: str = Field(pattern=_HASH)
    sector_context_manifest_hash: str = Field(pattern=_HASH)
    ordered_session_axis_hash: str = Field(pattern=_HASH)
    ordered_listing_axis_hash: str = Field(pattern=_HASH)
    ordered_factor_axis_hash: str = Field(pattern=_HASH)
    network_access_count: int = Field(ge=0)
    provider_access_count: int = Field(ge=0)
    holdout_access_count: int = Field(ge=0)
    current_or_production_pointer_read_count: int = Field(ge=0)
    pointer_mutation_count: int = Field(ge=0)
    source_workspace_write_count: int = Field(ge=0)
    resolution_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the declared methodology source resolution with absent optionals excluded.

        Args:
            values: Explicit resolution fields excluding the generated identity.

        Returns:
            Validated resolution with canonical resolution_hash over non-null fields.
        """
        provisional = cls.model_construct(**values, resolution_hash="0" * 64)
        identity = provisional.model_dump(
            mode="json", exclude={"resolution_hash"}, exclude_none=True
        )
        return cls(**values, resolution_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require zero forbidden accesses/mutations and exact non-null source resolution identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AuthoringError: Access/write counts are nonzero or canonical resolution_hash differs.
        """
        if any(
            value != 0
            for value in (
                self.network_access_count,
                self.provider_access_count,
                self.holdout_access_count,
                self.current_or_production_pointer_read_count,
                self.pointer_mutation_count,
                self.source_workspace_write_count,
            )
        ) or self.resolution_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"resolution_hash"}, exclude_none=True)
        ):
            raise AuthoringError("alpha_research.panel_source_resolution_invalid")
        return self


@dataclass(frozen=True, slots=True)
class PanelMethodologySourceRoots:
    """Runtime-only locations; no path is admitted into scientific config."""

    repository_root: Path
    panel_artifact_root: Path
    legacy_panel_artifact_root: Path
    feature_artifact_root: Path
    failed_baseline_workspace: Path
    sector_context_artifact_root: Path
    execution_outcome_artifact_root: Path | None = None
    development_overlay_artifact_root: Path | None = None


@dataclass(frozen=True, slots=True)
class ResolvedPanelMethodologySource:
    """Retain verified Panel arrays, outer folds and immutable Portfolio execution context."""

    arrays: PanelFeatureSourceArrays
    resolution: PanelMethodologySourceResolution
    outer_folds: tuple[PanelMethodologyOuterFold, ...]
    portfolio_market: ImmutablePortfolioMarketInputs
    portfolio_benchmark: ImmutablePortfolioBenchmark
    portfolio_fold_indices: npt.NDArray[np.int64]
    execution_clock: ExecutionClockContext


@dataclass(frozen=True, slots=True)
class ResolvedFixedPortfolioMarket:
    """Owner-resolved market facts on the Risk economic axis.

    The old Dynamic campaign surface is deliberately not extended here.  Its
    491 rows are one experiment window, not a durable market clock.  This
    successor composes the existing Target outcome, tradability, Sector,
    benchmark, and Risk input owners on the exact immutable axis they publish.
    """

    market: ImmutablePortfolioMarketInputs
    benchmark: ImmutablePortfolioBenchmark
    risk_input_binding_hash: str
    tradability_bundle_hash: str
    risk_return_surface_hash: str
    benchmark_surface_hash: str
    reference_marks: ResolvedReferenceMark


@dataclass(frozen=True, slots=True)
class ResolvedFixedPortfolioFacts:
    """Feature-free facts required by the frozen Alpha Portfolio consumer."""

    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    ordered_sector_ids: tuple[str, ...]
    sector_by_listing_id: Mapping[str, str]
    raw_simple_execution_returns: FloatArray
    total_return_target_evidence_hash: str
    outcome_method_binding_hash: str
    panel_derivation_recipe_hash: str
    sector_revision: str
    execution_clock: ExecutionClockContext
    alpha_panel_source_identity: AlphaPanelSourceIdentity


@dataclass(frozen=True, slots=True)
class ResolvedFixedPortfolioFactAuthority:
    """Exact factual handles and clocks needed by sealed preflight."""

    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    total_return_target_evidence_hash: str
    outcome_method_binding_hash: str
    panel_derivation_recipe_hash: str
    sector_revision: str
    execution_clock: ExecutionClockContext
    alpha_panel_source_identity: AlphaPanelSourceIdentity
    metadata_bytes_read: int


@dataclass(frozen=True, slots=True)
class ResolvedFixedPortfolioMarketAuthority:
    """Exact Market/Risk manifests for preflight, with no numerical matrices."""

    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    risk_input_binding_hash: str
    risk_return_surface_hash: str
    tradability_bundle_hash: str
    universe_epoch_hash: str
    benchmark_surface_hash: str
    state_transition: PortfolioStateTransitionBinding
    mark_surface: SessionMarkSurface
    metadata_bytes_read: int


@dataclass(frozen=True, slots=True)
class PanelMethodologyOuterFold:
    """Declare one exact outer fold with training/validation sessions and source manifest."""

    fold_index: int
    training_sessions: tuple[date, ...]
    validation_sessions: tuple[date, ...]
    source_manifest_hash: str


@dataclass(frozen=True, slots=True)
class _FailedBaselineIndependentSource:
    """Factor-independent lanes read from the immutable failed experiment graph."""

    old: DynamicPanelSourceArrays
    old_resolution: DynamicPanelSourceResolution
    baseline_report_hash: str
    panel_snapshot_hash: str
    methodology_surface_hash: str
    raw_formula_surface_hash: str
    curated_factor_checkpoint_hash: str
    total_return_target_evidence_hash: str
    execution_outcome_recipe_id: str
    outcome_method_binding_hash: str
    outer_folds: tuple[PanelMethodologyOuterFold, ...]
    portfolio_market: ImmutablePortfolioMarketInputs
    portfolio_benchmark: ImmutablePortfolioBenchmark
    portfolio_fold_indices: npt.NDArray[np.int64]
    execution_clock: ExecutionClockContext


def _read_baseline_report(root: Path) -> tuple[dict[str, Any], str]:
    path = root / _BASELINE_REPORT
    try:
        content = path.read_bytes()
        payload = json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise AuthoringError("alpha_research.failed_baseline_report_unavailable") from error
    if not isinstance(payload, dict):
        raise AuthoringError("alpha_research.failed_baseline_report_invalid")
    return cast(dict[str, Any], payload), str(canonical_hash(content.hex()))


def _curated_checkpoint_hash(
    *, feature_root: Path, methodology_surface_hash: str, factor_ids: tuple[str, ...]
) -> str:
    category = feature_root / "factor-research" / "campaign" / "curated-checkpoints"
    matches: list[CuratedFactorCheckpoint] = []
    for path in sorted(category.glob("*.json")):
        try:
            value = CuratedFactorCheckpoint.model_validate_json(path.read_bytes())
        except Exception as error:
            raise AuthoringError("alpha_research.factor_checkpoint_readback_failed") from error
        if (
            value.feature_source_hash == methodology_surface_hash
            and value.ordered_factor_ids == factor_ids
        ):
            matches.append(value)
    if len(matches) != 1:
        raise AuthoringError("alpha_research.factor_checkpoint_handle_ambiguous")
    return str(matches[0].checkpoint_hash)


def _sector_state_surface(
    *,
    root: Path,
    formation_sessions: tuple[object, ...],
    ordered_sector_ids: tuple[str, ...],
) -> tuple[FloatArray, str]:
    store = SectorContextStore(root)
    manifests: list[SectorContextManifest] = []
    manifest_root = store.root / "manifests"
    for path in sorted(manifest_root.glob("*.json")):
        try:
            value = SectorContextManifest.model_validate_json(path.read_bytes())
        except Exception as error:
            raise AuthoringError("alpha_research.sector_context_readback_failed") from error
        if value.sector_ids == ordered_sector_ids:
            manifests.append(value)
    if not manifests:
        raise AuthoringError("alpha_research.sector_context_handle_unresolved")
    requested = set(formation_sessions)
    maximum_overlap = max(len(requested & set(value.formation_sessions)) for value in manifests)
    selected = [
        value
        for value in manifests
        if len(requested & set(value.formation_sessions)) == maximum_overlap
    ]
    if maximum_overlap == 0 or len(selected) != 1:
        raise AuthoringError("alpha_research.sector_context_handle_ambiguous")
    surface = store.load(selected[0].manifest_hash)
    positions = {value: index for index, value in enumerate(surface.manifest.formation_sessions)}
    result: FloatArray = np.full(
        (len(formation_sessions), len(ordered_sector_ids), 4), np.nan, dtype=np.float64
    )
    for output_position, session in enumerate(formation_sessions):
        source_position = positions.get(session)
        if source_position is not None:
            result[output_position] = surface.values[source_position]
    result = np.ascontiguousarray(result, dtype=np.float64)
    result.setflags(write=False)
    return result, surface.manifest.manifest_hash


def _resolve_failed_baseline_independent_source(
    *,
    roots: PanelMethodologySourceRoots,
    authority: ResolvedResearchAuthority,
) -> _FailedBaselineIndependentSource:
    """Read Target/outcome/fold/Portfolio lanes without resolving old Feature context."""

    baseline = roots.failed_baseline_workspace.resolve()
    report, report_hash = _read_baseline_report(baseline)
    artifact_root = baseline / "artifacts"
    dynamic_store = DynamicPanelArtifactStore(artifact_root)
    try:
        dossier = dynamic_store.load_model_dossier(str(report["model_dossier_hash"]))
        program = dynamic_store.load_model_program(dossier.program_hash)
        surface = dynamic_store.load_surface_manifest(program.dynamic_panel_surface_hash)
        fold_manifests = tuple(
            dynamic_store.load_fold_context_manifest(value)
            for value in program.fold_context_manifest_hashes
        )
    except (KeyError, ValueError, FileNotFoundError) as error:
        raise AuthoringError("alpha_research.failed_baseline_graph_unavailable") from error
    methodology = load_development_methodology_surface_manifest(
        output_root=roots.feature_artifact_root,
        surface_hash=surface.relative_surface_hash,
    )
    panel_snapshot_hash = methodology.base_panel_snapshot_hash
    if panel_snapshot_hash != authority.panel_snapshot_hash:
        raise AuthoringError("alpha_research.panel_source_authority_mismatch")
    checkpoint_hash = _curated_checkpoint_hash(
        feature_root=roots.feature_artifact_root,
        methodology_surface_hash=methodology.surface_hash,
        factor_ids=surface.ordered_factor_ids,
    )
    service = DynamicPanelAlphaDevelopmentService(
        panel_artifact_root=roots.legacy_panel_artifact_root,
        feature_artifact_root=roots.feature_artifact_root,
        factor_artifact_reader=FactorResearchArtifactStore(roots.feature_artifact_root),
        target_artifact_root=artifact_root,
    )
    old, old_resolution, _frozen_at = service.resolve_source(
        DynamicPanelDevelopmentRequest(
            panel_snapshot_hash=panel_snapshot_hash,
            methodology_surface_hash=methodology.surface_hash,
            curated_factor_checkpoint_hash=checkpoint_hash,
            total_return_target_evidence_hash=surface.total_return_target_evidence_hash,
        )
    )
    target_store = AlphaDevelopmentArtifactStore(artifact_root)
    target_evidence = target_store.load_total_return_target_evidence(
        surface.total_return_target_evidence_hash
    )
    binding = target_store.load_total_return_target_binding(target_evidence.recipe_binding_hash)
    if roots.execution_outcome_artifact_root is None:
        raise AuthoringError("alpha_research.execution_outcome_root_missing")
    outcome_reader = CausalExecutionOutcomeDevelopmentReader(roots.execution_outcome_artifact_root)
    execution_clock = resolve_execution_clock_context(
        reader=outcome_reader,
        manifest_ref=outcome_reader.manifest_uri(binding.causal_outcome_snapshot_hash),
        sessions=authority.sessions,
    )
    if (
        execution_clock.events.method_binding_hash != binding.outcome_method_binding_hash
        or execution_clock.events.execution_recipe_id != binding.execution_outcome_recipe_id
    ):
        raise AuthoringError("alpha_research.execution_clock_not_this_target")
    if (
        old.formation_sessions != authority.sessions
        or old.ordered_listing_ids != authority.ordered_listing_ids
    ):
        raise AuthoringError("alpha_research.panel_source_common_axis_mismatch")
    if old_resolution.network_call_count != 0 or old_resolution.source_write_count != 0:
        raise AuthoringError("alpha_research.failed_baseline_source_not_read_only")
    portfolio_market, portfolio_benchmark, portfolio_fold_indices = (
        _failed_baseline_portfolio_inputs(baseline)
    )
    return _FailedBaselineIndependentSource(
        old=old,
        old_resolution=old_resolution,
        baseline_report_hash=report_hash,
        panel_snapshot_hash=panel_snapshot_hash,
        methodology_surface_hash=methodology.surface_hash,
        raw_formula_surface_hash=methodology.raw_formula_surface_hash,
        curated_factor_checkpoint_hash=checkpoint_hash,
        total_return_target_evidence_hash=surface.total_return_target_evidence_hash,
        execution_outcome_recipe_id=binding.execution_outcome_recipe_id,
        outcome_method_binding_hash=binding.outcome_method_binding_hash,
        outer_folds=tuple(
            PanelMethodologyOuterFold(
                fold_index=value.fold_index,
                training_sessions=tuple(value.training_sessions),
                validation_sessions=tuple(value.validation_sessions),
                source_manifest_hash=value.manifest_hash,
            )
            for value in sorted(fold_manifests, key=lambda item: item.fold_index)
        ),
        portfolio_market=portfolio_market,
        portfolio_benchmark=portfolio_benchmark,
        portfolio_fold_indices=portfolio_fold_indices,
        execution_clock=execution_clock,
    )


def _readonly(values: npt.NDArray[Any]) -> FloatArray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    result.setflags(write=False)
    return cast(FloatArray, result)


def _array_identity(values: npt.NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(values)
    return str(
        canonical_hash(
            {
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "content_sha256": sha256(contiguous.tobytes()).hexdigest(),
            }
        )
    )


def _listing_median(values: FloatArray) -> FloatArray:
    """Finite listing median without warning on legitimate all-missing warm-up."""

    output = np.full((values.shape[0], values.shape[2]), np.nan, dtype=np.float64)
    for session in range(values.shape[0]):
        for column in range(values.shape[2]):
            lane = values[session, :, column]
            finite = lane[np.isfinite(lane)]
            if finite.size:
                output[session, column] = float(np.median(finite))
    return _readonly(output)


def _failed_baseline_portfolio_inputs(
    baseline: Path,
) -> tuple[ImmutablePortfolioMarketInputs, ImmutablePortfolioBenchmark, npt.NDArray[np.int64]]:
    """A failed baseline's Portfolio inputs, which it sealed as a dynamic-panel input surface.

    That format's readback retired with SR (2026-09-27): no kept workspace holds one, so a
    baseline that names one is refused by name.
    """
    del baseline
    raise AuthoringError("alpha_research.failed_baseline_portfolio_format_retired")


def resolve_failed_experiment_baseline_source(
    *,
    roots: PanelMethodologySourceRoots,
    authority: ResolvedResearchAuthority,
    input_method_id: str,
) -> ResolvedPanelMethodologySource:
    """Resolve the exact stable inputs used by the immutable e2f43c4 graph."""
    if input_method_id != "E2F43C4_STABLE_STAGE2_INPUTS":
        raise AuthoringError("alpha_research.panel_input_method_not_installed")
    independent = _resolve_failed_baseline_independent_source(roots=roots, authority=authority)
    old = independent.old
    ordered_sector_ids = tuple(sorted(set(old.sector_by_listing_id.values())))
    sector_states, sector_manifest_hash = _sector_state_surface(
        root=roots.sector_context_artifact_root,
        formation_sessions=old.formation_sessions,
        ordered_sector_ids=ordered_sector_ids,
    )
    interaction_states = _listing_median(old.raw_absolute_state_values)
    raw_log = _readonly(old.raw_log_returns)
    raw_simple = _readonly(old.simple_economic_returns)
    sector_mapping = MappingProxyType(dict(old.sector_by_listing_id))
    sector_context, market_context = assemble_panel_context_arrays(
        formation_sessions=old.formation_sessions,
        holding_end_sessions=old.holding_end_sessions,
        ordered_listing_ids=old.ordered_listing_ids,
        ordered_sector_ids=ordered_sector_ids,
        sector_by_listing_id=sector_mapping,
        raw_log_execution_returns=raw_log,
        raw_simple_execution_returns=raw_simple,
        sector_state_values=sector_states,
        market_interaction_state_values=interaction_states,
    )
    arrays = PanelFeatureSourceArrays(
        formation_sessions=old.formation_sessions,
        holding_end_sessions=old.holding_end_sessions,
        ordered_listing_ids=old.ordered_listing_ids,
        ordered_factor_ids=old.ordered_factor_ids,
        absolute_state_factor_ids=old.ordered_factor_ids,
        ordered_sector_ids=ordered_sector_ids,
        sector_by_listing_id=sector_mapping,
        raw_formula_values=_readonly(old.raw_factor_values),
        total_return_target_z=_readonly(old.target_z_values),
        raw_log_execution_returns=raw_log,
        raw_simple_execution_returns=raw_simple,
        sector_context_values=sector_context,
        market_context_values=market_context,
        source_identity_hashes=MappingProxyType(
            {
                "panel": independent.panel_snapshot_hash,
                "methodology": independent.methodology_surface_hash,
                "raw_formula": independent.raw_formula_surface_hash,
                "factor_checkpoint": independent.curated_factor_checkpoint_hash,
                "target": independent.total_return_target_evidence_hash,
                "sector_context": sector_manifest_hash,
                "outcome_method": old.outcome_method_binding_hash,
            }
        ),
    )
    resolution = PanelMethodologySourceResolution.create(
        input_method_id=input_method_id,
        baseline_report_hash=independent.baseline_report_hash,
        panel_snapshot_hash=independent.panel_snapshot_hash,
        methodology_surface_hash=independent.methodology_surface_hash,
        raw_formula_surface_hash=independent.raw_formula_surface_hash,
        curated_factor_checkpoint_hash=independent.curated_factor_checkpoint_hash,
        total_return_target_evidence_hash=independent.total_return_target_evidence_hash,
        sector_revision=independent.old_resolution.sector_revision,
        execution_outcome_recipe_id=independent.execution_outcome_recipe_id,
        outcome_method_binding_hash=independent.outcome_method_binding_hash,
        sector_context_manifest_hash=sector_manifest_hash,
        ordered_session_axis_hash=str(
            canonical_hash([value.isoformat() for value in old.formation_sessions])
        ),
        ordered_listing_axis_hash=str(canonical_hash(list(old.ordered_listing_ids))),
        ordered_factor_axis_hash=str(canonical_hash(list(old.ordered_factor_ids))),
        network_access_count=0,
        provider_access_count=0,
        holdout_access_count=0,
        current_or_production_pointer_read_count=0,
        pointer_mutation_count=0,
        source_workspace_write_count=0,
    )
    return ResolvedPanelMethodologySource(
        arrays=arrays,
        resolution=resolution,
        outer_folds=independent.outer_folds,
        portfolio_market=independent.portfolio_market,
        portfolio_benchmark=independent.portfolio_benchmark,
        portfolio_fold_indices=independent.portfolio_fold_indices,
        execution_clock=independent.execution_clock,
    )


def _corrected_panel_recipe(
    *, resolver: ArtifactResolver, snapshot_hash: str, recipe_hash: str | None = None
) -> tuple[PanelDerivationRecipe, PanelClosureArtifactStore]:
    store = PanelClosureArtifactStore(resolver)
    if recipe_hash is not None:
        try:
            value = store.load_model(
                category="recipes", content_hash=recipe_hash, model=PanelDerivationRecipe
            )
        except (FileNotFoundError, ValueError) as error:
            raise AuthoringError(
                "alpha_research.panel_derivation_recipe_readback_failed"
            ) from error
        if value.snapshot_hash != snapshot_hash:
            raise AuthoringError("alpha_research.panel_derivation_recipe_not_this_snapshot")
        return value, store
    matches: list[PanelDerivationRecipe] = []
    for path in sorted((store.root / "recipes").glob("*.json")):
        try:
            value = store.load_model(
                category="recipes", content_hash=path.stem, model=PanelDerivationRecipe
            )
        except (FileNotFoundError, ValueError) as error:
            raise AuthoringError(
                "alpha_research.panel_derivation_recipe_readback_failed"
            ) from error
        if value.snapshot_hash == snapshot_hash:
            matches.append(value)
    if len(matches) != 1:
        raise AuthoringError("alpha_research.panel_derivation_recipe_handle_ambiguous")
    return matches[0], store


def _corrected_sector_map(
    *, store: PanelClosureArtifactStore, recipe: PanelDerivationRecipe
) -> SectorHistory:
    try:
        value = store.load_model(
            category="sector-maps", content_hash=recipe.sector_map_hash, model=SectorRevisionMap
        )
    except (FileNotFoundError, ValueError) as error:
        raise AuthoringError("alpha_research.panel_sector_map_readback_failed") from error
    # The Sector each session read: the recipe's map, then its reclassifications (V346).
    history = recipe_sector_history(recipe, value)
    if set(history) != set(recipe.listing_ids):
        raise AuthoringError("alpha_research.panel_sector_map_axis_mismatch")
    return history


def _formation_available_market_interaction_states(
    *, workspace: Path, sessions: tuple[date, ...]
) -> tuple[FloatArray, str]:
    feature_store = FeatureStateRepository(workspace)
    reference = feature_store.market_reference("SPY")
    if reference is None:
        raise AuthoringError("alpha_research.market_reference_unresolved")
    listing_id = str(reference["listing_id"])
    points = MarketDataRepository(workspace).provider_adjusted_closes(
        listing_id,
        start=sessions[0],
        through=sessions[-1],
    )
    by_session = {value.session_date: float(value.adjusted_close) for value in points}
    if set(sessions) - set(by_session):
        raise AuthoringError("alpha_research.market_reference_session_axis_incomplete")
    source = pd.DataFrame(
        {
            "session_date": sessions,
            "listing_id": (listing_id,) * len(sessions),
            "market_provider_adjusted_close": tuple(by_session[value] for value in sessions),
        }
    )
    try:
        frame = interaction_market_state_children(source)
    except ValueError as error:
        raise AuthoringError("alpha_research.market_interaction_formula_failed") from error
    values = _readonly(frame.to_numpy(dtype=np.float64))
    return values, str(
        canonical_hash(
            {
                "owner": "feature_engine.producers.factors.interactions",
                "market_reference_revision": str(reference["revision_hash"]),
                "session_axis": [value.isoformat() for value in sessions],
                "values_hash": _array_identity(values),
            }
        )
    )


def _corrected_sector_states(
    *,
    sessions: tuple[date, ...],
    holding_end_sessions: tuple[date, ...],
    listing_ids: tuple[str, ...],
    raw_log_returns: FloatArray,
    target_evidence_hash: str,
    sector_by_listing_id: Mapping[str, str],
    sector_revision: str,
) -> tuple[FloatArray, str]:
    try:
        return compile_sector_context_arrays(
            sessions=sessions,
            holding_end_sessions=holding_end_sessions,
            listing_ids=listing_ids,
            raw_log_returns=raw_log_returns,
            target_evidence_hash=target_evidence_hash,
            sector_by_listing_id=sector_by_listing_id,
            sector_revision=sector_revision,
        )
    except ValueError as error:
        raise AuthoringError("alpha_research.corrected_sector_context_failed") from error


def _rebind_portfolio_sector_authority(
    *,
    market: ImmutablePortfolioMarketInputs,
    panel_snapshot_hash: str,
    sector_revision: str,
    sector_by_listing_id: Mapping[str, str],
    ordered_sector_ids: tuple[str, ...],
) -> ImmutablePortfolioMarketInputs:
    # Each formation's exposure from the Sectors in force at it (V346).
    exposure, equal_weight = sector_exposures(
        market.ordered_listing_ids,
        ordered_sector_ids,
        sector_by_listing_id,
        sessions=market.formation_sessions,
    )
    return ImmutablePortfolioMarketInputs(
        formation_sessions=market.formation_sessions,
        economic_formation_sessions=market.economic_formation_sessions,
        ordered_listing_ids=market.ordered_listing_ids,
        sector_ids=ordered_sector_ids,
        sector_exposure_matrix=exposure,
        equal_weight_sector_exposure=equal_weight,
        decision_eligible=market.decision_eligible,
        execution_available=market.execution_available,
        realized_simple_returns=market.realized_simple_returns,
        passive_returns_by_session=market.passive_returns_by_session,
        causal_adv20=market.causal_adv20,
        tradability_bundle_hash=market.tradability_bundle_hash,
        universe_epoch_hash=market.universe_epoch_hash,
        sector_revision=sector_revision,
        source_surface_hash=str(
            canonical_hash(
                {
                    "market_source_surface_hash": market.source_surface_hash,
                    "panel_snapshot_hash": panel_snapshot_hash,
                    "sector_revision": sector_revision,
                    **reclassification_payload(sector_by_listing_id),
                    "sector_ids": list(ordered_sector_ids),
                    "sector_exposure_hash": _array_identity(exposure),
                }
            )
        ),
    )


def _risk_input_binding(covariance: ResolvedStageSixCovariance) -> RiskDevelopmentInputBinding:
    store = RiskArtifactStore(covariance.evidence_root)
    try:
        payload = store.load_json(
            category=RISK_INPUT_BINDING_CATEGORY,
            uri=store.uri(
                RISK_INPUT_BINDING_CATEGORY,
                covariance.surface.input_binding_hash,
            ),
            identity_field="input_binding_hash",
        )
        return RiskDevelopmentInputBinding.model_validate(payload)
    except Exception as error:
        raise AuthoringError("alpha_research.portfolio_risk_input_binding_invalid") from error


def _resolved_tradability_surfaces(
    *,
    artifact_root: Path,
    risk_binding: RiskDevelopmentInputBinding,
    execution_clock: ExecutionClockContext,
    bundle_hash: str | None = None,
) -> tuple[
    HistoricalTradabilityBundle,
    HistoricalDecisionTradabilitySurface,
    HistoricalExecutionAvailabilitySurface,
]:
    """Resolve one immutable bundle; never consult the tradability current pointer."""

    store = TradabilityArtifactStore(artifact_root)
    expected_sessions = risk_binding.formation_sessions
    try:
        expected_intended = tuple(
            execution_clock.entry_session_by_formation[session] for session in expected_sessions
        )
    except KeyError as error:
        raise AuthoringError("alpha_research.portfolio_execution_axis_incomplete") from error
    expected_schedule_hash = str(
        canonical_hash(tuple(zip(expected_sessions, expected_intended, strict=True)))
    )
    matches: list[
        tuple[
            HistoricalTradabilityBundle,
            HistoricalDecisionTradabilitySurface,
            HistoricalExecutionAvailabilitySurface,
        ]
    ] = []
    bundle_root = store.root / "bundles"
    paths = (
        (bundle_root / f"{bundle_hash}.json",)
        if bundle_hash is not None
        else (tuple(sorted(bundle_root.glob("*.json"))) if bundle_root.is_dir() else ())
    )
    for path in paths:
        try:
            bundle = HistoricalTradabilityBundle.model_validate(
                store.load_json(
                    category="bundles",
                    uri=store.uri("bundles", path.stem),
                    identity_field="bundle_hash",
                )
            )
            if bundle_hash is not None and bundle.bundle_hash != bundle_hash:
                raise ValueError("tradability bundle identity mismatch")
            if (
                bundle.universe_epoch_hash != risk_binding.return_epoch_hash
                or bundle.schedule_hash != expected_schedule_hash
                or bundle.formation_count != len(expected_sessions)
                or bundle.asset_count != len(risk_binding.ordered_listing_ids)
                or bundle.coverage_start != expected_sessions[0]
                or bundle.coverage_end != expected_sessions[-1]
            ):
                continue
            decision = HistoricalDecisionTradabilitySurface.model_validate(
                store.load_json(
                    category="decision/manifests",
                    uri=store.uri("decision/manifests", bundle.decision_surface_hash),
                    identity_field="surface_hash",
                )
            )
            execution = HistoricalExecutionAvailabilitySurface.model_validate(
                store.load_json(
                    category="execution/manifests",
                    uri=store.uri("execution/manifests", bundle.execution_surface_hash),
                    identity_field="surface_hash",
                )
            )
        except Exception as error:
            raise AuthoringError("alpha_research.portfolio_tradability_readback_failed") from error
        if (
            decision.formation_sessions == expected_sessions
            and execution.formation_sessions == expected_sessions
            and decision.intended_execution_sessions == expected_intended
            and execution.intended_execution_sessions == expected_intended
            and decision.ordered_listing_ids == risk_binding.ordered_listing_ids
            and execution.ordered_listing_ids == risk_binding.ordered_listing_ids
            and decision.universe_epoch_hash == risk_binding.return_epoch_hash
            and execution.universe_epoch_hash == risk_binding.return_epoch_hash
        ):
            matches.append((bundle, decision, execution))
    if len(matches) != 1:
        raise AuthoringError("alpha_research.portfolio_tradability_handle_ambiguous")
    return matches[0]


def resolve_fixed_portfolio_facts(
    *,
    roots: PanelMethodologySourceRoots,
    authority: ResolvedResearchAuthority,
    source_resolution_hash: str,
    execution_sessions: tuple[date, ...],
    development_overlay_method_id: str | None,
    alpha_panel_source_identity: AlphaPanelSourceIdentity | None = None,
    total_return_target_evidence_hash: str | None = None,
    outcome_method_binding_hash: str | None = None,
    panel_derivation_recipe_hash: str | None = None,
) -> ResolvedFixedPortfolioFacts:
    """Resolve only factual lanes needed by the frozen Portfolio consumer.

    This deliberately does not call the full Panel methodology source resolver:
    no Formula parquet, Feature cube, view recipe, or Feature materialization is
    opened. The target return lane, Panel/Sector identity, overlay manifest, and
    execution clock still come from their durable owners and are hash-checked.
    """
    exact_handles = (
        alpha_panel_source_identity,
        total_return_target_evidence_hash,
        outcome_method_binding_hash,
        panel_derivation_recipe_hash,
    )
    if any(value is None for value in exact_handles) != all(
        value is None for value in exact_handles
    ):
        raise AuthoringError("alpha_research.portfolio_fact_handle_group_invalid")
    exact_readback = alpha_panel_source_identity is not None
    artifact_root = roots.failed_baseline_workspace.resolve() / "artifacts"
    if exact_readback:
        target_evidence_hash = cast(str, total_return_target_evidence_hash)
    else:
        report, _report_hash = _read_baseline_report(roots.failed_baseline_workspace.resolve())
        dynamic_store = DynamicPanelArtifactStore(artifact_root)
        try:
            dossier = dynamic_store.load_model_dossier(str(report["model_dossier_hash"]))
            program = dynamic_store.load_model_program(dossier.program_hash)
            target_evidence_hash = dynamic_store.load_surface_manifest(
                program.dynamic_panel_surface_hash
            ).total_return_target_evidence_hash
        except (KeyError, ValueError, FileNotFoundError) as error:
            raise AuthoringError("alpha_research.failed_baseline_graph_unavailable") from error
    target_store = AlphaDevelopmentArtifactStore(artifact_root)
    try:
        evidence, _holding_sessions, simple_returns = (
            target_store.load_total_return_simple_return_lane(target_evidence_hash)
        )
        binding = target_store.load_total_return_target_binding(evidence.recipe_binding_hash)
    except (ValueError, FileNotFoundError) as error:
        raise AuthoringError("alpha_research.portfolio_target_lane_readback_failed") from error
    if (
        evidence.formation_sessions != authority.sessions
        or evidence.ordered_listing_ids != authority.ordered_listing_ids
    ):
        raise AuthoringError("alpha_research.corrected_target_common_axis_mismatch")

    resolver = ArtifactResolver(roots.panel_artifact_root.resolve())
    try:
        recipe, closure_store = _corrected_panel_recipe(
            resolver=resolver,
            snapshot_hash=authority.panel_snapshot_hash,
            recipe_hash=panel_derivation_recipe_hash,
        )
        sector_mapping = _corrected_sector_map(store=closure_store, recipe=recipe)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
        raise AuthoringError("alpha_research.corrected_panel_lineage_unavailable") from error
    if (
        set(recipe.listing_ids) != set(authority.ordered_listing_ids)
        or not set(authority.sessions).issubset(recipe.sessions)
        or set(authority.ordered_listing_ids) - set(sector_mapping)
    ):
        raise AuthoringError("alpha_research.corrected_panel_common_axis_mismatch")
    if alpha_panel_source_identity is None:
        try:
            manifest = resolver.load_feature_panel_manifest(authority.panel_manifest_ref)
            safe_summary = cast(dict[str, Any], manifest["safe_summary"])
            lineage = cast(dict[str, Any], safe_summary["lineage"])
        except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
            raise AuthoringError("alpha_research.corrected_panel_lineage_unavailable") from error
        overlay = _resolved_development_overlay_manifest(
            roots=roots,
            authority=authority,
            recipe=recipe,
            development_overlay_method_id=development_overlay_method_id,
        )
        factor_ids = tuple(
            sorted(
                (
                    *desktop_core_feature_bundle().factor_ids,
                    *(SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS if overlay is not None else ()),
                )
            )
        )
        source_identity = resolve_alpha_panel_source_identity(
            source_resolution_hash=source_resolution_hash,
            panel_snapshot_hash=authority.panel_snapshot_hash,
            ordered_session_axis_hash=str(
                canonical_hash([value.isoformat() for value in authority.sessions])
            ),
            ordered_listing_axis_hash=str(canonical_hash(list(authority.ordered_listing_ids))),
            ordered_factor_axis_hash=str(canonical_hash(list(factor_ids))),
            formula_observation_policy_hash=str(lineage["formula_observation_policy_hash"]),
            source_availability_policy_hash=source_availability_binding(
                str(cast(dict[str, Any], lineage["source_authorities"])["catalog_hash"])
            ),
            source_authority_binding_hash=str(lineage["source_authority_binding_hash"]),
            # The catalog the Panel was built under, a workspace's activations included (EX).
            catalog=feature_catalog_for(
                roots.panel_artifact_root.parent, str(lineage.get("catalog_hash", ""))
            ),
        )
    else:
        source_identity = alpha_panel_source_identity
        if (
            evidence.evidence_hash != total_return_target_evidence_hash
            or binding.outcome_method_binding_hash != outcome_method_binding_hash
            or source_identity.source_resolution_hash != source_resolution_hash
            or source_identity.panel_snapshot_hash != authority.panel_snapshot_hash
            or source_identity.ordered_session_axis_hash
            != canonical_hash([value.isoformat() for value in authority.sessions])
            or source_identity.ordered_listing_axis_hash
            != canonical_hash(list(authority.ordered_listing_ids))
            or not set(execution_sessions).issubset(evidence.formation_sessions)
        ):
            raise AuthoringError("alpha_research.portfolio_fact_handle_identity_mismatch")
    if roots.execution_outcome_artifact_root is None:
        raise AuthoringError("alpha_research.execution_outcome_root_missing")
    outcome_reader = CausalExecutionOutcomeDevelopmentReader(roots.execution_outcome_artifact_root)
    execution_clock = resolve_execution_clock_context(
        reader=outcome_reader,
        manifest_ref=outcome_reader.manifest_uri(binding.causal_outcome_snapshot_hash),
        sessions=execution_sessions,
        # Risk observes the oldest return window boundary at open(T-lookback-1).
        # The outcome row for the preceding formation is the owner that publishes
        # that open, hence one row beyond the authority's negative offset.
        history_margin_sessions=REQUIRED_LOOKBACK_SESSIONS + 2,
    )
    if (
        execution_clock.events.method_binding_hash != binding.outcome_method_binding_hash
        or execution_clock.events.execution_recipe_id != binding.execution_outcome_recipe_id
    ):
        raise AuthoringError("alpha_research.execution_clock_not_this_target")
    sealed_returns = _readonly(simple_returns)
    return ResolvedFixedPortfolioFacts(
        formation_sessions=evidence.formation_sessions,
        ordered_listing_ids=evidence.ordered_listing_ids,
        ordered_sector_ids=sector_ids(sector_mapping),
        sector_by_listing_id=sector_mapping,
        raw_simple_execution_returns=sealed_returns,
        total_return_target_evidence_hash=evidence.evidence_hash,
        outcome_method_binding_hash=binding.outcome_method_binding_hash,
        panel_derivation_recipe_hash=recipe.recipe_hash,
        sector_revision=recipe.sector_revision,
        execution_clock=execution_clock,
        alpha_panel_source_identity=source_identity,
    )


def resolve_fixed_portfolio_fact_authority(
    *,
    roots: PanelMethodologySourceRoots,
    authority: ResolvedResearchAuthority,
    source_resolution_hash: str,
    execution_sessions: tuple[date, ...],
    alpha_panel_source_identity: AlphaPanelSourceIdentity,
    total_return_target_evidence_hash: str,
    outcome_method_binding_hash: str,
    panel_derivation_recipe_hash: str,
) -> ResolvedFixedPortfolioFactAuthority:
    """Reopen the exact factual graph for preflight without its value lanes."""
    artifact_root = roots.failed_baseline_workspace.resolve() / "artifacts"
    target_store = AlphaDevelopmentArtifactStore(artifact_root)
    try:
        evidence = target_store.load_total_return_target_evidence(total_return_target_evidence_hash)
        binding = target_store.load_total_return_target_binding(evidence.recipe_binding_hash)
        resolver = ArtifactResolver(roots.panel_artifact_root.resolve())
        recipe, closure_store = _corrected_panel_recipe(
            resolver=resolver,
            snapshot_hash=authority.panel_snapshot_hash,
            recipe_hash=panel_derivation_recipe_hash,
        )
        sector_mapping = _corrected_sector_map(store=closure_store, recipe=recipe)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
        raise AuthoringError("alpha_research.portfolio_fact_metadata_readback_failed") from error
    source_identity = alpha_panel_source_identity
    if (
        evidence.evidence_hash != total_return_target_evidence_hash
        or binding.outcome_method_binding_hash != outcome_method_binding_hash
        or evidence.formation_sessions != authority.sessions
        or evidence.ordered_listing_ids != authority.ordered_listing_ids
        or set(recipe.listing_ids) != set(authority.ordered_listing_ids)
        or not set(authority.sessions).issubset(recipe.sessions)
        or set(authority.ordered_listing_ids) - set(sector_mapping)
        or source_identity.source_resolution_hash != source_resolution_hash
        or source_identity.panel_snapshot_hash != authority.panel_snapshot_hash
        or source_identity.ordered_session_axis_hash
        != canonical_hash([value.isoformat() for value in authority.sessions])
        or source_identity.ordered_listing_axis_hash
        != canonical_hash(list(authority.ordered_listing_ids))
        or not set(execution_sessions).issubset(evidence.formation_sessions)
    ):
        raise AuthoringError("alpha_research.portfolio_fact_handle_identity_mismatch")
    if roots.execution_outcome_artifact_root is None:
        raise AuthoringError("alpha_research.execution_outcome_root_missing")
    outcome_reader = CausalExecutionOutcomeDevelopmentReader(roots.execution_outcome_artifact_root)
    execution_clock = resolve_execution_clock_metadata(
        reader=outcome_reader,
        manifest_ref=outcome_reader.manifest_uri(binding.causal_outcome_snapshot_hash),
        sessions=execution_sessions,
        history_margin_sessions=REQUIRED_LOOKBACK_SESSIONS + 2,
    )
    if (
        execution_clock.events.method_binding_hash != binding.outcome_method_binding_hash
        or execution_clock.events.execution_recipe_id != binding.execution_outcome_recipe_id
    ):
        raise AuthoringError("alpha_research.execution_clock_not_this_target")
    return ResolvedFixedPortfolioFactAuthority(
        formation_sessions=evidence.formation_sessions,
        ordered_listing_ids=evidence.ordered_listing_ids,
        total_return_target_evidence_hash=evidence.evidence_hash,
        outcome_method_binding_hash=binding.outcome_method_binding_hash,
        panel_derivation_recipe_hash=recipe.recipe_hash,
        sector_revision=recipe.sector_revision,
        execution_clock=execution_clock,
        alpha_panel_source_identity=source_identity,
        metadata_bytes_read=(
            len(evidence.model_dump_json().encode("utf-8"))
            + len(binding.model_dump_json().encode("utf-8"))
            + len(recipe.model_dump_json().encode("utf-8"))
            + execution_clock.table.nbytes
        ),
    )


def _resolve_fixed_portfolio_reference_marks(
    *,
    market_workspace: Path,
    artifact_root: Path,
    playpen_root: Path,
    source: ResolvedFixedPortfolioFacts,
    risk_binding: RiskDevelopmentInputBinding,
    return_surface: CausalRiskReturnSurface,
    decision_sessions: tuple[date, ...],
    expected_state_transition: PortfolioStateTransitionBinding | None,
) -> ResolvedReferenceMark:
    """Publish once or reopen exactly the Market-owned close-mark authority."""

    if (
        not decision_sessions
        or decision_sessions != tuple(sorted(set(decision_sessions)))
        or not set(decision_sessions).issubset(risk_binding.formation_sessions)
    ):
        raise AuthoringError("alpha_research.portfolio_session_mark_axis_invalid")
    store = SessionMarkArtifactStore(artifact_root)
    try:
        if expected_state_transition is None:
            repository = MarketDataRepository(market_workspace)
            manifest = repository.load_universe_manifest_revision(
                risk_binding.universe_revision_sha256
            )
            listings = risk_binding.ordered_listing_ids
            if (
                manifest.profile.market_profile_id != return_surface.epoch.market_profile_id
                or not set(listings) <= {value.listing_id for value in manifest.listings}
            ):
                raise AuthoringError("alpha_research.portfolio_session_mark_epoch_invalid")
            watermark = str(
                repository.execution_source_watermark(
                    manifest, through=return_surface.last_formation_session
                )["watermark_hash"]
            )
            if watermark != return_surface.source_watermark_hash:
                raise AuthoringError("alpha_research.portfolio_session_mark_source_drifted")
            recipe_id = source.execution_clock.events.execution_recipe_id
            if recipe_id is None:
                raise AuthoringError("alpha_research.portfolio_execution_recipe_missing")
            recipe = build_installed_execution_outcome_method_catalog().resolve(recipe_id)
            try:
                surface = publish_market_session_marks(
                    market=repository,
                    store=store,
                    manifest=manifest,
                    listing_ids=listings,
                    sessions=decision_sessions,
                    corporate_action_identity=recipe.corporate_action_identity,
                    source_watermark_hash=watermark,
                    availability=resolve_session_mark_availability(),
                )
            except SessionMarkError as error:
                if str(error) == "market_data_ops.session_mark_source_absent":
                    raise AuthoringError(
                        "alpha_research.portfolio_session_mark_source_absent"
                    ) from error
                raise
        else:
            if expected_state_transition.mark_surface_hash is None:
                raise AuthoringError("alpha_research.portfolio_session_mark_handle_missing")
            surface = store.load_manifest(expected_state_transition.mark_surface_hash)
            recipe_id = source.execution_clock.events.execution_recipe_id
            if recipe_id is None:
                raise AuthoringError("alpha_research.portfolio_execution_recipe_missing")
            recipe = build_installed_execution_outcome_method_catalog().resolve(recipe_id)
        if (
            surface.epoch.universe_manifest_revision != risk_binding.universe_revision_sha256
            or surface.epoch.market_profile_id != return_surface.epoch.market_profile_id
            or not set(risk_binding.ordered_listing_ids) <= set(surface.epoch.ordered_listing_ids)
            or surface.source_watermark_hash != return_surface.source_watermark_hash
            or surface.corporate_action_identity != recipe.corporate_action_identity
            or decision_sessions[0] < surface.first_session
            or decision_sessions[-1] > surface.last_session
        ):
            raise AuthoringError("alpha_research.portfolio_session_mark_not_this_graph")
        binding = resolve_portfolio_state_transition_binding(
            surface=surface,
            rebalance_clock=EveryFormationClock().binding,
            playpen_root=playpen_root,
            execution_method_binding_hash=source.execution_clock.events.method_binding_hash,
        )
        if (
            expected_state_transition is not None
            and binding.binding_hash != expected_state_transition.binding_hash
        ):
            raise AuthoringError("alpha_research.portfolio_state_transition_not_this_graph")
        return resolve_reference_mark_lane(
            store=store,
            surface=surface,
            binding=binding,
            entry_session_by_formation=source.execution_clock.entry_session_by_formation,
            formation_sessions=decision_sessions,
            ordered_listing_ids=risk_binding.ordered_listing_ids,
        )
    except AuthoringError:
        raise
    except (SessionMarkError, ValueError) as error:
        raise AuthoringError("alpha_research.portfolio_session_mark_readback_failed") from error


def resolve_fixed_portfolio_market_authority(
    *,
    source: ResolvedFixedPortfolioFactAuthority,
    r0_covariance: ResolvedStageSixCovariance,
    r1_covariance: ResolvedStageSixCovariance | None,
    risk_return_artifact_root: Path,
    portfolio_artifact_root: Path,
    benchmark_artifact_root: Path,
    decision_sessions: tuple[date, ...],
    session_mark_artifact_root: Path,
    playpen_root: Path,
    risk_input_binding_hash: str,
    tradability_bundle_hash: str,
    risk_return_surface_hash: str,
    benchmark_surface_hash: str,
    state_transition: PortfolioStateTransitionBinding,
) -> ResolvedFixedPortfolioMarketAuthority:
    """Reopen exact Market/Risk manifests for preflight without value chunks."""
    r0_binding = _risk_input_binding(r0_covariance)
    sessions = r0_binding.formation_sessions
    listings = r0_binding.ordered_listing_ids
    if (
        r0_binding.input_binding_hash != risk_input_binding_hash
        or r0_covariance.formation_sessions != sessions
        or r0_covariance.ordered_listing_ids != listings
        or not set(listings).issubset(source.ordered_listing_ids)
        or r0_binding.panel_snapshot_hash != source.alpha_panel_source_identity.panel_snapshot_hash
        or (
            r1_covariance is not None
            and (
                _risk_input_binding(r1_covariance) != r0_binding
                or r1_covariance.formation_sessions != sessions
                or r1_covariance.ordered_listing_ids != listings
            )
        )
    ):
        raise AuthoringError("alpha_research.portfolio_market_authority_mismatch")
    try:
        return_surface = RiskReturnArtifactStore(risk_return_artifact_root).load_manifest(
            risk_return_surface_hash
        )
        bundle, decision_surface, execution_surface = _resolved_tradability_surfaces(
            artifact_root=portfolio_artifact_root,
            risk_binding=r0_binding,
            execution_clock=source.execution_clock,
            bundle_hash=tradability_bundle_hash,
        )
        benchmark = PortfolioBenchmarkStore(benchmark_artifact_root).load(benchmark_surface_hash)
        mark_store = SessionMarkArtifactStore(session_mark_artifact_root)
        if state_transition.mark_surface_hash is None:
            raise AuthoringError("alpha_research.portfolio_session_mark_handle_missing")
        mark_surface = mark_store.load_manifest(state_transition.mark_surface_hash)
        recipe_id = source.execution_clock.events.execution_recipe_id
        if recipe_id is None:
            raise AuthoringError("alpha_research.portfolio_execution_recipe_missing")
        recipe = build_installed_execution_outcome_method_catalog().resolve(recipe_id)
        resolved_transition = resolve_portfolio_state_transition_binding(
            surface=mark_surface,
            rebalance_clock=EveryFormationClock().binding,
            playpen_root=playpen_root,
            execution_method_binding_hash=source.execution_clock.events.method_binding_hash,
        )
    except AuthoringError:
        raise
    except (FileNotFoundError, SessionMarkError, ValueError) as error:
        raise AuthoringError("alpha_research.portfolio_market_metadata_readback_failed") from error
    expected_intended = tuple(
        source.execution_clock.entry_session_by_formation[session] for session in sessions
    )
    if (
        return_surface.surface_hash != r0_binding.return_surface_hash
        or return_surface.surface_hash != risk_return_surface_hash
        or return_surface.epoch.epoch_hash != r0_binding.return_epoch_hash
        or return_surface.epoch.panel_snapshot_hash != r0_binding.panel_snapshot_hash
        or return_surface.epoch.universe_manifest_revision != r0_binding.universe_revision_sha256
        or return_surface.epoch.ordered_listing_ids != listings
        or return_surface.source_watermark_hash != r0_binding.return_surface_source_watermark_hash
        or sessions[0] < return_surface.first_formation_session
        or sessions[-1] > return_surface.last_formation_session
        or decision_surface.formation_sessions != sessions
        or execution_surface.formation_sessions != sessions
        or decision_surface.intended_execution_sessions != expected_intended
        or execution_surface.intended_execution_sessions != expected_intended
        or benchmark.formation_sessions != sessions
        or benchmark.universe_manifest_revision != r0_binding.universe_revision_sha256
        or mark_surface.epoch.universe_manifest_revision != r0_binding.universe_revision_sha256
        or mark_surface.epoch.market_profile_id != return_surface.epoch.market_profile_id
        or not set(listings).issubset(mark_surface.epoch.ordered_listing_ids)
        or mark_surface.source_watermark_hash != return_surface.source_watermark_hash
        or mark_surface.corporate_action_identity != recipe.corporate_action_identity
        or mark_surface.first_session != decision_sessions[0]
        or mark_surface.last_session != decision_sessions[-1]
        or mark_surface.session_count != len(decision_sessions)
        or resolved_transition != state_transition
    ):
        raise AuthoringError("alpha_research.portfolio_market_metadata_not_this_graph")
    return ResolvedFixedPortfolioMarketAuthority(
        formation_sessions=sessions,
        ordered_listing_ids=listings,
        risk_input_binding_hash=r0_binding.input_binding_hash,
        risk_return_surface_hash=return_surface.surface_hash,
        tradability_bundle_hash=bundle.bundle_hash,
        universe_epoch_hash=r0_binding.return_epoch_hash,
        benchmark_surface_hash=benchmark.surface_hash,
        state_transition=resolved_transition,
        mark_surface=mark_surface,
        metadata_bytes_read=sum(
            len(value.model_dump_json().encode("utf-8"))
            for value in (
                return_surface,
                bundle,
                decision_surface,
                execution_surface,
                benchmark,
                mark_surface,
                resolved_transition,
            )
        ),
    )


def resolve_fixed_portfolio_market(
    *,
    source: ResolvedFixedPortfolioFacts,
    r0_covariance: ResolvedStageSixCovariance,
    r1_covariance: ResolvedStageSixCovariance | None,
    market_workspace: Path,
    risk_return_artifact_root: Path,
    portfolio_artifact_root: Path,
    benchmark_artifact_root: Path,
    decision_sessions: tuple[date, ...],
    session_mark_artifact_root: Path,
    playpen_root: Path,
    risk_input_binding_hash: str | None = None,
    tradability_bundle_hash: str | None = None,
    risk_return_surface_hash: str | None = None,
    benchmark_surface_hash: str | None = None,
    state_transition: PortfolioStateTransitionBinding | None = None,
) -> ResolvedFixedPortfolioMarket:
    """Compose fixed-Alpha Portfolio facts on Risk's 1,264-session state axis."""
    r0_binding = _risk_input_binding(r0_covariance)
    sessions = r0_binding.formation_sessions
    listings = r0_binding.ordered_listing_ids
    if (
        r0_covariance.formation_sessions != sessions
        or r0_covariance.ordered_listing_ids != listings
        or (
            risk_input_binding_hash is not None
            and r0_binding.input_binding_hash != risk_input_binding_hash
        )
        or not set(listings).issubset(source.ordered_listing_ids)
        or not set(listings).issubset(source.sector_by_listing_id)
        or r0_binding.panel_snapshot_hash != source.alpha_panel_source_identity.panel_snapshot_hash
        or (
            r1_covariance is not None
            and (
                _risk_input_binding(r1_covariance) != r0_binding
                or r1_covariance.formation_sessions != sessions
                or r1_covariance.ordered_listing_ids != listings
            )
        )
    ):
        raise AuthoringError("alpha_research.portfolio_market_authority_mismatch")

    return_store = RiskReturnArtifactStore(risk_return_artifact_root)
    try:
        return_surface = return_store.load_manifest(r0_binding.return_surface_hash)
        return_reader = CausalRiskReturnReader(risk_return_artifact_root)
        available_return_sessions = return_reader.available_sessions(return_surface)
    except Exception as error:
        raise AuthoringError("alpha_research.portfolio_return_surface_readback_failed") from error
    if (
        return_surface.epoch.epoch_hash != r0_binding.return_epoch_hash
        or return_surface.epoch.panel_snapshot_hash != r0_binding.panel_snapshot_hash
        or return_surface.epoch.universe_manifest_revision != r0_binding.universe_revision_sha256
        or return_surface.epoch.ordered_listing_ids != listings
        or return_surface.source_watermark_hash != r0_binding.return_surface_source_watermark_hash
        or (
            risk_return_surface_hash is not None
            and return_surface.surface_hash != risk_return_surface_hash
        )
        or not set(sessions).issubset(available_return_sessions)
    ):
        raise AuthoringError("alpha_research.portfolio_return_surface_not_this_risk_input")

    reference_marks = _resolve_fixed_portfolio_reference_marks(
        market_workspace=market_workspace,
        artifact_root=session_mark_artifact_root,
        playpen_root=playpen_root,
        source=source,
        risk_binding=r0_binding,
        return_surface=return_surface,
        decision_sessions=decision_sessions,
        expected_state_transition=state_transition,
    )

    bundle, decision_surface, execution_surface = _resolved_tradability_surfaces(
        artifact_root=portfolio_artifact_root,
        risk_binding=r0_binding,
        execution_clock=source.execution_clock,
        bundle_hash=tradability_bundle_hash,
    )
    decision, execution, adv20, intended = _tradability_matrices(
        artifacts=TradabilityArtifactStore(portfolio_artifact_root),
        decision=decision_surface,
        execution=execution_surface,
        sessions=sessions,
        listings=listings,
    )
    expected_intended = tuple(
        source.execution_clock.entry_session_by_formation[session] for session in sessions
    )
    if intended != expected_intended:
        raise AuthoringError("alpha_research.portfolio_tradability_clock_mismatch")

    source_positions = {session: index for index, session in enumerate(source.formation_sessions)}
    source_listing_positions = {
        listing_id: index for index, listing_id in enumerate(source.ordered_listing_ids)
    }
    try:
        rows: npt.NDArray[np.int64] = np.asarray(
            [source_positions[session] for session in sessions], dtype=np.int64
        )
        columns: npt.NDArray[np.int64] = np.asarray(
            [source_listing_positions[listing_id] for listing_id in listings], dtype=np.int64
        )
    except KeyError as error:
        raise AuthoringError("alpha_research.portfolio_outcome_axis_incomplete") from error
    realized = _readonly(source.raw_simple_execution_returns[np.ix_(rows, columns)])
    sector_ids = source.ordered_sector_ids
    # Each formation's exposure from the Sectors in force at it (V346).
    sector_exposure, equal_sector = sector_exposures(
        listings, sector_ids, source.sector_by_listing_id, sessions=sessions
    )
    market = ImmutablePortfolioMarketInputs(
        formation_sessions=sessions,
        economic_formation_sessions=sessions,
        ordered_listing_ids=listings,
        sector_ids=sector_ids,
        sector_exposure_matrix=sector_exposure,
        equal_weight_sector_exposure=equal_sector,
        decision_eligible=decision,
        execution_available=execution,
        realized_simple_returns=realized,
        passive_returns_by_session={},
        causal_adv20=adv20,
        tradability_bundle_hash=bundle.bundle_hash,
        universe_epoch_hash=r0_binding.return_epoch_hash,
        sector_revision=source.sector_revision,
        source_surface_hash=str(
            canonical_hash(
                {
                    "risk_input_binding_hash": r0_binding.input_binding_hash,
                    "risk_return_surface_hash": return_surface.surface_hash,
                    "target_evidence_hash": source.total_return_target_evidence_hash,
                    "outcome_method_binding_hash": source.outcome_method_binding_hash,
                    "tradability_bundle_hash": bundle.bundle_hash,
                    "sector_revision": source.sector_revision,
                    **reclassification_payload(source.sector_by_listing_id),
                    "realized_return_identity": _array_identity(realized),
                }
            )
        ),
    )
    benchmark_store = PortfolioBenchmarkStore(benchmark_artifact_root)
    if benchmark_surface_hash is None:
        benchmark_surface = build_portfolio_benchmark_surface(
            workspace=market_workspace,
            formation_sessions=sessions,
        )
        benchmark_store.publish(benchmark_surface)
    else:
        benchmark_surface = benchmark_store.load(benchmark_surface_hash)
    if benchmark_surface.universe_manifest_revision != r0_binding.universe_revision_sha256:
        raise AuthoringError("alpha_research.portfolio_benchmark_universe_mismatch")
    benchmark_log_returns = tuple(float(value) for value in benchmark_surface.log_returns)
    benchmark_simple_returns = tuple(
        float(value) for value in np.expm1(np.asarray(benchmark_log_returns, dtype=np.float64))
    )
    benchmark = ImmutablePortfolioBenchmark(
        formation_sessions=sessions,
        economic_formation_sessions=sessions,
        simple_returns=benchmark_simple_returns,
        log_returns=benchmark_log_returns,
        economic_simple_returns=benchmark_simple_returns,
        economic_log_returns=benchmark_log_returns,
        surface_hash=benchmark_surface.surface_hash,
    )
    return ResolvedFixedPortfolioMarket(
        market=market,
        benchmark=benchmark,
        risk_input_binding_hash=r0_binding.input_binding_hash,
        tradability_bundle_hash=bundle.bundle_hash,
        risk_return_surface_hash=return_surface.surface_hash,
        benchmark_surface_hash=benchmark_surface.surface_hash,
        reference_marks=reference_marks,
    )


def _observation_panel(
    formula_values: FloatArray, factor_ids: tuple[str, ...], factor_id: str
) -> FloatArray | None:
    """One Formula's panel, or absent when the Panel does not carry it.

    A Panel published before a Formula was installed simply has no column for
    it, and the market aggregates that read it are then absent rather than a
    refusal -- the view catalog declares the narrower market axis and preflight
    reports the column count it actually got.
    """

    if factor_id not in factor_ids:
        return None
    return _readonly(formula_values[:, :, factor_ids.index(factor_id)])


def _formation_observation_dollar_volume(
    *, workspace: Path, sessions: tuple[date, ...], listing_ids: tuple[str, ...]
) -> tuple[FloatArray, str]:
    """Read the Formula-owned raw close-times-volume for Market Context.

    The transport lane is not a second Formula: it reuses the stock Formula's
    arithmetic owner, then seals its own source axis for the Market aggregate.
    It performs no write or temporal projection.
    """

    if (
        not sessions
        or sessions != tuple(sorted(set(sessions)))
        or not listing_ids
        or len(set(listing_ids)) != len(listing_ids)
    ):
        raise AuthoringError("alpha_research.observation_dollar_volume_axis_invalid")
    repository = MarketDataRepository(workspace)
    rows = repository.raw_close_volume_observations(
        listing_ids,
        start=sessions[0],
        through=sessions[-1],
    )
    expected_row_count = len(sessions) * len(listing_ids)
    if rows.column_names != ["listing_id", "session_date", "close", "volume"]:
        raise AuthoringError("alpha_research.observation_dollar_volume_schema_invalid")
    if rows.num_rows != expected_row_count:
        raise AuthoringError("alpha_research.observation_dollar_volume_axis_incomplete")
    canonical_listings = tuple(sorted(listing_ids))
    observed_sessions = np.asarray(rows.column("session_date").to_numpy())
    observed_listings = np.asarray(rows.column("listing_id").to_numpy())
    expected_sessions: npt.NDArray[np.datetime64] = np.repeat(
        np.asarray(sessions, dtype="datetime64[D]"), len(listing_ids)
    )
    expected_listings = np.tile(np.asarray(canonical_listings, dtype=object), len(sessions))
    if not np.array_equal(observed_sessions, expected_sessions) or not np.array_equal(
        observed_listings, expected_listings
    ):
        raise AuthoringError("alpha_research.observation_dollar_volume_axis_invalid")
    close = np.asarray(rows.column("close").to_numpy(), dtype=np.float64)
    volume = np.asarray(rows.column("volume").to_numpy(), dtype=np.float64)
    canonical_values = raw_session_dollar_volume(close, volume).reshape(
        len(sessions), len(listing_ids)
    )
    canonical_position = {value: index for index, value in enumerate(canonical_listings)}
    values = canonical_values[:, [canonical_position[value] for value in listing_ids]]
    if not np.isfinite(values).all():
        raise AuthoringError("alpha_research.observation_dollar_volume_axis_incomplete")
    sealed = _readonly(values)
    return sealed, str(
        canonical_hash(
            {
                "source_lane_id": FORMATION_OBSERVATION_DOLLAR_VOLUME_SOURCE_LANE_ID,
                "owner": "market_data_ops.raw_daily_bar_current",
                "session_axis": [value.isoformat() for value in sessions],
                "listing_axis": list(listing_ids),
                "values_hash": _array_identity(sealed),
            }
        )
    )


def _resolved_development_overlay_manifest(
    *,
    roots: PanelMethodologySourceRoots,
    authority: ResolvedResearchAuthority,
    recipe: PanelDerivationRecipe,
    development_overlay_method_id: str | None,
) -> DevelopmentFeatureOverlayManifest | None:
    if development_overlay_method_id is None:
        return None
    if development_overlay_method_id != "SESSION_OBSERVATION_FORMULA_OVERLAY":
        raise AuthoringError("alpha_research.panel_development_overlay_method_not_installed")
    if roots.development_overlay_artifact_root is None:
        raise AuthoringError("alpha_research.panel_development_overlay_root_missing")
    binding = FeaturePanelBinding.create(
        manifest_revision=recipe.manifest_revision,
        sector_revision=recipe.sector_revision,
        catalog_hash=recipe.catalog_hash,
        spy_revision=recipe.spy_revision,
        policy_hash=recipe.policy_hash,
    )
    if binding.panel_binding_hash != recipe.panel_binding_hash:
        raise AuthoringError("alpha_research.panel_development_overlay_base_binding_mismatch")
    required = set(SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS)
    receipts = tuple(
        value for value in admit_factor_development_capabilities() if value.factor_id in required
    )
    if {value.factor_id for value in receipts if value.disposition == "ADMITTED"} != required:
        raise AuthoringError("alpha_research.panel_development_overlay_admission_incomplete")
    service = DevelopmentFeatureOverlayService()
    manifest = service.find_exact(
        output_root=roots.development_overlay_artifact_root,
        base_panel_snapshot_hash=authority.panel_snapshot_hash,
        source_identity=authority.source_watermark_hash,
        panel_binding=binding,
        admission_receipts=receipts,
    )
    if manifest is None:
        raise AuthoringError("alpha_research.panel_development_overlay_unresolved")
    manifest = load_development_feature_overlay_manifest(
        output_root=roots.development_overlay_artifact_root,
        overlay_hash=manifest.overlay_hash,
    )
    if (
        manifest.ordered_candidate_axis
        != tuple(sorted(SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS))
        or manifest.ordered_listing_axis != authority.ordered_listing_ids
        or not set(authority.sessions).issubset(
            date.fromisoformat(value) for value in manifest.ordered_session_axis
        )
        or manifest.raw_parquet_relative_path is None
        or tuple(value.admission_receipt_hash for value in manifest.columns)
        != tuple(value.receipt_hash for value in sorted(receipts, key=lambda item: item.factor_id))
    ):
        raise AuthoringError("alpha_research.panel_development_overlay_axis_mismatch")
    return manifest


def _development_overlay_raw_values(
    *,
    roots: PanelMethodologySourceRoots,
    authority: ResolvedResearchAuthority,
    recipe: PanelDerivationRecipe,
    base_rows: pd.DataFrame,
    development_overlay_method_id: str | None,
) -> tuple[pd.DataFrame, str | None]:
    manifest = _resolved_development_overlay_manifest(
        roots=roots,
        authority=authority,
        recipe=recipe,
        development_overlay_method_id=development_overlay_method_id,
    )
    if manifest is None:
        return base_rows, None
    try:
        verified_manifest, raw = read_raw_development_feature_overlay(
            output_root=roots.development_overlay_artifact_root,
            overlay_hash=manifest.overlay_hash,
        )
    except ValueError as error:
        raise AuthoringError("alpha_research.panel_development_overlay_readback_failed") from error
    if verified_manifest != manifest:
        raise AuthoringError("alpha_research.panel_development_overlay_identity_mismatch")
    raw["session_date"] = pd.to_datetime(raw["session_date"]).dt.date
    raw = raw.loc[raw["session_date"].isin(authority.sessions)].sort_values(
        ["session_date", "listing_id"], kind="mergesort"
    )
    expected_rows = len(authority.sessions) * len(authority.ordered_listing_ids)
    if len(raw) != expected_rows or raw.duplicated(["session_date", "listing_id"]).any():
        raise AuthoringError("alpha_research.panel_development_overlay_row_axis_mismatch")
    joined = base_rows.merge(
        raw.loc[
            :,
            [
                "session_date",
                "listing_id",
                *SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS,
            ],
        ],
        on=["session_date", "listing_id"],
        how="left",
        validate="one_to_one",
    )
    return joined, manifest.overlay_hash


def resolve_corrected_feature_t_panel_source(
    *,
    roots: PanelMethodologySourceRoots,
    authority: ResolvedResearchAuthority,
    input_method_id: str,
    development_overlay_method_id: str | None = None,
) -> ResolvedPanelMethodologySource:
    """Compose corrected Formula(T) values with independent target/market lanes."""
    if input_method_id != _CORRECTED_INPUT_METHOD_ID:
        raise AuthoringError("alpha_research.panel_input_method_not_installed")
    resolver = ArtifactResolver(roots.panel_artifact_root.resolve())
    try:
        manifest = resolver.load_feature_panel_manifest(authority.panel_manifest_ref)
    except (FileNotFoundError, ValueError, KeyError) as error:
        raise AuthoringError("alpha_research.corrected_panel_manifest_unavailable") from error
    safe_summary = manifest.get("safe_summary")
    lineage = safe_summary.get("lineage") if isinstance(safe_summary, dict) else None
    factor_summary = (
        safe_summary.get("factor_catalog_summary") if isinstance(safe_summary, dict) else None
    )
    if not isinstance(lineage, dict) or not isinstance(factor_summary, dict):
        raise AuthoringError("alpha_research.corrected_panel_lineage_unavailable")
    published_factor_ids = tuple(sorted(str(value) for value in factor_summary))
    base_factor_ids = tuple(desktop_core_feature_bundle().factor_ids)
    recipe, closure_store = _corrected_panel_recipe(
        resolver=resolver, snapshot_hash=authority.panel_snapshot_hash
    )
    sector_mapping = _corrected_sector_map(store=closure_store, recipe=recipe)
    if (
        set(authority.ordered_listing_ids) != set(recipe.listing_ids)
        or not set(authority.sessions).issubset(recipe.sessions)
        or published_factor_ids != tuple(sorted(recipe.factor_ids))
        or not set(base_factor_ids).issubset(recipe.factor_ids)
    ):
        raise AuthoringError("alpha_research.corrected_panel_common_axis_mismatch")

    rematerializer = ArtifactOnlyPanelRematerializer(resolver=resolver, store=closure_store)
    try:
        formula_table = rematerializer.read_base_formula_values(
            recipe.recipe_hash,
            sessions=authority.sessions,
            listing_ids=authority.ordered_listing_ids,
            factor_ids=base_factor_ids,
        )
    except (FileNotFoundError, ValueError, PanelRematerializationMismatch) as error:
        raise AuthoringError("alpha_research.corrected_formula_closure_unavailable") from error
    formula_frame = formula_table.to_pandas().sort_values(
        ["session_date", "listing_id"], kind="mergesort"
    )
    formula_frame, development_overlay_hash = _development_overlay_raw_values(
        roots=roots,
        authority=authority,
        recipe=recipe,
        base_rows=formula_frame,
        development_overlay_method_id=development_overlay_method_id,
    )
    factor_ids = tuple(
        sorted(
            (
                *base_factor_ids,
                *(
                    SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS
                    if development_overlay_hash is not None
                    else ()
                ),
            )
        )
    )
    formula_values = _readonly(
        formula_frame.loc[:, list(factor_ids)]
        .to_numpy(dtype=np.float64)
        .reshape(len(authority.sessions), len(authority.ordered_listing_ids), len(factor_ids))
    )

    legacy_snapshot = resolve_failed_baseline_panel_snapshot_hash(roots=roots)
    legacy_authority = ResolvedResearchAuthority.create(
        data_snapshot_handle="panel-remediation-baseline",
        universe_handle=authority.universe_handle,
        panel_snapshot_hash=legacy_snapshot,
        panel_manifest_ref=f"playpen://feature-panel/manifests/{legacy_snapshot}",
        universe_revision_sha256=authority.universe_revision_sha256,
        ordered_listing_ids=authority.ordered_listing_ids,
        sessions=authority.sessions,
        source_watermark_hash=authority.source_watermark_hash,
    )
    independent = _resolve_failed_baseline_independent_source(
        roots=roots, authority=legacy_authority
    )
    old = independent.old
    if (
        old.formation_sessions != authority.sessions
        or old.ordered_listing_ids != authority.ordered_listing_ids
    ):
        raise AuthoringError("alpha_research.corrected_target_common_axis_mismatch")
    ordered_sector_ids = sector_ids(sector_mapping)
    sector_states, sector_manifest_hash = _corrected_sector_states(
        sessions=authority.sessions,
        holding_end_sessions=old.holding_end_sessions,
        listing_ids=authority.ordered_listing_ids,
        raw_log_returns=_readonly(old.raw_log_returns),
        target_evidence_hash=independent.total_return_target_evidence_hash,
        sector_by_listing_id=sector_mapping,
        sector_revision=recipe.sector_revision,
    )
    market_interactions, market_interaction_hash = _formation_available_market_interaction_states(
        workspace=roots.panel_artifact_root.resolve().parent,
        sessions=authority.sessions,
    )
    observation_dollar_volume, observation_dollar_volume_hash = (
        _formation_observation_dollar_volume(
            workspace=roots.panel_artifact_root.resolve().parent,
            sessions=authority.sessions,
            listing_ids=authority.ordered_listing_ids,
        )
    )
    sector_context, market_context = assemble_panel_context_arrays(
        formation_sessions=authority.sessions,
        holding_end_sessions=old.holding_end_sessions,
        ordered_listing_ids=authority.ordered_listing_ids,
        ordered_sector_ids=ordered_sector_ids,
        sector_by_listing_id=sector_mapping,
        raw_log_execution_returns=_readonly(old.raw_log_returns),
        raw_simple_execution_returns=_readonly(old.simple_economic_returns),
        sector_state_values=sector_states,
        market_interaction_state_values=market_interactions,
        # The market aggregates of the close-to-close observation, available at
        # the formation session rather than three behind it. Absent for a Panel
        # published before the Formula was installed, and the view catalog then
        # declares eight market sources instead of ten.
        observation_returns=_observation_panel(formula_values, factor_ids, "close_to_close"),
        observation_volume_state=_observation_panel(formula_values, factor_ids, "volume_zscore_21"),
        observation_high_distance=_observation_panel(formula_values, factor_ids, "dist_52w_high"),
        observation_low_distance=_observation_panel(formula_values, factor_ids, "dist_52w_low"),
        observation_dollar_volume=observation_dollar_volume,
    )
    factor_axis_admission_hash = str(
        canonical_hash(
            {
                "panel_snapshot_hash": authority.panel_snapshot_hash,
                "panel_derivation_recipe_hash": recipe.recipe_hash,
                "base_formula_closure_hash": recipe.base_closure_hash,
                "factor_ids": list(factor_ids),
                "development_overlay_hash": development_overlay_hash,
                "semantics": "PRE_PANEL_FORMULA_VALUES_FROM_IDENTITY_CHECKED_CLOSURE",
            }
        )
    )
    arrays = PanelFeatureSourceArrays(
        formation_sessions=authority.sessions,
        holding_end_sessions=old.holding_end_sessions,
        ordered_listing_ids=authority.ordered_listing_ids,
        ordered_factor_ids=factor_ids,
        absolute_state_factor_ids=factor_ids,
        ordered_sector_ids=ordered_sector_ids,
        sector_by_listing_id=sector_mapping,
        raw_formula_values=formula_values,
        total_return_target_z=_readonly(old.target_z_values),
        raw_log_execution_returns=_readonly(old.raw_log_returns),
        raw_simple_execution_returns=_readonly(old.simple_economic_returns),
        sector_context_values=sector_context,
        market_context_values=market_context,
        source_identity_hashes=MappingProxyType(
            {
                "panel": authority.panel_snapshot_hash,
                "panel_recipe": recipe.recipe_hash,
                "raw_formula": recipe.base_closure_hash,
                "factor_axis_admission": factor_axis_admission_hash,
                "target": independent.total_return_target_evidence_hash,
                "sector_context": sector_manifest_hash,
                "market_interactions": market_interaction_hash,
                FORMATION_OBSERVATION_DOLLAR_VOLUME_SOURCE_LANE_ID: (
                    observation_dollar_volume_hash
                ),
                **(
                    {"development_overlay": development_overlay_hash}
                    if development_overlay_hash is not None
                    else {}
                ),
                "outcome_method": independent.outcome_method_binding_hash,
            }
        ),
    )
    resolution = PanelMethodologySourceResolution.create(
        input_method_id=input_method_id,
        baseline_report_hash=independent.baseline_report_hash,
        panel_snapshot_hash=authority.panel_snapshot_hash,
        development_overlay_hash=development_overlay_hash,
        panel_derivation_recipe_hash=recipe.recipe_hash,
        panel_base_formula_closure_hash=recipe.base_closure_hash,
        factor_axis_admission_hash=factor_axis_admission_hash,
        formula_observation_policy_hash=str(lineage["formula_observation_policy_hash"]),
        source_availability_policy_hash=source_availability_binding(
            str(dict(lineage["source_authorities"])["catalog_hash"])
        ),
        source_authority_binding_hash=str(lineage["source_authority_binding_hash"]),
        total_return_target_evidence_hash=(independent.total_return_target_evidence_hash),
        sector_revision=recipe.sector_revision,
        execution_outcome_recipe_id=independent.execution_outcome_recipe_id,
        outcome_method_binding_hash=independent.outcome_method_binding_hash,
        sector_context_manifest_hash=sector_manifest_hash,
        ordered_session_axis_hash=str(
            canonical_hash([value.isoformat() for value in authority.sessions])
        ),
        ordered_listing_axis_hash=str(canonical_hash(list(authority.ordered_listing_ids))),
        ordered_factor_axis_hash=str(canonical_hash(list(factor_ids))),
        network_access_count=0,
        provider_access_count=0,
        holdout_access_count=0,
        current_or_production_pointer_read_count=0,
        pointer_mutation_count=0,
        source_workspace_write_count=0,
    )
    return ResolvedPanelMethodologySource(
        arrays=arrays,
        resolution=resolution,
        outer_folds=independent.outer_folds,
        portfolio_market=_rebind_portfolio_sector_authority(
            market=independent.portfolio_market,
            panel_snapshot_hash=authority.panel_snapshot_hash,
            sector_revision=recipe.sector_revision,
            sector_by_listing_id=sector_mapping,
            ordered_sector_ids=ordered_sector_ids,
        ),
        portfolio_benchmark=independent.portfolio_benchmark,
        portfolio_fold_indices=independent.portfolio_fold_indices,
        execution_clock=independent.execution_clock,
    )


def resolve_installed_panel_methodology_source(
    *,
    roots: PanelMethodologySourceRoots,
    authority: ResolvedResearchAuthority,
    input_method_id: str,
    development_overlay_method_id: str | None = None,
) -> ResolvedPanelMethodologySource:
    """Dispatch the exact installed input method to its deterministic source resolver.

    Args:
        roots: Explicit declared methodology source locations.
        authority: Exact resolved research authority.
        input_method_id: Explicit installed input method.
        development_overlay_method_id: Optional exact development overlay method.

    Returns:
        Verified corrected-source or failed-experiment baseline source and exact execution context.
    """
    if input_method_id == _CORRECTED_INPUT_METHOD_ID:
        return resolve_corrected_feature_t_panel_source(
            roots=roots,
            authority=authority,
            input_method_id=input_method_id,
            development_overlay_method_id=development_overlay_method_id,
        )
    return resolve_failed_experiment_baseline_source(
        roots=roots, authority=authority, input_method_id=input_method_id
    )


def resolve_failed_baseline_panel_snapshot_hash(*, roots: PanelMethodologySourceRoots) -> str:
    """Resolve the immutable Panel identity behind the historical baseline handle."""
    report, _report_hash = _read_baseline_report(roots.failed_baseline_workspace.resolve())
    store = DynamicPanelArtifactStore(roots.failed_baseline_workspace.resolve() / "artifacts")
    try:
        dossier = store.load_model_dossier(str(report["model_dossier_hash"]))
        program = store.load_model_program(dossier.program_hash)
        surface = store.load_surface_manifest(program.dynamic_panel_surface_hash)
        methodology = load_development_methodology_surface_manifest(
            output_root=roots.feature_artifact_root,
            surface_hash=surface.relative_surface_hash,
        )
    except (KeyError, ValueError, FileNotFoundError) as error:
        raise AuthoringError("alpha_research.failed_baseline_graph_unavailable") from error
    return str(methodology.base_panel_snapshot_hash)


def resolve_corrected_panel_snapshot_hash(*, roots: PanelMethodologySourceRoots) -> str:
    """Resolve the unique installed Panel carrying a complete derivation closure."""
    resolver = ArtifactResolver(roots.panel_artifact_root.resolve())
    manifest_root = roots.panel_artifact_root.resolve() / "feature-panel" / "manifests"
    matches: list[str] = []
    for path in sorted(manifest_root.glob("*.json")):
        try:
            recipe, _store = _corrected_panel_recipe(resolver=resolver, snapshot_hash=path.stem)
            manifest = resolver.load_feature_panel_manifest(
                resolver.feature_panel_manifest_uri(path.stem)
            )
        except (FileNotFoundError, KeyError, ValueError, AuthoringError):
            continue
        if recipe.snapshot_hash == str(manifest.get("snapshot_hash", "")):
            matches.append(path.stem)
    if len(matches) != 1:
        raise AuthoringError("alpha_research.corrected_panel_handle_ambiguous")
    return matches[0]


def resolve_installed_development_covariance(
    *,
    evidence_root: Path,
    method_id: Literal["R0", "R1"],
    surface_hash: str | None = None,
) -> ResolvedStageSixCovariance:
    """Resolve an installed Risk method without caller-supplied artifact identity.

    A Risk evidence root may legitimately contain a full candidate domain.  R0
    is the installed EWMA/Ledoit-Wolf control; R1 is the predeclared fast/slow
    challenger with blend weight 0.75.  The Host derives the unique matching
    surface from its typed descriptor rather than requiring a hash in YAML.
    """
    category = evidence_root.resolve() / "risk-research" / DEVELOPMENT_SURFACE_CATEGORY
    paths = (
        (category / f"{surface_hash}.json",)
        if surface_hash is not None
        else tuple(sorted(category.glob("*.json")))
    )
    matches: list[RiskDevelopmentCovarianceSurface] = []
    for path in paths:
        try:
            surface = RiskDevelopmentCovarianceSurface.model_validate_json(path.read_bytes())
        except Exception as error:
            raise AuthoringError("research_authoring.risk_method_readback_failed") from error
        parameters = dict(surface.recipe_envelope.parameters)
        admitted = (
            method_id == "R0"
            and surface.capability_handle == "EWMA_STANDARDIZED_LEDOIT_WOLF_CORRELATION"
        ) or (
            method_id == "R1"
            and surface.capability_handle == "FAST_SLOW_LEDOIT_WOLF_CORRELATION"
            and parameters.get("blend_weight") == 0.75
        )
        if admitted and (surface_hash is None or surface.surface_hash == surface_hash):
            matches.append(surface)
    if len(matches) != 1:
        raise AuthoringError("research_authoring.risk_method_handle_ambiguous")
    return load_stage_six_covariance(
        evidence_root=evidence_root.resolve(), surface_hash=matches[0].surface_hash
    )


__all__ = [
    "PanelMethodologyOuterFold",
    "PanelMethodologySourceResolution",
    "PanelMethodologySourceRoots",
    "ResolvedFixedPortfolioFactAuthority",
    "ResolvedFixedPortfolioFacts",
    "ResolvedFixedPortfolioMarket",
    "ResolvedFixedPortfolioMarketAuthority",
    "ResolvedPanelMethodologySource",
    "resolve_corrected_feature_t_panel_source",
    "resolve_corrected_panel_snapshot_hash",
    "resolve_failed_baseline_panel_snapshot_hash",
    "resolve_failed_experiment_baseline_source",
    "resolve_fixed_portfolio_fact_authority",
    "resolve_fixed_portfolio_facts",
    "resolve_fixed_portfolio_market",
    "resolve_fixed_portfolio_market_authority",
    "resolve_installed_development_covariance",
    "resolve_installed_panel_methodology_source",
]
