"""Install the executor for one selected Desk kind over a workspace, read-only.

A Desk executor needs inputs only a workspace can answer for -- for Risk the
published causal return surface and the current sector map, for Factor the
published Panel and causal outcomes, for Alpha those plus a Factor development
checkpoint. Reading them is the composition root's job, not the Desk's:
``experiments/execution.py`` should not learn where a workspace keeps artifacts.

Installation is **per kind**. This used to build every installed executor
eagerly, which meant resolving Risk's return surface and sector state before
anything ran -- so a Factor experiment in a workspace that had never published a
return surface failed on an artifact it does not read. Resolving one Desk's
authority is not a precondition for another's.

Nothing here takes a writer lease on the source workspace and nothing moves a
pointer. A development run reads what the workspace already published; a missing
input is reported rather than published on the caller's behalf.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np

from alphalattice.capabilities.alpha_modeling.catalog import build_installed_alpha_model_catalog
from alphalattice.control.product_host.research_authoring.authority import (
    InstalledResearchSnapshot,
    WorkspaceResearchAuthorityResolver,
    exploration_sample_size,
    panel_sector_labels,
    universe_profile,
)
from alphalattice.control.product_host.research_authoring.feature_activations import (
    panel_feature_catalog,
)
from alphalattice.control.product_host.research_authoring.foundation import (
    build_alpha_development_foundation,
)
from alphalattice.control.product_host.research_authoring.model_extensions import (
    activated_models,
)
from alphalattice.control.product_host.research_authoring.panel_methodology_sources import (
    PanelMethodologySourceRoots,
    ResolvedFixedPortfolioFactAuthority,
    ResolvedFixedPortfolioFacts,
    ResolvedFixedPortfolioMarket,
    ResolvedFixedPortfolioMarketAuthority,
    resolve_corrected_panel_snapshot_hash,
    resolve_failed_baseline_panel_snapshot_hash,
    resolve_fixed_portfolio_fact_authority,
    resolve_fixed_portfolio_facts,
    resolve_fixed_portfolio_market,
    resolve_fixed_portfolio_market_authority,
    resolve_installed_development_covariance,
    resolve_installed_panel_methodology_source,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
    CausalOutcomeDevelopmentRows,
    DevelopmentOnlyExecutionOutcomeReader,
)
from alphalattice.foundation.factor_research.experiments.authoring import (
    FACTOR_EXPERIMENT_KIND,
)
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.contracts import panel_source_manifest_revision
from alphalattice.foundation.feature_engine.panels.development_input import (
    ResolvedDevelopmentFeatureInput,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.foundation.feature_engine.producers.factors.registry import (
    FeatureKernelRegistry,
)
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.investment.alpha_research.experiments.authoring import (
    ALPHA_EXPERIMENT_KIND,
    AlphaExperimentCompiler,
)
from alphalattice.investment.alpha_research.experiments.contracts import AlphaSplitPolicy
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactReadbackError,
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.experiments.development_execution import (
    AlphaExperimentExecutor,
)
from alphalattice.investment.alpha_research.experiments.development_mandate import (
    build_development_alpha_model_mandate,
)
from alphalattice.investment.alpha_research.experiments.lifecycle_authoring import (
    AlphaLifecycleExperiment,
    lifecycle_section,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_authoring import (
    PanelResearchMethodologyCompiler,
    PanelResearchMethodologyRequest,
    PanelResearchPreflight,
    PanelResearchRuntimeCaps,
    build_panel_research_preflight,
    methodology_section,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (
    preflight_panel_feature_plan,
)
from alphalattice.investment.alpha_research.scores.temporal_aggregation import (
    SEALED_PANEL_ALPHA_DECISION_SESSION_COUNT,
    SEALED_PANEL_ALPHA_FEATURE_COLUMN_COUNT,
    SEALED_PANEL_ALPHA_MODEL_RECIPE_ID,
    SEALED_PANEL_ALPHA_ROOT_HASH,
    SEALED_PANEL_ALPHA_VIEW_ID,
    AlphaPanelSourceIdentity,
    resolve_alpha_panel_source_identity,
)
from alphalattice.investment.alpha_research.simple_signal.authority import (
    WorkspaceSimpleSignalMaterializationAuthority,
    WorkspaceSimpleSignalSource,
)
from alphalattice.investment.alpha_research.simple_signal.contracts import (
    feature_id_for_simple_score_method,
)
from alphalattice.investment.alpha_research.targets.authority import (
    installed_alpha_target_methods,
)
from alphalattice.investment.portfolio_strategy_lab.policies.score_risk_cost import (
    ScoreRiskCostDevelopmentRecipe,
)
from alphalattice.investment.portfolio_strategy_lab.research_loop import (
    panel_methodology_execution,
)
from alphalattice.investment.portfolio_strategy_lab.research_loop.fixed_panel_score import (
    fixed_panel_alpha_source_identity,
)
from alphalattice.investment.portfolio_strategy_lab.research_loop.paired_alpha_portfolio import (
    portfolio_solver_call_upper_bound,
    portfolio_study_formation_limit,
    rank_buffered_r0_inner_candidate_set_hash,
)
from alphalattice.investment.risk_research.contracts import CausalRiskReturnSurface
from alphalattice.investment.risk_research.experiments.compiler import RISK_EXPERIMENT_KIND
from alphalattice.investment.risk_research.experiments.execution import RiskExperimentExecutor
from alphalattice.investment.risk_research.surfaces.returns import CausalRiskReturnReader
from alphalattice.kernel.quant.sector_history import SectorHistory
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.sector_treatment import sector_treatment
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskExperimentExecutor,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
)


@dataclass(frozen=True, slots=True)
class PanelPortfolioPreflightSource:
    """Host-installed exact handles for the fixed development Portfolio study.

    This is operational composition, not a second fact surface: each handle is
    immediately reopened and identity-checked by the original Alpha, Panel,
    Risk, Market, and execution owners before a Program receipt can be sealed.
    In particular, the Alpha source identity carries no Portfolio schedule.
    """

    r0_surface_hash: str
    r1_surface_hash: str | None
    alpha_panel_source_identity: AlphaPanelSourceIdentity
    total_return_target_evidence_hash: str
    outcome_method_binding_hash: str
    panel_derivation_recipe_hash: str
    risk_input_binding_hash: str
    risk_return_surface_hash: str
    market_tradability_bundle_hash: str
    market_universe_epoch_hash: str
    market_sector_revision: str
    market_source_surface_hash: str
    benchmark_surface_hash: str
    state_transition_payload: Mapping[str, object]

    @property
    def state_transition(
        self,
    ) -> Any:
        """Validate the host payload at the existing Portfolio contract owner."""
        return panel_methodology_execution.resolve_portfolio_state_transition_binding(
            self.state_transition_payload
        )

    @property
    def source_hash(self) -> str:
        """Bind metadata composition to its Portfolio Program, never Alpha."""
        return panel_methodology_execution.panel_portfolio_preflight_source_hash(
            r0_surface_hash=self.r0_surface_hash,
            r1_surface_hash=self.r1_surface_hash,
            alpha_panel_source_identity=self.alpha_panel_source_identity,
            total_return_target_evidence_hash=self.total_return_target_evidence_hash,
            outcome_method_binding_hash=self.outcome_method_binding_hash,
            panel_derivation_recipe_hash=self.panel_derivation_recipe_hash,
            risk_input_binding_hash=self.risk_input_binding_hash,
            risk_return_surface_hash=self.risk_return_surface_hash,
            market_tradability_bundle_hash=self.market_tradability_bundle_hash,
            market_universe_epoch_hash=self.market_universe_epoch_hash,
            market_sector_revision=self.market_sector_revision,
            market_source_surface_hash=self.market_source_surface_hash,
            benchmark_surface_hash=self.benchmark_surface_hash,
            state_transition=self.state_transition,
        )


_INSTALLED_PORTFOLIO_PREFLIGHT_SOURCE = PanelPortfolioPreflightSource(
    r0_surface_hash="d18f018b1164cae2b1f91b9efc4750cf4fedab7bf2f96f056ca71d8a5a57b955",
    r1_surface_hash="edeba7ac29317e1e24e0e5950e65054a77be0b43474d005ca57c3b1b5c64c8da",
    alpha_panel_source_identity=fixed_panel_alpha_source_identity(),
    total_return_target_evidence_hash="d4a0996d89201f8d8b640362934490ee1ae6a5e12271226ee1f157c8a34887a8",
    outcome_method_binding_hash="441f9f6b8b737e68ffc4379375ddb0f8edbef649b48621223d7a20c6e0f53573",
    panel_derivation_recipe_hash="3b1835364218d8a503755723c698caaa4bf551a47b6c1868d26e6344e715bd21",
    risk_input_binding_hash="e67557d180eb7e7b7086d2b02ff0091673fd7ee3d799299645a88574287b0483",
    risk_return_surface_hash="3a6a38167f3ab346dede5cc7e75bf3b77eed59308efafb2dfaa34ba4e4c08178",
    market_tradability_bundle_hash="a8bbab93e82b5ecdbccdda1629311c8c5ba7362f1ee0a5be964940dff17b5ca0",
    market_universe_epoch_hash="61704785b45cbcd2ad6e07ff6645f4ca116cfc32a4df80bdf4b40da99cb9dd2e",
    market_sector_revision="be3c6383651785b2a817f889e54f22dca1040489c239480c73e4fc0a6e33701f",
    market_source_surface_hash="b5908ee6c159cd471effebd941126c01156c4fb4072f8d0bbe30dbbdf2b25cf2",
    benchmark_surface_hash="4e3db3919f2510a398c0aba38574246475afc8530d17a619d79d5780ac5bd637",
    state_transition_payload={
        "binding_hash": "cb6181a1aad175060e01c2ca4e2eb82f3a004f6162cf934e46cd325e70925242",
        "execution_method_binding_hash": (
            "441f9f6b8b737e68ffc4379375ddb0f8edbef649b48621223d7a20c6e0f53573"
        ),
        "initial_state_semantics": "FLAT_BOOK_ALL_CASH",
        "mark_availability_policy_hash": (
            "a4c3cfb97cb35a2df38ea71d5cf25673ae66b021da4c4e4d787cd937da395229"
        ),
        "mark_availability_policy_id": "source-authority.provider-daily-bars",
        "mark_epoch_hash": "a07fb0c8f4a093d1d51e50b6330d9ebc443d77664daf64713e5f31363fbfbe58",
        "mark_manifest_ref": (
            "manifests/0513c7c2364eeb4f4059492178beddbe129a6d312e3e11eaadec4373714327de.json"
        ),
        "mark_price_basis": "split_adjusted",
        "mark_source_watermark_hash": (
            "10a5ef7837cc4b3e3f4519b50756c58d366a39e2defb06d9a01c94f80390544c"
        ),
        "mark_surface_hash": "0513c7c2364eeb4f4059492178beddbe129a6d312e3e11eaadec4373714327de",
        "rebalance_clock": {
            "binding_hash": "55c9d5709cbbfc2494f57f7f1935844a80e7cc47f03b4a41261691b6ddfa851d",
            "clock_id": "EVERY_FORMATION",
            "parameters": [],
        },
        "reference_mark_method": "MARKED_TO_MARKET_AT_CLOSE_T",
        "transition_implementation_hash": (
            "1224e82acc8d460e10eaebefa0f8dc210bb31367900e717bc24821f724cec8ed"
        ),
    },
)


@dataclass(frozen=True, slots=True)
class PanelMethodologyRuntimeRoots:
    """Host-only locations needed to resolve installed immutable handles."""

    source: PanelMethodologySourceRoots
    r0_evidence_root: Path | None = None
    r1_evidence_root: Path | None = None
    portfolio_market_workspace: Path | None = None
    portfolio_tradability_artifact_root: Path | None = None
    risk_return_artifact_root: Path | None = None
    portfolio_preflight_source: PanelPortfolioPreflightSource = (
        _INSTALLED_PORTFOLIO_PREFLIGHT_SOURCE
    )


def _load_panel_score_readiness(*, output_workspace: Path, root: Any) -> Any:
    handle = root.alpha_formation_readiness_receipt_hash
    surface_hash = root.alpha_score_surface_hash
    if handle is None or surface_hash is None:
        raise AuthoringError("alpha_research.panel_score_formation_readiness_unverified")
    try:
        return AlphaDevelopmentArtifactStore(
            output_workspace / "portfolio-development"
        ).load_panel_score_formation_readiness(
            handle,
            program_hash=root.program_hash,
            score_surface_hash=surface_hash,
            formation_sessions=root.common_formation_sessions,
        )
    except AlphaDevelopmentArtifactReadbackError as error:
        raise AuthoringError("alpha_research.panel_score_formation_readiness_unverified") from error


def build_panel_methodology_authority_resolver(
    *,
    workspace: Path,
    roots: PanelMethodologyRuntimeRoots,
    data_snapshot_handle: str = "panel-feature-t-successor",
) -> WorkspaceResearchAuthorityResolver:
    """Install one symbolic immutable Panel handle without consulting current."""
    if data_snapshot_handle == "panel-feature-t-successor":
        snapshot_hash = resolve_corrected_panel_snapshot_hash(roots=roots.source)
        artifact_root = roots.source.panel_artifact_root
    elif data_snapshot_handle == "panel-remediation-baseline":
        snapshot_hash = resolve_failed_baseline_panel_snapshot_hash(roots=roots.source)
        artifact_root = roots.source.legacy_panel_artifact_root
    else:
        raise AuthoringError("alpha_research.panel_snapshot_handle_not_installed")
    return WorkspaceResearchAuthorityResolver(
        workspace=workspace,
        artifact_root=artifact_root,
        installed_snapshots=(
            InstalledResearchSnapshot(
                handle=data_snapshot_handle,
                panel_snapshot_hash=snapshot_hash,
            ),
        ),
    )


class WorkspaceReturnSurfaceProvider:
    """Every published causal return surface in one workspace, read-only.

    This reads; it never selects. Which surface can answer for a given authority
    is Risk methodology and is decided by the Desk, after the authority has been
    resolved. The previous loader did both jobs and did the second one by
    accident: it required the manifests directory to hold exactly one file, so a
    workspace that had published a second immutable surface -- which is simply
    what happens as dates advance -- became unusable rather than ambiguous.

    Nothing here takes a writer lease or creates a directory.
    """

    def __init__(self, artifact_root: Path) -> None:
        """Open the declared workspace risk-return manifest namespace.

        Args:
            artifact_root: Explicit caller-owned source artifact root.
        """
        self._manifests = Path(artifact_root) / "data-operations" / "risk-returns" / "manifests"

    def published_surfaces(self) -> tuple[CausalRiskReturnSurface, ...]:
        """Read all declared published return surfaces in deterministic manifest-path order.

        Returns:
            Typed surface declarations; sorting grants no preference among ambiguous sources.

        Raises:
            AuthoringError: Manifest directory or published surface set is unavailable.
        """
        if not self._manifests.is_dir():
            raise AuthoringError("research_authoring.return_surface_unavailable")
        surfaces = tuple(
            CausalRiskReturnSurface(**json.loads(path.read_text(encoding="utf-8")))
            # Sorted for a deterministic order, which matters only for the
            # ambiguity error being reproducible -- never for choosing.
            for path in sorted(self._manifests.glob("*.json"))
        )
        if not surfaces:
            raise AuthoringError("research_authoring.return_surface_unavailable")
        return surfaces


def _artifact_root(workspace: Path, artifact_root: Path | None) -> Path:
    return Path(artifact_root) if artifact_root else Path(workspace) / "artifacts"


def _resolved_panel_authority(
    *,
    workspace: Path,
    artifact_root: Path,
    envelope: ResearchExperimentEnvelope,
    feature_catalog: FeatureCatalog | None = None,
    feature_kernels: FeatureKernelRegistry | None = None,
    feature_input: ResolvedDevelopmentFeatureInput | None = None,
) -> tuple[ResolvedResearchAuthority, dict[str, Any]]:
    """The resolved authority and the Panel manifest it names.

    Reuses ``WorkspaceResearchAuthorityResolver`` rather than repeating its
    handle rules, so an executor can never be installed against a different Panel
    than the one the Program was sealed with.

    The installed Feature owners travel with the call because the resolver now
    verifies the Panel's clock authority before it resolves anything. A workspace
    built under an installed catalog revision -- an externally authored Formula,
    say -- published a Panel only that revision can answer for, and a verifier
    holding the shipped catalog would refuse it. Left unset they are the shipped
    ones, which is every path but that.
    """

    authority = WorkspaceResearchAuthorityResolver(
        workspace=workspace,
        artifact_root=artifact_root,
        feature_catalog=feature_catalog,
        feature_kernels=feature_kernels,
        feature_input=feature_input,
    ).resolve(envelope)
    if authority.panel_snapshot_hash is None or authority.panel_manifest_ref is None:
        raise AuthoringError("research_authoring.panel_authority_required")
    resolver = ArtifactResolver(artifact_root)
    manifest = (
        feature_input.panel_manifest
        if feature_input
        else resolver.load_feature_panel_manifest(authority.panel_manifest_ref)
    )
    return authority, dict(manifest)


def _panel_factor_ids(panel_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    summary = panel_manifest.get("safe_summary")
    factors = summary.get("factor_catalog_summary") if isinstance(summary, dict) else None
    if not isinstance(factors, dict) or not factors:
        raise AuthoringError("research_authoring.panel_factor_axis_unavailable")
    return tuple(sorted(str(value) for value in factors))


def _sector_by_listing_id(
    *,
    workspace: Path,
    market_profile_id: str,
    panel_manifest: Mapping[str, Any],
    listing_ids: tuple[str, ...],
) -> tuple[SectorHistory, str | None]:
    """Resolve the selected Panel's historical coverage from its source workspace
    (``panel_sector_labels``)."""
    store = MarketDataRepository(workspace)
    manifest = store.load_universe_manifest_revision(panel_source_manifest_revision(panel_manifest))
    if manifest.profile.market_profile_id != market_profile_id:
        raise AuthoringError("research_authoring.universe_handle_unresolved")
    return panel_sector_labels(store, manifest, panel_manifest, listing_ids)


def _build_risk_executor(
    *, workspace: Path, artifact_root: Path, market_profile_id: str
) -> DeskExperimentExecutor:
    store = MarketDataRepository(workspace)
    manifest = store.current_quality_filtered_research_manifest(market_profile_id=market_profile_id)
    if manifest is None:
        raise AuthoringError("research_authoring.universe_handle_unresolved")
    feature_state = FeatureStateRepository(store.database, market_data=store)
    # The Sector each formation reads, from the store's history.
    sector = feature_state.sector_history(manifest)
    if sector is None:
        raise AuthoringError("research_authoring.sector_evidence_unavailable")
    ArtifactResolver(artifact_root)  # fail early if the artifact root is not usable

    def _freshness_probe(*, through: date) -> str:
        """Recompute the surface-scope watermark exactly as the publisher did.

        The Desk cannot do this: it needs the market store and the universe
        manifest. It is supplied here so the Desk can prove the source behind a
        surface has not moved, without comparing two hashes that were never
        computed at the same scope or by the same projection.
        """

        return str(store.execution_source_watermark(manifest, through=through)["watermark_hash"])

    return RiskExperimentExecutor(
        surface_provider=WorkspaceReturnSurfaceProvider(artifact_root),
        return_reader=CausalRiskReturnReader(artifact_root),
        sector_by_listing_id=sector,
        freshness_probe=_freshness_probe,
    )


def _compose_panel_methodology_executor(
    *,
    request: PanelResearchMethodologyRequest,
    preflight: PanelResearchPreflight,
    inputs: panel_methodology_execution.PanelMethodologyExecutionInputs,
    runtime_caps: PanelResearchRuntimeCaps | None = None,
    sector_revision: str,
    execution_outcome_recipe_id: str,
    panel_factor_ids: tuple[str, ...],
    factor_evidence_checkpoint_hash: str,
) -> DeskExperimentExecutor:
    """Assemble the one compiler/executor pair for prepared methodology inputs."""

    portfolio_stage = request.execution_stage in {"PORTFOLIO_ONLY", "END_TO_END"}
    methodology_compiler = PanelResearchMethodologyCompiler(
        plan=inputs.plan,
        source_resolution_hash=inputs.source_resolution_hash,
        preflight=preflight,
        fixed_feature_view_binding_hashes=dict(inputs.fixed_feature_view_binding_hash_by_method),
        portfolio_policy_recipe_hash=(
            (
                rank_buffered_r0_inner_candidate_set_hash()
                if request.portfolio_policy_ids == ("RANK_BUFFERED_SCORE_RISK_COST",)
                else ScoreRiskCostDevelopmentRecipe.create(
                    risk_aversion=100.0,
                    turnover_regularization=0.001,
                    transaction_cost_rate=0.0005,
                    sector_capacity=None,
                    score_scale=0.01,
                    normalization_reference_id="CROSS_SECTIONAL_TARGET_Z_SCORE_UNIT",
                ).recipe_hash
            )
            if portfolio_stage
            else None
        ),
        portfolio_preflight_source_hash=(
            inputs.portfolio_preflight_source_hash
            if request.execution_stage == "PORTFOLIO_ONLY"
            else None
        ),
    )
    compiler = AlphaExperimentCompiler(
        target_recipes=installed_alpha_target_methods(
            sector_revision=sector_revision,
            execution_outcome_recipe_id=execution_outcome_recipe_id,
        ),
        model_mandate=build_development_alpha_model_mandate(),
        model_catalog=build_installed_alpha_model_catalog(),
        panel_factor_ids=panel_factor_ids,
        factor_evidence_factor_ids=panel_factor_ids,
        factor_evidence_checkpoint_hash=factor_evidence_checkpoint_hash,
        methodology_compiler=methodology_compiler,
    )
    return panel_methodology_execution.PanelResearchMethodologyExecutor(
        compiler=compiler,
        request=request,
        preflight=preflight,
        inputs=inputs,
        runtime_caps=runtime_caps,
    )


def _build_panel_portfolio_metadata_executor(
    *,
    output: Path,
    request: PanelResearchMethodologyRequest,
    roots: PanelMethodologyRuntimeRoots,
    source_roots: PanelMethodologySourceRoots,
    caps: PanelResearchRuntimeCaps,
    authority: Any,
    alpha_root: Any,
    panel_score_readiness: Any,
    source: PanelPortfolioPreflightSource,
    receipt: panel_methodology_execution.PanelPortfolioPreflightReceipt | None,
    receipt_load_seconds: float,
) -> DeskExperimentExecutor:
    """Compose compact exact-owner metadata for first-time and sealed preflight."""

    source = replace(source, r1_surface_hash=None) if request.risk_method_ids == ("R0",) else source
    if roots.r0_evidence_root is None or (
        source.r1_surface_hash is not None and roots.r1_evidence_root is None
    ):
        raise AuthoringError("alpha_research.portfolio_covariance_roots_not_installed")
    if roots.portfolio_tradability_artifact_root is None or roots.risk_return_artifact_root is None:
        raise AuthoringError("alpha_research.fixed_portfolio_market_roots_not_installed")
    if receipt is not None and (
        receipt.r0_surface_hash != source.r0_surface_hash
        or receipt.r1_surface_hash != source.r1_surface_hash
        or receipt.alpha_panel_source_identity != source.alpha_panel_source_identity
        or receipt.total_return_target_evidence_hash != source.total_return_target_evidence_hash
        or receipt.outcome_method_binding_hash != source.outcome_method_binding_hash
        or receipt.panel_derivation_recipe_hash != source.panel_derivation_recipe_hash
        or receipt.risk_input_binding_hash != source.risk_input_binding_hash
        or receipt.risk_return_surface_hash != source.risk_return_surface_hash
        or receipt.market_tradability_bundle_hash != source.market_tradability_bundle_hash
        or receipt.market_universe_epoch_hash != source.market_universe_epoch_hash
        or receipt.market_sector_revision != source.market_sector_revision
        or receipt.market_source_surface_hash != source.market_source_surface_hash
        or receipt.benchmark_surface_hash != source.benchmark_surface_hash
        or receipt.state_transition != source.state_transition
        or (
            receipt.preflight_source_hash is not None
            and receipt.preflight_source_hash != source.source_hash
        )
    ):
        raise AuthoringError("alpha_research.portfolio_preflight_source_mismatch")
    metrics: list[tuple[str, float, int, int]] = []

    started = perf_counter()
    r0 = resolve_installed_development_covariance(
        evidence_root=roots.r0_evidence_root,
        method_id="R0",
        surface_hash=source.r0_surface_hash,
    )
    r1 = (
        resolve_installed_development_covariance(
            evidence_root=cast(Path, roots.r1_evidence_root),
            method_id="R1",
            surface_hash=source.r1_surface_hash,
        )
        if source.r1_surface_hash is not None
        else None
    )
    metrics.append(
        (
            "risk_covariance_manifests",
            perf_counter() - started,
            1 + int(r1 is not None),
            len(r0.surface.model_dump_json())
            + (len(r1.surface.model_dump_json()) if r1 is not None else 0),
        )
    )

    started = perf_counter()
    row_receipt, row_axes = panel_methodology_execution.load_fixed_alpha_row_axes(
        output_workspace=output,
        root_hash=SEALED_PANEL_ALPHA_ROOT_HASH,
        receipt_hash=alpha_root.alpha_row_axis_receipt_hash,
        panel_snapshot_hash=authority.panel_snapshot_hash,
    )
    feature_binding = panel_methodology_execution.resolve_fixed_alpha_feature_binding(
        output_workspace=output,
        root_hash=SEALED_PANEL_ALPHA_ROOT_HASH,
    )
    row_axis_bytes = sum(
        fold.row_count * np.dtype(fold.listing_position_dtype).itemsize
        for fold in row_receipt.folds
    )
    metrics.append(
        (
            "alpha_row_and_feature_authority",
            perf_counter() - started,
            2,
            row_axis_bytes + len(row_receipt.model_dump_json()),
        )
    )
    folds = (
        tuple(
            panel_methodology_execution.PanelMethodologyOuterFold(
                fold_index=fold.fold_index,
                training_sessions=(),
                validation_sessions=fold.validation_sessions,
                source_manifest_hash=receipt.ordered_fold_hashes[fold.fold_index],
            )
            for fold in row_receipt.folds
        )
        if receipt is not None
        else panel_methodology_execution.resolve_fixed_candidate_decision_folds(
            output_workspace=output,
            root_hash=SEALED_PANEL_ALPHA_ROOT_HASH,
            model_recipe_id=SEALED_PANEL_ALPHA_MODEL_RECIPE_ID,
        )
    )
    if (
        alpha_root.root_hash != SEALED_PANEL_ALPHA_ROOT_HASH
        or tuple(value.source_manifest_hash for value in folds) != alpha_root.ordered_fold_hashes
        or len(row_receipt.folds) != len(folds)
        or tuple(session for fold in folds for session in fold.validation_sessions)
        != alpha_root.common_formation_sessions
        or (receipt is not None and receipt.fixed_alpha_root_hash != alpha_root.root_hash)
        or (
            receipt is not None
            and receipt.fixed_alpha_row_axis_receipt_hash != row_receipt.receipt_hash
        )
        or (receipt is not None and len(receipt.ordered_fold_hashes) != len(folds))
        or (
            receipt is not None
            and tuple(sorted(feature_binding.feature_count_by_method.items()))
            != receipt.feature_count_by_method
        )
        or (
            receipt is not None
            and tuple(sorted(feature_binding.feature_axis_hash_by_method.items()))
            != receipt.feature_axis_hash_by_method
        )
        or (
            receipt is not None
            and tuple(sorted(feature_binding.view_binding_hash_by_method.items()))
            != receipt.feature_view_binding_hash_by_method
        )
        or feature_binding.source_resolution_hash
        != source.alpha_panel_source_identity.source_resolution_hash
        or alpha_root.source_resolution_hash != feature_binding.source_resolution_hash
        or alpha_root.feature_preflight_hash != feature_binding.feature_preflight_hash
        or row_receipt.source_resolution_hash != feature_binding.source_resolution_hash
        or row_receipt.panel_snapshot_hash != authority.panel_snapshot_hash
    ):
        raise AuthoringError("alpha_research.portfolio_preflight_alpha_identity_mismatch")

    started = perf_counter()
    facts: ResolvedFixedPortfolioFactAuthority = resolve_fixed_portfolio_fact_authority(
        roots=source_roots,
        authority=authority,
        source_resolution_hash=source.alpha_panel_source_identity.source_resolution_hash,
        execution_sessions=r0.formation_sessions,
        alpha_panel_source_identity=source.alpha_panel_source_identity,
        total_return_target_evidence_hash=source.total_return_target_evidence_hash,
        outcome_method_binding_hash=source.outcome_method_binding_hash,
        panel_derivation_recipe_hash=source.panel_derivation_recipe_hash,
    )
    metrics.append(
        (
            "fact_and_execution_metadata",
            perf_counter() - started,
            1,
            facts.metadata_bytes_read,
        )
    )

    started = perf_counter()
    market: ResolvedFixedPortfolioMarketAuthority = resolve_fixed_portfolio_market_authority(
        source=facts,
        r0_covariance=r0,
        r1_covariance=r1,
        risk_return_artifact_root=roots.risk_return_artifact_root,
        portfolio_artifact_root=roots.portfolio_tradability_artifact_root,
        benchmark_artifact_root=output / "portfolio-development",
        decision_sessions=tuple(session for fold in folds for session in fold.validation_sessions),
        session_mark_artifact_root=output / "portfolio-development",
        playpen_root=resolve_playpen_root(Path(__file__)),
        risk_input_binding_hash=source.risk_input_binding_hash,
        tradability_bundle_hash=source.market_tradability_bundle_hash,
        risk_return_surface_hash=source.risk_return_surface_hash,
        benchmark_surface_hash=source.benchmark_surface_hash,
        state_transition=source.state_transition,
    )
    metrics.append(
        (
            "risk_market_manifests",
            perf_counter() - started,
            1,
            market.metadata_bytes_read,
        )
    )
    if (
        facts.total_return_target_evidence_hash != source.total_return_target_evidence_hash
        or facts.outcome_method_binding_hash != source.outcome_method_binding_hash
        or facts.panel_derivation_recipe_hash != source.panel_derivation_recipe_hash
        or facts.sector_revision != source.market_sector_revision
        or market.risk_input_binding_hash != source.risk_input_binding_hash
        or market.risk_return_surface_hash != source.risk_return_surface_hash
        or market.tradability_bundle_hash != source.market_tradability_bundle_hash
        or market.universe_epoch_hash != source.market_universe_epoch_hash
        or market.benchmark_surface_hash != source.benchmark_surface_hash
        or market.state_transition != source.state_transition
    ):
        raise AuthoringError("alpha_research.portfolio_preflight_owner_identity_mismatch")

    axis = panel_methodology_execution.preflight_panel_portfolio_axis(
        fixed_row_axes=row_axes,
        ordered_score_listing_ids=row_receipt.ordered_listing_ids,
        outer_folds=folds,
        portfolio_market=market,
        r0_covariance=r0,
        r1_covariance=r1,
        maximum_formation_count=portfolio_study_formation_limit(request.portfolio_policy_ids),
    )
    if receipt is not None and (
        axis.formation_sessions != receipt.portfolio_formation_sessions
        or axis.economic_formation_sessions != receipt.portfolio_economic_formation_sessions
        or axis.decision_ranges != receipt.portfolio_decision_ranges
        or axis.passive_sessions != receipt.portfolio_passive_sessions
        or axis.ordered_listing_ids != receipt.portfolio_ordered_listing_ids
        or axis.session_axis_hash != receipt.portfolio_session_axis_hash
        or axis.economic_session_axis_hash != receipt.portfolio_economic_session_axis_hash
        or axis.listing_axis_hash != receipt.portfolio_listing_axis_hash
        or axis.axis_hash != receipt.portfolio_axis_hash
    ):
        raise AuthoringError("alpha_research.portfolio_preflight_axis_mismatch")

    started = perf_counter()
    simple_binding, _simple_artifact, simple_bytes = AlphaDevelopmentArtifactStore(
        output / "portfolio-development"
    ).load_simple_signed_score_metadata(
        artifact_hash=alpha_root.alpha_simple_score_value_artifact_hash,
        binding_hash=alpha_root.alpha_simple_score_binding_hash,
    )
    metrics.append(("alpha_simple_score_content_hash", perf_counter() - started, 1, simple_bytes))
    portfolio_listing_set = frozenset(axis.ordered_listing_ids)
    projected_simple_listing_ids = tuple(
        listing_id
        for listing_id in simple_binding.ordered_listing_ids
        if listing_id in portfolio_listing_set
    )
    if (
        tuple(
            value
            for value in simple_binding.ordered_formation_sessions
            if value in set(axis.formation_sessions)
        )
        != axis.formation_sessions
        or simple_binding.ordered_listing_ids != row_receipt.ordered_listing_ids
        or projected_simple_listing_ids != axis.ordered_listing_ids
        or alpha_root.alpha_simple_score_binding_hash != simple_binding.binding_hash
        or (
            receipt is not None
            and receipt.simple_score_binding_hash != alpha_root.alpha_simple_score_binding_hash
        )
        or (
            receipt is not None
            and receipt.simple_score_value_artifact_hash
            != alpha_root.alpha_simple_score_value_artifact_hash
        )
        or simple_binding.minimum_finite_listings_observed < cast(int, request.top_k)
    ):
        raise AuthoringError("alpha_research.portfolio_preflight_simple_score_mismatch")

    started = perf_counter()
    feature_axis_hash = feature_binding.feature_axis_hash_by_method.get(SEALED_PANEL_ALPHA_VIEW_ID)
    feature_count = feature_binding.feature_count_by_method.get(SEALED_PANEL_ALPHA_VIEW_ID)
    if feature_axis_hash is None or feature_count is None:
        raise AuthoringError("alpha_research.fixed_panel_score_feature_axis_absent")
    handoff = panel_methodology_execution.resolve_fixed_panel_score_temporal_handoff(
        output_workspace=output,
        root=alpha_root,
        row_axis_receipt=row_receipt,
        source=facts.alpha_panel_source_identity,
        feature_axis_hash=feature_axis_hash,
        feature_count=feature_count,
        readiness=panel_score_readiness,
    )
    panel_methodology_execution.admit_fixed_panel_portfolio_authorities(
        handoff=handoff,
        execution=facts.execution_clock,
        r0_covariance=r0,
        r1_covariance=r1,
        mark_surface=market.mark_surface,
        marked_sessions=axis.formation_sessions,
        tradability_bundle_hash=market.tradability_bundle_hash,
        universe_epoch_hash=market.universe_epoch_hash,
        axis=axis,
    )
    metrics.append(("temporal_admission", perf_counter() - started, 1, 0))

    preflight = build_panel_research_preflight(
        request=request,
        plan=None,
        fixed_feature_preflight_hash=feature_binding.feature_preflight_hash,
        fixed_feature_count_by_method=feature_binding.feature_count_by_method,
        source_resolution_hash=feature_binding.source_resolution_hash,
        fold_count=len(folds),
        formation_count=len(axis.formation_sessions),
        economic_formation_count=len(axis.economic_formation_sessions),
        portfolio_listing_count=len(axis.ordered_listing_ids),
        portfolio_session_axis_hash=axis.session_axis_hash,
        portfolio_economic_session_axis_hash=axis.economic_session_axis_hash,
        portfolio_listing_axis_hash=axis.listing_axis_hash,
        portfolio_axis_hash=axis.axis_hash,
        maximum_solver_calls_for_formation_count=lambda count: portfolio_solver_call_upper_bound(
            policy_ids=request.portfolio_policy_ids,
            formation_count=count,
        ),
        caps=caps,
    )
    if receipt is not None and preflight != receipt.preflight:
        raise AuthoringError("alpha_research.portfolio_preflight_budget_identity_mismatch")
    inputs = panel_methodology_execution.PanelMethodologyExecutionInputs(
        plan=None,
        source_resolution_hash=feature_binding.source_resolution_hash,
        feature_preflight_hash=feature_binding.feature_preflight_hash,
        outer_folds=folds,
        fixed_feature_column_count=feature_count,
        fixed_feature_axis_hash=feature_axis_hash,
        fixed_feature_count_by_method=tuple(
            sorted(feature_binding.feature_count_by_method.items())
        ),
        fixed_feature_axis_hash_by_method=tuple(
            sorted(feature_binding.feature_axis_hash_by_method.items())
        ),
        fixed_feature_view_binding_hash_by_method=tuple(
            sorted(feature_binding.view_binding_hash_by_method.items())
        ),
        fixed_alpha_row_axis_receipt=row_receipt,
        fixed_alpha_row_axes=row_axes,
        portfolio_axis_authority=market,
        r0_covariance=r0,
        r1_covariance=r1,
        portfolio_axis=axis,
        fixed_simple_score_binding_hash=simple_binding.binding_hash,
        fixed_simple_score_value_artifact_hash=alpha_root.alpha_simple_score_value_artifact_hash,
        total_return_target_evidence_hash=facts.total_return_target_evidence_hash,
        outcome_method_binding_hash=facts.outcome_method_binding_hash,
        panel_derivation_recipe_hash=facts.panel_derivation_recipe_hash,
        risk_input_binding_hash=market.risk_input_binding_hash,
        risk_return_surface_hash=market.risk_return_surface_hash,
        benchmark_surface_hash=market.benchmark_surface_hash,
        market_tradability_bundle_hash=market.tradability_bundle_hash,
        market_universe_epoch_hash=market.universe_epoch_hash,
        market_sector_revision=facts.sector_revision,
        market_source_surface_hash=source.market_source_surface_hash,
        portfolio_state_transition=market.state_transition,
        portfolio_preflight_source_hash=source.source_hash,
        fixed_simple_score_binding=simple_binding,
        alpha_panel_source_identity=facts.alpha_panel_source_identity,
        panel_score_readiness=panel_score_readiness,
        execution_clock=facts.execution_clock,
        owner_resolution_seconds=sum(value[1] for value in metrics),
        sealed_receipt_load_seconds=receipt_load_seconds,
        sealed_preflight_program_hash=(receipt.program_hash if receipt is not None else None),
        sealed_preflight_receipt=receipt,
        owner_resolution_metrics=tuple(metrics),
    )
    recipe_id = facts.execution_clock.events.execution_recipe_id
    if recipe_id is None:
        raise AuthoringError("alpha_research.portfolio_preflight_execution_recipe_missing")
    return _compose_panel_methodology_executor(
        request=request,
        preflight=preflight,
        inputs=inputs,
        sector_revision=facts.sector_revision,
        execution_outcome_recipe_id=recipe_id,
        panel_factor_ids=(),
        factor_evidence_checkpoint_hash=preflight.source_resolution_hash,
    )


def _build_panel_methodology_executor(
    *,
    workspace: Path,
    workspace_root: Path,
    envelope: ResearchExperimentEnvelope,
    document: Mapping[str, Any],
    roots: PanelMethodologyRuntimeRoots,
    caps: PanelResearchRuntimeCaps,
    sealed_preflight_program_hash: str | None = None,
    preflight_metadata_only: bool = False,
) -> DeskExperimentExecutor:
    section = methodology_section(document)
    if section is None:
        raise AuthoringError("alpha_research.methodology_not_selected")
    request = PanelResearchMethodologyRequest.from_mapping(section)
    output = (Path(workspace_root).resolve() / envelope.output_workspace).resolve()
    baseline = roots.source.failed_baseline_workspace.resolve()
    if output == baseline or output.is_relative_to(baseline) or baseline.is_relative_to(output):
        raise AuthoringError("alpha_research.remediation_output_overlaps_failed_baseline")
    alpha_root = None
    panel_score_readiness = None
    if request.execution_stage == "PORTFOLIO_ONLY":
        alpha_root = panel_methodology_execution.load_panel_methodology_root(
            output_workspace=output,
            root_hash=SEALED_PANEL_ALPHA_ROOT_HASH,
        )
        panel_score_readiness = _load_panel_score_readiness(
            output_workspace=output, root=alpha_root
        )
        if (
            alpha_root.alpha_simple_score_binding_hash is None
            or alpha_root.alpha_simple_score_value_artifact_hash is None
        ):
            raise AuthoringError("alpha_research.simple_signal_value_artifact_handle_required")
        if alpha_root.alpha_row_axis_receipt_hash is None:
            raise AuthoringError("alpha_research.fixed_alpha_row_axis_receipt_handle_required")
    if preflight_metadata_only and request.execution_stage != "PORTFOLIO_ONLY":
        raise AuthoringError("alpha_research.portfolio_metadata_preflight_stage_mismatch")
    if (
        request.execution_stage == "PORTFOLIO_ONLY"
        and sealed_preflight_program_hash is None
        and not preflight_metadata_only
    ):
        raise AuthoringError("alpha_research.sealed_portfolio_preflight_required")
    owner_started = perf_counter()
    resolver = build_panel_methodology_authority_resolver(
        workspace=workspace,
        roots=roots,
        data_snapshot_handle=envelope.data_snapshot_handle,
    )
    authority = resolver.resolve(envelope)
    source_roots = replace(
        roots.source,
        development_overlay_artifact_root=output / "feature-development",
    )
    fixed_facts: ResolvedFixedPortfolioFacts | None
    fixed_portfolio_market: ResolvedFixedPortfolioMarket | None
    if sealed_preflight_program_hash is not None:
        if request.execution_stage != "PORTFOLIO_ONLY":
            raise AuthoringError("alpha_research.sealed_portfolio_preflight_stage_mismatch")
        if roots.r0_evidence_root is None:
            raise AuthoringError("alpha_research.portfolio_covariance_roots_not_installed")
        load_started = perf_counter()
        receipt = panel_methodology_execution.load_panel_portfolio_preflight_receipt(
            output_workspace=output,
            program_hash=sealed_preflight_program_hash,
        )
        if receipt.r1_surface_hash is not None and roots.r1_evidence_root is None:
            raise AuthoringError("alpha_research.portfolio_covariance_roots_not_installed")
        if (
            receipt.preflight.request_hash != request.request_hash
            or receipt.authority_hash != authority.authority_hash
            or receipt.program_hash != sealed_preflight_program_hash
        ):
            raise AuthoringError("alpha_research.portfolio_preflight_identity_mismatch")
        if (
            receipt.simple_score_binding_hash is None
            or receipt.simple_score_value_artifact_hash is None
        ):
            raise AuthoringError("alpha_research.simple_signal_value_artifact_handle_required")
        if any(
            value is None
            for value in (
                receipt.total_return_target_evidence_hash,
                receipt.outcome_method_binding_hash,
                receipt.panel_derivation_recipe_hash,
                receipt.risk_input_binding_hash,
                receipt.risk_return_surface_hash,
            )
        ):
            raise AuthoringError("alpha_research.portfolio_preflight_owner_handle_required")
        if (
            roots.portfolio_market_workspace is None
            or roots.portfolio_tradability_artifact_root is None
            or roots.risk_return_artifact_root is None
        ):
            raise AuthoringError("alpha_research.fixed_portfolio_market_roots_not_installed")
        if preflight_metadata_only:
            return _build_panel_portfolio_metadata_executor(
                output=output,
                request=request,
                roots=roots,
                source_roots=source_roots,
                caps=caps,
                authority=authority,
                alpha_root=alpha_root,
                panel_score_readiness=panel_score_readiness,
                source=roots.portfolio_preflight_source,
                receipt=receipt,
                receipt_load_seconds=perf_counter() - load_started,
            )
        r0_exact = resolve_installed_development_covariance(
            evidence_root=roots.r0_evidence_root,
            method_id="R0",
            surface_hash=receipt.r0_surface_hash,
        )
        r1_exact = (
            resolve_installed_development_covariance(
                evidence_root=cast(Path, roots.r1_evidence_root),
                method_id="R1",
                surface_hash=receipt.r1_surface_hash,
            )
            if receipt.r1_surface_hash is not None
            else None
        )
        fixed_facts = resolve_fixed_portfolio_facts(
            roots=source_roots,
            authority=authority,
            source_resolution_hash=receipt.alpha_panel_source_identity.source_resolution_hash,
            development_overlay_method_id=None,
            alpha_panel_source_identity=receipt.alpha_panel_source_identity,
            total_return_target_evidence_hash=cast(str, receipt.total_return_target_evidence_hash),
            outcome_method_binding_hash=cast(str, receipt.outcome_method_binding_hash),
            panel_derivation_recipe_hash=cast(str, receipt.panel_derivation_recipe_hash),
            execution_sessions=r0_exact.formation_sessions,
        )
        fixed_portfolio_market = resolve_fixed_portfolio_market(
            source=fixed_facts,
            r0_covariance=r0_exact,
            r1_covariance=r1_exact,
            market_workspace=roots.portfolio_market_workspace,
            risk_return_artifact_root=roots.risk_return_artifact_root,
            portfolio_artifact_root=roots.portfolio_tradability_artifact_root,
            benchmark_artifact_root=output / "portfolio-development",
            decision_sessions=receipt.portfolio_formation_sessions,
            session_mark_artifact_root=output / "portfolio-development",
            playpen_root=resolve_playpen_root(Path(__file__)),
            risk_input_binding_hash=cast(str, receipt.risk_input_binding_hash),
            tradability_bundle_hash=receipt.market_tradability_bundle_hash,
            risk_return_surface_hash=cast(str, receipt.risk_return_surface_hash),
            benchmark_surface_hash=receipt.benchmark_surface_hash,
            state_transition=receipt.state_transition,
        )
        inputs = panel_methodology_execution.inputs_from_panel_portfolio_preflight_receipt(
            output_workspace=output,
            receipt=receipt,
            r0_covariance=r0_exact,
            r1_covariance=r1_exact,
            simple_score_source=WorkspaceSimpleSignalSource(
                score_evidence_root=output / "portfolio-development",
                panel_artifacts_root=roots.source.panel_artifact_root,
            ),
            portfolio_market=fixed_portfolio_market.market,
            portfolio_benchmark=fixed_portfolio_market.benchmark,
            reference_marks=fixed_portfolio_market.reference_marks,
            execution_clock=fixed_facts.execution_clock,
            alpha_panel_source_identity=fixed_facts.alpha_panel_source_identity,
            panel_score_readiness=cast(Any, panel_score_readiness),
            receipt_load_seconds=perf_counter() - load_started,
        )
        if (
            receipt.preflight.actual_feature_count != SEALED_PANEL_ALPHA_FEATURE_COLUMN_COUNT
            or receipt.preflight.actual_portfolio_formation_count
            != (
                portfolio_study_formation_limit(request.portfolio_policy_ids)
                or SEALED_PANEL_ALPHA_DECISION_SESSION_COUNT
            )
        ):
            raise AuthoringError("alpha_research.portfolio_preflight_fixed_axis_mismatch")
        recipe_id = fixed_facts.execution_clock.events.execution_recipe_id
        if recipe_id is None:
            raise AuthoringError("alpha_research.portfolio_preflight_execution_recipe_missing")
        return _compose_panel_methodology_executor(
            request=request,
            preflight=receipt.preflight,
            inputs=inputs,
            runtime_caps=caps,
            sector_revision=receipt.market_sector_revision,
            execution_outcome_recipe_id=recipe_id,
            panel_factor_ids=(),
            factor_evidence_checkpoint_hash=receipt.preflight.source_resolution_hash,
        )
    if preflight_metadata_only:
        return _build_panel_portfolio_metadata_executor(
            output=output,
            request=request,
            roots=roots,
            source_roots=source_roots,
            caps=caps,
            authority=authority,
            alpha_root=alpha_root,
            panel_score_readiness=panel_score_readiness,
            source=roots.portfolio_preflight_source,
            receipt=None,
            receipt_load_seconds=0.0,
        )
    portfolio_stage = request.execution_stage == "END_TO_END"
    if portfolio_stage and (roots.r0_evidence_root is None or roots.r1_evidence_root is None):
        raise AuthoringError("alpha_research.portfolio_covariance_roots_not_installed")
    r0 = (
        resolve_installed_development_covariance(
            evidence_root=cast(Path, roots.r0_evidence_root), method_id="R0"
        )
        if portfolio_stage
        else None
    )
    r1 = (
        resolve_installed_development_covariance(
            evidence_root=cast(Path, roots.r1_evidence_root), method_id="R1"
        )
        if portfolio_stage
        else None
    )
    owner_resolution_seconds = perf_counter() - owner_started
    feature_started = perf_counter()
    source = resolve_installed_panel_methodology_source(
        roots=source_roots,
        authority=authority,
        input_method_id=request.input_method_id,
        development_overlay_method_id=request.development_overlay_method_id,
    )
    if portfolio_stage and (
        request.fixed_score_method_id is None
        or feature_id_for_simple_score_method(request.fixed_score_method_id)
        not in source.arrays.ordered_factor_ids
    ):
        raise AuthoringError("alpha_research.simple_signal_feature_not_in_panel")
    plan = preflight_panel_feature_plan(
        source=source.arrays,
        selected_method_ids=request.feature_view_ids,
        # Score filters preserve the Alpha row axis and own their own
        # min-periods-one window depth. They may not shrink Feature eligibility.
        maximum_aggregation_span=1,
    )
    feature_materialization_count = 1
    feature_materialization_seconds = perf_counter() - feature_started
    folds = tuple(
        panel_methodology_execution.PanelMethodologyOuterFold(
            fold_index=value.fold_index,
            training_sessions=value.training_sessions,
            validation_sessions=value.validation_sessions,
            source_manifest_hash=value.source_manifest_hash,
        )
        for value in source.outer_folds
    )
    portfolio_market = source.portfolio_market
    portfolio_benchmark = source.portfolio_benchmark
    portfolio_fold_indices = source.portfolio_fold_indices
    source_resolution_hash = source.resolution.resolution_hash
    feature_preflight_hash = plan.preflight.preflight_hash
    sector_revision = source.resolution.sector_revision
    execution_outcome_recipe_id = source.resolution.execution_outcome_recipe_id
    panel_factor_ids = source.arrays.ordered_factor_ids
    factor_evidence_checkpoint_hash = (
        source.resolution.curated_factor_checkpoint_hash or source.resolution.resolution_hash
    )
    axis = None
    if portfolio_stage:
        if r0 is None or r1 is None:
            raise AuthoringError("alpha_research.portfolio_covariance_not_resolved")
        axis = panel_methodology_execution.preflight_panel_portfolio_axis(
            plan=plan,
            outer_folds=folds,
            portfolio_market=portfolio_market,
            r0_covariance=r0,
            r1_covariance=r1,
            maximum_formation_count=portfolio_study_formation_limit(request.portfolio_policy_ids),
        )
    preflight = build_panel_research_preflight(
        request=request,
        plan=plan,
        source_resolution_hash=source_resolution_hash,
        fold_count=len(folds),
        formation_count=len(axis.formation_sessions) if axis is not None else 0,
        economic_formation_count=(len(axis.economic_formation_sessions) if axis is not None else 0),
        portfolio_listing_count=len(axis.ordered_listing_ids) if axis is not None else 0,
        portfolio_session_axis_hash=axis.session_axis_hash if axis is not None else None,
        portfolio_economic_session_axis_hash=(
            axis.economic_session_axis_hash if axis is not None else None
        ),
        portfolio_listing_axis_hash=axis.listing_axis_hash if axis is not None else None,
        portfolio_axis_hash=axis.axis_hash if axis is not None else None,
        maximum_solver_calls_for_formation_count=lambda count: portfolio_solver_call_upper_bound(
            policy_ids=request.portfolio_policy_ids,
            formation_count=count,
        ),
        caps=caps,
    )
    simple_score_authority: WorkspaceSimpleSignalMaterializationAuthority | None = None
    if request.execution_stage in {"ALPHA_ONLY", "END_TO_END"}:
        if authority.panel_snapshot_hash is None or authority.panel_manifest_ref is None:
            raise AuthoringError("research_authoring.panel_authority_required")
        simple_score_authority = WorkspaceSimpleSignalMaterializationAuthority(
            panel_artifacts_root=roots.source.panel_artifact_root,
            panel_manifest_ref=authority.panel_manifest_ref,
            panel_snapshot_identity=authority.panel_snapshot_hash,
        )
    inputs = panel_methodology_execution.PanelMethodologyExecutionInputs(
        plan=plan,
        source_resolution_hash=source_resolution_hash,
        feature_preflight_hash=feature_preflight_hash,
        outer_folds=folds,
        portfolio_market=portfolio_market if portfolio_stage else None,
        portfolio_axis_authority=portfolio_market if portfolio_stage else None,
        portfolio_benchmark=portfolio_benchmark if portfolio_stage else None,
        portfolio_fold_indices=portfolio_fold_indices if portfolio_stage else None,
        r0_covariance=r0,
        r1_covariance=r1,
        portfolio_axis=axis,
        simple_score_authority=simple_score_authority,
        alpha_panel_source_identity=(
            resolve_alpha_panel_source_identity(
                source_resolution_hash=source.resolution.resolution_hash,
                panel_snapshot_hash=source.resolution.panel_snapshot_hash,
                ordered_session_axis_hash=source.resolution.ordered_session_axis_hash,
                ordered_listing_axis_hash=source.resolution.ordered_listing_axis_hash,
                ordered_factor_axis_hash=source.resolution.ordered_factor_axis_hash,
                formula_observation_policy_hash=source.resolution.formula_observation_policy_hash,
                source_availability_policy_hash=source.resolution.source_availability_policy_hash,
                source_authority_binding_hash=source.resolution.source_authority_binding_hash,
                catalog=panel_feature_catalog(workspace, source.resolution.panel_snapshot_hash),
            )
            if request.execution_stage in {"ALPHA_ONLY", "END_TO_END"}
            else None
        ),
        execution_clock=source.execution_clock if portfolio_stage else None,
        feature_materialization_count=feature_materialization_count,
        feature_materialization_seconds=feature_materialization_seconds,
        owner_resolution_seconds=owner_resolution_seconds,
    )
    return _compose_panel_methodology_executor(
        request=request,
        preflight=preflight,
        inputs=inputs,
        sector_revision=sector_revision,
        execution_outcome_recipe_id=execution_outcome_recipe_id,
        panel_factor_ids=panel_factor_ids,
        factor_evidence_checkpoint_hash=factor_evidence_checkpoint_hash,
    )


def _build_alpha_executor(
    *,
    workspace: Path,
    artifact_root: Path,
    envelope: ResearchExperimentEnvelope,
    document: Mapping[str, Any],
    factor_evidence_root: Path | None,
    workspace_root: Path,
    split_policy: AlphaSplitPolicy | None,
    panel_methodology_roots: PanelMethodologyRuntimeRoots | None = None,
    panel_runtime_caps: PanelResearchRuntimeCaps | None = None,
    outcome_artifact_root: Path | None = None,
    feature_authority_outcome_snapshot_handle: str | None = None,
    feature_catalog: FeatureCatalog | None = None,
    feature_kernels: FeatureKernelRegistry | None = None,
    sealed_preflight_program_hash: str | None = None,
    preflight_metadata_only: bool = False,
    feature_input: ResolvedDevelopmentFeatureInput | None = None,
) -> DeskExperimentExecutor:
    selected_lifecycle = lifecycle_section(document)
    if selected_lifecycle is not None:
        from alphalattice.control.product_host.composition.research_workspace import (
            component_training_selection,
            resolve_workspace_model_lifecycle,
        )

        component, training_authority = component_training_selection(envelope.data_snapshot_handle)
        if component != str(selected_lifecycle.get("component_recipe_id")):
            raise AuthoringError("alpha_research.lifecycle_component_not_installed")
        store, admission = resolve_workspace_model_lifecycle(
            workspace,
            component_id=component,
            artifact_root=artifact_root,
            training_authority_hash=training_authority,
        )
        method = AlphaLifecycleExperiment(store, admission)
        method.validate_declaration(document)
        return method
    if methodology_section(document) is not None:
        if panel_methodology_roots is None or panel_runtime_caps is None:
            raise AuthoringError("alpha_research.panel_methodology_runtime_not_installed")
        return _build_panel_methodology_executor(
            workspace=workspace,
            workspace_root=workspace_root,
            envelope=envelope,
            document=document,
            roots=panel_methodology_roots,
            caps=panel_runtime_caps,
            sealed_preflight_program_hash=sealed_preflight_program_hash,
            preflight_metadata_only=preflight_metadata_only,
        )
    if split_policy is None:
        raise AuthoringError("research_authoring.alpha_composition_incomplete")
    section = document.get("alpha")
    if not isinstance(section, dict):
        raise AuthoringError("alpha_research.authoring_section_missing")
    handle = section.get("factor_evidence_handle")
    if not isinstance(handle, str) or not handle:
        # The document names the *handle*; the Host decides where handles
        # resolve. A document that could also name the root could point the Host
        # at any directory on the machine and have it read as authority.
        raise AuthoringError("alpha_research.authoring_factor_evidence_handle_invalid")
    if factor_evidence_root is None:
        raise AuthoringError("research_authoring.factor_evidence_root_not_installed")
    outcome_handle = section.get("causal_outcome_snapshot_handle")
    if outcome_handle is not None and (not isinstance(outcome_handle, str) or not outcome_handle):
        raise AuthoringError("research_authoring.execution_outcome_handle_invalid")
    outcome_root = Path(outcome_artifact_root) if outcome_artifact_root else Path(artifact_root)

    authority, panel_manifest = _resolved_panel_authority(
        workspace=workspace,
        artifact_root=artifact_root,
        envelope=envelope,
        feature_catalog=feature_catalog,
        feature_kernels=feature_kernels,
        feature_input=feature_input,
    )
    assert authority.panel_snapshot_hash is not None and authority.panel_manifest_ref is not None
    snapshot_hash, manifest_ref = authority.panel_snapshot_hash, authority.panel_manifest_ref
    feature_reader = (
        feature_input.reader
        if feature_input
        else FeaturePanelReader(ArtifactResolver(artifact_root))
    )
    # The verified historical coverage axis the authority just read off the
    # Panel (`FeaturePanelReader.listing_ids`), not a second read of it.
    listing_ids = authority.ordered_listing_ids
    sector_by_listing_id, sector_coverage_hash = _sector_by_listing_id(
        workspace=workspace,
        market_profile_id=universe_profile(envelope.universe_handle),
        panel_manifest=panel_manifest,
        # An exploration sample fits on its names; its targets are standardized across
        # the whole cross-section, so every name of the Panel keeps its Sector.
        listing_ids=listing_ids
        if exploration_sample_size(envelope.universe_handle) is None
        else feature_reader.listing_ids(manifest_ref),
    )
    declared_outcome = section.get("execution_outcome_recipe_id")
    declared_outcome_id = (
        str(declared_outcome) if isinstance(declared_outcome, str) and declared_outcome else None
    )
    foundation, outcome_manifest_ref = build_alpha_development_foundation(
        panel_manifest=panel_manifest,
        panel_snapshot_hash=snapshot_hash,
        artifact_root=artifact_root,
        factor_evidence_root=factor_evidence_root,
        factor_evidence_handle=handle,
        source_workspace=workspace,
        output_workspace=Path(workspace_root) / envelope.output_workspace,
        execution_outcome_recipe_id=declared_outcome_id,
        outcome_artifact_root=outcome_root,
        causal_outcome_snapshot_handle=outcome_handle,
        feature_authority_outcome_snapshot_handle=feature_authority_outcome_snapshot_handle,
        sector_coverage_hash=sector_coverage_hash,
        feature_input=feature_input,
    )
    if section.get("foundation_admission_hash") is not None:
        from alphalattice.foundation.factor_research.experiments.development_evidence import (
            FactorDevelopmentCurationReader,
            FactorDevelopmentReceiptReader,
        )
        from alphalattice.foundation.research_foundation.publication.sponsorship import (
            admit_selected_factor_foundation,
            read_foundation_admission,
        )
        from alphalattice.investment.alpha_research.inputs.development_foundation import (
            AlphaDevelopmentFoundationBinding,
        )

        admission = read_foundation_admission(
            Path(workspace_root) / "artifacts", str(section["foundation_admission_hash"])
        )
        if admission.factor_receipt_hash != foundation.factor_development_receipt_hash:
            raise AuthoringError("research_foundation.factor_receipt_mismatch")
        receipt, checkpoint = FactorDevelopmentReceiptReader(factor_evidence_root).load(handle)
        decision = next(
            (
                v
                for v in FactorDevelopmentCurationReader(factor_evidence_root).submissions(
                    checkpoint.checkpoint_hash
                )
                if v.receipt_hash == admission.curation_receipt_hash
            ),
            None,
        )
        if (
            decision is None
            or admit_selected_factor_foundation(
                receipt=receipt,
                checkpoint=checkpoint,
                decision=decision,
                panel_manifest=panel_manifest,
                artifact_root=artifact_root,
                logical_panel_hash=foundation.logical_panel_hash,
                logical_semantic_index_hash=foundation.logical_semantic_index_hash,
                factor_task_id=admission.factor_task_id,
                input_id=admission.input_id,
                input_binding_hash=admission.input_binding_hash,
                factor_input_binding_hash=admission.factor_input_binding_hash,
            )
            != admission
        ):
            raise AuthoringError("research_foundation.source_changed")
        foundation = AlphaDevelopmentFoundationBinding.create(
            **{
                name: getattr(foundation, name)
                for name in type(foundation).model_fields
                if name not in {"foundation_hash", "research_foundation"}
            },
            research_foundation=admission.foundation,
        )
    # The sector map this workspace actually neutralized against, so two runs
    # over different sector revisions cannot share a recipe identity.
    lineage = dict(panel_manifest.get("safe_summary", {})).get("lineage", {})
    # Every installed target method, not only the frozen lanes. The successor
    # methods are parameterized by the outcome method the foundation resolved,
    # because the same composition over a different holding span is a different
    # method and must not share an identity with it.
    # Read off the seal the foundation resolved, never off the document. A
    # document that could name its own outcome method would be choosing what its
    # target is a transformation *of*, which is the authority resolution exists
    # to establish.
    # The reader era follows the snapshot the foundation resolved. A
    # development-only snapshot lives under the successor categories and is
    # resolved by the successor reader; handing it to the frozen reader would
    # report a published artifact missing rather than out of scope.
    outcome_reader: CausalOutcomeDevelopmentRows
    if foundation.feature_authority_outcome_snapshot_hash is not None:
        outcome_reader = DevelopmentOnlyExecutionOutcomeReader(outcome_root)
    else:
        outcome_reader = CausalExecutionOutcomeDevelopmentReader(outcome_root)
    outcome_seal = outcome_reader.resolve_method_seal(foundation.execution_outcome.snapshot_hash)
    if outcome_seal.disposition != "METHOD_BOUND":
        raise AuthoringError("alpha_research.development_outcome_method_unbound")
    recipes = installed_alpha_target_methods(
        sector_revision=str(dict(lineage)["sector_revision"]),
        execution_outcome_recipe_id=outcome_seal.method_bound.recipe_id,
        # What the Panel's sessions read.
        sector_history_treatment=sector_treatment(
            reclassified=bool(dict(lineage).get("sector_reclassifications"))
        ),
    )
    # The installed models and those a person activated in this workspace (EX).
    catalog = build_installed_alpha_model_catalog(activated_models(Path(workspace_root)))
    executor: DeskExperimentExecutor = AlphaExperimentExecutor(
        foundation=foundation,
        target_recipes=recipes,
        model_mandate=build_development_alpha_model_mandate(catalog=catalog),
        model_catalog=catalog,
        feature_reader=feature_reader,
        outcome_reader=outcome_reader,
        feature_panel_manifest_ref=manifest_ref,
        causal_outcome_manifest_ref=outcome_manifest_ref,
        ordered_listing_ids=listing_ids,
        sector_by_listing_id=sector_by_listing_id,
        split_policy=split_policy,
        frozen_at=datetime.combine(
            envelope.sessions.as_of.session, datetime.min.time(), tzinfo=UTC
        ),
        panel_factor_ids=_panel_factor_ids(panel_manifest),
    )
    return executor


def build_installed_desk_executors(
    *,
    envelope: ResearchExperimentEnvelope,
    workspace: Path,
    workspace_root: Path,
    document: Mapping[str, Any] | None = None,
    artifact_root: Path | None = None,
    factor_evidence_root: Path | None = None,
    alpha_split_policy: AlphaSplitPolicy | None = None,
    factor_evidence_policy: Any | None = None,
    factor_redundancy_policy: Any | None = None,
    outcome_artifact_root: Path | None = None,
    causal_outcome_snapshot_handle: str | None = None,
    feature_authority_outcome_snapshot_handle: str | None = None,
    panel_methodology_roots: PanelMethodologyRuntimeRoots | None = None,
    panel_runtime_caps: PanelResearchRuntimeCaps | None = None,
    feature_catalog: FeatureCatalog | None = None,
    feature_kernels: FeatureKernelRegistry | None = None,
    sealed_preflight_program_hash: str | None = None,
    preflight_metadata_only: bool = False,
    feature_input: ResolvedDevelopmentFeatureInput | None = None,
) -> tuple[DeskExperimentExecutor, ...]:
    """Install exactly the executor the selected kind needs, and no other.

    Returns a tuple because the workflow takes a tuple, not because more than one
    is ever built: one document selects one Desk, and resolving a second Desk's
    authority to run the first is how an unrelated missing artifact becomes a
    blocking failure.

    ``outcome_artifact_root`` lets a composer read causal outcomes from a root
    other than the workspace's own -- the case a study has when its inputs were
    prepared into a development root because the source workspace is read-only.
    The two outcome handles are Host-resolved selections, never document text:
    the arm's own snapshot for a Factor run, and the one-session snapshot the
    Factor evidence was screened on for an Alpha run whose own clock differs.
    """
    workspace = Path(workspace)
    root = _artifact_root(workspace, artifact_root)
    if envelope.kind == RISK_EXPERIMENT_KIND:
        return (
            _build_risk_executor(
                workspace=workspace,
                artifact_root=root,
                market_profile_id=envelope.universe_handle,
            ),
        )
    if envelope.kind == FACTOR_EXPERIMENT_KIND:
        from alphalattice.control.product_host.research_authoring.factor_inputs import (
            build_factor_executor,
        )

        return (
            build_factor_executor(
                workspace=workspace,
                artifact_root=root,
                envelope=envelope,
                evidence_policy=factor_evidence_policy,
                redundancy_policy=factor_redundancy_policy,
                outcome_artifact_root=outcome_artifact_root,
                causal_outcome_snapshot_handle=causal_outcome_snapshot_handle,
                feature_catalog=feature_catalog,
                feature_kernels=feature_kernels,
                feature_input=feature_input,
            ),
        )
    if envelope.kind == ALPHA_EXPERIMENT_KIND:
        selected_lifecycle = document is not None and lifecycle_section(document) is not None
        selected_panel_methodology = (
            document is not None and methodology_section(document) is not None
        )
        if (
            factor_evidence_root is None
            and not selected_panel_methodology
            and not selected_lifecycle
        ):
            # Host-owned configuration, deliberately not defaulted: guessing a
            # root is how a run silently binds to whichever development evidence
            # happens to be nearby.
            raise AuthoringError("research_authoring.factor_evidence_root_not_installed")
        if document is None or (alpha_split_policy is None and not selected_lifecycle):
            raise AuthoringError("research_authoring.alpha_composition_incomplete")
        return (
            _build_alpha_executor(
                workspace=workspace,
                artifact_root=root,
                envelope=envelope,
                document=document,
                factor_evidence_root=(
                    Path(factor_evidence_root) if factor_evidence_root is not None else None
                ),
                workspace_root=Path(workspace_root),
                split_policy=alpha_split_policy,
                outcome_artifact_root=outcome_artifact_root,
                panel_methodology_roots=panel_methodology_roots,
                panel_runtime_caps=panel_runtime_caps,
                feature_authority_outcome_snapshot_handle=(
                    feature_authority_outcome_snapshot_handle
                ),
                feature_catalog=feature_catalog,
                feature_kernels=feature_kernels,
                sealed_preflight_program_hash=sealed_preflight_program_hash,
                preflight_metadata_only=preflight_metadata_only,
                feature_input=feature_input,
            ),
        )
    raise AuthoringError("research_authoring.desk_executor_not_installed")


__all__ = [
    "PanelMethodologyRuntimeRoots",
    "WorkspaceReturnSurfaceProvider",
    "build_installed_desk_executors",
    "build_panel_methodology_authority_resolver",
]
