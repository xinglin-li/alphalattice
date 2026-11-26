"""Materialize local lifecycle research for the existing frozen Portfolio path."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from alphalattice.control.product_host.research_authoring.factor_inputs import (
    _copy_input,
    confined,
    factor_input_paths,
    file_digest,
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.portfolio_handoff import (
    LocalQAOutcomeRows,
    PortfolioMarketContext,
    prepare_portfolio_market_inputs,
)
from alphalattice.control.product_host.storage.inventory import require_storage_capacity
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.contracts import CausalExecutionSchedulePoint
from alphalattice.foundation.causal_outcomes.execution.methods import build_one_session_recipe
from alphalattice.foundation.causal_outcomes.execution.readers import (
    planned_local_qa_schedule,
    read_local_qa_execution_rows,
)
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import SectorRevisionMap
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    copy_sector_history,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.foundation.market_data_ops.publication.session_marks import (
    SessionMarkArtifactStore,
)
from alphalattice.foundation.market_data_ops.returns.sector_revision_identity import (
    sector_revision_hash,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.experiments.lifecycle_authoring import (
    AlphaLifecycleResearchReceipt,
)
from alphalattice.investment.alpha_research.experiments.score_rows import DevelopmentScoreMatrix
from alphalattice.investment.alpha_research.scores.model_renewal import read_lifecycle_projection
from alphalattice.investment.portfolio_strategy_lab.inputs.shared_lanes import outcome_returns
from alphalattice.investment.portfolio_strategy_lab.policies.buffered_rank_return import (
    complete_matured_rank_curve,
    rank_bucket_observations,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    FROZEN_RESEARCH_BOOK_RECIPES,
)
from alphalattice.investment.portfolio_strategy_lab.policies.lifecycle_research import (
    CATEGORY,
    LANES,
    LifecyclePortfolioAuthority,
    LifecycleResearchScoreSource,
    LifecycleScoreEvidence,
    LocalQAOutcomeBinding,
)
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioResearchArtifactStore,
)
from alphalattice.investment.risk_research.surfaces.decomposition import (
    INSTALLED_RISK_DECOMPOSITION_RECIPE,
)
from alphalattice.investment.risk_research.surfaces.returns import (
    CausalRiskReturnReader,
    RiskReturnArtifactStore,
)
from alphalattice.kernel.quant.sector_history import SectorHistory
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class FrozenPortfolioPreparationRequest(BaseModel):  # type: ignore[misc]
    """Declare exact Alpha/Risk parents and explicit local outcome policy for frozen preparation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    input_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    alpha_task_ids: tuple[UUID, ...] = Field(min_length=1)
    risk_task_id: UUID
    unavailable_return_policy: Literal["require_complete", "quarantine_listings"] = (
        "require_complete"
    )
    top_k: Literal[35] = 35


def local_research_economic_schedule(
    *, first: date, last: date, through: date
) -> tuple[CausalExecutionSchedulePoint, ...]:
    """Complete matured QA support, independent of a development-selection cutoff."""
    return tuple(
        point
        for point in planned_local_qa_schedule(first, through)
        if first <= point.formation_session <= last and point.holding_end_session <= through
    )


RISK_HISTORY_SESSIONS = (
    INSTALLED_RISK_DECOMPOSITION_RECIPE.conditional_volatility_initialization_sessions
    + INSTALLED_RISK_DECOMPOSITION_RECIPE.factor_fit_sessions
)
"""The Risk return sessions every book formation needs before it: the installed decomposition's
conditional-volatility initialization and its factor fit."""


def risk_return_surface(workspace: Path, risk: dict[str, Any]) -> tuple[Path, Any]:
    """A Risk study's own return surface in this workspace: its root and its manifest.

    Args:
        workspace: The workspace root.
        risk: The Risk study's exact readback.

    Returns:
        The surface's artifact root and its return manifest.

    Raises:
        ValueError: The study names no single return epoch, or this workspace holds no one
            surface for it.
    """
    return_hash = risk.get("risk_input", {}).get("return_surface_hash")
    if (
        not isinstance(return_hash, str)
        or len(return_hash) != 64
        or any(c not in "0123456789abcdef" for c in return_hash)
    ):
        raise ValueError("frozen_portfolio.single_return_epoch_required")
    matches = tuple(
        (workspace / "artifacts/research-risk-inputs").glob(
            f"*/data-operations/risk-returns/manifests/{return_hash}.json"
        )
    )
    if len(matches) != 1 or not matches[0].resolve().is_relative_to(workspace):
        raise ValueError("frozen_portfolio.return_surface_unresolved")
    source_root = matches[0].parents[3]
    return source_root, RiskReturnArtifactStore(source_root).load_manifest(return_hash)


def risk_history_shortfall(
    risk_sessions: tuple[date, ...] | list[date], formations: Iterable[date]
) -> tuple[date, int] | None:
    """The first formation with fewer Risk return sessions before it than `RISK_HISTORY_SESSIONS`.

    The one judgment of a strategy's Risk history, which its plan reads before any Task and
    its materialization reads again: a plan admits only what its materialization runs.

    Args:
        risk_sessions: The Risk return surface's available sessions, in order.
        formations: The book's formation sessions, each one the surface holds.

    Returns:
        That formation and the sessions held before it; None when every formation has its
        history.
    """
    positions = {session: index for index, session in enumerate(risk_sessions)}
    for session in formations:
        held = positions.get(session, 0)
        if held < RISK_HISTORY_SESSIONS:
            return session, held
    return None


def prepare_frozen_portfolio_authority(
    *,
    workspace: Path,
    request: FrozenPortfolioPreparationRequest,
    output_root: Path,
    read_experiment: Callable[[UUID], dict[str, Any]],
    implementation_hash: str,
    observed_at: datetime,
    cancelled: Callable[[], bool] = lambda: False,
) -> LifecyclePortfolioAuthority:
    """Prove parents and materialize small numeric lanes; no fit, book or activation.

    The caller owns Task admission and the output root. The installed immutable
    recipes own the component set and the calibration method, not a CLI script.
    """
    workspace = workspace.resolve()
    if not output_root.resolve().is_relative_to(workspace):
        raise ValueError("frozen_portfolio.output_outside_workspace")
    bundle = read_factor_bundle(workspace, request.input_binding_hash)
    source, artifacts = factor_input_paths(workspace, request.input_binding_hash)
    resolver = ArtifactResolver(artifacts)
    panel_ref = resolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
    panel = resolver.load_feature_panel_manifest(panel_ref)
    listings = tuple(FeaturePanelReader(resolver).listing_ids(panel_ref))
    recipes = FROZEN_RESEARCH_BOOK_RECIPES
    required_components = {v.component_id for recipe in recipes for v in recipe.components}
    calibrated = {
        v.component_id for recipe in recipes for v in recipe.components if v.weight_rule == "mu.iv0"
    }
    if len(calibrated) != 1:
        raise ValueError("frozen_portfolio.single_calibration_component_required")
    primary_id = next(iter(calibrated))
    store = PortfolioResearchArtifactStore(output_root)

    def checkpoint() -> None:
        if cancelled():
            raise ValueError("frozen_portfolio.cancelled_at_safe_checkpoint")

    def lane(values: npt.ArrayLike, dtype: str) -> str:
        checkpoint()
        array = np.ascontiguousarray(values, dtype=dtype)
        if not store.holds_array(category=LANES, content_hash=sha256(array.tobytes()).hexdigest()):
            require_storage_capacity(workspace, additional_bytes=array.nbytes)
        return store.publish_array(category=LANES, values=array)

    components = []
    score_values: dict[str, np.ndarray] = {}
    for task_id in request.alpha_task_ids:
        checkpoint()
        body = read_experiment(task_id)
        if (
            body.get("status") != "EXPERIMENT_PUBLISHED"
            or body.get("input_binding_hash") != request.input_binding_hash
        ):
            raise ValueError("frozen_portfolio.alpha_input_mismatch")
        receipt = AlphaLifecycleResearchReceipt.model_validate(body.get("lifecycle_research"))
        component_id = body["document"]["alpha"]["component_recipe_id"]
        if component_id not in required_components or component_id in score_values:
            raise ValueError("frozen_portfolio.component_selection_invalid")
        alpha_store = AlphaDevelopmentArtifactStore(
            confined(workspace, body["document"]["experiment"]["output_workspace"])
            / "alpha-lifecycle"
        )
        projections = [read_lifecycle_projection(alpha_store, h) for h in receipt.projection_files]
        if any(p.ordered_listing_ids != listings for p in projections):
            raise ValueError("frozen_portfolio.alpha_listing_axis_mismatch")
        values = np.stack([p.scores for p in projections])
        score_values[component_id] = values
        components.append(
            LifecycleScoreEvidence(
                component_id=component_id,
                task_id=task_id,
                program_hash=receipt.program_hash,
                receipt_hash=receipt.content_hash,
                component_recipe_hash=receipt.component_recipe_hash,
                lifecycle_hash=receipt.lifecycle.content_hash,
                formation_sessions=tuple(
                    date.fromisoformat(day) for day in receipt.formation_sessions
                ),
                scores_hash=lane(values, "<f8"),
            )
        )
    if {v.component_id for v in components} != required_components:
        raise ValueError("frozen_portfolio.complete_component_set_required")
    primary = next(v for v in components if v.component_id == primary_id)
    # This fixed-recipe Task admits post-observed local QA, not an ordinary
    # adaptive development experiment. Use the existing explicitly-scoped raw
    # observation owner; no sealed holdout reader/release is invoked.
    schedule = tuple(
        point.model_dump(mode="python")
        for point in local_research_economic_schedule(
            first=primary.formation_sessions[0],
            last=primary.formation_sessions[-1],
            through=bundle.sessions[-1],
        )
    )
    sessions = tuple(row["formation_session"] for row in schedule)
    if not sessions or not set(sessions) <= set(primary.formation_sessions):
        raise ValueError("frozen_portfolio.complete_economic_support_required")
    qa_rows, raw_bars, raw_axis = read_local_qa_execution_rows(
        store=MarketDataRepository(source),
        sessions=sessions,
        listing_ids=listings,
        through=bundle.sessions[-1],
        observed_at=observed_at,
    )
    del raw_bars, raw_axis
    qa_rows = qa_rows.sort_by([("formation_session", "ascending"), ("listing_id", "ascending")])
    qa_returns = outcome_returns(qa_rows, sessions=sessions, listings=listings)
    qa = LocalQAOutcomeBinding.create(
        input_binding_hash=request.input_binding_hash,
        method_recipe_hash=build_one_session_recipe().recipe_hash,
        source_rows_hash=canonical_hash(qa_rows["row_hash"].to_pylist()),
        returns_hash=sha256(np.ascontiguousarray(qa_returns, dtype="<f8").tobytes()).hexdigest(),
        formation_sessions=sessions,
        listing_axis_hash=canonical_hash(listings),
        through=bundle.sessions[-1],
    )
    positions = {day: i for i, day in enumerate(primary.formation_sessions)}
    primary_values = score_values[primary_id][[positions[day] for day in sessions]]
    # This is a numerical view, not a fabricated candidate/fold receipt.
    view = DevelopmentScoreMatrix(
        sessions, listings, primary_values, np.isfinite(primary_values), (), primary.scores_hash
    )
    context = PortfolioMarketContext(
        input_binding_hash=request.input_binding_hash,
        universe_revision=panel["safe_summary"]["lineage"]["manifest_revision"],
        panel_snapshot_hash=bundle.panel_snapshot_hash,
        outcome_snapshot_hash=bundle.outcome_snapshot_hash,
        formation_sessions=sessions,
        ordered_listing_ids=listings,
    )
    market = prepare_portfolio_market_inputs(
        source=context,
        source_workspace=source,
        artifact_root=artifacts,
        output=output_root,
        scores=view,
        schedule=schedule,
        cancellation=cancelled,
        spec=request,
        capacity=lambda amount: require_storage_capacity(workspace, additional_bytes=amount),
        local_qa=LocalQAOutcomeRows(qa, qa_rows),
    )
    checkpoint()
    assert market.market_decision_eligible is not None and market.transition_binding is not None
    assert market.tradability_decision_hash is not None
    decision = market.market_decision_eligible
    observations = rank_bucket_observations(
        scores=primary_values,
        realized_simple_returns=market.realized_simple_returns,
        decision_eligible=decision,
        rank_keys=np.arange(len(listings)),
    )
    curve, latest = complete_matured_rank_curve(
        observations=observations,
        observation_sessions=sessions,
        holding_end_sessions=tuple(row["holding_end_session"] for row in schedule),
        decision_sessions=sessions,
    )
    risk = read_experiment(request.risk_task_id)
    if (
        risk.get("status") != "EXPERIMENT_PUBLISHED"
        or risk.get("input_binding_hash") != request.input_binding_hash
    ):
        raise ValueError("frozen_portfolio.risk_input_mismatch")
    source_root, returns = risk_return_surface(workspace, risk)
    return_hash = str(risk["risk_input"]["return_surface_hash"])
    source_returns = RiskReturnArtifactStore(source_root)
    target_returns = RiskReturnArtifactStore(output_root)
    if (
        returns.epoch.ordered_listing_ids != listings
        or returns.epoch.universe_manifest_revision != context.universe_revision
    ):
        raise ValueError("frozen_portfolio.risk_epoch_mismatch")
    risk_sessions = CausalRiskReturnReader(source_root).available_sessions(returns)
    risk_positions = {session: index for index, session in enumerate(risk_sessions)}
    if not set(sessions) <= risk_positions.keys():
        raise ValueError("frozen_portfolio.risk_return_support_incomplete")
    if risk_history_shortfall(risk_sessions, sessions) is not None:
        raise ValueError("frozen_portfolio.risk_history_insufficient")
    for chunk in returns.chunks:
        checkpoint()
        origin = source_returns.resolve_chunk(chunk)
        destination = target_returns._chunk_path(chunk.content_hash)
        if not destination.exists():
            _copy_input(workspace, origin, destination, file_digest(origin))
        target_returns.resolve_chunk(chunk)
    target_returns.publish_manifest(returns)
    expected_sector = panel["safe_summary"]["lineage"]["sector_revision"]
    maps = {}
    # Captured inputs may omit the separate map. Its creating workspace may
    # retain that immutable artifact; verify its semantic revision against the
    # sealed input, never a current pointer or just a self-declared label.
    for source_root in (artifacts, workspace / "artifacts"):
        for value in PanelClosureArtifactStore(ArtifactResolver(source_root)).find_models(
            category="sector-maps",
            model=SectorRevisionMap,
            matches=lambda value: (
                value.manifest_revision == context.universe_revision
                and value.sector_revision == expected_sector
            ),
        ):
            if (
                sector_revision_hash(
                    manifest_revision=value.manifest_revision,
                    observations=[entry.model_dump(mode="json") for entry in value.entries],
                )
                != expected_sector
            ):
                raise ValueError("frozen_portfolio.sector_revision_not_its_evidence")
            maps[value.map_hash] = (value, source_root)
    if len(maps) != 1:
        raise ValueError("frozen_portfolio.sector_map_unresolved")
    sector, sector_root = next(iter(maps.values()))
    target_store = PanelClosureArtifactStore(ArtifactResolver(output_root))
    target_store.publish_json(
        category="sector-maps", content_hash=sector.map_hash, payload=sector.model_dump(mode="json")
    )
    # The history its sessions read, with the receipts that record it, so the book's Portfolio
    # reads what its Panel read; refused when the two disagree.
    history = copy_sector_history(
        PanelClosureArtifactStore(ArtifactResolver(sector_root)),
        target_store,
        sector.sector_revision,
    )
    held = history.subset(listing for listing in listings if listing in history)
    if (
        SectorHistory.of_panel(panel["safe_summary"]["lineage"], held.current).reclassifications
        != held.reclassifications
    ):
        raise ValueError("frozen_portfolio.sector_history_not_its_evidence")
    mark_store = SessionMarkArtifactStore(output_root)
    mark_surface = mark_store.load_manifest(market.market_binding["session_mark"])
    marks = mark_store.read_marks(mark_surface, sessions=sessions, listing_ids=listings)
    authority = LifecyclePortfolioAuthority.create(
        input_binding_hash=request.input_binding_hash,
        preparation_request_hash=canonical_hash(request.model_dump(mode="json")),
        preparation_implementation_hash=implementation_hash,
        components=tuple(sorted(components, key=lambda v: v.component_id)),
        formation_sessions=sessions,
        ordered_listing_ids=listings,
        decision_hash=lane(decision, "?"),
        execution_hash=lane(market.execution_available, "?"),
        returns_hash=lane(market.realized_simple_returns, "<f8"),
        adv_hash=lane(market.causal_adv20, "<f8"),
        marks_hash=lane(marks, "<f8"),
        entry_sessions=tuple(row["entry_session"] for row in schedule),
        holding_end_sessions=tuple(row["holding_end_session"] for row in schedule),
        transition=market.transition_binding,
        tradability_decision_hash=market.tradability_decision_hash,
        outcome_snapshot_hash=qa.content_hash,
        qa_outcome=qa,
        risk_return_surface_hash=return_hash,
        sector_map_hash=sector.map_hash,
        curve_hash=lane(curve, "<f8"),
        latest_calibration_hash=lane(latest, "<i8"),
        unavailable_return_policy=request.unavailable_return_policy,
        data_exclusions=market.data_exclusions,
    )
    store.publish(category=CATEGORY, value=authority, identity_field="authority_hash")
    path = store.root / CATEGORY / f"{authority.authority_hash}.json"
    for recipe in recipes:
        source_owner = LifecycleResearchScoreSource(manifest_path=path, recipe=recipe)
        source_owner.verify()
    return authority
