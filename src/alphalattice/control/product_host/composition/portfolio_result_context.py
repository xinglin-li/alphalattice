"""Source-bound reading context for a saved canonical Portfolio, never execution."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import confined
from alphalattice.control.product_host.research_authoring.timing import binding_temporal_scope
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import SectorRevisionMap
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    sector_history_as_of,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioExecutionProgram,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    FrozenStrategyPackage,
    recipe_hashes_match,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    FROZEN_RESEARCH_BOOK_RECIPES,
)
from alphalattice.investment.portfolio_strategy_lab.policies.lifecycle_research import (
    ARTIFACT_KEY,
    CATEGORY,
    LifecyclePortfolioAuthority,
)
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioResearchArtifactStore,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.static import PortfolioReportContext


def installed_source_binding(workspace: Path, manifest: ResearchWorkspaceManifest) -> str | None:
    """The research input the installed strategy's lifecycle authority was built on, if kept.

    Args:
        workspace: The Host's workspace.
        manifest: Its research workspace manifest.

    Returns:
        The input's binding hash, or None when no installed authority is kept.
    """
    reference = next(
        (a for a in manifest.strategy_artifacts if a.artifact_key == ARTIFACT_KEY), None
    )
    if reference is None:
        return None
    try:
        path = confined(workspace, reference.relative_path)
        authority = PortfolioResearchArtifactStore(path.parents[2]).load(
            category=CATEGORY,
            content_hash=path.stem,
            model=LifecyclePortfolioAuthority,
            identity_field="authority_hash",
        )
    except (ValueError, OSError, KeyError):
        return None
    return str(authority.input_binding_hash)


def installed_temporal_statements(
    workspace: Path, manifest: ResearchWorkspaceManifest, start: date, end: date
) -> tuple[str, ...]:
    """What an installed book can claim about time, from its strategy's research input (V347).

    Args:
        workspace: The Host's workspace.
        manifest: Its research workspace manifest.
        start: The book's first session.
        end: Its last.

    Returns:
        The statements, none when the installed input or its Panel is not kept.
    """
    binding = installed_source_binding(workspace, manifest)
    if binding is None:
        return ()
    scope = binding_temporal_scope(workspace, binding, window_start=start, window_end=end)
    return tuple(scope.get("statements", ())) if scope.get("status") == "RECORDED" else ()


def saved_portfolio_context(
    *,
    workspace: Path,
    packages: Mapping[str, FrozenStrategyPackage],
    manifest: ResearchWorkspaceManifest,
    program: PortfolioExecutionProgram,
    session: date,
) -> dict[str, Any]:
    """Unknown/missing historical sources stay explicit; never consult current data."""
    package = packages.get(program.strategy_package_hash or "")
    context: dict[str, Any] = {
        "strategy": None if package is None else package.model_dump(mode="json"),
        "source": {
            "status": "SOURCE_CONTEXT_NOT_AVAILABLE",
            "authority_hash": program.alpha_evidence_manifest_hash,
        },
        "sectors": {"status": "UNAVAILABLE", "by_listing": {}},
        "listing_labels": {},
    }
    recipe = next(
        (
            r
            for r in FROZEN_RESEARCH_BOOK_RECIPES
            if recipe_hashes_match(r.recipe_hash, program.policy_recipe_hash)
        ),
        None,
    )
    if recipe is not None:
        context["book_policy"] = {
            "policy_recipe_hash": recipe.recipe_hash,
            "configuration": "PACKAGE_FROZEN",
            "sleeve_notional": "PRESERVE_DRIFTED_SLEEVE_NOTIONAL",
            "initial_staging": "ALL_SLEEVES",
            "review_phase": recipe.review_phase,
            "aggregate_cap_start_formation": recipe.aggregate_cap_start_formation,
            "sizing_activation_formation": recipe.sizing_activation_formation,
            "notice": (
                "This exact frozen recipe governs the book; "
                "legacy generic report labels do not redefine it."
            ),
        }
    reference = next(
        (a for a in manifest.strategy_artifacts if a.artifact_key == ARTIFACT_KEY), None
    )
    if reference is None:
        return context
    try:
        path = confined(workspace, reference.relative_path)
        if path.stem != program.alpha_evidence_manifest_hash:
            raise ValueError("portfolio_application.saved_source_not_installed")
        store = PortfolioResearchArtifactStore(path.parents[2])
        authority = store.load(
            category=CATEGORY,
            content_hash=path.stem,
            model=LifecyclePortfolioAuthority,
            identity_field="authority_hash",
        )
        if (
            authority.sector_map_hash != program.sector_map_hash
            or session not in authority.formation_sessions
        ):
            raise ValueError("portfolio_application.saved_source_binding_mismatch")
        row = authority.formation_sessions.index(session)
        context["source"] = {
            "status": "BOUND_SOURCE_METADATA",
            "authority_hash": authority.authority_hash,
            "input_binding_hash": authority.input_binding_hash,
            "input_id": next(
                (
                    v.input_id
                    for v in manifest.experiment_inputs or ()
                    if v.binding_hash == authority.input_binding_hash
                ),
                None,
            ),
            "input_end": None
            if authority.qa_outcome is None
            else authority.qa_outcome.through.isoformat(),
            "alpha_tasks": [
                {
                    **v.model_dump(mode="json", exclude={"formation_sessions"}),
                    "target_recipe": next(
                        c.target_recipe
                        for c in package.component_plan
                        if c.component_id == v.component_id
                    ),
                    "allocation_basis_points": next(
                        c.allocation_basis_points
                        for c in package.component_plan
                        if c.component_id == v.component_id
                    ),
                }
                for v in authority.components
                if package is not None and v.component_id in package.component_ids
            ],
            "unavailable_return_policy": authority.unavailable_return_policy,
            "data_exclusion_count": len(authority.data_exclusions),
            "source_listing_count": len(authority.ordered_listing_ids),
        }
        context["temporal_scope"] = binding_temporal_scope(
            workspace,
            authority.input_binding_hash,
            window_start=authority.formation_sessions[0],
            window_end=session,
        )
        context["position_clock"] = {
            "formation_session": session.isoformat(),
            "decision_phase": "CLOSE_T",
            "entry_session": authority.entry_sessions[row].isoformat(),
            "execution_phase": "OPEN_T_PLUS_1",
            "holding_end_session": authority.holding_end_sessions[row].isoformat(),
            "state_transition_binding_hash": authority.transition.binding_hash,
        }
        context["sectors"], context["listing_labels"] = _saved_sector_context(
            path.parents[2], program.sector_map_hash, session
        )
    except (ValueError, OSError, KeyError) as error:
        context["source"] = {
            **context["source"],
            "status": "SOURCE_CONTEXT_UNAVAILABLE",
            "reason": str(error),
        }
    return context


def saved_portfolio_sectors(
    *,
    workspace: Path,
    manifest: ResearchWorkspaceManifest,
    program: PortfolioExecutionProgram,
    session: date,
) -> dict[str, Any]:
    """Read the book's sealed Sector history at a forward formation, never current labels.

    A forward formation need not occur in the original historical return window. Its
    classification still names the book's exact retained Sector revision and dated history.
    Missing or mismatched sources remain explicitly unavailable.
    """
    reference = next(
        (a for a in manifest.strategy_artifacts if a.artifact_key == ARTIFACT_KEY), None
    )
    if reference is None:
        return {"status": "UNAVAILABLE", "by_listing": {}}
    try:
        path = confined(workspace, reference.relative_path)
        if path.stem != program.alpha_evidence_manifest_hash:
            raise ValueError("portfolio_application.saved_source_not_installed")
        authority = PortfolioResearchArtifactStore(path.parents[2]).load(
            category=CATEGORY,
            content_hash=path.stem,
            model=LifecyclePortfolioAuthority,
            identity_field="authority_hash",
        )
        if authority.sector_map_hash != program.sector_map_hash:
            raise ValueError("portfolio_application.saved_source_binding_mismatch")
        sectors, _ = _saved_sector_context(path.parents[2], program.sector_map_hash, session)
        return sectors
    except (ValueError, OSError, KeyError) as error:
        return {"status": "UNAVAILABLE", "reason": str(error), "by_listing": {}}


def _saved_sector_context(
    artifact_root: Path, sector_map_hash: str, session: date
) -> tuple[dict[str, Any], dict[str, str]]:
    """One sealed classification reader shared by historical and forward positions."""
    closure_store = PanelClosureArtifactStore(ArtifactResolver(artifact_root))
    sector = closure_store.load_model(
        category="sector-maps", content_hash=sector_map_hash, model=SectorRevisionMap
    )
    return (
        {
            "status": "BOOK_EXECUTION",
            "sector_revision": sector.sector_revision,
            "coverage_hash": sector.map_hash,
            "formation_session": session.isoformat(),
            "by_listing": sector_history_as_of(closure_store, sector.sector_revision).at(session),
        },
        {value.listing_id: value.provider_symbol for value in sector.entries},
    )


def portfolio_report_context(
    *,
    workspace: Path,
    packages: Mapping[str, FrozenStrategyPackage],
    manifest: ResearchWorkspaceManifest,
    program: PortfolioExecutionProgram,
    session: date,
) -> PortfolioReportContext:
    """The static report consumes the same bound policy/names as dated CLI/Web readback."""
    context = saved_portfolio_context(
        workspace=workspace, packages=packages, manifest=manifest, program=program, session=session
    )
    return PortfolioReportContext(
        program_hash=program.program_hash,
        book_policy=context.get("book_policy", {}),
        listing_labels=context["listing_labels"],
    )
