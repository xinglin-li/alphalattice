"""Source-bound reading context for a saved canonical Portfolio, never execution."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID

from alphalattice.control.product_host.composition.decision_advancement import (
    DecisionAdvancementPlan,
)
from alphalattice.control.product_host.composition.portfolio_updates import (
    TASK_KIND as PORTFOLIO_UPDATE_TASK_KIND,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import confined
from alphalattice.control.product_host.research_authoring.timing import binding_temporal_scope
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.content_store import verified_model_read_scope
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import SectorRevisionMap
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    sector_history_as_of,
)
from alphalattice.interface.local_application.activity import refusal_code, returned_status
from alphalattice.interface.local_application.failure_codes import (
    FAILURE_DETAIL_WITHHELD,
    owner_failure_code,
    safe_failure_code,
)
from alphalattice.interface.local_application.portfolio_research import (
    OperationCaller,
    PortfolioResearchOperationRequest,
)
from alphalattice.interface.local_application.web import LocalWebError
from alphalattice.investment.alpha_research.scores.model_renewal import (
    verified_lifecycle_admissions,
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
    """What an installed book can claim about time, from its strategy's research input.

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


def forward_update_read_requests(
    operations: Any,
    *,
    source_book_task_id: UUID,
    strategy_package_id: str,
) -> tuple[PortfolioResearchOperationRequest, ...]:
    """Select the latest retained task of each update kind for this exact source book.

    Args:
        operations: The installed operation owners and their shared task registry.
        source_book_task_id: Historical book selected by the page.
        strategy_package_id: Package the selected historical result recorded.

    Returns:
        Exact task readbacks whose verified checkpoint names this book and package.
    """
    if operations.updates is None:
        return ()
    result: list[PortfolioResearchOperationRequest] = []
    tasks = tuple(reversed(operations.workspace_session.task_control_registry.tasks()))
    for operation, owner, kind in (
        (
            "RESEARCH_UPDATE_READBACK",
            operations.research_updates,
            None if operations.research_updates is None else operations.research_updates.task_kind,
        ),
        ("PORTFOLIO_UPDATE_READBACK", operations.updates, PORTFOLIO_UPDATE_TASK_KIND),
    ):
        if owner is None:
            continue
        for task in tasks:
            if task.task_kind != kind:
                continue
            plan = owner._plan_of(task, current=False)
            package = (
                plan.package_id
                if isinstance(plan, DecisionAdvancementPlan)
                else plan.strategy_package_id
            )
            if package != strategy_package_id:
                continue
            checkpoint = operations.updates.store.load_decision_checkpoint(plan.checkpoint_hash)
            if (
                checkpoint.book_task_id == source_book_task_id
                and checkpoint.package.strategy_id == strategy_package_id
            ):
                result.append(
                    PortfolioResearchOperationRequest(
                        operation=cast(
                            Literal["RESEARCH_UPDATE_READBACK", "PORTFOLIO_UPDATE_READBACK"],
                            operation,
                        ),
                        task_id=task.task_id,
                        strategy_package_id=strategy_package_id,
                    )
                )
                break
    return tuple(result)


@verified_model_read_scope(reuse_verified=True)
@verified_lifecycle_admissions()
def read_portfolio(
    operations: Any,
    request: PortfolioResearchOperationRequest,
    *,
    caller: OperationCaller = "HUMAN",
) -> dict[str, Any]:
    """Read one exact book and its retained Forward publications for any local consumer.

    Args:
        operations: The installed read owners and workspace session.
        request: Exact book, optional saved session and Forward selection.
        caller: The foreground or service caller performing this read.

    Returns:
        The verified book projection or its owner's refusal.
    """

    def execute(request: PortfolioResearchOperationRequest) -> dict[str, Any]:
        body = (
            operations.execute(request)
            if caller == "HUMAN"
            else operations.execute(request, caller=caller)
        )
        if caller == "SERVICE_AUTOMATION" and (returned_status(body) or "").startswith("REFUSED"):
            # A background refusal must not complete a successful warming cycle,
            # including one from a nested retained update readback.
            raise LocalWebError(safe_failure_code(refusal_code(body)) or FAILURE_DETAIL_WITHHELD)
        return cast(dict[str, Any], body)

    from alphalattice.interface.local_application.workbench_projection import (
        forward_holdings_view,
        portfolio_view,
    )

    performance = request.performance
    scope = request.portfolio_scope
    if scope is not None and (scope != "holdings" or performance == "latest"):
        raise LocalWebError("local_web.query_parameter_invalid:scope")

    def with_forward_performance(view: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        # A historical date selection changes only its saved positions. The client
        # keeps the same exact book's previously read rolling report; a full refresh
        # still verifies new daily publications and updates the report.
        if scope == "holdings":
            return view
        if body.get("reading_kind") != "INSTALLED_RESULT":
            if performance != "latest":
                return view
            return {
                **view,
                "forward_performance": {"status": "NO_INSTALLED_STRATEGY", "available": False},
                "forward_holdings": {"status": "NO_INSTALLED_STRATEGY", "available": False},
            }
        package = cast(dict[str, str], body["spec"])["strategy_package_id"]
        updates = [
            execute(request)
            for request in forward_update_read_requests(
                operations, source_book_task_id=task_id, strategy_package_id=package
            )
        ]
        realized = [
            cast(dict[str, Any], update["realized_performance"])
            for update in updates
            if update.get("strategy_package_id") == package
            and isinstance(update.get("realized_performance"), dict)
            and cast(dict[str, Any], update["realized_performance"]).get("source_book_task_id")
            == str(task_id)
        ]
        if not realized:
            if performance != "latest":
                return view
            return {
                **view,
                "forward_performance": {
                    "status": "NO_REALIZED_FORWARD_PUBLICATION",
                    "available": False,
                },
                "forward_holdings": {
                    "status": "NO_RECORDED_FORWARD_HOLDINGS",
                    "available": False,
                },
            }
        latest = max(
            realized,
            key=lambda row: (
                row["observed_through"],
                row["published_at"],
                row["publication_hash"],
            ),
        )
        # The book's exact quoted cost selects an already sealed return lane.
        # A quote outside the two recorded lanes remains explicitly unavailable.
        from alphalattice.capabilities.portfolio_backtesting.contracts import (
            PortfolioPerSideCostAssumption,
        )

        cost = PortfolioPerSideCostAssumption.from_bps_per_side(
            cast(dict[str, str], body["spec"])["cost_bps_per_side"]
        )
        lane = str(cost.cost_bps_per_side.normalize())
        selected = latest["cost_lanes"].get(lane)
        update = next(row for row in updates if row.get("realized_performance") == latest)
        from alphalattice.control.product_host.composition.rolling_portfolio_report import (
            read_rolling_report,
        )
        from alphalattice.investment.portfolio_strategy_lab.application import (
            decision_updates,
        )

        assert operations.application is not None and operations.updates is not None
        publication = operations.updates.store.content.load_model(
            category="decision-updates",
            content_hash=update["publication"]["content_hash"],
            model=decision_updates.PortfolioUpdatePublication,
            identity_field="content_hash",
        )
        checkpoint = operations.updates.store.load_decision_checkpoint(
            publication.input_checkpoint_hash or publication.checkpoint_hash
        )
        history = tuple(
            operations.updates.store.content.load_model(
                category="decision-updates",
                content_hash=row["content_hash"],
                model=decision_updates.PortfolioUpdatePublication,
                identity_field="content_hash",
            )
            for row in update["history"]
        )
        assert manifest is not None
        program = operations.application.ledger.load_program(manifest.program_hash)
        sectors = saved_portfolio_sectors(
            workspace=operations.application.workspace,
            manifest=operations.workspace_manifest,
            program=program,
            session=publication.pending_proposal.schedule.formation_session
            if publication.pending_proposal is not None
            else publication.book.schedule.formation_session,
        )
        rolling = read_rolling_report(
            application=operations.application, checkpoint=checkpoint, publications=history
        )
        # Benchmark values exist only for the base report's own recorded axis.
        # A new daily result never manufactures a benchmark observation.
        base_rows = {row["date"]: row for row in view.get("series", [])}
        for row in cast(list[dict[str, Any]], rolling["curve"]):
            base = base_rows.get(row["formation_session"], {})
            row["benchmark"] = base.get("benchmark")
            row["benchmarkDaily"] = base.get("benchmarkDaily")
        view = {
            **view,
            "rolling_performance": rolling,
            "forward_holdings": forward_holdings_view(
                checkpoint=checkpoint,
                publication=publication,
                history=history,
                source_book_task_id=str(task_id),
                strategy_package_id=package,
                reading_task_id=update["task_id"],
                publication_is_previous=update.get("publication_is_previous", False),
                sectors=sectors,
            ),
        }
        if performance != "latest":
            return view
        return {
            **view,
            "forward_performance": (
                {
                    "status": "RECORDED_REALIZED_FORWARD_WINDOW",
                    "available": True,
                    "reading_task_id": update["task_id"],
                    "publication_is_previous": update.get("publication_is_previous", False),
                    **selected,
                }
                if selected is not None
                else {
                    "status": "COST_LANE_NOT_RECORDED",
                    "available": False,
                    "cost_bps_per_side": lane,
                }
            ),
        }

    assert request.task_id is not None
    task_id = request.task_id
    task = str(task_id)
    manifest = (
        operations.application.pipeline.find_for_task(task_id)
        if operations.application is not None
        else None
    )
    if manifest is not None:
        selected = request.portfolio_session
        if selected is None:
            # The exact result's recorded report end, never today's date or
            # a newest result. The manifest's verified report supplies its
            # end, so the complete REPORT is requested only once.
            assert operations.application is not None
            saved = operations.application.ledger.load_report(manifest.report_hash)
            selected = saved.window_guard.selected_end.isoformat()
        body = execute(
            PortfolioResearchOperationRequest(
                operation="REPORT",
                result_hash=manifest.result_hash,
                portfolio_session=selected,
            )
        )
        if body.get("status") == "REFUSED":
            return body
        if body.get("originating_task_id") != task:
            raise LocalWebError("portfolio_application.result_readback_mismatch")
        return with_forward_performance(portfolio_view(body), body)
    body = execute(
        PortfolioResearchOperationRequest(
            operation="EXPERIMENT_READBACK",
            task_id=task_id,
            portfolio_session=request.portfolio_session,
        )
    )
    if body.get("status") == "REFUSED":
        return body
    return with_forward_performance(
        portfolio_view(body, sectors=book_sectors(operations.workspace_session.workspace, body)),
        body,
    )


_SECTOR_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


def book_sectors(workspace: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    """The sector each listing of a saved book was classified with by the book's own execution.

    Resolved exactly as the Portfolio handoff resolved it when the book ran: the sealed
    research input's own source workspace, its universe manifest at the book's revision, the
    Panel manifest of the book's snapshot, and the Feature state's Sector classification for
    the book's listing axis, as it stood at the session shown. A sealed input never changes,
    so the answer is kept per book source and session.
    Absent, with the owner's reason, when any hop fails; nothing is guessed.
    """
    from alphalattice.control.product_host.research_authoring.execution import (
        _sector_by_listing_id,
    )
    from alphalattice.control.product_host.research_authoring.factor_inputs import (
        factor_input_paths,
    )
    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository

    source = body.get("portfolio_source") or {}
    binding, snapshot = source.get("input_binding_hash"), source.get("panel_snapshot_hash")
    revision, listings = source.get("universe_revision"), source.get("ordered_listing_ids")
    if not all(isinstance(v, str) for v in (binding, snapshot, revision)) or not isinstance(
        listings, list
    ):
        return {
            "status": "UNAVAILABLE",
            "reason": "workbench.portfolio_source_incomplete",
            "by_listing": {},
        }
    shown = (body.get("position") or {}).get("session")
    key = (str(binding), str(snapshot), str(shown))
    cached = _SECTOR_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        source_workspace, artifact_root = factor_input_paths(workspace, str(binding))
        manifest = MarketDataRepository(source_workspace).load_universe_manifest_revision(
            str(revision)
        )
        resolver = ArtifactResolver(artifact_root)
        panel_manifest = resolver.load_feature_panel_manifest(
            resolver.feature_panel_manifest_uri(str(snapshot))
        )
        labels, coverage_hash = _sector_by_listing_id(
            workspace=source_workspace,
            market_profile_id=manifest.profile.market_profile_id,
            panel_manifest=panel_manifest,
            listing_ids=tuple(str(v) for v in listings),
        )
    except Exception as error:
        reason = owner_failure_code(error)
        if reason is None and not isinstance(error, (ValueError, OSError, KeyError, TypeError)):
            raise
        reason = reason or "workbench.book_sectors_unavailable"
        return {"status": "UNAVAILABLE", "reason": reason, "by_listing": {}}
    value = {
        "status": "BOOK_EXECUTION",
        "sector_revision": str(
            (panel_manifest.get("safe_summary") or {}).get("lineage", {}).get("sector_revision")
            or ""
        ),
        "coverage_hash": coverage_hash,
        "by_listing": (
            labels.at(date.fromisoformat(shown)) if isinstance(shown, str) else dict(labels)
        ),
    }
    _SECTOR_CACHE[key] = value
    return value
