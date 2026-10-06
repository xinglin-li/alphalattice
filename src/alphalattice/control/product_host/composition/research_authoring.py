"""Host composition of the installed research experiment compilers.

This is the one place that knows which Desks accept authored experiments, and
the one place that binds them to a concrete authority resolver. Adding a *method*
to a Desk does not change this module; only adding a whole new Desk kind does.
"""

from __future__ import annotations

from pathlib import Path

from alphalattice.control.product_host.research_authoring.authority import (
    InstalledResearchSnapshot,
    WorkspaceResearchAuthorityResolver,
)
from alphalattice.control.research_program.authoring.dispatcher import (
    ResearchExperimentDispatcher,
)
from alphalattice.control.research_program.authoring.workflow import ResearchProgramWorkflow
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.factor_research.experiments.authoring import (
    FactorExperimentCompiler,
    FactorInventoryEntry,
    factor_inventory_from_panel_manifest,
)
from alphalattice.foundation.factor_research.experiments.verification import (
    FactorEvidenceVerifier,
)
from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.investment.risk_research.experiments.compiler import RiskExperimentCompiler
from alphalattice.protocols.research_authoring.contracts import (
    DeskEvidenceVerifier,
    DeskExperimentCompiler,
    DeskExperimentExecutor,
    ResearchAuthorityResolver,
)


def installed_desk_compilers(
    *,
    factor_inventory: tuple[FactorInventoryEntry, ...] = (),
    alpha_compiler: DeskExperimentCompiler | None = None,
) -> tuple[DeskExperimentCompiler, ...]:
    """Install one compiler per admitted Desk kind.

    The Factor inventory is a dynamic ordered axis supplied by the Host, so
    adding or removing an ordinary Factor never edits this module. Each entry
    carries the identity of the *method* that computes it, read off the published
    Feature panel: a bare ID would let a Factor be rewritten under a stable name
    while the Desk's catalog identity stood still.

    Alpha's compiler arrives already built rather than constructed here, and for
    a reason the other two do not share: admitting an Alpha document requires the
    installed target recipes, the model mandate, the Panel's factor axis and the
    axis one Factor development checkpoint answered for. Three of those are read
    from artifacts, and the executor must validate against the same instance or
    the sealed Program and the run could disagree about what was installed.
    """
    compilers: list[DeskExperimentCompiler] = [RiskExperimentCompiler()]
    if factor_inventory:
        compilers.append(FactorExperimentCompiler(factor_inventory))
    if alpha_compiler is not None:
        compilers.append(alpha_compiler)
    return tuple(compilers)


def host_resolved_factor_inventory(
    *, workspace: Path, market_profile_id: str, artifact_root: Path | None = None
) -> tuple[FactorInventoryEntry, ...]:
    """Read the installed Factor axis off the Panel this workspace made active.

    The inventory used to be a caller argument defaulting to empty, so the CLI
    installed no Factor compiler at all and a Factor document could not even be
    validated. Resolving it here is what makes the canonical path Desk-complete
    without the Host learning any Factor methodology: it reads one published
    artifact and hands over what it found.

    Returns empty when the workspace has published no Panel. That is an ordinary
    state for a new workspace, and it must not be an error: a Risk experiment
    needs no Factor axis, and failing here would make one Desk's missing artifact
    block another Desk's run -- the same coupling the kind-aware executor
    installer exists to remove.
    """
    root = Path(artifact_root) if artifact_root else Path(workspace) / "artifacts"
    store = MarketDataRepository(Path(workspace))
    panel_state = PanelStateRepository(store.database, market_data=store)
    snapshot = panel_state.feature_panel_snapshot_for_active(market_profile_id)
    if snapshot is None:
        return ()
    manifest = ArtifactResolver(root).load_feature_panel_manifest(str(snapshot["manifest_uri"]))
    return factor_inventory_from_panel_manifest(dict(manifest))


def build_research_experiment_dispatcher(
    *,
    workspace: Path | None = None,
    artifact_root: Path | None = None,
    installed_snapshots: tuple[InstalledResearchSnapshot, ...] = (),
    authority: ResearchAuthorityResolver | None = None,
    compilers: tuple[DeskExperimentCompiler, ...] | None = None,
    factor_inventory: tuple[FactorInventoryEntry, ...] = (),
    alpha_compiler: DeskExperimentCompiler | None = None,
) -> ResearchExperimentDispatcher:
    """Compose the one installed dispatcher over a real workspace.

    Either a ``workspace`` or an already-built ``authority`` must be supplied.
    There is no default: an authoring boundary that silently invents its own
    session axis is exactly what let unresolved handles reach a Desk compiler.

    ``artifact_root`` is the same operator-named location the Desk executors
    already receive, and it belongs here for the same reason. Without it the
    authority resolver reads ``workspace/artifacts`` while the executor reads the
    named root, so a study whose Panel was published outside its source
    workspace refuses with ``panel_observation_clock_absent`` -- a statement
    about the Panel's clock, when the Panel was simply never looked at.
    """
    if authority is None:
        if workspace is None:
            raise ValueError("research_authoring.authority_not_installed")
        authority = WorkspaceResearchAuthorityResolver(
            workspace=workspace,
            artifact_root=artifact_root,
            installed_snapshots=installed_snapshots,
        )
    return ResearchExperimentDispatcher(
        compilers=compilers
        or installed_desk_compilers(
            factor_inventory=factor_inventory, alpha_compiler=alpha_compiler
        ),
        authority=authority,
    )


def build_research_program_workflow(
    *,
    workspace: Path,
    executors: tuple[DeskExperimentExecutor, ...],
    workspace_root: Path,
    artifact_root: Path | None = None,
    installed_snapshots: tuple[InstalledResearchSnapshot, ...] = (),
    authority: ResearchAuthorityResolver | None = None,
    compilers: tuple[DeskExperimentCompiler, ...] | None = None,
    factor_inventory: tuple[FactorInventoryEntry, ...] = (),
    alpha_compiler: DeskExperimentCompiler | None = None,
    verifier_kinds: tuple[str, ...] | None = None,
) -> ResearchProgramWorkflow:
    """Bind the Desk-neutral workflow to concrete compilers and executors.

    Executors are supplied rather than constructed here: a Desk executor needs
    workspace inputs that only the caller composing the run actually knows,
    such as which published return surface a development build reads.
    """
    return ResearchProgramWorkflow(
        dispatcher=build_research_experiment_dispatcher(
            workspace=workspace,
            artifact_root=artifact_root,
            installed_snapshots=installed_snapshots,
            authority=authority,
            compilers=compilers,
            factor_inventory=factor_inventory,
            alpha_compiler=alpha_compiler,
        ),
        executors=executors,
        workspace_root=workspace_root,
        # The real source workspace, so the workflow can refuse an output that
        # is the source, sits inside it, or contains it. The document's declared
        # baseline is caller input and cannot answer for this.
        source_workspace=workspace,
        verifiers=installed_desk_verifiers(verifier_kinds),
    )


def installed_desk_verifiers(
    kinds: tuple[str, ...] | None = None,
) -> tuple[DeskEvidenceVerifier, ...]:
    """Install one evidence verifier per Desk that publishes artifacts.

    Installed here rather than accepted as a caller argument: an optional
    verifier is one nobody passes, which is precisely how replay came to
    validate a single JSON file and report exact reuse.

    Installation is all the Host does. Each verifier is *owned* by its Desk,
    because how a surface relates to its diagnostics, its dossier and its chunks
    is that Desk's methodology -- and a Host that learned those relationships
    would accumulate every Desk's domain model as the remaining Desks arrive.

    Factor joins them here. It was the one Desk that could produce evidence and
    never replay it, and the missing verifier was reported honestly as
    ``evidence_verifier_not_installed`` rather than papered over -- which was the
    right answer only for as long as nothing walked the graph.

    Every one takes no arguments, including Factor's. That is not a convenience:
    a verifier assembled from caller-supplied inputs is one a caller can point at
    the wrong artifacts, and the whole point is that the sealed Program and the
    Host-resolved authority are the sides the graph did not produce.
    """
    selected = (
        kinds
        if kinds is not None
        else (
            "risk.covariance-development",
            "alpha.model-development",
            "factor.screening-development",
        )
    )
    verifiers: list[DeskEvidenceVerifier] = []
    if "risk.covariance-development" in selected:
        from alphalattice.investment.risk_research.experiments.verification import (
            RiskEvidenceVerifier,
        )

        verifiers.append(RiskEvidenceVerifier())
    if "alpha.model-development" in selected:
        from alphalattice.investment.alpha_research.experiments.verification import (
            AlphaEvidenceVerifier,
        )
        from alphalattice.investment.portfolio_strategy_lab.research_loop import (
            panel_methodology_execution,
        )

        verifiers.append(
            AlphaEvidenceVerifier(
                methodology_verifier=panel_methodology_execution.PanelMethodologyEvidenceVerifier()
            )
        )
    if "factor.screening-development" in selected:
        verifiers.append(FactorEvidenceVerifier())
    if "portfolio.policy-development" in selected:
        from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
            PortfolioExperimentVerifier,
        )

        verifiers.append(PortfolioExperimentVerifier())
    return tuple(verifiers)


__all__ = [
    "build_research_experiment_dispatcher",
    "build_research_program_workflow",
    "host_resolved_factor_inventory",
    "installed_desk_compilers",
    "installed_desk_verifiers",
]
