"""Publish execution outcomes for development workspace tests."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, cast

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.causal_outcomes.execution.publication import (
    CausalExecutionOutcomePublisher,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data_layer import history
from tests._support.development_universe import OBSERVED_AT

if TYPE_CHECKING:
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
    from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace


def publish_causal_outcomes(
    workspace: RealRiskWorkspace, *, at: datetime = OBSERVED_AT, recipe_id: str | None = None
) -> tuple[str, str]:
    """Publish the causal execution outcomes a Factor program reads, for real.

    Kept out of ``build_real_risk_workspace`` deliberately. Risk needs a return
    surface and never touches execution outcomes, so publishing them in the
    shared builder would slow every Risk case for an artifact none of them reads.

    Returns ``(snapshot_hash, manifest_ref)`` because a Factor program binds both
    and the reader resolves a manifest by hash rather than by URI -- the two are
    handed back together so a caller cannot pair a hash with someone else's ref.
    """

    store = MarketDataRepository(workspace.workspace)
    opened = not history.recording()
    if opened:
        history.open_history("outcomes-fixture", "Outcome fixture", workspace.workspace)
    try:
        published = CausalExecutionOutcomePublisher(
            # Annotated ``PanelStateRepository`` on the publisher, but every method it
            # reaches for -- the research manifest, the source watermark, raw bars and
            # actions -- belongs to the market store. The annotation is stale; the
            # market repository is what the code actually requires.
            store=cast("PanelStateRepository", store),
            resolver=ArtifactResolver(workspace.artifact_root),
            artifact_root=workspace.artifact_root,
            mutation_gate=WorkspaceMutationGate(),
        ).publish_daily(
            panel_manifest_ref=workspace.panel_manifest_ref,
            completed_at=at,
            **({} if recipe_id is None else {"recipe_id": recipe_id}),
        )
    finally:
        if opened:
            history.close_history()
    return str(published.manifest.snapshot_hash), str(published.manifest_artifact.uri)
