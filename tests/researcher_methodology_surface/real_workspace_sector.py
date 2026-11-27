"""Read the Sector revision written by the workspace fixture."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver

if TYPE_CHECKING:
    from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace


def _sector_revision(workspace: RealRiskWorkspace) -> str:
    from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
        PanelClosureArtifactStore,
    )
    from alphalattice.foundation.feature_engine.panels.closure_contracts import SectorRevisionMap

    store = PanelClosureArtifactStore(ArtifactResolver(workspace.artifact_root))
    maps = store.find_models(
        category="sector-maps", model=SectorRevisionMap, matches=lambda _: True
    )
    assert maps, "the shared workspace publishes a sector revision map"
    return maps[0].sector_revision
