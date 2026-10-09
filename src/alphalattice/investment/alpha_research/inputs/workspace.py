"""The Sector history an Alpha study reads, from the Panel closure store."""

from __future__ import annotations

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    sector_history_as_of,
)
from alphalattice.kernel.quant.sector_history import SectorHistory


def load_sector_history(*, resolver: ArtifactResolver, sector_revision: str) -> SectorHistory:
    """The Sector each session reads as of one published revision.

    Args:
        resolver: The workspace's artifacts.
        sector_revision: The revision the Panel's lineage names.

    Returns:
        The history: the revision's map, then the reclassifications in force.

    Raises:
        ValueError: `alpha_research.current_sector_map_unavailable` when the revision has no
            single map or its activations cannot be read back.
    """
    try:
        return sector_history_as_of(PanelClosureArtifactStore(resolver), sector_revision)
    except ValueError as error:
        raise ValueError("alpha_research.current_sector_map_unavailable") from error


__all__ = [
    "load_sector_history",
]
