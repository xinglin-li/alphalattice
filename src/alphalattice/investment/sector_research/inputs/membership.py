"""Sector membership resolution, read by the Desk that owns Sector state.

Alpha keeps its own reader for its own residualization; this one is not a shim
around it -- importing Alpha from here would close the ``alpha_research ->
sector_research`` edge Gate C created into a cycle. Both Desks read the same
Panel closure store, which is the single writer of sector revisions, so the two
readers cannot disagree about content, only about who is asking.
"""

from __future__ import annotations

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    sector_history_as_of,
)
from alphalattice.kernel.quant.sector_history import SectorHistory

from ..contracts import SectorResearchError


def load_sector_membership(*, resolver: ArtifactResolver, sector_revision: str) -> SectorHistory:
    """Resolve one published sector revision to the Sector each session reads (V346).

    Exactly one revision map may match. Zero is an unpublished handle; more than
    one would mean the revision identity is not an identity, and picking either
    by directory order would let filesystem enumeration decide membership. The
    history adds the reclassifications its activations recorded, each from its
    effective session; as a mapping it is the revision's own classification.

    Args:
        resolver: Artifact resolver owning the Panel closure maps and activation receipts.
        sector_revision: Exact requested classification revision.

    Returns:
        The revision's classification with the reclassifications in force.

    Raises:
        SectorResearchError: No single map matches the revision, or its activations cannot
            be read back.
    """
    try:
        return sector_history_as_of(PanelClosureArtifactStore(resolver), sector_revision)
    except ValueError as error:
        raise SectorResearchError("sector_research.sector_membership_unavailable") from error


__all__ = ["load_sector_membership"]
