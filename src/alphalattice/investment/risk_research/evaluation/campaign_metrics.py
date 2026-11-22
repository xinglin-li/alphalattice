"""Dense covariance matrices read back from Risk's packed chunks.

The paired route's covariance lanes (``portfolio_strategy_lab/campaign/authority.py``)
read the matrices a development build published without re-running an estimator.
``evaluation/formation.py`` owns every per-formation diagnostic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from alphalattice.investment.risk_research.contracts import HistoricalCovarianceChunk
from alphalattice.investment.risk_research.surfaces.artifacts import (
    HistoricalCovarianceReader,
    RiskArtifactStore,
)

type FloatArray = NDArray[np.float64]


def unpack_chunk_matrices(
    *,
    store: RiskArtifactStore,
    chunk: HistoricalCovarianceChunk,
    ordered_listing_ids: tuple[str, ...] | None = None,
) -> tuple[FloatArray, ...]:
    """Rebuild the verified covariance matrices in their declared formation order."""
    reader = HistoricalCovarianceReader(store.root.parent)
    with reader.lease(chunk, ordered_listing_ids=ordered_listing_ids) as lease:
        return tuple(lease.matrix(session) for session in chunk.formation_sessions)


__all__ = ["unpack_chunk_matrices"]
