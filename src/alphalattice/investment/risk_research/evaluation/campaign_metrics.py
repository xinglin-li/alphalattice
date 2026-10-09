"""Dense covariance matrices read back from Risk's packed chunks.

The paired route's covariance lanes (``portfolio_strategy_lab/campaign/authority.py``)
read the matrices a development build published without re-running an estimator.
``evaluation/formation.py`` owns every per-formation diagnostic.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from alphalattice.investment.risk_research.contracts import HistoricalCovarianceChunk
from alphalattice.investment.risk_research.estimators.covariance import (
    RiskNumericalError,
)
from alphalattice.investment.risk_research.surfaces.artifacts import RiskArtifactStore

type FloatArray = NDArray[np.float64]


def unpack_chunk_matrices(
    *, store: RiskArtifactStore, chunk: HistoricalCovarianceChunk
) -> tuple[FloatArray, ...]:
    """Rebuild the dense symmetric matrices a packed chunk stores.

    ``chunk_path`` verifies the payload against the chunk's own byte digest
    before returning, so a tampered or truncated chunk fails here rather than
    producing plausible numbers.
    """
    target = store.chunk_path(chunk)
    content = target.read_bytes()
    if hashlib.sha256(content).hexdigest() != chunk.packed_bytes_sha256:
        raise RiskNumericalError("risk_research.campaign_chunk_tampered")
    packed: FloatArray = np.frombuffer(content, dtype="<f8").reshape(
        chunk.matrix_count, chunk.packed_value_count // chunk.matrix_count
    )
    lower = np.tril_indices(chunk.asset_count)
    matrices: list[FloatArray] = []
    for row in range(chunk.matrix_count):
        matrix: FloatArray = np.zeros((chunk.asset_count, chunk.asset_count), dtype=np.float64)
        # Both triangles are written by assignment rather than rebuilt as
        # ``matrix + matrix.T``. That addition is numerically identical and
        # *byte*-lossy: a packed ``-0.0`` becomes ``+0.0`` on the mirrored side,
        # because ``0.0 + -0.0`` is ``+0.0``. The values compare equal and the
        # content hash does not, so a matrix carrying negative zeros -- which a
        # shrunk correlation produces routinely once an off-diagonal reaches
        # exactly zero -- would fail its own identity check on readback.
        # Assignment copies the bit pattern.
        matrix[lower] = packed[row]
        matrix[(lower[1], lower[0])] = packed[row]
        matrix.setflags(write=False)
        matrices.append(matrix)
    return tuple(matrices)


__all__ = ["unpack_chunk_matrices"]
