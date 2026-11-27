"""Canonical float-matrix content identity without numerical-engine imports."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from alphalattice.kernel.identity import identity_blake3, tagged_blake3

type FloatArray = NDArray[np.float64]


def matrix_content_hash(matrix: FloatArray) -> str:
    """Hash covariance shape and canonical contiguous little-endian float64 bytes.

    Args:
        matrix: Numerical array converted to little-endian float64 in C order.

    Returns:
        Tagged BLAKE3 binding rank, every dimension length and the canonical matrix bytes.
    """
    canonical = np.ascontiguousarray(matrix, dtype="<f8")
    digest = identity_blake3()
    digest.update(canonical.ndim.to_bytes(1, "big"))
    for dimension in canonical.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(canonical.tobytes(order="C"))
    return tagged_blake3(digest)


__all__ = ["matrix_content_hash"]
