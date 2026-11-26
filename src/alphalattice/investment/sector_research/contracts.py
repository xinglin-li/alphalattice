"""Shared Sector Research failure type and the one array-identity convention.

Every Sector Research refusal is a ``SectorResearchError`` carrying a stable
``sector_research.*`` code, so a caller can branch on what happened without
parsing prose. Absence is never one of these: a handle naming evidence that was
never published surfaces as ``FileNotFoundError`` from the store, because
"missing" and "broken" are different answers and collapsing them is how a
corrupt artifact gets reported as merely waiting.
"""

from __future__ import annotations

from hashlib import sha256

import numpy as np
import numpy.typing as npt

from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]


class SectorResearchError(ValueError):
    """Stable typed refusal raised before any Sector Research evidence is written."""


def sector_array_identity(values: FloatArray) -> str:
    """dtype, shape and content as one inseparable claim about an array.

    Bytes alone are ambiguous -- the same buffer read at a different dtype or
    reshaped is a different array with the same digest -- so all three travel
    together. This is the same convention the canonical Alpha target's lane
    identity uses; sharing the convention is what lets a consumer compare a
    Sector identity against an Alpha one without a translation step.
    """
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return str(
        canonical_hash(
            {
                "dtype": str(contiguous.dtype),
                "shape": [int(value) for value in contiguous.shape],
                "content": sha256(contiguous.tobytes()).hexdigest(),
            }
        )
    )


def grid_to_matrix(values: tuple[tuple[float | None, ...], ...]) -> FloatArray:
    """Rebuild the float64 matrix a durable value grid describes, ``NaN`` for absent.

    The durable form stores ``None`` where a cell is unavailable because strict
    JSON cannot carry ``NaN``; the identity hash is always computed over the
    rebuilt matrix so the in-memory and durable representations cannot drift.
    Python's JSON round-trips ``repr(float)`` exactly, so the rebuilt matrix is
    bitwise the one that was sealed.
    """
    return np.asarray(
        [[np.nan if value is None else float(value) for value in row] for row in values],
        dtype=np.float64,
    ).reshape(len(values), len(values[0]) if values else 0)


__all__ = [
    "FloatArray",
    "SectorResearchError",
    "grid_to_matrix",
    "sector_array_identity",
]
