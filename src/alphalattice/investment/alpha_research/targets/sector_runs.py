"""The Sector map in force at each formation, as the target compilers read it (V346).

A target surface is one cross-section per formation session. Each run of formations that
reads one Sector map (`sector_positions`: one run while no reclassification falls inside the
window) is demeaned and floored by its own Sectors, so a listing is residualized against the
Sector it belonged to at that formation, and a surface with one run is the one it always was.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from alphalattice.kernel.quant.cross_section import equal_sector_demean

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type SectorRuns = tuple[tuple[slice, tuple[str, ...], tuple[npt.NDArray[np.int64], ...]], ...]
"""Per run of formations: its rows, its Sectors and each Sector's listing positions."""


def demean_by_run(values: FloatArray, runs: SectorRuns) -> FloatArray:
    """Each formation's values less its Sector's equal-weight mean, under its run's map.

    Args:
        values: One row per formation, one column per listing.
        runs: The formations' runs (`sector_positions`).

    Returns:
        The demeaned values, in the input's shape.
    """
    demeaned: FloatArray = np.empty_like(values, dtype=np.float64)
    for rows, _sectors, positions in runs:
        demeaned[rows] = cast(
            FloatArray, equal_sector_demean(values[rows], sector_positions=positions)
        )
    return demeaned


def below_sector_sample(finite: BoolArray, runs: SectorRuns, minimum: int) -> BoolArray:
    """Whether a formation has a Sector with fewer finite members than the floor.

    Args:
        finite: Which cells are finite, one row per formation.
        runs: The formations' runs (`sector_positions`).
        minimum: The floor.

    Returns:
        One flag per formation.
    """
    below: BoolArray = np.zeros(finite.shape[0], dtype=np.bool_)
    for rows, _sectors, positions in runs:
        counts = np.column_stack([finite[rows][:, members].sum(axis=1) for members in positions])
        below[rows] = np.any(counts < minimum, axis=1)
    return below


__all__ = ["SectorRuns", "below_sector_sample", "demean_by_run"]
