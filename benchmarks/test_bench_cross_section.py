"""The robust cross-section numerics every Panel, Alpha target and Sector surface runs."""

from __future__ import annotations

import numpy as np
import pytest

from alphalattice.kernel.quant.cross_section import (
    equal_sector_demean,
    median_mad_winsor,
    robust_zscore,
)


@pytest.mark.benchmark
def test_median_mad_winsor(cross_section: np.ndarray) -> None:
    clipped, *_ = median_mad_winsor(cross_section)
    assert clipped.shape == cross_section.shape


@pytest.mark.benchmark
def test_robust_zscore(cross_section: np.ndarray) -> None:
    score, _, _ = robust_zscore(cross_section)
    assert score.shape == cross_section.shape


@pytest.mark.benchmark
@pytest.mark.parametrize("session_local", [False, True], ids=["historical", "session_local"])
def test_equal_sector_demean(
    cross_section: np.ndarray,
    sector_positions: tuple[np.ndarray, ...],
    session_local: bool,
) -> None:
    residual = equal_sector_demean(
        cross_section, sector_positions=sector_positions, session_local=session_local
    )
    assert residual.shape == cross_section.shape
