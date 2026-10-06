"""Typed boundary for Alpha target standardization methods.

`AlphaTargetPolicy` already binds the raw causal outcome, the ordered transform
sequence, winsorization, neutralization, and output semantics, so target
*identity* was never the gap. The gap was installation: the compiler chose the
lane standardization with a hard-coded branch, so a new standardization could not
be added without editing the shared compiler.

An adapter owns one standardization step and nothing else. Session admission,
coverage rules, sector sampling, causality, and evidence identity remain with the
Alpha Research Host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StandardizedTargetLane:
    """One standardized lane plus the dispersion the Host uses for admission."""

    values: FloatArray
    dispersion: FloatArray

    def __post_init__(self) -> None:
        """Require a two-dimensional target lane and one dispersion value per formation.

        Raises:
            ValueError: Target dimensionality or dispersion shape differs from the formation axis.
        """
        if self.values.ndim != 2 or self.dispersion.shape != (self.values.shape[0],):
            raise ValueError("ALPHA_TARGET_STANDARDIZED_LANE_INVALID")


class AlphaTargetStandardizationAdapter(Protocol):
    """Deterministic cross-sectional standardization installed by the Host."""

    standardization_id: str

    def standardize(self, values: FloatArray, *, mad_scale: float) -> StandardizedTargetLane:
        """Standardize declared target values through this installed adapter.

        Args:
            values: Formation-by-listing target values.
            mad_scale: Declared MAD scale for adapters that consume it.

        Returns:
            Standardized target lane with its formation dispersion.
        """
        ...


__all__ = [
    "AlphaTargetStandardizationAdapter",
    "StandardizedTargetLane",
]
