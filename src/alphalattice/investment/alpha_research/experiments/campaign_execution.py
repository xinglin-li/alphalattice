"""Chronologically honest fold execution of the Alpha development Program.

Every outer fold runs one complete search on a chronological inner split of its
own training portion, selects one method there, refits that method on the whole
outer-training portion, and issues exactly one prediction on the outer
validation fold. Outer validation rows never reach hyperparameter selection or
LightGBM early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .campaign import (
    AlphaConditionalMethodPlan,
    AlphaTrialPlan,
)

type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AlphaCampaignPredictionSurface:
    trial_evidence_hash: str
    artifact_hash: str
    validation_row_ids: tuple[str, ...]
    predictions: FloatArray

    def __post_init__(self) -> None:
        if (
            len(self.validation_row_ids) != len(self.predictions)
            or self.predictions.flags.writeable
            or not np.isfinite(self.predictions).all()
        ):
            raise ValueError("ALPHA_CAMPAIGN_PREDICTION_SURFACE_INVALID")


type ExecutablePlan = AlphaTrialPlan | AlphaConditionalMethodPlan


def alpha_campaign_prediction_artifact_hash(
    *,
    program_hash: str,
    horizon_sessions: Literal[1, 5],
    trial_hash: str,
    fold_index: int,
    prediction_value_hash: str,
    validation_row_axis_hash: str,
) -> str:
    return str(
        canonical_hash(
            {
                "program_hash": program_hash,
                "horizon_sessions": horizon_sessions,
                "trial_hash": trial_hash,
                "fold_index": fold_index,
                "prediction_value_hash": prediction_value_hash,
                "validation_row_axis_hash": validation_row_axis_hash,
            }
        )
    )


__all__ = ["AlphaCampaignPredictionSurface", "alpha_campaign_prediction_artifact_hash"]
