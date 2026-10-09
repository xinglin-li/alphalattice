"""Produce the installed factor-plus-idiosyncratic Risk surface.

This is the sole numerical owner for the public Risk representation.  It is a
direct, identity-bound rederivation of the implementation promoted from Batch
108; it neither imports the experiment worktree nor materialises an ``N x N``
covariance.  Foundation supplies the classification map and the existing Risk
return surface supplies the causal open-to-open history.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
import numpy.typing as npt

from alphalattice.foundation.feature_engine.panels.closure_contracts import SectorRevisionMap
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .decomposition import (
    INSTALLED_RISK_DECOMPOSITION_RECIPE,
    FactorIdiosyncraticRiskSurface,
    RiskAllocationProjection,
    RiskAttributionProjection,
    RiskDecompositionError,
    RiskDecompositionRecipe,
)

type FloatArray = npt.NDArray[np.float64]

RISK_PRODUCER_ID: Final = "SECTOR_DEGARCH_504_EXACT_CLOSURE"


def _array_hash(values: npt.NDArray[np.generic], *, dtype: npt.DTypeLike) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def _require_hash(value: str, *, code: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RiskDecompositionError(code)


@dataclass(frozen=True, slots=True)
class RiskDecompositionInputs:
    """One formation's causal inputs, already resolved by the composition owner."""

    formation_session: date
    history_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    open_to_open_log_returns: FloatArray
    classification: SectorRevisionMap
    return_surface_hash: str
    formation_sectors: Mapping[str, str] | None = None
    """Each listing whose Sector at this formation is not its current one, with that Sector's
    name; None while every listing reads its current Sector."""

    def validate(self, recipe: RiskDecompositionRecipe) -> None:
        """Require complete causal history, aligned inputs and classification coverage.

        Args:
            recipe: Recipe determining initialization and fitting history requirements.

        Raises:
            RiskDecompositionError: Listing/history axes, causal ordering, shape, infinity, source
                identity or classification coverage violate admission.
        """
        names = len(self.ordered_listing_ids)
        required = (
            recipe.conditional_volatility_initialization_sessions + recipe.factor_fit_sessions
        )
        if (
            names == 0
            or len(set(self.ordered_listing_ids)) != names
            or len(self.history_sessions) < required
            or self.history_sessions != tuple(sorted(set(self.history_sessions)))
            or self.history_sessions[-1] >= self.formation_session
            or self.open_to_open_log_returns.shape != (len(self.history_sessions), names)
            or np.isinf(self.open_to_open_log_returns).any()
        ):
            raise RiskDecompositionError("risk_research.producer_input_axis_invalid")
        _require_hash(
            self.return_surface_hash,
            code="risk_research.producer_return_surface_identity_invalid",
        )
        entries = {entry.listing_id: entry for entry in self.classification.entries}
        if any(listing_id not in entries for listing_id in self.ordered_listing_ids):
            raise RiskDecompositionError("risk_research.producer_classification_incomplete")
        if self.formation_sectors is not None and (
            not self.formation_sectors
            or set(self.formation_sectors) - set(self.ordered_listing_ids)
        ):
            raise RiskDecompositionError("risk_research.producer_formation_sectors_invalid")


@dataclass(frozen=True, slots=True)
class ProducedRiskSurface:
    """The one surface and its deliberately separated consumer projections."""

    surface: FactorIdiosyncraticRiskSurface
    allocation: RiskAllocationProjection
    attribution: RiskAttributionProjection


def _ewma_clock(
    returns: FloatArray,
    *,
    decay: float,
    initialization_sessions: int,
    variance_floor: float,
) -> tuple[FloatArray, FloatArray]:
    """Return historical sigma and the next-session sigma, strictly causal.

    Missing returns contribute a zero standardized residual and hold the EWMA
    state.  That is the exact promoted implementation rule, not a generic
    imputation choice.
    """

    clean = np.nan_to_num(returns, nan=0.0)
    seen = np.isfinite(returns)
    with np.errstate(invalid="ignore"):
        state = np.nanvar(
            np.where(seen[:initialization_sessions], returns[:initialization_sessions], np.nan),
            axis=0,
            ddof=1,
        )
    state = np.where(np.isfinite(state), state, 1e-8)
    history = np.full(returns.shape, np.nan, dtype=np.float64)
    for index in range(initialization_sessions, returns.shape[0]):
        history[index] = np.sqrt(np.maximum(state, variance_floor))
        step = np.square(clean[index])
        state = np.where(seen[index], decay * state + (1.0 - decay) * step, state)
    today = np.sqrt(np.maximum(state, variance_floor))
    return history, np.asarray(today, dtype=np.float64)


def _classification_design(
    *,
    ordered_listing_ids: tuple[str, ...],
    classification: SectorRevisionMap,
    formation_sectors: Mapping[str, str] | None = None,
) -> tuple[tuple[str, ...], FloatArray]:
    entries = {entry.listing_id: entry for entry in classification.entries}
    # A reclassified listing's earlier Sector, by its key where the map names one.
    keys = {entry.sector_name: entry.sector_key or entry.sector_name for entry in entries.values()}
    moved = formation_sectors or {}
    factor_by_listing = tuple(
        keys.get(moved[listing_id], moved[listing_id])
        if listing_id in moved
        else entries[listing_id].sector_key or entries[listing_id].sector_name
        for listing_id in ordered_listing_ids
    )
    factor_ids = tuple(sorted(set(factor_by_listing)))
    positions = {factor_id: index for index, factor_id in enumerate(factor_ids)}
    design: FloatArray = np.zeros((len(ordered_listing_ids), len(factor_ids)), dtype=np.float64)
    factor_positions = [positions[value] for value in factor_by_listing]
    design[np.arange(len(ordered_listing_ids)), factor_positions] = 1.0
    return factor_ids, design


class RiskSurfaceProducer:
    """Execute the installed recipe and emit both public projections."""

    def __init__(
        self,
        recipe: RiskDecompositionRecipe = INSTALLED_RISK_DECOMPOSITION_RECIPE,
    ) -> None:
        """Bind decomposition production to one declared frozen recipe.

        Args:
            recipe: Installed decomposition recipe by default, retained for subsequent production.
        """
        self.recipe = recipe

    def produce(self, inputs: RiskDecompositionInputs) -> ProducedRiskSurface:
        """Produce direct allocation volatility and factor/idiosyncratic attribution evidence.

        The producer uses causal EWMA state, standardized clipped/demeaned returns, equal-weight
        sector factor returns, sample factor covariance and floored specific variance. Producer
        identity binds raw input/source/classification and numerical recipe; observed numerical
        environment is excluded from producer identity.

        Args:
            inputs: Bound causal history and complete classification on the ordered listing axis.

        Returns:
            Produced surface with direct-volatility allocation and classification-bound attribution
            projections.

        Raises:
            RiskDecompositionError: Input admission fails or finite conditional volatility cannot be
                produced.
        """
        inputs.validate(self.recipe)
        required = (
            self.recipe.conditional_volatility_initialization_sessions
            + self.recipe.factor_fit_sessions
        )
        sessions = inputs.history_sessions[-required:]
        returns = np.asarray(inputs.open_to_open_log_returns[-required:], dtype=np.float64)
        history_volatility, today_volatility = _ewma_clock(
            returns,
            decay=self.recipe.conditional_volatility_decay,
            initialization_sessions=self.recipe.conditional_volatility_initialization_sessions,
            variance_floor=self.recipe.variance_floor,
        )
        fit_rows = self.recipe.factor_fit_sessions
        block = returns[-fit_rows:]
        volatility = history_volatility[-fit_rows:]
        if not np.isfinite(volatility).all():
            raise RiskDecompositionError("risk_research.producer_volatility_history_incomplete")
        standardized = np.nan_to_num(
            block / np.maximum(volatility, 1e-12),
            nan=0.0,
        )
        standardized = np.minimum(
            np.maximum(standardized, -self.recipe.residual_clip_bound),
            self.recipe.residual_clip_bound,
        )
        standardized = standardized - standardized.mean(axis=0)

        factor_ids, exposures = _classification_design(
            ordered_listing_ids=inputs.ordered_listing_ids,
            classification=inputs.classification,
            formation_sectors=inputs.formation_sectors,
        )
        counts = exposures.sum(axis=0)
        if np.any(counts <= 0.0):
            raise RiskDecompositionError("risk_research.producer_factor_empty")
        factor_returns = (standardized @ exposures) / counts
        specific_returns = standardized - factor_returns @ exposures.T
        factor_covariance = np.atleast_2d(np.cov(factor_returns, rowvar=False, ddof=1)).astype(
            np.float64, copy=False
        )
        denominator = max(fit_rows - len(factor_ids), 1)
        idiosyncratic_variance = np.maximum(
            np.square(specific_returns).sum(axis=0) / denominator,
            self.recipe.variance_floor,
        )
        producer_identity = canonical_hash(
            {
                "kind": "RiskSurfaceProducer",
                "producer_id": RISK_PRODUCER_ID,
                "recipe_hash": self.recipe.recipe_hash,
                "formation_session": inputs.formation_session.isoformat(),
                "history_first": sessions[0].isoformat(),
                "history_last": sessions[-1].isoformat(),
                "ordered_listing_ids": list(inputs.ordered_listing_ids),
                "return_values_hash": _array_hash(returns, dtype="<f8"),
                "return_surface_hash": inputs.return_surface_hash,
                "sector_map_hash": inputs.classification.map_hash,
                "sector_revision": inputs.classification.sector_revision,
                "manifest_revision": inputs.classification.manifest_revision,
                **(
                    {"formation_sectors": dict(sorted(inputs.formation_sectors.items()))}
                    if inputs.formation_sectors is not None
                    else {}
                ),
                # The stack it ran on is the Portfolio run's recorded environment,
                # never part of what it produced (LAWS.md ID6).
                "float": "IEEE754_FLOAT64",
            }
        )
        surface = FactorIdiosyncraticRiskSurface.create(
            recipe=self.recipe,
            formation_session=inputs.formation_session,
            ordered_listing_ids=inputs.ordered_listing_ids,
            ordered_factor_ids=factor_ids,
            exposures=exposures,
            factor_covariance=factor_covariance,
            idiosyncratic_variance=np.asarray(idiosyncratic_variance, dtype=np.float64),
            conditional_volatility=today_volatility,
            producer_identity=producer_identity,
        )
        return ProducedRiskSurface(
            surface=surface,
            allocation=RiskAllocationProjection.of(surface),
            attribution=RiskAttributionProjection.of(
                surface,
                classification_authority=(
                    "FOUNDATION_SECTOR_REVISION_MAP:"
                    f"{inputs.classification.sector_revision}:"
                    f"{inputs.classification.manifest_revision}"
                ),
            ),
        )


__all__ = [
    "RISK_PRODUCER_ID",
    "ProducedRiskSurface",
    "RiskDecompositionInputs",
    "RiskSurfaceProducer",
]
