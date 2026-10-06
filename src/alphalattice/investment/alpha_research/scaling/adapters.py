"""The installed dispersion-forecast implementations.

One adapter per installed method, each naming its executable identity by content
rather than by class name. The arithmetic here is deliberately small -- a lagged
control does not compute much -- but the identity still matters: a scale that
silently changed formula would move every downstream return signal while every
recipe id stayed the same.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

from alphalattice.kernel.shared_kernel.source_identity import (
    switched_source_identity,
)

from .contracts import (
    ASYMMETRIC_EWMA_XS_DISPERSION_RECIPE_ID,
    EWMA_XS_DISPERSION_RECIPE_ID,
    HAR_XS_DISPERSION_RECIPE_ID,
    LAGGED_XS_DISPERSION_RECIPE_ID,
    CrossSectionalScalingError,
    LaggedXsDispersionRecipe,
    RecursiveXsDispersionRecipe,
    ScalingImplementationBinding,
    XsDispersionRecipe,
)

LAGGED_XS_DISPERSION_IMPLEMENTATION = "alpha_research.scaling.adapters.LaggedXsDispersionAdapter"
EWMA_XS_DISPERSION_IMPLEMENTATION = "alpha_research.scaling.adapters.EwmaXsDispersionAdapter"
ASYMMETRIC_EWMA_XS_DISPERSION_IMPLEMENTATION = (
    "alpha_research.scaling.adapters.AsymmetricEwmaXsDispersionAdapter"
)
HAR_XS_DISPERSION_IMPLEMENTATION = "alpha_research.scaling.adapters.HarXsDispersionAdapter"

_LAGGED_XS_DISPERSION_OWNERS = ("alphalattice.investment.alpha_research.scaling.adapters",)


@lru_cache(maxsize=1)
def _implementation_content_hash() -> str:
    """Hash the bytes of the module that selects the scale.

    Anchored on ``__file__``, never the process working directory, and hashed by
    content under a stable semantic id, so the value is the same from any
    checkout path and different after any edit.
    """

    return str(
        switched_source_identity(
            {"alpha_research.scaling.adapters": Path(__file__)},
            semantic_owner="alpha_research.scaling",
            numerical_role="CROSS_SECTIONAL_SCALE",
        )
    )


class CrossSectionalScaleAdapter(Protocol):
    """What every installed scale method must offer."""

    @property
    def recipe_id(self) -> str:
        """Read the dispersion recipe route owned by this adapter.

        Returns:
            Exact installed dispersion recipe identifier.
        """
        ...

    @property
    def implementation_id(self) -> str:
        """Read the result-deciding dispersion implementation identifier.

        Returns:
            Exact installed numerical implementation identifier.
        """
        ...

    def describe_implementation_binding(self) -> ScalingImplementationBinding:
        """Describe the source implementation closure deciding this scaling output.

        Returns:
            Sealed implementation owner/content binding for this adapter.
        """
        ...

    def select(
        self,
        *,
        recipe: XsDispersionRecipe,
        realized: tuple[float | None, ...],
        index: int,
    ) -> tuple[float | None, int | None]:
        """Return the scale for one formation and the index it came from."""
        ...


class LaggedXsDispersionAdapter:
    """The causal control: the realized scale from exactly ``T - L``.

    The whole method is one index arithmetic and two refusals, which is the
    point of a control. Both refusals matter: a formation earlier than the lag
    has no matured predecessor at all, and a predecessor whose own formation was
    unavailable has no scale to lend. Neither is filled in.
    """

    @property
    def recipe_id(self) -> str:
        """Read the dispersion recipe route owned by this adapter.

        Returns:
            Exact installed dispersion recipe identifier.
        """
        return LAGGED_XS_DISPERSION_RECIPE_ID

    @property
    def implementation_id(self) -> str:
        """Read the result-deciding dispersion implementation identifier.

        Returns:
            Exact installed numerical implementation identifier.
        """
        return LAGGED_XS_DISPERSION_IMPLEMENTATION

    def describe_implementation_binding(self) -> ScalingImplementationBinding:
        """Describe the source implementation closure deciding this scaling output.

        Returns:
            Sealed implementation owner/content binding for this adapter.
        """
        return ScalingImplementationBinding.create(
            implementation_id=LAGGED_XS_DISPERSION_IMPLEMENTATION,
            implementation_owners=_LAGGED_XS_DISPERSION_OWNERS,
            implementation_content_hash=_implementation_content_hash(),
        )

    def select(
        self,
        *,
        recipe: LaggedXsDispersionRecipe,
        realized: tuple[float | None, ...],
        index: int,
    ) -> tuple[float | None, int | None]:
        """Select the positive realized dispersion at the declared source offset.

        Args:
            recipe: Admitted dispersion recipe and its declared source lag.
            realized: Ordered historical dispersion observations with explicit unavailable cells.
            index: Formation position for which causal scale is requested.

        Returns:
            Selected scale and source index, or (None, None) when warmup/source scale is
            unavailable.

        Raises:
            CrossSectionalScalingError: The formation index is outside the observation sequence.
        """
        if index < 0 or index >= len(realized):
            raise CrossSectionalScalingError("SCALING_FORMATION_INDEX_OUT_OF_RANGE")
        source_index = index - recipe.source_offset_sessions
        if source_index < 0:
            return (None, None)
        value = realized[source_index]
        if value is None or not (value > 0.0):
            return (None, None)
        return (float(value), source_index)


class _RecursiveScaleAdapter:
    recipe_id: str
    implementation_id: str

    def describe_implementation_binding(self) -> ScalingImplementationBinding:
        return ScalingImplementationBinding.create(
            implementation_id=self.implementation_id,
            implementation_owners=_LAGGED_XS_DISPERSION_OWNERS,
            implementation_content_hash=_implementation_content_hash(),
        )

    @staticmethod
    def _causal_source(
        recipe: RecursiveXsDispersionRecipe,
        realized: tuple[float | None, ...],
        index: int,
    ) -> int | None:
        if index < 0 or index >= len(realized):
            raise CrossSectionalScalingError("SCALING_FORMATION_INDEX_OUT_OF_RANGE")
        source_index = index - recipe.source_offset_sessions
        return source_index if source_index >= 0 else None


class EwmaXsDispersionAdapter(_RecursiveScaleAdapter):
    """Select causal dispersion with the admitted symmetric EWMA half-life.

    Missing, nonfinite and nonpositive updates are ignored after first valid state initialization.
    """

    recipe_id = EWMA_XS_DISPERSION_RECIPE_ID
    implementation_id = EWMA_XS_DISPERSION_IMPLEMENTATION

    def select(
        self,
        *,
        recipe: XsDispersionRecipe,
        realized: tuple[float | None, ...],
        index: int,
    ) -> tuple[float | None, int | None]:
        """Select causal dispersion with the admitted symmetric EWMA half-life.

        Missing, nonfinite and nonpositive updates are ignored after first valid state
        initialization.

        Args:
            recipe: Admitted dispersion recipe and its declared source lag.
            realized: Ordered historical dispersion observations with explicit unavailable cells.
            index: Formation position for which causal scale is requested.

        Returns:
            Positive retained/forecast scale and causal source index, or (None, None) when
            unavailable.

        Raises:
            CrossSectionalScalingError: Recipe type/route or causal formation/source admission
                fails.
        """
        if (
            not isinstance(recipe, RecursiveXsDispersionRecipe)
            or recipe.recipe_id != self.recipe_id
        ):
            raise CrossSectionalScalingError("SCALING_RECIPE_ADAPTER_MISMATCH")
        source_index = self._causal_source(recipe, realized, index)
        if source_index is None:
            return (None, None)
        assert recipe.symmetric_half_life_sessions is not None
        decay = float(np.exp(np.log(0.5) / recipe.symmetric_half_life_sessions))
        state: float | None = None
        for raw in realized[: source_index + 1]:
            if raw is None or not np.isfinite(raw) or raw <= 0.0:
                continue
            state = float(raw) if state is None else decay * state + (1.0 - decay) * float(raw)
        return (state, source_index) if state is not None else (None, None)


class AsymmetricEwmaXsDispersionAdapter(_RecursiveScaleAdapter):
    """Select causal dispersion with distinct admitted rise and decay half-lives.

    Updates above the previous state use the rise decay; other valid updates use the slower decay.
    Missing/nonfinite/nonpositive updates carry state.
    """

    recipe_id = ASYMMETRIC_EWMA_XS_DISPERSION_RECIPE_ID
    implementation_id = ASYMMETRIC_EWMA_XS_DISPERSION_IMPLEMENTATION

    def select(
        self,
        *,
        recipe: XsDispersionRecipe,
        realized: tuple[float | None, ...],
        index: int,
    ) -> tuple[float | None, int | None]:
        """Select causal dispersion with distinct admitted rise and decay half-lives.

        Updates above the previous state use the rise decay; other valid updates use the slower
        decay. Missing/nonfinite/nonpositive updates carry state.

        Args:
            recipe: Admitted dispersion recipe and its declared source lag.
            realized: Ordered historical dispersion observations with explicit unavailable cells.
            index: Formation position for which causal scale is requested.

        Returns:
            Positive retained/forecast scale and causal source index, or (None, None) when
            unavailable.

        Raises:
            CrossSectionalScalingError: Recipe type/route or causal formation/source admission
                fails.
        """
        if (
            not isinstance(recipe, RecursiveXsDispersionRecipe)
            or recipe.recipe_id != self.recipe_id
        ):
            raise CrossSectionalScalingError("SCALING_RECIPE_ADAPTER_MISMATCH")
        source_index = self._causal_source(recipe, realized, index)
        if source_index is None:
            return (None, None)
        assert recipe.rise_half_life_sessions is not None
        assert recipe.decay_half_life_sessions is not None
        up = float(np.exp(np.log(0.5) / recipe.rise_half_life_sessions))
        down = float(np.exp(np.log(0.5) / recipe.decay_half_life_sessions))
        state: float | None = None
        for raw in realized[: source_index + 1]:
            if raw is None or not np.isfinite(raw) or raw <= 0.0:
                continue
            if state is None:
                state = float(raw)
            else:
                decay = up if float(raw) > state else down
                state = decay * state + (1.0 - decay) * float(raw)
        return (state, source_index) if state is not None else (None, None)


class HarXsDispersionAdapter(_RecursiveScaleAdapter):
    """Forecast causal dispersion from daily, weekly and monthly log-scale history.

    The HAR fit uses an unpenalized intercept, RMS-normalized regressors and the declared ridge
    ratio. Insufficient warmup/fit rows or an unavailable final window produces no forecast.
    """

    recipe_id = HAR_XS_DISPERSION_RECIPE_ID
    implementation_id = HAR_XS_DISPERSION_IMPLEMENTATION

    def select(
        self,
        *,
        recipe: XsDispersionRecipe,
        realized: tuple[float | None, ...],
        index: int,
    ) -> tuple[float | None, int | None]:
        """Forecast causal dispersion from daily, weekly and monthly log-scale history.

        The HAR fit uses an unpenalized intercept, RMS-normalized regressors and the declared ridge
        ratio. Insufficient warmup/fit rows or an unavailable final window produces no forecast.

        Args:
            recipe: Admitted dispersion recipe and its declared source lag.
            realized: Ordered historical dispersion observations with explicit unavailable cells.
            index: Formation position for which causal scale is requested.

        Returns:
            Positive retained/forecast scale and causal source index, or (None, None) when
            unavailable.

        Raises:
            CrossSectionalScalingError: Recipe type/route or causal formation/source admission
                fails.
        """
        if (
            not isinstance(recipe, RecursiveXsDispersionRecipe)
            or recipe.recipe_id != self.recipe_id
        ):
            raise CrossSectionalScalingError("SCALING_RECIPE_ADAPTER_MISMATCH")
        source_index = self._causal_source(recipe, realized, index)
        if source_index is None or source_index < 84:
            return (None, None)
        values: np.ndarray = np.asarray(
            [np.nan if value is None else value for value in realized[: source_index + 1]],
            dtype=np.float64,
        )
        log_values = np.log(values)
        rows: list[list[float]] = []
        targets: list[float] = []
        for target_index in range(21, len(log_values)):
            window = log_values[target_index - 21 : target_index]
            if np.isfinite(log_values[target_index]) and np.isfinite(window).all():
                rows.append(
                    [1.0, float(window[-1]), float(np.mean(window[-5:])), float(np.mean(window))]
                )
                targets.append(float(log_values[target_index]))
        assert recipe.har_minimum_fit_rows is not None
        if len(rows) < recipe.har_minimum_fit_rows:
            return (None, None)
        matrix: np.ndarray = np.asarray(rows, dtype=np.float64)
        response: np.ndarray = np.asarray(targets, dtype=np.float64)
        scale = np.maximum(np.sqrt(np.mean(np.square(matrix[:, 1:]), axis=0)), 1e-12)
        normalized = matrix.copy()
        normalized[:, 1:] /= scale
        gram = normalized.T @ normalized / len(normalized)
        assert recipe.har_ridge_ratio is not None
        penalty = np.diag(
            [0.0, recipe.har_ridge_ratio, recipe.har_ridge_ratio, recipe.har_ridge_ratio]
        )
        coefficients = np.linalg.solve(gram + penalty, normalized.T @ response / len(normalized))
        last = log_values[-21:]
        if not np.isfinite(last).all():
            return (None, None)
        features = np.asarray(
            [1.0, last[-1] / scale[0], np.mean(last[-5:]) / scale[1], np.mean(last) / scale[2]]
        )
        forecast = float(np.exp(features @ coefficients))
        if not np.isfinite(forecast) or forecast <= 0.0:
            return (None, None)
        return (forecast, source_index)


__all__ = [
    "ASYMMETRIC_EWMA_XS_DISPERSION_IMPLEMENTATION",
    "EWMA_XS_DISPERSION_IMPLEMENTATION",
    "HAR_XS_DISPERSION_IMPLEMENTATION",
    "LAGGED_XS_DISPERSION_IMPLEMENTATION",
    "AsymmetricEwmaXsDispersionAdapter",
    "CrossSectionalScaleAdapter",
    "EwmaXsDispersionAdapter",
    "HarXsDispersionAdapter",
    "LaggedXsDispersionAdapter",
]
