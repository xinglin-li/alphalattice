"""``DISTRIBUTED_LAG_ELASTIC_NET``: regularized own and cross-sector lags.

The first challenger that can express a lead-lag relationship between sectors
rather than only a sector's own persistence. Each sector gets its own regression
of the next-session Sector mean constituent log return on the last `lag_count`
sessions of *every* sector, so the univariate term (a sector's own lags) and the
cross-series terms (other sectors' lags) are the same design matrix and the
penalty decides which survive.

Formula, over the matured training matrix the Host binds (rows are formation
sessions ascending, columns are the ordered sectors)::

    row i features = concat(T[i-1], T[i-2], ..., T[i-lag_count])   -> lag_count * S
    row i target   = T[i, s]                                        for sector s
    estimator      = ElasticNet(alpha, l1_ratio, fit_intercept=True)
    forecast_s     = estimator_s.predict(concat(T[-1], ..., T[-lag_count]))

Units: natural log return per session, same as the target, because the features
and the target are the same quantity at different times and the estimator is
linear with an intercept.

The regularization strength is expressed relative to the data rather than as an
absolute constant, which is what makes one frozen parameter point meaningful
across training windows of different length::

    alpha_max = max |X^T y| / (n_samples * l1_ratio)     -- the smallest penalty
                                                            that zeroes every
                                                            coefficient
    alpha     = alpha_max * alpha_max_multiplier_basis_points / 10_000

Both knobs are integers because a recipe's parameters are integers: the
multiplier is in basis points of `alpha_max` (1000 -> 0.10) and the mixing
weight is a percentage (50 -> an even L1/L2 split). Scaled integers keep the
declared identity exactly as wide as the method space, which a float bag would
not.

Features are deliberately not standardized. Every column is a one-session log
return of a sector, so the design matrix is already on one common scale, and a
fitted scaler would be a training-only preprocessing recipe this Desk would then
have to declare, publish and replay for no numerical benefit.

Finite-input policy: a training row is used only when its target and all of its
lagged features are finite, and the two failure modes stay distinct. Too few
usable rows is the declared warmup refusal. A sector whose most recent
`lag_count` sessions are not fully finite cannot be predicted at all at this
formation and refuses the same way, because substituting a zero for a missing
lag would be inventing an observation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNet

from ..contracts import FloatArray, SectorResearchError
from .contracts import (
    INSUFFICIENT_MATURED_HISTORY,
    SECTOR_SCIKIT_LEARN_STACK,
    BoundSectorForecastInput,
    SectorForecastRecipe,
    SectorForecastValues,
    SectorNumericalBinding,
    sector_numerical_thread_policy,
)
from .zero import SECTOR_FORECAST_CONTENT_FORMAT_ID

DISTRIBUTED_LAG_ELASTIC_NET_METHOD_ID = "DISTRIBUTED_LAG_ELASTIC_NET"
DISTRIBUTED_LAG_ELASTIC_NET_SCHEMA_ID = "sector-forecast/distributed-lag-elastic-net@1"

_PARAMETER_FIELDS = (
    "alpha_max_multiplier_basis_points",
    "forecast_horizon_sessions",
    "l1_ratio_percent",
    "lag_count",
    "minimum_history_sessions",
    "refit_every_sessions",
)

_MAX_ITERATIONS = 10_000
_TOLERANCE = 1e-8


def _lagged_design(values: FloatArray, *, lag_count: int) -> tuple[FloatArray, FloatArray]:
    """Rows of stacked lags and the aligned target rows, oldest first.

    Row ``i`` of the design carries sessions ``i-1 .. i-lag_count`` in that
    order, so a coefficient's position states which sector at which lag it
    weights. Returned as (design, targets) over the rows a target exists for.
    """

    session_count, sector_count = values.shape
    usable = session_count - lag_count
    if usable <= 0:
        return (
            np.zeros((0, lag_count * sector_count), dtype=np.float64),
            np.zeros((0, sector_count), dtype=np.float64),
        )
    design = np.empty((usable, lag_count * sector_count), dtype=np.float64)
    for lag in range(1, lag_count + 1):
        start = lag_count - lag
        block = values[start : start + usable, :]
        column = (lag - 1) * sector_count
        design[:, column : column + sector_count] = block
    return design, values[lag_count:, :]


class DistributedLagElasticNetAdapter:
    """Per-sector Elastic Net over own and cross-sector lags of the clean target."""

    method_id = DISTRIBUTED_LAG_ELASTIC_NET_METHOD_ID
    recipe_schema_id = DISTRIBUTED_LAG_ELASTIC_NET_SCHEMA_ID

    def describe_numerical_binding(self) -> SectorNumericalBinding:
        """Seal this installed adapter source-rule identity and deterministic numerical policy.

        Returns:
            Exact method/content-format, implementation, stack and arithmetic/warmup binding.
        """
        return SectorNumericalBinding.create(
            method_id=self.method_id,
            forecast_content_format_id=SECTOR_FORECAST_CONTENT_FORMAT_ID,
            implementation_sources={
                "sector_research.models.distributed_lag_elastic_net": Path(__file__),
                "sector_research.models.contracts": Path(__file__).with_name("contracts.py"),
            },
            deterministic_policy={
                "estimator": "sklearn.linear_model.ElasticNet",
                "selection": "cyclic",
                "fit_intercept": "True",
                "max_iter": str(_MAX_ITERATIONS),
                "tolerance": str(_TOLERANCE),
                "penalty_scale": "ALPHA_MAX_RELATIVE_BASIS_POINTS",
                "standardization": "NONE_COMMON_LOG_RETURN_SCALE",
                "warmup": "EXPLICIT_REFUSAL_BELOW_MINIMUM_HISTORY",
            },
            numerical_stack=SECTOR_SCIKIT_LEARN_STACK,
        )

    def validate_recipe(self, recipe: SectorForecastRecipe) -> None:
        """Require this installed method/schema route and its declared parameter shape.

        Positive declared parameters, 1-100 L1 ratio percentage and minimum history greater than lag
        depth are required.

        Args:
            recipe: Sealed recipe to validate before forecasting.

        Raises:
            SectorResearchError: Method/schema routing or the declared parameter rules fail.
        """
        if recipe.method_id != self.method_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise SectorResearchError("sector_research.recipe_route_invalid")
        if tuple(sorted(recipe.parameters)) != _PARAMETER_FIELDS or any(
            recipe.parameters[field] < 1 for field in _PARAMETER_FIELDS
        ):
            raise SectorResearchError("sector_research.recipe_parameters_invalid")
        if not 1 <= recipe.parameters["l1_ratio_percent"] <= 100:
            # l1_ratio must stay inside (0, 1]: sklearn's Elastic Net refuses a
            # pure ridge at 0 through this estimator, and above 1 is not a mix.
            raise SectorResearchError("sector_research.recipe_parameters_invalid")
        if recipe.parameters["minimum_history_sessions"] <= recipe.parameters["lag_count"]:
            # A minimum at or below the lag depth admits a fit with no rows.
            raise SectorResearchError("sector_research.recipe_parameters_invalid")

    def forecast(
        self,
        *,
        bound_input: BoundSectorForecastInput,
        recipe: SectorForecastRecipe,
    ) -> SectorForecastValues:
        """Fit lagged Elastic Net on complete matured rows and forecast each sector.

        Unavailable cells mark insufficient usable fit history or an unusable latest predictor. Fits
        use the declared single-thread scikit-learn policy and fixed alpha-max-relative penalty.

        Args:
            bound_input: Admitted matured observations and formation/sector axes.
            recipe: Exact method recipe checked before numerical work.

        Returns:
            Sector-axis-aligned values and per-cell unavailability reasons.

        Raises:
            SectorResearchError: Recipe routing/parameters or resulting forecast cells are invalid.
        """
        self.validate_recipe(recipe)
        lag_count = int(recipe.parameters["lag_count"])
        minimum_history = int(recipe.parameters["minimum_history_sessions"])
        l1_ratio = float(recipe.parameters["l1_ratio_percent"]) / 100.0
        multiplier = float(recipe.parameters["alpha_max_multiplier_basis_points"]) / 10_000.0

        observations = bound_input.training_values
        sector_count = len(bound_input.ordered_sectors)
        design, targets = _lagged_design(observations, lag_count=lag_count)
        complete_rows = np.isfinite(design).all(axis=1)

        # The prediction row is the most recent `lag_count` sessions, in the
        # same lag order the design uses.
        latest = np.concatenate([observations[-lag, :] for lag in range(1, lag_count + 1)]).reshape(
            1, lag_count * sector_count
        )
        latest_usable = bool(np.isfinite(latest).all())

        values: list[float | None] = []
        reasons: list[str | None] = []
        with sector_numerical_thread_policy():
            for column in range(sector_count):
                usable = complete_rows & np.isfinite(targets[:, column])
                if int(usable.sum()) < minimum_history or not latest_usable:
                    values.append(None)
                    reasons.append(INSUFFICIENT_MATURED_HISTORY)
                    continue
                features = np.ascontiguousarray(design[usable, :])
                response = np.ascontiguousarray(targets[usable, column])
                estimator = ElasticNet(
                    alpha=_alpha_for(features, response, l1_ratio=l1_ratio) * multiplier,
                    l1_ratio=l1_ratio,
                    fit_intercept=True,
                    max_iter=_MAX_ITERATIONS,
                    tol=_TOLERANCE,
                    selection="cyclic",
                )
                estimator.fit(features, response)
                predicted = float(estimator.predict(latest)[0])
                if not np.isfinite(predicted):
                    raise SectorResearchError("sector_research.forecast_values_nonfinite")
                values.append(predicted)
                reasons.append(None)
        return SectorForecastValues(
            method_id=self.method_id,
            forecast_formation_at=bound_input.forecast_formation_at,
            ordered_sectors=bound_input.ordered_sectors,
            values=tuple(values),
            unavailable_reasons=tuple(reasons),
        )


def _alpha_for(features: FloatArray, response: FloatArray, *, l1_ratio: float) -> float:
    """The smallest penalty that zeroes every coefficient, for this fit.

    Centred exactly as ``fit_intercept=True`` centres the fit, so the returned
    scale is the one the estimator actually faces.
    """

    centred = features - features.mean(axis=0)
    residual = response - response.mean()
    return float(np.max(np.abs(centred.T @ residual)) / (len(response) * l1_ratio))


__all__ = [
    "DISTRIBUTED_LAG_ELASTIC_NET_METHOD_ID",
    "DISTRIBUTED_LAG_ELASTIC_NET_SCHEMA_ID",
    "DistributedLagElasticNetAdapter",
]
