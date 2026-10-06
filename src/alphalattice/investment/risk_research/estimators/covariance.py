"""Pure numerical owner for the sole Risk Desk covariance recipe."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_info, threadpool_limits  # type: ignore[import-untyped]

from alphalattice.investment.risk_research.contracts import (
    CovarianceDiagnostics,
    CovarianceRecipe,
    default_covariance_recipe,
    seal_contract,
)
from alphalattice.kernel.shared_kernel.environment import recorded_environment
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .capability import RANDOMNESS_NONE
from .contracts import (
    SINGLE_THREAD_NUMERICAL_CAPABILITY,
    BoundRiskReturnInput,
    EstimatedCovariance,
    RiskEstimatorNumericalBinding,
    RiskEstimatorRecipeEnvelope,
    implementation_content_hash,
)
from .domains import COVARIANCE_PARAMETER_DOMAIN, RiskParameterDomain
from .matrix_identity import matrix_content_hash

type FloatArray = NDArray[np.float64]
RISK_NUMERICAL_THREAD_LIMIT = 1
COVARIANCE_ADAPTER_ID = "risk-covariance-ewma-standardized-ledoit-wolf"
COVARIANCE_RECIPE_SCHEMA_ID = "EWMA_STANDARDIZED_LEDOIT_WOLF_CORRELATION"
COVARIANCE_INPUT_SESSIONS = 315


class RiskNumericalError(ValueError):
    """A stable fail-closed numerical boundary error."""


@lru_cache(maxsize=1)
def _implementation_content_hash() -> str:
    """Bytes of the modules that actually compute a covariance estimate.

    Cached because it is read once per process and the files cannot change
    underneath a running interpreter.
    """

    from . import matrix_identity

    return implementation_content_hash(Path(__file__), Path(matrix_identity.__file__))


@lru_cache(maxsize=1)
def _numerical_backends() -> tuple[dict[str, str | None], ...]:
    fields = (
        "user_api",
        "internal_api",
        "prefix",
        "version",
        "threading_layer",
        "architecture",
    )
    pools: list[dict[str, str | None]] = []
    for pool in threadpool_info():
        pools.append(
            {field: str(pool[field]) if pool.get(field) is not None else None for field in fields}
        )
    return tuple(
        sorted(
            pools,
            key=lambda item: tuple(item[field] or "" for field in fields),
        )
    )


def numerical_environment() -> dict[str, Any]:
    """The environment an estimate was computed in, recorded beside it (LAWS.md ID6).

    Provenance: each estimate's diagnostics and each surface carry its hash, and no
    identity folds it or compares it, so a dependency upgrade moves no Risk identity;
    whether it moves a number is answered by U0.
    """
    return {
        **recorded_environment(("numpy", "scikit-learn", "threadpoolctl")),
        "thread_limit": RISK_NUMERICAL_THREAD_LIMIT,
        "numerical_backends": _numerical_backends(),
    }


@lru_cache(maxsize=1)
def numerical_environment_hash() -> str:
    """Hash the observed numerical environment as execution provenance.

    Returns:
        Canonical identity of numerical_environment(), retained separately from result content.
    """
    return str(canonical_hash(numerical_environment()))


@contextmanager
def risk_numerical_thread_policy() -> Iterator[None]:
    """Run the numerical recipe under its code-owned thread policy."""
    with threadpool_limits(limits=RISK_NUMERICAL_THREAD_LIMIT):
        if any(
            int(pool["num_threads"]) > RISK_NUMERICAL_THREAD_LIMIT
            for pool in threadpool_info()
            if pool.get("num_threads") is not None
        ):
            raise RiskNumericalError("risk_research.numerical_thread_policy_failed")
        yield


def _validate_inputs(returns: FloatArray, listing_ids: tuple[str, ...]) -> FloatArray:
    values = np.asarray(returns, dtype=np.float64)
    if values.shape != (COVARIANCE_INPUT_SESSIONS, len(listing_ids)) or len(listing_ids) == 0:
        raise RiskNumericalError("risk_research.covariance_input_shape_invalid")
    if len(listing_ids) != len(set(listing_ids)):
        raise RiskNumericalError("risk_research.covariance_axis_invalid")
    if not np.isfinite(values).all():
        raise RiskNumericalError("risk_research.covariance_input_non_finite")
    return values


def _standardized_residuals(
    returns: FloatArray, recipe: CovarianceRecipe
) -> tuple[FloatArray, FloatArray]:
    initialization = returns[: recipe.initialization_sessions]
    variance = np.var(initialization, axis=0, ddof=1, dtype=np.float64)
    if not np.isfinite(variance).all() or np.any(variance <= 0.0):
        raise RiskNumericalError("risk_research.ewma_initial_variance_invalid")
    residuals = np.empty(
        (recipe.standardized_residual_sessions, returns.shape[1]), dtype=np.float64
    )
    decay = recipe.ewma_decay
    for index, row in enumerate(returns[recipe.initialization_sessions :]):
        residuals[index] = row / np.sqrt(variance)
        variance = decay * variance + (1.0 - decay) * np.square(row)
    if not np.isfinite(residuals).all() or not np.isfinite(variance).all():
        raise RiskNumericalError("risk_research.ewma_standardization_non_finite")
    if np.any(variance <= 0.0):
        raise RiskNumericalError("risk_research.ewma_forecast_variance_invalid")
    return residuals, variance


def estimate_dynamic_covariance(
    *,
    returns: FloatArray,
    ordered_listing_ids: tuple[str, ...],
    formation_session: date,
    recipe: CovarianceRecipe | None = None,
) -> EstimatedCovariance:
    """Estimate one causal open-to-open covariance from exactly 315 returns."""
    active_recipe = recipe or default_covariance_recipe()
    values = _validate_inputs(returns, ordered_listing_ids)
    residuals, forecast_variance = _standardized_residuals(values, active_recipe)
    estimator = LedoitWolf(
        store_precision=active_recipe.ledoit_wolf_store_precision,
        assume_centered=active_recipe.ledoit_wolf_assume_centered,
    ).fit(residuals)
    residual_covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    residual_scale = np.sqrt(np.diag(residual_covariance))
    if not np.isfinite(residual_scale).all() or np.any(residual_scale <= 0.0):
        raise RiskNumericalError("risk_research.residual_covariance_diagonal_invalid")
    correlation = residual_covariance / np.outer(residual_scale, residual_scale)
    correlation = (correlation + correlation.T) * 0.5
    forecast_volatility = np.sqrt(forecast_variance)
    covariance = correlation * np.outer(forecast_volatility, forecast_volatility)
    covariance = (covariance + covariance.T) * 0.5

    if not np.isfinite(covariance).all() or not np.isfinite(correlation).all():
        raise RiskNumericalError("risk_research.covariance_non_finite")
    if not np.array_equal(covariance, covariance.T):
        raise RiskNumericalError("risk_research.covariance_not_symmetric")
    if not np.allclose(np.diag(correlation), 1.0, rtol=0.0, atol=1e-12):
        raise RiskNumericalError("risk_research.correlation_diagonal_invalid")
    if not np.allclose(np.diag(covariance), forecast_variance, rtol=1e-12, atol=1e-18):
        raise RiskNumericalError("risk_research.covariance_diagonal_inconsistent")

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    machine_tiny: float = float(np.finfo(np.float64).tiny)
    tolerance = max(maximum * 1e-12, machine_tiny)
    if minimum <= tolerance:
        raise RiskNumericalError("risk_research.covariance_not_positive_definite")
    condition = maximum / minimum
    if not np.isfinite(condition) or condition > 1e12:
        raise RiskNumericalError("risk_research.covariance_condition_exceeded")
    shrinkage = float(estimator.shrinkage_)
    if not 0.0 <= shrinkage <= 1.0:
        raise RiskNumericalError("risk_research.shrinkage_invalid")

    asset_count = len(ordered_listing_ids)
    off_diagonal_sum = float(correlation.sum() - np.trace(correlation))
    average_correlation = (
        off_diagonal_sum / (asset_count * (asset_count - 1)) if asset_count > 1 else 0.0
    )
    total_eigenvalue = float(eigenvalues.sum())
    annualized = forecast_volatility * np.sqrt(252.0)
    environment_hash = numerical_environment_hash()
    diagnostics = CovarianceDiagnostics(
        formation_session=formation_session,
        asset_count=asset_count,
        shrinkage=shrinkage,
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        condition_number=condition,
        trace=float(np.trace(covariance)),
        average_correlation=average_correlation,
        top_one_eigenvalue_share=maximum / total_eigenvalue,
        top_five_eigenvalue_share=float(eigenvalues[-min(5, asset_count) :].sum())
        / total_eigenvalue,
        annualized_volatility_minimum=float(annualized.min()),
        annualized_volatility_median=float(np.median(annualized)),
        annualized_volatility_maximum=float(annualized.max()),
        numerical_environment_hash=environment_hash,
        matrix_hash=matrix_content_hash(covariance),
    )
    covariance.setflags(write=False)
    forecast_volatility.setflags(write=False)
    eigenvalues.setflags(write=False)
    eigenvectors.setflags(write=False)
    return EstimatedCovariance(
        matrix=covariance,
        forecast_volatility=forecast_volatility,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        diagnostics=diagnostics,
    )


class CovarianceEstimatorAdapter:
    """Install the sole real Risk covariance recipe behind the estimator seam."""

    adapter_id = COVARIANCE_ADAPTER_ID
    recipe_schema_id = COVARIANCE_RECIPE_SCHEMA_ID

    def describe_numerical_binding(self) -> RiskEstimatorNumericalBinding:
        """Seal this adapter implementation closure and single-thread numerical policy.

        The binding declares EWMA variance standardization, Ledoit-Wolf correlation shrinkage and
        symmetric float64 output.

        Returns:
            Exact implementation, content format, deterministic policy and runtime-capability
            binding.
        """
        return RiskEstimatorNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimate_content_format_id="risk-covariance-dense-symmetric-float64",
            implementation_owners=(
                "alphalattice.investment.risk_research.estimators.covariance",
                "sklearn.covariance.LedoitWolf",
            ),
            # The bytes of the first-party half, so this binding is a statement
            # about code rather than about naming. sklearn's version is the
            # environment, recorded beside each estimate (LAWS.md ID6).
            implementation_content_hash=_implementation_content_hash(),
            deterministic_policy={
                "thread_limit": RISK_NUMERICAL_THREAD_LIMIT,
                "required_input_sessions": COVARIANCE_INPUT_SESSIONS,
                "standardization": "ewma-variance",
                "shrinkage": "ledoit-wolf-correlation",
                "symmetrization": "average-with-transpose",
            },
            required_runtime_capabilities=(SINGLE_THREAD_NUMERICAL_CAPABILITY,),
        )

    def validate_recipe(self, recipe: RiskEstimatorRecipeEnvelope) -> CovarianceRecipe:
        """Admit an exact adapter/schema route and its concrete covariance recipe.

        Args:
            recipe: Sealed Risk envelope carrying this adapter route and typed parameters.

        Returns:
            Validated CovarianceRecipe from the envelope parameters.

        Raises:
            RiskNumericalError: Adapter/schema routing or concrete recipe validation fails.
        """
        if recipe.adapter_id != self.adapter_id:
            raise RiskNumericalError("risk_research.covariance_adapter_route_invalid")
        if recipe.recipe_schema_id != self.recipe_schema_id:
            raise RiskNumericalError("risk_research.covariance_recipe_schema_invalid")
        try:
            active = CovarianceRecipe.model_validate(recipe.parameters)
        except ValueError as error:
            raise RiskNumericalError("risk_research.covariance_recipe_invalid") from error
        return active

    def estimate(
        self,
        *,
        recipe: RiskEstimatorRecipeEnvelope,
        inputs: BoundRiskReturnInput,
    ) -> EstimatedCovariance:
        """Estimate this method covariance from bound causal return history.

        Args:
            recipe: Sealed envelope admitted against this adapter and schema.
            inputs: Finite bound history with ordered listing and formation axes.

        Returns:
            Covariance estimate and diagnostics using EWMA variance standardization, Ledoit-Wolf
            correlation shrinkage and symmetric float64 output.

        Raises:
            RiskNumericalError: Recipe admission or the method numerical-input contract fails.
        """
        active = self.validate_recipe(recipe)
        return estimate_dynamic_covariance(
            returns=inputs.returns,
            ordered_listing_ids=inputs.ordered_listing_ids,
            formation_session=inputs.formation_session,
            recipe=active,
        )


class CovarianceCapability:
    """The one production Risk capability, installed explicitly by the Host.

    Everything the authoring chain needs for this method is reachable from here
    rather than from a compiler that named the method or a central domain table
    the catalog read backwards into.
    """

    capability_handle = COVARIANCE_RECIPE_SCHEMA_ID
    randomness_policy = RANDOMNESS_NONE
    """This estimator draws no random numbers, so a seed cannot change its output."""

    @property
    def adapter(self) -> CovarianceEstimatorAdapter:
        """Return this capability concrete covariance adapter.

        Returns:
            New adapter instance for this installed method and recipe schema.
        """
        return CovarianceEstimatorAdapter()

    @property
    def parameter_domain(self) -> RiskParameterDomain:
        """Return this method complete declared finite parameter domain.

        Returns:
            Named admissible axes and defaults for this covariance capability.
        """
        return COVARIANCE_PARAMETER_DOMAIN

    def seal(self, admitted: Mapping[str, object]) -> tuple[CovarianceRecipe, str]:
        """Seal admitted values through this schema's own contract."""
        recipe = seal_contract(CovarianceRecipe, "recipe_hash", **admitted)
        return recipe, str(recipe.recipe_hash)


__all__ = [
    "COVARIANCE_ADAPTER_ID",
    "COVARIANCE_INPUT_SESSIONS",
    "COVARIANCE_RECIPE_SCHEMA_ID",
    "RISK_NUMERICAL_THREAD_LIMIT",
    "CovarianceCapability",
    "CovarianceEstimatorAdapter",
    "EstimatedCovariance",
    "RiskNumericalError",
    "estimate_dynamic_covariance",
    "matrix_content_hash",
    "numerical_environment",
    "numerical_environment_hash",
    "risk_numerical_thread_policy",
]
