"""``FAST_SLOW_LEDOIT_WOLF_CORRELATION``: one blended correlation, two lookbacks.

R1. The control estimates one Ledoit-Wolf correlation over the whole matured
standardized-residual window; this method estimates two -- a slow one over the
same full window and a fast one over its most recent tail -- and blends them at a
declared weight::

    C_T = (1 - eta) * C_LW_slow + eta * C_LW_fast
    C_T = normalize_to_correlation(C_T)
    Sigma_T = D_T C_T D_T

`eta` is the method's declared parameter axis, admitted at 0.25, 0.50 and 0.75.
One adapter owns one formula; three adapters would have been three copies of it
whose only difference was a constant the recipe already carries.

Everything except the correlation lookback is deliberately identical to the
control, because a comparison in which two things moved answers nothing. The
same causally available raw return lane, the same 315-session window, the same
EWMA standardization, the same Ledoit-Wolf settings, and the same single latest
causal volatility diagonal. ``D_T`` is applied exactly once, to the blended
correlation, and is the same forecast volatility the control would have used.

Why the standardization is not reimplemented here: it is driven through the
control's own sealed ``CovarianceRecipe`` and its own
``_standardized_residuals``, so "the same causal standardization semantics" is a
fact about which code runs rather than a claim in a docstring. The Ledoit-Wolf
correlation step below *is* written out here, applied twice, and that is a
deliberate exception -- ``estimators/covariance.py`` is inside the frozen Risk
execution closure, so extracting a shared helper from it would move every
published covariance identity in the repository for a refactor that changes no
number. The five lines are mirrored knowingly; the golden test pins both against
the same construction.

The blend is normalized rather than assumed. Each component's diagonal is
``cov_ii / (sqrt(cov_ii) * sqrt(cov_ii))``, which is 1.0 to rounding and not
exactly 1.0, so the convex combination inherits a diagonal that is near one and
not one. Normalizing is what makes the result a correlation matrix in fact and
not by intention.

Both fitted shrinkage strengths are reported. A single blended number would hide
the thing a reader of this method most needs to see: the fast window is a
quarter the length of the slow one, so its Ledoit-Wolf shrinkage is normally the
larger of the two, and how much larger is the method's whole risk.

Units: one-session open-to-open log-return covariance, identical to the control.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.covariance import LedoitWolf

from alphalattice.investment.risk_research.contracts import (
    ADMISSIBLE_EWMA_DECAY,
    DEFAULT_EWMA_DECAY,
    CovarianceDiagnostics,
    CovarianceRecipe,
    seal_contract,
)
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
from .covariance import (
    COVARIANCE_INPUT_SESSIONS,
    RISK_NUMERICAL_THREAD_LIMIT,
    RiskNumericalError,
    _standardized_residuals,
    _validate_inputs,
    numerical_environment_hash,
)
from .domains import FAST_SLOW_PARAMETER_DOMAIN, RiskParameterDomain
from .matrix_identity import matrix_content_hash

type FloatArray = NDArray[np.float64]

FAST_SLOW_ADAPTER_ID = "risk-covariance-fast-slow-ledoit-wolf"
FAST_SLOW_RECIPE_SCHEMA_ID = "FAST_SLOW_LEDOIT_WOLF_CORRELATION"

ADMISSIBLE_BLEND_WEIGHT: tuple[float, ...] = (0.25, 0.50, 0.75)
"""The declared ``eta`` axis: three points, none of them an endpoint.

Zero and one are excluded on purpose. At zero the method is the control with an
extra normalization step, and at one it is a 63-session correlation with no slow
component at all; neither is the blended estimator this recipe names, and
admitting them would let one schema describe three different methods.
"""

SLOW_CORRELATION_SESSIONS = 252
FAST_CORRELATION_SESSIONS = 63


@lru_cache(maxsize=1)
def _implementation_content_hash() -> str:
    """Bytes of every first-party module whose edits move these numbers.

    The control's module is included because this method's standardization runs
    inside it: an edit there changes what R1 computes, and a binding that named
    only this file would not say so.
    """

    from . import covariance, matrix_identity

    return implementation_content_hash(
        Path(__file__), Path(covariance.__file__), Path(matrix_identity.__file__)
    )


class FastSlowCorrelationRecipe(BaseModel):  # type: ignore[misc]
    """The sealed R1 recipe: the control's window, plus a blend weight."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["FastSlowCorrelationRecipe"] = "FastSlowCorrelationRecipe"
    recipe_name: Literal["FAST_SLOW_LEDOIT_WOLF_CORRELATION"] = "FAST_SLOW_LEDOIT_WOLF_CORRELATION"
    initialization_sessions: Literal[63] = 63
    standardized_residual_sessions: Literal[252] = 252
    ewma_decay: float = Field(default=DEFAULT_EWMA_DECAY, ge=0.0, lt=1.0)
    slow_correlation_sessions: Literal[252] = 252
    fast_correlation_sessions: Literal[63] = 63
    blend_weight: float = Field(gt=0.0, lt=1.0)
    ledoit_wolf_store_precision: Literal[False] = False
    ledoit_wolf_assume_centered: Literal[False] = False
    output_unit: Literal["one-session-open-to-open-log-return-covariance"] = (
        "one-session-open-to-open-log-return-covariance"
    )
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_recipe(self) -> Self:
        # Membership by exact float equality, exactly as the control admits its
        # decay: an approximate check would admit values that hash differently
        # and so create recipe identities the declared domain never named.
        """Require admitted blend/decay values, ordered windows and exact recipe identity.

        Returns:
            This recipe after blend, decay, fast/slow-window and recipe_hash checks.

        Raises:
            ValueError: A parameter is undeclared, fast is not below slow, slow differs from the
                residual window or recipe_hash is inconsistent.
        """
        if not any(self.blend_weight == candidate for candidate in ADMISSIBLE_BLEND_WEIGHT):
            raise ValueError("risk_research.fast_slow_blend_weight_invalid")
        if not any(self.ewma_decay == candidate for candidate in ADMISSIBLE_EWMA_DECAY):
            raise ValueError("risk_research.fast_slow_decay_invalid")
        if self.fast_correlation_sessions >= self.slow_correlation_sessions:
            raise ValueError("risk_research.fast_slow_lookback_order_invalid")
        if self.slow_correlation_sessions != self.standardized_residual_sessions:
            # The slow component is the whole matured residual window. A shorter
            # one would silently discard rows the Host causally admitted.
            raise ValueError("risk_research.fast_slow_slow_lookback_invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"recipe_hash"}))
        if self.recipe_hash != expected:
            raise ValueError("risk_research.fast_slow_recipe_identity_invalid")
        return self

    def standardization_recipe(self) -> CovarianceRecipe:
        """The control's own sealed recipe for the shared standardization step.

        Constructed rather than duck-typed so the shared code runs against the
        contract it was written for, and so a future divergence between the two
        windows fails to seal instead of silently standardizing differently.
        """
        return seal_contract(
            CovarianceRecipe,
            "recipe_hash",
            initialization_sessions=self.initialization_sessions,
            standardized_residual_sessions=self.standardized_residual_sessions,
            ewma_decay=self.ewma_decay,
            ledoit_wolf_store_precision=self.ledoit_wolf_store_precision,
            ledoit_wolf_assume_centered=self.ledoit_wolf_assume_centered,
        )


def _ledoit_wolf_correlation(
    residuals: FloatArray, *, store_precision: bool, assume_centered: bool
) -> tuple[FloatArray, float]:
    """One Ledoit-Wolf correlation and the shrinkage that produced it."""

    estimator = LedoitWolf(store_precision=store_precision, assume_centered=assume_centered).fit(
        residuals
    )
    covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    scale = np.sqrt(np.diag(covariance))
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise RiskNumericalError("risk_research.residual_covariance_diagonal_invalid")
    correlation = covariance / np.outer(scale, scale)
    correlation = (correlation + correlation.T) * 0.5
    shrinkage = float(estimator.shrinkage_)
    if not 0.0 <= shrinkage <= 1.0:
        raise RiskNumericalError("risk_research.shrinkage_invalid")
    return correlation, shrinkage


def estimate_fast_slow_covariance(
    *,
    returns: FloatArray,
    ordered_listing_ids: tuple[str, ...],
    formation_session: date,
    recipe: FastSlowCorrelationRecipe,
) -> EstimatedCovariance:
    """Estimate one causal open-to-open covariance from a blended correlation."""
    values = _validate_inputs(returns, ordered_listing_ids)
    residuals, forecast_variance = _standardized_residuals(values, recipe.standardization_recipe())
    if residuals.shape[0] != recipe.slow_correlation_sessions:
        raise RiskNumericalError("risk_research.fast_slow_residual_axis_invalid")

    slow_correlation, slow_shrinkage = _ledoit_wolf_correlation(
        residuals,
        store_precision=recipe.ledoit_wolf_store_precision,
        assume_centered=recipe.ledoit_wolf_assume_centered,
    )
    fast_correlation, fast_shrinkage = _ledoit_wolf_correlation(
        residuals[-recipe.fast_correlation_sessions :],
        store_precision=recipe.ledoit_wolf_store_precision,
        assume_centered=recipe.ledoit_wolf_assume_centered,
    )

    eta = recipe.blend_weight
    blended = (1.0 - eta) * slow_correlation + eta * fast_correlation
    blend_scale = np.sqrt(np.diag(blended))
    if not np.isfinite(blend_scale).all() or np.any(blend_scale <= 0.0):
        raise RiskNumericalError("risk_research.fast_slow_blend_diagonal_invalid")
    correlation = blended / np.outer(blend_scale, blend_scale)
    correlation = (correlation + correlation.T) * 0.5

    forecast_volatility = np.sqrt(forecast_variance)
    covariance = correlation * np.outer(forecast_volatility, forecast_volatility)
    covariance = (covariance + covariance.T) * 0.5

    # The control's gates, applied to this method's output. Fail-closed and
    # never repaired: an estimate that cannot pass them is not returned.
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

    asset_count = len(ordered_listing_ids)
    off_diagonal_sum = float(correlation.sum() - np.trace(correlation))
    average_correlation = (
        off_diagonal_sum / (asset_count * (asset_count - 1)) if asset_count > 1 else 0.0
    )
    total_eigenvalue = float(eigenvalues.sum())
    annualized = forecast_volatility * np.sqrt(252.0)
    # A weighted summary of the two fitted strengths, blended exactly as the
    # correlations are. It is a diagnostic, not a claim that the blend is itself
    # a Ledoit-Wolf estimate at this intensity; both components travel
    # separately on the estimate so a reader never has to unpick this one.
    effective_shrinkage = (1.0 - eta) * slow_shrinkage + eta * fast_shrinkage
    diagnostics = CovarianceDiagnostics(
        formation_session=formation_session,
        asset_count=asset_count,
        shrinkage=effective_shrinkage,
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
        numerical_environment_hash=numerical_environment_hash(),
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
        component_shrinkages=(
            ("slow_ledoit_wolf_shrinkage", slow_shrinkage),
            ("fast_ledoit_wolf_shrinkage", fast_shrinkage),
        ),
    )


class FastSlowCovarianceAdapter:
    """Install the blended fast/slow correlation recipe behind the estimator seam."""

    adapter_id = FAST_SLOW_ADAPTER_ID
    recipe_schema_id = FAST_SLOW_RECIPE_SCHEMA_ID

    def describe_numerical_binding(self) -> RiskEstimatorNumericalBinding:
        """Seal this adapter implementation closure and single-thread numerical policy.

        The binding declares a declared fast/slow Ledoit-Wolf correlation blend, correlation
        normalization and causal EWMA volatility scaling.

        Returns:
            Exact implementation, content format, deterministic policy and runtime-capability
            binding.
        """
        return RiskEstimatorNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimate_content_format_id="risk-covariance-dense-symmetric-float64",
            implementation_owners=(
                "alphalattice.investment.risk_research.estimators.fast_slow",
                "alphalattice.investment.risk_research.estimators.covariance",
                "sklearn.covariance.LedoitWolf",
            ),
            implementation_content_hash=_implementation_content_hash(),
            deterministic_policy={
                "thread_limit": RISK_NUMERICAL_THREAD_LIMIT,
                "required_input_sessions": COVARIANCE_INPUT_SESSIONS,
                "standardization": "ewma-variance",
                "shrinkage": "ledoit-wolf-correlation-fast-slow-blend",
                "slow_correlation_sessions": SLOW_CORRELATION_SESSIONS,
                "fast_correlation_sessions": FAST_CORRELATION_SESSIONS,
                "blend": "convex-then-normalized-to-correlation",
                "volatility_diagonal": "single-latest-causal-ewma",
                "symmetrization": "average-with-transpose",
            },
            required_runtime_capabilities=(SINGLE_THREAD_NUMERICAL_CAPABILITY,),
        )

    def validate_recipe(self, recipe: RiskEstimatorRecipeEnvelope) -> FastSlowCorrelationRecipe:
        """Admit an exact adapter/schema route and its concrete covariance recipe.

        Args:
            recipe: Sealed Risk envelope carrying this adapter route and typed parameters.

        Returns:
            Validated FastSlowCorrelationRecipe from the envelope parameters.

        Raises:
            RiskNumericalError: Adapter/schema routing or concrete recipe validation fails.
        """
        if recipe.adapter_id != self.adapter_id:
            raise RiskNumericalError("risk_research.fast_slow_adapter_route_invalid")
        if recipe.recipe_schema_id != self.recipe_schema_id:
            raise RiskNumericalError("risk_research.fast_slow_recipe_schema_invalid")
        try:
            active = FastSlowCorrelationRecipe.model_validate(recipe.parameters)
        except ValueError as error:
            raise RiskNumericalError("risk_research.fast_slow_recipe_invalid") from error
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
            Covariance estimate and diagnostics using a declared fast/slow Ledoit-Wolf correlation
            blend, correlation normalization and causal EWMA volatility scaling.

        Raises:
            RiskNumericalError: Recipe admission or the method numerical-input contract fails.
        """
        active = self.validate_recipe(recipe)
        return estimate_fast_slow_covariance(
            returns=inputs.returns,
            ordered_listing_ids=inputs.ordered_listing_ids,
            formation_session=inputs.formation_session,
            recipe=active,
        )


class FastSlowCovarianceCapability:
    """Installs R1: adapter, declared domain, and its own recipe sealing."""

    capability_handle = FAST_SLOW_RECIPE_SCHEMA_ID
    randomness_policy = RANDOMNESS_NONE
    """Two deterministic Ledoit-Wolf fits and a fixed blend; no random draw."""

    @property
    def adapter(self) -> FastSlowCovarianceAdapter:
        """Return this capability concrete covariance adapter.

        Returns:
            New adapter instance for this installed method and recipe schema.
        """
        return FastSlowCovarianceAdapter()

    @property
    def parameter_domain(self) -> RiskParameterDomain:
        """Return this method complete declared finite parameter domain.

        Returns:
            Named admissible axes and defaults for this covariance capability.
        """
        return FAST_SLOW_PARAMETER_DOMAIN

    def seal(self, admitted: Mapping[str, object]) -> tuple[FastSlowCorrelationRecipe, str]:
        """Seal already admitted parameters into the method typed covariance recipe.

        Args:
            admitted: Parameter mapping checked against this capability declared domain.

        Returns:
            Typed recipe and its canonical recipe_hash.

        Raises:
            pydantic.ValidationError: Typed parameters or recipe consistency violate the concrete
                model.
        """
        recipe = seal_contract(FastSlowCorrelationRecipe, "recipe_hash", **admitted)
        return recipe, str(recipe.recipe_hash)


__all__ = [
    "ADMISSIBLE_BLEND_WEIGHT",
    "FAST_CORRELATION_SESSIONS",
    "FAST_SLOW_ADAPTER_ID",
    "FAST_SLOW_RECIPE_SCHEMA_ID",
    "SLOW_CORRELATION_SESSIONS",
    "FastSlowCorrelationRecipe",
    "FastSlowCovarianceAdapter",
    "FastSlowCovarianceCapability",
    "estimate_fast_slow_covariance",
]
