"""``DIAGONAL_SHRUNK_CORRELATION``: the control's correlation, pulled toward identity.

R2. The control estimates one Ledoit-Wolf correlation and applies the causal
volatility diagonal to it; this method takes the same correlation and shrinks it
toward the identity at a declared intensity::

    C_T = (1 - phi) * C_LW + phi * I
    Sigma_T = D_T C_T D_T

Everything else -- the raw return lane, the 315-session window, the EWMA
standardization, the Ledoit-Wolf settings, the single latest causal volatility
diagonal -- is the control's, run through the control's own sealed
``CovarianceRecipe``, exactly as R1 does it. The only thing that moves is how
much correlation structure survives.

**Why this method exists, and why its axis is the two endpoints.** The question
it was installed to answer is not "is there a better covariance"; it is whether
correlation structure changes a portfolio *at all*. The Dynamic Panel campaign
found the R0/R1 difference moved the predicted-over-realized variance ratio from
1.0681 to 0.9108 and produced weights that were **identical to the byte**,
because the policy consuming them was rank-only and a rank cannot see a
covariance. That confound is what this study exists to remove, and a method
whose two admissible points are "all the correlation" and "none of it" is the
sharpest available probe:

``correlation_shrinkage = 0.0``
    Reproduces the control's matrix **value for value**, and that is a test
    rather than an argument: ``np.array_equal`` against the control's estimate
    is exact, so every weight, log score and realized path downstream is
    identical.

    It is *not* byte-identical, and the difference is worth stating because it
    is easy to assume otherwise. ``(1 - 0) * C`` preserves a negative zero and
    ``+ 0 * I`` then destroys it -- IEEE-754 gives ``-0.0 + 0.0 == +0.0`` -- so a
    correlation entry the control published as ``-0.0`` arrives here as ``+0.0``.
    ``matrix_content_hash`` is deliberately signed-zero sensitive, so the two
    matrices carry different content identities while being the same numbers.
    Nothing scientific rides on that: the arm publishes under its own identity,
    as it should, and the equivalence that matters is the one on the values.
``correlation_shrinkage = 1.0``
    A pure diagonal. Every off-diagonal covariance is gone and only the causal
    volatility forecast remains.

If a covariance-consuming policy's weights and realized risk do not separate
between those two, then correlation structure is doing nothing in this
Portfolio, and the answer to "does the risk estimator move the optimizer" is no
-- decisively, and without appeal to a difference too small to see. That is why
the endpoints are admitted here and deliberately excluded in R1: R1's endpoints
would each be a *different method* wearing one schema, while these two are the
same formula at the boundary of its own axis, which is the whole point.

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
from .domains import DIAGONAL_PARAMETER_DOMAIN, RiskParameterDomain
from .matrix_identity import matrix_content_hash

type FloatArray = NDArray[np.float64]

DIAGONAL_ADAPTER_ID = "risk-covariance-diagonal-shrunk-ledoit-wolf"
DIAGONAL_RECIPE_SCHEMA_ID = "DIAGONAL_SHRUNK_CORRELATION"

ADMISSIBLE_CORRELATION_SHRINKAGE: tuple[float, ...] = (0.0, 1.0)
"""The declared ``phi`` axis: the two endpoints, and nothing between them.

Intermediate values are not admitted because this method is not a tuning knob.
It is a two-point contrast between "the control" and "no correlation at all",
and a grid of intensities in between would invite a search for the best one --
which is a different, and much weaker, experiment than the one it is here for.
"""


@lru_cache(maxsize=1)
def _implementation_content_hash() -> str:
    """Bytes of every first-party module whose edits move these numbers.

    The control's module is included because the standardization and the input
    validation run inside it, exactly as they do for the blended challenger.
    """

    from . import covariance, matrix_identity

    return implementation_content_hash(
        Path(__file__), Path(covariance.__file__), Path(matrix_identity.__file__)
    )


class DiagonalShrunkCorrelationRecipe(BaseModel):  # type: ignore[misc]
    """The sealed R2 recipe: the control's window, plus a shrink intensity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["DiagonalShrunkCorrelationRecipe"] = "DiagonalShrunkCorrelationRecipe"
    recipe_name: Literal["DIAGONAL_SHRUNK_CORRELATION"] = "DIAGONAL_SHRUNK_CORRELATION"
    initialization_sessions: Literal[63] = 63
    standardized_residual_sessions: Literal[252] = 252
    ewma_decay: float = Field(default=DEFAULT_EWMA_DECAY, ge=0.0, lt=1.0)
    correlation_shrinkage: float = Field(ge=0.0, le=1.0)
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
        """Require exact declared shrinkage/decay membership and canonical recipe identity.

        Returns:
            This recipe after exact parameter membership and recipe_hash checks.

        Raises:
            ValueError: Shrinkage, decay or recipe_hash differs from the declared finite
                domain/content.
        """
        if not any(
            self.correlation_shrinkage == candidate
            for candidate in ADMISSIBLE_CORRELATION_SHRINKAGE
        ):
            raise ValueError("risk_research.diagonal_shrinkage_invalid")
        if not any(self.ewma_decay == candidate for candidate in ADMISSIBLE_EWMA_DECAY):
            raise ValueError("risk_research.diagonal_decay_invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"recipe_hash"}))
        if self.recipe_hash != expected:
            raise ValueError("risk_research.diagonal_recipe_identity_invalid")
        return self

    def standardization_recipe(self) -> CovarianceRecipe:
        """The control's own sealed recipe for the shared standardization step.

        Constructed rather than duck-typed, for the reason R1 states: the shared
        code runs against the contract it was written for, and a divergence
        between the two windows fails to seal instead of standardizing
        differently in silence.
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
    """One Ledoit-Wolf correlation and the shrinkage that produced it.

    Mirrored from the control knowingly, for the reason R1 records: the control's
    module is inside the frozen Risk execution closure, so extracting a shared
    helper from it would move every published covariance identity in the
    repository for a refactor that changes no number.
    """

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


def estimate_diagonal_shrunk_covariance(
    *,
    returns: FloatArray,
    ordered_listing_ids: tuple[str, ...],
    formation_session: date,
    recipe: DiagonalShrunkCorrelationRecipe,
) -> EstimatedCovariance:
    """Estimate one causal covariance from a correlation shrunk toward identity."""
    values = _validate_inputs(returns, ordered_listing_ids)
    residuals, forecast_variance = _standardized_residuals(values, recipe.standardization_recipe())
    if residuals.shape[0] != recipe.standardized_residual_sessions:
        raise RiskNumericalError("risk_research.diagonal_residual_axis_invalid")

    fitted, ledoit_wolf_shrinkage = _ledoit_wolf_correlation(
        residuals,
        store_precision=recipe.ledoit_wolf_store_precision,
        assume_centered=recipe.ledoit_wolf_assume_centered,
    )
    phi = recipe.correlation_shrinkage
    # The stated formula, written as stated. At phi == 0 every value equals the
    # control's; the one thing that does not survive is a negative zero, because
    # adding ``+0.0`` to ``-0.0`` yields ``+0.0``. Special-casing the branch to
    # preserve the byte would make the endpoint a different code path from the
    # interior, which is a worse trade than a signed zero.
    correlation = (1.0 - phi) * fitted + phi * np.eye(fitted.shape[0], dtype=np.float64)
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
    # How much correlation survives, expressed on the same scale the control and
    # the blended challenger report: the fitted Ledoit-Wolf intensity, pulled the
    # rest of the way to identity by ``phi``. At ``phi == 1`` it is exactly 1.0,
    # which is the honest reading -- the correlation is entirely prior.
    effective_shrinkage = ledoit_wolf_shrinkage + phi * (1.0 - ledoit_wolf_shrinkage)
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
            ("ledoit_wolf_shrinkage", ledoit_wolf_shrinkage),
            ("identity_shrinkage", float(phi)),
        ),
    )


class DiagonalShrunkCovarianceAdapter:
    """Install the identity-shrunk correlation recipe behind the estimator seam."""

    adapter_id = DIAGONAL_ADAPTER_ID
    recipe_schema_id = DIAGONAL_RECIPE_SCHEMA_ID

    def describe_numerical_binding(self) -> RiskEstimatorNumericalBinding:
        """Seal this adapter implementation closure and single-thread numerical policy.

        The binding declares Ledoit-Wolf correlation shrunk toward the identity and causal EWMA
        volatility scaling.

        Returns:
            Exact implementation, content format, deterministic policy and runtime-capability
            binding.
        """
        return RiskEstimatorNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimate_content_format_id="risk-covariance-dense-symmetric-float64",
            implementation_owners=(
                "alphalattice.investment.risk_research.estimators.diagonal",
                "alphalattice.investment.risk_research.estimators.covariance",
                "sklearn.covariance.LedoitWolf",
            ),
            implementation_content_hash=_implementation_content_hash(),
            # Declaring the single-thread capability is not optional: BLAS thread
            # count moves eigenvalue bits, and an adapter that omits it seals a
            # null thread count while ambient BLAS decides the numbers.
            deterministic_policy={
                "thread_limit": RISK_NUMERICAL_THREAD_LIMIT,
                "required_input_sessions": COVARIANCE_INPUT_SESSIONS,
                "standardization": "ewma-variance",
                "shrinkage": "ledoit-wolf-correlation-then-identity-shrink",
                "identity_target": "correlation-identity",
                "volatility_diagonal": "single-latest-causal-ewma",
                "symmetrization": "average-with-transpose",
            },
            required_runtime_capabilities=(SINGLE_THREAD_NUMERICAL_CAPABILITY,),
        )

    def validate_recipe(
        self, recipe: RiskEstimatorRecipeEnvelope
    ) -> DiagonalShrunkCorrelationRecipe:
        """Admit an exact adapter/schema route and its concrete covariance recipe.

        Args:
            recipe: Sealed Risk envelope carrying this adapter route and typed parameters.

        Returns:
            Validated DiagonalShrunkCorrelationRecipe from the envelope parameters.

        Raises:
            RiskNumericalError: Adapter/schema routing or concrete recipe validation fails.
        """
        if recipe.adapter_id != self.adapter_id:
            raise RiskNumericalError("risk_research.diagonal_adapter_route_invalid")
        if recipe.recipe_schema_id != self.recipe_schema_id:
            raise RiskNumericalError("risk_research.diagonal_recipe_schema_invalid")
        try:
            active = DiagonalShrunkCorrelationRecipe.model_validate(recipe.parameters)
        except ValueError as error:
            raise RiskNumericalError("risk_research.diagonal_recipe_invalid") from error
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
            Covariance estimate and diagnostics using Ledoit-Wolf correlation shrunk toward the
            identity and causal EWMA volatility scaling.

        Raises:
            RiskNumericalError: Recipe admission or the method numerical-input contract fails.
        """
        active = self.validate_recipe(recipe)
        return estimate_diagonal_shrunk_covariance(
            returns=inputs.returns,
            ordered_listing_ids=inputs.ordered_listing_ids,
            formation_session=inputs.formation_session,
            recipe=active,
        )


class DiagonalShrunkCovarianceCapability:
    """Installs R2: adapter, declared domain, and its own recipe sealing."""

    capability_handle = DIAGONAL_RECIPE_SCHEMA_ID
    randomness_policy = RANDOMNESS_NONE
    """One deterministic Ledoit-Wolf fit and a fixed convex blend; no random draw."""

    @property
    def adapter(self) -> DiagonalShrunkCovarianceAdapter:
        """Return this capability concrete covariance adapter.

        Returns:
            New adapter instance for this installed method and recipe schema.
        """
        return DiagonalShrunkCovarianceAdapter()

    @property
    def parameter_domain(self) -> RiskParameterDomain:
        """Return this method complete declared finite parameter domain.

        Returns:
            Named admissible axes and defaults for this covariance capability.
        """
        return DIAGONAL_PARAMETER_DOMAIN

    def seal(self, admitted: Mapping[str, object]) -> tuple[DiagonalShrunkCorrelationRecipe, str]:
        """Seal already admitted parameters into the method typed covariance recipe.

        Args:
            admitted: Parameter mapping checked against this capability declared domain.

        Returns:
            Typed recipe and its canonical recipe_hash.

        Raises:
            pydantic.ValidationError: Typed parameters or recipe consistency violate the concrete
                model.
        """
        recipe = seal_contract(DiagonalShrunkCorrelationRecipe, "recipe_hash", **admitted)
        return recipe, str(recipe.recipe_hash)


__all__ = [
    "ADMISSIBLE_CORRELATION_SHRINKAGE",
    "DIAGONAL_ADAPTER_ID",
    "DIAGONAL_RECIPE_SCHEMA_ID",
    "DiagonalShrunkCorrelationRecipe",
    "DiagonalShrunkCovarianceAdapter",
    "DiagonalShrunkCovarianceCapability",
    "estimate_diagonal_shrunk_covariance",
]
