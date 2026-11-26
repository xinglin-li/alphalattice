"""A second Risk method, owned entirely by this case study.

Its whole purpose is to be *unlike* the production covariance capability while
still travelling the same chain. So it differs on every axis the chain is
supposed to be neutral about:

recipe schema      ``CASE_STUDY_SHRUNK_DIAGONAL``, whose parameters do not
                   overlap ``CovarianceRecipe`` at all
parameter domain   its own, with two admissible values on the axis that matters
sealing            its own contract, sealed by the capability rather than by
                   anything the Host knows about
numerical binding  its own executable content identity, taken from this file
mathematics        shrunk diagonal variance, not EWMA-standardised Ledoit-Wolf

Nothing in ``src/`` names it. It reaches the numerical path by being installed
into a catalog that this case study constructs explicitly, and by nothing else:
if the compiler, the development writer or the verifier had to recognise it, the
"pluggable method" claim would be false and this file could not exist without
editing them.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.risk_research.contracts import CovarianceDiagnostics
from alphalattice.investment.risk_research.estimators.contracts import (
    SINGLE_THREAD_NUMERICAL_CAPABILITY,
    BoundRiskReturnInput,
    EstimatedCovariance,
    RiskEstimatorNumericalBinding,
    RiskEstimatorRecipeEnvelope,
    implementation_content_hash,
)
from alphalattice.investment.risk_research.estimators.domains import (
    ParameterAxis,
    RiskParameterDomain,
)
from alphalattice.investment.risk_research.estimators.matrix_identity import matrix_content_hash
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = NDArray[np.float64]

ALTERNATE_ADAPTER_ID = "case-study-shrunk-diagonal"
ALTERNATE_RECIPE_SCHEMA_ID = "CASE_STUDY_SHRUNK_DIAGONAL"

ADMISSIBLE_SHRINKAGE = (0.10, 0.50)
"""Two admissible values, so a parameter change is observable in real numbers."""


@lru_cache(maxsize=1)
def alternate_numerical_environment_hash() -> str:
    """The environment *this* estimator's numbers actually depend on.

    Deliberately not the covariance capability's. This method uses NumPy and
    nothing else: no scikit-learn, no threadpoolctl, and no single-thread BLAS
    policy, because a diagonal matrix has no cross-asset reduction whose bits a
    thread count could move.

    Naming that difference is the point. The generic compiler and the development
    writer used to import the covariance module's environment and stamp it on
    every Program and every surface, so this method would have been described --
    and had its identity moved -- by a scikit-learn version it never loads.
    """

    return str(
        canonical_hash(
            {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "architecture": platform.machine(),
                "byte_order": sys.byteorder,
                "numpy": version("numpy"),
                "estimator": "shrunk-diagonal-variance",
            }
        )
    )


class ShrunkDiagonalRecipe(BaseModel):  # type: ignore[misc]
    """A recipe schema with nothing in common with ``CovarianceRecipe``.

    Different field names, different meanings, different identity field. If the
    chain could only carry the covariance schema, this contract could not reach
    an estimator at all -- which is exactly what it is here to demonstrate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ShrunkDiagonalRecipe"] = "ShrunkDiagonalRecipe"
    schema_name: Literal["CASE_STUDY_SHRUNK_DIAGONAL"] = "CASE_STUDY_SHRUNK_DIAGONAL"
    shrinkage_intensity: float = Field(ge=0.0, le=1.0)
    variance_floor: float = Field(gt=0.0)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def seal(cls, *, shrinkage_intensity: float, variance_floor: float) -> Self:
        values = {
            "kind": "ShrunkDiagonalRecipe",
            "schema_name": "CASE_STUDY_SHRUNK_DIAGONAL",
            "shrinkage_intensity": shrinkage_intensity,
            "variance_floor": variance_floor,
        }
        return cls(**values, recipe_hash=str(canonical_hash(values)))  # type: ignore[arg-type]

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if not any(self.shrinkage_intensity == value for value in ADMISSIBLE_SHRINKAGE):
            raise ValueError("case_study.shrunk_diagonal_intensity_invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"recipe_hash"}))
        if self.recipe_hash != expected:
            raise ValueError("case_study.shrunk_diagonal_identity_invalid")
        return self


ALTERNATE_PARAMETER_DOMAIN = RiskParameterDomain(
    recipe_schema_id=ALTERNATE_RECIPE_SCHEMA_ID,
    axes=(
        ParameterAxis("shrinkage_intensity", ADMISSIBLE_SHRINKAGE, ADMISSIBLE_SHRINKAGE[0]),
        ParameterAxis("variance_floor", (1e-10,), 1e-10),
    ),
)


class ShrunkDiagonalAdapter:
    """Shrink each asset's sample variance toward the cross-sectional mean.

    Deliberately simple and deterministic. The point is not the mathematics but
    that the authored ``shrinkage_intensity`` visibly moves the matrix, so
    "the parameter reached the estimator" is provable from the numbers.
    """

    adapter_id = ALTERNATE_ADAPTER_ID
    recipe_schema_id = ALTERNATE_RECIPE_SCHEMA_ID

    def describe_numerical_binding(self) -> RiskEstimatorNumericalBinding:
        return RiskEstimatorNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimate_content_format_id="risk-covariance-dense-symmetric-float64",
            implementation_owners=("case-study.researcher-methodology-surface.alternate",),
            # Its own executable content, stated exactly as the built-in
            # capability states its own.
            implementation_content_hash=implementation_content_hash(Path(__file__)),
            deterministic_policy={"estimator": "shrunk-diagonal-variance"},
            required_runtime_capabilities=(SINGLE_THREAD_NUMERICAL_CAPABILITY,),
        )

    def validate_recipe(self, recipe: RiskEstimatorRecipeEnvelope) -> ShrunkDiagonalRecipe:
        """Decode and re-validate the adapter's own schema from the envelope.

        The generic path never does this. An adapter is the only thing entitled
        to know what its parameters mean.
        """

        if recipe.adapter_id != self.adapter_id:
            raise ValueError("case_study.shrunk_diagonal_route_invalid")
        if recipe.recipe_schema_id != self.recipe_schema_id:
            raise ValueError("case_study.shrunk_diagonal_schema_invalid")
        return ShrunkDiagonalRecipe(**recipe.parameters)

    def estimate(
        self,
        *,
        recipe: RiskEstimatorRecipeEnvelope,
        inputs: BoundRiskReturnInput,
    ) -> EstimatedCovariance:
        active = self.validate_recipe(recipe)
        values = np.asarray(inputs.returns, dtype=np.float64)
        sample = np.maximum(np.var(values, axis=0, ddof=1), active.variance_floor)
        # The authored intensity, applied to real numbers.
        target = float(np.mean(sample))
        variance: FloatArray = (
            1.0 - active.shrinkage_intensity
        ) * sample + active.shrinkage_intensity * target
        covariance: FloatArray = np.diag(variance)
        volatility: FloatArray = np.sqrt(variance)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        annualized = volatility * np.sqrt(252.0)
        diagnostics = CovarianceDiagnostics(
            formation_session=inputs.formation_session,
            asset_count=len(inputs.ordered_listing_ids),
            shrinkage=active.shrinkage_intensity,
            minimum_eigenvalue=float(eigenvalues[0]),
            maximum_eigenvalue=float(eigenvalues[-1]),
            condition_number=float(eigenvalues[-1] / eigenvalues[0]),
            trace=float(np.trace(covariance)),
            average_correlation=0.0,
            top_one_eigenvalue_share=float(eigenvalues[-1] / eigenvalues.sum()),
            top_five_eigenvalue_share=float(
                eigenvalues[-min(5, len(eigenvalues)) :].sum() / eigenvalues.sum()
            ),
            annualized_volatility_minimum=float(annualized.min()),
            annualized_volatility_median=float(np.median(annualized)),
            annualized_volatility_maximum=float(annualized.max()),
            # Its own environment, recorded beside the estimate (LAWS.md ID6).
            numerical_environment_hash=alternate_numerical_environment_hash(),
            matrix_hash=matrix_content_hash(covariance),
        )
        for array in (covariance, volatility, eigenvalues, eigenvectors):
            array.setflags(write=False)
        return EstimatedCovariance(
            matrix=covariance,
            forecast_volatility=volatility,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            diagnostics=diagnostics,
        )


class ShrunkDiagonalCapability:
    """Installs the alternate method: adapter, domain, and its own sealing."""

    capability_handle = ALTERNATE_RECIPE_SCHEMA_ID
    randomness_policy = "NONE"

    @property
    def adapter(self) -> ShrunkDiagonalAdapter:
        return ShrunkDiagonalAdapter()

    @property
    def parameter_domain(self) -> RiskParameterDomain:
        return ALTERNATE_PARAMETER_DOMAIN

    def seal(self, admitted: Mapping[str, object]) -> tuple[ShrunkDiagonalRecipe, str]:
        recipe = ShrunkDiagonalRecipe.seal(
            shrinkage_intensity=float(admitted["shrinkage_intensity"]),  # type: ignore[arg-type]
            variance_floor=float(admitted["variance_floor"]),  # type: ignore[arg-type]
        )
        return recipe, recipe.recipe_hash


__all__ = [
    "ADMISSIBLE_SHRINKAGE",
    "ALTERNATE_ADAPTER_ID",
    "ALTERNATE_PARAMETER_DOMAIN",
    "ALTERNATE_RECIPE_SCHEMA_ID",
    "ShrunkDiagonalAdapter",
    "ShrunkDiagonalCapability",
    "ShrunkDiagonalRecipe",
    "alternate_numerical_environment_hash",
]
