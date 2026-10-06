"""Deterministic Alpha model implementations with no Agent runtime dependency."""

from .adapters.regularized_linear import (
    RegularizedLinearAdapter,
    RegularizedLinearFit,
    RegularizedLinearParameters,
    RegularizedLinearSearchDomain,
    build_regularized_linear_recipe,
    build_regularized_linear_search_domain,
    decode_regularized_linear_content,
    fit_regularized_linear,
)
from .catalog import (
    AlphaModelCapabilityIdentity,
    AlphaModelCatalog,
    AlphaModelCatalogBinding,
    build_installed_alpha_model_catalog,
)
from .contracts import (
    AlphaEstimatorContent,
    AlphaFitProvenanceReceipt,
    AlphaModelAdapter,
    AlphaModelFitResult,
    AlphaModelFitSidecar,
    AlphaModelNumericalBinding,
    AlphaModelPredictionResult,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
    AlphaModelStateProjection,
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
)
from .runtime import AlphaModelExecutionResult, AlphaModelRuntimeService

__all__ = [
    "AlphaEstimatorContent",
    "AlphaFitProvenanceReceipt",
    "AlphaModelAdapter",
    "AlphaModelCapabilityIdentity",
    "AlphaModelCatalog",
    "AlphaModelCatalogBinding",
    "AlphaModelExecutionResult",
    "AlphaModelFitResult",
    "AlphaModelFitSidecar",
    "AlphaModelNumericalBinding",
    "AlphaModelPredictionResult",
    "AlphaModelRecipeEnvelope",
    "AlphaModelRuntimeService",
    "AlphaModelSearchDomainEnvelope",
    "AlphaModelStateProjection",
    "BoundAlphaPredictionInput",
    "BoundAlphaTrainingInput",
    "RegularizedLinearAdapter",
    "RegularizedLinearFit",
    "RegularizedLinearParameters",
    "RegularizedLinearSearchDomain",
    "build_installed_alpha_model_catalog",
    "build_regularized_linear_recipe",
    "build_regularized_linear_search_domain",
    "decode_regularized_linear_content",
    "fit_regularized_linear",
]
