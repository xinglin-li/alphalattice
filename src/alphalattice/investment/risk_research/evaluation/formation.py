"""Formation-time covariance evaluation primitives."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
from numpy.typing import NDArray

from alphalattice.investment.risk_research.contracts import RiskFormationEvaluation
from alphalattice.investment.risk_research.estimators.contracts import EstimatedCovariance

type FloatArray = NDArray[np.float64]


def evaluate_formation(
    *,
    estimate: EstimatedCovariance,
    next_session: date,
    next_return: FloatArray,
    equal_weights: FloatArray,
    sector_weights: FloatArray,
    previous_matrix: FloatArray | None,
    previous_maximum_eigenvalue: float | None,
    realized_indices: tuple[int, ...] | None = None,
    realized_excluded_listing_ids: tuple[str, ...] | None = None,
    comparison_indices: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    comparison_dropped_previous: tuple[str, ...] | None = None,
    comparison_dropped_current: tuple[str, ...] | None = None,
) -> RiskFormationEvaluation:
    """Evaluate predicted portfolio variance, realized returns and formation diagnostics.

    Args:
        estimate: Covariance estimate on the admitted listing axis.
        next_session: Realization session associated with next_return.
        next_return: Realized return vector aligned to the estimate.
        equal_weights: Equal-weight control vector on that axis.
        sector_weights: Sector control vector on that axis.
        previous_matrix: Optional preceding covariance for relative-change diagnostics.
        previous_maximum_eigenvalue: Optional preceding leading eigenvalue; both previous inputs are
            needed.

    Returns:
        Formation evaluation with Gaussian negative log score, control calibration and optional
        matrix changes.
    """
    matrix = estimate.matrix
    realized_matrix = (
        matrix if realized_indices is None else matrix[np.ix_(realized_indices, realized_indices)]
    )
    equal_predicted = float(equal_weights @ realized_matrix @ equal_weights)
    sector_predicted = float(sector_weights @ realized_matrix @ sector_weights)
    equal_realized = float(equal_weights @ next_return) ** 2
    sector_realized = float(sector_weights @ next_return) ** 2
    trace_ratio = None
    frobenius = None
    leading_delta = None
    if previous_matrix is not None and previous_maximum_eigenvalue is not None:
        current_comparison = matrix
        if comparison_indices is not None:
            current_indices, previous_indices = comparison_indices
            current_comparison = matrix[np.ix_(current_indices, current_indices)]
            previous_matrix = previous_matrix[np.ix_(previous_indices, previous_indices)]
            previous_maximum_eigenvalue = float(np.linalg.eigvalsh(previous_matrix)[-1])
        previous_trace = float(np.trace(previous_matrix))
        current_trace = (
            estimate.diagnostics.trace
            if comparison_indices is None
            else float(np.trace(current_comparison))
        )
        trace_ratio = current_trace / previous_trace
        denominator = float(np.linalg.norm(previous_matrix, ord="fro"))
        frobenius = (
            float(np.linalg.norm(current_comparison - previous_matrix, ord="fro")) / denominator
        )
        current_maximum = (
            estimate.diagnostics.maximum_eigenvalue
            if comparison_indices is None
            else float(np.linalg.eigvalsh(current_comparison)[-1])
        )
        leading_delta = current_maximum / previous_maximum_eigenvalue - 1.0
    score = gaussian_log_score_per_asset(
        estimate, next_return, matrix=realized_matrix if realized_indices is not None else None
    )
    return RiskFormationEvaluation(
        formation_session=estimate.diagnostics.formation_session,
        next_session=next_session,
        matrix_hash=estimate.diagnostics.matrix_hash,
        shrinkage=estimate.diagnostics.shrinkage,
        minimum_eigenvalue=estimate.diagnostics.minimum_eigenvalue,
        maximum_eigenvalue=estimate.diagnostics.maximum_eigenvalue,
        condition_number=estimate.diagnostics.condition_number,
        trace=estimate.diagnostics.trace,
        average_correlation=estimate.diagnostics.average_correlation,
        top_one_eigenvalue_share=estimate.diagnostics.top_one_eigenvalue_share,
        top_five_eigenvalue_share=estimate.diagnostics.top_five_eigenvalue_share,
        annualized_volatility_minimum=estimate.diagnostics.annualized_volatility_minimum,
        annualized_volatility_median=estimate.diagnostics.annualized_volatility_median,
        annualized_volatility_maximum=estimate.diagnostics.annualized_volatility_maximum,
        gaussian_log_score_per_asset=score,
        equal_weight_predicted_variance=equal_predicted,
        equal_weight_realized_squared_return=equal_realized,
        sector_balanced_predicted_variance=sector_predicted,
        sector_balanced_realized_squared_return=sector_realized,
        trace_ratio=trace_ratio,
        frobenius_delta_ratio=frobenius,
        leading_eigenvalue_delta_ratio=leading_delta,
        realized_excluded_listing_ids=realized_excluded_listing_ids,
        comparison_dropped_previous=comparison_dropped_previous,
        comparison_dropped_current=comparison_dropped_current,
    )


def gaussian_log_score_per_asset(
    estimate: EstimatedCovariance, realized_return: FloatArray, *, matrix: FloatArray | None = None
) -> float:
    """Compute the Gaussian negative log-likelihood per asset; lower scores are better.

    Args:
        estimate: Positive-definite covariance estimate and its eigenvalue diagnostics.
        realized_return: Realized return vector aligned to the covariance asset axis.
        matrix: Optional covariance restricted to the realized listing axis.

    Returns:
        One-half of the Gaussian log-normalizer plus quadratic form, divided by asset count.
    """
    eigenvalues, eigenvectors = (
        (estimate.eigenvalues, estimate.eigenvectors) if matrix is None else np.linalg.eigh(matrix)
    )
    projected = eigenvectors.T @ realized_return
    quadratic = float(np.sum(np.square(projected) / eigenvalues))
    log_determinant = float(np.log(eigenvalues).sum())
    dimension = len(eigenvalues)
    return float(
        0.5 * (dimension * math.log(2.0 * math.pi) + log_determinant + quadratic) / dimension
    )
