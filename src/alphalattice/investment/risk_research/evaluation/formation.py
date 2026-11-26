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
    equal_predicted = float(equal_weights @ matrix @ equal_weights)
    sector_predicted = float(sector_weights @ matrix @ sector_weights)
    equal_realized = float(equal_weights @ next_return) ** 2
    sector_realized = float(sector_weights @ next_return) ** 2
    trace_ratio = None
    frobenius = None
    leading_delta = None
    if previous_matrix is not None and previous_maximum_eigenvalue is not None:
        previous_trace = float(np.trace(previous_matrix))
        trace_ratio = estimate.diagnostics.trace / previous_trace
        denominator = float(np.linalg.norm(previous_matrix, ord="fro"))
        frobenius = float(np.linalg.norm(matrix - previous_matrix, ord="fro")) / denominator
        leading_delta = estimate.diagnostics.maximum_eigenvalue / previous_maximum_eigenvalue - 1.0
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
        gaussian_log_score_per_asset=gaussian_log_score_per_asset(estimate, next_return),
        equal_weight_predicted_variance=equal_predicted,
        equal_weight_realized_squared_return=equal_realized,
        sector_balanced_predicted_variance=sector_predicted,
        sector_balanced_realized_squared_return=sector_realized,
        trace_ratio=trace_ratio,
        frobenius_delta_ratio=frobenius,
        leading_eigenvalue_delta_ratio=leading_delta,
    )


def gaussian_log_score_per_asset(
    estimate: EstimatedCovariance, realized_return: FloatArray
) -> float:
    """Compute the Gaussian negative log-likelihood per asset; lower scores are better.

    Args:
        estimate: Positive-definite covariance estimate and its eigenvalue diagnostics.
        realized_return: Realized return vector aligned to the covariance asset axis.

    Returns:
        One-half of the Gaussian log-normalizer plus quadratic form, divided by asset count.
    """
    projected = estimate.eigenvectors.T @ realized_return
    quadratic = float(np.sum(np.square(projected) / estimate.eigenvalues))
    log_determinant = float(np.log(estimate.eigenvalues).sum())
    dimension = estimate.matrix.shape[0]
    return float(
        0.5 * (dimension * math.log(2.0 * math.pi) + log_determinant + quadratic) / dimension
    )
