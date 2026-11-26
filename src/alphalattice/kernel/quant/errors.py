"""Typed failures for deterministic Quant capabilities."""

from __future__ import annotations

from typing import Final

from alphalattice.kernel.shared_kernel.domain.errors import RegisteredCodeError

QUANT_FAILURE_CODES: Final = frozenset(
    {
        "quant.invalid_spec",
        "quant.snapshot_mismatch",
        "quant.return_source_unavailable",
        "quant.insufficient_history",
        "quant.incomplete_window",
        "quant.non_finite_input",
        "quant.volatility_ineligible",
        "quant.covariance_ineligible",
        "quant.numerical_failure",
        "quant.dependency_unavailable",
        "quant.invalid_portfolio_spec",
        "quant.portfolio_input_mismatch",
        "quant.portfolio_covariance_unavailable",
        "quant.portfolio_volatility_unavailable",
        "quant.portfolio_solver_failure",
        "quant.portfolio_constraint_violation",
        "quant.risk_contribution_ineligible",
        "quant.portfolio_dependency_unavailable",
        "quant.parameter_owner_invalid",
        "quant.parameter_policy_violation",
        "quant.integration_input_mismatch",
        "quant.integration_artifact_invalid",
        "quant.integration_incomplete",
        "quant.factor_registry_invalid",
        "quant.factor_input_mismatch",
        "quant.factor_numerical_failure",
        "quant.factor_artifact_invalid",
        "quant.factor_incomplete",
        "quant.return_model_invalid_input",
        "quant.return_model_dependency_unavailable",
        "quant.return_model_numerical_failure",
        "quant.return_model_convergence_failure",
        "quant.forecast_portfolio_invalid_input",
        "quant.forecast_mapping_ineligible",
        "quant.horizon_risk_ineligible",
        "quant.forecast_portfolio_solver_failure",
        "quant.forecast_portfolio_constraint_violation",
        "quant.black_litterman_ineligible",
        "quant.forecast_portfolio_dependency_unavailable",
        "quant.forecast_portfolio_artifact_invalid",
        "quant.forecast_portfolio_incomplete",
        "quant.portfolio_overlay_invalid_input",
        "quant.portfolio_overlay_solver_failure",
        "quant.portfolio_overlay_constraint_violation",
    }
)


class QuantError(RegisteredCodeError):
    """Non-retryable Quant exception with a stable failure envelope."""

    category = "quant"
    label = "Quant"
    codes = QUANT_FAILURE_CODES


quant_failure = QuantError.typed_failure


__all__ = ["QUANT_FAILURE_CODES", "QuantError"]
