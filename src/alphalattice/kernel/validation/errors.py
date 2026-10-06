"""Typed failures for deterministic Validation protocol capabilities."""

from __future__ import annotations

from typing import Final

from alphalattice.kernel.shared_kernel.domain.errors import RegisteredCodeError

VALIDATION_FAILURE_CODES: Final = frozenset(
    {
        "validation.invalid_split_spec",
        "validation.timeline_invalid",
        "validation.future_data",
        "validation.benchmark_incomplete",
        "validation.insufficient_history",
        "validation.bootstrap_ineligible",
        "validation.attempt_input_mismatch",
        "validation.attempt_budget_exhausted",
        "validation.revision_budget_exhausted",
        "validation.holdout_not_releasable",
        "validation.holdout_already_accessed",
        "validation.holdout_result_conflict",
        "validation.dependency_unavailable",
        "validation.numerical_failure",
        "validation.invalid_evaluation_spec",
        "validation.evidence_input_mismatch",
        "validation.return_source_unavailable",
        "validation.metric_ineligible",
        "validation.cost_model_ineligible",
        "validation.uncertainty_ineligible",
        "validation.holdout_contaminated",
        "validation.artifact_invalid",
        "validation.publication_incomplete",
        "validation.factor_screening_invalid_input",
        "validation.factor_screening_leakage_detected",
        "validation.factor_screening_numerical_failure",
        "validation.factor_screening_artifact_invalid",
        "validation.factor_screening_incomplete",
        "validation.return_model_invalid_input",
        "validation.return_model_leakage_detected",
        "validation.return_model_artifact_invalid",
        "validation.return_model_incomplete",
        "validation.return_model_holdout_mismatch",
        "validation.research_schedule_invalid",
        "validation.research_schedule_ineligible",
        "validation.research_schedule_incomplete",
        "validation.research_schedule_artifact_invalid",
        "validation.portfolio_strategy_invalid_input",
        "validation.portfolio_strategy_dependency_unavailable",
        "validation.portfolio_strategy_target_ineligible",
        "validation.portfolio_strategy_return_unavailable",
        "validation.portfolio_strategy_metric_ineligible",
        "validation.portfolio_strategy_uncertainty_ineligible",
        "validation.portfolio_strategy_base_ineligible",
        "validation.portfolio_strategy_holdout_mismatch",
        "validation.portfolio_strategy_artifact_invalid",
        "validation.portfolio_strategy_incomplete",
        "validation.portfolio_overlay_invalid_input",
        "validation.portfolio_overlay_ineligible",
        "validation.portfolio_overlay_artifact_invalid",
    }
)


class ValidationProtocolError(RegisteredCodeError):
    """Non-retryable Validation exception with a stable failure envelope."""

    category = "validation"
    label = "Validation"
    codes = VALIDATION_FAILURE_CODES


__all__ = ["VALIDATION_FAILURE_CODES", "ValidationProtocolError"]
