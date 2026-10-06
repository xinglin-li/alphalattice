"""Pure evidence compilation and terminal candidate selection."""

from __future__ import annotations

from alphalattice.investment.portfolio_strategy_lab.contracts import (
    PortfolioEvidenceDossier,
    PortfolioExperimentProgram,
    PortfolioResearchReview,
    PortfolioTrialLedger,
    TrialState,
)


class PortfolioEvidenceError(ValueError):
    """Stable failure at the deterministic evidence boundary."""


def validate_terminal_evidence(
    *,
    program: PortfolioExperimentProgram,
    ledger: PortfolioTrialLedger,
    dossier: PortfolioEvidenceDossier,
    review: PortfolioResearchReview,
) -> None:
    """Require a complete initial or admitted structural attempt family."""
    structural = review.structural_experiment_executed
    additional = program.structural_attempt_budget if structural else 0
    expected_attempts = program.initial_attempt_budget + additional
    expected_phase = "COMPLETE_PROGRAM" if structural else "INITIAL_SEARCH"
    completed = sum(value.state is TrialState.COMPLETED for value in ledger.entries)
    failed = len(ledger.entries) - completed
    if (
        ledger.program_hash != program.program_hash
        or ledger.phase != expected_phase
        or ledger.attempt_budget != expected_attempts
        or len(ledger.entries) != expected_attempts
        or review.additional_attempt_count != additional
        or dossier.program_hash != program.program_hash
        or dossier.attempted_count != expected_attempts
        or dossier.completed_count != completed
        or dossier.failed_count != failed
    ):
        raise PortfolioEvidenceError("portfolio_strategy_lab.terminal_ledger_incomplete")


__all__ = ["PortfolioEvidenceError", "validate_terminal_evidence"]
