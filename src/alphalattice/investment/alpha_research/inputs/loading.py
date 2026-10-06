"""Program-scoped loading over frozen Alpha fold plans and bounded surfaces."""

from __future__ import annotations

from .folds import (
    AlphaCurrentRefitArrays,
    AlphaFoldArrayPlan,
    AlphaFoldArrays,
)
from .surfaces import prepare_alpha_program_array_workspace


def load_alpha_fold_arrays(plan: AlphaFoldArrayPlan, fold_index: int) -> AlphaFoldArrays:
    """Load exactly one fold and enforce causal training-outcome availability."""
    with (
        prepare_alpha_program_array_workspace(plan, train_session_count=None) as workspace,
        workspace.fold_lease(fold_index) as fold,
    ):
        return fold


def prepare_alpha_current_refit_arrays(
    plan: AlphaFoldArrayPlan,
    *,
    train_session_count: int,
) -> AlphaCurrentRefitArrays:
    """Build latest development-authorized training rows and one current feature surface."""
    with prepare_alpha_program_array_workspace(
        plan, train_session_count=train_session_count
    ) as workspace:
        return workspace.prepare_current_refit()


__all__ = ["load_alpha_fold_arrays", "prepare_alpha_current_refit_arrays"]
