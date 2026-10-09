"""Host-owned split, metric, and package policies for Alpha experiments."""

from __future__ import annotations

from ..evaluation.contracts import AlphaMetricPolicy, seal_evaluation_contract
from .contracts import (
    AlphaDevelopmentSplitPolicy,
    AlphaPackageIdentity,
    AlphaSplitPolicy,
    seal_contract,
)


class AlphaExperimentPolicyError(ValueError):
    """An installed numerical environment differs from the frozen policy."""


def load_alpha_split_policy() -> AlphaSplitPolicy:
    return seal_contract(AlphaSplitPolicy, {}, "policy_hash")


def derive_alpha_development_split_policy(
    *, geometry: AlphaSplitPolicy, maturity_lag_sessions: int, session_count: int
) -> AlphaDevelopmentSplitPolicy:
    """The installed geometry, embargoed by the outcome method's own maturity clock.

    A formation whose outcome is still open where a fold boundary falls carries a
    label that partly describes sessions on the other side of it. How many such
    formations exist is a property of the method -- ``maturity_lag_sessions - 1``,
    since the formation maturing exactly at the boundary is already closed -- so
    it is derived from the resolved seal rather than configured, and no caller
    can state an embargo the method does not imply.

    **The step grows with the embargo.** ``ResearchSplitSpec`` requires
    ``step_sessions >= validation_sessions + embargo_sessions``, and that
    invariant is right: a rolling step that did not cover the embargo would move
    the next validation window straight over the sessions the embargo had just
    excluded, so the exclusion would last exactly one fold and mean nothing. The
    frozen geometry steps by its validation length, which is only admissible
    while the embargo is zero. A development run therefore steps by
    ``validation + embargo``; validation windows keep their length and each
    subsequent fold starts one embargo later.

    **The fold count is predicted, not assumed.** Growing the step changes how
    many complete folds fit a fixed axis, so declaring the frozen five would
    either be luck or a lie. This computes the count from the same arithmetic
    ``build_research_split`` walks, and ``build_alpha_split_plan`` then requires
    the splitter to have produced exactly that many -- two independent
    computations of one number, which is a cross-check rather than a relaxation.

    The frozen ``AlphaSplitPolicy`` is untouched. It supplies the geometry and
    keeps its ``Literal[0]`` embargo and its sealed ``policy_hash`` for the Goal
    and current Programs that already carry them.
    """

    if maturity_lag_sessions < 2:
        raise AlphaExperimentPolicyError("ALPHA_DEVELOPMENT_EMBARGO_MATURITY_LAG_INVALID")
    embargo_sessions = maturity_lag_sessions - 1
    step_sessions = geometry.validation_sessions + embargo_sessions
    # The last index a window's embargo may reach, exactly as the splitter
    # computes it: the sealed holdout is removed first, then the purge that
    # protects it.
    final_boundary = session_count - geometry.sealed_holdout_sessions - geometry.purge_sessions
    reach = (
        final_boundary
        - geometry.train_sessions
        - geometry.purge_sessions
        - geometry.validation_sessions
        - embargo_sessions
    )
    expected_complete_folds = 0 if reach < 0 else reach // step_sessions + 1
    if expected_complete_folds < geometry.minimum_folds:
        raise AlphaExperimentPolicyError("ALPHA_DEVELOPMENT_EMBARGOED_AXIS_TOO_SHORT")
    return seal_contract(
        AlphaDevelopmentSplitPolicy,
        {
            "mode": "ROLLING",
            "train_sessions": geometry.train_sessions,
            "purge_sessions": geometry.purge_sessions,
            "validation_sessions": geometry.validation_sessions,
            "step_sessions": step_sessions,
            "embargo_sessions": embargo_sessions,
            "sealed_holdout_sessions": geometry.sealed_holdout_sessions,
            "minimum_folds": geometry.minimum_folds,
            "expected_complete_folds": expected_complete_folds,
        },
        "policy_hash",
    )


def load_alpha_metric_policy() -> AlphaMetricPolicy:
    return seal_evaluation_contract(
        AlphaMetricPolicy,
        {
            "metric_ids": (
                "MAE",
                "MSE",
                "ZERO_RELATIVE_OOS_R2",
                "RANK_IC",
                "ICIR",
                "GROSS_DECILE_SPREAD",
                "FOLD_COVERAGE",
                "FOLD_STABILITY",
                "FORMATION_DECILE_TURNOVER",
            )
        },
        "policy_hash",
    )


def load_alpha_package_identity() -> AlphaPackageIdentity:
    """The packages Alpha's contracts were validated with, as declared: never a gate.

    The installed versions are the environment, recorded beside each fit, and no identity
    compares them (LAWS.md ID6); whether an upgrade moves a number is answered
    by the saved-object replay check.
    """
    return seal_contract(AlphaPackageIdentity, {}, "package_hash")


__all__ = [
    "AlphaExperimentPolicyError",
    "derive_alpha_development_split_policy",
    "load_alpha_metric_policy",
    "load_alpha_package_identity",
    "load_alpha_split_policy",
]
