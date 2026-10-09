"""The installed Portfolio time-series split policy, and the one geometry it owns.

The anchored walk-forward's geometry used to live in three places at once: the
regularization runtime built folds from a literal tuple, the contract validated them
against a second literal dict, and this module restated a third for naming. Three
copies of one geometry are three chances to disagree, and nothing would have caught
it. So the spec lives here once and every consumer reads it from here. This is not a
cross-validation framework: it declares no rule that generates folds from an axis
length, and a policy that is not written out below cannot be resolved.

One policy is installed: ``PORTFOLIO_ANCHORED_THREE_FOLD``, the frozen
746-observation geometry the regularization program used, which its kept record
contracts still validate against.
"""

from __future__ import annotations

from typing import NamedTuple

PORTFOLIO_ANCHORED_THREE_FOLD = "PORTFOLIO_ANCHORED_THREE_FOLD"
"""The frozen anchored walk-forward the regularization program has always used."""

RESEARCH_SESSION_COUNT = 746
"""Observations on the frozen policy's axis. Named for the consumers that assert it."""


class PortfolioSplitPolicyError(ValueError):
    """Stable fail-closed boundary for an uninstalled or mismatched split policy."""


class AnchoredFoldGeometry(NamedTuple):
    """One fold, in both index spaces a consumer needs.

    The global indices are positions in the mandate's own session axis; the local
    ones are positions in the *observation* axis the policy runs on. The two
    differ wherever a policy excludes a global index, and carrying only one of
    them is what forced each consumer to re-derive the other.
    """

    fold_index: int
    training_global_end: int
    embargo_global_index: int
    validation_global_start: int
    validation_global_end: int
    training_local_stop: int
    validation_local_start: int
    validation_local_stop: int

    @property
    def validation_length(self) -> int:
        """Read validation length in the observation-index space.

        Returns:
            Exclusive local stop minus local start.
        """
        return self.validation_local_stop - self.validation_local_start


class _InstalledPolicy(NamedTuple):
    observation_count: int
    excluded_global_indices: tuple[int, ...]
    specs: tuple[tuple[int, int, int, int, int], ...]
    """``(fold, training_end, embargo, validation_start, validation_end)``, inclusive."""


_INSTALLED: dict[str, _InstalledPolicy] = {
    PORTFOLIO_ANCHORED_THREE_FOLD: _InstalledPolicy(
        observation_count=RESEARCH_SESSION_COUNT,
        # Global 494 is the program's own embargo between the development and
        # policy-OOS partitions: a session that exists on the mandate axis and is
        # deliberately not an observation.
        excluded_global_indices=(494,),
        specs=(
            (1, 367, 368, 369, 493),
            (2, 493, 494, 495, 620),
            (3, 620, 621, 622, 746),
        ),
    ),
}
"""Every installed policy, written out, pre-registered before any outcome."""


def research_global_indices(split_policy_id: str) -> tuple[int, ...]:
    """The mandate-axis position of each observation, in observation order."""
    policy = _policy(split_policy_id)
    excluded = set(policy.excluded_global_indices)
    total = policy.observation_count + len(excluded)
    return tuple(index for index in range(total) if index not in excluded)


def anchored_fold_geometry(split_policy_id: str) -> tuple[AnchoredFoldGeometry, ...]:
    """Resolve one policy's folds in both index spaces.

    The single owner of this arithmetic: the regularization record contracts read it
    from here, so a geometry change is one edit rather than several kept in step.
    """
    policy = _policy(split_policy_id)
    position = {
        value: index for index, value in enumerate(research_global_indices(split_policy_id))
    }
    folds: list[AnchoredFoldGeometry] = []
    for fold, training_end, embargo, start, stop in policy.specs:
        if not training_end < embargo < start <= stop:
            raise PortfolioSplitPolicyError(
                f"portfolio_strategy_lab.split_policy_fold_geometry_invalid:{fold}"
            )
        if embargo - training_end != 1:
            raise PortfolioSplitPolicyError(
                f"portfolio_strategy_lab.split_policy_embargo_invalid:{fold}"
            )
        try:
            folds.append(
                AnchoredFoldGeometry(
                    fold_index=fold,
                    training_global_end=training_end,
                    embargo_global_index=embargo,
                    validation_global_start=start,
                    validation_global_end=stop,
                    training_local_stop=position[training_end] + 1,
                    validation_local_start=position[start],
                    validation_local_stop=position[stop] + 1,
                )
            )
        except KeyError as error:
            raise PortfolioSplitPolicyError(
                f"portfolio_strategy_lab.split_policy_fold_outside_axis:{fold}"
            ) from error
    return tuple(folds)


def _policy(split_policy_id: str) -> _InstalledPolicy:
    policy = _INSTALLED.get(split_policy_id)
    if policy is None:
        raise PortfolioSplitPolicyError(
            f"portfolio_strategy_lab.split_policy_not_installed:{split_policy_id}"
        )
    return policy


__all__ = [
    "PORTFOLIO_ANCHORED_THREE_FOLD",
    "RESEARCH_SESSION_COUNT",
    "AnchoredFoldGeometry",
    "PortfolioSplitPolicyError",
    "anchored_fold_geometry",
    "research_global_indices",
]
