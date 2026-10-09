"""Declared trial domains, owned by the policy capability that is run in them.

The Stage 6 campaign enumerates a small pre-registered grid per policy; each
domain below says which parameter values that policy admits, and carries its own
identity so a Program records the domain it was admitted against. An axis's fields,
not a sampler, are its identity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from alphalattice.kernel.shared_kernel.identity import canonical_hash


@dataclass(frozen=True, slots=True)
class SearchAxis:
    """One named axis: integer choices, or a bounded float range.

    ``kind`` is closed at two values; its fields are the axis's identity inside
    ``PortfolioTrialDomain.domain_hash``.
    """

    name: str
    kind: Literal["categorical", "float"]
    choices: tuple[int, ...] = ()
    low: float = 0.0
    high: float = 0.0
    log: bool = False

    def __post_init__(self) -> None:
        """Require a named nonempty categorical or valid bounded numerical search axis.

        Raises:
            ValueError: Name/choices are empty, bounds are unordered or logarithmic bounds are
                nonpositive.
        """
        if not self.name:
            raise ValueError("portfolio_strategy_lab.search_axis_unnamed")
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError("portfolio_strategy_lab.search_axis_choices_empty")
            return
        if not self.low < self.high:
            raise ValueError("portfolio_strategy_lab.search_axis_bounds_invalid")
        if self.log and not (self.low > 0.0 and self.high > 0.0):
            # Refused at declaration, not at draw. A log axis with a zero or
            # negative bound is a space no sampler can draw from, so the mistake
            # belongs to whoever declared it -- and every value admitted against
            # it before this check would have been admitted against nothing.
            raise ValueError("portfolio_strategy_lab.search_axis_log_bounds_invalid")

    def admit(self, value: float | int) -> float | int:
        """Admit one value, exactly as given or not at all.

        The axis owns this because the axis owns the space. A value arriving from
        a request, a replayed ledger or a deliberately chosen point was once
        checked for its *name* and never its magnitude, so a top-k outside the
        declared choices or a negative value on a log axis became a policy
        nobody's domain admitted.

        Nothing is clamped, rounded or coerced. Repairing an inadmissible value
        produces a run whose parameters are not the ones anybody asked for, which
        is worse than refusing: it succeeds.

        Conversion is refused along with repair, because ``int(value)`` *is* a
        repair. It read ``20.9`` as the admitted choice ``20``, so a draw the
        domain never contained built a policy and the evidence recorded that
        domain's own hash asserting it had. ``bool`` is excluded explicitly:
        Python makes ``True`` an ``int`` equal to one, so a truth value would
        otherwise be admitted on any axis that happens to offer ``1``.
        """
        if self.kind == "categorical":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("portfolio_strategy_lab.search_axis_value_not_a_choice")
            if value not in self.choices:
                raise ValueError("portfolio_strategy_lab.search_axis_value_not_a_choice")
            return value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("portfolio_strategy_lab.search_axis_value_not_finite")
        admitted_float = float(value)
        if not math.isfinite(admitted_float):
            raise ValueError("portfolio_strategy_lab.search_axis_value_not_finite")
        if not self.low <= admitted_float <= self.high:
            raise ValueError("portfolio_strategy_lab.search_axis_value_out_of_bounds")
        if self.log and admitted_float <= 0.0:
            # Unreachable now that ``__post_init__`` requires positive log bounds,
            # and kept because the two checks answer to different owners: one to
            # whoever declared the axis, this one to whoever drew the value.
            raise ValueError("portfolio_strategy_lab.search_axis_value_not_positive")
        return admitted_float


@dataclass(frozen=True, slots=True)
class FrozenChoiceAxis:
    """One axis whose admissible values are an explicit frozen set of floats.

    ``SearchAxis`` cannot express this: its categorical kind is closed at integer
    choices, and widening it would change the identity of every domain that
    declares one. So a Sector band domain of exactly ``{0.01, 0.02}`` gets its
    own axis type here.
    """

    name: str
    choices: tuple[float, ...]

    def __post_init__(self) -> None:
        """Require a named nonempty unique finite measured choice set.

        Raises:
            ValueError: Name/choices are empty, choices repeat or any choice is nonfinite.
        """
        if not self.name or not self.choices:
            raise ValueError("portfolio_strategy_lab.frozen_choice_axis_invalid")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError("portfolio_strategy_lab.frozen_choice_axis_duplicated")
        if not all(math.isfinite(value) for value in self.choices):
            raise ValueError("portfolio_strategy_lab.frozen_choice_axis_not_finite")

    def admit(self, value: float | int) -> float:
        """Admit only a numeric value exactly equal to one declared measured choice.

        Args:
            value: Numeric proposed choice; booleans are refused.

        Returns:
            Matched choice normalized to float.

        Raises:
            ValueError: Value is not a supported numeric type or does not equal a declared choice.
        """
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("portfolio_strategy_lab.frozen_choice_value_not_a_choice")
        admitted = float(value)
        if not any(admitted == choice for choice in self.choices):
            raise ValueError("portfolio_strategy_lab.frozen_choice_value_not_a_choice")
        return admitted


@dataclass(frozen=True, slots=True)
class PortfolioTrialDomain:
    """The complete bounded space one enumerated Stage 6 capability may be run in.

    Enumerated rather than sampled, deliberately: the Stage 6 comparison is a
    small pre-registered grid over frozen band choices and a bounded
    ``lambda_r``, and a sampler would add an RNG stream to an identity nothing in
    the Campaign needs.
    """

    ordered_axes: tuple[SearchAxis | FrozenChoiceAxis, ...]

    def __post_init__(self) -> None:
        """Require a nonempty ordered domain with unique axis names.

        Raises:
            ValueError: Domain is empty or axis names repeat.
        """
        names = tuple(axis.name for axis in self.ordered_axes)
        if not names or len(set(names)) != len(names):
            raise ValueError("portfolio_strategy_lab.trial_domain_invalid")

    @property
    def domain_hash(self) -> str:
        """Hash ordered axis declarations including measured choices and numerical bounds.

        Returns:
            Canonical domain identity preserving declared axis order.
        """
        return str(
            canonical_hash(
                {
                    "kind": "PortfolioTrialDomain",
                    "ordered_axes": [
                        {
                            "name": axis.name,
                            "kind": "frozen_choice"
                            if isinstance(axis, FrozenChoiceAxis)
                            else axis.kind,
                            "choices": [float(value) for value in axis.choices],
                            "low": 0.0 if isinstance(axis, FrozenChoiceAxis) else axis.low,
                            "high": 0.0 if isinstance(axis, FrozenChoiceAxis) else axis.high,
                            "log": False if isinstance(axis, FrozenChoiceAxis) else axis.log,
                        }
                        for axis in self.ordered_axes
                    ],
                }
            )
        )

    def admit(self, parameters: Mapping[str, float | int]) -> dict[str, float | int]:
        """Admit exactly the declared parameter keys through each axis owner.

        Args:
            parameters: Proposed numeric mapping with exactly the declared axis names.

        Returns:
            Validated parameter mapping in declared axis order.

        Raises:
            ValueError: Keys differ or an axis refuses its proposed value.
        """
        declared = {axis.name: axis for axis in self.ordered_axes}
        if set(parameters) != set(declared):
            raise ValueError("portfolio_strategy_lab.development_parameters_not_in_domain")
        return {axis.name: axis.admit(parameters[axis.name]) for axis in self.ordered_axes}


SECTOR_BAND_AXES = (
    FrozenChoiceAxis(name="sector_absolute_deviation", choices=(0.01, 0.02)),
    FrozenChoiceAxis(name="sector_relative_deviation", choices=(0.25, 0.50)),
)
"""The Sector safety-band domain, frozen before any Portfolio outcome was seen."""


def stage_six_trial_domains() -> dict[str, PortfolioTrialDomain]:
    """The bounded domain each Stage 6 capability declares, keyed by policy id.

    Owned here rather than by the Campaign: the capability knows what it can be
    run with, and a runner that invented ranges would be describing a space
    nobody installed.
    """
    shared_pool = (
        SearchAxis(name="top_k", kind="categorical", choices=(50, 75, 100)),
        SearchAxis(name="cap_multiplier", kind="float", low=1.25, high=3.0, log=True),
    )
    return {
        "CURRENT_UNIVERSE_EQUAL_WEIGHT_EVERY_FORMATION": PortfolioTrialDomain(
            ordered_axes=(SearchAxis(name="top_k", kind="categorical", choices=(0,)),)
        ),
        "STRATIFIED_TOP_K_EQUAL_WEIGHT": PortfolioTrialDomain(
            ordered_axes=(*shared_pool, *SECTOR_BAND_AXES)
        ),
        # The covariance-blind control. The same pool and cap as the policies it
        # is a control for, and no axis of its own: a control with a knob stops
        # being one, because any difference it showed across Risk arms could be
        # blamed on the knob instead of on the arm.
        "TOP_K_EQUAL_WEIGHT": PortfolioTrialDomain(ordered_axes=shared_pool),
        "TOP_K_MINIMUM_VARIANCE": PortfolioTrialDomain(ordered_axes=shared_pool),
        "TOP_K_SCORE_RISK_COST": PortfolioTrialDomain(
            ordered_axes=(
                *shared_pool,
                SearchAxis(name="risk_aversion", kind="float", low=1e2, high=1e5, log=True),
                SearchAxis(
                    name="turnover_regularization", kind="float", low=1e-3, high=10.0, log=True
                ),
                # What one unit of score is worth. Declared wide, and with zero
                # admitted, because the conversion is exactly what a trial
                # measures when the score carries no units of its own.
                #
                # Zero is the most informative single point in the domain rather
                # than a degenerate one: ``stable_top_k`` selects the pool from
                # the *unscaled* score before this coefficient is applied, so
                # zero means "take the top K by score, then minimum-variance
                # them" -- which separates what the score is worth for choosing
                # names from what it is worth for sizing them.
                FrozenChoiceAxis(
                    name="score_utility_coefficient",
                    choices=(0.0, 1e-3, 1e-2, 1e-1, 1.0),
                ),
            )
        ),
        "RETURN_SCALED_TOTAL_SIGNAL_GLOBAL_QP": PortfolioTrialDomain(
            ordered_axes=(
                *shared_pool,
                *SECTOR_BAND_AXES,
                # No universal value is assumed. The bound is wide because the
                # stock component has return units and no calibration, so the
                # conversion to variance is exactly what the trial measures.
                SearchAxis(name="risk_aversion", kind="float", low=1e0, high=1e6, log=True),
                FrozenChoiceAxis(name="transaction_cost_rate", choices=(0.001,)),
                FrozenChoiceAxis(name="beta_guardrail", choices=(0.10, 0.20)),
            )
        ),
        # One external immutable package, not a search surface.  Its four
        # singleton axes make the existing request/Program point owner state
        # every parameter rather than accept a code default.
        "WHOLE_BOOK_HYSTERESIS_EQUAL_WEIGHT": PortfolioTrialDomain(
            ordered_axes=(
                FrozenChoiceAxis(name="top_k", choices=(50,)),
                FrozenChoiceAxis(name="exit_rank", choices=(150,)),
                FrozenChoiceAxis(name="cadence", choices=(3,)),
                FrozenChoiceAxis(name="burn_in", choices=(126,)),
            )
        ),
        "WHOLE_BOOK_HYSTERESIS_INVERSE_VOLATILITY": PortfolioTrialDomain(
            ordered_axes=(
                FrozenChoiceAxis(name="top_k", choices=(50,)),
                FrozenChoiceAxis(name="exit_rank", choices=(150,)),
                FrozenChoiceAxis(name="cadence", choices=(3,)),
                FrozenChoiceAxis(name="burn_in", choices=(126,)),
            )
        ),
        "WHOLE_BOOK_HYSTERESIS_CAUSAL_RANK_MU_DIAGONAL_TILT": PortfolioTrialDomain(
            ordered_axes=(
                FrozenChoiceAxis(name="top_k", choices=(50,)),
                FrozenChoiceAxis(name="exit_rank", choices=(100,)),
                FrozenChoiceAxis(name="cadence", choices=(3,)),
                FrozenChoiceAxis(name="burn_in", choices=(378,)),
                FrozenChoiceAxis(name="kappa", choices=(0.03,)),
                FrozenChoiceAxis(name="bucket_count", choices=(20,)),
                FrozenChoiceAxis(name="lookback", choices=(252,)),
                FrozenChoiceAxis(name="maximum_weight", choices=(0.06,)),
            )
        ),
    }


def installed_parameter_domains() -> dict[str, SearchAxis | FrozenChoiceAxis]:
    """The axes every declaring capability declares *identically*, keyed by name.

    Collected from the capabilities that own the domains rather than restated
    here, so a request is admitted against what is installed and adding an axis
    is one edit in its own owner.

    An axis two capabilities declare differently is deliberately **absent**.
    ``top_k`` is the real case: the anchor selects nothing and declares ``{0}``
    while the pooled methods declare ``{50, 75, 100}``, so "inside the domain"
    has no capability-independent answer and a request keyed only by axis name
    could not be checked against one. Those names are reported separately so a
    request naming them is refused for the right reason rather than as an axis
    nobody installed.
    """
    collected: dict[str, SearchAxis | FrozenChoiceAxis] = {}
    for name in sorted(set(_declared_axes()) - capability_specific_axis_names()):
        collected[name] = _declared_axes()[name][0]
    return collected


def capability_specific_axis_names() -> frozenset[str]:
    """Axis names whose meaning depends on which capability declares them."""
    return frozenset(
        name for name, axes in _declared_axes().items() if any(value != axes[0] for value in axes)
    )


def _declared_axes() -> dict[str, list[SearchAxis | FrozenChoiceAxis]]:
    declared: dict[str, list[SearchAxis | FrozenChoiceAxis]] = {}
    for domain in stage_six_trial_domains().values():
        for axis in domain.ordered_axes:
            declared.setdefault(axis.name, []).append(axis)
    return declared


__all__ = [
    "SECTOR_BAND_AXES",
    "FrozenChoiceAxis",
    "PortfolioTrialDomain",
    "SearchAxis",
    "capability_specific_axis_names",
    "installed_parameter_domains",
    "stage_six_trial_domains",
]
