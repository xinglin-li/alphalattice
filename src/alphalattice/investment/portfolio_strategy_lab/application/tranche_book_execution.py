"""Carry sleeve books across formations and let the shared engine own everything else.

The tranche adapter is a pure function of one formation. Something has to hold
the sleeves between formations, and this is that something -- deliberately the
executor rather than the policy, so the adapter neither derives nor stores
state it was not handed.

**It implements no drift, no fill, no turnover and no cost.** Those belong to
`portfolio_backtesting`, and the way this module stays out of them is worth
stating because it is the whole design: it never computes a return. The engine
hands it the executed book and the drifted book at each formation, and the
sleeves are *projected* onto those two vectors. A second drift implementation
here would be a second numerical path that agrees with the engine right up until
it does not.

The installed product recipe explicitly resets sleeve notionals to `1/T` at
each formation.  The engine still owns drift: this executor projects each
sleeve onto the drifted book, while the policy normalizes each projected sleeve
before applying the equal-notional recipe identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    BoolArray,
    FloatArray,
    PortfolioTargetDecision,
    PortfolioWalkForwardError,
)
from alphalattice.investment.alpha_research.scores.product_replay import (
    AlphaProductScoreProjection,
)
from alphalattice.investment.portfolio_strategy_lab.policies.buffered_equal_weight import (
    whole_book_hysteresis_selection,
)
from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
    BoundPolicyDecisionInput,
    CausalRankReturnCurveSlice,
)
from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import (
    FrozenSleeveWeightRule,
    TrancheBookRecipe,
    cap_and_renormalise,
    decide_tranche_book,
    due_sleeves,
    rank_bucket_mu,
    sleeve_target_weights,
)
from alphalattice.investment.risk_research.surfaces.decomposition import (
    RiskAllocationProjection,
    RiskAttributionProjection,
)

_MASS_TOLERANCE: Final = 1e-12


class TrancheExecutionError(PortfolioWalkForwardError):
    """Stable refusal for a tranche execution state or input failure."""


class EligiblePoolShort(TrancheExecutionError):
    """A walk refused before its book opens, by a formation too short for a rebalance.

    The formation has fewer tradable, scored names than a rebalance selects. It is the owner's
    stop, by that session, not a failure a resume clears: the same resolution refuses the same
    way.
    """


@dataclass(frozen=True, slots=True)
class TrancheFormationInputs:
    """Everything one formation needs, resolved by an owner before execution starts.

    The provider takes resolved arrays rather than a resolver, because a
    decision provider that could reach a store would be able to change what it
    consumes midway through a walk and no identity would record it.
    """

    formation_session: date
    scores: FloatArray
    decision_eligible: BoolArray
    risk_allocation: RiskAllocationProjection | None
    """The admitted per-name risk lane for this formation, or ``None`` for ``ew``.

    A projection rather than a covariance. The public path holds no `N x N`
    matrix at any point, and the executor is where that would have leaked back in
    -- taking a diagonal here would look identical downstream while quietly
    reintroducing an input nothing in this path validates.
    """

    causal_rank_return_curve: CausalRankReturnCurveSlice | None
    risk_attribution: RiskAttributionProjection | None = None
    """Report-only factor projection paired to ``risk_allocation``.

    It is validated here and consumed only after the policy has produced a
    target.  It never enters ``BoundPolicyDecisionInput``.
    """

    score_projection: AlphaProductScoreProjection | None = None
    """The exact aggregated score object these inputs carry, when replayed.

    The object travels with its values so the executor can recompute the identity
    immediately before the policy reads them. A detached hash beside a mutable
    score array would be provenance prose rather than a binding.
    """


def project_sleeves_onto_book(
    *,
    sleeves: FloatArray,
    reference_weights: FloatArray,
    pretrade_weights: FloatArray,
) -> FloatArray:
    """Re-express the carried sleeves against the book the engine actually holds.

    Two corrections happen here and neither is a return calculation.

    First a *reconciliation*: the executed book can differ from the target the
    sleeves were built to, because a fill can come up short. The shortfall is
    absorbed proportionally by whichever sleeves hold that name, which is the
    only attribution available when sleeves overlap and is at least unbiased
    between them.

    Then a *projection*: the engine's drifted book divided by its executed book
    is the per-name growth factor, up to one scalar that the renormalisation
    below cancels. Reading it from the engine is what keeps drift in exactly one
    place; recomputing it from returns would put a second copy here.
    """
    if sleeves.ndim != 2 or sleeves.shape[1] != reference_weights.size:
        raise TrancheExecutionError("portfolio_strategy_lab.tranche_execution_axis_invalid")
    if pretrade_weights.shape != reference_weights.shape:
        raise TrancheExecutionError("portfolio_strategy_lab.tranche_execution_book_axis_invalid")

    carried = np.asarray(sleeves, dtype=np.float64).copy()
    held = carried.sum(axis=0)
    executed = np.asarray(reference_weights, dtype=np.float64)
    reconcile = np.divide(
        executed,
        held,
        out=np.ones(executed.size, dtype=np.float64),
        where=held > _MASS_TOLERANCE,
    )
    # A name the sleeves do not hold cannot be attributed to any of them, so it
    # is left out rather than spread across sleeves that never chose it.
    reconcile[held <= _MASS_TOLERANCE] = 0.0
    carried = carried * reconcile[None, :]

    drifted = np.asarray(pretrade_weights, dtype=np.float64)
    growth = np.divide(
        drifted,
        executed,
        out=np.ones(executed.size, dtype=np.float64),
        where=np.abs(executed) > _MASS_TOLERANCE,
    )
    carried = carried * growth[None, :]

    total = float(carried.sum())
    if total <= _MASS_TOLERANCE:
        raise TrancheExecutionError("portfolio_strategy_lab.tranche_execution_book_empty")
    return np.asarray(carried / total, dtype=np.float64)


def rebuild_sleeve_state(
    *,
    sleeve_weights: tuple[FloatArray, ...],
    sleeve_shares: tuple[float, ...],
    target_weights: FloatArray,
) -> FloatArray:
    """Turn one allocation back into book-scale sleeves that sum to the target.

    The aggregate name cap is applied to the assembled book, so when it binds the
    sleeves must be scaled by the same per-name ratio. Without that feedback the
    carried state would keep claiming weight the book does not hold, and the two
    would drift apart silently over a walk.
    """
    scaled = np.vstack(
        [
            share * np.asarray(book, dtype=np.float64)
            for share, book in zip(sleeve_shares, sleeve_weights, strict=True)
        ]
    )
    uncapped = scaled.sum(axis=0)
    target = np.asarray(target_weights, dtype=np.float64)
    ratio = np.divide(
        target,
        uncapped,
        out=np.ones(target.size, dtype=np.float64),
        where=uncapped > _MASS_TOLERANCE,
    )
    return np.asarray(scaled * ratio[None, :], dtype=np.float64)


class _SleeveBookProvider:
    """What every sleeve book shares, and none of what makes them different.

    The carried state, the lanes a ledger reads back, the hold branch and the
    projection of the sleeves onto the engine's two books live here once. The
    selection arithmetic does not: the shared tranche book that resets every
    reviewed sleeve to ``1/T`` and the capped book that preserves drifted sleeve
    notional keep their own decision bodies verbatim, because those are the
    frozen policies and this class is only the bookkeeping they duplicated.
    """

    def __init__(
        self,
        *,
        formations: tuple[TrancheFormationInputs, ...],
        ordered_listing_ids: tuple[str, ...],
        initial_sleeve_weights: FloatArray | None,
        schedule_offset: int,
    ) -> None:
        if not formations:
            raise TrancheExecutionError("portfolio_strategy_lab.tranche_execution_no_formations")
        listings = len(ordered_listing_ids)
        if any(item.scores.size != listings for item in formations):
            raise TrancheExecutionError("portfolio_strategy_lab.tranche_execution_axis_invalid")
        if schedule_offset < 0:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_schedule_offset_invalid"
            )
        carried: FloatArray | None = None
        if initial_sleeve_weights is not None:
            carried = np.asarray(initial_sleeve_weights, dtype=np.float64)
            if carried.ndim != 2 or carried.shape[1] != listings:
                raise TrancheExecutionError(
                    "portfolio_strategy_lab.tranche_execution_sleeve_carry_axis_invalid"
                )
            if (
                not np.isfinite(carried).all()
                or bool(np.any(carried < 0.0))
                or float(carried.sum()) <= _MASS_TOLERANCE
            ):
                raise TrancheExecutionError(
                    "portfolio_strategy_lab.tranche_execution_sleeve_carry_invalid"
                )
            # Carried verbatim. A sealed state already sums to one within a few
            # ULP, and dividing it by its own mass would move exactly the bits a
            # continuation has to reproduce.
        self._formations = formations
        self._ordered_listing_ids = ordered_listing_ids
        # How many formations the policy had already decided when this segment
        # opened. Zero for a fresh book; the carried count for a continuation.
        self._schedule_offset = schedule_offset
        # The one piece of state that survives a formation. A continuation that
        # restored the book but not the sleeves would re-express the strategy
        # from scratch against a book it inherited, which is a different
        # strategy than the one that was frozen.
        self._sleeves: FloatArray | None = carried
        self._aggregate_cap_binding_counts: list[int] = []
        self._uncapped_targets: list[FloatArray] = []
        self._targets: list[FloatArray] = []
        self._risk_projections: list[str] = []
        self._score_projections: list[str] = []

    @property
    def formation_sessions(self) -> tuple[date, ...]:
        """The resolved input axis this provider will consume."""

        return tuple(item.formation_session for item in self._formations)

    @property
    def schedule_offset(self) -> int:
        """The schedule position this segment opened on."""

        return self._schedule_offset

    @property
    def sleeve_state(self) -> FloatArray | None:
        """The carried sleeves: inspection, readback, and sealing a path boundary.

        Copied on the way out, because a continuation seals this array and the
        caller must not be able to reach back into the provider's live state.
        """

        return None if self._sleeves is None else self._sleeves.copy()

    @property
    def consumed_risk_projection_hashes(self) -> tuple[str, ...]:
        """Every risk lane this walk actually consumed, in formation order."""

        return tuple(self._risk_projections)

    @property
    def consumed_score_projection_hashes(self) -> tuple[str, ...]:
        """Every replay score identity accepted before a book decision."""

        return tuple(self._score_projections)

    @property
    def aggregate_cap_binding_counts(self) -> tuple[int, ...]:
        """Names moved by the aggregate ceiling on each decided formation."""

        return tuple(self._aggregate_cap_binding_counts)

    @property
    def uncapped_targets_by_formation(self) -> tuple[FloatArray, ...]:
        """Pre-cap books retained for identity-bound successor replay."""

        return tuple(value.copy() for value in self._uncapped_targets)

    @property
    def target_weights_by_formation(self) -> tuple[FloatArray, ...]:
        """The exact books handed to the execution owner, capped and all.

        Sealed by the ledger because it is the *input* to the transition a
        strong replay reruns. The pre-cap lane above cannot stand in for it: on
        any formation where the aggregate ceiling bound, the two differ, and a
        replay driven by the pre-cap book would rederive fills that were never
        placed.
        """

        return tuple(value.copy() for value in self._targets)

    def _admit_score_lane(self, resolved: TrancheFormationInputs) -> None:
        projection = resolved.score_projection
        if projection is None:
            return
        try:
            projection.verify_content()
        except ValueError as error:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_score_projection_invalid"
            ) from error
        if projection.formation_session != resolved.formation_session:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_score_session_mismatch"
            )
        if projection.ordered_listing_ids != self._ordered_listing_ids:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_score_listing_axis_mismatch"
            )
        if not np.array_equal(resolved.scores, projection.scores, equal_nan=True):
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_score_values_mismatch"
            )
        if not np.array_equal(resolved.decision_eligible, projection.live):
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_score_support_mismatch"
            )
        self._score_projections.append(projection.projection_hash)

    def _open(
        self,
        *,
        formation_index: int,
        reference_weights: FloatArray,
        pretrade_weights: FloatArray,
        decision_mode: Literal["REBALANCE", "HOLD"],
    ) -> PortfolioTargetDecision | TrancheFormationInputs:
        """Re-express the sleeves against the engine's books, then hold or resolve.

        A hold is not a decision. The drifted book is carried unchanged and the
        sleeves are re-expressed against it, so the next review starts from what
        is actually held rather than from a stale target.
        """

        if self._sleeves is not None:
            self._sleeves = project_sleeves_onto_book(
                sleeves=self._sleeves,
                reference_weights=reference_weights,
                pretrade_weights=pretrade_weights,
            )
        if decision_mode == "HOLD":
            held = np.asarray(pretrade_weights, dtype=np.float64)
            self._targets.append(np.array(held, copy=True))
            return PortfolioTargetDecision(
                target_weights=held,
                predicted_variance=None,
                decision_mode="HOLD",
                requires_risk_forecast=False,
            )
        if formation_index < 0 or formation_index >= len(self._formations):
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_formation_invalid"
            )
        resolved = self._formations[formation_index]
        self._admit_score_lane(resolved)
        return resolved


class TrancheBookDecisionProvider(_SleeveBookProvider):
    """The stateful seam the engine calls once per formation.

    State is exactly one array: book-scale sleeve weights. Everything else it
    needs is resolved up front and immutable.
    """

    def __init__(
        self,
        *,
        recipe: TrancheBookRecipe,
        formations: tuple[TrancheFormationInputs, ...],
        ordered_listing_ids: tuple[str, ...],
        sector_exposure_matrix: FloatArray,
        equal_weight_sector_exposure: FloatArray,
        initial_sleeve_weights: FloatArray | None = None,
        schedule_offset: int = 0,
        attribution_required: bool = True,
    ) -> None:
        """Bind one tranche recipe to ordered score, Risk and sector formation inputs.

        Args:
            recipe: Admitted tranche policy recipe.
            formations: Ordered formation score/Risk projections.
            ordered_listing_ids: Exact common listing axis.
            sector_exposure_matrix: Sector exposure on that axis.
            equal_weight_sector_exposure: Sector reference for deviations.
            initial_sleeve_weights: Optional prior sleeve state.
            schedule_offset: Offset into the declared review schedule.
            attribution_required: Require Risk attribution when admitting each projection.

        Raises:
            TrancheExecutionError: A supplied Risk allocation or attribution listing axis differs.
        """
        super().__init__(
            formations=formations,
            ordered_listing_ids=ordered_listing_ids,
            initial_sleeve_weights=initial_sleeve_weights,
            schedule_offset=schedule_offset,
        )
        self._attribution_required = attribution_required
        """Whether each Risk lane comes paired with its factor attribution.

        The installed strategies' Risk is a factor model, whose report reads the attribution
        beside the holdings; a research study weighing by a development covariance's volatility
        has no factor block, so it carries none.
        """
        if any(
            item.risk_allocation is not None
            and tuple(item.risk_allocation.ordered_listing_ids) != ordered_listing_ids
            for item in formations
        ):
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_risk_listing_axis_mismatch"
            )
        if any(
            item.risk_attribution is not None
            and tuple(item.risk_attribution.ordered_listing_ids) != ordered_listing_ids
            for item in formations
        ):
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_risk_listing_axis_mismatch"
            )
        self._recipe = recipe
        self._sector_exposure_matrix = sector_exposure_matrix
        self._equal_weight_sector_exposure = equal_weight_sector_exposure
        self._reviewed: list[tuple[int, tuple[int, ...]]] = []
        self._sleeve_shares: list[tuple[float, ...]] = []
        self._risk_recipe_hash: str | None = None
        self._risk_attributions: list[RiskAttributionProjection] = []

    def _admit_risk_lane(self, resolved: TrancheFormationInputs) -> None:
        """Checks only the executor can make, because only it knows the session.

        The policy validates the axis and the values it is handed. Whether that
        projection belongs to *this* formation, and whether it was produced under
        the Risk recipe this run declared, are facts about the run rather than
        about the array, so they are checked here.
        """

        projection = resolved.risk_allocation
        if not self._recipe.consumes_risk:
            if projection is not None or resolved.risk_attribution is not None:
                raise TrancheExecutionError(
                    "portfolio_strategy_lab.tranche_execution_risk_lane_not_consumed"
                )
            return
        if projection is None:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_risk_projection_absent"
            )
        attribution = resolved.risk_attribution
        if attribution is None and self._attribution_required:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_risk_attribution_absent"
            )
        try:
            projection.verify_content()
        except ValueError as error:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_risk_projection_invalid"
            ) from error
        if projection.formation_session != resolved.formation_session:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_risk_session_mismatch"
            )
        if attribution is not None and (
            attribution.formation_session != resolved.formation_session
            or attribution.surface_hash != projection.surface_hash
            or attribution.recipe_hash != projection.recipe_hash
            or attribution.ordered_listing_ids != projection.ordered_listing_ids
        ):
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_risk_projection_pair_mismatch"
            )
        if self._risk_recipe_hash is None:
            self._risk_recipe_hash = projection.recipe_hash
        elif projection.recipe_hash != self._risk_recipe_hash:
            # One walk, one Risk recipe. A mid-walk change would silently splice
            # two representations into a single holdings ledger.
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_risk_recipe_changed"
            )

    def _admit_curve_lane(self, resolved: TrancheFormationInputs, *, formation_index: int) -> None:
        curve = resolved.causal_rank_return_curve
        if not self._recipe.consumes_mu:
            if curve is not None:
                raise TrancheExecutionError(
                    "portfolio_strategy_lab.tranche_execution_curve_not_consumed"
                )
            return
        if curve is None:
            raise TrancheExecutionError("portfolio_strategy_lab.tranche_execution_curve_absent")
        if curve.formation_index != formation_index:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_curve_formation_mismatch"
            )
        if curve.formation_session != resolved.formation_session:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_curve_session_mismatch"
            )

    @property
    def consumed_risk_attributions(self) -> tuple[RiskAttributionProjection, ...]:
        """Report-only projections in decision order, never fed back into policy."""
        return tuple(self._risk_attributions)

    @property
    def reviewed_sleeves_by_formation(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        """Read retained sleeve reviews in executed formation order.

        Returns:
            Ordered pairs of formation index and reviewed sleeve indices.
        """
        return tuple(self._reviewed)

    @property
    def sleeve_shares_by_formation(self) -> tuple[tuple[float, ...], ...]:
        """The recipe-stated shares actually used for each allocation."""
        return tuple(self._sleeve_shares)

    def __call__(
        self,
        *,
        formation_index: int,
        reference_weights: FloatArray,
        pretrade_weights: FloatArray,
        decision_mode: Literal["REBALANCE", "HOLD"],
    ) -> PortfolioTargetDecision:
        """Advance retained sleeves with admitted Risk/curve inputs or preserve a hold.

        Args:
            formation_index: Exact formation position on the retained source axis.
            reference_weights: Holdings reference for turnover and frozen carry.
            pretrade_weights: Drifted holdings before this decision.
            decision_mode: Explicit rebalance or hold disposition.

        Returns:
            Target decision plus retained review, cap and source-consumption history; this provider
            does not request an additional Risk forecast.

        Raises:
            TrancheExecutionError: Formation, prior state or exact Risk/curve admission fails.
        """
        opened = self._open(
            formation_index=formation_index,
            reference_weights=reference_weights,
            pretrade_weights=pretrade_weights,
            decision_mode=decision_mode,
        )
        if isinstance(opened, PortfolioTargetDecision):
            return opened
        resolved = opened
        carried = None if self._sleeves is None else tuple(self._sleeves)
        self._admit_risk_lane(resolved)
        self._admit_curve_lane(resolved, formation_index=formation_index)
        inputs = BoundPolicyDecisionInput(
            scores=resolved.scores,
            covariance=None,
            decision_eligible=resolved.decision_eligible,
            risk_allocation=resolved.risk_allocation,
            reference_weights=np.asarray(reference_weights, dtype=np.float64),
            # A Sector history's per-session lanes read the formation's.
            sector_exposure_matrix=(
                self._sector_exposure_matrix[formation_index]
                if self._sector_exposure_matrix.ndim == 3
                else self._sector_exposure_matrix
            ),
            equal_weight_sector_exposure=(
                self._equal_weight_sector_exposure[formation_index]
                if self._equal_weight_sector_exposure.ndim == 2
                else self._equal_weight_sector_exposure
            ),
            ordered_listing_ids=self._ordered_listing_ids,
            formation_index=formation_index,
            schedule_position=self._schedule_offset + formation_index,
            formation_session=resolved.formation_session,
            causal_rank_return_curve=resolved.causal_rank_return_curve,
            previous_sleeve_weights=carried,
        )
        allocation = decide_tranche_book(recipe=self._recipe, inputs=inputs)
        self._sleeves = rebuild_sleeve_state(
            sleeve_weights=allocation.sleeve_weights,
            sleeve_shares=allocation.sleeve_shares,
            target_weights=allocation.target_weights,
        )
        self._reviewed.append((formation_index, allocation.reviewed_sleeves))
        self._sleeve_shares.append(allocation.sleeve_shares)
        self._aggregate_cap_binding_counts.append(allocation.aggregate_cap_binding_count)
        self._uncapped_targets.append(allocation.uncapped_target_weights.copy())
        self._targets.append(np.asarray(allocation.target_weights, dtype=np.float64).copy())
        if allocation.risk_projection_hash is not None:
            self._risk_projections.append(allocation.risk_projection_hash)
        if resolved.risk_attribution is not None:
            self._risk_attributions.append(resolved.risk_attribution)
        return PortfolioTargetDecision(
            target_weights=allocation.target_weights,
            predicted_variance=None,
            requires_risk_forecast=False,
        )


@runtime_checkable
class ComponentBookProvider(Protocol):
    """One component book the engine can drive, and the state a ledger seals.

    The decision call is the engine's; everything else is what the executor has
    to read back afterwards to publish a path. Two very different books -- a
    risk-and-mu tranche book and a capped equal-weight one -- satisfy it, which
    is what lets the executor stop knowing which strategy it is running.
    """

    def __call__(
        self,
        *,
        formation_index: int,
        reference_weights: FloatArray,
        pretrade_weights: FloatArray,
        decision_mode: Literal["REBALANCE", "HOLD"],
    ) -> PortfolioTargetDecision:
        """Produce one component target under explicit holdings and decision disposition.

        Args:
            formation_index: Exact formation position on the retained source axis.
            reference_weights: Holdings reference for turnover and frozen carry.
            pretrade_weights: Drifted holdings before this decision.
            decision_mode: Explicit rebalance or hold disposition.

        Returns:
            Component target decision; deterministic implementation owns state and input admission.
        """
        ...

    @property
    def formation_sessions(self) -> tuple[date, ...]:
        """Read the retained formation axis.

        Returns:
            Formation sessions in declared source order.
        """
        ...

    @property
    def schedule_offset(self) -> int:
        """Read the retained review schedule offset.

        Returns:
            Declared integer schedule offset.
        """
        ...

    @property
    def sleeve_state(self) -> FloatArray | None:
        """Read current component sleeve holdings.

        Returns:
            Current sleeve matrix, or None before state is available.
        """
        ...

    @property
    def consumed_score_projection_hashes(self) -> tuple[str, ...]:
        """Read score projection identities consumed by component decisions.

        Returns:
            Ordered score projection hashes retained by the provider.
        """
        ...

    @property
    def consumed_risk_projection_hashes(self) -> tuple[str, ...]:
        """Read Risk projection identities consumed by component decisions.

        Returns:
            Ordered Risk projection hashes retained by the provider.
        """
        ...

    @property
    def aggregate_cap_binding_counts(self) -> tuple[int, ...]:
        """Read aggregate name-cap binding counts per formation.

        Returns:
            Retained cap counts on the executed formation axis.
        """
        ...

    @property
    def uncapped_targets_by_formation(self) -> tuple[FloatArray, ...]:
        """Read target history before aggregate name caps.

        Returns:
            Ordered uncapped target arrays retained by the provider.
        """
        ...

    @property
    def target_weights_by_formation(self) -> tuple[FloatArray, ...]:
        """Read final target history per formation.

        Returns:
            Ordered final target arrays retained by the provider.
        """
        ...


class CappedSleeveBookProvider(_SleeveBookProvider):
    """A sleeve book that keeps each reviewed sleeve's drifted notional.

    The other tranche provider in this module resets every reviewed sleeve to
    ``1/T``; this one does not, and applies an aggregate per-name cap from a
    declared formation onward. Both are closed-form membership and sleeve policy
    over books the shared engine executed and drifted -- neither computes a
    return, a fill, a turnover or a cost.
    """

    def __init__(
        self,
        *,
        top_k: int,
        exit_rank: int,
        tranches: int,
        aggregate_name_cap: float,
        aggregate_cap_start_formation: int,
        formations: tuple[TrancheFormationInputs, ...],
        ordered_listing_ids: tuple[str, ...],
        initial_sleeve_weights: FloatArray | None = None,
        schedule_offset: int = 0,
        weight_rule: FrozenSleeveWeightRule = "ew",
        review_phase: int = 0,
        sizing_activation_formation: int | None = None,
        sleeve_cap_equal_weight_multiple: float = 2.1,
    ) -> None:
        """Bind capped sleeve selection and review/sizing controls to exact formations.

        Args:
            top_k: Maximum selected listing count per sleeve.
            exit_rank: Rank beyond which an existing selection exits.
            tranches: Number of retained sleeves.
            aggregate_name_cap: Final aggregate single-name cap.
            aggregate_cap_start_formation: Formation activating the aggregate cap.
            formations: Ordered eligible score formations without Risk projections.
            ordered_listing_ids: Common listing axis.
            initial_sleeve_weights: Optional state with exactly the declared sleeve count.
            schedule_offset: Offset into the declared review schedule.
            weight_rule: Equal weight or admitted curve sizing rule.
            review_phase: Review phase inside the tranche cycle.
            sizing_activation_formation: Required activation for curve sizing.
            sleeve_cap_equal_weight_multiple: Positive per-sleeve cap multiple.

        Raises:
            TrancheExecutionError: Eligibility shape, forbidden Risk/curve input, sizing activation,
                phase, cap multiple or initial sleeve count is invalid.
        """
        super().__init__(
            formations=formations,
            ordered_listing_ids=ordered_listing_ids,
            initial_sleeve_weights=initial_sleeve_weights,
            schedule_offset=schedule_offset,
        )
        listings = len(ordered_listing_ids)
        if any(item.decision_eligible.shape != (listings,) for item in formations):
            raise TrancheExecutionError("portfolio_strategy_lab.component_book_axis_invalid")
        if any(
            item.risk_allocation is not None or item.risk_attribution is not None
            for item in formations
        ):
            raise TrancheExecutionError("portfolio_strategy_lab.component_book_consumes_no_risk")
        if weight_rule == "ew" and any(
            item.causal_rank_return_curve is not None for item in formations
        ):
            raise TrancheExecutionError("portfolio_strategy_lab.component_book_curve_not_consumed")
        if weight_rule == "mu.iv0" and sizing_activation_formation is None:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.component_book_sizing_activation_absent"
            )
        if review_phase < 0 or review_phase >= tranches:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.component_book_review_phase_invalid"
            )
        if sleeve_cap_equal_weight_multiple <= 0.0:
            raise TrancheExecutionError("portfolio_strategy_lab.component_book_sleeve_cap_invalid")
        if self._sleeves is not None and self._sleeves.shape[0] != tranches:
            raise TrancheExecutionError("portfolio_strategy_lab.component_book_sleeve_axis_invalid")
        self._top_k = top_k
        self._exit_rank = exit_rank
        self._tranches = tranches
        self._aggregate_name_cap = aggregate_name_cap
        self._aggregate_cap_start_formation = aggregate_cap_start_formation
        self._weight_rule = weight_rule
        self._review_phase = review_phase
        self._sizing_activation_formation = sizing_activation_formation
        self._sleeve_cap_equal_weight_multiple = sleeve_cap_equal_weight_multiple

    def __call__(
        self,
        *,
        formation_index: int,
        reference_weights: FloatArray,
        pretrade_weights: FloatArray,
        decision_mode: Literal["REBALANCE", "HOLD"],
    ) -> PortfolioTargetDecision:
        """Resolve a hold or capped sleeve target through the conditional target owner.

        Args:
            formation_index: Exact formation position on the retained source axis.
            reference_weights: Holdings reference for turnover and frozen carry.
            pretrade_weights: Drifted holdings before this decision.
            decision_mode: Explicit rebalance or hold disposition.

        Returns:
            Decision returned by resolve_conditional_target with retained sleeve state.
        """
        return self.resolve_conditional_target(
            formation_index=formation_index,
            reference_weights=reference_weights,
            pretrade_weights=pretrade_weights,
            decision_mode=decision_mode,
        )

    def resolve_conditional_target(
        self,
        *,
        formation_index: int,
        reference_weights: FloatArray,
        pretrade_weights: FloatArray,
        decision_mode: Literal["REBALANCE", "HOLD"],
    ) -> PortfolioTargetDecision:
        """Evaluate fixed formation inputs against an explicitly supplied book.

        The caller seals inputs and carry before future entry prices exist.
        A close-marked estimate and the eventual entry target use this same
        arithmetic on separate short-lived providers; an estimate never mutates
        the continuation's sleeves. Historical calls keep exactly this path.
        """
        opened = self._open(
            formation_index=formation_index,
            reference_weights=reference_weights,
            pretrade_weights=pretrade_weights,
            decision_mode=decision_mode,
        )
        if isinstance(opened, PortfolioTargetDecision):
            return opened
        resolved = opened
        position = self._schedule_offset + formation_index
        if self._sleeves is None:
            if position != 0:
                raise TrancheExecutionError(
                    "portfolio_strategy_lab.component_book_sleeve_state_absent"
                )
            due = tuple(range(self._tranches))
            sleeves: FloatArray = np.zeros(
                (self._tranches, len(self._ordered_listing_ids)), dtype=np.float64
            )
        else:
            due = due_sleeves(
                formation_index=position + self._review_phase,
                tranches=self._tranches,
            )
            sleeves = self._sleeves.copy()

        sizing_active = (
            self._weight_rule == "mu.iv0"
            and self._sizing_activation_formation is not None
            and position >= self._sizing_activation_formation
        )
        curve = resolved.causal_rank_return_curve
        ranked_positions: npt.NDArray[np.int64] | None = None
        eligible_count = 0
        if sizing_active:
            if curve is None or curve.disposition != "AVAILABLE":
                raise TrancheExecutionError("portfolio_strategy_lab.component_book_curve_absent")
            if curve.formation_index != formation_index:
                raise TrancheExecutionError(
                    "portfolio_strategy_lab.component_book_curve_formation_mismatch"
                )
            if curve.formation_session != resolved.formation_session:
                raise TrancheExecutionError(
                    "portfolio_strategy_lab.component_book_curve_session_mismatch"
                )
            bucket_means = np.asarray(curve.bucket_means, dtype=np.float64)
            if not np.isfinite(bucket_means).all():
                raise TrancheExecutionError(
                    "portfolio_strategy_lab.component_book_curve_not_finite"
                )
            eligible = np.flatnonzero(
                np.asarray(resolved.decision_eligible, dtype=np.bool_)
                & np.isfinite(resolved.scores)
            )
            ids = np.asarray(self._ordered_listing_ids, dtype=str)
            order = eligible[np.lexsort((ids[eligible], -resolved.scores[eligible]))]
            eligible_count = order.size
            ranked_positions = np.full(len(ids), order.size, dtype=np.int64)
            ranked_positions[order] = np.arange(order.size, dtype=np.int64)
        elif self._weight_rule == "ew" and curve is not None:
            raise TrancheExecutionError("portfolio_strategy_lab.component_book_curve_not_consumed")

        for sleeve_index in due:
            mass = float(sleeves[sleeve_index].sum())
            held = sleeves[sleeve_index] / mass if mass > _MASS_TOLERANCE else None
            share = mass if mass > _MASS_TOLERANCE else 1.0 / self._tranches
            selected = whole_book_hysteresis_selection(
                scores=resolved.scores,
                decision_eligible=resolved.decision_eligible,
                ordered_listing_ids=self._ordered_listing_ids,
                previous_target_weights=held,
                top_k=self._top_k,
                exit_rank=self._exit_rank,
            )
            if selected.size != self._top_k:
                raise TrancheExecutionError("portfolio_strategy_lab.component_book_support_invalid")
            means = None
            rule: FrozenSleeveWeightRule = "ew"
            if sizing_active:
                if ranked_positions is None or curve is None:
                    raise TrancheExecutionError(
                        "portfolio_strategy_lab.component_book_curve_absent"
                    )
                means = rank_bucket_mu(
                    selected=selected,
                    ranked_positions=ranked_positions,
                    eligible_count=eligible_count,
                    bucket_means=np.asarray(curve.bucket_means, dtype=np.float64),
                )
                rule = "mu.iv0"
            inner = sleeve_target_weights(
                mu=means,
                variance=np.ones(selected.size, dtype=np.float64),
                rule=rule,
                ceiling=self._sleeve_cap_equal_weight_multiple / float(self._top_k),
            )
            if inner.disposition != "CAPPED":
                raise TrancheExecutionError(
                    "portfolio_strategy_lab.component_book_sleeve_cap_not_honoured"
                )
            replacement: FloatArray = np.zeros(len(self._ordered_listing_ids), dtype=np.float64)
            replacement[selected] = share * inner.weights
            sleeves[sleeve_index] = replacement

        total = float(sleeves.sum())
        if total <= _MASS_TOLERANCE:
            raise TrancheExecutionError("portfolio_strategy_lab.component_book_empty")
        sleeves = np.asarray(sleeves / total, dtype=np.float64)
        uncapped = np.asarray(sleeves.sum(axis=0), dtype=np.float64)
        binding_count = 0
        target = uncapped
        if position >= self._aggregate_cap_start_formation and due:
            binding_count = int(np.sum(uncapped > self._aggregate_name_cap + _MASS_TOLERANCE))
            capped = cap_and_renormalise(uncapped, ceiling=self._aggregate_name_cap).weights
            ratio = np.divide(
                capped,
                uncapped,
                out=np.ones(capped.size, dtype=np.float64),
                where=uncapped > _MASS_TOLERANCE,
            )
            sleeves = np.asarray(sleeves * ratio[None, :], dtype=np.float64)
            target = np.asarray(capped, dtype=np.float64)

        self._sleeves = sleeves
        self._aggregate_cap_binding_counts.append(binding_count)
        self._uncapped_targets.append(uncapped.copy())
        self._targets.append(target.copy())
        return PortfolioTargetDecision(
            target_weights=target,
            predicted_variance=None,
            requires_risk_forecast=False,
        )


def _normalise_book(value: FloatArray, *, code: str) -> FloatArray:
    measured = np.asarray(value, dtype=np.float64)
    mass = float(measured.sum())
    if not np.isfinite(measured).all() or mass <= _MASS_TOLERANCE:
        raise TrancheExecutionError(code)
    if bool(np.any(measured < -_MASS_TOLERANCE)):
        raise TrancheExecutionError(code)
    return np.asarray(measured / mass, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ComponentBookPlan:
    """One component of a merged book: who it is, its share, and how it decides."""

    component_id: str
    allocation_basis_points: int
    sleeve_rows: int
    provider: ComponentBookProvider


class MergedComponentBookProvider:
    """Carry N independent component books and merge their targets post-trade.

    The engine executes, drifts, charges and evaluates the *merged* book, so a
    fill shortfall has to be reconciled back to the component books before the
    next decision. That reconciliation uses the engine's own reference/pre-trade
    pair; there is no second return calculation here.

    The book each component is reconciled from is read off that component's
    own sleeve state, never from a remembered target. A resumed path restores
    exactly the sleeves the boundary sealed and nothing else, so a merge that
    kept a private copy of its last targets could not be continued at all: the
    copy would be empty, every component would be reconciled against zero, and
    the restored book would be erased on the first decision.
    """

    def __init__(
        self,
        *,
        components: tuple[ComponentBookPlan, ...],
        allocation_total_basis_points: int = 10_000,
    ) -> None:
        """Bind an allocation-complete component plan on one formation axis and schedule.

        Args:
            components: At least two uniquely identified component book plans.
            allocation_total_basis_points: Declared total matched by component allocations.

        Raises:
            TrancheExecutionError: Component count/identity/allocation, common nonempty formation
                axis or schedule offsets disagree.
        """
        if len(components) < 2:
            # One component is not a merge. The single-book case is the provider
            # itself, resolved by `build_component_book`, so that a one-component
            # plan produces bit-identical numbers to running that book alone.
            raise TrancheExecutionError("portfolio_strategy_lab.component_merge_degenerate")
        ids = tuple(value.component_id for value in components)
        if len(set(ids)) != len(ids):
            raise TrancheExecutionError("portfolio_strategy_lab.component_merge_duplicated")
        if sum(value.allocation_basis_points for value in components) != (
            allocation_total_basis_points
        ):
            raise TrancheExecutionError("portfolio_strategy_lab.component_merge_allocation_invalid")
        axes = tuple(value.provider.formation_sessions for value in components)
        if not axes[0] or any(value != axes[0] for value in axes[1:]):
            raise TrancheExecutionError("portfolio_strategy_lab.component_merge_axis_invalid")
        offsets = {value.provider.schedule_offset for value in components}
        if len(offsets) != 1:
            raise TrancheExecutionError("portfolio_strategy_lab.component_merge_schedule_invalid")
        self._components = components
        self._shares = tuple(
            value.allocation_basis_points / float(allocation_total_basis_points)
            for value in components
        )
        self._formation_sessions = axes[0]
        self._schedule_offset = offsets.pop()
        self._uncapped_merged_history: list[FloatArray] = []
        self._merged_target_history: list[FloatArray] = []

    @property
    def components(self) -> tuple[ComponentBookPlan, ...]:
        """Read retained component book plans in declaration order.

        Returns:
            Ordered component plans with their deterministic providers.
        """
        return self._components

    @property
    def formation_sessions(self) -> tuple[date, ...]:
        """Read the retained formation axis.

        Returns:
            Formation sessions in declared source order.
        """
        return self._formation_sessions

    @property
    def schedule_offset(self) -> int:
        """Read the retained review schedule offset.

        Returns:
            Declared integer schedule offset.
        """
        return self._schedule_offset

    @property
    def sleeve_state(self) -> FloatArray | None:
        """Read current component sleeve holdings.

        Returns:
            Contiguous float64 vertical stack in component order, or None if any component lacks
            state.
        """
        states = [value.provider.sleeve_state for value in self._components]
        if any(value is None for value in states):
            return None
        return np.ascontiguousarray(
            np.vstack([value for value in states if value is not None]), dtype=np.float64
        )

    @property
    def consumed_score_projection_hashes(self) -> tuple[str, ...]:
        """Read score projection identities consumed by component decisions.

        Returns:
            Child consumption histories concatenated in component order.
        """
        return tuple(
            value
            for component in self._components
            for value in component.provider.consumed_score_projection_hashes
        )

    @property
    def consumed_risk_projection_hashes(self) -> tuple[str, ...]:
        """Read Risk projection identities consumed by component decisions.

        Returns:
            Child consumption histories concatenated in component order.
        """
        return tuple(
            value
            for component in self._components
            for value in component.provider.consumed_risk_projection_hashes
        )

    @property
    def aggregate_cap_binding_counts(self) -> tuple[int, ...]:
        """Read aggregate name-cap binding counts per formation.

        Returns:
            Per-formation sum of child binding counts; unequal history lengths are refused.
        """
        per_component = [
            component.provider.aggregate_cap_binding_counts for component in self._components
        ]
        return tuple(sum(values) for values in zip(*per_component, strict=True))

    @property
    def uncapped_targets_by_formation(self) -> tuple[FloatArray, ...]:
        """Read target history before aggregate name caps.

        Returns:
            Independent copies of the retained merged target arrays.
        """
        return tuple(value.copy() for value in self._uncapped_merged_history)

    @property
    def target_weights_by_formation(self) -> tuple[FloatArray, ...]:
        """Read final target history per formation.

        Returns:
            Independent copies of the retained merged target arrays.
        """
        return tuple(value.copy() for value in self._merged_target_history)

    def _books_before_decision(
        self,
        *,
        reference_weights: FloatArray,
        pretrade_weights: FloatArray,
    ) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
        """Each component's executed and drifted book, from its own sleeves.

        A component holds what its sleeves sum to. Scaling that by the engine's
        per-name fill ratio against the merged book gives the component's share
        of what was actually executed, and the engine's drift ratio then carries
        it to the decision instant. A fresh book has no sleeves anywhere and is
        reconciled from nothing; a book with sleeves in some components and not
        others is no state this merge ever produced.
        """

        states = [component.provider.sleeve_state for component in self._components]
        if all(value is None for value in states):
            zeros = {
                component.component_id: np.zeros(reference_weights.size, dtype=np.float64)
                for component in self._components
            }
            return zeros, {key: value.copy() for key, value in zeros.items()}
        if any(value is None for value in states):
            raise TrancheExecutionError("portfolio_strategy_lab.component_merge_state_inconsistent")
        books = {
            component.component_id: _normalise_book(
                np.asarray(state, dtype=np.float64).sum(axis=0),
                code="portfolio_strategy_lab.component_state_invalid",
            )
            for component, state in zip(self._components, states, strict=True)
        }
        merged = np.zeros(reference_weights.size, dtype=np.float64)
        for component, share in zip(self._components, self._shares, strict=True):
            merged += share * books[component.component_id]
        merged_target = _normalise_book(
            merged, code="portfolio_strategy_lab.component_merged_target_invalid"
        )
        reference = np.asarray(reference_weights, dtype=np.float64)
        pretrade = np.asarray(pretrade_weights, dtype=np.float64)
        fill_ratio = np.divide(
            reference,
            merged_target,
            out=np.zeros(reference.size, dtype=np.float64),
            where=merged_target > _MASS_TOLERANCE,
        )
        growth = np.divide(
            pretrade,
            reference,
            out=np.ones(reference.size, dtype=np.float64),
            where=reference > _MASS_TOLERANCE,
        )
        component_reference: dict[str, FloatArray] = {}
        component_pretrade: dict[str, FloatArray] = {}
        for component in self._components:
            reconciled = books[component.component_id] * fill_ratio
            component_reference[component.component_id] = _normalise_book(
                reconciled,
                code="portfolio_strategy_lab.component_fill_reconciliation_invalid",
            )
            component_pretrade[component.component_id] = _normalise_book(
                reconciled * growth,
                code="portfolio_strategy_lab.component_drift_invalid",
            )
        return component_reference, component_pretrade

    def __call__(
        self,
        *,
        formation_index: int,
        reference_weights: FloatArray,
        pretrade_weights: FloatArray,
        decision_mode: Literal["REBALANCE", "HOLD"],
    ) -> PortfolioTargetDecision:
        """Distribute carry, execute components and retain allocation-weighted merged targets.

        Args:
            formation_index: Exact formation position on the retained source axis.
            reference_weights: Holdings reference for turnover and frozen carry.
            pretrade_weights: Drifted holdings before this decision.
            decision_mode: Explicit rebalance or hold disposition.

        Returns:
            Hold preserves pretrade weights; rebalance combines normalized component targets by
            declared allocations. Both retain final/uncapped history without requesting a Risk
            forecast.

        Raises:
            TrancheExecutionError: Component holdings, source or target admission fails.
        """
        references, pretrades = self._books_before_decision(
            reference_weights=reference_weights,
            pretrade_weights=pretrade_weights,
        )
        decisions = {
            component.component_id: component.provider(
                formation_index=formation_index,
                reference_weights=references[component.component_id],
                pretrade_weights=pretrades[component.component_id],
                decision_mode=decision_mode,
            )
            for component in self._components
        }
        if decision_mode == "HOLD":
            held = np.asarray(pretrade_weights, dtype=np.float64)
            self._uncapped_merged_history.append(held.copy())
            self._merged_target_history.append(held.copy())
            return PortfolioTargetDecision(
                target_weights=held,
                predicted_variance=None,
                decision_mode="HOLD",
                requires_risk_forecast=False,
            )
        uncapped = np.zeros(reference_weights.size, dtype=np.float64)
        merged = np.zeros(reference_weights.size, dtype=np.float64)
        for component, share in zip(self._components, self._shares, strict=True):
            uncapped += share * component.provider.uncapped_targets_by_formation[-1]
            merged += share * _normalise_book(
                decisions[component.component_id].target_weights,
                code="portfolio_strategy_lab.component_target_invalid",
            )
        uncapped = _normalise_book(
            uncapped, code="portfolio_strategy_lab.component_uncapped_target_invalid"
        )
        merged = _normalise_book(
            merged, code="portfolio_strategy_lab.component_merged_target_invalid"
        )
        self._uncapped_merged_history.append(uncapped.copy())
        self._merged_target_history.append(merged.copy())
        return PortfolioTargetDecision(
            target_weights=merged,
            predicted_variance=None,
            decision_mode=decision_mode,
            requires_risk_forecast=False,
        )


def build_component_book(
    components: tuple[ComponentBookPlan, ...],
    *,
    allocation_total_basis_points: int = 10_000,
) -> ComponentBookProvider:
    """The one book the engine drives, whatever the component plan says.

    A single component carrying the whole allocation *is* the book: interposing
    a merge would renormalise a book that already sums to one and reconcile it
    against itself, which is arithmetic the predecessor path never did. The
    branch is on the plan's cardinality, not on which strategy produced it.
    """
    if not components:
        raise TrancheExecutionError("portfolio_strategy_lab.component_plan_empty")
    if len(components) == 1:
        if components[0].allocation_basis_points != allocation_total_basis_points:
            raise TrancheExecutionError("portfolio_strategy_lab.component_merge_allocation_invalid")
        return components[0].provider
    return MergedComponentBookProvider(
        components=components,
        allocation_total_basis_points=allocation_total_basis_points,
    )


def first_short_formation(
    formations: tuple[TrancheFormationInputs, ...], *, selected: int, walked: int
) -> date | None:
    """The first of the walked formations whose tradable, scored names are fewer than selected.

    A due sleeve selects ``selected`` names from those both decision-eligible and scored, and
    every formation of a walk has one due, so a formation with fewer cannot be decided.

    Args:
        formations: The component's resolved formations, in walk order.
        selected: How many names one rebalance selects.
        walked: How many of them the walk decides.

    Returns:
        That formation's session, or None.
    """
    for formation in formations[:walked]:
        pool = np.asarray(formation.decision_eligible, dtype=np.bool_) & np.isfinite(
            np.asarray(formation.scores, dtype=np.float64)
        )
        if int(np.count_nonzero(pool)) < selected:
            return formation.formation_session
    return None


@runtime_checkable
class ComponentBookFactory(Protocol):
    """A component the executor can open, once it knows where the walk resumes.

    Formations and axes are resolved long before execution; the opening sleeve
    state and the schedule position are not, because they belong to the *segment*
    rather than to the component. Keeping them out of the resolution is what lets
    one resolved plan serve both a flat start and a continuation.
    """

    @property
    def component_id(self) -> str:
        """Read the component declaration identity.

        Returns:
            Explicit component identity.
        """
        ...

    @property
    def allocation_basis_points(self) -> int:
        """Read the declared component allocation.

        Returns:
            Allocation in basis points.
        """
        ...

    @property
    def sleeve_rows(self) -> int:
        """Read the number of sleeve rows required by this component.

        Returns:
            Declared tranche count.
        """
        ...

    @property
    def formation_sessions(self) -> tuple[date, ...]:
        """Read the component formation axis.

        Returns:
            Ordered source formation sessions.
        """
        ...

    def first_short_formation(self, walked: int) -> date | None:
        """The first of the walked formations too short for one of this component's rebalances.

        Args:
            walked: How many of the component's formations the walk decides.

        Returns:
            The first formation whose tradable, scored names are fewer than a rebalance
            selects, or None.
        """
        ...

    def build(
        self, *, initial_sleeve_weights: FloatArray | None, schedule_offset: int
    ) -> ComponentBookProvider:
        """Construct the deterministic component provider with explicit resumed carry.

        Args:
            initial_sleeve_weights: Optional component carry on its declared listing/sleeve axes.
            schedule_offset: Offset for the resumed component review schedule.

        Returns:
            Component provider bound to declared recipe and formation inputs.
        """
        ...


@dataclass(frozen=True, slots=True)
class TrancheBookComponent:
    """A component decided by the shared tranche-book policy."""

    component_id: str
    allocation_basis_points: int
    recipe: TrancheBookRecipe
    formations: tuple[TrancheFormationInputs, ...]
    ordered_listing_ids: tuple[str, ...]
    sector_exposure_matrix: FloatArray
    equal_weight_sector_exposure: FloatArray

    @property
    def sleeve_rows(self) -> int:
        """Read the component recipe tranche count.

        Returns:
            Number of sleeve rows required by this component.
        """
        return self.recipe.tranches

    @property
    def formation_sessions(self) -> tuple[date, ...]:
        """Read component sessions in retained formation order.

        Returns:
            Tuple of formation_session values without sorting.
        """
        return tuple(value.formation_session for value in self.formations)

    def first_short_formation(self, walked: int) -> date | None:
        """The first walked formation too short for one of this tranche book's rebalances.

        Args:
            walked: How many of the component's formations the walk decides.

        Returns:
            The first formation with fewer tradable, scored names than ``top_k``, or None.
        """
        return first_short_formation(self.formations, selected=self.recipe.top_k, walked=walked)

    def build(
        self, *, initial_sleeve_weights: FloatArray | None, schedule_offset: int
    ) -> ComponentBookProvider:
        """Construct the retained Risk-admitted tranche provider.

        Construct the Risk-admitted tranche provider from retained declarations and resumed carry.

        Args:
            initial_sleeve_weights: Optional component carry on its declared listing/sleeve axes.
            schedule_offset: Offset for the resumed component review schedule.

        Returns:
            Provider with this component's recipe, listing/formation inputs and sizing/review
            controls.
        """
        return TrancheBookDecisionProvider(
            recipe=self.recipe,
            formations=self.formations,
            ordered_listing_ids=self.ordered_listing_ids,
            sector_exposure_matrix=self.sector_exposure_matrix,
            equal_weight_sector_exposure=self.equal_weight_sector_exposure,
            initial_sleeve_weights=initial_sleeve_weights,
            schedule_offset=schedule_offset,
        )


@dataclass(frozen=True, slots=True)
class CappedSleeveComponent:
    """A component decided by the drift-preserving capped sleeve book."""

    component_id: str
    allocation_basis_points: int
    top_k: int
    exit_rank: int
    tranches: int
    aggregate_name_cap: float
    aggregate_cap_start_formation: int
    formations: tuple[TrancheFormationInputs, ...]
    ordered_listing_ids: tuple[str, ...]
    weight_rule: FrozenSleeveWeightRule = "ew"
    review_phase: int = 0
    sizing_activation_formation: int | None = None
    sleeve_cap_equal_weight_multiple: float = 2.1

    @property
    def sleeve_rows(self) -> int:
        """Read the component recipe tranche count.

        Returns:
            Number of sleeve rows required by this component.
        """
        return self.tranches

    @property
    def formation_sessions(self) -> tuple[date, ...]:
        """Read component sessions in retained formation order.

        Returns:
            Tuple of formation_session values without sorting.
        """
        return tuple(value.formation_session for value in self.formations)

    def first_short_formation(self, walked: int) -> date | None:
        """The first walked formation too short for one of this capped book's rebalances.

        Args:
            walked: How many of the component's formations the walk decides.

        Returns:
            The first formation with fewer tradable, scored names than ``top_k``, or None.
        """
        return first_short_formation(self.formations, selected=self.top_k, walked=walked)

    def build(
        self, *, initial_sleeve_weights: FloatArray | None, schedule_offset: int
    ) -> ComponentBookProvider:
        """Construct the capped sleeve provider from retained declarations and resumed carry.

        Args:
            initial_sleeve_weights: Optional component carry on its declared listing/sleeve axes.
            schedule_offset: Offset for the resumed component review schedule.

        Returns:
            Provider with this component's recipe, listing/formation inputs and sizing/review
            controls.
        """
        return CappedSleeveBookProvider(
            top_k=self.top_k,
            exit_rank=self.exit_rank,
            tranches=self.tranches,
            aggregate_name_cap=self.aggregate_name_cap,
            aggregate_cap_start_formation=self.aggregate_cap_start_formation,
            formations=self.formations,
            ordered_listing_ids=self.ordered_listing_ids,
            initial_sleeve_weights=initial_sleeve_weights,
            schedule_offset=schedule_offset,
            weight_rule=self.weight_rule,
            review_phase=self.review_phase,
            sizing_activation_formation=self.sizing_activation_formation,
            sleeve_cap_equal_weight_multiple=self.sleeve_cap_equal_weight_multiple,
        )


def open_component_book(
    components: tuple[ComponentBookFactory, ...],
    *,
    initial_sleeve_weights: FloatArray | None,
    schedule_offset: int,
    formation_sessions: tuple[date, ...] | None = None,
    allocation_total_basis_points: int = 10_000,
) -> ComponentBookProvider:
    """Open every component over one carried sleeve state and merge the result.

    The carry is one array because the ledger seals one array. It is split back
    across components in plan order by each component's own sleeve-row count, so
    a continuation restores exactly the rows the component sealed rather than a
    reshaped view of somebody else's book.

    `formation_sessions` is the axis the engine will walk. Every component is
    checked against it here, before a single decision: a component resolved over
    a different calendar would otherwise be driven by formation *index* and pair
    each decision with somebody else's session.
    """
    if not components:
        raise TrancheExecutionError("portfolio_strategy_lab.component_plan_empty")
    if formation_sessions is not None:
        walked = len(formation_sessions)
        for component in components:
            axis = component.formation_sessions
            if len(axis) < walked or axis[:walked] != formation_sessions:
                raise TrancheExecutionError(
                    "portfolio_strategy_lab.component_workspace_session_axis_mismatch"
                )
    # Every formation the walk decides selects each component's names for a due sleeve from
    # those both tradable and scored; one with fewer is refused here, by its session, before
    # the book opens, never in the middle of the walk (class).
    for component in components:
        short = component.first_short_formation(
            len(component.formation_sessions) if formation_sessions is None else walked
        )
        if short is not None:
            raise EligiblePoolShort(f"portfolio_strategy_lab.eligible_pool_short:{short}")
    rows = tuple(value.sleeve_rows for value in components)
    carried: list[FloatArray | None] = [None] * len(components)
    if initial_sleeve_weights is not None:
        carry = np.asarray(initial_sleeve_weights, dtype=np.float64)
        if carry.ndim != 2 or carry.shape[0] != sum(rows):
            raise TrancheExecutionError(
                "portfolio_strategy_lab.component_sleeve_carry_axis_invalid"
            )
        start = 0
        for index, count in enumerate(rows):
            carried[index] = np.ascontiguousarray(carry[start : start + count], dtype=np.float64)
            start += count
    plans = tuple(
        ComponentBookPlan(
            component_id=component.component_id,
            allocation_basis_points=component.allocation_basis_points,
            sleeve_rows=component.sleeve_rows,
            provider=component.build(
                initial_sleeve_weights=carried[index], schedule_offset=schedule_offset
            ),
        )
        for index, component in enumerate(components)
    )
    return build_component_book(plans, allocation_total_basis_points=allocation_total_basis_points)


__all__ = [
    "CappedSleeveBookProvider",
    "CappedSleeveComponent",
    "ComponentBookFactory",
    "ComponentBookPlan",
    "ComponentBookProvider",
    "EligiblePoolShort",
    "MergedComponentBookProvider",
    "TrancheBookComponent",
    "TrancheBookDecisionProvider",
    "TrancheExecutionError",
    "TrancheFormationInputs",
    "build_component_book",
    "first_short_formation",
    "open_component_book",
    "project_sleeves_onto_book",
    "rebuild_sleeve_state",
]
