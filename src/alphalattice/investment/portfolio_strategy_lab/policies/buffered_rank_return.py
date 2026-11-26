"""Buffered Rank Return: a causal rank-return input and a diagonal closed-form allocation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    FloatArray,
    PortfolioTargetDecision,
    PortfolioWalkForwardError,
)
from alphalattice.capabilities.portfolio_inputs.signed_score.contracts import (
    PortfolioExecutionEvents,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .buffered_equal_weight import whole_book_equal_weight_target, whole_book_hysteresis_selection
from .contracts import (
    PORTFOLIO_POLICY_PACKAGE,
    BoundPolicyDecisionInput,
    CausalRankReturnCurveSlice,
    PortfolioPolicyAdapterBinding,
    PortfolioPolicyRecipe,
    PortfolioSelectionAllocationSemantics,
    portfolio_adapter_implementation_hash,
)

CAUSAL_RANK_MU_POLICY_ID = "WHOLE_BOOK_HYSTERESIS_CAUSAL_RANK_MU_DIAGONAL_TILT"
_BUCKETS = 20
CAUSAL_RANK_MU_LOOKBACK = 252
"""How many matured prior formations the rank-`mu` curve averages over.

Public because it is a *policy* parameter that upstream support checks must not
restate. A caller that hard-codes 252 beside this one is a caller that keeps its
own copy of a number this owner is allowed to change.
"""

_LOOKBACK = CAUSAL_RANK_MU_LOOKBACK
_MINIMUM_MEMBERS = 4 * _BUCKETS
_TOLERANCE = 1e-10
_MATURITY_RULE: Literal["HOLDING_END_OPEN_OWNER_READY_BY_FORMATION_CLOSE"] = (
    "HOLDING_END_OPEN_OWNER_READY_BY_FORMATION_CLOSE"
)

type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int64]


def _array_hash(values: np.ndarray, *, dtype: npt.DTypeLike) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def _readonly(values: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    values.setflags(write=False)
    return values


def rank_bucket_observations(
    *,
    scores: FloatArray,
    realized_simple_returns: FloatArray,
    decision_eligible: BoolArray,
    rank_keys: npt.NDArray[np.generic],
    bucket_count: int = _BUCKETS,
) -> FloatArray:
    """The shared per-formation arithmetic; the recipe supplies the tie order.

     uses listing identifiers. The frozen complete-observation policy uses
    positions on its declared axis. Those orders must not be conflated merely
    because the installed historical axis happens to be lexicographically sorted.
    """
    if (
        scores.ndim != 2
        or realized_simple_returns.shape != scores.shape
        or decision_eligible.shape != scores.shape
        or rank_keys.shape != (scores.shape[1],)
        or bucket_count < 1
    ):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.rank_observation_axis_invalid")
    observed: FloatArray = np.full((scores.shape[0], bucket_count), np.nan, dtype=np.float64)
    for index in range(scores.shape[0]):
        eligible = np.flatnonzero(
            decision_eligible[index]
            & np.isfinite(scores[index])
            & np.isfinite(realized_simple_returns[index])
        )
        if eligible.size < 4 * bucket_count:
            continue
        ranked = eligible[np.lexsort((rank_keys[eligible], -scores[index, eligible]))]
        for bucket in range(bucket_count):
            members = ranked[
                (bucket * ranked.size) // bucket_count : ((bucket + 1) * ranked.size)
                // bucket_count
            ]
            if members.size:
                observed[index, bucket] = float(np.mean(realized_simple_returns[index, members]))
    return observed


def complete_matured_observation_indices(
    *,
    complete_indices: IntArray,
    observation_sessions: tuple[date, ...],
    holding_end_sessions: tuple[date | None, ...],
    decision_session: date,
    lookback: int,
) -> tuple[int, ...]:
    """One membership rule shared by curve calculation and its published evidence."""
    return tuple(
        int(index)
        for index in complete_indices
        if observation_sessions[index] < decision_session
        and (end := holding_end_sessions[index]) is not None
        and end <= decision_session
    )[-lookback:]


def complete_matured_rank_curve(
    *,
    observations: FloatArray,
    observation_sessions: tuple[date, ...],
    holding_end_sessions: tuple[date | None, ...],
    decision_sessions: tuple[date, ...],
    lookback: int = _LOOKBACK,
) -> tuple[FloatArray, IntArray]:
    """Average the latest complete matured rows, not the latest calendar rows.

    Holding ends come from the execution owner on the complete exchange axis.
    Unknown maturities never qualify. This is the frozen Gate-Q window, distinct
    from last-matured-formations/partial-bucket-support rule below.
    """
    if (
        observations.ndim != 2
        or len(observation_sessions) != len(observations)
        or len(holding_end_sessions) != len(observations)
        or observation_sessions != tuple(sorted(set(observation_sessions)))
        or decision_sessions != tuple(sorted(set(decision_sessions)))
        or lookback < 1
        or any(
            end is not None and end <= session
            for session, end in zip(observation_sessions, holding_end_sessions, strict=True)
        )
    ):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.rank_maturity_axis_invalid")
    curves: FloatArray = np.full((len(decision_sessions), observations.shape[1]), np.nan)
    newest: IntArray = np.full(len(decision_sessions), -1, dtype=np.int64)
    valid = np.flatnonzero(np.isfinite(observations).all(axis=1))
    for row, session in enumerate(decision_sessions):
        selected = complete_matured_observation_indices(
            complete_indices=valid,
            observation_sessions=observation_sessions,
            holding_end_sessions=holding_end_sessions,
            decision_session=session,
            lookback=lookback,
        )
        if len(selected) == lookback:
            curves[row] = observations[list(selected)].mean(axis=0)
            newest[row] = selected[-1]
    return curves, newest


class WholeBookHysteresisCausalRankMuRecipe(BaseModel):  # type: ignore[misc]
    """The one fixed recipe supplied by the immutable baseline package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["WholeBookHysteresisCausalRankMuRecipe"] = "WholeBookHysteresisCausalRankMuRecipe"
    top_k: Literal[50] = 50
    exit_rank: Literal[100] = 100
    cadence: Literal[3] = 3
    burn_in: Literal[378] = 378
    kappa: float = 0.03
    bucket_count: Literal[20] = 20
    lookback: Literal[252] = 252
    maximum_weight: float = 0.06
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def policy_id(self) -> str:
        """Read this recipe's installed policy identity.

        Returns:
            The policy identity declared by this recipe owner.
        """
        return CAUSAL_RANK_MU_POLICY_ID

    @classmethod
    def create(cls) -> Self:
        """Seal the fixed causal rank-mu diagonal-tilt recipe.

        Returns:
            Validated installed recipe with canonical recipe_hash.
        """
        values: dict[str, object] = {
            "kind": "WholeBookHysteresisCausalRankMuRecipe",
            "top_k": 50,
            "exit_rank": 100,
            "cadence": 3,
            "burn_in": 378,
            "kappa": 0.03,
            "bucket_count": 20,
            "lookback": 252,
            "maximum_weight": 0.06,
        }
        return cls(**values, recipe_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the declared recipe parameters and canonical self identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The recipe hash differs or fixed kappa/maximum weight differs.
        """
        if self.kappa != 0.03 or self.maximum_weight != 0.06:
            raise ValueError("portfolio_strategy_lab.whole_book_rank_mu_recipe_parameters_invalid")
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise ValueError("portfolio_strategy_lab.whole_book_rank_mu_recipe_identity_invalid")
        return self


@dataclass(frozen=True, slots=True)
class CausalRankReturnCurve:
    """Owner-derived causal per-session rank-return curve over one exact axis."""

    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    score_authority_identity: str
    score_values_hash: str
    decision_eligibility_hash: str
    simple_return_identity: str
    execution_events_hash: str
    maturity_rule: Literal["HOLDING_END_OPEN_OWNER_READY_BY_FORMATION_CLOSE"]
    bucket_count: int
    lookback: int
    admitted_formation_counts: tuple[int, ...]
    bucket_support_counts: IntArray
    per_session_bucket_means: FloatArray
    curve_values: FloatArray
    curve_hash: str

    def at(self, formation_index: int) -> CausalRankReturnCurveSlice:
        """Read one immutable causal curve slice with explicit history/support disposition.

        Args:
            formation_index: Position on the retained formation axis.

        Returns:
            Read-only float64 bucket means, support counts and AVAILABLE/history/support
            disposition.

        Raises:
            PortfolioWalkForwardError: Formation position is outside the retained axis.
        """
        if formation_index < 0 or formation_index >= len(self.formation_sessions):
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.whole_book_rank_mu_curve_formation_invalid"
            )
        supports = tuple(int(value) for value in self.bucket_support_counts[formation_index])
        admitted = self.admitted_formation_counts[formation_index]
        if admitted < self.lookback:
            disposition: Literal[
                "AVAILABLE",
                "INSUFFICIENT_MATURED_FORMATION_HISTORY",
                "PARTIAL_BUCKET_SUPPORT",
            ] = "INSUFFICIENT_MATURED_FORMATION_HISTORY"
        elif any(value != self.lookback for value in supports):
            disposition = "PARTIAL_BUCKET_SUPPORT"
        else:
            disposition = "AVAILABLE"
        values = np.ascontiguousarray(self.curve_values[formation_index], dtype=np.float64)
        values.setflags(write=False)
        return CausalRankReturnCurveSlice(
            formation_index=formation_index,
            formation_session=self.formation_sessions[formation_index],
            bucket_means=values,
            bucket_support_counts=supports,
            admitted_formation_count=admitted,
            disposition=disposition,
            curve_hash=self.curve_hash,
        )


@dataclass(frozen=True, slots=True)
class CausalRankMuMaturitySupport:
    """The exact schedule-derived sessions on which the causal curve may exist.

    This is a cheap authority projection over sealed execution events. It does
    not build scores or a curve; RUN still verifies that every projected session
    has complete bucket support before any Program is published.
    """

    formation_sessions: tuple[date, ...]
    lookback: int
    execution_events_hash: str
    support_hash: str


def _matured_formation_indices(
    *, execution_events: PortfolioExecutionEvents, formation_index: int, lookback: int
) -> tuple[int, ...]:
    decision_at = execution_events.decision_at[formation_index]
    admitted = tuple(
        prior
        for prior, holding_end_at in enumerate(execution_events.exit_at)
        if prior < formation_index and holding_end_at <= decision_at
    )
    return admitted[-lookback:]


def causal_rank_mu_maturity_support(
    execution_events: PortfolioExecutionEvents, *, lookback: int = _LOOKBACK
) -> CausalRankMuMaturitySupport:
    """Project the causal rank-`mu` watermark from the schedule owner."""
    if lookback < 1:
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_rank_mu_curve_input_axis_invalid"
        )
    sessions = tuple(
        session
        for index, session in enumerate(execution_events.ordered_formation_sessions)
        if len(
            _matured_formation_indices(
                execution_events=execution_events,
                formation_index=index,
                lookback=lookback,
            )
        )
        == lookback
    )
    if not sessions:
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_rank_mu_maturity_support_absent"
        )
    identity = {
        "kind": "CausalRankMuMaturitySupport",
        "formation_sessions": tuple(value.isoformat() for value in sessions),
        "lookback": lookback,
        "execution_events_hash": execution_events.events_hash,
    }
    return CausalRankMuMaturitySupport(
        formation_sessions=sessions,
        lookback=lookback,
        execution_events_hash=execution_events.events_hash,
        support_hash=str(canonical_hash(identity)),
    )


def build_causal_rank_return_curve(
    *,
    formation_sessions: tuple[date, ...],
    ordered_listing_ids: tuple[str, ...],
    scores: FloatArray,
    score_authority_identity: str,
    decision_eligible: BoolArray,
    realized_simple_returns: FloatArray,
    execution_events: PortfolioExecutionEvents,
    bucket_count: int = _BUCKETS,
    lookback: int = _LOOKBACK,
) -> CausalRankReturnCurve:
    """Build holding-end-ready, per-session-average rank-return curve.

    This owner deliberately accepts outcomes only by the installed execution
    schedule: an earlier positional index is not evidence that its holding-end
    outcome was known at the current formation close.
    """
    rows = len(formation_sessions)
    columns = len(ordered_listing_ids)
    if (
        rows < 1
        or columns < bucket_count
        or len(set(ordered_listing_ids)) != columns
        or scores.shape != (rows, columns)
        or decision_eligible.shape != (rows, columns)
        or realized_simple_returns.shape != (rows, columns)
        or execution_events.ordered_formation_sessions != formation_sessions
        or len(execution_events.decision_at) != rows
        or len(execution_events.exit_at) != rows
        or not score_authority_identity
        or bucket_count < 1
        or lookback < 1
    ):
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_rank_mu_curve_input_axis_invalid"
        )

    minimum_members = 4 * bucket_count
    per_session = rank_bucket_observations(
        scores=scores,
        realized_simple_returns=realized_simple_returns,
        decision_eligible=decision_eligible,
        rank_keys=np.asarray(ordered_listing_ids, dtype=str),
        bucket_count=bucket_count,
    )

    curves: FloatArray = np.full((rows, bucket_count), np.nan, dtype=np.float64)
    supports: IntArray = np.zeros((rows, bucket_count), dtype=np.int64)
    admitted_counts: list[int] = []
    for index in range(rows):
        admitted = _matured_formation_indices(
            execution_events=execution_events,
            formation_index=index,
            lookback=lookback,
        )
        admitted_counts.append(len(admitted))
        if len(admitted) < lookback:
            continue
        window = per_session[np.asarray(admitted, dtype=np.int64)]
        finite = np.isfinite(window)
        supports[index] = finite.sum(axis=0, dtype=np.int64)
        sums = np.where(finite, window, 0.0).sum(axis=0)
        curves[index] = np.divide(
            sums,
            supports[index],
            out=np.full(bucket_count, np.nan, dtype=np.float64),
            where=supports[index] > 0,
        )

    per_session = cast(FloatArray, _readonly(np.ascontiguousarray(per_session, dtype=np.float64)))
    curves = cast(FloatArray, _readonly(np.ascontiguousarray(curves, dtype=np.float64)))
    supports = cast(IntArray, _readonly(np.ascontiguousarray(supports, dtype=np.int64)))
    identity = {
        "kind": "CausalRankReturnCurve",
        "score_authority_identity": score_authority_identity,
        "score_values_hash": _array_hash(scores, dtype=np.float64),
        "formation_sessions": tuple(value.isoformat() for value in formation_sessions),
        "ordered_listing_ids": ordered_listing_ids,
        "decision_eligibility_hash": _array_hash(decision_eligible, dtype=np.bool_),
        "simple_return_identity": _array_hash(realized_simple_returns, dtype=np.float64),
        "execution_events_hash": execution_events.events_hash,
        "maturity_rule": _MATURITY_RULE,
        "bucket_count": bucket_count,
        "lookback": lookback,
        "minimum_members_per_session": minimum_members,
        "admitted_formation_counts": tuple(admitted_counts),
        "bucket_support_counts_hash": _array_hash(supports, dtype=np.int64),
        "per_session_bucket_means_hash": _array_hash(per_session, dtype=np.float64),
        "curve_values_hash": _array_hash(curves, dtype=np.float64),
    }
    return CausalRankReturnCurve(
        formation_sessions=formation_sessions,
        ordered_listing_ids=ordered_listing_ids,
        score_authority_identity=score_authority_identity,
        score_values_hash=str(identity["score_values_hash"]),
        decision_eligibility_hash=str(identity["decision_eligibility_hash"]),
        simple_return_identity=str(identity["simple_return_identity"]),
        execution_events_hash=execution_events.events_hash,
        maturity_rule=_MATURITY_RULE,
        bucket_count=bucket_count,
        lookback=lookback,
        admitted_formation_counts=tuple(admitted_counts),
        bucket_support_counts=supports,
        per_session_bucket_means=per_session,
        curve_values=curves,
        curve_hash=str(canonical_hash(identity)),
    )


def _full_ranked_eligible(
    *, scores: FloatArray, decision_eligible: BoolArray, ordered_listing_ids: tuple[str, ...]
) -> IntArray:
    if (
        scores.ndim != 1
        or decision_eligible.shape != scores.shape
        or len(ordered_listing_ids) != scores.size
        or len(set(ordered_listing_ids)) != len(ordered_listing_ids)
    ):
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_rank_mu_decision_axis_invalid"
        )
    eligible = np.flatnonzero(decision_eligible & np.isfinite(scores))
    ids = np.asarray(ordered_listing_ids, dtype=str)
    ordered = eligible[np.lexsort((ids[eligible], -scores[eligible]))]
    return cast(IntArray, np.asarray(ordered, dtype=np.int64))


@dataclass(frozen=True, slots=True)
class DiagonalRankMuAllocation:
    """Retain causal rank-mu target weights, selected utilities and curve disposition."""

    target_weights: FloatArray
    selected_mu: FloatArray
    curve_disposition: str


def whole_book_diagonal_rank_mu_target(
    *,
    selected: IntArray,
    scores: FloatArray,
    covariance: FloatArray,
    decision_eligible: BoolArray,
    reference_weights: FloatArray,
    ordered_listing_ids: tuple[str, ...],
    curve: CausalRankReturnCurveSlice,
    top_k: int,
    bucket_count: int,
    kappa: float,
    maximum_weight: float,
) -> DiagonalRankMuAllocation:
    """Apply diagonal mean-variance tilt to one complete selected book."""
    if (
        covariance.ndim != 2
        or covariance.shape != (reference_weights.size, reference_weights.size)
        or selected.size != top_k
        or curve.bucket_means.shape != (bucket_count,)
        or maximum_weight <= 0.0
    ):
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_rank_mu_allocation_axis_invalid"
        )
    ranked = _full_ranked_eligible(
        scores=scores,
        decision_eligible=decision_eligible,
        ordered_listing_ids=ordered_listing_ids,
    )
    if ranked.size < top_k or np.any(~np.isin(selected, ranked)):
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_rank_mu_selection_invalid"
        )
    positions = np.full(scores.size, ranked.size, dtype=np.int64)
    positions[ranked] = np.arange(ranked.size, dtype=np.int64)
    buckets = np.clip((positions[selected] * bucket_count) // ranked.size, 0, bucket_count - 1)
    mu = np.where(np.isfinite(curve.bucket_means[buckets]), curve.bucket_means[buckets], 0.0)
    diagonal = np.diag(covariance)[selected]
    if np.any(~np.isfinite(diagonal)):
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_rank_mu_covariance_diagonal_invalid"
        )
    sigma2 = np.clip(diagonal, 1e-12, None)
    fallback = whole_book_equal_weight_target(
        selected=selected,
        decision_eligible=decision_eligible,
        reference_weights=reference_weights,
        top_k=top_k,
    )
    selected_mass = float(fallback[selected].sum())
    if selected_mass <= 0.0:
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_rank_mu_selected_mass_invalid"
        )
    raw = selected_mass / top_k + kappa * (mu - float(mu.mean())) / sigma2
    weights = np.clip(raw, 0.0, None)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_rank_mu_weight_mass_invalid"
        )
    weights *= selected_mass / total
    for _round in range(8):
        over = weights > maximum_weight + _TOLERANCE
        if not bool(over.any()):
            break
        spare = float((weights[over] - maximum_weight).sum())
        weights[over] = maximum_weight
        room = ~over
        room_mass = float(weights[room].sum())
        if not bool(room.any()) or room_mass <= 0.0:
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.whole_book_rank_mu_cap_redistribution_invalid"
            )
        weights[room] += spare * weights[room] / room_mass
    if (
        not np.isfinite(weights).all()
        or not np.isclose(float(weights.sum()), selected_mass, rtol=0.0, atol=_TOLERANCE)
        or bool(np.any(weights > maximum_weight + _TOLERANCE))
    ):
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_rank_mu_cap_not_enforced"
        )
    target = np.array(fallback, copy=True)
    target[selected] = weights
    target.setflags(write=False)
    selected_mu = np.ascontiguousarray(mu, dtype=np.float64)
    selected_mu.setflags(write=False)
    return DiagonalRankMuAllocation(
        target_weights=target,
        selected_mu=selected_mu,
        curve_disposition=curve.disposition,
    )


class WholeBookHysteresisCausalRankMuAdapter:
    """Direct allocation: causal rank-return input plus Risk diagonal only."""

    policy_id = CAUSAL_RANK_MU_POLICY_ID
    solver_backed = False
    selection_allocation_semantics = (
        PortfolioSelectionAllocationSemantics.closed_form_hysteresis_diagonal_rank_mu_tilt()
    )

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        """Bind this adapter's implementation, declared input consumption and allocation semantics.

        Returns:
            Exact adapter binding for previous-target hysteresis, this-formation causal curve and
            admitted covariance diagonal; it retains predicted variance.
        """
        return PortfolioPolicyAdapterBinding.create(
            policy_id=self.policy_id,
            adapter_implementation_hash=portfolio_adapter_implementation_hash(
                (PORTFOLIO_POLICY_PACKAGE, Path(__file__)),
                (
                    PORTFOLIO_POLICY_PACKAGE,
                    Path(whole_book_hysteresis_selection.__code__.co_filename),
                ),
            ),
            recipe_schema_id="WHOLE_BOOK_HYSTERESIS_CAUSAL_RANK_MU_DIAGONAL_TILT",
            solver_semantics="CLOSED_FORM_CAUSAL_RANK_MU_DIAGONAL_TILT_NO_OPTIMIZER",
            deterministic_policy={
                "selection": "stable-score-listing-id-top-k-with-previous-target-exit-rank",
                "allocation": "causal-rank-bucket-simple-return-mu-diagonal-covariance-tilt",
                "rank_curve_input": (
                    "holding-end-open-owner-ready-by-formation-close;"
                    "252-latest-admitted-per-session-bucket-means"
                ),
                "missing_mu": "nonfinite-bucket-mu-maps-to-zero-with-explicit-curve-disposition",
                "covariance_input": (
                    "owner-validated-diagonal-for-allocation-and-predicted-variance"
                ),
                "cap": "3-over-k-pro-rata-redistribution-at-most-eight-rounds",
                "book_semantics": "whole-book-not-tranche-or-phase-composite",
            },
            selection_allocation_semantics=self.selection_allocation_semantics,
            input_consumption_semantics="RISK_FORECAST_REQUIRED_NO_OPTIMIZER",
        )

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: object,
    ) -> PortfolioTargetDecision:
        """Allocate the selected book from causal rank-mu and covariance diagonal.

        Apply previous-target hysteresis, this-formation causal curve and admitted covariance
        diagonal; it retains predicted variance.

        Args:
            policy: Concrete admitted recipe for this installed adapter.
            inputs: Bound scores, eligibility, reference and declared auxiliary lanes.
            optimizer: Deterministic numerical owner; closed-form adapters do not use it.

        Returns:
            Target decision and the Risk forecast disposition produced by this allocation owner.

        Raises:
            PortfolioWalkForwardError: Required listing/formation, curve or covariance inputs are
                absent or incompatible.
        """
        del optimizer
        recipe = WholeBookHysteresisCausalRankMuRecipe.model_validate(policy)
        if inputs.ordered_listing_ids is None or inputs.formation_index is None:
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.whole_book_rank_mu_listing_or_formation_axis_absent"
            )
        curve = inputs.causal_rank_return_curve
        if curve is None or curve.formation_index != inputs.formation_index:
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.whole_book_rank_mu_curve_not_this_formation"
            )
        covariance = inputs.require_covariance()
        selected = whole_book_hysteresis_selection(
            scores=inputs.scores,
            decision_eligible=inputs.decision_eligible,
            ordered_listing_ids=inputs.ordered_listing_ids,
            previous_target_weights=inputs.previous_target_weights,
            top_k=recipe.top_k,
            exit_rank=recipe.exit_rank,
        )
        allocation = whole_book_diagonal_rank_mu_target(
            selected=selected,
            scores=inputs.scores,
            covariance=covariance,
            decision_eligible=inputs.decision_eligible,
            reference_weights=inputs.reference_weights,
            ordered_listing_ids=inputs.ordered_listing_ids,
            curve=curve,
            top_k=recipe.top_k,
            bucket_count=recipe.bucket_count,
            kappa=recipe.kappa,
            maximum_weight=recipe.maximum_weight,
        )
        return PortfolioTargetDecision(
            target_weights=allocation.target_weights,
            predicted_variance=float(
                allocation.target_weights @ covariance @ allocation.target_weights
            ),
            requires_risk_forecast=True,
        )


__all__ = [
    "CAUSAL_RANK_MU_LOOKBACK",
    "CAUSAL_RANK_MU_POLICY_ID",
    "CausalRankMuMaturitySupport",
    "CausalRankReturnCurve",
    "CausalRankReturnCurveSlice",
    "DiagonalRankMuAllocation",
    "WholeBookHysteresisCausalRankMuAdapter",
    "WholeBookHysteresisCausalRankMuRecipe",
    "build_causal_rank_return_curve",
    "causal_rank_mu_maturity_support",
    "whole_book_diagonal_rank_mu_target",
]
