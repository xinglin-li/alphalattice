"""The closed-form tranche book: sleeve schedule, membership, weighting and caps.

This is the public default strategy's rule owner. It is deliberately *only*
rules: pure functions over explicit inputs, plus one typed recipe. It holds no
state, carries nothing between formations, and implements no drift, turnover,
cost or failed fill -- those belong to the shared Backtesting state machine, and
a policy that grew its own copy is exactly the second numerical path the
delivery plan forbids.

Why a new owner rather than a variant of C6. All three installed C-policies
declare ``"book_semantics": "whole-book-not-tranche-or-phase-composite"``, which
is an explicit statement that they are not this. A tranche book runs `T`
independent sleeves on their own clocks and reviews exactly one per session, so
its schedule, its per-sleeve membership and its aggregation have no counterpart
there. Reusing the C6 recipe under a new name would have been an alias, not an
implementation.

What is reused exactly: ``whole_book_hysteresis_selection`` for one sleeve's membership,
and the causal rank-bucket curve slice the Host binds into the decision input.
Neither is copied.

**Numerical semantics are transcribed from the admitted construction, not
inferred from the control names.** Two of them would have been got wrong by
reading ``mu.iv1`` as "mu over sigma":

* ``mu.ivP`` does not use the bucket mean directly. Bucket means are simple
  returns and can be negative, which a long-only book cannot hold. The rule
  shifts by the minimum and adds a floor of a tenth of the resulting spread, so
  the weakest admitted bucket keeps a small positive weight instead of a zero
  or a negative one.
* the sleeve ceiling is ``2.1 / top_k`` of the *sleeve's own* notional, which is
  `2.1x` that sleeve's equal weight at every width. It is not the 6% aggregate
  name cap, which applies once, later, to the assembled book.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    FloatArray,
    PortfolioTargetDecision,
    PortfolioWalkForwardError,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .buffered_equal_weight import whole_book_hysteresis_selection
from .contracts import (
    PORTFOLIO_POLICY_PACKAGE,
    BoundPolicyDecisionInput,
    PerNameRiskScale,
    PortfolioPolicyAdapterBinding,
    PortfolioPolicyRecipe,
    PortfolioSelectionAllocationSemantics,
    portfolio_adapter_implementation_hash,
)

type IntArray = npt.NDArray[np.int64]

TRANCHE_BOOK_POLICY_ID: Final = "TRANCHE_BOOK_SLEEVE_SCHEDULE_HYSTERESIS_CLOSED_FORM"

type TrancheWeightRule = Literal["ew", "iv1", "iv2", "mu.iv1", "mu.iv2"]
type FrozenSleeveWeightRule = Literal["ew", "mu.iv0"]
type SleeveWeightRule = Literal["ew", "iv1", "iv2", "mu.iv0", "mu.iv1", "mu.iv2"]
ADMITTED_WEIGHT_RULES: Final[tuple[TrancheWeightRule, ...]] = (
    "ew",
    "iv1",
    "iv2",
    "mu.iv1",
    "mu.iv2",
)

TOP_K_RANGE: Final = (17, 75)
TRANCHES_RANGE: Final = (3, 10)
EXIT_RANK_MULTIPLE_RANGE: Final = (1.0, 6.0)

DEFAULT_TOP_K: Final = 35
DEFAULT_TRANCHES: Final = 3
DEFAULT_EXIT_RANK: Final = 70
DEFAULT_WEIGHT_RULE: Final[TrancheWeightRule] = "mu.iv1"

SLEEVE_CAP_EQUAL_WEIGHT_MULTIPLE: Final = 2.1
"""A sleeve name may hold at most `2.1x` that sleeve's equal weight."""

AGGREGATE_NAME_CAP: Final = 0.06
"""One name may hold at most 6% of the assembled book."""

SLEEVE_SHARE_POLICY: Final = "EQUAL_NOTIONAL_AT_EACH_FORMATION"
"""The frozen product plan's explicit aggregation identity."""

MU_FLOOR_SPREAD_SHARE: Final = 0.1
"""The weakest admitted bucket keeps a tenth of the lifted spread, never zero."""


_VARIANCE_FLOOR: Final = 1e-16
_CAP_TOLERANCE: Final = 1e-12

type CapDisposition = Literal[
    "CAPPED",
    "EQUAL_WEIGHT_CEILING_INFEASIBLE",
    "EQUAL_WEIGHT_NO_POSITIVE_MASS",
]
"""Only two ways out besides success, and both are genuine.

An earlier version carried a third, ``CEILING_UNREACHABLE``, for books where a
capped solution existed but the algorithm could not find one. That state was an
artifact of a defective water-filling loop rather than a property of the data,
and correcting the loop removed it: when ``ceiling * held >= 1`` a solution
always exists and is now always found.
"""


class TrancheBookError(PortfolioWalkForwardError):
    """Stable refusal for a tranche recipe or allocation failure.

    Subclasses the shared walk-forward error rather than ``ValueError`` because
    these refusals surface during a walk, where existing handlers already catch
    that type. Naming its own class keeps a tranche failure distinguishable from
    an unrelated policy's.
    """


@dataclass(frozen=True, slots=True)
class CappedWeights:
    """Capped weights plus how they were reached.

    The disposition is returned rather than logged because two of the three ways
    out are degenerate fallbacks to equal weight. A caller that cannot see which
    one happened would read a concentration-disciplined book and an emergency
    flat book as the same result.
    """

    weights: FloatArray
    disposition: CapDisposition


class TrancheBookRecipe(BaseModel):  # type: ignore[misc]
    """The public closed-form book's controls, with their admitted ranges.

    Defaults and ranges live together here so the runner, report, UI and Agent
    tools do not each restate them. Only the default tuple is the frozen product
    strategy; every other admitted value is a bounded development exploration
    with its own configuration identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["TrancheBookRecipe"] = "TrancheBookRecipe"
    top_k: int = Field(default=DEFAULT_TOP_K, ge=TOP_K_RANGE[0], le=TOP_K_RANGE[1])
    tranches: int = Field(default=DEFAULT_TRANCHES, ge=TRANCHES_RANGE[0], le=TRANCHES_RANGE[1])
    exit_rank: int = Field(default=DEFAULT_EXIT_RANK, ge=TOP_K_RANGE[0])
    weight_rule: TrancheWeightRule = DEFAULT_WEIGHT_RULE
    sleeve_cap_equal_weight_multiple: float = SLEEVE_CAP_EQUAL_WEIGHT_MULTIPLE
    aggregate_name_cap: float = AGGREGATE_NAME_CAP
    sleeve_share_policy: Literal["EQUAL_NOTIONAL_AT_EACH_FORMATION"] = SLEEVE_SHARE_POLICY
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        top_k: int = DEFAULT_TOP_K,
        tranches: int = DEFAULT_TRANCHES,
        exit_rank: int | None = None,
        weight_rule: TrancheWeightRule = DEFAULT_WEIGHT_RULE,
    ) -> Self:
        """Seal admitted tranche selection/sizing controls with frozen cap and sleeve-share rules.

        Args:
            top_k: Declared per-sleeve selection size.
            tranches: Number of retained sleeves.
            exit_rank: Explicit rank or twice top_k when omitted.
            weight_rule: Admitted deterministic sleeve sizing rule.

        Returns:
            Validated recipe with canonical recipe_hash.
        """
        values: dict[str, object] = {
            "kind": "TrancheBookRecipe",
            "top_k": top_k,
            "tranches": tranches,
            "exit_rank": 2 * top_k if exit_rank is None else exit_rank,
            "weight_rule": weight_rule,
            "sleeve_cap_equal_weight_multiple": SLEEVE_CAP_EQUAL_WEIGHT_MULTIPLE,
            "aggregate_name_cap": AGGREGATE_NAME_CAP,
            "sleeve_share_policy": SLEEVE_SHARE_POLICY,
        }
        return cls(**values, recipe_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_recipe(self) -> Self:
        # exit_rank is expressed against top_k rather than as a free integer,
        # because the measured ladder is flat in the multiple and not in the
        # absolute rank: 70 means "twice the sleeve", and it means that at every
        # width the user can select.
        """Require admitted exit-rank multiple, sizing rule, frozen caps and exact identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            TrancheBookError: Exit-rank range, weight rule, frozen cap/share policy or recipe_hash
                differs.
        """
        lower, upper = EXIT_RANK_MULTIPLE_RANGE
        if not lower * self.top_k <= self.exit_rank <= upper * self.top_k:
            raise TrancheBookError("portfolio_strategy_lab.tranche_exit_rank_out_of_range")
        if self.weight_rule not in ADMITTED_WEIGHT_RULES:
            raise TrancheBookError("portfolio_strategy_lab.tranche_weight_rule_not_admitted")
        if (
            self.sleeve_cap_equal_weight_multiple != SLEEVE_CAP_EQUAL_WEIGHT_MULTIPLE
            or self.aggregate_name_cap != AGGREGATE_NAME_CAP
            or self.sleeve_share_policy != SLEEVE_SHARE_POLICY
        ):
            raise TrancheBookError("portfolio_strategy_lab.tranche_caps_not_frozen")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"recipe_hash"}))
        if self.recipe_hash != expected:
            raise TrancheBookError("portfolio_strategy_lab.tranche_recipe_identity_invalid")
        return self

    @property
    def maximum_weight(self) -> float:
        """The largest share any one name may hold in the assembled book.

        Required by the ``PortfolioPolicyRecipe`` structural contract, which the
        catalog's own type signature names. Without it this recipe only happened
        to work because ``resolve`` reads ``policy_id`` alone; any generic
        consumer reading the declared contract would have failed on it.

        It is the aggregate name cap rather than the sleeve ceiling, because the
        sleeve ceiling bounds a name's share of one sleeve and this bounds its
        share of the book -- the quantity the contract is asking about.
        """
        return self.aggregate_name_cap

    @property
    def policy_id(self) -> str:
        """The handle ``PortfolioPolicyCatalog.resolve`` looks this recipe up by.

        Without it a recipe cannot be resolved through the installed catalog at
        all, which would leave the public book reachable only by calling its
        functions directly -- exactly the parallel path the delivery plan refuses.
        """
        return TRANCHE_BOOK_POLICY_ID

    @property
    def exit_rank_multiple(self) -> float:
        """Read exit rank relative to the declared sleeve width.

        Returns:
            exit_rank divided by top_k.
        """
        return self.exit_rank / self.top_k

    @property
    def sleeve_ceiling(self) -> float:
        """One sleeve name's maximum share of its own sleeve."""
        return self.sleeve_cap_equal_weight_multiple / self.top_k

    @property
    def consumes_mu(self) -> bool:
        """Read whether this sizing rule consumes causal rank-mu input.

        Returns:
            True exactly when weight_rule begins with mu.
        """
        return self.weight_rule.startswith("mu.")

    @property
    def consumes_risk(self) -> bool:
        """Read whether this sizing rule consumes per-name Risk scale.

        Returns:
            True for every rule other than ew.
        """
        return self.weight_rule != "ew"


def due_sleeves(*, formation_index: int, tranches: int) -> tuple[int, ...]:
    """Which sleeves are reviewed at this *schedule position*.

    Every sleeve is staged at position zero, because a book that reviewed one
    sleeve per session from empty would take `T` sessions to become invested and
    its first weeks would measure the ramp rather than the strategy. After that
    exactly one sleeve is due, which is what makes the schedule phase-free: the
    offset only relabels which sleeve goes first.

    The argument is a position on the *policy's* schedule, not an index into a
    caller's arrays. Those coincide for a path that runs from nothing in one
    segment, and they must not be assumed to coincide anywhere else: position
    zero means "this book has never traded", and a continued book has.
    """
    if tranches < 1 or formation_index < 0:
        raise TrancheBookError("portfolio_strategy_lab.tranche_schedule_axis_invalid")
    if formation_index == 0:
        return tuple(range(tranches))
    return (formation_index % tranches,)


def cap_and_renormalise(raw: FloatArray, *, ceiling: float) -> CappedWeights:
    """Water-fill to the ceiling, preserving the order the weighting rule produced.

    Capped names are **pinned**. That is the whole correctness argument, and the
    previous version got it wrong: it recomputed the over-ceiling set each pass
    as ``weights > ceiling``, so a name sitting at exactly the ceiling was not
    "over" and fell back into the redistribution pool, received more weight, and
    went over again. On ``[0.5, 0.3, 0.2]`` at ``0.34`` that oscillates forever
    and eventually reports failure, even though ``[0.34, 0.34, 0.32]`` is a
    perfectly good answer.

    The corrected loop keeps a monotonically growing capped set: pinned names
    hold exactly the ceiling, and the mass they do not use is shared among the
    still-free names in proportion to their original weights. Each pass pins at
    least one more name, so it terminates in at most one pass per held name, and
    it finds the solution whenever ``ceiling * held >= 1``.

    Everything stays confined to the support -- the names already carrying
    positive weight -- because the zeros can absorb nothing and a long-only book
    must not open a position the selection never made.
    """
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise TrancheBookError("portfolio_strategy_lab.tranche_weight_vector_invalid")
    if not np.isfinite(ceiling) or ceiling <= 0.0:
        raise TrancheBookError("portfolio_strategy_lab.tranche_ceiling_invalid")

    positive = np.clip(values, 0.0, None)
    total = float(positive.sum())
    if total <= 0.0:
        flat: FloatArray = np.full(values.size, 1.0 / values.size, dtype=np.float64)
        return CappedWeights(weights=flat, disposition="EQUAL_WEIGHT_NO_POSITIVE_MASS")

    base = positive / total
    support = base > 0.0
    held = int(support.sum())
    even: FloatArray = np.zeros(values.size, dtype=np.float64)
    even[support] = 1.0 / held

    # Genuine infeasibility: no allocation over the held names can meet the
    # ceiling, so the honest answer is the flat book across exactly those names.
    if ceiling * held < 1.0 - _CAP_TOLERANCE:
        return CappedWeights(weights=even, disposition="EQUAL_WEIGHT_CEILING_INFEASIBLE")

    capped = np.zeros(values.size, dtype=np.bool_)
    weights: FloatArray = np.zeros(values.size, dtype=np.float64)
    for _ in range(held + 1):
        free = support & ~capped
        remaining = 1.0 - ceiling * float(capped.sum())
        if not free.any() or remaining < -_CAP_TOLERANCE:
            return CappedWeights(weights=even, disposition="EQUAL_WEIGHT_CEILING_INFEASIBLE")
        weights = np.zeros(values.size, dtype=np.float64)
        weights[capped] = ceiling
        share = base[free]
        mass = float(share.sum())
        weights[free] = (
            remaining * share / mass
            if mass > 0.0
            else np.full(int(free.sum()), remaining / float(free.sum()))
        )
        newly = free & (weights > ceiling + _CAP_TOLERANCE)
        if not newly.any():
            break
        capped |= newly

    if float(weights.max()) > ceiling + _CAP_TOLERANCE:
        return CappedWeights(weights=even, disposition="EQUAL_WEIGHT_CEILING_INFEASIBLE")
    return CappedWeights(weights=weights, disposition="CAPPED")


def sleeve_target_weights(
    *,
    mu: FloatArray | None,
    variance: FloatArray,
    rule: SleeveWeightRule,
    ceiling: float,
) -> CappedWeights:
    """One sleeve's internal weights, before it is given a share of the book.

    ``mu`` is required by the ``mu.*`` rules and refused by the others, rather
    than defaulted: a rule that silently fell back to inverse volatility when its
    expected-return lane was missing would report itself as ``mu.iv1`` while
    trading ``iv1``.
    """
    variances = np.asarray(variance, dtype=np.float64)
    if variances.ndim != 1 or variances.size == 0:
        raise TrancheBookError("portfolio_strategy_lab.tranche_sleeve_axis_invalid")
    if rule == "ew":
        return cap_and_renormalise(np.full(variances.size, 1.0, dtype=np.float64), ceiling=ceiling)

    deviations = np.sqrt(np.clip(variances, _VARIANCE_FLOOR, None))
    exponent = float(rule.split("iv")[-1])
    raw: FloatArray = 1.0 / np.clip(deviations**exponent, _VARIANCE_FLOOR, None)

    if rule.startswith("mu."):
        if mu is None:
            raise TrancheBookError("portfolio_strategy_lab.tranche_mu_lane_absent")
        means = np.asarray(mu, dtype=np.float64)
        if means.shape != variances.shape or not np.isfinite(means).all():
            raise TrancheBookError("portfolio_strategy_lab.tranche_mu_axis_invalid")
        # Bucket means are simple returns and go negative. Lifting by the
        # minimum makes the worst admitted bucket exactly zero, and the floor
        # then keeps it a small positive weight rather than an exclusion the
        # membership rule never voted for.
        lifted = means - float(means.min())
        spread = float(lifted.max())
        tilt = lifted + MU_FLOOR_SPREAD_SHARE * spread if spread > 0.0 else np.ones_like(lifted)
        raw = raw * tilt
    elif mu is not None:
        raise TrancheBookError("portfolio_strategy_lab.tranche_mu_lane_not_consumed")

    return cap_and_renormalise(raw, ceiling=ceiling)


def rank_bucket_mu(
    *,
    selected: IntArray,
    ranked_positions: IntArray,
    eligible_count: int,
    bucket_means: FloatArray,
) -> FloatArray:
    """Map each selected name to its rank bucket's mean, absent buckets scoring zero.

    A non-finite bucket mean is a bucket the curve owner could not support at
    this formation. It becomes ``0.0`` here, which after the lift above is the
    neutral position in the tilt rather than a fabricated forecast.
    """
    buckets = np.asarray(bucket_means, dtype=np.float64)
    if buckets.ndim != 1 or buckets.size == 0 or eligible_count <= 0:
        raise TrancheBookError("portfolio_strategy_lab.tranche_bucket_axis_invalid")
    positions = np.asarray(ranked_positions, dtype=np.int64)[np.asarray(selected, dtype=np.int64)]
    slots = np.clip((positions * buckets.size) // eligible_count, 0, buckets.size - 1)
    chosen = buckets[slots]
    return np.asarray(np.where(np.isfinite(chosen), chosen, 0.0), dtype=np.float64)


def aggregate_sleeve_book(
    *,
    sleeve_books: tuple[FloatArray, ...],
    sleeve_shares: tuple[float, ...],
    aggregate_name_cap: float = AGGREGATE_NAME_CAP,
) -> CappedWeights:
    """Assemble the sleeves into one book and apply the aggregate name cap once.

    Sleeves overlap: the same name can sit in more than one, which is why the
    book holds fewer distinct names than ``top_k * tranches`` and why the
    aggregate cap is applied to the assembled vector rather than inside each
    sleeve. Applying it per sleeve would bind on a name no single sleeve
    over-weights.
    """
    if not sleeve_books or len(sleeve_books) != len(sleeve_shares):
        raise TrancheBookError("portfolio_strategy_lab.tranche_sleeve_count_invalid")
    width = sleeve_books[0].size
    if any(book.size != width for book in sleeve_books):
        raise TrancheBookError("portfolio_strategy_lab.tranche_sleeve_axis_mismatch")
    if any(share < 0.0 for share in sleeve_shares) or sum(sleeve_shares) <= 0.0:
        raise TrancheBookError("portfolio_strategy_lab.tranche_sleeve_share_invalid")

    book: FloatArray = np.zeros(width, dtype=np.float64)
    for weights, share in zip(sleeve_books, sleeve_shares, strict=True):
        book = book + share * np.asarray(weights, dtype=np.float64)
    return cap_and_renormalise(book, ceiling=aggregate_name_cap)


@dataclass(frozen=True, slots=True)
class TrancheBookAllocation:
    """One formation's assembled book, and the evidence behind it."""

    target_weights: FloatArray
    uncapped_target_weights: FloatArray
    risk_projection_hash: str | None
    """The exact risk lane these holdings were decided against, or ``None`` for ``ew``.

    Bound here so the holdings identity names its consumed risk input. It is the
    *projection* hash and never the surface hash: a report-only revision to the
    factor block rotates the surface while the consumed lane stands still, and
    the ledger must not move for that.
    """

    sleeve_weights: tuple[FloatArray, ...]
    sleeve_shares: tuple[float, ...]
    """The share each sleeve was given, returned rather than left implicit.

    The caller supplied the state these were read from, so returning them keeps
    one set of numbers in play: an executor that re-derived the shares to rebuild
    its own sleeve state could disagree with the allocation that was actually
    made, and nothing would catch it.
    """

    reviewed_sleeves: tuple[int, ...]
    sleeve_dispositions: tuple[CapDisposition, ...]
    book_disposition: CapDisposition
    aggregate_cap_binding_count: int
    curve_disposition: str | None


def _admitted_risk_scale(inputs: BoundPolicyDecisionInput, *, listings: int) -> PerNameRiskScale:
    """Admit the per-name lane, or refuse before any weight is decided.

    Every check here is one an unadmitted lane would pass silently: a projection
    for the wrong universe, one whose axis has drifted against the score axis, or
    one carrying a non-positive volatility that would divide into an infinite
    weight.
    """

    scale = inputs.risk_allocation
    if scale is None:
        raise TrancheBookError("portfolio_strategy_lab.tranche_risk_projection_absent")
    try:
        scale.verify_content()
    except ValueError as error:
        raise TrancheBookError("portfolio_strategy_lab.tranche_risk_projection_invalid") from error
    if inputs.formation_session is not None and scale.formation_session != inputs.formation_session:
        raise TrancheBookError("portfolio_strategy_lab.tranche_risk_session_mismatch")
    if inputs.ordered_listing_ids is not None and (
        tuple(scale.ordered_listing_ids) != tuple(inputs.ordered_listing_ids)
    ):
        raise TrancheBookError("portfolio_strategy_lab.tranche_risk_listing_axis_mismatch")
    volatility = np.asarray(scale.per_name_volatility, dtype=np.float64)
    if volatility.shape != (listings,):
        raise TrancheBookError("portfolio_strategy_lab.tranche_risk_axis_length_invalid")
    if not np.isfinite(volatility).all():
        raise TrancheBookError("portfolio_strategy_lab.tranche_risk_volatility_not_finite")
    if bool(np.any(volatility <= 0.0)):
        raise TrancheBookError("portfolio_strategy_lab.tranche_risk_volatility_not_positive")
    return scale


def decide_tranche_book(
    *,
    recipe: TrancheBookRecipe,
    inputs: BoundPolicyDecisionInput,
) -> TrancheBookAllocation:
    """Review the due sleeve, carry the rest, assemble the book.

    The sleeves that are not due are passed through untouched, which is the
    whole point of the schedule: only one sleeve trades at a formation, so the
    others must produce exactly the weights they already held. Recomputing them
    would turn a three-sleeve book into a whole-book policy that rebalances
    every session and trades three times as much.
    """
    if inputs.ordered_listing_ids is None:
        raise TrancheBookError("portfolio_strategy_lab.tranche_listing_axis_absent")
    if inputs.formation_index is None:
        raise TrancheBookError("portfolio_strategy_lab.tranche_formation_index_absent")

    listings = inputs.scores.size
    variance: FloatArray | None = None
    risk_projection_hash: str | None = None
    if recipe.consumes_risk:
        scale = _admitted_risk_scale(inputs, listings=listings)
        variance = np.square(np.asarray(scale.per_name_volatility, dtype=np.float64))
        risk_projection_hash = scale.projection_hash
    elif inputs.risk_allocation is not None:
        # ``ew`` consumes no risk at all. Accepting a lane it will not read would
        # let a book advertise a risk input it never used.
        raise TrancheBookError("portfolio_strategy_lab.tranche_risk_lane_not_consumed")

    eligible = np.flatnonzero(
        np.asarray(inputs.decision_eligible, dtype=np.bool_) & np.isfinite(inputs.scores)
    )
    ids = np.asarray(inputs.ordered_listing_ids, dtype=str)
    order = eligible[np.lexsort((ids[eligible], -inputs.scores[eligible]))]
    positions: IntArray = np.full(listings, order.size, dtype=np.int64)
    positions[order] = np.arange(order.size, dtype=np.int64)

    # The schedule position, which is the formation index only for a book that
    # started from nothing in this same segment.
    position = (
        inputs.formation_index if inputs.schedule_position is None else inputs.schedule_position
    )
    due = due_sleeves(formation_index=position, tranches=recipe.tranches)
    previous = inputs.previous_sleeve_weights
    shares = tuple(1.0 / recipe.tranches for _ in range(recipe.tranches))
    if previous is None:
        if set(due) != set(range(recipe.tranches)):
            raise TrancheBookError("portfolio_strategy_lab.tranche_sleeve_state_absent")
        prior: tuple[FloatArray | None, ...] = tuple(None for _ in range(recipe.tranches))
    else:
        if len(previous) != recipe.tranches or any(book.size != listings for book in previous):
            raise TrancheBookError("portfolio_strategy_lab.tranche_sleeve_state_axis_invalid")
        masses = tuple(float(np.asarray(book, dtype=np.float64).sum()) for book in previous)
        if any(mass < 0.0 for mass in masses) or sum(masses) <= 0.0:
            raise TrancheBookError("portfolio_strategy_lab.tranche_sleeve_state_mass_invalid")
        if any(mass <= 0.0 for mass in masses):
            raise TrancheBookError("portfolio_strategy_lab.tranche_sleeve_state_mass_invalid")
        # The frozen public plan says equal sleeve notionals.  Drift is still
        # read from the shared engine, but each carried sleeve is normalized
        # back to its own internal book before the equal 1/T shares are applied.
        # This is an explicit recipe identity, not an executor default.
        prior = tuple(
            np.asarray(book, dtype=np.float64) / mass
            for book, mass in zip(previous, masses, strict=True)
        )

    curve = inputs.causal_rank_return_curve
    if recipe.consumes_mu:
        # The frozen plan is explicit: "A missing or partial `mu` surface fails
        # closed for a `mu.*` rule; it does not silently become `iv1`." Partial
        # is therefore as fatal as absent. The predecessor C6 convention of
        # mapping unsupported buckets to zero and trading on is a *predecessor*
        # convention; carrying it here would have let a `mu.iv1` book quietly
        # trade an inverse-volatility tilt over the buckets its curve could not
        # support, while still reporting itself as `mu.iv1`.
        if curve is None:
            raise TrancheBookError("portfolio_strategy_lab.tranche_curve_absent")
        if curve.disposition != "AVAILABLE":
            raise TrancheBookError("portfolio_strategy_lab.tranche_curve_not_available")
        if curve.formation_index != inputs.formation_index:
            raise TrancheBookError("portfolio_strategy_lab.tranche_curve_formation_mismatch")
        if (
            inputs.formation_session is not None
            and curve.formation_session != inputs.formation_session
        ):
            raise TrancheBookError("portfolio_strategy_lab.tranche_curve_session_mismatch")
        if not np.isfinite(np.asarray(curve.bucket_means, dtype=np.float64)).all():
            raise TrancheBookError("portfolio_strategy_lab.tranche_curve_bucket_not_finite")
    elif curve is not None:
        raise TrancheBookError("portfolio_strategy_lab.tranche_curve_not_consumed")

    books: list[FloatArray] = []
    dispositions: list[CapDisposition] = []
    for sleeve in range(recipe.tranches):
        held = prior[sleeve]
        if sleeve not in due:
            if held is None:
                raise TrancheBookError("portfolio_strategy_lab.tranche_sleeve_state_absent")
            books.append(held)
            dispositions.append("CAPPED")
            continue

        selected = whole_book_hysteresis_selection(
            scores=inputs.scores,
            decision_eligible=inputs.decision_eligible,
            ordered_listing_ids=inputs.ordered_listing_ids,
            previous_target_weights=held,
            top_k=recipe.top_k,
            exit_rank=recipe.exit_rank,
        )
        mu: FloatArray | None = None
        if recipe.consumes_mu and curve is not None:
            mu = rank_bucket_mu(
                selected=selected,
                ranked_positions=positions,
                eligible_count=max(order.size, 1),
                bucket_means=curve.bucket_means,
            )
        sleeve_variance = (
            variance[selected] if variance is not None else np.ones(selected.size, dtype=np.float64)
        )
        inner = sleeve_target_weights(
            mu=mu,
            variance=sleeve_variance,
            rule=recipe.weight_rule,
            ceiling=recipe.sleeve_ceiling,
        )
        if inner.disposition != "CAPPED":
            # A sleeve that fell back to equal weight did not honour the sleeve
            # ceiling the recipe froze. Trading it would put a book on the tape
            # that violates its own stated concentration discipline, so the
            # public path refuses instead of returning a plausible-looking
            # vector whose disposition the caller can discard.
            raise TrancheBookError("portfolio_strategy_lab.tranche_sleeve_cap_not_honoured")
        book: FloatArray = np.zeros(listings, dtype=np.float64)
        book[selected] = inner.weights
        books.append(book)
        dispositions.append(inner.disposition)

    uncapped = np.zeros(listings, dtype=np.float64)
    for book, share in zip(books, shares, strict=True):
        uncapped = uncapped + share * book
    uncapped = uncapped / float(uncapped.sum())
    aggregate_cap_binding_count = int(np.sum(uncapped > recipe.aggregate_name_cap + _CAP_TOLERANCE))
    assembled = aggregate_sleeve_book(
        sleeve_books=tuple(books),
        sleeve_shares=shares,
        aggregate_name_cap=recipe.aggregate_name_cap,
    )
    if assembled.disposition != "CAPPED":
        # Same rule at the book level, and the more dangerous of the two: the
        # aggregate fallback is a flat book that can hold a name above the 6%
        # ceiling the recipe declares.
        raise TrancheBookError("portfolio_strategy_lab.tranche_aggregate_cap_not_honoured")
    return TrancheBookAllocation(
        target_weights=assembled.weights,
        uncapped_target_weights=uncapped,
        risk_projection_hash=risk_projection_hash,
        sleeve_weights=tuple(books),
        sleeve_shares=shares,
        reviewed_sleeves=due,
        sleeve_dispositions=tuple(dispositions),
        book_disposition=assembled.disposition,
        aggregate_cap_binding_count=aggregate_cap_binding_count,
        curve_disposition=None if curve is None else curve.disposition,
    )


class TrancheBookAdapter:
    """The public closed-form book: scheduled sleeves, hysteresis, diagonal risk only."""

    policy_id = TRANCHE_BOOK_POLICY_ID
    solver_backed = False
    selection_allocation_semantics = (
        PortfolioSelectionAllocationSemantics.closed_form_tranche_sleeve_schedule()
    )

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        """Bind closed-form sleeve scheduling, selection and declared scale-only Risk input.

        Returns:
            Exact binding for per-name Risk scale without a covariance forecast or optimizer.
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
            recipe_schema_id=TRANCHE_BOOK_POLICY_ID,
            solver_semantics="CLOSED_FORM_TRANCHE_SLEEVE_SCHEDULE_NO_OPTIMIZER",
            # Both of these default to the solver-backed answer, so omitting
            # them made a closed-form adapter declare that it required an
            # optimizer and state no selection semantics at all.
            selection_allocation_semantics=self.selection_allocation_semantics,
            input_consumption_semantics=("PER_NAME_RISK_SCALE_REQUIRED_NO_FORECAST_OR_OPTIMIZER"),
            deterministic_policy={
                "selection": (
                    "per-sleeve-stable-score-listing-id-top-k-with-previous-target-exit-rank"
                ),
                "schedule": "one-due-sleeve-per-formation-all-staged-on-the-first",
                "allocation": "lifted-rank-bucket-mu-over-diagonal-volatility-with-sleeve-ceiling",
                "aggregation": "declared-sleeve-shares-then-one-aggregate-name-cap",
                "mu_lane": "owner-supplied-causal-rank-bucket-curve-never-a-caller-made-lane",
            },
        )

    def validate_policy(self, policy: PortfolioPolicyRecipe) -> TrancheBookRecipe:
        # Annotated local rather than a bare return: ``model_validate`` is typed
        # ``Any`` here, and returning it directly would hand the caller an
        # unchecked value through a signature that promises a recipe.
        """Validate one concrete tranche recipe through its contract owner.

        Args:
            policy: Proposed recipe contract.

        Returns:
            Admitted TrancheBookRecipe.

        Raises:
            pydantic.ValidationError: Recipe fields or declared consistency fail.
        """
        recipe: TrancheBookRecipe = TrancheBookRecipe.model_validate(policy)
        return recipe

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: object,
    ) -> PortfolioTargetDecision:
        """Advance closed-form tranche allocation without publishing a diagonal Risk proxy.

        Args:
            policy: Concrete admitted installed recipe.
            inputs: Exact scores, eligibility, holdings reference and declared auxiliary lanes.
            optimizer: Deterministic solver owner; closed-form adapters do not use it.

        Returns:
            Target decision with no predicted_variance and no extra Risk forecast requirement; total
            Risk belongs to attribution/report owners.

        Raises:
            TrancheBookError: Recipe or deterministic tranche allocation admission fails.
        """
        del optimizer
        recipe = self.validate_policy(policy)
        allocation = decide_tranche_book(recipe=recipe, inputs=inputs)
        return PortfolioTargetDecision(
            target_weights=allocation.target_weights,
            # The allocation lane has no correlation term. Publishing its
            # diagonal quadratic form through this field would make shared
            # Backtesting calibrate it as a total covariance forecast. Total
            # predicted Risk belongs to the attribution projection/report owner.
            predicted_variance=None,
            requires_risk_forecast=False,
        )


INSTALLED_TRANCHE_BOOK_RECIPE: Final = TrancheBookRecipe.create()
"""The frozen public default: three sleeves of 35, exit 70, ``mu.iv1``."""


__all__ = [
    "ADMITTED_WEIGHT_RULES",
    "AGGREGATE_NAME_CAP",
    "DEFAULT_EXIT_RANK",
    "DEFAULT_TOP_K",
    "DEFAULT_TRANCHES",
    "DEFAULT_WEIGHT_RULE",
    "INSTALLED_TRANCHE_BOOK_RECIPE",
    "MU_FLOOR_SPREAD_SHARE",
    "SLEEVE_CAP_EQUAL_WEIGHT_MULTIPLE",
    "TRANCHE_BOOK_POLICY_ID",
    "CapDisposition",
    "CappedWeights",
    "FrozenSleeveWeightRule",
    "SleeveWeightRule",
    "TrancheBookAdapter",
    "TrancheBookAllocation",
    "TrancheBookError",
    "TrancheBookRecipe",
    "TrancheWeightRule",
    "aggregate_sleeve_book",
    "cap_and_renormalise",
    "decide_tranche_book",
    "due_sleeves",
    "rank_bucket_mu",
    "sleeve_target_weights",
]
