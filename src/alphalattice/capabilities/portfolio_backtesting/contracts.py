"""Typed causal path contracts shared by Portfolio research and Validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Final, Literal, Protocol, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]


class PortfolioWalkForwardError(ValueError):
    """Stable causal execution or evaluation failure."""


DEFAULT_REPORTING_COST_BPS: tuple[int, ...] = (5, 10, 20)
"""The reporting lanes this engine used as module literals until they were declared."""

DEFAULT_SELECTION_COST_BPS = 10
"""The lane a selection reads. Also a literal here until it became a parameter."""


class PortfolioTransactionCostConvention(BaseModel):  # type: ignore[misc]
    """The installed meaning of a Portfolio transaction-cost rate.

    A rate in basis points is not self-describing: it could be per fill side,
    per round trip, or per a notional that includes cash.  The causal execution
    path records only asset-weight turnover, so this contract says exactly how
    that stored lane is charged.  It is nested in current execution bindings;
    a missing convention is intentionally a legacy disposition, never an
    invitation to attach this label to an older artifact.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["PortfolioTransactionCostConvention"] = "PortfolioTransactionCostConvention"
    convention_id: Literal["ONE_WAY_ASSET_TURNOVER_BPS"] = "ONE_WAY_ASSET_TURNOVER_BPS"
    rate_basis: Literal["BASIS_POINTS_PER_ONE_WAY_ASSET_TURNOVER"] = (
        "BASIS_POINTS_PER_ONE_WAY_ASSET_TURNOVER"
    )
    turnover_definition: Literal["HALF_L1_EXECUTED_ASSET_WEIGHT_DELTA"] = (
        "HALF_L1_EXECUTED_ASSET_WEIGHT_DELTA"
    )
    cash_treatment: Literal["CASH_EXCLUDED_FROM_CAUSAL_ASSET_TURNOVER"] = (
        "CASH_EXCLUDED_FROM_CAUSAL_ASSET_TURNOVER"
    )
    net_simple_return_equation: Literal[
        "GROSS_SIMPLE_RETURN_MINUS_ONE_WAY_TURNOVER_TIMES_BPS_OVER_10000"
    ] = "GROSS_SIMPLE_RETURN_MINUS_ONE_WAY_TURNOVER_TIMES_BPS_OVER_10000"
    round_trip_interpretation: Literal[
        "FULL_A_TO_B_ASSET_ROTATION_CHARGES_DECLARED_BPS_ONCE_OUT_AND_BACK_TWICE"
    ] = "FULL_A_TO_B_ASSET_ROTATION_CHARGES_DECLARED_BPS_ONCE_OUT_AND_BACK_TWICE"
    human_readable_label: Literal["bps per one-way asset turnover; not a per-side rate"] = (
        "bps per one-way asset turnover; not a per-side rate"
    )
    convention_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def installed(cls) -> PortfolioTransactionCostConvention:
        """Build the installed one-way asset-turnover cost convention.

        Returns:
            The sealed convention with its fixed rate basis, cash treatment and return equation.
        """
        return cls._sealed(
            {
                "kind": "PortfolioTransactionCostConvention",
                "convention_id": "ONE_WAY_ASSET_TURNOVER_BPS",
                "rate_basis": "BASIS_POINTS_PER_ONE_WAY_ASSET_TURNOVER",
                "turnover_definition": "HALF_L1_EXECUTED_ASSET_WEIGHT_DELTA",
                "cash_treatment": "CASH_EXCLUDED_FROM_CAUSAL_ASSET_TURNOVER",
                "net_simple_return_equation": (
                    "GROSS_SIMPLE_RETURN_MINUS_ONE_WAY_TURNOVER_TIMES_BPS_OVER_10000"
                ),
                "round_trip_interpretation": (
                    "FULL_A_TO_B_ASSET_ROTATION_CHARGES_DECLARED_BPS_ONCE_OUT_AND_BACK_TWICE"
                ),
                "human_readable_label": "bps per one-way asset turnover; not a per-side rate",
            }
        )

    @classmethod
    def _sealed(cls, values: dict[str, str]) -> PortfolioTransactionCostConvention:
        return cls(**values, convention_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> PortfolioTransactionCostConvention:
        """Require the canonical identity of the complete cost convention.

        Returns:
            This validated convention.

        Raises:
            PortfolioWalkForwardError: The convention hash differs from its fields.
        """
        if self.convention_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"convention_hash"})
        ):
            raise PortfolioWalkForwardError(
                "portfolio_backtesting.transaction_cost_convention_identity_invalid"
            )
        return self

    def cost_fraction(self, *, one_way_turnover: FloatArray, cost_bps: float) -> FloatArray:
        """Convert one-way asset turnover and its basis-point rate into session costs.

        Args:
            one_way_turnover: Finite nonnegative asset-turnover values, excluding cash.
            cost_bps: Finite nonnegative rate charged per unit of one-way asset turnover.

        Returns:
            Session cost fractions, equal to turnover times cost_bps divided by 10,000.

        Raises:
            PortfolioWalkForwardError: A turnover value or the rate is nonfinite or negative.
        """
        turnover = np.asarray(one_way_turnover, dtype=np.float64)
        if (
            not np.isfinite(cost_bps)
            or cost_bps < 0.0
            or not np.isfinite(turnover).all()
            or bool(np.any(turnover < 0.0))
        ):
            raise PortfolioWalkForwardError("portfolio_backtesting.transaction_cost_input_invalid")
        return np.asarray(turnover * (cost_bps / 10_000.0), dtype=np.float64)

    def researcher_payload(self) -> dict[str, str]:
        """Expose the sealed cost convention as JSON-compatible descriptive fields.

        Returns:
            The complete convention payload, including its identity and rate interpretation.
        """
        return cast(dict[str, str], self.model_dump(mode="json"))


DEFAULT_PORTFOLIO_TRANSACTION_COST_CONVENTION = PortfolioTransactionCostConvention.installed()
"""The only convention new Portfolio causal execution may bind and publish."""


PER_SIDE_TO_PLATFORM_ONE_WAY_MULTIPLIER: Final = 2
"""A round trip charges the per-side rate twice: once selling, once buying."""


def platform_one_way_cost_from_per_side(per_side: int) -> int:
    """The one mapping from an externally quoted per-side rate into this engine's lane.

    The engine charges ``turnover * platform_bps / 10_000`` against one-way
    asset turnover, which is not a per-side rate -- its own installed label says
    so. A caller who thinks in per-side terms therefore needs a conversion, and
    the conversion is exactly one multiplication.

    It lives here, once, because the alternative is what the source already
    contained: the same ``2 *`` written out inside a baseline package validator,
    where a later reader cannot tell whether it is the authority or a copy of
    one. Both representations in the tree -- the baseline package's integer
    basis points and the public assumption's tenths -- call this function, so
    there is one arithmetic owner and no second definition free to drift.

    The argument is an integer in whatever unit the caller keeps, because
    doubling is unit-free. Sign is the only thing checked here; range and
    increment belong to the typed assumption below.
    """
    if per_side < 0:
        raise PortfolioWalkForwardError("portfolio_backtesting.per_side_cost_negative")
    return PER_SIDE_TO_PLATFORM_ONE_WAY_MULTIPLIER * per_side


COST_BPS_PER_SIDE_TENTHS_PER_BP: Final = 10
"""The fixed-point scale: the stored integer counts tenths of a basis point."""

COST_BPS_PER_SIDE_INCREMENT_TENTHS: Final = 5
"""The admitted increment, ``0.5`` bps, frozen with the schema rather than in a UI.

Chosen for a measurable reason rather than taste. Every multiple of ``0.5`` is
exactly representable in binary floating point, so the single conversion this
type performs -- tenths to the ``float`` rate the engine consumes -- is exact for
every admitted value, ``2.5`` included. A finer increment such as ``0.1`` would
admit values like ``0.3`` that no ``float`` holds exactly, which is the silently
rounded mapping this owner exists to refuse.
"""

MAXIMUM_COST_BPS_PER_SIDE_TENTHS: Final = 200
"""``20.0`` bps per side, the top of the admitted control range."""


class PortfolioPerSideCostAssumption(BaseModel):  # type: ignore[misc]
    """An externally quoted per-side cost, exact and convertible into the platform lane.

    Two things stay deliberately separate. The convention above states what the
    engine's stored rate means and never changes. This assumption is a caller's
    quote about their own broker: it changes freely and rotates only the
    economic descendants of a path, never its holdings.

    The value is an integer count of tenths of a basis point rather than a
    ``float`` figure in basis points. A per-side ladder that must express ``2.5``
    cannot be integer basis points, and holding it as a ``float`` would place an
    inexact value inside a content hash, so the exact representation is a scaled
    integer with a frozen increment.

    This type performs no cost arithmetic beyond the shared mapping. It converts
    and it refuses; ``PortfolioCostPolicy`` still owns charging.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["PortfolioPerSideCostAssumption"] = "PortfolioPerSideCostAssumption"
    quote_basis: Literal["PER_SIDE"] = "PER_SIDE"
    cost_bps_per_side_tenths: int = Field(ge=0, le=MAXIMUM_COST_BPS_PER_SIDE_TENTHS)
    assumption_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, *, cost_bps_per_side_tenths: int) -> Self:
        """Seal an exact integer quote in tenths of a basis point per side.

        Args:
            cost_bps_per_side_tenths: Nonnegative quote within the admitted range and
                half-basis-point increment.

        Returns:
            The validated fixed-point assumption and its canonical identity.

        Raises:
            PortfolioWalkForwardError: The quote is off the admitted increment or its identity is
                inconsistent.
        """
        values: dict[str, object] = {
            "kind": "PortfolioPerSideCostAssumption",
            "quote_basis": "PER_SIDE",
            "cost_bps_per_side_tenths": cost_bps_per_side_tenths,
        }
        return cls(**values, assumption_hash=canonical_hash(values))

    @classmethod
    def from_bps_per_side(cls, value: Decimal | str | int) -> Self:
        """Build from a quoted rate such as ``"2.5"``, refusing anything off the increment.

        A ``float`` is not accepted. ``0.1`` is not the value it prints as, and
        admitting one here would let an inexact quote through the door this type
        exists to guard. A caller holding a ``float`` states that intent by
        converting it explicitly.
        """
        quoted = value if isinstance(value, Decimal) else Decimal(value)
        scaled = quoted * COST_BPS_PER_SIDE_TENTHS_PER_BP
        if scaled != scaled.to_integral_value():
            raise PortfolioWalkForwardError("portfolio_backtesting.per_side_cost_not_fixed_point")
        return cls.create(cost_bps_per_side_tenths=int(scaled))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_assumption(self) -> Self:
        """Require the installed quote increment and canonical assumption identity.

        Returns:
            This validated per-side assumption.

        Raises:
            PortfolioWalkForwardError: The increment or assumption identity is invalid.
        """
        if self.cost_bps_per_side_tenths % COST_BPS_PER_SIDE_INCREMENT_TENTHS:
            raise PortfolioWalkForwardError("portfolio_backtesting.per_side_cost_increment_invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"assumption_hash"}))
        if self.assumption_hash != expected:
            raise PortfolioWalkForwardError("portfolio_backtesting.per_side_cost_identity_invalid")
        return self

    @property
    def cost_bps_per_side(self) -> Decimal:
        """The quoted rate, exact."""
        return Decimal(self.cost_bps_per_side_tenths) / COST_BPS_PER_SIDE_TENTHS_PER_BP

    @property
    def platform_one_way_cost_bps_tenths(self) -> int:
        """Convert the per-side quote to exact platform one-way fixed-point units.

        Returns:
            The shared owner's doubled rate, still in tenths of a basis point.
        """
        return platform_one_way_cost_from_per_side(self.cost_bps_per_side_tenths)

    @property
    def platform_one_way_cost_bps(self) -> Decimal:
        """Convert the exact per-side quote through the shared one-way rate owner.

        Returns:
            The exact Decimal platform one-way rate, twice the quoted per-side basis points.
        """
        return Decimal(self.platform_one_way_cost_bps_tenths) / COST_BPS_PER_SIDE_TENTHS_PER_BP

    @property
    def platform_cost_bps(self) -> float:
        """The engine's lane as the ``float`` its arithmetic takes.

        Exact for every admitted value, because the frozen increment keeps the
        platform rate a multiple of ``1.0``.
        """
        return self.platform_one_way_cost_bps_tenths / COST_BPS_PER_SIDE_TENTHS_PER_BP

    def researcher_payload(self) -> dict[str, str]:
        """Both rates and both bases, so a report can never show one unlabelled."""
        return {
            "quote_basis": "PER_SIDE",
            "cost_bps_per_side": f"{self.cost_bps_per_side}",
            "platform_one_way_cost_bps": f"{self.platform_one_way_cost_bps}",
            "mapping": "platform_one_way_cost_bps = 2 * cost_bps_per_side",
            "assumption_hash": self.assumption_hash,
        }


@dataclass(frozen=True, slots=True)
class PortfolioCostPolicy:
    """Which linear cost lanes a path is reported on, and which one decides.

    ``ExecutionClockBinding`` has declared ``reporting_cost_bps`` and
    ``selection_cost_bps`` since it was written, sealed both into its own
    ``binding_hash``, and validated that the selection lane is one of the
    reported ones. The evaluator then ignored the declaration and used
    ``(5, 10, 20)`` with lane 10 as module literals -- so an author could state a
    cost axis, watch it hash into the Program, and get a different one measured.

    Defaults reproduce those literals exactly, so an existing caller keeps its
    numbers to the bit.
    """

    reporting_bps: tuple[int, ...] = DEFAULT_REPORTING_COST_BPS
    selection_bps: int = DEFAULT_SELECTION_COST_BPS
    transaction_cost_convention: PortfolioTransactionCostConvention | None = (
        DEFAULT_PORTFOLIO_TRANSACTION_COST_CONVENTION
    )

    def __post_init__(self) -> None:
        """Require distinct reporting lanes, an included selection lane and supported convention.

        Raises:
            PortfolioWalkForwardError: The lanes are empty, negative or duplicated, selection is
                absent, or the convention is unsupported.
        """
        if not self.reporting_bps:
            raise PortfolioWalkForwardError("portfolio_backtesting.cost_policy_empty")
        if any(value < 0 for value in self.reporting_bps):
            raise PortfolioWalkForwardError("portfolio_backtesting.cost_policy_negative")
        if len(set(self.reporting_bps)) != len(self.reporting_bps):
            raise PortfolioWalkForwardError("portfolio_backtesting.cost_policy_duplicated")
        if self.selection_bps not in self.reporting_bps:
            raise PortfolioWalkForwardError("portfolio_backtesting.cost_policy_selection_absent")
        if (
            self.transaction_cost_convention is not None
            and self.transaction_cost_convention.convention_id != "ONE_WAY_ASSET_TURNOVER_BPS"
        ):
            raise PortfolioWalkForwardError("portfolio_backtesting.cost_policy_convention_invalid")

    @property
    def convention_disposition(self) -> str:
        """Identify whether the policy binds the installed convention or legacy arithmetic.

        Returns:
            The bound one-way convention disposition or the explicit legacy ambiguous disposition.
        """
        return (
            "BOUND_ONE_WAY_ASSET_TURNOVER_BPS"
            if self.transaction_cost_convention is not None
            else "LEGACY_AMBIGUOUS_COST_CONVENTION"
        )

    def cost_fraction(self, *, one_way_turnover: FloatArray, cost_bps: float) -> FloatArray:
        """Return the session cost under this policy's sealed convention.

        The legacy branch deliberately keeps historical arithmetic readable, but
        its caller must carry the legacy disposition rather than describing it
        as the installed per-one-way convention.
        """
        turnover = np.asarray(one_way_turnover, dtype=np.float64)
        if self.transaction_cost_convention is not None:
            return self.transaction_cost_convention.cost_fraction(
                one_way_turnover=turnover, cost_bps=cost_bps
            )
        if (
            not np.isfinite(cost_bps)
            or cost_bps < 0.0
            or not np.isfinite(turnover).all()
            or bool(np.any(turnover < 0.0))
        ):
            raise PortfolioWalkForwardError("portfolio_backtesting.transaction_cost_input_invalid")
        return np.asarray(turnover * (cost_bps / 10_000.0), dtype=np.float64)

    def net_simple_returns(
        self,
        *,
        gross_simple_returns: FloatArray,
        one_way_turnovers: FloatArray,
        cost_bps: float,
    ) -> FloatArray:
        """Subtract declared session costs from the gross simple-return path.

        Args:
            gross_simple_returns: Gross realized simple returns on the execution axis.
            one_way_turnovers: Realized asset turnovers with the same shape as the gross path.
            cost_bps: Finite nonnegative one-way turnover rate in basis points.

        Returns:
            The gross path minus the cost fractions produced by this policy.

        Raises:
            PortfolioWalkForwardError: Path shapes differ, or the cost rate/turnovers are invalid.
        """
        gross = np.asarray(gross_simple_returns, dtype=np.float64)
        turnover = np.asarray(one_way_turnovers, dtype=np.float64)
        if gross.shape != turnover.shape:
            raise PortfolioWalkForwardError("portfolio_backtesting.transaction_cost_axis_invalid")
        return np.asarray(gross - self.cost_fraction(one_way_turnover=turnover, cost_bps=cost_bps))

    def researcher_payload(self) -> dict[str, object]:
        """Publish the installed cost meaning or an explicit legacy interpretation limit.

        Returns:
            The descriptive convention payload; legacy arithmetic does not acquire a new convention
            label.
        """
        if self.transaction_cost_convention is None:
            return {
                "disposition": self.convention_disposition,
                "human_readable_label": (
                    "legacy cost arithmetic; rate basis and round-trip interpretation "
                    "were not sealed"
                ),
            }
        return {
            "disposition": self.convention_disposition,
            **self.transaction_cost_convention.researcher_payload(),
        }


class PortfolioTrialMetrics(BaseModel):  # type: ignore[misc]
    """Path metrics embedded unchanged in Strategy Lab and Validation evidence.

    Deliberately not grown. Seven sealed contracts nest this shape, each with its
    own identity function, so a field added here moves seven published
    identities to give six of them a number they never asked for. The declared
    cost axis and the selection-lane Sharpe live on
    ``PortfolioWalkForwardResult`` -- which is a return value, not evidence --
    and are sealed by whichever comparable row actually wants them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cumulative_gross_log_wealth: float
    cumulative_net_log_wealth_5bps: float
    cumulative_net_log_wealth_10bps: float
    cumulative_net_log_wealth_20bps: float
    realized_annualized_volatility: float = Field(ge=0.0)
    maximum_drawdown: float = Field(ge=0.0, le=1.0)
    mean_one_way_turnover: float = Field(ge=0.0)
    mean_hhi: float = Field(ge=0.0, le=1.0)
    mean_holding_count: float = Field(ge=0.0)
    mean_weighted_adv20: float = Field(gt=0.0)
    minimum_weighted_adv20: float = Field(gt=0.0)
    solver_failure_count: int = Field(ge=0)
    missing_execution_count: int = Field(ge=0)
    realized_to_predicted_variance_ratio: float | None = Field(default=None, gt=0.0)
    """``mean(realized**2) / mean(predicted)``. Above one is under-prediction.

    Named for the direction it actually computes. The previous name said
    predicted-over-realized while the arithmetic here said the reverse, and two
    other call sites computed the reciprocal under that same name -- so the
    field, not the formula, was where the confusion lived.
    """

    variance_calibration_support_count: int | None = Field(default=None, ge=0)
    """Formations that carried a decision-time forecast, so two ratios compare."""

    variance_calibration_disposition: (
        Literal["MEASURED", "NOT_APPLICABLE_POLICY_DOES_NOT_CONSUME_RISK_FORECAST"] | None
    ) = None
    """Whether the ratio is measured or deliberately absent for a closed-form policy.

    ``None`` remains the historical-readback state.  A current closed-form
    point cannot publish an invented ratio merely because the common metrics
    owner normally expects one.
    """

    predicted_realized_variance_ratio: float | None = Field(default=None, gt=0.0)
    """Read-only, historical, and **direction-ambiguous**. Never written now.

    Three writers filled this name and two of them stored the reciprocal of the
    third, so a value here cannot be interpreted without knowing which produced
    it. It stays declared because ``extra="forbid"`` would otherwise make every
    historical document carrying it unparseable -- which is what removing it did.
    """

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_one_calibration_direction(self) -> PortfolioTrialMetrics:
        """One artifact may not state the ratio in both directions at once.

        The legacy key and the successor name mean opposite things on at least
        two of the three writers that filled the old one. A document carrying
        both would let a reader pick whichever agreed with their expectation, and
        neither could be checked against the other.
        """
        if (
            self.predicted_realized_variance_ratio is not None
            and self.realized_to_predicted_variance_ratio is not None
        ):
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.variance_ratio_stated_in_two_directions"
            )
        if self.variance_calibration_disposition == (
            "NOT_APPLICABLE_POLICY_DOES_NOT_CONSUME_RISK_FORECAST"
        ) and (
            self.realized_to_predicted_variance_ratio is not None
            or self.variance_calibration_support_count not in (None, 0)
        ):
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.variance_calibration_not_applicable_stated"
            )
        return self


class PortfolioBootstrapSettings(Protocol):
    """Provide the declared resampling budget consumed by path evaluation."""

    @property
    def bootstrap_resamples(self) -> int:
        """Read the number of bootstrap replications declared by the mandate.

        Returns:
            The resampling count used by the metrics owner.
        """
        ...


class PortfolioBacktestWorkspace(Protocol):
    """Provide the aligned data surfaces consumed by causal Portfolio execution.

    Formation-by-listing execution availability, realized simple returns and causal
    ADV share the formation and ordered listing axes. Passive outcome rows are keyed
    by their session labels so embargo carry does not infer a calendar from indices.
    """

    @property
    def mandate(self) -> PortfolioBootstrapSettings:
        """Read the bootstrap settings declared for this path.

        Returns:
            The mandate resampling settings.
        """
        ...

    @property
    def formation_sessions(self) -> tuple[date, ...]:
        """Read ordered decision-session labels for the formation axis.

        Returns:
            The formation session axis.
        """
        ...

    @property
    def ordered_listing_ids(self) -> tuple[str, ...]:
        """Read listing identity order shared by all asset-axis surfaces.

        Returns:
            The ordered listing handles.
        """
        ...

    @property
    def execution_available(self) -> BoolArray:
        """Read which formation/listing orders can fill.

        Returns:
            The formation-by-listing execution-availability mask.
        """
        ...

    @property
    def realized_simple_returns(self) -> FloatArray:
        """Read realized simple returns for the formation outcome windows.

        Returns:
            The formation-by-listing realized return surface.
        """
        ...

    @property
    def causal_adv20(self) -> FloatArray:
        """Read the causal liquidity observations used for held-book capacity summaries.

        Returns:
            The aligned formation-by-listing ADV20 surface.
        """
        ...

    @property
    def sector_exposure_matrix(self) -> FloatArray:
        """Read the sector-by-listing exposures used to summarize each executed book.

        Returns:
            The sector exposure matrix.
        """
        ...

    @property
    def equal_weight_sector_exposure(self) -> FloatArray:
        """Read the equal-weight anchor used for sector deviations.

        Returns:
            The sector anchor, fixed or formation-specific.
        """
        ...

    @property
    def passive_returns_by_session(self) -> dict[date, FloatArray]:
        """Read the session-keyed outcome returns used for no-decision carry.

        Returns:
            Passive return rows aligned to the listing axis.
        """
        ...


@dataclass(frozen=True, slots=True)
class PortfolioTargetDecision:
    """Carry a target or explicit hold with the policy's decision-time audit.

    A HOLD retains the pretrade book and has no predicted variance. Rebalances that
    consume Risk require a forecast; direct closed-form policies must opt out
    explicitly. Missing optimizer objective/audit fields do not prove that no solve
    occurred. Failed execution can make the filled book differ from this target.

    Attributes:
        target_weights: Asset-axis weights returned by the decision provider.
        predicted_variance: Decision-time Risk forecast, absent on a hold or a policy that opts out.
        decision_mode: REBALANCE or HOLD as declared by the execution clock.
        requires_risk_forecast: Whether a rebalance must carry a covariance forecast.
        optimizer_objective_value: Optional service-owned post-solve objective total.
        optimization_audit: Optional owner-supplied optimization audit.
        solver_call_count: Recorded number of solver calls.
        sector_unconstrained_weights: Optional corresponding book before sector constraints.
    """

    target_weights: FloatArray
    predicted_variance: float | None
    """The forecast this decision was taken against, if this policy uses one.

    ``None`` rather than ``0.0`` means either a hold or a direct closed-form
    policy that explicitly does not consume covariance.  A zero is a number: it
    survives a round trip through a durable record and reads as a valid forecast
    to anything that does not also check this explicit declaration.
    """

    decision_mode: Literal["REBALANCE", "HOLD"] = "REBALANCE"
    """Whether a decision was taken at all.

    A hold is not a target decision: no adapter is called, no forecast is made,
    and the weights that come back are the book being carried rather than a
    choice. Recording that as a target made a path artifact look like a decision
    artifact, and put an ex-post quantity into decision-time risk calibration.
    """
    requires_risk_forecast: bool = True
    """Whether a rebalance is required to publish a covariance forecast.

    Existing policies default to ``True`` so a missing forecast stays a
    fail-closed error.  A direct policy must opt out explicitly; this is not an
    inference from a zero solver count.
    """
    optimizer_objective_value: float | None = None
    """The optimizer service's own post-solve total for this decision.

    Not read back from the solver -- the service computes it in Python from the
    terms it also publishes in the audit. The point is that it is computed
    *once, at the owner*: the consumer used to rebuild the same total from those
    terms, which is a second copy that would silently drop a term the objective
    later gained while still agreeing with itself.

    ``None`` on a hold, which solves nothing, and also on any adapter that does
    not carry it -- ``TOP_K_MINIMUM_VARIANCE`` and
    ``RETURN_SCALED_TOTAL_SIGNAL_GLOBAL_QP`` both solve and both leave it absent.
    A consumer must therefore treat absence as "not stated", never as "no solve".
    """

    optimization_audit: object | None = None
    solver_call_count: int = 0
    sector_unconstrained_weights: FloatArray | None = None


class RebalanceClockBinding(BaseModel):  # type: ignore[misc]
    """Which installed cadence a path was run on, sealed into the identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["RebalanceClockBinding"] = "RebalanceClockBinding"
    clock_id: str = Field(min_length=1, max_length=96)
    parameters: tuple[tuple[str, int], ...] = ()
    formation_indices: tuple[int, ...] | None = Field(default=None, exclude_if=lambda v: v is None)
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> RebalanceClockBinding:
        """Self-sealing, so a nested copy of this cannot be edited in place.

        It became worth checking when the binding started travelling inside a
        durable state transition: a clock id swapped there with the hash left
        alone would otherwise send a replay to a different installed cadence and
        get agreement from the hash it was compared against.
        """
        indices = self.formation_indices
        if indices is not None and (
            self.clock_id != "SCORED_FORMATIONS"
            or not indices
            or indices != tuple(sorted(set(indices)))
            or indices[0] != 0
        ):
            raise PortfolioWalkForwardError("portfolio_backtesting.scored_clock_axis_invalid")
        if self.clock_id == "SCORED_FORMATIONS" and indices is None:
            raise PortfolioWalkForwardError("portfolio_backtesting.scored_clock_axis_invalid")
        if self.binding_hash != canonical_hash(
            {
                "clock_id": self.clock_id,
                "parameters": dict(self.parameters),
                **({"formation_indices": indices} if indices is not None else {}),
            }
        ):
            raise PortfolioWalkForwardError(
                "portfolio_backtesting.rebalance_clock_binding_identity_invalid"
            )
        return self


class RebalanceClock(Protocol):
    """Whether one formation inside one segment is a decision or a hold."""

    @property
    def clock_id(self) -> str:
        """Read the installed cadence handle used for replay resolution.

        Returns:
            The named clock identity.
        """
        ...

    def rebalances(self, *, formation_index: int, segment_start_index: int) -> bool:
        """Decide whether this formation is a rebalance under the clock's anchor.

        Args:
            formation_index: Position on the complete formation axis.
            segment_start_index: Start of the current segment for segment-relative clocks.

        Returns:
            True for REBALANCE and false for HOLD.
        """
        ...

    @property
    def binding(self) -> RebalanceClockBinding:
        """Read the sealed clock ID, parameters and optional scored formation axis.

        Returns:
            The typed replay binding for this clock.
        """
        ...


class PortfolioDecisionProvider(Protocol):
    """Produce a target or carry the pretrade book under an explicit decision mode.

    The optimizer reference is valued by the installed reference-mark lane at the
    decision session: the open proxy retains its open weights, while a close-marked
    lane applies that session's causal mark. The pretrade book has already drifted
    through the previous outcome window and is what a HOLD keeps. They are separate
    arrays because replacing the held book with its optimizer reference creates
    turnover on a formation whose cadence declared no rebalance.
    """

    def __call__(
        self,
        *,
        formation_index: int,
        reference_weights: FloatArray,
        pretrade_weights: FloatArray,
        decision_mode: Literal["REBALANCE", "HOLD"],
    ) -> PortfolioTargetDecision:
        """Resolve one conditional target or explicit pretrade-book hold.

        Args:
            formation_index: Current position on the formation axis.
            reference_weights: Book valued by the installed reference lane for optimizer turnover.
            pretrade_weights: Drifted carried book preserved when decision_mode is HOLD.
            decision_mode: Clock-declared REBALANCE or HOLD.

        Returns:
            The target/hold weights, forecast policy and available optimization audit.
        """
        ...


@dataclass(frozen=True, slots=True)
class PortfolioWalkForwardResult:
    """Return evaluated continuous-path metrics and explicit reporting cost lanes.

    Attributes:
        metrics: Sealed-shape trial metrics used by downstream evidence owners.
        gross_log_returns: Realized gross log-return path.
        net_log_returns_5bps: Historical fixed five-basis-point reporting path.
        net_log_returns_10bps: Historical fixed ten-basis-point reporting path.
        net_log_returns_20bps: Historical fixed twenty-basis-point reporting path.
        net_log_returns_by_bps: Declared reporting rates paired with their net paths.
        net_log_wealth_by_bps: Declared rates paired with cumulative net log wealth.
        selection_cost_bps: Reporting lane used for selection summaries.
        sharpe: Selection-lane simple-return Sharpe, absent at zero deviation.
        annual_net_log_returns: Selection-lane log sums grouped by calendar year.
        chronological_block_net_log_returns: Selection-lane chronological block log sums.
        bootstrap_probability_net_positive: Fraction of seeded circular-block sums above zero.
        sector_exposure_summary: Mean and maximum absolute sector-anchor deviations.
    """

    metrics: PortfolioTrialMetrics
    gross_log_returns: tuple[float, ...]
    net_log_returns_5bps: tuple[float, ...]
    net_log_returns_10bps: tuple[float, ...]
    net_log_returns_20bps: tuple[float, ...]
    net_log_returns_by_bps: tuple[tuple[int, tuple[float, ...]], ...]
    net_log_wealth_by_bps: tuple[tuple[int, float], ...]
    selection_cost_bps: int
    sharpe: float | None
    annual_net_log_returns: tuple[tuple[int, float], ...]
    chronological_block_net_log_returns: tuple[float, ...]
    bootstrap_probability_net_positive: float
    sector_exposure_summary: dict[str, float]


@dataclass(frozen=True, slots=True)
class PortfolioWalkForwardState:
    """The book at one instant, and the reference the next decision prices against.

    Two states, not one book and one stray array. ``pretrade_*`` is the drifted
    book at the next formation's decision; ``optimizer_reference*`` is the
    executed book the optimizer measures turnover against, and it lives at its
    own instant -- the entry open it was filled at, or that session's close once
    it is re-marked.

    ``optimizer_reference_cash`` exists because the reference is a *portfolio*,
    not a weight vector. Advancing it with the drifted book's cash mixed two
    instants inside one budget identity, and ``drift_holdings`` refuses that the
    moment the two books hold different amounts of cash -- which is any book that
    is not fully invested.
    """

    pretrade_weights: FloatArray
    pretrade_cash: float
    optimizer_reference: FloatArray
    optimizer_reference_cash: float = 1.0
    """The cash beside ``optimizer_reference``, at the reference's own instant.

    Defaulted for the two fabricated states that never advance: a replay's
    placeholder and a validation stub. Every state the engine produces fills it.
    """


@dataclass(frozen=True, slots=True)
class PortfolioPassiveHoldResult:
    """One economic session carried without a score or target decision."""

    formation_session: date
    state_mode: Literal["NO_SCORE_PASSIVE_HOLD"]
    gross_simple_return: float
    one_way_turnover: Literal[0]
    optimizer_call_count: Literal[0]
    target_decision_count: Literal[0]
    final_state: PortfolioWalkForwardState


@dataclass(frozen=True, slots=True)
class PortfolioWalkForwardSegmentResult:
    """Carry exact execution-path evidence and the state at a segment boundary.

    The interval is start-inclusive and stop-exclusive on the formation axis.
    Executed weights come from fills rather than targets; missing fills and explicit
    hold modes remain visible. Empty optional mode/forecast/holdings lanes preserve
    historical readback without fabricating current decision evidence.

    Attributes:
        start_index: Inclusive formation-axis start.
        stop_index: Exclusive formation-axis stop.
        gross_simple_returns: Realized gross outcomes of executed books.
        one_way_turnovers: Realized half-L1 asset-weight changes at fills.
        predicted_variances: Decision-time forecasts, absent on holds or non-Risk policies.
        hhi: Squared-weight concentration per executed book.
        holding_counts: Number of weights above the execution tolerance per book.
        weighted_adv20: Executed-book causal liquidity summary, or unavailable capacity values.
        maximum_absolute_sector_deviations: Largest sector-anchor deviation per book.
        missed_execution_count: Number of nontrivial desired asset changes without available fills.
        final_state: Drifted book and separately carried optimizer reference.
        executed_weights: Exact post-fill asset weights per formation, empty only on legacy results.
        decision_modes: Explicit REBALANCE/HOLD lane, empty on legacy results.
        risk_forecast_required: Policy-declared forecast requirement per formation.
    """

    start_index: int
    stop_index: int
    gross_simple_returns: tuple[float, ...]
    one_way_turnovers: tuple[float, ...]
    predicted_variances: tuple[float | None, ...]
    """One entry per formation, ``None`` where the formation made no forecast."""

    hhi: tuple[float, ...]
    holding_counts: tuple[float, ...]
    weighted_adv20: tuple[float, ...]
    maximum_absolute_sector_deviations: tuple[float, ...]
    missed_execution_count: int
    final_state: PortfolioWalkForwardState
    executed_weights: tuple[tuple[float, ...], ...] = ()
    """The exact post-fill holdings path for durable public ledger readback.

    Empty remains the historical result shape.  Current segment execution fills
    this from the shared execution owner; report/storage code must not rebuild
    holdings from target decisions because failed fills can make them differ.
    """
    decision_modes: tuple[str, ...] = ()
    """``REBALANCE`` or ``HOLD`` per formation, empty on a legacy result.

    Carried so a consumer can tell a forecast from a carried book. Without it a
    hold's ``predicted_variance`` -- the variance of the drifted book, which only
    exists after the entry open -- is indistinguishable from a real decision-time
    forecast, and it went straight into risk calibration.
    """

    risk_forecast_required: tuple[bool, ...] = ()
    """Whether each formation's policy consumes a decision-time Risk forecast.

    Empty is legacy and conservatively means required. Closed-form policies carry
    an explicit false lane, so an all-absent forecast path is typed
    ``NOT_APPLICABLE`` rather than treated as a failed Risk calibration.
    """


@dataclass(frozen=True, slots=True)
class PortfolioWalkForwardSequenceResult:
    """Decision segments plus the economic sessions carried between them."""

    segments: tuple[PortfolioWalkForwardSegmentResult, ...]
    passive_holds: tuple[PortfolioPassiveHoldResult, ...]


FLAT_BOOK_ALL_CASH: Final = "FLAT_BOOK_ALL_CASH"
EXECUTED_BOOK_AT_OPEN_T_PROXY: Final = "EXECUTED_BOOK_AT_OPEN_T_PROXY"
MARKED_TO_MARKET_AT_CLOSE_T: Final = "MARKED_TO_MARKET_AT_CLOSE_T"

type ReferenceMarkMethod = Literal["EXECUTED_BOOK_AT_OPEN_T_PROXY", "MARKED_TO_MARKET_AT_CLOSE_T"]
"""How the optimizer reference is valued after execution. A closed domain.

Two installed methods and no third. Declared as a type rather than restated as a
``str`` field wherever it travels: a consumption binding that accepted any short
string could record ``"close"``, ``"MARKED_TO_MARKET"`` or a typo and validate,
and every downstream reader would then be comparing free text.
"""


class PortfolioStateTransitionBinding(BaseModel):  # type: ignore[misc]
    """What produced the holdings a Campaign decided and measured on.

    Holdings are **endogenous** -- the output of this strategy's own path, not a
    fact some upstream owner published -- so they carry no strategy-neutral input
    authority. This is what stands in its place, and it rotates on the four
    things that actually decide the book: the transition arithmetic, the
    rebalance clock, the execution method, and how the reference is marked.

    An earlier version of this contract was written and never constructed
    outside a test, and was deleted for it. It returns with a consumer: the
    campaign seals its hash into the consumption binding and replay re-derives it
    at its owners.

    The mark authority is **resolved**, not named. ``reference_mark_method``
    alone would let two surfaces, two price bases or two watermarks share one
    identity, so the surface hash, its epoch, its price basis and its source
    watermark are carried beside it -- absent together only for the open proxy,
    which is what a historical graph carries.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["PortfolioStateTransitionBinding"] = "PortfolioStateTransitionBinding"
    initial_state_semantics: Literal["FLAT_BOOK_ALL_CASH"] = FLAT_BOOK_ALL_CASH
    transition_implementation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_method_binding_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    rebalance_clock: RebalanceClockBinding
    """The cadence, in full, so a replay can rebuild it at the installed catalog.

    A hash alone recorded which clock ran and gave a verifier no way to ask the
    catalog for it: the id and its parameters lived only in the request, which is
    not part of the graph. Nesting the binding means the replay names an id,
    builds that clock with those parameters through the owner's own factory, and
    compares -- rather than comparing a stored hash to itself.
    """

    reference_mark_method: ReferenceMarkMethod
    mark_manifest_ref: str | None = Field(default=None, min_length=1, max_length=256)
    """The store-relative name of the surface, so a replay reopens *that* one.

    A hash alone is a claim a verifier cannot follow. The resolver used to find
    its surface by globbing a directory and requiring exactly one manifest, which
    supports one generation and no replay at all: a workspace holding two
    surfaces became unresolvable, and a workspace holding one different surface
    resolved silently to it. This is a canonical URI derived from the identity,
    not a filesystem path, so it names an artifact rather than a machine.
    """

    mark_surface_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mark_epoch_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mark_price_basis: str | None = Field(default=None, min_length=1, max_length=64)
    mark_source_watermark_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mark_availability_policy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mark_availability_policy_id: str | None = Field(default=None, min_length=1, max_length=96)
    """Which installed availability owner, by name, so a replay can re-resolve it.

    Beside the hash rather than instead of it: the name is what a verifier looks
    the policy up under at the Feature owner, and the hash is what it compares
    the answer against. A hash with no name can only be compared to a policy
    somebody already guessed.
    """

    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def marks_to_close(self) -> bool:
        """Identify whether the declared reference method uses a resolved close mark.

        Returns:
            True for MARKED_TO_MARKET_AT_CLOSE_T.
        """
        return self.reference_mark_method == MARKED_TO_MARKET_AT_CLOSE_T

    @classmethod
    def create(cls, **values: object) -> PortfolioStateTransitionBinding:
        """Seal transition, cadence, execution and resolved mark-authority fields.

        Args:
            values: Typed transition-binding fields; binding_hash is derived by the owner.

        Returns:
            The validated transition binding with a canonical identity.

        Raises:
            PortfolioWalkForwardError: Resolved mark authority or the resulting identity is
                inconsistent.
        """
        draft = cls.model_construct(**values, binding_hash="0" * 64)
        identity = draft.model_dump(mode="json", exclude={"binding_hash"})
        payload = draft.model_dump(exclude={"binding_hash"})
        return cast(
            "PortfolioStateTransitionBinding",
            cls.model_validate({**payload, "binding_hash": canonical_hash(identity)}),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_binding(self) -> PortfolioStateTransitionBinding:
        """Require complete close-mark authority or an authority-free proxy and a matching hash.

        Returns:
            This validated state-transition binding.

        Raises:
            PortfolioWalkForwardError: Close-mark authority is incomplete, a proxy carries
                authority, or the hash differs.
        """
        resolved = (
            self.mark_manifest_ref,
            self.mark_surface_hash,
            self.mark_epoch_hash,
            self.mark_price_basis,
            self.mark_source_watermark_hash,
            self.mark_availability_policy_hash,
            self.mark_availability_policy_id,
        )
        if self.marks_to_close and any(value is None for value in resolved):
            # A method name with no surface behind it is the free-text binding
            # review already rejected once.
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.state_transition_mark_authority_unresolved"
            )
        if not self.marks_to_close and any(value is not None for value in resolved):
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.state_transition_proxy_carries_mark_authority"
            )
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.state_transition_binding_identity_invalid"
            )
        return self
