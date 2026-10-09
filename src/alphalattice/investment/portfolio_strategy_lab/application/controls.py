"""The finite public control surface, described once for every entry point.

Desktop, the Local Application Service, the thin CLI and an installed Agent tool
all render or validate against this catalog. None of them restates a range, a
default or a refusal, because a control described in four places is a control
that will eventually disagree with itself.

The holdings controls are not redefined here either. ``TrancheBookRecipe`` owns
their ranges and defaults and this module reads them, so the policy and the
public schema cannot drift apart. What this module adds is the surface the policy
does not own -- cost, benchmark view, report unit, study window -- plus the one
thing no single owner could state: which artifacts each control invalidates.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    COST_BPS_PER_SIDE_INCREMENT_TENTHS,
    COST_BPS_PER_SIDE_TENTHS_PER_BP,
    MAXIMUM_COST_BPS_PER_SIDE_TENTHS,
)
from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import (
    ADMITTED_WEIGHT_RULES,
    DEFAULT_TOP_K,
    DEFAULT_TRANCHES,
    DEFAULT_WEIGHT_RULE,
    EXIT_RANK_MULTIPLE_RANGE,
    TOP_K_RANGE,
    TRANCHES_RANGE,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class PublicControlError(ValueError):
    """Stable refusal for a control that is not installed or not admitted."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


type PortfolioReportUnit = Literal[
    "MONTHLY_BETA_STRIPPED_LEDGER",
    "SIMPLE_CUMULATIVE",
    "CALENDAR_YEAR_TABLE",
]
ADMITTED_REPORT_UNITS: tuple[PortfolioReportUnit, ...] = (
    "MONTHLY_BETA_STRIPPED_LEDGER",
    "SIMPLE_CUMULATIVE",
    "CALENDAR_YEAR_TABLE",
)
DEFAULT_REPORT_UNIT: PortfolioReportUnit = "MONTHLY_BETA_STRIPPED_LEDGER"

REFUSED_REPORT_UNIT: Literal["ROLLING_EXCESS"] = "ROLLING_EXCESS"
"""Named so the refusal is a value the schema knows, not an unlisted string."""

FULL_SUPPORT_SELECTION: Literal["FULL_SUPPORT"] = "FULL_SUPPORT"
"""How a stored control receipt renders a study bound the request left open.

A receipt records every installed control, so "the caller supplied nothing" needs
a written value. Naming it here rather than spelling it inline keeps the receipt
that writes it and the reader that reverses it on the same word.
"""

type PortfolioBenchmarkView = Literal["anchor_only", "anchor_plus_spy"]
ADMITTED_BENCHMARK_VIEWS: tuple[PortfolioBenchmarkView, ...] = (
    "anchor_only",
    "anchor_plus_spy",
)
DEFAULT_BENCHMARK_VIEW: PortfolioBenchmarkView = "anchor_only"

PRIMARY_BENCHMARK: Literal["ELIGIBLE_UNIVERSE_EQUAL_WEIGHT"] = "ELIGIBLE_UNIVERSE_EQUAL_WEIGHT"
"""Always present. ``anchor_plus_spy`` adds a comparator; it never replaces this."""

DEFAULT_COST_BPS_PER_SIDE: str = "5"

EVIDENCE_COST_LADDER_BPS_PER_SIDE: tuple[str, ...] = ("0", "2.5", "5", "10", "20")
"""The fixed ladder rendered beside whatever cost the researcher selected."""

STUDY_WINDOW_DISPOSITION: Literal["DESCRIPTIVE_SUBWINDOW"] = "DESCRIPTIVE_SUBWINDOW"
"""A custom window describes an existing path. It never selects or promotes one."""

CAPACITY_DISPOSITION: Literal["NOT_MODELED"] = "NOT_MODELED"
"""Daily OHLCV does not authorize a market-impact or capacity claim."""

type ControlInvalidationClass = Literal[
    "EXECUTION_LEDGER",
    "ECONOMIC_OVERLAY",
    "BENCHMARK_DESCENDANT",
    "REPORT_UNIT_DESCENDANT",
    "STUDY_WINDOW_DESCENDANT",
]
"""What a change to a control rebuilds -- and, by omission, what it reuses exactly."""

type ControlDisposition = Literal["SHIP", "SHIP_GUARDED", "SHIP_LABELLED"]


class PublicControlDescriptor(_Contract):
    """One admitted control, described once for every entry point that offers it."""

    kind: Literal["PublicControlDescriptor"] = "PublicControlDescriptor"
    control_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    unit: str | None = None
    value_kind: Literal["INTEGER", "ENUM", "FIXED_POINT_DECIMAL", "DATE_OR_FULL_SUPPORT"]
    options: tuple[str, ...] = ()
    min: Decimal | None = None
    max: Decimal | None = None
    step: Decimal | None = None
    default_display: str = Field(min_length=1)
    default_value: str | None = None
    help: str = Field(min_length=1)
    guard: str | None = None
    refusal: str = Field(min_length=1)
    disposition: ControlDisposition
    invalidation_class: ControlInvalidationClass

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_rendering_contract(self) -> Self:
        """Require valid enum options/defaults and coherent numeric control bounds.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PublicControlError: Enum options repeat/are absent/empty, default is not admitted,
                enum/numeric properties are mixed, bounds are reversed or step is not positive.
        """
        if self.value_kind == "ENUM":
            if not self.options or any(value == "" for value in self.options):
                raise PublicControlError("portfolio_application.control_options_absent")
            if len(set(self.options)) != len(self.options):
                raise PublicControlError("portfolio_application.control_options_duplicated")
            if self.default_value not in self.options:
                raise PublicControlError("portfolio_application.control_default_not_an_option")
            if self.min is not None or self.max is not None or self.step is not None:
                raise PublicControlError("portfolio_application.control_enum_has_numeric_bounds")
        elif self.options:
            raise PublicControlError("portfolio_application.control_non_enum_has_options")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise PublicControlError("portfolio_application.control_bounds_reversed")
        if self.step is not None and self.step <= 0:
            raise PublicControlError("portfolio_application.control_step_not_positive")
        return self


class RefusedControlDescriptor(_Contract):
    """A refusal is a typed product result, not a disabled widget."""

    kind: Literal["RefusedControlDescriptor"] = "RefusedControlDescriptor"
    control_id: str = Field(min_length=1)
    refusal_code: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    """Why the control is refused, in the product's words."""
    evidence_reference: str = Field(min_length=1)
    """What the product serves or the law that decides it, never a plan."""


def _catalog_semantics(
    controls: tuple[PublicControlDescriptor, ...],
    refusals: tuple[RefusedControlDescriptor, ...],
) -> dict[str, object]:
    return {
        "kind": "PublicControlCatalog",
        "controls": [
            value.model_dump(
                mode="json",
                include={
                    "control_id",
                    "value_kind",
                    "options",
                    "min",
                    "max",
                    "step",
                    "default_value",
                    "disposition",
                    "invalidation_class",
                    "guard",
                    "refusal",
                },
            )
            for value in controls
        ],
        "refusals": [
            value.model_dump(mode="json", include={"control_id", "refusal_code"})
            for value in refusals
        ],
    }


class PublicControlCatalog(_Contract):
    """The finite installed control surface, with semantic and display identities.

    Generated from the owning types rather than transcribed: the holdings ranges
    come from ``TrancheBookRecipe`` and the cost bounds from the per-side cost
    assumption. `catalog_hash` binds executable admission semantics and is safe
    to carry into a Program; `presentation_hash` also binds labels and help, so a
    wording correction is tamper-evident without invalidating numerical work.
    An entry point renders this; it does not author it.
    """

    kind: Literal["PublicControlCatalog"] = "PublicControlCatalog"
    controls: tuple[PublicControlDescriptor, ...] = Field(min_length=1)
    refusals: tuple[RefusedControlDescriptor, ...] = Field(min_length=1)
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    presentation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require disjoint unique admitted/refused controls and exact semantic/presentation hashes.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PublicControlError: Control/refusal IDs repeat/overlap, guarded controls lack a guard
                family, or semantic/presentation identities differ.
        """
        ids = tuple(value.control_id for value in self.controls)
        if len(set(ids)) != len(ids):
            raise PublicControlError("portfolio_application.control_catalog_duplicated")
        if any(value.disposition == "SHIP_GUARDED" and not value.guard for value in self.controls):
            raise PublicControlError("portfolio_application.control_guard_family_absent")
        refused = tuple(value.control_id for value in self.refusals)
        if len(set(refused)) != len(refused):
            raise PublicControlError("portfolio_application.refusal_catalog_duplicated")
        if set(refused) & set(ids):
            raise PublicControlError("portfolio_application.control_both_admitted_and_refused")
        if self.catalog_hash != canonical_hash(_catalog_semantics(self.controls, self.refusals)):
            raise PublicControlError("portfolio_application.control_catalog_identity_invalid")
        presentation = self.model_dump(mode="json", exclude={"catalog_hash", "presentation_hash"})
        if self.presentation_hash != canonical_hash(presentation):
            raise PublicControlError("portfolio_application.control_presentation_identity_invalid")
        return self

    def descriptor(self, control_id: str) -> PublicControlDescriptor:
        """Resolve an explicitly installed admitted public control.

        Args:
            control_id: Exact admitted control identity.

        Returns:
            Matching control descriptor.

        Raises:
            PublicControlError: The admitted control identity is not installed.
        """
        for value in self.controls:
            if value.control_id == control_id:
                return value
        raise PublicControlError(f"portfolio_application.control_not_installed:{control_id}")

    def rebuilt_by(self, control_id: str) -> ControlInvalidationClass:
        """What changing this control rebuilds. Everything else is reused exactly."""
        return self.descriptor(control_id).invalidation_class

    def guard_family(self, control_id: str) -> str | None:
        """Read the installed public control guard family when one is declared.

        Args:
            control_id: Exact admitted control identity.

        Returns:
            Guard family string or None.

        Raises:
            PublicControlError: The admitted control identity is not installed.
        """
        return self.descriptor(control_id).guard

    def refusal(self, control_id: str) -> RefusedControlDescriptor:
        """Resolve an explicitly installed refused public control.

        Args:
            control_id: Exact refused control identity.

        Returns:
            Matching stable refusal descriptor.

        Raises:
            PublicControlError: The refused control identity is not installed.
        """
        for value in self.refusals:
            if value.control_id == control_id:
                return value
        raise PublicControlError(f"portfolio_application.refusal_not_installed:{control_id}")


def _decimal_display(tenths: int) -> str:
    return f"{Decimal(tenths) / COST_BPS_PER_SIDE_TENTHS_PER_BP}"


_REFUSED_CONTROL_ROWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "schedule_phase",
        "SCHEDULE_PHASE_SELECTION_REFUSED",
        "The book forms on its strategy's frozen schedule; its phase is part of the strategy.",
        "the strategy package's frozen controls (`strategy-book controls`)",
    ),
    (
        "research_effort",
        "SCIENTIFIC_BREADTH_SELECTION_REFUSED",
        "A study's breadth is fixed in its plan before it runs; a book never widens it.",
        "a study's plan (`study plan`)",
    ),
    (
        "alpha_recipe",
        "ALPHA_RECIPE_SELECTION_REFUSED",
        "The Alpha recipe is the strategy's; another recipe is another Alpha study.",
        "an Alpha study (`study controls --kind alpha.model-development`)",
    ),
    (
        "alpha_training_window",
        "ALPHA_TRAINING_WINDOW_SELECTION_REFUSED",
        "Training windows belong to the Alpha study whose models the book reads.",
        "an Alpha study (`study controls --kind alpha.model-development`)",
    ),
    (
        "alpha_seed_or_vintage",
        "ALPHA_SEED_OR_VINTAGE_SELECTION_REFUSED",
        "Seeds and vintages are sealed with the Alpha study's models.",
        "an Alpha study (`study controls --kind alpha.model-development`)",
    ),
    (
        "alpha_cadence_or_model_search",
        "ALPHA_MODEL_SEARCH_REFUSED",
        "A model search is an Alpha study of its own, never a book control.",
        "an Alpha study (`study controls --kind alpha.model-development`)",
    ),
    (
        "feature_or_target",
        "FEATURE_OR_TARGET_SELECTION_REFUSED",
        "Features and targets are chosen in the Factor and Alpha studies the book reads.",
        "a Factor study (`study controls`)",
    ),
    (
        "risk_estimator",
        "RISK_ESTIMATOR_SELECTION_REFUSED",
        "The Risk estimator is the Risk study's; the book reads the Risk surface it published.",
        "a Risk study (`study controls --kind risk.covariance-development`)",
    ),
    (
        "covariance_selector",
        "DENSE_COVARIANCE_SELECTION_REFUSED",
        "A covariance is a Risk study's choice, not the book's.",
        "a Risk study (`study controls --kind risk.covariance-development`)",
    ),
    (
        "sector_forecast",
        "SECTOR_FORECAST_CONTROL_REFUSED",
        "Sector forecasts are Sector research's; the book offers only its zero forecast.",
        "the strategy package's frozen controls (`strategy-book controls`)",
    ),
    (
        "optimizer_or_solver",
        "SOLVER_OR_OPTIMIZER_SELECTION_REFUSED",
        "The installed books weigh their tranches by fixed rules; no optimizer is installed.",
        "the strategy package's frozen controls (`strategy-book controls`)",
    ),
    (
        "qp_or_mvo_policy",
        "QP_OR_MINIMUM_VARIANCE_POLICY_REFUSED",
        "No quadratic or minimum-variance policy is installed.",
        "the strategy package's frozen controls (`strategy-book controls`)",
    ),
    (
        "regime_switching",
        "REGIME_SWITCHING_REFUSED",
        "No regime-switching policy is installed.",
        "the strategy package's frozen controls (`strategy-book controls`)",
    ),
    (
        "volatility_targeting",
        "VOLATILITY_TARGETING_REFUSED",
        "No volatility-targeting policy is installed.",
        "the strategy package's frozen controls (`strategy-book controls`)",
    ),
    (
        "satellite_or_factor_blend",
        "SATELLITE_OR_FACTOR_BLENDING_REFUSED",
        "A blend of satellites or factors would be another strategy, installed on its own.",
        "the strategy package's frozen controls (`strategy-book controls`)",
    ),
    (
        "rolling_excess_report_unit",
        "ROLLING_EXCESS_VIEW_REFUSED",
        "Rolling excess is not a report unit the book computes; its report units are listed.",
        "the book's report units (`strategy-book controls`)",
    ),
    (
        "shorting_or_leverage",
        "SHORTING_OR_LEVERAGE_REFUSED",
        "The books hold long positions only; shorting and leverage are outside them.",
        "the strategy package's frozen controls (`strategy-book controls`)",
    ),
    (
        "broker_or_order",
        "BROKER_OR_ORDER_SUBMISSION_REFUSED",
        "The product researches; it places no order and reaches no broker.",
        "LAWS OP5 (offline by default)",
    ),
    (
        "artifact_path",
        "ARTIFACT_PATH_INPUT_REFUSED",
        "An input is named by its owner's identity, never by a file path.",
        "LAWS OP8 (identity comes only from what an author may write)",
    ),
    (
        "current_pointer",
        "MUTABLE_CURRENT_POINTER_INPUT_REFUSED",
        "A book binds the input version it reads, never a pointer that moves.",
        "LAWS OP8 (identity comes only from what an author may write)",
    ),
    (
        "identity_hash_authority",
        "IDENTITY_HASH_AS_AUTHORITY_REFUSED",
        "A hash is what the product computes, never what a request asserts.",
        "LAWS OP8 (identity comes only from what an author may write)",
    ),
    (
        "protected_window_handle",
        "PROTECTED_WINDOW_HANDLE_REFUSED",
        "The protected holdout is sealed; no request opens it.",
        "a study's sealed holdout (`study show`)",
    ),
    (
        "cold_start_study_window",
        "COLD_START_VIA_DATE_PICKER_REFUSED",
        "A study window lies inside the full common support; a date does not start a book cold.",
        "the book's available interval (`strategy-book preview`)",
    ),
)
"""Each refused control: its refusal code, why in the product's words, and what the product
serves or the law that decides it, never a plan."""

REFUSED_CONTROL_IDS: frozenset[str] = frozenset(row[0] for row in _REFUSED_CONTROL_ROWS)


def build_public_control_catalog() -> PublicControlCatalog:
    """Derive the catalog from its owners, so it cannot describe a stale surface."""
    controls = (
        PublicControlDescriptor(
            control_id="top_k",
            label="Names selected per sleeve",
            unit="names per sleeve",
            value_kind="INTEGER",
            min=Decimal(TOP_K_RANGE[0]),
            max=Decimal(TOP_K_RANGE[1]),
            step=Decimal(1),
            default_display=str(DEFAULT_TOP_K),
            default_value=str(DEFAULT_TOP_K),
            help="How many highest-ranked names each sleeve selects before the exit buffer.",
            refusal="TOP_K_OUTSIDE_ADMITTED_RANGE",
            disposition="SHIP",
            invalidation_class="EXECUTION_LEDGER",
        ),
        PublicControlDescriptor(
            control_id="tranches",
            label="Sleeves",
            unit="sleeves",
            value_kind="INTEGER",
            min=Decimal(TRANCHES_RANGE[0]),
            max=Decimal(TRANCHES_RANGE[1]),
            step=Decimal(1),
            default_display=str(DEFAULT_TRANCHES),
            default_value=str(DEFAULT_TRANCHES),
            help="Formation sleeves in the installed staggered schedule.",
            guard="TRANCHE_SLEEVE_SCHEDULE_FAMILY",
            refusal="TRANCHE_COUNT_OUTSIDE_ADMITTED_RANGE",
            disposition="SHIP_GUARDED",
            invalidation_class="EXECUTION_LEDGER",
        ),
        PublicControlDescriptor(
            control_id="weight_rule",
            label="Weight rule",
            unit=None,
            value_kind="ENUM",
            options=ADMITTED_WEIGHT_RULES,
            default_display=DEFAULT_WEIGHT_RULE,
            default_value=DEFAULT_WEIGHT_RULE,
            help="The installed rule used to size selected names inside each sleeve.",
            refusal="WEIGHT_RULE_NOT_ADMITTED",
            disposition="SHIP",
            invalidation_class="EXECUTION_LEDGER",
        ),
        PublicControlDescriptor(
            control_id="exit_rank",
            label="Exit rank",
            unit="rank",
            value_kind="INTEGER",
            step=Decimal(1),
            default_display="2 * top_k",
            default_value=None,
            help="Buffered exit rank; its exact range is resolved from top_k and eligible names.",
            guard=(
                "top_k <= exit_rank <= min("
                f"{int(EXIT_RANK_MULTIPLE_RANGE[1])} * top_k, eligible_count)"
            ),
            refusal="portfolio_application.exit_rank_outside_band",
            disposition="SHIP_LABELLED",
            invalidation_class="EXECUTION_LEDGER",
        ),
        PublicControlDescriptor(
            control_id="cost_bps_per_side",
            label="Trading cost",
            unit="bps per side",
            value_kind="FIXED_POINT_DECIMAL",
            min=Decimal(0),
            max=Decimal(_decimal_display(MAXIMUM_COST_BPS_PER_SIDE_TENTHS)),
            step=Decimal(_decimal_display(COST_BPS_PER_SIDE_INCREMENT_TENTHS)),
            default_display=DEFAULT_COST_BPS_PER_SIDE,
            default_value=DEFAULT_COST_BPS_PER_SIDE,
            help="One-way transaction-cost assumption charged against one-way turnover.",
            refusal="portfolio_backtesting.per_side_cost_refusal_family",
            disposition="SHIP",
            invalidation_class="ECONOMIC_OVERLAY",
        ),
        PublicControlDescriptor(
            control_id="secondary_benchmark_view",
            label="Secondary benchmark view",
            unit=None,
            value_kind="ENUM",
            options=ADMITTED_BENCHMARK_VIEWS,
            default_display=DEFAULT_BENCHMARK_VIEW,
            default_value=DEFAULT_BENCHMARK_VIEW,
            help="Adds a descriptive comparator without replacing the primary benchmark.",
            refusal="portfolio_application.benchmark_view_not_installed",
            disposition="SHIP",
            invalidation_class="BENCHMARK_DESCENDANT",
        ),
        PublicControlDescriptor(
            control_id="report_unit",
            label="Report unit",
            unit=None,
            value_kind="ENUM",
            options=ADMITTED_REPORT_UNITS,
            default_display=DEFAULT_REPORT_UNIT,
            default_value=DEFAULT_REPORT_UNIT,
            help="Changes only the declared-path report projection.",
            refusal="portfolio_application.report_unit_not_installed",
            disposition="SHIP",
            invalidation_class="REPORT_UNIT_DESCENDANT",
        ),
        PublicControlDescriptor(
            control_id="study_start",
            label="Study start",
            unit="session",
            value_kind="DATE_OR_FULL_SUPPORT",
            default_display="full common support",
            default_value=None,
            help="Optional descriptive-window start inside the exact common support.",
            guard="STUDY_WINDOW_SUPPORT_FAMILY",
            refusal="portfolio_application.study_window_refusal_family",
            disposition="SHIP_GUARDED",
            invalidation_class="STUDY_WINDOW_DESCENDANT",
        ),
        PublicControlDescriptor(
            control_id="study_end",
            label="Study end",
            unit="session",
            value_kind="DATE_OR_FULL_SUPPORT",
            default_display="full common support",
            default_value=None,
            help="Optional descriptive-window end inside the exact common support.",
            guard="STUDY_WINDOW_SUPPORT_FAMILY",
            refusal="portfolio_application.study_window_refusal_family",
            disposition="SHIP_GUARDED",
            invalidation_class="STUDY_WINDOW_DESCENDANT",
        ),
    )
    refusals = tuple(
        RefusedControlDescriptor(
            control_id=control_id,
            refusal_code=code,
            reason=reason,
            evidence_reference=evidence,
        )
        for control_id, code, reason, evidence in _REFUSED_CONTROL_ROWS
    )
    presentation = PublicControlCatalog.model_construct(
        controls=controls,
        refusals=refusals,
        catalog_hash="0" * 64,
        presentation_hash="0" * 64,
    ).model_dump(mode="json", exclude={"catalog_hash", "presentation_hash"})
    return PublicControlCatalog(
        **presentation,
        catalog_hash=canonical_hash(_catalog_semantics(controls, refusals)),
        presentation_hash=canonical_hash(presentation),
    )


INSTALLED_PUBLIC_CONTROL_CATALOG: PublicControlCatalog = build_public_control_catalog()
"""The one catalog every entry point resolves. Built once, immutable, hashed."""


__all__ = [
    "ADMITTED_BENCHMARK_VIEWS",
    "ADMITTED_REPORT_UNITS",
    "CAPACITY_DISPOSITION",
    "DEFAULT_BENCHMARK_VIEW",
    "DEFAULT_COST_BPS_PER_SIDE",
    "DEFAULT_REPORT_UNIT",
    "EVIDENCE_COST_LADDER_BPS_PER_SIDE",
    "FULL_SUPPORT_SELECTION",
    "INSTALLED_PUBLIC_CONTROL_CATALOG",
    "PRIMARY_BENCHMARK",
    "REFUSED_CONTROL_IDS",
    "REFUSED_REPORT_UNIT",
    "STUDY_WINDOW_DISPOSITION",
    "ControlDisposition",
    "ControlInvalidationClass",
    "PortfolioBenchmarkView",
    "PortfolioReportUnit",
    "PublicControlCatalog",
    "PublicControlDescriptor",
    "PublicControlError",
    "RefusedControlDescriptor",
    "build_public_control_catalog",
]
