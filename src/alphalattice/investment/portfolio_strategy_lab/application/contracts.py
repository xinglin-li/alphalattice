"""The public Portfolio request: one control surface, one identity, one refusal set.

Every entry point -- Desktop, the Local Application Service, the thin CLI and an
installed Agent tool -- compiles through the types here. None of them restates a
range, a default, a cross-field rule or a refusal, because a control that is
described in four places is a control that will eventually disagree with itself.

The holdings controls are not redefined here either: ``TrancheBookRecipe`` owns
them, and this module reads its ranges and defaults so the policy and the public
schema cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Final, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioPerSideCostAssumption,
)
from alphalattice.investment.portfolio_strategy_lab.application.advancement import (
    PortfolioLedgerCoverage,
    ordered_axis_intersection,
    ordered_session_axis_hash,
)
from alphalattice.investment.portfolio_strategy_lab.application.controls import (
    ADMITTED_BENCHMARK_VIEWS,
    ADMITTED_REPORT_UNITS,
    CAPACITY_DISPOSITION,
    DEFAULT_BENCHMARK_VIEW,
    DEFAULT_COST_BPS_PER_SIDE,
    DEFAULT_REPORT_UNIT,
    EVIDENCE_COST_LADDER_BPS_PER_SIDE,
    FULL_SUPPORT_SELECTION,
    INSTALLED_PUBLIC_CONTROL_CATALOG,
    PRIMARY_BENCHMARK,
    REFUSED_REPORT_UNIT,
    STUDY_WINDOW_DISPOSITION,
    PortfolioBenchmarkView,
    PortfolioReportUnit,
)
from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import (
    ADMITTED_WEIGHT_RULES,
    DEFAULT_TOP_K,
    DEFAULT_TRANCHES,
    DEFAULT_WEIGHT_RULE,
    EXIT_RANK_MULTIPLE_RANGE,
    SLEEVE_SHARE_POLICY,
    TOP_K_RANGE,
    TRANCHES_RANGE,
    TrancheWeightRule,
    due_sleeves,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.risk import (
    PortfolioRiskAttributionFacts,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type ScoreSourceMode = Literal["HISTORICAL_ARRAY_REPLAY", "CURRENT_MODEL_SCORING"]
"""How the selected frozen package supplies formation scores."""

type PortfolioWeightRule = TrancheWeightRule | Literal["mu.iv0"]
"""A request value, including rules available only from frozen packages."""

REQUEST_WEIGHT_RULES: Final[tuple[PortfolioWeightRule, ...]] = (
    *ADMITTED_WEIGHT_RULES,
    "mu.iv0",
)


WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID: Final = "WORKSPACE_DEFAULT"
"""Typed request for the manifest's default package, never a package identity."""


class PortfolioApplicationError(ValueError):
    """Stable refusal for public application contract or lineage failures."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class PortfolioResearchSpec(_Contract):
    """One public request. Every entry point compiles through exactly this type.

    The four holdings controls, the cost, the benchmark view, the report unit and
    the study window are all here, and each one knows which artifacts it
    invalidates through ``INSTALLED_PUBLIC_CONTROL_CATALOG``. The projections
    below are what make that real: ``holdings_identity`` is what keys an
    execution ledger, so a cost or window change provably cannot rotate it.

    ``study_start`` and ``study_end`` are ``None`` for the full common-support
    window rather than being pre-filled with the resolved dates. A spec that
    hard-coded today's support would silently become a different request when the
    support advanced, and the default has to mean "all of it", not "all of it as
    of the moment I was built".
    """

    kind: Literal["PortfolioResearchSpec"] = "PortfolioResearchSpec"
    execution_mode: Literal["DEVELOPMENT_REPLAY"] = "DEVELOPMENT_REPLAY"
    strategy_package_id: str = Field(
        default=WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID, min_length=1, max_length=160
    )
    score_source_mode: ScoreSourceMode = "HISTORICAL_ARRAY_REPLAY"
    top_k: int = Field(default=DEFAULT_TOP_K, ge=TOP_K_RANGE[0], le=TOP_K_RANGE[1])
    tranches: int = Field(default=DEFAULT_TRANCHES, ge=TRANCHES_RANGE[0], le=TRANCHES_RANGE[1])
    exit_rank: int = Field(default=2 * DEFAULT_TOP_K, ge=TOP_K_RANGE[0])
    weight_rule: PortfolioWeightRule = DEFAULT_WEIGHT_RULE
    sleeve_share_policy: Literal["EQUAL_NOTIONAL_AT_EACH_FORMATION"] = SLEEVE_SHARE_POLICY
    aggregate_name_cap: float = Field(default=0.06, ge=0.0, le=1.0)
    cost: PortfolioPerSideCostAssumption
    benchmark: Literal["ELIGIBLE_UNIVERSE_EQUAL_WEIGHT"] = PRIMARY_BENCHMARK
    secondary_benchmark_view: PortfolioBenchmarkView = DEFAULT_BENCHMARK_VIEW
    report_unit: PortfolioReportUnit = DEFAULT_REPORT_UNIT
    study_start: date | None = None
    study_end: date | None = None
    study_window_disposition: Literal["DESCRIPTIVE_SUBWINDOW"] = STUDY_WINDOW_DISPOSITION
    sector_forecast_disposition: Literal["SECTOR_FORECAST_NOT_CONSUMED"] = (
        "SECTOR_FORECAST_NOT_CONSUMED"
    )
    spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        strategy_package_id: str = WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID,
        score_source_mode: ScoreSourceMode = "HISTORICAL_ARRAY_REPLAY",
        top_k: int = DEFAULT_TOP_K,
        tranches: int = DEFAULT_TRANCHES,
        exit_rank: int | None = None,
        weight_rule: PortfolioWeightRule = DEFAULT_WEIGHT_RULE,
        cost_bps_per_side: str = DEFAULT_COST_BPS_PER_SIDE,
        secondary_benchmark_view: str = DEFAULT_BENCHMARK_VIEW,
        report_unit: str = DEFAULT_REPORT_UNIT,
        study_start: date | None = None,
        study_end: date | None = None,
    ) -> Self:
        """The one constructor. A caller supplies values, never an identity.

        The two enum parameters are typed ``str`` rather than their literals on
        purpose. A caller reaching this from JSON, a CLI flag or an Agent tool
        argument has a string in hand, and refusing it here with a stable code is
        the difference between a product refusal and a schema traceback. The
        literals still constrain the field, so nothing unadmitted survives.
        """
        if not TOP_K_RANGE[0] <= top_k <= TOP_K_RANGE[1]:
            raise PortfolioApplicationError("TOP_K_OUTSIDE_ADMITTED_RANGE")
        if not TRANCHES_RANGE[0] <= tranches <= TRANCHES_RANGE[1]:
            raise PortfolioApplicationError("TRANCHE_COUNT_OUTSIDE_ADMITTED_RANGE")
        if weight_rule not in REQUEST_WEIGHT_RULES:
            raise PortfolioApplicationError("WEIGHT_RULE_NOT_ADMITTED")
        if report_unit == REFUSED_REPORT_UNIT:
            raise PortfolioApplicationError("portfolio_application.rolling_excess_view_refused")
        if report_unit not in ADMITTED_REPORT_UNITS:
            raise PortfolioApplicationError(
                f"portfolio_application.report_unit_not_installed:{report_unit}"
            )
        if secondary_benchmark_view not in ADMITTED_BENCHMARK_VIEWS:
            raise PortfolioApplicationError(
                f"portfolio_application.benchmark_view_not_installed:{secondary_benchmark_view}"
            )
        values: dict[str, object] = {
            "kind": "PortfolioResearchSpec",
            "execution_mode": "DEVELOPMENT_REPLAY",
            "strategy_package_id": strategy_package_id,
            "score_source_mode": score_source_mode,
            "top_k": top_k,
            "tranches": tranches,
            "exit_rank": 2 * top_k if exit_rank is None else exit_rank,
            "weight_rule": weight_rule,
            "sleeve_share_policy": SLEEVE_SHARE_POLICY,
            "aggregate_name_cap": 0.06,
            "cost": PortfolioPerSideCostAssumption.from_bps_per_side(cost_bps_per_side),
            "benchmark": PRIMARY_BENCHMARK,
            "secondary_benchmark_view": secondary_benchmark_view,
            "report_unit": report_unit,
            "study_start": study_start,
            "study_end": study_end,
            "study_window_disposition": STUDY_WINDOW_DISPOSITION,
            "sector_forecast_disposition": "SECTOR_FORECAST_NOT_CONSUMED",
        }
        identity = cls.model_construct(**values, spec_hash="0" * 64).model_dump(
            mode="json", exclude={"spec_hash"}
        )
        return cls(**identity, spec_hash=canonical_hash(identity))

    @classmethod
    def default(cls) -> Self:
        """The frozen product strategy, unchanged from Gate 8B."""
        return cls.create()

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require admitted aggregate cap, top-k exit band and ordered study window.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: Aggregate cap, exit rank or study bounds are not admitted, or
                canonical spec_hash differs.
        """
        if self.aggregate_name_cap != 0.06:
            raise PortfolioApplicationError("portfolio_application.aggregate_cap_not_admitted")
        # `exit_rank` is a cross-field rule, not a range: the admitted band moves
        # with `top_k`. Its `eligible_count` ceiling is resolution-time and is
        # applied by the Host, which is the only owner that knows the universe.
        floor, ceiling = self.exit_rank_band()
        if not floor <= self.exit_rank <= ceiling:
            raise PortfolioApplicationError("portfolio_application.exit_rank_outside_band")
        if (
            self.study_start is not None
            and self.study_end is not None
            and self.study_start > self.study_end
        ):
            raise PortfolioApplicationError("portfolio_application.study_window_axis_invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"spec_hash"}))
        if self.spec_hash != expected:
            raise PortfolioApplicationError("portfolio_application.spec_identity_invalid")
        return self

    def exit_rank_band(self) -> tuple[int, int]:
        """The spec-side band. The Host narrows the ceiling by `eligible_count`."""
        return self.top_k, int(EXIT_RANK_MULTIPLE_RANGE[1]) * self.top_k

    def holdings_identity(self) -> dict[str, object]:
        """Exactly the controls that can change a fill, and nothing else.

        This is what keys an execution ledger. Cost, benchmark view, report unit
        and study window are deliberately absent: they are descendants, and
        including them here is precisely the defect that would make a cost change
        re-run a walk-forward.
        """
        return {
            "execution_mode": self.execution_mode,
            "strategy_package_id": self.strategy_package_id,
            "score_source_mode": self.score_source_mode,
            "top_k": self.top_k,
            "tranches": self.tranches,
            "exit_rank": self.exit_rank,
            "weight_rule": self.weight_rule,
            "sleeve_share_policy": self.sleeve_share_policy,
            "aggregate_name_cap": self.aggregate_name_cap,
            "sector_forecast_disposition": self.sector_forecast_disposition,
        }

    @property
    def holdings_spec_hash(self) -> str:
        """Hash the declared holdings identity separately from reporting/economic views.

        Returns:
            Canonical hash of holdings_identity().
        """
        return str(canonical_hash(self.holdings_identity()))

    def is_default(self) -> bool:
        """Whether this is the frozen product strategy rather than an exploration."""
        return self.holdings_spec_hash == _DEFAULT_HOLDINGS_HASH and (
            self.cost.cost_bps_per_side_tenths == 50
            and self.secondary_benchmark_view == DEFAULT_BENCHMARK_VIEW
            and self.report_unit == DEFAULT_REPORT_UNIT
            and self.study_start is None
            and self.study_end is None
        )


_DEFAULT_HOLDINGS_HASH: str = str(
    canonical_hash(
        {
            "execution_mode": "DEVELOPMENT_REPLAY",
            "strategy_package_id": WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID,
            "score_source_mode": "HISTORICAL_ARRAY_REPLAY",
            "top_k": DEFAULT_TOP_K,
            "tranches": DEFAULT_TRANCHES,
            "exit_rank": 2 * DEFAULT_TOP_K,
            "weight_rule": DEFAULT_WEIGHT_RULE,
            "sleeve_share_policy": SLEEVE_SHARE_POLICY,
            "aggregate_name_cap": 0.06,
            "sector_forecast_disposition": "SECTOR_FORECAST_NOT_CONSUMED",
        }
    )
)
"""The frozen product strategy's holdings identity, stated once."""


class PortfolioControlReceipt(_Contract):
    """Every installed control and the value this request selected, stored.

    A comparison must be able to say which controls differ between two published
    results without holding either spec. Diffing report fields cannot do that --
    most controls do not appear on a report -- so the selected values are written
    down beside the result, once, in the catalog's own order.
    """

    kind: Literal["PortfolioControlReceipt"] = "PortfolioControlReceipt"
    control_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_control_ids: tuple[str, ...] = ()
    """The catalog order this historical receipt was created against.

    Stored rather than looked up from the process's installed catalog: adding a
    control tomorrow must not make yesterday's intact report unreadable.
    """

    selected: tuple[tuple[str, str], ...] = Field(min_length=1)
    """`(control_id, rendered value)` for every installed control, in catalog order."""

    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def of(cls, spec: PortfolioResearchSpec) -> Self:
        """Seal selected public controls in the installed semantic catalog order.

        Args:
            spec: Validated research spec supplying each installed control value.

        Returns:
            Typed receipt retaining control catalog identity, complete control axis and selected
            string values; absent study bounds select full support.
        """
        catalog = INSTALLED_PUBLIC_CONTROL_CATALOG
        control_ids = tuple(value.control_id for value in catalog.controls)
        values: dict[str, str] = {
            "top_k": str(spec.top_k),
            "tranches": str(spec.tranches),
            "weight_rule": spec.weight_rule,
            "exit_rank": str(spec.exit_rank),
            "cost_bps_per_side": str(spec.cost.cost_bps_per_side),
            "secondary_benchmark_view": spec.secondary_benchmark_view,
            "report_unit": spec.report_unit,
            "study_start": FULL_SUPPORT_SELECTION
            if spec.study_start is None
            else spec.study_start.isoformat(),
            "study_end": (
                FULL_SUPPORT_SELECTION if spec.study_end is None else spec.study_end.isoformat()
            ),
        }
        selected = tuple((value.control_id, values[value.control_id]) for value in catalog.controls)
        identity = cls.model_construct(
            control_catalog_hash=catalog.catalog_hash,
            catalog_control_ids=control_ids,
            selected=selected,
            receipt_hash="0" * 64,
        ).model_dump(mode="json", exclude={"receipt_hash"})
        return cls(**identity, receipt_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require complete unique selected controls and exact historical/current identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: Control identifiers repeat, selected order differs from bound
                order, or the applicable canonical receipt hash differs.
        """
        selected_ids = tuple(control_id for control_id, _ in self.selected)
        bound_ids = self.catalog_control_ids or selected_ids
        if len(set(bound_ids)) != len(bound_ids):
            raise PortfolioApplicationError("portfolio_application.control_receipt_duplicated")
        if selected_ids != bound_ids:
            # A receipt that omits a control cannot report it as unchanged.
            raise PortfolioApplicationError("portfolio_application.control_receipt_incomplete")
        identity = self.model_dump(mode="json", exclude={"receipt_hash"})
        if not self.catalog_control_ids:
            # Exact readback for receipts published before the catalog order was
            # made intrinsic. Their selected order is the historical order; no
            # current catalog is consulted.
            identity.pop("catalog_control_ids")
        if self.receipt_hash != canonical_hash(identity):
            raise PortfolioApplicationError(
                "portfolio_application.control_receipt_identity_invalid"
            )
        return self

    def changed_against(self, other: PortfolioControlReceipt) -> tuple[str, ...]:
        """Which installed controls differ. Both receipts name the same catalog."""
        left_ids = self.catalog_control_ids or tuple(value[0] for value in self.selected)
        right_ids = other.catalog_control_ids or tuple(value[0] for value in other.selected)
        if self.control_catalog_hash != other.control_catalog_hash or left_ids != right_ids:
            raise PortfolioApplicationError(
                "portfolio_application.control_receipt_catalog_mismatch"
            )
        right = dict(other.selected)
        return tuple(
            control_id for control_id, value in self.selected if right[control_id] != value
        )


PROGRAM_LEGACY_OPTIONAL_FIELDS: tuple[str, ...] = (
    "authorities_hash",
    "strategy_package_hash",
    "score_source_mode",
    "continued_from_state_hash",
    "numerical_input_assembly_hash",
)
"""Program bindings that a predecessor artifact may legitimately not carry.

Module-level rather than a class attribute: a leading-underscore tuple on a
pydantic model becomes a private-attribute descriptor and the loop over it would
silently stop looping.
"""

ASSEMBLED_NUMERICAL_INPUT_FIELDS: tuple[str, ...] = (
    "authorities_hash",
    "strategy_package_hash",
    "score_source_mode",
    "policy_recipe_hash",
    "policy_catalog_hash",
    "policy_adapter_binding_hash",
    "alpha_recipe_hash",
    "alpha_evidence_manifest_hash",
    "risk_recipe_hash",
    "risk_return_surface_hash",
    "sector_map_hash",
    "tradability_decision_hash",
    "execution_outcome_manifest_hash",
    "public_path_source_hash",
    "formation_sessions_hash",
    "ordered_listing_ids_hash",
    "continued_from_state_hash",
)
"""Exactly the bindings that decide a number, and nothing descriptive.

`continued_from_state_hash` belongs here for the same reason it belongs on the
Program: an opening book is an input to every fill that follows it.
"""


def numerical_input_assembly_hash(identity: Mapping[str, object]) -> str | None:
    """One digest over every bound numerical input, or `None` for a legacy shape.

    Takes the identity mapping rather than a Program so a caller can seal the
    assembly *while* building the Program, and so the validator can recompute it
    from the same mapping it verifies the Program hash against.
    """
    if identity.get("authorities_hash") is None:
        return None
    assembled: dict[str, object] = {"kind": "PortfolioNumericalInputAssembly"}
    for name in ASSEMBLED_NUMERICAL_INPUT_FIELDS:
        value = identity.get(name)
        if value is not None:
            # Absent optional inputs are omitted rather than hashed as null, so
            # a flat-start assembly keeps the shape it had before continuations
            # existed.
            assembled[name] = value
    return str(canonical_hash(assembled))


class PortfolioExecutionProgram(_Contract):
    """All owner-resolved inputs that can change holdings, and only those.

    Keyed by ``holdings_spec_hash`` rather than by the whole request. That single
    substitution is what makes the invalidation matrix true rather than intended:
    two specs differing only in cost, benchmark view, report unit or study window
    compile to the *same* program, so they reuse one execution ledger instead of
    re-running a walk-forward to reach identical fills.
    """

    kind: Literal["PortfolioExecutionProgram"] = "PortfolioExecutionProgram"
    holdings_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorities_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The exact PLAN authority receipt this Program was compiled against.

    `None` is accepted only when the Program hash proves the pre-binding legacy
    shape. Current lookup and execution paths require the field.
    """

    strategy_package_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The exact installed frozen strategy this run was compiled under.

    Whole-strategy identity lives here rather than being read out of
    `alpha_recipe_hash`. That overload is what let the Host, the compiler, the
    executor and the report each branch on an Alpha recipe to work out which
    product they were looking at; with the package bound directly none of them
    has to ask.

    `None` exactly for a Program sealed before packages were installed, which
    keeps every predecessor artifact recomputing to its published hash.
    """

    score_source_mode: Literal["HISTORICAL_ARRAY_REPLAY", "CURRENT_MODEL_SCORING"] | None = None
    """Which of the package's installed score sources produced these numbers.

    Two modes of one strategy are not one result: a historical array replay and
    a current model scoring answer the same question from different evidence, so
    they must never share a ledger.
    """

    policy_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_adapter_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    alpha_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    alpha_evidence_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    risk_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    risk_return_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sector_map_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    tradability_decision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_outcome_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Both bound because both decide fills.

    Eligibility and realised outcomes can be corrected while the session and
    listing axes stay exactly the same. Keying only on the axes would let a
    corrected authority reopen a ledger built from the superseded one.
    """

    public_path_source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_sessions_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_start: date
    formation_end: date
    formation_count: int = Field(gt=0)
    listing_count: int = Field(gt=0)
    execution_clock: Literal["EVERY_FORMATION"] = "EVERY_FORMATION"
    sector_forecast_disposition: Literal["SECTOR_FORECAST_NOT_CONSUMED"] = (
        "SECTOR_FORECAST_NOT_CONSUMED"
    )
    continued_from_state_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The sealed boundary a continuation starts from; `None` for a flat start.

    Part of the Program because it is a numerical input like any other: the same
    policy over the same authorities produces different fills from a different
    opening book. Keeping it out would let a continuation reuse a flat-start
    ledger through the by-program index and never notice it had.
    """

    numerical_input_assembly_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """One stored digest over every bound input that decides a number.

    Stored rather than derived. A derived property proves only that the reader
    and the writer run the same code, which is the one thing a replay may not
    assume; stored, it is part of what the Program *says* and a later edit to the
    recipe cannot silently re-answer it.

    `None` exactly for a Program sealed before the assembly was named. Such a
    Program may still be read back, and the absence is load-bearing: it may
    authorize neither unit reuse nor strong numerical replay, because there is
    nothing to compare a rederivation against.
    """

    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal authority-bound execution and derive its numerical input assembly identity.

        Absent legacy optional bindings are omitted from hashing rather than serialized as null,
        preserving compatible flat-start identity.

        Args:
            values: Declared program fields excluding generated program_hash; authorities_hash is
                required.

        Returns:
            Validated program with derived numerical_input_assembly_hash and canonical program_hash.

        Raises:
            PortfolioApplicationError: Required authority is absent or program/assembly consistency
                fails.
        """
        if values.get("authorities_hash") is None:
            raise PortfolioApplicationError("portfolio_application.program_authorities_absent")
        identity = {"kind": "PortfolioExecutionProgram", **values}
        identity["numerical_input_assembly_hash"] = numerical_input_assembly_hash(identity)
        # Mirrors the validator exactly. A caller that passes an absent optional
        # binding as an explicit ``None`` must land on the same hash as one that
        # omits it, or the two spellings of "flat start" would be two Programs.
        for name in PROGRAM_LEGACY_OPTIONAL_FIELDS:
            if identity.get(name) is None:
                identity.pop(name, None)
        return cls(**identity, program_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require ordered execution bounds and exact compatible program/assembly identities.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: Formation bounds are reversed, applicable program hash
                differs or a supplied assembly identity contradicts its bindings.
        """
        if self.formation_start > self.formation_end:
            raise PortfolioApplicationError("portfolio_application.program_axis_invalid")
        identity = self.model_dump(mode="json", exclude={"program_hash"})
        # Absent optional bindings are *popped* rather than hashed as null, so a
        # Program sealed before any of them existed recomputes to the hash it was
        # published under. Hashing the null instead would invalidate every
        # predecessor artifact on disk without a single line of evidence for it.
        for name in PROGRAM_LEGACY_OPTIONAL_FIELDS:
            if getattr(self, name) is None:
                identity.pop(name)
        if self.program_hash != canonical_hash(identity):
            raise PortfolioApplicationError("portfolio_application.program_identity_invalid")
        if self.numerical_input_assembly_hash is not None and (
            self.numerical_input_assembly_hash != numerical_input_assembly_hash(identity)
        ):
            # A stored assembly that does not describe the bindings beside it is
            # worse than none: it would authorize a replay of inputs the Program
            # never bound.
            raise PortfolioApplicationError("portfolio_application.program_assembly_invalid")
        return self

    @property
    def replayable(self) -> bool:
        """Whether this Program may authorize unit reuse or strong replay."""
        return self.numerical_input_assembly_hash is not None


LEDGER_LEGACY_OPTIONAL_FIELDS: tuple[str, ...] = (
    "consumed_score_receipt_hashes",
    "target_weights_hash",
    "realized_simple_returns_hash",
    "execution_available_hash",
    "initial_boundary",
    "final_boundary",
)
"""Replay children a ledger sealed before they were named may not carry.

Popped from the identity when absent for the same reason the Program's are: a
predecessor ledger has to keep recomputing to the hash it was published under.
"""


class SealedPortfolioBoundaryState(_Contract):
    """Every component of the walk-forward state at one end of a path.

    A continuation needs all of it and a replay needs all of it, and the two
    needs are the same need: the state a segment starts from is a numerical
    input, so it has to be sealed with the same care as the returns matrix.

    The four walk-forward components are here in full -- both books and both
    cash balances -- plus the executor-owned sleeve array, which is the only
    other thing that survives a formation. Sealing the reference alone was the
    defect this contract exists to close: a continuation restored from it looks
    right and prices its first trade against a book it never held.
    """

    kind: Literal["SealedPortfolioBoundaryState"] = "SealedPortfolioBoundaryState"
    listing_count: int = Field(gt=0)
    formation_session: date
    """The formation this boundary sits after, so continuity is checkable."""

    pretrade_weights_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pretrade_cash: float
    optimizer_reference_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    optimizer_reference_cash: float
    sleeve_weights_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The policy's book-scale sleeves, or `None` for a policy that holds none.

    Absent is a real answer here rather than a gap: a recipe with no sleeve
    dimension has no sleeve state, and a continuation that invented one would
    start a different strategy than the one that was frozen.
    """

    sleeve_count: int = Field(default=0, ge=0)
    decided_formation_count: int = Field(default=0, ge=0)
    """How many formations the policy had decided when this boundary was taken.

    The schedule position a continuation resumes on. Without it the protected
    segment opens at position zero, `due_sleeves` reads that as "never traded"
    and restages every sleeve -- so the first protected formation trades the
    whole book, and every number after it belongs to a different strategy than
    the one that was frozen.
    """

    boundary_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        identity = cls.model_construct(**values, boundary_hash="0" * 64).model_dump(
            mode="json", exclude={"boundary_hash"}
        )
        return cls(**identity, boundary_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if (self.sleeve_weights_hash is None) != (self.sleeve_count == 0):
            raise PortfolioApplicationError("portfolio_application.boundary_sleeve_axis_invalid")
        if not np.isfinite(self.pretrade_cash) or not np.isfinite(self.optimizer_reference_cash):
            raise PortfolioApplicationError("portfolio_application.boundary_state_invalid")
        if self.boundary_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"boundary_hash"})
        ):
            raise PortfolioApplicationError("portfolio_application.boundary_identity_invalid")
        return self


class PortfolioExecutionLedger(_Contract):
    """Continuous post-fill path, keyed by one exact execution program."""

    kind: Literal["PortfolioExecutionLedger"] = "PortfolioExecutionLedger"
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    executed_weights_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    executed_weights_shape: tuple[int, int]
    pre_cap_weights_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_cap_weights_shape: tuple[int, int]
    final_weights_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    gross_simple_returns: tuple[float, ...]
    one_way_turnovers: tuple[float, ...]
    decision_modes: tuple[str, ...]
    consumed_score_projection_hashes: tuple[str, ...]
    consumed_risk_projection_hashes: tuple[str, ...]
    consumed_score_receipt_hashes: tuple[str, ...] | None = None
    """One receipt per formation over every component row the scores came from.

    Distinct from the projection hashes above: a component book that carries no
    replay projection still has a score lane, and without this a merged book's
    only record of what it read would be the components' silence. `None` exactly
    for a ledger sealed before the receipt was named.
    """
    aggregate_cap_binding_counts: tuple[int, ...]
    missed_execution_count: int = Field(ge=0)
    anchor_simple_returns: tuple[float, ...]
    """The eligible-universe equal-weight anchor over the same axis.

    Program-determined, so it lives here rather than being re-derived by every
    descendant that needs a benchmark or a beta-stripped unit.
    """

    risk_facts: tuple[PortfolioRiskAttributionFacts, ...]
    """Report-side attribution, also program-determined: executed weights and the
    admitted attribution projections fix it, and no descendant control moves it."""

    median_holding_adv20_by_formation: tuple[float | None, ...]
    """Median 20-session dollar volume across the names held at each formation.

    One value per formation, not one for the path: a window ending in March must
    describe the book as it stood in March, and a single path-end figure cannot
    answer that.
    """

    target_weights_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The capped decision targets, one row per formation.

    The *input* to `execute_orders`, not its output, and that distinction is the
    whole reason it is sealed: with the targets, the returns and the availability
    matrix, a replay reruns the transition and derives the executed book, the
    turnover and the fills. Without them a replay can only read the outputs back
    and call that agreement.
    """

    realized_simple_returns_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The per-name outcome matrix drift consumes; the ledger seals only an
    aggregate of it otherwise, which is not enough to rederive the drifted book
    every later turnover is charged against."""

    execution_available_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """Which names could be traded at each formation: what turns an order into a
    fill or a miss."""

    initial_boundary: SealedPortfolioBoundaryState | None = None
    """The state the path started from -- flat for a development run, the frozen
    book for a continuation. A replay that assumed flat would silently rederive a
    different path for every continuation."""

    final_boundary: SealedPortfolioBoundaryState | None = None
    """The state the path ended at, in full. `final_weights_hash` above is one of
    its five components and is kept because existing readers cite it."""

    ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require aligned execution lanes, finite economics and a coherent terminal boundary.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: Weight/return/turnover/mode/cap/Risk/support axes differ,
                values violate finiteness/counts, boundary listing size differs, terminal
                session/reference contradicts final weights, or compatible ledger hash differs.
        """
        rows = len(self.formation_sessions)
        if (
            self.executed_weights_shape != (rows, len(self.ordered_listing_ids))
            or self.pre_cap_weights_shape != self.executed_weights_shape
            or len(self.gross_simple_returns) != rows
            or len(self.one_way_turnovers) != rows
            or len(self.decision_modes) != rows
            or len(self.aggregate_cap_binding_counts) != rows
            or len(self.anchor_simple_returns) != rows
            or len(self.risk_facts) != rows
            or len(self.median_holding_adv20_by_formation) != rows
            or any(value < 0 for value in self.aggregate_cap_binding_counts)
            or not all(np.isfinite(value) for value in self.anchor_simple_returns)
            or not all(np.isfinite(value) for value in self.gross_simple_returns)
            or not all(np.isfinite(value) and value >= 0.0 for value in self.one_way_turnovers)
        ):
            raise PortfolioApplicationError("portfolio_application.ledger_axis_invalid")
        listings = len(self.ordered_listing_ids)
        for boundary in (self.initial_boundary, self.final_boundary):
            if boundary is not None and boundary.listing_count != listings:
                raise PortfolioApplicationError(
                    "portfolio_application.ledger_boundary_axis_invalid"
                )
        if self.final_boundary is not None and (
            self.final_boundary.optimizer_reference_hash != self.final_weights_hash
            or self.final_boundary.formation_session != self.formation_sessions[-1]
        ):
            # The two views of the terminal state have to agree, or a reader
            # picks whichever one suits it.
            raise PortfolioApplicationError("portfolio_application.ledger_boundary_disagrees")
        identity = self.model_dump(mode="json", exclude={"ledger_hash"})
        for name in LEDGER_LEGACY_OPTIONAL_FIELDS:
            if getattr(self, name) is None:
                identity.pop(name)
        if self.ledger_hash != canonical_hash(identity):
            raise PortfolioApplicationError("portfolio_application.ledger_identity_invalid")
        return self

    @property
    def replayable(self) -> bool:
        """Whether every child a strong replay needs was sealed with this path."""
        return (
            self.target_weights_hash is not None
            and self.realized_simple_returns_hash is not None
            and self.execution_available_hash is not None
            and self.initial_boundary is not None
        )


class PortfolioEconomicLedger(_Contract):
    """Cost-only descendant; changing cost never rotates holdings."""

    kind: Literal["PortfolioEconomicLedger"] = "PortfolioEconomicLedger"
    execution_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cost_assumption_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cost_bps_per_side: str
    platform_one_way_cost_bps: str
    net_simple_returns: tuple[float, ...]
    cumulative_net_wealth: float = Field(ge=0.0)
    benchmark_beta: float | None
    """The book's beta to the primary anchor, from the Backtesting owner.

    `active_path_metrics` computes it as `cov(net, anchor, ddof=1) /
    var(anchor, ddof=1)`, the same expression `portfolio_economic_metrics` uses.
    It is estimated once over the **whole materialized path**, never inside a
    selected window: a descriptive window may slice what is displayed and may not
    move a coefficient.
    """

    benchmark_beta_disposition: Literal[
        "BACKTESTING_ACTIVE_PATH_BETA_NET_VS_ANCHOR_FULL_PATH",
        "ANCHOR_DEGENERATE_BETA_NOT_ESTIMABLE",
    ]
    """`None` only when the Backtesting owner refuses the anchor as degenerate.

    A beta over a benchmark that did not move is not a small number, it is not a
    number; the owner says so and this records which case happened. Report units
    that do not need beta keep working, because coupling every view to one
    coefficient would be a worse answer than the honest absence.
    """

    economic_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require finite net returns and beta value/disposition agreement.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: Net returns or estimated beta are nonfinite, beta presence
                contradicts its declared estimation disposition, or economic ledger identity
                differs.
        """
        if not all(np.isfinite(value) for value in self.net_simple_returns):
            raise PortfolioApplicationError("portfolio_application.economic_values_invalid")
        estimated = self.benchmark_beta is not None
        if estimated != (
            self.benchmark_beta_disposition
            == "BACKTESTING_ACTIVE_PATH_BETA_NET_VS_ANCHOR_FULL_PATH"
        ):
            raise PortfolioApplicationError(
                "portfolio_application.economic_beta_disposition_invalid"
            )
        if estimated and not np.isfinite(self.benchmark_beta):
            raise PortfolioApplicationError("portfolio_application.economic_values_invalid")
        if self.economic_ledger_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"economic_ledger_hash"})
        ):
            raise PortfolioApplicationError("portfolio_application.economic_identity_invalid")
        return self


class OwnerCoverage(_Contract):
    """One upstream owner's exact session support, named so a refusal can point at it."""

    kind: Literal["OwnerCoverage"] = "OwnerCoverage"
    owner_id: str = Field(min_length=1)
    lane: str = Field(min_length=1)
    sessions: tuple[date, ...] = Field(min_length=1)
    """The exact ordered axis, carried rather than summarised.

    The common watermark is the intersection of these axes, and an intersection
    cannot be computed from endpoints: two owners can share a first session, a
    last session and a count while disagreeing about a day in between.
    """

    first_session: date
    last_session: date
    session_count: int = Field(gt=0)
    session_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The exact ordered axis, so an interior gap cannot pass as agreement.

    Endpoints and a count are not enough to intersect two owners: a lane missing
    one Wednesday reports the same three numbers as a lane that never had that
    Wednesday, and the watermark would then claim support neither of them has.
    """

    identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def of(
        cls,
        *,
        owner_id: str,
        lane: str,
        sessions: tuple[date, ...],
        identity_hash: str,
    ) -> Self:
        """Build a row from the owner's exact axis rather than from its ends."""
        return cls(
            owner_id=owner_id,
            lane=lane,
            sessions=sessions,
            first_session=sessions[0],
            last_session=sessions[-1],
            session_count=len(sessions),
            session_axis_hash=ordered_session_axis_hash(sessions),
            identity_hash=identity_hash,
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_axis(self) -> Self:
        """Require coverage count, endpoints and session hash to describe the retained axis.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: The retained session axis disagrees with its declared
                count/endpoints/hash.
        """
        if (
            len(self.sessions) != self.session_count
            or self.sessions[0] != self.first_session
            or self.sessions[-1] != self.last_session
            or ordered_session_axis_hash(self.sessions) != self.session_axis_hash
        ):
            raise PortfolioApplicationError("portfolio_application.owner_coverage_axis_invalid")
        return self


class PortfolioSupportCoverage(_Contract):
    """The exact `CommonWatermark`, and the per-owner coverage that produced it.

    The watermark is an intersection, so on its own it can only say "no". The
    per-owner rows are what let a refusal say *which* lane ran out, which is the
    difference between a date picker that greys out and one a researcher can act
    on.
    """

    kind: Literal["PortfolioSupportCoverage"] = "PortfolioSupportCoverage"
    owners: tuple[OwnerCoverage, ...] = Field(min_length=1)
    common_watermark_start: date
    common_watermark_end: date
    common_session_count: int = Field(gt=0)
    alpha_training_window_sessions: Literal[1260] = 1260
    """Alpha's frozen training window, shown separately from score coverage.

    It is not a study bound and never narrows one; it is displayed beside the
    coverage rows so a reader does not mistake the replay's score support for the
    window those models were fitted on.
    """

    coverage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal declared cross-owner support coverage.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical coverage_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, coverage_hash="0" * 64).model_dump(
            mode="json", exclude={"coverage_hash"}
        )
        return cls(**identity, coverage_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require a nonempty ordered owner-axis intersection and exact support identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: Owner coverage has no common session or the common
                count/endpoints/hash differ from the derived intersection.
        """
        common = ordered_axis_intersection([value.sessions for value in self.owners])
        if not common:
            raise PortfolioApplicationError("portfolio_application.common_watermark_axis_invalid")
        if (
            common[0],
            common[-1],
            len(common),
        ) != (
            self.common_watermark_start,
            self.common_watermark_end,
            self.common_session_count,
        ):
            raise PortfolioApplicationError(
                "portfolio_application.common_watermark_not_the_owner_intersection"
            )
        if self.coverage_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"coverage_hash"})
        ):
            raise PortfolioApplicationError("portfolio_application.coverage_identity_invalid")
        return self

    def limiting_owners(self, *, start: date, end: date) -> tuple[str, ...]:
        """Which owners a request escapes. Empty when the request is inside support."""
        return tuple(
            value.owner_id
            for value in self.owners
            if start < value.first_session or end > value.last_session
        )


class PortfolioScheduleGuard(_Contract):
    """The sleeve schedule the selected `tranches` implies, rendered with every path.

    This ships with the control, not after it. The family is derived from the one
    tranche policy owner -- ``due_sleeves`` -- rather than restated, and the
    absence of a phase is a stated fact rather than a missing field: the offset
    only relabels which sleeve goes first, so there is nothing to select.
    """

    kind: Literal["PortfolioScheduleGuard"] = "PortfolioScheduleGuard"
    guard_family: Literal["TRANCHE_SLEEVE_SCHEDULE_FAMILY"] = "TRANCHE_SLEEVE_SCHEDULE_FAMILY"
    tranches: int = Field(ge=1)
    sleeve_share_policy: Literal["EQUAL_NOTIONAL_AT_EACH_FORMATION"] = SLEEVE_SHARE_POLICY
    warmup_formation_count: int = Field(ge=0)
    """Formations before every sleeve has traded once."""

    due_sleeve_cycle: tuple[tuple[int, ...], ...] = Field(min_length=1)
    """One full cycle of which sleeves are due, starting at the first formation."""

    schedule_phase_selectable: Literal[False] = False
    schedule_phase_refusal: Literal["SCHEDULE_PHASE_SELECTION_REFUSED"] = (
        "SCHEDULE_PHASE_SELECTION_REFUSED"
    )
    guard_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal declared tranche scheduling guard.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical guard_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, guard_hash="0" * 64).model_dump(
            mode="json", exclude={"guard_hash"}
        )
        return cls(**identity, guard_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the declared tranche/warmup scheduling cycle and exact guard identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: Due-sleeve cycle length differs from tranches plus warmup
                formations, or guard_hash differs.
        """
        if len(self.due_sleeve_cycle) != self.tranches + self.warmup_formation_count:
            raise PortfolioApplicationError("portfolio_application.schedule_cycle_length_invalid")
        if self.guard_hash != canonical_hash(self.model_dump(mode="json", exclude={"guard_hash"})):
            raise PortfolioApplicationError("portfolio_application.schedule_guard_identity_invalid")
        return self


class PortfolioStudyWindowGuard(_Contract):
    """The selected window, the full-support reference, and why it is descriptive.

    All three render together. A window shown without its full-support reference
    is a claim with the disconfirming half cropped out, which is exactly what the
    ``DESCRIPTIVE_SUBWINDOW`` disposition exists to prevent.
    """

    kind: Literal["PortfolioStudyWindowGuard"] = "PortfolioStudyWindowGuard"
    guard_family: Literal["STUDY_WINDOW_SUPPORT_FAMILY"] = "STUDY_WINDOW_SUPPORT_FAMILY"
    disposition: Literal["DESCRIPTIVE_SUBWINDOW"] = STUDY_WINDOW_DISPOSITION
    requested_start: date
    requested_end: date
    selected_start: date
    selected_end: date
    selected_session_count: int = Field(gt=0)
    clamped_to_materialized_path: bool
    """Whether a historical artifact honoured a narrower window than requested.

    Current construction always writes ``False``: causal rank-mu maturity is an
    owner coverage lane in PLAN, so a request outside the exact executable
    intersection is refused instead of being rewritten after admission. The
    field remains for identity-preserving readback of pre-remediation reports.
    """

    full_support_start: date
    full_support_end: date
    full_support_session_count: int = Field(gt=0)
    is_full_support: bool
    coverage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    may_select_or_promote: Literal[False] = False
    may_carry_claim_authority: Literal[False] = False
    prefix_formation_count: int = Field(ge=0)
    """Formations before the window that establish opening state and are never discarded."""

    guard_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal declared study-window selection guard.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical guard_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, guard_hash="0" * 64).model_dump(
            mode="json", exclude={"guard_hash"}
        )
        return cls(**identity, guard_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require a selected window within support and coherent selection/clamp flags.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: Selected bounds are reversed/outside support, full-support or
                materialized-clamp flags contradict the requested/selected bounds, or guard_hash
                differs.
        """
        if self.selected_start > self.selected_end:
            raise PortfolioApplicationError("portfolio_application.study_window_axis_invalid")
        if (
            self.selected_start < self.full_support_start
            or self.selected_end > self.full_support_end
        ):
            raise PortfolioApplicationError("portfolio_application.study_window_outside_support")
        if self.is_full_support != (
            self.selected_start == self.full_support_start
            and self.selected_end == self.full_support_end
        ):
            raise PortfolioApplicationError("portfolio_application.study_window_flag_invalid")
        if self.clamped_to_materialized_path != (
            self.requested_start != self.selected_start or self.requested_end != self.selected_end
        ):
            raise PortfolioApplicationError("portfolio_application.study_window_clamp_flag_invalid")
        if self.guard_hash != canonical_hash(self.model_dump(mode="json", exclude={"guard_hash"})):
            raise PortfolioApplicationError("portfolio_application.window_guard_identity_invalid")
        return self


def build_schedule_guard(*, tranches: int) -> PortfolioScheduleGuard:
    """Derive the schedule family from the policy owner, never from a restatement.

    ``due_sleeves`` is the same function the executor calls, so the guard a
    researcher reads and the schedule the book actually runs cannot disagree.
    """
    warmup = 1
    cycle = tuple(
        due_sleeves(formation_index=index, tranches=tranches) for index in range(tranches + warmup)
    )
    return PortfolioScheduleGuard.create(
        tranches=tranches,
        sleeve_share_policy=SLEEVE_SHARE_POLICY,
        warmup_formation_count=warmup,
        due_sleeve_cycle=cycle,
    )


def build_study_window_guard(
    *,
    requested_start: date,
    requested_end: date,
    selected_start: date,
    selected_end: date,
    selected_session_count: int,
    coverage: PortfolioSupportCoverage,
    ledger_sessions: tuple[date, ...],
) -> PortfolioStudyWindowGuard:
    """The window, its full-support reference and the prefix it never discards.

    ``prefix_formation_count`` is the number of admitted formations before the
    window. They establish the opening book and are counted here so a reader can
    see that the slice is an observation of a continuous path rather than a
    cold start at the visible date.
    """
    if (
        requested_start != selected_start
        or requested_end != selected_end
        or ledger_sessions[0] != coverage.common_watermark_start
        or ledger_sessions[-1] != coverage.common_watermark_end
        or len(ledger_sessions) != coverage.common_session_count
    ):
        raise PortfolioApplicationError("portfolio_application.study_window_not_honoured_exactly")
    prefix = sum(1 for value in ledger_sessions if value < selected_start)
    return PortfolioStudyWindowGuard.create(
        requested_start=requested_start,
        requested_end=requested_end,
        clamped_to_materialized_path=False,
        selected_start=selected_start,
        selected_end=selected_end,
        selected_session_count=selected_session_count,
        full_support_start=coverage.common_watermark_start,
        full_support_end=coverage.common_watermark_end,
        full_support_session_count=coverage.common_session_count,
        is_full_support=(
            selected_start == coverage.common_watermark_start
            and selected_end == coverage.common_watermark_end
        ),
        coverage_hash=coverage.coverage_hash,
        prefix_formation_count=prefix,
    )


class PortfolioBenchmarkComparison(_Contract):
    """The declared benchmark descendant. Rebuilt by the view, never by a fill."""

    kind: Literal["PortfolioBenchmarkComparison"] = "PortfolioBenchmarkComparison"
    execution_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    economic_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_benchmark: Literal["ELIGIBLE_UNIVERSE_EQUAL_WEIGHT"] = PRIMARY_BENCHMARK
    secondary_benchmark_view: PortfolioBenchmarkView
    primary_simple_returns: tuple[float, ...]
    secondary_benchmark_id: str | None = None
    secondary_simple_returns: tuple[float, ...] | None = None
    secondary_disposition: str = Field(min_length=1)
    comparison_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal declared primary/secondary benchmark comparison.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical comparison_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, comparison_hash="0" * 64).model_dump(
            mode="json", exclude={"comparison_hash"}
        )
        return cls(**identity, comparison_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require benchmark-view authority, aligned lanes and finite primary returns.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: Secondary benchmark ID contradicts its selected view, a
                supplied secondary lane length differs, primary returns are nonfinite or
                comparison_hash differs.
        """
        wants_secondary = self.secondary_benchmark_view == "anchor_plus_spy"
        if wants_secondary != (self.secondary_benchmark_id is not None):
            raise PortfolioApplicationError("portfolio_application.secondary_benchmark_mismatch")
        if self.secondary_simple_returns is not None and len(self.secondary_simple_returns) != len(
            self.primary_simple_returns
        ):
            raise PortfolioApplicationError("portfolio_application.benchmark_axis_invalid")
        if not all(np.isfinite(value) for value in self.primary_simple_returns):
            raise PortfolioApplicationError("portfolio_application.benchmark_values_invalid")
        if self.comparison_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"comparison_hash"})
        ):
            raise PortfolioApplicationError("portfolio_application.benchmark_identity_invalid")
        return self


HELD_WEIGHT_EPSILON: Final[float] = 1e-12
"""Below this a weight is not a position; it is the residue of a float subtraction.

The same threshold decides `window_end_distinct_names`, so a name counted as held
and a name listed in the book are the same set by construction rather than by
two authors agreeing.
"""


class PortfolioBookPosition(_Contract):
    """One name in the window-end book and the change that produced it."""

    kind: Literal["PortfolioBookPosition"] = "PortfolioBookPosition"
    listing_id: str = Field(min_length=1)
    weight: float = Field(ge=0.0)
    """Its share of the book at the window's end."""

    preceding_weight: float = Field(ge=0.0)
    """The same name's share at the change boundary; zero at a sealed opening."""

    weight_change: float
    """`weight - preceding_weight`, written down rather than left to a reader.

    Carried as a value because the report is the only thing allowed to compute
    it: a browser that subtracted these two numbers itself would be a second
    place where a domain figure is produced.
    """

    disposition: Literal["OPENED", "INCREASED", "HELD", "REDUCED", "EXITED"]

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_change(self) -> Self:
        """Require exact weight change and the corresponding held/opened/exited disposition.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: Weight subtraction differs, neither current nor preceding
                weight is held, or the epsilon-based disposition disagrees.
        """
        if self.weight_change != self.weight - self.preceding_weight:
            raise PortfolioApplicationError("portfolio_application.book_change_inconsistent")
        held = self.weight > HELD_WEIGHT_EPSILON
        was_held = self.preceding_weight > HELD_WEIGHT_EPSILON
        if not held and not was_held:
            raise PortfolioApplicationError("portfolio_application.book_position_absent")
        expected = (
            "OPENED"
            if not was_held
            else "EXITED"
            if not held
            else "INCREASED"
            if self.weight_change > HELD_WEIGHT_EPSILON
            else "REDUCED"
            if self.weight_change < -HELD_WEIGHT_EPSILON
            else "HELD"
        )
        if self.disposition != expected:
            raise PortfolioApplicationError("portfolio_application.book_disposition_invalid")
        return self


class PortfolioWindowEndBook(_Contract):
    """The names actually held when the selected window ends, and what moved.

    Every other holdings figure on the report is a count or a concentration
    statistic: how many names, how concentrated, how much turned over. None of
    them answers the question a person opens the report to ask, which is *which
    names, at what size, and what changed*. This is that answer, and it is a
    projection of the sealed executed weights over the ledger's own listing axis
    -- so it is bound to the same identity as everything else on the report and
    cannot describe a different book.

    The change is between two **formation-end** books. It therefore includes the
    drift between them and is not the traded turnover, which the path measures
    against the pre-trade book; the two answer different questions and the report
    says which is which rather than letting a reader assume.
    """

    kind: Literal["PortfolioWindowEndBook"] = "PortfolioWindowEndBook"
    formation_session: date
    """The window's last formation. These weights are the book held at it."""

    change_boundary: Literal[
        "PRECEDING_FORMATION",
        "SEALED_CONTINUATION_BOUNDARY",
        "FLAT_PATH_OPENING",
    ]
    """What the change is measured against, as three distinguishable things.

    `PRECEDING_FORMATION` is the ordinary case: the formation before this one on
    this path.

    `FLAT_PATH_OPENING` is the first formation of a path that started empty.
    There is no earlier book, so every position is an open.

    `SEALED_CONTINUATION_BOUNDARY` is the first formation of a path that started
    from a **frozen book** -- a continuation. Its predecessor is real, it is not
    this path's, and it is not flat. Calling that an opening was the defect this
    field exists to make unstateable: a continuation's first formation would have
    reported every carried name as newly opened, every reduction as an open, and
    no exit at all.
    """

    preceding_formation_session: date | None = None
    positions: tuple[PortfolioBookPosition, ...] = ()
    """Held at the window end or at the boundary, largest ending weight first.

    A name that left the book is a change and appears with a zero ending weight;
    omitting it would make an exit invisible in the one place it is the news.
    """

    held_count: int = Field(ge=0)
    opened_count: int = Field(ge=0)
    exited_count: int = Field(ge=0)
    absolute_weight_change_total: float = Field(ge=0.0)
    """Sum of `|weight_change|` over the rows below, in the order they are stored.

    Derived from the reported positions rather than recomputed from the arrays,
    and checked against them here: a total that came from a different reduction
    could disagree with the table it sits above, and a reader adding the column
    up would be the one who found out. Not turnover; see the class note.
    """

    listing_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The exact ordered listing axis these weights were read over."""

    book_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal declared window-end holdings and change boundary.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical book_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, book_hash="0" * 64).model_dump(
            mode="json", exclude={"book_hash"}
        )
        return cls(**identity, book_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require coherent opening/continuation holdings and exact change projections.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: Boundary clocks/preceding session disagree, listing positions
                repeat, held/opened/exited counts or absolute change total differ, a flat opening
                has prior mass/exits, or book_hash differs.
        """
        flat = self.change_boundary == "FLAT_PATH_OPENING"
        if flat != (self.preceding_formation_session is None):
            # A continuation boundary has a session and has to name it; only a
            # flat opening has nothing earlier to point at.
            raise PortfolioApplicationError("portfolio_application.book_boundary_mismatch")
        if self.preceding_formation_session is not None and (
            self.preceding_formation_session >= self.formation_session
        ):
            raise PortfolioApplicationError("portfolio_application.book_boundary_not_earlier")
        listings = tuple(value.listing_id for value in self.positions)
        if len(set(listings)) != len(listings):
            raise PortfolioApplicationError("portfolio_application.book_listing_duplicated")
        held = tuple(value for value in self.positions if value.weight > HELD_WEIGHT_EPSILON)
        if self.held_count != len(held):
            raise PortfolioApplicationError("portfolio_application.book_held_count_invalid")
        if self.opened_count != sum(
            value.disposition == "OPENED" for value in self.positions
        ) or self.exited_count != sum(value.disposition == "EXITED" for value in self.positions):
            raise PortfolioApplicationError("portfolio_application.book_change_count_invalid")
        if flat and any(value.preceding_weight > HELD_WEIGHT_EPSILON for value in self.positions):
            # The one claim a flat opening makes is that nothing was held before
            # it. A book that carries a position and calls itself flat is the
            # exact misreport this contract is here to refuse.
            raise PortfolioApplicationError("portfolio_application.book_flat_opening_is_not_flat")
        if flat and self.exited_count:
            raise PortfolioApplicationError("portfolio_application.book_opening_has_exit")
        if self.absolute_weight_change_total != sum(
            abs(value.weight_change) for value in self.positions
        ):
            raise PortfolioApplicationError("portfolio_application.book_change_total_inconsistent")
        if self.book_hash != canonical_hash(self.model_dump(mode="json", exclude={"book_hash"})):
            raise PortfolioApplicationError("portfolio_application.book_identity_invalid")
        return self


class PortfolioDeclaredPathReport(_Contract):
    """Typed report projection over one declared path; it performs no selection."""

    kind: Literal["PortfolioDeclaredPathReport"] = "PortfolioDeclaredPathReport"
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    economic_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_comparison_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_unit: PortfolioReportUnit
    ledger_coverage: PortfolioLedgerCoverage
    """The continuous path this report is a view of, stated as an interval.

    This is the installed reporting geometry. The Campaign codec it replaces
    pinned a formation count, so a workspace holding any other number of
    formations had no lawful way to report at all; that codec now reads back
    frozen artifacts and authorizes nothing current.
    """

    schedule_guard: PortfolioScheduleGuard
    window_guard: PortfolioStudyWindowGuard
    control_receipt: PortfolioControlReceipt
    window_end_book: PortfolioWindowEndBook
    """Which names are held at the window's end, at what size, and what changed."""

    risk_facts: tuple[PortfolioRiskAttributionFacts, ...]
    """Sliced to the selected window; the last is the attribution at its end."""

    window_end_distinct_names: int = Field(ge=0)
    window_end_effective_n: float = Field(ge=0.0)
    window_cumulative_net_wealth: float = Field(ge=0.0)
    """Compounded over the window only. The full-path figure is on the economic ledger."""

    mean_one_way_turnover: float = Field(ge=0.0)
    aggregate_cap_binding_sessions: int = Field(ge=0)
    aggregate_cap_binding_names_total: int = Field(ge=0)
    window_scope: Literal["ALL_FACTS_AT_OR_INSIDE_THE_SELECTED_WINDOW"] = (
        "ALL_FACTS_AT_OR_INSIDE_THE_SELECTED_WINDOW"
    )
    """Stated because the alternative is worse than wrong: a window that changed
    only the row table while holdings, Risk, cap counts and liquidity still
    described the path end would read as a windowed report and be a full-path
    one."""

    capacity_disposition: Literal["NOT_MODELED"] = "NOT_MODELED"
    """Never an estimate. Concentration and liquidity proxies render separately."""

    window_end_median_holding_adv20_dollar_volume: float | None = None
    """Median 20-session dollar volume across the names held at the window end.

    A liquidity descriptor in dollars, not a fraction and not a capacity
    estimate. `None` when the ADV lane is absent.
    """

    window_unit_rows: tuple[tuple[str, float], ...] = ()
    """The selected report unit's rows over the selected window; empty is truthful."""

    limitations: tuple[str, ...]
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_geometry(self) -> Self:
        """The report and its geometry must describe the same ledger and window."""
        if self.ledger_coverage.ledger_hash != self.execution_ledger_hash:
            raise PortfolioApplicationError("portfolio_application.report_geometry_ledger_mismatch")
        if self.ledger_coverage.program_hash != self.program_hash:
            raise PortfolioApplicationError(
                "portfolio_application.report_geometry_program_mismatch"
            )
        if not self.ledger_coverage.contains(
            start=self.window_guard.selected_start, end=self.window_guard.selected_end
        ):
            raise PortfolioApplicationError("portfolio_application.report_window_outside_geometry")
        if self.window_end_book.listing_axis_hash != self.ledger_coverage.listing_axis_hash:
            # A book read over a different listing axis is a book of a different
            # path, however plausible its names look.
            raise PortfolioApplicationError("portfolio_application.report_book_axis_mismatch")
        if self.window_end_book.formation_session != self.window_guard.selected_end:
            raise PortfolioApplicationError("portfolio_application.report_book_window_mismatch")
        if self.window_end_book.held_count != self.window_end_distinct_names:
            # The count and the names it counts must be the same set.
            raise PortfolioApplicationError("portfolio_application.report_book_count_mismatch")
        return self

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require report Risk facts inside the selected window and compatible nested identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: Risk dates/counts exceed the selected window or report_hash
                differs after compatible historical control-receipt normalization.
        """
        if self.risk_facts and (
            self.risk_facts[0].formation_session < self.window_guard.selected_start
            or self.risk_facts[-1].formation_session > self.window_guard.selected_end
        ):
            raise PortfolioApplicationError("portfolio_application.report_risk_outside_window")
        if len(self.risk_facts) not in (0, self.window_guard.selected_session_count):
            raise PortfolioApplicationError("portfolio_application.report_risk_axis_invalid")
        identity = self.model_dump(mode="json", exclude={"report_hash"})
        receipt = identity["control_receipt"]
        if isinstance(receipt, dict) and not receipt.get("catalog_control_ids"):
            # The receipt validator already proved the legacy nested identity.
            # Preserve the parent report's exact pre-field hash as well.
            receipt.pop("catalog_control_ids", None)
        if self.report_hash != canonical_hash(identity):
            raise PortfolioApplicationError("portfolio_application.report_identity_invalid")
        return self


class PortfolioResearchResult(_Contract):
    """Safe readback handles returned by RUN or exact reuse."""

    kind: Literal["PortfolioResearchResult"] = "PortfolioResearchResult"
    action: Literal["PUBLISHED", "REUSED_EXACT"]
    spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The whole request. Two specs can share a program and still differ here."""

    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    economic_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_uri: str
    html_uri: str
    export_command: str | None = None
    """The command that reproduced the request, held only by a result published before V403,
    whose hash covers its words; a later result's is composed from its Task's request when read,
    so a script's grammar never moves a result's identity."""

    result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require result identity excluding its action projection.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: The canonical result payload excluding action/result_hash
                differs.
        """
        members = self.model_dump(mode="json", exclude={"action", "result_hash"})
        if members["export_command"] is None:
            # Its identity binds the request (spec_hash), never the words of a command (V403).
            del members["export_command"]
        if self.result_hash != canonical_hash(members):
            raise PortfolioApplicationError("portfolio_application.result_identity_invalid")
        return self


class PortfolioPlanPreview(BaseModel):  # type: ignore[misc]
    """What PLAN may say, and the evidence that it said it without running anything.

    The four zero counters are not decoration. They are the machine-checkable
    form of "PLAN performs no numerical work", and the tests assert them against
    instrumented owners rather than trusting the field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["PortfolioPlanPreview"] = "PortfolioPlanPreview"
    workspace_id: str
    workspace_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    holdings_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorities_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy_package_id: str
    strategy_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_source_mode: ScoreSourceMode
    control_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage: PortfolioSupportCoverage
    schedule_guard: PortfolioScheduleGuard
    selected_study_start: str
    selected_study_end: str
    full_support_start: str
    full_support_end: str
    study_window_disposition: Literal["DESCRIPTIVE_SUBWINDOW"] = "DESCRIPTIVE_SUBWINDOW"
    candidate_formation_count: int = Field(gt=0)
    listing_count: int = Field(gt=0)
    eligible_count: int = Field(gt=0)
    cache_state: Literal[
        "RESULT_HIT",
        "EXECUTION_LEDGER_HIT",
        "FULL_NUMERICAL_MISS",
    ]
    """Three answers, because they mean three different amounts of work.

    A published result for this exact request is `RESULT_HIT` and costs nothing.
    A materialized path for this holdings configuration with a different
    descendant is `EXECUTION_LEDGER_HIT`: no walk, no replay, no Risk surface,
    only the descendants. Anything else is `FULL_NUMERICAL_MISS`. Collapsing the
    middle case into "miss" would tell a researcher that changing a report unit
    costs a walk-forward.
    """

    exact_cache_hit: bool
    cached_result_hash: str | None = None
    execution_ledger_coverage: tuple[str, str, int] | None = None
    """The materialized path's first session, last session and formation count.

    Reported separately from score and Risk coverage because they are different
    questions: an owner can have support the ledger has not been run over yet.
    `None` when no ledger exists for this holdings configuration.
    """

    comparison_plan: Literal["NO_INSTALLED_COMPARISON_OR_CV_PLAN"] = (
        "NO_INSTALLED_COMPARISON_OR_CV_PLAN"
    )
    """Stated rather than omitted.

    Portfolio cross-validation and configuration selection belong to the Lab, not
    to the public path. A preview that simply left the field out would read as an
    oversight; saying there is none is the honest form.
    """

    prefix_work_formation_count: int = Field(ge=0)
    """Formations that must run before the requested window can be projected.

    Zero on a cache hit. Otherwise the whole continuous path, because a study
    window is an observation control and may never become a cold start at the
    visible date.
    """

    estimated_score_replays: int = Field(ge=0)
    estimated_risk_surface_builds: int = Field(ge=0)
    work_estimate_basis: Literal["UPPER_BOUND_OVER_CANDIDATE_SUPPORT"] = (
        "UPPER_BOUND_OVER_CANDIDATE_SUPPORT"
    )
    """These are bounds, and saying so is the point.

    The executed axis is shorter than the candidate support by the rank-mu
    warm-up, and finding that boundary means building the curve -- numerical work
    ``PLAN`` must not do. So the estimate is taken over the candidate support and
    labelled as an upper bound rather than quietly reported as the exact count.
    """

    legal_recovery: tuple[str, ...]
    alpha_fit_count: Literal[0] = 0
    optimizer_call_count: Literal[0] = 0
    dense_covariance_materialization_count: Literal[0] = 0
    sector_forecast_call_count: Literal[0] = 0
    publication_count: Literal[0] = 0
    protected_read_count: Literal[0] = 0
    refusals: tuple[str, ...]
    preview_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def admission_hash(self) -> str:
        """What the run was authorised to do, without what the store happened to hold.

        `preview_hash` covers the whole preview including its cache state and
        work estimate, and those are observations about the store rather than
        part of the request: publishing a result changes them, so a task bound to
        the full preview could never be recovered after it had published. This
        covers the request, the authorities, the catalog, the window and the
        guards -- everything a later run must not silently differ on.
        """
        return str(
            canonical_hash(
                {
                    "workspace_id": self.workspace_id,
                    "workspace_manifest_hash": self.workspace_manifest_hash,
                    "spec_hash": self.spec_hash,
                    "holdings_spec_hash": self.holdings_spec_hash,
                    "authorities_hash": self.authorities_hash,
                    "strategy_catalog_hash": self.strategy_catalog_hash,
                    "strategy_package_id": self.strategy_package_id,
                    "strategy_package_hash": self.strategy_package_hash,
                    "score_source_mode": self.score_source_mode,
                    "control_catalog_hash": self.control_catalog_hash,
                    "coverage_hash": self.coverage.coverage_hash,
                    "schedule_guard_hash": self.schedule_guard.guard_hash,
                    "selected_study_start": self.selected_study_start,
                    "selected_study_end": self.selected_study_end,
                    "eligible_count": self.eligible_count,
                }
            )
        )

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal declared support/cache/work plan preview.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical preview_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, preview_hash="0" * 64).model_dump(
            mode="json", exclude={"preview_hash"}
        )
        return cls(**identity, preview_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require coherent support, cache-hit authority and numerical work estimates.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Study/support bounds, exact-cache flags/result identity, score/Risk/prefix
                work estimates or materialized ledger coverage contradict cache state, or
                preview_hash differs.
        """
        if (
            self.full_support_start != self.coverage.common_watermark_start.isoformat()
            or self.full_support_end != self.coverage.common_watermark_end.isoformat()
        ):
            raise ValueError("portfolio_application.plan_full_support_mismatch")
        if not (
            self.full_support_start
            <= self.selected_study_start
            <= self.selected_study_end
            <= self.full_support_end
        ):
            raise ValueError("portfolio_application.plan_study_window_outside_support")
        if self.exact_cache_hit != (self.cache_state == "RESULT_HIT"):
            raise ValueError("portfolio_application.plan_cache_state_invalid")
        if self.exact_cache_hit != (self.cached_result_hash is not None):
            raise ValueError("portfolio_application.plan_cache_state_invalid")
        # The work estimate has to follow the state, or the state is decoration.
        expected_work = self.cache_state == "FULL_NUMERICAL_MISS"
        if expected_work != bool(self.estimated_score_replays):
            raise ValueError("portfolio_application.plan_work_estimate_inconsistent")
        if expected_work != bool(self.estimated_risk_surface_builds):
            raise ValueError("portfolio_application.plan_work_estimate_inconsistent")
        if expected_work != bool(self.prefix_work_formation_count):
            raise ValueError("portfolio_application.plan_cache_work_inconsistent")
        # An `EXECUTION_LEDGER_HIT` is exactly the case that knows its coverage:
        # the path exists. A miss cannot, and a hit that reported none would be
        # claiming a path it could not describe.
        if (self.cache_state == "FULL_NUMERICAL_MISS") == (
            self.execution_ledger_coverage is not None
        ):
            raise ValueError("portfolio_application.plan_ledger_coverage_inconsistent")
        if self.execution_ledger_coverage is not None:
            start, end, count = self.execution_ledger_coverage
            # Separation, enforced: the materialized path is inside the
            # watermark and is not permitted to *be* it by construction. One is
            # what the owners can serve, the other is what has been run.
            if start < self.coverage.common_watermark_start.isoformat():
                raise ValueError("portfolio_application.plan_ledger_before_watermark")
            if end > self.coverage.common_watermark_end.isoformat():
                raise ValueError("portfolio_application.plan_ledger_after_watermark")
            if count > self.coverage.common_session_count:
                raise ValueError("portfolio_application.plan_ledger_exceeds_watermark")
        if self.preview_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"preview_hash"})
        ):
            raise ValueError("portfolio_application.plan_preview_identity_invalid")
        return self


class PortfolioReadouts(BaseModel):  # type: ignore[misc]
    """Every number a panel shows, taken from the report rather than recomputed.

    ``capacity`` is a disposition, not a value. Daily OHLCV does not authorise a
    market-impact claim, so the liquidity proxy travels beside it under its own
    name instead of being presented as one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["PortfolioReadouts"] = "PortfolioReadouts"
    distinct_names_held: int = Field(ge=0)
    effective_n: float = Field(ge=0.0)
    one_way_turnover_per_trading_session: float = Field(ge=0.0)
    turnover_basis: Literal["PER_TRADING_SESSION_NOT_ANNUAL"] = "PER_TRADING_SESSION_NOT_ANNUAL"
    aggregate_cap_binding_sessions: int = Field(ge=0)
    aggregate_cap_binding_names_total: int = Field(ge=0)
    cumulative_net_wealth: float = Field(ge=0.0)
    cost_bps_per_side: str
    cost_bps_round_trip: str
    platform_one_way_cost_bps: str
    evidence_cost_ladder_bps_per_side: tuple[str, ...] = EVIDENCE_COST_LADDER_BPS_PER_SIDE
    capacity: Literal["NOT_MODELED"] = CAPACITY_DISPOSITION
    median_holding_adv20_dollar_volume: float | None = None
    liquidity_proxy_disposition: Literal["CONCENTRATION_DESCRIPTOR_NOT_A_CAPACITY_ESTIMATE"] = (
        "CONCENTRATION_DESCRIPTOR_NOT_A_CAPACITY_ESTIMATE"
    )
    industry_attribution_available: bool
    readouts_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal declared retained portfolio read models.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical readouts_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, readouts_hash="0" * 64).model_dump(
            mode="json", exclude={"readouts_hash"}
        )
        return cls(**identity, readouts_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact retained portfolio read-model identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioApplicationError: The canonical readouts payload differs from readouts_hash.
        """
        if self.readouts_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"readouts_hash"})
        ):
            raise PortfolioApplicationError("portfolio_application.readouts_identity_invalid")
        return self


def export_command(*, workspace_id: str, spec: PortfolioResearchSpec) -> str:
    """The command that reproduces this exact request.

    Every non-default control is written out. The exported form carries values a
    caller can read and re-enter, never an identity to be replayed as authority:
    ``--spec-hash`` is a *check* the adapter verifies against the spec it rebuilt
    from these flags, not a way to name a request the compiler never saw.
    """
    parts = [f"alphalattice-portfolio --workspace {workspace_id} run"]
    if spec.strategy_package_id != WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID:
        parts.append(f"--strategy-package-id {spec.strategy_package_id}")
    parts.append(f"--score-source-mode {spec.score_source_mode}")
    if spec.top_k != DEFAULT_TOP_K:
        parts.append(f"--top-k {spec.top_k}")
    if spec.tranches != DEFAULT_TRANCHES:
        parts.append(f"--tranches {spec.tranches}")
    if spec.exit_rank != 2 * spec.top_k:
        parts.append(f"--exit-rank {spec.exit_rank}")
    if spec.weight_rule != DEFAULT_WEIGHT_RULE:
        parts.append(f"--weight-rule {spec.weight_rule}")
    if str(spec.cost.cost_bps_per_side) != DEFAULT_COST_BPS_PER_SIDE:
        parts.append(f"--cost-bps-per-side {spec.cost.cost_bps_per_side}")
    if spec.secondary_benchmark_view != DEFAULT_BENCHMARK_VIEW:
        parts.append(f"--secondary-benchmark-view {spec.secondary_benchmark_view}")
    if spec.report_unit != DEFAULT_REPORT_UNIT:
        parts.append(f"--report-unit {spec.report_unit}")
    if spec.study_start is not None:
        parts.append(f"--study-start {spec.study_start.isoformat()}")
    if spec.study_end is not None:
        parts.append(f"--study-end {spec.study_end.isoformat()}")
    parts.append(f"--spec-hash {spec.spec_hash}")
    return " ".join(parts)


__all__ = [
    "HELD_WEIGHT_EPSILON",
    "REQUEST_WEIGHT_RULES",
    "WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID",
    "OwnerCoverage",
    "PortfolioApplicationError",
    "PortfolioBenchmarkComparison",
    "PortfolioBookPosition",
    "PortfolioControlReceipt",
    "PortfolioDeclaredPathReport",
    "PortfolioEconomicLedger",
    "PortfolioExecutionLedger",
    "PortfolioExecutionProgram",
    "PortfolioPlanPreview",
    "PortfolioReadouts",
    "PortfolioResearchResult",
    "PortfolioResearchSpec",
    "PortfolioScheduleGuard",
    "PortfolioStudyWindowGuard",
    "PortfolioSupportCoverage",
    "PortfolioWeightRule",
    "PortfolioWindowEndBook",
    "ScoreSourceMode",
    "build_schedule_guard",
    "build_study_window_guard",
    "export_command",
]
