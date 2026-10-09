"""The public control surface: what it admits, what it refuses, and what it rebuilds.

Every entry point compiles through one spec, so these tests are written against
that spec rather than against any adapter. A direct caller reaching
``PortfolioResearchSpec.create`` gets exactly the bounds a UI would show, which
is the point: there is no second validation path to bypass.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    OwnerCoverage,
    PortfolioApplicationError,
    PortfolioResearchSpec,
    PortfolioScheduleGuard,
    PortfolioStudyWindowGuard,
    PortfolioSupportCoverage,
    build_schedule_guard,
    build_study_window_guard,
)
from alphalattice.investment.portfolio_strategy_lab.application.controls import (
    ADMITTED_BENCHMARK_VIEWS,
    ADMITTED_REPORT_UNITS,
    EVIDENCE_COST_LADDER_BPS_PER_SIDE,
    INSTALLED_PUBLIC_CONTROL_CATALOG,
    REFUSED_CONTROL_IDS,
    PublicControlError,
    build_public_control_catalog,
)
from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import (
    ADMITTED_WEIGHT_RULES,
    TOP_K_RANGE,
    TRANCHES_RANGE,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

HASH = "a" * 64
CATALOG = INSTALLED_PUBLIC_CONTROL_CATALOG


# ------------------------------------------------------------------ the catalog


def test_the_catalog_installs_exactly_the_nine_admitted_controls() -> None:
    assert tuple(value.control_id for value in CATALOG.controls) == (
        "top_k",
        "tranches",
        "weight_rule",
        "exit_rank",
        "cost_bps_per_side",
        "secondary_benchmark_view",
        "report_unit",
        "study_start",
        "study_end",
    )


def test_the_catalog_is_derived_from_its_owners_not_transcribed() -> None:
    """A range typed twice is a range that will eventually differ.

    The holdings bounds must be the policy owner's, so that widening `top_k` in
    `TrancheBookRecipe` cannot leave the public surface describing the old band.
    """

    assert CATALOG.descriptor("top_k").min == Decimal(TOP_K_RANGE[0])
    assert CATALOG.descriptor("top_k").max == Decimal(TOP_K_RANGE[1])
    assert CATALOG.descriptor("tranches").min == Decimal(TRANCHES_RANGE[0])
    assert CATALOG.descriptor("tranches").max == Decimal(TRANCHES_RANGE[1])
    assert CATALOG.descriptor("weight_rule").options == ADMITTED_WEIGHT_RULES
    assert CATALOG.descriptor("report_unit").options == ADMITTED_REPORT_UNITS
    assert CATALOG.descriptor("secondary_benchmark_view").options == (ADMITTED_BENCHMARK_VIEWS)


def test_every_control_is_a_complete_browser_contract() -> None:
    """A client renders fields and never has to interpret an admitted-surface sentence."""

    for control in CATALOG.controls:
        assert control.label
        assert control.help
        assert control.refusal
        assert control.default_display
        if control.value_kind == "ENUM":
            assert control.options
            assert control.default_value in control.options
        else:
            assert not control.options
        if control.value_kind in {"INTEGER", "FIXED_POINT_DECIMAL"}:
            assert control.step is not None
        if control.disposition == "SHIP_GUARDED":
            assert control.guard


def test_the_catalog_is_stable_and_tamper_evident() -> None:
    assert build_public_control_catalog().catalog_hash == CATALOG.catalog_hash
    assert build_public_control_catalog().presentation_hash == CATALOG.presentation_hash
    payload = CATALOG.model_dump(mode="json")
    payload["controls"][0]["default_display"] = "999"
    with pytest.raises(ValueError, match="control_presentation_identity_invalid"):
        type(CATALOG).model_validate(payload)


def test_display_copy_does_not_rotate_the_executable_catalog_identity() -> None:
    payload = CATALOG.model_dump(mode="json")
    payload["controls"][0]["help"] = "A clearer explanation of the same admitted control."
    payload["presentation_hash"] = canonical_hash(
        {
            key: value
            for key, value in payload.items()
            if key not in {"catalog_hash", "presentation_hash"}
        }
    )
    presentation = type(CATALOG).model_validate(payload)

    assert presentation.catalog_hash == CATALOG.catalog_hash
    assert presentation.presentation_hash != CATALOG.presentation_hash


def test_every_guarded_control_names_its_guard_family() -> None:
    """The guard ships with the control, so the catalog cannot describe one without it."""

    for control_id in ("tranches", "study_start", "study_end"):
        assert CATALOG.descriptor(control_id).disposition == "SHIP_GUARDED"
        assert CATALOG.guard_family(control_id)
    assert CATALOG.guard_family("tranches") == "TRANCHE_SLEEVE_SCHEDULE_FAMILY"
    assert CATALOG.guard_family("study_start") == "STUDY_WINDOW_SUPPORT_FAMILY"


def test_the_invalidation_class_of_every_control_is_declared() -> None:
    """This mapping is the invalidation matrix, stated where callers can read it."""

    assert CATALOG.rebuilt_by("top_k") == "EXECUTION_LEDGER"
    assert CATALOG.rebuilt_by("tranches") == "EXECUTION_LEDGER"
    assert CATALOG.rebuilt_by("weight_rule") == "EXECUTION_LEDGER"
    assert CATALOG.rebuilt_by("exit_rank") == "EXECUTION_LEDGER"
    assert CATALOG.rebuilt_by("cost_bps_per_side") == "ECONOMIC_OVERLAY"
    assert CATALOG.rebuilt_by("secondary_benchmark_view") == "BENCHMARK_DESCENDANT"
    assert CATALOG.rebuilt_by("report_unit") == "REPORT_UNIT_DESCENDANT"
    assert CATALOG.rebuilt_by("study_start") == "STUDY_WINDOW_DESCENDANT"
    assert CATALOG.rebuilt_by("study_end") == "STUDY_WINDOW_DESCENDANT"


def test_the_refusal_catalog_names_evidence_and_never_overlaps_the_admitted_set() -> None:
    """A refusal is a typed result with a reference, not a greyed-out widget."""

    assert len(CATALOG.refusals) == 23
    assert not REFUSED_CONTROL_IDS & {value.control_id for value in CATALOG.controls}
    for refusal in CATALOG.refusals:
        # Why, in the product's words, and what the product serves or a law; never a plan.
        assert refusal.reason and "implemented-plans" not in refusal.evidence_reference
        assert not refusal.evidence_reference.endswith(".md")
        assert refusal.refusal_code.endswith("_REFUSED")
    for control_id in (
        "schedule_phase",
        "research_effort",
        "alpha_recipe",
        "alpha_training_window",
        "risk_estimator",
        "optimizer_or_solver",
        "artifact_path",
        "current_pointer",
        "identity_hash_authority",
        "protected_window_handle",
        "broker_or_order",
    ):
        assert CATALOG.refusal(control_id).control_id == control_id


def test_an_uninstalled_control_is_refused_by_name() -> None:
    with pytest.raises(PublicControlError, match="control_not_installed:sigma_exponent"):
        CATALOG.descriptor("sigma_exponent")


def test_the_evidence_cost_ladder_is_the_fixed_five_rung_one() -> None:
    assert EVIDENCE_COST_LADDER_BPS_PER_SIDE == ("0", "2.5", "5", "10", "20")


# --------------------------------------------------------------------- the spec


def test_the_default_spec_is_the_frozen_product_strategy() -> None:
    spec = PortfolioResearchSpec.default()
    assert (spec.top_k, spec.tranches, spec.exit_rank, spec.weight_rule) == (35, 3, 70, "mu.iv1")
    assert spec.cost.cost_bps_per_side == Decimal(5)
    assert spec.secondary_benchmark_view == "anchor_only"
    assert spec.report_unit == "MONTHLY_BETA_STRIPPED_LEDGER"
    assert (spec.study_start, spec.study_end) == (None, None)
    assert spec.is_default()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", TOP_K_RANGE[0] - 1),
        ("top_k", TOP_K_RANGE[1] + 1),
        ("tranches", TRANCHES_RANGE[0] - 1),
        ("tranches", TRANCHES_RANGE[1] + 1),
    ],
)
def test_a_direct_caller_cannot_escape_the_installed_range(field: str, value: int) -> None:
    """There is no separate UI validator to be stricter than this one."""

    refusal = {
        "top_k": "TOP_K_OUTSIDE_ADMITTED_RANGE",
        "tranches": "TRANCHE_COUNT_OUTSIDE_ADMITTED_RANGE",
    }[field]
    with pytest.raises(ValueError, match=refusal):
        PortfolioResearchSpec.create(**{field: value})  # type: ignore[arg-type]


def test_an_uninstalled_weight_rule_has_the_catalogued_refusal() -> None:
    with pytest.raises(ValueError, match=CATALOG.descriptor("weight_rule").refusal):
        PortfolioResearchSpec.create(weight_rule="not-installed")  # type: ignore[arg-type]


def test_top_k_starts_at_the_aggregate_cap_feasibility_floor() -> None:
    """17 is `ceil(1 / 0.06)`; 16 names cannot fill a book under a 6% name cap."""

    assert TOP_K_RANGE[0] == 17
    assert PortfolioResearchSpec.create(top_k=17).top_k == 17


@pytest.mark.parametrize("weight_rule", ADMITTED_WEIGHT_RULES)
def test_every_admitted_weight_rule_compiles(weight_rule: str) -> None:
    spec = PortfolioResearchSpec.create(weight_rule=weight_rule)  # type: ignore[arg-type]
    assert spec.weight_rule == weight_rule


def test_exit_rank_is_a_cross_field_band_that_moves_with_top_k() -> None:
    """The band is `top_k <= exit_rank <= 6 * top_k`, so it is not a fixed range."""

    assert PortfolioResearchSpec.create(top_k=20).exit_rank_band() == (20, 120)
    assert PortfolioResearchSpec.create(top_k=20, exit_rank=20).exit_rank == 20
    assert PortfolioResearchSpec.create(top_k=20, exit_rank=120).exit_rank == 120
    with pytest.raises(ValueError, match="exit_rank_outside_band"):
        PortfolioResearchSpec.create(top_k=20, exit_rank=121)
    # 30 is a legal absolute rank at top_k=20 and illegal at top_k=35, which is
    # exactly why it cannot be validated as a plain integer range.
    assert PortfolioResearchSpec.create(top_k=20, exit_rank=30).exit_rank == 30
    with pytest.raises(ValueError, match="exit_rank_outside_band"):
        PortfolioResearchSpec.create(top_k=35, exit_rank=30)


def test_the_exit_rank_default_is_twice_top_k_at_every_width() -> None:
    for top_k in (17, 35, 75):
        assert PortfolioResearchSpec.create(top_k=top_k).exit_rank == 2 * top_k


def test_the_rolling_excess_view_is_refused_with_its_own_code() -> None:
    with pytest.raises(PortfolioApplicationError, match="rolling_excess_view_refused"):
        PortfolioResearchSpec.create(report_unit="ROLLING_EXCESS")


def test_an_uninstalled_enum_value_is_refused_rather_than_coerced() -> None:
    with pytest.raises(PortfolioApplicationError, match="report_unit_not_installed:WEEKLY"):
        PortfolioResearchSpec.create(report_unit="WEEKLY")
    with pytest.raises(PortfolioApplicationError, match="benchmark_view_not_installed:spy_only"):
        PortfolioResearchSpec.create(secondary_benchmark_view="spy_only")


@pytest.mark.parametrize("per_side", EVIDENCE_COST_LADDER_BPS_PER_SIDE)
def test_every_evidence_ladder_rung_is_an_admissible_cost(per_side: str) -> None:
    spec = PortfolioResearchSpec.create(cost_bps_per_side=per_side)
    assert spec.cost.cost_bps_per_side == Decimal(per_side)
    assert spec.cost.platform_one_way_cost_bps == Decimal(per_side) * 2


@pytest.mark.parametrize(
    ("cost", "code"),
    [("2.25", "per_side_cost_not_fixed_point"), ("20.5", "less than or equal")],
    ids=["off_the_frozen_increment", "above_the_admitted_ceiling"],
)
def test_a_cost_outside_the_admitted_grid_is_refused_not_rounded(cost: str, code: str) -> None:
    with pytest.raises(ValueError, match=code):
        PortfolioResearchSpec.create(cost_bps_per_side=cost)


def test_an_inverted_study_window_is_refused() -> None:
    with pytest.raises(ValueError, match="study_window_axis_invalid"):
        PortfolioResearchSpec.create(study_start=date(2024, 3, 1), study_end=date(2024, 1, 1))


def test_the_spec_round_trips_through_json_and_refuses_a_tampered_payload() -> None:
    spec = PortfolioResearchSpec.create(top_k=40, cost_bps_per_side="2.5", tranches=4)
    payload = spec.model_dump(mode="json")
    assert PortfolioResearchSpec.model_validate(payload) == spec
    payload["top_k"] = 41
    with pytest.raises(ValueError, match="spec_identity_invalid"):
        PortfolioResearchSpec.model_validate(payload)


# ------------------------------------------------- what each control invalidates


def test_only_holdings_controls_rotate_the_holdings_identity() -> None:
    """The invalidation matrix, proved on the identity that keys an execution ledger."""

    base = PortfolioResearchSpec.default()
    for changed in (
        PortfolioResearchSpec.create(cost_bps_per_side="10"),
        PortfolioResearchSpec.create(secondary_benchmark_view="anchor_plus_spy"),
        PortfolioResearchSpec.create(report_unit="SIMPLE_CUMULATIVE"),
        PortfolioResearchSpec.create(study_start=date(2023, 1, 3)),
        PortfolioResearchSpec.create(study_end=date(2024, 1, 3)),
    ):
        assert changed.spec_hash != base.spec_hash
        assert changed.holdings_spec_hash == base.holdings_spec_hash, (
            "a descendant control must not rotate the execution ledger key"
        )
    for changed in (
        PortfolioResearchSpec.create(top_k=40),
        PortfolioResearchSpec.create(tranches=4),
        PortfolioResearchSpec.create(weight_rule="iv1"),
        PortfolioResearchSpec.create(exit_rank=80),
    ):
        assert changed.holdings_spec_hash != base.holdings_spec_hash


def test_the_holdings_identity_carries_no_descendant_field() -> None:
    identity = PortfolioResearchSpec.default().holdings_identity()
    assert set(identity) == {
        "execution_mode",
        "strategy_package_id",
        "score_source_mode",
        "top_k",
        "tranches",
        "exit_rank",
        "weight_rule",
        "sleeve_share_policy",
        "aggregate_name_cap",
        "sector_forecast_disposition",
    }


def test_only_the_frozen_tuple_reports_as_the_default() -> None:
    assert PortfolioResearchSpec.default().is_default()
    for changed in (
        PortfolioResearchSpec.create(top_k=40),
        PortfolioResearchSpec.create(cost_bps_per_side="10"),
        PortfolioResearchSpec.create(report_unit="CALENDAR_YEAR_TABLE"),
        PortfolioResearchSpec.create(study_start=date(2023, 1, 3)),
    ):
        assert not changed.is_default()


# ------------------------------------------------------------------- the guards


def test_the_schedule_guard_comes_from_the_policy_owner() -> None:
    """One sleeve trades per formation after the opening, at every admitted width."""

    guard = build_schedule_guard(tranches=3)
    assert guard.due_sleeve_cycle == ((0, 1, 2), (1,), (2,), (0,))
    assert guard.warmup_formation_count == 1
    assert guard.sleeve_share_policy == "EQUAL_NOTIONAL_AT_EACH_FORMATION"
    wide = build_schedule_guard(tranches=10)
    assert wide.due_sleeve_cycle[0] == tuple(range(10))
    assert all(len(value) == 1 for value in wide.due_sleeve_cycle[1:])
    assert wide.guard_hash != guard.guard_hash


def test_the_schedule_guard_states_that_phase_is_not_selectable() -> None:
    """Absent is not the same as stated. The guard says so, with a refusal code."""

    guard = build_schedule_guard(tranches=4)
    assert guard.schedule_phase_selectable is False
    assert guard.schedule_phase_refusal == "SCHEDULE_PHASE_SELECTION_REFUSED"

    # Two independent refusals, and the semantic one fires first by design: a
    # restated `tranches` that no longer matches the cycle is caught as a wrong
    # schedule rather than merely as a wrong hash.
    payload = guard.model_dump(mode="json")
    payload["tranches"] = 5
    with pytest.raises(ValueError, match="schedule_cycle_length_invalid"):
        PortfolioScheduleGuard.model_validate(payload)

    reworded = guard.model_dump(mode="json")
    reworded["guard_hash"] = "b" * 64
    with pytest.raises(ValueError, match="schedule_guard_identity_invalid"):
        PortfolioScheduleGuard.model_validate(reworded)


def _daily(first: date, last: date) -> tuple[date, ...]:
    """A dense exact axis between two dates, for a coverage row that needs one."""

    return tuple(first + timedelta(days=index) for index in range((last - first).days + 1))


def _coverage(*, alpha_last: date = date(2024, 8, 12)) -> PortfolioSupportCoverage:
    alpha = _daily(date(2022, 7, 1), alpha_last)
    risk = _daily(date(2021, 1, 4), date(2024, 8, 20))
    return PortfolioSupportCoverage.create(
        owners=(
            OwnerCoverage.of(
                owner_id="alpha_replay",
                lane="product-score",
                sessions=alpha,
                identity_hash=HASH,
            ),
            OwnerCoverage.of(
                owner_id="risk_return_surface",
                lane="causal-open-return",
                sessions=risk,
                identity_hash=HASH,
            ),
        ),
        common_watermark_start=date(2022, 7, 1),
        common_watermark_end=alpha_last,
        common_session_count=len(alpha),
    )


def _executable_coverage(sessions: tuple[date, ...]) -> PortfolioSupportCoverage:
    """Exact support for guard tests, including the policy maturity owner."""

    return PortfolioSupportCoverage.create(
        owners=(
            OwnerCoverage.of(
                owner_id="causal_rank_mu",
                lane="holding-end-ready",
                sessions=sessions,
                identity_hash=HASH,
            ),
        ),
        common_watermark_start=sessions[0],
        common_watermark_end=sessions[-1],
        common_session_count=len(sessions),
    )


def test_the_common_watermark_must_be_the_owner_intersection() -> None:
    """A watermark that is not the intersection is a claim no owner supports."""

    coverage = _coverage()
    assert coverage.common_watermark_start == date(2022, 7, 1)
    payload = coverage.model_dump(mode="json")
    payload["common_watermark_end"] = "2024-08-20"
    with pytest.raises(ValueError, match="not_the_owner_intersection"):
        PortfolioSupportCoverage.model_validate(payload)


def test_coverage_names_the_owner_a_request_escapes() -> None:
    """A refusal that cannot say which lane ran out is a greyed-out date picker."""

    coverage = _coverage()
    assert coverage.limiting_owners(start=date(2022, 7, 1), end=date(2024, 8, 12)) == ()
    assert coverage.limiting_owners(start=date(2021, 1, 4), end=date(2024, 8, 12)) == (
        "alpha_replay",
    )
    assert set(coverage.limiting_owners(start=date(2020, 1, 2), end=date(2025, 1, 2))) == {
        "alpha_replay",
        "risk_return_surface",
    }


def test_the_alpha_training_window_is_reported_separately_from_coverage() -> None:
    """1,260 sessions is what the models were fitted on, not what the study spans."""

    coverage = _coverage()
    assert coverage.alpha_training_window_sessions == 1260
    assert coverage.common_session_count != coverage.alpha_training_window_sessions


def test_the_window_guard_carries_its_full_support_reference_and_prefix() -> None:
    sessions = tuple(date(2023, 1, 3) + timedelta(days=index) for index in range(40))
    guard = build_study_window_guard(
        requested_start=sessions[12],
        requested_end=sessions[-1],
        selected_start=sessions[12],
        selected_end=sessions[-1],
        selected_session_count=28,
        coverage=_executable_coverage(sessions),
        ledger_sessions=sessions,
    )
    assert guard.disposition == "DESCRIPTIVE_SUBWINDOW"
    assert guard.is_full_support is False
    assert (guard.full_support_start, guard.full_support_end) == (sessions[0], sessions[-1])
    assert guard.prefix_formation_count == 12
    assert guard.may_select_or_promote is False
    assert guard.may_carry_claim_authority is False


def test_the_full_window_reports_itself_as_full_support() -> None:
    sessions = tuple(date(2023, 1, 3) + timedelta(days=index) for index in range(40))
    guard = build_study_window_guard(
        requested_start=sessions[0],
        requested_end=sessions[-1],
        selected_start=sessions[0],
        selected_end=sessions[-1],
        selected_session_count=40,
        coverage=_executable_coverage(sessions),
        ledger_sessions=sessions,
    )
    assert guard.is_full_support is True
    assert guard.prefix_formation_count == 0


def test_a_window_outside_the_materialized_path_is_refused() -> None:
    sessions = tuple(date(2023, 1, 3) + timedelta(days=index) for index in range(40))
    with pytest.raises(ValueError, match="study_window_outside_support"):
        build_study_window_guard(
            requested_start=sessions[0] - timedelta(days=1),
            requested_end=sessions[-1],
            selected_start=sessions[0] - timedelta(days=1),
            selected_end=sessions[-1],
            selected_session_count=41,
            coverage=_executable_coverage(sessions),
            ledger_sessions=sessions,
        )


def test_the_window_guard_full_support_flag_cannot_lie() -> None:
    sessions = tuple(date(2023, 1, 3) + timedelta(days=index) for index in range(40))
    guard = build_study_window_guard(
        requested_start=sessions[5],
        requested_end=sessions[-1],
        selected_start=sessions[5],
        selected_end=sessions[-1],
        selected_session_count=35,
        coverage=_executable_coverage(sessions),
        ledger_sessions=sessions,
    )
    payload = guard.model_dump(mode="json")
    payload["is_full_support"] = True
    with pytest.raises(ValueError, match="study_window_flag_invalid"):
        PortfolioStudyWindowGuard.model_validate(payload)


def test_a_window_cannot_be_narrowed_after_admission() -> None:
    """A request outside executable support is refused, never rewritten."""

    sessions = tuple(date(2023, 1, 3) + timedelta(days=index) for index in range(40))
    with pytest.raises(ValueError, match="study_window_not_honoured_exactly"):
        build_study_window_guard(
            requested_start=sessions[0] - timedelta(days=30),
            requested_end=sessions[-1],
            selected_start=sessions[0],
            selected_end=sessions[-1],
            selected_session_count=40,
            coverage=_executable_coverage(sessions),
            ledger_sessions=sessions,
        )
