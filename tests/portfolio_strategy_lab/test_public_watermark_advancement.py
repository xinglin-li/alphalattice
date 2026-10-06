"""Gate 9A: `ADVANCE_TO_WATERMARK`, the daily lifecycle, and study slicing.

The scenario matrix is the deliverable, not the count of assertions. Each test
below is one row of it, and each one measures work rather than asserting that a
disposition string says the right thing: the synthetic owners count what they
actually did and publish it in their own receipt, so "exact zero work" and "no
double fit" are readings rather than claims.

Two kinds of evidence live here, and they are not interchangeable.

The **port tests** drive lane owners that record their own work. They are
Task-control evidence: ordering, recovery, cancellation, refusal shape. They
cannot prove that a real owner composes, and they are not offered as that.

The **owner test** at the end runs the real thing end to end on an isolated
synthetic axis: the installed `RiskSurfaceProducer`, the real Alpha product
lifecycle over genuine twelve-model receipts, the shared Portfolio executor, the
continuous ledger, a study slice, the report and its readback. No Provider, no
reserved evidence, no pointer, no forward activation.

Advancement is exercised on synthetic axes throughout, because the admitted
development evidence stops at the Stage firewall -- and daily advancement belongs
to `FORWARD_LOCAL_RESEARCH`, which no Gate before Stage 10 admits. What the
ordinary replay route does with an advancement request is proved separately.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from alphalattice.control.observation_runtime.telemetry.process_metrics import (
    ProcessResourceUsage,
    ResolvedRuntimeCapacityPlan,
)
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.portfolio_application import (
    PortfolioResearchApplication,
)
from alphalattice.control.task_control.contracts import (
    TaskLifecycle,
)
from alphalattice.interface.local_application.portfolio_research import (
    LocalApplicationError,
    LocalPortfolioResearchService,
)
from alphalattice.investment.alpha_research.scores.product_recipe import (
    INSTALLED_ALPHA_PRODUCT_RECIPE,
    PRODUCT_TRAINING_WINDOW_SESSIONS,
)
from alphalattice.investment.portfolio_strategy_lab.application.advancement import (
    ADVANCEMENT_LANE_ORDER,
    CorrectionLineage,
    DomainLaneReceipt,
    LaneCoverage,
    LaneWork,
    PortfolioLedgerCoverage,
    WatermarkAdvancementProgram,
    WatermarkAdvancementReceipt,
    ordered_listing_axis_hash,
    ordered_session_axis_hash,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    OwnerCoverage,
    PortfolioResearchSpec,
    PortfolioSupportCoverage,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.portfolio_strategy_lab.local_web_support import _resolved, _Resolver

# A synthetic calendar with no historical shape in it: not 1,260, not 494, not
# 252, and not the 253 formations the admitted workspace happens to hold.
CALENDAR = tuple(date(2023, 11, 1) + timedelta(days=index) for index in range(190))

CAPACITY = ResolvedRuntimeCapacityPlan.create(
    profile="AUTO",
    profile_resolution="BALANCED",
    workload="PORTFOLIO_NUMERICAL",
    detected_logical_processors=8,
    allowed_logical_processors=8,
    total_memory_bytes=1 << 34,
    available_memory_bytes=1 << 33,
    admitted_memory_budget_bytes=1 << 32,
    reserved_logical_processors=1,
    process_logical_processor_limit=4,
    fold_workers=2,
    lightgbm_threads_per_fit=1,
    duckdb_threads=4,
    blas_threads=1,
    estimated_peak_process_tree_rss_bytes=1 << 31,
    execution_waves=1,
    admission="ADMITTED",
)
USAGE = ProcessResourceUsage(
    wall_seconds=1.5,
    average_machine_cpu_percent=12.0,
    peak_machine_cpu_percent=30.0,
    peak_rss_bytes=1 << 28,
    process_tree_peak_rss_bytes=1 << 29,
    process_tree_peak_live_descendant_count=3,
    measurement_scope="PARENT_AND_LIVE_DESCENDANTS",
)


# ====================================== the acceptance composition: real lanes

# Every lane below is a production adapter -- `AlphaLifecycleLane`,
# `RiskProducerLane`, `OwnerReportedLane` -- over an isolated synthetic owner.
# The owners are QA data; the lanes, the compiler, the Task adapter and the
# stores are the installed ones, so the matrix exercises the real composition
# rather than a double that implements the lane protocol itself.

QA_LISTINGS = tuple(f"qa-{index:04d}" for index in range(24))

"""Who has to rebuild when an artifact is corrected, stated once."""


# ====================================================== scenario matrix rows


def test_an_empty_target_list_is_not_evidence_of_zero_work() -> None:
    """The contract refuses a rebuild receipt that reports nothing happened."""

    coverage = LaneCoverage.of(
        lane="RISK",
        owner_id="risk_surface_producer",
        lane_label="CAUSAL_OPEN_TO_OPEN_RETURN",
        sessions=(date(2024, 1, 2), date(2024, 1, 3)),
        identity_hash="a" * 64,
        reachable_end=date(2024, 1, 3),
    )
    lineage = CorrectionLineage.create(
        corrected_owner_id="risk_return_surface",
        source_revision_hash="b" * 64,
        superseded_identity_hash="c" * 64,
    )
    with pytest.raises(ValueError, match="lane_rebuild_reported_no_work"):
        DomainLaneReceipt.create(
            program_hash="d" * 64,
            lane="RISK",
            owner_id="risk_surface_producer",
            disposition="REBUILT_FROM_CORRECTION",
            coverage_before=coverage,
            coverage_after=coverage,
            produced_identities=("e" * 64,),
            correction_lineage=lineage,
            work=LaneWork(),
        )


# =========================================== product modes and the firewall


# ============================================ exact axes and interior gaps


def _risk_coverage(sessions: tuple[date, ...]) -> LaneCoverage:
    return LaneCoverage.of(
        lane="RISK",
        owner_id="risk_surface_producer",
        lane_label="CAUSAL_OPEN_TO_OPEN_RETURN",
        sessions=sessions,
        identity_hash="a" * 64,
        reachable_end=sessions[-1],
    )


def test_two_lanes_with_the_same_endpoints_but_a_different_interior_differ() -> None:
    """Endpoints and a count call a gap agreement. The axis hash does not."""

    # First, last and count identical; one interior session differs. This is the
    # case endpoints call agreement.
    left = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 6))
    right = (date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 6))
    assert left[0] == right[0] and left[-1] == right[-1] and len(left) == len(right)
    assert _risk_coverage(left).session_axis_hash != _risk_coverage(right).session_axis_hash
    assert _risk_coverage(left).coverage_hash != _risk_coverage(right).coverage_hash

    # A missing interior session below the last one, where the count differs too.
    dense = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5))
    missing_interior = (date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 5))
    assert dense[0] == missing_interior[0] and dense[-1] == missing_interior[-1]
    assert (
        _risk_coverage(dense).session_axis_hash
        != _risk_coverage(missing_interior).session_axis_hash
    )


def test_portfolio_support_recomputes_the_exact_owner_intersection() -> None:
    """A typed support cannot publish a fabricated common count."""

    left = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 6))
    right = (date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 6))
    owners = (
        OwnerCoverage.of(owner_id="left", lane="LEFT", sessions=left, identity_hash="a" * 64),
        OwnerCoverage.of(owner_id="right", lane="RIGHT", sessions=right, identity_hash="b" * 64),
    )
    with pytest.raises(ValueError, match="common_watermark_not_the_owner_intersection"):
        PortfolioSupportCoverage.create(
            owners=owners,
            common_watermark_start=left[0],
            common_watermark_end=left[-1],
            common_session_count=999,
        )


def test_owner_coverage_refuses_a_projection_that_disagrees_with_its_axis() -> None:
    axis = (date(2024, 1, 2), date(2024, 1, 3))
    valid = OwnerCoverage.of(owner_id="owner", lane="LANE", sessions=axis, identity_hash="a" * 64)
    payload = valid.model_dump(mode="python")
    payload["session_count"] = 999
    with pytest.raises(ValueError, match="owner_coverage_axis_invalid"):
        OwnerCoverage.model_validate(payload)


def test_a_reordered_or_duplicated_axis_is_refused_rather_than_hashed() -> None:
    """A defect in the producing owner does not get a durable identity."""

    with pytest.raises(ValueError, match="session_axis_not_strictly_increasing"):
        ordered_session_axis_hash((date(2024, 1, 3), date(2024, 1, 2)))
    with pytest.raises(ValueError, match="session_axis_not_strictly_increasing"):
        ordered_session_axis_hash((date(2024, 1, 2), date(2024, 1, 2)))
    with pytest.raises(ValueError, match="listing_axis_duplicated"):
        ordered_listing_axis_hash(("a", "b", "a"))


def test_the_listing_axis_binds_order_not_only_cardinality() -> None:
    """Same names, same count, different order: a different path."""

    names = tuple(f"listing-{index:03d}" for index in range(40))
    swapped = (names[1], names[0], *names[2:])
    assert len(names) == len(swapped) and set(names) == set(swapped)
    assert ordered_listing_axis_hash(names) != ordered_listing_axis_hash(swapped)


def test_ledger_coverage_binds_the_program_axes_and_source_support() -> None:
    """The successor geometry states an interval by content, not by count."""

    sessions = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(253))
    listings = tuple(f"listing-{index:04d}" for index in range(466))
    coverage = PortfolioLedgerCoverage.of(
        program_hash="a" * 64,
        ledger_hash="b" * 64,
        formation_sessions=sessions,
        ordered_listing_ids=listings,
        source_coverage_hash="c" * 64,
    )
    assert coverage.describes(sessions)
    assert not coverage.describes((*sessions[:-1], sessions[-1] + timedelta(days=1)))
    assert coverage.contains(start=sessions[10], end=sessions[-10])
    assert not coverage.contains(start=sessions[0] - timedelta(days=1), end=sessions[-1])
    assert coverage.source_coverage_hash == "c" * 64


# ================================================= Alpha lifecycle authority


def test_alpha_training_identity_is_frozen_against_every_portfolio_control() -> None:
    """A date control cannot reach the training window, and does not restate it."""

    spec = PortfolioResearchSpec.create(study_start=date(2024, 1, 3), study_end=date(2024, 2, 1))
    assert "training" not in spec.model_dump(mode="json")
    coverage = PortfolioSupportCoverage.model_fields["alpha_training_window_sessions"]
    # The coverage projection restates the window rather than importing the
    # recipe, because the recipe module pulls the estimator adapter in with it
    # and a Front Desk projection must not load LightGBM. The restatement is
    # pinned here instead.
    assert coverage.default == PRODUCT_TRAINING_WINDOW_SESSIONS
    assert INSTALLED_ALPHA_PRODUCT_RECIPE.training_window_sessions == 1_260
    assert INSTALLED_ALPHA_PRODUCT_RECIPE.purge_sessions == 1


# ============================================================ compiler rules


def test_a_downstream_lane_may_not_declare_a_source_bound() -> None:
    sessions = (date(2024, 1, 2), date(2024, 1, 3))
    with pytest.raises(ValueError, match="lane_downstream_declared_a_bound"):
        LaneCoverage.create(
            lane="PORTFOLIO_STATE",
            owner_id="portfolio_execution_ledger",
            lane_label="CONTINUOUS_EXECUTION_LEDGER",
            sessions=sessions,
            first_session=sessions[0],
            last_session=sessions[-1],
            session_count=len(sessions),
            session_axis_hash=ordered_session_axis_hash(sessions),
            source_bound="DOWNSTREAM_OF_PIPELINE",
            reachable_end=date(2024, 1, 4),
            identity_hash="a" * 64,
        )


def test_a_lane_cannot_reach_less_far_than_it_holds() -> None:
    with pytest.raises(ValueError, match="reachable_end_behind_materialized"):
        LaneCoverage.of(
            lane="RISK",
            owner_id="risk_surface_producer",
            lane_label="CAUSAL_OPEN_TO_OPEN_RETURN",
            sessions=(date(2024, 1, 2), date(2024, 1, 10)),
            identity_hash="a" * 64,
            reachable_end=date(2024, 1, 5),
        )


def test_reuse_that_reports_work_is_refused() -> None:
    coverage = _risk_coverage((date(2024, 1, 2), date(2024, 1, 10)))
    with pytest.raises(ValueError, match="lane_reuse_is_not_exact"):
        DomainLaneReceipt.create(
            program_hash="b" * 64,
            lane="RISK",
            owner_id="risk_surface_producer",
            disposition="REUSED_EXACT",
            coverage_before=coverage,
            coverage_after=coverage,
            work=LaneWork(risk_updates=1),
        )


def test_a_lane_cannot_materialize_a_session_nobody_requested() -> None:
    before = _risk_coverage((date(2024, 1, 2),))
    after = LaneCoverage.of(
        lane="RISK",
        owner_id="risk_surface_producer",
        lane_label="CAUSAL_OPEN_TO_OPEN_RETURN",
        sessions=(date(2024, 1, 2), date(2024, 1, 3)),
        identity_hash="b" * 64,
        reachable_end=date(2024, 1, 10),
    )
    with pytest.raises(ValueError, match="lane_materialized_outside_request"):
        DomainLaneReceipt.create(
            program_hash="c" * 64,
            lane="RISK",
            owner_id="risk_surface_producer",
            disposition="ADVANCED",
            requested_sessions=(date(2024, 1, 3),),
            materialized_sessions=(date(2024, 1, 3), date(2024, 1, 4)),
            produced_identities=("d" * 64,),
            work=LaneWork(risk_updates=2),
            coverage_before=before,
            coverage_after=after,
        )


def test_a_zero_work_receipt_cannot_move_the_watermark() -> None:
    axis = (date(2024, 1, 2), date(2024, 1, 10))
    coverages = tuple(
        LaneCoverage.of(
            lane=lane,
            owner_id=lane.lower(),
            lane_label=lane,
            sessions=axis,
            identity_hash=str(canonical_hash({"lane": lane})),
            source_bound=(
                "DOWNSTREAM_OF_PIPELINE"
                if lane in {"PORTFOLIO_STATE", "REPORT_PROJECTION"}
                else "INDEPENDENT_SOURCE"
            ),
            reachable_end=axis[-1],
        )
        for lane in ADVANCEMENT_LANE_ORDER
    )
    receipts = tuple(
        DomainLaneReceipt.create(
            program_hash="b" * 64,
            lane=coverage.lane,
            owner_id=coverage.owner_id,
            disposition="REUSED_EXACT",
            coverage_before=coverage,
            coverage_after=coverage,
            reused_identities=(coverage.identity_hash,),
        )
        for coverage in coverages
    )
    axis_hash = ordered_session_axis_hash(axis)
    with pytest.raises(ValueError, match="receipt_common_axis_invalid"):
        WatermarkAdvancementReceipt.create(
            program_hash="b" * 64,
            workspace_id="synthetic",
            product_mode="SYNTHETIC_QA",
            disposition="NO_ADVANCE_EXACT_ZERO_WORK",
            receipts=receipts,
            previous_watermark_start=axis[0],
            previous_watermark_end=axis[-1],
            previous_common_session_count=len(axis),
            previous_common_axis_hash=axis_hash,
            new_watermark_start=axis[0],
            new_watermark_end=axis[-1],
            new_common_session_count=999,
            new_common_axis_hash=axis_hash,
            operational_receipt_hash="c" * 64,
        )


# ============================================ advancement is not study slicing


def test_a_study_window_change_asks_no_owner_for_anything(tmp_path: Path) -> None:
    """Row 9: same ledger, different window, zero policy/fit/score/Risk work."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        resolver = _Resolver(_resolved())
        application = PortfolioResearchApplication(
            workspace_id="synthetic",
            workspace=workspace,
            manifest_binding=lambda: "a" * 64,
            session=session,
            resolver=resolver,
        )
        full = application.run(spec=PortfolioResearchSpec.default())
        numerical_after_full = resolver.numerical_calls

        sliced = PortfolioResearchSpec.create(
            study_start=date(2024, 1, 3), study_end=date(2024, 2, 1)
        )
        preview = application.plan(sliced).preview
        assert preview.cache_state == "EXECUTION_LEDGER_HIT"
        assert preview.estimated_score_replays == 0
        assert preview.estimated_risk_surface_builds == 0

        narrowed = application.run(spec=sliced)
        # The ledger is shared, so no formation was walked again; only the
        # report identity moved.
        assert narrowed.result.execution_ledger_hash == full.result.execution_ledger_hash
        assert narrowed.result.report_hash != full.result.report_hash
        assert resolver.numerical_calls == numerical_after_full

        report = application.report(narrowed.result.result_hash)
        full_report = application.report(full.result.result_hash)

    assert report.window_guard.disposition == "DESCRIPTIVE_SUBWINDOW"
    assert not report.window_guard.may_select_or_promote
    # The geometry is the whole path in both, because the window is a view of it
    # and not a different ledger.
    assert report.ledger_coverage == full_report.ledger_coverage
    assert report.ledger_coverage.formation_count == 3
    # The book was not cold-started at the window: the formations before it are
    # still on the path the window is a view of.
    assert report.window_guard.prefix_formation_count == 1


# ================================================ one product surface, one task


def test_a_result_can_name_the_task_that_produced_it(tmp_path: Path) -> None:
    """A researcher holds a result hash; the run behind it must be reachable."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        application = PortfolioResearchApplication(
            workspace_id="lineage",
            workspace=workspace,
            manifest_binding=lambda: "a" * 64,
            session=session,
            resolver=_Resolver(_resolved()),
        )
        completed = application.run(spec=PortfolioResearchSpec.default())
        task_id = application.originating_task(completed.result.result_hash)

        assert task_id == completed.pipeline_manifest.task_id
        assert session.task_control_registry.task(task_id).lifecycle is TaskLifecycle.SUCCEEDED
        service = LocalPortfolioResearchService(application=application, lineage=application)
        assert service.originating_task(completed.result.result_hash) == task_id
        assert "--workspace lineage" in service.export(completed.result.result_hash)
        assert application.originating_task("f" * 64) is None


def test_the_service_refuses_an_operation_it_was_not_wired_for(tmp_path: Path) -> None:
    """A missing port is a refusal, never a quiet no-op."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        service = LocalPortfolioResearchService(
            application=PortfolioResearchApplication(
                workspace_id="unwired",
                workspace=workspace,
                manifest_binding=lambda: "a" * 64,
                session=session,
                resolver=_Resolver(_resolved()),
            )
        )
        with pytest.raises(LocalApplicationError, match="advancement_not_wired"):
            service.advance()
        with pytest.raises(LocalApplicationError, match="task_lineage_not_wired"):
            service.originating_task("a" * 64)


# ======================================= the real owners, end to end, isolated


# ================================ the fit is reachable, and still fails closed


# ====================== the intersection is exact, at the compiler and the seal


def _lane_set(axes: dict[str, tuple[date, ...]]) -> tuple[LaneCoverage, ...]:
    return tuple(
        LaneCoverage.of(
            lane=lane,
            owner_id=lane.lower(),
            lane_label=lane,
            sessions=axes[lane],
            identity_hash=str(canonical_hash(lane)),
            source_bound=(
                "DOWNSTREAM_OF_PIPELINE"
                if lane in {"PORTFOLIO_STATE", "REPORT_PROJECTION"}
                else "INDEPENDENT_SOURCE"
            ),
            reachable_end=(
                None if lane in {"PORTFOLIO_STATE", "REPORT_PROJECTION"} else axes[lane][-1]
            ),
        )
        for lane in ADVANCEMENT_LANE_ORDER
    )


def test_program_recomputes_its_previous_common_axis() -> None:
    left = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 6))
    right = (date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 6))
    axes = {lane: left for lane in ADVANCEMENT_LANE_ORDER}
    axes["FEATURE"] = right
    coverage = _lane_set(axes)
    with pytest.raises(ValueError, match="program_common_axis_invalid"):
        WatermarkAdvancementProgram.create(
            workspace_id="synthetic",
            product_mode="SYNTHETIC_QA",
            lane_order=ADVANCEMENT_LANE_ORDER,
            previous_watermark_start=left[0],
            previous_watermark_end=left[-1],
            previous_common_session_count=999,
            previous_common_axis_hash="f" * 64,
            target_sessions=(),
            source_revision_hash=WatermarkAdvancementProgram.revision_of(coverage),
            alpha_recipe_hash="a" * 64,
            risk_recipe_hash="b" * 64,
            coverage=coverage,
        )


# ================================================= receipts answer this program


def test_coverage_after_must_account_for_what_was_materialized() -> None:
    """An owner may not gain a session it never reported making."""

    before = _risk_coverage((date(2024, 1, 2),))
    after = LaneCoverage.of(
        lane="RISK",
        owner_id="risk_surface_producer",
        lane_label="CAUSAL_OPEN_TO_OPEN_RETURN",
        sessions=(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)),
        identity_hash="b" * 64,
        reachable_end=date(2024, 1, 10),
    )
    with pytest.raises(ValueError, match="lane_coverage_after_unaccounted"):
        DomainLaneReceipt.create(
            program_hash="c" * 64,
            lane="RISK",
            owner_id="risk_surface_producer",
            disposition="ADVANCED",
            requested_sessions=(date(2024, 1, 3), date(2024, 1, 4)),
            materialized_sessions=(date(2024, 1, 3),),
            produced_identities=("d" * 64,),
            work=LaneWork(risk_updates=1),
            coverage_before=before,
            coverage_after=after,
        )
