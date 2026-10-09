"""Default public Portfolio application, Task recovery seam, and readback."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from alphalattice.capabilities.portfolio_backtesting.active_metrics import (
    ActiveMetricsError,
    active_path_metrics,
)
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.portfolio_application import (
    PlannedPortfolioResearch,
    PortfolioResearchApplication,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.workspace_runtime.content_store import (
    ContentAddressedStoreError,
)
from alphalattice.interface.local_application.portfolio_research import (
    LocalPortfolioResearchService,
)
from alphalattice.investment.portfolio_strategy_lab.application import (
    contracts as portfolio_application_contracts,
)
from alphalattice.investment.portfolio_strategy_lab.application import (
    resolution as portfolio_research_composition,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    HELD_WEIGHT_EPSILON,
    OwnerCoverage,
    PortfolioBenchmarkComparison,
    PortfolioControlReceipt,
    PortfolioDeclaredPathReport,
    PortfolioPlanPreview,
    PortfolioResearchResult,
    PortfolioResearchSpec,
    PortfolioSupportCoverage,
    export_command,
)
from alphalattice.investment.portfolio_strategy_lab.application.controls import (
    INSTALLED_PUBLIC_CONTROL_CATALOG,
)
from alphalattice.investment.portfolio_strategy_lab.application.executor import (
    ResolvedPortfolioExecution,
)
from alphalattice.investment.portfolio_strategy_lab.application.resolution import (
    ResolvedPortfolioAuthorities,
)
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    PortfolioResearchTaskAdapter,
    portfolio_research_task_contract,
)
from alphalattice.investment.portfolio_strategy_lab.inputs.shared_lanes import (
    PortfolioResearchCompositionError,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.static import (
    render_portfolio_research_html,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.units import (
    project_report_unit_rows,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from run_public_portfolio_research import build_parser, spec_from_args
from tests.portfolio_strategy_lab.local_web_support import (
    _HASH,
    TEST_PACKAGE,
    _Harness,
    _harness,
    _resolved,
    _Resolver,
    _run,
)

"""One installed package for the fixture, declared through the real contract.

The double stands in for a Host binding, so it has to answer the same questions
one does. Inventing a bare hash here would let the fixture pass a Program the
product could never compile.
"""


def test_default_path_runs_under_task_control_and_reuses_exact_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = PortfolioResearchSpec.default()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        application = PortfolioResearchApplication(
            workspace_id="synthetic",
            workspace=workspace,
            manifest_binding=lambda: _HASH,
            session=session,
            resolver=_Resolver(_resolved()),
        )
        planned = application.plan(spec)
        assert not planned.preview.exact_cache_hit
        assert planned.preview.optimizer_call_count == 0
        first = application.run(spec=spec, planned=planned)
        assert first.result.action == "PUBLISHED"
        assert len(application.report(first.result.result_hash).risk_facts) == 3
        assert "--workspace synthetic" in application.export(first.result.result_hash)

        second_plan = application.plan(spec)
        assert second_plan.preview.exact_cache_hit
        second = application.run(spec=spec, planned=second_plan)
        assert second.result.action == "REUSED_EXACT"
        assert second.result.result_hash == first.result.result_hash
        assert application.pipeline.load(second.pipeline_manifest.manifest_hash) == (
            second.pipeline_manifest
        )


def test_a_results_identity_binds_its_request_never_a_commands_words(tmp_path: Path) -> None:
    """A result's identity binds its request independently of the command's wording."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = PortfolioResearchSpec.default()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        application = PortfolioResearchApplication(
            workspace_id="synthetic",
            workspace=workspace,
            manifest_binding=lambda: _HASH,
            session=session,
            resolver=_Resolver(_resolved()),
        )
        result = application.run(spec=spec, planned=application.plan(spec)).result
        members = result.model_dump(
            mode="json", exclude={"action", "result_hash", "export_command"}
        )
        assert result.export_command is None
        assert result.result_hash == canonical_hash(members)
        command = export_command(workspace_id="synthetic", spec=spec)
        assert application.export(result.result_hash) == command

    earlier = {**members, "export_command": command}
    published = PortfolioResearchResult(
        action="PUBLISHED", **earlier, result_hash=canonical_hash(earlier)
    )
    assert published.export_command == command
    with pytest.raises(ValueError, match="result_identity_invalid"):
        PortfolioResearchResult(action="PUBLISHED", **members, result_hash=published.result_hash)


def test_only_one_application_session_can_own_a_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        assert session.writer_lease.held
        assert session.task_control_registry.database_path == (
            workspace / "runtime" / "research-task-control.duckdb"
        )
        assert session.runtime_path.parent == workspace / "runtime"
        with pytest.raises(RuntimeError, match="already owned"):
            WorkspaceApplicationSession.acquire(workspace)


def test_interrupted_verification_recovers_the_exact_published_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = PortfolioResearchSpec.default()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        application = PortfolioResearchApplication(
            workspace_id="recoverable",
            workspace=workspace,
            manifest_binding=lambda: _HASH,
            session=session,
            resolver=_Resolver(_resolved()),
        )
        planned = application.plan(spec)
        original_verify = PortfolioResearchTaskAdapter.verify_stage

        def _interrupt_after_publish(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError("simulated interruption after durable publication")

        monkeypatch.setattr(
            PortfolioResearchTaskAdapter,
            "verify_stage",
            _interrupt_after_publish,
        )
        with pytest.raises(RuntimeError, match="simulated interruption"):
            application.run(spec=spec, planned=planned)
        task = session.task_control_registry.tasks()[0]
        assert task.lifecycle is TaskLifecycle.RECOVERY_REQUIRED

        monkeypatch.setattr(PortfolioResearchTaskAdapter, "verify_stage", original_verify)
        recovered = application.recover(task_id=task.task_id)
        assert recovered.result.action == "REUSED_EXACT"
        assert session.task_control_registry.task(task.task_id).lifecycle is TaskLifecycle.SUCCEEDED


# ============================================================ Gate 8C behaviour


def test_plan_resolves_authorities_and_performs_no_numerical_work(tmp_path: Path) -> None:
    """Plan resolves authorities and performs no numerical work."""

    with _harness(tmp_path) as harness:
        preview = harness.application.plan(PortfolioResearchSpec.default()).preview
        assert harness.resolver.authority_calls == 1
        assert harness.resolver.numerical_calls == 0
        assert preview.alpha_fit_count == 0
        assert preview.optimizer_call_count == 0
        assert preview.dense_covariance_materialization_count == 0
        assert preview.sector_forecast_call_count == 0
        assert preview.publication_count == 0
        assert preview.protected_read_count == 0
        artifacts = harness.workspace / "runtime" / "artifacts"
        assert not artifacts.exists() or not any(artifacts.rglob("*.json"))


def test_plan_reports_support_coverage_work_estimate_and_legal_recovery(tmp_path: Path) -> None:
    with _harness(tmp_path) as harness:
        preview = harness.application.plan(PortfolioResearchSpec.default()).preview
        assert preview.coverage.common_session_count == 3
        assert {owner.owner_id for owner in preview.coverage.owners} == {
            "alpha_product_replay",
            "risk_return_surface",
        }
        assert preview.coverage.alpha_training_window_sessions == 1260
        assert preview.candidate_formation_count == 3
        assert preview.prefix_work_formation_count == 3
        assert preview.estimated_score_replays == 3
        assert preview.estimated_risk_surface_builds == 3
        assert preview.legal_recovery == (
            "REOPEN_EXACT_PUBLISHED_RESULT",
            "RESUME_RECOVERY_REQUIRED_TASK_WITH_SAME_PROGRAM",
        )
        assert "SCHEDULE_PHASE_SELECTION_REFUSED" in preview.refusals
        assert preview.control_catalog_hash == INSTALLED_PUBLIC_CONTROL_CATALOG.catalog_hash


def test_plan_ships_the_schedule_guard_beside_the_tranche_control(tmp_path: Path) -> None:
    with _harness(tmp_path) as harness:
        preview = harness.application.plan(PortfolioResearchSpec.create(tranches=5)).preview
        assert preview.schedule_guard.tranches == 5
        assert preview.schedule_guard.schedule_phase_selectable is False
        assert preview.schedule_guard.due_sleeve_cycle[0] == (0, 1, 2, 3, 4)


def test_an_out_of_support_window_is_refused_before_any_numerical_work(tmp_path: Path) -> None:
    """The refusal names the lane that ran out, and nothing was replayed."""

    with _harness(tmp_path) as harness:
        spec = PortfolioResearchSpec.create(study_start=date(2019, 1, 2))
        with pytest.raises(
            PortfolioResearchCompositionError,
            match="study_window_outside_support:alpha_product_replay",
        ):
            harness.application.plan(spec)
        assert harness.resolver.numerical_calls == 0


@pytest.mark.parametrize(
    "window",
    (
        {"study_start": date(2024, 1, 4)},
        {"study_end": date(2024, 1, 31)},
    ),
)
def test_a_non_session_window_bound_is_refused_by_plan(
    tmp_path: Path, window: dict[str, date]
) -> None:
    """Being between the endpoints is not membership in the exact session axis."""

    with _harness(tmp_path) as harness:
        spec = PortfolioResearchSpec.create(**window)  # type: ignore[arg-type]
        with pytest.raises(
            PortfolioResearchCompositionError,
            match="study_window_session_absent",
        ):
            harness.application.plan(spec)
        assert harness.resolver.numerical_calls == 0


def test_an_exit_rank_beyond_the_eligible_universe_is_refused_by_the_host(
    tmp_path: Path,
) -> None:
    """The spec owns the `6 * top_k` band; only the Host knows `eligible_count`."""

    with _harness(tmp_path) as harness:
        spec = PortfolioResearchSpec.create(top_k=20, exit_rank=120)
        with pytest.raises(
            PortfolioResearchCompositionError,
            match="exit_rank_exceeds_eligible_count:120>80",
        ):
            harness.application.plan(spec)
        assert harness.resolver.numerical_calls == 0


def test_run_refuses_a_preview_for_a_different_request(tmp_path: Path) -> None:
    with _harness(tmp_path) as harness:
        planned = harness.application.plan(PortfolioResearchSpec.default())
        with pytest.raises(ValueError, match="plan_spec_mismatch"):
            harness.application.run(spec=PortfolioResearchSpec.create(top_k=40), planned=planned)


def test_a_cost_change_reuses_the_execution_ledger_and_rebuilds_only_economics(
    tmp_path: Path,
) -> None:
    """A cost change reuses the execution ledger and rebuilds only economics."""

    with _harness(tmp_path) as harness:
        base = _run(harness, PortfolioResearchSpec.default())
        after_base = harness.resolver.numerical_calls
        assert after_base >= 1
        changed = _run(harness, PortfolioResearchSpec.create(cost_bps_per_side="10"))

        assert harness.resolver.numerical_calls == after_base, (
            "a descendant control re-walked a path that was already materialized"
        )

        assert changed.program_hash == base.program_hash
        left = harness.application.report(base.result_hash)
        right = harness.application.report(changed.result_hash)
        assert right.execution_ledger_hash == left.execution_ledger_hash
        assert right.economic_ledger_hash != left.economic_ledger_hash
        assert right.report_hash != left.report_hash

        base_economics = harness.application.ledger.load_economics(left.economic_ledger_hash)
        changed_economics = harness.application.ledger.load_economics(right.economic_ledger_hash)
        assert base_economics.cost_bps_per_side == "5"
        assert changed_economics.cost_bps_per_side == "10"
        assert changed_economics.execution_ledger_hash == base_economics.execution_ledger_hash


@pytest.mark.parametrize(
    ("changed", "rotates_execution"),
    [
        ({"top_k": 40}, True),
        ({"tranches": 4}, True),
        ({"weight_rule": "iv1"}, True),
        ({"exit_rank": 80}, True),
        ({"cost_bps_per_side": "2.5"}, False),
        ({"report_unit": "CALENDAR_YEAR_TABLE"}, False),
    ],
)
def test_each_control_rotates_exactly_what_the_catalog_says_it_does(
    tmp_path: Path, changed: dict[str, object], rotates_execution: bool
) -> None:
    """One parametrised proof per matrix row, read from the same catalog a UI reads."""

    with _harness(tmp_path) as harness:
        base = _run(harness, PortfolioResearchSpec.default())
        other = _run(harness, PortfolioResearchSpec.create(**changed))  # type: ignore[arg-type]
        left = harness.application.report(base.result_hash)
        right = harness.application.report(other.result_hash)
        assert (right.execution_ledger_hash != left.execution_ledger_hash) is rotates_execution
        assert (other.program_hash != base.program_hash) is rotates_execution
        for control_id in changed:
            expected = INSTALLED_PUBLIC_CONTROL_CATALOG.rebuilt_by(control_id)
            assert (expected == "EXECUTION_LEDGER") is rotates_execution


def test_unchanged_alpha_risk_and_score_identities_are_reused_exactly(
    tmp_path: Path,
) -> None:
    """Unchanged alpha risk and score identities are reused exactly."""

    with _harness(tmp_path) as harness:
        base = _run(harness, PortfolioResearchSpec.default())
        left = harness.application.report(base.result_hash)
        base_ledger = harness.application.ledger.load_execution(left.execution_ledger_hash)
        base_program = harness.application.program(base.result_hash)

        for spec in (
            PortfolioResearchSpec.create(cost_bps_per_side="20"),
            PortfolioResearchSpec.create(report_unit="SIMPLE_CUMULATIVE"),
        ):
            other = _run(harness, spec)
            program = harness.application.program(other.result_hash)
            ledger = harness.application.ledger.load_execution(
                harness.application.report(other.result_hash).execution_ledger_hash
            )
            assert program.alpha_recipe_hash == base_program.alpha_recipe_hash
            assert program.alpha_evidence_manifest_hash == (
                base_program.alpha_evidence_manifest_hash
            )
            assert program.risk_recipe_hash == base_program.risk_recipe_hash
            assert program.risk_return_surface_hash == base_program.risk_return_surface_hash
            assert ledger.consumed_score_projection_hashes == (
                base_ledger.consumed_score_projection_hashes
            )
            assert ledger.consumed_risk_projection_hashes == (
                base_ledger.consumed_risk_projection_hashes
            )
            assert ledger.executed_weights_hash == base_ledger.executed_weights_hash


def test_a_report_never_splices_two_holdings_configurations(tmp_path: Path) -> None:
    with _harness(tmp_path) as harness:
        left = harness.application.report(
            _run(harness, PortfolioResearchSpec.default()).result_hash
        )
        right = harness.application.report(
            _run(harness, PortfolioResearchSpec.create(top_k=40)).result_hash
        )
        assert left.execution_ledger_hash != right.execution_ledger_hash
        assert left.program_hash != right.program_hash
        # Each report names exactly one execution ledger, so there is no shape in
        # which rows from two configurations could appear together.
        assert left.economic_ledger_hash != right.economic_ledger_hash


def test_the_study_window_slices_the_ledger_and_keeps_its_prefix(tmp_path: Path) -> None:
    """The study window slices the ledger and keeps its prefix."""

    with _harness(tmp_path) as harness:
        sessions = tuple(_resolved().workspace.formation_sessions)
        full = _run(harness, PortfolioResearchSpec.default())
        sliced = _run(harness, PortfolioResearchSpec.create(study_start=sessions[1]))

        left = harness.application.report(full.result_hash)
        right = harness.application.report(sliced.result_hash)
        assert right.execution_ledger_hash == left.execution_ledger_hash
        assert sliced.program_hash == full.program_hash

        ledger = harness.application.ledger.load_execution(right.execution_ledger_hash)
        assert len(ledger.formation_sessions) == 3, "the continuous path is never truncated"

        assert left.window_guard.is_full_support is True
        assert left.window_guard.prefix_formation_count == 0
        assert right.window_guard.is_full_support is False
        assert right.window_guard.selected_start == sessions[1]
        assert right.window_guard.prefix_formation_count == 1
        assert right.window_guard.full_support_start == sessions[0]
        assert right.window_guard.disposition == "DESCRIPTIVE_SUBWINDOW"
        assert right.window_guard.may_select_or_promote is False
        assert right.window_guard.may_carry_claim_authority is False


def test_the_report_unit_changes_only_its_own_rows(tmp_path: Path) -> None:
    with _harness(tmp_path) as harness:
        base = _run(harness, PortfolioResearchSpec.default())
        other = _run(harness, PortfolioResearchSpec.create(report_unit="CALENDAR_YEAR_TABLE"))
        left = harness.application.report(base.result_hash)
        right = harness.application.report(other.result_hash)
        assert right.execution_ledger_hash == left.execution_ledger_hash
        assert right.economic_ledger_hash == left.economic_ledger_hash
        assert right.report_unit == "CALENDAR_YEAR_TABLE"
        assert left.report_unit == "MONTHLY_BETA_STRIPPED_LEDGER"


def test_the_benchmark_comparison_is_a_declared_descendant(tmp_path: Path) -> None:
    """The primary anchor is always present; the view only adds a comparator."""

    with _harness(tmp_path) as harness:
        base = _run(harness, PortfolioResearchSpec.default())
        report = harness.application.report(base.result_hash)
        assert report.benchmark_comparison_hash


def test_the_local_service_cli_and_agent_share_one_contract(tmp_path: Path) -> None:
    """The local service CLI and agent share one contract."""

    with _harness(tmp_path) as harness:
        service = LocalPortfolioResearchService(application=harness.application)
        assert service.controls.catalog_hash == INSTALLED_PUBLIC_CONTROL_CATALOG.catalog_hash

        parsed = build_parser().parse_args(
            [
                "--workspace",
                "synthetic",
                "run",
                "--top-k",
                "40",
                "--report-unit",
                "SIMPLE_CUMULATIVE",
            ]
        )
        assert spec_from_args(parsed) == PortfolioResearchSpec.create(
            top_k=40, report_unit="SIMPLE_CUMULATIVE"
        )

        preview = service.plan(PortfolioResearchSpec.default())
        result = service.run(PortfolioResearchSpec.default())
        readouts = service.readouts(result.result_hash)
        report = service.report(result.result_hash)

        # The Agent-facing projections that used to be asserted here were a
        # second, consumerless rendering path; the thin Agent bridge is Gate
        # 9C4's, and until it exists the service's own facts are the contract.
        assert preview.schedule_guard.schedule_phase_refusal == "SCHEDULE_PHASE_SELECTION_REFUSED"
        assert readouts.capacity == "NOT_MODELED"
        assert report.window_guard.disposition == "DESCRIPTIVE_SUBWINDOW"


def test_the_cli_cannot_reach_a_configuration_the_spec_refuses() -> None:
    """A direct caller gets the same bounds a UI would show."""

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--workspace", "w", "run", "--report-unit", "ROLLING_EXCESS"])
    with pytest.raises(ValueError, match="TOP_K_OUTSIDE_ADMITTED_RANGE"):
        spec_from_args(parser.parse_args(["--workspace", "w", "run", "--top-k", "500"]))
    with pytest.raises(ValueError, match="exported_spec_identity_invalid"):
        spec_from_args(parser.parse_args(["--workspace", "w", "run", "--spec-hash", "0" * 64]))


def test_the_service_readouts_come_from_typed_facts_and_never_promise_capacity(
    tmp_path: Path,
) -> None:
    with _harness(tmp_path) as harness:
        service = LocalPortfolioResearchService(application=harness.application)
        result = service.run(PortfolioResearchSpec.default())
        readouts = service.readouts(result.result_hash)
        report = service.report(result.result_hash)

        assert readouts.distinct_names_held == report.window_end_distinct_names
        assert readouts.effective_n == report.window_end_effective_n
        assert readouts.one_way_turnover_per_trading_session == report.mean_one_way_turnover
        assert readouts.turnover_basis == "PER_TRADING_SESSION_NOT_ANNUAL"
        assert readouts.capacity == "NOT_MODELED"
        assert readouts.liquidity_proxy_disposition.endswith("NOT_A_CAPACITY_ESTIMATE")
        assert readouts.evidence_cost_ladder_bps_per_side == ("0", "2.5", "5", "10", "20")
        assert readouts.cost_bps_per_side == "5"
        assert readouts.cost_bps_round_trip == "10"
        assert readouts.platform_one_way_cost_bps == "10"


def test_the_service_compares_two_configurations_without_selecting_one(
    tmp_path: Path,
) -> None:
    with _harness(tmp_path) as harness:
        service = LocalPortfolioResearchService(application=harness.application)
        left = service.run(PortfolioResearchSpec.default())
        right = service.run(PortfolioResearchSpec.create(report_unit="CALENDAR_YEAR_TABLE"))
        comparison = service.compare(left.result_hash, right.result_hash)
        assert comparison.disposition == "DECLARED_PATH_COMPARISON_NO_SELECTION"
        assert comparison.shares_execution_ledger is True
        assert "report_unit" in comparison.differing_controls
        with pytest.raises(ValueError, match="comparison_requires_two_configurations"):
            service.compare(left.result_hash, left.result_hash)


def test_the_service_changes_a_window_without_touching_other_controls(
    tmp_path: Path,
) -> None:
    with _harness(tmp_path) as harness:
        service = LocalPortfolioResearchService(application=harness.application)
        sessions = tuple(_resolved().workspace.formation_sessions)
        spec = PortfolioResearchSpec.create(top_k=40, cost_bps_per_side="2.5")
        resliced = service.with_study_window(spec, study_start=sessions[1], study_end=sessions[-1])
        assert resliced.holdings_spec_hash == spec.holdings_spec_hash
        assert resliced.cost == spec.cost
        assert resliced.study_start == sessions[1]
        assert service.full_support_reference(spec).common_session_count == 3


def test_the_rendered_page_is_offline_self_contained_and_truthful(tmp_path: Path) -> None:
    """The rendered page is offline self contained and truthful."""

    with _harness(tmp_path) as harness:
        service = LocalPortfolioResearchService(application=harness.application)
        result = service.run(PortfolioResearchSpec.default())
        page = service.open_html(result.result_hash)

        for forbidden in (
            "<script",
            "http://",
            "https://fonts.",
            "cdn.",
            "googleapis",
            "@import",
            "src=",
        ):
            assert forbidden not in page, f"page reaches outside itself: {forbidden}"
        assert "http" not in page, "a self-contained page contains no URL at all"

        # Units and qualifiers, not bare numbers.
        assert "bps per side" in page
        assert "one-way turnover" in page
        assert "NOT_MODELED" in page
        assert "liquidity proxy" in page
        assert "DESCRIPTIVE_SUBWINDOW" in page
        assert "SCHEDULE_PHASE_SELECTION_REFUSED" in page
        assert "not a study bound" in page
        assert "emergent from sleeve overlap" in page

        # The fixed evidence ladder, every rung.
        for rung in ("0 bps per side", "2.5 bps per side", "5 bps per side", "20 bps per side"):
            assert rung in page

        # Accessibility and layout affordances a static page can carry.
        assert 'href="#portfolio"' in page and "Skip to the Portfolio chapter" in page

        # The masthead cost contract: all three conventions, each labelled, before
        # any chapter. One unlabelled figure is the wrong number for someone.
        assert 'id="cost-contract"' in page
        assert "Per side" in page and "Round trip" in page
        assert "Platform charging convention" in page
        assert "charged against <strong>one-way turnover</strong>" in page
        # Source-bound Portfolio and Performance chapters, in that order.
        assert page.index('id="portfolio"') < page.index('id="performance"')
        assert page.index('id="cost-contract"') < page.index('id="portfolio"')
        assert "Nothing here was computed while rendering" in page
        assert "@media print" in page
        assert "@media (max-width:640px)" in page
        assert "focus-visible" in page
        assert 'role="img"' in page and "aria-label" in page
        assert 'class="scroll"' in page
        assert "<caption>" in page
        assert "scope='col'" in page


def test_the_window_end_book_names_the_positions_the_ledger_holds(tmp_path: Path) -> None:
    """The book is the sealed executed weights, not a second opinion about them."""

    with _harness(tmp_path) as harness:
        result = _run(harness, PortfolioResearchSpec.default())
        report = harness.application.report(result.result_hash)
        ledger = harness.application.ledger.load_execution(report.execution_ledger_hash)
        rows, columns = ledger.executed_weights_shape
        weights = np.frombuffer(
            harness.application.ledger.load_lane(
                category="executed-weights", content_hash=ledger.executed_weights_hash
            ),
            dtype="<f8",
        ).reshape(rows, columns)

    book = report.window_end_book
    listings = tuple(ledger.ordered_listing_ids)
    row = len(ledger.formation_sessions) - 1
    ending = {listings[index]: float(weights[row][index]) for index in range(len(listings))}
    preceding = {listings[index]: float(weights[row - 1][index]) for index in range(len(listings))}

    assert book.formation_session == ledger.formation_sessions[-1]
    assert book.change_boundary == "PRECEDING_FORMATION"
    assert book.preceding_formation_session == ledger.formation_sessions[-2]
    # Every held name is present, at the weight the ledger holds for it.
    assert book.held_count == report.window_end_distinct_names
    assert book.held_count > 0
    for position in book.positions:
        assert position.weight == ending[position.listing_id]
        assert position.preceding_weight == preceding[position.listing_id]
        assert position.weight_change == position.weight - position.preceding_weight
    # And no held name is missing from it.
    named = {position.listing_id for position in book.positions}
    assert {name for name, value in ending.items() if value > HELD_WEIGHT_EPSILON} <= named
    # Largest first, then by name: one total order, so two runs emit one table.
    ordered = [(-value.weight, value.listing_id) for value in book.positions]
    assert ordered == sorted(ordered)


def test_a_windowed_report_books_the_window_end_not_the_path_end(tmp_path: Path) -> None:
    """A window that ends early describes the book as it stood there."""

    with _harness(tmp_path) as harness:
        sessions = tuple(_resolved().workspace.formation_sessions)
        full = _run(harness, PortfolioResearchSpec.default())
        early = _run(harness, PortfolioResearchSpec.create(study_end=sessions[-2]))
        left = harness.application.report(full.result_hash)
        right = harness.application.report(early.result_hash)

    assert left.window_end_book.formation_session == sessions[-1]
    assert right.window_end_book.formation_session == sessions[-2]
    # Same path, same listing axis, different book: the window moved the answer
    # rather than the page's label for it.
    assert right.execution_ledger_hash == left.execution_ledger_hash
    assert right.window_end_book.listing_axis_hash == left.window_end_book.listing_axis_hash
    assert right.window_end_book.book_hash != left.window_end_book.book_hash


def test_a_flat_development_opening_is_unchanged(tmp_path: Path) -> None:
    """A development run that opens flat reports every position as an opening and no exits."""

    with _harness(tmp_path) as harness:
        sessions = tuple(_resolved().workspace.formation_sessions)
        result = _run(harness, PortfolioResearchSpec.create(study_end=sessions[0]))
        report = harness.application.report(result.result_hash)
        page = harness.application.ledger.load_html_by_uri(result.html_uri)
        ledger = harness.application.ledger.load_execution(report.execution_ledger_hash)
        opening = harness.application.ledger.load_opening_reference(ledger)

    book = report.window_end_book
    # The opening is flat because the sealed lane says so, not because the row
    # index is zero. That is the whole distinction this Gate turned on.
    assert not opening.any()
    assert book.formation_session == sessions[0]
    assert book.change_boundary == "FLAT_PATH_OPENING"
    assert book.preceding_formation_session is None
    assert book.exited_count == 0
    assert book.opened_count == book.held_count == len(book.positions)
    assert all(position.preceding_weight == 0.0 for position in book.positions)
    assert all(position.disposition == "OPENED" for position in book.positions)
    assert "opened flat" in page
    assert "FLAT_PATH_OPENING" in page


def test_a_book_from_another_formation_cannot_be_carried_by_the_report(
    tmp_path: Path,
) -> None:
    """The binding is load-bearing: a mismatched book is refused, not rendered."""

    with _harness(tmp_path) as harness:
        sessions = tuple(_resolved().workspace.formation_sessions)
        full = _run(harness, PortfolioResearchSpec.default())
        early = _run(harness, PortfolioResearchSpec.create(study_end=sessions[-2]))
        left = harness.application.report(full.result_hash)
        right = harness.application.report(early.result_hash)

        identity = left.model_dump(mode="json", exclude={"report_hash"})
        identity["window_end_book"] = right.window_end_book.model_dump(mode="json")
        payload = dict(identity, report_hash=canonical_hash(identity))
        with pytest.raises(ValueError, match="report_book_window_mismatch"):
            PortfolioDeclaredPathReport.model_validate(payload)


def test_a_listing_id_is_escaped_where_the_book_is_rendered(tmp_path: Path) -> None:
    """The book is the one table carrying identifiers from outside the report."""

    with _harness(tmp_path) as harness:
        result = _run(harness, PortfolioResearchSpec.default())
        report = harness.application.report(result.result_hash)
        economics = harness.application.ledger.load_economics(report.economic_ledger_hash)
        comparison = harness.application.ledger.load_comparison(report.benchmark_comparison_hash)
        hostile = report.window_end_book.positions[0].model_copy(
            update={"listing_id": '<script>alert("x")</script>'}
        )
        page = render_portfolio_research_html(
            spec=PortfolioResearchSpec.default(),
            report=report.model_copy(
                update={
                    "window_end_book": report.window_end_book.model_copy(
                        update={"positions": (hostile, *report.window_end_book.positions[1:])}
                    )
                }
            ),
            economics=economics,
            comparison=comparison,
            coverage=harness.application.plan(PortfolioResearchSpec.default()).preview.coverage,
        )

    assert "<script>alert" not in page
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in page


def test_the_page_shows_a_truthful_empty_state_rather_than_a_zero(tmp_path: Path) -> None:
    """An empty section says it is empty; it does not render a fabricated row."""

    with _harness(tmp_path) as harness:
        service = LocalPortfolioResearchService(application=harness.application)
        result = service.run(PortfolioResearchSpec.default())
        report = service.report(result.result_hash)
        economics = harness.application.ledger.load_economics(report.economic_ledger_hash)
        comparison_free = report.model_copy(update={"risk_facts": (), "window_unit_rows": ()})
        page = render_portfolio_research_html(
            spec=PortfolioResearchSpec.default(),
            report=comparison_free,
            economics=economics,
            comparison=PortfolioBenchmarkComparison.create(
                execution_ledger_hash=report.execution_ledger_hash,
                economic_ledger_hash=report.economic_ledger_hash,
                secondary_benchmark_view="anchor_only",
                primary_simple_returns=(0.0,),
                secondary_benchmark_id=None,
                secondary_simple_returns=None,
                secondary_disposition="SECONDARY_BENCHMARK_NOT_REQUESTED",
            ),
            coverage=harness.application.plan(PortfolioResearchSpec.default()).preview.coverage,
        )
        assert "No admitted Risk attribution for this path." in page
        assert "No formations fall inside the selected window." in page
        # The sharp form of the same rule: a section with nothing to show renders
        # its sentence, so no table is ever emitted with an empty body.
        assert "<tbody></tbody>" not in page


def test_the_default_gate_8b_path_is_unchanged_end_to_end(tmp_path: Path) -> None:
    """The installed default portfolio path supports planning, execution, exact reuse, reporting,
    and export end to end."""

    with _harness(tmp_path) as harness:
        service = LocalPortfolioResearchService(application=harness.application)
        spec = PortfolioResearchSpec.default()
        assert spec.is_default()

        preview = service.plan(spec)
        assert preview.exact_cache_hit is False
        published = service.run(spec)
        assert published.action == "PUBLISHED"
        assert service.plan(spec).exact_cache_hit is True
        assert service.run(spec).action == "REUSED_EXACT"

        assert service.report(published.result_hash).report_hash == published.report_hash
        assert service.open_html(published.result_hash).startswith("<meta charset=")

        exported = service.export(published.result_hash)
        assert exported == (
            "alphalattice-portfolio --workspace synthetic run "
            "--score-source-mode HISTORICAL_ARRAY_REPLAY "
            f"--spec-hash {spec.spec_hash}"
        )
        # The exported command carries no non-default flag, and replaying it
        # rebuilds the identical request.
        replayed = spec_from_args(
            build_parser().parse_args(
                ["--workspace", "synthetic", "run", "--spec-hash", spec.spec_hash]
            )
        )
        assert replayed.spec_hash == spec.spec_hash


def test_an_exported_command_round_trips_every_selected_control(tmp_path: Path) -> None:
    with _harness(tmp_path) as harness:
        service = LocalPortfolioResearchService(application=harness.application)
        sessions = tuple(_resolved().workspace.formation_sessions)
        spec = PortfolioResearchSpec.create(
            top_k=40,
            tranches=4,
            weight_rule="iv1",
            cost_bps_per_side="2.5",
            report_unit="CALENDAR_YEAR_TABLE",
            study_start=sessions[1],
        )
        result = service.run(spec)
        exported = service.export(result.result_hash)
        assert "--top-k 40" in exported
        assert "--tranches 4" in exported
        assert "--weight-rule iv1" in exported
        assert "--cost-bps-per-side 2.5" in exported
        assert "--report-unit CALENDAR_YEAR_TABLE" in exported
        assert f"--study-start {sessions[1].isoformat()}" in exported

        flags = exported.removeprefix("alphalattice-portfolio --workspace synthetic run ").split()
        replayed = spec_from_args(
            build_parser().parse_args(["--workspace", "synthetic", "run", *flags])
        )
        assert replayed.spec_hash == spec.spec_hash


def test_task_control_and_checkpoints_stay_in_the_runtime_authority(tmp_path: Path) -> None:
    """The Gate 8B store placement must survive everything Gate 8C added."""

    with _harness(tmp_path) as harness:
        service = LocalPortfolioResearchService(application=harness.application)
        service.run(PortfolioResearchSpec.default())
        runtime = harness.workspace / "runtime"
        assert harness.session.task_control_registry.database_path.parent == runtime.resolve()
        assert Path(harness.session.runtime_path).parent == runtime.resolve()
        assert not (harness.workspace / "research-task-control.duckdb").exists()
        assert not (harness.workspace / "market-data.duckdb").exists()
        for residue in (
            "audit-public-desktop-track-wrong-task-control-20260830.duckdb",
            "audit-public-desktop-track-wrong-checkpoints-20260830.sqlite",
            "audit-public-desktop-track-wrong-heartbeats-20260830.sqlite",
        ):
            assert not (harness.workspace / residue).exists()


def test_exact_reuse_performs_no_numerical_resolution(tmp_path: Path) -> None:
    """Exact reuse performs no numerical resolution."""

    with _harness(tmp_path) as harness:
        registry = harness.session.task_control_registry
        spec = PortfolioResearchSpec.default()
        first = harness.application.run(spec=spec)
        after_first = harness.resolver.numerical_calls
        assert after_first >= 1
        admitted = [task.task_id for task in registry.tasks()]

        second = harness.application.run(spec=spec)
        assert second.result.action == "REUSED_EXACT"
        assert second.result.result_hash == first.result.result_hash
        assert harness.resolver.numerical_calls == after_first, (
            "an exact reuse resolved numerical inputs it then threw away"
        )

        again = second
        assert again.result.action == "REUSED_EXACT"
        assert again.result.result_hash == first.result.result_hash
        assert again.pipeline_manifest == first.pipeline_manifest
        assert [task.task_id for task in registry.tasks()] == admitted
        pipeline = harness.application.pipeline
        assert pipeline.tasks_for_result(first.result.result_hash) == (
            first.pipeline_manifest.task_id,
        )

        index = pipeline.content.root / "index" / "by-result" / f"{first.result.result_hash}.json"
        index.unlink()
        named = harness.application.run(spec=spec)
        assert named.result.action == "REUSED_EXACT"
        assert named.pipeline_manifest.task_id not in admitted
        assert len(registry.tasks()) == len(admitted) + 1
        assert (
            harness.application.originating_task(first.result.result_hash)
            == named.pipeline_manifest.task_id
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"report_unit": "CALENDAR_YEAR_TABLE"},
        {"study_start": date(2024, 1, 3)},
    ],
)
def test_a_descendant_change_reads_the_stored_path_instead_of_rewalking(
    tmp_path: Path, changed: dict[str, object]
) -> None:
    """A descendant change reads the stored path instead of rewalking."""

    with _harness(tmp_path) as harness:
        base = harness.application.run(spec=PortfolioResearchSpec.default())
        after_base = harness.resolver.numerical_calls
        other = harness.application.run(
            spec=PortfolioResearchSpec.create(**changed)  # type: ignore[arg-type]
        )
        assert harness.resolver.numerical_calls == after_base, (
            "a descendant control re-walked a path that was already materialized"
        )
        left = harness.application.report(base.result.result_hash)
        right = harness.application.report(other.result.result_hash)
        assert right.execution_ledger_hash == left.execution_ledger_hash
        assert right.report_hash != left.report_hash


def test_the_spy_comparator_is_the_only_descendant_that_still_resolves(
    tmp_path: Path,
) -> None:
    """Only an external benchmark comparison resolves beyond the stored execution ledger."""

    with _harness(tmp_path) as harness:
        harness.application.run(spec=PortfolioResearchSpec.default())
        after_base = harness.resolver.numerical_calls
        harness.application.run(
            spec=PortfolioResearchSpec.create(secondary_benchmark_view="anchor_plus_spy")
        )
        assert harness.resolver.numerical_calls == after_base + 1


def test_plan_separates_ledger_coverage_from_owner_support(tmp_path: Path) -> None:
    """Plan separates ledger coverage from owner support."""

    with _harness(tmp_path) as harness:
        spec = PortfolioResearchSpec.default()
        before = harness.application.plan(spec).preview
        assert before.execution_ledger_coverage is None
        assert before.comparison_plan == "NO_INSTALLED_COMPARISON_OR_CV_PLAN"
        assert before.work_estimate_basis == "UPPER_BOUND_OVER_CANDIDATE_SUPPORT"

        harness.application.run(spec=spec)
        after = harness.application.plan(spec).preview
        sessions = tuple(_resolved().workspace.formation_sessions)
        assert after.execution_ledger_coverage == (
            sessions[0].isoformat(),
            sessions[-1].isoformat(),
            len(sessions),
        )
        # Coverage of the materialized path is not the same object as the
        # candidate support the owners could in principle serve.
        assert after.coverage.common_session_count == len(sessions)


def test_the_page_never_forces_the_body_to_scroll_sideways(tmp_path: Path) -> None:
    """The page never forces the body to scroll sideways."""

    with _harness(tmp_path) as harness:
        service = LocalPortfolioResearchService(application=harness.application)
        page = service.open_html(service.run(PortfolioResearchSpec.default()).result_hash)
        style = page.split("<style>", 1)[1].split("</style>", 1)[0]
        assert "overflow-x:hidden" in style
        # `nowrap` is not banned outright -- it is right for a table of short
        # labelled cells, which would otherwise break mid-word on a phone. It is
        # only ever allowed where the element sits in its own scroll container,
        # which is what stops it reaching the body.
        for block in style.split("}"):
            if "white-space:nowrap" not in block:
                continue
            selector = block.rsplit("{", 1)[0].strip().splitlines()[-1]
            assert selector.startswith("table."), selector
        # Every table is inside a container that scrolls on its own.
        assert page.count('<div class="scroll"><table') == page.count("<table")


# ================================================== Gate 8C remediation findings


def test_the_beta_stripped_unit_uses_the_backtesting_beta_owner(tmp_path: Path) -> None:
    """The beta stripped unit uses the backtesting beta owner."""

    with _harness(tmp_path) as harness:
        result = harness.application.run(spec=PortfolioResearchSpec.default()).result
        report = harness.application.report(result.result_hash)
        economics = harness.application.ledger.load_economics(report.economic_ledger_hash)
        ledger = harness.application.ledger.load_execution(report.execution_ledger_hash)

        net = np.asarray(economics.net_simple_returns, dtype=np.float64)
        anchor = np.asarray(ledger.anchor_simple_returns, dtype=np.float64)

        # 1. The stored beta is exactly the Backtesting owner's.
        expected = active_path_metrics(portfolio_simple=net, benchmark_simple=anchor)
        assert economics.benchmark_beta == expected.beta
        assert (
            economics.benchmark_beta_disposition
            == "BACKTESTING_ACTIVE_PATH_BETA_NET_VS_ANCHOR_FULL_PATH"
        )

        # 2. And it is the same expression the economic metric set uses.
        manual = float(np.cov(net, anchor, ddof=1)[0, 1] / np.var(anchor, ddof=1))
        assert economics.benchmark_beta == pytest.approx(manual, rel=1e-12)

        # 3. The rows are the stripped series, not the excess series. On this
        #    path the two differ, which is what makes the check load-bearing.
        stripped = project_report_unit_rows(
            report_unit="MONTHLY_BETA_STRIPPED_LEDGER",
            sessions=tuple(ledger.formation_sessions),
            net_simple_returns=net,
            anchor_simple_returns=anchor,
            benchmark_beta=economics.benchmark_beta,
        )
        excess = project_report_unit_rows(
            report_unit="MONTHLY_BETA_STRIPPED_LEDGER",
            sessions=tuple(ledger.formation_sessions),
            net_simple_returns=net,
            anchor_simple_returns=anchor,
            benchmark_beta=1.0,
        )
        assert report.window_unit_rows == stripped
        assert stripped != excess, "beta is 1.0 here, so this test proves nothing"


def test_beta_is_estimated_over_the_path_and_a_window_cannot_move_it(
    tmp_path: Path,
) -> None:
    """A descriptive window slices what is displayed; it does not fit anything."""

    with _harness(tmp_path) as harness:
        sessions = tuple(_resolved().workspace.formation_sessions)
        full = harness.application.run(spec=PortfolioResearchSpec.default()).result
        sliced = harness.application.run(
            spec=PortfolioResearchSpec.create(study_start=sessions[1])
        ).result
        left = harness.application.ledger.load_economics(
            harness.application.report(full.result_hash).economic_ledger_hash
        )
        right = harness.application.ledger.load_economics(
            harness.application.report(sliced.result_hash).economic_ledger_hash
        )
        assert right.benchmark_beta == left.benchmark_beta
        assert right.economic_ledger_hash == left.economic_ledger_hash


def test_a_degenerate_anchor_refuses_only_the_unit_that_needs_beta() -> None:
    """A degenerate anchor refuses only the unit that needs beta."""

    sessions = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 2, 1))
    net = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    flat = np.zeros(3, dtype=np.float64)
    with pytest.raises(ActiveMetricsError, match="active_benchmark"):
        active_path_metrics(portfolio_simple=net, benchmark_simple=flat)
    with pytest.raises(ValueError, match="report_unit_requires_beta"):
        project_report_unit_rows(
            report_unit="MONTHLY_BETA_STRIPPED_LEDGER",
            sessions=sessions,
            net_simple_returns=net,
            anchor_simple_returns=flat,
            benchmark_beta=None,
        )
    for unit in ("SIMPLE_CUMULATIVE", "CALENDAR_YEAR_TABLE"):
        assert project_report_unit_rows(
            report_unit=unit,  # type: ignore[arg-type]
            sessions=sessions,
            net_simple_returns=net,
            anchor_simple_returns=flat,
            benchmark_beta=None,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tradability_hash", "b" * 64),
        ("outcome_hash", "c" * 64),
    ],
)
def test_a_changed_input_authority_rotates_everything_and_reopens_nothing(
    tmp_path: Path, field: str, value: str
) -> None:
    """A changed input authority rotates everything and reopens nothing."""

    with _harness(tmp_path) as harness:
        base = harness.application.run(spec=PortfolioResearchSpec.default())
        base_program = harness.application.program(base.result.result_hash)
        base_task = harness.session.task_control_registry.tasks()[-1]

        # Same axes, same listings, same sessions -- one corrected authority.
        harness.resolver.resolved = _resolved(**{field: value})  # type: ignore[arg-type]

        moved = harness.application.run(spec=PortfolioResearchSpec.default())
        moved_program = harness.application.program(moved.result.result_hash)

        assert moved_program.formation_sessions_hash == base_program.formation_sessions_hash
        assert moved_program.ordered_listing_ids_hash == base_program.ordered_listing_ids_hash
        assert moved_program.program_hash != base_program.program_hash
        assert moved.result.execution_ledger_hash != base.result.execution_ledger_hash
        assert moved.result.result_hash != base.result.result_hash
        assert moved.result.action == "PUBLISHED", "a moved authority must not reuse"

        moved_task = harness.session.task_control_registry.tasks()[-1]
        assert moved_task.task_id != base_task.task_id
        assert (
            moved_task.input.payload["program"]["authorities_hash"]
            != (base_task.input.payload["program"]["authorities_hash"])
        )
        assert moved_task.input.payload["program"]["program_hash"] == moved_program.program_hash


def test_the_program_binds_both_input_authorities(tmp_path: Path) -> None:
    with _harness(tmp_path) as harness:
        result = harness.application.run(spec=PortfolioResearchSpec.default()).result
        program = harness.application.program(result.result_hash)
        assert program.tradability_decision_hash == "f" * 64
        assert program.execution_outcome_manifest_hash == "1" * 64


def test_task_admission_binds_the_admission_and_the_authorities(tmp_path: Path) -> None:
    """A task carries which plan authorised it, not only which request it ran."""

    with _harness(tmp_path) as harness:
        planned = harness.application.plan(PortfolioResearchSpec.default())
        harness.application.run(spec=PortfolioResearchSpec.default(), planned=planned)
        payload = harness.session.task_control_registry.tasks()[-1].input.payload
        assert payload["admission_hash"] == planned.preview.admission_hash
        assert payload["program"]["authorities_hash"] == planned.authorities.authorities_hash
        assert payload["workspace_manifest_hash"] == _HASH
        assert payload["strategy_catalog_hash"] == harness.resolver.strategy_catalog_hash
        assert payload["spec"] == PortfolioResearchSpec.default().model_dump(mode="json")
        assert payload["selected_strategy_package_id"] == TEST_PACKAGE.strategy_id
        assert payload["selected_strategy_package_hash"] == TEST_PACKAGE.package_hash
        assert payload["program"]["tradability_decision_hash"] == "f" * 64
        assert payload["program"]["execution_outcome_manifest_hash"] == "1" * 64


def test_the_admission_hash_excludes_cache_state_so_recovery_still_matches(
    tmp_path: Path,
) -> None:
    """The admission hash excludes cache state so recovery still matches."""

    with _harness(tmp_path) as harness:
        spec = PortfolioResearchSpec.default()
        before = harness.application.plan(spec).preview
        harness.application.run(
            spec=spec,
            planned=PlannedPortfolioResearch(
                preview=before,
                authorities=harness.resolver.resolve_authorities(
                    workspace=harness.workspace, spec=spec
                ),
            ),
        )
        after = harness.application.plan(spec).preview
        assert after.preview_hash != before.preview_hash, "the cache state moved"
        assert after.admission_hash == before.admission_hash, "the authorisation did not"


def test_recovery_refuses_a_task_admitted_under_different_authorities(
    tmp_path: Path,
) -> None:
    with _harness(tmp_path) as harness:
        spec = PortfolioResearchSpec.default()
        harness.application.run(spec=spec)
        task = harness.session.task_control_registry.tasks()[-1]
        harness.resolver.resolved = _resolved(tradability_hash="b" * 64)
        with pytest.raises(ValueError, match="recovery_admission_mismatch"):
            harness.application.recover(task_id=task.task_id)


def test_numerical_resolution_must_match_the_admitted_authorities(tmp_path: Path) -> None:
    """Two reads are not permission to publish two different authorities."""

    class SplitResolver(_Resolver):
        def resolve(
            self, *, workspace: Path, spec: PortfolioResearchSpec
        ) -> ResolvedPortfolioExecution:
            del workspace
            self.numerical_calls += 1
            return _resolved(
                spec.weight_rule,
                spec.secondary_benchmark_view,
                tradability_hash="a" * 64,
                outcome_hash="2" * 64,
            )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resolver = SplitResolver(_resolved(tradability_hash="f" * 64, outcome_hash="1" * 64))
    with WorkspaceApplicationSession.acquire(workspace) as session:
        application = PortfolioResearchApplication(
            workspace_id="split-authority",
            workspace=workspace,
            manifest_binding=lambda: _HASH,
            session=session,
            resolver=resolver,
        )
        planned = application.plan(PortfolioResearchSpec.default())
        with pytest.raises(
            PortfolioResearchCompositionError,
            match="numerical_resolution_authority_mismatch",
        ):
            application.run(spec=PortfolioResearchSpec.default(), planned=planned)
        # The mismatch is caught where the resolution first exists, which is
        # execution. Admission is deliberately off the numerical path now -- it
        # seals a Program from the authority receipt -- so a task record exists
        # and holds the refusal. What must not exist is a published path.
        admitted = session.task_control_registry.tasks()
        assert len(admitted) == 1
        assert admitted[0].lifecycle is not TaskLifecycle.SUCCEEDED
        assert not (workspace / "runtime" / "artifacts" / "portfolio-strategy-lab").exists()


def test_task_contract_refuses_a_program_from_different_authorities(tmp_path: Path) -> None:
    """The public function is fail-closed even when called outside the Host."""

    with _harness(tmp_path) as harness:
        spec = PortfolioResearchSpec.default()
        authorities = harness.resolver.resolve_authorities(workspace=harness.workspace, spec=spec)
        program = harness.application.compiler.compile(
            spec=spec,
            authorities=authorities,
            resolved=harness.resolver.resolved,
        )
        unbound = program.model_dump(mode="json", exclude={"program_hash", "authorities_hash"})
        with pytest.raises(ValueError, match="program_authorities_absent"):
            type(program).create(**unbound)
        with pytest.raises(ValueError, match="task_program_authorities_mismatch"):
            portfolio_research_task_contract(
                workspace_id="synthetic",
                spec=spec,
                program=program,
                admission_hash="a" * 64,
                authorities_hash="b" * 64,
                workspace_manifest_hash=_HASH,
                strategy_catalog_hash=harness.resolver.strategy_catalog_hash,
                selected_strategy_package_id=TEST_PACKAGE.strategy_id,
            )


def test_admitted_common_watermark_cannot_be_rewritten_to_a_shorter_ledger(
    tmp_path: Path,
) -> None:
    """A RUN either honours the admitted window exactly or publishes nothing."""

    class WiderAuthorityResolver(_Resolver):
        def resolve_authorities(
            self, *, workspace: Path, spec: PortfolioResearchSpec
        ) -> ResolvedPortfolioAuthorities:
            current = super().resolve_authorities(workspace=workspace, spec=spec)
            earlier = date(2023, 12, 29)
            sessions = (earlier, *current.candidate_sessions)
            owners = tuple(
                OwnerCoverage.of(
                    owner_id=value.owner_id,
                    lane=value.lane,
                    sessions=sessions,
                    identity_hash=value.identity_hash,
                )
                for value in current.coverage.owners
            )
            return ResolvedPortfolioAuthorities(
                coverage=PortfolioSupportCoverage.create(
                    owners=owners,
                    common_watermark_start=sessions[0],
                    common_watermark_end=sessions[-1],
                    common_session_count=len(sessions),
                ),
                candidate_sessions=sessions,
                ordered_listing_ids=current.ordered_listing_ids,
                eligible_count=current.eligible_count,
                strategy_package_id=current.strategy_package_id,
                strategy_package_hash=current.strategy_package_hash,
                score_source_mode=current.score_source_mode,
                policy_identity=current.policy_identity,
                alpha_recipe_hash=current.alpha_recipe_hash,
                alpha_evidence_manifest_hash=current.alpha_evidence_manifest_hash,
                risk_recipe_hash=current.risk_recipe_hash,
                risk_return_surface_hash=current.risk_return_surface_hash,
                sector_map_hash=current.sector_map_hash,
                tradability_decision_hash=current.tradability_decision_hash,
                execution_outcome_manifest_hash=current.execution_outcome_manifest_hash,
            )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resolver = WiderAuthorityResolver(_resolved())
    with WorkspaceApplicationSession.acquire(workspace) as session:
        application = PortfolioResearchApplication(
            workspace_id="wider-watermark",
            workspace=workspace,
            manifest_binding=lambda: _HASH,
            session=session,
            resolver=resolver,
        )
        planned = application.plan(PortfolioResearchSpec.default())
        assert planned.preview.selected_study_start == "2023-12-29"
        with pytest.raises(
            PortfolioResearchCompositionError,
            match="numerical_resolution_authority_mismatch",
        ):
            application.run(spec=PortfolioResearchSpec.default(), planned=planned)
        # As above: the refused run leaves a task that did not succeed, and no
        # widened ledger.
        admitted = session.task_control_registry.tasks()
        assert len(admitted) == 1
        assert admitted[0].lifecycle is not TaskLifecycle.SUCCEEDED
        assert not (workspace / "runtime" / "artifacts" / "portfolio-strategy-lab").exists()


def test_public_path_source_identity_contains_every_direct_remediation_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Path] = {}

    def capture(components: dict[str, Path], **metadata: object) -> str:
        del metadata
        captured.update(components)
        return "9" * 64

    monkeypatch.setattr(portfolio_research_composition, "switched_source_identity", capture)
    assert portfolio_research_composition._public_path_source_hash() == "9" * 64
    assert {
        "capabilities.portfolio_backtesting.active_metrics",
        "investment.portfolio_strategy_lab.application.contracts",
        "investment.portfolio_strategy_lab.application.controls",
        "investment.portfolio_strategy_lab.policies.buffered_rank_return",
        "investment.portfolio_strategy_lab.reporting.units",
    } <= captured.keys()


def test_authority_receipt_binds_the_full_axis_and_frozen_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal endpoints and counts cannot make different axes the same authority."""

    with _harness(tmp_path) as harness:
        spec = PortfolioResearchSpec.default()
        admitted = harness.resolver.resolve_authorities(workspace=harness.workspace, spec=spec)
        alternate = replace(
            admitted,
            candidate_sessions=(date(2024, 1, 2), date(2024, 1, 4), date(2024, 2, 1)),
        )
        assert admitted.candidate_sessions != alternate.candidate_sessions
        assert admitted.authorities_hash != alternate.authorities_hash

        frozen = admitted.authorities_hash
        monkeypatch.setattr(
            portfolio_research_composition,
            "_public_path_source_hash",
            lambda: "9" * 64,
        )
        assert admitted.authorities_hash == frozen
        assert replace(admitted, public_path_source_hash="9" * 64).authorities_hash != frozen


def test_a_custom_study_end_reports_every_fact_at_that_window_end(
    tmp_path: Path,
) -> None:
    """A custom study end reports every fact at that window end."""

    with _harness(tmp_path) as harness:
        sessions = tuple(_resolved().workspace.formation_sessions)
        full = harness.application.report(
            harness.application.run(spec=PortfolioResearchSpec.default()).result.result_hash
        )
        early = harness.application.report(
            harness.application.run(
                spec=PortfolioResearchSpec.create(study_end=sessions[0])
            ).result.result_hash
        )

        assert early.window_scope == "ALL_FACTS_AT_OR_INSIDE_THE_SELECTED_WINDOW"
        assert early.window_guard.selected_end == sessions[0]
        assert full.window_guard.selected_end == sessions[-1]

        # Risk attribution is sliced to the window and ends at the window end.
        assert len(early.risk_facts) == 1
        assert early.risk_facts[-1].formation_session == sessions[0]
        assert len(full.risk_facts) == len(sessions)

        # Cap diagnostics count the window's formations, not the path's.
        assert early.aggregate_cap_binding_sessions <= full.aggregate_cap_binding_sessions
        assert early.aggregate_cap_binding_names_total <= full.aggregate_cap_binding_names_total

        # Wealth compounds over the window only.
        economics = harness.application.ledger.load_economics(early.economic_ledger_hash)
        ledger = harness.application.ledger.load_execution(early.execution_ledger_hash)
        expected = float(1.0 + economics.net_simple_returns[0])
        assert early.window_cumulative_net_wealth == pytest.approx(expected)
        assert full.window_cumulative_net_wealth == pytest.approx(economics.cumulative_net_wealth)
        assert early.window_cumulative_net_wealth != full.window_cumulative_net_wealth

        # Liquidity is read at the window end from the per-formation lane.
        assert (
            early.window_end_median_holding_adv20_dollar_volume
            == (ledger.median_holding_adv20_by_formation[0])
        )
        assert (
            full.window_end_median_holding_adv20_dollar_volume
            == (ledger.median_holding_adv20_by_formation[-1])
        )

        # Turnover averages the window.
        assert early.mean_one_way_turnover == pytest.approx(ledger.one_way_turnovers[0])

        # Holdings are the executed book at that formation, which on a warming
        # book is a different set from the path end.
        assert early.window_end_distinct_names <= full.window_end_distinct_names


def test_a_report_cannot_carry_risk_facts_from_outside_its_window(tmp_path: Path) -> None:
    """The contract refuses it, so a future caller cannot reintroduce the defect."""

    with _harness(tmp_path) as harness:
        sessions = tuple(_resolved().workspace.formation_sessions)
        report = harness.application.report(
            harness.application.run(spec=PortfolioResearchSpec.default()).result.result_hash
        )
        # Drop one attribution while the guard still claims the full window. The
        # guard itself is untouched, so this exercises the report's own axis check.
        payload = report.model_dump(mode="json")
        payload["risk_facts"] = payload["risk_facts"][:-1]
        with pytest.raises(ValueError, match="report_risk_axis_invalid"):
            PortfolioDeclaredPathReport.model_validate(payload)

        # And a valid guard for a later window, against the full-path facts, is
        # refused by session rather than by count.
        later = harness.application.report(
            harness.application.run(
                spec=PortfolioResearchSpec.create(study_start=sessions[-1])
            ).result.result_hash
        )
        shifted = report.model_dump(mode="json")
        shifted["window_guard"] = later.window_guard.model_dump(mode="json")
        with pytest.raises(ValueError, match="report_risk_outside_window"):
            PortfolioDeclaredPathReport.model_validate(shifted)


def test_plan_distinguishes_result_hit_ledger_hit_and_full_miss(tmp_path: Path) -> None:
    """Plan distinguishes result hit ledger hit and full miss."""

    with _harness(tmp_path) as harness:
        spec = PortfolioResearchSpec.default()

        first = harness.application.plan(spec).preview
        assert first.cache_state == "FULL_NUMERICAL_MISS"
        assert first.exact_cache_hit is False
        assert first.estimated_score_replays == first.candidate_formation_count
        assert first.estimated_risk_surface_builds == first.candidate_formation_count
        assert first.prefix_work_formation_count == first.candidate_formation_count
        assert first.execution_ledger_coverage is None

        harness.application.run(spec=spec)

        again = harness.application.plan(spec).preview
        assert again.cache_state == "RESULT_HIT"
        assert again.exact_cache_hit is True
        assert (again.estimated_score_replays, again.estimated_risk_surface_builds) == (0, 0)
        assert again.prefix_work_formation_count == 0
        assert again.execution_ledger_coverage is not None

        descendant = harness.application.plan(
            PortfolioResearchSpec.create(report_unit="CALENDAR_YEAR_TABLE")
        ).preview
        assert descendant.cache_state == "EXECUTION_LEDGER_HIT"
        assert descendant.exact_cache_hit is False
        assert (descendant.estimated_score_replays, descendant.estimated_risk_surface_builds) == (
            0,
            0,
        )
        assert descendant.prefix_work_formation_count == 0
        assert descendant.execution_ledger_coverage is not None

        # And the estimate is truthful: running it enters no numerical owner.
        before = harness.resolver.numerical_calls
        harness.application.run(
            spec=PortfolioResearchSpec.create(report_unit="CALENDAR_YEAR_TABLE")
        )
        assert harness.resolver.numerical_calls == before


def test_a_plan_state_that_disagrees_with_its_work_estimate_is_refused(
    tmp_path: Path,
) -> None:
    with _harness(tmp_path) as harness:
        preview = harness.application.plan(PortfolioResearchSpec.default()).preview
        payload = preview.model_dump(mode="json")
        payload["cache_state"] = "EXECUTION_LEDGER_HIT"
        with pytest.raises(ValueError, match="plan_work_estimate_inconsistent"):
            PortfolioPlanPreview.model_validate(payload)


def test_the_watermark_and_the_materialized_ledger_never_impersonate_each_other(
    tmp_path: Path,
) -> None:
    """The watermark and the materialized ledger never impersonate each other."""

    with _harness(tmp_path) as harness:
        spec = PortfolioResearchSpec.default()
        harness.application.run(spec=spec)
        preview = harness.application.plan(spec).preview

        assert preview.execution_ledger_coverage is not None
        start, end, count = preview.execution_ledger_coverage
        assert count <= preview.coverage.common_session_count
        assert start >= preview.coverage.common_watermark_start.isoformat()
        assert end <= preview.coverage.common_watermark_end.isoformat()
        # Coverage names owners; it never names the ledger.
        assert "ledger" not in {owner.owner_id for owner in preview.coverage.owners}

        payload = preview.model_dump(mode="json")
        payload["execution_ledger_coverage"] = ["1999-01-04", end, count]
        with pytest.raises(ValueError, match="plan_ledger_before_watermark"):
            PortfolioPlanPreview.model_validate(payload)
        payload["execution_ledger_coverage"] = [start, "2099-01-04", count]
        with pytest.raises(ValueError, match="plan_ledger_after_watermark"):
            PortfolioPlanPreview.model_validate(payload)
        payload["execution_ledger_coverage"] = [
            start,
            end,
            preview.coverage.common_session_count + 1,
        ]
        with pytest.raises(ValueError, match="plan_ledger_exceeds_watermark"):
            PortfolioPlanPreview.model_validate(payload)


def test_comparison_names_every_changed_control_from_durable_receipts(
    tmp_path: Path,
) -> None:
    """Comparison names every changed control from durable receipts."""

    with _harness(tmp_path) as harness:
        service = LocalPortfolioResearchService(application=harness.application)
        sessions = tuple(_resolved().workspace.formation_sessions)
        left = service.run(PortfolioResearchSpec.default())
        right = service.run(
            PortfolioResearchSpec.create(
                top_k=40,
                exit_rank=60,
                cost_bps_per_side="10",
                report_unit="CALENDAR_YEAR_TABLE",
                study_start=sessions[1],
            )
        )
        comparison = service.compare(left.result_hash, right.result_hash)
        assert set(comparison.differing_controls) == {
            "top_k",
            "exit_rank",
            "cost_bps_per_side",
            "report_unit",
            "study_start",
        }
        assert comparison.disposition == "DECLARED_PATH_COMPARISON_NO_SELECTION"
        assert comparison.shares_execution_ledger is False


def test_a_control_that_moves_nothing_visible_is_still_named(tmp_path: Path) -> None:
    """The case a field diff gets wrong: same report shape, different book."""

    with _harness(tmp_path) as harness:
        service = LocalPortfolioResearchService(application=harness.application)
        left = service.run(PortfolioResearchSpec.default())
        right = service.run(PortfolioResearchSpec.create(cost_bps_per_side="10"))
        comparison = service.compare(left.result_hash, right.result_hash)
        # Same unit, same window, same guards -- one control differs, and it is
        # named from the receipt rather than inferred from what is on the page.
        assert comparison.differing_controls == ("cost_bps_per_side",)
        assert comparison.shares_execution_ledger is True


def test_the_control_receipt_covers_every_installed_control(tmp_path: Path) -> None:
    with _harness(tmp_path) as harness:
        report = harness.application.report(
            harness.application.run(spec=PortfolioResearchSpec.default()).result.result_hash
        )
        receipt = report.control_receipt
        assert receipt.control_catalog_hash == INSTALLED_PUBLIC_CONTROL_CATALOG.catalog_hash
        assert tuple(control_id for control_id, _ in receipt.selected) == tuple(
            value.control_id for value in INSTALLED_PUBLIC_CONTROL_CATALOG.controls
        )
        assert dict(receipt.selected)["top_k"] == "35"
        assert dict(receipt.selected)["study_start"] == "FULL_SUPPORT"

        payload = receipt.model_dump(mode="json")
        payload["selected"] = payload["selected"][:-1]
        with pytest.raises(ValueError, match="control_receipt_incomplete"):
            PortfolioControlReceipt.model_validate(payload)


def test_a_historical_control_receipt_does_not_consult_the_current_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catalog evolution cannot make an intact content-addressed report unreadable."""

    with _harness(tmp_path) as harness:
        result = harness.application.run(spec=PortfolioResearchSpec.default()).result
        report = harness.application.report(result.result_hash)
        monkeypatch.setattr(
            portfolio_application_contracts,
            "INSTALLED_PUBLIC_CONTROL_CATALOG",
            object(),
        )
        assert harness.application.ledger.load_report(report.report_hash) == report


def test_a_pre_catalog_ids_receipt_keeps_its_original_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compatibility lane is proven from the old bytes, not today's catalog."""

    with _harness(tmp_path) as harness:
        report = harness.application.report(
            harness.application.run(spec=PortfolioResearchSpec.default()).result.result_hash
        )
        payload = report.control_receipt.model_dump(mode="json")
        payload.pop("catalog_control_ids")
        payload["receipt_hash"] = canonical_hash(
            {key: value for key, value in payload.items() if key != "receipt_hash"}
        )
        monkeypatch.setattr(
            portfolio_application_contracts,
            "INSTALLED_PUBLIC_CONTROL_CATALOG",
            object(),
        )
        legacy = PortfolioControlReceipt.model_validate(payload)
        assert legacy.catalog_control_ids == ()
        assert legacy.receipt_hash == payload["receipt_hash"]


# ============================================ result-consumption compatibility


@dataclass
class _WatchedLedger:
    """The real ledger, with every read counted and one child read refusable.

    Counting is how the read claims in the C/D-4 record stay honest, and the
    refusal is how "this refusal never needed the report" stops being an
    argument about call order and becomes a test.
    """

    inner: Any
    reads: Counter[str] = field(default_factory=Counter)
    trace: list[str] = field(default_factory=list)
    refuse_reports: bool = False

    def seen(self, kind: str) -> None:
        """One place to record a touch, so the ports can share the sequence."""

        self.reads[kind] += 1
        self.trace.append(kind)

    def load_result(self, result_hash: str) -> Any:
        self.seen("results")
        return self.inner.load_result(result_hash)

    def load_report(self, report_hash: str) -> Any:
        self.seen("reports")
        if self.refuse_reports:
            raise ContentAddressedStoreError(f"content_store.artifact_missing:{report_hash}")
        return self.inner.load_report(report_hash)

    def load_economics(self, economic_hash: str) -> Any:
        self.seen("economics")
        return self.inner.load_economics(economic_hash)

    def load_html_by_uri(self, uri: str) -> str:
        self.seen("html")
        return str(self.inner.load_html_by_uri(uri))


@dataclass
class _WatchedApplication:
    """`PortfolioApplicationPort` over the real application and a watched ledger."""

    inner: Any
    ledger: _WatchedLedger

    @property
    def workspace_id(self) -> str:
        return str(self.inner.workspace_id)

    def plan(self, spec: PortfolioResearchSpec) -> Any:
        return self.inner.plan(spec)

    def run(self, *, spec: PortfolioResearchSpec) -> Any:
        return self.inner.run(spec=spec)

    def report(self, result_hash: str) -> Any:
        return self.inner.report(result_hash)

    def export(self, result_hash: str) -> str:
        return str(self.inner.export(result_hash))


@dataclass
class _WatchedLineage:
    """A wired lineage port that records the call, and can be made to fail.

    Wired-and-failing is the case that separates "this refusal never needed the
    lineage" from "this service has no lineage to ask": an unwired port is
    silent whatever the order, so it cannot witness an ordering defect.
    """

    inner: Any
    watch: _WatchedLedger
    fail: bool = False

    def originating_task(self, result_hash: str) -> Any:
        self.watch.seen("lineage")
        if self.fail:
            raise RuntimeError("probe.lineage_port_was_asked")
        return self.inner.originating_task(result_hash)


@dataclass
class _WatchedPrograms:
    """The same, for the Program port."""

    inner: Any
    watch: _WatchedLedger
    fail: bool = False

    def program(self, result_hash: str) -> Any:
        self.watch.seen("program")
        if self.fail:
            raise RuntimeError("probe.program_port_was_asked")
        return self.inner.program(result_hash)


def _watched(
    harness: _Harness, *, ports: bool = False, failing_ports: bool = False
) -> tuple[LocalPortfolioResearchService, _WatchedLedger]:
    """The service over a counted ledger, optionally over counted ports too.

    `ports=False` is the bare configuration the other cases here need. The two
    port flags exist because an ordering claim about export is only as good as
    its noisiest child, and the ports are children the ledger counter cannot
    see.
    """

    ledger = _WatchedLedger(inner=harness.application.ledger)
    wired = ports or failing_ports
    service = LocalPortfolioResearchService(
        application=_WatchedApplication(inner=harness.application, ledger=ledger),
        lineage=(
            _WatchedLineage(inner=harness.application, watch=ledger, fail=failing_ports)
            if wired
            else None
        ),
        programs=(
            _WatchedPrograms(inner=harness.application, watch=ledger, fail=failing_ports)
            if wired
            else None
        ),
    )
    return service, ledger


def test_export_manifest_needs_no_lineage_or_program_port(tmp_path: Path) -> None:
    """Export manifest needs no lineage or program port."""

    with _harness(tmp_path) as harness:
        spec = PortfolioResearchSpec.default()
        result = _run(harness, spec)
        service = LocalPortfolioResearchService(application=harness.application)

        manifest = service.export_manifest(result.result_hash, spec)
        assert manifest.result_hash == result.result_hash
        assert manifest.originating_task_id is None
        assert manifest.numerical_input_assembly_hash is None
        assert manifest.authorities_hash is None
        assert manifest.spec_hash == spec.spec_hash

        # Asking for the task itself still refuses, which is the difference
        # between "export does not need one" and "nothing needs one".
        with pytest.raises(ValueError, match="task_lineage_not_wired"):
            service.originating_task(result.result_hash)


def test_result_only_refusals_come_before_any_child_read(tmp_path: Path) -> None:
    """Result only refusals come before any child read."""

    with _harness(tmp_path) as harness:
        result = _run(harness, PortfolioResearchSpec.default())
        service, ledger = _watched(harness)
        ledger.refuse_reports = True

        with pytest.raises(ValueError, match="comparison_requires_two_configurations"):
            service.compare(result.result_hash, result.result_hash)
        assert ledger.reads["reports"] == 0
        assert ledger.reads["results"] == 2

        ledger.reads.clear()
        other = PortfolioResearchSpec.create(cost_bps_per_side="10")
        assert other.spec_hash != result.spec_hash
        with pytest.raises(ValueError, match="export_spec_is_not_this_result"):
            service.export_manifest(result.result_hash, other)
        assert ledger.reads["reports"] == 0
        assert ledger.reads["results"] == 1

        # The same two refusals with every child wired and every child failing.
        # A report the service never opens is one child; a port it never asks is
        # another, and an argument is evaluated before the call it is passed to,
        # so the lineage is the one an export refusal loses to most easily.
        failing, watch = _watched(harness, failing_ports=True)
        watch.refuse_reports = True

        with pytest.raises(ValueError, match="comparison_requires_two_configurations"):
            failing.compare(result.result_hash, result.result_hash)
        assert watch.trace == ["results", "results"]

        watch.trace.clear()
        with pytest.raises(ValueError, match="export_spec_is_not_this_result"):
            failing.export_manifest(result.result_hash, other)
        assert watch.trace == ["results"]

        # And through the form the Host reaches directly, which opens no result
        # of its own because its caller already did.
        watch.trace.clear()
        with pytest.raises(ValueError, match="export_spec_is_not_this_result"):
            failing.export_manifest_from(result, other, originating_task=None)
        assert watch.trace == []


def test_export_reads_its_children_in_the_declared_order(tmp_path: Path) -> None:
    """Export reads its children in the declared order."""

    with _harness(tmp_path) as harness:
        spec = PortfolioResearchSpec.default()
        result = _run(harness, spec)
        service, ledger = _watched(harness, ports=True)

        manifest = service.export_manifest(result.result_hash, spec)
        assert manifest.result_hash == result.result_hash
        assert ledger.trace == ["results", "reports", "program", "lineage"]

        # The Host's form: the caller has already opened the result and read the
        # lineage, so only the two it has not are read here.
        ledger.trace.clear()
        from_result = service.export_manifest_from(result, spec, originating_task=None)
        assert from_result.originating_task_id is None
        assert ledger.trace == ["reports", "program"]


def test_one_comparison_opens_each_artifact_once(tmp_path: Path) -> None:
    """One comparison opens each artifact once."""

    with _harness(tmp_path) as harness:
        left = _run(harness, PortfolioResearchSpec.default())
        right = _run(harness, PortfolioResearchSpec.create(cost_bps_per_side="10"))
        service, ledger = _watched(harness)

        comparison = service.compare(left.result_hash, right.result_hash)
        assert comparison.disposition == "DECLARED_PATH_COMPARISON_NO_SELECTION"
        assert dict(ledger.reads) == {"results": 2, "reports": 2, "economics": 2}


def test_a_later_operation_re_reads_rather_than_trusting_earlier_bytes(tmp_path: Path) -> None:
    """A later operation re reads rather than trusting earlier bytes."""

    with _harness(tmp_path) as harness:
        result = _run(harness, PortfolioResearchSpec.default())
        service, ledger = _watched(harness)

        first = service.readouts(result.result_hash)
        assert first.distinct_names_held >= 0
        assert dict(ledger.reads) == {"results": 1, "reports": 1, "economics": 1}

        report_file = harness.application.ledger.root / "reports" / f"{result.report_hash}.json"
        assert report_file.is_file()
        report_file.write_bytes(b"{}")

        with pytest.raises(ContentAddressedStoreError, match="artifact_tampered"):
            service.readouts(result.result_hash)
        assert ledger.reads["reports"] == 2


def test_every_task_that_published_a_result_is_named(tmp_path: Path) -> None:
    """Every task that published a result is named."""

    from alphalattice.control.product_host.publication.portfolio_research import (
        PortfolioResearchPipelineManifest,
        PortfolioResearchPipelineStore,
    )

    store = PortfolioResearchPipelineStore(tmp_path / "artifacts")
    first, second = uuid4(), uuid4()
    moment = datetime(2026, 9, 27, 12, tzinfo=UTC)
    manifests = [
        PortfolioResearchPipelineManifest.create(
            workspace_id="w",
            task_id=task,
            task_record_hash=f"{index}" * 64,
            program_hash="b" * 64,
            result_hash="c" * 64,
            report_hash="d" * 64,
            completed_at=moment + timedelta(minutes=index),
        )
        for index, task in enumerate((first, second), start=1)
    ]
    for value in manifests:
        store.publish(value)

    assert store.find_for_result("c" * 64) == manifests[0], "the first produced it"
    assert store.find_for_task(second) == manifests[1], "the second is found by its Task"
    assert store.tasks_for_result("c" * 64) == (first, second)
    assert store.find_for_task(uuid4()) is None


def test_result_collection_keeps_readable_rows_and_names_a_bad_result_index(
    tmp_path: Path,
) -> None:
    """V633/TE12: discovery keeps its sound rows; scientific selectors stay fail-closed."""

    from alphalattice.control.product_host.publication.portfolio_research import (
        PortfolioResearchPipelineManifest,
        PortfolioResearchPipelineStore,
    )

    store = PortfolioResearchPipelineStore(tmp_path / "artifacts")
    moment = datetime(2026, 10, 4, 12, tzinfo=UTC)
    values = [
        PortfolioResearchPipelineManifest.create(
            workspace_id="w",
            task_id=uuid4(),
            task_record_hash=f"{index}" * 64,
            program_hash="b" * 64,
            result_hash=result_hash,
            report_hash="d" * 64,
            completed_at=moment + timedelta(minutes=index),
        )
        for index, result_hash in enumerate(("c" * 64, "e" * 64), start=1)
    ]
    for value in values:
        store.publish(value)
    bad_index = store._result_index("e" * 64)
    bad_index.write_text("{", encoding="utf-8")

    readable, refusals = store.manifest_collection()
    assert [value.result_hash for value in readable] == ["c" * 64]
    assert len(refusals) == 1
    assert refusals[0]["status"] == "REFUSED"
    assert refusals[0]["result_hash"] == "e" * 64
    assert refusals[0]["entry_id"] == f"result:{'e' * 64}"
    assert refusals[0]["index_file"] == bad_index.name
    # This display scan cannot weaken the existing authority used to decide what was published.
    with pytest.raises(ValueError):
        store.manifests()
    with pytest.raises(ValueError):
        store.find_for_result("e" * 64)


def test_the_shared_axis_says_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (V336): no Risk return surface where the shared lanes are read is refused
    as absent, apart from several; a frozen axis sharing nothing with the Risk surface says
    its counts, in a code the Host serves as written (V306: 0 of 466 listings in 80)."""

    from alphalattice.interface.local_application.failure_codes import safe_failure_code
    from alphalattice.investment.portfolio_strategy_lab.application import resolution

    root = tmp_path / "runtime" / "artifacts"
    root.mkdir(parents=True)
    with pytest.raises(PortfolioResearchCompositionError) as absent:
        resolution._risk_surface(root)
    assert str(absent.value) == "portfolio_application.risk_return_surface_absent"
    manifests = root / "data-operations" / "risk-returns" / "manifests"
    manifests.mkdir(parents=True)
    for name in ("a" * 64, "b" * 64):
        (manifests / f"{name}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PortfolioResearchCompositionError) as several:
        resolution._risk_surface(root)
    assert str(several.value) == "portfolio_application.risk_return_surface_not_unique"

    # One surface and one sector map, read through the owners' stores.
    epoch = SimpleNamespace(ordered_listing_ids=("R1", "R2"), universe_manifest_revision="u")
    sectors = SimpleNamespace(
        manifest_revision="u", entries=tuple(SimpleNamespace(listing_id=v) for v in ("R1", "R2"))
    )
    (tmp_path / "surface" / "manifests").mkdir(parents=True)
    (tmp_path / "surface" / "manifests" / f"{'c' * 64}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "closure" / "sector-maps").mkdir(parents=True)
    (tmp_path / "closure" / "sector-maps" / f"{'d' * 64}.json").write_text("{}", encoding="utf-8")
    session = date(2026, 7, 1)
    monkeypatch.setattr(
        resolution,
        "RiskReturnArtifactStore",
        lambda _root: SimpleNamespace(
            root=tmp_path / "surface", load_manifest=lambda _hash: SimpleNamespace(epoch=epoch)
        ),
    )
    monkeypatch.setattr(
        resolution,
        "PanelClosureArtifactStore",
        lambda _resolver: SimpleNamespace(
            root=tmp_path / "closure", load_model=lambda **_fields: sectors
        ),
    )
    monkeypatch.setattr(
        resolution,
        "CausalRiskReturnReader",
        lambda _root: SimpleNamespace(available_sessions=lambda _surface: (session,)),
    )
    frozen = SimpleNamespace(ordered_listing_ids=("P1", "P2", "P3"), formation_sessions=(session,))
    support = SimpleNamespace(
        shared_artifacts=None,
        frozen_shared_market_source=frozen,
        ordered_listing_ids=None,
        formation_sessions=None,
    )
    with pytest.raises(PortfolioResearchCompositionError) as empty:
        resolution.SharedPortfolioInputResolver()._axis(workspace=tmp_path, support=support)
    code = str(empty.value)
    assert code == (
        "portfolio_application.frozen_shared_axis_empty:"
        "sessions=1,listings=0,package_listings=3,risk_listings=2"
    )
    assert safe_failure_code(code) == code
