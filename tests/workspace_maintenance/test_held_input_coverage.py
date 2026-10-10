"""Inclusive admitted-session coverage; refresh remains a separate operation."""

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from alphalattice.control.product_host.maintenance.data_update import (
    WorkspaceDataUpdateApplication,
)


def _plan(bounds=(2, 2, 2), ready="RESEARCH_READY"):
    days = tuple(None if value is None else date(2026, 10, value) for value in bounds)
    return SimpleNamespace(
        binding=SimpleNamespace(market_profile_id="us-current-index-research"),
        before=SimpleNamespace(
            data_through=days[0],
            adjusted_through=days[1],
            panel_through=days[2],
            readiness_status=ready,
            sources_checked_at=datetime(2026, 10, 2, 23, tzinfo=UTC),
            source_check_failed_at=None,
        ),
        request=SimpleNamespace(
            target_market_session=date(2026, 10, 2),
            request_clock=datetime(2026, 10, 3, 12, tzinfo=UTC),
            candidate_data_recheck=None,
            candidate_recheck=None,
            full_history_listing_ids=(),
        ),
        change=None,
    )


def _owner():
    owner = object.__new__(WorkspaceDataUpdateApplication)
    owner.provider = None
    return owner


@pytest.mark.parametrize(
    ("bounds", "ready", "expected"),
    [
        ((2, 2, 2), "RESEARCH_READY", True),
        ((5, 5, 5), "RESEARCH_READY", True),
        ((1, 2, 2), "RESEARCH_READY", False),
        ((2, 1, 2), "RESEARCH_READY", False),
        ((2, 2, 1), "RESEARCH_READY", False),
        ((None, 2, 2), "RESEARCH_READY", False),
        ((2, None, 2), "RESEARCH_READY", False),
        ((2, 2, 2), "FEATURE_BUILDING", False),
    ],
)
def test_all_three_held_bounds_include_the_target_only_when_research_ready(bounds, ready, expected):
    """an equal final session is held; one incomplete lane never is."""
    assert _owner().historical_inputs_cover(_plan(bounds, ready)) is expected


def test_zero_gap_refresh_planner_retains_overlap_but_does_not_decide_a_fetch():
    """the coordinator decides whether to invoke this refresh planner."""
    from alphalattice.foundation.market_data_ops.runtime.refresh import normal_refresh_plan

    day = date(2026, 10, 2)
    plan = normal_refresh_plan(latest_session=day, requested_as_of=day)
    assert plan.missing_calendar_days == 0
    assert plan.windows
    assert plan.windows[0].end == day
    assert plan.windows[0].start < day


def test_a_due_membership_check_is_named_apart_from_missing_market_data():
    plan = _plan()
    plan.before.sources_checked_at = datetime(2026, 10, 1, 23, tzinfo=UTC)
    assert _owner().network_work(plan) == {
        "MEMBERSHIP": "the membership source check for session 2026-10-02",
    }
    assert not _owner().historical_inputs_cover(plan)


@pytest.mark.parametrize("raw", (False, True))
def test_candidate_work_stays_planned_and_only_missing_source_needs_a_provider(qualified, raw):
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
    from tests.researcher_methodology_surface.real_workspace import OBSERVED_AT

    plan = _plan()
    target = date(2026, 7, 31)
    plan.request.target_market_session = target
    plan.request.request_clock = OBSERVED_AT
    plan.before.data_through = plan.before.adjusted_through = plan.before.panel_through = target
    owner = _owner()
    owner.session = SimpleNamespace(workspace=qualified)
    if raw:
        plan.request.candidate_data_recheck = object()
    else:
        from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
        from tests.portfolio_strategy_lab.held_friday_provider import prepare_friday_sector
        from tests.workspace_maintenance.local_data_provider import recording_provider

        parent = MarketDataRepository(qualified).source_admission_manifest(
            market_profile_id=plan.binding.market_profile_id
        )
        prepare_friday_sector(
            recording_provider(symbols=tuple(row.provider_symbol for row in parent.listings)),
            MarketDataRepository(qualified),
            parent,
            OBSERVED_AT,
            WorkspaceMutationGate(),
        )
        plan.request.candidate_recheck = SimpleNamespace(
            parent_manifest_revision=parent.revision_sha256
        )
    assert bool(owner.network_work(plan)) is raw
    assert not owner.historical_inputs_cover(plan)


def test_a_feature_retry_names_missing_parent_data_even_when_the_qualified_child_covers(qualified):
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository

    parent = MarketDataRepository(qualified).source_admission_manifest(
        market_profile_id="us-current-index-research"
    )
    plan = _plan()
    plan.request.candidate_recheck = SimpleNamespace(
        parent_manifest_revision=parent.revision_sha256
    )
    owner = _owner()
    owner.session = SimpleNamespace(workspace=qualified)
    assert owner.network_work(plan) == {
        "CANDIDATES": "the failed-candidate data retry through session 2026-10-02",
        "SECTOR": "the candidate Sector source check for session 2026-10-02",
    }
    assert owner.source_access(plan) == "EXPLICIT_NETWORK_REQUIRED"
    assert not owner.historical_inputs_cover(plan)


def test_a_due_candidate_sector_check_names_its_need_even_when_all_data_is_held(qualified):
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository

    parent = MarketDataRepository(qualified).source_admission_manifest(
        market_profile_id="us-current-index-research"
    )
    plan = _plan()
    target = date(2026, 7, 31)
    plan.request.target_market_session = target
    plan.before.data_through = plan.before.adjusted_through = plan.before.panel_through = target
    plan.request.candidate_recheck = SimpleNamespace(
        parent_manifest_revision=parent.revision_sha256
    )
    owner = _owner()
    owner.session = SimpleNamespace(workspace=qualified)
    assert owner.network_work(plan) == {
        "SECTOR": "the candidate Sector source check for session 2026-07-31"
    }
    assert owner.source_access(plan) == "EXPLICIT_NETWORK_REQUIRED"


@pytest.mark.parametrize("reference_held", (False, True))
def test_panel_completion_needs_a_provider_only_when_its_market_reference_is_missing(
    qualified,
    reference_held,
):
    plan = _plan((2, 2, 1), "FEATURE_BUILDING")
    target = date(2026, 7, 31) if reference_held else date(2026, 8, 3)
    plan.request.target_market_session = target
    plan.before.data_through = plan.before.adjusted_through = target
    plan.before.panel_through = date(2026, 7, 30)
    owner = _owner()
    owner.session = SimpleNamespace(workspace=qualified)
    work = owner.network_work(plan)
    assert work == (
        {}
        if reference_held
        else {"REFERENCE": "the SPY market reference through session 2026-08-03"}
    )


@pytest.mark.parametrize("due,missing", ((False, False), (True, False), (False, True)))
def test_research_plan_names_the_same_source_need_its_stage_refuses(due, missing, monkeypatch):
    from alphalattice.control.product_host.composition.decision_advancement import (
        STAGES,
        DecisionAdvancementApplication,
    )
    from alphalattice.control.product_host.composition.plain_refusals import explain
    from alphalattice.control.task_control.runner import (
        StageDisposition,
        StageExecutionResult,
    )

    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    data_plan = _plan((1, 1, 1) if missing else (2, 2, 2))
    if due:
        data_plan.before.sources_checked_at = datetime(2026, 10, 1, 23, tzinfo=UTC)
    plan = SimpleNamespace(
        checkpoint_hash="a" * 64,
        bindings=(),
        target=date(2026, 10, 2),
        content_hash="b" * 64,
        package_id="fixture",
        decision_sessions=(date(2026, 10, 2),),
        score_sessions=(),
        data_plan=data_plan,
    )
    owner = object.__new__(DecisionAdvancementApplication)
    checkpoint = SimpleNamespace(
        epoch_end=date(2026, 12, 31), recipe=SimpleNamespace(components=[])
    )
    owner.updates = SimpleNamespace(
        store=SimpleNamespace(load_decision_checkpoint=lambda _hash: checkpoint)
    )
    owner.data = _owner()
    planned = owner._plan_body(plan, False)
    assert planned["source_access"] == (
        "EXPLICIT_NETWORK_REQUIRED" if due or missing else "LOCAL_ADMITTED_INPUTS"
    )
    if not (due or missing):
        assert "no provider request" in planned["work"]
        return
    needs = owner.data.network_work(data_plan)
    assert all(need in planned["work"] for need in needs.values())
    owner.data.execute_step = lambda *_args, **_kwargs: StageExecutionResult(
        StageDisposition.BLOCKED, failure_code="workspace_data_update.source_access_not_admitted"
    )
    with pytest.raises(ValueError) as failed:
        owner._execute(plan, STAGES[1], SimpleNamespace())
    code = str(failed.value)
    words = explain(code)
    assert all(need in words["detail"] for need in needs.values())
    assert words["network_access"]["decided_by"] == "OPERATOR_OFFLINE_SWITCH"
    assert not words["network_access"]["network_allowed"]
    assert not words["network_access"]["next_requests"]
    assert words["next_action"] == "STOP_AND_TELL_PERSON_NETWORK_IS_HELD_OFFLINE"
    assert "ALPHALATTICE_NETWORK_DISABLED=1" in words["detail"]
    assert "restart" not in words["detail"]
    assert "network set" not in words["detail"]
    assert words["next_requests"] == {"network": {"operation": "NETWORK_ACCESS"}}
    from alphalattice.control.task_control.contracts import FAILURE_CODE_MAX_LENGTH
    from alphalattice.interface.local_application.failure_codes import safe_failure_code

    assert len(code) <= FAILURE_CODE_MAX_LENGTH
    assert safe_failure_code(code) == code


@pytest.mark.parametrize("provider_kind", ("default", "yfinance", "fixture"))
@pytest.mark.parametrize("missing", (False, True))
def test_plan_label_and_admission_share_the_actual_provider_boundary(
    tmp_path,
    monkeypatch,
    provider_kind,
    missing,
):
    from alphalattice.foundation.market_data_ops.sources.providers import YFinanceMarketDataProvider

    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    owner = _owner()
    owner.session = SimpleNamespace(workspace=tmp_path)
    owner.provider = {
        "default": None,
        "yfinance": YFinanceMarketDataProvider(tmp_path / "provider-cache"),
        "fixture": object(),
    }[provider_kind]
    plan = _plan((1, 1, 1) if missing else (2, 2, 2))
    assert owner.source_access(plan) == (
        "LOCAL_ADMITTED_INPUTS"
        if not missing
        else "HOST_SUPPLIED_PROVIDER"
        if provider_kind == "fixture"
        else "EXPLICIT_NETWORK_REQUIRED"
    )
    assert owner._source_access_admitted() is (provider_kind == "fixture")
