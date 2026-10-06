"""Shared real first-use workspace; source adapters are deterministic and offline."""

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    publish_research_workspace_manifest,
)
from alphalattice.control.product_host.maintenance.data_update import bind_existing_data_workspace
from alphalattice.control.product_host.research_authoring.factor_inputs import bind_factor_inputs
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.foundation.causal_outcomes.execution.methods import ONE_SESSION_RECIPE_ID
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from tests.portfolio_strategy_lab.local_web_support import _json
from tests.researcher_methodology_surface.real_workspace import (
    AS_OF,
    HISTORY_START,
    OBSERVED_AT,
    SeededWalkProvider,
    _source_loader_for,
    build_real_risk_workspace,
    publish_causal_outcomes,
)


@pytest.fixture(scope="session")
def fresh_prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("first-use")
    symbols = tuple(f"F{i:03d}" for i in range(120))
    schedule = materialize_calendar_schedule(
        ("XNAS",), start=HISTORY_START, end=AS_OF, as_of_timestamp=OBSERVED_AT
    )
    provider = SeededWalkProvider(symbols, tuple(v["session_date"] for v in schedule.to_pylist()))
    live = LocalPortfolioWebSession.from_workspace(root, clock=lambda: OBSERVED_AT)
    live.data_provider, live.data_source_loader = provider, _source_loader_for(symbols)
    with live:
        # Bound execution width for the four-worker run, not its data or checks.
        budget = live.operations.set_cpu_budget("2", chosen_by="EXTERNAL_AUTOMATION")
        assert budget["status"] == "CPU_BUDGET" and budget["cpu_budget"] == 2
        session, app = live.session, live.operations.preparation
        verify = app.verify_stage
        bound_progress = []

        def interrupt_after_publication(**kwargs):
            value = verify(**kwargs)
            if kwargs["work_item"].stage_id == "prepare_features":
                # The Feature owners' finer work progress, kept beside the Task: at this
                # moment it names the current execution and stage, so the page may show it.
                bound_progress.append(app.readback()["work_progress"])
            if kwargs["work_item"].stage_id == "publish_inputs":
                raise ConnectionError("simulated process interruption after input publication")
            return value

        app.verify_stage = interrupt_after_publication
        plan = _json(live, "/api/workspace/preparation/plan", method="POST", payload={})
        assert not (root / "market-data.duckdb").exists()
        with pytest.raises(ValueError, match="human_confirmation_required"):
            app.confirm(plan["plan_hash"], caller="INSTALLED_AGENT")
        admitted = _json(
            live,
            "/api/workspace/preparation/confirm",
            method="POST",
            payload={"preparation_plan_hash": plan["plan_hash"]},
        )
        from uuid import UUID

        task_id = UUID(admitted["task_id"])
        live.dispatcher.drain_for_tests(timeout=900)
        record = session.task_control_registry.task(task_id)
        assert record.lifecycle is TaskLifecycle.RECOVERY_REQUIRED, (
            record.failure_code,
            live.dispatcher.failure(task_id),
            app.readback(),
        )
        input_hash = record.input.input_hash
        (bound,) = bound_progress
        assert bound["availability"] == "BOUND" and bound["stage"] == "prepare_features"
        assert bound["execution_id"] == str(record.latest_execution_id)
        assert bound["stage_id"] and bound["status"] in {"RUNNING", "SUCCEEDED"}
        assert 0 <= bound["completed_units"] <= bound["total_units"]
        assert bound["unit_name"] and bound["heartbeat_sequence"] >= 1
        # Interrupted at publish_inputs: the kept observation is an earlier stage's and is said so.
        retained = app.readback()["work_progress"]
        assert retained["availability"] == "NOT_CURRENT" and retained["stage"] == "prepare_features"
        assert retained["stage_id"] == bound["stage_id"]
        publications = {
            p.name: p.read_bytes() for p in (root / "research-inputs/publications").glob("*.json")
        }

    # Restart through the real composition; no replay of source acquisition is allowed.
    class NoSourceCalls:
        def __getattribute__(self, name):
            raise AssertionError(f"source accessed during input readback recovery: {name}")

    reopened = LocalPortfolioWebSession.from_workspace(root, clock=lambda: OBSERVED_AT)
    reopened.data_provider = NoSourceCalls()
    with reopened as live:
        session, app = live.session, live.operations.preparation
        assert live.resumed_task_ids == (task_id,)
        live.dispatcher.drain_for_tests(timeout=120)
        record = session.task_control_registry.task(task_id)
        assert record.lifecycle is TaskLifecycle.SUCCEEDED, (
            record.failure_code,
            live.dispatcher.failure(task_id),
            app.readback(),
        )
        assert record.input.input_hash == input_hash
        assert {
            p.name: p.read_bytes() for p in (root / "research-inputs/publications").glob("*.json")
        } == publications
        assert app.confirm(plan["plan_hash"], caller="HUMAN").task_id == task_id
        count = len(session.task_control_registry.tasks())
        assert (
            _json(
                live,
                "/api/workspace/preparation/confirm",
                method="POST",
                payload={"preparation_plan_hash": plan["plan_hash"]},
            )["status"]
            == "REUSED_EXACT"
        )
        assert len(session.task_control_registry.tasks()) == count
        assert _json(live, "/api/experiments/controls")["status"] == "READY"
        storage = _json(live, "/api/workspace/storage")
        assert storage["budget"]["active_listing_count"] == 120
        assert storage["managed_bytes"] < storage["logical_bytes"]
        assert storage["capacity_status"] == "WITHIN_BUDGET"
        # A first use composes its Panel once, for the membership its qualification binds
        # (V311), so the plan has nothing to release: no chunk lies outside the published one.
        assert (
            _json(live, "/api/workspace/storage/plan", method="POST", payload={})["targets"] == {}
        )
        assert len(session.task_control_registry.tasks()) >= 1
        assert app.plan()["status"] == "ALREADY_PREPARED"
        view = app.readback()
    return root, task_id, view


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    """Consumer input from verified product-written data, not another cold-build test.

    The fresh fixture above retains every initialization/recovery assertion.
    This path only supplies a starting workspace via the supported local binding
    operations. Each process gets an independent writable database copy.
    """
    data = build_real_risk_workspace(
        tmp_path_factory.mktemp("prepared-input"),
        symbols=tuple(f"F{i:03d}" for i in range(120)),
        sector_size=6,
    )
    root = data.workspace
    publish_research_workspace_manifest(root, ResearchWorkspaceManifest.research_only("prepared"))
    bind_existing_data_workspace(root)
    outcome, _ref = publish_causal_outcomes(data, at=OBSERVED_AT, recipe_id=ONE_SESSION_RECIPE_ID)
    bind_factor_inputs(
        workspace=root, source=root, outcome_snapshot_hash=outcome, include_alpha_handoff=True
    )
    return root
