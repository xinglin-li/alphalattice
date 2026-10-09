"""A held Friday reaches Monday's positions through the real, offline CLI."""

from __future__ import annotations

import gc
import json
import shutil
import socket
import sys
from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest

from alphalattice.capabilities.alpha_modeling.runtime.service import AlphaModelRuntimeService
from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.foundation.market_data_ops.sources.universe import (
    bootstrap_from_candidate_manifest_document,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.interface.local_application.cli import main
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    PORTFOLIO_PUBLIC_TASK_KIND,
    portfolio_research_task_input,
)
from tests.portfolio_strategy_lab.held_friday_provider import (
    append_friday,
    prepare_friday_parent,
    prepare_friday_sector,
)
from tests.portfolio_strategy_lab.local_web_support import _json
from tests.workspace_maintenance.local_data_provider import (
    recording_provider,
)
from u0_probe import _copy


@pytest.mark.parametrize("status", ("BLOCKED", "DEFERRED", "PROPOSAL_PUBLISHED"))
def test_prior_positions_keep_the_updates_own_next_requests(monkeypatch, status):
    """adding the previous publication's review must keep the stop's way on."""
    from alphalattice.control.product_host.composition import portfolio_research_operations as ops

    publication = SimpleNamespace(content_hash="a" * 64)
    monkeypatch.setattr(ops.PortfolioUpdatePublication, "model_validate", lambda _doc: publication)
    monkeypatch.setattr(
        ops,
        "portfolio_update_positions",
        lambda *_args: SimpleNamespace(
            weights=(1.0,), changes=None, preceding=None, basis="CONDITIONAL_ESTIMATE"
        ),
    )
    kept = {
        "network": {"operation": "NETWORK_ACCESS"},
        "resume": {"operation": "RESEARCH_UPDATE_RUN", "update_plan_hash": "b" * 64},
    }
    answer = ops._position_rows(
        {
            "status": status,
            "task_id": "fixture-update",
            "publication": {},
            "history": [],
            "listing_labels": {"fixture-listing": "Fixture"},
            "next_requests": kept,
        }
    )
    assert answer["status"] == status
    assert all(answer["next_requests"][key] == value for key, value in kept.items())
    assert answer["next_requests"]["evidence_preview"] == {
        "operation": "EVIDENCE_PREVIEW",
        **answer["review_selector"],
    }


@pytest.mark.real_evidence
def test_a_held_friday_runs_offline_on_saturday_and_a_missing_monday_names_its_need(
    tmp_path, evidence_roots, monkeypatch, capsys, record_property
):
    """real installed fits and writers; synthetic Friday preparation on a QA copy.

    The held history and Data membership are retained. One missing current Sector
    observation, like Friday's new prices, is synthetic. This checks continuation and dates,
    not the simulated provider's prices or a scientific performance claim.
    """
    root = evidence_roots.require("ls1_daily_flows")
    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    # Native HTTP clients do not necessarily use Python's socket.connect.
    # No default-provider import is possible in this entirely offline fixture.
    monkeypatch.setitem(sys.modules, "yfinance", None)
    connect = socket.socket.connect

    def loopback_only(self, address):
        host = address[0] if isinstance(address, tuple) else address
        assert host in {"127.0.0.1", "localhost", "::1"}, "fixture attempted external access"
        return connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    backup_root = tmp_path / "backups"
    monkeypatch.setenv("ALPHALATTICE_BACKUP_ROOT", str(backup_root))

    def no_fit(*_args, **_kwargs):
        raise AssertionError("the September continuation must reuse the admitted fits")

    monkeypatch.setattr(AlphaModelRuntimeService, "fit", no_fit)
    workspace = tmp_path / "workspace"
    live = None
    try:
        _copy(root, workspace)
        (workspace / "runtime/local-research-connection.json").unlink(missing_ok=True)
        package = "RETURN_G6_MU_ONLY"
        # Finish all due retained candidate retries through their real owner too.
        # A prior failed listing's next retry is distinct from Friday's coverage.
        now = datetime(2026, 9, 11, 23, 5, tzinfo=UTC)
        live = LocalPortfolioWebSession.from_workspace(workspace, clock=lambda: now)
        market = MarketDataRepository(workspace)
        readiness = market.readiness.load("us-current-index-research")
        bootstrap = bootstrap_from_candidate_manifest_document(
            readiness.active_candidate_manifest_document
        )
        symbols = tuple(row.symbol for row in bootstrap.source_manifest.candidates)
        provider = recording_provider(now=now, symbols=symbols)
        universe = market.source_admission_manifest(market_profile_id="us-current-index-research")
        append_friday(provider, market, universe, date(2026, 9, 11))
        live.data_provider = provider
        live.data_source_loader = lambda **_kwargs: bootstrap
        live.start()
        prepare_friday_parent(
            provider, market, universe, date(2026, 9, 11), now, live.session.mutation_gate
        )

        def cli(*arguments):
            code = main(
                ["--workspace", str(workspace), "--view", "full", *arguments],
                serve=lambda _: 99,
            )
            answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
            return code, answer

        # A qualified child can be current while its retry parent still lacks Friday.
        # Finish that separately named preparation before claiming no upkeep is due.
        for _preparation_round in range(3):
            prepare_friday_sector(provider, market, universe, now, live.session.mutation_gate)
            code, prepared = cli("data-update", "plan")
            assert code == 0, prepared
            if prepared["data"]["status"] == "CONFIRMATION_REQUIRED":
                confirmed = _json(
                    live,
                    "/api/data-update/confirm",
                    method="POST",
                    payload={"update_plan_hash": prepared["data"]["plan_hash"]},
                )
                assert confirmed["task_id"], confirmed
            code, sent = cli("data-update", "run", "--plan", prepared["data"]["plan_hash"])
            assert code in {0, 3}, sent
            live.dispatcher.drain_for_tests()
            code, ready = cli("data-update", "show", "--task", sent["data"]["task_id"])
            assert code == 0, (
                ready["data"].get("task_lifecycle"),
                live.session.task_control_registry.task(UUID(sent["data"]["task_id"])).failure_code,
            )
            inputs = ready["data"]["inputs"]
            assert {
                inputs[name] for name in ("data_through", "adjusted_through", "panel_through")
            } == {"2026-09-11"}, ready
            parent = market.source_admission_manifest(market_profile_id="us-current-index-research")
            due = live.operations.data_update._readiness(market).plan_feature_candidate_recheck(
                observed_at=now
            )
            if (
                due is None
                and market.manifest_raw_through(parent) == date(2026, 9, 11)
                and market.manifest_provider_adjusted_through(parent) == date(2026, 9, 11)
            ):
                break
        else:
            pytest.fail("Friday fixture's complete candidate parent did not finish preparation")
        record_property("friday_preparation_rounds", _preparation_round + 1)
        books = [
            task
            for task in live.session.task_control_registry.tasks()
            if task.task_kind == PORTFOLIO_PUBLIC_TASK_KIND
            and task.lifecycle.value == "SUCCEEDED"
            and portfolio_research_task_input(task).selected_strategy_package_id == package
        ]
        book = str(max(books, key=lambda task: task.admitted_at).task_id)
        code, controls = cli("strategy-book", "controls", "--package", package)
        if controls["data"]["activation"]["status"] == "ACTIVE":
            assert (
                _json(
                    live,
                    "/api/strategy/deactivate",
                    method="POST",
                    payload={"strategy_package_id": package},
                )["status"]
                == "DEACTIVATED"
            )
        # A person's activation is fixture setup, through its human route;
        # the agent CLI must never claim that consent on the person's behalf.
        activated = _json(
            live, "/api/strategy/activate", method="POST", payload={"task_id": book}, timeout=300.0
        )
        assert activated["status"] == "ACTIVATED", activated
        # Qualification binds the reference to its child. Finish the complete
        # parent's own Friday Sector receipt too: a Saturday candidate recheck
        # must not mistake that child's receipt for the parent's publication.
        prepare_friday_sector(provider, market, universe, now, live.session.mutation_gate)
        # Saturday sees final Friday data; source access is closed throughout.
        now = datetime(2026, 9, 12, 12, tzinfo=UTC)
        live.operations.data_update.provider = None
        code, held_data = cli("data-update", "plan")
        assert code == 0 and held_data["data"]["source_access"] == "LOCAL_ADMITTED_INPUTS", {
            "work": held_data["data"].get("work"),
            "inputs": {
                key: value
                for key, value in held_data["data"].get("inputs", {}).items()
                if key.endswith("_through") or key == "sources_checked_at"
            },
            "source_check_due": held_data["data"].get("source_check_due"),
            "candidate_retry": held_data["data"].get("candidate_data_recheck") is not None,
            "change": held_data["data"].get("change") is not None,
        }
        code, held_sent = cli("data-update", "run", "--plan", held_data["data"]["plan_hash"])
        assert code in {0, 3}, held_sent
        live.dispatcher.drain_for_tests()
        code, held_result = cli("data-update", "show", "--task", held_sent["data"]["task_id"])
        held_record = live.session.task_control_registry.task(UUID(held_sent["data"]["task_id"]))
        formation = live.operations.data_update.changes.formation_scope(
            live.operations.data_update.last_plan.before,
            live.operations.data_update.last_plan.change,
        )
        assert code == 0 and held_result["data"]["task_lifecycle"] == "SUCCEEDED", {
            "failure": held_record.failure_code,
            "formation_listings": 0 if formation is None else len(formation.listings),
            "valuation_grants": len(live.operations.data_update.last_plan.valuation_grants),
            "feature_recheck": live.operations.data_update.last_plan.request.candidate_recheck
            is not None,
        }
        code, planned = cli("research-update", "plan", "--package", package)
        assert code == 0, planned
        assert planned["data"]["target_session"] == "2026-09-11"
        assert planned["data"]["source_access"] == "LOCAL_ADMITTED_INPUTS"
        assert "no provider request" in planned["data"]["work"]
        calls = list(provider.calls)
        code, sent = cli("research-update", "run", "--plan", planned["data"]["update_plan_hash"])
        assert code in {0, 3}, sent
        live.dispatcher.drain_for_tests()
        code, result = cli("research-update", "show", "--task", sent["data"]["task_id"])
        assert code == 0, result
        assert result["data"]["status"] == "PROPOSAL_PUBLISHED", result
        proposal = result["data"]["publication"]["pending_proposal"]
        assert proposal["schedule"]["formation_session"] == "2026-09-11"
        assert proposal["schedule"]["entry_session"] == "2026-09-14"
        assert any(weight > 0 for weight in proposal["estimated_weights"])
        assert provider.calls == calls
        record_property("weekend_clock", now.isoformat())
        record_property("formation_session", proposal["schedule"]["formation_session"])
        record_property("entry_session", proposal["schedule"]["entry_session"])
        record_property("weekend_provider_calls", len(provider.calls) - len(calls))
        # Monday after provider finality is genuinely absent, not an equal bound.
        now = datetime(2026, 9, 14, 23, tzinfo=UTC)
        code, missing = cli("research-update", "plan", "--package", package)
        assert code == 0, missing
        assert missing["data"]["target_session"] == "2026-09-14"
        assert missing["data"]["source_access"] == "EXPLICIT_NETWORK_REQUIRED"
        code, sent = cli("research-update", "run", "--plan", missing["data"]["update_plan_hash"])
        assert code in {0, 3}, sent
        live.dispatcher.drain_for_tests()
        code, refusal = cli("research-update", "show", "--task", sent["data"]["task_id"])
        assert code == 2, refusal
        assert refusal["failure_code"].startswith(
            "research_update.input_source_access_not_admitted:"
        )
        assert "2026-09-14,DATA" in refusal["failure_code"]
        assert "missing market data through session 2026-09-14" in refusal["detail"]
        assert "2026-09-14" in refusal["detail"]
        access = refusal["data"]["network_access"]
        assert access["decided_by"] == "OPERATOR_OFFLINE_SWITCH"
        assert not access["network_allowed"] and not access["next_requests"]
        assert refusal["data"]["next_action"] == "RESTART_WITHOUT_OPERATOR_OFFLINE_SWITCH"
        assert "ALPHALATTICE_NETWORK_DISABLED=1" in refusal["detail"]
        assert "restart the idle Host" in refusal["detail"]
        assert "network set" not in refusal["detail"]
        assert refusal["next_requests"]["network"] == {"operation": "NETWORK_ACCESS"}
        record_property("missing_target", missing["data"]["target_session"])
        record_property("missing_refusal", refusal["failure_code"])
    finally:
        if live is not None:
            live.stop()
        del live
        gc.collect()
        assert workspace.resolve().is_relative_to(tmp_path.resolve())
        shutil.rmtree(workspace)
        if backup_root.exists():
            assert backup_root.resolve().is_relative_to(tmp_path.resolve())
            shutil.rmtree(backup_root)
