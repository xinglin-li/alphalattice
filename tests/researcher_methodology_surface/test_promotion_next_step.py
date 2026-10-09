"""A Portfolio promotion carries its upstream Task's actual way on."""

from __future__ import annotations

import json

import pytest

from alphalattice.control.product_host.composition import resource_estimates
from run_alphalattice import main
from tests.portfolio_strategy_lab.local_web_support import _json
from tests.researcher_methodology_surface.factor_web_support import (
    _session,
)


def test_portfolio_promotion_keeps_its_upstream_stop_and_exact_recovery(
    sampled_alpha_case, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
):
    """Portfolio promotion keeps its upstream stop and exact recovery."""
    case, sampled = sampled_alpha_case
    root = case[0]
    with _session(root) as live:
        candidate = sampled["report"]["result"]["candidates"][0]["candidate_id"]
        draft = _json(
            live,
            "/api/experiments/portfolio-draft",
            method="POST",
            payload={"task_id": sampled["task_id"], "candidate_id": candidate},
        )
        assert draft["status"] == "PORTFOLIO_DRAFT_READY", draft
        plan = _json(
            live,
            "/api/experiments/plan",
            method="POST",
            payload={"experiment_document": draft["document"]},
        )
        assert plan["status"] == "PLANNED", plan
        sent = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": plan["plan_hash"]},
            timeout=180,
        )
        live.dispatcher.drain_for_tests()
        portfolio_id = sent.get("task_id") or sent["publication_task_id"]
        report = _json(live, f"/api/experiments/readback?task_id={portfolio_id}")
        assert (report["status"], report["research_lane"]) == (
            "EXPERIMENT_PUBLISHED",
            "EXPLORATION",
        ), report
        before = len(live.session.task_control_registry.tasks())
        monkeypatch.setattr(
            resource_estimates.ResourceGate,
            "stage_refusal",
            lambda self, task: resource_estimates.MEMORY_INSUFFICIENT,
        )
        admitted = _json(
            live,
            "/api/experiments/promote",
            method="POST",
            payload={"task_id": portfolio_id},
        )
        assert admitted["status"] == "UPSTREAM_PROMOTION_ADMITTED", admitted
        alpha_id = admitted["alpha_promotion"]["task_id"]
        assert admitted["task_id"] == admitted["follow_task_id"] == alpha_id
        assert admitted["promoted_from_task_id"] == portfolio_id != alpha_id
        assert admitted["next_requests"]["task"] == {"operation": "STATUS", "task_id": alpha_id}
        assert admitted["alpha_promotion"]["status"] in {"ADMITTED", "REUSED_IN_FLIGHT"}
        live.dispatcher.drain_for_tests()
        stopped = _json(live, f"/api/status?task_id={alpha_id}")
        assert stopped["lifecycle"] == "BLOCKED", stopped
        assert stopped["failure_code"] == resource_estimates.MEMORY_INSUFFICIENT

        request = tmp_path / "promotion.json"
        request.write_text(
            json.dumps({"operation": "EXPERIMENT_PROMOTE", "task_id": portfolio_id}),
            encoding="utf-8",
        )
        assert (
            main(["--workspace", str(root), "--view", "full", "request", "--file", str(request)])
            == 2
        )
        answer = json.loads(capsys.readouterr().out)
        body = answer["data"]
        upstream = body["alpha_promotion"]
        assert answer["status"] == body["status"] == upstream["status"] == "BLOCKED"
        assert answer["detail"] == body["detail"] == upstream["detail"]
        assert answer["next_requests"] == body["next_requests"] == upstream["next_requests"]
        assert (
            body["failure_code"]
            == upstream["failure_code"]
            == resource_estimates.MEMORY_INSUFFICIENT
        )
        assert body["task_id"] == upstream["task_id"] == alpha_id
        assert body["promoted_from_task_id"] == portfolio_id
        assert "promote" not in answer["next_requests"] and "follow_task_id" not in body
        assert len(live.session.task_control_registry.tasks()) == before + 1

        recovery_request = answer["next_requests"]["recovery"]
        assert recovery_request == {"operation": "TASK_RECOVERY", "task_id": alpha_id}
        request.write_text(json.dumps(recovery_request), encoding="utf-8")
        assert (
            main(["--workspace", str(root), "--view", "full", "request", "--file", str(request)])
            == 2
        )
        recovery = json.loads(capsys.readouterr().out)["data"]
        assert recovery["task_id"] == alpha_id and recovery["lifecycle"] == "BLOCKED"
        assert recovery["stop"]["code"] == resource_estimates.MEMORY_INSUFFICIENT
        assert recovery["next_requests"]["recover"] == {
            "operation": "RECOVER",
            "task_id": alpha_id,
            "expected_task_hash": recovery["task_record_hash"],
        }
