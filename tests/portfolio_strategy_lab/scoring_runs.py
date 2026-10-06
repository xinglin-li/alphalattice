"""One strategy-score run and one calibration run through a Local Web session.

The installed package id and the two request/run/readback helpers that the
strategy scoring, decision update and model lifecycle suites drove through each
other's modules. Each helper asserts the lifecycle it drives, because a caller
reading a stale or refused publication would otherwise measure the wrong thing.
Test support beside its owner; nothing here is product authority.
"""

from __future__ import annotations

from tests.portfolio_strategy_lab.local_web_support import _json

PACKAGE = "RETURN_G6_MU_ONLY"


def _plan_run(service, formation=None):
    payload = {"strategy_package_id": PACKAGE}
    if formation is not None:
        payload["formation_session"] = formation
    plan = _json(service, "/api/strategy-score/plan", method="POST", payload=payload)
    assert plan["status"] == "PLANNED", plan
    request = dict(plan["next_requests"]["run"])
    assert request == {
        "operation": "STRATEGY_SCORE_RUN",
        "score_plan_hash": plan["score_plan_hash"],
    }
    route = _json(service, "/api/session")["routes"][request.pop("operation")]
    sent = _json(
        service,
        route["path"],
        method=route["method"],
        payload=request,
    )
    assert sent["status"] == "ADMITTED", sent
    service.dispatcher.drain_for_tests()
    status = _json(service, f"/api/status?task_id={sent['task_id']}")
    assert status["lifecycle"] == "SUCCEEDED", status
    return plan, sent, _json(service, "/api/strategy-score")


def _calibration_run(service, score):
    plan = _json(
        service,
        "/api/strategy-calibration/plan",
        method="POST",
        payload={
            "strategy_package_id": PACKAGE,
            "score_snapshot_hash": score["score"]["snapshot_hash"],
        },
    )
    assert plan["status"] == "PLANNED", plan
    request = dict(plan["next_requests"]["run"])
    assert request == {
        "operation": "STRATEGY_CALIBRATION_RUN",
        "calibration_plan_hash": plan["calibration_plan_hash"],
    }
    route = _json(service, "/api/session")["routes"][request.pop("operation")]
    sent = _json(
        service,
        route["path"],
        method=route["method"],
        payload=request,
    )
    assert sent["status"] in {"ADMITTED", "REUSED_EXACT"}, sent
    service.dispatcher.drain_for_tests()
    result = _json(service, "/api/strategy-calibration")
    assert result["status"] == "PORTFOLIO_INPUT_PUBLISHED", result
    assert result["input"]["request_hash"] == plan["calibration_plan_hash"]
    assert result["input"]["score_snapshot_hash"] == score["score"]["snapshot_hash"]
    return plan, sent, result


__all__ = ["PACKAGE", "_calibration_run", "_plan_run"]
