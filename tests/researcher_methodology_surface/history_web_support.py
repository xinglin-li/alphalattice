"""History/comparison assertions over an existing authored research workspace."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlencode
from uuid import UUID

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.research_authoring.comparison import (
    compare_experiment_reports,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from tests.portfolio_strategy_lab.local_web_support import InstalledAgent, _json


def exercise_history(workspace: Path, portfolio_task_id: str) -> dict[str, object]:
    with LocalPortfolioWebSession.from_workspace(workspace) as live:
        original = live.operations.experiments._evidence

        def forbidden(*args, **kwargs):
            raise AssertionError("metadata discovery opened numerical descendants")

        live.operations.experiments._evidence = forbidden
        try:
            history = _json(live, "/api/research-history?history_limit=50")
        finally:
            live.operations.experiments._evidence = original
        assert history["status"] == "AVAILABLE", history
        row = next(v for v in history["entries"] if v["task_id"] == portfolio_task_id)
        assert row["book"]["experiment_task_id"] == portfolio_task_id
        bridge = InstalledAgent(live.operations)
        assert (
            json.loads(
                bridge.invoke(
                    PortfolioResearchAgentRequest(
                        operation="RESEARCH_HISTORY",
                        history_limit=50,
                    )
                )
            )
            == history
        )
        seen, cursor = [], None
        while True:
            query = {"history_limit": "1"}
            if cursor:
                query["history_cursor"] = cursor
            page = _json(live, "/api/research-history?" + urlencode(query))
            seen.extend(v["entry_id"] for v in page["entries"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
            assert (
                _json(
                    live,
                    "/api/research-history?"
                    + urlencode(
                        {
                            "history_kind": "risk.covariance-development",
                            "history_cursor": cursor,
                        }
                    ),
                )["status"]
                == "REFUSED"
            )
        assert seen == [v["entry_id"] for v in history["entries"]]
        assert _json(live, "/api/research-history?history_entry_id=unknown")["status"] == "REFUSED"
        risk = _json(live, "/api/research-history?history_kind=risk.covariance-development")
        assert risk["entries"] and all(
            v["kind"] == "risk.covariance-development" for v in risk["entries"]
        )
        assert _json(live, "/api/research-history?research_input_id=unrelated")["entries"] == []

        left = _json(live, "/api/experiments/readback?task_id=" + portfolio_task_id)
        document = deepcopy(left["document"])
        document["portfolio"]["cost_bps_per_side"] = "10"
        document["experiment"]["output_workspace"] = "managed"
        document["experiment"]["baseline_workspace"] = "managed"
        plan = _json(
            live,
            "/api/experiments/plan",
            method="POST",
            payload={
                "experiment_document": document,
            },
        )
        assert plan["status"] == "PLANNED", plan
        submitted = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={
                "experiment_plan_hash": plan["plan_hash"],
            },
        )
        right_task = submitted.get("task_id") or submitted["publication_task_id"]
        live.dispatcher.drain_for_tests(timeout=300)
        right = _json(live, "/api/experiments/readback?task_id=" + right_task)
        assert right["status"] == "EXPERIMENT_PUBLISHED", right
        assert right["position"]["weights"] == left["position"]["weights"]
        before = len(live.session.task_control_registry.tasks())
        query = urlencode({"left_task_id": portfolio_task_id, "right_task_id": right_task})
        comparison = _json(live, "/api/experiments/compare?" + query)
        assert comparison == json.loads(json.dumps(compare_experiment_reports(left, right)))
        assert comparison["disposition"] == "DECLARED_PATH_COMPARISON_NO_SELECTION"
        cost = next(
            m
            for d in comparison["dimensions"]
            for m in d["metrics"]
            if m["label"] == "platform_one_way_cost_bps"
        )
        assert cost["unit"] == "bps on platform one-way turnover"
        assert cost["left"] == left["result"]["cost_bps"]
        assert (
            json.loads(
                bridge.invoke(
                    PortfolioResearchAgentRequest(
                        operation="EXPERIMENT_COMPARE",
                        left_task_id=UUID(portfolio_task_id),
                        right_task_id=UUID(right_task),
                    )
                )
            )
            == comparison
        )
        for key in ("input_binding_hash", "formation_sessions", "ordered_listing_ids"):
            wrong = deepcopy(right)
            wrong["portfolio_source"][key] = (
                [] if isinstance(wrong["portfolio_source"][key], list) else "wrong"
            )
            with pytest.raises(ValueError, match="comparison_input_or_support_mismatch"):
                compare_experiment_reports(left, wrong)
        assert len(live.session.task_control_registry.tasks()) == before
        after = _json(live, "/api/research-history?history_limit=50")
    with LocalPortfolioWebSession.from_workspace(workspace) as live:
        assert _json(live, "/api/research-history?history_limit=50") == after
        assert _json(live, "/api/experiments/compare?" + query) == comparison
    return {"left": portfolio_task_id, "right": right_task, "entry_count": len(after["entries"])}
