"""A bounded Risk/report journey over an already computed Portfolio QA workspace."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlencode

import pytest
import yaml

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.research_authoring.risk_reports import RiskReportLinks
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from tests.portfolio_strategy_lab.local_web_support import InstalledAgent, _json


def exercise_risk_report(workspace: Path, portfolio_task_id: str) -> dict[str, object]:
    with LocalPortfolioWebSession.from_workspace(workspace) as live:
        rows = _json(live, "/api/experiments")["experiments"]
        parent = next(r for r in rows if r["task_id"] == portfolio_task_id)
        portfolio = _json(live, "/api/experiments/readback?task_id=" + portfolio_task_id)
        original_export = _json(live, "/api/experiments/export?task_id=" + portfolio_task_id)
        selection = {
            "research_input_id": parent["input_id"],
            "input_binding_hash": parent["input_binding_hash"],
        }
        controls = _json(
            live,
            "/api/experiments/controls?"
            + urlencode(
                {
                    **selection,
                    "experiment_kind": "risk.covariance-development",
                }
            ),
        )
        assert controls["status"] == "READY", controls
        document = controls["template"]
        day = portfolio["position"]["session"]
        document["experiment"]["sessions"].update(start=day, end=day)
        document["experiment"]["sessions"]["as_of"] = deepcopy(
            portfolio["document"]["experiment"]["sessions"]["as_of"]
        )
        invalid = deepcopy(document)
        invalid["risk"]["estimator"]["parameters"]["ewma_decay"] = 0.123
        count = len(live.session.task_control_registry.tasks())
        assert (
            _json(
                live,
                "/api/experiments/plan",
                method="POST",
                payload={
                    **selection,
                    "experiment_document": invalid,
                },
            )["status"]
            == "REFUSED"
        )
        assert len(live.session.task_control_registry.tasks()) == count
        plan = _json(
            live,
            "/api/experiments/plan",
            method="POST",
            payload={
                **selection,
                "experiment_document": document,
            },
        )
        assert plan["status"] == "PLANNED", plan
        assert plan["numerical_call_count"] == 0
        assert (
            _json(
                live,
                "/api/experiments/plan",
                method="POST",
                payload={
                    **selection,
                    "experiment_yaml": yaml.safe_dump(document),
                },
            )
            == plan
        )
        bridge = InstalledAgent(live.operations)
        assert (
            json.loads(
                bridge.invoke(
                    PortfolioResearchAgentRequest(
                        operation="EXPERIMENT_PLAN",
                        **selection,
                        experiment_document=document,
                    )
                )
            )
            == plan
        )
        submitted = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": plan["plan_hash"]},
        )
        task_id = submitted.get("task_id") or submitted["publication_task_id"]
        live.dispatcher.drain_for_tests(timeout=300)
        risk = _json(live, "/api/experiments/readback?task_id=" + task_id)
        assert risk["status"] == "EXPERIMENT_PUBLISHED", risk
        assert len(risk["result"]["evaluations"]) == 1
        assert risk["execution_numerical_call_count"] == 1
        assert risk["evidence"]["numerical_call_count"] == 0
        exported = _json(live, "/api/experiments/export?task_id=" + task_id)
        assert _facts(json.loads(exported["json"])) == _facts(risk)
        assert "Risk development diagnostics" in exported["html"]
        assert (
            _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )["status"]
            == "REUSED_EXACT"
        )
        link_request = {"task_id": portfolio_task_id, "risk_task_id": task_id}
        linked = _json(live, "/api/experiments/risk-link", method="POST", payload=link_request)
        assert linked["status"] in {"RISK_REPORT_LINKED", "REUSED_EXACT"}, linked
        assert (
            _json(live, "/api/experiments/risk-link", method="POST", payload=link_request)["status"]
            == "REUSED_EXACT"
        )
        assert (
            _json(
                live,
                "/api/experiments/risk-link",
                method="POST",
                payload={
                    **link_request,
                    "risk_task_id": portfolio_task_id,
                },
            )["status"]
            == "REFUSED"
        )
        wrong = deepcopy(risk)
        wrong["input_binding_hash"] = "0" * 64
        with pytest.raises(ValueError, match="input_or_axis_mismatch"):
            RiskReportLinks._link(portfolio, wrong, "HUMAN")
        export_request = linked["export_request"]
        assert export_request["risk_report_hash"] == linked["link"]["link_hash"]
        query = urlencode({k: v for k, v in export_request.items() if k != "operation"})
        combined = _json(live, "/api/experiments/risk-export?" + query)
        assert _facts(combined["portfolio"]) == _facts(portfolio)
        assert _facts(combined["risk"]) == _facts(risk)
        assert (
            json.loads(bridge.invoke(PortfolioResearchAgentRequest.model_validate(export_request)))
            == combined
        )
        assert (
            _json(live, "/api/experiments/export?task_id=" + portfolio_task_id) == original_export
        )
        assert "Original" in combined["html"] or "original" in combined["html"]
    with LocalPortfolioWebSession.from_workspace(workspace) as live:
        again = _json(live, "/api/experiments/readback?task_id=" + task_id)
        assert _facts(again) == _facts(risk)
        assert _json(live, "/api/experiments/risk-export?" + query) == combined
    return {
        "task_id": task_id,
        "link_hash": linked["link"]["link_hash"],
        "surface_hash": risk["risk_surface"]["surface_hash"],
    }


def exercise_risk_retry(workspace: Path, portfolio_task_id: str) -> dict[str, object]:
    """requirement: a Risk Task BLOCKED on a corrupt checkpoint has an explicit retry.

    Twenty-two formations, so the writer seals one 21-formation chunk and its
    checkpoint before the second chunk; the second chunk's publication is
    interrupted (a lost process), the checkpoint is corrupted, and the recovery
    attempt stops BLOCKED by name. A plain RUN of the same declaration answers
    that Task and the RECOVER it would take, never a silent retry; RECOVER
    without the repair reopens the Task and it stops again, by name, with no
    estimate; after the checkpoint is restored, RECOVER with the confirmed
    version reopens the same Task, which computes only the one unsealed
    formation and publishes once; a stale confirmation is refused.
    """

    import glob
    import time
    from unittest.mock import patch
    from uuid import UUID

    from alphalattice.investment.risk_research.surfaces.artifacts import RiskArtifactStore

    def chunk_files() -> int:
        pattern = str(workspace / "research-experiments/*/risk-research/covariance/chunks/*.bin")
        return len(glob.glob(pattern))

    with LocalPortfolioWebSession.from_workspace(workspace) as live:
        rows = _json(live, "/api/experiments")["experiments"]
        parent = next(r for r in rows if r["task_id"] == portfolio_task_id)
        portfolio = _json(live, "/api/experiments/readback?task_id=" + portfolio_task_id)
        selection = {
            "research_input_id": parent["input_id"],
            "input_binding_hash": parent["input_binding_hash"],
        }
        controls = _json(
            live,
            "/api/experiments/controls?"
            + urlencode({**selection, "experiment_kind": "risk.covariance-development"}),
        )
        document = controls["template"]
        sessions = [row["session"] for row in portfolio["series"]]
        day = portfolio["position"]["session"]
        start = sessions[sessions.index(day) - 21]
        document["experiment"]["sessions"].update(start=start, end=day)
        document["experiment"]["sessions"]["as_of"] = deepcopy(
            portfolio["document"]["experiment"]["sessions"]["as_of"]
        )
        plan = _json(
            live,
            "/api/experiments/plan",
            method="POST",
            payload={**selection, "experiment_document": document},
        )
        assert plan["status"] == "PLANNED", plan
        assert plan["execution_preview"]["expected_numerical_calls"] == 22
        original_publish = RiskArtifactStore.publish_covariance_chunk
        published: list[int] = []

        def interrupted(self, **kwargs):  # type: ignore[no-untyped-def]
            published.append(len(kwargs["formation_sessions"]))
            if len(published) == 2:
                raise RuntimeError("simulated lost process after the first checkpoint")
            return original_publish(self, **kwargs)

        with patch.object(RiskArtifactStore, "publish_covariance_chunk", interrupted):
            submitted = _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )
            task_id = submitted["task_id"]
            live.dispatcher.drain_for_tests(timeout=600)
        assert published == [21, 1]
        status = _json(live, "/api/status?task_id=" + task_id)
        assert status["lifecycle"] == "RECOVERY_REQUIRED", status
        pattern = str(
            workspace / "research-experiments/*/risk-research/development/build-checkpoints/*.json"
        )
        # The one-formation report journey left its own checkpoint; this is the
        # interrupted 22-formation build's, sealed after its first chunk.
        checkpoint = next(
            path
            for path in map(Path, glob.glob(pattern))
            if json.loads(path.read_bytes())["expected_formation_count"] == 22
        )
        saved = checkpoint.read_bytes()
        tampered = json.loads(saved)
        assert len(tampered["evaluations"]) == 21
        tampered["chunks"][0]["matrix_hashes"][0] = "0" * 64
        checkpoint.write_text(json.dumps(tampered), encoding="utf-8")
        view = _json(live, "/api/tasks/recovery?task_id=" + task_id)
        _json(
            live,
            "/api/recover",
            method="POST",
            payload={"task_id": task_id, "expected_task_hash": view["task_record_hash"]},
        )
        live.dispatcher.drain_for_tests(timeout=600)
        status = _json(live, "/api/status?task_id=" + task_id)
        assert status["lifecycle"] == "BLOCKED", status
        assert status["latest_failure_code"] == "risk_research.covariance_chunk_identity_invalid"
        chunks = chunk_files()
        # A plain RUN never reopens it; it names the Task and the explicit step.
        again = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": plan["plan_hash"]},
        )
        assert again["status"] == "BLOCKED" and again["task_id"] == task_id
        recover = again["next_requests"]["recover"]
        assert recover.pop("operation") == "RECOVER" and recover["task_id"] == task_id
        view = _json(live, "/api/tasks/recovery?task_id=" + task_id)
        action = next(a for a in view["actions"] if a["action"] == "RECOVER")
        assert action["available"] and "artifact" in action["reason"]
        assert recover["expected_task_hash"] == view["task_record_hash"]
        unconfirmed = _json(live, "/api/recover", method="POST", payload={"task_id": task_id})
        assert unconfirmed["failure_code"] == "local_application.expected_task_hash_required"
        # Unrepaired: reopened, stops again by name, nothing estimated.
        _json(live, "/api/recover", method="POST", payload=recover)
        live.dispatcher.drain_for_tests(timeout=600)
        status = _json(live, "/api/status?task_id=" + task_id)
        assert status["lifecycle"] == "BLOCKED"
        assert status["latest_failure_code"] == "risk_research.covariance_chunk_identity_invalid"
        assert chunk_files() == chunks
        # Restored: a stale confirmation is refused; the current one reopens the
        # same Task, which computes the one unsealed formation and publishes once.
        checkpoint.write_bytes(saved)
        stale = _json(live, "/api/recover", method="POST", payload=recover)
        assert stale["failure_code"] == "local_application.confirmation_stale"
        view = _json(live, "/api/tasks/recovery?task_id=" + task_id)
        # Two confirmations of the same BLOCKED version interleave: while the
        # first is re-checking the plan, the second is written first. The first
        # then finds the version it confirmed gone and applies nothing -- the
        # Task the competitor reopened is not reset to recovery a second time,
        # not started twice, and keeps the competitor's version.
        from alphalattice.control.product_host.composition.research_experiments import (
            ResearchExperimentApplication,
        )

        registry = live.session.task_control_registry
        confirmed = view["task_record_hash"]
        original_current = ResearchExperimentApplication._current
        competitor: list[str] = []

        def racing_current(self, plan, *, task_id=None):  # type: ignore[no-untyped-def]
            answer = original_current(self, plan, task_id=task_id)
            if not competitor:
                won = registry.mark_recovery_required(
                    task_id=UUID(str(task_id)),
                    failure_code="risk_research.covariance_chunk_identity_invalid",
                    observed_at=live.dispatcher.clock(),
                    allow_blocked=True,
                    expected_task_hash=confirmed,
                )
                competitor.append(won.record_hash)
            return answer

        with patch.object(ResearchExperimentApplication, "_current", racing_current):
            lost = _json(
                live,
                "/api/recover",
                method="POST",
                payload={"task_id": task_id, "expected_task_hash": confirmed},
            )
        assert lost["failure_code"] == "local_application.confirmation_stale"
        assert lost["refused_at"] == "TASK_CONTROL" and lost["lifecycle"] == "RECOVERY_REQUIRED"
        assert lost["task_record_hash"] == competitor[0], "the winner's version stands"
        assert registry.task(UUID(task_id)).record_hash == competitor[0]
        # The competitor's reopening is resumed against its own version.
        started = time.perf_counter()
        resumed = _json(
            live,
            "/api/recover",
            method="POST",
            payload={"task_id": task_id, "expected_task_hash": competitor[0]},
        )
        assert resumed["resumed_task_ids"] == [task_id]
        assert "retry" not in resumed, "an already reopened Task needs no second reopening"
        live.dispatcher.drain_for_tests(timeout=600)
        risk = _json(live, "/api/experiments/readback?task_id=" + task_id)
        assert risk["status"] == "EXPERIMENT_PUBLISHED", risk
        assert len(risk["result"]["evaluations"]) == 22
        assert risk["execution_numerical_call_count"] == 1
        assert (
            _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )["status"]
            == "REUSED_EXACT"
        )
        return {"task_id": task_id, "retry_wall_s": time.perf_counter() - started}


def _facts(body: dict) -> dict:  # type: ignore[type-arg]
    """A study answer's facts, without how this read proved them (`verification_basis`, L1)."""

    return {key: value for key, value in body.items() if key != "verification_basis"}
