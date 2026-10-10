"""Activation runs an installed reviewed strategy forward (LS1, V459; OW12).

On a copy of the registered QA installation built by the product
(`ls1-daily`: G6/G2 training inputs and lifecycle studies, a Risk study,
strategy preparation and installation, the Rebound Return Book over its whole support).
Offline: the research update decides the sessions the workspace's data already covers, and its
data step needs no source for them, which a provider refusing every fetch proves.
"""

from __future__ import annotations

import gc
import json
import shutil
import socket
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import Context
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Barrier, current_thread
from typing import Any
from uuid import UUID

import pytest

from alphalattice.capabilities.alpha_modeling.runtime.service import AlphaModelRuntimeService
from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.control.product_host.composition.plain_refusals import explain
from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from alphalattice.interface.local_application.cli import main
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from alphalattice.investment.portfolio_strategy_lab.application.advancement_task import (
    ADVANCEMENT_TASK_KIND,
)
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    PORTFOLIO_PUBLIC_TASK_KIND,
    portfolio_research_task_input,
)
from alphalattice.protocols.research_authoring.selection import load_safe_yaml_document
from tests.portfolio_strategy_lab.activation_review_support import (
    activation_cli,
    publish_activation_review,
)
from tests.portfolio_strategy_lab.local_web_support import _json, _request
from tests.workspace_maintenance.local_data_provider import (
    HeldDataProvider,
    recording_provider,
)
from u0_probe import _copy

PACKAGE = "RETURN_G6_MU_ONLY"


def test_history_retains_readable_cro_metadata_beside_a_corrupt_peer(tmp_path, monkeypatch):
    """V683/OP4/EV: one bad publication cannot hide peers or become a claim of no review."""
    from alphalattice.control.product_host.composition.evidence_review_projection import (
        EvidenceCroProjector,
    )
    from alphalattice.interface.local_application.answers import answer_problem
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        PortfolioReviewPublication,
    )
    from tests.alternative_evidence_desk.review_http_support import build_authority
    from tests.portfolio_strategy_lab.activation_review_support import activation_review_host

    with activation_review_host(
        tmp_path / "workspace",
        review_authority_for_report=lambda report: build_authority(
            tmp_path=tmp_path, report=report
        ),
    ) as book:
        live = book.live
        view = publish_activation_review(live, book.result_hash)
        store = live.review.review_publications.store
        bad_identity = "e" * 64
        (store.root / "cro-review-publications" / f"{bad_identity}.json").write_text(
            "{", encoding="utf-8"
        )

        def no_full_read(*_args, **_kwargs):
            raise AssertionError("History metadata opened a full Evidence or CRO read")

        with monkeypatch.context() as metadata:
            metadata.setattr(EvidenceCroProjector, "projection", no_full_read)
            metadata.setattr(live.review.review_publications, "read", no_full_read)
            history = _json(live, "/api/research-history?history_limit=50")
        assert history["status"] == "PARTIAL_METADATA_READBACK"
        assert answer_problem("RESEARCH_HISTORY", history) is None
        healthy = next(
            row
            for row in history["entries"]
            if row["entry_id"] == f"review:{view.publication.publication_hash}"
        )
        assert healthy["book_summary"]["state"] == "NOT_READ"
        assert healthy["book_summary"]["next_requests"]["review"] == {
            "operation": "EVIDENCE_CRO",
            "result_hash": book.result_hash,
        }
        blocked = next(
            row for row in history["blocked_entries"] if row["record_id"] == bad_identity
        )
        assert blocked["entry_id"] == f"review:{bad_identity}" and blocked["status"] == "REFUSED"
        assert blocked["failure_code"] and blocked["detail"]
        assert blocked["next_requests"] == {
            "workspace": {"operation": "WORKSPACE_SHOW"},
            "backups": {"operation": "WORKSPACE_BACKUPS"},
        }
        # Discovery is partial; the scientific store's original strict collection remains strict.
        with pytest.raises(ValueError):
            store.values("cro-review-publications", PortfolioReviewPublication)


class _NoFetch:
    """A provider that refuses every fetch: the store holds each session the update reaches."""

    name = "yfinance"

    def __getattr__(self, call: str) -> object:
        raise AssertionError(f"the data step fetched ({call}) a session the store holds")


def _settled(session: LocalPortfolioWebSession, task_id: str, timeout: float = 1800.0) -> str:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        record = session.session.task_control_registry.task(UUID(task_id))
        if record.lifecycle.value in {"SUCCEEDED", "BLOCKED", "CANCELLED"}:
            return str(record.lifecycle.value)
        time.sleep(2)
    raise TimeoutError(task_id)


@pytest.mark.real_evidence
@pytest.mark.untyped_failure
def test_a_reviewed_research_book_runs_forward_when_a_person_activates_it(
    tmp_path: Path, evidence_roots, monkeypatch, capsys, caplog
) -> None:
    """LS1/OW12: activation reuses sealed studies; updates publish positions until stopped."""

    from alphalattice.interface.local_application.client import LocalResearchClient
    from alphalattice.kernel.shared_kernel.spans import collect, readout

    root = evidence_roots.require("ls1_daily_flows")
    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")

    def no_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an activation and the sessions its studies scored fit no model")

    monkeypatch.setattr(AlphaModelRuntimeService, "fit", no_fit)
    workspace = tmp_path / "workspace"
    _copy(root, workspace)
    (workspace / "runtime/local-research-connection.json").unlink(missing_ok=True)
    live = LocalPortfolioWebSession.from_workspace(workspace)
    live.data_provider = _NoFetch()
    live.start()
    try:
        operations = live.operations
        books = [
            task
            for task in live.session.task_control_registry.tasks()
            if task.task_kind == PORTFOLIO_PUBLIC_TASK_KIND
            and task.lifecycle.value == "SUCCEEDED"
            and portfolio_research_task_input(task).selected_strategy_package_id == PACKAGE
        ]
        assert books, "the QA root holds no completed Rebound Return Book"
        book = str(max(books, key=lambda task: task.admitted_at).task_id)
        controls = operations.execute(
            PortfolioResearchOperationRequest(operation="CONTROLS", strategy_package_id=PACKAGE)
        )
        if controls["activation"]["status"] == "ACTIVE":
            stopped = _json(
                live,
                "/api/strategy/deactivate",
                method="POST",
                payload={"strategy_package_id": PACKAGE},
            )
            assert stopped["status"] == "DEACTIVATED", stopped
        offered = operations.execute(
            PortfolioResearchOperationRequest(operation="CONTROLS", strategy_package_id=PACKAGE)
        )["activation"]
        assert offered["next_requests"] == {
            "activate": {"operation": "STRATEGY_ACTIVATE", "task_id": book}
        }, offered
        recorded = live.application.pipeline.find_for_task(UUID(book))
        assert recorded is not None
        report = activation_cli(live, capsys, "result", "show", recorded.result_hash)
        review_standing = report["review_standing"]
        assert review_standing["book_selector"] == {"result_hash": recorded.result_hash}
        assert offered["review_standing"] == review_standing
        holdings = offered["review_holdings"]
        assert holdings["claim"] == "BOOK_LAST_HOLDINGS_NOT_NEXT_POSITIONS"
        assert holdings["book_task_id"] == book
        assert holdings["formation_session"] == report["book"]["formation_session"]
        assert holdings["entry_session"] > holdings["formation_session"]
        assert holdings["held_count"] == len(holdings["positions"]) > 0
        assert {row["listing_id"]: row["weight"] for row in holdings["positions"]} == {
            row["listing_id"]: row["weight"]
            for row in report["book"]["positions"]
            if row["weight"] != "0.000%"
        }
        shown = operations.execute(PortfolioResearchOperationRequest(operation="WORKSPACE_SHOW"))
        (forward,) = [i for i in shown["intents"] if i.get("strategy_package_id") == PACKAGE]
        assert (forward["flow"], forward["activation"], forward["next_requests"]) == (
            "RUN_FORWARD",
            offered,
            {"books": {"operation": "CONTROLS", "strategy_package_id": PACKAGE}},
        )
        book_controls = LocalResearchClient(workspace).request(forward["next_requests"]["books"])
        assert book_controls["strategy_package_id"] == PACKAGE
        assert book_controls["activation"] == offered
        daily = _json(live, "/api/research-update/automation")
        assert PACKAGE not in [row["strategy_package_id"] for row in daily["runs_forward"]]
        status, _headers, body = _request(
            live,
            "/api/research-update/automation",
            method="POST",
            payload={"automation_enabled": True, "automation_package_ids": [PACKAGE]},
        )
        assert (status, json.loads(body)) == (
            400,
            {"refused": "research_update.automation_package_not_installed"},
        )
        recovery = operations.execute(
            PortfolioResearchOperationRequest(
                **explain("research_update.automation_package_not_installed")["next_requests"][
                    "automation"
                ]
            )
        )
        assert recovery["runs_forward"] == daily["runs_forward"]
        activated = _json(
            live,
            "/api/strategy/activate",
            method="POST",
            payload={"task_id": book},
            timeout=300.0,
        )
        assert activated["status"] == "ACTIVATED", activated
        assert set(activated["stage_seconds"]) == {"book", "models", "seed", "checkpoint", "bind"}
        assert activated["review_standing"] == review_standing
        assert activated["strategy_dates"]["information_cutoff"] == "2026-09-10"
        assert (
            activated["strategy_dates"]["forward_book_first_decided_session"]
            == activated["next_decision_session"]
        )
        assert activated["strategy_dates"]["forward_book_start_basis"] == "ACTIVE"
        assert activated["strategy_dates"]["first_actionable_session"] > "2026-09-10"
        actionable_source = activated["strategy_dates"]["first_actionable_source"]
        assert datetime.fromisoformat(actionable_source["entry_at"]) > datetime.fromisoformat(
            actionable_source["activated_at"]
        )
        replay = activated["strategy_dates"]["replayed_in_sample_forward_sessions"]
        assert (replay["count"], replay["first_session"], replay["last_session"]) == (
            2,
            "2026-09-09",
            "2026-09-10",
        )
        renewals = activated["strategy_dates"]["model_renewals"]["components"]
        assert renewals and all(r["next_fit_vintage"] == "2026-10" for r in renewals)
        assert activated["fit_calls"] == 0
        assert activated["next_decision_session"] == "2026-09-09"
        assert all(model["reused_fits"] > 0 for model in activated["models"]), activated
        assert activated["calibration_seed_hash"] is not None
        again = _json(live, "/api/strategy/activate", method="POST", payload={"task_id": book})
        assert again["failure_code"] == "strategy_activation.already_active", again
        state = operations.execute(
            PortfolioResearchOperationRequest(operation="CONTROLS", strategy_package_id=PACKAGE)
        )["activation"]
        assert state["status"] == "ACTIVE" and state["book_task_id"] == book
        recorded = live.application.pipeline.find_for_task(UUID(book))
        assert recorded is not None
        assert (
            main(
                [
                    "--workspace",
                    str(workspace),
                    "--view",
                    "full",
                    "result",
                    "show",
                    recorded.result_hash,
                ],
                serve=lambda _: 99,
            )
            == 0
        )
        readback = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"]
        assert readback["strategy_dates"]["information_cutoff"] == "2026-09-10"
        assert readback["strategy_dates"]["book_sessions_after_cutoff"]["count"] == 0
        assert readback["strategy_dates"]["book_sessions_after_cutoff"]["first_session"] is None
        assert (
            main(
                [
                    "--workspace",
                    str(workspace),
                    "--view",
                    "full",
                    "result",
                    "show",
                    recorded.result_hash,
                    "--session",
                    "2026-09-08",
                ],
                serve=lambda _: 99,
            )
            == 0
        )
        positioned = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"]
        assert positioned["strategy_dates"] == readback["strategy_dates"]
        assert positioned["reading_context"]["strategy"]["strategy_id"] == PACKAGE
        assert state["next_requests"] == {
            "deactivate": {"operation": "STRATEGY_DEACTIVATE", "strategy_package_id": PACKAGE}
        }
        daily = _json(live, "/api/research-update/automation")
        (row,) = [row for row in daily["runs_forward"] if row["strategy_package_id"] == PACKAGE]
        assert (row["status"], row["book_task_id"]) == ("ACTIVE", book)
        assert row["first_forward_session"] == "2026-09-09"
        assert PACKAGE in daily["next_requests"]["enable"]["automation_package_ids"]
        assert operations.automation is not None and PACKAGE in operations.automation.installed
        assert operations.scoring is not None
        completed = operations.scoring.additional_publications
        assert completed is not None

        def repeat_certification():
            result = Context().run(completed)
            Context().run(completed)
            return result

        def completed_budget(ledger):
            observed = sum(
                row["count"]
                for row in readout(ledger)["spans"]
                if (row["category"], row["detail"]) == ("verify", "completed_scores")
            )
            assert observed == 1, (
                f"COUNT STOP op=completed_scores observed={observed} budget=1 "
                "way_on=retain_request_certification"
            )

        for planted in (False, True):
            with monkeypatch.context() as repeated:
                if planted:
                    repeated.setattr(
                        operations.scoring, "additional_publications", repeat_certification
                    )
                with collect() as ledger:
                    plan = operations.execute(
                        PortfolioResearchOperationRequest(
                            operation="RESEARCH_UPDATE_PLAN",
                            strategy_package_id=PACKAGE,
                            observed_through="2026-09-10",
                        ),
                        caller="EXTERNAL_AUTOMATION",
                    )
                if planted:
                    with pytest.raises(AssertionError, match="COUNT STOP") as excess:
                        completed_budget(ledger)
                    assert "observed=0 " not in str(excess.value)
                else:
                    completed_budget(ledger)
        assert plan["status"] == "PLANNED", plan
        assert plan["decision_sessions"] == ["2026-09-09", "2026-09-10"]
        assert plan["next_requests"] == {
            "run": {
                "operation": "RESEARCH_UPDATE_RUN",
                "update_plan_hash": plan["update_plan_hash"],
            }
        }
        shown = operations.execute(PortfolioResearchOperationRequest(operation="WORKSPACE_SHOW"))
        (forward,) = [i for i in shown["intents"] if i.get("strategy_package_id") == PACKAGE]
        assert forward["activation"]["status"] == "ACTIVE"
        assert forward["next_requests"] == {
            "update": {"operation": "RESEARCH_UPDATE_PLAN", "strategy_package_id": PACKAGE}
        }
        sent = operations.execute(
            PortfolioResearchOperationRequest(
                operation="RESEARCH_UPDATE_RUN", update_plan_hash=plan["update_plan_hash"]
            ),
            caller="EXTERNAL_AUTOMATION",
        )
        assert sent["status"] == "ADMITTED", sent
        assert _settled(live, str(sent["task_id"])) == "SUCCEEDED"
        assert operations.activations is not None and operations.research_updates is not None
        read_task = PortfolioResearchOperationRequest(
            operation="RESEARCH_UPDATE_READBACK", task_id=UUID(str(sent["task_id"]))
        )
        door = PortfolioResearchOperationRequest(operation="RESEARCH_UPDATE_AUTOMATION_READBACK")
        why = "Automation metadata opened full review/update readback"
        gate, caller = Barrier(2, timeout=30), current_thread()
        first = True

        def read_metadata():
            gate.wait()
            return operations.execute(door)

        def guard(read):
            def checked(*args, **kwargs):
                nonlocal first
                assert current_thread() is not metadata, why
                if current_thread() is caller and first:
                    first = False
                    gate.wait()
                return read(*args, **kwargs)

            return checked

        with ThreadPoolExecutor(max_workers=1) as reader, monkeypatch.context() as probe:
            metadata = reader.submit(current_thread).result(timeout=30)
            for owner, method in (
                (operations.activations, "read_review"),
                (operations.research_updates, "readback"),
            ):
                probe.setattr(owner, method, guard(getattr(owner, method)))
            daily = reader.submit(read_metadata)
            result = operations.execute(read_task, caller="EXTERNAL_AUTOMATION")
            daily = daily.result(timeout=30)
            assert result["status"] == "PROPOSAL_PUBLISHED", result
            probe.setattr(operations.automation, "readback", lambda: operations.execute(read_task))
            with pytest.raises(AssertionError, match=why) as bad:
                reader.submit(operations.execute, door).result(timeout=30)
        (running,) = [row for row in daily["runs_forward"] if row["strategy_package_id"] == PACKAGE]
        latest = running["latest_update"]
        assert "review_standing" not in running
        assert latest["task_id"] == str(sent["task_id"]) and latest["lifecycle"] == "SUCCEEDED"
        assert latest["verification"] == "METADATA_ONLY_SELECTED_READBACK_VERIFIES_DESCENDANTS"
        assert latest["target_session"] == result["update"]["target_session"]
        assert latest["claim"] == result["publication"]["claim"]
        assert latest["review_selector"] == result["review_selector"]
        assert latest["next_requests"]["readback"] == {
            "operation": "RESEARCH_UPDATE_READBACK",
            "task_id": str(sent["task_id"]),
        }

        proposal = result["publication"]["pending_proposal"]
        assert proposal["schedule"]["formation_session"] == "2026-09-10"
        assert proposal["schedule"]["entry_session"] == "2026-09-11"
        assert datetime.fromisoformat(
            proposal["schedule"]["formation_close_at"]
        ) < datetime.fromisoformat(proposal["schedule"]["entry_open_at"])
        assert any(weight > 0 for weight in proposal["estimated_weights"])
        from alphalattice.interface.local_application.portfolio_research import (
            PortfolioResearchRequestDocument,
        )

        offered_preview = result["next_requests"]["evidence_preview"]
        assert offered_preview == {"operation": "EVIDENCE_PREVIEW", **result["review_selector"]}
        preview = operations.execute(
            PortfolioResearchRequestDocument.model_validate(offered_preview).to_operation_request(),
            caller="EXTERNAL_AUTOMATION",
        )
        assert preview["status"] == "EVIDENCE_PREREQUISITES_MISSING", preview
        assert preview["failure_code"] == "REFUSED_NO_ADMITTED_EVIDENCE_AUTHORITY", preview

        def score_plan(session: str, component: str | None) -> dict[str, object]:
            return operations.execute(
                PortfolioResearchOperationRequest(
                    operation="STRATEGY_SCORE_PLAN",
                    strategy_package_id=PACKAGE,
                    formation_session=session,
                    component_id=component,
                ),
                caller="EXTERNAL_AUTOMATION",
            )

        component = None
        first = score_plan("2026-09-09", component)
        if str(first.get("failure_code", "")).startswith("strategy_score.component_required:"):
            component = str(first["failure_code"]).split(":", 1)[1].split(",")[0]
            first = score_plan("2026-09-09", component)
        second = score_plan("2026-09-10", component)
        assert first["status"] == second["status"] == "PLANNED", (first, second)
        assert first["score_plan_hash"] != second["score_plan_hash"]
        reopened = operations.scoring.prepare(str(first["score_plan_hash"]))
        assert reopened.formation_session.isoformat() == "2026-09-09"
        replay = next(
            task.task_id
            for task in live.session.task_control_registry.tasks()
            if isinstance(task.input.payload.get("plan"), dict)
            and task.input.payload["plan"].get("model_training_source") is not None
        )
        drafted = operations.execute(
            PortfolioResearchOperationRequest(
                operation="EXPERIMENT_PORTFOLIO_DRAFT", task_id=replay, candidate_id="any"
            ),
            caller="EXTERNAL_AUTOMATION",
        )
        assert drafted["failure_code"] == "portfolio_research.alpha_development_required", drafted
        assert drafted["prerequisites"]["flow"] == "BOOK", drafted
        enable = daily["next_requests"]["enable"]
        on = _json(
            live,
            "/api/research-update/automation",
            method="POST",
            payload={k: enable[k] for k in ("automation_enabled", "automation_package_ids")},
        )
        assert on["status"] == "ENABLED_SERVICE_LIFETIME", on
        assert "enable" not in on["next_requests"] and "disable" in on["next_requests"]
        off = on["next_requests"]["disable"]
        assert (
            _json(
                live,
                "/api/research-update/automation",
                method="POST",
                payload={k: off[k] for k in ("automation_enabled", "automation_package_ids")},
            )["status"]
            == "DISABLED"
        )
        stopped = _json(
            live,
            "/api/strategy/deactivate",
            method="POST",
            payload={"strategy_package_id": PACKAGE},
        )
        assert stopped["status"] == "DEACTIVATED", stopped
        after = operations.execute(
            PortfolioResearchOperationRequest(operation="CONTROLS", strategy_package_id=PACKAGE)
        )
        assert after["strategy_dates"]["forward_book_start_basis"] == "IF_ACTIVATED"
        assert after["strategy_dates"]["information_cutoff"] == "2026-09-10"
        assert after["activation"] == {
            "status": "INACTIVE",
            "strategy_dates": after["strategy_dates"],
            "review_standing": review_standing,
            "review_holdings": holdings,
            "next_requests": {"activate": {"operation": "STRATEGY_ACTIVATE", "task_id": book}},
        }
    finally:
        live.stop()
        operations = None
        del live
        gc.collect()
        assert workspace.resolve().is_relative_to(tmp_path.resolve())
        shutil.rmtree(workspace)

    raised = {r.args[2] for r in caplog.records if r.getMessage().startswith("operation.raised:")}
    assert raised == {f"AssertionError: {bad.value}"}


@pytest.mark.real_evidence
def test_installed_book_controls_saves_the_flat_declaration_its_preview_reads(
    tmp_path: Path, evidence_roots, monkeypatch, capsys
) -> None:
    """requirement (V538): controls gives the installed book's editable declaration; saving
    and loading it preserves the defaults, and the named preview accepts that flat document."""
    root = evidence_roots.require("ls1_daily_flows")
    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    workspace = tmp_path / "workspace"
    _copy(root, workspace)
    (workspace / "runtime/local-research-connection.json").unlink(missing_ok=True)
    live = LocalPortfolioWebSession.from_workspace(workspace)
    live.start()
    try:
        declaration = tmp_path / "book.yaml"
        saved = tmp_path / "controls.json"
        prefix = ["--workspace", str(workspace), "--view", "full"]
        assert (
            main(
                [
                    *prefix,
                    "strategy-book",
                    "controls",
                    "--package",
                    PACKAGE,
                    "--save-declaration",
                    str(declaration),
                    "--output",
                    str(saved),
                ],
                serve=lambda _: 99,
            )
            == 0
        )
        answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert "local_failure" not in answer
        body = json.loads(saved.read_text("utf-8"))
        template = body["template"]
        assert template == {
            "strategy_package_id": PACKAGE,
            "score_source_mode": body["default_score_source_mode"],
            **{row["control_id"]: row["default_value"] for row in body["controls"]},
        }
        assert not ({row["control_id"] for row in body["frozen"]} & template.keys())
        assert load_safe_yaml_document(declaration.read_text("utf-8")) == template
        assert body["next_requests"]["preview"] == {
            "operation": "PLAN",
            "spec": {"strategy_package_id": PACKAGE},
        }
        assert (
            main(
                [*prefix, "strategy-book", "preview", "--file", str(declaration)],
                serve=lambda _: 99,
            )
            == 0
        )
        preview = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert preview["outcome"] == "OK", preview
        assert preview["data"]["strategy_package_id"] == PACKAGE
        assert (
            main(
                [
                    *prefix,
                    "request",
                    "--from",
                    str(saved),
                    "--action",
                    "preview",
                    "--file",
                    str(declaration),
                ],
                serve=lambda _: 99,
            )
            == 0
        )
        continued = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert continued["outcome"] == "OK", continued
        assert continued["data"] == preview["data"]
        # A PLAN/spec envelope is not the flat declaration. With packages installed it
        # omitted the selector; that cannot mean the workspace has none installed (RR5d).
        missing = tmp_path / "missing.yaml"
        missing.write_text(json.dumps({"operation": "PLAN", "spec": template}), encoding="utf-8")
        unknown = tmp_path / "unknown.yaml"
        unknown.write_text(
            json.dumps({"strategy_package_id": "NOT_AN_INSTALLED_PACKAGE"}), encoding="utf-8"
        )
        for action in ("preview", "run"):
            for file, code in (
                (missing, "strategy_book.strategy_package_required"),
                (
                    unknown,
                    "local_application.strategy_package_not_installed:NOT_AN_INSTALLED_PACKAGE",
                ),
            ):
                assert (
                    main(
                        [*prefix, "strategy-book", action, "--file", str(file)], serve=lambda _: 99
                    )
                    == 2
                )
                refusal = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
                assert refusal["failure_code"] == code, refusal
                assert refusal["detail"] and PACKAGE in refusal["detail"]
                assert refusal["data"]["next_action"] == "SELECT_STRATEGY_PACKAGE_FROM_CONTROLS"
                assert refusal["next_requests"][f"controls:{PACKAGE}"] == {
                    "operation": "CONTROLS",
                    "strategy_package_id": PACKAGE,
                }
                if file == missing:
                    assert "strategy_package_id" in refusal["detail"]
                else:
                    assert "NOT_AN_INSTALLED_PACKAGE" in refusal["detail"]
        assert main([*prefix, "strategy-book", "controls"], serve=lambda _: 99) == 2
        unselected = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert unselected["failure_code"] == "strategy_book.strategy_package_required"
        assert "strategy_package_id" in unselected["detail"] and PACKAGE in unselected["detail"]
        for name, request in unselected["next_requests"].items():
            assert name == "controls:" + request["strategy_package_id"]
            assert request["strategy_package_id"] in unselected["detail"]
            assert request["operation"] == "CONTROLS"
    finally:
        live.stop()
        del live
        gc.collect()
        assert workspace.resolve().is_relative_to(tmp_path.resolve())
        shutil.rmtree(workspace)


@pytest.mark.parametrize("active", (False, True))
@pytest.mark.parametrize(
    "package,cutoff",
    (("RETURN_G6_MU_ONLY", "2026-07-03"), ("BALANCED_G2_G6_EQUAL_CAPITAL", "2026-07-05")),
)
def test_installed_strategy_dates_follow_sealed_records_on_the_cli(
    tmp_path, capsys, monkeypatch, active, package, cutoff
):
    """Installed strategy dates follow sealed records on the CLI."""

    from alphalattice.control.product_host.composition.entry import main as date_cli
    from tests.portfolio_strategy_lab.strategy_dates_support import date_host

    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    with date_host(tmp_path / "workspace", active=active) as live:
        prefix = ["--workspace", str(live.workspace), "--view", "full"]
        assert date_cli([*prefix, "strategy-book", "controls", "--package", package]) == 0
        controls = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"]
        dates = controls["strategy_dates"]
        assert dates["information_cutoff"] == cutoff
        assert dates["information_cutoff_status"] == "KNOWN"
        assert dates["forward_book_start_basis"] == ("ACTIVE" if active else "IF_ACTIVATED")
        assert dates["forward_book_first_decided_session"] == (
            "2026-07-23" if active else "2026-07-01"
        )
        assert dates["forward_book_start_detail"]
        assert dates["first_actionable_session"] == ("2026-07-23" if active else "2026-10-05")
        assert dates["first_actionable_basis"] == ("ACTIVE" if active else "IF_ACTIVATED")
        assert dates["first_actionable_source"]["latest_completed_session"] == (
            "2026-07-22" if active else "2026-10-02"
        )
        replay = dates["replayed_in_sample_forward_sessions"]
        assert replay["count"] == (0 if active else 2)
        assert replay["first_session"] == (None if active else "2026-07-01")
        assert replay["last_session"] == (None if active else "2026-07-02")
        assert "causal replay in-sample, never out-of-sample" in replay["detail"]
        assert dates == controls["activation"]["strategy_dates"]
        renewal = dates["model_renewals"]
        assert renewal["detail"]
        if active:
            assert renewal["status"] == "KNOWN"
            assert {r["component_id"] for r in renewal["components"]} == (
                {"G6_R0_FAST_REBOUND"}
                if package == "RETURN_G6_MU_ONLY"
                else {"G2_R0_TREND", "G6_R0_FAST_REBOUND"}
            )
            for component in renewal["components"]:
                assert component["remaining_fit_vintages"] == ["2026-01", "2026-04", "2026-07"]
                assert component["next_fit_vintage"] == "2026-01"
                assert component["renewal_through"] == "2026-09-30"
                assert component["grant_file"] and component["grant_hash"]
        else:
            assert renewal["status"] == "INACTIVE"
            assert renewal["components"] == []
            assert "inactive" in renewal["detail"] and "no fit vintage" in renewal["detail"]
        sources = dates["information_sources"]
        assert {s["kind"] for s in sources} >= {
            "FROZEN_RESEARCH_WINDOW",
            "POST_OBSERVED_SESSION_AXIS",
            "TRAINING_OBSERVATIONS",
            "TRAINING_OBSERVATION_AXIS",
            "PREPARED_REFIT_WINDOWS",
            "FIT_VINTAGES",
        }
        assert all(s["record_id"] and s["relative_path"] and s["field"] for s in sources)
        assert max(s["through"] for s in sources) == cutoff
        assert {s["component_id"] for s in sources if s["kind"] == "TRAINING_OBSERVATIONS"} == (
            {"G6_R0_FAST_REBOUND"}
            if package == "RETURN_G6_MU_ONLY"
            else {"G2_R0_TREND", "G6_R0_FAST_REBOUND"}
        )
        assert date_cli([*prefix, "workspace", "show"]) == 0
        listing = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"][
            "installed_strategies"
        ]
        assert next(p for p in listing if p["strategy_id"] == package)["strategy_dates"] == dates
        assert live.operations.activations.state(package)["strategy_dates"] == dates
        # The public owner is also the book readback's source. Equality is not after cutoff.
        day = date.fromisoformat(cutoff)

        book = live.operations.activations.dates(
            package,
            book_sessions=(
                day - timedelta(days=1),
                day,
                day + timedelta(days=1),
                day + timedelta(days=2),
            ),
        )
        assert book["book_sessions_after_cutoff"]["count"] == 2
        assert (
            book["book_sessions_after_cutoff"]["first_session"]
            == (day + timedelta(days=1)).isoformat()
        )


@pytest.mark.parametrize("gap", ("missing", "undated"))
def test_strategy_with_a_missing_training_date_says_what_is_unknown(
    tmp_path, capsys, monkeypatch, gap
):
    """V589: a missing sealed record cannot become study_end or a guessed information cutoff."""
    from alphalattice.control.product_host.composition.entry import main as date_cli
    from tests.portfolio_strategy_lab.strategy_dates_support import date_host

    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    with date_host(
        tmp_path / "workspace", missing=gap == "missing", undated=gap == "undated"
    ) as live:
        assert (
            date_cli(
                [
                    "--workspace",
                    str(live.workspace),
                    "--view",
                    "full",
                    "strategy-book",
                    "controls",
                    "--package",
                    PACKAGE,
                ]
            )
            == 0
        )
        data = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"][
            "strategy_dates"
        ]
        assert data["information_cutoff"] is None
        assert data["information_cutoff_status"] == "UNAVAILABLE"
        assert "cannot be determined" in data["information_cutoff_detail"]
        absent = [s for s in data["information_sources"] if s["status"] == "UNAVAILABLE"]
        assert absent and all(s["detail"] and s["relative_path"] for s in absent)
        assert data["latest_dated_information"] == "2026-07-02"
        assert data["forward_book_first_decided_session"] == "2026-07-01"
        assert data["first_actionable_session"] == "2026-10-05"
        assert data["replayed_in_sample_forward_sessions"]["count"] is None


@pytest.mark.parametrize("active", (False, True))
def test_date_read_locates_the_original_studys_fits_without_bulk_verification(tmp_path, active):
    """V589: inactive packages select training by the sealed study, with its actual fit metadata."""
    from tests.portfolio_strategy_lab.strategy_dates_support import date_host, study_date_owner

    with date_host(tmp_path / "workspace", active=active) as live:
        owner = study_date_owner(live)
        for package, cutoff in (
            (PACKAGE, "2026-07-03"),
            ("BALANCED_G2_G6_EQUAL_CAPITAL", "2026-07-05"),
        ):
            answer = owner.dates(package)
            assert answer["information_cutoff"] == cutoff
            fits = [s for s in answer["information_sources"] if s["kind"] == "FIT_VINTAGES"]
            assert fits and all(s["relative_path"].startswith("studies/") for s in fits)
            assert all(s["status"] == "KNOWN" and s["vintages"] for s in fits)
            assert answer["forward_book_first_decided_session"] == (
                "2026-07-23" if active else "2026-07-01"
            )
            renewal = answer["model_renewals"]
            if active:
                assert renewal["status"] == "UNAVAILABLE"
                assert renewal["components"]
                for component in renewal["components"]:
                    assert component["status"] == "UNAVAILABLE"
                    assert component["remaining_fit_vintages"] is None
                    assert component["next_fit_vintage"] is None
                    assert component["grant_file"] is None
                    assert "No active grant is bound" in component["detail"]
            else:
                assert renewal["status"] == "INACTIVE"
                assert renewal["components"] == []


def test_conditional_start_uses_the_activations_admitted_score_axis(tmp_path, capsys, monkeypatch):
    """V589: an interior component gap cannot silently become an activatable book axis."""
    from alphalattice.control.product_host.composition.entry import main as date_cli
    from tests.portfolio_strategy_lab.strategy_dates_support import date_host

    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    with date_host(tmp_path / "workspace", score_gap=True) as live:
        for package, known in ((PACKAGE, True), ("BALANCED_G2_G6_EQUAL_CAPITAL", False)):
            assert (
                date_cli(
                    [
                        "--workspace",
                        str(live.workspace),
                        "--view",
                        "full",
                        "strategy-book",
                        "controls",
                        "--package",
                        package,
                    ]
                )
                == 0
            )
            dates = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"][
                "strategy_dates"
            ]
            if known:
                assert dates["information_cutoff"] == "2026-07-03"
                assert dates["forward_book_first_decided_session"] == "2026-07-01"
            else:
                assert dates["information_cutoff"] is None
                assert dates["information_cutoff_detail"]
                assert dates["forward_book_first_decided_session"] is None
                assert dates["forward_book_start_basis"] == "UNAVAILABLE"
                assert dates["forward_book_start_detail"]


def test_a_position_row_whose_entry_open_has_passed_says_so() -> None:
    """requirement (the person's date): a conditional estimate read after its entry's open says
    the open has passed; before it, it stays a conditional execution."""
    from alphalattice.control.product_host.composition.strategy_activation import position_basis

    opened = datetime(2026, 10, 9, 13, 30, tzinfo=UTC)
    before, after = opened - timedelta(minutes=1), opened + timedelta(minutes=1)
    assert position_basis("CONDITIONAL_ESTIMATE", opened, before).endswith("conditional execution")
    assert position_basis("CONDITIONAL_ESTIMATE", opened, after).endswith("open has passed")
    assert position_basis("OBSERVED_RESEARCH_ENTRY", opened, after) == "Observed research entry"


@pytest.mark.parametrize("package", (PACKAGE, "BALANCED_G2_G6_EQUAL_CAPITAL"))
def test_a_forward_book_before_data_end_names_actionable_and_in_sample_replay(
    tmp_path, capsys, monkeypatch, package
):
    """V597: the first forward decision can precede both cutoff and actionable holdings."""
    from alphalattice.control.product_host.composition.entry import main as date_cli
    from tests.portfolio_strategy_lab.strategy_dates_support import date_host

    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    with date_host(tmp_path / "workspace", active=True, book_before_data_end=True) as live:
        prefix = ["--workspace", str(live.workspace), "--view", "full"]
        assert date_cli([*prefix, "strategy-book", "controls", "--package", package]) == 0
        dates = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"][
            "strategy_dates"
        ]
        assert dates["forward_book_first_decided_session"] == "2026-07-01"
        assert dates["information_cutoff"] > dates["forward_book_first_decided_session"]
        assert dates["first_actionable_session"] == "2026-07-23"
        assert dates["first_actionable_source"]["activated_at"] == "2026-07-22T23:00:00+00:00"
        replay = dates["replayed_in_sample_forward_sessions"]
        assert (replay["count"], replay["first_session"], replay["last_session"]) == (
            2,
            "2026-07-01",
            "2026-07-02",
        )
        assert "causal replay in-sample, never out-of-sample" in replay["detail"]
        assert live.operations.activations.offer(package)["strategy_dates"] == dates


@pytest.mark.parametrize(
    "clock,completed,actionable,entry_at",
    (
        ("2026-07-02T12:00Z", "2026-07-01", "2026-07-02", "2026-07-02T13:30Z"),
        ("2026-07-02T13:30Z", "2026-07-01", "2026-07-06", "2026-07-06T13:30Z"),
        ("2026-07-02T19:59Z", "2026-07-01", "2026-07-06", "2026-07-06T13:30Z"),
        ("2026-07-02T20:30Z", "2026-07-02", "2026-07-06", "2026-07-06T13:30Z"),
        ("2026-07-03T16:00Z", "2026-07-02", "2026-07-06", "2026-07-06T13:30Z"),
        ("2026-11-27T13:00Z", "2026-11-25", "2026-11-27", "2026-11-27T14:30Z"),
        ("2026-11-27T17:59Z", "2026-11-25", "2026-11-30", "2026-11-30T14:30Z"),
        ("2026-11-27T18:30Z", "2026-11-27", "2026-11-30", "2026-11-30T14:30Z"),
        ("2026-03-09T19:59Z", "2026-03-06", "2026-03-10", "2026-03-10T13:30Z"),
        ("2026-03-09T20:30Z", "2026-03-09", "2026-03-10", "2026-03-10T13:30Z"),
    ),
    ids=(
        "before-open",
        "at-open",
        "intraday",
        "after-close-before-provider-finality",
        "holiday",
        "early-close-before-open",
        "early-close-intraday",
        "early-close-after",
        "dst-intraday",
        "dst-after",
    ),
)
@pytest.mark.parametrize("active", (False, True), ids=("inactive", "active"))
def test_actionable_entry_is_strictly_after_activation_or_clock_on_the_cli(
    tmp_path, capsys, monkeypatch, active, clock, completed, actionable, entry_at
):
    """Actionable entry is strictly after activation or clock on the CLI."""

    from alphalattice.control.product_host.composition.entry import main as date_cli
    from tests.portfolio_strategy_lab.strategy_dates_support import date_host

    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    instant = datetime.fromisoformat(clock)
    observed = [instant]
    with date_host(
        tmp_path / "workspace", active=active, clock=lambda: observed[0], activated_at=instant
    ) as live:
        prefix = ["--workspace", str(live.workspace), "--view", "full"]
        for package in ("RETURN_G6_MU_ONLY", "BALANCED_G2_G6_EQUAL_CAPITAL"):
            assert date_cli([*prefix, "strategy-book", "controls", "--package", package]) == 0
            controls = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"]
            dates = controls["strategy_dates"]
            assert dates["first_actionable_session"] == actionable
            assert dates["first_actionable_basis"] == ("ACTIVE" if active else "IF_ACTIVATED")
            source = dates["first_actionable_source"]
            assert source["latest_completed_session"] == completed
            assert source["reference"] == ("ACTIVATION" if active else "CURRENT_CLOCK")
            if active:
                assert datetime.fromisoformat(source["activated_at"]) == instant
            else:
                assert source["observed_on"] == instant.date().isoformat()
            assert "observed_at" not in source
            entry = datetime.fromisoformat(source["entry_at"])
            assert entry == datetime.fromisoformat(entry_at)
            assert entry > instant
            assert source["formation_session"] < actionable
            assert "strictly after" in source["rule"]
            assert dates == controls["activation"]["strategy_dates"]
            # Every date producer must read the same result as the clock advances
            # within this market phase. Compare the complete answer, without masking fields.
            observed[0] += timedelta(microseconds=137)
            assert date_cli([*prefix, "strategy-book", "controls", "--package", package]) == 0
            repeated = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"]
            assert repeated == controls
            assert date_cli([*prefix, "workspace", "show"]) == 0
            listing = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"][
                "installed_strategies"
            ]
            assert next(p for p in listing if p["strategy_id"] == package)["strategy_dates"] == (
                dates
            )


@pytest.mark.parametrize(
    "clock,actionable,count,last",
    (
        ("2026-07-02T12:00Z", "2026-07-02", 1, "2026-07-01"),
        ("2026-07-02T19:59Z", "2026-07-06", 2, "2026-07-02"),
    ),
    ids=("before-open", "intraday"),
)
def test_in_sample_replay_stops_before_the_corrected_actionable_entry(
    tmp_path, capsys, monkeypatch, clock, actionable, count, last
):
    """The same frozen replay axis changes its boundary when the day's entry has passed."""
    from alphalattice.control.product_host.composition.entry import main as date_cli
    from tests.portfolio_strategy_lab.strategy_dates_support import date_host

    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    with date_host(
        tmp_path / "workspace",
        active=True,
        book_before_data_end=True,
        activated_at=datetime.fromisoformat(clock),
    ) as live:
        for package in ("RETURN_G6_MU_ONLY", "BALANCED_G2_G6_EQUAL_CAPITAL"):
            assert (
                date_cli(
                    [
                        "--workspace",
                        str(live.workspace),
                        "--view",
                        "full",
                        "strategy-book",
                        "controls",
                        "--package",
                        package,
                    ]
                )
                == 0
            )
            dates = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"][
                "strategy_dates"
            ]
            assert dates["first_actionable_session"] == actionable
            replay = dates["replayed_in_sample_forward_sessions"]
            assert (replay["count"], replay["first_session"], replay["last_session"]) == (
                count,
                "2026-07-01",
                last,
            )
            assert "never out-of-sample" in replay["detail"]


@pytest.mark.parametrize("active", (False, True), ids=("inactive", "active"))
def test_friday_intraday_activation_names_mondays_entry_on_the_cli(
    tmp_path, capsys, monkeypatch, active
):
    """Friday intraday activation names Monday's entry on the CLI."""
    from alphalattice.control.product_host.composition.entry import main as date_cli
    from tests.portfolio_strategy_lab.strategy_dates_support import date_host

    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    activation = datetime(2026, 10, 2, 16, 8, 5, tzinfo=UTC)
    now = datetime(2026, 10, 6, 16, 8, 5, tzinfo=UTC)
    with date_host(
        tmp_path / "workspace", active=active, activated_at=activation, observed_at=now
    ) as live:
        for package in ("RETURN_G6_MU_ONLY", "BALANCED_G2_G6_EQUAL_CAPITAL"):
            assert (
                date_cli(
                    [
                        "--workspace",
                        str(live.workspace),
                        "--view",
                        "full",
                        "strategy-book",
                        "controls",
                        "--package",
                        package,
                    ]
                )
                == 0
            )
            dates = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"][
                "strategy_dates"
            ]
            source = dates["first_actionable_source"]
            assert source["reference"] == ("ACTIVATION" if active else "CURRENT_CLOCK")
            assert dates["first_actionable_session"] == ("2026-10-05" if active else "2026-10-07")
            assert source["entry_at"] == (
                "2026-10-05T13:30:00+00:00" if active else "2026-10-07T13:30:00+00:00"
            )
            if active:
                assert datetime.fromisoformat(source["activated_at"]) == activation
            else:
                assert source["observed_on"] == now.date().isoformat()
            assert "observed_at" not in source


def test_conditional_actionable_readback_changes_at_the_entry_boundary(
    tmp_path, capsys, monkeypatch
):
    """Stable same-phase reads must still advance at the exact entry, without clock rounding."""
    from alphalattice.control.product_host.composition.entry import main as date_cli
    from tests.portfolio_strategy_lab.strategy_dates_support import date_host

    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    observed = [datetime(2026, 10, 2, 13, 29, 59, 999999, tzinfo=UTC)]
    with date_host(tmp_path / "workspace", clock=lambda: observed[0]) as live:
        for package in ("RETURN_G6_MU_ONLY", "BALANCED_G2_G6_EQUAL_CAPITAL"):
            dates = []
            for instant in (
                datetime(2026, 10, 2, 13, 29, 59, 999999, tzinfo=UTC),
                datetime(2026, 10, 2, 13, 30, tzinfo=UTC),
            ):
                observed[0] = instant
                assert (
                    date_cli(
                        [
                            "--workspace",
                            str(live.workspace),
                            "--view",
                            "full",
                            "strategy-book",
                            "controls",
                            "--package",
                            package,
                        ]
                    )
                    == 0
                )
                dates.append(
                    json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"][
                        "strategy_dates"
                    ]
                )
            assert [d["first_actionable_session"] for d in dates] == ["2026-10-02", "2026-10-05"]
            assert {d["first_actionable_source"]["observed_on"] for d in dates} == {"2026-10-02"}
            assert all("observed_at" not in d["first_actionable_source"] for d in dates)


def _seen(answer: dict, registry: DuckDbTaskControlRegistry, task: str | None) -> str:
    """A wait's answer whole, as text, beside its Task's lifecycle as Task Control holds it at
    the same instant (V602, V607): the lifecycle the answer read, its wait event and incident, so
    a Task read as running after it ended is told from an incident's wake."""
    held = None if task is None else registry.task(UUID(task)).lifecycle.value
    return json.dumps({"registry_lifecycle": held, "answer": answer}, indent=1, default=str)


def _no_membership_source(**_kwargs: object) -> object:
    raise ConnectionError("an offline test reads no membership source")


@contextmanager
def _forward_copy(
    tmp_path: Path,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    provider: object | None = None,
    provider_from: Callable[[Path], object] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Iterator[
    tuple[
        LocalPortfolioWebSession,
        Callable[..., tuple[int, dict]],
        Callable[[str], dict[str, int]],
    ]
]:
    """A copy of the research installation, its strategy activated by a person, offline: the
    provider's yfinance module absent, the membership source refusing, no socket off the machine
    and no model fitted. Yields the Host, the real CLI on it and a Task's attempts by stage
    (V600, V601). `provider_from` builds the provider from the copy before the Host opens it
    (V604)."""

    # The provider's yfinance module is absent: every fetch refuses, `data.provider_unavailable`.
    monkeypatch.setitem(sys.modules, "yfinance", None)
    connect = socket.socket.connect

    def loopback(self: socket.socket, address: object) -> object:
        host = address[0] if isinstance(address, tuple) else address
        assert host in {"127.0.0.1", "localhost", "::1"}, f"the test reached {address}"
        return connect(self, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", loopback)

    def no_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a forward update's resume fits no model")

    monkeypatch.setattr(AlphaModelRuntimeService, "fit", no_fit)
    backup_root = tmp_path / "backups"
    monkeypatch.setenv("ALPHALATTICE_BACKUP_ROOT", str(backup_root))
    workspace = tmp_path / "workspace"
    _copy(root, workspace)
    (workspace / "runtime/local-research-connection.json").unlink(missing_ok=True)
    if provider_from is not None:
        provider = provider_from(workspace)
    live = LocalPortfolioWebSession.from_workspace(
        workspace, clock=clock or (lambda: datetime.now(UTC))
    )
    live.data_provider = provider  # type: ignore[assignment]
    live.data_source_loader = _no_membership_source
    live.start()
    try:
        registry = live.session.task_control_registry
        offered = live.operations.execute(
            PortfolioResearchOperationRequest(operation="CONTROLS", strategy_package_id=PACKAGE)
        )["activation"]
        activated = _json(
            live,
            "/api/strategy/activate",
            method="POST",
            payload={"task_id": offered["next_requests"]["activate"]["task_id"]},
            timeout=300.0,
        )
        assert activated["status"] == "ACTIVATED", activated

        def cli(*argv: str) -> tuple[int, dict]:
            code = main(
                ["--workspace", str(workspace), "--view", "full", *argv], serve=lambda _: 99
            )
            return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])

        def attempts(task: str) -> dict[str, int]:
            _record, items = registry.task_with_work_items(UUID(task))
            return {item.stage_id: item.attempt_count for item in items if item.attempt_count}

        yield live, cli, attempts
    finally:
        live.stop()
        gc.collect()
        assert workspace.resolve().is_relative_to(tmp_path.resolve())
        shutil.rmtree(workspace)
        if backup_root.exists():
            assert backup_root.resolve().is_relative_to(tmp_path.resolve())
            shutil.rmtree(backup_root)


@pytest.mark.real_evidence
def test_an_update_stopped_on_the_network_resumes_by_the_way_its_stop_names(
    tmp_path: Path, evidence_roots, monkeypatch, capsys
) -> None:
    """An update stopped on the network resumes by the way its stop names."""

    # V620: reproduce the UX's operator-held stop first. Switch-off read controls below
    # use this copy's local provider guards; every read remains on the machine.
    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    root = evidence_roots.require("ls1_daily_flows")
    with _forward_copy(tmp_path, root, monkeypatch, capsys) as (live, cli, attempts):
        registry = live.session.task_control_registry

        def network(enabled: bool) -> None:
            set_ = _json(
                live,
                "/api/workspace/network",
                method="POST",
                payload={"network_enabled": enabled},
            )
            assert set_["network_allowed"] is enabled, set_

        def check_ways(task: str, *reads: tuple[str, ...]) -> None:
            """Read one saved stop under each effective decision without executing it."""
            held, counted = registry.task(UUID(task)).record_hash, attempts(task)
            for operator, enabled in ((True, True), (False, False), (False, True)):
                monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
                network(enabled)
                if operator:
                    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
                for arguments in reads:
                    code, answer = cli(*arguments)
                    assert code == 2
                    body, detail = answer["data"], answer["detail"]
                    access = body["network_access"]
                    assert answer["next_requests"]["network"] == {"operation": "NETWORK_ACCESS"}
                    if operator:
                        assert access["decided_by"] == "OPERATOR_OFFLINE_SWITCH"
                        assert not access["network_allowed"] and not access["next_requests"]
                        assert body["next_action"] == "RESTART_WITHOUT_OPERATOR_OFFLINE_SWITCH"
                        assert "ALPHALATTICE_NETWORK_DISABLED=1" in detail
                        assert "restart the idle Host" in detail and "network set" not in detail
                    elif not enabled:
                        assert access["decided_by"] == "WORKSPACE_CONTROL"
                        assert not access["network_allowed"] and "network set" in detail
                        assert access["next_requests"]["set"]["network_enabled"]
                    else:
                        assert (
                            access["network_allowed"] and "Network access is allowed now" in detail
                        )
                        assert body["next_action"] == "RETRY_THE_REFUSED_STEP"
                        assert "network set" not in detail
                    assert registry.task(UUID(task)).record_hash == held
                    assert attempts(task) == counted
            network(False)

        plan, stop = tmp_path / "plan.json", tmp_path / "stop.json"
        # The stop: the network closed, and a session the workspace's data does not hold.
        network(False)
        code, planned = cli(
            "research-update", "plan", "--package", PACKAGE, "--through", "2026-09-11",
            "--output", str(plan),
        )  # fmt: skip
        assert code == 0, planned
        assert planned["data"]["source_access"] == "EXPLICIT_NETWORK_REQUIRED", planned
        monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
        code, stopped = cli(
            "research-update", "run", "--from", str(plan), "--wait", "--output", str(stop)
        )
        task = stopped["data"]["task_id"]
        seen = _seen(stopped, registry, task)
        stopped_code = stopped["failure_code"]
        assert stopped_code.startswith(
            "research_update.input_source_access_not_admitted:2026-09-11,DATA"
        ), seen
        assert "missing market data through session 2026-09-11" in stopped["detail"], seen
        assert (code, stopped["failure_code"]) == (2, stopped_code), seen
        assert stopped["data"]["network_access"]["decided_by"] == "OPERATOR_OFFLINE_SWITCH", seen
        assert "restart the idle Host" in stopped["detail"], seen
        assert "network set" not in stopped["detail"], seen
        # Its read names the way its words give, bound to the stopped Task's own plan.
        assert stopped["data"]["next_requests"]["resume"] == {
            "operation": "RESEARCH_UPDATE_RUN",
            "update_plan_hash": planned["data"]["update_plan_hash"],
        }
        # V599: READBACK must name the same provider work as STATUS's wait,
        # while keeping both the person's network request and the bound resume.
        code, read = cli("research-update", "show", "--task", task)
        assert (code, read["failure_code"]) == (2, stopped_code), read
        assert "missing market data through session 2026-09-11" in read["detail"]
        assert "2026-09-11,DATA" not in read["detail"]
        assert read["next_requests"]["network"] == {"operation": "NETWORK_ACCESS"}
        assert read["next_requests"]["resume"] == stopped["next_requests"]["resume"]
        held = registry.task(UUID(task)).record_hash
        check_ways(task, ("research-update", "show", "--task", task), ("task", "show", task))
        # The rerun while the network stays closed is refused by that stop; nothing moves.
        monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
        network(False)
        code, refused = cli("research-update", "run", "--from", str(plan))
        assert (code, refused["failure_code"]) == (2, stopped_code), refused
        assert registry.task(UUID(task)).record_hash == held
        # A person allows the network: the offered resume takes up the same Task, and the wait
        # follows it rather than ending at its admission.
        monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
        network(True)
        code, resumed = cli("request", "--from", str(stop), "--action", "resume", "--wait")
        seen = _seen(resumed, registry, task)
        admission = resumed["data"]["admission"]
        assert admission["task_id"] == task, seen
        assert admission["lifecycle"] in {"QUEUED", "RUNNING"}, seen
        # It ran on past the network from the stage it stopped in, to the provider this test
        # refuses, whose refusal defers it as it defers a data update (V601); its verified first
        # stage was kept, and its stop is its own record.
        assert (code, resumed["data"]["task_id"]) == (3, task), seen
        assert resumed["data"]["lifecycle"] == "DEFERRED", seen
        assert attempts(task) == {"verify_request": 1, "update_inputs": 2}
        # A deferred update holds the workspace's one active place: cancelled, it frees it.
        code, cancelled = cli("task", "cancel", task)
        assert registry.task(UUID(task)).lifecycle.value == "CANCELLED", cancelled

        # The data update's own door: its Task, admitted while the network was open, stopped by
        # its closing before it ran, is resumed by the plan run again once a person reopens it.
        stopped_code = "workspace_data_update.source_access_not_admitted"
        monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
        network(True)
        data_plan = tmp_path / "data-plan.json"
        code, planned_data = cli("data-update", "plan", "--output", str(data_plan))
        assert code == 0, planned_data
        owner = live.operations.data_update
        assert owner is not None
        admitted = owner.admit(owner.prepare(str(planned_data["data"]["plan_hash"])))
        monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
        network(False)
        owner.execute(admitted.task_id)
        data_task = str(admitted.task_id)
        assert registry.task(admitted.task_id).failure_code == stopped_code
        code, refused = cli("data-update", "run", "--from", str(data_plan))
        assert (code, refused["failure_code"]) == (2, stopped_code), refused
        check_ways(
            data_task, ("data-update", "show", "--task", data_task), ("task", "show", data_task)
        )
        monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
        network(True)
        code, resumed = cli("data-update", "run", "--from", str(data_plan), "--wait")
        seen = _seen(resumed, registry, data_task)
        assert resumed["data"]["admission"]["task_id"] == data_task, seen
        # Past the network to the provider this test refuses, whose refusal defers a data update
        # to its retry time, as it defers any.
        assert (code, resumed["data"]["task_id"]) == (3, data_task), seen
        assert resumed["data"]["lifecycle"] == "DEFERRED", seen
        assert attempts(data_task) == {"validate_update_request": 1, "maintain_data_feature": 2}


@pytest.mark.real_evidence
def test_an_update_the_provider_defers_waits_with_it_and_resumes_once_due(
    tmp_path: Path, evidence_roots, monkeypatch, capsys
) -> None:
    """An update the provider defers waits with it and resumes once due."""

    clock = [datetime(2026, 9, 14, 23, tzinfo=UTC)]
    provider = recording_provider(now=clock[0])
    provider.unavailable = True
    # A recorded signal that the day's bars are not final yet: the provider asks to wait.
    provider.unavailable_code = "data.daily_bar_not_final"
    root = evidence_roots.require("ls1_daily_flows")
    with _forward_copy(
        tmp_path, root, monkeypatch, capsys, provider=provider, clock=lambda: clock[0]
    ) as (live, cli, attempts):
        registry = live.session.task_control_registry
        plan, deferred_at = tmp_path / "plan.json", tmp_path / "deferred.json"
        code, planned = cli(
            "research-update", "plan", "--package", PACKAGE, "--through", "2026-09-11",
            "--output", str(plan),
        )  # fmt: skip
        assert code == 0, planned
        code, deferred = cli(
            "research-update", "run", "--from", str(plan), "--wait", "--output", str(deferred_at)
        )
        # It waits with the provider: deferred, with its data stage's retry time, its read
        # saying so and offering the resume bound to its own plan.
        task, retry = deferred["data"]["task_id"], deferred["data"]["retry_after_at"]
        seen = _seen(deferred, registry, task)
        assert (code, deferred["data"]["lifecycle"]) == (3, "DEFERRED"), seen
        assert retry is not None, seen
        assert deferred["data"]["next_requests"]["resume"] == {
            "operation": "RESEARCH_UPDATE_RUN",
            "update_plan_hash": planned["data"]["update_plan_hash"],
        }
        asked, held = len(provider.calls), registry.task(UUID(task)).record_hash
        # A rerun before its time is refused with that time, and asks nothing of the provider.
        code, early = cli("research-update", "run", "--from", str(plan))
        assert (code, early["failure_code"]) == (2, "workspace_data_update.retry_not_due"), early
        assert early["data"]["retry_after_at"] == retry, early
        assert len(provider.calls) == asked
        assert registry.task(UUID(task)).record_hash == held
        # Once the time has passed, the resume takes up the same Task from the stage it deferred
        # in and asks the provider again, whose day is still not final: it defers again.
        clock[0] = datetime.fromisoformat(retry) + timedelta(minutes=1)
        code, resumed = cli("request", "--from", str(deferred_at), "--action", "resume", "--wait")
        seen = _seen(resumed, registry, task)
        assert resumed["data"]["admission"]["task_id"] == task, seen
        assert (code, resumed["data"]["task_id"]) == (3, task), seen
        assert resumed["data"]["lifecycle"] == "DEFERRED", seen
        assert len(provider.calls) > asked
        assert attempts(task) == {"verify_request": 1, "update_inputs": 2}


def _held_data(provider: list[HeldDataProvider]) -> Callable[[Path], HeldDataProvider]:
    """The copy's own bars through Monday 2026-09-14, refused until a test says the provider's
    bars are final (V604)."""

    def build(workspace: Path) -> HeldDataProvider:
        provider.append(HeldDataProvider(workspace, through=date(2026, 9, 14)))
        provider[0].unavailable = True
        # A recorded signal that the day's bars are not final yet: the provider asks to wait.
        provider[0].unavailable_code = "data.daily_bar_not_final"
        return provider[0]

    return build


def _until(found: Callable[[], object], what: str, live: LocalPortfolioWebSession) -> Any:
    """What `found` returns once it is something, read every second within half an hour; else
    every Task the Host holds, with its worker's failure where its command raised one."""
    end = time.monotonic() + 1800.0
    while time.monotonic() < end:
        if value := found():
            return value
        time.sleep(1)
    tasks = [
        (str(t.task_id), t.task_kind, t.lifecycle.value, t.failure_code)
        + ((failure,) if (failure := live.dispatcher.failure(t.task_id)) else ())
        for t in live.session.task_control_registry.tasks()
        if t.task_kind != "study_verification_sweep"
    ]
    raise AssertionError(f"{what}: {json.dumps(tasks, indent=1)}")


@pytest.mark.real_evidence
def test_the_daily_update_resumes_its_deferral_then_goes_on_to_the_next_session(
    tmp_path: Path, evidence_roots, monkeypatch, capsys
) -> None:
    """V604: the real Host's daily automation resumes Friday's deferral at its retry time,
    publishes Friday once the provider's bars are final, and plans Monday on that publication.
    Nothing queues behind the deferral. Offline: the provider serves the copy's own bars."""

    clock = [datetime(2026, 9, 11, 22, tzinfo=UTC)]  # Friday's bars are due
    held: list[HeldDataProvider] = []
    root = evidence_roots.require("ls1_daily_flows")
    with _forward_copy(
        tmp_path, root, monkeypatch, capsys, provider_from=_held_data(held), clock=lambda: clock[0]
    ) as (live, cli, attempts):
        provider, registry = held[0], live.session.task_control_registry
        automation = live.operations.automation
        assert automation is not None

        def updates() -> list[Any]:
            return [t for t in registry.tasks() if t.task_kind == ADVANCEMENT_TASK_KIND]

        def daily() -> dict[str, Any]:
            return cli("automation", "show")[1]["data"]

        def armed(at: str) -> bool:
            # A failed cycle ends the wait with the automation's read, never a timeout.
            read = daily()
            assert read["failure_code"] is None, json.dumps(read, indent=1, default=str)
            return bool(read["next_due_at"] == at)

        def wake(at: datetime) -> None:
            # The scheduler's own timer firing at `at`: it signals only work due by then.
            clock[0] = at
            assert automation.wake.signal_due_work(at), json.dumps(daily(), default=str)

        on = _json(
            live,
            "/api/research-update/automation",
            method="POST",
            payload={"automation_enabled": True, "automation_package_ids": [PACKAGE]},
        )
        assert on["status"] == "ENABLED_SERVICE_LIFETIME", on
        friday = _until(
            lambda: next((t for t in updates() if t.lifecycle.value == "DEFERRED"), None),
            "Friday's update deferred",
            live,
        )
        _, read = cli("research-update", "show", "--task", str(friday.task_id))
        retry, seen = read["data"]["retry_after_at"], _seen(read, registry, str(friday.task_id))
        assert (read["data"]["status"], retry is not None) == ("DEFERRED", True), seen
        # The automation wakes at the deferral's retry time, not at Monday's session, and nothing
        # queues behind the deferral, the Host's own sweep among them.
        _until(lambda: armed(retry), "armed at the retry time", live)
        assert [t.task_kind for t in registry.tasks() if t.lifecycle.value == "QUEUED"] == []
        # By its retry time the provider's bars are final: the automation resumes the same Task
        # from the stage it deferred in, and it publishes Friday.
        provider.unavailable = False
        wake(datetime.fromisoformat(retry) + timedelta(minutes=1))
        _until(
            lambda: registry.task(friday.task_id).lifecycle.value == "SUCCEEDED",
            "Friday published",
            live,
        )
        assert attempts(str(friday.task_id))["update_inputs"] == 2
        # Monday's bars are ready: its cycle plans Monday from Friday's publication, with no
        # person acting. Monday's positions published are (c)'s to read.
        _, published = cli("research-update", "show", "--task", str(friday.task_id))
        friday_publication = published["data"]["publication"]["content_hash"]
        _until(lambda: armed("2026-09-14T22:00:00+00:00"), "armed at Monday", live)
        wake(datetime(2026, 9, 14, 22, tzinfo=UTC))
        monday = _until(
            lambda: next((t for t in updates() if t.task_id != friday.task_id), None),
            "Monday planned",
            live,
        )
        planned = monday.input.payload["plan"]
        assert (planned["target"], planned["parent_hash"]) == (
            "2026-09-14",
            friday_publication,
        ), planned
        assert len(updates()) == 2, [(t.task_id, t.lifecycle.value) for t in updates()]
        assert daily()["status"] == "ENABLED_SERVICE_LIFETIME"
        # The person turns the daily update off: nothing reads Monday's update as the Host stops.
        _json(
            live,
            "/api/research-update/automation",
            method="POST",
            payload={"automation_enabled": False, "automation_package_ids": []},
        )
        cli("task", "cancel", str(monday.task_id))


@pytest.mark.real_evidence
def test_a_plan_answers_the_update_that_waits_and_a_cancelled_one_leaves_a_way_on(
    tmp_path: Path, evidence_roots, monkeypatch, capsys
) -> None:
    """A plan answers the update that waits and a cancelled one leaves a way on."""

    clock = [datetime(2026, 9, 14, 23, tzinfo=UTC)]
    held: list[HeldDataProvider] = []
    root = evidence_roots.require("ls1_daily_flows")
    with _forward_copy(
        tmp_path, root, monkeypatch, capsys, provider_from=_held_data(held), clock=lambda: clock[0]
    ) as (live, cli, _attempts):
        provider, registry = held[0], live.session.task_control_registry
        plan, deferred_at = tmp_path / "plan.json", tmp_path / "deferred.json"
        code, planned = cli(
            "research-update", "plan", "--package", PACKAGE, "--through", "2026-09-11",
            "--output", str(plan),
        )  # fmt: skip
        assert code == 0, planned
        code, deferred = cli(
            "research-update", "run", "--from", str(plan), "--wait", "--output", str(deferred_at)
        )
        task, retry = deferred["data"]["task_id"], deferred["data"]["retry_after_at"]
        seen = _seen(deferred, registry, task)
        assert (code, deferred["data"]["lifecycle"], retry is not None) == (3, "DEFERRED", True), (
            seen
        )
        # The next plan of the strategy, for the latest session, answers the update that waits.
        later = tmp_path / "later.json"
        code, again = cli("research-update", "plan", "--package", PACKAGE, "--output", str(later))
        assert code == 0, again
        assert again["data"]["update_plan_hash"] == planned["data"]["update_plan_hash"], again
        assert again["data"]["target_session"] == "2026-09-11", again
        code, early = cli("research-update", "run", "--from", str(later))
        assert (code, early["failure_code"]) == (2, "workspace_data_update.retry_not_due"), early
        assert early["data"]["retry_after_at"] == retry, early
        updates = [t.task_id for t in registry.tasks() if t.task_kind == ADVANCEMENT_TASK_KIND]
        assert updates == [UUID(task)], updates
        # A study verification asked for meanwhile waits for the running place, owned: the
        # Supervisor's pass finds nothing parked.
        code, sweep = cli("study", "verify-all")
        verifying = sweep["data"]["task_id"]
        assert registry.task(UUID(verifying)).lifecycle.value == "QUEUED", sweep
        live.operations.supervisor.supervise_once()
        code, waiting = cli("task", "show", verifying)
        seen = _seen(waiting, registry, verifying)
        assert waiting["data"]["lifecycle"] == "QUEUED", seen
        assert not waiting["data"].get("incident"), seen
        # The person cancels the deferral: the verification it held back runs.
        code, cancelled = cli("task", "cancel", task)
        assert registry.task(UUID(task)).lifecycle.value == "CANCELLED", cancelled
        _until(
            lambda: registry.task(UUID(verifying)).lifecycle.value == "SUCCEEDED",
            "the verification ran",
            live,
        )
        # The provider's deferral it left holds the workspace's inputs: the next plan is refused,
        # named by their state, its way on the data update that settles them.
        refusal = tmp_path / "refusal.json"
        code, refused = cli(
            "research-update", "plan", "--package", PACKAGE, "--output", str(refusal)
        )
        seen = _seen(refused, registry, None)
        assert refused["failure_code"] == (
            "strategy_score.workspace_inputs_not_ready:DATA_REVIEW_PENDING"
        ), seen
        assert code == 2 and "DATA_REVIEW_PENDING" in refused["detail"], seen
        assert refused["data"]["next_requests"] == {
            "data_update": {"operation": "DATA_UPDATE_PLAN"}
        }, seen
        data_plan = tmp_path / "data-plan.json"
        code, planned_data = cli(
            "request", "--from", str(refusal), "--action", "data_update", "--output", str(data_plan)
        )
        assert code == 0 and planned_data["data"]["status"] == "PLANNED", planned_data
        code, waits = cli("data-update", "run", "--from", str(data_plan), "--wait")
        data_task, data_retry = waits["data"]["task_id"], waits["data"]["retry_after_at"]
        seen = _seen(waits, registry, data_task)
        assert (code, waits["data"]["lifecycle"]) == (3, "DEFERRED"), seen
        # The data door's next plan answers the data update that waits.
        resumed_plan = tmp_path / "data-again.json"
        code, again = cli("data-update", "plan", "--output", str(resumed_plan))
        assert again["data"]["plan_hash"] == planned_data["data"]["plan_hash"], again
        # The provider's bars are final by its retry time: its run resumes the same Task, which
        # settles the workspace's inputs.
        provider.unavailable = False
        clock[0] = datetime.fromisoformat(data_retry) + timedelta(minutes=1)
        code, settled = cli("data-update", "run", "--from", str(resumed_plan), "--wait")
        seen = _seen(settled, registry, data_task)
        assert settled["data"]["task_id"] == data_task, seen
        assert registry.task(UUID(data_task)).lifecycle.value == "SUCCEEDED", seen
        # Monday's update plans on the settled inputs and publishes its positions.
        monday = tmp_path / "monday.json"
        code, planned = cli(
            "research-update", "plan", "--package", PACKAGE, "--output", str(monday)
        )
        assert (code, planned["data"]["target_session"]) == (0, "2026-09-14"), planned
        code, ran = cli("research-update", "run", "--from", str(monday), "--wait")
        seen = _seen(ran, registry, ran["data"].get("task_id"))
        assert (code, ran["data"]["lifecycle"]) == (0, "SUCCEEDED"), seen
        code, shown = cli("research-update", "show", "--package", PACKAGE)
        seen = _seen(shown, registry, ran["data"]["task_id"])
        assert shown["data"]["update"]["target_session"] == "2026-09-14", seen
        assert shown["data"]["position_rows"], seen


def test_current_performance_follows_exact_books_daily_publications_and_keeps_history_fixed(
    tmp_path, monkeypatch
):
    """CONTRACT: the current view reads the book/cost-bound realized window separately."""
    from alphalattice.control.product_host.composition import portfolio_result_context
    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from alphalattice.foundation.causal_outcomes.execution.readers import planned_local_qa_schedule
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        PortfolioDecisionCheckpoint,
        PortfolioUpdatePublication,
        advance_decision_state,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
        REBOUND_RETURN_BOOK_RECIPE,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash
    from tests.portfolio_strategy_lab.activation_review_support import activation_review_host
    from tests.portfolio_strategy_lab.synthetic_numerical import (
        build_numerical,
        prepared_for,
        snapshot_for,
    )

    with activation_review_host(tmp_path / "workspace") as book:
        live = book.live
        package = next(iter(live.application.resolver.installed_packages().values()))
        n = build_numerical(
            installed_package=package,
            book_recipe=REBOUND_RETURN_BOOK_RECIPE,
            first_day=next(
                point.formation_session
                for point in planned_local_qa_schedule(book.last_session, date(2026, 9, 30))
                if point.formation_session > book.last_session
            ),
        )
        n.checkpoint = PortfolioDecisionCheckpoint.create(
            **{
                k: getattr(n.checkpoint, k)
                for k in PortfolioDecisionCheckpoint.model_fields
                if k not in {"content_hash", "purpose", "activated_at", "book_task_id"}
            },
            purpose="PERSON_ACTIVATED_FORWARD_RESEARCH",
            activated_at=live.clock(),
            book_task_id=book.task_id,
        )
        live.application.ledger.publish_decision_checkpoint(n.checkpoint)
        html_hash, _ = live.application.ledger.publish_html("<html>Synthetic daily holdings</html>")
        prefix, publications, previous = [], [], None
        for offset in range(5):
            index = n.first + offset
            previous = advance_decision_state(
                checkpoint=n.checkpoint,
                previous=previous,
                prepared=prepared_for(n, index),
                observed=snapshot_for(n, index),
                plan_hash=canonical_hash(offset),
                published_at=live.clock(),
            )
            previous = PortfolioUpdatePublication.create(
                **{
                    k: getattr(previous, k)
                    for k in PortfolioUpdatePublication.model_fields
                    if k not in {"content_hash", "html_hash"}
                },
                html_hash=html_hash,
            )
            live.application.ledger.publish_decision_update(previous)
            prefix.append(previous)
            publications.append(
                live.operations.updates.publication_body(
                    n.checkpoint, previous, tuple(prefix), task_id=UUID(int=offset + 1)
                )
            )
        current = publications[3]
        original = PortfolioResearchOperations.execute
        requests = []

        def read(owner, request, **kwargs):
            if request.operation in {"RESEARCH_UPDATE_READBACK", "PORTFOLIO_UPDATE_READBACK"}:
                requests.append(request)
                assert request.strategy_package_id == package.strategy_id
                return (
                    current
                    if request.operation == "RESEARCH_UPDATE_READBACK"
                    else {"status": "NO_PORTFOLIO_UPDATE"}
                )
            return original(owner, request, **kwargs)

        monkeypatch.setattr(PortfolioResearchOperations, "execute", read)
        monkeypatch.setattr(
            portfolio_result_context,
            "forward_update_read_requests",
            lambda _operations, **_selection: (
                PortfolioResearchOperationRequest(
                    operation="RESEARCH_UPDATE_READBACK",
                    task_id=UUID(current["task_id"]),
                    strategy_package_id=package.strategy_id,
                ),
            ),
        )
        path = (
            f"/api/workbench/portfolio?task_id={book.task_id}&portfolio_session={book.last_session}"
        )
        pinned = _json(live, path)
        before_holdings_read = len(requests)
        selected_holdings = _json(live, path + "&scope=holdings")
        assert len(requests) == before_holdings_read  # No daily/model closure for a date change.
        assert "rolling_performance" not in selected_holdings
        assert "forward_holdings" not in selected_holdings
        assert selected_holdings["subject"] == pinned["subject"]
        assert selected_holdings["holdings"] == pinned["holdings"]
        first = _json(live, path + "&performance=latest")
        assert (
            live.operations.execute(
                PortfolioResearchOperationRequest(
                    operation="PORTFOLIO_READBACK",
                    task_id=book.task_id,
                    portfolio_session=book.last_session.isoformat(),
                    performance="latest",
                ),
                caller="EXTERNAL_AUTOMATION",
            )
            == first
        )
        assert first["forward_performance"]["available"] is True
        assert len(first["forward_performance"]["series"]) == 2
        assert (
            first["forward_performance"]["cost_bps_per_side"]
            == pinned["declaration"]["cost_bps_per_side"]
        )
        assert first["forward_performance"]["selected_window_metric_provenance"][
            "source_book_task_id"
        ] == str(book.task_id)
        assert (
            first["forward_holdings"]["publication_hash"] == current["publication"]["content_hash"]
        )
        assert first["forward_holdings"]["source_book_task_id"] == str(book.task_id)
        assert first["forward_holdings"]["formation_session"] != first["subject"]["session"]
        assert first["forward_holdings"]["holdings"] != first["holdings"]
        current = publications[4]
        second = _json(live, path + "&performance=latest")
        assert len(second["forward_performance"]["series"]) == 3
        assert (
            second["forward_performance"]["selected_window_metrics"]
            != first["forward_performance"]["selected_window_metrics"]
        )
        assert (
            second["forward_holdings"]["publication_hash"]
            != first["forward_holdings"]["publication_hash"]
        )
        assert (
            second["forward_holdings"]["observed_through"]
            != first["forward_holdings"]["observed_through"]
        )
        reopened = _json(live, path)
        for key in ("subject", "holdings", "holdings_basis", "position", "metrics", "series"):
            assert second[key] == reopened[key] == pinned[key]
        assert all(request.strategy_package_id == package.strategy_id for request in requests)
        current = {
            **current,
            "realized_performance": {
                **current["realized_performance"],
                "source_book_task_id": str(UUID(int=999)),
            },
        }
        # Identical strategy names still cannot borrow another book's activation history.
        assert _json(live, path + "&performance=latest")["forward_performance"] == {
            "status": "NO_REALIZED_FORWARD_PUBLICATION",
            "available": False,
        }


def test_activation_over_the_real_storage_cap_is_a_worded_http_refusal(tmp_path: Path) -> None:
    """Activation over the real storage cap is a worded HTTP refusal."""
    from alphalattice.control.product_host.composition.research_workspace import (
        read_research_workspace_manifest,
    )
    from alphalattice.control.product_host.storage.inventory import (
        managed_file_inventory,
        unique_managed_bytes,
    )
    from alphalattice.interface.local_application.cli_contract import (
        ANSWER_LANGUAGE,
        refusal_words,
        worded,
    )
    from tests.portfolio_strategy_lab.activation_review_support import activation_review_host
    from tests.portfolio_strategy_lab.storage_capacity_support import sparse_capacity_ballast

    with activation_review_host(tmp_path / "workspace") as book:
        live = book.live
        cap = live.operations.storage.capacity_cap_bytes()
        configured = _json(
            live,
            "/api/workspace/storage/cap",
            method="POST",
            payload={"storage_cap_bytes": str(cap)},
        )
        assert configured["capacity"]["cap_bytes"] == cap
        manifest = read_research_workspace_manifest(live.workspace)
        checkpoints = live.application.ledger.root / "decision-checkpoints"
        before = set(checkpoints.glob("*.json"))
        with sparse_capacity_ballast(live.workspace, cap_bytes=cap) as ballast:
            assert ballast.stat().st_size > cap
            status, _headers, raw = _request(
                live,
                "/api/strategy/activate",
                method="POST",
                payload={"task_id": str(book.task_id)},
            )
            refused = json.loads(raw)
            code = "storage.managed_capacity_exceeded"
            assert (status, refused["status"], refused["failure_code"]) == (
                200,
                "REFUSED",
                code,
            ), refused
            assert refused["detail"] == refusal_words(code)["detail"]
            assert refused["next_requests"] == {
                "cap": {"operation": "STORAGE_CAP_SHOW"},
                "storage": {"operation": "STORAGE_READBACK"},
                "cleanup": {"operation": "STORAGE_PLAN"},
            }
            assert unique_managed_bytes(managed_file_inventory(live.workspace)) > cap
            language = ANSWER_LANGUAGE.set("zh")
            try:
                assert worded(refused["detail"]) == worded(refusal_words(code)["detail"])
                assert worded(refused["detail"]) != refused["detail"]
            finally:
                ANSWER_LANGUAGE.reset(language)
            assert set(checkpoints.glob("*.json")) == before
            assert read_research_workspace_manifest(live.workspace) == manifest
        assert not ballast.exists()
