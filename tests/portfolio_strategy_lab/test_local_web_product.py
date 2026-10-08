"""Gate 9C: the local Web product, proved against a real loopback service.

Every test here boots the actual composition -- workspace session, application,
dispatcher, HTTP service -- and talks to it over a socket. Asserting against a
handler function would prove the handler; the questions this Gate asks are about
a listener, a session cookie, an `Origin` header and a browser that can be closed,
and none of those exist above the socket.

The fixture is the same isolated synthetic resolver the Portfolio suites use. No
Provider, no network beyond loopback, no protected evidence.
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import suppress
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    HANDLED_OPERATION_ROUTES,
    OPERATION_ROUTES,
    LocalPortfolioWebSession,
    LocalWebSessionError,
)
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    PortfolioRunCommand,
)
from alphalattice.control.task_control.queue import write_queue_setting
from alphalattice.control.task_control.registry import TaskQueueFull
from alphalattice.interface.local_application.dispatcher import CommandAdmission
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from alphalattice.interface.local_application.web import (
    BROWSER_REFUSED_PORTS,
    CONTENT_SECURITY_POLICY,
    SESSION_COOKIE,
    SESSION_HEADER,
    LocalWebApplication,
    LocalWebService,
    bind_browser_safe,
    session_cookie_name,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioResearchSpec,
)
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    PortfolioResearchTaskAdapter,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.static import (
    format_book_change,
    format_book_weight,
)
from tests.portfolio_strategy_lab.local_web_support import (
    InstalledAgent,
    _InterruptsOnce,
    _json,
    _manifest,
    _request,
    _resolved,
    _Resolver,
)

# the served pages and the bundle-reading tests read the built assets: built once per session
pytestmark = pytest.mark.usefixtures("workbench_build")

DEVELOPMENT_SESSIONS = _resolved().workspace.formation_sessions


def _agent(
    session: LocalPortfolioWebSession, request: PortfolioResearchAgentRequest
) -> dict[str, Any]:
    assert session.operations is not None
    bridge = InstalledAgent(session.operations)
    return json.loads(bridge.invoke(request))


def test_settings_storage_cap_uses_the_live_owner_and_preserves_invalid_input(
    live: LocalPortfolioWebSession,
) -> None:
    """V680: the Settings route sets execution configuration, without creating a Task."""
    initial = _json(live, "/api/workspace/storage/cap")
    assert initial["capacity"]["setting"]["cap_bytes"] == "auto"
    cap = str(20 * 1024**3)
    changed = _json(
        live, "/api/workspace/storage/cap", method="POST", payload={"storage_cap_bytes": cap}
    )
    assert changed["status"] == "CONFIGURED"
    assert changed["capacity"]["cap_bytes"] == int(cap)
    assert changed["capacity"]["setting"]["chosen_by"] == "HUMAN"
    status, _headers, body = _request(
        live, "/api/workspace/storage/cap", method="POST", payload={"storage_cap_bytes": "0"}
    )
    assert status == 200 and json.loads(body)["failure_code"] == "storage.cap_setting_invalid"
    assert _json(live, "/api/workspace/storage/cap")["capacity"]["cap_bytes"] == int(cap)
    assert _json(live, "/api/tasks")["tasks"] == []


def _run_to_completion(session: LocalPortfolioWebSession) -> str:
    """Admit one background run and wait for the task, not for the request."""

    admitted = _json(session, "/api/run", method="POST", payload={})
    if admitted["disposition"] == "REUSED_EXACT":
        assert admitted["task_id"] is None
        return str(admitted["result_hash"])
    assert admitted["disposition"] == "ADMITTED"
    session.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    status = _json(session, f"/api/status?task_id={admitted['task_id']}")
    assert status["lifecycle"] == "SUCCEEDED", status
    results = _json(session, "/api/results")["results"]
    assert results
    return str(results[0]["result_hash"])


# =========================================================== Agent parity


def test_cli_attaches_to_the_existing_host_and_reopens_exact_results(live, tmp_path):
    from alphalattice.interface.local_application.client import (
        LocalResearchClient,
        LocalResearchConnection,
    )

    script = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"

    def cli(*arguments, stdin=None):
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--workspace",
                str(live.workspace),
                "--view",
                "full",
                *arguments,
            ],
            capture_output=True,
            text=True,
            input=stdin,
            timeout=20,
        )
        assert live.web.application.session_token not in result.stdout
        assert live.web.application.external_token not in result.stdout
        return result.returncode, json.loads(result.stdout)

    client = LocalResearchClient(live.workspace)
    view = client.request()
    assert view["caller"] == "EXTERNAL_AUTOMATION" and "session_token" not in view
    assert "research_context" not in view
    # The CLI's copy is the Host's answer and the read it names (V571).
    tasks = cli("request", "--file", "-", stdin="operation: TASKS")[1]["data"]
    assert tasks.pop("read_request") == {"operation": "TASKS"}
    assert tasks == client.request({"operation": "TASKS"})
    code, conflict = cli("request", "--choices", "-", "--from", "-", stdin="{}")
    assert code == 1 and conflict["failure_code"] == "local_client.document_multiple_stdin_sources"
    workspace = cli("workspace", "show")[1]["data"]
    assert workspace["workspace_id"] == live.workspace_id
    context = workspace["research_context"]
    assert context == _json(live, "/api/session?context=1")["research_context"]
    assert {item["action"] for item in context["actions"]} == {
        "update_data",
        "capture_input",
        "research",
    }
    # This legacy fixture composes the Host without an on-disk research manifest.
    assert context["inputs"]["status"] == "REFUSED"
    assert context["tasks"]["tasks"] == []
    assert context["actions"][0]["configured"] is (
        live.operations.data_update is not None
        and live.operations.workspace_manifest.data_update is not None
    )
    contract = cli("schema", "show", "EXPERIMENT_RUN")[1]["data"]
    assert contract["required"] == ["experiment_plan_hash"]
    # The operation is its own constant (the product, 3de3f325): no enum of the other
    # operations rides along, and nothing else is referenced.
    assert contract["properties"]["operation"] == {
        "type": "string",
        "const": "EXPERIMENT_RUN",
        "description": "This operation, by its name.",
    }
    assert contract["$defs"] == {}
    delivery_contract = cli("schema", "show", "EXPERIMENT_DELIVERY_EXPORT")[1]["data"]
    # the operation rides along as a constant
    assert set(delivery_contract["$defs"]) == {"ResearchDeliveryCommentary"}
    analyst_contract = cli("schema", "show", "EVIDENCE_ANALYSIS_SUBMIT")[1]["data"]
    assert "AlternativeEvidenceAnswerFinding" in analyst_contract["$defs"]
    assert "EvidenceDirection" in analyst_contract["$defs"]  # A transitive finding reference.
    assert "PortfolioReviewAnswer" not in analyst_contract["$defs"]
    for name, definition in analyst_contract["$defs"].items():
        assert definition == view["operation_schema"]["$defs"][name]
    malformed = client.request({"operation": "EXPERIMENT_RUN"})
    assert "experiment_plan_hash" in malformed["message"]
    refusal = tmp_path / "review.yaml"
    refusal.write_text("operation: CRO_REVIEW", encoding="utf-8")
    refusal_output = tmp_path / "refused-review.json"
    code, refused = cli("request", "--file", str(refusal), "--output", str(refusal_output))
    assert code == 2 and refused["data"]["disposition"].startswith("REFUSED")
    # A refusal is the owner's answer and is saved like any other (CLI-6).
    assert json.loads(refusal_output.read_text(encoding="utf-8")) == refused["data"]
    empty_assessment = tmp_path / "assessment.json"
    empty_assessment.write_text("{}", encoding="utf-8")
    before_review = len(live.session.task_control_registry.tasks())
    # The template-reading submit verbs are gone from the grammar (Z1): the command is a
    # usage error like any unknown one, and nothing reaches an owner.
    code, rejected = cli("review-submit", "--from", "x.json", "--answer", str(empty_assessment))
    assert code == 1 and rejected["failure_code"] == "local_client.usage_invalid"
    assert len(live.session.task_control_registry.tasks()) == before_review
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    assert cli("strategy-book", "preview", "--file", str(spec))[0] == 0
    sent = cli("strategy-book", "run", "--file", str(spec))[1]["data"]
    assert sent["disposition"] == "ADMITTED"
    code, completed = cli("task", "show", sent["task_id"], "--wait", "--max-wait", "10")
    assert code == 0 and completed["data"]["lifecycle"] == "SUCCEEDED"
    result_hash = client.request({"operation": "RESULTS"})["results"][0]["result_hash"]
    discovered = cli("workspace", "show")[1]["data"]["research_context"]
    assert any(
        entry["entry_id"] == f"result:{result_hash}"
        for entry in discovered["recent_research"]["entries"]
    )
    code, report = cli("result", "show", result_hash)
    assert code == 0
    assert "history=result%3A" + result_hash in report["local_web_url"]
    code, brief = cli("--view", "compact", "result", "show", result_hash)
    # The compact view shows the hash by its first twelve characters (V393).
    assert code == 0 and brief["data"]["result_hash"] == report["data"]["result_hash"][:12]
    assert brief["navigation"]["kind"] == "portfolio_result"
    review_entry = client.navigation({"operation": "EVIDENCE_CRO", "result_hash": result_hash}, {})
    assert review_entry["kind"] == "book_review_entry" and "review_result=" in review_entry["url"]
    dated = client.selected_url(
        {
            "operation": "EXPERIMENT_READBACK",
            "task_id": sent["task_id"],
            "portfolio_session": "2024-08-09",
        },
        {},
    )
    assert "portfolio_session=2024-08-09" in dated
    assert "history=experiment%3A" + sent["task_id"] in dated
    for lifecycle in ("QUEUED", "RUNNING", "BLOCKED", "CANCELLED"):
        navigation = client.navigation(
            {"operation": "EXPERIMENT_RUN"},
            {"task_id": sent["task_id"], "lifecycle": lifecycle},
        )
        assert navigation["kind"] == "task" and "?task=" in navigation["url"]
    assert (
        client.navigation(
            {"operation": "EXPERIMENT_RUN"},
            {"task_id": sent["task_id"], "lifecycle": "SUCCEEDED"},
        )["kind"]
        == "experiment_result"
    )
    exported = cli("result", "export", result_hash)[1]["data"]
    # The CLI's copy keeps the request it read with (V460); the rest is the owner's answer.
    assert exported.pop("read_request") == {"operation": "EXPORT", "result_hash": result_hash}
    assert exported == live.operations.export(result_hash)
    output = tmp_path / "export.json"
    saved = cli("result", "export", result_hash, "--output", str(output))
    # Saving prints the compact view; the file holds the full answer, with the request it read
    # with, so `--from` reads it again (V460).
    whole = json.loads(output.read_text())
    assert saved[0] == 0 and whole.pop("read_request")["result_hash"] == result_hash
    assert whole == live.operations.export(result_hash)
    original = output.read_bytes()
    assert cli("result", "export", result_hash, "--output", str(output))[0] == 1
    assert output.read_bytes() == original
    count = len(live.session.task_control_registry.tasks())
    old = client.connection
    live.stop()
    assert not LocalResearchConnection.path(live.workspace).exists()
    live.start()
    assert LocalResearchConnection.read(live.workspace).instance != old.instance
    client.connection = dataclasses.replace(old, url=live.url)
    assert "external_workspace_or_instance_mismatch" in str(client.request())
    reused = cli("strategy-book", "run", "--file", str(spec))[1]["data"]
    assert reused["disposition"] == "REUSED_EXACT" and reused["task_id"] is None
    assert len(live.session.task_control_registry.tasks()) == count
    assert cli("result", "show", result_hash)[1]["data"] == report["data"]


def test_compact_output_is_non_authoritative_and_saved_exports_stay_exact(tmp_path):
    from alphalattice.interface.local_application.client import _save_output, compact_display

    document = {
        "status": "REVIEW_PUBLISHED",
        "publication_hash": "a" * 64,
        "review": {"recommendation": {"route": "REQUEST_EVIDENCE_REFRESH", "limits": ["PARTIAL"]}},
        "html": "<p>Unicode Ω and source evidence</p>" * 10000,
        "rows": list(range(1000)),
    }
    original = json.dumps(document, ensure_ascii=False)
    body = {"json": original, "review_publication_hash": "a" * 64, "status": "REVIEW_PUBLISHED"}
    view = compact_display(body)
    assert view["representation"] == "COMPACT_DISPLAY_NOT_AN_AUTHORITY_DOCUMENT"
    assert view["data"]["review_publication_hash"] == "a" * 64
    assert view["data"]["json"]["preview"]["review"]["recommendation"]["limits"] == ["PARTIAL"]
    assert view["omitted_sections"] and len(json.dumps(view)) < len(original) / 10
    assert body["json"] == original
    output = tmp_path / "exact.json"
    _save_output(output, body, json.dumps(body).encode(), "json")
    assert output.read_bytes() == original.encode("utf-8")

    # A receipt may exhaust the display budget, but not before the actual
    # result and selected-position facts. This is display ordering, not a
    # larger budget or a changed artifact.
    structured = {
        "status": "EXPERIMENT_PUBLISHED",
        "receipt": {f"entry-{i}": {f"trace-{j}": j for j in range(12)} for i in range(40)},
        "result": {"annualized_return": 0.1, "anchor_relative_return": None},
        "position": {"session": "2024-08-12", "cash": 0.0},
        "data_quality": {"policy": "quarantine_listings", "effective_listing_count": 457},
        # What the result can claim and what its flow needs next, read first (V368, V367).
        "standing": {
            "comparison": "NOT_APPLICABLE",
            "reasons": {},
            "statements": ["No comparison.", "Ran.", "Passed.", "Development.", "Nothing."],
        },
        "prerequisites": {
            "flow": "BOOK",
            "missing": [],
            "next_requests": {"portfolio-draft": {"task_id": "t-1", "candidate_id": None}},
        },
    }
    bounded = compact_display(structured)
    for key in ("result", "position", "data_quality", "standing", "prerequisites"):
        assert bounded["data"][key] == structured[key]
    assert any(path.startswith("receipt") for path in bounded["omitted_sections"])
    compared = compact_display(
        {
            "left": {"task_id": "left", **structured},
            "right": {"task_id": "right", **structured},
            "claim": "NO_WINNER",
        }
    )
    for side in ("left", "right"):
        assert compared["data"][side]["task_id"] == side
        assert compared["data"][side]["result"] == structured["result"]


def test_a_compact_answer_is_one_read_and_names_each_part_it_left_out() -> None:
    """requirement (V402, the outside review's F6): the compact view bounded nodes, not bytes,
    and eight mappings of thirty 899-character strings printed 218,529 bytes with nothing
    marked. The display stays within its budget, one read on every agent host; it names each
    part it left out as `--section` reads it, keeps standing whole and a refusal whole when it
    fits, and pages a long list from the list itself."""
    from alphalattice.interface.local_application.client import (
        COMPACT_ANSWER_BYTES,
        answer_part,
        compact_display,
    )
    from alphalattice.protocols.actor_execution.bundles import BUNDLE_FILE_BYTES

    assert COMPACT_ANSWER_BYTES == BUNDLE_FILE_BYTES
    body: dict[str, Any] = {
        f"group{g}": {f"text{i:02d}": "x" * 899 for i in range(30)} for g in range(8)
    }
    body["standing"] = {"comparison": "NOT_APPLICABLE", "statements": ["Ran."] * 5}
    assert len(json.dumps(body)) > 200_000
    view = compact_display(body)
    assert len(json.dumps(view)) <= COMPACT_ANSWER_BYTES
    assert view["data"]["standing"] == body["standing"] and view["omitted_sections"]
    for part in view["omitted_sections"]:
        answer_part(body, part)
    assert len(json.dumps(compact_display(body, budget=8 * 1024))) <= 8 * 1024
    # The page links are kept whole and a book's activation is drawn before the bulk: what AX's
    # agents read back from saved answers (AGENT-TIME R2).
    linked = {
        **body,
        "local_web_url": "http://127.0.0.1:1/workbench.html#page=book",
        "navigation": {"url": "http://127.0.0.1:1/workbench.html#page=book"},
        "activation": {"status": "INACTIVE", "detail": "Offered.", "reasons": ["REVIEWED"]},
    }
    shown = compact_display(linked)["data"]
    assert shown["local_web_url"] == linked["local_web_url"]
    assert shown["navigation"] == linked["navigation"]
    assert shown["activation"] == linked["activation"]

    refusal = {"refused": True, "failure_code": "x.y", "allowed": [f"OP_{i}" for i in range(20)]}
    assert compact_display(refusal, whole=True)["data"] == refusal

    rows = {"items": [{"row": i, "text": "y" * 500} for i in range(200)]}
    assert compact_display(rows)["omitted_sections"] == ["items.3:"]
    first = compact_display(answer_part(rows, "items")["value"], section="items")
    shown = len(first["data"]["preview"])
    assert first["omitted_sections"] == [f"items.{shown}:"]
    assert len(json.dumps(first)) <= COMPACT_ANSWER_BYTES
    page = first["omitted_sections"][0]
    second = compact_display(answer_part(rows, page)["value"], section=page)
    assert second["data"]["preview"][0]["row"] == shown
    assert second["omitted_sections"] == [f"items.{shown + len(second['data']['preview'])}:"]
    assert answer_part({"files": {"manifest.json": 1}}, "files.manifest.json")["value"] == 1


def test_cli_document_reader_keeps_utf8_and_byte_limits_for_files_and_stdin(tmp_path, monkeypatch):
    from io import BytesIO, TextIOWrapper

    from alphalattice.interface.local_application.client import _text

    document = tmp_path / "document.yaml"
    for payload, expected in (
        ("备注".encode(), "备注"),
        (b"a\r\nb\rc", "a\nb\nc"),
        (b"x" * 7, "too_large"),
        (b"\xff", "unreadable"),
    ):
        document.write_bytes(payload)
        for path in (document, Path("-")):
            monkeypatch.setattr(sys, "stdin", TextIOWrapper(BytesIO(payload), encoding="utf-8"))
            if expected not in {"too_large", "unreadable"}:
                assert _text(path, limit=6) == expected
            else:
                with pytest.raises(ValueError, match="document_" + expected):
                    _text(path, limit=6)


def test_continuation_references_preserve_owner_fields_and_reject_the_wrong_template(tmp_path):
    from alphalattice.interface.local_application.client import continuation
    from alphalattice.interface.local_application.operations import fields
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    allowed = fields("EXPERIMENT_PLAN")[1]

    task_id = uuid4()
    assert (
        PortfolioResearchOperationRequest(
            operation="EXPERIMENT_DRAFT", task_id=task_id
        ).input_binding_hash
        is None
    )
    reply = tmp_path / "draft.json"
    edited = tmp_path / "edited.yaml"
    edited.write_text("experiment: {}\n", encoding="utf-8")
    template = {
        "operation": "EXPERIMENT_PLAN",
        "origin_task_id": str(task_id),
        "research_input_id": "same-input",
        "input_binding_hash": "a" * 64,
        "factor_task_id": str(uuid4()),
        "curation_receipt_hash": "b" * 64,
    }
    reply.write_text(
        json.dumps({"status": "DRAFT_READY", "plan_request": template}), encoding="utf-8"
    )
    # `study plan --from draft.json --file edited.yaml`: the text is the field.
    edits = {"experiment_yaml": edited.read_text(encoding="utf-8")}
    assert continuation("EXPERIMENT_PLAN", reply, edits, allowed) == {**template, **edits}
    # The draft's references are kept (CG3): another binding beside the declaration is refused,
    # and the binding as the compact view shows it is the same one.
    with pytest.raises(ValueError, match="bound_reference_override:input_binding_hash"):
        continuation("EXPERIMENT_PLAN", reply, {**edits, "input_binding_hash": "d" * 64}, allowed)
    same = {**edits, "input_binding_hash": "a" * 12}
    assert continuation("EXPERIMENT_PLAN", reply, same, allowed) == {**template, **edits}
    for status in ("DRAFT_INCOMPLETE", "INPUT_COMPILER_PREFLIGHT_PASSED", "PORTFOLIO_DRAFT_READY"):
        reply.write_text(json.dumps({"status": status, "plan_request": template}), encoding="utf-8")
        assert continuation("EXPERIMENT_PLAN", reply, edits, allowed) == {**template, **edits}
    with pytest.raises(ValueError, match="plan_file_or_draft_document_required"):
        continuation("EXPERIMENT_PLAN", reply, {}, allowed)
    for field in ("document", "template"):
        declaration = {"experiment": {"kind": "owner-declared-kind"}}
        reply.write_text(
            json.dumps({"status": "DRAFT_READY", "plan_request": template, field: declaration}),
            encoding="utf-8",
        )
        assert continuation("EXPERIMENT_PLAN", reply, {}, allowed) == {
            **template,
            "experiment_document": declaration,
        }
    template["operation"] = "EXPERIMENT_RUN"
    reply.write_text(
        json.dumps({"status": "DRAFT_READY", "plan_request": template}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="continuation_bundle_invalid"):
        continuation("EXPERIMENT_PLAN", reply, edits, allowed)
    # V355: a controls answer plans on its own input, its template or the declaration given;
    # an answer of another kind is named, never taken for a stale draft.
    controls = {
        "status": "READY",
        "research_input_id": "the-input",
        "input_binding_hash": "c" * 64,
        "template": {"experiment": {"kind": "factor.screening-development"}},
    }
    reply.write_text(json.dumps(controls), encoding="utf-8")
    # The input the controls named travels with its binding (V406).
    planned = {
        "operation": "EXPERIMENT_PLAN",
        "research_input_id": "the-input",
        "input_binding_hash": "c" * 64,
    }
    assert continuation("EXPERIMENT_PLAN", reply, {}, allowed) == {
        **planned,
        "experiment_document": controls["template"],
    }
    assert continuation("EXPERIMENT_PLAN", reply, edits, allowed) == {**planned, **edits}
    with pytest.raises(ValueError, match="bound_reference_override:input_binding_hash"):
        continuation("EXPERIMENT_PLAN", reply, {"input_binding_hash": "d" * 64}, allowed)
    reply.write_text(json.dumps({"status": "AVAILABLE", "experiments": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="plan_source_unsupported:AVAILABLE"):
        continuation("EXPERIMENT_PLAN", reply, edits, allowed)


def test_a_controls_template_is_saved_as_its_declaration(tmp_path, monkeypatch, capsys):
    """regression (V362): `feature controls --declaration` refused the template its own answer
    carries; an answer's template is written as its editable declaration, in the loader's
    dialect, and reads back the same, while an answer with a `yaml` part keeps writing that."""

    from alphalattice.interface.local_application import cli, client
    from alphalattice.protocols.research_authoring.selection import load_safe_yaml_document

    template = {
        "input_binding_hash": "a" * 64,
        "base_revision_hash": "b" * 64,
        "edits": [],
        "reason": "A formula factor of my own.",
    }
    answers = iter(
        (
            {"status": "READY", "template": template},
            {"status": "READY", "template": {"x": 1}, "yaml": "y: 2\n"},
        )
    )

    class Host:
        """The running Host, answering each controls request with the next answer."""

        def __init__(self, workspace, **_kwargs):
            self.workspace, self.goal = workspace, None

        def exchange(self, _document):
            body = next(answers)
            return body, json.dumps(body).encode("utf-8")

        def selected_url(self, *_args):
            return None

    monkeypatch.setattr(client, "LocalResearchClient", Host)

    def controls(declaration):
        arguments = ["study", "controls", "--save-declaration", str(declaration)]
        cli.main(["--workspace", str(tmp_path), "--view", "full", *arguments], serve=lambda _: 99)
        capsys.readouterr()

    saved = tmp_path / "feature.yaml"
    controls(saved)
    assert load_safe_yaml_document(saved.read_text(encoding="utf-8")) == template
    written = tmp_path / "experiment.yaml"
    controls(written)
    assert written.read_text(encoding="utf-8") == "y: 2\n"


def test_a_trial_the_owner_refuses_to_compare_claims_no_change(tmp_path):
    """regression (V363): a trial whose two Alpha studies the owner refused to compare (their
    score support differs) still reported a change between their saved metrics; it stands
    NOT_COMPARED, the refusal says why, and the saved metrics stay each on its own."""

    from datetime import UTC, datetime
    from uuid import uuid4

    from alphalattice.control.product_host.composition.feature_trials import (
        FeatureTrial,
        FeatureTrials,
    )

    factor, baseline, tried = uuid4(), uuid4(), uuid4()
    metrics = {baseline: 0.010, tried: 0.014}

    class Owner:
        def readback(self, task_id):
            if task_id == factor:
                return {
                    "result": {
                        "evidence_report": {"items": [{"factor_id": "formula_x"}]},
                        "redundancy_structure": {"pair_evidence": []},
                    }
                }
            return {"receipt": {"child_lineage": [{"candidate_id": "c1"}]}}

        def summary(self, task_id):
            return {
                "result": {"candidates": [{"candidate_id": "c1", "mean_rank_ic": metrics[task_id]}]}
            }

        def operate(self, request, *, caller):
            raise ValueError("alpha_research.saved_comparison_score_support_mismatch")

    trials = FeatureTrials(
        workspace=tmp_path,
        experiments=Owner(),
        feature_builds=None,
        dispatcher=None,
        clock=lambda: datetime(2026, 9, 30, tzinfo=UTC),
    )
    trial = FeatureTrial(
        trial_id="a" * 64,
        feature_plan_hash="b" * 64,
        baseline_task_id=str(baseline),
        baseline_alpha_task_id=str(baseline),
        caller="HUMAN",
        requested_at=datetime(2026, 9, 30, tzinfo=UTC),
        state="COMPLETED",
        outcome="COMPARED",
        steps={"FACTOR_STUDY": str(factor), "ALPHA_STUDY": str(tried)},
        feature_factor_ids=("formula_x",),
    )
    alpha = trials._comparison(trial)["alpha_without_and_with"]
    assert alpha["standing"] == "NOT_COMPARED" and "change" not in alpha
    refused = alpha["owner_comparison"]
    assert refused["failure_code"] == "alpha_research.saved_comparison_score_support_mismatch"
    assert "not compared" in refused["detail"]
    assert (alpha["without"]["mean_rank_ic"], alpha["with"]["mean_rank_ic"]) == (0.010, 0.014)


def test_trial_and_feature_listings_keep_readable_siblings_with_damaged_records(
    tmp_path, monkeypatch
):
    """regression (V633, TE12): one corrupt Feature trial hid its readable siblings, and
    FeatureExtensions did not forward the damaged-record refusal with an inspection route."""

    from datetime import UTC, datetime
    from pathlib import Path
    from types import SimpleNamespace

    from alphalattice.control.product_host.composition.feature_trials import (
        FeatureTrial,
        FeatureTrials,
    )
    from alphalattice.control.product_host.research_authoring.feature_extensions import (
        FeatureExtensions,
    )

    class Builds:
        def definitions(self):
            return SimpleNamespace(store=SimpleNamespace(root=tmp_path / "plans"))

    trials = FeatureTrials(
        workspace=tmp_path,
        experiments=None,
        feature_builds=Builds(),
        dispatcher=None,
        clock=lambda: datetime(2026, 10, 4, tzinfo=UTC),
    )
    trials.root.mkdir(parents=True)
    healthy = FeatureTrial(
        trial_id="a" * 64,
        feature_plan_hash="d" * 64,
        baseline_task_id="task-1",
        baseline_alpha_task_id="task-1",
        caller="HUMAN",
        requested_at=datetime(2026, 10, 1, tzinfo=UTC),
        state="RUNNING",
        feature_factor_ids=("formula_x",),
    )
    healthy_path = trials.root / f"{healthy.trial_id}.json"
    healthy_path.write_text(healthy.model_dump_json(), encoding="utf-8")
    malformed_path = trials.root / f"{'b' * 64}.json"
    malformed_path.write_text("{", encoding="utf-8")
    unreadable_path = trials.root / f"{'c' * 64}.json"
    unreadable_path.write_text("{}", encoding="utf-8")

    read_bytes = Path.read_bytes

    def read_one_fails(path):
        if path == unreadable_path:
            raise OSError("the local record could not be read")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_one_fails)

    listed = trials.listing()
    assert [row["feature_trial_id"] for row in listed["trials"]] == [healthy.trial_id]
    assert [row["feature_trial_id"] for row in listed["damaged"]] == [
        malformed_path.stem,
        unreadable_path.stem,
    ]
    for row in listed["damaged"]:
        assert row["status"] == "REFUSED"
        assert row["failure_code"] == "feature_trial.record_damaged"
        assert row["detail"]
        assert row["next_requests"]["trials"]["operation"] == "FEATURE_TRIALS"
        assert row["next_requests"]["storage"]["operation"] == "STORAGE_READBACK"
    assert "the local record could not be read" not in str(listed["damaged"])

    extensions = FeatureExtensions(
        tmp_path,
        lambda: datetime(2026, 10, 4, tzinfo=UTC),
        trials=trials,
        goals=None,
    )
    extension_listing = extensions.listing()
    assert extension_listing["factors"] == []
    assert extension_listing["damaged"] == listed["damaged"]


def test_a_trial_offers_its_own_read_and_once_completed_each_review():
    """regression (V391): AX13's agents copied a trial's id 6 times and a feature plan's hash 11
    times, since the trial's answer offered no request; it offers its read, and once completed
    each factor's review, all filled for --from."""

    from datetime import UTC, datetime

    from alphalattice.control.product_host.composition.feature_trials import (
        FeatureTrial,
        trial_requests,
    )

    trial = FeatureTrial(
        trial_id="a" * 64,
        feature_plan_hash="b" * 64,
        baseline_task_id="c",
        baseline_alpha_task_id="c",
        caller="HUMAN",
        requested_at=datetime(2026, 10, 1, tzinfo=UTC),
        feature_factor_ids=("formula_x",),
    )
    show = {"operation": "FEATURE_TRIAL_READBACK", "feature_trial_id": "a" * 64}
    assert trial_requests(trial) == {"show": show}
    done = trial.model_copy(update={"state": "COMPLETED"})
    assert trial_requests(done) == {
        "show": show,
        "review": {
            "operation": "FEATURE_REVIEW",
            "feature_plan_hash": "b" * 64,
            "feature_factor_id": "formula_x",
        },
    }
    two = done.model_copy(update={"feature_factor_ids": ("formula_x", "formula_y")})
    assert set(trial_requests(two)) == {"show", "review:formula_x", "review:formula_y"}


def test_a_review_packets_trial_says_whether_its_alpha_studies_were_compared(tmp_path):
    """requirement (U56, V363): the packet a person activates from carries each trial's Alpha
    standing; one not compared carries the owner's words and no change."""

    from datetime import UTC, datetime

    from alphalattice.control.product_host.composition.feature_trials import FeatureTrial
    from alphalattice.control.product_host.research_authoring.feature_extensions import (
        FeatureExtensions,
    )

    refused = {
        "status": "REFUSED",
        "failure_code": "alpha_research.saved_comparison_score_support_mismatch",
        "detail": "The two Alpha studies scored different sessions, names or available rows.",
    }

    class Ledger:
        def __init__(self, alpha):
            self.alpha = alpha

        def readback(self, trial_id):
            feature = {"out_of_sample_evidence": [], "correlation_with_present_factors": []}
            return {"comparison": {"feature": feature, "alpha_without_and_with": self.alpha}}

    trial = FeatureTrial(
        trial_id="a" * 64,
        feature_plan_hash="b" * 64,
        baseline_task_id="t-1",
        baseline_alpha_task_id="t-1",
        caller="HUMAN",
        requested_at=datetime(2026, 9, 30, tzinfo=UTC),
        state="COMPLETED",
        outcome="COMPARED",
        feature_factor_ids=("formula_x",),
    )

    def packet_row(alpha):
        extensions = FeatureExtensions(
            tmp_path, lambda: datetime(2026, 9, 30, tzinfo=UTC), trials=Ledger(alpha), goals=None
        )
        return extensions._trial(trial, "formula_x")

    not_compared = packet_row({"standing": "NOT_COMPARED", "owner_comparison": refused})
    assert (not_compared["alpha_standing"], not_compared["alpha_change"]) == ("NOT_COMPARED", None)
    assert not_compared["alpha_refusal"] == {
        "failure_code": refused["failure_code"],
        "detail": refused["detail"],
    }
    compared = packet_row(
        {"standing": "COMPARED", "owner_comparison": {"status": "COMPARABLE"}, "change": {"x": 1.0}}
    )
    assert (compared["alpha_standing"], compared["alpha_change"]) == ("COMPARED", {"x": 1.0})
    assert "alpha_refusal" not in compared


def test_every_result_states_one_standing_from_its_owners_marks():
    """requirement (V368): ran, contract, compared, evidence and activation were separate marks
    in separate answers, so an agent read a COMPLETED trial whose comparison was refused as
    done; every result answer states them as one standing, the comparison first, each value
    with the owner's code that holds it and its statement, and a value missing its code, or a
    code beside a value that names none, is refused."""

    from alphalattice.control.product_host.composition.result_standing import (
        ResultStanding,
        packet_standing,
        study_standing,
        trial_standing,
    )

    def marks(standing):
        return (
            standing.comparison,
            standing.execution,
            standing.contract,
            standing.evidence,
            standing.activation,
        )

    published = {"status": "EXPERIMENT_PUBLISHED", "research_lane": "PROMOTION"}
    study = study_standing(published)
    assert marks(study) == (
        "NOT_APPLICABLE",
        "SUCCEEDED",
        "PASSED",
        "DEVELOPMENT",
        "NOT_ACTIVATABLE",
    )
    assert study.reasons == {} and len(study.statements) == 5
    superseded = "research_lane.sample_scheme_superseded"
    explored = study_standing(
        {
            **published,
            "research_lane": "EXPLORATION",
            "method_standing": "NOT_CURRENT",
            "method_refusal": superseded,
        }
    )
    assert (explored.contract, explored.evidence) == ("NOT_CURRENT", "EXPLORATION")
    assert explored.reasons == {"contract": superseded} and superseded in explored.statements[2]
    for disposition, evidence in (
        ("CURRENT_ALPHA_CANDIDATE_SET_READY", "QUALIFIED"),
        ("NO_STABLE_CURRENT_ALPHA_MODEL", "NOT_QUALIFIED"),
    ):
        qualified = {**published, "alpha_qualification": {"disposition": disposition}}
        assert study_standing(qualified).evidence == evidence
    assert study_standing({"status": "RUNNING"}).execution == "RUNNING"
    blocked = study_standing({"status": "BLOCKED", "failure_code": "data.input_changed"})
    assert (blocked.execution, blocked.evidence) == ("STOPPED", "NONE")
    assert blocked.reasons == {"execution": "data.input_changed"}

    refused = "alpha_research.saved_comparison_score_support_mismatch"
    not_compared = trial_standing(
        state="COMPLETED",
        outcome="COMPARED",
        stopped=None,
        comparison={"standing": "NOT_COMPARED", "owner_comparison": {"failure_code": refused}},
    )
    assert marks(not_compared)[:2] == ("NOT_COMPARED", "SUCCEEDED")
    assert not_compared.reasons == {"comparison": refused}
    assert (
        not_compared.statements[0].startswith("Not compared")
        and "no change" in (not_compared.statements[0])
    )
    compared = trial_standing(
        state="COMPLETED", outcome="COMPARED", stopped=None, comparison={"standing": "COMPARED"}
    )
    assert (compared.comparison, compared.evidence) == ("COMPARED", "DEVELOPMENT")
    screened = trial_standing(
        state="COMPLETED",
        outcome="FEATURE_NOT_ADMITTED_BY_SCREENING",
        stopped=None,
        comparison=None,
    )
    assert (screened.comparison, screened.evidence) == ("NOT_APPLICABLE", "SCREENED_OUT")
    stopped = trial_standing(
        state="STOPPED",
        outcome=None,
        stopped={"step": "ALPHA_STUDY", "failure_code": "task_blocked"},
        comparison=None,
    )
    assert (stopped.execution, stopped.reasons) == ("STOPPED", {"execution": "task_blocked"})

    # A packet reads its newest completed trial, its contract and a person's activation.
    rows = [
        {
            "state": "COMPLETED",
            "outcome": "COMPARED",
            "alpha_standing": "NOT_COMPARED",
            "alpha_refusal": {"failure_code": refused, "detail": "..."},
        },
        {"state": "STOPPED", "stopped": {"step": "FACTOR_STUDY", "failure_code": "x"}},
    ]
    offered = packet_standing(contract={"status": "PASSED"}, trials=rows, active=False, held=None)
    assert marks(offered) == (
        "NOT_COMPARED",
        "SUCCEEDED",
        "PASSED",
        "DEVELOPMENT",
        "A_PERSON_MAY_ACTIVATE",
    )
    failed = packet_standing(
        contract={"status": "FAILED", "failure_code": "goldens_outside_tolerance"},
        trials=[],
        active=False,
        held="feature_extension.contract_failed:goldens_outside_tolerance",
    )
    assert marks(failed) == ("NOT_APPLICABLE", "NOT_STARTED", "FAILED", "NONE", "HELD")
    assert failed.reasons == {
        "contract": "goldens_outside_tolerance",
        "activation": "feature_extension.contract_failed:goldens_outside_tolerance",
    }
    active = packet_standing(contract={"status": "PASSED"}, trials=rows, active=True, held=None)
    assert active.activation == "ACTIVE" and "activation" not in active.reasons

    for reasons in ({}, {"comparison": refused, "evidence": "x"}):
        with pytest.raises(ValueError, match=r"result_standing.reasons_mismatch"):
            ResultStanding.of(
                comparison="NOT_COMPARED",
                execution="SUCCEEDED",
                contract="NOT_CHECKED",
                evidence="DEVELOPMENT",
                activation="NOT_ACTIVATABLE",
                reasons=reasons,
            )


def test_each_flow_names_its_prerequisites_and_the_way_on(tmp_path):
    """requirement (V367): an agent found each entry's prerequisites by help, schemas and
    refusals (AX11 spent 26 launches before its first goal; AX10 built a feature before it
    learned its trial needed an Alpha study); each flow names the results it needs on its
    input, the completed ones the workspace holds, newest first and at most five of each, what
    is missing and the requests allowed next, a choice left as None."""

    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from alphalattice.control.product_host.composition.research_prerequisites import (
        holdings,
        prerequisites,
    )
    from alphalattice.foundation.factor_research.experiments.development_evidence import (
        FACTOR_DEVELOPMENT_CURATION_CATEGORY,
    )

    binding = "b" * 64
    selector = {"research_input_id": "in-1", "input_binding_hash": binding}

    def of(flow, **held):
        return prerequisites(flow, held, input_id="in-1", binding_hash=binding)

    factor = of("FACTOR_STUDY")
    assert (factor["needs"], factor["missing"]) == ([], [])
    # V433: a study is planned from a declaration; the controls write one, and the plan is a
    # template that leaves it to give, never a runnable command bound to be refused.
    from alphalattice.interface.local_application.cli_contract import choices

    for flow, kind in (
        ("FACTOR_STUDY", "factor.screening-development"),
        ("RISK_STUDY", "risk.covariance-development"),
    ):
        offered = of(flow)["next_requests"]
        assert offered == {
            "controls": {"operation": "EXPERIMENT_CONTROLS", **selector, "experiment_kind": kind},
            "plan": {"operation": "EXPERIMENT_PLAN", **selector, "experiment_document": None},
        }
        assert choices(offered["plan"]) == ["experiment_document"]
    nothing = of("ALPHA_STUDY")
    assert nothing["missing"] == ["CURATED_FACTOR_STUDY"]
    assert nothing["next_requests"] == {
        "factor": {
            "operation": "EXPERIMENT_CONTROLS",
            **selector,
            "experiment_kind": "factor.screening-development",
        }
    }
    assert "run a Factor study first" in nothing["detail"]
    fresh = {"task_id": "f-2", "curation_receipt_hashes": []}
    uncurated = of("ALPHA_STUDY", FACTOR_STUDY=[fresh])
    assert uncurated["next_requests"] == {
        "curation": {"operation": "EXPERIMENT_CURATION", "task_id": "f-2"}
    }
    once = {"task_id": "f-1", "curation_receipt_hashes": ["r1"]}
    curated = of("ALPHA_STUDY", FACTOR_STUDY=[fresh, once], CURATED_FACTOR_STUDY=[once])
    assert curated["missing"] == []
    assert curated["next_requests"] == {
        "handoff": {
            "operation": "EXPERIMENT_HANDOFF_PREVIEW",
            "task_id": "f-1",
            "curation_receipt_hash": "r1",
        }
    }
    two = {"task_id": "f-1", "curation_receipt_hashes": ["r1", "r2"]}
    twice = of("ALPHA_STUDY", FACTOR_STUDY=[two], CURATED_FACTOR_STUDY=[two])
    assert twice["next_requests"]["handoff"]["curation_receipt_hash"] is None
    alpha = [{"task_id": "a-1", "factor_task_id": "f-1"}]
    book = of("BOOK", ALPHA_STUDY=alpha)
    assert book["missing"] == []
    assert book["next_requests"] == {
        "portfolio-draft": {
            "operation": "EXPERIMENT_PORTFOLIO_DRAFT",
            "task_id": "a-1",
            "candidate_id": None,
        },
        "risk": {
            "operation": "EXPERIMENT_CONTROLS",
            **selector,
            "experiment_kind": "risk.covariance-development",
        },
    }
    assert (
        "risk"
        not in of("BOOK", ALPHA_STUDY=alpha, RISK_STUDY=[{"task_id": "r-1"}])["next_requests"]
    )
    trial = of("FEATURE_TRIAL", FACTOR_STUDY=[fresh])
    assert trial["missing"] == ["ALPHA_STUDY"] and set(trial["next_requests"]) == {"curation"}
    assert of("FEATURE_TRIAL", ALPHA_STUDY=alpha)["next_requests"] == {}
    review = of("BOOK_REVIEW", BOOK=[{"task_id": "p-1", "alpha_task_id": "a-1"}])
    assert review["next_requests"] == {
        "book": {"operation": "EXPERIMENT_READBACK", "task_id": "p-1"}
    }

    # The holdings: this input's completed studies by result, newest first, five at most.
    start = datetime(2026, 9, 30, tzinfo=UTC)
    folder = "research-experiments/factor-one"
    receipts = tmp_path / folder / FACTOR_DEVELOPMENT_CURATION_CATEGORY / "checkpoint"
    receipts.mkdir(parents=True)
    (receipts / ("c" * 64 + ".json")).write_text("{}", encoding="utf-8")

    def study(n, kind, *, input_hash=binding, alpha_source=None, portfolio_source=None):
        task = SimpleNamespace(task_id=f"t-{n}", admitted_at=start + timedelta(minutes=n))
        plan = SimpleNamespace(
            binding=SimpleNamespace(binding_hash=input_hash),
            program=SimpleNamespace(kind=kind),
            alpha_source=alpha_source,
            portfolio_source=portfolio_source,
            document={"experiment": {"output_workspace": folder}},
        )
        return task, plan

    studies = [
        study(1, "factor.screening-development"),
        study(2, "factor.screening-development", input_hash="d" * 64),
        study(3, "alpha.model-development", alpha_source=SimpleNamespace(factor_task_id="t-1")),
        study(4, "alpha.model-development"),
        *(study(10 + n, "risk.covariance-development") for n in range(7)),
        study(
            30,
            "portfolio.policy-development",
            portfolio_source=SimpleNamespace(alpha_task_id="t-3"),
        ),
    ]
    plans = {task.task_id: plan for task, plan in studies}
    held = holdings(
        [task for task, _plan in studies],
        lambda task: plans[task.task_id],
        binding_hash=binding,
        workspace=tmp_path,
    )
    assert held["FACTOR_STUDY"] == [{"task_id": "t-1", "curation_receipt_hashes": ["c" * 64]}]
    assert held["ALPHA_STUDY"] == [{"task_id": "t-3", "factor_task_id": "t-1"}]
    assert [row["task_id"] for row in held["RISK_STUDY"]] == [f"t-{n}" for n in range(16, 11, -1)]
    assert held["BOOK"] == [{"task_id": "t-30", "alpha_task_id": "t-3"}]


def test_an_older_curation_still_opens_alpha_beyond_the_newest_five(tmp_path):
    """regression (V497, the user's review): six Factor studies on one input, only the oldest
    curated, and `workspace show` said none was curated, sending the agent to curate again:
    the holdings kept the newest five before the curated ones were looked for. A prerequisite
    is judged over every study, its display over the newest five."""

    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from alphalattice.control.product_host.composition.research_prerequisites import (
        holdings,
        prerequisites,
    )
    from alphalattice.foundation.factor_research.experiments.development_evidence import (
        FACTOR_DEVELOPMENT_CURATION_CATEGORY,
    )

    binding = "b" * 64
    start = datetime(2026, 9, 30, tzinfo=UTC)
    receipts = tmp_path / "factor-0" / FACTOR_DEVELOPMENT_CURATION_CATEGORY / "checkpoint"
    receipts.mkdir(parents=True)
    (receipts / ("c" * 64 + ".json")).write_text("{}", encoding="utf-8")

    def factor(n):
        task = SimpleNamespace(task_id=f"t-{n}", admitted_at=start + timedelta(minutes=n))
        plan = SimpleNamespace(
            binding=SimpleNamespace(binding_hash=binding),
            program=SimpleNamespace(kind="factor.screening-development"),
            alpha_source=None,
            portfolio_source=None,
            document={"experiment": {"output_workspace": f"factor-{n}"}},
        )
        return task, plan

    studies = [factor(n) for n in range(7)]
    plans = {task.task_id: plan for task, plan in studies}
    held = holdings(
        [task for task, _plan in studies],
        lambda task: plans[task.task_id],
        binding_hash=binding,
        workspace=tmp_path,
    )
    assert [row["task_id"] for row in held["FACTOR_STUDY"]] == [f"t-{n}" for n in range(6, 1, -1)]
    answer = prerequisites("ALPHA_STUDY", held, input_id="in-1", binding_hash=binding)
    assert answer["missing"] == []
    assert answer["next_requests"] == {
        "handoff": {
            "operation": "EXPERIMENT_HANDOFF_PREVIEW",
            "task_id": "t-0",
            "curation_receipt_hash": "c" * 64,
        }
    }


def test_a_draft_onto_another_input_takes_that_inputs_revision():
    """regression (V498, the user's review): `study draft <A> --input B` kept A's revision, and
    (B, A's hash) was refused `research_input.version_not_admitted` unless `--binding` named
    B's. Another input named without a revision takes that input's declared anchor; the same
    input keeps the origin's exact revision; a revision named wins."""

    from alphalattice.control.product_host.composition.research_experiments import draft_revision

    prior = ("input-a", "a" * 64)
    assert draft_revision(prior, None, None) == ("input-a", "a" * 64)
    assert draft_revision(prior, "input-a", None) == ("input-a", "a" * 64)
    assert draft_revision(prior, "input-b", None) == ("input-b", None)
    assert draft_revision(prior, "input-b", "b" * 64) == ("input-b", "b" * 64)


def test_a_history_row_offers_its_reads_bound_to_it():
    """regression (V499, the user's review): `history list --entry E` answered the row's
    locators, and the agent copied them into `evidence show --update ... --basis ...`. Each row
    offers its reads bound to it: its Task, its book's review and, for a CRO review, its
    export; the CLI's `--from` then reads the row the agent chose."""

    from datetime import UTC, datetime
    from uuid import uuid4

    from alphalattice.control.product_host.composition.research_history import HistoryEntry
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector

    task = uuid4()
    book = BookSelector(
        update_task_id=task, update_publication_hash="p" * 64, position_basis="CONDITIONAL_ESTIMATE"
    )
    entry = HistoryEntry(
        "review:x",
        "CRO_REVIEW",
        datetime(2026, 10, 2, tzinfo=UTC),
        task,
        "SUCCEEDED",
        book=book,
        review_publication_hash="r" * 64,
    )
    offered = entry.body()["next_requests"]
    fields = book.request_fields()
    assert offered["task"] == {"operation": "STATUS", "task_id": str(task)}
    assert offered["review"] == {"operation": "EVIDENCE_CRO", **fields}
    assert offered["export"] == {
        "operation": "EVIDENCE_CRO_EXPORT",
        **fields,
        "review_publication_hash": "r" * 64,
    }


@pytest.mark.parametrize(
    "fields",
    [
        {"result_hash": "a" * 64},
        {"handoff_hash": "b" * 64},
        {
            "update_task_id": UUID("00000000-0000-0000-0000-000000000001"),
            "update_publication_hash": "c" * 64,
            "position_basis": "CONDITIONAL_ESTIMATE",
        },
        {
            "update_task_id": UUID("00000000-0000-0000-0000-000000000001"),
            "update_publication_hash": "c" * 64,
            "position_basis": "OBSERVED_RESEARCH_ENTRY",
        },
        {
            "experiment_task_id": UUID("00000000-0000-0000-0000-000000000002"),
            "experiment_receipt_hash": "d" * 64,
            "portfolio_session": "2026-10-02",
        },
    ],
    ids=["result", "handoff", "conditional-update", "observed-update", "experiment"],
)
def test_history_book_discovery_offers_exact_reads_without_a_full_projection(fields, monkeypatch):
    """V683/EV: discovery offers the complete selector but makes no current-standing claim."""
    from alphalattice.control.product_host.composition.evidence_review_projection import (
        EvidenceCroProjector,
    )
    from alphalattice.control.product_host.composition.research_history import HistoryEntry
    from alphalattice.interface.local_application.answers import EvidenceBookSummary
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector

    def no_projection(*_args, **_kwargs):
        raise AssertionError("Book discovery opened the full Evidence projection")

    monkeypatch.setattr(EvidenceCroProjector, "projection", no_projection)
    book = BookSelector(**fields)
    row = HistoryEntry(
        "metadata:book",
        "CRO_REVIEW",
        datetime(2026, 10, 2, tzinfo=UTC),
        None,
        "HISTORICAL_REVIEW",
        book=book,
    ).body()
    summary = row["book_summary"]
    EvidenceBookSummary.model_validate(summary)
    assert summary["state"] == "NOT_READ"
    assert summary["verification"] == "METADATA_ONLY_SELECTED_READBACK_VERIFIES_DESCENDANTS"
    assert summary["next_requests"]["review"] == {
        "operation": "EVIDENCE_CRO",
        **book.request_fields(),
    }
    assert summary["next_requests"]["review"] == row["next_requests"]["review"]
    assert "evidence_as_of" not in summary and "review_standing" not in summary


def test_a_request_that_fills_an_object_in_part_is_a_template_naming_the_fields_left(tmp_path):
    """regression (V373): AX12's agent ran the curation's `curate` as `next_commands` listed it
    and was refused for the author's choices it never gave; a request that fills an object in
    part (the receipt, not the choices) is a template naming each field left, its command
    writes no flag the CLI lacks, `request --from` refuses it unfilled, and a plan's own
    document, the owner's whole, is never read for choices."""

    from alphalattice.interface.local_application.cli_contract import choices, command

    task = str(uuid4())
    curate = {
        "operation": "EXPERIMENT_CURATE",
        "task_id": task,
        "experiment_curation": {"expected_receipt_hash": "a" * 64},
    }
    left = ["experiment_curation.choices", "experiment_curation.limitations_acknowledged"]
    assert choices(curate) == left
    line = command(curate, quoting="posix")
    assert line is not None and "--experiment-curation.choices" not in line
    assert line.endswith("with choices, limitations_acknowledged>")
    replan = {
        "operation": "EXPERIMENT_PLAN",
        "research_input_id": "in-1",
        "experiment_document": {"experiment": {"kind": "factor"}, "alpha": None},
    }
    assert choices(replan) == []


def test_a_provider_that_limits_or_changes_its_interface_is_worded_with_the_way_on():
    """requirement (V375, RR5): a first use or an update met a provider that limited the
    requests, did not answer or changed its interface, and the codes had no words; each is
    worded, and a deferred Task's answer says when it resumes, by which request, and, where
    research inputs are published, that they stand."""

    from alphalattice.control.product_host.composition.plain_refusals import deferral, explain

    for code in (
        "data.rate_limited",
        "data.provider_session_unstable",
        "workspace_preparation.retry_not_due",
        "workspace_data_update.retry_not_due",
    ):
        words = explain(code)["detail"]
        assert "provider" in words and "`retry_after_at`" in words, code
    # No owner defers on a timeout: the listing is recorded failed, or the step stopped (V396).
    timeout = explain("data.provider_timeout")["detail"]
    assert "did not answer in time" in timeout and "failed" in timeout, timeout
    assert "retry_after_at" not in timeout and "No retry time is set" in timeout, timeout
    for code in (
        "data.provider_fetch_failed",
        "data.current_universe_no_feature_ready_listings",
        "data.listing_updates_incomplete",
    ):
        assert "interface" in explain(code)["detail"], code
    resume = {"operation": "WORKSPACE_PREPARE_CONFIRM", "preparation_plan_hash": "a" * 64}
    first = deferral(
        {"status": "DEFERRED", "next_requests": {}},
        failure_code="data.rate_limited",
        retry_after_at="2026-10-01T12:00:00+00:00",
        resume=resume,
        held=False,
    )
    assert first["next_requests"] == {"resume": resume}
    assert first["retry_after_at"] == "2026-10-01T12:00:00+00:00"
    assert "inputs are unchanged" not in first["detail"]
    update = deferral(
        {"status": "PUBLISHED"},
        failure_code="data.remediation_wait",
        retry_after_at=None,
        resume=None,
        held=True,
    )
    assert update["detail"].startswith("The work was deferred")
    assert update["detail"].endswith("every study keeps reading them.")
    assert update["next_requests"] == {}


def test_next_request_fills_choices_preserves_references_and_requires_explicit_action(tmp_path):
    from argparse import Namespace

    from alphalattice.interface.local_application.client import (
        _next_request,
        continuation,
        continued,
    )
    from alphalattice.interface.local_application.operations import fields

    task = str(uuid4())
    template = {
        "operation": "EXPERIMENT_CURATE",
        "task_id": task,
        "experiment_curation": {"expected_receipt_hash": "a" * 64},
    }
    response = tmp_path / "response.json"
    choices = tmp_path / "choices.yaml"
    response.write_text(json.dumps({"next_requests": {"curate": template}}), encoding="utf-8")
    choices.write_text(
        "experiment_curation:\n  choices: []\n  limitations_acknowledged: []\n", encoding="utf-8"
    )
    args = Namespace(from_response=response, action="curate", file=choices)
    # Unfilled, the author's choices are named before anything is sent (V373).
    with pytest.raises(ValueError, match=r"needs_choice:experiment_curation\.choices,"):
        _next_request(Namespace(from_response=response, action="curate", file=None))
    filled = _next_request(args)
    assert filled == {
        **template,
        "experiment_curation": {
            **template["experiment_curation"],
            "choices": [],
            "limitations_acknowledged": [],
        },
    }
    assert template["experiment_curation"] == {"expected_receipt_hash": "a" * 64}
    # The compact display replaces the part it cut with a marker.
    cut = {**template, "experiment_curation": {"_omitted": "collection", "item_count": 1}}
    compact = {
        "representation": "COMPACT_DISPLAY_NOT_AN_AUTHORITY_DOCUMENT",
        "omitted_sections": ["next_requests.curate.experiment_curation"],
        "data": {"task_id": task, "next_requests": {"curate": cut}},
    }
    response.write_text(json.dumps(compact), encoding="utf-8")
    with pytest.raises(ValueError, match="compact_reference_requires_full_response:curate"):
        _next_request(args)
    # Intact scalar locators remain usable.
    shown = json.loads(response.read_text(encoding="utf-8"))["data"]
    assert continued("EXPERIMENT_READBACK", shown, {}, frozenset({"task_id"}))["task_id"] == task
    # A display cut elsewhere still holds the request whole, so it continues (V133).
    compact["omitted_sections"] = ["result"]
    compact["data"] = {
        "task_id": task,
        "result": {"_omitted": "collection", "item_count": 9},
        "next_requests": {"curate": template},
    }
    response.write_text(json.dumps(compact), encoding="utf-8")
    assert _next_request(args) == filled
    response.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(ValueError, match="operation_document_has_no_next_request"):
        _next_request(args)
    response.write_text(json.dumps({"next_requests": {"curate": template}}), encoding="utf-8")
    choices.write_text("experiment_curation:\n  expected_receipt_hash: bad\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match=r"bound_reference_override:experiment_curation.expected_receipt_hash"
    ):
        _next_request(args)
    args.action = None
    with pytest.raises(ValueError, match="next_request_required:curate"):
        _next_request(args)

    book, risk = tmp_path / "book.json", tmp_path / "risk.json"
    book.write_text(json.dumps({"task_id": task}), encoding="utf-8")
    risk_id = str(uuid4())
    risk.write_text(json.dumps({"task_id": risk_id}), encoding="utf-8")
    # `experiment link-risk --from book.json --risk-task-id <id>`: the book's Task is its locator.
    linked = continuation(
        "EXPERIMENT_LINK_RISK", book, {"risk_task_id": risk_id}, fields("EXPERIMENT_LINK_RISK")[1]
    )
    assert linked == {"operation": "EXPERIMENT_LINK_RISK", "task_id": task, "risk_task_id": risk_id}


def test_an_operation_command_starts_from_a_saved_answer(tmp_path, monkeypatch, capsys):
    """`<noun> <verb> --from <answer>` (Z1b): the answer supplies what the command does not
    give, so a required field it holds is not asked for; one stdin source per command."""

    from alphalattice.interface.local_application import cli, client

    sent = []

    class LocalClient:
        workspace = tmp_path
        goal = None  # no --goal named; every answer names its context (CG4)

        def exchange(self, document):
            sent.append(document)
            return {"status": "ADMITTED", "task_id": str(uuid4())}, b"{}"

        def selected_url(self, document, body):
            return ""

    monkeypatch.setattr(client, "LocalResearchClient", lambda *args, **kwargs: LocalClient())
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"status": "PLANNED", "plan_hash": "p" * 64}), encoding="utf-8")
    workspace = ["--workspace", str(tmp_path), "--view", "full"]
    assert cli.main([*workspace, "study", "run", "--from", str(plan)], serve=print) == 3
    assert sent == [{"operation": "EXPERIMENT_RUN", "experiment_plan_hash": "p" * 64}]
    capsys.readouterr()
    with pytest.raises(SystemExit) as stopped:
        cli.main([*workspace, "study", "run"], serve=print)
    assert stopped.value.code == 1 and "--plan" in capsys.readouterr().err
    code = cli.main(
        [*workspace, "curation", "submit", "--from", "-", "--file", "-"],
        serve=print,
    )
    answer = json.loads(capsys.readouterr().out)
    assert code == 1 and answer["failure_code"] == "local_client.document_multiple_stdin_sources"
    assert len(sent) == 1


def test_declaration_validation_feedback_names_fields_without_echoing_values(live, monkeypatch):
    from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
        PortfolioExperimentSpec,
    )

    def invalid_document(*args, **kwargs):
        return PortfolioExperimentSpec.model_validate(
            {"alpha_task_id": "known", "candidate_id": "known", "top_k": "private-input-marker"}
        )

    monkeypatch.setattr(live.operations.experiments, "operate", invalid_document)
    before = len(live.session.task_control_registry.tasks())
    reply = _json(
        live, "/api/experiments/plan", method="POST", payload={"experiment_yaml": "experiment: {}"}
    )
    assert reply["failure_code"] == "research_experiment.document_invalid"
    assert reply["fields"] == [["top_k"]]
    assert reply["next_action"] == "CORRECT_DECLARATION_AND_REPLAN"
    assert reply["next_requests"]["inputs"] == {"operation": "RESEARCH_INPUTS"}
    assert "private-input-marker" not in json.dumps(reply)
    assert len(live.session.task_control_registry.tasks()) == before


def test_declaration_comparison_preserves_absence_and_nested_parameter_changes():
    from alphalattice.control.product_host.composition.research_experiment_projection import (
        _field_changes,
    )

    before = {"portfolio": {"top_k": 35}, "old": None, "same": [1, 2]}
    after = {"portfolio": {"top_k": 40}, "new": None, "same": [1, 2]}
    changes = _field_changes(before, after)
    assert [c["path"] for c in changes] == [["new"], ["old"], ["portfolio", "top_k"]]
    assert changes[0]["before_present"] is False and changes[0]["after_present"] is True
    assert changes[1]["before_present"] is True and changes[1]["after_present"] is False
    assert (changes[2]["before"], changes[2]["after"]) == (35, 40)


@pytest.mark.parametrize("lines", ["stop\n", "ignored\nstop\n", ""])
def test_attached_launcher_stops_on_explicit_stdin_or_eof(tmp_path, monkeypatch, lines):
    from io import StringIO

    from scripts import run_local_portfolio_web

    from alphalattice.interface.local_application.cli import main

    calls = []

    class Session:
        launch_url = "http://127.0.0.1:12345/launch?key=k"  # printed and opened (HB)

        def start(self):
            calls.append("start")
            return "http://127.0.0.1:12345/"

        def stop(self):
            calls.append("stop")

    monkeypatch.setattr(
        run_local_portfolio_web.LocalPortfolioWebSession,
        "from_workspace",
        lambda *a, **k: Session(),
    )
    monkeypatch.setattr(sys, "stdin", StringIO(lines))
    assert (
        main(
            ["--workspace", str(tmp_path), "serve", "--no-browser", "--stop-on-stdin"],
            serve=run_local_portfolio_web.main,
        )
        == 0
    )
    assert calls == ["start", "stop"]


_WATCHED_STDIN = """
import subprocess, sys, threading
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/src"]
from scripts.run_local_portfolio_web import _wait_for_stop_on_stdin
watcher = threading.Thread(target=_wait_for_stop_on_stdin)
watcher.start()
# A process started while the Host waits for its stop inherits the stdin pipe (W10's worker).
subprocess.run([sys.executable, "-c", "pass"], check=True, timeout=60)
print("started", flush=True)
watcher.join()
"""


@pytest.mark.parametrize("ending", ["stop\n", ""])
def test_a_host_waiting_for_its_stop_lets_the_processes_it_starts_begin(ending: str) -> None:
    """regression (W10): a synchronous read pending on the stdin pipe held every process the
    Host started, which queries the stdin it inherits as it starts: the Task worker never
    began, and a golden or AX run served with `--stop-on-stdin` waited out its timeout."""

    root = Path(__file__).resolve().parents[2]
    host = subprocess.Popen(
        [sys.executable, "-c", _WATCHED_STDIN, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert host.stdout is not None and host.stdin is not None
    assert host.stdout.readline().strip() == "started"
    host.stdin.write("noise\n" + ending)
    host.stdin.close()
    assert host.wait(timeout=30) == 0


def test_cli_missing_dependency_names_locked_setup_but_does_not_hide_missing_product(
    monkeypatch, capsys
):
    import builtins

    from scripts import run_alphalattice

    original_import = builtins.__import__
    missing = "pydantic"

    def unavailable(name, *args, **kwargs):
        if name == "alphalattice.interface.local_application.cli":
            raise ModuleNotFoundError(name=missing)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    assert run_alphalattice.main(["--help"]) == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["missing_module"] == "pydantic"
    assert refusal["setup_command"][-2:] == ["--locked", "--all-extras"]
    assert refusal["claim_limit"] == "NO_INSTALLATION_OR_WORKSPACE_WORK_PERFORMED"
    missing = "alphalattice.missing_implementation"
    with pytest.raises(ModuleNotFoundError):
        run_alphalattice.main(["--help"])


def test_workspace_context_is_opt_in_and_one_unreadable_section_stays_explicit(live, monkeypatch):
    from alphalattice.interface.local_application.client import LocalResearchClient

    def unavailable_inputs(_self):
        raise ValueError("research_input.test_unreadable")

    monkeypatch.setattr(
        type(live.operations.input_capture.revisions), "versions", unavailable_inputs
    )
    before = len(live.session.task_control_registry.tasks())
    client = LocalResearchClient(live.workspace)
    assert "research_context" not in client.request()
    context = client.request(include_context=True)["research_context"]
    assert context["inputs"] == {
        "status": "REFUSED",
        "failure_code": "research_input.test_unreadable",
    }
    assert context["recent_research"]["status"] == "AVAILABLE"
    assert context["tasks"]["tasks"] == []
    # The six readers share one read boundary: the Task store is opened once
    # for the context (and once by the manifest refresh before it), read-only.
    from alphalattice.control.workspace_runtime import database

    opens: list[tuple[str, bool]] = []
    fresh = database._fresh_connection

    def counted(path, *, read_only):
        opens.append((path.name, read_only))
        return fresh(path, read_only=read_only)

    monkeypatch.setattr(database, "_fresh_connection", counted)
    assert context == _json(live, "/api/session?context=1")["research_context"]
    assert opens == [("research-task-control.duckdb", True)] * len(opens)
    assert len(opens) <= 2
    assert len(live.session.task_control_registry.tasks()) == before


@pytest.mark.parametrize("fault", ["human_token", "workspace", "actor_payload"])
def test_external_client_refuses_spoofing_before_operation_dispatch(live, monkeypatch, fault):
    from alphalattice.interface.local_application.client import LocalResearchClient

    client = LocalResearchClient(live.workspace)
    if fault == "human_token":
        client.connection = dataclasses.replace(
            client.connection, token=live.web.application.session_token
        )
    elif fault == "workspace":
        client.workspace = live.workspace.parent / "another-workspace"
    monkeypatch.setattr(
        type(live.operations),
        "execute",
        lambda *a, **k: pytest.fail("unauthorized operation dispatched"),
    )
    request = {"operation": "TASKS"}
    if fault == "actor_payload":
        request["caller"] = "HUMAN"
    result = client.request(request)
    assert "refused" in result or result.get("status") == "REFUSED"
    assert not live.session.task_control_registry.tasks()


def test_cli_serve_starts_from_outside_the_repo_and_stale_discovery_is_replaced(tmp_path):
    from alphalattice.interface.local_application.client import (
        LocalResearchClient,
        LocalResearchConnection,
    )

    root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "served-workspace"
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    environment["ALPHALATTICE_NETWORK_DISABLED"] = "1"
    child = subprocess.Popen(
        [
            sys.executable,
            str(root / "scripts/run_alphalattice.py"),
            "--workspace",
            str(workspace),
            "serve",
            "--no-browser",
        ],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 20
        while not LocalResearchConnection.path(workspace).exists():
            if child.poll() is not None or time.monotonic() >= deadline:
                pytest.fail("CLI did not start its own Host")
            time.sleep(0.05)
        client = LocalResearchClient(workspace)
        assert client.request()["caller"] == "EXTERNAL_AUTOMATION"
        instance = client.connection.instance
    finally:
        # This is the creation-bound handle of our isolated, task-free child.
        child.terminate()
        child.communicate(timeout=20)
    with LocalPortfolioWebSession.from_workspace(workspace) as reopened:
        assert LocalResearchConnection.read(workspace).instance != instance
        assert LocalResearchClient(workspace).request()["workspace_id"] == reopened.workspace_id
    assert not LocalResearchConnection.path(workspace).exists()


@pytest.mark.parametrize("fault", ("incomplete_body", "timeout_body", "timeout_open"))
def test_cli_reports_an_interrupted_response_body_without_retrying_work(
    live, monkeypatch, capsys, fault
):
    import http.client

    from alphalattice.interface.local_application.cli import main

    class InterruptedResponse:
        status = 200

        def read(self):
            if fault == "timeout_body":
                raise TimeoutError("private transport detail must not be printed")
            raise http.client.IncompleteRead(b"", 100)

    timeouts, closed = [], []

    class Connection:
        def __init__(self, host, port, *, timeout):
            assert host == "127.0.0.1"
            timeouts.append(timeout)

        def request(self, *args, **kwargs):
            if fault == "timeout_open":
                raise TimeoutError("private transport detail must not be printed")

        def getresponse(self):
            return InterruptedResponse()

        def close(self):
            closed.append(True)

    monkeypatch.setattr(http.client, "HTTPConnection", Connection)
    timeout_arguments = [] if fault == "timeout_body" else ["--request-timeout", "120"]
    code = main(
        ["--workspace", str(live.workspace), *timeout_arguments, "task", "list"],
        serve=lambda _: 99,
    )
    body = json.loads(capsys.readouterr().out)
    assert code == 4 and closed == [True]  # The connection is closed whatever failed.
    assert body["failure_code"] == "local_client.connection_lost_task_may_still_run"
    assert body["transport_reason"] == (
        "HTTP_WAIT_EXPIRED" if fault.startswith("timeout") else "CONNECTION_INTERRUPTED"
    )
    assert body["request_timeout_seconds"] == 120
    assert body["operation_outcome"] == "UNKNOWN"
    assert "private transport detail" not in json.dumps(body)
    assert timeouts == [120.0]  # One request, no retry after an uncertain response.
    for timeout in ("0", "-1", "nan", "inf", "601"):
        assert (
            main(
                [
                    "--workspace",
                    str(live.workspace),
                    "--request-timeout",
                    timeout,
                    "task",
                    "list",
                ],
                serve=lambda _: 99,
            )
            == 1
        )
        assert json.loads(capsys.readouterr().out)["failure_code"] == (
            "local_client.request_timeout_outside_0_600"
        )
    assert timeouts == [120.0]
    assert not live.session.task_control_registry.tasks()


def test_failed_output_keeps_compact_feedback_and_the_completed_operation(
    tmp_path, monkeypatch, capsys
):
    from alphalattice.interface.local_application import cli, cli_contract, client

    task = str(uuid4())
    payload = {"status": "ADMITTED", "task_id": task, "html": "private detail " * 10000}
    calls = []

    class LocalClient:
        workspace = tmp_path
        goal = None  # no --goal named; every answer names its context (CG4)

        def exchange(self, document):
            calls.append(document)
            return payload, json.dumps(payload).encode()

        def selected_url(self, document, body):
            return None

        def navigation(self, document, body):
            return None

    def refuse_output(*args, **kwargs):
        raise client.LocalResearchClientError("local_client.output_write_refused")

    monkeypatch.setattr(client, "LocalResearchClient", lambda *args, **kwargs: LocalClient())
    monkeypatch.setattr(client, "_save_output", refuse_output)
    for view in ("compact", "full"):
        code = cli.main(
            [
                "--workspace",
                str(tmp_path),
                "--view",
                view,
                "study",
                "export",
                task,
                "--output",
                str(tmp_path / "unwritten.html"),
                "--format",
                "html",
            ],
            serve=lambda _: 99,
        )
        raw = capsys.readouterr().out
        result = json.loads(raw)
        # The admitted operation keeps its own outcome; the unwritten file is reported beside
        # it, so nobody sends the RUN again (V129).
        assert result["outcome"] == "PENDING" and result["status"] == "ADMITTED"
        assert code == cli_contract.EXIT_CODES["PENDING"]
        assert len(calls) == (1 if view == "compact" else 2)
        assert result["local_failure"]["failure_code"] == "local_client.output_write_refused"
        assert result["local_failure"]["next_action"] == (
            "USE_STDOUT_OR_A_WRITABLE_UNUSED_OUTPUT_PATH"
        )
        assert "output_file" not in result
        if view == "compact":
            assert result["representation"] == "COMPACT_DISPLAY_NOT_AN_AUTHORITY_DOCUMENT"
            assert result["data"]["task_id"] == task[:12]  # as the compact view shows it (V393)
            assert result["data"]["status"] == "ADMITTED"
            assert result["data"]["html"]["_omitted"] == "text"
            assert len(raw) < 2000 and "private detail" not in raw
        else:
            # The CLI's copy keeps the request it read with (V460).
            assert result["data"].pop("read_request")["operation"] == "EXPERIMENT_EXPORT"
            assert result["data"] == payload
    assert not (tmp_path / "unwritten.html").exists()


def test_a_wait_that_loses_its_host_keeps_the_admission_in_the_output(
    tmp_path, monkeypatch, capsys
):
    """requirement (V139, WK): an admitted RUN whose --wait cannot reach its Host waits it out;
    at --max-wait it reads as pending, and --output keeps the admission as the receipt of the
    work it started."""

    from alphalattice.interface.local_application import cli, cli_contract, client

    task = str(uuid4())
    admitted = {"status": "ADMITTED", "task_id": task, "publication_task_id": task}

    class LocalClient:
        workspace = tmp_path
        goal = None  # no --goal named; every answer names its context (CG4)

        def exchange(self, document):
            return admitted, json.dumps(admitted).encode()

        def request(self, document, **_options):
            raise client.LocalResearchClientError("local_client.connection_lost_task_may_still_run")

        def selected_url(self, document, body):
            return None

        def navigation(self, document, body):
            return None

    monkeypatch.setattr(client, "LocalResearchClient", lambda *args, **kwargs: LocalClient())
    receipt = tmp_path / "run.json"
    code = cli.main(
        [
            "--workspace",
            str(tmp_path),
            "study",
            "run",
            "--plan",
            "a" * 64,
            "--wait",
            "--max-wait",
            "0.5",
            "--output",
            str(receipt),
        ],
        serve=lambda _: 99,
    )
    result = json.loads(capsys.readouterr().out)
    assert result["outcome"] == "PENDING" and code == cli_contract.EXIT_CODES["PENDING"]
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert saved["admission"] == admitted
    assert saved["wait_event"]["event"] == "MAX_WAIT_REACHED"
    assert result["output_file"] == str(receipt.resolve())


def test_human_and_agent_share_plan_task_status_report_and_freeze(
    live: LocalPortfolioWebSession,
) -> None:
    spec: dict[str, object] = {}
    browser_plan = _json(live, "/api/plan", method="POST", payload={"spec": spec})
    agent_plan = _agent(live, PortfolioResearchAgentRequest(operation="PLAN", spec=spec))
    assert agent_plan == browser_plan

    admitted = _agent(live, PortfolioResearchAgentRequest(operation="RUN", spec=spec))
    assert admitted["disposition"] == "ADMITTED"
    task_id = UUID(str(admitted["task_id"]))
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]

    agent_status = _agent(live, PortfolioResearchAgentRequest(operation="STATUS", task_id=task_id))
    browser_status = _json(live, f"/api/status?task_id={task_id}")
    assert agent_status == browser_status
    assert agent_status["lifecycle"] == "SUCCEEDED"

    agent_results = _agent(live, PortfolioResearchAgentRequest(operation="RESULTS"))
    assert agent_results == _json(live, "/api/results")
    assert len(agent_results["results"]) == 1
    result_hash = str(agent_results["results"][0]["result_hash"])
    report = _agent(
        live,
        PortfolioResearchAgentRequest(operation="REPORT", result_hash=result_hash),
    )
    assert report == _json(live, f"/api/report?result_hash={result_hash}")
    # The report offers its book's review bound to the result it read, never the default
    # book's, so an agent need not supply the selector itself (V598, S1).
    assert report["review_selector"] == {"result_hash": result_hash}
    assert report["next_requests"]["evidence_preview"] == {
        "operation": "EVIDENCE_PREVIEW",
        "result_hash": result_hash,
    }
    assert {request["operation"] for request in report["next_requests"].values()} == {
        "EVIDENCE_CRO",
        "EVIDENCE_PREVIEW",
    }
    assert report["next_requests"]["review"] == {
        "operation": "EVIDENCE_CRO",
        "result_hash": result_hash,
    }

    frozen = _agent(
        live,
        PortfolioResearchAgentRequest(operation="FREEZE", result_hash=result_hash),
    )
    candidate_hash = str(frozen["candidate_hash"])
    agent_finalization = _agent(
        live,
        PortfolioResearchAgentRequest(operation="FINALIZATION", candidate_hash=candidate_hash),
    )
    assert agent_finalization == _json(live, f"/api/finalization?candidate_hash={candidate_hash}")
    assert agent_finalization["disposition"] == "AWAITING_PROTECTED_AUTHORITY"
    assert agent_finalization["stage_11_action_available"] is False
    assert _agent(
        live, PortfolioResearchAgentRequest(operation="EVIDENCE_CRO", result_hash=result_hash)
    ) == _json(live, f"/api/evidence-cro?result_hash={result_hash}")


def test_result_and_history_collections_name_a_corrupt_result_index(
    live: LocalPortfolioWebSession,
) -> None:
    assert live.application is not None
    assert live.dispatcher is not None
    pipeline = live.application.pipeline
    admitted = _json(live, "/api/run", method="POST", payload={})
    assert admitted["disposition"] == "ADMITTED"
    healthy_task_id = str(admitted["task_id"])
    live.dispatcher.drain_for_tests()
    assert _json(live, f"/api/status?task_id={healthy_task_id}")["lifecycle"] == "SUCCEEDED"

    agent_results = _agent(live, PortfolioResearchAgentRequest(operation="RESULTS"))
    assert agent_results == _json(live, "/api/results")
    (healthy_result,) = [
        row for row in agent_results["results"] if row["task_id"] == healthy_task_id
    ]

    bad_hash = "e" * 64
    pipeline._result_index(bad_hash).write_text("{", encoding="utf-8")

    agent_results = _agent(live, PortfolioResearchAgentRequest(operation="RESULTS"))
    assert agent_results == _json(live, "/api/results")
    assert [row["result_hash"] for row in agent_results["results"]] == [
        healthy_result["result_hash"]
    ]
    (result_refusal,) = agent_results["refusals"]
    assert result_refusal["result_hash"] == bad_hash
    assert result_refusal["record_id"] == bad_hash
    assert result_refusal["entry_id"] == f"result:{bad_hash}"
    assert "task_id" not in result_refusal

    agent_history = _agent(
        live,
        PortfolioResearchAgentRequest(operation="RESEARCH_HISTORY", history_limit=50),
    )
    browser_history = _json(live, "/api/research-history?history_limit=50")
    assert agent_history == browser_history
    assert agent_history["status"] == "PARTIAL_METADATA_READBACK"
    (healthy_history,) = [
        row
        for row in agent_history["entries"]
        if row["entry_id"] == f"result:{healthy_result['result_hash']}"
    ]
    assert healthy_history["task_id"] == healthy_task_id
    assert healthy_history["status"] == "SUCCEEDED"
    blocked = next(
        row for row in agent_history["blocked_entries"] if row["result_hash"] == bad_hash
    )
    assert blocked["status"] == "REFUSED"
    assert blocked["kind"] == "INSTALLED_RESULT"
    assert blocked["entry_id"] == f"result:{bad_hash}"
    assert "task_id" not in blocked


@pytest.mark.parametrize(
    "association",
    ["indexed", "legacy", "redirected", "missing-keys", "wrong-shape"],
)
def test_results_selects_reused_task_metadata_without_scanning_reports(
    live: LocalPortfolioWebSession, monkeypatch, association: str
) -> None:
    """V683/OP4: the selected Task's sealed association is exact, and damage is a named refusal."""
    from datetime import timedelta

    from alphalattice.control.product_host.publication.portfolio_research import (
        PortfolioResearchPipelineManifest,
    )
    from alphalattice.interface.local_application.answers import answer_problem

    assert live.application is not None and live.operations is not None
    pipeline = live.application.pipeline
    first, reused = uuid4(), uuid4()
    moment = datetime(2026, 10, 2, tzinfo=UTC)
    values = [
        PortfolioResearchPipelineManifest.create(
            workspace_id=live.operations.workspace_manifest.workspace_id,
            task_id=task,
            task_record_hash=str(index) * 64,
            program_hash="b" * 64,
            result_hash="c" * 64,
            report_hash="d" * 64,
            completed_at=moment + timedelta(minutes=index),
        )
        for index, task in enumerate((first, reused), start=1)
    ]
    for value in values:
        pipeline.publish(value)

    def no_report(*_args, **_kwargs):
        raise AssertionError("Result metadata opened a report")

    monkeypatch.setattr(type(live.operations), "report", no_report)
    collection = _json(live, "/api/results")
    assert next(row for row in collection["results"] if row["result_hash"] == "c" * 64)[
        "task_id"
    ] == str(first)
    # A retained index is intentionally damaged here; the helper locates this fixture's
    # exact record without duplicating the store's on-disk addressing rule.
    index_path = pipeline._task_index(reused)
    if association == "legacy":
        index_path.unlink()
    elif association == "redirected":
        index_path.write_text(
            json.dumps({"task_id": str(reused), "manifest_hash": values[0].manifest_hash}),
            encoding="utf-8",
        )
    elif association in {"missing-keys", "wrong-shape"}:
        index_path.write_text("{}" if association == "missing-keys" else "[]", encoding="utf-8")
    # A broken unrelated result index cannot force the exact by-Task read to open reports.
    pipeline._result_index("e" * 64).write_text("{", encoding="utf-8")
    browser = _json(live, f"/api/results?task_id={reused}")
    agent = _agent(live, PortfolioResearchAgentRequest(operation="RESULTS", task_id=reused))
    assert agent == browser
    assert answer_problem("RESULTS", browser, {"task_id": str(reused)}) is None
    if association in {"indexed", "legacy"}:
        (row,) = browser["results"]
        assert row == {
            "result_hash": values[1].result_hash,
            "report_hash": values[1].report_hash,
            "program_hash": values[1].program_hash,
            "task_id": str(reused),
            "completed_at": values[1].completed_at.isoformat(),
        }
        assert not browser.get("refusals")
    else:
        assert browser["results"] == []
        (refusal,) = browser["refusals"]
        assert refusal["status"] == "REFUSED" and refusal["task_id"] == str(reused)
        assert refusal["record_id"] == f"task:{reused}"
        assert refusal["next_requests"]["task"] == {"operation": "STATUS", "task_id": str(reused)}
        assert refusal["detail"] and "handler_failed" not in refusal["failure_code"]
        if association == "redirected":
            assert refusal["failure_code"] == "portfolio_application.pipeline_task_index_tampered"


def test_results_names_an_invalid_task_selector(live: LocalPortfolioWebSession) -> None:
    """V683/OP4: the HTTP selector error remains the Task owner's typed refusal."""
    status, _headers, encoded = _request(live, "/api/results?task_id=not-a-task")
    body = json.loads(encoded)
    assert status == 400
    assert body["failure_code"] == "local_web.task_id_invalid"
    assert body["next_action"] == "COPY_FULL_TASK_ID_FROM_HISTORY"


@pytest.mark.parametrize("damage", ["bad-json", "different-canonical-id"])
def test_task_record_collections_keep_readable_rows_and_offer_named_authority_routes(
    live: LocalPortfolioWebSession, damage: str
) -> None:
    """V661: one unreadable canonical Task cannot hide peers or authorize a complete scan."""
    import duckdb

    from alphalattice.control.task_control.contracts import (
        ResearchGoal,
        ResearchPlan,
        TaskInputEnvelope,
        TaskRecord,
    )
    from alphalattice.control.task_control.registry import TaskRecordAuthorityError
    from alphalattice.interface.local_application.cli_contract import (
        REQUEST_PROVENANCE,
        RequestProvenance,
    )
    from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
        TASK_KIND as REVIEW_TASK_KIND,
    )
    from tests.workspace_task_runner.task_control_support import task_contract

    assert live.operations is not None
    assert live.session is not None
    registry = live.session.task_control_registry
    result_hash = _run_to_completion(live)
    healthy_result = next(
        row["task_id"]
        for row in _json(live, "/api/results")["results"]
        if row["result_hash"] == result_hash
    )
    write_queue_setting(
        live.workspace / "runtime", "4", chosen_by="HUMAN", chosen_at=datetime.now(UTC)
    )
    agent_session = "task-record-collection"
    other_session = "task-record-other-session"
    tasks = []
    for salt, session in (
        ("unreadable-canonical-record", agent_session),
        ("readable-queued-peer", agent_session),
        ("readable-other-session-peer", other_session),
    ):
        envelope, goal, plan = task_contract(salt=salt)
        if salt == "readable-queued-peer":
            # A known queued review command, constructed without executing specialist work.
            envelope = TaskInputEnvelope.create(
                task_kind=REVIEW_TASK_KIND,
                input_schema_id="task-record-collection.review-fixture",
                payload={"prepared_submission": {}, "salt": salt},
            )
            goal = ResearchGoal.create(
                goal_kind="RUN_PORTFOLIO_REVIEW",
                input_hash=envelope.input_hash,
                deliverable_kind="PortfolioReview",
                summary="A queued review command retained beside unreadable authority.",
            )
            plan = ResearchPlan.create(
                goal_hash=goal.goal_hash,
                workflow_definition_hash=plan.workflow_definition_hash,
                verifier_catalog_hash=plan.verifier_catalog_hash,
                work_items=plan.work_items,
            )
        provenance = REQUEST_PROVENANCE.set(RequestProvenance(vendor="codex", session=session))
        try:
            tasks.append(
                registry.admit(
                    input_envelope=envelope,
                    goal=goal,
                    plan=plan,
                    observed_at=datetime.now(UTC),
                ).record
            )
        finally:
            REQUEST_PROVENANCE.reset(provenance)
    damaged, peer, other_peer = tasks
    # Corrupt canonical JSON, not the QUEUED head's separate projection dependency.
    damaged, _command = registry.request_cancel(
        task_id=damaged.task_id,
        expected_task_hash=damaged.record_hash,
        observed_at=datetime.now(UTC),
    )
    reads = {
        "RESEARCH_HISTORY": "/api/research-history?history_limit=50",
        "EXPERIMENTS": "/api/experiments",
        "PENDING_DECISIONS": "/api/decisions",
        "TASK_GUARDIAN": "/api/tasks/guardian",
        "UPGRADE_OVERVIEW": "/api/upgrade",
        "DATA_ISSUES": "/api/workspace/data-issues",
        "TASKS": "/api/tasks",
        "SESSION_TASKS": f"/api/tasks?agent_session={agent_session}",
    }
    before = {operation: _json(live, path) for operation, path in reads.items()}
    assert any(row["task_id"] == healthy_result for row in before["RESEARCH_HISTORY"]["entries"])
    assert any(row["task_id"] == str(peer.task_id) for row in before["TASK_GUARDIAN"]["tasks"])
    assert [row["task_id"] for row in before["SESSION_TASKS"]["tasks"]] == [
        str(peer.task_id),
        str(damaged.task_id),
    ]
    other_before = _json(live, f"/api/tasks?agent_session={other_session}")
    assert [row["task_id"] for row in other_before["tasks"]] == [str(other_peer.task_id)]
    projection_before = registry.projection_collection((damaged.task_id,))

    alien = TaskRecord.from_identity(
        **{**damaged.model_dump(exclude={"record_hash"}), "task_id": uuid4()}
    )
    replacement = "{" if damage == "bad-json" else alien.model_dump_json()
    with duckdb.connect(str(registry.database_path)) as connection:
        connection.execute(
            "UPDATE workspace_task SET record_json = ? WHERE task_id = ?",
            [replacement, str(damaged.task_id)],
        )
        stored_before = connection.execute(
            "SELECT task_id, record_json FROM workspace_task ORDER BY admission_sequence"
        ).fetchall()
        projections_before = connection.execute(
            "SELECT task_id, projection_json FROM workspace_task_projection ORDER BY task_id"
        ).fetchall()

    def named_refusals(value):
        if isinstance(value, dict):
            if value.get("task_id") == str(damaged.task_id) and value.get("status") == "REFUSED":
                yield value
            for item in value.values():
                yield from named_refusals(item)
        elif isinstance(value, list):
            for item in value:
                yield from named_refusals(item)

    answers = {operation: _json(live, path) for operation, path in reads.items()}
    for operation, answer in answers.items():
        refusals = list(named_refusals(answer))
        assert refusals, (operation, answer)
        for refusal in refusals:
            assert refusal["failure_code"] == "task_control.database_authority_unreadable"
            assert not {
                "lifecycle",
                "task_kind",
                "progress",
                "artifact_refs",
                "task_record_hash",
            }.intersection(refusal), refusal
            assert refusal["detail"] and refusal["next_action"]
            assert refusal["next_requests"]["workspace"] == {"operation": "WORKSPACE_SHOW"}
            assert refusal["next_requests"]["backups"] == {"operation": "WORKSPACE_BACKUPS"}
        assert str(alien.task_id) not in json.dumps(answer)
    assert registry.projection_collection((damaged.task_id,)) == projection_before
    assert _agent(live, PortfolioResearchAgentRequest(operation="TASKS")) == answers["TASKS"]
    assert any(row["task_id"] == str(peer.task_id) for row in answers["TASKS"]["tasks"])
    session_tasks = answers["SESSION_TASKS"]
    assert (
        _agent(live, PortfolioResearchAgentRequest(operation="TASKS", agent_session=agent_session))
        == session_tasks
    )
    assert session_tasks["tasks"] == [before["SESSION_TASKS"]["tasks"][0]]
    assert session_tasks["next_cursor"] is None
    (session_refusal,) = session_tasks["refusals"]
    assert session_refusal["task_id"] == str(damaged.task_id)
    assert "attention" not in session_refusal
    # Refusals belong to this page and session, while cursors retain admission order.
    first_page = _json(live, f"/api/tasks?agent_session={agent_session}&history_limit=1")
    assert first_page["tasks"] == session_tasks["tasks"]
    assert first_page["next_cursor"] == str(peer.task_id)
    assert not first_page.get("refusals")
    assert (
        _agent(
            live,
            PortfolioResearchAgentRequest(
                operation="TASKS", agent_session=agent_session, history_limit=1
            ),
        )
        == first_page
    )
    last_page = _json(
        live,
        f"/api/tasks?agent_session={agent_session}&history_limit=1"
        f"&history_cursor={first_page['next_cursor']}",
    )
    assert last_page == {"tasks": [], "next_cursor": None, "refusals": [session_refusal]}
    assert (
        _agent(
            live,
            PortfolioResearchAgentRequest(
                operation="TASKS",
                agent_session=agent_session,
                history_limit=1,
                history_cursor=first_page["next_cursor"],
            ),
        )
        == last_page
    )
    assert _json(live, f"/api/tasks?agent_session={other_session}") == other_before
    assert (
        _agent(live, PortfolioResearchAgentRequest(operation="TASKS", agent_session=other_session))
        == other_before
    )
    history = answers["RESEARCH_HISTORY"]
    assert any(row["task_id"] == healthy_result for row in history["entries"])
    assert not any(row["task_id"] == str(damaged.task_id) for row in history["entries"])
    exact = _json(live, f"/api/research-history?history_entry_id=task:{damaged.task_id}")
    assert exact["task_id"] == str(damaged.task_id) and exact["status"] == "REFUSED"
    assert exact["next_requests"]["backups"] == {"operation": "WORKSPACE_BACKUPS"}
    assert any(row["task_id"] == str(peer.task_id) for row in answers["TASK_GUARDIAN"]["tasks"])
    assert answers["EXPERIMENTS"]["experiments"] == before["EXPERIMENTS"]["experiments"]
    assert answers["DATA_ISSUES"]["next_requests"] == {
        "pending": {"operation": "PENDING_DECISIONS"}
    }
    assert answers["DATA_ISSUES"]["continuations"] == []
    assert answers["DATA_ISSUES"]["delegations"] == []
    assert answers["DATA_ISSUES"]["task_refusals"]
    assert any(
        row.get("task_id") == str(damaged.task_id) and row["kind"] == "TASK_RECORD_UNREADABLE"
        for row in answers["PENDING_DECISIONS"]["decisions"]
    )
    assert "acknowledge" not in answers["UPGRADE_OVERVIEW"]["next_requests"]
    assert live.review is not None
    assert REVIEW_TASK_KIND in live.review.recovery_commands(owed_only=False)
    assert REVIEW_TASK_KIND in live.recoverable_task_kinds()
    with pytest.raises(TaskRecordAuthorityError):
        live.review.recovery_commands()
    with pytest.raises(TaskRecordAuthorityError):
        live.resume()
    # The bootstrap response still reads the workspace, while its collections name damage.
    assert _json(live, "/api/session?context=1")["workspace_id"] == live.workspace_id
    _agent(live, PortfolioResearchAgentRequest(operation="WORKSPACE_SHOW"))
    _json(live, "/api/workspace/backup")

    acknowledgement = live.workspace / "runtime" / "upgrade-overview.json"
    acknowledgement_before = acknowledgement.read_bytes() if acknowledgement.exists() else None
    _status, _headers, body = _request(
        live,
        "/api/upgrade/acknowledge",
        method="POST",
        payload={"upgrade_set_hash": answers["UPGRADE_OVERVIEW"]["installed"]["set_hash"]},
    )
    refused = json.loads(body)
    assert refused["failure_code"] == "task_control.database_authority_unreadable"
    assert (
        acknowledgement.read_bytes() if acknowledgement.exists() else None
    ) == acknowledgement_before
    with pytest.raises(TaskRecordAuthorityError):
        registry.tasks()
    with pytest.raises(TaskRecordAuthorityError):
        live.operations.supervisor.supervise_once()
    with duckdb.connect(str(registry.database_path), read_only=True) as connection:
        assert (
            connection.execute(
                "SELECT task_id, record_json FROM workspace_task ORDER BY admission_sequence"
            ).fetchall()
            == stored_before
        )
        assert (
            connection.execute(
                "SELECT task_id, projection_json FROM workspace_task_projection ORDER BY task_id"
            ).fetchall()
            == projections_before
        )


def test_human_and_agent_receive_the_same_refusal_before_task_admission(
    live: LocalPortfolioWebSession,
) -> None:
    spec = {"top_k": 500}
    before = _agent(live, PortfolioResearchAgentRequest(operation="TASKS"))
    status, _headers, body = _request(live, "/api/plan", method="POST", payload={"spec": spec})
    assert status == 400
    browser_refusal = json.loads(body)
    agent_refusal = _agent(live, PortfolioResearchAgentRequest(operation="PLAN", spec=spec))
    assert agent_refusal == browser_refusal
    assert "TOP_K_OUTSIDE_ADMITTED_RANGE" in agent_refusal["refused"]
    assert _agent(live, PortfolioResearchAgentRequest(operation="TASKS")) == before


# =============================================================== the service


def test_the_service_binds_only_to_loopback(live: LocalPortfolioWebSession) -> None:
    """No LAN listener, measured by trying to reach it from a routable address."""

    port = live.web.bound_port  # type: ignore[union-attr]
    assert live.url.startswith("http://127.0.0.1:")
    family_host = live.web._server.server_address[0]  # type: ignore[union-attr]
    assert family_host == "127.0.0.1"

    # Every non-loopback address this machine holds must refuse the connection.
    addresses = {
        info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    }
    reachable = []
    for address in addresses - {"127.0.0.1"}:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(1.5)
        try:
            probe.connect((address, port))
            reachable.append(address)
        except OSError:
            pass
        finally:
            probe.close()
    assert reachable == [], reachable


def test_every_response_carries_the_local_only_headers(live: LocalPortfolioWebSession) -> None:
    status, headers, body = _request(live, "/")
    assert status == 200
    assert headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
    assert "connect-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["Cross-Origin-Opener-Policy"] == "same-origin"
    assert "Access-Control-Allow-Origin" not in headers
    # The document issues no session; the launch URL does (HB).
    assert "Set-Cookie" not in headers
    assert "<title>AlphaLattice · Local Web</title>" in body.decode("utf-8")


def test_assets_are_cached_by_content_hash_and_nothing_else_is(
    live: LocalPortfolioWebSession,
) -> None:
    """Round 96: the host page names each asset by its content-hashed path (the build's
    manifest), and only those paths are immutable; the page, the plain names and every
    API answer stay `no-store`, so an older link reads the current build and the product's
    state never comes from a cache."""

    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "src/alphalattice/interface/local_application/assets/workbench-manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _status, headers, body = _request(live, "/")
    page = body.decode("utf-8")
    assert headers.get("Cache-Control") == "no-store"
    for name in ("workbench-prelude.js", "workbench.css", "workbench.js"):
        hashed = manifest[name]
        assert hashed in page, (name, hashed)
        status, hashed_headers, hashed_body = _request(live, hashed)
        assert status == 200
        assert hashed_headers.get("Cache-Control") == "public, max-age=31536000, immutable"
        status, plain_headers, plain_body = _request(live, f"/{name}")
        assert status == 200
        assert plain_headers.get("Cache-Control") == "no-store"
        assert plain_body == hashed_body
    # the dictionary is not on the page: a zh reader's prelude asks for it
    assert manifest["workbench.zh.js"] not in page
    assert _request(live, manifest["workbench.zh.js"])[0] == 200
    before_links = page.split("<link", 1)[0]
    assert "workbench-prelude" in before_links, "the prelude runs before the stylesheet"
    assert _request(live, "/api/session")[1].get("Cache-Control") == "no-store"
    # U13: the session names each operation's route, the table's and the handled ones.
    routes = _json(live, "/api/session")["routes"]
    assert routes == {
        op: {"method": m, "path": p} for m, p, op in (*OPERATION_ROUTES, *HANDLED_OPERATION_ROUTES)
    }
    assert live.web is not None
    assert len(routes) == len(OPERATION_ROUTES) + len(HANDLED_OPERATION_ROUTES)
    for operation, route in routes.items():
        assert (route["method"], route["path"]) in live.web.application.routes, operation
    assert routes["EVIDENCE_LEDGER"] == {"method": "GET", "path": "/api/evidence-cro/ledger"}
    assert routes["DATA_UPDATE_RUN"] == {"method": "POST", "path": "/api/data-update/run"}
    # The daily update's switch sends the requests its readback offers (U73).
    assert routes["RESEARCH_UPDATE_AUTOMATION_CONFIGURE"] == {
        "method": "POST",
        "path": "/api/research-update/automation",
    }
    assert {"EVIDENCE_REFRESH", "EVIDENCE_SELECT", "CRO_REVIEW", "EVIDENCE_CRO_EXPORT"} <= set(
        routes
    )


def test_no_asset_reaches_outside_the_machine(live: LocalPortfolioWebSession) -> None:
    """No CDN, font host, analytics beacon or absolute URL anywhere in the page."""

    pages = []
    for path in ("/", "/workbench.css", "/workbench.js"):
        _status, _headers, body = _request(live, path)
        pages.append(body.decode("utf-8"))
    joined = "\n".join(pages)
    # A DOM namespace identifies SVG elements; it is not a network resource.
    joined = joined.replace("http://www.w3.org/2000/svg", "")
    for marker in ("http://", "https://", "//cdn", "googleapis", "analytics", "fonts."):
        assert marker not in joined, marker


@pytest.mark.parametrize(
    ("label", "kwargs", "expected"),
    [
        ("bad host", {"host": "evil.example"}, "host_not_allowed"),
        (
            "bad origin",
            {"origin": "http://evil.example"},
            "origin_not_allowed",
        ),
        ("absent origin", {"omit_origin": True}, "origin_absent_on_mutation"),
        (
            "form content type",
            {"content_type": "application/x-www-form-urlencoded"},
            "content_type_not_json",
        ),
        ("absent token", {"token": None}, "session_token_absent"),
        ("absent cookie", {"cookie": None}, "session_token_absent"),
        ("wrong token", {"token": "0" * 43}, "session_token_invalid"),
        ("wrong cookie", {"cookie": "0" * 43}, "session_token_invalid"),
    ],
)
def test_a_mutation_is_refused_before_any_application_work(
    live: LocalPortfolioWebSession, label: str, kwargs: dict[str, Any], expected: str
) -> None:
    """Eight ways to be unauthorised, and no task admitted by any of them."""

    del label
    before = len(_json(live, "/api/tasks")["tasks"])
    status, _headers, body = _request(live, "/api/run", method="POST", payload={}, **kwargs)
    assert status == 403, (status, body[:300])
    assert expected in json.loads(body)["refused"]
    assert live.dispatcher.admissions == 0  # type: ignore[union-attr]
    assert len(_json(live, "/api/tasks")["tasks"]) == before


@pytest.mark.parametrize(
    "path",
    [
        "/api/experiments/training-inputs/prepare",
        "/api/workspace/storage/plan",
        "/api/workspace/storage/cap",
        # HB: plans and previews record what they plan, and the registry does not
        # list them as reads; each was marked read-only beside its row.
        "/api/research-inputs/plan",
        "/api/experiments/training-inputs/plan",
        "/api/research-strategies/plan",
        "/api/workspace/data-issues/preview",
        "/api/workspace/preparation/plan",
        "/api/experiments/plan",
        "/api/experiments/handoff",
        "/api/experiments/foundations/preview",
    ],
)
def test_a_route_that_writes_takes_the_write_check(
    live: LocalPortfolioWebSession, path: str
) -> None:
    """V184: admitting a training-input Task and writing storage plan files are
    writes, refused like any mutation without the session's token; since HB each
    route's check follows its operation in the registry (a POST that is not one of
    its reads), not a flag written beside the route."""

    status, _headers, body = _request(live, path, method="POST", payload={}, token=None)
    assert status == 403, (status, body[:300])
    assert "session_token_absent" in json.loads(body)["refused"]


def test_a_refused_write_ends_its_connection_without_a_reset(
    live: LocalPortfolioWebSession,
) -> None:
    """regression (V551, a flaky write check in two consolidated runs): a write refused before
    its body was answered and its socket closed with the announced body unread, and a socket
    closed with unread bytes is reset, so a client under load read a reset (WinError 10053)
    instead of the refusal, 12 of 720 times in a stress run. The refused request's announced
    body is read and dropped within the admitted bound, so the refusal arrives whole and the
    connection ends cleanly, even when the body arrives after the answer."""

    import socket
    import time

    body = json.dumps({"padding": "x" * 4096}).encode("utf-8")
    port = live.web.bound_port  # type: ignore[union-attr]
    head = "\r\n".join(
        (
            "POST /api/experiments/handoff HTTP/1.1",
            f"Host: 127.0.0.1:{port}",
            f"Origin: http://127.0.0.1:{port}",
            "Content-Type: application/json",
            f"Content-Length: {len(body)}",
            "",
            "",
        )
    ).encode("ascii")
    for _ in range(5):
        with socket.create_connection(("127.0.0.1", port), timeout=10) as client:
            client.sendall(head)
            time.sleep(0.2)  # the refusal is written before the body arrives
            client.sendall(body)
            received = b""
            while chunk := client.recv(65536):
                received += chunk
        assert received.startswith(b"HTTP/1.1 403"), received[:200]
        assert b"session_token_absent" in received, received[-300:]


def _launch_path(live: LocalPortfolioWebSession) -> str:
    selected = urlsplit(live.launch_url)
    return selected.path + ("?" + selected.query if selected.query else "")


def test_only_the_launch_url_gives_a_browser_the_session(
    live: LocalPortfolioWebSession,
) -> None:
    """V183: the session token went to any local caller that asked `/api/session`, and the
    web entry's caller is a person, so any local program could take a person's
    operations. The cookie is issued only at the launch URL, whose key the Host holds
    in memory, and the token is read only with that cookie; a refusal names the way on."""

    token = live.web.application.session_token  # type: ignore[union-attr]
    for path in ("/", "/workbench.html"):
        status, headers, _body = _request(live, path)
        assert status == 200 and "Set-Cookie" not in headers, path
    status, _headers, body = _request(live, "/api/session", cookie=None)
    refused = json.loads(body)
    assert status == 403 and refused["refused"] == "local_web.session_token_absent"
    assert refused["next_action"] == "REOPEN_FROM_LAUNCH_URL"
    assert token.encode() not in body
    status, _headers, body = _request(live, "/api/session", cookie="stale")
    assert status == 403 and json.loads(body)["refused"] == "local_web.session_token_invalid"
    for path in ("/launch", "/launch?key=guess", _launch_path(live) + "&key=again"):
        status, headers, body = _request(live, path)
        assert status == 403 and "Set-Cookie" not in headers, path
        assert json.loads(body)["refused"] == "local_web.launch_key_invalid"
    status, headers, _body = _request(live, _launch_path(live))
    assert status == 303 and headers["Location"] == "/workbench.html"
    assert f"{live.web.application.cookie_name}={token};" in headers["Set-Cookie"]
    assert "HttpOnly" in headers["Set-Cookie"] and "SameSite=Strict" in headers["Set-Cookie"]
    status, _headers, body = _request(live, "/api/session", cookie="valid")
    assert status == 200 and json.loads(body)["session_token"] == token


def test_browser_session_cookies_are_port_scoped_and_restart_replaces_only_its_own():
    """behavior: shared localhost cookie jars keep concurrent Hosts independent by port."""
    import http.cookiejar
    import urllib.error
    import urllib.request

    def application(label):
        web = LocalWebApplication()
        web.add_asset("/workbench.html", label.encode(), "text/html")

        @web.route("GET", "/api/session", session=True)
        def session(_query, _payload):
            return {"session_token": web.session_token, "label": label}

        @web.route("POST", "/api/write", mutates=True)
        def write(_query, _payload):
            return {"written_by": label}

        return web

    def open_url(opener, url):
        request = urllib.request.Request(url)
        try:
            with opener.open(request, timeout=5) as response:
                body = response.read()
                try:
                    value = json.loads(body or b"{}")
                except json.JSONDecodeError:
                    value = body.decode("utf-8", "replace")
                return response.status, value
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"{}")

    def write_url(opener, url, token):
        origin = url.split("/api/", 1)[0]
        request = urllib.request.Request(
            url,
            data=b"{}",
            method="POST",
            headers={
                SESSION_HEADER: token,
                "Origin": origin,
                "Content-Type": "application/json",
            },
        )
        try:
            with opener.open(request, timeout=5) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"{}")

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar)
    )
    first = LocalWebService(application("first"))
    second = LocalWebService(application("second"))
    services = [first, second]
    try:
        first_url = first.start().rstrip("/")
        second_url = second.start().rstrip("/")
        first_port, second_port = first.bound_port, second.bound_port
        assert first_port != second_port

        for service, base in ((first, first_url), (second, second_url)):
            status, _ = open_url(opener, f"{base}/launch?key={service.application.launch_key}")
            assert status == 200  # launch redirects to the served workbench asset

        first_cookie = session_cookie_name(first_port)
        second_cookie = session_cookie_name(second_port)
        assert first_cookie != second_cookie
        assert {cookie.name for cookie in jar} >= {first_cookie, second_cookie}

        first_status, first_session = open_url(opener, f"{first_url}/api/session")
        second_status, second_session = open_url(opener, f"{second_url}/api/session")
        first_token = first.application.session_token
        second_token = second.application.session_token
        assert first_status == second_status == 200
        assert first_session["session_token"] == first_token
        assert second_session["session_token"] == second_token
        assert write_url(opener, f"{first_url}/api/write", first_token)[0] == 200
        assert write_url(opener, f"{second_url}/api/write", second_token)[0] == 200
        status, refused = write_url(opener, f"{first_url}/api/write", second_token)
        assert status == 403 and refused["refused"] == "local_web.session_token_invalid"

        first.stop()
        replacement = LocalWebService(application("replacement"), port=first_port)
        services.append(replacement)
        replacement_url = replacement.start().rstrip("/")
        assert replacement.bound_port == first_port
        old_token = first_token
        new_token = replacement.application.session_token
        assert new_token != old_token
        status, _ = open_url(
            opener,
            f"{replacement_url}/launch?key={replacement.application.launch_key}",
        )
        assert status == 200
        assert open_url(opener, f"{replacement_url}/api/session")[1]["session_token"] == new_token

        old_cookie_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        old_cookie_request = urllib.request.Request(f"{replacement_url}/api/session")
        old_cookie_request.add_header("Cookie", f"{first_cookie}={old_token}")
        with pytest.raises(urllib.error.HTTPError) as error:
            old_cookie_opener.open(old_cookie_request, timeout=5)
        assert error.value.code == 403
        assert open_url(opener, f"{second_url}/api/session")[1]["session_token"] == second_token
        assert write_url(opener, f"{replacement_url}/api/write", new_token)[0] == 200
    finally:
        for service in reversed(services):
            service.stop(timeout=5)


def test_an_unknown_route_is_not_a_filesystem_path(live: LocalPortfolioWebSession) -> None:
    """No route opens a path, a store or a database, including by traversal."""

    for path in (
        "/../pyproject.toml",
        "/etc/passwd",
        "/runtime/artifacts",
        "/api/../app.js%00",
        "/api/store",
    ):
        status, _headers, _body = _request(live, path)
        assert status in {403, 404}, (path, status)


# ============================================================ plan, run, reuse


def test_plan_performs_no_numerical_or_task_work(live: LocalPortfolioWebSession) -> None:
    resolver = live.resolver
    assert isinstance(resolver, _Resolver)
    before_numerical = resolver.numerical_calls
    routes = _json(live, "/api/session")["routes"]
    route = routes["PLAN"]
    selected = {"top_k": 22, "exit_rank": 45, "cost_bps_per_side": "13"}
    plan = _json(live, route["path"], method=route["method"], payload={"spec": selected})

    assert plan["optimizer_call_count"] == 0
    assert resolver.numerical_calls == before_numerical
    assert _json(live, "/api/tasks")["tasks"] == []
    assert live.dispatcher.admissions == 0  # type: ignore[union-attr]
    assert plan["legal_recovery"]
    request = dict(plan["next_requests"]["run"])
    assert request["operation"] == "RUN"
    assert all(str(request["spec"][key]) == str(value) for key, value in selected.items())
    assert request["spec"]["strategy_package_id"] == plan["strategy_package_id"]
    assert request["spec"]["score_source_mode"] == plan["score_source_mode"]
    operation = request.pop("operation")
    admitted = _json(
        live, routes[operation]["path"], method=routes[operation]["method"], payload=request
    )
    assert admitted["disposition"] == "ADMITTED" and admitted["task_id"], admitted
    assert live.dispatcher.admissions == 1  # type: ignore[union-attr]


@pytest.mark.parametrize("prepared_input", [None, "b" * 64])
def test_a_session_portfolio_replan_keeps_its_optional_input_choice(
    live: LocalPortfolioWebSession, monkeypatch: pytest.MonkeyPatch, prepared_input
) -> None:
    """regression: a stopped update's bound request reaches its plan owner whole,
    including an explicit null selector, and admits no Task. Other invalid fields
    are still held at the custom HTTP door before that owner is called.
    """
    assert live.operations is not None and live.operations.updates is not None
    owner = live.operations.updates
    original = owner.plan
    seen = []

    def plan(package_id, input_hash, observed_through):
        seen.append((package_id, input_hash, observed_through))
        return original(package_id, input_hash, observed_through)

    monkeypatch.setattr(owner, "plan", plan)
    route = _json(live, "/api/session")["routes"]["PORTFOLIO_UPDATE_PLAN"]
    payload = {
        "strategy_package_id": "synthetic-uninstalled",
        "prepared_input_hash": prepared_input,
        "observed_through": "2026-09-30",
    }
    answer = _json(live, route["path"], method=route["method"], payload=payload)
    assert answer["status"] == "REFUSED", answer
    assert seen == [("synthetic-uninstalled", prepared_input, date(2026, 9, 30))]
    for invalid in (
        {**payload, "strategy_package_id": None},
        {**payload, "observed_through": None},
        {**payload, "prepared_input_hash": 1},
        {**payload, "unknown": "value"},
    ):
        status, _headers, body = _request(
            live, route["path"], method=route["method"], payload=invalid
        )
        assert status == 400, body
        assert json.loads(body)["refused"] == "portfolio_update.plan_fields_invalid"
    assert len(seen) == 1
    assert live.dispatcher is not None and live.dispatcher.admissions == 0
    assert _json(live, "/api/tasks")["tasks"] == []


def test_run_admission_returns_before_the_work_and_the_task_progresses(
    live: LocalPortfolioWebSession,
    monkeypatch,
) -> None:
    """The whole point of the dispatcher: an answer now, a task that moves after."""

    published, release = threading.Event(), threading.Event()
    execute = PortfolioRunCommand.execute

    def hold_after_publication(command, task_id):
        execute(command, task_id)
        published.set()
        assert release.wait(10), "test did not release the completed command"

    monkeypatch.setattr(PortfolioRunCommand, "execute", hold_after_publication)
    started = time.perf_counter()
    admitted = _json(live, "/api/run", method="POST", payload={})
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert admitted["disposition"] == "ADMITTED"
    assert admitted["task_id"]
    assert admitted["lifecycle"] in {"QUEUED", "RUNNING"}
    # Admission is not the work. A synchronous run of this fixture takes seconds.
    assert elapsed_ms < 2000.0, elapsed_ms

    try:
        assert published.wait(10)
        assert _json(live, "/api/results")["results"]
        held = _json(live, f"/api/status?task_id={admitted['task_id']}")
        assert held["lifecycle"] == "RUNNING"
        assert held["task_id"] == admitted["task_id"]
    finally:
        release.set()
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    final = _json(live, f"/api/status?task_id={admitted['task_id']}")
    assert final["lifecycle"] == "SUCCEEDED"
    assert final["verified_stage_count"] == final["total_stage_count"]
    assert final["worker_failure"] is None


def test_exact_reuse_performs_zero_numerical_work(live: LocalPortfolioWebSession) -> None:
    result_hash = _run_to_completion(live)
    resolver = live.resolver
    assert isinstance(resolver, _Resolver)
    before = resolver.numerical_calls
    admissions = live.dispatcher.admissions  # type: ignore[union-attr]
    tasks = _json(live, "/api/tasks")

    plan = _json(live, "/api/plan", method="POST", payload={})
    assert plan["exact_cache_hit"] is True
    assert plan["estimated_score_replays"] == 0
    assert plan["estimated_risk_surface_builds"] == 0
    request = dict(plan["next_requests"]["run"])
    assert request.pop("operation") == "RUN"
    route = _json(live, "/api/session")["routes"]["RUN"]
    reused = _json(live, route["path"], method=route["method"], payload=request)
    assert reused["disposition"] == "REUSED_EXACT"
    assert reused["result_hash"] == result_hash
    assert reused["task_id"] is None
    assert resolver.numerical_calls == before
    assert live.dispatcher.admissions == admissions  # type: ignore[union-attr]
    assert _json(live, "/api/tasks") == tasks


def test_capacity_is_the_workspaces_own_and_is_reported_truthfully(
    live: LocalPortfolioWebSession,
) -> None:
    """The refusal is Task Control's, surfaced -- not a capacity invented here.

    The workspace admits one active task and at most one queued behind it, and a
    freshly admitted task is queued until the worker picks it up. So *which*
    submission is refused depends on whether the worker has started, and the
    honest assertion is about the invariant rather than about an index: a
    refusal appears quickly, it carries no task id, and it never becomes a
    silent third queue slot.
    """

    application = live.application
    dispatcher = live.dispatcher
    assert application is not None and dispatcher is not None
    # One waiting place, as this test's invariant counts them; the operator sets it (V100).
    write_queue_setting(
        live.workspace / "runtime", 1, chosen_by="HUMAN", chosen_at=datetime.now(UTC)
    )

    submissions = [
        dispatcher.submit(
            PortfolioRunCommand(
                application=application, spec=PortfolioResearchSpec.create(top_k=20 + index)
            )
        )
        for index in range(4)
    ]
    dispositions = [value.disposition for value in submissions]

    assert dispositions[0] == "ADMITTED"
    assert "REFUSED_QUEUE_FULL" in dispositions, dispositions
    # Never more than two outstanding admissions before the first refusal.
    assert dispositions.index("REFUSED_QUEUE_FULL") <= 2, dispositions
    for value in submissions:
        if value.disposition == "REFUSED_QUEUE_FULL":
            assert value.task_id is None
            assert value.refusal_detail
        else:
            assert value.task_id is not None
            assert value.lifecycle in {"QUEUED", "RUNNING"}
    # And the refusal did not consume an admission.
    assert dispatcher.admissions == dispositions.count("ADMITTED")
    dispatcher.drain_for_tests()


def test_a_full_queue_refuses_a_run_before_its_plan(
    live: LocalPortfolioWebSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression (V100): the RUN resolved its sources, scope and caches, then learned the
    queue was full; it asks the queue first and plans nothing."""

    application = live.application
    assert application is not None and live.operations is not None
    registry = live.session.task_control_registry

    def full() -> None:
        raise TaskQueueFull("task_control.queue_full: 1 Tasks wait in the queue's 1 places")

    def never(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a full queue planned the run")

    monkeypatch.setattr(registry, "check_capacity", full)
    monkeypatch.setattr(application, "plan", never)
    answer = live.operations.run({})
    assert answer["disposition"] == "REFUSED_QUEUE_FULL" and answer["task_id"] is None
    assert "queue_full" in str(answer["refusal_detail"])


@pytest.mark.parametrize("unexpected_error", (False, True))
def test_cancellation_is_requested_not_enacted(
    live: LocalPortfolioWebSession, monkeypatch, unexpected_error
) -> None:
    """The dispatcher forwards; the runner decides at its own stage boundary."""

    from alphalattice.control.task_control.runner import TaskControlRunner

    run_next = TaskControlRunner.run_next
    cancellations = []

    def cancel_before_claim(runner, *, expected_task_id=None, expected_task_hash=None):
        # The application has read QUEUED, but the runner has not claimed it.
        # Drive the real HTTP cancel into that exact gap, not a timing lottery.
        cancellations.append(
            _json(live, "/api/cancel", method="POST", payload={"task_id": str(expected_task_id)})
        )
        if unexpected_error:
            raise ValueError("unrelated_worker_defect")
        return run_next(
            runner, expected_task_id=expected_task_id, expected_task_hash=expected_task_hash
        )

    monkeypatch.setattr(TaskControlRunner, "run_next", cancel_before_claim)
    admitted = _json(live, "/api/run", method="POST", payload={})
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    assert len(cancellations) == 1 and cancellations[0]["cancel_requested"] is True
    assert cancellations[0]["task_id"] == admitted["task_id"]
    final = _json(live, f"/api/status?task_id={admitted['task_id']}")
    # The command returned without claiming work; the existing idle finalizer,
    # not a killed worker or this test, owns terminal cancellation.
    assert final["lifecycle"] == "CANCELLED"
    assert final["worker_failure"] == (
        "ValueError: unrelated_worker_defect" if unexpected_error else None
    )


def test_the_dispatcher_joins_its_worker_on_close(tmp_path: Path) -> None:
    """No orphan thread survives shutdown."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-shutdown"),
        resolver=_Resolver(_resolved()),
    )
    session.start()
    _json(session, "/api/run", method="POST", payload={})
    session.stop()

    names = [thread.name for thread in threading.enumerate()]
    assert "local-application-dispatcher" not in names, names
    assert "local-web" not in names, names


def test_a_failed_start_leaves_no_lease_worker_or_socket(tmp_path: Path) -> None:
    """Start is all or nothing. A bind that cannot happen releases everything.

    The lease and the dispatcher worker are acquired *before* the socket, so a
    port collision used to leave a held workspace and a non-daemon thread with
    no URL to stop them through. The proof that they were released is that a
    second session starts on the same workspace and serves.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    taken = int(blocker.getsockname()[1])
    before = {thread.name for thread in threading.enumerate()}
    session = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-failed-start"),
        resolver=_Resolver(_resolved()),
        port=taken,
    )
    try:
        with pytest.raises(OSError):
            session.start()

        assert session.session is None
        assert session.dispatcher is None
        assert session.web is None
        assert {thread.name for thread in threading.enumerate()} - before == set()

        second = LocalPortfolioWebSession(
            workspace=workspace,
            workspace_manifest=_manifest("qa-failed-start"),
            resolver=_Resolver(_resolved()),
        )
        second.start()
        try:
            assert _json(second, "/api/session")["workspace_id"] == "qa-failed-start"
        finally:
            second.stop()
    finally:
        blocker.close()


def test_shutdown_joins_a_slow_authenticated_handler(live: LocalPortfolioWebSession) -> None:
    """A request thread is a writer, so stopping waits for it rather than past it."""

    entered = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def _slow(_query: object, _payload: object) -> object:
        entered.set()
        release.wait(timeout=20.0)
        order.append("handler")
        return {"slow": True}

    live.web.application.route("GET", "/api/qa-slow")(_slow)  # type: ignore[union-attr,arg-type]

    caller = threading.Thread(target=lambda: _request(live, "/api/qa-slow"), daemon=True)
    caller.start()
    assert entered.wait(timeout=20.0)

    def _stop() -> None:
        live.stop()
        order.append("stop")

    stopper = threading.Thread(target=_stop, daemon=True)
    stopper.start()
    # The handler is still inside the application: shutdown has not finished and
    # the workspace lease has not been released.
    time.sleep(0.4)
    assert stopper.is_alive()
    assert live.session is not None

    release.set()
    stopper.join(timeout=30.0)
    caller.join(timeout=30.0)
    assert not stopper.is_alive()
    assert order == ["handler", "stop"], order
    assert live.web is None
    assert live.session is None


def test_a_bounded_stop_will_not_release_the_lease_while_a_command_runs(
    tmp_path: Path,
) -> None:
    """The old bounded join returned with the worker alive. Now it refuses to."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-bounded"),
        resolver=_Resolver(_resolved()),
    )
    session.start()
    running = threading.Event()
    release = threading.Event()

    class _SlowCommand:
        command_kind = "qa_slow_command"

        def admit(self) -> CommandAdmission:
            return CommandAdmission(task_id=uuid4(), lifecycle="QUEUED")

        def execute(self, task_id: UUID) -> None:
            del task_id
            running.set()
            release.wait(timeout=60.0)

    try:
        session.dispatcher.submit(_SlowCommand())  # type: ignore[union-attr,arg-type]
        assert running.wait(timeout=20.0)

        with pytest.raises(LocalWebSessionError):
            session.stop(timeout=0.05)
        assert session.session is not None
        assert session.dispatcher is not None
        assert session.dispatcher.worker_alive

        contender = LocalPortfolioWebSession(
            workspace=workspace,
            workspace_manifest=_manifest("qa-bounded"),
            resolver=_Resolver(_resolved()),
        )
        with pytest.raises(RuntimeError, match="writer is already owned"):
            contender.start()

        release.set()
        session.stop()
        assert session.session is None
    finally:
        release.set()
        with suppress(Exception):
            session.stop()

    successor = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-bounded"),
        resolver=_Resolver(_resolved()),
    )
    successor.start()
    try:
        assert _json(successor, "/api/session")["workspace_id"] == "qa-bounded"
    finally:
        successor.stop()


def test_an_idle_keep_alive_connection_does_not_block_shutdown(
    tmp_path: Path,
) -> None:
    """A browser that walks away must not hold the workspace lease hostage.

    `HTTP/1.1` keeps the connection, so the handler thread sits in a read that a
    tab left open will never satisfy. Shutdown joins request threads -- rightly,
    they are writers -- so without a bound on that read, closing the service
    waits on a browser instead of on its own work.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-keepalive"),
        resolver=_Resolver(_resolved()),
    )
    session.start()
    port = session.web.bound_port  # type: ignore[union-attr]
    idle = socket.create_connection(("127.0.0.1", port), timeout=10.0)
    try:
        idle.sendall(
            "\r\n".join(
                (
                    "GET /api/session HTTP/1.1",
                    f"Host: 127.0.0.1:{port}",
                    # the session read answers the launch URL's browser alone (HB)
                    f"Cookie: {SESSION_COOKIE}={session.web.application.session_token}",  # type: ignore[union-attr]
                    "Connection: keep-alive",
                    "",
                    "",
                )
            ).encode("ascii")
        )
        assert idle.recv(4096).startswith(b"HTTP/1.1 200")
        # The socket stays open and silent, exactly as an abandoned tab leaves it.
        started = time.perf_counter()
        session.stop()
        elapsed = time.perf_counter() - started
    finally:
        idle.close()
        with suppress(Exception):
            session.stop()

    assert elapsed < 30.0, elapsed
    assert session.session is None
    assert "local-web-request" not in {thread.name for thread in threading.enumerate()}


def test_shutdown_drains_current_request_without_admitting_keepalive_followups():
    from http.client import HTTPConnection, HTTPException

    from alphalattice.interface.local_application.web import LocalWebApplication, LocalWebService

    app = LocalWebApplication()
    entered, release = threading.Event(), threading.Event()
    calls = []

    @app.route("GET", "/api/session")
    def read(_query, _body):
        calls.append("read")
        entered.set()
        assert release.wait(5)
        return {"status": "READ"}

    service = LocalWebService(app)
    service.start()
    connection = HTTPConnection("127.0.0.1", service.bound_port, timeout=3)
    try:
        connection.request("GET", "/api/session")
        assert entered.wait(3)
        assert not service.stop(timeout=0.05)  # The active request still owns its work.
        release.set()
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        with pytest.raises((OSError, HTTPException)):
            connection.request("GET", "/api/session")
            connection.getresponse().read()
        assert calls == ["read"]
    finally:
        release.set()
        connection.close()
        assert service.stop(timeout=5)


# ============================================== bounded and preflighted bodies


@pytest.mark.parametrize("failure", [BrokenPipeError, ConnectionResetError, ConnectionAbortedError])
def test_disconnected_response_closes_transport_without_retrying_or_masking_application_errors(
    failure, monkeypatch
):
    from io import BytesIO
    from types import SimpleNamespace

    from alphalattice.interface.local_application.web import (
        AdmittedRequest,
        LocalWebApplication,
        LocalWebResponse,
        _Handler,
    )

    application = LocalWebApplication()
    response = LocalWebResponse(200, b"committed result", "application/json")
    admitted = AdmittedRequest(None, None, {}, 0, False)
    calls = []
    monkeypatch.setattr(application, "preflight", lambda **_: admitted)

    def dispatch(*_args):
        calls.append("dispatch")
        return response

    monkeypatch.setattr(application, "dispatch", dispatch)
    handler = object.__new__(_Handler)
    handler.server = SimpleNamespace(application=application, stopping=threading.Event())
    handler.close_connection = False
    handler.path, handler.headers, handler.rfile = "/", {}, BytesIO()
    handler.send_response = lambda *_: None
    handler.send_header = lambda *_: None
    handler.end_headers = lambda: None

    def disconnected(_body):
        calls.append("write")
        raise failure("closed transport")

    handler.wfile = SimpleNamespace(write=disconnected)
    handler._serve("GET")
    assert calls == ["dispatch", "write"] and handler.close_connection
    monkeypatch.setattr(
        application, "dispatch", lambda *_: (_ for _ in ()).throw(ValueError("owner failure"))
    )
    with pytest.raises(ValueError, match="owner failure"):
        handler._serve("GET")


def _raw(session: LocalPortfolioWebSession, request: bytes) -> tuple[int, bytes]:
    """One handcrafted request. `urllib` cannot lie about `Content-Length`."""

    port = session.web.bound_port  # type: ignore[union-attr]
    connection = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    try:
        connection.sendall(request)
        chunks: list[bytes] = []
        while True:
            try:
                received = connection.recv(4096)
            except TimeoutError:
                break
            if not received:
                break
            chunks.append(received)
        payload = b"".join(chunks)
    finally:
        connection.close()
    status = int(payload.split(b" ", 2)[1]) if payload.startswith(b"HTTP/") else 0
    return status, payload


def _handcrafted(session: LocalPortfolioWebSession, *, token: str, length: str) -> bytes:
    port = session.web.bound_port  # type: ignore[union-attr]
    lines = (
        "POST /api/run HTTP/1.1",
        f"Host: 127.0.0.1:{port}",
        f"Origin: http://127.0.0.1:{port}",
        "Content-Type: application/json",
        f"{SESSION_HEADER}: {token}",
        f"Cookie: {SESSION_COOKIE}={token}",
        f"Content-Length: {length}",
        "",
        "",
    )
    return "\r\n".join(lines).encode("ascii")


def test_an_oversized_body_is_refused_before_it_is_read(
    live: LocalPortfolioWebSession,
) -> None:
    """The bound is checked against the declared length, not the bytes received."""

    token = live.web.application.session_token  # type: ignore[union-attr]
    started = time.perf_counter()
    status, payload = _raw(live, _handcrafted(live, token=token, length=str(200 * 1024 * 1024)))
    elapsed = time.perf_counter() - started

    assert status == 413, payload[:300]
    assert b"local_web.body_too_large" in payload
    assert b"Connection: close" in payload
    # No body was sent at all: a service that tried to read the announced 200 MB
    # would have blocked here until its socket timeout.
    assert elapsed < 4.0
    assert _json(live, "/api/tasks")["tasks"] == []


def test_an_unauthorised_oversized_request_does_no_application_work(
    live: LocalPortfolioWebSession,
) -> None:
    """Refused on authorisation -- before the length, the body and the handler."""

    application = live.web.application  # type: ignore[union-attr]
    served_before = application.served
    status, payload = _raw(live, _handcrafted(live, token="not-the-token", length="104857600"))

    assert status == 403, payload[:300]
    assert b"local_web.session_token_invalid" in payload
    assert b"Connection: close" in payload
    # The route it aimed at admits a task. None exists, so no handler ran.
    assert _json(live, "/api/tasks")["tasks"] == []
    assert _json(live, "/api/results")["results"] == []
    assert application.served == served_before + 2  # the two reads on this line

    deadline = time.monotonic() + 10.0
    while live.web.live_request_threads() and time.monotonic() < deadline:  # type: ignore[union-attr]
        time.sleep(0.02)
    assert live.web.live_request_threads() == ()  # type: ignore[union-attr]


@pytest.mark.parametrize("declared", ["-5", "abc", "12 34", "1.5"])
def test_a_malformed_content_length_is_refused_with_a_stable_reason(
    live: LocalPortfolioWebSession, declared: str
) -> None:
    """Negative, fractional and non-numeric all fail the same named way."""

    token = live.web.application.session_token  # type: ignore[union-attr]
    status, payload = _raw(live, _handcrafted(live, token=token, length=declared))

    assert status == 400, payload[:300]
    assert b"local_web.content_length_invalid" in payload
    assert _json(live, "/api/tasks")["tasks"] == []


# ================================================= recovery and presentation


@pytest.mark.parametrize("never_started", (False, True))
@pytest.mark.parametrize("external_recovery", (False, True, "http"))
def test_a_service_restart_recovers_the_same_task_and_publishes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    never_started: bool,
    external_recovery: bool | str,
) -> None:
    """A genuinely interrupted task is finished by the next service, once."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    interrupting = _InterruptsOnce(_Resolver(_resolved()))
    first = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-restart"),
        resolver=interrupting,
    )
    first.start()
    try:
        with monkeypatch.context() as patch:
            if never_started:
                # HTTP admits the actual Task; simulate death before dispatch.
                patch.setattr(first.dispatcher._work, "put", lambda item: None)
            admitted = _json(first, "/api/run", method="POST", payload={})
            task_id = admitted["task_id"]
            if not never_started:
                first.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        interrupted = _json(first, f"/api/status?task_id={task_id}")
        tasks_before = _json(first, "/api/tasks")["tasks"]
        results_before = _json(first, "/api/results")["results"]
    finally:
        first.stop()

    # The precondition this test exists for. Without it the restart below has
    # nothing to recover and every assertion after it is vacuous.
    assert interrupting.failures == (0 if never_started else 1)
    assert interrupted["lifecycle"] == ("QUEUED" if never_started else "RECOVERY_REQUIRED"), (
        interrupted
    )
    assert len(tasks_before) == 1
    assert results_before == []

    second = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-restart"),
        resolver=_Resolver(_resolved()),
    )
    if external_recovery:
        with monkeypatch.context() as patch:
            patch.setattr(LocalPortfolioWebSession, "resume", lambda *a, **k: ())
            second.start()
        # The injected method was bound during composition; restore the actual port.
        second.operations.recover_task = second.resume
    else:
        second.start()
    try:
        if external_recovery:
            from alphalattice.interface.local_application.client import LocalResearchClient

            recovered = (
                _json(second, "/api/recover", method="POST", payload={"task_id": task_id})
                if external_recovery == "http"
                else LocalResearchClient(workspace).request(
                    {"operation": "RECOVER", "task_id": task_id}
                )
            )
            resumed = tuple(UUID(v) for v in recovered["resumed_task_ids"])
        else:
            resumed = second.resumed_task_ids
        second.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        status = _json(second, f"/api/status?task_id={task_id}")
        tasks = _json(second, "/api/tasks")["tasks"]
        results = _json(second, "/api/results")["results"]
    finally:
        second.stop()

    # The same task, by identity, not a replacement that happens to look alike.
    assert [str(value) for value in resumed] == [task_id]
    assert status["task_id"] == task_id
    assert status["lifecycle"] == "SUCCEEDED", status
    assert [row["task_id"] for row in tasks] == [task_id]
    # One publication, not two: reopened, never re-admitted.
    assert len(results) == 1


@pytest.mark.parametrize("phase", ("WORK", "RESULT_PUBLISHED", "CANCEL"))
def test_local_web_reopens_after_forced_process_death(tmp_path: Path, phase: str) -> None:
    from tests.portfolio_strategy_lab.process_death import forced_process

    workspace = tmp_path / "workspace"
    with forced_process(workspace, __name__, phase) as interrupted:
        pass
    resolver = _Resolver(_resolved())
    service = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-hard-stop"),
        resolver=resolver,
    )
    service.start()
    try:
        service.dispatcher.drain_for_tests()
        task_id = interrupted["task_id"]
        # Read the list first: it must not depend on a status GET repairing it.
        listed = _json(service, "/api/tasks")["tasks"]
        assert len(listed) == 1
        assert listed[0]["lifecycle"] == ("CANCELLED" if phase == "CANCEL" else "SUCCEEDED")
        status = _json(service, f"/api/status?task_id={task_id}")
        task = service.session.task_control_registry.task(UUID(task_id))
        assert task.input.input_hash == interrupted["input_hash"]
        assert len(service.session.task_control_registry.tasks()) == 1
        results = _json(service, "/api/results")["results"]
        if phase == "CANCEL":
            assert status["lifecycle"] == "CANCELLED" and results == []
            assert service.resumed_task_ids == () and resolver.numerical_calls == 0
        else:
            assert service.resumed_task_ids == (UUID(task_id),)
            assert status["lifecycle"] == "SUCCEEDED" and len(results) == 1
            result_hash = results[0]["result_hash"]
            assert _json(service, f"/api/report?result_hash={result_hash}")
            assert _json(service, f"/api/export?result_hash={result_hash}")
            if phase == "RESULT_PUBLISHED":
                assert result_hash == interrupted["result_hash"] and resolver.numerical_calls == 0
    finally:
        service.stop()


def test_browser_close_and_reopen_is_presentation_only(
    live: LocalPortfolioWebSession,
) -> None:
    """A reload issues a new document and changes no task or domain state."""

    result_hash = _run_to_completion(live)
    before_tasks = _json(live, "/api/tasks")["tasks"]
    before_results = _json(live, "/api/results")["results"]

    for _ in range(3):
        status, headers, _body = _request(live, "/")
        assert status == 200
        status, headers, _body = _request(live, _launch_path(live))
        assert status == 303 and headers["Location"] == "/workbench.html"
        assert f"{live.web.application.cookie_name}=" in headers.get("Set-Cookie", "")

    assert _json(live, "/api/tasks")["tasks"] == before_tasks
    assert _json(live, "/api/results")["results"] == before_results
    assert _run_to_completion(live) == result_hash


def test_report_html_is_the_sealed_page_read_back(live: LocalPortfolioWebSession) -> None:
    """Immediately inspectable, and not re-rendered by the service."""

    result_hash = _run_to_completion(live)
    status, headers, body = _request(live, f"/report?result_hash={result_hash}")
    page = body.decode("utf-8")
    sealed = live.service.open_html(result_hash)  # type: ignore[union-attr]

    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert page == sealed
    # The sealed page styles itself inline. The response names that exact style
    # by hash rather than relaxing the policy, so the artifact renders as sealed
    # and nothing else inline is permitted.
    policy = headers["Content-Security-Policy"]
    assert "style-src 'self' 'sha256-" in policy
    assert "unsafe-inline" not in policy
    assert policy.startswith("default-src 'none'")
    assert "connect-src 'self'" in policy
    # A hash covers a `<style>` block and never a `style=` attribute, so a page
    # carrying one would render wrong under its own policy. The renderer keeps
    # every declaration in the hashed block; this is the check that keeps it there.
    assert 'style="' not in page
    # In DEVELOPMENT_REPLAY the page names what it is: a book that was selected
    # and held, not one that is held now.
    assert "Selected historical book and path" in page


def test_the_report_projection_carries_the_window_end_book(
    live: LocalPortfolioWebSession,
) -> None:
    """The workspace shows which names are held, written the way the page writes them."""

    result_hash = _run_to_completion(live)
    body = _json(live, f"/api/report?result_hash={result_hash}")
    book = body["book"]
    sealed = live.service.report(result_hash)  # type: ignore[union-attr]
    page = _request(live, f"/report?result_hash={result_hash}")[2].decode("utf-8")

    assert book["formation_session"] == sealed.window_end_book.formation_session.isoformat()
    assert book["held_count"] == sealed.window_end_book.held_count
    assert book["held_count"] == body["readouts"]["distinct_names_held"]
    assert len(book["positions"]) == len(sealed.window_end_book.positions)

    # The boundary is a typed fact, and both surfaces state the same one. A
    # workspace that said "opened flat" over a page that said otherwise would be
    # two reports of one book.
    assert book["change_boundary"] == sealed.window_end_book.change_boundary
    assert book["change_boundary"] in {
        "PRECEDING_FORMATION",
        "SEALED_CONTINUATION_BOUNDARY",
        "FLAT_PATH_OPENING",
    }
    assert book["preceding_formation_session"] == (
        None
        if sealed.window_end_book.preceding_formation_session is None
        else sealed.window_end_book.preceding_formation_session.isoformat()
    )
    assert book["change_boundary"] in page

    # Every value the browser will show is a string the report owner produced,
    # and the sealed page contains the same string for the same name.
    for projected, position in zip(
        book["positions"], sealed.window_end_book.positions, strict=True
    ):
        assert projected["listing_id"] == position.listing_id
        assert projected["weight"] == format_book_weight(position.weight)
        assert projected["weight_change_bp"] == format_book_change(position.weight_change)
        assert projected["disposition"] == position.disposition
        assert projected["weight"] in page
    assert book["absolute_weight_change_total"] == format_book_weight(
        sealed.window_end_book.absolute_weight_change_total
    )


def test_export_round_trips_from_workspace_id_and_a_spec_path(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """The manifest is enough to ask the same question again, and nothing more."""

    result_hash = _run_to_completion(live)
    manifest = _json(live, f"/api/export?result_hash={result_hash}")

    assert manifest["workspace_id"] == "qa-local-web"
    assert manifest["result_hash"] == result_hash
    assert manifest["command"] == (
        "alphalattice --workspace <dir> strategy-book run --file <spec.json>"
    )
    # The predecessor's root/hash argument list is not restored.
    for banned in ("--root", "--result", "--report-hash", "--program-hash"):
        assert banned not in manifest["command"], banned

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(manifest["spec"]), encoding="utf-8")
    reloaded = json.loads(spec_path.read_text(encoding="utf-8"))
    replayed = _json(live, "/api/plan", method="POST", payload={"spec": reloaded})
    assert replayed["spec_hash"] == manifest["spec_hash"]
    assert replayed["exact_cache_hit"] is True
    assert replayed["cached_result_hash"] == result_hash


def test_export_describes_the_result_not_whatever_is_in_the_form(
    live: LocalPortfolioWebSession,
) -> None:
    """A later plan must not change what an earlier result exports."""

    result_hash = _run_to_completion(live)
    exported = _json(live, f"/api/export?result_hash={result_hash}")

    # The researcher moves on: a different question is previewed, which is what
    # the Explore state now holds.
    moved_on = _json(live, "/api/plan", method="POST", payload={"spec": {"top_k": 22}})
    assert moved_on["spec_hash"] != exported["spec_hash"]

    again = _json(live, f"/api/export?result_hash={result_hash}")
    assert again["spec_hash"] == exported["spec_hash"]
    assert again["spec"] == exported["spec"]
    assert again["result_hash"] == result_hash
    assert again["spec"]["top_k"] != 22


# ================================================= freeze, finalize, masking


def test_freeze_records_the_exact_candidate_and_its_lineage(
    live: LocalPortfolioWebSession,
) -> None:
    result_hash = _run_to_completion(live)
    frozen = _json(live, "/api/freeze", method="POST", payload={"result_hash": result_hash})

    assert frozen["development_result_hash"] == result_hash
    assert frozen["development_task_id"]
    assert frozen["development_run_hash"]
    assert frozen["control_receipt_hash"]
    assert frozen["pre_protected_state_hash"]

    status = _json(live, f"/api/finalization?candidate_hash={frozen['candidate_hash']}")
    assert status["disposition"] == "AWAITING_PROTECTED_AUTHORITY"
    assert status["next_lawful_action"] == "WAIT_FOR_PROTECTED_AUTHORITY"
    assert status["stage_11_action_available"] is False
    assert status["handoff_hash"] is None
    assert status["released_result_hash"] is None
    assert "protected" in status["detail"]


def test_an_unfrozen_candidate_reports_not_frozen(live: LocalPortfolioWebSession) -> None:
    status = _json(live, f"/api/finalization?candidate_hash={'a' * 64}")
    assert status["disposition"] == "NOT_FROZEN"
    assert status["candidate"] is None


def test_process_cleanup_does_not_adopt_children_of_reused_parent_ids() -> None:
    from tests.portfolio_strategy_lab.process_death import _creation_filtered_tree

    parents = {20: 10, 30: 20, 40: 30, 50: 20}
    births = {10: 100, 20: 101, 30: 50, 40: 105, 50: 102}
    # 30 predates its alleged parent. Even its newer child 40 is not ours.
    assert _creation_filtered_tree(10, parents, births.get) == (10, 20, 50)
    births[20] = 90
    assert _creation_filtered_tree(10, parents, births.get) == (10,)


def test_the_evidence_and_cro_section_names_why_it_waits(
    live: LocalPortfolioWebSession,
) -> None:
    """The section is installed and says exactly what it is waiting for.

    Before any run there is no sealed book. After a run there is a development
    book, and this workspace admits no issuer registry or listing authority, so
    the section cannot name an issuer and says so -- rather than showing a
    provisional recommendation a later run would replace. Human and Agent read
    the same typed state and receive the same refusal.
    """

    application = live.web.application  # type: ignore[union-attr]
    registered = {path for _method, path in application.routes}
    assert {"/api/evidence-cro", "/api/evidence-refresh", "/api/cro-review"} <= registered

    body = _json(live, "/api/evidence-cro")
    assert body["state"] == "NO_BOOK_TO_REVIEW"
    assert body["issue_cards"] == []
    assert body["available_actions"] == []
    for path in ("/api/evidence-refresh", "/api/cro-review"):
        refused = _json(live, path, method="POST", payload={})
        assert refused["disposition"] == "REFUSED_NO_BOOK_TO_REVIEW"

    result_hash = _run_to_completion(live)
    body = _json(live, f"/api/evidence-cro?result_hash={result_hash}")
    assert body["state"] == "EVIDENCE_AUTHORITY_NOT_ADMITTED"
    assert "issuer registry" in body["explanation"]
    assert body["available_actions"] == []
    assert body == _json(live, "/api/evidence-cro"), "the latest result is the default book"
    assert body == _agent(
        live, PortfolioResearchAgentRequest(operation="EVIDENCE_CRO", result_hash=result_hash)
    )
    for path in ("/api/evidence-refresh", "/api/cro-review"):
        refused = _json(live, path, method="POST", payload={"result_hash": result_hash})
        assert refused["disposition"] == "REFUSED_NO_ADMITTED_EVIDENCE_AUTHORITY"
    agent_refusal = _agent(
        live, PortfolioResearchAgentRequest(operation="CRO_REVIEW", result_hash=result_hash)
    )
    assert agent_refusal["disposition"] == "REFUSED_NO_ADMITTED_EVIDENCE_AUTHORITY"
    # V405: the preview's refusal answers the commands that complete the setup, the workspace
    # filled in and only a person's choices left open; an installed book names no input.
    preview = _json(live, f"/api/evidence/preview?result_hash={result_hash}")
    assert preview["failure_code"] == "REFUSED_NO_ADMITTED_EVIDENCE_AUTHORITY", preview
    setup = preview["setup"]
    assert setup["pack"]["recipe"] == "hybrid-v2-minilm"
    assert setup["pack"]["install"].endswith("--install hybrid-v2-minilm --network")
    authority = setup["authority"]
    assert f'--workspace "{live.workspace.resolve()}"' in authority["install"]
    assert authority["check"].endswith("--preflight")
    assert authority["install"].endswith("--network-consent --install")
    assert "--research-input-id" not in authority["install"]
    assert authority["choose"][0]["arg"] == "--entities"
    # V591: each choice states what its options must hold, from the one table every refusal on
    # an option names (SETUP_OPTIONS).
    from alphalattice.control.product_host.composition.evidence_review_application import (
        SETUP_OPTIONS,
    )

    for choice in authority["choose"]:
        for flag in choice["arg"].split():
            if flag.startswith("--"):
                option = flag[2:].replace("-", "_")
                assert f"{flag} names {SETUP_OPTIONS[option]}" in choice["why"], choice

    page = _request(live, "/")[2].decode("utf-8")
    script = _request(live, "/workbench.js")[2].decode("utf-8")
    # The two primary actions are the ones the plan names, and nothing here
    # exposes a Task Board, an Agent role, a second report or a hardcoded route
    # (the workbench's script names the owner's route codes only as the words
    # it reads them back in; the page itself carries none).
    assert "Refresh evidence" in script
    assert "Review with the configured Provider" in script
    for word in ("Task Board", "Agent role", "debate"):
        assert word not in script, word
    for route in (
        "NO_MATERIAL_OBJECTION",
        "ACCEPT_WITH_LIMITS",
        "MATERIAL_OBJECTION",
        "DECLINE_ACTIVATION",
    ):
        assert route not in page, route


def test_development_replay_copy_makes_no_forward_claim(
    live: LocalPortfolioWebSession,
) -> None:
    """No `today`, no `current holdings`, no `intended trades`, anywhere a user reads."""

    result_hash = _run_to_completion(live)
    surfaces = [
        _request(live, "/")[2].decode("utf-8"),
        _request(live, "/workbench.js")[2].decode("utf-8"),
        _request(live, f"/report?result_hash={result_hash}")[2].decode("utf-8"),
    ]
    session = _json(live, "/api/session")
    assert session["execution_mode"] == "DEVELOPMENT_REPLAY"
    for surface in surfaces:
        lowered = surface.lower()
        for claim in (
            "today",
            "current holdings",
            "intended trade",
            "will trade",
            "recommended trade",
        ):
            assert claim not in lowered, claim


def test_every_visible_number_equals_a_typed_fact(live: LocalPortfolioWebSession) -> None:
    """The browser renders values; it does not produce them."""

    result_hash = _run_to_completion(live)
    body = _json(live, f"/api/report?result_hash={result_hash}")
    report = live.service.report(result_hash)  # type: ignore[union-attr]
    readouts = live.service.readouts(result_hash)  # type: ignore[union-attr]

    assert body["readouts"]["distinct_names_held"] == report.window_end_distinct_names
    assert body["readouts"]["effective_n"] == report.window_end_effective_n
    assert body["readouts"]["cumulative_net_wealth"] == report.window_cumulative_net_wealth
    assert body["readouts"]["cost_bps_per_side"] == readouts.cost_bps_per_side
    assert body["readouts"]["cost_bps_round_trip"] == readouts.cost_bps_round_trip
    assert body["window"]["selected_session_count"] == report.window_guard.selected_session_count
    assert [tuple(v) for v in body["controls"]] == list(report.control_receipt.selected)


def test_compare_names_differences_without_selecting(live: LocalPortfolioWebSession) -> None:
    first = _run_to_completion(live)
    admitted = _json(
        live,
        "/api/run",
        method="POST",
        payload={"spec": {**_default_spec_document(live), "cost_bps_per_side": "10"}},
    )
    assert admitted["disposition"] == "ADMITTED"
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    results = [item["result_hash"] for item in _json(live, "/api/results")["results"]]
    second = next(value for value in results if value != first)

    body = _json(live, f"/api/compare?left={first}&right={second}")
    assert body == _agent(
        live,
        PortfolioResearchAgentRequest(
            operation="COMPARE",
            left_result_hash=first,
            right_result_hash=second,
        ),
    )
    assert body["disposition"] == "DECLARED_PATH_COMPARISON_NO_SELECTION"
    # Cost never reaches a report field, so a field diff would have missed it.
    assert "cost_bps_per_side" in body["differing_controls"]
    assert body["shares_execution_ledger"] is True
    assert [dimension["dimension"] for dimension in body["dimensions"]] == [
        "HOLDINGS",
        "CONCENTRATION",
        "COST",
        "TURNOVER",
        "PERFORMANCE",
        "RISK",
        "LIMITATIONS",
    ]
    by_dimension = {item["dimension"]: item["metrics"] for item in body["dimensions"]}
    assert {item["label"] for item in by_dimension["HOLDINGS"]} >= {
        "Names held",
        "Absolute weight change",
    }
    assert {item["label"] for item in by_dimension["CONCENTRATION"]} >= {"Effective N"}
    assert {item["label"] for item in by_dimension["COST"]} >= {"Cost per side"}
    assert {item["label"] for item in by_dimension["TURNOVER"]} >= {"Mean one-way turnover"}
    assert {item["label"] for item in by_dimension["PERFORMANCE"]} >= {"Cumulative net wealth"}
    assert {item["label"] for item in by_dimension["RISK"]} >= {"Predicted volatility"}
    assert {item["label"] for item in by_dimension["LIMITATIONS"]} == {"Declared limitations"}
    # No ranking language, and no field that names a chosen side. `exit_rank` is
    # a control id and is not a ranking, so the check is on the vocabulary of
    # selection rather than on any substring.
    encoded = json.dumps(body).lower()
    for word in ("winner", "ranked", "preferred", "recommend", "better", "score_rank"):
        assert word not in encoded, word
    # the product (3de3f325): the reader tells the two comparisons apart by their kind
    assert body["kind"] == "INSTALLED_RESULT_COMPARISON"
    assert set(body) == {
        "kind",
        "disposition",
        "shares_execution_ledger",
        "differing_controls",
        "left",
        "right",
        "dimensions",
    }


def test_plan_leads_with_strategy_work_limits_and_lawful_actions(
    live: LocalPortfolioWebSession,
) -> None:
    plan = _json(live, "/api/plan", method="POST", payload={})

    assert plan["strategy"]["strategy_package_id"] == plan["strategy_package_id"]
    assert plan["strategy"]["package_hash"] == plan["strategy_package_hash"]
    assert plan["available_interval"] == {
        "start": plan["coverage"]["common_watermark_start"],
        "end": plan["coverage"]["common_watermark_end"],
        "sessions": plan["coverage"]["common_session_count"],
        "selected_start": plan["coverage"]["common_watermark_start"],
        "selected_end": plan["coverage"]["common_watermark_end"],
    }
    assert plan["cache"]["state"] == "FULL_NUMERICAL_MISS"
    assert plan["estimated_work"]["basis"] == "UPPER_BOUND_OVER_CANDIDATE_SUPPORT"
    assert plan["estimated_work"]["optimizer_calls"] == 0
    assert plan["limitations"]["capacity"] == "NOT_MODELED"
    assert plan["limitations"]["refused_capability_count"] == len(
        live.service.controls.refusals  # type: ignore[union-attr]
    )
    assert plan["next_lawful_actions"][0] == "ADMIT_DECLARED_PATH_RUN"
    assert {control["control_id"] for control in plan["available_controls"]} == {
        control.control_id
        for control in live.service.controls.controls  # type: ignore[union-attr]
    }
    exit_rank = next(
        control for control in plan["available_controls"] if control["control_id"] == "exit_rank"
    )
    assert exit_rank["min"] == str(PortfolioResearchSpec.default().top_k)
    assert exit_rank["max"]


def _default_spec_document(live: LocalPortfolioWebSession) -> dict[str, Any]:
    manifest_spec = PortfolioResearchSpec.default()
    return {
        "top_k": manifest_spec.top_k,
        "tranches": manifest_spec.tranches,
        "exit_rank": manifest_spec.exit_rank,
        "weight_rule": manifest_spec.weight_rule,
        "cost_bps_per_side": str(manifest_spec.cost.cost_bps_per_side),
        "secondary_benchmark_view": manifest_spec.secondary_benchmark_view,
        "report_unit": manifest_spec.report_unit,
        "study_start": None,
        "study_end": None,
    }


def test_every_registered_route_answers(live: LocalPortfolioWebSession) -> None:
    """Walk the whole routing table, because a route nobody calls is a route
    nobody has tested.

    Two real defects reached a browser before this existed: `/api/controls` read
    field names off the control catalog that the catalog does not have, and then
    iterated a tuple by the wrong name. Both are 500s on a route every other test
    happened to skip.
    """

    result_hash = _run_to_completion(live)
    frozen = _json(live, "/api/freeze", method="POST", payload={"result_hash": result_hash})
    application = live.web.application  # type: ignore[union-attr]

    query = {
        "/api/status": f"?task_id={_json(live, '/api/tasks')['tasks'][0]['task_id']}",
        "/api/report": f"?result_hash={result_hash}",
        "/api/export": f"?result_hash={result_hash}",
        "/report": f"?result_hash={result_hash}",
        "/api/compare": f"?left={result_hash}&right={result_hash}",
        "/api/finalization": f"?candidate_hash={frozen['candidate_hash']}",
    }
    answered: dict[str, int] = {}
    for method, path in sorted(application.routes):
        if method != "GET":
            continue
        if path.startswith("/api/client/"):
            from alphalattice.interface.local_application.client import LocalResearchClient

            client = LocalResearchClient(live.workspace)
            client_body = client.activity() if path == "/api/client/activity" else client.request()
            assert client_body["workspace_id"] == live.workspace_manifest.workspace_id
            assert "refused" not in client_body
            answered[path] = 200
            continue
        status, _headers, body = _request(live, f"{path}{query.get(path, '')}")
        answered[path] = status
        assert status in {200, 400}, (path, status, body[:200])
        if status == 400:
            # A refusal is a typed one, never a stack trace or a 500.
            assert "handler_failed" not in json.loads(body)["refused"], path

    # `/api/compare` refuses two of the same configuration by design; every
    # other read route answers.
    assert answered["/api/controls"] == 200
    assert answered["/api/session"] == 200
    assert answered["/api/results"] == 200
    assert answered["/api/report"] == 200
    assert answered["/api/finalization"] == 200
    for path in ("/", "/workbench.html", "/workbench.css", "/workbench.js"):
        assert _request(live, path)[0] == 200, path


def test_the_control_catalog_projection_matches_the_installed_catalog(
    live: LocalPortfolioWebSession,
) -> None:
    """Field for field, against the owner, so a rename cannot pass silently."""

    body = _json(live, "/api/controls")
    catalog = live.service.controls  # type: ignore[union-attr]

    assert body["catalog_hash"] == catalog.catalog_hash
    assert len(body["controls"]) == len(catalog.controls)
    for projected, control in zip(body["controls"], catalog.controls, strict=True):
        assert projected["control_id"] == control.control_id
        assert projected["label"] == control.label
        assert projected["unit"] == control.unit
        assert projected["value_kind"] == control.value_kind
        assert projected["options"] == list(control.options)
        assert projected["min"] == (None if control.min is None else str(control.min))
        assert projected["max"] == (None if control.max is None else str(control.max))
        assert projected["step"] == (None if control.step is None else str(control.step))
        assert projected["default_display"] == control.default_display
        assert projected["default_value"] == control.default_value
        assert projected["help"] == control.help
        assert projected["guard"] == control.guard
        assert projected["refusal"] == control.refusal
        assert projected["disposition"] == control.disposition
        assert "admitted_surface" not in projected
    assert [item["control_id"] for item in body["refused"]] == [
        item.control_id for item in catalog.refusals
    ]


def test_automation_settings_use_the_same_http_agent_permission_path(live):
    path = "/api/research-update/automation"
    before = len(live.session.task_control_registry.tasks())
    first = _json(live, path)
    assert first["status"] == "DISABLED"
    # Nothing runs forward here, so the daily update offers nothing to turn on (U73).
    assert (first["runs_forward"], first["next_requests"]) == ([], {})
    assert _agent(
        live,
        PortfolioResearchAgentRequest(
            operation="RESEARCH_UPDATE_AUTOMATION_READBACK",
        ),
    ) == _json(live, path)
    saved = _json(
        live,
        path,
        method="POST",
        payload={
            "automation_enabled": False,
            "automation_package_ids": [],
        },
    )
    assert saved["settings"]["chosen_by"] == "HUMAN"
    assert (
        _agent(
            live,
            PortfolioResearchAgentRequest(
                operation="RESEARCH_UPDATE_AUTOMATION_CONFIGURE",
                automation_enabled=False,
                automation_package_ids=(),
            ),
        )
        == saved
    )
    assert (
        _request(
            live,
            path,
            method="POST",
            payload={
                "automation_enabled": "false",
                "automation_package_ids": [],
            },
        )[0]
        == 400
    )
    assert (
        _request(
            live,
            path,
            method="POST",
            payload={
                "automation_enabled": True,
                "automation_package_ids": ["not-installed"],
            },
        )[0]
        == 400
    )
    assert len(live.session.task_control_registry.tasks()) == before
    assert not saved["source_or_fit_authority_granted"]
    workspace, manifest = live.workspace, live.workspace_manifest
    live.stop()
    with LocalPortfolioWebSession(
        workspace=workspace, workspace_manifest=manifest, resolver=_Resolver(_resolved())
    ) as reopened:
        assert _json(reopened, path)["settings"] == saved["settings"]
        assert len(reopened.session.task_control_registry.tasks()) == before


def test_automation_uses_durable_settings_and_sequential_wakes(tmp_path):
    from datetime import UTC, datetime

    from alphalattice.control.product_host.composition.research_update_automation import (
        ResearchUpdateAutomation,
    )
    from alphalattice.control.workspace_runtime.content_store import (
        CommittedIndex,
        ContentAddressedStore,
    )

    calls = []
    reached = threading.Event()

    def execute(request):
        calls.append(request.operation)
        if request.operation == "RESEARCH_UPDATE_PLAN":
            return {"status": "PLANNED", "update_plan_hash": "a" * 64}
        if calls.count("RESEARCH_UPDATE_RUN") % 2 == 0:
            reached.set()
        return {"status": "REUSED_EXACT", "task_id": None}

    index = CommittedIndex(
        tmp_path, ContentAddressedStore(tmp_path, uri_prefix="test://automation")
    )

    def owner():
        return ResearchUpdateAutomation(
            index=index,
            workspace_manifest_hash="b" * 64,
            installed_package_ids=("one", "two"),
            clock=lambda: datetime.now(UTC),
            execute=execute,
        )

    first = owner()
    first.start()
    try:
        assert not reached.wait(0.02) and not calls
        saved = first.configure(enabled=True, package_ids=("one", "two"), chosen_by="HUMAN")
        assert reached.wait(5)
        assert calls == ["RESEARCH_UPDATE_PLAN", "RESEARCH_UPDATE_RUN"] * 2
    finally:
        assert first.close(timeout=None)
    reached.clear()
    second = owner()
    second.start()
    try:
        assert second.readback()["settings"] == saved["settings"]
        # A restart re-asks operations, whose exact reuse retains Task identity.
        assert reached.wait(5)
        assert len(calls) == 8
        second.configure(enabled=False, package_ids=(), chosen_by="INSTALLED_AGENT")
        assert second.readback()["status"] == "DISABLED"
    finally:
        assert second.close(timeout=None)


def test_automation_attends_what_it_admitted_and_never_replans_a_stopped_update(tmp_path):
    """regression (V604): the daily update armed only the next session's ready time, so a
    deferral it admitted held the running place unattended past its retry time. It reads what
    it admitted once the worker is idle: a deferral is resumed at its retry time, an older
    target that published has the ready session planned at once, and a stopped update is left
    to its words -- planned again then, each run would admit a new Task that stops again."""

    from datetime import timedelta

    from alphalattice.control.product_host.composition.research_update_automation import (
        ResearchUpdateAutomation,
    )
    from alphalattice.control.workspace_runtime.content_store import (
        CommittedIndex,
        ContentAddressedStore,
    )

    clock = [datetime(2026, 9, 11, 22, 0, 30, tzinfo=UTC)]  # Friday's bars are due
    task = "11111111-1111-4111-8111-111111111111"
    read: dict[str, object] = {}
    calls: list[str] = []
    seen = threading.Condition()

    def execute(request):
        with seen:
            calls.append(request.operation)
            seen.notify_all()
        if request.operation == "RESEARCH_UPDATE_PLAN":
            return {"status": "PLANNED", "update_plan_hash": "a" * 64}
        if request.operation == "RESEARCH_UPDATE_RUN":
            return {"status": "ADMITTED", "task_id": task, "lifecycle": "QUEUED"}
        assert str(request.task_id) == task
        return dict(read)

    def until(count: int, operation: str) -> None:
        with seen:
            assert seen.wait_for(lambda: calls.count(operation) >= count, timeout=5), calls

    automation = ResearchUpdateAutomation(
        index=CommittedIndex(tmp_path, ContentAddressedStore(tmp_path, uri_prefix="test://a")),
        workspace_manifest_hash="b" * 64,
        installed_package_ids=("one",),
        clock=lambda: clock[0],
        execute=execute,
    )
    automation.start()
    try:
        automation.configure(enabled=True, package_ids=("one",), chosen_by="HUMAN")
        until(1, "RESEARCH_UPDATE_RUN")
        # Its update deferred: read at the next idle, the wake moves to its retry time.
        retry = datetime(2026, 9, 11, 22, 5, tzinfo=UTC)
        read.update(status="DEFERRED", retry_after_at=retry.isoformat())
        automation.command_completed()
        until(1, "RESEARCH_UPDATE_READBACK")
        end = time.monotonic() + 5
        while automation.readback()["next_due_at"] != retry.isoformat():
            assert time.monotonic() < end, automation.readback()
            time.sleep(0.01)
        assert calls.count("RESEARCH_UPDATE_PLAN") == 1
        # At its retry time the update is planned again: its plan resumes the same Task.
        clock[0] = retry + timedelta(minutes=1)
        assert automation.wake.signal_due_work(clock[0])
        until(2, "RESEARCH_UPDATE_RUN")
        # It published Friday on Monday, when Monday's bars are ready: Monday is planned at once.
        clock[0] = datetime(2026, 9, 14, 22, 30, tzinfo=UTC)
        read.clear()
        read.update(status="PROPOSAL_PUBLISHED", update={"target_session": "2026-09-11"})
        automation.command_completed()
        until(3, "RESEARCH_UPDATE_RUN")
        # Monday's stops: left to its words, and no plan follows however often the worker idles.
        read.clear()
        read.update(
            status="BLOCKED",
            failure_code="research_update.binding_changed",
            update={"target_session": "2026-09-11"},
        )
        automation.command_completed()
        until(3, "RESEARCH_UPDATE_READBACK")
        for _ in range(3):
            automation.command_completed()
        time.sleep(0.5)
        assert calls.count("RESEARCH_UPDATE_RUN") == 3, calls
        assert calls.count("RESEARCH_UPDATE_READBACK") == 3, calls
    finally:
        assert automation.close(timeout=None)


def test_a_chinese_request_is_sent_in_utf8_within_the_body_limit() -> None:
    """requirement (V147): two comments of 11,000 Chinese characters each fit the Host's body
    limit as their text does; escaped, the same request was twice its size and was refused."""

    from alphalattice.interface.local_application import cli_contract, client

    comment = "风险" * 5500
    document = {"operation": "EVIDENCE_REVIEW", "comments": [comment, comment]}
    body = client.request_body(document)
    assert len(body) < cli_contract.MAXIMUM_REQUEST_BODY_BYTES
    assert json.loads(body.decode("utf-8")) == document
    assert len(json.dumps(document).encode()) > cli_contract.MAXIMUM_REQUEST_BODY_BYTES


def _crash_child(workspace: str, phase: str) -> None:
    resolver = _Resolver(_resolved())
    service = LocalPortfolioWebSession(
        workspace=Path(workspace),
        workspace_manifest=_manifest("qa-hard-stop"),
        resolver=resolver,
    )
    resolve = resolver.resolve
    verify = PortfolioResearchTaskAdapter.verify_stage

    def stop(result_hash=None):  # type: ignore[no-untyped-def]
        registry = service.session.task_control_registry
        task = registry.active_task()
        assert task is not None
        if phase == "CANCEL":
            task, _ = registry.request_cancel(
                task_id=task.task_id,
                expected_task_hash=task.record_hash,
                observed_at=service.clock(),
            )
        print(
            "CRASH_BOUNDARY:"
            + json.dumps(
                {
                    "pid": os.getpid(),
                    "task_id": str(task.task_id),
                    "record_hash": task.record_hash,
                    "input_hash": task.input.input_hash,
                    "result_hash": result_hash,
                }
            ),
            flush=True,
        )
        threading.Event().wait()

    def resolve_or_stop(**kwargs):  # type: ignore[no-untyped-def]
        if phase in {"WORK", "CANCEL"}:
            stop()
        return resolve(**kwargs)

    def verify_or_stop(self, **kwargs):  # type: ignore[no-untyped-def]
        if phase == "RESULT_PUBLISHED":
            stop(kwargs["evidence"][0].content_hash)
        return verify(self, **kwargs)

    resolver.resolve = resolve_or_stop
    PortfolioResearchTaskAdapter.verify_stage = verify_or_stop
    service.start()
    _json(service, "/api/run", method="POST", payload={})
    service.dispatcher.drain_for_tests()
    service.stop()


def test_a_published_input_refreshes_every_manifest_holder(
    live: LocalPortfolioWebSession,
) -> None:
    """requirement (RX, V182): after an input is published every application reads
    the manifest the Host refreshed. They read one holder, so none keeps a copy of
    its own to fall behind and refuse its next plan (`workspace_manifest_changed`)
    though its inputs had not moved. The update automation keeps the hash its
    settings are approved under, so it asks for approval again, as a restart
    would."""

    from types import SimpleNamespace

    operations = live.operations
    assert operations is not None and operations.automation is not None
    applications = (
        operations.experiments,
        operations.data_update,
        operations.scoring,
        operations.calibration,
        operations.updates,
    )
    assert all(application is not None for application in applications)
    current = SimpleNamespace(manifest_hash="b" * 64)
    operations._hold(current)  # type: ignore[arg-type]

    assert operations.workspace_manifest is current
    assert all(application.manifest is current for application in applications)  # type: ignore[union-attr]
    assert operations.automation.manifest_hash == current.manifest_hash


def test_an_authored_id_is_sent_as_written_whatever_the_ledger_holds(tmp_path):
    """regression (V561, the user's review at de555b07; V544 in part): a criterion named
    `abcdefabcdef` was rewritten to a ledger hash that began so, since restoration read every
    field ending `_id`. Only a field the request contract types as an issued reference is read
    back, an issued history entry id among them; an authored id is sent as written with no, one
    or several ledger matches, and a typed reference still restores and refuses ambiguity."""

    import pytest

    from alphalattice.control.product_host.composition.reference_prefixes import ReferenceLedger
    from alphalattice.interface.local_application.client import (
        LocalResearchClientError,
        whole_references,
    )

    authored = "abcdefabcdef"
    first, second = authored + "0" * 52, authored + "1" * 52
    task = "927be8a3-305b-5237-9f9e-f765da4fa147"
    declared = {"goal_declaration": {"criteria": [{"criterion_id": authored, "text": "t"}]}}
    for held in ([], [first], [first, second]):
        if held:
            ReferenceLedger(tmp_path).record(held)
        request = {"operation": "GOAL_OPEN", **declared}
        whole_references(request, tmp_path)
        assert request["goal_declaration"]["criteria"][0]["criterion_id"] == authored
        for field in ("research_input_id", "component_id", "strategy_package_id"):
            sent = {"operation": "X", field: authored}
            whole_references(sent, tmp_path)
            assert sent[field] == authored, field
    ReferenceLedger(tmp_path).record({"next": [f"x --task-id {task}"]})
    entry = {"operation": "RESEARCH_HISTORY", "history_entry_id": f"experiment:{task[:12]}"}
    whole_references(entry, tmp_path)
    assert entry["history_entry_id"] == f"experiment:{task}"
    with pytest.raises(LocalResearchClientError, match="ambiguous"):
        whole_references({"result_hash": authored}, tmp_path)


def test_a_short_reference_is_read_back_as_the_one_value_it_begins(tmp_path):
    """requirement (V393): the Host keeps every hash and id it answers a client with, and the
    client reads a value of the shape the compact display gives one as the one value that
    begins so, at any depth, before the request is sent; one that begins several is refused
    with them, one that begins none is refused under a `_hash` key and elsewhere sent as
    written, since a name may take that shape."""

    import pytest

    from alphalattice.control.product_host.composition.reference_prefixes import ReferenceLedger
    from alphalattice.interface.local_application.client import (
        LocalResearchClientError,
        whole_references,
    )

    plan = "ab" * 32
    twin = "ab" * 6 + "cd" * 26
    task = "927be8a3-305b-5237-9f9e-f765da4fa147"
    ReferenceLedger(tmp_path).record({"plan_hash": plan, "next": [f"x --task-id {task}"]})
    request = {
        "operation": "EXPERIMENT_RUN",
        "experiment_plan_hash": plan[:12],
        "goal_submission": {"references": [{"request": {"task_id": task[:12]}}]},
        "research_input_id": "20261001-001",
    }
    whole_references(request, tmp_path)
    assert request == {
        "operation": "EXPERIMENT_RUN",
        "experiment_plan_hash": plan,
        "goal_submission": {"references": [{"request": {"task_id": task}}]},
        "research_input_id": "20261001-001",
    }
    with pytest.raises(LocalResearchClientError, match="unknown") as unknown:
        whole_references({"experiment_plan_hash": "0" * 12}, tmp_path)
    assert unknown.value.short_reference == {
        "field": "experiment_plan_hash",
        "value": "0" * 12,
        "candidates": [],
    }
    ReferenceLedger(tmp_path).record([twin, plan])
    with pytest.raises(LocalResearchClientError, match="ambiguous") as ambiguous:
        whole_references({"a": [{"result_hash": plan[:12]}]}, tmp_path)
    assert ambiguous.value.short_reference == {
        "field": "a.0.result_hash",
        "value": plan[:12],
        "candidates": sorted([plan, twin]),
    }
    assert (tmp_path / "runtime/reference-ledger.txt").read_text().split() == [plan, task, twin]


def test_a_watched_task_reads_back_from_the_short_id_the_display_gave(tmp_path):
    """regression (V488, the user's review): `activity list --watch` sent the compact display's
    short Task id as text and the feed refused it (`activity.watch_invalid`), while `task show`
    read the same short id back. The watch list is typed as Task references, so each short id
    reads back as its whole id before the request is sent."""

    from alphalattice.control.product_host.composition.reference_prefixes import ReferenceLedger
    from alphalattice.interface.local_application.client import whole_references

    task = "927be8a3-305b-5237-9f9e-f765da4fa147"
    ReferenceLedger(tmp_path).record([task])
    request = {"operation": "ACTIVITY_LIST", "watch": [task[:12]]}
    whole_references(request, tmp_path)
    assert request == {"operation": "ACTIVITY_LIST", "watch": [task]}


def test_authored_text_is_never_a_reference_and_one_reference_has_two_spellings(tmp_path):
    """requirement (V399, the CLI review's F1 and F2): only a field the request contract types
    as a hash or an id reads a shown beginning back, so authored text equal to one stays byte
    for byte and a typed field refuses a beginning the Host never answered with; a continuation
    takes a bound whole value and its shown beginning as one reference, the whole kept, and
    still refuses a different value."""

    import json

    import pytest

    from alphalattice.control.product_host.composition.reference_prefixes import ReferenceLedger
    from alphalattice.interface.local_application.client import (
        LocalResearchClientError,
        continuation,
        whole_references,
    )

    plan = "abcdef123456" + "0" * 52
    task = "927be8a3-305b-5237-9f9e-f765da4fa147"
    ReferenceLedger(tmp_path).record([plan, task])
    request = {
        "operation": "GOAL_OPEN",
        "goal_declaration": {"title": plan[:12], "objective": plan[:12], "notes": [task[:12]]},
        "goal_submission": {"references": [{"request": {"task_id": task[:12]}}]},
    }
    whole_references(request, tmp_path)
    assert request["goal_declaration"] == {
        "title": plan[:12],
        "objective": plan[:12],
        "notes": [task[:12]],
    }
    assert request["goal_submission"]["references"][0]["request"]["task_id"] == task
    with pytest.raises(LocalResearchClientError, match="unknown"):
        whole_references({"task_id": "00000000-000"}, tmp_path)

    answer = tmp_path / "answer.json"
    answer.write_text(
        json.dumps(
            {"next_requests": {"show": {"operation": "EXPERIMENT_READBACK", "task_id": task}}}
        ),
        encoding="utf-8",
    )
    allowed = frozenset({"task_id"})
    chosen = continuation("EXPERIMENT_READBACK", answer, {"task_id": task[:12]}, allowed)
    assert chosen == {"operation": "EXPERIMENT_READBACK", "task_id": task}
    with pytest.raises(LocalResearchClientError, match="bound_reference_override:task_id"):
        continuation(
            "EXPERIMENT_READBACK",
            answer,
            {"task_id": "00000000-0000-0000-0000-000000000000"},
            allowed,
        )


def test_a_books_risk_window_and_its_bound_section_are_refused_with_the_way_on():
    """regression (V501, V502; the fork's research chain on ls1-chain): a book refused for a
    Risk study short of its formations said "run it over the Alpha study's window", which a
    Risk study then refused, wordless, for its first sessions' missing lookback; and a changed
    window answered `authority_field_mismatch` with no words. Each names what holds and the
    way on: the Alpha study's scored formations, read from the Alpha study; the draft's bound
    `experiment` section."""

    import json

    from alphalattice.control.product_host.composition.plain_refusals import explain
    from alphalattice.interface.local_application import cli_contract

    source = {"task_id": "a-1", "candidate_id": "c-1"}
    short = explain("portfolio_research.risk_session_axis_incomplete", portfolio=source)
    assert "scored formations" in short["detail"] and "lookback" in short["detail"]
    assert short["next_requests"]["alpha"] == {"operation": "EXPERIMENT_READBACK", "task_id": "a-1"}
    bound = explain("portfolio_research.authority_field_mismatch", portfolio=source)
    assert bound["fields"] == [["experiment"]] and "bound to its draft" in bound["detail"]
    assert bound["next_requests"]["portfolio_draft"]["task_id"] == "a-1"
    words = json.loads(
        Path(cli_contract.__file__).with_name("refusal_words.json").read_text("utf-8")
    )
    unavailable = words.get("risk_research.formation_selection_requested_axis_unavailable")
    assert unavailable and unavailable["next_action"] == "START_THE_WINDOW_AFTER_THE_LOOKBACK"


def test_a_book_stopped_on_unsupported_returns_reads_its_owners_words_and_way_on():
    """regression (V504, RR5): a first-use book declared `require_complete`, as the Skill asked
    without the person's authorization, and stopped on `benchmark_support_absent`, whose
    recovery view said only that its owner recorded the code. The owner words both causes and
    the way on (the draft's declared quarantine, which a first-use goal's delegation gives),
    and a Task stopped on any worded owner code reads those words in its recovery view."""

    from alphalattice.control.product_host.composition.plain_refusals import explain
    from alphalattice.control.product_host.composition.task_recovery import stop_detail

    code = "portfolio_research.benchmark_support_absent"
    worded = explain(code, portfolio={"task_id": "a-1", "candidate_id": "c-1"})
    assert "require_complete" in worded["detail"] and "first-use" in worded["detail"]
    assert worded["fields"] == [["portfolio", "unavailable_return_policy"]]
    assert worded["next_requests"]["portfolio_draft"]["task_id"] == "a-1"
    assert stop_detail("research_experiment", code, "TASK_CONTROL") == explain(code)["detail"]
    unworded = stop_detail("research_experiment", "an_owner.code_without_words", "TASK_CONTROL")
    assert unworded.startswith("The Task stopped at a governed boundary")


def test_every_task_stop_sentence_fits_its_bound_and_reads_whole() -> None:
    """requirement (V565, V504, the class): a Task stopped on a code reads its owner's words
    whole in its recovery view, whose bound is `STOP_WORDS_BOUND`. Every registered code an
    owner words, every Evidence unit failure a Task stops on as an unprepared unit -- each code
    the Evidence package raises, a wrapped task failure, the source shortfalls with their facts,
    an unworded code as long as a failure code is kept -- a book's support causes and every data
    Task's stop fit it. V541's unit words had run to 729 and 860 characters, and the Task Center
    read them cut mid-word."""

    import ast
    import re

    from alphalattice.control.guanyin.data.workspace_maintenance import (
        maintenance_failure_detail,
    )
    from alphalattice.control.product_host.composition.plain_refusals import (
        SOURCE_SHORT_CODES,
        STOP_WORDS_BOUND,
        explain,
    )
    from alphalattice.control.product_host.composition.task_recovery import stop_detail

    repo = Path(__file__).resolve().parents[2]
    registered = json.loads((repo / "config/registries/refusals.json").read_text("utf-8"))
    worded = {}
    for code in registered["entries"]:
        try:
            words = explain(code).get("detail")
        except (KeyError, TypeError, ValueError):
            continue
        if words:
            worded[code] = words
    assert len(worded) > 50
    evidence = {
        code
        for path in (repo / "src/alphalattice/evidence/alternative_evidence").rglob("*.py")
        for code in re.findall(r'"(alternative_evidence\.[a-z_]+)', path.read_text("utf-8"))
    }
    units = {
        *evidence,
        "alternative_evidence.task_failed:knowledge.semantic_pack_not_pinned",
        *(code + ":0 of 78 issuers hold a source, 47 needed" for code in SOURCE_SHORT_CODES),
        "an_owner." + "x" * (120 - len("an_owner.")),
    }
    read_whole = [
        *("alternative_evidence.unit_not_prepared:" + code for code in sorted(units)),
        *(
            f"portfolio_research.benchmark_support_absent:{cause}:SUBJECT-7f3a"
            for cause in ("returns_unavailable", "no_eligible_name")
        ),
    ]
    for code in read_whole:
        worded[code] = explain(code)["detail"]
        assert stop_detail("research_experiment", code, "TASK_CONTROL") == worded[code], code
    long = {code: len(words) for code, words in worded.items() if len(words) > STOP_WORDS_BOUND}
    assert long == {}, long
    tree = ast.parse(
        (repo / "src/alphalattice/control/guanyin/data/workspace_maintenance.py").read_text("utf-8")
    )
    data = [
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]
    assert data and all(len(maintenance_failure_detail(code)) <= STOP_WORDS_BOUND for code in data)


def test_a_support_refusal_is_worded_with_its_cause_and_the_books_way_on(live, monkeypatch):
    """regression (V511, RR5): the words, the Task's status and its recovery view gave the
    support refusal no cause and no bound way on: its draft request named no Alpha study. Each
    cause is worded with where it holds and its own way on, bound to the Alpha study and
    candidate the stopped study's plan names; the bare code keeps V504's words."""

    from alphalattice.control.product_host.composition.plain_refusals import explain

    code = "portfolio_research.benchmark_support_absent"
    book = {"task_id": "alpha-1", "candidate_id": "alpha-candidate-1"}
    facts = "2 names,3 sessions,2020-03-03..2020-03-05"
    unavailable = explain(f"{code}:returns_unavailable:{facts}", portfolio=book)
    assert f"({facts})" in unavailable["detail"] and "quarantine_listings" in unavailable["detail"]
    assert unavailable["next_requests"]["portfolio_draft"] == {
        "operation": "EXPERIMENT_PORTFOLIO_DRAFT",
        "task_id": "alpha-1",
        "candidate_id": "alpha-candidate-1",
    }
    empty = explain(f"{code}:no_eligible_name:1 sessions,2020-03-06..2020-03-06", portfolio=book)
    assert "(1 sessions,2020-03-06..2020-03-06)" in empty["detail"]
    assert "either unavailable-return policy" in empty["detail"]
    assert empty["next_requests"] == {
        "alpha_draft": {"operation": "EXPERIMENT_DRAFT", "task_id": "alpha-1"}
    }
    assert explain(code)["detail"].startswith(
        "The book's walk met a formation session with no eligible name, or an eligible name"
    )
    # The stopped Task's way on reads the book its plan named, for a Portfolio code only.
    # Replanning also reads the retained Task through the composed owner's registry (V615).
    host = live.operations
    assert host is not None and live.application is not None
    asked: list[object] = []
    monkeypatch.setattr(host.experiments, "book_source", lambda task: asked.append(task) or book)
    task = live.application.admit(spec=PortfolioResearchSpec.create()).task_id
    way = host._stopped_way(task, f"{code}:returns_unavailable:{facts}", "BLOCKED")
    assert way["next_requests"]["portfolio_draft"]["task_id"] == "alpha-1" and asked == [task]
    host._stopped_way(task, "research_experiment.execution_binding_changed", "BLOCKED")
    assert asked == [task]


def test_prepared_training_inputs_offer_their_components_study(tmp_path):
    """regression (V530, RR5d): a succeeded training-input readback offered only its own read,
    so the agent searched the source for the next step. Prepared inputs offer each component's
    lifecycle study controls, bound to the input they were prepared on, and the Host's request
    contract takes the offered request as it stands."""

    from datetime import UTC, datetime
    from types import SimpleNamespace
    from uuid import uuid4

    from alphalattice.control.product_host.data_preparation.model_training import (
        ModelTrainingInputApplication,
        ModelTrainingInputPlan,
    )
    from alphalattice.control.task_control.contracts import TaskLifecycle
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    plan = ModelTrainingInputPlan.create(
        workspace_id="w",
        workspace_manifest_hash="m" * 64,
        input_id="factor-development",
        input_binding_hash="b" * 64,
        component_ids=("G2_R0_TREND",),
        listing_count=466,
        source_session_count=1264,
        implementation_hash="i" * 64,
    )
    task = SimpleNamespace(lifecycle=TaskLifecycle.SUCCEEDED, failure_code=None)
    session = SimpleNamespace(
        workspace=tmp_path, task_control_registry=SimpleNamespace(task=lambda _t: task)
    )
    owner = ModelTrainingInputApplication(
        session,  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 10, 2, tzinfo=UTC),
    )
    owner._of = lambda _task: plan  # type: ignore[method-assign]
    owner._publication = lambda _hash: SimpleNamespace(preparation_receipt_hash="r")  # type: ignore[method-assign]
    receipt = SimpleNamespace(bindings=(), model_dump=lambda mode: {})
    owner.store = SimpleNamespace(_load=lambda *_args: receipt)  # type: ignore[assignment]
    offered = owner.readback(uuid4())["next_requests"]
    controls = offered["controls:G2_R0_TREND"]
    assert controls == {
        "operation": "EXPERIMENT_CONTROLS",
        "research_input_id": "factor-development",
        "input_binding_hash": "b" * 64,
        "experiment_kind": "alpha.model-development",
        "component_id": "G2_R0_TREND",
    }
    PortfolioResearchOperationRequest(**controls)
    task.lifecycle = TaskLifecycle.RUNNING
    assert set(owner.readback(uuid4())["next_requests"]) == {"readback"}


def test_a_strategy_short_of_its_risk_window_names_it_and_offers_the_risk_study(tmp_path):
    """regression (V533, RR5d): a strategy plan whose Risk study did not cover its calibrated
    Alpha study's matured formations was refused `research_strategy.risk_parent_support_incomplete`
    with no words, no window and no request, and the agent found the window by reading the Risk
    study. The refusal names the window needed and the one the Risk study covers, its words
    say the way on, and it offers the Risk study's controls on the strategy's own input. Every
    code a strategy's declaration or readback can meet is worded."""

    import re
    from types import SimpleNamespace

    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceManifest,
        publish_research_workspace_manifest,
    )
    from alphalattice.interface.local_application.cli_contract import refusal_words
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    code = (
        "research_strategy.risk_parent_support_incomplete:"
        "needed 2019-10-07..2024-09-30, Risk study 2026-09-01..2026-09-30"
    )

    def plan(_document: object) -> None:
        raise ValueError(code)

    publish_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("s"))
    host = PortfolioResearchOperations.__new__(PortfolioResearchOperations)
    host.research_strategies = SimpleNamespace(plan=plan)  # type: ignore[assignment]
    host.workspace_session = SimpleNamespace(workspace=tmp_path)  # type: ignore[assignment]
    binding = "b" * 64
    answer = host._workspace_operation(
        PortfolioResearchOperationRequest(
            operation="RESEARCH_STRATEGY_PLAN", experiment_document={"input_binding_hash": binding}
        ),
        caller="HUMAN",
    )
    assert answer is not None and answer["failure_code"] == code
    assert answer["next_requests"] == {
        "risk_controls": {
            "operation": "EXPERIMENT_CONTROLS",
            "research_input_id": None,
            "input_binding_hash": binding,
            "experiment_kind": "risk.covariance-development",
        }
    }
    words = refusal_words(code)
    assert "needed 2019-10-07..2024-09-30, Risk study 2026-09-01..2026-09-30" in words["detail"]
    assert words["next_action"] == "STUDY_RISK_OVER_THE_NEEDED_WINDOW"
    source = (
        Path(__file__).resolve().parents[2]
        / "src/alphalattice/control/product_host/data_preparation/research_strategy.py"
    ).read_text(encoding="utf-8")
    internal = {  # integrity checks of the owner's own records, which no declaration reaches
        "research_strategy.evidence_binding_mismatch",
        "research_strategy.evidence_invalid",
        "research_strategy.materialized_inputs",
        "research_strategy.plan_invalid",
        "research_strategy.retry_requested",
        "research_strategy.stage_unknown",
        "research_strategy.task_contract_invalid",
    }
    reachable = set(re.findall(r'"(research_strategy\.[a-z_]+)', source)) - internal
    assert reachable and not [c for c in sorted(reachable) if not refusal_words(c)], reachable


def test_strategy_controls_offer_each_missing_components_first_step(tmp_path):
    """regression (V505, RR5): a fresh workspace asked to run a book forward read strategy
    controls naming the required components with no completed lifecycle study and offering
    only the strategy's plan, so its agent found no way to the components. Each missing
    component offers its first step, the input left to choose where the workspace has none
    or several; a component a completed lifecycle study holds is not missing."""

    from datetime import UTC, datetime
    from types import SimpleNamespace

    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceManifest,
        publish_research_workspace_manifest,
    )
    from alphalattice.control.product_host.data_preparation.research_strategy import (
        ResearchStrategyPreparation,
    )

    publish_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("fwd"))
    studies: list[dict[str, object]] = []
    owner = ResearchStrategyPreparation(
        SimpleNamespace(workspace=tmp_path),  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 10, 2, tzinfo=UTC),
        read_experiment=lambda _task: {},
        list_experiments=lambda: {"experiments": studies},
    )
    controls = owner.controls()
    required = controls["required_components"]
    assert required and controls["missing_components"] == required
    for component in required:
        assert controls["next_requests"][f"component:{component}"] == {
            "operation": "MODEL_TRAINING_INPUT_PLAN",
            "research_input_id": None,
            "component_id": component,
        }
    studies.append(
        {
            "lifecycle": "SUCCEEDED",
            "kind": "alpha.model-development",
            "component_recipe_id": required[0],
        }
    )
    again = owner.controls()
    assert again["missing_components"] == required[1:]
    assert f"component:{required[0]}" not in again["next_requests"]
    assert again["next_requests"]["plan"]["operation"] == "RESEARCH_STRATEGY_PLAN"
    # Risk is offered beside the components until a completed Risk study covers the window a
    # calibrated Alpha study names (FLOW-3); no window is known before one completes.
    assert again["risk_windows"] == []
    assert again["next_requests"]["risk"] == {
        "operation": "EXPERIMENT_CONTROLS",
        "research_input_id": None,
        "experiment_kind": "risk.covariance-development",
    }
    studies.append(
        {
            "lifecycle": "SUCCEEDED",
            "kind": "risk.covariance-development",
            "sessions": {"start": "2020-01-02", "end": "2026-09-10"},
        }
    )
    assert "risk" not in owner.controls()["next_requests"]


def test_the_first_intent_is_the_way_forward_and_names_it_in_words() -> None:
    """regression (FLOW-3, AX's first use of 2026-10-07): the lead followed the inputs' Factor
    flows, read first, and the Lab book they lead to was never the one a strategy activates.
    `workspace show` lists `RUN_FORWARD` first: with no strategy installed it names the
    research strategy's controls and why, and an installed strategy with no book yet offers
    its book's controls instead of being left out."""

    from types import SimpleNamespace

    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        INSTALLED_BOOK_WORDS,
        RUN_FORWARD_WORDS,
        PortfolioResearchOperations,
    )

    forward = PortfolioResearchOperations._forward_intents
    (none,) = forward(SimpleNamespace(_packages={}, activations=None))  # type: ignore[arg-type]
    assert none["status"] == "NO_RESEARCH_STRATEGY_INSTALLED"
    assert none["detail"] == RUN_FORWARD_WORDS and "need no Factor study" in RUN_FORWARD_WORDS
    assert none["next_requests"] == {
        "strategy_controls": {"operation": "RESEARCH_STRATEGY_CONTROLS"}
    }
    installed = SimpleNamespace(
        _packages={"PKG": object()},
        activations=object(),
        _activation_offer=lambda _package: {"status": "INACTIVE"},
    )
    (bookless,) = forward(installed)  # type: ignore[arg-type]
    assert (bookless["status"], bookless["detail"]) == ("NO_BOOK_YET", INSTALLED_BOOK_WORDS)
    assert bookless["next_requests"] == {
        "books": {"operation": "CONTROLS", "strategy_package_id": "PKG"}
    }


def test_a_strategy_short_of_risk_history_names_the_start_and_offers_the_risk_study(tmp_path):
    """regression (V596, DOC's RR5g-0 F3): a strategy plan admitted a Risk study whose evaluated
    window covered the book's formations, and its preparation then stopped after 90 s
    `frozen_portfolio.risk_history_insufficient`, unrecoverable, since each formation needs 63
    initialization and 504 factor-fit Risk return sessions before it. The plan judges that
    history by the same function its materialization reads and refuses before any Task, naming
    the first formation short of it, the sessions held and needed and the Risk start that would
    hold them, and offers the Risk study's controls on the strategy's own input."""

    from types import SimpleNamespace

    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceManifest,
        publish_research_workspace_manifest,
    )
    from alphalattice.interface.local_application.cli_contract import refusal_words
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    code = (
        "research_strategy.risk_history_insufficient:formation 2022-10-03 has 64 Risk return "
        "sessions before it, 567 needed, a Risk study starting on or before 2020-08-11"
    )

    def plan(_document: object) -> None:
        raise ValueError(code)

    publish_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("s"))
    host = PortfolioResearchOperations.__new__(PortfolioResearchOperations)
    host.research_strategies = SimpleNamespace(plan=plan)  # type: ignore[assignment]
    host.workspace_session = SimpleNamespace(workspace=tmp_path)  # type: ignore[assignment]
    binding = "b" * 64
    answer = host._workspace_operation(
        PortfolioResearchOperationRequest(
            operation="RESEARCH_STRATEGY_PLAN", experiment_document={"input_binding_hash": binding}
        ),
        caller="HUMAN",
    )
    assert answer is not None and answer["failure_code"] == code
    assert answer["next_requests"]["risk_controls"]["experiment_kind"] == (
        "risk.covariance-development"
    )
    assert answer["next_requests"]["risk_controls"]["input_binding_hash"] == binding
    words = refusal_words(code)
    assert "a Risk study starting on or before 2020-08-11" in words["detail"]
    assert words["next_action"] == "STUDY_RISK_FROM_THE_NEEDED_START"


def test_a_strategys_risk_history_is_judged_by_one_owner_for_its_plan_and_preparation():
    """requirement (V596's class, TE12): a plan admits only what its executor runs. The Risk
    history each formation needs (the installed decomposition's initialization and factor
    fit) is judged by one function, which both the strategy's plan and its materialization
    call; a formation holding one session fewer than the rule is short, one holding exactly
    the rule is not."""

    import inspect
    from datetime import date, timedelta

    from alphalattice.control.product_host.data_preparation.research_strategy import (
        ResearchStrategyPreparation,
    )
    from alphalattice.control.product_host.research_authoring import frozen_portfolio
    from alphalattice.investment.risk_research.surfaces.decomposition import (
        INSTALLED_RISK_DECOMPOSITION_RECIPE,
    )

    rule = (
        INSTALLED_RISK_DECOMPOSITION_RECIPE.conditional_volatility_initialization_sessions
        + INSTALLED_RISK_DECOMPOSITION_RECIPE.factor_fit_sessions
    )
    assert frozen_portfolio.RISK_HISTORY_SESSIONS == rule == 567
    sessions = [date(2018, 1, 1) + timedelta(days=day) for day in range(rule + 10)]
    short = frozen_portfolio.risk_history_shortfall(sessions, [sessions[rule - 1], sessions[-1]])
    assert short == (sessions[rule - 1], rule - 1)
    assert frozen_portfolio.risk_history_shortfall(sessions, [sessions[rule]]) is None
    for owner in (
        ResearchStrategyPreparation.plan,
        frozen_portfolio.prepare_frozen_portfolio_authority,
    ):
        assert "risk_history_shortfall(" in inspect.getsource(owner), owner


ALPHA, BETA = "ALPHA_BOOK", "BETA_BOOK"
"""Two strategies running forward on one workspace (V595)."""


def _research_update_plan(package: str, target: Any) -> Any:
    """A sealed research update plan of one strategy, each of its parts well formed (V595)."""
    from alphalattice.control.data_platform.maintenance.contracts import (
        MaintenanceTrigger,
        WorkspaceDataUpdateBinding,
        WorkspaceDataUpdatePlan,
        WorkspaceInputStatus,
        WorkspaceMaintenanceRequest,
    )
    from alphalattice.control.product_host.composition.decision_advancement import (
        DecisionAdvancementPlan,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceScoreInput,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    fixture = "a" * 64
    binding = WorkspaceDataUpdateBinding.seal(
        profile_file_hash=fixture, data_policy_hash=fixture, feature_catalog_hash=fixture
    )
    before = WorkspaceInputStatus.seal(
        manifest_revision="r1",
        data_revision_hash=fixture,
        data_through=target,
        adjusted_through=target,
        panel_hash=fixture,
        panel_through=target,
        readiness_status="READY",
        sources_checked_at=None,
        foundation_hash=None,
        foundation_panel_hash=None,
        foundation_disposition="MISSING",
    )
    request = WorkspaceMaintenanceRequest.create(
        market_profile_id=binding.market_profile_id,
        target_market_session=target,
        knowledge_cutoff_at=None,
        trigger=MaintenanceTrigger.USER_REQUEST,
        membership_revision="r1",
        data_policy_hash=fixture,
        feature_policy_hash=canonical_hash({"catalog": fixture, "invalidation": "domain-topology"}),
        requested_at=datetime(2026, 10, 1, tzinfo=UTC),
    )
    return DecisionAdvancementPlan.create(
        workspace_manifest_hash=fixture,
        catalog_hash=fixture,
        package_id=package,
        score_binding=ResearchWorkspaceScoreInput(
            strategy_package_id=package,
            strategy_package_hash=fixture,
            authority_relative_path="authority/fixture.json",
            authority_hash=fixture,
            source_kind="WORKSPACE_DATA_FEATURE",
        ),
        checkpoint_hash=fixture,
        parent_hash=None,
        target=target,
        decision_sessions=(target,),
        score_sessions=(target,),
        history_score_hashes=(),
        data_plan=WorkspaceDataUpdatePlan.seal(
            workspace_id="fixture",
            workspace_manifest_hash=fixture,
            binding=binding,
            before=before,
            request=request,
        ),
        implementation_hash=fixture,
        score_implementation_hash=fixture,
        calibration_implementation_hash=fixture,
        decision_implementation_hash=fixture,
    )


def _admitted(registry: Any, envelope: Any, goal: Any, plan: Any) -> str:
    return str(
        registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=datetime.now(UTC)
        ).record.task_id
    )


def test_a_cancelled_research_update_keeps_its_stop_and_offers_its_task_record(live, capsys):
    """OP4 regression: the V604 real replay left this cancelled read without words or a route.

    The same Task contract and writer-stopped cancellation code are reproduced through Task
    Control, without replaying scientific stages. The actual CLI follows the read-only route;
    neither read changes the cancelled record or offers a run or replan.
    """
    from alphalattice.control.product_host.composition.decision_advancement import task_contract
    from alphalattice.control.task_control.contracts import TaskExecutionCompatibility
    from alphalattice.interface.local_application.cli import main
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    registry = live.session.task_control_registry
    envelope, goal, workflow = task_contract(_research_update_plan(ALPHA, date(2026, 9, 29)))
    task_id = UUID(_admitted(registry, envelope, goal, workflow))
    started = registry.start_next(
        compatibility=TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash("CANCELLED_READBACK_FIXTURE"),
            workflow_definition_hash=workflow.workflow_definition_hash,
            input_schema_id=envelope.input_schema_id,
            domain_policy_hash=canonical_hash("CANCELLED_READBACK_FIXTURE"),
            framework_identity_hash=canonical_hash("CANCELLED_READBACK_FIXTURE"),
        ),
        worker_instance_id=uuid4(),
        observed_at=live.clock(),
        expected_task_id=task_id,
    )
    assert started is not None
    requested, _command = registry.request_cancel(
        task_id=task_id, expected_task_hash=started[0].record_hash, observed_at=live.clock()
    )
    # This fixture claimed no writer and executed no stage; the writer-stopped boundary is
    # acknowledged by Task Control, rather than manufacturing a cancelled Task or its code.
    cancelled = registry.finalize_cancel_after_writer_stopped(
        task_id=task_id, expected_task_hash=requested.record_hash, observed_at=live.clock()
    )
    assert live.operations.research_updates.readback(task_id) == {
        "status": "CANCELLED",
        "task_id": str(task_id),
        "failure_code": "TASK_CANCELLED_AFTER_WRITER_STOPPED",
    }
    saved = live.workspace.parent / "cancelled.json"
    assert (
        main(
            [
                "--workspace",
                str(live.workspace),
                "--view",
                "full",
                "research-update",
                "show",
                "--task",
                str(task_id),
                "--output",
                str(saved),
            ],
            serve=lambda _: 99,
        )
        == 2
    )
    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert answer["data"]["status"] == "CANCELLED"
    assert answer["failure_code"] == cancelled.failure_code
    assert "Task will not continue" in answer["detail"]
    assert answer["next_requests"] == {"task": {"operation": "STATUS", "task_id": str(task_id)}}
    assert (
        main(
            [
                "--workspace",
                str(live.workspace),
                "--view",
                "full",
                "request",
                "--from",
                str(saved),
                "--action",
                "task",
            ],
            serve=lambda _: 99,
        )
        == 2
    )
    followed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"]
    assert followed["task_id"] == str(task_id) and followed["lifecycle"] == "CANCELLED"
    assert followed["latest_failure_code"] == cancelled.failure_code
    assert registry.task(task_id).record_hash == cancelled.record_hash


def _planned_task(registry: Any, kind: str, plan: dict[str, object]) -> str:
    """A Task of `kind` whose sealed plan is `plan`: what a reader picking among Tasks reads."""
    from alphalattice.control.task_control.contracts import (
        ResearchGoal,
        ResearchPlan,
        TaskInputEnvelope,
        WorkItemDefinition,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    envelope = TaskInputEnvelope.create(
        task_kind=kind, input_schema_id=f"{kind}.fixture", payload={"plan": plan}
    )
    goal = ResearchGoal.create(
        goal_kind="FIXTURE",
        input_hash=envelope.input_hash,
        deliverable_kind="Fixture",
        summary="A Task its reader picks by the strategy its plan names.",
    )
    work = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash("fixture"),
        verifier_catalog_hash=canonical_hash("fixture"),
        work_items=(
            WorkItemDefinition.create(
                stage_id="only", dependency_ids=(), verifier_id="fixture.only"
            ),
        ),
    )
    return _admitted(registry, envelope, goal, work)


def test_completed_update_metadata_keeps_exact_target_claim_and_sealed_lookup(live, monkeypatch):
    """V683/EV/OP4: synthetic completion metadata is read without a scientific replay.

    This is a metadata-port fixture, not evidence that an advancement executed. The real
    Task contract, stage binding and publication seal readers remain in the path.
    """
    from alphalattice.control.product_host.composition.decision_advancement import (
        STAGES,
        task_contract,
    )
    from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        PortfolioUpdatePublication,
        advance_decision_state,
    )
    from tests.portfolio_strategy_lab.synthetic_numerical import (
        HASH,
        build_numerical,
        prepared_for,
        snapshot_for,
    )

    assert live.operations is not None
    operations = live.operations
    assert operations.research_updates is not None
    assert operations.activations is not None and operations.automation is not None
    owner = operations.research_updates
    numerical = build_numerical()
    package = numerical.checkpoint.package.strategy_id
    plan = _research_update_plan(package, numerical.sessions[numerical.first])
    publication = advance_decision_state(
        checkpoint=numerical.checkpoint,
        previous=None,
        prepared=prepared_for(numerical, numerical.first),
        observed=snapshot_for(numerical, numerical.first),
        plan_hash=HASH,
        published_at=datetime(2026, 10, 2, tzinfo=UTC),
    )
    assert isinstance(publication, PortfolioUpdatePublication)
    owner.ledger.content.publish_model(
        category="decision-candidates", value=publication, identity_field="content_hash"
    )
    # Seed only this isolated metadata receipt through its real sealed owner. No verifier is
    # replaced with a no-op, and this test makes no claim about completion of scientific work.
    owner._commit(plan, STAGES[5], (publication.content_hash,), HASH)
    envelope, goal, workflow = task_contract(plan)
    recorded_at = datetime(2026, 10, 2, tzinfo=UTC)
    task = TaskRecord.from_identity(
        task_id=uuid4(),
        task_kind=envelope.task_kind,
        input=envelope,
        goal=goal,
        plan=workflow,
        lifecycle=TaskLifecycle.SUCCEEDED,
        active_work_item_id=None,
        latest_execution_id=None,
        admitted_at=recorded_at,
        started_at=None,
        updated_at=recorded_at,
        failure_code=None,
        version=1,
    )
    monkeypatch.setattr(live.session.task_control_registry, "tasks", lambda: (task,))
    monkeypatch.setattr(operations.automation, "installed", (package,))

    def no_full_read(*_args, **_kwargs):
        raise AssertionError("Automation metadata opened full selected verification")

    monkeypatch.setattr(owner, "readback", no_full_read)
    monkeypatch.setattr(operations.activations, "state", no_full_read)
    monkeypatch.setattr(operations.activations, "read_review", no_full_read)
    answer = _json(live, "/api/research-update/automation")
    (running,) = answer["runs_forward"]
    latest = running["latest_update"]
    assert latest["task_id"] == str(task.task_id) and latest["lifecycle"] == "SUCCEEDED"
    assert latest["target_session"] == plan.target.isoformat()
    assert latest["claim"] == publication.claim
    assert latest["verification"] == "METADATA_ONLY_SELECTED_READBACK_VERIFIES_DESCENDANTS"
    assert latest["review_selector"] == {
        "update_task_id": str(task.task_id),
        "update_publication_hash": publication.content_hash,
        "position_basis": "CONDITIONAL_ESTIMATE",
    }
    assert latest["next_requests"]["readback"] == {
        "operation": "RESEARCH_UPDATE_READBACK",
        "task_id": str(task.task_id),
    }
    assert "review_standing" not in running
    # A second read must reopen the canonical publication: a broken seal is a named refusal,
    # never the earlier valid metadata returned from a cross-request cache.
    stored = owner.ledger.content.root / "decision-candidates" / f"{publication.content_hash}.json"
    payload = json.loads(stored.read_text(encoding="utf-8"))
    payload["published_at"] = "2026-10-03T00:00:00+00:00"
    stored.write_text(json.dumps(payload), encoding="utf-8")
    damaged = _json(live, "/api/research-update/automation")["runs_forward"][0]["latest_update"]
    assert damaged["status"] == "REFUSED" and damaged["failure_code"] and damaged["detail"]
    assert damaged["next_requests"]["task"] == {"operation": "STATUS", "task_id": str(task.task_id)}
    assert "review_selector" not in damaged and "claim" not in damaged


def test_a_strategys_latest_work_is_read_by_its_package_and_never_anothers(
    live, monkeypatch, capsys, tmp_path
):
    """requirement (V595, TE12): with two strategies running forward, a strategy's latest update
    is read by its package and never another strategy's; the unselected read answers as before
    while one strategy holds updates and is refused, offering each one's own, once two do; a Task
    named with another strategy is refused; and the daily update names each strategy's own latest
    update with its read bound to it. Through the real CLI and the automation route, and the same
    one reader for every kind of work planned per strategy: scores, calibration, Portfolio
    updates."""

    from datetime import date

    from alphalattice.control.product_host.composition.decision_advancement import task_contract
    from alphalattice.control.product_host.composition.portfolio_updates import (
        TASK_KIND as PORTFOLIO_UPDATE,
    )
    from alphalattice.control.product_host.composition.strategy_calibration import (
        TASK_KIND as CALIBRATION,
    )
    from alphalattice.control.product_host.composition.strategy_scoring import (
        TASK_KIND as SCORE,
    )
    from alphalattice.interface.local_application.cli import main

    registry = live.session.task_control_registry
    # A place for each Task this test admits, whatever the machine's processors (V100).
    write_queue_setting(
        live.workspace / "runtime", 16, chosen_by="HUMAN", chosen_at=datetime.now(UTC)
    )

    def update(package: str, target: date) -> str:
        return _admitted(registry, *task_contract(_research_update_plan(package, target)))

    def cli(*argv: str) -> tuple[int, dict[str, Any]]:
        code = main(
            ["--workspace", str(live.workspace), "--view", "full", *argv], serve=lambda _: 99
        )
        return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    first = update(ALPHA, date(2026, 9, 29))
    # One strategy's updates: the unselected read answers as before any strategy was named.
    _code, alone = cli("research-update", "show")
    assert alone["data"]["task_id"] == first, alone
    beta = update(BETA, date(2026, 9, 29))
    second = update(ALPHA, date(2026, 9, 30))
    code, refused = cli("research-update", "show")
    assert code == 2 and refused["failure_code"] == "research_update.strategy_package_required"
    assert refused["data"]["strategy_package_ids"] == [ALPHA, BETA]
    assert refused["data"]["next_requests"] == {
        f"readback:{package}": {
            "operation": "RESEARCH_UPDATE_READBACK",
            "strategy_package_id": package,
        }
        for package in (ALPHA, BETA)
    }
    for package, task in ((ALPHA, second), (BETA, beta)):
        _code, read = cli("research-update", "show", "--package", package)
        assert (read["data"]["task_id"], read["data"]["status"]) == (task, "QUEUED"), read
    code, crossed = cli("research-update", "show", "--task", beta, "--package", ALPHA)
    assert code == 2
    assert crossed["failure_code"] == f"research_update.task_of_another_strategy:{beta}"
    # The offered read, sent from the saved refusal as the CLI sends it.
    saved = tmp_path / "refused.json"
    assert cli("research-update", "show", "--output", str(saved))[0] == 2
    _code, followed = cli("request", "--from", str(saved), "--action", f"readback:{BETA}")
    assert followed["data"]["task_id"] == beta, followed

    # The daily update names each strategy's own latest update, its read bound to that Task.
    monkeypatch.setattr(live.operations.automation, "installed", (ALPHA, BETA))
    monkeypatch.setattr(live.operations.activations, "state", lambda _package: {"status": "ACTIVE"})
    rows = _json(live, "/api/research-update/automation")["runs_forward"]
    assert {row["strategy_package_id"]: row["latest_update"] for row in rows} == {
        package: {
            "task_id": task,
            "lifecycle": "QUEUED",
            "target_session": target,
            "next_requests": {
                "readback": {"operation": "RESEARCH_UPDATE_READBACK", "task_id": task}
            },
        }
        for package, task, target in ((ALPHA, second, "2026-09-30"), (BETA, beta, "2026-09-29"))
    }

    # Every kind of work planned per strategy is read the same way, by the one reader.
    for kind, command, owner, nested, none in (
        (PORTFOLIO_UPDATE, "portfolio-update", "portfolio_update", False, "NO_PORTFOLIO_UPDATE"),
        (SCORE, "score", "strategy_score", True, "NO_SCORE_PUBLICATION"),
        (CALIBRATION, "calibration", "strategy_calibration", True, "NO_CALIBRATION_PUBLICATION"),
    ):
        # No Task yet: the owner's own answer, its saved read keeping the strategy it names.
        _code, empty = cli(command, "show", "--package", ALPHA)
        assert empty["data"] == {
            "status": none,
            "task_id": None,
            "read_request": {
                "operation": f"{owner.upper()}_READBACK",
                "strategy_package_id": ALPHA,
            },
        }, empty
        for package in (ALPHA, BETA):
            named: dict[str, object] = {"strategy_package_id": package}
            _planned_task(registry, kind, {"binding": named} if nested else named)
        other = str(registry.tasks()[-1].task_id)
        code, refused = cli(command, "show")
        assert code == 2 and refused["failure_code"] == f"{owner}.strategy_package_required"
        assert sorted(refused["data"]["next_requests"]) == [f"readback:{ALPHA}", f"readback:{BETA}"]
        code, crossed = cli(command, "show", "--task", other, "--package", ALPHA)
        assert code == 2 and crossed["failure_code"] == f"{owner}.task_of_another_strategy:{other}"


def test_every_web_handler_preserves_every_typed_owner_exception(live, monkeypatch) -> None:
    """property (V679, TE12): source-discovered refusal classes cross every real registered
    handler via its public owner call. The original handler and exactly one reached owner
    call are checked, so an invalid request or replacement route cannot falsely pass.
    """
    from tests.portfolio_strategy_lab.typed_owner_refusals import typed_owner_cases
    from tests.portfolio_strategy_lab.web_refusal_support import exercise_registered_handler_matrix

    exercise_registered_handler_matrix(live, monkeypatch, typed_owner_cases())


@pytest.mark.parametrize("store", ("absent", "corrupt"))
def test_retention_decision_discovery_is_empty_only_before_a_store_exists(live, store) -> None:
    """regression: optional first-use discovery never hides an existing unreadable store."""
    import duckdb

    panel = live.operations.data_issues.panel
    assert not panel.path.exists()
    if store == "absent":
        assert panel.feature_input_raw_retention_decisions() == ()
        assert not panel.path.exists()
    else:
        panel.path.write_bytes(b"synthetic invalid database")
        with pytest.raises(duckdb.IOException):
            panel.feature_input_raw_retention_decisions()


def test_activation_keeps_typed_refusals_through_its_owner_and_observer(live, monkeypatch) -> None:
    """property (V679): real operations.execute and activation conversion remain in the
    path, with only the public activation owner's calls raising the discovered errors.
    """
    from tests.portfolio_strategy_lab.typed_owner_refusals import typed_owner_cases
    from tests.portfolio_strategy_lab.web_refusal_support import exercise_activation_owner_matrix

    exercise_activation_owner_matrix(live, monkeypatch, typed_owner_cases())


def test_a_raised_owner_refusal_keeps_its_transport_and_observer_code(live, monkeypatch) -> None:
    """regression (V679): observation cannot convert HTTP400 into HTTP200 or hide a
    typed RuntimeError in its FAILED record. Every discovered constructor crosses both.
    """
    from tests.portfolio_strategy_lab.typed_owner_refusals import typed_owner_cases
    from tests.portfolio_strategy_lab.web_refusal_support import exercise_observed_owner_matrix

    exercise_observed_owner_matrix(live, monkeypatch, typed_owner_cases())


@pytest.mark.untyped_failure
def test_a_genuine_fault_stays_a_fault_at_every_web_owner_entry(live, monkeypatch) -> None:
    """property (V679): each public handler dependency and the activation owner still
    returns HTTP500 with a stable incident for a genuine fault, withholding private text.
    """
    from tests.portfolio_strategy_lab.web_refusal_support import exercise_untyped_fault_matrix

    exercise_untyped_fault_matrix(live, monkeypatch)


def test_a_port_the_browser_refuses_is_never_the_workbenchs():
    """regression (STOPS-1, 2026-10-08): Windows gave a port-0 bind 1720, which Chromium refuses
    (ERR_UNSAFE_PORT), and the browser test could not open the Workbench. Each refused port is
    held while another is bound, then closed; a port the caller asked for is kept as asked."""

    class Bound:
        def __init__(self, port: int) -> None:
            self.server_address = ("127.0.0.1", port)
            self.closed = False

        def server_close(self) -> None:
            self.closed = True

    ports = iter((1720, 6000, 49152))
    made: list[Bound] = []

    def bind() -> Bound:
        made.append(Bound(next(ports)))
        return made[-1]

    kept = bind_browser_safe(bind, assigned=True)
    assert kept.server_address[1] == 49152 and not kept.closed
    assert [(server.server_address[1], server.closed) for server in made[:-1]] == [
        (1720, True),
        (6000, True),
    ]
    assert {1720, 5060, 6000, 10080} <= BROWSER_REFUSED_PORTS
    asked = bind_browser_safe(lambda: Bound(1720), assigned=False)
    assert asked.server_address[1] == 1720 and not asked.closed
