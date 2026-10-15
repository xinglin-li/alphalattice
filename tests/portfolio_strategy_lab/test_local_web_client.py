"""CLI and saved-document consumers of the Local Web."""

from __future__ import annotations

import dataclasses
import http.client
import http.cookiejar
import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from io import BytesIO, TextIOWrapper
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from tests.portfolio_strategy_lab.cli_support import _cli
from tests.portfolio_strategy_lab.local_web_support import (
    _json,
    _run_to_completion,
)


def test_a_controls_template_is_saved_as_its_declaration(tmp_path, monkeypatch, capsys):
    "`feature controls --declaration` refused the template its own answer"
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
            self.workspace, self.goal = (workspace, None)

        def exchange(self, _document):
            body = next(answers)
            return (body, json.dumps(body).encode("utf-8"))

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


def test_a_request_that_fills_an_object_in_part_is_a_template_naming_the_fields_left(tmp_path):
    "A request that fills an object in part is a template naming the fields left."
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


def test_an_operation_command_starts_from_a_saved_answer(tmp_path, monkeypatch, capsys):
    "`<noun> <verb> --from <answer>` (Z1b): the answer supplies what the command does not"
    from alphalattice.interface.local_application import cli, client

    sent = []

    class LocalClient:
        workspace = tmp_path
        goal = None

        def exchange(self, document):
            sent.append(document)
            return ({"status": "ADMITTED", "task_id": str(uuid4())}, b"{}")

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
    code = cli.main([*workspace, "curation", "submit", "--from", "-", "--file", "-"], serve=print)
    answer = json.loads(capsys.readouterr().out)
    assert code == 1 and answer["failure_code"] == "local_client.document_multiple_stdin_sources"
    assert len(sent) == 1


def test_declaration_validation_feedback_names_fields_without_echoing_values(
    read_only_live, monkeypatch
):
    from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
        PortfolioExperimentSpec,
    )

    def invalid_document(*args, **kwargs):
        return PortfolioExperimentSpec.model_validate(
            {"alpha_task_id": "known", "candidate_id": "known", "top_k": "private-input-marker"}
        )

    monkeypatch.setattr(read_only_live.operations.experiments, "operate", invalid_document)
    before = len(read_only_live.session.task_control_registry.tasks())
    reply = _json(
        read_only_live,
        "/api/experiments/plan",
        method="POST",
        payload={"experiment_yaml": "experiment: {}"},
    )
    assert reply["failure_code"] == "research_experiment.document_invalid"
    assert reply["fields"] == [["top_k"]]
    assert reply["next_action"] == "CORRECT_DECLARATION_AND_REPLAN"
    assert reply["next_requests"]["inputs"] == {"operation": "RESEARCH_INPUTS"}
    assert "private-input-marker" not in json.dumps(reply)
    assert len(read_only_live.session.task_control_registry.tasks()) == before


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
def test_external_client_refuses_spoofing_before_operation_dispatch(
    read_only_live, monkeypatch, fault
):
    from alphalattice.interface.local_application.client import LocalResearchClient

    tasks_before = read_only_live.session.task_control_registry.tasks()
    client = LocalResearchClient(read_only_live.workspace)
    if fault == "human_token":
        client.connection = dataclasses.replace(
            client.connection, token=read_only_live.web.application.session_token
        )
    elif fault == "workspace":
        client.workspace = read_only_live.workspace.parent / "another-workspace"
    monkeypatch.setattr(
        type(read_only_live.operations),
        "execute",
        lambda *a, **k: pytest.fail("unauthorized operation dispatched"),
    )
    request = {"operation": "TASKS"}
    if fault == "actor_payload":
        request["caller"] = "HUMAN"
    result = client.request(request)
    assert "refused" in result or result.get("status") == "REFUSED"
    assert read_only_live.session.task_control_registry.tasks() == tasks_before


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
        child.terminate()
        child.communicate(timeout=20)
    with LocalPortfolioWebSession.from_workspace(workspace) as reopened:
        assert LocalResearchConnection.read(workspace).instance != instance
        assert LocalResearchClient(workspace).request()["workspace_id"] == reopened.workspace_id
    assert not LocalResearchConnection.path(workspace).exists()


def test_failed_output_keeps_compact_feedback_and_the_completed_operation(
    tmp_path, monkeypatch, capsys
):
    from alphalattice.interface.local_application import cli, cli_contract, client

    task = str(uuid4())
    payload = {"status": "ADMITTED", "task_id": task, "html": "private detail " * 10000}
    calls = []

    class LocalClient:
        workspace = tmp_path
        goal = None

        def exchange(self, document):
            calls.append(document)
            return (payload, json.dumps(payload).encode())

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
        assert result["outcome"] == "PENDING" and result["status"] == "ADMITTED"
        assert code == cli_contract.EXIT_CODES["PENDING"]
        assert len(calls) == (1 if view == "compact" else 2)
        assert result["local_failure"]["failure_code"] == "local_client.output_write_refused"
        assert (
            result["local_failure"]["next_action"] == "USE_STDOUT_OR_A_WRITABLE_UNUSED_OUTPUT_PATH"
        )
        assert "output_file" not in result
        if view == "compact":
            assert result["representation"] == "COMPACT_DISPLAY_NOT_AN_AUTHORITY_DOCUMENT"
            assert result["data"]["task_id"] == task[:12]
            assert result["data"]["status"] == "ADMITTED"
            assert result["data"]["html"]["_omitted"] == "text"
            assert len(raw) < 2000 and "private detail" not in raw
        else:
            assert result["data"].pop("read_request")["operation"] == "EXPERIMENT_EXPORT"
            assert result["data"] == payload
    assert not (tmp_path / "unwritten.html").exists()


def test_a_wait_that_loses_its_host_keeps_the_admission_in_the_output(
    tmp_path, monkeypatch, capsys
):
    "requirement (, WK): an admitted RUN whose --wait cannot reach its Host waits it out;"
    from alphalattice.interface.local_application import cli, cli_contract, client

    task = str(uuid4())
    admitted = {"status": "ADMITTED", "task_id": task, "publication_task_id": task}

    class LocalClient:
        workspace = tmp_path
        goal = None

        def exchange(self, document):
            return (admitted, json.dumps(admitted).encode())

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


def test_a_chinese_request_is_sent_in_utf8_within_the_body_limit() -> None:
    "requirement (): two comments of 11,000 Chinese characters each fit the Host's body"
    from alphalattice.interface.local_application import cli_contract, client

    comment = "风险" * 5500
    document = {"operation": "EVIDENCE_REVIEW", "comments": [comment, comment]}
    body = client.request_body(document)
    assert len(body) < cli_contract.MAXIMUM_REQUEST_BODY_BYTES
    assert json.loads(body.decode("utf-8")) == document
    assert len(json.dumps(document).encode()) > cli_contract.MAXIMUM_REQUEST_BODY_BYTES


def test_continuation_references_preserve_owner_fields_and_reject_the_wrong_template(tmp_path):
    from alphalattice.interface.local_application.client import (
        _next_request,
        continuation,
        continued,
    )
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
    edits = {"experiment_yaml": edited.read_text(encoding="utf-8")}
    assert continuation("EXPERIMENT_PLAN", reply, edits, allowed) == {**template, **edits}
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
    controls = {
        "status": "READY",
        "research_input_id": "the-input",
        "input_binding_hash": "c" * 64,
        "template": {"experiment": {"kind": "factor.screening-development"}},
    }
    reply.write_text(json.dumps(controls), encoding="utf-8")
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
    with pytest.raises(ValueError, match="needs_choice:experiment_curation\\.choices,"):
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
    cut = {**template, "experiment_curation": {"_omitted": "collection", "item_count": 1}}
    compact = {
        "representation": "COMPACT_DISPLAY_NOT_AN_AUTHORITY_DOCUMENT",
        "omitted_sections": ["next_requests.curate.experiment_curation"],
        "data": {"task_id": task, "next_requests": {"curate": cut}},
    }
    response.write_text(json.dumps(compact), encoding="utf-8")
    with pytest.raises(ValueError, match="compact_reference_requires_full_response:curate"):
        _next_request(args)
    shown = json.loads(response.read_text(encoding="utf-8"))["data"]
    assert continued("EXPERIMENT_READBACK", shown, {}, frozenset({"task_id"}))["task_id"] == task
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
    book, risk = (tmp_path / "book.json", tmp_path / "risk.json")
    book.write_text(json.dumps({"task_id": task}), encoding="utf-8")
    risk_id = str(uuid4())
    risk.write_text(json.dumps({"task_id": risk_id}), encoding="utf-8")
    linked = continuation(
        "EXPERIMENT_LINK_RISK", book, {"risk_task_id": risk_id}, fields("EXPERIMENT_LINK_RISK")[1]
    )
    assert linked == {"operation": "EXPERIMENT_LINK_RISK", "task_id": task, "risk_task_id": risk_id}


def test_authored_text_is_never_a_reference_and_one_reference_has_two_spellings(tmp_path):
    "Authored text is never a reference and one reference has two spellings."
    from alphalattice.control.product_host.composition.reference_prefixes import ReferenceLedger
    from alphalattice.interface.local_application.client import (
        LocalResearchClientError,
        continuation,
        whole_references,
    )

    authored = "abcdefabcdef"
    first, second = (authored + "0" * 52, authored + "1" * 52)
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


def test_a_compact_answer_is_one_read_and_names_each_part_it_left_out() -> None:
    "A compact answer is one read and names each part it left out."
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
    assert len(json.dumps(body)) > 200000
    view = compact_display(body)
    assert len(json.dumps(view)) <= COMPACT_ANSWER_BYTES
    assert view["data"]["standing"] == body["standing"] and view["omitted_sections"]
    for part in view["omitted_sections"]:
        answer_part(body, part)
    assert len(json.dumps(compact_display(body, budget=8 * 1024))) <= 8 * 1024
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
    structured = {
        "status": "EXPERIMENT_PUBLISHED",
        "receipt": {f"entry-{i}": {f"trace-{j}": j for j in range(12)} for i in range(40)},
        "result": {"annualized_return": 0.1, "anchor_relative_return": None},
        "position": {"session": "2024-08-12", "cash": 0.0},
        "data_quality": {"policy": "quarantine_listings", "effective_listing_count": 457},
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


def test_cli_document_reader_keeps_utf8_and_byte_limits_for_files_and_stdin(
    tmp_path, monkeypatch, capsys, fake_host
):
    "The request file preserves UTF-8 and newlines and refuses invalid bytes or excess size."
    from alphalattice.interface.local_application.cli import main
    from alphalattice.interface.local_application.web import MAXIMUM_REQUEST_BODY_BYTES

    response = tmp_path / "answer.json"
    template = {
        "operation": "EXPERIMENT_PLAN",
        "research_input_id": "same-input",
        "input_binding_hash": "a" * 64,
    }
    response.write_text(json.dumps({"next_requests": {"preview": template}}), encoding="utf-8")
    document = tmp_path / "document.yaml"
    for payload, expected in (
        ("备注".encode(), "备注"),
        (b"a\r\nb\rc", "a\nb\nc"),
        (b"x" * (MAXIMUM_REQUEST_BODY_BYTES + 1), "too_large"),
        (b"\xff", "unreadable"),
    ):
        document.write_bytes(payload)
        for path in (document, Path("-")):
            monkeypatch.setattr(sys, "stdin", TextIOWrapper(BytesIO(payload), encoding="utf-8"))
            before = len(fake_host.sent)
            code = main(
                [
                    "--workspace",
                    str(tmp_path),
                    "request",
                    "--from",
                    str(response),
                    "--action",
                    "preview",
                    "--file",
                    str(path),
                ],
                serve=lambda _: 99,
            )
            reply = json.loads(capsys.readouterr().out)
            if expected not in {"too_large", "unreadable"}:
                assert code == 0, reply
                assert fake_host.sent[before:] == [{**template, "experiment_yaml": expected}]
            else:
                assert code == 1 and reply["failure_code"] == "local_client.document_" + expected, (
                    reply
                )
                assert len(fake_host.sent) == before
                if expected == "too_large":
                    assert reply["document_size"]["limit_bytes"] == MAXIMUM_REQUEST_BODY_BYTES


def test_declaration_comparison_preserves_absence_and_nested_parameter_changes():
    "The plan projection distinguishes absent fields from null and names nested changes."
    from alphalattice.control.product_host.composition.research_experiment_projection import (
        plan_impact,
    )

    before = {"portfolio": {"top_k": 35}, "old": None, "same": [1, 2]}
    after = {"portfolio": {"top_k": 40}, "new": None, "same": [1, 2]}
    common = {
        "program": SimpleNamespace(kind="factor.screening-development"),
        "binding": SimpleNamespace(input_id="same-input", binding_hash="a" * 64),
        "portfolio_source": None,
        "alpha_source": None,
        "model_training_source": None,
        "execution_preview": {},
        "origin_task_id": None,
        "implementation_hash": "b" * 64,
    }
    experiment = {"data_snapshot_handle": "snapshot", "sessions": {}}
    origin = SimpleNamespace(**common, document={"experiment": experiment, **before})
    plan = SimpleNamespace(**common, document={"experiment": experiment, **after})
    changes = plan_impact(plan, origin, None)["change"]["declared_fields"]
    assert [c["path"] for c in changes] == [["new"], ["old"], ["portfolio", "top_k"]]
    assert changes[0]["before_present"] is False and changes[0]["after_present"] is True
    assert changes[1]["before_present"] is True and changes[1]["after_present"] is False
    assert (changes[2]["before"], changes[2]["after"]) == (35, 40)


def test_cli_attaches_to_the_existing_host_and_reopens_exact_results(
    live, tmp_path, monkeypatch, capsys
):
    "CLI attachment reads the existing Host's public context and operation contracts."
    from alphalattice.interface.local_application.client import LocalResearchClient

    client = LocalResearchClient(live.workspace)
    view = client.request()
    assert view["caller"] == "EXTERNAL_AUTOMATION" and "session_token" not in view
    assert "research_context" not in view
    from alphalattice.interface.local_application.cli import main

    monkeypatch.setattr(sys, "stdin", TextIOWrapper(BytesIO(b"operation: TASKS"), encoding="utf-8"))
    assert (
        main(
            ["--workspace", str(live.workspace), "--view", "full", "request", "--file", "-"],
            serve=lambda _: 99,
        )
        == 0
    )
    tasks_output = capsys.readouterr().out
    tasks = json.loads(tasks_output)["data"]
    assert tasks.pop("read_request") == {"operation": "TASKS"}
    assert tasks == client.request({"operation": "TASKS"})
    monkeypatch.setattr(sys, "stdin", TextIOWrapper(BytesIO(b"{}"), encoding="utf-8"))
    code = main(
        ["--workspace", str(live.workspace), "request", "--choices", "-", "--from", "-"],
        serve=lambda _: 99,
    )
    conflict_output = capsys.readouterr().out
    conflict = json.loads(conflict_output)
    assert code == 1 and conflict["failure_code"] == "local_client.document_multiple_stdin_sources"
    _reply1 = _cli(live.workspace, "workspace", "show")
    workspace = _reply1[1]["data"]
    assert workspace["workspace_id"] == live.workspace_id
    context = workspace["research_context"]
    assert context == _json(live, "/api/session?context=1")["research_context"]
    assert {item["action"] for item in context["actions"]} == {
        "update_data",
        "capture_input",
        "research",
    }
    assert context["inputs"]["status"] == "REFUSED"
    assert context["tasks"]["tasks"] == []
    assert context["actions"][0]["configured"] is (
        live.operations.data_update is not None
        and live.operations.workspace_manifest.data_update is not None
    )
    _reply2 = _cli(live.workspace, "schema", "show", "EXPERIMENT_RUN")
    contract = _reply2[1]["data"]
    assert contract["required"] == ["experiment_plan_hash"]
    assert contract["properties"]["operation"] == {
        "type": "string",
        "const": "EXPERIMENT_RUN",
        "description": "This operation, by its name.",
    }
    assert contract["$defs"] == {}
    _reply3 = _cli(live.workspace, "schema", "show", "EXPERIMENT_DELIVERY_EXPORT")
    delivery_contract = _reply3[1]["data"]
    assert set(delivery_contract["$defs"]) == {"ResearchDeliveryCommentary"}
    _reply4 = _cli(live.workspace, "schema", "show", "EVIDENCE_ANALYSIS_SUBMIT")
    analyst_contract = _reply4[1]["data"]
    assert "AlternativeEvidenceAnswerFinding" in analyst_contract["$defs"]
    assert "EvidenceDirection" in analyst_contract["$defs"]
    assert "PortfolioReviewAnswer" not in analyst_contract["$defs"]
    for name, definition in analyst_contract["$defs"].items():
        assert definition == view["operation_schema"]["$defs"][name]
    malformed = client.request({"operation": "EXPERIMENT_RUN"})
    assert "experiment_plan_hash" in malformed["message"]
    transcript = "\n".join(
        [tasks_output, conflict_output, _reply1[2], _reply2[2], _reply3[2], _reply4[2]]
    )
    assert live.web.application.session_token not in transcript
    assert live.web.application.external_token not in transcript


def test_cli_saves_owner_refusals_and_rejects_removed_submit_commands(live, tmp_path):
    "Owner refusals are saved and a removed submit command admits no Task."
    refusal = tmp_path / "review.yaml"
    refusal.write_text("operation: CRO_REVIEW", encoding="utf-8")
    refusal_output = tmp_path / "refused-review.json"
    _reply1 = _cli(
        live.workspace, "request", "--file", str(refusal), "--output", str(refusal_output)
    )
    code, refused, _ = _reply1
    assert code == 2 and refused["data"]["disposition"].startswith("REFUSED")
    assert json.loads(refusal_output.read_text(encoding="utf-8")) == refused["data"]
    empty_assessment = tmp_path / "assessment.json"
    empty_assessment.write_text("{}", encoding="utf-8")
    before_review = len(live.session.task_control_registry.tasks())
    _reply2 = _cli(
        live.workspace, "review-submit", "--from", "x.json", "--answer", str(empty_assessment)
    )
    code, rejected, _ = _reply2
    assert code == 1 and rejected["failure_code"] == "local_client.usage_invalid"
    assert len(live.session.task_control_registry.tasks()) == before_review
    transcript = "\n".join([_reply1[2], _reply2[2]])
    assert live.web.application.session_token not in transcript
    assert live.web.application.external_token not in transcript


def test_cli_run_reads_and_saves_the_exact_completed_result(live, tmp_path):
    "A CLI run completes, discovers its result and saves the owner's exact export."
    from alphalattice.interface.local_application.client import LocalResearchClient

    client = LocalResearchClient(live.workspace)
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    _reply1 = _cli(live.workspace, "strategy-book", "preview", "--file", str(spec))
    assert _reply1[0] == 0
    _reply2 = _cli(live.workspace, "strategy-book", "run", "--file", str(spec))
    sent = _reply2[1]["data"]
    assert sent["disposition"] == "ADMITTED"
    _reply3 = _cli(live.workspace, "task", "show", sent["task_id"], "--wait", "--max-wait", "10")
    code, completed, _ = _reply3
    assert code == 0 and completed["data"]["lifecycle"] == "SUCCEEDED"
    result_hash = client.request({"operation": "RESULTS"})["results"][0]["result_hash"]
    _reply4 = _cli(live.workspace, "workspace", "show")
    discovered = _reply4[1]["data"]["research_context"]
    assert any(
        entry["entry_id"] == f"result:{result_hash}"
        for entry in discovered["recent_research"]["entries"]
    )
    _reply5 = _cli(live.workspace, "result", "show", result_hash)
    code, report, _ = _reply5
    assert code == 0
    assert "history=result%3A" + result_hash in report["local_web_url"]
    _reply6 = _cli(live.workspace, "--view", "compact", "result", "show", result_hash)
    code, brief, _ = _reply6
    assert code == 0 and brief["data"]["result_hash"] == report["data"]["result_hash"][:12]
    assert brief["navigation"]["kind"] == "portfolio_result"
    _reply7 = _cli(live.workspace, "result", "export", result_hash)
    exported = _reply7[1]["data"]
    assert exported.pop("read_request") == {"operation": "EXPORT", "result_hash": result_hash}
    assert exported == live.operations.export(result_hash)
    output = tmp_path / "export.json"
    _reply8 = _cli(live.workspace, "result", "export", result_hash, "--output", str(output))
    saved = _reply8
    whole = json.loads(output.read_text())
    assert saved[0] == 0 and whole.pop("read_request")["result_hash"] == result_hash
    assert whole == live.operations.export(result_hash)
    transcript = "\n".join(
        [
            _reply1[2],
            _reply2[2],
            _reply3[2],
            _reply4[2],
            _reply5[2],
            _reply6[2],
            _reply7[2],
            _reply8[2],
        ]
    )
    assert live.web.application.session_token not in transcript
    assert live.web.application.external_token not in transcript


def test_cli_reopens_exact_results_and_reuses_them_after_a_host_restart(live, tmp_path):
    "A restarted Host rejects its stale connection and reopens its exact result without admission."
    from alphalattice.interface.local_application.client import (
        LocalResearchClient,
        LocalResearchConnection,
    )

    client = LocalResearchClient(live.workspace)
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    result_hash = _run_to_completion(live)
    _reply1 = _cli(live.workspace, "result", "show", result_hash)
    code, report, _ = _reply1
    assert code == 0
    count = len(live.session.task_control_registry.tasks())
    old = client.connection
    live.stop()
    assert not LocalResearchConnection.path(live.workspace).exists()
    live.start()
    assert LocalResearchConnection.read(live.workspace).instance != old.instance
    client.connection = dataclasses.replace(old, url=live.url)
    assert "external_workspace_or_instance_mismatch" in str(client.request())
    _reply2 = _cli(live.workspace, "strategy-book", "run", "--file", str(spec))
    reused = _reply2[1]["data"]
    assert reused["disposition"] == "REUSED_EXACT" and reused["task_id"] is None
    assert len(live.session.task_control_registry.tasks()) == count
    _reply3 = _cli(live.workspace, "result", "show", result_hash)
    assert _reply3[1]["data"] == report["data"]
    transcript = "\n".join([_reply1[2], _reply2[2], _reply3[2]])
    assert live.web.application.session_token not in transcript
    assert live.web.application.external_token not in transcript


@pytest.mark.parametrize("fault", ("incomplete_body", "timeout_body", "timeout_open"))
def test_cli_reports_an_interrupted_response_body_without_retrying_work(
    read_only_live, monkeypatch, capsys, fault
):
    from alphalattice.interface.local_application.cli import main

    tasks_before = read_only_live.session.task_control_registry.tasks()

    class InterruptedResponse:
        status = 200

        def read(self):
            if fault == "timeout_body":
                raise TimeoutError("private transport detail must not be printed")
            raise http.client.IncompleteRead(b"", 100)

    timeouts, closed = ([], [])

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
        ["--workspace", str(read_only_live.workspace), *timeout_arguments, "task", "list"],
        serve=lambda _: 99,
    )
    body = json.loads(capsys.readouterr().out)
    assert code == 4 and closed == [True]
    assert body["failure_code"] == "local_client.connection_lost_task_may_still_run"
    assert body["transport_reason"] == (
        "HTTP_WAIT_EXPIRED" if fault.startswith("timeout") else "CONNECTION_INTERRUPTED"
    )
    assert body["request_timeout_seconds"] == 120
    assert body["operation_outcome"] == "UNKNOWN"
    assert "private transport detail" not in json.dumps(body)
    assert timeouts == [120.0]
    assert read_only_live.session.task_control_registry.tasks() == tasks_before


def test_cli_refuses_timeout_values_outside_the_public_range(tmp_path, capsys, fake_host):
    "Invalid request timeouts are refused before constructing a Host client or sending work."
    from alphalattice.interface.local_application.cli import main

    for timeout in ("0", "-1", "nan", "inf", "601"):
        assert (
            main(
                ["--workspace", str(tmp_path), "--request-timeout", timeout, "task", "list"],
                serve=lambda _: 99,
            )
            == 1
        )
        assert (
            json.loads(capsys.readouterr().out)["failure_code"]
            == "local_client.request_timeout_outside_0_600"
        )
    assert fake_host.workspaces == []
    assert fake_host.sent == []
