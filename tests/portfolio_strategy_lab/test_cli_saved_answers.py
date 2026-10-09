"""Saved answers retain their selection and lawful continuation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest

from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.interface.local_application import cli, client
from alphalattice.interface.local_application.answers import (
    answer_problem,
    continuation_problem,
    declaration_problem,
)
from alphalattice.interface.local_application.cli_contract import (
    client_refusal,
    command_table,
    declaration_contracts,
    envelope,
    exit_problem,
    named_read,
    outcome_of,
    request_problem,
    task_state,
)
from alphalattice.interface.local_application.client import LocalResearchClient
from alphalattice.interface.local_application.operations import OPERATIONS
from tests.portfolio_strategy_lab.cli_support import (
    _cli,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"


def test_every_exact_answer_link_preserves_its_selector_and_follow_scope() -> None:
    """Exact answer links preserve their object, revision, collection and follow scope."""

    client = object.__new__(LocalResearchClient)
    client.connection = SimpleNamespace(url="http://127.0.0.1:12345/")
    client.goal = None
    task, right, goal, explicit = (str(uuid4()) for _ in range(4))
    digest = "a" * 64
    cases = [
        (
            {"operation": "FEATURE_CATALOG_PLAN"},
            {"plan_hash": digest},
            {"feature_plan": [digest]},
            {},
        ),
        (
            {
                "operation": "EXPERIMENT_ALPHA_COMPARE",
                "left_task_id": task,
                "left_candidate_id": "ridge-1",
                "right_task_id": right,
                "right_candidate_id": "ridge-2",
            },
            {},
            {},
            {
                "page": ["alpha"],
                "study": [task],
                "alpha_left_task": [task],
                "alpha_left_candidate": ["ridge-1"],
                "alpha_right_task": [right],
                "alpha_right_candidate": ["ridge-2"],
            },
        ),
        (
            {"operation": "GOAL_SHOW"},
            {"goal_hash": digest},
            {},
            {"page": ["goal"], "goal": [digest]},
        ),
        (
            {"operation": "STRATEGY_ACTIVATE"},
            {"activation": {"book_task_id": task}},
            {},
            {"page": ["portfolio"], "book": [task]},
        ),
        (
            {"operation": "CRO_REVIEW"},
            {"review_publication_hash": digest},
            {"history": [f"review:{digest}"]},
            {},
        ),
        (
            {
                "operation": "CRO_READBACK",
                "experiment_task_id": task,
                "experiment_receipt_hash": digest,
                "portfolio_session": "2026-10-08",
            },
            {},
            {"review_experiment": [task], "r": [digest], "d": ["2026-10-08"]},
            {},
        ),
        (
            {
                "operation": "CRO_READBACK",
                "update_task_id": task,
                "update_publication_hash": digest,
                "position_basis": "CURRENT",
            },
            {},
            {"review_update": [task], "p": [digest], "b": ["CURRENT"]},
            {},
        ),
        (
            {"operation": "EXPERIMENT_READBACK"},
            {"task_id": task, "lifecycle": "SUCCEEDED"},
            {"history": [f"experiment:{task}"]},
            {},
        ),
        ({"operation": "REPORT", "result_hash": digest}, {}, {"history": [f"result:{digest}"]}, {}),
        ({"operation": "EXPERIMENT_PLAN"}, {"plan_hash": digest}, {"plan": [digest]}, {}),
        ({"operation": "DATA_UPDATE_READBACK"}, {}, {"panel": ["workspace"]}, {}),
    ]
    for document, body, query, fragment in cases:
        for attributed in (None, goal):
            answer = {**body, **({"attributed_goal_id": attributed} if attributed else {})}
            selected = client.selected_url(document, answer)
            split = urlsplit(selected)
            assert parse_qs(split.query) == query
            assert parse_qs(split.fragment) == {
                **fragment,
                "follow": [f"goal:{goal}" if attributed else "latest"],
            }
            assert client.navigation(document, answer)["url"] == selected
            if document["operation"] == "EXPERIMENT_ALPHA_COMPARE":
                assert client.navigation(document, answer)["kind"] == "alpha_comparison"
    client.goal = explicit
    assert parse_qs(urlsplit(client.selected_url({}, {})).fragment)["follow"] == [
        f"goal:{explicit}"
    ]
    assert parse_qs(urlsplit(client.selected_url({}, {"attributed_goal_id": goal})).fragment)[
        "follow"
    ] == [f"goal:{goal}"]
    client.goal = None
    assert parse_qs(
        urlsplit(client.selected_url({"operation": "GOAL_SHOW"}, {"goal_id": goal})).fragment
    )["follow"] == [f"goal:{goal}"]
    assert parse_qs(
        urlsplit(
            client.selected_url({"operation": "GOAL_NARRATIVE"}, {"goal": {"goal_id": goal}})
        ).fragment
    )["follow"] == [f"goal:{goal}"]
    for document in ({"operation": "REPORT"}, {"operation": "GOAL_NARRATIVE"}):
        answer = (
            {"goal": {"goal_id": goal}} if document["operation"] == "REPORT" else {"goal": goal}
        )
        assert parse_qs(urlsplit(client.selected_url(document, answer)).fragment)["follow"] == [
            "latest"
        ]

    before, current = "a" * 64, "b" * 64
    operations = [operation for operation in OPERATIONS if operation.startswith("GOAL_")]
    assert operations
    for operation in operations:
        document = {"operation": operation, "goal_hash": before}
        answer = {"goal_hash": current}
        selected = client.navigation(document, answer)
        assert selected["kind"] == "goal"
        assert parse_qs(urlsplit(selected["url"]).fragment) == {
            "page": ["goal"],
            "goal": [current],
            "follow": ["latest"],
        }
        assert parse_qs(urlsplit(client.selected_url(document, {})).fragment) == {
            "page": ["goal"],
            "goal": [before],
            "follow": ["latest"],
        }
        collection = client.navigation({"operation": operation}, {})
        assert collection["label"] == "Open Goals"
        assert parse_qs(urlsplit(collection["url"]).fragment) == {
            "page": ["goals"],
            "follow": ["latest"],
        }


def test_a_request_a_listed_item_offers_is_followed(tmp_path: Path, fake_host, capsys) -> None:
    """Saved offers preserve their selected subjects, choices and bound references."""
    task, other, alpha = (str(uuid4()), str(uuid4()), str(uuid4()))
    saved = tmp_path / "answer.json"
    acknowledge = {"operation": "UPGRADE_ACKNOWLEDGE", "upgrade_set_hash": "h"}
    decisions = {
        "status": "PENDING_DECISIONS",
        "decisions": [
            {
                "kind": "STOPPED_TASK",
                "task_id": value,
                "next_requests": {"recovery": {"operation": "TASK_RECOVERY", "task_id": value}},
            }
            for value in ("t-1", "t-2")
        ]
        + [{"kind": "UPGRADE", "next_requests": {"acknowledge": acknowledge}}],
    }
    saved.write_text(json.dumps({"data": decisions}), encoding="utf-8")
    assert sorted(client.offered_requests(decisions)) == [
        "acknowledge",
        "recovery:t-1",
        "recovery:t-2",
    ]
    chosen = argparse.Namespace(from_response=saved, action="recovery:t-2", file=None)
    assert client._next_request(chosen) == {"operation": "TASK_RECOVERY", "task_id": "t-2"}
    with pytest.raises(client.LocalResearchClientError, match="recovery:t-1,recovery:t-2"):
        client.continuation("TASK_RECOVERY", saved, {}, frozenset({"task_id"}))
    assert (
        client.continuation("UPGRADE_ACKNOWLEDGE", saved, {}, frozenset({"upgrade_set_hash"}))
        == acknowledge
    )
    assert client._next_commands(decisions, tmp_path) is None
    listed = client._next_commands(decisions, tmp_path, listed=True)
    assert listed is not None and sorted(listed) == ["acknowledge", "recovery:t-1", "recovery:t-2"]
    draft = {"operation": "EXPERIMENT_PORTFOLIO_DRAFT", "task_id": "t-1", "candidate_id": None}
    answer = {"status": "EXPERIMENT_PUBLISHED", "next_requests": {"portfolio-draft": draft}}
    assert client._next_commands(answer, tmp_path) is None
    templates = client._next_commands(answer, tmp_path, left=True)
    assert templates is not None
    assert templates["portfolio-draft"]["choose"] == ["candidate_id"]
    assert templates["portfolio-draft"]["command"].endswith(" --candidate <candidate_id>")
    saved.write_text(json.dumps(answer), encoding="utf-8")
    ask = argparse.Namespace(from_response=saved, action="portfolio-draft", file=None)
    with pytest.raises(client.LocalResearchClientError, match="needs_choice:candidate_id"):
        client._next_request(ask)
    choice = tmp_path / "choice.yaml"
    choice.write_text("candidate_id: c-7\n", encoding="utf-8")
    ask.file = choice
    assert client._next_request(ask) == {**draft, "candidate_id": "c-7"}
    left = client_refusal("local_client.next_request_needs_choice:candidate_id")
    assert left.next_action == "GIVE_THE_CHOICE_WITH_THE_SAME_ACTION"
    assert "--choices" in left.detail and "same action" in left.detail
    assert (
        client_refusal("local_client.next_request_required:a,b").next_action
        == "CHOOSE_ONE_OFFERED_NEXT_REQUEST"
    )
    curation = {"operation": "EXPERIMENT_CURATION", "task_id": task}
    flow = {
        "status": "OK",
        "intents": [
            {
                "research_input_id": "input-1",
                "flows": {
                    "factor": {"next_requests": {"curation": curation}},
                    "alpha": {"next_requests": {"controls": {"operation": "EXPERIMENT_CONTROLS"}}},
                    "risk": {"next_requests": {"controls": {"operation": "RESEARCH_INPUTS"}}},
                },
            }
        ],
        "admission": {"next_requests": {"curation": curation}},
    }
    offered = client.offered_requests(flow)
    assert offered["curation"] == curation
    assert {name for name in offered if name.startswith("controls")} == {
        "controls:intents.0.flows.alpha",
        "controls:intents.0.flows.risk",
    }
    saved.write_text(json.dumps(flow), encoding="utf-8")
    assert (
        client._next_request(argparse.Namespace(from_response=saved, action="curation", file=None))
        == curation
    )
    entries = [
        {
            "entry_id": f"review:{review}",
            "task_id": task,
            "next_requests": {
                "task": {"operation": "STATUS", "task_id": task},
                "export": {"operation": "EVIDENCE_CRO_EXPORT", "review_publication_hash": review},
            },
        }
        for review in ("a" * 64, "b" * 64)
    ]
    offered = client.offered_requests({"entries": entries})
    assert offered[f"export:review:{'a' * 64}"]["review_publication_hash"] == "a" * 64
    assert offered[f"export:review:{'b' * 64}"]["review_publication_hash"] == "b" * 64
    assert offered["task"] == {"operation": "STATUS", "task_id": task}
    alike = [
        {"task_id": task, "next_requests": {"export": {"operation": "EXPORT", "n": i}}}
        for i in (1, 2)
    ]
    kept = client.offered_requests({"items": alike})
    assert sorted(kept) == ["export:items.0", "export:items.1"], kept
    table = command_table()
    selector = {
        "experiment_task_id": task,
        "experiment_receipt_hash": "c" * 64,
        "portfolio_session": "2026-09-10",
    }
    review = {"operation": "EVIDENCE_CRO", **selector}
    book = {"status": "EXPERIMENT_PUBLISHED", "task_id": task, "next_requests": {"review": review}}
    links = {"operation": "EXPERIMENT_RISK_LINKS", "task_id": task}
    receipt = {"status": "RISK_REPORT_LINKED", "task_id": None, "next_requests": {"links": links}}
    export = {"operation": "EXPERIMENT_EXPORT", "task_id": task}
    named = {**receipt, "next_requests": {"export": export}}
    promotion = {"status": "UPSTREAM_PROMOTION_ADMITTED", "task_id": task, "follow_task_id": alpha}
    goal = str(uuid4())
    note = {"goal_statement": "A finding.", "change_reason": "Noted."}
    goal_answer, task_answer = (
        {"status": "OPEN", "goal_id": goal},
        {"status": "ADMITTED", "task_id": task},
    )
    replan = {
        "operation": "EXPERIMENT_PLAN",
        "research_input_id": "in-1",
        "input_binding_hash": "b",
        "experiment_document": {"experiment": {}},
        "origin_task_id": "t-1",
    }
    preview = {
        "status": "EXPIRED",
        "plan_hash": "p",
        "next_requests": {
            "replan": replan,
            "inspect": {"operation": "EXPERIMENT_PREVIEW_READBACK", "experiment_plan_hash": "p"},
        },
    }
    structured_replan = {
        **replan,
        "experiment_document": {"experiment": {"kind": "risk.covariance-development"}},
    }
    structured_replan.pop("origin_task_id")
    recovery = {
        "status": {"lifecycle": "BLOCKED", "task_id": "t-1"},
        "lifecycle": "BLOCKED",
        "next_requests": {"replan": structured_replan},
        "read_request": {"operation": "TASK_RECOVERY", "task_id": "t-1"},
    }
    score_replan = {
        "operation": "STRATEGY_SCORE_PLAN",
        "strategy_package_id": "PKG",
        "formation_session": "2026-09-29",
        "component_id": "G2",
    }
    score = {
        "status": "PLANNED",
        "score_plan_hash": "a" * 64,
        "strategy_package_id": "PKG",
        "formation_session": "2026-09-29",
        "next_requests": {"replan": score_replan},
    }
    plan_fields = frozenset(replan) - {"operation"} | {"experiment_yaml"}
    note_fields = frozenset({"goal_id", "goal_hash", "goal_statement", "change_reason"})
    cases = {
        ("EXPERIMENT_PLAN", plan_fields): [
            (preview, {}, replan),
            (
                preview,
                {"experiment_yaml": "experiment: {}\n"},
                {key: value for key, value in replan.items() if key != "experiment_document"}
                | {"experiment_yaml": "experiment: {}\n"},
            ),
        ],
        ("EXPERIMENT_PLAN", plan_fields - {"origin_task_id"}): [
            (recovery, {}, structured_replan),
            (recovery, {"research_input_id": "in-2"}, "bound_reference_override:research_input_id"),
            ({**recovery, "next_requests": {}}, {}, "plan_source_unsupported"),
        ],
        ("EVIDENCE_PREVIEW", frozenset(table["fields"]["EVIDENCE_PREVIEW"]["allowed"])): [
            (book, {}, {"operation": "EVIDENCE_PREVIEW", **selector}),
            (
                {
                    **book,
                    "next_requests": {
                        "review": review,
                        "other": {"operation": "EVIDENCE_CRO", "result_hash": "d" * 64},
                    },
                },
                {},
                "answer_names_two_books",
            ),
        ],
        ("EXPERIMENT_RISK_LINKS", frozenset(table["fields"]["EXPERIMENT_RISK_LINKS"]["allowed"])): [
            (receipt, {}, links),
            (receipt, {"task_id": task}, links),
            (named, {}, {"task_id": task}),
            (
                {
                    **named,
                    "next_requests": {"export": export, "other": {**export, "task_id": other}},
                },
                {},
                "response_reference_missing",
            ),
        ],
        ("STATUS", frozenset({"task_id"})): [
            (promotion, {}, {"operation": "STATUS", "task_id": alpha}),
            (
                {**promotion, "next_requests": {"task": {"operation": "STATUS", "task_id": alpha}}},
                {},
                {"task_id": alpha},
            ),
            (task_answer, {"task_id": other}, "bound_reference_override:task_id"),
        ],
        ("EXPERIMENT_READBACK", frozenset({"task_id"})): [(promotion, {}, {"task_id": task})],
        ("GOAL_NOTE", note_fields): [
            (goal_answer, note, {"goal_id": goal}),
            (task_answer, note, "response_reference_missing:goal_id"),
            (goal_answer, {**note, "goal_id": other}, "bound_reference_override:goal_id"),
        ],
        ("GOAL_OPEN", frozenset({"goal_id", "goal_declaration"})): [
            (
                goal_answer,
                {"goal_declaration": {}},
                {"operation": "GOAL_OPEN", "goal_declaration": {}},
            )
        ],
        (
            "STRATEGY_SCORE_PLAN",
            frozenset({"component_id", "formation_session", "strategy_package_id"}),
        ): [
            (score, {}, {"formation_session": "2026-09-29", "component_id": "G2"}),
            (score, {"formation_session": "2026-09-30"}, "bound_reference_override"),
        ],
    }
    for (operation, allowed), variants in cases.items():
        for answer, given, expected in variants:
            saved.write_text(json.dumps(answer), encoding="utf-8")
            for read in (
                lambda operation=operation, answer=answer, given=given, allowed=allowed: (
                    client.continued(operation, answer, given, allowed)
                ),
                lambda operation=operation, given=given, allowed=allowed: client.continuation(
                    operation, saved, given, allowed
                ),
            ):
                if isinstance(expected, str):
                    with pytest.raises(client.LocalResearchClientError, match=expected) as bound:
                        read()
                    if operation == "STRATEGY_SCORE_PLAN":
                        assert client_refusal(str(bound.value)).detail
                else:
                    got = read()
                    assert (
                        got if "operation" in expected else {key: got.get(key) for key in expected}
                    ) == expected, operation
    fake_host.answer = {"status": "ADMITTED", "task_id": str(uuid4())}
    sent = fake_host.sent
    run = {"operation": "DATA_UPDATE_RUN", "update_plan_hash": "c" * 64}
    saved.write_text(
        json.dumps({"status": "PLANNED", "plan_hash": "c" * 64, "next_requests": {"run": run}}),
        encoding="utf-8",
    )
    arguments = ["--workspace", str(tmp_path), "data-update", "run", "--from", str(saved)]
    assert cli.main(arguments, serve=lambda _: 99) == 3, capsys.readouterr().out
    assert sent == [run]
    fake_host.sent.clear()
    fake_host.answer = {"refused": "task_control.task_not_found"}
    saved.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "task_id": value,
                        "next_requests": {"recovery": {"operation": "STATUS", "task_id": value}},
                    }
                    for value in (task, other)
                ]
            }
        ),
        encoding="utf-8",
    )
    code = cli.main(
        [
            "--workspace",
            str(tmp_path),
            "request",
            "--from",
            str(saved),
            "--action",
            f"recovery:{task[:12]}",
        ],
        serve=lambda _: 99,
    )
    body = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (code, body["failure_code"]) == (2, "task_control.task_not_found"), body
    assert fake_host.sent == [{"operation": "STATUS", "task_id": task}]


def test_schema_show_writes_its_output_file(tmp_path: Path, capsys: Any) -> None:
    """Schema, model, restore and scaffold commands save complete success and refusal answers."""

    saved = tmp_path / "schema.json"
    code, answer, _ = _cli(tmp_path, "schema", "show", "STATUS", "--output", str(saved))
    assert code == 0 and answer["output_file"] == str(saved.resolve())
    assert json.loads(saved.read_text(encoding="utf-8"))["operation"] == "STATUS"

    passed, refused = tmp_path / "check.json", tmp_path / "refused.json"
    code, answer, _ = _cli(
        tmp_path, "model", "check", "regularized_linear", "--output", str(passed)
    )
    assert code == 0 and answer["output_file"] == str(passed.resolve())
    assert json.loads(passed.read_text(encoding="utf-8"))["status"] == "PASSED"
    code, answer, _ = _cli(tmp_path, "model", "check", "no_such_model", "--output", str(refused))
    assert code != 0 and answer["output_file"] == str(refused.resolve())
    assert json.loads(refused.read_text(encoding="utf-8"))["failure_code"] == (
        "model_extension.not_found:no_such_model"
    )

    workspace = ["--workspace", str(tmp_path / "workspace")]
    restored = {"status": "RESTORED", "workspace": str(tmp_path / "restored")}
    done = tmp_path / "restore.json"
    arguments = [*workspace, "backup", "restore", "--dir", str(tmp_path / "restored")]
    code = cli.main(
        [*arguments, "--output", str(done)], serve=lambda _: 99, restore=lambda *a, **k: restored
    )
    assert code == 0 and json.loads(done.read_text(encoding="utf-8")) == restored
    refused = tmp_path / "refused.json"
    assert cli.main([*arguments, "--output", str(refused)], serve=lambda _: 99) != 0
    assert json.loads(refused.read_text(encoding="utf-8"))["status"] == "REFUSED"
    capsys.readouterr()
    answer_file = tmp_path / "scaffold.json"
    declaration = tmp_path / "model.yaml"
    model = [*workspace, "model", "scaffold", "--save-declaration", str(declaration)]
    assert cli.main([*model, "--output", str(answer_file)], serve=lambda _: 99) == 0
    assert json.loads(answer_file.read_text(encoding="utf-8"))["status"] == "DECLARATION_WRITTEN"


def test_a_catalog_of_refused_controls_is_an_answer() -> None:
    """Catalog refusals, Task states and reused results have one outcome and a usable owner."""
    catalog = {
        "controls": [],
        "refused": [{"control_id": "turnover_cap", "refusal_code": "portfolio_control.frozen"}],
    }
    assert outcome_of(catalog) == "OK"
    answer = envelope(operation="CONTROLS", outcome="OK", body=catalog, elapsed_seconds=0.0)
    assert answer["failure_code"] is None
    refusal = {"refused": "task_control.task_not_found"}
    assert outcome_of(refusal) == "REFUSED"
    answer = envelope(operation="TASK", outcome="REFUSED", body=refusal, elapsed_seconds=0.0)
    assert answer["failure_code"] == "task_control.task_not_found"
    outcomes = {state: outcome_of({"lifecycle": state.value}) for state in TaskLifecycle}
    assert {state for state, outcome in outcomes.items() if outcome == "OK"} == {
        TaskLifecycle.SUCCEEDED
    }
    assert set(outcomes.values()) == {"OK", "PENDING", "REFUSED"}
    running = {"status": "NO_UPDATE_PUBLICATION", "task_lifecycle": "RUNNING"}
    assert task_state(running) == "RUNNING" and outcome_of(running) == "PENDING"
    bare = {"status": "REUSED_EXACT", "task_id": None, "receipt": {"plan_hash": "a" * 64}}
    problem = exit_problem(bare)
    assert problem is not None and "REUSED_EXACT" in problem
    assert exit_problem({**bare, "publication_task_id": str(uuid4())}) is None
    assert exit_problem({"status": "REUSED_EXACT", "result_hash": "b" * 64}) is None
    assert exit_problem({"status": "REUSED_EXACT", "observation_id": "o-1"}) is None
    sealed = {"operation": "EXPERIMENT_FOUNDATION_EXPORT", "foundation_admission_hash": "c" * 64}
    assert exit_problem({**bare, "next_requests": {"export": sealed}}) is None
    assert exit_problem({**bare, "attributed_goal_id": str(uuid4())}) is not None
    assert exit_problem({"status": "SUCCEEDED"}) is None


def test_an_answer_saved_as_json_or_yaml_reads_back_alike(tmp_path: Path) -> None:
    """Saved JSON and YAML preserve the whole answer while declarations have their own file."""

    texts = [
        "yes",
        "no",
        "on",
        "off",
        "true",
        "null",
        "~",
        "",
        " ",
        "1e3",
        "0x10",
        "007",
        "1_000",
        "2026-10-01",
        "2026-10-01T12:30:00+00:00",
        "12:30",
        ".inf",
        ".nan",
        "-0",
        "+1",
        "#hash",
        "a: b",
        "- item",
        "[x]",
        "{y}",
        "two\nlines",
        "trailing\n",
        " lead",
        "quote ' and \"",
        "back\\slash",
        "ü 中 🙂",
        "@at",
        "!tag",
        "&anchor",
        "*alias",
        "|",
        ">",
        "%",
        "x" * 300,
        "crlf\r\nline",
    ]
    answer = {
        "status": "OK",
        "texts": texts,
        "keys": {text: index for index, text in enumerate(texts) if text},
        "numbers": [0, -1, 1.5, 1e-7, 1e300, 2**63, 0.1 + 0.2, 7.0],
        "parts": [{"none": None, "empty": [], "map": {}}, [[1, [2, [3]]]], True, False],
        "next_requests": {"status": {"operation": "STATUS", "task_id": str(uuid4())}},
    }
    export = {"json": json.dumps(answer), "yaml": "status: OK\n", "html": "<p/>"}
    for name, body in (("answer", answer), ("export", export)):
        raw = json.dumps(body).encode("utf-8")
        client._save_output(tmp_path / f"{name}.json", body, raw, "json")
        client._save_output(tmp_path / f"{name}.yaml", body, raw, "yaml")
        as_json = client._response_document(tmp_path / f"{name}.json")
        as_yaml = client._response_document(tmp_path / f"{name}.yaml")
        assert as_yaml == as_json == json.loads(json.dumps(answer))

    answer = {
        "status": "DRAFT_READY",
        "plan_request": {"operation": "EXPERIMENT_PLAN", "origin_task_id": "t-1"},
        "document": {"experiment": {"kind": "factor.screening-development"}, "tolerance": "1e-10"},
        "yaml": "experiment:\n  kind: factor.screening-development\n",
    }
    saved = tmp_path / "draft.yaml"
    client._save_output(saved, answer, json.dumps(answer).encode("utf-8"), "yaml")
    assert client._response_document(saved) == answer
    declaration = tmp_path / "declaration.yaml"
    client._save_declaration(declaration, answer)
    assert declaration.read_text(encoding="utf-8") == answer["yaml"]
    with pytest.raises(client.LocalResearchClientError, match="declaration_unavailable"):
        client._save_declaration(tmp_path / "none.yaml", {"status": "OK"})


def test_an_offered_request_is_one_the_host_accepts_as_it_stands() -> None:
    """Offered requests, refusals and declarations obey every registered checker contract."""
    task = str(uuid4())
    assert request_problem({"operation": "STATUS", "task_id": task}) is None
    assert request_problem({"operation": "STATUS", "task_id": None}) is None
    assert request_problem({"operation": "STATUS"}) is None
    problem = request_problem({"evidence_unit_id": "u01"})
    assert problem is not None and "operation" in problem
    window = {"operation": "EVIDENCE_PACKET", "task_id": task, "evidence_expires_at": "2026-08-13"}
    assert request_problem(window) == "evidence_expires_at is no field of EVIDENCE_PACKET"
    stand_in = {"operation": "EXPERIMENT_PORTFOLIO_DRAFT", "candidate_id": "<candidate_id>"}
    assert request_problem(stand_in) == "candidate_id holds the stand-in <candidate_id>"
    numbered = {"operation": "STATUS", "task_id": 5}
    assert request_problem(numbered) == "task_id is integer, not string"
    table = command_table()
    samples = {
        "string": "s-1",
        "integer": 3,
        "number": 1.5,
        "boolean": True,
        "object": {},
        "array": [],
    }
    for operation, contract in table["fields"].items():
        request = {"operation": operation}
        for name in contract["allowed"]:
            kinds = table["types"].get(name, [])
            if kinds:
                request[name] = samples[kinds[0]]
        assert continuation_problem({"next_requests": {"next": request}}) is None, operation
        template = {"operation": operation, **dict.fromkeys(contract["required"])}
        assert continuation_problem({"next_requests": {"next": template}}) is None, operation
    table = command_table()
    read_operation = next(
        (
            operation
            for operation, fields in table["fields"].items()
            if fields["required"] == ["task_id"]
        )
    )
    body = {
        "status": "REFUSED",
        "failure_code": "local_client.response_reference_missing",
        "detail": "This saved answer names no subject for the next request.",
        "next_action": "SELECT_THE_NAMED_SUBJECT",
        "next_requests": {"read": {"operation": read_operation, "task_id": 42}},
    }
    for operation in (*table["fields"], "UNMODELED_PROJECTION"):
        problem = answer_problem(operation, body)
        assert problem is not None and "task_id" in problem, operation
    body["next_requests"]["read"]["task_id"] = str(uuid4())
    for operation in (*table["fields"], "UNMODELED_PROJECTION"):
        assert answer_problem(operation, body) is None, operation
    for operation, row in declaration_contracts().items():
        for field in row["fields"]:
            if field["name"] not in {"yaml", "template"}:
                continue
            declaration = "book: editable\n" if field["name"] == "yaml" else {"book": "editable"}
            assert declaration_problem(operation, {field["name"]: declaration}) is None
        assert declaration_problem(operation, {"yaml": "", "template": {}}) is not None
        assert (
            declaration_problem(operation, {"status": "REFUSED", "failure_code": "test.refused"})
            is None
        )
        for item in row.get("declaration_exceptions", ()):
            for value in item["values"]:
                assert declaration_problem(operation, {item["field"]: value}) is None
        assert declaration_problem(operation, {"status": "UNDECLARED_STATE"}) is not None


def test_each_registered_collection_read_checks_item_refusal_routes() -> None:
    """Partial collections keep healthy rows and worded routes for each refused row."""
    from alphalattice.interface.local_application.answers import (
        ANSWERS,
        collection_answer_fields,
    )
    from alphalattice.interface.local_application.cli_contract import refusal_problem, refusal_words

    operations = sorted(operation for operation in ANSWERS if collection_answer_fields(operation))
    assert operations
    global_refusal = {
        "status": "REFUSED",
        "failure_code": "goal.not_found",
        **refusal_words("goal.not_found"),
    }
    task = str(uuid4())
    for operation in operations:
        fields = collection_answer_fields(operation)
        assert (
            refusal_problem(global_refusal, operation=operation, collection_fields=fields) is None
        ), operation
        model = ANSWERS.get(operation)
        assert model is not None, operation
        assert fields, operation
        for field in fields:
            key = task
            refused_item = {
                "task_id": task,
                "status": "REFUSED",
                "failure_code": "goal.not_found",
                "detail": "This item could not be read.",
                "next_action": "REPLAN",
            }
            healthy_item = {"task_id": str(uuid4()), "status": "READY"}
            if operation == "ACTIVITY_LIST" and field == "tasks":
                body = {
                    "status": "PARTIAL",
                    field: {healthy_item["task_id"]: healthy_item, key: refused_item},
                }
            else:
                body = {"status": "PARTIAL", field: [healthy_item, refused_item]}
            problem = refusal_problem(body, operation=operation, collection_fields=fields)
            item_path = (
                f"{operation}.{field}[{key}]"
                if isinstance(body[field], dict)
                else f"{operation}.{field}[1]"
            )
            assert problem and item_path in problem, operation

            refused_item["next_requests"] = {"read-task": {"operation": "STATUS", "task_id": task}}
            assert refusal_problem(body, operation=operation, collection_fields=fields) is None, (
                operation
            )
            body["status"] = "REFUSED"
            problem = refusal_problem(body, operation=operation, collection_fields=fields)
            assert problem and "refused the collection while returning item rows" in problem

    task_fields = collection_answer_fields("TASKS")
    assert "tasks" in task_fields
    failed_work = {
        "task_id": task,
        "lifecycle": "BLOCKED",
        "latest_failure_code": "feature_trial.step_failed",
        "failure_code": "feature_trial.step_failed",
        "detail": "The Task's work stopped at a failed step; its record was read.",
    }
    readable_tasks = {"status": "TASKS", "tasks": [failed_work]}
    assert refusal_problem(readable_tasks, operation="TASKS", collection_fields=task_fields) is None


def test_every_read_reads_its_whole_selection_again_from_its_answer() -> None:
    """Every saved read preserves its explicit selection, defaults and absent first Task."""
    table = command_table()
    samples = {"string": "s-1", "integer": 3, "number": 1.5, "boolean": True}
    scalar_checked = defaults_checked = 0
    readbacks = []
    for operation in sorted(table["reads"]):
        fields = table["fields"][operation]
        allowed = frozenset(fields["allowed"])
        targets = {name: str(uuid4()) for name in ("task_id", "goal_id") if name in allowed}
        named = {
            name: (str(uuid4()) if name.endswith("_id") else "a" * 64)
            for name in sorted(set(fields["required"]) | ({"goal_id"} & allowed))
        }
        answer = {
            "status": "AVAILABLE",
            **named,
            "next_requests": {"next": {"operation": operation, **named, "next_page_marker": 2}},
        }
        saved = named_read({"operation": operation, **named}, answer)
        assert saved["read_request"] == {"operation": operation, **named}, operation
        again = client.continued(operation, saved, {}, allowed)
        assert "next_page_marker" not in again, (operation, again)
        assert {name: again.get(name) for name in named} == named, (operation, again)
        defaults_checked += 1
        if "task_id" in allowed and "task_id" not in fields["required"]:
            readbacks.append(operation)
            before = named_read({"operation": operation}, {"status": "NOT_STARTED"})
            assert client.continued(operation, before, {}, allowed) == {"operation": operation}
            task = str(uuid4())
            after = named_read({"operation": operation}, {"status": "SUCCEEDED", "task_id": task})
            assert client.continued(operation, after, {}, allowed) == {
                "operation": operation,
                "task_id": task,
            }
        for name in sorted(allowed - set(targets)):
            kinds = [kind for kind in table["types"].get(name, ()) if kind in samples]
            if not kinds:
                continue
            request = {"operation": operation, **targets, name: samples[kinds[0]]}
            echoed = {key: value for key, value in request.items() if key != "operation"}
            other = {"next_requests": {"next": {**request, name: samples[kinds[0]]}}}
            other["next_requests"]["next"].update(dict.fromkeys(targets, str(uuid4())))
            for body in ({"status": "OK"}, {"status": "OK", **echoed}, {"status": "OK", **other}):
                again = client.continued(operation, named_read(request, body), {}, allowed)
                assert {key: again.get(key) for key in request} == request, (operation, name)
            scalar_checked += 1
    assert scalar_checked > 100, scalar_checked
    assert defaults_checked == len(table["reads"]) and defaults_checked > 50
    assert "DATA_UPDATE_READBACK" in readbacks, readbacks
    status = frozenset(table["fields"]["STATUS"]["allowed"])
    bare = named_read({"operation": "STATUS"}, {"status": "AVAILABLE"})
    with pytest.raises(client.LocalResearchClientError, match="response_reference_missing"):
        client.continued("STATUS", bare, {}, status)
    assert named_read({"operation": "RUN", "spec": {}}, {"status": "OK"}) == {"status": "OK"}
    refusal = {"status": "REFUSED", "failure_code": "goal.not_found"}
    assert named_read({"operation": "REPORT", "result_hash": "a" * 64}, refusal) == refusal
    assert named_read({"operation": "REPORT"}, {"status": "OK"}) == {
        "status": "OK",
        "read_request": {"operation": "REPORT"},
    }
    goal, first, latest = str(uuid4()), "1" * 64, "2" * 64
    revision = named_read(
        {"operation": "GOAL_SHOW", "goal_id": goal, "goal_hash": first},
        {
            "status": "GOAL",
            "goal_id": goal,
            "goal_hash": first,
            "next_requests": {
                "show": {"operation": "GOAL_SHOW", "goal_id": goal},
                "prior_revision": {"operation": "GOAL_SHOW", "goal_hash": latest},
            },
        },
    )
    shown_again = client.continued(
        "GOAL_SHOW", revision, {}, frozenset(table["fields"]["GOAL_SHOW"]["allowed"])
    )
    assert (shown_again["goal_id"], shown_again["goal_hash"]) == (goal, first)


def test_an_unrelated_saved_answer_never_reads_default_controls() -> None:
    """A continuation without an input refuses before reading an anchor."""
    from alphalattice.interface.local_application.cli_contract import client_refusal, command_table
    from alphalattice.interface.local_application.client import LocalResearchClientError, continued

    allowed = frozenset(command_table()["fields"]["EXPERIMENT_CONTROLS"]["allowed"])
    with pytest.raises(
        LocalResearchClientError, match=r"local_client\.response_reference_missing"
    ) as caught:
        continued("EXPERIMENT_CONTROLS", {"status": "SUCCEEDED", "publication": {}}, {}, allowed)
    assert client_refusal(str(caught.value)).detail


def test_a_continued_page_keeps_the_ids_its_list_watches() -> None:
    """A continued page keeps the identities its list watches."""

    from alphalattice.interface.local_application.cli_contract import named_read
    from alphalattice.interface.local_application.client import continued

    task = str(uuid4())
    sent = {"operation": "ACTIVITY_LIST", "watch": [task], "limit": 1}
    page = named_read(sent, {"status": "ACTIVITY", "events": [], "next_cursor": "c-1"})
    assert page["read_request"] == {"operation": "ACTIVITY_LIST", "watch": [task], "limit": 1}
    following = continued(
        "ACTIVITY_LIST", page, {"after": "c-1"}, frozenset({"after", "limit", "watch"})
    )
    assert following["watch"] == [task] and following["limit"] == 1
    assert following["after"] == "c-1"


def test_a_saved_default_read_reads_its_own_page_again(
    tmp_path: Path, monkeypatch: Any, capsys: Any, fake_host: Any
) -> None:
    """A saved default read reads its own page again."""

    from alphalattice.interface.local_application import cli

    sent = fake_host.sent
    first = {
        "status": "AVAILABLE",
        "factors": [{"factor_id": "first-page"}],
        "next_requests": {"next": {"operation": "FEATURE_EXTENSIONS", "extensions_page": 2}},
    }

    fake_host.answer = lambda _: json.loads(json.dumps(first))
    saved = tmp_path / "p.json"
    listing = ["--workspace", str(tmp_path), "feature", "list"]
    assert cli.main([*listing, "--output", str(saved)], serve=lambda _: 99) == 0
    kept = json.loads(saved.read_text(encoding="utf-8"))
    assert kept["read_request"] == {"operation": "FEATURE_EXTENSIONS"}, kept
    capsys.readouterr()
    again = [*listing, "--from", str(saved), "--section", "factors.0.factor_id"]
    assert cli.main(again, serve=lambda _: 99) == 0
    assert "first-page" in capsys.readouterr().out
    assert sent == [{"operation": "FEATURE_EXTENSIONS"}] * 2, sent
    next_page = ["--workspace", str(tmp_path), "request", "--from", str(saved), "--action", "next"]
    assert cli.main(next_page, serve=lambda _: 99) == 0
    assert sent[-1] == {"operation": "FEATURE_EXTENSIONS", "extensions_page": 2}, sent


def test_a_read_alias_preserves_its_field_map_and_explicit_selection() -> None:
    """Read aliases preserve owner field maps and let an explicit selection replace a default."""
    table = command_table()
    plan, task = "a" * 64, str(uuid4())

    def allowed(operation: str) -> frozenset[str]:
        return frozenset(table["fields"][operation]["allowed"])

    report = named_read({"operation": "REPORT", "result_hash": plan}, {"status": "OK"})
    assert report["read_request"] == {"operation": "REPORT", "result_hash": plan}
    finding = named_read(
        {"operation": "CRO_REVIEW_FINDING", "finding_handle": "f-1", "result_hash": None},
        {"status": "FINDING"},
    )
    kept = {"operation": "CRO_REVIEW_FINDING", "finding_handle": "f-1"}
    assert finding == {"status": "FINDING", "read_request": kept}
    review = {"status": "FEATURE_REVIEW", "feature_plan_hash": plan, "factor_id": "f1"}
    again = client.continued("FEATURE_REVIEW", review, {}, allowed("FEATURE_REVIEW"))
    assert (again["feature_plan_hash"], again["feature_factor_id"]) == (plan, "f1")
    flagged = client.continued(
        "FEATURE_REVIEW", review, {"feature_factor_id": "f2"}, allowed("FEATURE_REVIEW")
    )
    assert flagged["feature_factor_id"] == "f2"
    packet = client.continued(
        "EVIDENCE_PACKET",
        {"status": "PACKET", "prepared_task_id": task, "prepared_unit_id": "u02"},
        {},
        allowed("EVIDENCE_PACKET"),
    )
    assert (packet["task_id"], packet["evidence_unit_id"]) == (task, "u02")
    assert (
        client.continued("CRO_REVIEW_FINDING", finding, {}, allowed("CRO_REVIEW_FINDING"))[
            "finding_handle"
        ]
        == "f-1"
    )
    day = {"operation": "EXPERIMENT_READBACK", "task_id": task, "portfolio_session": "2026-09-01"}
    shown = named_read(day, {"status": "EXPERIMENT_PUBLISHED", "task_id": task})
    assert client.continued("EXPERIMENT_READBACK", shown, {}, allowed("EXPERIMENT_READBACK")) == day
    moved = client.continued(
        "EXPERIMENT_READBACK",
        shown,
        {"portfolio_session": "2026-09-02"},
        allowed("EXPERIMENT_READBACK"),
    )
    assert moved["portfolio_session"] == "2026-09-02"
    package = {"operation": "CONTROLS", "strategy_package_id": "pkg-b"}
    controls = named_read(package, {"status": "CONTROLS", "strategy_package_id": "pkg-b"})
    assert client.continued("CONTROLS", controls, {}, allowed("CONTROLS")) == package


def test_a_short_reference_is_read_back_as_the_one_value_it_begins(live, tmp_path):
    """A short reference is read back as the one value it begins."""

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

    declaration = tmp_path / "goal.yaml"
    code, body, _ = _cli(live.workspace, "goal", "schema", "--save-declaration", str(declaration))
    assert code == 0, body
    opened = subprocess.run(
        [
            *(sys.executable, str(SCRIPT), "--workspace", str(live.workspace), "--view"),
            *("compact", "goal", "open", "--file", f"{declaration}"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    shown = json.loads(opened.stdout.strip().splitlines()[-1])
    assert opened.returncode == 0 and len(shown["data"]["goal_id"]) == 12, shown
    code, body, _ = _cli(live.workspace, "goal", "show", shown["data"]["goal_id"])
    assert code == 0 and body["data"]["goal_id"].startswith(shown["data"]["goal_id"]), body
    named, named_body, _ = _cli(live.workspace, "--goal", shown["data"]["goal_id"], "goal", "show")
    assert (named, named_body["data"]["goal_id"]) == (0, body["data"]["goal_id"]), named_body
    request = tmp_path / "request.yaml"
    goal_id = body["data"]["goal_id"]
    request.write_text(
        f"operation: GOAL_SHOW\ngoal_id: {goal_id}\ngoal_hash: '{'0' * 12}'\n", encoding="utf-8"
    )
    code, body, _ = _cli(live.workspace, "request", "--file", str(request))
    assert (code, body["failure_code"]) == (1, "local_client.short_reference_unknown"), body
    assert body["short_reference"]["field"] == "goal_hash" and "--from" in body["detail"]
