"""The CLI's one answer shape and its exit codes (binding plan, C1 rules 1 and 2).

Every command prints one envelope and exits by its outcome: 0 OK, 1 INVALID_INPUT, 2 REFUSED,
3 PENDING, 4 NO_HOST. `--output` holds the owner's answer whatever its status, and no answer
carries exception text or a path.
"""

from __future__ import annotations

import json
import re
import shlex
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.interface.local_application.failure_codes import FAILURE_DETAIL_WITHHELD
from tests.portfolio_strategy_lab.local_web_support import _json

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"
ENVELOPE = {
    "schema_version",
    "operation",
    "outcome",
    "status",
    "data",
    "failure_code",
    "detail",
    "next_requests",
    "timing",
}


def _cli(workspace: Path, *arguments: str) -> tuple[int, dict[str, Any], str]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--workspace", str(workspace), "--view", "full", *arguments],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode, json.loads(result.stdout.strip().splitlines()[-1]), result.stdout


def test_every_goal_navigation_uses_the_current_goal_routes() -> None:
    """P1/TE12: every registered Goal answer opens its exact revision or the Goals collection."""
    from types import SimpleNamespace
    from urllib.parse import parse_qs, urlsplit

    from alphalattice.interface.local_application.client import LocalResearchClient
    from alphalattice.interface.local_application.operations import OPERATIONS

    client = object.__new__(LocalResearchClient)
    client.connection = SimpleNamespace(url="http://127.0.0.1:12345")
    client.goal = None
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


def test_every_exact_answer_link_preserves_its_selector_and_follow_scope() -> None:
    """UIFOLLOW/TE12: early-return and query selectors retain their exact object and Goal."""
    from types import SimpleNamespace
    from urllib.parse import parse_qs, urlsplit

    from alphalattice.interface.local_application.client import LocalResearchClient

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
            {"operation": "EXPERIMENT_RUN"},
            {"task_id": task, "lifecycle": "RUNNING"},
            {"task": [task]},
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


@pytest.mark.parametrize(
    "network_consent",
    [
        pytest.param(None, id="defaults-without-consent"),
        pytest.param(False, id="bounds-without-consent"),
        pytest.param(True, id="consent-and-bounds"),
    ],
)
def test_serve_preserves_explicit_sec_consent_and_source_bounds(
    tmp_path: Path, network_consent: bool | None
) -> None:
    """The source-ways command reaches the launcher; bounds alone grant no consent."""
    from alphalattice.interface.local_application.cli import main

    launches: list[list[str]] = []
    source_arguments = (
        []
        if network_consent is None
        else [
            *(["--sec-network-consent"] if network_consent else []),
            "--sec-max-document-bytes",
            "2097152",
            "--sec-acquisition-window-seconds",
            "3600",
            "--sec-max-total-attempts",
            "300",
            "--sec-max-total-response-bytes",
            "100000000",
            "--sec-max-body-resources",
            "150",
            "--sec-campaign-id",
            "synthetic-cli-contract",
        ]
    )

    def launch(arguments: list[str]) -> int:
        launches.append(arguments)
        return 0

    assert (
        main(
            [
                "--workspace",
                str(tmp_path),
                "serve",
                "--no-browser",
                "--stop-on-stdin",
                *source_arguments,
            ],
            serve=launch,
        )
        == 0
    )
    assert launches == [
        [
            "--workspace",
            str(tmp_path),
            "--port",
            "0",
            "--no-browser",
            "--stop-on-stdin",
            *source_arguments,
        ]
    ]


def test_storage_cap_is_one_operator_setting_through_the_real_cli_and_http(
    live: LocalPortfolioWebSession,
) -> None:
    """V680: changing execution capacity preserves the workspace's sealed manifest and Tasks."""
    before = live.workspace_manifest.model_dump_json()
    tasks = _json(live, "/api/tasks")["tasks"]
    code, shown, _ = _cli(live.workspace, "storage", "cap")
    assert (code, shown["status"]) == (0, "AVAILABLE")
    assert shown["data"]["capacity"]["setting"]["cap_bytes"] == "auto"
    for cap in (13 * 1024**3, 20 * 1024**3):
        code, changed, _ = _cli(live.workspace, "storage", "set", "--cap-bytes", str(cap))
        assert (code, changed["status"]) == (0, "CONFIGURED")
        setting = changed["data"]["capacity"]["setting"]
        assert setting["cap_bytes"] == cap and setting["chosen_by"] == "EXTERNAL_AUTOMATION"
        assert _json(live, "/api/workspace/storage/cap")["capacity"]["cap_bytes"] == cap
    for invalid in ("0", "1.5", "unknown"):
        code, refusal, _ = _cli(live.workspace, "storage", "set", "--cap-bytes", invalid)
        assert (code, refusal["failure_code"]) == (2, "storage.cap_setting_invalid")
        assert "positive whole" in refusal["detail"]
        assert _json(live, "/api/workspace/storage/cap")["capacity"]["cap_bytes"] == 20 * 1024**3
    human = _json(
        live,
        "/api/workspace/storage/cap",
        method="POST",
        payload={"storage_cap_bytes": str(25 * 1024**3)},
    )
    assert human["capacity"]["setting"]["chosen_by"] == "HUMAN"
    code, shown, _ = _cli(live.workspace, "storage", "cap")
    assert code == 0 and shown["data"]["capacity"]["cap_bytes"] == 25 * 1024**3
    code, reset, _ = _cli(live.workspace, "storage", "set", "--cap-bytes", "auto")
    assert (code, reset["status"]) == (0, "CONFIGURED")
    assert reset["data"]["capacity"]["setting"]["cap_bytes"] == "auto"
    assert live.workspace_manifest.model_dump_json() == before
    assert _json(live, "/api/tasks")["tasks"] == tasks


def test_the_canary_door_operator_command_parses_through_the_real_cli(live) -> None:
    """V680 post-merge: the recovery words and owner use the CLI's actual grammar."""
    from alphalattice.control.product_host.composition.plain_refusals import explain
    from alphalattice.interface.local_application.cli_contract import refusal_words

    code = "alpha_modeling.lightgbm_thread_canary_mismatch"
    detail = refusal_words(code)["detail"]
    assert explain(code)["detail"] == detail
    command = re.findall(r"`([^`]+)`", detail)
    assert len(command) == 1
    exit_code, configured, _ = _cli(live.workspace, *shlex.split(command[0]))
    assert exit_code == 0 and configured["data"]["cpu_budget"] == 1


def test_every_answer_is_one_envelope_and_exits_by_its_outcome(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    code, body, _ = _cli(live.workspace, "task", "list")
    assert (code, body["outcome"]) == (0, "OK")
    assert body.keys() >= ENVELOPE and body["schema_version"] == 1
    assert body["operation"] == "TASKS" and isinstance(body["data"]["tasks"], list)

    saved = tmp_path / "refusal.json"
    code, body, stdout = _cli(live.workspace, "task", "show", str(uuid4()), "--output", str(saved))
    assert (code, body["outcome"], body["failure_code"]) == (
        2,
        "REFUSED",
        "task_control.task_not_found",
    )
    assert body["detail"] and body["next_requests"]
    # The refusal is the owner's answer and is saved like any other (CLI-6).
    assert json.loads(saved.read_text("utf-8"))["failure_code"] == "task_control.task_not_found"
    assert str(live.workspace) not in stdout

    code, body, _ = _cli(live.workspace, "frobnicate")
    assert (code, body["outcome"], body["failure_code"]) == (
        1,
        "INVALID_INPUT",
        "local_client.usage_invalid",
    )


def test_no_host_is_exit_four_in_its_own_words(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    empty = tmp_path / "no-host"
    (empty / "runtime").mkdir(parents=True)
    code, body, _ = _cli(empty, "task", "list")
    assert (code, body["outcome"], body["failure_code"]) == (
        4,
        "NO_HOST",
        "local_client.service_not_running",
    )

    # A record left behind by a Host that stopped: nothing listens on its port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    recorded = live.workspace / "runtime/local-research-connection.json"
    record = json.loads(recorded.read_text("utf-8"))
    record.update(workspace=str(empty.resolve()), url=f"http://127.0.0.1:{port}")
    (empty / "runtime/local-research-connection.json").write_text(json.dumps(record), "utf-8")
    code, body, _ = _cli(empty, "task", "list")
    assert (code, body["outcome"], body["failure_code"]) == (
        4,
        "NO_HOST",
        "local_client.service_not_running",
    )
    assert "Task may still run" not in json.dumps(body) and "No Host serves" in body["detail"]


def test_no_answer_carries_a_failure_field_that_is_not_a_code(
    live: LocalPortfolioWebSession, monkeypatch: Any
) -> None:
    operations = live.operations
    assert operations is not None
    monkeypatch.setattr(
        type(operations),
        "_execute",
        lambda _self, request, **kwargs: {
            "status": "REFUSED",
            "failure_code": "not a code: C:/secret/spec.yaml",
            "sections": [{"failure_code": "TASK_EXECUTION_INTERRUPTED"}],
        },
    )
    body = _json(live, "/api/plan", method="POST", payload={"spec": {}})
    assert body["failure_code"] == FAILURE_DETAIL_WITHHELD
    # An owner's own code, an enum name included, passes unchanged.
    assert body["sections"] == [{"failure_code": "TASK_EXECUTION_INTERRUPTED"}]
    assert "secret" not in json.dumps(body)


def test_lang_zh_words_an_answers_detail_and_keeps_its_codes(tmp_path: Path) -> None:
    """requirement (U19, CLI-17): `--lang zh` words an answer's detail from the UI owner's
    Chinese table; the code, the outcome and the next action stay English, for Agents."""

    empty = tmp_path / "no-host"
    (empty / "runtime").mkdir(parents=True)
    asked = ("--request-timeout", "700", "task", "list")
    _code, english, _ = _cli(empty, *asked)
    code, chinese, _ = _cli(empty, "--lang", "zh", *asked)
    assert (code, chinese["failure_code"]) == (1, "local_client.request_timeout_outside_0_600")
    kept = {"detail", "timing", "session_launches"}
    assert {k: v for k, v in chinese.items() if k not in kept} == {
        k: v for k, v in english.items() if k not in kept
    }
    assert english["detail"].startswith("--request-timeout")
    assert any("\u4e00" <= character <= "\u9fff" for character in chinese["detail"])


def test_every_answer_counts_the_agent_sessions_launches_help_included(tmp_path: Path) -> None:
    """requirement (V364): an agent reported 35 launches against a budget of 40 after 41,
    its help launches miscounted; every answer carries `session_launches`, the session's
    launches with help counted, and an answer outside an agent session carries none."""

    import os
    import tempfile

    session = f"v364-{uuid4()}"
    environment = {**os.environ, "CODEX_THREAD_ID": session, "CLAUDE_CODE_SESSION_ID": ""}
    empty = tmp_path / "no-host"
    (empty / "runtime").mkdir(parents=True)
    asked = ("--request-timeout", "700", "task", "list")

    def launch(*arguments: str, env: dict[str, str] = environment) -> dict[str, Any]:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--workspace", str(empty), *arguments],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        lines = result.stdout.strip().splitlines()
        return json.loads(lines[-1]) if lines and lines[-1].startswith("{") else {}

    try:
        assert launch(*asked)["session_launches"] == 1
        assert launch(*asked)["session_launches"] == 2
        launch("task", "list", "--help")
        assert launch(*asked)["session_launches"] == 4
        outside = {
            key: value
            for key, value in environment.items()
            if key not in {"CODEX_THREAD_ID", "CLAUDE_CODE_SESSION_ID"}
        }
        assert "session_launches" not in launch(*asked, env=outside)
    finally:
        count = (
            Path(tempfile.gettempdir()) / "alphalattice-cli-launches" / f"codex--{session}.count"
        )
        count.unlink(missing_ok=True)


def test_lang_zh_words_a_line_that_does_not_parse(capsys: pytest.CaptureFixture[str]) -> None:
    """regression (V582, an outside review at `ebe6e096`): `--lang zh` worded every answer but
    the parser's own refusal, since the line was parsed before its language was read. A line
    that does not parse reads in the language it names, wherever on the line it names it; a
    language the parser refuses reads in English."""

    from alphalattice.interface.local_application.cli import main

    def refused(*line: str) -> dict[str, Any]:
        with pytest.raises(SystemExit):
            main(list(line), serve=lambda _arguments: 0)
        return dict(json.loads(capsys.readouterr().out.strip().splitlines()[-1]))

    def chinese(text: str) -> bool:
        return any("一" <= character <= "鿿" for character in text)

    for line in (
        ("--lang", "zh", "--workspace", "w", "task", "show", "--typo", "x"),
        ("--workspace", "w", "task", "show", "x", "--typo", "--lang", "zh"),
    ):
        body = refused(*line)
        assert body["failure_code"] == "local_client.usage_invalid", line
        assert chinese(body["detail"]), (line, body["detail"])
    english = refused("--lang", "fr", "--workspace", "w", "task", "show")
    assert english["failure_code"] == "local_client.usage_invalid"
    assert not chinese(english["detail"])


def test_one_reader_names_the_session_a_request_and_its_launches_count_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression (V583, an outside review at `ebe6e096`): with both hosts' session variables
    set, an agent started inside another, the request named no session but its launches
    counted to Claude's. One reader, `agent_session`, decides both: such a call names no
    session and counts to none, and each host's own session counts its own. It is the only
    code that reads the hosts' session variables for the session; the Codex queue reads its
    thread to wake, by the one variable naming it."""

    import ast
    import tempfile

    from alphalattice.interface.local_application import cli_contract

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    claude = {"CLAUDE_CODE_SESSION_ID": "claude-1"}
    codex = {"CODEX_THREAD_ID": "codex-1"}
    both = {**claude, **codex}
    assert cli_contract.agent_session(both) is None
    assert cli_contract.agent_provenance_headers(both) == {}
    launches = [cli_contract.count_launch(env) for env in (claude, both, claude, codex)]
    assert launches == [1, None, 2, 1]
    assert cli_contract.agent_session(claude) == ("claude-code", "claude-1")
    assert cli_contract.agent_session({**codex, "CLAUDE_CODE_SESSION_ID": " "}) == (
        "codex",
        "codex-1",
    )
    variable = re.compile(r"environ\.get\(['\"](CODEX_THREAD_ID|CLAUDE_CODE_SESSION_ID)")
    words = ("AGENT_SESSION_VARIABLES", "CODEX_THREAD_ID", "CLAUDE_CODE_SESSION_ID")
    readers: dict[str, set[str]] = {}
    for path in (SCRIPT.parents[1] / "src" / "alphalattice").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if not any(word in text for word in words):
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef):
                body = ast.unparse(node)
                if "AGENT_SESSION_VARIABLES" in body or variable.search(body):
                    readers.setdefault(path.name, set()).add(node.name)
    assert readers == {
        "cli_contract.py": {"agent_session", "request_provenance"},
        "client.py": {"_codex_queue_ready"},
    }, readers


def test_a_printed_command_runs_as_printed_in_both_shells(tmp_path: Path) -> None:
    """regression (V123, V131, V449): a next command names the checkout's entry and the
    workspace, and quotes every part for the shell that reads it: PowerShell single-quotes a path
    with a space, doubles an apostrophe, quotes a leading `@` and calls a quoted program with
    `&`; POSIX quotes by `shlex`. The CLI reads the line back as the request it came from, a
    text beginning `@` included, which it would read as a file unless doubled (V449: this test
    asserted `'@file'` and never read the line back)."""

    from alphalattice.interface.local_application.cli import request_of
    from alphalattice.interface.local_application.cli_contract import command, entry

    workspace = tmp_path / "a book's workspace"
    request = {"operation": "ACTIVITY_LIST", "after": "@file"}
    prefix = entry(workspace)
    assert prefix == (
        "alphalattice",
        "--workspace",
        str(workspace.resolve()),
    )
    posix = command(request, prefix=prefix, quoting="posix")
    assert posix is not None and shlex.split(posix)[: len(prefix)] == list(prefix)
    assert request_of(shlex.split(posix)[1:]) == request
    powershell = command(
        request, prefix=("C:/My Tools/python.exe", *prefix[1:]), quoting="powershell"
    )
    assert powershell is not None and powershell.startswith("& 'C:/My Tools/python.exe' ")
    assert "'" + str(workspace.resolve()).replace("'", "''") + "'" in powershell
    assert powershell.endswith("--after '@@file'")


def test_a_value_is_read_one_way_only(tmp_path: Path) -> None:
    """regression (V126, V127, V132, V149): a yes-or-no field takes true or false alone; `@@`
    is a literal `@`; `@file` reads a YAML list for a list field; a YAML key written twice is
    refused at its line and column."""

    import pytest
    import yaml

    from alphalattice.interface.local_application import cli, client
    from alphalattice.protocols.research_authoring.selection import load_safe_yaml_document

    assert cli._value("network_enabled", "true") is True
    assert cli._value("network_enabled", "False") is False
    with pytest.raises(client.LocalResearchClientError, match="boolean_invalid"):
        cli._value("network_enabled", "flase")
    assert cli._value("agent_role", "@@home") == "@home"
    listed = tmp_path / "items.yaml"
    listed.write_text("- a\n- b\n", encoding="utf-8")
    assert cli._value("automation_package_ids", "@" + str(listed)) == ["a", "b"]
    with pytest.raises(yaml.constructor.ConstructorError, match="twice"):
        load_safe_yaml_document("a: 1\nb: 2\na: 3\n")


def test_an_empty_document_file_is_refused_by_its_own_code(tmp_path: Path) -> None:
    """regression (V222): an answer file left empty (PowerShell's `$input` wrote nothing) was
    refused as a document of the wrong shape; it is refused as empty, naming the file and the
    way on."""

    import pytest

    from alphalattice.interface.local_application import client
    from alphalattice.interface.local_application.cli_contract import client_refusal

    empty = tmp_path / "answer.json"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(client.LocalResearchClientError, match="document_empty") as refused:
        client._document(empty)
    assert refused.value.document_location == {
        "file": "answer.json",
        "expected": "a safe YAML mapping",
    }
    assert client_refusal("local_client.document_empty").next_action == (
        "WRITE_THE_DOCUMENT_INTO_THE_FILE"
    )


def test_every_refusal_the_client_raises_has_its_own_words() -> None:
    """regression (V482): six codes the client raises read the generic words.

    `local_client.answer_names_two_books` (V473) and five more told the agent to inspect the same
    Task, a way on that does not apply, and the compact refusal's entry lacked the colon its code
    carries. Every code spelled where the client's error is raised, a subject after a colon
    included, has an entry of its own.
    """
    import ast
    import itertools

    from alphalattice.interface.local_application import cli_contract

    def spellings(node: ast.expr) -> set[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.JoinedStr):
            head = itertools.takewhile(lambda part: isinstance(part, ast.Constant), node.values)
            return {"".join(str(part.value) for part in head) + "subject"}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return {a + b for a in spellings(node.left) for b in spellings(node.right)}
        if isinstance(node, ast.IfExp):
            return spellings(node.body) | spellings(node.orelse)
        return {"subject"}

    codes = {
        code
        for path in Path(cli_contract.__file__).parents[2].rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == "LocalResearchClientError"
        and node.args
        for code in spellings(node.args[0])
    }
    generic = cli_contract.client_refusal("no_owner.unworded")
    assert "local_client.answer_names_two_books" in codes
    assert not sorted(c for c in codes if cli_contract.client_refusal(c) == generic)


def test_every_answer_that_names_a_book_for_review_offers_that_review() -> None:
    """regression (V483, V487; V473's class): an update's positions and a recorded choice of
    analysis named their book's review fields but offered no request bound to them, so an
    agent's next read fell to the workspace's default book. Every answer whose rows name a
    `review_selector` offers `next_requests` beside it."""

    from alphalattice.interface.local_application import answers

    rows = json.loads(Path(answers.__file__).with_name("answers.json").read_text("utf-8"))
    naming = {
        operation
        for operation, answer in rows.items()
        if any(field["name"] == "review_selector" for field in answer["fields"])
    }
    assert {"EVIDENCE_SELECT", "RESEARCH_UPDATE_READBACK", "PORTFOLIO_UPDATE_READBACK"} <= naming
    assert not sorted(
        operation
        for operation in naming
        if not any(field["name"] == "next_requests" for field in rows[operation]["fields"])
    )


def test_a_document_over_the_request_bound_is_refused_with_its_size_and_the_way_to_fit(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """regression (V372): AX12's agent sent a 137,986-byte goal submission and was told the
    document could not be read and to check its schema; the refusal names the size, the bound
    and a way to fit (cite exact reads by their requests)."""

    from alphalattice.interface.local_application.cli_contract import MAXIMUM_REQUEST_BODY_BYTES

    large = tmp_path / "completion.json"
    large.write_text(json.dumps({"summary": "x" * MAXIMUM_REQUEST_BODY_BYTES}), encoding="utf-8")
    code, body, _ = _cli(
        live.workspace,
        "goal",
        "submit",
        str(uuid4()),
        "--revision",
        "a" * 64,
        "--file",
        f"{large}",
    )
    assert (code, body["failure_code"]) == (1, "local_client.document_too_large"), body
    assert body["document_size"] == {
        "bytes": large.stat().st_size,
        "limit_bytes": MAXIMUM_REQUEST_BODY_BYTES,
    }
    assert body["next_action"] == "SEND_A_SMALLER_DOCUMENT"
    assert "references" in body["detail"] and "cannot be read" not in body["detail"]


def test_a_document_path_with_no_file_is_named_apart_from_an_unreadable_document(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """regression (V390): AX13's Portfolio agent sent `request --file` with paths it had moved to
    a copy; each was refused `document_unreadable` with words about the schema, as if the file
    were bad JSON. A path with no file is named as missing, apart from a file that cannot be
    read."""

    missing = tmp_path / "copies" / "01-controls.json"
    code, body, _ = _cli(live.workspace, "request", "--file", str(missing))
    assert (code, body["failure_code"]) == (1, "local_client.document_missing"), body
    assert body["document_location"] == {"file": str(missing), "expected": "an existing file"}
    assert body["next_action"] == "CHECK_THE_FILE_PATH" and "schema" not in body["detail"]
    broken = tmp_path / "broken.json"
    broken.write_bytes(b"\xff\xfe not utf-8")
    code, body, _ = _cli(live.workspace, "request", "--file", str(broken))
    assert (code, body["failure_code"]) == (1, "local_client.document_unreadable"), body
    assert body["document_location"]["file"] == str(broken)


def test_a_nouns_help_names_every_verbs_flags() -> None:
    """requirement (V377): AX12's agent read a noun's help, then each verb's, to learn the
    flags (thirteen of its launches were help); one noun's help names every verb's operation,
    its required flags and, in brackets, its optional ones, and the options every verb takes."""

    import os

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "feature", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "COLUMNS": "240"},
    )
    text = " ".join(result.stdout.split())
    assert "controls Shows the formula language" in text and "input. --binding [--plan]" in text
    assert (
        "review Shows a formula factor's review packet" in text
        and "activation. <id> --plan" in text
    )
    assert "activate Activates a reviewed formula factor" in text
    assert "(a person completes it) <id> --plan" in text
    assert "--from (a saved answer)" in text


def test_an_unknown_operation_is_answered_with_the_nearest_names(tmp_path: Path) -> None:
    """regression (V111): `schema show` names the registry's nearest operations, where the
    parser answered only READ_HELP."""

    code, answer, _stderr = _cli(tmp_path, "schema", "show", "EXPERIMENT_COMPAR")
    assert code == 1 and answer["failure_code"] == "local_client.operation_unknown"
    assert "EXPERIMENT_COMPARE" in answer["nearest_operations"]


def test_a_command_a_text_names_is_one_the_cli_has(tmp_path: Path) -> None:
    """regression (V124): refusal words named `capabilities --operation` after that command had
    gone; the gate's walk refuses a text naming a command, or a flag, the CLI does not have."""

    from devtools.architecture.operation_registry import named_commands

    text = tmp_path / "case-study" / "README.md"
    text.parent.mkdir()
    text.write_text(
        "Run `schema show STATUS`, `study controls` and `request --from x`;"
        " not `capabilities --operation STATUS`, `study frobnicate` or"
        " `study show --sectoin run`. Prose such as `uv sync --locked` is no command.\n",
        encoding="utf-8",
    )
    found = named_commands(tmp_path)
    assert [line.split(" names ")[1] for line in found] == [
        "`capabilities`, a command the CLI does not have",
        "`study frobnicate`, a command the CLI does not have",
        "`study show --sectoin`, a flag the CLI does not have",
    ]


def _assert_commands_parse(
    commands: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Use request_of's own parser, stopping before file reads or product operations."""
    import argparse

    from alphalattice.interface.local_application.cli import request_of

    class Parsed(Exception):
        pass

    original_parse = argparse.ArgumentParser.parse_args

    def parse_only(parser, args=None, namespace=None):
        original_parse(parser, args, namespace)
        raise Parsed

    failures = []
    with monkeypatch.context() as context:
        context.setattr(argparse.ArgumentParser, "parse_args", parse_only)
        for location, command in commands:
            try:
                request_of(shlex.split(command.removeprefix("alphalattice "), comments=True))
            except Parsed:
                pass
            except SystemExit as error:
                if error.code != 0:
                    failures.append((location, command, capsys.readouterr().err))
            except ValueError as error:
                failures.append((location, command, str(error)))
            capsys.readouterr()
    assert commands and failures == []


def test_every_documented_command_line_parses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """regression (V611, TE12): documented alphalattice command lines parse through its
    own parser, including fenced/inline examples and generated capability bullets."""
    repository = SCRIPT.parents[1]
    roots = [repository / "AGENTS.md", repository / "README.md"]
    roots.extend(
        repository / name
        for name in (
            ".agents/skills/alphalattice-research",
            ".claude/skills/alphalattice-research",
            ".codex/agents",
            ".claude/agents",
            "docs/public-source",
        )
    )
    paths = []
    for root in roots:
        assert root.exists(), root
        files = (
            sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else [root]
        )
        assert files, root
        paths.extend(files)
    grammar = {
        "alphalattice <object> <action>",
        "alphalattice <noun> <verb>",
        "alphalattice <command> [flags]",
        "alphalattice ...",
        "uv run alphalattice ...",
    }
    fences = re.compile(r"^\s*```[^\n]*\n(.*?)^\s*```[ \t]*$", re.M | re.S)

    def examples(text: str) -> Iterator[tuple[int, str]]:
        lines = text.splitlines()
        index = 0
        while index < len(lines):
            number, line = index + 1, lines[index].strip()
            if line.startswith(("alphalattice ", "& $al $cli ")):
                while line.endswith(("\\", "`")) and index + 1 < len(lines):
                    index += 1
                    line = line[:-1] + " " + lines[index].strip()
                yield number, line
            index += 1
        prose = fences.sub(lambda block: re.sub(r"[^\n]", " ", block.group()), text)
        for match in re.finditer(r"`([^`]+)`", prose):
            command = " ".join(match.group(1).split())
            prefix = prose[prose.rfind("\n", 0, match.start()) + 1 : match.start()]
            if command.startswith(("alphalattice ", "uv run alphalattice ")) or (
                re.fullmatch(r"\s*-\s*", prefix)
                and len(command.split()) > 1
                and not command.startswith("--")
            ):
                yield prose.count("\n", 0, match.start()) + 1, command

    dummy_id = "00000000-0000-4000-8000-000000000001"
    values = dict.fromkeys(
        [
            "alpha-task-id",
            "alpha_task",
            "analyst-bundle-name",
            "binding",
            "book",
            "book-task-id",
            "candidate-id",
            "candidate_id",
            "declared-model-id",
            "dir",
            "evidence-task-id",
            "existing-workspace-path",
            "explicit-workspace",
            "factor-task-id",
            "factor_task",
            "feature_factor_id",
            "feature_plan_hash",
            "input",
            "input-id",
            "model",
            "out",
            "package",
            "prepared-task",
            "preview_request",
            "returned-action-name",
            "risk-kind",
            "risk-task-id",
            "risk_task",
            "task",
            "task-id",
            "trial",
            "unit",
        ],
        dummy_id,
    )
    values.update(
        {
            name: "documentation-path"
            for name in ("dir", "out", "existing-workspace-path", "explicit-workspace")
        }
    )
    values.update(
        {
            "binding": "1" * 64,
            "feature_plan_hash": "1" * 64,
            "risk-kind": "risk-covariance-development",
            "unit": "u01",
        }
    )
    variables = {
        "ws": "documentation-workspace",
        "baselineTask": dummy_id,
        "factor": "formula_factor",
        "featurePlan": "1" * 64,
        "inputBinding": "1" * 64,
    }

    commands, covered = [], set()
    for path in paths:
        for number, command in examples(path.read_text(encoding="utf-8")):
            if command in grammar:
                continue  # Explanatory grammar notation is not an invocation.
            covered.add(path)
            location = f"{path.relative_to(repository)}:{number}: {command}"
            normalized = command.removeprefix("& $al $cli ")
            names = set(re.findall(r"<([^<>]+)>", normalized))
            assert names <= values.keys(), (location, names - values.keys())
            normalized = re.sub(r"<([^<>]+)>", lambda match: values[match[1]], normalized)
            names = set(re.findall(r"\$([A-Za-z_]\w*)", normalized))
            assert names <= variables.keys(), (location, names - variables.keys())
            normalized = re.sub(r"\$([A-Za-z_]\w*)", lambda match: variables[match[1]], normalized)
            normalized = normalized.removeprefix("uv run ").removeprefix("alphalattice ")
            commands.append((location, normalized))
    assert all(any(root == path or root in path.parents for path in covered) for root in roots)
    _assert_commands_parse(commands, monkeypatch, capsys)


def test_every_refusal_and_answer_template_command_parses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """V687/TE12: owner recovery words and generated answer offers use the real parser."""
    from alphalattice.control.product_host.research_authoring.model_extensions import (
        ModelExtensions,
    )
    from alphalattice.interface.local_application import cli, cli_contract, client

    table = cli_contract.command_table()
    names = (
        set(table["commands"])
        | {" ".join(pair) for pair in cli.CLIENT_COMMANDS}
        | {"request", "serve"}
    )
    nouns = {name.split()[0] for name in names}
    heads = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    multi = "|".join(
        re.escape(name) for name in sorted(names, key=len, reverse=True) if " " in name
    )
    # This finds mentions; request_of alone decides whether their syntax is legal.
    raw = re.compile(
        rf"(?<![\w.-])(?:alphalattice\s+(?P<full>{heads})|(?P<plain>{multi}))(?![\w.-])"
    )
    tokens = re.compile(r"<[^<>]+>|'(?:[^']|'')*'|\"[^\"]*\"|[^\s,;()]+")
    commands: list[tuple[str, str]] = []

    def filled(text: str) -> str:
        def value(match: re.Match[str]) -> str:
            name = match[1]
            field = name.split(":", 1)[0]
            dummy_id = "00000000-0000-4000-8000-000000000001"
            special = {
                "operation": "STATUS",
                "command": "STATUS",
                "bytes": "21474836480",
                "id": dummy_id,
                "task-id": dummy_id,
                "task": dummy_id,
                "component": "G2_R0_TREND",
                "its readback": dummy_id,
                "workspace id": dummy_id,
                "package": dummy_id,
                "verified generation hash": "1" * 64,
                "file": "fixture.yaml",
                "that answer": "answer.json",
                "new directory": "fixture-directory",
                "backup root": "fixture-backups",
                "dir": "fixture-directory",
                "model-id": "fixture_model",
                "output": "answer.json",
                "declaration": "fixture.yaml",
                "study.yaml": "fixture.yaml",
                "spec.json": "fixture.json",
                "hash": "1" * 64,
                "strategy-package-id": "fixture_package",
                "unit": "u01",
            }
            if field in special:
                return shlex.quote(special[field])
            assert field in table["types"], (text, field)
            kinds = table["types"].get(field, [])
            if "object" in kinds:
                return shlex.quote("{}")
            if "array" in kinds:
                return shlex.quote("[]")
            if "boolean" in kinds:
                return "true"
            if "integer" in kinds or "number" in kinds:
                return "2"
            return shlex.quote(
                "1" * 64
                if "hash" in field or "binding" in field
                else "auto"
                if field == "storage_cap_bytes"
                else "fixture_model"
                if field == "model_id"
                else "2026-09-14"
                if field == "portfolio_session"
                else "CRO"
                if field == "agent_role"
                else dummy_id
            )

        return re.sub(r"<([^<>]+)>", value, text).removeprefix("alphalattice ")

    def offered(location: str, text: str) -> None:
        if text.startswith("alphalattice "):
            commands.append((location, filled(text)))
            return
        for match in re.finditer(r"`([^`]+)`", text):
            line = match[1]
            parts = line.split()
            if parts[0] == "alphalattice" or (
                parts[0] in nouns and (len(parts) > 1 or parts[0] in {"request", "serve"})
            ):
                commands.append((location, filled(line)))
        prose = re.sub(r"`[^`]+`", lambda match: " " * len(match[0]), text)
        # A pointer to an actual next_commands producer is not a bare invocation.
        prose = prose.replace("model check command shown here", "")
        for match in raw.finditer(prose):
            head = match["full"] or match["plain"]
            parser = cli._parser(cli._named(head.split()))
            terminal = next(
                part for part in cli._commands_of(parser) if part.prog == f"alphalattice {head}"
            )
            options = {
                option: action for action in terminal._actions for option in action.option_strings
            }
            tail = tokens.findall(prose[match.end() :])
            parts, index = head.split(), 0
            while index < len(tail):
                token = tail[index]
                if token.startswith("--"):
                    parts.append(token)
                    action = options.get(token.split("=", 1)[0])
                    index += 1
                    if (
                        "=" not in token
                        and (action is None or action.nargs != 0)
                        and index < len(tail)
                        and not tail[index].startswith("--")
                        and tail[index]
                        not in {"and", "or", "then", "with", "to", "which", "before", "after"}
                    ):
                        parts.append(tail[index])
                        index += 1
                elif token.startswith(("<", "'", '"')) or re.fullmatch(
                    r"[A-Z][A-Z0-9_]*|[0-9a-f-]{32,64}", token
                ):
                    parts.append(token)
                    index += 1
                else:
                    break
            commands.append((location, filled(" ".join(parts))))

    def walk(location: str, value: Any) -> None:
        if isinstance(value, str):
            offered(location, value)
        elif isinstance(value, dict):
            for name, inner in value.items():
                walk(f"{location}.{name}", inner)
        elif isinstance(value, list):
            for index, inner in enumerate(value):
                walk(f"{location}[{index}]", inner)

    root = SCRIPT.parents[1] / "src/alphalattice/interface/local_application"
    for name in ("refusal_words.json", "answers.json"):
        walk(name, json.loads((root / name).read_text(encoding="utf-8")))
    for code, refusal in cli_contract._CLIENT_REFUSALS:
        offered(f"client_refusal.{code}", refusal.detail)
    operations = sorted({op for rows in table["commands"].values() for op in rows})
    requests: dict[str, Any] = {}
    for operation in operations:
        walk(f"schema.{operation}", cli.schema(operation))
        request = {"operation": operation}
        line = cli_contract.command(request, quoting="posix")
        assert line is not None, operation
        commands.append((f"template.{operation}", filled(line)))
        requests[operation] = request
    for name in table["commands"]:
        walk(f"command_schema.{name}", cli.command_schema(name))
    for operation in operations:
        for field in table["inner_required"]:
            if field not in table["fields"][operation]["allowed"]:
                continue
            request = {
                "operation": operation,
                field: {"expected_receipt_hash": "1" * 64},
            }
            requests[f"{operation}.{field}.partial"] = request
            line = cli_contract.command(request, quoting="posix")
            assert line is not None
            commands.append((f"partial.{operation}", filled(line)))
    monkeypatch.setattr(client, "entry", lambda _workspace, _options: ("alphalattice",))
    for listed in (False, True):
        for left in (False, True):
            body = {"next_requests": requests}
            walk(
                f"next_commands.listed={listed}.left={left}",
                client._next_commands(body, tmp_path, listed=listed, left=left),
            )
    walk(
        "model.review_refusal",
        ModelExtensions._review_refusal("dummy_model", ValueError("unreadable")),
    )
    assert len(commands) > len(operations) * 3
    _assert_commands_parse(commands, monkeypatch, capsys)


def test_a_request_a_listed_item_offers_is_followed(tmp_path: Path) -> None:
    """regression (V135): the pending decisions carry each decision's next requests inside
    `decisions[]`, where `--list-next`, `--from --action` and `--from` read only the top level;
    two decisions offering one name are told apart by their Tasks."""

    import argparse

    import pytest

    from alphalattice.interface.local_application import client

    def stopped(task: str) -> dict[str, Any]:
        recovery = {"operation": "TASK_RECOVERY", "task_id": task}
        return {"kind": "STOPPED_TASK", "task_id": task, "next_requests": {"recovery": recovery}}

    acknowledge = {"operation": "UPGRADE_ACKNOWLEDGE", "upgrade_set_hash": "h"}
    answer = {
        "status": "PENDING_DECISIONS",
        "decisions": [
            stopped("t-1"),
            stopped("t-2"),
            {"kind": "UPGRADE", "next_requests": {"acknowledge": acknowledge}},
        ],
    }
    saved = tmp_path / "decisions.json"
    saved.write_text(json.dumps({"data": answer}), encoding="utf-8")
    assert sorted(client.offered_requests(answer)) == [
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
    # The envelope's own list stays the top level's; `--list-next` lists them all.
    assert client._next_commands(answer, tmp_path) is None
    listed = client._next_commands(answer, tmp_path, listed=True)
    assert listed is not None and sorted(listed) == ["acknowledge", "recovery:t-1", "recovery:t-2"]


def test_a_request_that_leaves_a_choice_is_shown_as_a_template(tmp_path: Path) -> None:
    """regression (V136): an Alpha result's `portfolio-draft` carries the Task alone and
    `candidate_id` is required, yet it was printed as a runnable command. It is a template
    naming the choice, following it asks for the choice, and the gate refuses an offer that
    neither fills nor names a required field."""

    import argparse

    import pytest

    from alphalattice.interface.local_application import client
    from devtools.architecture.operation_registry import incomplete_offers

    draft = {"operation": "EXPERIMENT_PORTFOLIO_DRAFT", "task_id": "t-1", "candidate_id": None}
    answer = {"status": "EXPERIMENT_PUBLISHED", "next_requests": {"portfolio-draft": draft}}
    assert client._next_commands(answer, tmp_path) is None
    templates = client._next_commands(answer, tmp_path, left=True)
    assert templates is not None
    assert templates["portfolio-draft"]["choose"] == ["candidate_id"]
    assert templates["portfolio-draft"]["command"].endswith(" --candidate <candidate_id>")
    saved = tmp_path / "alpha.json"
    saved.write_text(json.dumps(answer), encoding="utf-8")
    ask = argparse.Namespace(from_response=saved, action="portfolio-draft", file=None)
    with pytest.raises(client.LocalResearchClientError, match="needs_choice:candidate_id"):
        client._next_request(ask)
    choice = tmp_path / "choice.yaml"
    choice.write_text("candidate_id: c-7\n", encoding="utf-8")
    ask.file = choice
    assert client._next_request(ask) == {**draft, "candidate_id": "c-7"}

    owner = tmp_path / "src" / "alphalattice" / "owner.py"
    owner.parent.mkdir(parents=True)
    owner.write_text(
        'OFFER = {"operation": "EXPERIMENT_PORTFOLIO_DRAFT", "task_id": "t-1"}\n', encoding="utf-8"
    )
    (line,) = incomplete_offers(tmp_path)
    assert "without its required candidate_id" in line


def test_the_plan_command_follows_a_previews_replan(tmp_path: Path) -> None:
    """regression (V148): `experiment plan --from preview.json` refused the `replan` an expired
    preview offers, which only `request --from --action replan` followed; a declaration given
    replaces the offered one, as it replaces a draft's."""

    from alphalattice.interface.local_application import client

    replan = {
        "operation": "EXPERIMENT_PLAN",
        "research_input_id": "in-1",
        "input_binding_hash": "b",
        "experiment_document": {"experiment": {}},
        "origin_task_id": "t-1",
    }
    inspect = {"operation": "EXPERIMENT_PREVIEW_READBACK", "experiment_plan_hash": "p"}
    preview = {
        "status": "EXPIRED",
        "plan_hash": "p",
        "next_requests": {"replan": replan, "inspect": inspect},
    }
    saved = tmp_path / "preview.json"
    saved.write_text(json.dumps(preview), encoding="utf-8")
    allowed = frozenset(replan) - {"operation"} | {"experiment_yaml"}
    assert client.continuation("EXPERIMENT_PLAN", saved, {}, allowed) == replan
    edited = client.continuation(
        "EXPERIMENT_PLAN", saved, {"experiment_yaml": "experiment: {}\n"}, allowed
    )
    assert "experiment_document" not in edited and edited["origin_task_id"] == "t-1"
    assert edited["experiment_yaml"] == "experiment: {}\n"


def test_the_plan_continues_a_recovery_answer_with_a_structured_status(tmp_path: Path) -> None:
    """A stopped Task's recovery read offers its exact replan beside a Task status object."""
    from alphalattice.interface.local_application import client

    replan = {
        "operation": "EXPERIMENT_PLAN",
        "research_input_id": "in-1",
        "input_binding_hash": "b",
        "experiment_document": {"experiment": {"kind": "risk.covariance-development"}},
    }
    recovery = {
        "status": {"lifecycle": "BLOCKED", "task_id": "t-1"},
        "lifecycle": "BLOCKED",
        "next_requests": {"replan": replan},
        "read_request": {"operation": "TASK_RECOVERY", "task_id": "t-1"},
    }
    saved = tmp_path / "recovery.json"
    saved.write_text(json.dumps(recovery), encoding="utf-8")
    allowed = frozenset(replan) - {"operation"} | {"experiment_yaml"}
    assert client.continuation("EXPERIMENT_PLAN", saved, {}, allowed) == replan
    with pytest.raises(
        client.LocalResearchClientError, match="bound_reference_override:research_input_id"
    ):
        client.continuation("EXPERIMENT_PLAN", saved, {"research_input_id": "in-2"}, allowed)
    recovery["next_requests"] = {}
    saved.write_text(json.dumps(recovery), encoding="utf-8")
    with pytest.raises(client.LocalResearchClientError, match="plan_source_unsupported"):
        client.continuation("EXPERIMENT_PLAN", saved, {}, allowed)


def test_a_schema_says_which_command_writes_its_declaration() -> None:
    """regression (V125): `schema show` gave an `EXPERIMENT_PLAN` template holding only its
    operation, which the Host refuses when sent; it names the commands that write the
    declaration, and an operation that needs none is answered as before."""

    from alphalattice.interface.local_application import cli

    plan = cli.schema("EXPERIMENT_PLAN")
    assert "`study controls`" in plan["written_by"]
    assert "`study draft`" in plan["written_by"]
    assert plan["template"].splitlines()[1].startswith("# Its declaration is written elsewhere")
    assert "written_by" not in cli.schema("STATUS")


def test_an_answer_is_read_in_parts(live: LocalPortfolioWebSession) -> None:
    """regression (V112): an agent read the whole `experiment show` answer seven times to find
    one fact; `--section` prints one part, a path the answer lacks is reported beside the
    owner's outcome with the whole answer shown, and the readback's schema outlines a Factor
    result's parts from the Factor Desk's own models."""

    from alphalattice.interface.local_application import cli

    code, body, _ = _cli(live.workspace, "task", "list", "--section", "tasks")
    assert code == 0 and body["data"]["section"] == "tasks"
    assert isinstance(body["data"]["value"], list)
    code, body, _ = _cli(live.workspace, "task", "list", "--section", "tasks.frobnicate")
    assert (code, body["outcome"]) == (0, "OK") and "tasks" in body["data"]
    assert body["local_failure"]["failure_code"] == "local_client.section_unknown:tasks.frobnicate"
    # V319: a path written from the envelope (`data.`) is refused beside the owner's OK with the
    # answer's own parts, so the next call asks for one.
    code, body, _ = _cli(live.workspace, "task", "list", "--section", "data.tasks")
    assert (code, body["outcome"]) == (0, "OK")
    assert "tasks" in body["local_failure"]["sections"]
    parts = cli.schema("EXPERIMENT_READBACK")["answer_parts"]["factor.screening-development"]
    items = parts["result"]["evidence_report"]["items"]
    assert parts["result"]["evidence_report"]["hypothesis_count"] == "integer"
    assert items[0]["classification"] == "string"


def test_a_metric_read_alone_carries_its_unit(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """regression (V343): `--section position.one_way_turnover` answered the value alone, so an
    agent reported turnover with no unit; a part whose unit the answer names in its
    `metric_units` carries it, the comparison's own, and a part with none answers as before."""

    from alphalattice.control.product_host.research_authoring.comparison import POSITION_UNITS
    from alphalattice.interface.local_application import cli, client

    body = {
        "status": "OK",
        "position": {"one_way_turnover": 0.12, "session": "2024-01-03"},
        "metric_units": {f"position.{key}": unit for key, unit in POSITION_UNITS.items()},
    }

    class Host:
        """The running Host, answering any request with ``body``."""

        def __init__(self, workspace: Path, **_kwargs: Any) -> None:
            self.workspace, self.goal = workspace, None

        def exchange(self, _document: Any) -> tuple[dict[str, Any], bytes]:
            return body, json.dumps(body).encode("utf-8")

        def selected_url(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(client, "LocalResearchClient", Host)

    def section(path: str) -> Any:
        arguments = ["decision", "list", "--section", path]
        cli.main(["--workspace", str(tmp_path), *arguments], serve=lambda _: 99)
        return json.loads(capsys.readouterr().out)["data"]

    assert section("position.one_way_turnover") == {
        "section": "position.one_way_turnover",
        "value": 0.12,
        "unit": "portfolio fraction",
    }
    assert section("position.session") == {"section": "position.session", "value": "2024-01-03"}


def test_a_next_command_keeps_the_context_its_answer_came_in(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """regression (V406, the review of the CLI's repeated context): a printed next command named
    only the interpreter, the script and the workspace, so an agent that ran `--view compact` and
    copied it got the whole answer; it keeps the caller's view, language and named goal."""

    from alphalattice.interface.local_application import cli, client

    task, goal = str(uuid4()), str(uuid4())
    body = {
        "status": "SUCCEEDED",
        "next_requests": {"show": {"operation": "EXPERIMENT_READBACK", "task_id": task}},
    }

    class Host:
        """The running Host, answering any request with ``body``."""

        def __init__(self, workspace: Path, **kwargs: Any) -> None:
            self.workspace, self.goal = workspace, kwargs.get("goal")

        def exchange(self, _document: Any) -> tuple[dict[str, Any], bytes]:
            return body, json.dumps(body).encode("utf-8")

        def selected_url(self, *_args: Any) -> None:
            return None

        def navigation(self, *_args: Any) -> dict[str, str]:
            return {}

    monkeypatch.setattr(client, "LocalResearchClient", Host)
    context = ("--view", "full", "--lang", "zh", "--goal", goal)
    cli.main(["--workspace", str(tmp_path), *context, "decision", "list"], serve=lambda _: 99)
    whole = json.loads(capsys.readouterr().out)["next_commands"]["show"]
    assert f" --view full --lang zh --goal {goal} study show {task}" in whole, whole
    cli.main(["--workspace", str(tmp_path), "decision", "list"], serve=lambda _: 99)
    shown = json.loads(capsys.readouterr().out)["next_commands"]["show"]
    # The compact default names no view (V408), and a command the Host wrote stays whole, its
    # ids and paths as they run (V478).
    assert shown.endswith(f" study show {task}") and "--view" not in shown, shown


def test_the_reading_flags_reach_the_client(tmp_path: Path, monkeypatch: Any) -> None:
    """regression (V135, V112, V130): the CLI built the client's arguments field by field and
    left out `--list-next` and `--section`, so a listed item's requests were never listed; the
    reading flags, `--declaration` among them, reach the client."""

    from alphalattice.interface.local_application import cli, client

    seen: list[Any] = []

    def run(args: Any, **_kwargs: Any) -> int:
        seen.append(args)
        print(json.dumps({"next_commands": {"recovery:t-1": "a command"}}))
        return 0

    monkeypatch.setattr(client, "run", run)
    arguments = ["decision", "list", "--list-next", "--section", "decisions"]
    cli.main(["--workspace", str(tmp_path), *arguments], serve=lambda _: 99)
    assert seen and seen[0].list_next is True and seen[0].section == "decisions"
    declaration = tmp_path / "d.yaml"
    cli.main(
        ["--workspace", str(tmp_path), "study", "controls", "--save-declaration", str(declaration)],
        serve=lambda _: 99,
    )
    assert seen[-1].declaration == declaration


def test_a_contract_failure_answers_one_located_shape() -> None:
    """regression (V248): a document that failed its contract was answered in two shapes: the
    research case and experiment paths gave `fields` and `message` beside their code, and the
    other operations folded each field into the code (`fallback:field=reason`); every operation
    now answers the owner's code, the fields by path, each field's reason and the contract's
    words, never the value."""

    from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

    from alphalattice.interface.local_application.failure_codes import located_failure

    class Document(BaseModel):  # type: ignore[misc]
        model_config = ConfigDict(extra="forbid")
        top_k: int
        name: str

        @field_validator("name")
        @classmethod
        def _known(cls, value: str) -> str:
            if value != "known":
                raise ValueError("portfolio_research.name_unknown")
            return value

    with pytest.raises(ValidationError) as caught:
        Document.model_validate({"top_k": "private-input-marker", "name": "x", "extra": 1})
    answer = located_failure(caught.value, "portfolio_research.refused")
    assert answer["failure_code"] == "portfolio_research.refused"
    assert answer["fields"] == [["top_k"], ["name"], ["extra"]]
    assert answer["reasons"] == {
        "top_k": "int_parsing",
        "name": "portfolio_research.name_unknown",
        "extra": "extra_forbidden",
    }
    assert "name: portfolio_research.name_unknown" in str(answer["message"])
    assert "private-input-marker" not in json.dumps(answer)
    # Any other failure answers the code alone, the owner's where it raised one.
    assert located_failure(ValueError("portfolio_research.top_k_outside"), "x.refused") == {
        "failure_code": "portfolio_research.top_k_outside"
    }


_INTERNAL_CODE = re.compile(
    r"\b(?:V\d{2,3}|U\d{1,3}|(?:ID|OW|OP|PA|DA|EV|TE|PR|SC|LY|TY|ST|GY|WK|GR|AS|AX|NM|RR|FU|LS)"
    r"\d{1,3}|CLI-\d+)\b|\bGate [A-Z]\b|\bStage \d\b|\bPolicy [A-Z]\b"
    r"|\((?:[A-Z]{1,4}\d{0,3}[a-z]?)(?:, ?[A-Z]{1,4}\d{0,3}[a-z]?)*\)"
)
"""An internal row, law, card or stage code: `V451`, `OP13`, `(EX)`, `Gate I`, `Stage 6`."""
_PRODUCT_WORDS = frozenset({"(CLI)", "(CPU)", "(CRO)", "(ISO)", "(PM)", "(UTC)"})
"""Parenthesised capitals a reader knows, which are words, not codes."""


def test_the_product_text_names_no_internal_code() -> None:
    """requirement (V451, NM1's product-text half): what a reader of the product sees names no
    internal row, law, card or stage code: every operation's `schema show` (its request, answer
    and declaration prose), every command's help, and the answer, operation, refusal and label
    tables. Codes stay in the code's own comments and the plans."""

    import argparse

    from alphalattice.interface.local_application import cli

    texts: list[tuple[str, str]] = []

    def collect(where: str, value: object) -> None:
        if isinstance(value, str):
            texts.append((where, value))
        elif isinstance(value, dict):
            for key, item in value.items():
                collect(f"{where}.{key}", item)
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                collect(f"{where}[{index}]", item)

    for operation in sorted({name for names in cli._commands().values() for name in names}):
        collect(f"schema {operation}", cli.schema(operation))
    parsers = [("alphalattice", cli._parser())]
    while parsers:
        name, parser = parsers.pop()
        texts.append((f"{name} --help", parser.format_help()))
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                parsers.extend((f"{name} {word}", child) for word, child in action.choices.items())
    tables = Path(cli.__file__).parent
    for name in ("answers.json", "operations.json", "refusal_words.json"):
        collect(name, json.loads((tables / name).read_text(encoding="utf-8")))
    for row in json.loads((tables / "labels.json").read_text(encoding="utf-8"))["labels"]:
        for key in ("title", "summary", "title_zh", "summary_zh"):
            texts.append((f"labels.json {row['id']}.{key}", row[key]))
    named = {
        (where, match.group(0))
        for where, text in texts
        for match in _INTERNAL_CODE.finditer(text)
        if match.group(0) not in _PRODUCT_WORDS
    }
    assert len(texts) > 1000 and not named, sorted(named)[:20]


def test_an_answer_carries_the_titles_of_the_installed_ids_it_names() -> None:
    """requirement (V451, NM1's product-text half): the CLI names what an answer's installed ids
    are, from the one label table, beside the ids the owner wrote: a strategy, a component and a
    weight rule, as a key or a value; in Chinese under `--lang zh`; none when none is named."""

    from alphalattice.interface.local_application.cli_contract import ANSWER_LANGUAGE, envelope
    from alphalattice.interface.local_application.labels import label

    body = {
        "status": "OK",
        "strategies": {"RETURN_G6_MU_ONLY": {"weight_rule": "ew"}},
        "component_id": "G2_R0_TREND",
    }
    answer = envelope(operation="STRATEGY_SHOW", outcome="OK", body=body, elapsed_seconds=0.0)
    assert answer["titles"] == {
        "G2_R0_TREND": "Trend Candidate Scores",
        "RETURN_G6_MU_ONLY": "Rebound Return Book",
        "ew": "Equal Name Weights",
    }
    assert answer["data"] == body
    language = ANSWER_LANGUAGE.set("zh")
    try:
        chinese = envelope(operation="STRATEGY_SHOW", outcome="OK", body=body, elapsed_seconds=0)
    finally:
        ANSWER_LANGUAGE.reset(language)
    rebound = label("RETURN_G6_MU_ONLY")
    assert rebound is not None and chinese["titles"]["RETURN_G6_MU_ONLY"] == rebound.title_zh
    plain = envelope(operation="STATUS", outcome="OK", body={"status": "OK"}, elapsed_seconds=0)
    assert "titles" not in plain


def test_a_failed_item_answers_its_rule_never_an_empty_list() -> None:
    """regression (V453, AX15's finding): a Feature edit whose specification broke a rule was
    answered `value_error` at the edit, and the edits list as holding no item, so an agent met
    the same refusal twice; the answer now names the field, the rule's code and what the
    controls expect, and a list whose item failed adds nothing of its own."""

    from pydantic import ValidationError

    from alphalattice.foundation.feature_engine.catalog.research import ResearchFeatureChange
    from alphalattice.interface.local_application.failure_codes import located_failure

    specification = {
        "factor_id": "example_factor",
        "family": "liquidity",
        "formula_ref": "factor.example_factor.v1",
        "formula": "an example",
        "window_sessions": 5,
        "lag_sessions": 0,
        "return_convention": "dimensionless",
        "required_fields": ["volume_raw", "close_raw"],
        "literature_sources": ["urn:example"],
        "minimum_observations": 5,
        "absolute_tolerance": 0,
        "relative_tolerance": 0,
        "track": "model",
    }
    document = {
        "input_binding_hash": "0" * 64,
        "base_revision_hash": "0" * 64,
        "edits": [
            {"operation": "CREATE", "factor_id": "example_factor", "specification": specification}
        ],
        "reason": "an example",
    }
    with pytest.raises(ValidationError) as caught:
        ResearchFeatureChange.model_validate_json(json.dumps(document))
    answer = located_failure(caught.value, "feature_research.document_invalid")
    assert answer["reasons"] == {
        "edits.0.specification.required_fields": "factor_spec.values_sorted_unique"
    }
    assert "sorted and without repeats" in str(answer["message"])
    assert "volume_raw" not in json.dumps(answer)


def test_every_field_schema_prints_is_described() -> None:
    """requirement (SC3): a field's meaning is written once, in its contract, and `schema show`
    prints it: every request field of every operation, every field of a typed answer and every
    key of a Desk's declaration section (V249) carries its description."""

    from alphalattice.interface.local_application import cli
    from alphalattice.interface.local_application.operations import OPERATIONS

    bare: list[str] = []
    for operation in OPERATIONS:
        shown = cli.schema(operation)
        bare += [
            f"{operation}.{name}"
            for name, spec in shown["properties"].items()
            if not spec.get("description")
        ]
        answer = shown["answer_schema"] or {}
        for part in (answer, *answer.get("$defs", {}).values()):
            bare += [
                f"{operation} answer {part.get('title')}.{name}"
                for name, spec in part.get("properties", {}).items()
                if not spec.get("description")
            ]
    sections = cli.schema("EXPERIMENT_PLAN")["declaration_sections"]
    assert set(sections) == {
        "alpha",
        "alpha (MODEL_LIFECYCLE_REPLAY)",
        "factor",
        "portfolio",
        "risk",
    }
    # A Portfolio study sized by a linked Risk study is found from the CLI alone (V310).
    assert {"risk_task_id", "policy", "weight_rule"} <= set(sections["portfolio"]["properties"])
    # What the catalog holds and a study may not declare yet, each with its reason (V313).
    unavailable = {v for row in sections["portfolio"]["not_available"] for v in row["values"]}
    assert unavailable == {
        "mu.iv1",
        "mu.iv2",
        "RETURN_SCALED_TOTAL_SIGNAL",
        "RANK_BUFFERED_SCORE_RISK_COST",
        "STRATIFIED_TOP_K_EQUAL_WEIGHT",
    }
    # Each says its field, its values and why, in words; the row that tracks it stays in the
    # plans (NM1b).
    assert all(
        row["reason"] and set(row) == {"field", "values", "reason"}
        for row in sections["portfolio"]["not_available"]
    )
    for name, section in sections.items():
        for part in (section, *section.get("$defs", {}).values()):
            bare += [
                f"EXPERIMENT_PLAN section {name} {part.get('title')}.{field}"
                for field, spec in part.get("properties", {}).items()
                if not spec.get("description")
            ]
    assert bare == []


def test_the_declaration_dialect_reads_yaml_core_scalars_and_keeps_dates() -> None:
    """regression (V278): the declaration loader kept YAML 1.1's implicit types, so an authored
    `010` read as 8 and `yes` or `on` as true; it reads YAML 1.2's core rules, keeps the dates
    authored sessions are written as, and writes back what it reads."""

    from datetime import date

    from alphalattice.protocols.research_authoring.selection import (
        dump_declaration,
        load_safe_yaml_document,
    )

    text = "a: yes\nb: on\nc: 010\nd: 0x1A\ne: 0o17\nf: true\ng: 1e-10\nh: 2024-03-01\ni: 1_000\n"
    read = load_safe_yaml_document(text)
    assert read == {
        "a": "yes",
        "b": "on",
        "c": 10,
        "d": 26,
        "e": 15,
        "f": True,
        "g": 1e-10,
        "h": date(2024, 3, 1),
        "i": "1_000",
    }
    assert load_safe_yaml_document(dump_declaration(read)) == read
    kept = {"v": "010", "w": "true", "x": "1e-10"}
    assert load_safe_yaml_document(dump_declaration(kept)) == kept


def test_list_next_answers_in_the_one_envelope(live: LocalPortfolioWebSession) -> None:
    """regression (V261): `--list-next` printed tab-separated lines, and an answer with no next
    command went to stderr with stdout left empty; it answers in the one envelope, its data the
    listed requests as commands and templates."""

    code, body, stdout = _cli(live.workspace, "task", "list", "--list-next")
    assert (code, body["outcome"]) == (0, "OK") and len(stdout.strip().splitlines()) == 1
    assert body.keys() >= ENVELOPE
    assert set(body["data"]) == {"next_commands", "next_templates"}
    assert body["data"]["next_commands"] == body["next_commands"]


def test_an_answer_saved_as_yaml_reads_back_and_the_declaration_has_its_own_file(
    tmp_path: Path,
) -> None:
    """regression (V130): `--format yaml --output` saved only the editable declaration, which
    `--from` could not read; `--output` saves the whole answer in the format asked and `--from`
    reads it back, and `--declaration` saves the declaration alone."""

    import pytest

    from alphalattice.interface.local_application import client

    answer = {
        "status": "DRAFT_READY",
        "plan_request": {"operation": "EXPERIMENT_PLAN", "origin_task_id": "t-1"},
        "document": {"experiment": {"kind": "factor.screening-development"}, "tolerance": "1e-10"},
        "yaml": "experiment:\n  kind: factor.screening-development\n",
    }
    saved = tmp_path / "draft.yaml"
    client._save_output(saved, answer, json.dumps(answer).encode("utf-8"), "yaml")
    assert client._response_document(saved) == answer  # "1e-10" stays a string (V146)
    declaration = tmp_path / "declaration.yaml"
    client._save_declaration(declaration, answer)
    assert declaration.read_text(encoding="utf-8") == answer["yaml"]
    with pytest.raises(client.LocalResearchClientError, match="declaration_unavailable"):
        client._save_declaration(tmp_path / "none.yaml", {"status": "OK"})


def test_schema_show_writes_its_output_file(tmp_path: Path) -> None:
    """regression (V141): `schema show --output schema.json` exited 0 and wrote no file."""

    saved = tmp_path / "schema.json"
    code, answer, _ = _cli(tmp_path, "schema", "show", "STATUS", "--output", str(saved))
    assert code == 0 and answer["output_file"] == str(saved.resolve())
    assert json.loads(saved.read_text(encoding="utf-8"))["operation"] == "STATUS"


def test_a_model_check_writes_its_output_file_passed_or_refused(tmp_path: Path) -> None:
    """regression (V349, AX9): `model check --output` answered on stdout and wrote no file; the
    owner's answer is saved whole, a refused check's included."""

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


def test_an_operation_only_a_person_completes_is_marked_so(
    live: LocalPortfolioWebSession,
) -> None:
    """regression (V143): the CLI listed operations only a person may complete, whose requests
    it always sends as a client's and which are always refused, and `operation list` did not
    say so; each is marked there and in its help, and each owner refuses a client's request
    with its own person-only refusal."""

    from alphalattice.interface.local_application.operations import (
        COMMANDS,
        GRAMMAR,
        PERSON_ONLY,
        fields,
        flag_name,
    )

    code, body, _ = _cli(live.workspace, "operation", "list")
    contracts = body["data"]["operation_fields"]
    assert code == 0
    assert {op for op, contract in contracts.items() if contract.get("person_only")} == PERSON_ONLY
    values = {
        "network_enabled": "true",
        "usage_reading_enabled": "false",
        "input_pinned": "true",
        "task_id": str(uuid4()),
        "automation_enabled": "true",
        "automation_package_ids": '["a-package"]',
    }
    names = {op: command for command, ops in COMMANDS.items() for op in ops}
    for operation in sorted(PERSON_ONLY):
        noun, verb = names[operation]
        shown = GRAMMAR[operation].positional
        arguments = [
            part
            for name in sorted(fields(operation)[0])
            if name != shown
            for part in (f"--{flag_name(name)}", values.get(name, "a" * 64))
        ] + ([values.get(shown, "a" * 64)] if shown in fields(operation)[0] else [])
        code, body, _ = _cli(live.workspace, noun, verb, *arguments)
        assert body["outcome"] == "REFUSED" and "human" in body["failure_code"], (operation, body)


def test_a_declaration_that_does_not_parse_is_located_on_every_entry(
    live: LocalPortfolioWebSession,
) -> None:
    """regression (V145): a YAML syntax error was located (line and column) on
    `--experiment-document @file` and only named `document_unparsable` on `--experiment-yaml`,
    whose text the Host parses; the Host's refusal carries the line and column too."""

    body = _json(
        live,
        "/api/experiments/plan",
        method="POST",
        payload={"experiment_yaml": "experiment:\n  kind: [factor\n"},
    )
    assert body["failure_code"] == "research_authoring.document_unparsable"
    assert body["document_location"] == {"line": 3, "column": 1}


def test_a_catalog_of_refused_controls_is_an_answer() -> None:
    """regression (V333): the controls answer lists the controls a book refuses under
    `refused`; that catalog is an answer, and only a code there is a refusal."""

    from alphalattice.interface.local_application.cli_contract import envelope, outcome_of

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


def test_a_strategy_runs_forward_by_a_persons_activation_only(
    live: LocalPortfolioWebSession,
) -> None:
    """requirement (LS1, OW12, V143): a person runs a reviewed research book's strategy forward
    and stops it, in the Workbench; an agent's request is refused by name, a person's request
    naming no book or a strategy that does not run forward is refused with its words and way
    on, and the book's controls say whether the strategy runs forward."""

    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    code, body, _ = _cli(live.workspace, "strategy", "activate", str(uuid4()))
    assert code == 2 and body["failure_code"] == "strategy_activation.human_confirmation_required"
    absent = _json(live, "/api/strategy/activate", method="POST", payload={"task_id": str(uuid4())})
    assert absent["status"] == "REFUSED", absent
    assert absent["failure_code"] == "strategy_activation.book_task_absent"
    assert absent["detail"] and absent["next_requests"] == {"books": {"operation": "CONTROLS"}}
    idle = _json(
        live,
        "/api/strategy/deactivate",
        method="POST",
        payload={"strategy_package_id": "RETURN_G6_MU_ONLY"},
    )
    assert idle["status"] == "REFUSED" and idle["failure_code"] == "strategy_activation.not_active"
    # A refusal's way on carries the package its request named, never the default's (V474).
    package = {"operation": "CONTROLS", "strategy_package_id": "RETURN_G6_MU_ONLY"}
    assert idle["next_requests"] == {"books": package}
    code, scored, _ = _cli(live.workspace, "score", "plan", "--package", "RETURN_G6_MU_ONLY")
    assert code == 2 and scored["next_requests"]["books"] == package, scored
    assert "--package RETURN_G6_MU_ONLY" in scored["next_commands"]["books"]
    controls = live.operations.execute(PortfolioResearchOperationRequest(operation="CONTROLS"))
    # Inactive, with cutoff, conditional book/actionable dates, replay and renewal words.
    assert controls["activation"]["status"] == "INACTIVE"
    assert {
        "information_cutoff",
        "forward_book_first_decided_session",
        "first_actionable_session",
        "replayed_in_sample_forward_sessions",
        "model_renewals",
    } <= set(controls["activation"]["strategy_dates"])
    assert controls["activation"]["strategy_dates"]["model_renewals"]["components"] == []


def test_every_read_of_work_planned_per_strategy_is_keyed_by_its_strategy() -> None:
    """CONTRACT (V595, TE12): work planned for one strategy, each `<KIND>_PLAN` requiring its
    package, is read back by its Task or that strategy's own latest Task, never the workspace's
    latest, which with two strategies is one of them by guess. Over the operation table: each such
    readback takes `strategy_package_id` beside `task_id`; the door reads it through the one
    reader keying the latest by strategy, by its row in that reader's table; and no product code
    hands that owner's readback no Task, so the owner's own unkeyed latest, kept in a module its
    implementation's identity hashes, is never reached. A new kind planned per strategy fails
    here until its readback joins the reader."""

    import ast
    import inspect
    from typing import get_args

    from alphalattice.control.product_host.composition import portfolio_research_operations
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperation,
        PortfolioResearchOperationRequest,
    )

    contract = PortfolioResearchOperationRequest.field_contract
    readbacks = {
        operation.replace("_PLAN", "_READBACK")
        for operation in get_args(PortfolioResearchOperation.__value__)
        if operation.endswith("_PLAN") and "strategy_package_id" in contract(operation)[0]
    }
    for operation in readbacks:
        required, allowed = contract(operation)
        assert not required and {"task_id", "strategy_package_id"} <= allowed, operation
    tree = ast.parse(inspect.getsource(portfolio_research_operations))
    table = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and ast.unparse(node.target) == "_STRATEGY_READS"
    )
    assert isinstance(table, ast.Dict)
    assert {ast.literal_eval(key) for key in table.keys if key is not None} == readbacks

    def readers(node: ast.AST) -> set[str]:
        return {
            call.func.value.attr
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "readback"
            and isinstance(call.func.value, ast.Attribute)
        }

    owners: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and ast.unparse(node.test.left) == "request.operation"
            and "self._strategy_task(request)" in ast.unparse(node)
        ):
            owners[ast.literal_eval(node.test.comparators[0])] = readers(node)
    assert owners.keys() == readbacks and all(len(owner) == 1 for owner in owners.values()), owners
    held = set().union(*owners.values())
    root = Path(portfolio_research_operations.__file__).resolve().parents[3]
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if ".readback(" not in text:
            continue
        for call in ast.walk(ast.parse(text)):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "readback"
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr in held
            ):
                given = call.args[0] if call.args else None
                assert given is not None and ast.unparse(given) != "None", (path, ast.unparse(call))


def test_every_door_whose_way_on_reruns_a_plan_resumes_its_stopped_task() -> None:
    """CONTRACT (V600, V601, TE12): a stop or a deferral whose words send a person back to the
    same plan -- run it again, confirm it again, the plan runs again once its retry time has
    passed -- is resumed by that rerun, and its door answers the Task as it is now. Over the
    door words: each such stop is one its owner's rerun reopens (the data update's set of stops
    a rerun resumes, and a deferral once due, refused before it; the preparation's confirm,
    which reopens any stopped Task); every owner whose Tasks run the data update's stages
    admits through its `resume_stopped`, passes the stages' deferral through as its own, and
    reads that deferral in STATUS; and every door that submits those owners' work, or the
    preparation's, answers through `_run_answer`, the Task's current state. A new stop worded
    as a rerun, a new owner running the data stages or a new door fails here until it joins."""

    import ast
    import inspect
    import re

    from alphalattice.control.product_host.composition import portfolio_research_operations
    from alphalattice.control.product_host.data_preparation import application as preparation
    from alphalattice.control.product_host.maintenance import data_update
    from alphalattice.interface.local_application import cli_contract

    words = json.loads(
        Path(cli_contract.__file__).with_name("refusal_words.json").read_text("utf-8")
    )
    reruns = {
        code
        for code, entry in words.items()
        if re.search(r"\b(run the plan|confirm) again\b|\bplan runs again\b", str(entry["detail"]))
    }
    assert reruns == {
        "workspace_data_update.retry_not_due",
        "workspace_data_update.source_access_not_admitted",
        "workspace_preparation.source_access_not_admitted",
    }
    owner = data_update.WorkspaceDataUpdateApplication
    resume = inspect.getsource(owner.resume_stopped)
    for code in reruns:
        if code == "workspace_data_update.retry_not_due":
            # A deferral: reopened once its retry time has passed, refused before it (V601).
            assert "TaskLifecycle.DEFERRED" in resume and code in resume
        elif code.startswith("workspace_data_update."):
            assert owner.resumes(code), code
        else:
            assert code.startswith("workspace_preparation."), code
            confirm = inspect.getsource(preparation.WorkspacePreparationApplication.confirm)
            assert "allow_blocked=task.lifecycle is TaskLifecycle.BLOCKED" in confirm
            assert "self._sources_allowed()" in confirm

    def tree(module: object) -> ast.Module:
        return ast.parse(inspect.getsource(module))  # type: ignore[arg-type]

    root = Path(data_update.__file__).resolve().parents[2]
    running = {data_update.__name__}
    for path in (root / "product_host").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "data.execute_step(" in text:
            running.add(".".join(path.relative_to(root.parent.parent).with_suffix("").parts))
    commands = set()
    for name in running:
        module = __import__(name, fromlist=["_"])
        for node in ast.walk(tree(module)):
            if isinstance(node, ast.FunctionDef) and node.name == "admit":
                body = ast.unparse(node)
                if "task_control_registry" in body and "envelope" in body:
                    assert "resume_stopped(" in body, (name, node.lineno)
            if isinstance(node, ast.FunctionDef) and "data.execute_step(" in ast.unparse(node):
                # The data stages' deferral is passed through as the owner's own (V601).
                assert "StageDisposition.DEFERRED" in ast.unparse(node), (name, node.name)
            if isinstance(node, ast.ClassDef) and node.name.endswith("Command"):
                commands.add(node.name)
    assert commands >= {"DecisionAdvancementCommand", "WorkspaceDataUpdateCommand"}, commands
    # Each kind running the data stages reads its deferral in STATUS: when, and its resume.
    way = inspect.getsource(portfolio_research_operations.PortfolioResearchOperations._deferred_way)
    assert "DATA_UPDATE_TASK_KIND" in way and "DecisionAdvancementApplication.task_kind" in way
    commands.add(preparation.WorkspacePreparationCommand.__name__)
    door = tree(portfolio_research_operations)
    # Each submission of these owners' work: by the name it is held in, or the call itself.
    held_as = {
        id(node.value): ast.unparse(node.targets[0])
        for node in ast.walk(door)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
    }
    submitted: dict[str, str] = {}
    for call in ast.walk(door):
        if not (isinstance(call, ast.Call) and ast.unparse(call.func) == "self.dispatcher.submit"):
            continue
        made = call.args[0]
        if isinstance(made, ast.Call) and ast.unparse(made.func) in commands:
            submitted[ast.unparse(made.func)] = held_as.get(id(call), ast.unparse(call))
    assert submitted.keys() == commands, submitted
    answered = {
        ast.unparse(call.args[0])
        for call in ast.walk(door)
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "self._run_answer"
    }
    assert set(submitted.values()) <= answered, (submitted, answered)


def test_no_newer_plan_and_no_admitted_task_waits_behind_an_update_that_has_not_ended() -> None:
    """CONTRACT (V604, TE12): a deferral holds the workspace's one running place while no command
    drives it. Every owner whose update can defer answers, at its plan, the update of its own
    package that has not ended -- its run follows it, or resumes a deferral once due -- before it
    reads the workspace's inputs, never a newer plan queued behind it or refused while it holds
    them; the daily automation attends what it admitted, at a deferral's retry time, and tries a
    package refused while the inputs were not ready again once the worker is idle; a refusal for
    those inputs names their state and the request that settles them; and every Task kind
    admitted while the place was held is driven once it frees, never parked. A new deferring
    owner, a plan that reads before it answers, or a dispatcher path that drops a waiting command
    fails here until it joins."""

    import inspect

    from alphalattice.control.product_host.composition import (
        decision_advancement,
        portfolio_research_operations,
        research_update_automation,
    )
    from alphalattice.control.product_host.maintenance import data_update
    from alphalattice.interface.local_application import cli_contract, dispatcher

    root = Path(data_update.__file__).resolve().parents[2]
    deferring = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "product_host").rglob("*.py")
        if "StageDisposition.DEFERRED" in path.read_text(encoding="utf-8")
    )
    # The preparation refuses a new preparation by name while any Task is unfinished
    # (`workspace_preparation.finish_or_recover_existing_task`); the updates answer theirs.
    assert deferring == [
        "product_host/composition/decision_advancement.py",
        "product_host/data_preparation/application.py",
        "product_host/maintenance/data_update.py",
    ], deferring
    plans = {
        "research update": (
            inspect.getsource(decision_advancement.DecisionAdvancementApplication.plan),
            "TaskLifecycle.DEFERRED",
            "workspace_score_source_identity(",
        ),
        "data update": (
            inspect.getsource(data_update.WorkspaceDataUpdateApplication.plan),
            "self._waiting()",
            "read_workspace_inputs(",
        ),
    }
    for owner, (source, waits, reads) in plans.items():
        assert 0 <= source.find(waits) < source.find(reads), owner
    waiting = inspect.getsource(data_update.WorkspaceDataUpdateApplication._waiting)
    assert "TaskLifecycle.DEFERRED" in waiting
    # A research update's own data plan is sealed into its own Task, never a waiting one's.
    research_plan = plans["research update"][0]
    assert "self.data.plan(fresh=True)" in research_plan
    # The update that waits answers whatever target was asked: its match names the package alone.
    start = research_plan.find("TaskLifecycle.DEFERRED")
    waiting_pass = research_plan[start : research_plan.find("for task in tasks:", start)]
    assert "prior.package_id == package_id" in waiting_pass, waiting_pass
    assert "target" not in waiting_pass, waiting_pass
    # The automation attends what it admitted: a deferral at its retry time, a later session
    # once its resumed update publishes, and a package refused while the inputs were held.
    automation = research_update_automation.ResearchUpdateAutomation
    cycle = inspect.getsource(automation._cycle)
    assert "RESEARCH_UPDATE_READBACK" in cycle and "retry_after_at" in cycle
    assert "self._retry" in cycle and "self._after" in cycle and "self._watched" in cycle
    assert '"DEFERRED"' in inspect.getsource(automation._settle)
    assert "self._watched" in inspect.getsource(automation.command_completed)
    # A refusal for the inputs names their state and the request that settles them.
    words = cli_contract.refusal_words(
        "strategy_score.workspace_inputs_not_ready:DATA_REVIEW_PENDING"
    )
    assert "DATA_REVIEW_PENDING" in words["detail"] and "data-update plan" in words["detail"]
    operations = portfolio_research_operations.PortfolioResearchOperations
    way = inspect.getsource(operations._inputs_way)
    assert "read_workspace_inputs(" in way and "readiness_status" in way
    assert "running_place_holder()" in way and '"DATA_UPDATE_PLAN"' in way
    # Only an update holds the inputs with its data stage; another holder is no way on for them.
    assert "DATA_UPDATE_TASK_KIND" in way and "DecisionAdvancementApplication.task_kind" in way
    assert "self._inputs_way(" in inspect.getsource(operations._execute)
    # Every kind admitted while the place was held: its command is kept and driven once the
    # place frees, after any command returns and after a deferral's cancel; owned meanwhile.
    held = dispatcher.LocalBackgroundDispatcher
    assert "self._waits_its_turn(" in inspect.getsource(held._drain)
    assert "self._drive_waiting()" in inspect.getsource(held._drain)
    assert "self._drive_waiting()" in inspect.getsource(held.request_cancel)
    assert "self._waiting" in inspect.getsource(held.command_running)
    # The Host's own sweep never queues behind a deferral.
    assert "TaskLifecycle.DEFERRED" in inspect.getsource(operations.sweep_if_due)


def test_a_books_evidence_continues_that_book_never_the_default() -> None:
    """regression (V473, the user's review at 0b5dc227): `evidence preview --from` a book's
    readback sent the bare operation, which reads the workspace's default book or refuses with
    no book to review. A book request continued from an answer takes the book the answer's own
    requests select, bound as they bind it, and an answer naming two books is refused."""

    from alphalattice.interface.local_application import client
    from alphalattice.interface.local_application.cli_contract import command_table

    def allowed(operation: str) -> frozenset[str]:
        return frozenset(command_table()["fields"][operation]["allowed"])

    task, receipt = str(uuid4()), "c" * 64
    selector = {
        "experiment_task_id": task,
        "experiment_receipt_hash": receipt,
        "portfolio_session": "2026-09-10",
    }
    review = {"operation": "EVIDENCE_CRO", **selector}
    book = {"status": "EXPERIMENT_PUBLISHED", "task_id": task, "next_requests": {"review": review}}
    preview = client.continued("EVIDENCE_PREVIEW", book, {}, allowed("EVIDENCE_PREVIEW"))
    assert preview == {"operation": "EVIDENCE_PREVIEW", **selector}
    other = {"operation": "EVIDENCE_CRO", "result_hash": "d" * 64}
    two = {**book, "next_requests": {"review": review, "other": other}}
    with pytest.raises(client.LocalResearchClientError, match="answer_names_two_books"):
        client.continued("EVIDENCE_PREVIEW", two, {}, allowed("EVIDENCE_PREVIEW"))


def test_a_receipt_that_started_no_task_continues_the_task_it_names() -> None:
    """regression (V479, the user's review at baa0f206): `risk-link list --from` a link's
    receipt was refused (`response_reference_missing`), its book's Task named only inside the
    link. The receipt offers its links read bound to the Task, and a continuation of an answer
    naming no Task of its own takes the one Task its offered requests name; two refuse."""

    from alphalattice.interface.local_application import client
    from alphalattice.interface.local_application.cli_contract import command_table

    allowed = frozenset(command_table()["fields"]["EXPERIMENT_RISK_LINKS"]["allowed"])
    book, other = str(uuid4()), str(uuid4())
    links = {"operation": "EXPERIMENT_RISK_LINKS", "task_id": book}
    receipt = {"status": "RISK_REPORT_LINKED", "task_id": None, "next_requests": {"links": links}}
    assert client.continued("EXPERIMENT_RISK_LINKS", receipt, {}, allowed) == links
    assert client.continued("EXPERIMENT_RISK_LINKS", receipt, {"task_id": book}, allowed) == links
    export = {"operation": "EXPERIMENT_EXPORT", "task_id": book}
    named = {"status": "RISK_REPORT_LINKED", "task_id": None, "next_requests": {"export": export}}
    assert client.continued("EXPERIMENT_RISK_LINKS", named, {}, allowed)["task_id"] == book
    two = {**named, "next_requests": {"export": export, "other": {**export, "task_id": other}}}
    with pytest.raises(client.LocalResearchClientError, match="response_reference_missing"):
        client.continued("EXPERIMENT_RISK_LINKS", two, {}, allowed)


def test_a_door_refusal_filled_with_its_subject_reads_in_chinese(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (U76, V608): door subjects read through their Chinese keys, including V599's
    joined provider needs; an unkeyed part is carried as written, as the page reads it."""

    import json as json_module

    import alphalattice.interface.local_application as local_application
    from alphalattice.control.product_host.maintenance.data_update import NETWORK_WORK_WORDS
    from alphalattice.control.workspace_runtime.network_access import set_network_access
    from alphalattice.interface.local_application import cli_contract

    words_file = Path(local_application.__file__).with_name("refusal_words.json")
    table = json_module.loads(words_file.read_text(encoding="utf-8"))
    templated = [code for code, words in table.items() if "{subject}" in words["detail"]]
    assert templated, "the door's table words no sentence around a subject"
    # These existing Chinese keys describe the person-controlled, closed workspace.
    monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
    set_network_access(tmp_path, enabled=False)
    token = cli_contract.ANSWER_LANGUAGE.set("zh")
    try:
        for code in templated:
            detail = cli_contract.refusal_words(f"{code}:2026-09-10", workspace=tmp_path)["detail"]
            chinese = cli_contract.worded(detail)
            assert chinese != detail and "2026-09-10" in str(chinese), (code, chinese)
        session = "2026-09-14"
        unkeyed = "an unkeyed provider need"
        cases = (
            (
                ("DATA", "MEMBERSHIP", "CANDIDATES"),
                f"截至交易日 {session} 缺失的市场数据\uff1b交易日 {session} 的成员来源检查\uff1b"
                f"截至交易日 {session} 的失败候选数据重试",
            ),
            (("HISTORY",), "已准入的完整历史获取"),
            (
                ("REFERENCE", "SECTOR"),
                f"截至交易日 {session} 的 SPY 市场基准\uff1b交易日 {session} 的候选行业来源检查",
            ),
            (
                ("DATA", "UNKEYED", "MEMBERSHIP"),
                f"截至交易日 {session} 缺失的市场数据\uff1b{unkeyed}\uff1b"
                f"交易日 {session} 的成员来源检查",
            ),
        )
        for needs, expected in cases:
            subject = "; ".join(
                NETWORK_WORK_WORDS.get(need, unkeyed).format(session=session) for need in needs
            )
            detail = cli_contract.refusal_words(
                f"research_update.input_source_access_not_admitted:{subject}", workspace=tmp_path
            )["detail"]
            assert cli_contract.worded(detail) == (
                f"研究更新需要{expected}。此工作区已关闭的网络不允许访问数据提供方\uff1b"
                "请由人允许访问\uff08`network set`\uff09\uff0c然后再次运行已保存的计划。"
            ), needs
    finally:
        cli_contract.ANSWER_LANGUAGE.reset(token)


@pytest.mark.parametrize(
    ("code", "detail", "chinese"),
    (
        (
            "data.empty_payload",
            "The provider returned no daily price history for a listing in the requested period, "
            "so no bars were admitted. Read the data update's current record for the affected "
            "listing and period. Retry the same approved update later when the source history "
            "is available.",
            "提供方没有返回某个标的在所请求期间的每日价格历史\uff0c因此没有准入任何行情记录。"
            "请读取数据更新的当前记录\uff0c查明受影响的标的和期间。待来源历史可用后\uff0c重试同一个"
            "已获批准的更新。",
        ),
        (
            "data.sanitizer.corrupted_payload",
            "The provider price history failed the Data owner's validation and was not admitted. "
            "Read the update's current record for the affected listing. Retry the same approved "
            "update later after the source is corrected, or preview a new membership change "
            "excluding that listing and have a person approve it separately. A new exclusion "
            "does not discharge an earlier formation obligation.",
            "行情提供方的价格历史未通过数据模块的验证\uff0c没有准入。请读取更新的当前记录\uff0c查明受影响的"
            "标的。待来源修正后\uff0c重试同一个已获批准的更新\uff1b也可以预览排除该标的新成员变更\uff0c由人"
            "另行批准。"
            "新排除不会解除已有的组合形成义务。",
        ),
        (
            "workspace_data_update.transition_not_verified",
            "The saved approved membership transition could not be verified against the current "
            "Manifest, prior Panel and approved candidate document. Read the stopped update's "
            "current record and resolve the mismatch through their original owners before "
            "retrying the same approved update. Never edit a stored approval.",
            "已保存并获批的成员变更无法通过当前清单、此前面板及已批准候选文件的核验。"
            "请读取已停止更新的当前记录\uff0c通过这些记录的原所有者解决不一致后\uff0c再重试同一个"
            "已获批准的更新。不要修改已存储的批准记录。",
        ),
    ),
)
def test_data_history_stops_have_installed_words_and_preserve_the_owners_way(
    code: str, detail: str, chinese: str
) -> None:
    """regression (DUPD): refused history and transition bindings read as their causes
    in either installed language, preserving the owner's request and approval requirements.
    """
    from alphalattice.control.product_host.composition.task_recovery import stop_detail
    from alphalattice.interface.local_application import cli_contract

    request = {"show": {"operation": "DATA_UPDATE_READBACK"}}
    bare = {"status": "REFUSED", "failure_code": code}
    worded = cli_contract.worded_refusal(bare)
    assert worded == {
        **bare,
        "detail": detail,
        "next_action": "DATA_UPDATE_READBACK",
    }
    assert cli_contract.refusal_problem(worded) is None
    offered = cli_contract.worded_refusal({**bare, "next_requests": request})
    assert offered == {**bare, "detail": detail, "next_requests": request}
    assert cli_contract.refusal_problem(offered) is None
    if code.startswith("data."):
        for kind in (
            "workspace_preparation",
            "workspace_data_update",
            "research_input_capture",
        ):
            assert stop_detail(kind, code, "TASK_CONTROL") == detail
    for language, expected in (("en", detail), ("zh", chinese)):
        token = cli_contract.ANSWER_LANGUAGE.set(language)
        try:
            assert cli_contract.worded(detail) == expected
        finally:
            cli_contract.ANSWER_LANGUAGE.reset(token)
    token = cli_contract.ANSWER_LANGUAGE.set("zh")
    try:
        for source_detail, translated in (
            ("The provider price history failed validation.", "行情提供方的价格历史未通过验证。"),
            ("The provider price history could not be read.", "无法读取行情提供方的价格历史。"),
            ("The source failure cause was not recorded.", "没有记录来源失败的原因。"),
            ("Provider price history", "提供方价格历史"),
        ):
            assert cli_contract.worded(source_detail) == translated
    finally:
        cli_contract.ANSWER_LANGUAGE.reset(token)


def test_a_model_is_activated_by_a_person_only(live: LocalPortfolioWebSession) -> None:
    """requirement (EX, V143): anyone reads the models' review packets; an agent's activation
    is refused by name, since a person activates a model, in the Workbench."""

    listing = _json(live, "/api/models")
    assert listing["status"] == "AVAILABLE"
    assert {value["model_id"]: value["state"] for value in listing["models"]} == {
        "regularized_linear": "INSTALLED",
        "dynamic_panel_lightgbm": "INSTALLED",
    }
    code, body, _ = _cli(live.workspace, "model", "activate", "regularized_linear")
    assert (code, body["failure_code"]) == (2, "model_extension.human_confirmation_required"), body
    assert body["detail"] and body["next_requests"] == {"models": {"operation": "MODEL_EXTENSIONS"}}


def test_a_compact_answer_shows_references_short_and_they_can_be_sent_so(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """requirement (V393, V478): the compact display shows each typed reference by its first
    twelve characters, its list's items included, and a command may send what it shows; a path,
    a link, a command the Host wrote and every other text stay whole, so a bundle directory
    named after its Task still opens and its submit command still runs (the user's review at
    baa0f206); a short hash the Host never answered with is refused by the client, naming its
    field."""

    from alphalattice.interface.local_application.client import short_references

    task = "927be8a3-305b-5237-9f9e-f765da4fa147"
    note = f"run --task-id {task}; runtime/{task}/a.json, {task}.json"
    bundle = f"D:/ws/bundles/cro-{task}-r1"
    whole = {
        "note": note,
        "answer_file": f"{bundle}/answer.json",
        "submit_arguments": ["bundle", "submit", "--dir", bundle],
        "submit_command": f"alphalattice bundle submit --dir {bundle}",
    }
    shown = short_references(
        {"data": {"task_id": task, "task_ids": [task], **whole}, "url": f"/?t={task}"}
    )
    assert shown == {
        "data": {"task_id": task[:12], "task_ids": [task[:12]], **whole},
        "url": f"/?t={task}",
    }
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


def test_a_saved_goal_completion_runs_as_a_whole_request(live, tmp_path: Path) -> None:
    """P2-NH: the native lead submits GOAL_SHOW's file through request --file.

    The file already binds the Goal and revision. Passing it as goal submit's
    body would nest a whole request in GoalSubmission and refuse real completion.
    """
    import yaml

    from alphalattice.interface.local_application.goals import GoalSubmission

    declaration = tmp_path / "goal.json"
    declaration.write_text(
        json.dumps(
            {
                "title": "Read the completion request",
                "objective": "Check the saved request without research execution",
                "kind": "OPERATIONS",
                "criteria": [{"criterion_id": "read", "text": "Record what was read"}],
            }
        ),
        encoding="utf-8",
    )
    code, opened, _ = _cli(live.workspace, "goal", "open", "--file", str(declaration))
    assert code == 0, opened
    goal_id = opened["data"]["goal_id"]
    completion = tmp_path / "completion.yaml"
    code, shown, _ = _cli(
        live.workspace, "goal", "show", goal_id, "--save-declaration", str(completion)
    )
    assert code == 0, shown
    saved = yaml.safe_load(completion.read_text("utf-8"))
    assert saved["operation"] == "GOAL_SUBMIT" and saved["goal_id"] == goal_id
    assert saved["goal_hash"] == shown["data"]["goal_hash"]
    saved["goal_submission"].update(
        outcome="NOT_ACHIEVED", summary="The saved request was read; no research was executed."
    )
    saved["goal_submission"]["criteria"][0].update(
        answer="NOT_ASSESSED", note="This transport check makes no research judgment."
    )
    completion.write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8", newline="\n")
    code, submitted, _ = _cli(live.workspace, "request", "--file", str(completion))
    assert code == 0 and submitted["operation"] == "GOAL_SUBMIT", submitted
    assert submitted["data"]["goal_id"] == goal_id and submitted["data"]["state"] == "COMPLETE"
    code, readback, _ = _cli(live.workspace, "goal", "show", goal_id)
    assert code == 0, readback
    assert readback["data"]["goal"]["submission"] == GoalSubmission.model_validate(
        saved["goal_submission"]
    ).model_dump(mode="json")


def test_an_item_action_takes_its_id_as_shown(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """requirement (V399, the CLI review's F2): `--action recovery:<task>` selects an item's
    request whether the task id is whole or its shown beginning; the request then reaches the
    Host, which answers for that Task."""

    task = "927be8a3-305b-5237-9f9e-f765da4fa147"
    other = "11111111-2222-3333-4444-555555555555"
    answer = tmp_path / "answer.json"
    answer.write_text(
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
    code, body, _ = _cli(
        live.workspace, "request", "--from", str(answer), "--action", f"recovery:{task[:12]}"
    )
    assert (code, body["failure_code"]) == (2, "task_control.task_not_found"), body


def test_max_wait_bounds_a_wait_on_a_host_that_never_answers(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """requirement (V400, the CLI review's F7): `activity wait --max-wait` counts from the
    waiter's start, its first read included, and no request inside it waits past the time
    left, so a Host that accepts and never answers ends the wait as MAX_WAIT_REACHED in
    seconds, not after the client's 120-second HTTP limit."""

    import time

    empty = tmp_path / "silent"
    (empty / "runtime").mkdir(parents=True)
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(8)  # accepts into its backlog, never answers
        port = silent.getsockname()[1]
        record = json.loads(
            (live.workspace / "runtime/local-research-connection.json").read_text("utf-8")
        )
        record.update(workspace=str(empty.resolve()), url=f"http://127.0.0.1:{port}")
        (empty / "runtime/local-research-connection.json").write_text(json.dumps(record), "utf-8")
        started = time.monotonic()
        code, body, _ = _cli(empty, "activity", "wait", "--task", str(uuid4()), "--max-wait", "0.5")
        elapsed = time.monotonic() - started
    assert code == 3 and body["data"]["wait_event"]["event"] == "MAX_WAIT_REACHED", body
    assert elapsed < 30, elapsed


def test_the_grammar_reads_files_stdin_and_operands_as_gnu_does(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """requirement (LAWS OP2, the CLI grammar): a document is a plain path or `-` for stdin;
    `--` ends the options; a command acts on its positional operand; a command with two
    operations refuses both selectors; `--choices` needs `--from` and a field given twice is
    refused by name; `schema show` takes a command as it is run, a two-operation command's
    branches under `oneOf`; `--from -` chooses a branch as a file does (CG3)."""

    declaration = tmp_path / "goal.yaml"
    code, body, _ = _cli(live.workspace, "goal", "schema", "--save-declaration", str(declaration))
    assert code == 0, body
    # The saved declaration heads with its allowed values, which the loader ignores (AGENT-TIME
    # R2: AX's agents looked them up with `answer show`).
    head = declaration.read_text(encoding="utf-8").splitlines()
    assert head[0] == "# Allowed values:"
    assert any(line.startswith("# research.purpose: NEW_RESEARCH | ") for line in head)
    opened = subprocess.run(
        [
            *(sys.executable, str(SCRIPT), "--workspace", str(live.workspace), "--view", "full"),
            *("goal", "open", "--file", "-"),
        ],
        input=declaration.read_text(encoding="utf-8"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    answer = json.loads(opened.stdout.strip().splitlines()[-1])
    assert opened.returncode == 0, answer
    goal_id = answer["data"]["goal_id"]
    code, body, _ = _cli(live.workspace, "goal", "show", "--", goal_id)
    assert (code, body["data"]["goal_id"]) == (0, goal_id), body

    code, body, _ = _cli(live.workspace, "study", "show", str(uuid4()), "--plan", "a" * 64)
    assert code == 1 and "not both" in body["usage_error"]["observed"], body
    code, body, _ = _cli(live.workspace, "goal", "open", "--choices", str(declaration))
    assert code == 1 and "--from" in body["usage_error"]["observed"], body

    offered = tmp_path / "offered.json"
    offered.write_text(
        json.dumps(
            {"next_requests": {"open": {"operation": "GOAL_OPEN", "goal_declaration": None}}}
        ),
        encoding="utf-8",
    )
    choices = tmp_path / "choices.yaml"
    choices.write_text("goal_declaration: {title: twice}\n", encoding="utf-8")
    code, body, _ = _cli(
        live.workspace,
        *("goal", "open", "--from", str(offered), "--file", str(declaration)),
        *("--choices", str(choices)),
    )
    assert (code, body["failure_code"]) == (1, "local_client.field_given_twice:goal_declaration"), (
        body
    )

    code, body, _ = _cli(tmp_path, "schema", "show", "study", "plan")
    assert (code, body["data"]["operation"]) == (0, "EXPERIMENT_PLAN"), body
    assert body["data"]["command"] == "alphalattice study plan"
    # One declaration, its YAML text or its object, as the plan's owner takes it (V409).
    assert body["data"]["oneOf"] == [
        {"required": ["experiment_yaml"]},
        {"required": ["experiment_document"]},
    ]
    code, body, _ = _cli(tmp_path, "schema", "show", "study", "show")
    branches = body["data"]["oneOf"]
    assert (code, body["data"]["operations"]) == (0, [b["operation"] for b in branches]), body
    assert sorted(tuple(b["required"]) for b in branches) == [
        ("experiment_plan_hash",),
        ("task_id",),
    ]

    shown = subprocess.run(
        [
            *(sys.executable, str(SCRIPT), "--workspace", str(live.workspace), "--view", "full"),
            *("study", "show", "--from", "-"),
        ],
        input=json.dumps({"data": {"task_id": str(uuid4())}}),
        capture_output=True,
        text=True,
        timeout=120,
    )
    answer = json.loads(shown.stdout.strip().splitlines()[-1])
    assert answer["operation"] == "EXPERIMENT_READBACK" and "usage_error" not in answer, answer


def test_every_answer_names_its_context_and_the_help_opens_with_the_common_path(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """requirement (LAWS OP2, CG4: Git's and the agent CLIs' conventions): every answer names
    the workspace, the goal the Host counted the request toward and the agent session; the
    top-level help opens with the common path and lists every object in the research's order."""

    code, body, _ = _cli(live.workspace, "task", "list")
    assert code == 0 and body["context"]["workspace"] == str(live.workspace.resolve()), body
    assert body["context"]["goal"] is None
    declaration = tmp_path / "goal.yaml"
    assert _cli(live.workspace, "goal", "schema", "--save-declaration", str(declaration))[0] == 0
    code, opened, _ = _cli(live.workspace, "goal", "open", "--file", str(declaration))
    goal_id = opened["data"]["goal_id"]
    _code, planned, _ = _cli(live.workspace, "--goal", goal_id, "data-update", "plan")
    assert planned["context"]["goal"] == goal_id, planned

    shown = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60
    )
    words = " ".join(shown.stdout.split())
    assert "The common path: workspace show, goal take <id>, study controls" in words, words
    assert "Objects, in the research's order: Start and follow the work:" in words


def test_the_global_options_are_read_anywhere_on_the_line(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """regression (V412, AX14's first finding): an agent wrote `study show <task> --view full`
    and was refused twice, the global options being read only before the object; every command
    reads them after its action too, as GNU reads options, and a missing workspace is still
    refused by name."""

    from alphalattice.interface.local_application import cli, client

    seen: list[Path] = []

    class Host:
        """The running Host, answering any request with its status."""

        def __init__(self, workspace: Path, **kwargs: Any) -> None:
            self.workspace, self.goal = workspace, kwargs.get("goal")
            seen.append(workspace)

        def exchange(self, document: Any) -> tuple[dict[str, Any], bytes]:
            return {"status": "OK", "task_id": document["task_id"]}, b"{}"

        def selected_url(self, *_args: Any) -> None:
            return None

        def navigation(self, *_args: Any) -> dict[str, str]:
            return {}

    monkeypatch.setattr(client, "LocalResearchClient", Host)
    task = str(uuid4())
    after = ["study", "show", task, "--view", "full", "--workspace", str(tmp_path)]
    assert cli.main(after, serve=lambda _: 99) == 0
    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    read = {"operation": "EXPERIMENT_READBACK", "task_id": task}
    assert answer["data"] == {"status": "OK", "task_id": task, "read_request": read}, answer
    assert seen == [tmp_path]
    # Outside a bound session, wherever this runs, a line naming none is refused by name and
    # nothing is sent (V568).
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    assert cli.main(["task", "list"], serve=lambda _: 99) == 1
    refused = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert refused["failure_code"] == "local_client.workspace_unbound", refused
    assert "--workspace" in refused["detail"] and seen == [tmp_path]


def test_a_compact_answer_holds_its_next_requests_to_the_read(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """regression (V411, the DOC line's S1b-F17): a compact answer was one read only while its
    next requests were few; 200 of them printed 80,551 bytes, and 141,129 with `--list-next`.
    The compact answer keeps the owner's first next requests and counts the rest; `--list-next`
    lists every one, a page at a time within the read; `--next-from` alone is refused."""

    from alphalattice.interface.local_application import cli, client

    offered = {f"show_{i}": {"operation": "STATUS", "task_id": str(uuid4())} for i in range(200)}
    tasks = [
        {
            "task_id": str(uuid4()),
            "next_requests": {"recover": {"operation": "RECOVER", "task_id": str(uuid4())}},
        }
        for _ in range(60)
    ]
    body = {"status": "OK", "tasks": tasks, "next_requests": offered}

    class Host:
        """The running Host, answering with many next requests."""

        def __init__(self, workspace: Path, **kwargs: Any) -> None:
            self.workspace, self.goal = workspace, kwargs.get("goal")

        def exchange(self, _document: Any) -> tuple[dict[str, Any], bytes]:
            return body, json.dumps(body).encode("utf-8")

        def selected_url(self, *_args: Any) -> None:
            return None

        def navigation(self, *_args: Any) -> dict[str, str]:
            return {}

    monkeypatch.setattr(client, "LocalResearchClient", Host)

    def answer(*extra: str) -> tuple[int, dict[str, Any]]:
        cli.main(["--workspace", str(tmp_path), "task", "list", *extra], serve=lambda _: 99)
        line = capsys.readouterr().out.strip().splitlines()[-1]
        return len(line.encode("utf-8")), json.loads(line)

    size, compact = answer()
    kept = compact["next_requests"]
    assert size <= client.COMPACT_ANSWER_BYTES, size
    # The owner's first, in its order (the envelope prints its keys sorted).
    assert set(kept) == set(list(offered)[: len(kept)]) == set(compact["next_commands"])
    assert compact["next_left"] == {"count": 200 - len(kept), "list": "--list-next"}

    whole = answer("--list-next", "--view", "full")[1]["data"]
    listed = {**whole["next_commands"], **(whole["next_templates"] or {})}
    assert len(listed) == 260
    paged: dict[str, str] = {}
    start: str | None = None
    while True:
        size, page = answer("--list-next", *(("--next-from", start) if start else ()))
        assert size <= client.COMPACT_ANSWER_BYTES and page["next_commands"] is None, size
        assert not set(paged) & set(page["data"]["next_commands"])
        paged.update(page["data"]["next_commands"])
        if page["next_left"] is None:
            break
        start = page["next_left"]["list"].rsplit(" ", 1)[1]
    # Each listed once; a full view's command carries --view full and whole ids (V406, V393).
    assert set(paged) == set(listed)
    # Each page holds to the read whatever length the workspace path adds (FLOW-0): next_left's
    # key was left out of a page's room, a page one byte over at one temporary path's length.
    for pad in range(1, 161):
        command = ["--workspace", str(tmp_path / ("w" * pad)), "task", "list", "--list-next"]
        cli.main(command, serve=lambda _: 99)
        line = capsys.readouterr().out.strip().splitlines()[-1]
        assert len(line.encode("utf-8")) <= client.COMPACT_ANSWER_BYTES, pad
    refused = answer("--next-from", "3")[1]
    assert (refused["outcome"], refused["failure_code"]) == (
        "INVALID_INPUT",
        "local_client.next_from_invalid",
    ), refused


def test_a_data_update_runs_from_its_saved_plan(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """regression (V418, an outside review at 3fa785fd): the plan offered no next request, so
    `data-update run --from plan.json` could not take its hash and the card taught copying it;
    the run starts from the saved plan, the hash bound by the Host."""

    from alphalattice.interface.local_application import cli, client

    sent: list[dict[str, Any]] = []

    class Host:
        """The running Host, recording what it was sent."""

        def __init__(self, workspace: Path, **kwargs: Any) -> None:
            self.workspace, self.goal = workspace, kwargs.get("goal")

        def exchange(self, document: Any) -> tuple[dict[str, Any], bytes]:
            sent.append(dict(document))
            return {"status": "ADMITTED", "task_id": str(uuid4())}, b"{}"

        def selected_url(self, *_args: Any) -> None:
            return None

        def navigation(self, *_args: Any) -> dict[str, str]:
            return {}

    monkeypatch.setattr(client, "LocalResearchClient", Host)
    plan_hash = "c" * 64
    saved = tmp_path / "update-plan.json"
    run = {"operation": "DATA_UPDATE_RUN", "update_plan_hash": plan_hash}
    saved.write_text(
        json.dumps({"status": "PLANNED", "plan_hash": plan_hash, "next_requests": {"run": run}}),
        encoding="utf-8",
    )
    arguments = ["--workspace", str(tmp_path), "data-update", "run", "--from", str(saved)]
    # An admitted Task answers PENDING (exit 3, OP3); what matters is the request sent.
    assert cli.main(arguments, serve=lambda _: 99) == 3, capsys.readouterr().out
    assert sent == [run]


def test_a_flows_next_request_is_read_from_the_workspace_answer(tmp_path: Path) -> None:
    """regression (V425, an outside review at 97b65a25): `workspace show` offers each flow's
    next requests under `intents[].flows.*`, which the collector never entered, so `request
    --from workspace.json --action curation` failed and `--list-next` listed none. A request
    offered in two places is listed once; two flows offering one name are told apart by
    place."""

    import argparse

    from alphalattice.interface.local_application import client

    task = str(uuid4())
    curation = {"operation": "EXPERIMENT_CURATION", "task_id": task}
    answer = {
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
    offered = client.offered_requests(answer)
    assert offered["curation"] == curation
    assert {name for name in offered if name.startswith("controls")} == {
        "controls:intents.0.flows.alpha",
        "controls:intents.0.flows.risk",
    }
    saved = tmp_path / "workspace.json"
    saved.write_text(json.dumps(answer), encoding="utf-8")
    args = argparse.Namespace(from_response=saved, action="curation", file=None)
    assert client._next_request(args) == curation


def test_a_printed_command_with_an_inline_document_sends_its_request(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """regression (V432, an outside review at 0c2b62a0): a request carrying an
    `experiment_document` prints as `--file '{JSON}'`, which put the object in the text field
    `experiment_yaml` and was refused `string_type`; the printed command, run as printed,
    sends the request it came from."""

    import shlex

    from alphalattice.interface.local_application import cli, client
    from alphalattice.interface.local_application.cli_contract import command

    sent: list[dict[str, Any]] = []

    class Host:
        """The running Host, recording what it was sent."""

        def __init__(self, workspace: Path, **kwargs: Any) -> None:
            self.workspace, self.goal = workspace, kwargs.get("goal")

        def exchange(self, document: Any) -> tuple[dict[str, Any], bytes]:
            sent.append(dict(document))
            return {"status": "PLANNED", "plan_hash": "c" * 64}, b"{}"

        def selected_url(self, *_args: Any) -> None:
            return None

        def navigation(self, *_args: Any) -> dict[str, str]:
            return {}

    monkeypatch.setattr(client, "LocalResearchClient", Host)
    monkeypatch.setenv("ALPHALATTICE_SHELL", "posix")
    request = {
        "operation": "EXPERIMENT_PLAN",
        "research_input_id": "input-1",
        "experiment_document": {"experiment": {"kind": "factor.screening-development"}},
    }
    printed = command(request, prefix=("alphalattice", "--workspace", str(tmp_path)))
    assert printed is not None and "--file" in printed, printed
    arguments = shlex.split(printed)[1:]
    assert cli.main(arguments, serve=lambda _: 99) == 0, capsys.readouterr().out
    assert sent == [request]


def test_a_choice_left_keeps_its_action() -> None:
    """regression (V442, an outside review at bbb1b9e8): a next request still missing a choice
    (`portfolio-draft` without its candidate) was answered CHOOSE_ONE_OFFERED_NEXT_REQUEST,
    whose words lead to another action; it keeps the action and names `--choices`."""

    from alphalattice.interface.local_application.cli_contract import client_refusal

    left = client_refusal("local_client.next_request_needs_choice:candidate_id")
    assert left.next_action == "GIVE_THE_CHOICE_WITH_THE_SAME_ACTION"
    assert "--choices" in left.detail and "same action" in left.detail
    assert client_refusal("local_client.next_request_required:a,b").next_action == (
        "CHOOSE_ONE_OFFERED_NEXT_REQUEST"
    )


def test_a_saved_promotion_waits_on_the_task_it_started(tmp_path: Path) -> None:
    """regression (V446, an outside review at 933f3786): a promotion that started its Alpha first
    answered the Portfolio Task as `task_id` and the Alpha as `follow_task_id`; `task show --from
    promotion.json --wait` took the Portfolio Task and ended while the Alpha ran. A saved answer's
    STATUS follows the Task it started, offered or not; its other reads keep its own Task."""

    from alphalattice.interface.local_application import client

    portfolio, alpha = str(uuid4()), str(uuid4())
    answer = {
        "status": "UPSTREAM_PROMOTION_ADMITTED",
        "task_id": portfolio,
        "follow_task_id": alpha,
    }
    saved = tmp_path / "promotion.json"
    saved.write_text(json.dumps(answer), encoding="utf-8")
    status = client.continuation("STATUS", saved, {}, frozenset({"task_id"}))
    assert status == {"operation": "STATUS", "task_id": alpha}
    read = client.continuation("EXPERIMENT_READBACK", saved, {}, frozenset({"task_id"}))
    assert read["task_id"] == portfolio
    offered = {**answer, "next_requests": {"task": {"operation": "STATUS", "task_id": alpha}}}
    saved.write_text(json.dumps(offered), encoding="utf-8")
    assert client.continuation("STATUS", saved, {}, frozenset({"task_id"}))["task_id"] == alpha


def test_an_export_saved_as_yaml_continues_as_its_json_does(tmp_path: Path) -> None:
    """regression (V447, an outside review at 933f3786): `--format yaml --output` saved an
    export's outer container, its Task and next requests buried in a `json` string, where JSON
    saved the inner answer; both save the one answer, which `--from` reads alike."""

    from alphalattice.interface.local_application import client

    inner = {
        "status": "EXPERIMENT_EXPORTED",
        "task_id": str(uuid4()),
        "next_requests": {"delivery": {"operation": "EXPERIMENT_DELIVERY_EXPORT"}},
    }
    body = {"json": json.dumps(inner), "yaml": "status: EXPERIMENT_EXPORTED\n", "html": "<p/>"}
    raw = json.dumps(body).encode("utf-8")
    client._save_output(tmp_path / "export.json", body, raw, "json")
    client._save_output(tmp_path / "export.yaml", body, raw, "yaml")
    as_json = client._response_document(tmp_path / "export.json")
    as_yaml = client._response_document(tmp_path / "export.yaml")
    assert as_yaml == as_json == inner


def test_restore_and_a_written_declaration_save_their_answers(tmp_path: Path, capsys: Any) -> None:
    """regression (V448, an outside review at 933f3786): `backup restore --output` and `model
    scaffold --save-declaration --output` saved nothing; answer and refusal are saved alike."""

    from alphalattice.interface.local_application import cli

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


_AWKWARD = 'it\'s a "quoted" | %PATH% $HOME ~ value ü 中'
_SAMPLES: dict[str, list[Any]] = {
    "string": [_AWKWARD, "@handle", "-leads-with-a-dash", "", "two\nlines"],
    "integer": [7, 0, -3],
    "number": [0.25, -1.5],
    "boolean": [True, False],
    "object": [{"text": _AWKWARD, "at": "@x", "n": 1, "on": True, "items": [1, "two"]}, {}],
    "array": [["one two", "it's", "@x", "-y"], []],
}
"""Each kind's values, among them the ones a shell or the CLI's parser could misread."""


def _limited(limits: dict[str, str], name: str) -> object | None:
    """A value inside a field's declared limit (`one of A, B`, `from 1 to 100`), if it has one."""
    words = limits.get(name, "")
    if (one_of := re.match(r"one of ([A-Z_]+)", words)) is not None:
        return one_of.group(1)
    if (ranged := re.match(r"(?:from|above) (\d+)", words)) is not None:
        return int(ranged.group(1)) + 1
    return None


def _offered_shapes() -> Iterator[dict[str, Any]]:
    """Every operation the CLI sends, its required fields given, and each field it takes set in
    turn to each kind the grammar declares for it."""
    from alphalattice.interface.local_application.cli_contract import choices, command_table

    table = command_table()
    plain = {"string": "plain", "integer": 2, "number": 1.5, "boolean": True}
    plain |= {"object": {"a": 1}, "array": ["a"]}
    for operation in sorted({op for ops in table["commands"].values() for op in ops}):
        contract = table["fields"][operation]
        groups = table["alternatives"].get(operation, [])
        base: dict[str, Any] = {"operation": operation}
        for name in [*contract["required"], *(groups[0] if groups else [])]:
            fixed = _limited(table["limits"], name)
            base[name] = plain[table["types"][name][0]] if fixed is None else fixed
        for name in contract["allowed"]:
            fixed = _limited(table["limits"], name)
            kinds = table["types"][name]
            for value in [fixed] if fixed is not None else [v for k in kinds for v in _SAMPLES[k]]:
                request = {**base, name: value}
                given = [g for g in groups if all(request.get(field) is not None for field in g)]
                if not choices(request) and len(given) <= 1:
                    yield request


def _same(value: Any) -> Any:
    """A request as JSON compares it: a whole number given as a number field is that number."""
    if isinstance(value, dict):
        return {key: _same(inner) for key, inner in value.items() if inner is not None}
    if isinstance(value, list):
        return [_same(inner) for inner in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def test_every_printed_command_reads_back_as_the_request_it_came_from() -> None:
    """regression (V449, the seam sweep): the printer and the CLI's reading are one contract.
    Every operation the CLI sends, each field it takes set to each kind the grammar declares
    (a text beginning `@`, which the CLI reads as a file, or `-`, which its parser reads as an
    option, among them), printed as a POSIX command, is read back as that request."""

    from alphalattice.interface.local_application.cli import request_of
    from alphalattice.interface.local_application.cli_contract import command

    misread = []
    read = 0
    for request in _offered_shapes():
        line = command(request, prefix=("alphalattice", "--workspace", "W"), quoting="posix")
        if line is None:
            continue
        read += 1
        back = request_of(shlex.split(line)[1:])
        if _same(back) != _same(request):
            misread.append((line, back))
    assert read > 1500 and misread == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell is a Windows shell")
def test_a_command_printed_for_powershell_sends_the_request_it_came_from(tmp_path: Path) -> None:
    """regression (V449): Windows PowerShell 5.1 hands a native program an argument's double
    quotes bare, so a printed `--file '{"a":1}'` (V432) arrived as `{a:1}`. A request whose
    command would carry a double quote is printed whole, as ASCII JSON piped to `request --file
    -`; each command PowerShell runs sends the request it came from."""

    from alphalattice.interface.local_application.cli import request_of
    from alphalattice.interface.local_application.cli_contract import command

    echo = tmp_path / "echo.py"
    echo.write_text(
        "import json, sys\nprint(json.dumps({'argv': sys.argv[1:], 'stdin': sys.stdin.read()}))\n",
        encoding="utf-8",
    )
    task = str(uuid4())
    requests: list[dict[str, Any]] = [
        {
            "operation": "EXPERIMENT_PLAN",
            "experiment_document": {"experiment": {"q": 'say "hi" | 50% ü 中', "n": 1}},
        },
        {"operation": "STATUS", "task_id": 'it\'s "quoted"'},
        {"operation": "TASKS", "history_cursor": "@handle", "history_limit": 3},
        {"operation": "ACTIVITY_LIST", "after": "-leads with a dash ü 中", "limit": 2},
        {"operation": "CANCEL", "task_id": task},
    ]
    prefix = (sys.executable, str(echo), "--workspace", str(tmp_path / "workspace"))
    lines = [command(request, prefix=prefix, quoting="powershell") for request in requests]
    assert all(line is not None for line in lines)
    assert sum(" | " in str(line) for line in lines) == 2
    script = tmp_path / "lines.ps1"
    script.write_text("\n".join(str(line) for line in lines) + "\n", encoding="utf-8-sig")
    ran = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=True,
    )
    received = [json.loads(row) for row in ran.stdout.splitlines() if row.startswith("{")]
    assert len(received) == len(requests)
    for request, got in zip(requests, received, strict=True):
        if got["stdin"].strip():
            assert got["argv"][-3:] == ["request", "--file", "-"]
            assert json.loads(got["stdin"]) == request
        else:
            assert request_of(got["argv"]) == request


def test_an_answer_saved_as_json_or_yaml_reads_back_alike(tmp_path: Path) -> None:
    """regression (V447's class, V449): `--from` reads an answer saved in either format as the
    one answer, whatever its values: texts YAML would read as another kind (`yes`, `null`, a
    date, `1e3`, `007`, `12:30`), markers, line breaks and any script, an export's inner answer."""

    from alphalattice.interface.local_application import client

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


def test_two_items_offering_one_action_keep_both() -> None:
    """regression (V528, the user's review at 04f512f9): one Portfolio Task's two CRO reviews
    each offered their own export, both named `export:<task_id>`, and the collector kept the
    first alone, so `evidence export --from history.json` took one without a word. A history row
    is named by its entry; two items that still share a name each keep theirs by their place."""

    from alphalattice.interface.local_application import client

    task = str(uuid4())

    def row(review: str) -> dict[str, object]:
        export = {"operation": "EVIDENCE_CRO_EXPORT", "review_publication_hash": review}
        return {
            "entry_id": f"review:{review}",
            "task_id": task,
            "next_requests": {"task": {"operation": "STATUS", "task_id": task}, "export": export},
        }

    old, new = "a" * 64, "b" * 64
    offered = client.offered_requests({"entries": [row(old), row(new)]})
    assert offered[f"export:review:{old}"]["review_publication_hash"] == old
    assert offered[f"export:review:{new}"]["review_publication_hash"] == new
    assert offered["task"] == {"operation": "STATUS", "task_id": task}
    # Two items under one identifier and no entry: both kept, each by its place.
    alike = [
        {"task_id": task, "next_requests": {"export": {"operation": "EXPORT", "n": i}}}
        for i in (1, 2)
    ]
    kept = client.offered_requests({"items": alike})
    assert sorted(kept) == ["export:items.0", "export:items.1"], kept


def test_an_offered_request_is_one_the_host_accepts_as_it_stands() -> None:
    """regression (V449, the seam sweep): an Evidence answer offered a unit's id as a request
    with no operation, and a book's packet requests carried their window, fields no packet
    request takes. Every answer's offered requests are held to `request_problem` where every
    test's operations pass (V410's check); here, each problem it names."""

    from alphalattice.interface.local_application.cli_contract import request_problem

    task = str(uuid4())
    assert request_problem({"operation": "STATUS", "task_id": task}) is None
    assert request_problem({"operation": "STATUS", "task_id": None}) is None
    assert request_problem({"operation": "STATUS"}) is None
    assert request_problem({"evidence_unit_id": "u01"}) == "operation None is none the Host answers"
    window = {"operation": "EVIDENCE_PACKET", "task_id": task, "evidence_expires_at": "2026-08-13"}
    assert request_problem(window) == "evidence_expires_at is no field of EVIDENCE_PACKET"
    stand_in = {"operation": "EXPERIMENT_PORTFOLIO_DRAFT", "candidate_id": "<candidate_id>"}
    assert request_problem(stand_in) == "candidate_id holds the stand-in <candidate_id>"
    numbered = {"operation": "STATUS", "task_id": 5}
    assert request_problem(numbered) == "task_id is integer, not string"


def test_every_task_state_has_one_cli_outcome() -> None:
    """regression (V440's class, V449): a goal waiter ended on the states it listed and missed
    BLOCKED. The CLI's one table classes every state a Task can be in, so a state added to the
    lifecycle is classed before a waiter meets it: only a Task that succeeded is done."""

    from alphalattice.control.task_control.contracts import TaskLifecycle
    from alphalattice.interface.local_application.cli_contract import outcome_of

    outcomes = {state: outcome_of({"lifecycle": state.value}) for state in TaskLifecycle}
    assert {state for state, outcome in outcomes.items() if outcome == "OK"} == {
        TaskLifecycle.SUCCEEDED
    }
    assert set(outcomes.values()) == {"OK", "PENDING", "REFUSED"}


def test_a_command_from_a_saved_answer_acts_on_what_the_file_names(tmp_path: Path) -> None:
    """regression (V449, an outside review: context precedence): `goal note --from goal-a.json`
    sent no goal when the saved answer offered no note, and the Host took the session's own
    goal, another one. A command continued from a saved answer takes the goal the file names;
    a file naming none is refused, and a flag naming another is refused, never preferred. An
    opened goal's id stays its own."""

    from alphalattice.interface.local_application import client

    goal = str(uuid4())
    saved = tmp_path / "goal-a.json"
    saved.write_text(json.dumps({"status": "OPEN", "goal_id": goal}), encoding="utf-8")
    note = {"goal_statement": "A finding.", "change_reason": "Noted."}
    allowed = frozenset({"goal_id", "goal_hash", "goal_statement", "change_reason"})
    assert client.continuation("GOAL_NOTE", saved, note, allowed)["goal_id"] == goal
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"status": "ADMITTED", "task_id": str(uuid4())}), encoding="utf-8")
    with pytest.raises(client.LocalResearchClientError, match="response_reference_missing:goal_id"):
        client.continuation("GOAL_NOTE", task, note, allowed)
    other = {**note, "goal_id": str(uuid4())}
    with pytest.raises(client.LocalResearchClientError, match="bound_reference_override:goal_id"):
        client.continuation("GOAL_NOTE", saved, other, allowed)
    with pytest.raises(client.LocalResearchClientError, match="bound_reference_override:task_id"):
        client.continuation("STATUS", task, {"task_id": str(uuid4())}, frozenset({"task_id"}))
    opened = client.continuation(
        "GOAL_OPEN", saved, {"goal_declaration": {}}, frozenset({"goal_id", "goal_declaration"})
    )
    assert "goal_id" not in opened


def test_a_refusal_leaves_with_words_and_a_way_on() -> None:
    """regression (V449, OP4): of the 74 refusal codes the operation tests reach, 65 answered a
    code alone and 10 words without a way on. A refusal leaves the Host with its owner's words and
    way on, else its code's own from one table: returned by an operation, and raised to the web
    layer, which answered `{"refused": code}` alone."""

    from alphalattice.interface.local_application.cli_contract import (
        refusal_problem,
        worded_refusal,
    )
    from alphalattice.interface.local_application.web import AdmittedRequest, LocalWebApplication

    bare = {"status": "REFUSED", "failure_code": "goal.not_found"}
    assert refusal_problem(bare) == "refusal goal.not_found lacks words and a way on"
    worded = worded_refusal(bare)
    assert worded["detail"].startswith("No goal in this workspace has that id")
    assert worded["next_action"] == "COPY_THE_GOAL_ID_FROM_THE_GOAL_LIST"
    assert refusal_problem(worded) is None
    panel_code = "workspace_data_update.panel_binding_mismatch"
    panel_bare = {"status": "REFUSED", "failure_code": panel_code, "detail": None}
    panel_words = worded_refusal(panel_bare)
    assert panel_words["status"] == "REFUSED" and panel_words["failure_code"] == panel_code
    assert panel_words["next_action"] == "DATA_UPDATE_READBACK"
    assert refusal_problem(panel_words) is None
    panel_request = {"show": {"operation": "DATA_UPDATE_READBACK"}}
    panel = worded_refusal(
        {
            **panel_bare,
            "next_requests": panel_request,
        }
    )
    assert panel["status"] == "REFUSED" and panel["failure_code"] == panel_code
    assert panel["detail"] == (
        "The current Panel does not match the data binding required for this update. "
        "Read the data update's current record and the input state it reports; resolve "
        "that condition before planning the update again."
    )
    assert panel["detail"] == panel_words["detail"]
    assert panel["next_requests"] == panel_request
    assert refusal_problem(panel) is None
    for field in ("experiment.output_workspace", "experiment.universe_handle"):
        code = "factor_research.handoff_authority_field_mismatch:" + field
        handoff = worded_refusal({"status": "REFUSED", "failure_code": code})
        assert handoff["failure_code"] == code
        assert handoff["detail"] == (
            f"A Factor handoff fixes `{field}` of the Alpha study it opens: its kind, data, "
            "workspaces and publication intent come from the Factor study, and its universe is "
            "that study's or a declared sample of it. Leave the field out, or write the value "
            "`expected` gives, and plan again."
        )
        assert handoff["next_action"] == "EDIT_DECLARATION_AND_PLAN"
        assert refusal_problem(handoff) is None
    vague = {**bare, "detail": "A goal was not found.", "next_action": "REPLAN"}
    assert refusal_problem(vague) == "refusal goal.not_found lacks a usable way on"
    owned = {**bare, "detail": "Its owner's words.", "next_requests": {"goals": {}}}
    assert worded_refusal(owned) == owned
    assert refusal_problem(owned) == "refusal goal.not_found lacks a usable way on"
    offered = {
        **vague,
        "next_requests": {"task": {"operation": "STATUS", "task_id": str(uuid4())}},
    }
    assert refusal_problem(offered) is None
    unbound = {**vague, "next_requests": {"task": {"operation": "STATUS"}}}
    assert refusal_problem(unbound) == "refusal goal.not_found lacks a usable way on"
    page = "feature_extension.extensions_page_out_of_range:2 of 1"
    assert "(2 of 1)" in worded_refusal({"status": "REFUSED", "failure_code": page})["detail"]
    web = LocalWebApplication()

    @web.route("POST", "/api/raises")
    def raises(_query: Any, _payload: Any) -> object:
        raise ValueError("goal.not_found")

    admitted = AdmittedRequest(web.routes[("POST", "/api/raises")], None, {}, 0, False)
    raised = json.loads(web.dispatch(admitted, b"").body)
    assert raised["refused"] == raised["failure_code"] == "goal.not_found"
    assert (raised["detail"], raised["next_action"]) == (worded["detail"], worded["next_action"])


def test_each_registered_collection_read_checks_item_refusal_routes() -> None:
    """regression (V633, TE12): a partial collection can keep its healthy rows while one
    refused row names its own words and a usable route. List operations and their row arrays
    come from the public read and answer registries, not a second hand-maintained operation list."""
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
                # Activity joins Task records by their own keys; keep both healthy and refused
                # records addressable through that public map.
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

    # A blocked Task is a readable record of failed work, not a refusal to read that row. Its
    # failure code must not turn a healthy collection into a partial refusal.
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


def test_a_read_names_what_it_read_and_reads_again_from_itself() -> None:
    """regression (V449, an outside review at 4bfd3adc, its fourth item, and the sweep): `feature
    review --from review.json --section contract` sent neither selector the answer carried, and
    22 reads' answers did not read again from themselves; and (V454, an outside review at
    35d9a3a7) `study show --from` it read the latest day, not the one asked. The CLI's copy of a
    read's answer keeps the request it sent, its whole selection (`read_request`), the Host's
    answer staying its owner's (an export's record, a measured packet and a self-hash are
    sealed); `--from` reads the same read again from it, every reference, target and choice, a
    flag giving one winning; the Task stays a target."""

    from alphalattice.interface.local_application import client
    from alphalattice.interface.local_application.cli_contract import command_table, named_read

    table = command_table()
    plan, task = "a" * 64, str(uuid4())
    report = named_read({"operation": "REPORT", "result_hash": plan}, {"status": "OK"})
    assert report["read_request"] == {"operation": "REPORT", "result_hash": plan}
    assert named_read({"operation": "RUN", "spec": {}}, {"status": "OK"}) == {"status": "OK"}
    refusal = {"status": "REFUSED", "failure_code": "goal.not_found"}
    assert named_read({"operation": "REPORT", "result_hash": plan}, refusal) == refusal
    # A read that gave no field keeps its operation alone, its defaults its selection (V571).
    bare = {"status": "OK"}
    assert named_read({"operation": "REPORT"}, bare) == {
        **bare,
        "read_request": {"operation": "REPORT"},
    }
    finding = named_read(
        {"operation": "CRO_REVIEW_FINDING", "finding_handle": "f-1", "result_hash": None},
        {"status": "FINDING"},
    )
    kept = {"operation": "CRO_REVIEW_FINDING", "finding_handle": "f-1"}
    assert finding == {"status": "FINDING", "read_request": kept}

    def allowed(operation: str) -> frozenset[str]:
        return frozenset(table["fields"][operation]["allowed"])

    review = {"status": "FEATURE_REVIEW", "feature_plan_hash": plan, "factor_id": "f1"}
    again = client.continued("FEATURE_REVIEW", review, {}, allowed("FEATURE_REVIEW"))
    assert (again["feature_plan_hash"], again["feature_factor_id"]) == (plan, "f1")
    chosen = {"feature_factor_id": "f2"}
    flagged = client.continued("FEATURE_REVIEW", review, chosen, allowed("FEATURE_REVIEW"))
    assert flagged["feature_factor_id"] == "f2"
    packet = client.continued(
        "EVIDENCE_PACKET",
        {"status": "PACKET", "prepared_task_id": task, "prepared_unit_id": "u02"},
        {},
        allowed("EVIDENCE_PACKET"),
    )
    assert (packet["task_id"], packet["evidence_unit_id"]) == (task, "u02")
    again = client.continued("CRO_REVIEW_FINDING", finding, {}, allowed("CRO_REVIEW_FINDING"))
    assert again["finding_handle"] == "f-1"
    # The day a read was asked for reads back with it, and a flag naming another wins (V454).
    day = {"operation": "EXPERIMENT_READBACK", "task_id": task, "portfolio_session": "2026-09-01"}
    shown = named_read(day, {"status": "EXPERIMENT_PUBLISHED", "task_id": task})
    assert client.continued("EXPERIMENT_READBACK", shown, {}, allowed("EXPERIMENT_READBACK")) == day
    # A saved revision reads again as itself, not as the latest the answer offers next, nor
    # refused between two next reads of it (V460, the review at b39e55f2).
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
    shown_again = client.continued("GOAL_SHOW", revision, {}, allowed("GOAL_SHOW"))
    assert (shown_again["goal_id"], shown_again["goal_hash"]) == (goal, first)
    # A package the answer names reads back as well (V454, the review at 118f6378).
    package = {"operation": "CONTROLS", "strategy_package_id": "pkg-b"}
    controls = named_read(package, {"status": "CONTROLS", "strategy_package_id": "pkg-b"})
    assert client.continued("CONTROLS", controls, {}, allowed("CONTROLS")) == package
    later = {"portfolio_session": "2026-09-02"}
    moved = client.continued("EXPERIMENT_READBACK", shown, later, allowed("EXPERIMENT_READBACK"))
    assert moved["portfolio_session"] == "2026-09-02"


def test_every_read_reads_its_whole_selection_again_from_its_answer() -> None:
    """property (V454, TE12): for every read the Host answers and every field it takes as one
    value, the CLI's copy of an answer keeps it and `--from` reads it back: a day, a page and a
    target as well as a reference, so no choice a read was asked with falls to a default."""

    from alphalattice.interface.local_application import client
    from alphalattice.interface.local_application.cli_contract import command_table, named_read

    table = command_table()
    samples: dict[str, object] = {"string": "s-1", "integer": 3, "number": 1.5, "boolean": True}
    checked = 0
    for operation in sorted(table["reads"]):
        allowed = frozenset(table["fields"][operation]["allowed"])
        targets = {name: str(uuid4()) for name in ("task_id", "goal_id") if name in allowed}
        for name in sorted(allowed - set(targets)):
            kinds = [kind for kind in table["types"].get(name, ()) if kind in samples]
            if not kinds:
                continue
            request = {"operation": operation, **targets, name: samples[kinds[0]]}
            echoed = {key: value for key, value in request.items() if key != "operation"}
            # The answer naming none of it, or all of it as given (V454, the review at 118f6378:
            # a package the answer named fell to the default).
            # The same operation offered as a next request naming another value is not
            # what a saved read reads again (V460).
            other = {"next_requests": {"next": {**request, name: samples[kinds[0]]}}}
            other["next_requests"]["next"].update(dict.fromkeys(targets, str(uuid4())))
            for body in (
                {"status": "OK"},
                {"status": "OK", **echoed},
                {"status": "OK", **other},
            ):
                again = client.continued(operation, named_read(request, body), {}, allowed)
                assert {key: again.get(key) for key in request} == request, (operation, name)
            checked += 1
    assert checked > 100, checked


def test_every_offered_edge_keeps_its_bound_references_or_names_its_choice() -> None:
    """contract (V523): the operation registry generates the whole next-request class,
    including incomplete templates, rather than a list of reported continuation defects."""
    from alphalattice.interface.local_application.answers import continuation_problem
    from alphalattice.interface.local_application.cli_contract import command_table

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


def test_an_unrelated_saved_answer_never_reads_default_controls() -> None:
    """regression (V515): a continuation without an input refuses before reading an anchor."""
    from alphalattice.interface.local_application.cli_contract import client_refusal, command_table
    from alphalattice.interface.local_application.client import LocalResearchClientError, continued

    allowed = frozenset(command_table()["fields"]["EXPERIMENT_CONTROLS"]["allowed"])
    with pytest.raises(
        LocalResearchClientError, match=r"local_client\.response_reference_missing"
    ) as caught:
        continued("EXPERIMENT_CONTROLS", {"status": "SUCCEEDED", "publication": {}}, {}, allowed)
    assert client_refusal(str(caught.value)).detail


def test_every_refused_answer_still_holds_its_offered_request_contract() -> None:
    """contract (V523): a refusal's words do not excuse an invalid bound reference;
    each registered answer and a projection without a model holds the same boundary."""
    from alphalattice.interface.local_application.answers import answer_problem
    from alphalattice.interface.local_application.cli_contract import command_table

    table = command_table()
    read_operation = next(
        operation
        for operation, fields in table["fields"].items()
        if fields["required"] == ["task_id"]
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


def test_a_task_state_is_read_wherever_an_answer_names_it() -> None:
    """regression (V455, an outside review at 35d9a3a7): `data-update show --wait` answered OK
    while its Task ran, since the answer names the Task's state `task_lifecycle` beside its own
    `status`. The CLI reads a Task's state wherever an answer names it, and every answer field
    that names a lifecycle is one it reads or one declared not to be a Task's (TE12)."""

    from alphalattice.interface.local_application import answers
    from alphalattice.interface.local_application.cli_contract import (
        TASK_STATE_FIELDS,
        outcome_of,
        task_state,
    )

    running = {"status": "NO_UPDATE_PUBLICATION", "task_lifecycle": "RUNNING"}
    assert task_state(running) == "RUNNING" and outcome_of(running) == "PENDING"
    not_a_task_state = {
        "lifecycle_research": "a study's kind, a model lifecycle's research",
        "model_lifecycle": "a model's place in its lifecycle",
        "component_lifecycles": "each component's model lifecycle",
    }
    rows = json.loads(Path(answers.__file__).with_name("answers.json").read_text("utf-8"))
    named = {
        str(field["name"])
        for row in rows.values()
        for field in row["fields"]
        if "lifecycle" in str(field["name"])
    }
    assert named - set(TASK_STATE_FIELDS) == set(not_a_task_state)


def test_a_reuse_names_what_holds_its_work() -> None:
    """regression (V449, an outside review at 4bfd3adc, its third item): a data update's
    `REUSED_EXACT` answered `task_id: null` and its receipt, so `data-update show --from` it was
    refused. A reuse, an admission or a resubmission names its Task, or the result or record it
    is, by an id or a hash at its top or a request it offers naming one; the data update's names
    the Task that sealed the receipt it reused."""

    from alphalattice.interface.local_application.cli_contract import exit_problem

    bare = {"status": "REUSED_EXACT", "task_id": None, "receipt": {"plan_hash": "a" * 64}}
    assert exit_problem(bare) == "REUSED_EXACT names no Task, result or record that holds its work"
    assert exit_problem({**bare, "publication_task_id": str(uuid4())}) is None
    assert exit_problem({"status": "REUSED_EXACT", "result_hash": "b" * 64}) is None
    assert exit_problem({"status": "REUSED_EXACT", "observation_id": "o-1"}) is None
    sealed = {"operation": "EXPERIMENT_FOUNDATION_EXPORT", "foundation_admission_hash": "c" * 64}
    assert exit_problem({**bare, "next_requests": {"export": sealed}}) is None
    assert exit_problem({**bare, "attributed_goal_id": str(uuid4())}) is not None
    assert exit_problem({"status": "SUCCEEDED"}) is None


def test_a_bundle_submit_command_keeps_the_goal_it_was_prepared_under(tmp_path: Path) -> None:
    """regression (V449, an outside review at 4bfd3adc, its second item): a bundle prepared
    with `--goal G` printed a submit command without it, so the submission went unrecorded under
    G. The submit command is printed from the caller's entry, as every printed command is."""

    import shlex

    from alphalattice.interface.local_application import client
    from alphalattice.interface.local_application.cli_contract import entry

    workspace = tmp_path / "workspace"
    prefix = entry(workspace, ("--lang", "zh", "--goal", "goal-g"))
    body = {"status": "AGENT_BUNDLE_READY", "files": [{"name": "README.md", "text": "Read.\n"}]}
    bundle = client._write_bundle(tmp_path / "bundle", body, prefix=prefix)
    assert bundle["submit_arguments"][:6] == [*prefix[1:]]
    assert bundle["submit_arguments"][6:8] == ["bundle", "submit"]
    if "submit_command" in bundle:
        words = shlex.split(bundle["submit_command"], posix=False)
        assert "--goal" in words and "goal-g" in [w.strip("'\"") for w in words]


def test_a_document_that_does_not_read_says_so_in_its_own_terms() -> None:
    """regression (V449, AX15's finding): a YAML document with a syntax error was refused as
    "cannot be read as this operation's JSON" and sent to `schema show`; it says YAML or JSON
    and to correct the syntax where reading stopped."""

    from alphalattice.interface.local_application.cli_contract import client_refusal

    refusal = client_refusal("local_client.document_unreadable")
    assert "YAML or JSON" in refusal.detail
    assert refusal.next_action == "CORRECT_THE_DOCUMENT_WHERE_READING_STOPPED"
    assert "JSON;" not in client_refusal("local_client.document_missing").detail


def test_an_untyped_failure_is_told_from_a_product_code() -> None:
    """regression (V449, AX15's review IndexError): a crash reached the agent as a bare
    fingerprint. A failure without a product code is the fingerprint `public_failure` gives
    (`fallback:<exception class>:<digest>`), which the answer check refuses in every test that
    does not cause one on purpose, and whose words and way on the Host answers."""

    from alphalattice.interface.local_application.cli_contract import refusal_words
    from alphalattice.interface.local_application.failure_codes import (
        public_failure,
        untyped_failure,
    )

    crash = public_failure(
        IndexError("single positional indexer is out-of-bounds"), "local_web.handler_failed"
    )
    assert crash == "local_web.handler_failed:IndexError:503ea14e" and untyped_failure(crash)
    assert not untyped_failure("goal.not_found")
    assert not untyped_failure("feature_extension.extensions_page_out_of_range:2 of 1")
    words = refusal_words(crash)
    assert "503ea14e" not in words["next_action"] and "IndexError:503ea14e" in words["detail"]


def test_an_explicit_owner_code_survives_validation_without_its_private_detail() -> None:
    """regression (V679): the shared mapper reads structured codes, including validator
    causes, before exception prose; unsafe diagnostic suffixes keep only their named rule.
    Numeric transport status, paths, ordinary text and multi-argument faults are not codes.
    """
    from urllib.error import HTTPError

    from pydantic import BaseModel, ValidationError, field_validator

    from alphalattice.control.data_platform.maintenance.reconciliation import (
        MaintenanceReconciliationError,
    )
    from alphalattice.control.product_host.storage.inventory import StorageInventoryError
    from alphalattice.interface.local_application.failure_codes import (
        located_failure,
        owner_failure_code,
        public_failure,
        setup_failure,
        untyped_failure,
    )

    code = "storage.managed_capacity_exceeded"
    marker = "private synthetic payload /never/serve with token=not-a-secret"
    error = StorageInventoryError(code, marker)
    assert owner_failure_code(error) == public_failure(error, "local_web.handler_failed") == code
    assert setup_failure(error)["causes"] == [
        {"kind": "PRODUCT", "code": code, "failure_code": code}
    ]

    class Document(BaseModel):
        selected: int

        @field_validator("selected")
        @classmethod
        def admitted(cls, selected: int) -> int:
            raise MaintenanceReconciliationError(marker, failure_code=code)

    with pytest.raises(ValidationError) as seen:
        Document(selected=1)
    located = located_failure(seen.value, "local_application.request_refused")
    assert located["fields"] == [["selected"]]
    assert located["reasons"] == {"selected": code}
    assert marker not in json.dumps(located)
    assert public_failure(seen.value, "local_application.request_refused") == (
        f"local_application.request_refused:selected={code}"
    )
    for head in (
        "task_control.queue_full",
        "model_store.download_interrupted",
        "alternative_evidence.table_view_correspondence_unproved",
        "alternative_evidence.topic_coverage_absent",
    ):
        cause = RuntimeError(f"{head}: {marker}; unsafe diagnostic")
        assert (
            owner_failure_code(cause) == public_failure(cause, "local_web.handler_failed") == head
        )
    for fault in (
        RuntimeError("ordinary"),
        RuntimeError(marker),
        RuntimeError(code, marker),
        StorageInventoryError(marker, marker),
        HTTPError(None, 503, marker, None, None),
    ):
        assert owner_failure_code(fault) is None
        shown = public_failure(fault, "local_web.handler_failed")
        assert untyped_failure(shown) and marker not in shown


@pytest.mark.parametrize(
    ("operation", "field", "document"),
    [
        ("PLAN", "spec", {"strategy_package_id": "SYNTHETIC_SINGLE_BOOK", "top_k": 2}),
        (
            "EXPERIMENT_PLAN",
            "experiment_yaml",
            "experiment:\n  kind: factor.screening-development\n",
        ),
    ],
)
def test_an_offered_request_reads_its_operation_document_from_file(
    live, tmp_path: Path, monkeypatch, capsys, operation, field, document
) -> None:
    """contract (V538): generic --from --file loads the selected operation's document by
    its registry type, preserving text declarations and the offer's bound package."""
    from alphalattice.interface.local_application import cli, client

    saved, declaration = tmp_path / "answer.json", tmp_path / "declaration.yaml"
    bound = {"strategy_package_id": "SYNTHETIC_SINGLE_BOOK"} if field == "spec" else None
    offered = {"operation": operation, **({field: bound} if bound else {})}
    saved.write_text(json.dumps({"next_requests": {"preview": offered}}), encoding="utf-8")
    declaration.write_text(
        json.dumps(document) if isinstance(document, dict) else document,
        encoding="utf-8",
        newline="\n",
    )
    seen = []

    def exchange(_client, request):
        seen.append(request)
        body = {"status": "OK"}
        return body, json.dumps(body).encode("utf-8")

    monkeypatch.setattr(client.LocalResearchClient, "exchange", exchange)
    assert (
        cli.main(
            [
                "--workspace",
                str(live.workspace),
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
    capsys.readouterr()
    assert seen == [{"operation": operation, field: document}]


@pytest.mark.parametrize("case", ["bound_package", "no_document", "twice", "stdin"])
def test_an_offered_declaration_refuses_an_override_or_ambiguous_source_by_name(
    live, tmp_path: Path, case
) -> None:
    """contract (V538): the generic declaration path preserves bound selectors, cannot fill
    a non-document operation, and refuses duplicate fields or multiple stdin readers."""
    saved, declaration, choices = (
        tmp_path / name for name in ("answer.json", "book.yaml", "choices.json")
    )
    offered = {"operation": "PLAN", "spec": {"strategy_package_id": "SYNTHETIC_SINGLE_BOOK"}}
    document = {"strategy_package_id": "OTHER_PACKAGE"} if case == "bound_package" else {}
    if case == "no_document":
        offered = {"operation": "TASKS"}
    saved.write_text(json.dumps({"next_requests": {"preview": offered}}), encoding="utf-8")
    declaration.write_text(json.dumps(document), encoding="utf-8")
    arguments = ["request", "--from", str(saved), "--action", "preview", "--file", str(declaration)]
    expected = {
        "bound_package": "local_client.bound_reference_override:spec.strategy_package_id",
        "no_document": "local_client.operation_has_no_declaration_file:TASKS",
        "twice": "local_client.field_given_twice:spec",
        "stdin": "local_client.document_multiple_stdin_sources",
    }[case]
    if case == "twice":
        choices.write_text(json.dumps({"spec": {}}), encoding="utf-8")
        arguments += ["--choices", str(choices)]
    if case == "stdin":
        arguments = ["request", "--from", "-", "--action", "preview", "--file", "-"]
    code, answer, _ = _cli(live.workspace, *arguments)
    assert (code, answer["failure_code"]) == (1, expected), answer
    assert answer["detail"] and answer["next_action"]


def test_a_workspace_without_strategy_installation_words_that_absence(tmp_path: Path) -> None:
    """contract (V538): no installed strategy is a different refusal from an unselected
    installed book; the CLI's controls, preview and run doors all word the absence."""
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceManifest,
        publish_research_workspace_manifest,
    )

    workspace = tmp_path / "empty"
    publish_research_workspace_manifest(
        workspace, ResearchWorkspaceManifest.research_only("empty-book")
    )
    file = tmp_path / "book.json"
    file.write_text("{}", encoding="utf-8")
    with LocalPortfolioWebSession.from_workspace(workspace):
        for action in ("controls", "preview", "run"):
            code, answer, _ = _cli(
                workspace,
                "strategy-book",
                action,
                *([] if action == "controls" else ["--file", str(file)]),
            )
            assert (code, answer["failure_code"]) == (
                2,
                "research_workspace.strategy_not_installed",
            )
            assert "no installed strategy package" in answer["detail"]
            assert answer["data"]["next_action"] == "PREPARE_AND_INSTALL_A_RESEARCH_STRATEGY"
            assert answer["next_requests"]["strategies"] == {
                "operation": "RESEARCH_STRATEGY_CONTROLS"
            }


@pytest.mark.parametrize(
    "code",
    [
        "strategy_book.strategy_package_required",
        "local_application.strategy_package_not_installed:NOT_AN_INSTALLED_PACKAGE",
        "research_workspace.strategy_not_installed",
    ],
)
def test_every_book_selector_refusal_has_door_words_and_owner_context(code) -> None:
    """contract (V538): all selector refusals carry detail and next_action at the door, and
    the owner's available package context offers controls for each admitted choice."""
    from alphalattice.control.product_host.composition.plain_refusals import explain
    from alphalattice.interface.local_application.cli_contract import refusal_words, request_problem

    words = refusal_words(code)
    assert set(words) == {"detail", "next_action"} and all(words.values())
    packages = ("INSTALLED_A", "INSTALLED_B")
    answer = explain(code, installed_packages=packages)
    assert answer["detail"] and answer["next_action"]
    for request in answer["next_requests"].values():
        assert request_problem(request) is None
    if code != "research_workspace.strategy_not_installed":
        assert answer["next_requests"] == {
            f"controls:{package}": {"operation": "CONTROLS", "strategy_package_id": package}
            for package in packages
        }
        assert all(package in answer["detail"] for package in packages)


def test_the_answer_boundary_holds_every_registered_editable_declaration() -> None:
    """contract (V538/TE12): the observed answer boundary accepts each registered declaration
    kind and its named unavailable states; an unrecognised omission fails for every producer."""
    from alphalattice.interface.local_application.answers import declaration_problem
    from alphalattice.interface.local_application.cli_contract import declaration_contracts

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


def test_a_command_without_an_editable_answer_refuses_save_declaration(
    live, tmp_path: Path
) -> None:
    """contract (V538): a non-declaration command rejects the option in its grammar; generic
    request, whose operation is selected at run time, names declaration_unavailable instead."""
    file = tmp_path / "none.yaml"
    code, refusal, _ = _cli(live.workspace, "task", "list", "--save-declaration", str(file))
    assert (code, refusal["failure_code"]) == (1, "local_client.usage_invalid")
    assert not file.exists()
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"operation": "TASKS"}), encoding="utf-8")
    code, answer, _ = _cli(
        live.workspace, "request", "--file", str(request), "--save-declaration", str(file)
    )
    assert code == 0 and answer["outcome"] == "OK"  # the read itself succeeded
    assert answer["local_failure"]["failure_code"] == "local_client.declaration_unavailable"
    assert answer["local_failure"]["detail"] and answer["local_failure"]["next_action"]
    assert not file.exists()


@pytest.mark.parametrize(
    ("command", "operation"),
    [
        (["study", "show"], "EXPERIMENT_READBACK"),
        (["feature", "show", "--task"], "FEATURE_CATALOG_BUILD_READBACK"),
    ],
)
def test_shared_commands_refuse_declaration_saving_on_execution_readbacks_before_sending(
    live, tmp_path: Path, monkeypatch, capsys, command, operation
) -> None:
    """contract (V538/TE12): the plan branch of a shared read saves its declaration; the
    execution branch names its deliberate exception before sending an owner any request."""
    from alphalattice.interface.local_application import cli, client
    from alphalattice.interface.local_application.cli_contract import (
        DECLARATION_OPERATION_EXCEPTIONS,
    )

    def exchange(*_args, **_kwargs):
        raise AssertionError("a non-declaration execution read must not be sent")

    monkeypatch.setattr(client.LocalResearchClient, "exchange", exchange)
    file = tmp_path / "none.yaml"
    code = cli.main(
        [
            "--workspace",
            str(live.workspace),
            *command,
            str(uuid4()),
            "--save-declaration",
            str(file),
        ],
        serve=lambda _: 99,
    )
    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (code, answer["failure_code"]) == (
        1,
        "local_client.operation_has_no_editable_declaration:" + operation,
    )
    assert answer["detail"] and answer["next_action"]
    assert DECLARATION_OPERATION_EXCEPTIONS[operation] and not file.exists()


def test_a_continued_page_keeps_the_ids_its_list_watches() -> None:
    """regression (V556, the user's review at de555b07): `activity list --watch '["<T>"]'
    --limit 1 --output p.json` then `activity list --from p.json --after <cursor>` lost the
    watch, since the read kept scalar fields alone. The list of ids is kept with the limit, and
    the next page's cursor is the one given."""

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


def test_a_score_plan_plans_again_on_its_own_day_and_component() -> None:
    """regression (V562, the user's review at de555b07): `score plan --from p.json` kept the
    package and took the newest day. A score plan's answer offers its re-plan with its day and
    component bound, so the continuation keeps them; another day is another plan, which the
    bound answer refuses in words and a plan without `--from` makes (newest by default)."""

    from alphalattice.interface.local_application.client import continued

    answer = {
        "status": "PLANNED",
        "score_plan_hash": "a" * 64,
        "strategy_package_id": "PKG",
        "formation_session": "2026-09-29",
        "next_requests": {
            "replan": {
                "operation": "STRATEGY_SCORE_PLAN",
                "strategy_package_id": "PKG",
                "formation_session": "2026-09-29",
                "component_id": "G2",
            }
        },
    }
    allowed = frozenset({"component_id", "formation_session", "strategy_package_id"})
    again = continued("STRATEGY_SCORE_PLAN", answer, {}, allowed)
    assert (again["formation_session"], again["component_id"]) == ("2026-09-29", "G2")
    import pytest

    from alphalattice.interface.local_application.cli_contract import client_refusal
    from alphalattice.interface.local_application.client import LocalResearchClientError

    with pytest.raises(LocalResearchClientError, match="bound_reference_override") as bound:
        continued("STRATEGY_SCORE_PLAN", answer, {"formation_session": "2026-09-30"}, allowed)
    assert client_refusal(str(bound.value)).detail


def test_a_door_sentence_with_two_subjects_reads_in_chinese_with_both() -> None:
    """regression (V566, the UI line's BT note): the CLI's Chinese templates captured only the
    first `{subject}`, so a sentence with two (the uninstalled package and the installed ones)
    read in English under `--lang zh` though the page read it in Chinese. Every slot is a group
    of its own and is filled in its turn."""

    from alphalattice.interface.local_application import cli_contract

    keys = [key for key in cli_contract._zh() if key.count("{subject}") >= 2]
    assert keys, "no door sentence takes two subjects"
    token = cli_contract.ANSWER_LANGUAGE.set("zh")
    try:
        for key in keys:
            subjects = [
                "PKG_ONE",
                "PKG_TWO",
                *(f"PKG_{i}" for i in range(3, key.count("{subject}") + 1)),
            ]
            filled = key
            for subject in subjects:
                filled = filled.replace("{subject}", subject, 1)
            chinese = cli_contract.worded(filled)
            assert chinese != filled, key
            assert all(subject in str(chinese) for subject in subjects), chinese
            assert "{subject}" not in str(chinese), chinese
    finally:
        cli_contract.ANSWER_LANGUAGE.reset(token)


def test_a_saved_default_read_reads_its_own_page_again(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """regression (V571, an outside review at 9b3181e7): `feature list --output p.json` kept no
    `read_request`, since its read gave no field, so `feature list --from p.json` took the next
    page the answer offers for the same operation as the same read and answered the second
    page's first item, `0/OK`. A read that gives no field is kept as its operation alone and
    reads its own first page again; the next page stays a request asked for by name."""

    from alphalattice.interface.local_application import cli, client

    sent: list[dict[str, Any]] = []
    first = {
        "status": "AVAILABLE",
        "factors": [{"factor_id": "first-page"}],
        "next_requests": {"next": {"operation": "FEATURE_EXTENSIONS", "extensions_page": 2}},
    }

    class Host:
        """The running Host, answering the first page and recording what it was sent."""

        def __init__(self, workspace: Path, **kwargs: Any) -> None:
            self.workspace, self.goal = workspace, kwargs.get("goal")

        def exchange(self, document: Any) -> tuple[dict[str, Any], bytes]:
            sent.append(dict(document))
            return json.loads(json.dumps(first)), json.dumps(first).encode("utf-8")

        def selected_url(self, *_args: Any) -> None:
            return None

        def navigation(self, *_args: Any) -> dict[str, str]:
            return {}

    monkeypatch.setattr(client, "LocalResearchClient", Host)
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


def test_every_read_saved_with_its_defaults_reads_again_as_itself() -> None:
    """requirement (V571's class, TE12): every read the CLI saves keeps its request, its
    defaults included, so a continuation of that file reads the same read again before any next
    request of the same operation its answer offers, the references its answer names kept."""

    from alphalattice.interface.local_application import client
    from alphalattice.interface.local_application.cli_contract import command_table, named_read

    table = command_table()
    checked = 0
    for operation in sorted(table["reads"]):
        fields = table["fields"][operation]
        allowed = frozenset(fields["allowed"])
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
        checked += 1
    assert checked == len(table["reads"]) and checked > 50


def test_a_readback_saved_before_its_first_task_reads_again() -> None:
    """regression (V575, an outside review at 5f7e7375): `data-update show` saved before any
    Task could not be read again from its file: the continuation looked for the answer's Task
    before it applied the saved read, and refused when none was named. A read again as itself
    names no Task when its saved request named none and its operation needs none; a saved
    answer that names its Task still reads that Task. Pinned over every such readback."""

    from alphalattice.interface.local_application import client
    from alphalattice.interface.local_application.cli_contract import command_table, named_read

    table = command_table()
    readbacks = sorted(
        operation
        for operation in table["reads"]
        if "task_id" in table["fields"][operation]["allowed"]
        and "task_id" not in table["fields"][operation]["required"]
    )
    assert "DATA_UPDATE_READBACK" in readbacks, readbacks
    for operation in readbacks:
        allowed = frozenset(table["fields"][operation]["allowed"])
        before = named_read({"operation": operation}, {"status": "NOT_STARTED"})
        assert client.continued(operation, before, {}, allowed) == {"operation": operation}
        task = str(uuid4())
        after = named_read({"operation": operation}, {"status": "SUCCEEDED", "task_id": task})
        assert client.continued(operation, after, {}, allowed) == {
            "operation": operation,
            "task_id": task,
        }
    # A Task read still needs its Task: a status answer naming none is refused.
    status = frozenset(table["fields"]["STATUS"]["allowed"])
    bare = named_read({"operation": "STATUS"}, {"status": "AVAILABLE"})
    with pytest.raises(client.LocalResearchClientError, match="response_reference_missing"):
        client.continued("STATUS", bare, {}, status)


def _agent_project(tmp_path: Path, *, project: Path | None = None) -> Path:
    """A default Claude Code project with its configure-owned declaration and role card."""
    import shutil

    from alphalattice.interface.local_application.native_setup import declare_project

    checkout = SCRIPT.parents[1]
    project = tmp_path / "project" if project is None else project
    (project / ".claude" / "agents").mkdir(parents=True)
    (project / "notes" / "deep").mkdir(parents=True)
    shutil.copyfile(checkout / ".claude/settings.json", project / ".claude/settings.json")
    card = ".claude/agents/alphalattice_cro.md"
    shutil.copyfile(checkout / card, project / card)
    declare_project(project, "claude-code")
    return project


def _session_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Any:
    """`cli.main` in this process as an agent session runs it: its Claude Code session or none."""

    from alphalattice.interface.local_application.cli import main

    def run(*line: str, session: str | None) -> tuple[int, dict[str, Any]]:
        monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
        if session is None:
            monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
        else:
            monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", session)
        try:
            code = main(list(line), serve=lambda _arguments: 0)
        except SystemExit as stop:
            code = int(stop.code or 0)
        return code, dict(json.loads(capsys.readouterr().out.strip().splitlines()[-1]))

    return run


def test_a_workspace_left_out_is_the_bound_sessions_and_any_other_is_refused_in_words(
    live: LocalPortfolioWebSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """requirement (V568, the user's intent of 2026-10-03): after the initial phase an agent's
    commands leave `--workspace` out. `session bind` binds the session running it, its id read
    as every reader reads it, in the project holding the product's declarations; from any
    folder of the project the bound session's commands then work in its workspace, and every
    answer names how its workspace was chosen. A session not bound, another session, or none, is
    refused in words naming the full form and the bind, never given another workspace; and no
    unnamed session binds nothing. Another named Session may bind its own workspace."""

    from alphalattice.interface.local_application.native_bridge import BINDING_NAME

    project = _agent_project(tmp_path, project=live.workspace.parent)
    workspace = live.workspace
    monkeypatch.chdir(project / "notes" / "deep")
    run = _session_cli(monkeypatch, capsys)
    lead, other = str(uuid4()), str(uuid4())
    code, body = run("task", "list", session=lead)
    assert (code, body["failure_code"]) == (1, "local_client.workspace_unbound")
    assert "alphalattice --workspace <dir> session bind" in body["detail"]
    code, body = run("--workspace", str(workspace), "session", "bind", session=None)
    assert (code, body["failure_code"]) == (1, "local_client.session_unnamed")
    assert not (project / ".codex" / BINDING_NAME).exists()
    code, body = run("--workspace", str(workspace), "session", "bind", session=lead)
    assert (code, body["data"]["status"]) == (0, "BOUND")
    assert "foreground_attachment" not in body["data"], "FLOW-1: no hook asks to attach"
    assert (body["data"]["project"], body["data"]["session_id"]) == (str(project), lead)
    code, body = run("task", "list", session=other)
    assert (code, body["failure_code"]) == (1, "local_client.workspace_unbound")
    assert "alphalattice --workspace <dir>" in body["detail"]
    code, body = run("--workspace", str(workspace), "session", "bind", session=other)
    assert (code, body["data"]["status"]) == (0, "BOUND")
    code, body = run("task", "list", session=other)
    assert body["outcome"] == "OK", body
    assert body["context"]["workspace"] == str(workspace.resolve())
    assert body["context"]["workspace_from"] == "BINDING"
    code, body = run("task", "list", session=None)
    assert (code, body["failure_code"]) == (1, "local_client.workspace_unbound")
    # The bound session's own served workspace, from below the project.
    code, body = run("task", "list", session=lead)
    assert body["outcome"] == "OK", body
    assert body["context"]["workspace"] == str(workspace.resolve())
    assert body["context"]["workspace_from"] == "BINDING"
    code, body = run("--workspace", str(tmp_path), "task", "list", session=lead)
    assert body["context"]["workspace_from"] == "OPTION"
    assert body["context"]["workspace"] == str(tmp_path.resolve())


def test_a_bound_session_is_printed_its_commands_clean_and_any_other_the_full_form(
    live: LocalPortfolioWebSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """requirement (V568): in the bound session every command an answer prints is the clean
    `alphalattice <object> <action>`, whether the line named its workspace or left it out, and
    no string it prints names `--workspace`; another session, or a shell outside any, is printed
    the full form naming the workspace."""

    project = _agent_project(tmp_path, project=live.workspace.parent)
    monkeypatch.chdir(project)
    run = _session_cli(monkeypatch, capsys)
    lead = str(uuid4())
    named = ("--workspace", str(live.workspace))
    assert run(*named, "session", "bind", session=lead)[0] == 0

    def printed(body: dict[str, Any]) -> list[str]:
        found: list[str] = []
        pending: list[object] = [body]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
            elif isinstance(value, str) and value.startswith("alphalattice "):
                found.append(value)
        return found

    # An unknown Task is refused with its next requests, each printed as a command.
    asked = ("task", "show", str(uuid4()))
    for line in (asked, (*named, *asked)):
        code, body = run(*line, session=lead)
        assert code == 2, body
        commands = printed(body)
        assert commands and all("--workspace" not in command for command in commands), commands
        assert "--workspace" not in json.dumps(body)
    for session in (str(uuid4()), None):
        code, body = run(*named, *asked, session=session)
        commands = printed(body)
        assert commands and all(
            command.startswith(f"alphalattice --workspace {live.workspace.resolve()}")
            or command.startswith(f'alphalattice --workspace "{live.workspace.resolve()}"')
            or command.startswith(f"alphalattice --workspace '{live.workspace.resolve()}'")
            for command in commands
        ), commands


def test_session_unbind_removes_this_projects_binding_for_its_session_or_the_person(
    live: LocalPortfolioWebSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """requirement (V586): on the package leg no product command removed a binding, so a project
    bound to an ended session refused every later bind. P2-NH keeps a slot per real host/Session:
    a second Session binds without detaching the first, but never removes the first's record.
    A person detaches the sole record; multiple records require exact ownership. An inner
    project's unbind never reaches an outer project's binding."""

    from alphalattice.interface.local_application.native_bridge import BINDING_NAME

    project = _agent_project(tmp_path, project=live.workspace.parent)
    workspace = live.workspace
    monkeypatch.chdir(project / "notes" / "deep")
    run = _session_cli(monkeypatch, capsys)
    first, second = str(uuid4()), str(uuid4())
    bind = ("--workspace", str(workspace), "session", "bind")
    assert run(*bind, session=first)[0] == 0
    before = (project / ".codex" / BINDING_NAME).read_bytes()
    code, body = run("session", "unbind", session=second)
    assert (code, body["failure_code"]) == (
        2,
        "local_client.session_unbind_refused:native_bridge.session_mismatch",
    )
    assert "other Sessions' bindings stay" in body["detail"]
    assert "alphalattice --workspace <dir> session bind" in body["detail"]
    assert run(*bind, session=second)[0] == 0
    assert (project / ".codex" / BINDING_NAME).read_bytes() == before
    code, body = run("session", "unbind", session=None)
    assert (code, body["failure_code"]) == (
        2,
        "local_client.session_unbind_refused:native_bridge.binding_ambiguous",
    )
    assert "alphalattice session unbind" in body["detail"]
    assert body["next_action"] == "RESOLVE_THE_NAMED_CAUSE_THEN_UNBIND"
    code, body = run("session", "unbind", session=second)
    assert (code, body["data"]["status"], body["data"]["session_id"]) == (0, "DETACHED", second)
    assert (project / ".codex" / BINDING_NAME).read_bytes() == before
    code, body = run("session", "unbind", session=second)
    assert (code, body["failure_code"]) == (
        2,
        "local_client.session_unbind_refused:native_bridge.session_mismatch",
    )
    code, body = run("session", "unbind", session=first)
    assert (code, body["data"]["status"], body["data"]["session_id"]) == (0, "DETACHED", first)
    assert run("session", "unbind", session=first)[1]["data"]["status"] == "NOT_BOUND"
    assert run(*bind, session=second)[0] == 0
    code, body = run("session", "unbind", session=None)
    assert (code, body["data"]["status"], body["data"]["session_id"]) == (0, "DETACHED", second)
    assert not (project / ".codex" / BINDING_NAME).exists()
    # An inner project's unbind, by the person, never reaches the outer project's binding.
    assert run(*bind, session=first)[0] == 0
    inner = _agent_project(project / "notes")
    monkeypatch.chdir(inner)
    code, body = run("session", "unbind", session=None)
    assert (code, body["data"]["status"], body["data"]["project"]) == (0, "NOT_BOUND", str(inner))
    assert (project / ".codex" / BINDING_NAME).exists()


def test_every_product_text_naming_the_workspace_flag_is_a_kept_full_form() -> None:
    """requirement (V568): a bound session is printed no `--workspace`; the product's texts that
    name it are listed here, each with why it keeps the full form, so a new producer of a
    printed command names its reason or goes through `entry`."""

    import ast

    kept = {
        ("control/product_host/composition/entry.py", "serve"): "the launcher's own option",
        ("control/product_host/composition/evidence_authority_setup.py", "<module>"): (
            "its words: the launcher receives only --workspace"
        ),
        ("control/product_host/composition/evidence_authority_setup.py", "_parser"): (
            "the source setup script's own option"
        ),
        ("control/product_host/composition/evidence_authority_setup.py", "_acquisition_command"): (
            "the source setup script's own command, a script, not an object and an action; "
            "its acquisition offered again at one cutoff since V587"
        ),
        ("control/product_host/composition/evidence_authority_setup.py", "_import_command"): (
            "the source setup script's own command: a recorded import's check, its way on "
            "since V590"
        ),
        ("control/product_host/composition/evidence_review_application.py", "evidence_setup"): (
            "the source setup's script commands"
        ),
        ("control/product_host/composition/evidence_review_application.py", "source_ways"): (
            "the official serve way: the person's restart, in their own shell"
        ),
        ("control/product_host/composition/goals.py", "goal_prompt.command"): (
            "a /goal prompt opens another session's initial phase"
        ),
        ("control/product_host/composition/web_launcher.py", "<module>"): (
            "the Local Web launcher script's usage"
        ),
        ("control/product_host/composition/web_launcher.py", "main"): (
            "the Local Web launcher script's own option"
        ),
        ("evidence/alternative_evidence/runtime/policy.py", "<module>"): (
            "the source setup script's command template"
        ),
        ("interface/local_application/cli.py", "<module>"): "`session bind`'s help",
        ("interface/local_application/cli.py", "_command"): "serve's arguments to its launcher",
        ("interface/local_application/cli.py", "_global_options"): "the option and its help",
        ("interface/local_application/cli.py", "_repair"): (
            "a malformed --workspace's repair, the unexpected case"
        ),
        ("interface/local_application/cli_contract.py", "<module>"): (
            "the unbound refusals' way on and the binding's words"
        ),
        ("interface/local_application/cli_contract.py", "entry"): (
            "the full form, for every workspace but the bound one"
        ),
        ("interface/local_application/client.py", "_write_bundle"): (
            "where an entry's workspace ends, never printed"
        ),
        ("interface/local_application/native_setup.py", "main"): "the bridge setup's bind option",
        (
            "interface/local_application/portfolio_research.py",
            "LocalPortfolioResearchService._manifest",
        ): "an export's sealed reproduction command",
        ("investment/portfolio_strategy_lab/application/contracts.py", "export_command"): (
            "an export's sealed reproduction command"
        ),
    }
    source = SCRIPT.parents[1] / "src" / "alphalattice"
    found: set[tuple[str, str]] = set()

    def visit(node: ast.AST, owner: str, relative: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = owner
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = child.name if owner == "<module>" else f"{owner}.{child.name}"
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                if re.search(r"--workspace(?![-\w])", child.value):
                    found.add((relative, name))
                continue
            visit(child, name, relative)

    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(source).as_posix()
        visit(ast.parse(path.read_text(encoding="utf-8")), "<module>", relative)
    assert found == set(kept), (sorted(found - set(kept)), sorted(set(kept) - found))
    words = json.loads(
        (source / "interface/local_application/refusal_words.json").read_text("utf-8")
    )
    # backup restore's --workspace-id is a distinct option, not the global workspace flag.
    naming = {
        code for code, entry in words.items() if re.search(r"--workspace(?![-\w])", entry["detail"])
    }
    # The unexpected case's way on: a folder that holds no workspace names the right one.
    full_form_refusals = {
        "research_workspace.manifest_unreadable": "the folder cannot identify its workspace",
        "native_bridge.not_bound": "the initial session bind must explicitly name its workspace",
        "native_usage.read_limit_exceeded": (
            "unbind removes the implicit workspace; changing optional usage requires a fresh "
            "bind with that exact workspace, not a clean command against an absent binding"
        ),
        "native_bridge.existing_configuration_differs": (
            "changing this Session's scope first removes its implicit workspace"
        ),
        "native_bridge.binding_ambiguous": (
            "an unnamed Session cannot select another Session's implicit workspace"
        ),
        "native_bridge.binding_invalid": "an invalid record supplies no implicit workspace",
        "native_bridge.binding_limit_exceeded": (
            "an unbound Session can continue with an explicit workspace without replacing records"
        ),
        "native_bridge.binding_path_invalid": (
            "an unsafe record path supplies no implicit workspace"
        ),
        "native_bridge.binding_unreadable": "an unreadable record supplies no implicit workspace",
        "native_bridge.workspace_mismatch": (
            "another workspace cannot borrow this Session's binding; rebinding names its new scope"
        ),
    }
    assert naming == set(full_form_refusals)


@pytest.mark.parametrize("state", ["operator", "run", "closed", "open", "default", "unlocated"])
def test_every_network_way_on_reads_the_effective_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    """regression (V620, OP5, TE12): every switch-naming refusal refreshes stale advice
    from the effective owner; settings cannot lift an operator or run hold."""
    from contextlib import nullcontext

    import alphalattice.interface.local_application as local_application
    from alphalattice.control.product_host.composition.plain_refusals import explain
    from alphalattice.control.workspace_runtime.network_access import set_network_access
    from alphalattice.interface.local_application.cli_contract import (
        NETWORK_ACCESS_REFUSALS,
        refusal_words,
        worded_refusal,
    )
    from alphalattice.kernel.shared_kernel.environment import held_offline

    table = json.loads(Path(local_application.__file__).with_name("refusal_words.json").read_text())
    network_codes = {
        code
        for code, words in table.items()
        if words["next_action"] == "ASK_A_PERSON_TO_ALLOW_NETWORK_ACCESS"
    }
    assert (
        network_codes | {"evidence_review.workspace_network_not_allowed"} == NETWORK_ACCESS_REFUSALS
    )
    assert {
        code for code, words in table.items() if "network set" in words["detail"]
    } <= NETWORK_ACCESS_REFUSALS
    monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
    if state not in {"default", "unlocated"}:
        set_network_access(tmp_path, enabled=state != "closed")
    if state == "operator":
        monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    workspace = None if state == "unlocated" else tmp_path
    with held_offline() if state == "run" else nullcontext():
        for base in NETWORK_ACCESS_REFUSALS:
            code = base + (":2026-09-11,DATA" if base.startswith("research_update.") else "")
            owner = explain(code, workspace=workspace)
            fresh = worded_refusal(
                {
                    "status": "REFUSED",
                    "failure_code": code,
                    "detail": owner["detail"],
                    "network_access": {"decided_by": "WORKSPACE_CONTROL", "network_allowed": False},
                    "next_requests": {
                        "set": {"operation": "NETWORK_ACCESS_SET", "network_enabled": True}
                    },
                },
                workspace=workspace,
            )
            assert fresh["next_action"] == refusal_words(code, workspace=workspace)["next_action"]
            assert fresh["next_requests"] == {"network": {"operation": "NETWORK_ACCESS"}}
            if state in {"operator", "run"}:
                access = fresh["network_access"]
                assert access["decided_by"] == (
                    "OPERATOR_OFFLINE_SWITCH" if state == "operator" else "RUN_HELD_OFFLINE"
                )
                assert not access["network_allowed"] and not access["next_requests"]
                assert "network set" not in fresh["detail"]
                assert (
                    "ALPHALATTICE_NETWORK_DISABLED=1"
                    if state == "operator"
                    else "Wait for that run to finish"
                ) in fresh["detail"]
                if state == "operator":
                    assert "restart the idle Host" in fresh["detail"]
                else:
                    assert fresh["next_action"] == "WAIT_FOR_THE_OFFLINE_RUN_TO_FINISH"
            elif state in {"closed", "default"}:
                assert (
                    "set the workspace control"
                    if base == "evidence_review.workspace_network_not_allowed"
                    else "network set"
                ) in fresh["detail"]
                assert fresh["network_access"]["next_requests"]["set"]["network_enabled"]
            elif state == "open":
                assert fresh["network_access"]["network_allowed"]
                assert fresh["next_action"] == "RETRY_THE_REFUSED_STEP"
                assert "Network access is allowed now" in fresh["detail"]
                assert "network set" not in fresh["detail"]
            else:
                # No located control: a read is actionable without claiming its state.
                assert fresh["next_action"] == "READ_NETWORK_ACCESS"
                assert "network show" in fresh["detail"] and "network set" not in fresh["detail"]
                assert "network_access" not in refusal_words(code)
                assert "network_access" not in fresh
            if base.startswith("research_update."):
                assert "missing market data through session 2026-09-11" in fresh["detail"]
                assert "2026-09-11,DATA" not in fresh["detail"]


def test_the_installed_report_names_every_absent_performance_metric(
    live: LocalPortfolioWebSession,
) -> None:
    """regression (V617, OP4): old seals gain return-only readback from their own
    selected net path; unsupported benchmark metrics retain named absences."""
    import numpy as np

    from alphalattice.capabilities.portfolio_backtesting.metrics import (
        evaluate_net_simple_return_path,
    )

    admitted = _json(live, "/api/run", method="POST", payload={})
    assert live.dispatcher is not None and live.service is not None
    live.dispatcher.drain_for_tests()
    result_hash = _json(live, "/api/results")["results"][0]["result_hash"]
    saved = live.service.report(result_hash)
    saved_body = saved.model_dump(mode="json")
    store = live.service.application.ledger
    execution = store.load_execution(saved.execution_ledger_hash)
    economics = store.load_economics(saved.economic_ledger_hash)
    net = np.asarray(
        [
            economics.net_simple_returns[i]
            for i, session in enumerate(execution.formation_sessions)
            if saved.window_guard.selected_start <= session <= saved.window_guard.selected_end
        ],
        dtype=np.float64,
    )
    expected = evaluate_net_simple_return_path(net_simple_returns=net)
    day = saved.window_guard.selected_end.isoformat()
    code, undated, _ = _cli(live.workspace, "result", "show", result_hash)
    assert code == 0
    code, dated, _ = _cli(live.workspace, "result", "show", result_hash, "--session", day)
    assert code == 0
    body = dated["data"]
    json.dumps(body, allow_nan=False)
    measured = body["selected_window_metrics"]
    expected_metrics = {
        "cumulative_return": saved.window_cumulative_net_wealth - 1.0,
        "cost_bps": float(body["readouts"]["platform_one_way_cost_bps"]),
        **{
            name: value
            for name, value in expected.model_dump().items()
            if name != "cumulative_return" and np.isfinite(value)
        },
    }
    assert measured == expected_metrics
    assert undated["data"]["selected_window_metrics"] == measured
    provenance = body["selected_window_metric_provenance"]
    assert provenance == {
        "status": "DERIVED_FROM_SEALED_NET_RETURN_PATH",
        "report_hash": saved.report_hash,
        "execution_ledger_hash": execution.ledger_hash,
        "economic_ledger_hash": economics.economic_ledger_hash,
        "window_hash": saved.window_guard.guard_hash,
        "selected_start": saved.window_guard.selected_start.isoformat(),
        "selected_end": saved.window_guard.selected_end.isoformat(),
        "observation_count": len(net),
        "return_unit": "FRACTION",
        "annualization_sessions_per_year": 252,
        "volatility_degrees_of_freedom": 1,
        "sharpe_cash_return_per_session": 0.0,
        "sortino_downside_threshold": 0.0,
    }
    assert undated["data"]["selected_window_metric_provenance"] == provenance
    # Readback does not alter an old report's payload or scientific identity.
    assert live.service.report(result_hash).model_dump(mode="json") == saved_body
    absences = body["selected_window_metric_absences"]
    benchmark_absences = {
        "information_ratio",
        "benchmark_relative_return",
        "beta",
        "tracking_error",
        "zero_cash_jensen_alpha",
    }
    assert set(absences) == benchmark_absences | (
        {"sortino"} if not np.isfinite(expected.sortino) else set()
    )
    assert all(
        absences[name]
        == {
            "status": "UNAVAILABLE",
            "reason": "NOT_RECORDED_IN_DECLARED_PATH_REPORT",
            "detail": "The saved installed-strategy report does not record this metric "
            "for its selected window; this read does not estimate it.",
        }
        for name in benchmark_absences
    )
    if not np.isfinite(expected.sortino):
        assert absences["sortino"]["reason"] == "ZERO_DOWNSIDE_DEVIATION"
    assert undated["data"]["selected_window_metric_absences"] == absences
    view = _json(
        live, f"/api/workbench/portfolio?task_id={admitted['task_id']}&portfolio_session={day}"
    )
    assert view["metricAbsences"] == absences
    assert view["metrics"]["total"] == pytest.approx(
        body["selected_window_metrics"]["cumulative_return"] * 100
    )
    assert view["metrics"]["costBps"] == body["selected_window_metrics"]["cost_bps"]
    for alias, name, scale in (
        ("annual", "annualized_return", 100),
        ("vol", "annualized_volatility", 100),
        ("drawdown", "maximum_drawdown", 100),
        ("sharpe", "sharpe", 1),
        ("sortino", "sortino", 1),
    ):
        if name in measured:
            assert view["metrics"][alias] == pytest.approx(measured[name] * scale)
        else:
            assert view["metrics"][alias] is None
    for alias in (
        "informationRatio",
        "benchmarkRelative",
        "beta",
        "trackingError",
        "jensenAlpha",
    ):
        assert view["metrics"][alias] is None


def test_recovery_provenance_is_paired_typed_and_worded() -> None:
    """P3a: an existing owner request keeps its exact source pair or refuses it truthfully."""
    from uuid import UUID

    from alphalattice.interface.local_application.cli_contract import refusal_words
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
        PortfolioResearchRequestDocument,
    )

    source = UUID(int=83)
    context = {"recovery_task_id": str(source), "recovery_task_hash": "a" * 64}
    document = PortfolioResearchRequestDocument.model_validate(
        {"operation": "WORKSPACE_PREPARE_PLAN", **context}
    )
    request = document.to_operation_request()
    assert request.recovery_task_id == source and request.recovery_task_hash == "a" * 64
    assert (
        PortfolioResearchRequestDocument(operation="WORKSPACE_PREPARE_PLAN").recovery_task_id
        is None
    )
    for field in context:
        with pytest.raises(ValueError, match=r"portfolio_research\.recovery_context_pair_required"):
            PortfolioResearchRequestDocument.model_validate(
                {"operation": "WORKSPACE_PREPARE_PLAN", field: context[field]}
            )
    with pytest.raises(ValueError, match=r"portfolio_research\.recovery_task_id_invalid"):
        PortfolioResearchOperationRequest(
            operation="WORKSPACE_PREPARE_PLAN",
            recovery_task_id="not-a-task",
            recovery_task_hash="a" * 64,
        )  # type: ignore[arg-type]
    for code in (
        "portfolio_research.recovery_context_pair_required",
        "portfolio_research.recovery_task_id_invalid",
        "portfolio_research.recovery_request_not_offered",
    ):
        words = refusal_words(code)
        assert "recovery" in words["detail"]
        assert words["next_action"] == "READ_THE_TASK_AND_CONFIRM_AGAIN"


def test_a_book_review_follows_the_real_book_and_stops_where_evidence_needs_its_source(
    live, tmp_path
) -> None:
    """requirement (AGENT-TIME verb 2): through the real CLI and Host, `strategy-book review`
    runs the installed strategy's book with its default controls, follows its Task, finds its
    saved result and reads its report; where Evidence lacks its admitted source, that answer is
    the command's, its way on kept and its steps named."""
    from alphalattice.interface.local_application.client import LocalResearchClient

    package = LocalResearchClient(live.workspace).request({"operation": "CONTROLS"})[
        "strategy_package_id"
    ]
    code, answer, _ = _cli(
        live.workspace,
        "strategy-book",
        "review",
        "--package",
        package,
        "--dir",
        str(tmp_path / "analysts"),
        "--max-wait",
        "100",
    )
    data = answer["data"]
    assert answer["outcome"] == "REFUSED" and code != 0, answer
    assert data["status"] == "EVIDENCE_PREREQUISITES_MISSING"
    assert data["book_review"]["stopped_at"] == "evidence_preview"
    assert [step["step"] for step in data["book_review"]["steps"]] == [
        "controls",
        "book",
        "book_task",
        "book_result",
        "book_readback",
        "evidence_preview",
    ]
    assert "workspace" in data["next_requests"]
    assert not (tmp_path / "analysts").exists()
