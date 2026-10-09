"""The CLI's one answer shape and its exit codes (binding plan, C1 rules 1 and 2).

Every command prints one envelope and exits by its outcome: 0 OK, 1 INVALID_INPUT, 2 REFUSED,
3 PENDING, 4 NO_HOST. `--output` holds the owner's answer whatever its status, and no answer
carries exception text or a path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Any, get_args
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.control.product_host.research_authoring.comparison import POSITION_UNITS
from alphalattice.interface.local_application import cli, client
from alphalattice.interface.local_application.cli import main
from alphalattice.interface.local_application.cli_contract import (
    ANSWER_LANGUAGE,
    DECLARATION_OPERATION_EXCEPTIONS,
    MAXIMUM_REQUEST_BODY_BYTES,
    client_refusal,
    command_table,
    entry,
    envelope,
)
from alphalattice.interface.local_application.failure_codes import FAILURE_DETAIL_WITHHELD
from alphalattice.interface.local_application.labels import label
from tests.portfolio_strategy_lab.cli_support import (
    _agent_project,
    _cli,
    _inprocess_cli,
    _session_cli,
)
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

_INTERNAL_CODE = re.compile(
    r"\b(?:V\d{2,3}|U\d{1,3}|(?:ID|OW|OP|PA|DA|EV|TE|PR|SC|LY|TY|ST|GY|WK|GR|AS|AX|NM|RR|FU|LS)"
    r"\d{1,3}|CLI-\d+)\b|\bGate [A-Z]\b|\bStage \d\b|\bPolicy [A-Z]\b"
    r"|\((?:[A-Z]{1,4}\d{0,3}[a-z]?)(?:, ?[A-Z]{1,4}\d{0,3}[a-z]?)*\)"
)

_PRODUCT_WORDS = frozenset({"(CLI)", "(CPU)", "(CRO)", "(ISO)", "(PM)", "(UTC)"})


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


def test_storage_cap_is_one_operator_setting_through_the_real_cli(
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
    read_only_live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    live = read_only_live
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
    read_only_live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    live = read_only_live
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


def test_lang_zh_words_an_answers_detail_and_keeps_its_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Chinese words both owner refusals and parse refusals without changing their codes."""

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


def test_every_answer_counts_the_agent_sessions_launches_help_included(tmp_path: Path) -> None:
    """requirement (V364): an agent reported 35 launches against a budget of 40 after 41,
    its help launches miscounted; every answer carries `session_launches`, the session's
    launches with help counted, and an answer outside an agent session carries none."""

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


def test_one_reader_names_the_session_a_request_and_its_launches_count_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One session reader assigns both requests and their counted launches to the same
    unambiguous session."""

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


@pytest.mark.parametrize("case", ["empty", "large", "missing-and-unreadable"])
def test_an_empty_document_file_is_refused_by_its_own_code(
    case: str, tmp_path: Path, capsys: Any, fake_host: Any
) -> None:
    """Empty, oversized, missing and unreadable documents name their cause and way on."""
    if case == "empty":
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
    elif case == "large":
        large = tmp_path / "completion.json"
        large.write_text(
            json.dumps({"summary": "x" * MAXIMUM_REQUEST_BODY_BYTES}), encoding="utf-8"
        )
        code = cli.main(
            [
                "--workspace",
                str(tmp_path),
                "--view",
                "full",
                "goal",
                "submit",
                str(uuid4()),
                "--revision",
                "a" * 64,
                "--file",
                str(large),
            ],
            serve=lambda _: 99,
        )
        body = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert (code, body["failure_code"]) == (1, "local_client.document_too_large"), body
        assert body["document_size"] == {
            "bytes": large.stat().st_size,
            "limit_bytes": MAXIMUM_REQUEST_BODY_BYTES,
        }
        assert body["next_action"] == "SEND_A_SMALLER_DOCUMENT"
        assert "references" in body["detail"] and "cannot be read" not in body["detail"]
    else:
        missing = tmp_path / "copies" / "01-controls.json"
        code = cli.main(
            ["--workspace", str(tmp_path), "--view", "full", "request", "--file", str(missing)],
            serve=lambda _: 99,
        )
        body = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert (code, body["failure_code"]) == (1, "local_client.document_missing"), body
        assert body["document_location"] == {"file": str(missing), "expected": "an existing file"}
        assert body["next_action"] == "CHECK_THE_FILE_PATH" and "schema" not in body["detail"]
        broken = tmp_path / "broken.json"
        broken.write_bytes(b"\xff\xfe not utf-8")
        code = cli.main(
            ["--workspace", str(tmp_path), "--view", "full", "request", "--file", str(broken)],
            serve=lambda _: 99,
        )
        body = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert (code, body["failure_code"]) == (1, "local_client.document_unreadable"), body
        assert body["document_location"]["file"] == str(broken)
        refusal = client_refusal("local_client.document_unreadable")
        assert "YAML or JSON" in refusal.detail
        assert refusal.next_action == "CORRECT_THE_DOCUMENT_WHERE_READING_STOPPED"
        assert "JSON;" not in client_refusal("local_client.document_missing").detail
    assert fake_host.sent == []


def test_a_nouns_help_names_every_verbs_flags() -> None:
    """Object and top-level help show registered actions, flags, person-only doors and order."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "feature", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "COLUMNS": "240"},
    )
    text = " ".join(result.stdout.split())
    table = command_table()
    starts = {}
    for name, operations in table["commands"].items():
        if not name.startswith("feature "):
            continue
        verb = name.split()[1]
        purpose = " Or: ".join(dict.fromkeys(table["purposes"][op] for op in operations))
        assert f"{verb} {purpose}" in text
        starts[verb] = text.index(f"{verb} {purpose}")
    ordered = sorted(starts, key=starts.__getitem__)
    rows = {
        verb: text[starts[verb] : starts[ordered[i + 1]] if i + 1 < len(ordered) else len(text)]
        for i, verb in enumerate(ordered)
    }
    assert "--binding [--plan]" in rows["controls"]
    assert "<id> --plan" in rows["review"]
    assert "<id> --plan" in rows["activate"]
    assert "FEATURE_ACTIVATE" in table["person_only"] and "person" in rows["activate"]
    assert "--from (a saved answer)" in text

    shown = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60
    )
    words = " ".join(shown.stdout.split())
    positions = [words.index(command) for command in table["common_path"]]
    assert positions == sorted(positions)
    groups = [words.index(group + ":") for group, _nouns in table["help_groups"]]
    assert groups == sorted(groups) and positions[-1] < groups[0]
    for _group, nouns in table["help_groups"]:
        assert all(noun in words for noun, _purpose in nouns)


def test_an_unknown_operation_is_answered_with_the_nearest_names(tmp_path: Path) -> None:
    """Schema reads name unknown operations, declaration writers and typed answer sections."""

    code, answer, _stderr = _cli(tmp_path, "schema", "show", "EXPERIMENT_COMPAR")
    assert code == 1 and answer["failure_code"] == "local_client.operation_unknown"
    assert "EXPERIMENT_COMPARE" in answer["nearest_operations"]

    plan = cli.schema("EXPERIMENT_PLAN")
    assert "`study controls`" in plan["written_by"]
    assert "`study draft`" in plan["written_by"]
    declaration_note = plan["template"].splitlines()[1]
    assert declaration_note.startswith("#") and plan["written_by"] in declaration_note
    assert "written_by" not in cli.schema("STATUS")

    parts = cli.schema("EXPERIMENT_READBACK")["answer_parts"]["factor.screening-development"]
    items = parts["result"]["evidence_report"]["items"]
    assert parts["result"]["evidence_report"]["hypothesis_count"] == "integer"
    assert items[0]["classification"] == "string"


def test_a_metric_read_alone_carries_its_unit(tmp_path: Path, capsys: Any, fake_host: Any) -> None:
    """Section reads preserve the owner's outcome, available parts and each metric's unit."""
    fake_host.answer = {"status": "OK", "tasks": []}
    code = cli.main(
        ["--workspace", str(tmp_path), "--view", "full", "task", "list", "--section", "tasks"],
        serve=lambda _: 99,
    )
    body = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0 and body["data"]["section"] == "tasks"
    assert isinstance(body["data"]["value"], list)
    code = cli.main(
        [
            "--workspace",
            str(tmp_path),
            "--view",
            "full",
            "task",
            "list",
            "--section",
            "tasks.frobnicate",
        ],
        serve=lambda _: 99,
    )
    body = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (code, body["outcome"]) == (0, "OK") and "tasks" in body["data"]
    assert body["local_failure"]["failure_code"] == "local_client.section_unknown:tasks.frobnicate"
    code = cli.main(
        ["--workspace", str(tmp_path), "--view", "full", "task", "list", "--section", "data.tasks"],
        serve=lambda _: 99,
    )
    body = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (code, body["outcome"]) == (0, "OK")
    assert "tasks" in body["local_failure"]["sections"]
    fake_host.answer = {
        "status": "OK",
        "position": {"one_way_turnover": 0.12, "session": "2024-01-03"},
        "metric_units": {f"position.{key}": unit for key, unit in POSITION_UNITS.items()},
    }

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
    tmp_path: Path, capsys: Any, fake_host: Any
) -> None:
    """Next commands and bundle submission preserve the caller's view, language and goal."""

    task, goal = str(uuid4()), str(uuid4())
    body = {
        "status": "SUCCEEDED",
        "next_requests": {"show": {"operation": "EXPERIMENT_READBACK", "task_id": task}},
    }
    fake_host.answer = body
    context = ("--view", "full", "--lang", "zh", "--goal", goal)
    cli.main(["--workspace", str(tmp_path), *context, "decision", "list"], serve=lambda _: 99)
    whole = json.loads(capsys.readouterr().out)["next_commands"]["show"]
    assert f" --view full --lang zh --goal {goal} study show {task}" in whole, whole
    cli.main(["--workspace", str(tmp_path), "decision", "list"], serve=lambda _: 99)
    shown = json.loads(capsys.readouterr().out)["next_commands"]["show"]
    assert shown.endswith(f" study show {task}") and "--view" not in shown, shown

    workspace = tmp_path / "workspace"
    prefix = entry(workspace, ("--lang", "zh", "--goal", "goal-g"))
    body = {"status": "AGENT_BUNDLE_READY", "files": [{"name": "README.md", "text": "Read.\n"}]}
    bundle = client._write_bundle(tmp_path / "bundle", body, prefix=prefix)
    assert bundle["submit_arguments"][:6] == [*prefix[1:]]
    assert bundle["submit_arguments"][6:8] == ["bundle", "submit"]
    if "submit_command" in bundle:
        words = shlex.split(bundle["submit_command"], posix=False)
        assert "--goal" in words and "goal-g" in [w.strip("'\"") for w in words]


def test_the_product_text_names_no_internal_code() -> None:
    """The product text names no internal code."""

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
    """Answer titles follow installed label keys in either language without changing data."""

    body = {
        "status": "OK",
        "strategies": {"RETURN_G6_MU_ONLY": {"weight_rule": "ew"}},
        "component_id": "G2_R0_TREND",
    }
    answer = envelope(operation="STRATEGY_SHOW", outcome="OK", body=body, elapsed_seconds=0.0)
    identifiers = {"G2_R0_TREND", "RETURN_G6_MU_ONLY", "ew"}
    assert set(answer["titles"]) == identifiers
    for identifier in identifiers:
        installed = label(identifier)
        assert installed is not None and answer["titles"][identifier] == installed.title
    assert answer["data"] == body
    language = ANSWER_LANGUAGE.set("zh")
    try:
        chinese = envelope(operation="STRATEGY_SHOW", outcome="OK", body=body, elapsed_seconds=0)
    finally:
        ANSWER_LANGUAGE.reset(language)
    assert set(chinese["titles"]) == identifiers
    assert chinese["titles"] != answer["titles"]
    rebound = label("RETURN_G6_MU_ONLY")
    assert rebound is not None and chinese["titles"]["RETURN_G6_MU_ONLY"] == rebound.title_zh
    plain = envelope(operation="STATUS", outcome="OK", body={"status": "OK"}, elapsed_seconds=0)
    assert "titles" not in plain


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


def test_an_operation_only_a_person_completes_is_marked_so(
    read_only_live: LocalPortfolioWebSession,
) -> None:
    """An operation only a person completes is marked so."""
    live = read_only_live

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


def test_every_read_of_work_planned_per_strategy_is_keyed_by_its_strategy() -> None:
    """Every read of work planned per strategy is keyed by its strategy."""

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


def test_every_door_whose_way_on_reruns_a_plan_resumes_its_stopped_task() -> None:
    """Every door whose way on reruns a plan resumes its stopped task."""

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
    for code in reruns:
        if code == "workspace_data_update.retry_not_due":
            pass
        elif code.startswith("workspace_data_update."):
            assert owner.resumes(code), code
        else:
            assert code.startswith("workspace_preparation."), code


def test_no_newer_plan_and_no_admitted_task_waits_behind_an_update_that_has_not_ended() -> None:
    """No newer plan and no admitted task waits behind an update that has not ended."""

    from alphalattice.interface.local_application import cli_contract

    # The preparation refuses a new preparation by name while any Task is unfinished
    # (`workspace_preparation.finish_or_recover_existing_task`); the updates answer theirs.
    # A research update's own data plan is sealed into its own Task, never a waiting one's.
    # The update that waits answers whatever target was asked: its match names the package alone.
    # The automation attends what it admitted: a deferral at its retry time, a later session
    # once its resumed update publishes, and a package refused while the inputs were held.
    # A refusal for the inputs names their state and the request that settles them.
    words = cli_contract.refusal_words(
        "strategy_score.workspace_inputs_not_ready:DATA_REVIEW_PENDING"
    )
    assert "DATA_REVIEW_PENDING" in words["detail"] and "data-update plan" in words["detail"]


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
            "The approved stock-list change could not be verified: the active membership is "
            "neither the one it approved nor one its own run recorded or the gateway's admission "
            "derives from it, or its prior Panel or approved candidate document changed. Nothing "
            "ran. The research and history already recorded stay readable; to keep updating, "
            "prepare a new workspace from the current source. Never edit a stored approval.",
            "已获批准的股票清单变更无法核验\uff1a当前生效的成员既不是它批准的成员\uff0c也不是它自身运"
            "行所记录、或网关准入从中派生的成员\uff1b或者此前的面板或已批准的候选文件已经改变。没"
            "有执行任何操作。已记录的研究和历史仍可读取\uff1b要继续更新\uff0c请从当前来源准备一个新的"
            "工作区。不要修改已存储的批准记录。",
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


def test_a_compact_answer_shows_references_short_and_they_can_be_sent_so() -> None:
    """A compact answer shows references short and they can be sent so."""

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


def test_a_saved_goal_completion_runs_as_a_whole_request(live, tmp_path: Path) -> None:
    """A saved goal completion runs as a whole request."""
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


def test_max_wait_bounds_a_wait_on_a_host_that_never_answers(
    read_only_live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """The maximum wait bounds a request to a Host that never answers."""
    live = read_only_live

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


def test_the_global_options_are_read_anywhere_on_the_line(
    tmp_path: Path, monkeypatch: Any, capsys: Any, fake_host: Any
) -> None:
    """The global options are read anywhere on the line."""

    from alphalattice.interface.local_application import cli

    seen = fake_host.workspaces
    fake_host.answer = lambda document: {"status": "OK", "task_id": document["task_id"]}
    fake_host.raw = b"{}"

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
    tmp_path: Path, monkeypatch: Any, capsys: Any, fake_host: Any
) -> None:
    """A compact answer holds its next requests to the read."""

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
    fake_host.answer = body

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
    for pad in (1, 16, 32, 64, 128, 160):
        command = ["--workspace", str(tmp_path / ("w" * pad)), "task", "list", "--list-next"]
        cli.main(command, serve=lambda _: 99)
        line = capsys.readouterr().out.strip().splitlines()[-1]
        assert len(line.encode("utf-8")) <= client.COMPACT_ANSWER_BYTES, pad
    refused = answer("--next-from", "3")[1]
    assert (refused["outcome"], refused["failure_code"]) == (
        "INVALID_INPUT",
        "local_client.next_from_invalid",
    ), refused


def test_a_refusal_leaves_with_words_and_a_way_on() -> None:
    """A refusal leaves with words and a way on."""

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


@pytest.mark.parametrize(
    "case",
    [
        "book",
        "experiment",
        "bound_package",
        "no_document",
        "twice",
        "stdin",
        "unavailable",
        "study",
        "feature",
    ],
)
def test_an_offered_request_reads_its_operation_document_from_file(
    fake_host, tmp_path: Path, capsys, case
) -> None:
    """Declaration routing keeps bound selectors and refuses unsupported or ambiguous sources."""
    saved, declaration, choices = (
        tmp_path / name for name in ("answer.json", "declaration.yaml", "choices.json")
    )
    fake_host.answer = {"status": "OK"}
    file = tmp_path / "none.yaml"
    workspace = ["--workspace", str(tmp_path)]
    if case in {"book", "experiment"}:
        operation, field, document = {
            "book": ("PLAN", "spec", {"strategy_package_id": "SYNTHETIC_SINGLE_BOOK", "top_k": 2}),
            "experiment": (
                "EXPERIMENT_PLAN",
                "experiment_yaml",
                "experiment:\n  kind: factor.screening-development\n",
            ),
        }[case]
        bound = {"strategy_package_id": "SYNTHETIC_SINGLE_BOOK"} if field == "spec" else None
        offered = {"operation": operation, **({field: bound} if bound else {})}
        saved.write_text(json.dumps({"next_requests": {"preview": offered}}), encoding="utf-8")
        declaration.write_text(
            json.dumps(document) if isinstance(document, dict) else document,
            encoding="utf-8",
            newline="\n",
        )
        seen = fake_host.sent
        assert (
            cli.main(
                [
                    *workspace,
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
        return
    if case == "unavailable":
        code, refusal, _ = _inprocess_cli(
            tmp_path, "task", "list", "--save-declaration", str(file), capsys=capsys
        )
        assert (code, refusal["failure_code"]) == (1, "local_client.usage_invalid")
        assert not file.exists()
        assert fake_host.sent == []
        request = tmp_path / "request.json"
        request.write_text(json.dumps({"operation": "TASKS"}), encoding="utf-8")
        code = cli.main(
            [*workspace, "request", "--file", str(request), "--save-declaration", str(file)],
            serve=lambda _: 99,
        )
        answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert code == 0 and answer["outcome"] == "OK"
        assert answer["local_failure"]["failure_code"] == "local_client.declaration_unavailable"
        assert answer["local_failure"]["detail"] and answer["local_failure"]["next_action"]
        assert not file.exists()
        return
    if case in {"study", "feature"}:
        command, operation = {
            "study": (["study", "show"], "EXPERIMENT_READBACK"),
            "feature": (["feature", "show", "--task"], "FEATURE_CATALOG_BUILD_READBACK"),
        }[case]
        code = cli.main(
            [*workspace, *command, str(uuid4()), "--save-declaration", str(file)],
            serve=lambda _: 99,
        )
        answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert (code, answer["failure_code"]) == (
            1,
            "local_client.operation_has_no_editable_declaration:" + operation,
        )
        assert answer["detail"] and answer["next_action"]
        assert DECLARATION_OPERATION_EXCEPTIONS[operation] and not file.exists()
        assert fake_host.sent == []
        return
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
    code = cli.main([*workspace, *arguments], serve=lambda _: 99)
    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (code, answer["failure_code"]) == (1, expected), answer
    assert answer["detail"] and answer["next_action"]
    assert fake_host.sent == []


def test_a_bound_session_is_printed_its_commands_clean_and_any_other_the_full_form(
    live: LocalPortfolioWebSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bound session receives its way-on commands in the clean form."""

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
    code, body = run(*asked, session=lead)
    assert code == 2, body
    commands = printed(body)
    assert commands and all("--workspace" not in command for command in commands), commands
    assert "--workspace" not in json.dumps(body)


def test_a_book_review_follows_the_real_book_and_stops_where_evidence_needs_its_source(
    live, tmp_path
) -> None:
    """A book review follows the real book and stops where evidence needs its source."""
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


def test_a_command_schema_names_both_admitted_declaration_forms(
    tmp_path: Path, capsys: Any, fake_host: Any
) -> None:
    """A command's schema names its operation and either admitted declaration representation."""
    code, body, _ = _inprocess_cli(tmp_path, "schema", "show", "study", "plan", capsys=capsys)
    assert (code, body["data"]["operation"]) == (0, "EXPERIMENT_PLAN"), body
    assert body["data"]["command"] == "alphalattice study plan"
    assert body["data"]["oneOf"] == [
        {"required": ["experiment_yaml"]},
        {"required": ["experiment_document"]},
    ]
    assert fake_host.sent == []


def test_a_saved_answer_from_stdin_selects_its_read_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, fake_host: Any
) -> None:
    """A saved answer read from stdin selects its own read operation and exact task."""
    task = str(uuid4())
    monkeypatch.setattr(
        sys,
        "stdin",
        TextIOWrapper(BytesIO(json.dumps({"data": {"task_id": task}}).encode("utf8"))),
    )
    _code, answer, _ = _inprocess_cli(tmp_path, "study", "show", "--from", "-", capsys=capsys)
    assert answer["operation"] == "EXPERIMENT_READBACK" and "usage_error" not in answer, answer
    assert fake_host.sent == [{"operation": "EXPERIMENT_READBACK", "task_id": task}]


def test_the_option_terminator_keeps_the_following_id_as_an_operand(
    tmp_path: Path, capsys: Any, fake_host: Any
) -> None:
    """The GNU option terminator sends the following identifier as an operand."""
    goal_id = str(uuid4())
    fake_host.answer = {"status": "OK", "goal_id": goal_id}
    code, body, _ = _inprocess_cli(tmp_path, "goal", "show", "--", goal_id, capsys=capsys)
    assert (code, body["data"]["goal_id"]) == (0, goal_id), body
    assert fake_host.sent == [{"operation": "GOAL_SHOW", "goal_id": goal_id}]


def test_choices_require_the_saved_answer_they_fill(
    tmp_path: Path, capsys: Any, fake_host: Any
) -> None:
    """The choices flag requires the saved answer whose offered request it fills."""
    declaration = tmp_path / "goal.yaml"
    code, body, _ = _inprocess_cli(
        tmp_path, "goal", "open", "--choices", str(declaration), capsys=capsys
    )
    assert code == 1 and "--from" in body["usage_error"]["observed"], body
    assert fake_host.sent == []


def test_a_saved_declaration_reads_from_stdin_and_names_its_context(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """A saved declaration reads from stdin and every answer names its workspace and goal."""
    code, body, _ = _cli(live.workspace, "task", "list")
    assert code == 0 and body["context"]["workspace"] == str(live.workspace.resolve()), body
    assert body["context"]["goal"] is None
    declaration = tmp_path / "goal.yaml"
    code, body, _ = _cli(live.workspace, "goal", "schema", "--save-declaration", str(declaration))
    assert code == 0, body
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
    _code, planned, _ = _cli(live.workspace, "--goal", goal_id, "data-update", "plan")
    assert planned["context"]["goal"] == goal_id, planned


def test_a_shared_read_schema_names_each_alternative_selector(
    tmp_path: Path, capsys: Any, fake_host: Any
) -> None:
    """A shared read command's schema names each operation and its alternative selector."""
    code, body, _ = _inprocess_cli(tmp_path, "schema", "show", "study", "show", capsys=capsys)
    branches = body["data"]["oneOf"]
    assert (code, body["data"]["operations"]) == (0, [b["operation"] for b in branches]), body
    assert sorted(tuple(b["required"]) for b in branches) == [
        ("experiment_plan_hash",),
        ("task_id",),
    ]
    assert fake_host.sent == []


def test_exclusive_read_selectors_are_refused_before_a_request_is_sent(
    tmp_path: Path, capsys: Any, fake_host: Any
) -> None:
    """A command refuses two mutually exclusive selectors before it sends a request."""
    code, body, _ = _inprocess_cli(
        tmp_path, "study", "show", str(uuid4()), "--plan", "a" * 64, capsys=capsys
    )
    assert code == 1 and "not both" in body["usage_error"]["observed"], body
    assert fake_host.sent == []


def test_a_declaration_and_choices_never_give_one_field_twice(
    tmp_path: Path, capsys: Any, fake_host: Any
) -> None:
    """A file and choices cannot silently overwrite the same request field."""
    declaration = tmp_path / "goal.yaml"
    fake_host.answer = {"status": "AVAILABLE", "template": {"title": "A goal"}}
    code, _, _ = _inprocess_cli(
        tmp_path, "goal", "schema", "--save-declaration", str(declaration), capsys=capsys
    )
    assert code == 0
    fake_host.sent.clear()
    offered = tmp_path / "offered.json"
    offered.write_text(
        json.dumps(
            {"next_requests": {"open": {"operation": "GOAL_OPEN", "goal_declaration": None}}}
        ),
        encoding="utf-8",
    )
    choices = tmp_path / "choices.yaml"
    choices.write_text("goal_declaration: {title: twice}\n", encoding="utf-8")
    code, body, _ = _inprocess_cli(
        tmp_path,
        *("goal", "open", "--from", str(offered), "--file", str(declaration)),
        *("--choices", str(choices)),
        capsys=capsys,
    )
    assert (code, body["failure_code"]) == (1, "local_client.field_given_twice:goal_declaration"), (
        body
    )
    assert fake_host.sent == []
