"""Printed commands round-trip through the CLI grammar."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from alphalattice.interface.local_application import cli, client
from alphalattice.interface.local_application.cli import request_of
from alphalattice.interface.local_application.cli_contract import (
    choices,
    command,
    command_table,
    entry,
)
from tests.portfolio_strategy_lab.cli_support import (
    _assert_commands_parse,
)
from tests.structural.refusal_support import door_words, fill_subject

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"

_AWKWARD = 'it\'s a "quoted" | %PATH% $HOME ~ value ü 中'

_SAMPLES: dict[str, list[Any]] = {
    "string": [_AWKWARD, "@handle", "-leads-with-a-dash", "", "two\nlines"],
    "integer": [7, 0, -3],
    "number": [0.25, -1.5],
    "boolean": [True, False],
    "object": [{"text": _AWKWARD, "at": "@x", "n": 1, "on": True, "items": [1, "two"]}, {}],
    "array": [["one two", "it's", "@x", "-y"], []],
}


def test_every_refusal_and_answer_template_command_parses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Owner recovery words and generated answer offers use the real CLI parser."""
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
    walk("refusal_words.json", door_words())
    walk("answers.json", json.loads((root / "answers.json").read_text(encoding="utf-8")))
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


def test_a_door_refusal_filled_with_its_subject_reads_in_chinese(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chinese door words preserve subjects, provider needs and unkeyed values."""

    from alphalattice.control.product_host.maintenance.data_update import NETWORK_WORK_WORDS
    from alphalattice.control.workspace_runtime.network_access import set_network_access
    from alphalattice.interface.local_application import cli_contract

    table = door_words()
    templated = [code for code, words in table.items() if "{subject}" in words["detail"]]
    assert templated, "the door's table words no sentence around a subject"

    monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
    set_network_access(tmp_path, enabled=False)
    keys = [key for key in cli_contract._zh() if key.count("{subject}") >= 2]
    assert keys, "no door sentence takes two subjects"
    token = cli_contract.ANSWER_LANGUAGE.set("zh")
    try:
        for code in templated:
            detail = cli_contract.refusal_words(
                fill_subject(code + ":{subject}", "2026-09-10"), workspace=tmp_path
            )["detail"]
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


def test_every_printed_command_reads_back_as_the_request_it_came_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, fake_host: Any
) -> None:
    """Printed requests preserve their alternative, typed values, quoting and inline document."""
    misread = []
    read = 0
    seen = set()
    table = command_table()
    for request in _offered_shapes():
        line = command(request, prefix=("alphalattice", "--workspace", "W"), quoting="posix")
        if line is None:
            primary = next(name for name in table["primary"] if isinstance(request.get(name), str))
            line = command(
                {**request, primary: None},
                prefix=("alphalattice", "--workspace", "W"),
                quoting="posix",
            )
            assert line is not None
            placeholder = f"<{primary}>"
            if placeholder not in line:
                continue
            declaration = tmp_path / "declaration.yaml"
            declaration.write_text(request[primary], encoding="utf-8", newline="\n")
            line = line.replace(placeholder, shlex.quote(str(declaration)))
        read += 1
        back = request_of(shlex.split(line)[1:])
        if _same(back) != _same(request):
            misread.append((line, back))
        for group in table["alternatives"].get(request["operation"], []):
            if all(request.get(field) is not None for field in group):
                seen.add((request["operation"], tuple(group)))
    assert read > 1500 and misread == []
    assert seen == {
        (operation, tuple(group))
        for operation, groups in table["alternatives"].items()
        for group in groups
    }

    workspace = tmp_path / "a book's workspace"
    request = {"operation": "ACTIVITY_LIST", "after": "@file"}
    prefix = entry(workspace)
    assert prefix == ("alphalattice", "--workspace", str(workspace.resolve()))
    posix = command(request, prefix=prefix, quoting="posix")
    assert posix is not None and shlex.split(posix)[: len(prefix)] == list(prefix)
    assert request_of(shlex.split(posix)[1:]) == request
    powershell = command(
        request, prefix=("C:/My Tools/python.exe", *prefix[1:]), quoting="powershell"
    )
    assert powershell is not None and powershell.startswith("& 'C:/My Tools/python.exe' ")
    assert "'" + str(workspace.resolve()).replace("'", "''") + "'" in powershell
    assert powershell.endswith("--after '@@file'")

    assert request_of(["network", "set", "--enabled", "true"])["network_enabled"] is True
    assert request_of(["network", "set", "--enabled", "False"])["network_enabled"] is False
    with pytest.raises(client.LocalResearchClientError, match="boolean_invalid"):
        request_of(["network", "set", "--enabled", "flase"])
    assert (
        request_of(["bundle", "prepare", "--role", "@@home", "--dir", "B"])["agent_role"] == "@home"
    )
    listed = tmp_path / "items.yaml"
    listed.write_text("- a\n- b\n", encoding="utf-8")
    assert request_of(["automation", "set", "--enabled", "false", "--packages", "@" + str(listed)])[
        "automation_package_ids"
    ] == ["a", "b"]

    fake_host.answer = {"status": "PLANNED", "plan_hash": "c" * 64}
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
    assert fake_host.sent == [request]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell is a Windows shell")
def test_a_command_printed_for_powershell_sends_the_request_it_came_from(tmp_path: Path) -> None:
    """A command printed for PowerShell sends the request it came from."""

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


def _limited(limits: dict[str, str], name: str) -> object | None:
    """A sample satisfies each field's declared limit."""
    words = limits.get(name, "")
    if (one_of := re.match(r"one of ([A-Z_]+)", words)) is not None:
        return one_of.group(1)
    if (ranged := re.match(r"(?:from|above) (\d+)", words)) is not None:
        return int(ranged.group(1)) + 1
    return None


def _offered_shapes() -> Iterator[dict[str, Any]]:
    """Every alternative and admitted value kind round-trips through the printed grammar."""
    table = command_table()
    plain = {"string": "plain", "integer": 2, "number": 1.5, "boolean": True}
    plain |= {"object": {"a": 1}, "array": ["a"]}
    for operation in sorted({op for ops in table["commands"].values() for op in ops}):
        contract = table["fields"][operation]
        groups = table["alternatives"].get(operation, [])
        for group in groups or [()]:
            base: dict[str, Any] = {"operation": operation}
            for name in [*contract["required"], *group]:
                fixed = _limited(table["limits"], name)
                base[name] = plain[table["types"][name][0]] if fixed is None else fixed
            for name in contract["allowed"]:
                fixed = _limited(table["limits"], name)
                kinds = table["types"][name]
                values = [fixed] if fixed is not None else [v for k in kinds for v in _SAMPLES[k]]
                for value in values:
                    request = {**base, name: value}
                    given = [
                        g for g in groups if all(request.get(field) is not None for field in g)
                    ]
                    if not choices(request) and len(given) <= 1:
                        yield request


def _same(value: Any) -> Any:
    """Numbers and optional null fields compare as their JSON requests do."""
    if isinstance(value, dict):
        return {key: _same(inner) for key, inner in value.items() if inner is not None}
    if isinstance(value, list):
        return [_same(inner) for inner in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value
