"""Shared CLI invocations and parser admission support."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from alphalattice.interface.local_application import cli
from alphalattice.interface.local_application.cli import request_of
from alphalattice.interface.local_application.native_setup import declare_project

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"


def _cli(workspace: Path, *arguments: str) -> tuple[int, dict[str, Any], str]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--workspace", str(workspace), "--view", "full", *arguments],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return (result.returncode, json.loads(result.stdout.strip().splitlines()[-1]), result.stdout)


def _assert_commands_parse(
    commands: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Use request_of's own parser, stopping before file reads or product operations."""

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


"""Move the existing project/session builders to their shared support owner."""


def _agent_project(tmp_path: Path, *, project: Path | None = None) -> Path:
    """A synthetic agent project uses the configure-owned declaration and role card."""
    checkout = Path(__file__).resolve().parents[2]
    project = tmp_path / "project" if project is None else project
    (project / ".claude" / "agents").mkdir(parents=True)
    (project / "notes" / "deep").mkdir(parents=True)
    shutil.copyfile(checkout / ".claude/settings.json", project / ".claude/settings.json")
    card = ".claude/agents/alphalattice_cro.md"
    shutil.copyfile(checkout / card, project / card)
    declare_project(project, "claude-code")
    return project


def _session_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Any:
    """CLI entry in this process uses the named Claude Code session or the person."""

    def run(*line: str, session: str | None) -> tuple[int, dict[str, Any]]:
        monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
        if session is None:
            monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
        else:
            monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", session)
        try:
            code = cli.main(list(line), serve=lambda _arguments: 0)
        except SystemExit as stop:
            code = int(stop.code or 0)
        return code, dict(json.loads(capsys.readouterr().out.strip().splitlines()[-1]))

    return run


def _inprocess_cli(
    workspace: Path, *arguments: str, capsys: pytest.CaptureFixture[str]
) -> tuple[int, dict[str, Any], str]:
    """The real CLI entry answers in process with the same argv and full envelope."""
    try:
        code = cli.main(
            ["--workspace", str(workspace), "--view", "full", *arguments], serve=lambda _: 99
        )
    except SystemExit as stopped:
        code = int(stopped.code or 0)
    stdout = capsys.readouterr().out
    return code, json.loads(stdout.strip().splitlines()[-1]), stdout
