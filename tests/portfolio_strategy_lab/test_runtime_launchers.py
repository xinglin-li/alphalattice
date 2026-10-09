"""The installed entry composes product owners without checkout script imports."""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from alphalattice.control.product_host.composition import saved_object_readback
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    publish_research_workspace_manifest,
)


def test_product_probe_reopens_a_workspace_and_reads_saved_object_indexes(tmp_path: Path) -> None:
    """Requirement: sandbox readback starts actual product sessions with no harness."""
    harvest = tmp_path / "harvest"
    workspace = harvest / "workspace"
    publish_research_workspace_manifest(workspace, ResearchWorkspaceManifest.research_only("probe"))
    out = tmp_path / "readback.json"
    work = tmp_path / "copies"
    saved_object_readback.probe(harvest, work, out)
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["workspaces"] == 1
    by_op = {row["op"]: row for row in receipt["rows"]}
    assert set(by_op) == {"OPEN", "TASKS", "RESULTS", "FEATURE_TRIALS"}
    assert {op: row["verdict"] for op, row in by_op.items()} == {
        "OPEN": "OPENS",
        "TASKS": "OPENS",
        "FEATURE_TRIALS": "OPENS",
        "RESULTS": "REFUSES research_workspace.strategy_not_installed",
    }
    assert not work.exists()
    assert (workspace / "research-workspace.json").is_file()


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


@pytest.mark.parametrize("lines", ["stop\n", "ignored\nstop\n", ""])
def test_attached_launcher_stops_on_explicit_stdin_or_eof(tmp_path, monkeypatch, lines):

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


@pytest.mark.parametrize("ending", ["stop\n", ""])
def test_a_host_waiting_for_its_stop_lets_the_processes_it_starts_begin(ending: str) -> None:
    "a synchronous read pending on the stdin pipe held every process the"

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
