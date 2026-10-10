"""The path explorer walks every offered way on against a real served Host, offline."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from scripts.agent_eval.environments.explorer_host import ROOT, served

from devtools.evaluation.explorer import Explorer
from devtools.evaluation.forms import Scenario

SERVE = ["scripts/run_alphalattice.py", "--workspace", "{ws}", "serve", "--no-browser"]
SERVE.append("--stop-on-stdin")
FIRST_USE = ["first-use", "prepare", "--sentence", "Positions for 2026-10-12.", "--date"]
FIRST_USE.append("2026-10-12")
SESSION = {"CLAUDE_CODE_SESSION_ID": "explorer-session"}


def _scenario(name: str, route: str, faults: tuple[str, ...] = ()) -> Scenario:
    return Scenario(
        id=name,
        sentence=name.replace("_", " "),
        now="2026-10-10T15:00:00+00:00",
        injected_faults=faults,
        expected_route=((route,),),
        decider="agent",
        forbidden_acts=("restart_host",),
    )


def _walk(scenario, workspace, starts, out, env, *, heal=None, checkout=ROOT):
    launcher = [a.format(ws=workspace) for a in SERVE]
    log = out / f"{scenario.id}.host.log"
    with served(workspace, launcher, env=env, checkout=checkout, log=log) as (call, shell):
        (out / "offers").mkdir(parents=True, exist_ok=True)
        explorer = Explorer(call, shell, scenario, out / "offers", time_box=90, heal=heal)
        grade, trace = explorer.explore(starts)
    # The broken paths whole, and every answer on them, kept beside the walk.
    (out / f"{scenario.id}.grade.json").write_text(json.dumps(grade, indent=1), encoding="utf-8")
    (out / f"{scenario.id}.trace.jsonl").write_text(trace.dumps_jsonl(), encoding="utf-8")
    return grade


def test_a_first_use_without_the_retrieval_runtime_names_it(tmp_path: Path) -> None:
    """requirement: a first use on a product without the retrieval runtime names it first."""
    assert importlib.util.find_spec("fastembed") is None, "precondition: fastembed importable"
    assert not (ROOT / ".venv-retrieval").exists(), "precondition: a retrieval runtime is filled"
    scenario = _scenario("first_use_without_runtime", "retrieval_runtime", ("no_runtime",))
    assert _walk(scenario, tmp_path / "ws", [FIRST_USE], tmp_path, SESSION)["passed"]


def test_a_codex_wake_without_the_codex_command_names_the_wait_in_the_turn(tmp_path) -> None:
    """requirement: a missing dependency stops by name with its way on, never silently."""
    path = os.pathsep.join(
        d for d in os.environ["PATH"].split(os.pathsep) if not shutil.which("codex", path=d)
    )
    env = {"CODEX_THREAD_ID": "explorer-thread", "PATH": path}
    scenario = _scenario("codex_wake_without_codex", "wait inside this turn", ("no_codex",))
    starts = [[*FIRST_USE, "--notify", "codex-queue"]]
    assert _walk(scenario, tmp_path / "ws", starts, tmp_path, env)["passed"]


@pytest.mark.real_evidence
def test_the_evidence_setup_reaches_its_preflight_on_a_registered_copy(tmp_path: Path) -> None:
    """acceptance, on demand: a Host started before its runtime, on a registered copy of
    `first-use/after` served from a checkout of ALPHALATTICE_EXPLORER_SOURCE, walks red or
    green as ALPHALATTICE_EXPLORER_EXPECT names."""
    copy, source = (
        Path(os.environ["ALPHALATTICE_EXPLORER_QA_COPY"]),
        os.environ["ALPHALATTICE_EXPLORER_SOURCE"],
    )
    # Short, beside the copy in its run: a deep path makes Windows refuse the runtime's DLLs.
    checkout = copy.parents[2] / "temp" / f"co-{source}"
    shutil.rmtree(checkout, ignore_errors=True)
    parts = ["src", "scripts", "config", "pyproject.toml", "uv.lock", "README.md", "LICENSE"]
    archive = ["git", "-C", str(ROOT), "archive", source, *parts, "NOTICE"]
    with tarfile.open(fileobj=io.BytesIO(subprocess.run(archive, capture_output=True).stdout)) as t:
        t.extractall(checkout, filter="data")
    assets = Path("src/alphalattice/interface/local_application/assets")
    for built in (ROOT / assets).glob("workbench*.*"):  # the ignored Workbench build, not tested
        shutil.copy2(built, checkout / assets / built.name)
    assert importlib.util.find_spec("fastembed") is None, "precondition: fastembed importable"
    fill = [sys.executable, str(checkout / "scripts" / "create_retrieval_environment.py")]

    def documented_fill() -> None:  # the guide's own fill, once, at the first stop
        subprocess.run([*fill, "--offline"], cwd=checkout, capture_output=True, check=True)

    setup = "--research-input-id factor-development --research-input-hash "
    setup += "5a48f11317e60c5976384c3f3c65f482731c8dae4b7f74986705873a3cbc41c3"
    preflight = [
        "evidence",
        "install",
        f"--setup={setup} --acquire-sec --entities HAL SLB --preflight",
    ]
    evidence = _scenario("evidence_setup", "EVIDENCE_SOURCE_PREFLIGHT", ("filled_after_start",))
    first_use = _scenario("first_use_without_runtime", "retrieval_runtime", ("no_runtime",))
    # The first use meets a product without the runtime, so it walks before the fill.
    walks = [
        _walk(first_use, tmp_path / "fresh", [FIRST_USE], tmp_path, SESSION, checkout=checkout),
        _walk(
            evidence,
            copy,
            [preflight, preflight],
            tmp_path,
            SESSION,
            heal=documented_fill,
            checkout=checkout,
        ),
    ]
    expected = os.environ["ALPHALATTICE_EXPLORER_EXPECT"] == "green"
    assert [walk["passed"] for walk in walks] == [expected, expected], walks
