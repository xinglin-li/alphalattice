"""A test's real teardown judges the answers of the Hosts its own scopes started, and only those.

An inner pytest session runs real tests with the product's own root conftest: its Host stamping,
its wrapped door and its teardown observer. Its outcomes are what pytest itself reports.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

TEST_SOURCE = "src/devtools/evaluation/round_trip.py"
ROOT = Path(__file__).resolve().parents[3]
CASES = """
from uuid import UUID

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from tests._support.portfolio_strategy_lab_fixtures import live as live
from tests.portfolio_strategy_lab.local_web_support import _manifest, _resolved, _Resolver


@pytest.fixture(scope="module")
def module_host(tmp_path_factory):
    # The module scope under test is this file's own, never a shared fixture's.
    host = LocalPortfolioWebSession(
        workspace=tmp_path_factory.mktemp("module-host"),
        workspace_manifest=_manifest("qa-module-host"),
        resolver=_Resolver(_resolved()),
    )
    host.start()
    yield host
    host.stop()

TASK = str(UUID(int=1))
STOPPED = {
    "status": "ACCEPTED", "task_id": TASK, "command_kind": "STRATEGY_BOOK_RUN",
    "disposition": "ACCEPTED", "refusal_detail": None, "lifecycle": "RECOVERY_REQUIRED",
}
KEPT = []


def _stopped_run(operations, monkeypatch):
    monkeypatch.setattr(type(operations), "run", lambda self, spec: STOPPED)
    operations.execute(PortfolioResearchOperationRequest(operation="RUN", spec={}))


def test_a_module_hosts_stopped_answer(module_host, monkeypatch):
    _stopped_run(module_host.operations, monkeypatch)


def test_keeps_its_own_host(live):
    KEPT.append(live.operations)  # its work outlives the test, as a late answer does


def test_hears_the_kept_host_answer(monkeypatch):
    _stopped_run(KEPT[0], monkeypatch)
"""


@pytest.mark.proves("evaluation.answer_observer_scopes_real_hosts")
def test_real_teardowns_judge_own_hosts_and_say_foreign_ones(tmp_path: Path) -> None:
    """At pytest's own teardown: a module fixture's Host fails the test that heard its stopped
    answer, naming its Task and module; a Host one test kept answers during the next, which stays
    green and is warned of it, naming both tests."""
    (tmp_path / "test_cases.py").write_text(CASES, encoding="utf-8")
    path = os.pathsep.join([str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])
    inner = [
        *("-p", "tests.conftest", "-p", "no:cacheprovider", "-n", "0", "-rA"),
        *("--basetemp", str(tmp_path / "inner"), "--junitxml", str(tmp_path / "inner.xml")),
        "test_cases.py",
    ]
    result = subprocess.run(  # pytest's own teardown is the behaviour under test
        [sys.executable, "-m", "pytest", *inner],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": path},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    cases = {
        case.get("name"): [
            (child.tag, child.get("message", ""))
            for child in case
            if child.tag in ("error", "failure", "skipped")
        ]
        for case in ElementTree.parse(tmp_path / "inner.xml").iter("testcase")
    }
    task = "Task 00000000-0000-0000-0000-000000000001"
    [(kind, message)] = cases.pop("test_a_module_hosts_stopped_answer")
    assert kind == "error" and f"offers no way on ({task}" in message, cases
    assert "Host of test_cases.py)" in message, message
    assert cases == {"test_keeps_its_own_host": [], "test_hears_the_kept_host_answer": []}
    warned = result.stdout + result.stderr
    assert (
        "a Host of 'test_cases.py::test_keeps_its_own_host', "
        "during test_cases.py::test_hears_the_kept_host_answer)" in warned
    ), warned
