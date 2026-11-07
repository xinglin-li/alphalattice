"""A full pytest run records session fixture and sampled temporary footprint evidence, and a
parallel run keeps its longest-first order when a worker crashes."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from xdist.scheduler import LoadScheduling

from devtools.testing import collection
from tests.structural.pytest_support import child_environment

TEST_SOURCE = "src/devtools/testing/collection.py"


@pytest.mark.proves("testing.session_footprint")
@pytest.mark.parametrize(
    "workers,scenario",
    [
        (0, "normal"),
        (2, "normal"),
        (2, "golden-summary"),
        (0, "collision"),
        (0, "nested"),
        (0, "fixture-error"),
        (0, "other-attempt"),
    ],
)
def test_session_fixture_footprint_and_peak_are_recorded(tmp_path, workers, scenario, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    (tmp_path / "suite_marker.py").write_text("", encoding="utf-8")
    (tmp_path / "test_session_fixture.py").write_text(
        "import logging, pytest\n"
        "@pytest.fixture(scope='session')\n"
        "def seed(tmp_path_factory):\n"
        "    root=tmp_path_factory.mktemp('seed')\n"
        "    (root/'payload').write_bytes(b'x'*37)\n"
        "    logging.getLogger('devtools.testing.session_footprint').info("
        "'PLAYPEN_SESSION_WORKSPACE BUILD seed 0.001s')\n"
        + ("    raise RuntimeError('fixture control')\n" if scenario == "fixture-error" else "")
        + "    return root\n"
        "@pytest.fixture(scope='session')\n"
        "def no_root(): return 7\n"
        + ("@pytest.mark.large\n" if scenario == "golden-summary" else "")
        + "def test_seed(seed, no_root):\n"
        "    assert (seed/'payload').stat().st_size == 37 and no_root == 7\n"
        + (
            "    from devtools.evaluation.golden_workspaces import note_acquisition\n"
            "    note_acquisition(dict(recipe='probe', nested=False, cold=True, "
            "fill_seconds=1.25, lock_seconds=0.0, copy_seconds=0.0))\n"
            if scenario == "golden-summary"
            else ""
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "PYTEST_ADDOPTS", "-k parent_only_marker"
    )  # the child must not inherit the parent's selection
    env = child_environment(
        WK_SUITE_REPORT=str(tmp_path / "suite.json"),
        WK_SUITE_PLUGIN="suite_marker",
        WK_SUITE_ATTEMPT="f" * 32,
        WK_SUITE_SOURCE_SHA="a" * 40,
        PYTHONPATH=os.pathsep.join((str(tmp_path), str(root / "src"))),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_PLUGINS="",
        ALPHALATTICE_NETWORK_DISABLED="1",
    )
    if scenario == "nested":
        env["ALPHALATTICE_SESSION_FOOTPRINT_OWNER"] = "outer:fixture"
    if scenario == "other-attempt":
        (tmp_path / f"session-footprint-{'e' * 32}-main.json").write_text("other", encoding="utf8")
    if scenario == "collision":
        (tmp_path / f"session-footprint-{'f' * 32}-main.json").write_text(
            "occupied", encoding="utf8"
        )
        (tmp_path / f"session-footprint-{'f' * 32}-main.log").write_text(
            "occupied", encoding="utf8"
        )
    done = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-p",
            "devtools.testing.collection",
            "-p",
            "suite_marker",
            *(["-p", "xdist.plugin", "-n", str(workers)] if workers else []),
            "--rootdir",
            str(root),
            "--noconftest",
            "--basetemp",
            str(tmp_path / "base"),
            "--junitxml",
            str(tmp_path / "junit.xml"),
            "-o",
            "addopts=",
            "-q",
            str(tmp_path / "test_session_fixture.py"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf8",
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert done.returncode == (1 if scenario == "fixture-error" else 0), done.stdout + done.stderr
    if scenario == "golden-summary":
        summary = done.stdout.partition("Golden fills in Large tests")[2]
        assert all(value in summary for value in ("test_seed", "1.250", "probe:fill"))
    record = tmp_path / f"session-footprint-{'f' * 32}-main.json"
    if scenario == "nested":
        assert not record.exists()
        return
    if scenario == "collision":
        assert record.read_text(encoding="utf8") == "occupied"
        assert (tmp_path / f"session-footprint-{'f' * 32}-main.log").read_text(
            encoding="utf8"
        ) == "occupied"
        assert "test_size.session_footprint_unmeasured" in done.stdout + done.stderr
        return
    if scenario == "other-attempt":
        assert (tmp_path / f"session-footprint-{'e' * 32}-main.json").read_text(
            encoding="utf8"
        ) == "other"
    assert record.is_file(), "session footprint sidecar is missing"
    data = json.loads(record.read_text(encoding="utf8"))
    records = [
        json.loads(path.read_text(encoding="utf8"))
        for path in tmp_path.glob(f"session-footprint-{'f' * 32}-*.json")
    ]
    seed = next(row for item in records for row in item["fixtures"] if row["name"] == "seed")
    if scenario == "fixture-error":
        assert seed["bytes"] is None and "RuntimeError" in seed["reason"]
        return
    unknown = next(row for item in records for row in item["fixtures"] if row["name"] == "no_root")
    assert seed["bytes"] == 37 and seed["build_seconds"] >= 0
    assert unknown["bytes"] is None and unknown["reason"]
    assert data["scope"] == "unverified until joined to the attempt manifest"
    assert data["peak"]["sample_seconds"] >= 0
    assert data["peak"]["bytes"] >= 37 and any(
        "seed" in row["path"] for row in data["peak"]["contributors"]
    )
    assert any(
        "PLAYPEN_SESSION_WORKSPACE BUILD seed" in path.read_text(encoding="utf8")
        for path in tmp_path.glob("session-footprint-*.log")
    )


class _Worker:
    """The parts of an xdist worker controller the load scheduler drives."""

    shutting_down = False

    def __init__(self, name: str) -> None:
        self.gateway = SimpleNamespace(id=name)
        self.sent: list[int] = []

    def send_runtest_some(self, indices: list[int]) -> None:
        self.sent.extend(indices)

    def shutdown(self) -> None:
        self.shutting_down = True


def _crash_after_prefetch(scheduler: type) -> tuple[list[int], list[int]]:
    """Two workers start on cases 0-1 and 2-3; the second dies in case 2 with case 3 prefetched."""
    config = SimpleNamespace(
        getvalue=lambda name: ["2*popen"], getoption=lambda name, default=None: 2
    )
    schedule = scheduler(config)
    first, second = _Worker("gw0"), _Worker("gw1")
    for worker in (first, second):
        schedule.add_node(worker)
    for worker in (first, second):
        schedule.add_node_collection(worker, [f"t{index:02d}" for index in range(12)])
    schedule.schedule()
    schedule.mark_test_complete(first, 0, duration=1.0)
    assert schedule.remove_node(second) == "t02"
    schedule.mark_test_complete(first, 1, duration=1.0)
    return first.sent, schedule.pending


@pytest.mark.proves("testing.crash_requeue_keeps_longest_first")
def test_a_crashed_workers_prefetched_case_keeps_its_place_in_the_order() -> None:
    """A crash preserves queued priority and its timeout is not recorded as a fast test."""
    _, xdist_pending = _crash_after_prefetch(LoadScheduling)
    assert xdist_pending[-1] == 3
    sent, pending = _crash_after_prefetch(collection.ordered_requeue())
    assert sent[:4] == [0, 1, 4, 3] and pending == sorted(pending)
    node = (
        "tests/integration/product_host/test_input_revisions.py::"
        "test_update_capture_copy_and_recovery_leave_defaults_and_reports_unchanged"
    )
    other = "tests/test_scheduling_example.py::test_measured"
    fast = "tests/test_scheduling_example.py::test_fast"
    recorded = {node: 0.0, other: 1584.016, fast: 1.0}
    policy = collection.load_policy(Path(__file__).resolve().parents[3])
    hints = getattr(collection, "scheduling_hints", lambda *_: {})(recorded, policy)
    assert hints.get(node) == 900.0 and hints.get(other) == 1584.016 and recorded[node] == 0.0
    assert collection.scheduling_hints({fast: 0.0}, policy)[fast] == 0.0
    timeout = json.dumps({"event": "TIMEOUT", "nodeid": node, "elapsed_seconds": 900.02})
    assert (
        collection.merge_timing_hints(recorded, {node: 0.0}, {node}, ["log", timeout])[node]
        == 900.02
    )
    from tests import conftest as scheduler

    items = [SimpleNamespace(nodeid=name) for name in (node, fast, other)]
    config = SimpleNamespace(
        workerinput={"recorded_durations": hints, "recorded_distribution": "load"},
        getoption=lambda _name, default=None: default,
    )
    scheduler.pytest_collection_modifyitems(config, items)
    assert [item.nodeid for item in items] == [other, node, fast]
    controller = SimpleNamespace(
        _recorded_scheduling_hints=hints, getoption=lambda _name, default=None: default
    )
    worker = SimpleNamespace(workerinput={}, config=controller)
    scheduler.pytest_configure_node(worker)
    assert worker.workerinput["recorded_durations"][node] == 900.0
