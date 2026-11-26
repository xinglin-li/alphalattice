from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from devtools.testing import collection

TEST_SOURCE = "src/devtools/testing/collection.py"


@pytest.mark.proves("testing.worker_collection_refusal_is_named")
def test_a_workers_collection_refusal_is_named_before_it_ends_the_worker(capsys):
    """pytest sends a worker's collection in a finally, so xdist's controller sees only a crash on
    a test that never ran; the worker names its own refusal first, and the refusal still ends it."""
    for config, named in (
        (SimpleNamespace(workerinput={"workerid": "gw3"}), True),
        (SimpleNamespace(), False),  # the controller or a plain run: pytest itself reports it
    ):
        hook = collection.pytest_collection(SimpleNamespace(config=config))
        next(hook)
        with pytest.raises(pytest.UsageError, match=r"test_size.lane_unknown"):
            hook.throw(pytest.UsageError("test_size.lane_unknown"))
        named_line = "ERROR: [gw3] test_size.lane_unknown" in capsys.readouterr().err
        assert named_line is named
    hook = collection.pytest_collection(SimpleNamespace(config=SimpleNamespace()))
    next(hook)
    with pytest.raises(StopIteration) as finished:
        hook.send(["collected"])
    assert finished.value.value == ["collected"]  # an accepted collection passes through unchanged


@pytest.mark.proves("testing.worker_crash_evidence_is_named")
def test_worker_crash_evidence_names_the_running_node_and_actual_exit(tmp_path):
    """An observed process exit and fault stream are attributed to the scheduled node."""
    worker = SimpleNamespace(
        gateway=SimpleNamespace(
            id="gw2", _io=SimpleNamespace(popen=SimpleNamespace(wait=lambda timeout: 3221225477))
        ),
    )
    (tmp_path / "worker-gw2.fault.log").write_text(
        "Fatal Python error: Segmentation fault\n", encoding="utf-8"
    )
    node = "tests/unit/example/test_x.py::test_value"
    capture = getattr(collection, "worker_crash_evidence", None)
    assert callable(capture), "worker crash evidence owner is missing"
    record = capture(worker, node, tmp_path)
    assert record["worker"] == "gw2"
    assert record["node"] == node
    assert record["exit"] == 3221225477
    assert "Fatal Python error" in record["stack"]
    (tmp_path / "worker-gw2.fault.log").unlink()
    missing = capture(worker, node, tmp_path)
    assert missing["exit"] == 3221225477
    assert missing["stack"] is None and missing["stack_error"]
    worker.gateway._io.popen = SimpleNamespace(wait=lambda timeout: None)
    unavailable = capture(worker, node, tmp_path)
    assert unavailable["exit"] is None and unavailable["exit_error"]
    worker.gateway.id = "../outside"
    with pytest.raises(ValueError, match=r"testing\.worker_identity_unknown"):
        capture(worker, node, tmp_path)


@pytest.mark.proves("testing.worker_crash_evidence_is_named")
def test_worker_crash_report_carries_the_exit_and_fault_stack(tmp_path, monkeypatch):
    """The controller attaches evidence to the crash's own JUnit report."""
    monkeypatch.setenv("WK_SUITE_ATTEMPT", "a" * 32)
    scheduler = SimpleNamespace(
        collection=["tests/unit/example/test_x.py::test_value"], node2pending={}
    )
    config = SimpleNamespace(
        option=SimpleNamespace(xmlpath=str(tmp_path / "junit.xml")),
        pluginmanager=SimpleNamespace(getplugin=lambda name: SimpleNamespace(sched=scheduler)),
    )
    worker = SimpleNamespace(
        gateway=SimpleNamespace(
            id="gw2", _io=SimpleNamespace(popen=SimpleNamespace(wait=lambda timeout: 3221225477))
        ),
        config=config,
    )

    # Real xdist workers are hashable; the fixture uses identity equality.
    class IdentityWorker:
        pass

    actual = IdentityWorker()
    actual.gateway, actual.config = worker.gateway, config
    scheduler.node2pending[actual] = [0]
    fault = "Fatal Python error: fixture crash\n"
    (tmp_path / "worker-gw2.fault.log").write_text(fault, encoding="utf-8")
    down = getattr(collection, "pytest_testnodedown", None)
    crash = getattr(collection, "pytest_handlecrashitem", None)
    assert callable(down) and callable(crash), "worker crash hooks are missing"
    down(actual, "Not properly terminated")
    report = SimpleNamespace(node=actual, longrepr="worker 'gw2' crashed while running test")
    crash(scheduler.collection[0], report, scheduler)
    assert "gw2" in str(report.longrepr) and "3221225477" in str(report.longrepr)
    assert fault.strip() in str(report.longrepr)


@pytest.mark.proves("testing.paid_agent_runs_only_on_request")
@pytest.mark.parametrize(
    ("marker", "under_xdist"),
    [
        pytest.param(collection.PAID_AGENT, nullcontext, id="paid-agent"),
        pytest.param(
            collection.PERFORMANCE,
            lambda: pytest.raises(pytest.UsageError, match="serial_run_required"),
            id="performance-serial-only",
        ),
    ],
)
def test_a_paid_agent_test_runs_only_when_the_run_asks_for_it(monkeypatch, marker, under_xdist):
    """No stage, queue, nightly or full run selects a model turn that spends money, or a
    performance row, which measures only in a serial run with pytest-benchmark loaded."""
    marked = {"held": True, "free": False}
    items = [
        SimpleNamespace(
            name=name, get_closest_marker=lambda mark, held=held: (held and mark == marker) or None
        )
        for name, held in marked.items()
    ]
    dropped = []
    config = SimpleNamespace(
        hook=SimpleNamespace(pytest_deselected=lambda items: dropped.extend(items)),
        pluginmanager=SimpleNamespace(hasplugin=lambda name: name == "benchmark"),
        getoption=lambda name, default: default,
    )
    monkeypatch.delenv(collection.ON_REQUEST[marker], raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    kept = list(items)
    collection.deselect_on_request(config, kept)
    assert [i.name for i in kept] == ["free"] and [i.name for i in dropped] == ["held"]
    monkeypatch.setenv(collection.ON_REQUEST[marker], "1")
    kept = list(items)
    collection.deselect_on_request(config, kept)
    assert [i.name for i in kept] == ["held", "free"]
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    with under_xdist():
        collection.deselect_on_request(config, list(items))


@pytest.mark.proves("testing.shared_fixture_has_its_own_hang_guard")
def test_a_wider_fixtures_fill_has_the_hang_guard_to_itself_and_its_case_has_it_after():
    """The armed guard is armed afresh as a module fixture's fill starts and as it ends; a
    function fixture leaves it, as does a unit case, and with no guard armed a fill arms none."""
    events: list[str] = []

    def through(wrapper, **arguments):  # pytest-timeout's own implementation, under the wrapper
        hook = wrapper(**arguments)
        next(hook)
        with pytest.raises(StopIteration):
            hook.send(True)

    def cancel(item):
        events.append("cancel")
        through(collection.pytest_timeout_cancel_timer, item=item)

    def arm(item, settings):
        events.append(f"arm {settings}")
        through(collection.pytest_timeout_set_timer, item=item, settings=settings)

    hooks = SimpleNamespace(pytest_timeout_cancel_timer=cancel, pytest_timeout_set_timer=arm)
    case = SimpleNamespace(config=SimpleNamespace(hook=hooks), stash={})
    unit = SimpleNamespace(
        config=SimpleNamespace(hook=hooks),
        stash={collection.CONTRACT: SimpleNamespace(layer="unit")},
    )

    def fill(scope):
        hook = collection.pytest_fixture_setup(
            SimpleNamespace(argname="seed", scope=scope),
            SimpleNamespace(config=SimpleNamespace(stash={}), node=SimpleNamespace(stash={})),
        )
        next(hook)
        armed_during = list(events)
        with pytest.raises(StopIteration):
            hook.send("seed")
        return armed_during

    try:
        assert fill("module") == [] and events == []  # no guard armed: none is armed for it
        through(collection.pytest_timeout_set_timer, item=case, settings="900 s")
        assert fill("module") == ["cancel", "arm 900 s"]
        assert events == ["cancel", "arm 900 s", "cancel", "arm 900 s"]
        events.clear()
        assert fill("function") == [] and events == []
        through(collection.pytest_timeout_cancel_timer, item=case)
        assert collection.ARMED == {}
        through(collection.pytest_timeout_set_timer, item=unit, settings="60 s")
        assert fill("module") == [] and events == []  # a unit case's fixtures fill nothing
    finally:
        collection.ARMED.clear()
