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
