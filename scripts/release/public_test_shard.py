"""Bound the public pytest shard and retain the results of tests it reported."""

import argparse
import faulthandler
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import psutil
import pytest
import pytest_timeout
from _pytest.junitxml import bin_xml_escape, mangle_test_address

# The shard's own clock, bound at import: a test that fakes time.monotonic or time.time on
# the shared time module never reaches the events and deadlines a worker writes from its
# hooks (an exhausted fake ended an xdist worker, 11-23).
_monotonic, _wall = time.monotonic, time.time

_stream = None
_owner = "controller"
# Public runs stop hangs. The commit stage judges ordinary test size budgets.
HANG_SECONDS = 900
# The C watchdog gives pytest-timeout time to retain and exit after its diagnostic.
HANG_GRACE = 60


@pytest.hookimpl(wrapper=True)
def pytest_collection_modifyitems(items):
    result = yield
    for item in items:
        marker = item.get_closest_marker("timeout")
        limit = (
            (marker.args[0] if marker.args else marker.kwargs.get("timeout")) if marker else None
        )
        hang = max(HANG_SECONDS, limit or 0)  # a test's own longer bound stays
        item.add_marker(pytest.mark.timeout(hang, func_only=False), append=False)
        item.user_properties[:] = [
            (key, hang if key == "test_timeout_seconds" else value)
            for key, value in item.user_properties
        ]
    return result


def _event(kind, **fields):
    if _stream is not None:
        try:
            _stream.write(
                json.dumps({"event": kind, "at": round(_wall(), 3), "tick": _monotonic(), **fields})
                + "\n"
            )
            _stream.flush()
        except (OSError, TypeError, ValueError) as error:
            print(f"worker diagnostics unavailable: {type(error).__name__}", file=sys.stderr)


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    global _stream, _owner
    worker = getattr(config, "workerinput", {}).get("workerid")
    _owner = worker or "controller"
    try:
        path = Path(os.environ["PYTEST_WORKER_DIAGNOSTICS"]) / f"{_owner}-{os.getpid()}.log"
        _stream = path.open("w", encoding="utf-8", errors="backslashreplace", buffering=1)
        faulthandler.enable(file=_stream, all_threads=True)
    except Exception as error:
        print(f"worker diagnostics unavailable: {type(error).__name__}", file=sys.stderr)
        if _stream is not None:
            _stream.close()
        _stream = None
        return
    if worker is not None:
        _event("worker", worker=worker, pid=os.getpid())
    _watch("worker setup" if worker is not None else "controller setup")
    if worker is None:
        return

    def capture(original):
        def timed_out(item, settings):
            if not settings.disable_debugger_detection and pytest_timeout.is_debugging():
                return original(item, settings)
            try:
                _event("timeout", node=item.nodeid, seconds=settings.timeout)
                faulthandler.dump_traceback(file=_stream, all_threads=True)
            except Exception as error:
                print(f"worker diagnostics unavailable: {type(error).__name__}", file=sys.stderr)
            return original(item, settings)

        return timed_out

    pytest_timeout.timeout_timer = capture(pytest_timeout.timeout_timer)
    pytest_timeout.timeout_sigalrm = capture(pytest_timeout.timeout_sigalrm)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    """Print only a timeout for this crashed worker's still-active test."""
    if error is None:
        return
    worker = node.gateway.id
    if not re.fullmatch(r"gw[0-9]+", worker):
        return
    _watch("worker disconnected", seconds=None, owner=worker)
    reporter = node.config.pluginmanager.get_plugin("terminalreporter")
    try:
        logs = Path(os.environ["PYTEST_WORKER_DIAGNOSTICS"])
        files = sorted(
            path
            for path in logs.iterdir()
            if path.name.startswith(f"{worker}-") and path.suffix == ".log"
        )
        if len(files) != 1:
            raise ValueError("worker diagnostic is missing or ambiguous")
        path = files[0]
        active = timeout = None
        identified = False
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("{"):
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("worker event is not an object")
            if event.get("event") == "worker":
                pid = event.get("pid")
                if identified or type(pid) is not int or pid <= 0:
                    raise ValueError("worker identity is invalid")
                identified = event.get("worker") == worker and path.stem == f"{worker}-{pid}"
            elif not identified:
                raise ValueError("worker identity is unavailable")
            elif event.get("event") == "start":
                active, timeout = event.get("node"), None
                if not isinstance(active, str) or not active:
                    raise ValueError("active test identity is invalid")
            elif event.get("event") == "timeout" and active and event.get("node") == active:
                seconds = event.get("seconds")
                if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError("timeout bound is invalid")
                timeout = line
            elif event.get("event") == "end" and event.get("node") == active:
                active = timeout = None
        if not identified:
            raise ValueError("worker identity is unavailable")
        if active and timeout:
            message = f"Worker timeout: {timeout}"
            if reporter is not None:
                reporter.write_line(message)
            else:
                print(message, file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as problem:
        message = f"Worker {worker} timeout diagnostic unavailable: {type(problem).__name__}"
        if reporter is not None:
            reporter.write_line(message)
        else:
            print(message, file=sys.stderr)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    _event("start", node=item.nodeid)
    marker = item.get_closest_marker("timeout")
    limit = (marker.args[0] if marker.args else marker.kwargs.get("timeout")) if marker else None
    _watch("test protocol", node=item.nodeid, seconds=(limit or HANG_SECONDS) + HANG_GRACE)
    if _stream is not None and limit:
        # pytest-timeout's Windows exit can block. The C timer can exit without the GIL.
        faulthandler.dump_traceback_later(limit + HANG_GRACE, exit=True, file=_stream)
    try:
        return (yield)
    finally:
        if _stream is not None and limit:
            faulthandler.cancel_dump_traceback_later()
        _event("end", owner=_owner, node=item.nodeid)
        if _owner == "controller":
            _watch("worker coordination", seconds=None)


def _watch(phase, *, seconds=HANG_SECONDS, node=None, owner=None, pid=None):
    _event(
        "phase",
        owner=owner or _owner,
        phase=phase,
        node=node,
        until=None if seconds is None else _monotonic() + seconds,
        pid=os.getpid() if owner is None else pid,
    )


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    _watch("worker startup", owner=node.gateway.id)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    _watch("collection")


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtestloop(session):
    _watch("worker coordination", seconds=None)
    try:
        return (yield)
    finally:
        _watch("session finish")


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    _watch("session finish")
    try:
        return (yield)
    finally:
        _watch("worker shutdown" if _owner != "controller" else "process exit")


def _test_phase(item, phase):
    _event("test phase", owner=_owner, phase=phase, node=item.nodeid)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _test_phase(item, "fixture setup")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    _test_phase(item, "test call")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item):
    _test_phase(item, "fixture teardown")


def _record_failure(report, phase):
    if report.failed:
        crash = getattr(report.longrepr, "reprcrash", None)
        kind = getattr(crash, "message", "").partition(":")[0].partition("\n")[0]
        detail = (
            kind
            if re.fullmatch(r"(?:[A-Za-z_]\w*\.)*(?:[A-Za-z_]\w*(?:Error|Exception)|Failed)", kind)
            else "pytest report failed"
        )
        _event("failure", node=report.nodeid, phase=phase, detail=detail)


def pytest_runtest_logreport(report):
    """Retain the failed node and exception type without capturing child output."""
    _record_failure(report, report.when)


def pytest_collectreport(report):
    """Retain collection failures when a later worker or the controller stalls."""
    _record_failure(report, "collection")


@dataclass
class Phase:
    """The last reported phase of one process, with its original deadline."""

    name: str
    tick: float
    until: float | None
    node: str | None = None
    pid: int | None = None

    def label(self):
        """Identify the scope that was still running at expiry."""
        return self.name if self.node is None else f"{self.name} {self.node}"


class ShardWatch:
    """Follow a shard's process phases and retain failures as they arrive."""

    def __init__(self, began):
        self.phases = {"controller": Phase("pytest startup", began, began + HANG_SECONDS)}
        self.last_event = began
        self.failures = {}
        self._offsets = {}

    def observe(self, event):
        """Advance one process without renewing a test when it moves to teardown."""
        tick = event["tick"]
        self.last_event = max(self.last_event, tick)
        kind = event["event"]
        if kind == "failure":
            key = (event["node"], event["phase"])
            self.failures.setdefault(key, event["detail"])
        elif kind == "phase":
            owner = event["owner"]
            previous = self.phases.get(owner)
            if previous is None or tick >= previous.tick:
                self.phases[owner] = Phase(
                    event["phase"], tick, event["until"], event["node"], event["pid"]
                )
        elif kind == "test phase":
            previous = self.phases.get(event["owner"])
            if previous is not None and previous.node == event["node"] and tick >= previous.tick:
                previous.name = event["phase"]
        elif kind == "end" and (owner := event.get("owner")) != "controller":
            previous = self.phases.get(owner)
            if previous is not None and previous.node == event["node"] and tick >= previous.tick:
                # xdist enters a protocol with the next item or shutdown already assigned.
                self.phases[owner] = Phase(
                    "worker handoff", tick, tick + HANG_SECONDS, previous.node, previous.pid
                )

    def read(self, logs):
        """Read only the complete new rows, including rows flushed before an abrupt exit."""
        for path in sorted(file for file in logs.iterdir() if file.suffix == ".log"):
            with path.open("rb") as source:
                source.seek(self._offsets.get(path, 0))
                while (line := source.readline()).endswith(b"\n"):
                    self._offsets[path] = source.tell()
                    if line.startswith(b"{"):
                        try:
                            event = json.loads(line)
                        except ValueError as error:
                            raise ValueError(
                                f"public shard diagnostics malformed: {path.name}"
                            ) from error
                        if isinstance(event, dict) and "tick" in event:
                            self.observe(event)

    def expired(self, now, *, cutoff, alive):
        """Select the expired process phase or the shard's last admissible time."""
        running = sorted(
            (phase.until, owner, phase)
            for owner, phase in self.phases.items()
            if phase.until is not None and (phase.pid is None or alive(phase.pid))
        )
        if running and (running[0][0] <= now or now >= cutoff):
            _, owner, phase = running[0]
            return f"{owner}: {phase.label()}"
        if now >= cutoff or (not running and now >= self.last_event + HANG_SECONDS):
            return f"controller: {self.phases['controller'].label()}"
        return None


def _red_case(suite, classname, name, outcome, message, detail=None):
    case = ET.SubElement(suite, "testcase", classname=classname, name=name)
    ET.SubElement(case, outcome, message=message).text = detail
    field = outcome + "s"
    suite.set(field, str(int(suite.get(field, "0")) + 1))
    suite.set("tests", str(int(suite.get("tests", "0")) + 1))


def _write_red(junit, failures, reason):
    # Keep pytest's receipt if it reached its writer. A killed controller can
    # leave the file absent or midway through a write.
    try:
        root = ET.parse(junit).getroot()
    except (FileNotFoundError, ET.ParseError):
        root = ET.Element("testsuites")
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        suite = ET.SubElement(root, "testsuite", name="pytest", tests="0", failures="0", errors="0")
    reported = {
        (case.get("classname"), case.get("name"), result.tag)
        for case in suite.findall("testcase")
        for result in case
        if result.tag in {"failure", "error"}
    }
    for (node, phase), detail in sorted(failures.items()):
        *parents, name = mangle_test_address(node)
        classname, name = bin_xml_escape(".".join(parents)), bin_xml_escape(name)
        outcome = "failure" if phase == "call" else "error"
        if (classname, name, outcome) not in reported:
            _red_case(suite, classname, name, outcome, phase, bin_xml_escape(detail))
    visible = bin_xml_escape(reason)
    _red_case(suite, "public.shard", visible, "error", visible)
    staging = junit.with_suffix(".staging")
    ET.ElementTree(root).write(staging, encoding="utf-8", xml_declaration=True)
    staging.replace(junit)


def _process_alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def run_shard(args, junit, diagnostics, budget):
    """Run pytest in its own process tree and finish a stalled shard as a named red."""
    from alphalattice.kernel.shared_kernel import spawn, waits

    began = waits.clock()
    cutoff = began + (waits.bound(budget, ceiling=55 * 60) if budget > 0 else 0)
    watch = ShardWatch(began)
    diagnostics.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(Path(__file__).resolve().parent), env.get("PYTHONPATH")))
    )
    env["PYTEST_WORKER_DIAGNOSTICS"] = str(diagnostics)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "public_test_shard",
        *args,
        "--junitxml",
        str(junit),
    ]
    reason = "public shard deadline: pytest startup"
    code = 1
    if cutoff > began:
        reason = None
        try:
            with (
                spawn.reap_detached_sessions() as keep,
                spawn.InteractiveSession(timeout=cutoff - began + 1, stop_scope="tree") as session,
            ):
                child = session.start(
                    command, env=env, stdout=None, stderr=None, stdin=subprocess.DEVNULL
                )
                # an orphan its tests leave is reaped as it ends, never a zombie a wait outlasts
                keep(child.pid)
                while child.poll() is None:
                    watch.read(diagnostics)
                    now = waits.clock()
                    phase = watch.expired(now, cutoff=cutoff, alive=_process_alive)
                    if phase is not None:
                        cause = "deadline" if now >= cutoff else "timeout"
                        reason = f"public shard {cause}: {phase}"
                        break
                    waits.spaced(now, cutoff)
                if reason is None:
                    code = child.returncode
        except spawn.TreeNotStopped as error:
            reason = f"{reason or 'public shard process exit'}: {error}"
        except (OSError, ValueError) as error:
            reason = f"public shard diagnostics unavailable: {type(error).__name__}"
    try:
        watch.read(diagnostics)
    except (OSError, ValueError) as error:
        reason = reason or f"public shard diagnostics unavailable: {type(error).__name__}"
    if reason is None and not junit.is_file():
        phase = watch.phases["controller"].label()
        reason = f"pytest exited {code} during {phase} without a JUnit receipt"
    if reason is not None:
        _write_red(junit, watch.failures, reason)
        print(reason, file=sys.stderr, flush=True)
        return 1
    return code


def main():
    """Read the job's absolute cutoff and forward the agreed pytest arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--budget", required=True, type=float)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    return run_shard(
        arguments.pytest_args[1:] if arguments.pytest_args[:1] == ["--"] else arguments.pytest_args,
        arguments.junit.resolve(),
        Path(os.environ["PYTEST_WORKER_DIAGNOSTICS"]).resolve(),
        arguments.budget,
    )


if __name__ == "__main__":
    raise SystemExit(main())
