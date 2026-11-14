"""The test noun's pytest plugin: each collected test's registered size, owner and bound."""

from __future__ import annotations

import faulthandler
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from contextlib import suppress
from math import isfinite
from pathlib import Path
from types import CodeType
from typing import Any, TextIO

import pytest

from devtools.testing import audit, requirements, sizes
from devtools.testing.contracts import Backends, LayerContract
from devtools.testing.sizes import (
    _rule,
    admission_limit,
    changed_paths,
    classify,
    included,
    load_policy,
    moved,
    scoped,
    stage_files,
    tmp_entry_footprint,
    tmp_footprint,
)

REQUIRED = pytest.StashKey[tuple[tuple[str, ...], tuple[str, ...]]]()
CHANGED = pytest.StashKey[frozenset[str] | None]()  # a scoped run's changed paths, read once
CONTRACT = pytest.StashKey[LayerContract]()
OBSERVATION = pytest.StashKey[audit.ResourceObservation | None]()
OBSERVATION_PHASES = pytest.StashKey[dict[str, float]]()
OBSERVATION_STREAM = pytest.StashKey[TextIO | None]()
PHASES = pytest.StashKey[dict[str, float]]()
FAILED = pytest.StashKey[bool]()
FRAMEWORK = pytest.StashKey[frozenset[CodeType]]()
BACKENDS = pytest.StashKey[Backends]()
SIZE_FAMILIES = pytest.StashKey[tuple[str, ...]]()
SIZE_SELECTED = pytest.StashKey[bool]()
SIZE_FOOTPRINT = pytest.StashKey[tuple[int | None, float, str]]()
SIZE_TMP_PATH = pytest.StashKey[Path]()
SIZE_CREATED = pytest.StashKey[tuple[Path, frozenset[Path]] | str]()
SIZE_RECORDED = pytest.StashKey[bool]()
SIZE_FAILED = pytest.StashKey[bool]()
INSTRUCTION_COST = pytest.StashKey[bool]()
SIZE_POLICY = pytest.StashKey[dict[str, Any]]()
SIZE_CLOCK = pytest.StashKey[dict[str, float]]()
SESSION_FOOTPRINT = pytest.StashKey[sizes.SessionFootprint]()
FIND = 'register its requirement first (python scripts/tests_registry.py find "<words>")'


def _instruction_cost(config: Any) -> bool:
    """Only the advisory simulation omits wall-clock test limits, never a gating lane."""
    if (
        os.environ.get("ALPHALATTICE_CODSPEED_NO_TIME_BOUNDS") != "1"
        or os.environ.get("ALPHALATTICE_PIPELINE_STAGE")
        or os.environ.get("ALPHALATTICE_TEST_LANE")
    ):
        return False
    plugin = config.pluginmanager.get_plugin("codspeed_plugin")
    return bool(
        plugin
        and getattr(plugin, "is_codspeed_enabled", False)
        and getattr(getattr(plugin, "mode", None), "value", None) == "simulation"
    )


def _instruction_cost_timeouts(config: Any, items: list[Any]) -> None:
    if _instruction_cost(config):
        for item in items:
            item.stash[INSTRUCTION_COST] = True
            item.add_marker(pytest.mark.timeout(0, func_only=False), append=False)
            item.user_properties.append(
                ("test_timeout_mode", "off:codspeed-instruction-simulation")
            )


def pytest_report_header(config: Any) -> str | None:
    if _instruction_cost(config):
        return "CodSpeed instruction simulation: test wall-time bounds off; job deadline remains"
    return None


def _family(item: Any) -> str:
    """One collected parameter belongs to its static test-function family."""
    path = item.nodeid.split("::", 1)[0].replace("\\", "/")
    name = getattr(item, "originalname", None) or item.name.partition("[")[0]
    cls = getattr(item, "cls", None)
    return path + "::" + ((cls.__qualname__.replace(".", "::") + "::") if cls else "") + name  # type: ignore[no-any-return]  # pytest supplies a string name and class.


def _size_families(config: Any) -> tuple[str, ...] | None:
    """Share the running receipt's pinned workspace observation with workers."""
    events = os.environ.get("ALPHALATTICE_PIPELINE_EVENTS")
    if (
        os.environ.get("ALPHALATTICE_PIPELINE_STAGE") != "commit"
        or not events
        or os.environ.get("PYTEST_CURRENT_TEST")
    ):
        return None  # a pytest launched inside a test is not the stage's test row
    try:
        if hasattr(config, "workerinput"):
            if config.workerinput.get("alphalattice_size_scope") is False:
                return None  # a nested controller did not select the stage row
            if config.workerinput.get("alphalattice_size_scope") is not True:
                raise ValueError("the controller did not identify the size scope")
            values = config.workerinput["alphalattice_size_families"]
        else:
            owner = config.pluginmanager.get_plugin("devtools.pipeline.receipt")
            if owner is None and "ALPHALATTICE_PIPELINE_REQUIRED_TESTS" not in os.environ:
                return None  # the fast guard removes test-row selection before its own pytest
            if owner is None or not callable(getattr(owner, "size_families", None)):
                raise ValueError("the running receipt plugin is unavailable")
            values = owner.size_families(Path(events))
        return requirements.selection(values)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise pytest.UsageError(
            f"test_size.changed_unmeasured: restore the commit stage's source identity: {error}"
        ) from error


@pytest.hookimpl(optionalhook=True)  # type: ignore[untyped-decorator]
def pytest_configure_node(node: Any) -> None:
    node.workerinput["session_footprint"] = SESSION_FOOTPRINT in node.config.stash
    families = node.config.stash.get(SIZE_FAMILIES, None)
    if os.environ.get("ALPHALATTICE_PIPELINE_STAGE") == "commit" and os.environ.get(
        "ALPHALATTICE_PIPELINE_EVENTS"
    ):
        node.workerinput["alphalattice_size_scope"] = families is not None
        if families is not None:
            node.workerinput["alphalattice_size_families"] = list(families)


def worker_crash_evidence(worker: Any, node: str | None, directory: Path) -> dict[str, Any]:
    """Read the worker exit and fault stream; an absent value stays unknown."""
    worker_id = worker.gateway.id
    if not isinstance(worker_id, str) or re.fullmatch(r"gw[0-9]+", worker_id) is None:
        raise ValueError("testing.worker_identity_unknown: invalid xdist worker identity")
    record: dict[str, Any] = {"worker": worker_id, "node": node, "exit": None, "stack": None}
    try:
        process = worker.gateway._io.popen
        exit_code = process.wait(timeout=2)
        if type(exit_code) is not int:
            raise ValueError("process did not provide an exit code")
        record["exit"] = exit_code
    except (AttributeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        record["exit_error"] = f"worker exit unavailable: {type(error).__name__}: {error}"
    try:
        stack = (directory / f"worker-{worker_id}.fault.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if not stack:
            raise ValueError("fault stream is empty")
        record["stack"] = stack
    except (OSError, ValueError) as error:
        record["stack_error"] = f"fault stream unavailable: {error}"
    return record


@pytest.hookimpl(trylast=True)  # type: ignore[untyped-decorator]
def pytest_sessionstart(session: Any) -> None:
    """Put each managed suite worker's native fault output in its attempt directory."""
    config = session.config
    worker = getattr(config, "workerinput", None)
    if not os.environ.get("WK_SUITE_ATTEMPT") or not isinstance(worker, dict):
        return
    worker_id = worker.get("workerid")
    xml = getattr(config.option, "xmlpath", None)
    if not isinstance(worker_id, str) or re.fullmatch(r"gw[0-9]+", worker_id) is None or not xml:
        raise pytest.UsageError(
            "testing.worker_capture_unavailable: worker identity or JUnit path missing"
        )
    try:
        stream = (Path(xml).parent / f"worker-{worker_id}.fault.log").open("x", encoding="utf-8")
        faulthandler.enable(file=stream, all_threads=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise pytest.UsageError(f"testing.worker_capture_unavailable: {error}") from error
    config._worker_fault_stream = stream


@pytest.hookimpl(optionalhook=True, tryfirst=True)  # type: ignore[untyped-decorator]
def pytest_testnodedown(node: Any, error: Any) -> None:
    """Remember each xdist process and capture a crash before its scheduler entry goes away."""
    if not os.environ.get("WK_SUITE_ATTEMPT"):
        return
    config = node.config
    workers = getattr(config, "_worker_capture_nodes", None)
    if workers is None:
        workers = config._worker_capture_nodes = {}
    scheduler = getattr(config.pluginmanager.getplugin("dsession"), "sched", None)
    pending = getattr(scheduler, "node2pending", {}).get(node, ())
    collection = getattr(scheduler, "collection", None)
    running = collection[pending[0]] if error is not None and pending and collection else None
    workers[node.gateway.id] = (node, running, error is not None)
    xml = getattr(config.option, "xmlpath", None)
    if error is not None and xml:
        node._worker_crash_evidence = worker_crash_evidence(node, running, Path(xml).parent)


@pytest.hookimpl(optionalhook=True, trylast=True)  # type: ignore[untyped-decorator]
def pytest_handlecrashitem(crashitem: str, report: Any, sched: Any) -> None:
    """Carry the actual exit and fault stream in the failed node's existing JUnit record."""
    if not os.environ.get("WK_SUITE_ATTEMPT"):
        return
    record = getattr(getattr(report, "node", None), "_worker_crash_evidence", None)
    if not isinstance(record, dict) or record.get("node") != crashitem:
        report.longrepr = f"{report.longrepr}\nworker capture unmeasured: node identity unavailable"
        return
    exit_value = record.get("exit")
    stack = record.get("stack") or record.get("stack_error", "fault stream unavailable")
    exit_text = exit_value if exit_value is not None else "unknown"
    report.longrepr = f"{report.longrepr}\nworker {record['worker']} exit {exit_text}\n{stack}"


def ordered_requeue() -> type:
    """xdist's load scheduler, except that a crashed worker's prefetched cases return to their
    place in the collection's longest-first order. xdist appends them last; on one nightly that
    started two cases of over 900 s after the other workers had nearly run out of work."""
    from xdist.scheduler import LoadScheduling  # type: ignore[import-untyped]  # lazy: only -n

    class OrderedRequeue(LoadScheduling):  # type: ignore[misc]
        def remove_node(self, node: Any) -> str | None:
            # pytest-xdist 3.8's remove_node, pinned in pyproject; only the requeue is sorted.
            pending = self.node2pending.pop(node)
            if not pending:
                return None
            assert self.collection is not None
            crashed: str = self.collection[pending.pop(0)]
            self.pending[:] = sorted([*self.pending, *pending])
            for other in self.node2pending:
                self.check_schedule(other)
            return crashed

    return OrderedRequeue


@pytest.hookimpl(optionalhook=True)  # type: ignore[untyped-decorator]
def pytest_xdist_make_scheduler(config: Any, log: Any) -> Any:
    """xdist, in the main process: load keeps its order through a crash; others are xdist's."""
    if config.getoption("dist", "no") != "load":
        return None
    return ordered_requeue()(config, log)


@pytest.hookimpl(wrapper=True, trylast=True)  # type: ignore[untyped-decorator]
def pytest_fixture_setup(fixturedef: Any, request: Any) -> Any:
    clock = request.config.stash.get(SIZE_CLOCK, None)
    began = time.perf_counter() if clock is not None else 0.0
    item = request.node
    selected = fixturedef.argname == "tmp_path" and item.stash.get(SIZE_SELECTED, False)
    observer = request.config.stash.get(SESSION_FOOTPRINT, None)
    snapshot = None
    if observer is not None and fixturedef.scope == "session":
        try:
            snapshot = observer.before_fixture(fixturedef, request)
        except Exception as error:
            observer.errors.append("fixture setup observation unavailable: " + str(error))
    before = time.perf_counter() - began if clock is not None else 0.0
    try:
        value = yield
    except BaseException as error:
        if observer is not None and snapshot is not None:
            try:
                observer.failed_fixture(fixturedef, snapshot, error)
            except Exception as failure:
                observer.errors.append("failed fixture observation unavailable: " + str(failure))
        raise
    began = time.perf_counter() if clock is not None else 0.0
    if observer is not None and snapshot is not None:
        try:
            observer.after_fixture(fixturedef, request, snapshot)
        except Exception as error:
            observer.errors.append("fixture observation unavailable: " + str(error))
    if selected:

        def measure() -> None:
            started = time.perf_counter()
            try:
                size, error = tmp_footprint(value), ""
            except (OSError, ValueError) as failure:
                size, error = None, str(failure)
            elapsed = time.perf_counter() - started
            item.stash[SIZE_FOOTPRINT] = (size, elapsed, error)
            item.stash[SIZE_TMP_PATH] = value
            if clock is not None:
                clock["footprint_seconds"] += elapsed

        request.addfinalizer(measure)
    if clock is not None:
        clock["fixture_hook_seconds"] += before + time.perf_counter() - began
    return value


def _proven(
    item: Any, rows: dict[str, dict[str, Any]], policy: dict[str, Any]
) -> tuple[list[str], dict[str, Any], list[str], tuple[str, str, str, float]]:
    """The ids a test proves, its first requirement, its size markers and its classification.
    A node-specific size wins over the layer default; an unknown or retired id refuses."""
    ids = [str(arg) for mark in item.iter_markers("proves") for arg in mark.args]
    known = [rows[key] for key in ids if key in rows and not rows[key].get("retired")]
    if len(known) != len(ids):
        raise ValueError(f"requirements.unknown_id:{item.nodeid} proves {ids}; {FIND}")
    row = known[0] if known else {}
    primary = requirements.primary_layer(
        item.nodeid, {key: row["layer"] for key, row in zip(ids, known, strict=True)}
    )
    if primary:
        row = row | {"layer": primary}
    declared = [m.name for m in item.iter_markers() if m.name in policy["sizes"]]
    explicit = _rule(item.nodeid, policy["defaults"])
    layer = (
        []
        if _rule(item.nodeid, policy["large"]) or (explicit and "::" in explicit["pattern"])
        else [requirements.SIZES[row["layer"]]]
        if row
        else []
    )
    return ids, row, declared, classify(item.nodeid, declared + layer, policy)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)  # type: ignore[untyped-decorator]
def pytest_runtest_protocol(item: Any) -> Any:
    """Observe only managed pipeline tests; telemetry failure never changes their outcome."""
    scope = None
    entered = False
    if item.stash.get(SIZE_SELECTED, False):
        try:
            root = item.config._tmp_path_factory.getbasetemp()
            item.stash[SIZE_CREATED] = (root, frozenset(root.iterdir()))
        except (OSError, ValueError) as failure:
            item.stash[SIZE_CREATED] = str(failure)
    if os.environ.get("ALPHALATTICE_PIPELINE_EVENTS"):
        try:
            observation = audit.ResourceObservation()
            item.stash[OBSERVATION] = observation
            try:
                scope = audit.subscribe(observation.note)
                scope.__enter__()
                entered = True
            except Exception:
                observation.available = False
        except Exception:
            with suppress(Exception):
                item.stash[OBSERVATION] = None
    try:
        yield
    finally:
        if entered and scope is not None:
            with suppress(Exception):
                scope.__exit__(None, None, None)


def _measure_factory(item: Any) -> None:
    """Count new worker-base entries; existing shared directories are not attributed."""
    started = time.perf_counter()
    prior = item.stash.get(SIZE_FOOTPRINT, None)
    measured = item.stash.get(SIZE_TMP_PATH, None)
    try:
        if prior is not None and (prior[0] is None or prior[2]):
            raise ValueError(prior[2] or "temporary footprint observation missing")
        snapshot = item.stash[SIZE_CREATED]
        if isinstance(snapshot, str):
            raise ValueError(snapshot)
        root, before = snapshot
        size = (prior[0] if prior else 0) + sum(
            tmp_entry_footprint(path) for path in set(root.iterdir()) - before if path != measured
        )
        error = ""
    except (OSError, ValueError, KeyError) as failure:
        size, error = None, str(failure)
    elapsed = time.perf_counter() - started
    item.stash[SIZE_FOOTPRINT] = (size, elapsed + (prior[1] if prior else 0), error)
    clock = item.config.stash.get(SIZE_CLOCK, None)
    if clock is not None:
        clock["footprint_seconds"] += elapsed


@pytest.hookimpl(wrapper=True, trylast=True)  # type: ignore[untyped-decorator]
def pytest_runtest_setup(item: Any) -> Any:
    if OBSERVATION in item.stash:
        with suppress(Exception):
            item.stash[OBSERVATION].phase = "setup"
    level = dict(item.user_properties).get("requirement_layer") or requirements.layer(item.nodeid)
    if (
        item.stash.get(SIZE_SELECTED, False)
        and dict(item.user_properties).get("test_size") == "small"
    ):
        level = "unit"  # Small's no-subprocess/no-network contract uses the existing unit owner.
    if level not in {"unit", "integration"}:
        return (yield)
    contract = LayerContract(
        level, item.config._tmp_path_factory.getbasetemp(), item.config.stash.get(BACKENDS, None)
    )
    item.stash[CONTRACT] = contract
    definitions = item._fixtureinfo.name2fixturedefs
    functions = [values[-1].func for values in definitions.values() if values]
    framework = item.config.stash.get(FRAMEWORK, frozenset())
    contract.references(
        [item.obj, *(f for f in functions if getattr(f, "__code__", None) not in framework)]
    )
    with contract.active():
        contract.references(
            [f for f in functions if getattr(f, "__code__", None) in framework],
            module_inventory=framework,
        )
        return (yield)


@pytest.hookimpl(wrapper=True, trylast=True)  # type: ignore[untyped-decorator]
def pytest_runtest_call(item: Any) -> Any:
    if OBSERVATION in item.stash:
        with suppress(Exception):
            item.stash[OBSERVATION].phase = "call"
    if CONTRACT not in item.stash:
        return (yield)
    contract = item.stash[CONTRACT]
    contract.references(list(item.funcargs.values()))
    with contract.active():
        return (yield)


@pytest.hookimpl(wrapper=True, trylast=True)  # type: ignore[untyped-decorator]
def pytest_runtest_teardown(item: Any) -> Any:
    if OBSERVATION in item.stash:
        with suppress(Exception):
            item.stash[OBSERVATION].phase = "teardown"
    try:
        if CONTRACT not in item.stash:
            return (yield)
        with item.stash[CONTRACT].active():
            return (yield)
    finally:
        if item.stash.get(SIZE_SELECTED, False):
            _measure_factory(item)


def _observe_report(item: Any, report: Any) -> None:
    """Write one worker's node observations beside its JUnit, or leave an explicit gap."""
    if OBSERVATION not in item.stash:
        return
    phases = item.stash.setdefault(OBSERVATION_PHASES, {})
    if report.when in {"setup", "call", "teardown"}:
        phases[report.when] = report.duration
    if report.when != "teardown":
        return
    stream = item.config.stash.get(OBSERVATION_STREAM, None)
    if stream is None:
        return
    try:
        observation = item.stash[OBSERVATION]
        declared = item.stash[CONTRACT].layer if CONTRACT in item.stash else None
        value = (
            observation.snapshot(declared)
            if observation is not None
            else {"observed_layer": "unknown", "subprocesses": {}, "unknown": ["audit unavailable"]}
        )
        value.update(
            node=item.nodeid,
            phases={name: phases.get(name) for name in ("setup", "call", "teardown")},
        )
        if item.stash.get(SIZE_SELECTED, False):
            footprint = item.stash.get(SIZE_FOOTPRINT, None)
            value.update(
                tmp_footprint_bytes=footprint[0] if footprint else None,
                tmp_footprint_seconds=footprint[1] if footprint else None,
                tmp_footprint_error=footprint[2]
                if footprint
                else "temporary footprint observation missing",
            )
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        if item.stash.get(SIZE_SELECTED, False):
            item.stash[SIZE_RECORDED] = True
    except Exception as error:
        with suppress(Exception):
            print(f"test observation unavailable: {error}", file=sys.stderr)


def _size_report(item: Any, report: Any) -> None:
    """Refuse a changed test only after its complete phases and tmp_path have been observed."""
    if not item.stash.get(SIZE_SELECTED, False):
        return
    if report.skipped:
        report.outcome, report.longrepr = (
            "failed",
            f"test_size.admission_unmeasured: {item.nodeid}: the edited test was skipped; "
            "run it to measure its size",
        )
    if report.failed:
        item.stash[SIZE_FAILED] = True
    if report.when != "teardown" or item.stash.get(SIZE_FAILED, False):
        return
    phases = item.stash.get(OBSERVATION_PHASES, {})
    footprint = item.stash.get(SIZE_FOOTPRINT, None)
    try:
        if (
            set(phases) != {"setup", "call", "teardown"}
            or not item.stash.get(SIZE_RECORDED, False)
            or footprint is None
            or footprint[0] is None
            or footprint[2]
        ):
            raise ValueError(
                f"test_size.admission_unmeasured: {item.nodeid}: restore complete phases "
                "and tmp footprint at teardown"
            )
        policy = item.config.stash[SIZE_POLICY]
        size = dict(item.user_properties)["test_size"]
        seconds = sum(phases.values())
        bound = policy.get("admission_limits", {}).get(size, {})
        admission_limit(item.nodeid, size, seconds, footprint[0], policy)
    except (KeyError, TypeError, ValueError) as error:
        report.outcome, report.longrepr = "failed", str(error)
        if (
            str(error).startswith("test_size.admission_limit:")
            and footprint[0] <= bound["tmp_bytes"]
        ):
            # Over its seconds only: the gate measures it alone, as a timeout (lead_check retry).
            limit = bound["seconds"]
            report.longrepr += _retry_evidence(item, "teardown", limit, seconds, "admission")


def _golden_times(item: Any) -> None:
    """What a case's goldens cost, in its JUnit: a fill holds its nested parents, whose lock waits
    count as waits, so fill, lock and copy add up. Read from the owner only if a test loaded it."""
    owner = sys.modules.get("devtools.evaluation.golden_workspaces")
    if rows := owner.take_acquisitions() if owner else []:
        locks = sum(row["lock_seconds"] for row in rows)
        outer = [row for row in rows if not row["nested"]]
        waited = sum(row["lock_seconds"] for row in rows if row["nested"])
        fill = max(0.0, sum(row["fill_seconds"] for row in outer) - waited)  # 0 if the fill failed
        copy = sum(row["copy_seconds"] for row in outer)
        names = ",".join(f"{row['recipe']}:{'fill' if row['cold'] else 'copy'}" for row in rows)
        item.user_properties.extend(
            (
                ("golden_fill_seconds", round(fill, 3)),
                ("golden_lock_seconds", round(locks, 3)),
                ("golden_copy_seconds", round(copy, 3)),
                ("goldens", names),
            )
        )


def pytest_terminal_summary(terminalreporter: Any) -> None:
    """Report measured golden fills in Large tests from their existing teardown records."""
    fills: list[str] = []
    for reports in terminalreporter.stats.values():
        for report in reports:
            if getattr(report, "when", None) != "teardown":
                continue
            properties = dict(getattr(report, "user_properties", ()))
            value = properties.get("golden_fill_seconds")
            if (
                properties.get("test_size") == "large"
                and isinstance(value, int | float)
                and isfinite(value)
                and value > 0
            ):
                fills.append(
                    f"{report.nodeid}: golden fill {value:.3f} s ({properties.get('goldens', '')})"
                )
    if fills:
        terminalreporter.write_sep("=", "Golden fills in Large tests")
        for line in sorted(fills):
            terminalreporter.write_line(line)


def _timeline(item: Any, report: Any) -> None:
    """The run's timeline in its JUnit: which worker ran each case, from when to when."""
    if report.when == "call":
        return
    edge = ("started_at", report.start) if report.when == "setup" else ("ended_at", report.stop)
    item.user_properties.append((edge[0], round(edge[1], 3)))
    if report.when == "setup":
        item.user_properties.append(("worker", os.environ.get("PYTEST_XDIST_WORKER", "main")))
    else:
        _golden_times(item)
    report.user_properties = list(item.user_properties)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)  # type: ignore[untyped-decorator]
def pytest_runtest_makereport(item: Any, call: Any) -> Any:
    """Judge unit cost from its enclosing phases, using the existing solo-timeout route."""
    result = yield
    report = result.get_result()
    clock = item.config.stash.get(SIZE_CLOCK, None)
    began = (
        time.perf_counter() if clock is not None and item.stash.get(SIZE_SELECTED, False) else 0.0
    )
    _observe_report(item, report)
    if began:
        clock["record_seconds"] += time.perf_counter() - began
    began = time.perf_counter() if clock is not None else 0.0
    _size_report(item, report)
    if clock is not None:
        clock["report_seconds"] += time.perf_counter() - began
    _timeline(item, report)
    if CONTRACT not in item.stash or item.stash[CONTRACT].layer != "unit":
        return
    if report.failed:
        item.stash[FAILED] = True
    phases = item.stash.setdefault(PHASES, {})
    phases[report.when] = report.duration
    item.user_properties.append((report.when + "_seconds", report.duration))
    report.user_properties = list(item.user_properties)
    if report.when != "teardown" or item.stash.get(FAILED, False):
        return
    measured = set(phases) == {"setup", "call", "teardown"} and all(
        type(v) in (int, float) and isfinite(v) and v >= 0 for v in phases.values()
    )
    if not measured:
        report.outcome, report.longrepr = (
            "failed",
            (
                "testing.layer_contract_unmeasured: incomplete or invalid own phases; "
                "rerun the exact unit node"
            ),
        )
        return
    seconds = sum(phases.values())
    item.user_properties.append(("unit_own_seconds", seconds))
    report.user_properties = list(item.user_properties)
    limit = dict(item.user_properties)["test_timeout_seconds"]
    if seconds <= limit or item.stash.get(INSTRUCTION_COST, False):
        return
    report.outcome, report.longrepr = (
        "failed",
        (
            f"testing.layer_contract: unit own phases {seconds:g} s exceed {limit:g} s; "
            "share setup or move the test to integration"
        ),
    )
    report.longrepr += _retry_evidence(item, "teardown", limit, seconds, "unit-own-phases")


def _retry_evidence(item: Any, phase: str, limit: float, seconds: float, cause: str) -> str:
    """A node failed on its seconds: a TIMEOUT event in the run's log asks the gate to run it
    alone, which judges it again; an unwritable log is named in the failure."""
    event = dict(event="TIMEOUT", nodeid=item.nodeid, phase=phase, limit_seconds=limit)
    event.update(elapsed_seconds=seconds, cause=cause)
    log_file = item.config.getoption("log_file") or item.config.getini("log_file")
    if log_file:
        try:
            if receipts := item.config.pluginmanager.get_plugin("pytest_receipts"):
                slot = (
                    receipts.heavy_slot
                )  # the shared places, read as its own TIMEOUT event reads them
                event["heavy_place_holders"] = slot._read(slot.HOLDERS, None)
            with Path(log_file).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
        except OSError as error:
            return f"; testing.layer_contract_unmeasured: retry evidence: {error}"
    return ""


def gate(
    lane: str, marks: list[tuple[str, list[str], Path]], counted: dict[str, int] | None, suite: Path
) -> None:
    """An unregistered test of this checkout's suite does not run in a gate: refuse files holding
    more unmarked test functions than their legacy count. A child session's own tests are not the
    suite's, though it inherits the lane."""
    unmarked = {
        node.partition("[")[0]
        for node, ids, path in marks
        if not ids and path.is_relative_to(suite)
    }
    files = Counter(node.split("::", 1)[0] for node in unmarked)
    over = sorted(path for path, count in files.items() if count > (counted or {}).get(path, 0))
    if lane and over and counted is not None:  # the public export has no list
        raise ValueError(f"requirements.unregistered:{', '.join(over)}: a new test; {FIND}")


def _properties(
    item: Any,
    ids: list[str],
    row: dict[str, Any],
    declared: list[str],
    classified: tuple[str, str, str, float],
    *,
    size_hang_guard: float | None = None,
) -> None:
    """Attach each owner's existing size, real bound and requirement metadata once."""
    size, owner, declaration, seconds = classified
    if not declared:
        item.add_marker(size)
    previous = item.get_closest_marker("timeout")
    if size_hang_guard is not None and previous is None:
        seconds = max(
            seconds, size_hang_guard
        )  # admission measures completed phases; this is only a hang guard
    if previous:
        limit = previous.args[0] if previous.args else previous.kwargs.get("timeout")
        if isinstance(limit, (int, float)) and limit > 0:
            seconds = min(seconds, limit)
    item.add_marker(pytest.mark.timeout(seconds, func_only=False), append=False)
    item.user_properties.extend(
        [
            ("test_size", size),
            ("test_size_owner", owner),
            ("test_size_declaration", declaration),
            ("test_timeout_seconds", seconds),
            ("requirements", ids),
            ("requirement_layer", row.get("layer")),
            ("requirement_owner", row.get("owner")),
        ]
    )


def _nested_required(required: tuple[str, ...]) -> tuple[str, ...]:
    """Do not force a parent's changed families into its child pytest sessions."""
    events = os.environ.get("ALPHALATTICE_PIPELINE_EVENTS")
    if (
        not required
        or not events
        or os.environ.get("ALPHALATTICE_PIPELINE_STAGE") != "commit"
        or not os.environ.get("PYTEST_CURRENT_TEST")
    ):
        return required
    try:
        record = json.loads((Path(events).parent / "size-stage-selection.json").read_text("utf-8"))
        inherited = frozenset(requirements.selection(record["families"]))
        return tuple(node for node in required if node not in inherited)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise pytest.UsageError(
            f"test_size.changed_unmeasured: restore the parent stage selection: {error}"
        ) from error


def _session_footprint_enabled(config: Any) -> bool:
    """Recognize the suite plugin and its own product pytest session."""
    plugin = os.environ.get("WK_SUITE_PLUGIN", "")
    return bool(
        plugin
        and os.environ.get("WK_SUITE_REPORT")
        and not os.environ.get("PYTEST_CURRENT_TEST")
        and config.pluginmanager.has_plugin(plugin)
        and Path(config.rootpath).resolve() == Path(__file__).resolve().parents[3]
    )


def _start_session_footprint(config: Any) -> None:
    """Bind the optional observer only to the current verified suite and its workers."""
    if not _session_footprint_enabled(config):
        return
    worker_input = getattr(config, "workerinput", None)
    owner_key = "ALPHALATTICE_SESSION_FOOTPRINT_OWNER"
    if (worker_input is None and os.environ.get(owner_key)) or (
        worker_input is not None and worker_input.get("session_footprint") is not True
    ):
        return
    token = ""
    try:
        xml = getattr(config.option, "xmlpath", None)
        directory = Path(xml).parent if xml else Path(os.environ["WK_SUITE_REPORT"]).parent
        worker = worker_input.get("workerid", "main") if worker_input else "main"
        if not isinstance(worker, str) or not worker.isalnum():
            raise ValueError("invalid worker identity")
        base = getattr(config.option, "basetemp", None)
        root = Path(base).absolute() if base else None
        if worker != "main" and root is not None and root.name == "popen-" + worker:
            root = root.parent
        if worker_input is None:
            token = f"{os.getpid()}:{os.environ.get('WK_SUITE_ATTEMPT', '')}"
            os.environ[owner_key] = token
        observer = sizes.SessionFootprint(
            root,
            directory,
            worker,
            os.environ.get("WK_SUITE_ATTEMPT", ""),
            os.environ.get("WK_SUITE_SOURCE_SHA", ""),
        )
        observer.owner_token = token
        config.stash[SESSION_FOOTPRINT] = observer
    except Exception as error:
        if token and os.environ.get(owner_key) == token:
            os.environ.pop(owner_key, None)
        print(f"test_size.session_footprint_unmeasured: {error}", file=sys.stderr)


def pytest_configure(config: Any) -> None:
    """The producer's optional required nodes override only their own ordinary deselection."""
    value = os.environ.get("ALPHALATTICE_PIPELINE_REQUIRED_TESTS", "[]")
    source = "inline list"
    if value.startswith("@"):
        path = Path(value[1:])
        source = str(path)
        if not path.is_absolute():
            raise pytest.UsageError(
                f"test_size.required_selection_unavailable: {path}; name an absolute list file"
            )
        try:
            value = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError) as error:
            raise pytest.UsageError(
                f"test_size.required_selection_unavailable: {path}: {error}; "
                "restore the required list"
            ) from error
    try:
        required = requirements.selection(json.loads(value))
    except (ValueError, TypeError) as error:
        raise pytest.UsageError(
            f"test_size.required_selection_invalid: {source}: {error}; restore the required list"
        ) from None
    required = _nested_required(required)
    deselect = tuple(config.option.deselect or ()) if required else ()
    if required:
        config.option.deselect = []
    config.stash[REQUIRED] = required, deselect
    began = time.perf_counter()
    families = _size_families(config)
    if families is not None:
        config.stash[SIZE_FAMILIES] = families
        if families:
            config.stash[SIZE_CLOCK] = {
                "selection_seconds": time.perf_counter() - began,
                "fixture_hook_seconds": 0.0,
                "footprint_seconds": 0.0,
                "collection_seconds": 0.0,
                "record_seconds": 0.0,
                "report_seconds": 0.0,
            }
    if os.environ.get("ALPHALATTICE_PIPELINE_EVENTS"):
        try:
            xml = getattr(config.option, "xmlpath", None)
            directory = (
                Path(xml).parent if xml else Path(os.environ["ALPHALATTICE_PIPELINE_EVENTS"]).parent
            )
            worker = getattr(config, "workerinput", {}).get("workerid", "main")
            worker = worker if isinstance(worker, str) and worker.isalnum() else str(os.getpid())
            config.stash[OBSERVATION_STREAM] = (
                directory / f"node-observations-{worker}.jsonl"
            ).open("a", encoding="utf-8")
        except Exception as error:
            with suppress(Exception):
                print(f"test observation unavailable: {error}", file=sys.stderr)
    _start_session_footprint(config)


@pytest.hookimpl(trylast=True)  # type: ignore[untyped-decorator]
def pytest_sessionfinish(session: Any) -> None:
    """Retain component costs once per process; missing evidence cannot prove the stage budget."""
    config = session.config
    workers = getattr(config, "_worker_capture_nodes", None)
    if workers is not None and os.environ.get("WK_SUITE_ATTEMPT"):
        try:
            directory = Path(config.option.xmlpath).parent
            records = [
                worker_crash_evidence(node, running, directory)
                for node, running, _ in workers.values()
            ]
            data = {
                "schema": "alphalattice.pytest-workers.v1",
                "attempt": os.environ["WK_SUITE_ATTEMPT"],
                "source_sha": os.environ.get("WK_SUITE_SOURCE_SHA"),
                "workers": records,
            }
            with (directory / "worker-exits.json").open("x", encoding="utf-8") as stream:
                json.dump(data, stream, allow_nan=False)
                stream.write("\n")
        except (OSError, ValueError, AttributeError, KeyError) as error:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            print(f"testing.worker_capture_unavailable: {error}", file=sys.stderr)
    observer = session.config.stash.get(SESSION_FOOTPRINT, None)
    if observer is not None:
        try:
            observer.finish()
        except Exception as error:
            print(f"test_size.session_footprint_unmeasured: {error}", file=sys.stderr)
    clock = session.config.stash.get(SIZE_CLOCK, None)
    if clock is None:
        return
    events = os.environ.get("ALPHALATTICE_PIPELINE_EVENTS")
    try:
        if not events or any(not isfinite(value) or value < 0 for value in clock.values()):
            raise ValueError("missing or invalid size-observer timings")
        path = Path(events).parent / f"size-overhead-{os.getpid()}-{time.time_ns()}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(clock, stream, allow_nan=False)
    except (OSError, ValueError) as error:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        print(
            f"test_size.overhead_unmeasured: restore the stage's timing output: {error}",
            file=sys.stderr,
        )


def pytest_unconfigure(config: Any) -> None:
    stream = getattr(config, "_worker_fault_stream", None)
    if stream is not None:
        faulthandler.disable()
        stream.close()
    observer = config.stash.get(SESSION_FOOTPRINT, None)
    if observer is not None:
        try:
            observer.finish()
        except Exception as error:
            print(f"test_size.session_footprint_unmeasured: {error}", file=sys.stderr)
        owner_key = "ALPHALATTICE_SESSION_FOOTPRINT_OWNER"
        if observer.owner_token and os.environ.get(owner_key) == observer.owner_token:
            os.environ.pop(owner_key, None)
    stream = config.stash.get(OBSERVATION_STREAM, None)
    if stream is not None:
        with suppress(Exception):
            stream.close()


def _required_selection(config: Any, items: list[Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Refuse missing required families; a skipped or absent family is never passing evidence."""
    required, deselect = config.stash.get(REQUIRED, ((), ()))
    if any(not any(requirements.node_matches(i.nodeid, n) for i in items) for n in required):
        raise ValueError("test_size.required_selection_invalid")
    return required, deselect


def framework_fixtures(
    config: Any, items: list[Any], policy: dict[str, Any]
) -> frozenset[CodeType]:
    """Resolve exact registered root fixture names once, never a module or name pattern."""
    names = {row["node"] for row in policy.get("framework_fixtures", [])}
    functions = {
        values[-1].func
        for item in items
        for values in item._fixtureinfo.name2fixturedefs.values()
        if values
    }
    codes: set[CodeType] = set()
    for function in functions:
        code = getattr(function, "__code__", None)
        if code is None:
            continue
        try:
            path = Path(code.co_filename).absolute().relative_to(config.rootpath).as_posix()
        except ValueError:
            continue
        if f"{path}::{function.__qualname__}" in names:
            codes.add(code)
    return frozenset(codes)


RELEASE_ONLY = "release lane by rule (test-sizes.json): it runs at each promotion, not here"


def _size_attributes(
    item: Any,
    config: Any,
    families: frozenset[str],
    policy: dict[str, Any],
    ids: list[str],
    row: dict[str, Any],
    declared: list[str],
    classified: tuple[str, str, str, float],
) -> str | None:
    """Attach changed-test admission and return the Large rule's lane contract, if any."""
    size = classified[0]
    clock = config.stash.get(SIZE_CLOCK, None)
    began = time.perf_counter() if clock is not None else 0.0
    selected = _family(item) in families
    item.stash[SIZE_SELECTED] = selected
    large_rule = _rule(item.nodeid, policy["large"]) if size == "large" else None
    contract = (
        str(large_rule["size_contract"]) if large_rule and "size_contract" in large_rule else None
    )
    if clock is not None:
        clock["collection_seconds"] += time.perf_counter() - began
    if selected and size == "large" and not contract:
        raise ValueError(
            "test_size.large_contract: register a Large rule with "
            f"size_contract=nightly_only: {item.nodeid}"
        )
    _properties(
        item,
        ids,
        row,
        declared,
        classified,
        size_hang_guard=policy["limits"]["medium"] if selected and size == "small" else None,
    )
    return contract


def _lane_contract(item: Any, contract: str | None, lane: str) -> bool:
    """Whether a Large rule's lane contract omits the item here; the nightly lists a release-only
    one skipped, with its reason, so it is never a red."""
    if contract == "nightly_only" and lane != "nightly":
        print(f"test_size.nightly_only: {item.nodeid}; run in the nightly")
        return True
    if contract == "release_only" and lane == "nightly":
        item.add_marker(pytest.mark.skip(reason=RELEASE_ONLY))
    return False


def _stage_admits(stage: str, size: str, layer: str) -> bool:
    """Commit leaves slow and journey tests to their later scheduled tier."""
    return stage != "commit" or (size != "large" and layer not in {"e2e", "native"})


def _size_included(
    item: Any,
    size: str,
    lane: str,
    policy: dict[str, Any],
    changed: frozenset[str] | None,
    required: tuple[str, ...],
    deselect: tuple[str, ...],
    permitted: set[str],
    *,
    floor: bool = False,
) -> bool:
    """Keep required families and other admitted nodes under the existing lane policy."""
    return any(requirements.node_matches(item.nodeid, n) for n in required) or (
        item.nodeid.split("::", 1)[0].replace("\\", "/") in permitted
        and included(item.nodeid, size, lane, policy, changed, floor=floor)
        and not item.nodeid.startswith(deselect)
    )


FLOOR_MOVED = pytest.StashKey[dict[str, Any]]()


def _record_moved(
    config: Any,
    lane: str,
    items: list[Any],
    policy: dict[str, Any],
    changed: frozenset[str] | None,
    *,
    floor: bool = False,
) -> None:
    """Beside the run's JUnit, ``structural-moved.json``: the structural guards a commit stage
    leaves to post-merge, and whether this run ran them (a post-merge run, unscoped, always does),
    so a post-merge red among them is a red the stage no longer catches (the week's signal)."""
    xml = getattr(config.option, "xmlpath", None)
    found = sorted({n.replace("\\", "/") for n in (i.nodeid for i in items) if moved(n, policy)})
    if lane != "merge" or not xml or not found:
        return
    record = {
        "tier": "commit stage" if floor or changed is not None else "post-merge",
        "moved": found,
        "ran": [],
    }
    config.stash[FLOOR_MOVED] = record


def _write_moved(config: Any, record: dict[str, Any]) -> None:
    xml = config.option.xmlpath
    Path(xml).parent.joinpath("structural-moved.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"STRUCTURAL-MOVED {len(record['moved'])} guards leave the commit stage "
            f"for post-merge; this {record['tier']} run ran {len(record['ran'])} of them"
        )


def pytest_collection_finish(session: Any) -> None:
    """Name the moved guards actually kept after all collection hooks."""
    record = session.config.stash.get(FLOOR_MOVED, None)
    if record is not None:
        kept = {item.nodeid.replace("\\", "/") for item in session.items}
        record["ran"] = [node for node in record["moved"] if node in kept]
        _write_moved(session.config, record)


@pytest.hookimpl(wrapper=True, tryfirst=True)  # type: ignore[untyped-decorator]
def pytest_collection(session: Any) -> Any:
    """Name an xdist worker's collection refusal in the run's log. pytest sends the worker's
    collection to the controller in a finally, so a UsageError then ends the worker silently: the
    controller, which never collects, reports only a crash on a test that never ran."""
    try:
        return (yield)
    except pytest.UsageError as error:
        worker = getattr(session.config, "workerinput", {}).get("workerid")
        if worker:
            print(f"ERROR: [{worker}] {error}", file=sys.stderr, flush=True)
        raise


def pytest_itemcollected(item: Any) -> None:
    """Under CodSpeed every selected test is measured as its cost: it deselects unmarked tests."""
    if os.environ.get("CODSPEED_ENV") is not None or item.config.getoption("codspeed", False):
        item.add_marker(pytest.mark.benchmark)


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Materialize each test's registered size and declaring owner during collection."""

    lane = os.environ.get("ALPHALATTICE_TEST_LANE", "")
    floor = sizes.floor_mode(lane)
    changed = changed_paths(floor=floor) if lane in {"merge", "fast"} else None
    empty = pytest.mark.skip(reason="the change holds no Python file for this per_file scan")
    python = any(path.endswith(".py") for path in changed or ())
    config.stash[CHANGED] = changed
    kept: list[Any] = []
    omitted: list[Any] = []
    seen: set[str] = set()
    marks: list[tuple[str, list[str], Path]] = []
    families = frozenset(config.stash.get(SIZE_FAMILIES, ()))
    root = Path(__file__).resolve().parents[3]
    try:
        required, deselect = _required_selection(config, items)
        policy, rows, counted = (
            load_policy(root),
            requirements.load(root),
            requirements.legacy(root),
        )
        if lane and lane not in policy["lanes"]:
            raise ValueError("test_size.lane_unknown")
        config.stash[FRAMEWORK] = framework_fixtures(config, items, policy)
        config.stash[BACKENDS] = policy.get("unit_backends", [])
        config.stash[SIZE_POLICY] = policy
        stage = os.environ.get("ALPHALATTICE_PIPELINE_STAGE", "")
        files = {item.nodeid.split("::", 1)[0].replace("\\", "/") for item in items}
        permitted = set(
            stage_files(
                policy, stage, files, lane=lane, read=lambda path: (root / path).read_text("utf-8")
            )
        )
        for item in items:
            if (lane or required) and item.nodeid in seen:
                omitted.append(item)
                continue
            seen.add(item.nodeid)
            ids, row, declared, (size, owner, declaration, seconds) = _proven(item, rows, policy)
            marks.append((item.nodeid, ids, Path(item.path)))
            if not _stage_admits(stage, size, row.get("layer", "")):
                omitted.append(item)
                continue
            contract = _size_attributes(
                item,
                config,
                families,
                policy,
                ids,
                row,
                declared,
                (size, owner, declaration, seconds),
            )
            if _lane_contract(item, contract, lane):
                omitted.append(item)
                continue
            if scoped(item.nodeid, policy, changed, floor=floor):
                item.add_marker("per_file")  # it reads only the changed files, through scanned()
                if not python:
                    item.add_marker(empty)
            target = (
                kept
                if _size_included(
                    item, size, lane, policy, changed, required, deselect, permitted, floor=floor
                )
                else omitted
            )
            target.append(item)
        gate(lane, marks, counted, root / "tests")
    except ValueError as error:
        if str(error).startswith("test_size.large_contract:"):
            config.hook.pytest_collectreport(
                report=pytest.CollectReport(
                    nodeid=item.nodeid, outcome="failed", longrepr=str(error), result=[]
                )
            )
            items.clear()
            return
        raise pytest.UsageError(str(error)) from error
    _record_moved(config, lane, items, policy, changed, floor=floor)
    if lane or required or omitted:
        config.hook.pytest_deselected(items=omitted)
        items[:] = kept
    _instruction_cost_timeouts(config, items)


def timeout_hints(lines: list[str], failed: set[str]) -> dict[str, float]:
    """Use only this run's named failed timeout events as measured lower bounds."""
    hints: dict[str, float] = {}
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("event") != "TIMEOUT":
            continue
        node, elapsed = row.get("nodeid"), row.get("elapsed_seconds")
        if (
            not isinstance(node, str)
            or node not in failed
            or not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
        ):
            continue
        try:
            seconds = float(elapsed)
        except OverflowError:
            continue
        if isfinite(seconds) and seconds > 0:
            hints[node] = max(hints.get(node, 0.0), seconds)
    return hints


def scheduling_hints(recorded: dict[str, float], policy: dict[str, Any]) -> dict[str, float]:
    """Rank a zero-time Large case by its bound; keep the recorded measurements intact."""
    hints = dict(recorded)
    for node, seconds in recorded.items():
        if seconds != 0:
            continue
        try:
            size, _, _, bound = classify(node, (), policy)
        except ValueError:
            continue  # the normal collection owner will name a bad size declaration
        if size == "large":
            hints[node] = bound
    return hints


def merge_timing_hints(
    recorded: dict[str, float], measured: dict[str, float], failed: set[str], lines: list[str]
) -> dict[str, float]:
    """Keep failed durations conservative and retain a timed-out worker's measured lower bound."""
    merged = dict(recorded)
    merged.update(
        {
            name: round(max(seconds, merged.get(name, 0.0)) if name in failed else seconds, 3)
            for name, seconds in measured.items()
        }
    )
    for name, seconds in timeout_hints(lines, failed).items():
        merged[name] = max(merged.get(name, 0.0), seconds)
    return merged
