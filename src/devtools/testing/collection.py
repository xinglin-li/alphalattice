"""The test noun's pytest plugin: each collected test's registered size, owner and bound."""

from __future__ import annotations

import json
import os
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
SIZE_POLICY = pytest.StashKey[dict[str, Any]]()
SIZE_CLOCK = pytest.StashKey[dict[str, float]]()
FIND = 'register its requirement first (python scripts/tests_registry.py find "<words>")'


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
    families = node.config.stash.get(SIZE_FAMILIES, None)
    if os.environ.get("ALPHALATTICE_PIPELINE_STAGE") == "commit" and os.environ.get(
        "ALPHALATTICE_PIPELINE_EVENTS"
    ):
        node.workerinput["alphalattice_size_scope"] = families is not None
        if families is not None:
            node.workerinput["alphalattice_size_families"] = list(families)


@pytest.hookimpl(wrapper=True, trylast=True)  # type: ignore[untyped-decorator]
def pytest_fixture_setup(fixturedef: Any, request: Any) -> Any:
    clock = request.config.stash.get(SIZE_CLOCK, None)
    began = time.perf_counter() if clock is not None else 0.0
    item = request.node
    selected = fixturedef.argname == "tmp_path" and item.stash.get(SIZE_SELECTED, False)
    before = time.perf_counter() - began if clock is not None else 0.0
    value = yield
    began = time.perf_counter() if clock is not None else 0.0
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
        admission_limit(item.nodeid, size, sum(phases.values()), footprint[0], policy)
    except (KeyError, TypeError, ValueError) as error:
        report.outcome, report.longrepr = "failed", str(error)


def _timeline(item: Any, report: Any) -> None:
    """The run's timeline in its JUnit: which worker ran each case, from when to when."""
    if report.when == "call":
        return
    edge = ("started_at", report.start) if report.when == "setup" else ("ended_at", report.stop)
    item.user_properties.append((edge[0], round(edge[1], 3)))
    if report.when == "setup":
        item.user_properties.append(("worker", os.environ.get("PYTEST_XDIST_WORKER", "main")))
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
    if seconds <= limit:
        return
    report.outcome, report.longrepr = (
        "failed",
        (
            f"testing.layer_contract: unit own phases {seconds:g} s exceed {limit:g} s; "
            "share setup or move the test to integration"
        ),
    )
    event = dict(
        event="TIMEOUT",
        nodeid=item.nodeid,
        phase="teardown",
        limit_seconds=limit,
        elapsed_seconds=seconds,
        cause="unit-own-phases",
    )
    log_file = item.config.getoption("log_file") or item.config.getini("log_file")
    if log_file:
        try:
            with Path(log_file).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
        except OSError as error:
            report.longrepr += f"; testing.layer_contract_unmeasured: retry evidence: {error}"


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


def pytest_configure(config: Any) -> None:
    """The producer's optional required nodes override only their own ordinary deselection."""
    try:
        required = requirements.selection(
            json.loads(os.environ.get("ALPHALATTICE_PIPELINE_REQUIRED_TESTS", "[]"))
        )
    except (ValueError, TypeError):
        raise pytest.UsageError("test_size.required_selection_invalid") from None
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


@pytest.hookimpl(trylast=True)  # type: ignore[untyped-decorator]
def pytest_sessionfinish(session: Any) -> None:
    """Retain component costs once per process; missing evidence cannot prove the stage budget."""
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


RELEASE_ONLY = "release-lane test: it reads the release archives, so it runs at promotion, not here"


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
        print(f"test_size.nightly_only: {item.nodeid}; run in the nightly on pc1")
        return True
    if contract == "release_only" and lane == "nightly":
        item.add_marker(pytest.mark.skip(reason=RELEASE_ONLY))
    return False


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
    omitted: tuple[Any, ...] = (),
) -> None:
    """Beside the run's JUnit, ``structural-moved.json``: the structural guards a commit stage
    leaves to post-merge, and whether this run ran them (a post-merge run, unscoped, always does),
    so a post-merge red among them is a red the stage no longer catches (the week's signal)."""
    xml = getattr(config.option, "xmlpath", None)
    found = sorted({n.replace("\\", "/") for n in (i.nodeid for i in items) if moved(n, policy)})
    if lane != "merge" or not xml or not found:
        return
    excluded = {item.nodeid.replace("\\", "/") for item in omitted}
    record = {
        "tier": "commit stage" if floor or changed is not None else "post-merge",
        "moved": found,
        "ran": [
            n
            for n in found
            if n not in excluded and (changed is None or n.split("::")[0] in changed)
        ],
    }
    if floor:
        config.stash[FLOOR_MOVED] = record
    else:
        _write_moved(config, record)


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
    """A floor receipt names the moved guards actually kept after all collection hooks."""
    record = session.config.stash.get(FLOOR_MOVED, None)
    if record is not None:
        kept = {item.nodeid.replace("\\", "/") for item in session.items}
        record["ran"] = [node for node in record["moved"] if node in kept]
        _write_moved(session.config, record)


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
    _record_moved(config, lane, items, policy, changed, floor=floor, omitted=tuple(omitted))
    if lane or required or omitted:
        config.hook.pytest_deselected(items=omitted)
        items[:] = kept
