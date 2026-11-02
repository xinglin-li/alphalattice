"""The test noun's pytest plugin: each collected test's registered size, owner and bound."""

from __future__ import annotations

import json
import os
from collections import Counter
from math import isfinite
from pathlib import Path
from types import CodeType
from typing import Any

import pytest

from devtools.testing import requirements
from devtools.testing.contracts import Backends, LayerContract
from devtools.testing.sizes import (
    changed_paths,
    classify,
    included,
    load_policy,
    scoped,
    stage_files,
)

REQUIRED = pytest.StashKey[tuple[tuple[str, ...], tuple[str, ...]]]()
CHANGED = pytest.StashKey[frozenset[str] | None]()  # a scoped run's changed paths, read once
CONTRACT = pytest.StashKey[LayerContract]()
PHASES = pytest.StashKey[dict[str, float]]()
FRAMEWORK = pytest.StashKey[frozenset[CodeType]]()
BACKENDS = pytest.StashKey[Backends]()
FIND = 'register its requirement first (python scripts/tests_registry.py find "<words>")'


def _proven(
    item: Any, rows: dict[str, dict[str, Any]], policy: dict[str, Any]
) -> tuple[list[str], dict[str, Any], list[str], tuple[str, str, str, float]]:
    """The ids a test proves, its first requirement, its size markers and its classification, the
    requirement's layer giving its size; an unknown or retired id refuses."""
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
    layer = [requirements.SIZES[row["layer"]]] if row else []
    return ids, row, declared, classify(item.nodeid, declared + layer, policy)


@pytest.hookimpl(wrapper=True, trylast=True)  # type: ignore[untyped-decorator]
def pytest_runtest_setup(item: Any) -> Any:
    level = dict(item.user_properties).get("requirement_layer") or requirements.layer(item.nodeid)
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
    if CONTRACT not in item.stash:
        return (yield)
    contract = item.stash[CONTRACT]
    contract.references(list(item.funcargs.values()))
    with contract.active():
        return (yield)


@pytest.hookimpl(wrapper=True, trylast=True)  # type: ignore[untyped-decorator]
def pytest_runtest_teardown(item: Any) -> Any:
    if CONTRACT not in item.stash:
        return (yield)
    with item.stash[CONTRACT].active():
        return (yield)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)  # type: ignore[untyped-decorator]
def pytest_runtest_makereport(item: Any, call: Any) -> Any:
    """Judge unit cost from its enclosing phases, using the existing solo-timeout route."""
    result = yield
    report = result.get_result()
    if CONTRACT not in item.stash or item.stash[CONTRACT].layer != "unit":
        return
    phases = item.stash.setdefault(PHASES, {})
    phases[report.when] = report.duration
    item.user_properties.append((report.when + "_seconds", report.duration))
    report.user_properties = list(item.user_properties)
    if report.when != "teardown":
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
) -> None:
    """Attach each owner's existing size, real bound and requirement metadata once."""
    size, owner, declaration, seconds = classified
    if not declared:
        item.add_marker(size)
    previous = item.get_closest_marker("timeout")
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


def pytest_configure(config: Any) -> None:
    """The producer's optional required nodes override only their own ordinary deselection."""
    try:
        required = requirements.selection(
            json.loads(os.environ.get("ALPHALATTICE_PIPELINE_REQUIRED_TESTS", "[]"))
        )
    except (ValueError, TypeError):
        raise pytest.UsageError("test_size.required_selection_invalid") from None
    deselect = tuple(config.option.deselect or ()) if required else ()
    if required:
        config.option.deselect = []
    config.stash[REQUIRED] = required, deselect


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


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Materialize each test's registered size and declaring owner during collection."""

    lane = os.environ.get("ALPHALATTICE_TEST_LANE", "")
    changed = changed_paths() if lane in {"merge", "fast"} else None
    empty = pytest.mark.skip(reason="the change holds no Python file for this per_file scan")
    python = any(path.endswith(".py") for path in changed or ())
    config.stash[CHANGED] = changed
    kept: list[Any] = []
    omitted: list[Any] = []
    seen: set[str] = set()
    marks: list[tuple[str, list[str], Path]] = []
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
        stage = os.environ.get("ALPHALATTICE_PIPELINE_STAGE", "")
        files = {item.nodeid.split("::", 1)[0].replace("\\", "/") for item in items}
        permitted = set(stage_files(policy, stage, files, lane=lane))
        for item in items:
            if (lane or required) and item.nodeid in seen:
                omitted.append(item)
                continue
            seen.add(item.nodeid)
            ids, row, declared, (size, owner, declaration, seconds) = _proven(item, rows, policy)
            marks.append((item.nodeid, ids, Path(item.path)))
            _properties(item, ids, row, declared, (size, owner, declaration, seconds))
            if scoped(item.nodeid, policy, changed):
                item.add_marker("per_file")  # it reads only the changed files, through scanned()
                if not python:
                    item.add_marker(empty)
            forced = any(requirements.node_matches(item.nodeid, n) for n in required)
            (
                kept
                if forced
                or (
                    item.nodeid.split("::", 1)[0].replace("\\", "/") in permitted
                    and included(item.nodeid, size, lane, policy, changed)
                    and not item.nodeid.startswith(deselect)
                )
                else omitted
            ).append(item)
        gate(lane, marks, counted, root / "tests")
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    if lane or required:
        config.hook.pytest_deselected(items=omitted)
        items[:] = kept
