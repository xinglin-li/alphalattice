"""The test noun's pytest plugin: each collected test's registered size, owner and bound."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from devtools.testing import requirements
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
    declared = [m.name for m in item.iter_markers() if m.name in policy["sizes"]]
    layer = [requirements.SIZES[row["layer"]]] if row else []
    return ids, row, declared, classify(item.nodeid, declared + layer, policy)


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
