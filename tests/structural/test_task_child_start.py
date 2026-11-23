"""Every installed worker pool registers with the execution owner."""

from __future__ import annotations

import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from alphalattice.kernel.execution import process_execution


def test_every_process_pool_owner_is_in_the_child_start_class():
    root = Path(__file__).resolve().parents[2] / "src" / "alphalattice"
    registrations: Counter[str] = Counter()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        enrolled = {
            call.args[0]
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "register_pool"
            and call.args
        }
        for call in calls:
            if isinstance(call.func, ast.Name) and call.func.id in {
                "ThreadPoolExecutor",
                "ProcessPoolExecutor",
                "Parallel",
            }:
                assert call in enrolled, f"{path.relative_to(root)}:{call.lineno}"
                registrations[call.func.id] += 1
    assert registrations == {
        "ThreadPoolExecutor": 13,
        "ProcessPoolExecutor": 2,
        "Parallel": 1,
    }


@pytest.mark.proves("execution.registered_pools_keep_their_workers")
def test_registered_pools_keep_their_workers() -> None:
    """Register both pools and close the first without stopping the second's worker."""
    owner = process_execution()
    with (
        owner.register_pool(ThreadPoolExecutor(1), name="first-test-pool") as first,
        owner.register_pool(ThreadPoolExecutor(1), name="second-test-pool") as second,
    ):
        assert first is not second
        assert {"first-test-pool", "second-test-pool"} <= set(owner.registered_pool_names())
        first.shutdown()
        assert second.submit(lambda: 1).result(timeout=1) == 1
