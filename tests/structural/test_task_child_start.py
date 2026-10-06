"""V610: enumerate every process-pool owner so a new Task start cannot bypass the class."""

from __future__ import annotations

import ast
from pathlib import Path


def test_every_process_pool_owner_is_in_the_child_start_class():
    root = Path(__file__).resolve().parents[2] / "src" / "alphalattice"
    owners = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        starts = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ProcessPoolExecutor"
        ]
        if starts:
            owners.add(path.relative_to(root).as_posix())
    assert owners == {
        "control/task_control/child.py",
        "foundation/causal_outcomes/execution/publication.py",
    }, "Review every added start owner and its Task-kind coverage for V610."
