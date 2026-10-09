"""A request that resumes a Task resumes that Task only."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "alphalattice"


def test_every_owner_resume_names_the_task_its_request_chose() -> None:
    """regression (the user's review at 4bde3cb9): `strategy run --plan <B>` marked B for
    recovery, then resumed every recovery-required Task of its kind, A included, while its
    answer named only B. The input capture, the training inputs and the feature build did the
    same. An owner's resume of its kind names the Task its request chose (`only_task_id`); only
    the Host's restart picks up every Task left recovery-required, which builds no kind table
    of its own."""

    calls: list[tuple[str, int, bool]] = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "resume"
                and node.args
                and isinstance(node.args[0], ast.Dict)
            ):
                scoped = any(keyword.arg == "only_task_id" for keyword in node.keywords)
                calls.append((path.relative_to(SRC).as_posix(), node.lineno, scoped))
    assert len(calls) >= 4, calls
    assert all(scoped for _path, _line, scoped in calls), [c for c in calls if not c[2]]
