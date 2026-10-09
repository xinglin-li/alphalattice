"""Every capped client polling owner reaches the shared deadline pause and final read."""

from __future__ import annotations

import ast
from pathlib import Path


def test_every_capped_client_waiter_has_its_cap_rule_reviewed() -> None:
    """TE12: discover capped owners and polling sleeps, so a new waiter needs class review."""
    path = Path("src/alphalattice/interface/local_application/client.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    capped = set()
    sleeps = set()
    calls = {}
    for name, function in functions.items():
        names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
        attributes = {n.attr for n in ast.walk(function) if isinstance(n, ast.Attribute)}
        if "deadline" in names or "max_wait" in attributes:
            capped.add(name)
        calls[name] = {
            n.func.id
            for n in ast.walk(function)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "time"
            and n.func.attr == "sleep"
            for n in ast.walk(function)
        ):
            sleeps.add(name)
    # Each entry is exercised with a fake clock in test_wait_cap; a new capped owner
    # must join that evidence rather than silently inherit an assumed final read.
    assert capped == {
        "_sleep_before_read",
        "_read_through_restarts",
        "_follow",
        "_wait_for_goal",
        "_wait",
        "_chain",
    }, capped
    assert sleeps == {"_sleep_before_read"}, sleeps
    # A Codex wake is sent by the Host from the Task's journal, never by the client. An agent
    # verb's chain (AGENT-TIME verbs 2 and 3) takes one deadline from its cap in `_chain` and
    # waits only through `_follow`, in its `followed` method, reviewed with the deadline owners.
    for name in capped - {"_sleep_before_read", "_wait", "_chain"}:
        assert "_sleep_before_read" in calls[name], name
        if name != "_read_through_restarts":
            assert "_read_through_restarts" in calls[name], name
    assert {"_follow", "_wait_for_goal", "_read_through_restarts"} <= calls["_wait"]


def test_every_host_deadline_owner_has_its_wait_or_nonwait_reviewed() -> None:
    """Discover deadline owners in Host/transport/storage, including nested lock waits."""
    base = Path("src/alphalattice")
    found = set()
    for root in (
        "interface/local_application",
        "control/product_host",
        "control/workspace_runtime",
    ):
        for path in (base / root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for function in ast.walk(tree):
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                args = {
                    arg.arg
                    for arg in (
                        *function.args.posonlyargs,
                        *function.args.args,
                        *function.args.kwonlyargs,
                    )
                }
                assigned = {
                    target.id
                    for node in ast.walk(function)
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                    for target in (
                        [node.target] if isinstance(node, ast.AnnAssign) else node.targets
                    )
                    if isinstance(target, ast.Name)
                }
                if "deadline" in args | assigned or "max_wait" in args:
                    found.add((path.relative_to(base).as_posix(), function.name))
    client_path = "interface/local_application/client.py"
    reviewed = {
        (client_path, name)
        for name in (
            "_sleep_before_read",
            "_read_through_restarts",
            "_follow",
            "_wait_for_goal",
            "_wait",
            # AGENT-TIME verbs 2 and 3: one deadline bounds every follow of a verb's Tasks.
            "_chain",
            "followed",
        )
    }
    # Status and lock acquisition both re-read after their bounded pause. Joins inspect
    # liveness after the join. Coverage's deadline is an admission expiry, not a waiter.
    reviewed.update(
        {
            ("control/product_host/composition/portfolio_research_operations.py", "_moved_on"),
            ("control/workspace_runtime/database.py", "_acquire"),
            ("control/workspace_runtime/database.py", "wait"),
            ("interface/local_application/web.py", "join_request_threads"),
            ("interface/local_application/web.py", "stop"),
            (
                "control/product_host/composition/evidence_review_application.py",
                "_refresh_coverage",
            ),
        }
    )
    assert found == reviewed, {
        "unreviewed": sorted(found - reviewed),
        "gone": sorted(reviewed - found),
    }
