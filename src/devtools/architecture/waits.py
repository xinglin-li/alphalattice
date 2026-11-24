"""The commit's wait rules: an unbounded wait or loop is refused where it is written.

Each rule names a classic loop or wait anti-pattern and its repair through the one way to wait,
``alphalattice.capabilities.domain_contracts.waits``. The rules read Python source with the standard
library's ``ast`` alone, so they run alike on every platform. Only the wait primitive writes
``while True`` or sleeps: every other loop states its end in its test or bounds its passes with a
``for``, so a service's or monitor's endless loop cannot be written outside it. A poison item
retried ahead of the rest cannot be seen statically and stays a review question.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

PRIMITIVE = "src/alphalattice/capabilities/domain_contracts/waits.py"
WAIT = "alphalattice.capabilities.domain_contracts.waits"
RULES = {
    "while-true": f"a while True loop outside the wait primitive: state its end in the loop's "
    f"test, bound its passes or attempts with a for, or wait through {WAIT}.wait_until",
    "busy-wait": f"a loop that spins without blocking or sleeping: wait with {WAIT}.wait_until",
    "sleep-outside-wait": f"a bare sleep: pause through {WAIT}.pause or poll with "
    f"{WAIT}.wait_until",
    "swallowed-in-loop": "an exception swallowed inside a loop (except with pass or continue): "
    "name it, count it, and stop or raise past a bound",
    "call-without-timeout": "a blocking call with no timeout, or timeout=None: pass the time "
    f"left, from {WAIT}.wait_until's check",
    "wall-clock-deadline": "a deadline on time.time(), which jumps with the wall clock: use "
    "time.monotonic()",
    "deadline-reset-in-loop": "a deadline set again on each pass of the loop it bounds never "
    "arrives: set it once, before the loop",
    "blocking-under-lock": "a sleep or blocking call while a lock is held: release the lock "
    "first, or wait on a Condition",
}
BLOCKING = {"join", "wait", "result", "acquire", "communicate"}
"""Methods that block without an argument; ``get`` blocks only on a queue (a ContextVar's never)."""
SUBPROCESS = {"run", "call", "check_call", "check_output"}
WAITING = BLOCKING | {"sleep", "pause", "poll", "get", "fetch", "recv", "select"}
"""Calls that may wait, so a loop testing one is not a spin."""
SPIN = (ast.Pass, ast.Continue)


@dataclass(frozen=True)
class Finding:
    """One refused line: where, which rule, and its repair."""

    path: str
    line: int
    rule: str

    @property
    def message(self) -> str:
        """The rule's repair, as the refusal says it."""
        return RULES[self.rule]


def _walk(nodes: list[ast.stmt]) -> Iterator[ast.AST]:
    """Every node under ``nodes``, not entering a nested function or class."""
    stack: list[ast.AST] = list(nodes)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            stack.extend(ast.iter_child_nodes(node))


def _name(node: ast.AST) -> str:
    """A call target's last name: ``a.b.c`` is ``c``."""
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def _owner(node: ast.AST) -> str:
    """The name a method is called on: ``time.sleep`` is ``time``."""
    return _name(node.value) if isinstance(node, ast.Attribute) else ""


def _is_clock(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _name(node.func) in {"monotonic", "perf_counter"}


def _is_wall(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _name(node.func) == "time" and _owner(node.func) == "time"


def _exits(nodes: list[ast.stmt]) -> bool:
    """Whether a statement under ``nodes`` leaves the loop: break, return or raise."""
    return any(isinstance(n, ast.Break | ast.Return | ast.Raise) for n in _walk(nodes))


def _without_timeout(call: ast.Call, subprocesses: set[str]) -> bool:
    """A blocking call with no timeout, positional or named, or with ``timeout=None``."""
    named = {k.arg: k.value for k in call.keywords}
    blocking = named.get("blocking")
    if isinstance(blocking, ast.Constant) and blocking.value is False:
        return False
    if "timeout" in named:
        timeout = named["timeout"]
        return isinstance(timeout, ast.Constant) and timeout.value is None
    name, owner = _name(call.func), _owner(call.func)
    if name in SUBPROCESS and (owner == "subprocess" or name in subprocesses):
        return True
    if name == "urlopen":
        return len(call.args) < 3
    if name == "create_connection":
        return len(call.args) < 2
    if (
        name == "fetch" and "sub" in owner.lower()
    ):  # a NATS pull subscription: fetch(batch, timeout)
        return len(call.args) < 2
    queue = name == "get" and ("queue" in owner.lower() or owner in {"q", "inbox"})
    return isinstance(call.func, ast.Attribute) and (name in BLOCKING or queue) and not call.args


def _deadline_resets(loop: ast.While) -> Iterator[ast.Assign]:
    """Top-level assignments that set a monotonic deadline the loop itself tests."""
    tested = set(map(_name, ast.walk(loop.test)))
    tested |= {_name(n) for s in loop.body if isinstance(s, ast.If) for n in ast.walk(s.test)}
    for statement in loop.body:
        if (
            isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.BinOp)
            and isinstance(statement.value.op, ast.Add)
            and any(_is_clock(n) for n in ast.walk(statement.value))
            and any(_name(t) in tested for t in statement.targets)
        ):
            yield statement


def _spins(loop: ast.While) -> bool:
    """A loop that spins without waiting.

    An idle body (only pass or continue) spins when its test calls nothing, or when its test
    polls a negated predicate that does not wait (``while not ready(): pass``). A busy body spins
    when its test polls a predicate that does not wait while the body calls nothing, never leaves
    and changes nothing the test reads (``while not ready(): n += 1``). A test that does the work
    (``while step(): pass``) or waits (``while not stop.wait(1)``) is no spin.
    """
    idle = all(
        isinstance(s, SPIN) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        for s in loop.body
    )
    calls = [n for n in ast.walk(loop.test) if isinstance(n, ast.Call)]
    waits = any(
        _name(c.func) in WAITING or any(k.arg == "timeout" for k in c.keywords) for c in calls
    )
    if idle:
        negated = isinstance(loop.test, ast.UnaryOp) and isinstance(loop.test.op, ast.Not)
        return not calls or (negated and not waits)
    if not calls:
        return False
    body = list(_walk(loop.body))
    stored = {_name(n) for n in body if isinstance(getattr(n, "ctx", None), ast.Store)}
    return not (
        waits
        or any(isinstance(n, ast.Call) for n in body)
        or _exits(loop.body)
        or stored & {_name(n) for n in ast.walk(loop.test)}
    )


def _loop_rules(loop: ast.While) -> Iterator[tuple[ast.AST, str]]:
    """A constant test (``while True``, ``while 1``) hides the loop's end in its body."""
    if isinstance(loop.test, ast.Constant) and loop.test.value:
        yield loop, "while-true"
    if _spins(loop):
        yield loop, "busy-wait"
    for reset in _deadline_resets(loop):
        yield reset, "deadline-reset-in-loop"


def _swallowed(loop: ast.For | ast.While) -> Iterator[tuple[ast.AST, str]]:
    for node in _walk(loop.body):
        if isinstance(node, ast.ExceptHandler) and all(isinstance(s, SPIN) for s in node.body):
            yield node, "swallowed-in-loop"


def _sleep_named(node: ast.AST, sleeps: set[str]) -> bool:
    """``time.sleep``, or a name ``from time import sleep`` bound, read as a value."""
    if isinstance(node, ast.Name):
        return node.id in sleeps and isinstance(node.ctx, ast.Load)
    return _name(node) == "sleep" and _owner(node) == "time"


def _call_rule(call: ast.Call, sleeps: set[str], subprocesses: set[str]) -> str | None:
    named = isinstance(call.func, ast.Name) and call.func.id in sleeps
    if named or (_name(call.func) == "sleep" and _owner(call.func) == "time"):
        return "sleep-outside-wait"
    return "call-without-timeout" if _without_timeout(call, subprocesses) else None


def _wall_deadline(node: ast.AST) -> bool:
    """``time.time()`` added to a span or compared: a deadline on the wall clock."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_wall(node.left) or _is_wall(node.right)
    return isinstance(node, ast.Compare) and any(map(_is_wall, (node.left, *node.comparators)))


def _under_lock(node: ast.AST, subprocesses: set[str]) -> Iterator[tuple[ast.AST, str]]:
    """A sleep or a blocking call inside ``with <a lock>:``."""
    if isinstance(node, ast.With) and any(
        "lock" in _name(i.context_expr).lower() for i in node.items
    ):
        for call in _walk(node.body):
            if isinstance(call, ast.Call) and (
                _name(call.func) == "sleep" or _without_timeout(call, subprocesses)
            ):
                yield call, "blocking-under-lock"


def _imported(tree: ast.Module) -> tuple[set[str], set[str]]:
    """The names ``from time import sleep`` and ``from subprocess import run`` bind."""
    sleeps: set[str] = set()
    subprocesses: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"time", "subprocess"}:
            names = {a.asname or a.name for a in node.names}
            (sleeps if node.module == "time" else subprocesses).update(
                n for n in names if n == "sleep" or n in SUBPROCESS
            )
    return sleeps, subprocesses


def findings(path: str, text: str) -> list[Finding]:
    """Every rule's findings in one Python source; the wait primitive's own module holds none."""
    if path == PRIMITIVE:
        return []
    tree = ast.parse(text, filename=path)
    sleeps, subprocesses = _imported(tree)
    hits: list[tuple[ast.AST, str]] = []
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if id(node) not in called and _sleep_named(node, sleeps):  # a seam's default or a handle
            hits.append((node, "sleep-outside-wait"))
        if isinstance(node, ast.While):
            hits += _loop_rules(node)
        if isinstance(node, ast.For | ast.While):
            hits += _swallowed(node)
        if isinstance(node, ast.Call) and (rule := _call_rule(node, sleeps, subprocesses)):
            hits.append((node, rule))
        if _wall_deadline(node):
            hits.append((node, "wall-clock-deadline"))
        hits += _under_lock(node, subprocesses)
    unique = {(id(node), rule): (getattr(node, "lineno", 1), rule) for node, rule in hits}
    return [Finding(path, line, rule) for line, rule in sorted(unique.values())]


def counts(rows: list[Finding]) -> dict[str, int]:
    """The allowance's form: ``path::rule`` to its number of findings."""
    return dict(sorted(Counter(f"{f.path}::{f.rule}" for f in rows).items()))
