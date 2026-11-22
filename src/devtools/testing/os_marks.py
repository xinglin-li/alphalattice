"""A test that calls an OS-only API names the OS it needs, so other platforms skip it.

The product's CI runs macOS, Ubuntu and Windows. A test that reaches a Windows-only module or
name (ctypes' WinDLL, winreg, _winapi, msvcrt), reads an OSError's winerror it did not set, or
starts PowerShell or schtasks, fails on the other two unless a skipif on sys.platform, os.name or
platform.system() names its OS: on the test, its class or its module, through a mark alias, or
as a pytest.skip or return under a non-Windows branch. A use inside a branch that already tests
for Windows is the general test's own fork. The commit's fast guard runs this over the tests a
change writes (scripts/check_test_shape.py).
"""

from __future__ import annotations

import ast
import re

OS_NAMES = frozenset({"WinDLL", "windll", "winreg", "_winapi", "msvcrt"})
OS_COMMANDS = frozenset(
    {"powershell", "powershell.exe", "pwsh", "pwsh.exe", "schtasks", "schtasks.exe"}
)
STARTS = re.compile(r"^(?:subprocess\.)?(?:run|Popen|call|check_call|check_output)$|^os\.system$")
OS_MARK = re.compile(r"skipif\(\s*(?:not\s+)?(?:sys\.platform|os\.name|platform\.system\(\))")
RULE = "os-unmarked"
REPAIR = (
    'add @pytest.mark.skipif(sys.platform != "win32", reason=...) to the test or its module, '
    "or make it general"
)


def _tests(source: str) -> dict[str, ast.AST]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    found: dict[str, ast.AST] = {}
    for node in tree.body:
        members = node.body if isinstance(node, ast.ClassDef) else [node]
        for item in members:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith(
                "test_"
            ):
                name = f"{node.name}.{item.name}" if isinstance(node, ast.ClassDef) else item.name
                found[name] = item
    return found


def own_tests(new: str, *parents: str) -> dict[str, ast.AST]:
    """The tests a change writes: each differs from its version in every parent, so a merge
    brings in none."""
    befores = [{name: ast.dump(node) for name, node in _tests(old).items()} for old in parents]
    return {
        name: node
        for name, node in _tests(new).items()
        if all(before.get(name) != ast.dump(node) for before in befores)
    }


def _windows_branch(test: ast.AST) -> bool | None:
    """Whether a condition's true branch is Windows's own (True), the other platforms' (False),
    or neither (None)."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return True if any(_windows_branch(value) is True for value in test.values) else None
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return None
    left, right = test.left, test.comparators[0]
    pairs = {("os.name", "nt"), ("sys.platform", "win32"), ("platform.system()", "Windows")}
    pair = (
        (ast.unparse(left), right.value)
        if isinstance(right, ast.Constant) and isinstance(right.value, str)
        else None
    )
    reverse = (
        (ast.unparse(right), left.value)
        if isinstance(left, ast.Constant) and isinstance(left.value, str)
        else None
    )
    if pair not in pairs and reverse not in pairs:
        return None
    if isinstance(test.ops[0], ast.Eq):
        return True
    return False if isinstance(test.ops[0], ast.NotEq) else None


def _guarded(node: ast.AST) -> tuple[set[int], bool]:
    """The nodes inside a branch restricted to Windows, and whether a non-Windows branch skips
    or returns before the rest of the test."""
    found, skips = set(), False
    for branch in ast.walk(node):
        if not isinstance(branch, (ast.If, ast.IfExp)):
            continue
        windows = _windows_branch(branch.test)
        if windows is None:
            continue
        parts = (
            [branch.body if windows else branch.orelse]
            if isinstance(branch, ast.IfExp)
            else (branch.body if windows else branch.orelse)
        )
        found |= {id(x) for part in parts for x in ast.walk(part)}
        if windows is False and isinstance(branch, ast.If):
            skips |= isinstance(branch.body[-1], ast.Return) or any(
                isinstance(x, ast.Call) and ast.unparse(x.func) in {"pytest.skip", "skip"}
                for part in branch.body
                for x in ast.walk(part)
            )
    return found, skips


def _winerror(nodes: list[ast.AST]) -> set[str]:
    """winerror, when the test reads an OSError's winerror it never set itself."""
    contexts = {type(n.ctx) for n in nodes if isinstance(n, ast.Attribute) and n.attr == "winerror"}
    return {"winerror"} if ast.Load in contexts and ast.Store not in contexts else set()


def _commands(nodes: list[ast.AST]) -> set[str]:
    """The Windows-only commands a process start in these nodes names as its first argument."""
    found = set()
    for call in (n for n in nodes if isinstance(n, ast.Call) and n.args):
        if not STARTS.match(ast.unparse(call.func)):
            continue
        first = call.args[0]
        head = first.elts[0] if isinstance(first, (ast.List, ast.Tuple)) and first.elts else first
        words = (
            head.value.split()
            if isinstance(head, ast.Constant) and isinstance(head.value, str)
            else []
        )
        found |= {words[0].lower()} & OS_COMMANDS if words else set()
    return found


def _uses(nodes: list[ast.AST]) -> list[str]:
    """The OS-only names, unset winerror reads and Windows-only commands these nodes reach."""
    names = {n.id for n in nodes if isinstance(n, ast.Name)} | {
        n.attr for n in nodes if isinstance(n, ast.Attribute)
    }
    return sorted(names & OS_NAMES | _winerror(nodes) | _commands(nodes))


def _marks(body: list[ast.stmt]) -> tuple[set[str], set[str], bool]:
    """A module's OS-mark aliases, the classes it marks, and whether its pytestmark names the OS."""
    aliases = {
        target.id
        for n in body
        if isinstance(n, ast.Assign) and OS_MARK.search(ast.unparse(n.value))
        for target in n.targets
        if isinstance(target, ast.Name)
    }
    classes = {
        n.name for n in body if isinstance(n, ast.ClassDef) and _marked(n.decorator_list, aliases)
    }
    module = any(
        isinstance(n, ast.Assign)
        and "pytestmark" in ast.unparse(n.targets[0])
        and (
            OS_MARK.search(ast.unparse(n.value)) or any(a in ast.unparse(n.value) for a in aliases)
        )
        for n in body
    )
    return aliases, classes, module


def _marked(decorators: list[ast.expr], aliases: set[str]) -> bool:
    return any(OS_MARK.search(ast.unparse(d)) or ast.unparse(d) in aliases for d in decorators)


def unmarked(tests: dict[str, ast.AST], source: str) -> dict[str, str]:
    """Each test that reaches an OS-only API with no mark naming its OS: test -> the first use."""
    try:
        aliases, classes, module = _marks(ast.parse(source).body)
    except SyntaxError:
        return {}
    found = {}
    for name, node in {} if module else tests.items():
        if name.split(".")[0] in classes or _marked(getattr(node, "decorator_list", []), aliases):
            continue
        guarded, skipped = _guarded(node)
        used = [] if skipped else _uses([n for n in ast.walk(node) if id(n) not in guarded])
        if used:
            found[name] = used[0]
    return found


def refusals(path: str, new: str, *parents: str) -> list[str]:
    """The tests this change writes in `path` that reach an OS-only API without naming the OS,
    each refused with its repair."""
    return [
        f"{path}::{name}: {RULE}: uses {use} but names no OS: {REPAIR}"
        for name, use in sorted(unmarked(own_tests(new, *parents), new).items())
    ]
