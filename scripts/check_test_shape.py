"""Refuse growing source shape and test-writing violations in staged Git blobs.

The weekly JSON report records existing debt and static review candidates.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tokenize
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alphalattice.capabilities.domain_contracts import spawn
from devtools.architecture.git_inputs import read_blobs as _blobs
from devtools.architecture.git_inputs import staged_changes, staged_moves, unique_moves
from devtools.testing import os_marks
from devtools.testing.layout import (
    REGISTRIES,
    import_violations,
    new_data_file_problems,
    new_test_file_problems,
)
from devtools.testing.requirements import PROVES_LINE, file_holders
from devtools.testing.requirements import parse as parse_requirements
from devtools.testing.sizes import validate_policy

ROOT = Path(__file__).resolve().parents[1]
CAP_BYTES = 100 * 1024
SLEEP = re.compile(r"\bsleep\(\s*(\d+(?:\.\d+)?)\s*\)")
# a literal of five or more words that an assert or an includes() checks: a sentence pin
PIN = re.compile(r"""(?:\bincludes\(\s*|\bassert\s+)(['"`])([^'"`\n]{0,400})\1""")


ASSETS = "src/alphalattice/interface/local_application/assets/"
GENERATED = frozenset(
    ASSETS + "workbench" + suffix
    for suffix in (".html", ".css", ".js", ".zh.js", "-prelude.js", "-manifest.json")
)
# word catalogs are data: every new word needs its Chinese key, so a size ratchet would block every
# feature; the dead-key censuses hold their growth instead
LOCAL = "src/alphalattice/interface/local_application/"
WORD_CATALOGS = frozenset(
    {
        LOCAL + "refusal_words.json",
        LOCAL + "answers.json",
        LOCAL + "assets/workbench-source/js/data/zh.js",
    }
)
# third-party data the package ships whole, never edited: tiktoken's cl100k encoding (NOTICE), kept
# so a token count needs no download; its size is the upstream file's, checked by its SHA-256
VENDORED_DATA = frozenset({"src/alphalattice/protocols/actor_execution/cl100k_base.tiktoken.gz"})
CONTROL_OWNER = "src/alphalattice/capabilities/domain_contracts/environment.py"
WRITE = re.compile(
    r'\b(?:INSERT(?:\s+OR\s+\w+)?\s+INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO)\s+([\w."`]+)',
    re.I,
)
CODE = frozenset(
    {".py", ".js", ".cjs", ".mjs", ".ts", ".tsx", ".jsx", ".css", ".html", ".sql", ".sh", ".ps1"}
)
DEFINITION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
FUNCTION_TYPES = DEFINITION_TYPES[:-1]
TOKENS = re.compile(
    r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`"""
    r"|/\*.*?\*/|<!--.*?-->|//[^\n]*|===|!==|=>|\+\+|--|&&|\|\||\?\?"
    r"""|\w+|[^\w\s"'`/]|/(?![/*])""",
    re.S,
)


def _tree(text: str) -> ast.Module:
    try:
        return ast.parse(text)
    except SyntaxError:
        return ast.Module(body=[], type_ignores=[])


def _body(node: ast.AST):
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, DEFINITION_TYPES):
            yield child
            yield from _body(child)


def _definitions(tree: ast.AST, prefix: str = ""):
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, DEFINITION_TYPES):
            name = prefix + node.name
            yield name, node
            yield from _definitions(node, name + ".")
        else:
            yield from _definitions(node, prefix)


# Ruff's pytest-style rules a test change may not grow, with the fix each names
PT_FIXES = {
    "PT011": "give pytest.raises a match= for the expected message",
    "PT012": "keep one simple statement inside pytest.raises",
    "PT014": "drop the duplicate parametrize case",
    "PT015": "use pytest.fail for an assertion that always fails",
    "PT017": "assert the exception through pytest.raises",
}


def _ruff(snapshots: dict[str, str], *options: str) -> dict[str, list[dict]]:
    """Ruff's JSON rows for each snapshot, from one locked invocation."""
    if not snapshots:
        return {}
    with tempfile.TemporaryDirectory(prefix="test-shape-") as folder:
        root, names = Path(folder), {}
        for index, (key, text) in enumerate(snapshots.items()):
            path = root / f"{index:08}.py"
            path.write_text(text, encoding="utf-8", errors="surrogateescape", newline="\n")
            names[path.name] = key
        done = spawn.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--ignore-noqa",
                "--target-version",
                "py312",
                *options,
                "--output-format",
                "json",
                str(root),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if done.returncode not in (0, 1):
            raise ValueError("ruff could not read the staged snapshots")
        result = {key: [] for key in snapshots}
        for row in json.loads(done.stdout):
            result[names[Path(row["filename"]).name]].append(row)
        return result


def ruff_test_findings(snapshots: dict[str, str]) -> dict[str, tuple[tuple[str, int], ...]]:
    """Each staged test snapshot's PT findings as (rule, line), from one Ruff invocation."""
    rows = _ruff(snapshots, "--select", ",".join(PT_FIXES))
    if any(row["code"] not in PT_FIXES for found in rows.values() for row in found):
        raise ValueError("ruff returned a rule outside its PT selection")
    return {
        key: tuple(sorted((row["code"], row["location"]["row"]) for row in found))
        for key, found in rows.items()
    }


ANTIPATTERN_BASELINE = "config/registries/antipattern-baseline.json"
ANTIPATTERN_PRIVATE = (
    "config/" + Path(ANTIPATTERN_BASELINE).name.removesuffix(".json") + "-private.json"
)
ANTIPATTERN_SIDES = (ANTIPATTERN_BASELINE, ANTIPATTERN_PRIVATE)
ANTIPATTERN_SEMGREP = "scripts/antipattern_rules.yml"
ANTIPATTERN_CASES = "scripts/antipattern_examples.py.txt"
TEST_ANTIPATTERN_RULES = "scripts/test_antipattern_rules.yml"  # ast-grep: milliseconds a file
TEST_ANTIPATTERN_CASES = "scripts/test_antipattern_examples.py.txt"
ANTIPATTERN_RULES = "PGH,G,LOG,C901,PLR0912,PLR0915,S112,S113,DTZ005,ASYNC"
SIGNAL_RULES = frozenset({"C901", "PLR0912", "PLR0915", "test-too-many-mocks"})
"""Counts of size and form, measured and never refused: complying with a cap on them moves lines
and improves nothing (the 11-14 gate verdict, which the maintainers' own tools follow too).
The commit names each one over its bound in a function the change wrote as a review signal."""
WAIT_RULES = ("S112", "S113", "DTZ005", "ASYNC")


def _mock_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    names = {"patch", "Mock", "MagicMock", "AsyncMock"}
    if isinstance(func, ast.Name):
        return func.id in names
    if not isinstance(func, ast.Attribute):
        return False
    owner = func.value.id if isinstance(func.value, ast.Name) else ""
    return (
        func.attr in names
        or (func.attr == "object" and owner == "patch")
        or (func.attr in {"setattr", "setenv", "setitem"} and owner in {"monkeypatch", "binding"})
    )


def _mock_scores(path: str, definitions: list) -> dict[str, int]:
    scores = {}
    for name, node in definitions:
        if isinstance(node, FUNCTION_TYPES) and node.name.startswith("test_"):
            count = sum(_mock_call(item) for item in ast.walk(node))
            if count > 4:
                scores[f"{path}::test-too-many-mocks::{name}"] = count
    return scores


DOCSTRING_RULES = ("D10", "D419")
"""Ruff's docstring rules: a public library definition with no docstring, or an empty one. A
change answers for the definitions it writes; the rest is counted by the code health report."""


ARGUMENTS = 16_000
"""The characters of paths one Ruff call names: Windows refuses a command line past 32,767."""


def ruff_docstrings(
    snapshots: dict[str, str], config: str
) -> dict[str, tuple[tuple[str, int], ...]]:
    """Each source snapshot's docstring findings (rule, line), keyed by its path: one Ruff
    invocation per batch of paths under the repository's own configuration, each snapshot at its
    own path so the per-file-ignores apply as they do in the tree."""
    if not snapshots:
        return {}
    with tempfile.TemporaryDirectory(prefix="docstrings-") as folder:
        root = Path(folder)
        (root / "pyproject.toml").write_text(config, encoding="utf-8", newline="\n")
        batches: list[list[str]] = [[]]
        for path, text in snapshots.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text, encoding="utf-8", errors="surrogateescape", newline="\n")
            if batches[-1] and sum(len(p) + 1 for p in batches[-1]) + len(path) > ARGUMENTS:
                batches.append([])
            batches[-1].append(path)
        found: dict[str, list[tuple[str, int]]] = {path: [] for path in snapshots}
        for batch in batches:
            done = spawn.run(
                [
                    *(sys.executable, "-m", "ruff", "check", "--config", "pyproject.toml"),
                    *("--select", ",".join(DOCSTRING_RULES), "--no-cache"),
                    *("--output-format", "json", *batch),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if done.returncode not in (0, 1):
                raise ValueError("ruff could not read the staged sources' docstrings")
            for row in json.loads(done.stdout):
                path = Path(row["filename"]).resolve().relative_to(root.resolve()).as_posix()
                found[path].append((row["code"], row["location"]["row"]))
    return {path: tuple(sorted(rows)) for path, rows in found.items()}


def _as_written(node: ast.AST) -> str:
    """A definition's signature and body, without its decorators: what a change writes when it
    adds or edits it. A class holds its own statements only: each definition in it is judged
    alone, so adding, removing or editing a method leaves its class as written."""
    if isinstance(node, ast.ClassDef):
        own = [item for item in node.body if not isinstance(item, DEFINITION_TYPES)]
        node = ast.ClassDef(node.name, node.bases, node.keywords, own, [], node.type_params)
    elif isinstance(node, FUNCTION_TYPES):
        node = type(node)(node.name, node.args, node.body, [], node.returns, None, node.type_params)
    return ast.dump(node, include_attributes=False)


def undocumented(
    path: str, old: str | None, new: str, findings: tuple[tuple[str, int], ...]
) -> list[str]:
    """Each docstring finding on a definition this change adds, or whose signature or body it
    edits; a changed module answers for its own docstring. An untouched definition keeps its
    debt, which only falls: no change can add an undocumented one."""
    was = defaultdict(set)
    for name, node in _definitions(_tree(old or "")):
        was[name].add(_as_written(node))
    now = {node.lineno: (name, node) for name, node in _definitions(_tree(new))}
    found = []
    for rule, line in findings:
        name, node = now.get(line, ("the module", None))
        if node is None or _as_written(node) not in was[name]:
            found.append(
                f"{path}:{line}: {name} has no docstring (Ruff {rule}); "
                "document the definition this change writes"
            )
    return found


def ruff_antipatterns(snapshots: dict[str, str]) -> dict[str, int]:
    """Count rules by file and complex functions by their measured score."""
    rows = _ruff(snapshots, "--select", ANTIPATTERN_RULES)
    counts: Counter[str] = Counter()
    for path, findings in rows.items():
        definitions = list(_definitions(_tree(snapshots[path])))
        if path.startswith("tests/"):
            counts.update(_mock_scores(path, definitions))
        for row in findings:
            code = row["code"]
            if path.startswith("tests/") and code.startswith(WAIT_RULES):
                continue
            key, amount = f"{path}::{code}", 1
            if code in {"C901", "PLR0912", "PLR0915"}:
                score = re.search(r"\((\d+) > \d+\)", row["message"])
                line = row["location"]["row"]
                owners = [
                    (node.end_lineno - node.lineno, name)
                    for name, node in definitions
                    if isinstance(node, FUNCTION_TYPES) and node.lineno <= line <= node.end_lineno
                ]
                if not score or not owners:
                    raise ValueError(f"Ruff omitted the complexity owner for {path}:{line}")
                key += "::" + min(owners)[1]
                amount = int(score.group(1))
            counts[key] += amount
    return dict(counts)


RULE_FILES = (
    "scripts/check_test_shape.py",
    ANTIPATTERN_SEMGREP,
    TEST_ANTIPATTERN_RULES,
)


def seeded_rules(held: dict[str, int], baseline: dict[str, int], staged: list[str]) -> set[str]:
    """A rule the parent's allowance never held is a new rule's seed, admitted only with a change
    to a file that defines the rules; a finding a rule's landing missed is repaired, not seeded."""
    if not set(RULE_FILES) & set(staged):
        return set()
    return {key.partition("::")[2] for key in baseline} - {key.partition("::")[2] for key in held}


def antipattern_growth(held: dict[str, int], found: dict[str, int]) -> list[str]:
    """A finding or allowance may fall, but neither may grow above the held count; a size or form
    count (SIGNAL_RULES) is never refused."""
    return [
        f"{key}: anti-pattern findings grow {held.get(key, 0)} -> {count}"
        + (f"; {CHILD_REPAIRS[rule]}" if rule in CHILD_REPAIRS else "")
        for key, count in sorted(found.items())
        for rule in [key.split("::")[1]]
        if count > held.get(key, 0) and rule not in SIGNAL_RULES
    ]


# The test-child rules (the harness's twins, tools/lint.py):
# Static test rules on the children a test starts and the model turns it spends.
#
# ``unbounded_children``: a test's direct subprocess call with no finite timeout, or whose stdin
# is not proved closed to the runner's, can hang the run or read the runner's own stdin. Proved
# closed is ``stdin=DEVNULL`` or ``PIPE``, or ``input=`` given data (a string, bytes, an f-string, a
# sum or a call), never a name, attribute or item that may be None. A ``Popen`` is bounded only
# where its own function waits on the name it is bound to with a finite timeout (``wait``'s first
# argument, ``communicate``'s second, or ``timeout=``). A name counts as the value it is bound to (a
# parameter's default, or one assignment in the function or the module); any other name is
# unknown, never finite and never data.
#
# ``paid_turns``: a model turn outside the ``paid_agent`` class: a command starting a model CLI for
# a turn (``claude -p``, ``codex exec``) as a list, tuple or one string; any call reached through a
# provider SDK's import, under whatever alias, or an import of one by name; or a provider API host.
#
# ``parametrized_children``: a parametrized test whose own body starts a child process (a
# ``subprocess`` call, ``os.system`` or ``os.popen``, or an asyncio subprocess) runs that child once
# per case; the decision belongs in memory with one real journey per distinct path. It counts the
# cases (a literal list, or a module-level literal table, else one), so a grown table is growth.
# Children started through product code are not chased.
#
# Each finding is ``(rule, owner, line)``: the owner is the test function that holds it (its
# dotted name inside a class), or ``<module>``, so an allowance is keyed by file and function.

UNBOUNDED = "test-child-unbounded"
BARE_PYTHON = "test-child-bare-python"
TEXT_ENCODING = "test-child-text-encoding"
PAID = "test-paid-agent"
PARAMETRIZED = "test-parametrized-process"
CHILD_CALLS = frozenset({"run", "check_output", "check_call", "call", "Popen"})
WAIT_TIMEOUT_AT = {"wait": 0, "communicate": 1}  # Popen.wait(timeout), communicate(input, timeout)
MODEL_TURNS = (("claude", "-p"), ("claude", "--print"), ("codex", "exec"))
MODEL_PACKAGES = frozenset({"anthropic", "openai", "claude_agent_sdk", "claude_code_sdk"})
MODEL_HOSTS = ("api.anthropic.com", "api.openai.com")
CHILD_REPAIRS = {
    UNBOUNDED: "pass timeout= and stdin=subprocess.DEVNULL, or input=",
    BARE_PYTHON: "use sys.executable instead of a bare Python command",
    TEXT_ENCODING: "decode text explicitly with encoding='utf-8'",
    PAID: "mark it @pytest.mark.paid_agent, which no gate, queue, nightly or full run selects",
    PARAMETRIZED: "test the decision in memory and keep one real journey per distinct path",
}
OS_CHILDREN = frozenset({"system", "popen"})
ASYNC_CHILDREN = frozenset({"create_subprocess_exec", "create_subprocess_shell"})
UNKNOWN = ast.Name(id="<unknown>")
FUNCTIONS = ast.FunctionDef | ast.AsyncFunctionDef


def _owned(scope: ast.AST) -> Iterator[ast.AST]:
    """The nodes a function or module holds itself; a nested function's nodes are its own."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if not isinstance(node, FUNCTIONS):
            yield node
            stack.extend(ast.iter_child_nodes(node))


def _bound_names(nodes: Iterable[ast.AST]) -> dict[str, ast.AST]:
    """Names assigned once among these nodes, to their value; one assigned twice is unknown."""
    names: dict[str, ast.AST] = {}
    for node in nodes:
        targets: list[ast.expr]
        value: ast.expr | None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        for target in (t for t in targets if isinstance(t, ast.Name)):
            single = target.id not in names and len(targets) == 1
            names[target.id] = (value or UNKNOWN) if single else UNKNOWN
    return names


def _parameters(args: ast.arguments) -> dict[str, ast.AST]:
    """Each parameter bound to its default; one with none is unknown."""
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    pairs = [*zip(positional, defaults, strict=True)]
    pairs += [*zip(args.kwonlyargs, args.kw_defaults, strict=True)]
    return {arg.arg: default or UNKNOWN for arg, default in pairs}


def _resolved(value: ast.AST | None, names: dict[str, ast.AST]) -> ast.AST | None:
    for _ in range(5):  # a chain of names, bounded
        if not (isinstance(value, ast.Name) and value.id in names):
            break
        value = names[value.id]
    return value


def _finite(value: ast.AST | None, names: dict[str, ast.AST]) -> bool:
    """A positive finite number, or a sum or product of them; None, a name or a call never is."""
    value = _resolved(value, names)
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add | ast.Mult):
        return _finite(value.left, names) and _finite(value.right, names)
    number = value.value if isinstance(value, ast.Constant) else None
    return (
        isinstance(number, int | float)
        and not isinstance(number, bool)
        and 0 < number < float("inf")
    )


def _timeout(call: ast.Call, method: str) -> ast.AST | None:
    at = WAIT_TIMEOUT_AT[method]
    given = next((k.value for k in call.keywords if k.arg == "timeout"), None)
    return given if given is not None else (call.args[at] if len(call.args) > at else None)


def _stdin_closed(given: dict[str | None, ast.AST], names: dict[str, ast.AST]) -> bool:
    stdin, data = _resolved(given.get("stdin"), names), _resolved(given.get("input"), names)
    if (
        isinstance(stdin, ast.Attribute)
        and isinstance(stdin.value, ast.Name)
        and stdin.value.id == "subprocess"
    ):
        stdin = ast.Name(id=stdin.attr)
    return (
        (isinstance(stdin, ast.Name) and stdin.id in {"DEVNULL", "PIPE"})
        or isinstance(data, ast.JoinedStr | ast.BinOp | ast.Call)
        or (isinstance(data, ast.Constant) and isinstance(data.value, str | bytes))
    )


def _pipe(value: ast.AST | None, names: dict[str, ast.AST]) -> bool:
    value = _resolved(value, names)
    return (isinstance(value, ast.Name) and value.id == "PIPE") or (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "subprocess"
        and value.attr == "PIPE"
    )


def _test_owners(tree: ast.Module) -> dict[int, str]:
    """Each node's owning test function by id: its dotted name, or ``<module>``."""
    owner: dict[int, str] = {}

    def walk(node: ast.AST, name: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = name
            if isinstance(child, FUNCTIONS | ast.ClassDef):
                inner = child.name if name == "<module>" else f"{name}.{child.name}"
            owner[id(child)] = inner
            walk(child, inner)

    walk(tree, "<module>")
    return owner


def unbounded_children(tree: ast.Module, owner: Mapping[int, str]) -> list[tuple[str, str, int]]:
    found: list[tuple[str, str, int]] = []
    module = _bound_names(tree.body)
    scopes: list[ast.Module | ast.FunctionDef | ast.AsyncFunctionDef] = [tree]
    scopes += [n for n in ast.walk(tree) if isinstance(n, FUNCTIONS)]
    for scope in scopes:
        owned = list(_owned(scope))
        names = (
            module
            if isinstance(scope, ast.Module)
            else {**module, **_bound_names(owned), **_parameters(scope.args)}
        )
        waited = {
            n.func.value.id
            for n in owned
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in WAIT_TIMEOUT_AT
            and isinstance(n.func.value, ast.Name)
            and _finite(_timeout(n, n.func.attr), names)
        }
        communicated = {
            n.func.value.id
            for n in owned
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "communicate"
            and isinstance(n.func.value, ast.Name)
            and _finite(_timeout(n, "communicate"), names)
        }
        fed = {
            n.func.value.id
            for n in owned
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "communicate"
            and isinstance(n.func.value, ast.Name)
            and _finite(_timeout(n, "communicate"), names)
            and _stdin_closed(
                {
                    "input": next(
                        (k.value for k in n.keywords if k.arg == "input"),
                        n.args[0] if n.args else UNKNOWN,
                    )
                },
                names,
            )
        }
        held = {
            id(n.value): n.targets[0].id
            for n in owned
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        }
        held |= {
            id(n.context_expr): n.optional_vars.id
            for n in owned
            if isinstance(n, ast.withitem) and isinstance(n.optional_vars, ast.Name)
        }
        for node in owned:
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in CHILD_CALLS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
            ):
                continue
            given: dict[str | None, ast.AST] = {k.arg: k.value for k in node.keywords if k.arg}
            args = _resolved(node.args[0], names) if node.args else None
            executable = (
                _resolved(args.elts[0], names)
                if isinstance(args, ast.List | ast.Tuple) and args.elts
                else args
            )
            if (
                isinstance(executable, ast.Constant)
                and isinstance(executable.value, str)
                and executable.value.lower() in {"python", "python.exe", "python3"}
            ):
                found.append((BARE_PYTHON, owner.get(id(node), "<module>"), node.lineno))
            text_mode = any(
                isinstance(value := _resolved(given.get(key), names), ast.Constant)
                and value.value is True
                for key in ("text", "universal_newlines")
            )
            encoding = _resolved(given.get("encoding"), names)
            if (text_mode or encoding is not None) and not (
                isinstance(encoding, ast.Constant)
                and isinstance(encoding.value, str)
                and encoding.value.lower().replace("-", "") == "utf8"
            ):
                found.append((TEXT_ENCODING, owner.get(id(node), "<module>"), node.lineno))
            child = held.get(id(node))
            output_pipe = _pipe(given.get("stdout"), names) or _pipe(given.get("stderr"), names)
            input_pipe = _pipe(given.get("stdin"), names)
            bounded = (
                child in (communicated if output_pipe else waited)
                and (not input_pipe or child in fed)
                if node.func.attr == "Popen"
                else _finite(given.get("timeout"), names)
            )
            if not bounded or not _stdin_closed(given, names) or (input_pipe and child not in fed):
                found.append((UNBOUNDED, owner.get(id(node), "<module>"), node.lineno))
    return found


def _paid_marked(node: ast.AST) -> bool:
    """A definition or module marked paid_agent: the decorator, or pytestmark naming it."""
    marks = [d.func if isinstance(d, ast.Call) else d for d in getattr(node, "decorator_list", ())]
    if isinstance(node, ast.Module):
        values = [
            v
            for s in node.body
            if isinstance(s, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in s.targets)
            for v in ([*s.value.elts] if isinstance(s.value, ast.List | ast.Tuple) else [s.value])
        ]
        marks = [m.func if isinstance(m, ast.Call) else m for m in values]
    return any(isinstance(m, ast.Attribute) and m.attr == "paid_agent" for m in marks)


def _root(node: ast.AST) -> str:
    """The name a call's callee chain starts from: openai in openai.responses.create."""
    while isinstance(node, ast.Attribute | ast.Call | ast.Subscript):
        node = node.value if isinstance(node, ast.Attribute | ast.Subscript) else node.func
    return node.id if isinstance(node, ast.Name) else ""


def _provider(name: str | None) -> bool:
    return (name or "").split(".")[0] in MODEL_PACKAGES


def paid_turns(tree: ast.Module, owner: Mapping[int, str]) -> list[tuple[str, str, int]]:
    if _paid_marked(tree):
        return []
    nodes = list(ast.walk(tree))
    held: set[int] = set()
    found: list[tuple[str, str, int]] = []
    for node in nodes:
        if isinstance(node, FUNCTIONS | ast.ClassDef) and _paid_marked(node):
            held |= {id(n) for n in ast.walk(node)}
    bound = {
        (a.asname or a.name).split(".")[0]
        for n in nodes
        if isinstance(n, ast.Import)
        for a in n.names
        if _provider(a.name)
    }
    bound |= {
        a.asname or a.name
        for n in nodes
        if isinstance(n, ast.ImportFrom) and not n.level and _provider(n.module)
        for a in n.names
    }
    hosts = tuple(f"{scheme}{h}" for h in MODEL_HOSTS for scheme in ("", "http://", "https://"))
    # A model CLI's words count only as the command a subprocess call launches: an asserted argv
    # or a fake runner's command starts no turn (6 of 6 such words on the tree at 11-16 did not).
    launched = {
        id(command)
        for call in nodes
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in CHILD_CALLS
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "subprocess"
        for command in [*call.args[:1], *(k.value for k in call.keywords if k.arg == "args")]
    }
    for node in nodes:
        if id(node) in held:
            continue
        if id(node) not in launched:
            words: list[str] = []
        elif isinstance(node, ast.List | ast.Tuple):
            words = [
                e.value
                for e in node.elts[:2]
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            words = node.value.split()[:2]
        else:
            words = []
        client = isinstance(node, ast.Call) and (
            _root(node.func) in bound
            or (
                _root(node.func) in {"importlib", "import_module", "__import__"}
                and any(
                    isinstance(a, ast.Constant) and isinstance(a.value, str) and _provider(a.value)
                    for a in node.args
                )
            )
        )
        host = (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith(hosts)
        )
        if (tuple(words) in MODEL_TURNS or client or host) and isinstance(node, ast.expr):
            found.append((PAID, owner.get(id(node), "<module>"), node.lineno))
    return found


def _starts_child(call: ast.Call) -> bool:
    func = call.func
    if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
        return False
    module, name = func.value.id, func.attr
    return (
        (module == "subprocess" and name in CHILD_CALLS)
        or (module == "os" and name in OS_CHILDREN)
        or (module == "asyncio" and name in ASYNC_CHILDREN)
    )


TEST_MODULE_CODE = re.compile(r"(?:^|[\s;])(?:from\s+tests\.[\w.]+\s+import\b|import\s+tests\.)")
TEST_MODULE_TARGET = re.compile(r"^tests(?:\.\w+)+:\w+$")


def _names_a_test_module(value: ast.AST, after: ast.AST | None) -> bool:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        text = value.value
        return bool(TEST_MODULE_CODE.search(text) or TEST_MODULE_TARGET.match(text)) or (
            text.startswith("tests.")
            and isinstance(after, ast.Constant)
            and after.value == "-m"  # `-m tests.x`, read backwards from the module
        )
    if isinstance(value, ast.JoinedStr):  # f"{__name__}:target" names this test module
        parts = value.values
        return any(
            isinstance(part, ast.FormattedValue)
            and isinstance(part.value, ast.Name)
            and part.value.id == "__name__"
            and isinstance(following, ast.Constant)
            and str(following.value).startswith(":")
            for part, following in itertools.pairwise(parts)
        )
    return False


def test_module_children(tree: ast.Module) -> list[int]:
    """The lines where a child process is started on a test module (`tests.*`): a
    `module:function` target, an import in a `-c` script, or `-m`. The child imports it by
    name in a fresh interpreter, where `tests` is whichever package its import path finds first,
    possibly another checkout's; a source module resolves to this tree."""
    lines: set[int] = set()
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    module = _assigns(ast.Module([node for node in tree.body if isinstance(node, ast.Assign)], []))
    for scope in [tree, *(node for node in ast.walk(tree) if isinstance(node, functions))]:
        assigned = module if scope is tree else module | _assigns(scope)
        for call in ast.walk(scope):
            if isinstance(call, ast.Call) and _child_names_a_test_module(call, assigned):
                lines.add(call.lineno)
    return sorted(lines)


def _assigns(scope: ast.AST) -> dict[str, list[ast.AST]]:
    """Every value each plain name is assigned within a scope: a name a function assigns twice
    resolves to both, and one assigned in another function never shadows it."""
    found: dict[str, list[ast.AST]] = {}
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.setdefault(target.id, []).append(node.value)
    return found


def _child_names_a_test_module(call: ast.Call, assigned: Mapping[str, list[ast.AST]]) -> bool:
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if not (_starts_child(call) or name == "run_in_child"):
        return False
    for given in [*call.args, *(keyword.value for keyword in call.keywords)]:
        roots = assigned.get(given.id, [given]) if isinstance(given, ast.Name) else [given]
        for root in roots:
            nodes = list(ast.walk(root))
            for previous, node in zip([None, *nodes], nodes, strict=False):
                values = assigned.get(node.id, [node]) if isinstance(node, ast.Name) else [node]
                if any(_names_a_test_module(value, previous) for value in values):
                    return True
    return False


def _cases(argvalues: ast.AST, tables: Mapping[str, ast.AST]) -> int:
    """How many cases a parametrize argument holds: a literal list, tuple, set or dict, or a
    module-level literal table named directly, by ``.values()``, ``.keys()`` or ``list(...)``."""
    node = argvalues
    if isinstance(node, ast.Call) and not node.args and isinstance(node.func, ast.Attribute):
        node = node.func.value if node.func.attr in {"values", "keys", "items"} else node
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1:
        node = node.args[0] if node.func.id in {"list", "tuple"} else node
    if isinstance(node, ast.Name):
        node = tables.get(node.id, node)
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return len(node.elts)
    if isinstance(node, ast.Dict):
        return len(node.keys)
    return 1


def _parametrized(function: ast.AST, tables: Mapping[str, ast.AST]) -> int:
    """The cases a function's parametrize decorators give it, multiplied; 0 when it has none."""
    cases = 0
    for mark in getattr(function, "decorator_list", ()):
        if (
            isinstance(mark, ast.Call)
            and isinstance(mark.func, ast.Attribute)
            and mark.func.attr == "parametrize"
            and len(mark.args) > 1
        ):
            cases = max(cases, 1) * _cases(mark.args[1], tables)
    return cases


def _assigned(tree: ast.Module) -> dict[str, ast.AST]:
    """A module's top-level names to the values assigned to them."""
    return {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _imported_tables(path: str, tree: ast.Module) -> dict[str, tuple[str, str]]:
    """Each name a test module imports from a module under tests/: its path and its name there."""
    found: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        package = PurePosixPath(path).parent.parts
        base = package[: len(package) - node.level + 1] if node.level else ()
        module = "/".join([*base, *(node.module or "").split(".")]).strip("/") + ".py"
        if module.startswith("tests/"):
            found |= {a.asname or a.name: (module, a.name) for a in node.names}
    return found


def parametrized_children(
    tree: ast.Module, owner: Mapping[int, str], imported: Mapping[str, ast.AST]
) -> list[tuple[str, str, int]]:
    found: list[tuple[str, str, int]] = []
    tables = {**imported, **_assigned(tree)}
    for function in ast.walk(tree):
        if not isinstance(function, FUNCTIONS) or not function.name.startswith("test"):
            continue
        cases = _parametrized(function, tables)
        child = cases and next(
            (n for n in _owned(function) if isinstance(n, ast.Call) and _starts_child(n)), None
        )
        if isinstance(child, ast.Call):
            name = owner.get(id(function), function.name)
            found += [(PARAMETRIZED, name, child.lineno)] * cases
    return found


# A file whose text holds none of these words can hold no finding of the three rules below: each
# needs a subprocess, os or asyncio child call, a provider import or host, or a model CLI's word.
CHILD_WORDS = (
    "subprocess",
    "os.system",
    "os.popen",
    "claude",
    "codex",
    *MODEL_PACKAGES,
    *MODEL_HOSTS,
)


def child_antipatterns(
    snapshots: dict[str, str], modules: Mapping[str, str] | None = None
) -> dict[str, int]:
    """Each test's unbounded children, unclassed model turns and parametrized cases that start a
    child, keyed file::rule::function. ``modules`` holds other tests/ modules' texts by path, so a
    table a test imports is counted by its cases there."""
    texts, assigned = {**(modules or {}), **snapshots}, {}

    def table(module: str, name: str) -> ast.AST | None:
        if module not in assigned:
            text = texts.get(module)
            assigned[module] = _assigned(ast.parse(text)) if text is not None else {}
        return assigned[module].get(name)

    counts: Counter[str] = Counter()
    for path, text in snapshots.items():
        if not path.startswith("tests/") or not any(word in text for word in CHILD_WORDS):
            continue
        tree = ast.parse(text)
        owner = _test_owners(tree)
        imported = {
            name: value
            for name, (module, origin) in _imported_tables(path, tree).items()
            if (value := table(module, origin)) is not None
        }
        for rule, name, _ in (
            *unbounded_children(tree, owner),
            *paid_turns(tree, owner),
            *parametrized_children(tree, owner, imported),
        ):
            counts[f"{path}::{rule}::{name}"] += 1
    return dict(counts)


def _test_file(path: str) -> bool:
    name = PurePosixPath(path).name
    return path.startswith("tests/") and (name.startswith("test_") or name.endswith("_test.py"))


def _table_readers(
    sources: list[str],
    texts: dict[str, str],
    head: str,
    *,
    grep: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    read: Callable[[list[str]], dict[str, str]] | None = None,
) -> list[str]:
    """The test files a change did not touch that import a changed tests/ helper, whose tables
    their cases count; each is read into ``texts``, with every tests/ module a measured test
    imports from. ``grep`` runs Git and ``read`` reads blobs (default: this checkout's)."""
    grep, read = grep or _git, read or _blobs
    helpers = {
        PurePosixPath(p).stem
        for p in sources
        if p.startswith("tests/") and p.endswith(".py") and not _test_file(p)
    } - {"conftest", "__init__"}
    readers: list[str] = []
    if helpers:
        where = ["--cached"] if head == ":" else [head.rstrip(":")]
        words = [f"-e{stem}" for stem in sorted(helpers)]
        listing = grep("grep", "-l", "-F", *words, *where, "--", "tests")
        if listing.returncode > 1:
            raise ValueError("readers of the changed test helpers unreadable")
        named = {line.removeprefix(head) for line in listing.stdout.splitlines()}
        readers = sorted(
            p for p in named if _test_file(p) and p.endswith(".py") and p not in sources
        )
        texts.update(read([head + p for p in readers]))
    modules = {
        module
        for path in [*sources, *readers]
        if _test_file(path) and path.endswith(".py") and head + path in texts
        for module, _ in _imported_tables(path, ast.parse(texts[head + path])).values()
    }
    texts.update(read([head + m for m in sorted(modules) if head + m not in texts]))
    return readers


def _written_lines(old: str | None, new: str) -> set[int]:
    """The new text's line numbers a change wrote: every line of a new file."""
    lines = new.splitlines()
    if old is None:
        return set(range(1, len(lines) + 1))
    opcodes = SequenceMatcher(None, old.splitlines(), lines, autojunk=False).get_opcodes()
    return {n + 1 for tag, _, _, start, end in opcodes if tag != "equal" for n in range(start, end)}


def antipattern_signals(
    measured: dict[str, int], texts: dict[str, str], olds: dict[str, str | None]
) -> list[str]:
    """Each size or form count over its bound in a function the change wrote, as a review signal;
    ``texts`` holds each changed source and ``olds`` its parent's text (None for a new file)."""
    signals, spans, written = [], {}, {}
    for key, score in sorted(measured.items()):
        path, rule, name = [*key.split("::", 2), ""][:3]
        if rule not in SIGNAL_RULES or path not in texts:
            continue
        if path not in spans:
            spans[path] = {
                found: (node.lineno, node.end_lineno or node.lineno)
                for found, node in _definitions(_tree(texts[path]))
                if isinstance(node, FUNCTION_TYPES)
            }
            written[path] = _written_lines(olds.get(path), texts[path])
        start, end = spans[path].get(name, (0, -1))
        if any(start <= line <= end for line in written[path]):
            signals.append(
                f"test_shape.signal: {path}:{start}: {rule} {score} in {name} "
                "(a review signal, never refused)"
            )
    return signals


def antipattern_baseline(text: str) -> dict[str, int]:
    data = json.loads(text)
    baseline = data["counts"]
    if not isinstance(baseline, dict) or any(
        not isinstance(k, str) or "::" not in k or type(v) is not int or v < 1
        for k, v in baseline.items()
    ):
        raise ValueError("invalid anti-pattern allowance")
    return baseline


def antipattern_sides(texts: dict[str, str], prefix: str) -> dict[str, int]:
    counts = Counter(antipattern_baseline(texts[prefix + ANTIPATTERN_BASELINE]))
    if (private := texts.get(prefix + ANTIPATTERN_PRIVATE)) is not None:
        counts.update(antipattern_baseline(private))
    return dict(counts)


def load_antipattern_baseline(root: Path) -> dict[str, int]:
    texts = {ANTIPATTERN_BASELINE: (root / ANTIPATTERN_BASELINE).read_text(encoding="utf-8")}
    with contextlib.suppress(FileNotFoundError):  # the public checkout has no private side
        texts[ANTIPATTERN_PRIVATE] = (root / ANTIPATTERN_PRIVATE).read_text(encoding="utf-8")
    return antipattern_sides(texts, "")


def relocated_antipatterns(held: dict[str, int], moves: dict[str, str]) -> dict[str, int]:
    """Move a file's allowance with its move: Git's byte-identical one, or a row of the change's
    move table, whose findings are still measured against the allowance it carries."""
    back = {old: new for new, old in moves.items()}
    return {
        back.get(path, path) + "::" + rule: count
        for key, count in held.items()
        for path, _, rule in [key.partition("::")]
    }


def test_data_antipatterns(paths: list[str]) -> dict[str, int]:
    return {
        f"{path}::test-data-file": 1
        for path in paths
        if path.startswith("tests/")
        and Path(path).suffix.lower() not in CODE | {".md"}
        and not path.endswith((".test.yml", ".test.yaml", "/pytest.ini"))
    }


SCANNER_TEMP_LIMIT = 64
"""The longest TEMP semgrep is given on Windows: semgrep-core writes its own files under TEMP, and
past about 80 characters they cross the 260-character path limit and it exits 2 (1114-T1-13)."""


def scanner_temp(default: str, environ: dict[str, str], *, windows: bool) -> str | None:
    """Where semgrep's scratch and its TEMP go: the default (None), unless on Windows it is longer
    than SCANNER_TEMP_LIMIT, as a stage's run-allocated TEMP is; then the user's own temp folder."""
    if not windows or len(default) <= SCANNER_TEMP_LIMIT or not environ.get("LOCALAPPDATA"):
        return None
    return str(Path(environ["LOCALAPPDATA"]) / "Temp")


def _scanner_folder(prefix: str) -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(
        prefix=prefix,
        dir=scanner_temp(tempfile.gettempdir(), dict(os.environ), windows=os.name == "nt"),
    )


def _semgrep_environment(root: Path) -> dict[str, str]:
    short = scanner_temp(tempfile.gettempdir(), dict(os.environ), windows=os.name == "nt")
    return {
        **os.environ,
        **({"TEMP": short, "TMP": short} if short else {}),
        "SEMGREP_SETTINGS_FILE": str(root / "settings.yml"),
        "SEMGREP_LOG_FILE": str(root / "semgrep.log"),
        "SEMGREP_VERSION_CACHE_PATH": str(root / "version"),
    }


def _semgrep_selftest(rules: str, cases: str) -> None:
    with _scanner_folder("antipattern-cases-") as folder:
        root = Path(folder)
        (root / "rules.yml").write_text(rules, encoding="utf-8")
        (root / "rules.py").write_text(cases, encoding="utf-8")
        exe = Path(sys.executable).parent / ("semgrep.exe" if os.name == "nt" else "semgrep")
        result = spawn.run(
            [
                str(exe),
                "--test",
                "--config",
                "rules.yml",
                "rules.py",
                "--json",
                "--metrics=off",
                "--disable-version-check",
            ],
            cwd=root,
            env=_semgrep_environment(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            raise ValueError(f"semgrep rule cases failed: {result.stderr[-400:]}")
        report = json.loads(result.stdout)
        checks = {
            name: row
            for config in report["results"].values()
            for name, row in config["checks"].items()
        }
        expected = set(re.findall(r"(?m)^  - id: (\S+)$", rules))
        if (
            report.get("config_missing_tests")
            or set(checks) != expected
            or any(not row["passed"] for row in checks.values())
        ):
            raise ValueError("semgrep rule cases are incomplete or failed")


def semgrep_antipatterns(snapshots: dict[str, str], rules: str) -> dict[str, int]:
    """Run the slower rules on the same committed or staged source snapshots."""
    if not snapshots:
        return {}
    with _scanner_folder("antipattern-") as folder:
        root = Path(folder)
        config = root / "rules.yml"
        config.write_text(rules, encoding="utf-8")
        for path, text in snapshots.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", errors="surrogateescape", newline="\n")
        done = spawn.run(
            [
                str(
                    Path(sys.executable).parent / ("semgrep.exe" if os.name == "nt" else "semgrep")
                ),
                "scan",
                "--config",
                str(config),
                "--json",
                "--disable-nosem",
                "--disable-version-check",
                "--metrics=off",
                "--no-git-ignore",
                ".",
            ],
            cwd=root,
            env=_semgrep_environment(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if done.returncode:
            raise ValueError(f"semgrep exited {done.returncode}: {done.stderr[-400:]}")
        report = json.loads(done.stdout)
        if report.get("errors"):
            raise ValueError(f"semgrep could not measure its inputs: {report['errors']}")
        scanned = {Path(path).as_posix() for path in report["paths"]["scanned"]}
        if missing := set(snapshots) - scanned:
            raise ValueError(f"semgrep skipped source: {sorted(missing)[0]}")
        counts = Counter()
        for row in report["results"]:
            path = Path(row["path"]).as_posix()
            counts[f"{path}::{row['check_id'].split('.')[-1]}"] += 1
        return dict(counts)


def _astgrep_rows(root: Path, target: str) -> list[tuple[str, str, int]]:
    """Each ast-grep finding of root/rules.yml in root/target: its path under target, rule, line."""
    exe = Path(sys.executable).parent / ("ast-grep.exe" if os.name == "nt" else "ast-grep")
    done = spawn.run(
        [str(exe), "scan", "--rule", "rules.yml", "--json=compact", target],
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if done.returncode not in (0, 1):  # 1: an error-level finding
        raise ValueError(f"ast-grep exited {done.returncode}: {done.stderr[-400:]}")
    rows = json.loads(done.stdout)
    if not isinstance(rows, list):
        raise ValueError("ast-grep could not measure its inputs")
    return [
        (
            Path(row["file"]).relative_to(target).as_posix() if target != "." else row["file"],
            row["ruleId"],
            row["range"]["start"]["line"] + 1,
        )
        for row in rows
    ]


def astgrep_antipatterns(snapshots: dict[str, str], rules: str) -> dict[str, int]:
    """Run the ast-grep test rules on the same committed or staged test snapshots."""
    if not snapshots:
        return {}
    with _scanner_folder("test-antipattern-") as folder:
        root = Path(folder)
        (root / "rules.yml").write_text(rules, encoding="utf-8")
        for path, text in snapshots.items():
            target = root / "tree" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", errors="surrogateescape", newline="\n")
        return dict(Counter(f"{path}::{rule}" for path, rule, _ in _astgrep_rows(root, "tree")))


def _astgrep_selftest(rules: str, cases: str) -> None:
    """Each ast-grep rule fires on the line after its `# ruleid:` example and on no other."""
    with _scanner_folder("test-antipattern-cases-") as folder:
        root = Path(folder)
        (root / "rules.yml").write_text(rules, encoding="utf-8")
        (root / "cases.py").write_text(cases, encoding="utf-8", newline="\n")
        found = {(rule, line) for _, rule, line in _astgrep_rows(root, "cases.py")}
    marked = {
        (rule.strip(), n + 2)
        for n, text in enumerate(cases.splitlines())
        if text.strip().startswith("# ruleid:")
        for rule in text.split(":", 1)[1].split(",")
    }
    ids = set(re.findall(r"(?m)^id: (\S+)$", rules))
    if found != marked or {rule for rule, _ in marked} != ids:
        raise ValueError(f"ast-grep rule examples failed: {sorted(found ^ marked)[:4]}")


def auxiliary_antipatterns(snapshots: dict[str, str]) -> dict[str, int]:
    """Check staged workflows and shell scripts with their locked executables."""
    if not snapshots:
        return {}
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix="antipattern-tools-") as folder:
        root = Path(folder)
        for path, text in snapshots.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="\n")
        windows = ".exe" if os.name == "nt" else ""
        for tool, files, options in (
            (
                "actionlint",
                [p for p in snapshots if p.startswith(".github/workflows/")],
                ["-no-color", "-shellcheck=", "-pyflakes=", "-format", "{{json .}}"],
            ),
            (
                "shellcheck",
                [p for p in snapshots if p.endswith(".sh")],
                ["--norc", "--format=json1"],
            ),
        ):
            if not files:
                continue
            executable = Path(sys.executable).parent / (tool + windows)
            if not executable.is_file():
                raise ValueError(
                    f"{tool} is not installed at {executable}; sync the locked environment "
                    "(uv sync --locked --all-extras)"
                )
            done = spawn.run(
                [str(executable), *options, *files],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if done.returncode not in (0, 1):
                raise ValueError(f"{tool} exited {done.returncode}: {done.stderr[-400:]}")
            report = json.loads(done.stdout)
            rows = report if tool == "actionlint" else report["comments"]
            for row in rows:
                path = Path(row["filepath"] if tool == "actionlint" else row["file"]).as_posix()
                rule = row["kind"] if tool == "actionlint" else f"SC{row['code']}"
                counts[f"{path}::{tool}-{rule}"] += 1
        for path, text in snapshots.items():
            if path.endswith(".sh"):
                counts[f"{path}::shellcheck-inline-disable"] += sum(
                    bool(re.match(r"\s*#\s*shellcheck\b.*\bdisable\s*=", line, re.I))
                    for line in text.splitlines()
                )
    return {key: count for key, count in counts.items() if count}


def unbounded_loops(snapshots: dict[str, str]) -> dict[str, list[str]]:
    """Use the existing ast-grep owner to find literal loops lacking an exit or named bound."""
    if not snapshots:
        return {}
    with _scanner_folder("loop-check-") as folder:
        root = Path(folder)
        (root / "rules.yml").write_text(
            "id: unbounded-loop\nlanguage: python\nseverity: error\nmessage: unbounded loop\n"
            "rule:\n  pattern: 'while True: $$$BODY'\n",
            encoding="utf-8",
        )
        for path, source in snapshots.items():
            target = root / "tree" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8", errors="surrogateescape", newline="\n")
        found: dict[str, list[str]] = {}
        trees = {path: ast.parse(source) for path, source in snapshots.items()}
        for path, _, line in _astgrep_rows(root, "tree"):
            loop = next(
                node
                for node in ast.walk(trees[path])
                if isinstance(node, ast.While) and node.lineno == line
            )

            def scoped(node, nested=False, outer=loop):
                yield node, nested
                for child in ast.iter_child_nodes(node):
                    if not isinstance(
                        child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
                    ):
                        inner = (
                            node is not outer
                            and isinstance(node, (ast.For, ast.AsyncFor, ast.While))
                            and child in node.body
                        )
                        yield from scoped(child, nested or inner)

            nodes = list(scoped(loop))
            exits = any(
                isinstance(node, (ast.Return, ast.Raise))
                or (isinstance(node, ast.Break) and not nested)
                for node, nested in nodes
            )
            bounded = any(
                isinstance(node, ast.Name)
                and re.search(
                    r"deadline|timeout|limit|bound|maximum|end_at|stop_at|cap", node.id, re.I
                )
                for node, _ in nodes
            )
            if not exits and not bounded:
                found.setdefault(path, []).append(ast.dump(loop))
        return found


def _antipattern_problems(
    staged: list[str],
    sources: list[str],
    texts: dict[str, str],
    parents: tuple[str, ...],
    head: str,
    *,
    moves: dict[str, str],
    semgrep: bool = False,
    readers: Iterable[str] = (),
) -> list[str]:
    """``readers`` are test files the change left alone that import a changed tests/ helper."""
    baseline = antipattern_sides(texts, head)
    found = []
    carried = len(parents) > 1 and any(
        parent + ":" + ANTIPATTERN_BASELINE in texts for parent in parents
    )
    held: dict[str, int] | None = None  # a merge holds each count at its highest parent's: it adds
    for parent in parents:  # nothing either side had, so it refuses only what exceeds every parent
        previous = texts.get(parent + ":" + ANTIPATTERN_BASELINE)
        if previous is None:
            if not carried and (
                ANTIPATTERN_BASELINE not in staged or "scripts/check_test_shape.py" not in staged
            ):
                raise ValueError("anti-pattern baseline missing from parent")
        else:
            side = relocated_antipatterns(antipattern_sides(texts, parent + ":"), moves)
            held = {
                key: max(side.get(key, 0), (held or {}).get(key, 0))
                for key in {*side, *(held or {})}
            }
    if held is not None:
        seeds = seeded_rules(held, baseline, staged)
        found += [
            line
            for line in antipattern_growth(held, baseline)
            if line.split(": ", 1)[0].partition("::")[2] not in seeds
        ]
    missing = [path for path in sources if head + path not in texts]
    if missing:
        raise ValueError(f"changed source unreadable: {missing[0]}")
    python = {path: texts[head + path] for path in sources if path.endswith(".py")}
    loops = unbounded_loops(python)
    held: dict[str, Counter] = {}
    for parent in parents:
        previous = unbounded_loops(
            {path: texts.get(f"{parent}:{moves.get(path, path)}", "") for path in python}
        )
        for path, values in previous.items():
            held[path] = held.get(path, Counter()) | Counter(values)
    for path, values in loops.items():
        if Counter(values) - held.get(path, Counter()):
            found.append(f"{path}: unbounded-loop: add a deadline, bound or break condition")
    measured = ruff_antipatterns(python)
    helpers = {k.removeprefix(head): v for k, v in texts.items() if k.startswith(head + "tests/")}
    held_readers = {path: helpers[path] for path in readers if path in helpers}
    measured.update(child_antipatterns(python | held_readers, modules=helpers))
    olds = {path: texts.get(f"{parents[0]}:{moves.get(path, path)}") for path in python}
    for signal in antipattern_signals(measured, python, olds):
        print(signal, flush=True)
    measured.update(test_data_antipatterns(staged))
    measured.update(
        auxiliary_antipatterns(
            {path: texts[head + path] for path in sources if not path.endswith(".py")}
        )
    )
    if (test_rules := texts.get(head + TEST_ANTIPATTERN_RULES)) is not None:
        if TEST_ANTIPATTERN_RULES in staged or TEST_ANTIPATTERN_CASES in staged:
            _astgrep_selftest(test_rules, texts[head + TEST_ANTIPATTERN_CASES])
        tests = {
            p: texts[head + p] for p in sources if p.startswith("tests/") and p.endswith(".py")
        }
        measured.update(astgrep_antipatterns(tests, test_rules))
    if semgrep:
        if ANTIPATTERN_SEMGREP in staged or ANTIPATTERN_CASES in staged:
            _semgrep_selftest(texts[head + ANTIPATTERN_SEMGREP], texts[head + ANTIPATTERN_CASES])
        measured.update(
            semgrep_antipatterns(
                {
                    path: texts[head + path]
                    for path in sources
                    if path.endswith(".py") and not path.startswith("tests/")
                },
                texts[head + ANTIPATTERN_SEMGREP],
            )
        )
    return found + antipattern_growth(baseline, measured)


def _sql(node: ast.AST, values: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id, "")
    if isinstance(node, ast.JoinedStr):
        return "".join(_sql(part, values) or "?" for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _sql(node.left, values) + _sql(node.right, values)
    return ""


def _lint(path: str, tree: ast.Module) -> set[str]:
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update((item.asname or item.name, item.name) for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.update(
                (item.asname or item.name, (node.module or "") + "." + item.name)
                for item in node.names
            )

    def called(node):
        if isinstance(node, ast.Name):
            return imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return called(node.value) + "." + node.attr
        return ""

    found = set()
    if path.startswith("tests/unit/"):
        found.update(
            import_violations(
                tuple(imports.values())
                + tuple(called(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)),
                "unit",
            )
        )
    definitions = list(_definitions(tree))
    factories = {
        node.name
        for _name, node in definitions
        if called(getattr(node, "returns", None)) == "duckdb.DuckDBPyConnection"
    }
    for _name, unit in [("", tree), *definitions]:
        nodes = list(_body(unit))
        values = {}
        connections = {
            node.arg
            for node in nodes
            if isinstance(node, ast.arg)
            and node.annotation
            and called(node.annotation) == "duckdb.DuckDBPyConnection"
        }
        for node in nodes:
            if isinstance(node, ast.Assign):
                values.update(
                    (target.id, _sql(node.value, values))
                    for target in node.targets
                    if isinstance(target, ast.Name)
                )
            pairs = (
                [(target, node.value) for target in node.targets]
                if isinstance(node, ast.Assign)
                else [(item.optional_vars, item.context_expr) for item in node.items]
                if isinstance(node, ast.With | ast.AsyncWith)
                else []
            )
            connections.update(
                target.id
                for target, value in pairs
                if isinstance(target, ast.Name)
                and (
                    (isinstance(value, ast.Name) and value.id in connections)
                    or (
                        isinstance(value, ast.Call)
                        and (
                            called(value.func) == "duckdb.connect"
                            or called(value.func).rsplit(".", 1)[-1] in factories
                            or (
                                isinstance(value.func, ast.Attribute)
                                and value.func.attr == "cursor"
                                and called(value.func.value) in connections
                            )
                        )
                    )
                )
            )
        loop_calls = {
            id(child)
            for node in nodes
            if isinstance(node, ast.For | ast.AsyncFor | ast.While)
            for child in _body(node)
            if isinstance(child, ast.Call)
        }
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            if path != CONTROL_OWNER and called(node.func) in {
                "threadpoolctl.threadpool_limits",
                "threadpoolctl.ThreadpoolController",
            }:
                found.add(
                    f"{path}:{node.lineno}: threadpoolctl control; use numerical_thread_limit"
                )
            if (
                not node.args
                or not isinstance(node.func, ast.Attribute)
                or node.func.attr not in {"execute", "executemany", "sql"}
            ):
                continue
            writes = list(WRITE.finditer(_sql(node.args[0], values)))
            if any(
                match.group(1).lower().rsplit(".", 1)[-1].strip('"`') == "feature_ineligibility"
                for match in writes
            ):
                found.add(f"{path}:{node.lineno}: view mutation; write feature_ineligibility_run")
            if (
                writes
                and id(node) in loop_calls
                and node.func.attr in {"execute", "executemany"}
                and called(node.func.value) in connections
            ):
                found.add(
                    f"{path}:{node.lineno}: DuckDB row-loop write; batch rows at storage owner"
                )
    return found


@lru_cache(maxsize=64)
def _code_lines(path: str, text: str) -> list[tuple[int, str]]:
    if Path(path).suffix not in CODE or path in GENERATED:
        return []
    rows = defaultdict(list)
    if path.endswith(".py"):
        docs = {
            (node.body[0].value.lineno, node.body[0].value.col_offset)
            for node in ast.walk(_tree(text))
            if isinstance(node, (ast.Module, *DEFINITION_TYPES))
            and ast.get_docstring(node) is not None
        }
        try:
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if (
                    token.type
                    in {
                        tokenize.NAME,
                        tokenize.NUMBER,
                        tokenize.STRING,
                        tokenize.OP,
                        tokenize.ERRORTOKEN,
                    }
                    and token.start not in docs
                ):
                    rows[token.start[0]].append(token.string)
        except (tokenize.TokenError, IndentationError):
            return []
    else:
        line, position = 1, 0
        for token in TOKENS.finditer(text):
            line += text.count("\n", position, token.start())
            position = token.start()
            if not token.group().startswith(("/*", "//", "<!--")):
                rows[line].append(token.group())
    return [(line, " ".join(tokens)) for line, tokens in sorted(rows.items())]


def _windows(rows: list[tuple[int, str]]):
    return [
        (rows[index][0], tuple(text for _line, text in rows[index : index + 8]))
        for index in range(len(rows) - 7)
    ]


def _added_windows(path: str, old: str | None, new: str):
    before, after = _code_lines(path, old or ""), _code_lines(path, new)
    added = {
        index
        for tag, _a, _b, start, end in SequenceMatcher(
            a=[row[1] for row in before], b=[row[1] for row in after]
        ).get_opcodes()
        if tag in {"insert", "replace"}
        for index in range(start, end)
    }
    kept = Counter(window for _line, window in _windows(before))
    counts = Counter(window for _line, window in _windows(after))
    wanted = {
        window
        for index, (_line, window) in enumerate(_windows(after))
        if counts[window] > kept[window] and all(item in added for item in range(index, index + 8))
    }
    return after, wanted


def _node_environment() -> dict[str, str]:
    """Do not let inherited Node injection replace the verified scanner."""
    environment = dict(os.environ)
    for name in ("NODE_OPTIONS", "NODE_PATH"):
        environment.pop(name, None)
    return environment


def _npm_package(installed: Path, entry: dict, cache: Path) -> None:
    """Check installed bytes against the locked archive, not an installed stamp."""
    integrity = entry.get("integrity", "")
    if not isinstance(integrity, str) or not integrity.startswith("sha512-"):
        raise ValueError("jscpd.lock_unreadable: package integrity is missing")
    try:
        digest = base64.b64decode(integrity[7:], validate=True)
        if len(digest) != 64:
            raise ValueError("invalid SHA-512 digest")
        key = digest.hex()
        data = (cache / "_cacache/content-v2/sha512" / key[:2] / key[2:4] / key[4:]).read_bytes()
        if hashlib.sha512(data).digest() != digest:
            raise ValueError("cached archive hash differs from the lock")
        expected = {}
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for item in archive.getmembers():
                if item.isdir():
                    continue
                parts = item.name.split("/")
                if (
                    not item.isfile()
                    or len(parts) < 2
                    or parts[0] != "package"
                    or any(part in {"", ".", ".."} for part in parts[1:])
                ):
                    raise ValueError("unsafe archive member")
                stream = archive.extractfile(item)
                name = "/".join(parts[1:])
                if stream is None or name in expected:
                    raise ValueError("duplicate or unreadable archive member")
                expected[name] = hashlib.sha512(stream.read()).digest()
        if installed.is_symlink() or installed.is_junction():
            raise ValueError("installed package is a link")
        actual = {}

        def unreadable(error: OSError) -> None:
            raise error

        for folder, directories, files in os.walk(installed, followlinks=False, onerror=unreadable):
            for name in (*directories, *files):
                path = Path(folder) / name
                if path.is_symlink() or path.is_junction():
                    raise ValueError("installed package contains a link")
            for name in files:
                path = Path(folder) / name
                actual[path.relative_to(installed).as_posix()] = hashlib.sha512(
                    path.read_bytes()
                ).digest()
        if not expected or actual != expected:
            raise ValueError("installed package differs from its locked archive")
    except (OSError, ValueError, tarfile.TarError) as error:
        raise ValueError(
            f"jscpd.integrity_unavailable: {installed}: {error}; run npm ci and retry"
        ) from error


def jscpd_runner(
    root: Path, interpreter: Path, manifests: tuple[bytes, bytes] | None = None
) -> Path:
    """Bind every executable npm package to the staged lock's archive bytes."""
    package = Path("scripts/test-shape-tools")
    try:
        manifest, lock = manifests or tuple(
            (root / package / name).read_bytes() for name in ("package.json", "package-lock.json")
        )
        manifest, lock = json.loads(manifest), json.loads(lock)
        packages = lock["packages"]
        if manifest["devDependencies"] != packages[""]["devDependencies"]:
            raise ValueError("package manifest differs from its lock")
        entry = packages["node_modules/jscpd"]
        if entry["version"] != manifest["devDependencies"]["jscpd"]:
            raise ValueError("jscpd version differs from its lock")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"jscpd.lock_unreadable: restore the staged package manifests: {error}"
        ) from error
    modules = root / package / "node_modules"
    if not (modules / "jscpd/run-jscpd.js").is_file():
        if (
            interpreter.parent.name.lower() not in {"scripts", "bin"}
            or interpreter.parent.parent.name != ".venv"
        ):
            raise ValueError(
                "jscpd.unavailable: install the locked tool in scripts/test-shape-tools"
            )
        modules = interpreter.parent.parent.parent / package / "node_modules"
    if not (modules / "jscpd/run-jscpd.js").is_file():
        raise ValueError("jscpd.unavailable: install the locked tool in scripts/test-shape-tools")
    cache = os.environ.get("npm_config_cache") or os.environ.get("NPM_CONFIG_CACHE")  # noqa: SIM112
    npm, node = shutil.which("npm"), shutil.which("node")
    if not node or (not cache and not npm):
        raise ValueError("jscpd.unavailable: install Node.js and the locked tool")
    try:
        if not cache:
            cache = spawn.run(
                [npm, "config", "get", "cache", "--offline"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
                env=_node_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout.strip()
        _npm_package(modules / "jscpd", entry, Path(cache))
        choose = (
            "const m=require(process.argv[1]);const k=m.getPlatformKey();"
            "if(!k)process.exit(2);process.stdout.write(m.PLATFORM_MAP[k].packageName)"
        )
        name = spawn.run(
            [node, "-e", choose, str(modules / "jscpd/platform-map.js")],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
            env=_node_environment(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip()
        if not re.fullmatch(r"jscpd-[a-z0-9-]+", name):
            raise ValueError("platform package name is invalid")
        _npm_package(modules / name, packages["node_modules/" + name], Path(cache))
    except (OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        raise ValueError(f"jscpd.integrity_unavailable: {error}; run npm ci and retry") from error
    return modules / "jscpd/run-jscpd.js"


def clone_problems_batch(
    changes: list[tuple[str, str | None, str]],
    corpus: dict[str, str],
    *,
    manifests: tuple[bytes, bytes] | None = None,
) -> list[list[str]]:
    """Find exact added eight-line copies in staged blobs, using one jscpd scan."""
    prepared = [(path, *_added_windows(path, old, new)) for path, old, new in changes]
    wanted = set().union(*(windows for _path, _rows, windows in prepared))
    if not wanted:
        return [[] for _change in changes]
    anchors = re.compile(
        "|".join(re.escape("".join(max(window, key=len).split())) for window in wanted)
    )
    texts = dict(corpus)
    texts.update((path, new) for path, _old, new in changes)
    candidates = {
        path: _code_lines(path, text)
        for path, text in sorted(texts.items())
        if path not in GENERATED and anchors.search("".join(text.split()))
    }
    candidates = {path: rows for path, rows in candidates.items() if len(rows) >= 8}
    origins = defaultdict(list)
    if candidates:
        node = shutil.which("node")
        if not node:
            raise ValueError(
                "jscpd unavailable; run npm ci --ignore-scripts in scripts/test-shape-tools"
            )
        runner = jscpd_runner(ROOT, Path(sys.executable), manifests)
        with tempfile.TemporaryDirectory(prefix="test-clones-") as folder:
            root = Path(folder)
            inputs, report = root / "input", root / "report"
            inputs.mkdir()
            names = {}
            for index, (path, rows) in enumerate(candidates.items()):
                name = f"{index:08}.py"
                names[name] = (path, rows)
                # Each executable line becomes one opaque token, preserving literal
                # values and cross-language equality while jscpd finds line sequences.
                (inputs / name).write_text(
                    "".join(
                        "x"
                        + hashlib.sha256(line.encode("utf-8", "surrogateescape")).hexdigest()
                        + "()\n"
                        for _number, line in rows
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
            # jscpd counts the distance between line endpoints, so seven admits
            # an exact eight-line block; the existing window check filters shorter hits.
            spawn.run(
                [
                    node,
                    str(runner),
                    "--min-lines",
                    "7",
                    "--min-tokens",
                    "1",
                    "--skip-comments",
                    "--no-gitignore",
                    "--fail-on-empty",
                    "--workers",
                    "1",
                    "--max-size",
                    str(1 + max(len(rows) for rows in candidates.values()) * 68),
                    "--reporters",
                    "json",
                    "--output",
                    str(report),
                    "--no-tips",
                    str(inputs),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
                env=_node_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            result = json.loads((report / "jscpd-report.json").read_text(encoding="utf-8"))
            if result["statistics"]["total"]["sources"] != len(candidates):
                raise ValueError("jscpd omitted a staged source")
            matches = result["duplicates"]
            for match in matches:
                locations = []
                for key in ("firstFile", "secondFile"):
                    entry = match[key]
                    path, rows = names[Path(entry["name"]).name]
                    start, end = entry["start"], entry["end"]
                    if not (
                        type(start) is int and type(end) is int and 1 <= start <= end <= len(rows)
                    ):
                        raise ValueError("jscpd returned an invalid source range")
                    locations.append((path, dict(_windows(rows[start - 1 : end]))))
                (first, a), (second, b) = locations
                shared = set(a.values()) & set(b.values()) & wanted
                for path, windows in ((first, a), (second, b)):
                    for line, window in windows.items():
                        if window in shared:
                            origins[window].append((path, line))
    results = []
    for path, after, selected in prepared:
        found, last = [], -8
        for index, (line, window) in enumerate(_windows(after)):
            others = sorted(set(origins[window]) - {(path, line)})
            if window in selected and others and index >= last + 8:
                original, start = others[0]
                found.append(
                    f"{path}:{line}: copied 8 lines from {original}:{start}; call its owner"
                )
                last = index
        results.append(found)
    return results


def clone_problems(path: str, old: str | None, new: str, corpus: dict[str, str]) -> list[str]:
    return clone_problems_batch([(path, old, new)], corpus)[0]


def _docstring_lines(text: str) -> dict[str, int]:
    """A test marked `proves`, on itself, its class or its module, has its sentence in its
    registered requirement: it counts as one."""
    tree = _tree(text)
    held = file_holders("", ast.unparse(tree)) if tree.body else []
    marked = {test.node.split("::", 1)[1].replace("::", ".") for test in held if test.ids}
    return {
        name: len((ast.get_docstring(node) or "").splitlines()) or int(name in marked)
        for name, node in _definitions(tree)
        if isinstance(node, FUNCTION_TYPES) and node.name.startswith("test_")
    }


def _given(value: ast.AST | None) -> bool:
    return value is not None and not (isinstance(value, ast.Constant) and value.value is None)


def _names_every_case(ids: ast.AST | None) -> bool:
    """ids= names each case; a literal list with a None entry leaves pytest's value id there."""
    if isinstance(ids, ast.List | ast.Tuple):
        return bool(ids.elts) and all(_given(entry) for entry in ids.elts)
    return _given(ids)


def _named(case: ast.AST) -> bool:
    return (
        isinstance(case, ast.Call)
        and getattr(case.func, "attr", getattr(case.func, "id", "")) == "param"
        and any(word.arg == "id" and _given(word.value) for word in case.keywords)
    )


def _literal(value: ast.AST) -> bool:
    """A value pytest names by itself, stably: a str, bytes, number, bool or None literal, a
    signed number included."""
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub | ast.UAdd):
        value = value.operand
    return isinstance(value, ast.Constant)


def _several(names: ast.AST | None) -> bool:
    """Whether the argnames name more than one argument, so each case holds a value for each."""
    if isinstance(names, ast.Constant) and isinstance(names.value, str):
        return "," in names.value
    return isinstance(names, ast.List | ast.Tuple) and len(names.elts) > 1


def _self_named(case: ast.AST, several: bool) -> bool:
    """A case pytest gives a readable, stable id: one with id=, or whose values are all literals.
    Any other value (an object, a list, a dict, a name) pytest names by its position (argname0,
    argname1), which shifts when a case is added or removed."""
    if _named(case):
        return True
    if isinstance(case, ast.Call) and getattr(case.func, "attr", getattr(case.func, "id", "")) == (
        "param"
    ):
        values = case.args
    else:
        values = case.elts if several and isinstance(case, ast.List | ast.Tuple) else [case]
    return bool(values) and all(_literal(value) for value in values)


def _unnamed(node: ast.AST) -> list[str]:
    """A test's parametrize sites with a case pytest would name by its position, under no ids=
    naming every case. A literal case reads as its id already; a computed case list cannot be
    read, so it counts."""
    sites = []
    for mark in getattr(node, "decorator_list", ()):
        if not (isinstance(mark, ast.Call) and getattr(mark.func, "attr", "") == "parametrize"):
            continue
        given = {word.arg: word.value for word in mark.keywords}
        cases = mark.args[1] if len(mark.args) > 1 else given.get("argvalues")
        names = mark.args[0] if mark.args else given.get("argnames")
        listed = isinstance(cases, ast.List | ast.Tuple) and cases.elts
        if not _names_every_case(given.get("ids")) and not (
            listed and all(_self_named(case, _several(names)) for case in cases.elts)
        ):
            sites.append(f"line {mark.lineno} parametrize({ast.unparse(names) if names else ''})")
    return sites


def _tests(tree: ast.AST) -> dict[str, ast.AST]:
    return {
        name: node
        for name, node in _definitions(tree)
        if isinstance(node, FUNCTION_TYPES) and node.name.startswith("test_")
    }


def _owners(tree: ast.AST, found: tuple[tuple[str, int], ...]) -> Counter:
    """Each PT finding counted under the innermost function holding its line, decorators
    included, else the module."""
    spans = [
        (min([node.lineno] + [mark.lineno for mark in node.decorator_list]), node.end_lineno, name)
        for name, node in _definitions(tree)
        if isinstance(node, FUNCTION_TYPES)
    ]
    counts = Counter()
    for rule, line in found:
        holding = [span for span in spans if span[0] <= line <= span[1]]
        counts[(max(holding)[2] if holding else "<module>", rule)] += 1
    return counts


def _test_problems(
    path: str, old: str | None, new: str, ruff: tuple[tuple, tuple] | None
) -> list[str]:
    """An added or changed test names each parametrize case; no function grows a PT finding."""
    before, after = _tree(old or ""), _tree(new)
    was, found = _tests(before), []
    for name, node in _tests(after).items():
        if name in was and ast.dump(was[name]) == ast.dump(node):
            continue
        found += [
            f"{path}::{name}: {site} has a case pytest names by its position; "
            "give each such case pytest.param(..., id=...), or ids= (rule 5)"
            for site in _unnamed(node)
        ]
    if ruff is not None:
        held, now = _owners(before, ruff[0]), _owners(after, ruff[1])
        found += [
            f"{path}::{name}: Ruff {rule} findings grow {held[(name, rule)]} -> {count}; "
            f"{PT_FIXES[rule]}"
            for (name, rule), count in sorted(now.items())
            if count > held[(name, rule)]
        ]
    return found


def _window_flags(path: str, source: str) -> list[str]:
    """Test and tool children must inherit a hidden console, never request a new one."""
    forbidden = {"DETACHED_" + "PROCESS", "CREATE_NEW_" + "CONSOLE"}
    found = []
    for node in ast.walk(_tree(source)):
        value = (
            node.attr
            if isinstance(node, ast.Attribute)
            else node.id
            if isinstance(node, ast.Name)
            else node.name
            if isinstance(node, ast.alias)
            else node.args[1].value
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            else None
        )
        if value in forbidden:
            found.append(
                f"{path}:{node.lineno}: {value} opens a visible console; use CREATE_NO_WINDOW"
            )
    return found


def problems(
    path: str,
    old: str | None,
    new: str,
    ruff: tuple[tuple, tuple] | None = None,
    docstrings: tuple[tuple[str, int], ...] = (),
) -> list[str]:
    """Growth a staged source or test change refuses; existing debt only shrinks. A source's
    ``docstrings`` are its head's Ruff docstring findings (``ruff_docstrings``)."""
    if path in GENERATED:
        return []
    windows = (
        _window_flags(path, new)
        if path.startswith(("tests/", "scripts/")) and path.endswith(".py")
        else []
    )
    if path.startswith("scripts/"):
        return windows
    if path.startswith("src/"):
        found = []
        size = len(new.encode("utf-8", errors="surrogateescape"))
        grew = size > len((old or "").encode("utf-8", errors="surrogateescape"))
        if size > CAP_BYTES and grew and path not in WORD_CATALOGS | VENDORED_DATA:
            # A size cap is met by splitting, which leaves the code no better: review judges it.
            print(f"test_shape.signal: {path}: {size // 1024} KB (a review signal, never refused)")
        if path.endswith(".py"):
            found += undocumented(path, old, new, docstrings)
            before, after = _tree(old or ""), _tree(new)
            debt = Counter(line.split(": ", 1)[1] for line in _lint(path, before))
            current = sorted(_lint(path, after))
            counts = Counter(line.split(": ", 1)[1] for line in current)
            found += [
                line
                for line in current
                if counts[line.split(": ", 1)[1]] > debt[line.split(": ", 1)[1]]
            ]
        return found
    if not path.startswith("tests/"):
        return []
    found = windows

    def counted(text: str) -> int:  # tracing a test to its requirement never grows its file
        return len(PROVES_LINE.sub("", text).encode("utf-8", errors="surrogateescape"))

    size = counted(new)
    if size > CAP_BYTES and size > counted(old or ""):  # met by splitting files: review judges it
        print(f"test_shape.signal: {path}: {size // 1024} KB (a review signal, never refused)")
    # a line only re-indented is kept, not added
    before = {line.strip() for line in (old or "").splitlines()}
    added = [line for line in new.splitlines() if line.strip() not in before]
    if not path.startswith("tests/structural/"):
        for line in added:
            pin = PIN.search(line)
            if pin and len(pin.group(2).split()) >= 5:
                found.append(
                    f"{path}: pins '{pin.group(2)[:40]}'; check its key and facts (rule 4)"
                )
    if not path.endswith(".py"):
        return found
    found += _test_problems(path, old, new, ruff)
    children = test_module_children(_tree(new))
    if len(children) > len(test_module_children(_tree(old or ""))):
        found += [
            f"{path}:{line}: a child process starts on a test module (tests.*), which a fresh "
            "interpreter may import from another checkout; start it on a source module "
            "(devtools.testing.child_targets holds the test-only targets)"
            for line in children
        ]
    if path.startswith("tests/unit/"):
        found.extend(
            row for row in _lint(path, _tree(new)) if row.startswith("testing.layer_contract:")
        )
    if not path.startswith("tests/structural/") and any("inspect.getsource(" in x for x in added):
        found.append(f"{path}: reads source text; test the behaviour at the owner's seam (rule 3)")
    for line in added:
        match = SLEEP.search(line)
        if match and float(match.group(1)) >= 1:
            found.append(f"{path}: sleeps {match.group(1)} s; drive the clock or the step (rule 7)")
    was = _docstring_lines(old or "")
    for name, count in _docstring_lines(new).items():
        if name not in was and count == 0:
            found.append(f"{path}::{name}: missing requirement docstring (rule 1)")
        if count > 3 and count > was.get(name, 0):
            found.append(
                f"{path}::{name}: a {count}-line docstring; one requirement sentence (rule 1)"
            )
    return found


def _git(*args: str, root: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return spawn.run(
        ["git", *args],
        cwd=root,
        input="",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def weekly_report(root: Path = ROOT) -> dict:
    """Static review candidates, exact copies, seven-day growth and current shape debt."""
    paths = _git(
        "ls-files", "-z", "--", "src", "scripts", "tests", "config", root=root
    ).stdout.split("\0")
    texts = _blobs(["HEAD:" + path for path in paths if path], root)
    sources = {spec[5:]: text for spec, text in texts.items()}
    trees = {path: _tree(text) for path, text in sources.items() if path.endswith(".py")}

    def readers(tree):
        found = Counter()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                found.update([node.id])
            elif isinstance(node, ast.Attribute):
                found.update([node.attr])
            elif isinstance(node, ast.alias):
                found.update(node.name.split("."))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.update(re.findall(r"\b[A-Za-z_]\w*\b", node.value))
        return found

    mentions = sum((readers(tree) for tree in trees.values()), Counter())
    for path, text in sources.items():
        if path.startswith("config/"):
            mentions.update(re.findall(r"\b[A-Za-z_]\w*\b", text))
    candidates = [
        {"path": path, "line": node.lineno, "name": name, "kind": type(node).__name__}
        for path, tree in trees.items()
        if path.startswith("src/")
        for name, node in _definitions(tree)
        if mentions[node.name] <= readers(node)[node.name]
    ]
    index, copies = {}, []
    for path, text in sorted(sources.items()):
        if not path.startswith("src/"):
            continue
        last = -8
        for offset, (line, window) in enumerate(_windows(_code_lines(path, text))):
            if window in index and offset >= last + 8:
                original, start = index[window]
                copies.append(
                    {"original": original, "original_line": start, "copy": path, "line": line}
                )
                last = offset
            index.setdefault(window, (path, line))
    base = _git("rev-list", "-1", "--before=7 days ago", "HEAD", root=root).stdout.strip()
    growth = _git(
        "diff",
        "--numstat",
        base or _git("hash-object", "-t", "tree", "--stdin", root=root).stdout.strip(),
        "HEAD",
        "--",
        "src",
        root=root,
    )
    rows = []
    for row in growth.stdout.splitlines():
        added, removed, path = row.split("\t", 2)
        if added.isdecimal() and removed.isdecimal():
            rows.append(
                {
                    "path": path,
                    "added": int(added),
                    "removed": int(removed),
                    "net": int(added) - int(removed),
                }
            )
    rows.sort(key=lambda row: (-row["net"], row["path"]))
    frequency = Counter((copy["original"], copy["original_line"]) for copy in copies)
    copies.sort(key=lambda copy: -frequency[copy["original"], copy["original_line"]])
    return {
        "candidates": candidates,
        "clones": copies[:20],
        "growth": rows[:20],
        "debt": [
            problem
            for path, text in sources.items()
            if path.startswith("src/")
            for problem in problems(path, None, text)
        ],
        "limit": "Static reader candidates require review; dynamic readers are not proven absent.",
    }


def _data_file_problems(
    staged: list[str], moved: dict, parents: tuple[str, ...], head: str = ":"
) -> list[str]:
    """The rule for test data: a path no parent holds, under tests/, that is neither code nor
    prose. The tree is its baseline, and a byte-identical move keeps its standing."""
    data = [
        path
        for path in staged
        if path.startswith("tests/")
        and path not in moved
        and Path(path).suffix.lower() not in CODE | {".md"}
    ]
    if not data:
        return []
    try:
        held = set()
        for parent in parents:
            listed = _git("ls-tree", "-r", "-z", "--name-only", parent, "--", *data)
            if listed.returncode:
                raise ValueError(f"{parent} is unreadable")
            held.update(listed.stdout.split("\0"))
        fresh = [path for path in data if path not in held]
        if not fresh:
            return []
        policy = validate_policy(json.loads(_blobs([head + REGISTRIES[0]])[head + REGISTRIES[0]]))
        return new_data_file_problems(fresh, policy)
    except (KeyError, OSError, ValueError, subprocess.SubprocessError) as error:
        return [f"TEST-DATA unmeasured: {error}; restore the staged registry and retry"]


def _shape_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weekly", action="store_true", help="Print a read-only source review as JSON"
    )
    parser.add_argument(
        "--base",
        help="judge the committed head against its merge base with this ref (the commit stage)",
    )
    parser.add_argument("--semgrep", action="store_true", help="include the slower source rules")
    parser.add_argument(
        "--semgrep-only",
        action="store_true",
        help="only the slower source rules: the commit stage's share, as the hook runs the rest",
    )
    return parser.parse_args(argv)


def _semgrep_only(staged: list[str], sources: list[str], head: str) -> int:
    """The semgrep source rules alone, held to the anti-pattern baseline as ``--semgrep`` holds
    them: the hook and every stage run them as their own step, beside this check's other
    rules, so a guard run judges each rule once."""
    python = [path for path in sources if path.endswith(".py") and not path.startswith("tests/")]
    try:
        texts = _blobs(
            [head + path for path in (*python, *ANTIPATTERN_SIDES)]
            + [head + ANTIPATTERN_SEMGREP, head + ANTIPATTERN_CASES]
        )
        if ANTIPATTERN_SEMGREP in staged or ANTIPATTERN_CASES in staged:
            _semgrep_selftest(texts[head + ANTIPATTERN_SEMGREP], texts[head + ANTIPATTERN_CASES])
        measured = semgrep_antipatterns(
            {path: texts[head + path] for path in python}, texts[head + ANTIPATTERN_SEMGREP]
        )
        found = antipattern_growth(antipattern_sides(texts, head), measured)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(
            f"TEST-LAYOUT staged snapshot unreadable: {error}; restore the Git index and retry",
            file=sys.stderr,
        )
        return 1
    for line in found:
        print(line, file=sys.stderr)
    return 1 if found else 0


def _shape_moves(base: str | None) -> tuple[dict[str, str], dict[str, str]]:
    pairs = staged_changes(ROOT, base=base, tree="HEAD") if base else staged_changes(ROOT)
    return staged_moves(ROOT, pairs=pairs), unique_moves(pairs)


def unpackaged_test_dirs(changed: Iterable[str], tracked: Iterable[str]) -> list[str]:
    """Each test folder a changed test module or conftest sits in, or under, that is no package:
    without its ``__init__.py`` pytest imports the file by its bare name, so two same-named files
    collide."""
    packages = {path.rsplit("/", 1)[0] for path in tracked if path.endswith("/__init__.py")}
    missing: set[str] = set()
    for path in changed:
        name = path.rsplit("/", 1)[-1]
        collected = name == "conftest.py" or (name.startswith("test_") and name.endswith(".py"))
        if not (path.startswith("tests/") and collected):
            continue
        folder = path.rsplit("/", 1)[0]
        while (folder == "tests" or folder.startswith("tests/")) and folder not in packages:
            missing.add(folder)
            folder = folder.rsplit("/", 1)[0] if "/" in folder else ""
    return [
        f"{folder}: a test folder with no __init__.py imports its modules by bare name, so a "
        f"same-named test file elsewhere collides; add {folder}/__init__.py"
        for folder in sorted(missing)
    ]


def main(argv: list[str] | None = None) -> int:
    args = _shape_args(argv)
    if args.weekly:
        print(json.dumps(weekly_report(), ensure_ascii=False))
        return 0
    if args.base:  # the committed head against its merge base: the stage's snapshot
        found_base = _git("merge-base", args.base, "HEAD")
        if found_base.returncode:
            print(
                f"TEST-LAYOUT base {args.base} unreadable; name a reachable base", file=sys.stderr
            )
            return 1
        parents, head = (found_base.stdout.strip(),), "HEAD:"
        listing = _git("diff", "--name-only", "-z", "--diff-filter=ACMR", parents[0], "HEAD")
    else:  # the staged index against HEAD: the hook's snapshot
        head = ":"
        listing = _git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
    if listing.returncode:
        print(
            "TEST-LAYOUT changed files unreadable; restore the Git index and retry", file=sys.stderr
        )
        return 1
    staged = listing.stdout.split("\0")
    if not args.base:
        # A merge adds only what neither parent holds: the merged branch was checked there.
        merging = _git("rev-parse", "-q", "--verify", "MERGE_HEAD").returncode == 0
        parents = ("HEAD", "MERGE_HEAD") if merging else ("HEAD",)
    selected, anti_sources = (
        [
            path
            for path in staged
            if path.startswith("src/")
            or (path.startswith("scripts/") and path.endswith(".py"))
            or (path.startswith("tests/") and path.endswith((".py", ".cjs")))
        ],
        [
            path
            for path in staged
            if path.endswith((".py", ".sh"))
            or (path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml")))
        ],
    )
    if args.semgrep_only:
        return _semgrep_only(staged, anti_sources, head)
    try:
        moved, identical = _shape_moves(parents[0] if args.base else None)
        origin = {path: moved.get(path, path) for path in selected}
        candidates = [
            p
            for p in selected
            if p.startswith("tests/")
            and p.endswith(".py")
            and (Path(p).name.startswith("test_") or Path(p).name.endswith("_test.py"))
            and p not in moved
        ]
        texts = _blobs(
            [head + path for path in selected]
            + [head + path for path in anti_sources]
            + [  # where the change wrote, for the size signals
                f"{parents[0]}:{identical.get(path, path)}"
                for path in anti_sources
                if path.endswith(".py")
            ]
            + [parent + ":" + origin[path] for parent in parents for path in selected]
            + [parent + ":" + path for parent in parents for path in candidates]
            + [head + path for path in ANTIPATTERN_SIDES]
            + [head + TEST_ANTIPATTERN_RULES, head + TEST_ANTIPATTERN_CASES]
            + ([head + ANTIPATTERN_SEMGREP] if args.semgrep else [])
            + ([head + ANTIPATTERN_CASES] if args.semgrep else [])
            + [parent + ":" + path for parent in parents for path in ANTIPATTERN_SIDES]
            + ([head + "config/registries/requirements.json"] if candidates else [])
            + [head + "pyproject.toml"]
        )
        readers = _table_readers(anti_sources, texts, head)
        found, corpus = (
            _data_file_problems(staged, moved, parents, head)
            + _antipattern_problems(
                staged,
                anti_sources,
                texts,
                parents,
                head,
                moves=moved,
                semgrep=args.semgrep,
                readers=readers,
            ),
            None,
        )
    except (
        OSError,
        ValueError,
        IndexError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        print(
            f"TEST-LAYOUT staged snapshot unreadable: {error}; restore the Git index and retry",
            file=sys.stderr,
        )
        return 1
    tracked = (
        _git("ls-tree", "-r", "--name-only", "-z", "HEAD", "--", "tests")
        if args.base
        else _git("ls-files", "-z", "--", "tests")
    )
    if tracked.returncode:
        print("TEST-LAYOUT test listing unreadable; restore the Git index", file=sys.stderr)
        return 1
    found += unpackaged_test_dirs(staged, tracked.stdout.split("\0"))
    added = [p for p in candidates if all(parent + ":" + p not in texts for parent in parents)]
    if added:
        try:
            registry = texts.get(head + "config/registries/requirements.json")
            if registry is None:
                raise ValueError("the staged requirements registry is missing")
            rows = parse_requirements(registry)
            sources = _git("ls-files", "-z", "--", "src", "scripts")
            if sources.returncode:
                raise ValueError("staged source listing is unreadable")
            source_paths = set(sources.stdout.split("\0"))
            for path in added:
                if head + path not in texts:
                    raise ValueError(f"staged test {path} is unreadable")
                found += new_test_file_problems(path, texts[head + path], rows, source_paths)
        except (ValueError, TypeError, OSError) as error:
            found.append(
                f"TEST-LAYOUT placement unmeasured: {error}; "
                "restore the staged registry and source index"
            )
    python_sources = [p for p in selected if p.startswith("src/") and p.endswith(".py")]
    test_sources = [p for p in selected if p.startswith("tests/") and p.endswith(".py")]
    test_specs = [head + path for path in test_sources] + [
        parent + ":" + origin[path] for path in test_sources for parent in parents
    ]
    try:
        scores = ruff_test_findings({s: texts[s] for s in test_specs if s in texts})
        config = texts.get(head + "pyproject.toml")  # a tree without one declares no rules
        docstrings = (
            ruff_docstrings(
                {path: texts[head + path] for path in python_sources if head + path in texts},
                config,
            )
            if config is not None
            else {}
        )
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(
            f"TEST-LAYOUT Ruff measurement unavailable: {error}; restore Ruff and retry",
            file=sys.stderr,
        )
        return 1
    grouped, copies, locations = [], [], []
    for path in selected:
        if head + path in texts:
            new = texts[head + path]
            each = []
            for parent in parents:
                old = texts.get(parent + ":" + origin[path])
                measurement = (
                    (scores.get(parent + ":" + origin[path], ()), scores[head + path])
                    if path in test_sources
                    else None
                )
                try:
                    current = problems(path, old, new, measurement, docstrings.get(path, ()))
                except (ValueError, KeyError) as error:
                    print(
                        f"TEST-LAYOUT Ruff measurement unavailable: {error}; retry", file=sys.stderr
                    )
                    return 1
                if path.startswith("src/") and _added_windows(path, old, new)[1]:
                    if corpus is None:
                        listing = _git("ls-files", "-z", "--", "src", "scripts")
                        if listing.returncode:
                            print(
                                "TEST-LAYOUT clone inputs unreadable; restore the Git index",
                                file=sys.stderr,
                            )
                            return 1
                        paths = listing.stdout.split("\0")
                        corpus = {
                            spec[len(head) :]: text
                            for spec, text in _blobs(
                                [head + item for item in paths if item]
                            ).items()
                        }
                    copies.append((path, old, new))
                    locations.append((len(grouped), len(each)))
                each.append(set(current))
            grouped.append(each)
    if copies:
        try:
            names = (
                head + "scripts/test-shape-tools/package.json",
                head + "scripts/test-shape-tools/package-lock.json",
            )
            staged_manifests = _blobs(list(names))
            manifests = tuple(
                staged_manifests[name].encode("utf-8", errors="surrogateescape") for name in names
            )
            for (group, parent), matches in zip(
                locations,
                clone_problems_batch(copies, corpus or {}, manifests=manifests),
                strict=True,
            ):
                grouped[group][parent].update(matches)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            print(
                f"TEST-LAYOUT clone detection unmeasured: {error}; "
                "restore the locked tool and retry",
                file=sys.stderr,
            )
            return 1
    for each in grouped:
        found += sorted(set.intersection(*each))
    found += [  # a test the change writes names the OS its OS-only call needs
        line
        for path in test_sources
        for line in os_marks.refusals(
            path, texts[head + path], *(texts.get(p + ":" + origin[path], "") for p in parents)
        )
    ]
    for line in found:
        print(line, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
