"""The tests that read original data, and the Large rules that keep them out of the merge stage.

Original data is a QA copy, the QA evidence roots, the original research inputs, or a workspace
built from them: every root the tracked evidence-roots declaration declares (its owner is
devtools.architecture.evidence_roots), or one a local override names. A
test reaches one through the `evidence_roots` fixture, `resolve_evidence_root`, `EvidenceRoots`,
the `real_evidence` mark, the override's variable or file (by name or path), or a declared root's
name or environment key, in its own body, decorators, parameters or module's `pytestmark`, or
through a module-level name, a fixture it requests (as a parameter or by `usefixtures` on it, its
class or its module) or a helper function that reaches one, followed to a fixpoint.

Every id such a test can be collected under, the bare node and each parameter id, is kept out of
the merge stage by the Large rule collection resolves for it (`sizes._rule`, the longest matching
pattern). A test that keeps original data runs only in the
nightly (nightly_only) and names in `original_data` the one reason that keeps it: an upgrade or
restore replay of a 0.2.7 workspace on its QA copy. A test its area still decides (`deciding`) may
be release_only and name no reason until it is decided.
"""

import ast
import json
import re
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from devtools.testing.sizes import _rule

REASONS = frozenset({"upgrade replay", "restore replay"})
"""The reasons a test keeps original data: a 0.2.7 workspace's upgrade or restore, replayed on its
QA copy."""
CONTRACTS = frozenset({"nightly_only", "release_only"})
READERS = frozenset(
    {
        "evidence_roots",
        "resolve_evidence_root",
        "EvidenceRoots",
        "real_evidence",
        "ALPHALATTICE_EVIDENCE_ROOTS",
        "evidence-roots.local.json",
    }
)
"""The names a test reaches a declared root through, the local override's included."""
type Function = ast.FunctionDef | ast.AsyncFunctionDef


def _registry(path: Path, key: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))[key]
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"original_data.unreadable: {path}: {error}") from error


def markers(root: Path) -> frozenset[str]:
    """Every declared root's name and environment key, and the reader names; an unreadable
    declaration is refused by name."""
    declared = _registry(root / "config" / "evidence-roots.json", "roots")  # as its owner spells it
    keys = {row["environment_key"] for row in declared.values() if row.get("environment_key")}
    return frozenset({*declared, *keys, *READERS})


def large(root: Path) -> list[dict[str, Any]]:
    """The Large rules; an unreadable size registry is refused by name."""
    path = root / "config/registries/test-sizes.json"
    rules = _registry(path, "large")
    if not isinstance(rules, list) or not all(
        isinstance(rule, dict) and isinstance(rule.get("pattern"), str) for rule in rules
    ):
        raise ValueError(f"original_data.unreadable: {path}: not a list of rules")
    return rules


def _names(node: ast.AST) -> set[str]:
    """The identifiers, parameters (the fixtures a function requests), attributes (the helpers it
    calls through a module) and strings a node holds."""
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.arg):
            found.add(child.arg)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            found |= {child.value, re.split(r"[\\/]", child.value)[-1]}  # a path, by its file too
    return found


def _uses(marks: Iterable[ast.expr]) -> set[str]:
    """The fixtures `pytest.mark.usefixtures` requests by name among `marks`."""
    return {
        argument.value
        for mark in marks
        for call in ast.walk(mark)
        if isinstance(call, ast.Call) and ast.unparse(call.func).endswith("usefixtures")
        for argument in call.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }


def _functions(tree: ast.Module) -> Iterable[tuple[str, Function, list[ast.expr]]]:
    """Each module-level function and each class's method, by its node-id suffix, with its
    class's decorators."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node.name, node, []
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef):
                    yield f"{node.name}::{method.name}", method, node.decorator_list


def _fixture(function: Function) -> bool:
    """Whether a decorator declares it a fixture (`fixture`, `pytest.fixture(...)`); a
    `usefixtures` mark requests one."""
    return any(
        ast.unparse(getattr(decorator, "func", decorator)).rpartition(".")[2] == "fixture"
        for decorator in function.decorator_list
    )


def _module(name: str) -> str:
    """A dotted module name as its file under the root: `tests.x.y` -> tests/x/y.py."""
    return name.replace(".", "/") + ".py"


def _calls(relative: str, tree: ast.Module, function: Function) -> set[tuple[str, str]]:
    """The functions `function` calls, by their module file and name: one defined in its own
    module, imported from a module by name, or reached through an imported module's alias."""
    local = {name for name, *_ in _functions(tree) if "::" not in name}
    imported: dict[str, tuple[str, str]] = {}
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                imported[alias.asname or alias.name] = (_module(node.module), alias.name)
                aliases[alias.asname or alias.name] = _module(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = _module(alias.name)
    called: set[tuple[str, str]] = set()
    for child in ast.walk(function):
        if not isinstance(child, ast.Call):
            continue
        callee = child.func
        if isinstance(callee, ast.Name) and callee.id in local:
            called.add((relative, callee.id))
        elif isinstance(callee, ast.Name) and callee.id in imported:
            called.add(imported[callee.id])
        elif (
            isinstance(callee, ast.Attribute)
            and isinstance(callee.value, ast.Name)
            and callee.value.id in aliases
        ):
            called.add((aliases[callee.value.id], callee.attr))
    return called


def readers(root: Path, paths: Iterable[Path]) -> set[str]:
    """The node ids (one per test function) of the tests in `paths` that reach original data:
    directly, through a module-level name, a fixture it requests (pytest resolves fixtures by
    name) or a helper it calls (resolved by its import), followed to a fixpoint."""
    marks = markers(root)
    nodes: dict[tuple[str, str], tuple[set[str], set[str], set[tuple[str, str]], bool]] = {}
    marked: set[str] = set()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bound = {  # module-level names whose value reaches a declared root
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign) and _names(node.value) & marks
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        pytestmark = [
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
        ]
        if any(_names(mark) & (marks | bound) for mark in pytestmark):
            marked.add(relative)
        for name, function, owner in _functions(tree):
            direct = bool(_names(function) & (marks | bound))
            requested = {arg.arg for arg in function.args.args}
            requested |= _uses([*function.decorator_list, *owner, *pytestmark])
            nodes[(relative, name)] = (
                requested,
                {name} if _fixture(function) else set(),
                _calls(relative, tree, function),
                direct,
            )
    reaching = {key for key, (*_, direct) in nodes.items() if direct}
    size = -1
    while size != len(reaching):  # grows by at least one node a pass, so ends within len(nodes)
        size = len(reaching)  # a fixture or helper reaches when it reads a marker or uses one
        fixtures = {"evidence_roots", *(name for key in reaching for name in nodes[key][1])}
        reaching |= {
            key
            for key, (requested, _, called, _) in nodes.items()
            if requested & fixtures or called & reaching
        }
    return {
        f"{relative}::{name}"
        for relative, name in nodes
        if name.rpartition("::")[2].startswith("test")
        and not nodes[(relative, name)][1]
        and ((relative, name) in reaching or relative in marked)
    }


def _glob(pattern: str) -> list[str]:
    """A pattern's fnmatch tokens: `*`, `?`, a closed bracket class or one literal character."""
    tokens, at = [], 0
    while at < len(pattern):
        end = at + 1
        if pattern[at] == "[":
            end += pattern[end : end + 1] == "!"
            end += pattern[end : end + 1] == "]"
            end = pattern.find("]", end) + 1 or at + 1
        tokens.append(pattern[at:end])
        at = end
    return tokens


def _star(tokens: list[str], states: set[int]) -> set[int]:
    """`states`, with each state a run of `*` there may match nothing to."""
    grown = set(states)
    for at in states:
        while at < len(tokens) and tokens[at] == "*":
            at += 1
            grown.add(at)
    return grown


def _ids(pattern: str, node: str) -> tuple[bool, bool]:
    """Whether `pattern` matches some parameter id `node[...]`, and whether it matches every one:
    the pattern is run over `node[` and its remaining tokens must end the id with `]`."""
    tokens = _glob(pattern)
    states = _star(tokens, {0})
    for char in node + "[":
        live = [at for at in states if at < len(tokens) and fnmatchcase(char, tokens[at])]
        states = _star(tokens, {at + (tokens[at] != "*") for at in live})
    tails = [tokens[at:] for at in states if at < len(tokens)]
    some = any(fnmatchcase("]", tail[-1]) for tail in tails)
    every = any(
        set(tail) == {"*"} or (set(tail[:-1]) == {"*"} and fnmatchcase("]", tail[-1]))
        for tail in tails
    )
    return some, every


def _effective(node: str, rules: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """Every Large rule collection can resolve for one of `node`'s ids; None for an id with none.
    The bare node's is `_rule`'s. A parameter id's is the longest pattern matching it: a rule
    matching every id, or one at least as long matching some id."""
    found: list[dict[str, Any] | None] = [_rule(node, rules)]
    ids = [(rule, *_ids(str(rule["pattern"]), node)) for rule in rules]
    width = max((len(rule["pattern"]) for rule, _, every in ids if every), default=0)
    found += [rule for rule, some, _ in ids if some and len(rule["pattern"]) >= width]
    return found if width else [*found, None]


def unentered(root: Path, nodes: Iterable[str], deciding: frozenset[str]) -> list[str]:
    """Each node with an id no nightly_only or release_only Large rule keeps out of the merge
    stage, and each one not in `deciding` with an id whose rule is not nightly_only with a reason
    that keeps original data."""
    rules = large(root)
    refused = []
    for node in sorted(nodes):
        cases = _effective(node, rules)
        if any(rule is None or rule.get("size_contract") not in CONTRACTS for rule in cases):
            refused.append(f"{node}: a case has no nightly_only or release_only Large rule")
        elif node not in deciding and not all(
            rule is not None
            and rule.get("size_contract") == "nightly_only"
            and rule.get("original_data") in REASONS
            for rule in cases
        ):
            refused.append(
                f"{node}: kept, it needs a nightly_only rule naming an original_data reason"
            )
    return refused


__all__ = ["CONTRACTS", "READERS", "REASONS", "large", "markers", "readers", "unentered"]
