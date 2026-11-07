"""What a whole-tree census reads of one file, so the commit stage selects it by the edge.

A census reads every file under its root, so no import or input names it as a change's reader,
and it used to meet a change it refuses only after the merge. Each facet is a function of one
file's text: the facts of that file the census reads, or a superset of them. The stage selects a
census when a changed file under its row's ``within`` has different facts at base and head
(``selection_readers`` facet rows in config/registries/test-sizes.json). A census that builds
its own answer from its facet is exact; any other facet is a superset and selects more, never
less.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from collections.abc import Set as AbstractSet

LIFECYCLE_STATES = frozenset(
    {
        "QUEUED",
        "RUNNING",
        "DEFERRED",
        "REVIEW_PENDING",
        "CANCEL_REQUESTED",
        "CANCELLED",
        "SUCCEEDED",
        "BLOCKED",
        "RECOVERY_REQUIRED",
    }
)
"""The Task states a lifecycle set holds."""


def deadline_names(source: str) -> set[str]:
    """The deadline holder's exact function discovery: functions with a deadline or a wait cap."""
    found = set()
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = {
            arg.arg
            for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
        }
        assigned = {
            target.id
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
            if isinstance(target, ast.Name)
        }
        if "deadline" in args | assigned or "max_wait" in args:
            found.add(function.name)
    return found


def _members(node: ast.AST) -> frozenset[str]:
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
            if sub.value.id == "TaskLifecycle" and sub.attr in LIFECYCLE_STATES:
                found.add(sub.attr)
        elif (
            isinstance(sub, ast.Constant)
            and isinstance(sub.value, str)
            and sub.value in LIFECYCLE_STATES
        ):
            found.add(sub.value)
    return frozenset(found)


def _literal(node: ast.AST) -> bool:
    return isinstance(node, (ast.Set, ast.Tuple, ast.List)) or (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"frozenset", "set"}
    )


def lifecycle_sets(source: str) -> set[tuple[str, str]]:
    """Each set of two or more lifecycle states in one module: its scope and its members."""
    found: set[tuple[str, str]] = set()

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = f"{scope}.{child.name}" if scope else child.name
            if isinstance(child, (ast.Assign, ast.AnnAssign)) and child.value is not None:
                held = _members(child.value)
                if _literal(child.value) and len(held) >= 2:
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    name = ",".join(ast.unparse(target) for target in targets)
                    found.add((f"{scope}.{name}" if scope else name, ",".join(sorted(held))))
                    continue
            if isinstance(child, ast.Compare) and any(
                isinstance(op, (ast.In, ast.NotIn)) for op in child.ops
            ):
                for compared in child.comparators:
                    held = _members(compared)
                    if _literal(compared) and len(held) >= 2:
                        found.add((scope or "<module>", ",".join(sorted(held))))
            visit(child, inner)

    visit(ast.parse(source), "")
    return found


def class_declarations(source: str) -> set[str]:
    """Each top-level class as a constructor sees it: its name, bases, decorators, annotated
    fields, ``__init__`` signature and import bindings used by bases. The typed-exception census
    resolves exception classes across modules, including imported base aliases."""
    found = set()
    tree = ast.parse(source)
    bases_used = {
        part.id
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for base in node.bases
        for part in ast.walk(base)
        if isinstance(part, ast.Name)
    }
    for item in ast.walk(tree):
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            bound = {
                alias.asname
                or (alias.name.split(".")[0] if isinstance(item, ast.Import) else alias.name)
                for alias in item.names
            }
            if bound & bases_used:
                found.add(ast.dump(item, include_attributes=False))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        init = next(
            (
                ast.unparse(item.args)
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == "__init__"
            ),
            None,
        )
        fields = [ast.unparse(item) for item in node.body if isinstance(item, ast.AnnAssign)]
        bases = [ast.unparse(base) for base in (*node.bases, *node.keywords)]
        decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
        found.add(json.dumps([node.name, bases, decorators, fields, init]))
    return found


FACETS: dict[str, Callable[[str], AbstractSet[object]]] = {
    "deadline_owners": deadline_names,
    "lifecycle_sets": lifecycle_sets,
    "class_declarations": class_declarations,
}
"""The facets a ``selection_readers`` row may name, each a function of one file's text."""


def facet_changed(facet: str, before: str, after: str) -> bool:
    """Whether a file's facts for this facet differ between two texts (an absent side is '')."""
    return FACETS[facet](before) != FACETS[facet](after)
