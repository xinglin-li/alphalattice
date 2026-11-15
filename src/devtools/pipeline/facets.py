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
from collections.abc import Callable, Collection
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


def _read_field(item: ast.stmt) -> bool:
    """A class statement a constructor's census reads as a field: an annotated field, or a
    pydantic model's config, which the contract census reads."""
    return isinstance(item, ast.AnnAssign) or (
        isinstance(item, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "model_config" for t in item.targets)
    )


def class_declarations(source: str) -> set[str]:
    """Each top-level class as a constructor sees it: its name, bases, decorators, fields
    (``_read_field``), ``__init__`` signature and import bindings used by bases. The
    typed-exception census resolves exception classes across modules, including imported base
    aliases."""
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
        fields = [ast.unparse(item) for item in node.body if _read_field(item)]
        bases = [ast.unparse(base) for base in (*node.bases, *node.keywords)]
        decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
        found.add(json.dumps([node.name, bases, decorators, fields, init]))
    return found


def module_presence(source: str | None) -> set[str]:
    """Whether the file is there (``None`` is an absent side): its facts change only when it is
    added or deleted, an empty module included. A census over which modules exist reads no more
    of a file."""
    return set() if source is None else {"present"}


def defined_tests(source: str) -> set[str]:
    """The test functions a file defines, by name: they change when one is added, removed or
    renamed, which is all a census of named test holders reads of a test file."""
    return {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def mentions(source: str, words: Collection[str]) -> set[str]:
    """Each function, and each module or class statement outside one, whose text holds one of
    ``words``, by its syntax. A census that finds the calls or names ``words`` spell reads a
    subset of this, so a word added or removed, or any edit inside such a function, changes it."""
    if not any(word in source for word in words):
        return set()
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    bodies = [tree.body, *(n.body for n in nodes if isinstance(n, ast.ClassDef))]
    loose = {id(s) for body in bodies for s in body if not isinstance(s, ast.ClassDef)}
    found = set()
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or id(node) in loose:
            text = ast.get_source_segment(source, node) or ""
            if any(word in text for word in words):
                found.add(ast.dump(node, include_attributes=False))
    return found


FACETS: dict[str, Callable[[str], AbstractSet[object]]] = {
    "deadline_owners": deadline_names,
    "lifecycle_sets": lifecycle_sets,
    "class_declarations": class_declarations,
    "defined_tests": defined_tests,
}
"""The facets a ``selection_readers`` row may name, each a function of one file's text."""

PRESENCE_FACETS: dict[str, Callable[[str | None], AbstractSet[object]]] = {
    "module_presence": module_presence,
}
"""The facets a row may name that read an absent side as absent, never as an empty text."""

WORD_FACETS: dict[str, Callable[[str, Collection[str]], AbstractSet[object]]] = {
    "mentions": mentions,
}
"""The facets a row names with its ``words``, each a function of one file's text and them."""


def facet_changed(
    facet: str, before: str | None, after: str | None, words: Collection[str] = ()
) -> bool:
    """Whether a file's facts for this facet differ between two texts; ``None`` is an absent
    side, which every facet but a presence facet reads as ''."""
    if facet in PRESENCE_FACETS:
        return PRESENCE_FACETS[facet](before) != PRESENCE_FACETS[facet](after)
    before, after = before or "", after or ""
    if facet in WORD_FACETS:
        return WORD_FACETS[facet](before, words) != WORD_FACETS[facet](after, words)
    return FACETS[facet](before) != FACETS[facet](after)
