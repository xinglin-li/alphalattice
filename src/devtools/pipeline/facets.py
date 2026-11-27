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


_HAND = frozenset({"OrderedDict", "WeakKeyDictionary", "WeakValueDictionary"})
"""Stateful maps evicted by hand: their bound is not a count the scan reads."""
_MAPS = frozenset({"BoundedMemo", "dict", "Dict", "Mapping", "MutableMapping", *_HAND})
_PLAIN = frozenset({"dict", "defaultdict"})
"""A module-level map spelled as an empty literal or one of these calls: a cache once a function
writes it, holding no bound the scan can read."""

# What a cache's source holds its size to: a count, an owner's capacity it does not spell as a
# number ("capacity"), a hand-written eviction the scan cannot read ("hand"), or None for none.
type CacheBound = int | str | None


def _count(node: ast.expr, constants: dict[str, CacheBound]) -> CacheBound:
    """The size an argument spells: a positive literal or module constant, or BoundedMemo's own
    default capacity. Anything else, a None, an imported name or a computed value, is no bound
    the scan can prove, so a row that names one over it is refused."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, int) and node.value > 0 else None
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "default_capacity"
        and _named(node.func.value) == "BoundedMemo"
    ):
        return "capacity"
    return None


def _named(node: ast.expr) -> str | None:
    """A call's or a reference's last name: ``a.b(c)`` is ``b``."""
    target = node.func if isinstance(node, ast.Call) else node
    name = getattr(target, "id", None) or getattr(target, "attr", None)
    return name if isinstance(name, str) else None


def _functools_bound(
    function: ast.FunctionDef | ast.AsyncFunctionDef, constants: dict[str, CacheBound]
) -> CacheBound | bool:
    """The bound a functools decorator declares, or False for a function that caches nothing."""
    for decorator in function.decorator_list:
        if _named(decorator) == "cache":
            arguments = function.args
            keyed = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
            return 1 if not keyed and not arguments.vararg and not arguments.kwarg else None
        if _named(decorator) == "lru_cache":
            if not isinstance(decorator, ast.Call):
                return 128
            given = [*decorator.args, *(k.value for k in decorator.keywords if k.arg == "maxsize")]
            if given:
                return None if isinstance(given[0], ast.Starred) else _count(given[0], constants)
            # The default holds only for a call that gives no maximum: an unpacked one is unread.
            return None if any(k.arg is None for k in decorator.keywords) else 128
    return False


def _stored_types(node: ast.expr) -> set[str]:
    """Read the value's type and union members without descending into type arguments."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _stored_types(node.left) | _stored_types(node.right)
    if isinstance(node, ast.Subscript):
        if _named(node.value) in {"Optional", "Union"}:
            members = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            return set().union(*(_stored_types(member) for member in members))
        node = node.value
    return {name} if (name := _named(node)) else set()


def _context_held(
    names: list[str], annotation: ast.expr, aliases: dict[str, ast.expr]
) -> dict[str, CacheBound]:
    """A context variable that holds a mapping, with the bound its mapping's type declares."""
    if not isinstance(annotation, ast.Subscript):
        return {}
    spelled = _stored_types(annotation.slice)
    for alias in spelled & aliases.keys():
        spelled |= _stored_types(aliases[alias])
    if not spelled & _MAPS:
        return {}
    bound: CacheBound = (
        "capacity" if "BoundedMemo" in spelled else "hand" if spelled & _HAND else None
    )
    return dict.fromkeys(names, bound)


def _written_maps(tree: ast.Module) -> set[str]:
    """The module names a function writes into by key: an item assigned or a map method that
    stores one (``setdefault``, ``update``)."""
    written: set[str] = set()
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(function):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AugAssign | ast.AnnAssign)
                else []
            )
            written.update(
                t.value.id
                for t in targets
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
            )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"setdefault", "update"}
                and isinstance(node.func.value, ast.Name)
            ):
                written.add(node.func.value.id)
    return written


def _sized(node: ast.AST) -> str | None:
    """The map a ``len(name)`` call measures."""
    if (
        isinstance(node, ast.Call)
        and _named(node) == "len"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
    ):
        return node.args[0].id
    return None


def _over_capacity(node: ast.If | ast.While) -> tuple[str, list[ast.stmt]] | None:
    """The map a single size comparison tests and the statements that run once it is at or
    over its limit: the body for ``len(m) >= n`` (``>``, ``n <= len(m)``, ``n < len(m)``), an
    if's else for ``len(m) < n`` (``<=``, ``n > len(m)``, ``n >= len(m)``). Any other test
    names none, so it earns nothing."""
    test = node.test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return None
    left, right, op = test.left, test.comparators[0], test.ops[0]
    name, growing = _sized(left), isinstance(op, ast.Gt | ast.GtE)
    if name is None:
        name, growing = _sized(right), isinstance(op, ast.Lt | ast.LtE)
    if name is None or not isinstance(op, ast.Gt | ast.GtE | ast.Lt | ast.LtE):
        return None
    if growing:
        return name, node.body
    return (name, node.orelse) if isinstance(node, ast.If) else None


def _measured_maps(tree: ast.Module) -> set[str]:
    """The names evicted by hand: a branch or loop whose single ``len(name)`` comparison's
    at-or-over-limit side evicts from that map (``pop``, ``popitem``, ``clear`` or
    ``del name[key]``). A size compared only to log or warn, or an eviction on the side below
    the limit, earns nothing."""
    held: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If | ast.While):
            continue
        over = _over_capacity(node)
        if over is None:
            continue
        measured, side = {over[0]}, over[1]
        evicted = set()
        for inner in (c for statement in side for c in ast.walk(statement)):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr in {"pop", "popitem", "clear"}
                and isinstance(inner.func.value, ast.Name)
            ):
                evicted.add(inner.func.value.id)
            elif isinstance(inner, ast.Delete):
                evicted.update(
                    t.value.id
                    for t in inner.targets
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                )
        held |= measured & evicted
    return held


def cache_holdings(source: str) -> set[tuple[str, CacheBound, str]]:
    """Each cache a module holds, by the name the cache register's ``defined_by`` gives it after
    its path, with the bound its source declares and the kind that declares it: functools caches
    by their dotted scope (kind ``functools``), and module-level stateful maps (their class's name)
    and the context variables that hold one (``ContextVar``) by their name, and a plain map a
    function writes by key (``dict``, no bound the scan can read). The cache census
    builds its answer from this, so it selects exactly the edits that change the answer: a bound
    named through a module constant or a type alias, or a map's kind, included."""
    tree = ast.parse(source)
    constants = {
        target.id: _count(node.value, {})
        for node in tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign) and isinstance(node.value, ast.Constant)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    held: dict[str, tuple[CacheBound, str]] = {}
    scope: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            bound = _functools_bound(node, constants)
            if bound is not False:
                held[".".join([*scope, node.name])] = (bound, "functools")
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            scope.append(node.name)
            for child in ast.iter_child_nodes(node):
                visit(child)
            scope.pop()
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    aliases: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
            aliases[node.name.id] = node.value
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            aliases[node.targets[0].id] = node.value
    written, measured = _written_maps(tree), _measured_maps(tree)
    for node in tree.body:
        value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if (isinstance(value, ast.Dict) and not value.keys) or (
            isinstance(value, ast.Call)
            and _named(value) in _PLAIN
            and (_named(value) == "defaultdict" or not (value.args or value.keywords))
        ):
            held.update(
                {name: ("hand" if name in measured else None, "dict") for name in names}
                if set(names) <= written
                else {}
            )
            continue
        if not isinstance(value, ast.Call):
            continue
        kind = _named(value)
        if kind == "BoundedMemo":
            capacity = [k.value for k in value.keywords if k.arg == "capacity"]
            # BoundedMemo always holds a capacity: one it cannot read is its default or the owner's.
            bound = (_count(capacity[0], constants) if capacity else None) or "capacity"
            held.update(dict.fromkeys(names, (bound, kind)))
        elif kind in _HAND:
            held.update(dict.fromkeys(names, ("hand", kind)))
        elif kind == "ContextVar" and isinstance(node, ast.AnnAssign):
            context = _context_held(names, node.annotation, aliases)
            held.update({name: (bound, kind) for name, bound in context.items()})
    return {(name, bound, kind) for name, (bound, kind) in held.items()}


FACETS: dict[str, Callable[[str], AbstractSet[object]]] = {
    "deadline_owners": deadline_names,
    "lifecycle_sets": lifecycle_sets,
    "class_declarations": class_declarations,
    "defined_tests": defined_tests,
    "cache_holdings": cache_holdings,
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
