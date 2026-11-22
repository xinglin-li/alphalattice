"""Semantic source-content identity separated from physical repository layout."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any, cast

from alphalattice.kernel.identity import change_sha256, identity_sha256, integrity_sha256
from alphalattice.kernel.shared_kernel.project_layout import (
    resolve_playpen_root,
    resource_path,
    source_root,
    tracked_path,
)

from .identity import canonical_hash

_PHYSICAL_NAMESPACE_GROUPS = frozenset(
    {
        "capabilities",
        "control",
        "evidence",
        "foundation",
        "interface",
        "investment",
        "kernel",
        "oversight",
        "protocols",
    }
)


def source_component_id(*, package_id: str, source_path: Path) -> str:
    """Derive a stable component ID from its domain package-relative location."""
    if not _valid_component_id(package_id):
        raise ValueError("shared_kernel.source_component_id_invalid")
    try:
        package_index = max(
            index for index, part in enumerate(source_path.parts) if part == package_id
        )
    except ValueError as error:
        raise ValueError("shared_kernel.source_component_id_invalid") from error
    relative = source_path.parts[package_index + 1 :]
    if not relative:
        return package_id
    suffix_parts = list(relative)
    suffix_parts[-1] = Path(suffix_parts[-1]).stem
    if suffix_parts[-1] == "__init__":
        suffix_parts.pop()
    normalized = tuple(part.replace("-", "_") for part in suffix_parts)
    component_id = ".".join((package_id, *normalized))
    if not _valid_component_id(component_id):
        raise ValueError("shared_kernel.source_component_id_invalid")
    return component_id


def source_component_id_from_tracked_path(tracked_path: str) -> str:
    """Resolve a semantic component ID from old or namespaced tracked layout."""
    path = PurePosixPath(tracked_path.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("shared_kernel.source_component_id_invalid")
    parts = list(path.parts)
    if parts[0] == "src":
        parts.pop(0)
        if (
            len(parts) >= 3
            and parts[0] == "alphalattice"
            and parts[1] in _PHYSICAL_NAMESPACE_GROUPS
        ):
            parts = parts[2:]
    elif parts[0] == "agents":
        parts[0] = "agent_profiles"
    elif parts[0] == "skills":
        parts[0] = "agent_skills"
    elif len(parts) == 1 and parts[0] in {"pyproject.toml", "uv.lock"}:
        parts = ["build", "playpen", parts[0]]
    else:
        parts.insert(0, "playpen")
    parts[-1] = Path(parts[-1]).stem
    if parts[-1] == "__init__":
        parts.pop()
    component_id = ".".join(part.replace("-", "_") for part in parts)
    if not _valid_component_id(component_id):
        raise ValueError("shared_kernel.source_component_id_invalid")
    return component_id


_PARSED: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
"""A module's syntax digest and imports, by its bytes' SHA-256 and its package: the same bytes
are parsed once a process, and any edit is a new key, so the cache never answers stale."""


def source_syntax_sha256(path: Path) -> str:
    """Hash a module's executable AST without bare string statements.

    A module's identity as its syntax: the parsed tree without its docstrings and other bare
    string statements, and without its execution spans (``without_measurement``). A comment, a
    docstring, a span or the layout is not syntax, so editing one moves nothing; a changed name,
    value or statement does.

    Args:
        path: Python module file to parse.

    Returns:
        SHA-256 digest of the parsed syntax after removing bare string statements.
    """
    return _parsed(path, "")[0]


def source_module_imports(path: Path, *, package: str) -> tuple[str, ...]:
    """The modules one source file imports, as a rule closure's walk follows them.

    Args:
        path: The module's file.
        package: The package its relative imports resolve in.

    Returns:
        The imported modules' dotted names, outside `if TYPE_CHECKING:`.
    """
    return _parsed(path, package)[1]


def source_bytes_syntax_sha256(payload: bytes) -> str:
    """A module's syntax digest from its bytes, as ``source_syntax_sha256`` reads a file's.

    For a version no file holds, such as the parent commit's: the gate's neutral path (GN)
    compares a changed file with it.

    Args:
        payload: The module's source bytes.

    Returns:
        The digest ``source_syntax_sha256`` gives a file holding these bytes.
    """
    return _parsed_payload(payload, package="", filename="<bytes>")[0]


@dataclass(frozen=True, slots=True)
class NumberDecidingRule:
    """Select numerical packages while excluding their declared platform segments.

    Which modules can move a result: those of the number-deciding packages, except their
    platform (a module under a platform segment such as ``storage`` decides no number, B5).

    Attributes:
        packages: Owner package names whose modules may decide a numerical result.
        platform_segments: Package-relative segments excluded from the numerical walk.
    """

    packages: frozenset[str]
    platform_segments: frozenset[str]


def number_deciding_closure(
    entries: tuple[str, ...],
    *,
    root: Path,
    rule: NumberDecidingRule,
    excluded: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Walk and hash the admitted numerical imports of the declared entry modules.

    The entry modules and every module they import, transitively, that the ``rule`` counts
    as number-deciding: module name to its syntax digest. A package's ``__init__`` counts
    when an import passes through it; an import under ``if TYPE_CHECKING:`` is not followed,
    and neither is a module outside the packages or under a platform segment. An ``excluded``
    module is neither hashed nor walked through: the owner names what decides none of the
    numbers its identity covers (a component's own identity code, a sibling it never runs).

    Args:
        entries: Dotted entry-module names to visit.
        root: Repository root containing src.
        rule: Numerical-package and excluded-platform selection rule.
        excluded: Exact module names neither hashed nor traversed.

    Returns:
        Admitted module names mapped to their executable-syntax digests.
    """
    return dict(sorted(_walked(entries, root, rule, excluded, _parsed).items()))


def _walked[T](
    entries: tuple[str, ...],
    root: Path,
    rule: NumberDecidingRule,
    excluded: frozenset[str],
    read: Callable[[Path, str], tuple[T, tuple[str, ...]]],
) -> dict[str, T]:
    """The closure's walk (``number_deciding_closure``), each module read by ``read``."""
    source = source_root(root)

    def deciding(module: str) -> bool:
        parts = module.split(".")
        return (
            len(parts) >= 3
            and parts[0] == "alphalattice"
            and parts[1] in _PHYSICAL_NAMESPACE_GROUPS
            and parts[2] in rule.packages
            and not rule.platform_segments.intersection(parts[3:])
        )

    found: dict[str, T] = {}
    visited = set(excluded)
    pending = list(entries)
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = _module_file(source, module)
        if path is None:
            if module in entries:
                raise ValueError("shared_kernel.source_closure_entry_missing")
            continue
        found[module], imported = read(path, _package(module, path))
        for name in imported:
            parts = name.split(".")
            for end in range(3, len(parts) + 1):  # each package an import passes through
                prefix = ".".join(parts[:end])
                if prefix not in visited and deciding(prefix):
                    pending.append(prefix)
    return found


def _module_file(source: Path, module: str) -> Path | None:
    base = source.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _package(module: str, path: Path) -> str:
    """The package a module's relative imports resolve in."""
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def _parsed(path: Path, package: str) -> tuple[str, tuple[str, ...]]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError("shared_kernel.source_syntax_unreadable") from error
    return _parsed_payload(payload, package=package, filename=path.name)


def _parsed_payload(payload: bytes, *, package: str, filename: str) -> tuple[str, tuple[str, ...]]:
    key = _bytes_key(payload, package)
    kept = _PARSED.get(key)
    if kept is None:
        try:
            tree = ast.parse(payload, filename=filename)
        except (SyntaxError, ValueError) as error:
            raise ValueError("shared_kernel.source_syntax_unreadable") from error
        imported = _imports(tree, package=package)
        kept = _PARSED[key] = (_syntax_digest(tree), imported)
    return kept


SPAN_MODULE = "alphalattice.kernel.shared_kernel.spans"
"""The execution-span owner: a span measures where a stage's time goes and decides nothing."""
_SPAN_NAMES = frozenset({"span", "spanned"})


def without_measurement(tree: ast.Module) -> ast.Module:
    """The module's tree with its execution spans taken out, as an identity reads it (LAWS ID3).

    In a module that imports ``span`` or ``spanned`` from the span owner under those names and
    binds them nowhere else, each ``with span("<text>")`` (literal arguments, no ``as``) reads as
    its body, each ``@spanned("<text>")`` and that import are left out: a span measures and never
    decides (``shared_kernel.spans``), so adding or moving one moves no identity
    and no reuse pin. Any other form, or a module that rebinds either name, reads as written.
    The tree is changed in place and returned.

    Args:
        tree: A parsed module.

    Returns:
        The same tree without its spans.
    """
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == SPAN_MODULE
        for alias in node.names
        if alias.name in _SPAN_NAMES and alias.asname is None
    }
    if not imported or imported & _rebound(tree):
        return tree
    return cast(ast.Module, _WithoutMeasurement(frozenset(imported)).visit(tree))


def _rebound(tree: ast.Module) -> set[str]:
    """Names the module binds other than by importing them from the span owner."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if _plainly_bound(node, bound):
            continue
        if isinstance(node, ast.Import):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not (
            node.level == 0 and node.module == SPAN_MODULE
        ):
            bound.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            bound.update(alias.asname for alias in node.names if alias.asname)
    return bound


def _plainly_bound(node: ast.AST, bound: set[str]) -> bool:
    """Add the name an assignment, a definition or a parameter binds; say whether it was one."""
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
        bound.add(node.id)
    elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        bound.add(node.name)
    elif isinstance(node, ast.arg):
        bound.add(node.arg)
    else:
        return False
    return True


def _literal_call(node: ast.expr, names: frozenset[str], name: str) -> bool:
    """A call of the imported ``name`` whose arguments are all literal text or None."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        and name in names
        and all(_literal(argument) for argument in node.args)
        and all(keyword.arg == "detail" and _literal(keyword.value) for keyword in node.keywords)
    )


def _literal(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, str))


class _WithoutMeasurement(ast.NodeTransformer):
    """Read spans as their bodies and leave out their import (``without_measurement``)."""

    def __init__(self, names: frozenset[str]) -> None:
        self.names = names

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        if node.level != 0 or node.module != SPAN_MODULE:
            return node
        node.names = [
            alias for alias in node.names if not (alias.name in self.names and alias.asname is None)
        ]
        return node if node.names else None

    def visit_With(self, node: ast.With) -> ast.With | list[ast.stmt]:
        self.generic_visit(node)
        kept = [
            item
            for item in node.items
            if item.optional_vars is not None
            or not _literal_call(item.context_expr, self.names, "span")
        ]
        if len(kept) == len(node.items):
            return node
        if kept:
            node.items = kept
            return node
        return node.body

    def _undecorated(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        decorated = cast(ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, node)
        decorated.decorator_list = [
            decorator
            for decorator in decorated.decorator_list
            if not _literal_call(decorator, self.names, "spanned")
        ]
        return node

    visit_FunctionDef = _undecorated
    visit_AsyncFunctionDef = _undecorated
    visit_ClassDef = _undecorated


def _syntax_digest(tree: ast.Module) -> str:
    return _text_digest(_dumped(tree))


def _text_digest(text: str) -> str:
    """The SHA-256 of a syntax text: a module's dumped tree, or a canonical module's rendering."""
    return identity_sha256(text.encode("utf-8")).hexdigest()


def _bytes_key(payload: bytes, package: str) -> tuple[str, str]:
    """A parse cache's key: the module's bytes' SHA-256 and the package it resolves in."""
    return change_sha256(payload).hexdigest(), package


def _dumped(tree: ast.Module) -> str:
    """The tree as syntax: without its spans and bare strings, and without positions."""
    without_measurement(tree)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
            kept = [statement for statement in body if not _bare_string(statement)]
            node.body = kept or [ast.Pass()]  # type: ignore[attr-defined]
    return ast.dump(tree, include_attributes=False)


def _bare_string(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _imports(tree: ast.Module, *, package: str) -> tuple[str, ...]:
    typing_only = {
        id(inner)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and _type_checking(node.test)
        for child in node.body
        for inner in ast.walk(child)
    }
    names: list[str] = []
    for node in ast.walk(tree):
        if id(node) in typing_only:
            continue
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            stem = _import_stem(node, package)
            names.append(stem)
            names.extend(f"{stem}.{alias.name}" for alias in node.names if alias.name != "*")
    return tuple(names)


def _import_stem(node: ast.ImportFrom, package: str) -> str:
    """The module a `from ... import` names, its relative form resolved in ``package``."""
    if not node.level:
        return node.module or ""
    anchor = package.split(".")
    base = ".".join(anchor[: len(anchor) - node.level + 1])
    return f"{base}.{node.module}" if node.module else base


def _type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


CLOSURE_FORM = 2
"""The rule closure's form. Form 2 hashes the code that runs (``canonical_closure``). Form 1
hashed each module's syntax with its imports as written; the successor pass that moved every
role to form 2 keeps its values readable."""

_Reference = tuple[str, ...]
"""An imported binding as written: ``("module", dotted)`` or ``("from", module, name)``."""


@dataclass(frozen=True, slots=True)
class _Canonical:
    """One module's code with each imported name a numbered placeholder (``canonical_closure``).

    Attributes:
        key: The module's bytes' SHA-256 and its package.
        walked: The modules its imports name, as the closure's walk follows them.
        template: The dumped tree split at its placeholders: text, then each placeholder's
            number and the text after it.
        references: The binding each placeholder names.
        imports: Every binding the module imports, wherever its import statement is.
    """

    key: tuple[str, str]
    walked: tuple[str, ...]
    template: tuple[str, ...]
    references: tuple[_Reference, ...]
    imports: tuple[_Reference, ...]


_CANONICAL: dict[tuple[str, str], _Canonical] = {}
"""A module's canonical template by its bytes' SHA-256 and its package, as ``_PARSED``."""

_DIGESTS: dict[tuple[tuple[str, str], tuple[str, ...], tuple[str, ...]], str] = {}
"""A module's digest outside any ring, by its template's key and what its names resolve to: the
digest is a function of these alone, so a process computes each once."""


def canonical_closure(
    entries: tuple[str, ...], *, root: Path, rule: NumberDecidingRule
) -> dict[str, str]:
    """Hash each module of a rule closure as the code that runs, not as it is written.

    The modules are ``number_deciding_closure``'s. Each is read as its syntax without
    comments, docstrings, spans or layout, and with its imports resolved:

    - an import in a leading import block (``_Scan.covered``), whose name is bound by that one
      import alone and read only where the block binds it, is replaced at each read by what it
      names, and the statement is left out; its alias and which leading block holds it, the
      module's or the one function that reads it, stop mattering;
    - every other import keeps its place and the names it binds, its target named as below;
    - what an import names is the module it names, or the module and the name it takes from
      it; a ``from`` import of a closure submodule names that submodule. No re-export is
      resolved: reaching the same code through another module rotates, the safe side;
    - a closure module is named by its digest, never its path, so code moved to a new path
      with the same body keeps its digest; anything outside the closure is named by its
      dotted path.

    A module's digest covers its code, the set of what it imports, and the digest of every
    closure module it names (modules importing each other are hashed as one ring), so a
    changed constant, operator or called body anywhere in the closure moves every digest
    that reaches it.

    Args:
        entries: Dotted entry-module names.
        root: Repository root containing src.
        rule: Number-deciding selection rule.

    Returns:
        Each closure module's digest, by module name.
    """
    parsed: dict[str, _Canonical | None] = dict(
        _walked(entries, root, rule, frozenset(), _canonical_walk)
    )
    found = sorted(parsed)
    members = set(found)
    resolved: dict[_Reference, tuple[str, str]] = {}  # a name is used many times, resolved once

    def target(reference: _Reference) -> tuple[str, str]:
        if reference not in resolved:
            if reference[0] == "module":
                resolved[reference] = (reference[1], "")
            else:
                _, module, name = reference
                inner = f"{module}.{name}"
                resolved[reference] = (inner, "") if inner in members else (module, name)
        return resolved[reference]

    targets = {
        module: [target(reference) for reference in cast(_Canonical, parsed[module]).references]
        for module in found
    }
    imported = {
        module: [target(reference) for reference in cast(_Canonical, parsed[module]).imports]
        for module in found
    }
    deep: dict[str, str] = {}

    def marker(target: tuple[str, str], ring: Mapping[str, str]) -> str:
        module, name = target
        if module in deep:
            head = deep[module]
        elif module in ring:
            head = f"ring:{ring[module]}"
        else:
            return "@ext:" + (f"{module}.{name}" if name else module)
        return f"@{head}:{name}" if name else f"@{head}"

    def rendered(module: str, ring: Mapping[str, str]) -> str:
        held = cast(_Canonical, parsed[module])
        names = tuple(marker(target, ring) for target in targets[module])
        imports = tuple(sorted({marker(target, ring) for target in imported[module]}))
        key = (held.key, names, imports)
        if not ring and key in _DIGESTS:
            return _DIGESTS[key]
        digest = _text_digest(
            "".join(
                f"'{names[int(text)]}'" if index % 2 else text
                for index, text in enumerate(held.template)
            )
            + "\0".join(("", *imports))
        )
        if not ring:
            _DIGESTS[key] = digest
        return digest

    edges = {
        module: tuple(
            {target for target, _ in targets[module] + imported[module] if target in members}
        )
        for module in found
    }
    for component in _components(edges):
        (only,) = component if len(component) == 1 else ("",)
        if only and only not in edges[only]:  # no ring: its digest is its own
            deep[only] = rendered(only, {})
            continue
        shallow = {module: rendered(module, dict.fromkeys(component, "")) for module in component}
        ring = sorted(shallow.values())
        digests = {
            module: _text_digest("\0".join((rendered(module, shallow), *ring)))
            for module in component
        }
        deep.update(digests)
    return {module: deep[module] for module in found}


def _components(edges: Mapping[str, tuple[str, ...]]) -> list[list[str]]:
    """Tarjan's strongly connected components, each listed after every component it reaches."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def enter(node: str) -> None:
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)

    for start in sorted(edges):
        if start in index:
            continue
        enter(start)
        work = [(start, iter(sorted(edges[start])))]
        while work:
            node, children = work[-1]
            child = next(children, None)
            if child is not None:
                if child not in index:
                    enter(child)
                    work.append((child, iter(sorted(edges[child]))))
                elif child in on_stack:
                    low[node] = min(low[node], index[child])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[str] = []
                while not component or component[-1] != node:
                    component.append(stack.pop())
                    on_stack.discard(component[-1])
                components.append(sorted(component))
    return components


def _canonical_walk(path: Path, package: str) -> tuple[_Canonical, tuple[str, ...]]:
    held = _canonical(path, package)
    return held, held.walked


def _canonical(path: Path, package: str) -> _Canonical:
    key, payload = _keyed(path, package)
    kept = _CANONICAL.get(key)
    if kept is None:
        kept = _CANONICAL[key] = _canonical_tree(_tree(payload, path), key)
    return kept


def _keyed(path: Path, package: str) -> tuple[tuple[str, str], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError("shared_kernel.source_syntax_unreadable") from error
    return _bytes_key(payload, package), payload


def _tree(payload: bytes, path: Path) -> ast.Module:
    try:
        return ast.parse(payload, filename=path.name)
    except (SyntaxError, ValueError) as error:
        raise ValueError("shared_kernel.source_syntax_unreadable") from error


def _canonical_tree(tree: ast.Module, key: tuple[str, str]) -> _Canonical:
    scan = _Scan(tree, key[1])
    if scan.spans:  # a span measures and decides nothing (``without_measurement``)
        without_measurement(tree)
        scan = _Scan(tree, key[1], walked=scan.walked)
    single = {
        name: next(iter(references))
        for name, references in scan.named.items()
        if len(references) == 1
        and name not in scan.other
        and name not in scan.plain
        and scan.covered(name)
    }
    rooted = {
        name: frozenset(modules)
        for name, modules in scan.plain.items()
        if name not in scan.other and name not in scan.named and scan.covered(name)
    }
    template = _Template(single, rooted, key[1])
    return _Canonical(
        key=key,
        walked=scan.walked,
        template=template.written(tree),
        references=tuple(template.references),
        imports=tuple(scan.imports),
    )


_BINDING_KINDS = frozenset(
    {
        ast.Import,
        ast.ImportFrom,
        ast.If,
        ast.Name,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
        ast.arg,
        ast.ExceptHandler,
        ast.MatchAs,
        ast.MatchStar,
        ast.MatchMapping,
        ast.Global,
        ast.Nonlocal,
    }
)
"""The node kinds a scan reads: an import, a TYPE_CHECKING block, a scope, or a binding."""

_Scopes = tuple[int, ...]
"""Where a name is bound or read: its enclosing function and class scopes, outermost first, by
node id."""


class _Scan:
    """One pass over a module: what it imports and where, and every other name it binds.

    Attributes:
        walked: The modules its imports outside ``if TYPE_CHECKING:`` name, as ``_imports``
            gives them to the closure's walk.
        named: Each name an import binds (``as`` or ``from``), with what each import names.
        plain: Each name a plain ``import a.b`` binds, with the dotted modules it names.
        imports: Every binding the module imports, wherever its statement is.
        other: The names it binds otherwise, in any scope.
        spans: Whether it imports a span from the span owner.
        bound: Where each import binds each name it binds, and whether that import stands in
            a leading import block.
        used: Where each name is read.
        functions: The function scopes, by node id.
        leading: The imports in a leading import block, by node id: the imports that open the
            module's body or a function's, before its first statement that is not an import
            (prose aside, since it runs nothing).
    """

    def __init__(self, tree: ast.Module, package: str, walked: tuple[str, ...] | None = None):
        self.named: dict[str, set[_Reference]] = {}
        self.plain: dict[str, set[str]] = {}
        self.imports: list[_Reference] = []
        self.other: set[str] = set()
        self.spans = False
        self.bound: dict[str, list[tuple[_Scopes, bool]]] = {}
        self.used: dict[str, list[_Scopes]] = {}
        self.functions: set[int] = set()
        self.leading: set[int] = set(_leading(tree.body))
        found: list[str] = []
        # Each node: whether only TYPE_CHECKING reaches it, and its scopes.
        pending: list[tuple[ast.AST, bool, _Scopes]] = [
            (statement, False, ()) for statement in tree.body
        ]
        while pending:
            node, typing_only, scopes = pending.pop()
            kind = type(node)
            if kind in _BINDING_KINDS:
                if kind is ast.Import or kind is ast.ImportFrom:
                    seen = (scopes, id(node) in self.leading)
                    if kind is ast.Import:
                        self._import(cast(ast.Import, node), found, typing_only, seen)
                    else:
                        statement = cast(ast.ImportFrom, node)
                        self._import_from(statement, package, found, typing_only, seen)
                    continue
                if kind is ast.Name:
                    name = cast(ast.Name, node)
                    if type(name.ctx) is ast.Load:
                        self.used.setdefault(name.id, []).append(scopes)
                    else:
                        self.other.add(name.id)
                    continue
                if kind is ast.If and _type_checking(cast(ast.If, node).test):
                    test = cast(ast.If, node)
                    pending.append((test.test, typing_only, scopes))
                    pending.extend((c, True, scopes) for c in test.body)
                    pending.extend((c, typing_only, scopes) for c in test.orelse)
                    continue
                _bound_otherwise(node, self.other)
                if kind is ast.FunctionDef or kind is ast.AsyncFunctionDef or kind is ast.Lambda:
                    inner = (*scopes, id(node))
                    if kind is not ast.Lambda:
                        self.functions.add(id(node))
                        self.leading.update(_leading(cast(ast.FunctionDef, node).body))
                    body = cast(ast.FunctionDef | ast.Lambda, node).body
                    for child in ast.iter_child_nodes(node):
                        inside = child is body or (isinstance(body, list) and child in body)
                        # decorators, defaults, annotations: they run where it is defined
                        pending.append((child, typing_only, inner if inside else scopes))
                    continue
                if kind is ast.ClassDef:
                    inner = (*scopes, id(node))
                    body = cast(ast.ClassDef, node).body
                    for child in ast.iter_child_nodes(node):
                        place = inner if child in body else scopes
                        pending.append((child, typing_only, place))
                    continue
            pending.extend((child, typing_only, scopes) for child in ast.iter_child_nodes(node))
        self.walked = tuple(found) if walked is None else walked

    def covered(self, name: str) -> bool:
        """Whether ``name`` is read only where an import in a leading import block binds it.

        Then the import is read at each use and left out. A leading import block holds imports
        alone, so the module's binds before any read in the module runs, and a function's binds
        before any read in that function, or in one nested there, runs. An import nothing reads
        runs for its effect, and any other import (after a guard, an annotation, a call or any
        statement, or under a branch, a ``try`` or a class) is not proved to bind first: those
        read as written, at their place, and a move between them rotates.
        """
        uses = self.used.get(name, [])
        bindings = self.bound.get(name, [])
        if not uses or not all(leading for _, leading in bindings):
            return False
        if any(not where for where, _ in bindings):
            return True
        return all(any(scopes[: len(where)] == where for where, _ in bindings) for scopes in uses)

    def _import(
        self, node: ast.Import, found: list[str], typing_only: bool, seen: tuple[_Scopes, bool]
    ) -> None:
        for alias in node.names:
            if not typing_only:
                found.append(alias.name)
            self.imports.append(("module", alias.name))
            name = alias.asname or alias.name.partition(".")[0]
            self.bound.setdefault(name, []).append(seen)
            if alias.asname:
                self.named.setdefault(alias.asname, set()).add(("module", alias.name))
            else:
                self.plain.setdefault(name, set()).add(alias.name)

    def _import_from(
        self,
        node: ast.ImportFrom,
        package: str,
        found: list[str],
        typing_only: bool,
        seen: tuple[_Scopes, bool],
    ) -> None:
        stem = _import_stem(node, package)
        if not typing_only:
            found.append(stem)
            found.extend(f"{stem}.{alias.name}" for alias in node.names if alias.name != "*")
        if node.level == 0 and node.module == SPAN_MODULE:
            self.spans |= any(alias.name in _SPAN_NAMES for alias in node.names)
        for alias in node.names:
            if alias.name != "*":
                self.imports.append(("from", stem, alias.name))
                self.bound.setdefault(alias.asname or alias.name, []).append(seen)
                self.named.setdefault(alias.asname or alias.name, set()).add(
                    ("from", stem, alias.name)
                )


def _leading(body: list[ast.stmt]) -> list[int]:
    """The imports that open a body, before its first statement that is not an import or prose."""
    found: list[int] = []
    for statement in body:
        if isinstance(statement, ast.Import | ast.ImportFrom):
            found.append(id(statement))
        elif not _bare_string(statement):
            break
    return found


def _bound_otherwise(node: ast.AST, bound: set[str]) -> bool:
    """Add a name the node binds other than by an import, in any scope; say whether it binds."""
    if _plainly_bound(node, bound):
        return True
    if isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar) and node.name:
        bound.add(node.name)
    elif isinstance(node, ast.MatchMapping) and node.rest:
        bound.add(node.rest)
    elif isinstance(node, ast.Global | ast.Nonlocal):
        bound.update(node.names)
    else:
        return False
    return True


def _top_level_names(tree: ast.Module) -> frozenset[str]:
    """Names the module's top level binds other than by an import."""
    names: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(statement.name)  # what it holds binds in its own scope
            continue
        if isinstance(statement, ast.Import | ast.ImportFrom):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
    return frozenset(names)


class _Template:
    """Write a module's tree as text in one pass, each resolvable imported name a placeholder.

    A name read at each use is a placeholder there and its import is left out. Any other import
    keeps its place and the names it binds, its target a placeholder too, so its path's spelling
    is not read. Every bare string is left out, and an emptied block reads ``Pass()``. The text
    is split at the placeholders: text, then each placeholder's number and the text after.
    """

    def __init__(
        self,
        single: Mapping[str, _Reference],
        rooted: Mapping[str, frozenset[str]],
        package: str,
    ):
        self.single = single
        self.rooted = rooted
        self.package = package
        self.parts: list[str] = []
        self.text: list[str] = []
        self.references: list[_Reference] = []

    def written(self, tree: ast.Module) -> tuple[str, ...]:
        self.node(tree)
        return (*self.parts, "".join(self.text))

    def placeholder(self, reference: _Reference) -> None:
        self.parts += ("".join(self.text), str(len(self.references)))
        self.text = []
        self.references.append(reference)

    def node(self, node: ast.AST) -> None:
        kind = type(node)
        if kind is ast.Name:
            name = cast(ast.Name, node).id
            if name in self.single:
                return self.placeholder(self.single[name])
            if name in self.rooted:
                return self.placeholder(("module", name))
            self.text.append(f"Name({name!r},{type(cast(ast.Name, node).ctx).__name__})")
            return None
        if kind is ast.Attribute:
            dotted = _dotted(cast(ast.Attribute, node))
            if dotted is not None and dotted in self.rooted.get(dotted.partition(".")[0], ()):
                return self.placeholder(("module", dotted))
        if kind is ast.Import or kind is ast.ImportFrom:
            return self.imported(cast(ast.Import | ast.ImportFrom, node))
        self.text.append(kind.__name__ + "(")
        for field in kind._fields:
            value = getattr(node, field, None)
            if type(value) is list:
                self.listed(value)
            elif isinstance(value, ast.AST):
                self.node(value)
            else:
                self.text.append(repr(value))
            self.text.append(",")
        self.text.append(")")
        return None

    def listed(self, values: list[Any]) -> None:
        statements = bool(values) and isinstance(values[0], ast.stmt)
        self.text.append("[")
        written = 0
        for item in values:
            if statements and not self.kept(item):
                continue
            if isinstance(item, ast.AST):
                self.node(item)
            else:
                self.text.append(repr(item))
            self.text.append(",")
            written += 1
        if statements and not written:
            self.text.append("Pass(),")
        self.text.append("]")

    def imported(self, statement: ast.Import | ast.ImportFrom) -> None:
        """An import that stays: each name it binds, and its target as a placeholder."""
        stem = (
            _import_stem(statement, self.package) if isinstance(statement, ast.ImportFrom) else ""
        )
        self.text.append("Imports(")
        for alias in statement.names:
            if alias.name == "*":
                self.text.append(f"Star({stem!r}),")
                continue
            bound = alias.asname or (alias.name if stem else alias.name.partition(".")[0])
            self.text.append(f"({bound!r},")
            self.placeholder(("from", stem, alias.name) if stem else ("module", alias.name))
            self.text.append("),")
        self.text.append(")")

    def kept(self, statement: ast.stmt) -> bool:
        """Whether a statement is written: not a bare string, nor an import wholly resolved."""
        if isinstance(statement, ast.Import):
            statement.names = [
                alias
                for alias in statement.names
                if not (
                    alias.asname in self.single
                    if alias.asname
                    else alias.name.partition(".")[0] in self.rooted
                )
            ]
            return bool(statement.names)
        if isinstance(statement, ast.ImportFrom):
            statement.names = [
                alias
                for alias in statement.names
                if (alias.asname or alias.name) not in self.single
            ]
            return bool(statement.names)
        return not _bare_string(statement)


def _dotted(node: ast.expr) -> str | None:
    """``a.b.c`` for a chain of attributes on a name, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


IDENTITY_ROLES_PATH = Path("config") / "identity-roles.json"
"""The identity-roles table: each Host role, and which code can move a number."""


@cache
def number_deciding_rule(root: Path) -> NumberDecidingRule:
    """Which code can move a result, as the identity-roles table names it."""
    try:
        document = json.loads((root / IDENTITY_ROLES_PATH).read_text(encoding="utf-8"))
        rule = NumberDecidingRule(
            packages=frozenset(str(name) for name in document["number_deciding_packages"]),
            platform_segments=frozenset(str(name) for name in document["platform_segments"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("shared_kernel.identity_roles_unreadable") from error
    if not rule.packages:
        raise ValueError("shared_kernel.identity_roles_unreadable")
    return rule


def source_rule_closure_hash(
    *,
    root: Path,
    tracked_paths: tuple[str, ...],
    semantic_owner: str,
    numerical_role: str,
) -> str:
    """Hash one role's rule-derived module closure and tracked non-module bytes.

    A role's closure by rule: each tracked module is an entry, and the
    closure is what the entries import inside the number-deciding packages, each module
    hashed as the code that runs (``canonical_closure``) and named by its digest alone; a
    tracked file that is not a module under ``src`` (a lock, a Skill, a profile) is hashed by
    its bytes.

    Args:
        root: Repository root used to resolve tracked files and numerical imports.
        tracked_paths: Role-owned tracked paths that supply Python entries or exact byte inputs.
        semantic_owner: Stable owner identifier included in the closure binding.
        numerical_role: Stable numerical role identifier included in the closure binding.

    Returns:
        Canonical identity of the role, owner, numerical module closure and other tracked files.
    """
    python = {path: path.startswith("src/") and path.endswith(".py") for path in tracked_paths}
    modules = tuple(
        ".".join(PurePosixPath(path).with_suffix("").parts[1:])
        for path, module in python.items()
        if module
    )
    others = tuple(path for path, module in python.items() if not module)
    return cast(
        str,
        canonical_hash(
            {
                "semantic_owner": semantic_owner,
                "numerical_role": numerical_role,
                "form": CLOSURE_FORM,
                "modules": sorted(
                    canonical_closure(modules, root=root, rule=number_deciding_rule(root)).values()
                ),
                "files": {
                    source_component_id_from_tracked_path(path): _resource_sha256(root, path)
                    for path in others
                },
            }
        ),
    )


def _resource_sha256(root: Path, relative: str) -> str:
    path = resource_path(root, relative)
    if path.is_file():
        return integrity_sha256(path.read_bytes()).hexdigest()
    # Development runners are not runtime entries. The wheel carries the
    # accepted tree's byte commitments, preserving the same closure value.
    manifest = root / "config/release/source-hashes.json"
    if manifest.is_file():
        hashes = json.loads(manifest.read_text(encoding="utf-8"))
        value = hashes.get(relative)
        if isinstance(value, str) and len(value) == 64:
            return value
    raise ValueError(f"shared_kernel.source_resource_missing:{relative}")


IDENTITY_SWITCH_PATH = Path("config") / "identity-switch.json"
"""Each byte closure the rule replaced, with its rule value and the byte value it had then."""


@cache
def identity_switch(root: Path) -> Mapping[str, Mapping[str, str]]:
    """The switch table: for each switched closure, its rule and byte values at the switch.

    Args:
        root: The checkout whose ``config`` holds the table.

    Returns:
        The recorded values by closure key.

    Raises:
        ValueError: If the table cannot be read.
    """
    try:
        document = json.loads((root / IDENTITY_SWITCH_PATH).read_text(encoding="utf-8"))
        return {
            str(key): {"rule": str(value["rule"]), "byte": str(value["byte"])}
            for key, value in document["components"].items()
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError("shared_kernel.identity_switch_unreadable") from error


def switched_identity(key: str, rule_value: str, *, root: Path) -> str:
    """A closure's identity once its byte closure moved onto the rule (LAWS.md ID3).

    While the closure's rule value is the one the switch recorded, it keeps the byte value it
    had then, so everything sealed before the switch stays current; a comment or docstring no
    longer moves it, and a changed rule is a new identity.

    Args:
        key: The closure's key in the switch table.
        rule_value: Its rule closure's value now.
        root: The checkout whose ``config`` holds the table.

    Returns:
        The identity.
    """
    anchor = identity_switch(root).get(key)
    return anchor["byte"] if anchor is not None and anchor["rule"] == rule_value else rule_value


def source_switch_key(
    sources: Mapping[str, Path], *, semantic_owner: str, numerical_role: str
) -> str:
    """The switch table's key for a listed closure: its owner, role and entry names.

    Args:
        sources: The closure's entries by component id.
        semantic_owner: Who owns the closure.
        numerical_role: What kind of number it identifies.

    Returns:
        The key; a changed entry list is a new key.
    """
    entries = identity_sha256("\n".join(sorted(sources)).encode("utf-8")).hexdigest()[:16]
    return f"{semantic_owner}:{numerical_role}:{entries}"


def checkout_root(path: Path) -> Path:
    """The checkout a product source file sits in.

    The checkout is the directory holding ``src`` and the identity-roles table.

    Args:
        path: A file inside the checkout.

    Returns:
        The checkout's root.

    Raises:
        ValueError: If the file is in no checkout.
    """
    try:
        return resolve_playpen_root(path)
    except RuntimeError as error:
        raise ValueError("shared_kernel.source_checkout_unresolved") from error


def switched_source_identity(
    sources: Mapping[str, Path], *, semantic_owner: str, numerical_role: str
) -> str:
    """A listed byte closure moved onto the rule and kept at its switch value (ID3, W9b's sites).

    The listed files are the rule closure's entries (``source_rule_closure_hash``): each module
    and what it imports inside the number-deciding packages, as syntax, and any other listed
    file by its bytes. ``switched_identity`` keeps the value recorded at the switch.

    Args:
        sources: The closure's entries, each file by its component id.
        semantic_owner: Who owns the closure.
        numerical_role: What kind of number it identifies.

    Returns:
        The identity.

    Raises:
        ValueError: If the closure is empty or its files are in no checkout.
    """
    if not sources:
        raise ValueError("shared_kernel.source_inventory_invalid")
    root = checkout_root(next(iter(sources.values())))
    tracked = tuple(sorted(tracked_path(root, path) for path in sources.values()))
    rule = source_rule_closure_hash(
        root=root,
        tracked_paths=tracked,
        semantic_owner=semantic_owner,
        numerical_role=numerical_role,
    )
    key = source_switch_key(sources, semantic_owner=semantic_owner, numerical_role=numerical_role)
    return switched_identity(key, rule, root=root)


def _valid_component_id(value: str) -> bool:
    return bool(value) and all(
        part and part.replace("_", "").replace("-", "").isalnum() for part in value.split(".")
    )


__all__ = [
    "IDENTITY_ROLES_PATH",
    "IDENTITY_SWITCH_PATH",
    "NumberDecidingRule",
    "checkout_root",
    "identity_switch",
    "number_deciding_closure",
    "number_deciding_rule",
    "source_bytes_syntax_sha256",
    "source_component_id",
    "source_component_id_from_tracked_path",
    "source_module_imports",
    "source_rule_closure_hash",
    "source_switch_key",
    "source_syntax_sha256",
    "switched_identity",
    "switched_source_identity",
]
