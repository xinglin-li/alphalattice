"""Semantic source-content identity separated from physical repository layout."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Lock, get_ident
from typing import Any, cast
from uuid import uuid4

from alphalattice.capabilities.domain_contracts.bounded_memo import BoundedMemo
from alphalattice.capabilities.domain_contracts.project_layout import (
    resolve_playpen_root,
    resource_path,
    source_root,
    tracked_path,
)
from alphalattice.kernel.clock import wall_time_ns
from alphalattice.kernel.config import cache_root
from alphalattice.kernel.identity import (
    RACY_SECONDS,
    FileDigestTrust,
    Stamp,
    blake3_hex,
    change_xxh3_128_hex,
    identity_blake3,
    is_blake3,
    tagged_blake3,
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


_SOURCE_ADMISSION: ContextVar[Callable[[], None] | None] = ContextVar(
    "source_read_admission", default=None
)
_SOURCE_OBSERVER: ContextVar[Callable[[dict[str, str]], None] | None] = ContextVar(
    "source_closure_observer", default=None
)


@contextmanager
def source_read_admission(check: Callable[[], None]) -> Iterator[None]:
    """Check admission before, between and after the source reads."""
    check()
    token = _SOURCE_ADMISSION.set(check)
    try:
        yield
    finally:
        _SOURCE_ADMISSION.reset(token)
    check()


@contextmanager
def observe_source_closures(observe: Callable[[dict[str, str]], None]) -> Iterator[None]:
    """Read the modules reached by each closure for the identity reach check."""
    token = _SOURCE_OBSERVER.set(observe)
    try:
        yield
    finally:
        _SOURCE_OBSERVER.reset(token)


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


type SourceSyntaxKey = tuple[bytes, str, tuple[tuple[str, str], ...]]
type SourceSyntaxValue = tuple[str, tuple[str, ...]]
type SourceSyntaxProofs = dict[SourceSyntaxKey, SourceSyntaxValue]
_PARSED: BoundedMemo[SourceSyntaxKey, SourceSyntaxValue] = BoundedMemo(capacity=2048)
"""Syntax and imports indexed by exact bytes, package and import bindings."""


def source_syntax_digest(path: Path) -> str:
    """Hash a module's executable AST without bare string statements.

    A module's identity as its syntax: the parsed tree without its docstrings and other bare
    string statements, and without its execution spans (``without_measurement``). A comment, a
    docstring, a span or the layout is not syntax, so editing one moves nothing; a changed name,
    value or statement does.

    Args:
        path: Python module file to parse.

    Returns:
        Tagged BLAKE3 of the parsed syntax after removing bare string statements.
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


def source_bytes_syntax_digest(payload: bytes) -> str:
    """A module's syntax digest from its bytes, as ``source_syntax_digest`` reads a file's.

    For a version no file holds, such as the parent commit's: the gate's neutral path (GN)
    compares a changed file with it.

    Args:
        payload: The module's source bytes.

    Returns:
        The digest ``source_syntax_digest`` gives a file holding these bytes.
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
        module_bindings: Pairs of current dotted namespaces and their stable identity names.
    """

    packages: frozenset[str]
    platform_segments: frozenset[str]
    module_bindings: tuple[tuple[str, str], ...] = ()


def _bound_name(name: str, bindings: Mapping[str, str]) -> str:
    """Apply the most specific dotted namespace binding to a name."""
    prefix = name
    while bindings and prefix:
        if prefix in bindings:
            return bindings[prefix] + name[len(prefix) :]
        prefix = prefix.rpartition(".")[0]
    return name


def current_module_name(name: str, rule: NumberDecidingRule) -> str:
    """Resolve a stable module name to the module used by this checkout."""
    return _bound_name(name, {stable: module for module, stable in rule.module_bindings})


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
        Stable module names mapped to their executable-syntax digests.
    """
    moved = tuple(pair for pair in rule.module_bindings if pair[0] != pair[1])
    bindings = dict(moved)

    def read(path: Path, package: str) -> tuple[str, tuple[str, ...]]:
        return _parsed(path, package, bindings=moved)

    found = _walked(entries, root, rule, excluded, read)
    if (observe := _SOURCE_OBSERVER.get()) is not None:
        observe(found)
    return dict(sorted((_bound_name(module, bindings), digest) for module, digest in found.items()))


def _walked[T](
    entries: tuple[str, ...],
    root: Path,
    rule: NumberDecidingRule,
    excluded: frozenset[str],
    read: Callable[[Path, str], tuple[T, tuple[str, ...]]],
    *,
    is_file: Callable[[Path], bool] = Path.is_file,
) -> dict[str, T]:
    """The closure's walk (``number_deciding_closure``), each module read by ``read``."""
    source = source_root(root)
    stable = {module: binding for module, binding in rule.module_bindings if module != binding}
    current = {binding: module for module, binding in stable.items()}
    entries = tuple(_bound_name(entry, current) for entry in entries)
    excluded = frozenset(_bound_name(module, current) for module in excluded)

    def deciding(module: str) -> bool:
        parts = _bound_name(module, stable).split(".")
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
    check_admission = _SOURCE_ADMISSION.get()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        if check_admission is not None:
            check_admission()
        path = _module_file(source, module, is_file=is_file)
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


def _module_file(
    source: Path, module: str, *, is_file: Callable[[Path], bool] = Path.is_file
) -> Path | None:
    base = source.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if is_file(candidate):
            return candidate
    return None


def _package(module: str, path: Path) -> str:
    """The package a module's relative imports resolve in."""
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def _parsed(
    path: Path, package: str, *, bindings: tuple[tuple[str, str], ...] = ()
) -> tuple[str, tuple[str, ...]]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError("shared_kernel.source_syntax_unreadable") from error
    return _parsed_payload(payload, package=package, filename=path.name, bindings=bindings)


_SYNTAX_PROOFS = ContextVar[SourceSyntaxProofs | None]("source_syntax_proofs", default=None)


@contextmanager
def source_syntax_proofs(held: SourceSyntaxProofs | None = None) -> Iterator[SourceSyntaxProofs]:
    """Carry verified syntax to a worker sharing the caller's installed code.

    A lookup uses the worker's actual bytes, package and import binding. A changed source
    takes its own parse and the receiving owner still checks its full capability.
    """
    proofs = {} if held is None else held
    token = _SYNTAX_PROOFS.set(proofs)
    try:
        yield proofs
    finally:
        _SYNTAX_PROOFS.reset(token)


def _parsed_payload(
    payload: bytes, *, package: str, filename: str, bindings: tuple[tuple[str, str], ...] = ()
) -> tuple[str, tuple[str, ...]]:
    proofs = _SYNTAX_PROOFS.get()
    exact = (payload, package, bindings)
    if proofs is not None and (held := proofs.get(exact)) is not None:
        return held
    kept = _PARSED.get(exact)
    if kept is None:
        tree = _mutable_tree(payload, filename, change_xxh3_128_hex(payload))
        imported = _imports(tree, package=package)
        if bindings:
            _bind_syntax_imports(tree, dict(bindings))
        kept = _PARSED[exact] = (_syntax_digest(tree), imported)
    if proofs is not None:
        proofs[exact] = kept
    return kept


def _bind_syntax_imports(tree: ast.Module, bindings: Mapping[str, str]) -> None:
    """Give form 1's import syntax stable names while retaining its statements and aliases."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                alias.name = _bound_name(alias.name, bindings)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names = [
                _bound_name(f"{node.module}.{alias.name}", bindings).rsplit(".", 1)
                for alias in node.names
            ]
            stems = {stem for stem, _ in names}
            if len(stems) == 1:
                node.module = stems.pop()
                for alias, (_, name) in zip(node.names, names, strict=True):
                    alias.name = name


SPAN_MODULE = "alphalattice.capabilities.domain_contracts.spans"
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
    """Digest a module's dumped tree or a canonical module's rendering."""
    return tagged_blake3(identity_blake3(text.encode("utf-8")))


def _bytes_key(payload: bytes, package: str) -> tuple[str, str]:
    """Key the parse cache by the module's exact bytes and its package."""
    return change_xxh3_128_hex(payload), package


def _dumped(tree: ast.Module) -> str:
    """The tree as syntax: without its spans and bare strings, and without positions."""
    without_measurement(tree)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
            kept = [statement for statement in body if not _bare_string(statement)]
            node.body = kept or [ast.Pass()]  # type: ignore[attr-defined]
    # Python 3.13 drops empty fields by default; the identity keeps them, as 3.12 wrote it.
    return ast.dump(tree, include_attributes=False, show_empty=True)


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
    - a closure module is named by its digest, so code moved to a new path with the same
      body keeps its digest; imports outside the closure use their semantic binding when
      declared and their dotted name otherwise.

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
    return _canonical_digests(_walked(entries, root, rule, frozenset(), _canonical_walk), rule)


def _canonical_digests(parsed: dict[str, _Canonical], rule: NumberDecidingRule) -> dict[str, str]:
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
        module: [target(reference) for reference in parsed[module].references] for module in found
    }
    imported = {
        module: [target(reference) for reference in parsed[module].imports] for module in found
    }
    deep: dict[str, str] = {}
    bindings = {module: stable for module, stable in rule.module_bindings if module != stable}

    def marker(target: tuple[str, str], ring: Mapping[str, str]) -> str:
        module, name = target
        if module in deep:
            head = deep[module]
        elif module in ring:
            head = f"ring:{ring[module]}"
        else:
            qualified = f"{module}.{name}" if name else module
            return "@ext:" + _bound_name(qualified, bindings)
        return f"@{head}:{name}" if name else f"@{head}"

    def rendered(module: str, ring: Mapping[str, str]) -> str:
        held = parsed[module]
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
    digests = {module: deep[module] for module in found}
    if (observe := _SOURCE_OBSERVER.get()) is not None:
        observe(digests)
    return digests


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
        kept = _CANONICAL[key] = _cached_canonical(payload, key, path.name)
    return kept


def _cached_canonical(payload: bytes, key: tuple[str, str], filename: str) -> _Canonical:
    """The module's canonical form as the user's cache keeps it, else parsed and kept there.

    An entry is keyed by everything that decides the form: the BLAKE3 of the module's bytes, its
    package, the BLAKE3 of this module (the canonicaliser) and the interpreter's cache tag; the
    folder names the last two, the file `<bytes>-<package>.<the entry's own BLAKE3>.json` the rest.
    A hit needs the file's bytes to hash to the digest its name holds, the whole key to match and
    the form to be one the walk can read (`_shaped`). Anything else is a miss, parsed as before and
    written again beside it: the cache is never an error and never trusted beyond what it proves.
    """
    place = _cache_place()
    if place is None:
        return _canonical_tree(_mutable_tree(payload, filename, key[0]), key)
    folder, canonicaliser, tag = place
    named = {
        "bytes": tagged_blake3(identity_blake3(payload)),
        "package": key[1],
        "canonicaliser": canonicaliser,
        "cache_tag": tag,
    }
    stem = f"{blake3_hex(named['bytes'])}-{key[1]}"
    for entry, digest in _entries(folder, stem):
        held = _proved(entry, digest)
        if held is not None and held.get("key") == named and (form := _shaped(held, key)):
            return form
    canonical = _canonical_tree(_mutable_tree(payload, filename, key[0]), key)
    _keep(folder, stem, named, canonical)
    return canonical


def _entries(folder: Path, stem: str) -> list[tuple[Path, str]]:
    """The folder's entries for this stem with the digest each name holds; none when the folder
    is missing or cannot be listed, which the caller reads as a miss."""
    found = []
    try:
        with os.scandir(folder) as listing:
            for item in listing:
                named = _NAMED.fullmatch(item.name)
                if named is not None and named["stem"] == stem:
                    found.append((Path(item.path), named["digest"]))
    except OSError:
        return []
    return found


def _proved(entry: Path, digest: str) -> dict[str, Any] | None:
    """The entry's JSON when its bytes, read once, hash to the digest its name holds: the bytes
    decoded are the very bytes hashed, so a file changed in place cannot pass between the two."""
    try:
        data = FileDigestTrust.proven_bytes(entry, digest)
        if data is None:
            return None
        held = json.loads(data)
    except (OSError, ValueError):
        return None
    return held if isinstance(held, dict) else None


def _shaped(held: dict[str, Any], key: tuple[str, str]) -> _Canonical | None:
    """The stored form when the walk can read it: lists of strings, each reference and import a
    `("module", dotted)` or `("from", module, name)`, and the template's text alternating with
    the placeholder numbers 0, 1, ..., one per reference in order; else None."""

    def strings(value: object) -> tuple[str, ...] | None:
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return tuple(value)
        return None

    def bindings(value: object) -> tuple[_Reference, ...] | None:
        if not isinstance(value, list):
            return None
        found: list[_Reference] = []
        for item in value:
            match item:
                case ["module", str()] | ["from", str(), str()]:
                    found.append(cast(_Reference, tuple(item)))
                case _:
                    return None
        return tuple(found)

    walked, template = strings(held.get("walked")), strings(held.get("template"))
    references, imports = bindings(held.get("references")), bindings(held.get("imports"))
    if walked is None or template is None or references is None or imports is None:
        return None
    rest = iter(template)
    if next(rest, None) is None:
        return None
    try:  # after the first text, each placeholder number with the text that follows it
        numbers = [number for number, _text in zip(rest, rest, strict=True)]
    except ValueError:
        return None
    if numbers != [str(index) for index in range(len(references))]:
        return None
    return _Canonical(
        key=key, walked=walked, template=template, references=references, imports=imports
    )


def _own_digest() -> str | None:
    try:
        return FileDigestTrust.fresh_digest(Path(__file__))
    except OSError:
        return None


_CANONICALISER = _own_digest()
"""The BLAKE3 of this module's bytes, which key every cache entry; None when they cannot be read."""


def _cache_place() -> tuple[Path, str, str] | None:
    """`<cache root>/source-identity/<cache tag>-<canonicaliser digest>`, or None when the
    interpreter keeps no bytecode cache tag or this module's bytes cannot be read."""
    tag = sys.implementation.cache_tag
    if tag is None or _CANONICALISER is None:
        return None
    folder = cache_root() / "source-identity" / f"{tag}-{blake3_hex(_CANONICALISER)}"
    return folder, _CANONICALISER, tag


_NAMED = re.compile(r"(?P<stem>[0-9a-f]{64}-[\w.]*)\.(?P<digest>[0-9a-f]{64})\.json")
"""An entry's file: `<bytes digest>-<package>.<the entry's own BLAKE3>.json`."""


def _keep(folder: Path, stem: str, named: dict[str, str], canonical: _Canonical) -> None:
    """Write an entry whole: a partial file, named after its own BLAKE3 as it is renamed into
    place. Nothing else in the cache is touched; a failed write leaves the next process to parse."""
    partial = folder / f".{stem}.{os.getpid()}.{get_ident()}.partial"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        text = {
            "key": named,
            "walked": canonical.walked,
            "template": canonical.template,
            "references": canonical.references,
            "imports": canonical.imports,
        }
        partial.write_text(json.dumps(text, ensure_ascii=False), encoding="utf-8")
        own = blake3_hex(FileDigestTrust.fresh_digest(partial))
        os.replace(partial, folder / f"{stem}.{own}.json")
    except OSError:
        with suppress(OSError):
            partial.unlink(missing_ok=True)


def _keyed(path: Path, package: str) -> tuple[tuple[str, str], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError("shared_kernel.source_syntax_unreadable") from error
    return _bytes_key(payload, package), payload


_TREES: BoundedMemo[tuple[str, bytes], ast.Module] = BoundedMemo()
_TREE_LOCK = Lock()


def _mutable_tree(payload: bytes, filename: str, change_key: str) -> ast.Module:
    """Return a private tree from the observed source change key and exact bytes."""
    key = change_key, payload
    with _TREE_LOCK:
        tree = _TREES.get(key)
        if tree is None:
            tree = _TREES[key] = _tree(payload, filename)
    return deepcopy(tree)


def _tree(payload: bytes, filename: str) -> ast.Module:
    """A source module's one parse, which the work census names by its bytes."""
    try:
        return ast.parse(payload, filename=filename)
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


_FIELDS_SINCE_312 = frozenset({"default_value"})
"""Syntax fields Python added after 3.12 (PEP 696's type parameter defaults), written only when
present."""


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
            if value is None and field in _FIELDS_SINCE_312:
                continue  # an absent newer field writes nothing: one identity on every Python
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


def number_deciding_rule(root: Path) -> NumberDecidingRule:
    """Which code can move a result, as the identity-roles table names it."""
    try:
        document = json.loads((root / IDENTITY_ROLES_PATH).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("invalid identity roles table")
        bindings = document.get("module_bindings", {})
        if not isinstance(bindings, dict) or any(
            not isinstance(name, str)
            or not name.startswith("alphalattice.")
            or not all(part.isidentifier() for part in name.split("."))
            for name in (*bindings, *bindings.values())
        ):
            raise ValueError("invalid module binding")
        if len(set(bindings.values())) != len(bindings):
            raise ValueError("duplicate current module namespace")
        rule = NumberDecidingRule(
            packages=frozenset(str(name) for name in document["number_deciding_packages"]),
            platform_segments=frozenset(str(name) for name in document["platform_segments"]),
            module_bindings=tuple((module, binding) for binding, module in bindings.items()),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("shared_kernel.identity_roles_unreadable") from error
    if not rule.packages:
        raise ValueError("shared_kernel.identity_roles_unreadable")
    return rule


type _FileVersion = tuple[bool, Stamp | None]
type _RoleInput = tuple[Path, tuple[str, ...], str, str, NumberDecidingRule]
type _RoleClosure = tuple[tuple[tuple[Path, _FileVersion], ...], str]

_ROLE_CLOSURES = BoundedMemo[_RoleInput, _RoleClosure](capacity=128)
_MODULE_STAMPS = BoundedMemo[tuple[Path, str], tuple[Stamp, _Canonical]](capacity=2048)
_ROLE_READ = Lock()


class _RoleSources:
    def __init__(self) -> None:
        self.opened: dict[Path, _FileVersion] = {}
        self.confirmed: dict[Path, _FileVersion] = {}
        self.phase: str | None = None


_ACQUIRED_ROLE_SOURCES: ContextVar[_RoleSources | None] = ContextVar(
    "acquired_role_sources", default=None
)


@contextmanager
def source_role_acquisition(
    *, fresh: bool = False, shared: _RoleSources | None = None
) -> Iterator[None]:
    """Share source observations among the roles in the current request or Task phase; `shared`
    carries an open acquisition to work the phase hands to another thread."""
    if not fresh and shared is None and _ACQUIRED_ROLE_SOURCES.get() is not None:
        yield
        return
    token = _ACQUIRED_ROLE_SOURCES.set(_RoleSources() if shared is None else shared)
    try:
        yield
    finally:
        _ACQUIRED_ROLE_SOURCES.reset(token)


def open_source_role_acquisition() -> _RoleSources | None:
    """The phase's open source acquisition, for work it hands to another thread."""
    return _ACQUIRED_ROLE_SOURCES.get()


def open_source_phase() -> str | None:
    """The open acquisition's phase name, which a kept worker's calls in the phase share."""
    acquired = _ACQUIRED_ROLE_SOURCES.get()
    if acquired is None:
        return None
    if acquired.phase is None:
        acquired.phase = uuid4().hex
    return acquired.phase


def _file_version(path: Path) -> _FileVersion:
    return FileDigestTrust.version(path)


def _settled_stamp(stamp: Stamp | None, recorded_at: int) -> bool:
    return stamp is not None and recorded_at - max(stamp[-2:]) > RACY_SECONDS * 10**9


class _ClosureInputs:
    """Files and absent import candidates encountered while resolving a role."""

    def __init__(self, acquired: _RoleSources | None) -> None:
        self.recorded_at = wall_time_ns()
        self.acquired = acquired if acquired is not None else _RoleSources()
        self.versions: dict[Path, _FileVersion] = {}
        self.parsed: dict[tuple[Path, str], tuple[Stamp, _Canonical]] = {}

    def is_file(self, path: Path) -> bool:
        if path not in self.versions:
            version = self.acquired.confirmed.get(path)
            if version is None:
                if path not in self.acquired.opened:
                    self.acquired.opened[path] = _file_version(path)
                version = self.acquired.opened[path]
            self.versions[path] = version
        return self.versions[path][0]

    def canonical(self, path: Path, package: str) -> tuple[_Canonical, tuple[str, ...]]:
        stamp = self.versions[path][1]
        identity = path, package
        settled = _settled_stamp(stamp, self.recorded_at)
        held = _MODULE_STAMPS.get(identity) if settled else None
        if held is not None and held[0] == stamp:
            parsed = held[1]
        else:
            # New bytes need their own final check. A retained parse can carry the
            # observation already confirmed by another role in this acquisition.
            self.acquired.confirmed.pop(path, None)
            parsed = _canonical(path, package)
            if settled and stamp is not None:
                self.parsed[identity] = stamp, parsed
        return parsed, parsed.walked

    def stable(self) -> bool:
        confirmed = self.acquired.confirmed
        stable = _matches_role(self.versions.items(), self.recorded_at, confirmed)
        # The final check can also confirm the modules parsed in this acquisition.
        # When it stops at a changed input, independently confirm only the unvisited modules.
        for identity, (stamp, parsed) in self.parsed.items():
            path, _ = identity
            if path not in confirmed:
                confirmed[path] = _file_version(path)
            if confirmed[path] == (True, stamp):
                _MODULE_STAMPS[identity] = stamp, parsed
        return stable


def _matches_role(
    files: Iterable[tuple[Path, _FileVersion]],
    recorded_at: int,
    acquired: dict[Path, _FileVersion] | None = None,
) -> bool:
    check = _SOURCE_ADMISSION.get()
    for path, version in files:
        if check is not None:
            check()
        present, stamp = version
        if present and not _settled_stamp(stamp, recorded_at):
            return False
        observed = acquired.get(path) if acquired is not None else None
        if observed is None:
            observed = _file_version(path)
            if acquired is not None:
                acquired[path] = observed
        if observed != version:
            return False
    return True


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
    with _ROLE_READ:
        rule = number_deciding_rule(root)
        key = root, tracked_paths, semantic_owner, numerical_role, rule
        observing = _SOURCE_OBSERVER.get() is not None
        inputs = _ClosureInputs(None if observing else _ACQUIRED_ROLE_SOURCES.get())
        held = None if observing else _ROLE_CLOSURES.get(key)
        if held is not None and _matches_role(
            held[0], inputs.recorded_at, inputs.acquired.confirmed
        ):
            return held[1]
        python = {path: path.startswith("src/") and path.endswith(".py") for path in tracked_paths}
        modules = tuple(
            ".".join(PurePosixPath(path).with_suffix("").parts[1:])
            for path, module in python.items()
            if module
        )
        others = tuple(path for path, module in python.items() if not module)
        parsed = _walked(modules, root, rule, frozenset(), inputs.canonical, is_file=inputs.is_file)
        digest = cast(
            str,
            canonical_hash(
                {
                    "semantic_owner": semantic_owner,
                    "numerical_role": numerical_role,
                    "form": CLOSURE_FORM,
                    "modules": sorted(_canonical_digests(parsed, rule).values()),
                    "files": {
                        source_component_id_from_tracked_path(path): _resource_digest(
                            root, path, inputs
                        )
                        for path in others
                    },
                }
            ),
        )
        if not observing and inputs.stable():
            _ROLE_CLOSURES[key] = tuple(inputs.versions.items()), digest
        return digest


def _resource_digest(root: Path, relative: str, inputs: _ClosureInputs) -> str:
    path = resource_path(root, relative)
    if inputs.is_file(path):
        inputs.acquired.confirmed.pop(path, None)
        return FileDigestTrust.fresh_digest(path)
    # Development runners are not runtime entries. The wheel carries the
    # accepted tree's byte commitments, preserving the same closure value.
    manifest = root / "config/release/source-hashes.json"
    if inputs.is_file(manifest):
        inputs.acquired.confirmed.pop(manifest, None)
        hashes = json.loads(manifest.read_text(encoding="utf-8"))
        value = hashes.get(relative)
        if isinstance(value, str) and is_blake3(value):
            return value
    raise ValueError(f"shared_kernel.source_resource_missing:{relative}")


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
    """Hash the listed source files through the rule closure.

    The listed files are the rule closure's entries (``source_rule_closure_hash``): each module
    and what it imports inside the number-deciding packages, as syntax, and any other listed
    file by its bytes.

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
    return source_rule_closure_hash(
        root=root,
        tracked_paths=tracked,
        semantic_owner=semantic_owner,
        numerical_role=numerical_role,
    )


def _valid_component_id(value: str) -> bool:
    return bool(value) and all(
        part and part.replace("_", "").replace("-", "").isalnum() for part in value.split(".")
    )


__all__ = [
    "IDENTITY_ROLES_PATH",
    "NumberDecidingRule",
    "checkout_root",
    "current_module_name",
    "number_deciding_closure",
    "number_deciding_rule",
    "observe_source_closures",
    "source_bytes_syntax_digest",
    "source_component_id",
    "source_component_id_from_tracked_path",
    "source_module_imports",
    "source_rule_closure_hash",
    "source_syntax_digest",
    "switched_source_identity",
]
