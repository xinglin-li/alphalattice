"""Semantic source-content identity separated from physical repository layout."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath
from typing import cast

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
    value or statement does (binding plan R1).

    Args:
        path: Python module file to parse.

    Returns:
        SHA-256 digest of the parsed syntax after removing bare string statements.
    """
    return _parsed(path, package="")[0]


def source_module_imports(path: Path, *, package: str) -> tuple[str, ...]:
    """The modules one source file imports, as a rule closure's walk follows them.

    Args:
        path: The module's file.
        package: The package its relative imports resolve in.

    Returns:
        The imported modules' dotted names, outside `if TYPE_CHECKING:`.
    """
    return _parsed(path, package=package)[1]


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
    source = source_root(root)

    def file_of(module: str) -> Path | None:
        base = source.joinpath(*module.split("."))
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return candidate
        return None

    def deciding(module: str) -> bool:
        parts = module.split(".")
        return (
            len(parts) >= 3
            and parts[0] == "alphalattice"
            and parts[1] in _PHYSICAL_NAMESPACE_GROUPS
            and parts[2] in rule.packages
            and not rule.platform_segments.intersection(parts[3:])
        )

    found: dict[str, str] = {}
    pending = list(entries)
    while pending:
        module = pending.pop()
        if module in found or module in excluded:
            continue
        path = file_of(module)
        if path is None:
            if module in entries:
                raise ValueError("shared_kernel.source_closure_entry_missing")
            continue
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        found[module], imported = _parsed(path, package=package)
        for name in imported:
            parts = name.split(".")
            for end in range(3, len(parts) + 1):  # each package an import passes through
                prefix = ".".join(parts[:end])
                if deciding(prefix) and prefix not in found:
                    pending.append(prefix)
    return dict(sorted(found.items()))


def _parsed(path: Path, *, package: str) -> tuple[str, tuple[str, ...]]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError("shared_kernel.source_syntax_unreadable") from error
    return _parsed_payload(payload, package=package, filename=path.name)


def _parsed_payload(payload: bytes, *, package: str, filename: str) -> tuple[str, tuple[str, ...]]:
    key = (hashlib.sha256(payload).hexdigest(), package)
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
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            bound.add(node.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Import):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not (
            node.level == 0 and node.module == SPAN_MODULE
        ):
            bound.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            bound.update(alias.asname for alias in node.names if alias.asname)
    return bound


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
    without_measurement(tree)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
            kept = [statement for statement in body if not _bare_string(statement)]
            node.body = kept or [ast.Pass()]  # type: ignore[attr-defined]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


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
            if node.level:
                anchor = package.split(".")
                base = ".".join(anchor[: len(anchor) - node.level + 1])
                stem = f"{base}.{node.module}" if node.module else base
            else:
                stem = node.module or ""
            names.append(stem)
            names.extend(f"{stem}.{alias.name}" for alias in node.names if alias.name != "*")
    return tuple(names)


def _type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


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

    A role's closure by rule (binding plan R1): each tracked module is an entry, and the
    closure is what the entries import inside the number-deciding packages, each module
    hashed as its syntax (``number_deciding_closure``); a tracked file that is not a module
    under ``src`` (a lock, a Skill, a profile) is hashed by its bytes.

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
                "modules": number_deciding_closure(
                    modules, root=root, rule=number_deciding_rule(root)
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
        return hashlib.sha256(path.read_bytes()).hexdigest()
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
    entries = hashlib.sha256("\n".join(sorted(sources)).encode("utf-8")).hexdigest()[:16]
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
