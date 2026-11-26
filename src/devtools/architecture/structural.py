"""Deterministic structural checks for the ignored playpen closure."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from enum import StrEnum
from functools import partial
from importlib.util import resolve_name
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class GenerationModel(StrEnum):
    SINGLE_CURRENT = "SINGLE_CURRENT"
    BY_CATALOG = "BY_CATALOG"
    BY_REVISION = "BY_REVISION"
    OPERATIONAL_PROJECTION = "OPERATIONAL_PROJECTION"
    LEGACY_MIGRATION_ONLY = "LEGACY_MIGRATION_ONLY"


class StorageAuthorityGroup(_Contract):
    tables: tuple[str, ...] = Field(min_length=1)
    database: str = Field(min_length=1)
    """The file the tables live in: `workspace` (the market-data DuckDB), `task-control`,
    `task-heartbeats`, `observation`, `knowledge-index`, `knowledge-hybrid-index`."""
    semantic_role: str = Field(min_length=1)
    generation_model: GenerationModel
    authority_owner: str = Field(min_length=1)
    schema_writers: tuple[str, ...] = ()
    """Packages besides the authority's that create or alter these tables: the split the tree
    holds, a table's schema written by another package than its data."""
    retention_owner: str = Field(min_length=1)
    rebuildability: str = Field(min_length=1)
    artifact_relationship: str = Field(min_length=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_tables(self) -> StorageAuthorityGroup:
        if tuple(sorted(set(self.tables))) != tuple(sorted(self.tables)):
            raise ValueError("storage authority group tables must be unique")
        if self.artifact_relationship == "DUAL_AUTHORITY":
            raise ValueError("derived values cannot declare dual authority")
        return self


class StorageAuthorityRegistry(_Contract):
    schema_id: Literal["alphalattice.playpen.storage-authority-registry"] = Field(alias="schema")
    groups: tuple[StorageAuthorityGroup, ...] = Field(min_length=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_unique_tables(self) -> StorageAuthorityRegistry:
        tables = [(group.database, table) for group in self.groups for table in group.tables]
        if len(tables) != len(set(tables)):
            raise ValueError("storage authority registry contains duplicate tables")
        return self

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(sorted(table for group in self.groups for table in group.tables))


class ArtifactAuthorityGroup(_Contract):
    path_prefixes: tuple[str, ...] = Field(min_length=1)
    roots: tuple[Literal["artifacts", "runtime", "experiment", "workspace"], ...] = Field(
        min_length=1
    )
    """Where the prefixes lie: `artifacts` (the workspace's `artifacts/`, or a research
    input's captured copy), `runtime` (its `runtime/artifacts/`), `experiment` (an experiment's
    output) or `workspace` (the workspace root). A writer is constructed with its root: the roots
    listed are those its constructions name literally, and `artifacts` where they pass one on."""
    semantic_role: str = Field(min_length=1)
    authority_owner: str = Field(min_length=1)
    retention_owner: str = Field(min_length=1)
    rebuildability: str = Field(min_length=1)
    artifact_relationship: str = Field(min_length=1)
    eviction_authority: Literal[
        "NONE", "SEPARATE_USER_GOVERNED_PLAN_REQUIRED", "CURRENT_STATE_RETENTION_POLICY"
    ]
    retired: str | None = None
    """The commit that retired the writer, when no code writes these paths any more: the group
    stays so that a workspace's leftover files are named."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_paths(self) -> ArtifactAuthorityGroup:
        if self.path_prefixes != tuple(sorted(set(self.path_prefixes))):
            raise ValueError("artifact authority paths must be sorted and unique")
        if self.artifact_relationship == "DUAL_AUTHORITY":
            raise ValueError("artifacts cannot declare dual authority")
        return self


class ArtifactAuthorityRegistry(_Contract):
    schema_id: Literal["alphalattice.playpen.artifact-authority-registry"] = Field(alias="schema")
    groups: tuple[ArtifactAuthorityGroup, ...] = Field(min_length=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_unique_paths(self) -> ArtifactAuthorityRegistry:
        paths = [path for group in self.groups for path in group.path_prefixes]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact authority registry contains duplicate paths")
        return self

    @property
    def path_prefixes(self) -> tuple[str, ...]:
        return tuple(sorted(path for group in self.groups for path in group.path_prefixes))


def artifact_writer_violations(
    rows: tuple[Mapping[str, object], ...],
    registry: ArtifactAuthorityRegistry,
    *,
    source_classes: frozenset[str],
) -> tuple[str, ...]:
    """What writes a persistent path its registry entry does not name (G2; LAWS.md DA2).

    ``rows`` are the source-derived paths with their writers (``path_or_root``, ``writers`` with
    each writer's ``module::Class.method``). A path's registered owner is the most specific
    prefix's, and its class is among the path's writers; a path no prefix covers is refused,
    unless it is a broad root whose children are registered (its writer forms the category
    dynamically); and each registered owner is a class the source defines.
    """

    groups = [(prefix, group) for group in registry.groups for prefix in group.path_prefixes]
    violations: list[str] = []
    for row in rows:
        path = str(row["path_or_root"])
        writers = [
            str(value["writer"])
            for value in row["writers"]  # type: ignore[attr-defined]
        ]
        matches = [
            (prefix, group)
            for prefix, group in groups
            if path == prefix or path.startswith(prefix.rstrip("/") + "/")
        ]
        if not matches:
            if not any(prefix.startswith(path + "/") for prefix, _ in groups):
                violations.append(f"unregistered path {path}, written by {', '.join(writers)}")
            continue
        _, group = max(matches, key=lambda pair: len(pair[0]))
        owner = group.authority_owner.rsplit(".", 1)[-1]
        if not any(f"::{owner}." in writer or writer.endswith(f"::{owner}") for writer in writers):
            violations.append(
                f"{path} is written by {', '.join(writers)}, registered to {group.authority_owner}"
            )
    violations.extend(
        f"registered owner {group.authority_owner} is no class or function the source defines"
        for group in registry.groups
        if group.retired is None and group.authority_owner.rsplit(".", 1)[-1] not in source_classes
    )
    return tuple(sorted(set(violations)))


def source_class_names(sources: tuple[Path, ...]) -> frozenset[str]:
    """Every class, and every module-level function, the sources define."""

    names: set[str] = set()
    this = sys.modules[__name__]
    with kept_facts("class-names", this):  # each file's names, kept by its bytes
        for path in sources:
            names.update(
                kept_fact(
                    this,
                    "class-names",
                    path.read_bytes(),
                    _class_names_of,
                    lambda value: (
                        isinstance(value, list) and all(isinstance(item, str) for item in value)
                    ),
                )
            )
    return frozenset(names)


def _class_names_of(data: bytes) -> list[str]:
    """One module's classes, at any depth, and its module-level functions."""
    tree = ast.parse(file_text(data))
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    names.update(
        node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )
    return sorted(names)


class StoreSurface(_Contract):
    public_methods: tuple[str, ...]
    table_names: tuple[str, ...]
    connect_call_count: int = Field(ge=0)
    direct_legacy_panel_reference_count: int = Field(ge=0)
    source_bytes: int = Field(ge=0)
    source_lines: int = Field(ge=0)


class PerformanceBaselines(_Contract):
    schema_id: Literal["alphalattice.playpen.performance-baselines"] = Field(alias="schema")
    workspace_id: str = Field(min_length=1)
    captured_on: str = Field(min_length=1)
    source_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    available_reproduction_script_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    missing_original_scripts: tuple[str, ...]
    review_threshold_fraction: float = Field(gt=0.0, lt=1.0)
    measurements: dict[str, float]
    usage: str = Field(min_length=1)


class PackageArchitectureRole(StrEnum):
    DESK = "DESK"
    INDEPENDENT_OVERSIGHT = "INDEPENDENT_OVERSIGHT"
    DETERMINISTIC_CAPABILITY = "DETERMINISTIC_CAPABILITY"
    INTEGRATION_ADAPTER = "INTEGRATION_ADAPTER"
    CROSS_DOMAIN_PROTOCOL = "CROSS_DOMAIN_PROTOCOL"
    CONTROL_RUNTIME = "CONTROL_RUNTIME"
    COMPOSITION_ROOT = "COMPOSITION_ROOT"
    ENGINEERING_TOOL = "ENGINEERING_TOOL"


class PackageArchitecture(_Contract):
    package_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    product_area: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    capability_owner: str = Field(min_length=1)
    role: PackageArchitectureRole
    dependency_level: int = Field(ge=0)
    authority_kind: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    allowed_peer_dependencies: tuple[str, ...]
    artifact_writer: bool
    agent_free_import_roots: tuple[str, ...] = ()
    physical_namespace_target: str = Field(min_length=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_peer_inventory(self) -> PackageArchitecture:
        if self.allowed_peer_dependencies != tuple(sorted(set(self.allowed_peer_dependencies))):
            raise ValueError("package peer dependencies must be sorted and unique")
        if self.package_id in self.allowed_peer_dependencies:
            raise ValueError("package cannot authorize itself as a peer dependency")
        if self.agent_free_import_roots != tuple(sorted(set(self.agent_free_import_roots))):
            raise ValueError("agent-free import roots must be sorted and unique")
        return self


class ImportHygiene(_Contract):
    schema_id: Literal["alphalattice.import-hygiene"] = Field(alias="schema")
    legacy_lazy_imports: dict[str, Annotated[int, Field(strict=True, ge=1)]]


class PackageArchitectureRegistry(_Contract):
    schema_id: Literal["alphalattice.playpen.package-architecture"] = Field(alias="schema")
    note: str = ""
    packages: tuple[PackageArchitecture, ...] = Field(min_length=1)
    deferred_imports: dict[str, str] = Field(default_factory=dict)
    legacy_lazy_imports: dict[str, int] = Field(default_factory=dict)
    private_import_fences: dict[str, dict[str, str]] = Field(default_factory=dict)
    retired_imports: dict[str, str] = Field(default_factory=dict)
    """Each logical owner a cutover retired, with its reason; import_contracts.build bans it."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_inventory(self) -> PackageArchitectureRegistry:
        package_ids = tuple(value.package_id for value in self.packages)
        targets = tuple(value.physical_namespace_target for value in self.packages)
        if len(package_ids) != len(set(package_ids)):
            raise ValueError("package architecture registry repeats a package")
        if len(targets) != len(set(targets)):
            raise ValueError("package architecture registry repeats a physical target")
        known = set(package_ids)
        unknown = sorted(
            {
                peer
                for value in self.packages
                for peer in value.allowed_peer_dependencies
                if peer not in known
            }
        )
        if unknown:
            raise ValueError(f"package architecture registry has unknown peers: {unknown}")
        reasons = [
            *self.deferred_imports.values(),
            *(r for f in self.private_import_fences.values() for r in f.values()),
            *self.retired_imports.values(),
        ]
        if any(
            not re.fullmatch(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*", name)
            for name in self.retired_imports
        ):
            raise ValueError("a retired import is a dotted logical module name")
        if any(not reason.strip() for reason in reasons):
            raise ValueError("package import declarations require their reason")
        return self

    @property
    def by_package(self) -> dict[str, PackageArchitecture]:
        return {value.package_id: value for value in self.packages}


_CREATE_TABLE = re.compile(
    r"CREATE TABLE(?: IF NOT EXISTS)?\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE
)
_LEGACY_PANEL_TABLES = (
    "sector_neutral_panel_current",
    "sector_neutral_panel_revision",
)
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


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_storage_authority_registry(path: Path) -> StorageAuthorityRegistry:
    return StorageAuthorityRegistry.model_validate(_load_json(path))  # type: ignore[no-any-return]


def load_artifact_authority_registry(path: Path) -> ArtifactAuthorityRegistry:
    return ArtifactAuthorityRegistry.model_validate(_load_json(path))  # type: ignore[no-any-return]


def load_store_surface(path: Path) -> StoreSurface:
    return StoreSurface.model_validate(_load_json(path))  # type: ignore[no-any-return]


def load_performance_baselines(path: Path) -> PerformanceBaselines:
    return PerformanceBaselines.model_validate(_load_json(path))  # type: ignore[no-any-return]


def load_package_architecture_registry(path: Path) -> PackageArchitectureRegistry:
    legacy = ImportHygiene.model_validate(
        _load_json(path.parent.parent / "config/import-hygiene.json")
    )
    policy = PackageArchitectureRegistry.model_validate(_load_json(path))
    return policy.model_copy(update={"legacy_lazy_imports": legacy.legacy_lazy_imports})  # type: ignore[no-any-return]


def discover_store_surface(path: Path) -> StoreSurface:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    tree = ast.parse(text, filename=str(path))
    repository_names = {
        "MarketDataRepository",
        "FeatureStateRepository",
        "PanelStateRepository",
        "ResearchFoundationStateRepository",
        "WorkspaceReadinessRepository",
    }
    store_classes = tuple(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in repository_names
    )
    if not store_classes:
        raise ValueError("workspace repository class is missing")
    public_methods = tuple(
        sorted(
            node.name
            for store_class in store_classes
            for node in store_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        )
    )
    connect_calls = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_connect"
    )
    return StoreSurface(
        public_methods=public_methods,
        table_names=tuple(sorted(set(_CREATE_TABLE.findall(text)))),
        connect_call_count=connect_calls,
        direct_legacy_panel_reference_count=sum(
            text.count(table) for table in _LEGACY_PANEL_TABLES
        ),
        source_bytes=len(raw),
        source_lines=len(text.splitlines()),
    )


def discover_repository_surface(paths: tuple[Path, ...]) -> StoreSurface:
    """Combine the direct surfaces of the physical workspace repositories."""

    if not paths:
        raise ValueError("repository surface discovery requires at least one owner")
    surfaces = tuple(discover_store_surface(path) for path in paths)
    return StoreSurface(
        public_methods=tuple(
            sorted({method for surface in surfaces for method in surface.public_methods})
        ),
        table_names=tuple(sorted({table for surface in surfaces for table in surface.table_names})),
        connect_call_count=sum(surface.connect_call_count for surface in surfaces),
        direct_legacy_panel_reference_count=sum(
            surface.direct_legacy_panel_reference_count for surface in surfaces
        ),
        source_bytes=sum(surface.source_bytes for surface in surfaces),
        source_lines=sum(surface.source_lines for surface in surfaces),
    )


def store_surface_violations(current: StoreSurface, baseline: StoreSurface) -> tuple[str, ...]:
    violations: list[str] = []
    added_methods = sorted(set(current.public_methods) - set(baseline.public_methods))
    if added_methods:
        violations.append(f"store public methods grew: {added_methods}")
    added_tables = sorted(set(current.table_names) - set(baseline.table_names))
    if added_tables:
        violations.append(f"store tables grew: {added_tables}")
    if current.connect_call_count > baseline.connect_call_count:
        violations.append(
            "store _connect calls grew: "
            f"{baseline.connect_call_count} -> {current.connect_call_count}"
        )
    if current.direct_legacy_panel_reference_count > baseline.direct_legacy_panel_reference_count:
        violations.append(
            "legacy Panel references grew: "
            f"{baseline.direct_legacy_panel_reference_count} -> "
            f"{current.direct_legacy_panel_reference_count}"
        )
    return tuple(violations)


def storage_registry_violations(
    surface: StoreSurface, registry: StorageAuthorityRegistry
) -> tuple[str, ...]:
    discovered = set(surface.table_names)
    registered = set(registry.table_names)
    violations: list[str] = []
    missing = sorted(discovered - registered)
    stale = sorted(registered - discovered)
    if missing:
        violations.append(f"unregistered persistent tables: {missing}")
    if stale:
        violations.append(f"registry tables absent from store schema: {stale}")
    return tuple(violations)


class TableWrite(_Contract):
    """One static SQL statement that creates, alters, drops or writes a table."""

    table: str
    kind: Literal["schema", "data"]
    writer: str
    """The module and the class or top-level function the statement sits in."""
    package: str


_SCHEMA_WRITE = re.compile(
    r"\b(?:CREATE\s+(?:OR\s+REPLACE\s+)?(?:VIRTUAL\s+)?TABLE(?:\s+IF\s+NOT\s+EXISTS)?"
    r"|ALTER\s+TABLE|DROP\s+TABLE(?:\s+IF\s+EXISTS)?)\s+((?:[A-Za-z_]\w*\.)?[A-Za-z_]\w*)",
    re.IGNORECASE,
)
_TEMPORARY_TABLE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?TEMP(?:ORARY)?\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(\w+)",
    re.IGNORECASE,
)
_DATA_WRITE = re.compile(
    r"\b(?:INSERT(?:\s+OR\s+(?:REPLACE|IGNORE))?\s+INTO|DELETE\s+FROM|MERGE\s+INTO)"
    r"\s+((?:[A-Za-z_]\w*\.)?[A-Za-z_]\w*)|\bUPDATE\s+((?:[A-Za-z_]\w*\.)?[A-Za-z_]\w*)\s+SET\b",
    re.IGNORECASE,
)
_SQL_WORDS = frozenset({"IF", "NOT", "EXISTS", "SELECT", "AS", "VALUES", "TABLE"})


def discover_table_writes(
    sources: tuple[Path, ...], *, source_root: Path
) -> tuple[TableWrite, ...]:
    """Every static table write in ``sources``: a statement in a string, not a docstring.

    A table named by a placeholder is a scan limit, never a write; a temporary table the module
    creates is not persistent, and neither are its writes. A module that writes tables it names
    at run time declares them in a module-level ``WRITTEN_TABLES`` tuple, each a data write of
    that module.
    """

    found: set[TableWrite] = set()
    this = sys.modules[__name__]
    with kept_facts("table-writes", this):  # each file's writes, kept by its bytes
        for path in sources:
            module = _logical_module_for_path(path, source_root=source_root)
            found.update(
                TableWrite(table=table, kind=kind, writer=writer, package=package)
                for table, kind, writer, package in kept_fact(
                    this,
                    "table-writes " + module,
                    path.read_bytes(),
                    partial(_table_writes_of, module=module),
                    _string_rows,
                )
            )
    return tuple(sorted(found, key=lambda value: (value.table, value.kind, value.writer)))


def _table_writes_of(data: bytes, module: str) -> list[list[str]]:
    """One module's static table writes from its bytes, as [table, kind, writer, package]."""
    tree = ast.parse(file_text(data))
    package = module.split(".", 1)[0]
    found = _declared_written_tables(tree, module, package)
    texts = _statement_texts(tree)
    temporary = {
        match.group(1).lower()
        for node in texts
        for match in _TEMPORARY_TABLE.finditer(str(node.value))
    }
    classes = [
        (node.lineno, node.end_lineno or node.lineno, node.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    ]
    functions = [
        (node.lineno, node.end_lineno or node.lineno, node.name)
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    for node in texts:
        writer = _enclosing_writer(node.lineno, classes, functions)
        found.update(
            (table, kind, f"{module}:{writer}", package)
            for kind, table in _statement_writes(str(node.value))
            if table.upper() not in _SQL_WORDS and table.lower() not in temporary
        )
    return sorted(map(list, found))


def _statement_texts(tree: ast.Module) -> list[ast.Constant]:
    """Every string constant in a module that is not a docstring."""
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _enclosing_writer(
    line: int, classes: list[tuple[int, int, str]], functions: list[tuple[int, int, str]]
) -> str:
    """The innermost class around ``line``, else its top-level function, else the module."""
    inner = [value for value in classes if value[0] <= line <= value[1]]
    if inner:
        return min(inner, key=lambda value: value[1] - value[0])[2]
    return next((name for start, end, name in functions if start <= line <= end), "<module>")


def _declared_written_tables(
    tree: ast.Module, module: str, package: str
) -> set[tuple[str, str, str, str]]:
    """The tables a module names in its module-level ``WRITTEN_TABLES`` tuple, as data writes."""
    found: set[tuple[str, str, str, str]] = set()
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "WRITTEN_TABLES"
                for target in statement.targets
            )
            and isinstance(statement.value, ast.Tuple)
        ):
            found.update(
                (element.value, "data", f"{module}:WRITTEN_TABLES", package)
                for element in statement.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
    return found


def _statement_writes(text: str) -> list[tuple[str, str]]:
    """The (kind, table) of each persistent schema or data statement one string spells."""
    statements = [
        ("schema", match.group(1))
        for match in _SCHEMA_WRITE.finditer(text)
        if not _TEMPORARY_TABLE.match(text, match.start())
    ] + [("data", match.group(1) or match.group(2)) for match in _DATA_WRITE.finditer(text)]
    return [(kind, name.rsplit(".", 1)[-1]) for kind, name in statements]


def _string_rows(value: Any) -> bool:
    """A kept fact's shape: a list of rows, each a list of strings."""
    return isinstance(value, list) and all(
        isinstance(row, list) and all(isinstance(item, str) for item in row) for row in value
    )


def table_writer_violations(
    writes: tuple[TableWrite, ...],
    registry: StorageAuthorityRegistry,
    *,
    other_data_writers: Mapping[str, object],
    whole_tree: bool = True,
) -> tuple[str, ...]:
    """What writes a table outside its registered owners (G2; LAWS.md DA2, OW1).

    A table's schema is written by its authority's package or a listed schema writer, its data by
    its authority's package or a writer the baseline holds (`table::package`, only shrinking); a
    table the code creates is registered, and, read over the whole tree, a registered table is
    created somewhere.
    """

    owners: dict[str, set[str]] = {}
    schema: dict[str, set[str]] = {}
    for group in registry.groups:
        package = group.authority_owner.split(".", 1)[0]
        for table in group.tables:
            owners.setdefault(table, set()).add(package)
            schema.setdefault(table, set()).update({package, *group.schema_writers})
    violations: list[str] = []
    created = {value.table for value in writes if value.kind == "schema"}
    for value in writes:
        if value.table not in owners:
            if value.kind == "schema":
                violations.append(f"unregistered table {value.table}, created by {value.writer}")
            continue
        allowed = schema[value.table] if value.kind == "schema" else owners[value.table]
        if value.package not in allowed and (
            value.kind == "schema" or f"{value.table}::{value.package}" not in other_data_writers
        ):
            violations.append(
                f"{value.writer} writes the {value.kind} of {value.table}, which "
                f"{' and '.join(sorted(allowed))} own{'s' if len(allowed) == 1 else ''} "
                "(config/storage-authorities.json)"
            )
    if whole_tree:
        violations.extend(
            f"registered table {table} is created nowhere"
            for table in sorted(set(owners) - created)
        )
    return tuple(sorted(set(violations)))


def duckdb_access_mode_violations(paths: tuple[Path, ...]) -> tuple[str, ...]:
    """Require every file-backed DuckDB connection to declare its access mode."""

    violations: list[str] = []
    this = sys.modules[__name__]
    with kept_facts("duckdb-access-mode", this):  # each file's lines, kept by its bytes
        for path in paths:
            violations += [
                f"{path.as_posix()}:{line}: missing read_only=..."
                for line in kept_fact(
                    this,
                    "duckdb-access-mode",
                    path.read_bytes(),
                    partial(_unmoded_duckdb_lines, filename=str(path)),
                    _int_list,
                )
            ]
    return tuple(violations)


def _unmoded_duckdb_lines(data: bytes, filename: str) -> list[int]:
    """The lines of one file's file-backed DuckDB connections that declare no access mode."""
    lines: list[int] = []
    for node in ast.walk(ast.parse(file_text(data), filename=filename)):
        if not _is_duckdb_connect(node):
            continue
        assert isinstance(node, ast.Call)
        database = node.args[0] if node.args else None
        if isinstance(database, ast.Constant) and database.value == ":memory:":
            continue
        if not any(keyword.arg == "read_only" for keyword in node.keywords):
            lines.append(node.lineno)
    return lines


def _int_list(value: Any) -> bool:
    """A kept fact's shape: a list of line numbers."""
    return isinstance(value, list) and all(type(item) is int for item in value)


def _is_duckdb_connect(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "duckdb"
        and node.func.attr == "connect"
    )


def gate_after_instance_violations(paths: tuple[Path, ...]) -> tuple[str, ...]:
    """Refuse a unit that asks for a write gate while it holds a store's instance.

    Every writer takes the gate (`hold`) first and the store's retained instance
    (`retain(...)`) second; a unit that holds the instance and then asks for the gate waits on
    a writer holding the gate while it waits for that instance. Read in the source: in one
    `with`, no `hold` follows a `retain(...)`; inside a `retain(...)` block, or after one
    entered on an exit stack, no `hold` is asked for. What a called function takes is the
    lock-order recorder's (`tests/workspace_task_runner/lock_order.py`).
    """
    violations: list[str] = []
    this = sys.modules[__name__]
    with kept_facts("gate-after-instance", this):  # each file's lines, kept by its bytes
        for path in paths:
            violations += [
                f"{path.as_posix()}:{line}: hold() asked for while a store instance is retained"
                for line in kept_fact(
                    this,
                    "gate-after-instance",
                    path.read_bytes(),
                    partial(_late_hold_lines, filename=str(path)),
                    _int_list,
                )
            ]
    return tuple(violations)


def _late_hold_lines(data: bytes, filename: str) -> list[int]:
    """The lines of one file's ``hold`` calls asked for while a store instance is retained."""
    tree = ast.parse(file_text(data), filename=filename)
    late: set[ast.Call] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            items = [item.context_expr for item in node.items]
            first = next((i for i, item in enumerate(items) if _calls(item, "retain")), None)
            if first is not None:
                late.update(
                    item
                    for item in items[first + 1 :]
                    if isinstance(item, ast.Call) and _calls(item, "hold")
                )
                for statement in node.body:
                    late.update(_own_calls(statement, "hold"))
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(node, field, None)
            if not isinstance(statements, list):
                continue
            entered = next(
                (i for i, statement in enumerate(statements) if _enters_retained(statement)),
                None,
            )
            if entered is None:
                continue
            for statement in statements[entered + 1 :]:
                late.update(_own_calls(statement, "hold"))
    return [call.lineno for call in sorted(late, key=lambda call: (call.lineno, call.col_offset))]


def _calls(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    )


def _own_calls(statement: ast.stmt, name: str) -> list[ast.Call]:
    """The calls a statement makes itself, not those of a function or class it defines."""
    found: list[ast.Call] = []
    pending: list[ast.AST] = [statement]
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call) and _calls(node, name):
            found.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return found


def _enters_retained(statement: ast.stmt) -> bool:
    return any(
        any(_calls(argument, "retain") for argument in call.args)
        for call in _own_calls(statement, "enter_context")
    )


def assess_performance_measurement(
    *, measured: float, baseline: float, threshold_fraction: float
) -> Literal["WITHIN_BASELINE", "REVIEW_REQUIRED"]:
    if baseline <= 0 or measured < 0 or not 0 < threshold_fraction < 1:
        raise ValueError("performance comparison inputs are invalid")
    relative_delta = abs(measured - baseline) / baseline
    return "REVIEW_REQUIRED" if relative_delta > threshold_fraction else "WITHIN_BASELINE"


_DISCOVERED_IMPORTS: dict[tuple[tuple[str, ...], bytes], tuple[str, ...]] = {}
"""This process's answers by module place and content digest: the structural tests ask for the
same unchanged tree in test after test.
Never shared across processes or written to disk. It answers ahead of ``kept_fact``'s version
check, for this process's lifetime, whose loaded code a run never replaces; a test that swaps
``_imports_of`` in place clears it first."""


SCAN_CACHE_ENV = "ALPHALATTICE_SCAN_CACHE_DIR"
"""The directory a gate names for scans' per-file facts; absent, every fact is computed."""
SCAN_CACHE_BYTES = 64 * 1024 * 1024
"""The most one scan's store may hold on disk, its only file: a write that would pass it
rewrites the store from the run's own facts, or drops it when even those would not fit."""


def file_text(data: bytes, errors: str = "strict") -> str:
    """``data`` as ``Path.read_text(encoding="utf-8", errors=errors)`` would read it."""
    return data.decode("utf-8", errors).replace("\r\n", "\n").replace("\r", "\n")


class FileFacts:
    """Facts a whole-tree scan derives from one file's bytes alone, kept across processes.

    A fact is computed from the bytes this lookup reads and kept under their SHA-256, with a
    seal over the scan's code version, the fact's part, that key and the value; a row whose
    seal, JSON or shape (``valid``) does not hold is a miss and is recomputed, never served.
    The seal detects torn, stale, foreign or damaged rows; like the gate's mypy cache, the store
    trusts its writer, the line that owns the directory, and authenticates no other.
    The store is one SQLite file per scan, whatever its code version, in the directory the gate
    names (``SCAN_CACHE_ENV``), with no journal and no temporary file, so that one file is all it
    holds on disk and SQLite's own lock orders every writer, of any version. Another version's
    rows fail their seal and are misses, which this run's rows replace in place. Opening it
    drops a file past the cap, and keeps the store off when another process holds that file.
    Closing it writes the run's new facts only when the scan returned and the file stays within
    ``SCAN_CACHE_BYTES``; otherwise it rewrites the store in place from the facts this run used,
    or empties it when they alone would not fit. The byte cap is the store's only bound, so a
    small run, such as a test's synthetic tree or a scan whose code a test patched, never evicts
    a whole tree's facts; other versions' rows stay misses until the cap rewrites them away. An
    unreadable store is rebuilt; with no directory named, nothing is kept and every fact is
    computed.
    """

    def __init__(self, root: Path, scan: str, version: str) -> None:
        self.root, self.scan, self.version, self.hits, self.misses = root, scan, version, 0, 0
        self.state = "off"
        self._pending: dict[tuple[str, str], str] = {}
        self._used: dict[tuple[str, str], str] = {}
        self._db: sqlite3.Connection | None = None
        self._store: Path | None = None
        directory = os.environ.get(SCAN_CACHE_ENV)
        if not directory:
            return
        self._store = store = Path(directory) / f"{scan}.sqlite"  # one name: no second file
        try:
            store.parent.mkdir(parents=True, exist_ok=True)
            if store.exists() and store.stat().st_size > SCAN_CACHE_BYTES:
                store.unlink()
        except OSError as error:  # past the cap and held by another process: never read or grown
            self.state = f"off (an oversized store is held: {type(error).__name__})"
            return
        try:
            self._db = self._open(store)
            self.state = "persistent"
        except (OSError, sqlite3.DatabaseError):
            try:  # unreadable or damaged: a rebuilt store, never a read of the old one
                store.unlink(missing_ok=True)
                self._db, self.state = self._open(store), "rebuilt"
            except (OSError, sqlite3.DatabaseError) as error:
                self._db, self.state = None, f"off ({type(error).__name__})"

    @staticmethod
    def _open(store: Path) -> sqlite3.Connection:
        db = sqlite3.connect(store, isolation_level=None)  # transactions are close's, explicit
        try:
            # one file and nothing beside it: a torn write is a damaged or unsealed row, rebuilt
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("PRAGMA temp_store=MEMORY")
            db.execute(
                "CREATE TABLE IF NOT EXISTS facts "
                "(part TEXT, key TEXT, value TEXT, seal TEXT, PRIMARY KEY (part, key))"
            )
            db.execute("SELECT count(*) FROM facts").fetchone()  # a damaged file refuses here
        except sqlite3.DatabaseError:
            db.close()  # Windows cannot remove a file this process still holds open
            raise
        return db

    def _seal(self, part: str, key: str, value: str) -> str:
        return hashlib.sha256(f"{self.version}\0{part}\0{key}\0{value}".encode()).hexdigest()

    def get(
        self,
        part: str,
        path: Path,
        compute: Callable[[bytes], Any],
        valid: Callable[[Any], bool],
    ) -> Any:
        """``compute`` of the bytes read now at ``path``, as JSON gives it back; from the store
        when a sealed row of the right shape holds it for exactly those bytes."""
        return self.of(part, path.read_bytes(), compute, valid)

    def of(
        self,
        part: str,
        data: bytes,
        compute: Callable[[bytes], Any],
        valid: Callable[[Any], bool],
    ) -> Any:
        """``get`` for bytes already in hand, such as a Git blob a scan read from a tree."""
        if self._db is None:
            self.misses += 1
            return json.loads(json.dumps(compute(data)))
        key = hashlib.sha256(data).hexdigest()
        try:
            row = self._db.execute(
                "SELECT value, seal FROM facts WHERE part = ? AND key = ?", (part, key)
            ).fetchone()
            if row is not None and row[1] == self._seal(part, key, row[0]):
                fact = json.loads(row[0])
                if valid(fact):
                    self.hits += 1
                    self._used[(part, key)] = row[0]
                    return fact
        except (sqlite3.DatabaseError, TypeError, ValueError):
            pass  # an unreadable entry is a miss, recomputed below
        value = json.dumps(compute(data))
        self.misses += 1
        self._pending[(part, key)] = self._used[(part, key)] = value
        return json.loads(value)

    def retire_predecessors(self, prefix: str) -> int | None:
        """Once per store, remove older fact kinds under SQLite's writer lock."""
        if self._db is None:
            return None
        db = self._db
        try:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("BEGIN IMMEDIATE")
            count = None
            if db.execute("PRAGMA user_version").fetchone()[0] < 1:
                count = db.execute(
                    "DELETE FROM facts WHERE part IS NULL OR substr(part, 1, ?) != ?",
                    (len(prefix), prefix),
                ).rowcount
                db.execute("PRAGMA user_version=1")
            db.execute("COMMIT")
            return count
        except sqlite3.DatabaseError as error:
            with contextlib.suppress(sqlite3.DatabaseError):
                db.execute("ROLLBACK")
            raise ValueError("public scan cache predecessor retirement failed") from error

    def _bytes(self, rows: dict[tuple[str, str], str]) -> int:
        """Twice the rows' text, a bound on the pages SQLite gives them."""
        return 2 * sum(
            len(part) + len(key) + len(value) + 96 for (part, key), value in rows.items()
        )

    def _write(self, rows: dict[tuple[str, str], str]) -> None:
        assert self._db is not None
        self._db.executemany(
            "INSERT OR REPLACE INTO facts VALUES (?, ?, ?, ?)",
            [
                (part, key, value, self._seal(part, key, value))
                for (part, key), value in rows.items()
            ],
        )

    def close(self, *, returned: bool) -> str:
        """Write this run's facts if the scan ``returned``, within the store's byte cap, and say
        how it went. The measure and the write are one immediate transaction, SQLite's own write
        lock, so a concurrent writer of the same store waits for it rather than adding to a size
        it did not see; making room deletes rows inside the file, whose freed pages the next rows
        reuse, so the file never grows past what the cap admitted."""
        if self._db is None or self._store is None:
            return f"{self.scan} hits {self.hits} misses {self.misses} {self.state}"
        store = self._store
        try:
            if returned and self._pending:
                self._db.execute("BEGIN IMMEDIATE")
                grown = store.stat().st_size + self._bytes(self._pending)
                if grown <= SCAN_CACHE_BYTES:
                    self._write(self._pending)
                elif self._bytes(self._used) <= SCAN_CACHE_BYTES:
                    self._db.execute("DELETE FROM facts")
                    self._write(self._used)
                    self.state = "rewritten (bounded)"
                else:
                    self._db.execute("DELETE FROM facts")
                    self.state = "emptied (its facts pass the byte cap)"
                self._db.execute("COMMIT")
        except (OSError, sqlite3.DatabaseError) as error:
            self.state = f"unwritten ({type(error).__name__})"
        finally:
            if self._db is not None:
                self._db.close()
                self._db = None
        try:  # the estimate's backstop: a file past the cap is never kept
            if store.exists() and store.stat().st_size > SCAN_CACHE_BYTES:
                store.unlink()
                self.state = "dropped (over its byte cap)"
        except OSError as error:
            self.state = f"over its byte cap ({type(error).__name__})"
        return f"{self.scan} hits {self.hits} misses {self.misses} {self.state}"


def _plain(value: Any) -> str:
    """A constant as the same text in every process (sets sorted, patterns by their source);
    any other object by its type alone."""
    if value is None or isinstance(value, bool | int | float | str | bytes):
        return repr(value)
    if isinstance(value, re.Pattern):
        return repr((value.pattern, value.flags))
    if isinstance(value, tuple | list):
        return f"({','.join(map(_plain, value))})"
    if isinstance(value, set | frozenset):
        return f"{{{','.join(sorted(map(_plain, value)))}}}"
    if isinstance(value, dict):
        return f"{{{','.join(sorted(_plain(k) + ':' + _plain(v) for k, v in value.items()))}}}"
    return type(value).__qualname__


_IDENTITIES: dict[int, tuple[ModuleType, tuple[CodeType, ...], str]] = {}
"""Each module object's identity with the code it was taken from, held so no id is reused: kept
while those same code objects are its functions, so a rebound value is state, and taken anew
when they change, as ``importlib.reload`` re-executes every definition in the same object."""


def loaded_identity(*modules: ModuleType) -> str:
    """A digest of these modules as loaded in this process, never of their files on disk: each
    function's code (its bytecode, names, constants and nested code, through decorators and
    class members) and each immutable constant, taken anew whenever its functions' code
    changes (a new load or an in-place reload), so a file replaced after import cannot lend its
    version to facts the code already loaded derives, and a module's mutable state (a memo, a
    last result) is not its identity."""
    return hashlib.sha256("".join(map(_module_identity, modules)).encode()).hexdigest()[:16]


def _module_identity(module: ModuleType) -> str:
    codes = tuple(
        item.__code__
        for item in vars(module).values()
        if isinstance(item, FunctionType) and item.__module__ == module.__name__
    )
    known = _IDENTITIES.get(id(module))
    if (
        known is not None
        and codes
        and known[0] is module
        and len(known[1]) == len(codes)
        and all(old is new for old, new in zip(known[1], codes, strict=True))
    ):
        return known[2]
    digest = hashlib.sha256()

    def code(value: CodeType) -> None:
        digest.update(value.co_code + repr((value.co_names, value.co_varnames)).encode())
        for constant in value.co_consts:
            code(constant) if isinstance(constant, CodeType) else digest.update(
                _plain(constant).encode()
            )

    def add(item: Any, member: bool) -> None:
        item = getattr(item, "__func__", None) or getattr(item, "fget", None) or item
        while item is not None and (isinstance(item, FunctionType) or hasattr(item, "__wrapped__")):
            if isinstance(item, FunctionType):
                code(item.__code__)
                digest.update(_plain(item.__defaults__).encode())
            item = getattr(item, "__wrapped__", None)
        if isinstance(item, type) and not member:  # a dunder member only as code or fields: a
            for name, value in sorted(vars(item).items()):  # model's generated schema holds ids
                if not name.startswith("__") or name == "__annotations__" or callable(value):
                    digest.update(name.encode())
                    add(value, True)
        elif isinstance(item, str | bytes | int | float | tuple | frozenset | re.Pattern):
            digest.update(_plain(item).encode())

    for name, item in sorted(vars(module).items()):  # a definition is this module's only where
        defined = isinstance(item, FunctionType | type)  # written; a value counts by itself, so a
        own = not defined or item.__module__ == module.__name__  # pattern's "re" is not foreign
        if own and not name.startswith("__") and not isinstance(item, ModuleType):
            digest.update(name.encode())
            add(item, False)
    _IDENTITIES[id(module)] = module, codes, digest.hexdigest()
    return digest.hexdigest()


@contextlib.contextmanager
def kept_facts(scan: str, *derive: ModuleType) -> Iterator[FileFacts]:
    """``FileFacts`` for one run of ``scan``, versioned by the loaded code that derives its facts
    (these modules and this one, ``loaded_identity``), so a change to how a fact is derived is
    a miss even when a file changes under a running scan; written only when the body returns."""
    version = loaded_identity(*derive, sys.modules[__name__])
    facts = FileFacts(Path.cwd(), scan, version)
    _RUNNING.append((facts, {*derive, sys.modules[__name__]}))
    returned = False
    try:
        yield facts
        returned = True
    finally:
        _RUNNING.pop()
        facts.close(returned=returned)


_RUNNING: list[tuple[FileFacts, set[ModuleType]]] = []
"""The stores of the scans running in this process, innermost last, each with the modules whose
loaded code versions it."""


def kept_fact(
    module: ModuleType | None,
    part: str,
    data: bytes,
    compute: Callable[[bytes], Any],
    valid: Callable[[Any], bool],
) -> Any:
    """``compute(data)`` as JSON gives it back, from the innermost running scan's store when that
    store is versioned by ``module``, the code that computes it; computed now otherwise, so a
    fact is never kept under a version that does not cover its own code (``module`` None: a
    module loaded under a name it is not registered by, which keeps nothing)."""
    for facts, versioned in reversed(_RUNNING):
        if module is not None and module in versioned:
            return facts.of(part, data, compute, valid)
    return json.loads(json.dumps(compute(data)))


def discover_python_imports(path: Path, *, source_root: Path) -> tuple[str, ...]:
    """Return normalized absolute imports for one source-root Python module.

    The answer is a function of the module's place under the source root and its bytes alone,
    so it is kept by those and a changed file misses. A read, decode, parse or escape error is
    raised every time, never kept."""

    relative = path.relative_to(source_root).with_suffix("")
    module_parts = list(relative.parts)
    package_parts = module_parts if module_parts[-1] == "__init__" else module_parts[:-1]
    if package_parts and package_parts[-1] == "__init__":
        package_parts.pop()
    data = path.read_bytes()
    key = (tuple(module_parts), hashlib.sha256(data).digest())
    if (known := _DISCOVERED_IMPORTS.get(key)) is not None:
        return known
    found = tuple(
        kept_fact(
            sys.modules[__name__],
            "imports " + "/".join(module_parts),
            data,
            lambda bytes_: _imports_of(bytes_, path, package_parts),
            lambda value: isinstance(value, list) and all(isinstance(item, str) for item in value),
        )
    )
    _DISCOVERED_IMPORTS[key] = found
    return found


def _imports_of(data: bytes, path: Path, package_parts: list[str]) -> list[str]:
    """One module's normalized absolute imports from its bytes and its package's place."""
    imports: set[str] = set()
    text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")  # as read_text decodes
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(_logical_module_name(alias.name) for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            retained = len(package_parts) - (node.level - 1)
            if retained < 0:
                raise ValueError(f"relative import escapes source root: {path}:{node.lineno}")
            base_parts = [*package_parts[:retained]]
            if node.module:
                base_parts.extend(node.module.split("."))
            base = _logical_module_name(".".join(base_parts))
        else:
            base = _logical_module_name(node.module or "")
        if base:
            imports.add(base)
        imports.update(
            f"{base}.{alias.name}" if base else alias.name
            for alias in node.names
            if alias.name != "*"
        )
    return sorted(imports)


def package_root_file_violations(
    package_roots: Mapping[Path, tuple[str, ...]],
) -> tuple[str, ...]:
    """Require package roots to expose only an explicit stable file surface."""

    violations: list[str] = []
    for package_root, allowed_names in package_roots.items():
        allowed = set(allowed_names)
        for path in package_root.glob("*.py"):
            if path.name not in allowed:
                violations.append(f"{package_root.name}: unexpected root module {path.name}")
    return tuple(sorted(violations))


def empty_python_package_directories(root: Path) -> tuple[str, ...]:
    """Find tracked-style package directories containing no Python implementation."""

    empty: list[str] = []
    for init_path in root.rglob("__init__.py"):
        package = init_path.parent
        if not any(path.name != "__init__.py" for path in package.glob("*.py")) and not any(
            child.is_dir() and (child / "__init__.py").is_file() for child in package.iterdir()
        ):
            empty.append(package.relative_to(root).as_posix())
    return tuple(sorted(empty))


def discover_package_dependencies(
    paths: tuple[Path, ...], *, source_root: Path
) -> tuple[tuple[str, str], ...]:
    """Return top-level package edges for tracked Python sources."""

    modules_by_path = dict(discover_python_modules(paths, source_root=source_root))
    packages = {module.split(".", maxsplit=1)[0] for module in modules_by_path.values()}
    edges: set[tuple[str, str]] = set()
    with kept_facts("package-imports", sys.modules[__name__]):  # each module's, by its bytes
        for path, module in modules_by_path.items():
            owner = module.split(".", maxsplit=1)[0]
            if owner not in packages:
                continue
            for imported in discover_python_imports(path, source_root=source_root):
                target = imported.split(".", maxsplit=1)[0]
                if target in packages and target != owner:
                    edges.add((owner, target))
    return tuple(sorted(edges))


def package_architecture_violations(
    edges: tuple[tuple[str, str], ...],
    *,
    registered_packages: tuple[str, ...],
    registry: PackageArchitectureRegistry,
) -> tuple[str, ...]:
    """Enforce complete registration and the declared package dependency levels."""

    configured = registry.by_package
    violations: list[str] = []
    observed = set(registered_packages)
    declared = set(configured)
    for package_id in sorted(observed - declared):
        violations.append(f"unregistered package: {package_id}")
    for package_id in sorted(declared - observed):
        violations.append(f"registered package is absent: {package_id}")
    for owner, target in edges:
        if owner not in configured or target not in configured:
            continue
        owner_contract = configured[owner]
        target_contract = configured[target]
        if target_contract.dependency_level > owner_contract.dependency_level:
            violations.append(
                f"{owner} -> {target}: dependency points to higher level "
                f"{target_contract.dependency_level}"
            )
        elif (
            target_contract.dependency_level == owner_contract.dependency_level
            and target not in owner_contract.allowed_peer_dependencies
        ):
            violations.append(f"{owner} -> {target}: undeclared same-level dependency")
    return tuple(sorted(violations))


def _import_scope(node: ast.AST, parents: Mapping[ast.AST, ast.AST]) -> tuple[str, bool]:
    ancestors = []
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ancestors.append(parent)
        parent = parents.get(parent)
    return ".".join(n.name for n in reversed(ancestors)) or "<module>", any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ancestors
    )


def _imported_module(node: ast.AST, target: str, modules: set[str]) -> str:
    if isinstance(node, ast.Import | ast.Call) or target in modules:
        return target
    return target.rpartition(".")[0]


def import_contract_violations(
    sources: Mapping[str, str],
    *,
    registry: PackageArchitectureRegistry,
    modules: set[str] | None = None,
) -> tuple[str, ...]:
    """Enforce declared lazy modules and exact cross-owner private equality fences."""
    modules = {n.removesuffix(".__init__") for n in (sources if modules is None else modules)}
    roots = sorted(registry.packages, key=lambda r: -len(r.physical_namespace_target))

    def owner(name: str) -> str:
        for entry in roots:
            prefix = entry.physical_namespace_target
            if name == prefix or name.startswith(prefix + "."):
                return entry.package_id
        return ""

    out: list[str] = []
    retained = Counter(registry.legacy_lazy_imports)

    this = sys.modules[__name__]

    seen: set[tuple[str, str]] = set()  # this module's private imports, per module
    eager: set[str] = set()  # this module's top-level imports, per module

    def imported_module(kind: str, target: str) -> str:  # _imported_module, from a kept site
        return target if kind != "from" or target in modules else target.rpartition(".")[0]

    def inspect_module(qualified: str, source: str) -> None:
        module = qualified.removesuffix(".__init__")
        sites = kept_fact(
            this,
            "import-sites " + qualified,
            source.encode("utf-8"),
            partial(_import_sites, qualified=qualified),
            lambda value: isinstance(value, list) and all(len(row) == 8 for row in value),
        )
        seen.clear()
        eager.clear()
        eager.update(imported_module(site[1], t) for site in sites if site[6] for t in site[7])
        for line, kind, scope, deferred, lazy, text, _top, targets in sites:
            located = f"{module}:{line}"
            if deferred and not lazy:
                site = f"src/{qualified.replace('.', '/')}.py::{scope}::{text}"
                if retained[site] < 1:
                    out.append(f"{located}: unexplained-function-import {text}")
                retained[site] -= 1
            for target in targets:
                check_target(
                    module, located, scope, deferred, imported_module(kind, target), target
                )

    def check_target(
        module: str, located: str, scope: str, deferred: bool, imported: str, target: str
    ) -> None:
        if deferred and imported in eager:
            out.append(f"{located}: repeated-function-import {imported}")
        if not target.startswith(("alphalattice.", "devtools.")):
            return
        fence = registry.private_import_fences.get(module + "::" + scope, {})
        private = any(part.startswith("_") for part in target.split(".")) and owner(
            module
        ) != owner(target)
        if (
            deferred
            and imported not in registry.deferred_imports
            and not (private and target in fence)
        ):
            out.append(f"{located}: unlisted-function-import {imported}")
        if private:
            key = (scope, target)
            if target not in fence or key in seen:
                out.append(f"{located}: private-import {target}")
            seen.add(key)

    with kept_facts("import-sites", this):  # each module's import sites, kept by its bytes
        for qualified, source in sources.items():
            if qualified.startswith(("alphalattice.", "devtools.")):
                inspect_module(qualified, source)
    out.extend(
        f"stale-legacy-import {k}"
        for k, n in retained.items()
        if n > 0 and k.split("::", 1)[0][4:-3].replace("/", ".") in sources
    )
    return tuple(sorted(out))


def _import_sites(data: bytes, qualified: str) -> list[list[Any]]:
    """One module's import sites from its bytes, in walk order: [line, kind ("from" or "other"),
    scope, deferred, lazy-marked, text, top-level, targets]. The rules are the caller's."""
    module = qualified.removesuffix(".__init__")
    source = data.decode("utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    nodes = list(ast.walk(tree))  # walked once, read twice
    parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
    top = {id(statement) for statement in tree.body}
    constants: dict[str, object] = {}
    loaders = {"__import__", "importlib.import_module"}
    for item in nodes:
        match item:
            case (
                ast.Assign(targets=[ast.Name(id=name)], value=ast.Constant(value=value))
                | ast.AnnAssign(target=ast.Name(id=name), value=ast.Constant(value=value))
            ) if id(item) in top:
                constants[name] = value
            case ast.Import(names=names):
                loaders.update(
                    (a.asname or a.name) + ".import_module" for a in names if a.name == "importlib"
                )
            case ast.ImportFrom(module="importlib", names=names):
                loaders.update(a.asname or a.name for a in names if a.name == "import_module")
    package = module if qualified.endswith(".__init__") else module.rpartition(".")[0]
    constants["__package__"] = package

    def literal(node: ast.AST | None) -> object:
        return getattr(node, "value", constants.get(getattr(node, "id", "")))

    def targets_of(node: ast.AST) -> list[str]:
        match node:
            case ast.Import(names=names):
                return [a.name for a in names]
            case ast.ImportFrom(module=module, level=level, names=names):
                base = resolve_name("." * level + (module or ""), package) if level else module
                return [f"{base}.{a.name}" for a in names]
            case ast.Call(func=func) if ast.unparse(func) in loaders:
                args = {k.arg: k.value for k in node.keywords}
                if node.args:
                    args.setdefault("name", node.args[0])
                if len(node.args) > 1:
                    args.setdefault("package", node.args[1])
                name = literal(args.get("name"))
                context = literal(args["package"]) if "package" in args else package
                if isinstance(name, str) and (not name.startswith(".") or isinstance(context, str)):
                    return [resolve_name(name, str(context)) if name.startswith(".") else name]
        return []

    sites: list[list[Any]] = []
    for node in nodes:
        if not (targets := targets_of(node)):
            continue
        scope, deferred = _import_scope(node, parents)
        line = getattr(node, "lineno", 0)
        lazy = bool(re.search(r"#\s*lazy:\s*\S", lines[getattr(node, "lineno", 1) - 1]))
        kind = "from" if isinstance(node, ast.ImportFrom) else "other"
        sites.append(
            [line, kind, scope, deferred, lazy, ast.unparse(node), id(node) in top, targets]
        )
    return sites


def physical_namespace_violations(
    *, source_root: Path, registry: PackageArchitectureRegistry
) -> tuple[str, ...]:
    """Require every registered owner to occupy its declared physical namespace."""

    violations: list[str] = []
    all_sources: list[Path] = []
    for contract in registry.packages:
        target = source_root / Path(*contract.physical_namespace_target.split("."))
        if not target.is_dir():
            violations.append(
                f"{contract.package_id}: physical namespace is missing: "
                f"{contract.physical_namespace_target}"
            )
            continue
        sources = tuple(sorted(target.rglob("*.py")))
        all_sources.extend(sources)
        if not sources:
            violations.append(
                f"{contract.package_id}: physical namespace has no tracked-style Python source"
            )
            continue
        observed = {
            module.split(".", maxsplit=1)[0]
            for _, module in discover_python_modules(sources, source_root=source_root)
        }
        if observed != {contract.package_id}:
            violations.append(
                f"{contract.package_id}: physical namespace resolves to {sorted(observed)}"
            )
    payloads = {
        ".".join(p.relative_to(source_root).with_suffix("").parts): p.read_text(encoding="utf-8")
        for p in all_sources
    }
    return tuple(sorted([*violations, *import_contract_violations(payloads, registry=registry)]))


def discover_python_modules(
    paths: tuple[Path, ...], *, source_root: Path
) -> tuple[tuple[Path, str], ...]:
    """Return stable path-to-module identities for tracked Python sources."""

    modules_by_path: dict[Path, str] = {}
    for path in paths:
        module = _logical_module_for_path(path, source_root=source_root)
        if module:
            modules_by_path[path] = module
    return tuple(sorted(modules_by_path.items(), key=lambda item: item[1]))


def _logical_module_for_path(path: Path, *, source_root: Path) -> str:
    relative = path.relative_to(source_root).with_suffix("")
    parts = relative.parts[:-1] if relative.parts[-1] == "__init__" else relative.parts
    return _logical_module_name(".".join(parts))


def _logical_module_name(module: str) -> str:
    parts = module.split(".") if module else []
    if parts == ["devtools"]:
        return ""
    if parts == ["alphalattice"] or (
        len(parts) == 2 and parts[0] == "alphalattice" and parts[1] in _PHYSICAL_NAMESPACE_GROUPS
    ):
        return ""
    if len(parts) >= 3 and parts[0] == "alphalattice" and parts[1] in (_PHYSICAL_NAMESPACE_GROUPS):
        parts = parts[2:]
    elif len(parts) >= 2 and parts[0] == "devtools" and parts[1] in _DEVTOOLS_PACKAGES:
        parts = [_DEVTOOLS_PACKAGES[parts[1]], *parts[2:]]
    return ".".join(parts)


_DEVTOOLS_PACKAGES = {
    "architecture": "playpen_governance",
    "evaluation": "evaluation",
    "pipeline": "pipeline",
    "testing": "testing",
}
"""Each registered devtools package's id; an unregistered one keeps its dotted name."""


def discover_module_dependencies(
    paths: tuple[Path, ...], *, source_root: Path
) -> tuple[tuple[str, str], ...]:
    """Return import edges whose owners and targets are tracked source modules."""

    modules_by_path = dict(discover_python_modules(paths, source_root=source_root))
    modules = set(modules_by_path.values())
    edges: set[tuple[str, str]] = set()
    for path, owner in modules_by_path.items():
        for imported in discover_python_imports(path, source_root=source_root):
            if imported in modules and imported != owner:
                edges.add((owner, imported))
    return tuple(sorted(edges))


def dependency_closure(
    edges: tuple[tuple[str, str], ...], *, seeds: tuple[str, ...]
) -> tuple[str, ...]:
    """Return the stable transitive closure for explicit dependency seeds."""

    adjacency: dict[str, set[str]] = {}
    for owner, target in edges:
        adjacency.setdefault(owner, set()).add(target)
        adjacency.setdefault(target, set())
    for seed in seeds:
        adjacency.setdefault(seed, set())
    reachable = set(seeds)
    pending = list(reversed(sorted(seeds)))
    while pending:
        owner = pending.pop()
        for target in sorted(adjacency[owner], reverse=True):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    return tuple(sorted(reachable))


def mutual_package_dependencies(
    edges: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Project direct bidirectional package edges into stable unordered pairs."""

    edge_set = set(edges)
    return tuple(
        sorted(
            (owner, target)
            for owner, target in edge_set
            if owner < target and (target, owner) in edge_set
        )
    )


def package_dependency_components(
    edges: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, ...], ...]:
    """Return non-trivial strongly connected package components."""

    adjacency: dict[str, set[str]] = {}
    for owner, target in edges:
        adjacency.setdefault(owner, set()).add(target)
        adjacency.setdefault(target, set())

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in sorted(adjacency[node]):
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] != indices[node]:
            return
        component: list[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        if len(component) > 1:
            components.append(tuple(sorted(component)))

    for node in sorted(adjacency):
        if node not in indices:
            visit(node)
    return tuple(sorted(components))


def content_store_absence_violations(sources: Mapping[str, str]) -> tuple[str, ...]:
    """Find content-store readers that classify corruption without accounting for absence.

    Check comparisons, error translators and exception handlers around loader calls. Local
    helper calls are followed; a separately propagated store exception keeps its distinction.
    This is a syntax pin, not a proof of dynamic or cross-module dispatch: reader regressions
    hold those boundaries. New comparisons and blanket tamper translations enter the same pin.

    Args:
        sources: Source text keyed by its repository-relative file path.

    Returns:
        Located functions that compare tampering or translate a store read to tampering without
        an explicit missing branch, locally or in a reached in-module helper.
    """
    missing = "content_store.artifact_missing"
    tampered = "content_store.artifact_tampered"
    failures: set[str] = set()
    for path, source in sources.items():
        tree = ast.parse(source)
        function_nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        for function in function_nodes:
            functions.setdefault(function.name, []).append(function)
        store_module = any(  # once per file: it was walked again for every try block
            isinstance(n, ast.ImportFrom)
            and n.module is not None
            and n.module.endswith("workspace_runtime.content_store")
            for n in ast.walk(tree)
        )

        def reached(
            nodes: list[ast.AST],
            functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = functions,
        ) -> list[ast.AST]:
            result = list(nodes)
            seen: set[str] = set()
            for node in result:
                for call in ast.walk(node):
                    if not isinstance(call, ast.Call):
                        continue
                    name = (
                        call.func.id
                        if isinstance(call.func, ast.Name)
                        else call.func.attr
                        if isinstance(call.func, ast.Attribute)
                        else None
                    )
                    if name is not None and name in functions and name not in seen:
                        seen.add(name)
                        result.extend(functions[name])
            return result

        def strings(nodes: list[ast.AST]) -> set[str]:
            return {
                child.value
                for node in reached(nodes)
                for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            }

        for node in function_nodes:
            located = f"{path}:{node.lineno}:{node.name}"
            comparisons = [n for n in ast.walk(node) if isinstance(n, ast.Compare)]
            translates = any(
                isinstance(n, ast.Name) and n.id == "ContentAddressedStoreError"
                for n in ast.walk(node.args)
            ) and any(value.endswith("artifact_tampered") for value in strings([node]))
            if (
                any(
                    isinstance(c, ast.Constant) and c.value == tampered
                    for n in comparisons
                    for c in ast.walk(n)
                )
                or translates
            ) and (missing not in strings([node])):
                failures.add(located + ": corruption classification has no missing branch")
            for block in (n for n in ast.walk(node) if isinstance(n, ast.Try)):
                if not store_module:
                    continue
                loads = any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr.startswith("load_")
                    for part in reached(list(block.body))
                    for call in ast.walk(part)
                )
                if not loads:
                    continue
                for handler in block.handlers:
                    # A preceding typed propagation protects a later generic index handler.
                    prior = block.handlers[: block.handlers.index(handler)]
                    if any(
                        isinstance(h.type, ast.Name)
                        and h.type.id in {"ContentAddressedStoreError", "PortfolioPublicationError"}
                        and any(isinstance(n, ast.Raise) and n.exc is None for n in ast.walk(h))
                        for h in prior
                    ):
                        continue
                    values = strings([handler])
                    names = {n.id for n in ast.walk(handler) if isinstance(n, ast.Name)}
                    writes_tamper = any(v.endswith("_tampered") for v in values) or (
                        "tampered" in names
                        and any(v.endswith("_tampered") for v in strings([node]))
                    )
                    if writes_tamper and missing not in values:
                        failures.add(
                            located + ": loader exception becomes tampering without absence"
                        )
    return tuple(sorted(failures))


def evaluation_imports(sources: Mapping[str, str], entry: str) -> list[tuple[str, int, str]]:
    """Evaluation imports past its forms, stdlib, YAML and the `module:name` entry; dynamic ones."""
    (module, _, name), failures = entry.partition(":"), []
    for path, text in sources.items():
        for node in ast.walk(ast.parse(text)):
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.level:
                failures += [(path, node.lineno, "outside package")] if node.level != 1 else []
            elif isinstance(node, ast.ImportFrom) and not (
                node.module == module and all(alias.name == name for alias in node.names)
            ):
                names = [node.module or ""]
            for imported in names:
                if not (
                    imported.split(".")[0] in sys.stdlib_module_names | {"yaml"}
                    or imported.startswith("devtools.evaluation.")
                ):
                    failures.append((path, getattr(node, "lineno", 0), imported))
            if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id in {"__import__", "eval", "exec"})
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
            ):
                failures.append((path, node.lineno, "dynamic import"))
    return failures


__all__ = [
    "ArtifactAuthorityRegistry",
    "GenerationModel",
    "PackageArchitectureRegistry",
    "PackageArchitectureRole",
    "PerformanceBaselines",
    "StorageAuthorityRegistry",
    "StoreSurface",
    "assess_performance_measurement",
    "content_store_absence_violations",
    "dependency_closure",
    "discover_module_dependencies",
    "discover_package_dependencies",
    "discover_python_imports",
    "discover_python_modules",
    "discover_repository_surface",
    "discover_store_surface",
    "duckdb_access_mode_violations",
    "empty_python_package_directories",
    "evaluation_imports",
    "import_contract_violations",
    "load_artifact_authority_registry",
    "load_package_architecture_registry",
    "load_performance_baselines",
    "load_storage_authority_registry",
    "load_store_surface",
    "mutual_package_dependencies",
    "package_architecture_violations",
    "package_dependency_components",
    "package_root_file_violations",
    "physical_namespace_violations",
    "storage_registry_violations",
    "store_surface_violations",
]
