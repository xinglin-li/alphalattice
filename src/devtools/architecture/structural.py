"""Deterministic structural checks for the ignored playpen closure."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal

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
    holds, a table's schema written by another package than its data (V171)."""
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
    """Where the prefixes lie (V205): `artifacts` (the workspace's `artifacts/`, or a research
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
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names.update(node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        names.update(
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        )
    return frozenset(names)


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


class PackageArchitectureRegistry(_Contract):
    schema_id: Literal["alphalattice.playpen.package-architecture"] = Field(alias="schema")
    packages: tuple[PackageArchitecture, ...] = Field(min_length=1)

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
    return PackageArchitectureRegistry.model_validate(_load_json(path))  # type: ignore[no-any-return]


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
    that module (V264).
    """

    found: set[TableWrite] = set()
    for path in sources:
        module = _logical_module_for_path(path, source_root=source_root)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        texts = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ]
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
        package = module.split(".", 1)[0]
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
                    TableWrite(
                        table=element.value,
                        kind="data",
                        writer=f"{module}:WRITTEN_TABLES",
                        package=package,
                    )
                    for element in statement.value.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, str)
                )
        for node in texts:
            text = str(node.value)
            inner = [value for value in classes if value[0] <= node.lineno <= value[1]]
            writer = (
                min(inner, key=lambda value: value[1] - value[0])[2]
                if inner
                else next(
                    (name for start, end, name in functions if start <= node.lineno <= end),
                    "<module>",
                )
            )
            statements = [
                ("schema", match.group(1))
                for match in _SCHEMA_WRITE.finditer(text)
                if not _TEMPORARY_TABLE.match(text, match.start())
            ] + [("data", match.group(1) or match.group(2)) for match in _DATA_WRITE.finditer(text)]
            for kind, name in statements:
                table = name.rsplit(".", 1)[-1]
                if table.upper() in _SQL_WORDS or table.lower() in temporary:
                    continue
                found.add(
                    TableWrite(
                        table=table,
                        kind=kind,
                        writer=f"{module}:{writer}",
                        package=package,
                    )
                )
    return tuple(sorted(found, key=lambda value: (value.table, value.kind, value.writer)))


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
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not _is_duckdb_connect(node):
                continue
            assert isinstance(node, ast.Call)
            database = node.args[0] if node.args else None
            if isinstance(database, ast.Constant) and database.value == ":memory:":
                continue
            access_mode = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "read_only"),
                None,
            )
            if access_mode is None:
                violations.append(f"{path.as_posix()}:{node.lineno}: missing read_only=...")
    return tuple(violations)


def _is_duckdb_connect(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "duckdb"
        and node.func.attr == "connect"
    )


def gate_after_instance_violations(paths: tuple[Path, ...]) -> tuple[str, ...]:
    """Refuse a unit that asks for a write gate while it holds a store's instance (V81).

    Every writer takes the gate (`hold()`) first and the store's retained instance
    (`retain(...)`) second; a unit that holds the instance and then asks for the gate waits on
    a writer holding the gate while it waits for that instance. Read in the source: in one
    `with`, no `hold()` follows a `retain(...)`; inside a `retain(...)` block, or after one
    entered on an exit stack, no `hold()` is asked for. What a called function takes is the
    lock-order recorder's (`tests/workspace_task_runner/lock_order.py`).
    """
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
        violations += [
            f"{path.as_posix()}:{call.lineno}: hold() asked for while a store instance is retained"
            for call in sorted(late, key=lambda call: (call.lineno, call.col_offset))
        ]
    return tuple(violations)


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


def discover_python_imports(path: Path, *, source_root: Path) -> tuple[str, ...]:
    """Return normalized absolute imports for one source-root Python module."""

    relative = path.relative_to(source_root).with_suffix("")
    module_parts = list(relative.parts)
    package_parts = module_parts if module_parts[-1] == "__init__" else module_parts[:-1]
    if package_parts and package_parts[-1] == "__init__":
        package_parts.pop()
    imports: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
    return tuple(sorted(imports))


def retired_import_violations(
    paths: tuple[Path, ...],
    *,
    source_root: Path,
    retired_prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    """Find imports of owners that were retired by an atomic package cutover."""

    violations: list[str] = []
    for path in paths:
        for imported in discover_python_imports(path, source_root=source_root):
            if any(
                imported == prefix or imported.startswith(f"{prefix}.")
                for prefix in retired_prefixes
            ):
                violations.append(
                    f"{path.relative_to(source_root).as_posix()}: retired import {imported}"
                )
    return tuple(sorted(violations))


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


def package_dependency_violations(
    paths: tuple[Path, ...],
    *,
    source_root: Path,
    forbidden_dependencies: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Enforce named low-level-to-high-level dependency prohibitions."""

    violations: list[str] = []
    for path in paths:
        owner = _logical_module_for_path(path, source_root=source_root).split(".", maxsplit=1)[0]
        forbidden = forbidden_dependencies.get(owner, ())
        if not forbidden:
            continue
        for imported in discover_python_imports(path, source_root=source_root):
            if any(imported == prefix or imported.startswith(f"{prefix}.") for prefix in forbidden):
                violations.append(
                    f"{path.relative_to(source_root).as_posix()}: forbidden dependency {imported}"
                )
    return tuple(sorted(violations))


def discover_package_dependencies(
    paths: tuple[Path, ...], *, source_root: Path
) -> tuple[tuple[str, str], ...]:
    """Return top-level package edges for tracked Python sources."""

    modules_by_path = dict(discover_python_modules(paths, source_root=source_root))
    packages = {module.split(".", maxsplit=1)[0] for module in modules_by_path.values()}
    edges: set[tuple[str, str]] = set()
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


def physical_namespace_violations(
    *, source_root: Path, registry: PackageArchitectureRegistry
) -> tuple[str, ...]:
    """Require every registered owner to occupy its declared physical namespace."""

    violations: list[str] = []
    for contract in registry.packages:
        target = source_root / Path(*contract.physical_namespace_target.split("."))
        if not target.is_dir():
            violations.append(
                f"{contract.package_id}: physical namespace is missing: "
                f"{contract.physical_namespace_target}"
            )
            continue
        sources = tuple(sorted(target.rglob("*.py")))
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
    return tuple(sorted(violations))


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
    elif len(parts) >= 2 and parts[:2] == ["devtools", "architecture"]:
        parts = ["playpen_governance", *parts[2:]]
    return ".".join(parts)


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
                loads = any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr.startswith("load_")
                    for part in reached(list(block.body))
                    for call in ast.walk(part)
                )
                store_module = any(
                    isinstance(n, ast.ImportFrom)
                    and n.module is not None
                    and n.module.endswith("workspace_runtime.content_store")
                    for n in ast.walk(tree)
                )
                if not loads or not store_module:
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
    "load_artifact_authority_registry",
    "load_package_architecture_registry",
    "load_performance_baselines",
    "load_storage_authority_registry",
    "load_store_surface",
    "mutual_package_dependencies",
    "package_architecture_violations",
    "package_dependency_components",
    "package_dependency_violations",
    "package_root_file_violations",
    "physical_namespace_violations",
    "retired_import_violations",
    "storage_registry_violations",
    "store_surface_violations",
]
