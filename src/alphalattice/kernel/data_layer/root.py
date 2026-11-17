"""The 0.2.8 data root: its folders, its event log holding the seed registry, its search index.

machine/log.sqlite      the event log and every table projected from it (schema/log.sql), with
                        the seed registry
machine/index.sqlite    the search index (schema/index.sql), its vector table through sqlite-vec
machine/catalog.sqlite  the DuckLake catalog (schema/catalog.sql), made at its first use by the
                        store (`catalog`), using the installed extension wheels when the data
                        extra is present; nothing is ever fetched
data, blobs, staging    DuckLake's Parquet files, content-addressed blobs, compute staging

The DDL in schema/ is the 0.2.8 reference schemas' text, unchanged. Nothing reads a data root
yet: the Host's first append writes `registry.seeded`, the event every seed row names as its
registration.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePath
from typing import Any, Final

import sqlite_vec

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.project import register
from alphalattice.kernel.shared_kernel.environment import package_versions

SCHEMA_VERSION: Final = 4
"""`PRAGMA user_version` of the databases this code creates; it refuses any other. A change to
any schema file takes the next number, so a root an earlier build made is refused by name."""
FOLDERS: Final = ("machine", "data", "blobs", "staging", "extensions")
CATALOG_DEFERRED: Final = "catalog: deferred (ducklake extension absent)"
INDEX_DEFERRED: Final = "index: vector table deferred (extension loading unavailable)"
SPEC_FIELDS: Final = {
    "event_types": ("event_type", ("work", "subject", "required", "text", "durable")),
    "metrics": ("metric", ("unit", "direction", "params")),
    "artifact_types": ("artifact_type", ("storage", "table", "measures", "keep")),
    "tags": ("tag", ("values",)),
    "world_tables": ("world_table", ()),
}
"""Each seed list's definition family and the fields its rows keep as their spec."""
THREADS: Final = min(8, os.cpu_count() or 1)
"""Threads of a DuckDB session over the lake. Its statements are many and small, and each pipeline
fans out to every thread: past eight, a statement pays more in that fan-out than its scans gain."""


@dataclass(frozen=True)
class DataRoot:
    """What one call made now, and the parts no call makes yet, each with why."""

    path: Path
    created: tuple[str, ...]
    deferred: tuple[str, ...]


def schema_text(name: str) -> str:
    """One DDL file of the package: log.sql, catalog.sql, views.sql or index.sql."""
    return resources.files(__package__).joinpath("schema", name).read_text(encoding="utf-8")


def seed_rows(owner: str = "kernel", seq: int = 1) -> list[tuple]:
    """Every seed definition as a row, (family, name, version, then definition_version's columns
    after its version), registered by the event at `seq`. A seed row is version 1 unless it names
    its version and its relation to the version before it."""
    text = resources.files(__package__).joinpath("seed_registry.json").read_text(encoding="utf-8")
    seed = json.loads(text)
    out = [("family", name, {}, {}, words) for name, words in seed["families"].items()]
    for key, (family, fields) in SPEC_FIELDS.items():
        out += [
            (family, row["name"], row, {field: row[field] for field in fields}, row["description"])
            for row in seed[key]
        ]
    tail = (None, None, owner, seq, seq, None)  # code_ref, golden, owner, two seqs, reason
    return [
        (
            *(family, name, row.get("version", 1), "standard", row.get("relation"), None),
            *(json.dumps(spec, sort_keys=True), words, *tail),
        )
        for family, name, row, spec, words in out
    ]


def statements(text: str) -> list[str]:
    """A DDL file's statements without its comments (no literal in them holds `--` or `;`)."""
    bare = "\n".join(line.split("--", 1)[0] for line in text.splitlines())
    return [part.strip() for part in bare.split(";") if part.strip()]


def build_log(connection: sqlite3.Connection) -> None:
    """log.sql's tables in `connection`, with every seed definition registered at seq 1."""
    connection.executescript(schema_text("log.sql"))
    with connection:
        for family, name, version, *fields in seed_rows():
            register(connection, family, name, version, *fields)


def _build_index(connection: sqlite3.Connection) -> None:
    """index.sql whole when sqlite-vec loads; without extension loading (an interpreter built
    without it, or a policy refusing it) every table but the vector table (INDEX_DEFERRED)."""
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        vectors = True
    except (AttributeError, sqlite3.OperationalError):
        vectors = False
    for statement in statements(schema_text("index.sql")):
        if vectors or "USING vec0" not in statement:
            connection.execute(statement)


def _tables(connection: sqlite3.Connection) -> frozenset[str]:
    return frozenset(name for (name,) in connection.execute("SELECT name FROM sqlite_master"))


def _database(path: Path, build) -> tuple[bool, frozenset[str]]:
    """The database at `path` and its tables: built whole under a temporary name and moved into
    place, so a crash leaves none or all of it. An existing one is only read, and refused unless it
    is this schema's version; the read is immutable, so a WAL database gains no -shm or -wal file,
    and it needs no extension, since it reads only the schema."""
    if path.exists():
        uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            found = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = _tables(connection)
        if found != SCHEMA_VERSION:
            raise DataError(
                f"{path} is schema version {found}; this product reads version {SCHEMA_VERSION}",
                code="data_layer.schema_mismatch",
            )
        return False, tables
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.unlink(missing_ok=True)
    with closing(sqlite3.connect(partial, isolation_level=None)) as connection:
        build(connection)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        tables = _tables(connection)
    os.replace(partial, path)
    return True, tables


def load_extensions(connection) -> None:
    """Load engine-matched binaries from the installed wheels, never INSTALL or a repository."""
    import duckdb  # lazy: data is optional; log/index creation works without DuckDB

    version = dict(package_versions(("duckdb",)))["duckdb"]
    for name in ("sqlite_scanner", "ducklake"):
        path = resources.files(f"duckdb_extension_{name}").joinpath(
            "extensions", f"v{version}", f"{name}.duckdb_extension"
        )
        quoted = str(path).replace("\\", "/").replace("'", "''")
        try:
            connection.execute(f"LOAD '{quoted}'").complete()
        except duckdb.Error as error:
            raise DataError(
                f"The installed {name} extension could not load. "
                "Restore the locked data dependencies with uv sync --locked --extra data.",
                code="data_layer.extension_unavailable",
            ) from error


def sql_path(path: PurePath) -> str:
    """`path` for a DuckDB string literal: forward slashes, each quote doubled, and a Windows
    extended-length path (`//?/D:/...`, `//?/UNC/server/...` as forward slashes give them) in its
    ordinary spelling, which DuckDB can open and derive its WAL's name from."""
    text = path.as_posix()
    if text.startswith("//?/UNC/"):
        text = "//" + text[len("//?/UNC/") :]
    elif text.startswith("//?/"):
        text = text[len("//?/") :]
    return text.replace("'", "''")


def duckdb_session(root: Path) -> Any | None:
    """An in-memory DuckDB session with the SQLite scanner and DuckLake loaded from their installed
    wheels, every download off; None without the data extra. The caller closes it."""
    try:
        import duckdb  # lazy: DuckDB is the optional data extra; without it the session defers
    except ImportError:
        return None
    connection = duckdb.connect(
        ":memory:",
        config={
            "extension_directory": str(root / "extensions"),
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
            "threads": THREADS,
        },
    )
    try:
        load_extensions(connection)
    except ModuleNotFoundError:
        connection.close()
        return None
    except BaseException:
        connection.close()
        raise
    return connection


def catalog(root: Path) -> bool | None:
    """Create the SQLite-backed DuckLake catalog from installed wheels at its first use: True when
    made, False when present, None (CATALOG_DEFERRED) without the data extra. A root is made
    without it, since making it takes most of a root's creation and a root that never stores a
    world table never needs it."""
    if importlib.util.find_spec("duckdb") is None:
        return None
    path = root / "machine" / "catalog.sqlite"
    if path.exists():
        return False
    connection = duckdb_session(root)  # in memory: the catalog lives in the attached SQLite file
    if connection is None:
        return None
    try:
        data = sql_path(root / "data")
        connection.execute(
            f"ATTACH 'ducklake:sqlite:{sql_path(path)}' AS lake (DATA_PATH '{data}/')"
        ).complete()
        connection.execute("USE lake").complete()
        for statement in statements(schema_text("catalog.sql")):
            connection.execute(statement).complete()
    finally:
        connection.close()
    return True


def catalog_matches(lake: Any) -> None:
    """Refuse a catalog, attached as `lake`, whose tables are not catalog.sql's: a catalog is made
    once by its schema and, like the root's databases, never migrated or rebuilt in place. A
    table a later step adds (a feature set's) is not catalog.sql's and is not compared."""
    declared = {
        f"{schema}.{table}": re.findall(r"^  (\w+) ", body, re.M)
        for schema, table, body in re.findall(
            r"^CREATE TABLE (\w+)\.(\w+) \((.*?)^\);", schema_text("catalog.sql"), re.M | re.S
        )
    }
    held: dict[str, list[str]] = {}
    for schema, table, column in lake.execute(
        "SELECT schema_name, table_name, column_name FROM duckdb_columns()"
        " WHERE database_name = 'lake' ORDER BY schema_name, table_name, column_index"
    ).fetchall():
        held.setdefault(f"{schema}.{table}", []).append(column)
    moved = sorted(name for name, columns in declared.items() if held.get(name) != columns)
    if moved:
        raise DataError(
            f"the data root's catalog was made by another catalog schema ({', '.join(moved)}"
            " differ); it is never migrated in place: admit its workspaces into a new data root",
            code="data_layer.schema_mismatch",
        )


def create_data_root(path: Path) -> DataRoot:
    """Make the data root at `path`, or find it whole; running it again changes nothing."""
    created = []
    for folder in FOLDERS:
        if not (path / folder).is_dir():
            (path / folder).mkdir(parents=True, exist_ok=True)
            created.append(folder)
    tables: frozenset[str] = frozenset()
    for name, build in (("log.sqlite", build_log), ("index.sqlite", _build_index)):
        made, tables = _database(path / "machine" / name, build)
        if made:
            created.append(f"machine/{name}")
    return DataRoot(
        path=path,
        created=tuple(created),
        deferred=(INDEX_DEFERRED,) * ("passage_vec" not in tables),
    )
