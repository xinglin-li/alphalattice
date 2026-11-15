"""The 0.2.8 data root: its folders, its event log holding the seed registry, its search index.

machine/log.sqlite      the event log and every table projected from it (schema/log.sql), with
                        the seed registry
machine/index.sqlite    the search index (schema/index.sql), its vector table through sqlite-vec
machine/catalog.sqlite  the DuckLake catalog (schema/catalog.sql), made only when DuckLake loads
                        from extensions/, the product's own; nothing is ever fetched
data, blobs, staging    DuckLake's Parquet files, content-addressed blobs, compute staging

The DDL in schema/ is the 0.2.8 reference schemas' text, unchanged. Nothing reads a data root
yet: the Host's first append writes `registry.seeded`, the event every seed row names as its
registration.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Final

import sqlite_vec

from alphalattice.kernel.data.errors import DataError

SCHEMA_VERSION: Final = 1
"""`PRAGMA user_version` of the databases this code creates; it refuses any other."""
FOLDERS: Final = ("machine", "data", "blobs", "staging", "extensions")
CATALOG_DEFERRED: Final = "catalog: deferred (ducklake extension absent)"
INDEX_DEFERRED: Final = "index: vector table deferred (extension loading unavailable)"
SPEC_FIELDS: Final = {
    "event_types": ("event_type", ("work", "subject", "required", "text", "durable")),
    "metrics": ("metric", ("unit", "direction", "params")),
    "artifact_types": ("artifact_type", ("storage", "table", "measures", "keep")),
    "tags": ("tag", ("values",)),
}
"""Each seed list's definition family and the fields its rows keep as their spec."""


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
    """Every seed definition as a version-1 `definition` row, registered by the event at `seq`."""
    text = resources.files(__package__).joinpath("seed_registry.json").read_text(encoding="utf-8")
    seed = json.loads(text)
    out = [("family", name, {}, words) for name, words in seed["families"].items()]
    for key, (family, fields) in SPEC_FIELDS.items():
        out += [
            (family, row["name"], {field: row[field] for field in fields}, row["description"])
            for row in seed[key]
        ]
    tail = (None, None, owner, seq, seq, None)  # code_ref, golden, owner, two seqs, reason
    return [
        (family, name, 1, "standard", None, None, json.dumps(spec, sort_keys=True), words, *tail)
        for family, name, spec, words in out
    ]


def _statements(text: str) -> list[str]:
    """A DDL file's statements without its comments (no literal in them holds `--` or `;`)."""
    bare = "\n".join(line.split("--", 1)[0] for line in text.splitlines())
    return [part.strip() for part in bare.split(";") if part.strip()]


def _build_log(connection: sqlite3.Connection) -> None:
    connection.executescript(schema_text("log.sql"))
    with connection:
        rows = seed_rows()
        connection.executemany(f"INSERT INTO definition VALUES ({', '.join('?' * 14)})", rows)


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
    for statement in _statements(schema_text("index.sql")):
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


def _catalog(root: Path) -> bool | None:
    """The DuckLake catalog, made when DuckLake and DuckDB's SQLite scanner load from the root's
    own extensions folder with every download off; None when they do not (DuckDB absent, or the
    extensions not shipped)."""
    try:
        import duckdb  # lazy: DuckDB is the optional data extra; without it the catalog defers
    except ImportError:
        return None
    catalog = root / "machine" / "catalog.sqlite"
    if catalog.exists():
        return False
    connection = duckdb.connect(  # in memory: the catalog lives in the attached SQLite file
        ":memory:",
        config={
            "extension_directory": str(root / "extensions"),
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
        },
    )
    try:
        try:
            connection.execute("LOAD sqlite; LOAD ducklake;")
        except duckdb.Error:
            return None
        data = (root / "data").as_posix()
        connection.execute(
            f"ATTACH 'ducklake:sqlite:{catalog.as_posix()}' AS lake (DATA_PATH '{data}/')"
        )
        connection.execute("USE lake")
        for statement in _statements(schema_text("catalog.sql")):
            connection.execute(statement)
    finally:
        connection.close()
    return True


def create_data_root(path: Path) -> DataRoot:
    """Make the data root at `path`, or find it whole; running it again changes nothing."""
    created = []
    for folder in FOLDERS:
        if not (path / folder).is_dir():
            (path / folder).mkdir(parents=True, exist_ok=True)
            created.append(folder)
    tables: frozenset[str] = frozenset()
    for name, build in (("log.sqlite", _build_log), ("index.sqlite", _build_index)):
        made, tables = _database(path / "machine" / name, build)
        if made:
            created.append(f"machine/{name}")
    catalog = _catalog(path)
    if catalog:
        created.append("machine/catalog.sqlite")
    return DataRoot(
        path=path,
        created=tuple(created),
        deferred=(CATALOG_DEFERRED,) * (catalog is None)
        + (INDEX_DEFERRED,) * ("passage_vec" not in tables),  # the index's, read last
    )
