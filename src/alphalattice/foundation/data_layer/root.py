"""The 0.2.8 data root: its folders and its event log holding the seed registry and the knowledge
store.

machine/log.sqlite      the event log and every table projected from it (schema/log.sql), with
                        the seed registry and the knowledge store's sources, chunks and embeddings
machine/catalog.sqlite  the DuckLake catalog (schema/catalog.sql), made at its first use by the
                        store (`catalog`), using the installed extension wheels when the data
                        extra is present; nothing is ever fetched
data, blobs, staging    DuckLake's Parquet files, content-addressed blobs, compute staging

The Host records the seed lineages' minted IDs in `registry.seeded`, the first event of a new
root and the event each seed row names as its registration. Older roots keep the IDs already
held in the name table.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any, Final

from alphalattice.capabilities.data.errors import DataError
from alphalattice.capabilities.domain_contracts.environment import (
    package_versions,
)
from alphalattice.foundation.data_layer.project import register, uuid7
from alphalattice.kernel import storage
from alphalattice.kernel.config import DATA_ROOT_ENV as DATA_ROOT_ENV
from alphalattice.kernel.config import data_root as data_root
from alphalattice.kernel.resources import cpu_width

SCHEMA_VERSION: Final = 16
"""`PRAGMA user_version` of the databases this code creates; it refuses any other. A change to
any schema file, or to how world derives the rows a root holds, takes the next number, so a root
an earlier build made is refused by name."""
FOLDERS: Final = ("machine", "data", "blobs", "staging", "extensions")
CATALOG_DEFERRED: Final = "catalog: deferred (ducklake extension absent)"
SPEC_FIELDS: Final = {
    "event_types": ("event_type", ("work", "subject", "required", "text", "durable")),
    "metrics": ("metric", ("unit", "direction", "params")),
    "artifact_types": ("artifact_type", ("storage", "table", "measures", "keep")),
    "tags": ("tag", ("values",)),
    "world_tables": ("world_table", ()),
}
"""Each seed list's definition family and the fields its rows keep as their spec."""
LAKE_THREAD_CEILING: Final = 8
"""The lake's small statements pay more in fan-out past eight than their scans gain."""
READ_AHEAD: Final = 0
"""Scan jobs an instance opened over the lake reads ahead of decoding: none. DuckDB's default is
bounded only by memory, and each queued job holds its Parquet file open. The lake's files are many
and small (each admission writes one for each year its rows fall in), so a scan held one for each
file, past macOS's default of 256 per process. With none, a scan holds about one file per thread,
and over local files it is faster: on a 1,100-file catalog, every row in 0.14 s against 0.38 s."""


@dataclass(frozen=True)
class DataRoot:
    """What one call made now."""

    path: Path
    created: tuple[str, ...]


def schema_text(name: str) -> str:
    """One DDL file of the package: log.sql, catalog.sql or views.sql."""
    return resources.files(__package__).joinpath("schema", name).read_text(encoding="utf-8")


def seed_rows(
    owner: str = "kernel", seq: int = 1, *, historical_count: int | None = None
) -> list[tuple]:
    """Every seed definition as a row, (family, name, version, then definition_version's columns
    after its version), registered by the event at `seq`. A seed row is version 1 unless it names
    its version and its relation to the version before it. A v1 seed announcement's row count
    distinguishes the registry before and after event types were added."""
    text = resources.files(__package__).joinpath("seed_registry.json").read_text(encoding="utf-8")
    seed = json.loads(text)
    out = [("family", name, {}, {}, words) for name, words in seed["families"].items()]
    for key, (family, fields) in SPEC_FIELDS.items():
        out += [
            (family, row["name"], row, {field: row[field] for field in fields}, row["description"])
            for row in seed[key]
            if historical_count is None or row.get("first_seed_count", 0) <= historical_count
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


def build_log(
    connection: sqlite3.Connection,
    *,
    seed_ids: Mapping[tuple[str, str], str] | None = None,
    seed_version: int = 2,
) -> None:
    """Build the log's seed registry, minting ids for a new root and retaining them on replay."""
    connection.executescript(schema_text("log.sql"))
    with connection:
        for family, name, version, *fields in seed_rows():
            if family == "event_type" and name == "registry.seeded" and version > seed_version:
                continue
            if seed_ids is not None and (family, name) not in seed_ids:
                continue
            identity = None
            if version == 1:
                identity = uuid7() if seed_ids is None else seed_ids[family, name]
            register(connection, family, name, version, *fields, identity=identity)


def world_table_ids(connection: Any, *, attached: bool = False) -> dict[str, str]:
    """Resolve world tables from SQLite or from a reader's already attached log."""
    schema = "log." if attached else ""
    return dict(
        connection.execute(
            f"SELECT name, definition_id FROM {schema}definition_name WHERE family = 'world_table'"
        ).fetchall()
    )


def view_statements(table_ids: Mapping[str, str]) -> Iterator[str]:
    """Bind stored ids as constants so the catalog filters its table before the anti-join."""
    if table_ids:
        choices = " ".join(
            f"""WHEN '{name.replace("'", "''")}' THEN '{identity.replace("'", "''")}'"""
            for name, identity in table_ids.items()
        )
        yield f"CREATE MACRO world_table_id(table_name) AS CASE table_name {choices} END"
    yield from statements(schema_text("views.sql"))


def _database(path: Path, build) -> bool:
    """Whether the database at `path` was made now: built whole under a temporary name and moved
    into place, so a crash leaves none or all of it. An existing one is only read, and refused
    unless it is this schema's version; the read is immutable, so a WAL database gains no -shm or
    -wal file."""
    if path.exists():
        uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            found = connection.execute("PRAGMA user_version").fetchone()[0]
        if found != SCHEMA_VERSION:
            raise DataError(
                f"{path} is schema version {found}; this product reads version {SCHEMA_VERSION}",
                code="data_layer.schema_mismatch",
            )
        return False
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.unlink(missing_ok=True)
    with closing(sqlite3.connect(partial, isolation_level=None)) as connection:
        build(connection)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    os.replace(partial, path)
    return True


def load_extensions(connection) -> None:
    """Load engine-matched binaries from the installed wheels, never INSTALL or a repository."""
    import duckdb  # lazy: data is optional; log creation works without DuckDB

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


def bound_reads(connection: Any) -> None:
    """An instance opened over the lake reads ahead no scan job (READ_AHEAD), so a scan holds
    about one Parquet file open per thread, however many files the lake has. The setting is the
    instance's: its opener sets it, and a workspace's own instance, which the lake is only
    attached to, keeps DuckDB's default, under which its native tables read faster."""
    connection.execute(f"SET read_ahead_depth = {READ_AHEAD}").complete()


def sql_path(path: PurePath) -> str:
    """`path` for a DuckDB string literal: forward slashes, each quote doubled, and a Windows
    extended-length path (`//?/D:/...`, `//?/UNC/server/...` as forward slashes give them) in its
    ordinary spelling, which DuckDB can open and derive its WAL's name from."""
    return _without_extended_prefix(path.as_posix()).replace("'", "''")


def _without_extended_prefix(text: str) -> str:
    if text.startswith("//?/UNC/"):
        return "//" + text[len("//?/UNC/") :]
    return text.removeprefix("//?/")


def sqlite_read_only_uri(path: PurePath) -> str:
    """Build a SQLite file URI without treating a Windows path prefix as its authority."""
    if isinstance(path, PureWindowsPath):
        path = PureWindowsPath(_without_extended_prefix(path.as_posix()))
    return path.as_uri() + "?mode=ro"


def configure_read_cache(connection: Any) -> None:
    """Keep local Parquet data and its metadata in DuckDB's own caches."""
    connection.execute("SET parquet_metadata_cache = true").complete()
    connection.execute("SET cache_local_files = true").complete()


def lake_thread_config(threads: int | None = None) -> dict[str, int]:
    """Size both of a new lake instance's pools within its execution and Host limits."""
    width = cpu_width(min(threads or LAKE_THREAD_CEILING, LAKE_THREAD_CEILING))
    return {"threads": width, "async_threads": width}


def duckdb_session(root: Path, *, threads: int | None = None) -> Any | None:
    """An in-memory DuckDB session with the SQLite scanner and DuckLake loaded from their installed
    wheels, every download off; None without the data extra. The caller closes it."""
    try:
        import duckdb  # noqa: F401  # lazy: DuckDB is the optional data extra; without it the session defers
    except ImportError:
        return None
    connection = storage.connect(
        storage.MEMORY,
        config={
            "extension_directory": str(root / "extensions"),
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
            **lake_thread_config(threads),
        },
    )
    try:
        load_extensions(connection)
        bound_reads(connection)
    except ModuleNotFoundError:
        connection.close()
        return None
    except BaseException:
        connection.close()
        raise
    return connection


def enable_catalog_wal(root: Path) -> None:
    """Set the local catalog's journal mode before opening any DuckLake attachment."""
    path = root / "machine" / "catalog.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        if connection.execute("PRAGMA journal_mode = WAL").fetchone() != ("wal",):
            raise DataError(
                f"the local catalog at {path} could not enable WAL",
                code="data_layer.history_unavailable",
            )


def attach_catalog(connection: Any, root: Path) -> None:
    """Attach the writable catalog to an instance whose extensions are already loaded."""
    catalog_path = sql_path(root / "machine" / "catalog.sqlite")
    data = sql_path(root / "data")
    connection.execute(
        f"ATTACH 'ducklake:sqlite:{catalog_path}' AS lake"
        f" (DATA_PATH '{data}/', DATA_INLINING_ROW_LIMIT 1000)"
    ).complete()


def catalog(root: Path, *, owner: Any = None, keep_attached: bool = False) -> bool | None:
    """Create the SQLite-backed DuckLake catalog at its first use: True when made, False when
    present, None without the data extra. A writer with an owner can keep a newly created catalog
    attached instead of reopening it. A backup only needs it created."""
    if importlib.util.find_spec("duckdb") is None:
        return None
    path = root / "machine" / "catalog.sqlite"
    if path.exists():
        return False
    connection = duckdb_session(root) if owner is None else owner.cursor()
    if connection is None:
        return None
    attached = created = False
    try:
        enable_catalog_wal(root)
        attach_catalog(connection, root)
        attached = True
        connection.execute("USE lake").complete()
        # Each run of DDL commits once, in file order: a DuckLake commit per statement made a
        # catalog in two seconds. An option call stands alone, after its schema or table commits.
        run: list[str] = []
        for statement in [*statements(schema_text("catalog.sql")), None]:
            if statement is not None and not statement.upper().startswith("CALL "):
                run.append(statement)
                continue
            if run:
                connection.execute("BEGIN TRANSACTION").complete()
                for ddl in run:
                    connection.execute(ddl).complete()
                connection.execute("COMMIT").complete()
                run = []
            if statement is not None:
                connection.execute(statement).complete()
        created = True
    finally:
        connection.close()
        if owner is not None and attached and not (keep_attached and created):
            with closing(owner.cursor()) as detach:
                detach.execute("DETACH lake").complete()
    return True


def catalog_matches(lake: Any, *, name: str = "lake") -> None:
    """Refuse catalog tables whose columns differ from catalog.sql.

    Older 0.2.8 roots may lack an additive results table until its writer opens it.
    Tables a later step adds (a feature set's) are not catalog.sql's and are not compared.
    """
    declared = {
        f"{schema}.{table}": re.findall(r"^  (\w+) ", body, re.M)
        for schema, table, body in re.findall(
            r"^CREATE TABLE (\w+)\.(\w+) \((.*?)^\);", schema_text("catalog.sql"), re.M | re.S
        )
    }
    held: dict[str, list[str]] = {}
    for schema, table, column in lake.execute(
        "SELECT schema_name, table_name, column_name FROM duckdb_columns()"
        " WHERE database_name = ? ORDER BY schema_name, table_name, column_index",
        (name,),
    ).fetchall():
        held.setdefault(f"{schema}.{table}", []).append(column)
    moved = sorted(
        name
        for name, columns in declared.items()
        if held.get(name) != columns
        and not (
            name in {"results.validation_receipt", "results.sector_evaluation"} and name not in held
        )
    )
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
    if _database(path / "machine" / "log.sqlite", build_log):
        created.append("machine/log.sqlite")
    return DataRoot(path=path, created=tuple(created))
