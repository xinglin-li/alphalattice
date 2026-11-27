"""The threads DuckDB and Arrow read with: the operator's, from the CPU budget.

Both engines keep one setting per process. DuckDB keeps it per database instance, and every
connection to the instance shares it (`SET threads`); Arrow keeps its CPU pool. The Host applies
the budget's cores when it starts a Task (`apply_reader_threads`), and every DuckDB connection a
workspace reader opens takes the applied value (`connect_duckdb`), whichever reader opens it;
before anything is applied both engines keep their own defaults. Under the operator's thread share
(`thread_share`, which Arrow reads itself), every instance a reader opens is created with it for
its worker and async I/O pools, so neither starts wider, and no budget applied later widens it.
Every product opener of a file is this one, with one creation config, so a second opener in the
process matches it; a budget's later `SET` keeps that match. It opens through the storage port
(kernel.storage), whose driver is `DRIVER`, installed when this module is imported and by each
process's entry. Importing it loads neither engine: DuckDB loads at the first open, Arrow at the
first grant, so a CLI command that opens nothing pays for neither.
The product's queries order their rows and sum no floating-point values, so no result depends on
the count (a fixed query set gave the same bytes at 1 and 8 threads, 2026-09-26).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from alphalattice.capabilities.domain_contracts.environment import (
    duckdb_thread_config,
    set_duckdb_pool_width,
    thread_share,
)
from alphalattice.kernel import storage
from alphalattice.kernel.identity import identity_blake3, tagged_blake3
from alphalattice.kernel.resources import cpu_width, current_resource_lease

if TYPE_CHECKING:
    import duckdb

_LOCK = Lock()
_DIGEST_REGISTRATION = Lock()
_THREADS: list[int] = []


def reader_threads() -> int | None:
    """The threads applied to DuckDB and Arrow in this process; None before any."""
    return _THREADS[-1] if _THREADS else None


def apply_reader_threads(threads: int) -> int:
    """From now on, DuckDB connections opened here and Arrow's CPU pool use `threads`, within the
    operator's share."""
    threads = max(1, min(int(threads), thread_share() or int(threads)))
    with _LOCK:
        _THREADS[:] = [threads]
        import pyarrow as pa  # lazy: Arrow loads at the first grant, not at the driver's install

        pa.set_cpu_count(threads)
    return threads


def configure_duckdb(connection: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Give a DuckDB connection's instance the applied threads."""
    threads = reader_threads()
    if threads is not None:
        connection.execute(f"SET threads TO {threads}").complete()
    elif current_resource_lease() is not None:
        set_duckdb_pool_width(connection, cpu_width())
    return connection


def _identity_batch(values: Any) -> Any:
    """Apply the hashing owner's tagged identity to DuckDB's UTF-8 values."""
    import pyarrow as pa  # lazy: only an active digest batch needs Arrow.

    return pa.array(
        [tagged_blake3(identity_blake3(value.encode("utf-8"))) for value in values.to_pylist()],
        type=pa.string(),
    )


def _register_blake3_sql(connection: duckdb.DuckDBPyConnection) -> None:
    from duckdb.sqltypes import VARCHAR  # lazy: the driver has just opened DuckDB.

    connection.create_function(
        "alphalattice_blake3", _identity_batch, [VARCHAR], VARCHAR, type="arrow"
    )


def install_blake3_sql(connection: duckdb.DuckDBPyConnection) -> None:
    """Install the digest on an already open instance whose queries or views use it."""
    with _DIGEST_REGISTRATION:
        current = connection.execute(
            "SELECT 1 FROM duckdb_functions() WHERE function_name = 'alphalattice_blake3'"
            " AND database_name = 'system' AND function_type = 'scalar' AND internal"
        ).fetchall()
        if not current:
            _register_blake3_sql(connection)


class _DuckDB:
    """The DuckDB driver the storage port opens every instance through (kernel.storage)."""

    def connect(
        self, target: str, *, read_only: bool, config: Mapping[str, object]
    ) -> duckdb.DuckDBPyConnection:
        """Open `target` with `config`; a file takes the applied threads."""
        import duckdb  # lazy: DuckDB loads at the first open, not at the driver's install

        connection = duckdb.connect(target, read_only=read_only, config=dict(config))
        try:
            if target == storage.MEMORY:
                # The process's shared in-memory instance installs once, before opening cursors.
                _register_blake3_sql(connection)
                return connection
            return configure_duckdb(connection)
        except BaseException:
            connection.close()
            raise


DRIVER = _DuckDB()
"""The product's driver: installed in the storage port when this module is imported."""
storage.install_driver(DRIVER)


def connect_duckdb(path: Path | str, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB file created within the operator's share, with the applied threads."""
    connection: duckdb.DuckDBPyConnection = storage.connect(
        str(path),
        read_only=read_only,
        config={
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
            **duckdb_thread_config(),
        },
    )
    return connection


__all__ = ["DRIVER", "apply_reader_threads", "configure_duckdb", "connect_duckdb", "reader_threads"]
