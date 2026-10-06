"""The threads DuckDB and Arrow read with: the operator's, from the CPU budget (binding plan, B7).

Both engines keep one setting per process. DuckDB keeps it per database instance, and every
connection to the instance shares it (`SET threads`); Arrow keeps its CPU pool. The Host applies
the budget's cores when it starts a Task (`apply_reader_threads`), and every DuckDB connection a
workspace reader opens takes the applied value (`connect_duckdb`), whichever reader opens it;
before anything is applied both engines keep their own defaults. The product's queries order
their rows and sum no floating-point values, so no result depends on the count (a fixed query
set gave the same bytes at 1 and 8 threads, 2026-09-26).
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import duckdb
import pyarrow as pa

_LOCK = Lock()
_THREADS: list[int] = []


def reader_threads() -> int | None:
    """The threads applied to DuckDB and Arrow in this process; None before any."""
    return _THREADS[-1] if _THREADS else None


def apply_reader_threads(threads: int) -> int:
    """From now on, DuckDB connections opened here and Arrow's CPU pool use `threads`."""
    threads = max(1, int(threads))
    with _LOCK:
        _THREADS[:] = [threads]
        pa.set_cpu_count(threads)
    return threads


def configure_duckdb(connection: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Give a DuckDB connection's instance the applied threads."""
    threads = reader_threads()
    if threads is not None:
        connection.execute(f"SET threads TO {threads}")
    return connection


def connect_duckdb(path: Path | str, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB file with the applied threads."""
    return configure_duckdb(duckdb.connect(str(path), read_only=read_only))


__all__ = ["apply_reader_threads", "configure_duckdb", "connect_duckdb", "reader_threads"]
