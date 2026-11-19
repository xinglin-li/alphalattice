"""The setup that serves a workspace's derived views (workspace_runtime.derived_views): the data
root's catalog attached read-only to the workspace's DuckDB instance, once, and each view a TEMP
view on the connection; and the calls that read, keep and root derived partitions held as files
(workspace_runtime.panel_reads). Registered on import; data_layer alone writes the data root."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

from alphalattice.control.workspace_runtime import derived_views, panel_reads
from alphalattice.kernel.data_layer.root import (
    configure_read_cache,
    data_root,
    load_extensions,
    sql_path,
)


def serve(connection: Any, views: Sequence[tuple[str, str]]) -> None:
    """Attach this process's data root's catalog as `lake`, read-only, unless the instance holds
    it already, and make each view a TEMP view of this connection."""
    configure_read_cache(connection)
    if not connection.execute(
        "SELECT 1 FROM duckdb_databases() WHERE database_name = 'lake'"
    ).fetchall():
        root = data_root()
        catalog, data = sql_path(root / "machine" / "catalog.sqlite"), sql_path(root / "data")
        load_extensions(connection)
        connection.execute(
            f"ATTACH IF NOT EXISTS 'ducklake:sqlite:{catalog}' AS lake"
            f" (DATA_PATH '{data}/', READ_ONLY)"
        ).complete()
    for name, body in views:
        connection.execute(f"CREATE OR REPLACE TEMP VIEW {name} AS {body}").complete()


def body(table: str, digest: str) -> Path | None:
    """The body the log names for `table`'s file partition with `digest` (history's read)."""
    import alphalattice.kernel.data_layer.history as history  # lazy: history imports the log

    return history.derived_file_body(table, digest)


def keep(table: str, **partition: Any) -> None:
    """Hand one file to the Host to keep as a partition of `table` (history's write)."""
    import alphalattice.kernel.data_layer.history as history  # lazy: history imports the log

    history.keep_derived_file(table, **partition)


def root(table: str, digests: Collection[str], **lifecycle: Any) -> dict[str, int] | None:
    """Root, evict and collect a file table's partitions in the Host (history's lifecycle)."""
    import alphalattice.kernel.data_layer.history as history  # lazy: history imports the log

    return history.root_derived_files(table, digests, **lifecycle)


derived_views.register(serve)
panel_reads.register(body, keep, root)
