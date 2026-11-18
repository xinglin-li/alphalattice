"""The setup that serves a workspace's derived views (workspace_runtime.derived_views): the data
root's catalog attached read-only to the workspace's DuckDB instance, once, and each view a TEMP
view on the connection. Registered on import; data_layer alone loads DuckLake."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from alphalattice.control.workspace_runtime import derived_views
from alphalattice.kernel.data_layer.root import data_root, load_extensions, sql_path


def serve(connection: Any, views: Sequence[tuple[str, str]]) -> None:
    """Attach this process's data root's catalog as `lake`, read-only, unless the instance holds
    it already, and make each view a TEMP view of this connection."""
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


derived_views.register(serve)
