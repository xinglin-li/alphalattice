"""A workspace's derived views: a derived table of the data root's catalog, read under the name of
the workspace table it replaces.

A view is declared in the workspace's own file (`workspace_derived_view`: its name, the setting
that serves it, and its body over the catalog attached as `lake`). On each read-only connection
whose setting is `derived`, the registered setup, data_layer's, which alone loads DuckLake,
attaches the catalog read-only to the instance and makes each view a TEMP view. DuckDB resolves a
name to the TEMP view before the workspace's own table, from a persisted view as well, so every
reader of the table reads the catalog's rows. A writable connection makes no view and keeps
writing the workspace's table, which stays the rollback until it is retired. A connection that
would serve a view with no setup registered is refused by name: it never reads the workspace's
table in the view's place.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Final

import duckdb

TABLE: Final = "workspace_derived_view"
DERIVED: Final = "derived"
"""The value of a view's setting that serves it."""

Setup = Callable[[duckdb.DuckDBPyConnection, Sequence[tuple[str, str]]], None]
_setups: list[Setup] = []


def register(setup: Setup | None) -> None:
    """Install the one setup that serves a connection's derived views (data_layer's); None leaves
    the process with none, so a connection that would serve a view is refused."""
    _setups[:] = [] if setup is None else [setup]


def registered() -> Setup | None:
    """The setup installed, if any."""
    return _setups[0] if _setups else None


def declare(connection: duckdb.DuckDBPyConnection, view: str, *, setting: str, body: str) -> None:
    """Declare, or replace, the workspace's derived view `view`: `body` over `lake`, served while
    the environment's `setting` is `derived`."""
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE} (view_name VARCHAR PRIMARY KEY,"
        " setting VARCHAR NOT NULL, body VARCHAR NOT NULL)"
    )
    connection.execute(f"INSERT OR REPLACE INTO {TABLE} VALUES (?, ?, ?)", [view, setting, body])


def serve(connection: duckdb.DuckDBPyConnection) -> None:
    """Make the derived views this connection's settings serve, before its first statement;
    refused by name when one is served and no setup is registered. A process none of whose
    settings is derived reads nothing here, so the default costs a connection nothing."""
    if DERIVED not in os.environ.values():
        return
    declared = connection.execute(
        "SELECT 1 FROM duckdb_tables() WHERE database_name = current_database()"
        " AND schema_name = 'main' AND table_name = ?",
        [TABLE],
    ).fetchall()
    if not declared:
        return
    views = connection.execute(f"SELECT view_name, setting, body FROM main.{TABLE}").fetchall()
    served = [(name, body) for name, setting, body in views if os.environ.get(setting) == DERIVED]
    if not served:
        return
    if not _setups:
        raise RuntimeError("workspace_database.derived_setup_absent")
    _setups[0](connection, served)


__all__ = ["DERIVED", "TABLE", "declare", "register", "registered", "serve"]
