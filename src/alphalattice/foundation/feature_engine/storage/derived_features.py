"""Feature values as a derived method: the setting that chooses them, and the workspace's
partitions of the catalog's feature set table that a derived read answers from.

`ALPHALATTICE_FEATURE_READS` is `stored`, the default and the rollback, or `derived`. Under
`derived` the Host keeps each workspace's `feature_daily_current` in the catalog's table of the
feature set, `lake.derived.feature_set_<its lineage>`, one partition per catalog, month and
listing: a build's batch writes whole listings, and a day's append moves one month of each. A
partition's digest is its rows' content (`CONTENT`), so workspaces holding the same rows share one
copy, and workspaces holding other rows (other listings, values or build progress) never share
one. A workspace reads only the
partitions its own table names (`PARTS`), and roots them in the log's cache_root; a partition no
workspace roots is collected.

After each committed feature write, `mirror` digests the written listings' partitions and writes
those whose digest moved, in one snapshot per catalog. It roots them before the workspace reads
them, points the workspace at them, then drops the roots it no longer reads and collects. With no
listings (a build's start, a discarded part) it does the same over every partition, so a copy the
Host stopped before is completed. It declares the workspace's derived view
(workspace_runtime.derived_views), through which every reader of `feature_daily_current` takes the
catalog's rows. The workspace table stays written: it is the rollback until row U retires it.
"""

from __future__ import annotations

import os
from collections.abc import Collection
from typing import Any, Final

from alphalattice.control.workspace_runtime import derived_views
from alphalattice.control.workspace_runtime.database import WorkspaceDatabase
from alphalattice.foundation.market_data_ops.storage import world_reads
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.project import definition_id
from alphalattice.kernel.data_layer.store import feature_table

SETTING: Final = "ALPHALATTICE_FEATURE_READS"
CHOICES: Final = ("stored", "derived")
"""Where Feature values come from: the workspace's stored table, or the derived method."""
VIEW: Final = "feature_daily_current"
PARTS: Final = "feature_derived_part"
"""The workspace's table of the catalog's partitions its derived view reads, each by its digest."""
CONTENT: Final = (
    "'feature-rows-1:' || sha256(string_agg(concat_ws('|', CAST(session_date AS VARCHAR),"
    " row_hash, raw_input_hash, action_set_hash, market_reference_revision, cutoff_set_hash,"
    " coalesce(source_verification_receipt_hash, '-'),"
    " strftime(updated_at, '%Y-%m-%d %H:%M:%S.%f')), chr(10) ORDER BY session_date))"
)
"""A partition's digest, version 1: SHA-256 over its rows' identities in session order. A row's
identity is its row hash (the complete catalog row: its factor values and cutoffs,
materialization_identity) and the lineage the build stamps on it, so equal digests hold equal
rows. A copy is a function of its rows alone, and they carry its listings and build progress."""


def chosen() -> str:
    """The setting's value; refused by name when it names neither choice."""
    value = os.environ.get(SETTING, CHOICES[0])
    if value not in CHOICES:
        raise DataError(
            f"{SETTING} is {value!r}; set it to stored or derived, or unset it for stored",
            code="feature.feature_reads_unknown",
        )
    return value


def mirror(database: WorkspaceDatabase, stable_id: str, listing_ids: Collection[str]) -> None:
    """Under `derived`, bring the catalog's partitions of `listing_ids` level with the workspace's
    feature rows and point the workspace's derived view at them; every partition while the view
    is undeclared, or with no listings (a build's start, a discarded part), which completes a
    copy the Host stopped before. Refused by name outside the Host, which alone writes the
    catalog, and for a workspace the Host's log does not know, whose roots would have no owner."""
    if chosen() != "derived":
        return
    port = history.store()
    if port is None:
        raise DataError(
            f"{SETTING}=derived keeps Feature values in the data root's catalog, which only the"
            " Host writes; run the build in the Host, or unset the setting",
            code="feature.derived_reads_need_host",
        )
    workspace = world_reads.workspace(database.path)
    if workspace is None:
        raise DataError(
            f"{SETTING}=derived roots a workspace's Feature values under its id, and the Host's"
            f" log knows no workspace at {database.workspace}; open this workspace in the Host,"
            " or unset the setting",
            code="feature.derived_reads_need_workspace",
        )
    table = feature_table(definition_id("feature_set", stable_id))
    with database.transaction() as connection:
        connection.execute(
            f"CREATE TABLE IF NOT EXISTS {PARTS} (recipe_key VARCHAR NOT NULL, part VARCHAR NOT"
            " NULL, listing_id VARCHAR NOT NULL, input_digest VARCHAR NOT NULL,"
            " PRIMARY KEY (recipe_key, part))"
        )
        scope, parameters = "true", []
        if listing_ids and _declared(connection):
            scope, parameters = "list_contains(?::VARCHAR[], listing_id)", [sorted(listing_ids)]
        connection.execute(
            "CREATE OR REPLACE TEMP TABLE feature_parts_now AS"
            " SELECT recipe_key, listing_id, month,"
            " strftime(month, '%Y-%m') || '/' || CAST(CAST(listing_id AS UUID) AS VARCHAR) AS part,"
            " input_digest FROM (SELECT catalog_hash AS recipe_key, listing_id,"
            f" date_trunc('month', session_date) AS month, {CONTENT} AS input_digest"
            " FROM feature_daily_current"
            f" WHERE {scope} GROUP BY ALL)",
            parameters,
        )
        connection.execute(
            "CREATE OR REPLACE TEMP TABLE feature_parts_moved AS SELECT n.* FROM feature_parts_now"
            f" n LEFT JOIN {PARTS} h USING (recipe_key, part)"
            " WHERE h.input_digest IS DISTINCT FROM n.input_digest"
        )
        moved = connection.execute(
            "SELECT recipe_key, part, input_digest FROM feature_parts_moved ORDER BY ALL"
        ).fetchall()
        for recipe in sorted({row[0] for row in moved}):
            rows = connection.sql(
                "SELECT f.catalog_hash AS recipe_key, m.input_digest,"
                " CAST(f.listing_id AS UUID) AS listing_id, f.session_date,"
                " f.* EXCLUDE (listing_id, session_date, catalog_hash) FROM feature_daily_current f"
                " JOIN feature_parts_moved m ON m.recipe_key = f.catalog_hash"
                " AND m.listing_id = f.listing_id AND m.month = date_trunc('month', f.session_date)"
                " WHERE m.recipe_key = ?",
                params=[recipe],
            ).to_arrow_table()
            parts = connection.sql(
                "SELECT part, input_digest FROM feature_parts_moved WHERE recipe_key = ?",
                params=[recipe],
            ).to_arrow_table()
            port.derive(
                table, rows, recipe_key=recipe, parts=parts, compute_ms=0, rooted_by=workspace
            )  # rooted under the writer, before the view reads them or a collection sees them
        gone = connection.execute(
            f"DELETE FROM {PARTS} h WHERE {scope} AND NOT EXISTS (SELECT 1 FROM feature_parts_now"
            " n WHERE n.recipe_key = h.recipe_key AND n.part = h.part)",
            parameters,
        ).fetchall()[0][0]
        connection.execute(
            f"INSERT OR REPLACE INTO {PARTS} SELECT recipe_key, part, listing_id, input_digest"
            " FROM feature_parts_moved"
        )
        derived_views.declare(
            connection, VIEW, setting=SETTING, body=_body(connection, port, table)
        )
        changed = bool(moved or gone)
        read = (
            connection.execute(f"SELECT recipe_key, part, input_digest FROM {PARTS}").fetchall()
            if changed
            else []
        )
        connection.execute("DROP TABLE feature_parts_moved")
        connection.execute("DROP TABLE feature_parts_now")
    # Nothing moved or gone: the roots already match, and a crash before them left only extra
    # roots, which the next build that moves a value drops.
    if changed and port.log.rooted(workspace, table, read, exactly=True):
        port.collect(table)  # after the view stopped reading
    if not listing_ids:  # a build's start, after its reconcile: compact, and bound the bytes
        port.maintain()  # what the earlier builds' listing batches wrote, once per build


def _declared(connection: Any) -> bool:
    """Whether the workspace has declared its derived view, after its first copy."""
    return bool(
        connection.execute(
            "SELECT 1 FROM duckdb_tables() WHERE table_name = ?", [derived_views.TABLE]
        ).fetchall()
        and connection.execute(
            f"SELECT 1 FROM {derived_views.TABLE} WHERE view_name = ?", [VIEW]
        ).fetchall()
    )


def _body(connection: Any, port: Any, table: str) -> str:
    """The view's body: the workspace table's columns, in its order and types, from the catalog's
    partitions the workspace reads (`PARTS`), a column the catalog does not hold yet as NULL."""
    held = {row[0] for row in port.lake.execute(f"DESCRIBE lake.derived.{table}").fetchall()}
    columns = []
    for _, name, kind, *_ in connection.execute(
        "PRAGMA table_info('feature_daily_current')"
    ).fetchall():
        if name == "listing_id":
            columns.append("CAST(listing_id AS VARCHAR) AS listing_id")
        elif name == "catalog_hash":
            columns.append("recipe_key AS catalog_hash")
        elif name in held:
            columns.append(f'CAST("{name}" AS {kind}) AS "{name}"')
        else:
            columns.append(f'CAST(NULL AS {kind}) AS "{name}"')
    return (
        f"SELECT {', '.join(columns)} FROM lake.derived.{table} t WHERE EXISTS (SELECT 1 FROM"
        f" {PARTS} p WHERE p.recipe_key = t.recipe_key AND p.input_digest = t.input_digest)"
    )


__all__ = ["CHOICES", "CONTENT", "PARTS", "SETTING", "VIEW", "chosen", "mirror"]
