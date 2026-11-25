"""A workspace's references to committed market versions."""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.compute as pc

from alphalattice.control.workspace_runtime.database import bind_each_connection
from alphalattice.foundation.data_layer import history
from alphalattice.foundation.data_layer.root import sql_path
from alphalattice.kernel import storage
from alphalattice.kernel.identity import canonical_json, identity_blake3

VERSIONS = "market_catalog_version"
"""The workspace's exact current market version per local market key."""
ANCHOR = "market_catalog_anchor"
CACHE = "market_selection"
"""Each database instance's in-memory copy of the published selection's bodies: a read of the
shared catalog costs about ten times one of a native table, in every statement
(`bind_pending_reads`)."""
DATES = {
    "raw_bar": "session_date",
    "corporate_action": "effective_date",
    "provider_symbol": "valid_from",
}
OVERLAYS = {
    "raw_daily_bar_current": ("committed_raw_bars", "listing_id, provider, session_date"),
    "provider_adjusted_close_current": (
        "committed_adjusted_closes",
        "listing_id, provider, session_date",
    ),
    "corporate_action_current": (
        "committed_corporate_actions",
        "listing_id, provider, effective_date, action_kind",
    ),
    "provider_symbol_mapping": ("committed_mappings", "listing_id, provider, effective_from"),
}


def install(connection: Any) -> None:
    """Install the version index for a new catalog-backed workspace."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS market_catalog_version (
            world_table VARCHAR NOT NULL,
            listing_id VARCHAR NOT NULL,
            provider VARCHAR NOT NULL,
            day DATE NOT NULL,
            action_key VARCHAR NOT NULL,
            version_id UUID NOT NULL,
            catalog_listing_id UUID NOT NULL,
            payload_hash VARCHAR,
            observed_at TIMESTAMP,
            adjusted_hash VARCHAR,
            adjusted_at TIMESTAMP,
            exact_volume BIGINT,
            PRIMARY KEY (world_table, listing_id, provider, day, action_key)
        )
        """
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS market_catalog_anchor ("
        "singleton BOOLEAN PRIMARY KEY CHECK (singleton), snapshot_id BIGINT NOT NULL,"
        "cutoff_us BIGINT NOT NULL)"
    )


def enabled(connection: Any) -> bool:
    """Existing physical market stores have no version index."""
    return storage.first(
        connection.execute(
            "SELECT EXISTS (SELECT 1 FROM duckdb_tables() WHERE database_name = current_database()"
            " AND schema_name = 'main' AND table_name = ?)",
            [VERSIONS],
        )
    )[0]


def anchored(connection: Any) -> bool:
    """Whether the workspace has published its first version selection, read once per database
    instance: every connection a workspace opens asks."""
    return connection.market_fact(
        "published",
        lambda: (
            connection.market_fact("indexed", lambda: enabled(connection))
            and storage.first(connection.execute(f"SELECT 1 FROM {ANCHOR}")) is not None
        ),
    )


def closed_selection(connection: Any, database: Path) -> tuple[int, int] | None:
    """A fully published workspace selection that can be kept without market bodies."""
    if not enabled(connection):
        return None
    anchor = storage.first(connection.execute(f"SELECT snapshot_id, cutoff_us FROM {ANCHOR}"))
    if anchor is None or any(
        storage.first(connection.execute(f"SELECT 1 FROM {pending_table(database, table)} LIMIT 1"))
        for table in OVERLAYS
    ):
        raise ValueError("market_data_ops.catalog_selection_unpublished")
    return int(anchor[0]), int(anchor[1])


def write_selection_index(
    connection: Any, database: Path, destination: Path, *, omit: frozenset[str]
) -> tuple[int, int, str]:
    """Freeze the published index and supporting tables without any market bodies: the
    selection's snapshot and cutoff, and the frozen file's logical content (`content_hash`)."""
    if destination.exists():
        raise ValueError("market_data_ops.catalog_selection_unpublished")
    (source,) = storage.first(connection.execute("SELECT current_database()"))
    quoted = '"' + source.replace('"', '""') + '"'
    connection.execute(f"ATTACH '{sql_path(destination)}' AS frozen_selection")
    try:
        connection.execute("BEGIN TRANSACTION")
        try:
            selected = closed_selection(connection, database)
            if selected is None:
                raise ValueError("market_data_ops.catalog_selection_unpublished")
            connection.execute(f"COPY FROM DATABASE {quoted} TO frozen_selection (SCHEMA)")
            tables = connection.execute(
                "SELECT table_name FROM duckdb_tables() WHERE database_name = ?"
                " AND schema_name = 'main' ORDER BY table_name",
                [source],
            ).to_arrow_table()
            for name in tables["table_name"].to_pylist():
                if name in omit or name in OVERLAYS:
                    continue
                table = '"' + name.replace('"', '""') + '"'
                connection.execute(
                    f"INSERT INTO frozen_selection.main.{table} FROM {quoted}.main.{table}"
                )
            connection.execute("COMMIT")
            return (*selected, content_hash(connection, "frozen_selection"))
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    finally:
        connection.execute("DETACH frozen_selection")


_CONTENT_VERSION: Final = 1
"""The version of `content_hash`'s encoding: a change to it moves every sealed index's identity."""


def content_hash(connection: Any, catalog: str) -> str:
    """The logical content of the database `catalog` names: each main table's rows sorted by
    every column; per column, BLAKE3 (tagged `blake3:`) over its name, its Arrow type, its
    validity mask and its values with each null slot set to the type's zero and every NaN to one
    canonical NaN; then BLAKE3 (tagged) over the canonical JSON of the encoding's version and each
    table's name and column digests. Never a file's bytes or a writer's version, so neither
    DuckDB's padding nor a library upgrade moves it."""
    ((tables,),) = connection.execute(  # one row: the list of names
        "SELECT coalesce(list(table_name ORDER BY table_name), []) FROM duckdb_tables()"
        " WHERE database_name = ? AND schema_name = 'main'",
        [catalog],
    ).fetchall()
    contents = {}
    for name in tables:
        quoted = '"' + name.replace('"', '""') + '"'
        rows = connection.sql(f"SELECT * FROM {catalog}.main.{quoted} ORDER BY ALL")
        table = rows.to_arrow_table()
        contents[name] = {
            column: _column_digest(column, table.column(column).combine_chunks())
            for column in table.column_names
        }
    encoded = canonical_json({"version": _CONTENT_VERSION, "tables": contents}).encode("utf-8")
    return "blake3:" + identity_blake3(encoded).hexdigest()


def _column_digest(name: str, column: pa.Array) -> str:
    """One column's digest (`content_hash`), each part length-prefixed so no two columns share a
    byte stream; a type the index does not hold is refused by name."""
    kind = column.type
    if not len(column):
        values = b""
    elif (
        pa.types.is_string(kind) or pa.types.is_large_string(kind) or pa.types.is_string_view(kind)
    ):
        values = _text_bytes(pc.fill_null(column.cast(pa.large_string()), ""))
    elif pa.types.is_boolean(kind):
        values = _fixed_bytes(pc.fill_null(column, False).cast(pa.uint8()))
    elif pa.types.is_floating(kind):
        filled = pc.fill_null(column.cast(pa.float64()), 0.0)
        # one NaN for every NaN, built here: a scalar's conversion loads pandas, which a worker
        # that only imports this module never needs
        canonical = pa.scalar(float("nan"), pa.float64())
        values = _fixed_bytes(pc.if_else(pc.is_nan(filled), canonical, filled))
    elif pa.types.is_integer(kind) or pa.types.is_temporal(kind):
        signed = column.view(getattr(pa, f"int{kind.bit_width}")())
        values = _fixed_bytes(pc.fill_null(signed, 0))
    else:
        raise ValueError(f"market_data_ops.content_hash_type_unsupported: {name} {kind}")
    valid = _fixed_bytes(pc.is_valid(column).cast(pa.uint8()))
    digest = identity_blake3()
    for part in (name.encode("utf-8"), str(kind).encode("utf-8"), valid, values):
        digest.update(struct.pack("<Q", len(part)) + part)
    return "blake3:" + digest.hexdigest()


def _text_bytes(text: pa.LargeStringArray) -> bytes:
    """A text array's lengths, then its characters, from its offset (`_column_digest`)."""
    _validity, offsets, data = text.buffers()
    bounds = pa.Array.from_buffers(pa.int64(), len(text) + 1, [None, offsets], offset=text.offset)
    start, end = bounds[0].as_py(), bounds[-1].as_py()
    characters = b"" if data is None else data.to_pybytes()[start:end]
    return _fixed_bytes(pc.binary_length(text)) + characters


def _fixed_bytes(array: pa.Array) -> bytes:
    """A fixed-width array's values as little-endian bytes, from its offset, padding excluded."""
    if not len(array):
        return b""
    width = array.type.byte_width
    start = array.offset * width
    return array.buffers()[1].to_pybytes()[start : start + len(array) * width]


def pending_table(database: Path, table: str) -> str:
    """Name the physical write table, through a cursor's temporary read overlay."""
    catalog = database.stem.replace('"', '""')
    return f'"{catalog}".main.{table}'


def _source_columns(table: str) -> tuple[str, str, str]:
    """Read source stamps from the workspace rows while its publication still holds them."""
    if table == "raw_bar":
        return (
            "raw_daily_bar_current",
            "l.payload_hash, l.observed_at, a.payload_hash, a.updated_at,"
            " CASE WHEN try_cast(cast(l.volume AS DOUBLE) AS BIGINT)"
            " IS DISTINCT FROM l.volume THEN l.volume END",
            " LEFT JOIN provider_adjusted_close_current a"
            " ON a.listing_id = l.listing_id AND a.provider = l.provider"
            " AND a.session_date = l.session_date",
        )
    if table == "provider_symbol":
        return "provider_symbol_mapping", "NULL, NULL, NULL, NULL, NULL", ""
    return (
        "corporate_action_current",
        "l.payload_hash, l.observed_at, NULL, NULL, NULL",
        "",
    )


def place_versions(
    connection: Any,
    database: Path,
    batches: Mapping[str, Sequence[pa.Table]],
    provider_names: Mapping[str, str],
    world_listing: Callable[[str], Any],
    snapshot: int,
    cutoff_us: int,
) -> None:
    """Replace the changed keys' references in the workspace's open transaction."""
    listings = connection.execute("SELECT listing_id FROM listing").to_arrow_table()
    locals_ = listings["listing_id"].to_pylist()
    aliases = pa.table(
        {
            "local_id": pa.array(locals_, type=pa.string()),
            "world_id": pa.array(
                [str(world_listing(listing)) for listing in locals_], type=pa.string()
            ),
        }
    )
    providers = pa.table(
        {
            "provider_id": pa.array(list(provider_names), type=pa.string()),
            "provider": pa.array(list(provider_names.values()), type=pa.string()),
        }
    )
    connection.register("snapshot_listings", aliases)
    connection.register("snapshot_providers", providers)
    try:
        for table, parts in batches.items():
            date = DATES[table]
            source, stamps, adjusted = _source_columns(table)
            versions = pa.concat_tables(parts)
            connection.register("snapshot_rows", versions)
            action = "v.action_id" if table == "corporate_action" else "''"
            action_join = " AND l.action_kind = v.action_id" if table == "corporate_action" else ""
            source_date = "effective_from" if table == "provider_symbol" else date
            try:
                landed = storage.first(
                    connection.execute(
                        f"""
                    INSERT OR REPLACE INTO {VERSIONS}
                    SELECT ?, ids.local_id, p.provider, v.{date}, {action}, v.version_id,
                           v.listing_id, {stamps}
                    FROM snapshot_rows v
                    LEFT JOIN snapshot_listings ids ON ids.world_id = CAST(v.listing_id AS VARCHAR)
                    LEFT JOIN snapshot_providers p USING (provider_id)
                    JOIN {source} l
                      ON l.listing_id = ids.local_id AND l.provider = p.provider
                      AND l.{source_date} = v.{date}{action_join}
                    {adjusted}
                    """,
                        [table],
                    )
                )
                if landed != (versions.num_rows,):
                    raise ValueError("market_data_ops.catalog_version_count_moved")
                if table == "provider_symbol":
                    physical = pending_table(database, source)
                    connection.execute(
                        f"DELETE FROM {physical} p USING {VERSIONS} v, snapshot_rows r"
                        " WHERE v.world_table = 'provider_symbol' AND v.version_id = r.version_id"
                        " AND p.listing_id = v.listing_id AND p.provider = v.provider"
                        " AND p.effective_from = v.day"
                    )
            finally:
                connection.unregister("snapshot_rows")
    finally:
        connection.unregister("snapshot_providers")
        connection.unregister("snapshot_listings")
    connection.execute(
        f"INSERT OR REPLACE INTO {ANCHOR} VALUES (true, ?, ?)", [snapshot, cutoff_us]
    )


def version_relations(references: str, catalog: str) -> dict[str, str]:
    """The selected shared bodies, using the caller's qualified index and catalog."""
    return {
        "committed_raw_bars": f"""
        SELECT p.listing_id, p.provider, p.day AS session_date, r.open, r.high, r.low, r.close,
               coalesce(p.exact_volume, try_cast(r.volume AS BIGINT)) AS volume,
               p.payload_hash, p.observed_at
        FROM {references} p JOIN {catalog}.world.raw_bar r
          ON r.version_id = p.version_id AND r.listing_id = p.catalog_listing_id
          AND r.session_date = p.day
        WHERE p.world_table = 'raw_bar'
        """,
        "committed_adjusted_closes": f"""
        SELECT p.listing_id, p.provider, p.day AS session_date, r.adjusted_close,
               p.adjusted_hash AS payload_hash, p.adjusted_at AS updated_at
        FROM {references} p JOIN {catalog}.world.raw_bar r
          ON r.version_id = p.version_id AND r.listing_id = p.catalog_listing_id
          AND r.session_date = p.day
        WHERE p.world_table = 'raw_bar' AND p.adjusted_hash IS NOT NULL
        """,
        "committed_corporate_actions": f"""
        SELECT p.listing_id, p.provider, p.day AS effective_date, p.action_key AS action_kind,
               r.ratio AS new_shares_per_old_share, r.amount AS cash_amount, r.provisional,
               r.provenance, p.payload_hash,
               CASE WHEN r.retracted THEN 'RETRACTED' ELSE 'ACTIVE' END AS status, p.observed_at
        FROM {references} p JOIN {catalog}.world.corporate_action r
          ON r.version_id = p.version_id AND r.listing_id = p.catalog_listing_id
          AND r.effective_date = p.day
        WHERE p.world_table = 'corporate_action'
        """,
        "committed_mappings": f"""
        SELECT p.listing_id, p.provider, r.symbol AS provider_symbol,
               p.day AS effective_from, r.valid_to AS effective_to, r.lifecycle_state
        FROM {references} p JOIN {catalog}.world.provider_symbol r
          ON r.version_id = p.version_id AND r.listing_id = p.catalog_listing_id
          AND r.valid_from = p.day
        WHERE p.world_table = 'provider_symbol'
        """,
    }


def overlay(pending: str, committed: str, key: str) -> str:
    """Present one body per key, staged before committed."""
    return (
        f"SELECT * FROM {pending} UNION ALL"
        f" SELECT prior.* FROM {committed} prior ANTI JOIN {pending} written USING ({key})"
    )


def _published(connection: Any) -> object:
    return storage.first(connection.execute(f"SELECT snapshot_id, cutoff_us FROM {ANCHOR}"))


_BODIES = {
    "raw_bar": "committed_raw_bars",
    "corporate_action": "committed_corporate_actions",
    "provider_symbol": "committed_mappings",
}


def _require_held(connection: Any, relation: Callable[[str], str]) -> None:
    """Refuse a selection whose versions the attached catalog does not all hold, as for a
    workspace copied from another data root: each world table's versions against its bodies
    `relation` names. Its recovery is to carry them from the root it was published on
    (`world.carry_workspace` for a workspace, `world.carry_selection` for a research input)."""
    found = " UNION ALL ".join(
        f"SELECT '{table}' AS world_table, count(*) AS found FROM {relation(view)}"
        for table, view in _BODIES.items()
    )
    (short,) = storage.first(
        connection.execute(
            f"SELECT count(*) FROM (SELECT world_table, count(*) AS named FROM {VERSIONS} GROUP"
            " BY ALL)"
            f" JOIN ({found}) USING (world_table) WHERE named <> found"
        )
    )
    if short:
        raise ValueError("market_data_ops.catalog_selection_unavailable")


def _copy_selection(connection: Any) -> None:
    """Fill the instance's copy of the selection, refused when the catalog lacks part of it."""
    for name, sql in version_relations(VERSIONS, "workspace_market").items():
        connection.execute(
            f"CREATE OR REPLACE TABLE {CACHE}.main.{name} AS"
            f" SELECT * FROM ({sql}) ORDER BY listing_id"
        )
    _require_held(connection, lambda view: f"{CACHE}.main.{view}")


def write_back(connection: Any, database: Path) -> int:
    """Write the published selection's bodies, from the attached catalog, back into the
    workspace's pending tables and drop the selection, in one transaction: the workspace then
    holds its market data itself, as before its first publication. Refused by name when the
    catalog does not hold the whole selection. Returns the rows written."""
    committed = version_relations(VERSIONS, "workspace_market")
    _require_held(connection, lambda view: f"({committed[view]})")
    written = 0
    connection.execute("BEGIN TRANSACTION")
    try:
        for table, (view, key) in OVERLAYS.items():
            pending = pending_table(database, table)
            (count,) = storage.first(
                connection.execute(
                    f"INSERT INTO {pending} SELECT prior.* FROM ({committed[view]}) prior"
                    f" ANTI JOIN {pending} written USING ({key})"
                )
            )
            written += int(count)
        connection.execute(f"DELETE FROM {VERSIONS}")
        connection.execute(f"DELETE FROM {ANCHOR}")
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    return written


def bind_market_reads(connection: Any, database: Path) -> None:
    """Bind a workspace connection's market relations to its committed selection, once the
    workspace has published one (`anchored`), read through this process's data root."""
    if anchored(connection):
        port = history.store()
        root = history.data_root() if port is None else port.root
        bind_pending_reads(connection, database, root, fresh=True)


# Every repository's connection to a workspace file reads its market relations through the
# index: a Feature or score read on its own connection sees the rows a market read sees.
bind_each_connection(bind_market_reads)


@contextmanager
def transaction(database: Any) -> Iterator[Any]:
    """A writable transaction of the workspace database `database` whose market relations read
    the pending rows over the committed selection (`bind_market_reads`): committed on success,
    rolled back on failure, its connection closed."""
    connection = database.connect(read_only=False)  # bound at its open (`bind_market_reads`)
    try:
        with connection.snapshot():
            yield connection
    finally:
        connection.close()


def bind_pending_reads(
    connection: Any, database: Path, root: Path, *, fresh: bool = False, live: bool = False
) -> None:
    """Read each key from the staged delta first, then the workspace's committed selection: the
    instance's copy of it (`CACHE`), or the catalog's `live` rows, which a publishing transaction
    reads since it moves the selection. A `fresh` connection, just opened, has bound nothing."""
    if (
        not fresh
        and not live
        and storage.first(
            connection.execute(
                "SELECT 1 FROM duckdb_views() WHERE temporary AND view_name ="
                " 'raw_daily_bar_current'"
            )
        )
    ):
        return
    committed = version_relations(VERSIONS, "workspace_market")

    def view(table: str, copied: bool) -> str:
        body, key = OVERLAYS[table]
        relation = f"{CACHE}.main.{body}" if copied else f"({committed[body]})"
        return (
            f"CREATE OR REPLACE TEMP VIEW {table} AS"
            f" {overlay(pending_table(database, table), relation, key)}"
        )

    if live:  # a publishing transaction reads them at once, in place of any pending bind
        connection.bind_on_use({})
        connection.attach_market_catalog(root)
        connection.execute(";".join(view(table, False) for table in OVERLAYS))
        return
    held: list[bool] = []  # the selection is checked, and the copy refilled, once a view is read

    def bound(table: str) -> None:
        connection.attach_market_catalog(root)  # a connection that reads no market row needs none
        if not held:
            copied = connection.hold_market_selection(CACHE, _published, _copy_selection)
            if not copied:  # read live inside a transaction: refused as the copy's fill refuses
                _require_held(connection, lambda body: f"({committed[body]})")
            held.append(copied)
        connection.execute(view(table, held[0]))

    # Most opens read none of them: a view costs about half a millisecond to create.
    connection.bind_on_use({table: partial(bound, table) for table in OVERLAYS})
