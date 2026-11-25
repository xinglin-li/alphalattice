"""Market value reads from world: the session that serves them, and 0.2.7's market tables as
world's as-of reads, in 0.2.7's names and columns.

World is the one store of committed market truth, and serves every market read (`reading`) of
a workspace published into it but those of a market data update (`writing`), a stage writing the
workspace: the stage's reads take the workspace's own tables, its write buffer, and when its
outermost update ends the Host publishes what its writers journaled (world.publish). Nothing a
stage wrote is read outside it before then. A workspace this machine's log records no publish of,
and a process with no catalog to read, read the workspace's tables.
A world read runs on `session`: in the Host a cursor on its StorePort's session, and in any other
process the data root's catalog attached read-only (`reading_session`, without the log), opened
once. DuckDB never opens the log: the facts a read takes from it come through Python's sqlite3,
the Host's own connection or a read-only one (`_log_rows`), so a process opens the file with one
SQLite library, whose locks no second library's closing descriptor releases.
Each relation reads world as of a cutoff in microseconds: a selection's, or `LATEST`, all that is
known. A listing is named by its 0.2.7 id, which world keeps as its alias (world.LISTING_ALIAS),
and a provider by its lineage's name.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import pyarrow as pa

from alphalattice.control.workspace_runtime.database import release_workspace_database
from alphalattice.foundation.data_layer import history
from alphalattice.foundation.data_layer.history import catalog_session as session
from alphalattice.foundation.data_layer.root import sql_path
from alphalattice.kernel import storage

from . import market_snapshots, world
from .world import DATASET, DISPLAY_ALIAS, LATEST, LISTING_ALIAS, _log_rows

WORKSPACE: Final = {
    "bars": "raw_daily_bar_current",
    "adjusted": "provider_adjusted_close_current",
    "actions": "corporate_action_current",
    "action_evidence": "corporate_action_current",
    "mappings": "provider_symbol_mapping",
    "bar_revisions": "bar_revision",
    "action_revisions": "corporate_action_revision",
}
"""The workspace's own market tables, by the name a value read gives each relation."""
_UPDATES: ContextVar[Mapping[str, dict[str, Callable[[str], Any]]]] = ContextVar(
    "market_updates", default=MappingProxyType({})
)
"""The market data updates open in this context, a stage's own work, by their database's
resolved path: what the outermost of each publishes at its end, each dataset's publish, the
first an update of it named. Every thread starts with a context of its own, so a Host request or
another task reads world while a stage writes; a stage's worker threads only fetch from its
provider, and the stage applies and reads on its own thread."""
_served: set[tuple[str, str]] = set()
"""The (catalog, resolved database path) pairs world serves (`_serves`): a publish is never
undone, so a pair once found stays."""


def cutoff(at: datetime) -> int:
    """A reader's UTC instant in integer microseconds; legacy naive instants are UTC."""
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    return (at.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)


@contextmanager
def reading(database: Path, *, with_market_tables: bool = True) -> Iterator[Any | None]:
    """A cursor on world for the market and sector reads of the workspace in `database`; None,
    and the workspace's own tables answer, inside an update of it this context opened
    (`writing`), with no catalog to read, and while world holds no publish of it (`_serves`). The
    cursor's statements share one transaction, so they see one catalog snapshot.
    Universe readers use their own log lookups and need none of the market relation tables."""
    if str(database.resolve()) in _UPDATES.get():
        yield None
        return
    with session() as cursor:
        if cursor is None or not _serves(cursor, database):
            yield None
            return
        if with_market_tables:
            _log_tables(cursor)
        cursor.begin()
        try:
            yield cursor
        finally:
            cursor.rollback()


def _serves(cursor: Any, database: Path) -> bool:
    """Whether world serves the workspace in `database`, a bound copy's as its owner's: once
    this machine's log records a market publish of it (world.published). The Host publishes its
    own workspace at its open; another, one no Host here has opened, holds its rows only in its
    own tables."""
    found = cursor.execute("SELECT path FROM duckdb_databases() WHERE database_name = 'lake'")
    (catalog,) = storage.first(found)
    found.complete()  # the cursor's next statement needs its result finished
    key = (catalog, str(database.resolve()))
    if key not in _served:
        owner = workspace(database)
        if owner is None or not world.published(owner):
            return False
        _served.add(key)
    return True


def writes_to_world(database: Path) -> bool:
    """Whether this database is the open Host's own workspace with a live StorePort."""
    host = history.opened_log()
    return (
        host is not None
        and history.store() is not None
        and _folder(host[1]) == database.parent.resolve()
    )


@contextmanager
def writing(
    database: Path, publish: Callable[[str], Any], *, dataset: str = DATASET
) -> Iterator[None]:
    """One update of the workspace in `database`'s `dataset`: a stage writing it. Its reads, in
    this context, the stage's own work, take the workspace's own tables, the stage's buffer;
    every other reader keeps world. An update inside another of the same file joins it. When the
    outermost ends, in the Host whose workspace it is, each dataset's `publish` publishes into
    world, as that workspace's, what the updates' writers wrote, market data first; an update
    that ends before `database` exists wrote nothing. Elsewhere what was written waits for that
    Host's next publish."""
    path = str(database.resolve())
    open_updates = _UPDATES.get()
    if path in open_updates:
        open_updates[path].setdefault(dataset, publish)
        yield
        return
    publishes = {dataset: publish}
    token = _UPDATES.set({**open_updates, path: publishes})
    try:
        yield
    finally:
        _UPDATES.reset(token)
        host = history.opened_log()
        if host is not None and database.exists() and _folder(host[1]) == database.parent.resolve():
            for name in sorted(publishes, key=lambda name: name != DATASET):
                publishes[name](host[1])


def binding(database: Path) -> str | None:
    """The research-input binding whose copy `database` is, from its place in the workspace
    (research-inputs/<binding>/source/market-data.duckdb); None for any other database."""
    source = database.parent
    name = source.parent.name
    if source.name != "source" or source.parent.parent.name != "research-inputs":
        return None
    return name if len(name) == 64 and set(name) <= set("0123456789abcdef") else None


def _copy_workspace(database: Path) -> Path | None:
    """The workspace a research-input copy belongs to: the folder holding its `research-inputs/`,
    or, for a binding kept on the data root, the workspace its view `views/<workspace_id>`
    stands for (kernel.data_layer.inputs); None for a view of a workspace the log does not know."""
    holder = database.parents[3]
    if holder.parent.name != "views":
        return holder
    return history.research_input_workspace(holder)


def selected(database: Path) -> int | None:
    """The cutoff of the selection made for the research-input copy `database`, in the workspace
    that holds it: what its research may see. None for a database that is no copy, and for a copy
    bound with no selection (outside a Host, or while world did not serve its tables), which then
    reads its own tables."""
    found = binding(database)
    if found is None:
        return None
    sql = (
        "SELECT s.cutoff_us, w.folder FROM selection s JOIN workspace w USING"
        " (workspace_id) WHERE json_extract(s.spec, '$.binding_hash') = ? ORDER BY s.made_seq"
    )
    rows = _log_rows(sql, (found,))
    workspace = _copy_workspace(database)
    for cutoff, folder in rows:  # the folder as the log records it, the copy's in its own form
        if workspace and folder and Path(folder).is_dir() and os.path.samefile(folder, workspace):
            return int(cutoff)
    return None


def market_selection_held(selection: str, catalog_snapshot: int) -> bool:
    """A binding's market selection and catalog snapshot must be recorded together."""
    return bool(
        _log_rows(
            "SELECT 1 FROM selection WHERE selection_id = ?"
            " AND json_extract(spec, '$.market_catalog_snapshot') = ?",
            (selection, catalog_snapshot),
        )
    )


def map_cutoff(map_hash: str) -> int | None:
    """The cutoff of the selection a sector map's publication recorded, its spec naming the map's
    `map_hash` (the first made, in any workspace: the map's names are world's at each); None for a
    map with none: one published before 0.2.8, or while world did not serve its names."""
    rows = _log_rows(
        "SELECT cutoff_us FROM selection WHERE json_extract(spec, '$.map_hash') = ?"
        " ORDER BY made_seq LIMIT 1",
        (map_hash,),
    )
    return int(rows[0][0]) if rows else None


COMPARED: Final = {
    "bars": (
        "raw_daily_bar_current",
        "",
        "listing_id, provider, session_date, open, high, low, close, CAST(volume AS BIGINT)",
    ),
    "adjusted": (
        "provider_adjusted_close_current",
        "",
        "listing_id, provider, session_date, adjusted_close",
    ),
    "actions": (
        "corporate_action_current",
        " WHERE status = 'ACTIVE'",
        "listing_id, provider,"
        " effective_date, action_kind, new_shares_per_old_share, cash_amount, provisional,"
        " provenance",
    ),
    "mappings": (
        "provider_symbol_mapping",
        "",
        "listing_id, provider, provider_symbol, effective_from, effective_to, lifecycle_state",
    ),
}
"""Each relation a value read names, with the workspace table, rows and columns it answers for."""


def holds(database: Path, cutoff_us: int, *, listings: Sequence[str] | None = None) -> bool:
    """Whether world, as it held the market tables at `cutoff_us`, answers each value read
    exactly as the workspace database `database` does: every relation's rows equal, both ways.
    A copy is bound to world only then; marks alone show a past admission, and world may hold
    another workspace's later value of a key since. A family upgrade names its source listing
    scope; without one, every held listing is compared as for a whole research-input copy."""
    named = relations(cutoff_us, cutoff_us, listings, workspace_id=workspace_id(database))
    with session() as cursor:
        if cursor is None:
            return False
        _log_tables(cursor)
        alias = f"bound_{binding(database) or 'database'}"
        release_workspace_database(database)  # an instance of the file this process keeps open
        cursor.execute(f"ATTACH '{sql_path(database)}' AS {alias} (READ_ONLY)").fetchall()
        try:
            has_index = storage.first(
                cursor.execute(
                    "SELECT 1 FROM duckdb_tables() WHERE database_name = ? AND table_name = ?",
                    [alias, market_snapshots.VERSIONS],
                )
            )
            committed = (
                market_snapshots.version_relations(
                    f"{alias}.main.{market_snapshots.VERSIONS}", "lake"
                )
                if has_index
                else {}
            )
            for name, (table, rows, columns) in COMPARED.items():
                pending = f"{alias}.main.{table}"
                source = pending
                if committed and table in market_snapshots.OVERLAYS:
                    view, key = market_snapshots.OVERLAYS[table]
                    projected = market_snapshots.overlay(pending, f"({committed[view]})", key)
                    source = f"({projected})"
                own = f"SELECT {columns} FROM {source}{rows}"
                held = f"SELECT {columns} FROM {named[name]}"
                differ = storage.first(
                    cursor.execute(
                        f"SELECT EXISTS (({held}) EXCEPT ALL ({own}))"
                        f" OR EXISTS (({own}) EXCEPT ALL ({held}))"
                    )
                )[0]
                if differ:
                    return False
            return True
        finally:
            cursor.execute(f"DETACH {alias}").fetchall()


def workspace(database: Path) -> str | None:
    """The registered workspace owning this database, if its folder is unambiguous."""
    folder = _copy_workspace(database) if binding(database) is not None else database.parent
    if folder is None:
        return None
    rows = _log_rows(
        "SELECT workspace_id FROM workspace WHERE folder = ? AND status = 'active'",
        (str(folder.resolve()),),
    )
    if len(rows) > 1:
        raise ValueError("market_data.workspace_identity_ambiguous: repair the workspace registry")
    return str(rows[0][0]) if rows else None


def _folder(workspace_id: str) -> Path | None:
    """The folder the log records for a workspace, resolved; None when it records none."""
    found = _log_rows("SELECT folder FROM workspace WHERE workspace_id = ?", (workspace_id,))
    return None if not found or found[0][0] is None else Path(found[0][0]).resolve()


def workspace_id(database: Path) -> str | None:
    """The log's unambiguous active workspace, including a bound copy's owner."""
    return workspace(database)


_LOG_TABLES: Final = (
    (
        "market_aliases",
        f"SELECT name, definition_id FROM definition_name WHERE family = 'provider'"
        f" AND name IN ('{LISTING_ALIAS}', '{DISPLAY_ALIAS}')",
        pa.schema([("name", pa.string()), ("definition_id", pa.string())]),
    ),
    (
        "definition_lineage",
        "SELECT definition_id, name FROM definition_lineage",
        pa.schema([("definition_id", pa.string()), ("name", pa.string())]),
    ),
    (
        "admission",
        "SELECT admission_id, known_us FROM admission",
        pa.schema([("admission_id", pa.string()), ("known_us", pa.int64())]),
    ),
    (
        "market_revisions",
        "SELECT type, workspace_id, occurred_us, recorded_us, attrs FROM event"
        " WHERE type IN ('market.bar_revision', 'market.action_revision')",
        pa.schema(
            [
                ("type", pa.string()),
                ("workspace_id", pa.string()),
                ("occurred_us", pa.int64()),
                ("recorded_us", pa.int64()),
                ("attrs", pa.string()),
            ]
        ),
    ),
)
"""The log's tables a world session's relations join, each with the read that fills it and the
schema it is registered with."""


def _log_tables(cursor: Any) -> None:
    """Register on `cursor` the log's tables its relations join (`_LOG_TABLES`), as the log holds
    them now: after any admission this read made, whose providers they name."""
    for name, sql, schema in _LOG_TABLES:
        rows = [dict(zip(schema.names, row, strict=True)) for row in _log_rows(sql, ())]
        cursor.register(name, pa.Table.from_pylist(rows, schema=schema))


def relations(
    cutoff_us: int = LATEST,
    held_us: int | None = None,
    listings: Sequence[str] | None = None,
    *,
    workspace_id: str | None = None,
) -> dict[str, str]:
    """WORKSPACE's relations as world knew them at `cutoff_us`, by the same names; with
    `held_us`, a selection's cutoff, only the rows world held by then: a 0.2.7 row's knowledge
    time is derived from its valid date, so a row admitted later can be known earlier. Of
    `listings` only when the read names them (`scope`), else of every listing world holds.
    Revision observations and payload references belong to `workspace_id`; no identity gives
    no journal observations."""
    named = scope(listings)
    return {
        "bars": bars(cutoff_us, held_us, named, workspace_id=workspace_id),
        "adjusted": adjusted(cutoff_us, held_us, named, workspace_id=workspace_id),
        "actions": actions(cutoff_us, held_us, named, workspace_id=workspace_id),
        "action_evidence": actions(
            cutoff_us, held_us, named, evidence=True, workspace_id=workspace_id
        ),
        "mappings": mappings(cutoff_us, held_us, named),
        "bar_revisions": revisions("bar_revision", cutoff_us, held_us, workspace_id=workspace_id),
        "action_revisions": revisions(
            "corporate_action_revision", cutoff_us, held_us, workspace_id=workspace_id
        ),
    }


def scope(listings: Sequence[str] | None) -> tuple[str, str] | None:
    """The 0.2.7 ids a read names, resolved to their world ids once (world.world_listing): the
    relation that names a world row's listing by its 0.2.7 id, and the filter on a world row's
    listing (`{row}` its table's alias), so a world table's scan takes only those listings. One
    listing is named by constant and filtered by its id; several are joined to the read's ids, the
    join's filter reaching the scan. None for every listing, named through world's alias of each."""
    if listings is None:
        return None
    ids = sorted(set(listings))
    if not ids:
        return "(SELECT NULL::VARCHAR AS listing_id, NULL::UUID AS world WHERE false)", ""
    old = ["'" + listing.replace("'", "''") + "'" for listing in ids]
    new = [f"'{world.world_listing(listing)}'" for listing in ids]
    if len(ids) == 1:
        return (
            f"(SELECT {old[0]} AS listing_id, {new[0]}::UUID AS world)",
            f"{{row}}.listing_id = {new[0]}",
        )
    pairs = f"unnest([{', '.join(old)}]) AS listing_id, unnest([{', '.join(new)}])::UUID AS world"
    return f"(SELECT {pairs})", ""


def _asof(table: str, cutoff_us: int, held_us: int | None, *, versions: bool = False) -> str:
    """`table`'s versions world knew at `cutoff_us` (views.sql's `<table>_asof`), or every
    version with when it was replaced (`<table>_versions`, `versions`); with `held_us`, as world
    held them then: a version is replaced only by an admission committed by `held_us`, since a
    row is known at its writer's observed time, which a publish after `held_us` may precede."""
    if held_us is None:
        return f"{table}_versions" if versions else f"{table}_asof({cutoff_us})"
    held = (
        f"SELECT * FROM lake.world.superseded WHERE table_id = world_table_id('{table}')"
        f" AND by_admission_id IN (SELECT admission_id FROM admission WHERE known_us <= {held_us})"
    )
    if versions:
        return (
            f"(SELECT r.*, s.superseded_asof AS visible_to FROM lake.world.{table} r"
            f" LEFT JOIN ({held}) s ON s.version_id = r.version_id"
            " WHERE s.superseded_asof IS DISTINCT FROM TIMESTAMPTZ '-infinity')"
        )
    at = f"make_timestamptz({cutoff_us})"
    return (
        f"(SELECT r.* FROM lake.world.{table} r ANTI JOIN ({held} AND superseded_asof <= {at}) s"
        f" ON s.version_id = r.version_id WHERE r.known_asof <= {at} AND NOT r.retracted)"
    )


def _held_by(row: str, held_us: int | None) -> str:
    """The condition that a world row's admission was committed by `held_us`; true without it."""
    if held_us is None:
        return "true"
    admitted = f"SELECT admission_id FROM admission WHERE known_us <= {held_us}"
    return f"{row}.admission_id IN ({admitted})"


def _named(row: str, cutoff_us: int, held_us: int | None, named: tuple[str, str] | None) -> str:
    """The joins that name a world row's listing by its 0.2.7 id and its provider by name, and the
    filter on the row: its admission held by `held_us`, and its listing, the read's own ids when it
    names its listings, else world's alias of every listing, read at LATEST (an identity)."""
    provider = f" JOIN definition_lineage named ON named.definition_id = {row}.provider_id"
    where = f" WHERE {_held_by(row, held_us)}"
    if named is None:
        return (
            f" JOIN (SELECT listing_id AS world, symbol AS listing_id FROM"
            f" provider_symbol_asof({LATEST}) WHERE provider_id = "
            f"(SELECT definition_id FROM market_aliases WHERE name = '{LISTING_ALIAS}')) alias"
            f" ON alias.world = {row}.listing_id{provider}{where}"
        )
    ids, within = named
    if within:
        return f" CROSS JOIN {ids} alias{provider}{where} AND {within.format(row=row)}"
    return f" JOIN {ids} alias ON alias.world = {row}.listing_id{provider}{where}"


def revisions(
    table: str,
    cutoff_us: int = LATEST,
    held_us: int | None = None,
    *,
    workspace_id: str | None = None,
) -> str:
    """This workspace's original journal observations at the cutoff; none without its identity."""
    type_, columns = world.REVISIONS[table]
    fields = []
    for name in columns:
        type_sql = (
            "TIMESTAMP"
            if name == "observed_at"
            else ("DATE" if name in {"session_date", "effective_date"} else "VARCHAR")
        )
        fields.append(f"CAST(json_extract_string(attrs, '$.{name}') AS {type_sql}) AS {name}")
    held = "" if held_us is None else f" AND recorded_us <= {held_us}"
    workspace = "NULL" if workspace_id is None else "'" + workspace_id.replace("'", "''") + "'"
    return (
        f"(SELECT {', '.join(fields)} FROM market_revisions WHERE type = '{type_}'"
        f" AND workspace_id = {workspace} AND occurred_us <= {cutoff_us}{held})"
    )


def _payload(
    table: str, row: str, dated: str, cutoff_us: int, held_us: int | None, workspace_id: str | None
) -> str:
    """A current payload reference, including an equal-value rewrite's original revision fact."""
    kind = f" AND e.action_kind = {row}.action_id" if table == "corporate_action_revision" else ""
    rows = revisions(table, cutoff_us, held_us, workspace_id=workspace_id)
    return (
        f"coalesce((SELECT e.next_payload_hash FROM {rows} e"
        " WHERE e.listing_id = alias.listing_id AND e.provider = named.name"
        f" AND e.{dated} = {row}.{dated}{kind}"
        f" ORDER BY e.observed_at DESC, e.revision_id DESC LIMIT 1), {row}.source_payload_digest)"
    )


def bars(
    cutoff_us: int = LATEST,
    held_us: int | None = None,
    named: tuple[str, str] | None = None,
    *,
    workspace_id: str | None = None,
) -> str:
    """0.2.7's raw_daily_bar_current with its provider's adjusted close, as world knew it at
    `cutoff_us`: listing_id, provider, session_date, open, high, low, close, volume and
    adjusted_close and the original payload reference for audit evidence."""
    payload = _payload("bar_revision", "r", "session_date", cutoff_us, held_us, workspace_id)
    return (
        "(SELECT alias.listing_id, named.name AS provider, r.session_date, r.open,"
        " r.high, r.low, r.close, CAST(r.volume AS BIGINT) AS volume, r.adjusted_close,"
        f" {payload} AS payload_hash"
        f" FROM {_asof('raw_bar', cutoff_us, held_us)} r{_named('r', cutoff_us, held_us, named)})"
    )


def adjusted(
    cutoff_us: int = LATEST,
    held_us: int | None = None,
    named: tuple[str, str] | None = None,
    *,
    workspace_id: str | None = None,
) -> str:
    """0.2.7's provider_adjusted_close_current, as world knew it at `cutoff_us`: listing_id,
    provider, session_date and adjusted_close, where the bar's provider gave one."""
    source = bars(cutoff_us, held_us, named, workspace_id=workspace_id)
    return (
        "(SELECT listing_id, provider, session_date, adjusted_close"
        f" FROM {source} WHERE adjusted_close IS NOT NULL)"
    )


def actions(
    cutoff_us: int = LATEST,
    held_us: int | None = None,
    named: tuple[str, str] | None = None,
    *,
    evidence: bool = False,
    workspace_id: str | None = None,
) -> str:
    """0.2.7's active rows of corporate_action_current, as world knew them at `cutoff_us`:
    listing_id, provider, effective_date, action_kind, new_shares_per_old_share, cash_amount,
    provisional, provenance, status and payload reference. Normal reads hide withdrawals;
    audit evidence keeps the cutoff-visible withdrawal, as the original receipt's hash does."""
    source = _asof("corporate_action", cutoff_us, held_us, versions=evidence)
    payload = _payload(
        "corporate_action_revision", "c", "effective_date", cutoff_us, held_us, workspace_id
    )
    visible = (
        f" AND c.known_asof <= make_timestamptz({cutoff_us})"
        f" AND (c.visible_to IS NULL OR c.visible_to > make_timestamptz({cutoff_us}))"
        if evidence
        else ""
    )
    return (
        "(SELECT alias.listing_id, named.name AS provider, c.effective_date,"
        " c.action_id AS action_kind, c.ratio AS new_shares_per_old_share,"
        " c.amount AS cash_amount, c.provisional, c.provenance,"
        " CASE WHEN c.retracted THEN 'RETRACTED' ELSE 'ACTIVE' END AS status,"
        f" {payload} AS payload_hash"
        f" FROM {source} c{_named('c', cutoff_us, held_us, named)}{visible})"
    )


def mappings(
    cutoff_us: int = LATEST, held_us: int | None = None, named: tuple[str, str] | None = None
) -> str:
    """0.2.7's provider_symbol_mapping, as world knew it at `cutoff_us`: listing_id, provider,
    provider_symbol, effective_from, effective_to and the source's lifecycle state; world's own
    aliases are not mappings. A closed interval's opening version remains active before its end."""
    return (
        "(SELECT alias.listing_id, named.name AS provider, m.symbol AS provider_symbol,"
        " m.valid_from AS effective_from, m.valid_to AS effective_to, m.lifecycle_state"
        f" FROM {_asof('provider_symbol', cutoff_us, held_us)} m"
        f"{_named('m', cutoff_us, held_us, named)}"
        " AND m.provider_id NOT IN (SELECT definition_id FROM market_aliases))"
    )
