"""Market value reads from world: the setting that chooses them, the session that serves them,
and 0.2.7's market tables as world's as-of reads, in 0.2.7's names and columns.

`ALPHALATTICE_MARKET_READS` is `world`, the default, or `workspace`, the rollback; a Task child
inherits it. World serves a workspace's reads while it holds that workspace's market tables as
they are: their marks, the content fingerprint world.admit records, equal those of an admission
that took every row they mark (`reading`). The Host admits its own workspace's tables first when
they have changed; a Task child, which may not write the catalog, reads the workspace's own
tables until the Host has, and so does a process whose data root never admitted the workspace.
Inside a market data update (`writing`) the writer's reads take the workspace's tables, and the
Host admits once at its end.
A world read runs on `session`: in the Host a cursor on its StorePort's session, and in any other
process the data root's reading surface attached read-only (`reading_session`), opened once.
Each relation reads world as of a cutoff in microseconds: a selection's, or `LATEST`, all that is
known. A listing is named by its 0.2.7 id, which world keeps as its alias (world.LISTING_ALIAS),
and a provider by its lineage's name.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from alphalattice.control.workspace_runtime.database import (
    live_workspace_connections,
    store_fingerprint,
)
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.project import definition_id
from alphalattice.kernel.data_layer.reading import reading_session
from alphalattice.kernel.data_layer.root import sql_path

from . import world
from .world import DATASET, DISPLAY_ALIAS, LISTING_ALIAS

SETTING: Final = "ALPHALATTICE_MARKET_READS"
CHOICES: Final = ("world", "workspace")
"""Where market value reads come from: world's as-of reads, or the workspace's own tables."""
LATEST: Final = 253_402_300_799_999_999
"""A read's own moment as a cutoff, the end of year 9999: all that is known when it runs, since
no knowledge time is later than the read."""
_ALIAS: Final = definition_id("provider", LISTING_ALIAS)
_ALIASES: Final = f"('{_ALIAS}', '{definition_id('provider', DISPLAY_ALIAS)}')"
WORKSPACE: Final = {
    "bars": "raw_daily_bar_current",
    "adjusted": "provider_adjusted_close_current",
    "actions": "corporate_action_current",
    "mappings": "provider_symbol_mapping",
}
"""The workspace's own market tables, by the name a value read gives each relation."""
_held: dict[Path, Any] = {}
_served: dict[tuple[str, str, str], tuple[bool, int]] = {}
"""Each verdict, by catalog, database and its file's fingerprint: whether world serves it, and
the catalog snapshot it was reached at."""
_writing: dict[str, int] = {}
_lock = threading.Lock()


def chosen() -> str:
    """The setting's value; refused by name when it names neither choice."""
    value = os.environ.get(SETTING, CHOICES[0])
    if value not in CHOICES:
        raise DataError(
            f"{SETTING} is {value!r}; set it to world or workspace, or unset it for world",
            code="market_data.market_reads_unknown",
        )
    return value


@contextmanager
def session() -> Iterator[Any | None]:
    """A cursor on world's reading surface, closed after the block; None when this process's data
    root has no catalog yet. The Host's is its StorePort's session, which sees its own commits;
    another process attaches the catalog read-only once and takes a cursor per read."""
    port = history.store()
    if port is not None:
        cursor = port.lake.cursor()
    else:
        root = history.data_root()
        with _lock:
            surface = _held.get(root)
            if surface is None and (root / "machine" / "catalog.sqlite").is_file():
                surface = _held[root] = reading_session(root)
        cursor = None if surface is None else surface.cursor()
    try:
        yield cursor
    finally:
        if cursor is not None:
            cursor.close()


@contextmanager
def reading(database: Path, marks: Callable[[], dict[str, Any]]) -> Iterator[Any | None]:
    """A cursor on world while it serves the market reads of the workspace in `database`, else
    None and its own tables answer. World serves when the setting is world and the tables'
    `marks` equal those of a market admission that took every row (`_admitted`). A verdict is
    kept for its catalog and the database's file and WAL as they stand (`store_fingerprint`); one
    that world does not serve is read again once the catalog has a newer snapshot, since an
    admission may have brought the tables meanwhile. In the Host, tables of its own workspace that
    no admission holds are admitted first, unless this process holds the file open in an
    instance, which an admission's attach would meet; that read takes the workspace, and the next
    admits. The cursor's statements share one transaction, so they see one catalog snapshot."""
    path = str(database.resolve())
    if chosen() != "world" or path in _writing:
        yield None
        return
    with session() as cursor:
        if cursor is None:
            yield None
            return
        catalog = cursor.execute(
            "SELECT path FROM duckdb_databases() WHERE database_name = 'lake'"
        ).fetchone()[0]
        key = (catalog, path, json.dumps(store_fingerprint(database)))
        verdict = _served.get(key)
        if verdict is not None and not verdict[0] and verdict[1] != _snapshot(cursor):
            verdict = None  # world has committed since it did not serve these tables
        if verdict is None:
            current = marks()
            host = history.opened_log()
            served = _admitted(cursor, current, host)
            ours = host is not None and _folder(cursor, host[1]) == database.parent
            # an admission attaches the file, which an instance this process holds open refuses
            held = ours and live_workspace_connections(database) is not None
            if not served and ours and not held:
                world.admit(database, host[1])
                served = _admitted(cursor, current, host)
            verdict = (served, _snapshot(cursor))
            if served or not held:  # a held file's admission is owed to the next read
                _served[key] = verdict
        served = verdict[0]
        # contract: the workspace's own tables answer while world lacks their current content;
        # remove once the market data writers commit into world through StorePort staging as they
        # write, and the workspace setting is retired with the workspace's market tables.
        if not served:
            yield None
            return
        cursor.begin()
        try:
            yield cursor
        finally:
            cursor.rollback()


@contextmanager
def writing(database: Path, marks: Callable[[], dict[str, Any]]) -> Iterator[None]:
    """One market data update of the workspace in `database`, in this process. Its reads, every
    thread's, take the workspace's own tables, reading no marks and admitting nothing, so a loop
    that writes and reads listing by listing pays no admission per listing. When the outermost
    update ends, the Host admits the tables once, as `reading` would at its next read."""
    path = str(database.resolve())
    with _lock:
        _writing[path] = _writing.get(path, 0) + 1
    try:
        yield
    finally:
        with _lock:
            _writing[path] -= 1
            outermost = not _writing[path]
            if outermost:
                del _writing[path]
        if outermost and history.opened_log() is not None:
            with reading(database, marks):
                pass


def binding(database: Path) -> str | None:
    """The research-input binding whose copy `database` is, from its place in the workspace
    (research-inputs/<binding>/source/market-data.duckdb); None for any other database."""
    source = database.parent
    name = source.parent.name
    if source.name != "source" or source.parent.parent.name != "research-inputs":
        return None
    return name if len(name) == 64 and set(name) <= set("0123456789abcdef") else None


def selected(database: Path) -> int | None:
    """The cutoff of the selection made for the research-input copy `database`, in the workspace
    that holds it: what its research may see. None for a database that is no copy, and for a copy
    bound with no selection (outside a Host, or while world did not serve its tables), which then
    reads its own tables."""
    found = binding(database)
    if found is None:
        return None
    sql = (
        "SELECT s.cutoff_us, w.folder FROM {1}selection s JOIN {1}workspace w USING"
        " (workspace_id) WHERE {0}(s.spec, '$.binding_hash') = ? ORDER BY s.made_seq"
    )
    host = history.opened_log()
    if host is not None:  # the Host's own log, its open batch included
        rows = [tuple(row) for row in host[0].read(sql.format("json_extract", ""), (found,))]
    else:
        with session() as cursor:
            if cursor is None:
                return None
            rows = cursor.execute(sql.format("json_extract_string", "log."), [found]).fetchall()
    workspace = database.parents[3]
    for cutoff, folder in rows:  # the folder as the log records it, the copy's in its own form
        if folder and Path(folder).is_dir() and os.path.samefile(folder, workspace):
            return int(cutoff)
    return None


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
        "listing_id, provider, provider_symbol, effective_from, effective_to",
    ),
}
"""Each relation a value read names, with the workspace table, rows and columns it answers for."""


def holds(database: Path, cutoff_us: int) -> bool:
    """Whether world, as it held the market tables at `cutoff_us`, answers each value read
    exactly as the workspace database `database` does: every relation's rows equal, both ways.
    A copy is bound to world only then; marks alone show a past admission, and world may hold
    another workspace's later value of a key since."""
    named = relations(cutoff_us, cutoff_us)
    with session() as cursor:
        if cursor is None:
            return False
        alias = f"bound_{binding(database) or 'database'}"
        cursor.execute(f"ATTACH '{sql_path(database)}' AS {alias} (READ_ONLY)").fetchall()
        try:
            for name, (table, rows, columns) in COMPARED.items():
                own = f"SELECT {columns} FROM {alias}.{table}{rows}"
                held = f"SELECT {columns} FROM {named[name]}"
                differ = cursor.execute(
                    f"SELECT EXISTS (({held}) EXCEPT ALL ({own}))"
                    f" OR EXISTS (({own}) EXCEPT ALL ({held}))"
                ).fetchone()[0]
                if differ:
                    return False
            return True
        finally:
            cursor.execute(f"DETACH {alias}").fetchall()


def _snapshot(cursor: Any) -> int:
    """The catalog's current snapshot: a new one is committed by every admission."""
    return int(cursor.execute("SELECT * FROM lake.current_snapshot()").fetchone()[0])


def _admitted(cursor: Any, marks: dict[str, Any], host: tuple[Any, str] | None) -> bool:
    """Whether one of world's market admissions that took every row it marks (`taken`) recorded
    these marks: read from the Host's own log, its open batch included, or else from the log the
    session attached."""
    sql = (  # SQLite's json_extract gives text, as DuckDB's json_extract_string does
        "SELECT {0}(attrs, '$.marks'), {0}(attrs, '$.dropped.equal_time') FROM {1}event"
        " WHERE type = 'admission' AND {0}(attrs, '$.dataset') = ?"
    )
    if host is not None:
        admitted = host[0].read(sql.format("json_extract", ""), (DATASET,))
    else:
        admitted = cursor.execute(sql.format("json_extract_string", "log."), [DATASET]).fetchall()
    return bool(marks) and marks in taken(admitted)


def taken(admissions: Iterable[Sequence[Any]]) -> list[dict[str, Any]]:
    """The marks of the admissions, each as (marks, equal-time drops), that took every row they
    mark. One that dropped a row written at the same time as world's version of its key
    (`equal_time`) left world holding that other value; an older row's drop leaves world the
    newer one, which it serves."""
    return [json.loads(marks) for marks, equal in admissions if marks and equal is None]


def _folder(cursor: Any, workspace_id: str) -> Path | None:
    """The folder the log records for a workspace, resolved; None when it records none."""
    found = cursor.execute(
        "SELECT folder FROM log.workspace WHERE workspace_id = ?", [workspace_id]
    ).fetchone()
    return None if found is None or found[0] is None else Path(found[0]).resolve()


def relations(cutoff_us: int = LATEST, held_us: int | None = None) -> dict[str, str]:
    """WORKSPACE's relations as world knew them at `cutoff_us`, by the same names; with
    `held_us`, a selection's cutoff, only the rows world held by then: a 0.2.7 row's knowledge
    time is derived from its valid date, so a row admitted later can be known earlier."""
    return {
        "bars": bars(cutoff_us, held_us),
        "adjusted": adjusted(cutoff_us, held_us),
        "actions": actions(cutoff_us, held_us),
        "mappings": mappings(cutoff_us, held_us),
    }


def _held_by(row: str, held_us: int | None) -> str:
    """The condition that a world row's admission was committed by `held_us`; true without it."""
    if held_us is None:
        return "true"
    admitted = f"SELECT admission_id FROM log.admission WHERE known_us <= {held_us}"
    return f"{row}.admission_id IN ({admitted})"


def _named(row: str, cutoff_us: int) -> str:
    """The joins that name a world row's listing by its 0.2.7 id and its provider by name."""
    return (
        f" JOIN provider_symbol_asof({cutoff_us}) alias ON alias.listing_id = {row}.listing_id"
        f" AND alias.provider_id = '{_ALIAS}'"
        f" JOIN log.definition_lineage named ON named.definition_id = {row}.provider_id"
    )


def bars(cutoff_us: int = LATEST, held_us: int | None = None) -> str:
    """0.2.7's raw_daily_bar_current with its provider's adjusted close, as world knew it at
    `cutoff_us`: listing_id, provider, session_date, open, high, low, close, volume and
    adjusted_close."""
    return (
        "(SELECT alias.symbol AS listing_id, named.name AS provider, r.session_date, r.open,"
        " r.high, r.low, r.close, CAST(r.volume AS BIGINT) AS volume, r.adjusted_close"
        f" FROM raw_bar_asof({cutoff_us}) r{_named('r', cutoff_us)}"
        f" WHERE {_held_by('r', held_us)})"
    )


def adjusted(cutoff_us: int = LATEST, held_us: int | None = None) -> str:
    """0.2.7's provider_adjusted_close_current, as world knew it at `cutoff_us`: listing_id,
    provider, session_date and adjusted_close, where the bar's provider gave one."""
    return (
        "(SELECT listing_id, provider, session_date, adjusted_close"
        f" FROM {bars(cutoff_us, held_us)} WHERE adjusted_close IS NOT NULL)"
    )


def actions(cutoff_us: int = LATEST, held_us: int | None = None) -> str:
    """0.2.7's active rows of corporate_action_current, as world knew them at `cutoff_us`:
    listing_id, provider, effective_date, action_kind, new_shares_per_old_share, cash_amount,
    provisional, provenance and status. A retracted action is world's withdrawal, so no row."""
    return (
        "(SELECT alias.symbol AS listing_id, named.name AS provider, c.effective_date,"
        " c.action_id AS action_kind, c.ratio AS new_shares_per_old_share,"
        " c.amount AS cash_amount, c.provisional, c.provenance, 'ACTIVE' AS status"
        f" FROM corporate_action_asof({cutoff_us}) c{_named('c', cutoff_us)}"
        f" WHERE {_held_by('c', held_us)})"
    )


def mappings(cutoff_us: int = LATEST, held_us: int | None = None) -> str:
    """0.2.7's provider_symbol_mapping, as world knew it at `cutoff_us`: listing_id, provider,
    provider_symbol, effective_from and effective_to; world's own aliases are not mappings."""
    return (
        "(SELECT alias.symbol AS listing_id, named.name AS provider, m.symbol AS provider_symbol,"
        " m.valid_from AS effective_from, m.valid_to AS effective_to"
        f" FROM provider_symbol_asof({cutoff_us}) m{_named('m', cutoff_us)}"
        f" WHERE m.provider_id NOT IN {_ALIASES} AND {_held_by('m', held_us)})"
    )
