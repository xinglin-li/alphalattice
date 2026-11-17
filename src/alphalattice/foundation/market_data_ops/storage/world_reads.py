"""Market value reads from world: the setting that chooses them, the session that serves them,
and 0.2.7's market tables as world's as-of reads, in 0.2.7's names and columns.

`ALPHALATTICE_MARKET_READS` is `world`, the default, or `workspace`, the rollback; a Task child
inherits it. World serves a workspace's reads while it holds that workspace's market tables as
they are: their marks, the content fingerprint world.admit records, equal one of its admissions'
(`reading`). The Host admits its own workspace's tables first when they have changed; a Task
child, which may not write the catalog, reads the workspace's own tables until the Host has, and
so does a process whose data root never admitted the workspace. Inside a market data update
(`writing`) the writer's reads take the workspace's tables, and the Host admits once at its end.
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
from collections.abc import Callable, Iterator
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
    `marks` equal those of one of its market admissions. A verdict is kept for its catalog and the
    database's file and WAL as they stand (`store_fingerprint`); one that world does not serve is
    read again once the catalog has a newer snapshot, since an admission may have brought the
    tables meanwhile. In the Host, tables of its own workspace that no admission holds are
    admitted first, unless this process holds the file open in an instance, which an
    admission's attach would meet; that read takes the workspace, and the next admits. The
    cursor's statements share one transaction, so they see one catalog snapshot."""
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


def _snapshot(cursor: Any) -> int:
    """The catalog's current snapshot: a new one is committed by every admission."""
    return int(cursor.execute("SELECT * FROM lake.current_snapshot()").fetchone()[0])


def _admitted(cursor: Any, marks: dict[str, Any], host: tuple[Any, str] | None) -> bool:
    """Whether one of world's market admissions recorded these marks: read from the Host's own
    log, its open batch included, or else from the log the session attached."""
    sql = (  # SQLite's json_extract gives text, as DuckDB's json_extract_string does
        "SELECT {0}(attrs, '$.marks') FROM {1}event WHERE type = 'admission'"
        " AND {0}(attrs, '$.dataset') = ?"
    )
    if host is not None:
        admitted = host[0].read(sql.format("json_extract", ""), (DATASET,))
    else:
        admitted = cursor.execute(sql.format("json_extract_string", "log."), [DATASET]).fetchall()
    return bool(marks) and marks in [json.loads(row[0]) for row in admitted if row[0]]


def _folder(cursor: Any, workspace_id: str) -> Path | None:
    """The folder the log records for a workspace, resolved; None when it records none."""
    found = cursor.execute(
        "SELECT folder FROM log.workspace WHERE workspace_id = ?", [workspace_id]
    ).fetchone()
    return None if found is None or found[0] is None else Path(found[0]).resolve()


def relations(cutoff_us: int = LATEST) -> dict[str, str]:
    """WORKSPACE's relations as world knew them at `cutoff_us`, by the same names."""
    return {
        "bars": bars(cutoff_us),
        "adjusted": adjusted(cutoff_us),
        "actions": actions(cutoff_us),
        "mappings": mappings(cutoff_us),
    }


def _named(row: str, cutoff_us: int) -> str:
    """The joins that name a world row's listing by its 0.2.7 id and its provider by name."""
    return (
        f" JOIN provider_symbol_asof({cutoff_us}) alias ON alias.listing_id = {row}.listing_id"
        f" AND alias.provider_id = '{_ALIAS}'"
        f" JOIN log.definition_lineage named ON named.definition_id = {row}.provider_id"
    )


def bars(cutoff_us: int = LATEST) -> str:
    """0.2.7's raw_daily_bar_current with its provider's adjusted close, as world knew it at
    `cutoff_us`: listing_id, provider, session_date, open, high, low, close, volume and
    adjusted_close."""
    return (
        "(SELECT alias.symbol AS listing_id, named.name AS provider, r.session_date, r.open,"
        " r.high, r.low, r.close, CAST(r.volume AS BIGINT) AS volume, r.adjusted_close"
        f" FROM raw_bar_asof({cutoff_us}) r{_named('r', cutoff_us)})"
    )


def adjusted(cutoff_us: int = LATEST) -> str:
    """0.2.7's provider_adjusted_close_current, as world knew it at `cutoff_us`: listing_id,
    provider, session_date and adjusted_close, where the bar's provider gave one."""
    return (
        "(SELECT listing_id, provider, session_date, adjusted_close"
        f" FROM {bars(cutoff_us)} WHERE adjusted_close IS NOT NULL)"
    )


def actions(cutoff_us: int = LATEST) -> str:
    """0.2.7's active rows of corporate_action_current, as world knew them at `cutoff_us`:
    listing_id, provider, effective_date, action_kind, new_shares_per_old_share, cash_amount,
    provisional, provenance and status. A retracted action is world's withdrawal, so no row."""
    return (
        "(SELECT alias.symbol AS listing_id, named.name AS provider, c.effective_date,"
        " c.action_id AS action_kind, c.ratio AS new_shares_per_old_share,"
        " c.amount AS cash_amount, c.provisional, c.provenance, 'ACTIVE' AS status"
        f" FROM corporate_action_asof({cutoff_us}) c{_named('c', cutoff_us)})"
    )


def mappings(cutoff_us: int = LATEST) -> str:
    """0.2.7's provider_symbol_mapping, as world knew it at `cutoff_us`: listing_id, provider,
    provider_symbol, effective_from and effective_to; world's own aliases are not mappings."""
    return (
        "(SELECT alias.symbol AS listing_id, named.name AS provider, m.symbol AS provider_symbol,"
        " m.valid_from AS effective_from, m.valid_to AS effective_to"
        f" FROM provider_symbol_asof({cutoff_us}) m{_named('m', cutoff_us)}"
        f" WHERE m.provider_id NOT IN {_ALIASES})"
    )
