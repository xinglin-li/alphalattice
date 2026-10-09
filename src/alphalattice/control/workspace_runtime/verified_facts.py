"""What a workspace has verified once, recorded so that no later read computes it again.

- A **year fact** vouches for one calendar year of a market-store table's rows. It sits in the
  store (`verified_year_fact`) and answers only under the store's seal epoch
  (`WorkspaceDatabase.seal_epoch`) and the reader's basis (the rule or view it was made under).
  The rows' writers forget or renew the years they touch in their own transactions, so only an
  edit none of them made could leave a fact stale, and the epoch ends every fact then.
- A **file fact** is a file's digest while the file keeps its path, size, modification time and
  file id (decision 5). Under a workspace whose writer lease this process holds it is kept across
  processes in `runtime/verified-files.jsonl`; elsewhere it lives for the process. A reader's
  digest of a file modified within the last two seconds is not kept (a write in the same clock
  tick could leave the stamp unchanged); a writer's own digest of what it wrote is.

A missing fact only costs the full computation: both records are rebuildable caches, and a
store that predates the year-fact table holds none until its first writer creates it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import weakref
from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import Lock

import duckdb

FILE_FACTS = Path("runtime") / "verified-files.jsonl"
RACY_SECONDS = 2.0
"""How old a file's modification time must be before a reader's digest of it is kept."""

_Key = tuple[str, str]
_Entry = tuple[tuple[int, int, int], str]
_LOCK = Lock()
_KEPT: dict[Path, dict[_Key, _Entry]] = {}
"""Per workspace whose writer lease this process holds: its file facts by path and kind."""
_PROCESS: dict[_Key, _Entry] = {}
"""File facts outside a kept workspace, by absolute path and kind, for this process."""
_HELD: weakref.WeakKeyDictionary[duckdb.DuckDBPyConnection, bool] = weakref.WeakKeyDictionary()
"""Connections that have seen the year-fact table: one catalog read each, not one per call."""


def ensure_year_fact_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Create the store's year-fact table; its schema owners call this beside their own."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS verified_year_fact (
            kind VARCHAR NOT NULL, scope VARCHAR NOT NULL, year INTEGER NOT NULL,
            basis VARCHAR NOT NULL, epoch VARCHAR NOT NULL, value VARCHAR NOT NULL,
            PRIMARY KEY (kind, scope, year)
        )
        """
    )


def _year_facts_held(connection: duckdb.DuckDBPyConnection) -> bool:
    # Asked of the catalog, never by a failed read: a failed statement ends the caller's
    # transaction.
    if connection in _HELD:
        return True
    held = connection.execute(
        "SELECT count(*) FROM duckdb_tables() WHERE table_name = 'verified_year_fact'"
    ).fetchone()
    if held is None or not held[0]:
        return False
    with _LOCK:
        _HELD[connection] = True
    return True


def year_facts(
    connection: duckdb.DuckDBPyConnection,
    *,
    kind: str,
    scope: str,
    basis: str,
    epoch: str,
    years: Iterable[int],
) -> dict[int, str]:
    """The recorded values of `scope`'s years that still vouch under `basis` and `epoch`."""
    wanted = sorted(set(years))
    if not wanted or not _year_facts_held(connection):
        return {}
    rows = connection.execute(
        "SELECT year, value FROM verified_year_fact WHERE kind = ? AND scope = ? AND basis = ? "
        "AND epoch = ? AND list_contains(?::INTEGER[], year)",
        [kind, scope, basis, epoch, wanted],
    ).fetchall()
    return {int(year): str(value) for year, value in rows}


def record_year_facts(
    connection: duckdb.DuckDBPyConnection,
    *,
    kind: str,
    scope: str,
    basis: str,
    epoch: str,
    values: Mapping[int, str],
) -> None:
    """Record `scope`'s year values, verified now, in the caller's transaction."""
    if values:
        if not _year_facts_held(connection):
            ensure_year_fact_schema(connection)
        connection.executemany(
            "INSERT OR REPLACE INTO verified_year_fact VALUES (?, ?, ?, ?, ?, ?)",
            [[kind, scope, int(year), basis, epoch, value] for year, value in values.items()],
        )


def forget_year_facts(
    connection: duckdb.DuckDBPyConnection,
    kind: str,
    *,
    years: Iterable[int] | None = None,
    pairs: Iterable[tuple[str, int]] | None = None,
) -> None:
    """End the facts a write touches, in the writer's transaction.

    `years` ends those years of every scope, `pairs` those (scope, year) blocks, and neither
    ends every fact of the kind.
    """
    if not _year_facts_held(connection):
        return
    if years is None and pairs is None:
        connection.execute("DELETE FROM verified_year_fact WHERE kind = ?", [kind])
    if years is not None and (wanted := sorted(set(years))):
        connection.execute(
            "DELETE FROM verified_year_fact WHERE kind = ? AND list_contains(?::INTEGER[], year)",
            [kind, wanted],
        )
    if pairs is not None and (blocks := sorted(set(pairs))):
        connection.executemany(
            "DELETE FROM verified_year_fact WHERE kind = ? AND scope = ? AND year = ?",
            [[kind, scope, year] for scope, year in blocks],
        )


def keep_file_facts(workspace: Path) -> None:
    """Keep file facts for `workspace` across processes; its writer lease's holder calls this."""
    root = Path(os.path.abspath(workspace))
    entries: dict[_Key, _Entry] = {}
    try:
        lines = (root / FILE_FACTS).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        lines = []
    for line in lines:
        try:
            path, kind, size, mtime, ino, value = json.loads(line)
            entries[(str(path), str(kind))] = ((int(size), int(mtime), int(ino)), str(value))
        except (ValueError, TypeError):
            continue  # a damaged line costs that file one full digest
    if len(lines) > 2 * len(entries) + 1024:
        # Compacted: the last line per file, and only for a file still as it was recorded.
        entries = {key: entry for key, entry in entries.items() if _holds(root / key[0], entry)}
        staged = (root / FILE_FACTS).with_name(f"{FILE_FACTS.name}.{os.getpid()}.partial")
        try:
            staged.write_text("".join(map(_line, entries.items())), encoding="utf-8")
            os.replace(staged, root / FILE_FACTS)
        except OSError:
            staged.unlink(missing_ok=True)
    with _LOCK:
        _KEPT[root] = entries


def release_file_facts(workspace: Path) -> None:
    """Stop keeping `workspace`'s file facts in this process (its writer lease closed)."""
    with _LOCK:
        _KEPT.pop(Path(os.path.abspath(workspace)), None)


def file_fact(path: Path, kind: str) -> str | None:
    """The digest of `kind` recorded for `path`, while the file is as it was then."""
    entries, key = _place(path, kind)
    with _LOCK:
        held = entries.get(key)
    return held[1] if held is not None and _holds(path, held) else None


def record_file_fact(
    path: Path, kind: str, value: str, *, written: bool = False, stat: os.stat_result | None = None
) -> None:
    """Remember `path`'s digest of `kind`: verified by a reader, or `written` by its writer."""
    stat = stat if stat is not None else os.stat(path)
    if not written and time.time() - stat.st_mtime <= RACY_SECONDS:
        return
    entry = ((stat.st_size, stat.st_mtime_ns, stat.st_ino), value)
    entries, key = _place(path, kind)
    with _LOCK:
        if entries.get(key) == entry:
            return
        entries[key] = entry
        root = next((root for root, held in _KEPT.items() if held is entries), None)
        if root is not None:
            try:
                (root / FILE_FACTS).parent.mkdir(parents=True, exist_ok=True)
                with (root / FILE_FACTS).open("a", encoding="utf-8") as log:
                    log.write(_line((key, entry)))
            except OSError:
                pass  # kept for this process; the next one digests the file again


def file_sha256(path: Path) -> str:
    """A file's SHA-256: its fact, or hashed whole and remembered if the file held still."""
    held = file_fact(path, "sha256")
    if held is not None:
        return held
    before = os.stat(path)
    with open(path, "rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if _holds(path, ((before.st_size, before.st_mtime_ns, before.st_ino), digest)):
        record_file_fact(path, "sha256", digest, stat=before)
    return digest


def _holds(path: Path, entry: _Entry) -> bool:
    try:
        stat = os.stat(path)
    except OSError:
        return False
    return (stat.st_size, stat.st_mtime_ns, stat.st_ino) == entry[0]


def _place(path: Path, kind: str) -> tuple[dict[_Key, _Entry], _Key]:
    absolute = Path(os.path.abspath(path))
    with _LOCK:
        for root, entries in _KEPT.items():
            if absolute.is_relative_to(root):
                return entries, (absolute.relative_to(root).as_posix(), kind)
    return _PROCESS, (str(absolute), kind)


def _line(item: tuple[_Key, _Entry]) -> str:
    (path, kind), (stamp, value) = item
    return json.dumps([path, kind, *stamp, value], separators=(",", ":")) + "\n"
