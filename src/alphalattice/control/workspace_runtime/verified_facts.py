"""What a workspace has verified once, recorded so that no later read computes it again.

- A **year fact** vouches for one calendar year of a market-store table's rows. It sits in the
  store (`verified_year_fact`) and answers only under the store's seal epoch
  (`WorkspaceDatabase.seal_epoch`) and the reader's basis (the rule or view it was made under).
  The rows' writers forget or renew the years they touch in their own transactions, so only an
  edit none of them made could leave a fact stale, and the epoch ends every fact then.
- A **file fact** binds a verified value to its kind and the file's SHA-256. xxh3 indexes the
  complete canonical key; a hit compares that key. Under a workspace whose writer lease this
  process holds it is kept across processes in `runtime/verified-files.jsonl`; elsewhere it
  lives for the process. Legacy stamp-only rows are misses. A digest is reused while file identity,
  size, modification time and kernel change time hold; a reader waits for a recent stamp to settle.
  A writer's own digest can be kept immediately. Deliberately restoring timestamps is outside the
  accidental-change model.

A missing fact only costs the full computation: both records are rebuildable caches, and a
store that predates the year-fact table holds none until its first writer creates it.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import time
import weakref
from _thread import LockType
from collections.abc import Iterable, Mapping
from ctypes import wintypes
from pathlib import Path
from threading import Lock

import duckdb
import xxhash

from alphalattice.kernel.shared_kernel.bounded_memo import BoundedMemo
from alphalattice.kernel.shared_kernel.spans import span

if os.name == "nt":
    import msvcrt

FILE_FACTS = Path("runtime") / "verified-files.jsonl"
_CONTENT_KIND = "\x00content-v1:"
_STAMP_KIND = "\x00file-stamp-v1:"
RACY_SECONDS = 2.0
"""How old the newest modification or change time must be before a digest is kept."""

_Key = tuple[str, str]
_Entry = tuple[tuple[int, int, int], str]
_LOCK = Lock()
_DIGEST_LOCKS: weakref.WeakValueDictionary[str, LockType] = weakref.WeakValueDictionary()
_KEPT: dict[Path, BoundedMemo[_Key, _Entry]] = {}
"""Per workspace whose writer lease this process holds: its file facts by path and kind."""
_PROCESS: BoundedMemo[_Key, _Entry] = BoundedMemo()
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


def configure_file_facts(*, capacity: int | None = None) -> None:
    """Bound retained RAM facts, preserving active writer leases and the persistent journals."""
    global _PROCESS
    with _LOCK:
        for root, previous in [(None, _PROCESS), *_KEPT.items()]:
            retained = BoundedMemo[_Key, _Entry](capacity=capacity)
            retained.update(previous)
            if root is None:
                _PROCESS = retained
            else:
                _KEPT[root] = retained


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
            # contract: remove legacy stamp-only handling once workspace journals are compacted.
            if not str(kind).startswith(_CONTENT_KIND):
                continue  # a legacy stamp cannot establish the bytes of a reusable fact
            key = (str(path), str(kind))
            entries.pop(key, None)
            entries[key] = ((int(size), int(mtime), int(ino)), str(value))
        except (ValueError, TypeError):
            continue  # a damaged line costs that file one full digest
    # A changing file must not leave an unbounded series of content keys in the journal.
    capacity = _PROCESS.capacity
    if len(lines) > 2 * min(len(entries), capacity) + 1024:
        # A content key is checked at use; dropping older keys only costs a future miss.
        entries = dict(list(entries.items())[-capacity:])
        staged = (root / FILE_FACTS).with_name(f"{FILE_FACTS.name}.{os.getpid()}.partial")
        try:
            staged.write_text("".join(map(_line, entries.items())), encoding="utf-8")
            os.replace(staged, root / FILE_FACTS)
        except OSError:
            staged.unlink(missing_ok=True)
    with _LOCK:
        retained = BoundedMemo[_Key, _Entry](capacity=_PROCESS.capacity)
        retained.update(entries)
        _KEPT[root] = retained


def release_file_facts(workspace: Path) -> None:
    """Stop keeping `workspace`'s file facts in this process (its writer lease closed)."""
    with _LOCK:
        _KEPT.pop(Path(os.path.abspath(workspace)), None)


def _content_kind(kind: str, digest: str) -> str:
    canonical = json.dumps([kind, digest], ensure_ascii=False, separators=(",", ":"))
    return _CONTENT_KIND + xxhash.xxh3_128_hexdigest(canonical.encode("utf-8")) + ":" + canonical


def _content_shared(kind: str) -> bool:
    return kind in {"sha256", "columns"} or kind.startswith("columns-v2:")


def _content_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _stamp(path: Path) -> tuple[int, int, int, int, int]:
    """Read identity, size and both change clocks from the same open file."""
    with path.open("rb") as handle:
        stat = os.fstat(handle.fileno())
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetFileInformationByHandleEx.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
            basic = (ctypes.c_longlong * 5)()
            if not kernel.GetFileInformationByHandleEx(
                msvcrt.get_osfhandle(handle.fileno()), 0, ctypes.byref(basic), ctypes.sizeof(basic)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            changed = (basic[3] - 116444736000000000) * 100
        else:
            changed = stat.st_ctime_ns
        if changed <= 0:
            raise OSError("File change time is unavailable")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, changed


def _stamp_key(path: Path, stamp: tuple[int, ...]) -> _Key:
    return str(Path(os.path.abspath(path))), _STAMP_KIND + json.dumps(stamp, separators=(",", ":"))


def _settled(stamp: tuple[int, ...]) -> bool:
    return time.time_ns() - max(stamp[-2:]) > RACY_SECONDS * 10**9


def _held_digest(path: Path) -> str | None:
    try:
        stamp = _stamp(path)
    except OSError:
        return None
    with _LOCK:
        held = _PROCESS.get(_stamp_key(path, stamp))
    return held[1] if held is not None and _settled(stamp) else None


def _verified_digest(path: Path, *, settle: bool = False) -> str:
    absolute = str(Path(os.path.abspath(path)))
    with _LOCK:
        gate = _DIGEST_LOCKS.get(absolute)
        if gate is None:
            gate = Lock()
            _DIGEST_LOCKS[absolute] = gate
    with gate:
        return _read_digest(path, settle=settle)


def _read_digest(path: Path, *, settle: bool = False) -> str:
    try:
        before = _stamp(path)
    except OSError:
        with span("hash", "file_stamp_unavailable"):
            return _content_sha256(path)
    with _LOCK:
        held = _PROCESS.get(_stamp_key(path, before))
    if held is not None and _settled(before):
        return held[1]
    age = time.time_ns() - max(before[-2:])
    window = int(RACY_SECONDS * 10**9)
    if settle and 0 <= age <= window:
        # Wait without the shared cache lock, then read the stamp again before hashing.
        with span("wait", "file_stamp_settle"):
            time.sleep((window - age + 1_000_000) / 10**9)
        try:
            before = _stamp(path)
        except OSError:
            with span("hash", "file_stamp_unavailable"):
                return _content_sha256(path)
    digest = _content_sha256(path)
    try:
        after = _stamp(path)
    except OSError:
        with span("verify", "file_stamp_unavailable"):
            pass
        return digest
    if before == after and _settled(after):
        with _LOCK:
            _PROCESS[_stamp_key(path, after)] = ((after[2], after[3], after[1]), digest)
    elif before != after:
        with span("verify", "file_stamp_changed"):
            pass
    else:
        with span("verify", "file_stamp_unsettled"):
            pass
    return digest


def file_fact(path: Path, kind: str, *, _content: str | None = None) -> str | None:
    """The value proved for this exact file content, or None when absent or unreadable."""
    try:
        digest = (
            _content
            if _content is not None
            else (_held_digest(path) if kind == "sha256" else _verified_digest(path))
        )
    except OSError:
        return None
    if digest is None:
        entries, _ = _place(path, kind)
        with _LOCK:
            kept = entries is not _PROCESS and any(
                k[1].startswith(_CONTENT_KIND) and '"sha256"' in k[1] for k in entries
            )
        if not kept:
            return None
        try:
            digest = _verified_digest(path)
        except OSError:
            return None
    entries, key = _place(path, _content_kind(kind, digest), shared=_content_shared(kind))
    with _LOCK:
        held = entries.get(key)
    return held[1] if held is not None else None


def record_file_fact(
    path: Path,
    kind: str,
    value: str,
    *,
    written: bool = False,
    stat: os.stat_result | None = None,
    _content: str | None = None,
) -> None:
    """Remember `path`'s digest of `kind`: verified by a reader, or `written` by its writer."""
    stat = stat if stat is not None else os.stat(path)
    if not written and time.time() - stat.st_mtime <= RACY_SECONDS:
        return
    digest = _content if _content is not None else _verified_digest(path)
    if kind == "sha256" and value != digest:
        return
    entry = ((stat.st_size, stat.st_mtime_ns, stat.st_ino), value)
    if not _holds(path, entry):
        return
    entries, key = _place(path, _content_kind(kind, digest), shared=_content_shared(kind))
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
    """Reuse a settled file's digest while its complete stamp holds."""
    before = os.stat(path)
    digest = _verified_digest(path, settle=True)
    if _holds(path, ((before.st_size, before.st_mtime_ns, before.st_ino), digest)):
        record_file_fact(path, "sha256", digest, stat=before, _content=digest)
    return digest


def _holds(path: Path, entry: _Entry) -> bool:
    try:
        stat = os.stat(path)
    except OSError:
        return False
    return (stat.st_size, stat.st_mtime_ns, stat.st_ino) == entry[0]


def _place(
    path: Path, kind: str, *, shared: bool = False
) -> tuple[BoundedMemo[_Key, _Entry], _Key]:
    absolute = Path(os.path.abspath(path))
    with _LOCK:
        for root, entries in _KEPT.items():
            if absolute.is_relative_to(root):
                return entries, ("" if shared else absolute.relative_to(root).as_posix(), kind)
    return _PROCESS, ("" if shared else str(absolute), kind)


def _line(item: tuple[_Key, _Entry]) -> str:
    (path, kind), (stamp, value) = item
    return json.dumps([path, kind, *stamp, value], separators=(",", ":")) + "\n"
