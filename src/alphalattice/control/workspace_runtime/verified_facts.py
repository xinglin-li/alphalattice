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
  size, modification time and kernel change time hold. A reader hashes a recent stamp on
  every read. A writer can record its own digest immediately. Deliberately restoring
  timestamps is outside the accidental-change model.

- A **sealed file** is checked once, at write, in git's index form: its writer records the
  SHA-256 of the bytes it wrote with the file's identity stat (size, modification time, file
  id), and a read takes that digest while the stat holds, hashing the file again only once the
  stat changed. Inside ``full_content_proof()`` (an entry, an export, the scrub: LAWS.md EV2)
  every sealed file is proved from its bytes.

A missing fact only costs the full computation: both records are rebuildable caches, and a
store that predates the year-fact table holds none until its first writer creates it.
"""

from __future__ import annotations

import json
import os
import weakref
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import Lock

import duckdb

from alphalattice.kernel.clock import wall_time, wall_time_ns
from alphalattice.kernel.identity import (
    RACY_SECONDS,
    FileDigestTrust,
    change_xxh3_128_hex,
    file_fact_holds,
    integrity_sha256_path,
)
from alphalattice.kernel.shared_kernel.bounded_memo import BoundedMemo
from alphalattice.kernel.shared_kernel.spans import span

FILE_FACTS = Path("runtime") / "verified-files.jsonl"
_CONTENT_KIND = "\x00content-v1:"
_SEALED = "sealed-sha256"
"""A sealed file's fact: this path holds the bytes of this SHA-256 while its stat holds."""
_FULL_PROOF: ContextVar[dict[tuple[object, ...], object] | None] = ContextVar(
    "content_store_full_proof", default=None
)
"""Open inside ``full_content_proof()``: what a proof has shown, shared by its reads."""

_Key = tuple[str, str]
_Entry = tuple[tuple[int, int, int], str]
_LOCK = Lock()
_KEPT: dict[Path, BoundedMemo[_Key, _Entry]] = {}
"""Per workspace whose writer lease this process holds: its file facts by path and kind."""
_PROCESS: BoundedMemo[_Key, _Entry] = BoundedMemo()
"""File facts outside a kept workspace, by absolute path and kind, for this process."""
_DIGEST_TRUST = FileDigestTrust(
    process=lambda: _PROCESS, lock=_LOCK, span=span, time_ns=lambda: wall_time_ns()
)
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
        # One statement for every year: a row-by-row executemany costs a statement per year.
        years = {int(year): value for year, value in values.items()}
        connection.execute(
            "INSERT OR REPLACE INTO verified_year_fact"
            " SELECT ?, ?, unnest(?::INTEGER[]), ?, ?, unnest(?::VARCHAR[])",
            [kind, scope, list(years), basis, epoch, list(years.values())],
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
    return _CONTENT_KIND + change_xxh3_128_hex(canonical.encode("utf-8")) + ":" + canonical


def _content_shared(kind: str) -> bool:
    return kind in {"sha256", "columns"} or kind.startswith("columns-v2:")


@contextmanager
def full_content_proof() -> Iterator[None]:
    """Prove every sealed file read inside from its bytes, answering nothing kept before.

    The edges open it (LAWS.md EV2): an export, a publication, a stage's check of what it just
    wrote, and the scrub. What it proves is shared inside it, so each sealed value is proved once
    per proof; a nested proof joins the outer one. Kept values stay kept for ordinary reads.
    """
    if _FULL_PROOF.get() is not None:
        yield
        return
    token = _FULL_PROOF.set({})
    try:
        yield
    finally:
        _FULL_PROOF.reset(token)


def full_proof() -> bool:
    """Whether a read proves its sealed bytes now: inside ``full_content_proof()``."""
    return _FULL_PROOF.get() is not None


def sealed_file_holds(path: Path, sha256: str) -> bool:
    """Whether `path` holds the bytes whose SHA-256 is `sha256`, checked once (git's index).

    An ordinary read takes the digest its writer recorded, or an earlier read proved, while the
    file's identity stat holds, and hashes the file only when there is none or the stat changed;
    inside ``full_content_proof()`` the bytes are always hashed. As git does with a racily clean
    entry, a stat whose modification time is not yet settled is hashed: a rewrite of the same
    size within one clock tick would leave it unchanged. False for a missing, unreadable or
    different file.
    """
    held = None
    if not full_proof():
        entries, key = _place(path, _content_kind(_SEALED, sha256))
        with _LOCK:
            held = entries.get(key)
    equal, before = _DIGEST_TRUST.sealed(path, sha256, held)
    if equal and before is not None:
        record_file_fact(path, _SEALED, sha256, stat=before, _content=sha256)
    return equal


def record_sealed_file(path: Path, sha256: str) -> None:
    """Record, as its writer, that `path` now holds the bytes it wrote, whose SHA-256 is `sha256`,
    so no read hashes them again while the file's stat holds."""
    record_file_fact(path, _SEALED, sha256, written=True, _content=sha256)


def file_fact(path: Path, kind: str, *, _content: str | None = None) -> str | None:
    """The value proved for this exact file content, or None when absent or unreadable."""
    try:
        digest = (
            _content
            if _content is not None
            else (_DIGEST_TRUST.held(path) if kind == "sha256" else _DIGEST_TRUST.verified(path)[0])
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
            digest, _ = _DIGEST_TRUST.verified(path)
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
    if not written and wall_time() - stat.st_mtime <= RACY_SECONDS:
        return
    digest = _content if _content is not None else _DIGEST_TRUST.verified(path)[0]
    if kind == "sha256" and value != digest:
        return
    entry = ((stat.st_size, stat.st_mtime_ns, stat.st_ino), value)
    if not file_fact_holds(path, entry):
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


def file_sha256(path: Path, *, written: bool = False) -> str:
    """Hash now, recording the result for a settled file or for its own writer."""
    before = os.stat(path)
    if written:
        digest = integrity_sha256_path(path)
    else:
        digest, settled = _DIGEST_TRUST.verified(path)
        if not settled:
            return digest
    record_file_fact(path, "sha256", digest, written=written, stat=before, _content=digest)
    return digest


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
