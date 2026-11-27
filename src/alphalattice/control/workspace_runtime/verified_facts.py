"""What a workspace has verified once, recorded so that no later read computes it again.

- A **year fact** vouches for one calendar year of a market-store table's rows. It sits in the
  store (`verified_year_fact`) and answers only under the store's seal epoch
  (`WorkspaceDatabase.seal_epoch`) and the reader's basis (the rule or view it was made under).
  The rows' writers forget or renew the years they touch in their own transactions, so only an
  edit none of them made could leave a fact stale, and the epoch ends every fact then.
- A **file's digest** is the hashing owner's trust record (`FileDigestTrust`): taken once,
  by its writer after the bytes are on disk and renamed into place or by a read whose stamp
  held and was settled, and trusted until the file's stamp (identity, size, modification and
  change time) changes. A stamp inside the racy tick is hashed again. Under a workspace whose
  writer lease this process holds, the record is kept across processes in
  `runtime/file-trust.jsonl`, keyed by the path under the workspace; elsewhere it lives for
  the process. ``ALPHALATTICE_FILE_TRUST=0`` turns trust off: every read hashes. Inside
  ``full_content_proof()`` (an entry, an export, the scrub: LAWS.md EV2) every sealed file is
  proved from its bytes. Deliberately restoring timestamps is outside the accidental-change
  model.
- A **file fact** binds a value derived from a file's bytes to its kind and the file's
  BLAKE3 digest. xxh3 indexes the complete canonical key; a hit compares that key. It is kept
  beside the digests in `runtime/verified-files.jsonl`, or for the process.

A missing fact only costs the full computation: the records are rebuildable caches, and a
store that predates the year-fact table holds none until its first writer creates it.
"""

from __future__ import annotations

import errno
import json
import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import BinaryIO, TextIO, cast

import duckdb

from alphalattice.capabilities.domain_contracts.bounded_memo import BoundedMemo
from alphalattice.capabilities.domain_contracts.domain.errors import WorkspaceSecurityError
from alphalattice.capabilities.domain_contracts.environment import file_trust
from alphalattice.capabilities.domain_contracts.spans import span
from alphalattice.control.workspace_runtime.database import table_columns
from alphalattice.control.workspace_runtime.paths import resolve_confined
from alphalattice.kernel import storage
from alphalattice.kernel.clock import wall_time, wall_time_ns
from alphalattice.kernel.config import data_root
from alphalattice.kernel.identity import (
    RACY_SECONDS,
    FileDigestTrust,
    TrustEntry,
    bind_sealed,
    change_xxh3_128_hex,
    integrity_blake3_path,
    is_blake3,
    whole_digest,
)

FILE_FACTS = Path("runtime") / "verified-files.jsonl"
FILE_TRUST = Path("runtime") / "file-trust.jsonl"
FILE_TRUST_SCRUB = Path("runtime") / "file-trust-scrub.json"
_SCRUB_STARTED = '"scrub-start-v1"'
_SCRUB_CHECKED = '"scrub-checked-v1"'
_SCRUB_DISCARDED = '"scrub-discard-v1"'
_CONTENT_KIND = "\x00content-v2:"
_FULL_PROOF: ContextVar[BoundedMemo[tuple[object, ...], object] | None] = ContextVar(
    "content_store_full_proof", default=None
)
"""Open inside ``full_content_proof()``: what a proof has shown, shared by its reads."""

_Key = tuple[str, str]
_Entry = tuple[tuple[int, int, int], str]
_LOCK = Lock()
_Publish = Callable[[Callable[[], None]], bool]
_KEPT: dict[Path, BoundedMemo[_Key, _Entry]] = {}
"""Per workspace whose writer lease this process holds: its file facts by path and kind."""
_PROCESS: BoundedMemo[_Key, _Entry] = BoundedMemo()
"""File facts outside a kept workspace, by absolute path and kind, for this process."""
_TRUSTED: dict[Path, BoundedMemo[str, TrustEntry]] = {}
"""Per kept workspace: each file's trusted digest by its path under the workspace."""
_TRUSTED_PROCESS: BoundedMemo[str, TrustEntry] = BoundedMemo()
"""Trusted digests outside a kept workspace, by absolute path, for this process."""
_TRUST_LINES: dict[Path, int] = {}
"""Per kept workspace: the lines its trust journal holds, so its idle job compacts it in time."""
_TRUST_STALE: set[Path] = set()
"""Kept workspaces whose trust journal may name a file the workspace no longer holds: each one
loaded, and each whose scrub skipped a file. The idle job decides which files remain."""


class _TrustJournal:
    """The trust record's store: a kept workspace's map and journal, else the process map."""

    def __init__(self) -> None:
        """Serialize durable writes separately from the in-memory trust lookup."""
        self._writer = Lock()
        self._pending: dict[tuple[Path, str], TrustEntry] = {}

    def place(self, path: Path) -> tuple[MutableMapping[str, TrustEntry], str]:
        absolute = Path(os.path.abspath(path))
        _keep_data_root(absolute)
        with _LOCK:
            for root, entries in _TRUSTED.items():
                if absolute.is_relative_to(root):
                    return entries, absolute.relative_to(root).as_posix()
            return _TRUSTED_PROCESS, str(absolute)

    def kept(self, path: Path, key: str, entry: TrustEntry) -> None:
        absolute = Path(os.path.abspath(path))
        with _LOCK:
            root = next((root for root in _TRUSTED if absolute.is_relative_to(root)), None)
            if root is None or _TRUSTED[root].get(key) != entry:
                return
            self._pending[root, key] = entry
            acquired = self._writer.acquire(blocking=False)
        if acquired:
            self._flush()

    def mark_scrub(self, workspace: Path, marker: str, publish: _Publish) -> bool:
        """Seal the scan's journal boundary before hashing or advancing its cadence."""

        def publish_marker(append: Callable[[], None]) -> bool:
            def write() -> None:
                if not self._writer.acquire(blocking=False):
                    raise BlockingIOError(errno.EBUSY, os.strerror(errno.EBUSY))
                try:
                    append()
                finally:
                    self._writer.release()

            return publish(write)

        # Only kept() drains queued facts. The marker's bytes are published before
        # fsync, so foreground writers can persist their own facts during the sync.
        if not _append(workspace / FILE_TRUST, "\n" + marker + "\n", publish=publish_marker):
            return False
        _counted(Path(os.path.abspath(workspace)), 2)
        return True

    def _flush(self) -> None:
        # Write one batch. New arrivals stay queued for the next journal advance.
        try:
            with _LOCK:
                pending = {
                    (root, key): entry
                    for (root, key), entry in self._pending.items()
                    if (kept := _TRUSTED.get(root)) is not None and kept.get(key) == entry
                }
                self._pending.clear()
            for (root, key), entry in pending.items():
                if _append(root / FILE_TRUST, _trust_line((key, entry))):
                    _counted(root, 1)
        finally:
            self._writer.release()

    def compact(self, workspace: Path, publish: _Publish) -> bool:
        """Rewrite a due trust journal to its held files' entries, as its next load reads them.

        The journal is read from its own bytes, with no lock and outside admission; its markers
        apply as at a load, so a sealed discard holds. The probes and the staged file are made
        there too. Only the replace, the compaction's one visible write, goes through `publish`
        (the Host's idle boundary), and only while the journal is the one read: a fact appended
        since leaves it for the next idle run. False when it yields."""
        root = Path(os.path.abspath(workspace))
        if not _compaction_due(root):
            return True
        journal = root / FILE_TRUST
        try:
            with journal.open("rb") as stream:
                size = os.fstat(stream.fileno()).st_size
                lines = stream.read(size).decode("utf-8", errors="replace").splitlines()
        except FileNotFoundError:
            size, lines = -1, []
        durable = _entries(lines, _trust_entry, trust_journal=True)
        items = cast(list[tuple[str, TrustEntry]], list(durable.items())[-_PROCESS.capacity :])
        held, dropped = [], []
        for item in items:
            (held if _held(root, item[0]) else dropped).append(item)
        bound = 2 * min(len(items), _PROCESS.capacity) + 1024
        if not dropped and len(items) == len(durable) and len(lines) <= bound:
            with _LOCK:
                _TRUST_STALE.discard(root)  # every file it names is held: nothing to rewrite
            return True
        staged = journal.with_name(f"{journal.name}.{os.getpid()}.partial")
        replaced = False

        def replace() -> None:
            nonlocal replaced
            if not self._writer.acquire(blocking=False):
                return
            try:
                if _size(journal) == size:
                    os.replace(staged, journal)
                    replaced = True
                    with _LOCK:  # under the writer, before a later append is counted
                        _TRUST_LINES[root] = len(held)
                        _TRUST_STALE.discard(root)
                        kept = _TRUSTED.get(root)
                        for key, entry in dropped:  # the memo forgets them as the journal did
                            if kept is not None and kept.get(key) == entry:
                                del kept[key]
            finally:
                self._writer.release()

        try:  # a failed write raises to the idle job's bounded retry; the journal stays due
            with staged.open("w", encoding="utf-8") as stream:
                stream.write("".join(map(_trust_line, held)))
                stream.flush()
                os.fsync(stream.fileno())
            publish(replace)
        finally:
            staged.unlink(missing_ok=True)
        return replaced


def _counted(root: Path, lines: int) -> None:
    with _LOCK:
        if root in _TRUST_LINES:
            _TRUST_LINES[root] += lines


def _compaction_due(root: Path) -> bool:
    """Whether a kept workspace's trust journal may name a removed file or outgrew its bound."""
    with _LOCK:
        kept = _TRUSTED.get(root)
        if kept is None:
            return False
        bound = 2 * min(len(kept), kept.capacity) + 1024
        return root in _TRUST_STALE or _TRUST_LINES.get(root, 0) > bound


def _held(root: Path, key: str) -> bool:
    """Whether the workspace still holds a trusted file, read as the scrub reads it: an entry
    the confined resolver refuses, or an unreachable one, is dropped from this rebuildable cache."""
    try:
        return resolve_confined(root, key).is_file()
    except (OSError, WorkspaceSecurityError):
        return False


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return -1


_TRUST_JOURNAL = _TrustJournal()
_DIGEST_TRUST = FileDigestTrust(
    store=_TRUST_JOURNAL,
    lock=_LOCK,
    span=span,
    time_ns=lambda: wall_time_ns(),
    trusting=file_trust,
)
bind_sealed(_DIGEST_TRUST.sealed)  # the data layer's writers record their parts' digests here
_DATA_ROOTS: set[Path] = set()


def _keep_data_root(absolute: Path) -> None:
    """The data root's sealed parts keep their trust across processes, as a kept workspace's
    files do: its record is loaded at the first lookup or record beneath it, in every process
    that seals or reads a part there (the Host and its Task workers append to one journal)."""
    root = Path(os.path.abspath(data_root()))
    if root in _DATA_ROOTS or not absolute.is_relative_to(root):
        return
    with _LOCK:
        if root in _DATA_ROOTS or root in _TRUSTED:
            return
        _DATA_ROOTS.add(root)
    keep_file_facts(root, shared=True)


def held_digest(path: Path, *, holding: Callable[[str], bool] | None = None) -> str | None:
    """The digest the trust record holds for `path` while its stamp holds and is settled, of
    either the digest of its raw bytes or its writer's artifact identity. Never a hash.
    A recorded digest that `holding` refuses answers None before the file is stamped."""
    return _DIGEST_TRUST.held(path, holding=holding)


def named_digest(path: Path, observed: os.stat_result) -> str | None:
    """The digest the trust record names for the file `observed` saw, of either tag, without
    opening it; for binding a file fact, which a reader trusts only through `held_digest`."""
    return _DIGEST_TRUST.named(path, observed)


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
    # transaction. The file's schema is read once until a writer changes it (`table_columns`).
    return bool(table_columns(connection, "verified_year_fact"))


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
    return year_facts_by_scope(
        connection, kind=kind, basis=basis, epoch=epoch, wanted={scope: years}
    )[scope]


def year_facts_by_scope(
    connection: duckdb.DuckDBPyConnection,
    *,
    kind: str,
    basis: str,
    epoch: str,
    wanted: Mapping[str, Iterable[int]],
) -> dict[str, dict[int, str]]:
    """`year_facts` of many scopes, each its own years, in one read."""
    years = {scope: sorted(set(each)) for scope, each in wanted.items()}
    found: dict[str, dict[int, str]] = {scope: {} for scope in years}
    scopes = [scope for scope, each in years.items() if each]
    if not scopes or not _year_facts_held(connection):
        return found
    for scope, year, value in connection.execute(
        "SELECT scope, year, value FROM verified_year_fact WHERE kind = ? AND basis = ? "
        "AND epoch = ? AND list_contains(?::VARCHAR[], scope)",
        [kind, basis, epoch, scopes],
    ).fetchall():
        if int(year) in years[str(scope)]:
            found[str(scope)][int(year)] = str(value)
    return found


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
    record_year_facts_by_scope(
        connection, kind=kind, basis=basis, epoch=epoch, values={scope: values}
    )


def record_year_facts_by_scope(
    connection: duckdb.DuckDBPyConnection,
    *,
    kind: str,
    basis: str,
    epoch: str,
    values: Mapping[str, Mapping[int, str]],
) -> None:
    """`record_year_facts` of many scopes in one statement."""
    rows = [
        (scope, int(year), value)
        for scope, each in values.items()
        for year, value in {int(year): value for year, value in each.items()}.items()
    ]
    if rows:
        if not _year_facts_held(connection):
            ensure_year_fact_schema(connection)
        # One statement for every year: a row-by-row executemany costs a statement per year.
        scopes, years, kept = map(list, zip(*rows, strict=True))
        connection.execute(
            "INSERT OR REPLACE INTO verified_year_fact"
            " SELECT ?, unnest(?::VARCHAR[]), unnest(?::INTEGER[]), ?, ?, unnest(?::VARCHAR[])",
            [kind, scopes, years, basis, epoch, kept],
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
        scopes, block_years = map(list, zip(*blocks, strict=True))
        connection.execute(
            "DELETE FROM verified_year_fact AS fact USING (SELECT unnest(?::VARCHAR[]) AS scope,"
            " unnest(?::INTEGER[]) AS year) AS block"
            " WHERE fact.kind = ? AND fact.scope = block.scope AND fact.year = block.year",
            [scopes, block_years, kind],
        )


def configure_file_facts(*, capacity: int | None = None) -> None:
    """Bound retained RAM facts, preserving active writer leases and the persistent journals."""
    global _PROCESS, _TRUSTED_PROCESS
    with _LOCK:
        for kept in (_KEPT, _TRUSTED):
            for root, previous in list(kept.items()):
                retained = BoundedMemo(capacity=capacity)
                retained.update(previous)
                kept[root] = retained
        facts = BoundedMemo[_Key, _Entry](capacity=capacity)
        facts.update(_PROCESS)
        trusted = BoundedMemo[str, TrustEntry](capacity=capacity)
        trusted.update(_TRUSTED_PROCESS)
        _PROCESS, _TRUSTED_PROCESS = facts, trusted


def keep_file_facts(workspace: Path, *, shared: bool = False) -> None:
    """Keep file digests and facts for `workspace` across processes; its writer lease's holder
    calls this, once per process. A `shared` root's journals, which other processes append to
    while this one reads them (the data root's), are read here and never rewritten."""
    root = Path(os.path.abspath(workspace))
    facts, _ = _journal(root / FILE_FACTS, _fact_entry, _line, rewrite=not shared)
    # The lease's load probes no file: the idle job decides which of the journal's files remain.
    trusted, lines = _journal(root / FILE_TRUST, _trust_entry, _trust_line, rewrite=False)
    with _LOCK:
        for kept, entries in ((_KEPT, facts), (_TRUSTED, trusted)):
            retained = BoundedMemo(capacity=_PROCESS.capacity)
            retained.update(entries)
            kept[root] = retained
        _TRUST_LINES[root] = lines
        _TRUST_STALE.add(root)


def release_file_facts(workspace: Path) -> None:
    """Stop keeping `workspace`'s file digests and facts in this process (its lease closed)."""
    root = Path(os.path.abspath(workspace))
    with _LOCK:
        _KEPT.pop(root, None)
        _TRUSTED.pop(root, None)
        _TRUST_LINES.pop(root, None)
        _TRUST_STALE.discard(root)


def _journal(
    path: Path,
    entry: Callable[[list[object]], tuple[object, object]],
    line: Callable[[tuple[object, object]], str],
    *,
    rewrite: bool = True,
) -> tuple[dict[object, object], int]:
    """A journal's latest entry per key and the lines it holds. Without `rewrite` (the trust
    journal) nothing is rewritten here: its idle job compacts it."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        lines = []
    entries = _entries(lines, entry, trust_journal=path.name == FILE_TRUST.name)
    if not rewrite:
        return entries, len(lines)
    # A changing file must not leave an unbounded series of keys in the journal.
    capacity = _PROCESS.capacity
    if len(lines) > 2 * min(len(entries), capacity) + 1024:
        # An entry is checked at use; dropping older ones only costs a future miss.
        entries = dict(list(entries.items())[-capacity:])
        staged = path.with_name(f"{path.name}.{os.getpid()}.partial")
        try:
            staged.write_text("".join(map(line, entries.items())), encoding="utf-8")
            os.replace(staged, path)
        except OSError:
            staged.unlink(missing_ok=True)
    return entries, len(lines)


def _entries(
    lines: list[str], entry: Callable[[list[object]], tuple[object, object]], *, trust_journal: bool
) -> dict[object, object]:
    """A journal's latest entry per key; in the trust journal, as its scrub markers leave it."""
    entries: dict[object, object] = {}
    incomplete = False
    for text in lines:
        if trust_journal and text == _SCRUB_STARTED:
            if incomplete:
                entries.clear()  # an earlier process did not seal its scan
            incomplete = True
            continue
        if trust_journal and text in {_SCRUB_CHECKED, _SCRUB_DISCARDED}:
            if text == _SCRUB_DISCARDED:
                entries.clear()
            incomplete = False
            continue
        try:
            key, value = entry(json.loads(text))
        except (ValueError, TypeError):
            continue  # a damaged or older line costs that file one full digest
        entries.pop(key, None)
        entries[key] = value
    if incomplete:
        entries.clear()
    return entries


def _fact_entry(row: list[object]) -> tuple[_Key, _Entry]:
    path, kind, size, mtime, ino, value = row
    if not str(kind).startswith(_CONTENT_KIND):
        raise ValueError("a reusable file fact must bind its BLAKE3 digest")
    return (str(path), str(kind)), ((int(size), int(mtime), int(ino)), str(value))


def _trust_entry(row: list[object]) -> tuple[str, TrustEntry]:
    path, dev, ino, size, mtime, changed, digest, whole_file = row
    if not isinstance(digest, str) or not is_blake3(digest) or not isinstance(whole_file, bool):
        raise ValueError("a trust row must name a BLAKE3 digest and the bytes it covers")
    stamp = (int(dev), int(ino), int(size), int(mtime), int(changed))
    return str(path), TrustEntry(stamp, digest, whole_file)


def _append(journal: Path, line: str, *, publish: _Publish | None = None) -> bool:
    """Append a fact, or publish a durable scrub marker and sync it outside admission."""
    try:
        with ExitStack() as opened:
            log: TextIO | None = None

            def append() -> None:
                nonlocal log
                journal.parent.mkdir(parents=True, exist_ok=True)
                log = opened.enter_context(journal.open("a", encoding="utf-8"))
                log.write(line)
                if publish is not None:
                    log.flush()

            if publish is None:
                append()
            elif not publish(append):
                return False
            else:
                assert log is not None
                os.fsync(log.fileno())
    except OSError:
        if publish is not None:
            raise
        # Rebuildable facts stay in this process if the journal cannot take them.
        return False
    return True


def _content_kind(kind: str, digest: str) -> str:
    canonical = json.dumps([kind, digest], ensure_ascii=False, separators=(",", ":"))
    return _CONTENT_KIND + change_xxh3_128_hex(canonical.encode("utf-8")) + ":" + canonical


def _content_shared(kind: str) -> bool:
    return kind == "columns" or kind.startswith("columns-v2:")


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
    token = _FULL_PROOF.set(BoundedMemo())
    try:
        yield
    finally:
        _FULL_PROOF.reset(token)


def full_proof() -> bool:
    """Whether a read proves its sealed bytes now: inside ``full_content_proof()``."""
    return _FULL_PROOF.get() is not None


def sealed_file_holds(path: Path, digest: str) -> bool:
    """Whether `path` holds the bytes named by this BLAKE3 file digest.

    An ordinary read takes the digest its writer recorded, or an earlier read proved, while the
    file's stamp holds and is settled, and hashes the file otherwise; inside
    ``full_content_proof()`` the bytes are always hashed. False for a missing, unreadable or
    different file.
    """
    if not is_blake3(digest):
        return False
    try:
        if full_proof():
            with span("hash", "full_proof"), storage.outside(path):
                return integrity_blake3_path(path) == digest
        return _verified(path) == digest
    except OSError:
        return False


def record_sealed_file(path: Path, digest: str) -> None:
    """Record, as its writer after the atomic rename, that `path` now holds the bytes it wrote,
    named by `digest`, so no read hashes them again while the file's stamp holds."""
    _DIGEST_TRUST.written(path, digest)


class FileTrustScrub:
    """Re-prove a kept workspace's trust records one file at a time."""

    def __init__(
        self, workspace: Path, *, clock: Callable[[], datetime], period: timedelta
    ) -> None:
        """Restore the workspace's completed cadence and start with no in-flight scan."""
        self.workspace = Path(os.path.abspath(workspace))
        self.clock, self.period = clock, period
        now = clock()
        self.next_due = now
        try:
            receipt = resolve_confined(self.workspace, FILE_TRUST_SCRUB.as_posix())
            finished = datetime.fromisoformat(json.loads(receipt.read_bytes())["finished_at"])
            if finished.tzinfo is not None and finished <= now:
                self.next_due = finished + period
        except (OSError, ValueError, KeyError, TypeError, WorkspaceSecurityError):
            pass  # absent or unreadable cadence evidence makes a scrub due
        self._remaining: deque[str] | None = None
        self._journal_open = False
        self._checked = self._skipped = self._mismatches = 0
        self._first_mismatch: str | None = None

    def run(
        self,
        available: Callable[[], bool],
        *,
        publish: _Publish | None = None,
    ) -> bool:
        """Perform a due scrub, returning False if it yields to foreground work."""

        def write(action: Callable[[], None]) -> bool:
            if publish is not None:
                return publish(action)
            if not available():
                return False
            action()
            return True

        if not file_trust() or self.clock() < self.next_due:
            return True
        if not available():
            return False
        if self._remaining is None:
            with _LOCK:
                remaining = deque(_TRUSTED.get(self.workspace, ()))
            if not _TRUST_JOURNAL.mark_scrub(self.workspace, _SCRUB_STARTED, write):
                return False
            self._journal_open = True
            self._remaining = remaining
            self._checked = self._skipped = self._mismatches = 0
            self._first_mismatch = None
        while self._remaining:
            if not available():
                return False
            relative = self._remaining.popleft()
            try:
                result = _DIGEST_TRUST.scrub(resolve_confined(self.workspace, relative))
            except (OSError, WorkspaceSecurityError):
                result = None
            if result is None:
                self._skipped += 1
                with _LOCK:
                    _TRUST_STALE.add(self.workspace)  # its idle compaction decides if it remains
                continue
            self._checked += 1
            if not result:
                self._mismatches += 1
                if self._first_mismatch is None:
                    self._first_mismatch = relative
        return self._finish(write)

    def _finish(self, publish: _Publish) -> bool:
        if self._journal_open:
            # A detected mismatch makes the old durable cache unusable on restart.
            marker = _SCRUB_DISCARDED if self._mismatches else _SCRUB_CHECKED
            if not _TRUST_JOURNAL.mark_scrub(self.workspace, marker, publish):
                return False
            self._journal_open = False
        # The trust journal compacts only here, in the scheduled cycle, after its seal (an open
        # scan keeps its opening record) and before the receipt: a yield leaves the cycle due.
        if not _TRUST_JOURNAL.compact(self.workspace, publish):
            return False
        finished = self.clock()
        receipt = {
            "finished_at": finished.isoformat(),
            "checked": self._checked,
            "skipped": self._skipped,
            "mismatches": self._mismatches,
            "first_mismatch": self._first_mismatch,
        }
        target = resolve_confined(self.workspace, FILE_TRUST_SCRUB.as_posix())
        staged = resolve_confined(
            self.workspace, FILE_TRUST_SCRUB.with_suffix(".partial").as_posix()
        )
        payload = json.dumps(receipt, separators=(",", ":")) + "\n"

        def replace() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.write_text(payload, encoding="utf-8")
            os.replace(staged, target)

        if not publish(replace):
            return False
        self._remaining = None
        self.next_due = finished + self.period
        return True


def file_fact(path: Path, kind: str, *, _content: str | None = None) -> str | None:
    """The value proved for this exact file content, or None when absent or unreadable; for
    ``digest``, the trusted digest of the file's raw bytes."""
    try:
        digest = (
            _content
            if _content is not None
            else (
                _DIGEST_TRUST.held(path, whole_file=True) if kind == "digest" else _verified(path)
            )
        )
    except OSError:
        return None
    if digest is None or kind == "digest":
        return digest
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
    """Remember a value derived from `path`'s bytes: verified by a reader, or `written` by its
    writer. `stat` is the file's stat before the value was derived; a changed file keeps none."""
    stat = stat if stat is not None else os.stat(path)
    if not written and wall_time() - stat.st_mtime <= RACY_SECONDS:
        return
    digest = _content if _content is not None else _verified(path)
    entry = ((stat.st_size, stat.st_mtime_ns, stat.st_ino), value)
    try:
        now = os.stat(path)
    except OSError:
        return
    if (now.st_size, now.st_mtime_ns, now.st_ino) != entry[0]:
        return
    entries, key = _place(path, _content_kind(kind, digest), shared=_content_shared(kind))
    with _LOCK:
        if entries.get(key) == entry:
            return
        entries[key] = entry
        root = next((root for root, held in _KEPT.items() if held is entries), None)
        if root is not None:
            _append(root / FILE_FACTS, _line((key, entry)))


def verified_file_digest(
    path: Path,
    *,
    written: bool = False,
    handle: BinaryIO | None = None,
    kept: list[bytes] | None = None,
) -> str:
    """The file's BLAKE3 through the trust record, read off `handle` when the caller holds the
    file open; a hash taken of it reads its content into `kept` when given. Its `written` writer
    hashes and records it. Inside ``full_content_proof()`` a read hashes the bytes, answering
    nothing kept before."""
    if written:
        return _DIGEST_TRUST.written(path)
    if full_proof():
        with span("hash", "full_proof"):
            if handle is None:
                with storage.outside(path):
                    return integrity_blake3_path(path)
            return whole_digest(handle, kept)
    return _verified(path) if handle is None else _DIGEST_TRUST.verified(path, handle, kept=kept)[0]


def verified_bytes(path: Path) -> tuple[bytes, str]:
    """A file's bytes and its BLAKE3 through the trust record, both off one open, so the digest
    answered is that of the bytes returned while the file's stamp holds. A miss, and full proof,
    hash the very bytes returned: the file is read once."""
    kept: list[bytes] = []
    with storage.outside(path), path.open("rb") as handle:
        digest = verified_file_digest(path, handle=handle, kept=kept)
        if kept:
            return kept[-1], digest
        handle.seek(0)
        return handle.read(), digest


def _verified(path: Path) -> str:
    """The trust record's digest of `path`, its bytes read outside the engine
    (`storage.outside`): a workspace database instance that would refuse the read is let go."""
    with storage.outside(path):
        return _DIGEST_TRUST.verified(path)[0]


def _place(
    path: Path, kind: str, *, shared: bool = False
) -> tuple[BoundedMemo[_Key, _Entry], _Key]:
    absolute = Path(os.path.abspath(path))
    _keep_data_root(absolute)
    with _LOCK:
        for root, entries in _KEPT.items():
            if absolute.is_relative_to(root):
                return entries, ("" if shared else absolute.relative_to(root).as_posix(), kind)
    return _PROCESS, ("" if shared else str(absolute), kind)


def _line(item: tuple[_Key, _Entry]) -> str:
    (path, kind), (stamp, value) = item
    return json.dumps([path, kind, *stamp, value], separators=(",", ":")) + "\n"


def _trust_line(item: tuple[str, TrustEntry]) -> str:
    path, entry = item
    return (
        json.dumps([path, *entry.stamp, entry.digest, entry.whole_file], separators=(",", ":"))
        + "\n"
    )
