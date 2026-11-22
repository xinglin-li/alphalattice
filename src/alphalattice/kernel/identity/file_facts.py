"""File digests trusted until the file's stamp changes: the one record of each file's SHA-256."""

from __future__ import annotations

import os
import weakref
from _thread import LockType
from collections.abc import Callable, MutableMapping
from contextlib import AbstractContextManager
from io import BufferedReader
from pathlib import Path
from threading import Lock
from typing import BinaryIO, Protocol, cast

from ._digests import integrity_sha256_stream

RACY_SECONDS = 2.0
"""How old the newest modification or change time must be before a digest is kept."""


def integrity_sha256_path(path: Path) -> str:
    """Digest the opened file's bytes for an integrity comparison."""
    with path.open("rb") as handle:
        return integrity_sha256_stream(handle).hexdigest()


def _whole(handle: BinaryIO) -> str:
    handle.seek(0)
    return integrity_sha256_stream(cast(BufferedReader, handle)).hexdigest()


def file_stamp(path: Path) -> tuple[int, int, int, int, int]:
    """Read identity, size and both change clocks from the same open file."""
    with path.open("rb") as handle:
        return handle_stamp(handle)


def handle_stamp(handle: BinaryIO) -> tuple[int, int, int, int, int]:
    """Read identity, size and both change clocks of the file an open handle names."""
    stat = os.fstat(handle.fileno())
    if os.name == "nt":
        import ctypes  # lazy: the Windows file change clock needs a kernel handle.
        import msvcrt  # lazy: resolve the Windows file handle only for a stamp.
        from ctypes import wintypes  # lazy: the Windows file change clock's argument types.

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


Stamp = tuple[int, int, int, int, int]
"""A file's identity, size and both change clocks (``file_stamp``)."""
TrustEntry = tuple[Stamp, str]
"""A file's stamp when its digest was taken, and the digest."""


class TrustStore(Protocol):
    """Where the trust record keeps each file's entry; its composer owns memory and the journal."""

    def place(self, path: Path) -> tuple[MutableMapping[str, TrustEntry], str]:
        """The map that holds `path`'s entry and its key there: by kept root and the path under
        it, else by the absolute path for this process."""

    def kept(self, path: Path, key: str, entry: TrustEntry) -> None:
        """A new entry, for the composer to keep beyond this process when its root is kept."""


class FileDigestTrust:
    """The one record of each file's SHA-256: taken once after its write and trusted until its
    stamp changes.

    A read takes the recorded digest while the file's stamp equals the recorded one and is
    settled; a stamp whose newest clock is inside the racy tick (RACY_SECONDS) is hashed again,
    since a same-size rewrite in one clock tick would leave it unchanged. A digest a read takes is
    recorded only when the stamp held across the hash and is settled. A writer records the digest
    of the bytes its file holds after its atomic rename, once they are on disk (``written``).
    Every stamp and hash is read off one open file. With trust off (``trusting``), every read
    hashes and nothing is recorded.
    """

    def __init__(
        self,
        *,
        store: TrustStore,
        lock: LockType,
        span: Callable[[str, str], AbstractContextManager[object]],
        time_ns: Callable[[], int],
        trusting: Callable[[], bool],
    ) -> None:
        """Bind the record's store, lock, measurement, clock and switch."""
        self._store = store
        self._lock = lock
        self._span = span
        self._time_ns = time_ns
        self._trusting = trusting
        self._digest_locks: weakref.WeakValueDictionary[str, LockType] = (
            weakref.WeakValueDictionary()
        )

    def _settled(self, stamp: Stamp) -> bool:
        return self._time_ns() - max(stamp[-2:]) > RACY_SECONDS * 10**9

    def held(self, path: Path) -> str | None:
        """The recorded digest while its stamp holds and is settled, else None: never a hash."""
        if not self._trusting():
            return None
        try:
            stamp = file_stamp(path)
        except OSError:
            return None
        entries, key = self._store.place(path)
        with self._lock:
            held = entries.get(key)
        return held[1] if held is not None and held[0] == stamp and self._settled(stamp) else None

    def verified(self, path: Path, handle: BinaryIO | None = None) -> tuple[str, bool]:
        """The file's digest, and whether its stamp was settled and held so the digest is kept.

        The stamps and the hash are read off one open file: `handle`, one the caller holds, or the
        path opened once here, so a path replaced meanwhile never lends its stamp to other bytes.
        """
        absolute = str(Path(os.path.abspath(path)))
        with self._lock:
            gate = self._digest_locks.get(absolute)
            if gate is None:
                gate = Lock()
                self._digest_locks[absolute] = gate
        with gate:
            if handle is not None:
                return self._read_digest(path, handle)
            with path.open("rb") as opened:
                return self._read_digest(path, opened)

    def _read_digest(self, path: Path, handle: BinaryIO) -> tuple[str, bool]:
        try:
            before = handle_stamp(handle)
        except OSError:
            with self._span("hash", "file_stamp_unavailable"):
                return _whole(handle), False
        settled = self._settled(before)
        trusting = self._trusting()
        entries, key = self._store.place(path)
        if trusting:
            with self._lock:
                held = entries.get(key)
            if held is not None and held[0] == before and settled:
                return held[1], True
            with self._span("hash", "file_digest"):
                digest = _whole(handle)
        else:
            with self._span("hash", "file_trust_off"):  # the switch's mode, where readers look
                digest = _whole(handle)
        try:
            after = handle_stamp(handle)
        except OSError:
            with self._span("verify", "file_stamp_unavailable"):
                pass
            return digest, False
        if before == after and settled:
            if trusting:
                self._keep(path, entries, key, (after, digest))
            return digest, True
        with self._span(
            "verify", "file_stamp_changed" if before != after else "file_stamp_unsettled"
        ):
            pass
        return digest, False

    def written(self, path: Path, digest: str | None = None) -> str:
        """Record, as its writer after the atomic rename, the digest of the bytes the file holds,
        and return it.

        The hash, the flush to disk and the stamp are taken off one open of the file, so a crash
        cannot leave other bytes under this stamp and a path replaced since the writer's rename
        never lends its stamp to them. The stamp is read before the hash and again after the
        flush, so an edit in place meanwhile never pairs its stamp with the earlier bytes. A file
        whose bytes are not the writer's `digest` (it was replaced meanwhile), whose stamp moved
        while it was hashed, or that cannot be opened, flushed or stamped now (another process
        holds it unshared), keeps no entry: its next read hashes it. With trust off, a writer that
        names its digest is answered without a hash.
        """
        if digest is not None and not self._trusting():
            return digest
        try:
            handle = path.open("r+b")
        except OSError:
            return digest if digest is not None else integrity_sha256_path(path)
        with handle:
            try:
                before = handle_stamp(handle)
            except OSError:
                before = None
            with self._span("hash", "file_written"):
                held = _whole(handle)
            if before is None or not self._trusting() or (digest is not None and held != digest):
                return held
            try:
                os.fsync(handle.fileno())
                stamp = handle_stamp(handle)
            except OSError:
                return held
            if stamp != before:
                return held
        entries, key = self._store.place(path)
        self._keep(path, entries, key, (stamp, held))
        return held

    def _keep(
        self, path: Path, entries: MutableMapping[str, TrustEntry], key: str, entry: TrustEntry
    ) -> None:
        with self._lock:
            if entries.get(key) == entry:
                return
            entries[key] = entry
        self._store.kept(path, key, entry)
