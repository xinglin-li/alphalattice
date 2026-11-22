"""File digests whose existing trust depends on the file's stat."""

from __future__ import annotations

import json
import os
import weakref
from _thread import LockType
from collections.abc import Callable, MutableMapping
from contextlib import AbstractContextManager
from pathlib import Path
from threading import Lock

from ._digests import integrity_sha256_stream

RACY_SECONDS = 2.0
"""How old the newest modification or change time must be before a digest is kept."""
_STAMP_KIND = "\x00file-stamp-v1:"

FactKey = tuple[str, str]
FactEntry = tuple[tuple[int, int, int], str]


def integrity_sha256_path(path: Path) -> str:
    """Digest the opened file's bytes for an integrity comparison."""
    with path.open("rb") as handle:
        return integrity_sha256_stream(handle).hexdigest()


def file_stamp(path: Path) -> tuple[int, int, int, int, int]:
    """Read identity, size and both change clocks from the same open file."""
    with path.open("rb") as handle:
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


def _stamp_key(path: Path, stamp: tuple[int, ...]) -> FactKey:
    return str(Path(os.path.abspath(path))), _STAMP_KIND + json.dumps(stamp, separators=(",", ":"))


def file_fact_holds(path: Path, entry: FactEntry) -> bool:
    """Check the size, modification time and inode recorded with a digest."""
    try:
        stat = os.stat(path)
    except OSError:
        return False
    return (stat.st_size, stat.st_mtime_ns, stat.st_ino) == entry[0]


class FileDigestTrust:
    """Use the workspace's retained memo and span hooks for one stat-checked digest."""

    def __init__(
        self,
        *,
        process: Callable[[], MutableMapping[FactKey, FactEntry]],
        lock: LockType,
        span: Callable[[str, str], AbstractContextManager[object]],
        time_ns: Callable[[], int],
    ) -> None:
        """Bind the existing memo, lock, measurement and clock used to trust file stamps."""
        self._process = process
        self._lock = lock
        self._span = span
        self._time_ns = time_ns
        self._digest_locks: weakref.WeakValueDictionary[str, LockType] = (
            weakref.WeakValueDictionary()
        )

    def _settled(self, stamp: tuple[int, ...]) -> bool:
        return self._time_ns() - max(stamp[-2:]) > RACY_SECONDS * 10**9

    def held(self, path: Path) -> str | None:
        """Return the retained digest only while its file stamp still holds."""
        try:
            stamp = file_stamp(path)
        except OSError:
            return None
        with self._lock:
            held = self._process().get(_stamp_key(path, stamp))
        return held[1] if held is not None and self._settled(stamp) else None

    def verified(self, path: Path) -> tuple[str, bool]:
        """Return the digest and whether an unchanged, settled stamp lets a reader keep it."""
        absolute = str(Path(os.path.abspath(path)))
        with self._lock:
            gate = self._digest_locks.get(absolute)
            if gate is None:
                gate = Lock()
                self._digest_locks[absolute] = gate
        with gate:
            return self._read_digest(path)

    def _read_digest(self, path: Path) -> tuple[str, bool]:
        try:
            before = file_stamp(path)
        except OSError:
            with self._span("hash", "file_stamp_unavailable"):
                return integrity_sha256_path(path), False
        settled = self._settled(before)
        with self._lock:
            held = self._process().get(_stamp_key(path, before))
        if held is not None and settled:
            return held[1], True
        digest = integrity_sha256_path(path)
        try:
            after = file_stamp(path)
        except OSError:
            with self._span("verify", "file_stamp_unavailable"):
                pass
            return digest, False
        if before == after and settled:
            with self._lock:
                self._process()[_stamp_key(path, after)] = ((after[2], after[3], after[1]), digest)
            return digest, True
        if before != after:
            with self._span("verify", "file_stamp_changed"):
                pass
        else:
            with self._span("verify", "file_stamp_unsettled"):
                pass
        return digest, False

    def sealed(
        self, path: Path, expected: str, held: FactEntry | None
    ) -> tuple[bool, os.stat_result | None]:
        """Read a settled seal, or return the stat of bytes proved now."""
        settled = held is not None and self._time_ns() - held[0][1] > RACY_SECONDS * 10**9
        if settled and held is not None and file_fact_holds(path, held):
            return True, None
        try:
            before = os.stat(path)
            with self._span("hash", "sealed_file"):
                digest = integrity_sha256_path(path)
        except OSError:
            return False, None
        return digest == expected, before
