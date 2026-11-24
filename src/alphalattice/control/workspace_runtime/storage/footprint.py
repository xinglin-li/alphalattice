"""The physical file census shared by storage status and the footprint driver."""

from __future__ import annotations

import fnmatch
import json
import os
import stat
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

BASIS = "allocated bytes: st_blocks x 512, or FILE_STANDARD_INFO.AllocationSize on Windows"
_RULES: tuple[tuple[str, str, str, str], ...] = tuple(
    (scope, pattern, owner, category)
    for scope, pattern, owner, category in json.loads(
        Path(__file__).with_name("footprint_owners.json").read_text(encoding="utf-8")
    )
)


def owner_for(scope: str, relative: str) -> tuple[str, str] | None:
    """Read the first registered owner and category for a path in its scope."""
    for where, pattern, owner, category in _RULES:
        if where == scope and fnmatch.fnmatch(relative, pattern):
            return owner, category
    return None


def files(root: Path) -> Iterator[tuple[Path, tuple[int, int, tuple[int, int]]]]:
    """Read real file objects once per name without traversing aliases."""
    if root.is_symlink() or root.is_junction() or not root.is_dir():
        return

    def unreadable(error: OSError) -> None:
        if not isinstance(error, FileNotFoundError):
            raise error

    for folder, directories, names in root.walk(on_error=unreadable):
        directories[:] = [name for name in directories if not (folder / name).is_junction()]
        for name in names:
            path = folder / name
            if (reading := _file(path)) is not None:
                yield path, reading


def _file(path: Path) -> tuple[int, int, tuple[int, int]] | None:
    try:
        info = path.lstat()
        if stat.S_ISREG(info.st_mode):
            return _allocated(path, info), info.st_size, (info.st_dev, info.st_ino)
    except FileNotFoundError:
        # The directory entry disappeared between listing and reading it.
        return None
    return None


def allocated_by_category(root: Path) -> dict[str, int]:
    """Count each data-root file identity once, with unclaimed files visible."""
    totals: dict[str, int] = defaultdict(int)
    seen: set[tuple[int, int]] = set()
    for path, (allocated, _length, identity) in files(root):
        if identity not in seen:
            seen.add(identity)
            claim = owner_for("data-root", path.relative_to(root).as_posix())
            totals["unclaimed" if claim is None else claim[1]] += allocated
    return dict(sorted(totals.items()))


def _allocated(path: Path, info: os.stat_result) -> int:
    """The bytes the file system allocates to a file (`BASIS`): block- or cluster-rounded, fewer
    for a sparse or compressed one. A failure to read it is raised, never replaced by the length."""
    if sys.platform != "win32":
        return int(info.st_blocks) * 512
    import ctypes  # lazy: Windows only
    from ctypes import wintypes  # lazy: Windows file-allocation handle and record types

    class Standard(ctypes.Structure):  # FILE_STANDARD_INFO
        _fields_ = (
            ("allocation_size", ctypes.c_longlong),
            ("end_of_file", ctypes.c_longlong),
            ("number_of_links", wintypes.DWORD),
            ("delete_pending", wintypes.BOOLEAN),
            ("directory", wintypes.BOOLEAN),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    # FILE_READ_ATTRIBUTES, every share mode, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT.
    handle = kernel32.CreateFileW(str(path), 0x80, 0x7, None, 3, 0x00200000, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        standard = Standard()
        if not kernel32.GetFileInformationByHandleEx(
            handle, 1, ctypes.byref(standard), ctypes.sizeof(standard)
        ):  # FileStandardInfo
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)
    return int(standard.allocation_size)
