"""Sparse synthetic managed bytes for real storage-capacity boundary tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def sparse_capacity_ballast(workspace: Path, *, cap_bytes: int) -> Iterator[Path]:
    """Exceed an owner's real cap with a sparse file, removed even after a refused request.

    The storage owner counts file length, not allocated blocks. No budget is replaced and no
    payload is written. Windows uses native end-of-file controls: its CRT ``truncate`` can
    materialize zeros despite a sparse flag. Other systems prove a small hole is sparse before
    extending it to the real cap, then verify the final allocation too.
    """
    if cap_bytes <= 0:
        raise ValueError("storage_capacity_fixture.cap_must_be_positive")
    root = workspace.resolve()
    path = root / "staging/storage-capacity-test.sparse"
    path.parent.mkdir(parents=True, exist_ok=True)
    assert path.resolve().is_relative_to(root)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            length = cap_bytes + 1
            if os.name == "nt":
                import ctypes
                import msvcrt
                from ctypes import wintypes

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                control = kernel.DeviceIoControl
                control.argtypes = [
                    wintypes.HANDLE,
                    wintypes.DWORD,
                    wintypes.LPVOID,
                    wintypes.DWORD,
                    wintypes.LPVOID,
                    wintypes.DWORD,
                    ctypes.POINTER(wintypes.DWORD),
                    wintypes.LPVOID,
                ]
                control.restype = wintypes.BOOL
                pointer = kernel.SetFilePointerEx
                pointer.argtypes = [
                    wintypes.HANDLE,
                    ctypes.c_longlong,
                    ctypes.POINTER(ctypes.c_longlong),
                    wintypes.DWORD,
                ]
                pointer.restype = wintypes.BOOL
                end = kernel.SetEndOfFile
                end.argtypes = [wintypes.HANDLE]
                end.restype = wintypes.BOOL
                handle = msvcrt.get_osfhandle(stream.fileno())
                returned = wintypes.DWORD()
                fsctl_set_sparse = 0x000900C4
                if not control(
                    handle,
                    fsctl_set_sparse,
                    None,
                    0,
                    None,
                    0,
                    ctypes.byref(returned),
                    None,
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not pointer(handle, length, None, 0) or not end(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
                size = kernel.GetCompressedFileSizeW
                size.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
                size.restype = wintypes.DWORD
            else:
                stream.seek(1024**2)
                stream.truncate()
                probe = os.fstat(stream.fileno())
                assert probe.st_blocks * 512 < probe.st_size, (
                    "filesystem did not create a sparse hole"
                )
                stream.seek(length)
                stream.truncate()
                allocated = os.fstat(stream.fileno()).st_blocks * 512
        if os.name == "nt":
            high = wintypes.DWORD()
            ctypes.set_last_error(0)
            low = size(str(path), ctypes.byref(high))
            error = ctypes.get_last_error()
            if low == 0xFFFFFFFF and error:
                raise ctypes.WinError(error)
            allocated = (high.value << 32) | low
        assert path.stat().st_size == length
        assert allocated < 1024**2, "capacity ballast allocated a payload"
        yield path
    finally:
        if created:
            path.unlink(missing_ok=True)


__all__ = ["sparse_capacity_ballast"]
