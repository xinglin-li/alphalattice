"""Age a test file's Windows ChangeTime so a read lease can be exercised without sleeping."""

import ctypes
import os
import time
from ctypes import wintypes
from pathlib import Path


def age_lease_file(path: Path) -> None:
    if os.name != "nt":
        return

    class BasicInfo(ctypes.Structure):
        _fields_ = [
            ("created", ctypes.c_longlong),
            ("accessed", ctypes.c_longlong),
            ("written", ctypes.c_longlong),
            ("changed", ctypes.c_longlong),
            ("attributes", wintypes.DWORD),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.SetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.SetFileInformationByHandle.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(os.fspath(path), 0x100, 7, None, 3, 0, None)
    if handle in (None, ctypes.c_void_p(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = BasicInfo(changed=time.time_ns() // 100 + 116444736000000000 - 100_000_000)
        if not kernel.SetFileInformationByHandle(
            handle, 0, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)
