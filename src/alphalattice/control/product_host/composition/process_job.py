"""Own Windows children and their descendants until their parent owner closes."""

from __future__ import annotations

import ctypes
from ctypes import wintypes


class _Limits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _Limits),
        ("IoInfo", ctypes.c_ulonglong * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class ChildJob:
    """Keep the sole non-inheritable kill-on-close Job handle in its parent owner."""

    def __init__(self, *, affinity: int | None = None, priority: int | None = None) -> None:
        """Create this parent's Job with its caller's optional affinity and priority."""
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.ResumeThread.argtypes = [wintypes.HANDLE]
        self.kernel.ResumeThread.restype = wintypes.DWORD
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError("owned Job unavailable")
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if affinity is not None:
            limits.BasicLimitInformation.LimitFlags |= 0x10
            limits.BasicLimitInformation.Affinity = affinity
        if priority is not None:
            limits.BasicLimitInformation.LimitFlags |= 0x20
            limits.BasicLimitInformation.PriorityClass = priority
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            self.close()
            raise OSError("owned Job limit unavailable")

    def assign(self, process: int) -> None:
        """Assign the caller-owned process handle to this Job."""
        if not self.kernel.AssignProcessToJobObject(self.handle, process):
            raise ctypes.WinError(ctypes.get_last_error())

    def assign_and_resume(self, process: int, thread: int) -> None:
        """Admit the suspended child before it can run any Host code."""
        self.assign(process)
        if self.kernel.ResumeThread(thread) == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        """Release only this owner's handle, ending any remaining descendants."""
        import _winapi

        if self.handle:
            _winapi.CloseHandle(self.handle)
            self.handle = None
