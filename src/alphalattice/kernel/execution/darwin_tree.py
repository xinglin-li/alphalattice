"""Track Darwin process identities across session changes and reparenting."""

from __future__ import annotations

import ctypes
import os
import select
import subprocess
import sys
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any

_BOOT = """import os, sys
start, error = map(int, sys.argv[1:3])
os.set_inheritable(error, False)
if os.read(start, 1) != b"1":
    raise SystemExit(1)
os.close(start)
try:
    os.execvpe(sys.argv[3], sys.argv[4:], os.environ)
except OSError as exc:
    os.write(error, str(exc.errno).encode("ascii"))
"""


class _ProcessID(ctypes.Structure):
    _fields_ = [
        ("uuid", ctypes.c_ubyte * 16),
        ("unique", ctypes.c_uint64),
        ("original_parent", ctypes.c_uint64),
        ("version", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("reserved_a", ctypes.c_uint64),
        ("reserved_b", ctypes.c_uint64),
    ]


class DarwinTree:
    """Retain the root and observed ancestor IDs, which are independent of PIDs."""

    def __init__(self, root: int) -> None:
        """Bind the identity before releasing the owned command."""
        self._query = ctypes.CDLL("/usr/lib/libproc.dylib").proc_pidinfo
        self._query.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._query.restype = ctypes.c_int
        identity = self._identity(root)
        if identity is None:
            raise OSError("spawn.darwin_tree_unavailable")
        self._root = identity[0]
        self._seen = {self._root}

    def _identity(self, pid: int) -> tuple[int, int] | None:
        observed = _ProcessID()
        size = ctypes.sizeof(observed)
        if self._query(pid, 17, 0, ctypes.byref(observed), size) != size:
            return None
        return int(observed.unique), int(observed.original_parent)

    def remember(self, members: Iterable[Any]) -> None:
        """Retain original IDs after an observed ancestor exits."""
        for member in members:
            if member.is_running() and (identity := self._identity(member.pid)) is not None:
                self._seen.add(identity[0])

    def members(self, candidates: Iterable[Any]) -> tuple[bool, list[Any]]:
        """Find surviving members through original-parent links and verify the root."""
        children: dict[int, list[tuple[int, Any]]] = defaultdict(list)
        root_live = False
        for member in candidates:
            if member.is_running() and (identity := self._identity(member.pid)) is not None:
                unique, parent = identity
                root_live |= unique == self._root
                children[parent].append((unique, member))
        pending = deque(self._seen)
        visited = self._seen.copy()
        descendants = []
        while pending:
            for unique, member in children[pending.popleft()]:
                if unique != self._root:
                    descendants.append(member)
                if unique not in visited:
                    visited.add(unique)
                    pending.append(unique)
        return root_live, descendants


def start(args: Any, *, options: dict[str, Any]) -> tuple[subprocess.Popen[Any], DarwinTree]:
    """Hold the child on a pipe until its native original-parent identity is anchored."""
    original = list(args) if isinstance(args, list | tuple) else [args]
    executable = options.pop("executable", None)
    if options.pop("shell", False):
        original = [executable or "/bin/sh", "-c", *original]
    child_read, owner_write = os.pipe()
    owner_read, child_write = os.pipe()
    with (
        os.fdopen(child_read, "rb"),
        os.fdopen(owner_write, "wb") as release,
        os.fdopen(owner_read, "rb") as errors,
        os.fdopen(child_write, "wb") as child_errors,
    ):
        descriptors = set(options.pop("pass_fds", ())) | {child_read, child_write}
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                _BOOT,
                str(child_read),
                str(child_write),
                executable or original[0],
                *original,
            ],
            pass_fds=tuple(sorted(descriptors)),
            creationflags=options.pop("creationflags", getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            **options,
        )
        child_errors.close()
        try:
            tree = DarwinTree(process.pid)
            release.write(b"1")
            release.flush()
            if not select.select([errors], [], [], 5)[0]:
                raise OSError("spawn.darwin_tree_unavailable: command did not start")
            if error := errors.read():
                number = int(error)
                raise OSError(number, os.strerror(number), executable or original[0])
            process.args = args
            return process, tree
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            raise
