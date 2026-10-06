"""Cross-platform, non-blocking OS ownership for workspace mutations."""

from __future__ import annotations

import os
import socket
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import BinaryIO, Literal, cast
from uuid import uuid4

from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError
from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes

_PROCESS_STARTED_AT = datetime.now(UTC)
_LOCK_INITIALIZATION_GUARD = Lock()


def _apply_windows_lock(
    handle: BinaryIO,
    mode_name: Literal["LK_NBLCK", "LK_UNLCK"],
) -> None:
    members = vars(import_module("msvcrt"))
    locking = cast(Callable[[int, int, int], None], members["locking"])
    locking(handle.fileno(), int(members[mode_name]), 1)


class WorkspaceLock:
    """One-byte advisory lock with diagnostic owner metadata."""

    def __init__(self, path: Path) -> None:
        """Prepare a nonblocking OS writer lock and its ownership token.

        Args:
            path: Physical file path owned by the caller.
        """
        self.path = path
        self._file: BinaryIO | None = None
        self.owner_token = uuid4()

    def acquire(self) -> WorkspaceLock:
        """Acquire the exclusive OS lock and persist process ownership metadata.

        Returns:
            This acquired lock.

        Raises:
            WorkspaceConflictError: Another writer already owns the OS lock.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK_INITIALIZATION_GUARD:
            handle = self.path.open("a+b", buffering=0)
            if self.path.stat().st_size == 0:
                handle.write(b" ")
                handle.flush()
                os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                _apply_windows_lock(handle, "LK_NBLCK")
            else:
                fcntl = import_module("fcntl")
                members = vars(fcntl)
                flock = members["flock"]
                flock(
                    handle.fileno(),
                    int(members["LOCK_EX"]) | int(members["LOCK_NB"]),
                )
        except OSError as exc:
            handle.close()
            raise WorkspaceConflictError("workspace already has an active writer") from exc

        try:
            metadata = canonical_json_bytes(
                {
                    "host": socket.gethostname(),
                    "owner_token": self.owner_token,
                    "pid": os.getpid(),
                    "process_started_at": _PROCESS_STARTED_AT,
                }
            )
            handle.seek(0)
            handle.write(metadata)
            handle.truncate(len(metadata))
            handle.flush()
            os.fsync(handle.fileno())
            handle.seek(0)
        except BaseException:
            handle.close()
            raise
        self._file = handle
        return self

    def release(self) -> None:
        """Release the held OS lock and close its file handle; absence is a no-op."""
        if self._file is None:
            return
        handle = self._file
        self._file = None
        handle.seek(0)
        if os.name == "nt":
            _apply_windows_lock(handle, "LK_UNLCK")
        else:
            fcntl = import_module("fcntl")
            members = vars(fcntl)
            flock = members["flock"]
            flock(handle.fileno(), int(members["LOCK_UN"]))
        handle.close()

    def __enter__(self) -> WorkspaceLock:
        """Acquire the writer lock on context entry.

        Returns:
            This acquired lock.

        Raises:
            WorkspaceConflictError: Another writer already owns the OS lock.
        """
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the writer lock on context exit without suppressing exceptions.

        Args:
            exc_type: Exception type passed by the context manager, or None.
            exc_value: Exception passed by the context manager, or None.
            traceback: Exception traceback passed by the context manager, or None.
        """
        self.release()
