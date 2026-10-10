"""Cross-process writer lease for one local desktop workspace."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from alphalattice.control.workspace_runtime.verified_facts import (
    keep_file_facts,
    release_file_facts,
)


@dataclass
class WorkspaceWriterLease:
    """An advisory OS lock held for the lifetime of a writable runtime.

    Its holder is the workspace's one writer, so it also keeps the workspace's file facts
    (`verified_facts`) while it holds the lock.
    """

    _handle: BinaryIO | None
    _workspace: Path | None = None

    @property
    def held(self) -> bool:
        """Report whether this lease still retains its OS-lock handle.

        Returns:
            True while the handle remains held.
        """
        return self._handle is not None

    @classmethod
    def acquire(cls, workspace: Path) -> WorkspaceWriterLease:
        """Acquire the workspace's nonblocking process-lifetime writer lease.

        Args:
            workspace: Physical workspace root containing the control or writer-lock file.

        Returns:
            Held lease; the caller must close it when the writable runtime ends.

        Raises:
            RuntimeError: `workspace_runtime.writer_already_owned`, the workspace already has
                a writer owner, such as its running Host.
        """
        workspace.mkdir(parents=True, exist_ok=True)
        path = workspace / ".alphalattice-writer.lock"
        handle = path.open("a+b")
        try:
            handle.seek(0)
            if handle.read(1) == b"":
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - exercised by a future Linux playpen lane.
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("workspace_runtime.writer_already_owned") from exc
        keep_file_facts(workspace)
        return cls(_handle=handle, _workspace=workspace)

    def close(self) -> None:
        """Release the advisory writer lock and close its handle; absence is a no-op."""
        handle = self._handle
        if handle is None:
            return
        if self._workspace is not None:
            release_file_facts(self._workspace)
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - exercised by a future Linux playpen lane.
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        self._handle = None
