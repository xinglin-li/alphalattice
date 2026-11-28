"""Cross-process writer lease for one local desktop workspace."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO

from alphalattice.control.workspace_runtime.verified_facts import (
    keep_file_facts,
    release_file_facts,
)
from alphalattice.kernel.clock import IncreasingClock, hold_writer_clock

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _seed(handle: BinaryIO, workspace: Path) -> datetime | None:
    """The bound the root's last writer kept after its lock byte, in microseconds since the epoch;
    for a root from before the kept bound, the latest time its store's rows hold, read once."""
    handle.seek(1)
    kept = handle.read(8)
    if len(kept) != 8:
        if not (workspace / "market-data.duckdb").is_file():
            return None  # a new root: no earlier times
        import alphalattice.control.workspace_runtime.database as store  # lazy: a legacy store

        return store.WorkspaceDatabase(workspace).latest_stored_time()
    try:
        return _EPOCH + timedelta(microseconds=int.from_bytes(kept, "little"))
    except OverflowError:
        return None  # not a bound this lease wrote; the first kept bound replaces it


def _keeper(handle: BinaryIO) -> Callable[[datetime], None]:
    def keep(bound: datetime) -> None:
        handle.seek(1)
        handle.write(((bound - _EPOCH) // timedelta(microseconds=1)).to_bytes(8, "little"))
        handle.flush()

    return keep


@dataclass
class WorkspaceWriterLease:
    """An advisory OS lock held for the lifetime of a writable runtime.

    A writable holder also keeps the workspace's file facts (`verified_facts`) while
    it holds the lock.
    """

    _handle: BinaryIO | None
    _workspace: Path | None = None
    clock: IncreasingClock | None = None
    """This root's own increasing clock while the lease is held, for the writes that go to it."""
    _clock: Callable[[], None] = lambda: None
    """The end of the root's increasing clock (`hold_writer_clock`), seeded and kept in the lock
    file after its lock byte: a restart after the clock stepped back still hands out later times."""

    @property
    def held(self) -> bool:
        """Report whether this lease still retains its OS-lock handle.

        Returns:
            True while the handle remains held.
        """
        return self._handle is not None

    @classmethod
    def acquire(cls, workspace: Path, *, keep_facts: bool = True) -> WorkspaceWriterLease:
        """Acquire the workspace's nonblocking process-lifetime writer lease.

        Args:
            workspace: Physical workspace root containing the control or writer-lock file.
            keep_facts: Keep persistent file facts when the holder will write the workspace.

        Returns:
            Held lease; the caller must close it when the writable runtime ends.

        Raises:
            RuntimeError: `workspace_runtime.writer_already_owned`, the workspace already has
                a writer owner, such as its running Host.
        """
        workspace.mkdir(parents=True, exist_ok=True)
        path = workspace / ".alphalattice-writer.lock"
        path.touch()  # read and written in place: the kept bound follows the lock byte
        handle = path.open("r+b", buffering=0)  # each seek moves the descriptor the lock reads
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
        lease = cls(_handle=handle)
        try:
            seed = _seed(handle, workspace)
        except BaseException:
            lease.close()  # an unreadable store leaves the root unheld
            raise
        if keep_facts:
            keep_file_facts(workspace)
            lease._workspace = workspace
        lease.clock, lease._clock = hold_writer_clock(seed, _keeper(handle))
        return lease

    def close(self) -> None:
        """Release the advisory writer lock and close its handle; absence is a no-op."""
        handle = self._handle
        if handle is None:
            return
        self._clock()
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
