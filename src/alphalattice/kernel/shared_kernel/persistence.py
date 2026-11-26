"""Public durable-file persistence primitives shared by bounded contexts."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path

from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError


def fsync_directory(path: Path) -> None:
    """Flush directory metadata on POSIX; Windows has no equivalent here."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path: Path, data: bytes) -> None:
    """Create and durably write a new file without overwriting an existing target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(path.parent)


QUICK_REPLACE_DELAYS: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1)
"""Catalog swaps: a reader holds the file for an instant, or the swap is refused."""

DURABLE_REPLACE_DELAYS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.3)
"""Publications read by pages and scanners: about one second before refusal."""


def replace_with_retry(
    source: Path, target: Path, *, delays: Sequence[float] = QUICK_REPLACE_DELAYS
) -> None:
    """Replace ``target`` with ``source`` atomically, retrying a sharing violation.

    A reader holding the target for an instant -- a page polling the file, a
    scanner opening the fresh source -- makes ``os.replace`` raise
    ``PermissionError`` on Windows. The replace is retried after each delay in
    ``delays``; a refusal that outlasts them is raised as a typed, retryable
    conflict with the OS error as its cause. Any other failure is raised as
    it is. When the replace does not succeed the target is untouched.
    """
    for attempt, delay in enumerate((*delays, 0.0)):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            if attempt == len(delays):
                raise WorkspaceConflictError(
                    f"replacement of {target.name} is blocked by an open handle",
                    code="catalog.replace_blocked",
                ) from exc
            time.sleep(delay)
