"""A writer's times while its wall clock reads frozen or stepped back, for latest-row reads."""

from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from unittest import mock

import alphalattice.kernel.clock as clock
from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease


def stepped_walls(at: datetime) -> dict[str, tuple[datetime, datetime]]:
    """Two wall readings each: the second ties the first, or runs an hour back from it."""
    return {"frozen": (at, at), "stepped-back": (at, at - timedelta(hours=1))}


@contextmanager
def writer_wall(root: Path, readings: tuple[datetime, ...]) -> Iterator[None]:
    """A writer lease held on `root` while the wall reads `readings`, then the last of them."""
    left, last = iter(readings), [readings[0]]

    def now(tz: tzinfo | None = None) -> datetime:
        last[0] = next(left, last[0])
        return last[0]

    with (
        mock.patch.object(clock, "datetime", mock.Mock(now=now)),
        closing(WorkspaceWriterLease.acquire(root, keep_facts=False)),
    ):
        yield
