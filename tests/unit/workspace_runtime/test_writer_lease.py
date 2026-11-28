"""A data root's writer hands out strictly increasing times, across a restart after a step back."""

from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest

from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease
from alphalattice.kernel.clock import ReplayClock, utc_now, writing
from tests._support.stepped_clocks import stepped_walls, writer_wall

TEST_SOURCE = "src/alphalattice/control/workspace_runtime/writer_lease.py"
NOW, EPOCH = datetime(2026, 11, 27, 21, tzinfo=UTC), datetime(1970, 1, 1, tzinfo=UTC)
TICK, AHEAD = timedelta(microseconds=1), timedelta(hours=1)


def _kept(root):
    """The bound a lease keeps after its lock byte, in microseconds since the epoch."""
    held = (root / ".alphalattice-writer.lock").read_bytes()[1:9]
    return EPOCH + TICK * int.from_bytes(held, "little")


@pytest.mark.proves("workspace_runtime.a_writers_times_never_tie_or_run_back")
def test_a_writers_times_never_tie_or_run_back_even_across_a_restart(tmp_path):
    """requirement: each time is the wall clock's, or the last plus a microsecond, and above the
    root's earlier times; each root keeps its own bound, at most 1 ms past it, before it is out."""
    for name, readings in stepped_walls(NOW).items():
        with writer_wall(tmp_path / name, readings):
            assert (name, utc_now(), utc_now()) == (name, NOW, NOW + TICK)
    one_tick = ReplayClock(NOW, ticks=lambda: 0.0, wall=lambda: NOW)  # a rehearsal's clock
    assert (one_tick(), one_tick()) == (NOW, NOW + TICK)
    root, other = tmp_path / "root", tmp_path / "other"
    with closing(WorkspaceWriterLease.acquire(root, keep_facts=False)):
        handed = utc_now()
    assert handed < _kept(root) <= handed + timedelta(milliseconds=1)
    ahead = handed + AHEAD  # the clock stepped back an hour before the next start
    bound = b"0" + ((ahead - EPOCH) // TICK).to_bytes(8, "little")
    (root / ".alphalattice-writer.lock").write_bytes(bound)
    with (
        closing(WorkspaceWriterLease.acquire(root, keep_facts=False)),
        closing(WorkspaceWriterLease.acquire(other, keep_facts=False)) as second,
    ):
        assert (utc_now(), utc_now()) == (ahead + TICK, ahead + 2 * TICK)  # the first root
        with writing(second.clock):  # the other root's writes take its own clock
            own = utc_now()
    assert own < ahead and _kept(other) <= own + timedelta(milliseconds=1)
    assert _kept(root) > ahead + 2 * TICK
    assert utc_now() < ahead  # released: no root's clock answers
