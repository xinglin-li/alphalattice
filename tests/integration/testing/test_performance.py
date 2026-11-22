"""The kernel's performance holder: each kernel operation under a budget measured on the
workshop's main machine, or the running node's own when ALPHALATTICE_PERFORMANCE_BUDGETS names
its record, and a hot data loop's cost with the kernel's owners plugged in.

The rows run only in a serial run of their own (pytest-benchmark measures nothing under xdist):
    ALPHALATTICE_PERFORMANCE_RUN=1 uv run --group performance pytest -p no:xdist -m performance
        tests/integration/testing/test_performance.py --benchmark-precision=0.02
A change that adds a kernel operation adds its row to OPERATIONS and its per-batch owner call to
KERNEL_OWNERS as part of its own proof. Still to come with their implementations: a resource
grant, a Task submit, a dispatch through the door, a clock read and an open through storage.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from functools import partial

import numpy as np
import pyarrow as pa
import pytest

from alphalattice.kernel.data_layer import arrays
from alphalattice.kernel.shared_kernel.arrow_identity import canonical_row_hashes
from devtools.testing.performance import (
    Operation,
    judged,
    node_budget,
    overhead,
    within_bound,
)

TEST_SOURCE = "src/devtools/testing/performance.py"

_LANE = {"close": np.linspace(1.0, 2.0, 65_536), "volume": np.arange(65_536, dtype=np.int64)}

OPERATIONS = (
    # 1 MiB over two lanes: about 0.4 ms on the workshop's machine (11-22); the budget holds 2x.
    Operation("trusted digest", lambda: arrays.digest(_LANE), 0.001),
)

KERNEL_OWNERS: tuple[Callable[[pa.RecordBatch], object], ...] = ()
"""Each kernel owner a data loop calls once per batch (a lease check, a clock read, a span)."""


@pytest.mark.performance
@pytest.mark.proves("testing.each_kernel_operation_stays_within_its_budget")
@pytest.mark.parametrize("operation", OPERATIONS, ids=lambda operation: operation.name)
def test_each_kernel_operation_stays_within_its_budget(benchmark, operation):
    """A kernel operation's mean call is within its node's budget, its 99% interval settled under
    it; an operation the node lists no budget for is measured and reported, not judged."""
    budget = node_budget(operation.name, operation.budget)
    benchmark(operation.call)
    stats = benchmark.stats.stats
    if budget is None:
        pytest.skip(
            f"performance.unbudgeted: {operation.name} takes {stats.mean * 1e6:.1f} us a call; "
            "this node's budget record does not list it, so it is measured, not judged"
        )
    judged(operation.name, stats.mean, stats.stddev, stats.rounds, budget)


def _panel_batches(rows: int = 65_536, batch: int = 16_384) -> list[pa.RecordBatch]:
    """Ordered Panel closure rows, (session, listing, row hash), in the store's batch size."""
    listings = 64
    index = np.arange(rows)
    sessions = pa.array((index // listings).astype("datetime64[D]"), pa.date32())
    names = pa.array([f"listing-{value:04d}" for value in index % listings], pa.string())
    hashes = pa.array([hashlib.sha256(int(value).to_bytes(8)).hexdigest() for value in index])
    table = pa.table({"session_date": sessions, "listing_id": names, "row_hash": hashes})
    return table.to_batches(max_chunksize=batch)


def _hashed(batches: list[pa.RecordBatch], owners: tuple[Callable[..., object], ...]) -> list[str]:
    """Panel rows hashed by the kernel's canonical row identity, each owner called once a batch."""
    return [row for batch in batches for row in _hash_batch(batch, owners)]


def _hash_batch(batch: pa.RecordBatch, owners: tuple[Callable[..., object], ...]) -> list[str]:
    for owner in owners:
        owner(batch)
    return canonical_row_hashes(pa.Table.from_batches([batch]))


@pytest.mark.performance
@pytest.mark.proves("testing.a_hot_loop_costs_at_most_two_percent_more_with_kernel_owners")
def test_the_hot_loop_costs_at_most_two_percent_more_with_the_kernel_owners_plugged_in():
    """Panel row hashing with the kernel's owners plugged in stays within 2% of the bare loop,
    and the owners never change what it hashes."""
    batches = _panel_batches()
    assert _hashed(batches, KERNEL_OWNERS) == _hashed(batches, ())
    plugged = partial(_hash_batch, owners=KERNEL_OWNERS)
    cost = overhead(partial(_hash_batch, owners=()), plugged, batches)
    within_bound(*cost, "the Panel row-hash loop")
