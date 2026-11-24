"""The kernel's performance holder: each kernel operation under a budget measured on the
workshop's main machine, or the running node's own when ALPHALATTICE_PERFORMANCE_BUDGETS names
its record, and a hot data loop's cost with the kernel's owners plugged in.

The rows run only in a serial run of their own (pytest-benchmark measures nothing under xdist):
    ALPHALATTICE_PERFORMANCE_RUN=1 uv run --group performance pytest -p no:xdist -m performance
        tests/integration/testing/test_performance.py --benchmark-precision=0.02
A change that adds a kernel operation adds its row to OPERATIONS. Operations a data loop
calls once per batch also join KERNEL_OWNERS. Still to come with their implementations:
a resource grant, a clock read and an open through storage.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, MutableMapping
from contextlib import nullcontext
from functools import partial
from inspect import signature
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
import pyarrow as pa
import pytest

from alphalattice.capabilities.domain_contracts.arrow_identity import canonical_row_hashes
from alphalattice.foundation.data_layer import arrays
from alphalattice.interface.local_application import thin_cli
from alphalattice.kernel.door import Door, FamilyRoute
from alphalattice.kernel.execution import process_execution
from alphalattice.kernel.execution.events import Events
from alphalattice.kernel.identity import FileDigestTrust, TrustEntry
from devtools.testing.performance import (
    Operation,
    judged,
    node_budget,
    overhead,
    within_bound,
)
from tests._support.portfolio_strategy_lab_fixtures import live as live

TEST_SOURCE = "src/devtools/testing/performance.py"

_LANE = {"close": np.linspace(1.0, 2.0, 65_536), "volume": np.arange(65_536, dtype=np.int64)}


class _Held:
    """A trust record's store holding its entries in memory, for one process."""

    def __init__(self) -> None:
        self.entries: dict[str, TrustEntry] = {}

    def place(self, path: Path) -> tuple[MutableMapping[str, TrustEntry], str]:
        return self.entries, str(path)

    def kept(self, path: Path, key: str, entry: TrustEntry) -> None:
        return None


def _held_check() -> Callable[[], object]:
    """A stat-gated check of a settled file the trust record holds: what a trusted input costs a
    read once its digest is kept. This settled source file is hashed once here, never again."""
    trust = FileDigestTrust(
        store=_Held(),
        lock=Lock(),
        span=lambda _kind, _detail: nullcontext(),
        time_ns=time.time_ns,
        trusting=lambda: True,
    )
    source = Path(__file__)
    trust.verified(source)
    return lambda: trust.verified(source)


def _dispatch_interactive() -> None:
    process_execution().submit(lambda: None, lane="interactive").result()


def _thin_command(request: pytest.FixtureRequest) -> Callable[[], object]:
    """`study list` through the console's own entry (`thin_cli.invoke`) as an installed console
    runs it, against a running Host: its launcher checks, the endpoint record's lookup, the request
    and the Host's dispatch."""
    live = request.getfixturevalue("live")
    monkeypatch = request.getfixturevalue("monkeypatch")
    for _, variable in thin_cli.AGENT_SESSION_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    console = request.getfixturevalue("tmp_path") / "bin"
    console.mkdir()
    for name in ("alphalattice", "alphalattice.cmd"):  # an installed console on PATH
        (console / name).write_text("", encoding="utf-8")
        (console / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(console) + os.pathsep + os.environ.get("PATH", ""))
    argv = ["--workspace", str(live.workspace), "study", "list"]
    assert thin_cli.invoke(argv) == 0  # the thin path reached the Host
    return partial(thin_cli.invoke, argv)


def _door_echo(value: object) -> object:
    return value


def _door_call() -> Callable[[], object]:
    """One operation selection and return through the resident Door."""
    door: Door[object, object] = Door((FamilyRoute(("ECHO",), __name__, "_door_echo"),))
    return partial(door.dispatch, "ECHO", None)


def _event_delivered(request: pytest.FixtureRequest) -> Callable[[], object]:
    """One event delivered to a waiting subscriber, timed with its reply: a subscriber thread
    blocks on "ping" and publishes "pong" when it hears one; a call publishes ping and waits on
    pong, so each call is two deliveries to a blocked waiter."""
    hub, stopping, ready, seen = Events(), Event(), Event(), [0]

    def answer() -> None:
        with hub.subscribed("ping") as ping:
            ready.set()
            heard = 0
            while not stopping.is_set():
                heard = ping.wait(heard, timeout=60, what="a ping")
                hub.publish("pong")

    replies = hub.subscribed("pong")
    pong = replies.__enter__()
    thread = Thread(target=answer, name="event-answer", daemon=True)
    thread.start()
    assert ready.wait(5)

    def ended() -> None:
        stopping.set()
        hub.publish("ping")
        thread.join(5)
        replies.__exit__(None, None, None)

    request.addfinalizer(ended)

    def call() -> None:
        hub.publish("ping")
        seen[0] = pong.wait(seen[0], timeout=5, what="the reply")

    return call


OPERATIONS = (
    # 1 MiB over two lanes: about 0.4 ms on the workshop's machine (11-22); the budget holds 2x.
    Operation("trusted digest", lambda: arrays.digest(_LANE), 0.001),
    # One open, its stamp and a held entry: about 52 us on the workshop's machine (11-22, three runs
    # of 5,000); the budget holds 2x.
    Operation("a holding stat-gated fact check", _held_check(), 0.0001),
    # The serial nightly row averaged 66 us over 9,681 rounds on this machine (11-22).
    Operation("fast Task dispatch", _dispatch_interactive, 0.0001),
    # About 25 ms on the workshop's machine (11-22, 33 rounds, 99% interval 20 to 31 ms); the budget
    # holds 2x.
    Operation("a thin-CLI command to a running Host", _thin_command, 0.05),
    # Seven warm samples of 100,000 calls here gave a median of 0.089 us (11-22).
    Operation("Door dispatch", _door_call(), 0.0000002),
    # A delivery and its reply: 34 us mean on the workshop's machine (11-23, 539 rounds); the
    # budget holds 2x.
    Operation("one event delivered to a waiting subscriber", _event_delivered, 0.00007),
)

KERNEL_OWNERS: tuple[Callable[[pa.RecordBatch], object], ...] = ()
"""Each kernel owner a data loop calls once per batch (a lease check, a clock read, a span)."""


@pytest.mark.performance
@pytest.mark.proves("testing.each_kernel_operation_stays_within_its_budget")
@pytest.mark.parametrize("operation", OPERATIONS, ids=lambda operation: operation.name)
def test_each_kernel_operation_stays_within_its_budget(benchmark, operation, request):
    """A kernel operation's mean call is within its node's budget, its 99% interval settled under
    it; an operation the node lists no budget for is measured and reported, not judged. A row whose
    call takes pytest's request is first built from the fixtures it needs, a running Host's."""
    budget = node_budget(operation.name, operation.budget)
    call = operation.call(request) if signature(operation.call).parameters else operation.call
    benchmark(call)
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
