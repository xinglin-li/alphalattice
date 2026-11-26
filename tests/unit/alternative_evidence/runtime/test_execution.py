import os
import time

import pytest

from alphalattice.evidence.alternative_evidence.runtime import execution

TEST_SOURCE = "src/alphalattice/evidence/alternative_evidence/runtime/execution.py"


@pytest.mark.proves("alternative_evidence.a_later_unit_wait_ends_at_its_bound")
def test_a_later_unit_never_waits_without_end_on_the_first(monkeypatch):
    """A later unit's wait for the book's first unit fails by name when the first unit is still
    pending at the deadline, ends at once, unsettled, on a Task that ended or failed, and reads
    again on the Task's event rather than on a poll."""
    subject = ("task", "first-unit")
    monkeypatch.setattr(execution, "checkpoint", lambda: None)
    monkeypatch.setattr(execution, "FIRST_UNIT_WAIT_SECONDS", 0.05)
    with pytest.raises(RuntimeError, match=r"^alternative_evidence\.first_unit_wait_expired$"):
        execution.first_unit_settled(lambda: "PENDING", subject)
    assert execution.first_unit_settled(lambda: None, subject) is False
    monkeypatch.setattr(execution, "FIRST_UNIT_WAIT_SECONDS", 60.0)
    reads = []

    def state():  # the first unit's stage row lands just after the first read
        reads.append(time.monotonic())
        if len(reads) == 1:
            execution.EVENTS.publish(subject)
            return "PENDING"
        return "PREPARED"

    assert execution.first_unit_settled(state, subject) is True
    assert reads[1] - reads[0] < 0.5  # read again on the event, inside the old 0.5 s poll


@pytest.mark.proves("resources.task_grants_obey_the_budget")
@pytest.mark.parametrize(
    "case",
    [
        "grant",
        "over-budget",
        "release",
        "process-exit",
        "share-four-cores",
    ],
)
def test_resource_leases_keep_one_budget_and_reclaim_the_exact_holder(monkeypatch, case) -> None:
    """Only the Resources owner grants capacity, and a completed holder returns its whole grant."""
    from alphalattice.kernel.resources import (
        ResourceAmounts,
        ResourceLeases,
        ResourceRefused,
        carry_resource_lease,
        cpu_width,
        current_resource_lease,
        use_resource_lease,
        worker_width,
    )

    budget = ResourceAmounts(
        cpu=4, memory_bytes=1024, workers=4, disk_bytes=512, connections=4, network=2, spend=6
    )
    request = ResourceAmounts(
        cpu=2, memory_bytes=512, workers=2, disk_bytes=256, connections=2, network=1, spend=3
    )
    leases = ResourceLeases(budget)
    first = leases.grant("first", request)
    assert first.amounts == request and leases.available == request
    if case == "grant":
        return
    if case == "over-budget":
        with pytest.raises(ResourceRefused, match=r"^resources.capacity_exceeded:spend$"):
            leases.grant("too-expensive", ResourceAmounts(cpu=1, spend=4))
        assert leases.available == request
        return
    if case == "share-four-cores":
        from alphalattice.kernel import resources
        from alphalattice.kernel.config import Config, current, using
        from alphalattice.kernel.resources import ResourceLease

        second = leases.grant("second", request)
        assert first.amounts.cpu + second.amounts.cpu == 4
        with using(Config({})), use_resource_lease(second):
            assert worker_width(4) == cpu_width(4) == 2
            assert worker_width(1) == cpu_width(1) == 1
            read = carry_resource_lease(current_resource_lease)
        for name in resources.NATIVE_THREAD_VARIABLES:
            monkeypatch.setenv(name, "9")
        larger = ResourceLease("later", ResourceAmounts(cpu=8))
        for inlet, expected, later in (
            ({}, ("2", "2", "2"), 8),
            (
                {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "9"},
                ("1", "2", "2"),
                3,
            ),
        ):
            with using(Config(inlet)):
                with use_resource_lease(second):
                    resources.bound_native_threads(4)
                actual = tuple(os.environ[name] for name in resources.NATIVE_THREAD_VARIABLES)
                assert actual == expected
                assert current().child_environment() == inlet
                with use_resource_lease(larger):
                    assert cpu_width() == later
        assert read() is second
        assert current_resource_lease() is None
        with pytest.raises(ResourceRefused, match=r"^resources.capacity_exceeded:cpu$"):
            leases.grant("third", ResourceAmounts(cpu=1))
        assert leases.available == ResourceAmounts()
        return
    if case == "process-exit":
        leases.process_exited(first)
        assert leases.available == budget
        replacement = leases.grant("first", request)
        leases.process_exited(first)  # a late exit cannot reclaim a replacement holder
        assert leases.available == request
        leases.release(replacement)
    else:
        leases.release(first)
        leases.release(first)
    assert leases.available == budget
