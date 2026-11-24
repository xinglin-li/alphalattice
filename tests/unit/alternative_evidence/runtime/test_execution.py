import pytest

from alphalattice.evidence.alternative_evidence.runtime import execution

TEST_SOURCE = "src/alphalattice/evidence/alternative_evidence/runtime/execution.py"


@pytest.mark.proves("alternative_evidence.a_later_unit_wait_ends_at_its_bound")
def test_a_later_unit_never_waits_without_end_on_the_first(monkeypatch):
    """A later unit's wait for the book's first unit fails by name when the first unit is still
    pending at the deadline, and ends at once, unsettled, on a Task that ended or failed."""
    clock = iter([0.0, 0.0, 1.0, execution.FIRST_UNIT_WAIT_SECONDS + 1.0])
    monkeypatch.setattr(execution.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(execution, "pause", lambda _seconds: False)
    monkeypatch.setattr(execution, "checkpoint", lambda: None)
    with pytest.raises(RuntimeError, match=r"^alternative_evidence\.first_unit_wait_expired$"):
        execution.first_unit_settled(lambda: "PENDING")
    clock = iter([0.0, 0.0])
    assert execution.first_unit_settled(lambda: None) is False


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
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        second = leases.grant("second", request)
        assert first.amounts.cpu + second.amounts.cpu == 4
        with use_resource_lease(second):
            assert worker_width(4) == cpu_width(4) == 2
            assert worker_width(1) == cpu_width(1) == 1
            read = carry_resource_lease(current_resource_lease)
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
