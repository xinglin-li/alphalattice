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
