"""A stage's failure cause keeps a source's recorded facts and every earlier record's bytes."""

from __future__ import annotations

from alphalattice.control.task_control.contracts import StageFailureCause


def test_a_cause_recorded_before_source_facts_reads_back_byte_for_byte() -> None:
    """compatibility: a six-field cause recorded before source facts serializes as it was stored,
    so the stopped state that sealed it keeps its hash."""
    stored = {
        "exception_type": "ImportError",
        "detail": "DLL load failed",
        "step": "fit",
        "unit": "AAA",
        "first_session": "2026-09-01",
        "last_session": None,
    }
    assert StageFailureCause.model_validate(stored).model_dump(mode="json") == stored


def test_a_sources_unknown_row_count_and_sanitizer_code_are_kept() -> None:
    """requirement: a source's recorded row count and sanitizer subcode, unknown markers
    included, reach the stage's cause."""
    facts = {"exception_type": "UNKNOWN", "row_count": "UNKNOWN", "sanitizer_code": "UNKNOWN"}
    cause = StageFailureCause.from_facts(facts)
    assert cause is not None
    assert cause.model_dump(mode="json") | facts == cause.model_dump(mode="json")
