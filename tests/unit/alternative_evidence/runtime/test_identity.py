import pytest

from alphalattice.evidence.alternative_evidence.runtime.identity import (
    SUPPORTED_HISTORICAL_BINDINGS,
    observed_historical_bindings,
)

TEST_SOURCE = "src/alphalattice/evidence/alternative_evidence/runtime/identity.py"


@pytest.mark.proves("alternative_evidence.historical_identity_excludes_current_and_known")
def test_observed_history_excludes_current_and_supported_and_deduplicates_unknown():
    current = tuple("a" * 64 for _ in range(6))
    known = SUPPORTED_HISTORICAL_BINDINGS[0].bindings
    unknown = tuple("b" * 64 for _ in range(6))
    result = observed_historical_bindings((unknown, current, known, unknown), current=current)
    assert [item.bindings for item in result] == [unknown]
