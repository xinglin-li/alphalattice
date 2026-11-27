import pytest

from alphalattice.evidence.alternative_evidence.runtime import identity
from alphalattice.kernel.identity import canonical_hash

TEST_SOURCE = "src/alphalattice/evidence/alternative_evidence/runtime/identity.py"


@pytest.mark.proves("alternative_evidence.historical_identity_excludes_current_and_known")
def test_observed_history_excludes_current_and_supported_and_deduplicates_unknown(monkeypatch):
    current = (canonical_hash("current"),) * 6
    known = identity.HistoricalEvidenceBindings(*((canonical_hash("known"),) * 6))
    unknown = (canonical_hash("unknown"),) * 6
    monkeypatch.setattr(identity, "SUPPORTED_HISTORICAL_BINDINGS", (known,))
    result = identity.observed_historical_bindings(
        (unknown, current, known.bindings, unknown), current=current
    )
    assert [item.bindings for item in result] == [unknown]
