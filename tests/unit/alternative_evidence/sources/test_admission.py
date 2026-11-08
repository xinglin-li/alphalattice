import pytest

from alphalattice.evidence.alternative_evidence.sources.admission import (
    CONSENT,
    DEFAULT_SOURCE_CONSENT,
    evidence_source_consent,
)

TEST_SOURCE = "src/alphalattice/evidence/alternative_evidence/sources/admission.py"


@pytest.mark.proves("alternative_evidence.absent_consent_uses_the_default")
def test_an_absent_consent_uses_the_default_but_an_unreadable_one_does_not(tmp_path):
    assert evidence_source_consent(tmp_path) == DEFAULT_SOURCE_CONSENT
    target = tmp_path / CONSENT
    target.parent.mkdir(parents=True)
    target.write_text("{", encoding="utf-8")
    assert evidence_source_consent(tmp_path).actor == "UNREADABLE"
