import pytest

from alphalattice.evidence.alternative_evidence.sources.admission import (
    CONSENT,
    DEFAULT_SOURCE_CONSENT,
    evidence_source_consent,
)
from alphalattice.evidence.alternative_evidence.sources.campaign import require_campaign_id

TEST_SOURCE = "src/alphalattice/evidence/alternative_evidence/sources/admission.py"


@pytest.mark.proves("alternative_evidence.absent_consent_uses_the_default")
def test_an_absent_consent_uses_the_default_but_an_unreadable_one_does_not(tmp_path):
    consent = evidence_source_consent(tmp_path)
    assert consent == DEFAULT_SOURCE_CONSENT
    assert require_campaign_id(consent.campaign_id) == consent.campaign_id
    assert len(consent.campaign_id.removeprefix("consent-")) == 16
    target = tmp_path / CONSENT
    target.parent.mkdir(parents=True)
    target.write_text("{", encoding="utf-8")
    assert evidence_source_consent(tmp_path).actor == "UNREADABLE"
