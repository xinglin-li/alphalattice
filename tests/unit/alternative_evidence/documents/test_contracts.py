"""A document's availability, named by what it rests on."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from alphalattice.evidence.alternative_evidence.documents.contracts import (
    AlternativeEvidenceDocumentReference,
)

TEST_SOURCE = "src/alphalattice/evidence/alternative_evidence/documents/contracts.py"
ACCEPTED = datetime(2026, 2, 1, 21, 30, tzinfo=UTC)
CAPTURED = ACCEPTED + timedelta(days=30)


@pytest.mark.proves("alternative_evidence.availability_names_its_basis")
@pytest.mark.parametrize(
    ("source", "accepted", "available", "basis"),
    [
        pytest.param("SEC_EDGAR", ACCEPTED, ACCEPTED, "OFFICIAL_ACCEPTANCE", id="acceptance"),
        pytest.param("ISSUER_RECORDED", ACCEPTED, CAPTURED, "RECORDED_IMPORT_CAPTURE", id="import"),
        pytest.param(
            "ISSUER_RECORDED",
            None,
            CAPTURED,
            "RECORDED_IMPORT_CAPTURE_ACCEPTANCE_UNKNOWN",
            id="import, acceptance unknown",
        ),
        pytest.param("SEC_EDGAR", None, CAPTURED, "SOURCE_STATED", id="source stated"),
    ],
)
def test_a_documents_availability_names_what_it_rests_on(source, accepted, available, basis):
    """Availability equal to the acceptance is the acceptance; a recorded import's is its
    capture, its acceptance known or not; any other is the source's statement."""
    assert AlternativeEvidenceDocumentReference.availability_basis(source, accepted, available) == (
        basis
    )
