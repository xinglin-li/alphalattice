"""The per-document cap's one default (10,000,000 bytes), read by the source
policy, the product's admitted policy and the materializer's flag alike; an
explicit smaller cap enforced; the ceiling refused; a retained body between
2 and 10 MB reused under the default without a download; and a request
sealed with the earlier 2,000,000 read back exactly as sealed.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from alphalattice.control.product_host.composition.evidence_review_workspace import (
    live_evidence_policy,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    DEFAULT_SOURCE_DOCUMENT_BYTES,
    MAXIMUM_SOURCE_DOCUMENT_BYTES,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
)
from alphalattice.evidence.alternative_evidence.runtime.policy import AdmittedEvidencePolicy
from alphalattice.evidence.alternative_evidence.sources.admission import admit_official_source
from tests.alternative_evidence_desk.incremental_acquisition_support import (
    AAPL,
    TEN_K,
    _acquire,
    _outcomes,
    _request,
    _source_objects,
    _transport,
)
from tests.alternative_evidence_desk.planted_corpus import _runtime
from tests.alternative_evidence_desk.sec_scenario_transport import (
    NOW,
    SecScenarioTransport,
    filing_body,
)


def test_one_default_is_read_by_the_policy_the_product_and_the_materializer() -> None:
    """requirement (section 3): the normal default is 10,000,000 bytes, the
    contract's ceiling; the product's admitted policy and the materializer's
    `--maximum-document-bytes` read the same definition, so no user needs a
    flag to obtain a supported annual filing."""

    from scripts import materialize_evidence_cro_authority as materializer

    assert DEFAULT_SOURCE_DOCUMENT_BYTES == 10_000_000
    assert MAXIMUM_SOURCE_DOCUMENT_BYTES == 10_000_000
    assert AlternativeEvidenceSourcePolicy().maximum_document_bytes == DEFAULT_SOURCE_DOCUMENT_BYTES
    assert AdmittedEvidencePolicy().source_policy.maximum_document_bytes == (
        DEFAULT_SOURCE_DOCUMENT_BYTES
    )
    live = live_evidence_policy(AdmittedEvidencePolicy())
    assert live.source_policy.maximum_document_bytes == DEFAULT_SOURCE_DOCUMENT_BYTES
    parsed = materializer._parser().parse_args(["--workspace", "x"])
    assert parsed.maximum_document_bytes == DEFAULT_SOURCE_DOCUMENT_BYTES
    # An explicit override on either entry point is the operator's, and the
    # ceiling is refused wherever it is declared.
    assert (
        materializer._parser()
        .parse_args(["--workspace", "x", "--maximum-document-bytes", "2000000"])
        .maximum_document_bytes
        == 2_000_000
    )
    assert (
        admit_official_source(network_consent=False, maximum_document_bytes=2_000_000)
    ).maximum_document_bytes == 2_000_000
    with pytest.raises(ValidationError):
        AlternativeEvidenceSourcePolicy(maximum_document_bytes=MAXIMUM_SOURCE_DOCUMENT_BYTES + 1)
    with pytest.raises(ValidationError):
        admit_official_source(
            network_consent=True, maximum_document_bytes=MAXIMUM_SOURCE_DOCUMENT_BYTES + 1
        )
    with pytest.raises(ValidationError):
        AlternativeEvidenceSourcePolicy(maximum_document_bytes=999)


def test_a_retained_body_between_two_and_ten_megabytes_is_reused_under_the_default(
    tmp_path: Path,
) -> None:
    """requirement (section 3): a body of about three megabytes acquired
    under an explicit cap is reused without a download by a request under
    the default; an explicit smaller cap on a later request still defers it
    by name; the same request at the ceiling reuses it again."""

    transport = _transport()
    big = filing_body(TEN_K.accession, words=420_000)
    assert 2_000_000 < len(big) < 10_000_000
    transport.bodies[SecScenarioTransport.locator(AAPL, TEN_K)] = big
    runtime = _runtime(tmp_path)
    try:
        explicit = _request(cap=10_000_000)
        assert explicit.source_policy.maximum_document_bytes == 10_000_000
        registry, first, _set = _acquire(runtime, transport, explicit)
        assert _outcomes(first)[TEN_K.accession] == "FETCHED"
        transport.reset_calls()

        default = _request(as_of=NOW + timedelta(hours=1))
        assert default.source_policy == AlternativeEvidenceSourcePolicy(
            maximum_documents_per_issuer=3
        ), "the default policy names no cap of its own"
        _r, snapshot, source_set = _acquire(runtime, transport, default, registry=registry)
        assert transport.body_calls == [] and snapshot.status.value == "COMPLETE"
        assert _outcomes(snapshot)[TEN_K.accession] == "REUSED_LOCAL"
        assert next(
            d for d in source_set.documents if d.revision == TEN_K.accession
        ).content_bytes == (len(big))

        smaller = _request(as_of=NOW + timedelta(hours=2), cap=2_000_000)
        _r, deferred, _s = _acquire(runtime, transport, smaller, registry=registry)
        assert transport.body_calls == [] and _outcomes(deferred)[TEN_K.accession] == "DEFERRED"
        assert any(
            f"retained body of {len(big):,} bytes exceeds the 2,000,000-byte document cap" in v
            for v in deferred.limitations
        )
        assert len(_source_objects(runtime)) == 3
    finally:
        runtime.close()


def test_a_request_sealed_with_the_earlier_cap_reads_back_as_sealed(tmp_path: Path) -> None:
    """requirement (section 3): a historical request carries its cap
    explicitly and keeps it -- 2,000,000 stays 2,000,000 on readback, its
    hash unchanged -- while a request sealed today carries 10,000,000
    explicitly, so no representation is silently reinterpreted."""

    historical = _request(cap=2_000_000)
    payload = json.loads(historical.model_dump_json())
    assert payload["source_policy"]["maximum_document_bytes"] == 2_000_000
    read = AlternativeEvidenceRequest.model_validate(payload)
    assert read.source_policy.maximum_document_bytes == 2_000_000
    assert read.request_hash == historical.request_hash
    today = _request()
    assert json.loads(today.model_dump_json())["source_policy"]["maximum_document_bytes"] == (
        DEFAULT_SOURCE_DOCUMENT_BYTES
    )
    assert today.request_hash != historical.request_hash, "another cap is another request"
    # A representation that omitted the field would take the default; every
    # sealed request this system has written carries the field, so none
    # exists to reinterpret -- the retained workspaces were inspected for
    # it (259 of 259 requests carry it explicitly).
    assert "maximum_document_bytes" in AlternativeEvidenceSourcePolicy.model_fields
