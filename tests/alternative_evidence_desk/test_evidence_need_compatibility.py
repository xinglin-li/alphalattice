"""Historical candidate records read under their own contract. The candidate
records sealed at `486cea54` carried references without a target state
(the field arrived with K0 at `2a21d11a`, whose validator then required it
of every reference): such a record reads with its bytes and hash unchanged,
its references' target completeness *unspecified* -- never backfilled as
DELIVERED, never continued by a current plan -- while every current writer
records a state and the write boundary refuses a reference without one.

The historical fixture is the current writer's own record with the
`target_state` member removed from its REFERENCE needs: byte-for-byte the
serialization of `486cea54`, whose contract had no such field (an absent
optional member is what that writer emitted, and the identity hash of the
two dumps is the same). `fixtures/candidate-receipt-486cea54.json` retains
one such receipt beside the test, produced the same way."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alphalattice.evidence.alternative_evidence.analysis import packet as packet_module
from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceRetrievalAccessReceipt,
    EvidenceNeedRecord,
    LitigationMatterRecord,
)
from alphalattice.evidence.alternative_evidence.analysis.matters import EvidenceNeed
from alphalattice.evidence.alternative_evidence.contracts import seal_contract
from tests.alternative_evidence_desk.document_intelligence_support import (
    _document_with_text as _document,
)
from tests.alternative_evidence_desk.document_intelligence_support import _open_recorded
from tests.alternative_evidence_desk.litigation_support import _matters_filing

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "candidate-receipt-486cea54.json"


def historical_serialization(record: LitigationMatterRecord) -> dict[str, Any]:
    """The record as the `486cea54` writer serialized it: no reference
    carries a target state, because that contract had none."""

    payload = record.model_dump(mode="json")
    for need in payload["needs"]:
        if need["kind"] == "REFERENCE":
            need.pop("target_state", None)
    return payload


def test_a_historical_candidate_record_reads_unchanged_and_is_never_completed_or_continued(
    tmp_path: Path,
) -> None:
    """requirement (3): the old reference with the field absent reads under its
    own contract with its hash unchanged; the old non-reference record reads
    as before; the new writer emits the state and refuses a reference
    without one; an inconsistent state is refused; the summary discloses
    the unspecified references and never grants them completeness; the
    continuation owners refuse the historical prior by name."""

    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path, entities=("AAPL",), documents=(_document("AAPL", text=_matters_filing()),)
    )
    try:
        # The current writer is the integrated selection's.
        current_receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        record = current_receipt.litigation_matters
        assert record is not None
        references = [n for n in record.needs if n.kind == "REFERENCE"]
        assert references and all(n.target_state is not None for n in references)
        assert record.historical_reference_needs == 0
        # The historical form: the same bytes the old writer emitted, the same
        # identity, the references' completeness unspecified, nothing backfilled.
        historical_payload = historical_serialization(record)
        historical = LitigationMatterRecord.model_validate(historical_payload)
        assert historical.model_dump(mode="json") == historical_payload
        assert historical.historical_reference_needs == len(references)
        old_references = [n for n in historical.needs if n.kind == "REFERENCE"]
        assert all(
            n.target_state is None and n.target_completeness_unspecified for n in old_references
        )
        assert [n.status for n in old_references] == [n.status for n in references]
        assert not any(n.target_state == "DELIVERED" for n in old_references)
        assert [n for n in historical.needs if n.kind != "REFERENCE"] == [
            n for n in record.needs if n.kind != "REFERENCE"
        ]
        historical_receipt = seal_contract(
            AlternativeEvidenceRetrievalAccessReceipt,
            "receipt_hash",
            **{
                **{
                    name: getattr(current_receipt, name)
                    for name in AlternativeEvidenceRetrievalAccessReceipt.model_fields
                    if name != "receipt_hash"
                },
                "litigation_matters": historical,
            },
        )
        assert historical_receipt.receipt_hash != current_receipt.receipt_hash
        readback = AlternativeEvidenceRetrievalAccessReceipt.model_validate_json(
            historical_receipt.model_dump_json()
        )
        assert readback.receipt_hash == historical_receipt.receipt_hash
        assert readback == historical_receipt
        retained = AlternativeEvidenceRetrievalAccessReceipt.model_validate_json(
            FIXTURE.read_bytes()
        )
        assert retained.litigation_matters is not None
        assert retained.litigation_matters.historical_reference_needs > 0
        assert all(
            n.target_state is None
            for n in retained.litigation_matters.needs
            if n.kind == "REFERENCE"
        )
        # The summary discloses them as unspecified; nothing counts them delivered.
        summary = packet_module.litigation_matter_summary(historical_receipt)
        assert summary is not None
        assert summary["needs"]["target_completeness_unspecified"] == len(references)
        assert "cannot be continued" in summary["needs"]["rule"]
        assert (
            packet_module.litigation_matter_summary(current_receipt)["needs"][
                "target_completeness_unspecified"
            ]
            == 0
        )
        # The current write boundary: a reference states what its target got.
        reference_need = EvidenceNeed(
            document_key="d",
            matter_handle=None,
            kind="REFERENCE",
            priority=0,
            character_start=None,
            character_end=None,
            source_bytes=0,
            part=1,
            part_count=1,
            region_heading="NOTE 7",
            detail="note reference 'Note 9' -> region: NOTE 9",
        )
        inspected = next(iter(_inspected(runtime, document_set, generation, request)))
        with pytest.raises(ValueError, match="evidence_need_target_state_required"):
            packet_module._need_record(inspected, reference_need, "UNRESOLVED", None, "x")
        assert (
            packet_module._need_record(
                inspected, reference_need, "UNRESOLVED", None, "x", target_state="UNRESOLVED"
            ).target_state
            == "UNRESOLVED"
        )
        with pytest.raises(ValueError, match="evidence_need_invalid"):
            EvidenceNeedRecord(
                document_handle="DOC-AAPL-1",
                kind="REFERENCE",
                status="PROVIDED",
                span_handle="SPAN-M01-R0001",
                target_state="LOCATED",
            )
        with pytest.raises(ValueError, match="evidence_need_invalid"):
            EvidenceNeedRecord(
                document_handle="DOC-AAPL-1",
                kind="LEAD",
                matter_handle="M-AAPL-1-001",
                status="PENDING_ALLOWANCE",
                character_start=0,
                character_end=10,
                target_state="LOCATED",
            )
        # No current plan continues the historical prior: the runtime's
        # continuation owner refuses it by name.
        with pytest.raises(ValueError, match="continuation_prior_incompatible"):
            runtime.continue_evidence(
                request=request,
                document_set=document_set,
                generation=generation,
                prior_receipt=historical_receipt,
                prior_spans=spans,
                session_limit=3,
                window_limit=192,
            )
    finally:
        runtime.close()


def _inspected(runtime: Any, document_set: Any, generation: Any, request: Any) -> tuple[Any, ...]:
    session = runtime.retrieval.open_session(
        document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
    )
    try:
        return tuple(session.inspect_documents())
    finally:
        session.close()
