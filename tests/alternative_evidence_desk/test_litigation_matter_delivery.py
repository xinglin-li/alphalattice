"""Litigation matter windows are read through the verified reader under
their own per-session allowance, sealed into the receipt beside the question
bank's spans and the typed families', rendered in the packet with the
inventory's accounting, never counted as inspected when the allowance ran
out, and continued by the runtime as a successor receipt: the retrieval-stage
controls for the matter inventory's delivery under the integrated selection.
(The production plan's own reader of these windows went with the plan:
first-release integration T5.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphalattice.evidence.alternative_evidence.analysis import packet as packet_module
from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceRetrievalAccessReceipt,
    LitigationMatterRecord,
)
from alphalattice.evidence.alternative_evidence.contracts import seal_contract
from alphalattice.evidence.alternative_evidence.retrieval.session import (
    MAXIMUM_ISSUED_MATTER_WINDOWS,
)
from tests.alternative_evidence_desk.document_intelligence_support import (
    _document_with_text as _document,
)
from tests.alternative_evidence_desk.document_intelligence_support import (
    _obligation,
    _open_recorded,
)
from tests.alternative_evidence_desk.litigation_support import (
    BOILERPLATE,
    FEDERAL_DERIVATIVE,
    ITEM_CAPTIONS,
    SECURITIES,
    STATE_DERIVATIVE,
    _filing,
)
from tests.alternative_evidence_desk.matter_selection_support import _long_matter


def test_matter_windows_are_verified_spans_of_the_packet_with_their_own_accounting(
    tmp_path: Path,
) -> None:
    text = _filing(
        note_lines=[BOILERPLATE, SECURITIES, FEDERAL_DERIVATIVE, STATE_DERIVATIVE],
        item_lines=ITEM_CAPTIONS,
    )
    runtime, request, _registry, snapshot, document_set, generation = _open_recorded(
        tmp_path, documents=(_document(text=text),)
    )
    try:
        receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        matters = receipt.litigation_matters
        assert isinstance(matters, LitigationMatterRecord)
        assert matters.plan_offset == 0 and matters.prior_windows == 0
        assert matters.pending_windows == 0 and matters.read_windows == matters.planned_windows
        # The integrated selection deals the needs by topic lane, a window
        # joining the needs it touches: the note's three matters and the
        # item captions arrive in fewer, larger windows than one each.
        assert matters.read_windows >= 2
        assert all(handle.startswith("SPAN-M01-R") for handle in matters.span_handles)
        # The three budgets are three accounting entries of one receipt.
        assert set(matters.span_handles).isdisjoint(receipt.read_span_handles)
        typed = receipt.typed_disclosures
        assert typed is not None and set(matters.span_handles).isdisjoint(typed.span_handles)
        assert set(receipt.delivered_span_handles) == {value.span_handle for value in spans}
        (document,) = matters.documents
        assert document.inspection == "COMPLETE"
        assert [m.title[:16] for m in document.matters][:3] == [
            "Washtenaw County",
            "Nathan Silva v. ",
            "Todd Hellrigel v",
        ]
        assert all(m.named for m in document.matters)
        assert document.unassigned and document.unassigned[0].reason.startswith("lead-in")
        # Every read window is a whole excerpt: bytes were planned, not characters.
        read = [w for w in matters.windows if w.status == "READ"]
        assert all(w.delivered_whole for w in read)
        by_handle = {value.span_handle: value for value in spans}
        for window in read:
            span = by_handle[window.span_handle]
            assert span.character_start <= window.character_start
            assert span.character_end >= window.character_end
            assert not any("bounded" in value for value in span.limitations)
            assert window.source_bytes == len(
                text[window.character_start : window.character_end].encode("utf-8")
            )
        assert matters.read_source_bytes == sum(w.source_bytes for w in read)
        # The packet carries the accounting block, the provenance and the view.
        packet = packet_module.AlternativeEvidencePacket(
            request=request,
            obligation=_obligation(request),
            snapshot=snapshot,
            document_set=document_set,
            receipt=receipt,
            spans=spans,
        )
        rendered = packet_module.render_evidence_packet(packet)
        assert '"litigation_matters"' in rendered and '"named_proceeding": true' in rendered
        first = read[0]
        assert f"M01:MATTER:{first.matter_handle}" in rendered
        found_by, _facets = packet_module.span_provenance(receipt)
        # A window names first the matter it opened, then every need it serves.
        assert found_by[first.span_handle][0] == f"M01:MATTER:{first.matter_handle}"
        views = packet_module.matter_views(receipt)
        view = packet_module.matter_view(*views[first.span_handle])
        assert view["document_inspection"] == "COMPLETE" and view["part"] == 1
        summary = packet_module.litigation_matter_summary(receipt)
        assert summary is not None
        assert summary["allowance"]["windows_per_session"] == MAXIMUM_ISSUED_MATTER_WINDOWS
        assert summary["windows"] == {
            "planned": matters.planned_windows,
            "read": matters.read_windows,
            "pending": 0,
            "read_earlier": 0,
        }
        assert summary["next_reads"] == []
        assert "hash" not in json.dumps(summary)
        # A part of the delivery keeps the block and only its own windows.
        part = packet_module.render_evidence_packet(
            packet, span_handles=(first.span_handle,), delivery_part=(1, 3)
        )
        assert '"litigation_matters"' in part and first.span_handle in part
        assert read[1].span_handle not in json.loads(part.split("\n\n", 2)[1])["spans"][0].values()
        # A receipt whose record cites an unread window refuses by name.
        dumped = receipt.model_dump(mode="json", exclude={"receipt_hash"})
        tampered = json.loads(json.dumps(dumped))
        tampered["litigation_matters"]["span_handles"] = tampered["litigation_matters"][
            "span_handles"
        ][1:]
        tampered["litigation_matters"]["windows"][0]["status"] = "PENDING"
        tampered["litigation_matters"]["pending_windows"] = 1
        tampered["litigation_matters"]["read_windows"] -= 1
        with pytest.raises(ValueError, match="litigation_window_accounting_invalid"):
            AlternativeEvidenceRetrievalAccessReceipt.model_validate(
                {**tampered, "receipt_hash": "0" * 64}
            )
        tampered["litigation_matters"]["windows"][0]["span_handle"] = None
        with pytest.raises(ValueError, match="alternative_evidence"):
            AlternativeEvidenceRetrievalAccessReceipt.model_validate(
                {**tampered, "receipt_hash": "0" * 64}
            )
    finally:
        runtime.close()


def test_the_runtime_seals_a_continuation_receipt_over_the_prior_spans(tmp_path: Path) -> None:
    """requirement: a continuation is a successor receipt of the same
    request, document set and generation that carries the program's and the
    typed families' entries unchanged and the continued matter record over
    the prior spans plus the new windows; the prior receipt stays as it was;
    a foreign lineage, a mismatched span set or a PRIOR handle of this very
    session refuses by name."""

    # Matters longer than a window, one window each: more of them than one
    # session's allowance, so the first reading leaves windows pending.
    matters_text = [_long_matter(number) for number in range(1, MAXIMUM_ISSUED_MATTER_WINDOWS + 8)]
    text = _filing(note_lines=[BOILERPLATE, *matters_text])
    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path, documents=(_document(text=text),)
    )
    try:
        receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        first = receipt.litigation_matters
        assert first is not None and first.pending_windows >= 7
        with pytest.raises(ValueError, match="litigation_continuation_spans_mismatch"):
            runtime.continue_evidence(
                request=request,
                document_set=document_set,
                generation=generation,
                prior_receipt=receipt,
                prior_spans=spans[:-1],
                session_limit=5,
                window_limit=320,
            )
        with pytest.raises(ValueError, match="litigation_continuation_lineage_mismatch"):
            runtime.continue_evidence(
                request=request.model_copy(update={"request_hash": "f" * 64}),
                document_set=document_set,
                generation=generation,
                prior_receipt=receipt,
                prior_spans=spans,
                session_limit=5,
                window_limit=320,
            )
        successor, all_spans = runtime.continue_evidence(
            request=request,
            document_set=document_set,
            generation=generation,
            prior_receipt=receipt,
            prior_spans=spans,
            session_limit=5,
            window_limit=320,
        )
        continued = successor.litigation_matters
        assert continued is not None
        assert continued.continued_from == receipt.receipt_hash
        assert continued.session_index == 2 and continued.pending_windows == 0
        assert successor.receipt_hash != receipt.receipt_hash
        assert successor.read_span_handles == receipt.read_span_handles
        assert successor.typed_disclosures == receipt.typed_disclosures
        assert successor.search_call_count == receipt.search_call_count
        assert successor.delivered_span_handles == (
            *receipt.delivered_span_handles,
            *continued.span_handles,
        )
        assert tuple(value.span_handle for value in all_spans) == successor.delivered_span_handles
        assert all_spans[: len(spans)] == spans
        # Both receipts are published; the prior one is unchanged.
        published = runtime.artifacts.load(
            "retrieval-access-receipts",
            receipt.receipt_hash,
            AlternativeEvidenceRetrievalAccessReceipt,
        )
        assert published == receipt
        assert (
            runtime.artifacts.load(
                "retrieval-access-receipts",
                successor.receipt_hash,
                AlternativeEvidenceRetrievalAccessReceipt,
            )
            == successor
        )
        # The receipt validator refuses a PRIOR window of this session's own
        # series, and a PRIOR handle on a first reading.
        dumped = successor.model_dump(mode="json", exclude={"receipt_hash"})
        tampered = json.loads(json.dumps(dumped))
        tampered["litigation_matters"]["windows"][0]["span_handle"] = "SPAN-M02-R0999"
        with pytest.raises(ValueError, match="litigation_prior_window_unproved"):
            AlternativeEvidenceRetrievalAccessReceipt.model_validate(
                {**tampered, "receipt_hash": "0" * 64}
            )
        # A receipt sealed before plans carried an identity (no plan_hash, no
        # chain fields) still reads under its own hash; such a record cannot
        # be continued, and a chain record without its identity refuses.
        sealed = json.loads(receipt.model_dump_json(exclude={"receipt_hash"}))
        for key in ("plan_hash", "cumulative_read_windows"):
            sealed["litigation_matters"].pop(key)
        for document in sealed["litigation_matters"]["documents"]:
            document.pop("qualifications", None)
        retained = seal_contract(
            AlternativeEvidenceRetrievalAccessReceipt, "receipt_hash", **sealed
        )
        assert retained.litigation_matters.plan_hash is None
        assert retained.litigation_matters.session_index == 1
        # The absent fields stay absent from the identity: the hash a retained
        # receipt was sealed with is the hash it reads under.
        assert json.loads(retained.model_dump_json(exclude={"receipt_hash"})) == sealed
        with pytest.raises(ValueError, match="litigation_plan_changed"):
            runtime.continue_evidence(
                request=request,
                document_set=document_set,
                generation=generation,
                prior_receipt=retained,
                prior_spans=spans,
                session_limit=5,
                window_limit=320,
            )
        chain = json.loads(successor.model_dump_json())
        chain["litigation_matters"].pop("plan_hash")
        with pytest.raises(ValueError, match="litigation_continuation_chain_invalid"):
            AlternativeEvidenceRetrievalAccessReceipt.model_validate(
                {**chain, "receipt_hash": "0" * 64}
            )
        dumped = receipt.model_dump(mode="json", exclude={"receipt_hash"})
        tampered = json.loads(json.dumps(dumped))
        window = tampered["litigation_matters"]["windows"][-1]
        assert window["status"] == "PENDING"
        window["status"] = "PRIOR"
        window["span_handle"] = "SPAN-M00-R0001"
        tampered["litigation_matters"]["pending_windows"] -= 1
        tampered["litigation_matters"]["prior_windows"] += 1
        tampered["litigation_matters"]["plan_offset"] += 1
        tampered["litigation_matters"]["cumulative_read_windows"] += 1
        with pytest.raises(ValueError, match="litigation_prior_window_unproved"):
            AlternativeEvidenceRetrievalAccessReceipt.model_validate(
                {**tampered, "receipt_hash": "0" * 64}
            )
    finally:
        runtime.close()
