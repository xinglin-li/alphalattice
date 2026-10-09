"""Source continuation is a product operation of the one application owner:
a prepared packet with pending matter windows names, in its delivery, the
receipt and span set it sealed and an executable continuation request; the
application admits one more bounded reading session as its own Task under
the packet's request, obligation and admission; the successor packet keeps
every earlier span under its own handle and adds the new session's under a
new series; the same request asked twice is the completed Task; a stale or
forged prior, an exhausted or changed cumulative allowance, a tampered
source and an interruption before the commit boundary are refused, resumed
or retried by name, never counted as read; the compact summary carries the
whole scope in every part while the full inventory is an addressable
bounded detail; CLI and Web are one owner.
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.evidence.alternative_evidence.analysis.packet import (
    DEFAULT_DELIVERY_BUDGET_BYTES,
    serialized_response_bytes,
)
from alphalattice.evidence.alternative_evidence.retrieval.session import (
    MAXIMUM_ISSUED_MATTER_WINDOWS,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    AlternativeEvidenceDocumentTaskAdapter,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)
from tests.alternative_evidence_desk.document_intelligence_support import _document_with_text
from tests.alternative_evidence_desk.litigation_support import BOILERPLATE, _filing
from tests.alternative_evidence_desk.matter_selection_support import _long_matter
from tests.alternative_evidence_desk.review_http_support import (
    _continue,
    _operation,
    _parts,
    _raise_interruption,
    _receipt,
    _Service,
    _summary,
    build_authority,
    build_workspace,
    delivered_aliases,
    start_service,
)
from tests.portfolio_strategy_lab.local_web_support import _request

MATTER_COUNT = 2 * MAXIMUM_ISSUED_MATTER_WINDOWS + 6
"""A hundred and thirty-four named matters in one note, each longer than a
window so each is one: an initial session reads its sixty-four window
allowance, a continuation reads sixty-four more and a third session finishes
the plan -- a chain of three."""


ONE_UNIT = "u01"
"""The book's only unit: every book is prepared as a coverage run."""


def _ten_k(entities: tuple[str, ...]) -> tuple[RecordedEvidenceDocument, ...]:
    """One 10-K for the book's first issuer whose contingencies note holds
    more matters than one session's allowance."""

    text = _filing(note_lines=[BOILERPLATE, *(_long_matter(n) for n in range(1, MATTER_COUNT + 1))])
    return (_document_with_text(entities[0], text=text, form="10-K", revision="10-k-2026"),)


def _wire(service: _Service, path: str) -> tuple[int, dict[str, Any]]:
    status, _headers, raw = _request(service.session, path)
    assert status == 200, raw[:300]
    return len(raw), json.loads(raw)


def _accounting(block: dict[str, Any]) -> dict[str, Any]:
    """The matter block without the part's own `matters_in_part`."""

    return {k: v for k, v in block.items() if k != "matters_in_part"}


def test_continuation_is_an_executable_product_operation_over_pending_scope(
    tmp_path: Path,
) -> None:
    """requirement: pagination and source continuation are two requests; a
    continuation reads only what the sealed prior proved unread, under a
    new handle series, and the same request twice is one Task."""

    workspace, report = build_workspace(tmp_path)
    authority = build_authority(tmp_path=tmp_path, report=report, extra_documents=_ten_k)
    service = start_service(workspace, authority, tmp_path)
    try:
        selected = {"result_hash": service.result_hash()}
        prepared = service.post("/api/evidence/prepare", selected)
        assert prepared["disposition"] == "ADMITTED", prepared
        service.drain()
        task_id = prepared["task_id"]
        adapter = service.review.evidence_task_adapter
        writes_after_preparation = adapter.runtime.artifacts.write_count

        # The fixture's first packet sits at the default budget's edge; a
        # declared budget reads it whole (pagination is proved elsewhere).
        whole = _operation(
            service,
            {
                "operation": "EVIDENCE_PACKET",
                **selected,
                "task_id": task_id,
                "evidence_unit_id": ONE_UNIT,
                "delivery_budget_bytes": 2 * DEFAULT_DELIVERY_BUDGET_BYTES,
            },
        )
        assert whole["status"] == "EVIDENCE_ANALYST_PACKET_READY"
        summary = _summary(whole)
        assert summary["windows"]["read"] == MAXIMUM_ISSUED_MATTER_WINDOWS
        pending = summary["windows"]["pending"]
        assert pending > MAXIMUM_ISSUED_MATTER_WINDOWS and summary["windows"]["read_earlier"] == 0
        assert summary["continuation"]["state"] == "PENDING"
        assert summary["continuation"]["next_session"]["windows"] == MAXIMUM_ISSUED_MATTER_WINDOWS
        assert summary["reading_chain"]["session_index"] == 1
        assert summary["reading_chain"]["session_limit"] is None
        # Compact: counts and state per filing, no per-matter list, no hash.
        (filing,) = [d for d in summary["documents"] if d["document_type"] == "10-K"]
        assert filing["inspection"] == "PARTIAL" and filing["matters_inventoried"] >= MATTER_COUNT
        assert "matters" not in filing and filing["regions"] >= 1
        assert "hash" not in json.dumps(summary)
        # The two continuations are two requests: pagination reads no source.
        assert whole["delivery"]["next_request"] is None
        request = whole["continuation_request"]
        assert request["operation"] == "EVIDENCE_CONTINUE"
        assert request["task_id"] == task_id and request["evidence_unit_id"] == ONE_UNIT
        assert request["session_limit"] is None and request["window_limit"] is None
        # The request holds the operation's parameters and nothing else; the
        # scope, the cost and what to declare sit beside it.
        assert set(request) == {
            "operation",
            "result_hash",
            "task_id",
            "evidence_unit_id",
            "continuation_of",
            "continuation_spans",
            "session_limit",
            "window_limit",
        }
        scope = whole["continuation_scope"]
        assert scope["state"] == "PENDING" and scope["pending_windows"] == pending
        assert scope["pending_source_bytes"] == summary["source_bytes"]["pending"]
        assert scope["next_session"] == summary["continuation"]["next_session"]
        assert scope["declare"] == ["session_limit", "window_limit"]
        # Unfilled, the first continuation is refused by name at the typed
        # boundary: a first reading has no cumulative allowance to inherit.
        with pytest.raises(ValidationError, match="operation_field_required:session_limit"):
            PortfolioResearchRequestDocument.model_validate(request)
        receipt_hash, span_set_hash = adapter.prepared_receipt_identity(
            UUID(task_id), unit_id=ONE_UNIT
        )
        assert request["continuation_of"] == receipt_hash
        assert request["continuation_spans"] == span_set_hash
        first_receipt = _receipt(service, receipt_hash)
        m01 = list(first_receipt.litigation_matters.span_handles)
        assert m01 and all(handle.startswith("SPAN-M01-") for handle in m01)

        # A forged prior: a receipt this packet did not seal is refused by
        # name and admits nothing -- the only positional claim a continuation
        # can make is the receipt it names.
        forged = _continue(
            service,
            {**request, "continuation_of": "f" * 64},
            session_limit=5,
            window_limit=320,
        )
        assert forged["disposition"] == "REFUSED_CONTINUATION_PRIOR_MISMATCH"
        assert "task_id" not in forged
        assert adapter.runtime.artifacts.write_count == writes_after_preparation

        # A packet prepared under a retrieval contract this workspace has
        # since superseded: its reading plan is not continued, by name, and
        # nothing is written; the rotation is undone for the chain below.
        current_binding = adapter.runtime.retrieval_binding_hash
        adapter.runtime.retrieval_binding_hash = "1" * 64
        try:
            superseded = _continue(service, request, session_limit=5, window_limit=320)
        finally:
            adapter.runtime.retrieval_binding_hash = current_binding
        assert superseded["disposition"] == "REFUSED_PREPARATION_SUPERSEDED", superseded
        assert superseded["failure_code"] == "alternative_evidence.preparation_superseded"
        assert "task_id" not in superseded
        assert adapter.runtime.artifacts.write_count == writes_after_preparation

        # The continuation: one Task of one stage, its own packet.
        outcome = _continue(service, request, session_limit=5, window_limit=320)
        assert outcome["disposition"] == "ADMITTED", outcome
        continued_id = outcome["task_id"]
        assert continued_id != task_id
        assert outcome["next_requests"]["packet"]["task_id"] == continued_id
        continued_task = service.registry.task(UUID(continued_id))
        assert continued_task.lifecycle is TaskLifecycle.SUCCEEDED
        assert continued_task.input.payload["purpose"] == "CONTINUE_READING"
        assert [item.stage_id for item in continued_task.plan.work_items] == [
            "select_evidence_spans"
        ]
        assert adapter.runtime.retrieval.passage_embedding_pass_count == 1, "no re-embedding"
        parts = _parts(
            service, {"operation": "EVIDENCE_PACKET", **selected, "task_id": continued_id}
        )
        successor = parts[0]
        after = _summary(successor)
        # The compact block travels whole with every part, identically; the
        # matters a part states once for its own windows are the part's.
        assert all(_accounting(_summary(part)) == _accounting(after) for part in parts)
        assert after["windows"] == {
            "planned": MAXIMUM_ISSUED_MATTER_WINDOWS + pending,
            "read": MAXIMUM_ISSUED_MATTER_WINDOWS,
            "pending": pending - MAXIMUM_ISSUED_MATTER_WINDOWS,
            "read_earlier": MAXIMUM_ISSUED_MATTER_WINDOWS,
        }
        assert after["continuation"]["state"] == "PENDING"
        assert after["reading_chain"] == {
            **after["reading_chain"],
            "session_index": 2,
            "continues_earlier_session": True,
            "cumulative_read_windows": 2 * MAXIMUM_ISSUED_MATTER_WINDOWS,
            "session_limit": 5,
            "window_limit": 320,
            "remaining": {"sessions": 3, "windows": 320 - 2 * MAXIMUM_ISSUED_MATTER_WINDOWS},
        }
        assert _summary(whole)["reading_chain"]["remaining"] is None, (
            "a first reading has no allowance of its own to count down"
        )
        (filing_after,) = [d for d in after["documents"] if d["document_type"] == "10-K"]
        assert filing_after["inspection"] == "PARTIAL"
        # Every earlier span keeps its handle; the new session's are M02.
        delivered = [h for part in parts for h in part["delivery"]["delivered_span_handles"]]
        assert (
            delivered[: len(whole["delivery"]["delivered_span_handles"])]
            == (whole["delivery"]["delivered_span_handles"])
        )
        m02 = [handle for handle in delivered if handle.startswith("SPAN-M02-")]
        assert len(m02) == MAXIMUM_ISSUED_MATTER_WINDOWS and set(m01) <= set(delivered)
        # The earlier receipt is what it was; the successor carries the
        # program and typed entries unchanged.
        successor_hash, _spans = adapter.prepared_receipt_identity(UUID(continued_id))
        second_receipt = _receipt(service, successor_hash)
        assert _receipt(service, receipt_hash) == first_receipt
        assert second_receipt.read_span_handles == first_receipt.read_span_handles
        assert second_receipt.typed_disclosures == first_receipt.typed_disclosures
        assert second_receipt.litigation_matters.continued_from == receipt_hash
        # The session's own windows are a bounded delivery of their own: what
        # a consumer holding every earlier part has not yet received.
        own = _parts(service, outcome["next_requests"]["packet_session_windows"])
        own_handles = [h for part in own for h in part["delivery"]["delivered_span_handles"]]
        assert own_handles == m02
        assert all(part["status"] == "EVIDENCE_ANALYST_PACKET_PART" for part in own)
        assert all(part["delivery"]["delivery_scope"] == "SESSION_WINDOWS" for part in own)
        assert all(_accounting(_summary(part)) == _accounting(after) for part in own)
        assert own[0]["analysis_context_hash"] == successor["analysis_context_hash"]

        # The third session continues the continuation: its lineage resolves
        # link by link, its PRIOR windows carry both earlier series, and the
        # plan completes.
        second_request = own[0]["continuation_request"]
        assert second_request["continuation_of"] == successor_hash
        assert (
            second_request["task_id"] == continued_id and "evidence_unit_id" not in second_request
        )
        assert second_request["session_limit"] == 5 and second_request["window_limit"] == 320
        assert own[0]["continuation_scope"]["declare"] == []
        # A chain's request is executable exactly as returned.
        third = _continue(service, second_request)
        assert third["disposition"] == "ADMITTED", third
        third_id = third["task_id"]
        assert service.registry.task(UUID(third_id)).lifecycle is TaskLifecycle.SUCCEEDED
        third_parts = _parts(
            service, {"operation": "EVIDENCE_PACKET", **selected, "task_id": third_id}
        )
        final = _summary(third_parts[0])
        assert third_parts[0]["continuation_request"] is None
        assert third_parts[0]["continuation_scope"]["state"] == "COMPLETE"
        assert third_parts[0]["continuation_scope"]["pending_windows"] == 0
        assert final["windows"] == {
            "planned": MAXIMUM_ISSUED_MATTER_WINDOWS + pending,
            "read": pending - MAXIMUM_ISSUED_MATTER_WINDOWS,
            "pending": 0,
            "read_earlier": 2 * MAXIMUM_ISSUED_MATTER_WINDOWS,
        }
        assert final["continuation"]["state"] == "COMPLETE"
        assert final["continuation"]["remainders"] == {}, "nothing the plan named is left"
        assert final["reading_chain"]["session_index"] == 3
        assert final["reading_chain"]["remaining"] == {
            "sessions": 2,
            "windows": 320 - MAXIMUM_ISSUED_MATTER_WINDOWS - pending,
        }
        assert final["reading_chain"]["cumulative_read_windows"] == (
            MAXIMUM_ISSUED_MATTER_WINDOWS + pending
        )
        (filing_final,) = [d for d in final["documents"] if d["document_type"] == "10-K"]
        assert filing_final["inspection"] == "COMPLETE"
        all_delivered = [
            h for part in third_parts for h in part["delivery"]["delivered_span_handles"]
        ]
        m03 = [handle for handle in all_delivered if handle.startswith("SPAN-M03-")]
        assert len(m03) == pending - MAXIMUM_ISSUED_MATTER_WINDOWS
        # Every earlier span is still delivered under its handle, and the
        # matter windows keep their session order; the integrated selection
        # delivers a session's candidate reads after the matter windows.
        assert set(delivered) <= set(all_delivered)
        windows = [handle for handle in all_delivered if handle.startswith("SPAN-M")]
        assert windows == [handle for handle in delivered if handle.startswith("SPAN-M")] + m03
        third_hash, third_spans = adapter.prepared_receipt_identity(UUID(third_id))
        third_receipt = _receipt(service, third_hash)
        assert third_receipt.litigation_matters.continued_from == successor_hash
        prior = [
            w.span_handle for w in third_receipt.litigation_matters.windows if w.status == "PRIOR"
        ]
        assert prior == m01 + m02
        assert _receipt(service, successor_hash) == second_receipt
        assert third_receipt.read_span_handles == first_receipt.read_span_handles

        # An answer to the successor packet cites windows of two sessions and
        # reads back as one finding package across the continuation boundary;
        # the same answer again is the published Task, not a second one.
        entity = third_receipt.litigation_matters.documents[0].entity_id
        handle = "FIND-001"  # the Host's handle for the answer's first finding
        alias_of = delivered_aliases(third_parts)
        brief = {
            "findings": [
                {
                    "issuer": entity,
                    "topic": "LEGAL_REGULATORY",
                    "lifecycle": "ONGOING",
                    "direction": "ADVERSE",
                    "summary": "Controlled legal-matters finding citing matter windows of "
                    "series M01 and M03; not a judgment.",
                    "cite": [alias_of[value] for value in m01[:4] + m03[:4]],
                }
            ],
            "notes": "Controlled: one finding over windows of two reading sessions.",
        }
        template = third_parts[0]["submission_template"]
        assert template["task_id"] == third_id and "evidence_unit_id" not in template
        submission = {
            k: v for k, v in {**template, "analysis_answer": brief}.items() if k != "operation"
        }
        submitted = service.post("/api/evidence/analysis", submission)
        assert submitted["disposition"] == "ADMITTED", submitted
        service.drain()
        assert service.registry.task(UUID(submitted["task_id"])).lifecycle is (
            TaskLifecycle.SUCCEEDED
        )
        assert service.post("/api/evidence/analysis", submission)["disposition"] == "REUSED_EXACT"
        dossier = _operation(service, {"operation": "CRO_REVIEW_DOSSIER", **selected})
        assert dossier["status"] == "CRO_DOSSIER_READY", dossier.get("status")
        package = _operation(
            service, {"operation": "CRO_REVIEW_FINDING", **selected, "finding_handle": handle}
        )
        assert package["status"] == "FINDING_EVIDENCE_PACKAGE"
        support = package["support"]
        assert [e["span_handle"] for e in support] == m01[:4] + m03[:4]
        assert all(e["source_verified"] and e.get("litigation_matter") for e in support)
        # Every support entry carries its document's time, each meaning named
        # with its basis, and the same view the packet's index carries.
        index = {
            d["document_handle"]: d
            for d in json.loads(third_parts[0]["packet"].split("\n\n", 2)[1])["document_index"]
        }
        for entry in support:
            assert entry["time"] == index[entry["document_handle"]]["time"]
            basis = entry["time"]["availability_basis"]
            assert basis == "RECORDED_IMPORT_CAPTURE_ACCEPTANCE_UNKNOWN"
            assert entry["time"]["published_precision"] == "UNKNOWN"
        web_package = service.get(
            "/api/cro/finding?" + urllib.parse.urlencode({**selected, "finding_handle": handle})
        )
        assert web_package == json.loads(json.dumps(package, default=str))
        first_own = _parts(
            service,
            {
                "operation": "EVIDENCE_PACKET",
                **selected,
                "task_id": task_id,
                "evidence_unit_id": ONE_UNIT,
                "evidence_detail": "session_windows",
            },
        )
        assert [h for p in first_own for h in p["delivery"]["delivered_span_handles"]] == m01
        # Both packets bind their answers to their own excerpts: one alias per
        # span of the successor, whose spans hold both sessions' windows.
        enum = successor["response_schema"]["$defs"]["AlternativeEvidenceAnswerFinding"][
            "properties"
        ]["cite"]["items"]["enum"]
        successor_packet = adapter.prepared_packet(
            UUID(successor["submission_template"]["task_id"]), now=service.review.clock()
        )
        assert len(enum) == len(successor_packet.spans)
        assert set(m01) | set(m02) <= {span.span_handle for span in successor_packet.spans}

        # The same requests again are the completed Tasks, not sessions.
        writes = adapter.runtime.artifacts.write_count
        again = _continue(service, request, session_limit=5, window_limit=320)
        assert again["disposition"] == "REUSED_EXACT" and again["task_id"] == continued_id
        again_third = _continue(service, second_request, session_limit=5, window_limit=320)
        assert again_third["disposition"] == "REUSED_EXACT" and again_third["task_id"] == third_id
        assert adapter.runtime.artifacts.write_count == writes
        # A continuation of the completed packet has nothing pending; an
        # earlier receipt named against a later packet is stale. A
        # continuation's own Task is single: it is addressed without a unit.
        single = {k: v for k, v in request.items() if k != "evidence_unit_id"}
        nothing = _continue(
            service,
            {
                **single,
                "task_id": third_id,
                "continuation_of": third_hash,
                "continuation_spans": third_spans,
            },
            session_limit=5,
            window_limit=320,
        )
        assert nothing["disposition"] == "REFUSED_CONTINUATION_NOTHING_PENDING"
        stale = _continue(
            service, {**single, "task_id": continued_id}, session_limit=5, window_limit=320
        )
        assert stale["disposition"] == "REFUSED_CONTINUATION_PRIOR_MISMATCH"

        web_successor = service.get(
            "/api/evidence/packet?" + urllib.parse.urlencode({**selected, "task_id": continued_id})
        )
        assert web_successor == successor
        assert web_successor["delivery"]["response_bytes"] == serialized_response_bytes(successor)
        # The same request document on the wire, as returned with the limits
        # filled, through the operation's own route: the route is the
        # operation the document names, and never another one; the unfilled
        # first continuation is refused by name there too.
        web_outcome = service.post(
            "/api/evidence/continue", {**request, "session_limit": 5, "window_limit": 320}
        )
        assert web_outcome["disposition"] == "REUSED_EXACT"
        assert web_outcome["task_id"] == continued_id
        status, unfilled = service.request("/api/evidence/continue", method="POST", payload=request)
        assert status == 400
        assert "operation_field_required:session_limit,window_limit" in unfilled["refused"]
        status, mismatched = service.request(
            "/api/evidence/continue",
            method="POST",
            payload={**second_request, "operation": "EVIDENCE_PACKET"},
        )
        assert status == 400
        assert mismatched["refused"] == "local_web.operation_mismatch:EVIDENCE_PACKET"
    finally:
        service.session.stop()


def test_a_cumulative_allowance_is_honest_and_a_continuation_survives_restart_and_tamper(
    tmp_path: Path,
) -> None:
    """requirement: an exhausted cumulative allowance leaves an honest partial
    result and refuses further sessions by name; other limits are another
    request; an interruption before the commit boundary resumes after a
    real restart without a second publication; a tampered source blocks
    the session by name and the retried request reads the restored bytes."""

    workspace, report = build_workspace(tmp_path)
    authority = build_authority(tmp_path=tmp_path, report=report, extra_documents=_ten_k)
    first = start_service(workspace, authority, tmp_path)
    original_verify = AlternativeEvidenceDocumentTaskAdapter.verify_stage
    try:
        selected = {"result_hash": first.result_hash()}
        prepared = first.post("/api/evidence/prepare", selected)
        first.drain()
        task_id = prepared["task_id"]
        whole = _operation(
            first,
            {
                "operation": "EVIDENCE_PACKET",
                **selected,
                "task_id": task_id,
                "evidence_unit_id": ONE_UNIT,
            },
        )
        request = whole["continuation_request"]

        # Two windows of allowance left: the session reads two, reports four
        # pending, and the chain refuses by name under the same limits and
        # as another request under other limits.
        capped = _continue(
            first, request, session_limit=3, window_limit=MAXIMUM_ISSUED_MATTER_WINDOWS + 2
        )
        assert capped["disposition"] == "ADMITTED", capped
        capped_id = capped["task_id"]
        packet = _operation(
            first, {"operation": "EVIDENCE_PACKET", **selected, "task_id": capped_id}
        )
        summary = _summary(packet)
        pending = _summary(whole)["windows"]["pending"]
        assert summary["windows"]["read"] == 2 and summary["windows"]["pending"] == pending - 2
        assert summary["continuation"]["state"] == "PENDING"
        (filing,) = [d for d in summary["documents"] if d["document_type"] == "10-K"]
        assert filing["inspection"] == "PARTIAL"
        follow = packet["continuation_request"]
        assert follow["session_limit"] == 3 and follow["window_limit"] == 66
        exhausted = _continue(first, follow, session_limit=3, window_limit=66)
        assert exhausted["disposition"] == "REFUSED_CONTINUATION_BUDGET_EXHAUSTED"
        assert f"{pending - 2} window(s)" in exhausted["detail"]
        # An exhausted allowance names every channel the plan still owes,
        # never the matter windows alone (the first-release safety closeout).
        assert "sealed candidate(s)" in exhausted["detail"]
        assert "table(s) not dealt remain pending and unread" in exhausted["detail"]
        changed = _continue(first, follow, session_limit=5, window_limit=320)
        assert changed["disposition"] == "REFUSED_CONTINUATION_LIMITS_CHANGED"
        assert "task_id" not in exhausted and "task_id" not in changed

        # An interruption after the session published and before the stage
        # committed: RECOVERY_REQUIRED, resumed by the restarted service.
        def interrupt_once(
            self: Any, *, task: Any, execution: Any, work_item: Any, evidence: Any
        ) -> Any:
            if task.input.payload.get("purpose") == "CONTINUE_READING" and not getattr(
                interrupt_once, "fired", False
            ):
                interrupt_once.fired = True  # type: ignore[attr-defined]
                _raise_interruption()
            return original_verify(
                self, task=task, execution=execution, work_item=work_item, evidence=evidence
            )

        AlternativeEvidenceDocumentTaskAdapter.verify_stage = interrupt_once  # type: ignore[method-assign]
        interrupted = _continue(first, request, session_limit=5, window_limit=320)
        assert interrupted["disposition"] == "ADMITTED"
        interrupted_id = UUID(interrupted["task_id"])
        assert first.registry.task(interrupted_id).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        published = sorted(
            (
                first.review.evidence_task_adapter.runtime.artifacts.root
                / "retrieval-access-receipts"
            ).glob("*.json")
        )
    finally:
        AlternativeEvidenceDocumentTaskAdapter.verify_stage = original_verify  # type: ignore[method-assign]
        first.session.stop()

    second = start_service(workspace, authority, tmp_path)
    try:
        assert second.session.resumed_task_ids == (interrupted_id,)
        second.drain()
        assert second.registry.task(interrupted_id).lifecycle is TaskLifecycle.SUCCEEDED
        adapter = second.review.evidence_task_adapter
        assert (
            sorted((adapter.runtime.artifacts.root / "retrieval-access-receipts").glob("*.json"))
            == published
        ), "the resumed session republished nothing new"
        packet = _operation(
            second, {"operation": "EVIDENCE_PACKET", **selected, "task_id": str(interrupted_id)}
        )
        resumed_summary = _summary(packet)
        assert resumed_summary["reading_chain"]["session_index"] == 2
        assert resumed_summary["windows"]["read"] == MAXIMUM_ISSUED_MATTER_WINDOWS
        assert resumed_summary["windows"]["pending"] == pending - MAXIMUM_ISSUED_MATTER_WINDOWS
        again = _continue(second, request, session_limit=5, window_limit=320)
        assert again["disposition"] == "REUSED_EXACT"
        assert again["task_id"] == str(interrupted_id)

        # Tamper: a changed source byte blocks a fresh session by name and
        # publishes nothing; the restored bytes read again under a new Task
        # of the same input.
        document_set = adapter._document_set(second.registry.task(UUID(task_id)), ONE_UNIT)
        ten_k = next(d for d in document_set.documents if d.document_type == "10-K")
        blob = next(
            path
            for path in (tmp_path / "evidence" / "workspace" / "knowledge" / "blobs").rglob("*")
            if path.is_file() and path.stat().st_size == ten_k.byte_count
        )
        original = blob.read_bytes()
        writes = adapter.runtime.artifacts.write_count
        blob.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
        try:
            blocked = _continue(second, request, session_limit=2, window_limit=320)
        finally:
            blob.write_bytes(original)
        assert blocked["disposition"] == "ADMITTED"
        blocked_task = second.registry.task(UUID(blocked["task_id"]))
        assert blocked_task.lifecycle is TaskLifecycle.BLOCKED
        assert adapter.runtime.artifacts.write_count == writes
        retried = _continue(second, request, session_limit=2, window_limit=320)
        assert retried["disposition"] == "ADMITTED" and retried["task_id"] != blocked["task_id"]
        assert second.registry.task(UUID(retried["task_id"])).lifecycle is TaskLifecycle.SUCCEEDED
        packet = _operation(
            second, {"operation": "EVIDENCE_PACKET", **selected, "task_id": retried["task_id"]}
        )
        assert _summary(packet)["windows"]["read"] == min(pending, MAXIMUM_ISSUED_MATTER_WINDOWS)
    finally:
        second.session.stop()


@pytest.mark.parametrize(
    ("case", "session_limit", "window_limit", "bounded_review"),
    (
        ("undeclared", None, None, True),
        ("pending", 5, 320, False),
        ("unexhausted", 5, 320, False),
        ("windows_exhausted", 3, MAXIMUM_ISSUED_MATTER_WINDOWS + 2, True),
        ("sessions_exhausted", 2, 320, True),
    ),
)
def test_next_actions_settle_resumable_evidence_before_cro_and_keep_budget_stops_bounded(
    tmp_path: Path,
    case: str,
    session_limit: int | None,
    window_limit: int | None,
    bounded_review: bool,
) -> None:
    """behaviour: a declared reading with work still owed keeps its exact continuation door;
    an undeclared or exhausted allowance permits bounded CRO review with unread scope named."""

    workspace, report = build_workspace(tmp_path)
    authority = build_authority(tmp_path=tmp_path, report=report, extra_documents=_ten_k)
    service = start_service(workspace, authority, tmp_path)
    try:
        selected = {"result_hash": service.result_hash()}
        prepared = service.post("/api/evidence/prepare", selected)
        assert prepared["disposition"] == "ADMITTED", prepared
        service.drain()
        packet_request = {
            "operation": "EVIDENCE_PACKET",
            **selected,
            "task_id": prepared["task_id"],
            "evidence_unit_id": ONE_UNIT,
        }
        parts = _parts(
            service,
            {**packet_request, "delivery_budget_bytes": 2 * DEFAULT_DELIVERY_BUDGET_BYTES},
        )
        if case == "pending":
            remaining = _summary(parts[0])["reading_chain"]["remaining"]
            assert remaining is None
        if case != "undeclared":
            assert session_limit is not None and window_limit is not None
            continued = _continue(
                service,
                parts[0]["continuation_request"],
                session_limit=session_limit,
                window_limit=window_limit,
            )
            assert continued["disposition"] == "ADMITTED", continued
            packet_request = continued["next_requests"]["packet"]
            assert packet_request["task_id"] == continued["task_id"]
            assert "evidence_unit_id" not in packet_request
            parts = _parts(
                service,
                {**packet_request, "delivery_budget_bytes": 2 * DEFAULT_DELIVERY_BUDGET_BYTES},
            )

        summary = _summary(parts[0])
        assert summary["continuation"]["state"] == "PENDING"
        assert summary["windows"]["pending"] > 0
        remaining = summary["reading_chain"]["remaining"]
        if case == "undeclared":
            assert remaining is None
        elif case in {"pending", "unexhausted"}:
            assert remaining["sessions"] > 0 and remaining["windows"] > 0
        elif case == "windows_exhausted":
            assert remaining["windows"] == 0 and remaining["sessions"] > 0
        else:
            assert remaining["sessions"] == 0 and remaining["windows"] > 0

        # Only this packet's analysis is published: another reading of the same
        # obligation would require a real Evidence selection, not an inferred latest one.
        submitted = service.post(
            "/api/evidence/analysis",
            {
                key: value
                for key, value in {
                    **parts[0]["submission_template"],
                    "analysis_answer": {
                        "findings": [],
                        "notes": "Controlled whole packet read; its pending source remains unread.",
                    },
                }.items()
                if key != "operation"
            },
        )
        assert submitted["disposition"] == "ADMITTED", submitted
        service.drain()
        assert service.registry.task(UUID(submitted["task_id"])).lifecycle is (
            TaskLifecycle.SUCCEEDED
        )
        task_count = len(service.registry.tasks())
        section = _operation(service, {"operation": "EVIDENCE_CRO", **selected})
        if bounded_review:
            assert section["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW", section
            assert "unread" in section["explanation"].casefold()
            assert [action["action"] for action in section["required_actions"]] == [
                "READ_CRO_DOSSIER",
                "SUBMIT_CRO_ASSESSMENT",
            ]
            assert section["next_requests"]["dossier"] == {
                "operation": "CRO_REVIEW_DOSSIER",
                **selected,
            }
            assert set(section["next_requests"]) == {"dossier", "refresh", "review"}
            assert section["next_requests"]["refresh"] == {
                "operation": "EVIDENCE_REFRESH",
                **selected,
            }
            assert section["next_requests"]["review"] == {
                "operation": "CRO_REVIEW",
                **selected,
            }
            assert "cro_bundle" not in section["next_requests"]
            dossier = _operation(service, section["next_requests"]["dossier"])
            assert dossier["status"] == "CRO_DOSSIER_READY", dossier
            assert dossier["next_requests"]["cro_bundle"] == {
                "operation": "AGENT_BUNDLE_PREPARE",
                "agent_role": "CRO",
                **selected,
            }
            directory = str(tmp_path / "cro-bundle")
            bundle = _operation(
                service,
                {**dossier["next_requests"]["cro_bundle"], "bundle_directory": directory},
            )
            assert bundle["status"] == "AGENT_BUNDLE_READY", bundle
            assert bundle["agent_role"] == "CRO"
            assert bundle["bundle_directory"] == directory
            assert bundle["next_requests"]["submit"] == {
                "operation": "AGENT_ANSWER_SUBMIT",
                "bundle_directory": directory,
                "agent_answer": None,
            }
        else:
            assert section["state"] == "AWAITING_ALTERNATIVE_EVIDENCE", section
            assert [action["action"] for action in section["required_actions"]] == [
                "SETTLE_EVIDENCE_CONTINUATION",
                "ANALYZE_CONTINUED_EVIDENCE",
            ]
            packets = [
                request
                for request in section["next_requests"].values()
                if request["operation"] == "EVIDENCE_PACKET"
            ]
            assert packets == [packet_request]
            assert "REVIEW_WITH_CRO" not in section["available_actions"]
            assert not any(
                request["operation"].startswith("CRO_")
                or (
                    request["operation"] == "AGENT_BUNDLE_PREPARE"
                    and request.get("agent_role") == "CRO"
                )
                for request in section["next_requests"].values()
            )
            reopened = _operation(service, packets[0])
            assert _summary(reopened)["reading_chain"] == summary["reading_chain"]
        assert len(service.registry.tasks()) == task_count, "the returned reads admit no work"
    finally:
        service.session.stop()
