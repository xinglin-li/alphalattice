"""The budget bounds the response that is actually returned, and a delivery
that does not fit is followed to completion through executable parts.

On the frozen base the packet operation always returned the whole rendered
packet whatever the budget, the budget was measured on the packet text alone
(not on the schema, metadata and envelope the response carries), and the
"continuation" was prose with no request behind it. Here: the whole response
is delivered when it fits, otherwise complete spans in parts, each part
measured as the response it is (the local Web route's own serialization plus
a transport allowance), each carrying the whole packet's schema and
submission template and an executable request for the next part; stale,
out-of-range and mis-scoped continuations are refused by name; a span that
does not fit on its own is a typed outcome, never a cut; the CRO dossier is
delivered through the same mechanism.
"""

from __future__ import annotations

import json
import urllib.parse
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from alphalattice.control.product_host.composition import (
    evidence_review_delivery as delivery_module,
)
from alphalattice.evidence.alternative_evidence.analysis.packet import (
    DELIVERY_ENVELOPE_ALLOWANCE_BYTES,
    MINIMUM_DELIVERY_BUDGET_BYTES,
    serialized_response_bytes,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from tests.alternative_evidence_desk.planted_corpus import _NOW, _CitingActor
from tests.alternative_evidence_desk.review_dossiers import (
    _issue,
    _submission,
    answer_from_issues,
)
from tests.alternative_evidence_desk.review_http_support import (
    _Service,
    build_authority,
    build_workspace,
    start_service,
)
from tests.portfolio_strategy_lab.local_web_support import _request

LONG_MULTILINGUAL = (
    " Zusätzliche Erläuterung des Vorstands zur Liquiditätslage und zu den Lieferketten; "
    "管理层就流动性、监管调查与供应链中断作出补充说明并指出相关风险仍在评估之中。 "
    "Complément d'information sur les contentieux réglementaires et les garanties données. "
) * 12
"""Roughly 3 KB of accented and CJK text per claim: escaped six-fold on the
wire, so the response is measured where the bytes actually are."""


ONE_UNIT = "u01"
"""The small book's only unit: every book is prepared as a coverage run."""


def _prepared(tmp_path: Path, *, clock: Any = None) -> tuple[_Service, dict[str, str], str]:
    """A real service over the small book, the packet prepared natively."""

    workspace, report = build_workspace(tmp_path)
    authority = build_authority(tmp_path=tmp_path, report=report, document_suffix=LONG_MULTILINGUAL)
    service = start_service(workspace, authority, tmp_path, clock=clock)
    selected = {"result_hash": service.result_hash()}
    prepared = service.post("/api/evidence/prepare", selected)
    assert prepared["disposition"] == "ADMITTED", prepared
    service.drain()
    return service, selected, prepared["task_id"]


def _operation(service: _Service, document: dict[str, Any]) -> dict[str, Any]:
    """The CLI's path: the one operation owner, in process."""

    assert service.session.operations is not None
    return service.session.operations.execute(
        PortfolioResearchOperationRequest(**document), caller="HUMAN"
    )


def _wire_bytes(service: _Service, path: str) -> tuple[int, dict[str, Any]]:
    """The bytes the Web route puts on the wire for a GET, and the parsed body."""

    import json

    status, _headers, raw = _request(service.session, path)
    assert status == 200, raw[:300]
    return len(raw), json.loads(raw)


def test_the_whole_response_is_delivered_when_it_fits_and_measured_as_sent(
    tmp_path: Path,
) -> None:
    """requirement: the normal small path stays one response; its recorded
    bytes are the bytes the route serializes, envelope allowance named."""

    service, selected, task_id = _prepared(tmp_path)
    try:
        path = "/api/evidence/packet?" + urllib.parse.urlencode(
            {**selected, "task_id": task_id, "evidence_unit_id": ONE_UNIT}
        )
        wire, body = _wire_bytes(service, path)
        assert body["status"] == "EVIDENCE_ANALYST_PACKET_READY"
        delivery = body["delivery"]
        assert delivery["delivery_mode"] == "WHOLE_PACKET"
        assert (delivery["part"], delivery["part_count"]) == (1, 1)
        assert delivery["next_request"] is None and delivery["remaining_span_count"] == 0
        assert delivery["budget_source"] == "HOST_CONSERVATIVE_DEFAULT"
        assert delivery["envelope_allowance_bytes"] == DELIVERY_ENVELOPE_ALLOWANCE_BYTES
        assert delivery["response_bytes"] == wire == serialized_response_bytes(body)
        assert (
            delivery["response_bytes"] + DELIVERY_ENVELOPE_ALLOWANCE_BYTES
            <= (delivery["budget_bytes"])
        )
        aliases = body["response_schema"]["$defs"]["AlternativeEvidenceAnswerFinding"][
            "properties"
        ]["cite"]["items"]["enum"]

        rows = json.loads(body["packet"].split("\n\n", 2)[1])["spans"]
        # The answer cites by alias: one per delivered span, every one named.
        assert [row["span_handle"] for row in rows] == delivery["delivered_span_handles"]
        assert sorted(row["alias"] for row in rows) == sorted(aliases)
        assert body == _operation(
            service,
            {
                "operation": "EVIDENCE_PACKET",
                **selected,
                "task_id": task_id,
                "evidence_unit_id": ONE_UNIT,
            },
        )
    finally:
        service.session.stop()


def test_an_oversize_packet_is_delivered_in_complete_parts_through_executable_continuations(
    tmp_path: Path,
) -> None:
    """An oversize packet is delivered in complete parts through executable continuations."""

    service, selected, task_id = _prepared(tmp_path)
    try:
        whole = _operation(
            service,
            {
                "operation": "EVIDENCE_PACKET",
                **selected,
                "task_id": task_id,
                "evidence_unit_id": ONE_UNIT,
            },
        )
        all_handles = whole["delivery"]["delivered_span_handles"]
        budget = max(MINIMUM_DELIVERY_BUDGET_BYTES, whole["delivery"]["response_bytes"] // 2)
        first = _operation(
            service,
            {
                "operation": "EVIDENCE_PACKET",
                **selected,
                "task_id": task_id,
                "evidence_unit_id": ONE_UNIT,
                "delivery_budget_bytes": budget,
            },
        )
        assert first["status"] == "EVIDENCE_ANALYST_PACKET_PART", first["status"]
        assert first["delivery"]["delivery_mode"] == "PARTS"
        assert first["delivery"]["budget_source"] == "CONSUMER_DECLARED"
        assert first["delivery"]["part_count"] > 1

        # Follow the continuations through the operations owner (the CLI's path).
        parts = [first]
        while parts[-1]["delivery"]["next_request"] is not None:
            request = parts[-1]["delivery"]["next_request"]
            assert request["operation"] == "EVIDENCE_PACKET"
            assert request["delivery_part"] == parts[-1]["delivery"]["part"] + 1
            assert request["delivery_budget_bytes"] == budget
            assert request["analysis_context_hash"] == first["analysis_context_hash"]
            parts.append(_operation(service, request))
        assert len(parts) == first["delivery"]["part_count"]
        delivered = [h for part in parts for h in part["delivery"]["delivered_span_handles"]]
        assert delivered == all_handles, "every span once, in packet order"
        for index, part in enumerate(parts, start=1):
            delivery = part["delivery"]
            assert (delivery["part"], delivery["part_count"]) == (index, len(parts))
            assert delivery["response_bytes"] == serialized_response_bytes(part)
            assert delivery["response_bytes"] + DELIVERY_ENVELOPE_ALLOWANCE_BYTES <= budget
            assert (
                delivery["remaining_span_handles"]
                == all_handles[
                    len(
                        [h for p in parts[:index] for h in p["delivery"]["delivered_span_handles"]]
                    ) :
                ]
            )
            # Each part renders exactly its spans; the envelope's delivery names
            # the part it is (the packet text instructs no reader).
            text = part["packet"]
            assert all(f'"span_handle": "{h}"' in text for h in delivery["delivered_span_handles"])
            assert not any(
                f'"span_handle": "{h}"' in text.split('"spans"')[-1]
                for h in delivery["remaining_span_handles"]
            )
            assert '"delivery_part"' not in text
            # Every part carries the whole packet's schema and the same template.
            assert part["response_schema"] == whole["response_schema"]
            assert part["submission_template"] == whole["submission_template"]
            assert part["analysis_context_hash"] == whole["analysis_context_hash"]
        assert first["delivery"]["delivered_entity_ids"] or len(parts) > 1

        # The Web route delivers the same parts, measured as it sends them.
        for index, part in enumerate(parts, start=1):
            path = "/api/evidence/packet?" + urllib.parse.urlencode(
                {
                    **selected,
                    "task_id": task_id,
                    "evidence_unit_id": ONE_UNIT,
                    "delivery_part": index,
                    "delivery_budget_bytes": budget,
                    "analysis_context_hash": first["analysis_context_hash"],
                }
            )
            wire, body = _wire_bytes(service, path)
            assert body == part and wire == part["delivery"]["response_bytes"]

        # Stale, out-of-range, mis-scoped and unbounded continuations are refused.
        stale = {**parts[-1]["delivery"], "next_request": None}
        del stale
        for document, code in (
            (
                {**first["delivery"]["next_request"], "analysis_context_hash": "0" * 64},
                "alternative_evidence.delivery_continuation_stale",
            ),
            (
                {**first["delivery"]["next_request"], "delivery_part": len(parts) + 1},
                "alternative_evidence.delivery_part_unknown",
            ),
            (
                {
                    "operation": "EVIDENCE_PACKET",
                    **selected,
                    "task_id": task_id,
                    "evidence_unit_id": ONE_UNIT,
                    "delivery_part": 2,
                },
                "alternative_evidence.delivery_part_unknown",
            ),
            (
                {
                    "operation": "EVIDENCE_PACKET",
                    **selected,
                    "task_id": task_id,
                    "evidence_unit_id": ONE_UNIT,
                    "delivery_budget_bytes": MINIMUM_DELIVERY_BUDGET_BYTES - 1,
                },
                "alternative_evidence.delivery_budget_invalid",
            ),
        ):
            with pytest.raises(ValueError, match=code):
                _operation(service, document)
            status, _headers, raw = _request(
                service.session,
                "/api/evidence/packet?"
                + urllib.parse.urlencode({k: v for k, v in document.items() if k != "operation"}),
            )
            assert status == 400 and code in raw.decode("utf-8")
        with pytest.raises(ValueError, match="delivery_selector_invalid"):
            _operation(service, {**first["delivery"]["next_request"], "delivery_part": 0})

        # The answer, authored after every part, binds to the whole packet.
        packet = service.review.evidence_task_adapter.prepared_packet(
            UUID(task_id), now=service.review.clock(), unit_id=ONE_UNIT
        )
        answer = _CitingActor(topics={})(packet=packet).answer.model_dump(mode="json")
        submitted = service.post(
            "/api/evidence/analysis",
            {
                k: v
                for k, v in {**first["submission_template"], "analysis_answer": answer}.items()
                if k != "operation"
            },
        )
        assert submitted["disposition"] == "ADMITTED", submitted
        service.drain()
        assert service.evidence_cro()["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
    finally:
        service.session.stop()


def test_a_span_that_cannot_fit_on_its_own_is_a_typed_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: an indivisible unit over the budget is refused by name with
    its bytes, never truncated under its handle, never re-offered forever."""

    service, selected, task_id = _prepared(tmp_path)
    try:
        whole = _operation(
            service,
            {
                "operation": "EVIDENCE_PACKET",
                **selected,
                "task_id": task_id,
                "evidence_unit_id": ONE_UNIT,
            },
        )
        budget = max(MINIMUM_DELIVERY_BUDGET_BYTES, whole["delivery"]["response_bytes"] // 2)
        # The transport allowance grown past the budget: no span fits on its own.
        monkeypatch.setattr(delivery_module, "DELIVERY_ENVELOPE_ALLOWANCE_BYTES", budget)
        first_handle = whole["delivery"]["delivered_span_handles"][0]
        with pytest.raises(ValueError, match=f"delivery_budget_below_minimum_unit:{first_handle}:"):
            _operation(
                service,
                {
                    "operation": "EVIDENCE_PACKET",
                    **selected,
                    "task_id": task_id,
                    "evidence_unit_id": ONE_UNIT,
                    "delivery_budget_bytes": budget,
                },
            )
    finally:
        service.session.stop()


def test_the_dossier_is_delivered_through_the_same_mechanism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oversized dossier is delivered as complete bounded parts with executable continuations
    and intact citations."""

    from alphalattice.evidence.alternative_evidence.analysis import packet as packet_module

    monkeypatch.setattr(packet_module, "MINIMUM_DELIVERY_BUDGET_BYTES", 8 * 1024)
    observed = [_NOW]
    service, selected, task_id = _prepared(tmp_path, clock=lambda: observed[0])
    try:
        packet = service.review.evidence_task_adapter.prepared_packet(
            UUID(task_id), now=service.review.clock(), unit_id=ONE_UNIT
        )
        answer = _CitingActor(topics={})(packet=packet).answer.model_dump(mode="json")
        exported = _operation(
            service,
            {
                "operation": "EVIDENCE_PACKET",
                **selected,
                "task_id": task_id,
                "evidence_unit_id": ONE_UNIT,
            },
        )
        submitted = service.post(
            "/api/evidence/analysis",
            {
                k: v
                for k, v in {**exported["submission_template"], "analysis_answer": answer}.items()
                if k != "operation"
            },
        )
        assert submitted["disposition"] == "ADMITTED", submitted
        service.drain()
        whole = _operation(service, {"operation": "CRO_REVIEW_DOSSIER", **selected})
        assert whole["status"] == "CRO_DOSSIER_READY"
        assert whole["delivery"]["delivery_mode"] == "WHOLE_DOSSIER"
        assert whole["delivery"]["response_bytes"] == serialized_response_bytes(whole)
        # follow-up: an unchanged dossier's template is bound to the same
        # resolved cutoff, never to this GET's instant; compare the entire answer.
        observed[0] += timedelta(seconds=2, microseconds=137)
        assert _operation(service, {"operation": "CRO_REVIEW_DOSSIER", **selected}) == whole
        findings = [value["finding_handle"] for value in whole["dossier"]["findings"]]
        citations = [value["span_handle"] for value in whole["dossier"]["citations"]]
        assert len(findings) >= 2, "the small book yields one finding per issuer"
        # Two kilobytes short of the whole: the envelope and most findings fit
        # a part; the rest follow.
        budget = whole["delivery"]["response_bytes"] + DELIVERY_ENVELOPE_ALLOWANCE_BYTES - 2048
        parts = [
            _operation(
                service,
                {"operation": "CRO_REVIEW_DOSSIER", **selected, "delivery_budget_bytes": budget},
            )
        ]
        assert parts[0]["status"] == "CRO_DOSSIER_PART"
        while parts[-1]["delivery"]["next_request"] is not None:
            observed[0] += timedelta(seconds=2, microseconds=137)
            request = parts[-1]["delivery"]["next_request"]
            assert request["operation"] == "CRO_REVIEW_DOSSIER"
            assert request["review_dossier_hash"] == whole["dossier"]["dossier_hash"]
            parts.append(_operation(service, request))
        assert len(parts) == parts[0]["delivery"]["part_count"] > 1
        delivered_findings = [
            v["finding_handle"] for part in parts for v in part["dossier"]["findings"]
        ]
        delivered_citations = [
            v["span_handle"] for part in parts for v in part["dossier"]["citations"]
        ]
        assert delivered_findings == findings
        assert sorted(delivered_citations) == sorted(citations), "every citation exactly once"
        # Each issuer's reported checks travel in exactly one part, after the
        # findings, so no part grows with the book (first-release A4).
        delivered_checks = [entity for part in parts for entity in part["issuer_checks"]]
        assert sorted(delivered_checks) == sorted(whole["issuer_checks"])
        assert {
            entity: checks for part in parts for entity, checks in part["issuer_checks"].items()
        } == whole["issuer_checks"]
        assert parts[-1]["delivery"]["remaining_issuer_check_count"] == 0
        for index, part in enumerate(parts, start=1):
            assert part["delivery"]["response_bytes"] + DELIVERY_ENVELOPE_ALLOWANCE_BYTES <= budget
            assert part["delivery"]["response_bytes"] == serialized_response_bytes(part)
            assert part["assessment_schema"] == whole["assessment_schema"]
            assert part["submission_template"] == whole["submission_template"]
            assert part["dossier"]["dossier_hash"] == whole["dossier"]["dossier_hash"]
            assert part["dossier"]["dossier_partial"]["part"] == index
            assert part["dossier"]["dossier_partial"]["issuer_check_entities"] == list(
                part["issuer_checks"]
            )
            assert part["evidence_sources"] == whole["evidence_sources"]
            assert part["dossier"]["issuers"] == whole["dossier"]["issuers"]
        with pytest.raises(ValueError, match=r"chief_risk_officer\.delivery_continuation_stale"):
            _operation(
                service,
                {**parts[0]["delivery"]["next_request"], "review_dossier_hash": "0" * 64},
            )
        with pytest.raises(ValueError, match=r"chief_risk_officer\.delivery_part_unknown"):
            _operation(
                service,
                {**parts[0]["delivery"]["next_request"], "delivery_part": len(parts) + 1},
            )
        # The historical HTML/JSON export is not a model-context page and is
        # not bounded here; the assessment authored after every part is.
        issue = _issue(findings[0], whole["dossier"]["findings"][0]["affected_entities"][0])
        dossier = service.review.review_publications  # the owner's dossier reader is the Host's
        del dossier
        from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
            PortfolioReviewDossier,
        )

        sealed = PortfolioReviewDossier.model_validate(whole["dossier"])
        assessment = answer_from_issues(sealed, _submission(issue)).model_dump(mode="json")
        reviewed = service.post(
            "/api/cro/assessment",
            {
                k: v
                for k, v in {
                    **parts[-1]["submission_template"],
                    "review_answer": assessment,
                }.items()
                if k != "operation"
            },
        )
        assert reviewed["disposition"] == "ADMITTED", reviewed
        service.drain()
        assert service.evidence_cro()["state"] == "REVIEW_PUBLISHED"
    finally:
        service.session.stop()
