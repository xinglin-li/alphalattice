"""One finding is read as a bounded evidence package: the analyst's claim,
the exact excerpts it cites for and against with their source facts, the
question that found each passage, the unit's open questions, and -- once a
review is published -- the reviewer's disposition, read back exactly.

The dossier the CRO reviews carries findings and citation *handles*; a
reviewer who must judge materiality needs the verified text behind a
handle without the whole packet. The package is a read of what the product
proved was delivered and what was submitted; nothing in it is adjudicated.
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewDossier,
)
from tests.alternative_evidence_desk.planted_corpus import _CitingActor
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

ONE_UNIT = "u01"
"""The book's only unit: every book is prepared as a coverage run (C2)."""


def _operation(service: _Service, document: dict[str, Any]) -> dict[str, Any]:
    assert service.session.operations is not None
    return service.session.operations.execute(
        PortfolioResearchOperationRequest(**document), caller="HUMAN"
    )


def test_a_finding_is_read_as_its_evidence_package_before_and_after_review(
    tmp_path: Path,
) -> None:
    workspace, report = build_workspace(tmp_path)
    authority = build_authority(tmp_path=tmp_path, report=report)
    service = start_service(workspace, authority, tmp_path)
    try:
        selected = {"result_hash": service.result_hash()}
        prepared = service.post("/api/evidence/prepare", selected)
        assert prepared["disposition"] == "ADMITTED", prepared
        service.drain()
        task_id = prepared["task_id"]
        packet = service.review.evidence_task_adapter.prepared_packet(
            UUID(task_id), now=service.review.clock(), unit_id=ONE_UNIT
        )
        # The page's packet names which question found each passage and
        # instructs no one: the agent reads its bundle (C1).
        exported = _operation(
            service,
            {
                "operation": "EVIDENCE_PACKET",
                **selected,
                "task_id": task_id,
                "evidence_unit_id": ONE_UNIT,
            },
        )
        rendered = exported["packet"]
        assert '"found_by"' in rendered
        assert not any(
            f'"{key}"' in rendered
            for key in (
                "task_procedure",
                "question_kinds",
                "queries_run",
                "citation_rule",
                "span_rule",
                "inventory_detail",
            )
        )
        kinds = {value.kind for value in packet.receipt.queries}
        assert kinds == {"ADVERSE", "STATE"}
        answer = _CitingActor(topics={})(packet=packet).answer.model_dump(mode="json")
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
        dossier = _operation(service, {"operation": "CRO_REVIEW_DOSSIER", **selected})
        assert dossier["status"] == "CRO_DOSSIER_READY"
        finding = dossier["dossier"]["findings"][0]
        handle = finding["finding_handle"]

        # The Web route and the operation owner answer the same package.
        query = urllib.parse.urlencode({**selected, "finding_handle": handle})
        package = service.get(f"/api/cro/finding?{query}")
        in_process = _operation(
            service, {"operation": "CRO_REVIEW_FINDING", **selected, "finding_handle": handle}
        )
        assert package == json.loads(json.dumps(in_process, default=str))
        assert package["status"] == "FINDING_EVIDENCE_PACKAGE"
        assert package["atomic_claim"] == finding["summary"]
        assert package["affected_entities"] == finding["affected_entities"]
        assert package["event_state"]["topic"] == finding["topic"]
        assert package["review_dossier_hash"] == dossier["dossier"]["dossier_hash"]
        assert package["disposition"] is None, "no review yet: nothing to read back"
        assert len(package["support"]) == len(finding["supporting_span_handles"]) >= 1
        for excerpt in package["support"]:
            assert excerpt["excerpt"] and excerpt["source_verified"]
            assert excerpt["found_by"], "every delivered span was returned by a question"
            assert all(":" in item for item in excerpt["found_by"])
            assert excerpt["document_type"] and excerpt["revision_label"]
            assert excerpt["character_range"][0] < excerpt["character_range"][1]
        assert package["producing_publication"]["delivered_span_count"] >= 1

        # A stale dossier binding and an unknown finding are typed refusals.
        with pytest.raises(ValueError, match=r"chief_risk_officer\.delivery_continuation_stale"):
            _operation(
                service,
                {
                    "operation": "CRO_REVIEW_FINDING",
                    **selected,
                    "finding_handle": handle,
                    "review_dossier_hash": "0" * 64,
                },
            )
        with pytest.raises(ValueError, match=r"chief_risk_officer\.finding_handle_unknown"):
            _operation(
                service,
                {"operation": "CRO_REVIEW_FINDING", **selected, "finding_handle": "FIND-NOPE"},
            )

        # After the review, the same finding reads back with its disposition.
        sealed = PortfolioReviewDossier.model_validate(dossier["dossier"])
        issue = _issue(handle, finding["affected_entities"][0])
        assessment = answer_from_issues(sealed, _submission(issue)).model_dump(mode="json")
        reviewed = service.post(
            "/api/cro/assessment",
            {
                k: v
                for k, v in {
                    **dossier["submission_template"],
                    "review_answer": assessment,
                }.items()
                if k != "operation"
            },
        )
        assert reviewed["disposition"] == "ADMITTED", reviewed
        service.drain()
        section = service.evidence_cro()
        assert section["state"] == "REVIEW_PUBLISHED"
        historical = _operation(
            service,
            {
                "operation": "CRO_REVIEW_FINDING",
                **selected,
                "finding_handle": handle,
                "review_publication_hash": section["review_publication_hash"],
            },
        )
        assert historical["disposition"] is not None
        assert historical["disposition"]["finding_handle"] == handle
        assert json.loads(json.dumps(historical["support"], default=str)) == package["support"], (
            "the excerpts are the sealed ones"
        )
    finally:
        service.session.stop()
