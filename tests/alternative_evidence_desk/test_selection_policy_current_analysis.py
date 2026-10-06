"""An analysis produced under one matter selection never masquerades as the
answer to another: the application's policy match compares the effective
method and family configuration by the request's own identity, so the
current eligible analyses, the automatic evidence selection, the section's
state and the dossier all follow the selection the authority asks for now.
Since the retirement (first-release integration T5) the integrated selection
is the one current method: an authority moved to a retired selection finds
no answer in the integrated analysis, and the analysis stays readable as
history."""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any
from uuid import UUID

from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.evidence.alternative_evidence.contracts import (
    INTEGRATED_FAMILY_SPELLING,
    MATTER_FAMILY_CORPORATE_EVENT,
    MATTER_FAMILY_LITIGATION,
    MATTER_SELECTION_CANDIDATE,
    MATTER_SELECTION_INTEGRATED,
    MatterSelectionPolicy,
)
from alphalattice.evidence.alternative_evidence.runtime.policy import AdmittedEvidencePolicy
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    BookSelector,
)
from tests.alternative_evidence_desk.issuer_listing import ISSUER_TOPICS
from tests.alternative_evidence_desk.matter_selection_support import _filings
from tests.alternative_evidence_desk.planted_corpus import _CitingActor
from tests.alternative_evidence_desk.review_http_support import (
    _Service,
    build_authority,
    build_workspace,
    start_service,
)

COMBINED = MatterSelectionPolicy(
    method=MATTER_SELECTION_CANDIDATE,
    families=(MATTER_FAMILY_LITIGATION, MATTER_FAMILY_CORPORATE_EVENT),
)
INTEGRATED = MatterSelectionPolicy(
    method=MATTER_SELECTION_INTEGRATED, families=INTEGRATED_FAMILY_SPELLING
)


ONE_UNIT = "u01"
"""The book's only unit: every book is prepared as a coverage run (C2)."""


def _authority(tmp_path: Path, report: Any, policy: MatterSelectionPolicy | None) -> Any:
    authority = build_authority(
        tmp_path=tmp_path, report=report, extra_documents=_filings, model_authority_admitted=False
    )
    return authority.__class__(
        **{
            **{f: getattr(authority, f) for f in authority.__dataclass_fields__},
            "evidence_policy": AdmittedEvidencePolicy(
                admit_model_review=False, matter_selection=policy
            ),
        }
    )


def _publish_analysis(service: _Service, selected: dict[str, str]) -> str:
    """Prepare under the service's selection and publish a controlled analysis
    through the public operations; returns the publication hash."""

    preview = service.get("/api/evidence/preview?" + urllib.parse.urlencode(selected))
    prepare = preview["next_requests"]["prepare"]
    prepared = service.post(
        "/api/evidence/prepare", {k: v for k, v in prepare.items() if k != "operation"}
    )
    service.drain()
    task_id = UUID(prepared["task_id"])
    assert service.registry.task(task_id).lifecycle is TaskLifecycle.SUCCEEDED
    adapter = service.review.evidence_task_adapter
    exported = service.get(
        "/api/evidence/packet?"
        + urllib.parse.urlencode(
            {**selected, "task_id": str(task_id), "evidence_unit_id": ONE_UNIT}
        )
    )
    packet = adapter.prepared_packet(task_id, now=service.review.clock(), unit_id=ONE_UNIT)
    answer = _CitingActor(topics=ISSUER_TOPICS)(packet=packet).answer.model_dump(mode="json")
    document = {**exported["submission_template"], "analysis_answer": answer}
    submitted = service.post(
        "/api/evidence/analysis", {k: v for k, v in document.items() if k != "operation"}
    )
    service.drain()
    view = adapter.published_analysis(UUID(submitted["task_id"]), now=service.review.clock())
    return str(view.publication.publication_hash)


def _state(service: _Service, selected: dict[str, str]) -> dict[str, Any]:
    application = service.review
    resolved = application.resolve_book(BookSelector(result_hash=selected["result_hash"]))
    eligible = application.eligible_evidence(resolved)
    section = service.evidence_cro(selected["result_hash"])
    status_code, dossier = service.request("/api/cro/dossier?" + urllib.parse.urlencode(selected))
    current = None if not eligible else application.current_evidence(question=eligible[0][0])
    return {
        "eligible": [v.publication.publication_hash for _q, v in eligible],
        "current": None if current is None else current.publication.publication_hash,
        "state": section["state"],
        "dossier_status": status_code,
        "dossier_body_status": dossier.get("status") or dossier.get("disposition"),
    }


def test_an_analysis_under_another_selection_is_not_the_current_answer(tmp_path: Path) -> None:
    """requirement (3), restated by the retirement: a controlled analysis
    published under the integrated selection; the admitted selection moved
    to a retired one -- the production plan, the combined candidate: the
    analysis is neither eligible nor automatically selected, the section
    awaits evidence (its preview names the retirement) and the dossier does
    not prepare from it. Back under the integrated selection it is current
    again, and it stays readable as history throughout."""

    workspace, report = build_workspace(tmp_path)
    selected: dict[str, str] = {}
    integrated = start_service(workspace, _authority(tmp_path, report, INTEGRATED), tmp_path)
    try:
        selected = {"result_hash": integrated.result_hash()}
        integrated_hash = _publish_analysis(integrated, selected)
        before = _state(integrated, selected)
        assert before["eligible"] == [integrated_hash]
        assert before["current"] == integrated_hash
        assert before["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
        assert (
            before["dossier_status"] == 200 and before["dossier_body_status"] == "CRO_DOSSIER_READY"
        )
    finally:
        integrated.session.stop()

    for retired in (None, COMBINED):
        moved = start_service(workspace, _authority(tmp_path, report, retired), tmp_path)
        try:
            after = _state(moved, selected)
            assert after["eligible"] == [], after
            assert after["current"] is None
            assert after["state"] == "AWAITING_ALTERNATIVE_EVIDENCE", after
            assert (
                after["dossier_status"] != 200
                or after["dossier_body_status"] != "CRO_DOSSIER_READY"
            )
            # The integrated publication is still readable as history.
            readable = moved.review.evidence_publications.replay(
                integrated_hash, now=moved.review.clock()
            )
            assert readable.publication.publication_hash == integrated_hash
        finally:
            moved.session.stop()

    back = start_service(workspace, _authority(tmp_path, report, INTEGRATED), tmp_path)
    try:
        again = _state(back, selected)
        assert again["eligible"] == [integrated_hash]
        assert again["current"] == integrated_hash
        assert again["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
    finally:
        back.session.stop()
