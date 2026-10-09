"""The report is seven sections drawn from the accepted objects, and an
explicitly selected prior review is compared as facts.

Conclusion, coverage and work state, the material register, the
non-material / resolved / deferred register, follow-up actions, changes since
the selected prior report, and the evidence appendix -- rendered from the
same sealed review the JSON carries, so CLI, Web, HTML and JSON agree. The
comparison matches findings on issuers, topic and the analyst's words,
never by handle; a finding no longer found is reported as absent, never as
resolved; two reviews of different subjects are not comparable, by name.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from alphalattice.evidence.alternative_evidence.analysis.contracts import EvidenceTopic
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
    _NOW,
    _Service,
    build_authority,
    build_workspace,
    start_service,
)

SECTIONS = (
    "1. Conclusion",
    "2. Coverage and work state",
    "3. Material issue register",
    "4. Non-material, resolved and deferred register",
    "5. Follow-up actions",
    "6. Changes since the selected prior report",
    "7. Evidence appendix",
)


ONE_UNIT = "u01"
"""The book's only unit: every book is prepared as a coverage run."""


def _operation(service: _Service, document: dict[str, Any]) -> dict[str, Any]:
    assert service.session.operations is not None
    return service.session.operations.execute(
        PortfolioResearchOperationRequest(**document), caller="HUMAN"
    )


def _review(
    service: _Service,
    selected: dict[str, str],
    *,
    actor: _CitingActor,
    material: bool,
    evidence_as_of: str | None = None,
) -> str:
    """Prepare, analyse with `actor`, assess; return the review publication hash."""

    prepared = service.post(
        "/api/evidence/prepare",
        {**selected, **({} if evidence_as_of is None else {"evidence_as_of": evidence_as_of})},
    )
    assert prepared["disposition"] == "ADMITTED", prepared
    service.drain()
    task_id = prepared["task_id"]
    packet = service.review.evidence_task_adapter.prepared_packet(
        UUID(task_id), now=service.review.clock(), unit_id=ONE_UNIT
    )
    exported = _operation(
        service,
        {
            "operation": "EVIDENCE_PACKET",
            **selected,
            "task_id": task_id,
            "evidence_unit_id": ONE_UNIT,
        },
    )
    answer = actor(packet=packet).answer.model_dump(mode="json")
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
    # A second current analysis is an ambiguity the product refuses to
    # resolve itself: the person names the one read as of this cutoff.
    resolved = service.review.resolve_book(service.review.default_selector())
    eligible = service.review.eligible_evidence(resolved)
    if len(eligible) > 1:
        assert evidence_as_of is not None
        cutoff = datetime.fromisoformat(evidence_as_of)
        (view,) = [v for _, v in eligible if v.lineage.request.evidence_as_of == cutoff]
        chosen = service.post(
            "/api/evidence-select",
            {**selected, "analysis_publication_hash": view.publication.publication_hash},
        )
        assert chosen["chosen_by"] == "HUMAN", chosen
    dossier = _operation(service, {"operation": "CRO_REVIEW_DOSSIER", **selected})
    assert dossier.get("status") == "CRO_DOSSIER_READY", dossier
    sealed = PortfolioReviewDossier.model_validate(dossier["dossier"])
    finding = dossier["dossier"]["findings"][0]
    submission = (
        _submission(_issue(finding["finding_handle"], finding["affected_entities"][0]))
        if material
        else _submission()
    )
    assessment = answer_from_issues(sealed, submission).model_dump(mode="json")
    reviewed = service.post(
        "/api/cro/assessment",
        {
            k: v
            for k, v in {**dossier["submission_template"], "review_answer": assessment}.items()
            if k != "operation"
        },
    )
    assert reviewed["disposition"] == "ADMITTED", reviewed
    service.drain()
    section = service.evidence_cro()
    assert section["state"] == "REVIEW_PUBLISHED", section
    return str(section["review_publication_hash"])


def test_the_report_has_seven_sections_and_compares_a_selected_prior_review(
    tmp_path: Path,
) -> None:
    workspace, report = build_workspace(tmp_path)
    authority = build_authority(tmp_path=tmp_path, report=report)
    service = start_service(workspace, authority, tmp_path)
    try:
        selected = {"result_hash": service.result_hash()}
        first = _review(service, selected, actor=_CitingActor(topics={}), material=True)
        # A later cutoff, a different reading of one issuer: its operations
        # finding is gone, a legal one is new; the other issuer's words are
        # unchanged; the reviewer no longer records a material issue.
        prepared_scope = _operation(service, {"operation": "CRO_REVIEW_DOSSIER", **selected})
        entity = prepared_scope["dossier"]["issuers"][-1]["entity_id"]
        later = (_NOW.replace(hour=_NOW.hour + 1)).isoformat()
        second = _review(
            service,
            selected,
            actor=_CitingActor(topics={entity: EvidenceTopic.LEGAL_REGULATORY}),
            material=False,
            evidence_as_of=later,
        )
        assert second != first

        # The comparison, through the operation owner and the Web route alike.
        query = urllib.parse.urlencode(
            {**selected, "review_publication_hash": second, "prior_review_publication_hash": first}
        )
        web = service.get(f"/api/evidence-cro/export?{query}")
        cli = _operation(
            service,
            {
                "operation": "EVIDENCE_CRO_EXPORT",
                **selected,
                "review_publication_hash": second,
                "prior_review_publication_hash": first,
            },
        )
        assert web["export_hash"] == cli["export_hash"], "CLI and Web export the same object"
        changes = web["changes_since_prior_review"]
        assert changes["status"] == "REVIEW_CHANGES"
        assert changes["prior_publication_hash"] == first
        new = changes["findings"]["new"]
        gone = changes["findings"]["no_longer_found"]
        unchanged = changes["findings"]["unchanged"]
        assert [n["topic"] for n in new] == ["LEGAL_REGULATORY"]
        assert new[0]["affected_entities"] == [entity]
        assert len(gone) == 1 and gone[0]["affected_entities"] == [entity]
        assert "not thereby resolved" in gone[0]["meaning"]
        assert unchanged, "the other issuer's finding is stated in the same words"
        moved = changes["dispositions_moved"]
        assert moved and moved[0]["prior_disposition"] == "MATERIAL_ISSUE"
        # The second answer names nothing: the program lists the finding as
        # read and not named, never as judged immaterial.
        assert moved[0]["disposition"] == "NOT_ADDRESSED"
        assert changes["prior_evidence_as_of"] < changes["current_evidence_as_of"]
        # The typed disclosure families ran on both sides (the fixture's
        # releases carry no 10-K/10-Q, so nothing is observed): compared
        # under the same rules, with nothing to report and nothing invented.
        typed = changes["typed_disclosure_changes"]
        assert typed["status"] == "TYPED_DISCLOSURE_CHANGES", typed
        assert typed["current_rules"] == typed["prior_rules"]
        assert typed["changes"] == []
        assert json.loads(json.dumps(cli["changes_since_prior_review"], default=str)) == changes

        # The HTML carries the seven sections in order, from the same objects.
        html_text = web["html"]
        positions = [html_text.index(f"<h3>{title}</h3>") for title in SECTIONS]
        assert positions == sorted(positions), "the seven sections, in order"
        assert "Findings no longer found (not thereby resolved)" in html_text
        assert "Dispositions that moved: 1" in html_text
        assert re.search(r"<td>[A-Z]+</td><td>[A-Z, ]+</td><td>\d+\.\d{3}%</td>", html_text), (
            "every reviewed issuer is a row with its weight"
        )
        assert "Not answered by this assessment" not in html_text, "every finding was answered"
        first_html = service.get(
            "/api/evidence-cro/export?"
            + urllib.parse.urlencode({**selected, "review_publication_hash": first})
        )["html"]
        assert "ISSUE-" in first_html and "Recommendation: " in first_html, (
            "the material register carries the reviewer's risk and advice under its finding"
        )
        assert "verified excerpts those handles resolve to" in html_text

        # Without a prior report the section says so; a mismatched pair refuses by name.
        alone = service.get(
            "/api/evidence-cro/export?"
            + urllib.parse.urlencode({**selected, "review_publication_hash": second})
        )
        assert alone["changes_since_prior_review"] is None
        assert "No prior report was selected" in alone["html"]
        backwards = service.review.review_changes(first, second)
        assert backwards["status"] == "REVIEWS_NOT_COMPARABLE"
        assert "later cutoff" in backwards["reason"], backwards
    finally:
        service.session.stop()
