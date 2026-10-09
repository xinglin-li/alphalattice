"""Completion is what was delivered; a review names the major negatives it finds.

 and ER5 on the controlled fifty-issuer book through the real service. An
issuer whose documents state nothing is read and clean of reported findings --
not incomplete, not proof of no risk; an issuer with no source could not be
read, and its healthy unit peers go on without it; both stay in the book's
denominators and neither is counted as reviewed. A reviewer answers with the
risks it finds and is asked for no disposition: a finding it does not name is
read and not named, listed by the program, never a gap and never a refusal.
Controlled submissions test the plumbing, never a model's judgment.
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

from alphalattice.interface.local_application.evidence_cro import (
    _ANSWERED_ROUTE_IN_WORDS,
    render_review_export,
)
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewDossier,
)
from tests.alternative_evidence_desk.portfolio_coverage_support import (
    COVERAGE_BOOK_SIZE,
    COVERAGE_ENTITIES,
    COVERAGE_TOPICS,
    UNIT_COUNT,
    ClaimCitingActor,
    answer_unit,
    build_coverage_workspace,
    coverage_authority,
    coverage_documents,
    http_body,
    run_one,
    start_coverage_service,
)
from tests.alternative_evidence_desk.review_dossiers import ControlledRisk, controlled_answer

QUIET = "QA05"
MISSING = "QA17"


@pytest.fixture(scope="module")
def built_book(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return build_coverage_workspace(tmp_path_factory.mktemp("completion-book"))


@pytest.fixture
def book(built_book: Any, tmp_path: Path) -> Any:
    import shutil

    workspace, report = built_book
    copy = tmp_path / "workspace"
    shutil.copytree(workspace, copy)
    return copy, report


def _prepared_and_analysed(service: Any) -> dict[str, Any]:
    preview = run_one(service, {"operation": "EVIDENCE_PREVIEW"})
    prepared = run_one(service, preview["next_requests"]["prepare"])
    assert prepared["disposition"] == "ADMITTED", prepared
    service.drain()
    section = service.evidence_cro()
    packets = {k: v for k, v in section["next_requests"].items() if k.startswith("packet_")}
    for key in sorted(packets):
        answer_unit(service, packets[key])
        service.drain()
    return dict(service.evidence_cro())


def test_a_quiet_issuer_is_checked_and_a_sourceless_issuer_is_not_counted_as_reviewed(
    book: Any, tmp_path: Path
) -> None:
    """Requirement: completion comes from delivery facts. QA05's
    documents state no claim: it was read, nothing was reported, and it is
    reviewed. QA17 has no document: its unit's seven peers are prepared and
    analysed without it; QA17 is `SOURCE_MISSING`, named, in the denominator
    and not reviewed; the review of the book is partial, never full."""

    workspace, report = book
    documents = coverage_documents(
        tuple(entity for entity in COVERAGE_ENTITIES if entity != MISSING),
        quiet=frozenset({QUIET}),
    )
    authority = coverage_authority(
        workspace,
        report,
        documents=documents,
        analysis_actor=ClaimCitingActor(topics=COVERAGE_TOPICS),
    )
    service = start_coverage_service(workspace, authority, tmp_path)
    try:
        section = _prepared_and_analysed(service)
        assert section["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW", section["state"]
        progress = section["coverage_progress"]
        assert progress["units_prepared"] == UNIT_COUNT and progress["units_failed"] == 0
        assert progress["units_analyzed"] == UNIT_COUNT, "the sourceless issuer's unit went on"

        dossier_body = run_one(service, section["next_requests"]["dossier"])
        dossier = PortfolioReviewDossier.model_validate(dossier_body["dossier"])
        assert dossier.completion_schema == "issuer-review-state-v1"
        states = {issuer.entity_id: issuer.review_state for issuer in dossier.issuers}
        assert states[QUIET] == "EXECUTED_NO_FINDINGS"
        assert states[MISSING] == "SOURCE_MISSING"
        assert states[COVERAGE_ENTITIES[0]] == "EXECUTED_WITH_FINDINGS"
        assert set(states.values()) == {
            "EXECUTED_WITH_FINDINGS",
            "EXECUTED_NO_FINDINGS",
            "SOURCE_MISSING",
        }
        assert not any(QUIET in finding.affected_entities for finding in dossier.findings)
        # Reviewed weight counts the executed issuers only; the missing one is named.
        assert dossier.coverage.selected_issuer_coverage == pytest.approx(
            (COVERAGE_BOOK_SIZE - 1) / COVERAGE_BOOK_SIZE
        )
        assert dossier.coverage.reviewed_ending_weight_coverage < 1.0
        assert any(
            reason.startswith(f"{MISSING} is not reviewed: SOURCE_MISSING")
            for reason in dossier.coverage.unavailable_reasons
        )
        assert any(MISSING in value for value in dossier.coverage.missing_evidence)

        # An answer naming one risk is a whole answer: nothing is asked of the
        # other findings. The review judges what it read: the sourceless
        # issuer is a stated limit and an evidence gap, never a gate, and the
        # review's state is partial.
        answer = controlled_answer(dossier, ControlledRisk(COVERAGE_ENTITIES[0]))
        assert answer.risks, "the controlled reviewer names the first issuer's finding"
        schema = dossier_body["assessment_schema"]
        assert "finding_dispositions" not in json.dumps(schema)
        reviewed = service.post(
            "/api/cro/assessment",
            http_body(
                {
                    **dossier_body["submission_template"],
                    "review_answer": answer.model_dump(mode="json"),
                }
            ),
        )
        assert reviewed["disposition"] == "ADMITTED", reviewed
        service.drain()
        section = service.evidence_cro()
        assert section["state"] == "REVIEW_PUBLISHED"
        assert section["review_state"] == "PARTIAL"
        view = service.review.review_publications.read(section["review_publication_hash"])
        conclusions = {
            value.entity_id: value.conclusion for value in view.recommendation.issuer_conclusions
        }
        assert conclusions[MISSING] == "EVIDENCE_GAP"
        assert view.recommendation.route in {"MATERIAL_OBJECTION", "ACCEPT_WITH_LIMITS"}
        assert conclusions[COVERAGE_ENTITIES[0]] in {"ISSUE_STANDS", "ISSUE_NOTED_WITHIN_LIMITS"}
        assert view.receipt.outcome.review_state == "PARTIAL"
        assert any(MISSING in value for value in view.recommendation.limitations)
        assert view.receipt.outcome.disposition_completeness == "UNSTATED"
        assert view.receipt.answer == answer
        export = service.get(
            "/api/evidence-cro/export?"
            + urllib.parse.urlencode(
                {
                    "result_hash": service.result_hash(),
                    "review_publication_hash": section["review_publication_hash"],
                }
            )
        )
        cited = len(answer.risks[0].findings)
        assert (
            f"{cited} cited by a named risk, {len(dossier.findings) - cited} read and "
            "not named as a risk"
        ) in export["html"]

        # Historical readback stays a positive control: a receipt sealed
        # before answers renders its dispositions as sealed.
        historical = json.loads(json.dumps(export, default=str))
        historical["review"]["receipt"].pop("answer")
        historical["review"]["receipt"]["outcome"].pop("disposition_completeness", None)
        rendered = render_review_export(historical, None)
        assert "Finding dispositions: UNSTATED" in rendered
        assert "historical reading sealed before dispositions were required" in rendered
    finally:
        service.session.stop()


def test_a_subset_answer_is_a_review_and_the_rest_is_listed_as_not_named(
    book: Any, tmp_path: Path
) -> None:
    """requirement (ER5, restated by the answer format): a reviewer names the
    major negatives it finds and nothing else. The findings it does not name
    are listed by the program as read and not named -- the review is not
    partial for them and their issuers are not unadjudicated."""

    workspace, report = book
    authority = coverage_authority(
        workspace, report, analysis_actor=ClaimCitingActor(topics=COVERAGE_TOPICS)
    )
    service = start_coverage_service(workspace, authority, tmp_path)
    try:
        section = _prepared_and_analysed(service)
        dossier_body = run_one(service, section["next_requests"]["dossier"])
        dossier = PortfolioReviewDossier.model_validate(dossier_body["dossier"])
        assert len(dossier.evidence_children) == UNIT_COUNT
        answer = controlled_answer(dossier, ControlledRisk(COVERAGE_ENTITIES[0]))
        reviewed = service.post(
            "/api/cro/assessment",
            http_body(
                {
                    **dossier_body["submission_template"],
                    "review_answer": answer.model_dump(mode="json"),
                }
            ),
        )
        assert reviewed["disposition"] == "ADMITTED", reviewed
        service.drain()
        section = service.evidence_cro()
        assert section["state"] == "REVIEW_PUBLISHED"
        view = service.review.review_publications.read(section["review_publication_hash"])
        conclusions = {v.entity_id: v.conclusion for v in view.recommendation.issuer_conclusions}
        # The named issuer's risk was adjudicated (its route is the matrix's,
        # not this test's claim); an issuer read and not named is not a gap.
        assert conclusions[COVERAGE_ENTITIES[0]] in {"ISSUE_STANDS", "ISSUE_NOTED_WITHIN_LIMITS"}
        assert conclusions[COVERAGE_ENTITIES[1]] == "NO_ADVERSE_ISSUE"
        assert view.receipt.outcome.disposition_completeness == "UNSTATED"
        export = service.get(
            "/api/evidence-cro/export?"
            + urllib.parse.urlencode(
                {
                    "result_hash": service.result_hash(),
                    "review_publication_hash": section["review_publication_hash"],
                }
            )
        )
        assert "Read and not named as a risk by the reviewer" in export["html"]
        unnamed = next(
            f.finding_handle
            for f in dossier.findings
            if COVERAGE_ENTITIES[1] in f.affected_entities
        )
        assert unnamed in export["html"]
        assert "some findings were deferred" not in export["html"]
        # Written by the program (S6): the route in the answer format's words,
        # what the review did not read, and the issuers whose findings no risk names
        # -- in the report as in the export.
        route = str(view.recommendation.route)
        assert _ANSWERED_ROUTE_IN_WORDS[route] in export["html"]
        assert "What this review did not read (written by the program):" in export["html"]
        assert "Issuers whose findings no risk names (written by the program): " in export["html"]
        assert section["limitations"] == list(view.recommendation.limitations)
        assert any("read by the reviewer and not named" in v for v in section["limitations"])
        assert unnamed in section["not_addressed_findings"]
        assert COVERAGE_ENTITIES[1] in section["not_addressed_issuers"]
        assert COVERAGE_ENTITIES[0] not in section["not_addressed_issuers"]
        row = next(r for r in section["issuer_rows"] if r["entity_id"] == COVERAGE_ENTITIES[1])
        assert row["findings_summary"].endswith("read, none named as a risk")
        assert row["findings_summary_parts"] == [[row["findings_summary"], ""]]
    finally:
        service.session.stop()
