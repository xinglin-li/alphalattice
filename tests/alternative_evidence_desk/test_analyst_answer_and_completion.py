"""The Analyst answers with judgment only; completion is what was delivered.

An answer names findings -- issuer, topic, direction, a summary and the
aliases of the excerpts that state it -- and nothing else: no handle, no
hash, no per-check report, no review flag. The Host screens each finding on
its own and names every problem in plain words, maps the aliases back,
assigns the handles and writes the brief's texts. Each issuer's review state
is a delivery fact (`issuer-delivery-facts-v3`): a subset is an answer, an
empty list is an answer with no findings, and an issuer whose excerpts were
delivered and not named reads as read with nothing reported, never as a gap.
Completions sealed earlier read back as sealed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceAnalystAnswer,
    AnalysisCompletion,
    IssuerCheckOutcome,
)
from alphalattice.evidence.alternative_evidence.analysis.packet import span_aliases
from alphalattice.evidence.alternative_evidence.analysis.submissions import (
    AlternativeEvidenceBriefAuthorityError,
    accepted_analyst_answer,
    screen_analyst_answer,
    seal_alternative_evidence_analyst_brief,
)
from alphalattice.evidence.alternative_evidence.contracts import AlternativeEvidenceReviewStatus
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    COMPLETION_SCHEMA,
    MIXED_COMPLETION_SCHEMA,
    _completion_schema,
    _executed,
)
from alphalattice.protocols.actor_execution import ActorKind
from tests.alternative_evidence_desk.document_intelligence_support import (
    _obligation,
    _open_recorded,
)
from tests.alternative_evidence_desk.planted_corpus import (
    _NOW,
    PLAYPEN_ROOT,
    _recorded_document,
)

ENTITIES = ("AAPL", "MSFT", "NVDA")


def _inputs(tmp_path: Path) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Two issuers with documents and excerpts, one (NVDA) with no document;
    each issuer's excerpt aliases in reading order."""

    runtime, request, _registry, snapshot, document_set, generation = _open_recorded(
        tmp_path,
        entities=ENTITIES,
        documents=(_recorded_document("AAPL"), _recorded_document("MSFT")),
    )
    receipt, spans = runtime.select_evidence(
        request=request, document_set=document_set, generation=generation
    )
    runtime.close()
    aliases: dict[str, list[str]] = {}
    for alias, span in span_aliases(spans, request.ordered_entity_ids).items():
        aliases.setdefault(span.entity_id, []).append(alias)
    assert set(aliases) == {"AAPL", "MSFT"}, aliases
    common: dict[str, Any] = {
        "request": request,
        "obligation": _obligation(request),
        "snapshot": snapshot,
        "document_set": document_set,
        "generation": generation,
        "access_receipt": receipt,
        "resolved_spans": spans,
        "analysis_policy": runtime.analysis_policy,
        "decision_policy": runtime.decision_policy,
        "playpen_root": PLAYPEN_ROOT,
        "completed_at": _NOW,
        "actor_kind": ActorKind.HUMAN,
        "actor_id": "researcher@example.test",
    }
    return common, aliases


def _finding(issuer: str, cite: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "issuer": issuer,
        "topic": "OPERATIONS_SUPPLY",
        "direction": "ADVERSE",
        "summary": f"{issuer}: supplier interruption reduced component capacity.",
        "cite": cite,
        **extra,
    }


def _states(receipt: Any) -> dict[str, str]:
    completion = receipt.brief.completion
    assert completion is not None and completion.completion_schema == "issuer-delivery-facts-v3"
    assert all(value.checks_executed == () for value in completion.issuers)
    return {value.entity_id: str(value.state) for value in completion.issuers}


def test_completion_states_are_delivery_facts_and_a_subset_is_an_answer(tmp_path: Path) -> None:
    """requirement: an answer naming one issuer's finding completes every
    issuer by what was delivered -- the named one with findings, the one read
    and not named with none, the one without a document as missing -- and
    reports no check for anyone."""

    common, aliases = _inputs(tmp_path)
    answer = AlternativeEvidenceAnalystAnswer.model_validate(
        {"findings": [_finding("AAPL", aliases["AAPL"][:2])]}
    )
    receipt = seal_alternative_evidence_analyst_brief(**common, answer=answer)

    assert _states(receipt) == {
        "AAPL": "EXECUTED_WITH_FINDINGS",
        "MSFT": "EXECUTED_NO_FINDINGS",
        "NVDA": "SOURCE_MISSING",
    }
    assert receipt.brief.completion.executed_entity_ids == ("AAPL", "MSFT")
    (finding,) = receipt.brief.findings
    assert finding.finding_handle == "FIND-001", "the Host assigns the handle"
    assert finding.lifecycle is None, "an unstated lifecycle is not invented"
    spans = {span.span_handle for span in common["resolved_spans"]}
    assert set(finding.supporting_span_handles) <= spans, "aliases become span handles"
    # The actor binding names the answer as written; the brief's texts are the Host's.
    assert receipt.answer == answer
    assert receipt.submission.check_outcomes == ()
    assert not receipt.brief.requires_human_review
    assert receipt.brief.source_coverage_assessment.endswith("No admitted document for: NVDA.")


def test_an_empty_answer_is_complete_with_no_findings(tmp_path: Path) -> None:
    """requirement: an empty list is an answer, not an incomplete analysis --
    no review flag is asked for and none is set."""

    common, _aliases = _inputs(tmp_path)
    receipt = seal_alternative_evidence_analyst_brief(
        **common, answer=AlternativeEvidenceAnalystAnswer()
    )

    assert receipt.brief.review_status is AlternativeEvidenceReviewStatus.COMPLETE
    assert receipt.brief.findings == ()
    assert not receipt.brief.requires_human_review
    assert receipt.brief.executive_summary == "0 findings reported on 0 of 3 issuers."
    assert _states(receipt)["MSFT"] == "EXECUTED_NO_FINDINGS"


def test_each_finding_is_screened_on_its_own_in_plain_words(tmp_path: Path) -> None:
    """requirement: one bad item never refuses the others. Each problem names
    its item and what is wrong; spelling of a closed value is normalized;
    nothing omitted is a problem."""

    common, aliases = _inputs(tmp_path)
    entity_ids = common["request"].ordered_entity_ids
    table = span_aliases(common["resolved_spans"], entity_ids)
    written = {
        "findings": [
            _finding("aapl", [aliases["AAPL"][0].lower()], direction="adverse"),
            _finding("TSLA", aliases["AAPL"][:1]),
            _finding("MSFT", ["S999"]),
            _finding("MSFT", aliases["AAPL"][:1]),
            _finding(
                "MSFT", aliases["MSFT"][:1], contrary=aliases["MSFT"][:1], lifecycle="ONGOING"
            ),
            {"issuer": "MSFT", "topic": "WEATHER", "summary": "", "cite": []},
        ]
    }
    screened = screen_analyst_answer(written, entity_ids=entity_ids, aliases=table)

    assert screened.accepted_numbers == (1,)
    problems = {(value.item, value.text) for value in screened.problems}
    assert (2, "issuer 'TSLA' is not an issuer of this bundle; use one of AAPL, MSFT, NVDA.") in (
        problems
    )
    assert (3, "S999 is not an excerpt of this bundle.") in problems
    assert (4, f"{aliases['AAPL'][0]} is an excerpt of AAPL, not of MSFT.") in problems
    assert (
        5,
        f"{aliases['MSFT'][0]} is cited both for and against; keep it in one list.",
    ) in problems
    item_six = {text for item, text in problems if item == 6}
    assert "'direction' is missing." in item_six
    assert "'summary' is empty." in item_six
    assert "'cite' needs at least 1 entry." in item_six
    assert any(text.startswith("'topic' must be one of") for text in item_six)
    # The accepted part is spelled as the packet names it.
    accepted = accepted_analyst_answer(screened, entity_ids=entity_ids)
    assert accepted.findings[0].issuer == "AAPL"
    assert accepted.findings[0].cite == (aliases["AAPL"][0],)
    # The sealer refuses an answer holding an item the screen did not accept.
    with pytest.raises(AlternativeEvidenceBriefAuthorityError, match="answer_invalid"):
        seal_alternative_evidence_analyst_brief(
            **common,
            answer=AlternativeEvidenceAnalystAnswer.model_validate(
                {"findings": [_finding("MSFT", ["S999"])]}
            ),
        )


def test_historical_completions_read_as_sealed_and_the_aggregate_rule_is_per_issuer() -> None:
    """requirement: a v1 completion validates and serializes without new keys
    (its hashes stand); mixed legacy/new children never inflate the new
    child's coverage (the base counted every issuer when any child was
    legacy)."""

    v1 = AnalysisCompletion(
        completion_schema="issuer-check-outcomes-v1",
        required_checks=("supporting evidence",),
        issuers=(
            IssuerCheckOutcome(
                entity_id="OLD",
                state="EXECUTED_NO_FINDINGS",
                documents_admitted=1,
                spans_delivered=3,
                findings=0,
                contradicting_findings=0,
                checks_executed=("supporting evidence",),
            ),
        ),
    )
    dumped = v1.model_dump(mode="json")
    assert set(dumped["issuers"][0]) == {
        "entity_id",
        "state",
        "documents_admitted",
        "spans_delivered",
        "findings",
        "contradicting_findings",
        "checks_executed",
    }
    assert AnalysisCompletion.model_validate(dumped) == v1

    states = {
        "A": "UNSTATED",
        "B": "SOURCE_MISSING",
        "C": "EXECUTED_NO_FINDINGS",
        "D": "NOT_REPORTED",
    }
    assert _executed(states) == {"A", "C"}, "legacy A counts; missing B and unreported D do not"
    assert _completion_schema(states) == MIXED_COMPLETION_SCHEMA
    assert _completion_schema({"A": "UNSTATED"}) == ""
    assert _completion_schema({"B": "SOURCE_MISSING", "C": "EXECUTED_WITH_FINDINGS"}) == (
        COMPLETION_SCHEMA
    )
