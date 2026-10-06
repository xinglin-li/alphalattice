"""Three typed disclosure families are located, extracted and delivered from
the sections a filing keeps them in, without competing for a generic top-k
score: the stage's load-bearing controls.

Controlled fixtures cover the boundary shapes the retained corpus does not
show (a negative conclusion beside an ICFR one, a prior period's conclusion,
an actual unregistered sale, a bare "None", an 8-K reference, a termination
and a non-Rule arrangement, contents-only headings, an amendment of limited
scope, a truncated statement, a conflict, a negated conclusion verb, a none
with an inline or unread exception, a restated assertion with a changed
field). They prove the rules' semantics; they are not real-corpus coverage.
The real forms are measured separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alphalattice.evidence.alternative_evidence.analysis import packet as packet_module
from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceRetrievalAccessReceipt,
    TypedDisclosureRecord,
)
from alphalattice.evidence.alternative_evidence.analysis.disclosures import (
    CHANGE_CHANGED,
    CHANGE_CONTINUING,
    CHANGE_NEW_BASELINE,
    CHANGE_NEWLY_REPORTED,
    CHANGE_NOT_COMPARABLE,
    CHANGE_NOT_OBSERVED,
    CHANGE_RULE_OR_SOURCE,
    DEFINITIONS,
    TypedDisclosureFamily,
    TypedDisclosureState,
    compare_typed_disclosures,
    definitions_hash,
    extract_document,
)
from alphalattice.evidence.alternative_evidence.documents.structure import (
    DocumentStructure,
    parse_source_date,
)
from alphalattice.evidence.alternative_evidence.retrieval.session import (
    MAXIMUM_ISSUED_SOURCE_SPANS,
    MAXIMUM_SPAN_READS,
    MAXIMUM_TYPED_SPAN_READS,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from tests.alternative_evidence_desk.document_intelligence_support import (
    _obligation,
    _open_recorded,
)
from tests.alternative_evidence_desk.planted_corpus import _recorded_document

DCP = TypedDisclosureFamily.DISCLOSURE_CONTROLS_CONCLUSION.value
SALES = TypedDisclosureFamily.UNREGISTERED_EQUITY_SALES.value
PLANS = TypedDisclosureFamily.INSIDER_TRADING_ARRANGEMENTS.value
CONCENTRATION = TypedDisclosureFamily.CUSTOMER_CONCENTRATION.value
CYBER = TypedDisclosureFamily.CYBERSECURITY_THREAT_EFFECT.value

COVER_10Q = [
    "UNITED STATES",
    "SECURITIES AND EXCHANGE COMMISSION",
    "FORM 10-Q",
    "For the quarterly period ended June 30, 2026",
    "Indicate by check mark whether the registrant is a shell company. No",
]
COVER_10K = [
    "UNITED STATES",
    "SECURITIES AND EXCHANGE COMMISSION",
    "FORM 10-K",
    "For the fiscal year ended December 31, 2025",
    "Indicate by check mark whether the registrant is a shell company. No",
]
EVALUATION = (
    "Our management, with the participation of our principal executive officer and "
    "principal financial officer, evaluated the effectiveness of our disclosure controls "
    "and procedures as of the end of the period covered by this report."
)
DCP_EFFECTIVE = (
    "Based on that evaluation, our principal executive officer and principal financial "
    "officer concluded that our disclosure controls and procedures were effective at the "
    "reasonable assurance level."
)
DCP_NOT_EFFECTIVE = (
    "Based on that evaluation, our principal executive officer and principal financial "
    "officer concluded that, as of June 30, 2026, our disclosure controls and procedures "
    "were not effective because of the material weakness described below."
)
DCP_PRIOR = (
    "As previously reported, our principal executive officer and principal financial "
    "officer concluded that our disclosure controls and procedures were effective as of "
    "December 31, 2025."
)
ICFR_EFFECTIVE = (
    "Based on that assessment, management concluded that, as of December 31, 2025, the "
    "Company's internal control over financial reporting is effective."
)
ICFR_CHANGES = (
    "There were no changes in our internal control over financial reporting during the "
    "quarter ended June 30, 2026 that have materially affected, or are reasonably likely "
    "to materially affect, our internal control over financial reporting."
)
SALE = (
    "On May 12, 2026, we issued 40,000 shares of common stock to Harborline Capital in a "
    "private placement for an aggregate purchase price of $1,200,000 in a transaction "
    "exempt from registration under Section 4(a)(2) of the Securities Act."
)
SALE_TWO = (
    "On June 3, 2026, we issued 5,000 shares of common stock to a consultant as "
    "compensation for services, exempt from registration under Regulation D."
)
NO_SALES = (
    "There were no unregistered sales of equity securities during the quarter ended June 30, 2026."
)
REPURCHASES = (
    "Issuer Purchases of Equity Securities: during the quarter ended June 30, 2026 we "
    "repurchased 1,000 shares of our common stock under the program announced in 2025."
)
REFERENCE_8K = (
    "The unregistered sales of equity securities during the quarter were previously "
    "reported in our Current Report on Form 8-K filed on May 15, 2026."
)
PLANS_NONE = (
    "During the quarter ended June 30, 2026, no director or officer of the Company adopted "
    "or terminated a Rule 10b5-1 trading arrangement or non-Rule 10b5-1 trading arrangement, "
    "as each term is defined in Item 408(a) of Regulation S-K."
)
PLANS_NONE_EXCEPT = (
    "During the quarter ended June 30, 2026, there were no Rule 10b5-1 trading arrangements "
    "or non-Rule 10b5-1 trading arrangements adopted or terminated by any director or officer "
    "of the Company, except as follows:"
)
PLAN_ADOPTED = (
    "On May 8, 2026, Jane Q. Rivera, our Chief Financial Officer, adopted a Rule 10b5-1 "
    "trading plan for the sale of up to 12,000 shares of common stock between August 15, "
    "2026 and May 1, 2027. Ms. Rivera's plan will terminate on the earlier of May 1, 2027 "
    "and the date that all trades under the plan are completed."
)
PLAN_TERMINATED = (
    "On June 2, 2026, Mark T. Osei, Ph.D., a member of our Board of Directors, terminated "
    "a non-Rule 10b5-1 trading arrangement adopted on March 3, 2025 covering up to 4,000 "
    "shares of common stock."
)
PLAN_ISSUER = (
    "On June 20, 2026, the Company adopted a Rule 10b5-1 trading plan to repurchase shares "
    "under its repurchase program."
)


def _text(lines: list[str]) -> str:
    return "\n\n".join(lines) + "\n"


def _ten_q(
    *,
    item4: list[str] | None = None,
    item2: list[str] | None = None,
    item5: list[str] | None = None,
    contents: bool = False,
    omit_item4: bool = False,
) -> str:
    lines = list(COVER_10Q)
    if contents:
        lines += [
            "TABLE OF CONTENTS",
            "PART I. FINANCIAL INFORMATION",
            "ITEM 4. CONTROLS AND PROCEDURES",
            "PART II. OTHER INFORMATION",
            "ITEM 2. UNREGISTERED SALES OF EQUITY SECURITIES AND USE OF PROCEEDS",
            "ITEM 5. OTHER INFORMATION",
        ]
    lines += [
        "PART I. FINANCIAL INFORMATION",
        "ITEM 1. FINANCIAL STATEMENTS",
        "The condensed consolidated statements follow.",
        "ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION",
        "Results were in line with expectations for the period.",
    ]
    if not omit_item4:
        lines += [
            "ITEM 4. CONTROLS AND PROCEDURES",
            *(item4 or [EVALUATION, DCP_EFFECTIVE, ICFR_CHANGES]),
        ]
    lines += ["PART II. OTHER INFORMATION", "ITEM 1. LEGAL PROCEEDINGS", "None."]
    if item2 is not None:
        lines += ["ITEM 2. UNREGISTERED SALES OF EQUITY SECURITIES AND USE OF PROCEEDS", *item2]
    lines += ["ITEM 4. MINE SAFETY DISCLOSURES", "Not applicable."]
    if item5 is not None:
        lines += ["ITEM 5. OTHER INFORMATION", *item5]
    lines += ["ITEM 6. EXHIBITS", "See the exhibit index.", "SIGNATURES"]
    return _text(lines)


def _observe(text: str, form: str = "10-Q") -> dict[str, object]:
    structure = DocumentStructure(text, document_type=form)
    result = extract_document(text, structure)
    return {value.family.value: value for value in result.observations}


def _one(text: str, family: str, form: str = "10-Q"):
    return _observe(text, form)[family]


def _fields(instance) -> dict[str, str | None]:
    return {value.name: value.value for value in instance.fields}


# --------------------------------------------------------------------------
# Family 1: the disclosure-controls conclusion.


def test_dcp_conclusion_is_bound_with_its_evaluation_date_and_qualifier() -> None:
    observation = _one(_ten_q(), DCP)
    assert observation.state is TypedDisclosureState.EXTRACTED
    assert observation.part == "PART I" and observation.item == "4"
    (instance,) = observation.instances
    assert instance.polarity == "EFFECTIVE" and instance.action == "CONCLUDED"
    assert instance.period_end == parse_source_date("June 30, 2026")
    assert instance.period_basis.startswith("antecedent")
    assert instance.qualifiers == ("at the reasonable assurance level",)
    assert instance.character_start < observation.scope_range[1]  # type: ignore[index]
    # The antecedent evaluation sentence travels with the conclusion.
    assert instance.statement_text.startswith("Based on that evaluation")
    assert instance.character_start == _ten_q().index(EVALUATION)


def test_dcp_negative_conclusion_is_not_replaced_by_the_icfr_conclusion() -> None:
    """requirement: DCP versus ICFR/auditor opinion; the ICFR conclusion in
    the same section is never taken for the DCP one, and a not-effective
    conclusion with a material weakness stays negative and qualified."""

    text = _text(
        [
            *COVER_10K,
            "PART II",
            "ITEM 9A. CONTROLS AND PROCEDURES",
            EVALUATION,
            DCP_NOT_EFFECTIVE,
            "Management's Report on Internal Control over Financial Reporting",
            ICFR_EFFECTIVE,
            "The Company's independent registered public accounting firm has issued an "
            "unqualified opinion on the effectiveness of the Company's internal control over "
            "financial reporting.",
            "ITEM 9B. OTHER INFORMATION",
            "None.",
            "PART III",
            "ITEM 10. DIRECTORS",
        ]
    )
    observation = _one(text, DCP, "10-K")
    assert observation.state is TypedDisclosureState.EXTRACTED
    (instance,) = observation.instances
    assert instance.polarity == "NOT_EFFECTIVE"
    assert instance.period_end == parse_source_date("June 30, 2026")
    assert instance.period_basis == "stated in the sentence"
    assert any("material weakness" in value for value in instance.qualifiers)
    assert "ICFR conclusion in the same section (not this family)" in observation.context
    # ICFR only, no DCP conclusion at all: not found, with the ICFR noted.
    icfr_only = text.replace(DCP_NOT_EFFECTIVE, "")
    observation = _one(icfr_only, DCP, "10-K")
    assert observation.state is TypedDisclosureState.NOT_FOUND
    assert "ICFR conclusion" in observation.reason


def test_dcp_prior_period_conclusion_keeps_its_own_date_and_conflicts_only_within_a_period() -> (
    None
):
    """requirement: current versus prior period."""

    text = _ten_q(item4=[EVALUATION, DCP_EFFECTIVE, DCP_PRIOR, ICFR_CHANGES])
    observation = _one(text, DCP)
    assert observation.state is TypedDisclosureState.EXTRACTED
    assert "different periods" in observation.reason
    periods = {value.period_end for value in observation.instances}
    assert periods == {parse_source_date("June 30, 2026"), parse_source_date("December 31, 2025")}
    conflicting = _ten_q(item4=[EVALUATION, DCP_EFFECTIVE, DCP_NOT_EFFECTIVE, ICFR_CHANGES])
    observation = _one(conflicting, DCP)
    assert observation.state is TypedDisclosureState.AMBIGUOUS
    assert len(observation.instances) == 2


def test_dcp_missing_mandatory_section_and_truncated_statement_are_source_unavailable() -> None:
    """requirement: explicit none versus absent text; a mandatory section
    missing from the canonical text is not "not found", and a statement
    cut before its verb is not an assertion."""

    observation = _one(_ten_q(omit_item4=True, item2=[NO_SALES], item5=[PLANS_NONE]), DCP)
    assert observation.state is TypedDisclosureState.SOURCE_UNAVAILABLE
    assert "required on Form 10-Q" in observation.reason
    # An intact item in the same document keeps its own answer, and says the
    # text is incomplete elsewhere; an absent Part II item becomes unavailable.
    sales = _one(_ten_q(omit_item4=True, item2=[NO_SALES], item5=[PLANS_NONE]), SALES)
    assert sales.state is TypedDisclosureState.EXPLICIT_NONE
    assert any("incomplete elsewhere" in value for value in sales.context)
    absent = _one(_ten_q(omit_item4=True, item5=[PLANS_NONE]), SALES)
    assert absent.state is TypedDisclosureState.SOURCE_UNAVAILABLE
    truncated = _ten_q(
        item5=["During the quarter ended June 30, 2026, none of our directors or officers "]
    )
    observation = _one(truncated, PLANS)
    assert observation.state is TypedDisclosureState.SOURCE_UNAVAILABLE
    assert "truncated" in observation.reason
    assert observation.unrecognized[0].startswith("TRUNCATED: ")


def test_contents_headings_do_not_stand_for_the_body() -> None:
    """requirement: headings in contents only."""

    text = _ten_q(contents=True, item2=[NO_SALES], item5=[PLANS_NONE])
    observations = _observe(text)
    for family in (DCP, SALES, PLANS):
        observation = observations[family]
        assert observation.state in {
            TypedDisclosureState.EXTRACTED,
            TypedDisclosureState.EXPLICIT_NONE,
        }, family
        assert observation.instances
        # The region inspected is the body, after the contents block.
        assert observation.inspected_ranges[0][0] > text.index("TABLE OF CONTENTS")
        assert observation.inspected_ranges[0][0] > text.index("ITEM 1. FINANCIAL STATEMENTS")


# --------------------------------------------------------------------------
# Family 2: unregistered sales.


def test_unregistered_sales_are_told_from_repurchases_and_absence() -> None:
    """requirement: Item 2(a) versus repurchases; explicit none versus
    omitted text; a bare None is scoped to the item."""

    only_repurchases = _one(_ten_q(item2=[REPURCHASES]), SALES)
    assert only_repurchases.state is TypedDisclosureState.NOT_FOUND
    assert "repurchase" in only_repurchases.reason
    omitted = _one(_ten_q(), SALES)
    assert omitted.state is TypedDisclosureState.NOT_FOUND
    assert "omission is not an explicit none" in omitted.reason
    explicit = _one(_ten_q(item2=[NO_SALES, REPURCHASES]), SALES)
    assert explicit.state is TypedDisclosureState.EXPLICIT_NONE
    (instance,) = explicit.instances
    assert instance.period_end == parse_source_date("June 30, 2026")
    assert instance.subject == "unregistered sales of equity securities"
    bare = _one(_ten_q(item2=["None."]), SALES)
    assert bare.state is TypedDisclosureState.EXPLICIT_NONE
    assert bare.instances[0].qualifiers == ("stated as None",)
    # The same bare answer under Item 5 covers more than the plans family.
    bare_plans = _one(_ten_q(item5=["None."]), PLANS)
    assert bare_plans.state is TypedDisclosureState.NOT_FOUND


REPURCHASE_ROWS = [
    "The following table sets forth information relating to repurchases of our equity "
    "securities during the three months ended June 30, 2026:",
    "Period: April 1, 2026 \u2013 April 30, 2026; Total Number of Shares Purchased: 7,056,917 "
    "(2); Average Price Paid per Share (1): $178.49; Total Number of Shares Purchased as Part "
    "of Publicly Announced Plans or Programs: 7,056,917; Maximum Approximate Dollar Value of "
    "Shares that May Yet Be Purchased Under the Plans or Programs (Dollars in Billions): $16.9",
    "Period: May 1, 2026 \u2013 May 31, 2026; Total Number of Shares Purchased: \u2014; Average "
    "Price Paid per Share (1): $ \u2014; Total Number of Shares Purchased as Part of Publicly "
    "Announced Plans or Programs: \u2014; Maximum Approximate Dollar Value of Shares that May "
    "Yet Be Purchased Under the Plans or Programs (Dollars in Billions): $16.9",
    "Period: Total; Total Number of Shares Purchased: 7,056,917",
    "(1) Average price paid per share excludes commissions and the excise tax.",
]
"""A repurchase table as canonical extraction rules v4 carry it into Part II
Item 2: buybacks, never an unregistered sale."""


def test_a_carried_repurchase_table_is_not_an_unregistered_sale_and_is_delivered_whole() -> None:
    """requirement: with the repurchase table now in the region's text, the
    family still finds no unregistered sale -- no instance, no none, the
    repurchase noted as the other subject -- and the inspected scope it
    delivers runs through the whole table, rows and footnote, when the
    2,400-character cut lands inside a row; a table too large for the
    reader's per-span ceiling ends at its last whole row, never inside one,
    and the observation says the item continues."""

    from alphalattice.evidence.alternative_evidence.analysis.disclosures import _SCOPE_BYTES

    text = _ten_q(item2=REPURCHASE_ROWS)
    observation = _one(text, SALES)
    assert observation.state is TypedDisclosureState.NOT_FOUND
    assert observation.instances == () and observation.unrecognized == ()
    assert any("issuer repurchase disclosure present" in note for note in observation.context)
    assert observation.scope_range is not None
    assert not any("delivered scope is the item's first" in note for note in observation.context)
    # Padding sized so the 2,400-character cut lands inside the first row.
    for count in range(1, 60):
        padding = [
            f"Ordinary narrative about the use of proceeds, sentence {n:02d}." for n in range(count)
        ]
        long_text = _ten_q(item2=[*padding, *REPURCHASE_ROWS])
        first_row = long_text.index(REPURCHASE_ROWS[1])
        heading = long_text.index("ITEM 2. UNREGISTERED SALES")
        if first_row + 40 < heading + 2400 < first_row + len(REPURCHASE_ROWS[1]) - 40:
            break
    else:
        raise AssertionError("the fixture must cut into the first row")
    long_observation = _one(long_text, SALES)
    assert long_observation.scope_range is not None
    start, end = long_observation.scope_range
    footnote_end = long_text.index(REPURCHASE_ROWS[-1]) + len(REPURCHASE_ROWS[-1])
    assert start == heading and end == footnote_end, "rows and footnote delivered whole"
    assert len(long_text[start:end].encode("utf-8")) <= _SCOPE_BYTES
    assert long_text[end : end + 2] == "\n\n", "the scope ends at a paragraph boundary"
    # The whole item was delivered: nothing says it continues.
    assert not any("delivered scope is the item's first" in n for n in long_observation.context)
    # A table larger than the ceiling: the scope ends at a whole row.
    many = (
        [REPURCHASE_ROWS[0]]
        + [
            REPURCHASE_ROWS[1].replace("April", month)
            for month in (
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
                "Jan2",
                "Feb2",
            )
        ]
        + REPURCHASE_ROWS[3:]
    )
    big_text = _ten_q(item2=many)
    big_observation = _one(big_text, SALES)
    assert big_observation.scope_range is not None
    start, end = big_observation.scope_range
    assert len(big_text[start:end].encode("utf-8")) <= _SCOPE_BYTES
    assert big_text[end : end + 2] == "\n\n"
    delivered = big_text[start:end]
    assert delivered.count("Period: ") >= 4 and not delivered.rstrip().endswith("Period")
    assert big_text[end + 2 :].lstrip().startswith("Period: "), "ends before a whole row"
    assert any(
        "the item continues in the canonical text" in note for note in big_observation.context
    )


def test_actual_unregistered_sales_are_extracted_as_instances_with_fields() -> None:
    observation = _one(_ten_q(item2=[SALE, SALE_TWO, REPURCHASES]), SALES)
    assert observation.state is TypedDisclosureState.EXTRACTED
    first, second = observation.instances
    assert first.action == "SOLD_UNREGISTERED" and first.polarity == "AFFIRMATIVE"
    fields = _fields(first)
    assert fields["date"] == "May 12, 2026"
    assert fields["quantity"] == "40,000 shares"
    assert "Section 4(a)(2)" in (fields["exemption"] or "")
    assert "$1,200,000" in (fields["consideration"] or "")
    assert "Harborline Capital" in (fields["purchaser"] or "")
    assert first.period_end == parse_source_date("June 30, 2026")
    assert "consideration" in second.unknown_fields
    assert "Regulation D" in (_fields(second)["exemption"] or "")


def test_a_form_8k_reference_is_a_reference_requirement_not_a_sale_or_a_none() -> None:
    """requirement: cross-reference or exhibit missing."""

    observation = _one(_ten_q(item2=[REFERENCE_8K]), SALES)
    assert observation.state is TypedDisclosureState.REFERENCE_REQUIRED
    assert observation.references and "Form 8-K" in observation.references[0]
    assert observation.instances == ()


# --------------------------------------------------------------------------
# Family 3: trading arrangements.


def test_trading_arrangement_none_with_an_exception_yields_the_exception() -> None:
    """requirement: negation with an exception; no new plan actions versus
    an existing plan; the plan's own termination date is a term."""

    observation = _one(_ten_q(item5=[PLANS_NONE_EXCEPT, PLAN_ADOPTED]), PLANS)
    assert observation.state is TypedDisclosureState.EXTRACTED
    negation, adoption = observation.instances
    assert negation.polarity == "NONE_WITH_EXCEPTION"
    assert adoption.action == "ADOPTED" and adoption.polarity == "AFFIRMATIVE"
    fields = _fields(adoption)
    assert fields["person"] == "Jane Q. Rivera"
    assert fields["role"] == "Chief Financial Officer"
    assert fields["action_date"] == "2026-05-08"
    assert fields["arrangement_kind"] == "RULE_10B5_1"
    assert fields["shares_up_to"] == "12,000"
    assert fields["duration"] == "between August 15, 2026 and May 1, 2027"
    assert any("term of the plan, not a termination" in value for value in adoption.qualifiers)
    # A plain none says nothing about arrangements already in force.
    none = _one(_ten_q(item5=[PLANS_NONE]), PLANS)
    assert none.state is TypedDisclosureState.EXPLICIT_NONE
    assert none.instances[0].action == "NONE_STATED"
    assert _fields(none.instances[0])["actions_negated"] == "adopt+termi"
    arrangements = next(
        value
        for value in DEFINITIONS
        if value.family is TypedDisclosureFamily.INSIDER_TRADING_ARRANGEMENTS
    )
    assert "already in force" in arrangements.definition


def test_multiple_persons_actions_and_kinds_stay_multiple_and_the_issuer_is_not_a_person() -> None:
    """requirement: multiple directors/plans/actions; a termination and a
    non-Rule arrangement; the registrant's own plan is not a director's."""

    observation = _one(_ten_q(item5=[PLAN_ADOPTED, PLAN_TERMINATED, PLAN_ISSUER]), PLANS)
    assert observation.state is TypedDisclosureState.EXTRACTED
    adoption, termination = observation.instances
    assert _fields(termination)["person"] == "Mark T. Osei, Ph.D."
    assert termination.action == "TERMINATED"
    assert _fields(termination)["arrangement_kind"] == "NON_RULE_10B5_1"
    assert _fields(termination)["role"] == "member of our Board of Directors"
    assert _fields(adoption)["person"] == "Jane Q. Rivera"
    assert observation.unrecognized == (PLAN_ISSUER,)
    assert "1 relevant sentence(s) not recognised" in observation.reason
    conflicting = _one(_ten_q(item5=[PLANS_NONE, PLAN_ADOPTED]), PLANS)
    assert conflicting.state is TypedDisclosureState.AMBIGUOUS


def test_a_bulleted_list_of_arrangements_is_read_sentence_by_sentence() -> None:
    """requirement: NEE's Item 9B / Item 5 form -- a lead-in ("... adopted
    during the three months ended December 31, 2025 were as follows:") and
    one bullet per arrangement, each a dated statement behind a list glyph.
    Under rules v2 the glyph was the sentence's first word and every bullet
    was an unrecognised statement (state AMBIGUOUS, no instance). The glyph
    is layout; the lead-in stays visible as unrecognised."""

    lead_in = (
        "(c)\u00a0\u00a0\u00a0\u00a0Rule 10b5-1 trading arrangements terminated during the three "
        "months ended June\u00a030, 2026 were as follows:"
    )
    first = (
        "\u2022On April\u00a024,\u00a02026, Charles E. Sieving, Executive Vice President, Chief "
        "Legal, Environmental and Federal Regulatory Affairs Officer of NEE, terminated a "
        "Rule 10b5-1 trading arrangement that was intended to satisfy the affirmative "
        "defense of Rule 10b5-1(c) for the sale of 132,184 shares of NEE's common stock "
        "until December\u00a031,\u00a02026 that was originally entered into on January 27, 2026."
    )
    second = (
        "\u2022On April\u00a024,\u00a02026, John W. Ketchum, Chairman, President and Chief "
        "Executive Officer of NEE and Chairman of FPL, terminated a Rule 10b5-1 trading "
        "arrangement that was intended to satisfy the affirmative defense of Rule 10b5-1(c) "
        "for the sale of 88,605 shares of NEE's common stock until February\u00a017,\u00a02027 "
        "that was originally entered into on January 28, 2026."
    )
    observation = _one(_ten_q(item5=[lead_in, first, second]), PLANS)
    assert observation.state is TypedDisclosureState.EXTRACTED, observation.reason
    assert [value.action for value in observation.instances] == ["TERMINATED", "TERMINATED"]
    assert [_fields(v)["person"] for v in observation.instances] == [
        "Charles E. Sieving",
        "John W. Ketchum",
    ]
    assert {_fields(v)["action_date"] for v in observation.instances} == {"2026-04-24"}
    assert _fields(observation.instances[1])["duration"] == "until February 17, 2027"
    # Offsets are the source's: the range still starts at the glyph.
    text = _ten_q(item5=[lead_in, first, second])
    assert text[observation.instances[0].character_start] == "\u2022"
    assert len(observation.unrecognized) == 1 and "were as follows" in observation.unrecognized[0]


def test_an_amendment_of_limited_scope_refers_to_the_original_filing() -> None:
    """requirement: amendment with limited scope; absence of the item in an
    amendment is neither none nor not found."""

    text = _text(
        [
            *COVER_10K[:3],
            "FORM 10-K/A",
            "(Amendment No. 1)",
            "For the fiscal year ended December 31, 2025",
            "Indicate by check mark whether the registrant is a shell company. No",
            "EXPLANATORY NOTE",
            "The Company originally filed its Annual Report on Form 10-K for the year ended "
            "December 31, 2025 with the SEC on February 27, 2026 (the Original Form 10-K). "
            "This Amendment No. 1 on Form 10-K/A is being filed solely to provide the "
            "disclosure in Part II, Item 5 regarding the number of holders of record.",
            "PART II",
            "ITEM 5. MARKET FOR REGISTRANT'S COMMON EQUITY",
            "As of February 20, 2026, the number of holders of record was 1,125.",
        ]
    )
    observations = _observe(text, "10-K/A")
    assert observations[DCP].state is TypedDisclosureState.REFERENCE_REQUIRED
    assert "Form 10-K filed February 27, 2026" in observations[DCP].references[0]
    assert observations[PLANS].state is TypedDisclosureState.REFERENCE_REQUIRED
    assert observations[SALES].state is TypedDisclosureState.NOT_APPLICABLE
    assert observations[DCP].scope_range is not None
    unscoped = text.replace("is being filed solely to provide", "provides")
    assert _observe(unscoped, "10-K/A")[DCP].state is TypedDisclosureState.AMBIGUOUS


def test_part_decides_item_four_when_the_heading_carries_no_title() -> None:
    text = (
        _ten_q()
        .replace("ITEM 4. CONTROLS AND PROCEDURES", "ITEM 4.")
        .replace("ITEM 4. MINE SAFETY DISCLOSURES", "ITEM 4.")
    )
    observation = _one(text, DCP)
    assert observation.state is TypedDisclosureState.EXTRACTED
    assert observation.part == "PART I"


# --------------------------------------------------------------------------
# Explicit comparison with a selected prior observation.


def _sealed_record(
    tmp_path: Path, text: str, *, revision: str = "10-q-2026"
) -> TypedDisclosureRecord:
    """The sealed typed record the packet builds for one filing text, through
    the owning path (inspect, extract, issue, verified read): the contracts
    the report comparison actually receives, fields and all."""

    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path, documents=(_document(text=text, revision=revision),)
    )
    try:
        session = runtime.retrieval.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            record, _spans = packet_module.select_typed_disclosures(session=session)
        finally:
            session.close()
    finally:
        runtime.close()
    return record


def _later(text: str) -> str:
    """The same filing text as a later quarter's report."""

    return text.replace("June 30, 2026", "September 30, 2026")


def _compare(current: TypedDisclosureRecord, prior: TypedDisclosureRecord):
    rules = lambda record: (record.rules_id, record.definitions_hash)  # noqa: E731
    return {
        value.key: value
        for value in compare_typed_disclosures(
            current=current.observations,
            prior=prior.observations,
            current_rules=rules(current),
            prior_rules=rules(prior),
        )
    }


def test_prior_comparison_distinguishes_change_continuation_baseline_and_rule_moves(
    tmp_path: Path,
) -> None:
    """requirement: parser/version change versus a new event; disappearance
    is not resolution. On the sealed contracts the report receives."""

    prior = _sealed_record(
        tmp_path / "prior",
        _ten_q(item4=[EVALUATION, DCP_EFFECTIVE], item2=[NO_SALES], item5=[PLANS_NONE]),
        revision="10-q-2026-q1",
    )
    current = _sealed_record(
        tmp_path / "current",
        _later(
            _ten_q(
                item4=[EVALUATION, DCP_NOT_EFFECTIVE.replace("June 30", "September 30")],
                item5=[PLANS_NONE_EXCEPT, PLAN_ADOPTED],
            )
        ),
        revision="10-q-2026-q2",
    )
    changes = _compare(current, prior)
    assert changes[f"AAPL|{DCP}|10-Q|4"].kind == CHANGE_CHANGED
    assert "polarity EFFECTIVE -> NOT_EFFECTIVE" in " ".join(
        changes[f"AAPL|{DCP}|10-Q|4"].changed_fields
    )
    plans = changes[f"AAPL|{PLANS}|10-Q|5"]
    assert plans.kind == CHANGE_NEWLY_REPORTED
    assert any(
        "Jane Q. Rivera" in value and "newly stated" in value for value in plans.changed_fields
    )
    sales = changes[f"AAPL|{SALES}|10-Q|2"]
    assert sales.kind == CHANGE_NOT_COMPARABLE, "the later filing omits Item 2: not resolution"
    assert sales.current_state == "NOT_FOUND" and sales.prior_state == "EXPLICIT_NONE"
    # First use, the same revision, and a rule change.
    (baseline, *_rest) = [
        value
        for value in compare_typed_disclosures(
            current=current.observations,
            prior=(),
            current_rules=(current.rules_id, current.definitions_hash),
            prior_rules=(current.rules_id, current.definitions_hash),
        )
    ]
    assert baseline.kind == CHANGE_NEW_BASELINE
    same = _compare(prior, prior)
    assert {value.kind for value in same.values()} == {CHANGE_CONTINUING}
    moved = {
        value.key: value
        for value in compare_typed_disclosures(
            current=current.observations,
            prior=prior.observations,
            current_rules=(current.rules_id, current.definitions_hash),
            prior_rules=("alternative-evidence.typed-disclosures.v1", "a" * 64),
        )
    }
    assert {value.kind for value in moved.values()} == {CHANGE_RULE_OR_SOURCE}
    # A family rule identity that moved on one side is a rule change too,
    # whatever the record-level tuple says.
    rewired = tuple(
        value.model_copy(update={"rule_id": value.rule_id.replace(".v3", ".v1")})
        if value.family == PLANS
        else value
        for value in prior.observations
    )
    by_key = {
        value.key: value
        for value in compare_typed_disclosures(
            current=current.observations,
            prior=rewired,
            current_rules=(current.rules_id, current.definitions_hash),
            prior_rules=(current.rules_id, current.definitions_hash),
        )
    }
    assert by_key[f"AAPL|{PLANS}|10-Q|5"].kind == CHANGE_RULE_OR_SOURCE
    assert by_key[f"AAPL|{DCP}|10-Q|4"].kind == CHANGE_CHANGED
    # No current observation of a prior scope: not observed, not resolved.
    gone = _compare(prior, current)
    assert gone[f"AAPL|{SALES}|10-Q|2"].kind == CHANGE_NOT_COMPARABLE
    only_prior = {
        value.key: value
        for value in compare_typed_disclosures(
            current=tuple(v for v in current.observations if v.family != SALES),
            prior=prior.observations,
            current_rules=(current.rules_id, current.definitions_hash),
            prior_rules=(prior.rules_id, prior.definitions_hash),
        )
    }
    assert only_prior[f"AAPL|{SALES}|10-Q|2"].kind == CHANGE_NOT_OBSERVED
    assert "not thereby resolved" in only_prior[f"AAPL|{SALES}|10-Q|2"].detail


def test_comparison_reads_material_fields_not_only_event_labels(tmp_path: Path) -> None:
    """R3: the same person, action and date restated from 12,000 to 20,000
    shares is not a continuing assertion; a changed duration or qualifier
    is a change; an unchanged restatement continues; a re-reported period
    is a correction, never a newly occurring event."""

    plan_12k = _ten_q(item5=[PLANS_NONE_EXCEPT, PLAN_ADOPTED])
    prior = _sealed_record(tmp_path / "prior", plan_12k, revision="10-q-2026-q2")
    key = f"AAPL|{PLANS}|10-Q|5"
    # The same period re-reported (an amendment, a re-acquired revision)
    # with a corrected share count: a changed assertion, not a new action.
    corrected = _sealed_record(
        tmp_path / "corrected",
        plan_12k.replace("12,000 shares", "20,000 shares"),
        revision="10-q-2026-q2-a",
    )
    change = _compare(corrected, prior)[key]
    assert change.kind == CHANGE_CHANGED
    assert change.changed_fields == (
        "Jane Q. Rivera (Chief Financial Officer)|ADOPTED|2026-05-08: "
        "shares_up_to 12,000 -> 20,000",
    )
    assert "correction" in change.detail and "not a new event" in change.detail
    # A later filing restating the same action with a different quantity.
    later_20k = _sealed_record(
        tmp_path / "later20k",
        _later(plan_12k.replace("12,000 shares", "20,000 shares")),
        revision="10-q-2026-q3",
    )
    restated = _compare(later_20k, prior)[key]
    assert restated.kind == CHANGE_CHANGED and "restated with changed content" in restated.detail
    assert restated.changed_fields == change.changed_fields
    # A changed duration and a changed qualifier are changes too.
    later_duration = _sealed_record(
        tmp_path / "duration",
        _later(plan_12k.replace("May 1, 2027", "December 15, 2027")),
        revision="10-q-2026-q3",
    )
    duration = _compare(later_duration, prior)[key]
    assert duration.kind == CHANGE_CHANGED
    assert any(
        "duration between August 15, 2026 and May 1, 2027 -> " in v for v in duration.changed_fields
    )
    # The plan's stated termination term is a qualifier; a restatement
    # without it changes the assertion's qualification.
    later_unqualified = _sealed_record(
        tmp_path / "unqualified",
        _later(plan_12k.replace(PLAN_ADOPTED, PLAN_ADOPTED.split(". Ms. Rivera")[0] + ".")),
        revision="10-q-2026-q3",
    )
    unqualified = _compare(later_unqualified, prior)[key]
    assert unqualified.kind == CHANGE_CHANGED
    assert any(
        "qualifiers the stated termination is a term of the plan, not a termination action -> -"
        in v
        for v in unqualified.changed_fields
    )
    # A genuinely unchanged restatement in a later filing continues.
    later_same = _sealed_record(tmp_path / "same", _later(plan_12k), revision="10-q-2026-q3")
    continuing = _compare(later_same, prior)[key]
    assert continuing.kind == CHANGE_CONTINUING and continuing.changed_fields == ()
    # A new person in a later period is newly reported; the same person
    # added to a re-reported period is a change of that period's statement.
    later_new = _sealed_record(
        tmp_path / "new",
        _later(_ten_q(item5=[PLANS_NONE_EXCEPT, PLAN_ADOPTED, PLAN_TERMINATED])),
        revision="10-q-2026-q3",
    )
    newly = _compare(later_new, prior)[key]
    assert newly.kind == CHANGE_NEWLY_REPORTED
    assert [v for v in newly.changed_fields if "newly stated" in v] == [
        "Mark T. Osei, Ph.D. (member of our Board of Directors)|TERMINATED|2026-06-02: newly stated"
    ]
    amended = _sealed_record(
        tmp_path / "amended",
        _ten_q(item5=[PLANS_NONE_EXCEPT, PLAN_ADOPTED, PLAN_TERMINATED]),
        revision="10-q-2026-q2-a",
    )
    amendment = _compare(amended, prior)[key]
    assert amendment.kind == CHANGE_CHANGED and "1 newly stated" in amendment.detail
    # Same bytes, different observation: the parse moved, not the filing.
    parsed = _compare(
        corrected.model_copy(
            update={
                "observations": tuple(
                    v.model_copy(update={"revision_label": prior.observations[0].revision_label})
                    for v in corrected.observations
                )
            }
        ),
        prior,
    )[key]
    assert parsed.kind == CHANGE_RULE_OR_SOURCE and parsed.changed_fields == change.changed_fields


def test_two_same_day_sales_stay_two_assertions_and_a_change_to_one_cannot_disappear(
    tmp_path: Path,
) -> None:
    """R3 (remaining): two distinct sales on one day share subject, action
    and date; a change to the first must not be hidden by the second, a
    genuinely identical repetition folds to one, and when the source does
    not say which changed assertion corresponds to which, the comparison
    says so instead of pairing by guess."""

    same_day = (
        "On May 12, 2026, we issued 5,000 shares of common stock to a consultant as "
        "compensation for services, exempt from registration under Regulation D."
    )
    key = f"AAPL|{SALES}|10-Q|2"
    prior = _sealed_record(
        tmp_path / "prior", _ten_q(item2=[SALE, same_day]), revision="10-q-2026-q2"
    )
    sales = next(v for v in prior.observations if v.family == SALES)
    assert [_fields(v)["date"] for v in sales.instances] == ["May 12, 2026", "May 12, 2026"]
    corrected = _sealed_record(
        tmp_path / "corrected",
        _ten_q(item2=[SALE.replace("40,000 shares", "45,000 shares"), same_day]),
        revision="10-q-2026-q2-a",
    )
    change = _compare(corrected, prior)[key]
    assert change.kind == CHANGE_CHANGED
    assert change.changed_fields == (
        "sale of unregistered equity securities|SOLD_UNREGISTERED|May 12, 2026: "
        "quantity 40,000 shares -> 45,000 shares",
    )
    assert "consultant" not in " ".join(change.changed_fields), "the unchanged sale is silent"
    # The same two sales restated in the other order continue.
    reordered = _sealed_record(
        tmp_path / "reordered", _ten_q(item2=[same_day, SALE]), revision="10-q-2026-q2-b"
    )
    assert _compare(reordered, prior)[key].kind == CHANGE_CONTINUING
    # The same sale stated twice is one assertion, not a newly stated one.
    repeated = _sealed_record(
        tmp_path / "repeated", _ten_q(item2=[SALE, same_day, SALE]), revision="10-q-2026-q2-c"
    )
    assert _compare(repeated, prior)[key].kind == CHANGE_CONTINUING
    # Both same-day sales changed: no correspondence is invented.
    both = _sealed_record(
        tmp_path / "both",
        _ten_q(
            item2=[
                SALE.replace("40,000 shares", "45,000 shares"),
                same_day.replace("5,000 shares", "6,000 shares"),
            ]
        ),
        revision="10-q-2026-q2-d",
    )
    unresolved = _compare(both, prior)[key]
    assert unresolved.kind == CHANGE_CHANGED
    (entry,) = unresolved.changed_fields
    assert entry.startswith(
        "sale of unregistered equity securities|SOLD_UNREGISTERED|May 12, 2026: "
        "UNRESOLVED CORRESPONDENCE -- 2 prior and 2 current assertions"
    )
    # A third sale on the same day in a later filing is newly stated, not a
    # change to either existing one.
    third = _sealed_record(
        tmp_path / "third",
        _later(
            _ten_q(
                item2=[
                    SALE,
                    same_day,
                    "On May 12, 2026, we issued 800 shares of common stock to Ridgeline LLC "
                    "in a private placement exempt under Section 4(a)(2) of the Securities Act.",
                ]
            )
        ),
        revision="10-q-2026-q3",
    )
    newly = _compare(third, prior)[key]
    assert newly.kind == CHANGE_NEWLY_REPORTED
    assert newly.changed_fields == (
        "sale of unregistered equity securities|SOLD_UNREGISTERED|May 12, 2026: newly stated",
    )


# --------------------------------------------------------------------------
# The owning path: verified spans, the receipt, the packet and the budget.


def _document(
    entity_id: str = "AAPL", *, text: str, form: str = "10-Q", revision: str | None = None
) -> RecordedEvidenceDocument:
    base = _recorded_document(entity_id)
    return base.model_copy(
        update={
            "text": text,
            "document_type": form,
            "revision": revision if revision is not None else f"{form}-2026".lower(),
        }
    )


def test_typed_spans_are_verified_reads_and_the_receipt_accounts_for_them(tmp_path: Path) -> None:
    """requirement: source/tamper checks; the same verified reader; the
    receipt's own accounting; the rendered packet and the finding read."""

    text = _ten_q(item2=[SALE, REPURCHASES], item5=[PLANS_NONE_EXCEPT, PLAN_ADOPTED])
    runtime, request, _registry, snapshot, document_set, generation = _open_recorded(
        tmp_path, documents=(_document(text=text),)
    )
    try:
        receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        typed = receipt.typed_disclosures
        assert isinstance(typed, TypedDisclosureRecord)
        assert typed.definitions_hash == definitions_hash()
        assert typed.read_call_count >= 1 and typed.span_handles
        assert set(typed.span_handles).isdisjoint(receipt.read_span_handles)
        assert set(receipt.delivered_span_handles) == {value.span_handle for value in spans}
        by_family = {value.family: value for value in typed.observations}
        assert by_family[DCP].state == "EXTRACTED"
        assert by_family[SALES].state == "EXTRACTED"
        assert by_family[PLANS].state == "EXTRACTED"
        sale = by_family[SALES].instances[0]
        assert sale.span_handle is not None and sale.span_handle.startswith("SPAN-T01-R")
        span = next(value for value in spans if value.span_handle == sale.span_handle)
        assert SALE in " ".join(span.excerpt.split())
        assert (
            span.character_start <= sale.character_start < sale.character_end <= span.character_end
        )
        # The packet names the family that bound each typed span and carries
        # the accounting block; every part of a delivery keeps the block.
        packet = packet_module.AlternativeEvidencePacket(
            request=request,
            obligation=_obligation(request),
            snapshot=snapshot,
            document_set=document_set,
            receipt=receipt,
            spans=spans,
        )
        rendered = packet_module.render_evidence_packet(packet)
        assert f"T01:TYPED:{SALES}:EXTRACTED" in rendered
        assert '"typed_disclosures"' in rendered and '"SOLD_UNREGISTERED"' in rendered
        part = packet_module.render_evidence_packet(
            packet, span_handles=(sale.span_handle,), delivery_part=(1, 2)
        )
        assert '"typed_disclosures"' in part and '"SOLD_UNREGISTERED"' in part
        assert "Jane Q. Rivera" not in part, "the plan's span belongs to another part"
        assert "Jane Q. Rivera" in rendered
        found_by, _facets = packet_module.span_provenance(receipt)
        assert found_by[sale.span_handle] == (f"T01:TYPED:{SALES}:EXTRACTED",)
        # The typed span refuses a tampered blob like any other span.
        session = runtime.retrieval.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            session.inspect_documents()
            (handle,) = session.issue_source_spans(
                document_set.documents[0].workspace_document_id,
                ((text.index(SALE), text.index(SALE) + len(SALE)),),
            )
            (read,) = session.read_typed_spans(span_handles=(handle,))
            assert SALE in " ".join(read.excerpt.split())
            reference = document_set.documents[0]
            blob = next(
                path
                for path in (tmp_path / "workspace" / "knowledge" / "blobs").rglob("*")
                if path.is_file() and path.stat().st_size == reference.byte_count
            )
            original = blob.read_bytes()
            blob.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
            try:
                with pytest.raises((ValueError, KnowledgeRetrievalError)):
                    session.read_typed_spans(span_handles=(handle,))
            finally:
                blob.write_bytes(original)
            with pytest.raises(ValueError, match="span_character_range_invalid"):
                session.issue_source_spans(reference.workspace_document_id, ((5, 2),))
            with pytest.raises(ValueError, match="document_not_inspected"):
                session.issue_source_spans("unknown-document", ((0, 5),))
            with pytest.raises(ValueError, match="span_read_request_invalid"):
                session.read_typed_spans(span_handles=("SPAN-S01-R01",))
        finally:
            session.close()
    finally:
        runtime.close()
    # A receipt that cites a typed span it did not read refuses by name.
    with pytest.raises(ValueError, match="typed_disclosure_span_unread"):
        AlternativeEvidenceRetrievalAccessReceipt.model_validate(
            {
                **receipt.model_dump(mode="json", exclude={"receipt_hash"}),
                "typed_disclosures": {
                    **receipt.model_dump(mode="json")["typed_disclosures"],
                    "span_handles": [],
                },
                "receipt_hash": "0" * 64,
            }
        )


def _scope_delivery(tmp_path: Path, item2: list[str]) -> tuple[Any, Any, str]:
    """The sales family's delivered scope span through the runtime's own
    selection and the shared verified reader: the observation, the resolved
    span (None when no scope span was issued) and the canonical text."""

    text = _ten_q(item2=item2, item5=[PLAN_ADOPTED, PLAN_TERMINATED])
    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path, documents=(_document(text=text),)
    )
    try:
        receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
    finally:
        runtime.close()
    typed = receipt.typed_disclosures
    assert isinstance(typed, TypedDisclosureRecord)
    observation = next(value for value in typed.observations if value.family == SALES)
    if observation.scope_span_handle is not None:
        assert len(typed.span_handles) == 4, "four typed spans: one read, 4,096 bytes each"
    span = None
    if observation.scope_span_handle is not None:
        span = next(v for v in spans if v.span_handle == observation.scope_span_handle)
    return observation, span, text


def test_the_delivered_scope_respects_the_reader_byte_ceiling_on_every_path(
    tmp_path: Path,
) -> None:
    """requirement: the delivered scope is a range the shared reader returns
    whole. Three paths cut it: a region shorter than 2,400 characters whose
    UTF-8 bytes exceed the reader's per-span ceiling was returned unchecked
    (the reader then bounded it mid-row with a limitation); a paragraph
    crossing the 2,400th character whose end did not fit fell back to the
    raw character cut inside the row; and a single row longer than the
    ceiling was delivered as a fragment. Synthetic shapes, labelled: the
    dash rule line, the long row and the oversized row are built to the
    ceiling, not taken from a filing."""

    from alphalattice.evidence.alternative_evidence.analysis.disclosures import _SCOPE_BYTES

    lead = (
        "The following table sets forth information relating to repurchases of our equity "
        "securities during the three months ended June 30, 2026:"
    )
    rows = [REPURCHASE_ROWS[1], REPURCHASE_ROWS[2]]

    # 1. Short in characters, long in bytes: a rule line of em dashes.
    rule = "\u2014" * 1200
    observation, span, text = _scope_delivery(
        tmp_path / "short", [lead, *rows, rule, REPURCHASE_ROWS[-1]]
    )
    assert observation.scope_span_handle is not None and span is not None
    start, end = span.character_start, span.character_end
    assert not any("bounded" in v for v in span.limitations), "the reader returned the range whole"
    assert len(text[start:end].encode("utf-8")) <= _SCOPE_BYTES
    assert span.excerpt.count("Period: ") == 2, "both rows, whole, with their headings"
    assert rule not in span.excerpt and text[end - 2 : end] != "\u2014\u2014"
    assert any("not delivered" in note or "continues" in note for note in observation.context)

    # 2. A row that crosses the 2,400th character and does not fit whole.
    long_lead = " ".join(
        f"Narrative sentence {n:02d} about the use of proceeds." for n in range(46)
    )
    assert 2_200 < len(long_lead) < 2_400
    long_row = REPURCHASE_ROWS[1] + "; Notes: " + "detail " * 260
    observation, span, text = _scope_delivery(
        tmp_path / "long", [long_lead, long_row, REPURCHASE_ROWS[-1]]
    )
    assert span is not None
    assert span.excerpt.rstrip().endswith(long_lead.rstrip()), "the scope ends at the whole lead"
    assert "Period: " not in span.excerpt, "never a fragment of the row that did not fit"
    assert any("not delivered" in note for note in observation.context), observation.context

    # 3. The only unit is larger than the ceiling: no fragment, a named gap.
    huge_row = REPURCHASE_ROWS[1] + "; Notes: " + "detail " * 600
    observation, span, text = _scope_delivery(tmp_path / "huge", [huge_row])
    assert span is None and observation.scope_span_handle is None, "no misleading fragment"
    assert observation.state is not None
    assert any("not delivered" in note for note in observation.context), observation.context


def test_instances_beyond_the_typed_span_budget_stay_recorded_not_dropped(tmp_path: Path) -> None:
    """requirement: multi-instance overflow with no silent truncation."""

    plans = [
        f"On May {day}, 2026, Person{day} Q. Holder, our Vice President, adopted a Rule 10b5-1 "
        f"trading plan for the sale of up to {day * 100} shares of common stock until May 1, 2027."
        for day in range(1, 31)
    ]
    documents = tuple(
        _document(entity, text=_ten_q(item2=[NO_SALES], item5=plans))
        for entity in ("AAPL", "MSFT", "NVDA")
    )
    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path, entities=("AAPL", "MSFT", "NVDA"), documents=documents
    )
    try:
        session = runtime.retrieval.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            record, spans = packet_module.select_typed_disclosures(session=session)
        finally:
            session.close()
        assert len(record.span_handles) == MAXIMUM_ISSUED_SOURCE_SPANS == len(spans)
        plans_observations = [value for value in record.observations if value.family == PLANS]
        assert sum(len(value.instances) for value in plans_observations) == 90
        undelivered = sum(value.undelivered_instance_count for value in plans_observations)
        assert undelivered > 0
        unhandled = [
            instance
            for value in plans_observations
            for instance in value.instances
            if instance.span_handle is None
        ]
        assert len(unhandled) == undelivered
        assert all(instance.statement_text and instance.character_end > 0 for instance in unhandled)
    finally:
        runtime.close()


# --------------------------------------------------------------------------
# R1: the two read budgets are admitted separately and resolved by one reader.


def test_generic_and_typed_read_budgets_are_separate_under_one_verified_reader(
    tmp_path: Path,
) -> None:
    """R1: the generic thirty-two reads do not consume or block the typed
    families' sixteen, typed exhaustion leaves the generic accounting
    untouched, and a tampered source refuses through the typed path -- in
    the order a session is used: generic reads first, then typed selection
    in the same session."""

    text = _ten_q(item2=[SALE, REPURCHASES], item5=[PLANS_NONE_EXCEPT, PLAN_ADOPTED])
    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path, documents=(_document(text=text),)
    )
    try:
        session = runtime.retrieval.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            # A search issues generic spans; spending the generic budget on
            # one of them is admitted under the generic gate alone.
            handle = session.search(query="securities sold repurchased").hits[0].span_handle
            while session.span_read_count < MAXIMUM_SPAN_READS:
                session.read_spans(span_handles=(handle,))
            with pytest.raises(ValueError, match="span_read_budget_exhausted"):
                session.read_spans(span_handles=(handle,))
            assert session.typed_span_read_count == 0
            record, typed_spans = packet_module.select_typed_disclosures(session=session)
            assert record.read_call_count >= 1 and len(typed_spans) == len(record.span_handles)
            assert session.span_read_count == MAXIMUM_SPAN_READS, "typed reads are not generic"
            # v4: the concentration family reads a 10-Q too (its segment note is
            # absent here, so it is observed NOT_FOUND); the 10-K-only cyber family
            # is not sealed for a 10-Q.
            assert {value.family for value in record.observations} == {
                DCP,
                SALES,
                PLANS,
                CONCENTRATION,
            }
            # Typed exhaustion: the sixteenth read is the last; the program's
            # count neither moves nor is consulted.
            reference = document_set.documents[0]
            (extra,) = session.issue_source_spans(
                reference.workspace_document_id,
                ((text.index(SALE), text.index(SALE) + len(SALE)),),
            )
            while session.typed_span_read_count < MAXIMUM_TYPED_SPAN_READS:
                session.read_typed_spans(span_handles=(extra,))
            with pytest.raises(ValueError, match="typed_span_read_budget_exhausted"):
                session.read_typed_spans(span_handles=(extra,))
            assert session.span_read_count == MAXIMUM_SPAN_READS
            with pytest.raises(ValueError, match="span_read_budget_exhausted"):
                session.read_spans(span_handles=(handle,))
        finally:
            session.close()
        # A fresh session: the generic budget spent first, then the typed
        # read of a tampered source refuses by name, not by budget.
        session = runtime.retrieval.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            generic = session.search(query="securities sold repurchased").hits[0].span_handle
            while session.span_read_count < MAXIMUM_SPAN_READS:
                session.read_spans(span_handles=(generic,))
            session.inspect_documents()
            reference = document_set.documents[0]
            (handle,) = session.issue_source_spans(
                reference.workspace_document_id,
                ((text.index(SALE), text.index(SALE) + len(SALE)),),
            )
            blob = next(
                path
                for path in (tmp_path / "workspace" / "knowledge" / "blobs").rglob("*")
                if path.is_file() and path.stat().st_size == reference.byte_count
            )
            original = blob.read_bytes()
            blob.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
            try:
                with pytest.raises((ValueError, KnowledgeRetrievalError)) as refused:
                    session.read_typed_spans(span_handles=(handle,))
                assert "budget" not in str(refused.value)
            finally:
                blob.write_bytes(original)
            (read,) = session.read_typed_spans(span_handles=(handle,))
            assert SALE in " ".join(read.excerpt.split())
            with pytest.raises(ValueError, match="span_read_request_invalid"):
                session.read_typed_spans(span_handles=(handle, handle))
        finally:
            session.close()
    finally:
        runtime.close()


# --------------------------------------------------------------------------
# R2: negation and exceptions keep their scope; nothing becomes a fact or a
# blanket none that the source does not state.

NOT_CONCLUDED = (
    "Our management has not concluded that our disclosure controls and procedures were "
    "effective as of June 30, 2026."
)
UNABLE_TO_CONCLUDE = (
    "Because the evaluation is not complete, our principal executive officer and principal "
    "financial officer were unable to conclude that our disclosure controls and procedures "
    "were effective as of June 30, 2026."
)
WOULD_HAVE_CONCLUDED = (
    "Had the remediation been completed, our principal executive officer and principal "
    "financial officer would have concluded that our disclosure controls and procedures "
    "were effective as of June 30, 2026."
)
NO_SALES_EXCEPT_SALE = (
    "There were no unregistered sales of equity securities during the quarter ended June 30, "
    "2026, except for the 40,000 shares issued on May 12, 2026 in a private placement."
)
NO_SALES_EXCEPT_UNREAD = (
    "There were no unregistered sales of equity securities during the quarter ended June 30, "
    "2026, other than issuances under our registered equity incentive plans."
)
NO_SALES_EXCEPT_BELOW = (
    "Except as set forth below, there were no unregistered sales of equity securities during "
    "the quarter ended June 30, 2026."
)
PLANS_NONE_EXCEPT_INLINE = (
    "During the quarter ended June 30, 2026, no director or officer of the Company adopted "
    "or terminated a Rule 10b5-1 trading arrangement, except that on May 8, 2026, Jane Q. "
    "Rivera, our Chief Financial Officer, adopted a Rule 10b5-1 trading plan for the sale of "
    "up to 12,000 shares of common stock until May 1, 2027."
)
PLANS_NONE_BUT_ISSUER = (
    "During the quarter ended June 30, 2026, no director or officer of the Company adopted "
    "or terminated a Rule 10b5-1 trading arrangement, but the Company adopted a Rule 10b5-1 "
    "trading plan to repurchase shares under its repurchase program."
)


def test_a_negated_or_conditional_conclusion_states_no_conclusion() -> None:
    """R2: "has not concluded" is neither EFFECTIVE nor NOT_EFFECTIVE; the
    sentence stays visible as the reason; the affirmative and ordinary
    negative conclusions are unchanged."""

    for statement in (NOT_CONCLUDED, UNABLE_TO_CONCLUDE, WOULD_HAVE_CONCLUDED):
        observation = _one(_ten_q(item4=[EVALUATION, statement]), DCP)
        assert observation.state is TypedDisclosureState.AMBIGUOUS, statement
        assert observation.instances == ()
        assert observation.unrecognized == ("NO_CONCLUSION: " + statement,)
        assert "no conclusion was reached" in observation.reason
    # Beside a prior period's conclusion the current period is still unresolved.
    observation = _one(_ten_q(item4=[EVALUATION, DCP_PRIOR, NOT_CONCLUDED]), DCP)
    assert observation.state is TypedDisclosureState.AMBIGUOUS
    assert "states that none was reached" in observation.reason
    assert [value.polarity for value in observation.instances] == ["EFFECTIVE"]
    # Controls: the ordinary conclusions read as before.
    assert _one(_ten_q(), DCP).instances[0].polarity == "EFFECTIVE"
    negative = _one(_ten_q(item4=[EVALUATION, DCP_NOT_EFFECTIVE]), DCP)
    assert negative.state is TypedDisclosureState.EXTRACTED
    assert negative.instances[0].polarity == "NOT_EFFECTIVE"


def test_a_sales_none_with_an_exception_keeps_the_exception(tmp_path: Path) -> None:
    """R2: the exception to a none is read by the sale rule when it states a
    sale, stays visible and unresolved when it does not, and never becomes
    an unqualified EXPLICIT_NONE -- through extract_document and through the
    packet boundary the analyst reads."""

    observation = _one(_ten_q(item2=[NO_SALES_EXCEPT_SALE]), SALES)
    assert observation.state is TypedDisclosureState.EXTRACTED
    none, sale = observation.instances
    assert none.polarity == "NONE_WITH_EXCEPTION" and none.action == "NONE_STATED_WITH_EXCEPTION"
    assert none.qualifiers == (
        "negation with an exception: except for the 40,000 shares issued on May 12, 2026 in a "
        "private placement.",
    )
    assert sale.action == "SOLD_UNREGISTERED" and sale.polarity == "AFFIRMATIVE"
    assert _fields(sale) == {
        "date": "May 12, 2026",
        "quantity": "40,000 shares",
        "exemption": "private placement",
    }
    assert sale.unknown_fields == ("consideration", "purchaser")
    assert sale.period_end == parse_source_date("June 30, 2026")
    assert sale.statement_text == "the 40,000 shares issued on May 12, 2026 in a private placement"
    assert none.character_start < sale.character_start < sale.character_end <= none.character_end
    # An exception no rule reads: unresolved, with the exception visible.
    unread = _one(_ten_q(item2=[NO_SALES_EXCEPT_UNREAD]), SALES)
    assert unread.state is TypedDisclosureState.AMBIGUOUS
    assert [value.polarity for value in unread.instances] == ["NONE_WITH_EXCEPTION"]
    assert unread.unrecognized == (
        "EXCEPTION: issuances under our registered equity incentive plans",
    )
    assert "except as stated" in unread.reason and "requires reading" in unread.reason
    # A forward exception is the statements that follow; without them it is
    # unresolved, never a none.
    below = _one(_ten_q(item2=[NO_SALES_EXCEPT_BELOW, SALE_TWO]), SALES)
    assert below.state is TypedDisclosureState.EXTRACTED
    assert [value.action for value in below.instances] == [
        "NONE_STATED_WITH_EXCEPTION",
        "SOLD_UNREGISTERED",
    ]
    assert _fields(below.instances[1])["date"] == "June 3, 2026"
    alone = _one(_ten_q(item2=[NO_SALES_EXCEPT_BELOW]), SALES)
    assert alone.state is TypedDisclosureState.AMBIGUOUS
    # Controls: an ordinary none and a none beside a repurchase stay none.
    assert _one(_ten_q(item2=[NO_SALES]), SALES).state is TypedDisclosureState.EXPLICIT_NONE
    both = _one(
        _ten_q(
            item2=[
                "We did not sell any unregistered equity securities during the three months "
                "ended June 30, 2026, nor did we repurchase any shares of our common stock."
            ]
        ),
        SALES,
    )
    assert both.state is TypedDisclosureState.EXPLICIT_NONE
    # The period boundary date is not the sale date.
    dated = _one(
        _ten_q(
            item2=[
                "During the quarter ended June 30, 2026, we issued 5,000 shares of common stock "
                "to a consultant on May 3, 2026, exempt from registration under Regulation D."
            ]
        ),
        SALES,
    )
    assert _fields(dated.instances[0])["date"] == "May 3, 2026"
    # Through the packet: the sealed observation carries both assertions
    # with verified spans, and the rendered packet shows the exception.
    runtime, request, _registry, snapshot, document_set, generation = _open_recorded(
        tmp_path, documents=(_document(text=_ten_q(item2=[NO_SALES_EXCEPT_SALE])),)
    )
    try:
        receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        assert receipt.typed_disclosures is not None
        sealed = next(v for v in receipt.typed_disclosures.observations if v.family == SALES)
        assert sealed.state == "EXTRACTED"
        assert [value.polarity for value in sealed.instances] == [
            "NONE_WITH_EXCEPTION",
            "AFFIRMATIVE",
        ]
        assert all(
            value.span_handle in receipt.delivered_span_handles for value in sealed.instances
        )
        rendered = packet_module.render_evidence_packet(
            packet_module.AlternativeEvidencePacket(
                request=request,
                obligation=_obligation(request),
                snapshot=snapshot,
                document_set=document_set,
                receipt=receipt,
                spans=spans,
            ),
        )
        assert '"NONE_WITH_EXCEPTION"' in rendered and '"SOLD_UNREGISTERED"' in rendered
        assert "negation with an exception" in rendered
    finally:
        runtime.close()


def test_an_arrangement_none_with_an_exception_keeps_the_exception() -> None:
    """R2: an inline exception naming a person's action is that action; an
    exception the rule cannot read, or one that points to nothing, leaves
    the family unresolved instead of an extracted nothing."""

    inline = _one(_ten_q(item5=[PLANS_NONE_EXCEPT_INLINE]), PLANS)
    assert inline.state is TypedDisclosureState.EXTRACTED
    none, adopted = inline.instances
    assert none.polarity == "NONE_WITH_EXCEPTION"
    assert adopted.subject == "Jane Q. Rivera (Chief Financial Officer)"
    assert adopted.action == "ADOPTED"
    assert _fields(adopted) == {
        "person": "Jane Q. Rivera",
        "role": "Chief Financial Officer",
        "action_date": "2026-05-08",
        "arrangement_kind": "RULE_10B5_1",
        "shares_up_to": "12,000",
        "duration": "until May 1, 2027",
    }
    assert none.character_start < adopted.character_start < adopted.character_end
    issuer = _one(_ten_q(item5=[PLANS_NONE_BUT_ISSUER]), PLANS)
    assert issuer.state is TypedDisclosureState.AMBIGUOUS
    assert issuer.unrecognized == (
        "EXCEPTION: the Company adopted a Rule 10b5-1 trading plan to repurchase shares under "
        "its repurchase program",
    )
    forward_only = _one(_ten_q(item5=[PLANS_NONE_EXCEPT]), PLANS)
    assert forward_only.state is TypedDisclosureState.AMBIGUOUS
    assert forward_only.instances[0].polarity == "NONE_WITH_EXCEPTION"
    # Controls: the ordinary none and the none-then-instance shapes hold.
    assert _one(_ten_q(item5=[PLANS_NONE]), PLANS).state is TypedDisclosureState.EXPLICIT_NONE
    listed = _one(_ten_q(item5=[PLANS_NONE_EXCEPT, PLAN_ADOPTED]), PLANS)
    assert listed.state is TypedDisclosureState.EXTRACTED and len(listed.instances) == 2


# --------------------------------------------------------------------------
# Families 4 and 5 (v4): customer concentration and the cybersecurity effect.

ONE_BOTTLER = (
    "For the year ended December 31, 2025, one bottler accounted for 10% of our net "
    "operating revenues, which are reflected in our EMEA and Asia Pacific operating segments."
)
NO_BOTTLERS = (
    "No bottlers or customers represented 10% or more of our net operating revenues for the "
    "years ended December 31, 2024 and 2023."
)
DIRECT_CUSTOMERS = (
    "For fiscal year 2026, sales to one direct customer represented 22% of total revenue and "
    "sales to another direct customer represented 14% of total revenue, all of which were "
    "primarily attributable to the Compute & Networking segment."
)
GEOGRAPHIC = (
    "Revenue from sales to customers headquartered outside of the United States accounted "
    "for 31%, 41%, and 48% of total revenue for fiscal years 2026, 2025, and 2024, "
    "respectively."
)
NO_RELIANCE = (
    "We do not rely on any major customers as a source of sales, and the customers and "
    "long-lived assets of our reportable segments are predominantly in the U.S."
)
NO_US_REVENUE = (
    "There were no revenues from external customers in the U.S. for the year ended "
    "December 31, 2023."
)
CYBER_NOT_AFFECTED = (
    "Cybersecurity threats, including as a result of previous cybersecurity incidents, have "
    "not materially affected the Company during the past three fiscal years."
)
CYBER_NOT_AWARE = (
    "Based on the information available as of the date of this Form 10-K, we are not aware "
    "of any risks from actual cybersecurity incidents that have materially affected us or "
    "are reasonably likely to materially affect us, including our business strategy, "
    "results of operations or financial condition."
)
CYBER_FORWARD = (
    "A cybersecurity incident could materially affect our business strategy, results of "
    "operations or financial condition."
)
CYBER_AFFECTED = (
    "During the fourth quarter of 2025, a cybersecurity incident materially affected our "
    "results of operations, except as described in Item 7."
)
CYBER_PROGRAM = (
    "We have designated a Chief Information Security Officer to oversee the identification, "
    "assessment and management of material cybersecurity risks."
)


def _ten_k(*, item1c: list[str] | None = None, note: list[str] | None = None) -> str:
    lines = [
        *COVER_10K,
        "PART I",
        "ITEM 1. BUSINESS",
        "The Company sells beverages through bottling partners.",
        "ITEM 1A. RISK FACTORS",
        "Our business is subject to risks.",
    ]
    if item1c is not None:
        lines += ["ITEM 1C. CYBERSECURITY", *item1c]
    lines += [
        "PART II",
        "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        "NOTE 1: BASIS OF PRESENTATION",
        "The consolidated financial statements include the accounts of the Company.",
    ]
    if note is not None:
        lines += ["NOTE 20: OPERATING SEGMENTS", *note]
    lines += [
        "NOTE 21: SUBSEQUENT EVENTS",
        "None.",
        "ITEM 9A. CONTROLS AND PROCEDURES",
        EVALUATION,
        DCP_EFFECTIVE,
        "ITEM 9B. OTHER INFORMATION",
        "None.",
        "SIGNATURES",
    ]
    return _text(lines)


def test_customer_concentration_reads_the_segment_note_s_stated_shares_and_absences() -> None:
    """requirement (W2, typed rules v4; the three traced losses): in the
    segment note a counted customer beside a percentage of revenue is a
    `STATED` share with the shares and the base as fields; an absence at a
    threshold or of reliance is `ABSENT` with the threshold as its field;
    a share of customers by geography is kept visible as unrecognised and
    "no revenues from external customers in the U.S." is not read; the
    stated and absent instances coexist as assertions (a none scoped to a
    threshold contradicts no share); with no segment, concentration or
    customer note the family is NOT_FOUND, never a none; nothing names a
    customer or sums a share."""

    text = _ten_k(
        note=[ONE_BOTTLER, NO_BOTTLERS, DIRECT_CUSTOMERS, GEOGRAPHIC, NO_RELIANCE, NO_US_REVENUE]
    )
    observation = _one(text, CONCENTRATION, "10-K")
    assert observation.state is TypedDisclosureState.EXTRACTED
    assert observation.item == "8" and observation.region_title == "NOTE 20: OPERATING SEGMENTS"
    by_polarity: dict[str, list[str]] = {}
    for instance in observation.instances:
        by_polarity.setdefault(instance.polarity, []).append(instance.statement_text)
        assert instance.action.startswith("CONCENTRATION_")
        assert text[instance.character_start : instance.character_end] == instance.statement_text
    assert by_polarity == {
        "STATED": [ONE_BOTTLER, DIRECT_CUSTOMERS],
        "ABSENT": [NO_BOTTLERS, NO_RELIANCE],
    }
    stated = {i.statement_text: i for i in observation.instances}
    assert _fields(stated[ONE_BOTTLER]) == {"share_1": "10%", "base": "net operating revenues"}
    assert _fields(stated[DIRECT_CUSTOMERS]) == {
        "share_1": "22%",
        "share_2": "14%",
        "base": "revenue",
    }, "the base is the revenue word the share is of, not the sales before it"
    assert stated[ONE_BOTTLER].period_end == parse_source_date("December 31, 2025")
    assert _fields(stated[NO_BOTTLERS]) == {
        "threshold": "10% or more",
        "base": "net operating revenues",
    }
    assert "period" in stated[DIRECT_CUSTOMERS].unknown_fields
    assert observation.unrecognized == (GEOGRAPHIC,)
    assert "1 relevant sentence(s) not recognised" in observation.reason
    absent = _one(
        _ten_k(note=["The Company has three reportable segments."]), CONCENTRATION, "10-K"
    )
    assert absent.state is TypedDisclosureState.NOT_FOUND and not absent.instances
    none = _one(_ten_k(), CONCENTRATION, "10-K")
    assert none.state is TypedDisclosureState.NOT_FOUND
    assert "no note headed by 'segment', 'concentration', 'customer'" in none.reason
    assert _one(_ten_q(), CONCENTRATION).state is TypedDisclosureState.NOT_FOUND


def test_the_cybersecurity_effect_statement_is_read_from_item_1c() -> None:
    """requirement (W2, typed rules v4): Item 1C's statement of whether
    cybersecurity threats or incidents have materially affected the
    registrant is `NOT_AFFECTED` when the effect clause is negated in its
    own clause, `_QUALIFIED` when stated through awareness or belief or
    under an exception, `AFFECTED` when stated without a negation; a
    forward-looking sentence alone is a risk statement (NOT_FOUND with the
    context named) and a program sentence naming material risks is not
    the statement; the family is not sealed for a 10-Q."""

    plain = _one(_ten_k(item1c=[CYBER_PROGRAM, CYBER_NOT_AFFECTED]), CYBER, "10-K")
    assert plain.state is TypedDisclosureState.EXTRACTED and plain.item == "1C"
    (instance,) = plain.instances
    assert instance.polarity == "NOT_AFFECTED" and instance.action == "EFFECT_STATED"
    assert instance.period_text == "during the past three fiscal years"
    assert _fields(instance)["subject"] == "Cybersecurity threats"
    assert _fields(instance)["effect_clause"] == "have not materially affected"
    aware = _one(_ten_k(item1c=[CYBER_NOT_AWARE, CYBER_FORWARD]), CYBER, "10-K")
    assert aware.state is TypedDisclosureState.EXTRACTED
    (instance,) = aware.instances
    assert instance.polarity == "NOT_AFFECTED_QUALIFIED"
    assert any(
        value.startswith("stated through the registrant's awareness")
        for value in instance.qualifiers
    )
    assert "the sentence also states a forward-looking likelihood" in instance.qualifiers
    forward = _one(_ten_k(item1c=[CYBER_PROGRAM, CYBER_FORWARD]), CYBER, "10-K")
    assert forward.state is TypedDisclosureState.NOT_FOUND
    assert "a forward-looking cybersecurity sentence (not this family)" in forward.context
    affected = _one(_ten_k(item1c=[CYBER_AFFECTED]), CYBER, "10-K")
    (instance,) = affected.instances
    assert instance.polarity == "AFFECTED_QUALIFIED"
    assert instance.qualifiers == ("except as described in Item 7.",)
    assert instance.period_text == "During the fourth quarter of 2025"
    assert _observe(_ten_q())[CYBER].state is TypedDisclosureState.NOT_APPLICABLE
    assert _one(_ten_k(), CYBER, "10-K").state is TypedDisclosureState.NOT_FOUND


def test_typed_spans_serve_every_assertion_before_the_inspected_scopes(tmp_path: Path) -> None:
    """requirement (W2, lead decision B at the typed channel): the typed
    span budget is dealt in two passes -- every observation's assertions
    document by document, then the inspected scopes of the non-extracted
    observations -- and an observation of a family the form does not
    define is not sealed, so the later documents' assertions are never
    displaced by the earlier documents' none and not-found scopes."""

    annual = _ten_k(item1c=[CYBER_NOT_AFFECTED], note=[NO_BOTTLERS])
    quarterly = _ten_q(item2=[SALE], item5=[PLAN_ADOPTED])
    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path,
        documents=(
            _document(text=annual, form="10-K", revision="10-k-2025"),
            _document(text=quarterly),
        ),
    )
    try:
        session = runtime.retrieval.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            record, spans = packet_module.select_typed_disclosures(session=session)
        finally:
            session.close()
        assert {value.state for value in record.observations} <= {
            "EXTRACTED",
            "NOT_FOUND",
            "EXPLICIT_NONE",
        }, "no NOT_APPLICABLE observation is sealed"
        assert {(value.document_type, value.family) for value in record.observations} == {
            ("10-K", DCP),
            ("10-K", PLANS),
            ("10-K", CONCENTRATION),
            ("10-K", CYBER),
            ("10-Q", DCP),
            ("10-Q", SALES),
            ("10-Q", PLANS),
            ("10-Q", CONCENTRATION),
        }
        handles = list(record.span_handles)
        instance_handles = [
            instance.span_handle
            for value in record.observations
            for instance in value.instances
            if instance.span_handle is not None
        ]
        scope_handles = [
            value.scope_span_handle
            for value in record.observations
            if value.scope_span_handle is not None
        ]
        assert instance_handles and scope_handles
        assert handles == [*instance_handles, *scope_handles], (
            "assertions of every document first, then the inspected scopes"
        )
        assert len(spans) == len(handles)
    finally:
        runtime.close()


CYBER_NOT_ONLY = (
    "Cybersecurity incidents have not only materially affected our operations but also "
    "increased costs."
)
CYBER_NOT_DETERMINED = (
    "We have not yet determined whether the cybersecurity incident identified in March 2026 "
    "has materially affected our financial condition."
)
CYBER_UNABLE = (
    "We are unable to determine whether cybersecurity threats have materially affected our "
    "business strategy."
)
NO_CUSTOMER_BUT = (
    "No customer accounted for 10% or more of our revenue in 2025; however, one customer "
    "accounted for 12% of our accounts receivable at December 31, 2025."
)
NO_CUSTOMER_CORRELATIVE = (
    "No single customer accounted for 10% or more of our revenue, not only in 2025 but "
    "also in 2024."
)


def test_a_correlative_and_an_open_determination_are_not_a_cyber_negative(
    tmp_path: Path,
) -> None:
    """requirement (the first-release safety closeout, finding A; typed rules
    v5): the negation governs the asserted predicate only. "have not only
    materially affected our operations but also increased costs" asserts the
    effect (`AFFECTED`, no exception qualifier from its "but also"); "have
    not yet determined whether ... has materially affected" and "unable to
    determine whether ... have materially affected" are no conclusion
    (AMBIGUOUS, the sentence visible, no polarity); the affirmative, the
    scoped negative, the awareness-qualified negative and the forward-looking
    risk statement keep their readings. Proved at the extraction owner, on
    the sealed typed record and on the public typed projection -- under v4
    the correlative sealed as a qualified negative and the open determination
    as a negative or an affirmative."""

    def observation(sentence: str):
        return _one(_ten_k(item1c=[CYBER_PROGRAM, sentence]), CYBER, "10-K")

    not_only = observation(CYBER_NOT_ONLY)
    assert not_only.state is TypedDisclosureState.EXTRACTED
    (instance,) = not_only.instances
    assert instance.polarity == "AFFECTED" and instance.qualifiers == ()
    assert instance.statement_text == CYBER_NOT_ONLY
    for sentence in (CYBER_NOT_DETERMINED, CYBER_UNABLE):
        open_one = observation(sentence)
        assert open_one.state is TypedDisclosureState.AMBIGUOUS and not open_one.instances
        assert open_one.unrecognized == (f"NO_CONCLUSION: {sentence}",)
        assert "no conclusion was reached, so no polarity is read" in open_one.reason
    assert observation(CYBER_AFFECTED).instances[0].polarity == "AFFECTED_QUALIFIED"
    assert observation(CYBER_NOT_AFFECTED).instances[0].polarity == "NOT_AFFECTED"
    assert observation(CYBER_NOT_AWARE).instances[0].polarity == "NOT_AFFECTED_QUALIFIED"
    forward = observation(CYBER_FORWARD)
    assert forward.state is TypedDisclosureState.NOT_FOUND and not forward.instances
    # The sealed record and the public projection carry the same reading.
    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path,
        documents=(
            _document(text=_ten_k(item1c=[CYBER_NOT_ONLY]), form="10-K", revision="10-k-2025"),
            _document(text=_ten_k(item1c=[CYBER_UNABLE]), form="10-K", revision="10-k-2025-open"),
        ),
    )
    try:
        session = runtime.retrieval.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            record, _spans = packet_module.select_typed_disclosures(session=session)
        finally:
            session.close()
        cyber = {
            value.revision_label: value for value in record.observations if value.family == CYBER
        }
        sealed = cyber["10-k-2025"]
        assert sealed.state == "EXTRACTED" and sealed.instances[0].polarity == "AFFECTED"
        view = packet_module.typed_view(sealed, sealed.instances[0])
        assertion = view["assertion"]
        assert isinstance(assertion, dict)
        assert assertion["polarity"] == "AFFECTED" and assertion["qualifiers"] == ()
        open_sealed = cyber["10-k-2025-open"]
        assert open_sealed.state == "AMBIGUOUS" and not open_sealed.instances
        assert open_sealed.unrecognized == (f"NO_CONCLUSION: {CYBER_UNABLE}",)
        scope_view = packet_module.typed_view(open_sealed, None)
        assert scope_view["state"] == "AMBIGUOUS"
        assert scope_view["unrecognized"] == (f"NO_CONCLUSION: {CYBER_UNABLE}",)
    finally:
        runtime.close()


def test_a_concentration_absence_beside_another_assertion_stays_qualified() -> None:
    """requirement (finding A's analogue at the concentration family): an
    absence stated beside another assertion in the same sentence -- a share
    of another base after "however", a correlative over two years -- is
    never a unit-wide unqualified absence: the polarity is
    `ABSENT_QUALIFIED`, the rest of the sentence is its qualifier (the other
    share stays visible inside it, never a silent second assertion), and the
    threshold and the base of the absence are bound."""

    observation = _one(
        _ten_k(note=[NO_CUSTOMER_BUT, NO_CUSTOMER_CORRELATIVE]), CONCENTRATION, "10-K"
    )
    assert observation.state is TypedDisclosureState.EXTRACTED
    by_text = {i.statement_text: i for i in observation.instances}
    however = by_text[NO_CUSTOMER_BUT]
    assert however.polarity == "ABSENT_QUALIFIED"
    assert _fields(however) == {"threshold": "10% or more", "base": "revenue"}
    assert however.qualifiers == (
        "however, one customer accounted for 12% of our accounts receivable at December 31, 2025.",
    )
    correlative = by_text[NO_CUSTOMER_CORRELATIVE]
    assert correlative.polarity == "ABSENT_QUALIFIED"
    assert correlative.qualifiers == ("but also in 2024.",)
    assert observation.unrecognized == ()


CYBER_NOT_PREVENTED = (
    "Cybersecurity incidents were not prevented by our controls and have materially "
    "affected our operations."
)
CYBER_NOT_DETECTED = (
    "Although the cybersecurity incident was not detected for several weeks, it has "
    "materially affected our operations."
)
CYBER_NO_INCIDENT = (
    "No cybersecurity incident has materially affected the Company during the past three "
    "fiscal years."
)
CYBER_NOT_EXPERIENCED = (
    "To date, we have not experienced any cybersecurity incidents that have materially "
    "affected or are reasonably likely to materially affect our business strategy, results "
    "of operations or financial condition."
)
CYBER_IDIOM = (
    "Cybersecurity incidents, including, but not limited to, phishing, have materially "
    "affected our operations."
)
CYBER_MODAL_PASSIVE = (
    "While we are not currently aware of any cybersecurity threats that are reasonably "
    "likely to materially affect the Company, we could be materially affected by such "
    "threats in the future."
)
CYBER_COORDINATED_AWARENESS = (
    "In the last three fiscal years, we have not identified any material cybersecurity "
    "incidents and have not identified any material risks from cybersecurity threats, "
    "including as a result of any previous cybersecurity incidents, that have materially "
    "affected the Company."
)


def test_a_negation_of_another_predicate_never_negates_the_cyber_effect(
    tmp_path: Path,
) -> None:
    """requirement (the predicate-scope correction; typed rules v6, the cyber
    rule v3): a negation is read only where the rule binds it to the effect
    -- the predicate's own verb group, a negative subject that governs the
    predicate, or a negated awareness or experience verb whose object is the
    cyber subject. "Cybersecurity incidents were not prevented by our
    controls and have materially affected our operations" negates
    prevention, not the effect: under v5 it sealed as a certain
    `NOT_AFFECTED`; the rule cannot prove which polarity the sentence
    asserts, so it is AMBIGUOUS with the sentence visible as
    `SCOPE_UNRESOLVED` (the parser's limit), never the registrant's own
    `NO_CONCLUSION`. The controls: the affirmative, the predicate-scoped
    negative, the negative subject, "not only ... but also", the unmade
    determination, the awareness-qualified negative, the forward-looking
    risk statement, the negated experience verb over "any cybersecurity
    incidents" and an idiom's inert "not limited to". Proved at the
    extraction owner, on the sealed record (the sentence delivered whole as
    the observation's scope, its bytes untouched), on the public typed
    projection and at the comparison consumer, which compares nothing
    against an unresolved state and names a rule change as such."""

    def observation(sentence: str):
        return _one(_ten_k(item1c=[CYBER_PROGRAM, sentence]), CYBER, "10-K")

    for sentence in (CYBER_NOT_PREVENTED, CYBER_NOT_DETECTED):
        unresolved = observation(sentence)
        assert unresolved.state is TypedDisclosureState.AMBIGUOUS and not unresolved.instances
        assert unresolved.unrecognized == (f"SCOPE_UNRESOLVED: {sentence}",)
        assert "cannot bind to the effect or to another assertion" in unresolved.reason
        assert "no conclusion" not in unresolved.reason
    assert observation(CYBER_AFFECTED).instances[0].polarity == "AFFECTED_QUALIFIED"
    (scoped,) = observation(CYBER_NOT_AFFECTED).instances
    assert scoped.polarity == "NOT_AFFECTED" and scoped.qualifiers == ()
    (subject,) = observation(CYBER_NO_INCIDENT).instances
    assert subject.polarity == "NOT_AFFECTED"
    assert subject.qualifiers == ("the negation is read on the subject: No cybersecurity incident",)
    assert subject.period_text == "during the past three fiscal years"
    assert observation(CYBER_NOT_ONLY).instances[0].polarity == "AFFECTED"
    determination = observation(CYBER_NOT_DETERMINED)
    assert determination.state is TypedDisclosureState.AMBIGUOUS
    assert determination.unrecognized == (f"NO_CONCLUSION: {CYBER_NOT_DETERMINED}",)
    (aware,) = observation(CYBER_NOT_AWARE).instances
    assert aware.polarity == "NOT_AFFECTED_QUALIFIED"
    assert any(
        value.startswith("stated through the registrant's awareness") for value in aware.qualifiers
    )
    assert any(
        value.startswith("the negation is read on the governing verb: are not aware")
        for value in aware.qualifiers
    )
    assert observation(CYBER_FORWARD).state is TypedDisclosureState.NOT_FOUND
    (experienced,) = observation(CYBER_NOT_EXPERIENCED).instances
    assert experienced.polarity == "NOT_AFFECTED", (
        "a negated experience verb over the incidents is a negative subject"
    )
    assert "the sentence also states a forward-looking likelihood" in experienced.qualifiers
    assert any(
        value.startswith("the negation is read on the governing verb: have not experienced")
        for value in experienced.qualifiers
    )
    (idiom,) = observation(CYBER_IDIOM).instances
    assert idiom.polarity == "AFFECTED" and idiom.qualifiers == ()
    # "could be materially affected" is the modal passive of a risk statement,
    # not a realized effect: under v5 it sealed as a qualified negative through
    # the unrelated "not currently aware" (a real 10-K's sentence).
    modal = observation(CYBER_MODAL_PASSIVE)
    assert modal.state is TypedDisclosureState.NOT_FOUND and not modal.instances
    assert "a forward-looking cybersecurity sentence (not this family)" in modal.context
    # Two coordinated negated awareness verbs bind together (a real 10-K's shape).
    (coordinated,) = observation(CYBER_COORDINATED_AWARENESS).instances
    assert coordinated.polarity == "NOT_AFFECTED_QUALIFIED"
    # A section with a read conclusion beside an unresolved sentence is not a
    # conclusion: the scalar family stays AMBIGUOUS and says why.
    mixed = _one(_ten_k(item1c=[CYBER_NOT_AFFECTED, CYBER_NOT_PREVENTED]), CYBER, "10-K")
    assert mixed.state is TypedDisclosureState.AMBIGUOUS and len(mixed.instances) == 1
    assert "beside a sentence whose negation the rule cannot bind" in mixed.reason
    # The sealed record and the public projection: the unresolved sentence is
    # delivered whole as the observation's scope, bytes untouched.
    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path,
        documents=(
            _document(text=_ten_k(item1c=[CYBER_NOT_PREVENTED]), form="10-K", revision="10-k-2025"),
        ),
    )
    try:
        session = runtime.retrieval.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            record, spans = packet_module.select_typed_disclosures(session=session)
        finally:
            session.close()
        (sealed,) = [value for value in record.observations if value.family == CYBER]
        assert sealed.state == "AMBIGUOUS" and not sealed.instances
        assert sealed.unrecognized == (f"SCOPE_UNRESOLVED: {CYBER_NOT_PREVENTED}",)
        assert sealed.rule_id == "typed-disclosures.cybersecurity-threat-effect.v3"
        assert record.rules_id == "alternative-evidence.typed-disclosures.v6"
        assert sealed.scope_span_handle is not None
        (scope_span,) = [value for value in spans if value.span_handle == sealed.scope_span_handle]
        assert CYBER_NOT_PREVENTED in scope_span.excerpt, (
            "the source statement is delivered, not lost"
        )
        view = packet_module.typed_view(sealed, None)
        assert view["state"] == "AMBIGUOUS"
        assert view["unrecognized"] == (f"SCOPE_UNRESOLVED: {CYBER_NOT_PREVENTED}",)
        assert "cannot bind" in str(view["scope"]), "the projection carries the reason as its scope"
        # The comparison consumer: against the same sentence sealed as a
        # certain negative under the earlier rules, a rule change is named
        # (never a continuing or resolved assertion); under the same rules an
        # unresolved side compares nothing.
        earlier = sealed.model_copy(
            update={
                "state": "EXTRACTED",
                "rule_id": "typed-disclosures.cybersecurity-threat-effect.v2",
                "unrecognized": (),
                "reason": "1 source assertion(s) parsed under the definition",
            }
        )
        (moved,) = compare_typed_disclosures(
            current=(sealed,),
            prior=(earlier,),
            current_rules=(record.rules_id, record.definitions_hash),
            prior_rules=("alternative-evidence.typed-disclosures.v5", "b" * 64),
        )
        assert moved.kind == CHANGE_RULE_OR_SOURCE and "not a new event" in moved.detail
        (unresolved_side,) = compare_typed_disclosures(
            current=(sealed,),
            prior=(
                earlier.model_copy(
                    update={"rule_id": sealed.rule_id, "revision_label": "10-k-2024"}
                ),
            ),
            current_rules=(record.rules_id, record.definitions_hash),
            prior_rules=(record.rules_id, record.definitions_hash),
        )
        assert unresolved_side.kind == CHANGE_NOT_COMPARABLE
        assert "disappearance is not resolution" in unresolved_side.detail
    finally:
        runtime.close()
