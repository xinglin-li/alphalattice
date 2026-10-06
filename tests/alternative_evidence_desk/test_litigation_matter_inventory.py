"""The provisional litigation matter inventory: navigation over a filing's
Legal Proceedings item and contingencies note, never discovery proof or
extraction. Controlled fixtures cover the shapes the S4 scope names as
controls: two distinct matters with similar party names, a dated procedural
filing that continues a matter rather than opening one, a late-in-note
statement, text no signpost claims and a note whose end the structure
cannot see. (The production plan's reading plan went with the plan:
first-release integration T5; the integrated selection reads the inventory
through its own needs.)"""

from __future__ import annotations

from alphalattice.evidence.alternative_evidence.analysis.matters import (
    MATTER_INVENTORY_RULES_ID,
    litigation_regions,
    matter_inventory,
)
from alphalattice.evidence.alternative_evidence.documents.structure import DocumentStructure
from tests.alternative_evidence_desk.litigation_support import (
    BOILERPLATE,
    FEDERAL_DERIVATIVE,
    ITEM_CAPTIONS,
    SECURITIES,
    STATE_DERIVATIVE,
    _filing,
)


def test_similar_party_names_stay_distinct_and_procedural_filings_continue_a_matter() -> None:
    text = _filing(
        note_lines=[BOILERPLATE, SECURITIES, FEDERAL_DERIVATIVE, STATE_DERIVATIVE],
        item_lines=ITEM_CAPTIONS,
    )
    structure = DocumentStructure(text, document_type="10-Q")
    inventory = matter_inventory(text, structure, document_id="T")
    assert inventory.rules_id == MATTER_INVENTORY_RULES_ID
    assert [r.kind for r in inventory.regions] == ["NOTE", "ITEM"]
    assert not any(r.end_uncertain for r in inventory.regions)
    titles = [(m.title, m.basis) for m in inventory.matters]
    # Three note matters, all "v. Todd J. Vasos"-shaped, opened by their own
    # italic captions; the boilerplate stays visible and unassigned.
    assert titles[:3] == [
        (
            "Washtenaw County Employees' Retirement System v. Dollar General Corporation, et al.",
            "italic caption",
        ),
        ("Nathan Silva v. Todd J. Vasos, et al.", "italic caption"),
        ("Todd Hellrigel v. Todd J. Vasos et al.", "italic caption"),
    ]
    assert inventory.matters[1].aliases == ("Silva",)
    assert inventory.matters[2].aliases == ("Hellrigel",)
    assert inventory.matters[2].case_numbers == ("Case No. 24-0392-I",)
    assert [why for _s, _e, why in inventory.unassigned] == [
        "lead-in or boilerplate (no litigation cue)"
    ]
    assert text[inventory.unassigned[0][0] : inventory.unassigned[0][1]] == BOILERPLATE
    # Inside the Legal Proceedings item every caption is a proceeding; the
    # dated amended complaint continues Cruz instead of opening a matter.
    assert titles[3:] == [
        ("Cruz Litigation", "caption"),
        ("Faustina Plant Risk Management Plan", "caption"),
    ]
    cruz = inventory.matters[3]
    (cruz_range,) = cruz.ranges
    assert text[cruz_range[0] : cruz_range[1]] == "\n\n".join(ITEM_CAPTIONS[:2])
    # A late statement is inside its matter's single contiguous range.
    late = text.index("On April 23, 2026")
    (state_range,) = inventory.matters[2].ranges
    assert state_range[0] <= late < state_range[1]
    assert len({m.handle for m in inventory.matters}) == 5, "handles are local and distinct"


def test_named_is_the_owner_classification_and_group_is_the_source_heading() -> None:
    """requirement: a matter's `group` is the sub-heading the source sets it
    under -- a title, never a category -- and `named` is the owner's
    classification. A survey that compared the heading to a category word
    reported zero named matters over sixteen filings whose dated securities,
    antitrust and derivative filings the owner names; the count must be
    taken from the classification the delivered records use, not from the
    heading."""

    text = _filing(
        note_lines=[
            BOILERPLATE,
            "**Securities Actions**",
            "On May 12, 2023, a private securities class action lawsuit was filed in the U.S. "
            "District Court for the Central District of California against the Company and "
            "certain of its officers. The complaint alleges violations of the securities laws.",
            "On December 8, 2025, a private securities lawsuit was filed in the U.S. District "
            "Court for the Central District of California against the Company and certain of "
            "its officers alleging similar claims.",
            "**Derivative Actions**",
            "Ten shareholder derivative complaints have been filed against the Company and "
            "certain current and former officers and directors. Each of these actions is "
            "purportedly brought on behalf of the Company.",
            "**Tax Matters**",
            "Between December 2018 and August 2021, the Italian tax authorities issued "
            "assessments on the Company's Italian subsidiary totaling approximately $295 "
            "million, which the Company is contesting in litigation before the tax courts.",
        ],
    )
    structure = DocumentStructure(text, document_type="10-Q")
    inventory = matter_inventory(text, structure, document_id="T")
    by_group: dict[str, list[tuple[str, bool]]] = {}
    for matter in inventory.matters:
        by_group.setdefault(matter.group, []).append((matter.basis, matter.named))
    assert by_group == {
        "Securities Actions": [("dated filing", True), ("dated filing", True)],
        "Derivative Actions": [("filings", True)],
        "Tax Matters": [("sub-heading", False)],
    }
    assert not any(matter.group == "NAMED" for matter in inventory.matters)
    assert sum(1 for matter in inventory.matters if matter.named) == 3


def test_a_dash_run_in_label_a_caption_with_et_al_and_a_us_district_court_open_matters() -> None:
    """requirement: NEE's contingencies note -- "Legal Proceedings \u2013 NEE, FPL,
    ... are the named defendants in a purported shareholder securities class
    action lawsuit filed in the U.S. District Court for the Southern District
    of Florida ..." and "In November 2024, NEE was named as defendant in an
    antitrust lawsuit (Avangrid, Inc. et al. v. NextEra Energy, Inc.) filed in
    the U.S. District Court for the District of Massachusetts." Under v1 the
    dash was not a run-in label, "et al." and "v." ended the dated lead and
    "U.S. District Court" was not a court: three named proceedings were
    unassigned text."""

    securities = (
        "Legal Proceedings \u2013 NEE, FPL, and certain current and former executives, are the "
        "named defendants in a purported shareholder securities class action lawsuit filed in "
        "the U.S. District Court for the Southern District of Florida in January 2024. The "
        "complaint alleges that the defendants made false and misleading statements."
    )
    derivative = (
        "NEE, along with certain current and former executives and directors are the named "
        "defendants in purported shareholder derivative actions filed in the 15th Judicial "
        "Circuit in Palm Beach County, Florida in July 2023, March 2024 and May 2024."
    )
    antitrust = (
        "In November 2024, NEE was named as defendant in an antitrust lawsuit (Avangrid, Inc. "
        "et al. v. NextEra Energy, Inc.) filed in the U.S. District Court for the District of "
        "Massachusetts. The original complaint sought damages of $350 million."
    )
    text = _filing(note_lines=[BOILERPLATE, securities, derivative, antitrust])
    structure = DocumentStructure(text, document_type="10-K")
    inventory = matter_inventory(text, structure, document_id="NEE")
    assert inventory.rules_id.endswith(".v2")
    assert [(m.title[:32], m.basis, m.named) for m in inventory.matters] == [
        ("Legal Proceedings", "jurisdiction", True),
        ("NEE was named as defendant in an", "dated filing", True),
    ]
    assert inventory.matters[0].courts == (
        "U.S. District Court for the Southern District of Florida",
    )
    assert inventory.matters[1].courts == ("U.S. District Court for the District of Massachusetts",)
    # The derivative paragraph continues the run-in matter; nothing of the
    # three proceedings is unassigned, the boilerplate still is.
    (first_range,) = inventory.matters[0].ranges
    assert text[first_range[0] : first_range[1]] == securities + "\n\n" + derivative
    assert [text[s:e] for s, e, _why in inventory.unassigned] == [BOILERPLATE, "See Note 7."]


def test_a_note_whose_end_is_not_seen_is_reported_uncertain_and_bounded() -> None:
    text = _filing(note_lines=[BOILERPLATE, SECURITIES], close_note=False)
    structure = DocumentStructure(text, document_type="10-Q")
    (note, _item) = litigation_regions(text, structure)
    assert note.end_uncertain
    assert "Executive Overview" in note.end_basis
    assert text[note.body_start : note.body_end].rstrip().endswith("The motion remains pending.")
    inventory = matter_inventory(text, structure, document_id="T")
    assert [m.title[:16] for m in inventory.matters] == ["Washtenaw County"]
