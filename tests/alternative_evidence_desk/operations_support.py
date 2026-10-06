"""Controlled operations-disclosure fixtures shared by the operations
inventory and the routing tests, in the shapes the traced filings have: a
risk-factor paragraph with a dated store closure and impairment (DG), an
MD&A paragraph with a dated decision to close restaurants (DRI), a
`SIGNIFICANT OPERATING AND NONOPERATING ITEMS` note with an `Other
Operating Charges` sub-heading, dated charges and a cross-reference stub
(KO), a restructuring note (MMM), a segment disclosures note that is no
closure, an acquisitions note the corporate-event family keeps, and an
MD&A covenant sub-heading the structure is uncertain of (SYF). Real
filings' shapes, not their text."""

from __future__ import annotations

from tests.alternative_evidence_desk.event_support import ANNUAL_COVER

RISK_CLOSURE = (
    "Our growth strategy depends on new store openings. In the first quarter of 2025, we "
    "closed 45 pOpshelf stores and converted an additional six to Dollar General stores, as "
    "well as incurred significant impairment charges, the majority of which relate to the "
    "pOpshelf stores. There can be no assurance that future closures will not be necessary."
)
RISK_PLAIN = (
    "Our business is subject to risks from competition and consumer spending, which could "
    "adversely affect our results."
)
MDA_CLOSURE = (
    "On February 3, 2026, we announced the completion of this process and our decision to "
    "permanently close approximately half of our Bahama Breeze restaurants during the "
    "fourth quarter of fiscal 2026."
)
MDA_OPENINGS = (
    "In fiscal 2025, we opened 20 new restaurants and remodeled 15 others, consistent with "
    "our growth plans."
)
COVENANT_HEADING = "Covenants"
COVENANT_STATEMENT = (
    "The indentures pursuant to which our senior and subordinated unsecured notes have been "
    "issued include various covenants. We were in compliance with all of these covenants at "
    "June 30, 2026."
)
CREDIT_RATINGS = "Credit Ratings"
CREDIT_RATINGS_STATEMENT = "Our credit ratings were unchanged during the period."
OTHER_CHARGES_HEADING = "Other Operating Charges"
CHARGES_2025 = (
    "In 2025, the Company recorded other operating charges of $1,261 million. These "
    "charges consisted of $960 million related to the impairment of our BodyArmor "
    "trademark, $97 million related to the Company\u2019s productivity and reinvestment "
    "program, and $47 million related to the remeasurement of our contingent consideration "
    "liability."
)
CHARGES_2024 = (
    "In 2024, the Company recorded other operating charges of $4,163 million, which "
    "consisted primarily of $3,100 million related to the impairment of our fairlife "
    "trademark."
)
CROSS_REFERENCE = "Refer to Note 2 for additional information on our divestiture activities."
NONOPERATING_HEADING = "Other Nonoperating Items"
NONOPERATING = (
    "During 2025, the Company recognized a gain of $1,952 million related to the sale of our "
    "bottling operations in Africa, which was recorded in other income."
)
RESTRUCTURING_LEAD = (
    "Transformation Costs: In the third quarter of 2025, 3M began a transformation program "
    "intended as a structural change to reduce costs, and incurred restructuring charges of "
    "$85 million in severance and related benefits."
)
SEGMENT_NOTE = (
    "The Company has three reportable segments. Segment operating income is the measure "
    "reviewed by the chief operating decision maker."
)
ACQUISITION_LEAD = (
    "In March 2025, the Company sold a portion of its ownership interest in a bottling "
    "partner for $2.1 billion and recorded an impairment charge of $12 million on the "
    "retained interest."
)


def _operations_filing() -> str:
    lines = [
        *ANNUAL_COVER,
        "ITEM 1A. RISK FACTORS",
        RISK_PLAIN,
        RISK_CLOSURE,
        "PART II",
        "ITEM 7. MANAGEMENT\u2019S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS "
        "OF OPERATIONS",
        MDA_OPENINGS,
        MDA_CLOSURE,
        "Liquidity and Capital Resources",
        "We fund our operations from cash flow and borrowings.",
        COVENANT_HEADING,
        COVENANT_STATEMENT,
        CREDIT_RATINGS,
        CREDIT_RATINGS_STATEMENT,
        "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        "NOTE 2: ACQUISITIONS AND DIVESTITURES",
        ACQUISITION_LEAD,
        "NOTE 5. RESTRUCTURING ACTIONS",
        RESTRUCTURING_LEAD,
        "NOTE 18: SIGNIFICANT OPERATING AND NONOPERATING ITEMS",
        OTHER_CHARGES_HEADING,
        CHARGES_2025,
        CHARGES_2024,
        CROSS_REFERENCE,
        NONOPERATING_HEADING,
        NONOPERATING,
        "NOTE 21. SEGMENT DISCLOSURES",
        SEGMENT_NOTE,
        "PART IV",
        "ITEM 15. EXHIBITS AND FINANCIAL STATEMENT SCHEDULES",
        "See the exhibit index.",
        "SIGNATURES",
    ]
    return "\n\n".join(lines) + "\n"
