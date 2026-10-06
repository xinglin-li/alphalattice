"""Controlled corporate-event fixtures shared by the event inventory and
the matter delivery tests: a 10-K whose acquisitions note holds dated
sales, a run-in-titled sale, a purchase-price allocation and a pro forma
paragraph, a note named after its counterparty, a subsequent-events note,
a debt note with a dated issue and an MD&A with a dated sale (neither a
region); and an 8-K with a results item, an other-events item and the
exhibits item; and a late-filing notice (NT 10-Q) with its narrative and
other-information parts. Real filings' shapes, not their text."""

from __future__ import annotations

ANNUAL_COVER = [
    "UNITED STATES",
    "SECURITIES AND EXCHANGE COMMISSION",
    "FORM 10-K",
    "For the fiscal year ended December 31, 2025",
    "Indicate by check mark whether the registrant is a shell company. No",
    "PART I",
    "ITEM 1. BUSINESS",
    "The Company sells beverages through bottling partners.",
]
MDA_SALE = (
    "In April 2025, we sold our Canadian restaurants to a franchisee for $60 million and "
    "recognized a gain of $12 million."
)
CATEGORY_LEAD_IN = (
    "Acquisitions Our Company\u2019s acquisitions of businesses, equity method investments "
    "and nonmarketable securities during 2025 totaled $200 million."
)
CCEP_SALE = (
    "In March 2025, the Company sold a portion of our ownership interest in CCEP, an equity "
    "method investee, for net cash proceeds of $1,100 million and recognized a gain of $300 "
    "million."
)
INDIA_REFRANCHISING = (
    "In May 2025, the Company refranchised our bottling operations in certain territories "
    "in India that were previously classified as held for sale, for total consideration of "
    "$500 million."
)
PETS_BEST = (
    "Pets Best In March 2024, we sold our wholly-owned subsidiary, Pets Best Insurance "
    "Services, LLC, to Independence Pet Holdings, Inc. in exchange for cash and an equity "
    "interest in the buyer."
)
ALLOCATION = "The final allocation of the purchase price as of December 31, 2025 is as follows."
TABLE = "[Table of 12 rows not carried into the canonical text; the original retains it.]"
PRO_FORMA = (
    "The following unaudited pro forma financial information presents the combined results "
    "of the transactions described above as if they had occurred on January 1, 2024."
)
DEBT_ISSUE = (
    "In June 2025, we issued $1,000 million of senior notes due 2035 with a coupon of 4.9 "
    "percent; the proceeds were used for general corporate purposes."
)
GROQ_LEAD = (
    "In December 2025, we entered into a non-exclusive license agreement with Groq, Inc. for "
    "its language processing unit technology and hired certain Groq employees. We recorded "
    "$14.4 billion of goodwill and a $2.5 billion developed technology intangible asset; total "
    "consideration consists of $13.0 billion paid at closing and $4 billion payable within one "
    "year."
)
DIVIDEND_DECLARED = (
    "On June 24, 2026, the Board of Directors declared a cash dividend of $1.62 per share to "
    "be paid August 3, 2026 to shareholders of record on July 10, 2026."
)
EVALUATED = (
    "We have evaluated subsequent events through the date the financial statements were "
    "issued and identified no other events requiring disclosure."
)


def _event_filing() -> str:
    lines = [
        *ANNUAL_COVER,
        "ITEM 7. MANAGEMENT\u2019S DISCUSSION AND ANALYSIS",
        MDA_SALE,
        "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        "NOTE 1. SUMMARY OF SIGNIFICANT ACCOUNTING POLICIES",
        "Acquisitions are accounted for under the acquisition method of accounting.",
        "NOTE 2. ACQUISITIONS AND DIVESTITURES",
        CATEGORY_LEAD_IN,
        CCEP_SALE,
        INDIA_REFRANCHISING,
        PETS_BEST,
        ALLOCATION,
        TABLE,
        PRO_FORMA,
        "NOTE 3. DEBT",
        DEBT_ISSUE,
        "NOTE 4. GROQ",
        GROQ_LEAD,
        "NOTE 5. SUBSEQUENT EVENTS",
        DIVIDEND_DECLARED,
        EVALUATED,
        "NOTE 6. SEGMENT INFORMATION",
        "The Company manages its business as one segment.",
        "PART IV",
        "ITEM 15. EXHIBITS AND FINANCIAL STATEMENT SCHEDULES",
        "See the exhibit index.",
        "SIGNATURES",
    ]
    return "\n\n".join(lines) + "\n"


CURRENT_COVER = [
    "UNITED STATES",
    "SECURITIES AND EXCHANGE COMMISSION",
    "FORM 8-K",
    "CURRENT REPORT",
    "Date of Report (Date of earliest event reported): July 7, 2026",
]
RESULTS_ITEM = (
    "On July 7, 2026, the Company issued a press release announcing its results of operations "
    "for the quarter ended June 30, 2026. A copy of the press release is furnished as Exhibit "
    "99.1 to this report."
)
OTHER_EVENTS_ITEM = (
    "On July 7, 2026, the Board of Directors declared a quarterly cash dividend of $1.47 per "
    "share, payable August 14, 2026 to shareholders of record at the close of business on "
    "July 31, 2026."
)


def _current_report() -> str:
    lines = [
        *CURRENT_COVER,
        "Item 2.02 Results of Operations and Financial Condition.",
        RESULTS_ITEM,
        "Item 8.01 Other Events.",
        OTHER_EVENTS_ITEM,
        "Item 9.01 Financial Statements and Exhibits.",
        "(d) Exhibits. 99.1 Press release dated July 7, 2026.",
        "SIGNATURES",
    ]
    return "\n\n".join(lines) + "\n"


LATE_FILING_COVER = [
    "UNITED STATES",
    "SECURITIES AND EXCHANGE COMMISSION",
    "FORM 12b-25",
    "NOTIFICATION OF LATE FILING",
    "(Check one): Form 10-K Form 20-F Form 11-K [X] Form 10-Q Form 10-D Form N-CEN Form N-CSR",
    "For Period Ended: June 30, 2026",
    "PART I -- REGISTRANT INFORMATION",
    "Example Holdings, Inc. 100 Main Street, Springfield",
    "PART II -- RULES 12b-25(b) AND (c)",
    "[X] (a) The reason described in reasonable detail in Part III of this form could not be "
    "eliminated without unreasonable effort or expense.",
]
LATE_FILING_NARRATIVE = (
    "The Company could not file its Quarterly Report on Form 10-Q for the period ended June 30, "
    "2026 within the prescribed time period because the Audit Committee is reviewing the "
    "accounting for revenue recognized on certain multi-year contracts, and the Company expects "
    "to restate its financial statements for the fiscal year ended December 31, 2025."
)
LATE_FILING_CHANGE = (
    "(3) Is it anticipated that any significant change in results of operations from the "
    "corresponding period for the last fiscal year will be reflected by the earnings statements "
    "to be included in the subject report? [X] Yes. The Company expects to report a net loss of "
    "approximately $40 million for the quarter, compared with net income of $12 million a year "
    "earlier."
)


def _late_filing_notice() -> str:
    """A late-filing notice (Form 12b-25, filed as NT 10-Q): the registrant and
    the rule parts, the narrative (Part III), the other information (Part IV)
    and the signature. The official form's shape, invented text."""

    lines = [
        *LATE_FILING_COVER,
        "PART III -- NARRATIVE",
        LATE_FILING_NARRATIVE,
        "PART IV -- OTHER INFORMATION",
        LATE_FILING_CHANGE,
        "SIGNATURE",
        "Example Holdings, Inc. has caused this notification to be signed on its behalf.",
    ]
    return "\n\n".join(lines) + "\n"
