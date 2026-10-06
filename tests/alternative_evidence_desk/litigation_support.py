"""Controlled litigation-region fixtures shared by the matter inventory and
matter delivery tests: a 10-Q whose contingencies note holds three
derivative/securities matters with similar party names and whose Legal
Proceedings item holds captioned matters; the candidate delivery's note
shapes (two filings in one paragraph, a dated filing cut by a page break,
an accrual lead-in, a materiality closing, references to filings not held,
one matter longer than a window) and the filing that holds them. Real
filings' shapes, not their text."""

from __future__ import annotations

COVER = [
    "UNITED STATES",
    "SECURITIES AND EXCHANGE COMMISSION",
    "FORM 10-Q",
    "For the quarterly period ended June 30, 2026",
    "Indicate by check mark whether the registrant is a shell company. No",
    "PART I. FINANCIAL INFORMATION",
    "ITEM 1. FINANCIAL STATEMENTS",
]
BOILERPLATE = (
    "From time to time, the Company is a party to various legal matters in the ordinary "
    "course of its business. The Company has recorded accruals with respect to these "
    "matters, where appropriate."
)
SECURITIES = (
    "On November 27, 2023, the following putative shareholder class action lawsuit was filed "
    "in the United States District Court for the Middle District of Tennessee: *Washtenaw County "
    "Employees' Retirement System v. Dollar General Corporation, et al.* (Case No. 3:23-cv-01250) "
    '(the "Securities Litigation"). On June 23, 2025, the court granted defendants\' motion to '
    "dismiss without prejudice. On August 25, 2025, the lead plaintiffs filed a motion for leave "
    "to amend the second consolidated amended complaint. The motion remains pending."
)
FEDERAL_DERIVATIVE = (
    "On January 26, 2024, the following shareholder derivative action was filed in the United "
    "States District Court for the Middle District of Tennessee: *Nathan Silva v. Todd J. Vasos, "
    'et al.* (Case No. 3:24-cv-00083) ("Silva"). On May 2, 2024, the Silva action was '
    "dismissed."
)
STATE_DERIVATIVE = (
    "On March 26, 2024, the following shareholder derivative action was filed in the Chancery "
    "Court for Davidson County, Tennessee: *Todd Hellrigel v. Todd J. Vasos et al.* (Case No. "
    '24-0392-I) ("Hellrigel"). On May 20, 2024, the court entered an agreed order '
    "staying the Hellrigel action. On April 23, 2026, the court further extended the stay."
)
ITEM_CAPTIONS = [
    "Cruz Litigation. On August 27, 2020, a putative class action complaint was filed in the "
    "Circuit Court of the Thirteenth Judicial Circuit in Hillsborough County, Florida against "
    "our subsidiary. In late March 2023, the court denied defendants' motions to dismiss.",
    "On November 3, 2021, plaintiffs filed an amended complaint and Mosaic filed a motion to "
    "dismiss that complaint with prejudice on November 15, 2021.",
    "Faustina Plant Risk Management Plan. On September 14, 2022, EPA Region 6 issued a Notice of "
    "Potential Violation and Opportunity to Confer regarding compliance of our Faustina Plant.",
]


def _text(lines: list[str]) -> str:
    return "\n\n".join(lines) + "\n"


def _filing(
    *, note_lines: list[str], item_lines: list[str] | None = None, close_note: bool = True
) -> str:
    lines = [
        *COVER,
        "The condensed consolidated statements follow.",
        "NOTE 7. COMMITMENTS AND CONTINGENCIES",
        "**Legal proceedings**",
        *note_lines,
    ]
    if close_note:
        lines += ["NOTE 8. SEGMENT INFORMATION", "The Company manages its business as one segment."]
    else:
        lines += ["Executive Overview", "Net sales increased in the period."]
    lines += ["PART II. OTHER INFORMATION", "ITEM 1. LEGAL PROCEEDINGS"]
    lines += item_lines or ["See Note 7."]
    lines += ["ITEM 6. EXHIBITS", "See the exhibit index.", "SIGNATURES"]
    return _text(lines)


def _matter(number: int) -> str:
    """One named matter whose text carries multibyte quotes: bytes, not
    characters, are what the window ceiling is measured in."""

    return (
        f"On March {number % 28 + 1}, 2025, Plaintiff Number {number} filed a complaint in the "
        "United States District Court for the District of Delaware "
        f"(Case No. 1:25-cv-{number:05d}) alleging that the Company\u2019s products infringed "
        f"the plaintiff\u2019s patents (the \u201cPatent Action {number}\u201d). The Company "
        f"filed an answer denying the allegations. On June {number % 28 + 1}, 2025, the court "
        "denied the plaintiff\u2019s motion for a preliminary injunction, and discovery has "
        "commenced."
    )


ACCRUAL_LEAD_IN = (
    "The Company is involved in many claims, proceedings and litigations arising from its "
    "business. The Company has recorded an immaterial accrual with respect to some matters "
    "described below, in addition to other immaterial accruals for matters not described below."
)
MADERA_HOWELL = (
    "In January 2026, a class action on behalf of Washington employees was filed against the "
    "Company alleging failure to provide meal periods and rest breaks. The complaint seeks "
    "compensatory and exemplary damages. Madera v. Costco Wholesale Corp. (No. 26-2-02879-6, "
    "King County Superior Court). A second class action was filed in March 2026 alleging "
    "failure to provide meal periods and to compensate for violations, seeking similar relief. "
    "Howell v. Costco Wholesale Corp. (No. 26-2-09459-5; King County Superior Court). The "
    "Company has denied the material allegations of the complaints."
)
PAGE_CUT_HEAD = (
    "On April 11, 2025, the Commissioner of the Department of Licensing and Consumer Affairs "
    "and the Government of the United"
)
PAGE_CUT_TAIL = (
    "States Virgin Islands filed a lawsuit against the Company in the Superior Court of the "
    "Virgin Islands (Case No. ST-2025-CV-00123) concerning the environmental impacts of plastic "
    "packaging. The complaint seeks civil penalties and injunctive relief."
)
TRAILING_CLOSING = (
    "The Company does not believe that any pending claim, proceeding or litigation, either "
    "alone or in the aggregate, will have a material adverse effect on the Company"
    "\u2019s "
    "financial position; it is possible that an unfavorable outcome of some or all of the "
    "matters, however unlikely, could result in a charge that might be material to the "
    "results of an individual fiscal quarter or year."
)
BETWEEN_MATTERS = (
    "The Company believes these matters are without merit and intends to defend them "
    "vigorously; no accrual has been recorded for any of the matters described above."
)
ITEM_WITH_REFERENCES = [
    "See Note 7.",
    'For the risks these proceedings present, see "Risk Factors" in Part I.',
    "For a description of the tax dispute, see our Quarterly Report on Form 10-Q for the "
    "quarter ended March 31, 2026.",
]
GENERIC_ONLY = (
    "We are subject to private lawsuits, administrative proceedings and claims that arise in "
    "the ordinary course of our business. We do not believe that any pending legal proceedings "
    "will have a material adverse effect on our financial position or results of operations."
)


LONG_MATTER = (
    _matter(99)
    + " "
    + " ".join(
        f"On March {day}, 2025, the court entered scheduling order number {day} in the Patent "
        "Action 99, setting the next deadline."
        for day in range(1, 29)
    )
)
"""One matter longer than a window: its statements continue past its opening."""


def _matters_filing(*, extra: int = 0, long: bool = False) -> str:
    return _filing(
        note_lines=[
            ACCRUAL_LEAD_IN,
            SECURITIES,
            FEDERAL_DERIVATIVE,
            BETWEEN_MATTERS,
            MADERA_HOWELL,
            PAGE_CUT_HEAD,
            PAGE_CUT_TAIL,
            *(_matter(number) for number in range(1, extra + 1)),
            *([LONG_MATTER] if long else []),
            TRAILING_CLOSING,
        ],
        item_lines=ITEM_WITH_REFERENCES,
    )
