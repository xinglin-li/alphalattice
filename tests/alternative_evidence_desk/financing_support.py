"""Controlled financing-note fixtures shared by the financing inventory and
the matter delivery tests, in the shapes the development filings have: a
run-in sub-heading, a commercial paper paragraph with a balance and a
stated absence, a lines-of-credit paragraph with an unused capacity and
'no borrowings' inside it, a long-term debt sub-heading with a table not
carried and footnotes, a covenant statement that only refers to the notes,
a notes paragraph naming two instruments with an issuance and a repayment,
a stated absence of commercial paper, one sentence stating an absence for
one instrument beside an amount outstanding under another, a negative
qualified by an exception, a bulleted capacity list continuing its lead, a
matured facility, a securities note that is an asset note, and MD&A
liquidity text that is no region. Real filings' shapes, not their text."""

from __future__ import annotations

from tests.alternative_evidence_desk.event_support import ANNUAL_COVER

COMMERCIAL_PAPER = (
    "Loans and notes payable consist primarily of commercial paper issued in the United "
    "States. As of December 31, 2025 and 2024, we had $1,495 million and $1,139 million, "
    "respectively, in outstanding commercial paper borrowings. Our weighted-average interest "
    "rates for commercial paper outstanding were 3.9% and 5.0% as of December 31, 2025 and "
    "2024, respectively."
)
LINES_OF_CREDIT = (
    "In addition, we had $7,227 million in unused lines of credit and other short-term credit "
    "facilities as of December 31, 2025, of which $6,150 million was in corporate backup lines "
    "of credit for general purposes. These backup lines of credit expire at various times "
    "through 2030. There were no borrowings under these corporate backup lines of credit "
    "during 2025. These credit facilities are subject to normal banking terms and conditions."
)
LONG_TERM_LEAD = "The Company\u2019s long-term debt consisted of the following (in millions):"
TABLE = "[Table of 12 rows not carried into the canonical text; the original retains it.]"
FOOTNOTE = (
    "4 As of December 31, 2025 and 2024, the fair value of our long-term debt, including the "
    "current portion, was $39,385 million and $38,052 million, respectively."
)
COVENANT = (
    "As of December 31, 2025, we complied with the required covenants, which are "
    "non-financial in nature, under the outstanding notes."
)
TWO_NOTES = (
    "The fair value of the Senior Notes is estimated using Level 2 inputs. Other long-term "
    "debt consists of Guaranteed Senior Notes issued by the Company's Japan subsidiary. In "
    "March 2026, the Company's Japan subsidiary repaid $69 million of its Guaranteed Senior "
    "Notes. In June 2025, we issued $1,000 million of 4.9% Senior Notes due 2035."
)
NO_PAPER = "As of December 31, 2025, no commercial paper was outstanding."
MIXED_POLARITY = (
    "At December 31, 2025, no commercial paper was outstanding, but $25 million remained "
    "outstanding under the revolving credit facility."
)
QUALIFIED_NEGATIVE = (
    "No borrowings were outstanding under the term loan facility as of December 31, 2025, "
    "other than $12 million of letters of credit issued under it."
)
CAPACITY_NEGATIVE = (
    "As of July 26, 2026, our commercial paper program had a capacity of $25.0 billion, with "
    "no amounts outstanding."
)
TWO_IN_ONE_CLAUSE = (
    "We had no borrowings outstanding under the revolving credit facility and $25 million of "
    "commercial paper outstanding at year end."
)
PRONOMINAL_DRAW = (
    "Throughout fiscal 2025 no borrowings were outstanding under the term loan facility. In "
    "January 2026, the Company borrowed $200 million under it to fund the acquisition, and the "
    "amount remained outstanding at the date of this report."
)
PRONOMINAL_ABSENCE = (
    "At year end no amounts were outstanding under the subordinated debentures due 2035 "
    "program. It matures in 2035 and no amounts had been drawn under it at any time during "
    "the year."
)
CAPACITY_LEAD = (
    "We have undrawn committed and uncommitted capacity under our credit facilities from "
    "private lenders under our securitization programs, subject to customary borrowing "
    "conditions. At December 31, 2025, we had:"
)
CAPACITY_BULLET_1 = (
    "•an aggregate of $2.6 billion of undrawn capacity under our securitization "
    "financings, of which $2.1 billion was committed and $450 million was uncommitted, and"
)
CAPACITY_BULLET_2 = (
    "•an aggregate of $10.0 billion of available borrowing capacity through the Federal "
    "Reserve discount window based on the amount and type of assets pledged."
)
MATURED = (
    "In July 2025, our $500 million unsecured revolving credit facility with private lenders "
    "matured."
)
SECURITIES_NOTE = (
    "All of our debt securities are classified as available-for-sale and are held to "
    "generate interest income."
)
MDA_LIQUIDITY = (
    "We believe our cash, our commercial paper program and our $2 billion revolving credit "
    "facility are sufficient to meet our liquidity needs for the next twelve months."
)


def _financing_filing() -> str:
    lines = [
        *ANNUAL_COVER,
        "ITEM 7. MANAGEMENT\u2019S DISCUSSION AND ANALYSIS",
        MDA_LIQUIDITY,
        "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        "NOTE 4. DEBT SECURITIES",
        SECURITIES_NOTE,
        "NOTE 11: DEBT AND BORROWING ARRANGEMENTS",
        "Loans and Notes Payable",
        COMMERCIAL_PAPER,
        LINES_OF_CREDIT,
        "Long-Term Debt",
        LONG_TERM_LEAD,
        TABLE,
        FOOTNOTE,
        COVENANT,
        TWO_NOTES,
        NO_PAPER,
        MIXED_POLARITY,
        QUALIFIED_NEGATIVE,
        CAPACITY_NEGATIVE,
        TWO_IN_ONE_CLAUSE,
        PRONOMINAL_DRAW,
        PRONOMINAL_ABSENCE,
        "Additional Sources of Liquidity",
        CAPACITY_LEAD,
        CAPACITY_BULLET_1,
        CAPACITY_BULLET_2,
        MATURED,
        "NOTE 12: COMMITMENTS AND CONTINGENCIES",
        "The Company is involved in various legal proceedings.",
        "PART IV",
        "ITEM 15. EXHIBITS AND FINANCIAL STATEMENT SCHEDULES",
        "See the exhibit index.",
        "SIGNATURES",
    ]
    return "\n\n".join(lines) + "\n"
