"""A retained original with a debt table the canonical text does not carry:
the fixture the table-view boundary is proved on. Real markup shapes -- a
caption line, a scale line inside the table, a two-row heading set, a
footnote under the table -- none of it a real issuer's text."""

from __future__ import annotations

from alphalattice.evidence.alternative_evidence.documents.canonicalization import SecTableCarry
from alphalattice.kernel.live_evidence.online_sources import extract_canonical_markdown

INSTRUMENTS = (
    ("2.750% Senior Notes due 2027", "500", "500", "2027"),
    ("3.125% Senior Notes due 2029", "750", "750", "2029"),
    ("4.000% Senior Notes due 2032", "1,000", "1,000", "2032"),
    ("Term loan facility", "300", "350", "2028"),
    ("Revolving credit facility", "—", "150", "2030"),
    ("Finance lease obligations", "42", "39", "various"),
)


def debt_table_rows(count: int = len(INSTRUMENTS)) -> str:
    rows = []
    for index in range(count):
        name, current, prior, maturity = INSTRUMENTS[index % len(INSTRUMENTS)]
        suffix = "" if index < len(INSTRUMENTS) else f" (series {index})"
        rows.append(
            f"<tr><td>{name}{suffix}</td><td>$</td><td>{current}</td><td>$</td><td>{prior}</td>"
            f"<td>{maturity}</td></tr>"
        )
    return "".join(rows)


def original_html(
    *, rows: int = len(INSTRUMENTS), with_footnote: bool = True, headless_table: bool = False
) -> bytes:
    footnote = (
        "<p>(1) The revolving credit facility matures in 2030 and bears interest at SOFR plus "
        "a margin; no amounts were outstanding under it at December 31, 2025.</p>"
        if with_footnote
        else ""
    )
    # A second table of the note with no heading row a row could be read
    # over: the canonical text leaves its placeholder, the view is refused.
    headless = (
        "<p>Scheduled maturities of long-term debt are as follows:</p>"
        "<table><tr><td>2026</td><td>300</td></tr><tr><td>2027</td><td>500</td></tr>"
        "<tr><td>Thereafter</td><td>1,750</td></tr></table>"
        if headless_table
        else ""
    )
    body = f"""<html><body>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
<p>FORM 10-K</p>
<p>ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934</p>
<p>For the fiscal year ended December 31, 2025</p>
<p>PART I</p>
<p>ITEM 1. BUSINESS</p>
<p>The Company designs, manufactures and sells industrial equipment through two segments.</p>
<p>PART II</p>
<p>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</p>
<p>NOTES TO CONSOLIDATED FINANCIAL STATEMENTS</p>
<p>(In millions, except per share data)</p>
<p>NOTE 9. DEBT</p>
<p>The Company&#8217;s long-term debt consisted of the following:</p>
<table>
<tr><td colspan="6">Long-term debt (in millions)</td></tr>
<tr><td rowspan="2">Instrument</td><td colspan="2">December 31, 2025</td>
<td colspan="2">December 31, 2024</td><td rowspan="2">Maturity</td></tr>
<tr><td colspan="2">Principal</td><td colspan="2">Principal</td></tr>
{debt_table_rows(rows)}
</table>
{footnote}
{headless}
<p>In March 2025 the Company issued $1,000 million of 4.000% Senior Notes due 2032 and used
the proceeds to repay the 5.500% Senior Notes due 2025 at maturity.</p>
<p>NOTE 10. INCOME TAXES</p>
<p>The provision for income taxes was $120 million.</p>
<p>PART IV</p>
<p>ITEM 15. EXHIBITS AND FINANCIAL STATEMENT SCHEDULES</p>
<p>See the exhibit index.</p>
<p>SIGNATURES</p>
</body></html>"""
    return body.encode("utf-8")


def canonical_text(html: bytes) -> str:
    return extract_canonical_markdown(
        html, input_cap_bytes=10_000_000, tables=SecTableCarry("10-K")
    ).decode("utf-8")


def year_headed_html() -> bytes:
    """A 10-K whose note holds three uncarried tables of the refused shapes
    the development copy showed: a comparative table headed by a row of
    years alone, one headed by years beside a word, and a maturity
    schedule whose first row is data (a year beside an amount)."""

    body = """<html><body>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
<p>FORM 10-K</p>
<p>ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934</p>
<p>For the fiscal year ended December 31, 2025</p>
<p>PART I</p>
<p>ITEM 1. BUSINESS</p>
<p>The Company designs, manufactures and sells industrial equipment through two segments.</p>
<p>PART II</p>
<p>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</p>
<p>NOTES TO CONSOLIDATED FINANCIAL STATEMENTS</p>
<p>(In millions, except per share data)</p>
<p>NOTE 12. LEASES</p>
<p>The components of lease cost were as follows:</p>
<table>
<tr><td></td><td colspan="2">2025</td><td colspan="2">2024</td></tr>
<tr><td>Operating lease cost</td><td>$</td><td>32</td><td>$</td><td>34</td></tr>
<tr><td>Variable lease cost</td><td></td><td>5</td><td></td><td>4</td></tr>
</table>
<p>Premiums written by year were as follows:</p>
<table>
<tr><td></td><td colspan="2">2025</td><td colspan="2">2024</td><td>% Change</td></tr>
<tr><td>Gross premiums written</td><td>$</td><td>10,435</td><td>$</td><td>9,053</td>
<td>15.3</td></tr>
<tr><td>Net premiums written</td><td>$</td><td>8,100</td><td>$</td><td>7,200</td><td>12.5</td></tr>
<tr><td>Net premiums earned</td><td>$</td><td>7,900</td><td>$</td><td>7,100</td><td>11.3</td></tr>
</table>
<p>Scheduled maturities of long-term debt are as follows:</p>
<table>
<tr><td>2026</td><td>300</td></tr>
<tr><td>2027</td><td>500</td></tr>
<tr><td>Thereafter</td><td>1,750</td></tr>
</table>
<p>ITEM 9A. CONTROLS AND PROCEDURES</p>
<p>Our disclosure controls and procedures were effective as of December 31, 2025.</p>
<p>SIGNATURES</p>
</body></html>"""
    return body.encode("utf-8")
