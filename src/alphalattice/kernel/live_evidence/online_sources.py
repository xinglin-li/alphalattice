"""Private source-specific acquisition and HTML extraction primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from alphalattice.kernel.live_evidence.errors import LiveEvidenceError

BLS_ARCHIVE_INDEX_TEMPLATE = "https://www.bls.gov/bls/news-release/{release_id}.htm"
BLS_ARCHIVE_PATH = re.compile(
    r"^/news\.release/archives/(?P<release>cpi|empsit)_(?P<date>[0-9]{8})\.htm$"
)
BLS_INDEX_CAP_BYTES = 2 * 1024 * 1024
BLS_RELEASE_CAP_BYTES = 16 * 1024 * 1024
SEC_SUBMISSION_CAP_BYTES = 64 * 1024 * 1024
SEC_PRIMARY_CAP_BYTES = 32 * 1024 * 1024
CANONICAL_MARKDOWN_CAP_BYTES = 8 * 1024 * 1024
MAX_TREE_SIZE = 500_000
CANONICAL_EXTRACTION_RULES_ID = "live-evidence.canonical-markdown.v6"
"""v6 (2026-09-21): a paragraph set as a `div` that carries a cross-reference
link inside its prose is a paragraph (`_linked_paragraph_divs`). The
extractor keeps such a div a `div` (its link is kept as a reference for
boilerplate detection) and its Markdown writer closes a `div` with no line
break, so the block after it -- a note heading, the next paragraph -- was
glued to the paragraph's last sentence: COST's 10-K read "Please see Note 1
for additional information. Note 4—Debt" on one line and the structure
saw no debt note; every paragraph after a linked one in a filing set as
divs was joined the same way. A block whose text is mostly links (a table
of contents set as divs) is left to the extractor's own reading.
v5 (2026-09-19): a Markdown heading longer than a heading can be
(`_HEADING_BLOCK_CHARACTERS`) is a paragraph the source styled as one --
an exhibit-index footnote set as a heading tag -- and is carried as a
paragraph (`_demote_long_headings`); as a heading it named every chunk
after it and exceeded the citation contract's 256-character heading, which
failed the retrieval build of a whole unit (the live campaign's u05, MSTR's
and NEE's exhibit indexes). v4 (2026-09-18): a table of ordinary size in a section the caller's
`TableCarryPolicy` admits reaches the canonical text as compact rows that
each carry their own column headings (a repurchase table's "Period: ...;
Total Number of Shares Purchased: ...; Average Price Paid per Share: $...")
in the table's own place; a one-row layout table becomes the line it
displays ("ITEM 1. LEGAL PROCEEDINGS.") everywhere; every other table --
too large, too wide, without a heading row, a table of contents, or in a
section the policy does not admit -- is replaced by one line that says a
table was not carried. Every table was pruned whole before, so the issuer
repurchase table that answers Part II Item 2(c) never reached a packet and
a heading set as a two-cell table vanished. In a filing that sets most
paragraphs as `p`, a paragraph set as a `div` of inline content is read
as a `p` (`_paragraph_divs`): DG's 10-Q disclosure-controls conclusion was
such a paragraph and was lost. A cell longer than the
carried cell size is cut at a word and says in the cell how many
characters the original retains (`_bounded_cell`); a spanning cell
occupies its grid positions in every row it spans (`_table_rows`)."""
"""v2: a table with a single cell is a layout device, not a table; its
content is kept in the narrative (see `_unwrap_layout_tables`). v1 pruned
every table, and filings that set an inline-XBRL amount in a one-cell table
lost the number ("... not to exceed $" with nothing after it). An
extraction rules change rotates only the derived text of documents
extracted after it; recorded canonical text keeps its identity."""
MAX_ARCHIVE_HREFS = 4_096
MAX_ARCHIVE_HREF_CODEPOINTS = 2_048


def _schema(message: str) -> LiveEvidenceError:
    return LiveEvidenceError(message, code="live_evidence.schema_invalid")


_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _demote_long_headings(canonical: str) -> str:
    """A heading line longer than a heading can be is carried as the
    paragraph it is; the marker goes, the text stays where it was."""

    lines = canonical.split("\n")
    for index, line in enumerate(lines):
        match = _MARKDOWN_HEADING.match(line)
        if match is not None and len(match.group(2)) > _HEADING_BLOCK_CHARACTERS:
            lines[index] = match.group(2)
    return "\n".join(lines)


def extract_canonical_markdown(
    html: bytes,
    *,
    input_cap_bytes: int,
    tables: TableCarryPolicy | None = None,
) -> bytes:
    """Extract deterministic narrative Markdown from already bounded HTML.

    `tables` decides which tables are carried as rows (see
    `TableCarryPolicy`); without a policy no table is carried and each
    leaves a line saying so, while one-row layout tables still become the
    line they display.

    Args:
        html: Already acquired HTML bytes; this operation makes no network request.
        input_cap_bytes: Maximum admitted HTML byte count for the source.
        tables: Section-aware table carry policy, or None to omit ordinary tables.

    Returns:
        Bounded UTF-8 Markdown with normalized LF and a final newline.

    Raises:
        LiveEvidenceError: Input/output exceeds its cap, contains NUL, cannot be extracted,
            or yields no narrative.
    """
    if len(html) > input_cap_bytes:
        raise _schema("online evidence HTML exceeds the source cap")
    if b"\x00" in html:
        raise _schema("online evidence HTML contains a NUL byte")

    # The import is deliberately private to the optional live-evidence profile.
    from trafilatura import extract
    from trafilatura.settings import use_config

    config = use_config()
    config["DEFAULT"]["MAX_TREE_SIZE"] = str(MAX_TREE_SIZE)

    try:
        extracted = extract(
            _carry_tables(_unwrap_layout_tables(_unwrap_inline_xbrl(html)), tables),
            output_format="markdown",
            include_comments=False,
            include_tables=False,
            include_images=False,
            include_links=False,
            with_metadata=False,
            deduplicate=False,
            fast=False,
            prune_xpath="//table",
            config=config,
        )
    except (MemoryError, RecursionError, UnicodeError, ValueError) as exc:
        raise _schema("online evidence HTML extraction failed") from exc
    if extracted is None:
        raise _schema("online evidence HTML has no extractable narrative")
    canonical = _demote_long_headings(extracted.replace("\r\n", "\n").replace("\r", "\n").strip())
    if not canonical:
        raise _schema("online evidence HTML has no extractable narrative")
    if "\x00" in canonical:
        raise _schema("canonical Markdown contains a NUL character")
    payload = (canonical + "\n").encode("utf-8")
    if len(payload) > CANONICAL_MARKDOWN_CAP_BYTES:
        raise _schema("canonical Markdown exceeds the bounded policy")
    return payload


# The header is removed whole, then a hidden block set outside one (not the
# specification's placement, but never displayed either): each pattern
# closes at its own closing tag, so the references and resources inside a
# header are removed with it rather than unwrapped into prose.
_IX_HIDDEN_BLOCKS = (
    re.compile(rb"<ix:header\b[^>]*>.*?</ix:header\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(rb"<ix:hidden\b[^>]*>.*?</ix:hidden\s*>", re.IGNORECASE | re.DOTALL),
)
_IX_WRAPPER = re.compile(rb"</?ix:[A-Za-z]+\b[^>]*>", re.IGNORECASE)


def _unwrap_inline_xbrl(html: bytes) -> bytes:
    """Drop the hidden inline-XBRL header and unwrap every displayed inline-XBRL
    element to the text it displays, on the bytes, before any parser.

    Inline XBRL is markup over the displayed text: `ix:nonFraction`,
    `ix:nonNumeric`, `ix:continuation` and the rest wrap what the filing
    shows and add nothing visible, while `ix:header` holds the hidden facts,
    contexts and units the filing never displays. The extractor treated both
    wrongly: an unknown inline element inside a paragraph lost its tail, so
    a sentence read "was approximately $" and everything after the amount
    vanished (DG's retrieved 10-K: 76 of 145 narrative amounts, 27 sentences
    left at "$"); and the hidden header was emitted as one line of prose,
    so a filing's canonical text carried its undisplayed facts. Now the
    header is removed whole -- hidden facts are never prose -- and the
    wrappers are unwrapped so the sentence stays one sentence with its
    amount and its sign exactly as displayed.
    """

    for hidden in _IX_HIDDEN_BLOCKS:
        html = hidden.sub(b"", html)
    return _IX_WRAPPER.sub(b"", html)


_INNERMOST_TABLE = re.compile(rb"<table\b(?:(?!<table\b).)*?</table\s*>", re.IGNORECASE | re.DOTALL)
_CELL_OPEN = re.compile(rb"<t[dh]\b", re.IGNORECASE)
_CELL_CONTENT = re.compile(rb"<t[dh]\b[^>]*>(.*?)</t[dh]\s*>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(rb"<[^>]+>")
_MAXIMUM_UNWRAP_PASSES = 8


def _unwrap_layout_tables(html: bytes) -> bytes:
    """Replace every table that holds a single cell with that cell's text.

    Filing agents set an inline-XBRL amount in a one-cell table inside a
    sentence. An HTML parser closes the paragraph at the table and the
    extractor prunes tables, so the number is lost and the currency sign is
    left alone. The replacement happens on the bytes, before any parser sees
    the markup, so the sentence stays one sentence; a table with more than
    one cell stays a table and is pruned as before. Innermost tables are
    unwrapped first, a bounded number of times, so a layout cell inside a
    real table still counts toward that table's cells.
    """

    def unwrap(match: re.Match[bytes]) -> bytes:
        table = match.group(0)
        if len(_CELL_OPEN.findall(table)) != 1:
            return table
        cell = _CELL_CONTENT.search(table)
        if cell is None:
            return table
        text = b" ".join(_TAG.sub(b" ", cell.group(1)).split())
        return b" " + text + b" " if text else b" "

    current = html
    for _ in range(_MAXIMUM_UNWRAP_PASSES):
        replaced = _INNERMOST_TABLE.sub(unwrap, current)
        if replaced == current:
            break
        current = replaced
    return current


TABLE_ROW_LIMIT = 20
TABLE_COLUMN_LIMIT = 8
_TABLE_CELL_LIMIT = 400
_LAYOUT_LINE_CELLS = 4
_LAYOUT_LINE_CHARACTERS = 240
_HEADING_BLOCK_CHARACTERS = 200
_NUMERIC_CELL = re.compile(
    r"^[\s$()\-\u2013\u2014\d.,%/:]*\d[\s$()\-\u2013\u2014\d.,%/:]*$|^[\-\u2013\u2014]+$"
)
_CONTENTS_ROW = re.compile(r"^(?:part|item)\s+[0-9ivx]+", re.IGNORECASE)
_INVISIBLE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_BLOCK_TAGS = frozenset({"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"})

Row = list[tuple[int, int, str]]
"""One table row as `(physical column, span, text)` cells."""


class TableCarryPolicy(Protocol):
    """Which tables a document carries into its canonical text.

    The extractor walks the markup in source order, tells the policy every
    short block of text it passes (`observe`) -- the Part and Item headings
    among them -- and asks before each table whether a table standing here
    is carried (`carries`). The policy owns the section vocabulary and its
    state; the extractor owns the rendering.
    """

    def observe(self, text: str) -> None: ...

    def carries(self) -> bool: ...


def _local(tag: object) -> str:
    return tag.lower().rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _cell_text(cell: object) -> str:
    from lxml import etree

    element = cell
    # Text nodes joined with a space: a heading broken over lines inside the
    # cell ("Total Number" / "of Shares") stays two words.
    text = " ".join(element.itertext()) if isinstance(element, etree._Element) else ""
    return " ".join(_INVISIBLE.sub("", text).replace("\xa0", " ").split())


def _bounded_cell(text: str) -> str:
    """A cell within the carried cell size, whole; a longer cell as far as
    the size, cut at a word, saying in the cell how much the original
    retains -- never a silent cut that leaves a deadline, an exception or a
    unit past the 400th character reading as complete."""

    if len(text) <= _TABLE_CELL_LIMIT:
        return text
    cut = text.rfind(" ", _TABLE_CELL_LIMIT // 2, _TABLE_CELL_LIMIT)
    if cut == -1:
        cut = _TABLE_CELL_LIMIT
    kept = text[:cut].rstrip()
    omitted = len(text) - len(kept)
    return (
        f"{kept} [cell continues: {omitted} more characters not carried; the original retains them]"
    )


def _span_count(cell: object, attribute: str) -> int:
    from lxml import etree

    if not isinstance(cell, etree._Element):
        return 1
    try:
        return max(1, int(cell.get(attribute) or 1))
    except ValueError:
        return 1


def _take_occupied(
    occupied: dict[int, list[object]], opened: set[int], column: int, cells: Row
) -> int:
    """Carry the cells that span into this row from the rows above over the
    columns they occupy, and return the next free column."""

    while column in occupied and column not in opened:
        _left, span, text = occupied[column]
        if text:
            cells.append((column, int(span), str(text)))  # type: ignore[arg-type]
        column += int(span)  # type: ignore[arg-type]
    return column


def _table_rows(table: object) -> list[Row]:
    """Every row of one table as `(physical column, span, text)` cells, the
    physical column counted through `colspan` so a heading that spans three
    layout columns still covers the data cells set under it, and through
    `rowspan` so a cell that spans rows occupies its columns in each of
    them -- a two-row heading set keeps "High" under "Price Per Share", a
    holder spanning two dated lots stands in both. Blank cells are left out
    (they still occupy their columns) and a row of nothing but blanks is
    no row. A cell longer than the carried cell size says so in the cell."""

    from lxml import etree

    rows: list[Row] = []
    if not isinstance(table, etree._Element):
        return rows
    # Columns a spanning cell still occupies in the rows below: column ->
    # [rows left, column span, text].
    occupied: dict[int, list[object]] = {}
    for tr in table.iter():
        if _local(tr.tag) != "tr":
            continue
        column = 0
        cells: Row = []
        opened: set[int] = set()
        for td in tr:
            if _local(td.tag) not in {"td", "th"}:
                continue
            column = _take_occupied(occupied, opened, column, cells)
            span = _span_count(td, "colspan")
            text = _bounded_cell(_cell_text(td))
            if text:
                cells.append((column, span, text))
            rows_spanned = _span_count(td, "rowspan")
            if rows_spanned > 1:
                occupied[column] = [rows_spanned - 1, span, text]
                opened.add(column)
            column += span
        _take_occupied(occupied, opened, column, cells)
        for start in list(occupied):
            if start in opened:
                continue
            occupied[start][0] = int(occupied[start][0]) - 1  # type: ignore[arg-type]
            if int(occupied[start][0]) <= 0:  # type: ignore[arg-type]
                del occupied[start]
        if cells:
            rows.append(cells)
    return rows


def _layout_line(rows: list[Row]) -> str | None:
    """The line a one-row table of a few words is: a heading set as two
    cells ("ITEM 1." | "LEGAL PROCEEDINGS."), a check box and its label. A
    row that holds a number is a table, not a layout device."""

    if len(rows) != 1 or len(rows[0]) > _LAYOUT_LINE_CELLS:
        return None
    if sum(len(text) for _c, _s, text in rows[0]) > _LAYOUT_LINE_CHARACTERS:
        return None
    if any(_NUMERIC_CELL.match(text) for _c, _s, text in rows[0]):
        return None
    return " ".join(text for _c, _s, text in rows[0])


def _short_text(element: object) -> str | None:
    """The text of a block when it is short enough to be a heading, read no
    further than needed: a wrapper around a whole document is left alone
    after a few hundred characters."""

    from lxml import etree

    if not isinstance(element, etree._Element):
        return None
    pieces: list[str] = []
    total = 0
    for piece in element.itertext():
        pieces.append(piece)
        total += len(piece)
        if total > 4 * _HEADING_BLOCK_CHARACTERS:
            return None
    text = " ".join("".join(pieces).replace("\xa0", " ").split())
    if not text or len(text) > _HEADING_BLOCK_CHARACTERS:
        return None
    return text


NOT_CARRIED_PLACEHOLDER = re.compile(
    r"^\[Table of (?P<rows>\d+) rows not carried into the canonical text; "
    r"the original retains it\.\]$"
)
"""The line a table the canonical text does not carry leaves in its place:
its row count, so a view of the retained original can be matched to it."""


def _not_carried(rows: list[Row]) -> list[str]:
    return [
        f"[Table of {len(rows)} rows not carried into the canonical text; the original retains it.]"
    ]


@dataclass(frozen=True, slots=True)
class TableLines:
    """Narrative table rendering with notes, headings and source-order body rows.

    One table as lines a reader can take row by row: the caption and
    scale lines set inside the table (`notes`), the physical column
    headings (`headings`, by column), and every body row rendered over its
    headings (`rows`), in source order; `heading_rows` is how many heading
    rows the table set. Nothing is summed, ordered or interpreted.

    Attributes:
        notes: Caption and scale lines found before the body.
        headings: Deduplicated physical-column headings in column order.
        rows: Rendered body lines in source order; no numerical calculation is applied.
        heading_rows: Number of rows used to construct the headings.
    """

    notes: tuple[str, ...]
    headings: tuple[str, ...]
    rows: tuple[str, ...]
    heading_rows: int


_YEAR_CELL = re.compile(r"^(?:fiscal\s+|fy\s*)?(?:19|20)\d{2}[a-z]?$", re.IGNORECASE)
"""A cell that is a year: the column heading of a comparative table
("2025 | 2024"), numeric to `_NUMERIC_CELL` and so read as a body cell by
the canonical rendering. `table_lines` reads a leading row of years (or
of years beside words) as a heading row only when asked
(`year_headings`), so a view can read what the canonical text left."""


def _year_heading_row(row: Row) -> bool:
    """A multi-cell row before the body whose every non-blank cell is a
    year or a word (no other number): the source's column headings."""

    texts = [t.strip() for _c, _s, t in row if t.strip()]
    return (
        len(row) > 1
        and bool(texts)
        and any(_YEAR_CELL.match(t) for t in texts)
        and all(_YEAR_CELL.match(t) or not _NUMERIC_CELL.match(t) for t in texts)
    )


def table_lines(rows: list[Row], *, year_headings: bool = False) -> TableLines | None:
    """Render table rows over detected headings without canonical carry-size limits.

    The lines of a table of any size -- the rendering `_render_table`
    applies to a carried table, without its row and column limits -- or None
    when the table has no heading row a row could be read over. With
    `year_headings`, a leading row of years, or of years beside words, is a
    heading row too (the table view's reading; the canonical rendering
    keeps the default, so no canonical text moves).

    Args:
        rows: Source-order cells represented by physical column, span and text.
        year_headings: Whether leading year rows may serve as headings in this view.

    Returns:
        Rendered table, or None when no heading row can be identified.
    """
    notes: list[str] = []
    heading_rows: list[Row] = []
    body: list[Row] = []
    for row in rows:
        if not body and len(row) == 1 and not heading_rows:
            notes.append(row[0][2])
        elif not body and (
            (len(row) > 1 and not any(_NUMERIC_CELL.match(t) for _c, _s, t in row))
            or (year_headings and _year_heading_row(row))
        ):
            heading_rows.append(row)
        else:
            body.append(row)
    if not body and len(heading_rows) > 1:
        body, heading_rows = heading_rows[1:], heading_rows[:1]
    if not heading_rows:
        return None
    headings: dict[int, str] = {}
    for row in heading_rows:
        for column, span, text in row:
            for physical in range(column, column + span):
                existing = headings.get(physical, "")
                if text == existing:
                    continue
                headings[physical] = (existing + " " + text).strip()
    header_texts = {tuple(t for _c, _s, t in row) for row in heading_rows}
    lines: list[str] = []
    for row in body:
        if len(row) == 1:
            lines.append(row[0][2])
            continue
        if tuple(t for _c, _s, t in row) in header_texts:
            continue
        values: list[tuple[str, str]] = []
        for column, _span, text in row:
            heading = headings.get(column, "")
            if heading and (not values or values[-1][0] != heading):
                values.append((heading, text))
            elif values:
                previous_heading, previous = values[-1]
                values[-1] = (previous_heading, f"{previous} {text}".strip())
            else:
                values.append(("", text))
        rendered = []
        for heading, text in values:
            if not text.strip("$\u20ac\u00a3 "):
                continue
            value = re.sub(r"\$\s+(?=\d|\()", "$", text)
            value = re.sub(r"\(\s+", "(", value)
            value = re.sub(r"\s+\)", ")", value)
            value = re.sub(r"(\d)\s+%", r"\1%", value)
            rendered.append(f"{heading}: {value}" if heading else value)
        if rendered:
            lines.append("; ".join(rendered))
    return TableLines(
        notes=tuple(notes),
        headings=tuple(dict.fromkeys(headings[key] for key in sorted(headings))),
        rows=tuple(lines),
        heading_rows=len(heading_rows),
    )


def _render_table(rows: list[Row], *, carry: bool) -> list[str]:
    """The lines a table becomes, or a single line saying it was not carried.

    A one-row layout table is the line it displays wherever it stands. A
    table of contents leaves a line saying so. Elsewhere a table is carried
    only where the policy admits it and when it is of ordinary size with a
    heading row: the heading rows are the leading rows that hold words and
    no number (a one-cell caption or unit line above or among them is kept
    in its place); every other row is rendered as `heading: value; heading:
    value` over the headings that cover its cells' physical columns, so a
    row read on its own still says what each value is. A cell under the
    heading of the value before it (a currency sign set in its own cell) or
    under no heading (a footnote mark, a spacer) joins that value; a blank
    cell says nothing; a dash stays a dash; a heading row repeated after a
    page break is skipped, never read as a record. Nothing is summed,
    ordered or interpreted.
    """

    if not rows:
        return []
    line = _layout_line(rows)
    if line is not None:
        return [line]
    labelled = sum(1 for row in rows if _CONTENTS_ROW.match(row[0][2]))
    if len(rows) >= 4 and labelled * 2 >= len(rows):
        return [f"[Table of contents ({len(rows)} rows) not carried into the canonical text.]"]
    if not carry or len(rows) > TABLE_ROW_LIMIT:
        return _not_carried(rows)
    notes: list[str] = []
    heading_rows: list[Row] = []
    body: list[Row] = []
    for row in rows:
        if not body and len(row) == 1 and not heading_rows:
            notes.append(row[0][2])
        elif not body and len(row) > 1 and not any(_NUMERIC_CELL.match(t) for _c, _s, t in row):
            heading_rows.append(row)
        else:
            body.append(row)
    if not body and len(heading_rows) > 1:
        body, heading_rows = heading_rows[1:], heading_rows[:1]
    if not heading_rows:
        # A row without headings cannot be read on its own.
        return _not_carried(rows)
    headings: dict[int, str] = {}
    for row in heading_rows:
        for column, span, text in row:
            for physical in range(column, column + span):
                existing = headings.get(physical, "")
                if text == existing:
                    # A heading that spans two heading rows names its
                    # column once.
                    continue
                headings[physical] = (existing + " " + text).strip()
    distinct = list(dict.fromkeys(headings.values()))
    if len(distinct) > TABLE_COLUMN_LIMIT:
        return _not_carried(rows)
    header_texts = {tuple(t for _c, _s, t in row) for row in heading_rows}
    lines: list[str] = list(notes)
    for row in body:
        if len(row) == 1:
            lines.append(row[0][2])
            continue
        if tuple(t for _c, _s, t in row) in header_texts:
            continue
        values: list[tuple[str, str]] = []
        for column, _span, text in row:
            heading = headings.get(column, "")
            if heading and (not values or values[-1][0] != heading):
                values.append((heading, text))
            elif values:
                previous_heading, previous = values[-1]
                values[-1] = (previous_heading, f"{previous} {text}".strip())
            else:
                values.append(("", text))
        rendered = []
        for heading, text in values:
            if not text.strip("$\u20ac\u00a3 "):
                # A currency sign over a blank cell says nothing.
                continue
            value = re.sub(r"\$\s+(?=\d|\()", "$", text)
            value = re.sub(r"\(\s+", "(", value)
            value = re.sub(r"\s+\)", ")", value)
            value = re.sub(r"(\d)\s+%", r"\1%", value)
            rendered.append(f"{heading}: {value}" if heading else value)
        if rendered:
            lines.append("; ".join(rendered))
    # A table whose rows all say nothing is a gap, not an absence.
    return lines or _not_carried(rows)


def _parsed_tree(html: bytes) -> object | None:
    from lxml import etree

    # Bytes that are valid UTF-8 are read as UTF-8 whether or not the markup
    # declares a charset; otherwise the parser reads the declaration as the
    # extractor will. The output is UTF-8, which the extractor detects first.
    try:
        html.decode("utf-8")
        encoding: str | None = "utf-8"
    except UnicodeDecodeError:
        encoding = None
    parser = etree.HTMLParser(recover=True, huge_tree=True, encoding=encoding)
    try:
        tree = etree.fromstring(html, parser)
    except (etree.XMLSyntaxError, ValueError):
        return None
    return tree


def _table_decisions(tree: object, policy: TableCarryPolicy | None) -> dict[object, bool]:
    """Every table of the tree and whether it is carried: a walk in source
    order shows the policy each short block and decides each table; a table
    nested in a decided table takes its decision."""

    from lxml import etree

    decisions: dict[etree._Element, bool] = {}
    if not isinstance(tree, etree._Element):
        return decisions
    stack: list[etree._Element] = list(reversed(list(tree)))
    while stack:
        element = stack.pop()
        tag = _local(element.tag)
        if not tag:
            continue
        if tag == "table":
            line = _layout_line(_table_rows(element))
            if line is None:
                carry = policy is not None and policy.carries()
            else:
                carry = False
                if policy is not None:
                    policy.observe(line)
            for inner in element.iter():
                if _local(inner.tag) == "table":
                    decisions[inner] = carry
            continue
        if tag in _BLOCK_TAGS and policy is not None:
            text = _short_text(element)
            if text is not None:
                policy.observe(text)
        stack.extend(reversed([child for child in element if isinstance(child.tag, str)]))
    return decisions


def not_carried_tables(html: bytes, policy: TableCarryPolicy | None) -> tuple[list[Row], ...]:
    """Read omitted table rows in canonical placeholder order.

    The rows of every table the canonical text of this markup leaves a
    not-carried placeholder for, in the placeholders' order -- the same
    parse, walk and decisions as `_carry_tables`, so the k-th placeholder
    of the canonical text is the k-th table here. A table nested inside a
    table that is itself not carried leaves no placeholder of its own (its
    parent's cell text is discarded with the parent) and is left out; a
    layout line, a contents table and a carried table leave none.

    Args:
        html: Already acquired HTML markup to parse locally.
        policy: Section-aware carry policy, or None to use closed table admission.

    Returns:
        Source-order omitted tables, excluding nested omissions beneath an omitted parent.
    """
    from lxml import etree

    tree = _parsed_tree(html)
    if tree is None or not isinstance(tree, etree._Element):
        return ()
    decisions = _table_decisions(tree, policy)
    found: list[list[Row]] = []
    for table in tree.iter():
        if _local(table.tag) != "table" or table not in decisions:
            continue
        rows = _table_rows(table)
        lines = _render_table(rows, carry=decisions[table])
        if lines != _not_carried(rows):
            continue
        if any(
            _local(ancestor.tag) == "table"
            and ancestor in decisions
            and _render_table(_table_rows(ancestor), carry=decisions[ancestor])
            == _not_carried(_table_rows(ancestor))
            for ancestor in table.iterancestors()
        ):
            continue
        found.append(rows)
    return tuple(found)


def _carry_tables(html: bytes, policy: TableCarryPolicy | None) -> bytes:
    """Replace every table with the lines it becomes before the extractor
    prunes tables: a walk in source order shows the policy each short block
    and decides each table (`_table_decisions`); the tables are then
    rendered innermost first, so a table nested in a cell is rendered before
    its parent reads that cell's text. On the same tree a paragraph set as a
    `div` of inline content becomes a `p` (see `_paragraph_divs`)."""

    from lxml import etree

    if b"<table" not in html and b"<TABLE" not in html and b"<div" not in html:
        return html
    tree = _parsed_tree(html)
    if tree is None or not isinstance(tree, etree._Element):
        return html
    decisions = _table_decisions(tree, policy)
    tables = sorted(decisions, key=lambda element: -len(list(element.iterancestors())))
    for table in tables:
        _replace_table(table, _render_table(_table_rows(table), carry=decisions[table]))
    converted = _paragraph_divs(tree) + _linked_paragraph_divs(tree)
    if not decisions and not converted:
        return html
    return etree.tostring(tree, method="html", encoding="utf-8")


_STRUCTURAL_TAGS = frozenset(
    {
        "p",
        "div",
        "table",
        "ul",
        "ol",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
        "hr",
        "section",
        "article",
        "header",
        "footer",
        "nav",
        "aside",
        "form",
        "dl",
        "dt",
        "dd",
    }
)


def _paragraph_divs(tree: object) -> int:
    """In a document that sets most of its paragraphs as `p`, make every
    `div` that holds only inline content a `p` too, and say how many. The
    extractor reads `div` paragraphs only in a document with too few `p`
    paragraphs, so a filing that sets most paragraphs as `p` and a few as
    `div` (DG's 10-Q: "(a) Disclosure Controls and Procedures. Under the
    supervision ... concluded that our disclosure controls and procedures
    were effective") lost those few, a conclusion among them. A filing that
    sets its paragraphs as `div` is left whole to the extractor's div path,
    which reads them: converting them all lost footnotes and short
    paragraphs the `p` path discards (measured on NEE's and BKNG's 10-Ks).
    A `div` with a structural descendant is a container and stays one; a
    `div` with no text is layout and stays one."""

    from lxml import etree

    if not isinstance(tree, etree._Element):
        return 0
    paragraphs = 0
    candidates: list[etree._Element] = []
    for element in tree.iter():
        tag = _local(element.tag)
        if tag == "p":
            if "".join(element.itertext()).strip():
                paragraphs += 1
            continue
        if tag != "div":
            continue
        if any(_local(child.tag) in _STRUCTURAL_TAGS for child in element.iterdescendants()):
            continue
        if "".join(element.itertext()).strip():
            candidates.append(element)
    if not candidates or paragraphs < len(candidates):
        return 0
    for element in candidates:
        element.tag = "p"
    return len(candidates)


_LINKED_PROSE_LINK_SHARE = 3
"""A block is prose with a reference inside it when its links hold at most a
third of its characters; a block that is mostly links is navigation."""


def _linked_paragraph_divs(tree: object) -> int:
    """Make every `div` that holds only inline content, with prose and a
    cross-reference link inside it, a `p`, and say how many. The extractor
    keeps a linked div a `div` -- the link stays a reference so its link
    density can be judged -- and its Markdown writer closes a `div` with no
    line break, so whatever block follows is glued to the paragraph's last
    sentence. Prose is decided by share: a block whose links hold more than
    a third of its characters is a table of contents or a navigation line
    and keeps the extractor's own reading. A `div` with a structural
    descendant is a container and stays one."""

    from lxml import etree

    if not isinstance(tree, etree._Element):
        return 0
    converted = 0
    for element in tree.iter("div"):
        if any(_local(child.tag) in _STRUCTURAL_TAGS for child in element.iterdescendants()):
            continue
        links = [child for child in element.iterdescendants() if _local(child.tag) == "a"]
        if not links:
            continue
        text = " ".join("".join(element.itertext()).split())
        if not text:
            continue
        linked = sum(len(" ".join("".join(link.itertext()).split())) for link in links)
        if linked * _LINKED_PROSE_LINK_SHARE > len(text):
            continue
        element.tag = "p"
        converted += 1
    return converted


_INLINE_TAGS = frozenset(
    {"span", "a", "b", "i", "u", "em", "strong", "font", "sup", "sub", "small", "br"}
)


def _replace_table(table: object, lines: list[str]) -> None:
    """Put the lines in the table's place as paragraphs, and wrap the inline
    text that stood beside the table in its parent into paragraphs of its
    own: the extractor drops a bare span that follows a block, and a cover
    line set beside a check-box table ("For the quarterly period ended June
    30, 2026") was lost that way."""

    from lxml import etree

    if not isinstance(table, etree._Element):
        return
    parent = table.getparent()
    if parent is None:
        return
    replacement = etree.Element("div")
    for line in lines:
        paragraph = etree.SubElement(replacement, "p")
        paragraph.text = line
    before: list[etree._Element] = []
    previous = table.getprevious()
    while previous is not None and _local(previous.tag) in _INLINE_TAGS:
        before.append(previous)
        previous = previous.getprevious()
    leading_text = parent.text if previous is None else previous.tail
    after: list[etree._Element] = []
    following = table.getnext()
    while following is not None and _local(following.tag) in _INLINE_TAGS:
        after.append(following)
        following = following.getnext()
    trailing_text = table.tail
    replacement.tail = None
    parent.replace(table, replacement)
    if before or (leading_text and leading_text.strip()):
        paragraph = etree.Element("p")
        paragraph.text = leading_text
        for element in reversed(before):
            parent.remove(element)
            paragraph.append(element)
        if previous is None:
            parent.text = None
        else:
            previous.tail = None
        replacement.addprevious(paragraph)
    if after or (trailing_text and trailing_text.strip()):
        paragraph = etree.Element("p")
        paragraph.text = trailing_text
        for element in after:
            parent.remove(element)
            paragraph.append(element)
        replacement.addnext(paragraph)


__all__: list[str] = []
