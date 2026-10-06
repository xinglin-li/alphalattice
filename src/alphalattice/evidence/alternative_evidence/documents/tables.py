"""Bounded table-and-text evidence from the retained original of a filing.

The canonical text carries the tables of the disclosure items and leaves a
placeholder line where every other table stood ("[Table of 12 rows not
carried into the canonical text; the original retains it.]"). A financing
or operational fact often lives in such a table -- a debt schedule, a
maturity ladder, a share count -- beside the prose that names it. This
owner turns one placeholder into a verifiable table view: the k-th
placeholder of the canonical text is matched to the k-th table the
extractor's own walk of the retained original left uncarried
(`not_carried_tables`, the same parse, walk and decisions), proved by the
row count both state; the view is the extractor's own rendering of every
row over its headings, paged by whole rows under a byte ceiling with the
headings repeated, with the caption and scale lines the source set above
the table, the scale statement the document's structure binds to it, and
the footnote lines the canonical text carries right after the placeholder.
Nothing is summed, parsed into fields or mapped to a concept; a cell is the
source's words. A view names its parent original by content hash and the
parser it was rendered under, and is delivered only through the verified
session read, never as an ordinary canonical span.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from alphalattice.kernel.live_evidence.online_sources import (
    CANONICAL_EXTRACTION_RULES_ID,
    NOT_CARRIED_PLACEHOLDER,
    Row,
    TableCarryPolicy,
    not_carried_tables,
    table_lines,
)

from .structure import DocumentStructure

TABLE_VIEW_RULES_ID = "alternative-evidence.table-view.v2"
"""v2 (2026-09-20, the eight-topic completeness assignment, section 7): a
leading row of years, or of years beside words ("2025 | 2024", "2025 |
2024 | % Change", "(In years) | 2025 | 2024"), is the table's heading row
-- the source's own column headings, which the canonical rendering's
numeric test reads as a body row, so the table had no heading and was
refused. Measured over the 154 retained originals of the development
copy: 2,922 of 8,188 uncarried tables were refused, 1,926 of them (66%)
of this one shape; the cover-page and other mixed first rows stay
refused by name. The canonical text is untouched (`table_lines` keeps
its default), so no canonical byte and no derived set moves.
v1 (2026-09-22): a table view is rendered from the retained original
under the canonical extraction rules' own table reading
(`CANONICAL_EXTRACTION_RULES_ID`), matched to its placeholder by position
and row count, paged by whole rows under the window byte ceiling with the
headings repeated on every page, and carries the source's caption and
scale lines, the structure's scale statement and the canonical text's
footnote lines after the placeholder. A placeholder whose table cannot be
matched (no retained original, a row count that disagrees, a table without
a heading row) is a named representation gap, never a view."""

_FOOTNOTE_LINE = re.compile(
    r"^(?:\(\s*[0-9a-z]{1,3}\s*\)|[0-9]{1,2}[.)]\s|\*+|†|‡|§|_{3,}|"
    r"[a-z][.)]\s)",
    re.IGNORECASE,
)
"""How a footnote line under a table begins: a parenthesised mark, a
numbered mark, an asterisk, a dagger, a section sign, or a rule."""
_CAPTION_LEAD = re.compile(r"(?:following|as follows|consist(?:s|ed)? of|:\s*$)", re.IGNORECASE)
MAXIMUM_FOOTNOTE_LINES = 8
MAXIMUM_FOOTNOTE_CHARACTERS = 1_600
MAXIMUM_CAPTION_CHARACTERS = 300


@dataclass(frozen=True, slots=True)
class TablePlaceholder:
    """Locate one retained table's placeholder in canonical text.

    One not-carried table's place in the canonical text: the placeholder
    line's exact range, its ordinal among the placeholders (the k-th uncarried
    table of the original), the row count it states, the caption line before
    it when the source set one, the footnote lines after it, and what governs
    it -- the heading path, the family and the scale statement.
    """

    ordinal: int
    character_start: int
    character_end: int
    rows_total: int
    caption: tuple[int, int] | None
    footnotes: tuple[tuple[int, int], ...]
    path: tuple[str, ...]
    family: str
    unit_declaration: tuple[str, int, int] | None

    @property
    def region_start(self) -> int:
        """Return the placeholder's first character offset."""
        return self.caption[0] if self.caption is not None else self.character_start

    @property
    def region_end(self) -> int:
        """Return the placeholder's exclusive final character offset."""
        return self.footnotes[-1][1] if self.footnotes else self.character_end


def table_catalogue(text: str, structure: DocumentStructure) -> tuple[TablePlaceholder, ...]:
    """List retained-table placeholders in source order.

    Every not-carried table's placeholder in a canonical text, in source
    order with ordinals from 1, each with its caption, footnotes and the
    structure that governs it. No original is read here.
    """
    placeholders: list[TablePlaceholder] = []
    lines = text.split("\n")
    starts: list[int] = []
    position = 0
    for line in lines:
        starts.append(position)
        position += len(line) + 1
    for index, line in enumerate(lines):
        stripped = line.strip()
        match = NOT_CARRIED_PLACEHOLDER.match(stripped)
        if match is None:
            continue
        start = starts[index] + (len(line) - len(line.lstrip()))
        end = start + len(stripped)
        caption = _caption_before(lines, starts, index)
        footnotes = _footnotes_after(lines, starts, index)
        governing = structure.locate(start, end)
        placeholders.append(
            TablePlaceholder(
                ordinal=len(placeholders) + 1,
                character_start=start,
                character_end=end,
                rows_total=int(match.group("rows")),
                caption=caption,
                footnotes=footnotes,
                path=governing.path,
                family=governing.family,
                unit_declaration=governing.unit_declaration,
            )
        )
    return tuple(placeholders)


def _caption_before(lines: list[str], starts: list[int], index: int) -> tuple[int, int] | None:
    """Find a table caption immediately before its placeholder.

    The nearest non-empty line above the placeholder when it reads as the
    table's lead-in -- it ends with a colon or names what follows -- and is
    short; a heading or a paragraph of prose is not a caption.
    """
    for previous in range(index - 1, max(-1, index - 4), -1):
        candidate = lines[previous].strip()
        if not candidate:
            continue
        if NOT_CARRIED_PLACEHOLDER.match(candidate):
            return None
        if len(candidate) <= MAXIMUM_CAPTION_CHARACTERS and _CAPTION_LEAD.search(candidate):
            start = starts[previous] + (len(lines[previous]) - len(lines[previous].lstrip()))
            return start, start + len(candidate)
        return None
    return None


def _footnotes_after(
    lines: list[str], starts: list[int], index: int
) -> tuple[tuple[int, int], ...]:
    """Collect footnotes immediately after a table placeholder.

    The footnote lines set right after the placeholder: consecutive
    non-empty lines that begin with a footnote mark, up to a bounded count
    and length; the first line that is neither ends them.
    """
    found: list[tuple[int, int]] = []
    characters = 0
    for following in range(index + 1, min(len(lines), index + 3 * MAXIMUM_FOOTNOTE_LINES)):
        candidate = lines[following].strip()
        if not candidate:
            if found:
                # A blank line after a footnote may separate two of them.
                continue
            continue
        if not _FOOTNOTE_LINE.match(candidate) or NOT_CARRIED_PLACEHOLDER.match(candidate):
            break
        if len(found) >= MAXIMUM_FOOTNOTE_LINES or characters + len(candidate) > (
            MAXIMUM_FOOTNOTE_CHARACTERS
        ):
            break
        start = starts[following] + (len(lines[following]) - len(lines[following].lstrip()))
        found.append((start, start + len(candidate)))
        characters += len(candidate)
    return tuple(found)


@dataclass(frozen=True, slots=True)
class TableView:
    """Describe one bounded page of a retained table.

    One page of one table, rendered from the retained original: the
    rows `rows_from` to `rows_to` (1-based, inclusive) of `rows_total`,
    every row over its headings, with what the reader needs to read them
    -- the source's caption and scale lines, the structure's scale
    statement, the column headings, the footnotes -- and what remains.
    """

    rules_id: str
    parser_rules_id: str
    parent_source_content_hash: str
    ordinal: int
    rows_total: int
    rows_from: int
    rows_to: int
    headings: tuple[str, ...]
    notes: tuple[str, ...]
    caption: str
    scale: str
    footnotes: tuple[str, ...]
    lines: tuple[str, ...]
    text: str
    rows_clipped: int = 0
    """Rows of this page bounded at a word because one row alone exceeded
    the reader's ceiling: delivered in part, named in the line; a table
    holding one is never complete."""

    @property
    def remaining_rows(self) -> int:
        """Count rows after the rendered page of this table."""
        return self.rows_total - self.rows_to


class TableViewError(ValueError):
    """Name a refusal to render a retained table view.

    A placeholder that cannot become a view, by name: no retained original
    (`table_view_original_unavailable`), an original whose uncarried tables
    do not reach the placeholder's ordinal or disagree with its row count
    (`table_view_correspondence_unproved`), a table without a heading row
    (`table_view_headings_absent`), a page that begins past the table's
    rows (`table_view_rows_invalid`).
    """


def view_refusal(tables: tuple[list[Row], ...], placeholder: TablePlaceholder) -> str | None:
    """Return a stable refusal code when a table cannot be viewed.

    Why the placeholder's table could not become a view -- the refusal
    code `render_table_view` would raise -- or None when it renders: the
    original's uncarried tables do not reach the ordinal or disagree with
    the row count (`table_view_correspondence_unproved`), or the table has
    no heading row (`table_view_headings_absent`). `tables` is
    `not_carried_tables(original, policy)`, parsed once per document.
    """
    if placeholder.ordinal > len(tables):
        return "alternative_evidence.table_view_correspondence_unproved"
    rows = tables[placeholder.ordinal - 1]
    if len(rows) != placeholder.rows_total:
        return "alternative_evidence.table_view_correspondence_unproved"
    if table_lines(rows, year_headings=True) is None:
        return "alternative_evidence.table_view_headings_absent"
    return None


def render_table_view(
    original: bytes,
    *,
    parent_source_content_hash: str,
    policy: TableCarryPolicy | None,
    placeholder: TablePlaceholder,
    text: str,
    rows_from: int = 1,
    byte_ceiling: int,
) -> TableView:
    """Render one table page within the requested byte limit.

    The page of the placeholder's table that begins at `rows_from` and
    holds as many whole rows as fit `byte_ceiling` UTF-8 bytes with the
    headings, caption, scale and footnotes repeated on it; at least one row
    is always delivered, and a row alone larger than the ceiling is bounded
    at a word with the omission named.
    """
    tables = not_carried_tables(original, policy)
    if placeholder.ordinal > len(tables):
        raise TableViewError(
            "alternative_evidence.table_view_correspondence_unproved: the original's "
            f"uncarried tables ({len(tables)}) do not reach placeholder {placeholder.ordinal}"
        )
    rows = tables[placeholder.ordinal - 1]
    if len(rows) != placeholder.rows_total:
        raise TableViewError(
            "alternative_evidence.table_view_correspondence_unproved: placeholder "
            f"{placeholder.ordinal} states {placeholder.rows_total} rows, the original's "
            f"table has {len(rows)}"
        )
    rendered = table_lines(rows, year_headings=True)
    if rendered is None:
        raise TableViewError("alternative_evidence.table_view_headings_absent")
    if not 1 <= rows_from <= max(1, len(rendered.rows)):
        raise TableViewError("alternative_evidence.table_view_rows_invalid")
    caption = (
        "" if placeholder.caption is None else " ".join(text[slice(*placeholder.caption)].split())
    )
    scale = (
        ""
        if placeholder.unit_declaration is None
        else " ".join(placeholder.unit_declaration[0].split())
    )
    footnotes = tuple(" ".join(text[start:end].split()) for start, end in placeholder.footnotes)
    fixed = _frame(rendered, caption=caption, scale=scale, footnotes=footnotes)
    fixed_bytes = len("\n".join(fixed).encode("utf-8"))
    body: list[str] = []
    rows_to = rows_from - 1
    clipped = 0
    for line in rendered.rows[rows_from - 1 :]:
        candidate = line
        if body and fixed_bytes + len("\n".join([*body, candidate]).encode("utf-8")) > byte_ceiling:
            break
        if not body and fixed_bytes + len(candidate.encode("utf-8")) > byte_ceiling:
            room = max(80, byte_ceiling - fixed_bytes - 80)
            cut = candidate.encode("utf-8")[:room].decode("utf-8", errors="ignore")
            cut = cut[: cut.rfind(" ")] if " " in cut else cut
            candidate = f"{cut} [row continues: the original retains the rest]"
            clipped += 1
        body.append(candidate)
        rows_to += 1
    remaining = len(rendered.rows) - rows_to
    lines = (
        *fixed[: fixed.index("--")],
        *body,
        *fixed[fixed.index("--") + 1 :],
        f"[rows {rows_from}-{rows_to} of {len(rendered.rows)}"
        + (f"; {remaining} more row(s) not in this page]" if remaining else "]"),
    )
    return TableView(
        rules_id=TABLE_VIEW_RULES_ID,
        parser_rules_id=CANONICAL_EXTRACTION_RULES_ID,
        parent_source_content_hash=parent_source_content_hash,
        ordinal=placeholder.ordinal,
        rows_total=len(rendered.rows),
        rows_from=rows_from,
        rows_to=rows_to,
        headings=rendered.headings,
        notes=rendered.notes,
        caption=caption,
        scale=scale,
        footnotes=footnotes,
        lines=lines,
        text="\n".join(lines),
        rows_clipped=clipped,
    )


def _frame(rendered: object, *, caption: str, scale: str, footnotes: tuple[str, ...]) -> list[str]:
    """Frame rendered rows with caption, scale, and footnotes.

    The lines every page repeats around its rows, with a `--` marker
    where the rows go: the caption, the source's own lines inside the table,
    the scale statement, the column headings; then the footnotes.
    """
    from alphalattice.kernel.live_evidence.online_sources import TableLines

    assert isinstance(rendered, TableLines)
    head: list[str] = []
    if caption:
        head.append(f"Caption: {caption}")
    for note in rendered.notes:
        head.append(f"Table note: {note}")
    if scale:
        head.append(f"Scale statement (from the document's structure): {scale}")
    if rendered.headings:
        head.append("Columns: " + " | ".join(rendered.headings))
    tail = [f"Footnote: {value}" for value in footnotes]
    return [*head, "--", *tail]


__all__ = [
    "MAXIMUM_FOOTNOTE_LINES",
    "TABLE_VIEW_RULES_ID",
    "TablePlaceholder",
    "TableView",
    "TableViewError",
    "render_table_view",
    "table_catalogue",
    "view_refusal",
]
