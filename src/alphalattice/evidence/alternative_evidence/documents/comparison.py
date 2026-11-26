"""Conservative temporal comparison of disclosure units across filings.

A filing restates what an earlier filing said: a legal proceeding is
described again in the next quarter, a debt note repeats its instruments
with new balances, an 8-K item is summarized in the 10-Q. This owner aligns
a later filing's units with an earlier filing's by the units' own
identifiers -- the family that inventoried them and the title the source
gave them, plus a case number or a stated alias when the source states one
-- and compares the aligned texts with `difflib` under a size bound. What
it states is a machine fact about the texts: an exact repeat (the same
normalized text, so one read serves both and the other is a pointer with
its own source and time), a changed aligned unit (the same identifier, a
different text, with how many characters changed), a unit first observed
in the later filing, a unit of the earlier filing that no later unit
aligns with, or a correspondence the bound left unresolved. It never
infers that a matter was resolved, that a risk changed, or that absence
means anything: absence in a later filing is absence, reported as such.
Source times stay with the sources; nothing here assigns an instant.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

COMPARISON_RULES_ID = "alternative-evidence.temporal-comparison.v1"
"""v1 (2026-09-22): alignment by (family, normalized title, stated case
number or alias) between a filing and the previous filing of the same
issuer that inventoried the same family; exact repeats by normalized text
hash; changed aligned units by `SequenceMatcher(autojunk=False)` on texts
of at most `COMPARISON_TEXT_BOUND` characters each; everything larger is
compared by hash only and left unresolved when the hashes differ."""

COMPARISON_TEXT_BOUND = 24_000
"""Characters per side beyond which the character comparison is not run:
`SequenceMatcher` is quadratic in the worst case, and a unit longer than
this is a region, not a statement."""

ALIGNMENT_RATIO_FLOOR = 0.5
"""Below this similarity two units with the same identifier are not read as
one changed unit: the title repeated, the text did not."""

_WHITESPACE = re.compile(r"\s+")
_EMPHASIS = re.compile(r"[*_`]+")
_AMOUNT = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")
_SMALL = frozenset({"the", "of", "and", "a", "an", "in", "for", "to", "on", "et", "al", "v"})


@dataclass(frozen=True, slots=True)
class ComparableUnit:
    """Describe one filing disclosure unit for comparison.

    One disclosure unit as the comparison sees it: where it is, what the
    source calls it, and the identifiers that align it.
    """

    document_key: str
    handle: str
    family: str
    title: str
    text: str
    case_numbers: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UnitCorrespondence:
    """Record how a later filing unit aligns with an earlier one.

    What one unit of a later filing corresponds to in the earlier one
    (or, for `ABSENT_LATER`, what unit of the earlier filing no later unit
    aligns with).
    """

    entity_id: str
    family: str
    later_document_key: str
    later_handle: str | None
    earlier_document_key: str | None
    earlier_handle: str | None
    state: str
    """`EXACT_REPEAT`, `CHANGED_ALIGNED`, `FIRST_OBSERVED`, `ABSENT_LATER`
    or `UNRESOLVED`."""
    ratio: float | None
    changed_characters: int
    basis: str


def normalized_text(text: str) -> str:
    """Normalize markup and whitespace before passage comparison.

    The text with markup emphasis and whitespace runs removed: two
    renderings of one passage compare equal, two passages do not.
    """
    return _WHITESPACE.sub(" ", _EMPHASIS.sub("", text)).strip().casefold()


def text_digest(text: str) -> str:
    """Hash the normalized text used to compare disclosure units."""
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def alignment_key(unit: ComparableUnit) -> tuple[str, str]:
    """Identify a disclosure unit across filing revisions.

    `(family, identifier)`: a case number when the source states one,
    else the title's content words with amounts and small words removed --
    "Note 9. Debt" and "NOTE 9 — DEBT" are one identifier, "In re X
    Securities Litigation" and "X Derivative Action" are two.
    """
    if unit.case_numbers:
        return unit.family, f"case:{sorted(unit.case_numbers)[0].casefold()}"
    words = [
        word
        for word in re.findall(r"[a-z0-9&'\u2019.-]+", _AMOUNT.sub(" ", unit.title.casefold()))
        if word not in _SMALL and not word.replace(".", "").isdigit()
    ]
    return unit.family, "title:" + " ".join(words)[:120]


def compare_filings(
    *,
    entity_id: str,
    later: Sequence[ComparableUnit],
    earlier: Sequence[ComparableUnit],
) -> tuple[UnitCorrespondence, ...]:
    """Align later disclosure units with an earlier filing.

    Every unit of `later` against `earlier` (one filing each, one issuer),
    in `later`'s order, then every unit of `earlier` no later unit aligned
    with. An identifier shared by several earlier units aligns with the
    most similar one under the bound; a unit whose identifier no earlier
    unit shares is first observed in the later filing. Deterministic.
    """
    by_key: dict[tuple[str, str], list[ComparableUnit]] = {}
    for unit in earlier:
        by_key.setdefault(alignment_key(unit), []).append(unit)
    matched: set[tuple[str, str]] = set()
    results: list[UnitCorrespondence] = []
    for unit in later:
        candidates = [
            value
            for value in by_key.get(alignment_key(unit), [])
            if (value.document_key, value.handle) not in matched
        ]
        if not candidates:
            results.append(
                UnitCorrespondence(
                    entity_id=entity_id,
                    family=unit.family,
                    later_document_key=unit.document_key,
                    later_handle=unit.handle,
                    earlier_document_key=None,
                    earlier_handle=None,
                    state="FIRST_OBSERVED",
                    ratio=None,
                    changed_characters=0,
                    basis="no unit of the earlier filing shares the identifier",
                )
            )
            continue
        best: tuple[ComparableUnit, str, float | None, int, str] | None = None
        for candidate in candidates:
            state, ratio, changed, basis = _compare(unit.text, candidate.text)
            if best is None or _better((state, ratio), (best[1], best[2])):
                best = (candidate, state, ratio, changed, basis)
        assert best is not None
        candidate, state, ratio, changed, basis = best
        matched.add((candidate.document_key, candidate.handle))
        results.append(
            UnitCorrespondence(
                entity_id=entity_id,
                family=unit.family,
                later_document_key=unit.document_key,
                later_handle=unit.handle,
                earlier_document_key=candidate.document_key,
                earlier_handle=candidate.handle,
                state=state,
                ratio=ratio,
                changed_characters=changed,
                basis=basis,
            )
        )
    for unit in earlier:
        if (unit.document_key, unit.handle) in matched:
            continue
        results.append(
            UnitCorrespondence(
                entity_id=entity_id,
                family=unit.family,
                later_document_key=later[0].document_key if later else unit.document_key,
                later_handle=None,
                earlier_document_key=unit.document_key,
                earlier_handle=unit.handle,
                state="ABSENT_LATER",
                ratio=None,
                changed_characters=0,
                basis=(
                    "no unit of the later filing shares the identifier; absence is not resolution"
                ),
            )
        )
    return tuple(results)


def _better(candidate: tuple[str, float | None], held: tuple[str, float | None]) -> bool:
    order = {"EXACT_REPEAT": 3, "CHANGED_ALIGNED": 2, "UNRESOLVED": 1}
    if order[candidate[0]] != order[held[0]]:
        return order[candidate[0]] > order[held[0]]
    return (candidate[1] or 0.0) > (held[1] or 0.0)


def _compare(later: str, earlier: str) -> tuple[str, float | None, int, str]:
    if text_digest(later) == text_digest(earlier):
        return "EXACT_REPEAT", 1.0, 0, "the same normalized text"
    a, b = normalized_text(later), normalized_text(earlier)
    if len(a) > COMPARISON_TEXT_BOUND or len(b) > COMPARISON_TEXT_BOUND:
        return (
            "UNRESOLVED",
            None,
            0,
            f"texts differ and exceed the comparison bound of {COMPARISON_TEXT_BOUND} characters",
        )
    matcher = SequenceMatcher(None, b, a, autojunk=False)
    ratio = matcher.ratio()
    changed = sum(
        max(j2 - j1, i2 - i1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"
    )
    if ratio < ALIGNMENT_RATIO_FLOOR:
        return (
            "UNRESOLVED",
            ratio,
            changed,
            f"the identifier repeats but the texts share {ratio:.2f} of their characters",
        )
    return (
        "CHANGED_ALIGNED",
        ratio,
        changed,
        f"aligned by identifier; {changed} character(s) changed (ratio {ratio:.2f})",
    )


__all__ = [
    "ALIGNMENT_RATIO_FLOOR",
    "COMPARISON_RULES_ID",
    "COMPARISON_TEXT_BOUND",
    "ComparableUnit",
    "UnitCorrespondence",
    "alignment_key",
    "compare_filings",
    "normalized_text",
    "text_digest",
]
