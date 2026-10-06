"""The grounded recall rule for the Gate 9C6E casebook.

DEVELOPMENT_CALIBRATION. A retrieval development criterion, not a CRO decision
threshold, not independent evaluation, not prospective scientific validation.

The legacy casebook asked whether a probe term appeared anywhere in a selected
excerpt for that issuer. On real filings that is not a question about evidence:
"litigation" appears in the forward-looking-statement recital, "PFAS" appears as
an inline-XBRL element name, and "tax" appears in a risk-factor list. A packet
made of those passages tells a reviewer nothing while scoring perfectly.

Two grounded rules preceded this one and each was too weak in its own way. The
first compared a character range to a midpoint, so a one-character span counted
and the excerpt was never read. The second read the excerpt but trusted it: it
took the text a span *claimed* to hold and never established that the claim
agreed with the source at the range and revision the span declared, so an
invented excerpt under a real handle would have scored.

This rule closes that boundary. Every span is resolved through the verified
document set by the handle it carries; its declared issuer, document type and
revision must agree with what the set says that handle is; its excerpt must be
found inside the source bytes at the range it declares. Only an excerpt that
survives all of that is read, and only then is the annotated sentence looked
for inside it. Every annotation is likewise hashed against source bytes before
any scoring, so a casebook that has drifted from its corpus is refused rather
than re-scored.

Multi-fact cases are explicit. A case may name several sub-facts, each with its
own occurrences; the case declares whether recall means any sub-fact or all of
them, and both readings are reported so the declared one is never the only one
visible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CASEBOOK_PATH = Path(__file__).with_name("gate-9c6e-grounded-casebook.json")

RECALL_ANY = "any_sub_fact"
RECALL_ALL = "all_sub_facts"


class GroundedCasebookError(RuntimeError):
    """The casebook does not describe the corpus it is being scored against."""


@dataclass(frozen=True, slots=True)
class GroundedProbeOutcome:
    probe_id: str
    classification: str
    recalled: bool
    """Recall under the rule the case declares (`any_sub_fact` or `all_sub_facts`)."""

    any_sub_fact: bool
    all_sub_facts: bool
    matched_ranges: int
    verified_ranges: int
    sub_facts_matched: tuple[str, ...]
    sub_facts_total: int


def load_casebook(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or CASEBOOK_PATH).read_text(encoding="utf-8"))


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _sub_fact(value: dict[str, Any]) -> str:
    return str(value.get("sub_fact", "fact"))


def verify_casebook(
    *,
    casebook: dict[str, Any],
    document_set: Any,
    library: Any,
    document_bundle_hash: str,
) -> dict[tuple[str, str, str], tuple[str, int, str]]:
    """Prove the annotations still describe this corpus, and locate them.

    Returns a map from (entity, document_type, revision) to the workspace
    document, its revision and its exact text. Every annotated range is read
    back from the source bytes and hashed against the anchor recorded when the
    casebook was frozen, so a corpus, revision or anchor that has moved is a
    refusal rather than a silent re-score.
    """

    if casebook.get("document_bundle_hash") != document_bundle_hash:
        raise GroundedCasebookError(
            "casebook was frozen against another document bundle: "
            f"{casebook.get('document_bundle_hash')} != {document_bundle_hash}"
        )
    by_identity = {
        (item.entity_id, item.document_type, item.revision_label): item
        for item in document_set.documents
    }
    resolved: dict[tuple[str, str, str], tuple[str, int, str]] = {}
    for case in casebook["cases"]:
        for value in case["fact_ranges"]:
            key = (case["entity_id"], value["document_type"], value["revision"])
            reference = by_identity.get(key)
            if reference is None:
                raise GroundedCasebookError(f"annotated document is not admitted: {key}")
            if key not in resolved:
                _revision, content = library.read_revision(
                    reference.workspace_document_id, reference.workspace_revision
                )
                resolved[key] = (
                    reference.workspace_document_id,
                    reference.workspace_revision,
                    content.decode("utf-8"),
                )
            text = resolved[key][2]
            anchor = text[value["character_start"] : value["character_end"]]
            digest = hashlib.sha256(anchor.encode("utf-8")).hexdigest()
            if digest != value["anchor_sha256"]:
                raise GroundedCasebookError(
                    f"annotated range no longer holds its anchor: {case['probe_id']} {key}"
                )
    return resolved


def verify_spans(
    *,
    spans: tuple[Any, ...],
    document_set: Any,
    library: Any,
) -> tuple[tuple[tuple[str, str, str] | None, str | None], ...]:
    """What each span can be trusted to have shown, and from which document.

    A span is resolved by the handle the document set issued. Its own
    declaration -- issuer, document type, revision -- must agree with the set's
    record for that handle, so a crossed revision or issuer cannot score under a
    real handle. Its excerpt must then be found inside the source bytes at the
    range it declares. Anything that fails is returned with no excerpt: it may
    have been shown to the analyst, but nothing about it is established, and
    nothing unestablished is credited.
    """

    by_handle = {item.semantic_handle: item for item in document_set.documents}
    texts: dict[tuple[str, int], str] = {}
    verified: list[tuple[tuple[str, str, str] | None, str | None]] = []
    for span in spans:
        reference = by_handle.get(span.document_handle)
        if reference is None:
            verified.append((None, None))
            continue
        identity = (reference.entity_id, reference.document_type, reference.revision_label)
        declared = (span.entity_id, span.document_type, span.revision_label)
        if declared != identity:
            verified.append((identity, None))
            continue
        key = (reference.workspace_document_id, reference.workspace_revision)
        if key not in texts:
            _revision, content = library.read_revision(*key)
            texts[key] = content.decode("utf-8")
        text = texts[key]
        start, end = int(span.character_start), int(span.character_end)
        if not (0 <= start < end <= len(text)):
            verified.append((identity, None))
            continue
        excerpt = _collapse(span.excerpt)
        if not excerpt or excerpt not in _collapse(text[start:end]):
            verified.append((identity, None))
            continue
        verified.append((identity, excerpt))
    return tuple(verified)


def evaluate_grounded_recall(
    *,
    casebook: dict[str, Any],
    spans: tuple[Any, ...],
    document_set: Any,
    library: Any,
    document_bundle_hash: str,
) -> tuple[GroundedProbeOutcome, ...]:
    """Which annotated facts the packet actually put in front of the analyst.

    A range counts when a span that `verify_spans` accepted resolves to the
    annotated issuer, document and revision, and the annotated sentence is
    present in that span's verified excerpt. A sub-fact counts when any of its
    ranges counts. The case's declared rule decides whether recall means any
    sub-fact or all of them; both are reported.
    """

    resolved = verify_casebook(
        casebook=casebook,
        document_set=document_set,
        library=library,
        document_bundle_hash=document_bundle_hash,
    )
    verified = verify_spans(spans=spans, document_set=document_set, library=library)
    outcomes: list[GroundedProbeOutcome] = []
    for case in casebook["cases"]:
        rule = str(case.get("recall", RECALL_ANY))
        if rule not in (RECALL_ANY, RECALL_ALL):
            raise GroundedCasebookError(f"unknown recall rule {rule!r}: {case['probe_id']}")
        sub_facts: list[str] = []
        for value in case["fact_ranges"]:
            if _sub_fact(value) not in sub_facts:
                sub_facts.append(_sub_fact(value))
        matched = 0
        matched_sub_facts: list[str] = []
        for value in case["fact_ranges"]:
            key = (case["entity_id"], value["document_type"], value["revision"])
            _document_id, _revision, text = resolved[key]
            anchor = _collapse(text[value["character_start"] : value["character_end"]])
            if any(
                identity == key and excerpt is not None and anchor in excerpt
                for identity, excerpt in verified
            ):
                matched += 1
                if _sub_fact(value) not in matched_sub_facts:
                    matched_sub_facts.append(_sub_fact(value))
        any_sub_fact = bool(matched_sub_facts)
        all_sub_facts = len(matched_sub_facts) == len(sub_facts)
        outcomes.append(
            GroundedProbeOutcome(
                probe_id=case["probe_id"],
                classification=case["classification"],
                recalled=all_sub_facts if rule == RECALL_ALL else any_sub_fact,
                any_sub_fact=any_sub_fact,
                all_sub_facts=all_sub_facts,
                matched_ranges=matched,
                verified_ranges=len(case["fact_ranges"]),
                sub_facts_matched=tuple(matched_sub_facts),
                sub_facts_total=len(sub_facts),
            )
        )
    return tuple(outcomes)


def evaluable_cases(casebook: dict[str, Any]) -> tuple[str, ...]:
    """Probes that can be scored at all: a fact must exist in the corpus."""

    return tuple(
        case["probe_id"]
        for case in casebook["cases"]
        if case["classification"] == "SOURCE_CONFIRMED"
    )


def legacy_keyword_recall(
    *,
    casebook: dict[str, Any],
    spans: tuple[Any, ...],
) -> tuple[GroundedProbeOutcome, ...]:
    """The retired rule, kept so old and new outcomes stay comparable."""

    outcomes: list[GroundedProbeOutcome] = []
    for case in casebook["cases"]:
        reached = any(
            span.entity_id == case["entity_id"]
            and any(term in span.excerpt.casefold() for term in case["legacy_terms"])
            for span in spans
        )
        outcomes.append(
            GroundedProbeOutcome(
                probe_id=case["probe_id"],
                classification=case["classification"],
                recalled=reached,
                any_sub_fact=reached,
                all_sub_facts=reached,
                matched_ranges=int(reached),
                verified_ranges=len(case["fact_ranges"]),
                sub_facts_matched=(),
                sub_facts_total=0,
            )
        )
    return tuple(outcomes)


__all__ = [
    "CASEBOOK_PATH",
    "RECALL_ALL",
    "RECALL_ANY",
    "GroundedCasebookError",
    "GroundedProbeOutcome",
    "evaluable_cases",
    "evaluate_grounded_recall",
    "legacy_keyword_recall",
    "load_casebook",
    "verify_casebook",
    "verify_spans",
]
