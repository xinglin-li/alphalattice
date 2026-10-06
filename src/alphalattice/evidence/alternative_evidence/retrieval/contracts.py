"""Host and model-safe contracts for governed evidence retrieval."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast

from pydantic import Field, model_validator

from ..contracts import (
    AlternativeEvidenceContract,
    AlternativeEvidenceReadingDepth,
    validate_contract_identity,
)

RETRIEVAL_GENERATION_FORMAT: Literal["knowledge-hybrid-v4"] = "knowledge-hybrid-v4"
"""The kernel index schema the active generation record is sealed over."""


class _RetrievalGenerationRecord(AlternativeEvidenceContract):
    """What every retrieval generation record states, in either durable format."""

    kind: Literal["AlternativeEvidenceRetrievalGeneration"] = (
        "AlternativeEvidenceRetrievalGeneration"
    )
    document_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    workspace_snapshot_id: str
    workspace_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reading_depth: AlternativeEvidenceReadingDepth
    index_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_count: int = Field(ge=1, le=24)
    chunk_count: int = Field(ge=1)
    built_at: datetime
    generation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_generation(self) -> Self:
        if self.built_at.tzinfo is None or self.built_at.utcoffset() is None:
            raise ValueError("alternative_evidence.retrieval_generation_clock_invalid")
        validate_contract_identity(self, "generation_hash")
        return self


class AlternativeEvidenceRetrievalGeneration(_RetrievalGenerationRecord):
    """One request's use of one physical index over its effective corpus.

    One index, deliberately: hits are ranked against every admitted document
    together, so a query about one issuer's liquidity returns the passages that
    match it best rather than the best passage of each document in turn.

    The index is the kernel's `knowledge-hybrid-v4` generation for the corpus
    this request's document set resolves to; requests with different cutoffs
    whose eligible documents are the same bytes share it. The record keeps the
    request's own snapshot and cutoff (its question and provenance) and the
    corpus, manifest and vector-payload commitments the kernel verified, which
    anchor every later open: the database's own copy of its manifest is never
    the authority for itself.
    """

    generation_format: Literal["knowledge-hybrid-v4"] = RETRIEVAL_GENERATION_FORMAT
    corpus_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    vector_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HybridV3RetrievalGeneration(_RetrievalGenerationRecord):
    """A generation record sealed over the kernel's `knowledge-hybrid-v3` index.

    The durable format every retained workspace holds: keyed by its snapshot,
    with no corpus, manifest or vector commitment. Read exactly, as history,
    and opened only through the kernel's legacy path; never written again.
    Retired with the supported historical binding contract, by a recorded
    decision.
    """


WHOLE_FILINGS_FORMAT: Literal["whole-filings-v1"] = "whole-filings-v1"


class WholeFilingsGeneration(_RetrievalGenerationRecord):
    """Seal a whole-filing generation without a retrieval index.

    A unit's filings delivered whole, with no index built (W4): the unit is
    short enough for the bundle, so its record commits to its filings' bytes
    (`corpus_hash`, the digest of each canonical revision) and to nothing a
    session could open; its index id is that corpus and its spec the format.
    """

    generation_format: Literal["whole-filings-v1"] = WHOLE_FILINGS_FORMAT
    corpus_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


RetrievalGenerationRecord = (
    AlternativeEvidenceRetrievalGeneration | HybridV3RetrievalGeneration | WholeFilingsGeneration
)
CurrentRetrievalGeneration = AlternativeEvidenceRetrievalGeneration | WholeFilingsGeneration
"""The formats a new analysis stands on: an index, or a unit delivered whole."""


def parse_retrieval_generation(payload: dict[str, object]) -> RetrievalGenerationRecord:
    """Bounded durable-format dispatch: the committed format names itself."""
    if payload.get("generation_format") == WHOLE_FILINGS_FORMAT:
        return cast(WholeFilingsGeneration, WholeFilingsGeneration.model_validate(payload))
    if "generation_format" in payload:
        return cast(
            AlternativeEvidenceRetrievalGeneration,
            AlternativeEvidenceRetrievalGeneration.model_validate(payload),
        )
    return cast(HybridV3RetrievalGeneration, HybridV3RetrievalGeneration.model_validate(payload))


SPAN_HANDLE_PATTERN = r"^SPAN-[SWTMXCF][0-9]{2}-R[0-9]{2,4}$"
"""A span handle names its issuance: `SPAN-S<search>-R<rank>` for a hit of
a search, `SPAN-W<scan>-R<n>` for a window a structural scan admitted,
`SPAN-T01-R<n>` for an exact source range a typed disclosure rule bound,
`SPAN-M01-R<n>` for a window of a provisionally inventoried litigation matter
read whole under the matter budget, `SPAN-X<session>-R<n>` for a table
view rendered from the retained original at a placeholder of the canonical
text, under the same matter budget, `SPAN-C<session>-R<n>` for a residual
candidate a search of an earlier session of the chain returned and its
packet sealed as pending, read in this session at its sealed range under
the program's span-read budget without a search, `SPAN-F01-R<n>` for a piece
of a filing delivered whole, at its range in the canonical text (W4).
Each resolves through the same citation and the same verified read."""


class AlternativeEvidenceSearchHit(AlternativeEvidenceContract):
    """Describe one ranked retrieval hit with semantic handles and source availability.

    Attributes:
        span_handle: Retrievable SPAN handle.
        document_handle: Containing DOC handle.
        entity_id: Research issuer identifier.
        source_name: Source authority name.
        title: Source document title.
        preview: Bounded candidate text preview.
        rank: One-based bounded result rank.
        retrieval_score: Finite retrieval score.
        channels: Nonempty contributing retrieval channels.
        available_at: Evidence availability clock.
    """

    span_handle: str = Field(pattern=SPAN_HANDLE_PATTERN)
    document_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    entity_id: str = Field(min_length=1, max_length=32)
    source_name: str = Field(min_length=1, max_length=40)
    title: str = Field(min_length=1, max_length=300)
    preview: str = Field(min_length=1, max_length=1600)
    rank: int = Field(ge=1, le=180)
    """Position in the reranked order of one query's return: up to twenty per
    issuer group of the request (eight, and one trailing group), so a rank
    past twenty is an issuer's place, not a deeper cut."""
    # The retriever's cross-encoder score, carried rather than dropped: rank
    # only says where a hit came in its own query, and the packet has to
    # compare hits across queries.
    retrieval_score: float = Field(allow_inf_nan=False)
    channels: tuple[str, ...] = Field(min_length=1)
    available_at: datetime


class AlternativeEvidenceSearchResult(AlternativeEvidenceContract):
    """Return bounded ranked hits and the remaining retrieval budget for one query.

    Attributes:
        query: Bounded submitted query text.
        status: Declared search result status.
        hits: Ranked semantic hit records.
        remaining_searches: Remaining bounded search budget.
        reranked_pairs: Number of reranked query/document pairs.
    """

    query: str = Field(min_length=1, max_length=512)
    status: str
    hits: tuple[AlternativeEvidenceSearchHit, ...]
    remaining_searches: int = Field(ge=0, le=24)
    reranked_pairs: int = Field(default=0, ge=0)
    """Pairs the cross-encoder scored for this search: what a pair budget
    is charged; zero when the kernel reported none."""


class SourceUnitDeclaration(AlternativeEvidenceContract):
    """Retain the source's explicit amount scale and its exact location.

    The source's own statement of the scale its amounts are in -- "(In
    thousands, except per share data)" at the head of a note -- quoted
    verbatim from the same document at the range it occupies, so a reader of
    "$29,595 outstanding" knows the filing means thousands. It is the nearest
    such statement before the passage; an amount that names its own scale in
    the text ("$1.2 billion") is in that scale. Origin identifiable, never
    inferred: absent when the passage carries no bare amount or the source
    states no scale.
    """

    text: str = Field(min_length=1, max_length=200)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=1)


class TableViewBinding(AlternativeEvidenceContract):
    """Bind a rendered table view to original source bytes and the rendering method.

    What a table view was rendered from and how: the retained original
    (by content hash and media type) the canonical text's placeholder stands
    for, the extraction rules whose table reading rendered it, the table's
    ordinal among the original's uncarried tables and the row count both
    the placeholder and the original state, the rows this page holds and
    how many remain, and the column headings. The span's own range is the
    placeholder line in the canonical text, verified as any span is; the
    excerpt is the rendering, not source bytes at that range.
    """

    rules_id: str = Field(min_length=1, max_length=80)
    parser_rules_id: str = Field(min_length=1, max_length=80)
    parent_source_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_media_type: str = Field(min_length=1, max_length=40)
    table_ordinal: int = Field(ge=1)
    rows_total: int = Field(ge=0)
    rows_from: int = Field(ge=1)
    rows_to: int = Field(ge=0)
    remaining_rows: int = Field(ge=0)
    rows_clipped: int = Field(default=0, ge=0, exclude_if=lambda v: v == 0)
    """Rows of this page bounded at a word because one row alone exceeded
    the reader's ceiling (absent at 0, so every earlier span keeps its hash)."""
    headings: tuple[str, ...] = Field(default=(), max_length=64)
    caption: str = Field(default="", max_length=400)
    scale: str = Field(default="", max_length=240)
    footnote_count: int = Field(default=0, ge=0)


class AlternativeEvidenceResolvedSpan(AlternativeEvidenceContract):
    """Bind a delivered excerpt to source coordinates, provenance and optional derived-table proof.

    Attributes:
        span_handle: Delivered SPAN handle.
        document_handle: Containing DOC handle.
        entity_id: Research issuer identifier.
        source_name: Source authority name.
        source_right: Declared source access right.
        document_type: Source document type.
        revision_label: Source revision label.
        title: Source document title.
        start_line: One-based first source line.
        end_line: One-based last source line.
        character_start: Inclusive source character offset.
        character_end: Exclusive source character offset.
        utf8_byte_start: Inclusive UTF-8 byte offset.
        utf8_byte_end: Exclusive UTF-8 byte offset.
        excerpt: Delivered text, bounded independently of source document size.
        published_at: Optional aware publication clock.
        accepted_at: Optional aware acceptance clock.
        available_at: Aware evidence availability clock.
        immutable_source: Source immutability qualification.
        limitations: Explicit evidence limits.
        unit_declaration: Optional source amount-scale declaration.
        table_view: Optional proof that the excerpt is a derived table view.
    """

    span_handle: str = Field(pattern=SPAN_HANDLE_PATTERN)
    document_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    entity_id: str = Field(min_length=1, max_length=32)
    source_name: str = Field(min_length=1, max_length=40)
    source_right: str = Field(min_length=1, max_length=80)
    document_type: str = Field(min_length=1, max_length=40)
    revision_label: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=300)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=1)
    utf8_byte_start: int = Field(ge=0)
    utf8_byte_end: int = Field(ge=1)
    excerpt: str = Field(min_length=1, max_length=16_384)
    published_at: datetime | None = None
    accepted_at: datetime | None = None
    available_at: datetime
    immutable_source: bool
    limitations: tuple[str, ...] = ()
    unit_declaration: SourceUnitDeclaration | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """Absent from the identity when the source states no scale, so span sets
    sealed before this field keep their hashes."""
    table_view: TableViewBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """Present on a table view (`SPAN-X..`): the excerpt is the rendering of
    the retained original's table at the placeholder this span's range
    names, bound to that original and parser; absent on every source span,
    whose excerpt is the source bytes at its range, so earlier span sets
    keep their hashes."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_span(self) -> Self:
        """Validate source coordinates, aware clocks and ordinary/derived excerpt bounds.

        Returns:
            This validated contract.

        Raises:
            ValueError: Coordinates or clocks are invalid, ordinary excerpt bytes disagree, or a
                table lacks an X handle.
        """
        if self.end_line < self.start_line:
            raise ValueError("alternative_evidence.resolved_span_lines_invalid")
        if self.character_end <= self.character_start:
            raise ValueError("alternative_evidence.resolved_span_characters_invalid")
        if self.utf8_byte_end <= self.utf8_byte_start:
            raise ValueError("alternative_evidence.resolved_span_bytes_invalid")
        derived = self.table_view is not None
        if derived and not self.span_handle.startswith("SPAN-X"):
            # A rendering is delivered only under the table-view series; the
            # series' own anchor (the placeholder line, before rendering) is
            # a plain span of that range.
            raise ValueError("alternative_evidence.resolved_span_view_invalid")
        if not derived and (
            self.utf8_byte_end - self.utf8_byte_start != len(self.excerpt.encode("utf-8"))
        ):
            raise ValueError("alternative_evidence.resolved_span_byte_count_invalid")
        for value in (self.published_at, self.accepted_at, self.available_at):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("alternative_evidence.resolved_span_clock_invalid")
        return self


class AlternativeEvidenceResolvedSpanSet(AlternativeEvidenceContract):
    """Seal unique resolved spans against a request and retrieval generation.

    Attributes:
        request_hash: Research request identity.
        retrieval_generation_hash: Retrieval generation commitment.
        spans: Uniquely handled resolved excerpts.
        span_set_hash: Canonical resolved-span set identity.
    """

    kind: Literal["AlternativeEvidenceResolvedSpanSet"] = "AlternativeEvidenceResolvedSpanSet"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_generation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    spans: tuple[AlternativeEvidenceResolvedSpan, ...]
    span_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_span_set(self) -> Self:
        """Require unique resolved handles and canonical span-set identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Span handles repeat or canonical set identity differs.
        """
        handles = tuple(value.span_handle for value in self.spans)
        if len(handles) != len(set(handles)):
            raise ValueError("alternative_evidence.resolved_span_duplicate")
        validate_contract_identity(self, "span_set_hash")
        return self


class PairScoreBlockRecord(AlternativeEvidenceContract):
    """Commit one sealed cross-encoder score block by name and byte hash.

    One sealed block of cross-encoder pair scores: its name (the sha256
    of its bytes, under the reranker context's directory), how many pairs
    it holds and how long it is.
    """

    name: str = Field(pattern=r"^[0-9a-f]{64}[.]scores$")
    pair_count: int = Field(ge=1)
    byte_length: int = Field(ge=1)


class PairScoreCommitmentRecord(AlternativeEvidenceContract):
    """Commit every sealed score block retained by a retrieval session.

    The blocks of cross-encoder pair scores one session sealed, committed
    by the runtime that ran the model and named by the receipt of that
    session: the only authority under which a later reader may serve a
    block. A block on disk that no commitment names is a claim, never
    consulted. The commitment binds the complete derivation context (the
    reranker context hash: the pinned pack, its runtime versions, the
    rounding, the execution settings; the index spec) and, through each
    block's content hash, the exact pairs and their exact outputs.
    """

    kind: Literal["PairScoreCommitmentRecord"] = "PairScoreCommitmentRecord"
    reranker_context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_spec_logical_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_generation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    blocks: tuple[PairScoreBlockRecord, ...] = Field(min_length=1, max_length=64)
    sealed_at: datetime
    commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_commitment(self) -> Self:
        """Require distinct score-block names and the canonical commitment identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Score-block names repeat or canonical commitment identity differs.
        """
        names = tuple(block.name for block in self.blocks)
        if len(names) != len(set(names)):
            raise ValueError("alternative_evidence.pair_score_commitment_invalid")
        validate_contract_identity(self, "commitment_hash")
        return self


__all__ = [
    "RETRIEVAL_GENERATION_FORMAT",
    "SPAN_HANDLE_PATTERN",
    "WHOLE_FILINGS_FORMAT",
    "AlternativeEvidenceResolvedSpan",
    "AlternativeEvidenceResolvedSpanSet",
    "AlternativeEvidenceRetrievalGeneration",
    "AlternativeEvidenceSearchHit",
    "AlternativeEvidenceSearchResult",
    "CurrentRetrievalGeneration",
    "HybridV3RetrievalGeneration",
    "PairScoreBlockRecord",
    "PairScoreCommitmentRecord",
    "RetrievalGenerationRecord",
    "SourceUnitDeclaration",
    "TableViewBinding",
    "WholeFilingsGeneration",
    "parse_retrieval_generation",
]
