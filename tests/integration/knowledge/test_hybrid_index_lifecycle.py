"""The hybrid index across its lifecycle: build, reuse, reopen, priming and tamper.

The corpus is embedded exactly once, exact reuse embeds nothing, a cold reopen
verifies the committed vectors and embeds nothing, primed and cold readers
agree, chunks stay distinct spans, and an edited database, an edited source, a
mismatched or a coherently fabricated priming all refuse before any result.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from alphalattice.evidence.alternative_evidence.analysis.packet import (
    SPANS_PER_READ,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceClass,
)
from alphalattice.evidence.alternative_evidence.runtime.identity import (
    alternative_retrieval_binding_hash,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument
from alphalattice.kernel.knowledge.hybrid import (
    _trigram_applies,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    HybridIndexSpec,
)
from tests.alternative_evidence_desk.document_intelligence_support import (
    _built,
    _corpus_passes,
    _counted_runtime,
)
from tests.alternative_evidence_desk.planted_corpus import (
    _NOW,
    PLAYPEN_ROOT,
)

pytestmark = pytest.mark.usefixtures("review_on_the_store")  # review runs on the Host's store

_QUERY_TOP_K = 5


def _one_line_document(entity_id: str = "AAPL") -> RecordedEvidenceDocument:
    """An inline-XBRL shaped filing: one heading and one very long body line.

    Real inline XBRL arrives as a single line of many kilobytes. Chunking
    splits it at 512 codepoints, so every chunk reports the same source line
    range, which is exactly the case that used to collapse.
    """

    body = " ".join(
        f"Segment {index:03d} reports operating detail for {entity_id}." for index in range(200)
    )
    return RecordedEvidenceDocument(
        entity_id=entity_id,
        source_right="USER_PROVIDED_FOR_LOCAL_RESEARCH",
        evidence_class=AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,
        document_type="EARNINGS_RELEASE",
        revision="inline-xbrl-single-line",
        published_at=_NOW - timedelta(days=2),
        captured_at=_NOW - timedelta(days=1),
        available_at=_NOW - timedelta(days=1),
        text=f"# Inline filing for {entity_id}\n\n{body}",
        immutable_source=True,
    )


def test_build_and_first_query_embed_the_corpus_exactly_once(tmp_path: Path) -> None:
    """Build and first query embed the corpus exactly once."""

    runtime, passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)

    assert passes == [generation.chunk_count], passes
    assert generation.chunk_count > _QUERY_TOP_K, "the fixture must separate the two call kinds"

    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        result = session.search(query="operating", top_k=3)
    finally:
        session.close()

    assert result.hits, "the query really ran"
    assert _corpus_passes(passes, generation) == 1, (
        "the first query verified the projection, it did not re-embed the corpus"
    )
    assert runtime.retrieval.passage_embedding_pass_count == 1


def test_exact_in_process_reuse_does_not_embed_the_corpus_again(tmp_path: Path) -> None:
    """requirement: no corpus pass on exact in-process reuse."""

    runtime, passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)
    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        session.search(query="operating", top_k=3)
        after_first = _corpus_passes(passes, generation)
        for _ in range(3):
            session.search(query="routine context", top_k=3)
    finally:
        session.close()

    assert after_first == 1
    assert _corpus_passes(passes, generation) == 1, (
        "an open reader answers further queries with no corpus pass"
    )


def test_chunks_sharing_one_source_line_remain_distinct_spans(tmp_path: Path) -> None:
    """Chunks sharing one source line remain distinct spans."""

    runtime, _passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime, documents=(_one_line_document(),))
    assert generation.chunk_count > 1, "the fixture must really split into several chunks"

    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        result = session.search(query="operating detail", top_k=_QUERY_TOP_K)
        handles = tuple(hit.span_handle for hit in result.hits)
        identities = tuple(session.span_identity(handle) for handle in handles)
        spans = session.read_spans(span_handles=handles[:SPANS_PER_READ])
        # The same chunk asked for twice is the same span, which is the whole
        # point of an identity: deduplication must still work.
        assert session.span_identity(handles[0]) == session.span_identity(handles[0])
    finally:
        session.close()

    assert len(handles) > 1, "several passages matched"
    # Every hit is on the same source line, and every hit is its own span.
    line_ranges = {(value[0], value[1], value[3], value[4]) for value in identities}
    assert line_ranges == {("AAPL-inline-xbrl-single-line", 1, 1, 1)} or len(line_ranges) == 1
    assert len(set(identities)) == len(identities), "no two distinct chunks collapse"
    # Citations still say where in the source the passage is.
    for span in spans:
        assert span.start_line >= 1
        assert span.end_line >= span.start_line


@pytest.mark.proves("alternative_evidence.runtime_identity_binds_what_decides_selection")
def test_the_retrieval_binding_covers_the_owners_that_decide_selection(tmp_path: Path) -> None:
    """The retrieval binding covers the owners that decide selection."""

    tracked = (
        "config/identity-roles.json",
        "src/alphalattice/evidence/alternative_evidence/analysis/packet.py",
        "src/alphalattice/evidence/alternative_evidence/retrieval/contracts.py",
        "src/alphalattice/evidence/alternative_evidence/retrieval/service.py",
        "src/alphalattice/evidence/alternative_evidence/retrieval/session.py",
    )
    for relative in tracked:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PLAYPEN_ROOT / relative).read_bytes())
    baseline = alternative_retrieval_binding_hash(tmp_path)
    assert baseline == alternative_retrieval_binding_hash(PLAYPEN_ROOT), (
        "the copy must reproduce the live binding before anything is edited"
    )

    for relative in (
        "src/alphalattice/evidence/alternative_evidence/retrieval/session.py",
        "src/alphalattice/evidence/alternative_evidence/analysis/packet.py",
    ):
        edited = tmp_path / relative
        original = edited.read_bytes()
        edited.write_bytes(original + b"\n# a comment about the selection rule\n")
        assert alternative_retrieval_binding_hash(tmp_path) == baseline, relative
        edited.write_bytes(original + b"\nSELECTION_RULE_CHANGED = True\n")
        assert alternative_retrieval_binding_hash(tmp_path) != baseline, relative
        edited.write_bytes(original)
    assert alternative_retrieval_binding_hash(tmp_path) == baseline


def test_a_query_embeds_its_query_and_never_a_passage(tmp_path: Path) -> None:
    """A query embeds its query and never a passage."""

    queries: list[str] = []
    runtime, passes = _counted_runtime(tmp_path, queries=queries)
    request, document_set, generation = _built(runtime)
    assert _corpus_passes(passes, generation) == 1
    passage_calls_after_build = len(passes)

    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        for text in ("operating", "routine context", "background section"):
            assert session.search(query=text, top_k=3) is not None
    finally:
        session.close()

    assert len(passes) == passage_calls_after_build, (
        "a query must not put a single passage through the model"
    )
    assert queries == ["operating", "routine context", "background section"]


def test_two_chunks_from_one_line_return_different_excerpts(tmp_path: Path) -> None:
    """Two chunks from one line return different excerpts."""

    runtime, _passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime, documents=(_one_line_document(),))
    assert generation.chunk_count > 1

    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        result = session.search(query="operating detail", top_k=_QUERY_TOP_K)
        handles = tuple(hit.span_handle for hit in result.hits)
        spans = session.read_spans(span_handles=handles[:SPANS_PER_READ])
        identities = tuple(session.span_identity(handle) for handle in handles)
    finally:
        session.close()

    assert len(spans) > 1, "several passages matched"
    assert len({span.excerpt for span in spans}) == len(spans), (
        "each matched chunk returns its own text"
    )
    assert len({identity[2] for identity in identities}) == len(identities)
    # Every span is an exact slice of the sealed source, by byte offset.
    for span in spans:
        assert span.character_end > span.character_start
        assert span.utf8_byte_end > span.utf8_byte_start
        assert span.excerpt
    # And no span is the whole document or an arbitrary leading prefix.
    assert not all(span.character_start == 0 for span in spans)


@pytest.mark.parametrize(
    "heading",
    (
        pytest.param("## \u00a0", id="payx-8k-0000723531-26-000008"),
        "## " + "LongHeading" * 24,
    ),
)
def test_a_span_excerpt_round_trips_to_the_sealed_source_bytes(
    tmp_path: Path, heading: str
) -> None:
    """requirement: the excerpt is source bytes, verified against the revision."""

    runtime, _passes = _counted_runtime(tmp_path)
    broken_headings = f"# Filing\n{heading}\noperating detail before and after\n"
    document = _one_line_document().model_copy(update={"text": broken_headings})
    from pydantic import ValidationError

    try:
        request, document_set, generation = _built(runtime, documents=(document,))
    except ValidationError as error:
        if error.title != "KnowledgeCitation" or "heading_path" not in str(error):
            raise
        pytest.fail(f"retrieval rejected the filing heading: {error}")
    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        result = session.search(query="operating detail", top_k=3)
        spans = session.read_spans(span_handles=(result.hits[0].span_handle,))
        library = runtime.documents.library
        reference = next(iter(document_set.documents))
        _revision, content = library.read_revision(
            reference.workspace_document_id, reference.workspace_revision
        )
    finally:
        session.close()

    span = spans[0]
    assert content[span.utf8_byte_start : span.utf8_byte_end].decode("utf-8") == span.excerpt
    text = content.decode("utf-8")
    assert text[span.character_start : span.character_end] == span.excerpt
    # The line stays a truthful human coordinate for where the excerpt starts.
    assert span.start_line == text.count("\n", 0, span.character_start) + 1


def test_the_trigram_rule_is_declared_by_query_shape_not_by_a_clock() -> None:
    """The trigram rule is declared by query shape not by a clock."""

    spec = HybridIndexSpec.fixed_v2()
    size = spec.lexical_spec.chunk_size_codepoints

    # Serves a single contiguous probe, which is what trigram is for.
    assert _trigram_applies("impairment", spec)
    assert _trigram_applies("pfas", spec)
    assert _trigram_applies("mpairmen", spec), "substring probes are the point"

    # Declines a multi-word query, which it can only answer as a phrase.
    assert not _trigram_applies("equity offering convertible notes dilution", spec)
    assert not _trigram_applies("a b", spec)
    assert not _trigram_applies("line\nbreak", spec)

    # Declines what the tokenizer or the corpus cannot hold either way.
    assert not _trigram_applies("ab", spec)
    assert not _trigram_applies("x" * (size + 1), spec), (
        "a probe longer than a chunk cannot be contained in one"
    )
    assert _trigram_applies("x" * size, spec)
