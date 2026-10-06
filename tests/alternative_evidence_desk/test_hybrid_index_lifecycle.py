"""The hybrid index across its lifecycle: build, reuse, reopen, priming and tamper.

The corpus is embedded exactly once, exact reuse embeds nothing, a cold reopen
verifies the committed vectors and embeds nothing, primed and cold readers
agree, chunks stay distinct spans, and an edited database, an edited source, a
mismatched or a coherently fabricated priming all refuse before any result.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceRetrievalAccessReceipt,
    EvidenceTopic,
)
from alphalattice.evidence.alternative_evidence.analysis.packet import (
    EVIDENCE_QUERY_PROGRAM,
    SPANS_PER_READ,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceClass,
    AlternativeEvidenceRequest,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
    AlternativeEvidenceRetrievalGeneration,
    PairScoreCommitmentRecord,
)
from alphalattice.evidence.alternative_evidence.runtime.identity import (
    alternative_retrieval_binding_hash,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument
from alphalattice.kernel.knowledge import _reranking as reranking
from alphalattice.kernel.knowledge import hybrid as knowledge_hybrid
from alphalattice.kernel.knowledge._embeddings import (
    MODEL_WORK,
    serialize_vector,
)
from alphalattice.kernel.knowledge.hybrid import (
    HybridIndexPriming,
    WorkspaceHybridKnowledgeIndex,
    _hybrid_rows,
    _manifest,
    _trigram_applies,
    corpus_logical_hash,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    PAIR_SCORE_ROOT,
    HybridIndexSpec,
    PairScoreAdmission,
    SealedPairScoreBlock,
    reranker_context_hash,
)
from alphalattice.kernel.knowledge.retrieval import _build_chunks, _normalize_text
from alphalattice.kernel.knowledge.retrieval_contracts import RetrievalChannel
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from alphalattice.kernel.shared_kernel.domain.serialization import sha256_hex
from tests.alternative_evidence_desk import planted_corpus
from tests.alternative_evidence_desk.document_intelligence_support import (
    _built,
    _corpus_passes,
    _counted_runtime,
    _registry,
)
from tests.alternative_evidence_desk.planted_corpus import (
    _NOW,
    PLANTED_CLAIMS,
    PLAYPEN_ROOT,
    _recorded_document,
)

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
    """requirement: build plus immediate query is one full-corpus embedding pass.

    Before the owner seam existed the build embedded the corpus to publish the
    projection and the first query embedded it again to verify what had just
    been written, so the expensive half of retrieval was paid twice for one
    generation.
    """

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
    assert runtime.retrieval.generation_handoff_count == 1
    assert runtime.retrieval.independent_generation_reopen_count == 0


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


def test_a_cold_reopen_pays_one_fail_closed_verification_pass(tmp_path: Path) -> None:
    """requirement: cold reopen keeps its full verification, at zero passes.

    Priming is single use. A second session over the same generation has no
    vectors in hand, so it reads the payload the build committed, checks it
    against its digest, compares every stored vector with it byte for byte,
    re-derives the corpus and proves the rows against the manifest. The
    passage model is not run: the commitment is what is verified.
    """

    runtime, passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)

    primed = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        primed.search(query="operating", top_k=3)
    finally:
        primed.close()
    assert _corpus_passes(passes, generation) == 1

    cold = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        cold.search(query="operating", top_k=3)
    finally:
        cold.close()

    assert _corpus_passes(passes, generation) == 1, (
        "the cold reader verified the committed vectors; it did not embed the corpus"
    )
    assert runtime.retrieval.generation_handoff_count == 1
    assert runtime.retrieval.independent_generation_reopen_count == 1


def test_primed_and_cold_readers_agree_on_hits_and_packet_order(tmp_path: Path) -> None:
    """requirement: identical hits and packet ordering across the optimization.

    The primed reader skips the second corpus pass and nothing else. If it
    disagreed with a cold reader about a single rank, the optimization would
    have changed the product rather than its cost.
    """

    runtime, passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)

    def read(session: Any) -> tuple[Any, ...]:
        try:
            hits = tuple(
                (hit.span_handle, hit.rank, hit.entity_id, tuple(hit.channels))
                for query in EVIDENCE_QUERY_PROGRAM
                for hit in session.search(query=query.text, top_k=_QUERY_TOP_K).hits
            )
            identities = tuple(session.span_identity(handle) for handle, _r, _e, _c in hits)
            return hits, identities
        finally:
            session.close()

    primed = read(
        runtime.retrieval.open_session(
            document_set=document_set,
            generation=generation,
            evidence_as_of=request.evidence_as_of,
        )
    )
    assert _corpus_passes(passes, generation) == 1
    cold = read(
        runtime.retrieval.open_session(
            document_set=document_set,
            generation=generation,
            evidence_as_of=request.evidence_as_of,
        )
    )

    assert _corpus_passes(passes, generation) == 1, "the cold reader embedded nothing"
    assert runtime.retrieval.independent_generation_reopen_count == 1
    assert primed == cold


def test_chunks_sharing_one_source_line_remain_distinct_spans(tmp_path: Path) -> None:
    """requirement: same-line multi-chunk fixture preserves every distinct span.

    Keyed on `(document, revision, start_line, end_line)` alone, every chunk of
    a single-line filing is one candidate, so a packet that may hold many spans
    receives one per document however many distinct passages matched.
    """

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


def test_a_mismatched_or_tampered_priming_refuses_before_any_result(tmp_path: Path) -> None:
    """requirement: mismatched or tampered priming refuses before publication.

    Priming is a claim about a generation, so it is checked like one. A reader
    primed for one generation must not answer for another, and vectors that do
    not describe the published projection must not open it. The pending leases
    are reached directly because a healthy service never produces these states;
    the point is that the owner refuses them if anything ever does.
    """

    runtime, _passes = _counted_runtime(tmp_path)
    first_request, first_set, first = _built(runtime, documents=(_recorded_document(),))
    _second_request, _second_set, second = _built(
        runtime,
        entities=("MSFT",),
        documents=(_recorded_document("MSFT", revision="msft-q4"),),
    )
    pending = runtime.retrieval._pending_generations
    other_manifest = pending[second.generation_hash].manifest

    def refuse(replace: Any) -> str:
        lease = pending[first.generation_hash]
        original = lease._priming
        assert original is not None
        lease._priming = replace(original)
        session = runtime.retrieval.open_session(
            document_set=first_set,
            generation=first,
            evidence_as_of=first_request.evidence_as_of,
        )
        try:
            with pytest.raises(KnowledgeRetrievalError) as raised:
                session.search(query="operating", top_k=3)
        finally:
            session.close()
        pending[first.generation_hash] = lease
        return str(raised.value.failure.code)

    # A reader primed for another generation.
    assert (
        refuse(
            lambda value: HybridIndexPriming(
                manifest=other_manifest, chunks=value.chunks, vectors=value.vectors
            )
        )
        == "retrieval.snapshot_mismatch"
    )

    # A priming whose count does not match the published manifest, refused on
    # its own count before any projection is read.
    runtime_two, _ = _counted_runtime(tmp_path / "second")
    request_two, set_two, generation_two = _built(runtime_two, documents=(_recorded_document(),))
    lease_two = runtime_two.retrieval._pending_generations[generation_two.generation_hash]
    short = lease_two._priming
    assert short is not None
    lease_two._priming = HybridIndexPriming(
        manifest=short.manifest, chunks=short.chunks[:-1], vectors=short.vectors[:-1]
    )
    session = runtime_two.retrieval.open_session(
        document_set=set_two,
        generation=generation_two,
        evidence_as_of=request_two.evidence_as_of,
    )
    try:
        with pytest.raises(KnowledgeRetrievalError) as counted:
            session.search(query="operating", top_k=3)
    finally:
        session.close()
    assert counted.value.failure.code == "retrieval.index_corrupt"

    # Vectors that do not describe the published projection.
    runtime_three, _ = _counted_runtime(tmp_path / "third")
    request_three, set_three, generation_three = _built(
        runtime_three, documents=(_recorded_document(),)
    )
    lease_three = runtime_three.retrieval._pending_generations[generation_three.generation_hash]
    tampered = lease_three._priming
    assert tampered is not None
    lease_three._priming = HybridIndexPriming(
        manifest=tampered.manifest,
        chunks=tampered.chunks,
        vectors=tuple((0.0,) * len(vector) for vector in tampered.vectors),
    )
    session = runtime_three.retrieval.open_session(
        document_set=set_three,
        generation=generation_three,
        evidence_as_of=request_three.evidence_as_of,
    )
    try:
        with pytest.raises(KnowledgeRetrievalError) as corrupt:
            session.search(query="operating", top_k=3)
    finally:
        session.close()
    # Primed vectors are a claim to be the committed payload; these are not.
    assert corrupt.value.failure.code == "retrieval.vector_commitment_mismatch"


def test_the_retrieval_binding_covers_the_owners_that_decide_selection(tmp_path: Path) -> None:
    """requirement: correcting span identity rotates an identity, explicitly.

    A binding that hashes only what a generation *stores* lets the rule that
    decides what it *yields* change silently. `session.py` decides when two
    hits are one passage and `packet.py` decides which passages reach the
    analyst, so a change to either's code must move this hash; a comment moves
    nothing, the binding being their syntax (LAWS.md ID3).
    """

    tracked = (
        "config/identity-roles.json",
        "src/alphalattice/evidence/alternative_evidence/analysis/packet.py",
        "src/alphalattice/evidence/alternative_evidence/retrieval/contracts.py",
        "src/alphalattice/evidence/alternative_evidence/retrieval/service.py",
        "src/alphalattice/evidence/alternative_evidence/retrieval/session.py",
        "src/alphalattice/kernel/knowledge/retrieval.py",
        "src/alphalattice/kernel/knowledge/retrieval_contracts.py",
        "src/alphalattice/kernel/knowledge/hybrid.py",
        "src/alphalattice/kernel/knowledge/hybrid_contracts.py",
        "src/alphalattice/kernel/knowledge/_reranking.py",
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
    """requirement: zero passage embedding calls after the generation is open.

    Opening verifies the whole published projection against the model. Every
    query then re-embedded its `top_k` bodies to verify the hits it was about
    to return, re-proving what the open had already proved and paying the model
    per query for it. On the real corpus that was 8 calls per packet and the
    dominant per-packet cost.
    """

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


def test_a_database_edited_after_open_still_refuses(tmp_path: Path) -> None:
    """requirement: tampering after open must refuse, without re-embedding.

    The commitment held in memory is what a query verifies against, so the
    check has to be against the *stored* row rather than against itself.
    Rewriting a stored vector after the projection was verified must fail the
    hit rather than pass on a remembered value.
    """

    import sqlite3

    runtime, passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)
    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        assert session.search(query="operating", top_k=3).hits
        database = next(
            (tmp_path / "workspace" / ".system" / "knowledge-indexes").rglob("*.db"),
            None,
        )
        assert database is not None, "the published index must be on disk"
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("UPDATE dense_nodes SET vector_hash = ?", ("0" * 64,))
            connection.commit()
        with pytest.raises(KnowledgeRetrievalError) as refused:
            session.search(query="operating", top_k=3)
    finally:
        session.close()
    assert refused.value.failure.code == "retrieval.index_corrupt"
    # The refusal came from the stored row, not from re-embedding to check.
    assert len(passes) == 1


def test_two_chunks_from_one_line_return_different_excerpts(tmp_path: Path) -> None:
    """requirement: the matched chunk comes back, not the head of its line.

    A document with no Markdown heading is one whole-file section, so every
    chunk of it shares one line range. Resolving by line returned the same
    opening text for all of them; resolving by the chunk's own character range
    returns what was matched.
    """

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


def test_a_span_excerpt_round_trips_to_the_sealed_source_bytes(tmp_path: Path) -> None:
    """requirement: the excerpt is source bytes, verified against the revision."""

    runtime, _passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime, documents=(_one_line_document(),))
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
    """requirement: one deterministic rule, appropriate to trigram's purpose.

    Trigram asks the index for an exact run of characters. A multi-word product
    query is never that run, and asking anyway cost 190.7 seconds across the
    eight installed queries for zero rows. The rule is the query's shape and the
    index's own chunk size, so it decides the same way on any machine.
    """

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


def test_a_single_term_substring_query_still_reaches_the_trigram_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression: the rule must not retire substring and typo retrieval.

    This is the case the channel exists for: a probe that is a fragment of a
    word in the corpus, which no term index will match.
    """

    runtime, _passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime, documents=(_one_line_document(),))
    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        # "perating" is inside "operating" and is not a term in any index.
        fragment = session.search(query="perating", top_k=3)
        program = session.search(query="operating detail for AAPL segment", top_k=3)
    finally:
        session.close()

    # The hit must actually come through TRIGRAM. Asserting only that the probe
    # returns something passes even with the channel switched off, because the
    # dense channel answers every query.
    assert any(RetrievalChannel.TRIGRAM.value in hit.channels for hit in fragment.hits), [
        hit.channels for hit in fragment.hits
    ]
    assert program.hits, "and the product query still retrieves without trigram"
    assert not any(RetrievalChannel.TRIGRAM.value in hit.channels for hit in program.hits)

    # Switching the channel off must break the assertion above, which is what
    # makes this coverage load-bearing rather than decorative.
    monkeypatch.setattr(knowledge_hybrid, "_trigram_applies", lambda *_a, **_k: False)
    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        disabled = session.search(query="perating", top_k=3)
    finally:
        session.close()
    assert not any(RetrievalChannel.TRIGRAM.value in hit.channels for hit in disabled.hits), (
        "with the channel off no hit may claim it"
    )

    spec = HybridIndexSpec.fixed_v2()
    assert _trigram_applies(_normalize_text("perating"), spec)
    assert not _trigram_applies(_normalize_text("operating detail for AAPL segment"), spec)


def test_source_edited_after_open_still_refuses(tmp_path: Path) -> None:
    """requirement: per-query source verification keeps its commitment.

    The query no longer rebuilds all 9,725 chunks to prove the source; it
    re-reads and re-verifies the revisions the generation was built from. That
    has to catch an edited blob just as the rebuild did.
    """

    runtime, passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)
    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        assert session.search(query="operating", top_k=3).hits
        blob = next((tmp_path / "workspace" / "knowledge" / "blobs").iterdir())
        blob.write_bytes(blob.read_bytes() + b"\ntampered\n")
        with pytest.raises(KnowledgeRetrievalError) as refused:
            session.search(query="operating", top_k=3)
    finally:
        session.close()
    assert refused.value.failure.code == "retrieval.source_integrity"
    assert len(passes) == 1, "the refusal cost no corpus pass"


def test_a_coherently_fabricated_priming_is_refused_before_any_result(
    tmp_path: Path,
) -> None:
    """requirement: priming must prove it derives from the authoritative source.

    Verifying the database against the manifest and the vectors is not that
    proof, because all three can be made to agree with chunks that were never
    derived from the source. A chunk's identity folds its document, revision,
    content hash, heading path, ordinal, source range and body hash -- but not
    its title -- so titles can be rewritten while every chunk id, chunk hash,
    embedding input and vector stays exactly as built. Rebuild the projection
    and the manifest from those chunks and the whole generation is internally
    coherent and still a fabrication.

    The per-query chunk comparison used to catch this. It is now one derivation
    at primed open, which is where the claim is actually made.
    """

    runtime, passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)
    lease = runtime.retrieval._pending_generations[generation.generation_hash]
    priming = lease._priming
    assert priming is not None

    library = runtime.documents.library
    snapshot = library.read_snapshot(UUID(str(document_set.workspace_snapshot_id)))
    spec = HybridIndexSpec.fixed_v2()
    authentic = _build_chunks(snapshot, library, spec.lexical_spec)
    fabricated = tuple(replace(chunk, title="FABRICATED ISSUER TITLE") for chunk in authentic)

    # The fabrication is invisible to every identity the reader checks.
    assert [chunk.chunk_id for chunk in fabricated] == [chunk.chunk_id for chunk in authentic]
    assert [chunk.chunk_hash for chunk in fabricated] == [chunk.chunk_hash for chunk in authentic]

    rows = _hybrid_rows(fabricated, spec)
    forged = _manifest(
        corpus_hash=corpus_logical_hash(snapshot),
        document_count=len(snapshot.revisions),
        spec=spec,
        rows=rows,
        commitment=priming.manifest.vector_commitment,
    )
    assert forged != priming.manifest, "only the projection commitment moves"
    assert forged.ordered_chunk_ids == priming.manifest.ordered_chunk_ids

    # Publish a fully coherent index for the fabricated generation, at the same
    # path, with the vectors the genuine build produced and committed, and a
    # generation record that anchors the forged manifest -- every identity the
    # reader checks before deriving the corpus agrees with the fabrication.
    database = tmp_path / "workspace" / Path(*forged.database_path.split("/"))
    database.unlink()
    blobs = tuple(serialize_vector(vector) for vector in priming.vectors)
    WorkspaceHybridKnowledgeIndex._build_database(database, rows, blobs, forged)
    lease._priming = HybridIndexPriming(manifest=forged, chunks=fabricated, vectors=priming.vectors)
    anchored = seal_contract(
        AlternativeEvidenceRetrievalGeneration,
        "generation_hash",
        **{
            **generation.model_dump(exclude={"generation_hash", "index_manifest_hash"}),
            "index_manifest_hash": forged.logical_hash,
        },
    )
    runtime.retrieval._pending_generations[anchored.generation_hash] = lease

    before = len(passes)
    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=anchored,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        with pytest.raises(KnowledgeRetrievalError) as refused:
            session.search(query="operating", top_k=3)
    finally:
        session.close()
    assert refused.value.failure.code == "retrieval.source_integrity"
    assert len(passes) == before, "refused without a corpus pass"


def test_a_builds_own_priming_opens_without_deriving_the_corpus_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (the shared algorithm and cost initiative, E): the build
    derives the corpus from the authoritative library and proves the
    manifest against it; the reader it primes opens on that proof -- one
    derivation per build-and-open, not two -- while any priming this
    process's build did not mark, or marked for another manifest, is
    re-derived once at open and refused when it does not derive from the
    source. Measured on a real unit the second derivation cost 6.4 s of a
    27.6 s warm selection."""

    runtime, _passes = _counted_runtime(tmp_path)
    derivations: list[str] = []
    original = knowledge_hybrid._build_chunks

    def counted(snapshot: Any, library: Any, spec: Any) -> Any:
        derivations.append(str(snapshot.snapshot_id))
        return original(snapshot, library, spec)

    monkeypatch.setattr(knowledge_hybrid, "_build_chunks", counted)
    request, document_set, generation = _built(runtime)
    assert len(derivations) == 1, "the build derived the corpus once"
    lease = runtime.retrieval._pending_generations[generation.generation_hash]
    priming = lease._priming
    assert priming is not None and priming.derivation_proof is not None
    session = runtime.retrieval.open_session(
        document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
    )
    try:
        hits = session.search(query="operating", top_k=3)
        assert hits.hits, "the primed reader serves the query"
    finally:
        session.close()
    assert len(derivations) == 1, "the primed open rode on the build's derivation"
    # An independent reopen (no build in this process) derives once at open.
    reopened = runtime.retrieval.open_session(
        document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
    )
    try:
        reopened.search(query="operating", top_k=3)
    finally:
        reopened.close()
    assert len(derivations) == 2, "an independent reopen proves the corpus itself"
    # A priming that carries the genuine proof beside a fabricated manifest is
    # not this build's: it is re-derived and refused before any result.
    request2, document_set2, generation2 = _built(
        runtime, entities=("MSFT",), documents=(_recorded_document("MSFT"),)
    )
    lease2 = runtime.retrieval._pending_generations[generation2.generation_hash]
    genuine = lease2._priming
    assert genuine is not None
    library = runtime.documents.library
    snapshot = library.read_snapshot(UUID(str(document_set2.workspace_snapshot_id)))
    spec = HybridIndexSpec.fixed_v2()
    authentic = _build_chunks(snapshot, library, spec.lexical_spec)
    fabricated = tuple(replace(chunk, title="FABRICATED ISSUER TITLE") for chunk in authentic)
    rows = _hybrid_rows(fabricated, spec)
    forged = _manifest(
        corpus_hash=corpus_logical_hash(snapshot),
        document_count=len(snapshot.revisions),
        spec=spec,
        rows=rows,
        commitment=genuine.manifest.vector_commitment,
    )
    database = tmp_path / "workspace" / Path(*forged.database_path.split("/"))
    database.unlink()
    blobs = tuple(serialize_vector(vector) for vector in genuine.vectors)
    WorkspaceHybridKnowledgeIndex._build_database(database, rows, blobs, forged)
    lease2._priming = HybridIndexPriming(
        manifest=forged,
        chunks=fabricated,
        vectors=genuine.vectors,
        derivation_proof=genuine.derivation_proof,
    )
    anchored = seal_contract(
        AlternativeEvidenceRetrievalGeneration,
        "generation_hash",
        **{
            **generation2.model_dump(exclude={"generation_hash", "index_manifest_hash"}),
            "index_manifest_hash": forged.logical_hash,
        },
    )
    runtime.retrieval._pending_generations[anchored.generation_hash] = lease2
    before = len(derivations)
    session2 = runtime.retrieval.open_session(
        document_set=document_set2, generation=anchored, evidence_as_of=request2.evidence_as_of
    )
    try:
        with pytest.raises(KnowledgeRetrievalError) as refused:
            session2.search(query="operating", top_k=3)
    finally:
        session2.close()
    assert refused.value.failure.code == "retrieval.source_integrity"
    assert len(derivations) == before + 1, "a proof for another manifest is re-derived, not trusted"


def test_pair_scores_are_served_only_under_a_sealed_commitment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (review finding R2, reproduced on `ae9bc093`: a block that
    verified against its own name and passed a four-pair sample served an
    unsampled score changed from 0 to 9.9999): a block's name is a lookup
    key, not the authority to serve it. A reader serves only the blocks the
    admission it was opened with names -- the evidence runtime names those
    its sealed commitments bind, each commitment named by the receipt of
    the session that scored them -- and a block on disk under any other
    name (the rewrite under its recomputed name, an injected coherent
    block, a block no commitment names) is counted and never consulted.
    An admitted block that does not verify refuses by name; two admitted
    blocks that score one pair differently refuse by name; an admitted
    block missing from disk is a missing proof, scored again; a reader
    under another context sees none of it; a storage admission that
    refuses leaves the pairs unsealed and the selection whole; a reader
    without an admission keeps its in-process cache only. The four-pair
    sample stays as a drift diagnostic. Exact reuse: a later process with
    the commitment's blocks asks the model only for the sample and gets
    the same ordering."""

    runtime, _passes = _counted_runtime(tmp_path)
    scored: list[int] = []
    original = planted_corpus._FakeReranker.score

    def counted(self: Any, query: str, passages: list[str]) -> tuple[float, ...]:
        scored.append(len(passages))
        return original(self, query, passages)

    monkeypatch.setattr(planted_corpus._FakeReranker, "score", counted)
    monkeypatch.setattr(reranking, "PAIR_SCORE_BLOCK_MINIMUM", 1)
    request, document_set, generation = _built(
        runtime, documents=(_recorded_document(), _one_line_document())
    )
    queries = ("operating", "reports operating detail", "official quarterly update")
    spec = runtime.retrieval.index_spec
    root = tmp_path / "workspace" / PAIR_SCORE_ROOT / reranker_context_hash(spec)

    def search(
        admission: PairScoreAdmission | None,
    ) -> tuple[list[tuple[str, ...]], tuple[Any, ...], dict[str, int]]:
        session = runtime.retrieval.open_session(
            document_set=document_set,
            generation=generation,
            evidence_as_of=request.evidence_as_of,
            pair_scores=admission,
        )
        try:
            hits = [
                tuple(hit.span_handle for hit in session.search(query=query, top_k=3).hits)
                for query in queries
            ]
            sealed = session.seal_pair_scores()
            return hits, sealed, session.pair_score_facts()
        finally:
            session.close()

    # No admission: the in-process cache alone; nothing sealed, nothing read.
    first, sealed, facts = search(None)
    assert sealed == () and facts == {} and not root.exists()
    fresh = sum(scored)
    assert fresh > reranking.PAIR_SCORE_SAMPLE
    # An admission naming nothing yet: the model scores every pair and the
    # reader seals them; the blocks are what the caller commits.
    scored.clear()
    work0 = MODEL_WORK.snapshot()
    again, sealed, facts = search(PairScoreAdmission(admitted=frozenset(), admit=None))
    assert again == first and sum(scored) == fresh
    assert sealed and all(isinstance(b, SealedPairScoreBlock) for b in sealed)
    assert facts["sealed_pairs"] == fresh and facts["hits"] == 0
    work1 = MODEL_WORK.snapshot()
    assert work1["reranker_pair_store_writes"] - work0["reranker_pair_store_writes"] == fresh
    blocks = sorted(root.glob("*.scores"))
    assert [b.name for b in blocks] == sorted(b.name for b in sealed)
    committed = frozenset(b.name for b in sealed)
    # Exact reuse under the commitment: served from the blocks, the model
    # asked only for the drift sample, the same ordering.
    scored.clear()
    served, sealed_again, facts = search(PairScoreAdmission(admitted=committed, admit=None))
    assert served == first and sum(scored) == reranking.PAIR_SCORE_SAMPLE
    assert sealed_again == () and facts["hits"] == fresh and facts["misses"] == 0
    assert facts["blocks"] == len(blocks) and facts["unanchored"] == 0
    # The lead's counterexample: one unsampled score rewritten (0 -> 9.9999)
    # and the block renamed to its new content hash. It verifies against
    # its name and would pass the sample; no commitment names it, so it is
    # counted as unanchored and never consulted -- the admitted original
    # serves. Admitting nothing serves nothing: a block on disk is a claim.
    target = blocks[0]
    content = bytearray(target.read_bytes())
    header_end = content.index(b"\n", len(reranking._BLOCK_MAGIC)) + 1
    entries = content[header_end:]
    last = len(entries) // reranking._ENTRY.itemsize - 1
    offset = header_end + last * reranking._ENTRY.itemsize + 32
    content[offset : offset + 4] = (99999).to_bytes(4, "little", signed=True)
    rewritten = root / (sha256_hex(bytes(content)) + reranking.PAIR_SCORE_BLOCK_SUFFIX)
    rewritten.write_bytes(bytes(content))
    scored.clear()
    served, _sealed, facts = search(PairScoreAdmission(admitted=committed, admit=None))
    assert served == first and facts["unanchored"] == 1 and facts["hits"] == fresh
    scored.clear()
    served, sealed_more, facts = search(PairScoreAdmission(admitted=frozenset(), admit=None))
    assert served == first and sum(scored) == fresh, "no commitment, no reuse"
    assert facts["unanchored"] == len(blocks) + 1 and facts["hits"] == 0
    for block in sealed_more:  # the re-sealed blocks are the same bytes, same names
        assert block.name in committed
    # Two admitted blocks that score one pair differently: a conflict,
    # refused by name, never an arbitrary choice.
    with pytest.raises(KnowledgeRetrievalError) as conflict:
        search(PairScoreAdmission(admitted=committed | {rewritten.name}, admit=None))
    assert conflict.value.failure.code == "retrieval.pair_score_conflict"
    rewritten.unlink()
    # An admitted block whose bytes do not hash to its name refuses by name.
    sealed_bytes = target.read_bytes()
    target.write_bytes(sealed_bytes[:-1] + bytes([sealed_bytes[-1] ^ 0x01]))
    with pytest.raises(KnowledgeRetrievalError) as refused:
        search(PairScoreAdmission(admitted=committed, admit=None))
    assert refused.value.failure.code == "retrieval.pair_score_block_tampered"
    target.write_bytes(sealed_bytes)
    # An admitted block missing from disk is a missing proof: scored again.
    target.unlink()
    scored.clear()
    served, _sealed_missing, facts = search(PairScoreAdmission(admitted=committed, admit=None))
    assert served == first and facts["missing"] == 1 and sum(scored) > reranking.PAIR_SCORE_SAMPLE
    assert target.is_file() and target.read_bytes() == sealed_bytes, "sealed again, same bytes"
    # The sample is a drift diagnostic: a runtime that no longer scores as
    # the sealed context did is refused, whatever the commitment says.
    monkeypatch.setattr(
        planted_corpus._FakeReranker,
        "score",
        lambda self, query, passages: tuple(v + 1.0 for v in original(self, query, passages)),
    )
    with pytest.raises(KnowledgeRetrievalError) as drift:
        search(PairScoreAdmission(admitted=committed, admit=None))
    assert drift.value.failure.code == "retrieval.pair_score_runtime_drift"
    monkeypatch.setattr(planted_corpus._FakeReranker, "score", counted)
    # Another reranker context sees none of these blocks.
    other = reranking.PairScoreStore(
        root.parent / ("0" * 64),
        "0" * 64,
        spec.reranker_score_decimal_places,
        reranking.PAIR_SCORE_STORE_CAP_BYTES,
        admission=PairScoreAdmission(admitted=committed, admit=None),
    )
    other.load()
    assert other.block_count == 0 and other.missing == len(committed)
    # A storage admission that refuses: nothing placed, the pairs unsealed,
    # the selection whole; a full store likewise seals nothing more.
    for block in root.glob("*.scores"):
        block.unlink()

    def refuse(_additional: int) -> None:
        raise RuntimeError("storage.managed_capacity_exceeded")

    scored.clear()
    served, sealed_refused, facts = search(PairScoreAdmission(admitted=frozenset(), admit=refuse))
    assert served == first and sealed_refused == () and not list(root.glob("*.scores"))
    assert facts["unsealed_pairs"] == fresh and facts["sealed_pairs"] == 0
    monkeypatch.setattr(reranking, "PAIR_SCORE_STORE_CAP_BYTES", 1)
    _served, sealed_full, facts = search(PairScoreAdmission(admitted=frozenset(), admit=None))
    assert sealed_full == () and facts["refused_full"] == fresh
    assert not list(root.glob("*.scores"))


def test_the_runtime_commits_the_blocks_a_session_scored_and_the_receipt_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (review findings R2 and R4): the runtime seals what its
    session scored under the workspace's storage admission, commits the
    blocks (`PairScoreCommitmentRecord`, the reranker context bound) and
    names the commitment in the receipt; a later runtime over the same
    workspace admits exactly those blocks and serves another request over
    the same filings from them; a runtime whose storage admission refuses
    seals no block, commits nothing, names nothing, and selects the same
    evidence."""

    runtime, _passes = _counted_runtime(tmp_path)
    scored: list[int] = []
    original = planted_corpus._FakeReranker.score

    def counted(self: Any, query: str, passages: list[str]) -> tuple[float, ...]:
        scored.append(len(passages))
        return original(self, query, passages)

    monkeypatch.setattr(planted_corpus._FakeReranker, "score", counted)
    monkeypatch.setattr(reranking, "PAIR_SCORE_BLOCK_MINIMUM", 1)
    request, document_set, generation = _built(
        runtime, documents=(_recorded_document(), _one_line_document())
    )
    receipt, _spans = runtime.select_evidence(
        request=request, document_set=document_set, generation=generation
    )
    fresh = sum(scored)
    assert fresh > 0 and receipt.pair_score_commitment_hash is not None
    commitment = runtime.artifacts.load(
        "pair-score-commitments", receipt.pair_score_commitment_hash, PairScoreCommitmentRecord
    )
    assert commitment.reranker_context_hash == reranker_context_hash(runtime.retrieval.index_spec)
    assert commitment.retrieval_generation_hash == generation.generation_hash
    assert sum(b.pair_count for b in commitment.blocks) == fresh
    root = tmp_path / "workspace" / PAIR_SCORE_ROOT / commitment.reranker_context_hash
    assert {p.name for p in root.glob("*.scores")} == {b.name for b in commitment.blocks}
    since = commitment.sealed_at
    assert runtime.pair_score_admission(since=since).admitted == frozenset(
        b.name for b in commitment.blocks
    )
    # Another process over the same workspace, another request over the
    # same filings and one more (a later cutoff, another generation, so no
    # sealed selection is reused): its own selection, the shared filings'
    # pairs served from the committed blocks, the new filing's scored by
    # the model and committed in turn.
    later, _passes2 = _counted_runtime(tmp_path)
    later_request = seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        **{
            **request.model_dump(exclude={"request_hash"}),
            "evidence_as_of": request.evidence_as_of + timedelta(days=1),
            "acquisition_deadline": request.acquisition_deadline + timedelta(days=1),
        },
    )
    _snapshot, later_source_set = later.acquire_recorded(
        request=later_request,
        registry=_registry(("AAPL",)),
        documents=(
            _recorded_document(),
            _one_line_document(),
            _recorded_document(
                revision="issuer-release-2026-q4",
                claims=(PLANTED_CLAIMS[EvidenceTopic.LIQUIDITY_GOING_CONCERN],),
            ),
        ),
        published_at=later_request.evidence_as_of,
    )
    later_set = later.canonicalize(
        source_set=later_source_set, published_at=later_request.evidence_as_of
    )
    later_generation = later.build_retrieval(
        document_set=later_set, built_at=later_request.evidence_as_of
    )
    scored.clear()
    later_receipt, _later_spans = later.select_evidence(
        request=later_request, document_set=later_set, generation=later_generation
    )
    assert later_receipt.reused_from_receipt_hash is None, "its own selection"
    facts = later.pair_score_facts
    assert facts["hits"] > 0 and facts["unanchored"] == 0 and facts["missing"] == 0
    assert sum(scored) == facts["misses"] + reranking.PAIR_SCORE_SAMPLE, (scored, facts)
    assert later_receipt.pair_score_commitment_hash is not None, "the new filing's pairs"
    later_commitment = later.artifacts.load(
        "pair-score-commitments",
        later_receipt.pair_score_commitment_hash,
        PairScoreCommitmentRecord,
    )
    assert sum(b.pair_count for b in later_commitment.blocks) == facts["misses"]
    assert later.pair_score_admission(since=since).admitted >= frozenset(
        b.name for b in commitment.blocks
    )
    # A workspace whose storage admission refuses the blocks: the selection
    # completes, no block is placed, no commitment is sealed or named. The
    # admission refuses the routing's comparisons and the session's blocks
    # and admits what follows (the selection's own receipt and span set,
    # which are admitted writes too since section W).
    refusing, _passes3 = _counted_runtime(tmp_path / "refusing")
    admitted: list[int] = []

    def refuse_blocks(additional: int) -> None:
        if not refusing.pair_score_facts.get("unsealed_pairs"):
            raise RuntimeError("storage.managed_capacity_exceeded")
        admitted.append(additional)

    r_request, r_set, r_generation = _built(
        refusing, documents=(_recorded_document(), _one_line_document())
    )
    refusing.storage_admission = refuse_blocks
    r_receipt, r_spans = refusing.select_evidence(
        request=r_request, document_set=r_set, generation=r_generation
    )
    assert r_receipt.pair_score_commitment_hash is None
    assert refusing.pair_score_facts["unsealed_pairs"] > 0
    assert not list((tmp_path / "refusing" / "workspace" / PAIR_SCORE_ROOT).rglob("*.scores"))
    assert refusing.artifacts.values("pair-score-commitments", PairScoreCommitmentRecord) == ()
    assert [s.span_handle for s in r_spans]
    assert admitted, "the receipt and the span set were admitted, and placed"
    assert refusing.artifacts.exists("retrieval-access-receipts", r_receipt.receipt_hash)
    # A workspace whose admission refuses every write refuses the selection
    # by the budget's name: no receipt, no span set, nothing placed.
    denied, _passes4 = _counted_runtime(tmp_path / "denied")
    d_request, d_set, d_generation = _built(
        denied, documents=(_recorded_document(), _one_line_document())
    )

    def refuse(_additional: int) -> None:
        raise RuntimeError("storage.managed_capacity_exceeded")

    denied.storage_admission = refuse
    with pytest.raises(RuntimeError, match="managed_capacity_exceeded"):
        denied.select_evidence(request=d_request, document_set=d_set, generation=d_generation)
    assert (
        denied.artifacts.values(
            "retrieval-access-receipts", AlternativeEvidenceRetrievalAccessReceipt
        )
        == ()
    )
    assert not list((tmp_path / "denied" / "workspace" / PAIR_SCORE_ROOT).rglob("*.scores"))


def test_a_priming_proof_binds_the_chunks_the_reader_consumes(tmp_path: Path) -> None:
    """requirement (review finding R5): the build's derivation proof names
    the manifest and the snapshot, while the primed open consumed
    `priming.chunks` -- so a priming carrying the genuine manifest, vectors
    and proof beside altered or reordered chunks would hand the inventory
    and the scorer altered bodies as authoritative. The proof now stands
    only for the chunks that reproduce the manifest's source commitments
    (chunk ids, citations, embedding inputs, the projection), checked at
    the open without deriving the corpus; a changed body, a swapped body
    and a reordered chunk each refuse `retrieval.source_integrity` before
    any inventory or hit."""

    runtime, _passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(
        runtime, documents=(_recorded_document(), _one_line_document())
    )
    lease = runtime.retrieval._pending_generations[generation.generation_hash]
    genuine = lease._priming
    assert genuine is not None and genuine.derivation_proof is not None

    def opened_with(chunks: tuple[Any, ...]) -> Any:
        lease._priming = HybridIndexPriming(
            manifest=genuine.manifest,
            chunks=chunks,
            vectors=genuine.vectors,
            derivation_proof=genuine.derivation_proof,
        )
        runtime.retrieval._pending_generations[generation.generation_hash] = lease
        return runtime.retrieval.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )

    # A body altered under its own chunk id: the same count, the genuine
    # manifest, vectors and proof.
    altered = list(genuine.chunks)
    victim = altered[len(altered) // 2]
    altered[len(altered) // 2] = replace(
        victim, body=victim.body.replace("operating", "no material exposure")
    )
    session = opened_with(tuple(altered))
    try:
        with pytest.raises(KnowledgeRetrievalError) as refused:
            session.search(query="operating", top_k=3)
    finally:
        session.close()
    assert refused.value.failure.code == "retrieval.source_integrity"
    # Two bodies swapped between chunk ids.
    swapped = list(genuine.chunks)
    a, b = swapped[0], swapped[1]
    swapped[0], swapped[1] = replace(a, body=b.body), replace(b, body=a.body)
    session = opened_with(tuple(swapped))
    try:
        with pytest.raises(KnowledgeRetrievalError) as refused:
            session.search(query="operating", top_k=3)
    finally:
        session.close()
    assert refused.value.failure.code == "retrieval.source_integrity"
    # The chunks reordered whole.
    session = opened_with(tuple(reversed(genuine.chunks)))
    try:
        with pytest.raises(KnowledgeRetrievalError) as refused:
            session.search(query="operating", top_k=3)
    finally:
        session.close()
    assert refused.value.failure.code == "retrieval.source_integrity"
    # The genuine chunks open on the proof and serve the inventory and the hits.
    session = opened_with(genuine.chunks)
    try:
        assert session.search(query="operating", top_k=3).hits
        windows = session.retriever.inventory(
            session._binding_request(query="structural window inventory", top_k=1)
        )
        assert windows and all(
            w.body == next(c for c in genuine.chunks if c.chunk_id == w.citation.chunk_id).body
            for w in windows
        )
    finally:
        session.close()


def test_a_projection_is_proved_at_first_open_on_the_reader_s_own_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (record section Y, R0; the lead's reproduction on
    `4b0ba487`): a build's proof of the published projection is not the
    reader's. Rows deleted from the file between a build's materialization
    and the reader's first open never become hits, so no per-hit check can
    refuse their omission -- the primed open proves the projection on its
    own connection and refuses the damaged generation before any result:
    every content row deleted under an unchanged manifest; a term-index row
    deleted under an unchanged manifest; while an ordinary build and open
    proves once at the build and once at the open, a lease released without
    a session holds no handle on the index, an open that fails after the
    proof closes its connection, and an independent reopen agrees with the
    primed reader."""

    import sqlite3

    proofs: list[bool] = []
    original_verify = knowledge_hybrid._verify_projection

    def counted_verify(path: Path, manifest: Any, **kwargs: Any) -> Any:
        proofs.append(bool(kwargs.get("keep_open")))
        return original_verify(path, manifest, **kwargs)

    monkeypatch.setattr(knowledge_hybrid, "_verify_projection", counted_verify)

    def prepared(root: Path) -> tuple[Any, Any, Any, Any, Path]:
        runtime, _passes = _counted_runtime(root)
        request, document_set, generation = _built(runtime)
        database = next((root / "workspace" / ".system" / "knowledge-indexes").rglob("*.db"))
        return runtime, request, document_set, generation, database

    def opened(runtime: Any, request: Any, document_set: Any, generation: Any) -> Any:
        return runtime.retrieval.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )

    # The ordinary lifecycle: the first build publishes and the open proves
    # the projection (one proof, kept open); the same generation built again
    # materializes against the sealed record (one proof, closed) and its
    # open proves the projection again on the reader's own connection; the
    # independent reopen agrees with the primed reader.
    runtime, request, document_set, generation, database = prepared(tmp_path / "ordinary")
    try:
        session = opened(runtime, request, document_set, generation)
        try:
            primed_hits = [hit.rank for hit in session.search(query="operating", top_k=3).hits]
            assert primed_hits
        finally:
            session.close()
        assert proofs.count(True) == 1, "the open proved the projection on its own connection"
        proved_by_builds = proofs.count(False)
        again = runtime.build_retrieval(document_set=document_set, built_at=_NOW)
        assert again.generation_hash == generation.generation_hash
        priming = runtime.retrieval._pending_generations[generation.generation_hash]._priming
        assert priming is not None and not hasattr(priming, "verified")
        assert proofs.count(False) == proved_by_builds + 1, "the materialization proved the index"
        assert proofs.count(True) == 1, "and kept nothing for the reader"
        session = opened(runtime, request, document_set, generation)
        try:
            assert session.search(query="operating", top_k=3).hits
        finally:
            session.close()
        assert proofs.count(True) == 2, "the open proved the projection again, on its own"
        reopened = opened(runtime, request, document_set, generation)
        try:
            assert [
                hit.rank for hit in reopened.search(query="operating", top_k=3).hits
            ] == primed_hits
        finally:
            reopened.close()
        # A lease released without a session holds no handle on the index.
        runtime.build_retrieval(document_set=document_set, built_at=_NOW)
        runtime.retrieval.release_without_session(document_set=document_set, generation=generation)
        moved = database.with_suffix(".moved")
        database.rename(moved)
        moved.rename(database)
        # An open that fails after the projection was proved closes its
        # connection: nothing holds the index afterwards.
        runtime.build_retrieval(document_set=document_set, built_at=_NOW)
        failing = opened(runtime, request, document_set, generation)
        monkeypatch.setattr(
            failing.retriever, "_adapter_factory", lambda _root, _spec: 1 / 0, raising=False
        )
        try:
            with pytest.raises(ZeroDivisionError):
                failing.search(query="operating", top_k=3)
        finally:
            failing.close()
        database.rename(moved)
        moved.rename(database)
    finally:
        runtime.close()

    # The lead's reproduction: every content row deleted between the
    # materialization and the first open, the manifest unchanged. Refused
    # before any result, on the reader's own connection.
    runtime, request, document_set, generation, database = prepared(tmp_path / "omitted")
    try:
        runtime.build_retrieval(document_set=document_set, built_at=_NOW)
        with closing(sqlite3.connect(database)) as connection:
            assert connection.execute("SELECT COUNT(*) FROM content").fetchone()[0] > 0
            connection.execute("DELETE FROM content")
            connection.commit()
        session = opened(runtime, request, document_set, generation)
        try:
            with pytest.raises(KnowledgeRetrievalError) as refused:
                session.search(query="operating", top_k=3)
        finally:
            session.close()
        assert refused.value.failure.code == "retrieval.index_corrupt"
    finally:
        runtime.close()

    # A term-index row deleted under an unchanged manifest: refused the same
    # way, before any result.
    runtime, request, document_set, generation, database = prepared(tmp_path / "postings")
    try:
        runtime.build_retrieval(document_set=document_set, built_at=_NOW)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                "DELETE FROM term_fts WHERE rowid = (SELECT MIN(rowid) FROM term_fts)"
            )
            connection.commit()
        session = opened(runtime, request, document_set, generation)
        try:
            with pytest.raises(KnowledgeRetrievalError) as refused:
                session.search(query="operating", top_k=3)
        finally:
            session.close()
        assert refused.value.failure.code == "retrieval.index_corrupt"
    finally:
        runtime.close()
