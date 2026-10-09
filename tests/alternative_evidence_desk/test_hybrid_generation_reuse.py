"""Corpus-keyed generations and committed vectors.

A request at another cutoff over the same eligible bytes reuses the physical
generation and embeds nothing; a changed corpus is another generation; a new
process reopens a generation by verifying its committed vectors, not by
running the model; every way the commitment can be broken -- payload bytes,
a missing payload, vectors edited together with their in-database hashes, a
reordered mapping, a rewritten manifest, a corrupted posting list -- refuses
by name and reopens once restored; a legacy generation still opens through
its own path, at the one pass that path costs.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
    AlternativeEvidenceRetrievalGeneration,
    HybridV3RetrievalGeneration,
    parse_retrieval_generation,
)
from alphalattice.kernel.knowledge._embeddings import (
    load_vector_extension,
)
from alphalattice.kernel.knowledge.hybrid import (
    WorkspaceHybridKnowledgeIndex,
    _hybrid_rows,
    index_identity_of_database,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    RECIPE_MINILM_CPU,
    HybridIndexEvictionMarker,
    HybridIndexSpec,
    HybridKnowledgeIndexManifest,
)
from alphalattice.kernel.knowledge.retrieval import _build_chunks
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from tests.alternative_evidence_desk.document_intelligence_support import (
    _built,
    _corpus_passes,
    _counted_runtime,
    _registry,
)
from tests.alternative_evidence_desk.planted_corpus import _NOW, _recorded_document


def _release(*args: Any, **kwargs: Any) -> Any:
    """The planted release in its original form, for the tests whose ties and
    block keys were stated on its bytes: they build and search the index and
    never run a selection, so the form the integrated selection reads (a
    current report, the planted corpus's default since T5) would only move
    the fixtures."""

    return _recorded_document(*args, **{"document_type": "EARNINGS_RELEASE", **kwargs})


_QUERY = "operating"


def _request_at(cutoff: Any) -> AlternativeEvidenceRequest:
    return seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        ordered_entity_ids=("AAPL",),
        evidence_as_of=cutoff,
        acquisition_deadline=cutoff + timedelta(hours=1),
        evidence_classes=(AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,),
        source_policy=AlternativeEvidenceSourcePolicy(),
        ttl_seconds=86_400,
        mode=AlternativeEvidenceMode.RECORDED,
    )


def _prepare(runtime: Any, request: AlternativeEvidenceRequest, documents: tuple[Any, ...]) -> Any:
    """Acquire, canonicalize and build for one request; returns the document set and generation."""

    _snapshot, source_set = runtime.acquire_recorded(
        request=request,
        registry=_registry(),
        documents=documents,
        published_at=request.evidence_as_of,
    )
    document_set = runtime.canonicalize(source_set=source_set, published_at=request.evidence_as_of)
    generation = runtime.build_retrieval(document_set=document_set, built_at=request.evidence_as_of)
    return document_set, generation


def _hits(runtime: Any, request: Any, document_set: Any, generation: Any) -> tuple[Any, ...]:
    session = runtime.retrieval.open_session(
        document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
    )
    try:
        result = session.search(query=_QUERY, top_k=3)
        return tuple(
            (session.span_identity(hit.span_handle), hit.rank, round(hit.retrieval_score, 6))
            for hit in result.hits
        )
    finally:
        session.close()


def _physical(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    system = tmp_path / "workspace" / ".system"
    return (
        sorted((system / "knowledge-indexes").rglob("*.db")),
        sorted((system / "knowledge-vectors").glob("*.f32"))
        if (system / "knowledge-vectors").is_dir()
        else [],
    )


def test_a_new_cutoff_over_the_same_corpus_reuses_the_generation(tmp_path: Path) -> None:
    """requirement: another cutoff, same eligible bytes and spec -> one physical
    generation, zero new passage embeddings, its own question and provenance."""

    runtime, passes = _counted_runtime(tmp_path)
    first_request = _request_at(_NOW)
    first_set, first = _prepare(runtime, first_request, (_recorded_document(),))
    assert runtime.retrieval.passage_embedding_pass_count == 1
    first_hits = _hits(runtime, first_request, first_set, first)
    assert first_hits

    later_request = _request_at(_NOW + timedelta(days=1))
    later_set, later = _prepare(runtime, later_request, (_recorded_document(),))

    # A new question: its own request, document set, snapshot and record.
    assert later_request.request_hash != first_request.request_hash
    assert later_set.document_set_hash != first_set.document_set_hash
    assert later_set.workspace_snapshot_id != first_set.workspace_snapshot_id
    assert later.generation_hash != first.generation_hash
    assert later.built_at == later_request.evidence_as_of
    # The same physical generation, with nothing embedded for it.
    assert later.index_id == first.index_id
    assert later.corpus_hash == first.corpus_hash
    assert later.index_manifest_hash == first.index_manifest_hash
    assert later.vector_payload_sha256 == first.vector_payload_sha256
    assert runtime.retrieval.passage_embedding_pass_count == 1
    assert runtime.retrieval.generation_reuse_count == 1
    assert _corpus_passes(passes, first) == 1
    databases, payloads = _physical(tmp_path)
    assert len(databases) == 1 and len(payloads) == 1
    assert payloads[0].name == f"{first.vector_payload_sha256}.f32"
    # The reused generation answers the new question with the same hits and
    # the new snapshot on its citations.
    assert _hits(runtime, later_request, later_set, later) == first_hits
    assert _corpus_passes(passes, first) == 1


def _vector_root(tmp_path: Path) -> Path:
    return tmp_path / "workspace" / ".system" / "knowledge-vectors"


def _sidecars(tmp_path: Path) -> list[Path]:
    root = _vector_root(tmp_path)
    return sorted(root.glob("*.blocks.json")) if root.is_dir() else []


def _counts(runtime: Any) -> dict[str, int]:
    service = runtime.retrieval
    return {
        "passes": service.passage_embedding_pass_count,
        "incremental": service.incremental_build_count,
        "embedded": service.embedded_chunk_count,
        "reused": service.reused_chunk_count,
        "calls": service.embedding_call_count,
        "generation_reuse": service.generation_reuse_count,
    }


def test_a_changed_corpus_is_another_generation_composed_from_committed_blocks(
    tmp_path: Path,
) -> None:
    """requirement: adding a document embeds only that document; revising one
    re-embeds only the revision; removing one embeds nothing; every corpus is
    its own generation with its own question, and its vectors are the same
    numbers a full pass would give. Blocks come only from sealed generations."""

    runtime, passes = _counted_runtime(tmp_path)
    a = _release()
    b = _release(revision="issuer-release-2026-q2", claims=("Guidance was raised.",))
    c = _release(revision="issuer-release-2026-q1", claims=("Inventory fell.",))
    d = _release(revision="issuer-release-2025-q4", claims=("A plant was sold.",))

    # A/B/C: the first generation, every chunk embedded, one call per revision.
    request = _request_at(_NOW)
    _set, first = _prepare(runtime, request, (a, b, c))
    counts = _counts(runtime)
    assert counts == {
        "passes": 1,
        "incremental": 0,
        "embedded": first.chunk_count,
        "reused": 0,
        "calls": 3,
        "generation_reuse": 0,
    }
    assert len(passes) == 3 and sum(passes) == first.chunk_count
    assert len(_sidecars(tmp_path)) == 1
    first_hits = _hits(runtime, request, _set, first)

    # A/B/C/D: only D goes through the model; A, B and C come from the sealed
    # generation's blocks, proved as a whole. Another corpus, another question.
    later = _request_at(_NOW + timedelta(days=1))
    later_set, second = _prepare(runtime, later, (a, b, c, d))
    assert second.corpus_hash != first.corpus_hash and second.index_id != first.index_id
    assert second.chunk_count > first.chunk_count
    d_chunks = second.chunk_count - first.chunk_count
    assert _counts(runtime) == {
        "passes": 1,
        "incremental": 1,
        "embedded": first.chunk_count + d_chunks,
        "reused": first.chunk_count,
        "calls": 4,
        "generation_reuse": 0,
    }
    assert passes[3:] == [d_chunks], "one call, D's chunks only"
    # The composed generation is what a full pass gives: the same vectors for
    # A, B and C, so the hits over the shared passages are the first's.
    assert _hits(runtime, later, later_set, second)[: len(first_hits)] == first_hits
    # The database holds A/B/C/D; the store holds one block per distinct
    # revision (the synthetic encoder repeats vectors, so fewer files) and
    # two compositions; no payload was written as a whole.
    databases, _payloads = _physical(tmp_path)
    assert len(databases) == 2 and len(_sidecars(tmp_path)) == 2
    assert not (_vector_root(tmp_path) / f"{second.vector_payload_sha256}.f32").is_file()

    # B revised: B' is embedded, A, C and D are not.
    b_revised = _release(
        revision="issuer-release-2026-q2-amended", claims=("Guidance was raised twice.",)
    )
    revised_request = _request_at(_NOW + timedelta(days=2))
    _revised_set, third = _prepare(runtime, revised_request, (a, b_revised, c, d))
    assert third.corpus_hash != second.corpus_hash
    embedded_before = _counts(runtime)["embedded"]
    counts = _counts(runtime)
    assert counts["incremental"] == 2 and counts["passes"] == 1
    assert counts["embedded"] - (first.chunk_count + d_chunks) == passes[-1]
    assert passes[4:] == [passes[4]] and 0 < passes[4] < third.chunk_count
    assert counts["reused"] == first.chunk_count + (third.chunk_count - passes[4])
    del embedded_before

    # D removed from the corpus: nothing is embedded; the older snapshots and
    # their generations still read.
    removed_request = _request_at(_NOW + timedelta(days=3))
    _removed_set, fourth = _prepare(runtime, removed_request, (a, b_revised, c))
    assert fourth.corpus_hash != third.corpus_hash
    assert _counts(runtime)["embedded"] == counts["embedded"], "no model run"
    assert _counts(runtime)["calls"] == counts["calls"]
    assert len(passes) == 5
    assert _hits(runtime, later, later_set, second)[: len(first_hits)] == first_hits
    assert _hits(runtime, request, _set, first) == first_hits

    # A/B/C again at a later cutoff: the first generation itself is reused.
    again = _request_at(_NOW + timedelta(days=4))
    _again_set, fifth = _prepare(runtime, again, (a, b, c))
    assert fifth.index_id == first.index_id
    assert _counts(runtime)["generation_reuse"] == 1 and len(passes) == 5
    runtime.close()


def test_blocks_are_reused_only_under_the_same_encoder_context(tmp_path: Path) -> None:
    """requirement: a changed model, tokenizer, numerical profile or batch
    policy is another encoder context; text alone never justifies reuse."""

    from alphalattice.kernel.knowledge.hybrid_contracts import (
        embedding_context_hash,
        vector_asset_key,
    )

    runtime, passes = _counted_runtime(tmp_path)
    request = _request_at(_NOW)
    _set, first = _prepare(runtime, request, (_recorded_document(),))
    assert len(passes) == 1
    spec = runtime.retrieval.index_spec
    context = embedding_context_hash(spec)
    other = spec.model_copy(update={"batch_size": spec.batch_size + 1})
    assert embedding_context_hash(other) != context
    same_inputs = ("a" * 64, "b" * 64)
    assert vector_asset_key(context, same_inputs) != vector_asset_key(
        embedding_context_hash(other), same_inputs
    )
    assert vector_asset_key(context, same_inputs) != vector_asset_key(
        context, tuple(reversed(same_inputs))
    ), "the order of the inputs is part of the key"
    # A source asked under another context finds nothing in the sealed store.
    source = runtime.vector_block_source(_set)
    from uuid import UUID

    from alphalattice.kernel.knowledge.hybrid import _hybrid_rows
    from alphalattice.kernel.knowledge.retrieval import _build_chunks

    document_set = runtime.artifacts.load(
        "document-sets",
        first.document_set_hash,
        __import__(
            "alphalattice.evidence.alternative_evidence.documents.contracts",
            fromlist=["AlternativeEvidenceDocumentSet"],
        ).AlternativeEvidenceDocumentSet,
    )
    snapshot = runtime.documents.library.read_snapshot(UUID(document_set.workspace_snapshot_id))
    chunks = _build_chunks(snapshot, runtime.documents.library, spec.lexical_spec)
    inputs = tuple(str(row["embedding_input_hash"]) for row in _hybrid_rows(chunks, spec))
    assert source(context, inputs) is not None, "the sealed generation serves its block"
    assert source(embedding_context_hash(other), inputs) is None
    assert source(context, inputs[:-1]) is None, "a different revision's inputs are not a block"
    runtime.close()


def test_a_single_payload_generation_serves_slices_and_a_tampered_block_serves_nothing(
    tmp_path: Path,
) -> None:
    """requirement: a generation written as one payload (the retained v4 form)
    serves a revision's vectors as a slice cut by its anchored manifest; a
    block or sidecar that does not verify against its sealed record serves
    nothing and the build embeds instead of trusting it."""

    from alphalattice.kernel.knowledge.hybrid_contracts import (
        HybridVectorBlockSidecar,
    )
    from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes

    runtime, passes = _counted_runtime(tmp_path)
    a = _recorded_document()
    b = _recorded_document(revision="issuer-release-2026-q2", claims=("Guidance was raised.",))
    request = _request_at(_NOW)
    _set, first = _prepare(runtime, request, (a, b))
    assert len(passes) == 2
    root = _vector_root(tmp_path)
    (sidecar,) = _sidecars(tmp_path)
    # Rewrite the generation as one payload file, as the predecessor wrote it:
    # the blocks concatenated at the payload's address, no sidecar.
    parsed = HybridVectorBlockSidecar.model_validate_json(sidecar.read_bytes())
    payload = b"".join((root / f"{block.block_sha256}.f32").read_bytes() for block in parsed.blocks)
    (root / f"{first.vector_payload_sha256}.f32").write_bytes(payload)
    sidecar.unlink()
    for block in parsed.blocks:
        (root / f"{block.block_sha256}.f32").unlink()
    # A/B/C: A and B are cut from the single payload through the sealed
    # manifest; only C is embedded.
    c = _recorded_document(revision="issuer-release-2026-q1", claims=("Inventory fell.",))
    later = _request_at(_NOW + timedelta(days=1))
    _later_set, second = _prepare(runtime, later, (a, b, c))
    assert len(passes) == 3 and passes[-1] == second.chunk_count - first.chunk_count
    assert runtime.retrieval.reused_chunk_count == first.chunk_count
    runtime.close()

    # A block of the new generation rewritten coherently with its sidecar:
    # the composition no longer hashes to the sealed digest, so the store
    # serves nothing from it and the next corpus embeds every revision.
    restarted, later_passes = _counted_runtime(tmp_path)
    (sidecar,) = _sidecars(tmp_path)
    parsed = HybridVectorBlockSidecar.model_validate_json(sidecar.read_bytes())
    victim = parsed.blocks[-1]
    content = bytearray((root / f"{victim.block_sha256}.f32").read_bytes())
    content[0] ^= 0x01
    from alphalattice.kernel.shared_kernel.domain.serialization import sha256_hex

    altered_sha = sha256_hex(bytes(content))
    (root / f"{altered_sha}.f32").write_bytes(bytes(content))
    body = parsed.model_dump(mode="python")
    body["blocks"] = (*body["blocks"][:-1], {**body["blocks"][-1], "block_sha256": altered_sha})
    sidecar.write_bytes(canonical_json_bytes(HybridVectorBlockSidecar.model_validate(body)))
    d = _recorded_document(revision="issuer-release-2025-q4", claims=("A plant was sold.",))
    fourth = _request_at(_NOW + timedelta(days=2))
    _fourth_set, generation = _prepare(restarted, fourth, (a, b, c, d))
    # A and B still come from the untouched single-payload generation; C,
    # held only by the rewritten composition, is embedded again, as is D.
    assert sum(later_passes) == generation.chunk_count - first.chunk_count, (
        "nothing was trusted from the rewrite"
    )
    assert restarted.retrieval.reused_chunk_count == first.chunk_count
    # And the rewritten generation refuses its own cold open.
    with pytest.raises(KnowledgeRetrievalError) as refused:
        _hits(restarted, later, _later_set, second)
    assert refused.value.failure.code == "retrieval.vector_commitment_mismatch"
    restarted.close()


def test_a_new_process_reopens_the_committed_generation_without_embedding(
    tmp_path: Path,
) -> None:
    """requirement: cold reopen in another process verifies the commitment; no model."""

    runtime, passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)
    primed = _hits(runtime, request, document_set, generation)
    assert _corpus_passes(passes, generation) == 1
    runtime.close()

    restarted, later_passes = _counted_runtime(tmp_path)
    assert restarted.retrieval.passage_embedding_pass_count == 0
    cold = _hits(restarted, request, document_set, generation)
    assert cold == primed
    assert later_passes == [], "the reopened generation ran no passage embedding"
    assert restarted.retrieval.independent_generation_reopen_count == 1


def _manifest_of(database: Path) -> HybridKnowledgeIndexManifest:
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        raw = connection.execute("SELECT value FROM metadata WHERE key = 'manifest'").fetchone()[0]
    return HybridKnowledgeIndexManifest.model_validate_json(raw)


def test_a_broken_commitment_refuses_by_name_and_a_restored_one_reopens(
    tmp_path: Path,
) -> None:
    """requirement: payload bytes, a missing payload, vectors edited with their
    in-database hashes, a reordered mapping, a rewritten manifest and a
    corrupted posting list cannot open; restoring the bytes reopens."""

    runtime, _passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)
    _hits(runtime, request, document_set, generation)
    runtime.close()
    databases, payloads = _physical(tmp_path)
    database, payload = databases[0], payloads[0]
    database_bytes, payload_bytes = database.read_bytes(), payload.read_bytes()

    def cold() -> str | None:
        restarted, later_passes = _counted_runtime(tmp_path)
        try:
            _hits(restarted, request, document_set, generation)
        except KnowledgeRetrievalError as refused:
            assert later_passes == [], "a refusal costs no corpus pass"
            return str(refused.failure.code)
        finally:
            restarted.close()
        assert later_passes == []
        return None

    # (a) one byte of the payload
    flipped = bytearray(payload_bytes)
    flipped[len(flipped) // 2] ^= 0x01
    payload.write_bytes(bytes(flipped))
    assert cold() == "retrieval.vector_commitment_mismatch"
    payload.write_bytes(payload_bytes)
    assert cold() is None

    # (b) no payload at all
    payload.unlink()
    assert cold() == "retrieval.vector_payload_unavailable"
    payload.write_bytes(payload_bytes)
    assert cold() is None

    # (c) a vector edited together with the hash the database keeps beside it
    with closing(sqlite3.connect(database)) as connection:
        load_vector_extension(connection)
        row = connection.execute(
            "SELECT rowid, embedding FROM vec_chunks ORDER BY rowid"
        ).fetchone()
        blob = bytearray(row[1])
        blob[0] ^= 0x01
        edited = bytes(blob)
        connection.execute("DELETE FROM vec_chunks WHERE rowid = ?", (row[0],))
        connection.execute(
            "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)", (row[0], edited)
        )
        from alphalattice.kernel.shared_kernel.domain.serialization import sha256_hex

        connection.execute(
            "UPDATE dense_nodes SET vector_hash = ? WHERE rowid = ?", (sha256_hex(edited), row[0])
        )
        connection.commit()
    assert cold() == "retrieval.vector_commitment_mismatch"
    database.write_bytes(database_bytes)
    assert cold() is None

    # (d) two vectors swapped, hashes swapped with them: a reordered mapping
    with closing(sqlite3.connect(database)) as connection:
        load_vector_extension(connection)
        stored = connection.execute(
            "SELECT v.rowid, v.embedding, n.vector_hash FROM vec_chunks AS v "
            "JOIN dense_nodes AS n ON n.rowid = v.rowid ORDER BY v.rowid"
        ).fetchall()
        # Two rows whose vectors differ; the synthetic embedding gives some
        # passages one vector, and swapping equal vectors changes nothing.
        rows = [stored[0], next(row for row in stored[1:] if row[1] != stored[0][1])]
        for (rowid, _blob, _digest), (_other, blob, digest) in (
            (rows[0], rows[1]),
            (rows[1], rows[0]),
        ):
            connection.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rowid,))
            connection.execute(
                "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)", (rowid, blob)
            )
            connection.execute(
                "UPDATE dense_nodes SET vector_hash = ? WHERE rowid = ?", (digest, rowid)
            )
        connection.commit()
    assert cold() == "retrieval.vector_commitment_mismatch"
    database.write_bytes(database_bytes)
    assert cold() is None

    # (e) the manifest inside the database rewritten coherently: the caller's
    # sealed expectation of it is the authority, not the database
    manifest = _manifest_of(database)
    from alphalattice.kernel.knowledge.retrieval import _canonical_hash
    from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes

    forged_payload = manifest.model_dump(mode="python", exclude={"logical_hash"})
    forged_payload["document_count"] = manifest.document_count + 1
    forged = HybridKnowledgeIndexManifest.model_validate(
        {**forged_payload, "logical_hash": _canonical_hash(forged_payload)}
    )
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'manifest'",
            (canonical_json_bytes(forged).decode("utf-8"),),
        )
        connection.commit()
    assert cold() == "retrieval.snapshot_mismatch"
    database.write_bytes(database_bytes)
    assert cold() is None

    # (f) one byte of a term posting list: the content join still passes,
    # the FTS5 checksum does not
    with closing(sqlite3.connect(database)) as connection:
        segment = connection.execute(
            "SELECT id, block FROM term_fts_data ORDER BY length(block) DESC LIMIT 1"
        ).fetchone()
        block = bytearray(segment[1])
        block[len(block) // 2] ^= 0xFF
        connection.execute(
            "UPDATE term_fts_data SET block = ? WHERE id = ?", (bytes(block), segment[0])
        )
        connection.commit()
    assert cold() == "retrieval.index_corrupt"
    database.write_bytes(database_bytes)
    assert cold() is None


def _rewrite_coherently(database: Path, payload: Path) -> tuple[bytes, str, str]:
    """Other vectors at every layer the database and its neighbour control.

    A new payload at its own content address, the vec0 rows, the hashes
    beside them and the in-database manifest (its commitment and its logical
    hash) all rewritten to agree; the source and chunk commitments untouched.
    Returns the altered payload bytes, its digest and the rewritten manifest
    hash.
    """

    from alphalattice.kernel.knowledge.hybrid_contracts import HybridVectorCommitment
    from alphalattice.kernel.knowledge.retrieval import _canonical_hash
    from alphalattice.kernel.shared_kernel.domain.serialization import (
        canonical_json_bytes,
        sha256_hex,
    )

    manifest = _manifest_of(database)
    width = manifest.index_spec.embedding_dimension * 4
    altered = bytearray(payload.read_bytes())
    for index in range(manifest.chunk_count):
        altered[index * width + 2] ^= 0x40
    altered_bytes = bytes(altered)
    altered_sha = sha256_hex(altered_bytes)
    payload.with_name(f"{altered_sha}.f32").write_bytes(altered_bytes)
    commitment = HybridVectorCommitment.model_validate(
        {
            **manifest.vector_commitment.model_dump(mode="python"),
            "payload_sha256": altered_sha,
            "payload_path": f".system/knowledge-vectors/{altered_sha}.f32",
        }
    )
    body = manifest.model_dump(mode="python", exclude={"logical_hash"})
    body["vector_commitment"] = commitment.model_dump(mode="python")
    rewritten = HybridKnowledgeIndexManifest.model_validate(
        {**body, "logical_hash": _canonical_hash(body)}
    )
    assert rewritten.source_commitments() == manifest.source_commitments()
    with closing(sqlite3.connect(database)) as connection:
        load_vector_extension(connection)
        for index in range(manifest.chunk_count):
            blob = altered_bytes[index * width : (index + 1) * width]
            connection.execute("DELETE FROM vec_chunks WHERE rowid = ?", (index + 1,))
            connection.execute(
                "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)", (index + 1, blob)
            )
            connection.execute(
                "UPDATE dense_nodes SET vector_hash = ? WHERE rowid = ?",
                (sha256_hex(blob), index + 1),
            )
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'manifest'",
            (canonical_json_bytes(rewritten).decode("utf-8"),),
        )
        connection.commit()
    return altered_bytes, altered_sha, rewritten.logical_hash


def test_a_coherent_rewrite_cannot_be_laundered_through_a_new_cutoff(tmp_path: Path) -> None:
    """requirement: reuse and restoration are anchored to the commitment the
    sealed generation record holds, never to the database, its manifest or a
    cleanup marker -- which can all be rewritten together. A new cutoff over
    an index rewritten coherently refuses and seals nothing; an explicit
    rebuild repairs the index from the committed payload; a marker that lies
    is ignored in favour of the record; two records that disagree refuse.
    """

    runtime, _passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)
    first_hits = _hits(runtime, request, document_set, generation)
    runtime.close()
    databases, payloads = _physical(tmp_path)
    database, payload = databases[0], payloads[0]
    genuine_bytes = payload.read_bytes()

    _altered_bytes, altered_sha, altered_manifest = _rewrite_coherently(database, payload)
    assert _manifest_of(database).logical_hash == altered_manifest

    restarted, later_passes = _counted_runtime(tmp_path)
    # The original generation's cold read refuses: its record names M1/.
    with pytest.raises(KnowledgeRetrievalError) as refused:
        _hits(restarted, request, document_set, generation)
    assert refused.value.failure.code == "retrieval.snapshot_mismatch"

    # A genuinely new cutoff over the same corpus does not accept the rewrite:
    # the sealed record anchors it, the index differs, the build refuses and
    # seals no record naming the altered vectors.
    later = _request_at(_NOW + timedelta(days=1))
    records_before = sorted((restarted.artifacts.root / "retrieval-generations").glob("*.json"))
    with pytest.raises(KnowledgeRetrievalError) as laundering:
        _prepare(restarted, later, (_recorded_document(),))
    assert laundering.value.failure.code == "retrieval.snapshot_mismatch"
    assert later_passes == [], "no pass is spent on a store that does not verify"
    records_after = sorted((restarted.artifacts.root / "retrieval-generations").glob("*.json"))
    assert records_after == records_before
    assert restarted.retrieval.generation_reuse_count == 0
    assert all(
        restarted.artifacts.load_retrieval_generation(path.stem).vector_payload_sha256
        != altered_sha
        for path in records_after
    )

    # The explicit rebuild repairs the index from the committed payload:
    # no model, the committed manifest back in place, the original hits.
    facts = restarted.rebuild_retrieval(generation.index_id)
    assert facts["passage_embedding_pass"] is False
    assert facts["index_manifest_hash"] == generation.index_manifest_hash
    assert facts["vector_payload_sha256"] == generation.vector_payload_sha256
    assert _manifest_of(database).logical_hash == generation.index_manifest_hash
    assert _hits(restarted, request, document_set, generation) == first_hits
    assert later_passes == []
    # And the new cutoff now reuses the committed generation with no pass.
    later_set, later_generation = _prepare(restarted, later, (_recorded_document(),))
    assert later_generation.index_manifest_hash == generation.index_manifest_hash
    assert later_generation.vector_payload_sha256 == generation.vector_payload_sha256
    assert later_passes == [] and restarted.retrieval.generation_reuse_count == 1
    assert _hits(restarted, later, later_set, later_generation) == first_hits
    restarted.close()

    # Marker boundary: evict the index, then make the marker name the altered
    # vectors. Restoration follows the sealed records, not the marker.
    reopened, passes_again = _counted_runtime(tmp_path)
    relative = database.resolve().relative_to((tmp_path / "workspace").resolve()).as_posix()
    assert index_identity_of_database(relative) == generation.index_id, (
        "the identity a lease holds is the one the index file names"
    )
    reopened.retrieval.evict_generation(relative, plan_hash="a" * 64, evicted_at=_NOW)
    assert not database.is_file()
    marker_path = database.with_name(database.name + ".evicted.json")
    marker = HybridIndexEvictionMarker.model_validate_json(marker_path.read_bytes())
    lie = marker.model_dump(mode="python")
    lie["manifest_logical_hash"] = altered_manifest
    lie["vector_commitment"] = {
        **marker.vector_commitment.model_dump(mode="python"),  # type: ignore[union-attr]
        "payload_sha256": altered_sha,
        "payload_path": f".system/knowledge-vectors/{altered_sha}.f32",
    }
    from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes

    marker_path.write_bytes(canonical_json_bytes(HybridIndexEvictionMarker.model_validate(lie)))
    third = _request_at(_NOW + timedelta(days=2))
    third_set, third_generation = _prepare(reopened, third, (_recorded_document(),))
    assert passes_again == [], "restored from the committed payload, no model"
    assert third_generation.index_manifest_hash == generation.index_manifest_hash
    assert third_generation.vector_payload_sha256 == generation.vector_payload_sha256
    assert _manifest_of(database).logical_hash == generation.index_manifest_hash
    assert not marker_path.is_file()
    assert _hits(reopened, third, third_set, third_generation) == first_hits
    # With the committed payload gone as well, restoration runs the model
    # once and its output must reproduce the committed digest -- it does
    # here; the altered payload beside it is never read.
    reopened.retrieval.evict_generation(relative, plan_hash="b" * 64, evicted_at=_NOW)
    payload.unlink()
    fourth = _request_at(_NOW + timedelta(days=3))
    _fourth_set, fourth_generation = _prepare(reopened, fourth, (_recorded_document(),))
    assert _corpus_passes(passes_again, generation) == 1
    assert fourth_generation.vector_payload_sha256 == generation.vector_payload_sha256
    assert payload.read_bytes() == genuine_bytes
    reopened.close()

    # Two sealed records that disagree about one generation are two truths
    # under one identity: neither a new cutoff nor a rebuild picks one.
    forged = seal_contract(
        AlternativeEvidenceRetrievalGeneration,
        "generation_hash",
        **{
            **generation.model_dump(mode="python", exclude={"generation_hash"}),
            "index_manifest_hash": altered_manifest,
            "vector_payload_sha256": altered_sha,
            "built_at": _NOW + timedelta(days=9),
        },
    )
    divergent, _passes = _counted_runtime(tmp_path)
    divergent.artifacts.publish("retrieval-generations", forged.generation_hash, forged)
    with pytest.raises(ValueError, match="retrieval_generation_records_divergent"):
        _prepare(divergent, _request_at(_NOW + timedelta(days=4)), (_recorded_document(),))
    with pytest.raises(ValueError, match="retrieval_generation_records_divergent"):
        divergent.rebuild_retrieval(generation.index_id)
    divergent.close()


def test_an_unsealed_index_with_other_vectors_is_refused_not_replaced(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """requirement: an index no record names is not trusted, and when this
    build's own pass does not reproduce it, it is refused by name and left
    for cleanup rather than silently overwritten or silently adopted."""

    runtime, passes = _counted_runtime(tmp_path)
    request = _request_at(_NOW)
    publish = runtime.artifacts.publish

    def dies_before_sealing(category: str, identity: str, model: Any) -> None:
        if category == "retrieval-generations":
            raise RuntimeError("qa.interrupted_before_record_sealed")
        publish(category, identity, model)

    monkeypatch.setattr(runtime.artifacts, "publish", dies_before_sealing)
    with pytest.raises(RuntimeError, match="interrupted_before_record_sealed"):
        _prepare(runtime, request, (_recorded_document(),))
    monkeypatch.undo()
    databases, payloads = _physical(tmp_path)
    database, payload = databases[0], payloads[0]
    _rewrite_coherently(database, payload)

    with pytest.raises(KnowledgeRetrievalError) as refused:
        _prepare(runtime, request, (_recorded_document(),))
    assert refused.value.failure.code == "retrieval.index_publish_conflict"
    assert len(passes) == 2, "the pass that would have proved the orphan ran, and disproved it"
    assert database.is_file(), "nothing was overwritten"
    assert not list((runtime.artifacts.root / "retrieval-generations").glob("*.json"))
    runtime.close()


def test_a_legacy_generation_is_accounted_but_never_opened(tmp_path: Path) -> None:
    """requirement (D2, 2026-09-23): a `knowledge-hybrid-v3` generation is history.

    Its record still parses through the format dispatch -- the storage owner
    accounts its index bytes -- but a session never opens it: the request is
    refused by name before any index is read or any model runs, and the committed
    generation of the same corpus opens as before, embedding nothing.
    """

    runtime, _passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)
    recorded = generation.model_dump(
        exclude={
            "generation_format",
            "corpus_hash",
            "index_manifest_hash",
            "vector_payload_sha256",
            "generation_hash",
        }
    )
    legacy = seal_contract(HybridV3RetrievalGeneration, "generation_hash", **recorded)
    assert isinstance(
        parse_retrieval_generation(legacy.model_dump(mode="json")), HybridV3RetrievalGeneration
    )
    assert isinstance(
        parse_retrieval_generation(generation.model_dump(mode="json")),
        AlternativeEvidenceRetrievalGeneration,
    )
    runtime.close()

    restarted, later_passes = _counted_runtime(tmp_path)
    try:
        with pytest.raises(ValueError, match="retrieval_generation_retired"):
            _hits(restarted, request, document_set, legacy)
        assert _hits(restarted, request, document_set, generation)
        assert _corpus_passes(later_passes, generation) == 0, "nothing was embedded for either"
    finally:
        restarted.close()


def test_an_interrupted_build_reruns_its_uncommitted_work_and_publishes_once(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """requirement: death before or after the index is placed leaves one
    authoritative generation; uncommitted work reruns and is counted; work a
    sealed record commits to is reused; an orphan is never its own proof.

    Death while the index is staged: nothing was placed -- the payload is
    placed only after the staged index is verified and admitted -- and the
    next build embeds again (counted as a pass, never hidden) and publishes.
    Death after the index was placed but before its record was sealed: no
    durable commitment names the index, so the next build does not trust it;
    it embeds again and keeps the orphan only because this pass reproduced it
    exactly. Death after the record was sealed: the next build finds the
    commitment and reuses the generation with no pass.
    """

    runtime, passes = _counted_runtime(tmp_path)
    request = _request_at(_NOW)

    def dies_while_staging(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("qa.interrupted_before_index_publication")

    monkeypatch.setattr(
        WorkspaceHybridKnowledgeIndex, "_build_database", staticmethod(dies_while_staging)
    )
    with pytest.raises(RuntimeError, match="interrupted_before_index_publication"):
        _prepare(runtime, request, (_recorded_document(),))
    assert runtime.retrieval.passage_embedding_pass_count == 0, "nothing published, none counted"
    assert len(passes) == 1, "the model did run once for the attempt that died"
    databases, payloads = _physical(tmp_path)
    assert databases == [] and payloads == [], "nothing placed before the staged index verified"
    assert not list((tmp_path / "workspace" / ".system" / "staging").glob("*")), (
        "staging is cleaned up"
    )
    monkeypatch.undo()

    # Death between placing the index and sealing its record.
    publish = runtime.artifacts.publish

    def dies_before_sealing(category: str, identity: str, model: Any) -> None:
        if category == "retrieval-generations":
            raise RuntimeError("qa.interrupted_before_record_sealed")
        publish(category, identity, model)

    monkeypatch.setattr(runtime.artifacts, "publish", dies_before_sealing)
    with pytest.raises(RuntimeError, match="interrupted_before_record_sealed"):
        _prepare(runtime, request, (_recorded_document(),))
    monkeypatch.undo()
    databases, payloads = _physical(tmp_path)
    assert len(databases) == 1 and len(payloads) == 1, "placed, unsealed: an orphan"
    assert not list((runtime.artifacts.root / "retrieval-generations").glob("*.json"))
    assert len(passes) == 2

    document_set, generation = _prepare(runtime, request, (_recorded_document(),))
    assert len(passes) == 3, "the orphan was not trusted: this pass proved it"
    # The service counted the build that placed the orphan and the build
    # that proved it: both embedded, whatever happened to their records.
    assert runtime.retrieval.passage_embedding_pass_count == 2
    assert runtime.retrieval.generation_reuse_count == 0
    databases, payloads = _physical(tmp_path)
    assert len(databases) == 1 and len(payloads) == 1, "the same generation, once"
    assert _hits(runtime, request, document_set, generation)

    # Death after the record was sealed (the service raised before returning
    # the generation to its Task): the next build finds the commitment.
    def dies_after_sealing(self: Any, **kwargs: Any) -> Any:
        raise RuntimeError("qa.interrupted_after_index_publication")

    later = _request_at(_NOW + timedelta(days=1))
    monkeypatch.setattr(runtime.retrieval, "build", dies_after_sealing.__get__(runtime.retrieval))
    with pytest.raises(RuntimeError, match="interrupted_after_index_publication"):
        _prepare(runtime, later, (_recorded_document(),))
    monkeypatch.undo()
    _later_set, later_generation = _prepare(runtime, later, (_recorded_document(),))
    assert runtime.retrieval.passage_embedding_pass_count == 2
    assert runtime.retrieval.generation_reuse_count == 1
    assert later_generation.index_id == generation.index_id
    assert later_generation.index_manifest_hash == generation.index_manifest_hash
    # The model ran three times in all: the attempt that died staging, the
    # attempt that died unsealed, and the pass that proved the orphan. All
    # are real passes the adapter saw; the service counted the two that
    # placed an index. The committed generation was then reused with none.
    assert _corpus_passes(passes, generation) == 3


def test_another_spec_is_another_generation_and_never_opens_this_one(tmp_path: Path) -> None:
    """requirement: a changed spec cannot open a generation built under another;
    a spec sealed before F1, naming its threads, is another spec."""

    runtime, _passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)
    other_spec = HybridIndexSpec.sealed_before_f1(RECIPE_MINILM_CPU, 4)
    assert other_spec.logical_hash != runtime.retrieval.index_spec.logical_hash
    runtime.close()

    restarted, later_passes = _counted_runtime(tmp_path)
    restarted.retrieval.index_spec = other_spec
    try:
        with pytest.raises(ValueError, match="retrieval_generation_lineage_invalid"):
            _hits(restarted, request, document_set, generation)
        assert later_passes == []
    finally:
        restarted.close()


def test_a_generation_that_named_its_threads_serves_its_vectors_once_they_left(
    tmp_path: Path,
) -> None:
    """requirement (the final close-out, F1): a book whose generation was sealed
    naming its sessions' threads is built again under the spec without them at its
    next refresh, and reuses every vector: the model embeds no passage."""

    runtime, passes = _counted_runtime(tmp_path)
    runtime.retrieval.index_spec = HybridIndexSpec.sealed_before_f1(RECIPE_MINILM_CPU, 4)
    request, document_set, legacy = _built(runtime)
    assert _corpus_passes(passes, legacy) == 1
    before = _hits(runtime, request, document_set, legacy)
    runtime.close()

    restarted, later_passes = _counted_runtime(tmp_path)
    try:
        assert not restarted.retrieval.index_spec.names_execution
        rebuilt = restarted.build_retrieval(document_set=document_set, built_at=_NOW)
        assert rebuilt.index_spec_hash != legacy.index_spec_hash
        assert later_passes == [], "the vectors sealed before F1 were embedded again"
        assert restarted.block_source_refusals == []
        assert _hits(restarted, request, document_set, rebuilt) == before
    finally:
        restarted.close()


def _sidecar_body(sidecar: Path) -> dict[str, Any]:
    from alphalattice.kernel.knowledge.hybrid_contracts import HybridVectorBlockSidecar

    return HybridVectorBlockSidecar.model_validate_json(sidecar.read_bytes()).model_dump(
        mode="python"
    )


def _write_sidecar(sidecar: Path, body: dict[str, Any]) -> None:
    from alphalattice.kernel.knowledge.hybrid_contracts import HybridVectorBlockSidecar
    from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes

    sidecar.write_bytes(canonical_json_bytes(HybridVectorBlockSidecar.model_validate(body)))


def _revision_inputs(runtime: Any, generation: Any, revision_label: str) -> tuple[str, ...]:
    """The ordered encoder inputs of one revision of a sealed generation's corpus."""

    from alphalattice.evidence.alternative_evidence.documents.contracts import (
        AlternativeEvidenceDocumentSet,
    )

    document_set = runtime.artifacts.load(
        "document-sets", generation.document_set_hash, AlternativeEvidenceDocumentSet
    )
    snapshot = runtime.documents.library.read_snapshot(UUID(document_set.workspace_snapshot_id))
    spec = runtime.retrieval.index_spec
    chunks = _build_chunks(snapshot, runtime.documents.library, spec.lexical_spec)
    wanted = {
        item.workspace_document_id
        for item in document_set.documents
        if item.revision_label == revision_label
    }
    assert len(wanted) == 1, wanted
    return tuple(
        str(row["embedding_input_hash"])
        for chunk, row in zip(chunks, _hybrid_rows(chunks, spec), strict=True)
        if chunk.document_id in wanted
    )


def test_a_first_reading_proves_no_earlier_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (X6): the vector block source looks only through the
    generations that read one of the set's filings -- proving a generation
    reads every block it holds. A set of filings no generation read proves
    none (it proved every earlier generation whole); a set sharing a filing
    proves the generation that read it and reuses its vectors."""

    from alphalattice.evidence.alternative_evidence.runtime import service
    from tests.alternative_evidence_desk.planted_corpus import PLANTED_CLAIMS, EvidenceTopic

    proved: list[str] = []
    prove = service.verified_block_map

    def counted(root: Any, **values: Any) -> Any:
        proved.append(str(values["manifest_logical_hash"]))
        return prove(root, **values)

    monkeypatch.setattr(service, "verified_block_map", counted)
    runtime, passes = _counted_runtime(tmp_path)
    request = _request_at(_NOW)
    earlier = _recorded_document(revision="issuer-release-2026-q3")
    _set, first = _prepare(runtime, request, (earlier,))
    assert len(passes) == 1 and proved == []
    # A new filing, never read: nothing is proved.
    later = _request_at(_NOW + timedelta(days=1))
    new = _recorded_document(
        revision="issuer-release-2026-q4",
        claims=(PLANTED_CLAIMS[EvidenceTopic.LIQUIDITY_GOING_CONCERN],),
    )
    _new_set, _second = _prepare(runtime, later, (new,))
    assert proved == [], "a first reading proves no earlier generation"
    # The earlier filing again, beside the new one: its generation is proved.
    both = _request_at(_NOW + timedelta(days=2))
    _both_set, _third = _prepare(runtime, both, (earlier, new))
    assert first.index_manifest_hash in proved
    runtime.close()


def test_a_sidecar_key_cannot_map_a_block_to_other_inputs_or_another_context(
    tmp_path: Path,
) -> None:
    """requirement (R1): a reusable block proves its bytes AND its association
    with the exact ordered encoder inputs, context and position. A sidecar
    whose keys are swapped between two blocks of equal size -- every vector
    byte, block digest, order and payload digest preserved -- serves nothing,
    the build embeds, and the sealed generation still reads back whole; a
    key claiming another encoder context serves nothing under that context;
    the original metadata restores ordinary reuse."""

    from alphalattice.kernel.knowledge.hybrid_contracts import (
        embedding_context_hash,
        vector_asset_key,
    )
    from alphalattice.kernel.shared_kernel.domain.serialization import sha256_hex
    from tests.alternative_evidence_desk.planted_corpus import PLANTED_CLAIMS, EvidenceTopic

    runtime, passes = _counted_runtime(tmp_path)
    a = _release()
    b = _release(
        revision="issuer-release-2026-q2",
        claims=(PLANTED_CLAIMS[EvidenceTopic.LIQUIDITY_GOING_CONCERN],),
    )
    request = _request_at(_NOW)
    ab_set, first = _prepare(runtime, request, (a, b))
    assert len(passes) == 2
    first_hits = _hits(runtime, request, ab_set, first)
    root = _vector_root(tmp_path)
    (sidecar,) = _sidecars(tmp_path)
    original = sidecar.read_bytes()
    body = _sidecar_body(sidecar)
    block_a, block_b = body["blocks"]
    assert block_a["vector_count"] == block_b["vector_count"], "same size, different vectors"
    assert block_a["block_sha256"] != block_b["block_sha256"]
    bytes_a = (root / f"{block_a['block_sha256']}.f32").read_bytes()
    bytes_b = (root / f"{block_b['block_sha256']}.f32").read_bytes()
    context = embedding_context_hash(runtime.retrieval.index_spec)
    inputs_b = _revision_inputs(runtime, first, b.revision)
    assert block_b["asset_key"] == vector_asset_key(context, inputs_b)

    # Metadata only: the two keys exchanged; bytes, digests, order and the
    # whole-payload digest untouched.
    swapped = {
        **body,
        "blocks": (
            {**block_a, "asset_key": block_b["asset_key"]},
            {**block_b, "asset_key": block_a["asset_key"]},
        ),
    }
    _write_sidecar(sidecar, swapped)
    assert sha256_hex(bytes_a + bytes_b) == first.vector_payload_sha256

    # A new effective corpus asks for B's inputs: the swapped key would hand
    # it A's vectors. The real composition path must embed B instead.
    later = _request_at(_NOW + timedelta(days=1))
    _b_set, second = _prepare(runtime, later, (b,))
    assert passes[2:] == [second.chunk_count], "B was embedded, not reused"
    assert (root / f"{block_b['block_sha256']}.f32").read_bytes() == bytes_b
    (b_sidecar,) = [path for path in _sidecars(tmp_path) if path != sidecar]
    (b_block,) = _sidecar_body(b_sidecar)["blocks"]
    assert b_block["block_sha256"] == block_b["block_sha256"], "the sealed vectors are B's"
    assert second.vector_payload_sha256 == sha256_hex(bytes_b)
    refused = runtime.block_source_refusals
    assert [entry["index_id"] for entry in refused] == [first.index_id]
    assert refused[0]["code"] == "retrieval.vector_commitment_mismatch"
    # The tampered composition still reads back whole: the bytes it names
    # are the committed payload, and keys play no part in that proof.
    assert _hits(runtime, request, ab_set, first) == first_hits
    assert len(passes) == 3

    # A key claiming another encoder context: nothing is served under that
    # context; the conflicting sidecar serves nothing, and one bad candidate
    # does not refuse the independent valid one (B's own sealed generation).
    other = embedding_context_hash(
        runtime.retrieval.index_spec.model_copy(
            update={"batch_size": runtime.retrieval.index_spec.batch_size + 1}
        )
    )
    _write_sidecar(
        sidecar,
        {**body, "blocks": (block_a, {**block_b, "asset_key": vector_asset_key(other, inputs_b)})},
    )
    source = runtime.vector_block_source(ab_set)
    assert source(other, inputs_b) is None
    assert source(context, inputs_b) == bytes_b, "served by B's own untampered generation"
    assert [(entry["index_id"], entry["code"]) for entry in runtime.block_source_refusals] == [
        (first.index_id, "retrieval.vector_commitment_mismatch")
    ]
    # Restored metadata: the same verified inputs and context reuse again,
    # from the first generation too, and nothing is refused.
    sidecar.write_bytes(original)
    source = runtime.vector_block_source(ab_set)
    assert source(context, inputs_b) == bytes_b
    assert runtime.block_source_refusals == []
    assert source(other, inputs_b) is None
    runtime.close()


def test_removed_trigram_postings_are_refused_before_results_are_trusted(
    tmp_path: Path,
) -> None:
    """requirement (R3): the trigram index reads its columns from the content
    table, so a row whose postings were removed with a valid FTS5 operation
    still joins, counts and displays. The reader must prove postings against
    content before any result is trusted; restoring the bytes reads exactly."""

    runtime, _passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)
    before = _hits(runtime, request, document_set, generation)
    assert before, "the single-token probe reaches the trigram channel"
    runtime.close()
    (database,), _payloads = _physical(tmp_path)
    original = database.read_bytes()
    with closing(sqlite3.connect(database)) as connection:
        rowid, document_id, title, heading, body = connection.execute(
            "SELECT rowid, document_id, title_raw, heading_raw, body_raw FROM content "
            "WHERE body_raw LIKE '%operating%' ORDER BY rowid LIMIT 1"
        ).fetchone()
        connection.execute(
            "INSERT INTO trigram_fts(trigram_fts, rowid, identifier_raw, title_raw, "
            "heading_raw, body_raw) VALUES ('delete', ?, ?, ?, ?, ?)",
            (rowid, document_id, title, heading, body),
        )
        connection.commit()
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as ro:
        # The finding: ordinary checks accept the desynchronized index.
        assert ro.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert (
            ro.execute("SELECT COUNT(*) FROM trigram_fts").fetchone()[0] == generation.chunk_count
        )
        assert (
            ro.execute("SELECT body_raw FROM trigram_fts WHERE rowid = ?", (rowid,)).fetchone()[0]
            == body
        )
        assert rowid not in {
            row[0]
            for row in ro.execute(
                "SELECT rowid FROM trigram_fts WHERE trigram_fts MATCH 'operating'"
            )
        }, "the query has lost the row"

    restarted, later_passes = _counted_runtime(tmp_path)
    with pytest.raises(KnowledgeRetrievalError) as refused:
        _hits(restarted, request, document_set, generation)
    assert refused.value.failure.code == "retrieval.index_corrupt"
    assert later_passes == [], "a refusal costs no passage embedding"
    restarted.close()

    database.write_bytes(original)
    restored, later_passes = _counted_runtime(tmp_path)
    assert _hits(restored, request, document_set, generation) == before
    assert later_passes == []
    restored.close()


def test_an_evicted_generation_serves_no_blocks_until_its_index_is_rebuilt(
    tmp_path: Path,
) -> None:
    """requirement (R1, bounded fallback): a block's key is derived from the
    committed manifest, so a generation whose index is evicted proves no
    mapping -- refused by name, its revisions embedded again -- while its own
    cold readback and explicit rebuild still restore it from the retained
    blocks with no model; once the index is back, its blocks serve again."""

    from datetime import UTC, datetime

    runtime, passes = _counted_runtime(tmp_path)
    a = _recorded_document()
    b = _recorded_document(revision="issuer-release-2026-q2", claims=("Guidance was raised.",))
    request = _request_at(_NOW)
    ab_set, first = _prepare(runtime, request, (a, b))
    assert len(passes) == 2
    first_hits = _hits(runtime, request, ab_set, first)
    (database,), _payloads = _physical(tmp_path)
    relative = database.relative_to(tmp_path / "workspace").as_posix()
    runtime.retrieval.evict_generation(
        relative, plan_hash="a" * 64, evicted_at=datetime(2026, 9, 16, tzinfo=UTC)
    )
    assert not database.is_file()

    # A corpus sharing B: the evicted generation is refused by name and B is
    # embedded; nothing is guessed from the blocks alone.
    c = _recorded_document(revision="issuer-release-2026-q1", claims=("Inventory fell.",))
    later = _request_at(_NOW + timedelta(days=1))
    _bc_set, second = _prepare(runtime, later, (b, c))
    assert (
        passes[2:] == [second.chunk_count // 2, second.chunk_count // 2]
        or sum(passes[2:]) == second.chunk_count
    ), "both revisions embedded"
    assert [(entry["index_id"], entry["code"]) for entry in runtime.block_source_refusals] == [
        (first.index_id, "retrieval.vector_block_mapping_unproved")
    ]

    # The explicit rebuild restores the evicted generation from its retained
    # blocks, with no model run; its blocks then serve a further corpus.
    facts = runtime.rebuild_retrieval(first.index_id)
    assert facts["passage_embedding_pass"] is False and database.is_file()
    assert _hits(runtime, request, ab_set, first) == first_hits
    embedded_before = sum(passes)
    d = _recorded_document(revision="issuer-release-2025-q4", claims=("A plant was sold.",))
    _ad_set, third = _prepare(runtime, _request_at(_NOW + timedelta(days=2)), (a, d))
    assert sum(passes) - embedded_before == passes[-1] < third.chunk_count, "A came from a block"
    assert runtime.block_source_refusals == []
    runtime.close()
