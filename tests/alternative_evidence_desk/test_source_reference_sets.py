"""Acquired source bytes are kept once and referenced by every request that
acquired them; the retained inline form still reads.

Two preparations over the same filings hold one source object per distinct
content and two small reference sets; the canonicalizer resolves references
through the Workspace owner and refuses, by name, an object that is missing
or holds other bytes; a set sealed in the retained inline form canonicalizes
as it did.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.sources.contracts import (
    SOURCE_OBJECT_ROOT,
    AcquiredEvidenceDocumentSet,
    AcquiredEvidenceSourceReferenceSet,
    parse_source_set,
)
from tests.alternative_evidence_desk.document_intelligence_support import (
    _counted_runtime,
    _registry,
)
from tests.alternative_evidence_desk.planted_corpus import _NOW, _recorded_document


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


def _objects(tmp_path: Path) -> list[Path]:
    root = tmp_path / "workspace" / Path(*SOURCE_OBJECT_ROOT.split("/"))
    return sorted(root.iterdir()) if root.is_dir() else []


def test_source_bytes_are_kept_once_and_referenced_by_every_request(tmp_path: Path) -> None:
    runtime, _passes = _counted_runtime(tmp_path)
    documents = (
        _recorded_document(),
        _recorded_document(revision="issuer-release-2026-q2", claims=("Guidance was raised.",)),
    )
    first_request = _request_at(_NOW)
    _snapshot, first = runtime.acquire_recorded(
        request=first_request, registry=_registry(), documents=documents, published_at=_NOW
    )
    assert isinstance(first, AcquiredEvidenceSourceReferenceSet)
    assert [value.semantic_handle for value in first.documents] == ["DOC-AAPL-001", "DOC-AAPL-002"]
    objects = _objects(tmp_path)
    assert len(objects) == 2, "one object per distinct content"
    assert {path.name for path in objects} == {v.content_sha256 for v in first.documents}
    for reference in first.documents:
        content = (
            tmp_path / "workspace" / Path(*reference.content_object_path.split("/"))
        ).read_bytes()
        assert len(content) == reference.content_bytes
    stored = runtime.artifacts.root / "source-document-sets" / f"{first.source_set_hash}.json"
    assert stored.stat().st_size < 4096, "a reference set carries provenance, not bytes"
    assert b'"content":' not in stored.read_bytes()

    # A second request over the same filings: its own set, the same objects.
    later = _request_at(_NOW + timedelta(days=1))
    _snapshot, second = runtime.acquire_recorded(
        request=later, registry=_registry(), documents=documents, published_at=later.evidence_as_of
    )
    assert second.source_set_hash != first.source_set_hash
    assert _objects(tmp_path) == objects, "nothing added for bytes already held"
    assert {v.content_sha256 for v in second.documents} == {
        v.content_sha256 for v in first.documents
    }

    # The canonicalizer resolves references through the owner; the loaded set
    # is the reference form, verified by identity.
    loaded = runtime.artifacts.load_source_set(second.source_set_hash)
    assert loaded == second
    document_set = runtime.canonicalize(source_set=loaded, published_at=later.evidence_as_of)
    assert len(document_set.documents) == 2
    resolved = runtime.resolve_source_documents(loaded)
    assert [value.source_content_hash for value in resolved] == [
        value.source_content_hash for value in second.documents
    ]

    # An object that holds other bytes, then one that is gone: refused by name
    # at the boundary where the bytes are needed; the set itself still reads.
    target = objects[0]
    genuine = target.read_bytes()
    target.write_bytes(genuine[:-1] + bytes([genuine[-1] ^ 0x01]))
    with pytest.raises(ValueError, match="source_object_tampered"):
        runtime.canonicalize(source_set=loaded, published_at=later.evidence_as_of)
    target.unlink()
    with pytest.raises(ValueError, match="source_object_missing"):
        runtime.canonicalize(source_set=loaded, published_at=later.evidence_as_of)
    assert runtime.artifacts.load_source_set(second.source_set_hash) == second
    target.write_bytes(genuine)
    assert (
        runtime.canonicalize(source_set=loaded, published_at=later.evidence_as_of) == document_set
    )
    # An address that holds other bytes refuses a new publication rather than
    # adopting them.
    target.write_bytes(genuine[:-1] + bytes([genuine[-1] ^ 0x01]))
    with pytest.raises(ValueError, match="source_object_integrity"):
        runtime.acquire_recorded(
            request=_request_at(_NOW + timedelta(days=2)),
            registry=_registry(),
            documents=documents,
            published_at=_NOW + timedelta(days=2),
        )
    target.write_bytes(genuine)
    runtime.close()


def test_the_retained_inline_source_set_still_canonicalizes(tmp_path: Path) -> None:
    runtime, _passes = _counted_runtime(tmp_path)
    request = _request_at(_NOW)
    _snapshot, acquired = runtime.acquisition.build_recorded_evidence(
        request=request, registry=_registry(), documents=(_recorded_document(),), published_at=_NOW
    )
    inline = seal_contract(
        AcquiredEvidenceDocumentSet,
        "source_set_hash",
        request_hash=request.request_hash,
        source_snapshot_hash=_snapshot.snapshot_hash,
        documents=acquired,
        acquired_at=_NOW,
    )
    runtime.artifacts.publish("source-document-sets", inline.source_set_hash, inline)
    loaded = runtime.artifacts.load_source_set(inline.source_set_hash)
    assert isinstance(loaded, AcquiredEvidenceDocumentSet)
    assert loaded == inline
    assert runtime.resolve_source_documents(loaded) == acquired
    document_set = runtime.canonicalize(source_set=loaded, published_at=_NOW)
    assert len(document_set.documents) == 1
    assert _objects(tmp_path) == [], "reading an inline set publishes no object"
    payload = json.loads(
        (
            runtime.artifacts.root / "source-document-sets" / f"{inline.source_set_hash}.json"
        ).read_bytes()
    )
    assert parse_source_set(payload) == inline
    with pytest.raises(ValueError, match="source_set_kind_unknown"):
        parse_source_set({**payload, "kind": "Other"})
    runtime.close()
