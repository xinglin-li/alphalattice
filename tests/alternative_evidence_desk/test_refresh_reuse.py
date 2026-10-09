"""An unchanged refresh reuses what sealed commitments already prove --
the citation excerpts, the canonical documents and the program's selection
-- and computes afresh exactly what changed. Offline, over the scenario
transport and the planted retrieval kernel; a restart proves the reuse
stands on the sealed artifacts alone.
"""

from __future__ import annotations

import builtins
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceRetrievalAccessReceipt,
)
from alphalattice.evidence.alternative_evidence.analysis.packet import query_program_hash
from alphalattice.evidence.alternative_evidence.contracts import seal_contract
from alphalattice.evidence.alternative_evidence.documents import canonicalization
from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
    AlternativeEvidenceResolvedSpanSet,
)
from alphalattice.evidence.alternative_evidence.runtime import reuse
from alphalattice.evidence.alternative_evidence.runtime.reuse import SealedSelections
from alphalattice.evidence.alternative_evidence.runtime.service import (
    AlternativeEvidenceDocumentIntelligenceRuntime,
)
from alphalattice.evidence.alternative_evidence.sources import sec_edgar
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from tests.alternative_evidence_desk.incremental_acquisition_support import (
    AAPL,
    LATER_EIGHT_K,
    _acquire,
    _request,
    _transport,
)
from tests.alternative_evidence_desk.planted_corpus import _runtime
from tests.alternative_evidence_desk.sec_scenario_transport import (
    NOW,
    SecScenarioTransport,
    filing_body,
)


class _Extractions:
    """How many bodies the canonical extractor parsed, per caller."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.excerpts = 0
        self.canonicals = 0
        original = canonicalization.extract_canonical_markdown

        def for_excerpts(content: bytes, **kwargs: Any) -> bytes:
            self.excerpts += 1
            return original(content, **kwargs)

        def for_canonicals(content: bytes, **kwargs: Any) -> bytes:
            self.canonicals += 1
            return original(content, **kwargs)

        monkeypatch.setattr(sec_edgar, "extract_canonical_markdown", for_excerpts)
        monkeypatch.setattr(canonicalization, "extract_canonical_markdown", for_canonicals)

    def reset(self) -> None:
        self.excerpts = self.canonicals = 0


class _Sessions:
    """How many retrieval sessions the runtime opened."""

    def __init__(self, runtime: AlternativeEvidenceDocumentIntelligenceRuntime) -> None:
        self.opened = 0
        original = runtime.retrieval.open_session

        def counted(**kwargs: Any) -> Any:
            self.opened += 1
            return original(**kwargs)

        runtime.retrieval.open_session = counted  # type: ignore[method-assign]


def _prepare(
    runtime: AlternativeEvidenceDocumentIntelligenceRuntime,
    transport: SecScenarioTransport,
    *,
    as_of: Any,
    registry: Any = None,
    budget: int = 3,
) -> dict[str, Any]:
    request = _request(as_of=as_of, budget=budget)
    registry, snapshot, source_set = _acquire(runtime, transport, request, registry=registry)
    document_set = runtime.canonicalize(source_set=source_set, published_at=source_set.acquired_at)
    generation = runtime.build_retrieval(
        document_set=document_set, built_at=document_set.published_at
    )
    receipt, spans = runtime.select_evidence(
        request=request, document_set=document_set, generation=generation
    )
    return {
        "request": request,
        "registry": registry,
        "snapshot": snapshot,
        "source_set": source_set,
        "document_set": document_set,
        "generation": generation,
        "receipt": receipt,
        "spans": spans,
    }


def _assert_spans_anchor(
    runtime: AlternativeEvidenceDocumentIntelligenceRuntime, prepared: dict[str, Any]
) -> None:
    """Every delivered span reads back from its document's verified revision."""

    receipt, spans, document_set = prepared["receipt"], prepared["spans"], prepared["document_set"]
    assert set(receipt.delivered_span_handles) == {span.span_handle for span in spans}
    references = {value.semantic_handle: value for value in document_set.documents}
    for span in spans:
        reference = references[span.document_handle]
        _revision, content = runtime.documents.library.read_revision(
            reference.workspace_document_id, reference.workspace_revision
        )
        assert content[span.utf8_byte_start : span.utf8_byte_end].decode("utf-8") == span.excerpt


def test_an_unchanged_refresh_reuses_sealed_derivations_and_survives_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchanged refresh reuses sealed derivations and survives a restart."""

    extractions = _Extractions(monkeypatch)
    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        sessions = _Sessions(runtime)
        first = _prepare(runtime, transport, as_of=NOW)
        documents = len(first["document_set"].documents)
        assert documents == 3
        assert (extractions.excerpts, extractions.canonicals) == (documents, documents)
        assert sessions.opened == 1 and runtime.selection_reuse_count == 0
        assert first["receipt"].reused_from_receipt_hash is None
        assert first["receipt"].selection_policy_hash == runtime.analysis_policy_hash
        assert all(
            c.excerpt_binding_hash == runtime.document_binding_hash
            for c in first["snapshot"].citations
        )
        _assert_spans_anchor(runtime, first)

        extractions.reset()
        transport.reset_calls()
        second = _prepare(
            runtime, transport, as_of=NOW + timedelta(hours=1), registry=first["registry"]
        )
        assert transport.body_calls == [], "every body reused from the verified lookup"
        assert (extractions.excerpts, extractions.canonicals) == (0, 0)
        assert runtime.sealed_excerpts.reused == documents
        assert runtime.sealed_canonicals.reused == documents
        assert sessions.opened == 1, "no session for a selection a receipt already proves"
        assert runtime.selection_reuse_count == 1
        assert runtime.retrieval.generation_reuse_count == 1
        assert runtime.retrieval.generation_release_count == 1, "the build's lease is closed"
        assert runtime.retrieval.active_index_ids() == frozenset(), "nothing left held"
        # The derivations are the same commitments; the identities are this
        # request's own.
        assert second["request"].request_hash != first["request"].request_hash
        assert [c.excerpt for c in second["snapshot"].citations] == [
            c.excerpt for c in first["snapshot"].citations
        ]
        assert second["document_set"].documents == first["document_set"].documents
        assert second["document_set"].document_set_hash != first["document_set"].document_set_hash
        assert second["generation"].corpus_hash == first["generation"].corpus_hash
        assert second["receipt"].receipt_hash != first["receipt"].receipt_hash
        assert second["receipt"].reused_from_receipt_hash == first["receipt"].receipt_hash
        assert second["receipt"].request_hash == second["request"].request_hash
        assert second["receipt"].document_set_hash == second["document_set"].document_set_hash
        assert second["receipt"].queries == first["receipt"].queries
        assert second["receipt"].delivered_span_handles == first["receipt"].delivered_span_handles
        assert second["spans"] == first["spans"]
        _assert_spans_anchor(runtime, second)
    finally:
        runtime.close()

    # A restart: the reuse stands on the sealed artifacts, and names the
    # first receipt -- the selection's origin -- not the copy.
    extractions.reset()
    reopened = _runtime(tmp_path)
    try:
        sessions = _Sessions(reopened)
        transport.reset_calls()
        third = _prepare(
            reopened, transport, as_of=NOW + timedelta(hours=2), registry=first["registry"]
        )
        assert transport.body_calls == [] and (extractions.excerpts, extractions.canonicals) == (
            0,
            0,
        )
        assert sessions.opened == 0 and reopened.selection_reuse_count == 1
        assert third["receipt"].reused_from_receipt_hash == first["receipt"].receipt_hash
        assert third["spans"] == first["spans"]
        assert reopened.sealed_excerpts.scanned == 2 and reopened.sealed_canonicals.scanned == 2
        _assert_spans_anchor(reopened, third)
    finally:
        reopened.close()


def test_a_changed_source_reuses_the_unchanged_derivations_and_selects_afresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed source reuses the unchanged derivations and selects afresh."""

    extractions = _Extractions(monkeypatch)
    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        sessions = _Sessions(runtime)
        first = _prepare(runtime, transport, as_of=NOW)
        transport.add_filing(AAPL, LATER_EIGHT_K, filing_body(LATER_EIGHT_K.accession))
        extractions.reset()
        transport.reset_calls()
        second = _prepare(
            runtime,
            transport,
            as_of=NOW + timedelta(days=2),
            registry=first["registry"],
            budget=4,
        )
        assert transport.body_calls == [SecScenarioTransport.locator(AAPL, LATER_EIGHT_K)]
        assert (extractions.excerpts, extractions.canonicals) == (1, 1)
        assert runtime.sealed_excerpts.reused == 3 and runtime.sealed_canonicals.reused == 3
        assert len(second["document_set"].documents) == 4
        assert sessions.opened == 2 and runtime.selection_reuse_count == 0
        assert second["receipt"].reused_from_receipt_hash is None
        _assert_spans_anchor(runtime, second)

        other_policy = SealedSelections(
            runtime.artifacts,
            selection_policy_hash="f" * 64,
            program_hash=first["receipt"].query_program_hash,
        )
        production = first["request"].matter_selection_id
        assert (
            other_policy.find(
                document_set=first["document_set"],
                generation=first["generation"],
                matter_selection_id=production,
            )
            is None
        )
        assert (
            runtime.sealed_selections.find(
                document_set=first["document_set"],
                generation=first["generation"],
                matter_selection_id=production,
            )
            is not None
        )
        # The same content under another matter selection is another selection:
        # a production receipt never stands in for a candidate request.
        assert (
            runtime.sealed_selections.find(
                document_set=first["document_set"],
                generation=first["generation"],
                matter_selection_id="CANDIDATE_UNMET_NEEDS:LITIGATION,CORPORATE_EVENT",
            )
            is None
        )
    finally:
        runtime.close()


def test_a_damaged_canonical_revision_is_extracted_afresh_from_verified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A damaged canonical revision is extracted afresh from verified bytes."""

    extractions = _Extractions(monkeypatch)
    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        first = _prepare(runtime, transport, as_of=NOW)
        reference = first["document_set"].documents[0]
        revision, _content = runtime.documents.library.read_revision(
            reference.workspace_document_id, reference.workspace_revision
        )
        blob = runtime.documents.workspace.root / Path(*revision.blob_path.split("/"))
        original = blob.read_bytes()
        blob.write_bytes(original[:-1] + (b"!" if original[-1:] != b"!" else b"?"))
        extractions.reset()
        try:
            with pytest.raises(KnowledgeRetrievalError) as refused:
                _prepare(
                    runtime, transport, as_of=NOW + timedelta(hours=1), registry=first["registry"]
                )
            assert refused.value.failure.code == "retrieval.source_integrity"
            assert runtime.sealed_canonicals.damaged == 1
            assert runtime.sealed_canonicals.reused == 2 and extractions.canonicals == 1
        finally:
            blob.write_bytes(original)
        extractions.reset()
        third = _prepare(
            runtime, transport, as_of=NOW + timedelta(hours=2), registry=first["registry"]
        )
        assert extractions.canonicals == 0 and runtime.sealed_canonicals.reused == 5
        assert third["document_set"].documents == first["document_set"].documents
        assert third["receipt"].reused_from_receipt_hash == first["receipt"].receipt_hash
    finally:
        runtime.close()


def _shifted_span_set(
    runtime: AlternativeEvidenceDocumentIntelligenceRuntime, prepared: dict[str, Any]
) -> AlternativeEvidenceResolvedSpanSet:
    """A span set that carries exactly the receipt's handles under its
    request and generation, every range moved one character along the
    same verified source text: source-valid, self-consistent, and not the
    selection the receipt delivered."""

    references = {value.semantic_handle: value for value in prepared["document_set"].documents}
    shifted = []
    for span in prepared["spans"]:
        reference = references[span.document_handle]
        _revision, content = runtime.documents.library.read_revision(
            reference.workspace_document_id, reference.workspace_revision
        )
        text = content.decode("utf-8")
        start, end = span.character_start + 1, span.character_end + 1
        excerpt = text[start:end]
        shifted.append(
            span.model_copy(
                update={
                    "character_start": start,
                    "character_end": end,
                    "utf8_byte_start": len(text[:start].encode("utf-8")),
                    "utf8_byte_end": len(text[:end].encode("utf-8")),
                    "excerpt": excerpt,
                }
            )
        )
    return seal_contract(
        AlternativeEvidenceResolvedSpanSet,
        "span_set_hash",
        request_hash=prepared["request"].request_hash,
        retrieval_generation_hash=prepared["generation"].generation_hash,
        spans=tuple(shifted),
    )


def test_a_reused_selection_is_the_span_set_the_receipt_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reused selection is the span set the receipt names."""

    _Extractions(monkeypatch)
    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        sessions = _Sessions(runtime)
        first = _prepare(runtime, transport, as_of=NOW)
        receipt = first["receipt"]
        assert receipt.span_set_hash is not None
        store = runtime.artifacts
        named = store.root / "resolved-span-sets" / f"{receipt.span_set_hash}.json"
        assert named.is_file()
        original_bytes = named.read_bytes()
        substitute = _shifted_span_set(runtime, first)
        assert substitute.span_set_hash != receipt.span_set_hash
        assert tuple(s.span_handle for s in substitute.spans) == receipt.delivered_span_handles
        assert substitute.spans[0].excerpt != first["spans"][0].excerpt

        # Beside the original, under its own hash: the reuse still delivers
        # the original's spans, because the receipt names the original.
        store.publish("resolved-span-sets", substitute.span_set_hash, substitute)
        second = _prepare(
            runtime, transport, as_of=NOW + timedelta(hours=1), registry=first["registry"]
        )
        assert sessions.opened == 1 and runtime.selection_reuse_count == 1
        assert second["spans"] == first["spans"]
        assert second["receipt"].span_set_hash is not None
        assert second["receipt"].span_set_hash != receipt.span_set_hash, "its own set, its request"

        # Over the original's name: self-consistent, source-valid, and not the
        # commitment -- refused by name, not reused, not recomputed.
        named.write_bytes(store.serialized(substitute))
        with pytest.raises(ValueError, match="selection_commitment_corrupt"):
            _prepare(runtime, transport, as_of=NOW + timedelta(hours=2), registry=first["registry"])
        assert sessions.opened == 1, "no session ran over a corrupt commitment"
        named.write_bytes(original_bytes)
        third = _prepare(
            runtime, transport, as_of=NOW + timedelta(hours=3), registry=first["registry"]
        )
        assert third["spans"] == first["spans"] and sessions.opened == 1
    finally:
        runtime.close()

    # A receipt without the binding -- sealed before it was recorded -- is
    # no proof: the selection is computed again by a session.
    elsewhere = tmp_path.with_name(tmp_path.name + "-unbound")
    shutil.copytree(tmp_path, elsewhere)
    receipts = elsewhere / "artifacts" / "alternative-evidence" / "retrieval-access-receipts"
    for path in receipts.glob("*.json"):
        path.unlink()
    unbound = seal_contract(
        AlternativeEvidenceRetrievalAccessReceipt,
        "receipt_hash",
        request_hash=receipt.request_hash,
        document_set_hash=receipt.document_set_hash,
        retrieval_generation_hash=receipt.retrieval_generation_hash,
        query_program_hash=receipt.query_program_hash,
        queries=receipt.queries,
        read_span_handles=receipt.read_span_handles,
        search_call_count=receipt.search_call_count,
        span_read_call_count=receipt.span_read_call_count,
        span_groups=receipt.span_groups,
        typed_disclosures=receipt.typed_disclosures,
        litigation_matters=receipt.litigation_matters,
        selection_policy_hash=receipt.selection_policy_hash,
    )
    assert unbound.span_set_hash is None
    reopened = _runtime(elsewhere)
    try:
        reopened.artifacts.publish("retrieval-access-receipts", unbound.receipt_hash, unbound)
        sessions = _Sessions(reopened)
        again = _prepare(
            reopened, transport, as_of=NOW + timedelta(hours=4), registry=first["registry"]
        )
        assert sessions.opened == 1 and reopened.selection_reuse_count == 0
        assert again["receipt"].reused_from_receipt_hash is None
        assert again["spans"] == first["spans"], "computed again, the same selection"
    finally:
        reopened.close()


def test_an_unchanged_refresh_reuses_a_sealed_selection_without_building_its_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchanged refresh reuses a sealed selection without building its index."""

    _Extractions(monkeypatch)
    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        sessions = _Sessions(runtime)
        builds: list[str] = []
        original_build = runtime.retrieval.build

        def counted_build(**kwargs: Any) -> Any:
            builds.append(kwargs["document_set"].document_set_hash)
            return original_build(**kwargs)

        monkeypatch.setattr(runtime.retrieval, "build", counted_build)
        first = _prepare(runtime, transport, as_of=NOW)
        assert len(builds) == 1 and sessions.opened == 1

        def refresh(as_of: Any) -> dict[str, Any]:
            request = _request(as_of=as_of, budget=3)
            _registry, _snapshot, source_set = _acquire(
                runtime, transport, request, registry=first["registry"]
            )
            document_set = runtime.canonicalize(
                source_set=source_set, published_at=source_set.acquired_at
            )
            generation = runtime.build_retrieval(
                document_set=document_set, built_at=document_set.published_at, reuse_for=request
            )
            receipt, spans = runtime.select_evidence(
                request=request, document_set=document_set, generation=generation
            )
            return {
                "request": request,
                "document_set": document_set,
                "generation": generation,
                "receipt": receipt,
                "spans": spans,
            }

        # The unchanged refresh: no build, no session; the record is the one
        # the build would have sealed, and the selection is the reuse.
        second = refresh(NOW + timedelta(hours=1))
        assert len(builds) == 1 and sessions.opened == 1
        assert runtime.reuse_accounting()["generation_builds_avoided"] == 1
        assert runtime.reuse_accounting()["selection_reuses"] == 1
        assert second["receipt"].reused_from_receipt_hash == first["receipt"].receipt_hash
        assert [s.span_handle for s in second["spans"]] == [s.span_handle for s in first["spans"]]
        built = runtime.retrieval.build(
            document_set=second["document_set"],
            built_at=second["document_set"].published_at,
            anchor_for=runtime.generation_anchor,
        )
        assert built.generation_hash == second["generation"].generation_hash, (
            "the same identity the build seals"
        )
        runtime.retrieval.discard_pending(built.generation_hash)
        assert runtime.artifacts.exists("retrieval-generations", built.generation_hash)
        # A request under another policy hash builds and selects afresh.
        policy = runtime.analysis_policy_hash
        monkeypatch.setattr(runtime, "analysis_policy_hash", "0" * 64)
        runtime.sealed_selections = SealedSelections(
            runtime.artifacts, selection_policy_hash="0" * 64, program_hash=query_program_hash()
        )
        third = refresh(NOW + timedelta(hours=2))
        assert len(builds) == 3 and sessions.opened == 2  # the control build above, then this
        assert third["receipt"].reused_from_receipt_hash is None
        monkeypatch.setattr(runtime, "analysis_policy_hash", policy)
        runtime.sealed_selections = SealedSelections(
            runtime.artifacts, selection_policy_hash=policy, program_hash=query_program_hash()
        )
        # The index absent from disk: the build restores it, as before.
        database = next((tmp_path / "workspace" / ".system" / "knowledge-indexes").rglob("*.db"))
        database.unlink()
        fourth = refresh(NOW + timedelta(hours=3))
        assert len(builds) == 4 and sessions.opened == 2
        assert fourth["receipt"].reused_from_receipt_hash == first["receipt"].receipt_hash
        assert database.is_file(), "restored by the build"
        # A named span set that is not the one delivered: refused by name at
        # the build stage, never taken for a miss.
        named = (
            runtime.artifacts.root / "resolved-span-sets" / f"{first['receipt'].span_set_hash}.json"
        )
        original = named.read_bytes()
        substitute = _shifted_span_set(runtime, first)
        named.write_bytes(runtime.artifacts.serialized(substitute))
        try:
            with pytest.raises(ValueError, match="selection_commitment_corrupt"):
                refresh(NOW + timedelta(hours=4))
        finally:
            named.write_bytes(original)
        assert len(builds) == 4 and sessions.opened == 2
    finally:
        runtime.close()


@pytest.mark.parametrize("scan", ["ascending", "descending"])
def test_a_fresh_index_holds_the_origin_whatever_the_scan_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scan: str
) -> None:
    """A fresh index holds the origin whatever the scan order."""

    _Extractions(monkeypatch)
    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        first = _prepare(runtime, transport, as_of=NOW)
        second = _prepare(
            runtime, transport, as_of=NOW + timedelta(hours=1), registry=first["registry"]
        )
        assert second["receipt"].reused_from_receipt_hash == first["receipt"].receipt_hash
        assert second["receipt"].span_set_hash != first["receipt"].span_set_hash

        def order(paths: Any) -> list[Any]:
            return builtins.sorted(paths, reverse=scan == "descending")

        monkeypatch.setattr(reuse, "sorted", order, raising=False)
        fresh = SealedSelections(
            runtime.artifacts,
            selection_policy_hash=runtime.analysis_policy_hash,
            program_hash=query_program_hash(),
        )
        found = fresh.find(
            document_set=second["document_set"],
            generation=second["generation"],
            matter_selection_id=second["request"].matter_selection_id,
        )
        assert found is not None
        assert found[0].receipt_hash == first["receipt"].receipt_hash
        assert fresh.scanned == 2
    finally:
        runtime.close()
