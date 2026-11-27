"""The publication boundary of a session's selection: what is already on
disk is verified through the store's own boundary, never taken for the
record because its path exists (record section X, C0)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from alphalattice.evidence.alternative_evidence.analysis import packet
from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    RetrievalSupersessionRecord,
)
from alphalattice.evidence.alternative_evidence.contracts import seal_contract
from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    AlternativeEvidenceArtifactStore,
    AlternativeEvidencePublicationError,
    verified_evidence_records,
)
from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
    AlternativeEvidenceResolvedSpanSet,
)
from alphalattice.evidence.alternative_evidence.runtime.reuse import (
    RECEIPT_CATEGORY,
    SPAN_SET_CATEGORY,
)
from alphalattice.evidence.alternative_evidence.runtime.service import SUPERSESSION_CATEGORY
from alphalattice.foundation.data_layer.encoders import RerankScores
from alphalattice.kernel.identity import blake3_filename
from tests._support.identity_fixtures import fixture_digest
from tests.alternative_evidence_desk.document_intelligence_support import (
    _built,
    _counted_runtime,
)
from tests.alternative_evidence_desk.planted_corpus import _FakeReranker


class _Note(BaseModel):
    text: str
    note_hash: str


class _FrozenNote(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    note_hash: str


def _member(text: str) -> tuple[str, str, _Note]:
    identity = fixture_digest(text)
    return (SPAN_SET_CATEGORY, identity, _Note(text=text, note_hash=identity))


def _refuse(_additional: int) -> None:
    raise RuntimeError("storage.managed_capacity_exceeded")


def test_the_store_places_what_it_does_not_hold_and_verifies_what_it_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store places what it does not hold and verifies what it holds."""

    store = AlternativeEvidenceArtifactStore(tmp_path)
    first, second = _member("the set"), _member("its receipt")
    members = (first, second)
    sizes = [len(store.serialized(model)) for _category, _identity, model in members]
    paths = [
        store.root / category / f"{blake3_filename(identity)}.json"
        for category, identity, _m in members
    ]
    order: list[str] = []
    original_write = store._write

    def recording(target: Path, content: bytes) -> None:
        order.append(target.name)
        original_write(target, content)

    monkeypatch.setattr(store, "_write", recording)
    admitted: list[int] = []
    # Both missing: one admission of both, placed in the order given.
    assert store.place([first, second], admit=admitted.append) == 2
    assert admitted == [sum(sizes)] and order == [path.name for path in paths]
    assert all(path.is_file() for path in paths) and store.write_count == 2
    # Both held: a valid retry with no room for a byte writes nothing.
    assert store.place([first, second], admit=_refuse) == 0
    assert store.write_count == 2
    # One held, one missing: the missing bytes alone are admitted.
    paths[1].unlink()
    admitted.clear()
    assert store.place([first, second], admit=admitted.append) == 1
    assert admitted == [sizes[1]] and paths[1].is_file()
    # A refused admission places nothing.
    paths[1].unlink()
    with pytest.raises(RuntimeError, match="managed_capacity_exceeded"):
        store.place([first, second], admit=_refuse)
    assert paths[0].is_file() and not paths[1].is_file()
    # A held member with other bytes refuses by name before any admission
    # or write; the path exists and proves nothing; nothing is repaired.
    damaged = paths[0].read_bytes().replace(b"the set", b"another set")
    paths[0].write_bytes(damaged)
    admitted.clear()
    with pytest.raises(AlternativeEvidencePublicationError, match="artifact_identity_reused"):
        store.place([first, second], admit=admitted.append)
    assert admitted == [] and not paths[1].is_file() and paths[0].read_bytes() == damaged
    assert store.exists(first[0], first[1])
    with pytest.raises(AlternativeEvidencePublicationError, match="artifact_identity_reused"):
        store.holds(*first)
    with pytest.raises(AlternativeEvidencePublicationError, match="artifact_identity_reused"):
        store.publish(*first)
    assert paths[0].read_bytes() == damaged


def test_an_operation_verifies_each_record_once(tmp_path: Path) -> None:
    """An operation verifies each record once."""

    store = AlternativeEvidenceArtifactStore(tmp_path)
    note = _FrozenNote(text="the set", note_hash=fixture_digest("the set"))
    loose = _Note(text="its receipt", note_hash=fixture_digest("its receipt"))
    store.publish(SPAN_SET_CATEGORY, note.note_hash, note)
    store.publish(SPAN_SET_CATEGORY, loose.note_hash, loose)
    with verified_evidence_records():
        with verified_evidence_records():
            first = store.load(SPAN_SET_CATEGORY, note.note_hash, _FrozenNote)
        assert store.load(SPAN_SET_CATEGORY, note.note_hash, _FrozenNote) is first
        store.load(SPAN_SET_CATEGORY, loose.note_hash, _Note)
        store.load(SPAN_SET_CATEGORY, loose.note_hash, _Note)
    assert store.read_count == 3
    path = store.root / SPAN_SET_CATEGORY / f"{blake3_filename(note.note_hash)}.json"
    path.write_bytes(
        path.read_bytes().replace(note.note_hash.encode(), loose.note_hash.encode(), 1)
    )
    with pytest.raises(AlternativeEvidencePublicationError, match="artifact_tampered"):
        store.load(SPAN_SET_CATEGORY, note.note_hash, _FrozenNote)


def test_a_selection_is_published_only_over_verified_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selection is published only over verified artifacts."""

    runtime, _passes = _counted_runtime(tmp_path)
    try:
        request, document_set, generation = _built(runtime)
        admitted: list[int] = []
        runtime.storage_admission = admitted.append
        receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        assert receipt.span_set_hash is not None
        store = runtime.artifacts
        receipt_path = (
            store.root / RECEIPT_CATEGORY / f"{blake3_filename(receipt.receipt_hash)}.json"
        )
        set_path = store.root / SPAN_SET_CATEGORY / f"{blake3_filename(receipt.span_set_hash)}.json"
        assert receipt_path.is_file() and set_path.is_file()
        original_receipt, original_set = receipt_path.read_bytes(), set_path.read_bytes()
        assert len(original_receipt) + len(original_set) in admitted, "the pair, together"
        handles = [span.span_handle for span in spans]

        def replay() -> Any:
            again, again_spans = runtime.select_evidence(
                request=request, document_set=document_set, generation=generation
            )
            assert again.receipt_hash == receipt.receipt_hash
            assert [span.span_handle for span in again_spans] == handles
            return again

        # The receipt's own selection replayed while the workspace has no
        # room for a byte: verified, nothing written, no allowance asked.
        runtime.storage_admission = _refuse
        replay()
        assert receipt_path.read_bytes() == original_receipt
        # Interrupted after the set was placed and before the receipt: the
        # retry admits the receipt's bytes alone and places it.
        receipt_path.unlink()
        admitted.clear()
        runtime.storage_admission = admitted.append
        replay()
        assert admitted == [len(original_receipt)]
        assert receipt_path.read_bytes() == original_receipt
        # A receipt under the receipt's name with other bytes: refused by
        # name, never overwritten, never taken for the record.
        damaged_receipt = original_receipt.replace(b'"queries":', b'"queries": ')
        assert damaged_receipt != original_receipt
        receipt_path.write_bytes(damaged_receipt)
        admitted.clear()
        with pytest.raises(AlternativeEvidencePublicationError, match="artifact_identity_reused"):
            replay()
        assert admitted == [] and receipt_path.read_bytes() == damaged_receipt
        receipt_path.write_bytes(original_receipt)
        # Without a set to place, the set the receipt names must be there.
        set_path.unlink()
        with pytest.raises(FileNotFoundError, match="artifact_missing"):
            runtime._publish_selection(None, receipt)
        set_path.write_bytes(original_set)
        # A set that is not the receipt's named dependency is refused.
        other = seal_contract(
            AlternativeEvidenceResolvedSpanSet,
            "span_set_hash",
            request_hash=receipt.request_hash,
            retrieval_generation_hash=receipt.retrieval_generation_hash,
            spans=spans[:1],
        )
        assert other.span_set_hash != receipt.span_set_hash
        with pytest.raises(AlternativeEvidencePublicationError, match="selection_dependency"):
            runtime._publish_selection(other, receipt)
        assert not (
            store.root / SPAN_SET_CATEGORY / f"{blake3_filename(other.span_set_hash)}.json"
        ).is_file()
    finally:
        runtime.close()

    # Another process over the same artifacts, the receipt gone and the set
    # under its name holding other bytes: the selection is computed again
    # and refused by name at its publication -- the damaged set is not
    # taken for the set, and the receipt is not placed beside it.
    later, _passes2 = _counted_runtime(tmp_path)
    try:
        receipt_path.unlink()
        damaged_set = original_set.replace(b'"spans":[', b'"spans":[ ', 1)
        assert damaged_set != original_set
        set_path.write_bytes(damaged_set)
        admitted.clear()
        later.storage_admission = admitted.append
        with pytest.raises(AlternativeEvidencePublicationError, match="artifact_identity_reused"):
            later.select_evidence(request=request, document_set=document_set, generation=generation)
        assert admitted == [] and not receipt_path.is_file()
        assert set_path.read_bytes() == damaged_set
        # The set restored: the recomputed selection delivers the same spans
        # and names the held set, and places its receipt alone, admitted at
        # its own bytes.
        set_path.write_bytes(original_set)
        recomputed, recomputed_spans = later.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        assert [span.span_handle for span in recomputed_spans] == handles
        assert recomputed.span_set_hash == receipt.span_set_hash
        assert recomputed.reused_from_receipt_hash is None
        assert admitted == [len(store.serialized(recomputed))]
        assert set_path.read_bytes() == original_set
        assert (
            store.root / RECEIPT_CATEGORY / f"{blake3_filename(recomputed.receipt_hash)}.json"
        ).is_file()
    finally:
        later.close()


@pytest.mark.proves("alternative_evidence.retrieval_change_keeps_or_records_beside")
def test_a_selection_under_an_earlier_retrieval_binding_is_kept_or_recorded_beside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selection made again under a new binding is sealed under the current generation, naming
    the newest earlier receipt when passages and score types match, else recording the change
    beside it; a late rerank, a stop between the two writes and two upgrades included."""

    def select(folder: Path, *, binding: str | None, rerank: str, stop: bool = False) -> Any:
        runtime, _passes = _counted_runtime(folder)
        with monkeypatch.context() as patch:
            # This publication case compares a fully scored result with its fused fallback.
            patch.setattr(packet, "QUERY_TOP_K", 2)
            if binding is not None:
                for owner in (runtime, runtime.retrieval):
                    patch.setattr(owner, "retrieval_binding_hash", binding)
            if stop:  # the Host stops after the receipt is written, before its change's record
                place = runtime.artifacts.place

                def refused(items: Any, **options: Any) -> Any:
                    kept = [item for item in items if item[0] != SUPERSESSION_CATEGORY]
                    if len(kept) == len(items):
                        return place(items, **options)
                    place(kept, **options)
                    raise RuntimeError("stop")

                patch.setattr(runtime.artifacts, "place", refused)
            if rerank == "past the deadline":  # as the cross-encoder answers past its deadline
                patch.setattr(
                    _FakeReranker,
                    "score",
                    lambda self, query, passages, *, deadline: RerankScores((), timed_out=True),
                )
            if rerank == "reversed":
                score = _FakeReranker.score

                def reversed_scores(
                    self: _FakeReranker, query: str, passages: tuple[str, ...], *, deadline: float
                ) -> RerankScores:
                    result = score(self, query, passages, deadline=deadline)
                    return RerankScores(tuple(-v for v in result.scores), result.timed_out)

                patch.setattr(_FakeReranker, "score", reversed_scores)
            request, document_set, generation = _built(runtime)
            receipt, _spans = runtime.select_evidence(
                request=request, document_set=document_set, generation=generation
            )
        return runtime, receipt

    for name, before, now in (
        ("same", "scored", "scored"),
        ("changed", "reversed", "scored"),
        ("deadline", "scored", "past the deadline"),
        ("twice", "reversed", "scored"),
    ):
        earlier, prior = select(
            tmp_path / name, binding=fixture_digest("earlier retrieval binding"), rerank=before
        )
        earlier.close()
        if name == "twice":  # changed under one binding, then equal under the next
            earlier, prior = select(
                tmp_path / name, binding=fixture_digest("current retrieval binding"), rerank=now
            )
            earlier.close()
        if name == "changed":
            with pytest.raises(RuntimeError, match="stop"):
                select(tmp_path / name, binding=None, rerank=now, stop=True)
        runtime, receipt = select(tmp_path / name, binding=None, rerank=now)
        try:
            records = runtime.artifacts.values(SUPERSESSION_CATEGORY, RetrievalSupersessionRecord)
            assert (
                runtime.artifacts.load(RECEIPT_CATEGORY, prior.receipt_hash, type(prior)) == prior
            ), "the earlier receipt stays as sealed"
            if name == "twice":  # the newest earlier receipt is the one whose evidence it is
                (record,) = records
                assert record.receipt_hash == prior.receipt_hash
                assert receipt.reused_from_receipt_hash == prior.receipt_hash
            elif name == "same":  # the current chain, naming the receipt whose evidence it is
                assert receipt.retrieval_generation_hash != prior.retrieval_generation_hash
                assert receipt.reused_from_receipt_hash == prior.receipt_hash and records == ()
                runtime.close()
                runtime, again = select(tmp_path / name, binding=None, rerank=now)
                assert again == receipt
            elif name == "deadline":  # the fused order stands, recorded beside, and replays alike
                (record,) = records
                assert {(q.prior_score_type, q.score_type) for q in record.queries} == {
                    ("CROSS_ENCODER_LOGIT", "LINEAR_FUSION")
                }
                runtime.close()
                runtime, again = select(tmp_path / name, binding=None, rerank=now)
                assert again.receipt_hash == receipt.receipt_hash
                assert runtime.artifacts.values(
                    SUPERSESSION_CATEGORY, RetrievalSupersessionRecord
                ) == (record,)
            else:
                assert receipt.reused_from_receipt_hash is None
                (record,) = records
                assert (record.prior_receipt_hash, record.receipt_hash) == (
                    prior.receipt_hash,
                    receipt.receipt_hash,
                )
                assert record.queries and all(q.removed or q.added for q in record.queries)
                assert {q.score_type for q in record.queries} <= {"CROSS_ENCODER_LOGIT"}
        finally:
            runtime.close()
