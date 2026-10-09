"""The publication boundary of a session's selection: what is already on
disk is verified through the store's own boundary, never taken for the
record because its path exists (record section X, C0)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from alphalattice.evidence.alternative_evidence.contracts import seal_contract
from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    AlternativeEvidenceArtifactStore,
    AlternativeEvidencePublicationError,
    verified_evidence_records,
)
from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
    AlternativeEvidenceResolvedSpanSet,
    PairScoreCommitmentRecord,
)
from alphalattice.evidence.alternative_evidence.runtime.reuse import (
    PAIR_SCORE_COMMITMENT_CATEGORY,
    RECEIPT_CATEGORY,
    SPAN_SET_CATEGORY,
)
from alphalattice.kernel.knowledge import _reranking as reranking
from tests.alternative_evidence_desk.document_intelligence_support import (
    _built,
    _counted_runtime,
)


class _Note(BaseModel):
    text: str
    note_hash: str


class _FrozenNote(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    note_hash: str


def _member(text: str, digit: str) -> tuple[str, str, _Note]:
    return (SPAN_SET_CATEGORY, digit * 64, _Note(text=text, note_hash=digit * 64))


def _refuse(_additional: int) -> None:
    raise RuntimeError("storage.managed_capacity_exceeded")


def test_the_store_places_what_it_does_not_hold_and_verifies_what_it_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (C0, the owning operation): members the store does not
    hold are admitted as one sum and written in the given order; members it
    holds are verified and cost no admission, so a set held whole is a
    valid retry that writes nothing and needs no allowance; one held and
    one missing admits the missing bytes alone; a refused admission places
    nothing; a member on disk under its identity with other bytes refuses
    by name before any admission or write, and its path existing proves
    nothing."""

    store = AlternativeEvidenceArtifactStore(tmp_path)
    first, second = _member("the set", "1"), _member("its receipt", "2")
    members = (first, second)
    sizes = [len(store.serialized(model)) for _category, _identity, model in members]
    paths = [store.root / category / f"{identity}.json" for category, identity, _m in members]
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
    """regression: inside one operation's scope a record loaded by its
    identity is read and verified once and its later loads return it; an
    operation inside another joins the scope; a record that is not frozen is
    never kept; outside a scope every load reads the file and verifies it
    again, so a damaged record refuses by name."""

    store = AlternativeEvidenceArtifactStore(tmp_path)
    note = _FrozenNote(text="the set", note_hash="1" * 64)
    loose = _Note(text="its receipt", note_hash="2" * 64)
    store.publish(SPAN_SET_CATEGORY, note.note_hash, note)
    store.publish(SPAN_SET_CATEGORY, loose.note_hash, loose)
    with verified_evidence_records():
        with verified_evidence_records():
            first = store.load(SPAN_SET_CATEGORY, note.note_hash, _FrozenNote)
        assert store.load(SPAN_SET_CATEGORY, note.note_hash, _FrozenNote) is first
        store.load(SPAN_SET_CATEGORY, loose.note_hash, _Note)
        store.load(SPAN_SET_CATEGORY, loose.note_hash, _Note)
    assert store.read_count == 3
    path = store.root / SPAN_SET_CATEGORY / f"{note.note_hash}.json"
    path.write_bytes(path.read_bytes().replace(b"1111", b"2222", 1))
    with pytest.raises(AlternativeEvidencePublicationError, match="artifact_tampered"):
        store.load(SPAN_SET_CATEGORY, note.note_hash, _FrozenNote)


def test_a_selection_is_published_only_over_verified_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (C0, the public path): the receipt and span set of a
    selection are admitted together and placed in dependency order; the
    receipt's own selection replayed at capacity writes nothing and needs no
    allowance; a retry after an interruption between the two placements
    admits and places the receipt alone; a receipt or span set already
    under the name with other bytes (the lead's probe, on the owning path)
    refuses the selection by name -- nothing overwritten, nothing placed
    beside it; without a set to place the set the receipt names must be
    there and be the one delivered; a set that is not the receipt's named
    dependency is refused."""

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
        receipt_path = store.root / RECEIPT_CATEGORY / f"{receipt.receipt_hash}.json"
        set_path = store.root / SPAN_SET_CATEGORY / f"{receipt.span_set_hash}.json"
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
        assert not (store.root / SPAN_SET_CATEGORY / f"{other.span_set_hash}.json").is_file()
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
        # its own bytes (another receipt: this process's session served
        # the committed pair scores and committed no block of its own).
        set_path.write_bytes(original_set)
        recomputed, recomputed_spans = later.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        assert [span.span_handle for span in recomputed_spans] == handles
        assert recomputed.span_set_hash == receipt.span_set_hash
        assert recomputed.reused_from_receipt_hash is None
        assert admitted == [len(store.serialized(recomputed))]
        assert set_path.read_bytes() == original_set
        assert (store.root / RECEIPT_CATEGORY / f"{recomputed.receipt_hash}.json").is_file()
    finally:
        later.close()


def test_a_commitment_already_under_its_name_is_verified_not_assumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (C0, the commitment path): the pair-score commitment is
    placed through the same operation -- a record under the commitment's
    name that is not it refuses the selection by name and is never named by
    the receipt, while an admission refused for capacity leaves the blocks
    unanchored and the selection whole, as before."""

    monkeypatch.setattr(reranking, "PAIR_SCORE_BLOCK_MINIMUM", 1)
    runtime, _passes = _counted_runtime(tmp_path)
    try:
        request, document_set, generation = _built(runtime)
        original_place = runtime.artifacts.place

        def placing(members: Any, *, admit: Any) -> int:
            if members[0][0] == PAIR_SCORE_COMMITMENT_CATEGORY:
                raise AlternativeEvidencePublicationError(
                    "alternative_evidence.artifact_identity_reused"
                )
            return int(original_place(members, admit=admit))

        monkeypatch.setattr(runtime.artifacts, "place", placing)
        with pytest.raises(AlternativeEvidencePublicationError, match="artifact_identity_reused"):
            runtime.select_evidence(
                request=request, document_set=document_set, generation=generation
            )
        assert runtime.pair_score_commitments_refused == 0
        commitments = runtime.artifacts.values(
            PAIR_SCORE_COMMITMENT_CATEGORY, PairScoreCommitmentRecord
        )
        assert commitments == ()

        def refusing(members: Any, *, admit: Any) -> int:
            if members[0][0] == PAIR_SCORE_COMMITMENT_CATEGORY:
                raise RuntimeError("storage.managed_capacity_exceeded")
            return int(original_place(members, admit=admit))

        monkeypatch.setattr(runtime.artifacts, "place", refusing)
        receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        assert spans and receipt.pair_score_commitment_hash is None
        assert runtime.pair_score_commitments_refused == 1
    finally:
        runtime.close()
