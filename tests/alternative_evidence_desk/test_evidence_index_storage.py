"""Evidence index payloads under the workspace storage owner.

The store's bytes are accounted apart (source blobs, sealed artifacts, vector
payloads, index payloads, staging, task state); a published index is a
candidate only when no unfinished Task, open reader or pin needs it; a plan
is bound to the references it saw and stales on a new pin; an approved
cleanup journals its intent, releases the index through its owner (a marker
stays), survives an interruption, and reconciles predicted with freed bytes;
after eviction the review's exact export and the packet still read, a cold
open refuses by the plan's name, the next preparation of the same corpus and
the explicit rebuild restore the index from its committed vectors with no
model run, and a lost payload makes the explicit rebuild run the model once
and reproduce the committed digest.
"""

from __future__ import annotations

import sqlite3
import urllib.parse
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    EvidenceReviewAuthority,
)
from alphalattice.control.product_host.composition.research_workspace import (
    publish_research_workspace_manifest,
)
from alphalattice.control.product_host.storage.inventory import (
    managed_file_inventory,
    unique_managed_bytes,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceRetrievalAccessReceipt,
)
from alphalattice.evidence.alternative_evidence.retrieval.contracts import PairScoreCommitmentRecord
from alphalattice.evidence.alternative_evidence.runtime.policy import AdmittedEvidencePolicy
from alphalattice.evidence.alternative_evidence.runtime.service import (
    AlternativeEvidenceDocumentIntelligenceRuntime,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    AlternativeEvidenceDocumentTaskAdapter,
    AlternativeEvidenceDocumentTaskResources,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from alphalattice.protocols.actor_execution import ActorKind
from tests.alternative_evidence_desk.issuer_listing import (
    ISSUER_TOPICS,
    _documents,
    _listing_authority,
    _registry,
)
from tests.alternative_evidence_desk.planted_corpus import (
    _NOW,
    _CitingActor,
    _index_factory,
    _retriever_factory,
)
from tests.alternative_evidence_desk.review_dossiers import CitingReviewActor, ControlledRisk
from tests.alternative_evidence_desk.review_http_support import (
    _Service,
    _workspace_manifest,
    build_workspace,
    start_service,
)

INDEX_ROOT = "runtime/evidence-knowledge/.system/knowledge-indexes"


ONE_UNIT = "u01"
"""The book's only unit: every book is prepared as a coverage run."""


def _authority(workspace: Path, report: Any) -> EvidenceReviewAuthority:
    """The production shape: the evidence store lives under the workspace's runtime."""

    listings = tuple(value.listing_id for value in report.window_end_book.positions)
    runtime = AlternativeEvidenceDocumentIntelligenceRuntime(
        artifact_root=workspace / "runtime" / "artifacts",
        workspace_root=workspace / "runtime" / "evidence-knowledge",
        model_root=workspace / "runtime" / "models",
        index_factory=_index_factory,
        retriever_factory=_retriever_factory,
    )
    listing_authority = _listing_authority(listings)
    entities = tuple(sorted({value.ticker for value in listing_authority.entries}))
    return EvidenceReviewAuthority(
        model_authority_admitted=True,
        registry=_registry(),
        listing_authority=listing_authority,
        artifacts=runtime.artifacts,
        evidence_publications=runtime.publications,
        evidence_runtime=runtime,
        evidence_resources=AlternativeEvidenceDocumentTaskResources(
            recorded_registry=_registry(),
            recorded_documents=_documents(entities),
            analysis_actor=_CitingActor(topics=ISSUER_TOPICS),
        ),
        evidence_policy=AdmittedEvidencePolicy(),
        review_actor=CitingReviewActor(
            risks=(ControlledRisk("AAPL"),),
            actor_kind=ActorKind.HUMAN,
            actor_id="qa-evidence-storage",
        ),
    )


_NOTHING_UNREFERENCED = {
    "index_paths": [],
    "index_bytes": 0,
    "payload_paths": [],
    "payload_bytes": 0,
    "source_object_paths": [],
    "source_object_bytes": 0,
    "vector_reference_graph": {"status": "PROVED", "unproved": []},
}


def _relative(workspace: Path, path: Path) -> str:
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def _evidence(service: _Service) -> dict[str, Any]:
    body = service.get("/api/workspace/storage")["evidence"]
    assert body is not None
    return cast(dict[str, Any], body)


def _index_rows(service: _Service) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], _evidence(service)["indexes"])


def test_evidence_indexes_are_accounted_protected_evicted_and_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, report = build_workspace(tmp_path)
    publish_research_workspace_manifest(workspace, _workspace_manifest("qa-gate-9c5-http"))
    now = [_NOW]
    authority = _authority(workspace, report)
    service = start_service(workspace, authority, tmp_path, clock=lambda: now[0])
    try:
        runtime = authority.evidence_runtime
        assert runtime is not None
        result_hash = service.result_hash()
        selected = {"result_hash": result_hash}

        # Nothing prepared: the store is accounted, no index, nothing to release.
        empty = _evidence(service)
        assert empty["indexes"] == [] and empty["index_payload_bytes"] == 0

        refreshed = service.post("/api/evidence-refresh", selected)
        assert refreshed["disposition"] == "ADMITTED"
        service.drain()
        assert (
            service.registry.task(UUID(refreshed["task_id"])).lifecycle is TaskLifecycle.SUCCEEDED
        )
        reviewed = service.post("/api/cro-review", selected)
        assert reviewed["disposition"] == "ADMITTED"
        service.drain()
        section = service.evidence_cro(result_hash)
        assert section["state"] == "REVIEW_PUBLISHED"
        review_hash = section["review_publication_hash"]
        exported = service.get(
            "/api/evidence-cro/export?"
            + urllib.parse.urlencode({**selected, "review_publication_hash": review_hash})
        )
        assert exported["review_status"] == "EXACT_HISTORICAL_READBACK"
        assert runtime.retrieval.passage_embedding_pass_count == 1
        # A bare preparation at the same cutoff: its own Task, the same sealed
        # generation record, nothing embedded -- and, its selection reused
        # whole from the sealed receipt, no index materialized or proved
        # (record section Y) -- its packet reads.
        prepared = service.post("/api/evidence/prepare", selected)
        assert prepared["disposition"] == "ADMITTED"
        service.drain()
        prepared_id = UUID(prepared["task_id"])
        assert service.registry.task(prepared_id).lifecycle is TaskLifecycle.SUCCEEDED
        assert runtime.retrieval.passage_embedding_pass_count == 1
        assert runtime.retrieval.generation_reuse_count == 0
        assert runtime.generation_builds_avoided == 1
        packet_query = "/api/evidence/packet?" + urllib.parse.urlencode(
            {**selected, "task_id": str(prepared_id), "evidence_unit_id": ONE_UNIT}
        )
        packet = service.get(packet_query)
        assert packet["status"] == "EVIDENCE_ANALYST_PACKET_READY"
        adapter = service.review.evidence_task_adapter
        analysis = adapter.published_analysis(
            UUID(refreshed["task_id"]), now=now[0], unit_id=ONE_UNIT
        )
        lineage = analysis.lineage

        # Accounted apart: one index, its payload, the blobs and the artifacts.
        view = _evidence(service)
        (row,) = view["indexes"]
        assert row["generation_format"] == "knowledge-hybrid-v4"
        # Who a cleanup affects (first-release A8): the Task that built the
        # generation, by goal and lifecycle, beside the row -- shown to a
        # person, never among the references a storage plan binds.
        builder = service.registry.task(UUID(refreshed["task_id"]))
        assert {
            "task_id": refreshed["task_id"],
            "goal_kind": str(builder.goal.goal_kind),
            "lifecycle": "SUCCEEDED",
            "unit_id": ONE_UNIT,
        } in row["readers"]
        assert row["record_count"] == 1
        assert row["availability"] == "AVAILABLE" and row["eligible"] is True
        assert row["protected_by"] == [] and row["payload_available"] is True
        assert row["relative_path"].startswith(INDEX_ROOT)
        assert view["index_payload_bytes"] == row["database_bytes"] > 0
        assert view["vector_payload_bytes"] == row["payload_bytes"] > 0
        assert view["source_blob_bytes"] > 0 and view["sealed_artifact_bytes"] > 0
        assert view["unreferenced"] == _NOTHING_UNREFERENCED
        storage = service.get("/api/workspace/storage")
        assert (
            storage["managed_bytes"] >= view["index_payload_bytes"] + view["vector_payload_bytes"]
        )
        index_id = row["index_id"]
        database = workspace / Path(*row["relative_path"].split("/"))
        vector_root = workspace / "runtime" / "evidence-knowledge" / ".system" / "knowledge-vectors"
        vector_files = {path: path.read_bytes() for path in vector_root.iterdir()}
        assert database.is_file() and row["payload_available"] is True
        assert any(path.suffix == ".json" for path in vector_files), "composed from blocks"

        # A pin protects it; a plan made before the pin is stale afterwards.
        plan = service.post("/api/workspace/storage/plan")
        assert row["relative_path"] in plan["targets"]
        assert plan["reclaimable_bytes"] >= row["database_bytes"]
        pinned = service.post(
            "/api/workspace/storage/pin", {"input_binding_hash": index_id, "input_pinned": True}
        )
        assert pinned["status"] == "PINNED" and pinned["index_id"] == index_id
        (row,) = _index_rows(service)
        assert row["protected_by"] == ["USER_PINNED"] and row["eligible"] is False
        refused = service.post(
            "/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]}
        )
        assert (
            refused.items()
            >= {"status": "REFUSED", "failure_code": "storage.cleanup_plan_stale"}.items()
        )
        assert database.is_file(), "a stale plan released nothing"
        assert service.post("/api/workspace/storage/plan")["targets"] == {}
        service.post(
            "/api/workspace/storage/pin", {"input_binding_hash": index_id, "input_pinned": False}
        )

        # A vector object no sealed generation composes -- a block a build
        # placed before it died -- is a candidate, released through the
        # owner, but never while a build is composing; the blocks and the
        # sidecar the generation composes are never candidates.
        stray = vector_root / ("1" * 64 + ".f32")
        stray.write_bytes(b"\x00" * 1536)
        view = _evidence(service)
        assert view["unreferenced"]["payload_paths"] == [_relative(workspace, stray)]
        assert view["unreferenced"]["payload_bytes"] == 1536
        assert view["vector_object_count"] == len(vector_files) + 1
        plan = service.post("/api/workspace/storage/plan")
        assert _relative(workspace, stray) in plan["targets"]
        assert not any(path.endswith((".blocks.json",)) for path in plan["targets"])
        assert not any(_relative(workspace, path) in plan["targets"] for path in vector_files), (
            "composed blocks are not candidates"
        )
        runtime.retrieval._hold_building("f" * 64)
        assert service.post("/api/workspace/storage/plan")["targets"] == {
            key: value for key, value in plan["targets"].items() if key.endswith(".db")
        }, "no vector object is a candidate while a build is composing"
        assert (
            service.post(
                "/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]}
            ).items()
            >= {"status": "REFUSED", "failure_code": "storage.cleanup_plan_stale"}.items()
        )
        runtime.retrieval._release_building("f" * 64)
        plan = service.post("/api/workspace/storage/plan")
        stray_only = {k: v for k, v in plan["targets"].items() if k.endswith(".f32")}
        assert stray_only == {
            _relative(workspace, stray): plan["targets"][_relative(workspace, stray)]
        }
        service.post(
            "/api/workspace/storage/pin", {"input_binding_hash": index_id, "input_pinned": True}
        )
        plan = service.post("/api/workspace/storage/plan")
        assert list(plan["targets"]) == [_relative(workspace, stray)]
        released = service.post(
            "/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]}
        )
        assert released["status"] == "COMPLETED" and released["deleted_paths"] == 1
        assert not stray.is_file()
        assert {path: path.read_bytes() for path in vector_root.iterdir()} == vector_files
        service.post(
            "/api/workspace/storage/pin", {"input_binding_hash": index_id, "input_pinned": False}
        )
        (row,) = _index_rows(service)

        # Approved eviction of two targets -- the index and an index a build
        # left without a record -- interrupted after the first release and
        # resumed: each is released through its owner and leaves a marker;
        # the resume releases only what remains; the payload, the blobs, the
        # artifacts and every review stay.
        orphan = database.with_name("0" * 64 + database.name[64:])
        orphan.write_bytes(database.read_bytes())
        index_bytes = database.stat().st_size
        before = index_bytes + orphan.stat().st_size
        view = _evidence(service)
        assert view["unreferenced"]["index_paths"] == [_relative(workspace, orphan)]
        assert view["unreferenced"]["index_bytes"] == orphan.stat().st_size
        plan = service.post("/api/workspace/storage/plan")
        assert sorted(plan["targets"]) == sorted(
            [row["relative_path"], _relative(workspace, orphan)]
        )
        assert plan["reclaimable_bytes"] == before
        evict = runtime.retrieval.evict_generation
        calls: list[str] = []

        def interrupted(relative: str, **kwargs: Any) -> int:
            calls.append(relative)
            if len(calls) == 2:
                raise OSError("simulated cleanup interruption")
            return evict(relative, **kwargs)

        monkeypatch.setattr(runtime.retrieval, "evict_generation", interrupted)
        status, body = service.request(
            "/api/workspace/storage/confirm",
            method="POST",
            payload={"storage_plan_hash": plan["plan_hash"]},
        )
        assert (status, body.get("status")) != (200, "COMPLETED"), body
        assert len(calls) == 2 and not orphan.is_file() and database.is_file()
        assert orphan.with_name(orphan.name + ".evicted.json").is_file()
        assert service.get("/api/workspace/storage")["status"] == "RECOVERY_REQUIRED"
        # While the approved cleanup waits to be recovered the owner takes no action
        # on an index, so none is offered.
        assert all(value["available_actions"] == [] for value in _index_rows(service))
        assert (
            service.post("/api/workspace/storage/plan").items()
            >= {"status": "REFUSED", "failure_code": "storage.cleanup_recovery_required"}.items()
        )
        resumed = service.post(
            "/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]}
        )
        monkeypatch.undo()
        assert resumed["status"] == "COMPLETED" and resumed["deleted_paths"] == 1
        assert len(calls) == 3 and calls[2] == calls[1]
        assert (
            not database.is_file() and database.with_name(database.name + ".evicted.json").is_file()
        )
        assert {path: path.read_bytes() for path in vector_root.iterdir()} == vector_files
        after = _evidence(service)
        (row,) = after["indexes"]
        assert row["availability"] == "EVICTED_BY_RETENTION" and row["database_bytes"] == 0
        assert after["index_payload_bytes"] == 0
        assert after["vector_payload_bytes"] == row["payload_bytes"]
        assert after["unreferenced"]["index_paths"] == []
        assert plan["reclaimable_bytes"] == before, "predicted and freed bytes agree"
        assert service.get("/api/workspace/storage")["status"] == "AVAILABLE"
        # The completed plan is idempotent: its receipt answers, nothing moves.
        assert service.post(
            "/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]}
        ) == {"status": "COMPLETED", "deleted_paths": 0, "plan_hash": plan["plan_hash"]}

        # Historical readback needs no index: the exact export, the section and
        # the prepared packet read as before.
        again = service.get(
            "/api/evidence-cro/export?"
            + urllib.parse.urlencode({**selected, "review_publication_hash": review_hash})
        )
        assert again == exported
        assert service.evidence_cro(result_hash)["state"] == "REVIEW_PUBLISHED"
        assert service.get(packet_query) == packet
        assert adapter.prepared_packet(prepared_id, now=now[0], unit_id=ONE_UNIT).spans

        # A cold open of the evicted generation refuses by the plan's name.
        with pytest.raises(KnowledgeRetrievalError) as refused_open:
            session = runtime.retrieval.open_session(
                document_set=lineage.document_set,
                generation=lineage.generation,
                evidence_as_of=now[0],
            )
            try:
                session.search(query="operating", top_k=3)
            finally:
                session.close()
        assert refused_open.value.failure.code == "retrieval.index_evicted"
        assert plan["plan_hash"][:12] in str(refused_open.value)
        assert runtime.retrieval.passage_embedding_pass_count == 1

        # The next preparation of the same corpus -- a new cutoff, its own Task
        # -- restores the index from the committed payload: no model run, the
        # marker gone, the same generation identity under a second record.
        now[0] = _NOW + timedelta(hours=2)
        later = service.post("/api/evidence/prepare", selected)
        assert later["disposition"] == "ADMITTED" and later["task_id"] != str(prepared_id)
        service.drain()
        assert service.registry.task(UUID(later["task_id"])).lifecycle is TaskLifecycle.SUCCEEDED
        assert runtime.retrieval.passage_embedding_pass_count == 1
        # The evicted index is restored by a build (the one generation reuse
        # so far: the bare preparation above reused its selection whole and
        # built nothing).
        assert runtime.retrieval.generation_reuse_count == 1
        assert runtime.generation_builds_avoided == 1
        assert database.is_file()
        assert not database.with_name(database.name + ".evicted.json").is_file()
        (row,) = _index_rows(service)
        assert row["availability"] == "AVAILABLE" and row["record_count"] == 2
        assert row["index_id"] == index_id

        # Evict again, then the explicit rebuild: zero model, receipted, and a
        # cold open serves the same hits it served before.
        plan = service.post("/api/workspace/storage/plan")
        assert row["relative_path"] in plan["targets"]
        service.post("/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]})
        assert not database.is_file()
        (evicted,) = _index_rows(service)
        assert evicted["available_actions"] == ["REBUILD"], "the owner rebuilds an evicted index"
        rebuilt = service.post(
            "/api/workspace/storage/evidence-rebuild", {"evidence_index_id": index_id}
        )
        assert rebuilt["status"] == "REBUILT" and rebuilt["passage_embedding_pass"] is False
        assert rebuilt["index_id"] == index_id and rebuilt["database_bytes"] == index_bytes
        assert (
            workspace
            / "artifacts"
            / "storage-governance"
            / "evidence-rebuilds"
            / f"{rebuilt['receipt_hash']}.json"
        ).is_file()
        assert database.is_file()
        assert runtime.retrieval.passage_embedding_pass_count == 1
        (row,) = _index_rows(service)
        assert row["availability"] == "AVAILABLE"
        session = runtime.retrieval.open_session(
            document_set=lineage.document_set,
            generation=lineage.generation,
            evidence_as_of=_NOW,
        )
        try:
            hits = session.search(query="operating", top_k=3).hits
        finally:
            session.close()
        assert hits and runtime.retrieval.passage_embedding_pass_count == 1

        # A lost block: the rebuild runs the model and must reproduce the
        # committed digest; it did, so the block is back as well.
        plan = service.post("/api/workspace/storage/plan")
        service.post("/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]})
        block = max(
            (path for path in vector_root.iterdir() if path.suffix == ".f32"),
            key=lambda path: path.stat().st_size,
        )
        block_bytes = block.read_bytes()
        block.unlink()
        assert _index_rows(service)[0]["payload_available"] is False
        recomputed = service.post(
            "/api/workspace/storage/evidence-rebuild", {"evidence_index_id": index_id}
        )
        assert recomputed["passage_embedding_pass"] is True
        # The composition it belonged to can no longer be proved as a whole,
        # and no other sealed generation holds these revisions: every
        # revision is embedded again, and must reproduce the digest.
        assert recomputed["embedded_chunks"] == recomputed["chunk_count"]
        assert block.read_bytes() == block_bytes and database.is_file()
        assert {path: path.read_bytes() for path in vector_root.iterdir()} == vector_files

        # An unknown or legacy-only index refuses by name.
        status, unknown = service.request(
            "/api/workspace/storage/evidence-rebuild",
            method="POST",
            payload={"evidence_index_id": "0" * 64},
        )
        assert status == 400
        assert "alternative_evidence.retrieval_generation_unknown" in unknown["refused"]
    finally:
        service.session.stop()


def _prepare(service: _Service, selected: dict[str, str]) -> tuple[UUID, dict[str, Any]]:
    prepared = service.post("/api/evidence/prepare", selected)
    assert prepared["disposition"] == "ADMITTED", prepared
    service.drain()
    task_id = UUID(prepared["task_id"])
    status = service.get(f"/api/status?task_id={task_id}")
    return task_id, status


def _used_bytes(workspace: Path) -> int:
    return unique_managed_bytes(managed_file_inventory(workspace))


def test_evidence_writes_are_admitted_by_the_workspace_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence writes are admitted by the workspace budget."""

    workspace, report = build_workspace(tmp_path)
    publish_research_workspace_manifest(workspace, _workspace_manifest("qa-gate-9c5-http"))
    now = [_NOW]
    authority = _authority(workspace, report)
    service = start_service(workspace, authority, tmp_path, clock=lambda: now[0])
    try:
        runtime = authority.evidence_runtime
        assert runtime is not None
        storage = service.session.operations.storage  # type: ignore[union-attr]
        assert runtime.storage_admission is not None, "the Host bound the budget owner"

        def set_cap(value: int | str) -> None:
            configured = storage.set_cap(str(value), caller="test")
            assert configured["status"] == "CONFIGURED"
            assert configured["capacity"]["setting"]["cap_bytes"] == value

        set_cap(10 * 1024 * 1024 * 1024)
        selected = {"result_hash": service.result_hash()}
        store = workspace / "runtime" / "evidence-knowledge"

        # The first preparation, under the explicit cap: every write admitted.
        first_id, status = _prepare(service, selected)
        assert status["lifecycle"] == "SUCCEEDED"
        assert runtime.retrieval.passage_embedding_pass_count == 1
        (row,) = _index_rows(service)
        database = workspace / Path(*row["relative_path"].split("/"))
        source_sets = sorted((runtime.artifacts.root / "source-document-sets").glob("*.json"))
        source_set_bytes = max(path.stat().st_size for path in source_sets)
        # The session's cross-encoder pair scores were sealed under the same
        # admission and committed (review findings R2 and R4): the blocks
        # sit under the managed evidence root and count there, the
        # commitment names them, and the session's receipt names the
        # commitment; the filing comparisons the routing sealed count too.
        inventory = {name for name, _size, _key in managed_file_inventory(workspace)}
        blocks = sorted((store / ".system" / "knowledge-scores").rglob("*.scores"))
        assert blocks and all(
            block.relative_to(workspace).as_posix() in inventory for block in blocks
        )
        commitments = sorted((runtime.artifacts.root / "pair-score-commitments").glob("*.json"))
        assert len(commitments) == 1
        assert commitments[0].relative_to(workspace).as_posix() in inventory
        receipts = [
            AlternativeEvidenceRetrievalAccessReceipt.model_validate_json(path.read_bytes())
            for path in (runtime.artifacts.root / "retrieval-access-receipts").glob("*.json")
        ]
        assert {r.pair_score_commitment_hash for r in receipts if r.pair_score_commitment_hash} == {
            commitments[0].stem
        }
        # A session admits the pair scores committed within the window's
        # length, and none committed before it (X2).
        committed = PairScoreCommitmentRecord.model_validate_json(commitments[0].read_bytes())
        assert runtime.pair_score_admission(since=committed.sealed_at).admitted == {
            b.name for b in blocks
        }
        later = committed.sealed_at + timedelta(microseconds=1)
        assert runtime.pair_score_admission(since=later).admitted == frozenset()
        comparisons = sorted((runtime.artifacts.root / "filing-comparisons").glob("*.json"))
        assert all(c.relative_to(workspace).as_posix() in inventory for c in comparisons)

        # At the cap exactly, a preparation is refused where it would first
        # grow the workspace -- its source set -- by the budget's own name;
        # nothing of it is placed, and the Task says why.
        set_cap(_used_bytes(workspace))
        now[0] = _NOW + timedelta(hours=1)
        blocked_id, status = _prepare(service, selected)
        assert status["lifecycle"] == "BLOCKED"
        assert (
            status["latest_failure_code"]
            == "alternative_evidence.task_failed:storage.managed_capacity_exceeded"
        )
        assert sorted((runtime.artifacts.root / "source-document-sets").glob("*.json")) == (
            source_sets
        )
        # What is already there still reads: the exact packet of the first
        # preparation and a cold open of its generation cost no admission.
        packet = service.get(
            "/api/evidence/packet?"
            + urllib.parse.urlencode(
                {**selected, "task_id": str(first_id), "evidence_unit_id": ONE_UNIT}
            )
        )
        assert packet["status"] == "EVIDENCE_ANALYST_PACKET_READY"

        # Room for the request's own small artifacts, none for a build: the
        # index is evicted first, so the next preparation would have to place
        # it again. It is refused at the build, after the admitted stages
        # committed their own bytes, and leaves no staged or orphan file.
        set_cap(10 * 1024 * 1024 * 1024)
        plan = service.post("/api/workspace/storage/plan")
        assert row["relative_path"] in plan["targets"]
        service.post("/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]})
        assert not database.is_file()
        (row,) = _index_rows(service)
        assert row["availability"] == "EVICTED_BY_RETENTION" and row["payload_available"] is True
        records = sorted((runtime.artifacts.root / "retrieval-generations").glob("*.json"))
        # The request's own small artifacts include its run's record (every
        # book is a coverage run).
        run_record_bytes = max(
            path.stat().st_size
            for path in (runtime.artifacts.root / "evidence-coverage-runs").glob("*.json")
        )
        set_cap(_used_bytes(workspace) + source_set_bytes + run_record_bytes + 64 * 1024)
        now[0] = _NOW + timedelta(hours=2)
        refused_id, status = _prepare(service, selected)
        assert status["lifecycle"] == "BLOCKED"
        assert (
            status["latest_failure_code"]
            == "alternative_evidence.task_failed:storage.managed_capacity_exceeded"
        )
        assert not database.is_file(), "a refused build placed nothing"
        assert database.with_name(database.name + ".evicted.json").is_file()
        assert not list((store / ".system" / "staging").glob("*")), "no staged remains"
        assert sorted((runtime.artifacts.root / "retrieval-generations").glob("*.json")) == records
        assert runtime.retrieval.passage_embedding_pass_count == 1, "no model was run for it"
        view = _evidence(service)
        assert view["unreferenced"] == _NOTHING_UNREFERENCED
        # The recovery view names the stop and both ways to make room:
        # raise the operator's cap or confirm cleanup, keeping verified work.
        view = service.get(f"/api/tasks/recovery?task_id={refused_id}")
        assert view["stop"]["code"].endswith(":storage.managed_capacity_exceeded")
        assert view["stop"]["stage_id"] == f"{ONE_UNIT}_build_retrieval_generation"
        assert view["stop"]["detail"] == (
            "The stage's write exceeds the workspace storage cap. Raise the cap in Settings "
            "or preview and confirm a cleanup, then resume the Task's offered request; "
            "its verified stages and retained results stay intact."
        )
        # A blocked evidence Task is terminal for Task Control: the way
        # forward is room and a new preparation, which restores the index
        # from the retained payload without the model.
        status, body = service.request(
            "/api/recover",
            method="POST",
            payload={"task_id": str(refused_id), "expected_task_hash": status["task_record_hash"]},
        )
        assert status == 200 and body["disposition"] == "NOT_RECOVERY_REQUIRED"
        set_cap(10 * 1024 * 1024 * 1024)
        now[0] = _NOW + timedelta(hours=3)
        _restored_id, status = _prepare(service, selected)
        assert status["lifecycle"] == "SUCCEEDED"
        assert database.is_file() and runtime.retrieval.passage_embedding_pass_count == 1
        assert runtime.retrieval.generation_reuse_count == 1

        # Shared bytes count once: with room for the request's artifacts only
        # -- its source set, its receipt and its span set, the durable writes
        # a preparation over a present generation makes -- a further cutoff
        # is admitted, because the payload and the index add nothing.
        receipt_bytes = max(
            path.stat().st_size
            for path in (runtime.artifacts.root / "retrieval-access-receipts").glob("*.json")
        )
        span_set_bytes = max(
            path.stat().st_size
            for path in (runtime.artifacts.root / "resolved-span-sets").glob("*.json")
        )
        set_cap(
            _used_bytes(workspace) + source_set_bytes + receipt_bytes + span_set_bytes + 64 * 1024
        )
        now[0] = _NOW + timedelta(hours=4)
        _again_id, status = _prepare(service, selected)
        assert status["lifecycle"] == "SUCCEEDED"
        # The selection reused whole: no index built or proved for it.
        assert runtime.retrieval.generation_reuse_count == 1
        assert runtime.generation_builds_avoided == 1

        # The explicit rebuild obeys the same admission.
        set_cap(10 * 1024 * 1024 * 1024)
        plan = service.post("/api/workspace/storage/plan")
        service.post("/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]})
        assert not database.is_file()
        set_cap(_used_bytes(workspace))
        refused = service.post(
            "/api/workspace/storage/evidence-rebuild", {"evidence_index_id": row["index_id"]}
        )
        assert (
            refused.items()
            >= {"status": "REFUSED", "failure_code": "storage.managed_capacity_exceeded"}.items()
        )
        assert not database.is_file() and not list((store / ".system" / "staging").glob("*"))
        set_cap(10 * 1024 * 1024 * 1024)
        rebuilt = service.post(
            "/api/workspace/storage/evidence-rebuild", {"evidence_index_id": row["index_id"]}
        )
        assert rebuilt["status"] == "REBUILT" and rebuilt["passage_embedding_pass"] is False
        assert database.is_file()
        del blocked_id
        # The automatic operator cap resolves without the retention plan's
        # reference analysis: with the evidence generations' inventory refusing, the
        # real resolver still names the cap and an admission still completes,
        # while the readback -- whose plan needs that inventory -- is what
        # walks it. Before this (section W) every admission rebuilt the
        # inventory and proved every generation's vector payload again.
        set_cap("auto")
        from alphalattice.control.product_host.storage.evidence_references import (
            EvidenceStorageReferences,
        )

        def refuse(self: object) -> tuple[object, ...]:
            raise AssertionError("an admission must not inventory the evidence generations")

        monkeypatch.setattr(EvidenceStorageReferences, "rows", refuse)
        assert storage.capacity_cap_bytes() >= _used_bytes(workspace)
        storage.admit_evidence_bytes(1)
        with pytest.raises(AssertionError, match="must not inventory"):
            storage.readback()
    finally:
        service.session.stop()


def test_open_readers_and_unsealed_builds_hold_their_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Open readers and unsealed builds hold their generation."""

    workspace, report = build_workspace(tmp_path)
    publish_research_workspace_manifest(workspace, _workspace_manifest("qa-gate-9c5-http"))
    now = [_NOW]
    authority = _authority(workspace, report)
    service = start_service(workspace, authority, tmp_path, clock=lambda: now[0])
    try:
        runtime = authority.evidence_runtime
        assert runtime is not None
        selected = {"result_hash": service.result_hash()}
        refreshed = service.post("/api/evidence-refresh", selected)
        service.drain()
        adapter = service.review.evidence_task_adapter
        lineage = adapter.published_analysis(
            UUID(refreshed["task_id"]), now=now[0], unit_id=ONE_UNIT
        ).lineage
        (row,) = _index_rows(service)
        database = workspace / Path(*row["relative_path"].split("/"))
        assert row["protected_by"] == [] and row["eligible"] is True

        def open_reader() -> Any:
            return runtime.retrieval.open_session(
                document_set=lineage.document_set,
                generation=lineage.generation,
                evidence_as_of=now[0],
            )

        # An open session holds the generation until it closes.
        session = open_reader()
        (row,) = _index_rows(service)
        assert row["protected_by"] == ["ACTIVE_LEASE"] and row["eligible"] is False
        assert service.post("/api/workspace/storage/plan")["targets"] == {}
        session.close()
        (row,) = _index_rows(service)
        assert row["protected_by"] == [] and row["eligible"] is True

        # A hold taken between the plan and its apply stales the plan.
        plan = service.post("/api/workspace/storage/plan")
        assert row["relative_path"] in plan["targets"]
        session = open_reader()
        assert (
            service.post(
                "/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]}
            ).items()
            >= {"status": "REFUSED", "failure_code": "storage.cleanup_plan_stale"}.items()
        )
        assert database.is_file()
        session.close()

        # A hold taken inside the apply -- after its own reference check --
        # is refused at the target by the index owner; the cleanup stops
        # there, journalled, and resumes once the hold ends.
        plan = service.post("/api/workspace/storage/plan")
        evict = runtime.retrieval.evict_generation
        held: list[Any] = []

        def opens_then_evicts(relative: str, **kwargs: Any) -> int:
            held.append(open_reader())
            return evict(relative, **kwargs)

        monkeypatch.setattr(runtime.retrieval, "evict_generation", opens_then_evicts)
        assert (
            service.post(
                "/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]}
            ).items()
            >= {"status": "REFUSED", "failure_code": "storage.cleanup_target_in_use"}.items()
        )
        monkeypatch.undo()
        assert database.is_file()
        assert service.get("/api/workspace/storage")["status"] == "RECOVERY_REQUIRED"
        held.pop().close()
        resumed = service.post(
            "/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]}
        )
        assert resumed["status"] == "COMPLETED" and resumed["deleted_paths"] == 1
        assert not database.is_file()

        # A build holds its generation from before the model runs until its
        # lease is registered: an index placed but not yet sealed is not
        # abandoned storage. Seen from inside the sealing step, the plan
        # names no target and the index is held by the build's own lease
        # (here the restored index of an evicted generation, so the earlier
        # records still name it; a first generation would be an unsealed
        # file, which `targets()` skips by the same lease).
        publish = runtime.artifacts.publish
        seen: list[dict[str, Any]] = []
        storage = service.session.operations.storage  # type: ignore[union-attr]

        def plans_before_sealing(category: str, identity: str, model: Any) -> None:
            if category == "retrieval-generations" and not seen:
                readback = storage.readback()["evidence"]
                seen.append(
                    {
                        "targets": storage.plan()["targets"],
                        "active": sorted(runtime.retrieval.active_index_ids()),
                        "protected": [
                            (value["availability"], value["protected_by"])
                            for value in readback["indexes"]
                        ],
                    }
                )
            publish(category, identity, model)

        monkeypatch.setattr(runtime.artifacts, "publish", plans_before_sealing)
        now[0] = _NOW + timedelta(hours=1)
        _task_id, status = _prepare(service, selected)
        monkeypatch.undo()
        assert status["lifecycle"] == "SUCCEEDED"
        (observed,) = seen
        assert observed["targets"] == {}
        assert observed["active"] == [row["index_id"]]
        assert observed["protected"] == [("AVAILABLE", ["ACTIVE_LEASE"])]
        assert database.is_file()

        # A Task still owed its work holds the generation it built: the
        # interruption lands after the build's evidence was written; the
        # next service resumes the Task, and the hold ends with it.
        original_verify = AlternativeEvidenceDocumentTaskAdapter.verify_stage

        def interrupted_after_build(self: Any, **kwargs: Any) -> Any:
            if kwargs["work_item"].stage_id.endswith("select_evidence_spans"):
                raise RuntimeError("simulated interruption after the build")
            return original_verify(self, **kwargs)

        monkeypatch.setattr(
            AlternativeEvidenceDocumentTaskAdapter, "verify_stage", interrupted_after_build
        )
        now[0] = _NOW + timedelta(hours=2)
        prepared = service.post("/api/evidence/prepare", selected)
        service.drain()
        monkeypatch.undo()
        owed = UUID(prepared["task_id"])
        assert service.registry.task(owed).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        (row,) = _index_rows(service)
        assert row["protected_by"] == ["IN_FLIGHT_RECOVERY"] and row["eligible"] is False
        assert service.post("/api/workspace/storage/plan")["targets"] == {}
        service.session.stop()
        service = start_service(workspace, authority, tmp_path, clock=lambda: now[0])
        assert service.session.resumed_task_ids == (owed,)
        service.drain()
        assert service.registry.task(owed).lifecycle is TaskLifecycle.SUCCEEDED
        (row,) = _index_rows(service)
        assert row["protected_by"] == [] and row["eligible"] is True
        assert row["relative_path"] in service.post("/api/workspace/storage/plan")["targets"]
    finally:
        service.session.stop()


def _vector_targets(plan: dict[str, Any]) -> list[str]:
    return sorted(path for path in plan["targets"] if path.endswith((".f32", ".blocks.json")))


def test_an_unprovable_vector_reference_graph_refuses_vector_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unprovable vector reference graph refuses vector cleanup."""

    from alphalattice.kernel.knowledge.hybrid_contracts import (
        HybridVectorBlockSidecar,
    )
    from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes

    workspace, report = build_workspace(tmp_path)
    publish_research_workspace_manifest(workspace, _workspace_manifest("qa-gate-9c5-http"))
    now = [_NOW]
    authority = _authority(workspace, report)
    service = start_service(workspace, authority, tmp_path, clock=lambda: now[0])
    try:
        runtime = authority.evidence_runtime
        assert runtime is not None
        selected = {"result_hash": service.result_hash()}
        _first_id, status = _prepare(service, selected)
        assert status["lifecycle"] == "SUCCEEDED"
        # A second record over the same generation: shared blocks count once.
        now[0] = _NOW + timedelta(hours=1)
        _second_id, status = _prepare(service, selected)
        assert status["lifecycle"] == "SUCCEEDED"
        (row,) = _index_rows(service)
        assert row["record_count"] == 2 and row["payload_available"] is True
        assert row["payload_proof"] == "PROVED"
        vector_root = workspace / "runtime" / "evidence-knowledge" / ".system" / "knowledge-vectors"
        (sidecar,) = [path for path in vector_root.iterdir() if path.suffix == ".json"]
        blocks = [path for path in vector_root.iterdir() if path.suffix == ".f32"]
        original = {path: path.read_bytes() for path in vector_root.iterdir()}
        assert _evidence(service)["vector_object_count"] == len(original)
        stray = vector_root / ("1" * 64 + ".f32")
        stray.write_bytes(b"\x00" * 1536)
        view = _evidence(service)
        assert view["unreferenced"]["payload_paths"] == [_relative(workspace, stray)]
        assert view["unreferenced"]["vector_reference_graph"] == {
            "status": "PROVED",
            "unproved": [],
        }
        assert _vector_targets(service.post("/api/workspace/storage/plan")) == [
            _relative(workspace, stray)
        ]
        index_target = row["relative_path"]

        def graph() -> dict[str, Any]:
            return cast(
                dict[str, Any], _evidence(service)["unreferenced"]["vector_reference_graph"]
            )

        def refused_by(code: str) -> None:
            """No vector object is a candidate; the index still is; the row says why."""

            state = graph()
            assert state["status"] == "UNPROVED"
            (entry,) = state["unproved"]
            assert entry["index_id"] == row["index_id"] and entry["failure_code"] == code
            assert "rebuild" in entry["recovery"]
            unreferenced = _evidence(service)["unreferenced"]
            assert unreferenced["payload_paths"] == [] and unreferenced["payload_bytes"] == 0
            (current,) = _index_rows(service)
            assert current["payload_available"] is False and current["payload_proof"] == code
            plan = service.post("/api/workspace/storage/plan")
            assert _vector_targets(plan) == []
            assert index_target in plan["targets"], "index cleanup is independent"
            (refusal,) = plan["refusals"]
            assert refusal["failure_code"] == "storage.evidence_vector_references_unproved"
            assert refusal["generations"] == state["unproved"]

        # (1) The sidecar is missing: the blocks are still on disk and still
        # the sealed generation's vectors.
        sidecar_bytes = original[sidecar]
        sidecar.unlink()
        refused_by("retrieval.vector_payload_unavailable")
        sidecar.write_bytes(sidecar_bytes)
        assert graph() == {"status": "PROVED", "unproved": []}

        # (2) A truncated sidecar; the recovery step the refusal names is the
        # explicit rebuild: the index is present and committed, the model
        # composes the payload again (no other sealed generation holds these
        # revisions), the digest is reproduced, the blocks verify in place
        # and the damaged sidecar is replaced by the generation's own.
        sidecar.write_bytes(sidecar_bytes[: len(sidecar_bytes) // 2])
        refused_by("retrieval.vector_commitment_mismatch")
        passes = runtime.retrieval.passage_embedding_pass_count
        rebuilt = service.post(
            "/api/workspace/storage/evidence-rebuild", {"evidence_index_id": row["index_id"]}
        )
        assert rebuilt["status"] == "REBUILT" and rebuilt["passage_embedding_pass"] is True
        assert rebuilt["embedded_chunks"] == rebuilt["chunk_count"]
        assert runtime.retrieval.passage_embedding_pass_count == passes + 1
        assert sidecar.read_bytes() == sidecar_bytes
        assert {path: path.read_bytes() for path in vector_root.iterdir()} == {
            **original,
            stray: stray.read_bytes(),
        }
        assert graph() == {"status": "PROVED", "unproved": []}

        # (3) A structurally valid sidecar that omits a required block, and
        # one that misidentifies a block (names the stray's address).
        parsed = HybridVectorBlockSidecar.model_validate_json(sidecar_bytes)
        body = parsed.model_dump(mode="python")
        sidecar.write_bytes(
            canonical_json_bytes(
                HybridVectorBlockSidecar.model_validate({**body, "blocks": body["blocks"][:-1]})
            )
        )
        refused_by("retrieval.vector_commitment_mismatch")
        wrong = {
            **body["blocks"][-1],
            "block_sha256": "1" * 64,
            "vector_count": 1,
            "byte_length": 1536,
        }
        sidecar.write_bytes(
            canonical_json_bytes(
                HybridVectorBlockSidecar.model_validate(
                    {**body, "blocks": (*body["blocks"][:-1], wrong)}
                )
            )
        )
        refused_by("retrieval.vector_commitment_mismatch")
        sidecar.write_bytes(sidecar_bytes)
        assert graph() == {"status": "PROVED", "unproved": []}

        # (4) The same failure on a pinned generation: neither the index nor
        # any vector object is a candidate.
        service.post(
            "/api/workspace/storage/pin",
            {"input_binding_hash": row["index_id"], "input_pinned": True},
        )
        sidecar.unlink()
        assert service.post("/api/workspace/storage/plan")["targets"] == {}
        assert graph()["status"] == "UNPROVED"
        sidecar.write_bytes(sidecar_bytes)
        service.post(
            "/api/workspace/storage/pin",
            {"input_binding_hash": row["index_id"], "input_pinned": False},
        )

        # (5) Damage between preview and confirmation: the plan named the
        # stray over a proved graph (the index pinned, so the stray alone);
        # the confirmation rechecks the graph and refuses.
        service.post(
            "/api/workspace/storage/pin",
            {"input_binding_hash": row["index_id"], "input_pinned": True},
        )
        plan = service.post("/api/workspace/storage/plan")
        assert list(plan["targets"]) == [_relative(workspace, stray)]
        assert plan["refusals"] == []
        sidecar.unlink()
        assert (
            service.post(
                "/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]}
            ).items()
            >= {"status": "REFUSED", "failure_code": "storage.cleanup_target_protected"}.items()
        )
        assert stray.is_file() and all(path.is_file() for path in blocks)
        assert service.get("/api/workspace/storage")["status"] == "AVAILABLE", "nothing journalled"
        sidecar.write_bytes(sidecar_bytes)
        # Restored metadata restores ordinary behaviour: the genuine orphan
        # is released; the blocks and sidecar the generation composes stay.
        released = service.post(
            "/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]}
        )
        assert released["status"] == "COMPLETED" and released["deleted_paths"] == 1
        assert not stray.is_file()
        assert {path: path.read_bytes() for path in vector_root.iterdir()} == original
        service.post(
            "/api/workspace/storage/pin",
            {"input_binding_hash": row["index_id"], "input_pinned": False},
        )

        # (6) An approved cleanup of the index and a new orphan, interrupted
        # after the index was released; the graph breaks before the resume:
        # the resume is refused by the same protection and completes only
        # once the graph is proved again.
        stray.write_bytes(b"\x00" * 1536)
        plan = service.post("/api/workspace/storage/plan")
        assert sorted(plan["targets"]) == sorted([index_target, _relative(workspace, stray)])
        evict = runtime.retrieval.evict_generation
        calls: list[str] = []

        def interrupted(relative: str, **kwargs: Any) -> int:
            calls.append(relative)
            if len(calls) == 2:
                raise OSError("simulated cleanup interruption")
            return evict(relative, **kwargs)

        monkeypatch.setattr(runtime.retrieval, "evict_generation", interrupted)
        status_code, body = service.request(
            "/api/workspace/storage/confirm",
            method="POST",
            payload={"storage_plan_hash": plan["plan_hash"]},
        )
        monkeypatch.undo()
        assert (status_code, body.get("status")) != (200, "COMPLETED"), body
        database = workspace / Path(*index_target.split("/"))
        assert not database.is_file() and stray.is_file()
        assert service.get("/api/workspace/storage")["status"] == "RECOVERY_REQUIRED"
        sidecar.unlink()
        assert (
            service.post(
                "/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]}
            ).items()
            >= {"status": "REFUSED", "failure_code": "storage.cleanup_target_protected"}.items()
        )
        assert stray.is_file() and all(path.is_file() for path in blocks)
        sidecar.write_bytes(sidecar_bytes)
        resumed = service.post(
            "/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]}
        )
        assert resumed["status"] == "COMPLETED" and resumed["deleted_paths"] == 1
        assert not stray.is_file()
        assert {path: path.read_bytes() for path in vector_root.iterdir()} == original
        assert service.get("/api/workspace/storage")["status"] == "AVAILABLE"
        (row,) = _index_rows(service)
        assert row["availability"] == "EVICTED_BY_RETENTION" and row["payload_available"] is True
    finally:
        service.session.stop()


def test_a_cleaned_legacy_index_offers_no_impossible_rebuild(tmp_path: Path) -> None:
    """A historical index without payload commitments has no rebuild action."""
    from alphalattice.evidence.alternative_evidence.contracts import (
        AlternativeEvidenceReadingDepth,
        seal_contract,
    )
    from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
        HybridV3RetrievalGeneration,
    )

    workspace, report = build_workspace(tmp_path)
    publish_research_workspace_manifest(workspace, _workspace_manifest("legacy-index-storage"))
    authority = _authority(workspace, report)
    runtime = authority.evidence_runtime
    assert runtime is not None
    generation = seal_contract(
        HybridV3RetrievalGeneration,
        "generation_hash",
        document_set_hash="1" * 64,
        workspace_snapshot_id="synthetic-legacy-snapshot",
        workspace_snapshot_hash="2" * 64,
        retrieval_binding_hash="3" * 64,
        reading_depth=AlternativeEvidenceReadingDepth.FULL_FILING,
        index_id="4" * 64,
        index_spec_hash=runtime.retrieval.index_spec.logical_hash,
        document_count=1,
        chunk_count=1,
        built_at=_NOW,
    )
    runtime.artifacts.publish("retrieval-generations", generation.generation_hash, generation)
    database = (
        runtime.retrieval.workspace.root
        / ".system/knowledge-indexes/knowledge-hybrid-v3"
        / f"{generation.workspace_snapshot_hash}-{generation.index_spec_hash}.db"
    )
    database.parent.mkdir(parents=True, exist_ok=True)

    from alphalattice.kernel.knowledge.hybrid_contracts import HybridV3KnowledgeIndexManifest
    from alphalattice.kernel.shared_kernel.domain.serialization import (
        canonical_json_bytes,
        sha256_hex,
    )

    values = {
        "index_id": generation.index_id,
        "database_path": database.relative_to(runtime.retrieval.workspace.root).as_posix(),
        "snapshot_id": uuid4(),
        "snapshot_logical_hash": generation.workspace_snapshot_hash,
        "index_spec": runtime.retrieval.index_spec,
        "document_count": 1,
        "chunk_count": 1,
        "ordered_chunk_ids": ("6" * 64,),
        "ordered_citation_hashes": ("7" * 64,),
        "ordered_embedding_input_hashes": ("8" * 64,),
        "projection_logical_hash": "9" * 64,
    }
    draft = HybridV3KnowledgeIndexManifest.model_construct(**values, logical_hash="0" * 64)
    values = draft.model_dump(exclude={"logical_hash"})
    manifest = HybridV3KnowledgeIndexManifest.model_validate(
        {**values, "logical_hash": sha256_hex(canonical_json_bytes(values))}
    )
    # A valid historical manifest lets the real retention owner clean the index.
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute(
            "INSERT INTO metadata VALUES ('manifest', ?)",
            (canonical_json_bytes(manifest).decode("utf-8"),),
        )
    connection.close()
    service = start_service(workspace, authority, tmp_path, clock=lambda: _NOW)
    try:
        (available,) = _index_rows(service)
        assert available["generation_format"] == "knowledge-hybrid-v3"
        assert available["available_actions"] == ["PIN"]
        plan = service.post("/api/workspace/storage/plan")
        assert available["relative_path"] in plan["targets"]
        cleaned = service.post(
            "/api/workspace/storage/confirm", {"storage_plan_hash": plan["plan_hash"]}
        )
        assert cleaned["status"] == "COMPLETED" and not database.exists(), cleaned
        (evicted,) = _index_rows(service)
        assert evicted["availability"] == "EVICTED_BY_RETENTION"
        status, refused = service.request(
            "/api/workspace/storage/evidence-rebuild",
            method="POST",
            payload={"evidence_index_id": generation.index_id},
        )
        assert status == 400
        assert "retrieval_generation_legacy_not_rebuildable" in refused["refused"]
        assert evicted["available_actions"] == [], evicted
        assert "new Evidence preparation" in evicted["rebuild_limit"]
    finally:
        service.session.stop()
