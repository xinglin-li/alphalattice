"""The Host's sealing of the analysis, the eight-stage task and its recovery.

The Host seals, publishes and replays the analysis, refuses uncited, unscoped
and off-question briefs and forged policy bindings, agent configuration moves
provenance but never Host policy, the task runs eight stages without a model
or the network, recovery at the analysis stage keeps the index and publishes
once, and coverage and obligation checks block before downstream work.
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    resolve_task_control_database,
)
from alphalattice.control.task_control.runner import TaskControlRunner
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceAnalystAnswer,
    EvidenceStructureState,
    seal_research_obligation,
)
from alphalattice.evidence.alternative_evidence.analysis.cro_package import (
    compile_cro_alternative_evidence_package,
)
from alphalattice.evidence.alternative_evidence.analysis.packet import span_aliases
from alphalattice.evidence.alternative_evidence.analysis.submissions import (
    AlternativeEvidenceBriefAuthorityError,
    seal_alternative_evidence_analyst_brief,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceAdmission,
    AlternativeEvidenceRequest,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    AlternativeEvidencePublicationError,
)
from alphalattice.evidence.alternative_evidence.publication.contracts import (
    AlternativeEvidenceAnalysisPublication,
)
from alphalattice.evidence.alternative_evidence.runtime.service import (
    AlternativeEvidenceDocumentIntelligenceRuntime,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    AlternativeEvidenceDocumentTaskAdapter,
    AlternativeEvidenceDocumentTaskResources,
    SubmittedEvidenceAnalysis,
    alternative_evidence_document_task_contract,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution import ActorKind, seal_actor_submission
from tests.alternative_evidence_desk.document_intelligence_support import (
    _obligation,
    _open_recorded,
    _registry,
    _request,
)
from tests.alternative_evidence_desk.planted_corpus import (
    _NOW,
    PLAYPEN_ROOT,
    _CitingActor,
    _recorded_document,
    _runtime,
)


def _admission(
    request: AlternativeEvidenceRequest, *, model_review: bool = False
) -> AlternativeEvidenceAdmission:
    return seal_contract(
        AlternativeEvidenceAdmission,
        "admission_hash",
        request_hash=request.request_hash,
        network_consent=False,
        admit_live_official=False,
        admit_model_review=model_review,
        admitted_at=_NOW,
    )


def test_the_host_seals_publishes_and_replays_the_analysis(tmp_path: Path) -> None:
    """Brief -> CRO package -> publication -> exact replay, tamper-evident."""

    runtime, request, registry, snapshot, document_set, generation = _open_recorded(tmp_path)
    receipt, spans = runtime.select_evidence(
        request=request, document_set=document_set, generation=generation
    )
    obligation = _obligation(request)
    actor = _CitingActor()
    result = runtime.analyze(
        actor=actor,
        request=request,
        obligation=obligation,
        snapshot=snapshot,
        document_set=document_set,
        generation=generation,
        access_receipt=receipt,
        resolved_spans=spans,
        completed_at=_NOW,
    )
    assert len(actor.packets_seen) == 1
    assert result.analyst_receipt.decision_policy_hash == runtime.decision_policy.binding_hash
    assert result.brief.obligation_hash == obligation.obligation_hash

    package = compile_cro_alternative_evidence_package(
        request=request,
        snapshot=snapshot,
        document_set=document_set,
        generation=generation,
        access_receipt=receipt,
        brief=result.brief,
        resolved_spans=spans,
    )
    handle = result.brief.findings[0].finding_handle
    assert package.structure(handle).state is EvidenceStructureState.SINGLE_SOURCE
    assert package.structure(handle).supporting_document_count == 1
    # The one current report is what the window holds: a topic it does not
    # serve is no gap (no periodic report is a baseline since W1), so the
    # package names nothing missing -- and whatever it names is an issuer and
    # topic line.
    assert package.missing_evidence == ()
    assert all(line.startswith("AAPL ") for line in package.missing_evidence)

    view = runtime.publications.publish(
        request=request,
        registry=registry,
        snapshot=snapshot,
        document_set=document_set,
        generation=generation,
        access_receipt=receipt,
        analyst_receipt=result.analyst_receipt,
        cro_package=package,
        published_at=_NOW,
    )
    publication_hash = view.publication.publication_hash
    assert view.publication.obligation_hash == obligation.obligation_hash

    def files() -> tuple[tuple[Path, int], ...]:
        return tuple(
            (path, path.stat().st_mtime_ns)
            for path in sorted((tmp_path / "artifacts").rglob("*"))
            if path.is_file()
        )

    before = files()
    replay = runtime.publications.replay(publication_hash, now=_NOW + timedelta(minutes=1))
    assert replay.action == "REUSED_EXACT"
    assert replay.is_current
    assert replay.publication == view.publication
    assert replay.lineage.brief == result.brief
    assert files() == before, "replay is a read"

    later = _NOW + timedelta(seconds=request.ttl_seconds + 1)
    expired = runtime.publications.replay(publication_hash, now=later)
    assert not expired.is_current
    assert expired.current_eligibility == "EXPIRED"
    with pytest.raises(AlternativeEvidencePublicationError, match="expired"):
        runtime.publications.read(publication_hash, now=later)

    stored = next((tmp_path / "artifacts").rglob(f"{result.brief.brief_hash}.json"))
    payload = json.loads(stored.read_text(encoding="utf-8"))
    payload["executive_summary"] = "tampered"
    stored.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((AlternativeEvidencePublicationError, ValueError)):
        runtime.publications.verify(publication_hash)


def _sealer_inputs(tmp_path: Path) -> tuple[dict[str, Any], tuple[Any, ...], Any]:
    runtime, request, _registry_value, snapshot, document_set, generation = _open_recorded(tmp_path)
    receipt, spans = runtime.select_evidence(
        request=request, document_set=document_set, generation=generation
    )
    common: dict[str, Any] = {
        "request": request,
        "obligation": _obligation(request),
        "snapshot": snapshot,
        "document_set": document_set,
        "generation": generation,
        "access_receipt": receipt,
        "resolved_spans": spans,
        "analysis_policy": runtime.analysis_policy,
        "decision_policy": runtime.decision_policy,
        "playpen_root": PLAYPEN_ROOT,
        "completed_at": _NOW,
        "actor_kind": ActorKind.HUMAN,
        "actor_id": "researcher@example.test",
    }
    return common, spans, runtime


def _answer(spans: tuple[Any, ...], *, cite: tuple[str, ...], **finding: Any) -> Any:
    """One finding as an Analyst writes it, citing excerpts by their aliases."""

    alias_of = {
        span.span_handle: alias
        for alias, span in span_aliases(
            spans, tuple(dict.fromkeys(s.entity_id for s in spans))
        ).items()
    }
    return AlternativeEvidenceAnalystAnswer.model_validate(
        {
            "findings": [
                {
                    "issuer": "AAPL",
                    "topic": "OPERATIONS_SUPPLY",
                    "direction": "ADVERSE",
                    "summary": "Supplier interruption reduced component capacity.",
                    "cite": [alias_of.get(handle, handle) for handle in cite],
                    **finding,
                }
            ]
        }
    )


def test_the_sealer_refuses_uncited_unscoped_and_off_question_briefs(tmp_path: Path) -> None:
    """The Host, not the actor, decides what a finding may rest on."""

    common, spans, _runtime_value = _sealer_inputs(tmp_path)
    cited = spans[0].span_handle

    accepted = seal_alternative_evidence_analyst_brief(
        **common, answer=_answer(spans, cite=(cited,))
    )
    assert accepted.brief.findings[0].supporting_span_handles == (cited,)

    for answer in (
        _answer(spans, cite=("S99",)),
        _answer(spans, cite=(cited,), issuer="MSFT"),
        _answer(spans, cite=(cited,), contrary=[_answer(spans, cite=(cited,)).findings[0].cite[0]]),
    ):
        with pytest.raises(AlternativeEvidenceBriefAuthorityError, match="answer_invalid"):
            seal_alternative_evidence_analyst_brief(**common, answer=answer)
    with pytest.raises(ValueError, match="too_short"):
        _answer(spans, cite=())

    other = {**common, "obligation": _obligation(_request(("AAPL", "MSFT")))}
    with pytest.raises(AlternativeEvidenceBriefAuthorityError, match="brief_obligation_mismatch"):
        seal_alternative_evidence_analyst_brief(**other, answer=_answer(spans, cite=(cited,)))


def test_forged_host_policy_bindings_are_refused(tmp_path: Path) -> None:
    common, spans, runtime = _sealer_inputs(tmp_path)
    answer = _answer(spans, cite=(spans[0].span_handle,))

    def _forge(binding: Any) -> Any:
        values = {
            **binding.model_dump(exclude={"binding_hash"}),
            "validation_source_hash": "b" * 64,
        }
        return type(binding)(**values, binding_hash=str(canonical_hash(values)))

    forged_analysis = _forge(runtime.analysis_policy)
    forged_decision = _forge(runtime.decision_policy)
    assert type(forged_analysis).model_validate(forged_analysis) == forged_analysis
    for analysis, decision in (
        (forged_analysis, runtime.decision_policy),
        (runtime.analysis_policy, forged_decision),
    ):
        values = {**common, "analysis_policy": analysis, "decision_policy": decision}
        with pytest.raises(AlternativeEvidenceBriefAuthorityError, match="policy_unauthorized"):
            seal_alternative_evidence_analyst_brief(**values, answer=answer)


EVIDENCE_STAGES: tuple[str, ...] = (
    "admit_evidence_request",
    "resolve_official_sources",
    "acquire_source_evidence",
    "canonicalize_documents",
    "build_retrieval_generation",
    "select_evidence_spans",
    "analyze_evidence",
    "publish_evidence_analysis",
)


def _admitted_task(
    tmp_path: Path,
    runtime: AlternativeEvidenceDocumentIntelligenceRuntime,
    *,
    request: AlternativeEvidenceRequest,
    resources: AlternativeEvidenceDocumentTaskResources,
    prepare_only: bool = False,
) -> tuple[DuckDbTaskControlRegistry, Any, AlternativeEvidenceDocumentTaskAdapter]:
    envelope, goal, plan = alternative_evidence_document_task_contract(
        request=request,
        admission=_admission(request),
        obligation=_obligation(request),
        resource_binding_hash=resources.preparation_binding_hash
        if prepare_only
        else resources.binding_hash,
        prepare_only=prepare_only,
    )
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    admitted = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=_NOW)
    adapter = AlternativeEvidenceDocumentTaskAdapter(
        runtime=runtime, registry=registry, resources=resources
    )
    return registry, admitted, adapter


def test_the_task_runs_eight_stages_without_a_model_or_the_network(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    request = _request()
    actor = _CitingActor()
    registry, admitted, adapter = _admitted_task(
        tmp_path,
        runtime,
        request=request,
        resources=AlternativeEvidenceDocumentTaskResources(
            recorded_registry=_registry(),
            recorded_documents=(_recorded_document(),),
            analysis_actor=actor,
        ),
    )
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-checkpoints.sqlite"),
        clock=lambda: _NOW,
        heartbeat_seconds=0.01,
    )
    try:
        completed = runner.run_next()
    finally:
        runner.close()
    assert completed is not None
    assert completed.lifecycle is TaskLifecycle.SUCCEEDED, (
        completed.failure_code,
        getattr(completed, "failure_detail", None),
    )
    task_id = admitted.record.task_id
    safe = registry.safe_projection(task_id)
    assert safe.verified_stage_count == safe.total_stage_count == len(EVIDENCE_STAGES)
    receipts = registry.stage_receipts(task_id)
    assert len(receipts) == len(EVIDENCE_STAGES)
    assert {value.stage_id for value in receipts} == set(EVIDENCE_STAGES)
    publication_receipt = next(
        value for value in receipts if value.stage_id == "publish_evidence_analysis"
    )
    view = runtime.publications.read(
        publication_receipt.evidence[0].content_hash, now=_NOW + timedelta(minutes=1)
    )
    assert view.action == "READBACK"
    assert view.publication.obligation_hash == _obligation(request).obligation_hash
    assert len(actor.packets_seen) == 1
    assert runtime.retrieval.passage_embedding_pass_count == 1


def test_external_analysis_reopens_preparation_and_recovers_without_another_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    resources = AlternativeEvidenceDocumentTaskResources(
        recorded_registry=_registry(),
        recorded_documents=(_recorded_document(),),
    )
    request = _request()
    registry, admitted, adapter = _admitted_task(
        tmp_path,
        runtime,
        request=request,
        resources=resources,
        prepare_only=True,
    )
    checkpoint = str(tmp_path / "external-task-checkpoints.sqlite")
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=checkpoint,
        clock=lambda: _NOW,
    )
    try:
        completed = runner.run_next()
    finally:
        runner.close()
    assert completed is not None and completed.lifecycle is TaskLifecycle.SUCCEEDED
    assert len(registry.stage_receipts(completed.task_id)) == 6
    assert not (runtime.artifacts.root / "analyst-briefs").exists()
    packet = adapter.prepared_packet(completed.task_id, now=_NOW)
    assert runtime.retrieval.passage_embedding_pass_count == 1
    assert (
        resources.preparation_binding_hash
        == replace(resources, analysis_actor=_CitingActor()).preparation_binding_hash
    )
    runtime.close()

    # The new runtime opens saved packet bytes only, not an index or a model.
    runtime = _runtime(tmp_path)
    adapter = AlternativeEvidenceDocumentTaskAdapter(
        runtime=runtime, registry=registry, resources=resources
    )
    assert adapter.prepared_packet(completed.task_id, now=_NOW) == packet
    with pytest.raises(ValueError, match="brief_source_stale"):
        adapter.prepared_packet(
            completed.task_id, now=packet.snapshot.expires_at + timedelta(seconds=1)
        )
    answer = _CitingActor()(packet=packet).answer
    _, context = adapter.analysis_context(completed.task_id, now=_NOW)
    submitted = SubmittedEvidenceAnalysis(
        prepared_task_id=completed.task_id,
        packet_hash=context["packet_hash"],
        analysis_policy_hash=runtime.analysis_policy_hash,
        decision_policy_hash=runtime.decision_policy.binding_hash,
        answer=answer,
        actor_submission=seal_actor_submission(
            actor_kind=ActorKind.EXTERNAL_AUTOMATION,
            actor_id="external-researcher",
            submission_hash=canonical_hash(answer.model_dump(mode="json")),
        ),
    )
    for field in ("packet_hash", "analysis_policy_hash", "decision_policy_hash"):
        with pytest.raises(ValueError, match="external_analysis_binding_changed"):
            adapter.validate_submission(submitted.model_copy(update={field: "0" * 64}), now=_NOW)
    adapter.validate_submission(submitted, now=_NOW)
    contract = alternative_evidence_document_task_contract(
        request=request,
        admission=_admission(request),
        obligation=_obligation(request),
        resource_binding_hash=resources.preparation_binding_hash,
        submitted_analysis=submitted,
    )
    envelope, goal, plan = contract
    accepted = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=_NOW)
    assert accepted.record.task_id != admitted.record.task_id
    assert tuple(item.stage_id for item in plan.work_items) == EVIDENCE_STAGES[6:]
    original_verify = adapter.verify_stage
    interruptions: list[str] = []

    def interrupted(**kwargs: Any) -> Any:
        if kwargs["work_item"].stage_id == "analyze_evidence" and not interruptions:
            interruptions.append("analyze_evidence")
            raise RuntimeError("lost process after external analysis was sealed")
        return original_verify(**kwargs)

    monkeypatch.setattr(adapter, "verify_stage", interrupted)
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=checkpoint,
        clock=lambda: _NOW,
    )
    try:
        with suppress(RuntimeError):
            runner.run_next()
        assert registry.task(accepted.record.task_id).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        runner.recover(accepted.record.task_id)
    finally:
        runner.close()
    assert registry.task(accepted.record.task_id).lifecycle is TaskLifecycle.SUCCEEDED
    assert runtime.retrieval.passage_embedding_pass_count == 0
    publication = registry.stage_receipts(accepted.record.task_id)[-1].evidence[0]
    view = runtime.publications.read(publication.content_hash, now=_NOW)
    assert view.lineage.analyst_receipt.actor_submission == submitted.actor_submission
    assert view.lineage.analyst_receipt.model_call_count == 0
    assert (
        len(
            runtime.artifacts.values(
                "analysis-publications", AlternativeEvidenceAnalysisPublication
            )
        )
        == 1
    )
    assert adapter.published_analysis(accepted.record.task_id, now=_NOW) == view
    assert adapter.prepared_packet(completed.task_id, now=_NOW) == packet
    runtime.close()


def test_a_tampered_workspace_document_is_a_named_packet_refusal(tmp_path: Path) -> None:
    """The kernel's own code travels in the refusal; the restored blob reads again."""

    runtime = _runtime(tmp_path)
    resources = AlternativeEvidenceDocumentTaskResources(
        recorded_registry=_registry(), recorded_documents=(_recorded_document(),)
    )
    registry, _admitted, adapter = _admitted_task(
        tmp_path, runtime, request=_request(), resources=resources, prepare_only=True
    )
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "checkpoints.sqlite"),
        clock=lambda: _NOW,
    )
    try:
        completed = runner.run_next()
    finally:
        runner.close()
    assert completed is not None and completed.lifecycle is TaskLifecycle.SUCCEEDED
    packet = adapter.prepared_packet(completed.task_id, now=_NOW)
    reference = packet.document_set.documents[0]
    # The canonical blob the packet reads, not the original bytes the source
    # store keeps beside it (a recorded markdown source has the same size).
    blob = next(
        path
        for path in (tmp_path / "workspace" / "knowledge" / "blobs").rglob("*")
        if path.is_file() and path.stat().st_size == reference.byte_count
    )
    original = blob.read_bytes()
    blob.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
    try:
        with pytest.raises(ValueError, match="brief_document_unverifiable:retrieval\\."):
            adapter.prepared_packet(completed.task_id, now=_NOW)
    finally:
        blob.write_bytes(original)
    assert adapter.prepared_packet(completed.task_id, now=_NOW) == packet
    runtime.close()


def test_a_stage_failure_names_the_owners_own_code() -> None:
    """A knowledge failure carries its code; a contract refusal its message; the rest stays bare."""

    from pydantic import BaseModel, ValidationError, model_validator

    from alphalattice.control.task_control.contracts import FAILURE_CODE_MAX_LENGTH
    from alphalattice.evidence.alternative_evidence.runtime.task_adapter import _failure_code
    from alphalattice.kernel.knowledge.hybrid import _retrieval_error
    from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError

    corrupt = _retrieval_error(
        "hybrid projection differs from source", code="retrieval.index_corrupt"
    )
    assert isinstance(corrupt, KnowledgeRetrievalError)
    assert _failure_code(corrupt) == "alternative_evidence.task_failed:retrieval.index_corrupt"

    class _Refusing(BaseModel):
        value: str

        @model_validator(mode="after")
        def refuse(self) -> _Refusing:
            raise ValueError("alternative_evidence.identity_invalid")

    with pytest.raises(ValidationError) as caught:
        _Refusing(value="tampered")
    assert _failure_code(caught.value) == "alternative_evidence.identity_invalid"
    assert _failure_code(ValueError("alternative_evidence.stage_lineage_missing")) == (
        "alternative_evidence.stage_lineage_missing"
    )
    assert _failure_code(RuntimeError("lost process")) == "alternative_evidence.task_failed"
    assert len(_failure_code(_retrieval_error("x" * 300, code="retrieval." + "y" * 200))) <= (
        FAILURE_CODE_MAX_LENGTH
    )


def test_recovery_at_the_analysis_stage_keeps_the_index_and_publishes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost process owes the unverified stage, not the whole Task."""

    runtime = _runtime(tmp_path)
    request = _request()
    actor = _CitingActor()
    registry, admitted, adapter = _admitted_task(
        tmp_path,
        runtime,
        request=request,
        resources=AlternativeEvidenceDocumentTaskResources(
            recorded_registry=_registry(),
            recorded_documents=(_recorded_document(),),
            analysis_actor=actor,
        ),
    )
    original = AlternativeEvidenceDocumentTaskAdapter.verify_stage
    interruptions: list[str] = []

    def _interrupt(self: Any, *args: Any, **kwargs: Any) -> Any:
        stage = kwargs["work_item"].stage_id
        if stage == "analyze_evidence" and not interruptions:
            interruptions.append(str(stage))
            raise RuntimeError("simulated interruption inside the analysis stage")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AlternativeEvidenceDocumentTaskAdapter, "verify_stage", _interrupt)
    task_id = admitted.record.task_id
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-checkpoints.sqlite"),
        clock=lambda: _NOW,
        heartbeat_seconds=0.01,
    )
    try:
        with suppress(RuntimeError):
            runner.run_next()
        assert registry.task(task_id).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        assert runtime.retrieval.passage_embedding_pass_count == 1
        runner.recover(task_id)
    finally:
        runner.close()

    assert interruptions == ["analyze_evidence"]
    assert registry.task(task_id).lifecycle is TaskLifecycle.SUCCEEDED
    assert runtime.retrieval.passage_embedding_pass_count == 1, "the generation was preserved"
    publications = runtime.artifacts.values(
        "analysis-publications", AlternativeEvidenceAnalysisPublication
    )
    assert len(publications) == 1


def test_minimum_entity_coverage_blocks_before_downstream_work(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    request = _request()
    registry, admitted, adapter = _admitted_task(
        tmp_path,
        runtime,
        request=request,
        resources=AlternativeEvidenceDocumentTaskResources(
            recorded_registry=_registry(),
            recorded_documents=(),
            minimum_entity_coverage=1.0,
        ),
    )
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "coverage-rejection.sqlite"),
        clock=lambda: _NOW,
    )
    try:
        completed = runner.run_next()
    finally:
        runner.close()
    assert completed is not None and completed.lifecycle is TaskLifecycle.BLOCKED
    assert completed.failure_code == (
        "alternative_evidence.minimum_entity_coverage_not_met:"
        "0 of 1 issuers hold a source, 1 needed"
    )
    assert tuple(value.stage_id for value in registry.stage_receipts(admitted.record.task_id)) == (
        "admit_evidence_request",
        "resolve_official_sources",
    )
    for category in ("document-sets", "retrieval-generations", "analyst-briefs"):
        assert not (runtime.artifacts.root / category).exists()


def test_an_obligation_that_disagrees_with_the_request_is_refused_before_a_task() -> None:
    request = _request()
    admission = _admission(request)
    for obligation, code in (
        (_obligation(_request(("AAPL", "MSFT"))), "task_obligation_axis_mismatch"),
        (
            seal_research_obligation(
                **{
                    **_obligation(request).model_dump(exclude={"obligation_hash"}),
                    "evidence_as_of": _NOW - timedelta(days=1),
                }
            ),
            "task_obligation_cutoff_mismatch",
        ),
        (
            _obligation(request, approved_source_families=("ISSUER_WEBSITE",)),
            "task_obligation_sources_unadmitted",
        ),
    ):
        with pytest.raises(ValueError, match=code):
            alternative_evidence_document_task_contract(
                request=request,
                admission=admission,
                obligation=obligation,
                resource_binding_hash="0" * 64,
            )
