from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import httpx
import pytest

from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceAdmission,
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
    SecCompanyFactPoint,
    SecCompanyFactsSnapshot,
    SecIssuerRegistryEntry,
    SecIssuerRegistrySnapshot,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.sources.acquisition import (
    AlternativeEvidenceAcquisitionService,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument
from alphalattice.evidence.alternative_evidence.sources.sec_edgar import (
    HttpxSecOfficialTransport,
    SecEdgarSource,
    SecOfficialResponse,
)
from tests.alternative_evidence_desk.sec_fixture_transport import NOW, SecFixtureTransport


def _request(
    *,
    mode: AlternativeEvidenceMode = AlternativeEvidenceMode.RECORDED,
    evidence_classes: tuple[str, ...] = ("SEC_FILING", "SEC_COMPANYFACTS"),
) -> AlternativeEvidenceRequest:
    return seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        ordered_entity_ids=("AAPL", "MSFT"),
        evidence_as_of=NOW,
        acquisition_deadline=NOW + timedelta(minutes=15),
        evidence_classes=tuple(AlternativeEvidenceClass(value) for value in evidence_classes),
        source_policy=AlternativeEvidenceSourcePolicy(),
        ttl_seconds=3600,
        mode=mode,
    )


def _registry() -> SecIssuerRegistrySnapshot:
    return seal_contract(
        SecIssuerRegistrySnapshot,
        "registry_hash",
        captured_at=NOW,
        entries=(
            SecIssuerRegistryEntry(
                entity_id="AAPL", ticker="AAPL", cik="0000320193", legal_name="Apple Inc."
            ),
            SecIssuerRegistryEntry(
                entity_id="MSFT", ticker="MSFT", cik="0000789019", legal_name="Microsoft"
            ),
        ),
        source_content_hash="1" * 64,
    )


def _document(entity_id: str, *, available_at: datetime = NOW) -> RecordedEvidenceDocument:
    return RecordedEvidenceDocument(
        entity_id=entity_id,
        source_right="USER_PROVIDED_FOR_LOCAL_RESEARCH",
        evidence_class="SEC_FILING",
        document_type="10-Q",
        revision=f"{entity_id}-2026-Q2",
        published_at=NOW - timedelta(days=3),
        captured_at=available_at,
        available_at=available_at,
        text="Revenue growth slowed.",
        immutable_source=True,
        limitations=("Recorded test fixture.",),
    )


def test_recorded_acquisition_freezes_source_evidence_only(tmp_path: Path) -> None:
    service = AlternativeEvidenceAcquisitionService(artifact_root=tmp_path)
    request = _request()
    snapshot, documents = service.build_recorded_evidence(
        request=request,
        registry=_registry(),
        documents=(_document("AAPL"), _document("MSFT")),
        published_at=NOW,
    )

    assert snapshot.status == "COMPLETE"
    assert snapshot.available_source_count == 2
    assert len(documents) == 2
    assert not (tmp_path / "alternative-evidence" / "current" / "active.json").exists()


def test_recorded_material_beyond_an_issuers_capacity_is_deferred_and_named(
    tmp_path: Path,
) -> None:
    """Recorded material beyond an issuer's capacity is deferred and named."""

    service = AlternativeEvidenceAcquisitionService(artifact_root=tmp_path)
    request = seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        ordered_entity_ids=tuple(f"I{index}" for index in range(8)),
        evidence_as_of=NOW,
        acquisition_deadline=NOW + timedelta(minutes=15),
        evidence_classes=(AlternativeEvidenceClass.SEC_FILING,),
        source_policy=AlternativeEvidenceSourcePolicy(maximum_documents_per_issuer=4),
        ttl_seconds=3600,
        mode=AlternativeEvidenceMode.RECORDED,
    )
    library = (
        *(
            _document("I0", available_at=NOW - timedelta(days=days)).model_copy(
                update={"document_type": kind, "revision": f"I0-{kind}-{days}"}
            )
            for kind, days in (
                ("8-K", 1),
                ("8-K", 2),
                ("10-Q", 40),
                ("10-K", 200),
                ("10-Q", 130),
                ("8-K", 3),
            )
        ),
        _document("I1"),
    )
    registry = seal_contract(
        SecIssuerRegistrySnapshot,
        "registry_hash",
        captured_at=NOW - timedelta(days=1),
        entries=tuple(
            SecIssuerRegistryEntry(
                entity_id=f"I{index}", ticker=f"I{index}", cik=f"{index:010d}", legal_name="x"
            )
            for index in range(8)
        ),
        source_content_hash="1" * 64,
    )
    snapshot, documents = service.build_recorded_evidence(
        request=request, registry=registry, documents=library, published_at=NOW
    )
    kept = [value.revision for value in documents if value.entity_id == "I0"]
    # The policy budget of four is the issuer's own capacity, eight issuers
    # or one: the newest 10-K and 10-Q come first, then the two newest 8-Ks;
    # the older 10-Q and the oldest 8-K are deferred beyond the budget.
    assert kept == ["I0-8-K-1", "I0-8-K-2", "I0-10-Q-40", "I0-10-K-200"]
    deferral = next(value for value in snapshot.limitations if value.startswith("I0:"))
    assert "2 recorded document(s) deferred beyond the policy budget of 4" in deferral
    assert "10-Q I0-10-Q-130" in deferral and "8-K I0-8-K-3" in deferral
    assert "admitted-set share" not in deferral
    assert [value.revision for value in documents if value.entity_id == "I1"] == ["I1-2026-Q2"]
    alone = seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        **request.model_dump(exclude={"request_hash", "ordered_entity_ids"}),
        ordered_entity_ids=("I0",),
    )
    _snapshot, alone_documents = service.build_recorded_evidence(
        request=alone, registry=registry, documents=library, published_at=NOW
    )
    assert [value.revision for value in alone_documents] == kept, "the same selection alone"
    # The admitted set bounds the unit's selections together: eight issuers
    # of four documents each are 32, so eight non-baseline documents are
    # deferred round-robin, oldest first, and named for the unit.
    dense_library = tuple(
        _document(f"I{index}", available_at=NOW - timedelta(days=days)).model_copy(
            update={"document_type": kind, "revision": f"I{index}-{kind}-{days}"}
        )
        for index in range(8)
        for kind, days in (("8-K", 1), ("8-K", 2), ("10-Q", 40), ("10-K", 200))
    )
    dense_snapshot, dense_documents = service.build_recorded_evidence(
        request=request, registry=registry, documents=dense_library, published_at=NOW
    )
    assert len(dense_documents) == 24
    assert all(
        sorted(v.revision for v in dense_documents if v.entity_id == f"I{index}")
        == [f"I{index}-10-K-200", f"I{index}-10-Q-40", f"I{index}-8-K-1"]
        for index in range(8)
    ), "every issuer keeps its baselines and its newest event; the oldest event yields"
    unit_deferrals = [
        value
        for value in dense_snapshot.limitations
        if "beyond the admitted document set's capacity of 24 for this unit of 8 issuers" in value
    ]
    assert len(unit_deferrals) == 8 and all("8-K-2" in value for value in unit_deferrals)


class _PartiallyFailingSecTransport(SecFixtureTransport):
    def get(self, url: str, *, maximum_bytes: int) -> SecOfficialResponse:
        if "CIK0000789019" in url or "/789019/" in url:
            raise RuntimeError("fixture issuer source unavailable")
        return super().get(url, maximum_bytes=maximum_bytes)


def _live_admission(request: AlternativeEvidenceRequest) -> AlternativeEvidenceAdmission:
    return seal_contract(
        AlternativeEvidenceAdmission,
        "admission_hash",
        request_hash=request.request_hash,
        network_consent=True,
        admit_live_official=True,
        admit_model_review=False,
        admitted_at=NOW,
    )


def test_sec_acquisition_requires_explicit_consent_and_preserves_cutoff(tmp_path: Path) -> None:
    request = _request(mode=AlternativeEvidenceMode.LIVE_OFFICIAL)
    service = AlternativeEvidenceAcquisitionService(artifact_root=tmp_path)
    denied = seal_contract(
        AlternativeEvidenceAdmission,
        "admission_hash",
        request_hash=request.request_hash,
        network_consent=False,
        admit_live_official=False,
        admit_model_review=False,
        admitted_at=NOW,
    )
    with pytest.raises(ValueError, match=r"alternative_evidence\.live_admission_invalid"):
        service.build_live_evidence(
            request=request,
            admission=denied,
            source=SecEdgarSource(SecFixtureTransport()),
            published_at=NOW,
            clock=lambda: NOW,
        )

    registry, snapshot, _documents = service.build_live_evidence(
        request=request,
        admission=_live_admission(request),
        source=SecEdgarSource(SecFixtureTransport()),
        published_at=NOW,
        clock=lambda: NOW,
    )
    assert {value.cik for value in registry.entries} == {"0000320193", "0000789019"}
    assert snapshot.status == "COMPLETE"
    assert all(value.accepted_at <= request.evidence_as_of for value in snapshot.citations)

    _, partial, _ = service.build_live_evidence(
        request=request,
        admission=_live_admission(request),
        source=SecEdgarSource(_PartiallyFailingSecTransport()),
        registry=registry,
        published_at=NOW,
        clock=lambda: NOW,
    )
    assert partial.status == "PARTIAL"
    assert partial.available_source_count == partial.failed_source_count == 1


def test_live_calls_follow_the_admitted_evidence_classes(tmp_path: Path) -> None:
    service = AlternativeEvidenceAcquisitionService(artifact_root=tmp_path)
    filings = _request(
        mode=AlternativeEvidenceMode.LIVE_OFFICIAL,
        evidence_classes=("SEC_FILING",),
    )
    filings_transport = SecFixtureTransport()
    service.build_live_evidence(
        request=filings,
        admission=_live_admission(filings),
        source=SecEdgarSource(filings_transport),
        published_at=NOW,
        clock=lambda: NOW,
    )
    assert not any("/companyfacts/" in value for value in filings_transport.calls)

    facts = _request(
        mode=AlternativeEvidenceMode.LIVE_OFFICIAL,
        evidence_classes=("SEC_COMPANYFACTS",),
    )
    facts_transport = SecFixtureTransport()
    service.build_live_evidence(
        request=facts,
        admission=_live_admission(facts),
        source=SecEdgarSource(facts_transport),
        published_at=NOW,
        clock=lambda: NOW,
    )
    assert not any("/submissions/" in value for value in facts_transport.calls)
    assert any("/companyfacts/" in value for value in facts_transport.calls)


def test_http_transport_enforces_retry_size_failure_and_official_boundaries() -> None:
    calls = 0
    clock = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"content-type": "application/json", "retry-after": "0.75"},
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b'{"ok":true}',
        )

    retrying = HttpxSecOfficialTransport(
        user_agent="AlphaLattice research@example.com",
        maximum_attempts=2,
        transport=httpx.MockTransport(handler),
        monotonic=lambda: clock[0],
        sleeper=sleep,
    )
    response = retrying.get(
        "https://data.sec.gov/submissions/CIK0000320193.json",
        maximum_bytes=100,
    )
    assert response.status_code == 200 and calls == 2
    assert retrying.request_count == 2
    assert retrying.retry_count == retrying.rate_limited_count == 1
    assert retrying.response_bytes == len(b'{"ok":true}')
    assert sleeps[0] == 0.75
    assert any(value >= 0.2 for value in sleeps)

    wrong_mime = HttpxSecOfficialTransport(
        user_agent="AlphaLattice research@example.com",
        maximum_attempts=1,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "image/png"}, content=b"png"
            )
        ),
    )
    with pytest.raises(ValueError, match="sec_mime_invalid"):
        wrong_mime.get("https://www.sec.gov/files/company_tickers.json", maximum_bytes=10)

    declared_too_large = HttpxSecOfficialTransport(
        user_agent="AlphaLattice research@example.com",
        maximum_attempts=1,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    "content-length": "11",
                },
                content=b"01234567890",
            )
        ),
    )
    with pytest.raises(ValueError, match="sec_response_too_large"):
        declared_too_large.get(
            "https://www.sec.gov/files/company_tickers.json",
            maximum_bytes=10,
        )

    streamed_too_large = HttpxSecOfficialTransport(
        user_agent="AlphaLattice research@example.com",
        maximum_attempts=1,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=b"01234567890",
            )
        ),
    )
    with pytest.raises(ValueError, match="sec_response_too_large"):
        streamed_too_large.get(
            "https://www.sec.gov/files/company_tickers.json",
            maximum_bytes=10,
        )

    timeout = HttpxSecOfficialTransport(
        user_agent="AlphaLattice research@example.com",
        maximum_attempts=1,
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(
                httpx.ReadTimeout("fixture timeout", request=request)
            )
        ),
    )
    with pytest.raises(RuntimeError, match="sec_acquisition_failed"):
        timeout.get("https://www.sec.gov/files/company_tickers.json", maximum_bytes=100)
    with pytest.raises(ValueError, match="non_official_url_rejected"):
        retrying.get("https://example.com/not-sec", maximum_bytes=100)

    for transport in (
        retrying,
        wrong_mime,
        declared_too_large,
        streamed_too_large,
        timeout,
    ):
        transport.close()
        assert transport.closed


def test_mutable_companyfacts_requires_prior_as_of_snapshot() -> None:
    request = _request(mode=AlternativeEvidenceMode.LIVE_OFFICIAL)
    future = seal_contract(
        SecCompanyFactsSnapshot,
        "companyfacts_hash",
        entity_id="AAPL",
        cik="0000320193",
        captured_at=NOW + timedelta(seconds=1),
        facts=(
            SecCompanyFactPoint(
                concept="Assets",
                unit="USD",
                value=10.0,
                period_end="2026-06-30",
                filed_on="2026-08-01",
                form="10-Q",
                accession="0000320193-26-000001",
            ),
        ),
        source_content_hash="a" * 64,
    )
    with pytest.raises(ValueError, match="mutable_companyfacts_historical_rejected"):
        SecEdgarSource.companyfacts_citation(
            request=request,
            snapshot=future,
            semantic_handle="CIT-AAPL-001",
        )


def test_the_writer_passes_only_from_its_holder_and_a_routing_commits_its_own() -> None:
    """A writer transfers only from its current holder and each routing commits under its own
    ownership."""

    from alphalattice.evidence.alternative_evidence.runtime.reuse import SealedComparisons
    from alphalattice.evidence.alternative_evidence.runtime.service import EvidenceWriter

    writer = EvidenceWriter()
    entered: list[str] = []
    waiting = Thread(target=lambda: _enter(writer, entered))
    with writer.held():
        with writer.held():
            pass
        waiting.start()
        waiting.join(timeout=0.2)
        assert waiting.is_alive() and entered == [], "another stage waits for the writer"
        with writer.released():
            waiting.join(timeout=5.0)
            assert entered == ["entered"]
    with writer.released():
        pass  # not its holder: nothing to give up

    held: list[str] = []

    @contextmanager
    def holding() -> Iterator[None]:
        held.append("held")
        yield

    class _Store:
        def __init__(self) -> None:
            self.placed: list[list[str]] = []

        def holds(self, *_values: Any) -> bool:
            return False

        def place(self, members: Any, *, admit: Any) -> int:
            self.placed.append([name for _category, name, _record in members])
            return len(members)

    store = _Store()
    memo = SealedComparisons(store, held=holding)  # type: ignore[arg-type]
    memo.seal(SimpleNamespace(record_hash="a" * 64))  # type: ignore[arg-type]
    other = Thread(target=lambda: memo.seal(SimpleNamespace(record_hash="b" * 64)))  # type: ignore[arg-type]
    other.start()
    other.join()
    assert memo.commit() == frozenset() and store.placed == [["a" * 64]]
    assert held == ["held"]


def test_the_cpu_budget_splits_between_units_at_once_and_threads(tmp_path: Path) -> None:
    """The CPU budget splits between units at once and threads."""

    from alphalattice.evidence.alternative_evidence.runtime.execution import (
        CpuBudget,
        CpuBudgetStore,
        MachineLoad,
        parse_cpu_budget,
        plan_execution,
    )

    now = datetime(2026, 9, 25, tzinfo=UTC)

    def machine(busy: float | None, running: int = 0) -> MachineLoad:
        return MachineLoad(
            processors=32,
            busy_processors=busy,
            total_memory_bytes=1,
            available_memory_bytes=1,
            preparations_running=running,
        )

    auto = plan_execution(CpuBudget(), machine(6.4), units=12, task_id=None, planned_at=now)
    assert (auto.cores, auto.units_at_once, auto.threads_per_session) == (25, 6, 4)
    assert auto.threads_first_unit == 16 and auto.reason.startswith("auto: 25 of 32")
    idle = plan_execution(CpuBudget(), machine(0.0), units=12, task_id=None, planned_at=now)
    assert (idle.units_at_once, idle.threads_per_session) == (8, 4)
    busy = plan_execution(CpuBudget(), machine(31.0), units=12, task_id=None, planned_at=now)
    assert (busy.cores, busy.units_at_once, busy.threads_first_unit) == (4, 1, 4)
    one = plan_execution(CpuBudget(), machine(0.0), units=1, task_id=None, planned_at=now)
    assert (one.units_at_once, one.threads_first_unit) == (1, 16)
    shared = plan_execution(
        CpuBudget(cpu_budget=16), machine(0.0, running=1), units=12, task_id=None, planned_at=now
    )
    assert (shared.cores, shared.units_at_once, shared.threads_per_session) == (8, 2, 4)
    assert "shared with 1 other" in shared.reason
    single = plan_execution(
        CpuBudget(cpu_budget=1), machine(0.0), units=12, task_id=None, planned_at=now
    )
    assert (single.units_at_once, single.threads_per_session, single.threads_first_unit) == (
        1,
        1,
        1,
    )
    assert parse_cpu_budget("8") == 8 and parse_cpu_budget("auto") == "auto"
    for refused in ("0", "eight", -1, True, 1.5, 4096):
        with pytest.raises(ValueError, match=r"execution.cpu_budget_invalid"):
            parse_cpu_budget(refused)

    store = CpuBudgetStore(tmp_path)
    assert store.read() == CpuBudget() and store.last() is None
    assert store.write("12", chosen_by="INSTALLED_AGENT", chosen_at=now).cpu_budget == 12
    assert store.read().chosen_by == "INSTALLED_AGENT"
    store.record(shared)
    assert store.last() == shared
    (tmp_path / "execution" / "cpu-budget.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match=r"execution.cpu_budget_unreadable"):
        store.read()


def test_auto_reads_the_load_on_the_processors_it_may_use(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a process confined to some processors of a busy machine counts the
    busy work on those alone, so `auto` gives it the idle ones there, not one core."""

    from alphalattice.control.observation_runtime.telemetry import process_metrics
    from alphalattice.evidence.alternative_evidence.runtime import execution

    shares = [100.0, 100.0, 100.0, 100.0, 0.0, 50.0, 0.0, 0.0]

    def cpu_percent(*, interval: float, percpu: bool) -> list[float]:
        assert percpu
        return list(shares)

    monkeypatch.setattr(process_metrics.psutil, "cpu_percent", cpu_percent)
    assert process_metrics.busy_logical_processors(0.0, processor_ids=(4, 5, 6, 7)) == 0.5
    capacity = process_metrics.runtime_machine_capacity().model_copy(
        update={"allowed_logical_processor_ids": (4, 5, 6, 7)}
    )
    monkeypatch.setattr(execution, "runtime_machine_capacity", lambda: capacity)
    machine = execution.machine_load()
    assert (machine.processors, machine.busy_processors) == (4, 0.5)
    cores, reason = execution.budget_cores(execution.CpuBudget(), machine)
    assert cores == 3 and reason.startswith("auto: 3 of 4")


def _enter(writer: object, entered: list[str]) -> None:
    with writer.held():  # type: ignore[attr-defined]
        entered.append("entered")


def test_a_preparation_counts_its_stages_for_the_book_and_the_publisher() -> None:
    """A preparation counts its stages for the book and the publisher."""

    from alphalattice.control.observation_runtime.telemetry.progress import (
        WorkspaceProgressPublisher,
    )
    from alphalattice.evidence.alternative_evidence.runtime.progress import (
        ACQUIRE,
        BUILD,
        CANONICALIZE,
        SELECT,
        PreparationProgress,
        StageWork,
        publish_evidence_work,
    )

    now = [datetime(2026, 9, 25, 12, tzinfo=UTC)]
    progress = PreparationProgress(clock=lambda: now[0])
    published: list[StageWork] = []
    progress.sink = published.append
    task = uuid4()
    acquire = progress.begin(task, "u01", ACQUIRE, total=12)
    for _ in range(3):
        acquire.advance()
    now[0] += timedelta(seconds=2)
    acquire.advance()
    assert [(w.completed, w.total, w.running) for w in published] == [
        (0, 12, True),
        (4, 12, True),
    ], "its start, then at most one move a second"
    assert progress.units(task)["u01"].completed == 4
    acquire.finish(10)
    assert published[-1].completed == published[-1].total == 10 and not published[-1].running
    documents = progress.begin(task, "u01", CANONICALIZE)
    documents.expect(10)
    documents.advance(10)
    documents.finish(10)
    chunks = progress.begin(task, "u01", BUILD)
    chunks.expect(3607)
    chunks.advance(8)
    other = progress.begin(task, "u02", ACQUIRE, total=5)
    other.advance(2)
    assert progress.book(task, planned_filings=17, units_total=2, units_selected=0) == (
        (ACQUIRE, "filings", 12, 17),
        (CANONICALIZE, "documents", 10, 10),
        (BUILD, "chunks", 8, 3607),
        (SELECT, "units", 0, 2),
    )
    progress.drop(task, "u02")
    assert "u02" not in progress.units(task), "a stage that stopped leaves no count"
    assert progress.begin(task, "u01", "admit_evidence_request").advance() is None

    def failing(_work: StageWork) -> None:
        raise OSError("the projection is held by a reader")

    progress.sink = failing
    chunks.finish(3607)
    assert progress.units(task)["u01"].completed == 3607

    with tempfile.TemporaryDirectory() as root:
        publisher = WorkspaceProgressPublisher(Path(root))
        publish_evidence_work(publisher, progress.units(task)["u01"])
        latest = publisher.read()
        assert latest is not None and latest.operation_id == str(task)
        assert latest.stage_id == "u01_build_retrieval_generation"
        assert (latest.completed_units, latest.total_units, latest.unit_name) == (
            3607,
            3607,
            "chunks",
        )
        assert latest.status == "SUCCEEDED" and latest.current_item == "u01"
        progress.begin(task, "u03", BUILD)
        publish_evidence_work(publisher, progress.units(task)["u03"])
        assert publisher.read() == latest, "a stage with nothing to count yet is not published"


def test_the_writer_lets_a_returning_step_through_first_and_the_heaviest_unit_next() -> None:
    """The writer lets a returning step through first and the heaviest unit next."""

    from alphalattice.evidence.alternative_evidence.runtime.service import EvidenceWriter

    writer = EvidenceWriter()
    order: list[str] = []

    def begin(name: str, rank: int) -> None:
        with writer.held(rank=rank):
            order.append(name)

    def come_back(name: str) -> None:
        with writer.held():
            order.append(name)

    threads = [
        Thread(target=begin, args=("unit 3 begins", 2)),
        Thread(target=begin, args=("unit 2 begins", 1)),
        Thread(target=come_back, args=("a step comes back",)),
    ]
    with writer.held(rank=0):
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 5.0
        while len(writer._waiting) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(writer._waiting) == 3, "all three wait"
        writer.pause()
        order.append("the long stage goes on")
    for thread in threads:
        thread.join(timeout=5.0)
    assert order == [
        "a step comes back",
        "the long stage goes on",
        "unit 2 begins",
        "unit 3 begins",
    ]
