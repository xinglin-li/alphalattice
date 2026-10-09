"""Portfolio-driven incremental acquisition: only the bodies a scope still
lacks are fetched; verified local bodies are reused without a GET; each body
is committed durably as it arrives; failures, cancellations, corruption and
capacity refuse or defer by name and never launder. Every scenario runs
through the maintained acquisition owner and the runtime, over the
controlled transport seam -- no network, synthetic bodies labelled as such.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from alphalattice.evidence.alternative_evidence.contracts import (
    DEFAULT_SOURCE_DOCUMENT_BYTES,
)
from alphalattice.evidence.alternative_evidence.sources.acquisition import (
    AlternativeEvidenceTaskCancelled,
)
from alphalattice.evidence.alternative_evidence.sources.contracts import (
    SOURCE_COMMIT_CATEGORY,
    SOURCE_DEFERRAL_CATEGORY,
)
from alphalattice.evidence.alternative_evidence.sources.sec_edgar import (
    AlternativeEvidenceCommitRefused,
    SecEdgarSource,
)
from alphalattice.evidence.alternative_evidence.storage.inventory import INDEX_ROOT
from alphalattice.kernel.knowledge.hybrid_contracts import (
    HYBRID_INDEX_SCHEMA_VERSION,
)
from tests.alternative_evidence_desk.incremental_acquisition_support import (
    AAPL,
    EIGHT_K,
    LATER_EIGHT_K,
    MSFT,
    MSFT_TEN_K,
    MSFT_TEN_Q,
    TEN_K,
    TEN_Q,
    _acquire,
    _admission,
    _commits,
    _outcomes,
    _request,
    _source_objects,
    _transport,
)
from tests.alternative_evidence_desk.planted_corpus import _runtime
from tests.alternative_evidence_desk.sec_scenario_transport import (
    NOW,
    ScenarioFiling,
    SecScenarioTransport,
    filing_body,
)


def test_a_fresh_source_check_with_no_change_downloads_no_body_again(tmp_path: Path) -> None:
    """A fresh source check with no change downloads no body again."""

    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        registry, snapshot, source_set = _acquire(runtime, transport, _request())
        assert len(source_set.documents) == 3 and snapshot.status.value == "COMPLETE"
        assert len(transport.body_calls) == 3 and transport.inventory_calls == 1
        assert _commits(runtime) == 3, "each body is committed on its own as it arrives"
        first_calls = list(transport.calls)
        transport.reset_calls()

        later = NOW + timedelta(days=3)
        _registry, snapshot_again, source_set_again = _acquire(
            runtime, transport, _request(as_of=later), registry=registry
        )
        assert transport.inventory_calls == 1, "a fresh check reads the inventory once"
        assert transport.body_calls == [], f"no body is fetched again: {transport.body_calls}"
        assert transport.registry_calls == 0, "the registry the caller supplies is reused"
        assert {d.content_sha256 for d in source_set_again.documents} == {
            d.content_sha256 for d in source_set.documents
        }
        assert snapshot_again.status.value == "COMPLETE"
        accounting = snapshot_again.acquisition
        assert accounting is not None
        assert accounting.reused_local_count == 3 and accounting.fetched_count == 0
        assert accounting.inventory_request_count == 1 and accounting.body_request_count == 0
        assert accounting.fetched_bytes == 0 and accounting.reused_bytes > 0
        assert len(first_calls) == 5, "registry, inventory and three bodies on the first pass"
        assert len(_source_objects(runtime)) == 3, "three bodies, each held once"
        assert _commits(runtime) == 3, "a reuse commits nothing new"
    finally:
        runtime.close()


def test_one_added_filing_fetches_only_its_own_body(tmp_path: Path) -> None:
    """requirement (matrix row 4): the official inventory gains one filing
    between passes; the next pass fetches that body and nothing else, and
    names the reuse and the fetch separately."""

    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        registry, _snapshot, _set = _acquire(runtime, transport, _request(budget=4))
        transport.reset_calls()
        transport.add_filing(AAPL, LATER_EIGHT_K, filing_body(LATER_EIGHT_K.accession))
        later = NOW + timedelta(days=2)
        _r, snapshot, source_set = _acquire(
            runtime, transport, _request(as_of=later, budget=4), registry=registry
        )
        assert snapshot.status.value == "COMPLETE" and len(source_set.documents) == 4
        assert transport.body_calls == [SecScenarioTransport.locator(AAPL, LATER_EIGHT_K)]
        assert _outcomes(snapshot) == {
            TEN_K.accession: "REUSED_LOCAL",
            TEN_Q.accession: "REUSED_LOCAL",
            EIGHT_K.accession: "REUSED_LOCAL",
            LATER_EIGHT_K.accession: "FETCHED",
        }
        accounting = snapshot.acquisition
        assert accounting is not None
        assert accounting.fetched_count == 1 and accounting.reused_local_count == 3
        assert accounting.fetched_bytes == len(filing_body(LATER_EIGHT_K.accession))
        assert len(_source_objects(runtime)) == 4 and _commits(runtime) == 4
    finally:
        runtime.close()


def test_overlapping_portfolios_share_one_physical_copy_of_each_body(tmp_path: Path) -> None:
    """requirement (matrix rows 5-6): a second scope that overlaps the first
    reuses every body the first obtained -- one physical copy per body
    across the two requests' sets -- and fetches only the new issuer's."""

    transport = _transport(msft=True)
    runtime = _runtime(tmp_path)
    try:
        registry, _s, first_set = _acquire(runtime, transport, _request(("AAPL",)))
        transport.reset_calls()
        _r, snapshot, second_set = _acquire(
            runtime,
            transport,
            _request(("AAPL", "MSFT"), as_of=NOW + timedelta(hours=2)),
            registry=registry,
        )
        assert snapshot.status.value == "COMPLETE"
        assert transport.inventory_calls == 2, "one inventory read per issuer"
        assert sorted(transport.body_calls) == sorted(
            SecScenarioTransport.locator(MSFT, f) for f in (MSFT_TEN_K, MSFT_TEN_Q)
        )
        accounting = snapshot.acquisition
        assert accounting is not None
        assert accounting.reused_local_count == 3 and accounting.fetched_count == 2
        # Two sets, five logical references, five physical bodies -- the
        # three AAPL bodies are named by both sets and held once.
        references = len(first_set.documents) + len(second_set.documents)
        assert references == 8 and len(_source_objects(runtime)) == 5
        assert {d.content_sha256 for d in first_set.documents} <= {
            d.content_sha256 for d in second_set.documents
        }
    finally:
        runtime.close()


def test_two_securities_of_one_issuer_share_the_issuer_bodies(tmp_path: Path) -> None:
    """requirement (matrix row 7): two securities that map to one issuer are
    one set of filings; the second security reuses the bodies the first
    fetched, matched by CIK and accession rather than by the security."""

    registry_map = {"GOOGL": (AAPL, "Alphabet Inc."), "GOOG": (AAPL, "Alphabet Inc.")}
    transport = SecScenarioTransport(
        registry=registry_map,
        filings={AAPL: [EIGHT_K, TEN_Q, TEN_K]},
        bodies={
            SecScenarioTransport.locator(AAPL, f): filing_body(f.accession)
            for f in (EIGHT_K, TEN_Q, TEN_K)
        },
    )
    runtime = _runtime(tmp_path)
    try:
        _r, snapshot, source_set = _acquire(runtime, transport, _request(("GOOGL", "GOOG")))
        assert snapshot.status.value == "COMPLETE" and len(source_set.documents) == 6
        assert len(transport.body_calls) == 3, "each body of the issuer is fetched once"
        assert transport.inventory_calls == 2, "each security's plan reads the inventory"
        outcomes = [(v.entity_id, v.outcome) for v in snapshot.acquisition.documents]
        assert outcomes.count(("GOOGL", "FETCHED")) == 3
        assert outcomes.count(("GOOG", "REUSED_LOCAL")) == 3
        assert len(_source_objects(runtime)) == 3, "one physical copy per body"
        by_security = {
            security: {d.content_sha256 for d in source_set.documents if d.entity_id == security}
            for security in ("GOOGL", "GOOG")
        }
        assert by_security["GOOGL"] == by_security["GOOG"] and len(by_security["GOOG"]) == 3
        assert all(d.title.startswith(d.entity_id + " ") for d in source_set.documents)
    finally:
        runtime.close()


def test_a_failed_body_keeps_its_siblings_and_the_next_attempt_fetches_only_it(
    tmp_path: Path,
) -> None:
    """A failed body keeps its siblings and the next attempt fetches only it."""

    transport = _transport()
    transport.fail_body(AAPL, TEN_K, lambda: RuntimeError("scenario: source unavailable"))
    runtime = _runtime(tmp_path)
    try:
        registry, snapshot, source_set = _acquire(runtime, transport, _request())
        assert snapshot.status.value == "COMPLETE" and len(source_set.documents) == 2
        assert _outcomes(snapshot) == {
            TEN_K.accession: "FAILED",
            TEN_Q.accession: "FETCHED",
            EIGHT_K.accession: "FETCHED",
        }
        assert not any("REQUIRED_BASELINE_MISSING" in v for v in snapshot.limitations)
        assert any("source unavailable" in v for v in snapshot.limitations)
        assert len(_source_objects(runtime)) == 2 and _commits(runtime) == 2
        assert len(transport.body_calls) == 3, "each planned body was attempted once"

        transport.heal_body(AAPL, TEN_K)
        transport.reset_calls()
        _r, again, again_set = _acquire(
            runtime, transport, _request(as_of=NOW + timedelta(hours=1)), registry=registry
        )
        assert again.status.value == "COMPLETE" and len(again_set.documents) == 3
        assert transport.body_calls == [SecScenarioTransport.locator(AAPL, TEN_K)]
        assert _outcomes(again) == {
            TEN_K.accession: "FETCHED",
            TEN_Q.accession: "REUSED_LOCAL",
            EIGHT_K.accession: "REUSED_LOCAL",
        }
        assert len(_source_objects(runtime)) == 3 and _commits(runtime) == 3
    finally:
        runtime.close()


def test_a_failed_event_filing_leaves_the_issuer_available_and_named(tmp_path: Path) -> None:
    """A failed 8-K is not a missing baseline: the issuer's set is its two
    baselines, complete for the request, with the failure named in the
    accounting and the limitations rather than silently absent."""

    transport = _transport()
    transport.fail_body(AAPL, EIGHT_K, lambda: RuntimeError("scenario: 8-K refused"))
    runtime = _runtime(tmp_path)
    try:
        _r, snapshot, source_set = _acquire(runtime, transport, _request())
        assert snapshot.status.value == "COMPLETE" and len(source_set.documents) == 2
        assert _outcomes(snapshot)[EIGHT_K.accession] == "FAILED"
        assert snapshot.acquisition.failed_count == 1
        assert any(
            f"8-K {EIGHT_K.accession} failed -- RuntimeError: scenario: 8-K refused" in v
            for v in snapshot.limitations
        )
        assert not any("REQUIRED_BASELINE_MISSING" in v for v in snapshot.limitations)
    finally:
        runtime.close()


def test_a_cancellation_after_a_committed_body_resumes_without_refetching_it(
    tmp_path: Path,
) -> None:
    """requirement (plan section 3; matrix row 10): the Task is cancelled
    after the first body was committed. The cancellation is raised by name,
    the committed body stays, and the next attempt fetches only the rest."""

    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        with pytest.raises(AlternativeEvidenceTaskCancelled):
            _acquire(
                runtime,
                transport,
                _request(),
                should_cancel=lambda: len(transport.body_calls) >= 1,
            )
        assert len(transport.body_calls) == 1, "cancellation is honoured between bodies"
        committed = transport.body_calls[0]
        assert _commits(runtime) == 1 and len(_source_objects(runtime)) == 1

        transport.reset_calls()
        _r, snapshot, source_set = _acquire(
            runtime, transport, _request(as_of=NOW + timedelta(minutes=5))
        )
        assert snapshot.status.value == "COMPLETE" and len(source_set.documents) == 3
        assert committed not in transport.body_calls and len(transport.body_calls) == 2
        assert snapshot.acquisition.reused_local_count == 1
        assert snapshot.acquisition.fetched_count == 2
    finally:
        runtime.close()


def test_a_known_oversize_body_is_deferred_by_its_record_and_not_transferred_again(
    tmp_path: Path,
) -> None:
    """A known oversize body is deferred by its record and not transferred again."""

    transport = _transport()
    big = filing_body(EIGHT_K.accession, words=200_000)
    transport.bodies[SecScenarioTransport.locator(AAPL, EIGHT_K)] = big
    cap = 100_000
    assert len(big) > cap
    runtime = _runtime(tmp_path)
    try:
        registry, snapshot, source_set = _acquire(runtime, transport, _request(cap=cap))
        assert snapshot.status.value == "COMPLETE" and len(source_set.documents) == 2
        assert _outcomes(snapshot)[EIGHT_K.accession] == "DEFERRED"
        assert transport.body_calls_for(EIGHT_K.accession) == 1, "one bounded transfer"
        assert len(_source_objects(runtime)) == 2, "nothing of the oversize body is kept"
        deferrals = runtime.local_sources.deferrals()
        assert len(deferrals) == 1
        record = deferrals[0]
        assert record.resource_key == (
            "SEC_EDGAR",
            AAPL,
            EIGHT_K.accession,
            EIGHT_K.primary_document,
        )
        assert record.admitted_cap_bytes == cap and record.observed_bytes >= cap
        assert record.form == "8-K" and record.observed_at == NOW
        assert record.recheck_after is None, "no time-based recheck (W6)"
        sealed = runtime.artifacts.root / SOURCE_DEFERRAL_CATEGORY / f"{record.deferral_hash}.json"
        assert sealed.is_file()

        # The same resource under the same cap at a later cutoff, in this
        # runtime: reported deferred by the record, no transfer.
        transport.reset_calls()
        _r, again, _s = _acquire(
            runtime,
            transport,
            _request(as_of=NOW + timedelta(hours=1), cap=cap),
            registry=registry,
        )
        assert transport.body_calls == [], f"a known oversize body moved: {transport.body_calls}"
        assert _outcomes(again)[EIGHT_K.accession] == "DEFERRED"
        assert again.acquisition.reused_local_count == 2
        assert again.acquisition.body_request_count == 0
        observed = NOW.date().isoformat()
        assert any(
            f"known oversize: {record.observed_bytes:,}+ bytes observed on {observed} under a "
            f"{cap:,}-byte cap" in v
            for v in again.limitations
        )
    finally:
        runtime.close()

    # After a restart the sealed record is what the new runtime knows.
    reopened = _runtime(tmp_path)
    try:
        transport.reset_calls()
        _r, restarted, _s = _acquire(
            reopened,
            transport,
            _request(as_of=NOW + timedelta(days=2), cap=cap),
            registry=registry,
        )
        assert transport.body_calls == [] and _outcomes(restarted)[EIGHT_K.accession] == "DEFERRED"
        assert len(reopened.local_sources.deferrals()) == 1

        # A month on, the filing has left the window: the plan no longer
        # selects it, and nothing is transferred or deferred for it.
        transport.reset_calls()
        later = NOW + timedelta(days=31)
        _r, rechecked, _s = _acquire(
            reopened, transport, _request(as_of=later, cap=cap), registry=registry
        )
        assert transport.body_calls_for(EIGHT_K.accession) == 0
        assert EIGHT_K.accession not in _outcomes(rechecked)
        assert len(reopened.local_sources.deferrals()) == 1
        # A larger cap inside the window is a changed prerequisite: one bounded
        # retry, which here succeeds because the body fits under it; the sealed
        # record stays as history of what was observed.
        transport.reset_calls()
        _r, larger, larger_set = _acquire(
            reopened,
            transport,
            _request(as_of=NOW + timedelta(days=3), cap=2_000_000),
            registry=registry,
        )
        assert transport.body_calls_for(EIGHT_K.accession) == 1
        assert len(transport.body_calls) == 1
        assert _outcomes(larger)[EIGHT_K.accession] == "FETCHED" and len(larger_set.documents) == 3

    finally:
        reopened.close()


def test_an_explicit_accession_scope_retries_a_deferred_resource_once(tmp_path: Path) -> None:
    """requirement (Phase A3): an explicit retry -- the request naming the
    accession -- makes the deferred resource eligible for one bounded
    transfer under the same cap."""

    transport = _transport()
    big = filing_body(TEN_K.accession, words=200_000)
    transport.bodies[SecScenarioTransport.locator(AAPL, TEN_K)] = big
    cap = 100_000
    runtime = _runtime(tmp_path)
    try:
        registry, first, _s = _acquire(runtime, transport, _request(cap=cap))
        assert _outcomes(first)[TEN_K.accession] == "DEFERRED"
        transport.reset_calls()
        request = _request(as_of=NOW + timedelta(hours=1), cap=cap)
        _r, scoped, _s = runtime.acquire_live(
            request=request,
            admission=_admission(request),
            source=SecEdgarSource(transport),
            registry=registry,
            published_at=request.evidence_as_of,
            clock=lambda: request.evidence_as_of,
            accession_scopes={"AAPL": frozenset({TEN_K.accession})},
        )
        assert transport.body_calls_for(TEN_K.accession) == 1, "the explicit scope is one retry"
        assert _outcomes(scoped)[TEN_K.accession] == "DEFERRED"
        assert len(runtime.local_sources.deferrals()) == 1
        assert runtime.local_sources.deferrals()[0].observed_at == request.evidence_as_of
    finally:
        runtime.close()


def test_a_tampered_local_body_is_refused_by_name_and_never_refetched(tmp_path: Path) -> None:
    """A tampered local body is refused by name and never refetched."""

    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        registry, _snapshot, source_set = _acquire(runtime, transport, _request())
        ten_k = next(d for d in source_set.documents if d.revision == TEN_K.accession)
        target = runtime.documents.workspace.root / Path(*ten_k.content_object_path.split("/"))
        original = target.read_bytes()
        target.write_bytes(original[:-1] + b"?")
        transport.reset_calls()
        _r, snapshot, again_set = _acquire(
            runtime, transport, _request(as_of=NOW + timedelta(hours=1)), registry=registry
        )
        assert transport.body_calls == [], "a tamper is never healed by a refetch"
        assert snapshot.status.value == "COMPLETE" and len(again_set.documents) == 2
        assert _outcomes(snapshot) == {
            TEN_K.accession: "FAILED",
            TEN_Q.accession: "REUSED_LOCAL",
            EIGHT_K.accession: "REUSED_LOCAL",
        }
        assert any(
            "local integrity: alternative_evidence.source_object_tampered" in v
            for v in snapshot.limitations
        )
        assert not any(value.revision == TEN_K.accession for value in again_set.documents)
    finally:
        runtime.close()


def test_a_refused_durable_keep_stops_the_request_under_the_storage_owner_code(
    tmp_path: Path,
) -> None:
    """A refused durable keep stops the request under the storage owner's code."""

    class BudgetRefused(RuntimeError):
        failure_code = "storage.managed_capacity_exceeded"

    transport = _transport()
    runtime = _runtime(tmp_path)
    admitted: list[int] = []
    budget = len(filing_body(TEN_K.accession)) + 2_000  # one body and its commit record

    def admit(additional: int) -> None:
        if not admitted:
            # The issuer's preflight -- an estimate of every layer, before
            # any transfer -- passes here so the per-write rule is exercised.
            admitted.append(additional)
            return
        if sum(admitted[1:]) + additional > budget:
            raise BudgetRefused("Retained inputs exceed the managed budget.")
        admitted.append(additional)

    runtime.storage_admission = admit
    try:
        with pytest.raises(AlternativeEvidenceCommitRefused) as caught:
            _acquire(runtime, transport, _request())
        assert caught.value.failure_code == "storage.managed_capacity_exceeded"
        # The preflight sized three transfers at the cap across every layer;
        # the first body and its commit record were then admitted together
        # and kept; the second body's keep was refused before any of its
        # bytes were written, and no third transfer was started.
        assert admitted[0] > 3 * DEFAULT_SOURCE_DOCUMENT_BYTES, "the preflight covers every layer"
        assert len(admitted) == 2 and admitted[1] > len(filing_body(TEN_K.accession))
        assert len(transport.body_calls) == 2, "the refused body was the last transfer"
        assert len(_source_objects(runtime)) == 1 and _commits(runtime) == 1
        kept = transport.body_calls[0]

        runtime.storage_admission = None
        transport.reset_calls()
        _r, snapshot, source_set = _acquire(
            runtime, transport, _request(as_of=NOW + timedelta(minutes=1))
        )
        assert snapshot.status.value == "COMPLETE" and len(source_set.documents) == 3
        assert kept not in transport.body_calls and len(transport.body_calls) == 2
        assert len(_source_objects(runtime)) == 3
    finally:
        runtime.close()


def test_a_denied_transport_makes_no_reuse_claim_and_names_the_refusal(tmp_path: Path) -> None:
    """A denied transport makes no reuse claim and names the refusal."""

    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        registry, _s, _set = _acquire(runtime, transport, _request())
        transport.reset_calls()
        for url in [*transport.bodies, f"https://data.sec.gov/submissions/CIK{AAPL}.json"]:
            transport.failures[url] = lambda: RuntimeError("alternative_evidence.network_disabled")
        _r, snapshot, source_set = _acquire(
            runtime, transport, _request(as_of=NOW + timedelta(days=1)), registry=registry
        )
        assert snapshot.status.value == "UNAVAILABLE" and source_set.documents == ()
        assert transport.body_calls == []
        assert snapshot.acquisition.reused_local_count == 0
        assert any("network_disabled" in v for v in snapshot.limitations)
    finally:
        runtime.close()


def test_a_lost_index_is_rebuilt_from_committed_assets_without_a_download(
    tmp_path: Path,
) -> None:
    """A lost index is rebuilt from committed assets without a download."""

    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        _r, _snapshot, source_set = _acquire(runtime, transport, _request())
        document_set = runtime.canonicalize(source_set=source_set, published_at=NOW)
        generation = runtime.build_retrieval(document_set=document_set, built_at=NOW)
        database = (
            runtime.documents.workspace.root
            / Path(*INDEX_ROOT.split("/"))
            / HYBRID_INDEX_SCHEMA_VERSION
            / f"{generation.corpus_hash}-{generation.index_spec_hash}.db"
        )
        assert database.is_file()
        objects_before = _source_objects(runtime)
        transport.reset_calls()
        database.unlink()
        report = runtime.rebuild_retrieval(generation.index_id)
        assert database.is_file(), report
        assert transport.calls == [], "a rebuild never asks the source"
        assert _source_objects(runtime) == objects_before, "originals untouched"
        assert runtime.retrieval.passage_embedding_pass_count == 1, "vectors, not the model"
    finally:
        runtime.close()


def test_a_retained_body_over_a_later_request_cap_is_deferred_by_name(tmp_path: Path) -> None:
    """A retained body over a later request cap is deferred by name."""

    transport = _transport()
    big = filing_body(TEN_K.accession, words=60_000)
    transport.bodies[SecScenarioTransport.locator(AAPL, TEN_K)] = big
    runtime = _runtime(tmp_path)
    try:
        registry, first, _set = _acquire(runtime, transport, _request(cap=2_000_000))
        assert first.status.value == "COMPLETE" and _outcomes(first)[TEN_K.accession] == "FETCHED"
        transport.reset_calls()
        cap = 100_000
        assert len(big) > cap
        _r, snapshot, source_set = _acquire(
            runtime, transport, _request(as_of=NOW + timedelta(hours=1), cap=cap), registry=registry
        )
        assert transport.body_calls == [], "a held body is never fetched to be refused"
        assert _outcomes(snapshot)[TEN_K.accession] == "DEFERRED"
        assert any(
            f"retained body of {len(big):,} bytes exceeds the {cap:,}-byte document cap" in v
            for v in snapshot.limitations
        )
        assert snapshot.status.value == "COMPLETE" and len(source_set.documents) == 2
        assert len(_source_objects(runtime)) == 3, "the retained body stays retained"
    finally:
        runtime.close()


def test_a_storage_preflight_refuses_an_issuer_before_any_transfer(tmp_path: Path) -> None:
    """A storage preflight refuses an issuer before any transfer."""

    class DiskShort(RuntimeError):
        failure_code = "storage.disk_space_insufficient"

    transport = _transport()
    runtime = _runtime(tmp_path)
    asked: list[int] = []

    def admit(additional: int) -> None:
        asked.append(additional)
        if additional >= 1_000_000:  # a transfer's worth; the small sealed records pass
            raise DiskShort("Insufficient space for publication and recovery headroom.")

    runtime.storage_admission = admit
    try:
        with pytest.raises(AlternativeEvidenceCommitRefused) as caught:
            _acquire(runtime, transport, _request())
        assert caught.value.failure_code == "storage.disk_space_insufficient"
        assert transport.body_calls == [] and transport.inventory_calls == 1
        assert asked == [
            int(3 * DEFAULT_SOURCE_DOCUMENT_BYTES * runtime.storage_expansion.peak_per_raw)
        ]
        assert _source_objects(runtime) == {} and _commits(runtime) == 0

        # Bodies already held add nothing to the estimate: once the issuer is
        # held, the preflight asks for nothing and the check reuses.
        runtime.storage_admission = None
        _acquire(runtime, transport, _request(as_of=NOW + timedelta(minutes=1)))
        asked.clear()
        runtime.storage_admission = admit
        _r, snapshot, _s = _acquire(runtime, transport, _request(as_of=NOW + timedelta(minutes=2)))
        assert all(value < 1_000_000 for value in asked), asked
        assert snapshot.acquisition.reused_local_count == 3
    finally:
        runtime.close()


def test_an_amendment_fetches_only_itself_and_the_original_stays_as_sealed(
    tmp_path: Path,
) -> None:
    """An amendment fetches only itself and the original stays as sealed."""

    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        registry, _first, first_set = _acquire(runtime, transport, _request(budget=4))
        original = next(d for d in first_set.documents if d.revision == TEN_K.accession)
        amendment = ScenarioFiling(
            "0000320193-26-000090",
            "10-K/A",
            "2026-08-14",
            "2026-08-14T21:00:00.000Z",
            "aapl-10ka.htm",
            TEN_K.report_date,
        )
        transport.add_filing(AAPL, amendment, filing_body(amendment.accession))
        transport.reset_calls()
        _r, snapshot, source_set = _acquire(
            runtime, transport, _request(as_of=NOW + timedelta(days=3), budget=4), registry=registry
        )
        assert snapshot.status.value == "COMPLETE" and len(source_set.documents) == 4
        assert transport.body_calls == [SecScenarioTransport.locator(AAPL, amendment)]
        assert _outcomes(snapshot) == {
            TEN_K.accession: "REUSED_LOCAL",
            amendment.accession: "FETCHED",
            TEN_Q.accession: "REUSED_LOCAL",
            EIGHT_K.accession: "REUSED_LOCAL",
        }
        kept = next(d for d in source_set.documents if d.revision == TEN_K.accession)
        assert kept.content_sha256 == original.content_sha256
        assert runtime.documents.resolve_source_object(original).content == (
            runtime.documents.resolve_source_object(kept).content
        )
        assert len(_source_objects(runtime)) == 4 and _commits(runtime) == 4
    finally:
        runtime.close()


def test_a_corrupt_commit_record_is_a_named_integrity_refusal_not_a_cache_miss(
    tmp_path: Path,
) -> None:
    """A corrupt commit record is a named integrity refusal not a cache miss."""

    transport = _transport()
    first = _runtime(tmp_path)
    try:
        with pytest.raises(AlternativeEvidenceTaskCancelled):
            _acquire(
                first, transport, _request(), should_cancel=lambda: len(transport.body_calls) >= 1
            )
    finally:
        first.close()
    commits = sorted((first.artifacts.root / SOURCE_COMMIT_CATEGORY).glob("*.json"))
    assert len(commits) == 1 and len(_source_objects(first)) == 1
    sealed = commits[0].read_bytes()
    commits[0].write_bytes(sealed[:-40] + b"corrupted-tail-of-the-sealed-record-xxxx")
    damaged = commits[0].read_bytes()
    transport.reset_calls()

    reopened = _runtime(tmp_path)
    try:
        _r, snapshot, source_set = _acquire(
            reopened, transport, _request(as_of=NOW + timedelta(hours=1))
        )
        assert transport.body_calls == [], "a damaged store fetches nothing"
        assert snapshot.status.value == "UNAVAILABLE" and source_set.documents == ()
        assert all(value == "FAILED" for value in _outcomes(snapshot).values())
        assert snapshot.acquisition.body_request_count == 0
        assert any(
            "local integrity: alternative_evidence.source_commitment_corrupt" in v
            and SOURCE_COMMIT_CATEGORY in v
            for v in snapshot.limitations
        )
        assert (
            reopened.local_sources.damaged and commits[0].stem in reopened.local_sources.damaged[0]
        )
        assert sorted((reopened.artifacts.root / SOURCE_COMMIT_CATEGORY).glob("*.json")) == commits
        assert commits[0].read_bytes() == damaged, "nothing resealed or repaired in place"
        assert len(_source_objects(reopened)) == 1, "the retained body is untouched"
    finally:
        reopened.close()

    # Repaired by explicit action -- the sealed bytes restored -- the same
    # request reuses the committed body and fetches only the other two.
    commits[0].write_bytes(sealed)
    repaired = _runtime(tmp_path)
    try:
        _r, snapshot, source_set = _acquire(
            repaired, transport, _request(as_of=NOW + timedelta(hours=2))
        )
        assert snapshot.status.value == "COMPLETE" and len(source_set.documents) == 3
        assert len(transport.body_calls) == 2 and snapshot.acquisition.reused_local_count == 1
        assert _commits(repaired) == 3 and repaired.local_sources.damaged == ()
    finally:
        repaired.close()


def test_a_lost_body_under_an_intact_commit_is_restored_under_that_commit(
    tmp_path: Path,
) -> None:
    """A lost body under an intact commit is restored under that commit."""

    transport = _transport()
    runtime = _runtime(tmp_path)
    try:
        registry, _s, source_set = _acquire(runtime, transport, _request())
        ten_k = next(d for d in source_set.documents if d.revision == TEN_K.accession)
        target = runtime.documents.workspace.root / Path(*ten_k.content_object_path.split("/"))
        target.unlink()
        commits_before = sorted(
            p.read_bytes() for p in (runtime.artifacts.root / SOURCE_COMMIT_CATEGORY).glob("*.json")
        )
    finally:
        runtime.close()
    transport.reset_calls()
    reopened = _runtime(tmp_path)
    try:
        _r, snapshot, _again = _acquire(
            reopened, transport, _request(as_of=NOW + timedelta(hours=1)), registry=registry
        )
        assert snapshot.status.value == "COMPLETE"
        assert transport.body_calls == [SecScenarioTransport.locator(AAPL, TEN_K)]
        assert _outcomes(snapshot)[TEN_K.accession] == "FETCHED"
        assert target.is_file() and target.read_bytes() == filing_body(TEN_K.accession)
        commits_after = sorted(
            p.read_bytes()
            for p in (reopened.artifacts.root / SOURCE_COMMIT_CATEGORY).glob("*.json")
        )
        assert commits_after == commits_before, "restored under the intact commit, not resealed"
        assert reopened.local_sources.damaged == ()
    finally:
        reopened.close()
