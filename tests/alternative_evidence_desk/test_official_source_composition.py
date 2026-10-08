"""The official SEC source inside the maintained product composition: admitted
only by explicit consent and an open environment (offline is the default),
composed onto the live branch of the workspace admission, and driven through
the real public operation -- `EVIDENCE_PREPARE` over HTTP -- with an injected
transport of controlled responses: a fresh source check downloads no body the
workspace already holds, a second caller joins the refresh in flight, and the
preview reads the holdings and the source check back. No network anywhere.
"""

from __future__ import annotations

import shutil
import threading
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from alphalattice.control.product_host.composition.evidence_review_bundles import (
    EvidenceReviewBundles,
)
from alphalattice.control.product_host.composition.evidence_review_workspace import (
    admit_evidence_review_workspace,
    live_evidence_policy,
)
from alphalattice.control.workspace_runtime.network_access import set_network_access
from alphalattice.evidence.alternative_evidence.contracts import (
    WHOLE_FILING_BYTES,
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceSourcePolicy,
)
from alphalattice.evidence.alternative_evidence.runtime.policy import AdmittedEvidencePolicy
from alphalattice.evidence.alternative_evidence.sources.admission import admit_official_source
from alphalattice.evidence.alternative_evidence.sources.sec_edgar import (
    HttpxSecOfficialTransport,
    SecEdgarSource,
)
from alphalattice.interface.local_application.client import LocalResearchClient
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewDossier,
)
from tests.alternative_evidence_desk.issuer_listing import TICKERS
from tests.alternative_evidence_desk.planted_corpus import _NOW, CIKS
from tests.alternative_evidence_desk.review_http_support import (
    build_authority,
    build_workspace,
    start_service,
)
from tests.alternative_evidence_desk.review_package import CAPABILITY_HASH, _package
from tests.alternative_evidence_desk.sec_scenario_transport import (
    ScenarioFiling,
    SecScenarioTransport,
    filing_body,
)

ONE_UNIT = "u01"
"""The book's only unit: every book is prepared as a coverage run (C2)."""

LEAVES_WINDOW = datetime(2026, 7, 20, 21, tzinfo=UTC) + timedelta(days=30)
"""The day the scenario's earliest filing, each issuer's 10-K, leaves the 30-day window."""


def _issuer_filings(ticker: str) -> list[ScenarioFiling]:
    """Two periodic reports per issuer, accepted inside the 30-day window before the
    route's clock."""

    cik = CIKS[ticker]
    stem = ticker.casefold()
    return [
        ScenarioFiling(
            f"{cik}-26-000050",
            "10-Q",
            "2026-08-01",
            "2026-08-01T20:00:00.000Z",
            f"{stem}-10q.htm",
            "2026-03-28",
        ),
        ScenarioFiling(
            f"{cik}-25-000100",
            "10-K",
            "2026-07-20",
            "2026-07-20T21:00:00.000Z",
            f"{stem}-10k.htm",
            "2025-09-27",
        ),
    ]


def _scenario() -> SecScenarioTransport:
    filings = {CIKS[ticker]: _issuer_filings(ticker) for ticker in TICKERS}
    bodies = {
        SecScenarioTransport.locator(cik, filing): filing_body(filing.accession)
        for cik, values in filings.items()
        for filing in values
    }
    registry = {ticker: (CIKS[ticker], f"{ticker} Inc.") for ticker in TICKERS}
    transport = SecScenarioTransport(registry=registry, filings=filings, bodies=bodies)
    transport.retrieved_at = _NOW
    return transport


def test_the_official_source_is_admitted_only_by_consent_in_an_open_environment(
    tmp_path: Path,
) -> None:
    """requirement (plan section 4): explicit network consent and offline
    defaults. Without consent nothing is composed; with it the official
    client still needs the environment to allow the network and to name the
    SEC contact; an injected transport is admitted as such."""

    refused = admit_official_source(network_consent=False)
    assert refused.source is None and refused.transport_origin == "NONE"
    assert refused.refusal_code == "evidence_review.explicit_sec_network_consent_required"

    closed = admit_official_source(
        network_consent=True, environment={"ALPHALATTICE_NETWORK_DISABLED": "1"}
    )
    # Consent with the network disabled: the live branch over a transport
    # that refuses every request by name, so what is held reads back and
    # nothing leaves the process (matrix row 16).
    assert closed.admitted and closed.transport_origin == "DENIED"
    assert closed.refusal_code == "evidence_review.network_disabled"
    assert closed.network_access is not None
    assert closed.network_access.allowed is False
    assert closed.network_access.decided_by == "OPERATOR_OFFLINE_SWITCH"
    with pytest.raises(ValueError, match="network_disabled"):
        closed.source.acquire_registry(captured_at=_NOW)  # type: ignore[union-attr]
    assert closed.source.network_call_count == 1  # type: ignore[union-attr]

    set_network_access(tmp_path, enabled=True)
    unnamed = admit_official_source(network_consent=True, workspace_root=tmp_path, environment={})
    assert unnamed.source is None
    assert unnamed.refusal_code == "evidence_review.sec_user_agent_required"

    official = admit_official_source(
        network_consent=True,
        workspace_root=tmp_path,
        environment={"SEC_USER_AGENT": "QA qa@example.com"},
    )
    assert official.admitted and official.transport_origin == "OFFICIAL_HTTP"
    assert official.network_access is not None and official.network_access.allowed
    assert isinstance(official.source, SecEdgarSource)
    assert official.source.network_call_count == 0, "admission makes no request"
    official.source._transport.close()  # type: ignore[attr-defined]

    injected = admit_official_source(network_consent=True, transport=_scenario())
    assert injected.admitted and injected.transport_origin == "INJECTED"
    assert injected.network_consent is True and injected.refusal_code is None
    assert injected.network_access is None


def test_the_workspace_admission_takes_the_live_branch_only_with_an_admitted_source(
    tmp_path: Path,
) -> None:
    """requirement (plan section 4): the acquisition owner is wired through
    the maintained application composition. The same verified package is
    composed on the recorded branch by default, and on the live branch --
    the source as the Task resources' live source, the policy asking for SEC
    filings under LIVE_OFFICIAL with the consent recorded -- only when an
    official source was admitted. A refused admission changes nothing."""

    binding, _registry_path = _package(tmp_path)
    recorded = admit_evidence_review_workspace(
        workspace=tmp_path,
        binding=binding,
        semantic_capability_reader=lambda _runtime: CAPABILITY_HASH,
        official_source=admit_official_source(network_consent=False),
    )
    try:
        assert recorded.resources.live_source is None
        # The package was installed under the production plan (written
        # absent): the policy carries it as installed, and the Host refuses
        # new work under it by name with the re-install step.
        assert recorded.evidence_policy == AdmittedEvidencePolicy(
            admit_model_review=False, matter_selection=None
        )
    finally:
        recorded.runtime.close()

    transport = _scenario()
    live = admit_evidence_review_workspace(
        workspace=tmp_path,
        binding=binding,
        semantic_capability_reader=lambda _runtime: CAPABILITY_HASH,
        official_source=admit_official_source(network_consent=True, transport=transport),
    )
    try:
        assert isinstance(live.resources.live_source, SecEdgarSource)
        policy = live.evidence_policy
        assert policy.mode is AlternativeEvidenceMode.LIVE_OFFICIAL
        assert policy.evidence_classes == (AlternativeEvidenceClass.SEC_FILING,)
        assert policy.network_consent and policy.admit_live_official
        assert policy.admit_model_review is False, "consent to the source is not a model"
        # The recorded policy's, with a short unit delivered whole (W4).
        assert policy.source_policy == AlternativeEvidenceSourcePolicy(
            **{
                **AdmittedEvidencePolicy().source_policy.model_dump(mode="python"),
                "whole_filing_bytes": WHOLE_FILING_BYTES,
            }
        )
        assert transport.calls == [], "admission performs no acquisition"
        # The recorded package stays admitted beside the live source: the
        # registry and listing authority are the same verified ones.
        assert live.registry == recorded.registry
        assert live.resources.recorded_documents == recorded.resources.recorded_documents
    finally:
        live.runtime.close()

    denied_source = admit_official_source(
        network_consent=True, environment={"ALPHALATTICE_NETWORK_DISABLED": "1"}
    )
    held = admit_evidence_review_workspace(
        workspace=tmp_path,
        binding=binding,
        semantic_capability_reader=lambda _runtime: CAPABILITY_HASH,
        official_source=denied_source,
    )
    try:
        assert held.network_access == denied_source.network_access
        assert held.evidence_policy == live.evidence_policy
        assert held.resources.binding_hash == live.resources.binding_hash
        assert held.resources.live_source is not None
        assert held.resources.live_source.network_call_count == 0
    finally:
        held.runtime.close()


def test_the_public_operation_prepares_live_reuses_bodies_and_joins_a_refresh_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (plan sections 4-5; matrix rows 2, 3, 12, 17): the live
    branch through the real public operation over HTTP, with controlled
    responses. The first preparation fetches every selected body once and
    commits it; a second caller during it is answered with the Task in
    flight, not a second acquisition; the same intent later is the exact
    reuse (no source check at all); a new cutoff is a fresh source check
    that reads the inventory and downloads no body again; the preview reads
    the holdings and the source check back. The returned requests submit
    unchanged."""

    import socket

    real_connect = socket.socket.connect

    def connect(self: Any, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError(f"outbound network call: {address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)

    workspace, report = build_workspace(tmp_path)
    transport = _scenario()
    recorded = build_authority(tmp_path=tmp_path, report=report, model_authority_admitted=False)
    assert recorded.evidence_resources is not None
    authority = replace(
        recorded,
        evidence_resources=replace(
            recorded.evidence_resources, live_source=SecEdgarSource(transport)
        ),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
    )
    now = [_NOW]
    service = start_service(workspace, authority, tmp_path, clock=lambda: now[0])
    try:
        selected = {"result_hash": service.result_hash()}
        preview = service.get("/api/evidence/preview?" + f"result_hash={selected['result_hash']}")
        assert preview["status"] == "EVIDENCE_PREPARATION_READY"
        assert preview["source_mode"] == "LIVE_OFFICIAL"
        assert preview["source_work"] == "EXPLICITLY_ADMITTED_OFFICIAL_ACQUISITION"
        holdings = preview["source_inventory"]
        issuers = tuple(sorted(row["entity_id"] for row in holdings["issuers"]))
        assert issuers and set(issuers) <= set(TICKERS)
        assert holdings["issuers_with_retained_documents"] == 0
        assert holdings["retained_bytes"] == 0 and preview["source_check"] is None
        assert transport.calls == [], "the preview performs no acquisition"
        captured = preview["next_requests"]["prepare"]
        assert captured["evidence_as_of"] == _NOW.isoformat()
        expected_bodies = 2 * len(issuers)

        # A second caller while the first body is still transferring: the
        # transport holds that transfer until the second request has been
        # answered, so the answer is decided against a running Task.
        first_body = threading.Event()
        second_answered = threading.Event()

        def before_get(url: str) -> None:
            if "/Archives/edgar/data/" in url and not first_body.is_set():
                first_body.set()
                assert second_answered.wait(timeout=60), "the second caller never returned"

        transport.before_get = before_get
        prepared = service.post("/api/evidence/prepare", _submission(captured))
        assert prepared["disposition"] == "ADMITTED"
        assert first_body.wait(timeout=60), "the first transfer never started"
        joined = service.post("/api/evidence/prepare", _submission(captured))
        assert joined["disposition"] == "REUSED_IN_FLIGHT"
        assert joined["task_id"] == prepared["task_id"]
        assert joined["lifecycle"] in {"QUEUED", "RUNNING"}
        assert joined["evidence_as_of"] == captured["evidence_as_of"]
        assert "already" in joined["detail"] and "twice" in joined["detail"]
        # A request for another cutoff is another question (R2): the active
        # Task does not answer it. The existing queue admits it behind the
        # running one, and it is prepared with a fresh source check that
        # reuses every body the first Task committed.
        other_cutoff = (_NOW + timedelta(minutes=1)).isoformat()
        other = service.post(
            "/api/evidence/prepare",
            {
                **{
                    k: v
                    for k, v in _submission(captured).items()
                    if k != "preparation_binding_hash"
                },
                "evidence_as_of": other_cutoff,
            },
        )
        second_answered.set()
        assert other["disposition"] == "ADMITTED", other
        assert other["task_id"] != prepared["task_id"]
        assert other["evidence_as_of"] == other_cutoff
        service.drain()
        transport.before_get = None
        task_id = prepared["task_id"]
        assert service.get(f"/api/status?task_id={task_id}")["lifecycle"] == "SUCCEEDED"
        assert service.get(f"/api/status?task_id={other['task_id']}")["lifecycle"] == "SUCCEEDED"
        assert transport.registry_calls == 2 and transport.inventory_calls == 2 * len(issuers)
        assert len(transport.body_calls) == expected_bodies
        assert len(set(transport.body_calls)) == expected_bodies, "each body once, by one Task"
        adapter = service.review.evidence_task_adapter
        from uuid import UUID

        other_check = adapter.source_check(UUID(other["task_id"]), unit_id=ONE_UNIT)
        assert other_check["reused_local_count"] == expected_bodies
        assert other_check["body_request_count"] == 0
        assert other_check["evidence_as_of"] == other_cutoff
        tasks_after_first = len(service.registry.tasks())

        after = service.get("/api/evidence/preview?" + f"result_hash={selected['result_hash']}")
        assert after["prepared_task_id"] == task_id
        check = after["source_check"]
        assert check is not None
        assert check["snapshot_status"] == "COMPLETE"
        assert check["fetched_count"] == expected_bodies and check["reused_local_count"] == 0
        assert check["body_request_count"] == expected_bodies
        # The preparation read each filing index before packing; the Task took
        # those plans and read no index itself.
        assert check["inventory_request_count"] == 0
        assert check["fetched_bytes"] > 0 and check["reused_bytes"] == 0
        holdings = after["source_inventory"]
        assert holdings["issuers_with_retained_documents"] == len(issuers)
        assert all(row["retained_documents"] == 2 for row in holdings["issuers"])
        assert holdings["retained_bytes"] == check["fetched_bytes"]
        # A6 (R20): the workspace's retained documents, every registered issuer's, newest
        # accepted first, a page at a time; the read adds no Task.
        documents = service.get("/api/evidence/documents")
        assert documents["status"] == "EVIDENCE_DOCUMENTS"
        assert (documents["page"], documents["page_count"], documents["total"]) == (
            1,
            1,
            2 * len(issuers),
        )
        assert documents["retained_bytes"] == holdings["retained_bytes"]
        assert {row["entity_id"] for row in documents["documents"]} == {
            row["entity_id"] for row in holdings["issuers"]
        }
        accepted = [str(row["accepted_at"]) for row in documents["documents"]]
        assert accepted == sorted(accepted, reverse=True)
        assert documents["next_requests"] == {}
        # Over the Local Web a page number is a query string: the route reads it as the int it
        # names, so a page past the last is refused by that name, not as an invalid field.
        assert service.get("/api/evidence/documents?documents_page=1") == documents
        status, refused = service.request("/api/evidence/documents?documents_page=2")
        assert status != 200 and "documents_page_out_of_range:2 of 1" in str(refused), refused
        assert len(service.registry.tasks()) == tasks_after_first

        # The same captured intent later: the completed preparation, and no
        # source check of any kind.
        now[0] = _NOW + timedelta(hours=2)
        transport.reset_calls()
        again = service.post("/api/evidence/prepare", _submission(captured))
        assert again["disposition"] == "REUSED_EXACT" and again["task_id"] == task_id
        assert transport.calls == []
        assert len(service.registry.tasks()) == tasks_after_first

        # A new cutoff is a fresh source check: the inventory is read again,
        # every selected body is already held and verified, none is fetched.
        fresh = service.get("/api/evidence/preview?" + f"result_hash={selected['result_hash']}")
        assert fresh["prepared_task_id"] is None
        renewed = fresh["next_requests"]["prepare"]
        assert renewed["evidence_as_of"] == now[0].isoformat()
        submitted = service.post("/api/evidence/prepare", _submission(renewed))
        assert submitted["disposition"] == "ADMITTED"
        service.drain()
        assert service.get(f"/api/status?task_id={submitted['task_id']}")["lifecycle"] == (
            "SUCCEEDED"
        )
        assert transport.registry_calls == 1 and transport.inventory_calls == len(issuers)
        assert transport.body_calls == [], f"a body was fetched again: {transport.body_calls}"
        final = service.get("/api/evidence/preview?" + f"result_hash={selected['result_hash']}")
        assert final["prepared_task_id"] == submitted["task_id"]
        check = final["source_check"]
        assert check["reused_local_count"] == expected_bodies and check["fetched_count"] == 0
        assert check["body_request_count"] == 0 and check["fetched_bytes"] == 0
        assert check["reused_bytes"] == final["source_inventory"]["retained_bytes"]
        objects = (
            service.review.evidence_task_adapter.runtime.documents.workspace.root
            / ".system"
            / "source-objects"
        )
        assert len([p for p in objects.iterdir() if p.is_file()]) == expected_bodies
    finally:
        service.session.stop()


def _submission(request: dict[str, str]) -> dict[str, str]:
    """A returned request submitted unchanged, as the buttons submit it."""

    return {key: value for key, value in request.items() if key != "operation"}


def test_documents_lists_healthy_commitments_beside_a_named_refused_commitment(
    tmp_path: Path,
) -> None:
    """One unreadable source-set record refuses by identity without hiding other holdings."""
    workspace, report = build_workspace(tmp_path)
    transport = _scenario()
    recorded = build_authority(tmp_path=tmp_path, report=report, model_authority_admitted=False)
    assert recorded.evidence_resources is not None
    authority = replace(
        recorded,
        evidence_resources=replace(
            recorded.evidence_resources, live_source=SecEdgarSource(transport)
        ),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
    )
    service = start_service(workspace, authority, tmp_path)
    try:
        preview = service.get(f"/api/evidence/preview?result_hash={service.result_hash()}")
        admitted = service.post(
            "/api/evidence/prepare", _submission(preview["next_requests"]["prepare"])
        )
        assert admitted["disposition"] == "ADMITTED", admitted
        service.drain()
        assert service.get(f"/api/status?task_id={admitted['task_id']}")["lifecycle"] == (
            "SUCCEEDED"
        )
        artifact_root = service.review.evidence_task_adapter.runtime.artifacts.root
        source_set = next((artifact_root / "source-document-sets").glob("*.json"))
    finally:
        service.session.stop()

    source_set.write_text("{", encoding="utf-8")
    reopened = build_authority(tmp_path=tmp_path, report=report, model_authority_admitted=False)
    service = start_service(workspace, reopened, tmp_path)
    try:
        documents = service.get("/api/evidence/documents")
        assert documents["status"] == "EVIDENCE_DOCUMENTS"
        assert documents["documents"], "intact per-document commitments remain readable"
        assert documents["total"] > 0
        assert len(documents["refused_documents"]) == 1
        refusal = documents["refused_documents"][0]
        assert (refusal["status"], refusal["artifact_kind"], refusal["artifact_hash"]) == (
            "REFUSED",
            "source-document-sets",
            source_set.stem,
        )
        assert refusal["failure_code"] == "alternative_evidence.artifact_tampered"
        requests = refusal["next_requests"]
        assert requests["documents"] == {
            "operation": "EVIDENCE_DOCUMENTS",
            "documents_page": 1,
        }
        assert requests["storage"] == {"operation": "STORAGE_READBACK"}
        assert requests["workspace"] == {"operation": "WORKSPACE_SHOW"}
        assert requests["backups"] == {"operation": "WORKSPACE_BACKUPS"}
    finally:
        service.session.stop()


def test_a_parser_only_change_prepares_again_from_retained_bytes(tmp_path: Path) -> None:
    """requirement (plan section 3; matrix row 8): a canonicalization rule
    change supersedes the preparation but not the bytes. The next
    preparation under the rotated binding checks the inventory and
    canonicalizes again from the retained originals: zero body downloads,
    every selected body reused, the new document set sealed under the new
    binding."""

    workspace, report = build_workspace(tmp_path)
    transport = _scenario()
    recorded = build_authority(tmp_path=tmp_path, report=report, model_authority_admitted=False)
    assert recorded.evidence_resources is not None
    authority = replace(
        recorded,
        evidence_resources=replace(
            recorded.evidence_resources, live_source=SecEdgarSource(transport)
        ),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
    )
    now = [_NOW]
    service = start_service(workspace, authority, tmp_path, clock=lambda: now[0])
    try:
        query = f"result_hash={service.result_hash()}"
        preview = service.get("/api/evidence/preview?" + query)
        issuers = len(preview["source_inventory"]["issuers"])
        first = service.post(
            "/api/evidence/prepare", _submission(preview["next_requests"]["prepare"])
        )
        service.drain()
        assert service.get(f"/api/status?task_id={first['task_id']}")["lifecycle"] == "SUCCEEDED"
        fetched = len(transport.body_calls)
        assert fetched == 2 * issuers

        # The parser moves: the sealed preparation is history, not reuse.
        runtime = service.review.evidence_task_adapter.runtime
        runtime.document_binding_hash = "f" * 64
        transport.reset_calls()
        superseded = service.get("/api/evidence/preview?" + query)
        assert superseded["prepared_task_id"] is None
        again = service.post(
            "/api/evidence/prepare", _submission(superseded["next_requests"]["prepare"])
        )
        assert again["disposition"] == "ADMITTED" and again["task_id"] != first["task_id"]
        service.drain()
        assert service.get(f"/api/status?task_id={again['task_id']}")["lifecycle"] == "SUCCEEDED"
        assert transport.body_calls == [], f"bodies fetched again: {transport.body_calls}"
        # The index read at this cutoff is reused: what was filed up to a
        # cutoff does not move, so a parser change reads no index again.
        assert transport.inventory_calls == 0, "the cutoff's filing index read once"
        check = service.get("/api/evidence/preview?" + query)["source_check"]
        assert check["reused_local_count"] == fetched and check["fetched_count"] == 0
        adapter = service.review.evidence_task_adapter
        from uuid import UUID

        document_set = adapter._document_set(
            adapter.registry.task(UUID(again["task_id"])), ONE_UNIT
        )
        assert document_set.canonicalization_binding_hash == "f" * 64
    finally:
        service.session.stop()


def test_a_denied_source_reads_the_live_preparation_back_and_fetches_nothing(
    tmp_path: Path,
) -> None:
    """requirement (matrix row 16, seen on the campaign restart): a live
    preparation reopened with consent but no network reads back -- the
    prepared Task, its packet, the holdings -- and a new source check is
    refused by name with no outbound work and no local body presented as
    fresh."""

    workspace, report = build_workspace(tmp_path)
    transport = _scenario()
    recorded = build_authority(tmp_path=tmp_path, report=report, model_authority_admitted=False)
    assert recorded.evidence_resources is not None
    live = replace(
        recorded,
        evidence_resources=replace(
            recorded.evidence_resources, live_source=SecEdgarSource(transport)
        ),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
    )
    now = [_NOW]
    service = start_service(workspace, live, tmp_path, clock=lambda: now[0])
    try:
        query = f"result_hash={service.result_hash()}"
        preview = service.get("/api/evidence/preview?" + query)
        first = service.post(
            "/api/evidence/prepare", _submission(preview["next_requests"]["prepare"])
        )
        service.drain()
        task_id = first["task_id"]
        assert service.get(f"/api/status?task_id={task_id}")["lifecycle"] == "SUCCEEDED"
        fetched = len(transport.body_calls)
    finally:
        service.session.stop()

    denied = admit_official_source(
        network_consent=True, environment={"ALPHALATTICE_NETWORK_DISABLED": "1"}
    )
    assert denied.transport_origin == "DENIED" and denied.source is not None
    offline = replace(
        recorded,
        evidence_resources=replace(recorded.evidence_resources, live_source=denied.source),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
        network_access=denied.network_access,
    )
    service = start_service(workspace, offline, tmp_path, clock=lambda: now[0])
    try:
        query = f"result_hash={service.result_hash()}"
        again = service.get("/api/evidence/preview?" + query)
        assert again["prepared_task_id"] == task_id, "the live preparation reads back"
        assert again["source_check"]["fetched_count"] == fetched
        assert again["source_inventory"]["issuers_with_retained_documents"] > 0
        packet = service.get(
            f"/api/evidence/packet?result_hash={query.split('=')[1]}&task_id={task_id}"
            f"&evidence_unit_id={ONE_UNIT}"
        )
        assert packet["status"] == "EVIDENCE_ANALYST_PACKET_READY"
        reused = service.post(
            "/api/evidence/prepare", _submission(again["next_requests"]["prepare"])
        )
        assert reused["disposition"] == "REUSED_EXACT" and reused["task_id"] == task_id
        # A new cutoff refuses before admission or an inventory request;
        # a held body does not make its filing index current.
        now[0] = _NOW + timedelta(hours=1)
        later = service.get("/api/evidence/preview?" + query)
        assert later["status"] == "EVIDENCE_PREREQUISITES_MISSING"
        assert later["failure_code"] == "evidence_review.workspace_network_not_allowed"
        access = later["network_access"]
        assert access["network_allowed"] is False
        assert access["decided_by"] == "OPERATOR_OFFLINE_SWITCH"
        assert access["next_action"] == "RESTART_WITHOUT_OPERATOR_OFFLINE_SWITCH"
        assert later["next_requests"]["network"] == {"operation": "NETWORK_ACCESS"}
        assert "prepare" not in later["next_requests"]
        tasks_before = service.registry.tasks()
        fresh = service.post(
            "/api/evidence/prepare",
            _submission(
                {
                    **preview["next_requests"]["prepare"],
                    "evidence_as_of": now[0].isoformat(),
                    "preparation_binding_hash": later["preparation_binding_hash"],
                }
            ),
        )
        assert fresh["disposition"] == "REFUSED_NETWORK_ACCESS"
        assert fresh["failure_code"] == "evidence_review.workspace_network_not_allowed"
        assert "task_id" not in fresh
        assert fresh["network_access"] == access
        assert fresh["detail"] == "The command gave SEC consent. " + str(access["detail"])
        assert "Restart only an idle Host" in fresh["source_ways"]["official"]["before"]
        assert service.registry.tasks() == tasks_before
        refused = denied.source._transport.refused  # type: ignore[attr-defined]
        assert refused == [], "the refused preparation performs no source work"
        assert transport.body_calls[fetched:] == [], "the injected source was never used again"
        assert service.get("/api/evidence/preview?" + query)["prepared_task_id"] is None
    finally:
        service.session.stop()


def test_a_denied_source_requires_an_idle_restart_after_the_control_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (P4): opening the control cannot replace a denied transport;
    its refusal reports the current control and the existing idle-restart recovery."""
    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "0")
    workspace, report = build_workspace(tmp_path)
    recorded = build_authority(tmp_path=tmp_path, report=report, model_authority_admitted=False)
    assert recorded.evidence_resources is not None
    denied = admit_official_source(network_consent=True, workspace_root=workspace, environment={})
    assert denied.source is not None and denied.network_access is not None
    assert denied.network_access.decided_by == "DEFAULT"
    authority = replace(
        recorded,
        evidence_resources=replace(recorded.evidence_resources, live_source=denied.source),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
        network_access=denied.network_access,
    )
    service = start_service(workspace, authority, tmp_path, clock=lambda: _NOW)
    try:
        query = f"result_hash={service.result_hash()}"
        closed = service.get("/api/evidence/preview?" + query)
        assert closed["network_access"]["network_allowed"] is False
        set_network_access(workspace, enabled=True)
        opened = service.get("/api/evidence/preview?" + query)
        assert opened["status"] == "EVIDENCE_PREREQUISITES_MISSING"
        assert opened["network_access"]["network_allowed"] is True
        assert opened["network_access"]["decided_by"] == "WORKSPACE_CONTROL"
        assert opened["source_network_access"]["network_allowed"] is False
        assert opened["source_network_access"]["decided_by"] == "DEFAULT"
        assert "prepare" not in opened["next_requests"]
        assert "No workspace control is set" not in opened["detail"]
        assert "Restart only an idle Host" in opened["source_ways"]["official"]["before"]
        tasks_before = service.registry.tasks()
        refused = service.post(
            "/api/evidence/prepare",
            {
                "result_hash": service.result_hash(),
                "evidence_as_of": opened["evidence_as_of"],
                "preparation_binding_hash": opened["preparation_binding_hash"],
            },
        )
        assert refused["disposition"] == "REFUSED_NETWORK_ACCESS"
        assert refused["failure_code"] == "evidence_review.workspace_network_not_allowed"
        assert refused["network_access"] == opened["network_access"]
        assert refused["detail"] == opened["source_ways"]["official"]["before"]
        assert refused["next_action"] == "READ_NETWORK_ACCESS"
        assert service.registry.tasks() == tasks_before
        assert denied.source.network_call_count == 0
    finally:
        service.session.stop()


def test_a_real_source_reads_the_closed_control_before_any_new_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (P4): a real client admitted while open cannot acquire after
    the workspace closes; its controlled HTTP transport is never called."""
    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "0")
    workspace, report = build_workspace(tmp_path)
    set_network_access(workspace, enabled=True)
    admission = admit_official_source(
        network_consent=True,
        workspace_root=workspace,
        environment={"SEC_USER_AGENT": "QA qa@example.com"},
    )
    assert admission.source is not None and admission.network_access is not None
    assert admission.transport_origin == "OFFICIAL_HTTP" and admission.network_access.allowed
    admission.source.close()
    requests: list[httpx.Request] = []

    def controlled(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503)

    with HttpxSecOfficialTransport(
        user_agent="QA qa@example.com", transport=httpx.MockTransport(controlled)
    ) as transport:
        source = SecEdgarSource(transport)
        assert source.network_capable
        recorded = build_authority(tmp_path=tmp_path, report=report, model_authority_admitted=False)
        assert recorded.evidence_resources is not None
        authority = replace(
            recorded,
            evidence_resources=replace(recorded.evidence_resources, live_source=source),
            evidence_policy=live_evidence_policy(recorded.evidence_policy),
            network_access=admission.network_access,
        )
        service = start_service(workspace, authority, tmp_path, clock=lambda: _NOW)
        try:
            query = f"result_hash={service.result_hash()}"
            ready = service.get("/api/evidence/preview?" + query)
            assert ready["status"] == "EVIDENCE_PREPARATION_READY"
            set_network_access(workspace, enabled=False)
            closed = service.get("/api/evidence/preview?" + query)
            assert closed["status"] == "EVIDENCE_PREREQUISITES_MISSING"
            assert closed["network_access"]["network_allowed"] is False
            assert closed["network_access"]["decided_by"] == "WORKSPACE_CONTROL"
            assert closed["source_network_access"]["network_allowed"] is True
            assert "prepare" not in closed["next_requests"]
            tasks_before = service.registry.tasks()
            refused = service.post(
                "/api/evidence/prepare", _submission(ready["next_requests"]["prepare"])
            )
            assert refused["failure_code"] == "evidence_review.workspace_network_not_allowed"
            assert refused["network_access"] == closed["network_access"]
            assert "task_id" not in refused
            assert service.registry.tasks() == tasks_before
            assert source.network_call_count == 0 and transport.request_count == 0
            assert requests == []
        finally:
            service.session.stop()


def test_a_failed_live_run_is_not_retried_with_a_denied_source(tmp_path: Path) -> None:
    """requirement (P4): an already sealed failed run does not bypass the network
    refusal when its exact preparation is retried after a Host restart."""
    workspace, report = build_workspace(tmp_path)
    scenario = _scenario()
    scenario.failures["https://www.sec.gov/files/company_tickers.json"] = lambda: ValueError(
        "alternative_evidence.network_disabled"
    )
    recorded = build_authority(tmp_path=tmp_path, report=report, model_authority_admitted=False)
    assert recorded.evidence_resources is not None
    live = replace(
        recorded,
        evidence_resources=replace(
            recorded.evidence_resources, live_source=SecEdgarSource(scenario)
        ),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
    )
    service = start_service(workspace, live, tmp_path, clock=lambda: _NOW)
    try:
        preview = service.get(f"/api/evidence/preview?result_hash={service.result_hash()}")
        request = _submission(preview["next_requests"]["prepare"])
        admitted = service.post("/api/evidence/prepare", request)
        service.drain()
        assert service.get(f"/api/status?task_id={admitted['task_id']}")["lifecycle"] == "BLOCKED"
    finally:
        service.session.stop()
    denied = admit_official_source(
        network_consent=True, environment={"ALPHALATTICE_NETWORK_DISABLED": "1"}
    )
    assert denied.source is not None
    offline = replace(
        recorded,
        evidence_resources=replace(recorded.evidence_resources, live_source=denied.source),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
        network_access=denied.network_access,
    )
    service = start_service(workspace, offline, tmp_path, clock=lambda: _NOW)
    try:
        tasks_before = service.registry.tasks()
        refused = service.post("/api/evidence/prepare", request)
        assert refused["failure_code"] == "evidence_review.workspace_network_not_allowed"
        assert "task_id" not in refused
        assert service.registry.tasks() == tasks_before
        assert denied.source.network_call_count == 0
    finally:
        service.session.stop()


def test_a_holding_that_filed_nothing_in_the_window_is_named_and_not_packed(
    tmp_path: Path,
) -> None:
    """requirement (W1): the preparation reads each holding's filing index at its
    cutoff before packing. A holding whose index holds nothing in the 30-day
    window is named by the run as nothing filed and packed into no unit; the
    others are packed from the plans that read made, which the Task takes
    without reading an index again. A book where no holding filed anything
    prepares nothing, and says so."""

    from dataclasses import replace as moved
    from uuid import UUID

    def aged(ticker: str) -> list[ScenarioFiling]:
        # The same two periodic reports, accepted a quarter before the cutoff.
        return [
            moved(filing, filed_on="2026-05-01", accepted_at="2026-05-01T20:00:00.000Z")
            for filing in _issuer_filings(ticker)
        ]

    def serve(root: Path, transport: SecScenarioTransport) -> Any:
        workspace, report = build_workspace(root)
        recorded = build_authority(tmp_path=root, report=report, model_authority_admitted=False)
        assert recorded.evidence_resources is not None
        authority = replace(
            recorded,
            evidence_resources=replace(
                recorded.evidence_resources, live_source=SecEdgarSource(transport)
            ),
            evidence_policy=live_evidence_policy(recorded.evidence_policy),
        )
        return start_service(workspace, authority, root, clock=lambda: _NOW)

    transport = _scenario()
    service = serve(tmp_path / "one", transport)
    try:
        query = f"result_hash={service.result_hash()}"
        preview = service.get("/api/evidence/preview?" + query)
        held = sorted(row["entity_id"] for row in preview["source_inventory"]["issuers"])
        quiet = held[-1]
        transport.filings[CIKS[quiet]] = aged(quiet)
        prepared = service.post(
            "/api/evidence/prepare", _submission(preview["next_requests"]["prepare"])
        )
        assert prepared["disposition"] == "ADMITTED", prepared
        service.drain()
        adapter = service.review.evidence_task_adapter
        run = adapter.run_of(service.registry.task(UUID(prepared["task_id"])))
        assert run is not None and run.nothing_filed == (quiet,)
        assert quiet not in run.ordered_entity_ids
        assert sorted(run.ordered_entity_ids) == held[:-1]
        assert transport.inventory_calls == len(held), "each index read once, by the preparation"
        # The book's view reads the run whole: its holdings that filed nothing are
        # its own, so the prepared run is the book's, not a fresh packing.
        assert service.evidence_cro()["state"] == "ANALYST_PACKET_PREPARED"
        check = service.get("/api/evidence/preview?" + query)["source_check"]
        assert check["inventory_request_count"] == 0, "the Task took the plans that read made"
    finally:
        service.session.stop()

    silent = _scenario()
    for ticker in TICKERS:
        silent.filings[CIKS[ticker]] = aged(ticker)
    service = serve(tmp_path / "none", silent)
    try:
        preview = service.get("/api/evidence/preview?" + f"result_hash={service.result_hash()}")
        tasks_before = len(service.registry.tasks())
        nothing = service.post(
            "/api/evidence/prepare", _submission(preview["next_requests"]["prepare"])
        )
        assert nothing["disposition"] == "NOTHING_FILED", nothing
        assert "Having nothing to read is not a finding of no risk." in nothing["detail"]
        assert len(service.registry.tasks()) == tasks_before, "no Task admitted"
    finally:
        service.session.stop()


def test_a_holding_packed_on_its_reservation_that_filed_nothing_leaves_its_unit_s_floor(
    tmp_path: Path,
) -> None:
    """requirement (V587, TE12): a unit's preparation judges its floor as the packing and the
    installer do. A holding whose index could not be read at packing is packed on its
    reservation; when the unit's acquisition reads its index and it filed nothing in the
    window, it leaves the unit's share, while a holding whose filings failed to arrive stays
    in it -- at the strictest floor the unit is refused naming both by their numbers."""

    from dataclasses import replace as moved
    from uuid import UUID

    transport = _scenario()
    workspace, report = build_workspace(tmp_path)
    recorded = build_authority(tmp_path=tmp_path, report=report, model_authority_admitted=False)
    assert recorded.evidence_resources is not None
    authority = replace(
        recorded,
        evidence_resources=replace(
            recorded.evidence_resources,
            live_source=SecEdgarSource(transport),
            minimum_entity_coverage=1.0,
        ),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
    )
    service = start_service(workspace, authority, tmp_path, clock=lambda: _NOW)
    try:
        preview = service.get(f"/api/evidence/preview?result_hash={service.result_hash()}")
        held = sorted(row["entity_id"] for row in preview["source_inventory"]["issuers"])
        failed, quiet = held[0], held[-1]
        transport.filings[CIKS[quiet]] = [
            moved(filing, filed_on="2026-05-01", accepted_at="2026-05-01T20:00:00.000Z")
            for filing in _issuer_filings(quiet)
        ]
        for filing in transport.filings[CIKS[failed]]:
            transport.fail_body(
                CIKS[failed], filing, lambda: RuntimeError("scenario: body refused")
            )
        reads: list[str] = []

        def packing_read_fails(url: str) -> None:
            # The quiet holding's first index read, the preparation's at packing, fails.
            if url.endswith(f"/CIK{CIKS[quiet]}.json"):
                reads.append(url)
                if len(reads) == 1:
                    raise RuntimeError("scenario: index unavailable at packing")

        transport.before_get = packing_read_fails
        prepared = service.post(
            "/api/evidence/prepare", _submission(preview["next_requests"]["prepare"])
        )
        assert prepared["disposition"] == "ADMITTED", prepared
        service.drain()
        adapter = service.review.evidence_task_adapter
        task = service.registry.task(UUID(prepared["task_id"]))
        run = adapter.run_of(task)
        assert run is not None and run.nothing_filed == () and quiet in run.ordered_entity_ids
        assert len(reads) == 2, "read at packing, then by the unit's acquisition"
        (unit,) = adapter.unit_states(task).values()
        counted = len(held) - 1
        assert (unit["state"], unit["failed_stage"]) == ("FAILED", "acquire_source_evidence")
        assert unit["failure_code"] == (
            "alternative_evidence.minimum_entity_coverage_not_met:"
            f"{counted - 1} of {counted} issuers hold a source, {counted} needed, 1 filed nothing"
        )
        assert unit["uncovered_entity_ids"] == (failed,)
    finally:
        service.session.stop()


def _live_service(tmp_path: Path, transport: SecScenarioTransport, now: list[Any]) -> Any:
    """The route's service over the live official source and its controlled
    Analyst and reviewer, on a clock the scene moves."""

    workspace, report = build_workspace(tmp_path)
    recorded = build_authority(tmp_path=tmp_path, report=report)
    assert recorded.evidence_resources is not None
    authority = replace(
        recorded,
        evidence_resources=replace(
            recorded.evidence_resources, live_source=SecEdgarSource(transport)
        ),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
    )
    return start_service(workspace, authority, tmp_path, clock=lambda: now[0]), authority


def test_a_filing_read_once_is_carried_while_it_stays_in_the_window(tmp_path: Path) -> None:
    """requirement (W3, the Analyst's side of its verification): the filing is
    the unit of reuse. A day with no new filing prepares nothing -- no unit, no
    Task, no Analyst -- and the review reads the earlier findings as of that
    day; a day with a new filing reads only it, and the Analyst is told what
    the earlier readings found, never the old filing again."""

    from uuid import UUID

    from alphalattice.control.product_host.composition.evidence_review_projection import (
        EvidenceCroProjector,
    )

    transport = _scenario()
    now = [_NOW]
    service, _authority = _live_service(tmp_path, transport, now)
    try:
        review = service.review
        dispatcher = service.session.dispatcher
        first = review.refresh_evidence(dispatcher=dispatcher, evidence_as_of=_NOW)
        assert first.disposition == "ADMITTED", first
        service.drain()
        adapter = review.evidence_task_adapter
        day_one = adapter.run_of(service.registry.task(first.task_id))
        assert day_one is not None and day_one.read_filings == ()
        held = tuple(sorted(day_one.ordered_entity_ids))
        publication = adapter.published_analysis(first.task_id, now=_NOW, unit_id="u01")

        # The next day, nothing new was filed: each index is read at the new
        # cutoff, every filing in the window was read, nothing is prepared.
        now[0] = transport.retrieved_at = _NOW + timedelta(days=1)
        tasks, bodies = len(service.registry.tasks()), len(transport.body_calls)
        inventory = transport.inventory_calls
        second = review.refresh_evidence(dispatcher=dispatcher, evidence_as_of=now[0])
        assert second.disposition == "NOTHING_NEW", second
        assert len(service.registry.tasks()) == tasks, "no Task: nothing is left to read"
        assert transport.inventory_calls == inventory + len(held)
        assert len(transport.body_calls) == bodies, "no body fetched"
        scope = review.resolve_book(review.default_selector(None)).scope
        carried = review._sealed_run(scope, evidence_as_of=now[0])
        assert carried is not None and carried.units == ()
        assert carried.carried == held and len(carried.read_filings) == 2 * len(held)
        dossier = review._resolve_review_dossier(review.default_selector(None))
        assert not isinstance(dossier, type(second)), dossier
        assert dossier.evidence_as_of == now[0]
        assert [child.unit_id for child in dossier.evidence_children] == ["c01"]
        (child,) = dossier.evidence_children
        assert child.read_as_of == _NOW and sorted(child.ordered_entity_ids) == list(held)
        assert child.analysis_publication_hash == publication.publication.publication_hash
        assert dossier.findings, "the earlier findings carry"
        assert {issuer.entity_id for issuer in dossier.issuers} == set(held)

        # A day later one holding files a current report: only it is read.
        now[0] = transport.retrieved_at = _NOW + timedelta(days=2)
        quiet = held[0]
        new = ScenarioFiling(
            f"{CIKS[quiet]}-26-000200",
            "8-K",
            now[0].date().isoformat(),
            (now[0] - timedelta(hours=1)).isoformat().replace("+00:00", ".000Z"),
            f"{quiet.casefold()}-8k.htm",
        )
        transport.add_filing(CIKS[quiet], new, filing_body(new.accession))
        third = review.refresh_evidence(dispatcher=dispatcher, evidence_as_of=now[0])
        assert third.disposition == "ADMITTED", third
        service.drain()
        day_three = adapter.run_of(service.registry.task(third.task_id))
        assert day_three is not None
        assert [unit.ordered_entity_ids for unit in day_three.units] == [(quiet,)]
        assert day_three.carried == held[1:]
        (unit,) = day_three.units
        assert unit.request.read_accessions(quiet) == {
            value.accession for value in _issuer_filings(quiet)
        }
        assert transport.body_calls_for(new.accession) == 1
        assert all(
            transport.body_calls_for(value.accession) == 1 for value in _issuer_filings(quiet)
        ), "a filing read on the first day is never fetched again"
        lines = EvidenceReviewBundles(review)._earlier_findings(unit.request)
        assert lines and {entity for entity, _text in lines} == {quiet}
        assert all(f"found {_NOW.date().isoformat()}" in text for _entity, text in lines)
        dossier = review._resolve_review_dossier(review.default_selector(None))
        assert not isinstance(dossier, type(third)), dossier
        assert [child.unit_id for child in dossier.evidence_children] == ["c01", "u01"]
        carried_child, read_child = dossier.evidence_children
        assert read_child.ordered_entity_ids == (quiet,)
        assert sorted(carried_child.ordered_entity_ids) == list(held[1:])
        assert UUID(str(third.task_id)) != UUID(str(first.task_id))
        # Z1: the unit read at the cutoff keeps its analysis's expiry, the
        # carried reading the window's; the dossier states the earlier.
        assert read_child.evidence_expires_at == now[0] + timedelta(days=1)
        assert carried_child.evidence_expires_at == LEAVES_WINDOW
        assert dossier.evidence_expires_at == read_child.evidence_expires_at

        # X1: a reader chooses from the publication records and replays only
        # the analyses it returns -- the first day's, expired, is not replayed.
        publications = review.evidence_publications
        replays: list[str] = []
        replay = publications.replay
        publications.replay = lambda value, *, now: replays.append(value) or replay(value, now=now)
        resolved = review.resolve_book(review.default_selector(None))
        ((question, current),) = review.eligible_evidence(resolved)
        (selected,) = review._select_unit_evidence(resolved).values()
        assert selected.evidence == current
        assert review.current_evidence(question=question) == current
        assert (
            review.evidence_selection_label(
                obligation_hash=question.obligation.obligation_hash,
                publication_hash=current.publication.publication_hash,
            )
            == "UNIQUE_CURRENT"
        )
        publications.replay = replay
        assert replays == [current.publication.publication_hash] * 4

        # The day's review holds the first day's reading, carried and expired
        # since: the review is not refused for it, and the book's view reads it.
        assert review.review(dispatcher=dispatcher).disposition == "ADMITTED"
        service.drain()
        reviewed = review.published_review_for_book(resolved.book)
        assert reviewed is not None and reviewed.dossier.evidence_as_of == now[0]
        assert reviewed.dossier.evidence_children[0].read_as_of == _NOW
        projection = EvidenceCroProjector(review).projection()
        assert projection.state == "REVIEW_PUBLISHED", projection.state
        assert projection.evidence_selection == "UNIQUE_CURRENT"
    finally:
        service.session.stop()


def test_carried_findings_keep_an_issuer_whose_new_filing_cannot_be_read(tmp_path: Path) -> None:
    """A failed new source keeps the earlier finding and its issuer visible,
    without counting the issuer as reviewed. The same public route recovers
    after the source is available, keeping every dossier validation.
    """
    from pydantic import ValidationError

    transport = _scenario()
    now = [_NOW]
    service, _authority = _live_service(tmp_path, transport, now)
    try:
        client = LocalResearchClient(service.session.workspace)
        selected = {"result_hash": service.result_hash()}
        first = client.request({"operation": "EVIDENCE_REFRESH", **selected})
        assert first["disposition"] == "ADMITTED", first
        service.drain()
        original = client.request({"operation": "CRO_REVIEW_DOSSIER", **selected})
        assert original["status"] == "CRO_DOSSIER_READY", original
        finding = original["dossier"]["findings"][0]
        changed = finding["affected_entities"][0]
        now[0] = transport.retrieved_at = _NOW + timedelta(days=1)
        new = ScenarioFiling(
            f"{CIKS[changed]}-26-000200",
            "8-K",
            now[0].date().isoformat(),
            (now[0] - timedelta(hours=1)).isoformat().replace("+00:00", ".000Z"),
            f"{changed.casefold()}-unavailable.htm",
        )
        transport.add_filing(CIKS[changed], new, filing_body(new.accession))
        transport.fail_body(
            CIKS[changed], new, lambda: RuntimeError("fixture: new filing unavailable")
        )
        second = client.request({"operation": "EVIDENCE_REFRESH", **selected})
        assert second["disposition"] == "ADMITTED", second
        service.drain()
        exported = client.request({"operation": "CRO_REVIEW_DOSSIER", **selected})
        assert exported.get("status") == "CRO_DOSSIER_READY", exported
        dossier = PortfolioReviewDossier.model_validate(exported["dossier"])
        assert dossier.issuer(changed).review_state == "UNREVIEWED"
        assert any(changed in item.affected_entities for item in dossier.findings)
        assert not any(changed in child.ordered_entity_ids for child in dossier.evidence_children)
        assert dossier.coverage.reviewed_ending_weight_coverage < 1.0
        assert any(changed in reason for reason in dossier.coverage.unavailable_reasons)
        bundle = client.request(
            {
                "operation": "AGENT_BUNDLE_PREPARE",
                "agent_role": "CRO",
                "bundle_directory": str(tmp_path / "cro-bundle"),
                **selected,
            }
        )
        assert bundle["status"] == "AGENT_BUNDLE_READY", bundle
        # The compiler's issuer union repairs its own output, never the contract.
        damaged = dossier.model_dump(mode="json")
        damaged["issuers"] = [row for row in damaged["issuers"] if row["entity_id"] != changed]
        with pytest.raises(ValidationError, match=r"chief_risk_officer\.dossier_entity_invalid"):
            PortfolioReviewDossier.model_validate(damaged)
        stale = client.request(
            {
                "operation": "CRO_REVIEW_DOSSIER",
                **selected,
                "review_dossier_hash": "a" * 64,
                "review_read_at": dossier.evidence_as_of.isoformat(),
            }
        )
        assert stale["refused"].endswith("chief_risk_officer.delivery_continuation_stale")
        transport.heal_body(CIKS[changed], new)
        retry = client.request({"operation": "EVIDENCE_REFRESH", **selected})
        assert retry["disposition"] == "ADMITTED", retry
        service.drain()
        recovered = client.request({"operation": "CRO_REVIEW_DOSSIER", **selected})
        assert recovered["status"] == "CRO_DOSSIER_READY", recovered
        current = PortfolioReviewDossier.model_validate(recovered["dossier"])
        assert current.issuer(changed).review_state in {
            "EXECUTED_NO_FINDINGS",
            "EXECUTED_WITH_FINDINGS",
        }
    finally:
        service.session.stop()


def test_a_raised_risk_carries_forward_and_stays_open_after_its_filing_ages_out(
    tmp_path: Path,
) -> None:
    """requirement (W3, the CRO's side of its verification): the issuers'
    register. A day with no new filing calls no Analyst and no CRO: the last
    review carries forward as of that day, routed as the CRO last assessed it.
    A risk the CRO raised does not age out with its filing: forty days on,
    with every filing out of the window, it still reads open in the book's
    dossier, where the CRO reads it again -- and the register answers by
    issuer, never by book."""

    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        translate_submission,
    )
    from alphalattice.protocols.actor_execution import ActorKind

    transport = _scenario()
    now = [_NOW]
    service, authority = _live_service(tmp_path, transport, now)
    try:
        review = service.review
        dispatcher = service.session.dispatcher
        selector = review.default_selector(None)
        book = review.resolve_book(selector).book
        assert review.refresh_evidence(dispatcher=dispatcher, evidence_as_of=_NOW).disposition == (
            "ADMITTED"
        )
        service.drain()
        assert review.review(dispatcher=dispatcher).disposition == "ADMITTED"
        service.drain()
        first = review.published_review_for_book(book)
        assert first is not None
        (raised,) = first.receipt.submission.material_issues
        assert raised.affected_entities == ("AAPL",) and raised.open_issue_handle is None
        calls = len(authority.review_actor.observed_deadlines)

        # The next day nothing new was filed: no Analyst, no CRO.
        now[0] = transport.retrieved_at = _NOW + timedelta(days=1)
        tasks = len(service.registry.tasks())
        second = review.refresh_evidence(dispatcher=dispatcher, evidence_as_of=now[0])
        assert second.disposition == "NOTHING_NEW", second
        assert "carries forward" in second.detail and "no model call" in second.detail
        service.drain()
        assert len(service.registry.tasks()) == tasks + 1, "one review Task, no evidence Task"
        assert len(authority.review_actor.observed_deadlines) == calls, "the CRO was not called"
        carried = review.published_review_for_book(book)
        assert carried is not None and carried.dossier.evidence_as_of == now[0]
        assert carried.receipt.actor_submission.actor_kind is ActorKind.HOST_FALLBACK
        assert carried.receipt.model_call_count == 0
        assert carried.receipt.submission == translate_submission(
            first.receipt.submission, source=first.dossier, target=carried.dossier
        ), "the CRO's assessment as it sealed it, in the day's handles"
        assert carried.recommendation.route == first.recommendation.route
        # Z1: the carried review states its reading's expiry -- the day the
        # earliest filing it carries leaves the window -- in the recommendation
        # and the export alike, not the analysis's own, one day after its cutoff.
        assert first.dossier.evidence_expires_at == _NOW + timedelta(days=1)
        assert carried.dossier.evidence_expires_at == LEAVES_WINDOW
        assert carried.recommendation.evidence_expires_at == LEAVES_WINDOW
        exported = service.get(
            f"/api/evidence-cro/export?result_hash={service.result_hash()}"
            f"&review_publication_hash={carried.publication.publication_hash}"
        )
        assert "expiring 2026-08-19T21:00:00Z" in exported["html"]
        # The book's view reads the carried review on a day that read no unit
        # (found by Z5: it read "awaiting evidence").
        from alphalattice.control.product_host.composition.evidence_review_projection import (
            EvidenceCroProjector,
        )

        view = EvidenceCroProjector(review).projection()
        assert view.state == "REVIEW_PUBLISHED", view.state
        assert view.review_publication_hash == carried.publication.publication_hash
        assert view.evidence_selection == "UNIQUE_CURRENT"
        (opened,) = carried.dossier.open_issues
        assert opened.raised_on == _NOW.date() and opened.affected_entities == ("AAPL",)
        # A review a day: the book's newest is chosen from the records and read once.
        reads: list[str] = []
        read = review.review_publications.read
        review.review_publications.read = lambda value: reads.append(value) or read(value)
        assert review.published_review_for_book(book) == carried
        review.review_publications.read = read
        assert len(reads) == 1
        report = review.review_publications
        assert (
            report.find_for_basis(
                carried.receipt.review_basis or "", open_issues=carried.dossier.open_issues
            )
            is not None
        )

        # Forty days on every filing has left the window: nothing filed, and the
        # risk still reads, open, where the CRO reads the book again.
        now[0] = transport.retrieved_at = _NOW + timedelta(days=40)
        late = review.refresh_evidence(dispatcher=dispatcher, evidence_as_of=now[0])
        assert late.disposition == "NOTHING_FILED", late
        assert "the CRO reads the book again" in late.detail
        view = EvidenceCroProjector(review).projection()
        assert view.state == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW", view.state
        dossier = review._resolve_review_dossier(selector, evidence_as_of=now[0])
        assert not isinstance(dossier, type(late)), dossier
        (still,) = dossier.open_issues
        assert still.open_issue_handle == opened.open_issue_handle
        assert still.raised_on == _NOW.date()
        assert dossier.issuer("AAPL").review_state == "NOTHING_FILED"
        assert set(still.cited_finding_handles) <= dossier.finding_handles
        assert [child.unit_id for child in dossier.evidence_children] == ["c01"]
        # Carried only for the open issue, which does not age out with its
        # filing: the window from the cutoff bounds the reading (Z1).
        assert dossier.evidence_expires_at == now[0] + timedelta(days=30)
        assert review.review_publications.open_issues(frozenset({"AAPL"}), as_of=now[0])
        assert not review.review_publications.open_issues(frozenset({"MSFT"}), as_of=now[0])

        # The register is kept per review as each is published (X2); reviews
        # published before it was kept are read from their receipts alone,
        # the same register, and a read writes nothing.
        from alphalattice.oversight.chief_risk_officer.publication.portfolio_review import (
            PortfolioReviewPublicationService,
        )

        publications = review.review_publications
        register = publications.store.root / "cro-review-register"
        assert len(list(register.glob("*.json"))) == 2, "one record per review"
        shutil.rmtree(register)
        from_receipts = PortfolioReviewPublicationService(publications.store)
        for entities in ({"AAPL"}, {"MSFT"}):
            assert from_receipts.open_issues(
                frozenset(entities), as_of=now[0]
            ) == publications.open_issues(frozenset(entities), as_of=now[0])
        assert not register.exists()
    finally:
        service.session.stop()


def test_an_open_issue_on_a_reading_of_an_earlier_method_is_read_and_stays_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (AX1, X4 with E2): after the Evidence method changes, every
    earlier analysis reads as superseded, and the risks the CRO raised on them
    stay open in the register. A later review of the book still reads each
    open issue with the findings it rests on, marked as read under the earlier
    method, so the CRO can state it again or resolve it; a review that says
    nothing of it leaves it open. Before, the dossier refused
    `evidence_review_evidence_not_current` for every such book, with no way on."""

    from alphalattice.control.product_host.composition.evidence_review_application import (
        ReviewOutcome,
    )
    from alphalattice.evidence.alternative_evidence.runtime import service as evidence_service
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
        EARLIER_METHOD_NOTE,
    )
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        PortfolioReviewAnswer,
    )
    from alphalattice.oversight.chief_risk_officer.decision.views import render_review_bundle
    from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
        SubmittedPortfolioReviewActor,
    )
    from alphalattice.protocols.actor_execution import ActorKind

    transport = _scenario()
    now = [_NOW]
    workspace, report = build_workspace(tmp_path)

    def authority() -> Any:
        recorded = build_authority(tmp_path=tmp_path, report=report)
        assert recorded.evidence_resources is not None
        return replace(
            recorded,
            evidence_resources=replace(
                recorded.evidence_resources, live_source=SecEdgarSource(transport)
            ),
            evidence_policy=live_evidence_policy(recorded.evidence_policy),
        )

    # Day one, under an earlier build's Evidence method: the CRO raises a risk on AAPL.
    monkeypatch.setattr(
        evidence_service, "alternative_document_publication_binding_hash", lambda _root: "1" * 64
    )
    service = start_service(workspace, authority(), tmp_path, clock=lambda: now[0])
    try:
        review = service.review
        dispatcher = service.session.dispatcher
        selector = review.default_selector(None)
        assert review.refresh_evidence(dispatcher=dispatcher, evidence_as_of=_NOW).disposition == (
            "ADMITTED"
        )
        service.drain()
        assert review.review(dispatcher=dispatcher).disposition == "ADMITTED"
        service.drain()
        (raised,) = review.review_publications.open_issues(
            frozenset({"AAPL"}), as_of=_NOW + timedelta(days=1)
        )
    finally:
        service.session.stop()
    monkeypatch.undo()

    # Forty days on, under the installed method: every filing has left the window,
    # and the reading the issue rests on reads as superseded.
    now[0] = transport.retrieved_at = _NOW + timedelta(days=40)
    service = start_service(workspace, authority(), tmp_path, clock=lambda: now[0])
    try:
        review = service.review
        dispatcher = service.session.dispatcher
        late = review.refresh_evidence(dispatcher=dispatcher, evidence_as_of=now[0])
        assert late.disposition == "NOTHING_FILED", late
        dossier = review._resolve_review_dossier(selector, evidence_as_of=now[0])
        assert not isinstance(dossier, ReviewOutcome), dossier
        (still,) = dossier.open_issues
        assert still.open_issue_handle == raised.open_issue_handle
        earlier = {
            value.finding_handle
            for value in dossier.findings
            if EARLIER_METHOD_NOTE in value.limitations
        }
        assert set(still.cited_finding_handles) <= earlier, "its basis, marked"
        findings = "".join(
            text
            for name, text in render_review_bundle(dossier, task_procedure="").files
            if name.startswith("findings-")
        )
        assert EARLIER_METHOD_NOTE.split(":")[0] in findings

        silent = SubmittedPortfolioReviewActor(
            answer=PortfolioReviewAnswer(), actor_kind=ActorKind.HUMAN, actor_id="ax1-silent"
        )
        assert review.review(actor=silent, dispatcher=dispatcher).disposition == "ADMITTED"
        service.drain()
        (kept,) = review.review_publications.open_issues(
            frozenset({"AAPL"}), as_of=now[0] + timedelta(days=1)
        )
        assert kept.open_issue_handle == raised.open_issue_handle
        assert kept.assessed_on == _NOW.date(), "said nothing of it: counted as last assessed"
    finally:
        service.session.stop()


def test_a_holding_with_nothing_to_cite_carries_the_review(tmp_path: Path) -> None:
    """requirement (X3): the CRO's basis names only the holdings a finding or
    an open issue names, since every risk and every resolution cites findings.
    A holding that filed nothing, then files a report in which nothing is
    found, carries the last review with no model call; a band crossed by a
    holding with a finding is still a new question, one by a holding with
    none -- or its leaving the book -- is not."""

    from dataclasses import replace as moved

    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import review_basis
    from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import ExposureBand
    from alphalattice.protocols.actor_execution import ActorKind
    from tests.alternative_evidence_desk.issuer_listing import ISSUER_TOPICS
    from tests.alternative_evidence_desk.planted_corpus import _CitingActor

    quiet = "TSLA"

    class NothingFoundFor(_CitingActor):
        """The citing Analyst, finding nothing in the quiet holding's filings."""

        def __call__(self, *, packet: Any) -> Any:
            result = super().__call__(packet=packet)
            kept = tuple(value for value in result.answer.findings if value.issuer != quiet)
            return replace(result, answer=result.answer.model_copy(update={"findings": kept}))

    transport = _scenario()
    transport.filings[CIKS[quiet]] = [
        moved(filing, filed_on="2026-05-01", accepted_at="2026-05-01T20:00:00.000Z")
        for filing in _issuer_filings(quiet)
    ]
    now = [_NOW]
    workspace, report = build_workspace(tmp_path)
    recorded = build_authority(tmp_path=tmp_path, report=report)
    assert recorded.evidence_resources is not None
    authority = replace(
        recorded,
        evidence_resources=replace(
            recorded.evidence_resources,
            live_source=SecEdgarSource(transport),
            analysis_actor=NothingFoundFor(topics=ISSUER_TOPICS),
        ),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
    )
    service = start_service(workspace, authority, tmp_path, clock=lambda: now[0])
    try:
        review = service.review
        dispatcher = service.session.dispatcher
        book = review.resolve_book(review.default_selector(None)).book
        assert review.refresh_evidence(dispatcher=dispatcher, evidence_as_of=_NOW).disposition == (
            "ADMITTED"
        )
        service.drain()
        assert review.review(dispatcher=dispatcher).disposition == "ADMITTED"
        service.drain()
        first = review.published_review_for_book(book)
        assert first is not None and first.dossier.issuer(quiet).review_state == "NOTHING_FILED"
        calls = len(authority.review_actor.observed_deadlines)

        # The next day the quiet holding files a current report, read and
        # found to hold nothing: its state moves, and nothing it could be
        # cited by did.
        now[0] = transport.retrieved_at = _NOW + timedelta(days=1)
        filing = ScenarioFiling(
            f"{CIKS[quiet]}-26-000300",
            "8-K",
            now[0].date().isoformat(),
            (now[0] - timedelta(hours=1)).isoformat().replace("+00:00", ".000Z"),
            f"{quiet.casefold()}-8k.htm",
        )
        transport.add_filing(CIKS[quiet], filing, filing_body(filing.accession))
        assert review.refresh_evidence(
            dispatcher=dispatcher, evidence_as_of=now[0]
        ).disposition == ("ADMITTED")
        service.drain()
        assert review.review(dispatcher=dispatcher).disposition == "ADMITTED"
        service.drain()
        carried = review.published_review_for_book(book)
        assert carried is not None and carried.dossier.evidence_as_of == now[0]
        assert carried.dossier.issuer(quiet).review_state == "EXECUTED_NO_FINDINGS"
        assert len(authority.review_actor.observed_deadlines) == calls, "no model call"
        assert carried.receipt.actor_submission.actor_kind is ActorKind.HOST_FALLBACK
        assert carried.recommendation.route == first.recommendation.route

        dossier = carried.dossier
        policy = review._decision_policy().binding_hash
        basis = review_basis(dossier, decision_policy_hash=policy)

        def moved_band(entity: str) -> str:
            issuers = tuple(
                value.model_copy(
                    update={
                        "exposure_band": ExposureBand.CRITICAL
                        if value.exposure_band is not ExposureBand.CRITICAL
                        else ExposureBand.LOW
                    }
                )
                if value.entity_id == entity
                else value
                for value in dossier.issuers
            )
            return review_basis(
                dossier.model_copy(update={"issuers": issuers}), decision_policy_hash=policy
            )

        assert moved_band(quiet) == basis
        assert moved_band("AAPL") != basis
        without = tuple(value for value in dossier.issuers if value.entity_id != quiet)
        assert (
            review_basis(
                dossier.model_copy(update={"issuers": without}), decision_policy_hash=policy
            )
            == basis
        )
    finally:
        service.session.stop()


def test_a_raised_risk_stays_open_whichever_book_or_policy_raised_it(tmp_path: Path) -> None:
    """requirement (X4): the issuers' register is every fresh review's,
    whatever book it reviewed and whatever CRO decision policy sealed it. A
    risk one book's review raised is read by a second book holding the issuer
    in its first dossier after the policy rotated, stays open while reviews say
    nothing of it, and closes when a review of either book resolves it."""

    from alphalattice.control.product_host.composition import evidence_review_application
    from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
        PortfolioResearchSpec,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash
    from alphalattice.oversight.chief_risk_officer.decision import submissions
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
        BookSelector,
    )
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        PortfolioReviewAnswer,
        PortfolioReviewAnswerResolution,
    )
    from alphalattice.oversight.chief_risk_officer.decision.submissions import (
        CROHostPolicyBinding,
    )
    from alphalattice.oversight.chief_risk_officer.runtime import portfolio_review_task
    from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
        SubmittedPortfolioReviewActor,
    )
    from alphalattice.protocols.actor_execution import ActorKind
    from tests.portfolio_strategy_lab.local_web_support import _harness, _run

    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()
    with _harness(portfolio) as harness:
        first = _run(harness, PortfolioResearchSpec.default())
        second = _run(harness, PortfolioResearchSpec.create(cost_bps_per_side="20"))
        report = harness.application.report(first.result_hash)
        workspace = harness.workspace
    assert second.result_hash != first.result_hash
    one, other = (BookSelector(result_hash=value.result_hash) for value in (first, second))

    transport = _scenario()
    now = [_NOW]
    recorded = build_authority(tmp_path=tmp_path, report=report)
    assert recorded.evidence_resources is not None
    authority = replace(
        recorded,
        evidence_resources=replace(
            recorded.evidence_resources, live_source=SecEdgarSource(transport)
        ),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
    )
    service = start_service(workspace, authority, tmp_path, clock=lambda: now[0])
    rotation = pytest.MonkeyPatch()
    try:
        review = service.review
        dispatcher = service.session.dispatcher
        register = review.review_publications
        other_book = review.resolve_book(other).book
        assert "AAPL" in review.resolve_book(other).scope.ordered_entity_ids

        # The first book's review raises a risk on AAPL.
        outcome = review.refresh_evidence(selector=one, dispatcher=dispatcher, evidence_as_of=_NOW)
        assert outcome.disposition == "ADMITTED", outcome
        service.drain()
        assert review.review(selector=one, dispatcher=dispatcher).disposition == "ADMITTED"
        service.drain()
        (raised,) = register.open_issues(frozenset({"AAPL"}), as_of=_NOW + timedelta(days=1))

        # The CRO decision policy rotates, as a change to its owner rotates it;
        # the second book is read the next day.
        current = review._decision_policy()
        values = {
            **current.model_dump(mode="json", exclude={"binding_hash"}),
            "semantics_hash": "1" * 64,
        }
        rotated = CROHostPolicyBinding(**values, binding_hash=canonical_hash(values))
        for module in (submissions, portfolio_review_task, evidence_review_application):
            rotation.setattr(module, "build_portfolio_review_policy_binding", lambda _: rotated)
        now[0] = transport.retrieved_at = _NOW + timedelta(days=1)
        review.refresh_evidence(selector=other, dispatcher=dispatcher, evidence_as_of=now[0])
        service.drain()
        silent = SubmittedPortfolioReviewActor(
            answer=PortfolioReviewAnswer(), actor_kind=ActorKind.HUMAN, actor_id="x4-silent"
        )
        assert review.review(selector=other, actor=silent, dispatcher=dispatcher).disposition == (
            "ADMITTED"
        )
        service.drain()
        read = review.published_review_for_book(other_book)
        assert read is not None and read.receipt.decision_policy_hash == rotated.binding_hash
        (held,) = read.dossier.open_issues
        assert held.open_issue_handle == raised.open_issue_handle, "the other book's risk"
        assert set(held.cited_finding_handles) <= read.dossier.finding_handles
        (still,) = register.open_issues(frozenset({"AAPL"}), as_of=now[0] + timedelta(days=1))
        assert still.open_issue_handle == raised.open_issue_handle
        assert still.assessed_on == _NOW.date(), "said nothing of it: counted as last assessed"

        # A day later AAPL files a report; the first book's review reads it and
        # resolves the issue, under the new policy. (Without the new filing the
        # first book's dossier is the second's question -- the same holdings,
        # bands and findings -- and the second's review carries forward to it.)
        now[0] = transport.retrieved_at = _NOW + timedelta(days=2)
        filing = ScenarioFiling(
            f"{CIKS['AAPL']}-26-000400",
            "8-K",
            now[0].date().isoformat(),
            (now[0] - timedelta(hours=1)).isoformat().replace("+00:00", ".000Z"),
            "aapl-8k-settled.htm",
        )
        transport.add_filing(CIKS["AAPL"], filing, filing_body(filing.accession))
        outcome = review.refresh_evidence(
            selector=one, dispatcher=dispatcher, evidence_as_of=now[0]
        )
        assert outcome.disposition == "ADMITTED", outcome
        service.drain()
        resolving = SubmittedPortfolioReviewActor(
            answer=PortfolioReviewAnswer(
                resolved=(
                    PortfolioReviewAnswerResolution(
                        issue="O1", findings=("F1",), why="The later filing settles it."
                    ),
                )
            ),
            actor_kind=ActorKind.HUMAN,
            actor_id="x4-resolving",
        )
        assert review.review(selector=one, actor=resolving, dispatcher=dispatcher).disposition == (
            "ADMITTED"
        )
        service.drain()
        assert not register.open_issues(frozenset({"AAPL"}), as_of=now[0] + timedelta(days=1))

        # K2: published out of their order -- the last first -- into a store
        # holding only their parts, the three reviews leave the same register:
        # each publication settles the reviews after its own cutoff.
        from alphalattice.evidence.alternative_evidence.publication.artifacts import (
            AlternativeEvidenceArtifactStore,
        )
        from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
            PortfolioReviewPublication,
        )
        from alphalattice.oversight.chief_risk_officer.publication.portfolio_review import (
            PortfolioReviewPublicationService,
        )

        replayed = PortfolioReviewPublicationService(
            AlternativeEvidenceArtifactStore(tmp_path / "replayed")
        )
        for category in (
            "cro-review-dossiers",
            "cro-review-receipts",
            "cro-review-recommendations",
        ):
            shutil.copytree(register.store.root / category, replayed.store.root / category)
        reviews = sorted(
            register.store.values("cro-review-publications", PortfolioReviewPublication),
            key=lambda value: value.published_at,
        )
        assert len(reviews) == 3
        for value in (reviews[2], reviews[0], reviews[1]):
            replayed.publish(publication=value)
        for day in (1, 2, 3):
            as_of = _NOW + timedelta(days=day)
            assert replayed.open_issues(frozenset({"AAPL"}), as_of=as_of) == register.open_issues(
                frozenset({"AAPL"}), as_of=as_of
            ), day
    finally:
        rotation.undo()
        service.session.stop()


def _reads_after(
    tmp_path: Path, days: tuple[int, ...], measured: int, *, forget_days: bool = False
) -> dict[str, Counter[str]]:
    """What a book's view, a preparation and a CRO bundle read on the
    measured day, each in a fresh service over the workspace -- a new
    process -- after a preparation and a review on each of `days`, one
    holding filing a current report a day. The reviewer raises nothing, so
    what is held before the window leaves nothing open behind it. With
    `forget_days`, the records' days (Z2) and the register's index (K2) are
    then removed, as a workspace sealed before they were kept holds none, and
    the view is read twice more: what each read learned is `days`."""

    from alphalattice.control.product_host.composition.evidence_review_projection import (
        EvidenceCroProjector,
    )
    from alphalattice.evidence.alternative_evidence.publication.analysis import (
        AlternativeEvidenceAnalysisPublicationService,
    )
    from alphalattice.evidence.alternative_evidence.publication.artifacts import (
        AlternativeEvidenceArtifactStore,
    )
    from alphalattice.protocols.actor_execution import ActorKind
    from tests.alternative_evidence_desk.review_dossiers import CitingReviewActor

    transport = _scenario()
    now = [_NOW]
    workspace, report = build_workspace(tmp_path)

    def authority() -> Any:
        # A fresh runtime over the same evidence directory: a new process's caches.
        recorded = build_authority(tmp_path=tmp_path, report=report)
        assert recorded.evidence_resources is not None
        return replace(
            recorded,
            evidence_resources=replace(
                recorded.evidence_resources, live_source=SecEdgarSource(transport)
            ),
            evidence_policy=live_evidence_policy(recorded.evidence_policy),
            review_actor=CitingReviewActor(actor_kind=ActorKind.HUMAN, actor_id="x2-guard"),
        )

    def file_a_report(day: int) -> None:
        ticker = TICKERS[day % len(TICKERS)]
        filing = ScenarioFiling(
            f"{CIKS[ticker]}-26-{900 + day:06d}",
            "8-K",
            now[0].date().isoformat(),
            (now[0] - timedelta(hours=1)).isoformat().replace("+00:00", ".000Z"),
            f"{ticker.casefold()}-8k-{day}.htm",
        )
        transport.add_filing(CIKS[ticker], filing, filing_body(filing.accession))

    service = start_service(workspace, authority(), tmp_path, clock=lambda: now[0])
    try:
        for day in days:
            now[0] = transport.retrieved_at = _NOW + timedelta(days=day)
            if day:
                file_a_report(day)
            dispatcher = service.session.dispatcher
            assert service.review.refresh_evidence(
                dispatcher=dispatcher, evidence_as_of=now[0]
            ).disposition in {"ADMITTED", "NOTHING_NEW"}
            service.drain()
            assert service.review.review(dispatcher=dispatcher).disposition == "ADMITTED"
            service.drain()
    finally:
        service.session.stop()

    now[0] = transport.retrieved_at = _NOW + timedelta(days=measured)
    file_a_report(measured)
    counts: dict[str, Counter[str]] = {}
    counted: Counter[str] = Counter()
    load, load_set, verify = (
        AlternativeEvidenceArtifactStore.load,
        AlternativeEvidenceArtifactStore.load_source_set,
        (AlternativeEvidenceAnalysisPublicationService.verify),
    )

    def counted_load(self: Any, category: str, identity: str, model: Any) -> Any:
        counted[category] += 1
        return load(self, category, identity, model)

    def counted_set(self: Any, identity: str) -> Any:
        counted["source-document-sets"] += 1
        return load_set(self, identity)

    def counted_verify(self: Any, publication_hash: str) -> Any:
        counted["replay"] += 1
        return verify(self, publication_hash)

    for read in ("view", "preparation", "cro_bundle"):
        service = start_service(workspace, authority(), tmp_path, clock=lambda: now[0])
        review = service.review
        counted.clear()
        try:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(AlternativeEvidenceArtifactStore, "load", counted_load)
                patch.setattr(AlternativeEvidenceArtifactStore, "load_source_set", counted_set)
                patch.setattr(
                    AlternativeEvidenceAnalysisPublicationService, "verify", counted_verify
                )
                if read == "view":
                    EvidenceCroProjector(review).projection()
                elif read == "preparation":
                    review.refresh_evidence(
                        dispatcher=service.session.dispatcher, evidence_as_of=now[0]
                    )
                    service.drain()
                    runtime = review.evidence_task_adapter.runtime
                    counted["readings_scanned"] = runtime.sealed_readings.scanned
                    counted["excerpts_scanned"] = runtime.sealed_excerpts.scanned
                    counted["canonicals_scanned"] = runtime.sealed_canonicals.scanned
                    counted["retained_scanned"] = (
                        runtime.local_sources.scanned_sets + runtime.local_sources.scanned_commits
                    )
                else:
                    assert isinstance(
                        EvidenceReviewBundles(review).prepare_agent_bundle(
                            role="CRO", selector=None, directory=str(tmp_path / "cro")
                        ),
                        dict,
                    )
                counted["days_learned"] = review.review_publications.learned + sum(
                    days.learned
                    for days in review.evidence_task_adapter.runtime.record_days.values()
                )
                counted["register"] = review.review_publications.records_read
        finally:
            service.session.stop()
        counts[read] = Counter(
            {
                key: value
                for key, value in counted.items()
                if key
                in {
                    "replay",
                    "readings_scanned",
                    "requests",
                    "evidence-coverage-runs",
                    "cro-review-receipts",
                    "cro-review-recommendations",
                    "cro-review-dossiers",
                    "cro-review-publications",
                    "analysis-publications",
                    "snapshots",
                    "document-sets",
                    "source-document-sets",
                    "source-document-commits",
                    "excerpts_scanned",
                    "canonicals_scanned",
                    "retained_scanned",
                    "days_learned",
                    "register",
                }
            }
        )
    if forget_days:
        learned: Counter[str] = Counter()
        for read in ("relearn", "relearned"):
            service = start_service(workspace, authority(), tmp_path, clock=lambda: now[0])
            try:
                runtime = service.review.evidence_task_adapter.runtime
                if read == "relearn":
                    shutil.rmtree(runtime.artifacts.root / "record-days")
                    shutil.rmtree(runtime.artifacts.root / "cro-review-register" / "days")
                EvidenceCroProjector(service.review).projection()
                learned[read] = service.review.review_publications.learned + sum(
                    days.learned for days in runtime.record_days.values()
                )
            finally:
                service.session.stop()
        counts["days"] = learned
    return counts


def test_a_day_reads_the_same_however_many_days_the_workspace_holds(tmp_path: Path) -> None:
    """requirement (X2, Z2, K2): what a book's view, a preparation and a CRO
    bundle read -- analyses replayed, reviews read for the register and the
    basis and the register's own records, the readings ledger's scan, runs
    and requests, the analysis and
    review records a choice is made from, and the preparation's reuse
    indexes of snapshots, document and source sets and commits -- does not
    grow with the days a workspace holds. The same two recent days, measured
    the day after, read the same whether or not two days forty days earlier
    were prepared and reviewed too; only the names of the records are listed.
    The records' days are derived: a workspace that holds none reads each
    record once to learn its day, and the next read is as above."""

    recent = (40, 41)
    alone = _reads_after(tmp_path / "recent", recent, 42)
    held = _reads_after(tmp_path / "held", (0, 1, *recent), 42, forget_days=True)
    days = held.pop("days")
    assert held == alone, (alone, held)
    assert days["relearn"] > 0 and days["relearned"] == 0, days
    assert alone["preparation"]["readings_scanned"] >= 1
    assert alone["view"]["replay"] >= 1 and alone["cro_bundle"]["replay"] >= 1
    assert alone["view"]["register"] >= 1 and alone["cro_bundle"]["register"] >= 1
    assert alone["preparation"]["excerpts_scanned"] >= 1
    assert alone["preparation"]["retained_scanned"] >= 1
    assert all(value["days_learned"] == 0 for value in alone.values())


def test_a_short_unit_is_delivered_whole_without_an_index(tmp_path: Path) -> None:
    """requirement (W4): a unit whose filings together fit the bundle's file
    bound is delivered whole. Its generation commits to the filings' bytes and
    builds no index -- no passage embedded, no session opened, no question run,
    nothing reranked -- and its pieces together are each filing's canonical
    text, unchanged; the Analyst's bundle says the filings are whole."""

    from alphalattice.evidence.alternative_evidence.analysis.views import render_analyst_bundle
    from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
        WholeFilingsGeneration,
    )

    transport = _scenario()
    now = [_NOW]
    service, _authority = _live_service(tmp_path, transport, now)
    try:
        review = service.review
        adapter = review.evidence_task_adapter
        runtime = adapter.runtime
        opened: list[object] = []
        open_session = runtime.open_session
        runtime.open_session = lambda **values: opened.append(values) or open_session(**values)
        prepared = review.refresh_evidence(
            dispatcher=service.session.dispatcher, evidence_as_of=_NOW, prepare_only=True
        )
        assert prepared.disposition == "ADMITTED", prepared
        service.drain()
        task = service.registry.task(prepared.task_id)
        assert isinstance(adapter._generation(task, ONE_UNIT), WholeFilingsGeneration)
        assert runtime.retrieval.passage_embedding_pass_count == 0
        assert runtime.retrieval.embedded_chunk_count == 0
        assert opened == [], "no session opens over a unit delivered whole"
        packet = adapter.prepared_packet(prepared.task_id, now=_NOW, unit_id=ONE_UNIT)
        receipt = packet.receipt
        assert receipt.whole_filings is not None and receipt.queries == ()
        assert receipt.search_call_count == 0 and receipt.routing is None
        for reference in packet.document_set.documents:
            _revision, content = runtime.documents.library.read_revision(
                reference.workspace_document_id, reference.workspace_revision
            )
            delivered = "".join(
                span.excerpt
                for span in packet.spans
                if span.document_handle == reference.semantic_handle
            )
            assert delivered == content.decode("utf-8"), "the filing, whole and unchanged"
        index = dict(render_analyst_bundle(packet, task_procedure="").files)["README.md"]
        assert "Each filing is here whole" in index
    finally:
        service.session.stop()


def test_a_bundle_on_a_book_with_no_run_is_answered_as_it_was_read(tmp_path: Path) -> None:
    """regression (V255, EV1, OP6): a book with no sealed run of its own reads its dossier at
    the time it is read, the open issue another book's review raised carried into it. That
    cutoff was the clock, so the dossier moved between the bundle and its answer and every
    answer was refused stale (both AX1f runs); the bundle seals the time it read the dossier,
    and the answer and its Task's admission and publication meet that dossier.
    requirement (V256, OP10): a bundle names its book's last
    review with how it reads and under which CRO policy, never by a hash."""

    from uuid import UUID

    from alphalattice.control.task_control.contracts import TaskLifecycle
    from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
        PortfolioResearchSpec,
    )
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
        BookSelector,
    )
    from tests.alternative_evidence_desk.review_http_support import _operation
    from tests.portfolio_strategy_lab.local_web_support import _harness, _run

    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()
    with _harness(portfolio) as harness:
        first = _run(harness, PortfolioResearchSpec.default())
        second = _run(harness, PortfolioResearchSpec.create(cost_bps_per_side="20"))
        report = harness.application.report(first.result_hash)
        workspace = harness.workspace
    one, other = (BookSelector(result_hash=value.result_hash) for value in (first, second))
    transport = _scenario()
    now = [_NOW]
    recorded = build_authority(tmp_path=tmp_path, report=report)
    assert recorded.evidence_resources is not None
    authority = replace(
        recorded,
        evidence_resources=replace(
            recorded.evidence_resources, live_source=SecEdgarSource(transport)
        ),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
    )
    service = start_service(workspace, authority, tmp_path, clock=lambda: now[0])
    try:
        review = service.review
        dispatcher = service.session.dispatcher
        outcome = review.refresh_evidence(selector=one, dispatcher=dispatcher, evidence_as_of=_NOW)
        assert outcome.disposition == "ADMITTED", outcome
        service.drain()
        assert review.review(selector=one, dispatcher=dispatcher).disposition == "ADMITTED"
        service.drain()

        # The installed CRO policy moves with no recorded move: book one's review reads back
        # exactly, under an earlier policy, and its next bundle says so.
        review.review_publications._policy = "f" * 64
        again = _operation(
            service,
            {
                "operation": "AGENT_BUNDLE_PREPARE",
                "agent_role": "CRO",
                "result_hash": first.result_hash,
                "bundle_directory": str(tmp_path / "cro-one"),
            },
        )
        assert again["status"] == "AGENT_BUNDLE_READY", again
        index = next(item["text"] for item in again["files"] if item["name"] == "README.md")
        assert "## The book's last review" in index
        assert f"Published {_NOW.date().isoformat()}, on evidence as of" in index
        assert "sealed, under an earlier CRO policy" in " ".join(index.split())
        review.review_publications._policy = None  # the installed policy again

        now[0] = transport.retrieved_at = _NOW + timedelta(days=1)
        assert review._sealed_run(review.resolve_book(other).scope, evidence_as_of=None) is None
        directory = tmp_path / "cro"
        # A position basis beside a result's book is refused by its name, and the same
        # request without it is offered (V290).
        request = {
            "operation": "AGENT_BUNDLE_PREPARE",
            "agent_role": "CRO",
            "result_hash": second.result_hash,
            "bundle_directory": str(directory),
        }
        stray = _operation(service, {**request, "position_basis": "OBSERVED_RESEARCH_ENTRY"})
        assert stray["failure_code"].endswith("update_selector_invalid:position_basis"), stray
        assert stray["next_requests"] == {"without_position_basis": request}
        prepared = _operation(
            service,
            {
                "operation": "AGENT_BUNDLE_PREPARE",
                "agent_role": "CRO",
                "result_hash": second.result_hash,
                "bundle_directory": str(directory),
            },
        )
        assert prepared["status"] == "AGENT_BUNDLE_READY", prepared
        bundle = EvidenceReviewBundles(review).agent_bundle(str(directory))
        assert bundle is not None
        prepared_dossier_hash = bundle.submission["review_dossier_hash"]
        now[0] += timedelta(seconds=11)  # AX1f's two reads were 11 seconds apart
        answered = _operation(
            service,
            {
                "operation": "AGENT_ANSWER_SUBMIT",
                "bundle_directory": str(directory),
                "agent_answer": {
                    "risks": [],
                    "summary": "Nothing new in the evidence read.",
                    "read": [item["name"] for item in prepared["files"]],
                },
            },
        )
        assert answered.get("failure_code") != "agent_bundle.preparation_stale", answered
        assert answered["status"] == "ACCEPTED", answered
        service.drain()
        task = service.registry.task(UUID(answered["task_id"]))
        assert task.lifecycle is TaskLifecycle.SUCCEEDED, task.failure_code
        publication = review.review_publications.find_for_review_key(task.input.input_hash)
        assert publication is not None
        assert publication.publication.dossier_hash == prepared_dossier_hash

    finally:
        service.session.stop()
