"""The official history shards through the real HTTP transport, offline: the
one path shape the discovery needs is admitted, every other restriction of
the official boundary stands, and a listed shard is requested only when it
is named for the issuer being discovered. Backed by `httpx.MockTransport`
responses -- no network, no scenario transport.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
    SecIssuerRegistryEntry,
    SecIssuerRegistrySnapshot,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.sources.sec_edgar import (
    HttpxSecOfficialTransport,
    SecEdgarSource,
)

AAPL = "0000320193"
MSFT = "0000789019"
NOW = datetime(2026, 8, 12, 14, 0, tzinfo=UTC)
USER_AGENT = "AlphaLattice QA qa@example.com"
SHARD = f"https://data.sec.gov/submissions/CIK{AAPL}-submissions-001.json"


def _columns(rows: list[tuple[str, str, str, str, str, str]]) -> dict[str, list[str]]:
    keys = (
        "accessionNumber",
        "filingDate",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
        "reportDate",
    )
    return {key: [row[index] for row in rows] for index, key in enumerate(keys)}


RECENT = _columns(
    [
        (
            f"{AAPL}-26-000070",
            "2026-07-31",
            "2026-07-31T20:05:00.000Z",
            "8-K",
            "aapl-8k.htm",
            "2026-07-31",
        ),
        (
            f"{AAPL}-26-000050",
            "2026-05-02",
            "2026-05-02T20:00:00.000Z",
            "10-Q",
            "aapl-10q.htm",
            "2026-03-28",
        ),
    ]
)
OLDER = _columns(
    [
        (
            f"{AAPL}-25-000100",
            "2025-11-01",
            "2025-11-01T21:00:00.000Z",
            "10-K",
            "aapl-10k.htm",
            "2025-09-27",
        )
    ]
)


def _transport(handler) -> HttpxSecOfficialTransport:
    return HttpxSecOfficialTransport(
        user_agent=USER_AGENT,
        maximum_attempts=1,
        transport=httpx.MockTransport(handler),
        monotonic=lambda: 0.0,
        sleeper=lambda _seconds: None,
    )


def _json(payload: object, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, headers={"content-type": "application/json"}, content=json.dumps(payload).encode()
    )


def test_the_real_transport_admits_the_official_shard_path_and_nothing_wider() -> None:
    """The real transport admits the official shard path and nothing wider."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("-submissions-001.json"):
            return _json({"filings": {"recent": OLDER}})
        if request.url.path.endswith("-submissions-002.json"):
            return httpx.Response(302, headers={"location": SHARD})
        return _json({"unexpected": True})

    transport = _transport(handler)
    try:
        response = transport.get(SHARD, maximum_bytes=100_000)
        assert response.status_code == 200
        assert json.loads(response.content)["filings"]["recent"] == OLDER
        # The serialized request carries the official identifier and the
        # contact header, nothing else.
        assert [str(value.url) for value in seen] == [SHARD]
        assert seen[0].headers["user-agent"] == USER_AGENT
        assert not seen[0].url.query and seen[0].method == "GET"

        for url in (
            f"http://data.sec.gov/submissions/CIK{AAPL}-submissions-001.json",
            f"https://data.sec.gov.example.com/submissions/CIK{AAPL}-submissions-001.json",
            f"https://www.sec.gov/submissions/CIK{AAPL}-submissions-001.json",
            f"https://user:pw@data.sec.gov/submissions/CIK{AAPL}-submissions-001.json",
            f"https://data.sec.gov:8443/submissions/CIK{AAPL}-submissions-001.json",
            f"https://data.sec.gov/submissions/CIK{AAPL}-submissions-001.json?x=1",
            f"https://data.sec.gov/submissions/CIK{AAPL}-submissions-001.json#f",
            f"https://data.sec.gov/submissions/CIK{AAPL}-submissions-1.json",
            f"https://data.sec.gov/submissions/CIK{AAPL}-submissions-0001.json",
            f"https://data.sec.gov/submissions/CIK{AAPL[:9]}-submissions-001.json",
            f"https://data.sec.gov/submissions/CIK{AAPL}-submission-001.json",
            f"https://data.sec.gov/submissions/../CIK{AAPL}-submissions-001.json",
            f"https://data.sec.gov/submissions/CIK{AAPL}-submissions-001.json/",
        ):
            with pytest.raises(ValueError, match="non_official_url_rejected"):
                transport.get(url, maximum_bytes=100_000)
        assert len(seen) == 1, "no refused URL reached the wire"
        # A redirect is not followed: the official boundary is one response.
        with pytest.raises(ValueError, match="sec_http_status_invalid"):
            transport.get(SHARD.replace("-001", "-002"), maximum_bytes=100_000)
    finally:
        transport.close()


@pytest.mark.parametrize("status", [403, 503])
def test_a_setup_refusal_retains_the_official_transport_status_and_sanitizes_its_chain(
    status, monkeypatch, capsys, tmp_path
) -> None:
    """regression: direct and retry-exhausted SEC failures preserve the HTTP status through
    the setup entry, without leaking a response body, contact header or exception text."""
    from scripts import materialize_evidence_cro_authority as setup

    transport = _transport(
        lambda _request: httpx.Response(
            status,
            headers={"content-type": "application/json", "x-private": "NEVER-PRINT-ME"},
            text="NEVER-PRINT-ME",
        )
    )

    def acquire(_arguments):
        return transport.get(SHARD, maximum_bytes=100_000)

    monkeypatch.setattr(setup, "materialize", acquire)
    try:
        assert (
            setup.main(
                [
                    "--workspace",
                    str(tmp_path),
                    "--acquire-sec",
                    "--entities",
                    "AAPL",
                    "--network-consent",
                ]
            )
            == 2
        )
    finally:
        transport.close()
    payload = json.loads(capsys.readouterr().out)
    code = (
        "alternative_evidence.sec_http_status_invalid:403"
        if status == 403
        else ("alternative_evidence.sec_acquisition_failed")
    )
    assert payload["failure_code"] == code
    assert any(cause["failure_code"].endswith(f":{status}") for cause in payload["causes"])
    assert payload["detail"] and payload["next_action"] and payload["next_requests"]
    assert payload["context"]["source"] == "SEC" and payload["context"]["issuers"] == ["AAPL"]
    assert "--preflight" in payload["next_commands"]["preflight"]
    shown = json.dumps(payload)
    assert "NEVER-PRINT-ME" not in shown and USER_AGENT not in shown


def _request() -> AlternativeEvidenceRequest:
    return seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        ordered_entity_ids=("AAPL",),
        evidence_as_of=NOW,
        acquisition_deadline=NOW + timedelta(hours=1),
        evidence_classes=(AlternativeEvidenceClass.SEC_FILING,),
        source_policy=AlternativeEvidenceSourcePolicy(maximum_documents_per_issuer=3),
        ttl_seconds=86_400,
        mode=AlternativeEvidenceMode.LIVE_OFFICIAL,
    )


def _registry() -> SecIssuerRegistrySnapshot:
    return seal_contract(
        SecIssuerRegistrySnapshot,
        "registry_hash",
        captured_at=NOW - timedelta(days=1),
        entries=(
            SecIssuerRegistryEntry(
                entity_id="AAPL", ticker="AAPL", cik=AAPL, legal_name="Apple Inc."
            ),
        ),
        source_content_hash="c" * 64,
    )


def test_campaign_budgets_refuse_before_the_request_and_count_every_byte_read() -> None:
    """Campaign budgets refuse before the request and count every byte read."""

    served: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        served.append(request.url.path)
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"x" * 600)

    transport = HttpxSecOfficialTransport(
        user_agent=USER_AGENT,
        maximum_attempts=3,
        transport=httpx.MockTransport(handler),
        monotonic=lambda: 0.0,
        sleeper=lambda _seconds: None,
        maximum_total_attempts=2,
        maximum_total_response_bytes=1_000,
    )
    body = f"https://www.sec.gov/Archives/edgar/data/320193/{'0' * 18}/a.htm"
    try:
        first = transport.get(body, maximum_bytes=700)
        assert len(first.content) == 600 and transport.response_bytes == 600
        assert transport.attempts_remaining == 1 and transport.response_bytes_remaining == 400
        # 400 remain and the request may read 700: refused before the wire.
        with pytest.raises(ValueError, match="sec_campaign_byte_budget_exhausted"):
            transport.get(body.replace("a.htm", "b.htm"), maximum_bytes=700)
        assert transport.retry_count == 0, "a budget refusal is never retried"
        assert transport.budget_refusals["bytes"] == 1
        assert transport.attempts_remaining == 1, "a refusal consumes no attempt"
        # A cap that fits is admitted; the source's declared length exceeds
        # it, so nothing is read and the attempt is spent.
        with pytest.raises(ValueError, match="sec_response_too_large:600"):
            transport.get(body.replace("a.htm", "c.htm"), maximum_bytes=400)
        assert transport.response_bytes == 600 and transport.attempts_remaining == 0
        with pytest.raises(ValueError, match="sec_campaign_attempt_budget_exhausted"):
            transport.get(body.replace("a.htm", "d.htm"), maximum_bytes=100)
        prefix = "/Archives/edgar/data/320193/" + "0" * 18
        assert served == [prefix + "/a.htm", prefix + "/c.htm"], "the refusals never hit the wire"
        assert transport.budget_refusals == {"attempts": 1, "bytes": 1}
    finally:
        transport.close()


def test_the_body_resource_budget_refuses_a_new_body_before_any_request() -> None:
    """requirement (section 4): the source bounds distinct filing bodies; the
    body beyond the bound is refused by name before a request, and the
    refusal is the resource's outcome while its siblings stand."""

    from tests.alternative_evidence_desk.incremental_acquisition_support import (
        EIGHT_K,
        TEN_K,
        TEN_Q,
        _admission,
    )
    from tests.alternative_evidence_desk.planted_corpus import _runtime
    from tests.alternative_evidence_desk.sec_scenario_transport import (
        SecScenarioTransport,
        filing_body,
    )

    filings = {AAPL: [EIGHT_K, TEN_Q, TEN_K]}
    scenario = SecScenarioTransport(
        registry={"AAPL": (AAPL, "Apple Inc.")},
        filings=filings,
        bodies={
            SecScenarioTransport.locator(AAPL, f): filing_body(f.accession) for f in filings[AAPL]
        },
    )
    source = SecEdgarSource(scenario, maximum_body_resources=2)
    request = _request()
    with tempfile.TemporaryDirectory() as tmp:
        runtime = _runtime(Path(tmp))
        try:
            _r, snapshot, _source_set = runtime.acquire_live(
                request=request,
                admission=_admission(request),
                source=source,
                published_at=NOW,
                clock=lambda: NOW,
            )
        finally:
            runtime.close()
    outcomes = {v.accession: (v.outcome, v.detail) for v in snapshot.acquisition.documents}
    assert len(scenario.body_calls) == 2 and len(source.body_resources) == 2
    refused = [k for k, (o, d) in outcomes.items() if o == "FAILED"]
    assert len(refused) == 1
    assert "sec_campaign_body_budget_exhausted" in outcomes[refused[0]][1]
    assert sum(1 for o, _d in outcomes.values() if o == "FETCHED") == 2
    assert snapshot.acquisition.body_request_count == 2
