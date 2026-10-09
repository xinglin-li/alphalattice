"""One campaign's budget across the processes that spend it, offline: the
ledger the official transport reserves against before every request and
settles after it, resumed by a later process with only what remains,
refusing before another admitted request could exceed it, and refusing a
record it cannot trust. Backed by `httpx.MockTransport` -- no network.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from alphalattice.control.workspace_runtime.network_access import set_network_access
from alphalattice.evidence.alternative_evidence.sources.admission import (
    SEC_USER_AGENT_VARIABLE,
    admit_official_source,
    campaign_ledger_path,
)
from alphalattice.evidence.alternative_evidence.sources.campaign import (
    SecCampaignDeclaration,
    SecCampaignLedger,
)
from alphalattice.evidence.alternative_evidence.sources.sec_edgar import (
    HttpxSecOfficialTransport,
    SecEdgarSource,
)
from alphalattice.kernel.shared_kernel.environment import OFFLINE_SWITCH

USER_AGENT = "AlphaLattice QA qa@example.com"
ARCHIVE = f"https://www.sec.gov/Archives/edgar/data/320193/{'0' * 18}"
INVENTORY = "https://data.sec.gov/submissions/CIK0000320193.json"
DECLARATION = SecCampaignDeclaration(
    campaign_id="qa-campaign",
    maximum_total_attempts=4,
    maximum_total_response_bytes=1_500,
    maximum_body_resources=2,
    maximum_document_bytes=600,
)


class _Chunks(httpx.SyncByteStream):
    """A body the client yields in the chunks given, with no declared length."""

    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks


def _serve(served: list[str], *, size: int = 500) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        served.append(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"x" * size)

    return httpx.MockTransport(handler)


def _transport(ledger: SecCampaignLedger, served: list[str]) -> HttpxSecOfficialTransport:
    return HttpxSecOfficialTransport(
        user_agent=USER_AGENT,
        maximum_attempts=1,
        transport=_serve(served),
        monotonic=lambda: 0.0,
        sleeper=lambda _seconds: None,
        ledger=ledger,
    )


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(text) for text in path.read_text(encoding="utf-8").split("\n") if text]


def test_a_resumed_campaign_spends_only_its_remainder_and_refuses_before_exceeding_it(
    tmp_path: Path,
) -> None:
    """requirement (Phase A4): the campaign's totals and what it consumed --
    attempts, decoded bytes, distinct bodies -- survive a process restart;
    the process that resumes it receives only the remainder, continues
    within it, and is refused before a request whose cap does not fit the
    remainder is made. The admission line carries the declared totals and
    another declaration under the same id is another campaign."""

    path = tmp_path / "qa-campaign.jsonl"
    served: list[str] = []

    first = SecCampaignLedger.open(path, declaration=DECLARATION)
    assert first.resumed is False and first.summary()["bytes_remaining"] == 1_500
    with _transport(first, served) as transport:
        response = transport.get(f"{ARCHIVE}/a.htm", maximum_bytes=600)
        assert len(response.content) == 500
        assert transport.attempts_remaining == 3 and transport.response_bytes_remaining == 1_000
    lines = _lines(path)
    assert [line["kind"] for line in lines] == ["ADMISSION", "RESERVE", "SETTLE"]
    assert lines[1]["reserved_bytes"] == 600 and lines[1]["resource"] == "body"
    assert lines[2]["actual_bytes"] == 500 and lines[2]["outcome"] == "complete"

    # Another process: the same declaration resumes the campaign with what
    # remains, not a fresh full budget.
    second = SecCampaignLedger.open(path, declaration=DECLARATION)
    assert second.resumed is True
    assert second.attempts_consumed == 1 and second.bytes_consumed == 500
    assert second.remaining_attempts == 3 and second.remaining_bytes == 1_000
    assert second.body_resources == frozenset({f"{ARCHIVE}/a.htm"})
    with _transport(second, served) as transport:
        transport.get(f"{ARCHIVE}/b.htm", maximum_bytes=600)
        assert transport.response_bytes_remaining == 500 and transport.attempts_remaining == 2
        # 500 remain and the next request may read 600: refused before it
        # is made, and never retried.
        with pytest.raises(ValueError, match="sec_campaign_byte_budget_exhausted"):
            transport.get(f"{ARCHIVE}/c.htm", maximum_bytes=600)
        assert transport.budget_refusals == {"attempts": 0, "bytes": 1}
        assert transport.retry_count == 0
        # A request whose cap fits the remainder is still admitted.
        transport.get(INVENTORY, maximum_bytes=500)
        assert transport.response_bytes_remaining == 0
    assert served == ["a.htm", "b.htm", "CIK0000320193.json"], "the refusal never hit the wire"
    third = SecCampaignLedger.open(path, declaration=DECLARATION)
    assert third.summary() == {
        "campaign_id": "qa-campaign",
        "resumed": True,
        "attempts_consumed": 3,
        "attempts_remaining": 1,
        "bytes_consumed": 1_500,
        "bytes_remaining": 0,
        "unsettled_reservations": 0,
        "overrun_bytes": 0,
        "body_resources": 2,
        "body_resources_remaining": 0,
        "lines": 7,
    }
    with (
        _transport(third, served) as transport,
        pytest.raises(ValueError, match="sec_campaign_byte_budget_exhausted"),
    ):
        transport.get(INVENTORY, maximum_bytes=1)
    assert len(served) == 3

    # The same id with other totals is another campaign, not a widening.
    with pytest.raises(ValueError, match="sec_campaign_declaration_mismatch"):
        SecCampaignLedger.open(
            path,
            declaration=SecCampaignDeclaration(
                campaign_id="qa-campaign",
                maximum_total_attempts=4,
                maximum_total_response_bytes=2_000,
                maximum_body_resources=2,
                maximum_document_bytes=600,
            ),
        )


def test_an_unsettled_reservation_counts_at_its_cap_and_a_partial_transfer_at_what_was_read(
    tmp_path: Path,
) -> None:
    """requirement (Phase A4, crash boundary): a reservation no process
    settled -- the process died mid-transfer -- counts at the cap it
    reserved when the campaign is resumed, never at zero; a transfer that
    fails part-way is settled with the bytes actually read, the chunk that
    crossed the cap included, and the counters report the overrun rather
    than a clamped figure."""

    path = tmp_path / "qa-campaign.jsonl"
    ledger = SecCampaignLedger.open(path, declaration=DECLARATION)
    sequence = ledger.reserve(url=f"{ARCHIVE}/a.htm", resource="body", maximum_bytes=600)
    assert sequence == 1 and ledger.bytes_consumed == 600 and ledger.unsettled == 1
    ledger.close()  # the process died mid-transfer: its ownership went with it

    resumed = SecCampaignLedger.open(path, declaration=DECLARATION)
    assert resumed.unsettled == 1 and resumed.bytes_consumed == 600
    assert resumed.remaining_bytes == 900 and resumed.remaining_attempts == 3
    assert resumed.body_resources == frozenset({f"{ARCHIVE}/a.htm"})
    with pytest.raises(ValueError, match="sec_campaign_settlement_unknown"):
        resumed.settle(7, actual_bytes=0, outcome="failed")

    # A body served in 250-byte chunks with no declared length under a
    # 600-byte cap: read 250, 500, 750 -- refused at the chunk that crossed
    # the cap, with all 750 bytes counted.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/html"}, stream=_Chunks((b"y" * 250,) * 4)
        )

    transport = HttpxSecOfficialTransport(
        user_agent=USER_AGENT,
        maximum_attempts=1,
        transport=httpx.MockTransport(handler),
        monotonic=lambda: 0.0,
        sleeper=lambda _seconds: None,
        ledger=resumed,
    )
    with transport:
        with pytest.raises(ValueError, match="sec_response_too_large:750"):
            transport.get(f"{ARCHIVE}/b.htm", maximum_bytes=600)
        assert transport.response_bytes == 750
    lines = _lines(path)
    assert lines[-2]["kind"] == "RESERVE" and lines[-2]["reserved_bytes"] == 600
    settlement = lines[-1]
    assert settlement["kind"] == "SETTLE" and settlement["sequence"] == 2
    assert settlement["actual_bytes"] == 750 and settlement["outcome"] == "failed"
    assert resumed.bytes_consumed == 600 + 750 and resumed.remaining_bytes == 150
    assert resumed.summary()["overrun_bytes"] == 0

    # The remainder no longer covers a body's cap: the next body request is
    # refused before it is made, and the counters say what was consumed.
    with (
        _transport(resumed, []) as transport,
        pytest.raises(ValueError, match="sec_campaign_byte_budget_exhausted"),
    ):
        transport.get(f"{ARCHIVE}/c.htm", maximum_bytes=600)
    assert resumed.attempts_consumed == 2 and resumed.unsettled == 1


def test_a_ledger_that_was_edited_or_cut_short_is_refused_by_name(tmp_path: Path) -> None:
    """requirement (Phase A4): the ledger is the campaign's record, not a
    counter a caller may set -- an edited line breaks the chain, a ledger
    shorter than its head says is refused, and a line appended without the
    head moved (the one crash window) is accepted and the head repaired."""

    path = tmp_path / "qa-campaign.jsonl"
    head = tmp_path / "qa-campaign.jsonl.head"
    served: list[str] = []
    ledger = SecCampaignLedger.open(path, declaration=DECLARATION)
    with _transport(ledger, served) as transport:
        transport.get(f"{ARCHIVE}/a.htm", maximum_bytes=600)
        transport.get(f"{ARCHIVE}/b.htm", maximum_bytes=600)
    original = path.read_bytes()
    lines = original.split(b"\n")
    assert len(lines) == 6 and lines[-1] == b""

    # An edited settlement: the chain no longer holds.
    edited = original.replace(b'"actual_bytes":500', b'"actual_bytes":5', 1)
    assert edited != original
    path.write_bytes(edited)
    with pytest.raises(ValueError, match="sec_campaign_ledger_invalid"):
        SecCampaignLedger.open(path, declaration=DECLARATION)

    # Cut short by a reservation and its settlement: the head says more.
    path.write_bytes(b"\n".join(lines[:3]) + b"\n")
    with pytest.raises(ValueError, match="sec_campaign_ledger_invalid"):
        SecCampaignLedger.open(path, declaration=DECLARATION)

    # One line ahead of its head -- appended, then the process died before
    # the head moved: accepted, and the head repaired.
    path.write_bytes(original)
    head.write_text(json.loads(lines[3])["line_hash"], encoding="utf-8")
    with SecCampaignLedger.open(path, declaration=DECLARATION) as repaired:
        assert repaired.bytes_consumed == 1_000
    assert head.read_text(encoding="utf-8") == json.loads(lines[4])["line_hash"]

    # A head from another record, or none for a ledger past its first line.
    head.write_text("0" * 64, encoding="utf-8")
    with pytest.raises(ValueError, match="sec_campaign_ledger_invalid"):
        SecCampaignLedger.open(path, declaration=DECLARATION)
    head.unlink()
    with pytest.raises(ValueError, match="sec_campaign_ledger_invalid"):
        SecCampaignLedger.open(path, declaration=DECLARATION)


def test_the_source_and_transport_take_the_campaign_bounds_from_its_ledger(
    tmp_path: Path,
) -> None:
    """requirement (Phase A4): the bodies already attempted and every bound
    are the ledger's; an argument that restates a bound must agree, and a
    metadata request is not a body resource."""

    path = tmp_path / "qa-campaign.jsonl"
    served: list[str] = []
    ledger = SecCampaignLedger.open(path, declaration=DECLARATION)
    with _transport(ledger, served) as transport:
        transport.get(INVENTORY, maximum_bytes=600)
        transport.get(f"{ARCHIVE}/a.htm", maximum_bytes=600)
    assert ledger.body_resources == frozenset({f"{ARCHIVE}/a.htm"})

    resumed = SecCampaignLedger.open(path, declaration=DECLARATION)
    with _transport(resumed, served) as transport:
        source = SecEdgarSource(transport, ledger=resumed)
        assert source.body_resources == {f"{ARCHIVE}/a.htm"}
        assert transport.attempts_remaining == 2 and transport.response_bytes_remaining == 500
        with pytest.raises(ValueError, match="sec_campaign_declaration_mismatch"):
            SecEdgarSource(transport, maximum_body_resources=3, ledger=resumed)
        with pytest.raises(ValueError, match="sec_campaign_declaration_mismatch"):
            HttpxSecOfficialTransport(
                user_agent=USER_AGENT, maximum_total_attempts=9, ledger=resumed
            ).close()


def test_the_admission_opens_a_named_campaign_once_and_resumes_it_with_its_remainder(
    tmp_path: Path,
) -> None:
    """requirement (Phase A4): a named campaign is admitted once with every
    bound and the document cap, resumed by a later admission with the same
    declaration, refused under another, and needs the official client; an
    injected transport spends none of it and a denied environment opens no
    record."""

    workspace = tmp_path / "workspace"
    environment = {SEC_USER_AGENT_VARIABLE: USER_AGENT}
    bounds = {
        "maximum_document_bytes": 6_000,
        "maximum_total_attempts": 4,
        "maximum_total_response_bytes": 15_000,
        "maximum_body_resources": 2,
    }
    with pytest.raises(ValueError, match="campaign_declaration_incomplete"):
        admit_official_source(
            network_consent=True,
            environment=environment,
            campaign_id="qa",
            workspace_root=workspace,
        )
    with pytest.raises(ValueError, match="campaign_declaration_incomplete"):
        admit_official_source(
            network_consent=True, environment=environment, campaign_id="qa", **bounds
        )

    # The network is the workspace's, as a person sets it in Settings; an incomplete
    # declaration is refused above before any network is asked for.
    set_network_access(workspace, enabled=True)
    first = admit_official_source(
        network_consent=True,
        environment=environment,
        campaign_id="qa",
        workspace_root=workspace,
        **bounds,
    )
    assert first.source is not None
    first.source.close()
    assert first.transport_origin == "OFFICIAL_HTTP" and first.campaign is not None
    assert first.campaign["resumed"] is False and first.campaign["bytes_remaining"] == 15_000
    ledger_path = campaign_ledger_path(workspace, "qa")
    assert ledger_path.is_file()
    assert ledger_path.parent == workspace / "runtime/artifacts/alternative-evidence/sec-campaigns"

    # Spent by one process, resumed by the next with what remains.
    ledger = SecCampaignLedger.open(
        ledger_path,
        declaration=SecCampaignDeclaration(
            campaign_id="qa",
            maximum_total_attempts=4,
            maximum_total_response_bytes=15_000,
            maximum_body_resources=2,
            maximum_document_bytes=6_000,
        ),
    )
    ledger.reserve(url=f"{ARCHIVE}/a.htm", resource="body", maximum_bytes=6_000)
    ledger.close()
    second = admit_official_source(
        network_consent=True,
        environment=environment,
        campaign_id="qa",
        workspace_root=workspace,
        **bounds,
    )
    assert second.source is not None
    second.source.close()
    assert second.campaign is not None and second.campaign["resumed"] is True
    assert (
        second.campaign["bytes_remaining"] == 9_000
        and second.campaign["unsettled_reservations"] == 1
    )
    assert second.source.body_resources == {f"{ARCHIVE}/a.htm"}

    with pytest.raises(ValueError, match="sec_campaign_declaration_mismatch"):
        admit_official_source(
            network_consent=True,
            environment=environment,
            campaign_id="qa",
            workspace_root=workspace,
            **{**bounds, "maximum_total_attempts": 5},
        )
    injected = HttpxSecOfficialTransport(user_agent=USER_AGENT)
    try:
        with pytest.raises(ValueError, match="campaign_requires_official_transport"):
            admit_official_source(
                network_consent=True,
                transport=injected,
                campaign_id="qa",
                workspace_root=workspace,
                **bounds,
            )
    finally:
        injected.close()
    denied = admit_official_source(
        network_consent=True,
        environment={OFFLINE_SWITCH: "1"},
        campaign_id="qa-denied",
        workspace_root=workspace,
        **bounds,
    )
    assert denied.transport_origin == "DENIED" and denied.campaign is None
    assert not campaign_ledger_path(workspace, "qa-denied").exists()


def test_one_ledger_has_one_owner_and_a_stale_holder_cannot_append(tmp_path: Path) -> None:
    """requirement (R2): two independently opened holders of one ledger must
    not both spend its balance -- the second is refused before any request
    could leave, while the first holds the campaign; released, the second
    resumes with what the first spent. A line another writer appended
    behind a holder's back refuses that holder's next append by name."""

    path = tmp_path / "qa-campaign.jsonl"
    served: list[str] = []
    first = SecCampaignLedger.open(path, declaration=DECLARATION)
    try:
        assert first.held
        with pytest.raises(ValueError, match="sec_campaign_ledger_held"):
            SecCampaignLedger.open(path, declaration=DECLARATION)
        with _transport(first, served) as transport:
            transport.get(f"{ARCHIVE}/a.htm", maximum_bytes=600)
            transport.get(f"{ARCHIVE}/b.htm", maximum_bytes=600)
            assert transport.response_bytes_remaining == 500
            # Still one owner while the client that spends it is open.
            with pytest.raises(ValueError, match="sec_campaign_ledger_held"):
                SecCampaignLedger.open(path, declaration=DECLARATION)
    finally:
        first.close()
    assert not first.held
    with pytest.raises(ValueError, match="sec_campaign_ledger_closed"):
        first.reserve(url=f"{ARCHIVE}/c.htm", resource="body", maximum_bytes=600)
    second = SecCampaignLedger.open(path, declaration=DECLARATION)
    try:
        assert second.resumed and second.remaining_bytes == 500 and second.remaining_attempts == 2
        # A writer that got past the lock appended a line: this holder's view
        # is stale, and its next append is refused, not written on top.
        foreign = second._lines[-1]
        with path.open("ab") as handle:
            handle.write(
                (
                    json.dumps(
                        {**foreign, "sequence": 9, "previous": foreign["line_hash"]},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            )
        with pytest.raises(ValueError, match="sec_campaign_ledger_stale"):
            second.reserve(url=f"{ARCHIVE}/c.htm", resource="body", maximum_bytes=100)
        assert second.remaining_bytes == 500, "nothing was reserved on a stale view"
    finally:
        second.close()


def test_a_campaign_id_is_one_safe_path_component_or_nothing_is_created(
    tmp_path: Path,
) -> None:
    """requirement (R3): a backslash, a drive, a separator, a parent or
    hidden reference in the id is refused by the declaration and by the
    ledger path before any file or directory exists; ordinary ids keep
    working and a ledger written under one reopens."""

    workspace = tmp_path / "workspace"
    for bad in (
        "..\\evil",
        "C:evil",
        "/tmp/evil",
        "\\\\server\\share",
        "..",
        ".",
        ".hidden",
        "a/b",
        "a b",
        "",
        "x" * 121,
    ):
        with pytest.raises(ValueError, match="sec_campaign_id_invalid"):
            SecCampaignDeclaration(
                campaign_id=bad,
                maximum_total_attempts=4,
                maximum_total_response_bytes=1_500,
                maximum_body_resources=2,
                maximum_document_bytes=600,
            )
        with pytest.raises(ValueError, match="campaign_path_invalid"):
            campaign_ledger_path(workspace, bad)
        with pytest.raises(ValueError, match=r"campaign_path_invalid|sec_campaign_id_invalid"):
            admit_official_source(
                network_consent=True,
                environment={SEC_USER_AGENT_VARIABLE: USER_AGENT},
                campaign_id=bad,
                workspace_root=workspace,
                maximum_document_bytes=6_000,
                maximum_total_attempts=4,
                maximum_total_response_bytes=15_000,
                maximum_body_resources=2,
            )
    assert not workspace.exists(), "a refused id created nothing"

    ordinary = "qa-2026.09_19"
    path = campaign_ledger_path(workspace, ordinary)
    campaigns = (
        workspace.resolve() / "runtime" / "artifacts" / "alternative-evidence" / "sec-campaigns"
    )
    assert path == campaigns / f"{ordinary}.jsonl"
    declaration = SecCampaignDeclaration(
        campaign_id=ordinary,
        maximum_total_attempts=4,
        maximum_total_response_bytes=1_500,
        maximum_body_resources=2,
        maximum_document_bytes=600,
    )
    with SecCampaignLedger.open(path, declaration=declaration) as ledger:
        ledger.reserve(url=f"{ARCHIVE}/a.htm", resource="body", maximum_bytes=600)
    with SecCampaignLedger.open(path, declaration=declaration) as reopened:
        assert reopened.resumed and reopened.bytes_consumed == 600
