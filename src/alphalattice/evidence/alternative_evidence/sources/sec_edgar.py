"""Bounded acquisition and parsing for official SEC EDGAR endpoints only."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Protocol, Self, get_args
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, model_validator

from alphalattice.kernel.live_evidence.online_sources import extract_canonical_markdown
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import (
    ADMITTED_DOCUMENT_CAPACITY,
    AcquisitionOutcome,
    AlternativeEvidenceCitation,
    AlternativeEvidenceClass,
    AlternativeEvidenceDocumentAcquisition,
    AlternativeEvidenceRequest,
    SecCompanyFactPoint,
    SecCompanyFactsSnapshot,
    SecIssuerRegistryEntry,
    SecIssuerRegistrySnapshot,
    seal_contract,
)
from .campaign import SecCampaignLedger
from .contracts import (
    AcquiredEvidenceDocument,
    AcquiredEvidenceSourceDeferral,
    SecFilingInventoryEntry,
    SecFilingSelectionDeferral,
    SecFilingSelectionPlan,
    SecMaterialForm,
)

_SEC_DATA = "https://data.sec.gov"
_MATERIAL_FORMS = frozenset(get_args(SecMaterialForm))
_NEGATIVE_EVENT_ITEMS = frozenset({"1.03", "2.04", "2.06", "3.01", "4.01", "4.02"})
"""Current-report items that are major negatives by their number: bankruptcy,
an acceleration trigger, a material impairment, a delisting notice, an
auditor change and non-reliance on issued statements."""
_SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_REGISTRY_URL = "https://www.sec.gov/files/company_tickers.json"
_ALLOWED_MIME = {"application/json", "text/json", "text/plain", "text/html"}


class _OutcomeRecorder:
    """Appends one resource's outcome to the caller's accounting, if any."""

    def __init__(
        self,
        outcomes: list[AlternativeEvidenceDocumentAcquisition] | None,
        *,
        entity_id: str,
        filing: SecFilingInventoryEntry,
    ) -> None:
        self._outcomes = outcomes
        self._entity_id = entity_id
        self._filing = filing

    def __call__(
        self, outcome: AcquisitionOutcome, *, detail: str = "", content_bytes: int = 0
    ) -> None:
        if self._outcomes is None:
            return
        self._outcomes.append(
            AlternativeEvidenceDocumentAcquisition(
                entity_id=self._entity_id,
                cik=self._filing.cik,
                accession=self._filing.accession,
                document_name=self._filing.primary_document,
                form=self._filing.form,
                outcome=outcome,
                detail=detail[:400],
                content_bytes=content_bytes,
            )
        )


class LocalSourceLookup(Protocol):
    """Find locally verified filing bodies and sealed deferrals.

    A verified local body for an inventory entry, or None when the scope
    has never held it; an integrity refusal raises `ValueError`
    (`alternative_evidence.source_object_tampered`) and is never a fetch.
    `holds` is the cheap question -- is a body named for this entry and
    present -- asked before any transfer to size what a fetch would add;
    it verifies nothing. `deferral` is the latest sealed deferral of the
    entry's exact resource, or None.
    """

    def find(self, entry: SecFilingInventoryEntry) -> AcquiredEvidenceDocument | None: ...

    def holds(self, entry: SecFilingInventoryEntry) -> bool: ...

    def deferral(self, entry: SecFilingInventoryEntry) -> AcquiredEvidenceSourceDeferral | None: ...


class SourceDocumentDeferral(Protocol):
    """Seal one resource's bounded deferral as it is observed."""

    def __call__(
        self, entry: SecFilingInventoryEntry, *, observed_bytes: int, admitted_cap_bytes: int
    ) -> None: ...


class StoragePreflight(Protocol):
    """Check whether storage can retain an issuer's fetched bodies.

    Refuse, by the storage owner's own code, an issuer whose fetched bodies
    could not be kept at every layer; called once per issuer with the bytes
    its transfers would add, before the first of them.
    """

    def __call__(self, expected_fetch_bytes: int) -> None:
        """Reject an issuer whose expected fetch bytes cannot be retained."""
        ...


class SourceDocumentCommit(Protocol):
    """Keep one acquired body and its provenance durably, before the next."""

    def __call__(self, document: AcquiredEvidenceDocument) -> None: ...


class AlternativeEvidenceCommitRefused(RuntimeError):
    """Stop fetching when a durable source commit is refused.

    The durable keep of a fetched body was refused -- by the storage
    budget, the disk's headroom or an integrity check of the store -- so
    the request stops here: nothing further is fetched that could not be
    kept, everything committed before stays, and the refusal carries the
    owner's own code on `failure_code` for the Task to record.
    """

    def __init__(self, cause: BaseException) -> None:
        """Carry the storage owner's failure code into the refusal."""
        code = getattr(cause, "failure_code", None)
        self.failure_code = (
            code if isinstance(code, str) and code else "alternative_evidence.source_commit_refused"
        )
        super().__init__(f"{self.failure_code}: {cause}")


class SecOfficialResponse(BaseModel):  # type: ignore[misc]
    """Hold a bounded official response and its retrieval time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status_code: int
    content_type: str
    content: bytes
    retrieved_at: datetime

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_response(self) -> Self:
        """Require a timezone-aware retrieval timestamp."""
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("alternative_evidence.sec_response_clock_invalid")
        return self


class SecOfficialTransport(Protocol):
    """Fetch bounded responses from official SEC URLs."""

    def get(self, url: str, *, maximum_bytes: int) -> SecOfficialResponse:
        """Fetch one bounded response from an official SEC URL."""
        ...


class HttpxSecOfficialTransport:
    """Small official-only client; admission remains outside this class.

    A campaign's hard limits live here, checked before a request is made:
    `maximum_total_attempts` bounds every HTTP attempt (metadata, history
    shards, bodies and retries alike) and `maximum_total_response_bytes`
    bounds the decoded bytes read across the campaign, partial and failed
    transfers included. An attempt is admitted only while the remaining
    allowance covers the most the request may read (its per-request cap);
    an exhausted budget refuses by name (`sec_campaign_attempt_budget_exhausted`,
    `sec_campaign_byte_budget_exhausted`) and is never retried. With a
    `ledger` the allowance is the campaign's durable one: the attempt and
    its cap are written there before the request leaves and settled with
    what was read afterwards, so a process that resumes the campaign spends
    only what remains and a transfer no process settled counts at its cap.

    What the byte bound guarantees, exactly. The pre-read guard is the
    reservation: nothing is requested unless its cap fits the remainder.
    The response is then read in the decoded chunks the client yields
    (each decoded network read, in practice tens of kilobytes), every
    chunk is counted before it is checked, and a transfer that crosses its
    cap is abandoned at the chunk that crossed it -- so the campaign's
    consumption may exceed the declared total by at most one chunk of one
    transfer, and the counters report what was read, never a clamped
    figure. The count is of decoded bytes the client yielded: a compressed
    response moves fewer bytes on the wire, and when a transfer is
    abandoned the socket and the client's buffers may already hold bytes
    the count never saw. Post-read detection is not the guarantee; the
    reservation before the read is.
    """

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_seconds: float = 15.0,
        maximum_attempts: int = 3,
        transport: httpx.BaseTransport | None = None,
        requests_per_second: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        maximum_total_attempts: int | None = None,
        maximum_total_response_bytes: int | None = None,
        ledger: SecCampaignLedger | None = None,
    ) -> None:
        """Configure the official client, rate limit, and campaign budgets."""
        if "@" not in user_agent:
            raise ValueError("alternative_evidence.sec_user_agent_invalid")
        if ledger is not None:
            # The durable campaign's totals are the budget; the arguments
            # may only restate them.
            declared = ledger.declaration
            for given, owned in (
                (maximum_total_attempts, declared.maximum_total_attempts),
                (maximum_total_response_bytes, declared.maximum_total_response_bytes),
            ):
                if given is not None and given != owned:
                    raise ValueError("alternative_evidence.sec_campaign_declaration_mismatch")
            maximum_total_attempts = declared.maximum_total_attempts
            maximum_total_response_bytes = declared.maximum_total_response_bytes
        self._ledger = ledger
        if maximum_total_attempts is not None and maximum_total_attempts < 1:
            raise ValueError("alternative_evidence.sec_campaign_budget_invalid")
        if maximum_total_response_bytes is not None and maximum_total_response_bytes < 1:
            raise ValueError("alternative_evidence.sec_campaign_budget_invalid")
        self._user_agent = user_agent
        self._timeout_seconds = timeout_seconds
        self._maximum_attempts = maximum_attempts
        self._maximum_total_attempts = maximum_total_attempts
        self._maximum_total_response_bytes = maximum_total_response_bytes
        self.budget_refusals: dict[str, int] = {"attempts": 0, "bytes": 0}
        if requests_per_second <= 0.0 or requests_per_second > 5.0:
            raise ValueError("alternative_evidence.sec_request_rate_invalid")
        self._minimum_interval = 1.0 / requests_per_second
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._last_request_started: float | None = None
        self._closed = False
        self.request_count = 0
        self.retry_count = 0
        self.rate_limited_count = 0
        self.response_bytes = 0
        self._client = httpx.Client(
            timeout=self._timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": self._user_agent, "Accept-Encoding": "gzip"},
            transport=transport,
        )

    def close(self) -> None:
        """Close the HTTP client and release its campaign ledger."""
        if not self._closed:
            self._client.close()
            self._closed = True
            if self._ledger is not None:
                # The campaign's ownership ends with the client that spent it.
                self._ledger.close()

    def __enter__(self) -> Self:
        """Return the transport for a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the transport when the context ends."""
        self.close()

    @property
    def closed(self) -> bool:
        """Report whether the underlying HTTP client is closed."""
        return self._closed

    def _throttle(self) -> None:
        if self._closed:
            raise ValueError("alternative_evidence.sec_transport_closed")
        now = self._monotonic()
        if self._last_request_started is not None:
            remaining = self._minimum_interval - (now - self._last_request_started)
            if remaining > 0.0:
                self._sleeper(remaining)
                now = self._monotonic()
        self._last_request_started = now

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        value = response.headers.get("retry-after")
        if value is not None:
            try:
                return min(max(float(value), 0.0), 30.0)
            except ValueError:
                try:
                    parsed = parsedate_to_datetime(value)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=UTC)
                    return min(max((parsed - datetime.now(UTC)).total_seconds(), 0.0), 30.0)
                except (TypeError, ValueError, OverflowError):
                    pass
        return float(min(0.25 * (2**attempt), 1.0))

    @property
    def ledger(self) -> SecCampaignLedger | None:
        """Return the durable campaign ledger, when one is bound."""
        return self._ledger

    @property
    def attempts_remaining(self) -> int | None:
        """Count attempts left in the campaign budget."""
        if self._ledger is not None:
            return self._ledger.remaining_attempts
        if self._maximum_total_attempts is None:
            return None
        return max(0, self._maximum_total_attempts - self.request_count)

    @property
    def response_bytes_remaining(self) -> int | None:
        """Count response bytes left in the campaign budget."""
        if self._ledger is not None:
            return self._ledger.remaining_bytes
        if self._maximum_total_response_bytes is None:
            return None
        return max(0, self._maximum_total_response_bytes - self.response_bytes)

    def _require_budget(self, maximum_bytes: int) -> None:
        """Refuse a request whose cap exceeds the campaign remainder.

        Refuse before the request unless the remainder covers what it
        may read; a refusal here never reaches the wire.
        """
        if self.attempts_remaining == 0:
            self.budget_refusals["attempts"] += 1
            raise ValueError("alternative_evidence.sec_campaign_attempt_budget_exhausted")
        remaining = self.response_bytes_remaining
        if remaining is not None and remaining < maximum_bytes:
            self.budget_refusals["bytes"] += 1
            raise ValueError("alternative_evidence.sec_campaign_byte_budget_exhausted")

    def get(self, url: str, *, maximum_bytes: int) -> SecOfficialResponse:
        """Fetch a bounded official response with budgeted retries."""
        _require_official_url(url)
        error: Exception | None = None
        resource = "body" if url.startswith(_SEC_ARCHIVES) else "metadata"
        for attempt in range(self._maximum_attempts):
            response: httpx.Response | None = None
            reservation: int | None = None
            read = 0
            outcome = "failed"
            try:
                self._require_budget(maximum_bytes)
                if self._ledger is not None:
                    # Written before the request leaves: the attempt and the
                    # most it may read, so a death mid-transfer is never zero.
                    reservation = self._ledger.reserve(
                        url=url, resource=resource, maximum_bytes=maximum_bytes
                    )
                self._throttle()
                self.request_count += 1
                with self._client.stream("GET", url) as response:
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if response.status_code == 429 or 500 <= response.status_code < 600:
                        if response.status_code == 429:
                            self.rate_limited_count += 1
                        raise RuntimeError(
                            f"alternative_evidence.sec_retryable_status:{response.status_code}"
                        )
                    if response.status_code != 200:
                        raise ValueError(
                            f"alternative_evidence.sec_http_status_invalid:{response.status_code}"
                        )
                    if content_type not in _ALLOWED_MIME:
                        raise ValueError("alternative_evidence.sec_mime_invalid")
                    declared = response.headers.get("content-length")
                    if declared is not None and int(declared) > maximum_bytes:
                        # The size the source declared rides on the refusal, so
                        # the deferral can record what was observed.
                        raise ValueError(
                            f"alternative_evidence.sec_response_too_large:{int(declared)}"
                        )
                    chunks: list[bytes] = []
                    for chunk in response.iter_bytes():
                        # Every decoded byte read counts against the campaign,
                        # whether or not this transfer completes; the chunk
                        # that crosses the cap is counted, then refused.
                        read += len(chunk)
                        self.response_bytes += len(chunk)
                        if read > maximum_bytes:
                            raise ValueError(f"alternative_evidence.sec_response_too_large:{read}")
                        chunks.append(chunk)
                content = b"".join(chunks)
                outcome = "complete"
                return SecOfficialResponse(
                    status_code=response.status_code,
                    content_type=content_type,
                    content=content,
                    retrieved_at=datetime.now(UTC),
                )
            except (httpx.HTTPError, RuntimeError) as caught:
                error = caught
                if attempt + 1 < self._maximum_attempts:
                    self.retry_count += 1
                    delay = (
                        self._retry_delay(response, attempt)
                        if response is not None
                        else min(0.25 * (2**attempt), 1.0)
                    )
                    self._sleeper(delay)
            finally:
                if reservation is not None and self._ledger is not None:
                    # Settled with what was actually read, complete or not.
                    self._ledger.settle(reservation, actual_bytes=read, outcome=outcome)
        raise RuntimeError("alternative_evidence.sec_acquisition_failed") from error


class SecEdgarSource:
    """Resolve issuer identity and immutable filings under an explicit cutoff.

    `maximum_body_resources` is a campaign's bound on distinct filing bodies
    fetched through this source; a body beyond it is refused by name before
    any request (`sec_campaign_body_budget_exhausted`), and a body already
    attempted is not a second resource. With a `ledger` the bound and the
    bodies already attempted are the durable campaign's, from every process
    that spent it.
    """

    def __init__(
        self,
        transport: SecOfficialTransport,
        *,
        maximum_body_resources: int | None = None,
        ledger: SecCampaignLedger | None = None,
        maximum_documents_per_issuer: int | None = None,
    ) -> None:
        """Bind the admitted transport, filing-body budget and documents per issuer."""
        if maximum_body_resources is not None and maximum_body_resources < 1:
            raise ValueError("alternative_evidence.sec_campaign_budget_invalid")
        self._transport = transport
        self._network_call_count = 0
        self.body_request_count = 0
        self.body_resources: set[str] = set()
        """The distinct filing-body locators this campaign has attempted."""
        if ledger is not None:
            owned = ledger.declaration.maximum_body_resources
            if maximum_body_resources is not None and maximum_body_resources != owned:
                raise ValueError("alternative_evidence.sec_campaign_declaration_mismatch")
            maximum_body_resources = owned
            self.body_resources.update(ledger.body_resources)
        self._maximum_body_resources = maximum_body_resources
        self._documents_per_issuer = maximum_documents_per_issuer

    def documents_per_issuer(self, request: AlternativeEvidenceRequest) -> int:
        """The documents an issuer may give: the request's, never past this admission's.

        A request sealed under a wider budget (queued, or recovered after the consent narrowed)
        runs under the narrower one.
        """
        sealed = request.source_policy.maximum_documents_per_issuer
        return (
            sealed
            if self._documents_per_issuer is None
            else min(sealed, self._documents_per_issuer)
        )

    @property
    def network_call_count(self) -> int:
        """Count official requests issued through this source."""
        return self._network_call_count

    def index_reader(self) -> SecEdgarSource:
        """Make a source for separately counted index reads.

        A source over the same transport that counts its own requests: the
        index reads a preparation makes before packing, apart from what a
        running Task accounts for. It reads indexes and fetches no body.
        """
        return SecEdgarSource(
            self._transport, maximum_documents_per_issuer=self._documents_per_issuer
        )

    @property
    def network_capable(self) -> bool:
        """Whether this source owns the admitted real SEC HTTP transport."""
        return isinstance(self._transport, HttpxSecOfficialTransport)

    def _get(self, url: str, *, maximum_bytes: int) -> SecOfficialResponse:
        self._network_call_count += 1
        return self._transport.get(url, maximum_bytes=maximum_bytes)

    def close(self) -> None:
        """Close the bound transport if it owns a close method."""
        close = getattr(self._transport, "close", None)
        if callable(close):
            close()

    def acquire_registry(self, *, captured_at: datetime) -> SecIssuerRegistrySnapshot:
        """Fetch and seal the SEC issuer registry snapshot."""
        response = self._get(_REGISTRY_URL, maximum_bytes=2_000_000)
        payload = _json_object(response.content)
        entries: list[SecIssuerRegistryEntry] = []
        for raw in payload.values():
            if not isinstance(raw, Mapping):
                raise ValueError("alternative_evidence.registry_payload_invalid")
            ticker = str(raw.get("ticker", "")).strip().upper()
            cik = normalize_cik(raw.get("cik_str"))
            title = str(raw.get("title", "")).strip()
            entries.append(
                SecIssuerRegistryEntry(
                    entity_id=ticker,
                    ticker=ticker,
                    cik=cik,
                    legal_name=title,
                )
            )
        entries.sort(key=lambda value: value.ticker)
        return seal_contract(
            SecIssuerRegistrySnapshot,
            "registry_hash",
            captured_at=captured_at,
            entries=tuple(entries),
            source_content_hash=canonical_hash(json.loads(response.content)),
        )

    def acquire_filings(
        self,
        *,
        request: AlternativeEvidenceRequest,
        registry: SecIssuerRegistrySnapshot,
    ) -> tuple[AlternativeEvidenceCitation, ...]:
        """Acquire filing citations for every requested issuer."""
        citations: list[AlternativeEvidenceCitation] = []
        for entity_id in request.ordered_entity_ids:
            citations.extend(
                self.acquire_entity_filings(
                    request=request,
                    registry=registry,
                    entity_id=entity_id,
                    handle_start=len(citations) + 1,
                )
            )
        return tuple(citations)

    def acquire_entity_filings(
        self,
        *,
        request: AlternativeEvidenceRequest,
        registry: SecIssuerRegistrySnapshot,
        entity_id: str,
        handle_start: int,
    ) -> tuple[AlternativeEvidenceCitation, ...]:
        """Acquire one issuer's filings and compile their citations."""
        documents = self.acquire_entity_documents(
            request=request,
            registry=registry,
            entity_id=entity_id,
            handle_start=handle_start,
        )
        return self.citations_from_documents(
            request=request,
            documents=documents,
            handle_start=handle_start,
        )

    @staticmethod
    def citations_from_documents(
        *,
        request: AlternativeEvidenceRequest,
        documents: tuple[AcquiredEvidenceDocument, ...],
        handle_start: int,
        excerpt_binding_hash: str | None = None,
        known_excerpt: Callable[[AcquiredEvidenceDocument], str | None] | None = None,
    ) -> tuple[AlternativeEvidenceCitation, ...]:
        """Compile historical compact citations without fetching a source twice.

        The excerpt is the head of the canonical extraction of the whole
        body -- a full parse for 1,600 characters. A caller that holds a
        sealed excerpt of these exact bytes under the extraction binding it
        names (`excerpt_binding_hash`) supplies it through `known_excerpt`,
        and the body is not parsed again; the citation records the binding
        either way, so the next request can do the same.
        """
        citations: list[AlternativeEvidenceCitation] = []
        for offset, document in enumerate(documents):
            excerpt = None if known_excerpt is None else known_excerpt(document)
            if excerpt is None:
                markdown = extract_canonical_markdown(
                    document.content,
                    input_cap_bytes=request.source_policy.maximum_document_bytes,
                ).decode("utf-8")
                excerpt = re.sub(r"\s+", " ", markdown).strip()[:1600]
            citations.append(
                seal_contract(
                    AlternativeEvidenceCitation,
                    "citation_hash",
                    semantic_handle=(f"CIT-{document.entity_id}-{handle_start + offset:03d}"),
                    entity_id=document.entity_id,
                    source_name=document.source_name,
                    source_right=document.source_right,
                    evidence_class=document.evidence_class,
                    document_type=document.document_type,
                    revision=document.revision,
                    published_at=document.published_at,
                    accepted_at=document.accepted_at,
                    available_at=document.available_at,
                    excerpt=excerpt,
                    limitations=document.limitations,
                    immutable_source=document.immutable_source,
                    source_content_hash=document.source_content_hash,
                    excerpt_binding_hash=excerpt_binding_hash,
                )
            )
        return tuple(citations)

    def plan_entity_filings(
        self,
        *,
        request: AlternativeEvidenceRequest,
        registry: SecIssuerRegistrySnapshot,
        entity_id: str,
        accession_scope: frozenset[str] | None = None,
        read_earlier: frozenset[str] | None = None,
    ) -> SecFilingSelectionPlan:
        """Discover one issuer's cutoff-valid inventory and plan what to fetch.

        One official request (the submissions index); no filing is fetched.
        An accession scope names the exact originals to fetch instead of the
        window's filings. `read_earlier` names filings an earlier analysis
        read -- the request's own by default -- which the plan passes over.
        """
        if AlternativeEvidenceClass.SEC_FILING not in request.evidence_classes:
            raise ValueError("alternative_evidence.sec_filing_not_requested")
        if entity_id not in request.ordered_entity_ids:
            raise ValueError("alternative_evidence.issuer_not_requested")
        if request.source_policy.retired:
            # A request sealed under the retired window reads back; executed
            # again it is refused, and prepared under the current policy.
            raise ValueError("alternative_evidence.source_policy_retired")
        entries = {value.entity_id: value for value in registry.entries}
        entry = entries.get(entity_id)
        if entry is None:
            raise ValueError("alternative_evidence.issuer_not_in_registry")
        response = self._get(
            f"{_SEC_DATA}/submissions/CIK{entry.cik}.json",
            maximum_bytes=request.source_policy.maximum_document_bytes,
        )
        submissions = _json_object(response.content)
        inventory = _material_inventory_entries(
            submissions,
            entity_id=entry.entity_id,
            cik=entry.cik,
            evidence_as_of=request.evidence_as_of,
        )
        # The issuer's own selection under its policy budget: the admitted
        # document set bounds the unit's selections together
        # (`apply_unit_capacity`), never by dividing it among the issuers.
        return plan_sec_filing_selection(
            inventory,
            entity_id=entry.entity_id,
            cik=entry.cik,
            evidence_as_of=request.evidence_as_of,
            event_window_days=request.source_policy.sec_recent_8k_days,
            policy_budget=self.documents_per_issuer(request),
            unit_capacity=ADMITTED_DOCUMENT_CAPACITY,
            accession_scope=accession_scope,
            read_earlier=(
                request.read_accessions(entity_id) if read_earlier is None else read_earlier
            ),
        )

    def acquire_entity_selection(
        self,
        *,
        request: AlternativeEvidenceRequest,
        registry: SecIssuerRegistrySnapshot,
        entity_id: str,
        handle_start: int,
        accession_scope: frozenset[str] | None = None,
        local: LocalSourceLookup | None = None,
        commit: SourceDocumentCommit | None = None,
        check_cancel: Callable[[], None] | None = None,
        outcomes: list[AlternativeEvidenceDocumentAcquisition] | None = None,
        preflight: StoragePreflight | None = None,
        defer: SourceDocumentDeferral | None = None,
        plan: SecFilingSelectionPlan | None = None,
    ) -> tuple[SecFilingSelectionPlan, tuple[AcquiredEvidenceDocument, ...]]:
        """Acquire exactly one issuer's planned filing selection.

        Plan (or take the `plan` already made for this issuer, with the
        unit's capacity applied), then obtain exactly the planned
        accessions, in the plan's order: a verified local body without a
        request, the rest fetched and committed one by one.
        """
        if plan is None:
            plan = self.plan_entity_filings(
                request=request,
                registry=registry,
                entity_id=entity_id,
                accession_scope=accession_scope,
            )
        documents = self._acquire_filing_documents(
            request=request,
            registry=registry,
            entity_id=entity_id,
            entries=plan.selected,
            handle_start=handle_start,
            local=local,
            commit=commit,
            check_cancel=check_cancel,
            outcomes=outcomes,
            preflight=preflight,
            defer=defer,
            explicit_retry=accession_scope is not None,
        )
        return plan, documents

    def acquire_entity_documents(
        self,
        *,
        request: AlternativeEvidenceRequest,
        registry: SecIssuerRegistrySnapshot,
        entity_id: str,
        handle_start: int,
    ) -> tuple[AcquiredEvidenceDocument, ...]:
        """Acquire one issuer's selected filing bodies."""
        _plan, documents = self.acquire_entity_selection(
            request=request, registry=registry, entity_id=entity_id, handle_start=handle_start
        )
        return documents

    def _acquire_filing_documents(
        self,
        *,
        request: AlternativeEvidenceRequest,
        registry: SecIssuerRegistrySnapshot,
        entity_id: str,
        entries: tuple[SecFilingInventoryEntry, ...],
        handle_start: int,
        local: LocalSourceLookup | None = None,
        commit: SourceDocumentCommit | None = None,
        check_cancel: Callable[[], None] | None = None,
        outcomes: list[AlternativeEvidenceDocumentAcquisition] | None = None,
        preflight: StoragePreflight | None = None,
        defer: SourceDocumentDeferral | None = None,
        explicit_retry: bool = False,
    ) -> tuple[AcquiredEvidenceDocument, ...]:
        """Obtain an exact Host-admitted accession set without repeating discovery.

        Each planned resource is one of: reused from a verified local body
        (no request), fetched from its official locator and committed
        before the next, deferred (its body exceeds the document cap) or
        failed (the source refused it, or a local body did not verify --
        which is a named integrity refusal, never a refetch). A failed or
        deferred resource does not fail its siblings; the caller reads the
        outcomes and names what the issuer is missing.
        """
        if AlternativeEvidenceClass.SEC_FILING not in request.evidence_classes:
            raise ValueError("alternative_evidence.sec_filing_not_requested")
        if entity_id not in request.ordered_entity_ids:
            raise ValueError("alternative_evidence.issuer_not_requested")
        registry_entries = {value.entity_id: value for value in registry.entries}
        registry_entry = registry_entries.get(entity_id)
        if registry_entry is None:
            raise ValueError("alternative_evidence.issuer_not_in_registry")
        if len(entries) > self.documents_per_issuer(request):
            raise ValueError("alternative_evidence.selected_filing_budget_exceeded")
        if len({value.accession for value in entries}) != len(entries):
            raise ValueError("alternative_evidence.selected_filing_duplicate")
        cap = request.source_policy.maximum_document_bytes
        if preflight is not None:
            # What the transfers of this issuer would add, before the first:
            # the official inventory's stated size where it states one (the
            # whole submission, an upper bound on its primary document), the
            # request's cap otherwise; a body already held adds nothing.
            try:
                expected = sum(
                    min(value.size_bytes, cap) if value.size_bytes is not None else cap
                    for value in entries
                    if local is None
                    or (
                        not local.holds(value)
                        and _deferral_eligible(local.deferral(value), request, cap, explicit_retry)
                    )
                )
            except ValueError:
                # The store's commitments do not read: nothing will be
                # fetched below (each resource fails by that name), so
                # nothing is put to the storage owner either.
                expected = 0
            if expected:
                try:
                    preflight(expected)
                except Exception as error:
                    # The owner's refusal stops the request here, by its own
                    # code, before a body that could not be kept is fetched.
                    raise AlternativeEvidenceCommitRefused(error) from error
        documents: list[AcquiredEvidenceDocument] = []
        for offset, filing in enumerate(entries):
            if (
                filing.entity_id != entity_id
                or filing.cik != registry_entry.cik
                or filing.accepted_at > request.evidence_as_of
            ):
                raise ValueError("alternative_evidence.selected_filing_authority_invalid")
            if check_cancel is not None:
                check_cancel()
            handle = f"DOC-{entity_id}-{handle_start + offset:03d}"
            record = _OutcomeRecorder(outcomes, entity_id=entity_id, filing=filing)
            if local is not None:
                try:
                    held = local.find(filing)
                except ValueError as error:
                    record("FAILED", detail=f"local integrity: {error}")
                    continue
                if held is not None and len(held.content) > cap:
                    # Held, verified, and larger than this request admits: the
                    # request's cap governs a retained body exactly as it
                    # governs a transfer -- deferred by name, never carried
                    # into a reading that would refuse it later.
                    record(
                        "DEFERRED",
                        detail=f"retained body of {len(held.content):,} bytes exceeds the "
                        f"{cap:,}-byte document cap of this request",
                    )
                    continue
                if held is not None:
                    # The bytes and their provenance are the retained ones;
                    # the handle, the security and the title are this
                    # request's -- two securities of one issuer share bodies.
                    document = held.model_copy(
                        update={
                            "semantic_handle": handle,
                            "entity_id": entity_id,
                            "title": f"{entity_id} {filing.form} {filing.accession}",
                        }
                    )
                    documents.append(document)
                    record("REUSED_LOCAL", content_bytes=len(document.content))
                    continue
            known = None if local is None else local.deferral(filing)
            if known is not None and not _deferral_eligible(known, request, cap, explicit_retry):
                # Observed oversize before, under a cap no smaller than this
                # request's: deferred by that record's name, no transfer.
                record(
                    "DEFERRED",
                    detail=(
                        f"known oversize: {known.observed_bytes:,}+ bytes observed on "
                        f"{known.observed_at.date().isoformat()} under a "
                        f"{known.admitted_cap_bytes:,}-byte cap; not transferred again under "
                        f"this request's {cap:,}-byte cap; eligible under a larger cap or by "
                        "explicit accession scope"
                    ),
                )
                continue
            try:
                document = self._fetch_filing(
                    request=request,
                    registry_entry=registry_entry,
                    entity_id=entity_id,
                    filing=filing,
                    handle=handle,
                )
            except ValueError as error:
                if "too_large" in str(error):
                    observed = _observed_bytes(str(error), cap)
                    record(
                        "DEFERRED",
                        detail=f"body exceeds the {cap:,}-byte document cap: {error}",
                    )
                    if defer is not None:
                        defer(filing, observed_bytes=observed, admitted_cap_bytes=cap)
                else:
                    record("FAILED", detail=f"{type(error).__name__}: {error}")
                continue
            except Exception as error:  # a source failure is named, never swallowed
                record("FAILED", detail=f"{type(error).__name__}: {error}")
                continue
            if commit is not None:
                try:
                    commit(document)
                except Exception as error:
                    record("FAILED", detail=f"durable keep refused: {error}"[:400])
                    raise AlternativeEvidenceCommitRefused(error) from error
            documents.append(document)
            record("FETCHED", content_bytes=len(document.content))
        return tuple(documents)

    def _fetch_filing(
        self,
        *,
        request: AlternativeEvidenceRequest,
        registry_entry: SecIssuerRegistryEntry,
        entity_id: str,
        filing: SecFilingInventoryEntry,
        handle: str,
    ) -> AcquiredEvidenceDocument:
        accession = filing.accession
        primary = filing.primary_document
        accession_compact = accession.replace("-", "")
        cik_compact = str(int(registry_entry.cik))
        locator = f"{_SEC_ARCHIVES}/{cik_compact}/{accession_compact}/{primary}"
        if (
            self._maximum_body_resources is not None
            and locator not in self.body_resources
            and len(self.body_resources) >= self._maximum_body_resources
        ):
            raise ValueError("alternative_evidence.sec_campaign_body_budget_exhausted")
        self.body_resources.add(locator)
        self.body_request_count += 1
        filing_response = self._get(
            locator,
            maximum_bytes=request.source_policy.maximum_document_bytes,
        )
        if filing_response.content_type not in {"text/html", "text/plain"}:
            raise ValueError("alternative_evidence.sec_filing_media_type_invalid")
        media_type = filing_response.content_type
        return AcquiredEvidenceDocument(
            semantic_handle=handle,
            entity_id=entity_id,
            source_name="SEC_EDGAR",
            source_right="SEC_PUBLIC_OFFICIAL_ACCESS",
            evidence_class=AlternativeEvidenceClass.SEC_FILING,
            document_type=filing.form,
            revision=accession,
            # The official filing date, kept as a date: midnight UTC
            # of it under DATE precision, never a publication instant.
            published_at=datetime.combine(filing.filed_on, datetime.min.time(), tzinfo=UTC),
            published_precision="DATE",
            accepted_at=filing.accepted_at,
            available_at=filing.accepted_at,
            report_period_end=filing.report_date,
            retrieved_at=filing_response.retrieved_at,
            title=f"{entity_id} {filing.form} {accession}",
            media_type=media_type,
            content=filing_response.content,
            limitations=("SEC filing text is untrusted source data.",),
            immutable_source=True,
            source_content_hash=canonical_hash(
                {
                    "source_name": "SEC_EDGAR",
                    "revision": accession,
                    "media_type": media_type,
                    "content_hex": filing_response.content.hex(),
                }
            ),
            source_cik=registry_entry.cik,
            source_document_name=primary,
        )

    def acquire_companyfacts_snapshot(
        self,
        *,
        request: AlternativeEvidenceRequest,
        registry: SecIssuerRegistrySnapshot,
        entity_id: str,
    ) -> SecCompanyFactsSnapshot:
        """Fetch and seal one issuer's companyfacts snapshot."""
        if AlternativeEvidenceClass.SEC_COMPANYFACTS not in request.evidence_classes:
            raise ValueError("alternative_evidence.companyfacts_not_requested")
        if entity_id not in request.ordered_entity_ids:
            raise ValueError("alternative_evidence.issuer_not_requested")
        entries = {value.entity_id: value for value in registry.entries}
        entry = entries.get(entity_id)
        if entry is None:
            raise ValueError("alternative_evidence.issuer_not_in_registry")
        response = self._get(
            f"{_SEC_DATA}/api/xbrl/companyfacts/CIK{entry.cik}.json",
            maximum_bytes=5_000_000,
        )
        payload = _json_object(response.content)
        facts = _compact_companyfacts(payload)
        return seal_contract(
            SecCompanyFactsSnapshot,
            "companyfacts_hash",
            entity_id=entity_id,
            cik=entry.cik,
            captured_at=response.retrieved_at,
            facts=facts,
            source_content_hash=canonical_hash(json.loads(response.content)),
        )

    @staticmethod
    def companyfacts_citation(
        *,
        request: AlternativeEvidenceRequest,
        snapshot: SecCompanyFactsSnapshot,
        semantic_handle: str,
    ) -> AlternativeEvidenceCitation:
        """Build a cutoff-valid citation from a local companyfacts snapshot."""
        if snapshot.entity_id not in request.ordered_entity_ids:
            raise ValueError("alternative_evidence.companyfacts_entity_invalid")
        if snapshot.captured_at > request.evidence_as_of:
            raise ValueError("alternative_evidence.mutable_companyfacts_historical_rejected")
        facts = tuple(
            value
            for value in snapshot.facts
            if value.filed_on <= request.evidence_as_of.date().isoformat()
        )
        if not facts:
            raise ValueError("alternative_evidence.companyfacts_empty_at_cutoff")
        excerpt = "; ".join(
            f"{value.concept}={value.value:g} {value.unit} "
            f"(period {value.period_end}, filed {value.filed_on}, {value.form})"
            for value in facts
        )
        return seal_contract(
            AlternativeEvidenceCitation,
            "citation_hash",
            semantic_handle=semantic_handle,
            entity_id=snapshot.entity_id,
            source_name="SEC_EDGAR",
            source_right="SEC_PUBLIC_OFFICIAL_ACCESS",
            evidence_class=AlternativeEvidenceClass.SEC_COMPANYFACTS,
            document_type="COMPANYFACTS_SNAPSHOT",
            revision=snapshot.companyfacts_hash[:24],
            published_at=None,
            accepted_at=snapshot.captured_at,
            available_at=snapshot.captured_at,
            excerpt=excerpt[:1600],
            limitations=(
                "Companyfacts is a mutable SEC endpoint; this finding uses a prior local as-of "
                "snapshot and filing dates do not substitute for acceptance timestamps.",
            ),
            immutable_source=False,
            source_content_hash=snapshot.source_content_hash,
        )


def normalize_cik(value: object) -> str:
    """Validate and zero-pad an SEC central index key."""
    text = str(value).strip()
    if not text.isdigit() or len(text) > 10:
        raise ValueError("alternative_evidence.cik_invalid")
    return text.zfill(10)


def _material_inventory_entries(
    submissions: Mapping[str, object],
    *,
    entity_id: str,
    cik: str,
    evidence_as_of: datetime,
) -> tuple[SecFilingInventoryEntry, ...]:
    filings = submissions.get("filings")
    if not isinstance(filings, Mapping):
        raise ValueError("alternative_evidence.sec_submissions_invalid")
    recent = filings.get("recent")
    if not isinstance(recent, Mapping):
        raise ValueError("alternative_evidence.sec_submissions_invalid")
    required_names = (
        "accessionNumber",
        "filingDate",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
    )
    columns = {name: recent.get(name) for name in required_names}
    if any(not isinstance(value, list) for value in columns.values()):
        raise ValueError("alternative_evidence.sec_submissions_invalid")
    lengths = {len(value) for value in columns.values() if isinstance(value, list)}
    if len(lengths) != 1:
        raise ValueError("alternative_evidence.sec_submissions_invalid")
    row_count = lengths.pop() if lengths else 0
    report_dates = recent.get("reportDate")
    if report_dates is None:
        report_dates = [""] * row_count
    if not isinstance(report_dates, list) or len(report_dates) != row_count:
        raise ValueError("alternative_evidence.sec_submissions_invalid")
    # A current report's items, as the index states them; an index without the
    # column (or with another length) states none, which is not a refusal.
    item_values = recent.get("items")
    if not isinstance(item_values, list) or len(item_values) != row_count:
        item_values = [""] * row_count
    rows: list[dict[str, object]] = []
    for index in range(row_count):
        row = {name: cast_list(columns[name])[index] for name in required_names}
        row["reportDate"] = report_dates[index]
        accepted = _parse_sec_datetime(str(row["acceptanceDateTime"]))
        if accepted > evidence_as_of:
            continue
        form = str(row["form"])
        if form in _MATERIAL_FORMS:
            row["items"] = item_values[index]
            rows.append(row)
    rows.sort(key=lambda value: _parse_sec_datetime(str(value["acceptanceDateTime"])))
    sizes = recent.get("size")
    size_of: dict[str, int] = {}
    if isinstance(sizes, list) and len(sizes) == row_count:
        for index in range(row_count):
            value = sizes[index]
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                size_of[str(cast_list(columns["accessionNumber"])[index])] = value

    originals: dict[tuple[str, str], str] = {}
    entries: list[SecFilingInventoryEntry] = []
    for row in rows:
        form = str(row["form"])
        base_form = form.removesuffix("/A")
        report_date = str(row["reportDate"]).strip()
        relation_key = (base_form, report_date)
        amendment_of = originals.get(relation_key) if form.endswith("/A") else None
        accession = str(row["accessionNumber"])
        if not form.endswith("/A"):
            originals[relation_key] = accession
        entries.append(
            SecFilingInventoryEntry(
                entity_id=entity_id,
                cik=cik,
                accession=accession,
                form=form,
                report_date=(datetime.fromisoformat(report_date).date() if report_date else None),
                filed_on=datetime.fromisoformat(str(row["filingDate"])).date(),
                accepted_at=_parse_sec_datetime(str(row["acceptanceDateTime"])),
                primary_document=str(row["primaryDocument"]),
                size_bytes=size_of.get(accession),
                amendment_of_accession=amendment_of,
                amendment_relation=(
                    "LINKED"
                    if amendment_of is not None
                    else "ORIGINAL_NOT_IN_RECENT_SUBMISSIONS"
                    if form.endswith("/A")
                    else "NOT_APPLICABLE"
                ),
                items=_current_report_items(base_form, row["items"]),
            )
        )
    return tuple(entries)


def _current_report_items(base_form: str, value: object) -> tuple[str, ...]:
    """Extract modern current-report item numbers.

    A current report's modern item numbers (`2.02`); the index's older
    single-number items and every other form's field state none.
    """
    if base_form != "8-K":
        return ()
    return tuple(
        dict.fromkeys(
            item
            for item in (part.strip() for part in str(value).split(","))
            if re.fullmatch(r"[1-9]\.[0-9]{2}", item) is not None
        )
    )


def _signal_rank(entry: SecFilingInventoryEntry) -> int:
    """Rank filing signals for material negative evidence.

    0 for a filing that carries a major negative by its form or its item
    number -- a late-filing notice, a current report with a negative item --
    1 for a periodic report (10-K, 10-Q and their amendments), and 2 for every
    other current report.
    """
    if entry.form.startswith("NT ") or _NEGATIVE_EVENT_ITEMS.intersection(entry.items):
        return 0
    return 1 if entry.form.removesuffix("/A") in {"10-K", "10-Q"} else 2


def _observed_bytes(message: str, cap: int) -> int:
    """Report the size observed in an oversize refusal.

    What an oversize refusal observed: the declared or read size it
    carries, or the cap itself when it carries none (more than the cap).
    """
    _prefix, _sep, tail = message.partition("too_large:")
    try:
        return max(int(tail), cap) if tail else cap
    except ValueError:
        return cap


def _deferral_eligible(
    known: AcquiredEvidenceSourceDeferral | None,
    request: AlternativeEvidenceRequest,
    cap: int,
    explicit_retry: bool,
) -> bool:
    """Check whether a sealed resource deferral permits transfer.

    Whether a resource with a sealed deferral may be transferred by this
    request: never deferred, an explicit accession scope, or a cap larger
    than the one it was observed under. No time admits it (W6): under the
    30-day window the filing leaves the window before any recheck is due.
    """
    del request
    return known is None or explicit_retry or cap > known.admitted_cap_bytes


def plan_sec_filing_selection(
    entries: tuple[SecFilingInventoryEntry, ...],
    *,
    entity_id: str,
    cik: str,
    evidence_as_of: datetime,
    event_window_days: int,
    policy_budget: int,
    unit_capacity: int,
    accession_scope: frozenset[str] | None = None,
    read_earlier: frozenset[str] = frozenset(),
) -> SecFilingSelectionPlan:
    """Plan one issuer's selection from its cutoff-valid material inventory.

    What the issuer filed in the window, and nothing older: every 10-K, 10-Q,
    8-K and late-filing notice, amendments included, accepted in the
    `event_window_days` before the cutoff is a candidate -- major negatives
    first, then periodic reports, then the other current reports, each newest
    first. No baseline is kept from outside the window; an issuer with nothing
    in it has an empty plan, which the packing reports as nothing filed. The
    capacity is the bound, and what it cannot take is deferred by name. A
    candidate an earlier analysis read (`read_earlier`) is deferred as read
    earlier: its findings carry, and the capacity goes to what is new.
    """
    capacity = min(policy_budget, unit_capacity)
    ordered = tuple(sorted(entries, key=lambda value: (value.accepted_at, value.accession)))
    if accession_scope is not None:
        return _plan_scoped_selection(
            ordered,
            entity_id=entity_id,
            cik=cik,
            evidence_as_of=evidence_as_of,
            event_window_days=event_window_days,
            policy_budget=policy_budget,
            unit_capacity=unit_capacity,
            capacity=capacity,
            accession_scope=accession_scope,
        )
    window_start = evidence_as_of - timedelta(days=event_window_days)
    candidates = sorted(
        (entry for entry in reversed(ordered) if entry.accepted_at >= window_start),
        key=_signal_rank,
    )
    selected = [entry for entry in candidates if entry.accession not in read_earlier][:capacity]
    chosen = {entry.accession for entry in selected}
    deferred = tuple(
        SecFilingSelectionDeferral(
            accession=entry.accession,
            form=entry.form,
            accepted_at=entry.accepted_at,
            reason="READ_EARLIER" if entry.accession in read_earlier else "BEYOND_CAPACITY",
        )
        for entry in candidates
        if entry.accession not in chosen
    )
    beyond = sum(1 for value in deferred if value.reason == "BEYOND_CAPACITY")
    limitations: tuple[str, ...] = ()
    if beyond:
        limitations = (
            f"{entity_id}: {beyond} filing(s) inside the {event_window_days}-day window "
            f"deferred beyond the {capacity}-document capacity (policy budget {policy_budget}, "
            f"admitted-set share {unit_capacity}).",
        )
    return seal_contract(
        SecFilingSelectionPlan,
        "plan_hash",
        entity_id=entity_id,
        cik=cik,
        evidence_as_of=evidence_as_of,
        event_window_days=event_window_days,
        policy_budget=policy_budget,
        unit_capacity=unit_capacity,
        capacity=capacity,
        discovered_count=len(ordered),
        required=(),
        events=tuple(selected),
        deferred=deferred,
        superseded_baseline_count=0,
        outside_window_event_count=len(ordered) - len(candidates),
        limitations=limitations,
    )


def _plan_scoped_selection(
    ordered: tuple[SecFilingInventoryEntry, ...],
    *,
    entity_id: str,
    cik: str,
    evidence_as_of: datetime,
    event_window_days: int,
    policy_budget: int,
    unit_capacity: int,
    capacity: int,
    accession_scope: frozenset[str],
) -> SecFilingSelectionPlan:
    """Select only the issuer's named accessions under its budget.

    Exactly the named accessions the issuer's own index holds under the
    cutoff; every other discovered filing deferred as outside the scope; a
    named accession the index does not hold stated by name, never fetched
    from a guessed location.
    """
    if not accession_scope or len(accession_scope) > 4:
        raise ValueError("alternative_evidence.selection_accession_scope_invalid")
    found = tuple(entry for entry in ordered if entry.accession in accession_scope)
    limitations = [
        f"{entity_id}: accession {accession} is not in the cutoff-valid material "
        f"submissions of CIK {cik}; not fetched."
        for accession in sorted(accession_scope - {entry.accession for entry in found})
    ]
    if len(found) > capacity:
        raise ValueError("alternative_evidence.selection_capacity_below_baselines")
    deferred = tuple(
        SecFilingSelectionDeferral(
            accession=entry.accession,
            form=entry.form,
            accepted_at=entry.accepted_at,
            reason="OUTSIDE_ACCESSION_SCOPE",
        )
        for entry in ordered
        if entry.accession not in accession_scope
    )
    limitations.append(
        f"{entity_id}: acquisition scoped to {len(accession_scope)} named accession(s); "
        f"{len(found)} found in the index, {len(deferred)} other discovered filing(s) "
        "deferred as outside the scope."
    )
    return seal_contract(
        SecFilingSelectionPlan,
        "plan_hash",
        entity_id=entity_id,
        cik=cik,
        evidence_as_of=evidence_as_of,
        accession_scope=tuple(sorted(accession_scope)),
        event_window_days=event_window_days,
        policy_budget=policy_budget,
        unit_capacity=unit_capacity,
        capacity=capacity,
        discovered_count=len(ordered),
        required=found,
        events=(),
        deferred=deferred,
        superseded_baseline_count=0,
        outside_window_event_count=0,
        limitations=tuple(limitations),
    )


def apply_unit_capacity(
    plans: tuple[SecFilingSelectionPlan, ...], *, capacity: int = ADMITTED_DOCUMENT_CAPACITY
) -> tuple[SecFilingSelectionPlan, ...]:
    """Apply the admitted document capacity to unit plans.

    The unit's plans under one admitted document set: when their
    selections together exceed `capacity`, event filings are deferred
    `BEYOND_UNIT_CAPACITY`, round-robin from the issuer holding the most
    selected events, that issuer's oldest first, until the unit fits; the
    baselines are never deferred, and a unit whose baselines alone exceed
    the capacity is refused by name. A plan that changes is sealed again
    with its new deferrals and a limitation naming the packing; a plan that
    fits is returned as it is. Deterministic: the same plans give the same
    result, whatever order they arrive in.
    """
    ordered = sorted(plans, key=lambda plan: plan.entity_id)
    if len({plan.entity_id for plan in ordered}) != len(ordered):
        raise ValueError("alternative_evidence.unit_plans_issuer_duplicate")
    events: dict[str, list[SecFilingInventoryEntry]] = {
        plan.entity_id: list(plan.events) for plan in ordered
    }
    deferred: dict[str, list[SecFilingInventoryEntry]] = {}
    required = sum(len(plan.required) for plan in ordered)
    if required > capacity:
        raise ValueError("alternative_evidence.unit_capacity_below_baselines")
    while required + sum(len(values) for values in events.values()) > capacity:
        count, entity_id = max((len(values), key) for key, values in events.items())
        if count == 0:
            raise ValueError("alternative_evidence.unit_capacity_below_baselines")
        # A routine event, oldest first, before any major negative.
        oldest = min(
            events[entity_id],
            key=lambda entry: (-_signal_rank(entry), entry.accepted_at, entry.accession),
        )
        events[entity_id].remove(oldest)
        deferred.setdefault(entity_id, []).append(oldest)
    result: list[SecFilingSelectionPlan] = []
    for plan in ordered:
        moved = deferred.get(plan.entity_id)
        if not moved:
            result.append(plan)
            continue
        result.append(
            seal_contract(
                SecFilingSelectionPlan,
                "plan_hash",
                **{
                    name: value
                    for name, value in plan
                    if name not in {"plan_hash", "events", "deferred", "limitations"}
                },
                events=tuple(events[plan.entity_id]),
                deferred=(
                    *plan.deferred,
                    *(
                        SecFilingSelectionDeferral(
                            accession=entry.accession,
                            form=entry.form,
                            accepted_at=entry.accepted_at,
                            reason="BEYOND_UNIT_CAPACITY",
                        )
                        for entry in sorted(moved, key=lambda e: (e.accepted_at, e.accession))
                    ),
                ),
                limitations=(
                    *plan.limitations,
                    f"{plan.entity_id}: {len(moved)} selected event filing(s) deferred beyond "
                    f"the admitted document set's capacity of {capacity} for this unit of "
                    f"{len(ordered)} issuers (pack fewer issuers per unit): "
                    + ", ".join(
                        f"{entry.form} {entry.accession}"
                        for entry in sorted(moved, key=lambda e: (e.accepted_at, e.accession))
                    )
                    + ".",
                ),
            )
        )
    by_entity = {plan.entity_id: plan for plan in result}
    return tuple(by_entity[plan.entity_id] for plan in plans)


def _json_object(content: bytes) -> dict[str, object]:
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("alternative_evidence.sec_json_invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("alternative_evidence.sec_json_invalid")
    return payload


def _compact_companyfacts(payload: Mapping[str, object]) -> tuple[SecCompanyFactPoint, ...]:
    raw_facts = payload.get("facts")
    if not isinstance(raw_facts, Mapping):
        raise ValueError("alternative_evidence.companyfacts_payload_invalid")
    us_gaap = raw_facts.get("us-gaap")
    if not isinstance(us_gaap, Mapping):
        return ()
    allowed = (
        "Assets",
        "Liabilities",
        "Revenues",
        "NetIncomeLoss",
        "CashAndCashEquivalentsAtCarryingValue",
    )
    points: list[SecCompanyFactPoint] = []
    for concept in allowed:
        value = us_gaap.get(concept)
        if not isinstance(value, Mapping):
            continue
        units = value.get("units")
        if not isinstance(units, Mapping):
            continue
        candidates: list[SecCompanyFactPoint] = []
        for unit, records in units.items():
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, Mapping) or record.get("form") not in {"10-K", "10-Q"}:
                    continue
                try:
                    candidates.append(
                        SecCompanyFactPoint(
                            concept=concept,
                            unit=str(unit),
                            value=float(record["val"]),
                            period_end=str(record["end"]),
                            filed_on=str(record["filed"]),
                            form=str(record["form"]),
                            accession=str(record["accn"]),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        if candidates:
            points.append(max(candidates, key=lambda item: (item.filed_on, item.period_end)))
    return tuple(points)


def _parse_sec_datetime(value: str) -> datetime:
    normalized = value.rstrip("Z")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _require_official_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("alternative_evidence.non_official_url_rejected")
    if parsed.netloc == "data.sec.gov" and (
        re.fullmatch(r"/submissions/CIK\d{10}\.json", parsed.path)
        # An issuer's older submissions, the official shape exactly: the
        # main file names them `CIK##########-submissions-NNN.json`.
        or re.fullmatch(r"/submissions/CIK\d{10}-submissions-\d{3}\.json", parsed.path)
        or re.fullmatch(r"/api/xbrl/companyfacts/CIK\d{10}\.json", parsed.path)
    ):
        return
    if parsed.netloc == "www.sec.gov" and (
        parsed.path == "/files/company_tickers.json"
        or re.fullmatch(
            r"/Archives/edgar/data/[1-9]\d*/\d{18}/[A-Za-z0-9][A-Za-z0-9._-]*",
            parsed.path,
        )
    ):
        return
    raise ValueError("alternative_evidence.non_official_url_rejected")


def cast_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("alternative_evidence.sec_submissions_invalid")
    return value


__all__ = [
    "AlternativeEvidenceCommitRefused",
    "HttpxSecOfficialTransport",
    "SecEdgarSource",
    "SecOfficialResponse",
    "SecOfficialTransport",
    "StoragePreflight",
    "normalize_cik",
]
