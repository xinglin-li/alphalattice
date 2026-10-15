"""The official SEC source, admitted for live acquisitions only by explicit consent.

Offline is the default: without the operator's consent nothing is composed, and with it the
official client still needs the workspace's network access and the product's SEC contact. A denied
transport keeps a consented but offline process from presenting local holdings as fresh, and a
durable campaign's ledger lives beside the Evidence artifact store.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal, TypedDict

from alphalattice.control.workspace_runtime.network_access import NetworkAccess, network_access
from alphalattice.evidence.alternative_evidence.contracts import (
    DEFAULT_SOURCE_DOCUMENT_BYTES,
    AlternativeEvidenceAdmission,
    AlternativeEvidenceSourcePolicy,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.sources.campaign import (
    CAMPAIGN_LEDGER_ROOT,
    SecCampaignDeclaration,
    SecCampaignLedger,
    require_campaign_id,
)
from alphalattice.evidence.alternative_evidence.sources.sec_edgar import (
    HttpxSecOfficialTransport,
    SecEdgarSource,
    SecOfficialResponse,
    SecOfficialTransport,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

SEC_USER_AGENT_VARIABLE = "SEC_USER_AGENT"
SEC_CONTACT = "sec-contact.json"
"""The product's own SEC contact, shipped with it: the organization and the dedicated address it
maintains, and its request rate, well below the SEC's fair-access limit."""


def _sec_contact() -> dict[str, Any]:
    try:
        text = resources.files(__package__).joinpath(SEC_CONTACT).read_text(encoding="utf-8")
    except FileNotFoundError:  # a build without its contact: refused as its configuration
        return {}
    contact = json.loads(text)
    return contact if isinstance(contact, dict) else {}


def sec_user_agent(environment: Mapping[str, str] | None = None) -> str | None:
    """The User-Agent the product names itself by to the SEC, or None when this build has none.

    The product's contact (`SEC_CONTACT`) unless an operator's `SEC_USER_AGENT` names another
    address. A build with neither has a configuration gap for its maintainer; the person is
    never asked for a name or an address.
    """
    override = (
        (os.environ if environment is None else environment)
        .get(SEC_USER_AGENT_VARIABLE, "")
        .strip()
    )
    if "@" in override:
        return override
    contact = _sec_contact()
    organization = str(contact.get("organization") or "").strip()
    email = str(contact.get("email") or "").strip()
    return f"{organization} {email}" if organization and "@" in email else None


def sec_request_rate() -> float:
    """The product's SEC request rate, requests per second."""
    return float(_sec_contact().get("requests_per_second") or 1.0)


CONSENT = Path("runtime") / "evidence-source-consent.json"
"""The person's consent to official SEC acquisition in a workspace, and the budget it admits."""


@dataclass(frozen=True, slots=True)
class EvidenceSourceConsent:
    """What official SEC acquisition may take: per issuer, in documents and in bytes overall.

    `actor` is `HUMAN`, the delegation a first use's agent acted under, or `DEFAULT` when no
    consent was recorded (the product's default budget, which a delegation may grant).
    """

    documents_per_issuer: int = 3
    total_documents: int = 600
    total_bytes: int = 3_000_000_000
    actor: str = "DEFAULT"
    granted_at: str | None = None

    def __post_init__(self) -> None:
        """Refuse a budget the source policy or the totals cannot hold."""
        counts = (self.documents_per_issuer, self.total_documents, self.total_bytes)
        if any(type(value) is not int for value in counts) or not isinstance(self.actor, str):
            raise ValueError("evidence_review.consent_budget_invalid")
        AlternativeEvidenceSourcePolicy(maximum_documents_per_issuer=self.documents_per_issuer)
        if self.total_documents < self.documents_per_issuer or self.total_bytes < 1:
            raise ValueError("evidence_review.consent_budget_invalid")

    def within(self, budget: EvidenceSourceConsent) -> bool:
        """Whether this consent asks for no more than `budget`, in every bound."""
        return (
            self.documents_per_issuer <= budget.documents_per_issuer
            and self.total_documents <= budget.total_documents
            and self.total_bytes <= budget.total_bytes
        )

    @property
    def campaign_id(self) -> str:
        """The durable campaign this consent's totals are spent from, one per budget.

        A grant repeated at the same budget spends what the last one left.
        """
        digest: str = canonical_hash(
            [self.documents_per_issuer, self.total_documents, self.total_bytes]
        )
        return "consent-" + digest[:16]

    def body(self) -> dict[str, object]:
        """The consent as its owner answers it."""
        return {
            "documents_per_issuer": self.documents_per_issuer,
            "total_documents": self.total_documents,
            "total_bytes": self.total_bytes,
            "actor": self.actor,
            "granted_at": self.granted_at,
        }


DEFAULT_SOURCE_CONSENT = EvidenceSourceConsent()
"""The default budget: at most three documents per issuer, within the product's totals."""


def evidence_source_consent(workspace: Path | None) -> EvidenceSourceConsent:
    """The workspace's recorded consent; the default budget when none is recorded.

    A record that does not read as a consent grants nothing and says so (`UNREADABLE`).
    """
    if workspace is None:
        return DEFAULT_SOURCE_CONSENT
    try:
        value = json.loads((workspace / CONSENT).read_text(encoding="utf-8"))
        return EvidenceSourceConsent(**{k: value[k] for k in DEFAULT_SOURCE_CONSENT.body()})
    except FileNotFoundError:
        return DEFAULT_SOURCE_CONSENT
    except (OSError, ValueError, KeyError, TypeError):
        return replace(DEFAULT_SOURCE_CONSENT, actor="UNREADABLE")


def record_evidence_source_consent(
    workspace: Path, consent: EvidenceSourceConsent
) -> EvidenceSourceConsent:
    """Write the workspace's consent atomically and answer it."""
    target = workspace / CONSENT
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f"{target.name}.partial")
    staged.write_text(json.dumps(consent.body(), sort_keys=True), encoding="utf-8")
    os.replace(staged, target)
    return consent


def seal_live_setup_admission(
    request_hash: str, admitted_at: datetime
) -> AlternativeEvidenceAdmission:
    """Seal a setup admission after its explicit network permission checks.

    Args:
        request_hash: The admitted acquisition request.
        admitted_at: The setup's admission clock.

    Returns:
        Live official-source admission without model-review authority.
    """
    return seal_contract(
        AlternativeEvidenceAdmission,
        "admission_hash",
        request_hash=request_hash,
        network_consent=True,
        admit_live_official=True,
        admit_model_review=False,
        admitted_at=admitted_at,
    )


class _Declared(TypedDict, total=False):
    """What the operator declared for an admission, stated on it whatever it admits."""

    maximum_document_bytes: int | None
    maximum_documents_per_issuer: int
    acquisition_window_seconds: int | None
    campaign_budget: Mapping[str, int] | None
    campaign: Mapping[str, object] | None
    campaign_ledger: SecCampaignLedger | None


class DeniedSecOfficialTransport:
    """The official source with the network denied.

    The admission's own transport, not the acquisition owner's: every request is refused by
    name before anything leaves the process, and counted. What the workspace already holds
    reads back under the live policy; a source check fails by this name, so nothing local is
    presented as fresh.
    """

    def __init__(self) -> None:
        """Start with no request refused."""
        self.refused: list[str] = []

    def get(self, url: str, *, maximum_bytes: int) -> SecOfficialResponse:
        """Refuse a request by name, and count it.

        Args:
            url: The official URL asked for.
            maximum_bytes: The cap the request carried.

        Raises:
            ValueError: Always, as `alternative_evidence.network_disabled`.
        """
        self.refused.append(url)
        raise ValueError("alternative_evidence.network_disabled")

    def close(self) -> None:
        """Close nothing: no connection was opened."""
        return None


@dataclass(frozen=True, slots=True)
class OfficialSourceAdmission:
    """An official SEC source admitted for this process's live acquisitions.

    `source` is the maintained acquisition owner over the transport this
    admission composed: the official HTTP client only when the operator
    consented and the environment allows the network, or a transport the
    caller injected (recorded or controlled responses, for a rehearsal or a
    test) -- named as such by `transport_origin`. `refusal_code` says why no
    source was admitted; the workspace then stays on its recorded package.
    """

    source: SecEdgarSource | None
    network_consent: bool
    transport_origin: Literal["OFFICIAL_HTTP", "INJECTED", "DENIED", "NONE"]
    refusal_code: str | None = None
    """Why no official client was composed. Set with `transport_origin`
    `DENIED` too: consent was given but the environment keeps the network
    disabled, so the live branch is composed over a transport that refuses
    every request by name -- what the workspace holds reads back, nothing
    outbound happens, and a source check is refused as `network_disabled`
    rather than answered from stale local inventory."""
    network_access: NetworkAccess | None = None
    """The effective permission read when a real source was composed; absent for
    an injected transport. Runtime admission data, never a source binding."""
    maximum_document_bytes: int | None = None
    """The per-document cap the operator declared for this source, when one
    was declared; the recorded policy's cap otherwise. Stated at admission,
    before any request, never raised after a refusal."""
    maximum_documents_per_issuer: int = DEFAULT_SOURCE_CONSENT.documents_per_issuer
    """The documents per issuer the workspace's consent admits (`evidence_source_consent`);
    every request under this source seals it, and the rest of an issuer's filings are deferred
    by name."""
    acquisition_window_seconds: int | None = None
    """The acquisition window the operator declared for this source's
    requests, when one was declared -- the deadline every request under it
    carries from its cutoff; the recorded policy's window otherwise. An
    operational choice of the admission, not a change to the policy's
    default."""
    campaign_budget: Mapping[str, int] | None = None
    """The campaign's hard limits the transport and source enforce before
    each request, as declared: `attempts`, `response_bytes`, `body_resources`."""
    campaign: Mapping[str, object] | None = None
    """The durable campaign this admission spends, when one was named and an
    official client was composed: its id, whether it was resumed, what it
    has consumed and what remains (`SecCampaignLedger.summary`)."""
    campaign_ledger: SecCampaignLedger | None = None
    """The ledger itself, which the transport and the source spend: read
    for the campaign's balance at any later moment, never written here."""

    @property
    def admitted(self) -> bool:
        """Whether a source was composed."""
        return self.source is not None


def admit_official_source(
    *,
    network_consent: bool,
    environment: Mapping[str, str] | None = None,
    transport: SecOfficialTransport | None = None,
    maximum_document_bytes: int | None = None,
    acquisition_window_seconds: int | None = None,
    maximum_total_attempts: int | None = None,
    maximum_total_response_bytes: int | None = None,
    maximum_body_resources: int | None = None,
    campaign_id: str | None = None,
    workspace_root: Path | None = None,
) -> OfficialSourceAdmission:
    """Compose the official source, or say by name why not.

    Offline is the default: without the operator's explicit consent nothing
    is composed; with it, the official client needs network access (the
    workspace's network control, `network_access`: the operator's
    `ALPHALATTICE_NETWORK_DISABLED=1` keeps the process offline whatever the
    workspace says) and the product's SEC contact (`sec_user_agent`). With consent and the network
    still disabled the live branch is composed over a denied transport
    (`DENIED`): the workspace's live preparations read back, no request
    leaves the process, and a source check is refused by name. An injected
    transport is admitted under the consent alone -- it is the caller's
    controlled responses, not the network -- and is recorded as injected. A
    declared per-document cap is validated by the source policy contract,
    which admits nothing above its maximum. A declared acquisition window
    and campaign budget (total attempts, total decoded bytes, distinct body
    resources) are the admission's operational choices, enforced by the
    transport and the source before each request. A `campaign_id` names a
    durable campaign under the workspace (`campaign_ledger_path`): the first
    admission writes its totals -- every budget bound and the document cap
    are then required -- and a later admission with the same id and totals
    resumes it with only what remains; the same id with other totals is
    refused as another campaign, and a campaign needs the official client
    (an injected transport spends none of it).
    """
    if maximum_document_bytes is not None:
        AlternativeEvidenceSourcePolicy(maximum_document_bytes=maximum_document_bytes)
    if acquisition_window_seconds is not None and acquisition_window_seconds < 1:
        raise ValueError("evidence_review.acquisition_window_invalid")
    budget = {
        key: value
        for key, value in (
            ("attempts", maximum_total_attempts),
            ("response_bytes", maximum_total_response_bytes),
            ("body_resources", maximum_body_resources),
        )
        if value is not None
    }
    if any(value < 1 for value in budget.values()):
        raise ValueError("evidence_review.campaign_budget_invalid")
    # The workspace's consent bounds every declared total, and fills those not declared.
    consent = evidence_source_consent(workspace_root)
    for key, bound in (
        ("body_resources", consent.total_documents),
        ("response_bytes", consent.total_bytes),
    ):
        if budget.get(key, 0) > bound:
            raise ValueError("evidence_review.budget_exceeds_consent")
        budget.setdefault(key, bound)
    maximum_body_resources = budget["body_resources"]
    maximum_total_response_bytes = budget["response_bytes"]
    declared: _Declared = {
        "maximum_document_bytes": maximum_document_bytes,
        "maximum_documents_per_issuer": consent.documents_per_issuer,
        "acquisition_window_seconds": acquisition_window_seconds,
        "campaign_budget": budget or None,
    }
    declaration: SecCampaignDeclaration | None = None
    if campaign_id is not None:
        if workspace_root is None or len(budget) != 3 or maximum_document_bytes is None:
            raise ValueError("evidence_review.campaign_declaration_incomplete")
        if transport is not None:
            raise ValueError("evidence_review.campaign_requires_official_transport")
        declaration = SecCampaignDeclaration(
            campaign_id=campaign_id,
            maximum_total_attempts=budget["attempts"],
            maximum_total_response_bytes=budget["response_bytes"],
            maximum_body_resources=budget["body_resources"],
            maximum_document_bytes=maximum_document_bytes,
        )
    if not network_consent:
        return OfficialSourceAdmission(
            source=None,
            network_consent=False,
            transport_origin="NONE",
            refusal_code="evidence_review.explicit_sec_network_consent_required",
            **declared,
        )
    if transport is not None:
        return OfficialSourceAdmission(
            source=SecEdgarSource(
                transport,
                maximum_body_resources=maximum_body_resources,
                maximum_documents_per_issuer=consent.documents_per_issuer,
            ),
            network_consent=True,
            transport_origin="INJECTED",
            **declared,
        )
    values = dict(os.environ if environment is None else environment)
    # The workspace's typed network control decides, the operator's offline switch first.
    access = network_access(workspace_root, values)
    if not access.allowed:
        return OfficialSourceAdmission(
            source=SecEdgarSource(
                DeniedSecOfficialTransport(),
                maximum_body_resources=maximum_body_resources,
                maximum_documents_per_issuer=consent.documents_per_issuer,
            ),
            network_consent=True,
            transport_origin="DENIED",
            refusal_code="evidence_review.network_disabled",
            network_access=access,
            **declared,
        )
    user_agent = sec_user_agent(values)
    if user_agent is None:
        return OfficialSourceAdmission(
            source=None,
            network_consent=True,
            transport_origin="NONE",
            refusal_code="evidence_review.sec_contact_not_configured",
            **declared,
        )
    ledger: SecCampaignLedger | None = None
    if declaration is not None and workspace_root is not None:
        ledger = SecCampaignLedger.open(
            campaign_ledger_path(workspace_root, declaration.campaign_id),
            declaration=declaration,
        )
        declared["campaign"] = ledger.summary()
        declared["campaign_ledger"] = ledger
    return OfficialSourceAdmission(
        source=SecEdgarSource(
            HttpxSecOfficialTransport(
                user_agent=user_agent,
                requests_per_second=sec_request_rate(),
                maximum_total_attempts=maximum_total_attempts,
                maximum_total_response_bytes=maximum_total_response_bytes,
                ledger=ledger,
            ),
            maximum_body_resources=maximum_body_resources,
            ledger=ledger,
            maximum_documents_per_issuer=consent.documents_per_issuer,
        ),
        network_consent=True,
        transport_origin="OFFICIAL_HTTP",
        network_access=access,
        **declared,
    )


NOT_GRANTED = frozenset({"DEFAULT", "UNREADABLE"})
"""The consent actors that grant no acquisition: none recorded, or a record that does not read."""

CONSENT_ATTEMPTS_PER_DOCUMENT = 4
"""Requests a consented document may take: its filing index, its body and their retries."""


def official_source_state(
    workspace: Path, environment: Mapping[str, str] | None = None
) -> tuple[object, ...]:
    """What the workspace's admission of the official source depends on, read cheaply.

    Its consent, its network control and the product's contact (`admit_workspace_source`).
    """
    return (
        evidence_source_consent(workspace).body(),
        network_access(workspace, environment).allowed,
        sec_user_agent(environment),
    )


def admit_workspace_source(
    workspace: Path, environment: Mapping[str, str] | None = None
) -> OfficialSourceAdmission:
    """The official source as the workspace admits it now, with no Host restart.

    A recorded consent is the consent (`evidence_source_consent`); its totals are spent from one
    durable campaign, so they hold across the Host's restarts. With none recorded, nothing is
    admitted and the Evidence review stays on its recorded package.
    """
    consent = evidence_source_consent(workspace)
    if consent.actor in NOT_GRANTED:
        return admit_official_source(
            network_consent=False, workspace_root=workspace, environment=environment
        )
    return admit_official_source(
        network_consent=True,
        workspace_root=workspace,
        environment=environment,
        maximum_document_bytes=DEFAULT_SOURCE_DOCUMENT_BYTES,
        maximum_total_attempts=CONSENT_ATTEMPTS_PER_DOCUMENT * consent.total_documents,
        maximum_total_response_bytes=consent.total_bytes,
        maximum_body_resources=consent.total_documents,
        campaign_id=consent.campaign_id,
    )


REQUESTS_PER_DOCUMENT = 2
"""A document's filing index and its body, without retries."""


def acquisition_scope(
    issuers: int, consent: EvidenceSourceConsent, *, issuers_per_unit: int
) -> dict[str, object]:
    """What official acquisition of these issuers would take under the consent, before it.

    Documents and bytes are upper bounds: an issuer that filed nothing in the window takes none.
    Requests and seconds are an estimate without retries: each issuer's filing index is one
    request and each document two, at the product's request rate.
    """
    documents = min(issuers * consent.documents_per_issuer, consent.total_documents)
    requests = 1 + issuers + REQUESTS_PER_DOCUMENT * documents
    return {
        "issuers": issuers,
        "units": -(-issuers // issuers_per_unit),
        "documents_per_issuer": consent.documents_per_issuer,
        "documents_at_most": documents,
        "bytes_at_most": min(documents * DEFAULT_SOURCE_DOCUMENT_BYTES, consent.total_bytes),
        "requests_estimated": requests,
        "seconds_estimated": round(requests / sec_request_rate()),
        "beyond_one_package": max(issuers - issuers_per_unit, 0),
    }


def campaign_ledger_path(workspace_root: Path, campaign_id: str) -> Path:
    """Return where a workspace keeps one campaign's ledger.

    Beside the evidence artifact store, under the managed storage the inventory counts. The id
    is one safe path component (`require_campaign_id`) and the path is proved to stay inside the
    campaigns directory before anything is created.

    Args:
        workspace_root: The workspace's root.
        campaign_id: The campaign's id.

    Returns:
        The ledger's path.

    Raises:
        ValueError: `evidence_review.campaign_path_invalid` for an id with a separator, a
            drive, a parent reference or an absolute path.
    """
    try:
        require_campaign_id(campaign_id)
    except ValueError as error:
        raise ValueError("evidence_review.campaign_path_invalid") from error
    campaigns = (
        workspace_root.resolve() / "runtime" / "artifacts" / "alternative-evidence"
    ) / CAMPAIGN_LEDGER_ROOT
    path = (campaigns / f"{campaign_id}.jsonl").resolve()
    if path.parent != campaigns or path.name != f"{campaign_id}.jsonl":
        raise ValueError("evidence_review.campaign_path_invalid")
    return path
