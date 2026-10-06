"""The official SEC source, admitted for live acquisitions only by explicit consent.

Offline is the default: without the operator's consent nothing is composed, and with it the
official client still needs the workspace's network access and a named SEC contact. A denied
transport keeps a consented but offline process from presenting local holdings as fresh, and a
durable campaign's ledger lives beside the Evidence artifact store.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

from alphalattice.control.workspace_runtime.network_access import network_access
from alphalattice.evidence.alternative_evidence.contracts import (
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

SEC_USER_AGENT_VARIABLE = "SEC_USER_AGENT"


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
    maximum_document_bytes: int | None = None
    """The per-document cap the operator declared for this source, when one
    was declared; the recorded policy's cap otherwise. Stated at admission,
    before any request, never raised after a refusal."""
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
    workspace says) and the SEC contact named (`SEC_USER_AGENT` with an
    address). With consent and the network
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
    declared: _Declared = {
        "maximum_document_bytes": maximum_document_bytes,
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
            source=SecEdgarSource(transport, maximum_body_resources=maximum_body_resources),
            network_consent=True,
            transport_origin="INJECTED",
            **declared,
        )
    values = dict(os.environ if environment is None else environment)
    # The workspace's typed network control decides, the operator's offline switch first (V53).
    if not network_access(workspace_root, values).allowed:
        return OfficialSourceAdmission(
            source=SecEdgarSource(
                DeniedSecOfficialTransport(), maximum_body_resources=maximum_body_resources
            ),
            network_consent=True,
            transport_origin="DENIED",
            refusal_code="evidence_review.network_disabled",
            **declared,
        )
    user_agent = values.get(SEC_USER_AGENT_VARIABLE, "").strip()
    if "@" not in user_agent:
        return OfficialSourceAdmission(
            source=None,
            network_consent=True,
            transport_origin="NONE",
            refusal_code="evidence_review.sec_user_agent_required",
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
                maximum_total_attempts=maximum_total_attempts,
                maximum_total_response_bytes=maximum_total_response_bytes,
                ledger=ledger,
            ),
            maximum_body_resources=maximum_body_resources,
            ledger=ledger,
        ),
        network_consent=True,
        transport_origin="OFFICIAL_HTTP",
        **declared,
    )


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
