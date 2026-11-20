"""The sealed evidence-review capability package the workspace and HTTP route
suites install.

Test support beside its owner; nothing here is product authority or evidence.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from alphalattice.control.product_host.composition.evidence_review_workspace import (
    EvidenceReviewArtifactBinding,
    EvidenceReviewModelProfile,
    EvidenceReviewWorkspaceManifest,
    RecordedEvidenceDocumentBundle,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceEvidenceReview,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceClass,
    SecIssuerRegistryEntry,
    SecIssuerRegistrySnapshot,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import (
    RecordedEvidenceDocument,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    AdmittedListingTicker,
    AdmittedListingTickerAuthority,
    seal_portfolio_evidence_contract,
)

NOW = datetime(2026, 8, 12, 18, tzinfo=UTC)

CAPABILITY_HASH = "b" * 64

DEEPSEEK_ENVIRONMENT = {
    "DEEPSEEK_API_KEY": "test-only-not-a-real-secret",
    "DEEPSEEK_BASE_URL": "https://api.deepseek.example/v1",
}


def _write(path: Path, value: Any) -> EvidenceReviewArtifactBinding:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump_json(indent=2).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    identity = next(
        getattr(value, name)
        for name in ("registry_hash", "authority_hash", "bundle_hash")
        if hasattr(value, name)
    )
    return EvidenceReviewArtifactBinding(
        relative_path=path.relative_to(path.parent.parent).as_posix(),
        file_sha256=hashlib.sha256(payload).hexdigest(),
        content_hash=identity,
    )


def _package(
    workspace: Path,
    *,
    semantic_capability_hash: str = CAPABILITY_HASH,
    managed: bool = True,
    corpus: tuple[RecordedEvidenceDocument, ...] | None = None,
) -> tuple[ResearchWorkspaceEvidenceReview, Path]:
    """The authority a workspace ships.

    `semantic_capability_hash` is a parameter because a manifest verified
    through the *real* reader must declare the capability that reader
    actually computes, not a stub constant.
    """

    authority = workspace / "authority"
    issuers = (
        ("AAPL", "0000320193", "Apple Inc."),
        ("MSFT", "0000789019", "Microsoft Corporation"),
    )
    entities = {"AAPL"} if corpus is None else {document.entity_id for document in corpus}
    registry = seal_contract(
        SecIssuerRegistrySnapshot,
        "registry_hash",
        captured_at=NOW - timedelta(days=1),
        entries=tuple(
            SecIssuerRegistryEntry(entity_id=ticker, ticker=ticker, cik=cik, legal_name=name)
            for ticker, cik, name in issuers
            if ticker in entities
        ),
        source_content_hash="c" * 64,
    )
    listing = seal_portfolio_evidence_contract(
        AdmittedListingTickerAuthority,
        "authority_hash",
        entries=tuple(
            AdmittedListingTicker(listing_id=f"US-{entry.ticker}", ticker=entry.ticker)
            for entry in registry.entries
        ),
    )
    documents = RecordedEvidenceDocumentBundle.create(
        corpus
        if corpus is not None
        else (
            RecordedEvidenceDocument(
                entity_id="AAPL",
                source_right="USER_PROVIDED_FOR_LOCAL_RESEARCH",
                evidence_class=AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,
                document_type="10-Q",
                revision="0000320193-26-000079",
                published_at=NOW - timedelta(days=3),
                captured_at=NOW - timedelta(days=2),
                available_at=NOW - timedelta(days=2),
                text="Recorded issuer filing with a material legal contingency.",
                immutable_source=True,
                limitations=("Local recorded copy; verify against the official filing.",),
            ),
        )
    )
    registry_binding = _write(authority / "registry.json", registry)
    listing_binding = _write(authority / "listing.json", listing)
    documents_binding = _write(authority / "documents.json", documents)
    (authority / "semantic-model").mkdir(parents=True, exist_ok=True)
    profile = (
        None
        if not managed
        else EvidenceReviewModelProfile.create(
            model_name="deepseek-v4-flash",
            base_url=DEEPSEEK_ENVIRONMENT["DEEPSEEK_BASE_URL"],
        )
    )
    manifest = EvidenceReviewWorkspaceManifest.create(
        authority_id="qa-real-authority",
        issuer_registry=registry_binding,
        listing_authority=listing_binding,
        recorded_documents=documents_binding,
        semantic_model_relative_path="authority/semantic-model",
        semantic_capability_hash=semantic_capability_hash,
        model_profile=profile,
    )
    manifest_path = authority / "evidence-review-authority.json"
    payload = manifest.model_dump_json(indent=2).encode("utf-8") + b"\n"
    manifest_path.write_bytes(payload)
    return (
        ResearchWorkspaceEvidenceReview(
            relative_path="authority/evidence-review-authority.json",
            file_sha256=hashlib.sha256(payload).hexdigest(),
        ),
        authority / "registry.json",
    )
