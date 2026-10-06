"""Pointer-free publication and recursive readback for actor-neutral analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal, Protocol

from ..analysis.contracts import (
    AlternativeEvidenceAnalystBrief,
    AlternativeEvidenceAnalystBriefReceipt,
    AlternativeEvidenceRetrievalAccessReceipt,
    CROAlternativeEvidencePackage,
)
from ..contracts import (
    AlternativeEvidenceRequest,
    AlternativeEvidenceSnapshot,
    SecIssuerRegistrySnapshot,
    seal_contract,
)
from ..documents.contracts import AlternativeEvidenceDocumentSet
from ..retrieval.contracts import (
    CurrentRetrievalGeneration,
    RetrievalGenerationRecord,
)
from .artifacts import (
    AlternativeEvidenceArtifactStore,
    AlternativeEvidencePublicationError,
    validate_document_lineage,
)
from .contracts import AlternativeEvidenceAnalysisPublication, bindings_current


@dataclass(frozen=True, slots=True)
class AlternativeEvidenceAnalysisLineage:
    """Hold the verified records behind one analysis publication."""

    request: AlternativeEvidenceRequest
    registry: SecIssuerRegistrySnapshot
    snapshot: AlternativeEvidenceSnapshot
    document_set: AlternativeEvidenceDocumentSet
    generation: RetrievalGenerationRecord
    access_receipt: AlternativeEvidenceRetrievalAccessReceipt
    analyst_receipt: AlternativeEvidenceAnalystBriefReceipt
    brief: AlternativeEvidenceAnalystBrief
    cro_package: CROAlternativeEvidencePackage


AnalysisCurrentEligibility = Literal["CURRENT", "EXPIRED", "SUPERSEDED", "NOT_EVALUATED"]
"""Whether this publication may be *used now*, which is a separate question.

`NOT_EVALUATED` is what an exact verification returns: it proved the lineage and
was never asked about the clock. An immutable publication that stopped being
readable the moment it expired would not be immutable. `SUPERSEDED` is a
publication sealed under a supported historical binding contract: exact
readback and export by its handle stay open, current use does not, whatever
the clock says.
"""

CURRENT_BINDING_CONTRACT = "CURRENT"


@dataclass(frozen=True, slots=True)
class AlternativeEvidenceAnalysisPublicationView:
    """Pair a publication with its verified lineage and current standing."""

    action: str
    publication: AlternativeEvidenceAnalysisPublication
    lineage: AlternativeEvidenceAnalysisLineage
    current_eligibility: AnalysisCurrentEligibility = "NOT_EVALUATED"
    binding_contract: str = CURRENT_BINDING_CONTRACT
    """`CURRENT`, or `HISTORICAL:<contract hash>` naming the supported historical
    tuple the publication was verified under."""

    @property
    def is_current(self) -> bool:
        """Report whether this publication remains usable under the current contract."""
        return self.current_eligibility == "CURRENT"

    @property
    def is_historical(self) -> bool:
        """Report whether a supported historical contract verified this publication."""
        return self.binding_contract != CURRENT_BINDING_CONTRACT


class HistoricalBindingsPort(Protocol):
    """A supported historical binding tuple, as the identity owner records it."""

    @property
    def bindings(self) -> tuple[str, str, str, str, str, str]:
        """Return the six hashes in this supported binding tuple."""
        ...

    @property
    def contract_hash(self) -> str:
        """Return the identity of this historical binding contract."""
        ...


class AlternativeEvidenceAnalysisPublicationService:
    """Publish immutable analysis lineages without creating a current marker.

    Two boundaries, kept apart: `verify` reopens the recorded graph under the
    current binding contract or one of the explicitly supported historical
    ones and says which; `read` and `replay` decide current use, which a
    historical contract never grants.
    """

    def __init__(
        self,
        store: AlternativeEvidenceArtifactStore,
        *,
        acquisition_binding_hash: str,
        canonicalization_binding_hash: str,
        retrieval_binding_hash: str,
        analysis_policy_hash: str,
        decision_policy_hash: str,
        publication_binding_hash: str,
        supported_historical_bindings: tuple[HistoricalBindingsPort, ...] = (),
    ) -> None:
        """Bind the artifact store and current or supported historical contracts."""
        self.store = store
        self.expected_bindings = (
            acquisition_binding_hash,
            canonicalization_binding_hash,
            retrieval_binding_hash,
            analysis_policy_hash,
            decision_policy_hash,
            publication_binding_hash,
        )
        # Bounded and explicit: a tuple is verified as history only when it is
        # listed here, and the current tuple is never listed as history.
        self.historical_contracts: dict[tuple[str, ...], str] = {
            tuple(value.bindings): f"HISTORICAL:{value.contract_hash[:12]}"
            for value in supported_historical_bindings
            if tuple(value.bindings) != self.expected_bindings
        }

    def binding_contract(self, publication: AlternativeEvidenceAnalysisPublication) -> str | None:
        """Under which contract this publication's bindings verify, or nothing."""
        recorded = (
            publication.acquisition_binding_hash,
            publication.canonicalization_binding_hash,
            publication.retrieval_binding_hash,
            publication.analysis_policy_hash,
            publication.decision_policy_hash,
            publication.publication_binding_hash,
        )
        if recorded == self.expected_bindings or bindings_current(recorded, self.expected_bindings):
            return CURRENT_BINDING_CONTRACT
        return self.historical_contracts.get(recorded)

    def standing(
        self, publication: AlternativeEvidenceAnalysisPublication, *, now: datetime
    ) -> AnalysisCurrentEligibility | None:
        """The standing rule, read from the record.

        Nothing when no admitted contract verifies its bindings, `SUPERSEDED` under a
        historical contract, `EXPIRED` past its expiry, `CURRENT` otherwise. `replay`
        reports it after verifying the lineage; a reader choosing which publications to
        replay applies it to the records alone (X1), so the rule has one owner (V96).
        """
        contract = self.binding_contract(publication)
        if contract is None:
            return None
        if contract != CURRENT_BINDING_CONTRACT:
            return "SUPERSEDED"
        return "EXPIRED" if now > publication.expires_at else "CURRENT"

    def publish(
        self,
        *,
        request: AlternativeEvidenceRequest,
        registry: SecIssuerRegistrySnapshot,
        snapshot: AlternativeEvidenceSnapshot,
        document_set: AlternativeEvidenceDocumentSet,
        generation: CurrentRetrievalGeneration,
        access_receipt: AlternativeEvidenceRetrievalAccessReceipt,
        analyst_receipt: AlternativeEvidenceAnalystBriefReceipt,
        cro_package: CROAlternativeEvidencePackage,
        published_at: datetime,
    ) -> AlternativeEvidenceAnalysisPublicationView:
        """Seal and retain an analysis publication after validating its lineage."""
        brief = analyst_receipt.brief
        validate_document_lineage(
            request=request,
            registry=registry,
            snapshot=snapshot,
            document_set=document_set,
            generation=generation,
            access_receipt=access_receipt,
            brief=brief,
            cro_package=cro_package,
        )
        observed_bindings = (
            snapshot.acquisition_binding_hash,
            document_set.canonicalization_binding_hash,
            generation.retrieval_binding_hash,
            brief.review_binding_hash,
            analyst_receipt.decision_policy_hash,
            self.expected_bindings[-1],
        )
        if not bindings_current(observed_bindings, self.expected_bindings):
            raise AlternativeEvidencePublicationError(
                "alternative_evidence.analysis_publication_binding_mismatch"
            )
        publication = seal_contract(
            AlternativeEvidenceAnalysisPublication,
            "publication_hash",
            request_hash=request.request_hash,
            registry_hash=registry.registry_hash,
            source_snapshot_hash=snapshot.snapshot_hash,
            document_set_hash=document_set.document_set_hash,
            retrieval_generation_hash=generation.generation_hash,
            access_receipt_hash=access_receipt.receipt_hash,
            analyst_brief_receipt_hash=analyst_receipt.receipt_hash,
            analyst_brief_hash=brief.brief_hash,
            cro_package_hash=cro_package.package_hash,
            obligation_hash=cro_package.obligation_hash,
            acquisition_binding_hash=snapshot.acquisition_binding_hash,
            canonicalization_binding_hash=document_set.canonicalization_binding_hash,
            retrieval_binding_hash=generation.retrieval_binding_hash,
            analysis_policy_hash=brief.review_binding_hash,
            decision_policy_hash=analyst_receipt.decision_policy_hash,
            publication_binding_hash=self.expected_bindings[-1],
            published_at=published_at,
            expires_at=snapshot.expires_at,
        )
        # The package is part of what is published: the service owns its write
        # so that a direct publish and the Task's publish leave the same store.
        self.store.publish("cro-packages", cro_package.package_hash, cro_package)
        try:
            existing = self.read(publication.publication_hash, now=published_at)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing.publication != publication:
                raise AlternativeEvidencePublicationError(
                    "alternative_evidence.analysis_publication_identity_reused"
                )
            return replace(existing, action="REUSED_EXACT")
        self.store.publish("analysis-publications", publication.publication_hash, publication)
        durable = self.read(publication.publication_hash, now=published_at)
        return replace(durable, action="PUBLISHED")

    def read(
        self,
        publication_hash: str,
        *,
        now: datetime,
    ) -> AlternativeEvidenceAnalysisPublicationView:
        """Verify exactly, then require the artifact to still be usable now."""
        view = self.verify(publication_hash)
        _aware_now(now)
        if view.is_historical:
            # Verified as history; current use needs the current contract.
            raise AlternativeEvidencePublicationError(
                "alternative_evidence.analysis_publication_authority_superseded"
            )
        if now > view.publication.expires_at:
            raise AlternativeEvidencePublicationError(
                "alternative_evidence.analysis_publication_expired"
            )
        return replace(view, action="READBACK", current_eligibility="CURRENT")

    def verify(self, publication_hash: str) -> AlternativeEvidenceAnalysisPublicationView:
        """Reopen and validate the whole immutable lineage. No clock is consulted."""
        publication = self.store.load(
            "analysis-publications",
            publication_hash,
            AlternativeEvidenceAnalysisPublication,
        )
        request = self.store.load("requests", publication.request_hash, AlternativeEvidenceRequest)
        registry = self.store.load(
            "registries", publication.registry_hash, SecIssuerRegistrySnapshot
        )
        snapshot = self.store.load(
            "snapshots", publication.source_snapshot_hash, AlternativeEvidenceSnapshot
        )
        document_set = self.store.load(
            "document-sets", publication.document_set_hash, AlternativeEvidenceDocumentSet
        )
        generation = self.store.load_retrieval_generation(publication.retrieval_generation_hash)
        access_receipt = self.store.load(
            "retrieval-access-receipts",
            publication.access_receipt_hash,
            AlternativeEvidenceRetrievalAccessReceipt,
        )
        analyst_receipt = self.store.load(
            "analyst-brief-receipts",
            publication.analyst_brief_receipt_hash,
            AlternativeEvidenceAnalystBriefReceipt,
        )
        brief = self.store.load(
            "analyst-briefs", publication.analyst_brief_hash, AlternativeEvidenceAnalystBrief
        )
        cro_package = self.store.load(
            "cro-packages", publication.cro_package_hash, CROAlternativeEvidencePackage
        )
        contract = self.binding_contract(publication)
        if (
            analyst_receipt.brief != brief
            or access_receipt.receipt_hash != brief.access_receipt_hash
            or publication.published_at > publication.expires_at
            or contract is None
        ):
            raise AlternativeEvidencePublicationError(
                "alternative_evidence.analysis_publication_authority_mismatch"
            )
        validate_document_lineage(
            request=request,
            registry=registry,
            snapshot=snapshot,
            document_set=document_set,
            generation=generation,
            access_receipt=access_receipt,
            brief=brief,
            cro_package=cro_package,
        )
        if not (
            publication.obligation_hash == cro_package.obligation_hash == brief.obligation_hash
        ):
            raise AlternativeEvidencePublicationError(
                "alternative_evidence.analysis_obligation_lineage_mismatch"
            )
        return AlternativeEvidenceAnalysisPublicationView(
            action="VERIFIED",
            publication=publication,
            lineage=AlternativeEvidenceAnalysisLineage(
                request=request,
                registry=registry,
                snapshot=snapshot,
                document_set=document_set,
                generation=generation,
                access_receipt=access_receipt,
                analyst_receipt=analyst_receipt,
                brief=brief,
                cro_package=cro_package,
            ),
            binding_contract=contract,
        )

    def replay(
        self,
        publication_hash: str,
        *,
        now: datetime,
    ) -> AlternativeEvidenceAnalysisPublicationView:
        """Reopen history and *report* whether it is still current."""
        verified = self.verify(publication_hash)
        _aware_now(now)
        eligibility = self.standing(verified.publication, now=now)
        assert eligibility is not None  # verify refused a publication no contract admits
        return replace(verified, action="REUSED_EXACT", current_eligibility=eligibility)


def _aware_now(now: datetime) -> None:
    if now.tzinfo is None or now.utcoffset() is None:
        raise AlternativeEvidencePublicationError(
            "alternative_evidence.analysis_readback_clock_invalid"
        )


__all__ = [
    "CURRENT_BINDING_CONTRACT",
    "AlternativeEvidenceAnalysisLineage",
    "AlternativeEvidenceAnalysisPublicationService",
    "AlternativeEvidenceAnalysisPublicationView",
    "AnalysisCurrentEligibility",
    "HistoricalBindingsPort",
]
