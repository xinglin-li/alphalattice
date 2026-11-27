"""Independent source, retrieval and publication identities.

Every identity in this module is Host-owned and actor-neutral. None of them may
name an Agent profile, a Skill, a model, a harness preset or an Agent framework
version. The Agent's own binding lives beside the Agent, in
`alternative_evidence.agent.analyst`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from alphalattice.capabilities.domain_contracts.identity import canonical_hash, schema_structure
from alphalattice.capabilities.domain_contracts.project_layout import resolve_playpen_root
from alphalattice.capabilities.domain_contracts.source_identity import source_rule_closure_hash
from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    CROAlternativeEvidencePackage,
)
from alphalattice.evidence.alternative_evidence.analysis.submissions import (
    ALTERNATIVE_ANALYSIS_POLICY_OWNER,
    ALTERNATIVE_BRIEF_RECEIPT_CATEGORY,
    ALTERNATIVE_DECISION_POLICY_OWNER,
    ALTERNATIVE_POLICY_VERSION,
    AlternativeEvidenceHostPolicyBinding,
    build_alternative_analysis_policy_binding,
    build_alternative_evidence_decision_policy_binding,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceCitation,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSnapshot,
    SecIssuerRegistrySnapshot,
)
from alphalattice.evidence.alternative_evidence.documents.contracts import (
    AlternativeEvidenceDocumentSet,
)
from alphalattice.evidence.alternative_evidence.publication.contracts import (
    AlternativeEvidenceAnalysisPublication,
)
from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
    AlternativeEvidenceRetrievalGeneration,
)
from alphalattice.evidence.alternative_evidence.sources.contracts import (
    AcquiredEvidenceDocumentSet,
    AcquiredEvidenceSourceReferenceSet,
)


def alternative_evidence_playpen_root(source_path: Path) -> Path:
    """Resolve the Playpen checkout without coupling callers to package depth."""
    return resolve_playpen_root(source_path)


@dataclass(frozen=True, slots=True)
class HistoricalEvidenceBindings:
    """Record a binding tuple retained for publication readback.

    The publication reader verifies this tuple as history, never as current
    authority.

    The six hashes are the ones a publication records. A tuple is listed here
    only when a user workspace may hold artifacts sealed under it; listing it
    changes nothing about those artifacts and grants no current eligibility.
    Retirement is a recorded decision, not a side effect of an edit: the
    fifteen tuples that served only QA copies retired when the user declared
    those copies history (D2, 2026-09-23); their results are in the records.
    """

    acquisition_binding_hash: str
    canonicalization_binding_hash: str
    retrieval_binding_hash: str
    analysis_policy_hash: str
    decision_policy_hash: str
    publication_binding_hash: str

    @property
    def bindings(self) -> tuple[str, str, str, str, str, str]:
        """Return the six binding hashes in publication order."""
        return (
            self.acquisition_binding_hash,
            self.canonicalization_binding_hash,
            self.retrieval_binding_hash,
            self.analysis_policy_hash,
            self.decision_policy_hash,
            self.publication_binding_hash,
        )

    @property
    def contract_hash(self) -> str:
        """Return the identity of the retained binding tuple."""
        return str(
            canonical_hash({"kind": "historical-evidence-bindings", "bindings": self.bindings})
        )


SUPPORTED_HISTORICAL_BINDINGS: tuple[HistoricalEvidenceBindings, ...] = ()


@dataclass(frozen=True, slots=True)
class ObservedEvidenceBindings:
    """A tuple this workspace's publications were sealed under that no build lists.

    Its publications read back as history under a contract of their own kind,
    fully verified and never current authority; new work refuses them as
    superseded. Before this, a tuple nobody listed -- the QA
    tuples D2 retired, or one a change forgot to list -- refused the readback of
    every review resting on it.
    """

    bindings: tuple[str, str, str, str, str, str]

    @property
    def contract_hash(self) -> str:
        """Return the identity of the unlisted binding tuple."""
        return str(
            canonical_hash({"kind": "unlisted-evidence-bindings", "bindings": self.bindings})
        )


def observed_historical_bindings(
    recorded: Iterable[tuple[str, str, str, str, str, str]],
    *,
    current: tuple[str, ...],
) -> tuple[ObservedEvidenceBindings, ...]:
    """Return recorded tuples outside current and supported historical bindings."""
    listed = {value.bindings for value in SUPPORTED_HISTORICAL_BINDINGS}
    return tuple(
        ObservedEvidenceBindings(bindings)
        for bindings in sorted(set(recorded))
        if bindings != tuple(current) and bindings not in listed
    )


def alternative_acquisition_binding_hash(playpen_root: Path) -> str:
    """Hash the syntax of source acquisition and snapshot-sealing owners.

    The acquisition service's module is an entry of the closure (its admission, its recorded
    and live builds and its snapshot seal), not the text of four of its methods read at
    runtime, so a comment moves nothing. Installed library versions are provenance
    (LAWS.md ID6), recorded beside a result and never here.
    """
    return str(
        canonical_hash(
            {
                "files": _files(
                    playpen_root,
                    (
                        "src/alphalattice/evidence/alternative_evidence/sources/acquisition.py",
                        "src/alphalattice/evidence/alternative_evidence/sources/recorded.py",
                        "src/alphalattice/evidence/alternative_evidence/sources/sec_edgar.py",
                    ),
                ),
                "schemas": {
                    "request": schema_structure(AlternativeEvidenceRequest),
                    "registry": schema_structure(SecIssuerRegistrySnapshot),
                    "citation": schema_structure(AlternativeEvidenceCitation),
                    "snapshot": schema_structure(AlternativeEvidenceSnapshot),
                    "source_documents": schema_structure(AcquiredEvidenceDocumentSet),
                    "source_references": schema_structure(AcquiredEvidenceSourceReferenceSet),
                },
            }
        )
    )


def alternative_document_binding_hash(playpen_root: Path) -> str:
    """Hash document canonicalization owners and their schema."""
    return str(
        canonical_hash(
            {
                "files": _files(
                    playpen_root,
                    (
                        "src/alphalattice/evidence/alternative_evidence/documents/contracts.py",
                        "src/alphalattice/evidence/alternative_evidence/documents/canonicalization.py",
                        "src/alphalattice/evidence/alternative_evidence/documents/quality.py",
                        "src/alphalattice/evidence/alternative_evidence/documents/structure.py",
                        "src/alphalattice/evidence/alternative_evidence/documents/workspace.py",
                        "src/alphalattice/evidence/alternative_evidence/documents/library.py",
                    ),
                ),
                "schema": schema_structure(AlternativeEvidenceDocumentSet),
                # The store's revisions are bound by content: each revision's SHA-256 is in
                # the snapshot the document set seals (LAWS.md ID8).
                "workspace_owners": _files(
                    playpen_root, ("src/alphalattice/evidence/live_evidence/online_sources.py",)
                ),
            }
        )
    )


def alternative_retrieval_binding_hash(playpen_root: Path) -> str:
    """Every owner that decides what a generation yields, not only what it stores.

    ``session.py`` decides when two hits are one passage and ``packet.py``
    decides which passages reach the analyst and seals the access receipt.
    Neither was hashed here before 2026-09-03, so span identity and packet
    selection could change without any published identity moving. They are
    part of the binding now, which is why correcting span identity rotates
    this hash rather than silently reusing artifacts built under the old rule.
    """
    return str(
        canonical_hash(
            {
                "files": _files(
                    playpen_root,
                    (
                        "src/alphalattice/evidence/alternative_evidence/analysis/packet.py",
                        "src/alphalattice/evidence/alternative_evidence/retrieval/contracts.py",
                        "src/alphalattice/evidence/alternative_evidence/retrieval/service.py",
                        "src/alphalattice/evidence/alternative_evidence/retrieval/session.py",
                    ),
                ),
                "schema": schema_structure(AlternativeEvidenceRetrievalGeneration),
                # The store is bound by content, not by its source: a generation seals the
                # recipe, models and search settings (`index_spec_hash`) and the chunk and
                # vector sets its search reads (LAWS.md ID8).
            }
        )
    )


def alternative_document_publication_binding_hash(playpen_root: Path) -> str:
    """Hash analysis publication owners and their schemas."""
    return str(
        canonical_hash(
            {
                "files": _files(
                    playpen_root,
                    (
                        "src/alphalattice/evidence/alternative_evidence/analysis/cro_package.py",
                        "src/alphalattice/evidence/alternative_evidence/publication/analysis.py",
                        "src/alphalattice/evidence/alternative_evidence/publication/artifacts.py",
                        "src/alphalattice/evidence/alternative_evidence/publication/contracts.py",
                    ),
                ),
                "schemas": {
                    "cro_package": schema_structure(CROAlternativeEvidencePackage),
                    "analysis_publication": (
                        schema_structure(AlternativeEvidenceAnalysisPublication)
                    ),
                },
            }
        )
    )


def current_evidence_binding_tuple(playpen_root: Path) -> tuple[str, str, str, str, str, str]:
    """Return the six currently installed bindings in publication order.

    The tuple the binding gate records (`devtools.architecture.evidence_bindings`).
    """
    return (
        alternative_acquisition_binding_hash(playpen_root),
        alternative_document_binding_hash(playpen_root),
        alternative_retrieval_binding_hash(playpen_root),
        build_alternative_analysis_policy_binding(playpen_root).binding_hash,
        build_alternative_evidence_decision_policy_binding(playpen_root).binding_hash,
        alternative_document_publication_binding_hash(playpen_root),
    )


def _files(root: Path, relative_paths: tuple[str, ...]) -> str:
    return source_rule_closure_hash(
        root=root,
        tracked_paths=relative_paths,
        semantic_owner="alternative_evidence",
        numerical_role="EVIDENCE_RUNTIME_BINDING",
    )


__all__ = [
    "ALTERNATIVE_ANALYSIS_POLICY_OWNER",
    "ALTERNATIVE_BRIEF_RECEIPT_CATEGORY",
    "ALTERNATIVE_DECISION_POLICY_OWNER",
    "ALTERNATIVE_POLICY_VERSION",
    "SUPPORTED_HISTORICAL_BINDINGS",
    "AlternativeEvidenceHostPolicyBinding",
    "HistoricalEvidenceBindings",
    "ObservedEvidenceBindings",
    "alternative_acquisition_binding_hash",
    "alternative_document_binding_hash",
    "alternative_document_publication_binding_hash",
    "alternative_evidence_playpen_root",
    "alternative_retrieval_binding_hash",
    "build_alternative_analysis_policy_binding",
    "build_alternative_evidence_decision_policy_binding",
    "current_evidence_binding_tuple",
    "observed_historical_bindings",
]
