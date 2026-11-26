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
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash


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


SUPPORTED_HISTORICAL_BINDINGS: tuple[HistoricalEvidenceBindings, ...] = (
    # The tuple at f2af8500d859b824ed202e292a0eedd117f8529c, the backend baseline
    # the lead accepted (unchanged through the first-release integration's
    # T1-T3, c40b6cad..9b1de175): the QA copies of its acceptance (ae-fr-live,
    # ae-fr-fresh-*) and any workspace run on the accepted baseline sealed
    # analyses and reviews under it. T4's repair of the report period rotates
    # the canonicalization binding (structure.py). Listed so those artifacts
    # re-read exactly as sealed; it grants no current eligibility.
    HistoricalEvidenceBindings(
        acquisition_binding_hash=(
            "392ef08ab318b0c184f5a35d51981201196f120714426723848c71415f82c346"
        ),
        canonicalization_binding_hash=(
            "9922289905e55bd26ca771760f92894b65c0acfd5b7811af97e4e1b7e800c536"
        ),
        retrieval_binding_hash=("ffc203914422a488e17a4469b7fe538ebd5421d3213549b2f6a2f9159680ed48"),
        analysis_policy_hash=("d3e3cbeb09c8f86d0de7c227e48bae10194a1d0ebfd6bdf69d4add7af0a04114"),
        decision_policy_hash=("e0b12bbe19341c4f8212b9843ada1790b0c6adaed8a2fbccc9a511c2b7c08577"),
        publication_binding_hash=(
            "01a32a4c23e9a24d3a924c1bcec229ddd9332868721f637093ffeca53403f120"
        ),
    ),
    # The tuple of release 0.1.3 (develop through bd11bbccf): a person's workspace may hold
    # analyses and reviews sealed under it. The SEC source's clamp to its admission's documents
    # per issuer (2026-10-09) rotates the acquisition binding (sec_edgar.py). Listed so those
    # artifacts re-read exactly as sealed; it grants no current eligibility.
    HistoricalEvidenceBindings(
        acquisition_binding_hash=(
            "259473047bff6003faf3cd2e8f72d42e2f4a7a87ddf89ca3ce53adbd57b7cd47"
        ),
        canonicalization_binding_hash=(
            "8b5a16e890540e4fad64af6fc3e04046b024173e4b586e5419e2837c0856ca44"
        ),
        retrieval_binding_hash=("45ee521713056588758ff624627b3b77db7230b7ce2b7961567e34a94e726c59"),
        analysis_policy_hash=("1e0a84f859aade29d3415002c27a851bfa63034cda358c4653261d80c036d301"),
        decision_policy_hash=("fd1161bf40ca1077ad8d9a5a1a18f7ec2990d7ed5cb60b6a9ccad3e8fef4993c"),
        publication_binding_hash=(
            "906d733cbae3d8904964959d1ebd72f8182f0d316f8e54aa19d69e28f575742a"
        ),
    ),
    # No kept workspace holds a publication under the removed bindings. Their readback
    # is observed history, never current authority (`ObservedEvidenceBindings`).
)


@dataclass(frozen=True, slots=True)
class ObservedEvidenceBindings:
    """A tuple this workspace's publications were sealed under that no build lists.

    Its publications read back as history under a contract of their own kind,
    fully verified and never current authority; new work refuses them as
    superseded (binding plan, E2). Before this, a tuple nobody listed -- the QA
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
                    ),
                ),
                "schema": schema_structure(AlternativeEvidenceDocumentSet),
                "workspace_owners": _files(
                    playpen_root,
                    (
                        "src/alphalattice/kernel/live_evidence/online_sources.py",
                        "src/alphalattice/kernel/knowledge/retrieval.py",
                    ),
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
                "workspace_owners": _files(
                    playpen_root,
                    (
                        "src/alphalattice/kernel/knowledge/retrieval.py",
                        "src/alphalattice/kernel/knowledge/retrieval_contracts.py",
                        "src/alphalattice/kernel/knowledge/hybrid.py",
                        "src/alphalattice/kernel/knowledge/hybrid_contracts.py",
                        "src/alphalattice/kernel/knowledge/_reranking.py",
                    ),
                ),
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
