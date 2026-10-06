"""Stage-owned document intelligence without duplicating source or Workspace owners."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from heapq import heappop, heappush
from itertools import count
from pathlib import Path
from threading import Condition, Lock, get_ident, local
from typing import Protocol, cast
from uuid import UUID

from pydantic import BaseModel

from alphalattice.evidence.alternative_evidence.publication.usage import (
    record_provider_stage_usage,
)
from alphalattice.kernel.knowledge._embeddings import VERIFIED_PACKS
from alphalattice.kernel.knowledge.hybrid import (
    VectorBlockSource,
    committed_database_relative,
    committed_payload_slice,
    verified_block_map,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    RECIPE_MINILM_CPU,
    HybridGenerationAnchor,
    HybridIndexSpec,
    PairScoreAdmission,
    reranker_context_hash,
    vector_asset_key,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.protocols.actor_execution import ActorKind, AgentExecutionBinding
from alphalattice.protocols.actor_execution.answers import AnswerProblem
from alphalattice.protocols.actor_execution.usage import ProviderStageUsage

from ..analysis.contracts import (
    AlternativeEvidenceAnalystAnswer,
    AlternativeEvidenceAnalystBrief,
    AlternativeEvidenceAnalystBriefReceipt,
    AlternativeEvidenceResearchObligation,
    AlternativeEvidenceRetrievalAccessReceipt,
    WholeFilingsRecord,
)
from ..analysis.matters import TOPIC_LANES_ALLOCATION_ID
from ..analysis.packet import (
    AlternativeEvidencePacket,
    continue_routed_reads,
    delivered_ranges,
    plan_table_pages,
    query_program_hash,
    route_session,
    routing_record,
    select_matter_evidence,
    select_routed_evidence,
    shared_inventory,
    table_view_renderable,
)
from ..analysis.submissions import seal_alternative_evidence_analyst_brief
from ..contracts import (
    AlternativeEvidenceAdmission,
    AlternativeEvidenceReadingDepth,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSnapshot,
    SecCompanyFactsSnapshot,
    SecIssuerRegistrySnapshot,
    seal_contract,
)
from ..documents.canonicalization import canonicalize_source_documents
from ..documents.contracts import AlternativeEvidenceDocumentSet
from ..documents.workspace import AlternativeEvidenceDocumentPublisher
from ..publication.analysis import AlternativeEvidenceAnalysisPublicationService
from ..publication.artifacts import (
    AlternativeEvidenceArtifactStore,
    AlternativeEvidencePublicationError,
)
from ..publication.contracts import (
    RETRIEVAL_BINDING_ROLE,
    AlternativeEvidenceAnalysisPublication,
)
from ..retrieval.contracts import (
    WHOLE_FILINGS_FORMAT,
    AlternativeEvidenceResolvedSpan,
    AlternativeEvidenceResolvedSpanSet,
    AlternativeEvidenceRetrievalGeneration,
    PairScoreBlockRecord,
    PairScoreCommitmentRecord,
    RetrievalGenerationRecord,
    WholeFilingsGeneration,
)
from ..retrieval.service import (
    AlternativeEvidenceRetrievalService,
    IndexFactory,
    RetrieverFactory,
)
from ..retrieval.session import AlternativeEvidenceRetrievalSession
from ..sources.acquisition import AlternativeEvidenceAcquisitionService
from ..sources.contracts import (
    SOURCE_COMMIT_CATEGORY,
    SOURCE_DEFERRAL_CATEGORY,
    AcquiredEvidenceDocument,
    AcquiredEvidenceDocumentReference,
    AcquiredEvidenceDocumentSet,
    AcquiredEvidenceSourceCommit,
    AcquiredEvidenceSourceDeferral,
    AcquiredEvidenceSourceReferenceSet,
    AcquiredEvidenceSourceSet,
    SecFilingInventoryEntry,
)
from ..sources.recorded import RecordedEvidenceDocument
from ..sources.sec_edgar import SecEdgarSource
from .execution import CpuBudgetStore
from .history import require_current_selection
from .identity import (
    SUPPORTED_HISTORICAL_BINDINGS,
    ObservedEvidenceBindings,
    alternative_document_binding_hash,
    alternative_document_publication_binding_hash,
    alternative_evidence_playpen_root,
    alternative_retrieval_binding_hash,
    build_alternative_analysis_policy_binding,
    build_alternative_evidence_decision_policy_binding,
    observed_historical_bindings,
)
from .reuse import (
    ANALYSIS_PUBLICATION_CATEGORY,
    DOCUMENT_SET_CATEGORY,
    PAIR_SCORE_COMMITMENT_CATEGORY,
    RECEIPT_CATEGORY,
    SNAPSHOT_CATEGORY,
    SOURCE_SET_CATEGORY,
    SPAN_SET_CATEGORY,
    RecordDays,
    SealedCanonicals,
    SealedComparisons,
    SealedExcerpts,
    SealedReadings,
    SealedSelections,
    record_days,
    sealed_field,
)


@dataclass(frozen=True, slots=True)
class AlternativeEvidenceActorSubmission:
    """Carry an accepted actor answer and its execution facts.

    The Host records dropped content, calls, repairs, and usage.
    """

    answer: AlternativeEvidenceAnalystAnswer
    actor_kind: ActorKind
    actor_id: str
    dropped: tuple[AnswerProblem, ...] = ()
    agent_execution: AgentExecutionBinding | None = None
    model_call_count: int = 0
    protocol_repair_count: int = 0
    provider_usage: ProviderStageUsage | None = None
    """What the Provider reported for this stage, or nothing when it reported none."""


def _carried(
    receipt: AlternativeEvidenceRetrievalAccessReceipt, *, without: Collection[str]
) -> dict[str, object]:
    """A sealed receipt's own records carried into a successor as the models
    they are (a python dump would hand the sealer dicts in model-typed
    fields: the same identity, serialized under a warning on every
    session): every field but the receipt's identity and the ones the
    successor seals itself -- a reuse over another request, a continuation
    over the same one.
    """
    return {
        name: getattr(receipt, name)
        for name in type(receipt).model_fields
        if name != "receipt_hash" and name not in without
    }


@dataclass(frozen=True, slots=True)
class AlternativeEvidenceAnalysisResult:
    """Carry the sealed analysis receipts, brief, and resolved spans."""

    access_receipt: AlternativeEvidenceRetrievalAccessReceipt
    analyst_receipt: AlternativeEvidenceAnalystBriefReceipt
    brief: AlternativeEvidenceAnalystBrief
    resolved_spans: tuple[AlternativeEvidenceResolvedSpan, ...]


class AlternativeEvidenceAnalysisActor(Protocol):
    """One tool-free extraction over one Host-built packet.

    No Host policy crosses this boundary. An actor declares the identity of its
    own process; the Host keeps its analysis and decision policies private and
    applies them at the sealer. A Human or external automation never learns --
    and cannot influence -- the identity its submission is judged under.
    """

    @property
    def process_binding_hash(self) -> str:
        """Return the actor process binding without Host policy."""
        ...

    def __call__(self, *, packet: AlternativeEvidencePacket) -> AlternativeEvidenceActorSubmission:
        """Produce one submission from a Host-built packet."""
        ...


class SubmittedAlternativeEvidenceAnalysisActor:
    """One already-screened answer, from a Human, an Agent or an automation."""

    def __init__(
        self,
        *,
        actor_kind: ActorKind,
        actor_id: str,
        answer: AlternativeEvidenceAnalystAnswer,
        dropped: tuple[AnswerProblem, ...] = (),
        agent_execution: AgentExecutionBinding | None = None,
    ) -> None:
        """Bind a screened answer to its admitted actor identity."""
        if actor_kind not in {
            ActorKind.HUMAN,
            ActorKind.EXTERNAL_AUTOMATION,
            ActorKind.INSTALLED_AGENT,
        }:
            raise ValueError("alternative_evidence.actor_kind_not_admitted")
        if (agent_execution is not None) != (actor_kind is ActorKind.INSTALLED_AGENT):
            raise ValueError("alternative_evidence.actor_execution_evidence_invalid")
        self.actor_kind = actor_kind
        self.actor_id = actor_id
        self.answer = AlternativeEvidenceAnalystAnswer.model_validate(answer)
        self.dropped = dropped
        self.agent_execution = agent_execution
        self.packets_seen: list[AlternativeEvidencePacket] = []

    @property
    def process_binding_hash(self) -> str:
        """Return the submitted-answer actor process binding."""
        return str(
            canonical_hash(
                {
                    "owner": "alternative_evidence.runtime.submitted_analysis",
                    "schema": schema_structure(AlternativeEvidenceAnalystAnswer),
                }
            )
        )

    def __call__(self, *, packet: AlternativeEvidencePacket) -> AlternativeEvidenceActorSubmission:
        """Record the packet and return the already screened answer."""
        self.packets_seen.append(packet)
        return AlternativeEvidenceActorSubmission(
            answer=self.answer,
            dropped=self.dropped,
            actor_kind=self.actor_kind,
            actor_id=self.actor_id,
            agent_execution=self.agent_execution,
        )


@dataclass(frozen=True, slots=True)
class EvidenceStorageExpansion:
    """Estimate peak storage expansion per fetched source byte.

    Include the original itself, its canonical text, the vector
    blocks of its passages and the index projection of its generation.
    Measured on the retained QA workspace ae-f1-fresh (2026-09-18: 95.9 MB
    of originals, canonical text 0.057 of that, vectors 2.72 times the
    canonical text, an index projection 20.5 times the canonical text of its
    corpus); a bound for the preflight, never a promise, and stated here so
    a different measurement replaces it in one place.
    """

    canonical_per_raw: float = 0.06
    vector_per_canonical: float = 2.8
    index_per_canonical: float = 21.0
    staging_per_raw: float = 1.0
    """The staged copy of a body before it is renamed into place."""

    @property
    def peak_per_raw(self) -> float:
        derived = self.canonical_per_raw * (
            1.0 + self.vector_per_canonical + self.index_per_canonical
        )
        return 1.0 + self.staging_per_raw + derived


SOURCE_COMMITMENT_CORRUPT = "alternative_evidence.source_commitment_corrupt"
"""A sealed source set or per-document commit that does not read as sealed.
The lookup is rebuildable; the commitments it is rebuilt from are the
authoritative provenance of every retained body, and one that is damaged is
an integrity failure of the store -- named, never a cache miss that a
refetch and a replacement record would paper over."""


def _commitment_failure_code(error: OSError | ValueError) -> str:
    """The public item refusal for one damaged source commitment."""
    if isinstance(error, FileNotFoundError):
        return "alternative_evidence.artifact_missing"
    if isinstance(error, OSError):
        return "alternative_evidence.artifact_unavailable"
    return "alternative_evidence.artifact_tampered"


class VerifiedSourceLookup:
    """Retained SEC bodies by resource identity, for reuse without a request.

    Rebuilt from the store's sealed commitments -- every reference of every
    source set and every per-document commit -- and kept current as this
    runtime commits more. A body is committed at or after its filing was
    accepted, so a filing is looked for only in the commitments sealed since
    the day before (Z2); what lists every holding reads them all. A hit is the reference whose
    issuer and accession match the inventory entry (the document name too
    when the reference recorded one); its bytes are read from their content
    address and verified against the reference before the document is
    returned, so a mutable index row confers nothing and a body that does not
    verify is a named integrity refusal, never a silent refetch. A
    commitment that does not read as sealed damages the store's provenance:
    the intact commitments are still indexed, but no resource is served or
    fetched over a damaged store until it is repaired by explicit action --
    the runtime never overwrites or reseals a historical artifact.
    """

    def __init__(
        self,
        artifacts: AlternativeEvidenceArtifactStore,
        documents: AlternativeEvidenceDocumentPublisher,
        *,
        source_days: RecordDays,
        commit_days: RecordDays,
        deferral_days: RecordDays,
    ) -> None:
        self._artifacts = artifacts
        self._documents = documents
        self._unread = (source_days.reader(), commit_days.reader(), deferral_days.reader())
        self._by_accession: dict[str, list[AcquiredEvidenceDocumentReference]] = {}
        self._committed: set[tuple[str, str]] = set()
        """`(accession, content sha)` pairs a per-document commit names."""
        self._deferrals: dict[tuple[str, str, str, str], AcquiredEvidenceSourceDeferral] = {}
        """The latest sealed deferral by resource identity."""
        self.scanned_sets = 0
        self.scanned_commits = 0
        self.damaged: tuple[str, ...] = ()
        """The commitments that did not read as sealed, each named with its
        category, identity and the reason; empty for an intact store."""
        self.damaged_artifacts: tuple[tuple[str, str, str], ...] = ()
        """Independently unreadable commitments as ``(category, identity, code)``.

        Collection readers may name these items while retaining intact records.
        Strict source-use paths still refuse the whole source commitment index.
        """

    def _index(
        self, accepted_at: datetime | None
    ) -> dict[str, list[AcquiredEvidenceDocumentReference]]:
        """The commitments sealed since the day before `accepted_at` -- every
        one when None -- each read once.
        """
        sources, commits, deferrals = self._unread
        damaged = list(self.damaged)
        damaged_artifacts = list(self.damaged_artifacts)
        index = self._by_accession
        for identity in sources.since(accepted_at):
            try:
                source_set = self._artifacts.load_source_set(identity)
            except (OSError, ValueError) as error:
                damaged_artifacts.append(
                    ("source-document-sets", identity, _commitment_failure_code(error))
                )
                damaged.append(f"source-document-sets/{identity}: {error}")
                continue
            self.scanned_sets += 1
            if isinstance(source_set, AcquiredEvidenceSourceReferenceSet):
                for reference in source_set.documents:
                    self._add(index, reference)
        for identity in commits.since(accepted_at):
            try:
                commit = self._artifacts.load(
                    SOURCE_COMMIT_CATEGORY, identity, AcquiredEvidenceSourceCommit
                )
            except (OSError, ValueError) as error:
                damaged_artifacts.append(
                    (SOURCE_COMMIT_CATEGORY, identity, _commitment_failure_code(error))
                )
                damaged.append(f"{SOURCE_COMMIT_CATEGORY}/{identity}: {error}")
                continue
            self.scanned_commits += 1
            self._add(index, commit.reference)
            self._committed.add((commit.reference.revision, commit.reference.content_sha256))
        for identity in deferrals.since(accepted_at):
            try:
                deferral = self._artifacts.load(
                    SOURCE_DEFERRAL_CATEGORY, identity, AcquiredEvidenceSourceDeferral
                )
            except (OSError, ValueError) as error:
                damaged_artifacts.append(
                    (SOURCE_DEFERRAL_CATEGORY, identity, _commitment_failure_code(error))
                )
                damaged.append(f"{SOURCE_DEFERRAL_CATEGORY}/{identity}: {error}")
                continue
            self._add_deferral(deferral)
        self.damaged = tuple(damaged)
        self.damaged_artifacts = tuple(damaged_artifacts)
        return index

    def _require_intact(self, accepted_at: datetime | None) -> None:
        self._index(accepted_at)
        if self.damaged:
            raise ValueError(f"{SOURCE_COMMITMENT_CORRUPT}: {'; '.join(self.damaged)[:600]}")

    def _add_deferral(self, deferral: AcquiredEvidenceSourceDeferral) -> None:
        held = self._deferrals.get(deferral.resource_key)
        if held is None or deferral.observed_at > held.observed_at:
            self._deferrals[deferral.resource_key] = deferral

    def add_deferral(self, deferral: AcquiredEvidenceSourceDeferral) -> None:
        self._add_deferral(deferral)

    def deferral(self, entry: SecFilingInventoryEntry) -> AcquiredEvidenceSourceDeferral | None:
        """Return the latest sealed deferral for this exact resource.

        A damaged store raises an integrity refusal.
        """
        self._require_intact(entry.accepted_at)
        return self._deferrals.get(
            ("SEC_EDGAR", entry.cik, entry.accession, entry.primary_document)
        )

    def deferrals(self) -> tuple[AcquiredEvidenceSourceDeferral, ...]:
        self._index(None)
        return tuple(self._deferrals[key] for key in sorted(self._deferrals))

    def committed(self, reference: AcquiredEvidenceDocumentReference) -> bool:
        """Check whether a commit already names these accession bytes.

        A body restored under that commit needs no second record.
        """
        self._index(reference.accepted_at)
        return (reference.revision, reference.content_sha256) in self._committed

    @staticmethod
    def _add(
        index: dict[str, list[AcquiredEvidenceDocumentReference]],
        reference: AcquiredEvidenceDocumentReference,
    ) -> None:
        if reference.source_name != "SEC_EDGAR":
            return
        held = index.setdefault(reference.revision, [])
        if all(value.content_sha256 != reference.content_sha256 for value in held):
            held.append(reference)

    def add(self, reference: AcquiredEvidenceDocumentReference, *, committed: bool) -> None:
        self._add(self._by_accession, reference)
        if committed:
            self._committed.add((reference.revision, reference.content_sha256))

    def holdings(
        self, *, entity_id: str, cik: str | None
    ) -> tuple[AcquiredEvidenceDocumentReference, ...]:
        """List an issuer's committed references, one per accession.

        Match by CIK where recorded or by security otherwise. This listing
        does not read document bytes.
        """
        found: dict[str, AcquiredEvidenceDocumentReference] = {}
        for accession, references in self._index(None).items():
            for reference in references:
                matches = (
                    reference.source_cik == cik
                    if reference.source_cik is not None and cik is not None
                    else reference.entity_id == entity_id
                )
                if matches and accession not in found:
                    found[accession] = reference
        return tuple(found[key] for key in sorted(found))

    def holds(self, entry: SecFilingInventoryEntry) -> bool:
        """Check cheaply whether a reference names a present body.

        Preflight uses this without verifying bytes. A damaged store raises
        the same integrity refusal as ``find``.
        """
        self._require_intact(entry.accepted_at)
        for value in self._by_accession.get(entry.accession, ()):
            matches = (
                value.source_cik == entry.cik
                if value.source_cik is not None
                else value.entity_id == entry.entity_id
            )
            if matches and self._documents.holds_source_object(value.content_sha256):
                return True
        return False

    def find(self, entry: SecFilingInventoryEntry) -> AcquiredEvidenceDocument | None:
        self._require_intact(entry.accepted_at)
        candidates = [
            value
            for value in self._by_accession.get(entry.accession, ())
            if (
                value.source_cik == entry.cik
                if value.source_cik is not None
                else value.entity_id == entry.entity_id
            )
            and (
                value.source_document_name is None
                or value.source_document_name == entry.primary_document
            )
        ]
        # A reference that recorded the document name is the surer match.
        candidates.sort(key=lambda value: value.source_document_name is None)
        for reference in candidates:
            try:
                return self._documents.resolve_source_object(reference)
            except ValueError as error:
                if "source_object_missing" in str(error):
                    continue  # bytes evicted: not held, never a refusal
                raise
        return None


WHOLE_PIECE_CHARACTERS = 4_000
"""The most one piece of a filing delivered whole holds: pieces end at line
bounds, and a longer line is cut at the bound."""


def _whole_pieces(text: str) -> tuple[tuple[int, int], ...]:
    """A text's pieces as (start, end) character offsets: consecutive lines up
    to `WHOLE_PIECE_CHARACTERS`, together the text itself.
    """
    pieces: list[tuple[int, int]] = []
    start = position = 0
    for line in text.splitlines(keepends=True):
        while len(line) > WHOLE_PIECE_CHARACTERS:
            if position > start:
                pieces.append((start, position))
                start = position
            pieces.append((position, position + WHOLE_PIECE_CHARACTERS))
            position += WHOLE_PIECE_CHARACTERS
            start = position
            line = line[WHOLE_PIECE_CHARACTERS:]
        if position + len(line) - start > WHOLE_PIECE_CHARACTERS and position > start:
            pieces.append((start, position))
            start = position
        position += len(line)
    if position > start:
        pieces.append((start, position))
    return tuple(pieces)


class ChunkCount(Protocol):
    """A build's progress: the chunks the index cuts, then each batch embedded."""

    def expect(self, total: int) -> None: ...

    def advance(self, count: int = 1) -> None: ...


class EvidenceWriter:
    """One preparation stage at a time touches the runtime's shared state.

    The stores, the sealed caches, the storage admission and the workspace's
    own lock were written for one stage at a time, and they still see one:
    a stage runs holding the writer. It gives the writer up for the spans that
    touch nothing shared -- a model call inside a build
    (`yielding_during_inference`) and a unit's routed selection over its own
    session -- so units prepare at once where the time goes and in turn
    everywhere else. Held again by the thread that holds it, it is a no-op.

    Who goes next is decided, not raced: a thread coming back to a stage it
    is in (from a model call, from its routing, or a routing's commit) goes
    before a stage that has not begun, and among each the unit that runs
    first in the book (its `rank`, heaviest first) goes first. A long stage
    (an acquisition, a canonicalization) `pause`s at its safe points -- a
    body committed, a document taken up -- so the units between two model
    calls pass through instead of waiting for the whole stage. Measured on
    the median book's first run of units at once, a build stalled there.
    """

    _RETURNING = 0
    _BEGINNING = 1

    def __init__(self) -> None:
        """Initialize deterministic writer turns for preparation stages."""
        self._turn = Condition(Lock())
        self._owner: int | None = None
        self._waiting: list[tuple[int, int, int]] = []
        self._tickets = count()
        self._ranks = local()

    def _acquire(self, kind: int, rank: int) -> None:
        with self._turn:
            ticket = (kind, rank, next(self._tickets))
            heappush(self._waiting, ticket)
            while self._owner is not None or self._waiting[0] != ticket:
                self._turn.wait()
            heappop(self._waiting)
            self._owner = get_ident()

    def _release(self) -> None:
        with self._turn:
            self._owner = None
            self._turn.notify_all()

    def _rank(self) -> int:
        return int(getattr(self._ranks, "rank", 0))

    @contextmanager
    def held(self, *, rank: int | None = None) -> Iterator[None]:
        """Hold the writer for one stage step.

        A unit at ``rank`` begins a stage; a call without ``rank`` returns to
        a stage already in progress.
        """
        if self._owner == get_ident():
            yield
            return
        if rank is None:
            self._acquire(self._RETURNING, self._rank())
        else:
            self._ranks.rank = rank
            self._acquire(self._BEGINNING, rank)
        try:
            yield
        finally:
            self._release()

    @contextmanager
    def released(self) -> Iterator[None]:
        """Give the writer up for a span, if this thread holds it; take it back after."""
        if self._owner != get_ident():
            yield
            return
        self._release()
        try:
            yield
        finally:
            self._acquire(self._RETURNING, self._rank())

    def pause(self) -> None:
        """Yield at a long stage's safe point to returning work.

        Resume before a stage that has not begun.
        """
        if self._owner != get_ident():
            return
        with self._turn:
            if not self._waiting or self._waiting[0][0] != self._RETURNING:
                return
        self._release()
        self._acquire(self._RETURNING, self._rank())


class AlternativeEvidenceDocumentIntelligenceRuntime:
    """Compose existing owners into the single active document-intelligence route."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        workspace_root: Path,
        model_root: Path,
        index_factory: IndexFactory | None = None,
        retriever_factory: RetrieverFactory | None = None,
        retrieval_recipe: str = RECIPE_MINILM_CPU,
    ) -> None:
        """Compose acquisition, documents, retrieval, and sealed reuse owners."""
        self.playpen_root = alternative_evidence_playpen_root(Path(__file__))
        self.retrieval_recipe = retrieval_recipe
        self.acquisition = AlternativeEvidenceAcquisitionService(artifact_root=artifact_root)
        self.artifacts = self.acquisition.artifacts
        self.document_binding_hash = alternative_document_binding_hash(self.playpen_root)
        self.retrieval_binding_hash = alternative_retrieval_binding_hash(self.playpen_root)
        self.analysis_policy = build_alternative_analysis_policy_binding(self.playpen_root)
        self.decision_policy = build_alternative_evidence_decision_policy_binding(self.playpen_root)
        self.analysis_policy_hash = self.analysis_policy.binding_hash
        self.publication_binding_hash = alternative_document_publication_binding_hash(
            self.playpen_root
        )
        self.recorded_acquisition_document_count = 0
        self.canonicalization_document_count = 0
        self.selection_reuse_count = 0
        self.generation_builds_avoided = 0
        """Builds `build_retrieval` did not run because the selection the
        caller was about to make is reused whole from a sealed receipt
        (`reuse_for`): no index materialized, opened or proved."""
        self.storage_admission: Callable[[int], None] | None = None
        """The workspace's capacity admission, `(additional_bytes)`, bound by
        the Host that owns the storage budget; every write this runtime makes
        -- a source set, canonical blobs, a vector payload, an index -- is
        admitted through it before it is placed. Unbound, nothing is admitted
        and nothing is refused: a library use without a Host has no budget."""
        self.block_source_refusals: list[dict[str, str]] = []
        """Sealed generations the last block source refused to serve, by index
        id, generation hash and the kernel's failure code: a composition that
        did not prove as a whole, a sidecar whose keys the committed manifest
        does not derive, or a manifest not on disk. Each is an explicit
        fallback to embedding, never a silent one."""
        self.documents = AlternativeEvidenceDocumentPublisher(workspace_root)
        self.record_days = record_days(self.artifacts)
        """The day each record the reuse indexes and the analysis readers
        choose from was sealed, kept beside the records (Z2)."""
        self.local_sources = VerifiedSourceLookup(
            self.artifacts,
            self.documents,
            source_days=self.record_days[SOURCE_SET_CATEGORY],
            commit_days=self.record_days[SOURCE_COMMIT_CATEGORY],
            deferral_days=self.record_days[SOURCE_DEFERRAL_CATEGORY],
        )
        self.storage_expansion = EvidenceStorageExpansion()
        # Derived work sealed commitments prove for unchanged inputs, reused
        # instead of computed again: see `runtime.reuse`.
        self.sealed_excerpts = SealedExcerpts(self.artifacts, self.record_days[SNAPSHOT_CATEGORY])
        self.sealed_canonicals = SealedCanonicals(
            self.artifacts,
            self.documents,
            binding_hash=self.document_binding_hash,
            source_days=self.record_days[SOURCE_SET_CATEGORY],
            document_days=self.record_days[DOCUMENT_SET_CATEGORY],
        )
        self.sealed_selections = SealedSelections(
            self.artifacts,
            selection_policy_hash=self.analysis_policy_hash,
            program_hash=query_program_hash(),
        )
        self.writer = EvidenceWriter()
        self.sealed_comparisons = SealedComparisons(
            self.artifacts, admit=self.admit_storage, held=self.writer.held
        )
        self.pair_score_facts: dict[str, int] = {}
        """The last session's pair-score store counts (hits, misses, blocks,
        unanchored, missing, sealed and unsealed pairs)."""
        self.pair_score_commitments_refused = 0
        """Commitments the storage admission refused: their blocks stay
        unanchored and unserved."""
        self.retrieval = AlternativeEvidenceRetrievalService(
            workspace_root=workspace_root,
            model_root=model_root,
            retrieval_binding_hash=self.retrieval_binding_hash,
            index_factory=index_factory,
            retriever_factory=retriever_factory,
            index_spec=HybridIndexSpec.for_recipe(retrieval_recipe),
        )
        self.execution = CpuBudgetStore(artifact_root.parent)
        self.foreground = VERIFIED_PACKS.foreground
        """While a reader the lead waits on runs (an Analyst's bundle), this process's
        model work waits at its next call or batch (F2)."""
        """The workspace's CPU budget and the log of what each preparation ran with:
        how many units of a book prepare at once and each session's threads
        (`runtime.execution`), the operator's and never identity."""
        self.publications = AlternativeEvidenceAnalysisPublicationService(
            self.artifacts,
            acquisition_binding_hash=self.acquisition.acquisition_binding_hash,
            canonicalization_binding_hash=self.document_binding_hash,
            retrieval_binding_hash=self.retrieval_binding_hash,
            analysis_policy_hash=self.analysis_policy_hash,
            decision_policy_hash=self.decision_policy.binding_hash,
            publication_binding_hash=self.publication_binding_hash,
            supported_historical_bindings=(
                *SUPPORTED_HISTORICAL_BINDINGS,
                *self._observed_bindings(),
            ),
        )
        self.sealed_readings = SealedReadings(
            self.artifacts,
            expected_bindings=self.publications.expected_bindings,
            days=self.record_days[ANALYSIS_PUBLICATION_CATEGORY],
        )
        self._set_revisions: dict[str, frozenset[tuple[str, str]]] = {}
        """Each sealed document set's filings, by its hash: immutable, read once."""

    def _observed_bindings(self) -> tuple[ObservedEvidenceBindings, ...]:
        """The tuples this workspace's analyses record that no build lists (E2).

        Read once, when the runtime is composed; each is handed to the publication
        service as history, so those analyses and the reviews resting on them read
        back verified, labelled historical, and never as current authority.
        """
        current = (
            self.acquisition.acquisition_binding_hash,
            self.document_binding_hash,
            self.retrieval_binding_hash,
            self.analysis_policy_hash,
            self.decision_policy.binding_hash,
            self.publication_binding_hash,
        )
        recorded = (
            (
                value.acquisition_binding_hash,
                value.canonicalization_binding_hash,
                value.retrieval_binding_hash,
                value.analysis_policy_hash,
                value.decision_policy_hash,
                value.publication_binding_hash,
            )
            for value in self.artifacts.values(
                ANALYSIS_PUBLICATION_CATEGORY, AlternativeEvidenceAnalysisPublication
            )
        )
        return observed_historical_bindings(recorded, current=current)

    def days_of(self, category: str, field: str) -> RecordDays:
        """Return the dated index for a category sealed by another owner.

        Read each record's day from ``field`` (Z2); keep one index per category.
        """
        held = self.record_days.get(category)
        if held is None:
            held = self.record_days[category] = RecordDays(
                self.artifacts, category, sealed_field(self.artifacts, category, field)
            )
        return held

    def admit_storage(self, additional_bytes: int) -> None:
        """Ask the Host's capacity owner to admit a runtime write.

        An unbound library runtime has no workspace budget to enforce.
        """
        if self.storage_admission is not None:
            self.storage_admission(additional_bytes)

    def close(self) -> None:
        """Close the retrieval owner and its held resources."""
        self.retrieval.close()

    def reuse_accounting(self) -> dict[str, int]:
        """Report work replaced by sealed commitments.

        Count reused excerpts, canonical documents, and selections; damaged
        canonical revisions; and generation leases released when a receipt
        already proved the selection.
        """
        return {
            "excerpt_reuses": self.sealed_excerpts.reused,
            "canonical_reuses": self.sealed_canonicals.reused,
            "canonical_revisions_damaged": self.sealed_canonicals.damaged,
            "selection_reuses": self.selection_reuse_count,
            "generation_releases": self.retrieval.generation_release_count,
            "generation_builds_avoided": self.generation_builds_avoided,
        }

    def acquire_recorded(
        self,
        *,
        request: AlternativeEvidenceRequest,
        registry: SecIssuerRegistrySnapshot,
        documents: tuple[RecordedEvidenceDocument, ...],
        published_at: datetime,
    ) -> tuple[AlternativeEvidenceSnapshot, AcquiredEvidenceSourceReferenceSet]:
        """Acquire recorded documents and seal their source references."""
        self.recorded_acquisition_document_count += len(documents)
        snapshot, acquired = self.acquisition.build_recorded_evidence(
            request=request,
            registry=registry,
            documents=documents,
            published_at=published_at,
        )
        self.record_days[SNAPSHOT_CATEGORY].note(snapshot.snapshot_hash, snapshot.published_at)
        source_set = self._publish_source_set(
            request_hash=request.request_hash,
            source_snapshot_hash=snapshot.snapshot_hash,
            acquired=acquired,
            acquired_at=published_at,
        )
        return snapshot, source_set

    def _publish_source_set(
        self,
        *,
        request_hash: str,
        source_snapshot_hash: str,
        acquired: tuple[AcquiredEvidenceDocument, ...],
        acquired_at: datetime,
    ) -> AcquiredEvidenceSourceReferenceSet:
        """Keep each document's bytes once and seal this request's references to them."""
        references = tuple(
            self.documents.publish_source_object(document, admit=self.storage_admission)
            for document in acquired
        )
        source_set = seal_contract(
            AcquiredEvidenceSourceReferenceSet,
            "source_set_hash",
            request_hash=request_hash,
            source_snapshot_hash=source_snapshot_hash,
            documents=references,
            acquired_at=acquired_at,
        )
        self.artifacts.place(
            [(SOURCE_SET_CATEGORY, source_set.source_set_hash, source_set)],
            admit=self.admit_storage,
        )
        self.record_days[SOURCE_SET_CATEGORY].note(source_set.source_set_hash, acquired_at)
        return source_set

    def resolve_source_documents(
        self, source_set: AcquiredEvidenceSourceSet
    ) -> tuple[AcquiredEvidenceDocument, ...]:
        """Resolve either durable source-set form with verified bytes."""
        if isinstance(source_set, AcquiredEvidenceDocumentSet):
            return source_set.documents
        return tuple(
            self.documents.resolve_source_object(reference) for reference in source_set.documents
        )

    def acquire_live(
        self,
        *,
        request: AlternativeEvidenceRequest,
        admission: AlternativeEvidenceAdmission,
        source: SecEdgarSource,
        published_at: datetime,
        prior_companyfacts: tuple[SecCompanyFactsSnapshot, ...] = (),
        registry: SecIssuerRegistrySnapshot | None = None,
        should_cancel: object = None,
        accession_scopes: Mapping[str, frozenset[str]] | None = None,
        clock: Callable[[], datetime] | None = None,
        fetched: Callable[[int], None] | None = None,
    ) -> tuple[
        SecIssuerRegistrySnapshot,
        AlternativeEvidenceSnapshot,
        AcquiredEvidenceSourceReferenceSet,
    ]:
        """Acquire live source evidence under the storage admission.

        Reuse verified local bodies. Commit each fetched body's bytes,
        provenance, and resource identity as it arrives so a later failure
        cannot lose it. ``fetched(1)`` observes each completed commitment.
        """

        def commit(document: AcquiredEvidenceDocument) -> None:
            # The bytes and their commit record are admitted together, before
            # either is written: a budget that cannot take both refuses the
            # body whole, never a body without the record that makes it
            # evidence.
            reference = self.documents.source_reference_of(document)
            record = seal_contract(
                AcquiredEvidenceSourceCommit,
                "commit_hash",
                reference=reference,
                committed_at=published_at,
            )
            # A body restored under an intact commit -- its bytes were lost,
            # its provenance was not -- is kept under that commit; a second
            # record for the same bytes would be provenance twice.
            needs_record = not self.local_sources.committed(reference) and (
                not self.artifacts.holds(SOURCE_COMMIT_CATEGORY, record.commit_hash, record)
            )
            additional = 0
            if not self.documents.holds_source_object(reference.content_sha256):
                additional += len(document.content)
            if needs_record:
                additional += len(self.artifacts.serialized(record))
            if additional and self.storage_admission is not None:
                self.storage_admission(additional)
            self.documents.publish_source_object(document)
            if needs_record:
                self.artifacts.publish(SOURCE_COMMIT_CATEGORY, record.commit_hash, record)
                self.record_days[SOURCE_COMMIT_CATEGORY].note(
                    record.commit_hash, record.committed_at
                )
            self.local_sources.add(reference, committed=True)
            if fetched is not None:
                fetched(1)
            # A body and its record are placed: a safe point of a long stage.
            self.writer.pause()

        def defer(
            entry: SecFilingInventoryEntry, *, observed_bytes: int, admitted_cap_bytes: int
        ) -> None:
            # The observation sealed on its own, by the resource's identity:
            # the next request finds it before any transfer.
            record = seal_contract(
                AcquiredEvidenceSourceDeferral,
                "deferral_hash",
                source_cik=entry.cik,
                accession=entry.accession,
                document_name=entry.primary_document,
                form=entry.form,
                accepted_at=entry.accepted_at,
                inventory_size_bytes=entry.size_bytes,
                admitted_cap_bytes=admitted_cap_bytes,
                observed_bytes=observed_bytes,
                observed_at=published_at,
            )
            self.artifacts.place(
                [(SOURCE_DEFERRAL_CATEGORY, record.deferral_hash, record)],
                admit=self.admit_storage,
            )
            self.record_days[SOURCE_DEFERRAL_CATEGORY].note(
                record.deferral_hash, record.observed_at
            )
            self.local_sources.add_deferral(record)

        def preflight(expected_fetch_bytes: int) -> None:
            # The issuer's transfers at every layer, put to the storage owner
            # once before the first of them; the writes that follow are each
            # admitted again at their own size.
            if self.storage_admission is not None:
                self.storage_admission(
                    int(expected_fetch_bytes * self.storage_expansion.peak_per_raw)
                )

        def known_excerpt(document: AcquiredEvidenceDocument) -> str | None:
            # A sealed citation of these exact bytes under this extraction
            # binding: the body is cited, not parsed again.
            return self.sealed_excerpts.find(document, binding_hash=self.document_binding_hash)

        registry, snapshot, acquired = self.acquisition.build_live_evidence(
            request=request,
            admission=admission,
            source=source,
            published_at=published_at,
            prior_companyfacts=prior_companyfacts,
            registry=registry,
            should_cancel=should_cancel or (lambda: False),  # type: ignore[arg-type]
            accession_scopes=accession_scopes,
            local=self.local_sources,
            commit=commit,
            preflight=preflight,
            defer=defer,
            excerpt_binding_hash=self.document_binding_hash,
            known_excerpt=known_excerpt,
            **({} if clock is None else {"clock": clock}),
        )
        self.record_days[SNAPSHOT_CATEGORY].note(snapshot.snapshot_hash, snapshot.published_at)
        self.sealed_excerpts.add(snapshot)
        source_set = self._publish_source_set(
            request_hash=request.request_hash,
            source_snapshot_hash=snapshot.snapshot_hash,
            acquired=acquired,
            acquired_at=published_at,
        )
        return registry, snapshot, source_set

    def canonicalize(
        self,
        *,
        source_set: AcquiredEvidenceSourceSet,
        published_at: datetime,
        reading_depth: AlternativeEvidenceReadingDepth = (
            AlternativeEvidenceReadingDepth.FULL_FILING
        ),
        began: Callable[[int], None] | None = None,
    ) -> AlternativeEvidenceDocumentSet:
        """Canonicalize a source set into a sealed document set.

        ``began(1)`` hears each document as its canonical text is taken up.
        """
        self.canonicalization_document_count += len(source_set.documents)

        def known_canonical(document: AcquiredEvidenceDocument) -> bytes | None:
            # The canonical text a sealed document set proves for these exact
            # bytes under the current binding, read through the library's
            # verified revision; the source bytes themselves were resolved
            # and hashed for this request just above.
            if began is not None:
                began(1)
            # The document before this one is taken up whole: a safe point.
            self.writer.pause()
            return self.sealed_canonicals.find(document, reading_depth=reading_depth)

        admitted, rejected = canonicalize_source_documents(
            self.resolve_source_documents(source_set),
            reading_depth=reading_depth,
            known_canonical=known_canonical,
        )
        value = self.documents.publish(
            request_hash=source_set.request_hash,
            source_snapshot_hash=source_set.source_snapshot_hash,
            canonicalization_binding_hash=self.document_binding_hash,
            reading_depth=reading_depth,
            documents=admitted,
            rejections=rejected,
            published_at=published_at,
            admit=self.storage_admission,
        )
        self.artifacts.publish(DOCUMENT_SET_CATEGORY, value.document_set_hash, value)
        self.record_days[DOCUMENT_SET_CATEGORY].note(value.document_set_hash, value.published_at)
        self.sealed_canonicals.add(source_set, value)
        return value

    def build_retrieval(
        self,
        *,
        document_set: AlternativeEvidenceDocumentSet,
        built_at: datetime,
        reuse_for: AlternativeEvidenceRequest | None = None,
        chunks: ChunkCount | None = None,
    ) -> AlternativeEvidenceRetrievalGeneration:
        """Build or reuse the sealed generation for a document set.

        Materialize under the workspace's admission, or, with ``reuse_for``, use
        the sealed record a selection of that request will be reused over
        without opening a session (`_reusable_generation`), in which case no
        index is materialized or proved here: what the selection stands on
        is the receipt and the span set, proved as every reuse proves them,
        and the index it never opens is neither verified nor claimed to be.
        `chunks` hears the chunks the index cuts, counted before the model
        runs, and each batch as it is embedded.
        """
        value = (
            None
            if reuse_for is None
            else self._reusable_generation(document_set, reuse_for, built_at=built_at)
        )
        if value is not None:
            self.generation_builds_avoided += 1
        else:
            from alphalattice.kernel.knowledge._embeddings import (
                observing_passages,
                withhold_inference_yield,
                yielding_during_inference,
            )

            if chunks is not None:
                chunks.expect(self._chunk_count(document_set))

            def anchor_for(index_id: str) -> HybridGenerationAnchor | None:
                # A committed generation is materialized under the workspace's
                # lock, its vectors embedded there only when its payload was
                # lost: that build keeps the writer through the model call.
                anchor = self.generation_anchor(index_id)
                if anchor is not None:
                    withhold_inference_yield()
                return anchor

            with (
                yielding_during_inference(self.writer.released),
                observing_passages(None if chunks is None else chunks.advance),
            ):
                value = self.retrieval.build(
                    document_set=document_set,
                    built_at=built_at,
                    anchor_for=anchor_for,
                    admit=self.storage_admission,
                    block_for=self.vector_block_source(document_set),
                )
            # The commitment is re-read from the sealed records at the moment
            # of sealing this one: a record sealed meanwhile that names other
            # vectors for the same generation means two truths under one
            # identity, and this record is not added as a third.
            anchor = self.generation_anchor(value.index_id)
            if anchor is not None and (
                anchor.manifest_logical_hash != value.index_manifest_hash
                or anchor.payload_sha256 != value.vector_payload_sha256
            ):
                self.retrieval.discard_pending(value.generation_hash)
                raise ValueError("alternative_evidence.retrieval_generation_divergent")
        self.artifacts.publish("retrieval-generations", value.generation_hash, value)
        return value

    def _chunk_count(self, document_set: AlternativeEvidenceDocumentSet) -> int:
        """The chunks the index will cut from this document set, by the index's
        own chunking (0.35 s for a unit of 3,607 chunks): the denominator of a
        build's progress, never an input to it.
        """
        from alphalattice.kernel.knowledge.retrieval import _build_chunks

        library = self.documents.library
        snapshot = library.read_snapshot(UUID(document_set.workspace_snapshot_id))
        return len(_build_chunks(snapshot, library, self.retrieval.index_spec.lexical_spec))

    @staticmethod
    def whole_delivery(
        request: AlternativeEvidenceRequest, document_set: AlternativeEvidenceDocumentSet
    ) -> bool:
        """Check whether the unit qualifies for whole-filing delivery (W4).

        Its policy names a bound and its filings together fit within it.
        """
        bound = request.source_policy.whole_filing_bytes
        held = sum(value.byte_count for value in document_set.documents)
        return bool(document_set.documents) and 0 < held <= bound

    def _whole_texts(self, document_set: AlternativeEvidenceDocumentSet) -> tuple[str, ...]:
        """Each filing's canonical text, from the revision the document set names."""
        texts = []
        for reference in document_set.documents:
            _revision, content = self.documents.library.read_revision(
                reference.workspace_document_id, reference.workspace_revision
            )
            if len(content) != reference.byte_count:
                raise ValueError("alternative_evidence.workspace_document_tampered")
            texts.append(content.decode("utf-8"))
        return tuple(texts)

    def seal_whole_generation(
        self, *, document_set: AlternativeEvidenceDocumentSet, built_at: datetime
    ) -> WholeFilingsGeneration:
        """Seal a whole-filing generation from verified canonical text."""
        texts = self._whole_texts(document_set)
        corpus = str(
            canonical_hash(
                [
                    [reference.workspace_document_id, reference.workspace_revision, text]
                    for reference, text in zip(document_set.documents, texts, strict=True)
                ]
            )
        )
        value = seal_contract(
            WholeFilingsGeneration,
            "generation_hash",
            document_set_hash=document_set.document_set_hash,
            workspace_snapshot_id=document_set.workspace_snapshot_id,
            workspace_snapshot_hash=document_set.workspace_snapshot_hash,
            retrieval_binding_hash=self.retrieval_binding_hash,
            reading_depth=document_set.reading_depth,
            index_id=corpus,
            index_spec_hash=str(canonical_hash({"format": WHOLE_FILINGS_FORMAT})),
            document_count=len(document_set.documents),
            chunk_count=sum(len(_whole_pieces(text)) for text in texts),
            built_at=built_at,
            corpus_hash=corpus,
        )
        self.artifacts.publish("retrieval-generations", value.generation_hash, value)
        return value

    def select_whole(
        self,
        *,
        request: AlternativeEvidenceRequest,
        document_set: AlternativeEvidenceDocumentSet,
        generation: WholeFilingsGeneration,
    ) -> tuple[
        AlternativeEvidenceRetrievalAccessReceipt, tuple[AlternativeEvidenceResolvedSpan, ...]
    ]:
        """Deliver every filing in a short unit whole (W4).

        Split each canonical text at line bounds in document-set order; the pieces form
        the text itself; no session, no question, no rerank.
        """
        require_current_selection(request)
        texts = self._whole_texts(document_set)
        spans: list[AlternativeEvidenceResolvedSpan] = []
        for reference, text in zip(document_set.documents, texts, strict=True):
            for start, end in _whole_pieces(text):
                before = text[:start]
                spans.append(
                    AlternativeEvidenceResolvedSpan(
                        span_handle=f"SPAN-F01-R{len(spans) + 1:03d}",
                        document_handle=reference.semantic_handle,
                        entity_id=reference.entity_id,
                        source_name=reference.source_name,
                        source_right=reference.source_right,
                        document_type=reference.document_type,
                        revision_label=reference.revision_label,
                        title=reference.title,
                        # Lines as the packet reader verifies them: the newlines
                        # before each bound, plus one.
                        start_line=text.count("\n", 0, start) + 1,
                        end_line=text.count("\n", 0, end) + 1,
                        character_start=start,
                        character_end=end,
                        utf8_byte_start=len(before.encode("utf-8")),
                        utf8_byte_end=len(text[:end].encode("utf-8")),
                        excerpt=text[start:end],
                        published_at=reference.published_at,
                        accepted_at=reference.accepted_at,
                        available_at=reference.available_at,
                        immutable_source=reference.immutable_source,
                    )
                )
        span_set = self._seal_span_set(request, generation, tuple(spans))
        receipt = seal_contract(
            AlternativeEvidenceRetrievalAccessReceipt,
            "receipt_hash",
            request_hash=request.request_hash,
            document_set_hash=document_set.document_set_hash,
            retrieval_generation_hash=generation.generation_hash,
            query_program_hash=query_program_hash(),
            queries=(),
            read_span_handles=tuple(value.span_handle for value in spans),
            search_call_count=0,
            span_read_call_count=0,
            selection_policy_hash=self.analysis_policy_hash,
            span_set_hash=span_set.span_set_hash,
            whole_filings=WholeFilingsRecord(
                bound_bytes=request.source_policy.whole_filing_bytes,
                document_count=len(document_set.documents),
                delivered_bytes=sum(value.byte_count for value in document_set.documents),
            ),
        )
        self._publish_selection(span_set, receipt)
        return receipt, tuple(spans)

    def vector_block_source(
        self, document_set: AlternativeEvidenceDocumentSet
    ) -> VectorBlockSource:
        """One revision's committed vectors from a generation this runtime sealed.

        Asked with the encoder context and the revision's exact encoder
        inputs, the source looks through the sealed committed records: a
        generation composed from blocks is proved as a whole (every block
        read, the concatenation hashed back to the committed digest) and its
        blocks are keyed by what the committed manifest derives for their
        position -- the manifest's encoder context and ordered encoder inputs
        -- never by what the sidecar says; a generation written as one
        payload serves a slice cut by that same manifest where those inputs
        appear contiguously under the same context. Nothing is served from a
        database, a sidecar or a file that does not verify against a sealed
        record, and a generation whose manifest is not on disk proves no
        mapping. Every refusal is recorded in `block_source_refusals` with
        its code; the revision is then embedded. Proofs are kept for the life
        of one build.

        Only a generation whose document set holds one of this set's filings
        is looked through (X6): a revision's vectors are committed by a
        generation that read its filing, and proving a generation reads every
        block it holds. A first reading of new filings proves none, where it
        read every earlier generation of the workspace whole.
        """
        root = self.retrieval.workspace.root
        wanted = self._revisions(document_set.document_set_hash, document_set)
        records = [
            value
            for value in self._committed_records()
            if isinstance(value, AlternativeEvidenceRetrievalGeneration)
            and wanted & self._revisions(value.document_set_hash)
        ]
        records.sort(key=lambda value: value.built_at, reverse=True)
        proved: dict[tuple[str, str], dict[str, bytes] | None] = {}
        self.block_source_refusals = []

        def block_for(context_hash: str, inputs: tuple[str, ...]) -> bytes | None:
            key = vector_asset_key(context_hash, inputs)
            for record in records:
                pair = (record.index_manifest_hash, record.vector_payload_sha256)
                if pair not in proved:
                    try:
                        proved[pair] = verified_block_map(
                            root,
                            database_relative=committed_database_relative(
                                record.corpus_hash, record.index_spec_hash
                            ),
                            manifest_logical_hash=record.index_manifest_hash,
                            payload_sha256=record.vector_payload_sha256,
                        )
                    except KnowledgeRetrievalError as refused:
                        proved[pair] = {}
                        self.block_source_refusals.append(
                            {
                                "index_id": record.index_id,
                                "generation_hash": record.generation_hash,
                                "code": refused.failure.code,
                            }
                        )
                block_map = proved[pair]
                if block_map is not None:
                    content = block_map.get(key)
                    if content is not None:
                        return content
                    continue
                try:
                    content = committed_payload_slice(
                        root,
                        database_relative=committed_database_relative(
                            record.corpus_hash, record.index_spec_hash
                        ),
                        manifest_logical_hash=record.index_manifest_hash,
                        payload_sha256=record.vector_payload_sha256,
                        context_hash=context_hash,
                        embedding_input_hashes=inputs,
                    )
                except KnowledgeRetrievalError:
                    content = None
                if content is not None:
                    return cast(bytes, content)
            return None

        return block_for

    def _revisions(
        self, document_set_hash: str, document_set: AlternativeEvidenceDocumentSet | None = None
    ) -> frozenset[tuple[str, str]]:
        """The filings a sealed document set holds, by issuer and revision
        label (a workspace document id is the set's own); none for a set
        that does not read back, which proves no vectors.
        """
        held = self._set_revisions.get(document_set_hash)
        if held is None:
            try:
                value = document_set or self.artifacts.load(
                    "document-sets", document_set_hash, AlternativeEvidenceDocumentSet
                )
            except (ValueError, FileNotFoundError):
                return frozenset()
            held = frozenset(
                (reference.entity_id, reference.revision_label) for reference in value.documents
            )
            self._set_revisions[document_set_hash] = held
        return held

    def _committed_records(self) -> tuple[RetrievalGenerationRecord, ...]:
        directory = self.artifacts.root / "retrieval-generations"
        if not directory.is_dir():
            return ()
        return tuple(
            self.artifacts.load_retrieval_generation(path.stem)
            for path in sorted(directory.glob("*.json"))
        )

    def _reusable_generation(
        self,
        document_set: AlternativeEvidenceDocumentSet,
        request: AlternativeEvidenceRequest,
        *,
        built_at: datetime,
    ) -> AlternativeEvidenceRetrievalGeneration | None:
        """The generation record a selection of this request would be reused
        over without a session -- sealed here exactly as the build would seal
        it, from the one commitment the sealed records hold for the corpus's
        index -- or None when the normal build must run. Proved before it is
        returned: committed records at the current retrieval binding and
        index spec name this corpus's index and agree on one commitment
        (`generation_anchor`, which refuses two truths as the build would);
        the index is held on disk under its identity and not evicted
        (presence only: an absent or evicted index is restored by the build
        as before); a sealed receipt of this content under the current
        policy names a span set that is the one it delivered
        (`SealedSelections.find`, which refuses a named set that is not and
        never treats it as a miss); and the document set's revisions and the
        record's lineage verify (`verify_lineage`). The record claims what a
        built one claims -- which commitment this document set resolves to
        -- and not that the index on disk was proved: nothing opened it. A
        request that asks for a retired policy is refused at the selection,
        as ever; nothing here reinterprets a request or a policy.
        """
        index_id, held = self.retrieval.committed_index(document_set)
        committed = [
            value
            for value in self._generation_records(index_id)
            if isinstance(value, AlternativeEvidenceRetrievalGeneration)
            and is_current(
                RETRIEVAL_BINDING_ROLE, value.retrieval_binding_hash, self.retrieval_binding_hash
            )
            and value.index_spec_hash == self.retrieval.index_spec.logical_hash
        ]
        anchor = self.generation_anchor(index_id)
        if not committed or anchor is None or not held:
            return None
        template = committed[0]
        generation = seal_contract(
            AlternativeEvidenceRetrievalGeneration,
            "generation_hash",
            document_set_hash=document_set.document_set_hash,
            workspace_snapshot_id=document_set.workspace_snapshot_id,
            workspace_snapshot_hash=document_set.workspace_snapshot_hash,
            retrieval_binding_hash=self.retrieval_binding_hash,
            reading_depth=document_set.reading_depth,
            corpus_hash=template.corpus_hash,
            index_id=template.index_id,
            index_spec_hash=self.retrieval.index_spec.logical_hash,
            index_manifest_hash=anchor.manifest_logical_hash,
            vector_payload_sha256=anchor.payload_sha256,
            document_count=template.document_count,
            chunk_count=template.chunk_count,
            built_at=built_at,
        )
        if (
            self.sealed_selections.find(
                document_set=document_set,
                generation=generation,
                matter_selection_id=request.matter_selection_id,
            )
            is None
        ):
            return None
        self.retrieval.verify_lineage(document_set, generation)
        return generation

    def generation_anchor(self, index_id: str) -> HybridGenerationAnchor | None:
        """Return the durable commitment sealed for a generation.

        Every committed record naming the index must commit to the same
        manifest and payload; records that disagree are two truths under one
        identity and refuse until resolved. None when no committed record
        names the index -- a corpus never sealed, or sealed only in the legacy
        format -- so the caller embeds under its own admission rather than
        trusting anything on disk.
        """
        anchors = {
            (value.index_manifest_hash, value.vector_payload_sha256)
            for value in self._generation_records(index_id)
            if isinstance(value, AlternativeEvidenceRetrievalGeneration)
        }
        if not anchors:
            return None
        if len(anchors) > 1:
            raise ValueError("alternative_evidence.retrieval_generation_records_divergent")
        ((manifest_hash, payload_sha256),) = anchors
        return HybridGenerationAnchor(
            manifest_logical_hash=manifest_hash, payload_sha256=payload_sha256
        )

    def _generation_records(self, index_id: str) -> tuple[RetrievalGenerationRecord, ...]:
        """Every sealed generation record naming one index, verified by identity."""
        return tuple(value for value in self._committed_records() if value.index_id == index_id)

    def rebuild_retrieval(self, index_id: str) -> dict[str, object]:
        """Restore or repair one index by the identity its sealed records carry.

        The committed records naming the index must agree on one commitment,
        which anchors the rebuild. A legacy record names a generation this
        runtime never rebuilds: it refuses, and a new preparation is the way
        forward.
        """
        records = self._generation_records(index_id)
        if not records:
            raise ValueError("alternative_evidence.retrieval_generation_unknown")
        committed = [
            value for value in records if isinstance(value, AlternativeEvidenceRetrievalGeneration)
        ]
        if not committed:
            raise ValueError("alternative_evidence.retrieval_generation_legacy_not_rebuildable")
        self.generation_anchor(index_id)
        generation = max(committed, key=lambda value: value.built_at)
        document_set = self.artifacts.load(
            "document-sets", generation.document_set_hash, AlternativeEvidenceDocumentSet
        )
        return self.retrieval.rebuild_generation(
            document_set=document_set,
            generation=generation,
            admit=self.storage_admission,
            block_for=self.vector_block_source(document_set),
        )

    def original_reader(
        self, document_set: AlternativeEvidenceDocumentSet
    ) -> Callable[[str], tuple[bytes, str, str] | None]:
        """Build a reader for verified original document bytes.

        Map workspace document IDs to original bytes, content SHA-256, and
        media type through the sealed source set and verified object store.
        Table views render from those originals. Return ``None`` for a
        document without an original reference; refuse a referenced original
        that is missing or fails verification.
        """
        references: dict[str, AcquiredEvidenceDocumentReference] = {}
        source_set = self._source_set_of(document_set)
        if isinstance(source_set, AcquiredEvidenceSourceReferenceSet):
            by_handle = {value.semantic_handle: value for value in source_set.documents}
            for document in document_set.documents:
                found = by_handle.get(document.semantic_handle)
                if (
                    found is not None
                    and found.entity_id == document.entity_id
                    and found.revision == document.revision_label
                    and found.document_type == document.document_type
                ):
                    references[document.workspace_document_id] = found

        def original_for(document_id: str) -> tuple[bytes, str, str] | None:
            reference = references.get(document_id)
            if reference is None:
                return None
            resolved = self.documents.resolve_source_object(reference)
            return resolved.content, reference.content_sha256, reference.media_type

        return original_for

    def _source_set_of(
        self, document_set: AlternativeEvidenceDocumentSet
    ) -> AcquiredEvidenceSourceSet | None:
        """The sealed source set a document set was canonicalized from: the
        one of its request whose snapshot it names.
        """
        root = self.artifacts.root / "source-document-sets"
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            try:
                value = self.artifacts.load_source_set(path.stem)
            except (ValueError, OSError):
                continue
            if (
                value.request_hash == document_set.request_hash
                and value.source_snapshot_hash == document_set.source_snapshot_hash
            ):
                return value
        return None

    def open_session(
        self,
        *,
        document_set: AlternativeEvidenceDocumentSet,
        generation: RetrievalGenerationRecord,
        evidence_as_of: datetime,
    ) -> AlternativeEvidenceRetrievalSession:
        """Open a generation session with retained originals and pair scores.

        The session renders table views from retained originals and serves
        pair scores admitted by the workspace's sealed commitments. Selections
        open one session for the generation.
        """
        return self.retrieval.open_session(
            document_set=document_set,
            generation=generation,
            evidence_as_of=evidence_as_of,
            original_for=self.original_reader(document_set),
            pair_scores=self.pair_score_admission(
                since=datetime.now(UTC)
                - timedelta(
                    days=self.artifacts.load(
                        "requests", document_set.request_hash, AlternativeEvidenceRequest
                    ).source_policy.sec_recent_8k_days
                    + 1
                )
            ),
        )

    def pair_score_admission(self, *, since: datetime) -> PairScoreAdmission:
        """Return the pair-score blocks and admission a reader may use.

        Read blocks named by workspace commitments under the current reranker
        context (`PairScoreCommitmentRecord`, each named by the receipt of
        the session that scored them), and the workspace's storage
        admission for new blocks. A block under any other name is never
        served; a commitment of another context admits nothing here.

        Only a commitment sealed since `since` admits (X2): a reader loads
        every block it admits, and a filing's pairs are committed as it is
        read, in the window -- so a commitment older than the window's
        length holds filings no reading now can take, and the blocks a
        reader loads do not grow with the days a workspace holds.
        """
        context = reranker_context_hash(self.retrieval.index_spec)
        admitted = {
            block.name
            for record in self.artifacts.values(
                PAIR_SCORE_COMMITMENT_CATEGORY, PairScoreCommitmentRecord
            )
            if record.reranker_context_hash == context and record.sealed_at >= since
            for block in record.blocks
        }
        return PairScoreAdmission(admitted=frozenset(admitted), admit=self.admit_storage)

    def commit_pair_scores(
        self,
        session: AlternativeEvidenceRetrievalSession,
        generation: RetrievalGenerationRecord,
    ) -> str | None:
        """Seal and commit pair-score blocks from a session.

        Return the commitment hash for the receipt, or ``None`` when the session
        sealed nothing (nothing scored, or the storage admission refused the
        blocks or the commitment). Called before the session closes.
        """
        sealed = session.seal_pair_scores()
        self.pair_score_facts = session.pair_score_facts()
        if not sealed:
            return None
        record = seal_contract(
            PairScoreCommitmentRecord,
            "commitment_hash",
            reranker_context_hash=reranker_context_hash(self.retrieval.index_spec),
            index_spec_logical_hash=self.retrieval.index_spec.logical_hash,
            retrieval_generation_hash=generation.generation_hash,
            blocks=tuple(
                PairScoreBlockRecord(
                    name=block.name, pair_count=block.pair_count, byte_length=block.byte_length
                )
                for block in sealed
            ),
            sealed_at=datetime.now(UTC),
        )
        try:
            self.artifacts.place(
                [(PAIR_SCORE_COMMITMENT_CATEGORY, record.commitment_hash, record)],
                admit=self.admit_storage,
            )
        except AlternativeEvidencePublicationError:
            # A record under this commitment's name that is not this
            # commitment: refused by name, never named by the receipt.
            raise
        except Exception:
            # The blocks were placed under admission; without their
            # commitment they stay unanchored and are never served.
            self.pair_score_commitments_refused += 1
            return None
        return record.commitment_hash

    def select_evidence(
        self,
        *,
        request: AlternativeEvidenceRequest,
        document_set: AlternativeEvidenceDocumentSet,
        generation: RetrievalGenerationRecord,
    ) -> tuple[
        AlternativeEvidenceRetrievalAccessReceipt, tuple[AlternativeEvidenceResolvedSpan, ...]
    ]:
        """Run and seal the integrated selection of one unit.

        Select one discovery, typed families under their budgets, routed
        topics in topic lanes, table views, and residual questions. Seal one
        receipt with its routing; one session serves one unit.

        A selection is a function of the generation's content, the document
        references, the program and the selection policy. When a sealed
        receipt of exactly that already exists -- another request over the
        same filings -- no session opens: the receipt is sealed again for
        this request, naming the one it stands on, over the spans that
        receipt delivered (`SealedSelections`). A request under a retired
        selection -- the production plan (the omitted field), the candidate
        needs allocation, a retired residual policy -- is refused by name
        before anything is found or opened (`matter_selection_retired`).
        """
        require_current_selection(request)
        found = self.sealed_selections.find(
            document_set=document_set,
            generation=generation,
            matter_selection_id=request.matter_selection_id,
        )
        if found is not None:
            prior, spans = found
            # What a session would have verified first, and the build's lease
            # closed: no session takes it, and a held lease is not evictable.
            self.retrieval.release_without_session(document_set=document_set, generation=generation)
            if (prior.request_hash, prior.document_set_hash, prior.retrieval_generation_hash) == (
                request.request_hash,
                document_set.document_set_hash,
                generation.generation_hash,
            ):
                # This request's own sealed receipt: a replay, the same identity.
                receipt = prior
                span_set = None
            else:
                # The span set is sealed first so the receipt can name it: the
                # binding the next reuse stands on. The prior's own records
                # carry over; this request's identity, policy, set and
                # provenance are sealed afresh, and a reuse scored nothing,
                # so it names no commitment.
                span_set = self._seal_span_set(request, generation, spans)
                receipt = seal_contract(
                    AlternativeEvidenceRetrievalAccessReceipt,
                    "receipt_hash",
                    **_carried(
                        prior,
                        without={
                            "request_hash",
                            "document_set_hash",
                            "retrieval_generation_hash",
                            "selection_policy_hash",
                            "reused_from_receipt_hash",
                            "span_set_hash",
                            "pair_score_commitment_hash",
                        },
                    ),
                    request_hash=request.request_hash,
                    document_set_hash=document_set.document_set_hash,
                    retrieval_generation_hash=generation.generation_hash,
                    selection_policy_hash=self.analysis_policy_hash,
                    reused_from_receipt_hash=prior.reused_from_receipt_hash or prior.receipt_hash,
                    span_set_hash=span_set.span_set_hash,
                )
            self.selection_reuse_count += 1
        else:
            # The integrated selection: one discovery, the topics routed, the
            # units in topic lanes, the tables as views, the bank over the
            # residual scope -- sealed as one receipt with its routing.
            session = self.open_session(
                document_set=document_set,
                generation=generation,
                evidence_as_of=request.evidence_as_of,
            )
            try:
                # The routed selection reads its own session's generation and
                # writes nothing shared but the comparisons it seals, which take
                # the writer back to commit: other units' stages run meanwhile.
                with self.writer.released():
                    routed = select_routed_evidence(
                        session=session,
                        entity_ids=request.ordered_entity_ids,
                        comparisons=self.sealed_comparisons,
                    )
                search_count, read_count = session.search_count, session.span_read_count
                commitment_hash = self.commit_pair_scores(session, generation)
            finally:
                session.close()
            spans = routed.spans
            span_set = self._seal_span_set(request, generation, spans)
            receipt = seal_contract(
                AlternativeEvidenceRetrievalAccessReceipt,
                "receipt_hash",
                request_hash=request.request_hash,
                document_set_hash=document_set.document_set_hash,
                retrieval_generation_hash=generation.generation_hash,
                query_program_hash=query_program_hash(),
                queries=routed.queries,
                read_span_handles=routed.read_span_handles,
                search_call_count=search_count,
                span_read_call_count=read_count,
                span_groups=routed.span_groups,
                typed_disclosures=routed.typed,
                litigation_matters=routed.matters,
                routing=routed.routing,
                selection_policy_hash=self.analysis_policy_hash,
                span_set_hash=span_set.span_set_hash,
                pair_score_commitment_hash=commitment_hash,
            )
        self._publish_selection(span_set, receipt)
        self.sealed_selections.add(receipt, document_set=document_set, generation=generation)
        return receipt, spans

    def _seal_span_set(
        self,
        request: AlternativeEvidenceRequest,
        generation: RetrievalGenerationRecord,
        spans: tuple[AlternativeEvidenceResolvedSpan, ...],
    ) -> AlternativeEvidenceResolvedSpanSet:
        """Seal the delivered spans under the request and the generation,
        before the receipt that names the set; `_publish_selection` places
        both under one admission.
        """
        return seal_contract(
            AlternativeEvidenceResolvedSpanSet,
            "span_set_hash",
            request_hash=request.request_hash,
            retrieval_generation_hash=generation.generation_hash,
            spans=spans,
        )

    def _publish_selection(
        self,
        span_set: AlternativeEvidenceResolvedSpanSet | None,
        receipt: AlternativeEvidenceRetrievalAccessReceipt,
    ) -> None:
        """Place a session's span set and receipt under one storage admission
        of the bytes the workspace does not hold yet -- the same rule as every
        other durable write (review finding R4) -- the span set first, so the
        receipt never names a set that is not there. What is already there
        is verified, never taken for the record because its path exists
        (`AlternativeEvidenceArtifactStore.place`): a damaged or substituted
        span set or receipt refuses by name before any admission or write,
        held valid bytes cost no admission, and a valid retry -- after an
        interruption between the two placements, or with the workspace at
        its cap -- writes only what is missing. Without a set to place (the
        receipt's own sealed selection replayed) the set the receipt names
        must be the one it delivered, proved the way a reuse proves it.
        """
        members: list[tuple[str, str, BaseModel]] = []
        if span_set is None:
            if self.sealed_selections.spans_of(receipt) is None:
                raise FileNotFoundError("alternative_evidence.artifact_missing")
        else:
            if receipt.span_set_hash != span_set.span_set_hash:
                raise AlternativeEvidencePublicationError(
                    "alternative_evidence.selection_dependency_mismatch"
                )
            members.append((SPAN_SET_CATEGORY, span_set.span_set_hash, span_set))
        members.append((RECEIPT_CATEGORY, receipt.receipt_hash, receipt))
        self.artifacts.place(members, admit=self.admit_storage)

    def continue_evidence(
        self,
        *,
        request: AlternativeEvidenceRequest,
        document_set: AlternativeEvidenceDocumentSet,
        generation: RetrievalGenerationRecord,
        prior_receipt: AlternativeEvidenceRetrievalAccessReceipt,
        prior_spans: tuple[AlternativeEvidenceResolvedSpan, ...],
        session_limit: int,
        window_limit: int,
    ) -> tuple[
        AlternativeEvidenceRetrievalAccessReceipt, tuple[AlternativeEvidenceResolvedSpan, ...]
    ]:
        """Continue matter reading under the same request and generation.

        Open another session over the document set and continue the litigation
        plan proved by the prior sealed receipt. Seal a successor receipt
        that carries the program's and the typed families' entries unchanged
        and the continued matter record, over the prior spans plus the
        windows this session read. The program and the typed families are
        not rerun: continuation is source reading, never a search.

        The prior receipt must be of this very request, document set and
        generation and must carry a matter record; the prior spans must be
        exactly the handles that receipt delivered. Anything else refuses by
        name before a session opens, a chain of a retired selection among
        it. The integrated chain continues through `select_matter_evidence`
        and `continue_routed_reads`, which prove the prior is of its very
        plan before reading.
        """
        require_current_selection(request)
        if (
            prior_receipt.request_hash != request.request_hash
            or prior_receipt.document_set_hash != document_set.document_set_hash
            or prior_receipt.retrieval_generation_hash != generation.generation_hash
        ):
            raise ValueError("alternative_evidence.litigation_continuation_lineage_mismatch")
        if (
            prior_receipt.selection_policy_hash is not None
            and prior_receipt.selection_policy_hash != self.analysis_policy_hash
        ):
            # The chain's plan was sealed under another analysis policy: its
            # pending candidates, pages and windows are that policy's; nothing
            # of them is read under this one. A new preparation seals a plan
            # under the current identity.
            raise ValueError("alternative_evidence.litigation_continuation_policy_changed")
        prior = prior_receipt.litigation_matters
        if prior is None:
            raise ValueError("alternative_evidence.litigation_continuation_without_prior")
        if prior.allocation_rules_id != TOPIC_LANES_ALLOCATION_ID or prior_receipt.routing is None:
            # A record of a retired selection: read as sealed, never continued.
            raise ValueError("alternative_evidence.litigation_continuation_prior_incompatible")
        if (
            tuple(value.span_handle for value in prior_spans)
            != prior_receipt.delivered_span_handles
        ):
            raise ValueError("alternative_evidence.litigation_continuation_spans_mismatch")
        session = self.open_session(
            document_set=document_set,
            generation=generation,
            evidence_as_of=request.evidence_as_of,
        )
        routing = prior_receipt.routing
        continued_candidate_spans: tuple[AlternativeEvidenceResolvedSpan, ...] = ()
        continued_table_spans: tuple[AlternativeEvidenceResolvedSpan, ...] = ()
        try:
            # The integrated chain: the same discovery and routing are
            # recomputed (deterministic over the same generation); the
            # units continue in their topic lanes under the allowance
            # less this session's table pages; the sealed pending plan's
            # table pages (partial tables resumed at their next row, then
            # tables not yet dealt, topic-fair) and residual candidates
            # (per issuer, by cell, round by round) are read without a
            # search; the delivered tables follow the sealed page
            # identities, never a prefix of the routing's table needs.
            assert routing is not None
            inventory = shared_inventory(session)
            topic_routing = route_session(
                session,
                inventory,
                prior_receipt.typed_disclosures,
                entity_ids=request.ordered_entity_ids,
                comparisons=self.sealed_comparisons,
                bindings=routing.comparison_bindings(),
            )
            refusals = list(routing.table_view_refusals)
            refused_by_document: dict[str, dict[int, str]] = {}
            renderable = table_view_renderable(
                session, topic_routing, inventory, refusals, refused_by_document
            )
            pages = plan_table_pages(
                prior=routing,
                routing=topic_routing,
                inspected=inventory.inspected,
                renderable=renderable,
            )
            candidates_pending = any(c.state == "PENDING" for c in routing.candidates())
            matters, matter_spans, _trace = select_matter_evidence(
                session=session,
                prior=prior,
                continued_from=prior_receipt.receipt_hash,
                session_limit=session_limit,
                window_limit=window_limit,
                inventory=inventory,
                routing=topic_routing,
                allowance_reserve=len(pages),
                pending_elsewhere=bool(pages) or candidates_pending,
            )
            unrenderable = tuple(
                (key, refusal.table_ordinal)
                for refusal in refusals
                for key, document in inventory.inspected.items()
                if document.document_handle == refusal.document_handle
            )
            continued = continue_routed_reads(
                session=session,
                prior=routing,
                inventory=inventory,
                routing=topic_routing,
                session_index=matters.session_index,
                pages=pages,
                refusals=tuple(refusals),
                unrenderable=unrenderable,
                # What the chain has delivered through other channels:
                # the prior spans and this session's windows cover the
                # pending candidates they hold.
                delivered=delivered_ranges((*prior_spans, *matter_spans)),
            )
            routing = routing_record(
                topic_routing,
                matters=matters,
                inspected=inventory.inspected,
                needs_by_document=inventory.needs_by_document,
                table_view_span_handles=continued.table_view_span_handles,
                table_views=continued.table_views,
                refusals=continued.refusals,
                unrenderable=continued.unrenderable,
                delivered_tables=continued.delivered_tables,
                scope_windows={
                    (cell.entity_id, cell.topic): cell.residual_windows for cell in routing.cells
                },
                residual_by_cell={
                    (cell.entity_id, cell.topic): cell.residual_hits for cell in routing.cells
                },
                pair_budget=routing.residual_rerank_pair_budget,
                reranked_pairs=routing.residual_reranked_pairs,
                questions=(
                    routing.questions_run,
                    routing.questions_skipped_no_scope,
                    routing.questions_skipped_budget,
                ),
                unrouted_windows=routing.unrouted_windows,
                candidates=continued.pending_candidates,
                candidates_beyond=routing.candidates_beyond_plan,
                candidate_span_handles=continued.candidate_span_handles,
                residual_selection_rules_id=routing.residual_selection_rules_id,
            )
            continued_candidate_spans = continued.candidate_spans
            continued_table_spans = continued.table_spans
        finally:
            session.close()
        # The packet's order is the receipt's: the prior's residual, typed and
        # matter spans, this session's windows, the chain's candidate reads
        # (the prior sessions' then this session's), then the table views
        # (the prior sessions' pages then this session's) -- the order
        # `delivered_span_handles` derives.
        prior_matters = prior_receipt.litigation_matters
        tables = set(() if routing is None else routing.table_view_span_handles)
        candidates = set(() if routing is None else routing.candidate_span_handles)
        spans = (
            *(
                span
                for span in prior_spans
                if span.span_handle not in tables and span.span_handle not in candidates
            ),
            *matter_spans,
            *(span for span in prior_spans if span.span_handle in candidates),
            *continued_candidate_spans,
            *(span for span in prior_spans if span.span_handle in tables),
            *continued_table_spans,
        )
        assert prior_matters is not None
        span_set = self._seal_span_set(request, generation, spans)
        receipt = seal_contract(
            AlternativeEvidenceRetrievalAccessReceipt,
            "receipt_hash",
            **_carried(prior_receipt, without={"litigation_matters", "span_set_hash", "routing"}),
            litigation_matters=matters,
            routing=routing,
            span_set_hash=span_set.span_set_hash,
        )
        self._publish_selection(span_set, receipt)
        return receipt, spans

    def analyze(
        self,
        *,
        actor: AlternativeEvidenceAnalysisActor,
        request: AlternativeEvidenceRequest,
        obligation: AlternativeEvidenceResearchObligation,
        snapshot: AlternativeEvidenceSnapshot,
        document_set: AlternativeEvidenceDocumentSet,
        generation: RetrievalGenerationRecord,
        access_receipt: AlternativeEvidenceRetrievalAccessReceipt,
        resolved_spans: tuple[AlternativeEvidenceResolvedSpan, ...],
        completed_at: datetime,
    ) -> AlternativeEvidenceAnalysisResult:
        """Screen an actor submission and seal its analysis result."""
        packet = AlternativeEvidencePacket(
            request=request,
            obligation=obligation,
            snapshot=snapshot,
            document_set=document_set,
            receipt=access_receipt,
            spans=resolved_spans,
        )
        actor_submission = actor(packet=packet)
        decision = seal_alternative_evidence_analyst_brief(
            request=request,
            obligation=obligation,
            snapshot=snapshot,
            document_set=document_set,
            generation=generation,
            access_receipt=access_receipt,
            resolved_spans=resolved_spans,
            analysis_policy=self.analysis_policy,
            decision_policy=self.decision_policy,
            playpen_root=self.playpen_root,
            answer=actor_submission.answer,
            dropped=actor_submission.dropped,
            completed_at=completed_at,
            actor_kind=actor_submission.actor_kind,
            actor_id=actor_submission.actor_id,
            agent_execution=actor_submission.agent_execution,
            model_call_count=actor_submission.model_call_count,
            protocol_repair_count=actor_submission.protocol_repair_count,
        )
        brief = decision.brief
        self.artifacts.publish("analyst-brief-receipts", decision.receipt_hash, decision)
        record_provider_stage_usage(
            self.artifacts,
            stage="ALTERNATIVE_EVIDENCE_ANALYST",
            subject_hash=decision.receipt_hash,
            usage=actor_submission.provider_usage,
            model_call_count=actor_submission.model_call_count,
        )
        self.artifacts.publish("analyst-briefs", brief.brief_hash, brief)
        return AlternativeEvidenceAnalysisResult(
            access_receipt=access_receipt,
            analyst_receipt=decision,
            brief=brief,
            resolved_spans=resolved_spans,
        )


__all__ = [
    "AlternativeEvidenceActorSubmission",
    "AlternativeEvidenceAnalysisActor",
    "AlternativeEvidenceAnalysisResult",
    "AlternativeEvidenceDocumentIntelligenceRuntime",
    "EvidenceWriter",
    "SubmittedAlternativeEvidenceAnalysisActor",
]
