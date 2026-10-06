"""Derived work a sealed commitment already proves, reused by the runtime.

Three derivations of an unchanged refresh are pure functions of sealed
inputs. A citation's excerpt is the head of the canonical extraction of the
body's bytes under one extraction binding; a canonical document is the
extraction of the same bytes under one canonicalization binding and
reading depth; the program's selection is a function of the generation's
content, the document references, the query program and the selection
policy (`AlternativeEvidenceRetrievalAccessReceipt`: "two runs over the
same generation read the same spans"). Measured on the 64-issuer campaign
copy, an unchanged refresh spent about 83 s extracting excerpts for reused
bodies, 74 s extracting the same bodies again into canonical documents and
96 s (warmed; about 480 s after a restart, the cross-encoder scoring every
pair again) selecting the same spans from the same generations.

Each index here reads only what the artifact store and the knowledge
library already hold -- snapshots, source-document and document sets,
receipts and span sets, the library's own revisions -- through the store's
own readers, which verify every artifact against the identity its path
names, and answers only when every part of the key matches the commitment;
a reused canonical text is verified by the revision that names it, a reused
selection is the span set the receipt names by hash (`span_set_hash`),
never a set that merely carries the same handles. Nothing outlives its
commitments and nothing is skipped that verifies a source: the bytes a
reuse is keyed on are the ones the verified lookup resolved and hashed for
this request. A named commitment that is on disk and does not verify is a
refusal (`selection_commitment_corrupt`), not a recomputation; a receipt
that names no span set (sealed before the binding was recorded) proves
nothing and is recomputed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from threading import local
from typing import TYPE_CHECKING

from pydantic import BaseModel

from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import is_current, latest

from ..analysis.contracts import (
    CONTEXT_COMPLETE_SELECTION_RULES_ID,
    EXECUTED_STATES,
    AlternativeEvidenceAnalystBrief,
    AlternativeEvidenceRetrievalAccessReceipt,
    ComparedUnitRecord,
    FilingComparisonRecord,
)
from ..contracts import (
    AlternativeEvidenceReadingDepth,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSnapshot,
)
from ..documents.contracts import AlternativeEvidenceDocumentSet
from ..publication.artifacts import (
    AlternativeEvidenceArtifactStore,
    AlternativeEvidencePublicationError,
)
from ..publication.contracts import (
    ANALYSIS_POLICY_ROLE,
    CANONICALIZATION_BINDING_ROLE,
    RETRIEVAL_BINDING_ROLE,
    AlternativeEvidenceAnalysisPublication,
    bindings_current,
)
from ..retrieval.contracts import (
    AlternativeEvidenceResolvedSpan,
    AlternativeEvidenceResolvedSpanSet,
    AlternativeEvidenceRetrievalGeneration,
    RetrievalGenerationRecord,
)
from ..sources.contracts import (
    SOURCE_COMMIT_CATEGORY,
    SOURCE_DEFERRAL_CATEGORY,
    AcquiredEvidenceDocument,
    AcquiredEvidenceDocumentReference,
    AcquiredEvidenceSourceCommit,
    AcquiredEvidenceSourceDeferral,
    AcquiredEvidenceSourceReferenceSet,
    AcquiredEvidenceSourceSet,
)
from .history import receipt_matter_selection_id

if TYPE_CHECKING:
    from ..documents.workspace import AlternativeEvidenceDocumentPublisher

SNAPSHOT_CATEGORY = "snapshots"
SOURCE_SET_CATEGORY = "source-document-sets"
DOCUMENT_SET_CATEGORY = "document-sets"
RECEIPT_CATEGORY = "retrieval-access-receipts"
SPAN_SET_CATEGORY = "resolved-span-sets"
SELECTION_COMMITMENT_CORRUPT = "alternative_evidence.selection_commitment_corrupt"
RECORD_DAYS_ROOT = "record-days"


class RecordDays:
    """Index the sealing day of each record in one category.

    Keep days beside the records
    (`record-days/<category>/<day>.<identity>`, empty files, never an artifact
    category), so a reader that can need only the records sealed since a
    moment reads those and no other, however many days the workspace holds
    (Z2). The records' names are listed; only the records asked for are read.

    Derived and rebuildable: a record without its day -- sealed before the
    days were kept, or by a write that stopped between the record and its day
    -- is read once to learn it, a cost and never a record passed over; one
    that does not read back is returned to every reader, which refuses it by
    name as it always did.
    """

    def __init__(
        self,
        artifacts: AlternativeEvidenceArtifactStore,
        category: str,
        sealed_at: Callable[[str], datetime],
    ) -> None:
        """Index sealing dates for records in ``category``."""
        self._records = artifacts.root / category
        self._days = artifacts.root / RECORD_DAYS_ROOT / category
        self._sealed_at = sealed_at
        self._known: dict[str, date | None] | None = None
        self.learned = 0

    def _index(self) -> dict[str, date | None]:
        if self._known is None:
            known: dict[str, date | None] = {}
            for path in self._days.iterdir() if self._days.is_dir() else ():
                day, _, identity = path.name.partition(".")
                try:
                    known[identity] = date.fromisoformat(day)
                except ValueError:
                    continue
            self._known = known
        return self._known

    def note(self, identity: str, sealed_at: datetime) -> None:
        """Keep a record's day as it is sealed."""
        day = sealed_at.astimezone(UTC).date()
        known = self._index()
        if known.get(identity) != day:
            self._days.mkdir(parents=True, exist_ok=True)
            (self._days / f"{day.isoformat()}.{identity}").touch()
            known[identity] = day

    def by_day(self, moment: datetime | None) -> tuple[tuple[str, ...], ...]:
        """Group eligible record identities by day, newest first.

        Include records sealed since the day before ``moment`` or every record
        when ``moment`` is ``None``. Place unknown days first.
        """
        names = (
            [path.stem for path in self._records.glob("*.json")] if self._records.is_dir() else []
        )
        known = self._index()
        for identity in names:
            if identity not in known:
                self.learned += 1
                try:
                    self.note(identity, self._sealed_at(identity))
                except (ValueError, KeyError, OSError):
                    known[identity] = None
        first = None if moment is None else moment.astimezone(UTC).date() - timedelta(days=1)
        groups: dict[date, list[str]] = {}
        for identity in names:
            day = known[identity] or date.max
            if first is None or day >= first:
                groups.setdefault(day, []).append(identity)
        return tuple(tuple(sorted(groups[day])) for day in sorted(groups, reverse=True))

    def since(self, moment: datetime | None) -> tuple[str, ...]:
        """Return eligible record identities in descending sealing-day order.

        Include every record when ``moment`` is ``None``.
        """
        return tuple(identity for group in self.by_day(moment) for identity in group)

    def reader(self) -> _Unread:
        """One index's reads of this category: each record once."""
        return _Unread(self)


class _Unread:
    """Track records one index has read from a category.

    Read each record once; an earlier moment reads only the records it adds.
    """

    def __init__(self, days: RecordDays) -> None:
        self._days = days
        self._read: set[str] = set()
        self._first: date | None = None
        self._whole = False

    def since(self, moment: datetime | None) -> tuple[str, ...]:
        first = None if moment is None else moment.astimezone(UTC).date()
        if self._whole or (self._first is not None and first is not None and first >= self._first):
            return ()
        fresh = tuple(value for value in self._days.since(moment) if value not in self._read)
        self._read.update(fresh)
        if first is None:
            self._whole = True
        else:
            self._first = first if self._first is None else min(self._first, first)
        return fresh

    def held(self, identity: str) -> None:
        """Mark a record sealed and indexed by this runtime as already read."""
        self._read.add(identity)


class SealedExcerpts:
    """Index citation excerpts by source bytes and extraction binding.

    The excerpts come from sealed snapshots.

    A snapshot holds filings accepted at or before it was sealed, so a filing
    is looked for only in the snapshots sealed since the day before it was
    accepted (Z2).
    """

    def __init__(self, artifacts: AlternativeEvidenceArtifactStore, days: RecordDays) -> None:
        """Read sealed snapshots through their dated record index."""
        self._artifacts = artifacts
        self._unread = days.reader()
        self._by_key: dict[tuple[str, str], str] = {}
        self.reused = 0
        self.scanned = 0

    def _index(self, accepted_at: datetime | None) -> dict[tuple[str, str], str]:
        for identity in self._unread.since(accepted_at):
            try:
                snapshot = self._artifacts.load(
                    SNAPSHOT_CATEGORY, identity, AlternativeEvidenceSnapshot
                )
            except (ValueError, FileNotFoundError):
                # A snapshot that does not read back proves nothing here;
                # the readers that depend on it refuse it by name.
                continue
            self.scanned += 1
            self._add(self._by_key, snapshot)
        return self._by_key

    @staticmethod
    def _add(index: dict[tuple[str, str], str], snapshot: AlternativeEvidenceSnapshot) -> None:
        for citation in snapshot.citations:
            if citation.excerpt_binding_hash is not None:
                binding = latest(CANONICALIZATION_BINDING_ROLE, citation.excerpt_binding_hash)
                index.setdefault((citation.source_content_hash, binding), citation.excerpt)

    def add(self, snapshot: AlternativeEvidenceSnapshot) -> None:
        """Index a snapshot already held by this runtime."""
        self._unread.held(snapshot.snapshot_hash)
        self._add(self._by_key, snapshot)

    def find(self, document: AcquiredEvidenceDocument, *, binding_hash: str) -> str | None:
        """Find an excerpt sealed for the document bytes and binding.

        Follow recorded binding moves; return ``None`` without a match.
        """
        key = (document.source_content_hash, latest(CANONICALIZATION_BINDING_ROLE, binding_hash))
        value = self._index(document.accepted_at).get(key)
        if value is not None:
            self.reused += 1
        return value


class SealedCanonicals:
    """Index canonical documents proved by sealed document sets.

    Reuse requires matching source bytes, the current canonicalization
    binding, and full-filing reading depth.

    The join is by the acquisition: a document set names the source
    snapshot it canonicalized, the source-document set of that snapshot
    names each document's bytes, and the two share their semantic handles.
    The canonical text itself comes back through the library's own
    revision read, which verifies the blob against the revision it names.
    A filing is looked for only in the sets sealed since the day before it
    was accepted (Z2); its source set was sealed before its document set,
    in the same preparation.
    """

    def __init__(
        self,
        artifacts: AlternativeEvidenceArtifactStore,
        publisher: AlternativeEvidenceDocumentPublisher,
        *,
        binding_hash: str,
        source_days: RecordDays,
        document_days: RecordDays,
    ) -> None:
        """Join dated source and document sets under the current binding."""
        self._artifacts = artifacts
        self._publisher = publisher
        self._binding_hash = binding_hash
        self._unread_sources = source_days.reader()
        self._unread_documents = document_days.reader()
        self._sources: dict[str, dict[str, AcquiredEvidenceDocumentReference]] = {}
        self._by_key: dict[tuple[str, str], tuple[str, int]] = {}
        self.reused = 0
        self.damaged = 0
        self.scanned = 0

    def _index(self, accepted_at: datetime | None) -> dict[tuple[str, str], tuple[str, int]]:
        for identity in self._unread_sources.since(accepted_at):
            try:
                value = self._artifacts.load_source_set(identity)
            except (ValueError, OSError):
                # Unreadable or not the set its path names: no commitment.
                continue
            if isinstance(value, AcquiredEvidenceSourceReferenceSet):
                self._sources[value.source_snapshot_hash] = {
                    document.semantic_handle: document for document in value.documents
                }
        for identity in self._unread_documents.since(accepted_at):
            try:
                document_set = self._artifacts.load(
                    DOCUMENT_SET_CATEGORY, identity, AlternativeEvidenceDocumentSet
                )
            except (ValueError, FileNotFoundError):
                continue
            self.scanned += 1
            references = self._sources.get(document_set.source_snapshot_hash)
            if references is not None:
                self._join(self._by_key, document_set, references)
        return self._by_key

    def _join(
        self,
        index: dict[tuple[str, str], tuple[str, int]],
        document_set: AlternativeEvidenceDocumentSet,
        references: dict[str, AcquiredEvidenceDocumentReference],
    ) -> None:
        if (
            not is_current(
                CANONICALIZATION_BINDING_ROLE,
                document_set.canonicalization_binding_hash,
                self._binding_hash,
            )
            or document_set.reading_depth is not AlternativeEvidenceReadingDepth.FULL_FILING
        ):
            return
        for reference in document_set.documents:
            source = references.get(reference.semantic_handle)
            if (
                source is None
                or source.media_type != "text/html"
                or source.document_type != reference.document_type
                or source.entity_id != reference.entity_id
                or source.revision != reference.revision_label
                or source.source_name != reference.source_name
            ):
                continue
            index.setdefault(
                (source.source_content_hash, source.document_type),
                (reference.workspace_document_id, reference.workspace_revision),
            )

    def add(
        self, source_set: AcquiredEvidenceSourceSet, document_set: AlternativeEvidenceDocumentSet
    ) -> None:
        """Index source and document sets already held by this runtime."""
        self._unread_documents.held(document_set.document_set_hash)
        if isinstance(source_set, AcquiredEvidenceSourceReferenceSet):
            self._unread_sources.held(source_set.source_set_hash)
            references = {document.semantic_handle: document for document in source_set.documents}
            self._sources[source_set.source_snapshot_hash] = references
            self._join(self._by_key, document_set, references)

    def find(
        self,
        document: AcquiredEvidenceDocument,
        *,
        reading_depth: AlternativeEvidenceReadingDepth,
    ) -> bytes | None:
        """Read verified canonical text for the source bytes, if proved.

        Return ``None`` when no set proves it or the library revision no longer
        verifies. A damaged revision increments ``damaged`` so the caller can
        extract the document again from verified source bytes.
        """
        if (
            reading_depth is not AlternativeEvidenceReadingDepth.FULL_FILING
            or document.media_type != "text/html"
        ):
            return None
        located = self._index(document.accepted_at).get(
            (document.source_content_hash, document.document_type)
        )
        if located is None:
            return None
        try:
            _revision, content = self._publisher.library.read_revision(*located)
        except KnowledgeRetrievalError:
            self.damaged += 1
            return None
        self.reused += 1
        return bytes(content)


SelectionKey = tuple[str, str, str, str, str, str, str, str]


class SealedSelections:
    """Reuse selections proved by sealed retrieval receipts.

    A receipt binds generation content, document references, the query
    program, and selection policy. Another request over the same content can
    reuse the selection without a session.
    """

    def __init__(
        self,
        artifacts: AlternativeEvidenceArtifactStore,
        *,
        selection_policy_hash: str,
        program_hash: str,
    ) -> None:
        """Index receipts eligible under the program and selection policy."""
        self._artifacts = artifacts
        self._policy_hash = selection_policy_hash
        self._program_hash = program_hash
        self._by_key: dict[SelectionKey, AlternativeEvidenceRetrievalAccessReceipt] | None = None
        self.reused = 0
        self.scanned = 0

    @staticmethod
    def key(
        document_set: AlternativeEvidenceDocumentSet,
        generation: RetrievalGenerationRecord,
        matter_selection_id: str,
    ) -> SelectionKey | None:
        """Build the content and matter key for a selection.

        Return ``None`` for a legacy generation without a corpus commitment.
        """
        if not isinstance(generation, AlternativeEvidenceRetrievalGeneration):
            return None
        references = canonical_hash(
            {
                "documents": [item.model_dump(mode="json") for item in document_set.documents],
                "rejections": [item.model_dump(mode="json") for item in document_set.rejections],
                "reading_depth": document_set.reading_depth.value,
            }
        )
        return (
            generation.corpus_hash,
            generation.index_manifest_hash,
            generation.vector_payload_sha256,
            generation.index_spec_hash,
            latest(RETRIEVAL_BINDING_ROLE, generation.retrieval_binding_hash),
            generation.reading_depth.value,
            str(references),
            matter_selection_id,
        )

    def _eligible(self, receipt: AlternativeEvidenceRetrievalAccessReceipt) -> bool:
        # A first reading by the installed program under this policy that
        # names the span set it delivered; a continued matter chain or a
        # structural scan is another selection, and a receipt sealed before
        # the binding was recorded proves nothing.
        matters = receipt.litigation_matters
        return (
            receipt.selection_policy_hash is not None
            and is_current(ANALYSIS_POLICY_ROLE, receipt.selection_policy_hash, self._policy_hash)
            and receipt.query_program_hash == self._program_hash
            and receipt.span_set_hash is not None
            and receipt.structural_scan is None
            # The integrated selection may have skipped every question for
            # want of residual scope: its routing record is the program's
            # account, and a zero-query selection under it is sealed reuse.
            and (bool(receipt.queries) or receipt.routing is not None)
            and (matters is None or matters.continued_from is None)
            # A receipt dealt under a retired residual policy (section V's
            # opt-ins, retired in section X) reads back and is never the
            # default's selection: its routing names the retired rule, or
            # its queries hold the gap searches (`Q-<TOPIC>-GAP<n>`). The
            # receipt's own matter selection id names neither.
            and (
                receipt.routing is None
                or receipt.routing.residual_selection_rules_id
                != CONTEXT_COMPLETE_SELECTION_RULES_ID
            )
            and not any("-GAP" in query.query_id for query in receipt.queries)
        )

    def _index(self) -> dict[SelectionKey, AlternativeEvidenceRetrievalAccessReceipt]:
        if self._by_key is None:
            index: dict[SelectionKey, AlternativeEvidenceRetrievalAccessReceipt] = {}
            root = self._artifacts.root / RECEIPT_CATEGORY
            for path in sorted(root.glob("*.json")) if root.is_dir() else ():
                try:
                    receipt = self._artifacts.load(
                        RECEIPT_CATEGORY, path.stem, AlternativeEvidenceRetrievalAccessReceipt
                    )
                except (ValueError, FileNotFoundError):
                    continue
                self.scanned += 1
                if not self._eligible(receipt):
                    continue
                key = self._key_of(receipt)
                if key is not None:
                    self._hold(index, key, receipt)
            self._by_key = index
        return self._by_key

    @staticmethod
    def _hold(
        index: dict[SelectionKey, AlternativeEvidenceRetrievalAccessReceipt],
        key: SelectionKey,
        receipt: AlternativeEvidenceRetrievalAccessReceipt,
    ) -> None:
        # Every reuse seals a copy that carries the origin's records under its
        # own span set, so one key holds the origin and its copies. The origin
        # is the proof, whatever order the scan meets them in (hash order
        # varies between workspaces); a copy stands in only while no origin
        # is held.
        held = index.get(key)
        if held is None or (
            held.reused_from_receipt_hash is not None and receipt.reused_from_receipt_hash is None
        ):
            index[key] = receipt

    def _key_of(self, receipt: AlternativeEvidenceRetrievalAccessReceipt) -> SelectionKey | None:
        try:
            generation = self._artifacts.load_retrieval_generation(
                receipt.retrieval_generation_hash
            )
            document_set = self._artifacts.load(
                DOCUMENT_SET_CATEGORY, receipt.document_set_hash, AlternativeEvidenceDocumentSet
            )
        except (ValueError, OSError):
            return None
        return self.key(document_set, generation, receipt_matter_selection_id(receipt))

    def add(
        self,
        receipt: AlternativeEvidenceRetrievalAccessReceipt,
        *,
        document_set: AlternativeEvidenceDocumentSet,
        generation: RetrievalGenerationRecord,
    ) -> None:
        """Index a newly sealed receipt when it meets reuse requirements."""
        if self._eligible(receipt):
            key = self.key(document_set, generation, receipt_matter_selection_id(receipt))
            if key is not None:
                self._hold(self._index(), key, receipt)

    def find(
        self,
        *,
        document_set: AlternativeEvidenceDocumentSet,
        generation: RetrievalGenerationRecord,
        matter_selection_id: str,
    ) -> (
        tuple[
            AlternativeEvidenceRetrievalAccessReceipt, tuple[AlternativeEvidenceResolvedSpan, ...]
        ]
        | None
    ):
        """Find a proved receipt and its spans for the same selection.

        Match content, program, policy, and ``matter_selection_id``. Return
        ``None`` when no receipt or named span set exists, allowing computation
        again. Refuse a present set that differs from the receipt's commitment.
        """
        key = self.key(document_set, generation, matter_selection_id)
        if key is None:
            return None
        prior = self._index().get(key)
        if prior is None:
            return None
        spans = self.spans_of(prior)
        if spans is None:
            return None
        self.reused += 1
        return prior, spans

    def spans_of(
        self, receipt: AlternativeEvidenceRetrievalAccessReceipt
    ) -> tuple[AlternativeEvidenceResolvedSpan, ...] | None:
        """Read and verify the span set named by a receipt.

        Return ``None`` for an absent commitment or set. Refuse a present set
        whose request, generation, or delivered handles differ.
        """
        if receipt.span_set_hash is None:
            return None
        try:
            span_set = self._artifacts.load(
                SPAN_SET_CATEGORY, receipt.span_set_hash, AlternativeEvidenceResolvedSpanSet
            )
        except FileNotFoundError:
            return None
        except (AlternativeEvidencePublicationError, ValueError) as error:
            raise ValueError(
                f"{SELECTION_COMMITMENT_CORRUPT}: span set {receipt.span_set_hash[:12]} named by "
                f"receipt {receipt.receipt_hash[:12]}: {error}"
            ) from error
        spans = tuple(span_set.spans)
        if (
            span_set.request_hash != receipt.request_hash
            or span_set.retrieval_generation_hash != receipt.retrieval_generation_hash
            or tuple(span.span_handle for span in spans) != receipt.delivered_span_handles
        ):
            raise ValueError(
                f"{SELECTION_COMMITMENT_CORRUPT}: span set {receipt.span_set_hash[:12]} is not "
                f"the one receipt {receipt.receipt_hash[:12]} delivered"
            )
        return spans


COMPARISON_CATEGORY = "filing-comparisons"
PAIR_SCORE_COMMITMENT_CATEGORY = "pair-score-commitments"


class SealedComparisons:
    """Reuse sealed temporal comparisons of filing pairs.

    ``FilingComparisonRecord`` is named by its whole-record hash. A
    routing resolves a pair only through the binding its chain's sealed
    receipt carries (closure -> record hash): a record on disk under any
    other name is never consulted. Three outcomes, kept apart: the bound
    record is absent -- a missing proof, the pair is computed and sealed
    again; the bound record is there and verifies -- reused; the bound
    record is there and does not verify -- the store's `artifact_tampered`
    when its bytes no longer hash to the bound identity, the contract's
    `identity_invalid` when its own seals disagree, and
    `comparison_binding_mismatch` when a verified record answers another
    closure -- a refusal by name, never a recomputation. A computed record
    is sealed only under the workspace's storage admission -- the routing's
    computed records together, in one admission at `commit`, since the
    admission's own cost (the workspace inventory and the cap) is paid a
    call, not a byte; refused, they are used for this routing and not
    sealed (`unsealed`) nor bound, so the chain computes the pair again
    rather than trusting an unadmitted write.

    A routing stages and commits on its own thread: units routed at once
    each commit their own records, placed under `held` (the runtime's
    writer), never another routing's.
    """

    def __init__(
        self,
        artifacts: AlternativeEvidenceArtifactStore,
        *,
        admit: Callable[[int], None] | None = None,
        held: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> None:
        """Stage comparison records under the supplied storage admission."""
        self._artifacts = artifacts
        self._admit = admit
        self._held = held
        self.reused = 0
        self.computed = 0
        self.unsealed = 0
        self.missing = 0
        self.admissions = 0
        self._routing = local()

    @property
    def _staged(self) -> dict[str, FilingComparisonRecord]:
        staged: dict[str, FilingComparisonRecord] | None = getattr(self._routing, "staged", None)
        if staged is None:
            staged = self._routing.staged = {}
        return staged

    def find(self, closure_hash: str, record_hash: str) -> tuple[ComparedUnitRecord, ...] | None:
        """Return verified compared units for a bound record, if present."""
        try:
            record = self._artifacts.load(COMPARISON_CATEGORY, record_hash, FilingComparisonRecord)
        except FileNotFoundError:
            self.missing += 1
            return None
        if record.comparison_hash != closure_hash:
            raise AlternativeEvidencePublicationError(
                "alternative_evidence.comparison_binding_mismatch"
            )
        self.reused += 1
        return record.units

    def seal(self, record: FilingComparisonRecord) -> str | None:
        """Stage a computed record under its content hash for commit.

        A record already held by the store is its own seal.
        """
        self.computed += 1
        if record.record_hash not in self._staged and not self._artifacts.holds(
            COMPARISON_CATEGORY, record.record_hash, record
        ):
            self._staged[record.record_hash] = record
        return record.record_hash

    def commit(self) -> frozenset[str]:
        """Place staged records under one storage admission.

        Return refused record hashes for the routing to leave unbound;
        ``unsealed`` counts them.
        """
        staged, self._routing.staged = self._staged, {}
        if not staged:
            return frozenset()

        def admit(additional_bytes: int) -> None:
            assert self._admit is not None
            self.admissions += 1
            self._admit(additional_bytes)

        try:
            with self._held():
                self._artifacts.place(
                    [(COMPARISON_CATEGORY, name, record) for name, record in staged.items()],
                    admit=None if self._admit is None else admit,
                )
        except AlternativeEvidencePublicationError:
            # A record under a staged record's name that is not it: refused
            # by name, never bound.
            raise
        except Exception:
            self.unsealed += len(staged)
            return frozenset(staged)
        return frozenset()


ANALYSIS_PUBLICATION_CATEGORY = "analysis-publications"


@dataclass(frozen=True, slots=True)
class _Reading:
    accession: str
    read_as_of: datetime
    published_at: datetime
    publication_hash: str


class SealedReadings:
    """Index filings read by current analyses, grouped by issuer.

    The filing is the unit of reuse (W3).

    A filing is read when a publication under the current binding tuple holds
    it in its document set and its issuer's checks executed. Such a filing is
    not read again while it stays in the window: the selection passes over it
    and its findings carry with the publication that read it. A publication
    under another tuple read under another policy and proves nothing here.
    Each reading is as of its analysis's cutoff, so a run at an earlier cutoff
    never takes a reading made after it.

    Only a filing still in the window is carried, so only an analysis
    published since the window's first day can hold one: the ledger reads
    those, chosen by the day each was published (Z2), and no other record
    (X2), however many days the workspace holds. A reading is made at or after
    the filing it reads and published at or after its cutoff.
    """

    def __init__(
        self,
        artifacts: AlternativeEvidenceArtifactStore,
        *,
        expected_bindings: tuple[str, ...],
        days: RecordDays,
    ) -> None:
        """Read publications under the expected bindings and dated index."""
        self._artifacts = artifacts
        self._expected = expected_bindings
        self._days = days
        self._by_entity: dict[str, list[_Reading]] | None = None
        self._since: datetime | None = None
        self.scanned = 0

    def _index(self, since: datetime) -> dict[str, list[_Reading]]:
        if self._by_entity is None or self._since is None or since < self._since:
            index: dict[str, list[_Reading]] = {}
            for identity in self._days.since(since):
                try:
                    publication = self._artifacts.load(
                        ANALYSIS_PUBLICATION_CATEGORY,
                        identity,
                        AlternativeEvidenceAnalysisPublication,
                    )
                    if (
                        not bindings_current(self._bindings(publication), self._expected)
                        or publication.published_at < since
                    ):
                        continue
                    request = self._artifacts.load(
                        "requests", publication.request_hash, AlternativeEvidenceRequest
                    )
                    documents = self._artifacts.load(
                        DOCUMENT_SET_CATEGORY,
                        publication.document_set_hash,
                        AlternativeEvidenceDocumentSet,
                    )
                    brief = self._artifacts.load(
                        "analyst-briefs",
                        publication.analyst_brief_hash,
                        AlternativeEvidenceAnalystBrief,
                    )
                except (ValueError, FileNotFoundError):
                    # A publication that does not read back whole proves no
                    # reading; its readers refuse it by name.
                    continue
                self.scanned += 1
                self._add(index, publication, request, documents, brief)
            self._by_entity = index
            self._since = since
        return self._by_entity

    @staticmethod
    def _bindings(publication: AlternativeEvidenceAnalysisPublication) -> tuple[str, ...]:
        return (
            publication.acquisition_binding_hash,
            publication.canonicalization_binding_hash,
            publication.retrieval_binding_hash,
            publication.analysis_policy_hash,
            publication.decision_policy_hash,
            publication.publication_binding_hash,
        )

    @staticmethod
    def _add(
        index: dict[str, list[_Reading]],
        publication: AlternativeEvidenceAnalysisPublication,
        request: AlternativeEvidenceRequest,
        documents: AlternativeEvidenceDocumentSet,
        brief: AlternativeEvidenceAnalystBrief,
    ) -> None:
        completion = brief.completion
        for document in documents.documents:
            state = None if completion is None else completion.state_of(document.entity_id)
            if completion is not None and state not in EXECUTED_STATES:
                continue
            index.setdefault(document.entity_id, []).append(
                _Reading(
                    accession=document.revision_label,
                    read_as_of=request.evidence_as_of,
                    published_at=publication.published_at,
                    publication_hash=publication.publication_hash,
                )
            )

    def add(
        self,
        publication: AlternativeEvidenceAnalysisPublication,
        *,
        request: AlternativeEvidenceRequest,
        documents: AlternativeEvidenceDocumentSet,
        brief: AlternativeEvidenceAnalystBrief,
    ) -> None:
        """Index a publication sealed after the index was read."""
        self._days.note(publication.publication_hash, publication.published_at)
        if self._bindings(publication) == self._expected and self._by_entity is not None:
            self._add(self._by_entity, publication, request, documents, brief)

    def read_by(
        self, entity_id: str, *, evidence_as_of: datetime, window_days: int
    ) -> dict[str, str]:
        """Return eligible accession-to-publication readings for an issuer.

        Keep the first publication that read each filing at or before the
        cutoff. A later reading cannot displace it, even when two books were
        prepared together. Include a filing dated on the window's first day.
        """
        since = evidence_as_of - timedelta(days=window_days + 1)
        first: dict[str, _Reading] = {}
        for value in self._index(since).get(entity_id, ()):
            if value.read_as_of > evidence_as_of:
                continue
            held = first.get(value.accession)
            if held is None or (value.published_at, value.publication_hash) < (
                held.published_at,
                held.publication_hash,
            ):
                first[value.accession] = value
        return {accession: value.publication_hash for accession, value in first.items()}


def sealed_field(
    artifacts: AlternativeEvidenceArtifactStore, category: str, field: str
) -> Callable[[str], datetime]:
    """Read a record's sealing day from its own field.

    This supplies a listing hint for a category another owner seals. The
    reader that loads the record performs the verified read.
    """

    def read(identity: str) -> datetime:
        body = json.loads((artifacts.root / category / f"{identity}.json").read_bytes())
        return datetime.fromisoformat(str(body[field]))

    return read


def record_days(artifacts: AlternativeEvidenceArtifactStore) -> dict[str, RecordDays]:
    """Build dated indexes for reuse and analysis reader categories.

    Each record's day is its own sealing time (Z2).
    """

    def sealed(category: str, model: type[BaseModel], field: str) -> Callable[[str], datetime]:
        return lambda identity: getattr(artifacts.load(category, identity, model), field)

    return {
        SNAPSHOT_CATEGORY: RecordDays(
            artifacts,
            SNAPSHOT_CATEGORY,
            sealed(SNAPSHOT_CATEGORY, AlternativeEvidenceSnapshot, "published_at"),
        ),
        SOURCE_SET_CATEGORY: RecordDays(
            artifacts,
            SOURCE_SET_CATEGORY,
            lambda identity: artifacts.load_source_set(identity).acquired_at,
        ),
        SOURCE_COMMIT_CATEGORY: RecordDays(
            artifacts,
            SOURCE_COMMIT_CATEGORY,
            sealed(SOURCE_COMMIT_CATEGORY, AcquiredEvidenceSourceCommit, "committed_at"),
        ),
        SOURCE_DEFERRAL_CATEGORY: RecordDays(
            artifacts,
            SOURCE_DEFERRAL_CATEGORY,
            sealed(SOURCE_DEFERRAL_CATEGORY, AcquiredEvidenceSourceDeferral, "observed_at"),
        ),
        DOCUMENT_SET_CATEGORY: RecordDays(
            artifacts,
            DOCUMENT_SET_CATEGORY,
            sealed(DOCUMENT_SET_CATEGORY, AlternativeEvidenceDocumentSet, "published_at"),
        ),
        ANALYSIS_PUBLICATION_CATEGORY: RecordDays(
            artifacts,
            ANALYSIS_PUBLICATION_CATEGORY,
            sealed(
                ANALYSIS_PUBLICATION_CATEGORY,
                AlternativeEvidenceAnalysisPublication,
                "published_at",
            ),
        ),
    }


def analysis_records(
    artifacts: AlternativeEvidenceArtifactStore, days: RecordDays | None, since: datetime
) -> tuple[AlternativeEvidenceAnalysisPublication, ...]:
    """Read the analysis publications published since a moment, by the day each was published.

    A reader of recent analyses reads no older record, however many days the workspace holds
    (Z2).

    Args:
        artifacts: The Evidence artifact store.
        days: The analysis publications' dated index, or None where no runtime keeps one.
        since: The earliest publication time to read.

    Returns:
        The publications of those days; every publication when no index is kept.
    """
    if days is None:
        return artifacts.values(
            ANALYSIS_PUBLICATION_CATEGORY, AlternativeEvidenceAnalysisPublication
        )
    return tuple(
        artifacts.load(
            ANALYSIS_PUBLICATION_CATEGORY, identity, AlternativeEvidenceAnalysisPublication
        )
        for identity in days.since(since)
    )


def analysis_records_newest_first(
    artifacts: AlternativeEvidenceArtifactStore,
    days: RecordDays | None,
    read: dict[str, AlternativeEvidenceAnalysisPublication],
    *,
    waiting: set[tuple[str, ...]],
) -> Iterator[AlternativeEvidenceAnalysisPublication]:
    """Yield every analysis publication, newest first, a day at a time.

    A publication not yet in ``read`` is read, and added to it, only while ``waiting`` still
    holds a unit.

    Args:
        artifacts: The Evidence artifact store.
        days: The analysis publications' dated index, or None where no runtime keeps one.
        read: The publications already read, by hash; extended as more are read.
        waiting: The units still unanswered; reading stops once it is empty.

    Yields:
        Each day's publications, the newest first.
    """
    groups = (
        (
            tuple(
                value.publication_hash
                for value in artifacts.values(
                    ANALYSIS_PUBLICATION_CATEGORY, AlternativeEvidenceAnalysisPublication
                )
            ),
        )
        if days is None
        else days.by_day(None)
    )
    for group in groups:
        if not waiting:
            return
        for identity in group:
            if identity not in read:
                read[identity] = artifacts.load(
                    ANALYSIS_PUBLICATION_CATEGORY, identity, AlternativeEvidenceAnalysisPublication
                )
        day = [read[identity] for identity in group]
        yield from sorted(day, key=lambda value: value.published_at, reverse=True)


__all__ = [
    "ANALYSIS_PUBLICATION_CATEGORY",
    "COMPARISON_CATEGORY",
    "PAIR_SCORE_COMMITMENT_CATEGORY",
    "RECEIPT_CATEGORY",
    "RECORD_DAYS_ROOT",
    "SOURCE_SET_CATEGORY",
    "SPAN_SET_CATEGORY",
    "RecordDays",
    "SealedCanonicals",
    "SealedComparisons",
    "SealedExcerpts",
    "SealedReadings",
    "SealedSelections",
    "analysis_records",
    "analysis_records_newest_first",
    "record_days",
    "sealed_field",
]
