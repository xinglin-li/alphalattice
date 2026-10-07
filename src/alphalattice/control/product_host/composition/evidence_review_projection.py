"""Project the Evidence and CRO state ladder over a book's exact units.

The Evidence & CRO section's state: one ladder over a book's units, from the
refresh in progress to the published review, with the unit progress and packet requests
it carries. One owner, composed by the review application and read by the operation owner.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, NamedTuple, cast

from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord
from alphalattice.evidence.alternative_evidence.analysis.packet import (
    litigation_continuation_scope,
    litigation_matter_summary,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceMode,
)
from alphalattice.evidence.alternative_evidence.runtime.coverage import (
    UNIT_LIMIT,
    AlternativeEvidenceCoverageRun,
    coverage_unit_ids,
)
from alphalattice.evidence.alternative_evidence.runtime.progress import StageWork
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.interface.local_application.evidence_cro import (
    MANAGED_WORK_NOTE,
    EvidenceCroBook,
    EvidenceCroCoverageProgress,
    EvidenceCroEvidenceVersion,
    EvidenceCroProjection,
    EvidenceCroRequiredAction,
    EvidenceCroStageWork,
    EvidenceCroUnitProgress,
    EvidenceSelectionLabel,
    NextRequests,
    alternative_evidence_expired,
    alternative_evidence_ready_for_review,
    alternative_evidence_superseded,
    analyst_packet_prepared,
    awaiting_alternative_evidence,
    book_projection,
    evidence_authority_not_admitted,
    evidence_refresh_in_progress,
    evidence_selection_ambiguous,
    no_book_to_review,
    project_published_review,
    source_authority_not_admitted,
)
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.investment.portfolio_strategy_lab.reporting.static import (
    format_book_change,
    format_book_weight,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    BookSelector,
    PortfolioReviewInputIncomplete,
)
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewDossier,
    PortfolioReviewPublication,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.coverage.selections import (
    recorded_selection,
)

from .evidence_review_application import (
    DOSSIER_STANDINGS,
    EvidenceSelectionAmbiguous,
    ResolvedBookScope,
    ReviewOutcome,
    _CurrentEvidenceSelection,
    source_ways,
)
from .evidence_review_delivery import EvidenceReviewDelivery
from .plain_refusals import SOURCE_SHORT_CODES, STALE_PACKET, unit_failure_words

if TYPE_CHECKING:
    from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
        AlternativeEvidenceDocumentTaskAdapter,
    )

    from .evidence_review_application import EvidenceReviewApplication


class _UnitPacket(NamedTuple):
    """A prepared unit's packet request and the window it was prepared for, which no packet
    request takes: an offered request is one the Host accepts as it stands (V449)."""

    request: dict[str, str]
    evidence_as_of: str
    evidence_expires_at: str


class _UnitWay(NamedTuple):
    """A failed unit's words, way on and issuers without a source (V541)."""

    detail: str
    next_action: str
    issuers_without_source: tuple[str, ...] | None


class _FailedUnit(NamedTuple):
    """A unit the book's newest run failed to prepare: its issuers, its code, its words and,
    when it failed for its sources, the issuers without one (V541)."""

    unit_id: str
    entity_ids: tuple[str, ...]
    failure_code: str
    detail: str
    issuers_without_source: tuple[str, ...] | None

    @property
    def sources_short(self) -> bool:
        """Whether the unit failed for too few source documents."""
        return self.failure_code.partition(":")[0] in SOURCE_SHORT_CODES


class EvidenceCroProjector:
    """The one typed state of the section, over the application it reads."""

    @staticmethod
    def book_summary(selector: BookSelector) -> dict[str, object]:
        """Discover an exact book without opening its evidence or claiming current standing.

        Args:
            selector: The book identity already checked by its metadata owner.

        Returns:
            Metadata-only standing and the exact full read that verifies descendants.
        """
        return {
            "verification": "METADATA_ONLY_SELECTED_READBACK_VERIFIES_DESCENDANTS",
            "state": "NOT_READ",
            "next_requests": {"review": {"operation": "EVIDENCE_CRO", **selector.request_fields()}},
        }

    @staticmethod
    def review_metadata(
        app: EvidenceReviewApplication,
    ) -> tuple[tuple[PortfolioReviewPublication, ...], tuple[tuple[str, str], ...]]:
        """Discover verified review metadata, retaining failures beside readable peers.

        Args:
            app: The Evidence owner whose sealed publication store is read.

        Returns:
            Strictly loaded publication metadata and named record or location refusals.
            Full selected readback still verifies every descendant at its owner.
        """
        store = app.review_publications.store
        directory = store.root / "cro-review-publications"
        paths = []
        refused: list[tuple[str, str]] = []
        try:
            for path in directory.iterdir():
                if path.name.endswith(".json"):
                    paths.append(path)
        except FileNotFoundError:
            if not paths:
                return (), ()
            refused.append(("cro-review-publications", "research_history.entry_unreadable"))
        except OSError as error:
            refused.append(
                (
                    "cro-review-publications",
                    public_failure(error, "research_history.entry_unreadable"),
                )
            )
        readable: list[PortfolioReviewPublication] = []
        for path in sorted(paths):
            try:
                readable.append(
                    store.load("cro-review-publications", path.stem, PortfolioReviewPublication)
                )
            except (ValueError, OSError) as error:
                refused.append(
                    (path.stem, public_failure(error, "research_history.entry_unreadable"))
                )
        return tuple(readable), tuple(refused)

    def __init__(
        self, app: EvidenceReviewApplication, delivery: EvidenceReviewDelivery | None = None
    ) -> None:
        """Bind the evidence/CRO projection to one application and delivery owner.

        Args:
            app: Deterministic review application owner.
            delivery: Optional existing bounded delivery owner; otherwise construct it for this
                application.
        """
        self.app = app
        # The published review a state names is read through the one delivery owner.
        self.delivery = delivery if delivery is not None else EvidenceReviewDelivery(app)

    def projection(
        self,
        selector: BookSelector | None = None,
        *,
        publication_hash: str | None = None,
        reading: bool = False,
        entity_id: str | None = None,
        topic: str | None = None,
        last_days: int | None = None,
    ) -> EvidenceCroProjection:
        """One typed state for the Evidence & CRO section. Never a guess."""
        chosen = self.app.default_selector(selector)
        if chosen is None:
            return no_book_to_review()
        selector_fields = chosen.request_fields()

        def bound(operation: str, **fields: str) -> dict[str, str]:
            return {"operation": operation, **selector_fields, **fields}

        # Managed work is the one thing a missing credential refuses; every
        # native entry below is offered on the evidence alone.
        managed = self.app.model_authority_admitted

        def finish(
            projection: EvidenceCroProjection,
            *,
            next_requests: NextRequests,
            refresh: bool = False,
            review: bool = False,
        ) -> EvidenceCroProjection:
            if managed and refresh:
                next_requests["refresh"] = bound("EVIDENCE_REFRESH")
            if managed and review:
                next_requests["review"] = bound("CRO_REVIEW")
            explanation = projection.explanation
            if not managed and (refresh or review):
                explanation = explanation + " " + MANAGED_WORK_NOTE
            # The actions a state offers are the requests it offers, never a list of
            # its own: a book awaiting its analysis offers no CRO review (V198).
            offered = tuple(
                action
                for action, available in (
                    ("REFRESH_EVIDENCE", managed and refresh),
                    ("REVIEW_WITH_CRO", managed and review),
                )
                if available
            )
            return replace(
                projection,
                explanation=explanation,
                available_actions=offered,
                next_requests=next_requests,
            )

        if publication_hash is not None:
            # The existing export owner verifies exact book/receipt/source agreement,
            # including old authority and expiry readback. Do not rewrite its export; read
            # on the review it verified (V151).
            snapshot, verified, analyses = self.delivery.verified_read_view(
                chosen, publication_hash
            )
            assert verified is not None, "an export of a named review verifies it"
            historical_view = verified
            historical_subject = historical_view.publication.experiment_subject
            projected = project_published_review(
                recommendation=historical_view.recommendation,
                dossier=historical_view.dossier,
                receipt=historical_view.receipt,
                percent=format_book_weight,
                change=format_book_change,
                book=book_projection(
                    authority=historical_view.publication.book_authority,
                    result_hash=historical_view.publication.result_hash,
                    held_count=historical_view.dossier.held_count,
                    formation_session=None
                    if historical_subject is None
                    else historical_subject.portfolio_session.isoformat(),
                    experiment_subject=cast(
                        dict[str, object] | None, snapshot.get("experiment_subject")
                    ),
                    update_subject=cast(dict[str, object] | None, snapshot.get("update_subject")),
                ),
            )
            # The older dossier did not seal the detailed position breakdown.
            # Never borrow today's issuer mapping/eligibility to fill that gap.
            return replace(
                projected,
                review_publication_hash=publication_hash,
                published_reading=self.delivery.published_reading(
                    historical_view,
                    verified=analyses,
                    entity_id=entity_id,
                    topic=topic,
                    last_days=last_days,
                )
                if reading
                else None,
                available_actions=(),
                explanation=(
                    "Exact historical review; no current eligibility or new work is granted. "
                    "Detailed position-scope breakdown was not sealed in this record."
                    + (
                        ""
                        if historical_view.under_installed_policy
                        else " It was sealed under an earlier CRO review policy; it reads as "
                        "recorded."
                    )
                ),
                next_requests={
                    "export": bound("EVIDENCE_CRO_EXPORT", review_publication_hash=publication_hash)
                },
            )
        if not self.app.has_evidence_authority:
            return evidence_authority_not_admitted()
        try:
            resolved = self.app.resolve_book(chosen)
        except PortfolioReviewInputIncomplete as error:
            subject = error.book.update_subject
            experiment_subject = error.book.experiment_subject
            return EvidenceCroProjection(
                state="REVIEW_INPUT_INCOMPLETE",
                explanation=(
                    "The first recorded holdings session has no sealed preceding session. "
                    "It stays readable; missing changes cannot be invented for review."
                    if experiment_subject is not None
                    else "This historical entry did not seal per-listing pretrade weights. "
                    "It stays readable; missing changes are not zero and cannot authorize a review."
                ),
                book=EvidenceCroBook(
                    authority=str(error.book.authority),
                    explanation="Research holdings, not venue execution.",
                    result_hash=None,
                    update_subject=None if subject is None else subject.model_dump(mode="json"),
                    experiment_subject=None
                    if experiment_subject is None
                    else experiment_subject.model_dump(mode="json"),
                ),
                available_actions=(),
            )
        book = self.app._book_projection(resolved)
        if self.app.evidence_publications is None:
            # Issuers are named; what is missing is the source package and the
            # pinned retrieval runtime, which no credential supplies.
            return source_authority_not_admitted(book=book)
        active = self.app.active_evidence_refresh(scope=resolved.scope)
        if active is not None:
            return finish(
                replace(
                    evidence_refresh_in_progress(
                        book=book,
                        task_id=str(active.task_id),
                        lifecycle=str(active.lifecycle.value),
                    ),
                    coverage_progress=self._coverage_progress(resolved, selections=None),
                ),
                next_requests={"status": {"operation": "STATUS", "task_id": str(active.task_id)}},
            )
        try:
            selections = self.app._select_unit_evidence(resolved)
        except EvidenceSelectionAmbiguous:
            # The choice is the typed required action and the versions offered;
            # no request is composed for it because the person must name one.
            return finish(
                replace(
                    evidence_selection_ambiguous(
                        book=book, versions=self._eligible_versions(resolved)
                    ),
                    coverage_progress=self._coverage_progress(resolved, selections=None),
                ),
                next_requests={},
            )
        return self._coverage_projection(
            chosen,
            resolved,
            book=book,
            selections=selections,
            finish=finish,
            bound=bound,
            now=self.app.clock(),
            reading=reading,
            entity_id=entity_id,
            topic=topic,
            last_days=last_days,
        )

    def _coverage_projection(
        self,
        chosen: BookSelector,
        resolved: ResolvedBookScope,
        *,
        book: EvidenceCroBook,
        selections: dict[tuple[str, ...], _CurrentEvidenceSelection],
        finish: Callable[..., EvidenceCroProjection],
        bound: Callable[..., dict[str, str]],
        now: datetime,
        reading: bool,
        entity_id: str | None,
        topic: str | None,
        last_days: int | None,
    ) -> EvidenceCroProjection:
        """The state of a book, read over its units.

        A refresh in progress was answered before this; here, a review
        already published against exactly these analyses, else the analyses
        ready for the dossier (complete, or partial with the missing units
        named), else the packets prepared and waiting for their analyses,
        else what stands in the way. A book of one unit reads the same states
        with its one unit; progress rides along for a wider one.
        """

        progress = self._coverage_progress(resolved, selections=selections)
        current = {
            key: value
            for key, value in selections.items()
            if value.disposition == "CURRENT" and value.evidence is not None
        }
        # Eligible earlier readings can support a dossier even when every unit
        # lacks a current analysis. The dossier owner decides their eligibility.
        dossier = self.app._resolve_review_dossier(chosen)
        if isinstance(dossier, ReviewOutcome) and dossier.disposition in DOSSIER_STANDINGS:
            # A dossier these analyses cannot make is the book's standing, said with its way on;
            # the book's read never fails on it (V546).
            packaged = self.app.evidence_policy.mode is AlternativeEvidenceMode.RECORDED
            return finish(
                replace(
                    awaiting_alternative_evidence(book=book),
                    explanation=dossier.detail,
                    coverage_progress=progress,
                    source_ways=source_ways(self.app.workspace, entities=None, book=chosen)
                    if packaged
                    else None,
                ),
                next_requests=dict(dossier.next_requests or {}),
            )
        if isinstance(dossier, PortfolioReviewDossier):
            recorded = self.app.published_review_for_book(
                resolved.book, analysis_publication_hash=dossier.analysis_publication_hash
            )
            historical = None
            if (
                recorded is not None
                and not current
                and recorded.dossier.dossier_hash != dossier.dossier_hash
            ):
                # The carried readings alone name no day: a review answers a
                # day that read no unit only when sealed on that day's dossier.
                historical, recorded = recorded, None
            if recorded is not None and not recorded.under_installed_policy:
                # Sealed under an earlier CRO review policy: the review reads back
                # by its handle, and the book waits for one.
                historical, recorded = recorded, None
            versions = self._eligible_versions(resolved)
            choices = versions if len(versions) > len(current) else ()
            partial = len(current) < len(selections)
            if recorded is None:
                pending: dict[str, str] = {}
                stops: set[str] = set()
                unit_ids = coverage_unit_ids(tuple(selections))
                for key, value in current.items():
                    assert value.evidence is not None
                    receipt = value.evidence.lineage.access_receipt
                    scope = litigation_continuation_scope(receipt)
                    summary = litigation_matter_summary(receipt)
                    if scope is None or summary is None:
                        continue
                    if scope["state"] == "PENDING":
                        chain = cast(dict[str, object], summary["reading_chain"])
                        remaining = cast(dict[str, int] | None, chain["remaining"])
                        if remaining is None:
                            # A first reading has no cumulative continuation allowance.
                            # It permits a bounded review, not invented reading authority.
                            stops.add("No cumulative continuation allowance is declared.")
                        elif all(remaining[name] > 0 for name in ("sessions", "windows")):
                            pending[receipt.receipt_hash] = unit_ids[key]
                        else:
                            stops.add("The declared continuation allowance is exhausted.")
                    elif scope["state"] == "NOTHING_RESUMABLE":
                        stops.add("The sealed reading plan has no resumable work.")
                adapter = self.app.evidence_task_adapter
                preparations = (
                    adapter.prepared_receipts() if pending and adapter is not None else {}
                )
                continuation_packets: NextRequests = {}
                for receipt_hash, unit in pending.items():
                    location = preparations.get(receipt_hash)
                    if location is None:
                        # The sealed analysis remains readable even if its preparation
                        # Task was not retained. Never invent an executable continuation.
                        stops.add("The selected analysis's preparation Task is not retained.")
                        continue
                    task_id, unit_id = location
                    continuation_packets[f"packet_{unit}"] = bound(
                        "EVIDENCE_PACKET",
                        task_id=task_id,
                        **({"evidence_unit_id": unit_id} if unit_id is not None else {}),
                    )
                if continuation_packets:
                    return finish(
                        replace(
                            awaiting_alternative_evidence(book=book),
                            explanation=(
                                "The selected analysis still has source the sealed reading plan "
                                "can continue. Read its exact packet and settle its continuation "
                                "within the declared session and window allowance before CRO. "
                                "Follow each continuation answer's successor packet, publish its "
                                "Analyst answer, then read this book's Evidence again."
                            ),
                            coverage_progress=progress,
                            required_actions=(
                                EvidenceCroRequiredAction(
                                    "SETTLE_EVIDENCE_CONTINUATION",
                                    None,
                                    "Read the packet's continuation scope and remaining allowance; "
                                    "continue the sealed plan within that allowance.",
                                    True,
                                ),
                                EvidenceCroRequiredAction(
                                    "ANALYZE_CONTINUED_EVIDENCE",
                                    None,
                                    "Publish the Analyst answer to the returned successor packet "
                                    "before requesting this book's CRO review.",
                                    True,
                                ),
                            ),
                        ),
                        next_requests=continuation_packets,
                    )
                projection = replace(
                    alternative_evidence_ready_for_review(
                        book=book,
                        evidence_as_of=dossier.evidence_as_of.isoformat(),
                        evidence_expires_at=dossier.evidence_expires_at.isoformat(),
                    ),
                    eligible_versions=choices,
                    coverage_progress=progress,
                    source_ways=self._unit_source_ways(chosen, resolved, progress),
                )
                if not current:
                    projection = replace(
                        projection,
                        explanation=(
                            "Eligible earlier readings support this dossier. Read the dossier"
                            " and submit a bounded CRO assessment."
                        ),
                    )
                if partial:
                    coverage_words = (
                        f"{len(current)} of {len(selections)} units have a current analysis;"
                        " the review reads those exactly and names the rest as not reviewed."
                        if current
                        else f"{len(current)} of {len(selections)} units have a current analysis;"
                        " current coverage gaps remain named."
                    )
                    projection = replace(
                        projection,
                        explanation=" ".join((projection.explanation, coverage_words)),
                    )
                if stops:
                    projection = replace(
                        projection,
                        explanation=" ".join((projection.explanation, *sorted(stops)))
                        + (
                            " Pending source and other unread ranges stay unread; the CRO "
                            "assessment is bounded to the recorded analysis, not every source."
                        ),
                    )
                projection = replace(
                    projection,
                    required_actions=(
                        EvidenceCroRequiredAction(
                            "READ_CRO_DOSSIER",
                            None,
                            "Read this book's dossier and the recorded limits of its Evidence.",
                            False,
                        ),
                        EvidenceCroRequiredAction(
                            "SUBMIT_CRO_ASSESSMENT",
                            None,
                            "Submit the CRO assessment through this book's offered bundle.",
                            False,
                        ),
                    ),
                )
                next_requests: NextRequests = {
                    "dossier": bound("CRO_REVIEW_DOSSIER"),
                }
                if historical is not None:
                    handle = historical.publication.publication_hash
                    next_requests["export"] = bound(
                        "EVIDENCE_CRO_EXPORT", review_publication_hash=handle
                    )
                    historical_words = (
                        "The published review reads back by its handle; it answers its"
                        " sealed dossier, not this newly compiled dossier."
                        if historical.under_installed_policy
                        else "The review published on this dossier reads back by its handle;"
                        " this build no longer gives its recommendation, so it is not the"
                        " current answer."
                    )
                    projection = replace(
                        projection,
                        review_publication_hash=handle,
                        explanation=" ".join((projection.explanation, historical_words)),
                    )
                analysed = coverage_unit_ids(tuple(selections))
                next_requests.update(
                    {
                        key: packet.request
                        for key, packet in self._unit_packets(
                            resolved,
                            bound=bound,
                            now=now,
                            exclude=frozenset(analysed[key] for key in current),
                        ).items()
                    }
                )
                # A way on a stale row names in words is offered bound (S1): the preview, which
                # prepares its unit again under the authority now held (V567).
                stale_way = refusal_words(STALE_PACKET)["next_action"]
                if progress is not None and any(
                    row.next_action == stale_way for row in progress.units
                ):
                    next_requests["preview"] = bound("EVIDENCE_PREVIEW")
                return finish(projection, next_requests=next_requests, review=True, refresh=True)
            # How each unit's analysis came to be the one; a carried child
            # is an earlier reading, never a selection, and its analysis may
            # have expired since (W3). A review of carried readings alone
            # chose among no analyses.
            children = recorded.dossier.evidence_children
            labels = {
                self.app.evidence_selection_label(
                    obligation_hash=child.obligation_hash,
                    publication_hash=child.analysis_publication_hash,
                )
                for child in children
                if child.read_as_of is None
            } or (
                {"UNIQUE_CURRENT"}
                if children
                else {
                    self.app.evidence_selection_label(
                        obligation_hash=recorded.dossier.obligation_hash,
                        publication_hash=recorded.dossier.analysis_publication_hash,
                    )
                }
            )
            label: EvidenceSelectionLabel = (
                "UNIQUE_CURRENT"
                if labels == {"UNIQUE_CURRENT"}
                else "EXPLICIT_OLDER_CURRENT_SELECTION"
                if "EXPLICIT_OLDER_CURRENT_SELECTION" in labels
                else "EXPLICIT_CURRENT_SELECTION"
            )
            published_reading = (
                self.delivery.published_reading(
                    recorded,
                    verified=tuple(
                        value.evidence
                        for value in selections.values()
                        if value.evidence is not None
                    ),
                    entity_id=entity_id,
                    topic=topic,
                    last_days=last_days,
                )
                if reading
                else None
            )
            projected = project_published_review(
                recommendation=recorded.recommendation,
                dossier=recorded.dossier,
                receipt=recorded.receipt,
                percent=format_book_weight,
                change=format_book_change,
                book=book,
                selection=label,
                scope=resolved.scope,
                exposure=resolved.projection,
            )
            return finish(
                replace(
                    projected,
                    review_publication_hash=recorded.publication.publication_hash,
                    published_reading=published_reading,
                    eligible_versions=choices,
                    coverage_progress=progress,
                ),
                next_requests={
                    "export": bound(
                        "EVIDENCE_CRO_EXPORT",
                        review_publication_hash=recorded.publication.publication_hash,
                    )
                },
            )
        adapter = self.app.evidence_task_adapter
        if adapter is None:
            # Published analyses would be readable; preparing one is not
            # possible until the source package and runtime are admitted.
            return source_authority_not_admitted(book=book)
        packets = self._unit_packets(resolved, bound=bound, now=now)
        dispositions = {value.disposition for value in selections.values()}
        if packets:
            first = next(iter(packets.values()))
            projection = replace(
                analyst_packet_prepared(
                    book=book,
                    task_id=first.request["task_id"],
                    evidence_as_of=first.evidence_as_of,
                    evidence_expires_at=first.evidence_expires_at,
                ),
                coverage_progress=progress,
                source_ways=self._unit_source_ways(chosen, resolved, progress),
            )
            return finish(
                projection,
                next_requests={
                    **{key: packet.request for key, packet in packets.items()},
                    "preview": bound("EVIDENCE_PREVIEW"),
                },
                refresh=True,
            )
        if "EXPIRED" in dispositions or "SUPERSEDED" in dispositions:
            stale_key = next(
                key
                for key, value in selections.items()
                if value.disposition in {"EXPIRED", "SUPERSEDED"}
            )
            stale = selections[stale_key]
            assert stale.question is not None and stale.evidence is not None
            make = (
                alternative_evidence_expired
                if stale.disposition == "EXPIRED"
                else alternative_evidence_superseded
            )
            projection = replace(
                make(
                    book=book,
                    evidence_as_of=stale.question.obligation.evidence_as_of.isoformat(),
                    evidence_expires_at=stale.evidence.publication.expires_at.isoformat(),
                ),
                coverage_progress=progress,
            )
            stale_requests: NextRequests = {"preview": bound("EVIDENCE_PREVIEW")}
            historical = self.app.published_review_for_book(
                resolved.book,
                analysis_publication_hash=stale.evidence.publication.publication_hash,
            )
            if historical is not None:
                # The review published against the stale analysis stays
                # readable by its handle; it grants no current eligibility.
                handle = historical.publication.publication_hash
                stale_requests["export"] = bound(
                    "EVIDENCE_CRO_EXPORT", review_publication_hash=handle
                )
                projection = replace(
                    projection,
                    review_publication_hash=handle,
                    explanation=projection.explanation
                    + " The review published against it remains readable by its handle.",
                )
            return finish(projection, next_requests=stale_requests, refresh=True)
        documents = tuple(
            value
            for value in adapter.resources.recorded_documents
            if value.entity_id in resolved.scope.ordered_entity_ids
        )
        detail = None
        expired = self._expired_packet(resolved, now=now)
        failed = self._failed_units(resolved, progress)
        total = 1 if progress is None else progress.units_total
        packaged = self.app.evidence_policy.mode is AlternativeEvidenceMode.RECORDED
        # Nothing these sources can prepare: a refresh would fail the same way, so none is offered.
        standing = _sources_cannot_cover(failed, total=total, recorded=packaged)
        if failed:
            detail = _failed_words(
                failed, total=total, recorded=packaged, standing=standing, one_unit=progress is None
            )
        elif expired is not None:
            detail = (
                f"The last prepared packets (as of {expired[0].isoformat()}) expired at "
                f"{expired[1].isoformat()} and no analysis was submitted against them."
            )
        elif packaged and not documents:
            detail = (
                "The admitted source package records no documents for this book's "
                "issuers, so preparation would find nothing to read."
            )
        projection = awaiting_alternative_evidence(book=book, detail=detail)
        if standing:
            # The book's standing is the whole explanation: preparing again is no way on here.
            projection = replace(projection, explanation=str(detail))
        return finish(
            replace(
                projection,
                coverage_progress=progress,
                source_ways=self._source_ways(chosen, failed, total=total) if packaged else None,
            ),
            next_requests={"preview": bound("EVIDENCE_PREVIEW")},
            refresh=not standing,
        )

    def _unit_source_ways(
        self,
        chosen: BookSelector,
        resolved: ResolvedBookScope,
        progress: EvidenceCroCoverageProgress | None,
    ) -> dict[str, object] | None:
        """The ways on for a book prepared in part: for its units that failed for their sources
        under the recorded package, beside the units it reviews (V541)."""

        if self.app.evidence_policy.mode is not AlternativeEvidenceMode.RECORDED:
            return None
        return self._source_ways(
            chosen,
            self._failed_units(resolved, progress),
            total=1 if progress is None else progress.units_total,
        )

    def _source_ways(
        self, chosen: BookSelector, failed: list[_FailedUnit], *, total: int
    ) -> dict[str, object] | None:
        """The ways on for the units that failed for their sources under the recorded package:
        the person's official acquisition, and a package only for a book of one unit, which one
        package covers whole (V541, V546)."""

        short = [unit for unit in failed if unit.sources_short]
        if not short:
            return None
        whole = short[0].entity_ids if total == 1 else None
        return source_ways(self.app.workspace, entities=whole, book=chosen)

    def _failed_units(
        self, resolved: ResolvedBookScope, progress: EvidenceCroCoverageProgress | None
    ) -> list[_FailedUnit]:
        """The units the book's newest run failed to prepare, heaviest first: from its progress
        table, or, for a book of one unit, which has none, from the run's own unit states."""

        if progress is not None:
            return [
                _FailedUnit(
                    row.unit_id,
                    row.entity_ids,
                    row.failure_code or "",
                    row.detail or "",
                    row.issuers_without_source,
                )
                for row in progress.units
                if row.state == "FAILED"
            ]
        adapter = self.app.evidence_task_adapter
        latest = next(self.app._run_tasks(resolved.scope), None)
        if adapter is None or latest is None or adapter.run_of(latest) is None:
            return []
        failed = []
        for unit_id, value in sorted(adapter.unit_states(latest).items()):
            if value["state"] != "FAILED":
                continue
            entities = tuple(cast(tuple[str, ...], value["ordered_entity_ids"]))
            way = self._unit_way(value, entities)
            failed.append(
                _FailedUnit(
                    unit_id,
                    entities,
                    str(value["failure_code"]),
                    way.detail,
                    way.issuers_without_source,
                )
            )
        return failed

    def _unit_way(self, state: dict[str, object], entities: tuple[str, ...]) -> _UnitWay:
        """A failed unit's words, way on and issuers without a source (V541): those its failure
        sealed, every issuer of a unit that held no document, and none named by a failure sealed
        before, which recorded none."""

        code = str(state["failure_code"])
        words = unit_failure_words(
            code, recorded=self.app.evidence_policy.mode is AlternativeEvidenceMode.RECORDED
        )
        uncovered = tuple(cast(tuple[str, ...], state.get("uncovered_entity_ids") or ()))
        return _UnitWay(
            detail=words["detail"],
            next_action=words["next_action"],
            issuers_without_source=uncovered
            if uncovered
            else entities
            if code.partition(":")[0] == "alternative_evidence.document_set_empty"
            else None,
        )

    def _expired_packet(
        self, resolved: ResolvedBookScope, *, now: datetime
    ) -> tuple[datetime, datetime] | None:
        """The cutoff and expiry of the newest run of this book that prepared
        a unit whose packet has since expired, when one did."""

        adapter = self.app.evidence_task_adapter
        if adapter is None:
            return None
        for task in self.app._run_tasks(resolved.scope):
            run = adapter.run_of(task)
            if run is None:
                continue
            states = adapter.unit_states(task)
            for unit in run.units:
                if states[unit.unit_id]["state"] != "PREPARED":
                    continue
                try:
                    evidence_as_of, expires_at = adapter.preparation_window(task, unit.unit_id)
                except (ValueError, KnowledgeRetrievalError):
                    continue
                if now > expires_at:
                    return evidence_as_of, expires_at
            return None
        return None

    def _eligible_versions(
        self, resolved: ResolvedBookScope
    ) -> tuple[EvidenceCroEvidenceVersion, ...]:
        """What a person is choosing between, and which one is already chosen."""

        recorded = recorded_selection(self.app.artifacts, resolved.scope.scope_hash)
        chosen = None if recorded is None else recorded.analysis_publication_hash
        return tuple(
            EvidenceCroEvidenceVersion(
                analysis_publication_hash=view.publication.publication_hash,
                evidence_as_of=question.obligation.evidence_as_of.isoformat(),
                evidence_expires_at=view.publication.expires_at.isoformat(),
                issuer_count=len(question.obligation.ordered_entity_ids),
                is_selected=view.publication.publication_hash == chosen,
            )
            for question, view in self.app.eligible_evidence(resolved)
        )

    def _unit_packets(
        self,
        resolved: ResolvedBookScope,
        *,
        bound: Callable[..., dict[str, str]],
        now: datetime,
        exclude: frozenset[str] = frozenset(),
    ) -> dict[str, _UnitPacket]:
        """A packet request per prepared unit whose packet still reads, keyed
        `packet_<unit>`, each beside the cutoff and expiry it was prepared
        for; `exclude` names the units already answered."""

        adapter = self.app.evidence_task_adapter
        if adapter is None:
            return {}
        requests: dict[str, _UnitPacket] = {}
        for task in self.app._run_tasks(resolved.scope):
            run = adapter.run_of(task)
            if run is None or adapter.stale_authority(task) is not None:
                # A run prepared under authority the Host no longer holds offers no packet, which
                # the Host would refuse; its rows say so, with the preview (V547, U79).
                continue
            states = adapter.unit_states(task)
            for unit in run.units:
                key = f"packet_{unit.unit_id}"
                if (
                    key in requests
                    or unit.unit_id in exclude
                    or states[unit.unit_id]["state"] != "PREPARED"
                ):
                    continue
                try:
                    evidence_as_of, expires_at = adapter.preparation_window(task, unit.unit_id)
                except (ValueError, KnowledgeRetrievalError):
                    continue
                if now > expires_at:
                    continue
                requests[key] = _UnitPacket(
                    request=bound(
                        "EVIDENCE_PACKET", task_id=str(task.task_id), evidence_unit_id=unit.unit_id
                    ),
                    evidence_as_of=evidence_as_of.isoformat(),
                    evidence_expires_at=expires_at.isoformat(),
                )
        return dict(sorted(requests.items()))

    def _coverage_progress(
        self,
        resolved: ResolvedBookScope,
        *,
        selections: dict[tuple[str, ...], _CurrentEvidenceSelection] | None,
    ) -> EvidenceCroCoverageProgress | None:
        """Where every unit of a wide book stands; nothing for a book that fits one.

        Prepared is read from the newest coverage Task's own stage receipts;
        analyzed from the unit's current publication; reviewed from a
        published recommendation that read that publication. The
        denominators are the whole book.
        """

        scope = resolved.scope
        units = self.app._units(scope)
        sealed = self.app._sealed_run(scope, evidence_as_of=None)
        carried: tuple[str, ...] = () if sealed is None else sealed.carried
        if len(units) <= 1 and not carried:
            return None
        adapter = self.app.evidence_task_adapter
        ids = coverage_unit_ids(units) if units else {}
        positions = {value.entity_id: value for value in scope.selected_issuers}
        states: dict[tuple[str, ...], dict[str, object]] = {}
        works: dict[str, EvidenceCroStageWork] = {}
        preparation: tuple[EvidenceCroStageWork, ...] = ()
        run_hash: str | None = None if sealed is None else sealed.run_hash
        nothing_filed: tuple[str, ...] = () if sealed is None else sealed.nothing_filed
        latest = next(self.app._run_tasks(scope), None)
        stale: str | None = None
        if adapter is not None and latest is not None:
            run = adapter.run_of(latest)
            if run is not None and (sealed is None or run.run_hash == sealed.run_hash):
                run_hash = run.run_hash
                nothing_filed = run.nothing_filed
                # Packets prepared under a package an install has since replaced (V547).
                stale = adapter.stale_authority(latest)
                states = {
                    tuple(cast(tuple[str, ...], value["ordered_entity_ids"])): value
                    for value in adapter.unit_states(latest).values()
                }
                states = {
                    key: {**value, "task_id": str(latest.task_id)} for key, value in states.items()
                }
                if latest.lifecycle in {TaskLifecycle.RUNNING, TaskLifecycle.CANCEL_REQUESTED}:
                    works, preparation = _preparation_work(adapter, latest, run, states)
        # A review reads an analysis only after it is published: the reviews
        # older than every analysis asked about are not read (W3's daily ones).
        analysed = [
            value.evidence.publication.published_at
            for value in (selections or {}).values()
            if value.evidence is not None
        ]
        reviewed_hashes = (
            {
                publication_hash
                for view in self.app._published_reviews_for_book(resolved.book, since=min(analysed))
                for publication_hash in view.dossier.evidence_publication_hashes
            }
            if analysed
            else set()
        )
        rows: list[EvidenceCroUnitProgress] = []
        prepared_weight = 0.0
        analyzed_weight = 0.0
        for entities in units:
            state = states.get(entities)
            selected = None if selections is None else selections.get(entities)
            analyzed = (
                selected is not None
                and selected.disposition == "CURRENT"
                and selected.evidence is not None
            )
            publication_hash = (
                selected.evidence.publication.publication_hash
                if analyzed and selected is not None and selected.evidence is not None
                else None
            )
            reviewed = publication_hash is not None and publication_hash in reviewed_hashes
            prepared = state is not None and state["state"] == "PREPARED"
            failed = state is not None and state["state"] == "FAILED"
            way = self._unit_way(state, entities) if failed and state is not None else None
            if prepared and stale is not None:
                words = refusal_words(stale)
                way = _UnitWay(words["detail"], words["next_action"], None)
            weight = math.fsum(positions[entity].ending_weight for entity in entities)
            if prepared or analyzed:
                prepared_weight += weight
            if analyzed:
                analyzed_weight += weight
            label = (
                "REVIEWED"
                if reviewed
                else "ANALYZED"
                if analyzed
                else "PREPARED"
                if prepared
                else "FAILED"
                if failed
                else "PENDING"
                if state is not None
                else "NOT_STARTED"
            )
            rows.append(
                EvidenceCroUnitProgress(
                    unit_id=ids[entities],
                    entity_ids=entities,
                    listing_count=sum(len(positions[entity].listing_ids) for entity in entities),
                    ending_weight=format_book_weight(weight),
                    state=label,
                    stages_done=0 if state is None else int(cast(int, state["stages_done"])),
                    stages_expected=0
                    if state is None
                    else int(cast(int, state["stages_expected"])),
                    failure_code=None if state is None else cast(str | None, state["failure_code"]),
                    packet_task_id=None if not prepared or state is None else str(state["task_id"]),
                    packet_unit_id=None if not prepared or state is None else ids[entities],
                    analysis_publication_hash=publication_hash,
                    evidence_as_of=(
                        selected.question.obligation.evidence_as_of.isoformat()
                        if analyzed and selected is not None and selected.question is not None
                        else None
                    ),
                    work=works.get(ids[entities]) if label == "PENDING" else None,
                    **({} if way is None or label not in {"FAILED", "PREPARED"} else way._asdict()),
                )
            )
        # A holding that filed nothing in the window, or has nothing new in it,
        # mapped to its issuer; it is named apart from the units, never counted
        # as unmapped.
        mapped = (
            {entity for entities in units for entity in entities}
            | set(nothing_filed)
            | set(carried)
        )
        book_listings = len(resolved.projection.positions)
        mapped_listings = sum(len(positions[entity].listing_ids) for entity in mapped)
        return EvidenceCroCoverageProgress(
            run_hash=run_hash,
            unit_limit=UNIT_LIMIT,
            book_listings=book_listings,
            mapped_issuers=len(mapped),
            unmapped_listings=book_listings - mapped_listings,
            units_total=len(rows),
            units_prepared=sum(
                1 for row in rows if row.state in {"PREPARED", "ANALYZED", "REVIEWED"}
            ),
            units_failed=sum(1 for row in rows if row.state == "FAILED"),
            units_pending=sum(1 for row in rows if row.state in {"PENDING", "NOT_STARTED"}),
            units_analyzed=sum(1 for row in rows if row.state in {"ANALYZED", "REVIEWED"}),
            units_reviewed=sum(1 for row in rows if row.state == "REVIEWED"),
            issuers_prepared=sum(
                len(row.entity_ids)
                for row in rows
                if row.state in {"PREPARED", "ANALYZED", "REVIEWED"}
            ),
            issuers_failed=sum(len(row.entity_ids) for row in rows if row.state == "FAILED"),
            issuers_analyzed=sum(
                len(row.entity_ids) for row in rows if row.state in {"ANALYZED", "REVIEWED"}
            ),
            prepared_weight=format_book_weight(prepared_weight),
            analyzed_weight=format_book_weight(analyzed_weight),
            complete=all(row.state in {"PREPARED", "ANALYZED", "REVIEWED"} for row in rows),
            units=tuple(rows),
            issuers_nothing_filed=len(nothing_filed),
            nothing_filed_weight=(
                format_book_weight(
                    math.fsum(positions[entity].ending_weight for entity in nothing_filed)
                )
                if nothing_filed
                else None
            ),
            issuers_carried=len(carried),
            carried_weight=(
                format_book_weight(math.fsum(positions[entity].ending_weight for entity in carried))
                if carried
                else None
            ),
            preparation=preparation,
        )


def _sources_cannot_cover(failed: list[_FailedUnit], *, total: int, recorded: bool) -> bool:
    """Whether every unit of the book failed for too few sources under the recorded package, so
    that the book cannot be reviewed until its sources change (V541)."""
    return recorded and len(failed) == total and all(unit.sources_short for unit in failed)


def _failed_words(
    failed: list[_FailedUnit], *, total: int, recorded: bool, standing: bool, one_unit: bool
) -> str:
    """What the book's failed units say together, and the book's standing (V541).

    A book the installed sources cannot cover (`standing`) cannot be reviewed until they change,
    and that is said with its ways on; a source shortfall is never offered a retry under the
    recorded package. A book of one unit has no progress table, so its unit's code, words and
    issuers without a source are said here.
    """
    short = [unit for unit in failed if unit.sources_short]
    if standing:
        words = (
            "This book cannot be reviewed under the installed sources: every unit failed for too "
            "few source documents, each naming the coverage it reached and the issuers without "
            "one, and preparing again under these sources fails the same way. It can be reviewed "
            "after the person consents to official SEC acquisition, which prepares every unit at "
            "one cutoff; a package covers one unit, and units prepared under different packages "
            "cannot be reviewed together (`source_ways`)."
        )
    else:
        words = " ".join(
            part
            for part in (
                f"{len(failed)} of {total} {'unit' if total == 1 else 'units'} failed to prepare, "
                "each with its code and words.",
                (
                    f"{len(short)} failed for too few source documents, which preparing again "
                    "under the installed sources does not change; the ways on are the person's "
                    "(`source_ways`)."
                    if recorded
                    else "Under official acquisition, preparing again retries the filings that "
                    "could not be obtained."
                )
                if short
                else "",
                "Preparing again retries the others and keeps every unit that completed."
                if len(short) < len(failed)
                else "",
            )
            if part
        )
    if not one_unit:
        return words
    unit = failed[0]
    without = unit.issuers_without_source
    return f"{words} The unit failed with `{unit.failure_code}`: {unit.detail}" + (
        " Without a source: " + ", ".join(without) + "." if without else ""
    )


def _preparation_work(
    adapter: AlternativeEvidenceDocumentTaskAdapter,
    task: TaskRecord,
    run: AlternativeEvidenceCoverageRun,
    states: dict[tuple[str, ...], dict[str, object]],
) -> tuple[dict[str, EvidenceCroStageWork], tuple[EvidenceCroStageWork, ...]]:
    """What the running Task's stages report (section 10.10): each unit's current stage,
    and the book's four counts -- filings against the run's source counts, units against
    the run's units. Telemetry of this process; a restarted Host reports none."""

    def shown(work: StageWork) -> EvidenceCroStageWork:
        return EvidenceCroStageWork(
            stage=work.stage,
            unit_name=work.unit_name,
            completed=work.completed,
            total=work.total,
            running=work.running,
            updated_at=work.updated_at.isoformat(),
        )

    reported = adapter.progress.units(task.task_id)
    counts = dict(run.source_counts)
    book = adapter.progress.book(
        task.task_id,
        planned_filings=sum(
            counts.get(entity, 0) for unit in run.units for entity in unit.ordered_entity_ids
        ),
        units_total=len(run.units),
        units_selected=sum(1 for value in states.values() if value["state"] == "PREPARED"),
    )
    latest = max((work.updated_at for work in reported.values()), default=None)
    return (
        {unit: shown(work) for unit, work in reported.items() if unit is not None},
        tuple(
            EvidenceCroStageWork(
                stage=stage,
                unit_name=unit_name,
                completed=completed,
                total=total,
                running=completed < total,
                updated_at=None if latest is None else latest.isoformat(),
            )
            for stage, unit_name, completed, total in book
        ),
    )
