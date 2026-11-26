"""Official and recorded source acquisition for document intelligence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from ..contracts import (
    ADMITTED_DOCUMENT_CAPACITY,
    AlternativeEvidenceAcquisitionAccounting,
    AlternativeEvidenceAdmission,
    AlternativeEvidenceCitation,
    AlternativeEvidenceClass,
    AlternativeEvidenceDocumentAcquisition,
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSnapshot,
    AlternativeEvidenceSnapshotStatus,
    SecCompanyFactsSnapshot,
    SecIssuerRegistrySnapshot,
    seal_contract,
    snapshot_expiry,
)
from ..publication.artifacts import AlternativeEvidenceArtifactStore
from ..runtime.identity import (
    alternative_acquisition_binding_hash,
    alternative_evidence_playpen_root,
)
from .contracts import (
    AcquiredEvidenceDocument,
    SecFilingInventoryRead,
    SecFilingSelectionPlan,
    inventory_read_key,
)
from .recorded import RecordedEvidenceDocument, RecordedEvidenceSource
from .sec_edgar import (
    AlternativeEvidenceCommitRefused,
    LocalSourceLookup,
    SecEdgarSource,
    SourceDocumentCommit,
    SourceDocumentDeferral,
    StoragePreflight,
    apply_unit_capacity,
)


class AlternativeEvidenceAcquisitionError(ValueError):
    """Reject an invalid source acquisition request or result."""

    pass


class AlternativeEvidenceTaskCancelled(RuntimeError):
    """Stop acquisition when the owning task is cancelled."""

    pass


class AlternativeEvidenceAcquisitionService:
    """Acquire and freeze source evidence without review or current authority."""

    def __init__(
        self,
        *,
        artifact_root: Path,
    ) -> None:
        """Bind the artifact store and recorded source resolver."""
        playpen_root = alternative_evidence_playpen_root(Path(__file__))
        self.acquisition_binding_hash = alternative_acquisition_binding_hash(playpen_root)
        self.artifacts = AlternativeEvidenceArtifactStore(artifact_root)
        self.recorded = RecordedEvidenceSource()

    def admit(
        self,
        *,
        request: AlternativeEvidenceRequest,
        admission: AlternativeEvidenceAdmission,
        now: datetime,
    ) -> None:
        """Validate request authority, deadline, mode, and source classes."""
        if admission.request_hash != request.request_hash:
            raise AlternativeEvidenceAcquisitionError(
                "alternative_evidence.request_authority_mismatch"
            )
        if now > request.acquisition_deadline:
            raise AlternativeEvidenceAcquisitionError(
                "alternative_evidence.acquisition_deadline_exceeded"
            )
        live = request.mode is AlternativeEvidenceMode.LIVE_OFFICIAL
        if live != (admission.network_consent and admission.admit_live_official):
            raise AlternativeEvidenceAcquisitionError("alternative_evidence.live_admission_invalid")
        if live and not set(request.evidence_classes).issubset(
            {
                AlternativeEvidenceClass.SEC_FILING,
                AlternativeEvidenceClass.SEC_COMPANYFACTS,
            }
        ):
            raise AlternativeEvidenceAcquisitionError(
                "alternative_evidence.live_evidence_class_not_supported"
            )

    def build_recorded_evidence(
        self,
        *,
        request: AlternativeEvidenceRequest,
        registry: SecIssuerRegistrySnapshot,
        documents: tuple[RecordedEvidenceDocument, ...],
        published_at: datetime,
    ) -> tuple[AlternativeEvidenceSnapshot, tuple[AcquiredEvidenceDocument, ...]]:
        """Freeze selected recorded documents and publish their citations."""
        if request.mode is not AlternativeEvidenceMode.RECORDED:
            raise AlternativeEvidenceAcquisitionError("alternative_evidence.recorded_mode_required")
        acquired = self.recorded.acquire_documents(request=request, documents=documents)
        citations = self.recorded.resolve(request=request, documents=documents)
        _selected, deferrals = self.recorded.select_documents(request=request, documents=documents)
        snapshot = self._snapshot(
            request=request,
            registry=registry,
            citations=citations,
            failed_entities=(),
            published_at=published_at,
            additional_limitations=(
                "Recorded issuer material uses actual local captured and available time.",
                *deferrals,
            ),
        )
        self._publish_source_evidence(request=request, registry=registry, snapshot=snapshot)
        return snapshot, acquired

    def _inventory_read(
        self, request: AlternativeEvidenceRequest, entity_id: str
    ) -> SecFilingInventoryRead | None:
        """Load a prior cutoff-valid filing inventory read, when present.

        The index read at this request's cutoff under its window and budget,
        if the packing read one.
        """
        key = inventory_read_key(
            entity_id=entity_id,
            evidence_as_of=request.evidence_as_of,
            event_window_days=request.source_policy.sec_recent_8k_days,
            policy_budget=request.source_policy.maximum_documents_per_issuer,
            unit_capacity=ADMITTED_DOCUMENT_CAPACITY,
        )
        if not self.artifacts.exists("sec-inventory-reads", key):
            return None
        return self.artifacts.load("sec-inventory-reads", key, SecFilingInventoryRead)

    def build_live_evidence(
        self,
        *,
        request: AlternativeEvidenceRequest,
        admission: AlternativeEvidenceAdmission,
        source: SecEdgarSource,
        published_at: datetime,
        prior_companyfacts: tuple[SecCompanyFactsSnapshot, ...] = (),
        registry: SecIssuerRegistrySnapshot | None = None,
        should_cancel: Callable[[], bool] = lambda: False,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        accession_scopes: Mapping[str, frozenset[str]] | None = None,
        local: LocalSourceLookup | None = None,
        commit: SourceDocumentCommit | None = None,
        preflight: StoragePreflight | None = None,
        defer: SourceDocumentDeferral | None = None,
        excerpt_binding_hash: str | None = None,
        known_excerpt: Callable[[AcquiredEvidenceDocument], str | None] | None = None,
    ) -> tuple[
        SecIssuerRegistrySnapshot,
        AlternativeEvidenceSnapshot,
        tuple[AcquiredEvidenceDocument, ...],
    ]:
        """Acquire the request's selected resources under its admission.

        For each issuer the official inventory is read and the selection
        planned -- the issuer's own selection under its policy budget --
        then the unit's plans are bounded together by the admitted
        document set (`apply_unit_capacity`: event filings deferred by
        name when the selections together exceed it, never a division of
        the set among the issuers); each planned resource is then reused
        from a verified local body when the caller supplies a lookup (no
        request), or fetched and, when the caller supplies a commit, kept
        durably before the next. A
        resource that fails or is deferred is named in the snapshot's
        accounting and limitations; its issuer keeps what it did obtain, and
        a required baseline it lacks fails the issuer for this request --
        its obtained bodies stay committed for the next attempt, which
        fetches only what is still missing. Cancellation is checked between
        resources: committed bodies stay, an uncommitted transfer is repeated
        on a later attempt. A refused durable keep stops the request under
        the storage owner's own code; it is never an issuer's failure.
        """
        self.admit(request=request, admission=admission, now=published_at)
        if accession_scopes and not set(accession_scopes) <= set(request.ordered_entity_ids):
            raise AlternativeEvidenceAcquisitionError("alternative_evidence.issuer_not_requested")
        if request.mode is not AlternativeEvidenceMode.LIVE_OFFICIAL:
            raise AlternativeEvidenceAcquisitionError("alternative_evidence.live_mode_required")
        registry = registry or source.acquire_registry(captured_at=published_at)
        citations: list[AlternativeEvidenceCitation] = []
        acquired_documents: list[AcquiredEvidenceDocument] = []
        failed: list[str] = []
        prior_by_entity = {value.entity_id: value for value in prior_companyfacts}
        companyfacts_limitations: list[str] = []
        selection_limitations: list[str] = []
        outcomes: list[AlternativeEvidenceDocumentAcquisition] = []
        inventory_requests_before = source.network_call_count
        body_requests_before = source.body_request_count

        def check_cancel() -> None:
            if should_cancel():
                raise AlternativeEvidenceTaskCancelled(
                    "alternative_evidence.current_task_cancelled"
                )

        # Every issuer's own plan first (one inventory request each), then
        # the unit's plans under one admitted document set; a plan that
        # cannot be made fails its issuer here, by name, and the others go on.
        plans: dict[str, SecFilingSelectionPlan] = {}
        planning_failures: dict[str, str] = {}
        ciks = {value.entity_id: value.cik for value in registry.entries}
        if AlternativeEvidenceClass.SEC_FILING in request.evidence_classes:
            for entity_id in request.ordered_entity_ids:
                if clock() > request.acquisition_deadline:
                    raise AlternativeEvidenceAcquisitionError(
                        "alternative_evidence.acquisition_deadline_exceeded"
                    )
                check_cancel()
                scope = None if accession_scopes is None else accession_scopes.get(entity_id)
                read = None if scope is not None else self._inventory_read(request, entity_id)
                if (
                    read is not None
                    and read.plan.cik == ciks.get(entity_id)
                    and frozenset(
                        value.accession
                        for value in read.plan.deferred
                        if value.reason == "READ_EARLIER"
                    )
                    == request.read_accessions(entity_id)
                ):
                    # The index this cutoff's packing read, passing over what
                    # this request names as read earlier: the same plan, no
                    # second request for it.
                    plans[entity_id] = read.plan
                    continue
                try:
                    plans[entity_id] = source.plan_entity_filings(
                        request=request,
                        registry=registry,
                        entity_id=entity_id,
                        accession_scope=scope,
                    )
                except (AlternativeEvidenceTaskCancelled, AlternativeEvidenceCommitRefused):
                    raise
                except Exception as error:  # every source failure is named, never swallowed
                    planning_failures[entity_id] = f"{type(error).__name__}: {str(error)[:200]}"
            bounded = apply_unit_capacity(
                tuple(plans[e] for e in request.ordered_entity_ids if e in plans)
            )
            plans = {plan.entity_id: plan for plan in bounded}
        for entity_id in request.ordered_entity_ids:
            if clock() > request.acquisition_deadline:
                raise AlternativeEvidenceAcquisitionError(
                    "alternative_evidence.acquisition_deadline_exceeded"
                )
            if should_cancel():
                raise AlternativeEvidenceTaskCancelled(
                    "alternative_evidence.current_task_cancelled"
                )
            if entity_id in planning_failures:
                failed.append(entity_id)
                selection_limitations.append(
                    f"{entity_id}: filing acquisition failed -- {planning_failures[entity_id]}; "
                    "no plan could be made, nothing is part of this request's set."
                )
            elif AlternativeEvidenceClass.SEC_FILING in request.evidence_classes:
                try:
                    entity_outcomes: list[AlternativeEvidenceDocumentAcquisition] = []
                    plan, entity_documents = source.acquire_entity_selection(
                        request=request,
                        registry=registry,
                        entity_id=entity_id,
                        handle_start=len(acquired_documents) + 1,
                        accession_scope=(
                            None if accession_scopes is None else accession_scopes.get(entity_id)
                        ),
                        local=local,
                        commit=commit,
                        check_cancel=check_cancel,
                        outcomes=entity_outcomes,
                        preflight=preflight,
                        defer=defer,
                        plan=plans[entity_id],
                    )
                    outcomes.extend(entity_outcomes)
                    # The plan is the record of what was discovered, selected,
                    # deferred and why; sealed beside the snapshot and named
                    # in its limitations so a reader can find it by hash.
                    self.artifacts.publish("sec-selection-plans", plan.plan_hash, plan)
                    selection_limitations.extend(plan.limitations)
                    reused = sum(1 for value in entity_outcomes if value.outcome == "REUSED_LOCAL")
                    fetched = sum(1 for value in entity_outcomes if value.outcome == "FETCHED")
                    earlier = sum(1 for value in plan.deferred if value.reason == "READ_EARLIER")
                    selection_limitations.append(
                        f"{entity_id}: SEC selection plan {plan.plan_hash[:12]} -- "
                        f"{plan.discovered_count} cutoff-valid filing(s) discovered, "
                        f"{len(plan.events)} in the {plan.event_window_days}-day window "
                        f"selected ({reused} reused from verified local bytes, "
                        f"{fetched} fetched), "
                        + (f"{earlier} read earlier, " if earlier else "")
                        + f"{len(plan.deferred) - earlier} deferred by the plan, "
                        f"{plan.outside_window_event_count} outside the window."
                    )
                    obtained = {value.revision for value in entity_documents}
                    for value in entity_outcomes:
                        if value.outcome in {"DEFERRED", "FAILED"}:
                            selection_limitations.append(
                                f"{entity_id}: {value.form} {value.accession} "
                                f"{value.outcome.lower()} -- {value.detail}"
                            )
                    if plan.selected and not obtained:
                        # The window selected filings and none could be read:
                        # the issuer's evidence is unavailable, never empty.
                        failed.append(entity_id)
                        selection_limitations.append(
                            f"{entity_id}: NO_SELECTED_FILING_OBTAINED -- none of the "
                            f"{len(plan.selected)} filing(s) selected in the window could be "
                            "obtained; nothing is part of this request's set."
                        )
                    else:
                        acquired_documents.extend(entity_documents)
                        citations.extend(
                            source.citations_from_documents(
                                request=request,
                                documents=entity_documents,
                                handle_start=len(citations) + 1,
                                excerpt_binding_hash=excerpt_binding_hash,
                                known_excerpt=known_excerpt,
                            )
                        )
                except (AlternativeEvidenceTaskCancelled, AlternativeEvidenceCommitRefused):
                    raise
                except Exception as error:  # every source failure is named, never swallowed
                    failed.append(entity_id)
                    selection_limitations.append(
                        f"{entity_id}: filing acquisition failed -- "
                        f"{type(error).__name__}: {str(error)[:200]}; bodies committed "
                        "before the failure are retained for a later attempt, none is "
                        "part of this request's set."
                    )
            if AlternativeEvidenceClass.SEC_COMPANYFACTS in request.evidence_classes:
                prior = prior_by_entity.get(entity_id)
                if prior is not None:
                    try:
                        citations.append(
                            source.companyfacts_citation(
                                request=request,
                                snapshot=prior,
                                semantic_handle=f"CIT-{entity_id}-{len(citations) + 1:03d}",
                            )
                        )
                    except ValueError:
                        companyfacts_limitations.append(
                            f"{entity_id}: prior companyfacts snapshot was not usable at cutoff."
                        )
                try:
                    captured = source.acquire_companyfacts_snapshot(
                        request=request,
                        registry=registry,
                        entity_id=entity_id,
                    )
                    self.artifacts.publish(
                        "companyfacts-snapshots", captured.companyfacts_hash, captured
                    )
                except Exception:
                    companyfacts_limitations.append(
                        f"{entity_id}: new companyfacts snapshot acquisition failed."
                    )
            if should_cancel():
                raise AlternativeEvidenceTaskCancelled(
                    "alternative_evidence.current_task_cancelled"
                )
            if clock() > request.acquisition_deadline:
                raise AlternativeEvidenceAcquisitionError(
                    "alternative_evidence.acquisition_deadline_exceeded"
                )
        available_entities = {value.entity_id for value in citations}
        accounting = AlternativeEvidenceAcquisitionAccounting(
            inventory_request_count=(
                source.network_call_count
                - inventory_requests_before
                - (source.body_request_count - body_requests_before)
            ),
            body_request_count=source.body_request_count - body_requests_before,
            fetched_count=sum(1 for value in outcomes if value.outcome == "FETCHED"),
            reused_local_count=sum(1 for value in outcomes if value.outcome == "REUSED_LOCAL"),
            deferred_count=sum(1 for value in outcomes if value.outcome == "DEFERRED"),
            failed_count=sum(1 for value in outcomes if value.outcome == "FAILED"),
            fetched_bytes=sum(
                value.content_bytes for value in outcomes if value.outcome == "FETCHED"
            ),
            reused_bytes=sum(
                value.content_bytes for value in outcomes if value.outcome == "REUSED_LOCAL"
            ),
            documents=tuple(outcomes),
        )
        snapshot = self._snapshot(
            request=request,
            registry=registry,
            citations=tuple(citations),
            # A filing request may fail while a prior, cutoff-valid companyfacts snapshot
            # still supplies evidence for that issuer.  Source accounting is per issuer,
            # so a partially successful issuer cannot be both available and failed.
            failed_entities=tuple(value for value in failed if value not in available_entities),
            published_at=published_at,
            additional_limitations=(
                "Automatic LIVE acquisition is restricted to SEC EDGAR official endpoints.",
                "Mutable companyfacts is excluded unless a prior as-of snapshot exists.",
                *selection_limitations,
                *companyfacts_limitations,
            ),
            acquisition=accounting,
        )
        self._publish_source_evidence(request=request, registry=registry, snapshot=snapshot)
        return registry, snapshot, tuple(acquired_documents)

    def _snapshot(
        self,
        *,
        request: AlternativeEvidenceRequest,
        registry: SecIssuerRegistrySnapshot,
        citations: tuple[AlternativeEvidenceCitation, ...],
        failed_entities: tuple[str, ...],
        published_at: datetime,
        additional_limitations: tuple[str, ...],
        acquisition: AlternativeEvidenceAcquisitionAccounting | None = None,
    ) -> AlternativeEvidenceSnapshot:
        if published_at > request.acquisition_deadline:
            raise AlternativeEvidenceAcquisitionError(
                "alternative_evidence.acquisition_deadline_exceeded"
            )
        available_entities = {value.entity_id for value in citations}
        requested = set(request.ordered_entity_ids)
        if not available_entities.issubset(requested) or not set(failed_entities).issubset(
            requested
        ):
            raise AlternativeEvidenceAcquisitionError(
                "alternative_evidence.snapshot_entity_axis_invalid"
            )
        missing_entities = requested - available_entities - set(failed_entities)
        expected = len(request.ordered_entity_ids)
        available = len(available_entities)
        failed = len(set(failed_entities))
        missing = len(missing_entities)
        if available == expected:
            status = AlternativeEvidenceSnapshotStatus.COMPLETE
        elif available:
            status = AlternativeEvidenceSnapshotStatus.PARTIAL
        elif failed:
            status = AlternativeEvidenceSnapshotStatus.UNAVAILABLE
        else:
            status = AlternativeEvidenceSnapshotStatus.EMPTY
        limitations = tuple(
            dict.fromkeys(
                (
                    *additional_limitations,
                    *(value for item in citations for value in item.limitations),
                )
            )
        )
        return seal_contract(
            AlternativeEvidenceSnapshot,
            "snapshot_hash",
            request_hash=request.request_hash,
            registry_hash=registry.registry_hash,
            acquisition_binding_hash=self.acquisition_binding_hash,
            status=status,
            expected_source_count=expected,
            available_source_count=available,
            missing_source_count=missing,
            failed_source_count=failed,
            citations=citations,
            limitations=limitations,
            published_at=published_at,
            expires_at=snapshot_expiry(request, published_at),
            acquisition=acquisition,
        )

    def _publish_source_evidence(
        self,
        *,
        request: AlternativeEvidenceRequest,
        registry: SecIssuerRegistrySnapshot,
        snapshot: AlternativeEvidenceSnapshot,
    ) -> None:
        self.artifacts.publish("requests", request.request_hash, request)
        self.artifacts.publish("registries", registry.registry_hash, registry)
        self.artifacts.publish("snapshots", snapshot.snapshot_hash, snapshot)


__all__ = [
    "AlternativeEvidenceAcquisitionError",
    "AlternativeEvidenceAcquisitionService",
    "AlternativeEvidenceTaskCancelled",
]
