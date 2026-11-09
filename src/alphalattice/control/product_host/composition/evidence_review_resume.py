"""A book's review resumed by a retry (idempotency keys), never prepared again beside itself.

The Evidence review application's: its issuer scope binds the sealed book, so a retry names the
same attempt. While a current analysis answers a unit of the newest preparation, the preview keeps
that cutoff and offers an Analyst only an unanswered unit: no second analysis for the agent to pick.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Mapping
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, cast
from uuid import UUID

from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
)
from alphalattice.evidence.alternative_evidence.runtime.coverage import (
    COVERAGE_RUN_PURPOSE,
    sources_short,
)
from alphalattice.evidence.alternative_evidence.runtime.reuse import analysis_records
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    PortfolioEvidenceReviewError,
)

if TYPE_CHECKING:
    from alphalattice.control.product_host.composition.application_session import (
        WorkspaceApplicationSession,
    )
    from alphalattice.control.task_control.contracts import TaskRecord
    from alphalattice.evidence.alternative_evidence.publication.analysis import (
        AlternativeEvidenceAnalysisPublicationService,
    )
    from alphalattice.evidence.alternative_evidence.publication.artifacts import (
        AlternativeEvidenceArtifactStore,
    )
    from alphalattice.evidence.alternative_evidence.publication.contracts import (
        AlternativeEvidenceAnalysisPublication,
    )
    from alphalattice.evidence.alternative_evidence.runtime.coverage import (
        AlternativeEvidenceCoverageRun,
    )
    from alphalattice.evidence.alternative_evidence.runtime.policy import AdmittedEvidencePolicy
    from alphalattice.evidence.alternative_evidence.runtime.reuse import RecordDays
    from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
        AlternativeEvidenceDocumentTaskAdapter,
    )
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector
    from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
        PortfolioIssuerScope,
    )

CURRENT_WITHIN = timedelta(seconds=604_800)
"""The longest analysis TTL a request admits (`AlternativeEvidenceRequest.ttl_seconds`):
an analysis is published at or after its cutoff, so one published before now less this
has expired, and a reader of current analyses reads no older record (Z2)."""


class _Review(Protocol):
    artifacts: AlternativeEvidenceArtifactStore
    evidence_publications: AlternativeEvidenceAnalysisPublicationService | None
    evidence_policy: AdmittedEvidencePolicy
    evidence_task_adapter: AlternativeEvidenceDocumentTaskAdapter | None

    @property
    def clock(self) -> Callable[[], datetime]: ...

    session: WorkspaceApplicationSession

    def packet_requests(
        self, selector: BookSelector | None, *, task_id: UUID
    ) -> dict[str, dict[str, str]]: ...

    def _publication_days(self) -> RecordDays | None: ...

    def _sealed_run(
        self, scope: PortfolioIssuerScope, *, evidence_as_of: datetime | None
    ) -> AlternativeEvidenceCoverageRun | None: ...

    def _run_tasks(
        self, scope: PortfolioIssuerScope, *, evidence_as_of: datetime | None = None
    ) -> Iterator[TaskRecord]: ...

    def _active_refresh_task(self, *, scope: PortfolioIssuerScope) -> TaskRecord | None: ...


def current_answers(
    app: _Review, obligation_hashes: Collection[str], *, now: datetime
) -> list[AlternativeEvidenceAnalysisPublication]:
    """Every current analysis answering one of these obligations, the records read once."""
    service = app.evidence_publications
    if service is None:
        return []
    found = []
    for value in analysis_records(app.artifacts, app._publication_days(), now - CURRENT_WITHIN):
        if (
            value.obligation_hash not in obligation_hashes
            or service.standing(value, now=now) != "CURRENT"
        ):
            continue
        request = app.artifacts.load("requests", value.request_hash, AlternativeEvidenceRequest)
        if app.evidence_policy.matches(request):
            found.append(value)
    return found


def answered_units(app: _Review, run: AlternativeEvidenceCoverageRun) -> frozenset[str]:
    """The run's units a current analysis already answers, by unit id."""
    owed = {unit.obligation.obligation_hash: unit.unit_id for unit in run.units}
    return frozenset(
        owed[value.obligation_hash] for value in current_answers(app, owed, now=app.clock())
    )


def same_preparation(
    run: AlternativeEvidenceCoverageRun, *, prepare_only: bool
) -> Callable[[TaskRecord], bool]:
    """The rule Task Control asks at admission: one book's preparation under one package."""

    def same(task: TaskRecord) -> bool:
        payload = task.input.payload
        return (
            payload.get("purpose") == COVERAGE_RUN_PURPOSE
            and payload.get("prepare_only") is prepare_only
            and payload.get("scope_hash") == run.scope_hash
            and payload.get("resource_binding_hash") == run.resource_binding_hash
        )

    return same


def preparation_in_flight(
    app: _Review, scope: PortfolioIssuerScope, *, prepare_only: bool
) -> tuple[TaskRecord, AlternativeEvidenceCoverageRun] | None:
    """Find the one preparation in flight for this book's review, if any.

    A book has one preparation in flight, whatever cutoff a later request names: a second run
    beside it re-reads the same filing indices and packs the same positions into other units, so a
    repeated review loses the coverage the first was gathering. The caller joins the Task queued,
    running or owed its recovery, at that Task's cutoff. Another package, another purpose, or a
    single preparation queued before every book was a run is another question, and the existing
    queue admits it or refuses it as busy. A completed preparation for the exact run stays
    `REUSED_EXACT`: each caller asks this after that reuse.
    """
    active, adapter = app._active_refresh_task(scope=scope), app.evidence_task_adapter
    if active is None or adapter is None:
        return None
    run = adapter.run_of(active)
    if (
        run is None
        or active.input.payload.get("prepare_only") is not prepare_only
        or run.resource_binding_hash != adapter.run_binding_hash(prepare_only=prepare_only)
    ):
        return None
    return active, run


def review_run(
    app: _Review, scope: PortfolioIssuerScope, newest: AlternativeEvidenceCoverageRun
) -> AlternativeEvidenceCoverageRun:
    """Choose the run a book's review stands on.

    The newest, whenever it accounts for any issuer: a run that carried or answered some says
    what the new filings left unread. Only a newest run that accounts for none gives way, to a
    run of this same scope and package from the last analysis TTL, packed otherwise, that accounts
    for most. Analyses answer a run's units, so a later run that packed the same positions
    otherwise and was never answered would leave every analyzed unit out of the review. A later
    run packed the same is that review's next preparation: it stands, and reuses what it can.
    """
    adapter = app.evidence_task_adapter
    if adapter is None or newest.carried or newest.nothing_filed:
        return newest
    runs = [newest]

    def packing(run: AlternativeEvidenceCoverageRun) -> set[tuple[str, ...]]:
        return {tuple(unit.ordered_entity_ids) for unit in run.units}

    for task in app._run_tasks(scope):
        run = adapter.run_of(task)
        if (
            run is not None
            and run.run_hash not in {value.run_hash for value in runs}
            and packing(run) != packing(newest)
            and run.resource_binding_hash == newest.resource_binding_hash
            and run.evidence_as_of >= app.clock() - CURRENT_WITHIN
        ):
            runs.append(run)
    if len(runs) == 1:
        return newest
    owed = {unit.obligation.obligation_hash for run in runs for unit in run.units}
    answered = {value.obligation_hash for value in current_answers(app, owed, now=app.clock())}

    def covered(run: AlternativeEvidenceCoverageRun) -> int:
        held = {*run.carried, *run.nothing_filed}
        for unit in run.units:
            if unit.obligation.obligation_hash in answered:
                held.update(unit.ordered_entity_ids)
        return len(held)

    return newest if covered(newest) else max(runs, key=covered)


def resumed(
    app: _Review, scope: PortfolioIssuerScope
) -> tuple[AlternativeEvidenceCoverageRun, frozenset[str]] | None:
    """The scope's newest preparation its package still binds, with its answered units if any."""
    run, adapter = app._sealed_run(scope, evidence_as_of=None), app.evidence_task_adapter
    if run is None or adapter is None:
        return None
    if run.resource_binding_hash != adapter.run_binding_hash(prepare_only=True):
        return None
    answered = answered_units(app, run)
    return (run, answered) if answered else None


def units_short_of_sources(
    inventory: Mapping[str, object], mode: AlternativeEvidenceMode, *, floor: float
) -> list[dict[str, object]]:
    """The units RECORDED mode cannot prepare, each with the code it would fail.

    A packet reads only the documents the workspace holds. A unit whose issuers holding one fall
    short of the installed floor fails `alternative_evidence.minimum_entity_coverage_not_met`
    at its acquisition, the same rule its refusal applies (`sources_short`); one whose issuers
    hold none under a floor that admits that fails `alternative_evidence.document_set_empty`.
    Under official acquisition the filing index is read at the run, so none is named.

    Args:
        inventory: The preview's source inventory: each issuer, its unit and its documents.
        mode: The Host's evidence mode.
        floor: The installed package's minimum issuer coverage.

    Returns:
        Each such unit in the run's order, heaviest first: its id, its code and its issuers
        without a source; empty outside RECORDED mode.
    """
    if mode is not AlternativeEvidenceMode.RECORDED:
        return []
    units: dict[str, list[tuple[str, int]]] = {}
    for issuer in cast(list[dict[str, object]], inventory.get("issuers") or []):
        units.setdefault(str(issuer.get("unit_id")), []).append(
            (str(issuer.get("entity_id")), int(cast(int, issuer.get("documents") or 0)))
        )
    named: list[dict[str, object]] = []
    for unit_id, issuers in units.items():
        entities = tuple(entity for entity, _documents in issuers)
        sourced = {entity for entity, documents in issuers if documents}
        short = sources_short(entities, sourced, floor=floor)
        if short is not None or not sourced:
            named.append(
                {
                    "unit_id": unit_id,
                    "failure_code": "alternative_evidence.document_set_empty"
                    if short is None
                    else str(short),
                    "issuers_without_source": list(entities if short is None else short.uncovered),
                }
            )
    return named


def unanswered(
    requests: dict[str, dict[str, str]], answered: Collection[str]
) -> dict[str, dict[str, str]]:
    """These offers less each unit's packet (`packet_<unit>`) a current analysis answers."""
    return {
        name: value
        for name, value in requests.items()
        if not (name.startswith("packet_") and name.removeprefix("packet_") in answered)
    }


def _matching_task(
    task: TaskRecord,
    adapter: AlternativeEvidenceDocumentTaskAdapter,
    scope: PortfolioIssuerScope,
    subjects: Callable[[UUID], tuple[dict[str, Any], ...]],
    matches: Callable[[dict[str, Any]], bool],
) -> bool:
    active = {
        TaskLifecycle.QUEUED,
        TaskLifecycle.RUNNING,
        TaskLifecycle.RECOVERY_REQUIRED,
        TaskLifecycle.SUCCEEDED,
    }
    return (
        task.task_kind == adapter.task_kind
        and task.input.payload.get("purpose") == COVERAGE_RUN_PURPOSE
        and task.input.payload.get("prepare_only") is True
        and task.lifecycle in active
        and task.input.payload.get("scope_hash") == scope.scope_hash
        and task.input.payload.get("resource_binding_hash")
        == adapter.run_binding_hash(prepare_only=True)
        and any(matches(subject) for subject in subjects(task.task_id))
    )


def _current_run(
    run: AlternativeEvidenceCoverageRun,
    scope: PortfolioIssuerScope,
    adapter: AlternativeEvidenceDocumentTaskAdapter,
    policy: AdmittedEvidencePolicy,
    now: datetime,
) -> bool:
    return (
        run.evidence_as_of <= now
        and run.scope_hash == scope.scope_hash
        and run.resource_binding_hash == adapter.run_binding_hash(prepare_only=True)
        and (run.network_consent, run.admit_live_official, run.admit_model_review)
        == (policy.network_consent, policy.admit_live_official, False)
        and all(policy.matches(unit.request) for unit in run.units)
    )


def _usable_packets(
    task: TaskRecord,
    run: AlternativeEvidenceCoverageRun,
    adapter: AlternativeEvidenceDocumentTaskAdapter,
    policy: AdmittedEvidencePolicy,
    now: datetime,
) -> bool:
    if task.lifecycle is not TaskLifecycle.SUCCEEDED:
        return True
    states = adapter.unit_states(task)
    if any(value["state"] not in {"PREPARED", "FAILED"} for value in states.values()):
        return False
    prepared = [unit for unit in run.units if states[unit.unit_id]["state"] == "PREPARED"]
    if not prepared and now > run.evidence_as_of + timedelta(seconds=policy.ttl_seconds):
        return False
    try:
        for unit in prepared:
            adapter.prepared_packet(task.task_id, now=now, unit_id=unit.unit_id)
    except ValueError as error:
        if str(error) not in {
            "alternative_evidence.brief_source_stale",
            "alternative_evidence.preparation_superseded",
        }:
            raise
        return False
    return True


def resumable_preparation(
    app: _Review,
    scope: PortfolioIssuerScope,
    *,
    now: datetime,
    matches: Callable[[dict[str, Any]], bool],
) -> tuple[TaskRecord, AlternativeEvidenceCoverageRun] | None:
    """Select this publication's still-bound preparation without hiding failed units."""
    adapter = app.evidence_task_adapter
    if adapter is None:
        return None
    registry, policy = app.session.task_control_registry, app.evidence_policy
    tasks = sorted(
        registry.tasks(), key=lambda task: (task.updated_at, str(task.task_id)), reverse=True
    )
    for task in tasks:
        if not _matching_task(task, adapter, scope, registry.admission_subjects, matches):
            continue
        try:
            run = adapter.run_of(task)
        except OSError as error:
            raise PortfolioEvidenceReviewError(
                "alternative_evidence.artifact_unavailable"
            ) from error
        if run is None:
            raise PortfolioEvidenceReviewError("alternative_evidence.artifact_unavailable")
        if _current_run(run, scope, adapter, policy, now) and _usable_packets(
            task, run, adapter, policy, now
        ):
            return task, run
    return None


def preview_units(
    adapter: AlternativeEvidenceDocumentTaskAdapter,
    run: AlternativeEvidenceCoverageRun,
    selector: dict[str, str],
    offers: dict[str, dict[str, str]],
    *,
    now: datetime,
) -> tuple[list[dict[str, object]], int]:
    """Project the fresh preview's units and their current exact matches."""
    units: list[dict[str, object]] = []
    count = 0
    for unit in run.units:
        found = adapter.completed_unit(
            unit.preparation_intent_hash, now=now, evidence_as_of=unit.request.evidence_as_of
        )
        if found is not None:
            count += 1
            offers[f"packet_{unit.unit_id}"] = {
                "operation": "EVIDENCE_PACKET",
                **selector,
                "task_id": str(found[0]),
                **({} if found[1] is None else {"evidence_unit_id": found[1]}),
            }
        units.append(
            {
                "unit_id": unit.unit_id,
                "ordered_entity_ids": list(unit.ordered_entity_ids),
                "request_hash": unit.request.request_hash,
                "obligation_hash": unit.obligation.obligation_hash,
                "preparation_intent_hash": unit.preparation_intent_hash,
                "prepared_task_id": None if found is None else str(found[0]),
                "prepared_unit_id": None if found is None else found[1],
                "source_check": None
                if found is None
                else adapter.prepared_source_check(str(found[0]), found[1]),
            }
        )
    return units, count


def resume_offers(
    app: _Review,
    selector: BookSelector,
    resumable: tuple[TaskRecord, AlternativeEvidenceCoverageRun] | None,
) -> dict[str, dict[str, str]]:
    """Offer only prepared units of the selected completed Task."""
    if (
        resumable is None
        or app.evidence_task_adapter is None
        or resumable[0].lifecycle is not TaskLifecycle.SUCCEEDED
    ):
        return {}
    states = app.evidence_task_adapter.unit_states(resumable[0])
    return {
        name: offer
        for name, offer in app.packet_requests(selector, task_id=resumable[0].task_id).items()
        if name.startswith("analyst_bundle")
        and offer.get("evidence_unit_id")
        and states[offer["evidence_unit_id"]]["state"] == "PREPARED"
    }


def preview_binding(
    run: AlternativeEvidenceCoverageRun, scope: PortfolioIssuerScope
) -> dict[str, object]:
    """The preparation identity explained alongside a preview."""
    return {
        "run_hash": run.run_hash,
        "issuer_scope_hash": scope.scope_hash,
        "resource_binding_hash": run.resource_binding_hash,
        "unit_count": len(run.units),
        "unit_limit": run.unit_limit,
        "packing_rules_id": run.packing_rules_id,
    }
