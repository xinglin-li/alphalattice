"""The Evidence & CRO route, driven through the owner the Local Web session holds.

Every test goes through `EvidenceReviewApplication` over one real Portfolio
development result published to a real ledger: book -> scope -> obligation ->
Alternative Evidence Task -> dossier -> review Task -> publication -> projection.
No handoff is needed any more: a development result is a reviewable book, and
the same evidence serves a validated handoff over the same report.

Recorded documents, a citing automation actor and Submitted review actors. No
Provider, no network, no protected or post-2024-08-12 evidence.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from alphalattice.control.product_host.composition.evidence_review_application import (
    EvidenceReviewApplication,
    EvidenceSelectionAmbiguous,
    ResolvedEvidenceQuestion,
)
from alphalattice.control.product_host.composition.evidence_review_delivery import (
    EvidenceReviewDelivery,
)
from alphalattice.control.product_host.composition.evidence_review_projection import (
    EvidenceCroProjector,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    EvidenceStructureState,
)
from alphalattice.evidence.alternative_evidence.publication.contracts import (
    AlternativeEvidenceAnalysisPublication,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    AlternativeEvidenceDocumentTaskAdapter,
    AlternativeEvidenceDocumentTaskResources,
    alternative_evidence_document_task_contract,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument
from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher
from alphalattice.interface.local_application.evidence_cro import (
    alternative_evidence_ready_for_review,
    evidence_cro_body,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioResearchSpec,
)
from alphalattice.investment.portfolio_strategy_lab.application.finalization import (
    ValidatedPortfolioHandoff,
)
from alphalattice.investment.portfolio_strategy_lab.publication.finalization_ledger import (
    PortfolioFinalizationStore,
)
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    BookSelector,
    PortfolioEvidenceReviewError,
    project_unit_obligation,
)
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    CRORiskSeverity,
    PortfolioReviewPublication,
    PortfolioReviewRoute,
    RequiredActionKind,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    BookAuthority,
    ExposureBand,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.coverage.selections import (
    recorded_selection,
)
from alphalattice.oversight.chief_risk_officer.publication.portfolio_review import (
    PortfolioReviewPublicationService,
)
from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
    PortfolioReviewTaskAdapter,
    SubmittedPortfolioReviewActor,
)
from alphalattice.protocols.actor_execution import ActorKind
from tests.alternative_evidence_desk.issuer_listing import (
    ISSUER_TOPICS,
    TICKERS,
    _documents,
    _listing_authority,
    _registry,
)
from tests.alternative_evidence_desk.planted_corpus import (
    _NOW,
    PLAYPEN_ROOT,
    _CitingActor,
    _runtime,
)
from tests.alternative_evidence_desk.review_dossiers import CitingReviewActor, ControlledRisk
from tests.portfolio_strategy_lab.local_web_support import _harness, _run

# ============================================================ the fixtures


def _reviewer(*risks: ControlledRisk, actor_id: str = "gate-9c5") -> SubmittedPortfolioReviewActor:
    """A controlled reviewer answering the given risks by the aliases of the
    dossier it is handed."""

    return CitingReviewActor(risks=risks, actor_kind=ActorKind.HUMAN, actor_id=actor_id)


class _Route:
    """One booted review owner over one real development result."""

    def __init__(self, harness: Any, tmp_path: Path, *, omit_listing: bool = False) -> None:
        result = _run(harness, PortfolioResearchSpec.default())
        self.harness = harness
        self.result_hash = result.result_hash
        self.report = harness.application.report(result.result_hash)
        listings = tuple(value.listing_id for value in self.report.window_end_book.positions)
        self.registry = _registry()
        self.listing_authority = _listing_authority(
            listings, omit=listings[0] if omit_listing else None
        )
        self.runtime = _runtime(tmp_path / "evidence")
        self.runtime_path = tmp_path / "checkpoints.sqlite"
        self.application = self.build_application()

    def build_application(self, **overrides: Any) -> EvidenceReviewApplication:
        values: dict[str, Any] = {
            "workspace_id": "qa-gate-9c5",
            "workspace": self.harness.workspace,
            "session": self.harness.session,
            "ledger": self.harness.application.ledger,
            "artifacts": self.runtime.artifacts,
            "evidence_publications": self.runtime.publications,
            "review_publications": PortfolioReviewPublicationService(self.runtime.artifacts),
            "playpen_root": PLAYPEN_ROOT,
            "latest_result_hash": lambda: self.result_hash,
            "finalization": PortfolioFinalizationStore(
                self.harness.workspace / "runtime" / "artifacts"
            ),
            "registry": self.registry,
            "listing_authority": self.listing_authority,
            "clock": lambda: _NOW,
            "runtime_path": self.runtime_path,
        }
        values.update(overrides)
        return EvidenceReviewApplication(**values)

    @property
    def task_registry(self) -> Any:
        return self.harness.session.task_control_registry

    def resolved(self, application: EvidenceReviewApplication | None = None) -> Any:
        application = application or self.application
        return application.resolve_book(cast(BookSelector, application.default_selector()))

    def wire_evidence(
        self,
        *,
        documents: tuple[RecordedEvidenceDocument, ...] | None = None,
        actor: Any | None = None,
    ) -> None:
        """Give the application the real evidence Task adapter over recorded documents."""

        scope = self.resolved().scope
        self.application.evidence_task_adapter = AlternativeEvidenceDocumentTaskAdapter(
            runtime=self.runtime,
            registry=self.task_registry,
            resources=AlternativeEvidenceDocumentTaskResources(
                recorded_registry=self.registry,
                recorded_documents=(
                    documents if documents is not None else _documents(scope.ordered_entity_ids)
                ),
                analysis_actor=actor or _CitingActor(topics=ISSUER_TOPICS),
            ),
        )

    def refresh(self, *, evidence_as_of: Any = None) -> Any:
        dispatcher = LocalBackgroundDispatcher(status_port=self.task_registry)
        try:
            outcome = self.application.refresh_evidence(
                dispatcher=dispatcher, evidence_as_of=evidence_as_of
            )
            dispatcher.drain_for_tests()
        finally:
            dispatcher.close()
        return outcome

    def review(
        self,
        actor: Any | None,
        *,
        drain: bool = False,
        application: EvidenceReviewApplication | None = None,
    ) -> Any:
        """Ask for one review through the dispatcher, as the HTTP route does."""

        application = application or self.application
        dispatcher = LocalBackgroundDispatcher(status_port=self.task_registry)
        try:
            outcome = application.review(dispatcher=dispatcher, actor=actor)
            if drain:
                dispatcher.drain_for_tests()
        finally:
            dispatcher.close()
        return outcome

    def current_evidence(self, application: EvidenceReviewApplication | None = None) -> Any:
        application = application or self.application
        question = _question(application)
        return application.current_evidence(question=question)


def _question(application: EvidenceReviewApplication) -> ResolvedEvidenceQuestion:
    """The one question of the harness's one-unit book at `_NOW`, composed from
    the owners the application uses (its scope, the obligation projection)."""

    resolved = application.resolve_book(cast(BookSelector, application.default_selector()))
    return ResolvedEvidenceQuestion(
        resolved=resolved,
        obligation=project_unit_obligation(
            ordered_entity_ids=resolved.scope.ordered_entity_ids,
            evidence_as_of=_NOW,
            approved_source_families=application.evidence_policy.approved_source_families,
        ),
    )


@pytest.fixture
def route(tmp_path: Path) -> Any:
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir(parents=True, exist_ok=True)
    with _harness(portfolio) as harness:
        yield _Route(harness, tmp_path)


@pytest.fixture
def gapped_route(tmp_path: Path) -> Any:
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir(parents=True, exist_ok=True)
    with _harness(portfolio) as harness:
        yield _Route(harness, tmp_path, omit_listing=True)


# ============================================================= book and scope


def test_the_latest_development_result_is_the_default_book(route: Any) -> None:
    resolved = route.resolved()

    assert resolved.book.authority is BookAuthority.DEVELOPMENT_RESULT
    assert resolved.book.result_hash == route.result_hash
    assert resolved.book.report_hash == route.report.report_hash
    assert resolved.projection.held_count == len(route.report.window_end_book.positions)
    assert set(resolved.scope.ordered_entity_ids) == set(TICKERS)
    assert resolved.scope.mapping_coverage == 1.0
    assert resolved.scope.reviewed_ending_weight_coverage == 1.0
    projection = EvidenceCroProjector(route.application).projection()
    # No evidence Task adapter is wired yet: the truthful state names the source
    # package and runtime as what is missing, not a refresh that would refuse.
    assert projection.state == "EVIDENCE_AUTHORITY_NOT_ADMITTED"
    assert "source package" in projection.explanation
    assert projection.next_requests == {} and projection.available_actions == ()
    assert projection.book is not None
    assert projection.book.authority == "DEVELOPMENT_RESULT"
    assert projection.book.result_hash == route.result_hash
    route.wire_evidence()
    projection = EvidenceCroProjector(route.application).projection()
    assert projection.state == "AWAITING_ALTERNATIVE_EVIDENCE"
    assert list(projection.next_requests) == ["preview", "refresh"]


def test_an_unmapped_listing_stays_in_the_coverage_denominator(gapped_route: Any) -> None:
    scope = gapped_route.resolved().scope

    assert len(scope.mapping_failures) == 1
    assert scope.mapping_failures[0].reason == "LISTING_NOT_IN_ADMITTED_AUTHORITY"
    assert scope.unmapped_ending_weight > 0.0
    assert scope.mapping_coverage < 1.0
    assert scope.reviewed_ending_weight_coverage < 1.0
    assert any("did not map" in value for value in scope.unavailable_reasons)


def test_a_workspace_without_evidence_authority_names_the_wait(route: Any) -> None:
    """Typed zero-work refusals, and no Task, until the owner admits the inputs."""

    tasks_before = len(route.task_registry.tasks())

    bare = route.build_application(registry=None, listing_authority=None)
    assert EvidenceCroProjector(bare).projection().state == "EVIDENCE_AUTHORITY_NOT_ADMITTED"
    assert EvidenceCroProjector(bare).projection().available_actions == ()
    reviewer = _reviewer()
    assert (
        route.review(reviewer, application=bare).disposition
        == "REFUSED_NO_ADMITTED_EVIDENCE_AUTHORITY"
    )
    dispatcher = LocalBackgroundDispatcher(status_port=route.task_registry)
    try:
        refused = bare.refresh_evidence(dispatcher=dispatcher)
    finally:
        dispatcher.close()
    assert refused.disposition == "REFUSED_NO_ADMITTED_EVIDENCE_AUTHORITY"

    empty = route.build_application(latest_result_hash=lambda: None)
    assert EvidenceCroProjector(empty).projection().state == "NO_BOOK_TO_REVIEW"
    assert route.review(reviewer, application=empty).disposition == "REFUSED_NO_BOOK_TO_REVIEW"

    # Authority admitted, but neither an evidence runtime nor a review actor.
    dispatcher = LocalBackgroundDispatcher(status_port=route.task_registry)
    try:
        no_runtime = route.application.refresh_evidence(dispatcher=dispatcher)
    finally:
        dispatcher.close()
    assert no_runtime.disposition == "REFUSED_NO_ADMITTED_EVIDENCE_RUNTIME"
    assert route.review(None).disposition == "REFUSED_NO_ADMITTED_REVIEW_ACTOR"

    assert reviewer.observed_deadlines == []
    assert len(route.task_registry.tasks()) == tasks_before


# ===================================================== the evidence half


def test_the_refresh_command_binds_the_obligation_into_the_task_input(route: Any) -> None:
    """requirement: the refresh's Task input names the book's run by hash, and
    the run's unit carries exactly the book's obligation, request and admission
    -- the question a Task answers is fixed by its input (every book is a run, C2)."""

    route.wire_evidence()
    outcome = route.refresh()
    assert outcome.disposition == "ADMITTED"
    assert outcome.task_id is not None

    question = _question(route.application)
    task = route.task_registry.task(outcome.task_id)
    adapter = route.application.evidence_task_adapter
    run = adapter.run_of(task)
    assert run is not None and task.input.payload["run_hash"] == run.run_hash
    (unit,) = run.units
    assert unit.obligation.obligation_hash == question.obligation.obligation_hash
    assert tuple(unit.request.ordered_entity_ids) == question.resolved.scope.ordered_entity_ids
    assert adapter.unit_authority(task, unit.unit_id).admission.network_consent is False
    assert task.lifecycle is TaskLifecycle.SUCCEEDED


def test_a_disagreeing_obligation_is_refused_before_a_task_exists(route: Any) -> None:
    question = _question(route.application)
    reversed_axis = question.resolved.scope.ordered_entity_ids[::-1]
    request = route.application.evidence_policy.request(
        ordered_entity_ids=reversed_axis, evidence_as_of=_NOW
    )
    admission = route.application.evidence_policy.admission(request=request, admitted_at=_NOW)
    before = len(route.task_registry.tasks())
    with pytest.raises(ValueError, match="task_obligation_axis_mismatch"):
        alternative_evidence_document_task_contract(
            request=request,
            admission=admission,
            obligation=question.obligation,
            resource_binding_hash="0" * 64,
        )
    assert len(route.task_registry.tasks()) == before


# ========================================================== the whole route


def test_the_whole_route_reaches_a_published_recommendation(route: Any) -> None:
    """Book -> scope -> obligation -> evidence Task -> dossier -> review Task -> section."""

    route.wire_evidence()
    assert route.refresh().disposition == "ADMITTED"
    evidence = route.current_evidence()
    assert evidence is not None and evidence.is_current
    handles = {value.finding_handle for value in evidence.lineage.brief.findings}
    # The Host assigns the handles in answer order; each issuer's own topic.
    assert handles == {f"FIND-{index:03d}" for index in range(1, len(ISSUER_TOPICS) + 1)}
    assert {
        (value.affected_entities, value.topic) for value in evidence.lineage.brief.findings
    } == {((entity,), topic) for entity, topic in ISSUER_TOPICS.items()}
    for handle in handles:
        structure = evidence.lineage.cro_package.structure(handle)
        assert structure.state is EvidenceStructureState.SUPPORTED, handle
        assert structure.supporting_document_count == 2

    actor = _reviewer(ControlledRisk("AAPL"))
    outcome = route.review(actor, drain=True)
    assert outcome.disposition == "ADMITTED"
    assert outcome.task_id is not None
    assert route.task_registry.task(outcome.task_id).lifecycle is TaskLifecycle.SUCCEEDED
    assert len(actor.observed_deadlines) == 1
    assert actor.observed_deadlines[0] > 0

    view = route.application.published_review_for_book(route.resolved().book)
    assert view is not None
    assert view.publication.book_authority is BookAuthority.DEVELOPMENT_RESULT
    assert view.publication.result_hash == route.result_hash
    assert view.dossier.issuer("AAPL").exposure_band in {
        ExposureBand.HIGH,
        ExposureBand.CRITICAL,
    }
    assert view.recommendation.route is PortfolioReviewRoute.MATERIAL_OBJECTION
    action = view.recommendation.required_actions[0]
    assert action.action is RequiredActionKind.RECONSIDER_CANDIDATE
    assert action.entity_id == "AAPL"
    assert view.recommendation.action_activation == "NOT_AUTHORIZED"
    # regression (V90): a read takes the review's standing from its receipt's policy against
    # the installed one and the recorded moves, never by recompiling; under a policy its
    # receipt does not lead to, the review reads back as recorded and is not reused.
    assert view.under_installed_policy
    moved = PortfolioReviewPublicationService(
        route.runtime.artifacts, installed_policy=lambda: "e" * 64
    )
    earlier = moved.read(view.publication.publication_hash)
    assert earlier.recommendation == view.recommendation and not earlier.under_installed_policy
    assert moved.find_for_review_key(view.publication.review_key) is None
    # The recommendation asks a person to act (V46): its readback says so, and the pending
    # decisions list it with the request that opens its export.
    from alphalattice.control.product_host.composition.pending_decisions import (
        pending_decisions,
    )
    from alphalattice.control.product_host.composition.upgrade_overview import upgrade_overview

    marker = view.person_action
    assert marker is not None and marker["route"] == "MATERIAL_OBJECTION"
    assert {
        "action": "RECONSIDER_CANDIDATE",
        "entity_id": "AAPL",
        "reason": action.reason,
        "blocking": action.blocking,
    } in marker["actions"]
    overview = upgrade_overview(
        workspace=route.runtime_path.parent,
        registry=route.task_registry,
        review=route.application,
        resume_refusal=lambda _task: None,
        command_running=lambda _task_id: False,
    )
    listed = pending_decisions(tasks=(), awaiting={}, data_issues={}, overview=overview)
    [decision] = [item for item in listed["decisions"] if item["kind"] == "CRO_RECOMMENDATION"]
    assert decision["review_publication_hash"] == view.publication.publication_hash
    assert decision["person_action"] == marker
    assert decision["next_requests"]["export"]["operation"] == "EVIDENCE_CRO_EXPORT"
    # regression (V119): a damaged review is named in its own row, never in the way of the
    # page, and the overview reads the Tasks the request already read.
    damaged = "d" * 64
    folder = route.runtime.artifacts.root / "cro-review-publications"
    (folder / f"{damaged}.json").write_text("{", encoding="utf-8")
    overview = upgrade_overview(
        workspace=route.runtime_path.parent,
        registry=route.task_registry,
        review=route.application,
        resume_refusal=lambda _task: None,
        command_running=lambda _task_id: False,
        tasks=(),
    )
    rows = {row["review_publication_hash"]: row for row in overview["reviews"]}
    assert rows[damaged]["state"] == "UNREADABLE" and rows[damaged]["review_key"] is None
    assert overview["studies"] == [] and overview["tasks"] == []
    listed = pending_decisions(tasks=(), awaiting={}, data_issues={}, overview=overview)
    assert [
        item["review_publication_hash"]
        for item in listed["decisions"]
        if item["kind"] == "CRO_RECOMMENDATION"
    ] == [view.publication.publication_hash]
    (folder / f"{damaged}.json").unlink()

    projection = EvidenceCroProjector(route.application).projection()
    assert projection.state == "REVIEW_PUBLISHED"
    assert projection.evidence_selection == "UNIQUE_CURRENT"
    assert projection.required_actions[0].action == "RECONSIDER_CANDIDATE"
    assert len(projection.issuer_rows) == len(TICKERS)
    assert projection.issue_cards
    assert projection.citations
    assert projection.claim_limits
    # The owner's resolved publication identity crosses the HTTP boundary for an
    # installed result too; it is the same public locator history and the export
    # route take, so a client reads it here instead of inferring it. No other
    # store identity leaves with it.
    assert projection.review_publication_hash == view.publication.publication_hash
    body_json = evidence_cro_body(projection)
    assert body_json["review_publication_hash"] == view.publication.publication_hash
    body = json.dumps(body_json)
    assert set(re.findall(r"[0-9a-f]{64}", body)) <= {
        route.result_hash,
        view.publication.publication_hash,
    }, "no store identity"
    assert projection.book is not None
    unpublished = evidence_cro_body(alternative_evidence_ready_for_review(book=projection.book))
    assert "review_publication_hash" in unpublished
    assert unpublished["review_publication_hash"] is None, "absent identity stays explicit"
    body = json.dumps(evidence_cro_body(projection))
    assert projection.review_publication_hash is not None
    assert set(re.findall(r"[0-9a-f]{64}", body)) <= {
        route.result_hash,
        projection.review_publication_hash,
    }, "no store identity beyond the book and the review on display"


def test_an_exact_reuse_admits_no_task_and_calls_no_actor(route: Any) -> None:
    route.wire_evidence()
    route.refresh()
    actor = _reviewer(ControlledRisk("AAPL"))
    assert route.review(actor, drain=True).disposition == "ADMITTED"
    calls = len(actor.observed_deadlines)
    tasks = len(route.task_registry.tasks())

    again = route.review(actor)
    assert again.disposition == "REUSED_EXACT"
    assert again.task_id is None
    assert again.review is not None
    assert len(actor.observed_deadlines) == calls
    assert len(route.task_registry.tasks()) == tasks

    # A different assessment is a different review, not a reuse.
    other = _reviewer(ControlledRisk("AAPL", severity=CRORiskSeverity.MEDIUM))
    assert route.review(other, drain=True).disposition == "ADMITTED"
    assert len(route.task_registry.tasks()) == tasks + 1


@pytest.mark.parametrize("carried_review", (False, True))
def test_expired_evidence_is_a_typed_refusal_not_a_stale_review(
    route: Any, carried_review: bool
) -> None:
    route.wire_evidence()
    route.refresh()
    application = route.application
    if carried_review:
        assert route.review(_reviewer(ControlledRisk("AAPL")), drain=True).disposition == "ADMITTED"
        # A different book has no coverage run to anchor its carried-only dossier.
        handoff = ValidatedPortfolioHandoff.create(
            workspace_id="qa-gate-9c5",
            finalization_task_id="00000000-0000-4000-8000-000000000009",
            candidate_hash="c" * 64,
            package_hash="d" * 64,
            validation_receipt_hash="e" * 64,
            released_report_hash=route.report.report_hash,
            released_at=_NOW,
        )
        application = route.build_application(handoff=handoff)
        application.evidence_task_adapter = route.application.evidence_task_adapter
    application.clock = lambda: _NOW + timedelta(days=2)
    actor = _reviewer(ControlledRisk("AAPL")) if carried_review else _reviewer()

    assert route.current_evidence(application) is None
    outcome = route.review(actor, drain=carried_review, application=application)
    projector = EvidenceCroProjector(application)
    projection = projector.projection()
    if not carried_review:
        assert outcome.disposition == "REFUSED_ALTERNATIVE_EVIDENCE_EXPIRED"
        assert actor.observed_deadlines == []
        assert projection.state == "ALTERNATIVE_EVIDENCE_EXPIRED"
        assert projection.required_actions[0].action == "PREPARE_EVIDENCE"
        return
    assert outcome.disposition == "ADMITTED" and outcome.task_id is not None
    assert route.task_registry.task(outcome.task_id).lifecycle is TaskLifecycle.SUCCEEDED
    assert projection.state == "REVIEW_PUBLISHED" and projection.review_state == "PARTIAL"
    handle = projection.review_publication_hash
    assert handle is not None

    application.clock = lambda: _NOW + timedelta(days=2, minutes=1)
    moved = projector.projection()
    assert moved.state == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
    assert moved.review_state is None, "the sealed review is not today's dossier"
    assert "Eligible earlier readings support this dossier." in moved.explanation
    assert "0 of 1 units have a current analysis; current coverage gaps remain named." in (
        moved.explanation
    )
    assert "A current Alternative Evidence analysis covers" not in moved.explanation
    assert "No recommendation has been published yet" not in moved.explanation
    assert moved.review_publication_hash == handle
    assert moved.next_requests["export"]["review_publication_hash"] == handle
    assert route.current_evidence(application) is None
    historical = projector.projection(publication_hash=handle)
    assert historical.review_state == "PARTIAL" and historical.available_actions == ()


def test_the_same_evidence_serves_a_handoff_and_a_development_book(route: Any) -> None:
    """Evidence is about issuers; a review is about a book.

    One analysis answers the issuer question for both the development result
    and a validated handoff released over the same report. The reviews differ:
    an objection asks Portfolio to reconsider the candidate on the one, and not
    to activate on the other, and each is published under its own key.
    """

    route.wire_evidence()
    route.refresh()
    development = route.current_evidence()
    assert development is not None

    handoff = ValidatedPortfolioHandoff.create(
        workspace_id="qa-gate-9c5",
        finalization_task_id="00000000-0000-4000-8000-000000000009",
        candidate_hash="c" * 64,
        package_hash="d" * 64,
        validation_receipt_hash="e" * 64,
        released_report_hash=route.report.report_hash,
        released_at=_NOW,
    )
    released = route.build_application(handoff=handoff)
    resolved = route.resolved(released)
    assert resolved.book.authority is BookAuthority.VALIDATED_HANDOFF
    assert resolved.book.handoff_hash == handoff.handoff_hash
    reused = route.current_evidence(released)
    assert reused is not None
    assert reused.publication.publication_hash == development.publication.publication_hash
    assert (
        EvidenceCroProjector(released).projection().state == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
    )

    actor = _reviewer(ControlledRisk("AAPL"))
    assert route.review(actor, drain=True, application=released).disposition == "ADMITTED"
    handoff_view = released.published_review_for_book(resolved.book)
    assert handoff_view is not None
    assert handoff_view.publication.result_hash is None
    assert handoff_view.dossier.handoff_hash == handoff.handoff_hash
    assert handoff_view.recommendation.route is PortfolioReviewRoute.MATERIAL_OBJECTION
    assert (
        handoff_view.recommendation.required_actions[0].action is RequiredActionKind.DO_NOT_ACTIVATE
    )

    assert route.application.published_review_for_book(route.resolved().book) is None
    assert route.review(actor, drain=True).disposition == "ADMITTED"
    development_view = route.application.published_review_for_book(route.resolved().book)
    assert development_view is not None
    assert development_view.publication.review_key != handoff_view.publication.review_key
    assert (
        development_view.recommendation.required_actions[0].action
        is RequiredActionKind.RECONSIDER_CANDIDATE
    )
    publications = route.application.artifacts.values(
        "analysis-publications", AlternativeEvidenceAnalysisPublication
    )
    assert len(publications) == 1, "one analysis served both books"


# ================================================================ recovery


@pytest.mark.parametrize("never_started", (False, True))
def test_a_review_task_resumes_exactly_once_after_an_interruption(
    route: Any, monkeypatch: Any, never_started: bool
) -> None:
    """A lost process owes the work, not a new Task.

    The adapter's own verify raises once, Task Control marks the Task
    `RECOVERY_REQUIRED` from its own failure path, and recovery goes through the
    production seam: `recovery_commands()` rebuilds the command from the durable
    Task input and the dispatcher resumes it under the admitted identity.
    """

    route.wire_evidence()
    route.refresh()
    actor = _reviewer(ControlledRisk("AAPL"))
    route.application.review_actor = actor

    original = PortfolioReviewTaskAdapter.verify_stage

    def _interrupt(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("simulated interruption inside the review Task")

    monkeypatch.setattr(PortfolioReviewTaskAdapter, "verify_stage", _interrupt)
    if never_started:
        dispatcher = LocalBackgroundDispatcher(status_port=route.task_registry)
        try:
            monkeypatch.setattr(dispatcher, "start", lambda: None)
            outcome = route.application.review(dispatcher=dispatcher, actor=actor)
        finally:
            dispatcher.close()
    else:
        outcome = route.review(actor, drain=True)
    assert outcome.disposition == "ADMITTED"
    assert outcome.task_id is not None
    registry = route.task_registry
    assert registry.task(outcome.task_id).lifecycle is (
        TaskLifecycle.QUEUED if never_started else TaskLifecycle.RECOVERY_REQUIRED
    )
    assert route.application.published_review_for_book(route.resolved().book) is None

    monkeypatch.setattr(PortfolioReviewTaskAdapter, "verify_stage", original)
    commands = route.application.recovery_commands()
    assert set(commands) == {PortfolioReviewTaskAdapter.task_kind}
    dispatcher = LocalBackgroundDispatcher(status_port=registry)
    try:
        resumed = dispatcher.resume(cast(Any, commands))
        dispatcher.drain_for_tests()
    finally:
        dispatcher.close()

    assert resumed == (outcome.task_id,), "the original identity, not a new Task"
    assert registry.task(outcome.task_id).lifecycle is TaskLifecycle.SUCCEEDED
    view = route.application.published_review_for_book(route.resolved().book)
    assert view is not None
    stored = route.application.review_publications.store.values(
        "cro-review-publications", PortfolioReviewPublication
    )
    assert len(stored) == 1, "recovery published once, not twice"


@pytest.mark.parametrize("never_started", (False, True))
def test_an_evidence_task_resumes_exactly_once_after_an_interruption(
    route: Any, monkeypatch: Any, never_started: bool
) -> None:
    """The same recovery contract on the evidence half: the command is rebuilt
    from the run the Task was admitted with (every book is a run, C2)."""

    route.wire_evidence()
    original = AlternativeEvidenceDocumentTaskAdapter.verify_stage

    def _interrupt(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("simulated interruption inside the evidence Task")

    monkeypatch.setattr(AlternativeEvidenceDocumentTaskAdapter, "verify_stage", _interrupt)
    if never_started:
        dispatcher = LocalBackgroundDispatcher(status_port=route.task_registry)
        try:
            monkeypatch.setattr(dispatcher, "start", lambda: None)
            outcome = route.application.refresh_evidence(dispatcher=dispatcher)
        finally:
            dispatcher.close()
    else:
        outcome = route.refresh()
    assert outcome.disposition == "ADMITTED"
    assert outcome.task_id is not None
    registry = route.task_registry
    assert registry.task(outcome.task_id).lifecycle is (
        TaskLifecycle.QUEUED if never_started else TaskLifecycle.RECOVERY_REQUIRED
    )
    assert route.current_evidence() is None
    assert (
        EvidenceCroProjector(route.application).projection().state == "EVIDENCE_REFRESH_IN_PROGRESS"
    )

    monkeypatch.setattr(AlternativeEvidenceDocumentTaskAdapter, "verify_stage", original)
    commands = route.application.recovery_commands()
    assert set(commands) == {AlternativeEvidenceDocumentTaskAdapter.task_kind}
    recovered = commands[AlternativeEvidenceDocumentTaskAdapter.task_kind]
    admitted_input = registry.task(outcome.task_id).input.payload
    run = recovered.run  # type: ignore[attr-defined]
    assert run is not None and run.run_hash == admitted_input["run_hash"]
    (unit,) = run.units
    dispatcher = LocalBackgroundDispatcher(status_port=registry)
    try:
        resumed = dispatcher.resume(cast(Any, commands))
        dispatcher.drain_for_tests()
    finally:
        dispatcher.close()

    assert resumed == (outcome.task_id,)
    assert registry.task(outcome.task_id).lifecycle is TaskLifecycle.SUCCEEDED
    view = route.current_evidence()
    assert view is not None
    assert view.publication.obligation_hash == unit.obligation.obligation_hash
    assert (
        EvidenceCroProjector(route.application).projection().state
        == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
    )


# ============================================== which publication answers this


def _second_analysis_in_the_same_store(route: Any) -> None:
    """A second answer to the same obligation, from amended text, a minute later."""

    scope = route.resolved().scope
    route.wire_evidence(
        documents=_documents(
            scope.ordered_entity_ids,
            revisions=("amended",),
            suffix=" The statement was later amended.",
        )
    )
    route.application.clock = lambda: _NOW + timedelta(minutes=1)
    try:
        assert route.refresh(evidence_as_of=_NOW).disposition == "ADMITTED"
    finally:
        route.application.clock = lambda: _NOW


def test_resolving_an_ambiguity_needs_no_provider_credential(route: Any) -> None:
    """requirement: choosing which admitted analysis is in force is a local decision.

    Nothing here asks a model anything: two analyses are already published, and
    picking one between them reads artifacts and writes a record. The section
    used to refuse the whole state before it read any of it, so a workspace with
    no credential could neither see the choice nor make it.

    The state after choosing is the honest one for a workspace that cannot do
    model work: the analysis is current and ready for review, the dossier and
    its assessment submission are the next step, and only the managed review
    is withheld.
    """

    route.wire_evidence()
    route.refresh()
    first = route.current_evidence()
    assert first is not None
    _second_analysis_in_the_same_store(route)

    application = route.application
    # A state's offered actions are the requests it offers (V198): with managed
    # work admitted, a choice between two analyses offers neither a refresh nor
    # a CRO review.
    assert application.model_authority_admitted
    managed = EvidenceCroProjector(application).projection()
    assert managed.state == "EVIDENCE_SELECTION_AMBIGUOUS" and managed.next_requests == {}
    assert managed.available_actions == ()
    application.model_authority_admitted = False

    ambiguous = EvidenceCroProjector(application).projection()
    assert ambiguous.state == "EVIDENCE_SELECTION_AMBIGUOUS"
    assert len(ambiguous.eligible_versions) == 2, "both choices are offered"
    assert not any(value.is_selected for value in ambiguous.eligible_versions)

    selector = application.default_selector()
    assert selector is not None
    chosen = first.publication.publication_hash
    selection = application.select_evidence(
        selector=selector, analysis_publication_hash=chosen, chosen_by="HUMAN"
    )
    assert selection.analysis_publication_hash == chosen
    assert selection.chosen_by == "HUMAN"

    settled = EvidenceCroProjector(application).projection()
    assert settled.state == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW", (
        "the analysis is current; the native dossier is the next step"
    )
    assert settled.available_actions == (), "managed review needs the credential"
    assert list(settled.next_requests) == ["dossier"], "no managed request is composed"
    assert settled.next_requests["dossier"]["operation"] == "CRO_REVIEW_DOSSIER"
    assert "credential" in settled.explanation
    assert settled.book is not None, "the book it concerns is still named"

    # The choice is recorded and survives a fresh read of the same store.
    recorded = recorded_selection(application.artifacts, route.resolved().scope.scope_hash)
    assert recorded is not None
    assert recorded.analysis_publication_hash == chosen

    # An ineligible publication is still refused, credential or not.
    with pytest.raises(PortfolioEvidenceReviewError, match="not_eligible"):
        application.select_evidence(
            selector=selector, analysis_publication_hash="0" * 64, chosen_by="HUMAN"
        )

    # And admitting the credential again offers the review, without re-choosing.
    application.model_authority_admitted = True
    assert (
        EvidenceCroProjector(application).projection().state
        == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
    )


def test_two_current_answers_are_an_ambiguity_until_one_is_selected(route: Any) -> None:
    """Which analysis was read is part of what a recommendation means."""

    route.wire_evidence()
    route.refresh()
    first = route.current_evidence()
    assert first is not None
    _second_analysis_in_the_same_store(route)
    matching = [
        value
        for value in route.application.artifacts.values(
            "analysis-publications", AlternativeEvidenceAnalysisPublication
        )
        if value.obligation_hash == first.publication.obligation_hash
    ]
    assert len(matching) == 2, "the same store now holds two answers"
    with pytest.raises(EvidenceSelectionAmbiguous):
        route.current_evidence()

    actor = _reviewer()
    before = len(route.task_registry.tasks())
    outcome = route.review(actor)
    assert outcome.disposition == "REFUSED_EVIDENCE_SELECTION_AMBIGUOUS"
    assert outcome.task_id is None
    assert actor.observed_deadlines == []
    assert len(route.task_registry.tasks()) == before
    assert (
        EvidenceCroProjector(route.application).projection().state == "EVIDENCE_SELECTION_AMBIGUOUS"
    )

    route.application.selected_analysis_publication_hash = first.publication.publication_hash
    assert (
        EvidenceCroProjector(route.application).projection().state
        == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
    )
    assert route.review(actor, drain=True).disposition == "ADMITTED"
    projection = EvidenceCroProjector(route.application).projection()
    assert projection.state == "REVIEW_PUBLISHED"
    assert projection.evidence_selection == "EXPLICIT_OLDER_CURRENT_SELECTION"


def test_an_orphaned_sibling_analysis_never_closes_the_section_over_a_current_one(
    route: Any,
) -> None:
    """requirement: a publication sealed under an authority the workspace has
    rotated past and never listed is history the section cannot read; it must
    not make a current answer to the same obligation unreadable. Seen on a QA
    copy whose earlier journeys ran on an uncommitted tree: after the next
    review the Evidence & CRO section raised `analysis_publication_authority_mismatch`
    for the whole book, from the selection label replaying every sibling.
    The orphan is excluded from the label; its own review still refuses by name.
    """

    from alphalattice.evidence.alternative_evidence.publication.analysis import (
        AlternativeEvidenceAnalysisPublicationService,
    )
    from alphalattice.evidence.alternative_evidence.publication.artifacts import (
        AlternativeEvidencePublicationError,
    )
    from alphalattice.evidence.alternative_evidence.runtime.identity import (
        HistoricalEvidenceBindings,
    )

    route.wire_evidence()
    assert route.refresh().disposition == "ADMITTED"
    first = route.current_evidence()
    assert first is not None and first.is_current
    assert route.review(_reviewer(ControlledRisk("AAPL")), drain=True).disposition == ("ADMITTED")
    application = route.application
    published = application.published_review_for_book(route.resolved().book)
    assert published is not None
    orphaned_review = published.publication.publication_hash
    selector = cast(BookSelector, application.default_selector())

    # The bindings rotate and the old tuple is not listed: the first analysis
    # and its review are orphans.
    current = application.evidence_publications
    assert current is not None
    sealed = HistoricalEvidenceBindings(*current.expected_bindings)
    rotated = ["1" * 64 if index == 2 else value for index, value in enumerate(sealed.bindings)]
    rotated_service = AlternativeEvidenceAnalysisPublicationService(
        current.store,
        acquisition_binding_hash=rotated[0],
        canonicalization_binding_hash=rotated[1],
        retrieval_binding_hash=rotated[2],
        analysis_policy_hash=rotated[3],
        decision_policy_hash=rotated[4],
        publication_binding_hash=rotated[5],
        supported_historical_bindings=(),
    )
    application.evidence_publications = rotated_service
    runtime = application.evidence_task_adapter.runtime
    runtime.retrieval_binding_hash = rotated[2]
    runtime.retrieval.retrieval_binding_hash = rotated[2]  # the next generation's binding
    runtime.publications = rotated_service  # what the next analysis seals under
    assert route.current_evidence() is None

    # A fresh preparation and analysis under the new bindings, the same
    # obligation, then its review.
    _second_analysis_in_the_same_store(route)
    second = route.current_evidence()
    assert second is not None and second.is_current
    assert second.publication.publication_hash != first.publication.publication_hash
    assert second.publication.obligation_hash == first.publication.obligation_hash
    assert route.review(_reviewer(ControlledRisk("AAPL")), drain=True).disposition == ("ADMITTED")
    projection = EvidenceCroProjector(application).projection()
    assert projection.state == "REVIEW_PUBLISHED"
    assert projection.evidence_selection == "UNIQUE_CURRENT", "the orphan is no candidate"
    assert projection.review_publication_hash != orphaned_review
    exported = EvidenceReviewDelivery(application).export_review(
        selector, projection.review_publication_hash
    )
    assert exported["review_status"] == "EXACT_HISTORICAL_READBACK"
    with pytest.raises(AlternativeEvidencePublicationError, match="authority_mismatch"):
        EvidenceReviewDelivery(application).export_review(selector, orphaned_review)


def test_a_rotated_binding_keeps_old_analyses_and_reviews_readable_never_current(
    route: Any,
) -> None:
    """requirement: exact-handle history survives an implementation change; no
    current eligibility follows from being decodable.

    The whole route is driven under the bindings in force, then the
    workspace's retrieval binding rotates (the reader is rebuilt with another
    hash, exactly what an edit to a retrieval owner does). The tuple the
    publications carry is listed as a supported historical contract: the
    analysis verifies as history, its review still exports by handle, the
    section says the analysis was superseded and asks for a new preparation,
    a managed review is refused by name, the packet prepared under the old
    binding is no longer offered, and nothing on disk was rewritten. Without
    the listing the same tuple refuses as it always did.
    """

    from alphalattice.evidence.alternative_evidence.publication.analysis import (
        AlternativeEvidenceAnalysisPublicationService,
    )
    from alphalattice.evidence.alternative_evidence.publication.artifacts import (
        AlternativeEvidencePublicationError,
    )
    from alphalattice.evidence.alternative_evidence.runtime.identity import (
        HistoricalEvidenceBindings,
    )

    route.wire_evidence()
    assert route.refresh().disposition == "ADMITTED"
    evidence = route.current_evidence()
    assert evidence is not None and evidence.is_current
    assert route.review(_reviewer(ControlledRisk("AAPL")), drain=True).disposition == ("ADMITTED")
    application = route.application
    published = application.published_review_for_book(route.resolved().book)
    assert published is not None
    review_hash = published.publication.publication_hash
    selector = cast(BookSelector, application.default_selector())
    before = EvidenceReviewDelivery(application).export_review(selector, review_hash)
    assert before["review_status"] == "EXACT_HISTORICAL_READBACK"
    analysis_hash = evidence.publication.publication_hash
    store_files = {
        path: path.read_bytes() for path in sorted(route.runtime.artifacts.root.rglob("*.json"))
    }

    current = application.evidence_publications
    assert current is not None
    sealed = HistoricalEvidenceBindings(*current.expected_bindings)
    rotated = ["1" * 64 if index == 2 else value for index, value in enumerate(sealed.bindings)]

    def reader(historical: tuple[HistoricalEvidenceBindings, ...]) -> Any:
        return AlternativeEvidenceAnalysisPublicationService(
            current.store,
            acquisition_binding_hash=rotated[0],
            canonicalization_binding_hash=rotated[1],
            retrieval_binding_hash=rotated[2],
            analysis_policy_hash=rotated[3],
            decision_policy_hash=rotated[4],
            publication_binding_hash=rotated[5],
            supported_historical_bindings=historical,
        )

    # Unsupported: the rotation refuses the old tuple exactly as before.
    unsupported = reader(())
    with pytest.raises(AlternativeEvidencePublicationError, match="authority_mismatch"):
        unsupported.verify(analysis_hash)
    application.evidence_publications = unsupported
    with pytest.raises(AlternativeEvidencePublicationError, match="authority_mismatch"):
        EvidenceReviewDelivery(application).export_review(selector, review_hash)

    # Supported as history: exact readback, export, and a state that says so.
    supported = reader((sealed,))
    verified = supported.verify(analysis_hash)
    assert verified.binding_contract == f"HISTORICAL:{sealed.contract_hash[:12]}"
    assert verified.is_historical and not verified.is_current
    assert verified.publication == evidence.publication, "the artifact was not rewritten"
    assert supported.replay(analysis_hash, now=_NOW).current_eligibility == "SUPERSEDED"
    with pytest.raises(AlternativeEvidencePublicationError, match="authority_superseded"):
        supported.read(analysis_hash, now=_NOW)
    application.evidence_publications = supported
    application.evidence_task_adapter.runtime.retrieval_binding_hash = rotated[2]
    after = EvidenceReviewDelivery(application).export_review(selector, review_hash)
    assert after["review_status"] == "EXACT_HISTORICAL_READBACK"
    assert after["review"] == before["review"] and after["evidence"] == before["evidence"]
    assert after["export_hash"] == before["export_hash"]
    projection = EvidenceCroProjector(application).projection()
    assert projection.state == "ALTERNATIVE_EVIDENCE_SUPERSEDED"
    assert projection.review_publication_hash == review_hash
    assert projection.required_actions[0].action == "PREPARE_EVIDENCE"
    assert projection.next_requests["export"]["review_publication_hash"] == review_hash
    assert "preview" in projection.next_requests
    assert route.current_evidence() is None, "no current eligibility is inferred"
    actor = _reviewer()
    assert route.review(actor).disposition == "REFUSED_ALTERNATIVE_EVIDENCE_SUPERSEDED"
    assert actor.observed_deadlines == []
    pinned = EvidenceCroProjector(application).projection(publication_hash=review_hash)
    assert pinned.state == "REVIEW_PUBLISHED" and pinned.available_actions == ()
    assert {
        path: path.read_bytes() for path in sorted(route.runtime.artifacts.root.rglob("*.json"))
    } == store_files, "history is read, never resealed"


def test_pending_decisions_read_each_owner_standing() -> None:
    """requirement (RX, V185-V187): the pending list reads each owner's own
    standing, never a reading of its own. A later upgrade stands whatever was
    acknowledged before (the overview's `show`); a data issue past the owner's
    first page is pending too; and the CRO's newest review of each book is the
    one that asks or not, so an older review's action does not stand beside a
    newer review that asks none, and two asking reviews of one book list once."""

    from alphalattice.control.product_host.composition.pending_decisions import (
        all_data_issues,
        pending_decisions,
    )

    def listed(**parts: Any) -> dict[str, Any]:
        values: dict[str, Any] = {"tasks": (), "awaiting": {}, "data_issues": {}, "overview": {}}
        return pending_decisions(**{**values, **parts})

    offered = {"acknowledge": {"operation": "UPGRADE_ACKNOWLEDGE", "upgrade_set_hash": "b" * 64}}
    later = {"acknowledged": {"set_hash": "a" * 64}, "show": True, "next_requests": offered}
    assert [item["kind"] for item in listed(overview=later)["decisions"]] == ["UPGRADE"]
    assert listed(overview={**later, "show": False})["decisions"] == []

    pages = {
        None: {
            "next_requests": {
                f"preview:{index:064x}:hold": {"operation": "DATA_ISSUE_PREVIEW"}
                for index in range(25)
            },
            "next_cursor": f"{24:064x}",
        },
        f"{24:064x}": {
            "next_requests": {f"preview:{25:064x}:hold": {"operation": "DATA_ISSUE_PREVIEW"}},
            "next_cursor": None,
        },
    }
    issues = all_data_issues(lambda cursor: pages[cursor])
    assert listed(data_issues=issues)["counts"] == {"DATA_ISSUE": 26}

    asks = {"route": "MATERIAL_OBJECTION", "actions": []}

    def review(handle: str, day: int, action: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "review_publication_hash": handle * 64,
            "review_key": f"key-{handle}",
            "book_key": "book",
            "published_at": f"2026-09-{day:02d}T00:00:00+00:00",
            "state": "CURRENT",
            "person_action": action,
            "next_requests": {},
        }

    settled = {"reviews": [review("a", 1, asks), review("b", 2, None)]}
    assert listed(overview=settled)["decisions"] == [], "the newer review asks none"
    twice = {"reviews": [review("a", 1, asks), review("c", 3, asks)]}
    [only] = listed(overview=twice)["decisions"]
    assert only["review_publication_hash"] == "c" * 64


def test_a_carried_review_names_the_book_it_read(tmp_path: Path) -> None:
    """regression (V427, an outside review at 97b65a25): a CRO bundle whose review carries
    forward answered without the book it read, so `evidence show --from` the answer read the
    default book; it names its book and its Task, as a submitted answer does."""

    from alphalattice.control.product_host.composition.evidence_review_application import (
        ReviewOutcome,
    )
    from alphalattice.control.product_host.composition.evidence_review_bundles import (
        EvidenceReviewBundles,
    )
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector

    book = BookSelector(result_hash="b" * 64)
    task = UUID(int=7)

    class Review:
        """The review owner as far as a carried review reaches it."""

        playpen_root = Path(__file__).resolve().parents[2]

        def _review_selector(self, selector: BookSelector) -> BookSelector:
            return selector

        def clock(self) -> datetime:
            return datetime(2026, 10, 1, tzinfo=UTC)

        def _resolve_review_dossier(self, chosen: BookSelector, *, read_at: datetime) -> object:
            return object()

        def _carry_forward(self, dossier: object, *, dispatcher: object) -> ReviewOutcome:
            return ReviewOutcome(disposition="ADMITTED", detail="Carried.", task_id=task)

    carried = EvidenceReviewBundles(Review()).prepare_agent_bundle(  # type: ignore[arg-type]
        role="CRO", selector=book, directory=str(tmp_path / "bundle"), dispatcher=object()
    )
    assert isinstance(carried, ReviewOutcome)
    assert carried.disposition == "REVIEW_CARRIED_FORWARD"
    assert carried.next_requests == {
        "task": {"operation": "STATUS", "task_id": str(task)},
        "book": {"operation": "EVIDENCE_CRO", **book.request_fields()},
    }


def test_a_carried_review_offers_no_packet_of_the_reviews_task() -> None:
    """regression (V434, an outside review at 0c2b62a0): with nothing new the refresh carried
    the CRO's review forward, and EVIDENCE_PREPARE offered packet and Analyst-bundle requests for
    the review's Task, each refused `alternative_evidence.current_authority_mismatch`; a review's
    Task has no packet, and the answer offers its Task and its book."""

    from types import SimpleNamespace

    from alphalattice.control.product_host.composition.evidence_review_application import (
        EvidenceReviewApplication,
        ReviewOutcome,
    )
    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector

    book = BookSelector(result_hash="b" * 64)
    review_task = UUID(int=9)
    review = EvidenceReviewApplication.__new__(EvidenceReviewApplication)
    review.evidence_task_adapter = SimpleNamespace(task_kind="alternative_evidence.refresh")
    review.session = SimpleNamespace(
        task_control_registry=SimpleNamespace(
            task=lambda _task_id: SimpleNamespace(task_kind="chief_risk_officer.review")
        )
    )
    review.default_selector = lambda selector=None: book  # type: ignore[method-assign]
    assert review.packet_requests(book, task_id=review_task) == {}

    def refresh_evidence(**_kwargs: object) -> ReviewOutcome:
        return ReviewOutcome(
            disposition="REUSED_EXACT", detail="Nothing new: carried.", task_id=review_task
        )

    review.refresh_evidence = refresh_evidence  # type: ignore[method-assign]
    operations = PortfolioResearchOperations.__new__(PortfolioResearchOperations)
    operations.review = review  # type: ignore[assignment]
    operations.dispatcher = object()  # type: ignore[assignment]
    answer = operations._review_operation(
        PortfolioResearchOperationRequest(operation="EVIDENCE_PREPARE", result_hash="b" * 64),
        caller="EXTERNAL_AUTOMATION",
    )
    assert answer is not None
    assert answer["next_requests"] == {
        "task": {"operation": "STATUS", "task_id": str(review_task)},
        "book": {"operation": "EVIDENCE_CRO", **book.request_fields()},
    }


def test_a_recorded_preview_names_the_units_it_cannot_prepare() -> None:
    """regression (V445, AX14's finding; V541, RR5d's): offline, a preview in RECORDED mode
    offered `prepare` while most units held no recorded document, and the coverage Task then
    succeeded with them failed. The units that cannot prepare are named before any run with the
    code each would fail: short of the installed floor, `minimum_entity_coverage_not_met` naming
    its reach and need and the issuers without a source, as the unit's own refusal does; with no
    document under a floor that admits it, `document_set_empty`. V445 named the second where the
    first comes first. Under official acquisition none is named: the run reads the index."""

    from alphalattice.control.product_host.composition.evidence_review_application import (
        units_short_of_sources,
    )
    from alphalattice.evidence.alternative_evidence.contracts import AlternativeEvidenceMode

    inventory = {
        "issuers": [
            {"entity_id": "TSN", "unit_id": "u01", "documents": 0},
            {"entity_id": "TPL", "unit_id": "u01", "documents": 3},
            {"entity_id": "ERIE", "unit_id": "u02", "documents": 0},
            {"entity_id": "LII", "unit_id": "u02", "documents": 0},
            {"entity_id": "APO", "unit_id": "u03", "documents": 0},
        ]
    }
    recorded = AlternativeEvidenceMode.RECORDED
    short = "alternative_evidence.minimum_entity_coverage_not_met:"
    assert units_short_of_sources(inventory, recorded, floor=0.6) == [
        {
            "unit_id": "u01",
            "failure_code": short + "1 of 2 issuers hold a source, 2 needed",
            "issuers_without_source": ["TSN"],
        },
        {
            "unit_id": "u02",
            "failure_code": short + "0 of 2 issuers hold a source, 2 needed",
            "issuers_without_source": ["ERIE", "LII"],
        },
        {
            "unit_id": "u03",
            "failure_code": short + "0 of 1 issuers hold a source, 1 needed",
            "issuers_without_source": ["APO"],
        },
    ]
    empty = "alternative_evidence.document_set_empty"
    assert units_short_of_sources(inventory, recorded, floor=0.0) == [
        {"unit_id": "u02", "failure_code": empty, "issuers_without_source": ["ERIE", "LII"]},
        {"unit_id": "u03", "failure_code": empty, "issuers_without_source": ["APO"]},
    ]
    official = AlternativeEvidenceMode.LIVE_OFFICIAL
    assert units_short_of_sources(inventory, official, floor=0.6) == []


def test_the_source_ways_offer_a_package_only_where_one_covers_the_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression (V546, V547, RR5d): RR5d's lead installed one recorded package per unit, each
    replacing the last, and the units it prepared then stood at different cutoffs that no review
    could read together. The ways on prefer official acquisition, which prepares every unit at
    one cutoff; a package is offered only for a book of one unit, which one package covers, and
    the package rule is said either way."""

    from alphalattice.control.product_host.composition.evidence_review_application import (
        PACKAGE_RULE,
        source_ways,
    )

    many = source_ways(tmp_path, entities=None, book=None)
    assert set(many) == {"official", "package_rule"}
    assert many["package_rule"] == PACKAGE_RULE
    assert "cannot be reviewed together" in PACKAGE_RULE and "goes stale" in PACKAGE_RULE
    official = cast(dict[str, str], many["official"])
    assert official["serve"].endswith("serve --sec-network-consent")
    assert "one cutoff" in official["before"]
    # V620: a setup way names the deciding operator hold before consent advice.
    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    held = cast(dict[str, str], source_ways(tmp_path, entities=None, book=None)["official"])
    assert "ALPHALATTICE_NETWORK_DISABLED=1" in held["before"]
    assert "restart the idle Host" in held["before"]
    from alphalattice.control.workspace_runtime.network_access import set_network_access

    monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
    set_network_access(tmp_path, enabled=False)
    closed = cast(dict[str, str], source_ways(tmp_path, entities=None, book=None)["official"])
    assert "This workspace's network control decides" in closed["before"]
    assert "ALPHALATTICE_NETWORK_DISABLED=1" not in closed["before"]
    one = source_ways(tmp_path, entities=("AAPL", "MSFT"), book=None)
    package = cast(dict[str, object], one["package"])
    assert package["entities"] == ["AAPL", "MSFT"]
    assert "--entities AAPL MSFT" in str(package["install"])


def test_a_review_selector_is_refused_in_words_with_the_request_that_works() -> None:
    """regression (V546, RR5d's FINDING 21:24): `review dossier --result <receipt> --session
    <day>` and `--study <replay> --receipt <r> --session <day>` were refused
    `evidence_review_experiment_selector_invalid` and `task_kind_mismatch` with no words. A
    study's part beside another book is named, with the request without it; a selector missing a
    part says which three name a study's book; and a study's selector naming a Task of another
    kind -- a public development replay, whose book is its published result -- offers the request
    by the receipt it named."""

    from alphalattice.control.product_host.composition.plain_refusals import explain
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
        BookSelector,
        PortfolioEvidenceReviewError,
    )

    receipt = "a" * 64
    with pytest.raises(PortfolioEvidenceReviewError) as stray:
        BookSelector(result_hash=receipt, portfolio_session="2026-09-29")
    code = str(stray.value)
    assert code == "product_host.evidence_review_experiment_selector_invalid:portfolio_session"
    book = {"operation": "CRO_REVIEW_DOSSIER", "result_hash": receipt}
    words = explain(code, book=book)
    assert "`portfolio_session` names a study's book" in words["detail"]
    assert words["next_requests"] == {"without_portfolio_session": book}
    with pytest.raises(PortfolioEvidenceReviewError) as missing:
        BookSelector(experiment_task_id=UUID(int=1), portfolio_session="2026-09-29")
    assert str(missing.value) == "product_host.evidence_review_experiment_selector_invalid"
    words = explain(str(missing.value))
    assert "three parts together" in words["detail"]
    assert words["next_action"] == "NAME_THE_STUDY_TASK_RECEIPT_AND_SESSION"
    replay = "product_host.evidence_review_study_selector_not_a_study"
    kind = "portfolio_public_development_replay"
    words = explain(f"{replay}:{kind}", task_id="72455e96", book=book)
    assert f"Task 72455e96 is a `{kind}` Task, not a study" in words["detail"]
    assert words["next_requests"] == {"result": book}
    words = explain(f"{replay}:{kind}", task_id="72455e96")
    assert words["next_requests"] == {"task": {"operation": "STATUS", "task_id": "72455e96"}}


def test_every_dossier_and_bundle_refusal_on_the_review_route_has_words_and_a_way_on() -> None:
    """requirement (V546, V547, the class): every refusal the review route's owners raise that a
    request or the book's state can reach -- the book its selector names, a dossier, a packet or
    dossier part, a bundle, an answer -- reads in words with a way on, the codes the selector's
    readers raise among them; the codes that only a broken Host or record reaches are named
    here, each with why. Every refusing outcome the route returns carries its words in place."""

    import ast

    from alphalattice.control.product_host.composition.evidence_review_bundles import (
        PACKET_SELECTOR_CODES,
        agent_bundle_refusal,
    )
    from alphalattice.control.product_host.composition.plain_refusals import explain
    from alphalattice.interface.local_application.cli_contract import refusal_words

    def worded(code: str) -> dict[str, Any]:
        """A code's words: the door table's, the plain refusals', or a bundle's own in place."""
        words = refusal_words(code) or explain(code)
        if not words.get("detail") and code.startswith("agent_bundle."):
            try:
                bundle = agent_bundle_refusal(code)
            except KeyError:
                return {}
            return {"detail": bundle["message"], "next_action": bundle["next_action"]}
        return words

    root = Path(__file__).resolve().parents[2] / "src/alphalattice"
    owners = (
        "control/product_host/composition/evidence_review_application.py",
        "control/product_host/composition/evidence_review_delivery.py",
        "control/product_host/composition/evidence_review_bundles.py",
        # The book a request's selector names, opened (V546).
        "oversight/chief_risk_officer/decision/book_evidence.py",
    )
    # What the selector's readers raise for the book a request names: an update's or a study's.
    readers = (
        "product_host.evidence_review_update_not_configured",
        "product_host.evidence_review_update_task_mismatch",
        "product_host.evidence_review_publication_origin_unavailable",
        "portfolio_update.publication_not_bound_to_task",
        "research_update.publication_not_bound_to_task",
        "portfolio_research.session_outside_report",
        "portfolio_research.date_selector_not_portfolio",
    )
    internal = {
        # The Host's own composition: a workspace served without these is refused earlier, by
        # name, as its setup; every Host that serves a review reads studies.
        "product_host.evidence_review_service_absent",
        "product_host.evidence_review_adapter_absent",
        "product_host.evidence_refresh_command_incomplete",
        "product_host.evidence_review_experiment_not_configured",
        # A sealed record that no longer reads or measures as it was sealed: an update whose
        # listing labels and weights disagree, a handoff whose report the ledger holds otherwise,
        # an analysis selected for the review that answers another question than the review's.
        "product_host.evidence_review_update_axis_mismatch",
        "product_host.evidence_review_report_mismatch",
        "product_host.evidence_review_obligation_mismatch",
        "product_host.evidence_review_axis_mismatch",
        "product_host.evidence_review_scope_mismatch",
        "product_host.evidence_coverage_run_unreadable",
        "alternative_evidence.delivery_measurement_unstable",
        "chief_risk_officer.prepared_submission_invalid",
        "chief_risk_officer.submission_identity_mismatch",
        "chief_risk_officer.submitted_actor_invalid",
        "chief_risk_officer.review_not_prepared",
        "chief_risk_officer.review_actor_unavailable",
        # Who may submit or select: a person or an agent through its bundle; the Host's own
        # automation never does.
        "alternative_evidence.external_submission_entry_required",
        "chief_risk_officer.external_submission_entry_required",
        "product_host.evidence_selection_actor_invalid",
        # A slot another writer filled: the Host is its workspace's one writer, and one
        # binding's answers take turns (V557).
        "agent_bundle.answer_slot_taken",
    }
    raised: set[str] = set()
    outcomes = 0
    for name in owners:
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and node.exc.args:
                argument = node.exc.args[0]
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and re.fullmatch(r"[a-z_]+\.[a-z_]+(:.*)?", argument.value)
                ):
                    raised.add(argument.value.partition(":")[0])
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ReviewOutcome":
                keywords = {value.arg: value.value for value in node.keywords}
                disposition = keywords.get("disposition")
                if isinstance(disposition, ast.Constant) and str(disposition.value).startswith(
                    "REFUSED"
                ):
                    outcomes += 1
                    assert "detail" in keywords, (name, disposition.value)
    assert outcomes > 10 and len(raised) > 30
    assert internal <= raised, internal - raised
    reached = sorted((raised | set(readers)) - internal - PACKET_SELECTOR_CODES)
    assert [code for code in reached if not worded(code).get("detail")] == []
    for code in reached:
        words = worded(code)
        assert words.get("next_action") or words.get("next_requests"), code
    for code in (
        "chief_risk_officer.dossier_cutoff_invalid:5 citations of 1 unit after the cutoff "
        "2026-10-03T00:26:23Z, the earliest 2026-10-03T00:46:23Z",
        "product_host.evidence_review_dossier_refused:chief_risk_officer.dossier_children_invalid",
        "alternative_evidence.task_resource_authority_mismatch:package",
    ):
        words = refusal_words(code)
        assert code.partition(":")[2] in words["detail"] and words["next_action"], code


def test_every_evidence_unit_failure_has_words_and_a_way_on() -> None:
    """requirement (V541, the class): a failed coverage unit reads with words and a way on
    whatever its code: every code the Evidence package raises, the owner's code a task failure
    carries, and a refusal of the preparation's own work, which names its code. A unit short of
    sources is offered the sources (`source_ways`) under the recorded package and a retry under
    official acquisition, never a bare code."""

    from alphalattice.control.product_host.composition.plain_refusals import (
        SOURCE_SHORT_CODES,
        unit_failure_words,
    )

    root = Path(__file__).resolve().parents[2] / "src/alphalattice/evidence/alternative_evidence"
    codes = {
        code
        for path in root.rglob("*.py")
        for code in re.findall(r'"(alternative_evidence\.[a-z_]+)', path.read_text("utf-8"))
    }
    assert len(codes) > 300, len(codes)
    wrapped = "alternative_evidence.task_failed:knowledge.semantic_pack_not_pinned"
    for code in sorted({*codes, wrapped}):
        words = unit_failure_words(code)
        assert words.get("detail") and words.get("next_action"), code
    generic = unit_failure_words("alternative_evidence.source_object_tampered")
    assert "`alternative_evidence.source_object_tampered`" in generic["detail"]
    assert generic["next_action"] == "PREPARE_AGAIN_THEN_REPORT_THE_CODE"
    for code in SOURCE_SHORT_CODES:
        words = unit_failure_words(code + ":0 of 8 issuers hold a source, 5 needed")
        assert "`source_ways`" in words["detail"], code
        assert words["next_action"] == "ASK_FOR_OFFICIAL_ACQUISITION_OR_A_COVERING_PACKAGE"
        assert unit_failure_words(code, recorded=False)["next_action"] == (
            "PREVIEW_AND_PREPARE_AGAIN"
        )
