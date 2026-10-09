"""The CRO's reading of one sealed Portfolio book against independent issuer evidence.

Everything here is deterministic. The review opens a sealed book, projects it,
maps it onto admitted issuers, derives an evidence-only obligation from the
resulting scope, reopens an exact Alternative Evidence publication, and
assembles the dossier a bounded actor may reason over. The Host composes it with
the Evidence route and the Task runner (W1: moved from the Host).

Two rules shape the module:

- **Evidence never learns the Portfolio.** The obligation projected toward
  Alternative Evidence carries an issuer axis, a cutoff, approved source
  families and required checks -- no weight, no change, no band, no report
  identity -- so a weight-only edit to the book cannot rotate the evidence
  identity and an analyst cannot be steered by how large a position is.
- **The Portfolio is never replayed here.** Holdings come from sealed readback;
  exposure changes and concentration are projected from those verified facts
  (including sealed HHI for authored books), not a new fit or backtest.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Literal, cast
from uuid import UUID

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    EXECUTED_STATES,
    AlternativeEvidenceResearchObligation,
    AnalysisCompletion,
    seal_research_obligation,
)
from alphalattice.evidence.alternative_evidence.analysis.read_model import finding_filings
from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceReadFiling,
    AlternativeEvidenceRequest,
    SecIssuerRegistrySnapshot,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.publication.analysis import (
    AlternativeEvidenceAnalysisPublicationView,
)
from alphalattice.evidence.alternative_evidence.runtime.coverage import (
    UNIT_LIMIT,
    UNIT_PACKING_RULES_ID,
    AlternativeEvidenceCoverageRun,
    coverage_unit_ids,
    coverage_units,
    seal_coverage_run,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    coverage_unit,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioDeclaredPathReport,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioUpdatePositions,
    PortfolioUpdatePublication,
    portfolio_update_positions,
)
from alphalattice.investment.portfolio_strategy_lab.application.finalization import (
    ValidatedPortfolioHandoff,
)
from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
    PortfolioReplayReceipt,
)
from alphalattice.investment.portfolio_strategy_lab.publication.finalization_ledger import (
    PortfolioFinalizationStore,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PortfolioLedgerStore,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    OpenIssueState,
    PortfolioReviewAnswer,
    PortfolioReviewCoverage,
    PortfolioReviewDossier,
    PortfolioReviewDossierCitation,
    PortfolioReviewDossierFinding,
    PortfolioReviewDossierIssuer,
    PortfolioReviewEvidenceChild,
    PortfolioReviewOpenIssue,
    PortfolioReviewPublication,
    PortfolioReviewUnresolvedQuestion,
    finding_aliases,
    review_dispositions,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    POSITION_EPSILON,
    AdmittedListingTickerAuthority,
    BookAuthority,
    PortfolioExperimentReviewSubject,
    PortfolioExposureProjection,
    PortfolioIssuerScope,
    PortfolioListingPosition,
    PortfolioUpdateReviewSubject,
    book_key,
    seal_portfolio_evidence_contract,
    transition_of,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.coverage.issuer_scope import (
    compile_portfolio_issuer_scope,
)

COMPLETION_SCHEMA = "issuer-review-state-v1"
"""The dossier's completion rule when every child brief carries a completion:
an issuer is reviewed only when its checks executed; a source-missing,
unread, unreported or incomplete issuer stays in the book's denominators and
is named. Absent when every child brief predates completions (the legacy
rule: each issuer of the request counted as reviewed)."""

MIXED_COMPLETION_SCHEMA = "issuer-review-state-v1/legacy-children"
"""Some children carry completions and some predate them. The rule is applied
per issuer: a legacy child's issuers keep the legacy reading and are marked
`UNSTATED`; a versioned child's issuers are reviewed only when executed. One
legacy child never upgrades another child's known missing or incomplete
state."""


def _review_states(
    completion: AnalysisCompletion | None, entities: tuple[str, ...]
) -> dict[str, str]:
    """Each issuer's review state from its analysis's completion, or `UNSTATED`."""
    if completion is None:
        return {entity: "UNSTATED" for entity in entities}
    states = {}
    for entity in entities:
        state = completion.state_of(entity)
        states[entity] = "UNSTATED" if state is None else str(state)
    return states


def _executed(states: dict[str, str]) -> set[str]:
    """Return issuers treated as reviewed under their own completion schema.

    A legacy UNSTATED issuer uses the legacy rule. A versioned issuer counts
    only when its checks executed; another child's legacy state never upgrades it.
    """
    executed = {s.value for s in EXECUTED_STATES}
    return {entity for entity, value in states.items() if value == "UNSTATED" or value in executed}


def _completion_schema(states: dict[str, str]) -> str:
    """Which rule the dossier's coverage followed, for its readers."""
    unstated = sum(1 for value in states.values() if value == "UNSTATED")
    if not states or unstated == len(states):
        return ""
    return COMPLETION_SCHEMA if unstated == 0 else MIXED_COMPLETION_SCHEMA


EVIDENCE_REQUIRED_CHECKS: tuple[str, ...] = (
    "supporting evidence",
    "contradicting evidence",
    "cutoff and source limitations",
    "whether human review is required",
)

EVIDENCE_QUESTION = (
    "For these issuers, in the order given, identify issuer-specific downside, "
    "contradiction, uncertainty, or de-risk evidence in what each filed recently: the "
    "approved sources in the packet, at or before the cutoff. Do not infer Portfolio "
    "weights, Alpha, ranking, or activation."
)
"""The one question every unit asks. The same words for one unit or seven: a
unit's obligation differs from another's in its issuers and cutoff alone."""

REVIEW_CLAIM_LIMITS: tuple[str, ...] = (
    "Limited to the reviewed issuers, the approved source families and the stated cutoff.",
    "Absence of adverse evidence is not evidence of absence; unreviewed holdings are named.",
    "No route here is an activation, an order, or a change to any weight.",
    "Nothing here claims a risk is priced in, captured by Alpha, or represented by "
    "volatility or covariance.",
)
"""Carried onto every recommendation, including `NO_MATERIAL_OBJECTION`."""


class PortfolioEvidenceReviewError(ValueError):
    """The review could not be assembled exactly and admissibly."""


class PortfolioReviewInputIncomplete(PortfolioEvidenceReviewError):
    """Retain the sealed book when its review projection lacks required inputs."""

    def __init__(self, book: SealedBook) -> None:
        """Attach the book whose inputs prevented a complete review.

        Args:
            book: Sealed book requiring additional inputs for projection.
        """
        super().__init__("product_host.evidence_review_input_incomplete")
        self.book = book


@dataclass(frozen=True, slots=True)
class SealedBook:
    """One book a review may be about, and the authority that sealed it."""

    authority: BookAuthority
    report: PortfolioDeclaredPathReport | None
    result_hash: str | None
    candidate_hash: str | None
    handoff_hash: str | None
    update_subject: PortfolioUpdateReviewSubject | None = None
    update_positions: PortfolioUpdatePositions | None = None
    update_listing_ids: tuple[str, ...] = ()
    experiment_subject: PortfolioExperimentReviewSubject | None = None
    experiment_positions: tuple[PortfolioListingPosition, ...] | None = None
    experiment_effective_n: float = 0.0
    temporal_statements: tuple[str, ...] = ()
    """What the book's window can claim about time, from its readback's Panel marks:
    T0, the initial cohort, survivorship, the Sector treatment and the price basis."""

    @property
    def report_hash(self) -> str | None:
        """Return the authored report's seal when this book carries a report.

        Returns:
            Report hash, or None for a book without an authored report.
        """
        return None if self.report is None else str(self.report.report_hash)


@dataclass(frozen=True, slots=True)
class BookSelector:
    """One complete, mutually exclusive reference to a sealed book."""

    result_hash: str | None = None
    handoff_hash: str | None = None
    update_task_id: UUID | None = None
    update_publication_hash: str | None = None
    position_basis: str | None = None
    experiment_task_id: UUID | None = None
    experiment_receipt_hash: str | None = None
    portfolio_session: str | None = None

    def request_fields(self) -> dict[str, str]:
        """The book as an operation document names it: each field it sets, as text.

        Returns:
            Only selected operation fields, each serialized as text.
        """
        return {key: str(value) for key, value in asdict(self).items() if value is not None}

    def __post_init__(self) -> None:
        """Validate complete update and experiment selectors and their exclusivity.

        Raises:
            PortfolioEvidenceReviewError: An update or experiment selector is
                incomplete, its position basis is invalid, or selectors conflict.
        """
        update = (self.update_task_id, self.update_publication_hash, self.position_basis)
        experiment = (self.experiment_task_id, self.experiment_receipt_hash, self.portfolio_session)
        named = [
            name
            for name, value in zip(
                ("experiment_task_id", "experiment_receipt_hash", "portfolio_session"),
                experiment,
                strict=True,
            )
            if value is not None
        ]
        another = (self.result_hash, self.handoff_hash, self.update_task_id)
        if named and len(named) < len(experiment) and any(v is not None for v in another):
            # A study's parts beside another book do not apply: named, so the request without
            # them is offered, as a stray position basis is.
            raise PortfolioEvidenceReviewError(
                "product_host.evidence_review_experiment_selector_invalid:" + ",".join(named)
            )
        if any(v is not None for v in experiment) and any(v is None for v in experiment):
            raise PortfolioEvidenceReviewError(
                "product_host.evidence_review_experiment_selector_invalid"
            )
        if (
            self.position_basis is not None
            and self.update_task_id is None
            and self.update_publication_hash is None
        ):
            # A position basis names an update's book only; beside another book it is refused
            # by its name, with the request that works.
            raise PortfolioEvidenceReviewError(
                "product_host.evidence_review_update_selector_invalid:position_basis"
            )
        if any(v is not None for v in update) and (
            any(v is None for v in update)
            or self.position_basis not in {"CONDITIONAL_ESTIMATE", "OBSERVED_RESEARCH_ENTRY"}
        ):
            raise PortfolioEvidenceReviewError(
                "product_host.evidence_review_update_selector_invalid"
            )
        if (
            sum(
                v is not None
                for v in (
                    self.result_hash,
                    self.handoff_hash,
                    self.update_task_id,
                    self.experiment_task_id,
                )
            )
            > 1
        ):
            raise PortfolioEvidenceReviewError("product_host.evidence_review_selector_ambiguous")


@dataclass(frozen=True, slots=True)
class PortfolioEvidenceReviewInputs:
    """Everything the review reads, all of it already exact."""

    ledger: PortfolioLedgerStore | None
    finalization: PortfolioFinalizationStore | None
    registry: SecIssuerRegistrySnapshot | None
    listing_authority: AdmittedListingTickerAuthority | None
    portfolio_report_link: str
    handoff: ValidatedPortfolioHandoff | None = None
    """A handoff the host was given, beside those the finalization store holds."""
    read_update: Callable[[UUID, str], dict[str, object]] | None = None
    read_experiment: Callable[[UUID, str], dict[str, object]] | None = None
    installed_temporal_statements: Callable[[date, date], tuple[str, ...]] | None = None
    """The installed strategy's research input stated for a window: what an update, a
    handoff or a result book can claim about time."""


def _installed_statements(
    inputs: PortfolioEvidenceReviewInputs, start: date, end: date
) -> tuple[str, ...]:
    reader = inputs.installed_temporal_statements
    return () if reader is None else reader(start, end)


def open_sealed_book(inputs: PortfolioEvidenceReviewInputs, selector: BookSelector) -> SealedBook:
    """Open exactly one sealed book and name its authority.

    A handoff book is opened through the handoff's released report. A result is
    a development book unless the finalization store holds a candidate frozen
    from it, in which case it is a frozen candidate. The report validates its
    own identity on load, so the chain selector -> report -> book is checked by
    the owners that wrote each link.

    Args:
        inputs: Sealed stores, read callbacks, and admitted authority for the review.
        selector: One complete reference to the book to open.

    Returns:
        Verified book and the authority that sealed it.

    Raises:
        PortfolioEvidenceReviewError: The selected book is absent, its reader is
            unavailable, or its publication, report, position basis, or axis disagrees.
    """
    if selector.experiment_task_id is not None:
        return _open_experiment_book(inputs, selector)
    if selector.update_task_id is not None:
        if inputs.read_update is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_review_update_not_configured")
        assert selector.update_publication_hash is not None
        body = inputs.read_update(selector.update_task_id, selector.update_publication_hash)
        value = PortfolioUpdatePublication.model_validate(body.get("publication"))
        if value.content_hash != selector.update_publication_hash:
            raise PortfolioEvidenceReviewError(
                "product_host.evidence_review_update_publication_mismatch"
            )
        history = tuple(
            PortfolioUpdatePublication.model_validate(v)
            for v in cast(list[object], body.get("history", []))
        )
        positions = portfolio_update_positions(value, history)
        if positions.basis != selector.position_basis:
            raise PortfolioEvidenceReviewError("product_host.evidence_review_update_basis_mismatch")
        labels = body["listing_labels"]
        if not isinstance(labels, dict) or len(labels) != len(positions.weights):
            raise PortfolioEvidenceReviewError("product_host.evidence_review_update_axis_mismatch")
        subject = PortfolioUpdateReviewSubject(
            update_task_id=UUID(str(body["task_id"])),
            update_publication_hash=value.content_hash,
            position_basis=positions.basis,
            position_hash=positions.position_hash,
            checkpoint_hash=value.input_checkpoint_hash or value.checkpoint_hash,
            strategy_package_id=str(body["strategy_package_id"]),
            observed_through=value.observed_through.isoformat(),
            formation_session=positions.schedule.formation_session.isoformat(),
            entry_session=positions.schedule.entry_session.isoformat(),
        )
        formation = positions.schedule.formation_session
        return SealedBook(
            subject.book_authority,
            None,
            None,
            None,
            None,
            subject,
            positions,
            tuple(labels),
            temporal_statements=_installed_statements(inputs, formation, formation),
        )
    if selector.handoff_hash is not None:
        if inputs.ledger is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_review_handoff_unknown")
        handoff = inputs.handoff
        if handoff is None or handoff.handoff_hash != selector.handoff_hash:
            if inputs.finalization is None:
                raise PortfolioEvidenceReviewError("product_host.evidence_review_handoff_unknown")
            try:
                handoff = inputs.finalization.load_handoff(selector.handoff_hash)
            except ValueError as error:
                _refuse_if_absent(error, "product_host.evidence_review_handoff_unknown")
                raise
        report = inputs.ledger.load_report(handoff.released_report_hash)
        if report.report_hash != handoff.released_report_hash:
            raise PortfolioEvidenceReviewError("product_host.evidence_review_report_mismatch")
        return SealedBook(
            authority=BookAuthority.VALIDATED_HANDOFF,
            report=report,
            result_hash=None,
            candidate_hash=handoff.candidate_hash,
            handoff_hash=handoff.handoff_hash,
            temporal_statements=_installed_statements(
                inputs, report.window_guard.selected_start, report.window_guard.selected_end
            ),
        )
    if selector.result_hash is None:
        raise PortfolioEvidenceReviewError("product_host.evidence_review_book_unselected")
    if inputs.ledger is None:
        raise PortfolioEvidenceReviewError("product_host.evidence_review_result_unknown")
    try:
        result = inputs.ledger.load_result(selector.result_hash)
    except ValueError as error:
        _refuse_if_absent(error, "product_host.evidence_review_result_unknown")
        raise
    report = inputs.ledger.load_report(result.report_hash)
    candidate_hash = None
    if inputs.finalization is not None:
        candidate_hash = inputs.finalization.find_candidate_for_result(selector.result_hash)
    return SealedBook(
        authority=(
            BookAuthority.FROZEN_CANDIDATE
            if candidate_hash is not None
            else BookAuthority.DEVELOPMENT_RESULT
        ),
        report=report,
        result_hash=selector.result_hash,
        candidate_hash=candidate_hash,
        handoff_hash=None,
        temporal_statements=_installed_statements(
            inputs, report.window_guard.selected_start, report.window_guard.selected_end
        ),
    )


def _refuse_if_absent(error: ValueError, code: str) -> None:
    """Refuse a book the workspace does not hold by its own code.

    The content-store owner identifies absence; a present artifact that does not validate
    keeps its corruption refusal. No caller inspects a filesystem cause.
    """
    if str(error).partition(":")[0] == "content_store.artifact_missing":
        raise PortfolioEvidenceReviewError(code) from error


def _open_experiment_book(
    inputs: PortfolioEvidenceReviewInputs, selector: BookSelector
) -> SealedBook:
    if inputs.read_experiment is None:
        raise PortfolioEvidenceReviewError("product_host.evidence_review_experiment_not_configured")
    assert selector.experiment_task_id is not None and selector.portfolio_session is not None
    body = inputs.read_experiment(selector.experiment_task_id, selector.portfolio_session)
    if body.get("status") != "EXPERIMENT_PUBLISHED" or body.get("task_id") != str(
        selector.experiment_task_id
    ):
        raise PortfolioEvidenceReviewError("product_host.evidence_review_experiment_not_published")
    receipt = PortfolioReplayReceipt.model_validate(body.get("receipt"))
    if receipt.receipt_hash != selector.experiment_receipt_hash:
        raise PortfolioEvidenceReviewError(
            "product_host.evidence_review_experiment_receipt_mismatch"
        )
    selected = cast(dict[str, object], body["position"])
    previous = cast(dict[str, object] | None, body["preceding_position"])
    day = date.fromisoformat(selector.portfolio_session)
    index = receipt.source.formation_sessions.index(day)
    if selected["session"] != selector.portfolio_session or (
        (previous is None) != (index == 0)
        or (
            previous is not None
            and previous["session"] != receipt.source.formation_sessions[index - 1].isoformat()
        )
    ):
        raise PortfolioEvidenceReviewError("product_host.evidence_review_experiment_date_mismatch")
    weights = cast(list[float], selected["weights"])
    before = None if previous is None else cast(list[float], previous["weights"])
    document = cast(dict[str, object], body["document"])
    experiment = cast(dict[str, object], document["experiment"])
    as_of = cast(dict[str, object], cast(dict[str, object], experiment["sessions"])["as_of"])
    subject = PortfolioExperimentReviewSubject(
        experiment_task_id=selector.experiment_task_id,
        experiment_receipt_hash=receipt.receipt_hash,
        program_hash=receipt.program_hash,
        input_binding_hash=receipt.source.input_binding_hash,
        portfolio_session=day,
        preceding_session=None if index == 0 else receipt.source.formation_sessions[index - 1],
        research_as_of_session=date.fromisoformat(str(as_of["session"])),
        research_as_of_phase=cast("Literal['OPEN', 'OFFICIAL_CLOSE']", as_of["phase"]),
        position_hash=str(
            canonical_hash(
                {
                    "receipt": receipt.receipt_hash,
                    "session": selected["session"],
                    "listing_ids": receipt.source.ordered_listing_ids,
                    "weights": weights,
                    "preceding": before,
                }
            )
        ),
    )
    positions = (
        None
        if before is None
        else tuple(
            PortfolioListingPosition(
                listing_id=listing,
                ending_weight=ending,
                preceding_weight=preceding,
                signed_change=ending - preceding,
                transition=transition_of(ending=ending, preceding=preceding),
            )
            for listing, ending, preceding in zip(
                receipt.source.ordered_listing_ids, weights, before, strict=True
            )
            if ending > 0 or preceding > 0
        )
    )
    hhi = float(cast(float, selected["hhi"]))
    scope = cast(
        dict[str, object],
        cast(dict[str, object], body.get("timing") or {}).get("temporal_scope") or {},
    )
    return SealedBook(
        BookAuthority.DEVELOPMENT_RESULT,
        None,
        None,
        None,
        None,
        experiment_subject=subject,
        experiment_positions=positions,
        experiment_effective_n=1.0 / hhi if hhi > 0 else 0.0,
        temporal_statements=(
            tuple(str(value) for value in cast(list[object], scope.get("statements") or []))
            if scope.get("status") == "RECORDED"
            else ()
        ),
    )


def book_listing_count(book: SealedBook) -> int:
    """How many listings the book's review maps, the rows `project_portfolio_exposure` keeps.

    It needs no listing authority, so a scope is said before any source package exists.
    """
    if book.experiment_positions is not None:
        return len(book.experiment_positions)
    if book.update_positions is not None:
        facts = book.update_positions
        before = facts.preceding or tuple(0.0 for _ in facts.weights)
        rows = zip(facts.weights, before, strict=True)
        return sum(ending > 0 or prior > 0 for ending, prior in rows)
    assert book.report is not None
    return len(book.report.window_end_book.positions)


def project_portfolio_exposure(
    inputs: PortfolioEvidenceReviewInputs, book: SealedBook
) -> PortfolioExposureProjection:
    """Read the sealed report's window-end book. Nothing is recomputed.

    Every row survives. A listing the admitted authority carries no symbol for
    keeps its weight and its change and travels on with `ticker=None`, so it
    lands in the mapping gaps and in the coverage denominators.

    Args:
        inputs: Review stores and admitted listing authority.
        book: Sealed authored, update, or experiment book to project.

    Returns:
        Content-bound listing positions, changes, and concentration from sealed facts.

    Raises:
        PortfolioEvidenceReviewError: Listing authority is absent or exposure is empty.
        PortfolioReviewInputIncomplete: The book lacks inputs required for projection.
    """
    if inputs.listing_authority is None:
        raise PortfolioEvidenceReviewError("product_host.evidence_review_authority_absent")
    if book.experiment_subject is not None:
        if book.experiment_positions is None:
            raise PortfolioReviewInputIncomplete(book)
        tickers = inputs.listing_authority.ticker_by_listing
        return seal_portfolio_evidence_contract(
            PortfolioExposureProjection,
            "projection_hash",
            book_authority=book.authority,
            report_hash=None,
            experiment_subject=book.experiment_subject,
            window_end_book_hash=book.experiment_subject.position_hash,
            listing_authority_hash=inputs.listing_authority.authority_hash,
            formation_session=book.experiment_subject.portfolio_session.isoformat(),
            change_boundary=book.experiment_subject.position_basis,
            held_count=sum(p.ending_weight > 0 for p in book.experiment_positions),
            window_end_effective_n=book.experiment_effective_n,
            positions=tuple(
                p.model_copy(update={"ticker": tickers.get(p.listing_id)})
                for p in book.experiment_positions
            ),
        )
    if book.update_subject is not None:
        facts = book.update_positions
        assert facts is not None
        if facts.preceding is None:
            raise PortfolioReviewInputIncomplete(book)
        tickers = inputs.listing_authority.ticker_by_listing
        rows = tuple(
            PortfolioListingPosition(
                listing_id=listing,
                ticker=tickers.get(listing),
                ending_weight=ending,
                preceding_weight=before,
                signed_change=ending - before,
                transition=transition_of(ending=ending, preceding=before),
            )
            for listing, ending, before in zip(
                book.update_listing_ids, facts.weights, facts.preceding, strict=True
            )
            if ending > 0 or before > 0
        )
        return seal_portfolio_evidence_contract(
            PortfolioExposureProjection,
            "projection_hash",
            book_authority=book.authority,
            report_hash=None,
            update_subject=book.update_subject,
            window_end_book_hash=facts.position_hash,
            listing_authority_hash=inputs.listing_authority.authority_hash,
            formation_session=book.update_subject.formation_session,
            change_boundary=facts.basis,
            held_count=sum(v > 0 for v in facts.weights),
            window_end_effective_n=facts.effective_n,
            positions=rows,
        )
    assert book.report is not None
    window_end = book.report.window_end_book
    tickers = inputs.listing_authority.ticker_by_listing
    positions = tuple(
        PortfolioListingPosition(
            listing_id=position.listing_id,
            ticker=tickers.get(position.listing_id),
            ending_weight=position.weight,
            preceding_weight=position.preceding_weight,
            signed_change=position.weight_change,
            transition=transition_of(ending=position.weight, preceding=position.preceding_weight),
        )
        for position in window_end.positions
    )
    if not positions:
        raise PortfolioEvidenceReviewError("product_host.evidence_review_exposure_empty")
    return seal_portfolio_evidence_contract(
        PortfolioExposureProjection,
        "projection_hash",
        book_authority=book.authority,
        report_hash=book.report_hash,
        update_subject=book.update_subject,
        result_hash=book.result_hash,
        window_end_book_hash=window_end.book_hash,
        listing_authority_hash=inputs.listing_authority.authority_hash,
        candidate_hash=book.candidate_hash,
        handoff_hash=book.handoff_hash,
        formation_session=window_end.formation_session.isoformat(),
        change_boundary=window_end.change_boundary,
        held_count=window_end.held_count,
        window_end_effective_n=float(book.report.window_end_effective_n),
        positions=positions,
    )


def compile_portfolio_scope(
    inputs: PortfolioEvidenceReviewInputs, book: SealedBook
) -> tuple[PortfolioExposureProjection, PortfolioIssuerScope]:
    """Project the book and map it, in the one order that binds the lineage.

    Args:
        inputs: Review inputs including listing and issuer mapping authority.
        book: Verified book whose exposure and issuer scope are requested.

    Returns:
        Exposure projection and its admitted issuer mapping.

    Raises:
        PortfolioEvidenceReviewError: Review authority or sealed exposure is unavailable.
    """
    if inputs.registry is None:
        raise PortfolioEvidenceReviewError("product_host.evidence_review_authority_absent")
    projection = project_portfolio_exposure(inputs, book)
    return projection, compile_portfolio_issuer_scope(
        projection=projection, registry=inputs.registry
    )


def project_unit_obligation(
    *,
    ordered_entity_ids: tuple[str, ...],
    evidence_as_of: datetime,
    approved_source_families: tuple[str, ...],
) -> AlternativeEvidenceResearchObligation:
    """The only thing Alternative Evidence learns about one unit of this book.

    Args:
        ordered_entity_ids: Issuers to research in their admitted order.
        evidence_as_of: Latest availability time admitted for evidence.
        approved_source_families: Source families the unit may read.

    Returns:
        Sealed research question and checks without Portfolio weights or positions.
    """
    return seal_research_obligation(
        question=EVIDENCE_QUESTION,
        ordered_entity_ids=ordered_entity_ids,
        evidence_as_of=evidence_as_of,
        approved_source_families=approved_source_families,
        required_checks=EVIDENCE_REQUIRED_CHECKS,
    )


def project_coverage_run(
    *,
    scope: PortfolioIssuerScope,
    evidence_as_of: datetime,
    approved_source_families: tuple[str, ...],
    request_for: Callable[
        [tuple[str, ...], datetime, tuple[AlternativeEvidenceReadFiling, ...]],
        AlternativeEvidenceRequest,
    ],
    resource_binding_hash: str,
    network_consent: bool,
    admit_live_official: bool,
    admit_model_review: bool,
    unit_limit: int = UNIT_LIMIT,
    source_counts: Mapping[str, int] | None = None,
    nothing_filed: frozenset[str] = frozenset(),
    carried: frozenset[str] = frozenset(),
    read_filings: tuple[AlternativeEvidenceReadFiling, ...] = (),
) -> AlternativeEvidenceCoverageRun:
    """Every unit this book needs at this cutoff, sealed as one run.

    With each issuer's logical source count (`source_counts`,
    `UNIT_PACKING_RULES_ID`) the book is packed heaviest first, and the issuers
    whose filing index at the cutoff holds nothing in the window
    (`nothing_filed`) or nothing an earlier analysis did not read (`carried`)
    are named by the run instead of packed; without counts it is cut at the
    issuer limit alone. Each unit's request names what an earlier analysis
    read of its issuers (`read_filings`), so only what is new is read; a book
    with nothing new is a run of no unit. The run's identity has no clock of
    its own beyond the cutoff, so a preview's captured intent and its later
    submission are one run; the counts it was packed from are part of it.

    Args:
        scope: Admitted issuers and their coverage weights.
        evidence_as_of: Evidence cutoff shared by all units.
        approved_source_families: Source families each unit may read.
        request_for: Build a unit request from issuers, cutoff, and previously read filings.
        resource_binding_hash: Installed Evidence resource authority captured by the run.
        network_consent: Whether the run has network consent.
        admit_live_official: Whether live official acquisition is admitted.
        admit_model_review: Whether model review is admitted.
        unit_limit: Maximum issuers admitted to one unit.
        source_counts: Logical source counts used for source-aware packing, when available.
        nothing_filed: Issuers with no filing in the window.
        carried: Issuers whose relevant filings were already read.
        read_filings: Earlier read filings attached to each unit's request.

    Returns:
        Sealed run naming packed units and issuers whose earlier or absent filings suffice.

    Raises:
        PortfolioEvidenceReviewError: The admitted issuer scope is empty.
    """
    if not scope.selected_issuers:
        raise PortfolioEvidenceReviewError("product_host.evidence_review_scope_empty")
    units = []
    cut = coverage_units(
        scope.ordered_entity_ids,
        priority_rank=scope.priority_rank,
        unit_limit=unit_limit,
        source_counts=source_counts,
        weight_rank=(
            None
            if source_counts is None
            else {issuer.entity_id: issuer.weight_rank for issuer in scope.selected_issuers}
        ),
        nothing_filed=nothing_filed,
        carried=carried,
    )
    ids = coverage_unit_ids(cut) if cut else {}
    read = tuple(sorted(read_filings, key=lambda value: (value.entity_id, value.accession)))
    for entities in cut:
        request = request_for(
            entities,
            evidence_as_of,
            tuple(value for value in read if value.entity_id in entities),
        )
        obligation = project_unit_obligation(
            ordered_entity_ids=entities,
            evidence_as_of=evidence_as_of,
            approved_source_families=approved_source_families,
        )
        units.append(
            coverage_unit(
                unit_id=ids[entities],
                ordered_entity_ids=entities,
                request=request,
                obligation=obligation,
                resource_binding_hash=resource_binding_hash,
                network_consent=network_consent,
                admit_live_official=admit_live_official,
                admit_model_review=admit_model_review,
            )
        )
    return seal_coverage_run(
        scope_hash=scope.scope_hash,
        evidence_as_of=evidence_as_of,
        unit_limit=unit_limit,
        resource_binding_hash=resource_binding_hash,
        network_consent=network_consent,
        admit_live_official=admit_live_official,
        admit_model_review=admit_model_review,
        units=tuple(units),
        packing_rules_id=None if source_counts is None else UNIT_PACKING_RULES_ID,
        source_counts=(
            ()
            if source_counts is None
            else tuple(
                sorted((entity, source_counts[entity]) for entity in scope.ordered_entity_ids)
            )
        ),
        nothing_filed=tuple(sorted(nothing_filed)),
        carried=tuple(sorted(carried)),
        read_filings=read,
    )


def compile_portfolio_review_dossier(
    *,
    inputs: PortfolioEvidenceReviewInputs,
    book: SealedBook,
    projection: PortfolioExposureProjection,
    scope: PortfolioIssuerScope,
    obligation: AlternativeEvidenceResearchObligation,
    evidence: AlternativeEvidenceAnalysisPublicationView,
) -> PortfolioReviewDossier:
    """Assemble the one dossier, refusing anything that does not line up exactly.

    Args:
        inputs: Review authorities and report link.
        book: Verified book being reviewed.
        projection: Exposure projection of that book.
        scope: Admitted issuer mapping of the projection.
        obligation: Exact evidence research obligation for that scope.
        evidence: Verified publication answering the obligation.

    Returns:
        Sealed dossier joining the book, issuer exposure, findings, and citation lineage.

    Raises:
        PortfolioEvidenceReviewError: Scope, obligation, issuer axis, or evidence
            standing disagrees with the requested review.
    """
    publication = evidence.publication
    lineage = evidence.lineage
    if not evidence.is_current:
        raise PortfolioEvidenceReviewError("product_host.evidence_review_evidence_not_current")
    if publication.obligation_hash != obligation.obligation_hash:
        raise PortfolioEvidenceReviewError("product_host.evidence_review_obligation_mismatch")
    if tuple(lineage.request.ordered_entity_ids) != scope.ordered_entity_ids:
        raise PortfolioEvidenceReviewError("product_host.evidence_review_axis_mismatch")
    if scope.exposure_projection_hash != projection.projection_hash:
        raise PortfolioEvidenceReviewError("product_host.evidence_review_scope_mismatch")

    package = lineage.cro_package
    citations = tuple(
        PortfolioReviewDossierCitation(
            span_handle=span.span_handle,
            document_handle=span.document_handle,
            entity_id=span.entity_id,
            available_at=span.available_at,
        )
        for span in package.verified_spans
    )
    findings = []
    for finding in lineage.brief.findings:
        structure = package.structure(finding.finding_handle)
        findings.append(
            PortfolioReviewDossierFinding(
                finding_handle=finding.finding_handle,
                affected_entities=finding.affected_entities,
                topic=finding.topic,
                lifecycle=finding.lifecycle,
                direction=finding.direction,
                summary=finding.summary,
                supporting_span_handles=finding.supporting_span_handles,
                contradicting_span_handles=finding.contradicting_span_handles,
                limitations=finding.limitations,
                supporting_document_count=structure.supporting_document_count,
                contradicting_document_count=structure.contradicting_document_count,
                structure=structure.state,
            )
        )
    states = _review_states(lineage.brief.completion, scope.ordered_entity_ids)
    executed = _executed(states)
    versioned = all(value != "UNSTATED" for value in states.values())
    issuers = tuple(
        PortfolioReviewDossierIssuer(
            entity_id=issuer.entity_id,
            tickers=issuer.tickers,
            ending_weight=issuer.ending_weight,
            signed_change=issuer.signed_change,
            transition=issuer.transition,
            weight_rank=issuer.weight_rank,
            exposure_band=issuer.exposure_band,
            selection_reason=issuer.selection_reason,
            review_state=states[issuer.entity_id],
        )
        for issuer in scope.selected_issuers
    )
    unexecuted = tuple(
        f"{issuer.entity_id} is not reviewed: {states[issuer.entity_id]}"
        for issuer in scope.selected_issuers
        if issuer.entity_id not in executed
    )
    executed_listings = {
        listing
        for issuer in scope.selected_issuers
        if issuer.entity_id in executed
        for listing in issuer.listing_ids
    }
    coverage = PortfolioReviewCoverage(
        reviewed_ending_weight_coverage=_ratio(
            math.fsum(
                value.ending_weight
                for value in projection.positions
                if value.listing_id in executed_listings
            ),
            math.fsum(value.ending_weight for value in projection.positions),
        )
        if versioned
        else scope.reviewed_ending_weight_coverage,
        reviewed_absolute_change_coverage=_ratio(
            math.fsum(
                abs(value.signed_change)
                for value in projection.positions
                if value.listing_id in executed_listings
            ),
            math.fsum(abs(value.signed_change) for value in projection.positions),
        )
        if versioned
        else scope.reviewed_absolute_change_coverage,
        mapping_coverage=scope.mapping_coverage,
        selected_issuer_coverage=_ratio(float(len(executed)), float(scope.mapped_issuer_count))
        if versioned
        else scope.selected_issuer_coverage,
        missing_evidence=package.missing_evidence,
        unavailable_reasons=(*scope.unavailable_reasons, *unexecuted),
    )
    return seal_contract(
        PortfolioReviewDossier,
        "dossier_hash",
        book_authority=book.authority,
        report_hash=book.report_hash,
        update_subject=book.update_subject,
        experiment_subject=book.experiment_subject,
        result_hash=book.result_hash,
        candidate_hash=book.candidate_hash,
        handoff_hash=book.handoff_hash,
        issuer_scope_hash=scope.scope_hash,
        exposure_projection_hash=projection.projection_hash,
        registry_hash=scope.registry_hash,
        analysis_publication_hash=publication.publication_hash,
        analyst_brief_hash=publication.analyst_brief_hash,
        cro_package_hash=publication.cro_package_hash,
        obligation_hash=obligation.obligation_hash,
        evidence_as_of=lineage.request.evidence_as_of,
        evidence_expires_at=publication.expires_at,
        issuers=issuers,
        findings=tuple(findings),
        citations=citations,
        coverage=coverage,
        analyst_requires_human_review=lineage.brief.requires_human_review,
        mapping_failure_count=len(scope.mapping_failures),
        held_count=projection.held_count,
        window_end_effective_n=projection.window_end_effective_n,
        claim_limits=REVIEW_CLAIM_LIMITS + _book_claim_limits(book),
        limitations=(*package.limitations, *book.temporal_statements),
        portfolio_report_link=_portfolio_report_link(inputs, book),
        unresolved_questions=tuple(
            PortfolioReviewUnresolvedQuestion(unit_id="u01", text=value[:600])
            for value in lineage.brief.unresolved_questions
        ),
        completion_schema=COMPLETION_SCHEMA if versioned else "",
    )


def _book_claim_limits(book: SealedBook) -> tuple[str, ...]:
    return (
        ()
        if book.experiment_subject is None
        else (
            "POST_OBSERVED_DEVELOPMENT_NOT_INDEPENDENT_VALIDATION; executed research "
            "holdings are not venue execution or a current recommendation.",
            "Evidence has its own cutoff; later evidence is retrospective, not "
            "decision-time support.",
        )
    ) + (
        ()
        if book.update_subject is None
        else (
            "POST_OBSERVED_QA_NOT_TIMELY_ADVICE; no protected validation "
            "or verified venue execution.",
            "Review evidence has its own cutoff; evidence after the decision is "
            "retrospective, not decision-time support.",
            f"Position basis: {book.update_subject.position_basis}; "
            f"decision formation: {book.update_subject.formation_session}.",
        )
    )


def _portfolio_report_link(inputs: PortfolioEvidenceReviewInputs, book: SealedBook) -> str:
    if book.experiment_subject is not None:
        return (
            f"/?review_experiment={book.experiment_subject.experiment_task_id}"
            f"&r={book.experiment_subject.experiment_receipt_hash}"
            f"&d={book.experiment_subject.portfolio_session}"
        )
    if book.update_subject is None:
        return inputs.portfolio_report_link
    return (
        f"/?review_update={book.update_subject.update_task_id}"
        f"&p={book.update_subject.update_publication_hash}&b={book.update_subject.position_basis}"
    )


def _qualifier(publication_hash: str) -> str:
    return f"P{publication_hash[:8].upper()}"


def _qualified_finding(handle: str, qualifier: str) -> str:
    return f"FIND-{qualifier}-{handle.removeprefix('FIND-')}"


def _qualified_span(handle: str, qualifier: str) -> str:
    return f"{qualifier}:{handle}"


def qualify_span_handle(handle: str, *, publication_hash: str) -> str:
    """The handle a child publication's span carries in an aggregate dossier.

    Args:
        handle: Child publication's local span handle.
        publication_hash: Publication identity used to qualify the handle.

    Returns:
        Span handle qualified for use alongside other child publications.
    """
    return _qualified_span(handle, _qualifier(publication_hash))


def _ratio(part: float, whole: float) -> float:
    """The scope's own rule: coverage over an empty book is zero, not one."""
    if whole <= POSITION_EPSILON:
        return 0.0
    return min(1.0, part / whole)


@dataclass(frozen=True, slots=True)
class CarriedReading:
    """Carry an earlier analysis's relevant readings into the current review.

    Findings on filings the run names as read carry at its cutoff (W3),
    and the analysis answers for the carried holdings it read latest.
    """

    evidence: AlternativeEvidenceAnalysisPublicationView
    filings: frozenset[tuple[str, str]]
    """(issuer, accession) of every filing it read that the run names."""
    entities: tuple[str, ...]
    """The carried holdings whose review state it states; none when every
    holding it read was read again at the cutoff."""
    as_of: datetime
    """The run's cutoff: nothing new was filed by then."""
    expires_at: datetime
    """The day its earliest carried filing leaves the window: a carried
    reading is current while its filings stay in it, whatever its analysis's
    own expiry (Z1)."""
    issue_findings: frozenset[str] = frozenset()
    """Findings, by the handle its brief states, that an open issue of the
    issuers' register rests on: carried whether or not their filing is still in
    the window, since the issue does not age out with it (W3)."""


EARLIER_METHOD_NOTE = (
    "Read under an earlier version of the Evidence method: an open issue rests on it, and it "
    "was not read again under the current one."
)
"""Carried beside each finding of a superseded reading an open issue rests on."""


def carried_reading_admitted(eligibility: str | None, *, open_issue_only: bool) -> bool:
    """Whether a review carries an earlier reading, one rule for the selection and the dossier.

    A reading of filings still in the window is carried while current or expired; one sealed
    under a superseded contract is prepared again, its holdings named until then. A reading an
    open issue alone rests on is carried under any verified contract, a superseded one with
    `EARLIER_METHOD_NOTE` on its findings: the register keeps the issue open until a review
    resolves it (X4), and a review that could not read its basis could never resolve it.

    Args:
        eligibility: Verified standing of the earlier reading, when known.
        open_issue_only: Whether an open issue is the reason to retain the reading.

    Returns:
        Whether the reading may be carried under the shared standing rule.
    """
    if eligibility in {"CURRENT", "EXPIRED"}:
        return True
    return open_issue_only and eligibility == "SUPERSEDED"


def compile_portfolio_coverage_dossier(
    *,
    inputs: PortfolioEvidenceReviewInputs,
    book: SealedBook,
    projection: PortfolioExposureProjection,
    scope: PortfolioIssuerScope,
    children: tuple[
        tuple[
            str, AlternativeEvidenceResearchObligation, AlternativeEvidenceAnalysisPublicationView
        ],
        ...,
    ],
    unreviewed: tuple[tuple[str, tuple[str, ...], str], ...] = (),
    nothing_filed: tuple[str, ...] = (),
    carried: tuple[CarriedReading, ...] = (),
    open_issues: tuple[OpenIssueState, ...] = (),
) -> PortfolioReviewDossier:
    """Assemble the one dossier of a book analysed as several units.

    `carried` are the earlier analyses whose filings in the window no one
    need read again (W3): each is a child named `c01`, `c02`, ... beside the
    units, carrying only its findings on the filings the run names, each
    finding narrowed to the book's holdings; it answers for the carried
    holdings it read latest, and says when it read them (`read_as_of`).

    `open_issues` are the issuers' register's issues open on the book's
    holdings (W3): each is in the dossier with the findings it rests on, as
    the CRO last stated it, so every book that holds the issuer reads it.

    `nothing_filed` are the holdings whose filing index at the cutoff held
    nothing in the window: no unit read them, and they stay in the dossier with
    their weight, stated as nothing filed -- never as no risk -- and counted
    apart from what was read and what was not.

    Each child is one unit (its id, its obligation) with that unit's exact,
    current publication; every finding and span keeps its child's identity in
    its handle, so `SPAN-S01-R01` from two units never collide and a
    reviewer's citation names one span. Coverage is recomputed over the book:
    an issuer is reviewed when its unit's publication is here, and a unit
    without one (`unreviewed`: id, issuers, reason) keeps its listings and
    weight in the denominator. Nothing is averaged and no new severity rule
    appears: the route reads the same findings, structures and coverage
    components it always read.

    One child and nothing unreviewed is the ordinary dossier, bare handles
    and all: the aggregate form exists only where more than one publication
    is read or a unit is missing.

    Args:
        inputs: Review authorities and report link.
        book: Verified book being reviewed.
        projection: Exposure projection of the book.
        scope: Admitted issuer mapping of that projection.
        children: Unit IDs, exact obligations, and their verified evidence publications.
        unreviewed: Unit IDs, issuer axes, and reasons for missing publications.
        nothing_filed: Issuers with no filing in the evidence window.
        carried: Earlier readings and the holdings and findings they still answer for.
        open_issues: Register issues whose evidence remains relevant to this review.

    Returns:
        Sealed dossier with child-qualified citations and coverage over the whole book.

    Raises:
        PortfolioEvidenceReviewError: The scope is empty or a child's scope,
            obligation, issuer axis, or evidence standing disagrees with the review.
    """
    if (
        len(children) == 1
        and not unreviewed
        and not nothing_filed
        and not carried
        and not open_issues
    ):
        _unit_id, obligation, evidence = children[0]
        return compile_portfolio_review_dossier(
            inputs=inputs,
            book=book,
            projection=projection,
            scope=scope,
            obligation=obligation,
            evidence=evidence,
        )
    if not children and not carried:
        raise PortfolioEvidenceReviewError("product_host.evidence_review_scope_empty")
    if scope.exposure_projection_hash != projection.projection_hash:
        raise PortfolioEvidenceReviewError("product_host.evidence_review_scope_mismatch")
    positions = {value.entity_id: value for value in scope.selected_issuers}
    reviewed: list[str] = []
    findings: list[PortfolioReviewDossierFinding] = []
    citations: list[PortfolioReviewDossierCitation] = []
    child_records: list[PortfolioReviewEvidenceChild] = []
    missing: list[str] = []
    limitations: list[str] = []
    requires_human_review = False
    states: dict[str, str] = {}
    questions: list[PortfolioReviewUnresolvedQuestion] = []
    for unit_id, obligation, evidence in children:
        publication = evidence.publication
        lineage = evidence.lineage
        if not evidence.is_current:
            raise PortfolioEvidenceReviewError("product_host.evidence_review_evidence_not_current")
        if publication.obligation_hash != obligation.obligation_hash:
            raise PortfolioEvidenceReviewError("product_host.evidence_review_obligation_mismatch")
        if tuple(lineage.request.ordered_entity_ids) != obligation.ordered_entity_ids:
            raise PortfolioEvidenceReviewError("product_host.evidence_review_axis_mismatch")
        if any(entity not in positions for entity in obligation.ordered_entity_ids):
            raise PortfolioEvidenceReviewError("product_host.evidence_review_scope_mismatch")
        qualifier = _qualifier(publication.publication_hash)
        package = lineage.cro_package
        reviewed.extend(obligation.ordered_entity_ids)
        states.update(_review_states(lineage.brief.completion, obligation.ordered_entity_ids))
        questions.extend(
            PortfolioReviewUnresolvedQuestion(unit_id=unit_id, text=value[:600])
            for value in lineage.brief.unresolved_questions
        )
        citations.extend(
            PortfolioReviewDossierCitation(
                span_handle=_qualified_span(span.span_handle, qualifier),
                document_handle=span.document_handle,
                entity_id=span.entity_id,
                available_at=span.available_at,
            )
            for span in package.verified_spans
        )
        for finding in lineage.brief.findings:
            structure = package.structure(finding.finding_handle)
            findings.append(
                PortfolioReviewDossierFinding(
                    finding_handle=_qualified_finding(finding.finding_handle, qualifier),
                    affected_entities=finding.affected_entities,
                    topic=finding.topic,
                    lifecycle=finding.lifecycle,
                    direction=finding.direction,
                    summary=finding.summary,
                    supporting_span_handles=tuple(
                        _qualified_span(value, qualifier)
                        for value in finding.supporting_span_handles
                    ),
                    contradicting_span_handles=tuple(
                        _qualified_span(value, qualifier)
                        for value in finding.contradicting_span_handles
                    ),
                    limitations=finding.limitations,
                    supporting_document_count=structure.supporting_document_count,
                    contradicting_document_count=structure.contradicting_document_count,
                    structure=structure.state,
                )
            )
        missing.extend(f"{unit_id}: {value}" for value in package.missing_evidence)
        limitations.extend(package.limitations)
        requires_human_review = requires_human_review or lineage.brief.requires_human_review
        child_records.append(
            PortfolioReviewEvidenceChild(
                unit_id=unit_id,
                ordered_entity_ids=obligation.ordered_entity_ids,
                obligation_hash=obligation.obligation_hash,
                analysis_publication_hash=publication.publication_hash,
                analyst_brief_hash=publication.analyst_brief_hash,
                cro_package_hash=publication.cro_package_hash,
                evidence_as_of=lineage.request.evidence_as_of,
                evidence_expires_at=publication.expires_at,
            )
        )
    carried_findings = 0
    for index, reading in enumerate(carried, start=1):
        publication = reading.evidence.publication
        lineage = reading.evidence.lineage
        eligibility = reading.evidence.current_eligibility
        if not carried_reading_admitted(eligibility, open_issue_only=not reading.filings):
            raise PortfolioEvidenceReviewError("product_host.evidence_review_evidence_not_current")
        noted = (EARLIER_METHOD_NOTE,) if eligibility == "SUPERSEDED" else ()
        if any(entity not in positions for entity in reading.entities):
            raise PortfolioEvidenceReviewError("product_host.evidence_review_scope_mismatch")
        unit_id = f"c{index:02d}"
        qualifier = _qualifier(publication.publication_hash)
        package = lineage.cro_package
        reviewed.extend(reading.entities)
        states.update(_review_states(lineage.brief.completion, reading.entities))
        cited = finding_filings(lineage)
        kept: set[str] = set()
        for finding in lineage.brief.findings:
            affected = tuple(entity for entity in finding.affected_entities if entity in positions)
            if not affected or not (
                cited[finding.finding_handle] & reading.filings
                or finding.finding_handle in reading.issue_findings
            ):
                continue
            structure = package.structure(finding.finding_handle)
            kept.update((*finding.supporting_span_handles, *finding.contradicting_span_handles))
            carried_findings += 1
            findings.append(
                PortfolioReviewDossierFinding(
                    finding_handle=_qualified_finding(finding.finding_handle, qualifier),
                    affected_entities=affected,
                    topic=finding.topic,
                    lifecycle=finding.lifecycle,
                    direction=finding.direction,
                    summary=finding.summary,
                    supporting_span_handles=tuple(
                        _qualified_span(value, qualifier)
                        for value in finding.supporting_span_handles
                    ),
                    contradicting_span_handles=tuple(
                        _qualified_span(value, qualifier)
                        for value in finding.contradicting_span_handles
                    ),
                    limitations=(*finding.limitations, *noted),
                    supporting_document_count=structure.supporting_document_count,
                    contradicting_document_count=structure.contradicting_document_count,
                    structure=structure.state,
                )
            )
        citations.extend(
            PortfolioReviewDossierCitation(
                span_handle=_qualified_span(span.span_handle, qualifier),
                document_handle=span.document_handle,
                entity_id=span.entity_id,
                available_at=span.available_at,
            )
            for span in package.verified_spans
            if span.span_handle in kept
        )
        requires_human_review = requires_human_review or lineage.brief.requires_human_review
        child_records.append(
            PortfolioReviewEvidenceChild(
                unit_id=unit_id,
                ordered_entity_ids=reading.entities,
                obligation_hash=publication.obligation_hash,
                analysis_publication_hash=publication.publication_hash,
                analyst_brief_hash=publication.analyst_brief_hash,
                cro_package_hash=publication.cro_package_hash,
                evidence_as_of=reading.as_of,
                evidence_expires_at=reading.expires_at,
                read_as_of=lineage.request.evidence_as_of,
            )
        )
    if carried:
        limitations.append(
            f"{carried_findings} finding(s) carried from {len(carried)} earlier reading(s) -- "
            "of filings still in the window, or that an open issue rests on -- none of them "
            "read again; each says when it was found."
        )
    held_findings = {value.finding_handle for value in findings}
    opened: list[PortfolioReviewOpenIssue] = []
    for state in open_issues:
        entities = tuple(e for e in state.issue.affected_entities if e in positions)
        handles = tuple(
            handle
            for publication_hash, finding_handle, _found in state.findings
            if (handle := _qualified_finding(finding_handle, _qualifier(publication_hash)))
            in held_findings
        )
        if not entities or not handles:
            limitations.append(
                f"Open issue {state.open_issue_handle} from {state.raised_on.isoformat()} "
                "could not be read back whole and is named here instead."
            )
            continue
        opened.append(
            PortfolioReviewOpenIssue(
                open_issue_handle=state.open_issue_handle,
                affected_entities=entities,
                raised_on=state.raised_on,
                assessed_on=state.assessed_on,
                cited_finding_handles=handles,
                causal_channel=state.issue.causal_channel,
                severity_if_true=state.issue.severity_if_true,
                evidence_interpretation=state.issue.evidence_interpretation,
                recommendation=state.issue.recommendation,
            )
        )
    unavailable = list(scope.unavailable_reasons)
    for unit_id, entities, reason in unreviewed:
        if any(entity in reviewed for entity in entities):
            raise PortfolioEvidenceReviewError("product_host.evidence_review_scope_mismatch")
        unavailable.append(f"unit {unit_id} ({', '.join(entities)}) is not reviewed: {reason}")
    reviewed_set = set(reviewed)
    if len(reviewed_set) != len(reviewed):
        raise PortfolioEvidenceReviewError("product_host.evidence_review_scope_mismatch")
    completion_schema = _completion_schema(states)
    executed = _executed(states)
    unavailable.extend(
        f"{entity} is not reviewed: {states[entity]}"
        for entity in reviewed
        if entity not in executed
    )
    reviewed_set = reviewed_set & executed
    if any(entity not in positions or entity in reviewed for entity in nothing_filed):
        raise PortfolioEvidenceReviewError("product_host.evidence_review_scope_mismatch")
    reviewed_listings = {
        listing for entity in reviewed_set for listing in positions[entity].listing_ids
    }
    nothing_listings = {
        listing for entity in nothing_filed for listing in positions[entity].listing_ids
    }
    book_ending = math.fsum(value.ending_weight for value in projection.positions)
    book_change = math.fsum(abs(value.signed_change) for value in projection.positions)
    reviewed_ending = math.fsum(
        value.ending_weight
        for value in projection.positions
        if value.listing_id in reviewed_listings
    )
    reviewed_change = math.fsum(
        abs(value.signed_change)
        for value in projection.positions
        if value.listing_id in reviewed_listings
    )
    # Every issuer a child analysed stays in the dossier with its state; the
    # coverage above counts only those whose checks executed.
    issuers = tuple(
        PortfolioReviewDossierIssuer(
            entity_id=issuer.entity_id,
            tickers=issuer.tickers,
            ending_weight=issuer.ending_weight,
            signed_change=issuer.signed_change,
            transition=issuer.transition,
            weight_rank=issuer.weight_rank,
            exposure_band=issuer.exposure_band,
            selection_reason=issuer.selection_reason,
            review_state=(
                "NOTHING_FILED"
                if issuer.entity_id in nothing_filed
                else states.get(issuer.entity_id, "UNSTATED")
            ),
        )
        for issuer in scope.selected_issuers
        if issuer.entity_id in reviewed or issuer.entity_id in nothing_filed
    )
    # An earlier finding or open issue can still name a holding whose new
    # source was not read. Keep the issuer without counting it as reviewed.
    named = {value.entity_id for value in issuers}
    issuers += tuple(
        PortfolioReviewDossierIssuer(
            entity_id=issuer.entity_id,
            tickers=issuer.tickers,
            ending_weight=issuer.ending_weight,
            signed_change=issuer.signed_change,
            transition=issuer.transition,
            weight_rank=issuer.weight_rank,
            exposure_band=issuer.exposure_band,
            selection_reason=issuer.selection_reason,
            review_state="UNREVIEWED",
        )
        for issuer in scope.selected_issuers
        if issuer.entity_id not in named
        and (
            any(issuer.entity_id in value.affected_entities for value in findings)
            or any(issuer.entity_id in value.affected_entities for value in opened)
        )
    )
    weight_parts: dict[str, object] = {}
    if nothing_filed:
        weight_parts = {
            "nothing_filed_ending_weight_coverage": _ratio(
                math.fsum(
                    value.ending_weight
                    for value in projection.positions
                    if value.listing_id in nothing_listings
                ),
                book_ending,
            ),
            # Exact: an empty sum when every holding is read or filed nothing.
            "unreached_ending_weight_coverage": _ratio(
                math.fsum(
                    value.ending_weight
                    for value in projection.positions
                    if value.listing_id not in reviewed_listings
                    and value.listing_id not in nothing_listings
                ),
                book_ending,
            ),
            "nothing_filed_window_days": (
                children[0][2].lineage if children else carried[0].evidence.lineage
            ).request.source_policy.sec_recent_8k_days,
        }
    coverage = PortfolioReviewCoverage(
        reviewed_ending_weight_coverage=_ratio(reviewed_ending, book_ending),
        reviewed_absolute_change_coverage=_ratio(reviewed_change, book_change),
        mapping_coverage=scope.mapping_coverage,
        selected_issuer_coverage=_ratio(float(len(reviewed_set)), float(scope.mapped_issuer_count)),
        missing_evidence=tuple(dict.fromkeys(missing)),
        unavailable_reasons=tuple(unavailable),
        **weight_parts,
    )
    children_sorted = tuple(sorted(child_records, key=lambda value: value.unit_id))
    aggregate: dict[str, object] = {
        "kind": "PortfolioEvidenceAggregate",
        "children": [[value.unit_id, value.analysis_publication_hash] for value in children_sorted],
    }
    return seal_contract(
        PortfolioReviewDossier,
        "dossier_hash",
        book_authority=book.authority,
        report_hash=book.report_hash,
        update_subject=book.update_subject,
        experiment_subject=book.experiment_subject,
        result_hash=book.result_hash,
        candidate_hash=book.candidate_hash,
        handoff_hash=book.handoff_hash,
        issuer_scope_hash=scope.scope_hash,
        exposure_projection_hash=projection.projection_hash,
        registry_hash=scope.registry_hash,
        analysis_publication_hash=str(canonical_hash(aggregate)),
        analyst_brief_hash=str(
            canonical_hash({**aggregate, "briefs": [v.analyst_brief_hash for v in children_sorted]})
        ),
        cro_package_hash=str(
            canonical_hash({**aggregate, "packages": [v.cro_package_hash for v in children_sorted]})
        ),
        obligation_hash=str(
            canonical_hash(
                {**aggregate, "obligations": [v.obligation_hash for v in children_sorted]}
            )
        ),
        evidence_as_of=min(value.evidence_as_of for value in children_sorted),
        evidence_expires_at=min(value.evidence_expires_at for value in children_sorted),
        evidence_children=children_sorted,
        unresolved_questions=tuple(questions),
        completion_schema=completion_schema,
        issuers=issuers,
        findings=tuple(findings),
        citations=tuple(citations),
        coverage=coverage,
        analyst_requires_human_review=requires_human_review,
        mapping_failure_count=len(scope.mapping_failures),
        held_count=projection.held_count,
        window_end_effective_n=projection.window_end_effective_n,
        claim_limits=REVIEW_CLAIM_LIMITS + _book_claim_limits(book),
        limitations=tuple(
            dict.fromkeys(
                (
                    *limitations,
                    *book.temporal_statements,
                    f"Analysed as {len(children)} unit(s) of at most {UNIT_LIMIT} issuers"
                    + (f" and {len(carried)} earlier reading(s)" if carried else "")
                    + "; finding and span handles are qualified by their unit's publication.",
                )
            )
        ),
        portfolio_report_link=_portfolio_report_link(inputs, book),
        open_issues=tuple(opened),
    )


def portfolio_review_key_payload(
    *,
    dossier: PortfolioReviewDossier,
    decision_policy_hash: str,
    typed_user_authority: str,
    actor_kind: str,
    actor_id: str,
    process_binding_hash: str,
    response_schema_hash: str,
    submission_hash: str | None = None,
) -> dict[str, object]:
    """The review key: everything that must change before new cognition is bought.

    It deliberately binds no wall clock and no mutable pointer. Asking the same
    question twice must be free; asking a genuinely new one must require a
    changed input.

    Args:
        dossier: Exact dossier whose evidence and book identities enter the key.
        decision_policy_hash: Installed decision policy captured by the request.
        typed_user_authority: User authority recorded in the request key.
        actor_kind: Kind of actor associated with the request.
        actor_id: Actor identifier associated with the request.
        process_binding_hash: Process binding captured by the request.
        response_schema_hash: Answer schema binding captured by the request.
        submission_hash: Optional sealed submission identity.

    Returns:
        Key fields, including an update or experiment subject when the dossier has one.
    """
    payload: dict[str, object] = {
        "report_hash": dossier.report_hash,
        "book_authority": str(dossier.book_authority),
        "issuer_scope_hash": dossier.issuer_scope_hash,
        "analysis_publication_hash": dossier.analysis_publication_hash,
        "cro_package_hash": dossier.cro_package_hash,
        "obligation_hash": dossier.obligation_hash,
        "dossier_hash": dossier.dossier_hash,
        "decision_policy_hash": decision_policy_hash,
        "typed_user_authority": typed_user_authority,
        "actor_kind": actor_kind,
        "actor_id": actor_id,
        "process_binding_hash": process_binding_hash,
        "response_schema_hash": response_schema_hash,
        "execution_semantics": "TOOL_FREE_SINGLE_SEMANTIC_EXECUTION",
    }
    if submission_hash is not None:
        payload["submission_hash"] = submission_hash
    if dossier.update_subject is not None:
        payload["update_subject"] = dossier.update_subject.model_dump(mode="json")
    if dossier.experiment_subject is not None:
        payload["experiment_subject"] = dossier.experiment_subject.model_dump(mode="json")
    return payload


def review_book_key(publication: PortfolioReviewPublication) -> str:
    """Name the book a published review is of, by `book_key`.

    Args:
        publication: The review's publication record.

    Returns:
        The digest `book_key` gives the book it reviewed.
    """
    return book_key(
        publication.report_hash,
        publication.update_subject,
        publication.experiment_subject,
        publication.book_authority,
    )


def assessment_schema(dossier: PortfolioReviewDossier) -> dict[str, object]:
    """Return the answer's schema with the dossier's finding aliases as the choices.

    Args:
        dossier: The dossier the answer is about.

    Returns:
        `PortfolioReviewAnswer`'s JSON schema, its findings limited to the dossier's aliases
        (and its risks to none when the dossier has no finding or no issuer).
    """
    schema = PortfolioReviewAnswer.model_json_schema()
    if not dossier.finding_handles or not dossier.issuers:
        schema["properties"]["risks"]["maxItems"] = 0
    else:
        ref = schema["properties"]["risks"]["items"]["$ref"].split("/")[-1]
        properties = schema["$defs"][ref]["properties"]
        properties["findings"]["items"]["enum"] = list(finding_aliases(dossier))
    return cast(dict[str, object], schema)


def assessment_schema_hash(dossier: PortfolioReviewDossier) -> str:
    """Hash what an answer's schema binds: its structure and its choices among it (SC3).

    Args:
        dossier: The dossier the answer is about.

    Returns:
        The hash of the schema's structure.
    """
    return str(canonical_hash(schema_structure(assessment_schema(dossier))))


def finding_dispositions(view: Any) -> dict[str, str]:
    """Return how a published review disposed of each of its dossier's findings.

    Args:
        view: The review's publication view (its receipt and its dossier).

    Returns:
        Each finding's handle with its disposition.
    """
    return {
        value.finding_handle: str(value.disposition)
        for value in review_dispositions(
            view.receipt.submission,
            answered=view.receipt.answer is not None,
            finding_handles=tuple(f.finding_handle for f in view.dossier.findings),
        )
    }


def typed_disclosure_records(
    dossier: PortfolioReviewDossier,
    replay: Callable[[str], AlternativeEvidenceAnalysisPublicationView],
) -> tuple[list[Any], tuple[str, str] | None]:
    """Collect the typed disclosure observations a dossier's evidence read, with their rules.

    Args:
        dossier: The dossier whose evidence publications are read.
        replay: Reads one evidence publication by its hash.

    Returns:
        Every observation of the publications that ran the families, and the rules (their id
        and definitions' hash) of the last that did; None for the rules when none did.
    """
    observations: list[Any] = []
    rules: tuple[str, str] | None = None
    for publication_hash in dossier.evidence_publication_hashes:
        typed = replay(publication_hash).lineage.access_receipt.typed_disclosures
        if typed is None:
            continue
        observations.extend(typed.observations)
        rules = (typed.rules_id, typed.definitions_hash)
    return observations, rules


__all__ = [
    "EVIDENCE_QUESTION",
    "EVIDENCE_REQUIRED_CHECKS",
    "REVIEW_CLAIM_LIMITS",
    "BookSelector",
    "PortfolioEvidenceReviewError",
    "PortfolioEvidenceReviewInputs",
    "SealedBook",
    "assessment_schema",
    "assessment_schema_hash",
    "book_listing_count",
    "compile_portfolio_coverage_dossier",
    "compile_portfolio_review_dossier",
    "compile_portfolio_scope",
    "finding_dispositions",
    "open_sealed_book",
    "portfolio_review_key_payload",
    "project_coverage_run",
    "project_portfolio_exposure",
    "project_unit_obligation",
    "qualify_span_handle",
    "review_book_key",
    "typed_disclosure_records",
]
