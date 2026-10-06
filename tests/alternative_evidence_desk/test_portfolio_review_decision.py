"""The Portfolio evidence review in isolation: scope, obligation, route, key.

Pure functions and single contracts. The product path -- one sealed book
through a real ledger, a real Alternative Evidence Task, a real review Task and
the interface projection -- is proved in `test_evidence_review_route.py`, and
over HTTP in `test_evidence_review_http_route.py`. This file asks the questions
that need no workspace: how the scope selects, what the obligation may say, what
the route function does with an assessment, and what the review key binds.

Non-reserved synthetic fixtures only. No Provider, no network, no protected or
post-2024-08-12 evidence.
"""

from __future__ import annotations

import ast
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    EvidenceStructureState,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    SecIssuerRegistryEntry,
    SecIssuerRegistrySnapshot,
    seal_contract,
)
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    EVIDENCE_REQUIRED_CHECKS,
    portfolio_review_key_payload,
    project_unit_obligation,
)
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PORTFOLIO_REVIEW_ROUTE_POLICY,
    CROEvidenceInterpretation,
    CROEvidenceRelevance,
    CROPositionImpactDirection,
    CRORiskConfidence,
    CRORiskSeverity,
    PortfolioReviewAnswer,
    PortfolioReviewDossier,
    PortfolioReviewMaterialIssue,
    PortfolioReviewRecommendation,
    PortfolioReviewRoute,
    RequiredActionKind,
    ReviewState,
)
from alphalattice.oversight.chief_risk_officer.decision.submissions import (
    CRODecisionAuthorityError,
    build_portfolio_review_policy_binding,
    screen_review_answer,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    BookAuthority,
    ExposureBand,
    PortfolioExposureProjection,
    PortfolioIssuerScope,
    PortfolioListingPosition,
    exposure_band,
    seal_portfolio_evidence_contract,
    transition_of,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.coverage.issuer_scope import (
    compile_portfolio_issuer_scope,
)
from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
    portfolio_review_task_contract,
)
from alphalattice.protocols.actor_execution import ActorKind, AgentExecutionBinding
from alphalattice.protocols.actor_execution.answers import AnswerProblem
from tests.alternative_evidence_desk.review_dossiers import (
    _NOW,
    ENTITY,
    FINDING,
    REPO_ROOT,
    ControlledRisk,
    _dossier,
    _finding,
    _issue,
    _issuer,
    _seal,
    _submission,
    controlled_answer,
)

APPROVED_SOURCES: tuple[str, ...] = ("SEC_EDGAR_OFFICIAL",)

ISSUERS: tuple[tuple[str, str, str], ...] = (
    ("AAPL", "AAPL", "0000320193"),
    ("MSFT", "MSFT", "0000789019"),
    ("NVDA", "NVDA", "0001045810"),
    ("TSLA", "TSLA", "0001318605"),
    ("AMZN", "AMZN", "0001018724"),
)

WIDE_TICKERS: tuple[str, ...] = tuple(f"W{index:02d}" for index in range(1, 12))
"""Eleven synthetic issuers: wider than one evidence unit, narrower than two."""

WIDE_ISSUERS: tuple[tuple[str, str, str], ...] = tuple(
    (ticker, ticker, f"{9_200_000_000 + index:010d}") for index, ticker in enumerate(WIDE_TICKERS)
)

BASE_ROWS: tuple[tuple[str, str | None, float, float], ...] = (
    ("AAPL-XNAS", "AAPL", 0.30, 0.20),
    ("MSFT-XNAS", "MSFT", 0.25, 0.25),
    ("NVDA-XNAS", "NVDA", 0.20, 0.35),
    ("TSLA-XNAS", "TSLA", 0.15, 0.00),
    ("AMZN-XNAS", "AMZN", 0.10, 0.20),
)


def _registry(entries: tuple[tuple[str, str, str], ...] = ISSUERS) -> SecIssuerRegistrySnapshot:
    return seal_contract(
        SecIssuerRegistrySnapshot,
        "registry_hash",
        captured_at=_NOW - timedelta(days=1),
        entries=tuple(
            SecIssuerRegistryEntry(
                entity_id=entity_id, ticker=ticker, cik=cik, legal_name=f"{entity_id} Inc."
            )
            for entity_id, ticker, cik in entries
        ),
        source_content_hash="1" * 64,
    )


def _projection(
    rows: tuple[tuple[str, str | None, float, float], ...] = BASE_ROWS,
    *,
    authority: BookAuthority = BookAuthority.DEVELOPMENT_RESULT,
) -> PortfolioExposureProjection:
    """One exposure projection, sealed directly. The Host path is proved elsewhere."""

    handoff = authority is BookAuthority.VALIDATED_HANDOFF
    return seal_portfolio_evidence_contract(
        PortfolioExposureProjection,
        "projection_hash",
        book_authority=authority,
        report_hash="c" * 64,
        result_hash=None if handoff else "a" * 64,
        window_end_book_hash="d" * 64,
        listing_authority_hash="e" * 64,
        candidate_hash=None if authority is BookAuthority.DEVELOPMENT_RESULT else "b" * 64,
        handoff_hash="f" * 64 if handoff else None,
        formation_session="2024-06-03",
        change_boundary="PRECEDING_FORMATION",
        held_count=len(rows),
        window_end_effective_n=float(len(rows)),
        positions=tuple(
            PortfolioListingPosition(
                listing_id=listing_id,
                ticker=ticker,
                ending_weight=ending,
                preceding_weight=preceding,
                signed_change=ending - preceding,
                transition=transition_of(ending=ending, preceding=preceding),
            )
            for listing_id, ticker, ending, preceding in rows
        ),
    )


def _scope(
    rows: tuple[tuple[str, str | None, float, float], ...] = BASE_ROWS,
    *,
    registry: SecIssuerRegistrySnapshot | None = None,
) -> PortfolioIssuerScope:
    return compile_portfolio_issuer_scope(
        projection=_projection(rows),
        registry=registry if registry is not None else _registry(),
    )


# ============================================== the Portfolio-to-Evidence scope


def test_the_scope_selects_by_the_canonical_lexicographic_rule() -> None:
    """Opened and increased first, then unchanged holds, then what shrank."""

    scope = _scope()

    assert scope.ordered_entity_ids == ("TSLA", "AAPL", "MSFT", "NVDA", "AMZN")
    assert [value.selection_reason for value in scope.selected_issuers] == [
        "OPENED_OR_INCREASED_BY_POSITIVE_CHANGE",
        "OPENED_OR_INCREASED_BY_POSITIVE_CHANGE",
        "HELD_BY_ENDING_WEIGHT",
        "REDUCED_OR_EXITED_BY_ABSOLUTE_CHANGE",
        "REDUCED_OR_EXITED_BY_ABSOLUTE_CHANGE",
    ]
    assert {value.entity_id: value.weight_rank for value in scope.selected_issuers} == {
        "AAPL": 1,
        "MSFT": 2,
        "NVDA": 3,
        "TSLA": 4,
        "AMZN": 5,
    }
    assert scope.reviewed_ending_weight_coverage == 1.0
    assert scope.mapping_coverage == 1.0
    assert scope.mapping_failures == ()
    assert scope == _scope(), "a pure function of its inputs"


def test_a_listing_with_no_admitted_ticker_stays_in_the_denominator() -> None:
    rows = (("ZZZZ-XNAS", None, 0.20, 0.20), *BASE_ROWS[1:])
    scope = _scope(rows)

    assert "AAPL" not in scope.ordered_entity_ids
    assert len(scope.mapping_failures) == 1
    assert scope.mapping_failures[0].reason == "LISTING_NOT_IN_ADMITTED_AUTHORITY"
    assert scope.mapping_failures[0].ending_weight == 0.20
    assert scope.mapping_coverage < 1.0
    assert scope.reviewed_ending_weight_coverage < 1.0
    assert any("did not map" in value for value in scope.unavailable_reasons)


def test_a_ticker_outside_the_registry_is_a_different_gap() -> None:
    rows = (("ZZZZ-XNAS", "ZZZZ", 0.20, 0.20), *BASE_ROWS[1:])
    scope = _scope(rows)

    assert scope.mapping_failures[0].reason == "TICKER_NOT_IN_SEC_REGISTRY"
    assert scope.mapping_failures[0].ticker == "ZZZZ"


def test_every_mapped_issuer_is_in_scope_and_eight_is_only_a_unit() -> None:
    """requirement: a book wider than one evidence request names every mapped
    issuer in priority order; the request axis (eight) cuts execution units,
    never the scope, and priority decides unit order, never membership."""

    from alphalattice.evidence.alternative_evidence.runtime.coverage import (
        UNIT_LIMIT,
        coverage_units,
    )

    rows = tuple(
        (f"{ticker}-XNAS", ticker, 0.02 + 0.001 * index, 0.02)
        for index, ticker in enumerate(WIDE_TICKERS)
    )
    scope = _scope(rows, registry=_registry(WIDE_ISSUERS))

    assert len(scope.selected_issuers) == len(WIDE_TICKERS) == 11
    assert scope.selected_issuer_coverage == 1.0
    assert scope.reviewed_ending_weight_coverage == 1.0
    assert scope.unavailable_reasons == ()
    assert tuple(value.selection_rank for value in scope.selected_issuers) == tuple(range(1, 12))
    units = coverage_units(scope.ordered_entity_ids, priority_rank=scope.priority_rank)
    assert sorted(len(unit) for unit in units) == [3, UNIT_LIMIT]
    assert sorted(entity for unit in units for entity in unit) == sorted(WIDE_TICKERS)
    # The unit holding the rank-one issuer is prepared first.
    assert scope.ordered_entity_ids[0] in units[0]
    # Membership by entity id, so the same issuers under other weights cut the
    # same units; order by the best rank inside each unit.
    other = _scope(
        tuple(
            (listing, ticker, 0.02, 0.02 + 0.001 * index)
            for index, (listing, ticker, _e, _p) in enumerate(rows)
        ),
        registry=_registry(WIDE_ISSUERS),
    )
    assert other.ordered_entity_ids != scope.ordered_entity_ids
    assert {
        frozenset(unit)
        for unit in coverage_units(other.ordered_entity_ids, priority_rank=other.priority_rank)
    } == {frozenset(unit) for unit in units}
    # A book that fits one unit is the one request it always was: priority
    # order, untouched, so an existing preparation keeps its identity.
    narrow = scope.ordered_entity_ids[:UNIT_LIMIT]
    assert coverage_units(narrow, priority_rank=scope.priority_rank) == (narrow,)
    with pytest.raises(ValueError, match="coverage_unit_limit_invalid"):
        coverage_units(scope.ordered_entity_ids, priority_rank={}, unit_limit=UNIT_LIMIT + 1)
    with pytest.raises(ValueError, match="coverage_entities_invalid"):
        coverage_units(("A", "A"), priority_rank={})


def test_units_are_packed_from_each_issuer_s_own_source_count() -> None:
    """requirement (6A, W1): with each issuer's logical source count the units
    are packed under the issuer limit and the admitted document set, heaviest
    holding first, so an issuer's selection is never decided by how many
    issuers share its unit and the heaviest holdings are read first; inside a
    unit the issuers keep the scope's order, so a book whose selections fit
    one unit is the one request it always was; an issuer that filed nothing in
    the window is not packed; an issuer whose own selection exceeds the set,
    a missing count and a missing weight rank are refused by name."""

    from alphalattice.evidence.alternative_evidence.contracts import ADMITTED_DOCUMENT_CAPACITY
    from alphalattice.evidence.alternative_evidence.runtime.coverage import (
        UNIT_LIMIT,
        coverage_units,
    )

    rows = tuple(
        (f"{ticker}-XNAS", ticker, 0.02 + 0.001 * index, 0.02)
        for index, ticker in enumerate(WIDE_TICKERS)
    )
    scope = _scope(rows, registry=_registry(WIDE_ISSUERS))
    entities = scope.ordered_entity_ids
    weight = {issuer.entity_id: issuer.weight_rank for issuer in scope.selected_issuers}
    heaviest = sorted(entities, key=lambda entity: (weight[entity], entity))

    def pack(names, counts, **extra):
        return coverage_units(
            names,
            priority_rank=scope.priority_rank,
            source_counts=counts,
            weight_rank=weight,
            **extra,
        )

    # Ten documents each: at most two issuers fit the 24-document set.
    dense = dict.fromkeys(entities, 10)
    units = pack(entities, dense)
    assert all(len(unit) <= 2 for unit in units) and len(units) == 6
    assert all(sum(dense[e] for e in unit) <= ADMITTED_DOCUMENT_CAPACITY for unit in units)
    assert sorted(e for unit in units for e in unit) == sorted(entities)
    assert [frozenset(unit) for unit in units] == [
        frozenset(heaviest[i : i + 2]) for i in range(0, len(entities), 2)
    ], "packed heaviest first, two per unit"
    assert all(list(unit) == sorted(unit, key=entities.index) for unit in units), (
        "inside a unit the scope's order"
    )
    assert heaviest[0] in units[0], "the heaviest holding's unit is prepared first"
    # Three each: the issuer limit binds first (eight per unit), as before.
    sparse = dict.fromkeys(entities, 3)
    assert sorted(len(unit) for unit in pack(entities, sparse)) == [3, UNIT_LIMIT]
    # Mixed counts: a unit closes when the next issuer would overflow the set.
    mixed = {e: (12 if i % 3 == 0 else 3) for i, e in enumerate(sorted(entities))}
    packed = pack(entities, mixed)
    assert all(sum(mixed[e] for e in unit) <= ADMITTED_DOCUMENT_CAPACITY for unit in packed)
    assert all(len(unit) <= UNIT_LIMIT for unit in packed)
    # Weights decide membership and order: reweighted, the new heaviest leads.
    other = _scope(
        tuple(
            (listing, ticker, 0.02 + 0.001 * (len(rows) - i), 0.02)
            for i, (listing, ticker, _e, _p) in enumerate(rows)
        ),
        registry=_registry(WIDE_ISSUERS),
    )
    other_weight = {issuer.entity_id: issuer.weight_rank for issuer in other.selected_issuers}
    other_units = coverage_units(
        other.ordered_entity_ids,
        priority_rank=other.priority_rank,
        source_counts=dense,
        weight_rank=other_weight,
    )
    assert min(other_units[0], key=other_weight.__getitem__) == min(
        other_weight, key=other_weight.__getitem__
    )
    # A book whose selections fit one unit keeps its order and identity.
    narrow = entities[:UNIT_LIMIT]
    assert pack(narrow, sparse) == (narrow,)
    # Eight issuers whose selections do not fit one unit are packed, not cut.
    assert len(pack(narrow, dense)) == 4
    # An issuer that filed nothing in the window is named by the run, not packed.
    quiet = pack(narrow, {**sparse, narrow[0]: 0}, nothing_filed=frozenset({narrow[0]}))
    assert quiet == (narrow[1:],)
    with pytest.raises(ValueError, match="coverage_issuer_exceeds_unit_capacity"):
        pack(narrow, {**dense, narrow[0]: 25})
    with pytest.raises(ValueError, match="coverage_source_counts_missing"):
        pack(narrow, {narrow[0]: 3})
    with pytest.raises(ValueError, match="coverage_weight_rank_missing"):
        coverage_units(narrow, priority_rank={}, source_counts=sparse)


def test_a_run_of_many_units_carries_a_plan_task_control_admits() -> None:
    """regression (the retained book's PREPARE, 2026-09-20): a book packed
    from each issuer's source count needs more units than the issuer-limit
    cut ever produced -- fourteen for the 64-issuer book -- and its Task plan
    carries one set of stages per unit (84 at preparation), which Task
    Control's plan of at most 64 work items refused at submission. The plan
    bound is sized for every unit the run contract admits across every
    stage, the recovery view carries every stage of such a plan, and the
    adapter names a run the plan could not carry by its size."""

    from annotated_types import MaxLen

    from alphalattice.control.product_host.composition.task_recovery import TaskRecoveryView
    from alphalattice.control.task_control.contracts import PLAN_WORK_ITEM_LIMIT, ResearchPlan
    from alphalattice.evidence.alternative_evidence.runtime.coverage import RUN_UNIT_LIMIT
    from alphalattice.evidence.alternative_evidence.runtime.policy import AdmittedEvidencePolicy
    from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
        _STAGES,
        coverage_run_task_contract,
    )
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
        project_coverage_run,
    )

    def bound(model: type[Any], field: str) -> int | None:
        return next(
            (
                value.max_length
                for value in model.model_fields[field].metadata
                if isinstance(value, MaxLen)
            ),
            None,
        )

    assert RUN_UNIT_LIMIT * len(_STAGES) <= PLAN_WORK_ITEM_LIMIT
    assert bound(ResearchPlan, "work_items") == PLAN_WORK_ITEM_LIMIT
    assert bound(TaskRecoveryView, "stages") == PLAN_WORK_ITEM_LIMIT
    rows = tuple(
        (f"{ticker}-XNAS", ticker, 0.02 + 0.001 * index, 0.02)
        for index, ticker in enumerate(WIDE_TICKERS)
    )
    scope = _scope(rows, registry=_registry(WIDE_ISSUERS))
    policy = AdmittedEvidencePolicy()
    as_of = datetime(2026, 9, 19, 4, 58, 49, tzinfo=UTC)
    # Thirteen documents each: no two issuers share a 24-document unit.
    run = project_coverage_run(
        scope=scope,
        evidence_as_of=as_of,
        approved_source_families=policy.approved_source_families,
        request_for=lambda entities, when, read: policy.request(
            ordered_entity_ids=entities, evidence_as_of=when, read_filings=read
        ),
        resource_binding_hash="1" * 64,
        network_consent=False,
        admit_live_official=False,
        admit_model_review=False,
        source_counts=dict.fromkeys(scope.ordered_entity_ids, 13),
    )
    assert len(run.units) == len(WIDE_TICKERS) == 11
    for prepare_only, stage_count in ((True, 6), (False, len(_STAGES))):
        _envelope, _goal, plan = coverage_run_task_contract(run=run, prepare_only=prepare_only)
        assert len(plan.work_items) == 11 * stage_count
    assert len(plan.work_items) == 88 > 64


@pytest.mark.parametrize(
    ("ending", "change", "rank", "expected"),
    (
        (0.005, 0.0, 30, ExposureBand.LOW),
        (0.03, 0.0, 10, ExposureBand.MEDIUM),
        (0.01, 0.006, 20, ExposureBand.MEDIUM),
        (0.06, 0.0, 10, ExposureBand.HIGH),
        (0.03, -0.025, 10, ExposureBand.HIGH),
        (0.03, 0.0, 3, ExposureBand.HIGH),
        (0.12, 0.0, 1, ExposureBand.CRITICAL),
        (0.06, 0.055, 2, ExposureBand.CRITICAL),
    ),
)
def test_the_exposure_band_is_a_function_of_weight_change_and_rank(
    ending: float, change: float, rank: int, expected: ExposureBand
) -> None:
    """Position size decides priority and exposure; never how severe a finding is."""

    assert exposure_band(ending_weight=ending, signed_change=change, weight_rank=rank) is expected


def test_the_evidence_obligation_carries_no_portfolio_fact() -> None:
    """Alternative Evidence learns the issuers, the cutoff and the sources. Nothing else."""

    scope = _scope()
    obligation = project_unit_obligation(
        ordered_entity_ids=scope.ordered_entity_ids,
        evidence_as_of=_NOW,
        approved_source_families=APPROVED_SOURCES,
    )

    assert obligation.ordered_entity_ids == scope.ordered_entity_ids
    assert obligation.required_checks == EVIDENCE_REQUIRED_CHECKS
    facts = {
        key: value for key, value in obligation.model_dump(mode="json").items() if key != "question"
    }
    rendered = json.dumps(facts).casefold()
    for banned in ("weight", "0.3", "0.25", "increased", "reduced", "rank", "band", "critical"):
        assert banned not in rendered, banned
    assert "do not infer portfolio weights" in obligation.question.casefold()
    assert obligation == project_unit_obligation(
        ordered_entity_ids=scope.ordered_entity_ids,
        evidence_as_of=_NOW,
        approved_source_families=APPROVED_SOURCES,
    )


# ================================================================ the route


def test_the_actor_schema_cannot_express_a_number() -> None:
    """No probability, expected loss, score, weight, route or target can be submitted."""

    schema = PortfolioReviewAnswer.model_json_schema()
    types = set(re.findall(r'"type": "([a-z]+)"', json.dumps(schema)))
    assert types <= {"string", "array", "object", "boolean"}, types

    names = {
        name
        for definition in (schema, *schema.get("$defs", {}).values())
        for name in definition.get("properties", {})
    }
    for banned in ("probability", "expected_loss", "score", "target_weight", "activation", "route"):
        assert not any(banned in name for name in names), (banned, sorted(names))
    # Judgment only: no handle, hash, disposition, relevance or mitigation to state.
    for ritual in ("handle", "hash", "disposition", "relevance", "mitigation", "entities"):
        assert not any(ritual in name for name in names), (ritual, sorted(names))


@pytest.mark.parametrize(
    ("kwargs", "band", "expected", "rule_id"),
    (
        ({}, ExposureBand.HIGH, PortfolioReviewRoute.MATERIAL_OBJECTION, "R-OBJECTION"),
        ({}, ExposureBand.CRITICAL, PortfolioReviewRoute.MATERIAL_OBJECTION, "R-OBJECTION"),
        ({}, ExposureBand.MEDIUM, PortfolioReviewRoute.ACCEPT_WITH_LIMITS, "R-LIMITS"),
        ({}, ExposureBand.LOW, PortfolioReviewRoute.ACCEPT_WITH_LIMITS, "R-LIMITS"),
        (
            {"confidence": CRORiskConfidence.CONTESTED},
            ExposureBand.HIGH,
            PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED,
            "R-HIGH-UNSETTLED",
        ),
        (
            {"confidence": CRORiskConfidence.LIMITED},
            ExposureBand.CRITICAL,
            PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED,
            "R-HIGH-UNSETTLED",
        ),
        (
            {"confidence": CRORiskConfidence.CONTESTED},
            ExposureBand.MEDIUM,
            PortfolioReviewRoute.ACCEPT_WITH_LIMITS,
            "R-LIMITS",
        ),
        (
            {"severity": CRORiskSeverity.MEDIUM},
            ExposureBand.CRITICAL,
            PortfolioReviewRoute.ACCEPT_WITH_LIMITS,
            "R-LIMITS",
        ),
        (
            {"severity": CRORiskSeverity.MEDIUM, "confidence": CRORiskConfidence.CONTESTED},
            ExposureBand.HIGH,
            PortfolioReviewRoute.ACCEPT_WITH_LIMITS,
            "R-LIMITS",
        ),
        (
            {"severity": CRORiskSeverity.LOW},
            ExposureBand.CRITICAL,
            PortfolioReviewRoute.NO_MATERIAL_OBJECTION,
            "R-LOW",
        ),
    ),
)
def test_the_four_axis_route_matrix_is_deterministic(
    kwargs: dict[str, Any], band: ExposureBand, expected: PortfolioReviewRoute, rule_id: str
) -> None:
    """Severity, exposure band and the capped confidence decide together (the
    seam plan's section 7): a high-severity risk on an exposed holding objects
    when its findings support it and asks a person -- advisory, never
    blocking -- when they are contested or limited; any other material risk is
    accepted with its limits; a low one is listed without objection. Position
    size changes what the book must do about a finding, never whether it is
    true.
    """

    dossier = _dossier(band=band)
    receipt, recommendation = _seal(
        dossier, controlled_answer(dossier, ControlledRisk(ENTITY, **kwargs))
    )

    assert recommendation.route is expected
    assert receipt.outcome.issue_evaluations[0].rule_id == rule_id
    assert receipt.outcome.issue_evaluations[0].exposure_band is band
    if expected is PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED:
        assert not any(value.blocking for value in recommendation.required_actions)


def test_an_irrelevant_or_favourable_issue_is_not_evaluated() -> None:
    receipt, recommendation = _seal(
        _dossier(),
        _submission(
            _issue(
                FINDING,
                ENTITY,
                relevance=CROEvidenceRelevance.NOT_RELEVANT,
                direction=CROPositionImpactDirection.NONE,
            ),
            _issue(
                FINDING,
                ENTITY,
                direction=CROPositionImpactDirection.FAVOURABLE,
                handle="ISSUE-2",
            ),
        ),
    )

    assert recommendation.route is PortfolioReviewRoute.NO_MATERIAL_OBJECTION
    assert receipt.outcome.issue_evaluations == ()
    assert recommendation.issuer_conclusions[0].conclusion == "NO_ADVERSE_ISSUE"


@pytest.mark.parametrize(
    ("structure", "effective", "expected"),
    (
        (
            EvidenceStructureState.SUPPORTED,
            CROEvidenceInterpretation.SUPPORTED,
            PortfolioReviewRoute.MATERIAL_OBJECTION,
        ),
        (
            EvidenceStructureState.SINGLE_SOURCE,
            CROEvidenceInterpretation.LIMITED,
            PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED,
        ),
        (
            EvidenceStructureState.CONTESTED,
            CROEvidenceInterpretation.CONTESTED,
            PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED,
        ),
        (
            EvidenceStructureState.UNSUPPORTED,
            CROEvidenceInterpretation.INSUFFICIENT,
            PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED,
        ),
    ),
)
def test_the_citation_structure_caps_what_the_actor_may_claim(
    structure: EvidenceStructureState,
    effective: CROEvidenceInterpretation,
    expected: PortfolioReviewRoute,
) -> None:
    """The Host counted the documents; the actor's `SUPPORTED` cannot exceed them.

    One document is a single source, a contradicting document is a contest, no
    document is a gap. An objection therefore needs two independent documents
    behind the finding, whatever the reviewer said; anything less on an
    exposed holding goes to a person, advisory.
    """

    receipt, recommendation = _seal(
        _dossier(structure=structure), _submission(_issue(FINDING, ENTITY))
    )

    evaluation = receipt.outcome.issue_evaluations[0]
    assert evaluation.stated_interpretation is CROEvidenceInterpretation.SUPPORTED
    assert evaluation.effective_interpretation is effective
    assert evaluation.structure is structure
    assert recommendation.route is expected


def test_a_holding_that_filed_nothing_is_counted_apart_and_never_as_no_risk() -> None:
    """requirement (W1): a holding whose filing index held nothing in the window
    is in the dossier with its weight and in no unit. The coverage counts weight
    three ways -- read, filed nothing, not read -- and a review with nothing
    unread is complete, one with weight unread partial. The holding's
    conclusion is NOTHING_FILED, and the limitation says it is not a finding of
    no risk."""
    from alphalattice.interface.local_application.evidence_cro import (
        EvidenceCroBook,
        evidence_cro_body,
        project_published_review,
    )

    quiet = _issuer("QA99", weight_rank=9).model_copy(update={"review_state": "NOTHING_FILED"})

    def reviewed(unreached: float) -> tuple[Any, Any]:
        dossier = _dossier(
            issuers=(_issuer(), quiet),
            reviewed_ending_weight_coverage=0.9 - unreached,
            nothing_filed_ending_weight_coverage=0.1,
            unreached_ending_weight_coverage=unreached,
        )
        receipt, recommendation = _seal(dossier, controlled_answer(dossier, ControlledRisk(ENTITY)))
        projected = project_published_review(
            recommendation=recommendation,
            dossier=dossier,
            receipt=receipt,
            percent=lambda value: f"{value:.3%}",
            change=lambda value: f"{value:+.3%}",
            book=EvidenceCroBook(
                authority="DEVELOPMENT_RESULT",
                explanation="Synthetic sealed book",
                result_hash=None,
            ),
        )
        coverage = evidence_cro_body(projected)["coverage"]
        assert coverage["reviewed_ending_weight"] == f"{0.9 - unreached:.3%}"
        assert coverage["nothing_filed_ending_weight"] == "10.000%"
        assert coverage["unreached_ending_weight"] == f"{unreached:.3%}"
        assert coverage["nothing_filed_window_days"] == 30
        assert coverage["accounted_ending_weight"] == f"{1.0 - unreached:.3%}"
        return receipt, recommendation

    _receipt, complete = reviewed(0.0)
    assert complete.review_state is ReviewState.COMPLETE
    conclusions = {value.entity_id: value.conclusion for value in complete.issuer_conclusions}
    assert conclusions["QA99"] == "NOTHING_FILED"
    assert any(
        "10.00% is in holdings that filed nothing with the SEC in the last 30 days, which is "
        "not a finding of no risk." in line
        for line in complete.limitations
    )
    _receipt, partial = reviewed(0.05)
    assert partial.review_state is ReviewState.PARTIAL
    assert any("5.00% was not read." in line for line in partial.limitations)
    # A historical seal without the owner's no-filing partition cannot acquire
    # accounting proof from the UI or from the complement of its read share.
    legacy = _dossier(reviewed_ending_weight_coverage=0.4)
    receipt, recommendation = _seal(legacy, controlled_answer(legacy, ControlledRisk(ENTITY)))
    legacy_coverage = evidence_cro_body(
        project_published_review(
            recommendation=recommendation,
            dossier=legacy,
            receipt=receipt,
            percent=lambda value: f"{value:.3%}",
            change=lambda value: f"{value:+.3%}",
            book=EvidenceCroBook(
                authority="DEVELOPMENT_RESULT",
                explanation="Synthetic sealed book",
                result_hash=None,
            ),
        )
    )["coverage"]
    assert legacy_coverage["reviewed_ending_weight"] == "40.000%"
    assert legacy_coverage["nothing_filed_ending_weight"] is None
    assert legacy_coverage["unreached_ending_weight"] is None
    assert legacy_coverage["nothing_filed_window_days"] is None
    assert legacy_coverage["accounted_ending_weight"] is None


@pytest.mark.parametrize(
    ("dossier_kwargs", "limitation", "state"),
    (
        (
            {"missing_evidence": ("The Q3 filing was not available.",)},
            "The analysis records 1 gap(s) in what could be read",
            ReviewState.PARTIAL,
        ),
        (
            {"analyst_requires_human_review": True},
            "The analysis asked for a person to read its sources.",
            ReviewState.COMPLETE,
        ),
        (
            {"reviewed_ending_weight_coverage": 0.2},
            "The review read issuers carrying 20.00% of the book's ending weight, below the "
            "60% it is designed for",
            ReviewState.PARTIAL,
        ),
        (
            {"issuers": (), "findings": ()},
            "No issuer in this book could be reviewed; nothing was read.",
            ReviewState.UNAVAILABLE,
        ),
    ),
)
def test_deterministic_evidence_facts_are_stated_limits_never_gates(
    dossier_kwargs: dict[str, Any], limitation: str, state: ReviewState
) -> None:
    """requirement (S3): judge first, state the gaps. Each fact is the Host's
    and is written beside the route as a limitation; none stops the review,
    and none becomes a claim against an issuer."""

    dossier = _dossier(**dossier_kwargs)
    receipt, recommendation = _seal(dossier, controlled_answer(dossier, ControlledRisk(ENTITY)))

    expected = (
        PortfolioReviewRoute.MATERIAL_OBJECTION
        if dossier.findings
        else PortfolioReviewRoute.NO_MATERIAL_OBJECTION
    )
    assert recommendation.route is expected
    assert recommendation.review_state is state
    assert any(value.startswith(limitation) for value in receipt.outcome.limitations)
    assert recommendation.limitations == receipt.outcome.limitations
    assert not any(value.startswith("G-") for value in receipt.outcome.rule_ids)


def test_low_coverage_is_adjudicated_and_its_limit_stated() -> None:
    """requirement: a review of part of a book judges what it read. At 20%
    coverage the supported, high-severity risk still stands for its issuer --
    formerly a guard returned before adjudication -- and the coverage is a
    stated limit; an empty answer on the same dossier concludes nothing
    against the issuer and says no major negative was found."""

    dossier = _dossier(reviewed_ending_weight_coverage=0.2)
    receipt, recommendation = _seal(dossier, controlled_answer(dossier, ControlledRisk(ENTITY)))
    conclusion = {value.entity_id: value for value in recommendation.issuer_conclusions}[ENTITY]
    assert recommendation.route is PortfolioReviewRoute.MATERIAL_OBJECTION
    assert conclusion.conclusion == "ISSUE_STANDS"
    assert receipt.outcome.review_state is ReviewState.PARTIAL

    empty_receipt, empty = _seal(dossier, controlled_answer(dossier))
    assert empty.route is PortfolioReviewRoute.NO_MATERIAL_OBJECTION
    assert empty.reasons[0].startswith("No major negative was found in the evidence read.")
    assert {value.entity_id: value.conclusion for value in empty.issuer_conclusions}[ENTITY] == (
        "NO_ADVERSE_ISSUE"
    )
    assert "1 of 1 finding(s) were read by the reviewer and not named as a risk." in (
        empty_receipt.outcome.limitations
    )
    assert empty.policy_version == "chief_risk_officer.portfolio-evidence-review"


def test_a_sealed_review_reads_back_and_a_retired_policy_name_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: a published review reopens on restart under strict validation
    and keeps the identity it was stored under; a review naming a retired policy --
    the numbered names only the QA copies held, history since 2026-09-23 (D2) --
    is refused at the same boundary, sealed anew or read back.

    The recommendation goes through the real store, whose reader re-validates and
    requires the identity to match the name it was filed under (an earlier version
    of this test used `model_construct`, which skips the validator).
    """

    from alphalattice.evidence.alternative_evidence.publication.artifacts import (
        AlternativeEvidenceArtifactStore,
    )
    from alphalattice.oversight.chief_risk_officer.decision import portfolio_review
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        READABLE_PORTFOLIO_REVIEW_ROUTE_POLICIES,
    )

    assert READABLE_PORTFOLIO_REVIEW_ROUTE_POLICIES == (PORTFOLIO_REVIEW_ROUTE_POLICY,)
    store = AlternativeEvidenceArtifactStore(tmp_path)
    _receipt, recommendation = _seal(_dossier(), _submission())
    assert recommendation.policy_version == PORTFOLIO_REVIEW_ROUTE_POLICY
    assert type(recommendation).model_validate(recommendation.model_dump(mode="json")) == (
        recommendation
    )
    identity = recommendation.recommendation_hash
    store.publish("cro-review-recommendations", identity, recommendation)
    assert store.load("cro-review-recommendations", identity, type(recommendation)) == (
        recommendation
    )
    assert (store.root / "cro-review-recommendations" / f"{identity}.json").is_file()

    retired = (
        "chief_risk_officer.portfolio-review-route.v2",
        "chief_risk_officer.portfolio-review-route.v3",
    )
    for name in retired:
        stored = recommendation.model_dump(mode="json")
        stored["policy_version"] = name
        with pytest.raises(ValueError, match="policy_invalid"):
            PortfolioReviewRecommendation.model_validate(stored)
        monkeypatch.setattr(portfolio_review, "PORTFOLIO_REVIEW_ROUTE_POLICY", name)
        with pytest.raises(ValueError, match="policy_invalid"):
            _seal(_dossier(), _submission())

    # Nothing already filed was rewritten by any of that.
    again = store.load("cro-review-recommendations", identity, PortfolioReviewRecommendation)
    assert again.recommendation_hash == identity


def test_mapping_gaps_are_named_without_blocking_the_route() -> None:
    _receipt, recommendation = _seal(
        _dossier(mapping_failure_count=2, unavailable_reasons=("2 holding(s) did not map.",)),
        _submission(),
    )

    assert recommendation.route is PortfolioReviewRoute.NO_MATERIAL_OBJECTION
    assert recommendation.review_state is ReviewState.PARTIAL
    gap = next(
        value
        for value in recommendation.required_actions
        if value.action is RequiredActionKind.RESOLVE_ISSUER_MAPPING
    )
    assert not gap.blocking
    assert "2 holding(s)" in gap.reason


@pytest.mark.parametrize(
    ("authority", "action"),
    (
        (BookAuthority.DEVELOPMENT_RESULT, RequiredActionKind.RECONSIDER_CANDIDATE),
        (BookAuthority.FROZEN_CANDIDATE, RequiredActionKind.RECONSIDER_CANDIDATE),
        (BookAuthority.VALIDATED_HANDOFF, RequiredActionKind.DO_NOT_ACTIVATE),
    ),
)
def test_the_required_action_follows_the_book_authority(
    authority: BookAuthority, action: RequiredActionKind
) -> None:
    """One route grammar; what an objection asks for depends on what the book is."""

    _receipt, recommendation = _seal(
        _dossier(book_authority=authority), _submission(_issue(FINDING, ENTITY))
    )

    assert recommendation.route is PortfolioReviewRoute.MATERIAL_OBJECTION
    assert recommendation.book_authority is authority
    assert recommendation.required_actions[0].action is action
    assert recommendation.required_actions[0].entity_id == ENTITY
    assert recommendation.required_actions[0].blocking
    assert recommendation.action_activation == "NOT_AUTHORIZED"
    assert recommendation.issuer_conclusions[0].conclusion == "ISSUE_STANDS"


SECOND_ENTITY = "MSFT"


def _two_issuer_issue(
    entities: tuple[str, ...] = (ENTITY, SECOND_ENTITY),
) -> PortfolioReviewMaterialIssue:
    """One adverse issue naming both issuers, as the contract permits."""

    return _issue(FINDING, ENTITY, entities=entities)


def _two_issuer_dossier(
    *,
    first_band: ExposureBand = ExposureBand.LOW,
    second_band: ExposureBand = ExposureBand.CRITICAL,
) -> PortfolioReviewDossier:
    """One finding naming two issuers whose exposures are nothing alike."""

    return _dossier(
        issuers=(
            _issuer(ENTITY, band=first_band, weight_rank=7),
            _issuer(SECOND_ENTITY, band=second_band, weight_rank=1),
        ),
        findings=(_finding(entities=(ENTITY, SECOND_ENTITY)),),
    )


# regression: an issue may name up to eight issuers, and they do not share an
# exposure band. Evaluating only `affected_entities[0]` at the maximum band
# across all of them charged a small position with a large one's exposure and
# dropped the exposed issuer out of the conclusions entirely.
def test_a_two_issuer_issue_is_evaluated_once_per_issuer_at_its_own_band() -> None:
    receipt, recommendation = _seal(_two_issuer_dossier(), _submission(_two_issuer_issue()))

    evaluations = {value.entity_id: value for value in receipt.outcome.issue_evaluations}
    assert set(evaluations) == {ENTITY, SECOND_ENTITY}, "both named issuers are evaluated"
    assert evaluations[ENTITY].exposure_band is ExposureBand.LOW
    assert evaluations[SECOND_ENTITY].exposure_band is ExposureBand.CRITICAL

    # The exposed issuer earns the objection; the small position does not.
    assert evaluations[SECOND_ENTITY].route is PortfolioReviewRoute.MATERIAL_OBJECTION
    assert evaluations[SECOND_ENTITY].rule_id == "R-OBJECTION"
    assert evaluations[ENTITY].route is PortfolioReviewRoute.ACCEPT_WITH_LIMITS
    assert evaluations[ENTITY].rule_id == "R-LIMITS"
    assert recommendation.route is PortfolioReviewRoute.MATERIAL_OBJECTION

    conclusions = {value.entity_id: value for value in recommendation.issuer_conclusions}
    assert conclusions[SECOND_ENTITY].conclusion == "ISSUE_STANDS"
    assert conclusions[SECOND_ENTITY].exposure_band is ExposureBand.CRITICAL
    assert conclusions[ENTITY].conclusion == "ISSUE_NOTED_WITHIN_LIMITS"
    assert conclusions[ENTITY].exposure_band is ExposureBand.LOW
    assert conclusions[SECOND_ENTITY].adverse_issue_count == 1
    assert conclusions[ENTITY].adverse_issue_count == 1

    # The blocking action names the issuer that is actually exposed.
    blocking = tuple(value for value in recommendation.required_actions if value.blocking)
    assert tuple(value.entity_id for value in blocking) == (SECOND_ENTITY,)
    assert blocking[0].action is RequiredActionKind.RECONSIDER_CANDIDATE


def test_a_high_exposure_second_issuer_cannot_be_hidden_or_reattributed() -> None:
    """The exposed issuer is never silent, and never speaks through another."""

    _receipt, recommendation = _seal(_two_issuer_dossier(), _submission(_two_issuer_issue()))

    conclusions = {value.entity_id: value for value in recommendation.issuer_conclusions}
    assert conclusions[SECOND_ENTITY].conclusion != "NO_ADVERSE_ISSUE"
    assert conclusions[SECOND_ENTITY].finding_count == 1
    # The low-exposure issuer is never charged with the other's exposure.
    assert all(
        value.exposure_band is ExposureBand.LOW
        for value in _receipt.outcome.issue_evaluations
        if value.entity_id == ENTITY
    )
    assert SECOND_ENTITY in " ".join(recommendation.reasons)


def test_the_order_the_actor_lists_its_issuers_in_does_not_change_the_outcome() -> None:
    dossier = _two_issuer_dossier()
    forward_receipt, forward_recommendation = _seal(
        dossier, _submission(_two_issuer_issue((ENTITY, SECOND_ENTITY)))
    )
    reversed_receipt, reversed_recommendation = _seal(
        dossier, _submission(_two_issuer_issue((SECOND_ENTITY, ENTITY)))
    )

    assert forward_receipt.outcome == reversed_receipt.outcome
    assert forward_recommendation.route is reversed_recommendation.route
    assert forward_recommendation.issuer_conclusions == reversed_recommendation.issuer_conclusions
    assert forward_recommendation.required_actions == reversed_recommendation.required_actions


def test_a_single_issuer_issue_is_unchanged_by_per_issuer_evaluation() -> None:
    receipt, recommendation = _seal(_dossier(), _submission(_issue(FINDING, ENTITY)))

    assert len(receipt.outcome.issue_evaluations) == 1
    assert receipt.outcome.issue_evaluations[0].entity_id == ENTITY
    assert receipt.outcome.issue_evaluations[0].exposure_band is ExposureBand.HIGH
    assert recommendation.route is PortfolioReviewRoute.MATERIAL_OBJECTION
    assert recommendation.issuer_conclusions[0].conclusion == "ISSUE_STANDS"


SECOND_FINDING = "FIND-MSFT-LEGAL-REGULATORY"


def _crowded_dossier() -> PortfolioReviewDossier:
    """Two exposed issuers, each with its own corroborated finding."""

    return _dossier(
        issuers=(
            _issuer(ENTITY, band=ExposureBand.CRITICAL, weight_rank=1),
            _issuer(SECOND_ENTITY, band=ExposureBand.HIGH, weight_rank=2),
        ),
        findings=(
            _finding(),
            _finding(handle=SECOND_FINDING, entity=SECOND_ENTITY),
        ),
    )


def _crowded_issues() -> tuple[PortfolioReviewMaterialIssue, ...]:
    """Eight objections against one issuer, then one against another.

    Nine triggering evaluations against an eight-item presentation bound: the
    ninth issuer is the one a per-issue slice loses.
    """

    first = tuple(_issue(FINDING, ENTITY, handle=f"ISSUE-{index:02d}") for index in range(1, 9))
    return (*first, _issue(SECOND_FINDING, SECOND_ENTITY, handle="ISSUE-09"))


def _issuer_actions(recommendation: PortfolioReviewRecommendation) -> set[tuple[str | None, str]]:
    return {(value.entity_id, str(value.action)) for value in recommendation.required_actions}


# regression: reasons and actions are per issuer, not per issue. Slicing the
# triggering evaluations to eight let one issuer's eight objections fill the
# presentation bound and drop a ninth issuer whose conclusion still stood.
def test_every_triggering_issuer_earns_an_action_however_many_issues_precede_it() -> None:
    dossier = _crowded_dossier()
    receipt, recommendation = _seal(dossier, _submission(*_crowded_issues()))

    assert recommendation.route is PortfolioReviewRoute.MATERIAL_OBJECTION
    conclusions = {value.entity_id: value.conclusion for value in recommendation.issuer_conclusions}
    assert conclusions == {ENTITY: "ISSUE_STANDS", SECOND_ENTITY: "ISSUE_STANDS"}

    assert _issuer_actions(recommendation) == {
        (ENTITY, str(RequiredActionKind.RECONSIDER_CANDIDATE)),
        (SECOND_ENTITY, str(RequiredActionKind.RECONSIDER_CANDIDATE)),
    }, "an issuer whose conclusion stands is never left without an action"
    named = " ".join(recommendation.reasons)
    assert ENTITY in named and SECOND_ENTITY in named

    # Every per-issue evaluation survives: nine issues, nine evaluations.
    assert len(receipt.outcome.issue_evaluations) == len(_crowded_issues())
    assert {value.entity_id for value in receipt.outcome.issue_evaluations} == {
        ENTITY,
        SECOND_ENTITY,
    }


def test_repeated_issues_for_one_issuer_do_not_consume_another_issuers_slot() -> None:
    dossier = _crowded_dossier()
    sparse = _seal(dossier, _submission(_issue(FINDING, ENTITY), _crowded_issues()[-1]))[1]
    crowded = _seal(dossier, _submission(*_crowded_issues()))[1]

    assert _issuer_actions(sparse) == _issuer_actions(crowded)
    assert len(crowded.required_actions) == len(sparse.required_actions)
    # The per-issue counts differ honestly; what may not differ is which issuers
    # are concluded on and how.
    assert {value.entity_id: value.conclusion for value in crowded.issuer_conclusions} == {
        value.entity_id: value.conclusion for value in sparse.issuer_conclusions
    }
    assert [value.adverse_issue_count for value in crowded.issuer_conclusions] == [8, 1]
    assert [value.adverse_issue_count for value in sparse.issuer_conclusions] == [1, 1]


def test_reversing_the_issue_order_does_not_change_the_issuer_action_set() -> None:
    dossier = _crowded_dossier()
    issues = _crowded_issues()
    forward_receipt, forward = _seal(dossier, _submission(*issues))
    reversed_receipt, backward = _seal(dossier, _submission(*reversed(issues)))

    assert _issuer_actions(forward) == _issuer_actions(backward)
    assert forward.required_actions == backward.required_actions
    assert forward.issuer_conclusions == backward.issuer_conclusions
    assert forward_receipt.outcome.rule_ids == reversed_receipt.outcome.rule_ids
    # The Host numbers an answer's risks in the order written, so a reason
    # names a different handle; what it says about each issuer may not move.
    assert [re.sub(r"ISSUE-\d+", "ISSUE", value) for value in forward.reasons] == [
        re.sub(r"ISSUE-\d+", "ISSUE", value) for value in backward.reasons
    ]
    assert sorted(
        (value.entity_id, value.route, value.rule_id)
        for value in forward_receipt.outcome.issue_evaluations
    ) == sorted(
        (value.entity_id, value.route, value.rule_id)
        for value in reversed_receipt.outcome.issue_evaluations
    )


def test_no_material_objection_never_says_safe() -> None:
    _receipt, recommendation = _seal(_dossier(), _submission())

    assert recommendation.route is PortfolioReviewRoute.NO_MATERIAL_OBJECTION
    rendered = " ".join((*recommendation.reasons, *recommendation.claim_limits)).casefold()
    assert "not a statement that the portfolio or any issuer is safe" in rendered
    assert "limited to the reviewed issuers" in rendered
    assert "priced in" in rendered
    assert recommendation.action_activation == "NOT_AUTHORIZED"


def test_an_answer_that_lost_items_is_never_published_as_clean() -> None:
    """requirement (first release, false assurance): items the Host dropped
    from the reviewer's answer after two corrections are unfinished work, not
    a clean review. An empty accepted part with a dropped risk does not say no
    major negative was found: the review is partial, a person is asked to read
    the answer as written, and the drop is a stated limitation. A route the
    admitted risks set higher stands; an answer that lost nothing is unchanged."""

    dossier = _dossier()
    lost = (AnswerProblem(item=1, text="F9 is not a finding of this bundle."),)
    receipt, recommendation = _seal(dossier, controlled_answer(dossier), dropped=lost)
    assert recommendation.route is PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED
    assert recommendation.review_state is ReviewState.PARTIAL
    assert not any("No major negative" in value for value in recommendation.reasons)
    assert receipt.outcome.rule_ids == ("R-ANSWER-INCOMPLETE",)
    assert [value.action for value in recommendation.required_actions] == [
        RequiredActionKind.HUMAN_REVIEW
    ]
    assert not any(value.blocking for value in recommendation.required_actions)
    assert (
        "The reviewer's last answer had 1 item(s) the Host could not admit after two "
        "corrections; they are not part of this review. Item 1: F9 is not a finding of "
        "this bundle."
    ) in receipt.outcome.limitations

    _clean_receipt, clean = _seal(dossier, controlled_answer(dossier))
    assert clean.route is PortfolioReviewRoute.NO_MATERIAL_OBJECTION
    assert clean.review_state is ReviewState.COMPLETE

    standing_receipt, standing = _seal(
        dossier, controlled_answer(dossier, ControlledRisk(ENTITY)), dropped=lost
    )
    assert standing.route is PortfolioReviewRoute.MATERIAL_OBJECTION
    assert standing.review_state is ReviewState.PARTIAL
    assert "R-ANSWER-INCOMPLETE" not in standing_receipt.outcome.rule_ids
    assert any(
        value.startswith("The reviewer's last answer had 1 item(s)")
        for value in (standing_receipt.outcome.limitations)
    )


def test_expired_evidence_is_refused_never_routed() -> None:
    """requirement (S3): stale evidence is handled before a review; no route
    asks anyone to refresh."""

    with pytest.raises(ValueError, match="review_evidence_expired"):
        _seal(_dossier(), _submission(), evidence_is_current=False)


def test_an_ungrounded_citation_is_refused_before_it_can_route() -> None:
    """A prompt-injected finding alias cannot produce any route, adverse or not:
    the screen names it and the sealer refuses an answer holding it."""

    injected = {
        "risks": [
            {
                "findings": ["F99"],
                "why": "Injected by document text.",
                "severity": "HIGH",
                "confidence": "SUPPORTED",
                "recommendation": "Sell everything.",
            }
        ]
    }
    screened = screen_review_answer(injected, dossier=_dossier())
    assert screened.items == ()
    assert [value.text for value in screened.problems] == ["F99 is not a finding of this bundle."]
    with pytest.raises(CRODecisionAuthorityError, match="answer_invalid"):
        _seal(_dossier(), PortfolioReviewAnswer.model_validate(injected))


def test_the_affected_issuers_are_the_cited_findings_own() -> None:
    """The answer names findings, never issuers: the Host derives who is
    affected from what is cited, so no answer can reach an issuer outside
    the dossier."""

    dossier = _crowded_dossier()
    receipt, _ = _seal(dossier, controlled_answer(dossier, ControlledRisk(SECOND_ENTITY)))

    (issue,) = receipt.submission.material_issues
    assert issue.affected_entities == (SECOND_ENTITY,)
    assert issue.issue_handle == "ISSUE-001"
    assert issue.portfolio_mitigation is None
    assert receipt.answer is not None and receipt.answer.risks[0].findings == ("F2",)


def _agent_binding() -> AgentExecutionBinding:
    return AgentExecutionBinding(
        profile_id="qa-scripted",
        mode="STRUCTURED_OUTPUT",
        profile_hash="a" * 64,
        document_hash="b" * 64,
        response_protocol="TOOL_FREE_STRUCTURED_OUTPUT",
        response_protocol_hash="c" * 64,
        concrete_schema_hash="d" * 64,
    )


def test_every_actor_kind_reaches_the_same_seam() -> None:
    """Human, Installed Agent and automation: one sealer, one route, one outcome."""

    submission = _submission(_issue(FINDING, ENTITY))
    human_receipt, human = _seal(_dossier(), submission)
    agent_receipt, agent = _seal(
        _dossier(),
        submission,
        actor_kind=ActorKind.INSTALLED_AGENT,
        agent_execution=_agent_binding(),
    )
    automation_receipt, automation = _seal(
        _dossier(), submission, actor_kind=ActorKind.EXTERNAL_AUTOMATION
    )

    assert human_receipt.outcome == agent_receipt.outcome == automation_receipt.outcome
    assert {human.actor_kind, agent.actor_kind, automation.actor_kind} == {
        ActorKind.HUMAN,
        ActorKind.INSTALLED_AGENT,
        ActorKind.EXTERNAL_AUTOMATION,
    }
    for value in (agent, automation):
        assert value.route is human.route
        assert value.reasons == human.reasons
        assert value.required_actions == human.required_actions
        assert value.issuer_conclusions == human.issuer_conclusions


def test_the_review_key_binds_no_clock_and_no_pointer() -> None:
    """Reuse must not decay with time, and must not follow a mutable marker."""

    payload = portfolio_review_key_payload(
        dossier=_dossier(),
        decision_policy_hash="0" * 64,
        typed_user_authority="qa reviewer",
        actor_kind="HUMAN",
        actor_id="gate-9c5",
        process_binding_hash="1" * 64,
        response_schema_hash="2" * 64,
    )

    rendered = str(payload).casefold()
    for banned in ("now", "current", "pointer", "timestamp", "published_at", "marker"):
        assert banned not in rendered, banned
    assert payload["execution_semantics"] == "TOOL_FREE_SINGLE_SEMANTIC_EXECUTION"
    assert "submission_hash" not in payload
    assert payload["book_authority"] == str(BookAuthority.DEVELOPMENT_RESULT)

    envelope, _goal, plan = portfolio_review_task_contract(review_key_payload=payload)
    assert len(plan.work_items) == 3
    assert envelope.input_schema_id == "cro-portfolio-review-authority"


# ============================================== the actor stage deadline


class _StageClock:
    """A scripted monotonic clock: one reading per call, then it holds."""

    def __init__(self, readings: tuple[float, ...]) -> None:
        self.readings = list(readings)
        self.observed: list[float] = []

    def __call__(self) -> float:
        value = self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]
        self.observed.append(value)
        return value


STAGE_BUDGET = 10.0


class _ManualClock:
    """A clock that moves only when something spends time, not when it is read.

    Reading it cannot be what advances it: the question is whether the stage
    anchor is taken before or after preparation, and a clock scripted by read
    order answers that question the same way either way.
    """

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# regression: the budget covers the whole actor stage. Anchoring the deadline
# after profile loading and dossier rendering handed the first model a full
# budget the stage had already partly spent.


# regression: the stage deadline is one budget for the whole actor stage, not a
# timeout handed to each attempt. Passing the full deadline to the repair let a
# ten-second stage block for twenty, because the Task can only observe the
# overrun after the actor returns.


# ================================================== contraction and closure


def _module_scope_imports(tree: ast.Module) -> set[str]:
    """What importing this module actually loads.

    Only module-scope statements count. An import inside a function body runs
    when that function is called, and an import under `TYPE_CHECKING` never
    runs at all, so neither is loaded by importing the module. The Host names
    the Agent adapters in annotations and constructs them inside the admission
    function; a reader of this closure must not see either as a load.
    """

    found: set[str] = set()
    pending: list[ast.stmt] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.If):
            test = node.test
            named = getattr(test, "id", None) or getattr(test, "attr", None)
            if named != "TYPE_CHECKING":
                pending.extend(node.body)
                pending.extend(node.orelse)
        elif isinstance(node, ast.Try):
            pending.extend(node.body)
            pending.extend(node.orelse)
            pending.extend(node.finalbody)
            for handler in node.handlers:
                pending.extend(handler.body)
    return found


def _internal_import_closure(start: str) -> set[str]:
    """Every `alphalattice` module importing one entry point actually loads."""

    src = REPO_ROOT / "src"
    imports: dict[str, set[str]] = {}
    for path in src.rglob("*.py"):
        parts = list(path.relative_to(src).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        tree = ast.parse(path.read_bytes().decode("utf-8"))
        imports[".".join(parts)] = _module_scope_imports(tree)

    seen: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(
            target for target in imports.get(current, ()) if target.startswith("alphalattice.")
        )
    return seen


def test_the_local_web_composition_reaches_the_review_seam_and_no_agent_adapter() -> None:
    """The owners are on the product's own import closure; the Agent adapters are not.

    A route that exists only in tests is not a product route. And a Host that
    imported an Agent adapter would load LangChain to answer a GET.
    """

    closure = _internal_import_closure(
        "alphalattice.control.product_host.composition.local_web_session"
    )
    for owner in (
        "alphalattice.control.product_host.composition.evidence_review_application",
        "alphalattice.oversight.chief_risk_officer.decision.book_evidence",
        "alphalattice.oversight.chief_risk_officer.portfolio_evidence.coverage.issuer_scope",
        "alphalattice.oversight.chief_risk_officer.decision.portfolio_review",
        "alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task",
        "alphalattice.oversight.chief_risk_officer.publication.portfolio_review",
        "alphalattice.interface.local_application.evidence_cro",
    ):
        assert owner in closure, owner
    outside = sorted(
        name
        for name in closure
        if ".agent." in name
        and name.startswith(
            ("alphalattice.oversight.chief_risk_officer", "alphalattice.evidence.alternative")
        )
    )
    assert outside == [], outside
    legacy = sorted(name for name in closure if "opening" in name or "validated_portfolio" in name)
    assert legacy == [], legacy


def test_a_renderer_change_rotates_no_review_identity() -> None:
    """No rendering surface is inside any identity on this path."""

    binding = build_portfolio_review_policy_binding(REPO_ROOT)
    decision_only = source_rule_closure_hash(
        root=REPO_ROOT.resolve(),
        tracked_paths=(
            "src/alphalattice/oversight/chief_risk_officer/decision/portfolio_review.py",
            "src/alphalattice/oversight/chief_risk_officer/decision/submissions.py",
        ),
        semantic_owner="chief_risk_officer",
        numerical_role="CRO_HOST_POLICY",
    )
    with_renderer = source_rule_closure_hash(
        root=REPO_ROOT.resolve(),
        tracked_paths=(
            "src/alphalattice/oversight/chief_risk_officer/decision/portfolio_review.py",
            "src/alphalattice/oversight/chief_risk_officer/decision/submissions.py",
            "src/alphalattice/interface/local_application/evidence_cro.py",
        ),
        semantic_owner="chief_risk_officer",
        numerical_role="CRO_HOST_POLICY",
    )

    assert binding.validation_source_hash == decision_only
    assert with_renderer != decision_only, "the control arm must actually differ"
    assert _dossier() == _dossier(), "the dossier is a pure function of its inputs"


def test_an_open_issue_is_stated_again_resolved_or_counted_as_last_assessed() -> None:
    """requirement (W3): an open issue of the issuers' register stands in front
    of the reviewer until a review resolves it. A risk that carries it states it
    again as assessed now; a resolution citing a finding closes it; one the
    reviewer says nothing of counts as last assessed -- routed, never dropped.
    An answer naming an issue the bundle does not hold is told so."""

    from alphalattice.evidence.alternative_evidence.contracts import seal_contract
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        CROEvidenceInterpretation,
        CRORiskConfidence,
        CRORiskSeverity,
        CROSeverityIfTrue,
        PortfolioReviewAnswer,
        PortfolioReviewAnswerResolution,
        PortfolioReviewAnswerRisk,
        PortfolioReviewDossier,
        PortfolioReviewOpenIssue,
    )
    from alphalattice.oversight.chief_risk_officer.decision.submissions import (
        screen_review_answer,
    )

    base = _dossier()
    opened = PortfolioReviewOpenIssue(
        open_issue_handle="OPEN-" + "A" * 12,
        affected_entities=(ENTITY,),
        raised_on=(_NOW - timedelta(days=40)).date(),
        assessed_on=(_NOW - timedelta(days=5)).date(),
        cited_finding_handles=(base.findings[0].finding_handle,),
        causal_channel="An earlier review's reason.",
        severity_if_true=CROSeverityIfTrue.HIGH,
        evidence_interpretation=CROEvidenceInterpretation.SUPPORTED,
        recommendation="Hold at the current weight.",
    )
    dossier = seal_contract(
        PortfolioReviewDossier,
        "dossier_hash",
        **{**base.model_dump(exclude={"dossier_hash"}), "open_issues": (opened,)},
    )
    silent, routed = _seal(dossier, PortfolioReviewAnswer())
    (issue,) = silent.submission.material_issues
    assert issue.open_issue_handle == opened.open_issue_handle
    assert issue.carried_on == opened.assessed_on
    assert routed.route is PortfolioReviewRoute.MATERIAL_OBJECTION, "it counts as it stood"

    again = PortfolioReviewAnswer(
        risks=(
            PortfolioReviewAnswerRisk(
                findings=("F1",),
                why="The latest filing narrows it.",
                severity=CRORiskSeverity.LOW,
                confidence=CRORiskConfidence.SUPPORTED,
                recommendation="Watch the next report.",
                carries="O1",
            ),
        )
    )
    stated, routed = _seal(dossier, again)
    (issue,) = stated.submission.material_issues
    assert issue.open_issue_handle == opened.open_issue_handle and issue.carried_on is None
    assert routed.route is PortfolioReviewRoute.NO_MATERIAL_OBJECTION

    closed = PortfolioReviewAnswer(
        resolved=(
            PortfolioReviewAnswerResolution(
                issue="O1", findings=("F1",), why="The later filing settles it."
            ),
        )
    )
    resolved, _routed = _seal(dossier, closed)
    assert resolved.submission.material_issues == ()
    assert [value.open_issue_handle for value in resolved.submission.resolved_issues] == [
        opened.open_issue_handle
    ]

    screened = screen_review_answer(
        {
            "risks": [{**again.risks[0].model_dump(mode="json"), "carries": "O2"}],
            "resolved": [{"issue": "O1", "findings": ["F9"], "why": "No such finding."}],
        },
        dossier=dossier,
    )
    words = " ".join(value.text for value in screened.problems)
    assert "O2 is not an open issue of this bundle." in words
    assert "F9 is not in this bundle." in words
    assert not screened.items and not screened.resolved


def test_a_book_holding_more_open_issues_than_an_answer_can_address_still_seals() -> None:
    """requirement (CS, V211): an answer states at most sixteen risks and
    resolves at most sixteen issues, and every open issue it leaves unnamed is
    carried as last assessed. A register of more open issues than that once
    made every answer unsealable (`material_issues=too_long`, AX1b: 42 open
    issues, 49 to 51 issues sealed). The carried ones sit outside the
    reviewer's bound, each is routed, none is dropped; a seventeenth issue of
    the reviewer's own is refused by name."""

    from alphalattice.evidence.alternative_evidence.contracts import seal_contract
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        MAXIMUM_REVIEWER_ISSUES,
        CROEvidenceInterpretation,
        CRORiskConfidence,
        CRORiskSeverity,
        CROSeverityIfTrue,
        PortfolioReviewAnswer,
        PortfolioReviewAnswerRisk,
        PortfolioReviewAssessmentSubmission,
        PortfolioReviewDossier,
        PortfolioReviewOpenIssue,
    )

    base = _dossier()
    opened = tuple(
        PortfolioReviewOpenIssue(
            open_issue_handle=f"OPEN-{index:012X}",
            affected_entities=(ENTITY,),
            raised_on=(_NOW - timedelta(days=40)).date(),
            assessed_on=(_NOW - timedelta(days=5)).date(),
            cited_finding_handles=(base.findings[0].finding_handle,),
            causal_channel=f"An earlier review's reason, number {index}.",
            severity_if_true=CROSeverityIfTrue.LOW,
            evidence_interpretation=CROEvidenceInterpretation.SUPPORTED,
        )
        for index in range(42)
    )
    dossier = seal_contract(
        PortfolioReviewDossier,
        "dossier_hash",
        **{**base.model_dump(exclude={"dossier_hash"}), "open_issues": opened},
    )
    answer = PortfolioReviewAnswer(
        risks=tuple(
            PortfolioReviewAnswerRisk(
                findings=("F1",),
                why=f"The cited finding is adverse for a held issuer, reading {index}.",
                severity=CRORiskSeverity.LOW,
                confidence=CRORiskConfidence.SUPPORTED,
                recommendation="Watch the next report.",
            )
            for index in range(MAXIMUM_REVIEWER_ISSUES)
        )
    )

    receipt, _recommendation = _seal(dossier, answer)
    issues = receipt.submission.material_issues
    carried = [value for value in issues if value.carried_on is not None]
    assert len(issues) - len(carried) == MAXIMUM_REVIEWER_ISSUES
    assert sorted(value.open_issue_handle or "" for value in carried) == sorted(
        value.open_issue_handle for value in opened
    ), "every open issue the answer left unnamed is carried, none dropped"
    evaluated = {value.issue_handle for value in receipt.outcome.issue_evaluations}
    assert {value.issue_handle for value in carried} <= evaluated, "each is routed as it stood"

    written = receipt.submission.model_dump(mode="json")
    own = next(value for value in written["material_issues"] if "carried_on" not in value)
    seventeen = {
        **written,
        "material_issues": [*written["material_issues"], {**own, "issue_handle": "ISSUE-999"}],
    }
    with pytest.raises(
        ValueError, match=re.escape("chief_risk_officer.submission_own_issues_exceeded")
    ):
        PortfolioReviewAssessmentSubmission.model_validate(seventeen)


def test_a_seal_that_meets_a_record_bound_is_refused_with_its_bounds() -> None:
    """requirement (CS, V212): when an answer within its format would seal a
    record past one of the record's bounds, the agent is told which bound, how
    many and how many are allowed, and to tell its lead -- never a bare
    `request_refused:<field>=too_long` with no way on."""

    from pydantic import ValidationError

    from alphalattice.control.product_host.composition.evidence_review_bundles import (
        agent_bundle_refusal,
        sealed_bounds_exceeded,
    )
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        MAXIMUM_OPEN_ISSUES,
        MAXIMUM_REVIEWER_ISSUES,
        CRORiskConfidence,
        CRORiskSeverity,
        PortfolioReviewAnswer,
        PortfolioReviewAnswerRisk,
        PortfolioReviewAssessmentSubmission,
    )

    risk = PortfolioReviewAnswerRisk(
        findings=("F1",),
        why="The cited finding is adverse for a held issuer.",
        severity=CRORiskSeverity.LOW,
        confidence=CRORiskConfidence.SUPPORTED,
        recommendation="Watch the next report.",
    )
    receipt, _recommendation = _seal(_dossier(), PortfolioReviewAnswer(risks=(risk,)))
    written = receipt.submission.model_dump(mode="json")
    issue = written["material_issues"][0]
    limit = MAXIMUM_REVIEWER_ISSUES + MAXIMUM_OPEN_ISSUES
    past = {
        **written,
        "material_issues": [
            {**issue, "issue_handle": f"ISSUE-{index:03d}", "carried_on": "2026-08-01"}
            for index in range(limit + 1)
        ],
    }
    with pytest.raises(ValidationError) as caught:
        PortfolioReviewAssessmentSubmission.model_validate(past)

    bounds = sealed_bounds_exceeded(caught.value)
    assert bounds == ({"field": "material_issues", "count": limit + 1, "at_most": limit},)
    refusal = agent_bundle_refusal("agent_bundle.seal_bound_exceeded", role="CRO", bounds=bounds)
    assert refusal["failure_code"] == "agent_bundle.seal_bound_exceeded"
    assert refusal["bounds"] == [dict(value) for value in bounds]
    assert refusal["next_action"] == "TELL_YOUR_LEAD" and "Host's limit" in str(refusal["message"])


def test_a_resolution_past_the_answer_bound_is_corrected_by_its_author() -> None:
    """regression (V576, an outside review at d7f94002): seventeen legitimate resolutions passed
    the screen, which bounded the risks but counted no resolution, and the record's model then
    refused them as `agent_bundle.seal_bound_exceeded`, whose words call it the Host's limit and
    send the specialist to its lead; the README stated no bound for `resolved`. The screen reads
    the first sixteen and asks its author to correct, as it does for risks, and the README states
    the bound, the answer model's own."""

    from alphalattice.evidence.alternative_evidence.contracts import seal_contract
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        CROEvidenceInterpretation,
        CROSeverityIfTrue,
        PortfolioReviewDossier,
        PortfolioReviewOpenIssue,
    )
    from alphalattice.oversight.chief_risk_officer.decision.submissions import (
        MAXIMUM_ANSWER_RESOLUTIONS,
    )
    from alphalattice.oversight.chief_risk_officer.decision.views import render_review_bundle

    bound = MAXIMUM_ANSWER_RESOLUTIONS
    assert bound == 16
    base = _dossier()
    issues = tuple(
        PortfolioReviewOpenIssue(
            open_issue_handle=f"OPEN-{number:012X}",
            affected_entities=(ENTITY,),
            raised_on=(_NOW - timedelta(days=40)).date(),
            assessed_on=(_NOW - timedelta(days=5)).date(),
            cited_finding_handles=(base.findings[0].finding_handle,),
            causal_channel="An earlier review's reason.",
            severity_if_true=CROSeverityIfTrue.HIGH,
            evidence_interpretation=CROEvidenceInterpretation.SUPPORTED,
            recommendation="Hold at the current weight.",
        )
        for number in range(1, bound + 2)
    )
    dossier = seal_contract(
        PortfolioReviewDossier,
        "dossier_hash",
        **{**base.model_dump(exclude={"dossier_hash"}), "open_issues": issues},
    )
    written = {
        "resolved": [
            {"issue": f"O{number}", "findings": ["F1"], "why": "The later filing settles it."}
            for number in range(1, bound + 2)
        ]
    }
    screened = screen_review_answer(written, dossier=dossier)
    words = " ".join(value.text for value in screened.problems)
    assert f"The answer has {bound + 1} resolutions; only the first {bound} are read." in words
    assert len(screened.resolved) == bound
    readme = dict(render_review_bundle(dossier, task_procedure="Review the book.").files)
    lines = readme["README.md"].splitlines()
    stated = next(line for line in lines if line.startswith("- `resolved`"))
    assert f"at most {bound} in one answer" in stated, stated


def test_every_bound_a_specialist_answer_sets_is_screened_and_stated() -> None:
    """requirement (V576's class, TE12): an answer past a bound its own model sets is corrected
    by its author, never refused as the Host's limit. Every top-level bound of the CRO's and the
    Evidence Analyst's answer models is the number its screen reads, and its README states it
    from that number."""

    import inspect

    from alphalattice.evidence.alternative_evidence.analysis import (
        submissions as analyst_screen,
    )
    from alphalattice.evidence.alternative_evidence.analysis import views as analyst_views
    from alphalattice.evidence.alternative_evidence.analysis.contracts import (
        ANALYST_ANSWER_TEXT_FIELDS,
        AlternativeEvidenceAnalystAnswer,
    )
    from alphalattice.oversight.chief_risk_officer.decision import submissions as cro_screen
    from alphalattice.oversight.chief_risk_officer.decision import views as cro_views
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        REVIEW_ANSWER_TEXT_FIELDS,
        PortfolioReviewAnswer,
    )

    def bounds(model: Any) -> dict[str, int]:
        return {
            name: limit
            for name, field in model.model_fields.items()
            for value in field.metadata
            if isinstance(limit := getattr(value, "max_length", None), int)
        }

    cro = {
        "risks": ("MAXIMUM_ANSWER_RISKS", cro_screen.MAXIMUM_ANSWER_RISKS),
        "resolved": ("MAXIMUM_ANSWER_RESOLUTIONS", cro_screen.MAXIMUM_ANSWER_RESOLUTIONS),
        "summary": ("REVIEW_ANSWER_TEXT_FIELDS", REVIEW_ANSWER_TEXT_FIELDS["summary"]),
    }
    analyst = {
        "findings": ("MAXIMUM_ANSWER_FINDINGS", analyst_screen.MAXIMUM_ANSWER_FINDINGS),
        "notes": ("ANALYST_ANSWER_TEXT_FIELDS", ANALYST_ANSWER_TEXT_FIELDS["notes"]),
    }
    for model, screened, views in (
        (PortfolioReviewAnswer, cro, cro_views),
        (AlternativeEvidenceAnalystAnswer, analyst, analyst_views),
    ):
        assert bounds(model) == {name: number for name, (_, number) in screened.items()}, model
        readme = inspect.getsource(views)
        for name, (constant, _number) in screened.items():
            assert f"`{name}`" in readme and constant in readme, (model, name)


def test_published_issuer_summary_separates_generated_words_from_authored_text() -> None:
    """V671: punctuation in a carried CRO reason is never a translation boundary."""
    from alphalattice.interface.local_application.evidence_cro import (
        EvidenceCroBook,
        evidence_cro_body,
        project_published_review,
    )
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        CROSeverityIfTrue,
        PortfolioReviewOpenIssue,
    )

    base = _dossier()
    authored = (
        "Author's words; open issue from 2026-01-01: unchanged; no new filing; read 2026-02-01."
    )
    opened = PortfolioReviewOpenIssue(
        open_issue_handle="OPEN-" + "A" * 12,
        affected_entities=(ENTITY,),
        raised_on=(_NOW - timedelta(days=40)).date(),
        assessed_on=(_NOW - timedelta(days=5)).date(),
        cited_finding_handles=(base.findings[0].finding_handle,),
        causal_channel=authored,
        severity_if_true=CROSeverityIfTrue.HIGH,
        evidence_interpretation=CROEvidenceInterpretation.SUPPORTED,
        recommendation="Hold at the current weight.",
    )
    dossier = seal_contract(
        PortfolioReviewDossier,
        "dossier_hash",
        **{**base.model_dump(exclude={"dossier_hash"}), "open_issues": (opened,)},
    )
    receipt, recommendation = _seal(dossier, PortfolioReviewAnswer())
    projection = project_published_review(
        recommendation=recommendation,
        dossier=dossier,
        receipt=receipt,
        percent=lambda value: str(value),
        change=lambda value: str(value),
        book=EvidenceCroBook(authority="DEVELOPMENT_RESULT", explanation="", result_hash="a" * 64),
    )
    row = next(row for row in projection.issuer_rows if row.entity_id == ENTITY)
    prefix = f"open issue from {opened.raised_on.isoformat()}"
    assert row.findings_summary_parts[0] == (prefix, authored)
    assert row.findings_summary_parts[-1][1] == ""
    assert row.findings_summary == f"{prefix}: {authored}; " + row.findings_summary_parts[-1][0]
    body = evidence_cro_body(projection)
    wire_row = next(row for row in body["issuer_rows"] if row["entity_id"] == ENTITY)
    assert wire_row["findings_summary_parts"] == [list(part) for part in row.findings_summary_parts]
    assert wire_row["findings_summary"] == row.findings_summary
