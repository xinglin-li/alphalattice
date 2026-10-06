"""A controlled fifty-issuer book for the full-portfolio evidence route.

Fifty synthetic issuers, each with two recorded official documents stating one
planted claim, mapped one-to-one onto a fifty-name synthetic book. What this
proves is scheduling, isolation, accounting and reuse at fifty-name scale; it
proves nothing about financial recall or real-world latency, and no filing is
copied under another issuer's name. Test support beside its owner; nothing
here is product authority or evidence.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from alphalattice.control.product_host.composition.local_web_session import (
    EvidenceReviewAuthority,
)
from alphalattice.evidence.alternative_evidence.analysis.contracts import EvidenceTopic
from alphalattice.evidence.alternative_evidence.contracts import (
    SecIssuerRegistryEntry,
    SecIssuerRegistrySnapshot,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.runtime.policy import AdmittedEvidencePolicy
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    AlternativeEvidenceDocumentTaskResources,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioResearchSpec,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    AdmittedListingTicker,
    AdmittedListingTickerAuthority,
    seal_portfolio_evidence_contract,
)
from alphalattice.protocols.actor_execution import ActorKind
from tests.alternative_evidence_desk.planted_corpus import (
    _NOW,
    PLANTED_CLAIMS,
    _CitingActor,
    _recorded_document,
    _runtime,
)
from tests.alternative_evidence_desk.review_dossiers import CitingReviewActor, ControlledRisk
from tests.portfolio_strategy_lab.local_web_support import (
    _harness,
    _resolved,
    _Resolver,
    _run,
)

COVERAGE_BOOK_SIZE = 50
COVERAGE_UNIVERSE = 120
"""A universe wide enough for a fifty-name book's exit rank (twice its top-k)."""

COVERAGE_ENTITIES: tuple[str, ...] = tuple(f"QA{index:02d}" for index in range(1, 51))
"""Fifty synthetic issuers. Synthetic tickers, synthetic CIKs, no real company."""

_TOPICS: tuple[EvidenceTopic, ...] = tuple(EvidenceTopic)
COVERAGE_TOPICS: dict[str, EvidenceTopic] = {
    entity: _TOPICS[index % len(_TOPICS)] for index, entity in enumerate(COVERAGE_ENTITIES)
}


def coverage_registry(entities: tuple[str, ...] = COVERAGE_ENTITIES) -> SecIssuerRegistrySnapshot:
    return seal_contract(
        SecIssuerRegistrySnapshot,
        "registry_hash",
        captured_at=_NOW - timedelta(days=1),
        entries=tuple(
            SecIssuerRegistryEntry(
                entity_id=entity,
                ticker=entity,
                cik=f"{9_100_000_000 + index:010d}",
                legal_name=f"{entity} Synthetic Holdings Inc.",
            )
            for index, entity in enumerate(entities, start=1)
        ),
        source_content_hash="2" * 64,
    )


def coverage_documents(
    entities: tuple[str, ...] = COVERAGE_ENTITIES,
    *,
    revisions: tuple[str, ...] = ("release", "transcript"),
    suffix: str = "",
    revised: frozenset[str] = frozenset(),
    quiet: frozenset[str] = frozenset(),
) -> tuple[RecordedEvidenceDocument, ...]:
    """Two official documents per issuer stating that issuer's planted claim.

    `revised` names issuers whose documents carry an amended revision label and
    an added sentence: the two-issuer delta workload, where only those issuers'
    documents change while every other issuer's bytes stay byte-identical.
    `quiet` names issuers whose documents state no planted claim at all: real
    material that yields no finding, as opposed to no material.
    """

    documents = []
    for entity in entities:
        topic = COVERAGE_TOPICS[entity]
        for revision in revisions:
            amended = entity in revised
            claim = (
                PLANTED_CLAIMS[topic]
                + suffix
                + (" The amended filing restates the prior figure." if amended else "")
            )
            base = _recorded_document(
                entity,
                claims=() if entity in quiet else (claim,),
                revision=f"{entity.casefold()}-2026-q3-{revision}"
                + ("-amended" if amended else ""),
                # Current reports, the form the integrated selection reads;
                # the revision names which is the release and which the
                # transcript.
                document_type="8-K",
            )
            # Every issuer's text is its own: the background names the issuer
            # and the document on every line, so no two issuers share an
            # encoder input and a chunk count is a count of real work. The
            # shared-text shortcut the planted corpus takes would otherwise
            # let fifty issuers cost sixteen documents of embedding.
            documents.append(
                base.model_copy(
                    update={
                        "text": base.text.replace(
                            "Background section",
                            f"{entity} {revision} background section",
                        )
                    }
                )
            )
    return tuple(documents)


def coverage_listing_authority(
    listing_ids: tuple[str, ...],
    *,
    entities: tuple[str, ...] = COVERAGE_ENTITIES,
    omit: frozenset[str] = frozenset(),
    shared: dict[str, str] | None = None,
) -> AdmittedListingTickerAuthority:
    """Map the book's listings onto the synthetic issuers, one each, in listing order.

    `omit` leaves listings without an admitted symbol (the mapping-failure
    workload); `shared` maps a listing onto another listing's issuer so one
    issuer carries two positions (the shared-issuer case).
    """

    ordered = tuple(sorted(listing_ids))
    by_listing = {listing: entities[index % len(entities)] for index, listing in enumerate(ordered)}
    for listing, other in (shared or {}).items():
        by_listing[listing] = by_listing[other]
    return seal_portfolio_evidence_contract(
        AdmittedListingTickerAuthority,
        "authority_hash",
        entries=tuple(
            AdmittedListingTicker(listing_id=listing, ticker=by_listing[listing])
            for listing in ordered
            if listing not in omit
        ),
    )


def build_coverage_workspace(
    tmp_path: Path,
    *,
    top_k: int = COVERAGE_BOOK_SIZE,
    universe: int = COVERAGE_UNIVERSE,
    tranches: int = 3,
    exit_rank: int | None = None,
    sessions: tuple[Any, ...] | None = None,
    score_rotation: int = 0,
) -> Any:
    """One real Portfolio path over a wider universe, published, its lease released.

    `universe` widens the listing universe with the book (a wider book needs twice its
    top-k for its exit rank; the scale scenes pass both). `tranches`, `exit_rank`,
    `sessions` and `score_rotation` give the book its real width: `top_k` is names per
    sleeve, and with a rotating cross-section the book at the window end is the union
    of the live sleeves (the scale scene: 6 sleeves of 70 under an exit rank of 420).
    """

    portfolio = tmp_path / "portfolio"
    portfolio.mkdir(parents=True, exist_ok=True)
    with _harness(portfolio) as harness:
        harness.application.resolver = _Resolver(
            _resolved(
                top_k=top_k,
                listing_count=universe,
                tranches=tranches,
                exit_rank=exit_rank,
                sessions=sessions,
                score_rotation=score_rotation,
            ),
            score_rotation=score_rotation,
        )
        result = _run(
            harness,
            PortfolioResearchSpec.create(top_k=top_k, tranches=tranches, exit_rank=exit_rank),
        )
        report = harness.application.report(result.result_hash)
        return harness.workspace, report


def coverage_authority(
    workspace: Path,
    report: Any,
    *,
    entities: tuple[str, ...] = COVERAGE_ENTITIES,
    omit: frozenset[str] = frozenset(),
    shared: dict[str, str] | None = None,
    documents: tuple[RecordedEvidenceDocument, ...] | None = None,
    model_authority_admitted: bool = True,
    minimum_entity_coverage: float = 0.0,
    runtime: Any | None = None,
    analysis_actor: Any | None = None,
) -> EvidenceReviewAuthority:
    """The production shape over the controlled book: the store under the runtime.

    `runtime` lets a driver hand in a counted runtime over the same roots; by
    default the runtime lives under the workspace's `runtime/` like the
    materialized one does.
    """

    from alphalattice.evidence.alternative_evidence.runtime.service import (
        AlternativeEvidenceDocumentIntelligenceRuntime,
    )
    from tests.alternative_evidence_desk.planted_corpus import _index_factory, _retriever_factory

    listings = tuple(value.listing_id for value in report.window_end_book.positions)
    if runtime is None:
        runtime = AlternativeEvidenceDocumentIntelligenceRuntime(
            artifact_root=workspace / "runtime" / "artifacts",
            workspace_root=workspace / "runtime" / "evidence-knowledge",
            model_root=workspace / "runtime" / "models",
            index_factory=_index_factory,
            retriever_factory=_retriever_factory,
        )
    listing_authority = coverage_listing_authority(
        listings, entities=entities, omit=omit, shared=shared
    )
    mapped = tuple(sorted({value.ticker for value in listing_authority.entries}))
    registry = coverage_registry(entities)
    first = mapped[0] if mapped else entities[0]
    return EvidenceReviewAuthority(
        model_authority_admitted=model_authority_admitted,
        registry=registry,
        listing_authority=listing_authority,
        artifacts=runtime.artifacts,
        evidence_publications=runtime.publications,
        evidence_runtime=runtime,
        evidence_resources=AlternativeEvidenceDocumentTaskResources(
            recorded_registry=registry,
            recorded_documents=coverage_documents(mapped) if documents is None else documents,
            analysis_actor=analysis_actor or _CitingActor(topics=COVERAGE_TOPICS),
            minimum_entity_coverage=minimum_entity_coverage,
        ),
        evidence_policy=AdmittedEvidencePolicy(),
        review_actor=CitingReviewActor(
            risks=(ControlledRisk(first),),
            actor_kind=ActorKind.HUMAN,
            actor_id="qa-portfolio-coverage",
        ),
    )


class ClaimCitingActor(_CitingActor):
    """A controlled analyst that cites only spans stating the issuer's planted
    claim -- so an issuer whose documents state nothing yields no finding --
    and leaves one note per answer, as a real analyst may."""

    def __call__(self, *, packet: Any) -> Any:
        from alphalattice.evidence.alternative_evidence.analysis.contracts import (
            AlternativeEvidenceAnalystAnswer,
        )
        from tests.alternative_evidence_desk.planted_corpus import packet_aliases

        answered = super().__call__(packet=packet)
        alias_of = packet_aliases(packet)
        kept = []
        for finding in answered.answer.findings:
            claim = PLANTED_CLAIMS[COVERAGE_TOPICS[finding.issuer]]
            cited = tuple(
                alias_of[span.span_handle]
                for span in packet.spans
                if span.entity_id == finding.issuer and claim in " ".join(span.excerpt.split())
            )
            if cited:
                kept.append(finding.model_copy(update={"cite": cited[:8]}))
        revised = AlternativeEvidenceAnalystAnswer(
            findings=tuple(kept),
            notes="Whether the stated matters recur next quarter is not answered by these sources.",
        )
        return replace(answered, answer=revised)


UNIT_COUNT = -(-COVERAGE_BOOK_SIZE // 8)
"""Fifty issuers in units of eight: seven units, the last of two."""


def run_one(service: Any, request: dict[str, Any], *, caller: str = "HUMAN") -> Any:
    """The CLI's path: the same operation owner the buttons reach, in process."""

    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    assert service.session.operations is not None
    return service.session.operations.execute(
        PortfolioResearchOperationRequest(**request), caller=caller
    )


def http_body(request: dict[str, Any]) -> dict[str, Any]:
    """The same request as an HTTP body: the route names the operation, the body does not."""

    return {key: value for key, value in request.items() if key != "operation"}


def answer_unit(service: Any, packet_request: dict[str, Any]) -> dict[str, Any]:
    """Export one unit's packet, answer it with the authority's controlled
    analyst, submit the answer through the native submission path."""

    from uuid import UUID

    exported = run_one(service, packet_request)
    assert exported["status"] == "EVIDENCE_ANALYST_PACKET_READY", exported
    adapter = service.review.evidence_task_adapter
    packet = adapter.prepared_packet(
        UUID(packet_request["task_id"]),
        now=service.review.clock(),
        unit_id=packet_request.get("evidence_unit_id"),
    )
    actor = adapter.resources.analysis_actor or _CitingActor(topics=COVERAGE_TOPICS)
    answer = actor(packet=packet).answer.model_dump(mode="json")
    submitted = run_one(service, {**exported["submission_template"], "analysis_answer": answer})
    assert submitted["disposition"] in {"ADMITTED", "REUSED_EXACT"}, submitted
    return dict(submitted)


def plain_runtime(tmp_path: Path) -> Any:
    """The runtime alone, for the reference driver and owner-level probes."""

    return cast(Any, _runtime(tmp_path))


def start_coverage_service(
    workspace: Path,
    authority: EvidenceReviewAuthority | None,
    tmp_path: Path,
    *,
    clock: Any = None,
    top_k: int = COVERAGE_BOOK_SIZE,
    universe: int = COVERAGE_UNIVERSE,
    tranches: int = 3,
    exit_rank: int | None = None,
    score_rotation: int = 0,
) -> Any:
    """The real local service over the controlled book: bound socket, session
    token, the one dispatcher -- the shape every button and CLI request takes."""

    from alphalattice.control.product_host.composition.local_web_session import (
        LocalPortfolioWebSession,
    )
    from tests.alternative_evidence_desk.review_http_support import (
        _Service,
        _workspace_manifest,
    )

    session = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_workspace_manifest("qa-portfolio-coverage"),
        resolver=_Resolver(
            _resolved(
                top_k=top_k,
                listing_count=universe,
                tranches=tranches,
                exit_rank=exit_rank,
                score_rotation=score_rotation,
            ),
            score_rotation=score_rotation,
        ),
        clock=clock or (lambda: _NOW),
        review_authority=authority,
    )
    session.start()
    return _Service(session, tmp_path)


__all__ = [
    "COVERAGE_BOOK_SIZE",
    "COVERAGE_ENTITIES",
    "COVERAGE_TOPICS",
    "COVERAGE_UNIVERSE",
    "UNIT_COUNT",
    "ClaimCitingActor",
    "answer_unit",
    "build_coverage_workspace",
    "coverage_authority",
    "coverage_documents",
    "coverage_listing_authority",
    "coverage_registry",
    "http_body",
    "plain_runtime",
    "run_one",
    "start_coverage_service",
]
