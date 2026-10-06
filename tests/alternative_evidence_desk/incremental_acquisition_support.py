"""Fixtures and helpers of the incremental-acquisition scenarios: a small
official-shaped inventory for two issuers, live requests under the source
policy's default or an explicit cap, and the readers of a runtime's retained
objects, commits and per-resource outcomes. Test support beside its owner;
nothing here is product authority or evidence."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceAdmission,
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.runtime.service import (
    AlternativeEvidenceDocumentIntelligenceRuntime,
)
from alphalattice.evidence.alternative_evidence.sources.contracts import (
    SOURCE_COMMIT_CATEGORY,
    SOURCE_OBJECT_ROOT,
)
from alphalattice.evidence.alternative_evidence.sources.sec_edgar import SecEdgarSource
from tests.alternative_evidence_desk.document_intelligence_support import INTEGRATED_SELECTION
from tests.alternative_evidence_desk.sec_scenario_transport import (
    NOW,
    ScenarioFiling,
    SecScenarioTransport,
    filing_body,
)

AAPL = "0000320193"
MSFT = "0000789019"
REGISTRY = {"AAPL": (AAPL, "Apple Inc."), "MSFT": (MSFT, "Microsoft")}

TEN_K = ScenarioFiling(
    "0000320193-25-000100",
    "10-K",
    "2026-07-20",
    "2026-07-20T21:00:00.000Z",
    "aapl-10k.htm",
    "2025-09-27",
)
TEN_Q = ScenarioFiling(
    "0000320193-26-000050",
    "10-Q",
    "2026-08-01",
    "2026-08-01T20:00:00.000Z",
    "aapl-10q.htm",
    "2026-03-28",
)
EIGHT_K = ScenarioFiling(
    "0000320193-26-000070",
    "8-K",
    "2026-07-31",
    "2026-07-31T20:05:00.000Z",
    "aapl-8k.htm",
    "2026-07-31",
)
LATER_EIGHT_K = ScenarioFiling(
    "0000320193-26-000081",
    "8-K",
    "2026-08-13",
    "2026-08-13T20:10:00.000Z",
    "aapl-8k-2.htm",
    "2026-08-13",
)
MSFT_TEN_K = ScenarioFiling(
    "0000789019-25-000090",
    "10-K",
    "2026-07-25",
    "2026-07-25T21:00:00.000Z",
    "msft-10k.htm",
    "2025-06-30",
)
MSFT_TEN_Q = ScenarioFiling(
    "0000789019-26-000040",
    "10-Q",
    "2026-08-05",
    "2026-08-05T21:00:00.000Z",
    "msft-10q.htm",
    "2026-03-31",
)


def _transport(*, msft: bool = False) -> SecScenarioTransport:
    filings = {AAPL: [EIGHT_K, TEN_Q, TEN_K]}
    bodies = {
        SecScenarioTransport.locator(AAPL, f): filing_body(f.accession) for f in filings[AAPL]
    }
    if msft:
        filings[MSFT] = [MSFT_TEN_Q, MSFT_TEN_K]
        for filing in filings[MSFT]:
            bodies[SecScenarioTransport.locator(MSFT, filing)] = filing_body(filing.accession)
    return SecScenarioTransport(registry=REGISTRY, filings=filings, bodies=bodies)


def _request(
    entities: tuple[str, ...] = ("AAPL",), *, as_of=NOW, budget: int = 3, cap: int | None = None
) -> AlternativeEvidenceRequest:
    """A live request under the integrated selection; `cap` None takes the
    source policy's own default."""

    return seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        ordered_entity_ids=entities,
        evidence_as_of=as_of,
        acquisition_deadline=as_of + timedelta(hours=1),
        evidence_classes=(AlternativeEvidenceClass.SEC_FILING,),
        source_policy=AlternativeEvidenceSourcePolicy(
            maximum_documents_per_issuer=budget,
            **({} if cap is None else {"maximum_document_bytes": cap}),
        ),
        ttl_seconds=86_400,
        mode=AlternativeEvidenceMode.LIVE_OFFICIAL,
        matter_selection=INTEGRATED_SELECTION,
    )


def _admission(request: AlternativeEvidenceRequest) -> AlternativeEvidenceAdmission:
    return seal_contract(
        AlternativeEvidenceAdmission,
        "admission_hash",
        request_hash=request.request_hash,
        network_consent=True,
        admit_live_official=True,
        admit_model_review=False,
        admitted_at=request.evidence_as_of,
    )


def _acquire(
    runtime: AlternativeEvidenceDocumentIntelligenceRuntime,
    transport: SecScenarioTransport,
    request: AlternativeEvidenceRequest,
    *,
    registry=None,
    should_cancel=None,
):
    at = request.evidence_as_of
    return runtime.acquire_live(
        request=request,
        admission=_admission(request),
        source=SecEdgarSource(transport),
        registry=registry,
        published_at=at,
        clock=lambda: at,
        should_cancel=should_cancel,
    )


def _source_objects(runtime: AlternativeEvidenceDocumentIntelligenceRuntime) -> dict[str, int]:
    root = runtime.documents.workspace.root / Path(*SOURCE_OBJECT_ROOT.split("/"))
    return (
        {p.name: p.stat().st_size for p in root.iterdir() if p.is_file()} if root.is_dir() else {}
    )


def _commits(runtime: AlternativeEvidenceDocumentIntelligenceRuntime) -> int:
    directory = runtime.artifacts.root / SOURCE_COMMIT_CATEGORY
    return len(list(directory.glob("*.json"))) if directory.is_dir() else 0


def _outcomes(snapshot) -> dict[str, str]:
    assert snapshot.acquisition is not None
    return {value.accession: value.outcome for value in snapshot.acquisition.documents}


__all__ = [
    "AAPL",
    "EIGHT_K",
    "LATER_EIGHT_K",
    "MSFT",
    "MSFT_TEN_K",
    "MSFT_TEN_Q",
    "REGISTRY",
    "TEN_K",
    "TEN_Q",
    "_acquire",
    "_admission",
    "_commits",
    "_outcomes",
    "_request",
    "_source_objects",
    "_transport",
]
