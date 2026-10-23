"""A book's review resumed by a retry (idempotency keys), never prepared again beside itself.

The Evidence review application's: its issuer scope binds the sealed book, so a retry names the
same attempt. While a current analysis answers a unit of the newest preparation, the preview keeps
that cutoff and offers an Analyst only an unanswered unit: no second analysis for the agent to pick.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from alphalattice.evidence.alternative_evidence.contracts import AlternativeEvidenceRequest
from alphalattice.evidence.alternative_evidence.runtime.reuse import analysis_records

if TYPE_CHECKING:
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
    clock: Callable[[], datetime]

    def _publication_days(self) -> RecordDays | None: ...

    def _sealed_run(
        self, scope: PortfolioIssuerScope, *, evidence_as_of: datetime | None
    ) -> AlternativeEvidenceCoverageRun | None: ...


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


def unanswered(
    requests: dict[str, dict[str, str]], answered: Collection[str]
) -> dict[str, dict[str, str]]:
    """These offers less each unit's packet (`packet_<unit>`) a current analysis answers."""
    return {
        name: value
        for name, value in requests.items()
        if not (name.startswith("packet_") and name.removeprefix("packet_") in answered)
    }
