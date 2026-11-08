from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest

from alphalattice.control.product_host.composition.evidence_review_application import (
    EvidenceReviewApplication,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    AlternativeEvidenceDocumentTaskAdapter,
)

TEST_SOURCE = "src/alphalattice/control/product_host/composition/evidence_review_application.py"


@pytest.mark.proves("composition.unstarted_evidence_task_reports_its_requested_cutoff")
def test_an_evidence_task_without_a_run_keeps_its_request_cutoff():
    cutoff = datetime(2026, 1, 5, tzinfo=UTC)
    request = seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        ordered_entity_ids=("AAPL",),
        evidence_as_of=cutoff,
        acquisition_deadline=cutoff + timedelta(minutes=15),
        evidence_classes=(AlternativeEvidenceClass.SEC_FILING,),
        source_policy=AlternativeEvidenceSourcePolicy(),
        ttl_seconds=3600,
        mode=AlternativeEvidenceMode.RECORDED,
    )
    task = SimpleNamespace(
        task_id=UUID(int=1),
        task_kind=AlternativeEvidenceDocumentTaskAdapter.task_kind,
        input=SimpleNamespace(payload={"request": request.model_dump(mode="json")}),
    )
    registry = SimpleNamespace(admission_subjects=lambda _: ())
    owner = SimpleNamespace(
        session=SimpleNamespace(task_control_registry=registry), evidence_task_adapter=None
    )
    answer = EvidenceReviewApplication.task_subject(owner, task)
    assert answer["subject_context"]["evidence_as_of"] == cutoff.isoformat()
