from datetime import UTC, datetime
from uuid import UUID

import pytest

from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    TaskInputEnvelope,
    WorkItemDefinition,
)
from alphalattice.control.task_control.ledger import (
    REQUESTS_DIRECTORY,
    TaskAdmissionRequest,
    read_frozen,
)

TEST_SOURCE = "src/alphalattice/control/task_control/ledger.py"


@pytest.mark.proves("task_control.frozen_requests_refuse_content_address_changes")
def test_a_frozen_request_refuses_valid_content_under_another_address(tmp_path, monkeypatch):
    """A 0.2.7 request reads back with its exact text; valid content under another address is
    refused by name."""
    from alphalattice.capabilities.domain_contracts.bounded_memo import stage_digests
    from alphalattice.control.task_control import contracts

    hashed = []

    def count(value):
        hashed.append(value)
        return canonical_hash(value)

    with monkeypatch.context() as probe, stage_digests():
        probe.setattr(contracts, "canonical_hash", count)
        envelope = TaskInputEnvelope.create(
            task_kind="test.task", input_schema_id="test.input", payload={"score": -0.0}
        )
        assert TaskInputEnvelope.model_validate_json(envelope.model_dump_json()) == envelope
        assert len(hashed) == 1
        changed = envelope.model_dump() | {"payload": {"score": 0.0}}
        with pytest.raises(ValueError, match="task input envelope hash is invalid"):
            TaskInputEnvelope.model_validate(changed)
        assert len(hashed) == 2
    goal = ResearchGoal.create(
        goal_kind="TEST_GOAL",
        input_hash=envelope.input_hash,
        deliverable_kind="test.result",
        summary="test",
    )
    item = WorkItemDefinition.create(stage_id="first", dependency_ids=(), verifier_id="test.verify")
    plan = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash="a" * 64,
        verifier_catalog_hash="b" * 64,
        work_items=(item,),
    )
    request = TaskAdmissionRequest(
        task_id=UUID(int=1),
        task_kind=envelope.task_kind,
        input=envelope,
        goal=goal,
        plan=plan,
        admitted_at=datetime(2026, 1, 1, tzinfo=UTC),
        admission_sequence=1,
    )
    text = request.model_dump_json()  # as a 0.2.7 workspace kept it, named by its content
    path = tmp_path / REQUESTS_DIRECTORY / f"{canonical_hash(request.model_dump(mode='json'))}.json"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")
    assert read_frozen(tmp_path) == ((request, text),)
    path.write_text(
        request.model_copy(update={"admission_sequence": 2}).model_dump_json(), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"task_control\.admission_request_unreadable"):
        read_frozen(tmp_path)
