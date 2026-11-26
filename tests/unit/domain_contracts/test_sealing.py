from __future__ import annotations

from datetime import UTC, datetime
from typing import Self
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from alphalattice.capabilities.domain_contracts.bounded_memo import stage_digests
from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.capabilities.domain_contracts.sealing import (
    scoped_model_digest,
    seal_scoped_model,
)
from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord
from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from tests.workspace_task_runner.task_control_support import task_contract

TEST_SOURCE = "src/alphalattice/capabilities/domain_contracts/sealing.py"
HASHED: list[object] = []


def _digest(payload: object) -> str:
    HASHED.append(payload)
    return canonical_hash(payload)


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    values: tuple[float, ...]
    content_hash: str

    @classmethod
    def create(cls, **values: object) -> Self:
        return seal_scoped_model(cls, values, field="content_hash", digest=_digest)

    @model_validator(mode="after")
    def identity(self) -> Self:
        if self.content_hash != scoped_model_digest(self, "content_hash", digest=_digest):
            raise ValueError("record_invalid")
        return self


def _task_identity() -> dict[str, object]:
    envelope, goal, plan = task_contract(salt="sealing")
    at = datetime(2026, 9, 29, 12, tzinfo=UTC)
    return {
        "task_id": uuid4(),
        "task_kind": envelope.task_kind,
        "input": envelope,
        "goal": goal,
        "plan": plan,
        "lifecycle": TaskLifecycle.QUEUED,
        "active_work_item_id": None,
        "latest_execution_id": None,
        "admitted_at": at,
        "started_at": None,
        "updated_at": at,
        "version": 1,
    }


@pytest.mark.proves("domain_contracts.a_record_is_sealed_once")
def test_a_sealed_record_is_encoded_once_and_checked_again_when_read():
    """requirement: a sealed record is hashed once in a stage; its validator and a read back
    compare the exact payload, so a record whose constructed payload differs is refused."""
    HASHED.clear()
    with stage_digests():
        record = _Record.create(values=(1.5, -0.0))
        assert record.content_hash == canonical_hash({"values": [1.5, -0.0]})
        again = _Record.model_validate_json(record.model_dump_json())
        assert again == record and len(HASHED) == 1
        with pytest.raises(ValidationError, match="record_invalid"):
            _Record(values=(2.0,), content_hash=record.content_hash)
    # A record sealed from a mapping its model fills in (a defaulted field) is refused.
    with stage_digests(), pytest.raises(ValidationError, match="task record hash is invalid"):
        TaskRecord.from_identity(**_task_identity())
    # The registry keeps its verified record private: each reader gets its own payload.
    stored = TaskRecord.from_identity(**_task_identity(), failure_code=None)
    document = stored.model_dump_json()
    first, second = (
        DuckDbTaskControlRegistry._verified_record(str(stored.task_id), document) for _ in "ab"
    )
    assert first == second == stored and first.input.payload is not second.input.payload
