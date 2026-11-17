"""A 0.2.7 store's sealed documents, read against the log's."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from alphalattice.control.task_control.contracts import WorkItemLifecycle, WorkItemState
from alphalattice.control.task_control.upgrade import sealed_equal
from tests.workspace_task_runner.task_control_support import digest

TEST_SOURCE = "src/alphalattice/control/task_control/upgrade.py"


@pytest.mark.proves("task_control.an_older_spelling_is_the_same_sealed_state")
def test_an_older_store_spelling_of_a_work_item_is_the_same_sealed_state() -> None:
    """requirement: a work item an older store spelled without a field added since is the same
    sealed state, and one whose sealed value differs is not."""
    now = datetime(2026, 8, 5, 15, tzinfo=UTC)
    item = WorkItemState.from_identity(
        task_id=uuid4(),
        stage_id="resolve_inputs",
        definition_hash=digest("resolve_inputs"),
        lifecycle=WorkItemLifecycle.PENDING,
        attempt_count=0,
        evidence=(),
        failure_code=None,
        started_at=None,
        updated_at=now,
        version=1,
    )
    logged = item.model_dump_json()
    older = json.loads(logged)
    assert older.pop("failure_cause") is None  # the field an older store's spelling lacks
    table = "workspace_task_work_item"
    assert sealed_equal(table, logged, json.dumps(older))
    assert not sealed_equal(table, logged, json.dumps(older | {"attempt_count": 1}))  # its hash
    fields = item.model_dump(exclude={"state_hash"}) | {"attempt_count": 1, "version": 2}
    moved = WorkItemState.from_identity(**fields)
    assert not sealed_equal(table, logged, moved.model_dump_json())
