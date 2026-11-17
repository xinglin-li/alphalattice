"""A workspace's records, read back from the Host's history exactly or refused by name."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from alphalattice.control.observation_runtime.contracts import (
    ObservationAuthority,
    ObservationDraft,
    ObservationRecord,
    ObservationRetentionClass,
    ObservationSensitivity,
)
from alphalattice.control.observation_runtime.log_reads import (
    OBSERVATION_ID,
    TYPE_OF,
    LogObservations,
)
from alphalattice.control.observation_runtime.policy import default_observation_policies
from alphalattice.control.product_host.composition.workspace_activity import (
    OWN,
    WorkspaceActivity,
    record_of,
    trace_record,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import history

TEST_SOURCE = "src/alphalattice/control/observation_runtime/log_reads.py"
NOW = datetime(2026, 8, 8, 12, tzinfo=UTC)


@pytest.fixture
def kept(
    tmp_path: Path, recording_history: str
) -> Iterator[tuple[WorkspaceActivity, LogObservations]]:
    """One workspace's activity writing into this process's history, and its reads."""
    activity = WorkspaceActivity(tmp_path, recording_history, WorkspaceMutationGate(), "reads")
    opened = history.opened_log()
    assert opened is not None
    try:
        yield activity, LogObservations(opened[0], recording_history)
    finally:
        activity.close()


def _keep(
    activity: WorkspaceActivity,
    sequence: int,
    *,
    run_id: str = "run-1",
    schema_kind: str = "TaskControlTransition",
    status: str = "RUNNING",
    correlation_ids: tuple[str, ...] = ("foundation-1", "task-1"),
) -> ObservationRecord:
    draft = _draft(sequence, run_id, schema_kind, correlation_ids, {"status": status})
    disposition, record = activity._keep(draft)
    assert disposition == "APPENDED"
    return record


def _draft(
    sequence: int,
    run_id: str,
    schema_kind: str,
    correlation_ids: tuple[str, ...],
    payload: dict[str, object],
) -> ObservationDraft:
    policy = default_observation_policies().policy(schema_kind, 1)
    return ObservationDraft(
        schema_kind=schema_kind,
        schema_version=1,
        occurred_at=NOW + timedelta(seconds=sequence),
        source_kind="TASK_CONTROL",
        source_id="task-1",
        source_sequence=sequence,
        task_id="task-1",
        run_id=run_id,
        stage_id="alpha",
        correlation_ids=correlation_ids,
        authority=ObservationAuthority.TASK_CONTROL_ASSERTION,
        sensitivity=ObservationSensitivity.PUBLIC_SAFE,
        retention_class=ObservationRetentionClass.RUN_OPERATIONAL,
        inline_safe_payload=payload | {"stage_id": "alpha"},
        policy_hash=policy.policy_hash,
    )


@pytest.mark.proves("observation_runtime.history_reads_answer_exactly_or_refuse")
def test_a_correlated_read_follows_its_run_schema_and_whole_handle(kept) -> None:
    """An exact link follows no prefix, payload, other run or schema; a record's id is the id of
    the event that keeps it, and read by it the record is the one kept, a public *_token field
    included; a credential in one, and a kind the history keeps no event for, are refused."""
    activity, reads = kept
    handle = 'answer-"quoted"-\\handle'
    exact = _keep(activity, 1, correlation_ids=(handle,))
    _keep(activity, 2, run_id="other-run", correlation_ids=(handle,))
    _keep(activity, 3, correlation_ids=(handle + "-later",))
    _keep(activity, 4, status=handle, correlation_ids=())
    with pytest.raises(ValueError, match=r"^observation\.payload_not_admitted$"):
        _keep(activity, 5, schema_kind="TaskCommandObserved", correlation_ids=(handle,))
    assert reads.unique_correlated_observation("run-1", "TaskControlTransition", handle) == exact
    assert reads.read(exact.observation_id) == exact
    assert OBSERVATION_ID.fullmatch(exact.observation_id)  # the log's event id, not a hash
    for selectors in (
        ("missing-run", "TaskControlTransition", handle),
        ("run-1", "TaskCommandObserved", handle),
        ("run-1", "TaskControlTransition", "answer-"),
        ("run-1", "TaskControlTransition", "missing-answer"),
    ):
        assert reads.unique_correlated_observation(*selectors) is None
    with pytest.raises(ValueError, match=r"^observation\.tampered$"):
        reads.read("0" * 64)  # no record has that id: refused as the ledger refused it
    tokens = {"case_token": "ab" * 32, "session_token": "member-session-1"}  # public, by policy
    named = _draft(6, "run-1", "TaskControlTransition", (), {"subject": tokens})
    kept_by = trace_record(
        named, NOW, {}, activity.workspace_id
    )  # the name rule reads them as keys
    assert kept_by is not None
    assert reads.read(kept_by) == record_of(named, NOW, kept_by)  # whole, through its text
    enveloped = _draft(8, "run-1", "TaskControlTransition", (), {"status": "DONE"})
    envelope = {  # as rows written before the envelope retired keep it: read past, never refused
        "kind": "ObservationEnvelope",
        "schema_kind": "TaskControlTransition",
        "schema_version": 1,
        "sensitivity": "PUBLIC_SAFE",
        "payload_hash": "a" * 64,
        "policy_hash": "b" * 64,
        "payload_ref": None,
        "availability": "AVAILABLE",
        "observation_id": "c" * 64,
        "supersedes_observation_id": None,
    }
    own = enveloped.model_dump(mode="json", include=set(OWN)) | {"observed_at": NOW.isoformat()}
    payload = dict(enveloped.inline_safe_payload or {})
    older = history.fact(
        TYPE_OF["TaskControlTransition"],
        payload | {"observation": own | envelope},
        workspace=activity.workspace_id,
    )
    assert reads.read(older) == record_of(enveloped, NOW, older)
    key = {"session_token": "sk-" + "a" * 24}  # a credential's shape, which the log refuses
    held = _draft(7, "run-1", "TaskControlTransition", (), {"subject": key})
    with pytest.raises(DataError, match="credential"):
        trace_record(held, NOW, {}, activity.workspace_id)


@pytest.mark.proves("observation_runtime.history_reads_answer_exactly_or_refuse")
@pytest.mark.parametrize(
    ("selectors", "code"),
    [
        pytest.param(("run-1", "TaskControlTransition", "task-1"), "ambiguous", id="reused"),
        pytest.param(("", "schema", "correlation"), "selector_invalid", id="no-run"),
        pytest.param(("run", "", "correlation"), "selector_invalid", id="no-schema"),
        pytest.param(("run", "schema", ""), "selector_invalid", id="no-handle"),
    ],
)
def test_a_correlated_read_refuses_two_matches_and_empty_selectors(
    kept, selectors: tuple[str, str, str], code: str
) -> None:
    """A reused handle selects neither record, and an empty selector is refused by name."""
    activity, reads = kept
    for sequence in (1, 2, 3):
        _keep(activity, sequence)
    with pytest.raises(ValueError, match=rf"^observation\.correlation_{code}$"):
        reads.unique_correlated_observation(*selectors)
