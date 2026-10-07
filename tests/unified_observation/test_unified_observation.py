from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic

import pytest

from alphalattice.control.observation_runtime.contracts import (
    ObservationAuthority,
    ObservationAuthorityReadback,
    ObservationAuthorityReference,
    ObservationAvailability,
    ObservationDraft,
    ObservationRetentionClass,
    ObservationRetentionPolicy,
    ObservationSensitivity,
    build_observation_authority_readback,
    build_observation_authority_reference,
    build_retention_policy,
)
from alphalattice.control.observation_runtime.ledger import (
    ObservationLedger,
    ObservationStorageError,
    UnifiedObservationPort,
)
from alphalattice.control.observation_runtime.policy import default_observation_policies
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.control.workspace_runtime.storage.capacity import StorageCapStore
from alphalattice.kernel.shared_kernel.identity import canonical_hash

NOW = datetime(2026, 8, 8, 12, tzinfo=UTC)


class _FixtureAuthorityStore:
    owner_kind = "ALPHA_SPARSE_LOOP_ARTIFACT_STORE"

    def __init__(self) -> None:
        self._records: set[tuple[str, str]] = set()

    def register(
        self,
        record_kind: str,
        payload: object,
        *,
        task_id: str | None = "task-1",
        run_id: str | None = "run-1",
        subject_id: str | None = "decision-1",
    ) -> ObservationAuthorityReference:
        record_hash = canonical_hash(payload)
        self._records.add((record_kind, record_hash))
        return build_observation_authority_reference(
            owner_kind=self.owner_kind,
            record_kind=record_kind,
            record_hash=record_hash,
            task_id=task_id,
            run_id=run_id,
            subject_id=subject_id,
        )

    def read_observation_authority(
        self, reference: ObservationAuthorityReference
    ) -> ObservationAuthorityReadback:
        if (
            reference.owner_kind != self.owner_kind
            or (reference.record_kind, reference.record_hash) not in self._records
        ):
            raise FileNotFoundError(reference.record_hash)
        return build_observation_authority_readback(reference)

    def remove(self, reference: ObservationAuthorityReference) -> None:
        self._records.remove((reference.record_kind, reference.record_hash))


def _runtime(tmp_path: Path) -> tuple[ObservationLedger, UnifiedObservationPort]:
    gate = WorkspaceMutationGate()
    authority_store = _FixtureAuthorityStore()
    ledger = ObservationLedger(tmp_path / "observations.sqlite", gate=gate)
    return ledger, UnifiedObservationPort(
        ledger=ledger,
        policies=default_observation_policies(authority_reader=authority_store),
        clock=lambda: NOW,
    )


def _draft(
    *,
    sequence: int,
    status: str = "RUNNING",
    run_id: str = "run-1",
    retention: ObservationRetentionClass = ObservationRetentionClass.RUN_OPERATIONAL,
    supersedes: str | None = None,
) -> ObservationDraft:
    policies = default_observation_policies()
    policy = policies.policy("TaskControlTransition", 1)
    return ObservationDraft(
        schema_kind="TaskControlTransition",
        schema_version=1,
        occurred_at=NOW + timedelta(seconds=sequence),
        source_kind="TASK_CONTROL",
        source_id="task-1",
        source_sequence=sequence,
        task_id="task-1",
        run_id=run_id,
        stage_id="alpha",
        correlation_ids=("foundation-1", "task-1"),
        authority=ObservationAuthority.TASK_CONTROL_ASSERTION,
        sensitivity=ObservationSensitivity.PUBLIC_SAFE,
        retention_class=retention,
        inline_safe_payload={"status": status, "stage_id": "alpha"},
        policy_hash=policy.policy_hash,
        supersedes_observation_id=supersedes,
    )


def test_append_is_idempotent_and_sequence_collision_fails(tmp_path: Path) -> None:
    ledger, port = _runtime(tmp_path)
    first = port.emit(_draft(sequence=1))
    replay = port.emit(_draft(sequence=1))

    assert first.disposition == "APPENDED"
    assert replay.disposition == "REUSED_EXACT"
    assert replay.observation_id == first.observation_id
    assert ledger.read(first.observation_id).inline_safe_payload == {
        "stage_id": "alpha",
        "status": "RUNNING",
    }

    with pytest.raises(ValueError, match=r"observation\.source_sequence_collision"):
        port.emit(_draft(sequence=1, status="BLOCKED"))


def test_correction_appends_and_retains_prior(tmp_path: Path) -> None:
    ledger, port = _runtime(tmp_path)
    first = port.emit(_draft(sequence=1))
    corrected = port.emit(_draft(sequence=2, status="BLOCKED", supersedes=first.observation_id))

    assert corrected.observation_id != first.observation_id
    assert ledger.read(corrected.observation_id).supersedes_observation_id == first.observation_id
    assert ledger.read(first.observation_id).availability is ObservationAvailability.AVAILABLE


def test_unique_correlation_read_uses_exact_run_schema_and_array_membership(tmp_path: Path) -> None:
    """An exact RETURNED-envelope link cannot follow a prefix, payload, or another run."""
    ledger, port = _runtime(tmp_path)
    correlation = 'answer-"quoted"-\\handle'
    exact = port.emit(_draft(sequence=1).model_copy(update={"correlation_ids": (correlation,)}))
    port.emit(
        _draft(sequence=2, run_id="other-run").model_copy(
            update={"correlation_ids": (correlation,)}
        )
    )
    port.emit(_draft(sequence=3).model_copy(update={"correlation_ids": (correlation + "-later",)}))
    port.emit(_draft(sequence=4, status=correlation).model_copy(update={"correlation_ids": ()}))
    other_schema = "TaskCommandObserved"
    other = port.emit(
        _draft(sequence=5).model_copy(
            update={
                "schema_kind": other_schema,
                "policy_hash": port.policies.policy(other_schema, 1).policy_hash,
                "correlation_ids": (correlation,),
            }
        )
    )

    assert ledger.unique_correlated_observation(
        "run-1", "TaskControlTransition", correlation
    ) == ledger.read(exact.observation_id)
    assert ledger.unique_correlated_observation("run-1", other_schema, correlation) == ledger.read(
        other.observation_id
    )
    for run_id, schema_kind, correlation_id in (
        ("missing-run", "TaskControlTransition", correlation),
        ("run-1", "OtherSchema", correlation),
        ("run-1", "TaskControlTransition", "answer-"),
        ("run-1", "TaskControlTransition", "missing-answer"),
    ):
        assert ledger.unique_correlated_observation(run_id, schema_kind, correlation_id) is None
    ledger.close()


def test_unique_correlation_read_refuses_multiple_envelopes(tmp_path: Path) -> None:
    """A reused correlation does not select the newest record or credit either candidate."""
    ledger, port = _runtime(tmp_path)
    for sequence in (1, 2, 3):
        port.emit(_draft(sequence=sequence))
    with pytest.raises(ValueError, match=r"^observation\.correlation_ambiguous$"):
        ledger.unique_correlated_observation("run-1", "TaskControlTransition", "task-1")
    ledger.close()


@pytest.mark.parametrize("selector", ["run", "schema"])
def test_unique_correlation_read_does_not_decode_outside_its_scope(
    tmp_path: Path, selector: str
) -> None:
    """An unreadable envelope outside the exact run/schema cannot poison this link."""
    ledger, port = _runtime(tmp_path)
    exact = port.emit(_draft(sequence=1))
    unrelated = port.emit(_draft(sequence=2, run_id="other-run"))
    with sqlite3.connect(ledger.database_path) as writer:
        writer.execute(
            "UPDATE unified_observation SET run_id = ?, schema_kind = ?, envelope_json = ? "
            "WHERE observation_id = ?",
            (
                "other-run" if selector == "run" else "run-1",
                "OtherSchema" if selector == "schema" else "TaskControlTransition",
                "not-json",
                unrelated.observation_id,
            ),
        )
    assert ledger.unique_correlated_observation(
        "run-1", "TaskControlTransition", "task-1"
    ) == ledger.read(exact.observation_id)
    ledger.close()


@pytest.mark.parametrize("corruption", ["indexed_column", "envelope_json"])
def test_unique_correlation_read_preserves_store_integrity_refusals(
    tmp_path: Path, corruption: str
) -> None:
    """A matching row is verified through the ledger's existing read contract."""
    ledger, port = _runtime(tmp_path)
    exact = port.emit(_draft(sequence=1))
    with sqlite3.connect(ledger.database_path) as writer:
        if corruption == "indexed_column":
            writer.execute(
                "UPDATE unified_observation SET source_id = ? WHERE observation_id = ?",
                ["altered-source", exact.observation_id],
            )
        else:
            writer.execute(
                "UPDATE unified_observation SET envelope_json = ? WHERE observation_id = ?",
                ["not-json", exact.observation_id],
            )
    if corruption == "indexed_column":
        with pytest.raises(ValueError, match=r"^observation\.tampered$"):
            ledger.unique_correlated_observation("run-1", "TaskControlTransition", "task-1")
    else:
        with pytest.raises(ObservationStorageError) as caught:
            ledger.unique_correlated_observation("run-1", "TaskControlTransition", "task-1")
        assert caught.value.failure_code == "observation.storage_open_failed"
    ledger.close()


@pytest.mark.parametrize(
    "selectors", [("", "schema", "correlation"), ("run", "", "correlation"), ("run", "schema", "")]
)
def test_unique_correlation_read_refuses_empty_selectors(
    tmp_path: Path, selectors: tuple[str, str, str]
) -> None:
    ledger, _port = _runtime(tmp_path)
    with pytest.raises(ValueError, match=r"^observation\.correlation_selector_invalid$"):
        ledger.unique_correlated_observation(*selectors)
    ledger.close()


def test_unknown_fields_and_secret_like_fields_fail_before_append(tmp_path: Path) -> None:
    ledger, port = _runtime(tmp_path)
    draft = _draft(sequence=1).model_copy(
        update={"inline_safe_payload": {"status": "RUNNING", "api_key": "not-recorded"}}
    )
    with pytest.raises(ValueError, match=r"observation\.redaction_failed"):
        port.emit(draft)
    assert ledger.all_observations() == ()

    draft = _draft(sequence=1).model_copy(
        update={"inline_safe_payload": {"status": "RUNNING", "unknown": "no"}}
    )
    with pytest.raises(ValueError, match=r"observation\.payload_not_admitted"):
        port.emit(draft)
    assert ledger.all_observations() == ()


@pytest.mark.parametrize(
    ("sensitivity", "retention"),
    (
        (ObservationSensitivity.LOCAL_SENSITIVE, ObservationRetentionClass.RUN_OPERATIONAL),
        (ObservationSensitivity.PRIVATE_RUNTIME, ObservationRetentionClass.RUN_OPERATIONAL),
        (
            ObservationSensitivity.PUBLIC_SAFE,
            ObservationRetentionClass.INTERNAL_COGNITION_ELIGIBLE,
        ),
    ),
)
def test_public_port_rejects_private_and_internal_cognition_payloads_before_append(
    tmp_path: Path,
    sensitivity: ObservationSensitivity,
    retention: ObservationRetentionClass,
) -> None:
    ledger, port = _runtime(tmp_path)
    draft = _draft(sequence=1).model_copy(
        update={"sensitivity": sensitivity, "retention_class": retention}
    )

    with pytest.raises(ValueError, match=r"observation\.payload_not_admitted"):
        port.emit(draft)

    assert ledger.all_observations() == ()


def test_public_api_does_not_expose_a_raw_ledger_append_bypass(tmp_path: Path) -> None:
    ledger, port = _runtime(tmp_path)
    unsafe = _draft(sequence=1).model_copy(
        update={
            "sensitivity": ObservationSensitivity.PRIVATE_RUNTIME,
            "retention_class": ObservationRetentionClass.INTERNAL_COGNITION_ELIGIBLE,
        }
    )

    assert not hasattr(port, "ledger")
    assert not hasattr(ledger, "append")
    with pytest.raises(ValueError, match=r"observation\.payload_not_admitted"):
        ledger._append_validated(
            unsafe,
            observed_at=NOW,
            capability=object(),  # type: ignore[arg-type]
        )

    assert ledger.all_observations() == ()


def test_durable_decision_requires_external_authority_reference(tmp_path: Path) -> None:
    ledger, port = _runtime(tmp_path)
    without_authority = _draft(sequence=1, retention=ObservationRetentionClass.DURABLE_DECISION)

    with pytest.raises(ValueError, match=r"observation\.required_decision_missing"):
        port.emit(without_authority)

    assert ledger.all_observations() == ()


def test_durable_decision_rejects_hash_shaped_but_unreadable_authority(tmp_path: Path) -> None:
    ledger, port = _runtime(tmp_path)
    unreadable = build_observation_authority_reference(
        owner_kind="ALPHA_SPARSE_LOOP_ARTIFACT_STORE",
        record_kind="DecisionTrace",
        record_hash="c" * 64,
    )
    policy = port.policies.policy("DecisionTraceObserved", 1)
    draft = ObservationDraft(
        schema_kind="DecisionTraceObserved",
        schema_version=1,
        occurred_at=NOW,
        source_kind="TASK_CONTROLLER",
        source_id="decision-1",
        source_sequence=0,
        task_id="task-1",
        run_id="run-1",
        authority=ObservationAuthority.HOST_DECISION,
        sensitivity=ObservationSensitivity.PUBLIC_SAFE,
        retention_class=ObservationRetentionClass.DURABLE_DECISION,
        inline_safe_payload={
            "decision_trace_hash": unreadable.record_hash,
            "external_authority": unreadable.model_dump(mode="json"),
            "scientific_stop_preserved": True,
            "risk_admitted": False,
        },
        policy_hash=policy.policy_hash,
    )

    with pytest.raises(ValueError, match=r"observation\.required_decision_missing"):
        port.emit(draft)
    assert ledger.all_observations() == ()


def test_retention_evicts_transient_ring_and_compacts_completed_run(tmp_path: Path) -> None:
    ledger, port = _runtime(tmp_path)
    transient_ids: list[str] = []
    for sequence in range(5):
        transient_ids.append(
            port.emit(
                _draft(
                    sequence=sequence,
                    retention=ObservationRetentionClass.TRANSIENT_OPERATIONAL,
                )
            ).observation_id
        )
    for sequence in range(5, 8):
        port.emit(_draft(sequence=sequence))

    before = ledger.approximate_bytes()
    physical_before = ledger.physical_store_bytes()
    policy = ObservationRetentionPolicy.model_construct(
        workspace_managed_cap_bytes=physical_before * 10,
        ledger_cap_bytes=physical_before * 10,
        high_water_bytes=1,
        cleanup_target_bytes=before - 1,
        transient_ring_size=2,
        policy_hash="1" * 64,
    )
    report = ledger.apply_retention(
        policy=policy,
        protected_run_ids=(),
        completed_run_ids=("run-1",),
    )

    assert report.evicted_count == 3
    assert report.compacted_count == 3
    assert report.run_summaries[0].run_id == "run-1"
    assert all(
        ledger.read(value).availability is ObservationAvailability.EVICTED_BY_RETENTION
        for value in transient_ids[:3]
    )
    assert all(
        ledger.read(value).availability is ObservationAvailability.AVAILABLE
        for value in transient_ids[3:]
    )
    assert report.bytes_after < report.bytes_before
    assert report.physical_bytes_after <= policy.ledger_cap_bytes


def test_append_lifecycle_enforces_transient_ring_and_physical_cap(tmp_path: Path) -> None:
    policy = ObservationRetentionPolicy.model_construct(
        workspace_managed_cap_bytes=10 * 1024 * 1024,
        ledger_cap_bytes=10 * 1024 * 1024,
        high_water_bytes=1,
        cleanup_target_bytes=1024 * 1024,
        transient_ring_size=2,
        policy_hash="3" * 64,
    )
    authority_store = _FixtureAuthorityStore()
    ledger = ObservationLedger(
        tmp_path / "observations.sqlite",
        gate=WorkspaceMutationGate(),
        retention_policy=policy,
    )
    port = UnifiedObservationPort(
        ledger=ledger,
        policies=default_observation_policies(authority_reader=authority_store),
        clock=lambda: NOW,
    )
    observation_ids = tuple(
        port.emit(
            _draft(
                sequence=sequence,
                retention=ObservationRetentionClass.TRANSIENT_OPERATIONAL,
            )
        ).observation_id
        for sequence in range(5)
    )

    assert all(
        ledger.read(observation_id).availability is ObservationAvailability.EVICTED_BY_RETENTION
        for observation_id in observation_ids[:3]
    )
    assert all(
        ledger.read(observation_id).availability is ObservationAvailability.AVAILABLE
        for observation_id in observation_ids[3:]
    )
    assert ledger.physical_store_bytes() <= policy.ledger_cap_bytes


def test_protected_root_over_budget_fails_without_partial_compaction(tmp_path: Path) -> None:
    ledger, port = _runtime(tmp_path)
    appended = port.emit(_draft(sequence=1))
    policy = ObservationRetentionPolicy.model_construct(
        workspace_managed_cap_bytes=1,
        ledger_cap_bytes=1,
        high_water_bytes=1,
        cleanup_target_bytes=1,
        transient_ring_size=1,
        policy_hash="2" * 64,
    )
    with pytest.raises(ValueError, match=r"observation\.retention_root_over_budget"):
        ledger.apply_retention(
            policy=policy,
            protected_run_ids=("run-1",),
            completed_run_ids=("run-1",),
        )
    assert ledger.read(appended.observation_id).availability is ObservationAvailability.AVAILABLE


def test_sqlite_wal_reader_does_not_block_single_writer(tmp_path: Path) -> None:
    ledger, port = _runtime(tmp_path)
    first = port.emit(_draft(sequence=1))
    uri = f"{ledger.database_path.as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as reader:
        reader.execute("BEGIN")
        assert reader.execute("SELECT count(*) FROM unified_observation").fetchone()[0] == 1
        second = port.emit(_draft(sequence=2, status="COMPLETED"))
        assert second.disposition == "APPENDED"
        assert reader.execute("SELECT count(*) FROM unified_observation").fetchone()[0] == 1
    assert ledger.read(first.observation_id).availability is ObservationAvailability.AVAILABLE
    assert ledger.read(second.observation_id).availability is ObservationAvailability.AVAILABLE


def test_sqlite_writer_configuration_matches_derived_projection_policy(tmp_path: Path) -> None:
    StorageCapStore(tmp_path).write(200 * 1024**2, chosen_by="HUMAN", chosen_at=NOW)
    ledger, _port = _runtime(tmp_path)
    assert ledger._writer.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert ledger._writer.execute("PRAGMA synchronous").fetchone()[0] == 1
    assert ledger._writer.execute("PRAGMA busy_timeout").fetchone()[0] == 5_000
    assert ledger._writer.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 1_000
    cap = StorageCapStore(tmp_path).capacity(measured_data_bytes=0).cap_bytes
    assert ledger._writer.execute("PRAGMA journal_size_limit").fetchone()[0] == max(
        1, build_retention_policy(workspace_managed_cap_bytes=cap).ledger_cap_bytes // 100
    )
    assert ledger._writer.execute("PRAGMA auto_vacuum").fetchone()[0] == 2


def test_an_open_observation_ledger_follows_the_workspace_cap_without_reidentifying_records(
    tmp_path: Path,
) -> None:
    """V680: settings changed outside the ledger affect the next append and reopening."""
    store = StorageCapStore(tmp_path)
    store.write(200 * 1024**2, chosen_by="HUMAN", chosen_at=NOW)
    ledger = ObservationLedger(
        tmp_path / "observations.sqlite",
        gate=WorkspaceMutationGate(),
        storage_cap_reader=lambda: store.capacity(measured_data_bytes=0).cap_bytes,
    )
    port = UnifiedObservationPort(
        ledger=ledger, policies=default_observation_policies(), clock=lambda: NOW
    )
    first = port.emit(_draft(sequence=1))
    original = ledger.read(first.observation_id)
    ledger.set_retention_roots(protected_run_ids=("run-1",), completed_run_ids=())
    store.write(1, chosen_by="HUMAN", chosen_at=NOW)
    with pytest.raises(ValueError, match=r"observation\.retention_root_over_budget"):
        port.emit(_draft(sequence=2))
    assert ledger.read(first.observation_id) == original
    assert len(ledger.all_observations()) == 1
    store.write(400 * 1024**2, chosen_by="HUMAN", chosen_at=NOW)
    replay = port.emit(_draft(sequence=1))
    assert replay.observation_id == first.observation_id and replay.disposition == "REUSED_EXACT"
    assert port.emit(_draft(sequence=2)).disposition == "APPENDED"
    ledger.close()
    reopened = ObservationLedger(
        tmp_path / "observations.sqlite",
        gate=WorkspaceMutationGate(),
        storage_cap_reader=lambda: store.capacity(measured_data_bytes=0).cap_bytes,
    )
    assert reopened.read(first.observation_id) == original
    reopened.close()


def test_second_sqlite_writer_fails_with_bounded_safe_detail(tmp_path: Path) -> None:
    ledger, port = _runtime(tmp_path)
    with sqlite3.connect(ledger.database_path, isolation_level=None) as competing_writer:
        competing_writer.execute("BEGIN IMMEDIATE")
        started = monotonic()
        with pytest.raises(ObservationStorageError) as caught:
            port.emit(_draft(sequence=1))
        elapsed = monotonic() - started
        competing_writer.execute("ROLLBACK")

    assert caught.value.failure_code == "observation.storage_busy"
    assert "locked" in caught.value.safe_detail.lower()
    assert 4.0 <= elapsed < 7.0
    assert tuple((tmp_path / "observation-storage-failures").glob("*.json"))


def test_failed_write_rolls_back_and_reuses_persistent_writer(tmp_path: Path) -> None:
    ledger, port = _runtime(tmp_path)
    writer_identity = id(ledger._writer)

    def fail_after_insert(connection: sqlite3.Connection) -> None:
        connection.execute(
            "INSERT INTO observation_store_metadata VALUES (?, ?)",
            ("must-rollback", "{}"),
        )
        raise RuntimeError("fixture failure")

    with pytest.raises(RuntimeError, match="fixture failure"):
        ledger._write(fail_after_insert)

    assert ledger.store_metadata("must-rollback") is None
    assert port.emit(_draft(sequence=1)).disposition == "APPENDED"
    assert port.emit(_draft(sequence=2)).disposition == "APPENDED"
    assert id(ledger._writer) == writer_identity


def test_bounded_read_has_tail_cursor_gap_reset_and_restart_semantics(tmp_path: Path) -> None:
    """The workspace activity feed's contract: what a reader can rely on across
    a poll, a producer replay, retention, a service restart and a rebuilt store."""

    from alphalattice.control.observation_runtime.contracts import ObservationReadCursor

    ledger, port = _runtime(tmp_path)
    assert ledger.observations_after(None, limit=10).items == ()
    assert ledger.head_ordinal() == 0
    ids = [port.emit(_draft(sequence=index)).observation_id for index in range(6)]

    # No cursor: the newest rows, in commit order, and the cursor at the head.
    tail = ledger.observations_after(None, limit=4)
    assert tail.disposition == "TAIL" and tail.more
    assert [item.envelope.observation_id for item in tail.items] == ids[2:]
    assert [item.ordinal for item in tail.items] == [3, 4, 5, 6]
    assert tail.next_cursor.ordinal == tail.head_ordinal == 6

    # A producer replay is REUSED_EXACT at the store and never a second row.
    assert port.emit(_draft(sequence=5)).disposition == "REUSED_EXACT"
    quiet = ledger.observations_after(tail.next_cursor, limit=10)
    assert quiet.disposition == "CONTINUED" and quiet.items == () and not quiet.more
    assert quiet.next_cursor == tail.next_cursor

    # The same cursor read twice returns the same rows; a bounded page reports `more`.
    cursor = ObservationReadCursor(store_epoch=ledger.store_epoch, ordinal=2)
    first = ledger.observations_after(cursor, limit=2)
    assert [item.ordinal for item in first.items] == [3, 4] and first.more
    assert first == ledger.observations_after(cursor, limit=2)
    rest = ledger.observations_after(first.next_cursor, limit=10)
    assert [item.ordinal for item in rest.items] == [5, 6] and not rest.more

    # Retention keeps every row addressable: an evicted payload is a visible gap.
    port.emit(_draft(sequence=7, retention=ObservationRetentionClass.TRANSIENT_OPERATIONAL))
    port.emit(_draft(sequence=8, retention=ObservationRetentionClass.TRANSIENT_OPERATIONAL))
    physical = ledger.physical_store_bytes()
    ledger.apply_retention(
        policy=ObservationRetentionPolicy.model_construct(
            workspace_managed_cap_bytes=physical * 10,
            ledger_cap_bytes=physical * 10,
            high_water_bytes=1,
            cleanup_target_bytes=ledger.approximate_bytes() - 1,
            transient_ring_size=1,
            policy_hash="1" * 64,
        ),
        protected_run_ids=(),
        completed_run_ids=(),
    )
    after_gap = ledger.observations_after(tail.next_cursor, limit=10)
    assert [item.ordinal for item in after_gap.items] == [7, 8]
    assert after_gap.unavailable_count == 1
    evicted = after_gap.items[0].envelope
    assert evicted.availability is ObservationAvailability.EVICTED_BY_RETENTION
    assert evicted.inline_safe_payload is None and evicted.schema_kind == "TaskControlTransition"

    # A service restart reopens the same store: same epoch, ordinals continue.
    epoch = ledger.store_epoch
    ledger.close()
    reopened = ObservationLedger(ledger.database_path, gate=WorkspaceMutationGate())
    assert reopened.store_epoch == epoch
    resumed = reopened.observations_after(after_gap.next_cursor, limit=10)
    assert resumed.disposition == "CONTINUED" and resumed.items == ()
    assert resumed.head_ordinal == 8

    # A cursor this store cannot honour is reset with a fresh tail, never continued.
    foreign = ObservationReadCursor(store_epoch="f" * 32, ordinal=3)
    reset = reopened.observations_after(foreign, limit=3)
    assert reset.disposition == "RESET" and [i.ordinal for i in reset.items] == [6, 7, 8]
    beyond = ObservationReadCursor(store_epoch=epoch, ordinal=99)
    assert reopened.observations_after(beyond, limit=3).disposition == "RESET"
    rebuilt = ObservationLedger(tmp_path / "rebuilt.sqlite", gate=WorkspaceMutationGate())
    assert rebuilt.store_epoch != epoch
    assert rebuilt.observations_after(after_gap.next_cursor, limit=3).disposition == "RESET"
    rebuilt.close()

    # Malformed cursors and out-of-bound pages are typed refusals, not scans.
    for text in ("", "abc", f"{epoch}:", f"{epoch}:-1", "not-hex:3", f"{epoch}:3:4"):
        with pytest.raises(ValueError, match=r"observation\.cursor_invalid"):
            ObservationReadCursor.parse(text)
    assert ObservationReadCursor.parse(f"{epoch}:3").encode() == f"{epoch}:3"
    for limit in (0, 501):
        with pytest.raises(ValueError, match=r"observation\.read_limit_invalid"):
            reopened.observations_after(None, limit=limit)
    reopened.close()
