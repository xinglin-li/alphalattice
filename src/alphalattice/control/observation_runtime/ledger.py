"""Bounded SQLite-WAL Unified Observation Ledger for one workspace."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.control.workspace_runtime.storage.capacity import StorageCapStore
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import (
    ObservationAppendResult,
    ObservationAvailability,
    ObservationDraft,
    ObservationEnvelope,
    ObservationProjectionCheckpoint,
    ObservationReadCursor,
    ObservationReadItem,
    ObservationReadPage,
    ObservationRetentionClass,
    ObservationRetentionPolicy,
    ObservationRetentionReport,
    ObservationRunSummary,
    build_retention_policy,
    observation_identity,
)
from .policy import ObservationPolicyRegistry

_Result = TypeVar("_Result")

_BUSY_TIMEOUT_SECONDS = 5.0
_BUSY_TIMEOUT_MILLISECONDS = 5_000
_SCHEMA_VERSION = 1
_STORE_EPOCH_KEY = "store_epoch"
MAXIMUM_READ_PAGE = 500
"""The largest bounded read one call may ask for; the store never scans past it."""

_ROW_COLUMNS = (
    "source_kind, source_id, source_sequence, schema_kind, schema_version, "
    "run_id, task_id, stage_id, payload_hash, policy_hash, availability, "
    "envelope_json, observation_id"
)


class ObservationStorageError(RuntimeError):
    """Bounded, user-safe observation-store failure."""

    def __init__(self, failure_code: str, detail: str) -> None:
        """Keep a stable failure code and bounded user-safe detail."""
        safe_detail = " ".join(str(detail).split())[:1000] or type(self).__name__
        super().__init__(safe_detail)
        self.failure_code = failure_code
        self.safe_detail = safe_detail


class _ObservationAppendCapability:
    __slots__ = ()


_OBSERVATION_APPEND_CAPABILITY = _ObservationAppendCapability()


class UnifiedObservationPort:
    """The only public append boundary: validate/redact, then persist."""

    def __init__(
        self,
        *,
        ledger: ObservationLedger,
        policies: ObservationPolicyRegistry,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Bind the append boundary to a ledger, policies and observation clock."""
        self._ledger = ledger
        self._policies = policies
        self._clock = clock

    @property
    def policies(self) -> ObservationPolicyRegistry:
        """Return the installed schema policies used by this append port."""
        return self._policies

    def emit(self, draft: ObservationDraft) -> ObservationAppendResult:
        """Redact and admit a draft before the ledger persists it.

        Args:
            draft: Proposed public-safe observation.

        Returns:
            The append or exact-reuse receipt.

        """
        safe = self._policies.validate_and_redact(draft)
        return self._ledger._append_validated(
            safe,
            observed_at=self._clock(),
            capability=_OBSERVATION_APPEND_CAPABILITY,
        )

    def observations_for_run(self, run_id: str) -> tuple[ObservationEnvelope, ...]:
        """Expose bounded readback without leaking the Ledger write owner."""
        return self._ledger.observations_for_run(run_id)


class ObservationLedger:
    """Record safe assertions without replacing their source authorities."""

    def __init__(
        self,
        database_path: Path,
        *,
        gate: WorkspaceMutationGate,
        retention_policy: ObservationRetentionPolicy | None = None,
        storage_cap_reader: Callable[[], int] | None = None,
    ) -> None:
        """Open one workspace ledger under its mutation and retention rules.

        Args:
            database_path: SQLite WAL store path.
            gate: Authority for serialized store mutations.
            retention_policy: Optional workspace-specific storage cap.
            storage_cap_reader: Live workspace owner cap; standalone stores use their directory.

        Raises:
            ObservationStorageError: The store cannot be opened or verified.

        """
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.gate = gate
        self._storage_cap_reader = storage_cap_reader
        if self._storage_cap_reader is None and retention_policy is None:
            self._storage_cap_reader = lambda: (
                StorageCapStore(self.database_path.parent)
                .capacity(measured_data_bytes=self.physical_store_bytes())
                .cap_bytes
            )
        try:
            if retention_policy is not None:
                self._retention_policy = retention_policy
            else:
                assert self._storage_cap_reader is not None
                self._retention_policy = build_retention_policy(
                    workspace_managed_cap_bytes=self._storage_cap_reader()
                )
        except (OSError, ValueError, RuntimeError) as error:
            raise ObservationStorageError(
                "observation.storage_open_failed",
                "The workspace storage setting could not be read.",
            ) from error
        self._protected_run_ids: tuple[str, ...] = ()
        self._completed_run_ids: tuple[str, ...] = ()
        self._store_epoch = ""
        self._closed = False
        try:
            self._writer = sqlite3.connect(
                self.database_path,
                timeout=_BUSY_TIMEOUT_SECONDS,
                isolation_level=None,
                check_same_thread=False,
            )
            self._bootstrap()
        except Exception as exc:
            writer = getattr(self, "_writer", None)
            if writer is not None:
                writer.close()
            if isinstance(exc, ObservationStorageError):
                raise
            raise _storage_error(exc, operation="open", database_path=self.database_path) from exc

    def __enter__(self) -> ObservationLedger:
        """Return the open ledger for a context-managed lifetime."""
        self._require_open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        """Close the ledger when its context ends."""
        self.close()

    def close(self) -> None:
        """Close the writer once; later reads and writes are refused."""
        if self._closed:
            return
        self._closed = True
        self._writer.close()

    def _require_open(self) -> None:
        if self._closed:
            raise ObservationStorageError(
                "observation.storage_open_failed", "observation store is closed"
            )

    def _bootstrap(self) -> None:
        def operation() -> None:
            is_new = not _has_user_tables(self._writer)
            if is_new:
                self._writer.execute("PRAGMA auto_vacuum=INCREMENTAL")
                self._writer.execute("VACUUM")
            journal_mode = str(self._writer.execute("PRAGMA journal_mode=WAL").fetchone()[0])
            if journal_mode.lower() != "wal":
                self._raise_storage(
                    "observation.storage_open_failed", "observation WAL mode was not admitted"
                )
            self._writer.execute("PRAGMA synchronous=NORMAL")
            self._writer.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MILLISECONDS}")
            self._writer.execute("PRAGMA wal_autocheckpoint=1000")
            journal_bytes = max(1, self._retention_policy.ledger_cap_bytes // 100)
            self._writer.execute(f"PRAGMA journal_size_limit={journal_bytes}")
            self._writer.execute(
                """
                CREATE TABLE IF NOT EXISTS unified_observation (
                    observation_id TEXT PRIMARY KEY,
                    source_kind TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_sequence INTEGER NOT NULL,
                    schema_kind TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    run_id TEXT,
                    task_id TEXT,
                    stage_id TEXT,
                    occurred_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    authority_kind TEXT NOT NULL,
                    sensitivity_class TEXT NOT NULL,
                    retention_class TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    inline_safe_payload_json TEXT,
                    payload_ref_json TEXT,
                    policy_hash TEXT NOT NULL,
                    supersedes_observation_id TEXT,
                    availability TEXT NOT NULL,
                    envelope_json TEXT NOT NULL,
                    UNIQUE(source_kind, source_id, source_sequence)
                )
                """
            )
            self._writer.execute(
                """
                CREATE TABLE IF NOT EXISTS observation_projection_checkpoint (
                    projection_kind TEXT NOT NULL,
                    projection_key TEXT NOT NULL,
                    through_observation_id TEXT NOT NULL,
                    checkpoint_hash TEXT NOT NULL,
                    checkpoint_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (projection_kind, projection_key)
                )
                """
            )
            # Refusals, counted: the one durable trace of where requests fail, kept when
            # the transient rows that carried them are evicted (AC, OP14). Codes and kinds
            # only: no session, no request, no text.
            self._writer.execute(
                """
                CREATE TABLE IF NOT EXISTS refusal_count (
                    day TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    failure_code TEXT NOT NULL,
                    caller TEXT NOT NULL,
                    vendor TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    first_at TEXT NOT NULL,
                    last_at TEXT NOT NULL,
                    PRIMARY KEY (day, operation, failure_code, caller, vendor)
                )
                """
            )
            self._writer.execute(
                """
                CREATE TABLE IF NOT EXISTS observation_store_metadata (
                    metadata_key TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            self._writer.execute(
                """
                CREATE INDEX IF NOT EXISTS unified_observation_run_order_idx
                ON unified_observation(
                    run_id, observed_at, source_kind, source_id,
                    source_sequence, observation_id
                )
                """
            )
            self._writer.execute(
                """
                CREATE INDEX IF NOT EXISTS unified_observation_retention_idx
                ON unified_observation(
                    retention_class, availability, run_id,
                    source_kind, source_id, schema_kind, source_sequence
                )
                """
            )
            # A reader of one kind (the declared sessions) walks that kind in commit
            # order without scanning the other kinds between its rows.
            self._writer.execute(
                """
                CREATE INDEX IF NOT EXISTS unified_observation_kind_order_idx
                ON unified_observation(schema_kind)
                """
            )
            self._writer.execute(
                """
                INSERT INTO observation_store_metadata VALUES ('schema', ?)
                ON CONFLICT(metadata_key) DO NOTHING
                """,
                [json.dumps({"schema_version": _SCHEMA_VERSION}, sort_keys=True)],
            )
            # The epoch names this physical store for bounded readers. Written
            # once; a rebuilt or migrated store gets its own, so a cursor from
            # the old store is reset instead of continued against new ordinals.
            self._writer.execute(
                """
                INSERT INTO observation_store_metadata VALUES (?, ?)
                ON CONFLICT(metadata_key) DO NOTHING
                """,
                [
                    _STORE_EPOCH_KEY,
                    json.dumps({"epoch": secrets.token_hex(16)}, sort_keys=True),
                ],
            )

        self.gate.run(operation)
        epoch = self.store_metadata(_STORE_EPOCH_KEY)
        if epoch is None or not isinstance(epoch.get("epoch"), str):
            self._raise_storage("observation.storage_integrity_failed", "store epoch is invalid")
        assert epoch is not None
        self._store_epoch = str(epoch["epoch"])

    @property
    def store_epoch(self) -> str:
        """The identity bounded read cursors are scoped to; stable across restarts."""
        return self._store_epoch

    def head_ordinal(self) -> int:
        """Commit ordinal of the newest row, 0 for an empty store."""

        def operation(connection: Any) -> int:
            row = connection.execute("SELECT COALESCE(MAX(rowid), 0) FROM unified_observation")
            return int(row.fetchone()[0])

        return self._read(operation)

    def observations_after(
        self, cursor: ObservationReadCursor | None, *, limit: int
    ) -> ObservationReadPage:
        """Read at most `limit` rows in commit order, bounded by the cursor.

        No cursor reads the tail: the newest `limit` rows, so a fresh reader
        starts from current activity rather than the beginning of the store. A
        cursor from another epoch, or past this store's head, cannot be honoured
        and is answered with `RESET` plus a fresh tail. Retention never deletes
        rows, so a reader that was away while payloads were evicted still sees
        each row, with its availability, in its place.
        """
        if not 1 <= limit <= MAXIMUM_READ_PAGE:
            raise ValueError("observation.read_limit_invalid")

        def operation(connection: Any) -> ObservationReadPage:
            head = int(
                connection.execute(
                    "SELECT COALESCE(MAX(rowid), 0) FROM unified_observation"
                ).fetchone()[0]
            )
            if cursor is None or cursor.store_epoch != self._store_epoch or cursor.ordinal > head:
                disposition: Literal["TAIL", "RESET"] = "TAIL" if cursor is None else "RESET"
                rows = connection.execute(
                    f"""
                    SELECT rowid, {_ROW_COLUMNS} FROM unified_observation
                    ORDER BY rowid DESC LIMIT ?
                    """,
                    [limit + 1],
                ).fetchall()
                more = len(rows) > limit
                selected = list(reversed(rows[:limit]))
                return ObservationReadPage(
                    disposition=disposition,
                    store_epoch=self._store_epoch,
                    head_ordinal=head,
                    items=tuple(self._read_item(row) for row in selected),
                    more=more,
                )
            rows = connection.execute(
                f"""
                SELECT rowid, {_ROW_COLUMNS} FROM unified_observation
                WHERE rowid > ? ORDER BY rowid LIMIT ?
                """,
                [cursor.ordinal, limit + 1],
            ).fetchall()
            return ObservationReadPage(
                disposition="CONTINUED",
                store_epoch=self._store_epoch,
                head_ordinal=head,
                items=tuple(self._read_item(row) for row in rows[:limit]),
                more=len(rows) > limit,
            )

        return self._read(operation)

    def observations_of_kind(
        self, schema_kind: str, *, limit: int, before: int | None = None
    ) -> ObservationReadPage:
        """Read the newest rows of one schema kind in commit order.

        The tail of a kind, not of the store: a reader of the declared sessions
        is not bounded by how many product operations were recorded between two
        team events. `before` (an ordinal) reads the rows older than it, so the
        reader can walk further back one page at a time. Retention keeps every
        row; an evicted payload shows as unavailable in its place.
        """
        if not 1 <= limit <= MAXIMUM_READ_PAGE:
            raise ValueError("observation.read_limit_invalid")
        if not schema_kind or (before is not None and before < 1):
            raise ValueError("observation.read_cursor_invalid")

        def operation(connection: Any) -> ObservationReadPage:
            head = int(
                connection.execute(
                    "SELECT COALESCE(MAX(rowid), 0) FROM unified_observation"
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT rowid, {_ROW_COLUMNS} FROM unified_observation
                WHERE schema_kind = ? AND rowid < ? ORDER BY rowid DESC LIMIT ?
                """,
                [schema_kind, head + 1 if before is None else before, limit + 1],
            ).fetchall()
            return ObservationReadPage(
                disposition="TAIL",
                store_epoch=self._store_epoch,
                head_ordinal=head,
                items=tuple(self._read_item(row) for row in reversed(rows[:limit])),
                more=len(rows) > limit,
            )

        return self._read(operation)

    @staticmethod
    def _read_item(row: Any) -> ObservationReadItem:
        return ObservationReadItem(
            ordinal=int(row[0]), envelope=ObservationLedger._validated_envelope(row[1:])
        )

    def _append_validated(
        self,
        draft: ObservationDraft,
        *,
        observed_at: datetime,
        capability: _ObservationAppendCapability,
    ) -> ObservationAppendResult:
        if capability is not _OBSERVATION_APPEND_CAPABILITY:
            raise ValueError("observation.payload_not_admitted")
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observation clock must be timezone-aware")
        if draft.inline_safe_payload is not None:
            payload = draft.inline_safe_payload
        else:
            if draft.payload_ref is None:
                raise ValueError("observation.payload_not_admitted")
            payload = draft.payload_ref.model_dump(mode="json")
        payload_hash = canonical_hash(payload)
        provisional = ObservationEnvelope.model_construct(
            **draft.model_dump(mode="python", exclude={"kind"}),
            observed_at=observed_at.astimezone(UTC),
            payload_hash=payload_hash,
            availability=ObservationAvailability.AVAILABLE,
            observation_id="",
        )
        envelope = ObservationEnvelope(
            **provisional.model_dump(mode="python", exclude={"observation_id"}),
            observation_id=observation_identity(provisional),
        )

        def operation(connection: Any) -> ObservationAppendResult:
            existing_row = connection.execute(
                """
                SELECT observation_id, envelope_json FROM unified_observation
                WHERE source_kind = ? AND source_id = ? AND source_sequence = ?
                """,
                [envelope.source_kind, envelope.source_id, envelope.source_sequence],
            ).fetchone()
            if existing_row is not None:
                existing = ObservationEnvelope.model_validate_json(str(existing_row[1]))
                if existing.observation_id != envelope.observation_id:
                    raise ValueError("observation.source_sequence_collision")
                return _append_result("REUSED_EXACT", existing)
            if envelope.supersedes_observation_id is not None:
                prior = self._load_row(connection, envelope.supersedes_observation_id)
                if prior is None:
                    raise ValueError("observation.required_decision_missing")
                if (
                    prior.source_kind != envelope.source_kind
                    or prior.source_id != envelope.source_id
                    or prior.source_sequence >= envelope.source_sequence
                ):
                    raise ValueError("observation.source_sequence_collision")
            connection.execute(
                """
                INSERT INTO unified_observation VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    envelope.observation_id,
                    envelope.source_kind,
                    envelope.source_id,
                    envelope.source_sequence,
                    envelope.schema_kind,
                    envelope.schema_version,
                    envelope.run_id,
                    envelope.task_id,
                    envelope.stage_id,
                    _db_time(envelope.occurred_at),
                    _db_time(envelope.observed_at),
                    envelope.authority.value,
                    envelope.sensitivity.value,
                    envelope.retention_class.value,
                    envelope.payload_hash,
                    _json(envelope.inline_safe_payload),
                    _json(envelope.payload_ref),
                    envelope.policy_hash,
                    envelope.supersedes_observation_id,
                    envelope.availability.value,
                    envelope.model_dump_json(),
                ],
            )
            return _append_result("APPENDED", envelope)

        self._enforce_retention_on_append()
        result = self._write(operation)
        self._enforce_retention_on_append()
        return result

    def set_retention_roots(
        self, *, protected_run_ids: Sequence[str], completed_run_ids: Sequence[str]
    ) -> None:
        """Provide the current owner-resolved roots used by append-time retention."""
        self._protected_run_ids = tuple(sorted(set(protected_run_ids)))
        self._completed_run_ids = tuple(sorted(set(completed_run_ids)))

    def read(self, observation_id: str) -> ObservationEnvelope:
        """Read one verified envelope by its logical observation ID.

        Raises:
            ValueError: The observation is absent or tampered.

        """

        def operation(connection: Any) -> ObservationEnvelope:
            value = self._load_row(connection, observation_id)
            if value is None:
                raise ValueError("observation.tampered")
            return value

        return self._read(operation)

    def availability(self, observation_id: str) -> ObservationAvailability:
        """Return the stored availability or a missing-or-tampered verdict."""
        try:
            return self.read(observation_id).availability
        except (ValueError, json.JSONDecodeError):
            return ObservationAvailability.MISSING_OR_TAMPERED

    def observations_for_run(self, run_id: str) -> tuple[ObservationEnvelope, ...]:
        """Read a run's observations in stable occurrence and source order."""

        def operation(connection: Any) -> tuple[ObservationEnvelope, ...]:
            rows = connection.execute(
                """
                SELECT observation_id FROM unified_observation
                WHERE run_id = ?
                ORDER BY observed_at, source_kind, source_id, source_sequence, observation_id
                """,
                [run_id],
            ).fetchall()
            return tuple(self._require_row(connection, str(row[0])) for row in rows)

        return self._read(operation)

    def all_observations(self) -> tuple[ObservationEnvelope, ...]:
        """Read every observation in stable occurrence and source order."""

        def operation(connection: Any) -> tuple[ObservationEnvelope, ...]:
            rows = connection.execute(
                """
                SELECT observation_id FROM unified_observation
                ORDER BY observed_at, source_kind, source_id, source_sequence, observation_id
                """
            ).fetchall()
            return tuple(self._require_row(connection, str(row[0])) for row in rows)

        return self._read(operation)

    def save_projection_checkpoint(
        self, checkpoint: ObservationProjectionCheckpoint, *, observed_at: datetime
    ) -> None:
        """Persist a verified projection checkpoint without moving it backward.

        Args:
            checkpoint: Projection position and source cursors to save.
            observed_at: Aware timestamp for this checkpoint write.

        Raises:
            ValueError: The clock, source row or progression is invalid.

        """
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observation checkpoint clock must be timezone-aware")
        self.read(checkpoint.through_observation_id)

        def operation(connection: Any) -> None:
            existing = connection.execute(
                """
                SELECT checkpoint_json FROM observation_projection_checkpoint
                WHERE projection_kind = ? AND projection_key = ?
                """,
                [checkpoint.projection_kind, checkpoint.projection_key],
            ).fetchone()
            if existing is not None:
                current = ObservationProjectionCheckpoint.model_validate_json(str(existing[0]))
                if current.checkpoint_hash == checkpoint.checkpoint_hash:
                    return
                if checkpoint.observation_count < current.observation_count:
                    raise ValueError("observation.projection_checkpoint_mismatch")
            connection.execute(
                """
                INSERT INTO observation_projection_checkpoint VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (projection_kind, projection_key) DO UPDATE SET
                    through_observation_id = excluded.through_observation_id,
                    checkpoint_hash = excluded.checkpoint_hash,
                    checkpoint_json = excluded.checkpoint_json,
                    updated_at = excluded.updated_at
                """,
                [
                    checkpoint.projection_kind,
                    checkpoint.projection_key,
                    checkpoint.through_observation_id,
                    checkpoint.checkpoint_hash,
                    checkpoint.model_dump_json(),
                    _db_time(observed_at),
                ],
            )

        self._write(operation)

    def load_projection_checkpoint(
        self, projection_kind: str, projection_key: str
    ) -> ObservationProjectionCheckpoint | None:
        """Load a checkpoint only if its source observation still verifies."""

        def operation(connection: Any) -> ObservationProjectionCheckpoint | None:
            row = connection.execute(
                """
                SELECT checkpoint_json FROM observation_projection_checkpoint
                WHERE projection_kind = ? AND projection_key = ?
                """,
                [projection_kind, projection_key],
            ).fetchone()
            if row is None:
                return None
            value = cast(
                ObservationProjectionCheckpoint,
                ObservationProjectionCheckpoint.model_validate_json(str(row[0])),
            )
            self._require_row(connection, value.through_observation_id)
            return value

        return self._read(operation)

    def store_metadata(self, key: str) -> dict[str, Any] | None:
        """Read verified store metadata by key, if it exists."""

        def operation(connection: Any) -> dict[str, Any] | None:
            row = connection.execute(
                "SELECT metadata_json FROM observation_store_metadata WHERE metadata_key = ?",
                [key],
            ).fetchone()
            if row is None:
                return None
            value = json.loads(str(row[0]))
            if not isinstance(value, dict):
                self._raise_storage(
                    "observation.storage_integrity_failed", "store metadata is invalid"
                )
            return cast(dict[str, Any], value)

        return self._read(operation)

    def _save_store_metadata(self, key: str, value: dict[str, Any]) -> None:
        if not key:
            raise ValueError("observation metadata key is required")

        def operation(connection: Any) -> None:
            connection.execute(
                """
                INSERT INTO observation_store_metadata VALUES (?, ?)
                ON CONFLICT(metadata_key) DO UPDATE SET metadata_json = excluded.metadata_json
                """,
                [key, json.dumps(value, sort_keys=True, separators=(",", ":"))],
            )

        self._write(operation)

    def integrity_check(self) -> None:
        """Refuse a store whose SQLite quick check does not pass."""

        def operation(connection: Any) -> None:
            row = connection.execute("PRAGMA quick_check").fetchone()
            if row is None or str(row[0]).lower() != "ok":
                self._raise_storage(
                    "observation.storage_integrity_failed", "SQLite quick_check failed"
                )

        self._read(operation)

    def physical_store_bytes(self) -> int:
        """Sum bytes in the database and its live WAL companions."""
        return sum(
            path.stat().st_size for path in _store_paths(self.database_path) if path.is_file()
        )

    def _prepare_for_promotion(self) -> None:
        def gated() -> None:
            self._require_open()
            try:
                result = self._writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if result is None or int(result[0]) != 0:
                    self._raise_storage(
                        "observation.storage_busy", "WAL promotion checkpoint was busy"
                    )
            except ObservationStorageError:
                raise
            except sqlite3.Error as exc:
                raise _storage_error(
                    exc,
                    operation="promotion checkpoint",
                    database_path=self.database_path,
                ) from exc

        self.gate.run(gated)

    def count_refusal(
        self,
        *,
        operation: str,
        failure_code: str,
        caller: str,
        vendor: str | None,
        refused_at: datetime,
    ) -> None:
        """Count one refusal under its day, operation, code, caller kind and agent vendor.

        Args:
            operation: The operation refused, or `UNKNOWN` when the request named none.
            failure_code: The typed code the refusal carried.
            caller: The caller kind the entry boundary supplied.
            vendor: The agent vendor the request's session named, if any.
            refused_at: When the refusal was answered, timezone-aware.
        """
        moment = refused_at.astimezone(UTC).isoformat()
        day = moment[:10]

        def operation_(connection: Any) -> None:
            connection.execute(
                """
                INSERT INTO refusal_count VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT (day, operation, failure_code, caller, vendor)
                DO UPDATE SET count = count + 1, last_at = excluded.last_at
                """,
                [day, operation, failure_code, caller, vendor or "", moment, moment],
            )

        self._write(operation_)

    def refusal_counts(self, *, since_day: str) -> tuple[dict[str, object], ...]:
        """Read the refusals counted on or after a day, the most frequent first.

        Args:
            since_day: The first day read, as `YYYY-MM-DD`.

        Returns:
            One row a day, operation, code, caller kind and vendor, with its count and its first
            and last moments.
        """

        def operation_(connection: Any) -> tuple[dict[str, object], ...]:
            rows = connection.execute(
                """
                SELECT day, operation, failure_code, caller, vendor, count, first_at, last_at
                FROM refusal_count WHERE day >= ?
                ORDER BY count DESC, day DESC, operation, failure_code, caller, vendor
                LIMIT ?
                """,
                [since_day, MAXIMUM_READ_PAGE],
            ).fetchall()
            return tuple(
                {
                    "day": row[0],
                    "operation": row[1],
                    "failure_code": row[2],
                    "caller": row[3],
                    "vendor": row[4] or None,
                    "count": int(row[5]),
                    "first_at": row[6],
                    "last_at": row[7],
                }
                for row in rows
            )

        return self._read(operation_)

    def approximate_bytes(self) -> int:
        """Estimate the ledger's logical retained payload bytes."""
        return self._read(self._approximate_bytes)

    def apply_retention(
        self,
        *,
        policy: ObservationRetentionPolicy,
        protected_run_ids: Sequence[str],
        completed_run_ids: Sequence[str],
    ) -> ObservationRetentionReport:
        """Compact or evict eligible payloads while protecting owner roots.

        Args:
            policy: Installed cap, watermarks and transient ring size.
            protected_run_ids: Runs whose rows must remain available.
            completed_run_ids: Finished runs eligible for compaction.

        Returns:
            A content-hashed report of the resulting storage changes.

        Raises:
            ValueError: Protected roots alone exceed the ledger cap.

        """
        protected = tuple(sorted(set(protected_run_ids)))
        completed = tuple(sorted(set(completed_run_ids)))

        physical_before = self.physical_store_bytes()

        def operation(
            connection: Any,
        ) -> tuple[int, int, int, int, tuple[ObservationRunSummary, ...]]:
            before = self._approximate_bytes(connection)
            if max(before, physical_before) <= policy.high_water_bytes:
                return before, before, 0, 0, ()

            protected_placeholders = ",".join("?" for _ in protected) or "NULL"
            protected_bytes = int(
                connection.execute(
                    f"""
                    SELECT COALESCE(SUM(length(envelope_json)), 0)
                    FROM unified_observation
                    WHERE run_id IN ({protected_placeholders})
                    """,
                    list(protected),
                ).fetchone()[0]
            )
            if protected_bytes > policy.ledger_cap_bytes:
                raise ValueError("observation.retention_root_over_budget")

            evicted = 0
            transient_groups = connection.execute(
                """
                SELECT run_id, source_kind, source_id, schema_kind
                FROM unified_observation
                WHERE retention_class = ? AND availability = ?
                GROUP BY 1, 2, 3, 4
                """,
                [
                    ObservationRetentionClass.TRANSIENT_OPERATIONAL.value,
                    ObservationAvailability.AVAILABLE.value,
                ],
            ).fetchall()
            for run_id, source_kind, source_id, schema_kind in transient_groups:
                ids = connection.execute(
                    """
                    SELECT observation_id FROM unified_observation
                    WHERE run_id IS NOT DISTINCT FROM ? AND source_kind = ?
                      AND source_id = ? AND schema_kind = ?
                      AND retention_class = ? AND availability = ?
                    ORDER BY source_sequence DESC, observed_at DESC
                    LIMIT -1 OFFSET ?
                    """,
                    [
                        run_id,
                        source_kind,
                        source_id,
                        schema_kind,
                        ObservationRetentionClass.TRANSIENT_OPERATIONAL.value,
                        ObservationAvailability.AVAILABLE.value,
                        policy.transient_ring_size,
                    ],
                ).fetchall()
                for row in ids:
                    self._set_unavailable(
                        connection,
                        str(row[0]),
                        ObservationAvailability.EVICTED_BY_RETENTION,
                    )
                    evicted += 1

            compacted = 0
            summaries: list[ObservationRunSummary] = []
            for run_id in completed:
                if run_id in protected:
                    continue
                rows = connection.execute(
                    """
                    SELECT observation_id FROM unified_observation
                    WHERE run_id = ? AND retention_class = ? AND availability = ?
                    ORDER BY source_kind, source_id, source_sequence
                    """,
                    [
                        run_id,
                        ObservationRetentionClass.RUN_OPERATIONAL.value,
                        ObservationAvailability.AVAILABLE.value,
                    ],
                ).fetchall()
                values = [self._require_row(connection, str(row[0])) for row in rows]
                if not values:
                    continue
                summaries.append(_summarize_run(run_id, values))
                for value in values:
                    self._set_unavailable(
                        connection, value.observation_id, ObservationAvailability.COMPACTED
                    )
                    compacted += 1

            after = self._approximate_bytes(connection)
            if after > policy.cleanup_target_bytes:
                raise ValueError("observation.retention_root_over_budget")
            return before, after, compacted, evicted, tuple(summaries)

        before, after, compacted, evicted, summaries = self._write(operation)
        self._maintain_physical_store()
        physical_after = self.physical_store_bytes()
        if physical_after > policy.ledger_cap_bytes:
            raise ValueError("observation.retention_root_over_budget")
        return _retention_report(
            policy,
            before,
            after,
            physical_before,
            physical_after,
            compacted,
            evicted,
            protected,
            summaries,
        )

    def _enforce_retention_on_append(self) -> None:
        if self._storage_cap_reader is not None:

            def refresh() -> None:
                assert self._storage_cap_reader is not None
                self._retention_policy = build_retention_policy(
                    workspace_managed_cap_bytes=self._storage_cap_reader()
                )
                journal_bytes = max(1, self._retention_policy.ledger_cap_bytes // 100)
                self._writer.execute(f"PRAGMA journal_size_limit={journal_bytes}")

            self.gate.run(refresh)
        if max(self.approximate_bytes(), self.physical_store_bytes()) <= (
            self._retention_policy.high_water_bytes
        ):
            return
        self.apply_retention(
            policy=self._retention_policy,
            protected_run_ids=self._protected_run_ids,
            completed_run_ids=self._completed_run_ids,
        )

    @staticmethod
    def _load_row(connection: Any, observation_id: str) -> ObservationEnvelope | None:
        row = connection.execute(
            f"SELECT {_ROW_COLUMNS} FROM unified_observation WHERE observation_id = ?",
            [observation_id],
        ).fetchone()
        if row is None:
            return None
        return ObservationLedger._validated_envelope(row)

    @staticmethod
    def _validated_envelope(row: Any) -> ObservationEnvelope:
        """One row in `_ROW_COLUMNS` order, checked against its own envelope.

        The indexed columns are a projection of the envelope; a row whose
        columns disagree with its JSON, or whose stored id is not the envelope's
        own identity, has been altered outside the append path.
        """
        value = cast(
            ObservationEnvelope,
            ObservationEnvelope.model_validate_json(str(row[11])),
        )
        columns = (
            value.source_kind,
            value.source_id,
            value.source_sequence,
            value.schema_kind,
            value.schema_version,
            value.run_id,
            value.task_id,
            value.stage_id,
            value.payload_hash,
            value.policy_hash,
            value.availability.value,
        )
        if tuple(row[:11]) != columns or value.observation_id != str(row[12]):
            raise ValueError("observation.tampered")
        return value

    def _require_row(self, connection: Any, observation_id: str) -> ObservationEnvelope:
        value = self._load_row(connection, observation_id)
        if value is None:
            raise ValueError("observation.tampered")
        return value

    @staticmethod
    def _approximate_bytes(connection: Any) -> int:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(
                length(envelope_json)
              + COALESCE(length(inline_safe_payload_json), 0)
              + COALESCE(length(payload_ref_json), 0)
            ), 0)
            FROM unified_observation
            """
        ).fetchone()
        return int(row[0])

    def _set_unavailable(
        self,
        connection: Any,
        observation_id: str,
        availability: ObservationAvailability,
    ) -> None:
        current = self._require_row(connection, observation_id)
        if current.payload_ref is not None:
            payload_ref = current.payload_ref.model_copy(update={"availability": availability})
            payload_ref = payload_ref.model_copy(
                update={
                    "ref_hash": canonical_hash(
                        payload_ref.model_dump(mode="json", exclude={"ref_hash"})
                    )
                }
            )
        else:
            payload_ref = None
        updated = current.model_copy(
            update={
                "inline_safe_payload": None,
                "payload_ref": payload_ref,
                "availability": availability,
            }
        )
        connection.execute(
            """
            UPDATE unified_observation SET
                inline_safe_payload_json = NULL,
                payload_ref_json = ?,
                availability = ?,
                envelope_json = ?
            WHERE observation_id = ?
            """,
            [_json(payload_ref), availability.value, updated.model_dump_json(), observation_id],
        )

    def _write(self, operation: Callable[[Any], _Result]) -> _Result:
        def gated() -> _Result:
            self._require_open()
            try:
                self._writer.execute("BEGIN IMMEDIATE")
                result = operation(self._writer)
                self._writer.execute("COMMIT")
                return result
            except Exception as exc:
                if self._writer.in_transaction:
                    self._writer.execute("ROLLBACK")
                if isinstance(exc, (ValueError, ObservationStorageError)):
                    raise
                if isinstance(exc, sqlite3.Error):
                    raise _storage_error(
                        exc, operation="write", database_path=self.database_path
                    ) from exc
                raise

        return cast(_Result, self.gate.run(gated))

    def _read(self, operation: Callable[[Any], _Result]) -> _Result:
        self._require_open()
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                _read_only_uri(self.database_path),
                timeout=_BUSY_TIMEOUT_SECONDS,
                uri=True,
            )
            connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MILLISECONDS}")
            return operation(connection)
        except (ValueError, ObservationStorageError):
            raise
        except sqlite3.Error as exc:
            raise _storage_error(exc, operation="read", database_path=self.database_path) from exc
        finally:
            if connection is not None:
                connection.close()

    def _maintain_physical_store(self) -> None:
        def gated() -> None:
            self._require_open()
            try:
                # PASSIVE never waits for readers. Reusable pages are reclaimed
                # incrementally without turning retention into a global pause.
                self._writer.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                self._writer.execute("PRAGMA incremental_vacuum")
            except sqlite3.Error as exc:
                raise _storage_error(
                    exc, operation="maintenance", database_path=self.database_path
                ) from exc

        self.gate.run(gated)

    def _raise_storage(self, failure_code: str, detail: str) -> None:
        error = ObservationStorageError(failure_code, detail)
        _write_storage_failure(self.database_path, error)
        raise error


def _append_result(
    disposition: Literal["APPENDED", "REUSED_EXACT"], envelope: ObservationEnvelope
) -> ObservationAppendResult:
    provisional = ObservationAppendResult.model_construct(
        disposition=disposition,
        observation_id=envelope.observation_id,
        source_sequence=envelope.source_sequence,
        availability=envelope.availability,
        result_hash="",
    )
    identity = provisional.model_dump(mode="json", exclude={"result_hash"})
    return ObservationAppendResult(**identity, result_hash=canonical_hash(identity))


def _summarize_run(
    run_id: str, observations: Iterable[ObservationEnvelope]
) -> ObservationRunSummary:
    values = tuple(observations)
    ranges: dict[str, list[int]] = defaultdict(list)
    decisions: list[str] = []
    for item in values:
        ranges[f"{item.source_kind}:{item.source_id}"].append(item.source_sequence)
        if item.retention_class is ObservationRetentionClass.DURABLE_DECISION:
            decisions.append(item.observation_id)
    identity = {
        "run_id": run_id,
        "compacted_observation_count": len(values),
        "retained_decision_ids": tuple(sorted(decisions)),
        "source_ranges": {key: (min(items), max(items)) for key, items in sorted(ranges.items())},
        "compacted_payload_digest": canonical_hash(
            tuple((item.observation_id, item.payload_hash) for item in values)
        ),
    }
    provisional = ObservationRunSummary.model_construct(**identity, summary_hash="")
    canonical_identity = provisional.model_dump(mode="json", exclude={"summary_hash"})
    return ObservationRunSummary(
        **canonical_identity,
        summary_hash=canonical_hash(canonical_identity),
    )


def _retention_report(
    policy: ObservationRetentionPolicy,
    before: int,
    after: int,
    physical_before: int,
    physical_after: int,
    compacted: int,
    evicted: int,
    protected: tuple[str, ...],
    summaries: tuple[ObservationRunSummary, ...],
) -> ObservationRetentionReport:
    values: dict[str, Any] = {
        "policy_hash": policy.policy_hash,
        "bytes_before": before,
        "bytes_after": after,
        "physical_bytes_before": physical_before,
        "physical_bytes_after": physical_after,
        "compacted_count": compacted,
        "evicted_count": evicted,
        "protected_run_ids": protected,
        "run_summaries": summaries,
    }
    provisional = ObservationRetentionReport.model_construct(**values, report_hash="")
    identity = provisional.model_dump(mode="json", exclude={"report_hash"})
    return ObservationRetentionReport(**identity, report_hash=canonical_hash(identity))


def _json(value: object | None) -> str | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _db_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _has_user_tables(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        """
        SELECT count(*) FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        """
    ).fetchone()
    return bool(row and int(row[0]) > 0)


def _read_only_uri(path: Path) -> str:
    return f"{path.as_uri()}?mode=ro"


def _store_paths(path: Path) -> tuple[Path, Path, Path]:
    return (path, Path(f"{path}-wal"), Path(f"{path}-shm"))


def _storage_error(
    exc: BaseException, *, operation: str, database_path: Path
) -> ObservationStorageError:
    detail = " ".join(str(exc).split())[:1000]
    lowered = detail.lower()
    code = (
        "observation.storage_busy"
        if "locked" in lowered or "busy" in lowered
        else "observation.storage_open_failed"
    )
    error = ObservationStorageError(code, f"{operation}: {detail or type(exc).__name__}")
    _write_storage_failure(database_path, error)
    return error


def _write_storage_failure(database_path: Path, error: ObservationStorageError) -> None:
    try:
        root = database_path.parent / "observation-storage-failures"
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "failure_code": error.failure_code,
            "failure_type": type(error).__name__,
            "user_safe_detail": error.safe_detail,
        }
        identity = canonical_hash(payload)
        target = root / f"{identity}.json"
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if target.is_file():
            return
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_bytes(encoded)
        temporary.replace(target)
    except Exception:
        return


__all__ = [
    "MAXIMUM_READ_PAGE",
    "ObservationLedger",
    "ObservationStorageError",
    "UnifiedObservationPort",
]
