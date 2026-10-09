"""DuckDB authority for workspace maintenance cycles and audit chains."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar, cast

import duckdb

from alphalattice.control.data_platform.maintenance.contracts import (
    CANCELLED_AT_SAFE_CHECKPOINT,
    ActionAuditChainReceipt,
    ActionAuditScope,
    DataRemediationFailureReceipt,
    MaintenancePhase,
    MaintenanceStatus,
    MaintenanceTrigger,
    MarketDataChangeSet,
    WorkspaceDataUpdateReceipt,
    WorkspaceMaintenanceCycle,
    WorkspaceMaintenanceRequest,
)
from alphalattice.control.workspace_runtime.database import open_workspace_database
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.contracts import canonical_hash
from alphalattice.foundation.feature_engine.inputs.contracts import FeatureCandidateRecheck
from alphalattice.foundation.market_data_ops.sources.contracts import CandidateDataRecheck


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


_Result = TypeVar("_Result")

_TABLES = frozenset(
    {
        "workspace_data_update_receipt",
        "workspace_maintenance_cycle",
        "data_remediation_failure_receipt",
    }
)
"""The registry's own tables: present after one bootstrap of the store."""


class DuckDbWorkspaceMaintenanceRegistry:
    """One writer-gated registry in the existing workspace DuckDB."""

    def __init__(
        self, database_path: Path, *, gate: WorkspaceMutationGate, initialize: bool = True
    ) -> None:
        """Resolve the gated DuckDB maintenance registry and optionally initialize tables.

        Args:
            database_path: Workspace DuckDB path for maintenance records.
            gate: Workspace mutation owner governing registry writes.
            initialize: Whether to bootstrap maintenance tables during construction.
        """
        self.database_path = database_path.resolve()
        self.gate = gate
        if initialize:
            self._bootstrap()

    def _bootstrap(self) -> None:
        # A read first. Every data-update readback constructs this registry,
        # and a writable open with three DDL statements per readback forced
        # the store's instance writable under every reader (and refused the
        # session's read boundary); the tables are created only when the
        # store lacks one -- a workspace's first composition -- or the store
        # does not exist yet, which the writable open creates.
        if self.database_path.is_file():

            def present(connection: duckdb.DuckDBPyConnection) -> bool:
                names = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'main'"
                    ).fetchall()
                }
                return names >= _TABLES

            if self._read(present):
                return

        def operation() -> None:
            connection = open_workspace_database(self.database_path, read_only=False)
            try:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS workspace_data_update_receipt "
                    "(plan_hash VARCHAR PRIMARY KEY, receipt_hash VARCHAR NOT NULL, "
                    "receipt_json VARCHAR NOT NULL, published_at TIMESTAMP NOT NULL)"
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workspace_maintenance_cycle (
                        cycle_id VARCHAR PRIMARY KEY,
                        request_hash VARCHAR NOT NULL UNIQUE,
                        request_json VARCHAR NOT NULL,
                        phase VARCHAR NOT NULL,
                        status VARCHAR NOT NULL,
                        change_set_json VARCHAR,
                        child_task_refs_json VARCHAR NOT NULL,
                        effect_receipts_json VARCHAR NOT NULL,
                        retry_after_at TIMESTAMP,
                        failure_code VARCHAR,
                        transport_workers INTEGER NOT NULL DEFAULT 4,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    ALTER TABLE workspace_maintenance_cycle
                    ADD COLUMN IF NOT EXISTS transport_workers INTEGER DEFAULT 4
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workspace_maintenance_event (
                        event_id VARCHAR PRIMARY KEY,
                        cycle_id VARCHAR NOT NULL,
                        sequence BIGINT NOT NULL,
                        kind VARCHAR NOT NULL,
                        details_json VARCHAR NOT NULL,
                        recorded_at TIMESTAMP NOT NULL,
                        UNIQUE(cycle_id, sequence)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS action_audit_chain_receipt (
                        receipt_hash VARCHAR PRIMARY KEY,
                        listing_id VARCHAR NOT NULL,
                        provider VARCHAR NOT NULL,
                        audit_scope VARCHAR NOT NULL,
                        previous_receipt_hash VARCHAR,
                        full_anchor_receipt_hash VARCHAR NOT NULL,
                        covered_through_session DATE NOT NULL,
                        receipt_json VARCHAR NOT NULL,
                        observed_at TIMESTAMP NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS data_remediation_failure_receipt (
                        receipt_hash VARCHAR PRIMARY KEY,
                        maintenance_id VARCHAR NOT NULL,
                        case_token VARCHAR,
                        failure_code VARCHAR NOT NULL,
                        receipt_json VARCHAR NOT NULL,
                        observed_at TIMESTAMP NOT NULL
                    )
                    """
                )
            finally:
                connection.close()

        self.gate.run(operation)

    def _write(self, operation: Callable[[duckdb.DuckDBPyConnection], _Result]) -> _Result:
        def gated() -> _Result:
            connection = open_workspace_database(self.database_path, read_only=False)
            try:
                connection.execute("BEGIN TRANSACTION")
                result = operation(connection)
                connection.execute("COMMIT")
                return result
            except Exception:
                connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

        with self.gate.hold():
            return gated()

    def _read(self, operation: Callable[[duckdb.DuckDBPyConnection], _Result]) -> _Result:
        def gated() -> _Result:
            connection = open_workspace_database(self.database_path, read_only=True)
            try:
                return operation(connection)
            finally:
                connection.close()

        with self.gate.hold():
            return gated()

    def admit(
        self, request: WorkspaceMaintenanceRequest, *, observed_at: datetime
    ) -> WorkspaceMaintenanceCycle:
        """Create or reuse the request-derived durable maintenance cycle.

        Args:
            request: Sealed request whose scope this operation executes or projects.
            observed_at: Operational observation clock; callers supply an aware instant.

        Returns:
            The admitted cycle with recorded request and initial/current state.

        Raises:
            ValueError: Admission time is naive or the request identity is already bound to another
                cycle.
        """
        if observed_at.tzinfo is None:
            raise ValueError("maintenance admission time must be timezone-aware")
        now = observed_at.astimezone(UTC).replace(tzinfo=None)
        cycle_id: str = canonical_hash(["workspace-maintenance", request.request_hash])

        def operation(connection: duckdb.DuckDBPyConnection) -> str:
            row = connection.execute(
                "SELECT cycle_id FROM workspace_maintenance_cycle WHERE request_hash = ?",
                [request.request_hash],
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO workspace_maintenance_cycle (
                        cycle_id, request_hash, request_json, phase, status,
                        change_set_json, child_task_refs_json, effect_receipts_json,
                        retry_after_at, failure_code, transport_workers, created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, 'preflight', 'running', NULL, '[]', '[]',
                        NULL, NULL, 4, ?, ?
                    )
                    """,
                    [cycle_id, request.request_hash, _json(asdict(request)), now, now],
                )
                self._append_event(
                    connection,
                    cycle_id=cycle_id,
                    kind="workspace_maintenance.admitted",
                    details={"request_hash": request.request_hash},
                    observed_at=now,
                )
            elif str(row[0]) != cycle_id:
                raise ValueError("maintenance request hash is bound to another cycle")
            return cycle_id

        return self.cycle(self._write(operation))

    def update(
        self,
        cycle_id: str,
        *,
        phase: MaintenancePhase,
        status: MaintenanceStatus,
        observed_at: datetime,
        change_set: MarketDataChangeSet | None = None,
        child_task_refs: tuple[str, ...] | None = None,
        effect_receipts: tuple[str, ...] | None = None,
        retry_after_at: datetime | None = None,
        failure_code: str | None = None,
        transport_workers: int | None = None,
    ) -> WorkspaceMaintenanceCycle:
        """Persist phase/status and supplied cycle effects under the mutation owner.

        Args:
            cycle_id: Durable maintenance cycle identifier.
            phase: Maintenance phase to record.
            status: Maintenance lifecycle state to record.
            observed_at: Operational observation clock; callers supply an aware instant.
            change_set: Observed market-data/membership changes to propagate.
            child_task_refs: Optional replacement for the recorded child task references.
            effect_receipts: Optional replacement for committed effect receipt references.
            retry_after_at: Optional next admissible retry clock.
            failure_code: Stable failure code retained for diagnosis.
            transport_workers: Optional replacement for the recorded transport worker count.

        Returns:
            The updated durable cycle.

        Raises:
            ValueError: The named cycle does not exist.
        """
        now = observed_at.astimezone(UTC).replace(tzinfo=None)

        def operation(connection: duckdb.DuckDBPyConnection) -> None:
            result = connection.execute(
                """
                UPDATE workspace_maintenance_cycle
                SET phase = ?, status = ?, change_set_json = COALESCE(?, change_set_json),
                    child_task_refs_json = COALESCE(?, child_task_refs_json),
                    effect_receipts_json = COALESCE(?, effect_receipts_json), retry_after_at = ?,
                    failure_code = ?, transport_workers = COALESCE(?, transport_workers),
                    updated_at = ?
                WHERE cycle_id = ?
                """,
                [
                    phase.value,
                    status.value,
                    _json(asdict(change_set)) if change_set is not None else None,
                    _json(child_task_refs) if child_task_refs is not None else None,
                    _json(effect_receipts) if effect_receipts is not None else None,
                    retry_after_at.astimezone(UTC).replace(tzinfo=None) if retry_after_at else None,
                    failure_code,
                    transport_workers,
                    now,
                    cycle_id,
                ],
            )
            if result.rowcount == 0:
                raise ValueError("workspace maintenance cycle does not exist")
            self._append_event(
                connection,
                cycle_id=cycle_id,
                kind=f"workspace_maintenance.{status.value}",
                details={"phase": phase.value, "failure_code": failure_code},
                observed_at=now,
            )

        self._write(operation)
        return self.cycle(cycle_id)

    MARKET_DATA_SCOPE_EVENT = "workspace_maintenance.market_data_scope"

    def record_market_data_scope(
        self,
        cycle_id: str,
        *,
        maintenance_id: str,
        manifest_revision: str,
        as_of_session: date,
        observed_at: datetime,
    ) -> None:
        """Seal the actual maintenance-run scope in this cycle's event log.

        Record the maintenance run this cycle admitted, in the cycle's own event log, so a
        later reader resolves the cycle's listing units by the run the coordinator actually
        keyed them under -- a membership transition or a candidate recheck runs under a
        manifest the request does not name. Idempotent: the same scope is recorded once.
        """
        details = {
            "maintenance_id": maintenance_id,
            "manifest_revision": manifest_revision,
            "as_of_session": as_of_session.isoformat(),
        }
        if self.market_data_scope(cycle_id) == details:
            return
        now = observed_at.astimezone(UTC).replace(tzinfo=None)

        def operation(connection: duckdb.DuckDBPyConnection) -> None:
            self._append_event(
                connection,
                cycle_id=cycle_id,
                kind=self.MARKET_DATA_SCOPE_EVENT,
                details=details,
                observed_at=now,
            )

        self._write(operation)

    def market_data_scope(self, cycle_id: str) -> dict[str, str] | None:
        """Resolve the latest recorded market-data scope for this cycle.

        The maintenance run this cycle recorded (its latest scope event), or None for a
        cycle that recorded none (before this record existed, or before its market-data
        phase).
        """
        details = self._latest_event(cycle_id, self.MARKET_DATA_SCOPE_EVENT)
        if details is None:
            return None
        keys = ("maintenance_id", "manifest_revision", "as_of_session")
        return {key: str(details[key]) for key in keys}

    WORKING_MANIFEST_EVENT = "workspace_maintenance.working_manifest"

    def record_working_manifest(
        self, cycle_id: str, *, manifest_revision: str, authority: str, observed_at: datetime
    ) -> None:
        """Record the membership this cycle made active, in its own event log.

        A cycle that moves the active membership -- a transition's qualified root or a child
        it derives (Sector exclusion, gateway quarantine, baseline qualification) -- owns that
        working membership until its Panel publishes. Recovery reads it here instead of
        deriving it again from the journal; the same membership is recorded once, and only
        on an existing cycle that has not finished.

        Raises:
            ValueError: The cycle does not exist or has finished.
        """
        if self.working_manifest(cycle_id) == manifest_revision:
            return
        details = {"manifest_revision": manifest_revision, "authority": authority}
        now = observed_at.astimezone(UTC).replace(tzinfo=None)

        def operation(connection: duckdb.DuckDBPyConnection) -> None:
            row = connection.execute(
                "SELECT status FROM workspace_maintenance_cycle WHERE cycle_id = ?", [cycle_id]
            ).fetchone()
            if row is None:
                raise ValueError("workspace maintenance cycle does not exist")
            if MaintenanceStatus(str(row[0])) in {
                MaintenanceStatus.COMPLETED,
                MaintenanceStatus.NOOP,
                MaintenanceStatus.CANCELLED,
            }:
                raise ValueError("workspace maintenance cycle has finished")
            self._append_event(
                connection,
                cycle_id=cycle_id,
                kind=self.WORKING_MANIFEST_EVENT,
                details=details,
                observed_at=now,
            )

        self._write(operation)

    def working_manifest(self, cycle_id: str) -> str | None:
        """The membership this cycle last made active, or None when it moved none."""
        details = self._latest_event(cycle_id, self.WORKING_MANIFEST_EVENT)
        return None if details is None else str(details["manifest_revision"])

    AUDIT_SUPPLEMENT_EVENT = "workspace_maintenance.audit_supplement"

    def record_audit_supplement(
        self,
        cycle_id: str,
        *,
        listing_ids: tuple[str, ...],
        requirement_receipt: str,
        audit_plan_hash: str,
        observed_at: datetime,
    ) -> None:
        """Admit full-history audits a stopped cycle receipted as required, onto that cycle.

        The cycle keeps its sealed request; its next run reads these listings beside the
        request's own. The requirement must be one this cycle receipted, and the same
        admission is recorded once.

        Raises:
            ValueError: `workspace_data_update.audit_requirement_unverified` when the cycle
                holds no such requirement receipt.
        """
        if requirement_receipt not in self.cycle(cycle_id).effect_receipts:
            raise ValueError("workspace_data_update.audit_requirement_unverified")
        details = {
            "listing_ids": sorted(listing_ids),
            "requirement_receipt": requirement_receipt,
            "audit_plan_hash": audit_plan_hash,
        }
        if self._latest_event(cycle_id, self.AUDIT_SUPPLEMENT_EVENT) == details:
            return
        now = observed_at.astimezone(UTC).replace(tzinfo=None)

        def operation(connection: duckdb.DuckDBPyConnection) -> None:
            self._append_event(
                connection,
                cycle_id=cycle_id,
                kind=self.AUDIT_SUPPLEMENT_EVENT,
                details=details,
                observed_at=now,
            )

        self._write(operation)

    def audit_supplement(self, cycle_id: str) -> tuple[str, ...]:
        """Every listing a supplement admitted onto this cycle, sorted."""
        return tuple(
            sorted({v for item in self._supplements(cycle_id) for v in item["listing_ids"]})
        )

    def admitted_audit_plans(self, cycle_id: str) -> frozenset[str]:
        """The audit plans whose confirmation admitted a supplement onto this cycle."""
        return frozenset(str(item["audit_plan_hash"]) for item in self._supplements(cycle_id))

    def _supplements(self, cycle_id: str) -> list[dict[str, Any]]:
        def operation(connection: duckdb.DuckDBPyConnection) -> Any:
            return connection.execute(
                "SELECT details_json FROM workspace_maintenance_event "
                "WHERE cycle_id = ? AND kind = ?",
                [cycle_id, self.AUDIT_SUPPLEMENT_EVENT],
            ).fetchall()

        return [dict(json.loads(str(row))) for (row,) in self._read(operation)]

    def _latest_event(self, cycle_id: str, kind: str) -> dict[str, Any] | None:
        def operation(connection: duckdb.DuckDBPyConnection) -> Any:
            return connection.execute(
                """
                SELECT details_json FROM workspace_maintenance_event
                WHERE cycle_id = ? AND kind = ? ORDER BY sequence DESC LIMIT 1
                """,
                [cycle_id, kind],
            ).fetchone()

        row = self._read(operation)
        return None if row is None else dict(json.loads(str(row[0])))

    def cancel(self, cycle_id: str, *, observed_at: datetime) -> WorkspaceMaintenanceCycle:
        """Record that the Task running this cycle was cancelled at a safe checkpoint.

        Only a cycle that was still open (running, deferred, review pending or
        blocked) can be cancelled; a completed cycle keeps its completion, and
        cancelling twice records nothing new. The phase is kept so the record
        says where the work stopped; no retry time survives.
        """
        now = observed_at.astimezone(UTC).replace(tzinfo=None)

        def operation(connection: duckdb.DuckDBPyConnection) -> None:
            row = connection.execute(
                "SELECT status, phase FROM workspace_maintenance_cycle WHERE cycle_id = ?",
                [cycle_id],
            ).fetchone()
            if row is None:
                raise ValueError("workspace maintenance cycle does not exist")
            status = MaintenanceStatus(str(row[0]))
            if status is MaintenanceStatus.CANCELLED:
                return
            if status in {MaintenanceStatus.COMPLETED, MaintenanceStatus.NOOP}:
                raise ValueError("workspace maintenance cycle is already complete")
            connection.execute(
                """
                UPDATE workspace_maintenance_cycle
                SET status = ?, retry_after_at = NULL, failure_code = ?, updated_at = ?
                WHERE cycle_id = ?
                """,
                [MaintenanceStatus.CANCELLED.value, CANCELLED_AT_SAFE_CHECKPOINT, now, cycle_id],
            )
            self._append_event(
                connection,
                cycle_id=cycle_id,
                kind="workspace_maintenance.cancelled",
                details={"phase": str(row[1]), "failure_code": CANCELLED_AT_SAFE_CHECKPOINT},
                observed_at=now,
            )

        self._write(operation)
        return self.cycle(cycle_id)

    def record_action_audit(self, receipt: ActionAuditChainReceipt) -> bool:
        """Persist an immutable audit receipt once with its required predecessor.

        Args:
            receipt: Immutable receipt to persist.

        Returns:
            Whether a new receipt was inserted rather than identical content reused.

        Raises:
            ValueError: A receipt identity has different bytes or its predecessor is absent.
        """

        def operation(connection: duckdb.DuckDBPyConnection) -> bool:
            prior = connection.execute(
                "SELECT receipt_json FROM action_audit_chain_receipt WHERE receipt_hash = ?",
                [receipt.receipt_hash],
            ).fetchone()
            document = _json(asdict(receipt))
            if prior is not None:
                if str(prior[0]) != document:
                    raise ValueError("action audit receipt hash was reused with new content")
                return False
            if receipt.previous_receipt_hash is not None:
                previous = connection.execute(
                    "SELECT receipt_hash FROM action_audit_chain_receipt WHERE receipt_hash = ?",
                    [receipt.previous_receipt_hash],
                ).fetchone()
                if previous is None:
                    raise ValueError("action audit receipt chain has a missing predecessor")
            connection.execute(
                "INSERT INTO action_audit_chain_receipt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    receipt.receipt_hash,
                    receipt.listing_id,
                    receipt.provider,
                    receipt.audit_scope.value,
                    receipt.previous_receipt_hash,
                    receipt.full_anchor_receipt_hash,
                    receipt.covered_through_session,
                    document,
                    receipt.observed_at.astimezone(UTC).replace(tzinfo=None),
                ],
            )
            return True

        return bool(self._write(operation))

    def latest_action_audit(
        self, listing_id: str, provider: str, *, full_only: bool = False
    ) -> ActionAuditChainReceipt | None:
        """Resolve the latest recorded audit for one listing/provider scope.

        Args:
            listing_id: Exact listing identifier whose evidence is recorded.
            provider: Bound market data provider for acquisition or receipt verification.
            full_only: Whether to restrict lookup to full-history audits.

        Returns:
            The latest matching audit receipt, or None when absent.
        """
        return self.latest_action_audits((listing_id,), provider, full_only=full_only).get(
            listing_id
        )

    def latest_action_audits(
        self,
        listing_ids: tuple[str, ...],
        provider: str,
        *,
        full_only: bool = False,
    ) -> dict[str, ActionAuditChainReceipt]:
        """Read the newest audit per listing in one authoritative scan."""
        resolved: dict[str, ActionAuditChainReceipt] = {}
        for receipt in self.action_audits(listing_ids, provider, full_only=full_only):
            resolved.setdefault(receipt.listing_id, receipt)
        return resolved

    def action_audits(
        self,
        listing_ids: tuple[str, ...],
        provider: str,
        *,
        full_only: bool = False,
    ) -> tuple[ActionAuditChainReceipt, ...]:
        """Read audit chains newest-first without reopening DuckDB per listing."""
        ordered_ids = tuple(sorted(set(listing_ids)))
        if not ordered_ids:
            return ()

        def operation(connection: duckdb.DuckDBPyConnection) -> Any:
            clause = "AND audit_scope = 'full'" if full_only else ""
            placeholders = ",".join("?" for _ in ordered_ids)
            return connection.execute(
                f"""
                SELECT listing_id, receipt_json FROM action_audit_chain_receipt
                WHERE provider = ? AND listing_id IN ({placeholders}) {clause}
                ORDER BY listing_id, observed_at DESC, receipt_hash DESC
                """,
                [provider, *ordered_ids],
            ).fetchall()

        rows = self._read(operation)
        resolved: list[ActionAuditChainReceipt] = []
        for _listing_id, document in rows:
            payload = json.loads(str(document))
            resolved.append(
                ActionAuditChainReceipt(
                    receipt_hash=str(payload["receipt_hash"]),
                    listing_id=str(payload["listing_id"]),
                    provider=str(payload["provider"]),
                    audit_scope=ActionAuditScope(str(payload["audit_scope"])),
                    previous_receipt_hash=payload.get("previous_receipt_hash"),
                    full_anchor_receipt_hash=str(payload["full_anchor_receipt_hash"]),
                    history_start=datetime.fromisoformat(str(payload["history_start"])).date(),
                    history_end=datetime.fromisoformat(str(payload["history_end"])).date(),
                    covered_through_session=datetime.fromisoformat(
                        str(payload["covered_through_session"])
                    ).date(),
                    window_action_hash=str(payload["window_action_hash"]),
                    action_set_hash=str(payload["action_set_hash"]),
                    raw_evidence_hash=str(payload["raw_evidence_hash"]),
                    mapping_revision=str(payload["mapping_revision"]),
                    data_policy_hash=str(payload["data_policy_hash"]),
                    provider_receipt_hash=str(payload["provider_receipt_hash"]),
                    observed_at=datetime.fromisoformat(str(payload["observed_at"])),
                )
            )
        return tuple(resolved)

    def record_data_remediation_failure(self, receipt: DataRemediationFailureReceipt) -> bool:
        """Persist a terminal Agent failure without recording an effect."""

        def operation(connection: duckdb.DuckDBPyConnection) -> bool:
            document = _json(asdict(receipt))
            row = connection.execute(
                "SELECT receipt_json FROM data_remediation_failure_receipt WHERE receipt_hash = ?",
                [receipt.receipt_hash],
            ).fetchone()
            if row is not None:
                if str(row[0]) != document:
                    raise ValueError("Data remediation failure receipt identity was reused")
                return False
            connection.execute(
                "INSERT INTO data_remediation_failure_receipt VALUES (?, ?, ?, ?, ?, ?)",
                [
                    receipt.receipt_hash,
                    receipt.maintenance_id,
                    receipt.case_token,
                    receipt.failure_code,
                    document,
                    receipt.observed_at.astimezone(UTC).replace(tzinfo=None),
                ],
            )
            return True

        return bool(self._write(operation))

    def latest_data_remediation_failure(self) -> DataRemediationFailureReceipt | None:
        """The newest recorded remediation failure, for the readback that names it."""

        def operation(connection: duckdb.DuckDBPyConnection) -> Any:
            return connection.execute(
                """
                SELECT receipt_json FROM data_remediation_failure_receipt
                ORDER BY observed_at DESC, receipt_hash DESC LIMIT 1
                """
            ).fetchone()

        row = self._read(operation)
        if row is None:
            return None
        return DataRemediationFailureReceipt.read_document(json.loads(str(row[0])))

    @staticmethod
    def read_data_update_receipt(
        database_path: Path,
        plan_hash: str | None = None,
    ) -> WorkspaceDataUpdateReceipt | None:
        """Read existing publication without initializing maintenance storage."""
        if not database_path.is_file():
            return None
        with open_workspace_database(database_path, read_only=True) as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'workspace_data_update_receipt'"
                ).fetchone()
                is None
            ):
                return None
            where, values = ("", []) if plan_hash is None else ("WHERE plan_hash = ?", [plan_hash])
            row = connection.execute(
                "SELECT plan_hash, receipt_hash, receipt_json "
                f"FROM workspace_data_update_receipt {where} "
                "ORDER BY published_at DESC, receipt_hash LIMIT 1",
                values,
            ).fetchone()
        if row is None:
            return None
        receipt = WorkspaceDataUpdateReceipt.model_validate_json(row[2])
        if receipt.plan_hash != row[0] or receipt.content_hash != row[1]:
            raise ValueError("workspace_data_update.receipt_binding_invalid")
        return cast(WorkspaceDataUpdateReceipt, receipt)

    def publish_data_update_receipt(self, receipt: WorkspaceDataUpdateReceipt) -> None:
        """Publish one immutable completed update receipt per plan identity.

        Args:
            receipt: Immutable receipt to persist.

        Raises:
            ValueError: The plan identity already binds different receipt content.
        """
        document = receipt.model_dump_json()

        def operation(connection: duckdb.DuckDBPyConnection) -> None:
            prior = connection.execute(
                "SELECT receipt_json FROM workspace_data_update_receipt WHERE plan_hash = ?",
                [receipt.plan_hash],
            ).fetchone()
            if prior is not None:
                if prior[0] != document:
                    raise ValueError("workspace_data_update.receipt_identity_reused")
                return
            connection.execute(
                "INSERT INTO workspace_data_update_receipt VALUES (?, ?, ?, ?)",
                [
                    receipt.plan_hash,
                    receipt.content_hash,
                    document,
                    receipt.completed_at.astimezone(UTC).replace(tzinfo=None),
                ],
            )

        self._write(operation)

    @staticmethod
    def read_update_retry_after(database_path: Path, request_hash: str) -> datetime | None:
        """Read a cycle's retry clock through a read-only database connection.

        Args:
            database_path: Workspace DuckDB path for maintenance records.
            request_hash: Maintenance request identity used to locate the durable cycle.

        Returns:
            The UTC retry clock, or None when no retry is recorded.

        Raises:
            ValueError: No cycle has the requested request identity.
        """
        with open_workspace_database(database_path, read_only=True) as connection:
            row = connection.execute(
                "SELECT retry_after_at FROM workspace_maintenance_cycle WHERE request_hash = ?",
                [request_hash],
            ).fetchone()
        if row is None:
            raise ValueError("workspace_data_update.cycle_absent")
        return row[0].replace(tzinfo=UTC) if row[0] is not None else None

    def cycle(self, cycle_id: str) -> WorkspaceMaintenanceCycle:
        """Reconstruct the durable request, change set and effect state for a cycle.

        Args:
            cycle_id: Durable maintenance cycle identifier.

        Returns:
            The stored cycle with reconstructed typed scope and aware database clocks.

        Raises:
            ValueError: The named cycle does not exist.
        """

        def operation(connection: duckdb.DuckDBPyConnection) -> Any:
            return connection.execute(
                """
                SELECT request_json, phase, status, change_set_json, child_task_refs_json,
                       effect_receipts_json, retry_after_at, failure_code,
                       transport_workers, updated_at
                FROM workspace_maintenance_cycle WHERE cycle_id = ?
                """,
                [cycle_id],
            ).fetchone()

        row = self._read(operation)
        if row is None:
            raise ValueError("workspace maintenance cycle does not exist")
        request_payload = json.loads(str(row[0]))
        recheck = request_payload.get("candidate_recheck")
        data_recheck = request_payload.get("candidate_data_recheck")
        request = WorkspaceMaintenanceRequest(
            market_profile_id=str(request_payload["market_profile_id"]),
            target_market_session=datetime.fromisoformat(
                str(request_payload["target_market_session"])
            ).date(),
            knowledge_cutoff_at=datetime.fromisoformat(str(request_payload["knowledge_cutoff_at"]))
            if request_payload["knowledge_cutoff_at"] is not None
            else None,
            trigger=MaintenanceTrigger(str(request_payload["trigger"])),
            membership_revision=str(request_payload["membership_revision"]),
            data_policy_hash=str(request_payload["data_policy_hash"]),
            feature_policy_hash=str(request_payload["feature_policy_hash"]),
            request_hash=str(request_payload["request_hash"]),
            full_history_listing_ids=tuple(request_payload.get("full_history_listing_ids", ())),
            requested_at=datetime.fromisoformat(str(request_payload["requested_at"]))
            if request_payload.get("requested_at") is not None
            else None,
            candidate_recheck=FeatureCandidateRecheck(
                parent_manifest_revision=str(recheck["parent_manifest_revision"]),
                listing_ids=tuple(recheck["listing_ids"]),
            )
            if recheck is not None
            else None,
            candidate_data_recheck=CandidateDataRecheck(
                prior_onboarding_id=str(data_recheck["prior_onboarding_id"]),
                candidate_manifest_hash=str(data_recheck["candidate_manifest_hash"]),
                qualified_parent_revision=str(data_recheck["qualified_parent_revision"]),
                listing_ids=tuple(data_recheck["listing_ids"]),
                history_start=datetime.fromisoformat(str(data_recheck["history_start"])).date(),
            )
            if data_recheck is not None
            else None,
        )
        change_set = None
        if row[3] is not None:
            payload = json.loads(str(row[3]))
            from alphalattice.control.data_platform.maintenance.contracts import (
                ListingMarketDataChange,
            )

            change_set = MarketDataChangeSet(
                listing_changes=tuple(
                    ListingMarketDataChange(
                        listing_id=str(item["listing_id"]),
                        new_session_start=(
                            datetime.fromisoformat(str(item["new_session_start"])).date()
                            if item.get("new_session_start")
                            else None
                        ),
                        new_sessions=tuple(
                            datetime.fromisoformat(str(value)).date()
                            for value in item.get("new_sessions", ())
                        ),
                        raw_correction_start=(
                            datetime.fromisoformat(str(item["raw_correction_start"])).date()
                            if item.get("raw_correction_start")
                            else None
                        ),
                        action_correction_start=(
                            datetime.fromisoformat(str(item["action_correction_start"])).date()
                            if item.get("action_correction_start")
                            else None
                        ),
                        raw_correction_sessions=tuple(
                            datetime.fromisoformat(str(value)).date()
                            for value in item.get("raw_correction_sessions", ())
                        ),
                        adjusted_return_change_sessions=tuple(
                            datetime.fromisoformat(str(value)).date()
                            for value in item.get("adjusted_return_change_sessions", ())
                        ),
                        source_receipt_hashes=tuple(item["source_receipt_hashes"]),
                    )
                    for item in payload["listing_changes"]
                ),
                spy_correction_start=(
                    datetime.fromisoformat(str(payload["spy_correction_start"])).date()
                    if payload.get("spy_correction_start")
                    else None
                ),
                spy_return_change_sessions=tuple(
                    datetime.fromisoformat(str(value)).date()
                    for value in payload.get("spy_return_change_sessions", ())
                ),
                sector_revision_changed=bool(payload["sector_revision_changed"]),
                membership_additions=tuple(payload["membership_additions"]),
                membership_removals=tuple(payload["membership_removals"]),
                receipt_hashes=tuple(payload["receipt_hashes"]),
                change_set_hash=str(payload["change_set_hash"]),
            )
        updated = row[9].replace(tzinfo=UTC) if row[9].tzinfo is None else row[9]
        retry = row[6].replace(tzinfo=UTC) if row[6] and row[6].tzinfo is None else row[6]
        return WorkspaceMaintenanceCycle(
            cycle_id=cycle_id,
            request=request,
            phase=MaintenancePhase(str(row[1])),
            status=MaintenanceStatus(str(row[2])),
            market_data_change_set=change_set,
            child_task_refs=tuple(json.loads(str(row[4]))),
            effect_receipts=tuple(json.loads(str(row[5]))),
            retry_after_at=retry,
            failure_code=str(row[7]) if row[7] is not None else None,
            transport_workers=int(row[8]),
            updated_at=updated,
        )

    def latest_cycle(self, market_profile_id: str) -> WorkspaceMaintenanceCycle | None:
        """Resolve the most recently updated cycle for one market profile.

        Args:
            market_profile_id: Workspace market profile identifier.

        Returns:
            The latest matching cycle, or None when the table or matching profile is absent.
        """

        def operation(connection: duckdb.DuckDBPyConnection) -> str | None:
            if (
                connection.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_name = 'workspace_maintenance_cycle'"
                ).fetchone()[0]
                == 0
            ):
                return None
            rows = connection.execute(
                """
                SELECT cycle_id, request_json FROM workspace_maintenance_cycle
                ORDER BY updated_at DESC, cycle_id DESC
                """
            ).fetchall()
            for cycle_id, request_json in rows:
                payload = json.loads(str(request_json))
                if payload.get("market_profile_id") == market_profile_id:
                    return str(cycle_id)
            return None

        cycle_id = self._read(operation)
        return self.cycle(cycle_id) if cycle_id is not None else None

    def reconcile_stale_running_cycles(
        self,
        market_profile_id: str,
        *,
        observed_at: datetime,
        stale_after: timedelta = timedelta(seconds=30),
    ) -> tuple[str, ...]:
        """Terminalize orphaned cycles before a new process starts work.

        The caller must already hold the workspace writer lease. That excludes
        a still-live prior process; age is therefore used only to avoid
        rewriting a cycle that was admitted moments before the same process
        finished composition.
        """
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("stale-cycle observation must be timezone-aware")
        if stale_after.total_seconds() <= 0:
            raise ValueError("stale-cycle threshold must be positive")
        now = observed_at.astimezone(UTC).replace(tzinfo=None)
        cutoff = now - stale_after

        def operation(connection: duckdb.DuckDBPyConnection) -> tuple[str, ...]:
            rows = connection.execute(
                """
                SELECT cycle_id, request_json
                FROM workspace_maintenance_cycle
                WHERE status = 'running' AND updated_at < ?
                ORDER BY cycle_id
                """,
                [cutoff],
            ).fetchall()
            reconciled: list[str] = []
            for cycle_id, request_json in rows:
                request = json.loads(str(request_json))
                if request.get("market_profile_id") != market_profile_id:
                    continue
                connection.execute(
                    """
                    UPDATE workspace_maintenance_cycle
                    SET status = 'blocked',
                        failure_code = 'workspace_maintenance.worker_lost',
                        retry_after_at = NULL
                    WHERE cycle_id = ? AND status = 'running'
                    """,
                    [cycle_id],
                )
                self._append_event(
                    connection,
                    cycle_id=str(cycle_id),
                    kind="workspace_maintenance.worker_lost",
                    details={"stale_after_seconds": stale_after.total_seconds()},
                    observed_at=now,
                )
                reconciled.append(str(cycle_id))
            return tuple(reconciled)

        return self._write(operation)

    @staticmethod
    def _append_event(
        connection: duckdb.DuckDBPyConnection,
        *,
        cycle_id: str,
        kind: str,
        details: Mapping[str, object],
        observed_at: datetime,
    ) -> None:
        sequence = int(
            connection.execute(
                "SELECT COALESCE(max(sequence), 0) + 1 FROM workspace_maintenance_event "
                "WHERE cycle_id = ?",
                [cycle_id],
            ).fetchone()[0]
        )
        event_id = canonical_hash([cycle_id, sequence, kind, details])
        connection.execute(
            "INSERT INTO workspace_maintenance_event VALUES (?, ?, ?, ?, ?, ?)",
            [event_id, cycle_id, sequence, kind, _json(details), observed_at],
        )
