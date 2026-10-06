"""Single repository owner for the cross-domain workspace readiness projection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb

from alphalattice.control.workspace_runtime.database import WorkspaceRepository


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class WorkspaceReadinessRecord:
    """Persisted market-profile readiness and active/pending manifest bindings.

    Attributes:
        market_profile_id: Market-profile key owning this readiness record.
        status: Current readiness state.
        active_manifest_id: Installed active manifest identifier, or None.
        active_manifest_revision: Installed active manifest revision, or None.
        active_membership_fingerprint: Identity of active membership, or None.
        active_candidate_manifest_document: Active candidate manifest mapping, or None.
        pending_membership_fingerprint: Identity of pending membership, or None.
        pending_candidate_manifest_document: Pending candidate manifest mapping, or None.
        last_checked_at: Most recent source-check timestamp, or None.
        last_changed_at: Most recent readiness-change timestamp, or None.
        failure_code: Stable readiness refusal code, or None.
        updated_at: Time this persisted readiness record was updated.
        source_check_failed_at: Retained source-check failure timestamp, or None.
    """

    market_profile_id: str
    status: str
    active_manifest_id: str | None
    active_manifest_revision: str | None
    active_membership_fingerprint: str | None
    active_candidate_manifest_document: dict[str, object] | None
    pending_membership_fingerprint: str | None
    pending_candidate_manifest_document: dict[str, object] | None
    last_checked_at: datetime | None
    last_changed_at: datetime | None
    failure_code: str | None
    updated_at: datetime
    source_check_failed_at: datetime | None = None


class WorkspaceReadinessRepository(WorkspaceRepository):
    """Own readiness schema, codec, and generic projection mutations."""

    @staticmethod
    def ensure_schema(connection: duckdb.DuckDBPyConnection) -> None:
        """Ensure the readiness table and its source-check failure timestamp column exist.

        Args:
            connection: Open workspace database connection used for this operation.
        """
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workspace_readiness (
                market_profile_id VARCHAR PRIMARY KEY,
                status VARCHAR NOT NULL,
                active_manifest_id VARCHAR,
                active_manifest_revision VARCHAR,
                active_membership_fingerprint VARCHAR,
                active_candidate_manifest_json VARCHAR,
                pending_membership_fingerprint VARCHAR,
                pending_candidate_manifest_json VARCHAR,
                last_checked_at TIMESTAMP,
                last_changed_at TIMESTAMP,
                failure_code VARCHAR,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )
        connection.execute(
            "ALTER TABLE workspace_readiness "
            "ADD COLUMN IF NOT EXISTS source_check_failed_at TIMESTAMP"
        )

    def load(self, market_profile_id: str) -> WorkspaceReadinessRecord | None:
        """Read one readiness record while supporting stores without the failure-time column.

        Args:
            market_profile_id: Exact market-profile key selecting the readiness record.

        Returns:
            Persisted record, or None when the file, table or profile row is absent.
        """
        if not self.path.exists():
            return None
        connection = self._connect(read_only=True)
        try:
            try:
                columns = {
                    item[1]
                    for item in connection.execute(
                        "PRAGMA table_info('workspace_readiness')"
                    ).fetchall()
                }
                failed_column = (
                    "source_check_failed_at"
                    if "source_check_failed_at" in columns
                    else "NULL::TIMESTAMP"
                )
                row = connection.execute(
                    f"""
                    SELECT market_profile_id, status, active_manifest_id, active_manifest_revision,
                           active_membership_fingerprint, active_candidate_manifest_json,
                           pending_membership_fingerprint, pending_candidate_manifest_json,
                           last_checked_at, last_changed_at, failure_code, updated_at,
                           {failed_column}
                    FROM workspace_readiness WHERE market_profile_id = ?
                    """,
                    [market_profile_id],
                ).fetchone()
            except duckdb.CatalogException:
                return None
        finally:
            connection.close()
        if row is None:
            return None
        return WorkspaceReadinessRecord(
            market_profile_id=str(row[0]),
            status=str(row[1]),
            active_manifest_id=str(row[2]) if row[2] is not None else None,
            active_manifest_revision=str(row[3]) if row[3] is not None else None,
            active_membership_fingerprint=str(row[4]) if row[4] is not None else None,
            active_candidate_manifest_document=self._json_object(row[5]),
            pending_membership_fingerprint=str(row[6]) if row[6] is not None else None,
            pending_candidate_manifest_document=self._json_object(row[7]),
            last_checked_at=row[8],
            last_changed_at=row[9],
            failure_code=str(row[10]) if row[10] is not None else None,
            updated_at=row[11],
            source_check_failed_at=row[12],
        )

    def save(
        self,
        *,
        market_profile_id: str,
        status: str,
        active_manifest_id: str | None,
        active_manifest_revision: str | None,
        active_membership_fingerprint: str | None,
        active_candidate_manifest_document: Mapping[str, object] | None,
        pending_membership_fingerprint: str | None,
        pending_candidate_manifest_document: Mapping[str, object] | None,
        last_checked_at: datetime | None,
        last_changed_at: datetime | None,
        failure_code: str | None,
        observed_at: datetime,
        source_check_failed_at: datetime | None = None,
    ) -> WorkspaceReadinessRecord:
        """Upsert readiness bindings and verify the resulting record is durable.

        Timestamps are stored as naive UTC. An omitted source-check failure timestamp preserves an
        existing value; active and pending documents are encoded as JSON.

        Args:
            market_profile_id: Exact market-profile key selecting the readiness record.
            status: Readiness state supplied by the deterministic readiness owner.
            active_manifest_id: Installed active manifest identifier, or None.
            active_manifest_revision: Installed active manifest revision, or None.
            active_membership_fingerprint: Identity of active manifest membership, or None.
            active_candidate_manifest_document: Active candidate manifest document, or None.
            pending_membership_fingerprint: Identity of pending candidate membership, or None.
            pending_candidate_manifest_document: Pending candidate manifest document, or None.
            last_checked_at: Most recent source-check time, or None.
            last_changed_at: Most recent readiness-change time, or None.
            failure_code: Stable readiness refusal code, or None.
            observed_at: Observation time stored as a naive UTC timestamp.
            source_check_failed_at: Source-check failure time; None preserves an existing stored
                failure time.

        Returns:
            Read-back persisted record.

        Raises:
            AssertionError: No record can be read after the write.
        """
        active_json = self._encode(active_candidate_manifest_document)
        pending_json = self._encode(pending_candidate_manifest_document)
        connection = self._connect()
        try:
            self.ensure_schema(connection)
            connection.execute(
                """
                INSERT INTO workspace_readiness (
                    market_profile_id, status, active_manifest_id, active_manifest_revision,
                    active_membership_fingerprint, active_candidate_manifest_json,
                    pending_membership_fingerprint, pending_candidate_manifest_json,
                    last_checked_at, last_changed_at, failure_code, updated_at,
                    source_check_failed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (market_profile_id) DO UPDATE SET
                    status = excluded.status,
                    active_manifest_id = excluded.active_manifest_id,
                    active_manifest_revision = excluded.active_manifest_revision,
                    active_membership_fingerprint = excluded.active_membership_fingerprint,
                    active_candidate_manifest_json = excluded.active_candidate_manifest_json,
                    pending_membership_fingerprint = excluded.pending_membership_fingerprint,
                    pending_candidate_manifest_json = excluded.pending_candidate_manifest_json,
                    last_checked_at = excluded.last_checked_at,
                    last_changed_at = excluded.last_changed_at,
                    failure_code = excluded.failure_code,
                    updated_at = excluded.updated_at,
                    source_check_failed_at = coalesce(
                        excluded.source_check_failed_at, workspace_readiness.source_check_failed_at)
                """,
                [
                    market_profile_id,
                    status,
                    active_manifest_id,
                    active_manifest_revision,
                    active_membership_fingerprint,
                    active_json,
                    pending_membership_fingerprint,
                    pending_json,
                    _utc_naive(last_checked_at) if last_checked_at is not None else None,
                    _utc_naive(last_changed_at) if last_changed_at is not None else None,
                    failure_code,
                    _utc_naive(observed_at),
                    _utc_naive(source_check_failed_at)
                    if source_check_failed_at is not None
                    else None,
                ],
            )
        finally:
            connection.close()
        record = self.load(market_profile_id)
        if record is None:
            raise AssertionError("workspace readiness write was not durable")
        return record

    @staticmethod
    def reconcile_incomplete_panel_identity(
        connection: duckdb.DuckDBPyConnection,
        *,
        market_profile_id: str,
        observed_at: datetime,
    ) -> None:
        """Demote research-ready records whose active panel binding lacks required identities.

        Only matching RESEARCH_READY rows with an incomplete active panel binding move to
        FEATURE_BUILDING with the content-identity refusal code.

        Args:
            connection: Open workspace database connection used for this operation.
            market_profile_id: Exact market-profile key selecting the readiness record.
            observed_at: Observation time stored as a naive UTC timestamp.
        """
        WorkspaceReadinessRepository.ensure_schema(connection)
        connection.execute(
            """
            UPDATE workspace_readiness
            SET status = 'FEATURE_BUILDING',
                failure_code = 'workspace_readiness.feature_panel_content_identity_required',
                updated_at = ?
            WHERE market_profile_id = ? AND status = 'RESEARCH_READY'
              AND EXISTS (
                  SELECT 1 FROM active_feature_panel_binding AS binding
                  WHERE binding.market_profile_id = workspace_readiness.market_profile_id
                    AND (
                        binding.panel_binding_hash IS NULL
                        OR binding.panel_content_hash IS NULL
                        OR binding.materialization_receipt_hash IS NULL
                    )
              )
            """,
            [_utc_naive(observed_at), market_profile_id],
        )

    @staticmethod
    def research_ready_for_snapshot(
        connection: duckdb.DuckDBPyConnection, *, snapshot_hash: str, observed_at: datetime
    ) -> None:
        """Marks research-ready the profile whose active Panel the snapshot publishes (V155).

        The Feature repository runs it on the connection that registers the snapshot.
        """
        connection.execute(
            """
            UPDATE workspace_readiness AS readiness
            SET status = 'RESEARCH_READY', failure_code = NULL, updated_at = ?
            WHERE EXISTS (
                SELECT 1
                FROM active_feature_panel_binding AS panel
                JOIN feature_panel_snapshot_manifest AS snapshot
                  ON snapshot.panel_content_hash = panel.panel_content_hash
                 AND snapshot.as_of_session = panel.as_of_session
                 AND snapshot.knowledge_cutoff_at = panel.knowledge_cutoff_at
                 AND snapshot.temporal_identity_hash = panel.temporal_identity_hash
                WHERE panel.market_profile_id = readiness.market_profile_id
                  AND snapshot.snapshot_hash = ?
                  AND snapshot.lifecycle = 'ACTIVE'
            )
            """,
            [_utc_naive(observed_at), snapshot_hash],
        )

    @staticmethod
    def blocked_by_inactive_snapshot(
        connection: duckdb.DuckDBPyConnection, *, snapshot_hash: str, observed_at: datetime
    ) -> None:
        """Blocks the profile whose active Panel's snapshot was retired or quarantined (V155).

        The Feature repository runs it on the connection that changes the snapshot's lifecycle.
        """
        connection.execute(
            """
            UPDATE workspace_readiness AS readiness
            SET status = 'BLOCKED', failure_code = 'feature_panel.snapshot_not_active',
                updated_at = ?
            WHERE EXISTS (
                SELECT 1
                FROM active_feature_panel_binding AS panel
                JOIN feature_panel_snapshot_manifest AS snapshot
                  ON snapshot.panel_binding_hash = panel.panel_binding_hash
                 AND snapshot.panel_content_hash = panel.panel_content_hash
                 AND snapshot.as_of_session = panel.as_of_session
                 AND snapshot.knowledge_cutoff_at = panel.knowledge_cutoff_at
                 AND snapshot.temporal_identity_hash = panel.temporal_identity_hash
                WHERE panel.market_profile_id = readiness.market_profile_id
                  AND snapshot.snapshot_hash = ?
                  AND snapshot.lifecycle <> 'ACTIVE'
            )
            """,
            [_utc_naive(observed_at), snapshot_hash],
        )

    @staticmethod
    def feature_building(
        connection: duckdb.DuckDBPyConnection, *, market_profile_id: str, observed_at: datetime
    ) -> None:
        """Marks a profile's Features building (V155).

        The Feature repository runs it inside the transaction that activates a Panel.
        """
        connection.execute(
            """
            UPDATE workspace_readiness SET status = 'FEATURE_BUILDING', failure_code = NULL,
                updated_at = ? WHERE market_profile_id = ?
            """,
            [_utc_naive(observed_at), market_profile_id],
        )

    @staticmethod
    def feature_building_for_manifest(
        connection: duckdb.DuckDBPyConnection,
        *,
        market_profile_id: str,
        manifest_id: str,
        manifest_revision: str,
        membership_fingerprint: str,
        candidate_manifest_json: str,
        checked_at: datetime,
        changed_at: datetime,
    ) -> None:
        """Records an activated manifest and marks its Features building (V155).

        The market store runs it inside the transaction that activates the manifest.
        """
        connection.execute(
            """
            INSERT INTO workspace_readiness (
                market_profile_id, status, active_manifest_id, active_manifest_revision,
                active_membership_fingerprint, active_candidate_manifest_json,
                pending_membership_fingerprint, pending_candidate_manifest_json,
                last_checked_at, last_changed_at, failure_code, updated_at
            ) VALUES (?, 'FEATURE_BUILDING', ?, ?, ?, ?,
                                                    NULL, NULL, ?, ?, NULL, ?)
            ON CONFLICT (market_profile_id) DO UPDATE SET
                status = 'FEATURE_BUILDING',
                active_manifest_id = excluded.active_manifest_id,
                active_manifest_revision = excluded.active_manifest_revision,
                active_membership_fingerprint = excluded.active_membership_fingerprint,
                active_candidate_manifest_json = excluded.active_candidate_manifest_json,
                pending_membership_fingerprint = NULL,
                pending_candidate_manifest_json = NULL,
                last_checked_at = excluded.last_checked_at,
                last_changed_at = excluded.last_changed_at,
                failure_code = NULL,
                updated_at = excluded.updated_at
            """,
            [
                market_profile_id,
                manifest_id,
                manifest_revision,
                membership_fingerprint,
                candidate_manifest_json,
                _utc_naive(checked_at),
                _utc_naive(changed_at),
                _utc_naive(changed_at),
            ],
        )

    @staticmethod
    def feature_building_for_source_change(
        connection: duckdb.DuckDBPyConnection,
        *,
        market_profile_id: str,
        manifest_revision: str,
        membership_fingerprint: str,
        candidate_manifest_json: str,
        checked_at: datetime,
        changed_at: datetime,
    ) -> None:
        """Records a source change that left the active set unchanged (V155).

        The market store runs it inside the transaction that records the change; only the
        profile whose active manifest is `manifest_revision` moves.
        """
        connection.execute(
            """
            UPDATE workspace_readiness
            SET status = 'FEATURE_BUILDING', active_membership_fingerprint = ?,
                active_candidate_manifest_json = ?, pending_membership_fingerprint = NULL,
                pending_candidate_manifest_json = NULL, last_checked_at = ?,
                last_changed_at = ?, failure_code = NULL, updated_at = ?
            WHERE market_profile_id = ? AND active_manifest_revision = ?
            """,
            [
                membership_fingerprint,
                candidate_manifest_json,
                _utc_naive(checked_at),
                _utc_naive(changed_at),
                _utc_naive(changed_at),
                market_profile_id,
                manifest_revision,
            ],
        )

    @staticmethod
    def _encode(value: Mapping[str, object] | None) -> str | None:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":")) if value is not None else None
        )

    @staticmethod
    def _json_object(value: object) -> dict[str, object] | None:
        if value is None:
            return None
        try:
            decoded = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise ValueError("workspace readiness JSON is invalid") from exc
        if not isinstance(decoded, dict):
            raise ValueError("workspace readiness JSON is not an object")
        return decoded


__all__ = ["WorkspaceReadinessRecord", "WorkspaceReadinessRepository"]
