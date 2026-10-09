"""Repository for post-membership research-foundation rebuild authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb

from alphalattice.control.workspace_runtime.database import WorkspaceRepository


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class FeatureUniverseRebuildRequirement:
    """Durable lifecycle for the complete post-membership verified prefix."""

    transition_id: str
    prior_manifest_revision: str | None
    next_manifest_revision: str
    lifecycle: str
    created_at: datetime
    updated_at: datetime | None
    panel_snapshot_hash: str | None
    factor_result_hash: str | None
    factor_slate_hash: str | None
    execution_outcome_hash: str | None
    foundation_hash: str | None
    revision_marker_hash: str | None
    failure_code: str | None


class ResearchFoundationStateRepository(WorkspaceRepository):
    """Own the active lifecycle after a Market membership transition is activated."""

    @staticmethod
    def require_rebuild(
        connection: duckdb.DuckDBPyConnection,
        *,
        transition_id: str,
        prior_manifest_revision: str | None,
        next_manifest_revision: str,
        at: datetime,
    ) -> None:
        """Record that an activated membership transition needs its Foundation rebuilt.

        This owner defines the transition; Market Data's activation runs it inside its own
        transaction, as the data platform composes it, so activation stays atomic.

        Args:
            connection: The activation's connection, inside its transaction.
            transition_id: The activated transition.
            prior_manifest_revision: The manifest it replaced, if any.
            next_manifest_revision: The manifest it activated.
            at: When it was activated.
        """
        connection.execute(
            """
            INSERT INTO feature_universe_rebuild_requirement (
                transition_id, prior_manifest_revision, next_manifest_revision,
                lifecycle, created_at, updated_at
            ) VALUES (?, ?, ?, 'REQUIRED', ?, ?)
            ON CONFLICT (transition_id) DO NOTHING
            """,
            [
                transition_id,
                prior_manifest_revision,
                next_manifest_revision,
                _utc_naive(at),
                _utc_naive(at),
            ],
        )

    def feature_universe_rebuild_requirement(
        self, transition_id: str
    ) -> FeatureUniverseRebuildRequirement:
        """Read the durable rebuild lifecycle for one activated membership transition.

        Args:
            transition_id: Activated transition whose Foundation lifecycle is requested.

        Returns:
            Recorded lifecycle, timestamps, published children and optional failure code.

        Raises:
            ValueError: No rebuild requirement exists for the requested transition.
        """
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT transition_id, prior_manifest_revision, next_manifest_revision,
                       lifecycle, created_at, updated_at, panel_snapshot_hash,
                       factor_result_hash, factor_slate_hash, execution_outcome_hash,
                       foundation_hash, revision_marker_hash, failure_code
                FROM feature_universe_rebuild_requirement
                WHERE transition_id = ?
                """,
                [transition_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise ValueError("feature universe rebuild requirement does not exist")
        return FeatureUniverseRebuildRequirement(
            transition_id=str(row[0]),
            prior_manifest_revision=str(row[1]) if row[1] is not None else None,
            next_manifest_revision=str(row[2]),
            lifecycle=str(row[3]),
            created_at=row[4],
            updated_at=row[5],
            panel_snapshot_hash=str(row[6]) if row[6] is not None else None,
            factor_result_hash=str(row[7]) if row[7] is not None else None,
            factor_slate_hash=str(row[8]) if row[8] is not None else None,
            execution_outcome_hash=str(row[9]) if row[9] is not None else None,
            foundation_hash=str(row[10]) if row[10] is not None else None,
            revision_marker_hash=str(row[11]) if row[11] is not None else None,
            failure_code=str(row[12]) if row[12] is not None else None,
        )

    def feature_universe_rebuild_for_manifest(
        self, manifest_revision: str
    ) -> FeatureUniverseRebuildRequirement | None:
        """Resolve the most recent rebuild requirement naming a manifest revision.

        Args:
            manifest_revision: Activated membership manifest to look up.

        Returns:
            Latest requirement ordered by creation clock and transition ID, or None if absent.
        """
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT transition_id
                FROM feature_universe_rebuild_requirement
                WHERE next_manifest_revision = ?
                ORDER BY created_at DESC, transition_id DESC
                LIMIT 1
                """,
                [manifest_revision],
            ).fetchone()
        finally:
            connection.close()
        return self.feature_universe_rebuild_requirement(str(row[0])) if row else None

    def start_feature_universe_rebuild(
        self, transition_id: str, *, observed_at: datetime
    ) -> FeatureUniverseRebuildRequirement:
        """Enter RUNNING from a required, running or blocked rebuild and verify readback.

        Args:
            transition_id: Existing membership transition to rebuild.
            observed_at: Recorded lifecycle-update clock.

        Returns:
            Durable RUNNING requirement with its prior failure code cleared.

        Raises:
            ValueError: The requirement is absent or cannot enter RUNNING.
        """
        connection = self._connect()
        try:
            connection.execute(
                """
                UPDATE feature_universe_rebuild_requirement
                SET lifecycle = 'RUNNING', updated_at = ?, failure_code = NULL
                WHERE transition_id = ? AND lifecycle IN ('REQUIRED', 'RUNNING', 'BLOCKED')
                """,
                [_utc_naive(observed_at), transition_id],
            )
        finally:
            connection.close()
        result = self.feature_universe_rebuild_requirement(transition_id)
        if result.lifecycle != "RUNNING":
            raise ValueError("feature universe rebuild cannot enter RUNNING")
        return result

    def fulfill_feature_universe_rebuild(
        self,
        transition_id: str,
        *,
        panel_snapshot_hash: str,
        factor_result_hash: str,
        factor_slate_hash: str,
        execution_outcome_hash: str,
        foundation_hash: str,
        revision_marker_hash: str,
        observed_at: datetime,
    ) -> FeatureUniverseRebuildRequirement:
        """Seal the complete rebuilt prefix into a running or already fulfilled requirement.

        Args:
            transition_id: Existing activated transition whose prefix was rebuilt.
            panel_snapshot_hash: Qualified Panel publication identity.
            factor_result_hash: Verified Factor evidence result identity.
            factor_slate_hash: Published candidate slate identity.
            execution_outcome_hash: Qualified execution-outcome snapshot identity.
            foundation_hash: Verified downstream Foundation identity.
            revision_marker_hash: Marker sealing the verified rebuilt prefix.
            observed_at: Recorded lifecycle-update clock.

        Returns:
            Durable FULFILLED requirement with all child identities verified on readback.

        Raises:
            ValueError: A child handle has invalid length, the requirement is absent,
                or lifecycle/child readback differs from the submitted prefix.
        """
        values = (
            panel_snapshot_hash,
            factor_result_hash,
            factor_slate_hash,
            execution_outcome_hash,
            foundation_hash,
            revision_marker_hash,
        )
        if any(len(value) != 64 for value in values):
            raise ValueError("feature universe rebuild child identity is invalid")
        connection = self._connect()
        try:
            connection.execute(
                """
                UPDATE feature_universe_rebuild_requirement
                SET lifecycle = 'FULFILLED', updated_at = ?, panel_snapshot_hash = ?,
                    factor_result_hash = ?, factor_slate_hash = ?,
                    execution_outcome_hash = ?, foundation_hash = ?,
                    revision_marker_hash = ?, failure_code = NULL
                WHERE transition_id = ? AND lifecycle IN ('RUNNING', 'FULFILLED')
                """,
                [_utc_naive(observed_at), *values, transition_id],
            )
        finally:
            connection.close()
        result = self.feature_universe_rebuild_requirement(transition_id)
        expected = (
            result.panel_snapshot_hash,
            result.factor_result_hash,
            result.factor_slate_hash,
            result.execution_outcome_hash,
            result.foundation_hash,
            result.revision_marker_hash,
        )
        if result.lifecycle != "FULFILLED" or expected != values:
            raise ValueError("feature universe rebuild fulfillment readback differs")
        return result

    def block_feature_universe_rebuild(
        self, transition_id: str, *, failure_code: str, observed_at: datetime
    ) -> FeatureUniverseRebuildRequirement:
        """Record a required/running/blocked rebuild failure and verify its durable code.

        Args:
            transition_id: Existing activated transition whose rebuild cannot proceed.
            failure_code: Nonempty stable cause of the blocked rebuild.
            observed_at: Recorded lifecycle-update clock.

        Returns:
            Durable BLOCKED requirement carrying the requested failure code.

        Raises:
            ValueError: The code is empty, requirement is absent, or blocked readback differs.
        """
        if not failure_code:
            raise ValueError("blocked feature universe rebuild requires a failure code")
        connection = self._connect()
        try:
            connection.execute(
                """
                UPDATE feature_universe_rebuild_requirement
                SET lifecycle = 'BLOCKED', updated_at = ?, failure_code = ?
                WHERE transition_id = ? AND lifecycle IN ('REQUIRED', 'RUNNING', 'BLOCKED')
                """,
                [_utc_naive(observed_at), failure_code, transition_id],
            )
        finally:
            connection.close()
        result = self.feature_universe_rebuild_requirement(transition_id)
        if result.lifecycle != "BLOCKED" or result.failure_code != failure_code:
            raise ValueError("feature universe rebuild block readback differs")
        return result


__all__ = [
    "FeatureUniverseRebuildRequirement",
    "ResearchFoundationStateRepository",
]
