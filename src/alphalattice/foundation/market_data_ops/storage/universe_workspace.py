"""Read the workspace's frozen universe facts for admission and rollback."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime

import duckdb

from alphalattice.control.workspace_runtime.database import WorkspaceRepository
from alphalattice.foundation.market_data_ops.publication.projection import _canonical_hash
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.sources.membership import (
    MembershipEvent,
    UniverseBootstrapRecord,
)


def _membership_digest(listing_ids: Iterable[str]) -> str:
    """The digest of a manifest's member listings, in no particular order."""
    return _canonical_hash(sorted(set(listing_ids)))


def _utc_aware(value: datetime) -> datetime:
    """Project stored UTC wall time back into an absolute runtime contract."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _table_exists(connection: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(row is not None and int(row[0]) > 0)


class UniverseWorkspaceReader(WorkspaceRepository):
    """The shared owner of workspace universe reads."""

    def _workspace_load_universe_manifest(self, manifest_id: str) -> UniverseManifest:
        """Restore a frozen manifest from workspace-owned durable membership rows."""
        connection = self._connect(read_only=True)
        try:
            # Obligations arrived after workspaces existed: a manifest stored
            # before the column carries none, and a reader must not need the
            # writer's upgrade to say so.
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info('universe_manifest')").fetchall()
            }
            obligations_column = (
                "u.qualification_obligations_json"
                if "qualification_obligations_json" in columns
                else "NULL"
            )
            membership_column = "u.membership_sha256" if "membership_sha256" in columns else "NULL"
            manifest_row = connection.execute(
                f"""
                SELECT u.manifest_id, u.as_of_date, u.revision_sha256,
                       u.universe_membership_basis, u.is_point_in_time_historical,
                       p.market_profile_id, p.display_name, p.market, p.currency, p.calendar_id,
                       p.provider, p.daily_price_basis, p.data_validity_class,
                       u.membership_fingerprint, u.qualification_policy_hash,
                       u.universe_policy_type, u.universe_components_json,
                       u.survivorship_bias_warning, u.research_use_class,
                       {obligations_column}, {membership_column}, u.listing_count
                FROM universe_manifest u
                JOIN market_profile p ON p.market_profile_id = u.market_profile_id
                WHERE u.manifest_id = ?
                """,
                [manifest_id],
            ).fetchone()
            if manifest_row is None:
                raise ValueError("workspace manifest does not exist")
            listing_rows = connection.execute(
                """
                SELECT l.listing_id, l.display_symbol, l.mic, l.issuer_external_id,
                       (
                           SELECT psm.provider_symbol FROM provider_symbol_mapping psm
                           WHERE psm.listing_id = l.listing_id AND psm.provider = ?
                           ORDER BY psm.effective_from DESC LIMIT 1
                       ) AS provider_symbol
                FROM universe_manifest_listing uml
                JOIN listing l ON l.listing_id = uml.listing_id
                WHERE uml.manifest_id = ?
                ORDER BY l.display_symbol
                """,
                [str(manifest_row[10]), manifest_id],
            ).fetchall()
        finally:
            connection.close()
        if not listing_rows or any(row[4] is None for row in listing_rows):
            raise ValueError("workspace manifest membership or provider mapping is incomplete")
        # Its revision names its members, but the table keeps neither the candidate manifest an
        # acquisition revision binds nor a fixed copy of the profile (a later manifest of the
        # profile updates it), so the read proves the members it assembled against the digest
        # sealed at registration, or, on a manifest written before it, against the member count
        # it stored (EV2).
        members = [str(row[0]) for row in listing_rows]
        sealed_members, sealed_count = manifest_row[20], manifest_row[21]
        proved = (
            _membership_digest(members) == sealed_members
            if sealed_members is not None
            else len(members) == sealed_count
        )
        if not proved:
            raise ValueError("market_data_ops.universe_manifest_membership_changed")
        profile = self._profile_from_row(manifest_row)
        return UniverseManifest(
            manifest_id=str(manifest_row[0]),
            profile=profile,
            listings=tuple(
                ManifestListing(
                    listing_id=str(row[0]),
                    symbol=str(row[1]),
                    mic=str(row[2]),
                    provider_symbol=str(row[4]),
                    issuer_external_id=str(row[3]) if row[3] is not None else None,
                )
                for row in listing_rows
            ),
            revision_sha256=str(manifest_row[2]),
            universe_membership_basis=str(manifest_row[3]),
            is_point_in_time_historical=bool(manifest_row[4]),
            membership_fingerprint=(
                str(manifest_row[13]) if manifest_row[13] is not None else None
            ),
            qualification_policy_hash=(
                str(manifest_row[14]) if manifest_row[14] is not None else None
            ),
            universe_policy_type=str(manifest_row[15]),
            universe_components=tuple(json.loads(str(manifest_row[16]))),
            survivorship_bias_warning=bool(manifest_row[17]),
            research_use_class=str(manifest_row[18]),
            qualification_obligations=(
                tuple(str(item) for item in json.loads(str(manifest_row[19])))
                if manifest_row[19] is not None
                else ()
            ),
        )

    def _workspace_universe_bootstrap(
        self, market_profile_id: str
    ) -> UniverseBootstrapRecord | None:
        """Return the rollback workspace's frozen bootstrap boundary."""
        if not self.path.exists():
            return None
        connection = self._connect(read_only=True)
        try:
            if not _table_exists(connection, "universe_bootstrap"):
                return None
            row = connection.execute(
                """
                SELECT t0_session, history_start, cohort_hash, cohort_listing_ids_json,
                       manifest_revision, candidate_manifest_hash, qualification_policy_hash,
                       feature_input_policy_hash, source_observed_at, admitted_at,
                       panel_snapshot_hash, derivation, initialization_assumption, record_hash
                FROM universe_bootstrap WHERE market_profile_id = ?
                """,
                [market_profile_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return UniverseBootstrapRecord(
            market_profile_id=market_profile_id,
            t0_session=row[0],
            history_start=row[1],
            cohort_listing_ids=tuple(str(item) for item in json.loads(str(row[3]))),
            cohort_hash=str(row[2]),
            manifest_revision=str(row[4]),
            candidate_manifest_hash=str(row[5]),
            qualification_policy_hash=str(row[6]),
            feature_input_policy_hash=str(row[7]),
            source_observed_at=_utc_aware(row[8]),
            admitted_at=_utc_aware(row[9]),
            panel_snapshot_hash=str(row[10]),
            derivation=row[11],
            initialization_assumption=str(row[12]),
            record_hash=str(row[13]),
        )

    def _workspace_membership_events(self, market_profile_id: str) -> tuple[MembershipEvent, ...]:
        """Return the rollback workspace's membership journal in sequence order."""
        if not self.path.exists():
            return ()
        connection = self._connect(read_only=True)
        try:
            if not _table_exists(connection, "universe_membership_event"):
                return ()
            rows = connection.execute(
                """
                SELECT sequence, listing_id, event_kind, effective_session, observed_at,
                       decided_at, authority, reference_hash, manifest_revision, event_hash
                FROM universe_membership_event
                WHERE market_profile_id = ?
                ORDER BY sequence
                """,
                [market_profile_id],
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            MembershipEvent(
                market_profile_id=market_profile_id,
                sequence=int(row[0]),
                listing_id=str(row[1]),
                kind=row[2],
                effective_session=row[3],
                observed_at=_utc_aware(row[4]),
                decided_at=_utc_aware(row[5]),
                authority=str(row[6]),
                reference_hash=str(row[7]),
                manifest_revision=str(row[8]),
                event_hash=str(row[9]),
            )
            for row in rows
        )

    @staticmethod
    def _profile_from_row(row: tuple[object, ...]) -> MarketProfile:
        return MarketProfile(
            market_profile_id=str(row[5]),
            display_name=str(row[6]),
            market=str(row[7]),
            currency=str(row[8]),
            calendar_id=str(row[9]),
            provider=str(row[10]),
            daily_price_basis=str(row[11]),
            manifest_as_of=row[1],
            data_validity_class=str(row[12]),
        )
