"""Operational DuckDB projection for Panel factor/session availability.

Availability is keyed by the identity of the session's cross-section, the
catalog and the policy: what the numbers were computed over. A membership
change at one session therefore leaves every earlier session's rows in
place, and a build whose manifest changed for governance reasons alone
finds the rows it needs under the key it computes. The binding-keyed table
Panels recorded before this rule stays as written and is read only through
their recorded lineage.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime

import duckdb
import pyarrow as pa

from alphalattice.control.workspace_runtime.reader_threads import install_blake3_sql


class PanelAvailabilityRepository:
    """Persist availability only; Panel values remain immutable artifacts."""

    def upsert(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        cross_section_by_session: Mapping[str, str],
        catalog_hash: str,
        policy_hash: str,
        availability: Sequence[Mapping[str, object]],
        materialization_receipt_hash: str,
        observed_at: datetime,
    ) -> None:
        """Upsert availability by cross-section, catalog, policy, session, and factor.

        Record changed rows as revisions while leaving immutable Panel values
        in their artifact store.
        """
        if not availability:
            return
        stage_name = "panel_availability_input_stage"
        try:
            connection.register(
                stage_name,
                _availability_stage(
                    availability,
                    cross_section_by_session=cross_section_by_session,
                    materialization_receipt_hash=materialization_receipt_hash,
                ),
            )
            _record(
                connection,
                stage=stage_name,
                catalog_hash=catalog_hash,
                policy_hash=policy_hash,
                observed=_utc_naive(observed_at),
            )
        finally:
            with suppress(Exception):
                connection.unregister(stage_name)

    def carry(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        cross_section_by_session: Mapping[str, str],
        from_catalog_hash: str,
        to_catalog_hash: str,
        policy_hash: str,
        factor_ids: Sequence[str],
        observed_at: datetime,
    ) -> int:
        """Record a catalog's cells under a catalog that only adds columns to it.

        Each cell keeps what its batch measured, its binding and its receipt:
        that batch computed it, and the new catalog computes it alike. Under
        the new catalog its availability hash binds that catalog, recorded by
        the same step as an upsert's. A cell the new catalog already holds is
        kept.

        Returns:
            The number of cells carried.
        """
        if not cross_section_by_session or not factor_ids:
            return 0
        wanted = "panel_availability_carry_sessions"
        stage = "panel_availability_carry_stage"
        connection.register(
            wanted,
            pa.table(
                {
                    "session_date": pa.array(list(cross_section_by_session), pa.string()),
                    "cross_section_identity": pa.array(
                        list(cross_section_by_session.values()), pa.string()
                    ),
                }
            ),
        )
        try:
            # The columns and order of an upsert's normalized stage.
            connection.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE {stage} AS
                SELECT source.cross_section_identity,
                       CAST(source.session_date AS VARCHAR) AS session_date,
                       source.factor_id, source.universe_size, source.computed_count,
                       source.coverage, source.sector_counts_json, source.winsor_lower,
                       source.winsor_upper, source.residual_median, source.residual_mad,
                       source.status, source.reason, source.panel_binding_hash,
                       source.small_sector_warning, source.small_sector_names_json,
                       source.materialization_receipt_hash
                FROM panel_cross_section_availability AS source
                JOIN {wanted} AS wanted
                  ON CAST(source.session_date AS VARCHAR) = wanted.session_date
                 AND source.cross_section_identity = wanted.cross_section_identity
                WHERE source.catalog_hash = ? AND source.policy_hash = ?
                  AND list_contains(?::VARCHAR[], source.factor_id)
                  AND NOT EXISTS (
                      SELECT 1 FROM panel_cross_section_availability AS held
                      WHERE held.cross_section_identity = source.cross_section_identity
                        AND held.catalog_hash = ? AND held.policy_hash = source.policy_hash
                        AND held.session_date = source.session_date
                        AND held.factor_id = source.factor_id
                  )
                """,
                [from_catalog_hash, policy_hash, list(factor_ids), to_catalog_hash],
            )
            counted = connection.execute(f"SELECT count(*) FROM {stage}").fetchone()
            carried = int(counted[0]) if counted is not None else 0
            if carried:
                _record(
                    connection,
                    stage=stage,
                    catalog_hash=to_catalog_hash,
                    policy_hash=policy_hash,
                    observed=_utc_naive(observed_at),
                )
        finally:
            with suppress(Exception):
                connection.unregister(wanted)
            with suppress(Exception):
                connection.execute(f"DROP TABLE IF EXISTS {stage}")
        return carried


def _record(
    connection: duckdb.DuckDBPyConnection,
    *,
    stage: str,
    catalog_hash: str,
    policy_hash: str,
    observed: datetime,
) -> None:
    """Hash a staged set of cells under a catalog and record them, revisions first."""
    hashed_stage = "panel_availability_hashed_stage"
    install_blake3_sql(connection)
    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {hashed_stage} AS
            SELECT
                ?::VARCHAR AS catalog_hash,
                ?::VARCHAR AS policy_hash,
                *,
                system.main.alphalattice_blake3(to_json(struct_pack(
                    cross_section := cross_section_identity,
                    catalog_hash := ?::VARCHAR,
                    policy_hash := ?::VARCHAR,
                    session := session_date,
                    factor := factor_id,
                    universe_size := universe_size,
                    computed_count := computed_count,
                    coverage := coverage,
                    sector_counts_json := sector_counts_json,
                    winsor_lower := winsor_lower,
                    winsor_upper := winsor_upper,
                    residual_median := residual_median,
                    residual_mad := residual_mad,
                    status := status,
                    reason := reason,
                    panel_binding_hash := panel_binding_hash,
                    small_sector_warning := small_sector_warning,
                    small_sector_names_json := small_sector_names_json
                ))) AS availability_hash
            FROM {stage}
            """,
            [catalog_hash, policy_hash, catalog_hash, policy_hash],
        )
        connection.execute(
            f"""
            INSERT INTO panel_cross_section_availability_revision
            SELECT
                system.main.alphalattice_blake3(concat_ws('|', stage.cross_section_identity,
                    stage.catalog_hash, stage.policy_hash,
                    CAST(stage.session_date AS VARCHAR), stage.factor_id,
                    current.availability_hash, stage.availability_hash)),
                stage.cross_section_identity, stage.catalog_hash,
                stage.policy_hash, stage.session_date, stage.factor_id,
                current.availability_hash, stage.availability_hash,
                stage.materialization_receipt_hash, ?::TIMESTAMP
            FROM {hashed_stage} AS stage
            JOIN panel_cross_section_availability AS current
              ON current.cross_section_identity = stage.cross_section_identity
             AND current.catalog_hash = stage.catalog_hash
             AND current.policy_hash = stage.policy_hash
             AND current.session_date = stage.session_date
             AND current.factor_id = stage.factor_id
            WHERE current.availability_hash <> stage.availability_hash
            ON CONFLICT DO NOTHING
            """,
            [observed],
        )
        connection.execute(
            f"""
            INSERT INTO panel_cross_section_availability (
                cross_section_identity, catalog_hash, policy_hash,
                session_date, factor_id, universe_size, computed_count, coverage,
                sector_counts_json, winsor_lower, winsor_upper, residual_median,
                residual_mad, status, reason, panel_binding_hash,
                small_sector_warning, small_sector_names_json, availability_hash,
                materialization_receipt_hash
            ) SELECT
                cross_section_identity, catalog_hash, policy_hash,
                session_date, factor_id, universe_size, computed_count, coverage,
                sector_counts_json, winsor_lower, winsor_upper, residual_median,
                residual_mad, status, reason, panel_binding_hash,
                small_sector_warning, small_sector_names_json, availability_hash,
                materialization_receipt_hash
            FROM {hashed_stage}
            ON CONFLICT (
                cross_section_identity, catalog_hash, policy_hash, session_date, factor_id
            ) DO UPDATE SET
                universe_size = excluded.universe_size,
                computed_count = excluded.computed_count,
                coverage = excluded.coverage,
                sector_counts_json = excluded.sector_counts_json,
                winsor_lower = excluded.winsor_lower,
                winsor_upper = excluded.winsor_upper,
                residual_median = excluded.residual_median,
                residual_mad = excluded.residual_mad,
                status = excluded.status,
                reason = excluded.reason,
                panel_binding_hash = excluded.panel_binding_hash,
                small_sector_warning = excluded.small_sector_warning,
                small_sector_names_json = excluded.small_sector_names_json,
                availability_hash = excluded.availability_hash,
                materialization_receipt_hash = excluded.materialization_receipt_hash
            """
        )
    finally:
        with suppress(Exception):
            connection.execute(f"DROP TABLE IF EXISTS {hashed_stage}")


_STAGE_COLUMNS = (
    "cross_section_identity",
    "session_date",
    "factor_id",
    "universe_size",
    "computed_count",
    "coverage",
    "sector_counts_json",
    "winsor_lower",
    "winsor_upper",
    "residual_median",
    "residual_mad",
    "status",
    "reason",
    "panel_binding_hash",
    "small_sector_warning",
    "small_sector_names_json",
    "materialization_receipt_hash",
)
"""The normalized stage's columns, in the order every upsert stages them."""


def _availability_stage(
    availability: Sequence[Mapping[str, object]],
    *,
    cross_section_by_session: Mapping[str, str],
    materialization_receipt_hash: str,
) -> pa.Table:
    """The cells normalized into the stage's columns, one pass and one Arrow array per column:
    each column holds the values, and takes the type, a table built row by row from the same
    normalized cells would. A session's sector counts and small-sector names repeat across its
    factors, so each distinct one is encoded once."""
    columns: dict[str, list[object]] = {name: [] for name in _STAGE_COLUMNS}
    encoded_counts: dict[tuple[tuple[object, object], ...], str] = {}
    encoded_names: dict[tuple[object, ...], str] = {}
    for item in availability:
        binding_hash = str(item.get("panel_binding_hash") or item.get("panel_hash") or "")
        if not binding_hash:
            raise ValueError("Panel availability is missing its binding hash")
        raw_small_names = item.get("small_sector_names", ())
        if not isinstance(raw_small_names, Sequence) or isinstance(raw_small_names, (str, bytes)):
            raise ValueError("Panel availability small-sector names are invalid")
        session = str(item["session_date"])
        try:
            cross_section = cross_section_by_session[session]
        except KeyError as exc:
            raise ValueError(f"Panel availability session {session} has no cross-section") from exc
        for name, value in (
            ("cross_section_identity", cross_section),
            ("session_date", session),
            ("factor_id", str(item["factor_id"])),
            ("universe_size", int(str(item["universe_size"]))),
            ("computed_count", int(str(item["computed_count"]))),
            ("coverage", float(str(item["coverage"]))),
            ("sector_counts_json", _encoded_counts(item["sector_counts"], encoded_counts)),
            ("winsor_lower", item.get("winsor_lower")),
            ("winsor_upper", item.get("winsor_upper")),
            ("residual_median", item.get("residual_median")),
            ("residual_mad", item.get("residual_mad")),
            ("status", str(item["status"])),
            ("reason", item.get("reason")),
            ("panel_binding_hash", binding_hash),
            ("small_sector_warning", bool(item.get("small_sector_warning"))),
            ("small_sector_names_json", _encoded_names(raw_small_names, encoded_names)),
            ("materialization_receipt_hash", materialization_receipt_hash),
        ):
            columns[name].append(value)
    return pa.table({name: pa.array(values) for name, values in columns.items()})


def _encoded_counts(counts: object, seen: dict[tuple[tuple[object, object], ...], str]) -> str:
    """The sector counts as their stage JSON, keys sorted; equal counts encode alike, so each
    distinct mapping is encoded once."""
    key = tuple(counts.items()) if isinstance(counts, Mapping) else None
    if key is None:
        return json.dumps(counts, sort_keys=True, separators=(",", ":"))
    encoded = seen.get(key)
    if encoded is None:
        encoded = seen[key] = json.dumps(counts, sort_keys=True, separators=(",", ":"))
    return encoded


def _encoded_names(names: Sequence[object], seen: dict[tuple[object, ...], str]) -> str:
    """The small-sector names as their stage JSON, sorted, each distinct list encoded once."""
    key = tuple(names)
    encoded = seen.get(key)
    if encoded is None:
        encoded = seen[key] = json.dumps(
            tuple(sorted(str(value) for value in names)), separators=(",", ":")
        )
    return encoded


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Panel availability timestamp must be timezone-aware")
    return value.astimezone(UTC).replace(tzinfo=None)


__all__ = ["PanelAvailabilityRepository"]
