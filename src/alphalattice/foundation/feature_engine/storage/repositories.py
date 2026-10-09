"""Feature and panel repositories over the shared workspace database."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alphalattice.control.workspace_runtime.database import (
    WorkspaceDatabase,
    WorkspaceRepository,
    checkpoint_workspace_database,
)
from alphalattice.control.workspace_runtime.storage.readiness import (
    WorkspaceReadinessRepository,
)
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.catalog.layer import FeatureCatalogLayer
from alphalattice.foundation.feature_engine.contracts import (
    PanelCrossSectionRange,
    PanelSourceExclusion,
)
from alphalattice.foundation.feature_engine.inputs.contracts import (
    ListingQuarantine,
    QualificationDomain,
    quarantine_qualification_domain,
)
from alphalattice.foundation.feature_engine.panels.materialization_identity import (
    feature_materialization_coverage,
    feature_materialization_receipt_hash,
    feature_row_content_hash,
    feature_row_is_numerically_identical,
)
from alphalattice.foundation.feature_engine.publication.current_storage import (
    FeatureStorageLayout,
    assert_current_catalog,
    assert_rows_carry_installed_catalog,
    canonical_cutoff_set,
    ensure_feature_current_schema,
    feature_storage_layout,
    runtime_view_hash,
)
from alphalattice.foundation.feature_engine.publication.persistence import admitted_writes
from alphalattice.foundation.feature_engine.storage.contracts import (
    BindableSectorEvidence,
    FeatureIneligibilityRun,
    FeatureMaterializationWrite,
    FeatureRowIdentity,
    FeatureSourceWindow,
    PanelContentIdentity,
    SectorReferenceState,
)
from alphalattice.foundation.market_data_ops.publication.projection import (
    ADJUSTED_CLOSE_DIAGNOSTIC_POLICY_HASH,
    DailyPriceBasis,
    FeatureAdmissionBlocked,
    action_set_hash,
    project_as_traded_series,
    project_research_series,
    provider_adjusted_close_evidence_hash,
    provider_adjusted_ratio_diagnostics,
)
from alphalattice.foundation.market_data_ops.returns.sector_revision_identity import (
    sector_revision_hash,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    FailureEvidence,
    ProviderAdjustedClosePoint,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.sources.sector_forward import sector_effective_session
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    ActionAuditReceipt,
    FeaturePersistenceSqlProfiler,
    FeaturePersistenceTimingSink,
    MarketDataRepository,
    _canonical_hash,
    _selected_providers,
    _source_prefix_arrow_hash,
    _utc_aware,
    _utc_naive,
    raw_bars_from_table,
)
from alphalattice.kernel.quant.sector_history import SectorHistory, SectorReclassification


class _FeatureInputAdmissionLike(Protocol):
    """Storage-facing shape of one Gateway admission."""

    evidence_scope: str


@runtime_checkable
class _FeatureInputGatewayResultLike(Protocol):
    """Storage-facing shape of one admitted Gateway result."""

    research_manifest: UniverseManifest | None
    quarantines: tuple[ListingQuarantine, ...]
    deferred: object | None
    agent_cases: tuple[object, ...]
    admission: _FeatureInputAdmissionLike | None
    failure_reasons: tuple[str, ...]


@runtime_checkable
class _QuarantineContinuationLike(Protocol):
    """Storage-facing shape of one standing-quarantine continuation."""

    continuation_hash: str
    continued_from_quarantine_hash: str
    quarantine: ListingQuarantine

    def document(self) -> dict[str, object]: ...


_INELIGIBILITY_RUN_COLUMNS = (
    "run_id",
    "listing_id",
    "catalog_hash",
    "factor_id",
    "reason",
    "first_session",
    "last_session",
    "first_observation_count",
    "observation_cap",
    "materialization_receipt_hash",
    "updated_at",
)
_INELIGIBILITY_FACT_COLUMNS = (
    "listing_id",
    "session_date",
    "catalog_hash",
    "factor_id",
    "reason",
    "observation_count",
    "materialization_receipt_hash",
    "updated_at",
)
"""One diagnostic fact per session and factor, as ``_compress_ineligibility_rows`` takes them."""


def _staged_rows(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> pa.Table:
    """The Arrow table of non-empty ``rows``, each a value per column in ``columns`` order.

    Built by column: what `pa.Table.from_pylist` builds from a mapping per row -- it
    gathers each column's values and converts them with `pa.array` -- without the
    mappings (1.19M rows over a first build's Feature rows).
    """
    by_column = zip(*rows, strict=True)
    return pa.table(
        {name: pa.array(values) for name, values in zip(columns, by_column, strict=True)}
    )


def _canonical_stage(
    columns: Sequence[str],
    cells: Sequence[tuple[date, str, str]],
    factor_values: np.ndarray,
    *,
    constants: Mapping[str, object],
    verification: tuple[str | None, ...],
) -> pa.Table:
    """The Arrow table of canonical Feature rows, in ``columns`` order.

    One row per cell of ``cells`` -- its session, cutoff column value and row hash --
    with ``factor_values`` its factor cells (a row per cell, a column per factor, the
    factors ending ``columns``) and ``constants`` the columns every row shares. Typed
    as ``_staged_rows`` types a row list's values: strings, dates, a microsecond
    timestamp and doubles, NaN kept. ``verification`` holds, when given, the
    ``source_verification_receipt_hash`` every row carries.
    """
    count = len(cells)
    sessions, cutoffs, row_hashes = (list(values) for values in zip(*cells, strict=True))
    by_cell = {
        "session_date": pa.array(sessions, pa.date32()),
        "cutoff_set_hash": pa.array(cutoffs, pa.string()),
        "input_cutoffs_json": pa.array(cutoffs, pa.string()),
        "row_hash": pa.array(row_hashes, pa.string()),
    }
    payload: dict[str, pa.Array] = {}
    for name in columns[: len(columns) - factor_values.shape[1]]:
        if name in by_cell:
            payload[name] = by_cell[name]
        elif name == "updated_at":
            payload[name] = pa.array([constants[name]] * count, pa.timestamp("us"))
        else:
            payload[name] = pa.array([constants[name]] * count, pa.string())
    factors = columns[len(columns) - factor_values.shape[1] :]
    for index, name in enumerate(factors):
        payload[name] = pa.array(factor_values[:, index], pa.float64())
    if verification:
        payload["source_verification_receipt_hash"] = pa.array(
            [verification[0]] * count, pa.string()
        )
    return pa.table(payload)


def _ineligibility_facts(
    ineligibility: Sequence[Mapping[str, object]] | pd.DataFrame,
    *,
    range_start: date,
    range_end: date,
    rebuilt_factor_ids: Sequence[str],
) -> tuple[list[date], list[str], list[str], list[int]]:
    """Each diagnostic fact's session, factor, reason and observation count, checked.

    A fact's session is its value's ISO date; it must lie in the rebuilt range, and its
    factor in the rebuilt scope. A frame is read column by column. A frame whose facts do
    not all pass, or whose values the columns cannot take as they are, and every sequence
    of mappings, go through the checks fact by fact, so the first failing fact raises what
    it always raised.
    """
    required = ("session_date", "factor_id", "reason", "observation_count")
    if isinstance(ineligibility, pd.DataFrame):
        if not set(required).issubset(ineligibility.columns):
            raise ValueError("feature ineligibility frame is missing required columns")
        columns = [ineligibility[name].tolist() for name in required]
        try:
            parsed = {value: date.fromisoformat(str(value)) for value in set(columns[0])}
            factors = [str(value) for value in columns[1]]
            in_range = not parsed or (
                range_start <= min(parsed.values()) and max(parsed.values()) <= range_end
            )
            if in_range and set(factors) <= set(rebuilt_factor_ids):
                return (
                    [parsed[value] for value in columns[0]],
                    factors,
                    [str(value) for value in columns[2]],
                    [int(value) for value in columns[3]],
                )
        except (TypeError, ValueError):
            pass
        items: Iterable[tuple[object, ...]] = zip(*columns, strict=True)
    else:
        items = (tuple(item[name] for name in required) for item in ineligibility)
    sessions: list[date] = []
    factors = []
    reasons: list[str] = []
    observations: list[int] = []
    for session_value, factor_value, reason_value, observation_value in items:
        session = date.fromisoformat(str(session_value))
        if not range_start <= session <= range_end:
            raise ValueError("feature ineligibility is outside rebuilt range")
        if str(factor_value) not in rebuilt_factor_ids:
            raise ValueError("feature ineligibility is outside rebuilt factor scope")
        sessions.append(session)
        factors.append(str(factor_value))
        reasons.append(str(reason_value))
        observations.append(int(cast(Any, observation_value)))
    return sessions, factors, reasons, observations


@dataclass
class _FeatureWriteBatch:
    """What a batch's listing writes share in their one transaction (V92).

    The table's shape and the installed catalog, read once; the cutoff sets an earlier write
    of the batch inserted, which a later one does not offer again; the receipts, which no write
    of the batch reads, inserted once before the commit; and, when every write is of another
    listing, the ineligibility runs, which only their own listing's write reads or replaces.
    """

    storage_layout: FeatureStorageLayout
    has_verification: bool
    offered_cutoff_sets: set[str]
    receipts: list[list[object]]
    runs: list[list[object]] | None


_RECEIPT_STAGE_COLUMNS = (
    "receipt_hash",
    "listing_id",
    "range_start",
    "range_end",
    "catalog_hash",
    "raw_input_hash",
    "action_set_hash",
    "market_reference_revision",
    "idempotency_key",
    "coverage_summary_json",
    "observed_at",
)
"""A deferred receipt's values, in the order a listing's write gathers them."""


def _first_by_identity(rows: Sequence[list[object]]) -> list[list[object]]:
    """Each row whose identity (its first value) no earlier row has, in order."""
    first: dict[object, list[object]] = {}
    for row in rows:
        first.setdefault(row[0], row)
    return list(first.values())


def _storage_shape(connection: duckdb.DuckDBPyConnection) -> tuple[FeatureStorageLayout, bool]:
    """The current table's layout and whether its rows carry a source verification receipt."""
    return feature_storage_layout(connection), "source_verification_receipt_hash" in {
        item[1]
        for item in connection.execute("PRAGMA table_info('feature_daily_current')").fetchall()
    }


@dataclass(frozen=True)
class FeatureSourceInputs:
    """One listing's stored source inputs, as the store reads them for its input block.

    What the block is projected from (``projected_feature_rows``) and the lineage it is recorded
    under: read by a build's writer and projected where the listing is computed, so the writer
    builds no object per bar (V92).
    """

    listing_id: str
    bars: pa.Table
    """Its raw bars over the block's sessions, as ``MarketDataRepository.raw_bar_table`` reads."""
    actions: tuple[CorporateActionEvent, ...]
    """Every recorded corporate action of the listing."""
    adjusted_close: pa.Table
    """The provider's adjusted closes over the bars' sessions (``session_date``,
    ``adjusted_close``), in session order."""
    daily_price_basis: DailyPriceBasis
    allow_missing_adjusted: bool
    as_traded: bool
    raw_hash: str
    action_hash: str


def projected_feature_rows(inputs: FeatureSourceInputs) -> list[dict[str, object]]:
    """One listing's feature input block, projected from its stored source inputs.

    Args:
        inputs: What ``FeatureStateRepository.feature_source_inputs`` read.

    Returns:
        One row per bar: the projected bar, the provider's adjusted close (NaN where an
        availability-aware read carries a missing one) and, for an as-traded read, the
        session's as-traded prices and re-basing indices.

    Raises:
        ValueError: The provider's adjusted series does not cover the bars' sessions.
    """
    bars = raw_bars_from_table(inputs.bars)
    scoped_actions = tuple(
        item
        for item in inputs.actions
        if bars[0].session_date <= item.effective_date <= bars[-1].session_date
    )
    projected = project_research_series(
        bars, scoped_actions, daily_price_basis=inputs.daily_price_basis
    )
    traded = (
        {
            item.session_date: item
            for item in project_as_traded_series(
                bars, inputs.actions, daily_price_basis=inputs.daily_price_basis
            )
        }
        if inputs.as_traded
        else {}
    )
    adjusted_by_session = {
        session: float(value)
        for session, value in zip(
            inputs.adjusted_close.column("session_date").to_pylist(),
            inputs.adjusted_close.column("adjusted_close").to_pylist(),
            strict=True,
        )
    }
    raw_sessions = {item.session_date for item in projected}
    if set(adjusted_by_session) - raw_sessions or (
        not inputs.allow_missing_adjusted and raw_sessions - set(adjusted_by_session)
    ):
        raise ValueError("provider adjusted series does not cover the feature input sessions")
    payload = []
    for item in projected:
        row = item.__dict__.copy()
        row["provider_adjusted_close"] = adjusted_by_session.get(item.session_date, math.nan)
        if inputs.as_traded:
            row.update(
                {
                    key: value
                    for key, value in traded[item.session_date].__dict__.items()
                    if key != "session_date"
                }
            )
        payload.append(row)
    return payload


class FeatureStateRepository(WorkspaceRepository):
    """Own Feature rows, eligibility, reference data, and sector state."""

    def __init__(
        self,
        workspace: Path | WorkspaceDatabase,
        *,
        market_data: MarketDataRepository | None = None,
        installed_catalog: FeatureCatalog | None = None,
    ) -> None:
        """Initialize Feature storage with its market repository and catalog."""
        super().__init__(workspace)
        self._market_data = market_data or MarketDataRepository(self.database)
        self._feature_factor_ids_by_catalog: dict[str, tuple[str, ...]] = {}
        self._installed_catalog = installed_catalog
        self._catalog_layer: FeatureCatalogLayer | None = None
        """The catalog revision this workspace's storage is shaped by.

        Defaults to the shipped catalog, so production is unchanged. It has to be
        composition state for the same reason ``FeatureFoundationService`` holds
        one: the materializer, the persistence factor axis and the panel
        publisher were all already installable, but the DuckDB current-feature
        schema was built from ``FeatureCatalog.load()`` regardless. A workspace
        could therefore compute an extension factor and then fail to persist it,
        because the table had no column of that name. Every writer has to agree
        about which factors exist, and storage is a writer.
        """

    def _resolve_installed_catalog(self) -> FeatureCatalog:
        """Resolve the shaping catalog, loading the shipped one when none was set.

        Deliberately private: the store's public surface is guarded against
        growth, and nothing outside this repository needs to ask it which catalog
        it was composed with -- the composer already knows.
        """
        if self._installed_catalog is None:
            self._installed_catalog = FeatureCatalog.load()
        return self._installed_catalog

    def _resolve_catalog_layer(self) -> FeatureCatalogLayer:
        """The parts whose rows compose the installed catalog's (V92), derived once."""
        if self._catalog_layer is None:
            self._catalog_layer = FeatureCatalogLayer.over(self._resolve_installed_catalog())
        return self._catalog_layer

    @staticmethod
    def _ensure_feature_candidate_schema(connection: duckdb.DuckDBPyConnection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_input_candidate_binding (
                market_profile_id VARCHAR PRIMARY KEY,
                candidate_manifest_id VARCHAR NOT NULL,
                candidate_manifest_revision VARCHAR NOT NULL,
                observed_at TIMESTAMP NOT NULL
            )
            """
        )

    def bind_feature_input_candidate(
        self, manifest: UniverseManifest, *, observed_at: datetime
    ) -> None:
        """Own the stable pre-quarantine candidate binding for Feature rechecks."""
        connection = self._connect()
        try:
            self._ensure_feature_candidate_schema(connection)
            connection.execute(
                """
                INSERT INTO feature_input_candidate_binding VALUES (?, ?, ?, ?)
                ON CONFLICT (market_profile_id) DO UPDATE SET
                    candidate_manifest_id = excluded.candidate_manifest_id,
                    candidate_manifest_revision = excluded.candidate_manifest_revision,
                    observed_at = excluded.observed_at
                """,
                [
                    manifest.profile.market_profile_id,
                    manifest.manifest_id,
                    manifest.revision_sha256,
                    _utc_naive(observed_at),
                ],
            )
        finally:
            connection.close()

    def feature_input_candidate(self, market_profile_id: str) -> UniverseManifest | None:
        """Load the Feature-owned candidate through the Market manifest reader."""
        if not self.path.exists():
            return None
        connection = self._connect(read_only=True)
        try:
            try:
                row = connection.execute(
                    """
                    SELECT candidate_manifest_id, candidate_manifest_revision
                    FROM feature_input_candidate_binding
                    WHERE market_profile_id = ?
                    """,
                    [market_profile_id],
                ).fetchone()
            except duckdb.CatalogException:
                return None
        finally:
            connection.close()
        if row is None:
            return None
        manifest = self._market_data.load_universe_manifest(str(row[0]))
        if manifest.revision_sha256 != str(row[1]):
            raise ValueError("feature-input candidate binding has drifted")
        return manifest

    def _ensure_feature_ineligibility_view(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Migrate the old row table once, then expose a diagnostic-only view."""
        object_row = connection.execute(
            """
            SELECT table_type FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'feature_ineligibility'
            """
        ).fetchone()
        if object_row is not None and str(object_row[0]) == "BASE TABLE":
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.execute(
                    "ALTER TABLE feature_ineligibility RENAME TO feature_ineligibility_legacy"
                )
                scopes = connection.execute(
                    """
                    SELECT DISTINCT listing_id, catalog_hash
                    FROM feature_ineligibility_legacy ORDER BY listing_id, catalog_hash
                    """
                ).fetchall()
                for listing_id, catalog_hash in scopes:
                    legacy = connection.execute(
                        """
                        SELECT listing_id, session_date, catalog_hash, factor_id, reason,
                               observation_count, materialization_receipt_hash, updated_at
                        FROM feature_ineligibility_legacy
                        WHERE listing_id = ? AND catalog_hash = ?
                        ORDER BY factor_id, session_date
                        """,
                        [listing_id, catalog_hash],
                    ).fetchall()
                    positions = {
                        row[0]: index
                        for index, row in enumerate(
                            connection.execute(
                                """
                                SELECT session_date FROM feature_daily_runtime
                                WHERE listing_id = ? AND catalog_hash = ? ORDER BY session_date
                                """,
                                [listing_id, catalog_hash],
                            ).fetchall()
                        )
                    }
                    runs = self._compress_ineligibility_rows(
                        {
                            name: [row[index] for row in legacy]
                            for index, name in enumerate(_INELIGIBILITY_FACT_COLUMNS)
                        },
                        positions,
                    )
                    if runs:
                        # The same staged-relation shape the live write path
                        # already uses below. Per row this migration paid an
                        # index probe for its conflict clause, and a workspace
                        # can carry tens of thousands of legacy runs.
                        legacy_stage = "feature_ineligibility_legacy_run_stage"
                        connection.register(
                            legacy_stage, _staged_rows(_INELIGIBILITY_RUN_COLUMNS, runs)
                        )
                        try:
                            connection.execute(
                                f"""
                                INSERT INTO feature_ineligibility_run
                                SELECT * FROM {legacy_stage}
                                ON CONFLICT (run_id) DO NOTHING
                                """
                            )
                        finally:
                            connection.unregister(legacy_stage)
                connection.execute("DROP TABLE feature_ineligibility_legacy")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        connection.execute(
            """
            CREATE OR REPLACE VIEW feature_ineligibility AS
            WITH expanded AS (
                SELECT
                    run.run_id,
                    run.listing_id,
                    feature.session_date,
                    run.catalog_hash,
                    run.factor_id,
                    run.reason,
                    run.first_observation_count,
                    run.observation_cap,
                    row_number() OVER (
                        PARTITION BY run.run_id ORDER BY feature.session_date
                    ) - 1 AS run_offset,
                    run.materialization_receipt_hash,
                    run.updated_at
                FROM feature_ineligibility_run AS run
                JOIN feature_daily_runtime AS feature
                  ON feature.listing_id = run.listing_id
                 AND feature.catalog_hash = run.catalog_hash
                 AND feature.session_date BETWEEN run.first_session AND run.last_session
            )
            SELECT
                listing_id,
                session_date,
                catalog_hash,
                factor_id,
                reason,
                least(first_observation_count + run_offset, observation_cap)::INTEGER
                    AS observation_count,
                materialization_receipt_hash,
                updated_at
            FROM expanded
            """
        )

    @staticmethod
    def _compress_ineligibility_rows(
        facts: Mapping[str, Sequence[object]],
        session_positions: Mapping[date, int],
    ) -> list[list[object]]:
        """Losslessly encode contiguous diagnostic facts as bounded runs.

        ``facts`` holds each fact's value in every column of
        ``_INELIGIBILITY_FACT_COLUMNS``, one fact per position, observation
        counts as integers. The facts are grouped by factor (as text), then
        session, ties in the order given; a run continues while factor, reason,
        receipt and time stay the same, the session is the next feature session
        and the observation count steps as the run's second fact stepped (0 or
        1). Held by column, not as a mapping per fact: a first build hands over
        about a million facts, and the mappings, their sort and their lookups
        cost its single writer about 4 s.
        """
        sessions = facts["session_date"]
        factors = facts["factor_id"]
        reasons = facts["reason"]
        receipts = facts["materialization_receipt_hash"]
        updated = facts["updated_at"]
        observations = [int(value) for value in facts["observation_count"]]
        positions = [
            session_positions.get(session) if isinstance(session, date) else None
            for session in sessions
        ]
        if None in positions:
            raise ValueError("ineligibility row has no feature-session position")
        factor_ranks = {name: rank for rank, name in enumerate(sorted({str(v) for v in factors}))}
        order: list[int] = np.lexsort(
            (
                np.fromiter(
                    (cast(date, session).toordinal() for session in sessions),
                    dtype=np.int64,
                    count=len(sessions),
                ),
                np.fromiter(
                    (factor_ranks[str(value)] for value in factors),
                    dtype=np.int64,
                    count=len(factors),
                ),
            )
        ).tolist()
        # (first fact, last fact, observation cap) of each run, in sorted order.
        groups: list[tuple[int, int, int]] = []
        first = previous = -1
        cap = 0
        observation_step: int | None = None
        for index in order:
            position = cast(int, positions[index])
            observation = observations[index]
            if previous < 0:
                first, cap = index, observation
                previous = index
                continue
            next_step = observation - observations[previous]
            if (
                factors[index] == factors[previous]
                and reasons[index] == reasons[previous]
                and receipts[index] == receipts[previous]
                and updated[index] == updated[previous]
                and position == cast(int, positions[previous]) + 1
                and next_step in (0, 1)
                and (observation_step is None or next_step == observation_step)
            ):
                if observation_step is None:
                    observation_step = next_step
                cap = max(cap, observation)
            else:
                groups.append((first, previous, cap))
                first, cap = index, observation
                observation_step = None
            previous = index
        if previous >= 0:
            groups.append((first, previous, cap))
        encoded: list[list[object]] = []
        for first, last, cap in groups:
            payload = {
                "listing_id": facts["listing_id"][first],
                "catalog_hash": facts["catalog_hash"][first],
                "factor_id": factors[first],
                "reason": reasons[first],
                "first_session": sessions[first],
                "last_session": sessions[last],
                "first_observation_count": observations[first],
                "observation_cap": cap,
                "materialization_receipt_hash": receipts[first],
                "updated_at": updated[first],
            }
            encoded.append([_canonical_hash(payload), *payload.values()])
        return encoded

    def assert_installed_catalog_rows(self, *, catalog_hash: str) -> None:
        """Refuse live rows whose computing catalog is not the installed one, or its part."""
        layer = self._resolve_catalog_layer()
        parts = (
            layer.part_hashes
            if layer.layered and catalog_hash == layer.catalog.binding.catalog_hash
            else ()
        )
        connection = self._connect(read_only=True)
        try:
            assert_rows_carry_installed_catalog(connection, catalog_hash=catalog_hash, parts=parts)
        finally:
            connection.close()

    def retire_rows_outside_layer(self, *, _connection: duckdb.DuckDBPyConnection) -> int:
        """Delete the rows and runs no part of the installed catalog computed.

        Rows of two catalogs share no key, so after a catalog rotation the
        earlier catalog's rows stay beside the new ones until the build has
        written every part; then they answer for nothing, and would refuse every
        publication (``assert_installed_catalog_rows``). That holds as well for
        a listing an earlier manifest held and a manifest transition dropped
        (V398), and for the column of a factor a person deactivated (V92).

        Returns:
            The number of rows deleted.
        """
        admitted = list(self._resolve_catalog_layer().part_hashes)
        counted = _connection.execute(
            "SELECT count(*) FROM feature_daily_current "
            "WHERE NOT list_contains(?::VARCHAR[], catalog_hash)",
            [admitted],
        ).fetchone()
        retired = int(counted[0]) if counted is not None else 0
        if retired:
            for table in ("feature_daily_current", "feature_ineligibility_run"):
                _connection.execute(
                    f"DELETE FROM {table} WHERE NOT list_contains(?::VARCHAR[], catalog_hash)",
                    [admitted],
                )
            self._clear_feature_year_seals(_connection)
        return retired

    def discard_part_rows(
        self, catalog_hash: str, *, _connection: duckdb.DuckDBPyConnection
    ) -> int:
        """Delete the rows and runs an interrupted first load wrote for a column part (V92).

        Its genesis head attests no row, so the part goes back to empty and its build writes it
        again. Only a column part of the installed catalog's layer is discarded, never the base.

        Returns:
            The number of rows deleted.
        """
        columns = {part.binding.catalog_hash for part in self._resolve_catalog_layer().columns}
        if catalog_hash not in columns:
            raise ValueError("feature.catalog_outside_layer")
        _connection.execute("BEGIN TRANSACTION")
        try:
            counted = _connection.execute(
                "SELECT count(*) FROM feature_daily_current WHERE catalog_hash = ?", [catalog_hash]
            ).fetchone()
            for table in ("feature_daily_current", "feature_ineligibility_run"):
                _connection.execute(f"DELETE FROM {table} WHERE catalog_hash = ?", [catalog_hash])
            self._clear_feature_year_seals(_connection)
            _connection.execute("COMMIT")
        except BaseException:
            _connection.execute("ROLLBACK")
            raise
        return int(counted[0]) if counted is not None else 0

    @contextmanager
    def feature_build_connection(self):
        """Reuse one writer-compatible connection across a deterministic build."""
        connection = self._connect()
        try:
            yield connection
            checkpoint_workspace_database(connection)
        finally:
            connection.close()

    def ensure_current_storage(self) -> None:
        """Install the active catalog-shaped Feature current schema."""
        connection = self._connect()
        try:
            self._ensure_feature_current_storage(connection)
            self._ensure_feature_ineligibility_view(connection)
        finally:
            connection.close()

    def _feature_factor_ids(self, catalog_hash: str) -> tuple[str, ...]:
        cached = self._feature_factor_ids_by_catalog.get(catalog_hash)
        if cached is not None:
            return cached
        layer = self._resolve_catalog_layer()
        if catalog_hash == layer.catalog.binding.catalog_hash:
            factor_ids = tuple(layer.catalog.factor_ids)
        elif catalog_hash in layer.part_hashes:
            factor_ids = tuple(layer.part(catalog_hash).factor_ids)
        else:
            raise ValueError("feature catalog hash does not match the active binding")
        self._feature_factor_ids_by_catalog[catalog_hash] = factor_ids
        return factor_ids

    def _ensure_feature_current_storage(self, connection: duckdb.DuckDBPyConnection) -> None:
        layer = self._resolve_catalog_layer()
        ensure_feature_current_schema(
            connection,
            catalog_hash=layer.catalog.binding.catalog_hash,
            factor_ids=layer.catalog.factor_ids,
            activated_at=_utc_naive(datetime.now(UTC)),
            parts=layer.storage_parts(),
        )

    def projected_feature_frame(
        self,
        manifest: UniverseManifest,
        *,
        listing_id: str,
        through: date,
        start: date | None = None,
        allow_missing_adjusted: bool = False,
        as_traded: bool = False,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> tuple[list[dict[str, object]], str, str]:
        """Return one provider-first feature input block and its lineage.

        Legacy action-derived projection columns remain available to old
        snapshots.  The active desktop catalog consumes the separately stored
        provider adjusted close and never treats the local projection as its
        total-return authority. Availability-aware callers may carry missing
        adjusted observations as NaN; unmatched extra adjusted dates still
        refuse. The default stays strict and no raw values are substituted.
        ``as_traded`` adds each session's prices and share volume as they traded
        and the indices that re-base them, from every recorded split, later ones
        included (V345), for a formula reading a point-in-time leaf.

        The block is ``projected_feature_rows`` over ``feature_source_inputs``.
        """
        inputs = self.feature_source_inputs(
            manifest,
            listing_id=listing_id,
            through=through,
            start=start,
            allow_missing_adjusted=allow_missing_adjusted,
            as_traded=as_traded,
            _connection=_connection,
        )
        return projected_feature_rows(inputs), inputs.raw_hash, inputs.action_hash

    def feature_source_inputs(
        self,
        manifest: UniverseManifest,
        *,
        listing_id: str,
        through: date,
        start: date | None = None,
        allow_missing_adjusted: bool = False,
        as_traded: bool = False,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> FeatureSourceInputs:
        """One listing's stored source inputs and the lineage of the block they project to.

        The reads of ``projected_feature_frame``, by columns where a listing has a row per
        session: a build's writer reads them and the worker that computes the listing projects
        them (``projected_feature_rows``), so the writer builds no object per bar (V92).
        """
        self._market_data._assert_manifest_scope(manifest, (listing_id,))
        connection = _connection or self._connect(read_only=True)
        owns_connection = _connection is None
        try:
            bars = self._market_data._raw_bar_table(
                connection, listing_id, start=start, through=through
            )
            if not bars.num_rows:
                raise ValueError("listing has no raw bars for feature materialization")
            first_session = cast(date, bars.column("session_date")[0].as_py())
            provider = self._market_data._provider_for_listing(connection, listing_id)
            # What ``actions`` reads, under the provider read once for the listing's inputs.
            actions = self._market_data._current_actions(connection, listing_id, provider)
            raw_hash = self._market_data._raw_evidence_hash(
                connection,
                listing_id=listing_id,
                provider=provider,
                history_start=first_session,
                history_end=through,
            )
            adjusted_close = cast(
                pa.Table,
                connection.execute(
                    """
                    SELECT session_date, adjusted_close
                    FROM provider_adjusted_close_current
                    WHERE listing_id = ? AND provider = ? AND session_date BETWEEN ? AND ?
                    ORDER BY session_date
                    """,
                    [listing_id, provider, first_session, through],
                ).to_arrow_table(),
            )
        finally:
            if owns_connection:
                connection.close()
        return FeatureSourceInputs(
            listing_id=listing_id,
            bars=bars,
            actions=actions,
            adjusted_close=adjusted_close,
            daily_price_basis=manifest.profile.daily_price_basis,
            allow_missing_adjusted=allow_missing_adjusted,
            as_traded=as_traded,
            raw_hash=raw_hash,
            action_hash=action_set_hash(actions),
        )

    def feature_source_inputs_by_listing(
        self,
        starts: Mapping[str, tuple[UniverseManifest, date]],
        *,
        through: date,
        allow_missing_adjusted: bool = False,
        as_traded: bool = False,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> dict[str, FeatureSourceInputs]:
        """``feature_source_inputs`` of many listings, one read per table, each from its start.

        ``starts`` maps a listing to its manifest and first session. A listing with no bars from
        its start, or no provider mapping, is left out: its own read refuses it by name.
        """
        market = self._market_data
        for listing_id, (manifest, _start) in starts.items():
            market._assert_manifest_scope(manifest, (listing_id,))
        connection = _connection or self._connect(read_only=True)
        owns_connection = _connection is None
        try:
            tables = market._raw_bar_tables(
                connection,
                {listing: start for listing, (_m, start) in starts.items()},
                through=through,
            )
            providers = _selected_providers(
                connection, [listing for listing, table in tables.items() if table.num_rows]
            )
            held = sorted(
                listing for listing, provider in providers.items() if provider is not None
            )
            if not held:
                return {}
            firsts = {
                listing: cast(date, tables[listing].column("session_date")[0].as_py())
                for listing in held
            }
            actions = market.actions_by_listing(held, _connection=connection)
            raw_hashes = market._raw_evidence_hashes(
                connection,
                {
                    listing: (cast(str, providers[listing]), firsts[listing], through)
                    for listing in held
                },
            )
            adjusted = cast(
                pa.Table,
                connection.execute(
                    """
                    SELECT listing_id, provider, session_date, adjusted_close
                    FROM provider_adjusted_close_current
                    WHERE listing_id IN (SELECT unnest(?::VARCHAR[]))
                      AND session_date BETWEEN ? AND ?
                    ORDER BY listing_id, session_date
                    """,
                    [held, min(firsts.values()), through],
                ).to_arrow_table(),
            )
        finally:
            if owns_connection:
                connection.close()
        listings = adjusted.column("listing_id").to_pylist()
        selected = [
            index
            for index, (listing, provider, session) in enumerate(
                zip(
                    listings,
                    adjusted.column("provider").to_pylist(),
                    adjusted.column("session_date").to_pylist(),
                    strict=True,
                )
            )
            if provider == providers[listing] and session >= firsts[listing]
        ]
        kept = adjusted.take(pa.array(selected, type=pa.int64()))
        kept_listings = [listings[index] for index in selected]
        bounds: dict[str, tuple[int, int]] = {}
        for index, listing in enumerate(kept_listings):
            bounds[listing] = (bounds.get(listing, (index, index))[0], index + 1)
        out: dict[str, FeatureSourceInputs] = {}
        for listing in held:
            first, last = bounds.get(listing, (0, 0))
            manifest = starts[listing][0]
            out[listing] = FeatureSourceInputs(
                listing_id=listing,
                bars=tables[listing],
                actions=actions[listing],
                adjusted_close=kept.slice(first, last - first).select(
                    ["session_date", "adjusted_close"]
                ),
                daily_price_basis=manifest.profile.daily_price_basis,
                allow_missing_adjusted=allow_missing_adjusted,
                as_traded=as_traded,
                raw_hash=raw_hashes[listing],
                action_hash=action_set_hash(actions[listing]),
            )
        return out

    def upsert_feature_materialization(
        self,
        *,
        listing_id: str,
        catalog_hash: str,
        rows: Sequence[Mapping[str, object]] | pd.DataFrame,
        ineligibility: Sequence[Mapping[str, object]] | pd.DataFrame,
        raw_input_hash: str,
        action_set_hash_value: str,
        market_reference_revision: str,
        idempotency_key: str,
        revision_reason: str,
        observed_at: datetime,
        factor_ids: Sequence[str] | None = None,
        rows_are_canonical: bool = False,
        source_window: FeatureSourceWindow | None = None,
        row_identities: Sequence[FeatureRowIdentity] | None = None,
        existing_rows: Mapping[date, Mapping[str, object]] | None = None,
        timing_sink: FeaturePersistenceTimingSink | None = None,
        sql_profiler: FeaturePersistenceSqlProfiler | None = None,
        _connection: duckdb.DuckDBPyConnection | None = None,
        _manage_transaction: bool = True,
        _batch: _FeatureWriteBatch | None = None,
    ) -> str:
        """Atomically merge one listing/time feature block with no-op replay.

        The receipt is a safe operation record.  A current row only creates a
        revision when its canonical business content really differs; repeated
        feature builds therefore cannot grow the revision history.

        ``row_identities`` are the canonical rows' identities the closure
        coordinator derived from these same rows for this same write (one per
        row, in row order); given, they are consumed instead of derived again
        here. Without them every identity is derived from the rows.

        ``existing_rows`` are the runtime rows held at these rows' sessions, by
        session (each a mapping of the runtime row's columns), as the closure
        coordinator read them for its whole transition before this write, with
        nothing written in between; given, they are consumed instead of read
        again here, per listing. Without them they are read.
        """

        def record(stage: str, started: float) -> None:
            if timing_sink is not None:
                timing_sink(stage, time.perf_counter() - started)

        def profile(
            connection: duckdb.DuckDBPyConnection, stage: str
        ) -> AbstractContextManager[None]:
            if sql_profiler is None:
                return nullcontext()
            return sql_profiler(connection, stage)

        started = time.perf_counter()
        catalog_factor_ids = self._feature_factor_ids(catalog_hash)
        rebuilt_factor_ids = (
            tuple(sorted(set(str(item) for item in factor_ids)))
            if factor_ids is not None
            else catalog_factor_ids
        )
        if not rebuilt_factor_ids or not set(rebuilt_factor_ids).issubset(catalog_factor_ids):
            raise ValueError("feature materialization factor scope is outside the catalog")
        record("catalog_load", started)
        started = time.perf_counter()
        canonical_frame: pd.DataFrame | None = None
        normalized_rows: tuple[Mapping[str, object], ...] = ()
        if isinstance(rows, pd.DataFrame):
            required_columns = {
                "listing_id",
                "session_date",
                "input_cutoffs_json",
                *catalog_factor_ids,
            }
            if not required_columns.issubset(rows.columns):
                raise ValueError("feature row frame is missing required columns")
            if rows.empty:
                raise ValueError("feature materialization must contain at least one row")
            if not rows["listing_id"].eq(listing_id).all():
                raise ValueError("feature rows have another listing identity")
            if rows_are_canonical:
                canonical_frame = rows
            else:
                normalized_rows = tuple(rows.to_dict("records"))
        else:
            if rows_are_canonical:
                raise ValueError("canonical feature rows must use the columnar frame contract")
            normalized_rows = tuple(rows)
        if canonical_frame is None and not normalized_rows:
            raise ValueError("feature materialization must contain at least one row")

        def normalized_session(value: object) -> date:
            return value if isinstance(value, date) else date.fromisoformat(str(value))

        row_sessions = (
            tuple(normalized_session(value) for value in canonical_frame["session_date"])
            if canonical_frame is not None
            else tuple(normalized_session(row["session_date"]) for row in normalized_rows)
        )
        if canonical_frame is None and any(
            str(row.get("listing_id")) != listing_id for row in normalized_rows
        ):
            raise ValueError("feature rows have another listing identity")
        range_start = min(row_sessions)
        range_end = max(row_sessions)
        coverage = feature_materialization_coverage(
            rows=len(row_sessions),
            factor_count=len(catalog_factor_ids),
            rebuilt_factor_count=len(rebuilt_factor_ids),
            ineligible_count=len(ineligibility),
            source_window=source_window.to_payload() if source_window is not None else None,
        )
        receipt_hash = feature_materialization_receipt_hash(
            listing_id=listing_id,
            range_start=range_start,
            range_end=range_end,
            catalog_hash=catalog_hash,
            raw_input_hash=raw_input_hash,
            action_set_hash=action_set_hash_value,
            market_reference_revision=market_reference_revision,
            idempotency_key=idempotency_key,
            coverage=coverage,
        )
        observed = _utc_naive(observed_at)
        record("prepare_contract", started)
        if (_connection is None) != _manage_transaction:
            raise ValueError("feature persistence connection ownership is inconsistent")
        owns_connection = _connection is None
        if owns_connection:
            started = time.perf_counter()
            connection = self._connect()
            record("connect", started)
        else:
            connection = _connection
        assert connection is not None
        try:
            storage_layout, has_verification = (
                _storage_shape(connection)
                if _batch is None
                else (_batch.storage_layout, _batch.has_verification)
            )
            if source_window is not None and not has_verification:
                raise ValueError("feature.source_verification_storage_unavailable")
            if storage_layout == "CURRENT_BY_ROW_CATALOG":
                # The rows of a layered catalog are its parts': each is written under its own.
                layer = self._resolve_catalog_layer()
                if _batch is None:
                    assert_current_catalog(
                        connection, catalog_hash=layer.catalog.binding.catalog_hash
                    )
                if catalog_hash not in layer.part_hashes:
                    raise ValueError("feature catalog hash does not match current storage binding")
                columns = (
                    "listing_id",
                    "session_date",
                    "catalog_hash",
                    "raw_input_hash",
                    "action_set_hash",
                    "market_reference_revision",
                    "cutoff_set_hash",
                    "row_hash",
                    "updated_at",
                    *catalog_factor_ids,
                )
            else:
                columns = (
                    "listing_id",
                    "session_date",
                    "catalog_hash",
                    "raw_input_hash",
                    "action_set_hash",
                    "market_reference_revision",
                    "input_cutoffs_json",
                    "row_hash",
                    "updated_at",
                    *catalog_factor_ids,
                )
            quoted_columns = ", ".join(f'"{item}"' for item in columns)
            if _manage_transaction:
                started = time.perf_counter()
                with profile(connection, "begin"):
                    connection.execute("BEGIN TRANSACTION")
                record("begin", started)
            receipt_row: list[object] = [
                receipt_hash,
                listing_id,
                range_start,
                range_end,
                catalog_hash,
                raw_input_hash,
                action_set_hash_value,
                market_reference_revision,
                idempotency_key,
                json.dumps(coverage, sort_keys=True, separators=(",", ":")),
                observed,
            ]
            if _batch is not None:
                _batch.receipts.append(receipt_row)
            else:
                started = time.perf_counter()
                with profile(connection, "receipt_write"):
                    connection.execute(
                        """
                        INSERT INTO feature_materialization_receipt
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED', ?, ?)
                        ON CONFLICT (receipt_hash) DO NOTHING
                        """,
                        receipt_row,
                    )
                record("receipt_write", started)
            started = time.perf_counter()
            existing_by_session: dict[date, Sequence[object]]
            if existing_rows is not None:
                # As the read below lays them out, by column name: (row hash, cutoffs, factors).
                existing_by_session = {
                    session: (
                        row["row_hash"],
                        row["input_cutoffs_json"],
                        *(row[factor_id] for factor_id in catalog_factor_ids),
                    )
                    for session, row in existing_rows.items()
                }
            else:
                with profile(connection, "existing_row_read"):
                    factor_projection = ", ".join(
                        f'"{factor_id}"' for factor_id in catalog_factor_ids
                    )
                    held_rows = connection.execute(
                        f"""
                        SELECT session_date, row_hash, input_cutoffs_json, {factor_projection}
                        FROM feature_daily_runtime
                        WHERE listing_id = ? AND catalog_hash = ?
                          AND session_date BETWEEN ? AND ?
                        """,
                        [listing_id, catalog_hash, range_start, range_end],
                    ).fetchall()
                existing_by_session = {row[0]: row[1:] for row in held_rows}
            record("existing_row_read", started)
            inserts: list[list[object]] = []
            revisions: list[list[object]] = []
            cutoff_sets: dict[str, str] = {}
            # The canonical rows to insert: their positions in the frame and each one's
            # (session, cutoff column value, row hash), staged column by column below.
            staged_positions: list[int] = []
            staged_cells: list[tuple[date, str, str]] = []
            # Verification may advance without changing a value or its original
            # provenance. Partial/legacy writes prove no full row.
            verified_receipt = (
                receipt_hash
                if source_window is not None and set(rebuilt_factor_ids) == set(catalog_factor_ids)
                else None
            )

            def storage_values(
                *,
                session: date,
                cutoff_json: str,
                row_hash: str,
                row_values: Sequence[object],
                cutoff_set: tuple[str, str] | None = None,
            ) -> list[object]:
                if storage_layout == "LEGACY_BY_CATALOG":
                    return [
                        listing_id,
                        session,
                        catalog_hash,
                        raw_input_hash,
                        action_set_hash_value,
                        market_reference_revision,
                        cutoff_json,
                        row_hash,
                        observed,
                        *row_values,
                    ]
                canonical_json, cutoff_hash = cutoff_set or canonical_cutoff_set(
                    cutoff_json, factor_ids=catalog_factor_ids
                )
                cutoff_sets[cutoff_hash] = canonical_json
                return [
                    listing_id,
                    session,
                    # The row records what computed it. Reading the identity from
                    # the current binding at query time is what let a rotation
                    # relabel values nothing recomputed.
                    catalog_hash,
                    raw_input_hash,
                    action_set_hash_value,
                    market_reference_revision,
                    cutoff_hash,
                    row_hash,
                    observed,
                    *row_values,
                ]

            started = time.perf_counter()
            if canonical_frame is not None:
                cutoff_values = canonical_frame["input_cutoffs_json"].to_numpy()
                if not all(isinstance(value, str) for value in cutoff_values):
                    raise ValueError("canonical feature rows require serialized input cutoffs")
                try:
                    first_cutoffs = json.loads(str(cutoff_values[0]))
                except json.JSONDecodeError as exc:
                    raise ValueError("feature row has invalid factor-level input cutoffs") from exc
                if not isinstance(first_cutoffs, Mapping) or set(first_cutoffs) != set(
                    catalog_factor_ids
                ):
                    raise ValueError("feature row input-cutoff scope does not match catalog")
                canonical_first = json.dumps(first_cutoffs, sort_keys=True, separators=(",", ":"))
                if canonical_first != cutoff_values[0]:
                    raise ValueError("canonical feature row cutoffs are not canonically encoded")
                # One float per cell, as the rows are stored; a row's cells become
                # Python floats only for a row hashed or compared by value here.
                factor_matrix = canonical_frame.loc[:, list(catalog_factor_ids)].to_numpy(
                    dtype=float
                )
                if row_identities is not None and (
                    len(row_identities) != len(row_sessions)
                    or any(
                        identity.session_date != session
                        for identity, session in zip(row_identities, row_sessions, strict=True)
                    )
                ):
                    raise ValueError("feature row identities do not align with the rows")
                canonical_items = zip(
                    range(len(row_sessions)),
                    row_sessions,
                    cutoff_values,
                    row_identities if row_identities is not None else (None,) * len(row_sessions),
                    strict=True,
                )
            else:
                canonical_items = zip((), (), (), (), strict=True)

            for position, session, cutoff_value, identity in canonical_items:
                cutoff_json = str(cutoff_value)
                row_values: list[float] | None = None
                if identity is not None:
                    row_hash = identity.row_hash
                    cutoff_set: tuple[str, str] | None = (
                        identity.canonical_cutoffs,
                        identity.cutoff_set_hash,
                    )
                else:
                    parsed = json.loads(cutoff_json)
                    assert isinstance(parsed, Mapping)
                    row_values = factor_matrix[position].tolist()
                    row_hash = feature_row_content_hash(
                        listing_id=listing_id,
                        session_date=session,
                        catalog_hash=catalog_hash,
                        input_cutoffs=parsed,
                        factor_ids=catalog_factor_ids,
                        values=row_values,
                    )
                    cutoff_set = None
                existing = existing_by_session.get(session)
                existing_hash = str(existing[0]) if existing is not None else None
                if existing_hash == row_hash:
                    continue
                if existing is not None:
                    parsed_cutoffs = json.loads(cutoff_json)
                    if not isinstance(parsed_cutoffs, Mapping) or set(parsed_cutoffs) != set(
                        catalog_factor_ids
                    ):
                        raise ValueError("feature row input-cutoff scope does not match catalog")
                    if feature_row_is_numerically_identical(
                        existing_cutoffs=str(existing[1]),
                        existing_values=existing[2:],
                        next_cutoffs=parsed_cutoffs,
                        next_values=(
                            row_values
                            if row_values is not None
                            else factor_matrix[position].tolist()
                        ),
                    ):
                        continue
                if storage_layout == "LEGACY_BY_CATALOG":
                    cutoff_column = cutoff_json
                else:
                    canonical_json, cutoff_column = cutoff_set or canonical_cutoff_set(
                        cutoff_json, factor_ids=catalog_factor_ids
                    )
                    cutoff_sets[cutoff_column] = canonical_json
                staged_positions.append(position)
                staged_cells.append((session, cutoff_column, row_hash))
                if existing_hash is not None:
                    revision_id = _canonical_hash(
                        [listing_id, session.isoformat(), catalog_hash, existing_hash, row_hash]
                    )
                    revisions.append(
                        [
                            revision_id,
                            listing_id,
                            session,
                            catalog_hash,
                            existing_hash,
                            row_hash,
                            revision_reason,
                            receipt_hash,
                            observed,
                        ]
                    )

            for row in normalized_rows:
                session = normalized_session(row["session_date"])
                cutoff_value = row.get("input_cutoffs_json")
                if isinstance(cutoff_value, str):
                    try:
                        parsed_cutoffs = json.loads(cutoff_value)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "feature row has invalid factor-level input cutoffs"
                        ) from exc
                else:
                    parsed_cutoffs = cutoff_value
                if not isinstance(parsed_cutoffs, Mapping):
                    raise ValueError("feature row is missing factor-level input cutoffs")
                if set(parsed_cutoffs) != set(catalog_factor_ids):
                    raise ValueError("feature row input-cutoff scope does not match catalog")
                cutoffs = dict(parsed_cutoffs)
                cutoff_json = (
                    cutoff_value
                    if isinstance(cutoff_value, str)
                    else json.dumps(cutoffs, sort_keys=True, separators=(",", ":"))
                )
                row_hash = feature_row_content_hash(
                    listing_id=listing_id,
                    session_date=session,
                    catalog_hash=catalog_hash,
                    input_cutoffs=cutoffs,
                    factor_ids=catalog_factor_ids,
                    values=tuple(row.get(factor_id) for factor_id in catalog_factor_ids),
                )
                existing = existing_by_session.get(session)
                existing_hash = str(existing[0]) if existing is not None else None
                if existing_hash == row_hash:
                    continue
                if existing is not None and feature_row_is_numerically_identical(
                    existing_cutoffs=str(existing[1]),
                    existing_values=existing[2:],
                    next_cutoffs=cutoffs,
                    next_values=tuple(row.get(factor_id) for factor_id in catalog_factor_ids),
                ):
                    continue
                values = storage_values(
                    session=session,
                    cutoff_json=cutoff_json,
                    row_hash=row_hash,
                    row_values=tuple(row.get(factor_id) for factor_id in catalog_factor_ids),
                )
                inserts.append(values)
                if existing_hash is not None:
                    revision_id = _canonical_hash(
                        [listing_id, session.isoformat(), catalog_hash, existing_hash, row_hash]
                    )
                    revisions.append(
                        [
                            revision_id,
                            listing_id,
                            session,
                            catalog_hash,
                            existing_hash,
                            row_hash,
                            revision_reason,
                            receipt_hash,
                            observed,
                        ]
                    )
            record("row_normalize_hash", started)
            if inserts or staged_positions:
                # A set an earlier write of the batch inserted is in this transaction's
                # table already, so it is not offered again (V92).
                offered = _batch.offered_cutoff_sets if _batch is not None else set()
                ordered = sorted(item for item in cutoff_sets.items() if item[0] not in offered)
                offered.update(item[0] for item in ordered)
                if ordered:
                    # One cutoff set per session, so a full listing offers ~2,515
                    # of them and every listing after the first offers the same
                    # ones again -- the sessions are shared, so all but the first
                    # listing's are conflicts that do nothing. Submitted through
                    # `executemany` that is one statement per row: 1.17M
                    # single-row inserts over a 466-listing rebuild, and 3.3 s of
                    # the 3.5 s each listing takes. Offered as one relation it is
                    # a single insert, and the conflict clause still decides what
                    # is kept.
                    cutoff_stage = "feature_cutoff_set_stage"
                    connection.register(
                        cutoff_stage,
                        pa.table(
                            {
                                "cutoff_set_hash": pa.array(
                                    [item[0] for item in ordered], pa.string()
                                ),
                                "input_cutoffs_json": pa.array(
                                    [item[1] for item in ordered], pa.string()
                                ),
                                "factor_count": pa.array(
                                    [len(catalog_factor_ids)] * len(ordered), pa.int32()
                                ),
                            }
                        ),
                    )
                    try:
                        with profile(connection, "cutoff_set_write"):
                            connection.execute(
                                f"""
                                INSERT INTO feature_input_cutoff_set
                                SELECT cutoff_set_hash, input_cutoffs_json, factor_count
                                FROM {cutoff_stage}
                                ON CONFLICT (cutoff_set_hash) DO NOTHING
                                """
                            )
                    finally:
                        connection.unregister(cutoff_stage)
                stage_name = "feature_daily_stage"
                started = time.perf_counter()
                inserted_columns = quoted_columns
                if staged_positions:
                    # The canonical rows by column, as `_staged_rows` would type them
                    # from a row list: the frame's own floats for the factors, not a
                    # Python float per cell and back. The verification receipt goes in
                    # with them rather than by a second statement over the same rows.
                    stage_table = _canonical_stage(
                        columns,
                        staged_cells,
                        factor_matrix[staged_positions],
                        constants={
                            "listing_id": listing_id,
                            "catalog_hash": catalog_hash,
                            "raw_input_hash": raw_input_hash,
                            "action_set_hash": action_set_hash_value,
                            "market_reference_revision": market_reference_revision,
                            "updated_at": observed,
                        },
                        verification=(verified_receipt,) if has_verification else (),
                    )
                    if has_verification:
                        inserted_columns += ', "source_verification_receipt_hash"'
                else:
                    stage_table = _staged_rows(columns, inserts)
                record("current_arrow_encode", started)
                started = time.perf_counter()
                connection.register(
                    stage_name,
                    stage_table,
                )
                insert_mode = "INSERT OR REPLACE" if revisions else "INSERT"
                with profile(connection, "current_write"):
                    connection.execute(
                        f"{insert_mode} INTO feature_daily_current ({inserted_columns}) "
                        f"SELECT {inserted_columns} FROM {stage_name}"
                    )
                    self._clear_feature_year_seals(connection, row_sessions)
                connection.unregister(stage_name)
                record("current_write", started)
            if revisions:
                started = time.perf_counter()
                with profile(connection, "revision_write"):
                    revision_columns = (
                        "revision_id",
                        "listing_id",
                        "session_date",
                        "catalog_hash",
                        "prior_row_hash",
                        "next_row_hash",
                        "revision_reason",
                        "materialization_receipt_hash",
                        "observed_at",
                    )
                    revision_stage_name = "feature_revision_stage"
                    revision_stage = _staged_rows(revision_columns, revisions)
                    connection.register(revision_stage_name, revision_stage)
                    connection.execute(
                        """
                        INSERT INTO feature_daily_revision
                        SELECT revision_id, listing_id, session_date, catalog_hash,
                               prior_row_hash, next_row_hash, revision_reason,
                               materialization_receipt_hash, observed_at
                        FROM feature_revision_stage
                        ON CONFLICT DO NOTHING
                        """
                    )
                    connection.unregister(revision_stage_name)
                record("revision_write", started)
            # Durable mutation is run-native.  The compatibility view can
            # expand millions of rows and is never an engine input.
            started = time.perf_counter()
            with profile(connection, "rle_overlap_read"):
                rebuilt_factor_marks = ", ".join("?" for _ in rebuilt_factor_ids)
                overlapping_runs = connection.execute(
                    f"""
                    SELECT run_id, listing_id, catalog_hash, factor_id, reason,
                           first_session, last_session, first_observation_count,
                           observation_cap, materialization_receipt_hash, updated_at
                    FROM feature_ineligibility_run
                    WHERE listing_id = ? AND catalog_hash = ?
                      AND factor_id IN ({rebuilt_factor_marks})
                      AND first_session <= ? AND last_session >= ?
                    ORDER BY factor_id, first_session
                    """,
                    [
                        listing_id,
                        catalog_hash,
                        *rebuilt_factor_ids,
                        range_end,
                        range_start,
                    ],
                ).fetchall()
            record("rle_overlap_read", started)
            started = time.perf_counter()
            retained_runs: list[list[object]] = []
            if overlapping_runs:
                overlap_start = min(row[5] for row in overlapping_runs)
                overlap_end = max(row[6] for row in overlapping_runs)
                with profile(connection, "rle_session_read"):
                    feature_sessions = [
                        row[0]
                        for row in connection.execute(
                            """
                            SELECT session_date FROM feature_daily_runtime
                            WHERE listing_id = ? AND catalog_hash = ?
                              AND session_date BETWEEN ? AND ?
                            ORDER BY session_date
                            """,
                            [listing_id, catalog_hash, overlap_start, overlap_end],
                        ).fetchall()
                    ]
                session_positions = {
                    session: position for position, session in enumerate(feature_sessions)
                }

                def retained_fragment(
                    run: tuple[object, ...], first_session: date, last_session: date
                ) -> list[object]:
                    old_first = run[5]
                    if old_first not in session_positions:
                        raise ValueError("stored ineligibility run has no feature-session origin")
                    first_offset = session_positions[first_session] - session_positions[old_first]
                    last_offset = session_positions[last_session] - session_positions[old_first]
                    first_observation = min(int(run[7]) + first_offset, int(run[8]))
                    last_observation = min(int(run[7]) + last_offset, int(run[8]))
                    payload = {
                        "listing_id": run[1],
                        "catalog_hash": run[2],
                        "factor_id": run[3],
                        "reason": run[4],
                        "first_session": first_session,
                        "last_session": last_session,
                        "first_observation_count": first_observation,
                        "observation_cap": last_observation,
                        "materialization_receipt_hash": run[9],
                        "updated_at": run[10],
                    }
                    return [_canonical_hash(payload), *payload.values()]

                for run in overlapping_runs:
                    run_first = run[5]
                    run_last = run[6]
                    if run_first < range_start:
                        left_sessions = [
                            session
                            for session in feature_sessions
                            if run_first <= session < range_start and session <= run_last
                        ]
                        if left_sessions:
                            retained_runs.append(
                                retained_fragment(run, left_sessions[0], left_sessions[-1])
                            )
                    if run_last > range_end:
                        right_sessions = [
                            session
                            for session in feature_sessions
                            if run_first <= session <= run_last and session > range_end
                        ]
                        if right_sessions:
                            retained_runs.append(
                                retained_fragment(run, right_sessions[0], right_sessions[-1])
                            )
                with profile(connection, "rle_overlap_delete"):
                    connection.execute(
                        f"""
                        DELETE FROM feature_ineligibility_run
                        WHERE listing_id = ? AND catalog_hash = ?
                          AND factor_id IN ({rebuilt_factor_marks})
                          AND first_session <= ? AND last_session >= ?
                        """,
                        [
                            listing_id,
                            catalog_hash,
                            *rebuilt_factor_ids,
                            range_end,
                            range_start,
                        ],
                    )
            fact_sessions, fact_factors, fact_reasons, fact_observations = _ineligibility_facts(
                ineligibility,
                range_start=range_start,
                range_end=range_end,
                rebuilt_factor_ids=rebuilt_factor_ids,
            )
            facts = len(fact_sessions)
            input_positions = {session: index for index, session in enumerate(sorted(row_sessions))}
            new_runs = self._compress_ineligibility_rows(
                {
                    "listing_id": [listing_id] * facts,
                    "session_date": fact_sessions,
                    "catalog_hash": [catalog_hash] * facts,
                    "factor_id": fact_factors,
                    "reason": fact_reasons,
                    "observation_count": fact_observations,
                    "materialization_receipt_hash": [receipt_hash] * facts,
                    "updated_at": [observed] * facts,
                },
                input_positions,
            )
            runs = [*retained_runs, *new_runs]
            record("rle_prepare", started)
            if runs and _batch is not None and _batch.runs is not None:
                _batch.runs.extend(runs)
            elif runs:
                run_columns = _INELIGIBILITY_RUN_COLUMNS
                started = time.perf_counter()
                run_stage = "feature_ineligibility_run_stage"
                connection.register(run_stage, _staged_rows(run_columns, runs))
                with profile(connection, "rle_write"):
                    connection.execute(
                        f"""
                        INSERT INTO feature_ineligibility_run
                        SELECT * FROM {run_stage}
                        ON CONFLICT (run_id) DO NOTHING
                        """
                    )
                connection.unregister(run_stage)
                record("rle_write", started)
            if has_verification:
                # Every row at these sessions carries the verification: an inserted
                # canonical row already does; each other row is stamped here.
                inserted = {cell[0] for cell in staged_cells}
                unstamped = [session for session in row_sessions if session not in inserted]
                if unstamped:
                    connection.execute(
                        """UPDATE feature_daily_current SET source_verification_receipt_hash = ?
                           WHERE listing_id = ? AND catalog_hash = ?
                             AND session_date IN (SELECT unnest(?))""",
                        [verified_receipt, listing_id, catalog_hash, unstamped],
                    )
                    self._clear_feature_year_seals(connection, unstamped)
            if _manage_transaction:
                started = time.perf_counter()
                with profile(connection, "commit"):
                    connection.execute("COMMIT")
                record("commit", started)
        except Exception:
            if _manage_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            if owns_connection:
                started = time.perf_counter()
                connection.close()
                record("close", started)
        return receipt_hash

    def upsert_feature_materialization_batch(
        self,
        writes: Sequence[FeatureMaterializationWrite],
        *,
        timing_sink: FeaturePersistenceTimingSink | None = None,
        sql_profiler: FeaturePersistenceSqlProfiler | None = None,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> tuple[str, ...]:
        """Atomically persist a bounded batch while retaining listing receipts."""
        bounded = tuple(writes)
        if not bounded:
            raise ValueError("feature materialization batch is empty")
        width = min(
            len(self._feature_factor_ids(catalog_hash))
            for catalog_hash in {write.catalog_hash for write in bounded}
        )
        if len(bounded) > admitted_writes(width):
            raise ValueError("feature materialization batch exceeds the bounded desktop limit")

        def record(stage: str, started: float) -> None:
            if timing_sink is not None:
                timing_sink(stage, time.perf_counter() - started)

        owns_connection = _connection is None
        if owns_connection:
            started = time.perf_counter()
            connection = self._connect()
            record("batch_connect", started)
        else:
            connection = _connection
        assert connection is not None
        try:
            started = time.perf_counter()
            connection.execute("BEGIN TRANSACTION")
            record("batch_begin", started)
            storage_layout, has_verification = _storage_shape(connection)
            if storage_layout == "CURRENT_BY_ROW_CATALOG":
                assert_current_catalog(
                    connection,
                    catalog_hash=self._resolve_catalog_layer().catalog.binding.catalog_hash,
                )
            listing_ids = [write.listing_id for write in bounded]
            batch = _FeatureWriteBatch(
                storage_layout=storage_layout,
                has_verification=has_verification,
                offered_cutoff_sets=set(),
                receipts=[],
                runs=[] if len(set(listing_ids)) == len(listing_ids) else None,
            )
            receipts = tuple(
                self.upsert_feature_materialization(
                    listing_id=write.listing_id,
                    catalog_hash=write.catalog_hash,
                    rows=write.rows,
                    ineligibility=write.ineligibility,
                    raw_input_hash=write.raw_input_hash,
                    action_set_hash_value=write.action_set_hash_value,
                    market_reference_revision=write.market_reference_revision,
                    idempotency_key=write.idempotency_key,
                    revision_reason=write.revision_reason,
                    observed_at=write.observed_at,
                    factor_ids=write.factor_ids,
                    rows_are_canonical=write.rows_are_canonical,
                    source_window=write.source_window,
                    row_identities=write.row_identities,
                    existing_rows=write.existing_rows,
                    timing_sink=timing_sink,
                    sql_profiler=sql_profiler,
                    _connection=connection,
                    _manage_transaction=False,
                    _batch=batch,
                )
                for write in bounded
            )
            self._write_batch_rows(connection, batch, record=record)
            started = time.perf_counter()
            connection.execute("COMMIT")
            record("batch_commit", started)
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            if owns_connection:
                started = time.perf_counter()
                connection.close()
                record("batch_close", started)
        return receipts

    @staticmethod
    def _write_batch_rows(
        connection: duckdb.DuckDBPyConnection,
        batch: _FeatureWriteBatch,
        *,
        record: Callable[[str, float], None],
    ) -> None:
        """Insert what the batch's writes gathered: their receipts, and their runs if deferred.

        Each row once, the first offered kept, as the conflict clause keeps the first written.
        """
        if batch.receipts:
            started = time.perf_counter()
            connection.register(
                "feature_receipt_stage",
                _staged_rows(_RECEIPT_STAGE_COLUMNS, _first_by_identity(batch.receipts)),
            )
            try:
                connection.execute(
                    """
                    INSERT INTO feature_materialization_receipt
                    SELECT receipt_hash, listing_id, range_start, range_end, catalog_hash,
                           raw_input_hash, action_set_hash, market_reference_revision,
                           idempotency_key, 'COMPLETED', coverage_summary_json, observed_at
                    FROM feature_receipt_stage
                    ON CONFLICT (receipt_hash) DO NOTHING
                    """
                )
            finally:
                connection.unregister("feature_receipt_stage")
            record("receipt_write", started)
        if batch.runs:
            started = time.perf_counter()
            connection.register(
                "feature_ineligibility_run_stage",
                _staged_rows(_INELIGIBILITY_RUN_COLUMNS, _first_by_identity(batch.runs)),
            )
            try:
                connection.execute(
                    """
                    INSERT INTO feature_ineligibility_run
                    SELECT * FROM feature_ineligibility_run_stage
                    ON CONFLICT (run_id) DO NOTHING
                    """
                )
            finally:
                connection.unregister("feature_ineligibility_run_stage")
            record("rle_write", started)

    def find_feature_ineligibility_runs(
        self,
        *,
        listing_ids: Sequence[str],
        factor_ids: Sequence[str],
        start: date,
        end: date,
    ) -> tuple[FeatureIneligibilityRun, ...]:
        """Read physical intervals; internal engines must not expand the compatibility view."""
        listing_scope = tuple(sorted(set(listing_ids)))
        factor_scope = tuple(sorted(set(factor_ids)))
        if not listing_scope or not factor_scope:
            return ()
        if start > end:
            raise ValueError("feature ineligibility range start is after end")
        listing_placeholders = ", ".join("?" for _ in listing_scope)
        factor_placeholders = ", ".join("?" for _ in factor_scope)
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                f"""
                SELECT run_id, listing_id, catalog_hash, factor_id, reason,
                       first_session, last_session, first_observation_count,
                       observation_cap, materialization_receipt_hash, updated_at
                FROM feature_ineligibility_run
                WHERE listing_id IN ({listing_placeholders})
                  AND factor_id IN ({factor_placeholders})
                  AND first_session <= ? AND last_session >= ?
                ORDER BY listing_id, factor_id, first_session
                """,
                [*listing_scope, *factor_scope, end, start],
            ).fetchall()
        finally:
            connection.close()
        return tuple(FeatureIneligibilityRun(*row) for row in rows)

    def feature_rows(
        self,
        *,
        listing_ids: Sequence[str],
        catalog_hash: str,
        start: date,
        end: date,
        factor_ids: Sequence[str] | None = None,
        include_lineage: bool = True,
        include_values: bool = True,
        include_verification: bool = False,
        as_frame: bool = False,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> list[dict[str, object]] | pd.DataFrame:
        """Trusted engine read-back; UI and agents never receive this payload."""
        if not listing_ids:
            return []
        catalog_factors = self._feature_factor_ids(catalog_hash)
        factors = tuple(factor_ids) if factor_ids is not None else catalog_factors
        if not factors or not set(factors).issubset(catalog_factors):
            raise ValueError("feature row projection contains an unknown factor")
        columns = ", ".join(f'"{factor}"' for factor in factors)
        if not include_values:
            columns = ""
        leading_columns = (
            "listing_id, session_date, catalog_hash, raw_input_hash, action_set_hash, "
            "market_reference_revision, input_cutoffs_json, row_hash"
            if include_lineage
            else "listing_id, session_date"
        )
        marks = ", ".join("?" for _ in listing_ids)
        connection = _connection or self._connect(read_only=True)
        owns_connection = _connection is None
        try:
            if include_verification:
                names = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info('feature_daily_runtime')"
                    ).fetchall()
                }
                verification = (
                    "source_verification_receipt_hash"
                    if "source_verification_receipt_hash" in names
                    else "NULL"
                )
                leading_columns += f", {verification} AS source_verification_receipt_hash"
            result = connection.execute(
                f"""
                SELECT {leading_columns}{(", " + columns) if columns else ""}
                FROM feature_daily_runtime
                WHERE listing_id IN ({marks}) AND catalog_hash = ?
                  AND session_date BETWEEN ? AND ?
                ORDER BY session_date, listing_id
                """,
                [*listing_ids, catalog_hash, start, end],
            )
            if as_frame:
                rows = result.to_arrow_table().to_pandas()
            else:
                names = [column[0] for column in result.description]
                rows = [dict(zip(names, row, strict=True)) for row in result.fetchall()]
        finally:
            if owns_connection:
                connection.close()
        return rows

    def feature_source_prefix_proof(
        self,
        *,
        listing_ids: Sequence[str],
        catalog_hash: str,
        start: date,
        end: date,
        factor_ids: Sequence[str],
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> str:
        """Prove actual selected Feature values over an exact historical source scope.

        This is the bounded row selection of ``feature_rows``, hashed from the
        live keys, selected values and actual cutoff/source-verification lineage
        rather than trusting stored row hashes or lineage counters. Separate
        null bits distinguish a null cell from a valid NaN;
        actual row keys distinguish a missing row from an all-null observation.
        Numerical bits, including negative zero, are retained in canonical Arrow.

        Args:
            listing_ids: Nonempty unique listing scope, normalized to sorted order.
            catalog_hash: Exact installed computing catalog.
            start: First included session.
            end: Last included session.
            factor_ids: Nonempty unique stored factor selection, normalized to sorted order.
            _connection: Optional connection already held in a consistent read snapshot.

        Returns:
            SHA256 binding request scope and exact current selected keys, values and nulls.

        Raises:
            ValueError: The request or selected catalog factors are invalid.
        """
        return self.feature_source_prefix_proofs(
            listing_ids=listing_ids,
            catalog_hash=catalog_hash,
            start=start,
            ends=(end,),
            factor_ids=factor_ids,
            _connection=_connection,
        )[0]

    def feature_source_prefix_proofs(
        self,
        *,
        listing_ids: Sequence[str],
        catalog_hash: str,
        start: date,
        ends: Sequence[date],
        factor_ids: Sequence[str],
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> tuple[str, ...]:
        """Prove selected Feature prefixes from their year seals and one read of the open year.

        Each proof (``FeatureSourcePrefixProofV2``) binds its scope and selection, the seal of
        every calendar year before its cutoff's (`_feature_year_seals`), and the open year's
        selected rows from its first session (or ``start``) to the cutoff, hashed as canonical
        Arrow from one read (`_prefix_arrow_hashes`). A seal covers every listing and factor of
        its year, and every write to a year's rows removes it, so a change anywhere in the prefix
        moves the proof; only the open year is read each time.

        Args:
            listing_ids: Nonempty unique listing scope, normalized to sorted order.
            catalog_hash: Exact installed computing catalog.
            start: First included session.
            ends: Nonempty requested cutoff sequence; duplicate cutoffs are allowed.
            factor_ids: Nonempty unique stored factor selection, normalized to sorted order.
            _connection: Optional connection already held in a consistent read snapshot.

        Returns:
            Exact source proofs in the requested cutoff order.

        Raises:
            ValueError: The request or selected catalog factors are invalid.
        """
        scope, factors, cutoffs = tuple(listing_ids), tuple(factor_ids), tuple(ends)
        if (
            not scope
            or not factors
            or not cutoffs
            or any(not isinstance(value, str) or not value for value in (*scope, *factors))
            or len(set(scope)) != len(scope)
            or len(set(factors)) != len(factors)
            or any(start > end for end in cutoffs)
        ):
            raise ValueError("feature.source_prefix_request_invalid")
        scope, factors = tuple(sorted(scope)), tuple(sorted(factors))
        if not set(factors) <= set(self._feature_factor_ids(catalog_hash)):
            raise ValueError("feature row projection contains an unknown factor")
        boundary = (
            self.database.read_transaction() if _connection is None else nullcontext(_connection)
        )
        with boundary as connection:
            statement = self._feature_prefix_statement(connection, factors, scoped=True)
            # The open year of each cutoff is read whole from its first session (or `start`);
            # every earlier year answers by its seal.
            opens: dict[date, list[date]] = {}
            for end in dict.fromkeys(cutoffs):
                opens.setdefault(max(start, date(end.year, 1, 1)), []).append(end)
            open_values: dict[date, str] = {}
            for open_start, open_ends in opens.items():
                open_values.update(
                    self._prefix_arrow_hashes(
                        connection, statement, [scope, catalog_hash], open_start, open_ends
                    )
                )
            seals = self._feature_year_seals(
                connection, catalog_hash, range(start.year, max(cutoffs).year)
            )
        return tuple(
            _canonical_hash(
                {
                    "kind": "FeatureSourcePrefixProofV2",
                    "listing_ids": scope,
                    "catalog_hash": catalog_hash,
                    "start": start,
                    "end": end,
                    "factor_ids": factors,
                    "closed_years": [[year, seals[year]] for year in range(start.year, end.year)],
                    "open_values": open_values[end],
                }
            )
            for end in cutoffs
        )

    def _feature_prefix_statement(
        self,
        connection: duckdb.DuckDBPyConnection,
        factors: Sequence[str],
        *,
        scoped: bool,
    ) -> str:
        """The proved projection of the runtime rows, its listing scope first when ``scoped``.

        Parameters: the scope (when ``scoped``), the catalog, the first and the last session.
        """
        projection = ", ".join(
            f'COALESCE("{factor}", 0::DOUBLE) AS value_{index}, "{factor}" IS NULL AS null_{index}'
            for index, factor in enumerate(factors)
        )
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info('feature_daily_runtime')").fetchall()
        }
        verification = (
            "source_verification_receipt_hash"
            if "source_verification_receipt_hash" in columns
            else "NULL::VARCHAR"
        )
        listing_filter = "listing_id IN (SELECT unnest(?)) AND " if scoped else ""
        return f"""
            SELECT listing_id, session_date, catalog_hash,
                   COALESCE(raw_input_hash, '') AS raw_input_hash,
                   raw_input_hash IS NULL AS raw_input_is_null,
                   COALESCE(action_set_hash, '') AS action_set_hash,
                   action_set_hash IS NULL AS action_set_is_null,
                   COALESCE(market_reference_revision, '') AS market_reference_revision,
                   market_reference_revision IS NULL AS market_reference_is_null,
                   COALESCE(input_cutoffs_json, '') AS input_cutoffs_json,
                   input_cutoffs_json IS NULL AS cutoffs_is_null,
                   COALESCE({verification}, '') AS source_verification_receipt_hash,
                   {verification} IS NULL AS verification_is_null,
                   {projection}
            FROM feature_daily_runtime
            WHERE {listing_filter}catalog_hash = ?
              AND session_date BETWEEN ? AND ?
            ORDER BY session_date, listing_id
            """

    @staticmethod
    def _prefix_arrow_hashes(
        connection: duckdb.DuckDBPyConnection,
        statement: str,
        leading: list[object],
        start: date,
        ends: Sequence[date],
    ) -> dict[date, str]:
        """Each cutoff's rows from ``start``, hashed as canonical Arrow from one read.

        The largest prefix uses the original SQL projection. Earlier prefixes are exported
        through the same connection to preserve the original Arrow bytes, including null flags
        and negative zero; slicing or taking Arrow rows alone does not preserve that encoding. A
        prefix that still spans multiple original chunks uses their original buffers before
        normalization. A single-chunk prefix of a multi-chunk export retains per-cutoff SQL,
        including its original Boolean padding. Prefixes are hashed and released in turn,
        retaining only the full table and one earlier prefix. Arrow and IPC buffers are not
        automatically bounded by DuckDB's memory limit.
        """
        latest = max(ends)
        rows = connection.execute(statement, [*leading, start, latest]).to_arrow_table()
        multiple_chunks = any(column.num_chunks > 1 for column in rows.columns)
        session_axis = rows.column("session_date").to_numpy(zero_copy_only=False)
        hashes: dict[date, str] = {}
        for end in dict.fromkeys(ends):
            if end == latest:
                prefix = rows
            elif multiple_chunks:
                stop = int(np.searchsorted(session_axis, np.datetime64(end), side="right"))
                bounded = rows.slice(0, stop)
                if all(column.num_chunks > 1 for column in bounded.columns):
                    prefix = bounded
                else:
                    prefix = connection.execute(statement, [*leading, start, end]).to_arrow_table()
            else:
                prefix = (
                    connection.from_arrow(rows)
                    .filter(f"session_date <= DATE '{end.isoformat()}'")
                    .order("session_date, listing_id")
                    .to_arrow_table()
                )
            hashes[end] = _source_prefix_arrow_hash(prefix)
            del prefix
        return hashes

    def _feature_year_seals(
        self, connection: duckdb.DuckDBPyConnection, catalog_hash: str, years: Iterable[int]
    ) -> dict[int, str]:
        """Each year's seal digest: a stored seal that still vouches, else computed now.

        A seal vouches under the runtime view it was made under and in the store's current
        seal epoch (`WorkspaceDatabase.seal_epoch`); a write to a year's rows removed it in the
        write's own transaction. One computed here is not stored: it is only slower.
        """
        wanted = tuple(years)
        if not wanted:
            return {}
        stored = {
            int(year): str(digest)
            for year, digest in connection.execute(
                """
                SELECT year, digest FROM feature_year_seal
                WHERE catalog_hash = ? AND view_hash = ? AND epoch = ?
                  AND list_contains(?::INTEGER[], year)
                """,
                [
                    catalog_hash,
                    runtime_view_hash(connection),
                    self.database.seal_epoch(),
                    list(wanted),
                ],
            ).fetchall()
        }
        return {
            year: stored[year]
            if year in stored
            else self._feature_year_digest(connection, catalog_hash, year)
            for year in wanted
        }

    def _feature_year_digest(
        self, connection: duckdb.DuckDBPyConnection, catalog_hash: str, year: int
    ) -> str:
        """One calendar year of the catalog's rows: every listing and stored factor, hashed.

        A superset of any proof's scope and selection, so a change outside them can only cost
        a rebuild, never pass unseen.
        """
        factors = tuple(sorted(self._feature_factor_ids(catalog_hash)))
        rows = connection.execute(
            self._feature_prefix_statement(connection, factors, scoped=False),
            [catalog_hash, date(year, 1, 1), date(year + 1, 1, 1) - timedelta(days=1)],
        ).to_arrow_table()
        return _source_prefix_arrow_hash(rows)

    def seal_closed_feature_years(
        self, catalog_hash: str, *, _connection: duckdb.DuckDBPyConnection
    ) -> tuple[int, ...]:
        """Seal each closed year of the catalog's rows that no seal of this epoch vouches for.

        A year is closed once a later year holds a row: a daily write no longer reaches it, so
        its seal holds until a correction's write removes it. Runs in its own transaction on
        the build's connection.

        Returns:
            The years sealed now.
        """
        connection = _connection
        connection.execute("BEGIN TRANSACTION")
        try:
            bounds = connection.execute(
                "SELECT min(session_date), max(session_date) FROM feature_daily_runtime "
                "WHERE catalog_hash = ?",
                [catalog_hash],
            ).fetchone()
            sealed: list[int] = []
            if bounds is not None and bounds[0] is not None:
                view_hash, epoch = runtime_view_hash(connection), self.database.seal_epoch()
                held = {
                    int(row[0])
                    for row in connection.execute(
                        "SELECT year FROM feature_year_seal "
                        "WHERE catalog_hash = ? AND view_hash = ? AND epoch = ?",
                        [catalog_hash, view_hash, epoch],
                    ).fetchall()
                }
                for year in range(bounds[0].year, bounds[1].year):
                    if year in held:
                        continue
                    connection.execute(
                        "INSERT OR REPLACE INTO feature_year_seal VALUES (?, ?, ?, ?, ?)",
                        [
                            catalog_hash,
                            year,
                            view_hash,
                            epoch,
                            self._feature_year_digest(connection, catalog_hash, year),
                        ],
                    )
                    sealed.append(year)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        return tuple(sealed)

    @staticmethod
    def _clear_feature_year_seals(
        connection: duckdb.DuckDBPyConnection, sessions: Iterable[date] | None = None
    ) -> None:
        """A write to Feature rows ends the seals of their years, in its own transaction.

        Every catalog's: a layered catalog's rows compose from its parts' rows. None clears all.
        """
        if sessions is None:
            connection.execute("DELETE FROM feature_year_seal")
            return
        years = sorted({session.year for session in sessions})
        if years:
            connection.execute(
                "DELETE FROM feature_year_seal WHERE list_contains(?::INTEGER[], year)", [years]
            )

    def materialization_source_windows(
        self,
        *,
        listing_id: str,
        catalog_hash: str,
        receipt_hashes: Sequence[str],
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> dict[str, tuple[FeatureSourceWindow, date, date]]:
        """Verify existing receipts before a caller compares their numerical inputs."""
        if not receipt_hashes:
            return {}
        connection = _connection or self._connect(read_only=True)
        try:
            rows = connection.execute(
                """SELECT receipt_hash, range_start, range_end, raw_input_hash, action_set_hash,
                          market_reference_revision, idempotency_key,
                          coverage_summary_json, work_status
                   FROM feature_materialization_receipt WHERE listing_id = ? AND catalog_hash = ?
                     AND receipt_hash IN (SELECT unnest(?))""",
                [listing_id, catalog_hash, tuple(receipt_hashes)],
            ).fetchall()
        finally:
            if _connection is None:
                connection.close()
        result = {}
        factor_count = len(self._feature_factor_ids(catalog_hash))
        for identity, start, end, raw, actions, market, key, encoded, status in rows:
            coverage = json.loads(encoded)
            if not isinstance(coverage, dict):
                raise ValueError("feature.source_verification_receipt_invalid")
            expected = feature_materialization_receipt_hash(
                listing_id=listing_id,
                range_start=start,
                range_end=end,
                catalog_hash=catalog_hash,
                raw_input_hash=raw,
                action_set_hash=actions,
                market_reference_revision=market,
                idempotency_key=key,
                coverage=coverage,
            )
            if (
                expected != identity
                or status != "COMPLETED"
                or (
                    coverage.get("factor_count") != factor_count
                    or coverage.get("rebuilt_factor_count") != factor_count
                )
                or not isinstance(coverage.get("source_window"), dict)
            ):
                raise ValueError("feature.source_verification_receipt_invalid")
            window = FeatureSourceWindow.from_payload(coverage["source_window"])
            if not window.first_output_session <= start <= end <= window.input_end:
                raise ValueError("feature.source_verification_receipt_invalid")
            result[str(identity)] = (window, start, end)
        if result.keys() != set(receipt_hashes):
            raise ValueError("feature.source_verification_receipt_absent")
        return result

    def feature_sessions(
        self,
        *,
        listing_ids: Sequence[str],
        catalog_hash: str,
        start: date,
        end: date,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> list[date]:
        """Return the bounded panel schedule without loading a feature matrix."""
        if not listing_ids:
            return []
        marks = ", ".join("?" for _ in listing_ids)
        connection = _connection or self._connect(read_only=True)
        owns_connection = _connection is None
        try:
            rows = connection.execute(
                f"""
                SELECT DISTINCT session_date FROM feature_daily_runtime
                WHERE listing_id IN ({marks}) AND catalog_hash = ?
                  AND session_date BETWEEN ? AND ?
                ORDER BY session_date
                """,
                [*listing_ids, catalog_hash, start, end],
            ).fetchall()
        finally:
            if owns_connection:
                connection.close()
        return [row[0] for row in rows]

    def feature_revision_count(self, listing_id: str) -> int:
        """Count persisted Feature revisions for one listing."""
        connection = self._connect(read_only=True)
        try:
            return int(
                connection.execute(
                    "SELECT count(*) FROM feature_daily_revision WHERE listing_id = ?", [listing_id]
                ).fetchone()[0]
            )
        finally:
            connection.close()

    def upsert_market_reference(
        self,
        *,
        reference_id: str,
        listing_id: str,
        provider: str,
        symbol: str,
        revision_hash: str,
        action_audit_receipt_hash: str | None,
        latest_session: date | None,
        observed_at: datetime,
    ) -> None:
        """Store the current market reference and its source revision."""
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO market_reference_current VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (reference_id) DO UPDATE SET
                    listing_id = excluded.listing_id, provider = excluded.provider,
                    symbol = excluded.symbol, revision_hash = excluded.revision_hash,
                    action_audit_receipt_hash = excluded.action_audit_receipt_hash,
                    latest_session = excluded.latest_session, updated_at = excluded.updated_at
                """,
                [
                    reference_id,
                    listing_id,
                    provider,
                    symbol,
                    revision_hash,
                    action_audit_receipt_hash,
                    latest_session,
                    _utc_naive(observed_at),
                ],
            )
        finally:
            connection.close()

    def market_reference(self, reference_id: str = "SPY") -> dict[str, object] | None:
        """Return the current market reference record, if present."""
        connection = self._connect(read_only=True)
        try:
            result = connection.execute(
                """
                SELECT reference_id, listing_id, provider, symbol, revision_hash,
                       action_audit_receipt_hash, latest_session, updated_at
                FROM market_reference_current WHERE reference_id = ?
                """,
                [reference_id],
            )
            row = result.fetchone()
            if row is None:
                return None
            return dict(zip((item[0] for item in result.description), row, strict=True))
        finally:
            connection.close()

    def verified_market_reference(
        self,
        manifest: UniverseManifest,
        *,
        reference_id: str,
        requested_as_of: date,
    ) -> dict[str, object] | None:
        """Read back an already-published market reference without a TTL fetch.

        ``reusable_action_audit_receipt`` answers an acquisition-budget question:
        whether a recent receipt can suppress a new provider audit.  A published
        market reference is different authority.  It remains reusable after the
        audit TTL expires provided its referenced receipt and every current input
        still verify.  This method therefore recomputes the receipt evidence,
        diagnostics, and reference revision rather than treating age as drift.
        """
        if len(manifest.listings) != 1:
            raise ValueError("market-reference verification requires one listing")
        listing_id = manifest.listings[0].listing_id
        connection = self._connect(read_only=True)
        try:
            result = connection.execute(
                """
                SELECT reference.reference_id, reference.listing_id, reference.provider,
                       reference.symbol, reference.revision_hash,
                       reference.action_audit_receipt_hash,
                       reference.latest_session, reference.updated_at,
                       receipt.receipt_hash, receipt.listing_id, receipt.provider,
                       receipt.manifest_revision, receipt.mapping_revision,
                       receipt.requested_as_of, receipt.history_start, receipt.history_end,
                       receipt.raw_evidence_hash, receipt.action_set_hash,
                       receipt.action_evidence_hash,
                       receipt.provider_adjusted_close_evidence_hash,
                       receipt.max_adjusted_close_difference_bps,
                       receipt.adjusted_close_mismatch_count,
                       receipt.first_adjusted_close_mismatch_session,
                       receipt.diagnostic_policy_hash, receipt.observed_at
                FROM market_reference_current AS reference
                LEFT JOIN action_audit_receipt AS receipt
                  ON receipt.receipt_hash = reference.action_audit_receipt_hash
                WHERE reference.reference_id = ?
                """,
                [reference_id],
            )
            row = result.fetchone()
            if row is None or row[8] is None:
                return None
            current = dict(
                zip(
                    (
                        "reference_id",
                        "listing_id",
                        "provider",
                        "symbol",
                        "revision_hash",
                        "action_audit_receipt_hash",
                        "latest_session",
                        "updated_at",
                    ),
                    row[:8],
                    strict=True,
                )
            )
            receipt = ActionAuditReceipt(*row[8:])
            provider = str(current["provider"])
            if (
                str(current["listing_id"]) != listing_id
                or current["latest_session"] != requested_as_of
                or receipt.listing_id != listing_id
                or receipt.provider != provider
                or receipt.manifest_revision != manifest.revision_sha256
                or receipt.requested_as_of != requested_as_of
                or receipt.history_end < requested_as_of
                or receipt.diagnostic_policy_hash != ADJUSTED_CLOSE_DIAGNOSTIC_POLICY_HASH
                or receipt.provider_adjusted_close_evidence_hash is None
                or receipt.max_adjusted_close_difference_bps is None
                or receipt.adjusted_close_mismatch_count is None
            ):
                return None
            receipt_payload = {
                key: value for key, value in receipt.__dict__.items() if key != "receipt_hash"
            }
            if _canonical_hash(receipt_payload) != receipt.receipt_hash:
                return None
            if receipt.mapping_revision != self._market_data._mapping_revision(
                connection,
                listing_id=listing_id,
                provider=provider,
                as_of_session=requested_as_of,
            ):
                return None
            if receipt.raw_evidence_hash != self._market_data._raw_evidence_hash(
                connection,
                listing_id=listing_id,
                provider=provider,
                history_start=receipt.history_start,
                history_end=receipt.history_end,
            ):
                return None
            actions = self._market_data._current_actions(connection, listing_id, provider)
            if receipt.action_set_hash != action_set_hash(
                actions
            ) or receipt.action_evidence_hash != self._market_data._action_evidence_hash(
                connection, listing_id=listing_id, provider=provider
            ):
                return None
            raw_rows = connection.execute(
                """
                SELECT listing_id, provider, session_date, open, high, low, close, volume
                FROM raw_daily_bar_current
                WHERE listing_id = ? AND provider = ? AND session_date BETWEEN ? AND ?
                ORDER BY session_date
                """,
                [listing_id, provider, receipt.history_start, receipt.history_end],
            ).fetchall()
            adjusted_rows = connection.execute(
                """
                SELECT session_date, adjusted_close
                FROM provider_adjusted_close_current
                WHERE listing_id = ? AND provider = ? AND session_date BETWEEN ? AND ?
                ORDER BY session_date
                """,
                [listing_id, provider, receipt.history_start, receipt.history_end],
            ).fetchall()
            bars = tuple(RawDailyBar(*item) for item in raw_rows)
            adjusted = tuple(
                ProviderAdjustedClosePoint(listing_id, provider, session, float(value))
                for session, value in adjusted_rows
            )
            if (
                not bars
                or {item.session_date for item in bars} != {item.session_date for item in adjusted}
                or provider_adjusted_close_evidence_hash(adjusted)
                != receipt.provider_adjusted_close_evidence_hash
            ):
                return None
            scoped_actions = tuple(
                item
                for item in actions
                if receipt.history_start <= item.effective_date <= receipt.history_end
            )
            diagnostics = provider_adjusted_ratio_diagnostics(
                bars,
                scoped_actions,
                adjusted,
                daily_price_basis=manifest.profile.daily_price_basis,
            )
            mismatches = tuple(item for item in diagnostics if item.difference_bps > 5.0)
            if (
                receipt.max_adjusted_close_difference_bps
                != max((item.difference_bps for item in diagnostics), default=0.0)
                or receipt.adjusted_close_mismatch_count != len(mismatches)
                or receipt.first_adjusted_close_mismatch_session
                != (mismatches[0].session_date if mismatches else None)
            ):
                return None
            all_rows = connection.execute(
                """
                SELECT session_date, close FROM raw_daily_bar_current
                WHERE listing_id = ? AND provider = ? AND session_date <= ?
                ORDER BY session_date
                """,
                [listing_id, provider, requested_as_of],
            ).fetchall()
            if not all_rows or all_rows[-1][0] != requested_as_of:
                return None
            revision = _canonical_hash(
                {
                    "reference": reference_id,
                    "raw_sessions": [
                        (session.isoformat(), float(close)) for session, close in all_rows
                    ],
                    "action_receipt": receipt.receipt_hash,
                }
            )
            return current if revision == current["revision_hash"] else None
        finally:
            connection.close()

    def consume_workspace_tokens(
        self,
        *,
        bucket_id: str,
        requested: int,
        now: datetime,
        capacity: int = 20,
        refill_every_seconds: int = 2,
    ) -> tuple[bool, datetime]:
        """Consume a serial provider budget without sleeping or queues."""
        if requested != 1 or capacity != 20 or refill_every_seconds != 2:
            raise ValueError(
                "sector token policy is fixed at 20 tokens and one token per two seconds"
            )
        observed = _utc_naive(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            row = connection.execute(
                """
                SELECT available_tokens, last_refilled_at FROM workspace_token_bucket
                WHERE bucket_id = ?
                """,
                [bucket_id],
            ).fetchone()
            if row is None:
                available = float(capacity)
                last = observed
                connection.execute(
                    """
                    INSERT INTO workspace_token_bucket VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [bucket_id, capacity, available, refill_every_seconds, last, observed],
                )
            else:
                available = float(row[0])
                last = row[1]
                elapsed = max(0.0, (observed - last).total_seconds())
                added = math.floor(elapsed / refill_every_seconds)
                if added:
                    available = min(float(capacity), available + added)
                    last = last + timedelta(seconds=added * refill_every_seconds)
            if available >= 1.0:
                available -= 1.0
                granted = True
            else:
                granted = False
            connection.execute(
                """
                UPDATE workspace_token_bucket
                SET available_tokens = ?, last_refilled_at = ?, updated_at = ?
                WHERE bucket_id = ?
                """,
                [available, last, observed, bucket_id],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        retry_after = now if granted else (last.replace(tzinfo=UTC) + timedelta(seconds=2))
        return granted, retry_after

    def save_sector_progress(
        self,
        *,
        manifest_revision: str,
        cursor_listing_id: str | None,
        status: str,
        deferred_retry_id: str | None,
        retry_after_at: datetime | None,
        observed_at: datetime,
        failure_code: str | None = None,
        observed_workers: int | None = None,
        next_workers: int | None = None,
        transport_policy_hash: str | None = None,
    ) -> None:
        """Persist the resumable sector-reference cursor and transport state."""
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO sector_reference_progress (
                    manifest_revision, cursor_listing_id, status, deferred_retry_id,
                    retry_after_at, failure_code, observed_workers, next_workers,
                    transport_policy_hash, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (manifest_revision) DO UPDATE SET
                    cursor_listing_id = excluded.cursor_listing_id, status = excluded.status,
                    deferred_retry_id = excluded.deferred_retry_id,
                    retry_after_at = excluded.retry_after_at,
                    failure_code = excluded.failure_code,
                    observed_workers = excluded.observed_workers,
                    next_workers = excluded.next_workers,
                    transport_policy_hash = excluded.transport_policy_hash,
                    updated_at = excluded.updated_at
                """,
                [
                    manifest_revision,
                    cursor_listing_id,
                    status,
                    deferred_retry_id,
                    _utc_naive(retry_after_at) if retry_after_at else None,
                    failure_code,
                    observed_workers,
                    next_workers,
                    transport_policy_hash,
                    _utc_naive(observed_at),
                ],
            )
        finally:
            connection.close()

    def sector_progress(self, manifest_revision: str) -> dict[str, object] | None:
        """Return persisted sector-reference progress for a manifest revision."""
        connection = self._connect(read_only=True)
        try:
            result = connection.execute(
                "SELECT * FROM sector_reference_progress WHERE manifest_revision = ?",
                [manifest_revision],
            )
            row = result.fetchone()
            return (
                dict(zip((item[0] for item in result.description), row, strict=True))
                if row is not None
                else None
            )
        finally:
            connection.close()

    def stage_sector_observation(
        self,
        *,
        manifest_revision: str,
        listing_id: str,
        provider: str,
        provider_symbol: str,
        sector_name: str,
        sector_key: str | None,
        payload_hash: str,
        evidence_hash: str,
        observed_at: datetime,
    ) -> None:
        """Persist one accepted fetch so a rate-limit resume continues at its cursor."""
        self.stage_sector_observations(
            manifest_revision=manifest_revision,
            observations=(
                {
                    "listing_id": listing_id,
                    "provider": provider,
                    "provider_symbol": provider_symbol,
                    "sector_name": sector_name,
                    "sector_key": sector_key,
                    "payload_hash": payload_hash,
                    "evidence_hash": evidence_hash,
                },
            ),
            observed_at=observed_at,
        )

    def stage_sector_observations(
        self,
        *,
        manifest_revision: str,
        observations: Sequence[Mapping[str, object]],
        observed_at: datetime,
    ) -> None:
        """Persist a validated sector fan-in without one connection per listing."""
        prepared = tuple(
            (
                manifest_revision,
                str(item["listing_id"]),
                str(item["provider"]),
                str(item["provider_symbol"]),
                str(item["sector_name"]),
                str(item["sector_key"]) if item.get("sector_key") else None,
                str(item["payload_hash"]),
                str(item["evidence_hash"]),
                _utc_naive(observed_at),
            )
            for item in observations
        )
        if not prepared:
            return

        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            # One statement per listing means one index probe per listing for the
            # conflict clause. The rows are offered once instead; `DO UPDATE`
            # cannot resolve the same key twice inside a single statement, so a
            # repeated listing keeps its last observation, which is what the
            # per-row loop left behind.
            collapsed: dict[tuple[object, object], tuple[object, ...]] = {}
            for row in prepared:
                collapsed[(row[0], row[1])] = row
            staged = list(collapsed.values())
            stage_name = "sector_reference_staging_stage"
            connection.register(
                stage_name,
                pa.table(
                    {
                        "manifest_revision": pa.array([str(r[0]) for r in staged]),
                        "listing_id": pa.array([str(r[1]) for r in staged]),
                        "provider": pa.array([str(r[2]) for r in staged]),
                        "provider_symbol": pa.array([str(r[3]) for r in staged]),
                        "sector_name": pa.array([str(r[4]) for r in staged]),
                        "sector_key": pa.array(
                            [None if r[5] is None else str(r[5]) for r in staged]
                        ),
                        "payload_hash": pa.array([str(r[6]) for r in staged]),
                        "evidence_hash": pa.array([str(r[7]) for r in staged]),
                        "retrieved_at": pa.array([r[8] for r in staged], pa.timestamp("us")),
                    }
                ),
            )
            try:
                connection.execute(
                    f"""
                    INSERT INTO sector_reference_staging
                    SELECT manifest_revision, listing_id, provider, provider_symbol,
                           sector_name, sector_key, payload_hash, evidence_hash, retrieved_at
                    FROM {stage_name}
                    ON CONFLICT (manifest_revision, listing_id) DO UPDATE SET
                        provider = excluded.provider,
                        provider_symbol = excluded.provider_symbol,
                        sector_name = excluded.sector_name, sector_key = excluded.sector_key,
                        payload_hash = excluded.payload_hash,
                        evidence_hash = excluded.evidence_hash,
                        retrieved_at = excluded.retrieved_at
                    """
                )
            finally:
                connection.unregister(stage_name)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def staged_sector_observations(self, manifest_revision: str) -> list[dict[str, object]]:
        """Return staged sector observations ordered by listing identifier."""
        connection = self._connect(read_only=True)
        try:
            result = connection.execute(
                """
                SELECT listing_id, provider, provider_symbol, sector_name, sector_key,
                       payload_hash, evidence_hash
                FROM sector_reference_staging WHERE manifest_revision = ? ORDER BY listing_id
                """,
                [manifest_revision],
            )
            names = [column[0] for column in result.description]
            return [dict(zip(names, row, strict=True)) for row in result.fetchall()]
        finally:
            connection.close()

    def reusable_current_sector_observations(
        self,
        manifest: UniverseManifest,
        *,
        observed_at: datetime,
        every_days: int = 30,
    ) -> list[dict[str, object]]:
        """Return fresh, identity-matched observations for a partial revision.

        This is a migration/onboarding optimization, not an activation path.
        Callers may stage the returned evidence and fetch only missing members;
        :meth:`activate_sector_revision` still requires the complete manifest.
        """
        if every_days < 1:
            raise ValueError("sector reuse interval must be positive")
        expected = {item.listing_id: item.provider_symbol for item in manifest.listings}
        connection = self._connect(read_only=True)
        try:
            result = connection.execute(
                """
                SELECT current.listing_id, current.provider, current.provider_symbol,
                       current.sector_name, current.sector_key, current.payload_hash,
                       current.evidence_hash, current.retrieved_at
                FROM sector_classification_current AS current
                JOIN universe_manifest_listing AS member USING (listing_id)
                WHERE member.manifest_id = ? AND current.provider = ?
                ORDER BY current.listing_id
                """,
                [manifest.manifest_id, manifest.profile.provider],
            )
            names = [column[0] for column in result.description]
            rows = [dict(zip(names, row, strict=True)) for row in result.fetchall()]
        finally:
            connection.close()
        cutoff = _utc_naive(observed_at) - timedelta(days=every_days)
        return [
            {key: value for key, value in row.items() if key != "retrieved_at"}
            for row in rows
            if expected.get(str(row["listing_id"])) == str(row["provider_symbol"])
            and row["retrieved_at"] >= cutoff
        ]

    def clear_sector_staging(self, manifest_revision: str) -> None:
        """Remove one consumed staging scope after a governed manifest transition."""
        connection = self._connect()
        try:
            connection.execute(
                "DELETE FROM sector_reference_staging WHERE manifest_revision = ?",
                [manifest_revision],
            )
        finally:
            connection.close()

    def activate_sector_revision(
        self,
        *,
        manifest: UniverseManifest,
        observations: Sequence[Mapping[str, object]],
        observed_at: datetime,
    ) -> tuple[str, bool, str, date | None]:
        """Atomically publish a complete Yahoo-current-sector revision only.

        A listing whose Sector changed is recorded as a reclassification in force from the
        session its update observed it, never on a session the active Panel published (V346);
        with no Panel published yet the new classification is the backfill.

        Returns:
            The revision, whether a listing's payload changed, the store receipt and the
            effective session of this revision's reclassifications (None when it has none).
        """
        expected = {item.listing_id: item for item in manifest.listings}
        if {str(item.get("listing_id")) for item in observations} != set(expected):
            raise ValueError("sector revision must be complete for the active manifest")
        safe_records = tuple(
            {
                "listing_id": str(item["listing_id"]),
                "provider": str(item["provider"]),
                "provider_symbol": str(item["provider_symbol"]),
                "sector_name": str(item["sector_name"]),
                "sector_key": str(item["sector_key"]) if item.get("sector_key") else None,
                "payload_hash": str(item["payload_hash"]),
                "evidence_hash": str(item["evidence_hash"]),
            }
            for item in sorted(observations, key=lambda row: str(row["listing_id"]))
        )
        revision = sector_revision_hash(
            manifest_revision=manifest.revision_sha256,
            observations=safe_records,
        )
        receipt_hash = _canonical_hash(
            {
                "manifest": manifest.revision_sha256,
                "sector_revision": revision,
                "items": len(safe_records),
            }
        )
        observed = _utc_naive(observed_at)
        connection = self._connect()
        changed = False
        try:
            connection.execute("BEGIN TRANSACTION")
            prior_rows = connection.execute(
                """
                SELECT listing_id, payload_hash, sector_revision, sector_name
                FROM sector_classification_current
                WHERE listing_id IN (
                    SELECT listing_id FROM universe_manifest_listing WHERE manifest_id = ?
                )
                """,
                [manifest.manifest_id],
            ).fetchall()
            prior_by_listing = {
                str(row[0]): (str(row[1]), str(row[2]), str(row[3])) for row in prior_rows
            }
            changed = bool(prior_rows) and any(
                prior_by_listing.get(item["listing_id"], (None, None, None))[0]
                != item["payload_hash"]
                for item in safe_records
            )
            reclassified = {
                item["listing_id"]
                for item in safe_records
                if item["listing_id"] in prior_by_listing
                and prior_by_listing[item["listing_id"]][2] != item["sector_name"]
            }
            effective_session: date | None = None
            if reclassified:
                published = connection.execute(
                    "SELECT as_of_session FROM active_feature_panel_binding "
                    "WHERE market_profile_id = ?",
                    [manifest.profile.market_profile_id],
                ).fetchone()
                if published is not None and published[0] is not None:
                    effective_session = sector_effective_session(
                        _utc_aware(observed_at),
                        first_unpublished=published[0] + timedelta(days=1),
                    )
            for item in safe_records:
                prior = prior_by_listing.get(item["listing_id"])
                connection.execute(
                    """
                    INSERT INTO sector_classification_current VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (listing_id, provider) DO UPDATE SET
                        provider_symbol = excluded.provider_symbol,
                        sector_name = excluded.sector_name,
                        sector_key = excluded.sector_key,
                        payload_hash = excluded.payload_hash,
                        evidence_hash = excluded.evidence_hash,
                        sector_revision = excluded.sector_revision,
                        retrieved_at = excluded.retrieved_at
                    """,
                    [
                        item["listing_id"],
                        item["provider"],
                        item["provider_symbol"],
                        item["sector_name"],
                        item["sector_key"],
                        item["payload_hash"],
                        item["evidence_hash"],
                        revision,
                        observed,
                    ],
                )
                if prior is None or prior[0] != item["payload_hash"]:
                    revision_id = _canonical_hash(
                        [
                            item["listing_id"],
                            item["provider"],
                            prior[0] if prior else None,
                            item["payload_hash"],
                            revision,
                        ]
                    )
                    connection.execute(
                        """
                        INSERT INTO sector_classification_revision (
                            revision_id, listing_id, provider, prior_payload_hash,
                            next_payload_hash, sector_revision, revision_kind, retrieved_at,
                            prior_sector_name, sector_name, effective_session
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT DO NOTHING
                        """,
                        [
                            revision_id,
                            item["listing_id"],
                            item["provider"],
                            prior[0] if prior else None,
                            item["payload_hash"],
                            revision,
                            "INSERTED" if prior is None else "CORRECTED",
                            observed,
                            prior[2] if prior else None,
                            item["sector_name"],
                            effective_session if item["listing_id"] in reclassified else None,
                        ],
                    )
            connection.execute(
                """
                INSERT INTO sector_reference_receipt VALUES (?, ?, ?, ?, 'COMPLETED', NULL, ?)
                ON CONFLICT (receipt_hash) DO NOTHING
                """,
                [receipt_hash, manifest.revision_sha256, revision, len(safe_records), observed],
            )
            if changed:
                requirement_id = _canonical_hash(
                    [manifest.revision_sha256, prior_rows[0][2] if prior_rows else None, revision]
                )
                connection.execute(
                    """
                    INSERT INTO sector_panel_rebuild_requirement VALUES (?, ?, ?, ?, 'REQUIRED', ?)
                    ON CONFLICT DO NOTHING
                    """,
                    [
                        requirement_id,
                        manifest.revision_sha256,
                        prior_rows[0][2] if prior_rows else None,
                        revision,
                        observed,
                    ],
                )
            connection.execute(
                "DELETE FROM sector_reference_staging WHERE manifest_revision = ?",
                [manifest.revision_sha256],
            )
            connection.execute(
                """
                INSERT INTO sector_reference_progress (
                    manifest_revision, cursor_listing_id, status, deferred_retry_id,
                    retry_after_at, failure_code, observed_workers, next_workers,
                    transport_policy_hash, updated_at
                ) VALUES (?, NULL, 'COMPLETED', NULL, NULL, NULL, NULL, NULL, NULL, ?)
                ON CONFLICT (manifest_revision) DO UPDATE SET
                    cursor_listing_id = NULL, status = 'COMPLETED', deferred_retry_id = NULL,
                    retry_after_at = NULL, failure_code = NULL, observed_workers = NULL,
                    next_workers = NULL, transport_policy_hash = NULL,
                    updated_at = excluded.updated_at
                """,
                [manifest.revision_sha256, observed],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return revision, changed, receipt_hash, effective_session

    def bindable_sector_evidence(
        self,
        manifest: UniverseManifest,
        *,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> BindableSectorEvidence | None:
        """Return exactly what a rebinding would carry over, without mutating.

        The closure ledger must publish a derived sector map *before* the
        rebinding activates its revision, and it can only do that honestly from
        the same rows under the same construction. Reading them through a
        second query would let the published map and the activated revision
        disagree while both looked well formed.
        """
        expected = {item.listing_id: item.provider_symbol for item in manifest.listings}
        connection = _connection or self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT current.listing_id, current.provider,
                       current.provider_symbol, current.sector_name,
                       current.sector_key, current.payload_hash,
                       current.evidence_hash, current.retrieved_at
                FROM sector_classification_current AS current
                JOIN universe_manifest_listing AS member USING (listing_id)
                WHERE member.manifest_id = ? AND current.provider = ?
                ORDER BY current.listing_id
                """,
                [manifest.manifest_id, manifest.profile.provider],
            ).fetchall()
        finally:
            if _connection is None:
                connection.close()
        if len(rows) != len(expected):
            return None
        if any(
            expected.get(str(row[0])) != str(row[2])
            or not str(row[3]).strip()
            or row[5] is None
            or row[6] is None
            for row in rows
        ):
            return None
        safe_records = tuple(
            {
                "listing_id": str(row[0]),
                "provider": str(row[1]),
                "provider_symbol": str(row[2]),
                "sector_name": str(row[3]),
                "sector_key": str(row[4]) if row[4] is not None else None,
                "payload_hash": str(row[5]),
                "evidence_hash": str(row[6]),
            }
            for row in rows
        )
        revision = sector_revision_hash(
            manifest_revision=manifest.revision_sha256,
            observations=safe_records,
        )
        return BindableSectorEvidence(
            observations=safe_records,
            sector_revision=revision,
            receipt_hash=_canonical_hash(
                {
                    "manifest": manifest.revision_sha256,
                    "sector_revision": revision,
                    "items": len(safe_records),
                }
            ),
            source_observed_at=max(row[7] for row in rows),
        )

    def bind_current_sector_evidence_to_manifest(
        self,
        manifest: UniverseManifest,
    ) -> tuple[str, str] | None:
        """Bind unchanged current-sector evidence to one derived manifest.

        A quality-governance child changes membership authority, not the
        provider observation time.  Reusing the current rows must therefore
        preserve every ``retrieved_at`` value and publish the child receipt at
        the original evidence time.  This method never records classification
        revisions or resets the sector freshness clock.

        The revision it activates is bound to manifest identity, so the closure
        ledger needs a sector map for it. Reach this through
        ``SectorRevisionMapActivationCoordinator.bind_to_manifest`` rather than
        directly, or the ledger will hold no map for the activated revision.
        """
        connection = self._connect()
        transaction_started = False
        try:
            evidence = self.bindable_sector_evidence(manifest, _connection=connection)
            if evidence is None:
                return None
            safe_records = evidence.observations
            revision = evidence.sector_revision
            receipt_hash = evidence.receipt_hash
            source_observed_at = evidence.source_observed_at
            existing = connection.execute(
                """
                SELECT 1 FROM sector_reference_receipt
                WHERE receipt_hash = ? AND manifest_revision = ?
                  AND sector_revision = ? AND status = 'COMPLETED'
                """,
                [receipt_hash, manifest.revision_sha256, revision],
            ).fetchone()
            current_revisions = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT DISTINCT current.sector_revision
                    FROM sector_classification_current AS current
                    JOIN universe_manifest_listing AS member USING (listing_id)
                    WHERE member.manifest_id = ? AND current.provider = ?
                    """,
                    [manifest.manifest_id, manifest.profile.provider],
                ).fetchall()
            }
            if existing is not None and current_revisions == {revision}:
                return revision, receipt_hash

            connection.execute("BEGIN TRANSACTION")
            transaction_started = True
            connection.execute(
                """
                UPDATE sector_classification_current AS current
                SET sector_revision = ?
                FROM universe_manifest_listing AS member
                WHERE member.manifest_id = ?
                  AND current.listing_id = member.listing_id
                  AND current.provider = ?
                """,
                [revision, manifest.manifest_id, manifest.profile.provider],
            )
            connection.execute(
                """
                INSERT INTO sector_reference_receipt
                VALUES (?, ?, ?, ?, 'COMPLETED', NULL, ?)
                ON CONFLICT (receipt_hash) DO NOTHING
                """,
                [
                    receipt_hash,
                    manifest.revision_sha256,
                    revision,
                    len(safe_records),
                    source_observed_at,
                ],
            )
            connection.execute(
                """
                INSERT INTO sector_reference_progress (
                    manifest_revision, cursor_listing_id, status,
                    deferred_retry_id, retry_after_at, failure_code,
                    observed_workers, next_workers, transport_policy_hash,
                    updated_at
                ) VALUES (?, NULL, 'COMPLETED', NULL, NULL, NULL,
                          NULL, NULL, NULL, ?)
                ON CONFLICT (manifest_revision) DO UPDATE SET
                    cursor_listing_id = NULL, status = 'COMPLETED',
                    deferred_retry_id = NULL, retry_after_at = NULL,
                    failure_code = NULL, observed_workers = NULL,
                    next_workers = NULL, transport_policy_hash = NULL,
                    updated_at = excluded.updated_at
                """,
                [manifest.revision_sha256, source_observed_at],
            )
            connection.execute("COMMIT")
            transaction_started = False
        except Exception:
            if transaction_started:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return revision, receipt_hash

    def current_sector_revision(
        self, manifest: UniverseManifest
    ) -> tuple[str, dict[str, str]] | None:
        """Return a complete current sector revision and its listing map."""
        state = self.current_sector_state(manifest)
        if state is None:
            return None
        return state.sector_revision, state.sector_by_listing_id

    def current_sector_state(
        self,
        manifest: UniverseManifest,
        *,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> SectorReferenceState | None:
        """Return sector state only when every manifest listing shares one revision."""
        owns_connection = _connection is None
        connection = _connection or self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT current.listing_id, current.sector_name, current.sector_revision,
                       current.retrieved_at
                FROM sector_classification_current AS current
                JOIN universe_manifest_listing AS members
                  ON current.listing_id = members.listing_id
                WHERE members.manifest_id = ?
                ORDER BY current.listing_id
                """,
                [manifest.manifest_id],
            ).fetchall()
        finally:
            if owns_connection:
                connection.close()
        if len(rows) != len(manifest.listings):
            return None
        revisions = {str(row[2]) for row in rows}
        if len(revisions) != 1:
            return None
        sector_by_listing = {str(row[0]): str(row[1]) for row in rows}
        distribution: dict[str, int] = {}
        for sector in sector_by_listing.values():
            distribution[sector] = distribution.get(sector, 0) + 1
        return SectorReferenceState(
            sector_revision=next(iter(revisions)),
            sector_by_listing_id=sector_by_listing,
            sector_observed_at=_utc_aware(max(row[3] for row in rows)),
            sector_distribution=dict(sorted(distribution.items())),
        )

    def sector_history(
        self, manifest: UniverseManifest, *, listing_ids: Sequence[str] = ()
    ) -> SectorHistory | None:
        """The Sector each session reads, for a manifest's members and these listings (V346).

        Args:
            manifest: The manifest whose current classification the history ends at.
            listing_ids: Listings outside the manifest (a Panel axis's former members).

        Returns:
            The history, or None when the manifest has no complete current classification.
        """
        state = self.current_sector_state(manifest)
        if state is None:
            return None
        current = dict(state.sector_by_listing_id)
        missing = [listing_id for listing_id in listing_ids if listing_id not in current]
        if missing:
            current.update(self.sector_classifications(missing))
        connection = self._connect(read_only=True)
        try:
            # A store sealed before the forward rule (a research input's read-only copy) lacks
            # the columns its writer adds on open; it holds no forward reclassification, as a
            # row written before the rule does (V346, LS1's acceptance).
            columns = {
                str(row[0])
                for row in connection.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'sector_classification_revision'"
                ).fetchall()
            }
            rows = (
                connection.execute(
                    """
                    SELECT listing_id, effective_session, prior_sector_name, sector_name
                    FROM sector_classification_revision
                    WHERE effective_session IS NOT NULL
                    ORDER BY effective_session, listing_id, retrieved_at
                    """
                ).fetchall()
                if {"effective_session", "prior_sector_name", "sector_name"} <= columns
                else []
            )
        finally:
            connection.close()
        return SectorHistory(
            current_revision=state.sector_revision,
            current=current,
            reclassifications=tuple(
                SectorReclassification(
                    listing_id=str(row[0]),
                    effective_session=row[1],
                    prior_sector=str(row[2]),
                    sector=str(row[3]),
                )
                for row in rows
                if str(row[0]) in current
            ),
        )

    def sector_classifications(self, listing_ids: Sequence[str]) -> dict[str, str]:
        """Return the current sector of listings outside the manifest's own state.

        A Panel's calculation axis holds every listing that is a member of
        some session of its history, and a listing that has left the manifest
        keeps its last observed classification here: the current
        classification applied across history, exactly as for a member. An
        axis listing with no row, or with rows from more than one provider,
        is reported absent and the caller refuses rather than guesses.
        """
        if not listing_ids:
            return {}
        marks = ", ".join("?" for _ in listing_ids)
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                f"""
                SELECT listing_id, sector_name, count(*) OVER (PARTITION BY listing_id)
                FROM sector_classification_current
                WHERE listing_id IN ({marks})
                ORDER BY listing_id
                """,
                list(listing_ids),
            ).fetchall()
        finally:
            connection.close()
        return {str(row[0]): str(row[1]) for row in rows if int(row[2]) == 1}

    def sector_revision_refresh_due(
        self, manifest: UniverseManifest, *, observed_at: datetime, every_days: int = 30
    ) -> bool:
        """Return whether the active Yahoo-current-sector revision needs a full check.

        A receipt is evidence of a complete revision, never a guarantee that
        current Yahoo metadata has not changed.  This method deliberately uses
        only a bounded freshness clock; it does not infer stability from raw
        market-bar activity.
        """
        if every_days < 1:
            raise ValueError("sector refresh interval must be positive")
        current = self.current_sector_revision(manifest)
        if current is None:
            return True
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT observed_at FROM sector_reference_receipt
                WHERE manifest_revision = ? AND sector_revision = ? AND status = 'COMPLETED'
                ORDER BY observed_at DESC LIMIT 1
                """,
                [manifest.revision_sha256, current[0]],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return True
        return _utc_naive(observed_at) - row[0] >= timedelta(days=every_days)


def _supersede_unbound_snapshots(
    connection: duckdb.DuckDBPyConnection, *, observed_at: datetime, keeping: str | None = None
) -> None:
    """When a Panel is stale, the one rule: an ACTIVE snapshot no active binding names (V176).

    Registration runs it for the snapshots beside the one it registers (``keeping``), and a
    Feature build after Market Data's bootstrap for all of them.
    """
    connection.execute(
        """
        UPDATE feature_panel_snapshot_manifest AS snapshot
        SET lifecycle = 'SUPERSEDED',
            lifecycle_reason = 'not_current_active_panel',
            lifecycle_updated_at = ?
        WHERE snapshot.snapshot_hash IS DISTINCT FROM ?
          AND COALESCE(snapshot.lifecycle, 'ACTIVE') = 'ACTIVE'
          AND NOT EXISTS (
              SELECT 1 FROM active_feature_panel_binding AS panel
              WHERE panel.panel_binding_hash = snapshot.panel_binding_hash
                AND panel.panel_content_hash = snapshot.panel_content_hash
                AND panel.as_of_session = snapshot.as_of_session
                AND panel.knowledge_cutoff_at = snapshot.knowledge_cutoff_at
                AND panel.temporal_identity_hash = snapshot.temporal_identity_hash
          )
        """,
        [_utc_naive(observed_at), keeping],
    )


class PanelStateRepository(WorkspaceRepository):
    """Own panel snapshots, activation, and input-quality state."""

    def __init__(
        self,
        workspace: Path | WorkspaceDatabase,
        *,
        market_data: MarketDataRepository | None = None,
    ) -> None:
        """Initialize Panel storage with its market-data repository."""
        super().__init__(workspace)
        self._market_data = market_data or MarketDataRepository(self.database)

    @contextmanager
    def panel_write_connection(self, *, memory_limit_bytes: int = 1_000_000_000):
        """Reuse one trusted connection while retaining per-chunk transactions."""
        if memory_limit_bytes < 256_000_000:
            raise ValueError("panel DuckDB memory limit must be at least 256 MB")
        connection = self._connect()
        try:
            connection.execute(f"SET memory_limit = '{int(memory_limit_bytes)}B'")
            yield connection
            checkpoint_workspace_database(connection)
        finally:
            connection.close()

    @contextmanager
    def feature_panel_read_transaction(self):
        """Hold one consistent DuckDB snapshot while immutable chunks are encoded."""
        with self.database.read_transaction() as connection:
            yield connection

    def active_feature_panel(
        self,
        market_profile_id: str,
        *,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> dict[str, object] | None:
        """Return the active Feature Panel binding for a market profile."""
        owns_connection = _connection is None
        connection = _connection or self._connect(read_only=True)
        try:
            result = connection.execute(
                "SELECT * FROM active_feature_panel_binding WHERE market_profile_id = ?",
                [market_profile_id],
            )
            row = result.fetchone()
            return (
                dict(zip((column[0] for column in result.description), row, strict=True))
                if row is not None
                else None
            )
        finally:
            if owns_connection:
                connection.close()

    def panel_source_state_hash(
        self,
        *,
        listing_ids: Sequence[str],
        catalog_hash: str,
        as_of_session: date,
        allow_missing_market_observations: bool = False,
    ) -> str:
        """Hash current base rows and, when admitted, confirmed source absences.

        Missing Feature computation is never evidence of missing market data.
        Only the current partial-coverage policy may carry an absent raw bar as
        an absent Feature row; its kernel still enforces the nominal coverage
        denominator. A recovered source changes this commitment before reuse.
        """
        if not listing_ids:
            raise ValueError("panel source state requires active listings")
        connection = self._connect(read_only=True)
        try:
            placeholders = ", ".join("?" for _ in listing_ids)
            rows = connection.execute(
                f"""
                SELECT listing_id, row_hash
                FROM feature_daily_runtime
                WHERE catalog_hash = ? AND session_date = ?
                  AND listing_id IN ({placeholders})
                ORDER BY listing_id
                """,
                [catalog_hash, as_of_session, *listing_ids],
            ).fetchall()
            missing = tuple(sorted(set(listing_ids) - {str(row[0]) for row in rows}))
            if missing and allow_missing_market_observations:
                has_source = connection.execute(
                    """
                    SELECT DISTINCT listing_id FROM raw_daily_bar_current
                    WHERE session_date = ? AND listing_id IN (SELECT unnest(?))
                    """,
                    [as_of_session, missing],
                ).fetchall()
                if has_source:
                    raise ValueError("feature.panel_member_row_not_materialized")
                rows = sorted([*rows, *((listing_id, None) for listing_id in missing)])
        finally:
            connection.close()
        if len(rows) != len(set(listing_ids)):
            raise ValueError("panel source state is missing an active as-of feature row")
        return _canonical_hash(
            {
                "catalog_hash": catalog_hash,
                "as_of_session": as_of_session.isoformat(),
                "rows": [
                    (str(row[0]), str(row[1]) if row[1] is not None else None) for row in rows
                ],
            }
        )

    def record_panel_materialization(
        self,
        *,
        receipt_hash: str,
        market_profile_id: str,
        manifest_revision: str,
        sector_revision: str,
        catalog_hash: str,
        spy_revision: str,
        policy_hash: str,
        panel_binding_hash: str,
        content: PanelContentIdentity,
        admission_summary: Mapping[str, object],
        temporal_risk: Mapping[str, object],
        source_state_hash: str,
        temporal_identity_hash: str,
        knowledge_cutoff_at: datetime,
        observed_at: datetime,
    ) -> None:
        """Persist a Panel materialization receipt and its content identity."""
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO panel_materialization_receipt (
                    receipt_hash, market_profile_id, manifest_revision, sector_revision,
                    catalog_hash, spy_revision, policy_hash, panel_binding_hash,
                    panel_content_hash, history_start, as_of_session, row_count,
                    availability_count, admission_summary_json, temporal_risk_json,
                    source_state_hash, temporal_identity_hash, knowledge_cutoff_at, observed_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                ) ON CONFLICT (receipt_hash) DO NOTHING
                """,
                [
                    receipt_hash,
                    market_profile_id,
                    manifest_revision,
                    sector_revision,
                    catalog_hash,
                    spy_revision,
                    policy_hash,
                    panel_binding_hash,
                    content.panel_content_hash,
                    content.history_start,
                    content.as_of_session,
                    content.row_count,
                    content.availability_count,
                    json.dumps(
                        admission_summary,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ),
                    json.dumps(temporal_risk, sort_keys=True, separators=(",", ":"), default=str),
                    source_state_hash,
                    temporal_identity_hash,
                    _utc_naive(knowledge_cutoff_at),
                    _utc_naive(observed_at),
                ],
            )
        finally:
            connection.close()

    def panel_availability_rows(
        self,
        *,
        catalog_hash: str,
        policy_hash: str,
        start: date,
        end: date,
        cross_sections: Sequence[PanelCrossSectionRange] | None = None,
        manifest_revision: str | None = None,
        sector_revision: str | None = None,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> list[dict[str, object]]:
        """Availability rows of one Panel, under the key its rows were computed with.

        ``cross_sections`` names, per range of sessions, the cross-section
        identity a Panel under the session-cross-section rule holds there;
        rows are read from the cross-section table. A binding-keyed Panel
        passes its manifest and sector revisions instead and reads the table
        it recorded into. Exactly one key is accepted.
        """
        by_cross_section = cross_sections is not None
        by_binding = manifest_revision is not None or sector_revision is not None
        if by_cross_section == by_binding:
            raise ValueError("Panel availability needs exactly one key: cross-sections or binding")
        owns_connection = _connection is None
        connection = _connection or self._connect(read_only=True)
        try:
            if cross_sections is not None:
                if not cross_sections:
                    return []
                predicate = " OR ".join(
                    "(cross_section_identity = ? AND session_date BETWEEN ? AND ?)"
                    for _ in cross_sections
                )
                parameters: list[object] = [catalog_hash, policy_hash, start, end]
                for item in cross_sections:
                    parameters.extend(
                        [item.cross_section_identity, item.first_session, item.last_session]
                    )
                result = connection.execute(
                    f"""
                    SELECT session_date, factor_id, universe_size, computed_count, coverage,
                           sector_counts_json, status, reason, small_sector_warning,
                           small_sector_names_json, availability_hash,
                           panel_binding_hash, materialization_receipt_hash,
                           cross_section_identity
                    FROM panel_cross_section_availability
                    WHERE catalog_hash = ? AND policy_hash = ?
                      AND session_date BETWEEN ? AND ? AND ({predicate})
                    ORDER BY session_date, factor_id
                    """,
                    parameters,
                )
            else:
                if manifest_revision is None or sector_revision is None:
                    raise ValueError("binding-keyed Panel availability needs both revisions")
                result = connection.execute(
                    """
                    SELECT session_date, factor_id, universe_size, computed_count, coverage,
                           sector_counts_json, status, reason, small_sector_warning,
                           small_sector_names_json, availability_hash,
                           panel_binding_hash, materialization_receipt_hash
                    FROM panel_factor_availability
                    WHERE manifest_revision = ? AND sector_revision = ? AND catalog_hash = ?
                      AND policy_hash = ? AND session_date BETWEEN ? AND ?
                    ORDER BY session_date, factor_id
                    """,
                    [manifest_revision, sector_revision, catalog_hash, policy_hash, start, end],
                )
            names = [column[0] for column in result.description]
            return [dict(zip(names, row, strict=True)) for row in result.fetchall()]
        finally:
            if owns_connection:
                connection.close()

    def register_feature_panel_snapshot(
        self,
        *,
        snapshot_hash: str,
        panel_binding_hash: str,
        panel_content_hash: str,
        manifest_uri: str,
        metadata_hash: str,
        history_start: date,
        as_of_session: date,
        knowledge_cutoff_at: datetime,
        temporal_identity_hash: str,
        active_listing_count: int,
        chunk_count: int,
        observed_at: datetime,
    ) -> None:
        """Register an available Feature Panel snapshot by content hash."""
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO feature_panel_snapshot_manifest (
                    snapshot_hash, panel_binding_hash, panel_content_hash,
                    manifest_uri, metadata_hash,
                    history_start, as_of_session, knowledge_cutoff_at, temporal_identity_hash,
                    active_listing_count, chunk_count, lifecycle, lifecycle_reason,
                    lifecycle_updated_at, physical_availability, eviction_plan_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', NULL, ?, 'AVAILABLE', NULL, ?)
                ON CONFLICT (snapshot_hash) DO UPDATE SET
                    panel_binding_hash = COALESCE(
                        feature_panel_snapshot_manifest.panel_binding_hash,
                        excluded.panel_binding_hash
                    ),
                    physical_availability = 'AVAILABLE',
                    eviction_plan_hash = NULL
                """,
                [
                    snapshot_hash,
                    panel_binding_hash,
                    panel_content_hash,
                    manifest_uri,
                    metadata_hash,
                    history_start,
                    as_of_session,
                    _utc_naive(knowledge_cutoff_at),
                    temporal_identity_hash,
                    active_listing_count,
                    chunk_count,
                    _utc_naive(observed_at),
                    _utc_naive(observed_at),
                ],
            )
            connection.execute(
                """
                UPDATE feature_panel_snapshot_manifest
                SET lifecycle = 'SUPERSEDED',
                    lifecycle_reason = 'replaced_by_newer_snapshot',
                    lifecycle_updated_at = ?
                WHERE snapshot_hash <> ? AND panel_binding_hash = ?
                  AND lifecycle <> 'QUARANTINED'
                """,
                [_utc_naive(observed_at), snapshot_hash, panel_binding_hash],
            )
            # Every other snapshot still ACTIVE is judged against the active
            # bindings now, not at the next workspace open: the projection
            # published right after this registration is what the reader
            # admits by, and it must say what the database says. A snapshot
            # under yesterday's binding is not the current active Panel once
            # today's is registered; its bytes and manifest stay readable to
            # closure, recovery and retention, which never consult lifecycle
            # for that.
            _supersede_unbound_snapshots(connection, observed_at=observed_at, keeping=snapshot_hash)
            connection.execute(
                """
                UPDATE feature_panel_snapshot_manifest AS snapshot
                SET lifecycle = CASE
                        WHEN EXISTS (
                            SELECT 1 FROM active_feature_panel_binding AS panel
                            WHERE panel.panel_binding_hash = snapshot.panel_binding_hash
                              AND panel.panel_content_hash = snapshot.panel_content_hash
                              AND panel.as_of_session = snapshot.as_of_session
                              AND panel.knowledge_cutoff_at = snapshot.knowledge_cutoff_at
                              AND panel.temporal_identity_hash = snapshot.temporal_identity_hash
                        ) THEN 'ACTIVE'
                        ELSE 'SUPERSEDED'
                    END,
                    lifecycle_reason = CASE
                        WHEN EXISTS (
                            SELECT 1 FROM active_feature_panel_binding AS panel
                            WHERE panel.panel_binding_hash = snapshot.panel_binding_hash
                              AND panel.panel_content_hash = snapshot.panel_content_hash
                              AND panel.as_of_session = snapshot.as_of_session
                              AND panel.knowledge_cutoff_at = snapshot.knowledge_cutoff_at
                              AND panel.temporal_identity_hash = snapshot.temporal_identity_hash
                        ) THEN NULL
                        ELSE 'not_current_active_panel'
                    END,
                    lifecycle_updated_at = ?
                WHERE snapshot.snapshot_hash = ? AND snapshot.lifecycle <> 'QUARANTINED'
                """,
                [_utc_naive(observed_at), snapshot_hash],
            )
            WorkspaceReadinessRepository.research_ready_for_snapshot(
                connection, snapshot_hash=snapshot_hash, observed_at=observed_at
            )
        finally:
            connection.close()

    def feature_panel_snapshot_for_active(self, market_profile_id: str) -> dict[str, object] | None:
        """Resolve the immutable snapshot that exactly matches the active panel."""
        connection = self._connect(read_only=True)
        try:
            result = connection.execute(
                """
                SELECT snapshot.snapshot_hash, snapshot.manifest_uri,
                       snapshot.metadata_hash, snapshot.panel_content_hash,
                       snapshot.as_of_session, snapshot.knowledge_cutoff_at,
                       snapshot.temporal_identity_hash,
                       panel.manifest_revision, panel.sector_revision,
                       panel.catalog_hash, panel.spy_revision, panel.policy_hash,
                       panel.panel_binding_hash, panel.temporal_risk_hash
                FROM active_feature_panel_binding AS panel
                JOIN feature_panel_snapshot_manifest AS snapshot
                  ON snapshot.panel_binding_hash = panel.panel_binding_hash
                 AND snapshot.panel_content_hash = panel.panel_content_hash
                 AND snapshot.as_of_session = panel.as_of_session
                 AND snapshot.knowledge_cutoff_at = panel.knowledge_cutoff_at
                 AND snapshot.temporal_identity_hash = panel.temporal_identity_hash
                WHERE panel.market_profile_id = ?
                  AND snapshot.lifecycle = 'ACTIVE'
                ORDER BY snapshot.created_at DESC, snapshot.snapshot_hash DESC
                LIMIT 1
                """,
                [market_profile_id],
            )
            row = result.fetchone()
            if row is None:
                return None
            names = [column[0] for column in result.description]
            return dict(zip(names, row, strict=True))
        finally:
            connection.close()

    def supersede_unbound_feature_panel_snapshots(self, *, observed_at: datetime) -> None:
        """Supersede every ACTIVE snapshot no active binding names: Feature's one rule (V176).

        Args:
            observed_at: When the rule ran, recorded on each row it moves.
        """
        connection = self._connect()
        try:
            _supersede_unbound_snapshots(connection, observed_at=observed_at)
        finally:
            connection.close()

    def feature_panel_snapshot_lifecycles(self) -> tuple[dict[str, object], ...]:
        """Return the bounded lifecycle projection maintained for read admission."""
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT snapshot_hash, lifecycle, lifecycle_reason,
                       physical_availability, eviction_plan_hash
                FROM feature_panel_snapshot_manifest
                ORDER BY snapshot_hash
                """
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            {
                "snapshot_hash": str(snapshot_hash),
                "lifecycle": str(lifecycle),
                "reason": str(reason) if reason is not None else None,
                "physical_availability": str(physical_availability),
                "eviction_plan_hash": (
                    str(eviction_plan_hash) if eviction_plan_hash is not None else None
                ),
            }
            for snapshot_hash, lifecycle, reason, physical_availability, eviction_plan_hash in rows
        )

    def set_feature_panel_snapshot_lifecycle(
        self,
        *,
        snapshot_hash: str,
        lifecycle: str,
        reason: str,
        observed_at: datetime,
    ) -> None:
        """Retire or quarantine one registered snapshot; reactivation is publication-owned."""
        if lifecycle not in {"SUPERSEDED", "QUARANTINED"}:
            raise ValueError("snapshot lifecycle maintenance cannot activate a snapshot")
        if not reason.strip():
            raise ValueError("snapshot lifecycle maintenance requires a reason")
        connection = self._connect()
        try:
            result = connection.execute(
                """
                UPDATE feature_panel_snapshot_manifest
                SET lifecycle = ?, lifecycle_reason = ?, lifecycle_updated_at = ?
                WHERE snapshot_hash = ?
                  AND NOT (lifecycle = 'QUARANTINED' AND ? = 'SUPERSEDED')
                RETURNING snapshot_hash
                """,
                [
                    lifecycle,
                    reason,
                    _utc_naive(observed_at),
                    snapshot_hash,
                    lifecycle,
                ],
            ).fetchone()
            if result is None:
                exists = connection.execute(
                    "SELECT lifecycle FROM feature_panel_snapshot_manifest WHERE snapshot_hash = ?",
                    [snapshot_hash],
                ).fetchone()
                if exists is None:
                    raise KeyError(f"unknown feature panel snapshot: {snapshot_hash}")
                raise ValueError("a quarantined snapshot cannot be downgraded to superseded")
            WorkspaceReadinessRepository.blocked_by_inactive_snapshot(
                connection, snapshot_hash=snapshot_hash, observed_at=observed_at
            )
        finally:
            connection.close()

    def feature_input_quality_disclosure(
        self, *, result_manifest_revision: str
    ) -> dict[str, object]:
        """Return a safe quality-universe disclosure for a panel snapshot."""
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT admission_hash, admitted_listing_count, quarantined_listing_count,
                       admission_json
                FROM feature_input_admission
                WHERE research_manifest_revision = ?
                ORDER BY created_at DESC
                """,
                [result_manifest_revision],
            ).fetchall()
            # One revision can hold admissions of two scopes: a Sector-only
            # exclusion derives the revision, and the Gateway's full-quality
            # admission re-satisfies it as the same manifest (the obligations
            # already name the Gateway's policy). A full-quality admission is
            # what qualifies the revision, whichever was recorded last.
            row = next(
                (
                    item
                    for item in rows
                    if json.loads(str(item[3])).get("evidence_scope", "FULL_QUALITY")
                    == "FULL_QUALITY"
                ),
                rows[0] if rows else None,
            )
            if row is not None:
                payload = json.loads(str(row[3]))
                admitted = int(row[1])
                excluded = int(row[2])
                return {
                    "gateway_qualified": payload.get("evidence_scope", "FULL_QUALITY")
                    == "FULL_QUALITY",
                    "quality_admission_hash": str(row[0]),
                    "candidate_listing_count": admitted + excluded,
                    "admitted_listing_count": admitted,
                    "quality_exclusion_count": excluded,
                    "quarantine_reason_counts": dict(
                        sorted(payload.get("quarantine_reason_counts", {}).items())
                    ),
                    "caveated_listing_count": len(payload.get("caveat_receipts", [])),
                    "caveat_receipts": payload.get("caveat_receipts", []),
                }
            count_row = connection.execute(
                """
                SELECT listing_count FROM universe_manifest WHERE revision_sha256 = ?
                ORDER BY as_of_date DESC LIMIT 1
                """,
                [result_manifest_revision],
            ).fetchone()
        finally:
            connection.close()
        admitted_count = int(count_row[0]) if count_row is not None else 0
        return {
            "gateway_qualified": False,
            "quality_admission_hash": None,
            "candidate_listing_count": admitted_count,
            "admitted_listing_count": admitted_count,
            "quality_exclusion_count": 0,
            "quarantine_reason_counts": {},
        }

    def admits_feature_candidate_subset(
        self,
        *,
        candidate_revision: str,
        result_revision: str,
        as_of_session: date,
        qualification_hash: str | None = None,
        allow_sector_preparation: bool = False,
    ) -> bool:
        """Read a dated candidate admission; Sector proof permits preparation only."""
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """WITH RECURSIVE lineage(revision) AS (
                    SELECT ? UNION SELECT a.research_manifest_revision
                    FROM feature_input_admission a JOIN lineage l
                      ON a.candidate_manifest_revision = l.revision
                )
                SELECT admission_hash, admission_json FROM feature_input_admission
                WHERE candidate_manifest_revision IN (SELECT revision FROM lineage)
                  AND research_manifest_revision = ? AND market_as_of_session = ?""",
                [candidate_revision, result_revision, as_of_session],
            ).fetchall()
        finally:
            connection.close()
        for expected, document in rows:
            payload = json.loads(document)
            actual = payload.pop("admission_hash")
            if actual != expected or _canonical_hash(payload) != expected:
                raise ValueError("feature.baseline_admission_identity_mismatch")
            if (
                allow_sector_preparation
                and qualification_hash is None
                and payload.get("evidence_scope") == "SECTOR_ONLY"
                and payload.get("candidate_manifest_revision") == candidate_revision
            ):
                return True  # Declared preparation, never baseline/ENTRY qualification.
            if (
                payload.get("evidence_scope") == "BASE_FEATURE_CANDIDATES"
                and payload.get("feature_qualification_hash")
                and (
                    qualification_hash is None
                    or payload["feature_qualification_hash"] == qualification_hash
                )
            ):
                return True
        return False

    def feature_panel_reachability_roots(self) -> tuple[str, ...]:
        """Return authority-owned snapshot roots without exposing physical paths."""
        prefix = "playpen://feature-panel/manifests/"
        roots: set[str] = set()

        def collect(value: object) -> None:
            if isinstance(value, str):
                if value.startswith(prefix):
                    roots.add(value)
                return
            if isinstance(value, dict):
                for nested in value.values():
                    collect(nested)
                return
            if isinstance(value, list):
                for nested in value:
                    collect(nested)

        connection = self._connect(read_only=True)
        try:
            roots.update(
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT manifest_uri FROM feature_panel_snapshot_manifest
                    WHERE lifecycle = 'ACTIVE'
                    """
                ).fetchall()
            )
            table_names = {
                str(row[0])
                for row in connection.execute(
                    "SELECT table_name FROM information_schema.tables"
                ).fetchall()
            }
            if "research_task_artifact" in table_names:
                roots.update(
                    str(row[0])
                    for row in connection.execute(
                        """
                        SELECT uri FROM research_task_artifact
                        WHERE uri LIKE 'playpen://feature-panel/manifests/%'
                        """
                    ).fetchall()
                )
            if {"research_task", "research_task_execution"}.issubset(table_names):
                projections = connection.execute(
                    """
                    SELECT execution.status_projection_json
                    FROM research_task_execution AS execution
                    JOIN research_task AS task USING (task_id)
                    WHERE task.lifecycle IN (
                        'running', 'deferred', 'review_pending', 'completed'
                    ) AND execution.status_projection_json IS NOT NULL
                    """
                ).fetchall()
                for (payload,) in projections:
                    collect(json.loads(str(payload)))
        finally:
            connection.close()
        return tuple(sorted(roots))

    def activate_feature_panel(
        self,
        *,
        market_profile_id: str,
        manifest_revision: str,
        sector_revision: str,
        catalog_hash: str,
        spy_revision: str,
        policy_hash: str,
        panel_binding_hash: str,
        panel_content_hash: str,
        history_start: date,
        as_of_session: date,
        materialization_receipt_hash: str,
        admission_summary_hash: str,
        temporal_risk_hash: str,
        temporal_identity_hash: str,
        source_state_hash: str,
        knowledge_cutoff_at: datetime,
        observed_at: datetime,
    ) -> None:
        """Publish one complete panel binding, then and only then mark readiness."""
        observed = _utc_naive(observed_at)
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                INSERT INTO active_feature_panel_binding (
                    market_profile_id, manifest_revision, sector_revision, catalog_hash,
                    spy_revision, policy_hash, panel_hash, panel_binding_hash,
                    panel_content_hash, history_start, as_of_session,
                    materialization_receipt_hash, admission_summary_hash,
                    temporal_risk_hash, temporal_identity_hash, source_state_hash,
                    knowledge_cutoff_at, activated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (market_profile_id) DO UPDATE SET
                    manifest_revision = excluded.manifest_revision,
                    sector_revision = excluded.sector_revision,
                    catalog_hash = excluded.catalog_hash,
                    spy_revision = excluded.spy_revision,
                    policy_hash = excluded.policy_hash,
                    panel_hash = excluded.panel_hash,
                    panel_binding_hash = excluded.panel_binding_hash,
                    panel_content_hash = excluded.panel_content_hash,
                    history_start = excluded.history_start,
                    as_of_session = excluded.as_of_session,
                    materialization_receipt_hash = excluded.materialization_receipt_hash,
                    admission_summary_hash = excluded.admission_summary_hash,
                    temporal_risk_hash = excluded.temporal_risk_hash,
                    temporal_identity_hash = excluded.temporal_identity_hash,
                    source_state_hash = excluded.source_state_hash,
                    knowledge_cutoff_at = excluded.knowledge_cutoff_at,
                    activated_at = excluded.activated_at
                """,
                [
                    market_profile_id,
                    manifest_revision,
                    sector_revision,
                    catalog_hash,
                    spy_revision,
                    policy_hash,
                    panel_binding_hash,
                    panel_binding_hash,
                    panel_content_hash,
                    history_start,
                    as_of_session,
                    materialization_receipt_hash,
                    admission_summary_hash,
                    temporal_risk_hash,
                    temporal_identity_hash,
                    source_state_hash,
                    _utc_naive(knowledge_cutoff_at),
                    observed,
                ],
            )
            WorkspaceReadinessRepository.feature_building(
                connection, market_profile_id=market_profile_id, observed_at=observed
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def freeze_snapshot(
        self,
        manifest: UniverseManifest,
        output_path: Path,
        *,
        as_of_session: date,
        observed_at: datetime,
        requested_listing_ids: Sequence[str] | None = None,
        run_scoped_alias_overrides: Mapping[str, Mapping[str, str]] | None = None,
    ) -> str:
        """Write a feature-only immutable handoff after deterministic admission.

        A provider adapter must first call :meth:`complete_action_audit`.  This
        method only accepts a matching receipt, produces no adjusted table in
        DuckDB, and blocks all listed inputs on known ambiguity or a same-source
        adjusted-close mismatch greater than five basis points.
        """
        observed_at = _utc_naive(observed_at)
        selected = tuple(manifest.listings)
        if requested_listing_ids is not None:
            requested = set(requested_listing_ids)
            self._market_data._assert_manifest_scope(manifest, requested)
            selected = tuple(item for item in selected if item.listing_id in requested)
        if not selected:
            raise ValueError("feature snapshot must request at least one listing")
        rows: list[dict[str, object]] = []
        hashes: set[str] = set()
        receipt_hashes: set[str] = set()
        failures: list[FailureEvidence] = []
        prepared: list[tuple[str, tuple[object, ...], ActionAuditReceipt]] = []
        for listing in selected:
            bars = self._market_data.raw_bars(listing.listing_id, through=as_of_session)
            if not bars:
                failures.append(
                    FailureEvidence(
                        listing.listing_id,
                        manifest.profile.market_profile_id,
                        "DATA_UNREADY",
                        as_of_session,
                        as_of_session,
                        observed_at,
                    )
                )
                continue
            receipt = self._market_data.reusable_action_audit_receipt(
                manifest,
                listing_id=listing.listing_id,
                provider=manifest.profile.provider,
                requested_as_of=as_of_session,
                now=observed_at,
            )
            if receipt is None:
                failures.append(
                    FailureEvidence(
                        listing.listing_id,
                        manifest.profile.market_profile_id,
                        "ACTION_AUDIT_REQUIRED",
                        bars[0].session_date,
                        as_of_session,
                        observed_at,
                    )
                )
                continue
            try:
                actions = self._market_data.actions(listing.listing_id)
                projection = project_research_series(
                    bars,
                    actions,
                    daily_price_basis=manifest.profile.daily_price_basis,
                )
            except FeatureAdmissionBlocked as exc:
                failures.append(
                    FailureEvidence(
                        listing.listing_id,
                        manifest.profile.market_profile_id,
                        exc.code,
                        bars[0].session_date,
                        as_of_session,
                        observed_at,
                    )
                )
                continue
            if receipt.adjusted_close_mismatch_count:
                failures.append(
                    FailureEvidence(
                        listing.listing_id,
                        manifest.profile.market_profile_id,
                        "LISTING_REQUIRES_REVIEW:ADJ_CLOSE_RATIO_MISMATCH",
                        bars[0].session_date,
                        as_of_session,
                        observed_at,
                    )
                )
                continue
            prepared.append((listing.listing_id, projection, receipt))
        if failures:
            self._market_data.record_failures(failures)
            codes = ", ".join(sorted({item.failure_code for item in failures}))
            raise FeatureAdmissionBlocked("FEATURE_ADMISSION_BLOCKED", codes)
        for _listing_id, projection, receipt in prepared:
            hashes.add(projection[0].action_set_hash)
            receipt_hashes.add(receipt.receipt_hash)
            rows.extend(
                {
                    "listing_id": item.listing_id,
                    "session_date": item.session_date,
                    "open_raw": item.open_raw,
                    "high_raw": item.high_raw,
                    "low_raw": item.low_raw,
                    "close_raw": item.close_raw,
                    "volume_raw": item.volume_raw,
                    "cash_dividend": item.cash_dividend,
                    "new_shares_per_old_share": item.new_shares_per_old_share,
                    "split_ratio": item.split_ratio,
                    "open_split_adjusted": item.open_split_adjusted,
                    "high_split_adjusted": item.high_split_adjusted,
                    "low_split_adjusted": item.low_split_adjusted,
                    "close_split_adjusted": item.close_split_adjusted,
                    "volume_split_adjusted": item.volume_split_adjusted,
                    "close_total_return_adjusted": item.close_total_return_adjusted,
                    "action_set_hash": item.action_set_hash,
                    "action_audit_receipt_hash": receipt.receipt_hash,
                    "anchor_session": item.anchor_session,
                    "manifest_revision": manifest.revision_sha256,
                    "daily_price_basis": manifest.profile.daily_price_basis,
                }
                for item in projection
            )
        alias_overrides = {
            str(listing_id): {
                str(key): str(value)
                for key, value in sorted(override.items(), key=lambda item: str(item[0]))
            }
            for listing_id, override in sorted(
                (run_scoped_alias_overrides or {}).items(), key=lambda item: str(item[0])
            )
        }
        alias_override_hash = _canonical_hash(alias_overrides)
        snapshot_hash = _canonical_hash(
            {
                "manifest": manifest.revision_sha256,
                "universe_membership_basis": manifest.universe_membership_basis,
                "is_point_in_time_historical": manifest.is_point_in_time_historical,
                "actions": sorted(hashes),
                "receipts": sorted(receipt_hashes),
                "run_scoped_alias_overrides": alias_overrides,
                "rows": rows,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows)
        metadata = dict(table.schema.metadata or {})
        metadata[b"alphalattice.snapshot_kind"] = b"FeatureInputSnapshot"
        metadata[b"alphalattice.manifest_revision"] = manifest.revision_sha256.encode("utf-8")
        metadata[b"alphalattice.universe_membership_basis"] = (
            manifest.universe_membership_basis.encode("utf-8")
        )
        metadata[b"alphalattice.is_point_in_time_historical"] = (
            b"true" if manifest.is_point_in_time_historical else b"false"
        )
        metadata[b"alphalattice.daily_price_basis"] = manifest.profile.daily_price_basis.encode(
            "utf-8"
        )
        metadata[b"alphalattice.action_set_hashes"] = ",".join(sorted(hashes)).encode("utf-8")
        metadata[b"alphalattice.action_audit_receipts"] = ",".join(sorted(receipt_hashes)).encode(
            "utf-8"
        )
        metadata[b"alphalattice.run_scoped_alias_overrides"] = json.dumps(
            alias_overrides, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        metadata[b"alphalattice.alias_override_hash"] = alias_override_hash.encode("utf-8")
        metadata[b"alphalattice.snapshot_hash"] = snapshot_hash.encode("utf-8")
        pq.write_table(table.replace_schema_metadata(metadata), output_path, compression="zstd")
        return snapshot_hash

    def record_feature_input_gateway_result(
        self,
        *,
        candidate_manifest: UniverseManifest,
        result: object,
        temporal_boundary: object,
        observed_at: datetime,
    ) -> None:
        """Persist only typed Gateway identities and safe summaries.

        The method intentionally accepts the playpen contract as ``object`` so
        this low-level store does not own the Feature Input policy module.  It
        validates the exact expected types locally before opening a transaction.
        """
        from alphalattice.foundation.feature_engine.contracts import TemporalKnowledgeBoundary

        if not isinstance(result, _FeatureInputGatewayResultLike):
            raise TypeError("result is not a FeatureInputGatewayResult")
        if not isinstance(temporal_boundary, TemporalKnowledgeBoundary):
            raise TypeError("temporal boundary has the wrong contract")
        if observed_at.tzinfo is None:
            raise ValueError("feature-input persistence time must be timezone-aware")
        self._market_data.bootstrap(candidate_manifest)
        if result.research_manifest is not None:
            self._market_data.bootstrap(result.research_manifest)
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            for quarantine in result.quarantines:
                if quarantine.continued_from_quarantine_hash is not None:
                    # A continued row is recorded with its continuation record
                    # by ``record_quarantine_continuation`` before the result
                    # is assessed; the result only carries it forward.
                    continue
                self._write_listing_quarantine(
                    connection,
                    candidate_manifest.revision_sha256,
                    quarantine,
                    observed_at=observed_at,
                    effective_session=temporal_boundary.market_as_of_session,
                )
            if result.deferred is not None:
                deferred = result.deferred
                connection.execute(
                    """UPDATE feature_input_provider_deferred
                       SET lifecycle = 'SUPERSEDED', resolved_at = ?
                       WHERE manifest_revision = ? AND lifecycle = 'ACTIVE'
                       AND deferred_retry_id != ?""",
                    [
                        _utc_naive(observed_at),
                        candidate_manifest.revision_sha256,
                        deferred.deferred_retry_id,
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO feature_input_provider_deferred VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, NULL
                    ) ON CONFLICT (deferred_retry_id) DO NOTHING
                    """,
                    [
                        deferred.deferred_retry_id,
                        candidate_manifest.revision_sha256,
                        deferred.provider,
                        deferred.failure_code,
                        json.dumps(deferred.affected_listing_ids, separators=(",", ":")),
                        _utc_naive(deferred.retry_after_at),
                        deferred.observed_workers,
                        deferred.next_workers,
                        deferred.evidence_hash,
                        deferred.policy_hash,
                        _utc_naive(observed_at),
                    ],
                )
            for case in result.agent_cases:
                option_catalog_hash = _canonical_hash(
                    [(option.option_id, option.option_hash) for option in case.options]
                )
                connection.execute(
                    """
                    INSERT INTO feature_input_agent_case (
                        case_token, manifest_revision, case_kind, failure_code, evidence_hash,
                        listing_ids_json, option_catalog_hash, rediagnosis_count, lifecycle,
                        created_at, case_json, case_record_hash
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?
                    ) ON CONFLICT (case_token) DO NOTHING
                    """,
                    [
                        case.case_token,
                        case.manifest_revision,
                        case.case_kind,
                        case.failure_code,
                        case.evidence_hash,
                        json.dumps(case.listing_ids, separators=(",", ":")),
                        option_catalog_hash,
                        case.rediagnosis_count,
                        _utc_naive(observed_at),
                        json.dumps(case.document(), sort_keys=True, separators=(",", ":")),
                        _canonical_hash(case.document()),
                    ],
                )
            # Open cases are cleared by evidence whenever the result names no
            # case for them: an admission, or a refusal on grounds other than
            # a listing's failure (Panel impact, incomplete Sector evidence
            # over failures that are gone).
            cleared_by_refusal = result.deferred is None and bool(result.failure_reasons)
            if (
                result.agent_cases
                or cleared_by_refusal
                or (
                    result.admission is not None
                    and result.admission.evidence_scope == "FULL_QUALITY"
                )
            ):
                tokens = [case.case_token for case in result.agent_cases]
                connection.execute(
                    """UPDATE feature_input_agent_case SET lifecycle = ?
                       WHERE manifest_revision = ? AND lifecycle = 'OPEN'"""
                    + (
                        " AND case_token NOT IN (" + ",".join("?" for _ in tokens) + ")"
                        if tokens
                        else ""
                    ),
                    [
                        "SUPERSEDED" if tokens else "CLEARED_BY_EVIDENCE",
                        candidate_manifest.revision_sha256,
                        *tokens,
                    ],
                )
                connection.execute(
                    """UPDATE feature_input_provider_deferred
                       SET lifecycle = 'RESOLVED', resolved_at = ?
                       WHERE manifest_revision = ? AND lifecycle = 'ACTIVE'""",
                    [_utc_naive(observed_at), candidate_manifest.revision_sha256],
                )
            if result.admission is not None:
                if result.research_manifest is None:
                    raise ValueError("feature-input admission has no research manifest")
                admission = result.admission
                admission_payload = {
                    "candidate_manifest_revision": admission.candidate_manifest_revision,
                    "admitted_listing_ids": admission.admitted_listing_ids,
                    "quarantined_listing_ids": admission.quarantined_listing_ids,
                    "quarantine_reason_counts": admission.quarantine_reason_counts,
                    "quality_policy_hash": admission.quality_policy_hash,
                    "temporal_identity_hash": admission.temporal_identity_hash,
                    "knowledge_cutoff_at": admission.knowledge_cutoff_at.isoformat(),
                    "admission_hash": admission.admission_hash,
                }
                if admission.evidence_scope != "FULL_QUALITY":
                    admission_payload["evidence_scope"] = admission.evidence_scope
                if admission.caveat_receipts:
                    admission_payload["caveat_receipts"] = admission.caveat_receipts
                if admission.feature_qualification_hash is not None:
                    admission_payload["feature_qualification_hash"] = (
                        admission.feature_qualification_hash
                    )
                if admission.nominal_membership_hash is not None:
                    admission_payload["nominal_membership_hash"] = admission.nominal_membership_hash
                connection.execute(
                    """
                    INSERT INTO feature_input_admission VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    ) ON CONFLICT (admission_hash) DO NOTHING
                    """,
                    [
                        admission.admission_hash,
                        admission.candidate_manifest_revision,
                        result.research_manifest.manifest_id,
                        result.research_manifest.revision_sha256,
                        admission.quality_policy_hash,
                        admission.temporal_identity_hash,
                        temporal_boundary.market_as_of_session,
                        _utc_naive(admission.knowledge_cutoff_at),
                        len(admission.admitted_listing_ids),
                        len(admission.quarantined_listing_ids),
                        json.dumps(
                            admission_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        ),
                        _utc_naive(observed_at),
                    ],
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def record_feature_input_delegation(self, document: dict[str, object]) -> None:
        """Persist a Human-issued grant through the Data decision owner.

        Only the same-origin Human operation calls this writer. The hash detects
        accidental record changes; it is not an authorization supplied by a caller.
        """
        grant_hash = str(document["grant_hash"])
        body = {key: value for key, value in document.items() if key != "grant_hash"}
        if _canonical_hash(body) != grant_hash:
            raise ValueError("feature_input.delegation_invalid")
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS feature_input_delegation (
                    grant_hash VARCHAR PRIMARY KEY,
                    case_token VARCHAR NOT NULL,
                    preparation_task_id VARCHAR NOT NULL,
                    document_json VARCHAR NOT NULL,
                    document_hash VARCHAR NOT NULL,
                    revoked_at TIMESTAMP
                )"""
            )
            previous = connection.execute(
                "SELECT document_json FROM feature_input_delegation WHERE grant_hash = ?",
                [grant_hash],
            ).fetchone()
            if previous is None:
                connection.execute(
                    """INSERT INTO feature_input_delegation
                    (grant_hash, case_token, preparation_task_id, document_json, document_hash)
                    VALUES (?, ?, ?, ?, ?)""",
                    [
                        grant_hash,
                        document["case_token"],
                        document["preparation_task_id"],
                        encoded,
                        _canonical_hash(document),
                    ],
                )
            elif previous[0] != encoded:
                raise ValueError("feature_input.delegation_conflict")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def feature_input_delegation_documents(
        self, grant_hash: str | None = None
    ) -> tuple[tuple[dict[str, object], bool], ...]:
        """Read stored grants and revocation state; no external manifest is accepted."""
        if not self.path.is_file():
            return ()
        connection = self._connect(read_only=True)
        try:
            exists = connection.execute(
                """SELECT 1 FROM information_schema.tables
                   WHERE table_name = 'feature_input_delegation' LIMIT 1"""
            ).fetchone()
            if exists is None:
                return ()
            query = """SELECT grant_hash, case_token, preparation_task_id,
                              document_json, document_hash, revoked_at
                       FROM feature_input_delegation"""
            values: list[object] = []
            if grant_hash is not None:
                query += " WHERE grant_hash = ?"
                values.append(grant_hash)
            query += " ORDER BY grant_hash"
            rows = connection.execute(query, values).fetchall()
        finally:
            connection.close()
        documents = []
        for stored_hash, case_token, task_id, encoded, record_hash, revoked_at in rows:
            document = json.loads(encoded)
            if not isinstance(document, dict):
                raise ValueError("feature_input.delegation_record_tampered")
            body = {key: value for key, value in document.items() if key != "grant_hash"}
            if (
                document.get("grant_hash") != stored_hash
                or document.get("case_token") != case_token
                or document.get("preparation_task_id") != task_id
                or _canonical_hash(body) != stored_hash
                or _canonical_hash(document) != record_hash
            ):
                raise ValueError("feature_input.delegation_record_tampered")
            documents.append((document, revoked_at is not None))
        return tuple(documents)

    def revoke_feature_input_delegation(self, grant_hash: str, at: datetime) -> None:
        """Revoke a Feature input delegation, failing if its grant is absent."""
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            result = connection.execute(
                """UPDATE feature_input_delegation
                   SET revoked_at = COALESCE(revoked_at, ?)
                   WHERE grant_hash = ? RETURNING grant_hash""",
                [_utc_naive(at), grant_hash],
            ).fetchone()
            if result is None:
                raise ValueError("feature_input.delegation_not_found")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def feature_input_case_documents(
        self,
        manifest_revision: str | None,
        *,
        lifecycle: Literal["OPEN", "RESOLVED"] = "OPEN",
        limit: int | None = None,
        after_token: str | None = None,
    ) -> tuple[str, ...]:
        """Exact stored catalogs; legacy summary-only cases require a fresh assessment."""
        connection = self._connect(read_only=True)
        try:
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info('feature_input_agent_case')"
                ).fetchall()
            }
            if "case_json" not in columns:
                return ()
            if limit is not None and not 1 <= limit <= 51:
                raise ValueError("feature_input.case_limit_invalid")
            query = """SELECT case_json, case_record_hash, case_token FROM feature_input_agent_case
                       WHERE lifecycle = ? AND case_json IS NOT NULL"""
            values: list[object] = [lifecycle]
            if manifest_revision is not None:
                query += " AND manifest_revision = ?"
                values.append(manifest_revision)
            if after_token is not None:
                query += " AND case_token > ?"
                values.append(after_token)
            query += (
                " ORDER BY created_at DESC, case_token"
                if lifecycle == "RESOLVED"
                else " ORDER BY case_token"
            )
            if limit is not None:
                query += " LIMIT ?"
                values.append(limit)
            rows = connection.execute(query, values).fetchall()
        finally:
            connection.close()
        documents = []
        for document, record_hash, case_token in rows:
            payload = json.loads(document)
            if _canonical_hash(payload) != record_hash or payload.get("case_token") != case_token:
                raise ValueError("feature_input.case_record_tampered")
            documents.append(str(document))
        return tuple(documents)

    def feature_input_decision_for_receipt(
        self, execution_receipt_hash: str
    ) -> tuple[str, dict[str, object]] | None:
        """Return the case and resolution whose executed effect sealed this receipt.

        A quarantine carries the receipt of the decision that authorized it;
        this is the way back from the row to the decision, verified by the
        same record hashes every other reader checks.
        """
        connection = self._connect(read_only=True)
        try:
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info('feature_input_agent_case')"
                ).fetchall()
            }
            if "resolution_json" not in columns:
                return None
            row = connection.execute(
                """SELECT case_json, case_record_hash, case_token, resolution_json, resolution_hash
                   FROM feature_input_agent_case
                   WHERE case_json IS NOT NULL AND resolution_json IS NOT NULL
                     AND json_extract_string(resolution_json, '$.effect.execution_receipt_hash') = ?
                   ORDER BY created_at, case_token LIMIT 1""",
                [execution_receipt_hash],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        case_payload = json.loads(row[0])
        if _canonical_hash(case_payload) != row[1] or case_payload.get("case_token") != row[2]:
            raise ValueError("feature_input.case_record_tampered")
        resolution = json.loads(row[3])
        if _canonical_hash(resolution) != row[4]:
            raise ValueError("feature_input.resolution_tampered")
        return str(row[0]), dict(resolution)

    def feature_input_raw_retention_decisions(self) -> tuple[tuple[str, dict[str, object]], ...]:
        """Read verified raw-retention cases and resolutions across manifest revisions."""
        # A workspace before data preparation holds no retention decisions. Existing
        # stores still cross the normal strict read and verification boundary.
        if not self.path.exists():
            return ()
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """SELECT case_json, case_record_hash, case_token, resolution_json, resolution_hash
                   FROM feature_input_agent_case
                   WHERE case_json IS NOT NULL AND resolution_json IS NOT NULL
                     AND json_extract_string(resolution_json, '$.effect.status') = ?
                   ORDER BY created_at, case_token""",
                ["raw_value_retained_with_caveat"],
            ).fetchall()
        finally:
            connection.close()
        decisions = []
        for case_document, case_hash, case_token, resolution_document, resolution_hash in rows:
            case_payload = json.loads(case_document)
            if (
                _canonical_hash(case_payload) != case_hash
                or case_payload.get("case_token") != case_token
            ):
                raise ValueError("feature_input.case_record_tampered")
            resolution = json.loads(resolution_document)
            if _canonical_hash(resolution) != resolution_hash:
                raise ValueError("feature_input.resolution_tampered")
            decisions.append((str(case_document), dict(resolution)))
        return tuple(decisions)

    def feature_input_resolution(self, case_token: str) -> dict[str, object] | None:
        """Return a hash-verified resolution for a Feature input case."""
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """SELECT resolution_json, resolution_hash FROM feature_input_agent_case
                   WHERE case_token = ?""",
                [case_token],
            ).fetchone()
        finally:
            connection.close()
        if row is None or row[0] is None:
            return None
        payload = json.loads(row[0])
        if _canonical_hash(payload) != row[1]:
            raise ValueError("feature_input.resolution_tampered")
        return dict(payload)

    def record_feature_input_resolution(self, case_token: str, receipt: dict[str, object]) -> None:
        """Store the Host-sealed choice, not a claim that its effect has run."""
        self._record_feature_input_resolution(case_token, "receipt", receipt)

    def record_feature_input_effect(self, case_token: str, effect: dict[str, object]) -> None:
        """Keep the decision intact and bind its one completed deterministic effect."""
        self._record_feature_input_resolution(case_token, "effect", effect)

    def _record_feature_input_resolution(
        self, case_token: str, field: Literal["receipt", "effect"], value: dict[str, object]
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            row = connection.execute(
                """SELECT lifecycle, resolution_json, resolution_hash FROM feature_input_agent_case
                   WHERE case_token = ? AND case_json IS NOT NULL""",
                [case_token],
            ).fetchone()
            if row is None:
                raise ValueError("feature_input.case_not_available")
            payload = json.loads(row[1]) if row[1] is not None else {}
            if row[1] is not None and _canonical_hash(payload) != row[2]:
                raise ValueError("feature_input.resolution_tampered")
            # A refused option and an elapsed wait are decisions that ran
            # their course without resolving the case; a new choice archives
            # them, it does not conflict with them. (Whether a wait has
            # elapsed is the application's to judge before it records.)
            if (
                field == "receipt"
                and payload.get("receipt") != value
                and (
                    payload.get("effect", {}).get("failure_reasons")
                    or payload.get("effect", {}).get("retry_after_at")
                )
            ):
                payload.setdefault("prior_decisions", []).append(
                    {"receipt": payload.pop("receipt"), "effect": payload.pop("effect")}
                )
            if field in payload and payload[field] != value:
                raise ValueError("feature_input.resolution_conflict:" + field)
            if field == "effect" and "receipt" not in payload:
                raise ValueError("feature_input.resolution_absent")
            if field == "receipt" and field not in payload and row[0] != "OPEN":
                raise ValueError("feature_input.case_no_longer_pending")
            if field not in payload:
                payload[field] = value
                connection.execute(
                    """UPDATE feature_input_agent_case
                       SET resolution_json = ?, resolution_hash = ?, lifecycle = ?
                       WHERE case_token = ?""",
                    [
                        json.dumps(payload, sort_keys=True, separators=(",", ":")),
                        _canonical_hash(payload),
                        "RESOLVED"
                        if field == "effect"
                        and value.get("status")
                        in {
                            "raw_value_retained_with_caveat",
                            "quarantine_ready",
                            "requalification_ready",
                        }
                        and not value.get("failure_reasons")
                        else "OPEN",
                        case_token,
                    ],
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def feature_input_effect_documents(self, manifest_revision: str) -> tuple[str, ...]:
        """Return verified effect documents for a manifest revision."""
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """SELECT resolution_json, resolution_hash FROM feature_input_agent_case
                   WHERE manifest_revision = ? AND resolution_json IS NOT NULL
                   ORDER BY created_at, case_token""",
                [manifest_revision],
            ).fetchall()
        finally:
            connection.close()
        effects = []
        for document, record_hash in rows:
            payload = json.loads(document)
            if _canonical_hash(payload) != record_hash:
                raise ValueError("feature_input.resolution_tampered")
            if payload.get("effect") is not None:
                effects.append(json.dumps(payload["effect"], sort_keys=True, separators=(",", ":")))
        return tuple(effects)

    def has_unresolved_feature_input(self, manifest_revision: str) -> bool:
        """Check for an unresolved incident despite any previous admission."""
        connection = self._connect(read_only=True)
        try:
            return (
                connection.execute(
                    """SELECT 1 FROM feature_input_agent_case
                   WHERE manifest_revision = ? AND lifecycle = 'OPEN'
                   UNION ALL SELECT 1 FROM feature_input_provider_deferred
                   WHERE manifest_revision = ? AND lifecycle = 'ACTIVE' LIMIT 1""",
                    [manifest_revision, manifest_revision],
                ).fetchone()
                is not None
            )
        finally:
            connection.close()

    def active_listing_quarantines(
        self,
        candidate_manifest_revision: str,
        *,
        include_profile_history: bool = False,
        qualification_domain: QualificationDomain | None = None,
    ) -> tuple[ListingQuarantine, ...]:
        """Return recoverable quarantine contracts without exposing SQL upstream."""
        connection = self._connect(read_only=True)
        try:
            chained = self._quarantine_continuation_columns(connection)
            rows = connection.execute(
                f"""
                SELECT listing_id, reason_codes_json, evidence_hash, agent_proposal_hash,
                       execution_receipt_hash, recheck_after_at, quarantine_hash,
                       {"continued_from_quarantine_hash" if chained else "NULL"}
                FROM listing_quarantine
                WHERE (candidate_manifest_revision = ? OR (? AND candidate_manifest_revision IN (
                    SELECT old.revision_sha256 FROM universe_manifest old
                    JOIN universe_manifest current USING (market_profile_id)
                    WHERE current.revision_sha256 = ?
                ))) AND lifecycle = 'ACTIVE'
                ORDER BY listing_id, created_at DESC, quarantine_hash DESC
                """,
                [candidate_manifest_revision, include_profile_history, candidate_manifest_revision],
            ).fetchall()
        finally:
            connection.close()
        latest: dict[str, ListingQuarantine] = {}
        for row in rows:
            listing_id = str(row[0])
            reasons = tuple(json.loads(row[1]))
            if (
                qualification_domain is not None
                and quarantine_qualification_domain(reasons) != qualification_domain
            ):
                continue
            if listing_id in latest:
                continue
            latest[listing_id] = self._quarantine_from_row(row)
        return tuple(latest[key] for key in sorted(latest))

    @staticmethod
    def _quarantine_continuation_columns(connection: duckdb.DuckDBPyConnection) -> bool:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info('listing_quarantine')").fetchall()
        }
        return "continued_from_quarantine_hash" in columns and "continuation_json" in columns

    @staticmethod
    def _quarantine_from_row(row: tuple[Any, ...]) -> ListingQuarantine:
        return ListingQuarantine(
            listing_id=str(row[0]),
            reason_codes=tuple(json.loads(row[1])),
            evidence_hash=str(row[2]),
            agent_proposal_hash=str(row[3]) if row[3] is not None else None,
            execution_receipt_hash=str(row[4]),
            recheck_after_at=_utc_aware(row[5]),
            quarantine_hash=str(row[6]),
            continued_from_quarantine_hash=str(row[7]) if row[7] is not None else None,
        )

    def quarantine_lineage(
        self, *, market_profile_id: str, listing_id: str
    ) -> tuple[tuple[ListingQuarantine, str, str | None], ...]:
        """Every quarantine row of one listing in the profile, oldest first.

        Each item is the row, its lifecycle, and the continuation document
        the row carries when it continues another row (a JSON string the
        Gateway's record reads back and verifies). The readback chains them by
        ``continued_from_quarantine_hash``.
        """
        connection = self._connect(read_only=True)
        try:
            chained = self._quarantine_continuation_columns(connection)
            rows = connection.execute(
                f"""
                SELECT listing_id, reason_codes_json, evidence_hash, agent_proposal_hash,
                       execution_receipt_hash, recheck_after_at, quarantine_hash,
                       {"continued_from_quarantine_hash" if chained else "NULL"},
                       lifecycle, {"continuation_json" if chained else "NULL"}
                FROM listing_quarantine
                WHERE listing_id = ? AND candidate_manifest_revision IN (
                    SELECT revision_sha256 FROM universe_manifest WHERE market_profile_id = ?)
                ORDER BY created_at, quarantine_hash
                """,
                [listing_id, market_profile_id],
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            (
                self._quarantine_from_row(row),
                str(row[8]),
                str(row[9]) if row[9] is not None else None,
            )
            for row in rows
        )

    def panel_source_exclusions(
        self, *, market_profile_id: str, sessions: Sequence[date], listing_ids: Sequence[str]
    ) -> tuple[PanelSourceExclusion, ...]:
        """Freeze the dated quality projection, independently of nominal membership."""
        if not sessions or not listing_ids:
            return ()
        connection = self._connect(read_only=True)
        try:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info('listing_quarantine')").fetchall()
            }
            dated = "effective_session" in columns
            rows = connection.execute(
                f"""SELECT q.listing_id, q.reason_codes_json, q.quarantine_hash, q.lifecycle,
                           {"q.effective_session" if dated else "NULL::DATE"},
                           {"q.cleared_effective_session" if dated else "NULL::DATE"}, q.cleared_at
                FROM listing_quarantine q
                WHERE q.listing_id IN (SELECT unnest(?))
                  AND q.candidate_manifest_revision IN (
                    SELECT revision_sha256 FROM universe_manifest WHERE market_profile_id = ?)
                ORDER BY q.created_at, q.quarantine_hash""",
                [tuple(listing_ids), market_profile_id],
            ).fetchall()
        finally:
            connection.close()
        result = []
        for listing, reasons, identity, lifecycle, first, until, cleared_at in rows:
            reasons = tuple(json.loads(reasons))
            if first is None:
                if lifecycle != "ACTIVE":
                    continue  # An undated historical clearance asserts no invented interval.
                first = sessions[-1]
                reasons = (*reasons, "LEGACY_UNDATED_QUALITY_SCOPE")
            if until is None and cleared_at is not None:
                until = _utc_aware(cleared_at).date()
                reasons = (*reasons, "LEGACY_QUALITY_CLEARANCE_DATE")
            begin = max(first, sessions[0])
            end = (
                min(sessions[-1], until - timedelta(days=1)) if until is not None else sessions[-1]
            )
            if begin <= end:
                result.append(
                    PanelSourceExclusion(str(listing), begin, end, str(identity), reasons)
                )
        return tuple(result)

    @staticmethod
    def _write_listing_quarantine(
        connection: duckdb.DuckDBPyConnection,
        candidate_manifest_revision: str,
        item: ListingQuarantine,
        *,
        observed_at: datetime,
        effective_session: date | None,
        continuation_json: str | None = None,
    ) -> None:
        if (item.continued_from_quarantine_hash is None) != (continuation_json is None):
            raise ValueError("a continued quarantine carries its continuation record")
        connection.execute(
            """INSERT INTO listing_quarantine (
                quarantine_hash, candidate_manifest_revision, listing_id, reason_codes_json,
                evidence_hash, agent_proposal_hash, execution_receipt_hash, recheck_after_at,
                lifecycle, created_at, cleared_at, qualification_receipt_hash,
                effective_session, cleared_effective_session,
                continued_from_quarantine_hash, continuation_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, NULL, NULL, ?, NULL, ?, ?)
            ON CONFLICT (quarantine_hash) DO UPDATE SET
                effective_session = coalesce(
                    listing_quarantine.effective_session, excluded.effective_session)
            """,
            [
                item.quarantine_hash,
                candidate_manifest_revision,
                item.listing_id,
                json.dumps(item.reason_codes, separators=(",", ":")),
                item.evidence_hash,
                item.agent_proposal_hash,
                item.execution_receipt_hash,
                _utc_naive(item.recheck_after_at),
                _utc_naive(observed_at),
                effective_session,
                item.continued_from_quarantine_hash,
                continuation_json,
            ],
        )

    def record_quarantine_continuation(
        self,
        *,
        candidate_manifest_revision: str,
        continuation: object,
        observed_at: datetime,
        effective_session: date | None = None,
    ) -> None:
        """Append one continued quarantine row chained to the active row it continues.

        The continued row keeps the original execution receipt; the row it
        continues stays as recorded (its own hash covers its recheck time, so
        nothing is rewritten) and is cleared with the chain by the same
        requalification. Idempotent by the continued row's hash.
        """
        if not isinstance(continuation, _QuarantineContinuationLike):
            raise TypeError("continuation is not a QuarantineContinuation")
        item = continuation.quarantine
        document = continuation.document()
        if (
            item.continued_from_quarantine_hash != continuation.continued_from_quarantine_hash
            or document.get("continuation_hash") != continuation.continuation_hash
            or not re.fullmatch(r"[0-9a-f]{64}", continuation.continuation_hash)
        ):
            raise ValueError("feature_input.continuation_identity_mismatch")
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            previous = connection.execute(
                """SELECT lifecycle, listing_id, execution_receipt_hash FROM listing_quarantine
                   WHERE quarantine_hash = ?""",
                [continuation.continued_from_quarantine_hash],
            ).fetchone()
            if (
                previous is None
                or previous[0] != "ACTIVE"
                or previous[1] != item.listing_id
                or previous[2] != item.execution_receipt_hash
            ):
                raise ValueError("feature_input.continuation_source_not_active")
            self._write_listing_quarantine(
                connection,
                candidate_manifest_revision,
                item,
                observed_at=observed_at,
                effective_session=effective_session,
                continuation_json=json.dumps(document, sort_keys=True, separators=(",", ":")),
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def record_listing_quarantines(
        self,
        *,
        candidate_manifest_revision: str,
        quarantines: Sequence[object],
        observed_at: datetime,
        effective_session: date | None = None,
    ) -> None:
        """Append recoverable quarantine facts; never alter manifest membership."""
        if not quarantines or any(not isinstance(item, ListingQuarantine) for item in quarantines):
            raise ValueError("listing quarantine write requires typed non-empty input")
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            for item in quarantines:
                assert isinstance(item, ListingQuarantine)
                self._write_listing_quarantine(
                    connection,
                    candidate_manifest_revision,
                    item,
                    observed_at=observed_at,
                    effective_session=effective_session,
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def clear_listing_quarantine(
        self,
        *,
        quarantine_hash: str,
        qualification_receipt_hash: str,
        cleared_at: datetime,
        effective_session: date | None = None,
        qualification_domain: QualificationDomain | None = None,
    ) -> bool:
        """Clear by immutable identity after deterministic requalification."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT lifecycle, listing_id, candidate_manifest_revision, "
                "effective_session, created_at, reason_codes_json "
                "FROM listing_quarantine WHERE quarantine_hash = ?",
                [quarantine_hash],
            ).fetchone()
            if row is None:
                raise ValueError("listing quarantine does not exist")
            if row[0] == "CLEARED":
                return False
            if (
                qualification_domain is not None
                and quarantine_qualification_domain(tuple(json.loads(row[5])))
                != qualification_domain
            ):
                raise ValueError("feature.quarantine_clearance_domain_mismatch")
            if _utc_naive(cleared_at) < row[4]:
                raise ValueError("quality clearance precedes its observed quarantine")
            clear_session = effective_session or _utc_aware(cleared_at).date()
            if row[3] is not None and clear_session < row[3]:
                raise ValueError("quality clearance precedes the exclusion session")
            candidates = connection.execute(
                """SELECT quarantine_hash, reason_codes_json FROM listing_quarantine
                WHERE listing_id = ? AND lifecycle = 'ACTIVE'
                  AND created_at <= ?
                  AND (effective_session IS NULL OR effective_session <= ?)
                  AND candidate_manifest_revision IN (
                    SELECT prior.revision_sha256 FROM universe_manifest prior
                    JOIN universe_manifest current USING (market_profile_id)
                    WHERE current.revision_sha256 = ?)
                """,
                [
                    row[1],
                    _utc_naive(cleared_at),
                    clear_session,
                    row[2],
                ],
            ).fetchall()
            hashes = [
                str(value[0])
                for value in candidates
                if qualification_domain is None
                or quarantine_qualification_domain(tuple(json.loads(value[1])))
                == qualification_domain
            ]
            connection.execute(
                """UPDATE listing_quarantine SET lifecycle = 'CLEARED', cleared_at = ?,
                          qualification_receipt_hash = ?, cleared_effective_session = ?
                   WHERE lifecycle = 'ACTIVE' AND quarantine_hash IN (SELECT unnest(?))""",
                [_utc_naive(cleared_at), qualification_receipt_hash, clear_session, hashes],
            )
            return True
        finally:
            connection.close()

    def front_desk_quarantine_projections(self) -> dict[str, tuple[tuple[str, ...], datetime]]:
        """Project active quarantine reason/time by symbol without evidence details."""
        if not self.path.exists():
            return {}
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT listing.display_symbol, quarantine.reason_codes_json,
                       quarantine.recheck_after_at, quarantine.created_at
                FROM listing_quarantine AS quarantine
                JOIN listing ON listing.listing_id = quarantine.listing_id
                WHERE quarantine.lifecycle = 'ACTIVE'
                ORDER BY listing.display_symbol, quarantine.created_at DESC
                """
            ).fetchall()
        finally:
            connection.close()
        projections: dict[str, tuple[tuple[str, ...], datetime]] = {}
        for symbol, reasons, recheck, _ in rows:
            normalized = str(symbol)
            if normalized not in projections:
                projections[normalized] = (tuple(json.loads(reasons)), _utc_aware(recheck))
        return projections


__all__ = [
    "FeatureSourceInputs",
    "FeatureStateRepository",
    "PanelStateRepository",
    "projected_feature_rows",
]
