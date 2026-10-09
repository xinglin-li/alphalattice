"""One-workspace DuckDB owner for raw observations, action evidence, and snapshots."""

from __future__ import annotations

import functools
import hashlib
import json
import math
from bisect import bisect_left
from collections.abc import Buffer, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from io import RawIOBase
from itertools import pairwise
from typing import Any, Protocol

import duckdb
import pyarrow as pa

from alphalattice.control.workspace_runtime.database import (
    WorkspaceRepository,
    checkpoint_workspace_database,
)
from alphalattice.control.workspace_runtime.storage.readiness import (
    WorkspaceReadinessRecord,
    WorkspaceReadinessRepository,
)
from alphalattice.control.workspace_runtime.verified_facts import (
    ensure_year_fact_schema,
    forget_year_facts,
    record_year_facts,
    year_facts,
)
from alphalattice.foundation.market_data_ops.publication.projection import (
    ADJUSTED_CLOSE_DIAGNOSTIC_POLICY_HASH,
    action_set_hash,
    provider_adjusted_close_evidence_hash,
    provider_adjusted_ratio_diagnostics,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    DataHealthReport,
    FailureEvidence,
    HealthSubject,
    ProviderAdjustedClosePoint,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.sources.membership import (
    MembershipEvent,
    MembershipSchedule,
    UniverseBootstrapRecord,
    UniverseSourceObservation,
    membership_effective_session,
    resolve_membership_schedule,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import SanitizedBatch

FeaturePersistenceTimingSink = Callable[[str, float], None]
FeaturePersistenceSqlProfiler = Callable[
    [duckdb.DuckDBPyConnection, str], AbstractContextManager[None]
]

_SOURCE_OBSERVATION_TABLE = """
    CREATE TABLE IF NOT EXISTS universe_source_observation (
        observation_hash VARCHAR PRIMARY KEY,
        market_profile_id VARCHAR NOT NULL,
        observed_at TIMESTAMP NOT NULL,
        previous_observed_at TIMESTAMP,
        candidate_membership_hash VARCHAR NOT NULL,
        source_identity_hash VARCHAR NOT NULL,
        first_eligible_session DATE NOT NULL
    )
"""


@contextmanager
def _staged_rows(
    connection: duckdb.DuckDBPyConnection,
    name: str,
    columns: Mapping[str, pa.Array],
) -> Iterator[str]:
    """Offer a bounded row set to DuckDB as one relation.

    DuckDB runs `executemany`, and a Python loop around `execute`, one statement
    per row: measured on a real workspace that is 79 us per row into a plain
    table and 923 us per row when the target carries a key and the statement an
    `ON CONFLICT` clause, because each row then takes its own index probe. The
    same rows offered as one relation cost 3.1 us each. Callers keep their
    conflict clause, so what is inserted and what is skipped is unchanged.
    """
    connection.register(name, pa.table(dict(columns)))
    try:
        yield name
    finally:
        connection.unregister(name)


def _last_by_key(
    items: Sequence[tuple[object, ...]], key_index: int = 0
) -> list[tuple[object, ...]]:
    """Keep the last row per key, which is what `DO UPDATE` in a loop leaves."""
    collapsed: dict[object, tuple[object, ...]] = {}
    for item in items:
        collapsed[item[key_index]] = item
    return list(collapsed.values())


def _first_by_key(
    items: Sequence[tuple[object, ...]], key_index: int = 0
) -> list[tuple[object, ...]]:
    """Keep the first row per key, which is what `DO NOTHING` in a loop leaves."""
    collapsed: dict[object, tuple[object, ...]] = {}
    for item in items:
        collapsed.setdefault(item[key_index], item)
    return list(collapsed.values())


class ActionAuditScopeInsufficient(ValueError):
    """A rolling action audit needs a wider provider observation to reconcile safely."""


def _membership_digest(listing_ids: Iterable[str]) -> str:
    """The digest of a manifest's member listings, in no particular order."""
    return _canonical_hash(sorted(set(listing_ids)))


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _SourcePrefixHashSink(RawIOBase):
    """Hash the original IPC writes without retaining another complete table buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.digest = hashlib.sha256()

    def writable(self) -> bool:
        return True

    def write(self, value: Buffer) -> int:
        view = memoryview(value)
        self.digest.update(view)
        return view.nbytes


def _source_prefix_arrow_hash(table: pa.Table) -> str:
    """Hash normalized source columns without depending on their batch layout."""
    normalized = table.replace_schema_metadata(None).combine_chunks()
    with _SourcePrefixHashSink() as sink:
        with pa.ipc.new_stream(sink, normalized.schema) as writer:
            writer.write_table(normalized)
        return sink.digest.hexdigest()


def _utc_naive(value: datetime) -> datetime:
    """Store timestamps as UTC wall time in DuckDB's portable TIMESTAMP type."""
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _utc_aware(value: datetime) -> datetime:
    """Project stored UTC wall time back into an absolute runtime contract."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _plain_number(value: object) -> str | None:
    """How `_canonical_hash` spells a finite float or an int; None for anything else."""
    if type(value) is float:
        return repr(value) if math.isfinite(value) else None
    if type(value) is int:
        return repr(value)
    return None


@functools.cache
def _plain_text(value: str) -> str | None:
    """How `_canonical_hash` spells a string it writes unescaped; None for one it escapes.

    With ``ensure_ascii=False`` the encoder escapes only a quote, a backslash and a
    control character (one below the space). Cached: a store's listing and provider
    names, a few thousand short strings, repeat on every bar.
    """
    if '"' in value or "\\" in value or any(character < " " for character in value):
        return None
    return f'"{value}"'


def _bar_hash(bar: RawDailyBar) -> str:
    """A raw bar's payload hash: `_canonical_hash` of its eight fields.

    For the shapes a sanitized bar holds -- plain strings, a date, finite floats, an int
    -- the encoder's bytes are written directly rather than by building and encoding a
    mapping per bar (1.25M of them at first use); any other shape is encoded by
    `_canonical_hash` itself.
    """
    listing = _plain_text(bar.listing_id) if type(bar.listing_id) is str else None
    provider = _plain_text(bar.provider) if type(bar.provider) is str else None
    close = _plain_number(bar.close)
    high = _plain_number(bar.high)
    low = _plain_number(bar.low)
    open_ = _plain_number(bar.open)
    volume = _plain_number(bar.volume)
    if (
        listing is None
        or provider is None
        or type(bar.session_date) is not date
        or close is None
        or high is None
        or low is None
        or open_ is None
        or volume is None
    ):
        return _canonical_hash(
            {
                "listing_id": bar.listing_id,
                "provider": bar.provider,
                "session_date": bar.session_date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        )
    payload = (
        f'{{"close":{close},"high":{high},"listing_id":{listing},"low":{low},'
        f'"open":{open_},"provider":{provider},'
        f'"session_date":"{bar.session_date.isoformat()}","volume":{volume}}}'
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _action_hash(event: CorporateActionEvent) -> str:
    return _canonical_hash(
        {
            "listing_id": event.listing_id,
            "provider": event.provider,
            "effective_date": event.effective_date,
            "action_kind": event.action_kind,
            "new_shares_per_old_share": event.new_shares_per_old_share,
            "cash_amount": event.cash_amount,
            "provisional": event.provisional,
            "provenance": event.provenance,
        }
    )


@dataclass(frozen=True)
class ActionAuditReceipt:
    """Seal the evidence and diagnostic scope of one corporate-action audit."""

    receipt_hash: str
    listing_id: str
    provider: str
    manifest_revision: str
    mapping_revision: str
    requested_as_of: date
    history_start: date
    history_end: date
    raw_evidence_hash: str
    action_set_hash: str
    action_evidence_hash: str
    provider_adjusted_close_evidence_hash: str | None
    max_adjusted_close_difference_bps: float | None
    adjusted_close_mismatch_count: int | None
    first_adjusted_close_mismatch_session: date | None
    diagnostic_policy_hash: str | None
    observed_at: datetime


@dataclass(frozen=True)
class ProviderAdjustedSeriesRevision:
    """One series-level receipt; backward restatements never create row revisions."""

    receipt_hash: str
    listing_id: str
    provider: str
    prior_series_hash: str | None
    next_series_hash: str
    scope_start: date
    scope_end: date
    full_history: bool
    changed_value_count: int
    changed_return_sessions: tuple[date, ...]
    session_set_changed: bool
    uniform_rescale: bool
    source_receipt_hash: str
    observed_at: datetime


@dataclass(frozen=True)
class FeatureSourceDelta:
    """A bounded, content-addressed source-session delta for feature maintenance."""

    listing_id: str
    sessions: tuple[date, ...]
    evidence_hash: str


@dataclass(frozen=True)
class CurrentUniverseOnboardingListing:
    """Record one listing's progress through current-universe onboarding."""

    onboarding_id: str
    listing_id: str
    symbol: str
    calendar_id: str
    state: str
    failure_code: str | None
    raw_through: date | None
    updated_at: datetime


@dataclass(frozen=True)
class CurrentUniverseOnboardingRun:
    """Track the durable scope and lifecycle of an onboarding run."""

    onboarding_id: str
    candidate_manifest_hash: str
    acquisition_manifest_id: str
    acquisition_manifest_revision: str
    history_start: date
    as_of_session: date
    lifecycle: str
    quality_admission_hash: str | None
    research_manifest_id: str | None
    research_manifest_revision: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class HydrationDeferred:
    """Provider-wide transport pause; never a listing-quality decision."""

    onboarding_id: str
    deferred_retry_id: str
    failure_code: str
    retry_after_at: datetime
    observed_workers: int
    next_workers: int
    affected_listing_set_hash: str
    transport_policy_hash: str
    created_at: datetime


@dataclass(frozen=True)
class CurrentUniverseQualityAdmission:
    """The durable deterministic explanation for one quality-gate decision."""

    onboarding_id: str
    listing_id: str
    eligible: bool
    expected_sessions: int
    observed_sessions: int
    missing_sessions: int
    missing_ratio: float
    maximum_consecutive_gap: int
    reasons: tuple[str, ...]
    evaluated_at: datetime


def _maintenance_retry_grant(
    receipt_hash: Any, wait_until: Any, consumed_at: Any
) -> MaintenanceRetryGrant | None:
    """Read the grant a listing row carries, if any.

    Both the receipt and the wait name the grant (the requeue writes them
    together); a row with only one of them -- or written before the columns
    existed -- holds no grant, so it can neither exempt the unit from the
    attempt budget nor be spent.
    """
    if receipt_hash is None or wait_until is None:
        return None
    return MaintenanceRetryGrant(
        execution_receipt_hash=str(receipt_hash),
        wait_until=_utc_aware(wait_until),
        consumed_at=_utc_aware(consumed_at) if consumed_at is not None else None,
    )


def _table_exists(connection: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(row is not None and int(row[0]) > 0)


ADJUSTED_YEAR = "adjusted_year"
"""The year fact a closed year of a listing's adjusted series is held by: its (session, close)
pairs' digest, recorded by the refresh that read or wrote that year
(`_reconcile_provider_adjusted_series`, the series' only writer)."""
_ADJUSTED_YEAR_BASIS = "provider_adjusted_series.year_blocks.v1"
"""The rule the series and year digests are made by (`_provider_adjusted_series_hash`)."""


def _adjusted_scope(listing_id: str, provider: str) -> str:
    return f"{provider}:{listing_id}"


_RAW_BAR_COLUMNS = (
    "listing_id",
    "provider",
    "session_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
"""A raw bar's columns, in ``RawDailyBar``'s field order."""

_SELECTED_PROVIDER_SQL = """
    WITH selected_provider AS (
        SELECT scope.listing_id, selected.provider
        FROM unnest(?::VARCHAR[]) AS scope(listing_id)
        LEFT JOIN LATERAL (
            SELECT provider FROM provider_symbol_mapping
            WHERE listing_id = scope.listing_id
            ORDER BY effective_from DESC LIMIT 1
        ) AS selected ON TRUE
    )
"""
"""Each listing's latest provider mapping, ``_provider_for_listing``'s rule, for a listing set."""


def _selected_providers(
    connection: duckdb.DuckDBPyConnection, listing_ids: Sequence[str]
) -> dict[str, str | None]:
    """Each listing's latest provider mapping by `_SELECTED_PROVIDER_SQL`, None without one."""
    if not listing_ids:
        return {}
    return {
        str(listing): None if provider is None else str(provider)
        for listing, provider in connection.execute(
            _SELECTED_PROVIDER_SQL + "SELECT listing_id, provider FROM selected_provider",
            [list(listing_ids)],
        ).fetchall()
    }


def _raw_bar_query(
    listing_id: str, *, start: date | None, through: date | None
) -> tuple[str, list[object]]:
    """One listing's raw bars within optional session bounds, in session order."""
    sql = f"SELECT {', '.join(_RAW_BAR_COLUMNS)} FROM raw_daily_bar_current WHERE listing_id = ?"
    params: list[object] = [listing_id]
    if start is not None:
        sql += " AND session_date >= ?"
        params.append(start)
    if through is not None:
        sql += " AND session_date <= ?"
        params.append(through)
    return sql + " ORDER BY session_date", params


def raw_bars_from_table(table: pa.Table) -> tuple[RawDailyBar, ...]:
    """The bars ``MarketDataRepository.raw_bars`` reads, from the same rows read as columns.

    Args:
        table: One listing's raw bars, as ``MarketDataRepository._raw_bar_table`` reads them.

    Returns:
        Bars in session order.
    """
    columns = (table.column(name).to_pylist() for name in _RAW_BAR_COLUMNS)
    return tuple(RawDailyBar(*row) for row in zip(*columns, strict=True))


def _quality_admission(row: tuple[Any, ...]) -> CurrentUniverseQualityAdmission:
    """Convert one stored qualification row into its admission record."""
    return CurrentUniverseQualityAdmission(
        onboarding_id=str(row[0]),
        listing_id=str(row[1]),
        eligible=bool(row[2]),
        expected_sessions=int(row[3]),
        observed_sessions=int(row[4]),
        missing_sessions=int(row[5]),
        missing_ratio=float(row[6]),
        maximum_consecutive_gap=int(row[7]),
        reasons=tuple(str(item) for item in json.loads(str(row[8]))),
        evaluated_at=row[9],
    )


@dataclass(frozen=True)
class QualityGovernanceInputs:
    """What a quality-governance pass reads for a whole listing set, in one read.

    ``admissions``: each listing's newest qualification (the row
    ``latest_quality_admission`` returns), absent when it has none.
    ``raw_close_series``: every listing's raw closes through the target
    session as ``raw_bars`` returns them, projected to
    ``listing_id, session_date, close`` and ordered by listing then session.
    ``actions``: each listing's active actions under its current provider,
    exactly and in the order ``actions`` returns them; a listing without a
    provider mapping is absent, where the per-listing reader raises.
    """

    admissions: Mapping[str, CurrentUniverseQualityAdmission]
    raw_close_series: pa.Table
    actions: Mapping[str, tuple[CorporateActionEvent, ...]]


@dataclass(frozen=True)
class MaintenanceRetryGrant:
    """One attempt past the operation's budget, granted by an elapsed wait.

    Identified by the disposition's execution receipt and the wait instant
    that elapsed: one wait grants one attempt, and the same wait cannot grant
    again. Held until the attempt begins (``consumed_at``); a held grant
    survives batches and restarts because it lives on the listing's row.
    """

    execution_receipt_hash: str
    wait_until: datetime
    consumed_at: datetime | None = None

    @property
    def held(self) -> bool:
        """Report whether this retry grant remains unspent."""
        return self.consumed_at is None


@dataclass(frozen=True)
class CurrentUniverseMaintenanceListing:
    """Record one listing's state in a current-universe maintenance run."""

    maintenance_id: str
    listing_id: str
    symbol: str
    state: str
    failure_code: str | None
    raw_through: date | None
    change_document: dict[str, object] | None
    attempt_count: int
    updated_at: datetime
    retry_grant: MaintenanceRetryGrant | None = None


@dataclass(frozen=True)
class CurrentUniverseMaintenanceRun:
    """Track the durable scope and lifecycle of a maintenance run."""

    maintenance_id: str
    research_manifest_id: str
    research_manifest_revision: str
    as_of_session: date
    lifecycle: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ManifestTransitionRecord:
    """Record a proposed, approved, or activated manifest transition."""

    transition_id: str
    market_profile_id: str
    prior_manifest_revision: str | None
    next_manifest_revision: str | None
    membership_fingerprint: str
    additions: tuple[str, ...]
    removals: tuple[str, ...]
    lifecycle: str
    approved_at: datetime | None
    activated_at: datetime | None
    created_at: datetime


class ActivationStep(Protocol):
    """A caller's step run inside a manifest's activation, writing state its owner defines."""

    def __call__(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        transition_id: str,
        prior_manifest_revision: str | None,
        next_manifest_revision: str,
        at: datetime,
    ) -> None:
        """Write the owner's state for one activated transition, inside the activation's."""


class MarketDataRepository(WorkspaceRepository):
    """A single-process, single-workspace owner of mutable provider observations.

    Fetching and normalization are deliberately outside this owner.  The host
    validates a bounded payload in memory, then this class performs one short
    transaction.  It supplies no multi-process coordination, automatic
    lifecycle mutation, provider arbitration, or agent-controlled SQL.
    """

    @property
    def readiness(self) -> WorkspaceReadinessRepository:
        """Expose the lower-level readiness projection owner without duplicating SQL."""
        return WorkspaceReadinessRepository(self.database)

    def bootstrap(self, manifest: UniverseManifest) -> None:
        """Create or upgrade market-data tables for the supplied manifest.

        Args:
            manifest: Frozen universe whose profile scopes the repository.

        """
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS market_profile (
                    market_profile_id VARCHAR PRIMARY KEY,
                    display_name VARCHAR NOT NULL,
                    market VARCHAR NOT NULL,
                    currency VARCHAR NOT NULL,
                    calendar_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    daily_price_basis VARCHAR NOT NULL,
                    data_validity_class VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                ALTER TABLE market_profile
                ADD COLUMN IF NOT EXISTS daily_price_basis VARCHAR DEFAULT 'split_adjusted'
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS universe_manifest (
                    manifest_id VARCHAR PRIMARY KEY,
                    market_profile_id VARCHAR NOT NULL,
                    as_of_date DATE NOT NULL,
                    revision_sha256 VARCHAR NOT NULL,
                    data_validity_class VARCHAR NOT NULL,
                    listing_count INTEGER NOT NULL,
                    universe_membership_basis VARCHAR NOT NULL DEFAULT 'CURRENT_ACTIVE_SURVIVORS',
                    is_point_in_time_historical BOOLEAN NOT NULL DEFAULT FALSE
                )
                """
            )
            connection.execute(
                """
                ALTER TABLE universe_manifest
                ADD COLUMN IF NOT EXISTS universe_membership_basis VARCHAR
                DEFAULT 'CURRENT_ACTIVE_SURVIVORS'
                """
            )
            connection.execute(
                """
                ALTER TABLE universe_manifest
                ADD COLUMN IF NOT EXISTS is_point_in_time_historical BOOLEAN DEFAULT FALSE
                """
            )
            connection.execute(
                """
                ALTER TABLE universe_manifest
                ADD COLUMN IF NOT EXISTS membership_fingerprint VARCHAR
                """
            )
            connection.execute(
                """
                ALTER TABLE universe_manifest
                ADD COLUMN IF NOT EXISTS qualification_policy_hash VARCHAR
                """
            )
            connection.execute(
                """
                ALTER TABLE universe_manifest
                ADD COLUMN IF NOT EXISTS universe_policy_type VARCHAR
                DEFAULT 'CURRENT_SURVIVOR_COMPOSITE'
                """
            )
            connection.execute(
                """
                ALTER TABLE universe_manifest
                ADD COLUMN IF NOT EXISTS universe_components_json VARCHAR
                DEFAULT '["SP500","NASDAQ100","DJIA"]'
                """
            )
            connection.execute(
                """
                ALTER TABLE universe_manifest
                ADD COLUMN IF NOT EXISTS survivorship_bias_warning BOOLEAN DEFAULT TRUE
                """
            )
            connection.execute(
                """
                ALTER TABLE universe_manifest
                ADD COLUMN IF NOT EXISTS research_use_class VARCHAR
                DEFAULT 'CURRENT_UNIVERSE_RESEARCH_ONLY'
                """
            )
            connection.execute(
                """
                ALTER TABLE universe_manifest
                ADD COLUMN IF NOT EXISTS qualification_obligations_json VARCHAR
                """
            )
            # The members a manifest was registered with, sealed beside its revision, so a
            # read proves the membership it assembles (V269); NULL on a manifest written before.
            connection.execute(
                """
                ALTER TABLE universe_manifest
                ADD COLUMN IF NOT EXISTS membership_sha256 VARCHAR
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS universe_manifest_listing (
                    manifest_id VARCHAR NOT NULL,
                    listing_id VARCHAR NOT NULL,
                    PRIMARY KEY (manifest_id, listing_id)
                )
                """
            )
            # The forward membership history: one bootstrap cohort per
            # profile, then only the events that changed membership. Both are
            # append-only; the per-session interval index is recomputed from
            # them and never stored.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS universe_bootstrap (
                    market_profile_id VARCHAR PRIMARY KEY,
                    t0_session DATE NOT NULL,
                    history_start DATE NOT NULL,
                    cohort_hash VARCHAR NOT NULL,
                    cohort_listing_ids_json VARCHAR NOT NULL,
                    manifest_revision VARCHAR NOT NULL,
                    candidate_manifest_hash VARCHAR NOT NULL,
                    qualification_policy_hash VARCHAR NOT NULL,
                    feature_input_policy_hash VARCHAR NOT NULL,
                    source_observed_at TIMESTAMP NOT NULL,
                    admitted_at TIMESTAMP NOT NULL,
                    panel_snapshot_hash VARCHAR NOT NULL,
                    derivation VARCHAR NOT NULL,
                    initialization_assumption VARCHAR NOT NULL,
                    record_hash VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS universe_membership_event (
                    event_hash VARCHAR PRIMARY KEY,
                    market_profile_id VARCHAR NOT NULL,
                    sequence INTEGER NOT NULL,
                    listing_id VARCHAR NOT NULL,
                    event_kind VARCHAR NOT NULL,
                    effective_session DATE NOT NULL,
                    observed_at TIMESTAMP NOT NULL,
                    decided_at TIMESTAMP NOT NULL,
                    authority VARCHAR NOT NULL,
                    reference_hash VARCHAR NOT NULL,
                    manifest_revision VARCHAR NOT NULL,
                    UNIQUE (market_profile_id, sequence)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS listing (
                    listing_id VARCHAR PRIMARY KEY,
                    market_profile_id VARCHAR NOT NULL,
                    display_symbol VARCHAR NOT NULL,
                    mic VARCHAR NOT NULL,
                    issuer_external_id VARCHAR,
                    UNIQUE (market_profile_id, display_symbol)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_symbol_mapping (
                    listing_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    provider_symbol VARCHAR NOT NULL,
                    effective_from DATE NOT NULL,
                    effective_to DATE,
                    lifecycle_state VARCHAR NOT NULL,
                    PRIMARY KEY (listing_id, provider, effective_from)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_daily_bar_current (
                    listing_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    session_date DATE NOT NULL,
                    open DOUBLE NOT NULL,
                    high DOUBLE NOT NULL,
                    low DOUBLE NOT NULL,
                    close DOUBLE NOT NULL,
                    volume BIGINT NOT NULL,
                    payload_hash VARCHAR NOT NULL,
                    observed_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (listing_id, provider, session_date)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS bar_revision (
                    revision_id VARCHAR PRIMARY KEY,
                    listing_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    session_date DATE NOT NULL,
                    prior_payload_hash VARCHAR NOT NULL,
                    next_payload_hash VARCHAR NOT NULL,
                    observed_at TIMESTAMP NOT NULL,
                    reason VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS corporate_action_current (
                    listing_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    effective_date DATE NOT NULL,
                    action_kind VARCHAR NOT NULL,
                    new_shares_per_old_share DOUBLE,
                    cash_amount DOUBLE,
                    provisional BOOLEAN NOT NULL,
                    provenance VARCHAR NOT NULL,
                    payload_hash VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    observed_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (listing_id, provider, effective_date, action_kind)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS corporate_action_revision (
                    revision_id VARCHAR PRIMARY KEY,
                    listing_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    effective_date DATE NOT NULL,
                    action_kind VARCHAR NOT NULL,
                    revision_kind VARCHAR NOT NULL,
                    prior_payload_hash VARCHAR,
                    next_payload_hash VARCHAR,
                    observed_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS action_audit_receipt (
                    receipt_hash VARCHAR PRIMARY KEY,
                    listing_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    manifest_revision VARCHAR NOT NULL,
                    mapping_revision VARCHAR NOT NULL,
                    requested_as_of DATE NOT NULL,
                    history_start DATE NOT NULL,
                    history_end DATE NOT NULL,
                    raw_evidence_hash VARCHAR NOT NULL,
                    action_set_hash VARCHAR NOT NULL,
                    action_evidence_hash VARCHAR NOT NULL,
                    provider_adjusted_close_evidence_hash VARCHAR,
                    max_adjusted_close_difference_bps DOUBLE,
                    adjusted_close_mismatch_count INTEGER,
                    first_adjusted_close_mismatch_session DATE,
                    diagnostic_policy_hash VARCHAR,
                    observed_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_adjusted_close_current (
                    listing_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    session_date DATE NOT NULL,
                    adjusted_close DOUBLE NOT NULL,
                    payload_hash VARCHAR NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (listing_id, provider, session_date)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_adjusted_series_revision (
                    receipt_hash VARCHAR PRIMARY KEY,
                    listing_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    prior_series_hash VARCHAR,
                    next_series_hash VARCHAR NOT NULL,
                    scope_start DATE NOT NULL,
                    scope_end DATE NOT NULL,
                    full_history BOOLEAN NOT NULL,
                    changed_value_count INTEGER NOT NULL,
                    changed_return_sessions_json VARCHAR NOT NULL,
                    session_set_changed BOOLEAN NOT NULL,
                    uniform_rescale BOOLEAN NOT NULL,
                    source_receipt_hash VARCHAR NOT NULL,
                    observed_at TIMESTAMP NOT NULL
                )
                """
            )
            ensure_year_fact_schema(connection)
            # Old ignored playpen stores may contain a diagnostic column on
            # raw bars and an earlier receipt layout.  Keep them readable, but
            # make all new writes independent from that legacy column.
            connection.execute(
                """
                ALTER TABLE action_audit_receipt
                ADD COLUMN IF NOT EXISTS provider_adjusted_close_evidence_hash VARCHAR
                """
            )
            connection.execute(
                """
                ALTER TABLE action_audit_receipt
                ADD COLUMN IF NOT EXISTS max_adjusted_close_difference_bps DOUBLE
                """
            )
            connection.execute(
                """
                ALTER TABLE action_audit_receipt
                ADD COLUMN IF NOT EXISTS adjusted_close_mismatch_count INTEGER
                """
            )
            connection.execute(
                """
                ALTER TABLE action_audit_receipt
                ADD COLUMN IF NOT EXISTS first_adjusted_close_mismatch_session DATE
                """
            )
            connection.execute(
                """
                ALTER TABLE action_audit_receipt
                ADD COLUMN IF NOT EXISTS diagnostic_policy_hash VARCHAR
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS data_quality (
                    listing_id VARCHAR PRIMARY KEY,
                    latest_session DATE,
                    quality_state VARCHAR NOT NULL,
                    summary VARCHAR NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_attempt (
                    attempt_id VARCHAR PRIMARY KEY,
                    listing_id VARCHAR NOT NULL,
                    failure_code VARCHAR,
                    range_start DATE NOT NULL,
                    range_end DATE NOT NULL,
                    outcome VARCHAR NOT NULL,
                    observed_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS remediation_execution (
                    proposal_hash VARCHAR PRIMARY KEY,
                    option_id VARCHAR NOT NULL,
                    executed_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_input_admission (
                    admission_hash VARCHAR PRIMARY KEY,
                    candidate_manifest_revision VARCHAR NOT NULL,
                    research_manifest_id VARCHAR NOT NULL,
                    research_manifest_revision VARCHAR NOT NULL,
                    quality_policy_hash VARCHAR NOT NULL,
                    temporal_identity_hash VARCHAR NOT NULL,
                    market_as_of_session DATE NOT NULL,
                    knowledge_cutoff_at TIMESTAMP NOT NULL,
                    admitted_listing_count INTEGER NOT NULL,
                    quarantined_listing_count INTEGER NOT NULL,
                    admission_json VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS listing_quarantine (
                    quarantine_hash VARCHAR PRIMARY KEY,
                    candidate_manifest_revision VARCHAR NOT NULL,
                    listing_id VARCHAR NOT NULL,
                    reason_codes_json VARCHAR NOT NULL,
                    evidence_hash VARCHAR NOT NULL,
                    agent_proposal_hash VARCHAR,
                    execution_receipt_hash VARCHAR NOT NULL,
                    recheck_after_at TIMESTAMP NOT NULL,
                    lifecycle VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    cleared_at TIMESTAMP,
                    qualification_receipt_hash VARCHAR
                )
                """
            )
            for definition in (
                "effective_session DATE",
                "cleared_effective_session DATE",
                # A row continuing a standing quarantine names the row it
                # continues and carries the typed continuation record; rows
                # written before continuation existed read back with both
                # absent.
                "continued_from_quarantine_hash VARCHAR",
                "continuation_json VARCHAR",
            ):
                connection.execute(
                    f"ALTER TABLE listing_quarantine ADD COLUMN IF NOT EXISTS {definition}"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_input_agent_case (
                    case_token VARCHAR PRIMARY KEY,
                    manifest_revision VARCHAR NOT NULL,
                    case_kind VARCHAR NOT NULL,
                    failure_code VARCHAR NOT NULL,
                    evidence_hash VARCHAR NOT NULL,
                    listing_ids_json VARCHAR NOT NULL,
                    option_catalog_hash VARCHAR NOT NULL,
                    rediagnosis_count INTEGER NOT NULL,
                    lifecycle VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
                """
            )
            for definition in (
                "case_json VARCHAR",
                "case_record_hash VARCHAR",
                "resolution_json VARCHAR",
                "resolution_hash VARCHAR",
            ):
                connection.execute(
                    f"ALTER TABLE feature_input_agent_case ADD COLUMN IF NOT EXISTS {definition}"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_input_provider_deferred (
                    deferred_retry_id VARCHAR PRIMARY KEY,
                    manifest_revision VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    failure_code VARCHAR NOT NULL,
                    affected_listing_ids_json VARCHAR NOT NULL,
                    retry_after_at TIMESTAMP NOT NULL,
                    observed_workers INTEGER NOT NULL,
                    next_workers INTEGER NOT NULL,
                    evidence_hash VARCHAR NOT NULL,
                    policy_hash VARCHAR NOT NULL,
                    lifecycle VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    resolved_at TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS current_universe_onboarding (
                    onboarding_id VARCHAR PRIMARY KEY,
                    candidate_manifest_hash VARCHAR NOT NULL,
                    candidate_manifest_json VARCHAR NOT NULL,
                    acquisition_manifest_id VARCHAR NOT NULL,
                    acquisition_manifest_revision VARCHAR NOT NULL,
                    history_start DATE NOT NULL,
                    as_of_session DATE NOT NULL,
                    lifecycle VARCHAR NOT NULL,
                    quality_admission_hash VARCHAR,
                    research_manifest_id VARCHAR,
                    research_manifest_revision VARCHAR,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS current_universe_onboarding_listing (
                    onboarding_id VARCHAR NOT NULL,
                    listing_id VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    calendar_id VARCHAR NOT NULL,
                    state VARCHAR NOT NULL,
                    failure_code VARCHAR,
                    raw_through DATE,
                    updated_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (onboarding_id, listing_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS current_universe_hydration_deferred (
                    deferred_retry_id VARCHAR PRIMARY KEY,
                    onboarding_id VARCHAR NOT NULL,
                    failure_code VARCHAR NOT NULL,
                    retry_after_at TIMESTAMP NOT NULL,
                    observed_workers INTEGER NOT NULL,
                    next_workers INTEGER NOT NULL,
                    affected_listing_set_hash VARCHAR NOT NULL,
                    transport_policy_hash VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    resolved_at TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS current_universe_quality_admission (
                    onboarding_id VARCHAR NOT NULL,
                    listing_id VARCHAR NOT NULL,
                    eligible BOOLEAN NOT NULL,
                    expected_sessions INTEGER NOT NULL,
                    observed_sessions INTEGER NOT NULL,
                    missing_sessions INTEGER NOT NULL,
                    missing_ratio DOUBLE NOT NULL,
                    maximum_consecutive_gap INTEGER NOT NULL,
                    reasons_json VARCHAR NOT NULL,
                    evaluated_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (onboarding_id, listing_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS current_universe_maintenance (
                    maintenance_id VARCHAR PRIMARY KEY,
                    research_manifest_id VARCHAR NOT NULL,
                    research_manifest_revision VARCHAR NOT NULL,
                    as_of_session DATE NOT NULL,
                    lifecycle VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS current_universe_maintenance_listing (
                    maintenance_id VARCHAR NOT NULL,
                    listing_id VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    state VARCHAR NOT NULL,
                    failure_code VARCHAR,
                    raw_through DATE,
                    change_json VARCHAR,
                    updated_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (maintenance_id, listing_id)
                )
                """
            )
            connection.execute(
                """
                ALTER TABLE current_universe_maintenance_listing
                ADD COLUMN IF NOT EXISTS change_json VARCHAR
                """
            )
            connection.execute(
                """
                ALTER TABLE current_universe_maintenance_listing
                ADD COLUMN IF NOT EXISTS attempt_count INTEGER DEFAULT 0
                """
            )
            # A governed retry grant lives on the listing's row: the receipt
            # and wait instant that granted it, and when the attempt consumed
            # it. Rows written before grants existed read back without one.
            for definition in (
                "retry_grant_receipt_hash VARCHAR",
                "retry_grant_wait_until TIMESTAMP",
                "retry_grant_consumed_at TIMESTAMP",
            ):
                connection.execute(
                    "ALTER TABLE current_universe_maintenance_listing "
                    f"ADD COLUMN IF NOT EXISTS {definition}"
                )
            self.readiness.ensure_schema(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS manifest_transition (
                    transition_id VARCHAR PRIMARY KEY,
                    market_profile_id VARCHAR NOT NULL,
                    prior_manifest_revision VARCHAR,
                    next_manifest_revision VARCHAR,
                    membership_fingerprint VARCHAR NOT NULL,
                    additions_json VARCHAR NOT NULL,
                    removals_json VARCHAR NOT NULL,
                    lifecycle VARCHAR NOT NULL,
                    approved_at TIMESTAMP,
                    activated_at TIMESTAMP,
                    created_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_universe_rebuild_requirement (
                    transition_id VARCHAR PRIMARY KEY,
                    prior_manifest_revision VARCHAR,
                    next_manifest_revision VARCHAR NOT NULL,
                    lifecycle VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP,
                    panel_snapshot_hash VARCHAR,
                    factor_result_hash VARCHAR,
                    factor_slate_hash VARCHAR,
                    execution_outcome_hash VARCHAR,
                    foundation_hash VARCHAR,
                    revision_marker_hash VARCHAR,
                    failure_code VARCHAR
                )
                """
            )
            for definition in (
                "updated_at TIMESTAMP",
                "panel_snapshot_hash VARCHAR",
                "factor_result_hash VARCHAR",
                "factor_slate_hash VARCHAR",
                "execution_outcome_hash VARCHAR",
                "foundation_hash VARCHAR",
                "revision_marker_hash VARCHAR",
                "failure_code VARCHAR",
            ):
                connection.execute(
                    "ALTER TABLE feature_universe_rebuild_requirement "
                    f"ADD COLUMN IF NOT EXISTS {definition}"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_daily_revision (
                    revision_id VARCHAR PRIMARY KEY,
                    listing_id VARCHAR NOT NULL,
                    session_date DATE NOT NULL,
                    catalog_hash VARCHAR NOT NULL,
                    prior_row_hash VARCHAR NOT NULL,
                    next_row_hash VARCHAR NOT NULL,
                    revision_reason VARCHAR NOT NULL,
                    materialization_receipt_hash VARCHAR NOT NULL,
                    observed_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_ineligibility_run (
                    run_id VARCHAR PRIMARY KEY,
                    listing_id VARCHAR NOT NULL,
                    catalog_hash VARCHAR NOT NULL,
                    factor_id VARCHAR NOT NULL,
                    reason VARCHAR NOT NULL,
                    first_session DATE NOT NULL,
                    last_session DATE NOT NULL,
                    first_observation_count INTEGER NOT NULL,
                    observation_cap INTEGER NOT NULL,
                    materialization_receipt_hash VARCHAR NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_materialization_receipt (
                    receipt_hash VARCHAR PRIMARY KEY,
                    listing_id VARCHAR NOT NULL,
                    range_start DATE NOT NULL,
                    range_end DATE NOT NULL,
                    catalog_hash VARCHAR NOT NULL,
                    raw_input_hash VARCHAR NOT NULL,
                    action_set_hash VARCHAR NOT NULL,
                    market_reference_revision VARCHAR NOT NULL,
                    idempotency_key VARCHAR NOT NULL,
                    work_status VARCHAR NOT NULL,
                    coverage_summary_json VARCHAR NOT NULL,
                    observed_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS market_reference_current (
                    reference_id VARCHAR PRIMARY KEY,
                    listing_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    revision_hash VARCHAR NOT NULL,
                    action_audit_receipt_hash VARCHAR,
                    latest_session DATE,
                    updated_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sector_classification_current (
                    listing_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    provider_symbol VARCHAR NOT NULL,
                    sector_name VARCHAR NOT NULL,
                    sector_key VARCHAR,
                    payload_hash VARCHAR NOT NULL,
                    evidence_hash VARCHAR NOT NULL,
                    sector_revision VARCHAR NOT NULL,
                    retrieved_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (listing_id, provider)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sector_classification_revision (
                    revision_id VARCHAR PRIMARY KEY,
                    listing_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    prior_payload_hash VARCHAR,
                    next_payload_hash VARCHAR NOT NULL,
                    sector_revision VARCHAR NOT NULL,
                    revision_kind VARCHAR NOT NULL,
                    retrieved_at TIMESTAMP NOT NULL
                )
                """
            )
            # A reclassification's Sectors and the session it takes effect from (V346); a row
            # written before the forward rule keeps them NULL and stays in the backfill.
            for definition in (
                "prior_sector_name VARCHAR",
                "sector_name VARCHAR",
                "effective_session DATE",
            ):
                connection.execute(
                    "ALTER TABLE sector_classification_revision "
                    f"ADD COLUMN IF NOT EXISTS {definition}"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sector_reference_receipt (
                    receipt_hash VARCHAR PRIMARY KEY,
                    manifest_revision VARCHAR NOT NULL,
                    sector_revision VARCHAR NOT NULL,
                    listing_count INTEGER NOT NULL,
                    status VARCHAR NOT NULL,
                    cursor_listing_id VARCHAR,
                    observed_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sector_reference_progress (
                    manifest_revision VARCHAR PRIMARY KEY,
                    cursor_listing_id VARCHAR,
                    status VARCHAR NOT NULL,
                    deferred_retry_id VARCHAR,
                    retry_after_at TIMESTAMP,
                    failure_code VARCHAR,
                    observed_workers INTEGER,
                    next_workers INTEGER,
                    transport_policy_hash VARCHAR,
                    updated_at TIMESTAMP NOT NULL
                )
                """
            )
            for definition in (
                "failure_code VARCHAR",
                "observed_workers INTEGER",
                "next_workers INTEGER",
                "transport_policy_hash VARCHAR",
            ):
                connection.execute(
                    f"ALTER TABLE sector_reference_progress ADD COLUMN IF NOT EXISTS {definition}"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sector_reference_staging (
                    manifest_revision VARCHAR NOT NULL,
                    listing_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    provider_symbol VARCHAR NOT NULL,
                    sector_name VARCHAR NOT NULL,
                    sector_key VARCHAR,
                    payload_hash VARCHAR NOT NULL,
                    evidence_hash VARCHAR NOT NULL,
                    retrieved_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (manifest_revision, listing_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_token_bucket (
                    bucket_id VARCHAR PRIMARY KEY,
                    capacity INTEGER NOT NULL,
                    available_tokens DOUBLE NOT NULL,
                    refill_every_seconds INTEGER NOT NULL,
                    last_refilled_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sector_panel_rebuild_requirement (
                    requirement_id VARCHAR PRIMARY KEY,
                    manifest_revision VARCHAR NOT NULL,
                    prior_sector_revision VARCHAR,
                    next_sector_revision VARCHAR NOT NULL,
                    lifecycle VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS panel_factor_availability (
                    manifest_revision VARCHAR NOT NULL,
                    sector_revision VARCHAR NOT NULL,
                    catalog_hash VARCHAR NOT NULL,
                    policy_hash VARCHAR NOT NULL,
                    session_date DATE NOT NULL,
                    factor_id VARCHAR NOT NULL,
                    universe_size INTEGER NOT NULL,
                    computed_count INTEGER NOT NULL,
                    coverage DOUBLE NOT NULL,
                    sector_counts_json VARCHAR NOT NULL,
                    winsor_lower DOUBLE,
                    winsor_upper DOUBLE,
                    residual_median DOUBLE,
                    residual_mad DOUBLE,
                    status VARCHAR NOT NULL,
                    reason VARCHAR,
                    panel_hash VARCHAR NOT NULL,
                    panel_binding_hash VARCHAR NOT NULL,
                    small_sector_warning BOOLEAN NOT NULL,
                    small_sector_names_json VARCHAR NOT NULL,
                    availability_hash VARCHAR NOT NULL,
                    materialization_receipt_hash VARCHAR NOT NULL,
                    PRIMARY KEY (
                        manifest_revision, sector_revision, catalog_hash, policy_hash,
                        session_date, factor_id
                    )
                )
                """
            )
            connection.execute(
                "ALTER TABLE panel_factor_availability "
                "ADD COLUMN IF NOT EXISTS panel_binding_hash VARCHAR"
            )
            connection.execute(
                "ALTER TABLE panel_factor_availability "
                "ADD COLUMN IF NOT EXISTS small_sector_warning BOOLEAN DEFAULT FALSE"
            )
            connection.execute(
                "ALTER TABLE panel_factor_availability "
                "ADD COLUMN IF NOT EXISTS small_sector_names_json VARCHAR DEFAULT '[]'"
            )
            connection.execute(
                "ALTER TABLE panel_factor_availability "
                "ADD COLUMN IF NOT EXISTS availability_hash VARCHAR"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS panel_factor_availability_revision (
                    revision_id VARCHAR PRIMARY KEY,
                    manifest_revision VARCHAR NOT NULL,
                    sector_revision VARCHAR NOT NULL,
                    catalog_hash VARCHAR NOT NULL,
                    policy_hash VARCHAR NOT NULL,
                    session_date DATE NOT NULL,
                    factor_id VARCHAR NOT NULL,
                    prior_availability_hash VARCHAR NOT NULL,
                    next_availability_hash VARCHAR NOT NULL,
                    materialization_receipt_hash VARCHAR NOT NULL,
                    observed_at TIMESTAMP NOT NULL
                )
                """
            )
            # Availability under the session cross-section identity rule: one
            # row per (cross-section, catalog, policy, session, factor), so a
            # membership change at one session leaves every earlier session's
            # availability -- and the partition it describes -- untouched. The
            # binding-keyed tables above stay as written for the snapshots
            # that recorded them; nothing writes them any more.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS panel_cross_section_availability (
                    cross_section_identity VARCHAR NOT NULL,
                    catalog_hash VARCHAR NOT NULL,
                    policy_hash VARCHAR NOT NULL,
                    session_date DATE NOT NULL,
                    factor_id VARCHAR NOT NULL,
                    universe_size INTEGER NOT NULL,
                    computed_count INTEGER NOT NULL,
                    coverage DOUBLE NOT NULL,
                    sector_counts_json VARCHAR NOT NULL,
                    winsor_lower DOUBLE,
                    winsor_upper DOUBLE,
                    residual_median DOUBLE,
                    residual_mad DOUBLE,
                    status VARCHAR NOT NULL,
                    reason VARCHAR,
                    panel_binding_hash VARCHAR NOT NULL,
                    small_sector_warning BOOLEAN NOT NULL,
                    small_sector_names_json VARCHAR NOT NULL,
                    availability_hash VARCHAR NOT NULL,
                    materialization_receipt_hash VARCHAR NOT NULL,
                    PRIMARY KEY (
                        cross_section_identity, catalog_hash, policy_hash, session_date, factor_id
                    )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS panel_cross_section_availability_revision (
                    revision_id VARCHAR PRIMARY KEY,
                    cross_section_identity VARCHAR NOT NULL,
                    catalog_hash VARCHAR NOT NULL,
                    policy_hash VARCHAR NOT NULL,
                    session_date DATE NOT NULL,
                    factor_id VARCHAR NOT NULL,
                    prior_availability_hash VARCHAR NOT NULL,
                    next_availability_hash VARCHAR NOT NULL,
                    materialization_receipt_hash VARCHAR NOT NULL,
                    observed_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS panel_materialization_receipt (
                    receipt_hash VARCHAR PRIMARY KEY,
                    market_profile_id VARCHAR NOT NULL,
                    manifest_revision VARCHAR NOT NULL,
                    sector_revision VARCHAR NOT NULL,
                    catalog_hash VARCHAR NOT NULL,
                    spy_revision VARCHAR NOT NULL,
                    policy_hash VARCHAR NOT NULL,
                    panel_binding_hash VARCHAR NOT NULL,
                    panel_content_hash VARCHAR NOT NULL,
                    history_start DATE NOT NULL,
                    as_of_session DATE NOT NULL,
                    row_count BIGINT NOT NULL,
                    availability_count BIGINT NOT NULL,
                    admission_summary_json VARCHAR NOT NULL,
                    temporal_risk_json VARCHAR NOT NULL,
                    source_state_hash VARCHAR NOT NULL,
                    temporal_identity_hash VARCHAR,
                    knowledge_cutoff_at TIMESTAMP,
                    observed_at TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                "ALTER TABLE panel_materialization_receipt "
                "ADD COLUMN IF NOT EXISTS temporal_identity_hash VARCHAR"
            )
            connection.execute(
                "ALTER TABLE panel_materialization_receipt "
                "ADD COLUMN IF NOT EXISTS knowledge_cutoff_at TIMESTAMP"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS active_feature_panel_binding (
                    market_profile_id VARCHAR PRIMARY KEY,
                    manifest_revision VARCHAR NOT NULL,
                    sector_revision VARCHAR NOT NULL,
                    catalog_hash VARCHAR NOT NULL,
                    spy_revision VARCHAR NOT NULL,
                    policy_hash VARCHAR NOT NULL,
                    panel_hash VARCHAR NOT NULL,
                    panel_binding_hash VARCHAR,
                    panel_content_hash VARCHAR,
                    history_start DATE,
                    as_of_session DATE,
                    materialization_receipt_hash VARCHAR,
                    admission_summary_hash VARCHAR,
                    temporal_risk_hash VARCHAR,
                    temporal_identity_hash VARCHAR,
                    source_state_hash VARCHAR,
                    knowledge_cutoff_at TIMESTAMP,
                    activated_at TIMESTAMP NOT NULL
                )
                """
            )
            for definition in (
                "panel_binding_hash VARCHAR",
                "panel_content_hash VARCHAR",
                "history_start DATE",
                "as_of_session DATE",
                "materialization_receipt_hash VARCHAR",
                "admission_summary_hash VARCHAR",
                "temporal_risk_hash VARCHAR",
                "temporal_identity_hash VARCHAR",
                "source_state_hash VARCHAR",
                "knowledge_cutoff_at TIMESTAMP",
            ):
                connection.execute(
                    "ALTER TABLE active_feature_panel_binding "
                    f"ADD COLUMN IF NOT EXISTS {definition}"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_panel_snapshot_manifest (
                    snapshot_hash VARCHAR PRIMARY KEY,
                    panel_binding_hash VARCHAR,
                    panel_content_hash VARCHAR NOT NULL,
                    manifest_uri VARCHAR NOT NULL,
                    metadata_hash VARCHAR NOT NULL,
                    history_start DATE NOT NULL,
                    as_of_session DATE NOT NULL,
                    knowledge_cutoff_at TIMESTAMP,
                    temporal_identity_hash VARCHAR,
                    active_listing_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    lifecycle VARCHAR NOT NULL DEFAULT 'ACTIVE',
                    lifecycle_reason VARCHAR,
                    lifecycle_updated_at TIMESTAMP,
                    physical_availability VARCHAR NOT NULL DEFAULT 'AVAILABLE',
                    eviction_plan_hash VARCHAR,
                    created_at TIMESTAMP NOT NULL
                )
                """
            )
            for definition in (
                "panel_binding_hash VARCHAR",
                "knowledge_cutoff_at TIMESTAMP",
                "temporal_identity_hash VARCHAR",
                "lifecycle VARCHAR DEFAULT 'ACTIVE'",
                "lifecycle_reason VARCHAR",
                "lifecycle_updated_at TIMESTAMP",
                "physical_availability VARCHAR DEFAULT 'AVAILABLE'",
                "eviction_plan_hash VARCHAR",
            ):
                connection.execute(
                    "ALTER TABLE feature_panel_snapshot_manifest "
                    f"ADD COLUMN IF NOT EXISTS {definition}"
                )
            connection.execute(
                """
                UPDATE feature_panel_snapshot_manifest
                SET physical_availability = 'AVAILABLE'
                WHERE physical_availability IS NULL
                """
            )
            connection.execute(
                """
                UPDATE feature_panel_snapshot_manifest AS snapshot
                SET panel_binding_hash = panel.panel_binding_hash
                FROM active_feature_panel_binding AS panel
                WHERE snapshot.panel_binding_hash IS NULL
                  AND snapshot.panel_content_hash = panel.panel_content_hash
                  AND snapshot.as_of_session = panel.as_of_session
                  AND snapshot.knowledge_cutoff_at = panel.knowledge_cutoff_at
                  AND snapshot.temporal_identity_hash = panel.temporal_identity_hash
                """
            )
            # When a Panel is stale is Feature's rule alone: a Feature build runs it after
            # this bootstrap and republishes the projection with it (V176).
            profile = manifest.profile
            connection.execute(
                """
                INSERT INTO market_profile (
                    market_profile_id, display_name, market, currency, calendar_id, provider,
                    daily_price_basis, data_validity_class
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (market_profile_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    provider = excluded.provider,
                    daily_price_basis = excluded.daily_price_basis,
                    data_validity_class = excluded.data_validity_class
                """,
                [
                    profile.market_profile_id,
                    profile.display_name,
                    profile.market,
                    profile.currency,
                    profile.calendar_id,
                    profile.provider,
                    profile.daily_price_basis,
                    profile.data_validity_class,
                ],
            )
            connection.execute(
                """
                INSERT INTO universe_manifest (
                    manifest_id, market_profile_id, as_of_date, revision_sha256,
                    data_validity_class, listing_count, universe_membership_basis,
                    is_point_in_time_historical, membership_fingerprint,
                    qualification_policy_hash, universe_policy_type,
                    universe_components_json, survivorship_bias_warning,
                    research_use_class, qualification_obligations_json, membership_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (manifest_id) DO UPDATE SET
                    as_of_date = excluded.as_of_date,
                    revision_sha256 = excluded.revision_sha256,
                    listing_count = excluded.listing_count,
                    universe_membership_basis = excluded.universe_membership_basis,
                    is_point_in_time_historical = excluded.is_point_in_time_historical,
                    membership_fingerprint = excluded.membership_fingerprint,
                    qualification_policy_hash = excluded.qualification_policy_hash,
                    universe_policy_type = excluded.universe_policy_type,
                    universe_components_json = excluded.universe_components_json,
                    survivorship_bias_warning = excluded.survivorship_bias_warning,
                    research_use_class = excluded.research_use_class,
                    qualification_obligations_json = excluded.qualification_obligations_json,
                    membership_sha256 = excluded.membership_sha256
                """,
                [
                    manifest.manifest_id,
                    profile.market_profile_id,
                    profile.manifest_as_of,
                    manifest.revision_sha256,
                    profile.data_validity_class,
                    len(manifest.listings),
                    manifest.universe_membership_basis,
                    manifest.is_point_in_time_historical,
                    manifest.membership_fingerprint,
                    manifest.qualification_policy_hash,
                    manifest.universe_policy_type,
                    json.dumps(manifest.universe_components, separators=(",", ":")),
                    manifest.survivorship_bias_warning,
                    manifest.research_use_class,
                    (
                        json.dumps(list(manifest.qualification_obligations), separators=(",", ":"))
                        if manifest.qualification_obligations
                        else None
                    ),
                    _membership_digest(listing.listing_id for listing in manifest.listings),
                ],
            )
            # Four inserts per listing became four statements per listing; a
            # 466-listing manifest issued 1,864 of them, each taking its own
            # index probe for its conflict clause. The four relations below are
            # the same rows in the same order, and each keeps its own clause.
            # `listing` resolves conflicts with DO UPDATE, so a repeated
            # listing_id must keep the last row exactly as the loop left it;
            # the DO NOTHING targets keep the first.
            identity_rows = _last_by_key(
                [
                    (
                        listing.listing_id,
                        listing.symbol,
                        listing.mic,
                        listing.issuer_external_id,
                    )
                    for listing in manifest.listings
                ]
            )
            mapping_rows = _first_by_key(
                [(listing.listing_id, listing.provider_symbol) for listing in manifest.listings]
            )
            quality_rows = _first_by_key(
                [
                    (listing.listing_id, _utc_naive(datetime.now(UTC)))
                    for listing in manifest.listings
                ]
            )
            membership_rows = _first_by_key(
                [(listing.listing_id,) for listing in manifest.listings]
            )
            if identity_rows:
                with _staged_rows(
                    connection,
                    "listing_identity_stage",
                    {
                        "listing_id": pa.array([str(row[0]) for row in identity_rows]),
                        "display_symbol": pa.array([str(row[1]) for row in identity_rows]),
                        "mic": pa.array([str(row[2]) for row in identity_rows]),
                        "issuer_external_id": pa.array(
                            [None if row[3] is None else str(row[3]) for row in identity_rows]
                        ),
                    },
                ) as stage:
                    connection.execute(
                        f"""
                        INSERT INTO listing
                        SELECT listing_id, ?, display_symbol, mic, issuer_external_id
                        FROM {stage}
                        ON CONFLICT (listing_id) DO UPDATE SET
                            display_symbol = excluded.display_symbol,
                            mic = excluded.mic,
                            issuer_external_id = excluded.issuer_external_id
                        """,
                        [profile.market_profile_id],
                    )
                with _staged_rows(
                    connection,
                    "provider_mapping_stage",
                    {
                        "listing_id": pa.array([str(row[0]) for row in mapping_rows]),
                        "provider_symbol": pa.array([str(row[1]) for row in mapping_rows]),
                    },
                ) as stage:
                    connection.execute(
                        f"""
                        INSERT INTO provider_symbol_mapping
                        SELECT listing_id, ?, provider_symbol, ?, NULL, 'ACTIVE'
                        FROM {stage}
                        ON CONFLICT (listing_id, provider, effective_from) DO NOTHING
                        """,
                        [profile.provider, profile.manifest_as_of],
                    )
                with _staged_rows(
                    connection,
                    "data_quality_stage",
                    {
                        "listing_id": pa.array([str(row[0]) for row in quality_rows]),
                        "observed_at": pa.array(
                            [row[1] for row in quality_rows], pa.timestamp("us")
                        ),
                    },
                ) as stage:
                    connection.execute(
                        f"""
                        INSERT INTO data_quality
                        SELECT listing_id, NULL, 'DATA_UNREADY', 'no validated bar yet',
                               observed_at
                        FROM {stage}
                        ON CONFLICT (listing_id) DO NOTHING
                        """
                    )
                with _staged_rows(
                    connection,
                    "manifest_listing_stage",
                    {"listing_id": pa.array([str(row[0]) for row in membership_rows])},
                ) as stage:
                    connection.execute(
                        f"""
                        INSERT INTO universe_manifest_listing
                        SELECT ?, listing_id FROM {stage}
                        ON CONFLICT (manifest_id, listing_id) DO NOTHING
                        """,
                        [manifest.manifest_id],
                    )
            self.readiness.reconcile_incomplete_panel_identity(
                connection,
                market_profile_id=profile.market_profile_id,
                observed_at=datetime.now(UTC),
            )
            checkpoint_workspace_database(connection)
        finally:
            connection.close()

    def load_universe_manifest(self, manifest_id: str) -> UniverseManifest:
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
        # it stored (EV2, V269).
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

    def record_universe_bootstrap(self, record: UniverseBootstrapRecord) -> None:
        """Freeze T0 and U0 once; a second, different record for the profile is refused."""
        existing = self.universe_bootstrap(record.market_profile_id)
        if existing is not None:
            if existing.record_hash != record.record_hash:
                raise ValueError("universe bootstrap is already recorded with another identity")
            return
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO universe_bootstrap VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (market_profile_id) DO NOTHING
                """,
                [
                    record.market_profile_id,
                    record.t0_session,
                    record.history_start,
                    record.cohort_hash,
                    json.dumps(list(record.cohort_listing_ids), separators=(",", ":")),
                    record.manifest_revision,
                    record.candidate_manifest_hash,
                    record.qualification_policy_hash,
                    record.feature_input_policy_hash,
                    _utc_naive(record.source_observed_at),
                    _utc_naive(record.admitted_at),
                    record.panel_snapshot_hash,
                    record.derivation,
                    record.initialization_assumption,
                    record.record_hash,
                ],
            )
        finally:
            connection.close()

    def universe_bootstrap(self, market_profile_id: str) -> UniverseBootstrapRecord | None:
        """Return the profile's frozen bootstrap boundary, if recorded."""
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

    def record_universe_source_observation(self, observation: UniverseSourceObservation) -> None:
        """Idempotently record a bounded explicit source check, upgrading only this owned table."""
        expected_session = membership_effective_session(
            observed_at=observation.observed_at,
            decided_at=observation.observed_at,
            not_before=observation.observed_at.date() - timedelta(days=2),
        )
        if observation.first_eligible_session != expected_session:
            raise ValueError("universe_membership.observation_session_mismatch")
        values = (
            observation.observation_hash,
            observation.market_profile_id,
            _utc_naive(observation.observed_at),
            _utc_naive(observation.previous_observed_at)
            if observation.previous_observed_at is not None
            else None,
            observation.candidate_membership_hash,
            observation.source_identity_hash,
            observation.first_eligible_session,
        )
        connection = self._connect()
        try:
            connection.execute(_SOURCE_OBSERVATION_TABLE)
            stored = connection.execute(
                "INSERT INTO universe_source_observation VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (observation_hash) DO NOTHING RETURNING *",
                values,
            ).fetchone()
            if stored is None:
                stored = connection.execute(
                    "SELECT * FROM universe_source_observation WHERE observation_hash = ?",
                    [observation.observation_hash],
                ).fetchone()
            if stored != values:
                raise ValueError("universe_membership.observation_conflict")
        finally:
            connection.close()

    def universe_source_observations(
        self, market_profile_id: str, *, limit: int = 20
    ) -> tuple[UniverseSourceObservation, ...]:
        """Read recent checks without requiring a writer migration of an older workspace."""
        if limit < 1:
            raise ValueError("universe_membership.observation_limit_invalid")
        if not self.path.exists():
            return ()
        connection = self._connect(read_only=True)
        try:
            if not _table_exists(connection, "universe_source_observation"):
                return ()
            rows = connection.execute(
                "SELECT observation_hash, observed_at, previous_observed_at, "
                "candidate_membership_hash, source_identity_hash, first_eligible_session "
                "FROM universe_source_observation WHERE market_profile_id = ? "
                "ORDER BY observed_at DESC, observation_hash LIMIT ?",
                [market_profile_id, limit],
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            UniverseSourceObservation(
                observation_hash=str(row[0]),
                market_profile_id=market_profile_id,
                observed_at=_utc_aware(row[1]),
                previous_observed_at=_utc_aware(row[2]) if row[2] is not None else None,
                candidate_membership_hash=str(row[3]),
                source_identity_hash=str(row[4]),
                first_eligible_session=row[5],
            )
            for row in rows
        )

    def append_membership_events(self, events: Sequence[MembershipEvent]) -> None:
        """Append membership events in journal order; a gap or a rewrite is refused."""
        if not events:
            return
        profiles = {item.market_profile_id for item in events}
        if len(profiles) != 1:
            raise ValueError("membership events must belong to one market profile")
        profile = profiles.pop()
        bootstrap = self.universe_bootstrap(profile)
        if bootstrap is None:
            raise ValueError("membership events require a recorded universe bootstrap")
        ordered = sorted(events, key=lambda item: item.sequence)
        for event in ordered:
            if (
                membership_effective_session(
                    observed_at=event.observed_at,
                    decided_at=event.decided_at,
                    not_before=event.effective_session,
                )
                != event.effective_session
            ):
                raise ValueError("universe_membership.effective_session_precedes_knowledge")
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            last = connection.execute(
                """
                SELECT max(sequence), max(effective_session)
                FROM universe_membership_event WHERE market_profile_id = ?
                """,
                [profile],
            ).fetchone()
            next_sequence = int(last[0]) + 1 if last is not None and last[0] is not None else 1
            latest_effective = (
                last[1] if last is not None and last[1] is not None else bootstrap.t0_session
            )
            for event in ordered:
                if event.sequence != next_sequence:
                    raise ValueError("membership event sequence is not the journal's next")
                if event.effective_session < bootstrap.t0_session:
                    raise ValueError("membership event takes effect before the bootstrap session")
                # Event ordering is separate from source-observation freshness;
                # unchanged checks live in universe_source_observation.
                if event.effective_session < latest_effective:
                    raise ValueError("membership event takes effect before the journal's latest")
                latest_effective = event.effective_session
                connection.execute(
                    """
                    INSERT INTO universe_membership_event VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        event.event_hash,
                        event.market_profile_id,
                        event.sequence,
                        event.listing_id,
                        event.kind,
                        event.effective_session,
                        _utc_naive(event.observed_at),
                        _utc_naive(event.decided_at),
                        event.authority,
                        event.reference_hash,
                        event.manifest_revision,
                    ],
                )
                next_sequence += 1
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def membership_events(self, market_profile_id: str) -> tuple[MembershipEvent, ...]:
        """Return the profile's membership journal in sequence order."""
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

    def research_listing_sources(self, manifest: UniverseManifest) -> dict[str, str]:
        """Coverage and its admitting source, not current/session eligibility.

        Keep the initial cohort and each journal entry for historical samples,
        lookbacks and exited holdings. Current mappings take precedence; prior
        names retain their last entry's source. No per-day copy or array scan.
        """
        bootstrap = self.universe_bootstrap(manifest.profile.market_profile_id)
        sources = (
            dict.fromkeys(bootstrap.cohort_listing_ids, bootstrap.manifest_revision)
            if bootstrap is not None
            else {}
        )
        if bootstrap is not None:
            for event in self.membership_events(manifest.profile.market_profile_id):
                if event.kind == "ENTRY":
                    sources[event.listing_id] = event.manifest_revision
        sources.update((item.listing_id, manifest.revision_sha256) for item in manifest.listings)
        return sources

    def membership_schedule(
        self,
        market_profile_id: str,
        *,
        sessions: Sequence[date],
        fallback_listing_ids: Sequence[str],
    ) -> MembershipSchedule:
        """Per-session membership over ``sessions``, replayed from the journal.

        Before the bootstrap is recorded every session holds
        ``fallback_listing_ids`` -- the manifest whose cohort is being built.
        """
        return resolve_membership_schedule(
            sessions=sessions,
            bootstrap=self.universe_bootstrap(market_profile_id),
            events=self.membership_events(market_profile_id),
            fallback_listing_ids=fallback_listing_ids,
        )

    def load_universe_manifest_revision(self, manifest_revision: str) -> UniverseManifest:
        """Restore one unambiguous manifest from its immutable content revision."""
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT manifest_id FROM universe_manifest
                WHERE revision_sha256 = ? ORDER BY manifest_id
                """,
                [manifest_revision],
            ).fetchall()
        finally:
            connection.close()
        if len(rows) != 1:
            raise ValueError("workspace manifest revision is missing or ambiguous")
        return self.load_universe_manifest(str(rows[0][0]))

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

    @staticmethod
    def _source_admission_identity(
        connection: duckdb.DuckDBPyConnection, market_profile_id: str
    ) -> tuple[str, str] | None:
        onboarding = connection.execute(
            """SELECT research_manifest_id, research_manifest_revision
               FROM current_universe_onboarding
               WHERE lifecycle = 'COMPLETED' AND research_manifest_id IS NOT NULL
                 AND acquisition_manifest_id LIKE ?
               ORDER BY updated_at DESC LIMIT 1""",
            [f"{market_profile_id}:acquisition:%"],
        ).fetchone()
        if onboarding is None:
            return None
        activated = connection.execute(
            """SELECT u.manifest_id, u.revision_sha256
               FROM manifest_transition t
               JOIN universe_manifest u ON u.revision_sha256 = t.next_manifest_revision
               JOIN workspace_readiness r ON r.market_profile_id = t.market_profile_id
                 AND r.active_membership_fingerprint = t.membership_fingerprint
               WHERE t.market_profile_id = ?
                 AND t.lifecycle IN ('ACTIVATED', 'SOURCE_CHANGED_ACTIVE_SET_UNCHANGED')
               ORDER BY t.activated_at DESC, t.created_at DESC LIMIT 1""",
            [market_profile_id],
        ).fetchone()
        chosen = activated or onboarding
        return str(chosen[0]), str(chosen[1])

    def source_admission_manifest(self, *, market_profile_id: str) -> UniverseManifest | None:
        """Return the admitted source root before Feature-only subsets."""
        if not self.path.exists():
            return None
        with self._connect(read_only=True) as connection:
            root = self._source_admission_identity(connection, market_profile_id)
        if root is None:
            return None
        manifest = self.load_universe_manifest(root[0])
        if (
            manifest.revision_sha256 != root[1]
            or manifest.profile.market_profile_id != market_profile_id
        ):
            raise ValueError("source admission manifest binding has drifted")
        return manifest

    def current_quality_filtered_research_manifest(
        self,
        *,
        market_profile_id: str,
    ) -> UniverseManifest | None:
        """Return the latest host-admitted research manifest for Front Desk admission.

        An activated source transition owns the nominal candidate roster. Raw
        requalification may omit an existing member that still belongs to the
        observed source; that omission is usability, not a membership exit.
        Feature admissions refine the activated root, never an arbitrary pointer.
        Legacy onboarding-only workspaces retain their original admission root.
        """
        if not self.path.exists():
            return None
        connection = self._connect(read_only=True)
        try:
            root_row = self._source_admission_identity(connection, market_profile_id)
            if root_row is None:
                return None
            admission_rows = connection.execute(
                """
                SELECT candidate_manifest_revision, research_manifest_id,
                       research_manifest_revision, created_at, admission_hash
                FROM feature_input_admission
                ORDER BY created_at DESC, admission_hash DESC
                """,
            ).fetchall()
            readiness_row = connection.execute(
                """
                SELECT active_manifest_id, active_manifest_revision
                FROM workspace_readiness
                WHERE market_profile_id = ?
                """,
                [market_profile_id],
            ).fetchone()
        finally:
            connection.close()

        root_id, root_revision = str(root_row[0]), str(root_row[1])
        edges_by_parent: dict[str, list[tuple[str, str]]] = {}
        for row in admission_rows:
            edges_by_parent.setdefault(str(row[0]), []).append((str(row[1]), str(row[2])))

        if readiness_row is not None and readiness_row[0] is not None:
            selected = (str(readiness_row[0]), str(readiness_row[1]))
            reachable = {root_revision}
            pending = [root_revision]
            manifest_ids = {root_revision: root_id}
            while pending:
                parent_revision = pending.pop()
                for child_id, child_revision in edges_by_parent.get(parent_revision, ()):
                    known_id = manifest_ids.get(child_revision)
                    if known_id is not None and known_id != child_id:
                        raise ValueError(
                            "feature-input admission lineage has conflicting identities"
                        )
                    manifest_ids[child_revision] = child_id
                    if child_revision not in reachable:
                        reachable.add(child_revision)
                        pending.append(child_revision)
            if selected[1] not in reachable or manifest_ids.get(selected[1]) != selected[0]:
                raise ValueError("workspace readiness active manifest is outside admission lineage")
        else:
            # Legacy/onboarding-only workspaces do not yet have an active
            # readiness pointer. Follow the latest durable admission at each
            # parent until the stable terminal membership is reached.
            selected = (root_id, root_revision)
            visited: set[str] = set()
            while selected[1] not in visited:
                visited.add(selected[1])
                children = edges_by_parent.get(selected[1], ())
                if not children:
                    break
                child = children[0]
                if child[1] == selected[1]:
                    if child[0] != selected[0]:
                        raise ValueError(
                            "feature-input admission self-edge changed manifest identity"
                        )
                    break
                selected = child

        manifest = self.load_universe_manifest(str(selected[0]))
        if manifest.revision_sha256 != str(selected[1]):
            raise ValueError("quality-filtered research manifest binding has drifted")
        root = self.load_universe_manifest(root_id)
        if manifest.profile.market_profile_id != market_profile_id or not {
            item.listing_id for item in manifest.listings
        }.issubset({item.listing_id for item in root.listings}):
            raise ValueError("quality-filtered research manifest escaped onboarding authority")
        return manifest

    def record_manifest_transition(
        self,
        *,
        transition_id: str,
        market_profile_id: str,
        prior_manifest_revision: str | None,
        membership_fingerprint: str,
        additions: Sequence[str],
        removals: Sequence[str],
        lifecycle: str,
        approved_at: datetime | None,
        created_at: datetime,
    ) -> ManifestTransitionRecord:
        """Persist a proposed manifest transition and return its durable record.

        Args:
            transition_id: Stable identifier for the proposal.
            market_profile_id: Market scope of the proposed transition.
            prior_manifest_revision: Revision being replaced, if any.
            membership_fingerprint: Identity of the proposed member set.
            additions: Listing identities entering the member set.
            removals: Listing identities leaving the member set.
            lifecycle: Initial transition state.
            approved_at: Approval instant, if already approved.
            created_at: Proposal creation instant.

        Returns:
            The persisted transition record.

        """
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO manifest_transition VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT (transition_id) DO NOTHING
                """,
                [
                    transition_id,
                    market_profile_id,
                    prior_manifest_revision,
                    membership_fingerprint,
                    json.dumps(tuple(sorted(additions)), separators=(",", ":")),
                    json.dumps(tuple(sorted(removals)), separators=(",", ":")),
                    lifecycle,
                    _utc_naive(approved_at) if approved_at is not None else None,
                    _utc_naive(created_at),
                ],
            )
        finally:
            connection.close()
        return self.manifest_transition(transition_id)

    def manifest_transition(self, transition_id: str) -> ManifestTransitionRecord:
        """Load one manifest transition by its durable identifier.

        Raises:
            ValueError: The transition is missing or its members are invalid.

        """
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT transition_id, market_profile_id, prior_manifest_revision,
                       next_manifest_revision, membership_fingerprint, additions_json,
                       removals_json, lifecycle, approved_at, activated_at, created_at
                FROM manifest_transition WHERE transition_id = ?
                """,
                [transition_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise ValueError("manifest transition does not exist")
        additions = json.loads(str(row[5]))
        removals = json.loads(str(row[6]))
        if not isinstance(additions, list) or not isinstance(removals, list):
            raise ValueError("manifest transition members are invalid")
        return ManifestTransitionRecord(
            transition_id=str(row[0]),
            market_profile_id=str(row[1]),
            prior_manifest_revision=str(row[2]) if row[2] is not None else None,
            next_manifest_revision=str(row[3]) if row[3] is not None else None,
            membership_fingerprint=str(row[4]),
            additions=tuple(str(item) for item in additions),
            removals=tuple(str(item) for item in removals),
            lifecycle=str(row[7]),
            approved_at=row[8],
            activated_at=row[9],
            created_at=row[10],
        )

    def approve_manifest_transition(self, transition_id: str, *, approved_at: datetime) -> None:
        """Mark a proposed transition approved for onboarding.

        Raises:
            ValueError: The transition cannot be approved in its current state.

        """
        connection = self._connect()
        try:
            result = connection.execute(
                """
                UPDATE manifest_transition
                SET lifecycle = 'ONBOARDING_IN_PROGRESS', approved_at = ?
                WHERE transition_id = ? AND lifecycle IN ('PROPOSED', 'ONBOARDING_IN_PROGRESS')
                """,
                [_utc_naive(approved_at), transition_id],
            )
            if result.rowcount == 0:
                raise ValueError("manifest transition cannot be approved")
        finally:
            connection.close()

    def activate_workspace_manifest(
        self,
        *,
        transition_id: str,
        manifest: UniverseManifest,
        membership_fingerprint: str,
        candidate_manifest_document: Mapping[str, object],
        checked_at: datetime,
        changed_at: datetime,
        within_activation: ActivationStep,
    ) -> WorkspaceReadinessRecord:
        """Atomically promote a completed manifest without rewriting history.

        ``within_activation`` is the caller's step, run inside this transaction: the state it
        writes belongs to another owner, which defines it (V159), so activation stays atomic.
        """
        self.bootstrap(manifest)
        candidate_json = json.dumps(
            candidate_manifest_document, sort_keys=True, separators=(",", ":")
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            transition = connection.execute(
                """
                SELECT lifecycle, prior_manifest_revision
                FROM manifest_transition WHERE transition_id = ?
                """,
                [transition_id],
            ).fetchone()
            if transition is None:
                raise ValueError("manifest transition does not exist")
            if str(transition[0]) != "ONBOARDING_IN_PROGRESS":
                raise ValueError("manifest transition is not ready for activation")
            connection.execute(
                """
                UPDATE manifest_transition
                SET next_manifest_revision = ?, lifecycle = 'ACTIVATED', activated_at = ?
                WHERE transition_id = ?
                """,
                [manifest.revision_sha256, _utc_naive(changed_at), transition_id],
            )
            within_activation(
                connection,
                transition_id=transition_id,
                prior_manifest_revision=None if transition[1] is None else str(transition[1]),
                next_manifest_revision=manifest.revision_sha256,
                at=changed_at,
            )
            WorkspaceReadinessRepository.feature_building_for_manifest(
                connection,
                market_profile_id=manifest.profile.market_profile_id,
                manifest_id=manifest.manifest_id,
                manifest_revision=manifest.revision_sha256,
                membership_fingerprint=membership_fingerprint,
                candidate_manifest_json=candidate_json,
                checked_at=checked_at,
                changed_at=changed_at,
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        record = self.readiness.load(manifest.profile.market_profile_id)
        if record is None:
            raise AssertionError("activated workspace manifest is missing")
        return record

    def record_source_change_without_active_set_change(
        self,
        *,
        transition_id: str,
        active_manifest: UniverseManifest,
        membership_fingerprint: str,
        candidate_manifest_document: Mapping[str, object],
        checked_at: datetime,
        changed_at: datetime,
    ) -> WorkspaceReadinessRecord:
        """Advance source provenance without manufacturing a Panel revision."""
        candidate_json = json.dumps(
            candidate_manifest_document, sort_keys=True, separators=(",", ":")
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            transition = connection.execute(
                "SELECT lifecycle, prior_manifest_revision FROM manifest_transition "
                "WHERE transition_id = ?",
                [transition_id],
            ).fetchone()
            if transition is None or str(transition[0]) != "ONBOARDING_IN_PROGRESS":
                raise ValueError("manifest transition is not ready for source-only activation")
            if str(transition[1]) != active_manifest.revision_sha256:
                raise ValueError("source-only transition differs from the active manifest")
            connection.execute(
                """
                UPDATE manifest_transition
                SET next_manifest_revision = ?,
                    lifecycle = 'SOURCE_CHANGED_ACTIVE_SET_UNCHANGED', activated_at = ?
                WHERE transition_id = ?
                """,
                [active_manifest.revision_sha256, _utc_naive(changed_at), transition_id],
            )
            WorkspaceReadinessRepository.feature_building_for_source_change(
                connection,
                market_profile_id=active_manifest.profile.market_profile_id,
                manifest_revision=active_manifest.revision_sha256,
                membership_fingerprint=membership_fingerprint,
                candidate_manifest_json=candidate_json,
                checked_at=checked_at,
                changed_at=changed_at,
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        record = self.readiness.load(active_manifest.profile.market_profile_id)
        if record is None:
            raise ValueError("source-only readiness activation was not durable")
        return record

    def admit_current_universe_onboarding(
        self,
        manifest: UniverseManifest,
        *,
        onboarding_id: str,
        candidate_manifest_hash: str,
        candidate_manifest_document: Mapping[str, object],
        history_start: date,
        as_of_session: date,
        calendar_by_listing_id: Mapping[str, str],
        observed_at: datetime,
        requested_listing_ids: tuple[str, ...] = (),
    ) -> CurrentUniverseOnboardingRun:
        """Persist a resumable full-universe onboarding intent and its units of work."""
        if history_start > as_of_session:
            raise ValueError("onboarding history start is after as_of session")
        known_listing_ids = {listing.listing_id for listing in manifest.listings}
        selected_ids = set(requested_listing_ids) if requested_listing_ids else known_listing_ids
        if selected_ids - known_listing_ids or set(calendar_by_listing_id) != selected_ids:
            raise ValueError("onboarding calendar scope does not match acquisition manifest")
        if any(calendar not in {"XNYS", "XNAS"} for calendar in calendar_by_listing_id.values()):
            raise ValueError("onboarding has an unsupported listing calendar")
        self.bootstrap(manifest)
        observed = _utc_naive(observed_at)
        document_json = json.dumps(
            candidate_manifest_document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            existing = connection.execute(
                """
                SELECT candidate_manifest_hash, acquisition_manifest_id,
                       acquisition_manifest_revision, history_start, as_of_session
                FROM current_universe_onboarding WHERE onboarding_id = ?
                """,
                [onboarding_id],
            ).fetchone()
            expected = (
                candidate_manifest_hash,
                manifest.manifest_id,
                manifest.revision_sha256,
                history_start,
                as_of_session,
            )
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO current_universe_onboarding (
                        onboarding_id, candidate_manifest_hash, candidate_manifest_json,
                        acquisition_manifest_id, acquisition_manifest_revision, history_start,
                        as_of_session, lifecycle, quality_admission_hash, research_manifest_id,
                        research_manifest_revision, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'RUNNING', NULL, NULL, NULL, ?, ?)
                    """,
                    [
                        onboarding_id,
                        candidate_manifest_hash,
                        document_json,
                        manifest.manifest_id,
                        manifest.revision_sha256,
                        history_start,
                        as_of_session,
                        observed,
                        observed,
                    ],
                )
            elif tuple(existing) != expected:
                raise ValueError(
                    "onboarding identifier is already bound to different immutable input"
                )
            elif {
                row[0]
                for row in connection.execute(
                    "SELECT listing_id FROM current_universe_onboarding_listing "
                    "WHERE onboarding_id = ?",
                    [onboarding_id],
                ).fetchall()
            } != selected_ids:
                raise ValueError("onboarding identifier is already bound to a different work scope")
            seeded = _first_by_key(
                [
                    (
                        listing.listing_id,
                        listing.symbol,
                        calendar_by_listing_id[listing.listing_id],
                    )
                    for listing in manifest.listings
                    if listing.listing_id in selected_ids
                ]
            )
            if seeded:
                with _staged_rows(
                    connection,
                    "onboarding_listing_stage",
                    {
                        "listing_id": pa.array([str(row[0]) for row in seeded]),
                        "symbol": pa.array([str(row[1]) for row in seeded]),
                        "calendar_id": pa.array([str(row[2]) for row in seeded]),
                    },
                ) as stage:
                    connection.execute(
                        f"""
                        INSERT INTO current_universe_onboarding_listing (
                            onboarding_id, listing_id, symbol, calendar_id, state, failure_code,
                            raw_through, updated_at
                        )
                        SELECT ?, listing_id, symbol, calendar_id, 'PENDING', NULL, NULL, ?
                        FROM {stage}
                        ON CONFLICT (onboarding_id, listing_id) DO NOTHING
                        """,
                        [onboarding_id, observed],
                    )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return self.current_universe_onboarding_run(onboarding_id)

    def current_universe_onboarding_run(self, onboarding_id: str) -> CurrentUniverseOnboardingRun:
        """Load one durable current-universe onboarding run.

        Raises:
            ValueError: The onboarding run does not exist.

        """
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT onboarding_id, candidate_manifest_hash, acquisition_manifest_id,
                       acquisition_manifest_revision, history_start, as_of_session, lifecycle,
                       quality_admission_hash, research_manifest_id, research_manifest_revision,
                       created_at, updated_at
                FROM current_universe_onboarding WHERE onboarding_id = ?
                """,
                [onboarding_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise ValueError("current-universe onboarding does not exist")
        return CurrentUniverseOnboardingRun(*row)

    def resumable_current_universe_onboarding_input(
        self,
        *,
        market_profile_id: str,
    ) -> tuple[str, dict[str, object], date, date] | None:
        """Read the latest nonterminal onboarding input without exposing any raw data."""
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT onboarding_id, candidate_manifest_json, history_start, as_of_session
                FROM current_universe_onboarding
                WHERE lifecycle IN ('RUNNING', 'DEFERRED')
                  AND acquisition_manifest_id LIKE ?
                  AND (SELECT count(*) FROM current_universe_onboarding_listing unit
                       WHERE unit.onboarding_id = current_universe_onboarding.onboarding_id)
                    = (SELECT listing_count FROM universe_manifest manifest
                       WHERE manifest.manifest_id = acquisition_manifest_id)
                ORDER BY updated_at DESC LIMIT 1
                """,
                [f"{market_profile_id}:acquisition:%"],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        try:
            document = json.loads(str(row[1]))
        except json.JSONDecodeError as exc:
            raise ValueError("persisted candidate manifest JSON is invalid") from exc
        if not isinstance(document, dict):
            raise ValueError("persisted candidate manifest JSON is not an object")
        return str(row[0]), document, row[2], row[3]

    def current_universe_onboarding_listings(
        self, onboarding_id: str
    ) -> tuple[CurrentUniverseOnboardingListing, ...]:
        """List an onboarding run's listing units in symbol order."""
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT onboarding_id, listing_id, symbol, calendar_id, state, failure_code,
                       raw_through, updated_at
                FROM current_universe_onboarding_listing
                WHERE onboarding_id = ? ORDER BY symbol
                """,
                [onboarding_id],
            ).fetchall()
        finally:
            connection.close()
        return tuple(CurrentUniverseOnboardingListing(*row) for row in rows)

    def latest_current_universe_onboarding_disclosure(
        self, *, market_profile_id: str
    ) -> dict[str, object] | None:
        """Return a safe terminal-state summary for operational reporting.

        The disclosure deliberately contains counts and governed failure codes,
        never provider payloads, raw observations, SQL, or physical paths.
        """
        if not self.path.exists():
            return None
        connection = self._connect(read_only=True)
        try:
            run = connection.execute(
                """
                SELECT onboarding_id, acquisition_manifest_revision, history_start,
                       as_of_session, lifecycle, research_manifest_revision,
                       (SELECT listing_count FROM universe_manifest m
                        WHERE m.manifest_id = acquisition_manifest_id)
                FROM current_universe_onboarding
                WHERE acquisition_manifest_id LIKE ?
                  AND lifecycle = 'COMPLETED'
                ORDER BY updated_at DESC LIMIT 1
                """,
                [f"{market_profile_id}:acquisition:%"],
            ).fetchone()
            if run is None:
                return None
            state_rows = connection.execute(
                """
                SELECT state, COUNT(*)
                FROM current_universe_onboarding_listing
                WHERE onboarding_id = ?
                GROUP BY state ORDER BY state
                """,
                [run[0]],
            ).fetchall()
            failure_rows = connection.execute(
                """
                SELECT failure_code, COUNT(*)
                FROM current_universe_onboarding_listing
                WHERE onboarding_id = ? AND failure_code IS NOT NULL
                GROUP BY failure_code ORDER BY failure_code
                """,
                [run[0]],
            ).fetchall()
            quality_reason_rows = connection.execute(
                """
                SELECT reasons_json
                FROM current_universe_quality_admission
                WHERE onboarding_id = ? AND eligible = FALSE
                ORDER BY listing_id
                """,
                [run[0]],
            ).fetchall()
        finally:
            connection.close()
        state_counts = {str(row[0]): int(row[1]) for row in state_rows}
        failure_reason_counts = {str(row[0]): int(row[1]) for row in failure_rows}
        for row in quality_reason_rows:
            reasons = json.loads(str(row[0]))
            if not isinstance(reasons, list):
                raise ValueError("quality-admission reasons are not a list")
            for reason in reasons:
                key = str(reason)
                failure_reason_counts[key] = failure_reason_counts.get(key, 0) + 1
        disclosure = {
            "onboarding_id": str(run[0]),
            "acquisition_manifest_revision": str(run[1]),
            "history_start": run[2],
            "as_of_session": run[3],
            "lifecycle": str(run[4]),
            "research_manifest_revision": str(run[5]) if run[5] is not None else None,
            "candidate_listing_count": sum(state_counts.values()),
            "state_counts": state_counts,
            "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        }
        if sum(state_counts.values()) < int(run[6]):
            disclosure["work_scope"] = "CANDIDATE_REQUALIFICATION"
            disclosure["source_candidate_count"] = int(run[6])
        return disclosure

    def current_hydration_deferred(self, onboarding_id: str) -> HydrationDeferred | None:
        """Return the latest unresolved provider-wide hydration pause, if any."""
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT onboarding_id, deferred_retry_id, failure_code, retry_after_at,
                       observed_workers, next_workers, affected_listing_set_hash,
                       transport_policy_hash, created_at
                FROM current_universe_hydration_deferred
                WHERE onboarding_id = ? AND resolved_at IS NULL
                ORDER BY created_at DESC LIMIT 1
                """,
                [onboarding_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return HydrationDeferred(*row[:3], _utc_aware(row[3]), *row[4:8], _utc_aware(row[8]))

    def hydration_worker_limit(self, onboarding_id: str) -> int | None:
        """Return the latest durable degradation; a task never auto-escalates."""
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT next_workers FROM current_universe_hydration_deferred
                WHERE onboarding_id = ? ORDER BY created_at DESC LIMIT 1
                """,
                [onboarding_id],
            ).fetchone()
        finally:
            connection.close()
        return int(row[0]) if row is not None else None

    def hydration_defer_count(self, onboarding_id: str) -> int:
        """Count prior provider-wide pauses for deterministic backoff selection."""
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT COUNT(*) FROM current_universe_hydration_deferred
                WHERE onboarding_id = ?
                """,
                [onboarding_id],
            ).fetchone()
        finally:
            connection.close()
        return int(row[0]) if row is not None else 0

    def record_hydration_deferred(self, deferred: HydrationDeferred) -> None:
        """Persist a provider-wide pause and move its onboarding run to DEFERRED.

        Args:
            deferred: Pause identity, retry time, and reduced worker limit.

        Raises:
            ValueError: Worker counts or a reused pause identity are invalid.

        """
        if deferred.observed_workers < 1 or deferred.next_workers < 1:
            raise ValueError("hydration deferred worker counts must be positive")
        if deferred.next_workers > deferred.observed_workers:
            raise ValueError("hydration deferred cannot increase concurrency")
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            existing = connection.execute(
                """
                SELECT onboarding_id, failure_code, retry_after_at, observed_workers,
                       next_workers, affected_listing_set_hash, transport_policy_hash
                FROM current_universe_hydration_deferred
                WHERE deferred_retry_id = ?
                """,
                [deferred.deferred_retry_id],
            ).fetchone()
            expected = (
                deferred.onboarding_id,
                deferred.failure_code,
                _utc_naive(deferred.retry_after_at),
                deferred.observed_workers,
                deferred.next_workers,
                deferred.affected_listing_set_hash,
                deferred.transport_policy_hash,
            )
            if existing is not None and tuple(existing) != expected:
                raise ValueError("hydration deferred identity was reused with different content")
            connection.execute(
                """
                INSERT INTO current_universe_hydration_deferred VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL
                ) ON CONFLICT (deferred_retry_id) DO NOTHING
                """,
                [
                    deferred.deferred_retry_id,
                    deferred.onboarding_id,
                    deferred.failure_code,
                    _utc_naive(deferred.retry_after_at),
                    deferred.observed_workers,
                    deferred.next_workers,
                    deferred.affected_listing_set_hash,
                    deferred.transport_policy_hash,
                    _utc_naive(deferred.created_at),
                ],
            )
            connection.execute(
                """
                UPDATE current_universe_onboarding
                SET lifecycle = 'DEFERRED', updated_at = ? WHERE onboarding_id = ?
                """,
                [_utc_naive(deferred.created_at), deferred.onboarding_id],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def resolve_hydration_deferred(
        self, deferred_retry_id: str, *, observed_at: datetime
    ) -> HydrationDeferred:
        """Resolve a recorded hydration pause and return its durable record."""
        connection = self._connect()
        now = _utc_naive(observed_at)
        try:
            connection.execute("BEGIN TRANSACTION")
            row = connection.execute(
                """
                SELECT onboarding_id, deferred_retry_id, failure_code, retry_after_at,
                       observed_workers, next_workers, affected_listing_set_hash,
                       transport_policy_hash, created_at, resolved_at
                FROM current_universe_hydration_deferred WHERE deferred_retry_id = ?
                """,
                [deferred_retry_id],
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown hydration deferred: {deferred_retry_id}")
            if row[9] is not None:
                raise ValueError("hydration deferred is already resolved")
            if now < row[3]:
                raise ValueError("hydration deferred retry is not due")
            deferred = HydrationDeferred(
                *row[:3], _utc_aware(row[3]), *row[4:8], _utc_aware(row[8])
            )
            connection.execute(
                """
                UPDATE current_universe_hydration_deferred SET resolved_at = ?
                WHERE deferred_retry_id = ?
                """,
                [now, deferred_retry_id],
            )
            connection.execute(
                """
                UPDATE current_universe_onboarding
                SET lifecycle = 'RUNNING', updated_at = ? WHERE onboarding_id = ?
                """,
                [now, deferred.onboarding_id],
            )
            connection.execute("COMMIT")
            return deferred
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def update_current_universe_onboarding_listing(
        self,
        *,
        onboarding_id: str,
        listing_id: str,
        state: str,
        observed_at: datetime,
        failure_code: str | None = None,
        raw_through: date | None = None,
    ) -> None:
        """Record one deterministic unit-of-work outcome without provider payloads."""
        connection = self._connect()
        try:
            result = connection.execute(
                """
                UPDATE current_universe_onboarding_listing
                SET state = ?, failure_code = ?, raw_through = COALESCE(?, raw_through),
                    updated_at = ?
                WHERE onboarding_id = ? AND listing_id = ?
                """,
                [
                    state,
                    failure_code,
                    raw_through,
                    _utc_naive(observed_at),
                    onboarding_id,
                    listing_id,
                ],
            )
            if result.rowcount == 0:
                raise ValueError("onboarding listing does not exist")
        finally:
            connection.close()

    def record_current_universe_quality_admission(
        self,
        *,
        onboarding_id: str,
        listing_id: str,
        eligible: bool,
        expected_sessions: int,
        observed_sessions: int,
        missing_sessions: int,
        missing_ratio: float,
        maximum_consecutive_gap: int,
        reasons: Sequence[str],
        observed_at: datetime,
    ) -> None:
        """Persist the original quality thresholds before action-audit admission."""
        state = "QUALITY_ELIGIBLE" if eligible else "QUALITY_INELIGIBLE"
        summary = "quality gate passed" if eligible else "; ".join(reasons)
        observed = _utc_naive(observed_at)
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                INSERT INTO current_universe_quality_admission VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (onboarding_id, listing_id) DO UPDATE SET
                    eligible = excluded.eligible,
                    expected_sessions = excluded.expected_sessions,
                    observed_sessions = excluded.observed_sessions,
                    missing_sessions = excluded.missing_sessions,
                    missing_ratio = excluded.missing_ratio,
                    maximum_consecutive_gap = excluded.maximum_consecutive_gap,
                    reasons_json = excluded.reasons_json,
                    evaluated_at = excluded.evaluated_at
                """,
                [
                    onboarding_id,
                    listing_id,
                    eligible,
                    expected_sessions,
                    observed_sessions,
                    missing_sessions,
                    missing_ratio,
                    maximum_consecutive_gap,
                    json.dumps(tuple(reasons), separators=(",", ":")),
                    observed,
                ],
            )
            result = connection.execute(
                """
                UPDATE current_universe_onboarding_listing
                SET state = ?, failure_code = NULL, updated_at = ?
                WHERE onboarding_id = ? AND listing_id = ?
                """,
                [state, observed, onboarding_id, listing_id],
            )
            if result.rowcount == 0:
                raise ValueError("onboarding listing does not exist")
            if not eligible:
                connection.execute(
                    """
                    UPDATE data_quality SET quality_state = 'QUALITY_INELIGIBLE', summary = ?,
                        updated_at = ? WHERE listing_id = ?
                    """,
                    [summary, observed, listing_id],
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def current_universe_quality_admissions(
        self, onboarding_id: str
    ) -> tuple[CurrentUniverseQualityAdmission, ...]:
        """Return canonical quality-gate evidence without returning raw bars."""
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT onboarding_id, listing_id, eligible, expected_sessions,
                       observed_sessions, missing_sessions, missing_ratio,
                       maximum_consecutive_gap, reasons_json, evaluated_at
                FROM current_universe_quality_admission
                WHERE onboarding_id = ?
                ORDER BY listing_id
                """,
                [onboarding_id],
            ).fetchall()
        finally:
            connection.close()
        admissions: list[CurrentUniverseQualityAdmission] = []
        for row in rows:
            try:
                reasons_value = json.loads(str(row[8]))
            except json.JSONDecodeError as exc:
                raise ValueError("persisted quality-admission reasons are invalid") from exc
            if not isinstance(reasons_value, list) or not all(
                isinstance(reason, str) for reason in reasons_value
            ):
                raise ValueError("persisted quality-admission reasons are invalid")
            admissions.append(
                CurrentUniverseQualityAdmission(
                    onboarding_id=str(row[0]),
                    listing_id=str(row[1]),
                    eligible=bool(row[2]),
                    expected_sessions=int(row[3]),
                    observed_sessions=int(row[4]),
                    missing_sessions=int(row[5]),
                    missing_ratio=float(row[6]),
                    maximum_consecutive_gap=int(row[7]),
                    reasons=tuple(reasons_value),
                    evaluated_at=row[9],
                )
            )
        return tuple(admissions)

    def latest_quality_admission(self, listing_id: str) -> CurrentUniverseQualityAdmission | None:
        """Return the newest deterministic ten-year qualification for one listing."""
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT onboarding_id, listing_id, eligible, expected_sessions,
                       observed_sessions, missing_sessions, missing_ratio,
                       maximum_consecutive_gap, reasons_json, evaluated_at
                FROM current_universe_quality_admission
                WHERE listing_id = ?
                ORDER BY evaluated_at DESC, onboarding_id DESC LIMIT 1
                """,
                [listing_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return _quality_admission(row)

    def quality_governance_inputs(
        self, listing_ids: Sequence[str], *, through: date
    ) -> QualityGovernanceInputs:
        """Read a listing set's quality-governance inputs in one connection.

        Four set-scoped statements over the rows the per-listing readers
        (``latest_quality_admission``, ``raw_bars`` through ``through``,
        ``actions`` with its provider lookup) return for each listing, so a
        governance pass over a whole manifest costs four scans instead of
        three connection requests and one full-width bar set per listing.
        The per-listing readers stay for their other consumers; this is the
        same rows in the same order, not a second definition of them.
        """
        if not listing_ids or len(set(listing_ids)) != len(listing_ids):
            raise ValueError("market_data_ops.quality_governance_inputs_request_invalid")
        requested = tuple(sorted(listing_ids))
        connection = self._connect(read_only=True)
        try:
            admission_rows = connection.execute(
                """
                SELECT onboarding_id, listing_id, eligible, expected_sessions,
                       observed_sessions, missing_sessions, missing_ratio,
                       maximum_consecutive_gap, reasons_json, evaluated_at
                FROM (
                    SELECT *, row_number() OVER (
                        PARTITION BY listing_id
                        ORDER BY evaluated_at DESC, onboarding_id DESC
                    ) AS newest
                    FROM current_universe_quality_admission
                    WHERE listing_id IN (SELECT unnest(?))
                )
                WHERE newest = 1
                """,
                [requested],
            ).fetchall()
            series = connection.execute(
                """
                SELECT listing_id, session_date, close
                FROM raw_daily_bar_current
                WHERE listing_id IN (SELECT unnest(?)) AND session_date <= ?
                ORDER BY listing_id, session_date
                """,
                [requested, through],
            ).to_arrow_table()
            mapped = connection.execute(
                """
                SELECT listing_id FROM (
                    SELECT listing_id, row_number() OVER (
                        PARTITION BY listing_id ORDER BY effective_from DESC
                    ) AS newest
                    FROM provider_symbol_mapping
                    WHERE listing_id IN (SELECT unnest(?))
                )
                WHERE newest = 1
                """,
                [requested],
            ).fetchall()
            action_rows = connection.execute(
                """
                SELECT action.listing_id, action.provider, action.effective_date,
                       action.action_kind, action.new_shares_per_old_share,
                       action.cash_amount, action.provisional, action.provenance
                FROM corporate_action_current AS action
                JOIN (
                    SELECT listing_id, provider FROM (
                        SELECT listing_id, provider, row_number() OVER (
                            PARTITION BY listing_id ORDER BY effective_from DESC
                        ) AS newest
                        FROM provider_symbol_mapping
                        WHERE listing_id IN (SELECT unnest(?))
                    )
                    WHERE newest = 1
                ) AS mapping
                  ON mapping.listing_id = action.listing_id
                 AND mapping.provider = action.provider
                WHERE action.status = 'ACTIVE'
                ORDER BY action.listing_id, action.effective_date, action.action_kind
                """,
                [requested],
            ).fetchall()
        finally:
            connection.close()
        actions: dict[str, list[CorporateActionEvent]] = {str(row[0]): [] for row in mapped}
        for row in action_rows:
            actions[str(row[0])].append(CorporateActionEvent(*row))
        return QualityGovernanceInputs(
            admissions={str(row[1]): _quality_admission(row) for row in admission_rows},
            raw_close_series=series,
            actions={listing_id: tuple(events) for listing_id, events in actions.items()},
        )

    def admit_current_universe_maintenance(
        self,
        manifest: UniverseManifest,
        *,
        maintenance_id: str,
        as_of_session: date,
        observed_at: datetime,
    ) -> CurrentUniverseMaintenanceRun:
        """Persist one resumable 45-day refresh over a frozen research manifest."""
        self.bootstrap(manifest)
        observed = _utc_naive(observed_at)
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            existing = connection.execute(
                """
                SELECT research_manifest_id, research_manifest_revision, as_of_session
                FROM current_universe_maintenance WHERE maintenance_id = ?
                """,
                [maintenance_id],
            ).fetchone()
            expected = (manifest.manifest_id, manifest.revision_sha256, as_of_session)
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO current_universe_maintenance VALUES (?, ?, ?, ?, 'RUNNING', ?, ?)
                    """,
                    [maintenance_id, *expected, observed, observed],
                )
            elif tuple(existing) != expected:
                raise ValueError("maintenance identifier is bound to different immutable input")
            seeded = _first_by_key(
                [(listing.listing_id, listing.symbol) for listing in manifest.listings]
            )
            if seeded:
                with _staged_rows(
                    connection,
                    "maintenance_listing_stage",
                    {
                        "listing_id": pa.array([str(row[0]) for row in seeded]),
                        "symbol": pa.array([str(row[1]) for row in seeded]),
                    },
                ) as stage:
                    connection.execute(
                        f"""
                        INSERT INTO current_universe_maintenance_listing (
                            maintenance_id, listing_id, symbol, state, failure_code, raw_through,
                            change_json, attempt_count, updated_at
                        )
                        SELECT ?, listing_id, symbol, 'PENDING', NULL, NULL, NULL, 0, ?
                        FROM {stage}
                        ON CONFLICT (maintenance_id, listing_id) DO NOTHING
                        """,
                        [maintenance_id, observed],
                    )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return self.current_universe_maintenance_run(maintenance_id)

    def current_universe_maintenance_run(
        self, maintenance_id: str
    ) -> CurrentUniverseMaintenanceRun:
        """Load one durable current-universe maintenance run."""
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT maintenance_id, research_manifest_id, research_manifest_revision,
                       as_of_session, lifecycle, created_at, updated_at
                FROM current_universe_maintenance WHERE maintenance_id = ?
                """,
                [maintenance_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise ValueError("current-universe maintenance does not exist")
        return CurrentUniverseMaintenanceRun(*row)

    def current_universe_maintenance_counts(self, maintenance_id: str) -> tuple[int, int, int]:
        """One maintenance's units, and those UPDATED and FAILED, counted in the engine.

        The counts ``current_universe_maintenance_listings`` would give, without reading or
        parsing any unit's change document: what a run's progress reports.
        """
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT count(*), count(*) FILTER (WHERE state = 'UPDATED'),
                       count(*) FILTER (WHERE state = 'FAILED')
                FROM current_universe_maintenance_listing WHERE maintenance_id = ?
                """,
                [maintenance_id],
            ).fetchone()
        finally:
            connection.close()
        assert row is not None
        return int(row[0]), int(row[1]), int(row[2])

    def current_universe_maintenance_listings(
        self, maintenance_id: str
    ) -> tuple[CurrentUniverseMaintenanceListing, ...]:
        """Every unit of one maintenance, readable from a table bootstrap has not upgraded.

        The plan and the pages read maintenance records without writing, so
        a workspace whose table predates ``attempt_count`` or the grant
        columns is read by the columns it has: no attempts counted, no grant
        held. A grant is read only from a row carrying both its receipt and
        its wait; a half-present pair is not a grant.
        """
        connection = self._connect(read_only=True)
        try:
            columns = self._maintenance_listing_columns(connection)

            def present(name: str, absent: str) -> str:
                return name if name in columns else absent

            rows = connection.execute(
                f"""
                SELECT maintenance_id, listing_id, symbol, state, failure_code, raw_through,
                       change_json, {present("attempt_count", "0")}, updated_at,
                       {present("retry_grant_receipt_hash", "NULL")},
                       {present("retry_grant_wait_until", "NULL")},
                       {present("retry_grant_consumed_at", "NULL")}
                FROM current_universe_maintenance_listing
                WHERE maintenance_id = ? ORDER BY symbol
                """,
                [maintenance_id],
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            CurrentUniverseMaintenanceListing(
                *row[:6],
                json.loads(str(row[6])) if row[6] is not None else None,
                int(row[7]),
                row[8],
                _maintenance_retry_grant(row[9], row[10], row[11]),
            )
            for row in rows
        )

    @staticmethod
    def _maintenance_listing_columns(connection: duckdb.DuckDBPyConnection) -> set[str]:
        return {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info('current_universe_maintenance_listing')"
            ).fetchall()
        }

    def seed_current_universe_maintenance_verified_prefix(
        self,
        *,
        source_maintenance_id: str,
        target_maintenance_id: str,
        excluded_listing_ids: tuple[str, ...],
        observed_at: datetime,
    ) -> tuple[str, ...]:
        """Reuse durable successful listing units across an authorization boundary.

        A listing-scoped full-history authorization changes the maintenance
        execution identity, but it does not invalidate successful rolling units
        from the same manifest, as-of session, and policy generation.  Preserve
        their exact change documents so downstream feature invalidation still
        sees the verified prefix; authorized listings always remain pending.
        """
        if source_maintenance_id == target_maintenance_id:
            return ()
        observed = _utc_naive(observed_at)
        excluded = set(excluded_listing_ids)
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            source = connection.execute(
                """
                SELECT research_manifest_id, research_manifest_revision, as_of_session
                FROM current_universe_maintenance WHERE maintenance_id = ?
                """,
                [source_maintenance_id],
            ).fetchone()
            if source is None:
                connection.execute("COMMIT")
                return ()
            target = connection.execute(
                """
                SELECT research_manifest_id, research_manifest_revision, as_of_session
                FROM current_universe_maintenance WHERE maintenance_id = ?
                """,
                [target_maintenance_id],
            ).fetchone()
            if target is None:
                raise ValueError("target current-universe maintenance does not exist")
            if tuple(source) != tuple(target):
                raise ValueError("verified maintenance prefix has different immutable inputs")
            rows = connection.execute(
                """
                SELECT listing_id, raw_through, change_json
                FROM current_universe_maintenance_listing
                WHERE maintenance_id = ? AND state = 'UPDATED' AND change_json IS NOT NULL
                ORDER BY listing_id
                """,
                [source_maintenance_id],
            ).fetchall()
            seeded: list[str] = []
            for listing_id, raw_through, change_json in rows:
                identity = str(listing_id)
                if identity in excluded:
                    continue
                seeded_row = connection.execute(
                    """
                    UPDATE current_universe_maintenance_listing
                    SET state = 'UPDATED', failure_code = NULL, raw_through = ?,
                        change_json = ?, updated_at = ?
                    WHERE maintenance_id = ? AND listing_id = ? AND state = 'PENDING'
                    RETURNING listing_id
                    """,
                    [
                        raw_through,
                        change_json,
                        observed,
                        target_maintenance_id,
                        identity,
                    ],
                ).fetchone()
                if seeded_row is not None:
                    seeded.append(identity)
            connection.execute("COMMIT")
            return tuple(seeded)
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def begin_current_universe_maintenance_listing_attempt(
        self,
        *,
        maintenance_id: str,
        listing_id: str,
        observed_at: datetime,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> int:
        """Record one resumable listing attempt before any Provider call.

        The attempt consumes a held retry grant: the grant was for this
        attempt, whatever its outcome, so a later restart or continuation
        cannot spend it again.

        ``_connection`` is the caller's open transaction, which the caller commits.
        """
        connection = _connection or self._connect()
        try:
            row = connection.execute(
                """
                UPDATE current_universe_maintenance_listing
                SET attempt_count = attempt_count + 1, updated_at = ?,
                    retry_grant_consumed_at = CASE
                        WHEN retry_grant_receipt_hash IS NOT NULL
                             AND retry_grant_consumed_at IS NULL THEN ?
                        ELSE retry_grant_consumed_at END
                WHERE maintenance_id = ? AND listing_id = ? AND state = 'PENDING'
                RETURNING attempt_count
                """,
                [_utc_naive(observed_at), _utc_naive(observed_at), maintenance_id, listing_id],
            ).fetchone()
            if row is None:
                raise ValueError("maintenance listing is not pending")
            return int(row[0])
        finally:
            if _connection is None:
                connection.close()

    def requeue_failed_current_universe_maintenance_listings(
        self,
        *,
        maintenance_id: str,
        maximum_attempts: int,
        excluded_failure_codes: tuple[str, ...] = (),
        retry_grants: Mapping[str, tuple[str, datetime]] | None = None,
        observed_at: datetime,
    ) -> tuple[str, ...]:
        """Reopen only failed units that still have one governed attempt left.

        ``retry_grants`` maps a failed unit to the elapsed wait (execution
        receipt, wait instant) that grants it one more attempt past the
        budget. Granting is recorded on the row and held until the attempt
        begins; the same wait grants once, so a unit already granted by this
        receipt and instant (held or consumed) is not reopened by it again.
        A non-retryable failure is never reopened, granted or not.
        """
        if maximum_attempts < 1:
            raise ValueError("maximum maintenance attempts must be positive")
        connection = self._connect()
        try:
            exclusion_clause = ""
            exclusion_parameters: list[object] = []
            if excluded_failure_codes:
                placeholders = ",".join("?" for _ in excluded_failure_codes)
                exclusion_clause = f"AND failure_code NOT IN ({placeholders})"
                exclusion_parameters.extend(excluded_failure_codes)
            connection.execute("BEGIN TRANSACTION")
            rows = connection.execute(
                f"""
                UPDATE current_universe_maintenance_listing
                SET state = 'PENDING', failure_code = NULL, updated_at = ?
                WHERE maintenance_id = ? AND state = 'FAILED' AND attempt_count < ?
                  {exclusion_clause}
                RETURNING listing_id
                """,
                [_utc_naive(observed_at), maintenance_id, maximum_attempts, *exclusion_parameters],
            ).fetchall()
            for listing_id, (receipt_hash, wait_until) in sorted((retry_grants or {}).items()):
                rows.extend(
                    connection.execute(
                        f"""
                        UPDATE current_universe_maintenance_listing
                        SET state = 'PENDING', failure_code = NULL, updated_at = ?,
                            retry_grant_receipt_hash = ?, retry_grant_wait_until = ?,
                            retry_grant_consumed_at = NULL
                        WHERE maintenance_id = ? AND listing_id = ? AND state = 'FAILED'
                          AND NOT (retry_grant_receipt_hash IS NOT DISTINCT FROM ?
                                   AND retry_grant_wait_until IS NOT DISTINCT FROM ?)
                          {exclusion_clause}
                        RETURNING listing_id
                        """,
                        [
                            _utc_naive(observed_at),
                            receipt_hash,
                            _utc_naive(wait_until),
                            maintenance_id,
                            listing_id,
                            receipt_hash,
                            _utc_naive(wait_until),
                            *exclusion_parameters,
                        ],
                    ).fetchall()
                )
            if rows:
                connection.execute(
                    """
                    UPDATE current_universe_maintenance
                    SET lifecycle = 'RUNNING', updated_at = ?
                    WHERE maintenance_id = ?
                    """,
                    [_utc_naive(observed_at), maintenance_id],
                )
            connection.execute("COMMIT")
            return tuple(sorted(str(row[0]) for row in rows))
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def update_current_universe_maintenance_listing(
        self,
        *,
        maintenance_id: str,
        listing_id: str,
        state: str,
        observed_at: datetime,
        failure_code: str | None = None,
        raw_through: date | None = None,
        change_document: dict[str, object] | None = None,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Update one maintenance listing's state and observed evidence.

        ``_connection`` is the caller's open transaction, which the caller commits.
        """
        connection = _connection or self._connect()
        try:
            result = connection.execute(
                """
                UPDATE current_universe_maintenance_listing
                SET state = ?, failure_code = ?, raw_through = COALESCE(?, raw_through),
                    change_json = COALESCE(?, change_json), updated_at = ?
                WHERE maintenance_id = ? AND listing_id = ?
                """,
                [
                    state,
                    failure_code,
                    raw_through,
                    json.dumps(change_document, sort_keys=True, default=str)
                    if change_document is not None
                    else None,
                    _utc_naive(observed_at),
                    maintenance_id,
                    listing_id,
                ],
            )
            if result.rowcount == 0:
                raise ValueError("maintenance listing does not exist")
        finally:
            if _connection is None:
                connection.close()

    def set_current_universe_maintenance_lifecycle(
        self,
        maintenance_id: str,
        *,
        lifecycle: str,
        observed_at: datetime,
    ) -> None:
        """Set the lifecycle and observation time of a maintenance run."""
        connection = self._connect()
        try:
            result = connection.execute(
                """
                UPDATE current_universe_maintenance SET lifecycle = ?, updated_at = ?
                WHERE maintenance_id = ?
                """,
                [lifecycle, _utc_naive(observed_at), maintenance_id],
            )
            if result.rowcount == 0:
                raise ValueError("current-universe maintenance does not exist")
        finally:
            connection.close()

    def set_current_universe_onboarding_lifecycle(
        self,
        onboarding_id: str,
        *,
        lifecycle: str,
        observed_at: datetime,
    ) -> None:
        """Set the lifecycle and observation time of an onboarding run."""
        connection = self._connect()
        try:
            result = connection.execute(
                """
                UPDATE current_universe_onboarding SET lifecycle = ?, updated_at = ?
                WHERE onboarding_id = ?
                """,
                [lifecycle, _utc_naive(observed_at), onboarding_id],
            )
            if result.rowcount == 0:
                raise ValueError("current-universe onboarding does not exist")
        finally:
            connection.close()

    def complete_current_universe_onboarding(
        self,
        *,
        onboarding_id: str,
        research_manifest: UniverseManifest,
        quality_admission_hash: str,
        observed_at: datetime,
    ) -> CurrentUniverseOnboardingRun:
        """Publish the quality-qualified manifest link as the terminal outcome."""
        self.bootstrap(research_manifest)
        connection = self._connect()
        try:
            result = connection.execute(
                """
                UPDATE current_universe_onboarding
                SET lifecycle = 'COMPLETED', quality_admission_hash = ?, research_manifest_id = ?,
                    research_manifest_revision = ?, updated_at = ?
                WHERE onboarding_id = ?
                """,
                [
                    quality_admission_hash,
                    research_manifest.manifest_id,
                    research_manifest.revision_sha256,
                    _utc_naive(observed_at),
                    onboarding_id,
                ],
            )
            if result.rowcount == 0:
                raise ValueError("current-universe onboarding does not exist")
        finally:
            connection.close()
        return self.current_universe_onboarding_run(onboarding_id)

    @staticmethod
    def _assert_manifest_scope(manifest: UniverseManifest, listing_ids: Iterable[str]) -> None:
        allowed = {item.listing_id for item in manifest.listings}
        if any(listing_id not in allowed for listing_id in listing_ids):
            raise ValueError("data operation contains a listing outside the frozen manifest")

    @staticmethod
    def _current_actions(
        connection: duckdb.DuckDBPyConnection, listing_id: str, provider: str
    ) -> tuple[CorporateActionEvent, ...]:
        rows = connection.execute(
            """
            SELECT listing_id, provider, effective_date, action_kind,
                   new_shares_per_old_share, cash_amount, provisional, provenance
            FROM corporate_action_current
            WHERE listing_id = ? AND provider = ? AND status = 'ACTIVE'
            ORDER BY effective_date, action_kind
            """,
            [listing_id, provider],
        ).fetchall()
        return tuple(CorporateActionEvent(*row) for row in rows)

    @staticmethod
    def _record_action_revision(
        connection: duckdb.DuckDBPyConnection,
        event: CorporateActionEvent,
        *,
        revision_kind: str,
        prior_payload_hash: str | None,
        next_payload_hash: str | None,
        observed_at: datetime,
    ) -> None:
        revision_id = _canonical_hash(
            {
                "listing_id": event.listing_id,
                "provider": event.provider,
                "effective_date": event.effective_date,
                "action_kind": event.action_kind,
                "revision_kind": revision_kind,
                "prior_payload_hash": prior_payload_hash,
                "next_payload_hash": next_payload_hash,
            }
        )
        connection.execute(
            """
            INSERT INTO corporate_action_revision VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            [
                revision_id,
                event.listing_id,
                event.provider,
                event.effective_date,
                event.action_kind,
                revision_kind,
                prior_payload_hash,
                next_payload_hash,
                observed_at,
            ],
        )

    def _upsert_actions(
        self,
        connection: duckdb.DuckDBPyConnection,
        events: Sequence[CorporateActionEvent],
        *,
        observed_at: datetime,
    ) -> tuple[int, int]:
        """Upsert many events; the counts of INSERTED and of CORRECTED or RESTORED.

        The outcome of every event is the one ``_upsert_action`` decides --
        the current rows of the events' listings and providers are read
        once, an event without a current row is inserted, the rest are
        judged one by one -- and the inserts, the bulk of a first
        hydration, land as one relation instead of one statement each.
        """
        if not events:
            return 0, 0
        scopes = sorted({(event.listing_id, event.provider) for event in events})
        existing: dict[tuple[str, str, date, str], tuple[str, str]] = {}
        for listing_id, provider in scopes:
            for row in connection.execute(
                """
                SELECT effective_date, action_kind, payload_hash, status
                FROM corporate_action_current WHERE listing_id = ? AND provider = ?
                """,
                [listing_id, provider],
            ).fetchall():
                existing[(listing_id, provider, row[0], str(row[1]))] = (str(row[2]), str(row[3]))
        inserted = corrected = 0
        fresh: list[CorporateActionEvent] = []
        seen: set[tuple[str, str, date, str]] = set()
        for event in events:
            key = (event.listing_id, event.provider, event.effective_date, event.action_kind)
            if key in existing or key in seen:
                outcome = self._upsert_action(connection, event, observed_at=observed_at)
                inserted += int(outcome == "INSERTED")
                corrected += int(outcome in {"CORRECTED", "RESTORED"})
                continue
            seen.add(key)
            fresh.append(event)
        if fresh:
            stage = "corporate_action_stage"
            connection.register(
                stage,
                pa.table(
                    {
                        "listing_id": pa.array([e.listing_id for e in fresh], pa.string()),
                        "provider": pa.array([e.provider for e in fresh], pa.string()),
                        "effective_date": pa.array([e.effective_date for e in fresh], pa.date32()),
                        "action_kind": pa.array([e.action_kind for e in fresh], pa.string()),
                        "new_shares_per_old_share": pa.array(
                            [e.new_shares_per_old_share for e in fresh], pa.float64()
                        ),
                        "cash_amount": pa.array([e.cash_amount for e in fresh], pa.float64()),
                        "provisional": pa.array([e.provisional for e in fresh], pa.bool_()),
                        "provenance": pa.array([e.provenance for e in fresh], pa.string()),
                        "payload_hash": pa.array([_action_hash(e) for e in fresh], pa.string()),
                        "observed_at": pa.array([observed_at] * len(fresh), pa.timestamp("us")),
                    }
                ),
            )
            try:
                connection.execute(
                    f"""
                    INSERT INTO corporate_action_current
                    SELECT listing_id, provider, effective_date, action_kind,
                           new_shares_per_old_share, cash_amount, provisional, provenance,
                           payload_hash, 'ACTIVE', observed_at
                    FROM {stage}
                    """
                )
            finally:
                connection.unregister(stage)
            inserted += len(fresh)
        return inserted, corrected

    def _upsert_action(
        self,
        connection: duckdb.DuckDBPyConnection,
        event: CorporateActionEvent,
        *,
        observed_at: datetime,
    ) -> str:
        """Return INSERTED, CORRECTED, RESTORED, or UNCHANGED."""
        digest = _action_hash(event)
        existing = connection.execute(
            """
            SELECT payload_hash, status FROM corporate_action_current
            WHERE listing_id = ? AND provider = ? AND effective_date = ? AND action_kind = ?
            """,
            [event.listing_id, event.provider, event.effective_date, event.action_kind],
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO corporate_action_current VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?)
                """,
                [
                    event.listing_id,
                    event.provider,
                    event.effective_date,
                    event.action_kind,
                    event.new_shares_per_old_share,
                    event.cash_amount,
                    event.provisional,
                    event.provenance,
                    digest,
                    observed_at,
                ],
            )
            return "INSERTED"
        prior_hash, prior_status = str(existing[0]), str(existing[1])
        if prior_hash == digest and prior_status == "ACTIVE":
            return "UNCHANGED"
        revision_kind = (
            "RESTORED" if prior_status == "RETRACTED" and prior_hash == digest else "CORRECTED"
        )
        connection.execute(
            """
            UPDATE corporate_action_current
            SET new_shares_per_old_share = ?, cash_amount = ?, provisional = ?, provenance = ?,
                payload_hash = ?, status = 'ACTIVE', observed_at = ?
            WHERE listing_id = ? AND provider = ? AND effective_date = ? AND action_kind = ?
            """,
            [
                event.new_shares_per_old_share,
                event.cash_amount,
                event.provisional,
                event.provenance,
                digest,
                observed_at,
                event.listing_id,
                event.provider,
                event.effective_date,
                event.action_kind,
            ],
        )
        self._record_action_revision(
            connection,
            event,
            revision_kind=revision_kind,
            prior_payload_hash=prior_hash,
            next_payload_hash=digest,
            observed_at=observed_at,
        )
        return revision_kind

    def _retract_absent_actions(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        listing_id: str,
        provider: str,
        observed_natural_keys: set[tuple[date, str]],
        history_start: date,
        history_end: date,
        observed_at: datetime,
    ) -> int:
        rows = connection.execute(
            """
            SELECT listing_id, provider, effective_date, action_kind,
                   new_shares_per_old_share, cash_amount, provisional, provenance, payload_hash
            FROM corporate_action_current
            WHERE listing_id = ? AND provider = ? AND status = 'ACTIVE'
              AND effective_date BETWEEN ? AND ?
            """,
            [listing_id, provider, history_start, history_end],
        ).fetchall()
        retracted = 0
        for row in rows:
            event = CorporateActionEvent(*row[:8])
            if (event.effective_date, event.action_kind) in observed_natural_keys:
                continue
            prior_hash = str(row[8])
            connection.execute(
                """
                UPDATE corporate_action_current SET status = 'RETRACTED', observed_at = ?
                WHERE listing_id = ? AND provider = ? AND effective_date = ? AND action_kind = ?
                """,
                [
                    observed_at,
                    event.listing_id,
                    event.provider,
                    event.effective_date,
                    event.action_kind,
                ],
            )
            self._record_action_revision(
                connection,
                event,
                revision_kind="RETRACTED",
                prior_payload_hash=prior_hash,
                next_payload_hash=None,
                observed_at=observed_at,
            )
            retracted += 1
        return retracted

    def apply_validated_batch(
        self,
        manifest: UniverseManifest,
        batch: SanitizedBatch,
        *,
        ingestion_id: str,
        observed_at: datetime,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> dict[str, int]:
        """Apply a normal, bounded refresh atomically.

        The normal 45-day overlap may add or correct observed action rows, but
        never retracts an action that is simply absent from a partial refresh.
        Only :meth:`complete_action_audit` has that authority.
        ``_connection`` is the caller's open transaction, which the caller commits.
        """
        observed_at = _utc_naive(observed_at)
        self._assert_manifest_scope(manifest, (item.listing_id for item in batch.bars))
        self._assert_manifest_scope(manifest, (item.listing_id for item in batch.actions))
        if not batch.bars:
            raise ValueError("validated batch must contain at least one raw bar")
        bar_rows = [
            {
                "listing_id": bar.listing_id,
                "provider": bar.provider,
                "session_date": bar.session_date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "payload_hash": _bar_hash(bar),
                "observed_at": observed_at,
            }
            for bar in batch.bars
        ]
        stage_name = "validated_raw_bar_stage"
        connection = _connection or self._connect()
        inserted = corrected = action_inserted = action_corrected = 0
        try:
            if _connection is None:
                connection.execute("BEGIN TRANSACTION")
            connection.register(stage_name, pa.Table.from_pylist(bar_rows))
            inserted, corrected = connection.execute(
                f"""
                SELECT
                    count(*) FILTER (WHERE current.payload_hash IS NULL),
                    count(*) FILTER (
                        WHERE current.payload_hash IS NOT NULL
                          AND current.payload_hash <> s.payload_hash
                    )
                FROM {stage_name} AS s
                LEFT JOIN raw_daily_bar_current AS current
                  ON current.listing_id = s.listing_id
                 AND current.provider = s.provider
                 AND current.session_date = s.session_date
                """
            ).fetchone()
            inserted = int(inserted)
            corrected = int(corrected)
            if corrected:
                changed = connection.execute(
                    f"""
                    SELECT
                        s.listing_id,
                        s.provider,
                        s.session_date,
                        current.payload_hash AS prior_hash,
                        s.payload_hash AS next_hash
                    FROM {stage_name} AS s
                    JOIN raw_daily_bar_current AS current
                      ON current.listing_id = s.listing_id
                     AND current.provider = s.provider
                     AND current.session_date = s.session_date
                    WHERE current.payload_hash <> s.payload_hash
                    ORDER BY s.listing_id, s.provider, s.session_date
                    """
                ).fetchall()
                connection.executemany(
                    """
                    INSERT INTO bar_revision
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'provider_fact_correction')
                    ON CONFLICT DO NOTHING
                    """,
                    [
                        [
                            _canonical_hash(
                                [
                                    str(listing_id),
                                    str(provider),
                                    session_date,
                                    str(prior_hash),
                                    str(next_hash),
                                ]
                            ),
                            str(listing_id),
                            str(provider),
                            session_date,
                            str(prior_hash),
                            str(next_hash),
                            observed_at,
                        ]
                        for listing_id, provider, session_date, prior_hash, next_hash in changed
                    ],
                )
            connection.execute(
                f"""
                MERGE INTO raw_daily_bar_current AS current
                USING {stage_name} AS staged
                   ON current.listing_id = staged.listing_id
                  AND current.provider = staged.provider
                  AND current.session_date = staged.session_date
                WHEN MATCHED AND current.payload_hash <> staged.payload_hash THEN
                    UPDATE SET
                        open = staged.open,
                        high = staged.high,
                        low = staged.low,
                        close = staged.close,
                        volume = staged.volume,
                        payload_hash = staged.payload_hash,
                        observed_at = staged.observed_at
                WHEN NOT MATCHED THEN
                    INSERT (
                        listing_id, provider, session_date, open, high, low, close, volume,
                        payload_hash, observed_at
                    ) VALUES (
                        staged.listing_id, staged.provider, staged.session_date,
                        staged.open, staged.high, staged.low, staged.close, staged.volume,
                        staged.payload_hash, staged.observed_at
                    )
                """
            )
            connection.execute(
                f"""
                INSERT INTO data_quality
                SELECT
                    listing_id,
                    max(session_date),
                    'READY',
                    'validated raw payload',
                    max(observed_at)
                FROM {stage_name}
                GROUP BY listing_id
                ON CONFLICT (listing_id) DO UPDATE SET
                    latest_session = greatest(
                        data_quality.latest_session, excluded.latest_session
                    ),
                    quality_state = 'READY',
                    summary = excluded.summary,
                    updated_at = excluded.updated_at
                """
            )
            batch_inserted, batch_corrected = self._upsert_actions(
                connection, tuple(batch.actions), observed_at=observed_at
            )
            action_inserted += batch_inserted
            action_corrected += batch_corrected
            connection.execute(
                """
                INSERT INTO provider_attempt VALUES (?, ?, NULL, ?, ?, 'SUCCESS', ?)
                ON CONFLICT (attempt_id) DO NOTHING
                """,
                [
                    ingestion_id,
                    batch.bars[0].listing_id,
                    min(item.session_date for item in batch.bars),
                    max(item.session_date for item in batch.bars),
                    observed_at,
                ],
            )
            if _connection is None:
                connection.execute("COMMIT")
            return {
                "inserted": inserted,
                "corrected": corrected,
                "action_inserted": action_inserted,
                "action_corrected": action_corrected,
            }
        except Exception:
            if _connection is None:
                connection.execute("ROLLBACK")
            raise
        finally:
            with suppress(duckdb.Error):
                connection.unregister(stage_name)
            if _connection is None:
                connection.close()

    def _mapping_revision(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        listing_id: str,
        provider: str,
        as_of_session: date,
    ) -> str:
        rows = connection.execute(
            """
            SELECT provider_symbol, effective_from, effective_to, lifecycle_state
            FROM provider_symbol_mapping
            WHERE listing_id = ? AND provider = ? AND effective_from <= ?
              AND (effective_to IS NULL OR effective_to >= ?)
            ORDER BY effective_from
            """,
            [listing_id, provider, as_of_session, as_of_session],
        ).fetchall()
        if not rows:
            # This current-only manifest deliberately has no historical mapping
            # lifecycle yet.  Its one active mapping is a bootstrap identity,
            # not point-in-time listing evidence; future lifecycle authority
            # must remove this fallback rather than infer history from it.
            rows = connection.execute(
                """
                SELECT provider_symbol, effective_from, effective_to, lifecycle_state
                FROM provider_symbol_mapping
                WHERE listing_id = ? AND provider = ?
                ORDER BY effective_from
                """,
                [listing_id, provider],
            ).fetchall()
        if not rows:
            raise ValueError("listing has no provider symbol mapping")
        return _canonical_hash(rows)

    def provider_mapping_revision(
        self, *, listing_id: str, provider: str, as_of_session: date
    ) -> str:
        """Return one listing's provider-mapping identity at a session."""
        return self.provider_mapping_revisions(
            (listing_id,), provider=provider, as_of_session=as_of_session
        )[listing_id]

    def provider_mapping_revisions(
        self,
        listing_ids: tuple[str, ...],
        *,
        provider: str,
        as_of_session: date,
    ) -> dict[str, str]:
        """Resolve current mapping identities for many listings in one read."""
        ordered_ids = tuple(sorted(set(listing_ids)))
        if not ordered_ids:
            return {}
        placeholders = ",".join("?" for _ in ordered_ids)
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                f"""
                SELECT listing_id, provider_symbol, effective_from, effective_to,
                       lifecycle_state
                FROM provider_symbol_mapping
                WHERE provider = ? AND listing_id IN ({placeholders})
                ORDER BY listing_id, effective_from
                """,
                [provider, *ordered_ids],
            ).fetchall()
        finally:
            connection.close()
        grouped: dict[str, list[tuple[object, ...]]] = {
            listing_id: [] for listing_id in ordered_ids
        }
        for listing_id, *mapping in rows:
            grouped[str(listing_id)].append(tuple(mapping))
        revisions: dict[str, str] = {}
        for listing_id in ordered_ids:
            mappings = grouped[listing_id]
            if not mappings:
                raise ValueError("listing has no provider symbol mapping")
            active = [
                row
                for row in mappings
                if row[1] <= as_of_session and (row[2] is None or row[2] >= as_of_session)
            ]
            revisions[listing_id] = _canonical_hash(active or mappings)
        return revisions

    def historical_revision_since(
        self,
        *,
        listing_id: str,
        provider: str,
        observed_after: datetime,
        before_session: date,
    ) -> date | None:
        """Return the earliest newly observed correction outside the rolling window."""
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT min(affected_session) FROM (
                    SELECT session_date AS affected_session FROM bar_revision
                    WHERE listing_id = ? AND provider = ? AND observed_at > ?
                      AND session_date < ?
                    UNION ALL
                    SELECT effective_date AS affected_session
                    FROM corporate_action_revision
                    WHERE listing_id = ? AND provider = ? AND observed_at > ?
                      AND effective_date < ?
                )
                """,
                [
                    listing_id,
                    provider,
                    _utc_naive(observed_after),
                    before_session,
                    listing_id,
                    provider,
                    _utc_naive(observed_after),
                    before_session,
                ],
            ).fetchone()
        finally:
            connection.close()
        return row[0] if row is not None else None

    def historical_revisions_since(
        self,
        observed_after_by_listing: Mapping[str, datetime],
        *,
        provider: str,
        before_session: date,
    ) -> dict[str, date]:
        """Resolve old-session corrections for many audit anchors in one scan."""
        anchors = tuple(sorted(observed_after_by_listing.items()))
        if not anchors:
            return {}
        values = ",".join("(?, ?)" for _ in anchors)
        parameters: list[object] = []
        for listing_id, observed_after in anchors:
            parameters.extend((listing_id, _utc_naive(observed_after)))
        parameters.extend((provider, before_session, provider, before_session))
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                f"""
                WITH anchors(listing_id, observed_after) AS (VALUES {values}),
                revisions AS (
                    SELECT revision.listing_id,
                           revision.session_date AS affected_session
                    FROM bar_revision AS revision
                    JOIN anchors USING (listing_id)
                    WHERE revision.provider = ?
                      AND revision.observed_at > anchors.observed_after
                      AND revision.session_date < ?
                    UNION ALL
                    SELECT revision.listing_id,
                           revision.effective_date AS affected_session
                    FROM corporate_action_revision AS revision
                    JOIN anchors USING (listing_id)
                    WHERE revision.provider = ?
                      AND revision.observed_at > anchors.observed_after
                      AND revision.effective_date < ?
                )
                SELECT listing_id, min(affected_session)
                FROM revisions GROUP BY listing_id
                """,
                parameters,
            ).fetchall()
        finally:
            connection.close()
        return {str(listing_id): affected for listing_id, affected in rows}

    def _raw_evidence_hash(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        listing_id: str,
        provider: str,
        history_start: date,
        history_end: date,
    ) -> str:
        return self._raw_evidence_hashes(
            connection, {listing_id: (provider, history_start, history_end)}
        )[listing_id]

    @staticmethod
    def _raw_evidence_hashes(
        connection: duckdb.DuckDBPyConnection, ranges: Mapping[str, tuple[str, date, date]]
    ) -> dict[str, str]:
        """Each listing's raw-evidence hash over its own provider and sessions, read once.

        A listing's current payload hashes and bar revisions in its range, by session (and
        revisions by observation), under one provider: the evidence an action audit and a
        Feature write bind. ``ranges`` maps a listing to its provider and first and last session.
        """
        if not ranges:
            return {}
        scope = sorted(ranges)
        low = min(start for _provider, start, _end in ranges.values())
        high = max(end for _provider, _start, end in ranges.values())
        current: dict[str, list[tuple[object, ...]]] = {listing: [] for listing in scope}
        for listing, provider, *row in connection.execute(
            """
            SELECT listing_id, provider, session_date, payload_hash FROM raw_daily_bar_current
            WHERE listing_id IN (SELECT unnest(?::VARCHAR[])) AND session_date BETWEEN ? AND ?
            ORDER BY listing_id, session_date
            """,
            [scope, low, high],
        ).fetchall():
            selected, start, end = ranges[listing]
            if provider == selected and start <= row[0] <= end:
                current[listing].append(tuple(row))
        revisions: dict[str, list[tuple[object, ...]]] = {listing: [] for listing in scope}
        for listing, provider, *row in connection.execute(
            """
            SELECT listing_id, provider, session_date, prior_payload_hash, next_payload_hash,
                   observed_at
            FROM bar_revision
            WHERE listing_id IN (SELECT unnest(?::VARCHAR[])) AND session_date BETWEEN ? AND ?
            ORDER BY listing_id, session_date, observed_at
            """,
            [scope, low, high],
        ).fetchall():
            selected, start, end = ranges[listing]
            if provider == selected and start <= row[0] <= end:
                revisions[listing].append(tuple(row))
        return {
            listing: _canonical_hash({"current": current[listing], "revisions": revisions[listing]})
            for listing in scope
        }

    def _action_evidence_hash(
        self, connection: duckdb.DuckDBPyConnection, *, listing_id: str, provider: str
    ) -> str:
        current = connection.execute(
            """
            SELECT effective_date, action_kind, payload_hash, status
            FROM corporate_action_current WHERE listing_id = ? AND provider = ?
            ORDER BY effective_date, action_kind
            """,
            [listing_id, provider],
        ).fetchall()
        revisions = connection.execute(
            """
            SELECT effective_date, action_kind, revision_kind, prior_payload_hash, next_payload_hash
            FROM corporate_action_revision WHERE listing_id = ? AND provider = ?
            ORDER BY effective_date, action_kind, revision_id
            """,
            [listing_id, provider],
        ).fetchall()
        return _canonical_hash({"current": current, "revisions": revisions})

    @staticmethod
    def _adjusted_year_digests(values: Mapping[date, float]) -> dict[int, str]:
        years: dict[int, list[tuple[str, float]]] = {}
        for session in sorted(values):
            years.setdefault(session.year, []).append((session.isoformat(), float(values[session])))
        return {year: _canonical_hash(pairs) for year, pairs in years.items()}

    @classmethod
    def _provider_adjusted_series_hash(
        cls, values: Mapping[date, float], held: Mapping[int, str] | None = None
    ) -> str:
        """A series' digest over its calendar years' digests, in order (year blocks).

        `held` gives the digests of closed years whose values are not in `values` (their facts).
        """
        return _canonical_hash(
            sorted({**(held or {}), **cls._adjusted_year_digests(values)}.items())
        )

    @staticmethod
    def _provider_adjusted_return_changes(
        prior: Mapping[date, float],
        current: Mapping[date, float],
        *,
        refreshed_from: date | None = None,
    ) -> tuple[date, ...]:
        """The sessions whose log return differs between two adjusted series.

        `refreshed_from` names the first session a rolling refresh replaced, every earlier one
        kept as it was: a return can then change only from the kept session before it, so the
        earlier returns, which read the same two closes on both sides, are not compared.
        """
        if refreshed_from is not None:
            since = max((session for session in prior if session < refreshed_from), default=None)
            if since is not None:
                prior = {session: close for session, close in prior.items() if session >= since}
                current = {session: close for session, close in current.items() if session >= since}

        def returns(values: Mapping[date, float]) -> dict[date, float]:
            sessions = sorted(values)
            return {
                session: math.log(float(values[session]) / float(values[previous]))
                for previous, session in pairwise(sessions)
            }

        old_returns = returns(prior)
        new_returns = returns(current)
        return tuple(
            session
            for session in sorted(set(old_returns) | set(new_returns))
            if session not in old_returns
            or session not in new_returns
            or abs(new_returns[session] - old_returns[session]) > 1e-12
        )

    def _reconcile_provider_adjusted_series(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        listing_id: str,
        provider: str,
        points: Sequence[ProviderAdjustedClosePoint],
        scope_start: date,
        scope_end: date,
        full_history: bool,
        earliest_session: date | None,
        source_receipt_hash: str,
        observed_at: datetime,
    ) -> ProviderAdjustedSeriesRevision:
        """Write a refresh of the listing's adjusted series and record its revision.

        A rolling refresh reads the stored series only from its scope's calendar year: each
        earlier year, back to the first raw bar's, is held by its fact (`ADJUSTED_YEAR`), or the
        whole series is read. Every closed year it read is recorded as it leaves it; a full
        refresh first forgets every year of the series it replaces.
        """
        ordered = tuple(sorted(points, key=lambda item: item.session_date))
        if not ordered or len({item.session_date for item in ordered}) != len(ordered):
            raise ValueError("provider adjusted series must be non-empty and session-unique")
        if any(
            item.listing_id != listing_id
            or item.provider != provider
            or not scope_start <= item.session_date <= scope_end
            or not math.isfinite(item.adjusted_close)
            or item.adjusted_close <= 0.0
            for item in ordered
        ):
            raise ValueError("provider adjusted series is outside its declared scope")
        scope = _adjusted_scope(listing_id, provider)
        facts: dict[int, str] = {}
        held: dict[int, str] = {}
        if not full_history and earliest_session is not None:
            # The written years too, so a fact this refresh does not renew can be ended.
            facts = year_facts(
                connection,
                kind=ADJUSTED_YEAR,
                scope=scope,
                basis=_ADJUSTED_YEAR_BASIS,
                epoch=self.database.seal_epoch(),
                years=range(earliest_session.year, scope_end.year + 1),
            )
            closed = range(earliest_session.year, scope_start.year)
            if all(year in facts for year in closed):
                held = {year: facts[year] for year in closed}

        def stored(start: date) -> dict[date, float]:
            rows = connection.execute(
                """
                SELECT session_date, adjusted_close FROM provider_adjusted_close_current
                WHERE listing_id = ? AND provider = ? AND session_date >= ? ORDER BY session_date
                """,
                [listing_id, provider, start],
            ).fetchall()
            return {row[0]: float(row[1]) for row in rows}

        observed = {item.session_date: float(item.adjusted_close) for item in ordered}
        prior = stored(date(scope_start.year, 1, 1) if held else date.min)
        if held and not any(session < ordered[0].session_date for session in prior):
            # The refresh's first return reads the session before it, in a closed year.
            held, prior = {}, stored(date.min)
        prior_in_scope = {
            session: value
            for session, value in prior.items()
            if scope_start <= session <= scope_end
        }
        if set(prior_in_scope) - set(observed):
            raise ActionAuditScopeInsufficient(
                "provider adjusted refresh omitted a canonical session"
            )
        next_values = dict(prior)
        next_values.update(observed)
        added_sessions = set(observed) - set(prior)
        if full_history and prior and any(session <= max(prior) for session in added_sessions):
            raise ValueError("full provider adjusted series changed its historical session set")
        if full_history:
            next_values = observed
        prior_hash = self._provider_adjusted_series_hash(prior, held) if prior or held else None
        next_hash = self._provider_adjusted_series_hash(next_values, held)
        changed_values = sum(
            1 for session, value in observed.items() if prior.get(session) != value
        )
        if full_history:
            forget_year_facts(
                connection, ADJUSTED_YEAR, pairs=[(scope, session.year) for session in prior]
            )
        session_set_changed = bool(prior or held) and set(prior) != set(next_values)
        return_changes = (
            self._provider_adjusted_return_changes(
                prior, next_values, refreshed_from=None if full_history else min(observed)
            )
            if prior
            else ()
        )
        uniform_rescale = bool(changed_values) and not return_changes and not session_set_changed
        stage_name = "provider_adjusted_close_stage"
        stage = pa.Table.from_pylist(
            [
                {
                    "listing_id": listing_id,
                    "provider": provider,
                    "session_date": item.session_date,
                    "adjusted_close": float(item.adjusted_close),
                    "payload_hash": _canonical_hash(
                        [listing_id, provider, item.session_date.isoformat(), item.adjusted_close]
                    ),
                    "updated_at": observed_at,
                }
                for item in ordered
            ]
        )
        connection.register(stage_name, stage)
        try:
            connection.execute(
                """
                INSERT OR REPLACE INTO provider_adjusted_close_current
                SELECT listing_id, provider, session_date, adjusted_close, payload_hash, updated_at
                FROM provider_adjusted_close_stage
                """
            )
        finally:
            connection.unregister(stage_name)
        # The closed years this refresh read, as it leaves them, are held for the next one; a
        # written year's fact it does not renew ends.
        renewed = (
            {
                year: digest
                for year, digest in self._adjusted_year_digests(next_values).items()
                if year < scope_end.year
            }
            if min(next_values).year < scope_end.year
            else {}
        )
        record_year_facts(
            connection,
            kind=ADJUSTED_YEAR,
            scope=scope,
            basis=_ADJUSTED_YEAR_BASIS,
            epoch=self.database.seal_epoch(),
            values=renewed,
        )
        forget_year_facts(
            connection,
            ADJUSTED_YEAR,
            pairs=[
                (scope, session.year)
                for session in observed
                if session.year in facts and session.year not in renewed
            ],
        )
        payload = {
            "listing_id": listing_id,
            "provider": provider,
            "prior_series_hash": prior_hash,
            "next_series_hash": next_hash,
            "scope_start": scope_start,
            "scope_end": scope_end,
            "full_history": full_history,
            "changed_value_count": changed_values,
            "changed_return_sessions": [item.isoformat() for item in return_changes],
            "session_set_changed": session_set_changed,
            "uniform_rescale": uniform_rescale,
            "source_receipt_hash": source_receipt_hash,
        }
        revision = ProviderAdjustedSeriesRevision(
            receipt_hash=_canonical_hash(payload),
            listing_id=listing_id,
            provider=provider,
            prior_series_hash=prior_hash,
            next_series_hash=next_hash,
            scope_start=scope_start,
            scope_end=scope_end,
            full_history=full_history,
            changed_value_count=changed_values,
            changed_return_sessions=return_changes,
            session_set_changed=session_set_changed,
            uniform_rescale=uniform_rescale,
            source_receipt_hash=source_receipt_hash,
            observed_at=_utc_aware(observed_at),
        )
        if prior_hash != next_hash:
            connection.execute(
                """
                INSERT INTO provider_adjusted_series_revision VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                ) ON CONFLICT (receipt_hash) DO NOTHING
                """,
                [
                    revision.receipt_hash,
                    revision.listing_id,
                    revision.provider,
                    revision.prior_series_hash,
                    revision.next_series_hash,
                    revision.scope_start,
                    revision.scope_end,
                    revision.full_history,
                    revision.changed_value_count,
                    json.dumps(
                        [item.isoformat() for item in revision.changed_return_sessions],
                        separators=(",", ":"),
                    ),
                    revision.session_set_changed,
                    revision.uniform_rescale,
                    revision.source_receipt_hash,
                    observed_at,
                ],
            )
        return revision

    def provider_adjusted_closes(
        self,
        listing_id: str,
        *,
        through: date,
        start: date | None = None,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> tuple[ProviderAdjustedClosePoint, ...]:
        """Read provider-adjusted closes for one listing within a date range.

        ``_connection`` is the caller's connection: it sees that transaction's own writes.
        """
        connection = _connection or self._connect(read_only=True)
        try:
            provider = self._provider_for_listing(connection, listing_id)
            rows = connection.execute(
                """
                SELECT session_date, adjusted_close FROM provider_adjusted_close_current
                WHERE listing_id = ? AND provider = ? AND session_date BETWEEN ? AND ?
                ORDER BY session_date
                """,
                [listing_id, provider, start or date.min, through],
            ).fetchall()
            return tuple(
                ProviderAdjustedClosePoint(listing_id, provider, row[0], float(row[1]))
                for row in rows
            )
        finally:
            if _connection is None:
                connection.close()

    def provider_adjusted_close_panel(
        self,
        *,
        listing_ids: Sequence[str],
        sessions: Sequence[date],
    ) -> tuple[ProviderAdjustedClosePoint, ...]:
        """Bulk-read existing provider Adj Close rows for a frozen research slice.

        This is a Data Operations read boundary.  Factor Research receives only
        the immutable outcome snapshot published from these rows and never a
        connection, SQL string, or mutable store handle.
        """
        listings = tuple(sorted(set(str(value) for value in listing_ids)))
        ordered_sessions = tuple(sorted(set(sessions)))
        if not listings or len(listings) != len(tuple(listing_ids)):
            raise ValueError("adjusted-close panel listings must be non-empty and unique")
        if not ordered_sessions or len(ordered_sessions) != len(tuple(sessions)):
            raise ValueError("adjusted-close panel sessions must be non-empty and unique")
        connection = self._connect(read_only=True)
        try:
            # Both axes are already deduplicated above, and both targets carry a
            # key, so a per-row insert pays an index probe for each of the ~3,000
            # rows a full panel read requests. The key stays -- it is the
            # uniqueness guard -- but the rows arrive in one statement.
            connection.execute(
                "CREATE TEMP TABLE requested_factor_listing(listing_id VARCHAR PRIMARY KEY)"
            )
            with _staged_rows(
                connection,
                "requested_factor_listing_stage",
                {"listing_id": pa.array([str(value) for value in listings])},
            ) as stage:
                connection.execute(
                    f"INSERT INTO requested_factor_listing SELECT listing_id FROM {stage}"
                )
            connection.execute(
                "CREATE TEMP TABLE requested_factor_session(session_date DATE PRIMARY KEY)"
            )
            with _staged_rows(
                connection,
                "requested_factor_session_stage",
                {"session_date": pa.array(list(ordered_sessions), pa.date32())},
            ) as stage:
                connection.execute(
                    f"INSERT INTO requested_factor_session SELECT session_date FROM {stage}"
                )
            rows = connection.execute(
                """
                WITH current_provider AS (
                    SELECT listing_id, provider,
                           row_number() OVER (
                               PARTITION BY listing_id ORDER BY effective_from DESC
                           ) AS provider_rank
                    FROM provider_symbol_mapping
                    WHERE listing_id IN (SELECT listing_id FROM requested_factor_listing)
                )
                SELECT adjusted.listing_id, adjusted.provider,
                       adjusted.session_date, adjusted.adjusted_close
                FROM provider_adjusted_close_current AS adjusted
                JOIN current_provider AS provider
                  ON provider.listing_id = adjusted.listing_id
                 AND provider.provider = adjusted.provider
                 AND provider.provider_rank = 1
                JOIN requested_factor_session AS requested
                  ON requested.session_date = adjusted.session_date
                ORDER BY adjusted.listing_id, adjusted.session_date
                """
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            ProviderAdjustedClosePoint(
                listing_id=str(listing_id),
                provider=str(provider),
                session_date=session,
                adjusted_close=float(adjusted_close),
            )
            for listing_id, provider, session, adjusted_close in rows
        )

    def provider_adjusted_revision(
        self, source_receipt_hash: str, *, _connection: duckdb.DuckDBPyConnection | None = None
    ) -> ProviderAdjustedSeriesRevision | None:
        """Read a series revision linked to one source receipt, if present.

        ``_connection`` is the caller's connection: it sees that transaction's own writes.
        """
        connection = _connection or self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT receipt_hash, listing_id, provider, prior_series_hash,
                       next_series_hash, scope_start, scope_end, full_history,
                       changed_value_count, changed_return_sessions_json,
                       session_set_changed, uniform_rescale, source_receipt_hash,
                       observed_at
                FROM provider_adjusted_series_revision
                WHERE source_receipt_hash = ? ORDER BY observed_at DESC LIMIT 1
                """,
                [source_receipt_hash],
            ).fetchone()
            if row is None:
                return None
            return ProviderAdjustedSeriesRevision(
                receipt_hash=str(row[0]),
                listing_id=str(row[1]),
                provider=str(row[2]),
                prior_series_hash=str(row[3]) if row[3] is not None else None,
                next_series_hash=str(row[4]),
                scope_start=row[5],
                scope_end=row[6],
                full_history=bool(row[7]),
                changed_value_count=int(row[8]),
                changed_return_sessions=tuple(
                    date.fromisoformat(item) for item in json.loads(str(row[9]))
                ),
                session_set_changed=bool(row[10]),
                uniform_rescale=bool(row[11]),
                source_receipt_hash=str(row[12]),
                observed_at=_utc_aware(row[13]),
            )
        finally:
            if _connection is None:
                connection.close()

    def provider_adjusted_semantic_ledger(
        self,
    ) -> tuple[ProviderAdjustedSeriesRevision, ...]:
        """Return the monotonic semantic revision ledger for frozen-label preparation."""
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT receipt_hash, listing_id, provider, prior_series_hash,
                       next_series_hash, scope_start, scope_end, full_history,
                       changed_value_count, changed_return_sessions_json,
                       session_set_changed, uniform_rescale, source_receipt_hash,
                       observed_at
                FROM provider_adjusted_series_revision
                ORDER BY observed_at, receipt_hash
                """
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            ProviderAdjustedSeriesRevision(
                receipt_hash=str(row[0]),
                listing_id=str(row[1]),
                provider=str(row[2]),
                prior_series_hash=str(row[3]) if row[3] is not None else None,
                next_series_hash=str(row[4]),
                scope_start=row[5],
                scope_end=row[6],
                full_history=bool(row[7]),
                changed_value_count=int(row[8]),
                changed_return_sessions=tuple(
                    date.fromisoformat(item) for item in json.loads(str(row[9]))
                ),
                session_set_changed=bool(row[10]),
                uniform_rescale=bool(row[11]),
                source_receipt_hash=str(row[12]),
                observed_at=_utc_aware(row[13]),
            )
            for row in rows
        )

    def provider_adjusted_semantic_deltas_since(
        self,
        manifest: UniverseManifest,
        *,
        observed_after: datetime,
        through: date,
    ) -> tuple[ProviderAdjustedSeriesRevision, ...]:
        """Return manifest-scoped adjusted-return revisions after a verified panel prefix."""
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT revision.receipt_hash, revision.listing_id, revision.provider,
                       revision.prior_series_hash, revision.next_series_hash,
                       revision.scope_start, revision.scope_end, revision.full_history,
                       revision.changed_value_count,
                       revision.changed_return_sessions_json,
                       revision.session_set_changed, revision.uniform_rescale,
                       revision.source_receipt_hash, revision.observed_at
                FROM provider_adjusted_series_revision AS revision
                JOIN universe_manifest_listing AS member USING (listing_id)
                WHERE member.manifest_id = ? AND revision.provider = ?
                  AND revision.observed_at > ? AND revision.scope_start <= ?
                ORDER BY revision.observed_at, revision.receipt_hash
                """,
                [
                    manifest.manifest_id,
                    manifest.profile.provider,
                    _utc_naive(observed_after),
                    through,
                ],
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            ProviderAdjustedSeriesRevision(
                receipt_hash=str(row[0]),
                listing_id=str(row[1]),
                provider=str(row[2]),
                prior_series_hash=str(row[3]) if row[3] is not None else None,
                next_series_hash=str(row[4]),
                scope_start=row[5],
                scope_end=row[6],
                full_history=bool(row[7]),
                changed_value_count=int(row[8]),
                changed_return_sessions=tuple(
                    session
                    for session in (date.fromisoformat(item) for item in json.loads(str(row[9])))
                    if session <= through
                ),
                session_set_changed=bool(row[10]),
                uniform_rescale=bool(row[11]),
                source_receipt_hash=str(row[12]),
                observed_at=_utc_aware(row[13]),
            )
            for row in rows
        )

    def raw_bar_semantic_deltas_since(
        self,
        manifest: UniverseManifest,
        *,
        observed_after: datetime,
        through: date,
    ) -> tuple[FeatureSourceDelta, ...]:
        """Return exact corrected raw sessions after a verified panel prefix."""
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT revision.listing_id, revision.session_date,
                       revision.prior_payload_hash, revision.next_payload_hash,
                       revision.observed_at
                FROM bar_revision AS revision
                JOIN universe_manifest_listing AS member USING (listing_id)
                WHERE member.manifest_id = ? AND revision.provider = ?
                  AND revision.observed_at > ? AND revision.session_date <= ?
                ORDER BY revision.listing_id, revision.session_date,
                         revision.observed_at, revision.revision_id
                """,
                [
                    manifest.manifest_id,
                    manifest.profile.provider,
                    _utc_naive(observed_after),
                    through,
                ],
            ).fetchall()
        finally:
            connection.close()
        grouped: dict[str, list[tuple[object, ...]]] = {}
        for row in rows:
            grouped.setdefault(str(row[0]), []).append(tuple(row[1:]))
        return tuple(
            FeatureSourceDelta(
                listing_id=listing_id,
                sessions=tuple(sorted({row[0] for row in revisions})),
                evidence_hash=_canonical_hash(
                    {
                        "kind": "raw_bar_semantic_delta",
                        "listing_id": listing_id,
                        "revisions": revisions,
                    }
                ),
            )
            for listing_id, revisions in sorted(grouped.items())
        )

    def feature_source_appends(
        self,
        manifest: UniverseManifest,
        *,
        catalog_hash: str,
        through: date,
        include_incomplete: bool = False,
    ) -> tuple[FeatureSourceDelta, ...]:
        """Find tail work beyond each listing's materialized Feature prefix.

        Strict callers require both sources. An availability-aware caller may
        schedule incomplete rows, whose evidence retains the absent adjusted
        hash as None. This is invalidation scope, never numerical admission.
        """
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                WITH latest_feature AS (
                    SELECT member.listing_id, max(feature.session_date) AS latest_session
                    FROM universe_manifest_listing AS member
                    LEFT JOIN feature_daily_runtime AS feature
                      ON feature.listing_id = member.listing_id
                     AND feature.catalog_hash = ?
                    WHERE member.manifest_id = ?
                    GROUP BY member.listing_id
                )
                SELECT raw.listing_id, raw.session_date, raw.payload_hash,
                       adjusted.payload_hash
                FROM latest_feature AS latest
                JOIN raw_daily_bar_current AS raw
                  ON raw.listing_id = latest.listing_id AND raw.provider = ?
                LEFT JOIN provider_adjusted_close_current AS adjusted
                  ON adjusted.listing_id = raw.listing_id
                 AND adjusted.provider = raw.provider
                 AND adjusted.session_date = raw.session_date
                WHERE (latest.latest_session IS NULL
                       OR raw.session_date > latest.latest_session)
                  AND raw.session_date <= ?
                ORDER BY raw.listing_id, raw.session_date
                """,
                [catalog_hash, manifest.manifest_id, manifest.profile.provider, through],
            ).fetchall()
        finally:
            connection.close()
        missing_adjusted = tuple((str(row[0]), row[1]) for row in rows if row[3] is None)
        if missing_adjusted and not include_incomplete:
            raise ValueError("feature append source is missing provider adjusted close")
        grouped: dict[str, list[tuple[object, ...]]] = {}
        for listing_id, session, raw_hash, adjusted_hash in rows:
            grouped.setdefault(str(listing_id), []).append(
                (session, str(raw_hash), str(adjusted_hash) if adjusted_hash is not None else None)
            )
        return tuple(
            FeatureSourceDelta(
                listing_id=listing_id,
                sessions=tuple(row[0] for row in source_rows),
                evidence_hash=_canonical_hash(
                    {
                        "kind": "feature_source_append",
                        "listing_id": listing_id,
                        "rows": source_rows,
                    }
                ),
            )
            for listing_id, source_rows in sorted(grouped.items())
        )

    def verified_feature_source_session_sets(
        self,
        manifest: UniverseManifest,
        *,
        catalog_hash: str,
        through: date,
    ) -> frozenset[str]:
        """Return listings whose source calendar is a verified tail append.

        The already materialized feature rows are the consumed session prefix.
        Raw and provider-adjusted sources must agree exactly, every historical
        source session through the materialized prefix must still have a
        feature row, and no feature row may outlive either source.  Sessions
        beyond the prefix are therefore the only admissible set change.
        """
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                WITH members AS (
                    SELECT listing_id FROM universe_manifest_listing
                    WHERE manifest_id = ?
                ),
                latest_feature AS (
                    SELECT member.listing_id,
                           max(feature.session_date) AS latest_session
                    FROM members AS member
                    LEFT JOIN feature_daily_runtime AS feature
                      ON feature.listing_id = member.listing_id
                     AND feature.catalog_hash = ?
                     AND feature.session_date <= ?
                    GROUP BY member.listing_id
                )
                SELECT member.listing_id,
                       EXISTS (
                           SELECT 1 FROM raw_daily_bar_current AS raw
                           WHERE raw.listing_id = member.listing_id
                             AND raw.provider = ? AND raw.session_date <= ?
                       )
                       AND NOT EXISTS (
                           SELECT 1 FROM raw_daily_bar_current AS raw
                           LEFT JOIN provider_adjusted_close_current AS adjusted
                             ON adjusted.listing_id = raw.listing_id
                            AND adjusted.provider = raw.provider
                            AND adjusted.session_date = raw.session_date
                           LEFT JOIN feature_daily_runtime AS feature
                             ON feature.listing_id = raw.listing_id
                            AND feature.catalog_hash = ?
                            AND feature.session_date = raw.session_date
                           WHERE raw.listing_id = member.listing_id
                             AND raw.provider = ? AND raw.session_date <= ?
                             AND (
                                 adjusted.session_date IS NULL
                                 OR (
                                     latest.latest_session IS NOT NULL
                                     AND raw.session_date <= latest.latest_session
                                     AND feature.session_date IS NULL
                                 )
                             )
                       )
                       AND NOT EXISTS (
                           SELECT 1
                           FROM provider_adjusted_close_current AS adjusted
                           LEFT JOIN raw_daily_bar_current AS raw
                             ON raw.listing_id = adjusted.listing_id
                            AND raw.provider = adjusted.provider
                            AND raw.session_date = adjusted.session_date
                           WHERE adjusted.listing_id = member.listing_id
                             AND adjusted.provider = ?
                             AND adjusted.session_date <= ?
                             AND raw.session_date IS NULL
                       )
                       AND NOT EXISTS (
                           SELECT 1 FROM feature_daily_runtime AS feature
                           LEFT JOIN raw_daily_bar_current AS raw
                             ON raw.listing_id = feature.listing_id
                            AND raw.provider = ?
                            AND raw.session_date = feature.session_date
                           LEFT JOIN provider_adjusted_close_current AS adjusted
                             ON adjusted.listing_id = feature.listing_id
                            AND adjusted.provider = ?
                            AND adjusted.session_date = feature.session_date
                           WHERE feature.listing_id = member.listing_id
                             AND feature.catalog_hash = ?
                             AND feature.session_date <= ?
                             AND (raw.session_date IS NULL
                                  OR adjusted.session_date IS NULL)
                       ) AS verified
                FROM members AS member
                JOIN latest_feature AS latest USING (listing_id)
                ORDER BY member.listing_id
                """,
                [
                    manifest.manifest_id,
                    catalog_hash,
                    through,
                    manifest.profile.provider,
                    through,
                    catalog_hash,
                    manifest.profile.provider,
                    through,
                    manifest.profile.provider,
                    through,
                    manifest.profile.provider,
                    manifest.profile.provider,
                    catalog_hash,
                    through,
                ],
            ).fetchall()
        finally:
            connection.close()
        return frozenset(str(row[0]) for row in rows if bool(row[1]))

    def complete_action_audit(
        self,
        manifest: UniverseManifest,
        *,
        listing_id: str,
        provider: str,
        observed_actions: Sequence[CorporateActionEvent],
        observed_adjusted_closes: Sequence[ProviderAdjustedClosePoint],
        history_start: date,
        history_end: date,
        requested_as_of: date,
        observed_at: datetime,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> tuple[ActionAuditReceipt, dict[str, int]]:
        """Reconcile one bounded or full provider action observation and issue a receipt.

        The caller is the deterministic provider adapter. Missing current actions
        are marked ``RETRACTED`` only inside the declared scope, never silently
        deleted. Adjusted closes remain audit evidence only.
        ``_connection`` is the caller's open transaction, which the caller commits.
        """
        observed_at = _utc_naive(observed_at)
        self._assert_manifest_scope(manifest, [listing_id])
        if history_start > history_end or history_end < requested_as_of:
            raise ValueError("action audit range must cover requested as_of")
        events = tuple(observed_actions)
        if any(
            item.listing_id != listing_id
            or item.provider != provider
            or not history_start <= item.effective_date <= history_end
            for item in events
        ):
            raise ValueError("action audit observation is outside its declared scope")
        natural_keys = {(item.effective_date, item.action_kind) for item in events}
        if len(natural_keys) != len(events):
            raise ValueError("action audit contains duplicate natural keys")
        adjusted_closes = tuple(observed_adjusted_closes)
        if any(
            item.listing_id != listing_id
            or item.provider != provider
            or not history_start <= item.session_date <= history_end
            for item in adjusted_closes
        ):
            raise ValueError("adjusted-close audit observation is outside its declared scope")
        connection = _connection or self._connect()
        inserted = corrected = retracted = 0
        try:
            if _connection is None:
                connection.execute("BEGIN TRANSACTION")
            inserted, corrected = self._upsert_actions(connection, events, observed_at=observed_at)
            retracted = self._retract_absent_actions(
                connection,
                listing_id=listing_id,
                provider=provider,
                observed_natural_keys=natural_keys,
                history_start=history_start,
                history_end=history_end,
                observed_at=observed_at,
            )
            active_actions = self._current_actions(connection, listing_id, provider)
            scoped_actions = tuple(
                item
                for item in active_actions
                if history_start <= item.effective_date <= history_end
            )
            rows = connection.execute(
                """
                SELECT listing_id, provider, session_date, open, high, low, close, volume
                FROM raw_daily_bar_current
                WHERE listing_id = ? AND provider = ? AND session_date BETWEEN ? AND ?
                ORDER BY session_date
                """,
                [listing_id, provider, history_start, history_end],
            ).fetchall()
            bars = tuple(RawDailyBar(*row) for row in rows)
            expected_sessions = {bar.session_date for bar in bars}
            observed_sessions = {item.session_date for item in adjusted_closes}
            if (
                not bars
                or expected_sessions != observed_sessions
                or len(adjusted_closes) != len(observed_sessions)
            ):
                raise ActionAuditScopeInsufficient(
                    "adjusted-close audit does not exactly cover canonical sessions"
                )
            diagnostics = provider_adjusted_ratio_diagnostics(
                bars,
                scoped_actions,
                adjusted_closes,
                daily_price_basis=manifest.profile.daily_price_basis,
            )
            mismatches = tuple(item for item in diagnostics if item.difference_bps > 5.0)
            adjusted_close_evidence = provider_adjusted_close_evidence_hash(adjusted_closes)
            action_digest = action_set_hash(active_actions)
            mapping_revision = self._mapping_revision(
                connection,
                listing_id=listing_id,
                provider=provider,
                as_of_session=requested_as_of,
            )
            raw_digest = self._raw_evidence_hash(
                connection,
                listing_id=listing_id,
                provider=provider,
                history_start=history_start,
                history_end=history_end,
            )
            action_evidence = self._action_evidence_hash(
                connection, listing_id=listing_id, provider=provider
            )
            receipt_payload = {
                "listing_id": listing_id,
                "provider": provider,
                "manifest_revision": manifest.revision_sha256,
                "mapping_revision": mapping_revision,
                "requested_as_of": requested_as_of,
                "history_start": history_start,
                "history_end": history_end,
                "raw_evidence_hash": raw_digest,
                "action_set_hash": action_digest,
                "action_evidence_hash": action_evidence,
                "provider_adjusted_close_evidence_hash": adjusted_close_evidence,
                "max_adjusted_close_difference_bps": max(
                    (item.difference_bps for item in diagnostics), default=0.0
                ),
                "adjusted_close_mismatch_count": len(mismatches),
                "first_adjusted_close_mismatch_session": (
                    mismatches[0].session_date if mismatches else None
                ),
                "diagnostic_policy_hash": ADJUSTED_CLOSE_DIAGNOSTIC_POLICY_HASH,
                "observed_at": observed_at,
            }
            receipt = ActionAuditReceipt(_canonical_hash(receipt_payload), **receipt_payload)
            earliest_session = connection.execute(
                """
                SELECT min(session_date) FROM raw_daily_bar_current
                WHERE listing_id = ? AND provider = ?
                """,
                [listing_id, provider],
            ).fetchone()[0]
            self._reconcile_provider_adjusted_series(
                connection,
                listing_id=listing_id,
                provider=provider,
                points=adjusted_closes,
                scope_start=history_start,
                scope_end=history_end,
                full_history=history_start == earliest_session,
                earliest_session=earliest_session,
                source_receipt_hash=receipt.receipt_hash,
                observed_at=observed_at,
            )
            connection.execute(
                """
                INSERT INTO action_audit_receipt (
                    receipt_hash, listing_id, provider, manifest_revision, mapping_revision,
                    requested_as_of, history_start, history_end, raw_evidence_hash, action_set_hash,
                    action_evidence_hash, provider_adjusted_close_evidence_hash,
                    max_adjusted_close_difference_bps, adjusted_close_mismatch_count,
                    first_adjusted_close_mismatch_session, diagnostic_policy_hash, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (receipt_hash) DO NOTHING
                """,
                [
                    receipt.receipt_hash,
                    receipt.listing_id,
                    receipt.provider,
                    receipt.manifest_revision,
                    receipt.mapping_revision,
                    receipt.requested_as_of,
                    receipt.history_start,
                    receipt.history_end,
                    receipt.raw_evidence_hash,
                    receipt.action_set_hash,
                    receipt.action_evidence_hash,
                    receipt.provider_adjusted_close_evidence_hash,
                    receipt.max_adjusted_close_difference_bps,
                    receipt.adjusted_close_mismatch_count,
                    receipt.first_adjusted_close_mismatch_session,
                    receipt.diagnostic_policy_hash,
                    receipt.observed_at,
                ],
            )
            if _connection is None:
                connection.execute("COMMIT")
            return receipt, {
                "inserted": inserted,
                "corrected": corrected,
                "retracted": retracted,
            }
        except Exception:
            if _connection is None:
                connection.execute("ROLLBACK")
            raise
        finally:
            if _connection is None:
                connection.close()

    def reusable_action_audit_receipt(
        self,
        manifest: UniverseManifest,
        *,
        listing_id: str,
        provider: str,
        requested_as_of: date,
        now: datetime,
        allow_equivalent_manifest: bool = False,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> ActionAuditReceipt | None:
        """Return a still-valid full-history receipt; TTL is a budget, never proof of truth.

        This answers the acquisition-budget question -- may a recent audit
        suppress a new full-history provider audit? -- so only a receipt that
        covers the listing's whole current history as of ``requested_as_of``
        can answer it, and it must still prove the current bytes over that whole
        range (``_action_audit_receipt_verifies``).
        """
        now = _utc_naive(now)
        self._assert_manifest_scope(manifest, [listing_id])
        connection = _connection or self._connect(read_only=True)
        owns_connection = _connection is None
        try:
            earliest = connection.execute(
                """
                SELECT min(session_date) FROM raw_daily_bar_current
                WHERE listing_id = ? AND provider = ? AND session_date <= ?
                """,
                [listing_id, provider, requested_as_of],
            ).fetchone()[0]
            if earliest is None:
                return None
            candidates = self._action_audit_candidates(
                connection,
                listing_id=listing_id,
                provider=provider,
                requested_as_of=requested_as_of,
                manifest_revision=None if allow_equivalent_manifest else manifest.revision_sha256,
            )
            if not candidates:
                return None
            evidence = self._action_audit_evidence(
                connection,
                listing_id=listing_id,
                provider=provider,
                requested_as_of=requested_as_of,
                history_start=earliest,
                history_end=requested_as_of,
            )
            # Equal observation times do not order revisions. A hash tie-break
            # may put superseded evidence first, so test candidates against the
            # current bytes rather than treating that arbitrary first row as
            # the only possible proof. Every original reuse check still holds.
            for receipt in candidates:
                if receipt.history_start > earliest or receipt.history_end < requested_as_of:
                    continue
                if self._action_audit_receipt_verifies(receipt, evidence, now=now):
                    return receipt
            return None
        finally:
            if owns_connection:
                connection.close()

    def verified_action_audit_receipt(
        self,
        manifest: UniverseManifest,
        *,
        receipt_hash: str,
        listing_id: str,
        provider: str,
        requested_as_of: date,
        now: datetime,
        allow_historical: bool = False,
        allow_equivalent_manifest: bool = False,
    ) -> ActionAuditReceipt | None:
        """Verify an exact consumed receipt over its recorded range, not a new audit budget.

        Historical verification retains the recorded observation and scope, ignoring
        only acquisition freshness and an equivalent membership manifest revision.
        Equivalent manifests can also be admitted without relaxing freshness.
        """
        self._assert_manifest_scope(manifest, [listing_id])
        with self._connect(read_only=True) as connection:
            receipt = self._rebindable_action_audit_receipt(
                connection,
                listing_id=listing_id,
                provider=provider,
                requested_as_of=requested_as_of,
                now=now,
                manifest_revision=(
                    None
                    if allow_historical or allow_equivalent_manifest
                    else manifest.revision_sha256
                ),
                receipt_hash=receipt_hash,
                allow_historical=allow_historical,
            )
        if receipt is None or receipt.observed_at > _utc_naive(now):
            return None
        payload = {key: value for key, value in receipt.__dict__.items() if key != "receipt_hash"}
        return receipt if _canonical_hash(payload) == receipt_hash else None

    # -- evidence reuse validation: one owner for every reuse question ------

    def _action_audit_candidates(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        listing_id: str,
        provider: str,
        requested_as_of: date,
        manifest_revision: str | None,
    ) -> tuple[ActionAuditReceipt, ...]:
        """Receipts for one listing and session, newest first; one manifest's or any."""
        manifest_clause = "" if manifest_revision is None else "AND receipt.manifest_revision = ?"
        parameters: list[object] = [listing_id, provider, requested_as_of]
        if manifest_revision is not None:
            parameters.append(manifest_revision)
        rows = connection.execute(
            f"""
            SELECT receipt.receipt_hash, receipt.listing_id, receipt.provider,
                   receipt.manifest_revision, receipt.mapping_revision,
                   receipt.requested_as_of, receipt.history_start, receipt.history_end,
                   receipt.raw_evidence_hash, receipt.action_set_hash,
                   receipt.action_evidence_hash,
                   receipt.provider_adjusted_close_evidence_hash,
                   receipt.max_adjusted_close_difference_bps,
                   receipt.adjusted_close_mismatch_count,
                   receipt.first_adjusted_close_mismatch_session,
                   receipt.diagnostic_policy_hash, receipt.observed_at
            FROM action_audit_receipt AS receipt
            WHERE receipt.listing_id = ? AND receipt.provider = ?
              AND receipt.requested_as_of = ?
              {manifest_clause}
            ORDER BY receipt.observed_at DESC, receipt.receipt_hash DESC
            """,
            parameters,
        ).fetchall()
        return tuple(ActionAuditReceipt(*row) for row in rows)

    def _action_audit_evidence(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        listing_id: str,
        provider: str,
        requested_as_of: date,
        history_start: date,
        history_end: date,
    ) -> dict[str, object]:
        """Read the listing's current bytes over one audited range.

        The mapping revision as of the requested session, the raw evidence
        digest and the provider adjusted-close evidence digest over exactly
        ``history_start..history_end`` (``None`` when the adjusted series does
        not cover that range's sessions exactly), and the current action set
        and action evidence, which every receipt digests without a range.
        """
        raw_sessions = {
            item[0]
            for item in connection.execute(
                """
                SELECT session_date FROM raw_daily_bar_current
                WHERE listing_id = ? AND provider = ? AND session_date BETWEEN ? AND ?
                """,
                [listing_id, provider, history_start, history_end],
            ).fetchall()
        }
        adjusted_rows = connection.execute(
            """
            SELECT session_date, adjusted_close FROM provider_adjusted_close_current
            WHERE listing_id = ? AND provider = ? AND session_date BETWEEN ? AND ?
            ORDER BY session_date
            """,
            [listing_id, provider, history_start, history_end],
        ).fetchall()
        adjusted_points = tuple(
            ProviderAdjustedClosePoint(listing_id, provider, session, float(adjusted_close))
            for session, adjusted_close in adjusted_rows
        )
        adjusted_digest = (
            provider_adjusted_close_evidence_hash(adjusted_points)
            if {item.session_date for item in adjusted_points} == raw_sessions
            else None
        )
        return {
            "mapping_revision": self._mapping_revision(
                connection,
                listing_id=listing_id,
                provider=provider,
                as_of_session=requested_as_of,
            ),
            "raw_evidence_hash": self._raw_evidence_hash(
                connection,
                listing_id=listing_id,
                provider=provider,
                history_start=history_start,
                history_end=history_end,
            ),
            "action_set_hash": action_set_hash(
                self._current_actions(connection, listing_id, provider)
            ),
            "action_evidence_hash": self._action_evidence_hash(
                connection, listing_id=listing_id, provider=provider
            ),
            "provider_adjusted_close_evidence_hash": adjusted_digest,
        }

    @staticmethod
    def _action_audit_receipt_current(
        receipt: ActionAuditReceipt, *, now: datetime, allow_historical: bool = False
    ) -> bool:
        """Check freshness, complete diagnostics, and the installed policy.

        A receipt these refuse is refused by the complete validator whatever
        the current bytes say, so a caller may skip the byte evidence for it. The
        rules live here once; ``_action_audit_receipt_verifies`` applies them
        before the byte comparison.
        Historical reads admit older observations, never future ones.
        """
        return not (
            (not allow_historical and _utc_naive(now) - receipt.observed_at >= timedelta(hours=24))
            or (allow_historical and receipt.observed_at > _utc_naive(now))
            or receipt.provider_adjusted_close_evidence_hash is None
            or receipt.max_adjusted_close_difference_bps is None
            or receipt.adjusted_close_mismatch_count is None
            or receipt.diagnostic_policy_hash != ADJUSTED_CLOSE_DIAGNOSTIC_POLICY_HASH
        )

    @staticmethod
    def _action_audit_receipt_verifies(
        receipt: ActionAuditReceipt,
        evidence: Mapping[str, object],
        *,
        now: datetime,
        allow_historical: bool = False,
    ) -> bool:
        """Every reuse check: fresh, same mapping, same bytes, complete diagnostics.

        ``evidence`` is the current ``_action_audit_evidence`` over the range the
        caller holds the receipt to -- the whole history for the acquisition
        budget, the receipt's own recorded range for rebinding.
        Historical verification changes only the acquisition freshness check.
        """
        return MarketDataRepository._action_audit_receipt_current(
            receipt, now=now, allow_historical=allow_historical
        ) and not (
            receipt.mapping_revision != evidence["mapping_revision"]
            or receipt.raw_evidence_hash != evidence["raw_evidence_hash"]
            or receipt.action_set_hash != evidence["action_set_hash"]
            or receipt.action_evidence_hash != evidence["action_evidence_hash"]
            or evidence["provider_adjusted_close_evidence_hash"] is None
            or receipt.provider_adjusted_close_evidence_hash
            != evidence["provider_adjusted_close_evidence_hash"]
        )

    def _rebindable_action_audit_receipt(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        listing_id: str,
        provider: str,
        requested_as_of: date,
        now: datetime,
        manifest_revision: str | None,
        receipt_hash: str | None = None,
        candidate_hashes: frozenset[str] | None = None,
        allow_historical: bool = False,
    ) -> ActionAuditReceipt | None:
        """Find another manifest's receipt that still proves its own range.

        Membership revision is not market-data content, so a receipt can be
        carried to a derived manifest -- but only the receipt as recorded: a
        daily maintenance audits a rolling window and a full history only
        when one is due, and the child inherits exactly the scope its
        ancestor audited, re-verified against current bytes over that scope
        with the same checks as the acquisition-budget question. Nothing is
        re-observed and no observation time moves. ``candidate_hashes``
        narrows the receipts judged to the ones a caller has already found
        new; the order and the checks are the same.
        """
        return next(
            self._rebindable_action_audit_receipts(
                connection,
                listing_id=listing_id,
                provider=provider,
                requested_as_of=requested_as_of,
                now=now,
                manifest_revision=manifest_revision,
                receipt_hash=receipt_hash,
                candidate_hashes=candidate_hashes,
                allow_historical=allow_historical,
            ),
            None,
        )

    def _rebindable_action_audit_receipts(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        listing_id: str,
        provider: str,
        requested_as_of: date,
        now: datetime,
        manifest_revision: str | None,
        receipt_hash: str | None = None,
        candidate_hashes: frozenset[str] | None = None,
        allow_historical: bool = False,
    ) -> Iterator[ActionAuditReceipt]:
        """Every candidate receipt that still proves its own range, in the candidates' order.

        Each receipt is judged on its own recorded range: a full-history
        receipt and a rolling one may both prove current bytes, and neither
        stands in for the other; nor does a later receipt over the same
        range stand in for an earlier one, since the bytes may have moved
        and returned between them. Only the validator decides. The byte-free
        checks go first, so a receipt they refuse costs no evidence digest,
        and current evidence over one range is read once for every receipt
        that shares it, so a call costs one digest per range judged.
        """
        candidates = self._action_audit_candidates(
            connection,
            listing_id=listing_id,
            provider=provider,
            requested_as_of=requested_as_of,
            manifest_revision=manifest_revision,
        )
        evidence_by_range: dict[tuple[date, date], dict[str, object]] = {}
        for receipt in candidates:
            if receipt_hash is not None and receipt.receipt_hash != receipt_hash:
                continue
            if candidate_hashes is not None and receipt.receipt_hash not in candidate_hashes:
                continue
            if receipt.history_end < requested_as_of:
                continue
            if not self._action_audit_receipt_current(
                receipt, now=now, allow_historical=allow_historical
            ):
                continue
            audited = (receipt.history_start, receipt.history_end)
            evidence = evidence_by_range.get(audited)
            if evidence is None:
                evidence = self._action_audit_evidence(
                    connection,
                    listing_id=listing_id,
                    provider=provider,
                    requested_as_of=requested_as_of,
                    history_start=receipt.history_start,
                    history_end=receipt.history_end,
                )
                evidence_by_range[audited] = evidence
            if self._action_audit_receipt_verifies(
                receipt, evidence, now=now, allow_historical=allow_historical
            ):
                yield receipt

    @staticmethod
    def _rebound_receipt(receipt: ActionAuditReceipt, manifest_revision: str) -> ActionAuditReceipt:
        payload = {key: value for key, value in receipt.__dict__.items() if key != "receipt_hash"}
        payload["manifest_revision"] = manifest_revision
        return ActionAuditReceipt(_canonical_hash(payload), **payload)

    def bind_available_action_audit_receipts_to_manifest(
        self,
        target_manifest: UniverseManifest,
        *,
        requested_as_of: date,
        now: datetime,
    ) -> tuple[str, ...]:
        """Bind currently equivalent listing evidence to a derived manifest.

        Membership revision is not market-data content.  A recently validated
        receipt from an ancestor manifest may therefore be rebound only after
        rechecking the current mapping, raw rows, actions, adjusted series,
        diagnostic policy, coverage, and TTL -- over the range that receipt
        audited (``_rebindable_action_audit_receipt``).  Missing or drifted
        listings are omitted so the caller can fetch only that remainder.
        Answers the listings bound by this call; a listing whose copies
        already cover every other manifest's evidence for the session is
        not re-judged, so repeated calls cost the candidate query alone.
        """
        if not target_manifest.listings:
            return ()
        now = _utc_naive(now)
        prepared: list[ActionAuditReceipt] = []
        connection = self._connect()
        try:
            # Only another manifest's receipt can produce a new binding, and a
            # binding copies that receipt as recorded, so a listing is a
            # candidate exactly when another manifest holds same-session
            # evidence this manifest holds no copy of: content, not presence.
            # Nothing about instants prunes here. A receipt sealed earlier may
            # prove today's bytes while a later one does not -- over another
            # range, and over the same range too: a full audit that sealed a
            # tail that has since been restored by a rolling audit leaves the
            # older full receipt valid and the later one refused, and the
            # restoring receipt covers only the window. Whether a receipt
            # still proves its range is the validator's answer alone. A
            # listing whose copies cover everything the other manifests hold
            # is skipped without re-validation (the coordinator calls this
            # once per cycle); a copy bound earlier that no longer proves the
            # bytes does not hide evidence the parent audited since; an
            # unheld receipt the validator refuses is judged again next call
            # until its freshness ends, one digest per range. This query
            # admits no evidence: every candidate below still goes through
            # the complete reuse validator, whose byte-free refusals
            # (freshness, completeness, policy) are the only pruning before
            # the bytes are read, so nothing that could verify today is left
            # unjudged.
            candidates: dict[str, set[str]] = {}
            for row in connection.execute(
                """
                    SELECT other.listing_id, other.receipt_hash FROM action_audit_receipt AS other
                    WHERE other.provider = ? AND other.requested_as_of = ?
                      AND other.manifest_revision != ?
                      AND NOT EXISTS (
                          SELECT 1 FROM action_audit_receipt AS bound
                          WHERE bound.listing_id = other.listing_id
                            AND bound.provider = other.provider
                            AND bound.requested_as_of = other.requested_as_of
                            AND bound.manifest_revision = ?
                            AND bound.mapping_revision IS NOT DISTINCT FROM other.mapping_revision
                            AND bound.history_start = other.history_start
                            AND bound.history_end = other.history_end
                            AND bound.raw_evidence_hash = other.raw_evidence_hash
                            AND bound.action_set_hash = other.action_set_hash
                            AND bound.action_evidence_hash = other.action_evidence_hash
                            AND bound.provider_adjusted_close_evidence_hash
                                IS NOT DISTINCT FROM other.provider_adjusted_close_evidence_hash
                            AND bound.max_adjusted_close_difference_bps
                                IS NOT DISTINCT FROM other.max_adjusted_close_difference_bps
                            AND bound.adjusted_close_mismatch_count
                                IS NOT DISTINCT FROM other.adjusted_close_mismatch_count
                            AND bound.first_adjusted_close_mismatch_session
                                IS NOT DISTINCT FROM other.first_adjusted_close_mismatch_session
                            AND bound.diagnostic_policy_hash
                                IS NOT DISTINCT FROM other.diagnostic_policy_hash
                            AND bound.observed_at = other.observed_at
                      )
                    """,
                [
                    target_manifest.profile.provider,
                    requested_as_of,
                    target_manifest.revision_sha256,
                    target_manifest.revision_sha256,
                ],
            ).fetchall():
                candidates.setdefault(str(row[0]), set()).add(str(row[1]))
            for listing in target_manifest.listings:
                new_evidence = candidates.get(listing.listing_id)
                if not new_evidence:
                    continue
                # Only the evidence found new is judged (a receipt whose copy
                # the manifest already holds would be chosen again by its
                # order and bind nothing), and every receipt of it that
                # proves its own range is bound: the manifest inherits the
                # full-history and the rolling evidence alike, so its
                # acquisition-budget question is answered by whichever the
                # ancestor holds.
                prepared.extend(
                    self._rebound_receipt(receipt, target_manifest.revision_sha256)
                    for receipt in self._rebindable_action_audit_receipts(
                        connection,
                        listing_id=listing.listing_id,
                        provider=target_manifest.profile.provider,
                        requested_as_of=requested_as_of,
                        now=now,
                        manifest_revision=None,
                        candidate_hashes=frozenset(new_evidence),
                    )
                )
            if prepared:
                self._insert_rebound_receipts(connection, prepared)
        finally:
            connection.close()
        return tuple(sorted({receipt.listing_id for receipt in prepared}))

    @staticmethod
    def _insert_rebound_receipts(
        connection: duckdb.DuckDBPyConnection, prepared: Sequence[ActionAuditReceipt]
    ) -> None:
        """One transaction for every rebound receipt; a refusal leaves none behind.

        Offered as one relation (`_staged_rows`) rather than a statement per receipt, a
        derived manifest's hundreds of receipts each paid the key probe of the conflict
        clause. Of receipts with one hash the first is offered, as the statement per
        receipt kept the first and skipped the rest.
        """
        first: dict[str, ActionAuditReceipt] = {}
        for receipt in prepared:
            first.setdefault(receipt.receipt_hash, receipt)
        receipts = tuple(first.values())
        connection.execute("BEGIN TRANSACTION")
        try:
            with _staged_rows(
                connection,
                "rebound_action_audit_receipt_stage",
                {
                    "receipt_hash": pa.array([r.receipt_hash for r in receipts], pa.string()),
                    "listing_id": pa.array([r.listing_id for r in receipts], pa.string()),
                    "provider": pa.array([r.provider for r in receipts], pa.string()),
                    "manifest_revision": pa.array(
                        [r.manifest_revision for r in receipts], pa.string()
                    ),
                    "mapping_revision": pa.array(
                        [r.mapping_revision for r in receipts], pa.string()
                    ),
                    "requested_as_of": pa.array([r.requested_as_of for r in receipts], pa.date32()),
                    "history_start": pa.array([r.history_start for r in receipts], pa.date32()),
                    "history_end": pa.array([r.history_end for r in receipts], pa.date32()),
                    "raw_evidence_hash": pa.array(
                        [r.raw_evidence_hash for r in receipts], pa.string()
                    ),
                    "action_set_hash": pa.array([r.action_set_hash for r in receipts], pa.string()),
                    "action_evidence_hash": pa.array(
                        [r.action_evidence_hash for r in receipts], pa.string()
                    ),
                    "provider_adjusted_close_evidence_hash": pa.array(
                        [r.provider_adjusted_close_evidence_hash for r in receipts], pa.string()
                    ),
                    "max_adjusted_close_difference_bps": pa.array(
                        [r.max_adjusted_close_difference_bps for r in receipts], pa.float64()
                    ),
                    "adjusted_close_mismatch_count": pa.array(
                        [r.adjusted_close_mismatch_count for r in receipts], pa.int32()
                    ),
                    "first_adjusted_close_mismatch_session": pa.array(
                        [r.first_adjusted_close_mismatch_session for r in receipts], pa.date32()
                    ),
                    "diagnostic_policy_hash": pa.array(
                        [r.diagnostic_policy_hash for r in receipts], pa.string()
                    ),
                    "observed_at": pa.array([r.observed_at for r in receipts], pa.timestamp("us")),
                },
            ) as stage:
                connection.execute(
                    f"""
                    INSERT INTO action_audit_receipt (
                        receipt_hash, listing_id, provider, manifest_revision, mapping_revision,
                        requested_as_of, history_start, history_end, raw_evidence_hash,
                        action_set_hash, action_evidence_hash,
                        provider_adjusted_close_evidence_hash,
                        max_adjusted_close_difference_bps, adjusted_close_mismatch_count,
                        first_adjusted_close_mismatch_session, diagnostic_policy_hash,
                        observed_at
                    )
                    SELECT receipt_hash, listing_id, provider, manifest_revision,
                        mapping_revision, requested_as_of, history_start, history_end,
                        raw_evidence_hash, action_set_hash, action_evidence_hash,
                        provider_adjusted_close_evidence_hash,
                        max_adjusted_close_difference_bps, adjusted_close_mismatch_count,
                        first_adjusted_close_mismatch_session, diagnostic_policy_hash,
                        observed_at
                    FROM {stage}
                    ON CONFLICT (receipt_hash) DO NOTHING
                    """
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def latest_action_audit_receipt(
        self, *, listing_id: str, provider: str
    ) -> ActionAuditReceipt | None:
        """Return the newest full provider audit for coordinator chain seeding."""
        return self.latest_action_audit_receipts((listing_id,), provider=provider).get(listing_id)

    def latest_action_audit_receipts(
        self,
        listing_ids: tuple[str, ...],
        *,
        provider: str,
        requested_as_of: date | None = None,
    ) -> dict[str, ActionAuditReceipt]:
        """Return the newest provider audits in one authoritative scan.

        Without ``requested_as_of``: the newest full-history audit per listing
        (the coordinator's chain seed). With it: the newest audit sealed for
        that session in whatever scope it was run -- the rolling window of a
        daily maintenance or a full history -- so governance that runs after
        the maintenance reads the same receipt the maintenance itself
        reported, not a differently scoped older one.
        """
        ordered_ids = tuple(sorted(set(listing_ids)))
        if not ordered_ids:
            return {}
        placeholders = ",".join("?" for _ in ordered_ids)
        if requested_as_of is None:
            scope_clause = "AND receipt.history_start <= earliest.earliest_session"
            scope_parameters: list[object] = []
        else:
            scope_clause = "AND receipt.requested_as_of = ?"
            scope_parameters = [requested_as_of]
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                f"""
                WITH earliest AS (
                    SELECT listing_id, min(session_date) AS earliest_session
                    FROM raw_daily_bar_current
                    WHERE provider = ? AND listing_id IN ({placeholders})
                    GROUP BY listing_id
                )
                SELECT receipt_hash, listing_id, provider, manifest_revision, mapping_revision,
                       requested_as_of, history_start, history_end, raw_evidence_hash,
                       action_set_hash, action_evidence_hash,
                       provider_adjusted_close_evidence_hash,
                       max_adjusted_close_difference_bps, adjusted_close_mismatch_count,
                       first_adjusted_close_mismatch_session, diagnostic_policy_hash, observed_at
                FROM action_audit_receipt AS receipt
                JOIN earliest USING (listing_id)
                WHERE receipt.provider = ?
                  AND receipt.listing_id IN ({placeholders})
                  {scope_clause}
                QUALIFY row_number() OVER (
                    PARTITION BY receipt.listing_id
                    ORDER BY receipt.observed_at DESC, receipt.requested_as_of DESC,
                             receipt.receipt_hash DESC
                ) = 1
                """,
                [provider, *ordered_ids, provider, *ordered_ids, *scope_parameters],
            ).fetchall()
        finally:
            connection.close()
        return {str(row[1]): ActionAuditReceipt(*row) for row in rows}

    def full_action_audit_receipt_hashes(self, receipt_hashes: tuple[str, ...]) -> frozenset[str]:
        """Validate that provider receipts cover the complete current raw range."""
        ordered_hashes = tuple(sorted(set(receipt_hashes)))
        if not ordered_hashes:
            return frozenset()
        placeholders = ",".join("?" for _ in ordered_hashes)
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                f"""
                WITH earliest AS (
                    SELECT listing_id, provider, min(session_date) AS earliest_session
                    FROM raw_daily_bar_current
                    GROUP BY listing_id, provider
                )
                SELECT receipt.receipt_hash
                FROM action_audit_receipt AS receipt
                JOIN earliest USING (listing_id, provider)
                WHERE receipt.receipt_hash IN ({placeholders})
                  AND receipt.history_start <= earliest.earliest_session
                """,
                list(ordered_hashes),
            ).fetchall()
        finally:
            connection.close()
        return frozenset(str(row[0]) for row in rows)

    def bind_action_audit_receipts_to_manifest(
        self,
        source_manifest: UniverseManifest,
        target_manifest: UniverseManifest,
        *,
        requested_as_of: date,
        now: datetime,
    ) -> None:
        """Bind unchanged audit evidence to a derived manifest revision.

        This does not refresh the receipt TTL and does not re-observe provider
        facts.  It only gives a quality-filtered child manifest an exact
        revision-bound receipt for evidence already accepted by its parent:
        the parent's receipt for this session, in the scope the parent
        audited (a daily rolling window or a full history), after that scope
        still verifies against current bytes. A listing whose parent receipt
        is missing, expired or no longer matches refuses the whole binding.
        """
        source_ids = {item.listing_id for item in source_manifest.listings}
        target_ids = {item.listing_id for item in target_manifest.listings}
        if not target_ids.issubset(source_ids):
            raise ValueError("target manifest is not a subset of source audit scope")
        now = _utc_naive(now)
        prepared: list[ActionAuditReceipt] = []
        connection = self._connect()
        try:
            for listing in target_manifest.listings:
                receipt = self._rebindable_action_audit_receipt(
                    connection,
                    listing_id=listing.listing_id,
                    provider=target_manifest.profile.provider,
                    requested_as_of=requested_as_of,
                    now=now,
                    manifest_revision=source_manifest.revision_sha256,
                )
                if receipt is None:
                    raise ValueError(
                        "source manifest has no reusable action-audit receipt: "
                        f"{listing.listing_id} as of {requested_as_of.isoformat()}"
                    )
                prepared.append(self._rebound_receipt(receipt, target_manifest.revision_sha256))
            self._insert_rebound_receipts(connection, prepared)
        finally:
            connection.close()

    def record_failures(
        self,
        failures: Iterable[FailureEvidence],
        *,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Persist listing and marketwide failure evidence in one transaction.

        ``_connection`` is the caller's open transaction, which the caller commits.
        """
        connection = _connection or self._connect()
        try:
            if _connection is None:
                connection.execute("BEGIN TRANSACTION")
            for evidence in failures:
                observed_at = _utc_naive(evidence.observed_at)
                attempt_id = _canonical_hash(
                    [
                        evidence.listing_id,
                        evidence.failure_code,
                        evidence.range_start,
                        evidence.range_end,
                        observed_at,
                    ]
                )
                connection.execute(
                    """
                    INSERT INTO provider_attempt VALUES (?, ?, ?, ?, ?, 'FAILURE', ?)
                    ON CONFLICT DO NOTHING
                    """,
                    [
                        attempt_id,
                        evidence.listing_id,
                        evidence.failure_code,
                        evidence.range_start,
                        evidence.range_end,
                        observed_at,
                    ],
                )
                connection.execute(
                    """
                    UPDATE data_quality SET quality_state = 'LISTING_REQUIRES_REVIEW',
                    summary = ?, updated_at = ? WHERE listing_id = ?
                    """,
                    [evidence.failure_code, observed_at, evidence.listing_id],
                )
            if _connection is None:
                connection.execute("COMMIT")
        except Exception:
            if _connection is None:
                connection.execute("ROLLBACK")
            raise
        finally:
            if _connection is None:
                connection.close()

    def health_report(
        self, manifest: UniverseManifest, *, run_id: str, case_token: str, as_of_session: date
    ) -> DataHealthReport:
        """Build a typed health report for the manifest's listing scope.

        Args:
            manifest: Frozen listings and profile to report.
            run_id: Identifier of the requesting maintenance run.
            case_token: Related maintenance case identifier.
            as_of_session: Session whose readiness is reported.

        Returns:
            Health findings for the manifest's listings.

        """
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT l.listing_id, l.display_symbol, q.latest_session, q.quality_state,
                    (SELECT CASE WHEN p.outcome = 'FAILURE' THEN p.failure_code ELSE NULL END
                     FROM provider_attempt p WHERE p.listing_id = l.listing_id
                     ORDER BY p.observed_at DESC LIMIT 1) AS provider_failure
                FROM listing l JOIN data_quality q USING (listing_id)
                WHERE l.market_profile_id = ? ORDER BY l.display_symbol
                """,
                [manifest.profile.market_profile_id],
            ).fetchall()
        finally:
            connection.close()
        return DataHealthReport(
            run_id=run_id,
            case_token=case_token,
            market_profile_id=manifest.profile.market_profile_id,
            manifest_revision=manifest.revision_sha256,
            as_of_session=as_of_session,
            subjects=tuple(
                HealthSubject(
                    listing_id=str(row[0]),
                    symbol=str(row[1]),
                    latest_session=row[2],
                    expected_session=as_of_session,
                    quality_state=str(row[3]),
                    provider_failure=str(row[4]) if row[4] is not None else None,
                )
                for row in rows
            ),
        )

    def raw_bars(
        self,
        listing_id: str,
        *,
        start: date | None = None,
        through: date | None = None,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> tuple[RawDailyBar, ...]:
        """Read raw daily bars for one listing and optional session bounds.

        Args:
            listing_id: Durable listing identity.
            start: First included session, if bounded.
            through: Last included session, if bounded.
            _connection: Existing read connection for a larger transaction.

        Returns:
            Bars in session order.

        """
        connection = _connection or self._connect(read_only=True)
        owns_connection = _connection is None
        try:
            rows = connection.execute(
                *_raw_bar_query(listing_id, start=start, through=through)
            ).fetchall()
        finally:
            if owns_connection:
                connection.close()
        return tuple(RawDailyBar(*row) for row in rows)

    def raw_bars_by_listing(
        self,
        listing_ids: Sequence[str],
        *,
        start: date | None = None,
        through: date | None = None,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> dict[str, tuple[RawDailyBar, ...]]:
        """``raw_bars`` of many listings in one read: each one's rows, filter and order.

        Args:
            listing_ids: Durable listing identities; each answers, empty without bars.
            start: First included session, if bounded.
            through: Last included session, if bounded.
            _connection: Existing read connection for a larger transaction.

        Returns:
            Each listing's bars in session order.
        """
        sql = (
            f"SELECT {', '.join(_RAW_BAR_COLUMNS)} FROM raw_daily_bar_current "
            "WHERE listing_id IN (SELECT unnest(?::VARCHAR[]))"
        )
        params: list[object] = [list(listing_ids)]
        if start is not None:
            sql += " AND session_date >= ?"
            params.append(start)
        if through is not None:
            sql += " AND session_date <= ?"
            params.append(through)
        connection = _connection or self._connect(read_only=True)
        owns_connection = _connection is None
        try:
            rows = connection.execute(sql + " ORDER BY listing_id, session_date", params).fetchall()
        finally:
            if owns_connection:
                connection.close()
        bars: dict[str, list[RawDailyBar]] = {listing_id: [] for listing_id in listing_ids}
        for row in rows:
            bars[row[0]].append(RawDailyBar(*row))
        return {listing_id: tuple(values) for listing_id, values in bars.items()}

    @staticmethod
    def _raw_bar_table(
        connection: duckdb.DuckDBPyConnection,
        listing_id: str,
        *,
        start: date | None,
        through: date | None,
    ) -> pa.Table:
        """The rows ``raw_bars`` reads, as columns: a reader that hands a listing's bars to
        another process builds no object per bar (``raw_bars_from_table`` builds them there)."""
        table: pa.Table = connection.execute(
            *_raw_bar_query(listing_id, start=start, through=through)
        ).to_arrow_table()
        return table

    @staticmethod
    def _raw_bar_tables(
        connection: duckdb.DuckDBPyConnection, starts: Mapping[str, date], *, through: date
    ) -> dict[str, pa.Table]:
        """``_raw_bar_table`` of many listings in one read, each from its own start."""
        if not starts:
            return {}
        table: pa.Table = connection.execute(
            f"SELECT {', '.join(_RAW_BAR_COLUMNS)} FROM raw_daily_bar_current "
            "WHERE listing_id IN (SELECT unnest(?::VARCHAR[])) AND session_date BETWEEN ? AND ? "
            "ORDER BY listing_id, session_date",
            [sorted(starts), min(starts.values()), through],
        ).to_arrow_table()
        listings = table.column("listing_id").to_pylist()
        sessions = table.column("session_date").to_pylist()
        bounds: dict[str, tuple[int, int]] = {}
        for index, listing in enumerate(listings):
            bounds[listing] = (bounds.get(listing, (index, index))[0], index + 1)
        out: dict[str, pa.Table] = {}
        for listing, start in starts.items():
            first, last = bounds.get(listing, (0, 0))
            begin = bisect_left(sessions, start, first, last)
            out[listing] = table.slice(begin, last - begin)
        return out

    def raw_bar_sessions_by_listing(
        self,
        listing_ids: Sequence[str],
        *,
        through: date | None = None,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> dict[str, tuple[date, ...]]:
        """``raw_bar_sessions`` of many listings in one read: each one's session axis.

        Args:
            listing_ids: Durable listing identities; each answers, empty without bars.
            through: Last included session, if bounded.
            _connection: Existing read connection for a larger transaction.

        Returns:
            Each listing's ordered sessions.
        """
        connection = _connection or self._connect(read_only=True)
        owns_connection = _connection is None
        try:
            sql = (
                "SELECT listing_id, session_date FROM raw_daily_bar_current "
                "WHERE listing_id IN (SELECT unnest(?::VARCHAR[]))"
            )
            params: list[object] = [list(listing_ids)]
            if through is not None:
                sql += " AND session_date <= ?"
                params.append(through)
            rows = connection.execute(sql + " ORDER BY listing_id, session_date", params).fetchall()
        finally:
            if owns_connection:
                connection.close()
        out: dict[str, list[date]] = {listing: [] for listing in listing_ids}
        for listing, session in rows:
            out[listing].append(session)
        return {listing: tuple(sessions) for listing, sessions in out.items()}

    def raw_bar_sessions(
        self,
        listing_id: str,
        *,
        through: date | None = None,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> tuple[date, ...]:
        """Return the ordered session axis of ``raw_bars`` without its bars.

        The Feature build needs every listing's full calendar to plan its
        targets but none of the bar values for that; this is the same rows,
        filter and order projected to their dates.
        """
        return self.raw_bar_sessions_by_listing(
            (listing_id,), through=through, _connection=_connection
        )[listing_id]

    def raw_close_volume_observations(
        self,
        listing_ids: tuple[str, ...],
        *,
        start: date,
        through: date,
    ) -> pa.Table:
        """Read one exact listing set's bounded raw close/volume rows in one scan."""
        if (
            not listing_ids
            or len(set(listing_ids)) != len(listing_ids)
            or any(not value for value in listing_ids)
            or start > through
        ):
            raise ValueError("market_data_ops.raw_close_volume_request_invalid")
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT listing_id, session_date, close, volume
                FROM raw_daily_bar_current
                WHERE listing_id IN (SELECT unnest(?))
                  AND session_date >= ?
                  AND session_date <= ?
                ORDER BY session_date, listing_id
                """,
                [tuple(sorted(listing_ids)), start, through],
            ).to_arrow_table()
        finally:
            connection.close()
        return rows

    def market_data_source_prefix_proof(
        self,
        *,
        listing_ids: Sequence[str],
        start: date,
        through: date,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> str:
        """Prove exact current raw values and cutoff-admitted actions in one bounded scan.

        Raw rows have the same listing/date selection as ``raw_bars``, including
        every stored provider. Actions use ``actions``' latest provider mapping,
        ACTIVE status and effective dates through the cutoff, with no lower
        action-date bound. Stored payload hashes and revision journals are not
        evidence that these current values still equal a previous prefix.

        Args:
            listing_ids: Nonempty unique listing scope, normalized to sorted order.
            start: First included raw session.
            through: Last included raw session and admitted action effective date.
            _connection: Optional connection already held in a consistent read snapshot.

        Returns:
            SHA256 binding request scope, actual keys, values, nulls and selected providers.

        Raises:
            ValueError: The request is invalid or a listing has no provider mapping.
        """
        scope = tuple(listing_ids)
        if (
            not scope
            or any(not isinstance(value, str) or not value for value in scope)
            or len(set(scope)) != len(scope)
            or start > through
        ):
            raise ValueError("market_data_ops.source_prefix_request_invalid")
        scope = tuple(sorted(scope))
        providers_sql = _SELECTED_PROVIDER_SQL
        boundary = (
            self.database.read_transaction() if _connection is None else nullcontext(_connection)
        )
        with boundary as connection:
            providers = connection.execute(
                providers_sql
                + "SELECT listing_id, provider FROM selected_provider ORDER BY listing_id",
                [scope],
            ).to_arrow_table()
            if providers.column("provider").null_count:
                raise ValueError("listing has no provider mapping")
            raw = connection.execute(
                """
                SELECT listing_id, provider, session_date,
                       COALESCE(open, 0::DOUBLE) AS open,
                       COALESCE(high, 0::DOUBLE) AS high,
                       COALESCE(low, 0::DOUBLE) AS low,
                       COALESCE(close, 0::DOUBLE) AS close,
                       COALESCE(volume, 0::BIGINT) AS volume,
                       open IS NULL AS open_is_null, high IS NULL AS high_is_null,
                       low IS NULL AS low_is_null, close IS NULL AS close_is_null,
                       volume IS NULL AS volume_is_null
                FROM raw_daily_bar_current
                WHERE listing_id IN (SELECT unnest(?))
                  AND session_date BETWEEN ? AND ?
                ORDER BY listing_id, session_date, provider
                """,
                [scope, start, through],
            ).to_arrow_table()
            actions = connection.execute(
                providers_sql
                + """
                SELECT action.listing_id, action.provider, action.effective_date,
                       action.action_kind,
                       COALESCE(action.new_shares_per_old_share, 0::DOUBLE)
                           AS new_shares_per_old_share,
                       COALESCE(action.cash_amount, 0::DOUBLE) AS cash_amount,
                       action.provisional, action.provenance,
                       action.new_shares_per_old_share IS NULL AS shares_is_null,
                       action.cash_amount IS NULL AS cash_is_null
                FROM corporate_action_current AS action
                JOIN selected_provider AS selected
                  ON selected.listing_id = action.listing_id
                 AND selected.provider = action.provider
                WHERE action.status = 'ACTIVE' AND action.effective_date <= ?
                ORDER BY action.listing_id, action.effective_date, action.action_kind
                """,
                [scope, through],
            ).to_arrow_table()
        return _canonical_hash(
            {
                "kind": "MarketSourcePrefixProofV1",
                "listing_ids": scope,
                "start": start,
                "through": through,
                "providers": _source_prefix_arrow_hash(providers),
                "raw": _source_prefix_arrow_hash(raw),
                "actions": _source_prefix_arrow_hash(actions),
            }
        )

    def listing_scope(
        self, manifest: UniverseManifest, *, listing_ids: Sequence[str]
    ) -> tuple[ManifestListing, ...]:
        """Resolve covered securities without asserting they are current research members.

        The caller's frozen Panel/holding authority supplies the scope. Data
        resolves identifiers only within the bound market/provider; it does not
        manufacture a larger current Universe manifest to make a consumer pass.
        """
        requested = tuple(sorted(set(listing_ids)))
        known = {item.listing_id: item for item in manifest.listings}
        missing = tuple(item for item in requested if item not in known)
        if missing:
            connection = self._connect(read_only=True)
            try:
                rows = connection.execute(
                    """
                    SELECT l.listing_id, l.display_symbol, l.mic, l.issuer_external_id,
                           p.provider_symbol
                    FROM listing l JOIN provider_symbol_mapping p ON p.listing_id = l.listing_id
                    WHERE l.market_profile_id = ? AND p.provider = ?
                      AND l.listing_id IN (SELECT unnest(?))
                    QUALIFY row_number() OVER (
                        PARTITION BY l.listing_id ORDER BY p.effective_from DESC
                    ) = 1
                    """,
                    [manifest.profile.market_profile_id, manifest.profile.provider, missing],
                ).fetchall()
            finally:
                connection.close()
            for row in rows:
                known[str(row[0])] = ManifestListing(
                    listing_id=str(row[0]),
                    symbol=str(row[1]),
                    mic=str(row[2]),
                    issuer_external_id=str(row[3]) if row[3] is not None else None,
                    provider_symbol=str(row[4]),
                )
        if not requested or set(requested) - known.keys():
            raise ValueError("market_data.covered_listing_identity_unavailable")
        return tuple(known[item] for item in requested)

    def execution_source_watermark(
        self,
        manifest: UniverseManifest,
        *,
        through: date,
        listing_ids: Sequence[str] | None = None,
    ) -> dict[str, object]:
        """Return payload-free source state for causal-outcome exact reuse.

        Current row values are never returned. New/deleted rows change the row
        counts; corrected rows and actions change the append-only revision
        counters. The immutable outcome manifest remains the authority for the
        exact source-row hashes consumed by an initial publication.
        """
        manifest_listing_ids = tuple(sorted(item.listing_id for item in manifest.listings))
        listing_ids = (
            tuple(item.listing_id for item in self.listing_scope(manifest, listing_ids=listing_ids))
            if listing_ids is not None
            else manifest_listing_ids
        )
        connection = self._connect(read_only=True)
        try:
            raw = connection.execute(
                """
                SELECT count(*), min(session_date), max(session_date),
                       count(DISTINCT listing_id)
                FROM raw_daily_bar_current
                WHERE listing_id IN (SELECT unnest(?)) AND session_date <= ?
                """,
                [listing_ids, through],
            ).fetchone()
            bar_revision = connection.execute(
                """
                SELECT count(*), max(observed_at)
                FROM bar_revision
                WHERE listing_id IN (SELECT unnest(?)) AND session_date <= ?
                """,
                [listing_ids, through],
            ).fetchone()
            actions = connection.execute(
                """
                SELECT count(*), min(effective_date), max(effective_date)
                FROM corporate_action_current
                WHERE listing_id IN (SELECT unnest(?)) AND effective_date <= ?
                """,
                [listing_ids, through],
            ).fetchone()
            action_revision = connection.execute(
                """
                SELECT count(*), max(observed_at)
                FROM corporate_action_revision
                WHERE listing_id IN (SELECT unnest(?)) AND effective_date <= ?
                """,
                [listing_ids, through],
            ).fetchone()
        finally:
            connection.close()
        values = {
            "manifest_revision": manifest.revision_sha256,
            "through": through,
            "listing_count": len(listing_ids),
            "raw_row_count": int(raw[0]),
            "raw_first_session": raw[1],
            "raw_last_session": raw[2],
            "raw_listing_count": int(raw[3]),
            "bar_revision_count": int(bar_revision[0]),
            "bar_revision_latest_observed_at": bar_revision[1],
            "action_row_count": int(actions[0]),
            "action_first_session": actions[1],
            "action_last_session": actions[2],
            "action_revision_count": int(action_revision[0]),
            "action_revision_latest_observed_at": action_revision[1],
        }
        if listing_ids != manifest_listing_ids:
            values["coverage_listing_set_hash"] = _canonical_hash(listing_ids)
        return {**values, "watermark_hash": _canonical_hash(values)}

    def manifest_raw_range(self, manifest: UniverseManifest) -> tuple[date, date] | None:
        """Return the canonical session range covered by one manifest."""
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT min(bar.session_date), max(bar.session_date)
                FROM raw_daily_bar_current AS bar
                JOIN universe_manifest_listing AS member USING (listing_id)
                WHERE member.manifest_id = ?
                """,
                [manifest.manifest_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None or row[0] is None or row[1] is None:
            return None
        return row[0], row[1]

    def listing_raw_ranges(
        self, listing_ids: Sequence[str], *, through: date | None = None
    ) -> dict[str, tuple[date, date]]:
        """Read local coverage for a frozen listing set in one scan.

        Bounded by ``through`` when given; the whole held history otherwise
        (a listing with no bars is absent from the answer).
        """
        if not listing_ids:
            return {}
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT listing_id, min(session_date), max(session_date)
                FROM raw_daily_bar_current
                WHERE listing_id IN (SELECT unnest(?)) AND (? IS NULL OR session_date <= ?)
                GROUP BY listing_id ORDER BY listing_id
                """,
                [list(listing_ids), through, through],
            ).fetchall()
        finally:
            connection.close()
        return {str(row[0]): (row[1], row[2]) for row in rows}

    def manifest_raw_through(self, manifest: UniverseManifest) -> date | None:
        """Return the latest session covered by every listing in a manifest."""
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT min(latest_session)
                FROM (
                    SELECT member.listing_id, max(bar.session_date) AS latest_session
                    FROM universe_manifest_listing AS member
                    LEFT JOIN raw_daily_bar_current AS bar USING (listing_id)
                    WHERE member.manifest_id = ?
                    GROUP BY member.listing_id
                )
                """,
                [manifest.manifest_id],
            ).fetchone()
        finally:
            connection.close()
        return row[0] if row is not None and row[0] is not None else None

    def manifest_provider_adjusted_through(self, manifest: UniverseManifest) -> date | None:
        """Return provider Adj Close coverage shared by every manifest listing.

        Raw bars alone do not make the desktop feature authority current.  This
        query follows each listing's latest provider mapping so a stale row
        from a superseded provider cannot make maintenance look complete.
        """
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                WITH current_provider AS (
                    SELECT listing_id, provider,
                           row_number() OVER (
                               PARTITION BY listing_id ORDER BY effective_from DESC
                           ) AS provider_rank
                    FROM provider_symbol_mapping
                ), listing_coverage AS (
                    SELECT member.listing_id,
                           max(adjusted.session_date) AS latest_session
                    FROM universe_manifest_listing AS member
                    LEFT JOIN current_provider AS provider
                      ON provider.listing_id = member.listing_id
                     AND provider.provider_rank = 1
                    LEFT JOIN provider_adjusted_close_current AS adjusted
                      ON adjusted.listing_id = member.listing_id
                     AND adjusted.provider = provider.provider
                    WHERE member.manifest_id = ?
                    GROUP BY member.listing_id
                )
                SELECT min(latest_session) FROM listing_coverage
                """,
                [manifest.manifest_id],
            ).fetchone()
        finally:
            connection.close()
        return row[0] if row is not None and row[0] is not None else None

    def market_reference_raw_range(self, reference_id: str) -> tuple[date, date] | None:
        """Return a reference's raw coverage without exposing its listing identity."""
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT min(bar.session_date), max(bar.session_date)
                FROM market_reference_current AS reference
                JOIN raw_daily_bar_current AS bar USING (listing_id)
                WHERE reference.reference_id = ?
                """,
                [reference_id],
            ).fetchone()
        finally:
            connection.close()
        if row is None or row[0] is None or row[1] is None:
            return None
        return row[0], row[1]

    def actions(
        self,
        listing_id: str,
        *,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> tuple[CorporateActionEvent, ...]:
        """Read the current corporate-action observations for one listing.

        Args:
            listing_id: Durable listing identity.
            _connection: Existing read connection for a larger transaction.

        Returns:
            Active action observations in effective-date order.

        """
        connection = _connection or self._connect(read_only=True)
        owns_connection = _connection is None
        try:
            rows = self._current_actions(
                connection, listing_id, self._provider_for_listing(connection, listing_id)
            )
        finally:
            if owns_connection:
                connection.close()
        return rows

    def actions_by_listing(
        self,
        listing_ids: Sequence[str],
        *,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> dict[str, tuple[CorporateActionEvent, ...]]:
        """``actions`` of many listings in one read, each under its latest provider mapping.

        Args:
            listing_ids: Durable listing identities; each answers, empty without actions.
            _connection: Existing read connection for a larger transaction.

        Returns:
            Each listing's active action observations in effective-date order.

        Raises:
            ValueError: A listing has no provider mapping, as ``actions`` refuses it.
        """
        scope = list(listing_ids)
        connection = _connection or self._connect(read_only=True)
        owns_connection = _connection is None
        try:
            if any(
                provider is None for provider in _selected_providers(connection, scope).values()
            ):
                raise ValueError("listing has no provider mapping")
            rows = connection.execute(
                _SELECTED_PROVIDER_SQL
                + """
                SELECT action.listing_id, action.provider, action.effective_date,
                       action.action_kind, action.new_shares_per_old_share,
                       action.cash_amount, action.provisional, action.provenance
                FROM corporate_action_current AS action
                JOIN selected_provider AS selected
                  ON selected.listing_id = action.listing_id
                 AND selected.provider = action.provider
                WHERE action.status = 'ACTIVE'
                ORDER BY action.listing_id, action.effective_date, action.action_kind
                """,
                [scope],
            ).fetchall()
        finally:
            if owns_connection:
                connection.close()
        events: dict[str, list[CorporateActionEvent]] = {listing_id: [] for listing_id in scope}
        for row in rows:
            events[row[0]].append(CorporateActionEvent(*row))
        return {listing_id: tuple(values) for listing_id, values in events.items()}

    def _active_action_set_hash(
        self,
        listing_id: str,
        *,
        _connection: duckdb.DuckDBPyConnection | None = None,
    ) -> str:
        """Return the exact active corporate-action identity for one listing."""
        return action_set_hash(self.actions(listing_id, _connection=_connection))

    @staticmethod
    def _provider_for_listing(connection: duckdb.DuckDBPyConnection, listing_id: str) -> str:
        row = connection.execute(
            """
            SELECT provider FROM provider_symbol_mapping
            WHERE listing_id = ? ORDER BY effective_from DESC LIMIT 1
            """,
            [listing_id],
        ).fetchone()
        if row is None:
            raise ValueError("listing has no provider mapping")
        return str(row[0])

    def revision_count(self, listing_id: str) -> int:
        """Count stored raw-bar revisions for one listing."""
        connection = self._connect(read_only=True)
        try:
            return int(
                connection.execute(
                    "SELECT count(*) FROM bar_revision WHERE listing_id = ?", [listing_id]
                ).fetchone()[0]
            )
        finally:
            connection.close()

    def maintenance_revision_fingerprint(
        self, manifest: UniverseManifest, *, listing_ids: Sequence[str] | None = None
    ) -> str:
        """One metadata scan of governed revision journals, including same-day corrections."""
        source = self.execution_source_watermark(
            manifest, through=date.max, listing_ids=listing_ids
        )
        listing_ids = (
            tuple(listing_ids)
            if listing_ids is not None
            else tuple(item.listing_id for item in manifest.listings)
        )
        connection = self._connect(read_only=True)
        try:
            adjusted = connection.execute(
                "SELECT listing_id, count(*), max(observed_at) "
                "FROM provider_adjusted_series_revision "
                "WHERE listing_id IN (SELECT unnest(?)) "
                "GROUP BY listing_id ORDER BY listing_id",
                [listing_ids],
            ).fetchall()
            return _canonical_hash({"source": source, "adjusted_revisions": adjusted})
        finally:
            connection.close()

    def action_revision_count(self, listing_id: str) -> int:
        """Count stored corporate-action revisions for one listing."""
        connection = self._connect(read_only=True)
        try:
            return int(
                connection.execute(
                    "SELECT count(*) FROM corporate_action_revision WHERE listing_id = ?",
                    [listing_id],
                ).fetchone()[0]
            )
        finally:
            connection.close()

    def record_remediation_execution(
        self, proposal_hash: str, option_id: str, *, executed_at: datetime
    ) -> bool:
        """Record one remediation execution and report whether it was new.

        Args:
            proposal_hash: Identity of the accepted remediation proposal.
            option_id: Selected deterministic remediation option.
            executed_at: Execution time recorded with the receipt.

        Returns:
            ``True`` if this execution was first recorded.

        """
        executed_at = _utc_naive(executed_at)
        connection = self._connect()
        try:
            exists = connection.execute(
                "SELECT 1 FROM remediation_execution WHERE proposal_hash = ?", [proposal_hash]
            ).fetchone()
            if exists is not None:
                return False
            connection.execute(
                "INSERT INTO remediation_execution VALUES (?, ?, ?)",
                [proposal_hash, option_id, executed_at],
            )
            return True
        finally:
            connection.close()

    def remediation_count(self) -> int:
        """Count recorded remediation executions."""
        connection = self._connect(read_only=True)
        try:
            return int(
                connection.execute("SELECT count(*) FROM remediation_execution").fetchone()[0]
            )
        finally:
            connection.close()

    def prove_atomic_rollback(self, listing_id: str) -> None:
        """Fixture-only proof that a failed batch cannot leave a quality update behind."""
        connection = self._connect()
        try:
            before = connection.execute(
                "SELECT quality_state FROM data_quality WHERE listing_id = ?", [listing_id]
            ).fetchone()[0]
            row = connection.execute(
                """
                SELECT listing_id, provider, session_date, open, high, low, close, volume,
                       payload_hash, observed_at
                FROM raw_daily_bar_current WHERE listing_id = ? LIMIT 1
                """,
                [listing_id],
            ).fetchone()
            assert row is not None
            try:
                connection.execute("BEGIN TRANSACTION")
                connection.execute(
                    "UPDATE data_quality SET quality_state = 'DATA_UNREADY' WHERE listing_id = ?",
                    [listing_id],
                )
                connection.execute(
                    """
                    INSERT INTO raw_daily_bar_current (
                        listing_id, provider, session_date, open, high, low, close, volume,
                        payload_hash, observed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    list(row),
                )
            except duckdb.ConstraintException:
                connection.execute("ROLLBACK")
            else:
                raise AssertionError("duplicate raw bar unexpectedly committed")
            after = connection.execute(
                "SELECT quality_state FROM data_quality WHERE listing_id = ?", [listing_id]
            ).fetchone()[0]
            if before != after:
                raise AssertionError("failed transaction changed data_quality")
        finally:
            connection.close()

    def prove_same_issuer_can_have_multiple_listings(self, market_profile_id: str) -> None:
        """Fixture-only proof that issuer identifiers are never the listing primary key."""
        connection = self._connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.executemany(
                "INSERT INTO listing VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        "fixture-listing-class-a",
                        market_profile_id,
                        "FIXTUREA",
                        "XNAS",
                        "SEC:0000123456",
                    ),
                    (
                        "fixture-listing-class-b",
                        market_profile_id,
                        "FIXTUREB",
                        "XNYS",
                        "SEC:0000123456",
                    ),
                ],
            )
            count = connection.execute(
                "SELECT count(*) FROM listing WHERE issuer_external_id = ?", ["SEC:0000123456"]
            ).fetchone()[0]
            if count != 2:
                raise AssertionError("issuer identity collapsed distinct listings")
            connection.execute("ROLLBACK")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
