"""Bounded construction and readback of causal tradability surfaces."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq

from alphalattice.control.observation_runtime.telemetry.progress import WorkProgressUpdate
from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor
from alphalattice.foundation.causal_outcomes.execution.compile import execution_session_status
from alphalattice.foundation.market_data_ops.sources.contracts import RawDailyBar
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data.enums import ExecutionSessionStatus
from alphalattice.kernel.shared_kernel.arrow_identity import canonical_row_hashes
from alphalattice.kernel.shared_kernel.identity import _CANONICAL_ENCODER, canonical_hash

from .contracts import (
    DecisionTradabilityReason,
    DecisionTradabilityStatus,
    ExecutionObservationMethod,
    HistoricalDecisionTradabilitySurface,
    HistoricalExecutionAvailabilitySurface,
    HistoricalTradabilityBundle,
    TradabilitySurfaceChunk,
    seal_tradability_contract,
    tradability_chunk_identity,
)

_PREFIX = "playpen://data-operations/tradability/"
_DECISION_SCHEMA_ID = "decision-tradability-snapshot"
_EXECUTION_SCHEMA_ID = "execution-availability-observation"


def _read_raw_bar_table(
    store: MarketDataRepository,
    listing_ids: tuple[str, ...],
    *,
    start: date,
    through: date,
) -> pa.Table:
    """Read one bounded OHLCV block without growing the central store API."""
    if not listing_ids or listing_ids != tuple(sorted(set(listing_ids))):
        raise TradabilitySurfaceError("data_tradability.listing_axis_invalid")
    if start > through:
        raise TradabilitySurfaceError("data_tradability.source_range_invalid")
    connection = store._connect(read_only=True)
    try:
        return connection.execute(
            """
            SELECT listing_id, provider, session_date, open, high, low, close, volume
            FROM raw_daily_bar_current
            WHERE listing_id IN (SELECT unnest(?))
              AND session_date BETWEEN ? AND ?
            ORDER BY session_date, listing_id
            """,
            [list(listing_ids), start, through],
        ).to_arrow_table()
    finally:
        connection.close()


def _decision_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("formation_session", pa.date32(), nullable=False),
            pa.field("intended_execution_session", pa.date32(), nullable=False),
            pa.field("listing_id", pa.string(), nullable=False),
            pa.field("decision_status", pa.string(), nullable=False),
            pa.field("reason_code", pa.string(), nullable=False),
            pa.field("causal_adv20", pa.float64(), nullable=True),
            pa.field("observed_through", pa.date32(), nullable=False),
            pa.field("source_row_hash", pa.string(), nullable=False),
            pa.field("row_hash", pa.string(), nullable=False),
        ]
    )


def _execution_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("formation_session", pa.date32(), nullable=False),
            pa.field("intended_execution_session", pa.date32(), nullable=False),
            pa.field("listing_id", pa.string(), nullable=False),
            pa.field("execution_status", pa.string(), nullable=False),
            pa.field("observation_method", pa.string(), nullable=False),
            pa.field("observed_through", pa.date32(), nullable=False),
            pa.field("source_row_hash", pa.string(), nullable=False),
            pa.field("row_hash", pa.string(), nullable=False),
        ]
    )


class TradabilitySurfaceError(ValueError):
    """Stable Data-owned failure at the tradability authority boundary."""


class TradabilityBuildCancelled(RuntimeError):
    """Raised only between immutable chunk publications."""


@dataclass(frozen=True, slots=True)
class PublishedTradabilitySurfaces:
    """Carry built decision/execution authorities and their published artifact descriptors.

    Attributes:
        decision: Sealed planned-order eligibility manifest.
        decision_artifact: Published decision manifest descriptor.
        execution: Sealed observed execution-availability manifest.
        execution_artifact: Published execution manifest descriptor.
        bundle: Sealed linkage of the two manifests.
        bundle_artifact: Published bundle descriptor.
    """

    decision: HistoricalDecisionTradabilitySurface
    decision_artifact: ArtifactDescriptor
    execution: HistoricalExecutionAvailabilitySurface
    execution_artifact: ArtifactDescriptor
    bundle: HistoricalTradabilityBundle
    bundle_artifact: ArtifactDescriptor


@dataclass(frozen=True, slots=True)
class PortfolioTradabilityRiskInputs:
    """Narrow Risk evidence projected for Portfolio input construction."""

    return_surface_hash: str
    covariance_surface_hash: str
    epoch_hash: str
    market_profile_id: str
    ordered_listing_ids: tuple[str, ...]
    universe_manifest_revision: str
    formation_sessions: tuple[date, ...]
    formation_count: int
    asset_count: int
    available_return_sessions: tuple[date, ...]


@dataclass(frozen=True, slots=True)
class PortfolioTradabilityMarketInputs:
    """Market-only projection for a policy that consumes no covariance."""

    source_identity_hash: str
    epoch_hash: str
    market_profile_id: str
    ordered_listing_ids: tuple[str, ...]
    universe_manifest_revision: str
    formation_sessions: tuple[date, ...]
    available_return_sessions: tuple[date, ...]

    @property
    def asset_count(self) -> int:
        """Count the declared market-only listing axis.

        Returns:
            The number of ordered listing identities.
        """
        return len(self.ordered_listing_ids)

    @property
    def formation_count(self) -> int:
        """Count the declared market-only formation axis.

        Returns:
            The number of formation session labels.
        """
        return len(self.formation_sessions)


@dataclass(frozen=True, slots=True)
class _TradabilityBuildInputs:
    authority_hash: str
    epoch_hash: str
    formations: tuple[date, ...]
    available_sessions: tuple[date, ...]
    session_positions: Mapping[date, int]
    intended_execution_sessions: tuple[date, ...]
    schedule_hash: str
    ordered_listing_ids: tuple[str, ...]
    source_watermark_hash: str


class TradabilityArtifactStore:
    """Own immutable Parquet chunks and compact tradability JSON authorities."""

    def __init__(
        self, artifact_root: Path, *, capacity: Callable[[int], None] = lambda _bytes: None
    ) -> None:
        """Choose immutable tradability storage and its new-chunk capacity callback.

        Args:
            artifact_root: Workspace artifact root containing the Data operations subtree.
            capacity: Callback admitting the byte count before a new Parquet payload is written.
        """
        self.root = artifact_root.resolve() / "data-operations" / "tradability"
        self._capacity = capacity

    @staticmethod
    def uri(category: str, content_hash: str) -> str:
        """Construct a content-addressed handle in the tradability owner namespace.

        Args:
            category: Owner artifact category, such as decision/chunks or current/markers.
            content_hash: Artifact identity appended to the category.

        Returns:
            The playpen URI; read/publish boundaries validate identities separately.
        """
        return f"{_PREFIX}{category}/{content_hash}"

    @staticmethod
    def _require_hash(value: str) -> None:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise TradabilitySurfaceError("data_tradability.artifact_identity_invalid")

    @staticmethod
    def _atomic_write(target: Path, content: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        staged.write_bytes(content)
        os.replace(staged, target)
        staged.unlink(missing_ok=True)

    def _path(self, category: str, content_hash: str, suffix: str) -> Path:
        self._require_hash(content_hash)
        return self.root / category / f"{content_hash}.{suffix}"

    def publish_chunk(
        self,
        *,
        surface_kind: Literal["DECISION", "EXECUTION"],
        table: pa.Table,
    ) -> TradabilitySurfaceChunk:
        """Publish a canonical ordered Parquet payload for at most 21 formation sessions.

        Rows are sorted by formation and listing, then the owner attaches schema metadata
        and writes Zstandard-compressed Parquet. Identity binds the exact payload bytes,
        canonical metadata, session labels and row count. Existing identical bytes reuse
        the chunk; the same identity with different bytes is refused.

        Args:
            surface_kind: DECISION or EXECUTION schema family.
            table: Nonempty Arrow table with the owner's declared column names.

        Returns:
            The validated content-addressed chunk reference.

        Raises:
            TradabilitySurfaceError: Column names, row/session count or existing identity bytes are
                invalid.
        """
        schema = _decision_schema() if surface_kind == "DECISION" else _execution_schema()
        if table.schema.names != schema.names or table.num_rows == 0:
            raise TradabilitySurfaceError("data_tradability.chunk_schema_invalid")
        ordered = table.combine_chunks().sort_by(
            [("formation_session", "ascending"), ("listing_id", "ascending")]
        )
        sessions = tuple(sorted(set(ordered.column("formation_session").to_pylist())))
        if not sessions or len(sessions) > 21:
            raise TradabilitySurfaceError("data_tradability.chunk_session_axis_invalid")
        metadata = {
            b"alphalattice.snapshot_kind": (
                _DECISION_SCHEMA_ID if surface_kind == "DECISION" else _EXECUTION_SCHEMA_ID
            ).encode(),
            b"alphalattice.surface_kind": surface_kind.encode(),
        }
        buffer = pa.BufferOutputStream()
        pq.write_table(ordered.replace_schema_metadata(metadata), buffer, compression="zstd")
        content = buffer.getvalue().to_pybytes()
        payload_sha256 = hashlib.sha256(content).hexdigest()
        serialized_metadata = json.dumps(
            {key.decode(): value.decode() for key, value in sorted(metadata.items())},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        metadata_hash = hashlib.sha256(serialized_metadata).hexdigest()
        content_hash = canonical_hash(
            tradability_chunk_identity(
                surface_kind=surface_kind,
                formation_sessions=sessions,
                row_count=ordered.num_rows,
                payload_sha256=payload_sha256,
                metadata_hash=metadata_hash,
            )
        )
        target = self._path(f"{surface_kind.lower()}/chunks", content_hash, "parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != content:
                raise TradabilitySurfaceError("data_tradability.chunk_identity_reused")
        else:
            self._capacity(len(content))
            self._atomic_write(target, content)
        return TradabilitySurfaceChunk(
            surface_kind=surface_kind,
            formation_sessions=sessions,
            row_count=ordered.num_rows,
            payload_sha256=payload_sha256,
            metadata_hash=metadata_hash,
            content_hash=content_hash,
            uri=self.uri(f"{surface_kind.lower()}/chunks", content_hash),
        )

    def resolve_chunk(self, chunk: TradabilitySurfaceChunk) -> Path:
        """Resolve a chunk only after verifying its URI, payload bytes, metadata and rows.

        Args:
            chunk: Typed immutable chunk reference requested by a consumer.

        Returns:
            The verified local Parquet path.

        Raises:
            TradabilitySurfaceError: URI, payload digest, Arrow metadata or row count differs from
                the reference.
        """
        category = f"{chunk.surface_kind.lower()}/chunks"
        if chunk.uri != self.uri(category, chunk.content_hash):
            raise TradabilitySurfaceError("data_tradability.chunk_uri_invalid")
        path = self._path(category, chunk.content_hash, "parquet")
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != chunk.payload_sha256
        ):
            raise TradabilitySurfaceError("data_tradability.chunk_tampered")
        parquet = pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        serialized = json.dumps(
            {key.decode(): value.decode() for key, value in sorted(metadata.items())},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if hashlib.sha256(serialized).hexdigest() != chunk.metadata_hash:
            raise TradabilitySurfaceError("data_tradability.chunk_metadata_tampered")
        if parquet.metadata.num_rows != chunk.row_count:
            raise TradabilitySurfaceError("data_tradability.chunk_row_count_invalid")
        return path

    def publish_json(
        self,
        *,
        category: str,
        payload: Mapping[str, object],
        identity_field: str,
    ) -> ArtifactDescriptor:
        """Publish canonical JSON with a validated named identity field.

        Args:
            category: Owner artifact category used for storage and URI derivation.
            payload: JSON-compatible contract payload including its identity field.
            identity_field: Canonical content hash field excluded from its own preimage.

        Returns:
            The content/metadata/URI artifact descriptor, reusing existing identical bytes.

        Raises:
            TradabilitySurfaceError: The identity is malformed, its preimage differs or stored bytes
                reuse it inconsistently.
        """
        content_hash = str(payload[identity_field])
        self._require_hash(content_hash)
        identity = dict(payload)
        identity.pop(identity_field)
        if canonical_hash(identity) != content_hash:
            raise TradabilitySurfaceError("data_tradability.json_identity_invalid")
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        target = self._path(category, content_hash, "json")
        if target.exists() and target.read_bytes() != content:
            raise TradabilitySurfaceError("data_tradability.json_identity_reused")
        if not target.exists():
            self._atomic_write(target, content)
        return ArtifactDescriptor(
            kind=f"data-tradability-{category.replace('/', '-')}",
            content_hash=content_hash,
            metadata_hash=hashlib.sha256(content).hexdigest(),
            uri=self.uri(category, content_hash),
        )

    def load_json(self, *, category: str, uri: str, identity_field: str) -> dict[str, object]:
        """Reopen a category-scoped JSON artifact and verify its requested identity.

        Args:
            category: Expected owner artifact category.
            uri: Content-addressed handle in that category.
            identity_field: Payload field expected to equal the handle's identity.

        Returns:
            The verified JSON object.

        Raises:
            TradabilitySurfaceError: URI scope, requested identity or canonical payload commitment
                differs.
            FileNotFoundError: The content-addressed artifact does not exist.
        """
        prefix = self.uri(category, "")
        if not uri.startswith(prefix):
            raise TradabilitySurfaceError("data_tradability.artifact_uri_invalid")
        content_hash = uri[len(prefix) :]
        payload = json.loads(self._path(category, content_hash, "json").read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get(identity_field) != content_hash:
            raise TradabilitySurfaceError("data_tradability.json_requested_identity_invalid")
        identity = dict(payload)
        identity.pop(identity_field)
        if canonical_hash(identity) != content_hash:
            raise TradabilitySurfaceError("data_tradability.json_tampered")
        return cast(dict[str, object], payload)


class HistoricalTradabilityBuilder:
    """Build both surfaces from one bounded source read, published per 21-formation chunk."""

    def __init__(
        self,
        *,
        market_store: MarketDataRepository,
        artifact_root: Path,
        progress_sink: Callable[[WorkProgressUpdate], object] | None = None,
        capacity: Callable[[int], None] = lambda _bytes: None,
    ) -> None:
        """Bind the local Market source and bounded tradability artifact publication owner.

        Args:
            market_store: Local Market repository supplying source bars and session coverage.
            artifact_root: Workspace artifact root for immutable output surfaces.
            progress_sink: Optional safe progress-observation callback.
            capacity: Capacity admission callback before publishing a new Parquet chunk.
        """
        self.market_store = market_store
        self.artifacts = TradabilityArtifactStore(artifact_root, capacity=capacity)
        self.progress_sink = progress_sink
        self._inputs: _TradabilityBuildInputs | None = None

    def current_surfaces_match(
        self,
        *,
        risk_inputs: PortfolioTradabilityRiskInputs,
        decision: HistoricalDecisionTradabilitySurface,
        execution: HistoricalExecutionAvailabilitySurface,
        bundle: HistoricalTradabilityBundle,
    ) -> bool:
        """Compare current surfaces with every source-derived build identity."""
        inputs = self._prepare_inputs(risk_inputs)
        return (
            bundle.universe_epoch_hash == inputs.epoch_hash
            and bundle.schedule_hash == inputs.schedule_hash
            and decision.universe_epoch_hash == inputs.epoch_hash
            and execution.universe_epoch_hash == inputs.epoch_hash
            and decision.ordered_listing_ids == inputs.ordered_listing_ids
            and execution.ordered_listing_ids == inputs.ordered_listing_ids
            and decision.source_watermark_hash == inputs.source_watermark_hash
            and execution.source_watermark_hash == inputs.source_watermark_hash
            and decision.schedule_hash == inputs.schedule_hash
            and execution.schedule_hash == inputs.schedule_hash
            and decision.formation_sessions == inputs.formations
            and execution.formation_sessions == inputs.formations
            and decision.intended_execution_sessions == inputs.intended_execution_sessions
            and execution.intended_execution_sessions == inputs.intended_execution_sessions
        )

    def build(
        self,
        *,
        risk_inputs: PortfolioTradabilityRiskInputs | None = None,
        market_inputs: PortfolioTradabilityMarketInputs | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> PublishedTradabilitySurfaces:
        """Build planned eligibility and observed execution availability from one authority.

        Exactly one Risk or market-only input authority is required. One bounded local
        source read covers the first formation's ADV20 history through the final intended
        execution session. Publication is chunked in at most 21 formations, with optional
        cancellation between immutable chunk publications. Current-universe scope and
        daily-bar execution limitations remain explicit in both resulting manifests.

        Args:
            risk_inputs: Optional resolved Risk return/covariance input authority.
            market_inputs: Optional market-only authority for a policy consuming no covariance.
            should_cancel: Optional cancellation predicate checked between chunk publications.

        Returns:
            Published decision/execution manifests, their bundle and artifact descriptors.

        Raises:
            TradabilitySurfaceError: Authority choice, source coverage or output contract evidence
                is invalid.
            TradabilityBuildCancelled: The cancellation predicate requests a stop between chunks.
        """
        if (risk_inputs is None) == (market_inputs is None):
            raise TradabilitySurfaceError("data_tradability.one_authority_required")
        inputs = self._prepare_inputs(risk_inputs if risk_inputs is not None else market_inputs)
        formations = inputs.formations
        available = inputs.available_sessions
        positions = inputs.session_positions
        intended = inputs.intended_execution_sessions
        schedule_hash = inputs.schedule_hash
        listing_ids = inputs.ordered_listing_ids
        watermark_hash = inputs.source_watermark_hash
        decision_chunks: list[TradabilitySurfaceChunk] = []
        execution_chunks: list[TradabilitySurfaceChunk] = []
        decision_counts: Counter[str] = Counter()
        execution_counts: Counter[str] = Counter()
        chunks = tuple(formations[index : index + 21] for index in range(0, len(formations), 21))
        # The whole build's source in one bounded read: the first formation's
        # ADV20 history through the last formation's execution session. Read
        # per chunk, the same bars were queried again for every overlapping
        # window and each source scan was paid once per chunk; the surfaces
        # are the same rows either way, since every row reads only the bars
        # of its own formation, history and execution session.
        raw = _read_raw_bar_table(
            self.market_store,
            listing_ids,
            start=available[positions[formations[0]] - 19],
            through=available[positions[formations[-1]] + 1],
        )
        bars = _bar_map(raw, listing_ids)
        del raw
        # A bar's source fragment is the same in every history window it
        # appears in; encoded once here rather than twenty times per formation.
        fragments: dict[tuple[date, str], str] = {}
        for chunk_index, chunk_formations in enumerate(chunks, start=1):
            if should_cancel is not None and should_cancel():
                raise TradabilityBuildCancelled
            decision_cells: list[tuple[object, ...]] = []
            execution_cells: list[tuple[object, ...]] = []
            for formation in chunk_formations:
                formation_position = positions[formation]
                execution_session = available[formation_position + 1]
                history = available[formation_position - 19 : formation_position + 1]
                for listing_id in listing_ids:
                    status, reason, adv20, history_bars = _decision_cell(
                        listing_id=listing_id,
                        formation_session=formation,
                        history=history,
                        bars=bars,
                    )
                    decision_cells.append(
                        (
                            formation,
                            execution_session,
                            listing_id,
                            status.value,
                            reason.value,
                            adv20,
                            formation,
                            _history_identity(history_bars, fragments),
                        )
                    )
                    decision_counts[status.value] += 1
                    bar = bars.get((execution_session, listing_id))
                    execution_status, method = _execution_cell(bar)
                    execution_cells.append(
                        (
                            formation,
                            execution_session,
                            listing_id,
                            execution_status.value,
                            method.value,
                            execution_session,
                            None if bar is None else bar.open,
                            None if bar is None else bar.volume,
                        )
                    )
                    execution_counts[execution_status.value] += 1
            decision_chunks.append(
                self.artifacts.publish_chunk(
                    surface_kind="DECISION", table=_decision_table(decision_cells)
                )
            )
            execution_chunks.append(
                self.artifacts.publish_chunk(
                    surface_kind="EXECUTION", table=_execution_table(execution_cells)
                )
            )
            self._progress(
                operation_id=schedule_hash,
                completed=chunk_index,
                total=len(chunks),
                current_item=chunk_formations[-1].isoformat(),
            )
        limitations = (
            "Current-universe research only; historical membership is not point-in-time.",
            "Daily-bar execution eligibility is assumed unless venue evidence is available.",
            "ADV20 is a capacity descriptor; Portfolio owns liquidity policy.",
        )
        common = {
            "universe_epoch_hash": inputs.epoch_hash,
            "ordered_listing_ids": listing_ids,
            "source_watermark_hash": watermark_hash,
            "schedule_hash": schedule_hash,
            "formation_sessions": formations,
            "intended_execution_sessions": intended,
            "data_validity_class": "CURRENT_UNIVERSE_RESEARCH_ONLY",
            "limitations": limitations,
        }
        decision = seal_tradability_contract(
            HistoricalDecisionTradabilitySurface,
            "surface_hash",
            **common,
            chunks=tuple(decision_chunks),
            eligible_row_count=decision_counts[DecisionTradabilityStatus.PLANNED_ORDER_ELIGIBLE],
            unavailable_row_count=decision_counts[
                DecisionTradabilityStatus.PLANNED_ORDER_UNAVAILABLE
            ],
        )
        execution = seal_tradability_contract(
            HistoricalExecutionAvailabilitySurface,
            "surface_hash",
            **common,
            chunks=tuple(execution_chunks),
            execution_status_counts=dict(sorted(execution_counts.items())),
        )
        decision_artifact = self.artifacts.publish_json(
            category="decision/manifests",
            payload=decision.model_dump(mode="json"),
            identity_field="surface_hash",
        )
        execution_artifact = self.artifacts.publish_json(
            category="execution/manifests",
            payload=execution.model_dump(mode="json"),
            identity_field="surface_hash",
        )
        bundle = seal_tradability_contract(
            HistoricalTradabilityBundle,
            "bundle_hash",
            universe_epoch_hash=inputs.epoch_hash,
            schedule_hash=schedule_hash,
            decision_surface_hash=decision.surface_hash,
            execution_surface_hash=execution.surface_hash,
            formation_count=len(formations),
            asset_count=len(listing_ids),
            coverage_start=formations[0],
            coverage_end=formations[-1],
        )
        bundle_artifact = self.artifacts.publish_json(
            category="bundles",
            payload=bundle.model_dump(mode="json"),
            identity_field="bundle_hash",
        )
        self._progress(
            operation_id=schedule_hash,
            completed=len(chunks),
            total=len(chunks),
            current_item=None,
            status="SUCCEEDED",
        )
        return PublishedTradabilitySurfaces(
            decision=decision,
            decision_artifact=decision_artifact,
            execution=execution,
            execution_artifact=execution_artifact,
            bundle=bundle,
            bundle_artifact=bundle_artifact,
        )

    def _prepare_inputs(
        self,
        risk_inputs: PortfolioTradabilityRiskInputs | PortfolioTradabilityMarketInputs | None,
    ) -> _TradabilityBuildInputs:
        if risk_inputs is None:
            raise TradabilitySurfaceError("data_tradability.one_authority_required")
        authority_hash = canonical_hash(asdict(risk_inputs))
        cached = self._inputs
        if cached is not None:
            if cached.authority_hash != authority_hash:
                raise TradabilitySurfaceError("data_tradability.builder_authority_changed")
            return cached
        if (
            risk_inputs.asset_count != len(risk_inputs.ordered_listing_ids)
            or risk_inputs.formation_count != len(risk_inputs.formation_sessions)
            or not risk_inputs.formation_sessions
            or risk_inputs.formation_sessions != tuple(sorted(set(risk_inputs.formation_sessions)))
            or risk_inputs.available_return_sessions
            != tuple(sorted(set(risk_inputs.available_return_sessions)))
        ):
            raise TradabilitySurfaceError("data_tradability.risk_authority_mismatch")
        formations = risk_inputs.formation_sessions
        available = risk_inputs.available_return_sessions
        positions = {session: index for index, session in enumerate(available)}
        try:
            intended = tuple(available[positions[session] + 1] for session in formations)
        except (KeyError, IndexError) as error:
            raise TradabilitySurfaceError("data_tradability.next_session_unavailable") from error
        first_position = positions[formations[0]]
        if first_position < 19:
            raise TradabilitySurfaceError("data_tradability.adv20_warmup_unavailable")
        schedule_hash = canonical_hash(tuple(zip(formations, intended, strict=True)))
        try:
            manifest = self.market_store.load_universe_manifest_revision(
                risk_inputs.universe_manifest_revision
            )
            listing_ids = risk_inputs.ordered_listing_ids
            self.market_store.listing_scope(manifest, listing_ids=listing_ids)
        except (KeyError, ValueError) as error:
            raise TradabilitySurfaceError("data_tradability.universe_epoch_mismatch") from error
        if manifest.profile.market_profile_id != risk_inputs.market_profile_id or (
            listing_ids != tuple(sorted(set(listing_ids)))
        ):
            raise TradabilitySurfaceError("data_tradability.universe_epoch_mismatch")
        # Market usability also serves historical samples and held leavers. It
        # does not grant membership; the Portfolio consumer applies dated
        # membership before selecting positions from these covered observations.
        watermark = self.market_store.execution_source_watermark(
            manifest, through=intended[-1], listing_ids=listing_ids
        )
        watermark_hash = str(watermark["watermark_hash"])
        prepared = _TradabilityBuildInputs(
            authority_hash=authority_hash,
            epoch_hash=risk_inputs.epoch_hash,
            formations=formations,
            available_sessions=available,
            session_positions=positions,
            intended_execution_sessions=intended,
            schedule_hash=schedule_hash,
            ordered_listing_ids=listing_ids,
            source_watermark_hash=watermark_hash,
        )
        self._inputs = prepared
        return prepared

    def _progress(
        self,
        *,
        operation_id: str,
        completed: int,
        total: int,
        current_item: str | None,
        status: Literal["RUNNING", "SUCCEEDED"] = "RUNNING",
    ) -> None:
        if self.progress_sink is not None:
            self.progress_sink(
                WorkProgressUpdate(
                    operation_id=operation_id,
                    stage_id="build_historical_tradability",
                    status=status,
                    completed_units=completed,
                    total_units=total,
                    unit_name="chunks",
                    current_item=current_item,
                )
            )


def _bar_map(table: pa.Table, listing_ids: tuple[str, ...]) -> dict[tuple[date, str], RawDailyBar]:
    allowed = set(listing_ids)
    result: dict[tuple[date, str], RawDailyBar] = {}
    # Column by column rather than ``to_pylist``: the rows are only ever read
    # positionally here, and a dict per bar was most of what building this map cost.
    names = ("listing_id", "provider", "session_date", "open", "high", "low", "close", "volume")
    columns = [table.column(name).to_pylist() for name in names]
    for listing, provider, session_date, open_, high, low, close, volume in zip(
        *columns, strict=True
    ):
        listing_id = str(listing)
        if listing_id not in allowed:
            raise TradabilitySurfaceError("data_tradability.source_axis_mismatch")
        bar = RawDailyBar(
            listing_id=listing_id,
            provider=str(provider),
            session_date=session_date,
            open=float(open_),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=int(volume),
        )
        key = (bar.session_date, listing_id)
        if key in result:
            raise TradabilitySurfaceError("data_tradability.source_duplicate")
        result[key] = bar
    return result


def _valid_formation_bar(bar: RawDailyBar | None) -> bool:
    return bool(
        bar is not None
        and all(math.isfinite(value) and value > 0.0 for value in (bar.open, bar.close))
        and bar.volume > 0
    )


def _decision_cell(
    *,
    listing_id: str,
    formation_session: date,
    history: tuple[date, ...],
    bars: Mapping[tuple[date, str], RawDailyBar],
) -> tuple[
    DecisionTradabilityStatus,
    DecisionTradabilityReason,
    float | None,
    tuple[RawDailyBar | None, ...],
]:
    """The decision rule for one ``(formation, listing)`` cell, and the bars it read."""
    formation_bar = bars.get((formation_session, listing_id))
    history_bars = tuple(bars.get((session, listing_id)) for session in history)
    if not _valid_formation_bar(formation_bar):
        return (
            DecisionTradabilityStatus.PLANNED_ORDER_UNAVAILABLE,
            DecisionTradabilityReason.FORMATION_MARKET_DATA_UNAVAILABLE,
            None,
            history_bars,
        )
    if len(history) != 20 or not all(_valid_formation_bar(bar) for bar in history_bars):
        return (
            DecisionTradabilityStatus.PLANNED_ORDER_UNAVAILABLE,
            DecisionTradabilityReason.ADV20_HISTORY_INCOMPLETE,
            None,
            history_bars,
        )
    dollar_volumes = (
        cast(RawDailyBar, bar).close * cast(RawDailyBar, bar).volume for bar in history_bars
    )
    return (
        DecisionTradabilityStatus.PLANNED_ORDER_ELIGIBLE,
        DecisionTradabilityReason.CAUSAL_MARKET_DATA_COMPLETE,
        sum(dollar_volumes) / 20.0,
        history_bars,
    )


def _history_identity(
    history_bars: tuple[RawDailyBar | None, ...],
    fragments: dict[tuple[date, str], str] | None = None,
) -> str:
    """``canonical_hash`` of a decision row's source: its history bars, or null where absent.

    The canonical encoder writes a sequence as its members' encodings joined by
    ``,`` between ``[`` and ``]``, so encoding each bar's member once and
    joining is the same document it would write for the whole tuple; a build
    keeps the encodings in ``fragments`` across the windows a bar appears in.
    """
    parts: list[str] = []
    for bar in history_bars:
        if bar is None:
            parts.append("null")
            continue
        key = (bar.session_date, bar.listing_id)
        text = None if fragments is None else fragments.get(key)
        if text is None:
            text = _CANONICAL_ENCODER.encode(
                (bar.session_date, bar.open, bar.close, bar.volume, bar.provider)
            )
            if fragments is not None:
                fragments[key] = text
        parts.append(text)
    return hashlib.sha256(("[" + ",".join(parts) + "]").encode("utf-8")).hexdigest()


def _decision_row(
    *,
    listing_id: str,
    formation_session: date,
    intended_execution_session: date | None,
    history: tuple[date, ...],
    bars: Mapping[tuple[date, str], RawDailyBar],
) -> dict[str, object]:
    status, reason, adv20, history_bars = _decision_cell(
        listing_id=listing_id, formation_session=formation_session, history=history, bars=bars
    )
    values: dict[str, object] = {
        "formation_session": formation_session,
        "intended_execution_session": intended_execution_session,
        "listing_id": listing_id,
        "decision_status": status.value,
        "reason_code": reason.value,
        "causal_adv20": adv20,
        "observed_through": formation_session,
        "source_row_hash": _history_identity(history_bars),
    }
    values["row_hash"] = canonical_hash(values)
    return values


def _decision_table(cells: list[tuple[object, ...]]) -> pa.Table:
    """A chunk's decision rows as ``_decision_row`` states them, sealed from the columns.

    Each cell is the row's members in schema order less the row identity; the
    identity of every row is the canonical hash of its members, taken from the
    columns at once rather than from a dict per row.
    """
    schema = _decision_schema()
    names = schema.names[:-1]
    columns = [
        pa.array(list(column), type=schema.field(name).type)
        for name, column in zip(names, zip(*cells, strict=True), strict=True)
    ]
    row_hashes = canonical_row_hashes(pa.Table.from_arrays(columns, names=names))
    return pa.Table.from_arrays([*columns, pa.array(row_hashes, pa.string())], schema=schema)


def _execution_table(cells: list[tuple[object, ...]]) -> pa.Table:
    """A chunk's execution rows as ``_execution_row`` states them, sealed from the columns.

    Each cell carries the row's members in schema order less the two
    identities, then the source open and raw volume (``None`` where no bar was
    observed); the source identity and the row identity are the canonical
    hashes ``_execution_row`` states, taken from the columns at once.
    """
    schema = _execution_schema()
    names = schema.names[:-2]
    fields = list(zip(*cells, strict=True))
    columns = [
        pa.array(list(column), type=schema.field(name).type)
        for name, column in zip(names, fields[: len(names)], strict=True)
    ]
    source = pa.Table.from_arrays(
        [
            columns[names.index("listing_id")],
            columns[names.index("intended_execution_session")],
            pa.array(list(fields[-2]), pa.float64()),
            pa.array(list(fields[-1]), pa.int64()),
            columns[names.index("execution_status")],
        ],
        names=["listing_id", "session_date", "open_split_adjusted", "volume_raw", "status"],
    )
    source_hashes = pa.array(canonical_row_hashes(source), pa.string())
    row_hashes = canonical_row_hashes(
        pa.Table.from_arrays([*columns, source_hashes], names=[*names, "source_row_hash"])
    )
    return pa.Table.from_arrays(
        [*columns, source_hashes, pa.array(row_hashes, pa.string())], schema=schema
    )


def decision_eligible_at_close(
    *,
    listing_id: str,
    formation_session: date,
    history: tuple[date, ...],
    bars: Mapping[tuple[date, str], RawDailyBar],
) -> bool:
    """Decision-only qualification; no future open or fill observation is read.

    The cell rule alone: the row this used to build also sealed the history
    identity and the row hash, which a yes/no question never reads, and the
    calibration owner asks it once per (formation, listing) over its whole
    history -- a hundred thousand rows per decision session on the QA book.
    """
    status, _reason, _adv20, _history_bars = _decision_cell(
        listing_id=listing_id, formation_session=formation_session, history=history, bars=bars
    )
    return status is DecisionTradabilityStatus.PLANNED_ORDER_ELIGIBLE


def _execution_cell(
    bar: RawDailyBar | None,
) -> tuple[ExecutionSessionStatus, ExecutionObservationMethod]:
    """The execution rule for one observed (or unobserved) entry session bar."""
    return execution_session_status(bar), (
        ExecutionObservationMethod.POST_SESSION_DAILY_BAR
        if bar is not None
        else ExecutionObservationMethod.POST_SESSION_DAILY_BAR_ABSENCE
    )


def _execution_row(
    *,
    listing_id: str,
    formation_session: date,
    intended_execution_session: date,
    bar: RawDailyBar | None,
) -> dict[str, object]:
    status, method = _execution_cell(bar)
    source = {
        "listing_id": listing_id,
        "session_date": intended_execution_session,
        "open_split_adjusted": None if bar is None else bar.open,
        "volume_raw": None if bar is None else bar.volume,
        "status": status.value,
    }
    values: dict[str, object] = {
        "formation_session": formation_session,
        "intended_execution_session": intended_execution_session,
        "listing_id": listing_id,
        "execution_status": status.value,
        "observation_method": method.value,
        "observed_through": intended_execution_session,
        "source_row_hash": canonical_hash(source),
    }
    values["row_hash"] = canonical_hash(values)
    return values


__all__ = [
    "HistoricalTradabilityBuilder",
    "PortfolioTradabilityMarketInputs",
    "PortfolioTradabilityRiskInputs",
    "PublishedTradabilitySurfaces",
    "TradabilityArtifactStore",
    "TradabilityBuildCancelled",
    "TradabilitySurfaceError",
]
