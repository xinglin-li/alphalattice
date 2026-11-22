"""Data-owned causal log-return surface for the Risk Desk."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast, overload

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor, ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.contracts import panel_source_manifest_revision
from alphalattice.foundation.feature_engine.panels.market_observation import (
    panel_market_observation,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.foundation.feature_engine.panels.semantic_index import (
    FeaturePanelSemanticIndexService,
)
from alphalattice.foundation.market_data_ops.publication.projection import project_research_series
from alphalattice.foundation.market_data_ops.returns.execution import open_to_open_log_return
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.investment.risk_research.contracts import (
    CausalRiskReturnChunk,
    CausalRiskReturnSurface,
    RiskReturnOmissions,
    RiskUniverseEpoch,
    seal_contract,
)
from alphalattice.kernel.identity import integrity_sha256
from alphalattice.kernel.shared_kernel.arrow_identity import (
    canonical_hash_with_row_lists,
    canonical_row_hashes,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = NDArray[np.float64]
_PREFIX = "playpen://data-operations/risk-returns/"
_SCHEMA_ID = "causal-risk-open-return"
_READ_WINDOW_CACHE_CAP_BYTES = 64 * 1024 * 1024


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("formation_session", pa.date32(), nullable=False),
            pa.field("return_start_session", pa.date32(), nullable=False),
            pa.field("listing_id", pa.string(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("open_total_return_log", pa.float64(), nullable=False),
            pa.field("entry_source_row_hash", pa.string(), nullable=False),
            pa.field("exit_source_row_hash", pa.string(), nullable=False),
            pa.field("action_set_hash", pa.string(), nullable=False),
            pa.field("row_hash", pa.string(), nullable=False),
        ]
    )


class RiskReturnSurfaceError(ValueError):
    """Stable Data-owned failure for the Risk return boundary."""


@dataclass(frozen=True, slots=True)
class RiskReturnClosure:
    """Compact proof that a return-surface manifest and all chunks reopened."""

    surface_hash: str
    chunk_count: int
    closure_hash: str


@dataclass(frozen=True, slots=True)
class RiskReturnWindow:
    """A complete return window and the listings excluded for recorded gaps."""

    ordered_listing_ids: tuple[str, ...]
    excluded_listing_ids: tuple[str, ...]
    returns: FloatArray


@dataclass(frozen=True, slots=True)
class _CachedReturnWindow:
    surface_hash: str
    formation_sessions: tuple[date, ...]
    matrix: FloatArray


class RiskReturnArtifactStore:
    """Publish immutable Parquet chunks and one content-addressed manifest."""

    def __init__(self, artifact_root: Path) -> None:
        """Bind Risk return publication and readback to the caller artifact root.

        Args:
            artifact_root: Caller root containing the data-operations/risk-returns store.
        """
        self.root = artifact_root.resolve() / "data-operations" / "risk-returns"

    @staticmethod
    def _atomic_write(target: Path, content: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        staged.write_bytes(content)
        os.replace(staged, target)
        staged.unlink(missing_ok=True)

    @staticmethod
    def _require_hash(value: str) -> None:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise RiskReturnSurfaceError("risk_research.return_artifact_identity_invalid")

    def _chunk_path(self, content_hash: str) -> Path:
        self._require_hash(content_hash)
        return self.root / "chunks" / f"{content_hash}.parquet"

    def _manifest_path(self, surface_hash: str) -> Path:
        self._require_hash(surface_hash)
        return self.root / "manifests" / f"{surface_hash}.json"

    @staticmethod
    def chunk_uri(content_hash: str) -> str:
        """Name the Risk return chunk URI for a declared content identity.

        Args:
            content_hash: Chunk content identity used in the URI.

        Returns:
            Identity-addressed return chunk URI.
        """
        return f"{_PREFIX}chunks/{content_hash}"

    @staticmethod
    def manifest_uri(surface_hash: str) -> str:
        """Name the Risk return manifest URI for a declared surface identity.

        Args:
            surface_hash: Surface identity used in the URI.

        Returns:
            Identity-addressed return manifest URI.
        """
        return f"{_PREFIX}manifests/{surface_hash}"

    def publish_chunk(self, table: pa.Table) -> CausalRiskReturnChunk:
        """Publish sorted causal-return rows as an identity-addressed Parquet chunk.

        Identity binds schema and ordered formation/listing/row-hash tuples. Missing files are
        written through a staged replacement; existing files are retained and checked against the
        chunk metadata commitment.

        Args:
            table: Nonempty return rows sorted by formation and listing before publication.

        Returns:
            Year, coverage, content/metadata identities and URI of the published chunk.

        Raises:
            RiskReturnSurfaceError: Rows are empty or existing chunk metadata fails verification.
        """
        ordered = table.combine_chunks().sort_by(
            [("formation_session", "ascending"), ("listing_id", "ascending")]
        )
        if ordered.num_rows == 0:
            raise RiskReturnSurfaceError("risk_research.return_chunk_empty")
        identity_rows = tuple(
            (str(row[0]), str(row[1]), str(row[2]))
            for row in zip(
                ordered.column("formation_session").to_pylist(),
                ordered.column("listing_id").to_pylist(),
                ordered.column("row_hash").to_pylist(),
                strict=True,
            )
        )
        content_hash = canonical_hash({"schema": _SCHEMA_ID, "rows": identity_rows})
        metadata = {
            b"alphalattice.snapshot_kind": b"CausalRiskReturnChunk",
            b"alphalattice.chunk_hash": content_hash.encode(),
        }
        target = self._chunk_path(content_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        if not target.exists():
            pq.write_table(ordered.replace_schema_metadata(metadata), staged, compression="zstd")
            os.replace(staged, target)
        staged.unlink(missing_ok=True)
        parquet = pq.ParquetFile(target)
        actual = parquet.schema_arrow.metadata or {}
        if actual.get(b"alphalattice.chunk_hash") != content_hash.encode():
            raise RiskReturnSurfaceError("risk_research.return_chunk_tampered")
        serialized_metadata = json.dumps(
            {key.decode(): value.decode() for key, value in sorted(actual.items())},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        sessions = ordered.column("formation_session").to_pylist()
        return CausalRiskReturnChunk(
            year=sessions[0].year,
            row_count=ordered.num_rows,
            first_formation_session=sessions[0],
            last_formation_session=sessions[-1],
            content_hash=content_hash,
            metadata_hash=integrity_sha256(serialized_metadata).hexdigest(),
            uri=self.chunk_uri(content_hash),
        )

    def publish_manifest(self, surface: CausalRiskReturnSurface) -> ArtifactDescriptor:
        """Publish a sealed return manifest with exact-byte replay admission.

        Args:
            surface: Validated causal-return surface to serialize.

        Returns:
            Artifact descriptor binding surface identity, serialized metadata digest and URI.

        Raises:
            RiskReturnSurfaceError: An existing manifest at that identity contains different bytes.
        """
        content = json.dumps(
            surface.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        target = self._manifest_path(surface.surface_hash)
        if target.exists() and target.read_bytes() != content:
            raise RiskReturnSurfaceError("risk_research.return_surface_identity_reused")
        if not target.exists():
            self._atomic_write(target, content)
        return ArtifactDescriptor(
            kind="causal-risk-return-surface",
            content_hash=surface.surface_hash,
            metadata_hash=integrity_sha256(content).hexdigest(),
            uri=self.manifest_uri(surface.surface_hash),
        )

    def load_manifest(self, surface_hash: str) -> CausalRiskReturnSurface:
        """Load and validate the return manifest stored under a requested surface identity.

        Args:
            surface_hash: Identity selecting the stored manifest filename.

        Returns:
            Model-validated causal-return surface from the stored JSON.

        Raises:
            FileNotFoundError: The selected manifest is absent.
            pydantic.ValidationError: The stored manifest violates the surface contract.
        """
        target = self._manifest_path(surface_hash)
        if not target.is_file():
            raise FileNotFoundError("Risk return surface is missing")
        return cast(
            CausalRiskReturnSurface,
            CausalRiskReturnSurface.model_validate_json(target.read_bytes()),
        )

    def verify_closure(self, surface_hash: str) -> RiskReturnClosure:
        """Open the manifest and validate every immutable Parquet child."""
        surface = self._closure_surface(surface_hash)
        for chunk in surface.chunks:
            self.resolve_chunk(chunk)
        return self._closure(surface)

    def _closure_surface(self, surface_hash: str) -> CausalRiskReturnSurface:
        surface = self.load_manifest(surface_hash)
        if surface.surface_hash != surface_hash:
            raise RiskReturnSurfaceError("risk_research.return_surface_identity_mismatch")
        return surface

    @staticmethod
    def _closure(surface: CausalRiskReturnSurface) -> RiskReturnClosure:
        identity = {
            "kind": "RiskReturnClosure",
            "surface_hash": surface.surface_hash,
            "chunks": tuple(chunk.content_hash for chunk in surface.chunks),
        }
        return RiskReturnClosure(
            surface_hash=surface.surface_hash,
            chunk_count=len(surface.chunks),
            closure_hash=canonical_hash(identity),
        )

    def find_manifest(
        self, *, panel_snapshot_hash: str, source_watermark_hash: str
    ) -> CausalRiskReturnSurface | None:
        """Find the first stored manifest matching both Panel and source watermark.

        Args:
            panel_snapshot_hash: Required source Panel identity.
            source_watermark_hash: Required market-source watermark identity.

        Returns:
            First match in sorted manifest path order, or None when no manifest matches.
        """
        root = self.root / "manifests"
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            surface = self.load_manifest(path.stem)
            if (
                surface.epoch.panel_snapshot_hash == panel_snapshot_hash
                and surface.source_watermark_hash == source_watermark_hash
            ):
                return surface
        return None

    def resolve_chunk(self, chunk: CausalRiskReturnChunk) -> Path:
        """Verify chunk URI, metadata, row content identities and declared coverage.

        Args:
            chunk: Declared yearly return chunk to resolve and verify.

        Returns:
            Local Parquet path after metadata, reconstructed row/chunk hashes and coverage checks.

        Raises:
            RiskReturnSurfaceError: URI, schema, metadata, row identity, session type/count/bounds
                or year differs from the declaration.
        """
        if chunk.uri != self.chunk_uri(chunk.content_hash):
            raise RiskReturnSurfaceError("risk_research.return_chunk_uri_invalid")
        target = self._chunk_path(chunk.content_hash)
        try:
            table = pq.read_table(target, columns=_schema().names).combine_chunks()
        except Exception as error:
            raise RiskReturnSurfaceError("risk_research.return_chunk_tampered") from error
        metadata = table.schema.metadata or {}
        if metadata.get(b"alphalattice.chunk_hash") != chunk.content_hash.encode():
            raise RiskReturnSurfaceError("risk_research.return_chunk_tampered")
        serialized = json.dumps(
            {key.decode(): value.decode() for key, value in sorted(metadata.items())},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if integrity_sha256(serialized).hexdigest() != chunk.metadata_hash:
            raise RiskReturnSurfaceError("risk_research.return_chunk_metadata_tampered")
        ordered = table.sort_by([("formation_session", "ascending"), ("listing_id", "ascending")])
        if ordered.num_rows == 0 or ordered.num_rows != chunk.row_count:
            raise RiskReturnSurfaceError("risk_research.return_chunk_tampered")
        # Every row's identity re-derived from its cells -- the hash the
        # publisher sealed, computed from the columns rather than from one
        # Python object per row -- then the chunk's identity from the sequence
        # of (session, listing, row hash) the publisher hashed.
        recorded = ordered.column("row_hash")
        if canonical_row_hashes(ordered.drop_columns(["row_hash"])) != recorded.to_pylist():
            raise RiskReturnSurfaceError("risk_research.return_chunk_tampered")
        sessions = ordered.column("formation_session")
        if not pa.types.is_date(sessions.type) or sessions.null_count:
            raise RiskReturnSurfaceError("risk_research.return_chunk_tampered")
        identity = pa.table(
            {
                "formation_session": sessions,
                "listing_id": ordered.column("listing_id"),
                "row_hash": recorded,
            }
        )
        if (
            canonical_hash_with_row_lists({"schema": _SCHEMA_ID}, key="rows", table=identity)
            != chunk.content_hash
            or sessions[0].as_py() != chunk.first_formation_session
            or sessions[-1].as_py() != chunk.last_formation_session
            or not pc.all(pc.equal(pc.year(sessions), chunk.year)).as_py()
        ):
            raise RiskReturnSurfaceError("risk_research.return_chunk_tampered")
        return target


def derive_risk_return_rows(
    *,
    manifest: UniverseManifest,
    listing_id: str,
    symbol: str,
    bars: Sequence[RawDailyBar],
    actions: Sequence[CorporateActionEvent],
    required_sessions: tuple[date, ...],
    traded_sessions: Sequence[date] | None = None,
) -> tuple[dict[str, object], ...]:
    """Derive adjacent-session log gross returns from split-adjusted opens and dividends.

    Args:
        manifest: Universe manifest declaring split-adjusted daily price basis.
        listing_id: Listing identity attached to each return row.
        symbol: Display symbol retained in each sealed row.
        bars: Raw daily bars projected through the declared price basis.
        actions: Corporate actions used by the source projection.
        required_sessions: Ordered source sessions whose adjacent pairs form return rows.
        traded_sessions: Sessions the Panel says traded; None retains strict complete coverage.

    Returns:
        Sealed row mappings with entry/exit/action source identities and canonical row_hash.

    Raises:
        RiskReturnSurfaceError: Price basis is unsupported, projected sessions repeat or a required
            source session is absent.
    """
    if manifest.profile.daily_price_basis != "split_adjusted":
        raise RiskReturnSurfaceError("risk_research.return_price_basis_unsupported")
    projected = project_research_series(
        bars,
        actions,
        daily_price_basis=manifest.profile.daily_price_basis,
    )
    by_session = {row.session_date: row for row in projected}
    if len(by_session) != len(projected):
        raise RiskReturnSurfaceError("risk_research.return_source_duplicate")
    rows: list[dict[str, object]] = []
    traded = None if traded_sessions is None else set(traded_sessions)
    for previous_session, formation_session in pairwise(required_sessions):
        previous = by_session.get(previous_session)
        current = by_session.get(formation_session)
        if previous is None or current is None:
            if (
                traded is None
                or (previous is None and previous_session in traded)
                or (current is None and formation_session in traded)
            ):
                raise RiskReturnSurfaceError("risk_research.return_source_session_missing")
            continue
        value = open_to_open_log_return(
            entry_open=previous.open_split_adjusted,
            exit_open=current.open_split_adjusted,
            period_dividend=current.cash_dividend,
        )
        entry_hash = canonical_hash(
            {
                "listing_id": listing_id,
                "session": previous_session,
                "open_split_adjusted": previous.open_split_adjusted,
                "action_set_hash": previous.action_set_hash,
            }
        )
        exit_hash = canonical_hash(
            {
                "listing_id": listing_id,
                "session": formation_session,
                "open_split_adjusted": current.open_split_adjusted,
                "period_dividend": current.cash_dividend,
                "action_set_hash": current.action_set_hash,
            }
        )
        identity: dict[str, object] = {
            "formation_session": formation_session,
            "return_start_session": previous_session,
            "listing_id": listing_id,
            "symbol": symbol,
            "open_total_return_log": value,
            "entry_source_row_hash": entry_hash,
            "exit_source_row_hash": exit_hash,
            "action_set_hash": current.action_set_hash,
        }
        rows.append({**identity, "row_hash": canonical_hash(identity)})
    return tuple(rows)


def listing_sources(
    store: Any, listings: Sequence[Any], sessions: tuple[date, ...]
) -> dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]]:
    """Each listing's bars over `sessions` and its actions within them, from the store's bars and
    actions of every listing read at once."""
    ids = [listing.listing_id for listing in listings]
    bars = store.raw_bars_by_listing(ids, start=sessions[0], through=sessions[-1])
    actions = store.actions_by_listing(ids)
    return {
        listing_id: (
            bars[listing_id],
            tuple(
                v for v in actions[listing_id] if sessions[0] <= v.effective_date <= sessions[-1]
            ),
        )
        for listing_id in ids
    }


def listing_returns(
    manifest: UniverseManifest,
    listing: Any,
    sessions: tuple[date, ...],
    sources: Mapping[str, tuple[tuple[Any, ...], tuple[Any, ...]]],
    traded_sessions: Sequence[date] | None = None,
) -> tuple[dict[str, object], ...]:
    """One listing's causal returns over `sessions`, from its bars and actions in `sources`."""
    bars, actions = sources[listing.listing_id]
    return derive_risk_return_rows(
        manifest=manifest,
        listing_id=listing.listing_id,
        symbol=listing.symbol,
        bars=bars,
        actions=actions,
        required_sessions=sessions,
        traded_sessions=traded_sessions,
    )


class CausalRiskReturnSurfacePublisher:
    """Compile current-universe causal Risk returns from the clean market store."""

    def __init__(
        self,
        *,
        store: MarketDataRepository,
        resolver: ArtifactResolver,
        artifact_root: Path,
        mutation_gate: WorkspaceMutationGate,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Bind causal-return publication to source readers, artifact resolver and mutation gate.

        Args:
            store: Market-data repository providing revision-bound source reads.
            resolver: Resolver for the source Panel manifest and semantic index.
            artifact_root: Caller-owned destination for Risk return artifacts.
            mutation_gate: Workspace publication mutation owner.
            progress: Optional textual source-read progress callback.
        """
        self.store = store
        self.resolver = resolver
        self.artifacts = RiskReturnArtifactStore(artifact_root)
        self.index_service = FeaturePanelSemanticIndexService(resolver)
        self.mutation_gate = mutation_gate
        self.progress = progress or (lambda _message: None)

    def publish(
        self,
        *,
        panel_manifest_ref: str,
        market_profile_id: str,
        required_sessions: tuple[date, ...] | None = None,
        required_listing_ids: tuple[str, ...] | None = None,
    ) -> tuple[CausalRiskReturnSurface, ArtifactDescriptor]:
        """Publish complete scoped causal returns against an exact Panel and market revision.

        The publisher requires at least 316 ordered source sessions, yielding at least 315 returns.
        Optional listings preserve the Panel subsequence order; optional sessions form a complete
        index interval. Read-only source access precedes gated chunk/manifest publication and
        manifest readback.

        Args:
            panel_manifest_ref: Source Panel manifest reference.
            market_profile_id: Required source market profile.
            required_sessions: Optional complete ordered source-session interval.
            required_listing_ids: Optional nonempty ordered subsequence of the Panel listing axis.

        Returns:
            Sealed causal-return surface and its published artifact descriptor.

        Raises:
            RiskReturnSurfaceError: Profile, listing/session scope, source returns or publication
                readback violates the contract.
        """
        panel = self.resolver.load_feature_panel_manifest(panel_manifest_ref)
        index, _index_ref, _created = self.index_service.obtain(panel_manifest_ref)
        manifest = self.store.load_universe_manifest_revision(panel_source_manifest_revision(panel))
        if manifest.profile.market_profile_id != market_profile_id:
            raise RiskReturnSurfaceError("risk_research.return_universe_unavailable")
        listing_ids = FeaturePanelReader(self.resolver).listing_ids(panel_manifest_ref)
        absent_sessions = self._observed_absences(manifest, panel, listing_ids)
        if required_listing_ids is not None:
            requested_listings = set(required_listing_ids)
            if (
                not required_listing_ids
                or tuple(listing for listing in listing_ids if listing in requested_listings)
                != required_listing_ids
            ):
                raise RiskReturnSurfaceError("risk_research.return_listing_scope_invalid")
            listing_ids = required_listing_ids
        listings = self.store.listing_scope(manifest, listing_ids=listing_ids)
        sessions = tuple(item.session_date for item in index.sessions)
        if required_sessions is not None:
            if not required_sessions or required_sessions != tuple(sorted(set(required_sessions))):
                raise RiskReturnSurfaceError("risk_research.return_session_axis_invalid")
            scoped = tuple(
                day for day in sessions if required_sessions[0] <= day <= required_sessions[-1]
            )
            if scoped != required_sessions:
                raise RiskReturnSurfaceError("risk_research.return_session_axis_invalid")
            sessions = scoped
        if len(sessions) < 316 or sessions != tuple(sorted(set(sessions))):
            raise RiskReturnSurfaceError("risk_research.return_session_axis_invalid")
        epoch = seal_contract(
            RiskUniverseEpoch,
            "epoch_hash",
            market_profile_id=market_profile_id,
            universe_manifest_revision=manifest.revision_sha256,
            panel_snapshot_hash=str(panel["snapshot_hash"]),
            ordered_listing_ids=listing_ids,
        )
        watermark = self.store.execution_source_watermark(
            manifest, through=sessions[-1], listing_ids=listing_ids
        )
        source_watermark_hash = str(watermark["watermark_hash"])
        rows_by_year: dict[int, list[dict[str, object]]] = {}
        sources = listing_sources(self.store, listings, sessions)
        omissions: list[RiskReturnOmissions] = []
        for index_value, listing in enumerate(listings, start=1):
            absent = set(absent_sessions.get(listing.listing_id, ()))
            listing_rows = listing_returns(
                manifest,
                listing,
                sessions,
                sources,
                tuple(day for day in sessions if day not in absent) if absent else None,
            )
            absent_formations = {row["formation_session"] for row in listing_rows}
            pairs = tuple(
                (start, end) for start, end in pairwise(sessions) if end not in absent_formations
            )
            if pairs:
                omissions.append(
                    RiskReturnOmissions(
                        listing_id=listing.listing_id, count=len(pairs), pairs=pairs
                    )
                )
            for row in listing_rows:
                formation = row["formation_session"]
                if not isinstance(formation, date):
                    raise RiskReturnSurfaceError("risk_research.return_formation_session_invalid")
                rows_by_year.setdefault(formation.year, []).append(row)
            self.progress(f"risk_return_source {index_value}/{len(listing_ids)}")
        omissions.sort(key=lambda item: listing_ids.index(item.listing_id))
        chunks: list[CausalRiskReturnChunk] = []
        with self.mutation_gate.try_hold(timeout_seconds=30.0):
            for _year, rows in sorted(rows_by_year.items()):
                chunks.append(
                    self.artifacts.publish_chunk(pa.Table.from_pylist(rows, schema=_schema()))
                )
            surface = seal_contract(
                CausalRiskReturnSurface,
                "surface_hash",
                epoch=epoch,
                first_formation_session=sessions[1],
                last_formation_session=sessions[-1],
                formation_count=len(sessions) - 1,
                source_watermark_hash=source_watermark_hash,
                chunks=tuple(chunks),
                limitations=(
                    (
                        "Historical return coverage is not formation eligibility; "
                        "use the paired Panel's dated members. Initial backfill is non-PIT."
                        if isinstance(panel.get("safe_summary"), dict)
                        and panel["safe_summary"].get("membership")
                        else "Current-universe research surface; membership is not point-in-time."
                    ),
                    "Valid market tail returns are retained without winsorization.",
                ),
                **({"omissions": tuple(omissions)} if omissions else {}),
            )
            descriptor = self.artifacts.publish_manifest(surface)
        if self.artifacts.load_manifest(surface.surface_hash) != surface:
            raise RiskReturnSurfaceError("risk_research.return_surface_readback_failed")
        return surface, descriptor

    def _observed_absences(
        self, manifest: UniverseManifest, panel: Mapping[str, object], listing_ids: tuple[str, ...]
    ) -> Mapping[str, tuple[date, ...]]:
        """The sessions the Panel recorded its listings had no bar for, while the source holds them.

        The record is the Panel's own, bound by its snapshot hash; it is evidence only while the
        market source still holds what the Panel read, so its watermark is recomputed over the
        Panel's whole listing axis through the recorded session. A Panel without a record has no
        absence, and every missing bar stays refused.
        """
        try:
            observation = panel_market_observation(panel)
        except ValueError as error:
            raise RiskReturnSurfaceError(
                "risk_research.return_market_observation_unverified"
            ) from error
        if observation is None:
            return {}
        current = self.store.execution_source_watermark(
            manifest, through=observation.through, listing_ids=listing_ids
        )
        if current["watermark_hash"] != observation.source_watermark_hash:
            raise RiskReturnSurfaceError("risk_research.return_market_observation_unverified")
        return observation.absent_sessions


class CausalRiskReturnReader:
    """Read only bounded formation slices into a canonical dense matrix."""

    def __init__(self, artifact_root: Path) -> None:
        """Bind verified causal-return reads and bounded caches to an artifact root.

        Args:
            artifact_root: Caller root containing the persisted causal-return store.
        """
        self.artifacts = RiskReturnArtifactStore(artifact_root)
        self._verified_chunks: dict[str, tuple[CausalRiskReturnChunk, Path]] = {}
        self._available_cache: tuple[str, tuple[date, ...]] | None = None
        self._window_cache: _CachedReturnWindow | None = None

    def verify_closure(self, surface_hash: str) -> RiskReturnClosure:
        """The store's closure verification, through this reader's once-per-chunk proof.

        The execution that publishes or selects a surface proves its closure
        and then reads its windows; proved through the store and again through
        the reader, every chunk was verified twice per run. Proved here, the
        reads that follow reuse the same proof for the life of this reader --
        one execution -- and a new reader proves again.
        """
        surface = self.artifacts._closure_surface(surface_hash)
        for chunk in surface.chunks:
            self._resolve_chunk(chunk)
        return self.artifacts._closure(surface)

    def _resolve_chunk(self, chunk: CausalRiskReturnChunk) -> Path:
        # Return chunks are immutable content-addressed artifacts. One reader
        # certifies each chunk once, then reuses that result for its execution.
        cached = self._verified_chunks.get(chunk.content_hash)
        if cached is not None:
            cached_chunk, path = cached
            if cached_chunk != chunk or not path.is_file():
                raise RiskReturnSurfaceError("risk_research.return_chunk_identity_conflict")
            return path
        path = self.artifacts.resolve_chunk(chunk)
        self._verified_chunks[chunk.content_hash] = (chunk, path)
        return path

    def available_sessions(self, surface: CausalRiskReturnSurface) -> tuple[date, ...]:
        """Read the sorted unique formation axis from verified chunks and cache by surface identity.

        Args:
            surface: Sealed return surface whose published coverage is being read.

        Returns:
            Sorted unique formation sessions across the surface chunks.

        Raises:
            RiskReturnSurfaceError: Verified chunk sessions do not match the declared formation
                count.
        """
        if self._available_cache is not None and self._available_cache[0] == surface.surface_hash:
            return self._available_cache[1]
        sessions: set[date] = set()
        for chunk in surface.chunks:
            table = pq.read_table(self._resolve_chunk(chunk), columns=["formation_session"])
            sessions.update(table.column("formation_session").to_pylist())
        sessions.update(end for item in surface.omissions or () for _start, end in item.pairs)
        ordered = tuple(sorted(sessions))
        if len(ordered) != surface.formation_count:
            raise RiskReturnSurfaceError("risk_research.return_session_count_invalid")
        self._available_cache = (surface.surface_hash, ordered)
        return ordered

    @overload
    def read_sessions(
        self,
        surface: CausalRiskReturnSurface,
        formation_sessions: tuple[date, ...],
        *,
        _mode: Literal["matrix"] = "matrix",
    ) -> FloatArray: ...

    @overload
    def read_sessions(
        self,
        surface: CausalRiskReturnSurface,
        formation_sessions: tuple[date, ...],
        *,
        _mode: Literal["window"],
    ) -> RiskReturnWindow: ...

    @overload
    def read_sessions(
        self,
        surface: CausalRiskReturnSurface,
        formation_sessions: tuple[date, ...],
        *,
        _mode: Literal["prepare"],
    ) -> None: ...

    def read_sessions(
        self,
        surface: CausalRiskReturnSurface,
        formation_sessions: tuple[date, ...],
        *,
        _mode: Literal["matrix", "window", "prepare"] = "matrix",
    ) -> FloatArray | RiskReturnWindow | None:
        """Read verified returns; finite subsets are exposed through read_window."""
        if not formation_sessions or formation_sessions != tuple(sorted(set(formation_sessions))):
            raise RiskReturnSurfaceError("risk_research.return_read_axis_invalid")
        matrix: FloatArray | None = None
        cached = self._window_cache
        if cached is not None and cached.surface_hash == surface.surface_hash:
            try:
                start = cached.formation_sessions.index(formation_sessions[0])
            except ValueError:
                start = -1
            stop = start + len(formation_sessions)
            if start >= 0 and cached.formation_sessions[start:stop] == formation_sessions:
                matrix = cached.matrix[start:stop]
                matrix.setflags(write=False)
        if matrix is None:
            requested = set(formation_sessions)
            chunks = tuple(
                chunk
                for chunk in surface.chunks
                if chunk.first_formation_session <= formation_sessions[-1]
                and chunk.last_formation_session >= formation_sessions[0]
            )
            tables = tuple(
                pq.read_table(
                    self._resolve_chunk(chunk),
                    columns=["formation_session", "listing_id", "open_total_return_log"],
                    filters=[("formation_session", "in", list(formation_sessions))],
                )
                for chunk in chunks
            )
            if not tables:
                raise RiskReturnSurfaceError("risk_research.return_read_empty")
            table = pa.concat_tables(tables).combine_chunks()
            sessions = table.column("formation_session").to_pylist()
            listings = table.column("listing_id").to_pylist()
            values = np.asarray(
                pc.cast(table.column("open_total_return_log"), pa.float64()).to_numpy(
                    zero_copy_only=False
                ),
                dtype=np.float64,
            )
            session_positions = {value: index for index, value in enumerate(formation_sessions)}
            listing_positions = {
                value: index for index, value in enumerate(surface.epoch.ordered_listing_ids)
            }
            matrix = np.full(
                (len(formation_sessions), len(listing_positions)), np.nan, dtype=np.float64
            )
            seen: set[tuple[int, int]] = set()
            for row_index, (session, listing_id) in enumerate(zip(sessions, listings, strict=True)):
                if session not in requested or listing_id not in listing_positions:
                    raise RiskReturnSurfaceError("risk_research.return_read_axis_mismatch")
                key = (session_positions[session], listing_positions[listing_id])
                if key in seen:
                    raise RiskReturnSurfaceError("risk_research.return_read_duplicate")
                seen.add(key)
                matrix[key] = values[row_index]
            declared = {
                (session_positions[end], listing_positions[item.listing_id])
                for item in surface.omissions or ()
                for _start, end in item.pairs
                if end in session_positions
            }
            if (
                matrix.size - len(seen) != len(declared)
                or any(key in seen for key in declared)
                or not np.isfinite(values).all()
            ):
                raise RiskReturnSurfaceError("risk_research.return_read_incomplete")
            matrix.setflags(write=False)
            if matrix.nbytes <= _READ_WINDOW_CACHE_CAP_BYTES:
                self._window_cache = _CachedReturnWindow(
                    surface_hash=surface.surface_hash,
                    formation_sessions=formation_sessions,
                    matrix=matrix,
                )
        if _mode == "prepare":
            return None
        if _mode == "window":
            requested = set(formation_sessions)
            excluded = {
                item.listing_id
                for item in surface.omissions or ()
                if any(end in requested for _start, end in item.pairs)
            }
            axis = surface.epoch.ordered_listing_ids
            selected = tuple(item for item in axis if item not in excluded)
            if not selected:
                raise RiskReturnSurfaceError("risk_research.return_read_empty")
            values = matrix[:, [index for index, item in enumerate(axis) if item not in excluded]]
            if not np.isfinite(values).all():
                raise RiskReturnSurfaceError("risk_research.return_read_incomplete")
            values.setflags(write=False)
            return RiskReturnWindow(
                selected, tuple(item for item in axis if item in excluded), values
            )
        if not np.isfinite(matrix).all():
            raise RiskReturnSurfaceError("risk_research.return_read_incomplete")
        return matrix

    def read_window(
        self, surface: CausalRiskReturnSurface, formation_sessions: tuple[date, ...]
    ) -> RiskReturnWindow:
        """Read a finite window, excluding listings with recorded gaps in it."""
        return self.read_sessions(surface, formation_sessions, _mode="window")

    def prepare_windows(
        self, surface: CausalRiskReturnSurface, formation_sessions: tuple[date, ...]
    ) -> None:
        """Verify and retain the bounded source for subsequent finite window reads."""
        self.read_sessions(surface, formation_sessions, _mode="prepare")


__all__ = [
    "CausalRiskReturnReader",
    "CausalRiskReturnSurfacePublisher",
    "RiskReturnArtifactStore",
    "RiskReturnClosure",
    "RiskReturnSurfaceError",
    "RiskReturnWindow",
    "derive_risk_return_rows",
]
