"""Governed streaming reader for immutable Feature Panel snapshots."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.contracts import canonical_hash
from alphalattice.foundation.feature_engine.panels.identity import (
    panel_chunk_hash,
    panel_rows_match_their_hashes,
    panel_schema_hash,
)

_VALUES_VERIFIED: set[str] = set()
"""The byte digests of the chunk files whose values this process has checked: a
chunk is read many times in one study, and its values need checking once per content."""


@dataclass(frozen=True)
class FeaturePanelReadRequest:
    """Select factor columns and sessions with bounded Arrow scanning.

    Attributes:
        manifest_ref: Immutable snapshot manifest to admit before reading.
        start_session: Inclusive first session.
        end_session: Inclusive last session.
        factor_columns: Nonempty, sorted, unique factor names.
        exact_sessions: Optional sorted subset within the session range.
        include_row_hash: Include each row's content digest in the output.
        batch_size: Maximum rows per batch.
        batch_readahead: Number of batches Arrow may prefetch.
        fragment_readahead: Number of fragments Arrow may prefetch.
        use_threads: Allow Arrow to scan using its worker threads.
    """

    manifest_ref: str
    start_session: date
    end_session: date
    factor_columns: tuple[str, ...]
    exact_sessions: tuple[date, ...] | None = None
    include_row_hash: bool = False
    batch_size: int = 65_536
    batch_readahead: int = 4
    fragment_readahead: int = 2
    use_threads: bool = True

    def __post_init__(self) -> None:
        """Refuse reversed ranges, invalid columns, or invalid scan bounds.

        Raises:
            ValueError: The range, columns, session subset, or scan bounds are invalid.
        """
        if self.start_session > self.end_session:
            raise ValueError("feature panel read range is reversed")
        if not self.factor_columns or self.factor_columns != tuple(
            sorted(set(self.factor_columns))
        ):
            raise ValueError("factor columns must be non-empty, sorted, and unique")
        if self.batch_size < 1:
            raise ValueError("feature panel batch size must be positive")
        if self.batch_readahead < 0 or self.fragment_readahead < 0:
            raise ValueError("feature panel readahead cannot be negative")
        if self.exact_sessions is not None:
            if not self.exact_sessions or self.exact_sessions != tuple(
                sorted(set(self.exact_sessions))
            ):
                raise ValueError("exact sessions must be non-empty, sorted, and unique")
            if (
                self.exact_sessions[0] < self.start_session
                or self.exact_sessions[-1] > self.end_session
            ):
                raise ValueError("exact sessions are outside the requested panel range")


class FeaturePanelReader:
    """Validate a frozen manifest, then yield bounded Arrow RecordBatches.

    The reader never connects to DuckDB and never materializes the complete
    panel with ``to_table()``.  Each worker may construct its own reader over
    the immutable chunks.
    """

    _IDENTITY_COLUMNS = ("session_date", "listing_id")

    def __init__(self, resolver: ArtifactResolver) -> None:
        """Use the workspace's resolver to verify snapshot artifacts.

        Args:
            resolver: Resolver owning manifest, lifecycle, and chunk access.
        """
        self.resolver = resolver

    def batches(self, request: FeaturePanelReadRequest) -> Iterator[pa.RecordBatch]:
        """Yield verified rows for the requested factor and session selection.

        Args:
            request: Snapshot, columns, sessions, and bounded scan settings.

        Yields:
            Batches containing session and listing keys, selected factors, and
            the row digest when requested.

        Raises:
            ValueError: Admission, snapshot identity, chunk integrity, or the
                requested range or columns fail verification.
        """
        manifest = self.admitted_manifest(request.manifest_ref, consumer="Factor Research")
        self._validate_request_identity(manifest, request)
        chunk_paths = self._qualified_chunk_paths(manifest, request)
        if not chunk_paths:
            raise ValueError("feature panel read range has no snapshot chunks")
        dataset = ds.dataset(chunk_paths, format="parquet")
        columns = [
            *self._IDENTITY_COLUMNS,
            *request.factor_columns,
            *(["row_hash"] if request.include_row_hash else []),
        ]
        missing = sorted(set(columns) - set(dataset.schema.names))
        if missing:
            raise ValueError(f"feature panel request contains unknown columns: {missing}")
        predicate = (ds.field("session_date") >= pa.scalar(request.start_session)) & (
            ds.field("session_date") <= pa.scalar(request.end_session)
        )
        if request.exact_sessions is not None:
            predicate = predicate & ds.field("session_date").isin(request.exact_sessions)
        scanner = dataset.scanner(
            columns=columns,
            filter=predicate,
            batch_size=request.batch_size,
            batch_readahead=request.batch_readahead,
            fragment_readahead=request.fragment_readahead,
            use_threads=request.use_threads,
        )
        yield from scanner.to_batches()

    def available_sessions(self, manifest_ref: str) -> tuple[date, ...]:
        """Read the immutable panel calendar without consulting mutable authority.

        Args:
            manifest_ref: Immutable snapshot whose admitted calendar is requested.

        Returns:
            Sorted sessions spanning the manifest's full history range.

        Raises:
            ValueError: Admission, catalog, chunk integrity, or calendar coverage fails.
        """
        manifest = self.admitted_manifest(manifest_ref, consumer="Factor Research calendar")
        summary = manifest.get("safe_summary")
        if not isinstance(summary, dict):
            raise ValueError("feature panel snapshot has no safe summary")
        factors = summary.get("factor_catalog_summary")
        if not isinstance(factors, dict) or not factors:
            raise ValueError("feature panel snapshot has no factor catalog")
        start = date.fromisoformat(str(manifest["history_start"]))
        end = date.fromisoformat(str(manifest["as_of_session"]))
        request = FeaturePanelReadRequest(
            manifest_ref=manifest_ref,
            start_session=start,
            end_session=end,
            factor_columns=(min(str(value) for value in factors),),
            batch_size=65_536,
            batch_readahead=0,
            fragment_readahead=0,
        )
        observed: set[date] = set()
        for batch in self.batches(request):
            sessions = batch.column(batch.schema.get_field_index("session_date"))
            observed.update(sessions.unique().to_pylist())
        ordered = tuple(sorted(observed))
        if not ordered or ordered[0] != start or ordered[-1] != end:
            raise ValueError("feature panel calendar does not cover its manifest range")
        return ordered

    def listing_ids(self, manifest_ref: str) -> tuple[str, ...]:
        """The verified historical coverage axis, including members that later left.

        Args:
            manifest_ref: Immutable snapshot whose historical listing coverage is requested.

        Returns:
            Sorted unique listings matching the manifest's listing-set digest.

        Raises:
            ValueError: Snapshot admission or the listing-axis digest fails verification.
        """
        observed: set[str] = set()
        for batch in self.identity_batches(manifest_ref, include_row_hash=False):
            observed.update(
                batch.column(batch.schema.get_field_index("listing_id")).unique().to_pylist()
            )
        result = tuple(sorted(observed))
        manifest = self.resolver.load_feature_panel_manifest(manifest_ref)
        if not result or canonical_hash(result) != manifest.get("listing_set_hash"):
            raise ValueError("feature_panel.listing_axis_mismatch")
        return result

    def listing_axis(self, manifest_ref: str) -> tuple[str, ...]:
        """The listing axis of an admitted snapshot, verified against ``listing_set_hash``.

        The same admission every research read passes -- physically
        available bytes, an ACTIVE lifecycle, a consistent temporal identity
        and a Feature Input Gateway admission -- and, per chunk the manifest
        names, the chunk's content-addressed descriptor, its schema and its
        row count. Only the ``listing_id`` column is read, and the manifest's
        ``listing_set_hash`` must be the hash of exactly that axis. The rows'
        logical content (their hashes) is what a reader of row values
        verifies before consuming them (``batches``, ``identity_batches``);
        this axis binds no row value, so it does not re-read them.

        Args:
            manifest_ref: Immutable snapshot to admit and verify.

        Returns:
            Sorted unique listings from verified chunk descriptors.

        Raises:
            ValueError: Admission, temporal identity, chunk schema or row count,
                or the listing-axis digest is invalid.
        """
        manifest = self.admitted_manifest(manifest_ref, consumer="Panel coverage")
        self._validate_snapshot_identity(manifest)
        raw_chunks = manifest.get("chunks")
        if not isinstance(raw_chunks, list):
            raise ValueError("feature panel manifest has no chunk catalog")
        schema_hash = str(manifest.get("schema_hash", ""))
        observed: set[str] = set()
        for item in raw_chunks:
            if not isinstance(item, dict):
                raise ValueError("feature panel chunk entry is invalid")
            path = self.resolver.resolve_feature_panel_chunk_ref(
                uri=str(item["uri"]),
                content_hash=str(item["chunk_hash"]),
                metadata_hash=str(item["metadata_hash"]),
            )
            parquet = pq.ParquetFile(path)
            if panel_schema_hash(parquet.schema_arrow) != schema_hash:
                raise ValueError("feature panel chunk schema hash mismatch")
            if parquet.metadata.num_rows != int(item["row_count"]):
                raise ValueError("feature panel chunk row count mismatch")
            column = parquet.read(columns=["listing_id"]).column("listing_id")
            observed.update(str(value) for value in column.unique().to_pylist())
        result = tuple(sorted(observed))
        if not result or canonical_hash(result) != manifest.get("listing_set_hash"):
            raise ValueError("feature_panel.listing_axis_mismatch")
        return result

    def admitted_manifest(self, manifest_ref: str, *, consumer: str) -> dict[str, object]:
        """The verified manifest of a snapshot research may read now.

        One admission for every research read of a snapshot, whichever
        columns it reads: the bytes must be physically available and the
        lifecycle ACTIVE. A superseded or quarantined snapshot is refused
        here before a chunk is opened; its already-published derivatives
        remain readable through their own owners, which is a different
        question from admitting new research over it.

        Args:
            manifest_ref: Snapshot manifest reference resolved in this workspace.
            consumer: Consumer name used in refusal messages.

        Returns:
            Manifest of a physically available, active snapshot.

        Raises:
            ValueError: Snapshot bytes are unavailable or its lifecycle is not active.
        """
        lifecycle = self.resolver.feature_panel_snapshot_lifecycle(manifest_ref)
        if lifecycle.get("physical_availability", "AVAILABLE") != "AVAILABLE":
            raise ValueError(
                f"{consumer} requires physically available Feature Panel snapshot bytes "
                f"({lifecycle.get('physical_availability')})"
            )
        if lifecycle.get("lifecycle") != "ACTIVE":
            reason = lifecycle.get("reason") or "no lifecycle reason recorded"
            raise ValueError(
                f"{consumer} requires an ACTIVE Feature Panel snapshot "
                f"({lifecycle.get('lifecycle')}: {reason})"
            )
        return self.resolver.load_feature_panel_manifest(manifest_ref)

    def identity_batches(
        self, manifest_ref: str, *, include_row_hash: bool = True
    ) -> Iterator[pa.RecordBatch]:
        """Read only session/listing/row-hash identity for a one-time semantic index.

        Args:
            manifest_ref: Immutable snapshot to admit and verify.
            include_row_hash: Include each row's digest beside its session and listing.

        Yields:
            Batches of identity columns from qualified snapshot chunks.

        Raises:
            ValueError: Catalog, admission, temporal identity, or chunk integrity is invalid.
        """
        manifest = self.resolver.load_feature_panel_manifest(manifest_ref)
        summary = manifest.get("safe_summary")
        factors = summary.get("factor_catalog_summary") if isinstance(summary, dict) else None
        if not isinstance(factors, dict) or not factors:
            raise ValueError("feature panel snapshot has no factor catalog")
        request = FeaturePanelReadRequest(
            manifest_ref=manifest_ref,
            start_session=date.fromisoformat(str(manifest["history_start"])),
            end_session=date.fromisoformat(str(manifest["as_of_session"])),
            factor_columns=(min(str(value) for value in factors),),
            include_row_hash=True,
            batch_readahead=0,
            fragment_readahead=0,
        )
        self.admitted_manifest(manifest_ref, consumer="semantic index")
        self._validate_request_identity(manifest, request)
        paths = self._qualified_chunk_paths(manifest, request)
        dataset = ds.dataset(paths, format="parquet")
        yield from dataset.scanner(
            columns=["session_date", "listing_id", *(["row_hash"] if include_row_hash else [])],
            batch_readahead=0,
            fragment_readahead=0,
            use_threads=True,
        ).to_batches()

    def _qualified_chunk_paths(
        self,
        manifest: dict[str, object],
        request: FeaturePanelReadRequest,
    ) -> list[str]:
        raw_chunks = manifest.get("chunks")
        if not isinstance(raw_chunks, list):
            raise ValueError("feature panel manifest has no chunk catalog")
        panel_binding_hash = str(manifest.get("panel_binding_hash", ""))
        schema_hash = str(manifest.get("schema_hash", ""))
        summary = manifest.get("safe_summary")
        catalog = summary.get("factor_catalog_summary") if isinstance(summary, dict) else None
        row_hash_factor_ids = (
            tuple(sorted(str(value) for value in catalog)) if isinstance(catalog, dict) else None
        )
        paths: list[str] = []
        for item in raw_chunks:
            if not isinstance(item, dict):
                raise ValueError("feature panel chunk entry is invalid")
            first = date.fromisoformat(str(item["first_session"]))
            last = date.fromisoformat(str(item["last_session"]))
            if last < request.start_session or first > request.end_session:
                continue
            content_hash = str(item["chunk_hash"])
            path = self.resolver.resolve_feature_panel_chunk_ref(
                uri=str(item["uri"]),
                content_hash=content_hash,
                metadata_hash=str(item["metadata_hash"]),
            )
            # A chunk is hashed with the binding of the build that wrote it.
            # A manifest that reuses another build's partition records that
            # origin; one written before partition reuse recorded none, and
            # every such writer used the snapshot's own binding.
            origin = item.get("origin_binding_hash")
            self._validate_chunk_content(
                path,
                expected_chunk_hash=content_hash,
                expected_schema_hash=schema_hash,
                panel_binding_hash=str(origin) if origin is not None else panel_binding_hash,
                year=int(item["year"]),
                expected_row_count=int(item["row_count"]),
                row_hash_factor_ids=row_hash_factor_ids,
            )
            paths.append(str(path))
        return paths

    @classmethod
    def _validate_request_identity(
        cls, manifest: dict[str, object], request: FeaturePanelReadRequest
    ) -> None:
        start = date.fromisoformat(str(manifest["history_start"]))
        end = date.fromisoformat(str(manifest["as_of_session"]))
        if request.start_session < start or request.end_session > end:
            raise ValueError("feature panel read request is outside the frozen snapshot")
        summary = cls._validate_snapshot_identity(manifest)
        factor_summary = summary.get("factor_catalog_summary")
        if not isinstance(factor_summary, dict):
            raise ValueError("feature panel snapshot has no admitted factor catalog")
        unknown_factors = sorted(set(request.factor_columns) - set(factor_summary))
        if unknown_factors:
            raise ValueError(
                f"feature panel request contains factors outside the admitted catalog: "
                f"{unknown_factors}"
            )

    @staticmethod
    def _validate_snapshot_identity(manifest: dict[str, object]) -> dict[str, object]:
        """The snapshot's own admission: its temporal identity and Gateway admission.

        Every research read requires it, whatever it reads; the safe summary
        it was read from is returned for the request-specific checks.
        """
        summary = manifest.get("safe_summary")
        if not isinstance(summary, dict):
            raise ValueError("feature panel snapshot has no safe summary")
        temporal = summary.get("temporal_boundary")
        if not isinstance(temporal, dict):
            raise ValueError("feature panel snapshot has no temporal knowledge boundary")
        if temporal.get("market_as_of_session") != manifest.get("as_of_session"):
            raise ValueError("feature panel temporal market session is inconsistent")
        if not temporal.get("knowledge_cutoff_at") or not temporal.get("temporal_identity_hash"):
            raise ValueError("feature panel temporal knowledge identity is incomplete")
        temporal_identity = dict(temporal)
        claimed_temporal_hash = temporal_identity.pop("temporal_identity_hash")
        if canonical_hash(temporal_identity) != claimed_temporal_hash:
            raise ValueError("feature panel temporal knowledge hash is invalid")
        quality = summary.get("quality_governance")
        if not isinstance(quality, dict) or quality.get("gateway_qualified") is not True:
            raise ValueError("Factor Research requires a Feature Input Gateway admission")
        if not quality.get("quality_admission_hash"):
            raise ValueError("feature panel quality admission hash is missing")
        if int(quality.get("admitted_listing_count", -1)) + int(
            quality.get("quality_exclusion_count", -1)
        ) != int(quality.get("candidate_listing_count", -1)):
            raise ValueError("feature panel quality disclosure counts do not reconcile")
        return summary

    @staticmethod
    def _validate_chunk_content(
        path: object,
        *,
        expected_chunk_hash: str,
        expected_schema_hash: str,
        panel_binding_hash: str,
        year: int,
        expected_row_count: int,
        row_hash_factor_ids: tuple[str, ...] | None,
    ) -> None:
        parquet = pq.ParquetFile(path)
        logical_schema_hash = panel_schema_hash(parquet.schema_arrow)
        if logical_schema_hash != expected_schema_hash:
            raise ValueError("feature panel chunk schema hash mismatch")
        identity = parquet.read(columns=["row_hash"])
        if identity.num_rows != expected_row_count:
            raise ValueError("feature panel chunk row count mismatch")
        # The identity the writer sealed, computed by its owner over the
        # chunk's row hashes and whole schema.
        actual = panel_chunk_hash(
            identity, panel_binding_hash=panel_binding_hash, year=year, schema=parquet.schema_arrow
        )
        if actual != expected_chunk_hash:
            raise ValueError("feature panel chunk logical content hash mismatch")
        # The row hashes are bound; the values must be the ones they were hashed from
        # (LAWS.md EV2), checked once per file content in this process.
        digest = hashlib.sha256(Path(str(path)).read_bytes()).hexdigest()
        if digest not in _VALUES_VERIFIED:
            if not panel_rows_match_their_hashes(
                parquet.read(), row_hash_factor_ids=row_hash_factor_ids
            ):
                raise ValueError("feature panel chunk values do not match their row hashes")
            _VALUES_VERIFIED.add(digest)
