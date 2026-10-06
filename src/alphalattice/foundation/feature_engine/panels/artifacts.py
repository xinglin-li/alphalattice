"""Bounded annual composition for immutable Sector-Neutral Panel artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.contracts import (
    PANEL_ROW_IDENTITY_BY_BINDING,
    PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
    PanelCrossSectionRange,
    PanelMembership,
    canonical_hash,
)
from alphalattice.foundation.feature_engine.panels.identity import (
    CROSS_SECTION_IDENTITY_COLUMN,
    hash_panel_rows,
    panel_chunk_hash,
    panel_schema_hash,
)
from alphalattice.foundation.feature_engine.storage.contracts import PanelContentIdentity
from alphalattice.kernel.quant.sector_history import SectorHistory

# The columns a finalization and the content identity read from a written
# partition; every other column is a factor value no step here decodes.
_IDENTITY_COLUMNS = ("session_date", "listing_id", "row_hash")


@dataclass(frozen=True)
class PanelCompositionBinding:
    """Bind a Panel build to source revisions, factors, and row identity rule."""

    manifest_revision: str
    sector_revision: str
    catalog_hash: str
    policy_hash: str
    panel_binding_hash: str
    history_start: date
    as_of_session: date
    factor_ids: tuple[str, ...]
    # Which rule the rows' identities follow. Under the binding rule the
    # manifest and sector revisions above are hash inputs of every row; under
    # the cross-section rule they are the provenance of this build and each
    # row binds its own session's cross-section instead.
    row_identity_basis: str = PANEL_ROW_IDENTITY_BY_BINDING

    def __post_init__(self) -> None:
        """Check the factor axis, date range, and row identity rule."""
        if self.factor_ids != tuple(sorted(set(self.factor_ids))):
            raise ValueError("Panel composition factor IDs must be sorted and unique")
        if self.history_start > self.as_of_session:
            raise ValueError("Panel composition date range is inverted")
        if self.row_identity_basis not in {
            PANEL_ROW_IDENTITY_BY_BINDING,
            PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
        }:
            raise ValueError("Panel composition row identity basis is unknown")


@dataclass(frozen=True)
class PreparedPanelChunk:
    """Describe an immutable annual chunk and the binding that wrote it."""

    year: int
    first_session: date
    last_session: date
    row_count: int
    chunk_hash: str
    metadata_hash: str
    uri: str
    # The Panel binding of the build that wrote this file. ``chunk_hash`` is
    # evaluated with it, and the file's metadata names it. A composition may
    # reuse a partition another build wrote under an earlier binding; ``None``
    # is only the constructor default for callers that describe a chunk of
    # the composition's own binding, and the session always resolves it.
    origin_binding_hash: str | None = None
    # Under the cross-section rule: the identity each range of the year's
    # sessions was computed over, with its member count. Empty for a chunk
    # under the binding rule, whose rows all share the binding's identity.
    cross_sections: tuple[PanelCrossSectionRange, ...] = ()


@dataclass(frozen=True)
class PanelPartitionOriginRecord:
    """One build whose cells a composition holds: the binding's own inputs.

    The SPY revision, and under the cross-section rule the manifest and
    sector revisions too, are what a rematerializer needs to rebuild the
    exact binding a partition or receipt group was produced under; ``None``
    only in a staging document written before these were recorded.
    """

    spy_revision: str | None
    manifest_revision: str | None = None
    sector_revision: str | None = None
    # The catalog the build computed under, when it is not the composition's: cells of a
    # catalog that the composition's only adds columns to, carried with their batches (V92).
    catalog_hash: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return the recorded source revisions as a staging payload."""
        payload: dict[str, object] = {
            "spy_revision": self.spy_revision,
            "manifest_revision": self.manifest_revision,
            "sector_revision": self.sector_revision,
        }
        if self.catalog_hash is not None:
            payload["catalog_hash"] = self.catalog_hash
        return payload


@dataclass(frozen=True)
class PanelColumnExtension:
    """The catalog a composition's catalog only adds columns to (V92).

    A base Panel built under it holds every column but the added ones, each
    value the one the composition's catalog computes: its partitions are
    merged with the added columns year by year instead of recomputed.
    """

    catalog_hash: str
    factor_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreparedPanelComposition:
    """Record the staged chunks, reused years, and source origins."""

    composition_hash: str
    binding: PanelCompositionBinding
    content: PanelContentIdentity
    schema_hash: str
    chunks: tuple[PreparedPanelChunk, ...]
    reused_chunk_count: int
    written_chunk_count: int
    reused_years: tuple[int, ...] = ()
    # Every build whose cells this composition holds (chunk origins and the
    # bindings stamped on its availability rows), each with the revisions its
    # binding was created from.
    partition_origins: Mapping[str, PanelPartitionOriginRecord] = field(default_factory=dict)
    # The membership the rows were computed over; None for a composition
    # under the binding rule, whose axis is its manifest.
    membership: PanelMembership | None = None
    listing_ids: tuple[str, ...] = ()

    @property
    def cross_sections(self) -> tuple[PanelCrossSectionRange, ...]:
        """Collect the cross-section ranges recorded by all chunks."""
        return tuple(item for chunk in self.chunks for item in chunk.cross_sections)


@dataclass(frozen=True)
class _PatchFragment:
    year: int
    factor_ids: tuple[str, ...]
    table: pa.Table


@dataclass(frozen=True)
class _YearAxis:
    """Every (session, listing) row a year holds, each session's members in order, as keys.

    A row's key is its session's place in the year times the listing axis's length plus its
    listing's place on that axis, so the year's keys ascend and a table's rows are located by
    one vector search rather than a lookup per row (V92).
    """

    sessions: pa.Array
    listings: pa.Array
    keys: np.ndarray

    def positions(self, table: pa.Table) -> np.ndarray:
        """Each row's place in the year, or -1 for a row the year does not hold."""
        session = _located(table.column("session_date"), self.sessions)
        listing = _located(table.column("listing_id"), self.listings)
        found: np.ndarray = np.full(table.num_rows, -1, dtype=np.intp)
        if not len(self.keys):
            return found
        keys = session.astype(np.int64) * len(self.listings) + listing
        at = np.searchsorted(self.keys, keys)
        held = (
            (session >= 0)
            & (listing >= 0)
            & (self.keys[np.minimum(at, len(self.keys) - 1)] == keys)
        )
        found[held] = at[held]
        return found

    def columns(self) -> tuple[np.ndarray, np.ndarray]:
        """The year's sessions and listings, row by row, as the values a frame holds."""
        width = len(self.listings)
        return (
            np.asarray(self.sessions.to_pylist(), dtype=object)[self.keys // width],
            np.asarray(self.listings.to_pylist(), dtype=object)[self.keys % width],
        )


def _located(column: pa.ChunkedArray, axis: pa.Array) -> np.ndarray:
    """Each value's place on an axis, or -1 for a value the axis does not hold."""
    located = pc.index_in(column, value_set=axis.cast(column.type))
    places: np.ndarray = np.asarray(
        pc.fill_null(located, -1).to_numpy(zero_copy_only=False), dtype=np.int64
    )
    return places


class PanelArtifactCompositionOwner:
    """Compose annual-wide immutable chunks without persisting Panel values."""

    _STAGING_KIND = "PanelArtifactComposition"

    def __init__(self, resolver: ArtifactResolver) -> None:
        """Locate composition staging under the artifact resolver."""
        self.resolver = resolver
        self._staging_root: Path = Path(resolver.root) / "feature-panel" / "composition-staging"

    def begin(
        self,
        *,
        operation_id: str,
        binding: PanelCompositionBinding,
        base_manifest: Mapping[str, object] | None,
        sessions: Sequence[date],
        listing_ids: Sequence[str],
        spy_revision: str | None = None,
        membership: PanelMembership | None = None,
        sector_history: SectorHistory | None = None,
        extension: PanelColumnExtension | None = None,
    ) -> PanelCompositionSession:
        """Open a bounded Panel composition session for one operation."""
        return PanelCompositionSession(
            owner=self,
            operation_id=operation_id,
            binding=binding,
            base_manifest=base_manifest,
            sessions=sessions,
            listing_ids=listing_ids,
            spy_revision=spy_revision,
            membership=membership,
            sector_history=sector_history,
            extension=extension,
        )

    def load(self, *, panel_binding_hash: str, panel_content_hash: str) -> PreparedPanelComposition:
        """Load and verify a staged composition by binding and content hash."""
        path = self._composition_path(panel_binding_hash, panel_content_hash)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") != self._STAGING_KIND:
            raise ValueError("Panel composition staging kind is invalid")
        claimed_hash = str(payload.get("composition_hash", ""))
        identity = dict(payload)
        identity.pop("composition_hash", None)
        if claimed_hash != canonical_hash(identity):
            raise ValueError("Panel composition staging identity is invalid")
        binding_payload = cast(dict[str, object], payload["binding"])
        binding = PanelCompositionBinding(
            manifest_revision=str(binding_payload["manifest_revision"]),
            sector_revision=str(binding_payload["sector_revision"]),
            catalog_hash=str(binding_payload["catalog_hash"]),
            policy_hash=str(binding_payload["policy_hash"]),
            panel_binding_hash=str(binding_payload["panel_binding_hash"]),
            history_start=date.fromisoformat(str(binding_payload["history_start"])),
            as_of_session=date.fromisoformat(str(binding_payload["as_of_session"])),
            factor_ids=tuple(
                str(value) for value in cast(list[object], binding_payload["factor_ids"])
            ),
            row_identity_basis=str(
                binding_payload.get("row_identity_basis", PANEL_ROW_IDENTITY_BY_BINDING)
            ),
        )
        content_payload = cast(dict[str, object], payload["content"])
        content = PanelContentIdentity(
            panel_content_hash=str(content_payload["panel_content_hash"]),
            row_count=int(cast(int, content_payload["row_count"])),
            availability_count=int(cast(int, content_payload["availability_count"])),
            history_start=date.fromisoformat(str(content_payload["history_start"])),
            as_of_session=date.fromisoformat(str(content_payload["as_of_session"])),
        )
        chunks = tuple(
            prepared_chunk(item, default_origin=binding.panel_binding_hash)
            for item in cast(list[dict[str, object]], payload["chunks"])
        )
        for chunk in chunks:
            self.resolver.resolve_feature_panel_chunk_ref(
                uri=chunk.uri,
                content_hash=chunk.chunk_hash,
                metadata_hash=chunk.metadata_hash,
            )
        if (
            binding.panel_binding_hash != panel_binding_hash
            or content.panel_content_hash != panel_content_hash
        ):
            raise ValueError("Panel composition staging lookup identity is invalid")
        recorded_origins = payload.get("partition_origins")
        if recorded_origins is None:
            # Written before partition reuse: every cell is the composition's.
            origins: dict[str, PanelPartitionOriginRecord] = {
                binding.panel_binding_hash: PanelPartitionOriginRecord(None)
            }
        elif isinstance(recorded_origins, dict):
            origins = {str(key): _origin_record(value) for key, value in recorded_origins.items()}
        else:
            raise ValueError("Panel composition staging origins are invalid")
        membership_payload = payload.get("membership")
        membership = (
            PanelMembership.from_payload(cast(Mapping[str, object], membership_payload))
            if isinstance(membership_payload, Mapping)
            else None
        )
        if binding.row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION and (
            membership is None
        ):
            raise ValueError("Panel composition under the cross-section rule has no membership")
        return PreparedPanelComposition(
            composition_hash=claimed_hash,
            binding=binding,
            content=content,
            schema_hash=str(payload["schema_hash"]),
            chunks=chunks,
            reused_chunk_count=0,
            written_chunk_count=0,
            partition_origins=origins,
            membership=membership,
            listing_ids=(
                membership.listing_ids
                if membership is not None
                else tuple(str(value) for value in payload.get("listing_ids", ()))
            ),
        )

    def _publish(self, composition: PreparedPanelComposition) -> None:
        identity = _composition_identity(composition)
        if canonical_hash(identity) != composition.composition_hash:
            raise ValueError("Panel composition hash does not match its payload")
        payload = {**identity, "composition_hash": composition.composition_hash}
        target = self._composition_path(
            composition.binding.panel_binding_hash,
            composition.content.panel_content_hash,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
        if target.exists():
            if target.read_bytes() != serialized:
                raise ValueError("Panel composition identity was reused with different content")
            return
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        staged.write_bytes(serialized)
        os.replace(staged, target)

    def _composition_path(self, panel_binding_hash: str, panel_content_hash: str) -> Path:
        return self._staging_root / panel_binding_hash / f"{panel_content_hash}.json"


class PanelCompositionSession:
    """Accumulate bounded patches and finalize one year at a time."""

    def __init__(
        self,
        *,
        owner: PanelArtifactCompositionOwner,
        operation_id: str,
        binding: PanelCompositionBinding,
        base_manifest: Mapping[str, object] | None,
        sessions: Sequence[date],
        listing_ids: Sequence[str],
        spy_revision: str | None = None,
        membership: PanelMembership | None = None,
        sector_history: SectorHistory | None = None,
        extension: PanelColumnExtension | None = None,
    ) -> None:
        """Bind source axes and determine session membership for composition."""
        self.owner = owner
        self.operation_id = operation_id
        self.binding = binding
        self.extension = extension
        self.spy_revision = spy_revision
        self.base_manifest = dict(base_manifest) if base_manifest is not None else None
        self._sessions = tuple(
            session
            for session in sorted(set(sessions))
            if binding.history_start <= session <= binding.as_of_session
        )
        self._listing_ids = tuple(sorted(set(str(value) for value in listing_ids)))
        if not self._sessions or not self._listing_ids:
            raise ValueError("Panel composition requires sessions and listings")
        self._sessions_by_year = {
            year: tuple(value for value in self._sessions if value.year == year)
            for year in range(self._sessions[0].year, self._sessions[-1].year + 1)
        }
        self.membership = membership
        self._identity_by_session: dict[date, str] = {}
        self._cross_sections_by_year: dict[int, tuple[PanelCrossSectionRange, ...]] = {}
        if binding.row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
            if membership is None or sector_history is None:
                raise ValueError(
                    "Panel composition under the cross-section rule needs membership and sectors"
                )
            if tuple(sorted(membership.listing_ids)) != self._listing_ids:
                raise ValueError("Panel composition membership axis differs from its listings")
            for year, year_sessions in self._sessions_by_year.items():
                if not year_sessions:
                    continue
                ranges = _cross_sections(membership, year_sessions, sector_history)
                self._cross_sections_by_year[year] = ranges
                for item in ranges:
                    for session in year_sessions:
                        if item.first_session <= session <= item.last_session:
                            self._identity_by_session[session] = item.cross_section_identity
        elif membership is not None and not membership.is_uniform:
            raise ValueError("Panel composition under the binding rule holds one member set")
        self._fragments: list[_PatchFragment] = []
        self._active_patch_year: int | None = None
        self._prepared_by_year: dict[int, PreparedPanelChunk] = {}
        self._reused_years: list[int] = []
        self._base_axes: dict[int, tuple[date, ...]] = {}
        self._identity_tables: dict[str, pa.Table] = {}

    def members(self, session: date) -> tuple[str, ...]:
        """The listings whose rows the session holds, in physical (sorted) order."""
        if self.membership is None:
            return self._listing_ids
        return tuple(sorted(self.membership.members(session)))

    def cross_sections(self, year: int) -> tuple[PanelCrossSectionRange, ...]:
        """Return the cross-section ranges recorded for a year."""
        return self._cross_sections_by_year.get(year, ())

    def all_cross_sections(self) -> tuple[PanelCrossSectionRange, ...]:
        """Every session's cross-section over the whole calendar, as ranges."""
        return tuple(
            item
            for year in sorted(self._cross_sections_by_year)
            for item in self._cross_sections_by_year[year]
        )

    def cross_section_by_session(self) -> dict[str, str]:
        """Session ISO date -> cross-section identity, for availability keyed by it."""
        return {
            session.isoformat(): identity for session, identity in self._identity_by_session.items()
        }

    def stage_patch(
        self,
        *,
        rows: pd.DataFrame | Sequence[Mapping[str, object]],
        factor_ids: Sequence[str],
        materialization_receipt_hash: str,
    ) -> None:
        """Stage ordered factor rows for annual chunk composition."""
        factors = tuple(sorted(set(str(value) for value in factor_ids)))
        if not factors or not set(factors).issubset(self.binding.factor_ids):
            raise ValueError("Panel composition patch has unknown factor IDs")
        frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        required = {"session_date", "listing_id", *factors}
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError("Panel composition patch is incomplete")
        frame = frame.loc[:, ["session_date", "listing_id", *factors]].copy()
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
        frame["listing_id"] = frame["listing_id"].astype(str)
        frame["materialization_receipt_hash"] = materialization_receipt_hash
        if frame.duplicated(["session_date", "listing_id"]).any():
            raise ValueError("Panel composition patch contains duplicate rows")
        for year, year_frame in frame.groupby(frame["session_date"].map(lambda value: value.year)):
            year = int(year)
            if self._active_patch_year is not None and year < self._active_patch_year:
                raise ValueError("Panel composition patches must be staged chronologically")
            if self._active_patch_year is not None and year > self._active_patch_year:
                self._seal_patch_year(self._active_patch_year)
            self._active_patch_year = year
            table = pa.Table.from_pandas(
                year_frame.loc[
                    :, ["session_date", "listing_id", "materialization_receipt_hash", *factors]
                ],
                preserve_index=False,
            )
            self._fragments.append(_PatchFragment(year=year, factor_ids=factors, table=table))

    def finalize(
        self,
        *,
        availability: Sequence[Mapping[str, object]],
        progress: Callable[[int, int, int], object] | None = None,
    ) -> PreparedPanelComposition:
        """Seal annual chunks and publish the prepared composition."""
        if self._active_patch_year is not None:
            self._seal_patch_year(self._active_patch_year)
            self._active_patch_year = None
        base_chunks = self._base_chunks()
        prepared: list[PreparedPanelChunk] = []
        schema_hash: str | None = None
        reused = 0
        written = len(self._prepared_by_year)
        work_years = tuple(year for year, values in self._sessions_by_year.items() if values)
        reusable = set(self.reusable_years())
        for completed, year in enumerate(work_years, start=1):
            chunk = self._prepared_by_year.get(year)
            base = base_chunks.get(year)
            if chunk is None:
                if base is None or year not in reusable:
                    raise ValueError(f"Panel composition has no source for year {year}")
                chunk = base
                reused += 1
                self._reused_years.append(year)
            prepared.append(chunk)
            # Finalization judges a partition by its identity columns and its
            # schema; the factor columns are decoded by no step here, so the
            # partition is read once, projected, and the content identity
            # below reads the same projection.
            table = self._identity_table(chunk)
            current_schema_hash = panel_schema_hash(self._chunk_schema(chunk))
            if chunk is base:
                # A partition another build wrote is named by this composition
                # only after its bytes still prove the identity the base
                # manifest recorded for it, under the binding that wrote it.
                assert chunk.origin_binding_hash is not None
                if table.num_rows != chunk.row_count or (
                    panel_chunk_hash(
                        table,
                        panel_binding_hash=chunk.origin_binding_hash,
                        year=year,
                        schema=self._chunk_schema(chunk),
                    )
                    != chunk.chunk_hash
                ):
                    raise ValueError(f"Panel composition base chunk for {year} failed identity")
                if _session_axis(table) != self._sessions_by_year[year]:
                    raise ValueError(
                        f"Panel composition base chunk for {year} does not hold the calendar"
                    )
            if schema_hash is None:
                schema_hash = current_schema_hash
            elif schema_hash != current_schema_hash:
                raise ValueError("Panel composition annual schemas differ")
            if progress is not None:
                progress(completed, len(work_years), year)
        if not prepared or schema_hash is None:
            raise ValueError("Panel composition produced no chunks")
        content = panel_content_identity(
            binding=self.binding,
            chunks=prepared,
            load_table=self._identity_table,
            availability=availability,
        )
        self._identity_tables.clear()
        origins = self._partition_origins(prepared, availability)
        composition = PreparedPanelComposition(
            composition_hash=canonical_hash(
                _composition_identity_parts(
                    binding=self.binding,
                    content=content,
                    schema_hash=schema_hash,
                    chunks=prepared,
                    partition_origins=origins,
                    membership=self.membership,
                    listing_ids=self._listing_ids,
                )
            ),
            binding=self.binding,
            content=content,
            schema_hash=schema_hash,
            chunks=tuple(prepared),
            reused_chunk_count=reused,
            written_chunk_count=written,
            reused_years=tuple(self._reused_years),
            partition_origins=origins,
            membership=self.membership,
            listing_ids=self._listing_ids,
        )
        self.owner._publish(composition)
        return composition

    def _partition_origins(
        self,
        chunks: Sequence[PreparedPanelChunk],
        availability: Sequence[Mapping[str, object]],
    ) -> dict[str, PanelPartitionOriginRecord]:
        """Resolve every binding whose cells the composition holds to its inputs.

        The composition's own binding resolves to the revisions it was
        given; any other binding must be an origin the compatible base
        manifest recorded (or the base's own). A binding neither knows is a
        cell written by a build no published snapshot describes, and a
        composition that cannot say where a cell came from must not be sealed.
        """
        present: set[str] = {
            chunk.origin_binding_hash for chunk in chunks if chunk.origin_binding_hash is not None
        }
        for item in availability:
            binding_hash = item.get("panel_binding_hash")
            if binding_hash:
                present.add(str(binding_hash))
        base_origins = (
            manifest_partition_origins(self.base_manifest)
            if self.base_manifest is not None and self.base_compatible
            else {}
        )
        base_catalog = (
            _manifest_catalog_hash(self.base_manifest) if self.base_manifest is not None else None
        )
        origins: dict[str, PanelPartitionOriginRecord] = {}
        for binding_hash in sorted(present):
            if binding_hash == self.binding.panel_binding_hash:
                origins[binding_hash] = PanelPartitionOriginRecord(
                    spy_revision=self.spy_revision,
                    manifest_revision=self.binding.manifest_revision,
                    sector_revision=self.binding.sector_revision,
                )
            elif binding_hash in base_origins:
                recorded = base_origins[binding_hash]
                # An origin the base recorded without a catalog shares the base's (V92).
                catalog = cast(str | None, recorded.get("catalog_hash")) or base_catalog
                origins[binding_hash] = PanelPartitionOriginRecord(
                    spy_revision=cast(str | None, recorded["spy_revision"]),
                    manifest_revision=cast(str | None, recorded.get("manifest_revision")),
                    sector_revision=cast(str | None, recorded.get("sector_revision")),
                    catalog_hash=catalog if catalog != self.binding.catalog_hash else None,
                )
            else:
                raise ValueError(
                    f"Panel composition cannot resolve the origin of binding {binding_hash}"
                )
        return origins

    def _seal_patch_year(self, year: int) -> None:
        fragments = tuple(self._fragments)
        if not fragments or any(item.year != year for item in fragments):
            raise ValueError("Panel composition patch-year state is invalid")
        # A year the base cannot serve even as a merge base (another calendar
        # or membership) is composed from its fragments alone; the plan forced
        # every one of its sessions, so no row is left for the base to supply.
        table = self._compose_year(
            year=year,
            sessions=self._sessions_by_year[year],
            base=(
                self._base_chunks().get(year)
                if self._base_prefix_length(year) is not None
                else None
            ),
            fragments=fragments,
        )
        current_schema_hash = panel_schema_hash(table.schema)
        chunk_hash = panel_chunk_hash(
            table,
            panel_binding_hash=self.binding.panel_binding_hash,
            year=year,
        )
        descriptor = self.owner.resolver.publish_feature_panel_chunk(
            table=table,
            content_hash=chunk_hash,
            metadata={
                "panel_binding_hash": self.binding.panel_binding_hash,
                "calendar_year": str(year),
                "schema_hash": current_schema_hash,
            },
        )
        chunk_sessions = cast(list[date], table.column("session_date").to_pylist())
        self._prepared_by_year[year] = PreparedPanelChunk(
            year=year,
            first_session=min(chunk_sessions),
            last_session=max(chunk_sessions),
            row_count=table.num_rows,
            chunk_hash=chunk_hash,
            metadata_hash=descriptor.metadata_hash,
            uri=descriptor.uri,
            origin_binding_hash=self.binding.panel_binding_hash,
            cross_sections=self.cross_sections(year),
        )
        self._fragments.clear()

    @property
    def base_compatible(self) -> bool:
        """Whether the base snapshot's partitions may serve this composition.

        Under the binding rule a partition's rows are determined by the
        membership, sector map, catalog, policy and Formula values of its
        year (the row hash binds the first four per row), so compatibility is
        the composition binding read from the base manifest's sealed lineage
        -- not binding equality, which no two days share, because the SPY
        revision is the market-reference state the build saw and a new SPY
        session changes it without changing any closed year.

        Under the cross-section rule each row binds its own session's
        cross-section, so the manifest and sector revisions of the base are
        not part of compatibility either: a base under the same rule,
        catalog, policy, history start and factor axis may serve any year
        whose cross-sections it recorded as the ones this build computes.
        A base built under the catalog this composition's only adds columns
        to serves the same way, as the base of every year's merge
        (``base_extended``, V92).
        """
        if self.base_manifest is None:
            return False
        if self.base_manifest.get("panel_binding_hash") == self.binding.panel_binding_hash:
            return True
        return (
            manifest_composition_binding(self.base_manifest)
            == _composition_binding_key(self.binding)
            or self.base_extended
        )

    @property
    def base_extended(self) -> bool:
        """Whether the base serves only through the composition's column extension (V92).

        Its partitions then lack the added columns: no year is reused whole,
        and each is merged with the added columns' values.
        """
        if self.extension is None or self.base_manifest is None:
            return False
        return manifest_composition_binding(self.base_manifest) == _composition_binding_key(
            replace(
                self.binding,
                catalog_hash=self.extension.catalog_hash,
                factor_ids=self.extension.factor_ids,
            )
        )

    def _added_factor_ids(self) -> frozenset[str]:
        """The columns a merge takes from patches alone: the extension's added ones."""
        if not self.base_extended:
            return frozenset()
        assert self.extension is not None
        return frozenset(self.binding.factor_ids) - frozenset(self.extension.factor_ids)

    def reusable_years(self) -> tuple[int, ...]:
        """Years whose base partition holds exactly the current calendar and membership."""
        if self.base_extended:
            return ()
        return tuple(
            year
            for year, sessions in self._sessions_by_year.items()
            if sessions and self._base_prefix_length(year) == len(sessions)
        )

    def unreusable_years(self) -> tuple[int, ...]:
        """Years no base partition can serve even as the base of a merge.

        A compatible base may lack a year (the calendar grew into it), or hold
        it under another calendar or membership. Those years are recomputed
        whole. A year the base holds as a prefix of the current calendar --
        the year a new session lands in, or the year a membership change
        takes effect in -- is not among them: its base partition is merged
        with the sessions beyond the prefix.
        """
        return tuple(
            year
            for year, sessions in self._sessions_by_year.items()
            if sessions and self._base_prefix_length(year) is None
        )

    def unreusable_sessions(self) -> tuple[date, ...]:
        """Sessions the plan must compute regardless of source deltas.

        Every session of an unreusable year, and the sessions of a prefix-held
        year that lie beyond its base prefix.
        """
        forced: list[date] = []
        for year, sessions in self._sessions_by_year.items():
            if not sessions:
                continue
            held = self._base_prefix_length(year)
            forced.extend(sessions if held is None else sessions[held:])
        return tuple(forced)

    def _base_prefix_length(self, year: int) -> int | None:
        """How many leading sessions of the year's calendar the base partition serves.

        Decided on the partition's actual session axis, not on the endpoints
        and row count its manifest entry records: two calendars can share
        both (a session replaced by another between the same endpoints). The
        axis must be a prefix of the requested calendar -- every session it
        holds requested, none requested lying between two it holds -- and
        its rows that axis times the membership it recorded. Under the
        cross-section rule the prefix is further cut at the first session
        whose recorded cross-section is not the one this build computes: the
        sessions before a membership change are served, the ones from it on
        are recomputed. The full calendar is the prefix of its own length;
        ``None`` means no partition can serve the year even as the base of a
        merge.
        """
        base = self._base_chunks().get(year)
        sessions = self._sessions_by_year.get(year, ())
        if base is None or not sessions:
            return None
        axis = self._base_axis(year, base)
        if axis != sessions[: len(axis)]:
            return None
        if self.binding.row_identity_basis == PANEL_ROW_IDENTITY_BY_BINDING:
            if base.row_count != len(axis) * len(self._listing_ids):
                return None
            return len(axis)
        if not base.cross_sections:
            return None
        recorded = _identity_by_session(base.cross_sections, axis)
        if recorded is None:
            return None
        member_counts = {
            item.cross_section_identity: item.member_count for item in base.cross_sections
        }
        if base.row_count != sum(member_counts[recorded[session]] for session in axis):
            return None
        held = 0
        for session in axis:
            if recorded[session] != self._identity_by_session.get(session):
                break
            held += 1
        return held if held else None

    def _base_axis(self, year: int, base: PreparedPanelChunk) -> tuple[date, ...]:
        """The sessions a base partition holds, read once from its bytes.

        The same column projection the recovery binding reads at publication;
        cached for the session so the file is not scanned again by every
        caller that judges the year. The manifest entry must describe the
        file it names: a partition whose bytes disagree with its recorded
        endpoints or row count is refused, not silently recomputed.
        """
        if year in self._base_axes:
            return self._base_axes[year]
        path = self.owner.resolver.resolve_feature_panel_chunk_ref(
            uri=base.uri, content_hash=base.chunk_hash, metadata_hash=base.metadata_hash
        )
        table = pq.read_table(path, columns=["session_date"])
        axis = _session_axis(table)
        if (
            not axis
            or table.num_rows != base.row_count
            or axis[0] != base.first_session
            or axis[-1] != base.last_session
        ):
            raise ValueError(f"Panel composition base chunk for {year} contradicts its manifest")
        self._base_axes[year] = axis
        return axis

    def _base_chunks(self) -> dict[int, PreparedPanelChunk]:
        if not self.base_compatible:
            return {}
        assert self.base_manifest is not None
        default_origin = str(self.base_manifest["panel_binding_hash"])
        result: dict[int, PreparedPanelChunk] = {}
        for item in cast(list[dict[str, object]], self.base_manifest.get("chunks", [])):
            chunk = prepared_chunk(item, default_origin=default_origin)
            result[chunk.year] = chunk
        return result

    def _chunk_path(self, chunk: PreparedPanelChunk) -> Path:
        path: Path = self.owner.resolver.resolve_feature_panel_chunk_ref(
            uri=chunk.uri,
            content_hash=chunk.chunk_hash,
            metadata_hash=chunk.metadata_hash,
        )
        return path

    def _load_base_table(self, chunk: PreparedPanelChunk) -> pa.Table:
        return pq.read_table(self._chunk_path(chunk)).replace_schema_metadata(None)

    def _identity_table(self, chunk: PreparedPanelChunk) -> pa.Table:
        """The partition's identity columns, read once per finalization."""
        table = self._identity_tables.get(chunk.chunk_hash)
        if table is None:
            table = pq.read_table(self._chunk_path(chunk), columns=list(_IDENTITY_COLUMNS))
            self._identity_tables[chunk.chunk_hash] = table
        return table

    def _chunk_schema(self, chunk: PreparedPanelChunk) -> pa.Schema:
        """The whole schema of a written partition, from its metadata alone."""
        return pq.read_schema(self._chunk_path(chunk)).remove_metadata()

    def _year_axis(self, sessions: Sequence[date]) -> _YearAxis:
        """Every (session, listing) row the year holds: each session's members."""
        place = {listing_id: index for index, listing_id in enumerate(self._listing_ids)}
        width = len(self._listing_ids)
        try:
            keys: np.ndarray = np.fromiter(
                (
                    index * width + place[listing_id]
                    for index, session in enumerate(sessions)
                    for listing_id in self.members(session)
                ),
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError("Panel composition member is outside its listing axis") from exc
        if (np.diff(keys) <= 0).any():
            raise ValueError("Panel composition year axis is not in session and listing order")
        return _YearAxis(
            sessions=pa.array(list(sessions), pa.date32()),
            listings=pa.array(self._listing_ids, pa.string()),
            keys=keys,
        )

    def _hash_rows(self, raw: pa.Table) -> pa.Table:
        if self.binding.row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
            sessions = sorted(self._identity_by_session)
            located = pc.index_in(
                raw.column("session_date"), value_set=pa.array(sessions, pa.date32())
            )
            if located.null_count:
                raise ValueError("Panel composition row has no cross-section identity")
            identities = pa.array(
                [self._identity_by_session[session] for session in sessions], pa.string()
            ).take(located)
            raw = raw.append_column(CROSS_SECTION_IDENTITY_COLUMN, identities)
        return hash_panel_rows(
            raw,
            manifest_revision=self.binding.manifest_revision,
            sector_revision=self.binding.sector_revision,
            catalog_hash=self.binding.catalog_hash,
            policy_hash=self.binding.policy_hash,
            factor_ids=self.binding.factor_ids,
            identity_basis=self.binding.row_identity_basis,
        )

    def _compose_year(
        self,
        *,
        year: int,
        sessions: Sequence[date],
        base: PreparedPanelChunk | None,
        fragments: Sequence[_PatchFragment],
    ) -> pa.Table:
        columns = [
            "session_date",
            "listing_id",
            "materialization_receipt_hash",
            *self.binding.factor_ids,
        ]
        axis = self._year_axis(sessions)
        if base is None and all(
            fragment.factor_ids == self.binding.factor_ids for fragment in fragments
        ):
            raw = pa.concat_tables(
                [fragment.table.select(columns) for fragment in fragments],
                promote_options="default",
            )
            found = axis.positions(raw)
            if (
                len(found) != len(axis.keys)
                or (found < 0).any()
                or len(np.unique(found)) != len(found)
            ):
                raise ValueError(f"Panel composition is not complete for {year}")
            return self._hash_rows(raw)
        added = self._added_factor_ids()
        # Every row the year holds, by position: the base's cells, then each patch's in staging
        # order, each later write replacing an earlier one, as the frame the rows are hashed
        # from always took them; a base row outside the year's members is not kept.
        size = len(axis.keys)
        values: dict[str, np.ndarray] = {
            factor_id: np.full(size, np.nan, dtype=float) for factor_id in self.binding.factor_ids
        }
        receipts: np.ndarray = np.full(size, None, dtype=object)
        covered: dict[str, np.ndarray] = {
            factor_id: np.zeros(size, dtype=bool) for factor_id in sorted(added)
        }
        if base is not None:
            base_table = self._load_base_table(base)
            base_axis = _session_axis(base_table)
            if base_axis != tuple(sessions)[: len(base_axis)]:
                raise ValueError(f"Panel composition base chunk for {year} is not a prefix")
            # A base under the catalog this one only adds columns to holds every
            # column but the added ones, which every row takes from a patch (V92).
            missing = set(columns) - set(base_table.column_names)
            if not missing <= added:
                raise ValueError(f"Panel composition base chunk for {year} lacks its columns")
            held = axis.positions(base_table)
            kept = held >= 0
            receipts[held[kept]] = np.asarray(
                base_table.column("materialization_receipt_hash").to_pylist(), dtype=object
            )[kept]
            for factor_id in self.binding.factor_ids:
                if factor_id not in missing:
                    values[factor_id][held[kept]] = base_table.column(factor_id).to_numpy(
                        zero_copy_only=False
                    )[kept]
        for fragment in fragments:
            at = axis.positions(fragment.table)
            if (at < 0).any():
                raise ValueError(f"Panel patch is outside the admitted {year} universe")
            for factor_id in fragment.factor_ids:
                values[factor_id][at] = fragment.table.column(factor_id).to_numpy(
                    zero_copy_only=False
                )
                if factor_id in covered:
                    covered[factor_id][at] = True
            receipts[at] = np.asarray(
                fragment.table.column("materialization_receipt_hash").to_pylist(), dtype=object
            )
        if pd.isna(receipts).any() or not all(item.all() for item in covered.values()):
            raise ValueError(f"Panel composition has unmaterialized rows for {year}")
        session_column, listing_column = axis.columns()
        raw = pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "session_date": session_column,
                    "listing_id": listing_column,
                    "materialization_receipt_hash": receipts,
                    **values,
                }
            ).loc[:, columns],
            preserve_index=False,
        )
        return self._hash_rows(raw)


def panel_content_identity(
    *,
    binding: PanelCompositionBinding,
    chunks: Sequence[PreparedPanelChunk],
    load_table: Callable[[PreparedPanelChunk], pa.Table],
    availability: Sequence[Mapping[str, object]],
) -> PanelContentIdentity:
    """Hash ordered Panel chunks and availability into content identity."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "manifest_revision": binding.manifest_revision,
                "sector_revision": binding.sector_revision,
                "catalog_hash": binding.catalog_hash,
                "policy_hash": binding.policy_hash,
                "panel_binding_hash": binding.panel_binding_hash,
                "history_start": binding.history_start.isoformat(),
                "as_of_session": binding.as_of_session.isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    row_count = 0
    for chunk in chunks:
        table = load_table(chunk)
        joined = pc.binary_join_element_wise(
            pa.scalar("R"),
            pc.cast(table.column("session_date"), pa.string()),
            table.column("listing_id"),
            table.column("row_hash"),
            pa.scalar("|"),
        )
        lines = pc.binary_join_element_wise(joined, pa.scalar("\n"), pa.scalar(""))
        data = lines.combine_chunks().buffers()[2]
        if data is None:
            raise ValueError("Panel row identity buffer is missing")
        digest.update(memoryview(data))
        row_count += table.num_rows
    ordered_availability = sorted(
        availability,
        key=lambda item: (str(item["session_date"]), str(item["factor_id"])),
    )
    for item in ordered_availability:
        availability_hash = str(item.get("availability_hash") or "")
        if not availability_hash:
            raise ValueError("Panel availability lacks a content hash")
        digest.update(
            (
                f"A|{date.fromisoformat(str(item['session_date'])).isoformat()}|"
                f"{item['factor_id']}|{availability_hash}\n"
            ).encode()
        )
    if row_count == 0 or not ordered_availability:
        raise ValueError("Panel content identity has no durable rows")
    return PanelContentIdentity(
        panel_content_hash=digest.hexdigest(),
        row_count=row_count,
        availability_count=len(ordered_availability),
        history_start=binding.history_start,
        as_of_session=binding.as_of_session,
    )


def _session_axis(table: pa.Table) -> tuple[date, ...]:
    """The distinct sessions of a Panel table, in calendar order."""
    return tuple(sorted(set(cast(list[date], table.column("session_date").to_pylist()))))


def _identity_by_session(
    ranges: Sequence[PanelCrossSectionRange], sessions: Sequence[date]
) -> dict[date, str] | None:
    """Each session's recorded cross-section identity, or None if a session has none."""
    result: dict[date, str] = {}
    for session in sessions:
        for item in ranges:
            if item.first_session <= session <= item.last_session:
                result[session] = item.cross_section_identity
                break
        else:
            return None
    return result


def _composition_identity(composition: PreparedPanelComposition) -> dict[str, object]:
    return _composition_identity_parts(
        binding=composition.binding,
        content=composition.content,
        schema_hash=composition.schema_hash,
        chunks=composition.chunks,
        partition_origins=composition.partition_origins,
        membership=composition.membership,
        listing_ids=composition.listing_ids,
    )


def _composition_identity_parts(
    *,
    binding: PanelCompositionBinding,
    content: PanelContentIdentity,
    schema_hash: str,
    chunks: Sequence[PreparedPanelChunk],
    partition_origins: Mapping[str, PanelPartitionOriginRecord],
    membership: PanelMembership | None,
    listing_ids: tuple[str, ...],
) -> dict[str, object]:
    identity: dict[str, object] = {
        "kind": PanelArtifactCompositionOwner._STAGING_KIND,
        "binding": _binding_payload(binding),
        "content": _content_payload(content),
        "schema_hash": schema_hash,
        "chunks": [_chunk_payload(item) for item in chunks],
        "partition_origins": {
            key: value.to_payload() for key, value in sorted(partition_origins.items())
        },
    }
    if membership is not None:
        identity["membership"] = membership.to_payload()
    elif listing_ids:
        identity["listing_ids"] = list(listing_ids)
    return identity


def _binding_payload(binding: PanelCompositionBinding) -> dict[str, object]:
    payload = asdict(binding)
    payload["history_start"] = binding.history_start.isoformat()
    payload["as_of_session"] = binding.as_of_session.isoformat()
    payload["factor_ids"] = list(binding.factor_ids)
    return payload


def _content_payload(content: PanelContentIdentity) -> dict[str, object]:
    return {
        "panel_content_hash": content.panel_content_hash,
        "row_count": content.row_count,
        "availability_count": content.availability_count,
        "history_start": content.history_start.isoformat(),
        "as_of_session": content.as_of_session.isoformat(),
    }


def _chunk_payload(chunk: PreparedPanelChunk) -> dict[str, object]:
    if chunk.origin_binding_hash is None:
        raise ValueError("Panel composition chunk has no origin binding")
    payload = asdict(chunk)
    payload["first_session"] = chunk.first_session.isoformat()
    payload["last_session"] = chunk.last_session.isoformat()
    payload["cross_sections"] = [item.to_payload() for item in chunk.cross_sections]
    return payload


def _origin_record(value: object) -> PanelPartitionOriginRecord:
    if value is None or isinstance(value, str):
        return PanelPartitionOriginRecord(spy_revision=value)
    if isinstance(value, Mapping):
        spy = value.get("spy_revision")
        manifest = value.get("manifest_revision")
        sector = value.get("sector_revision")
        catalog = value.get("catalog_hash")
        return PanelPartitionOriginRecord(
            spy_revision=str(spy) if spy is not None else None,
            manifest_revision=str(manifest) if manifest is not None else None,
            sector_revision=str(sector) if sector is not None else None,
            catalog_hash=str(catalog) if catalog is not None else None,
        )
    raise ValueError("Panel composition staging origin entry is invalid")


def prepared_chunk(item: Mapping[str, object], *, default_origin: str) -> PreparedPanelChunk:
    """Read one recorded chunk entry; an entry without an origin is a legacy one.

    Every writer before partition reuse hashed each chunk with the snapshot's
    own binding and recorded nothing else, so an absent ``origin_binding_hash``
    *is* that binding -- a recorded fact of the sealed document, not a default.
    An entry without ``cross_sections`` is a chunk under the binding rule.
    """
    origin = item.get("origin_binding_hash")
    recorded = item.get("cross_sections") or ()
    if not isinstance(recorded, (list, tuple)):
        raise ValueError("Panel chunk cross-sections are invalid")
    return PreparedPanelChunk(
        year=int(cast(int, item["year"])),
        first_session=date.fromisoformat(str(item["first_session"])),
        last_session=date.fromisoformat(str(item["last_session"])),
        row_count=int(cast(int, item["row_count"])),
        chunk_hash=str(item["chunk_hash"]),
        metadata_hash=str(item["metadata_hash"]),
        uri=str(item["uri"]),
        origin_binding_hash=str(origin) if origin is not None else default_origin,
        cross_sections=tuple(
            PanelCrossSectionRange.from_payload(cast(Mapping[str, object], entry))
            for entry in recorded
        ),
    )


def _cross_sections(
    membership: PanelMembership, sessions: Sequence[date], history: SectorHistory
) -> tuple[PanelCrossSectionRange, ...]:
    """Each session's cross-section under the Sector map in force at it (V346), as ranges.

    A run of sessions that reads one map is identified as before; ranges that meet across a
    run's edge with one identity are one range, as one map would have made them.
    """
    ranges: list[PanelCrossSectionRange] = []
    for run_sessions, sectors in history.runs(sessions):
        for item in membership.cross_sections(run_sessions, sectors):
            previous = ranges[-1] if ranges else None
            if (
                previous is not None
                and previous.cross_section_identity == item.cross_section_identity
                and previous.member_count == item.member_count
            ):
                ranges[-1] = PanelCrossSectionRange(
                    previous.first_session,
                    item.last_session,
                    item.cross_section_identity,
                    item.member_count,
                )
            else:
                ranges.append(item)
    return tuple(ranges)


def _composition_binding_key(binding: PanelCompositionBinding) -> tuple[object, ...]:
    if binding.row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
        return (
            binding.row_identity_basis,
            binding.catalog_hash,
            binding.policy_hash,
            binding.history_start.isoformat(),
            binding.factor_ids,
        )
    return (
        binding.manifest_revision,
        binding.sector_revision,
        binding.catalog_hash,
        binding.policy_hash,
        binding.history_start.isoformat(),
        binding.factor_ids,
    )


def manifest_row_identity_basis(manifest: Mapping[str, object]) -> str:
    """The row identity rule a published snapshot manifest records.

    Absent from every manifest written before the cross-section rule, whose
    rows all bind their build's manifest and sector revisions.
    """
    summary = manifest.get("safe_summary")
    lineage = summary.get("lineage") if isinstance(summary, Mapping) else None
    if isinstance(lineage, Mapping):
        basis = lineage.get("row_identity_basis")
        if basis is not None:
            return str(basis)
    return str(PANEL_ROW_IDENTITY_BY_BINDING)


def manifest_composition_binding(manifest: Mapping[str, object]) -> tuple[object, ...] | None:
    """The composition binding a published snapshot manifest records, or None.

    Read from the sealed lineage and factor summary: the fields that decide
    whether another build's partitions describe the same Panel rows.
    """
    summary = manifest.get("safe_summary")
    if not isinstance(summary, Mapping):
        return None
    lineage = summary.get("lineage")
    factors = summary.get("factor_catalog_summary")
    if not isinstance(lineage, Mapping) or not isinstance(factors, Mapping):
        return None
    try:
        basis = manifest_row_identity_basis(manifest)
        if basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
            return (
                basis,
                str(lineage["catalog_hash"]),
                str(lineage["policy_hash"]),
                date.fromisoformat(str(manifest["history_start"])).isoformat(),
                tuple(sorted(str(value) for value in factors)),
            )
        return (
            str(lineage["manifest_revision"]),
            str(lineage["sector_revision"]),
            str(lineage["catalog_hash"]),
            str(lineage["policy_hash"]),
            date.fromisoformat(str(manifest["history_start"])).isoformat(),
            tuple(sorted(str(value) for value in factors)),
        )
    except (KeyError, ValueError):
        return None


def _manifest_catalog_hash(manifest: Mapping[str, object]) -> str | None:
    """The catalog a published snapshot manifest's lineage names, or None."""
    summary = manifest.get("safe_summary")
    lineage = summary.get("lineage") if isinstance(summary, Mapping) else None
    if not isinstance(lineage, Mapping) or lineage.get("catalog_hash") is None:
        return None
    return str(lineage["catalog_hash"])


def manifest_partition_origins(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Every build whose cells a snapshot holds: binding -> revisions, receipt.

    A manifest written before partition reuse holds cells of exactly one
    build, its own; its lineage is that origin. A later manifest records the
    table explicitly under ``lineage.partition_origins``, and one under the
    cross-section rule records each origin's manifest and sector revisions
    with its SPY revision. A manifest without a lineage at all (a bare test
    double) names only its own binding, with no revisions to report.
    """
    summary = manifest.get("safe_summary")
    lineage = summary.get("lineage") if isinstance(summary, Mapping) else None
    own = manifest.get("panel_binding_hash")
    origins: dict[str, dict[str, object]] = {}
    if not isinstance(lineage, Mapping):
        if own is not None:
            origins[str(own)] = {
                "spy_revision": None,
                "materialization_receipt_hash": None,
                "manifest_revision": None,
                "sector_revision": None,
            }
        return origins
    recorded = lineage.get("partition_origins")
    if isinstance(recorded, Mapping):
        for binding_hash, entry in recorded.items():
            if not isinstance(entry, Mapping) or "spy_revision" not in entry:
                raise ValueError("feature Panel partition origin entry is invalid")
            origins[str(binding_hash)] = {
                "spy_revision": str(entry["spy_revision"]),
                "materialization_receipt_hash": (
                    str(entry["materialization_receipt_hash"])
                    if entry.get("materialization_receipt_hash") is not None
                    else None
                ),
                # A binding-rule manifest recorded only the SPY revision: its
                # origins all share its own manifest and sector revisions.
                "manifest_revision": str(
                    entry.get("manifest_revision") or lineage["manifest_revision"]
                ),
                "sector_revision": str(entry.get("sector_revision") or lineage["sector_revision"]),
            }
            # Cells a catalog this one only adds columns to computed (V92); an
            # origin recorded without one shares the manifest's catalog.
            if entry.get("catalog_hash") is not None:
                origins[str(binding_hash)]["catalog_hash"] = str(entry["catalog_hash"])
    if own is not None:
        origins.setdefault(
            str(own),
            {
                "spy_revision": str(lineage["spy_revision"]),
                "materialization_receipt_hash": None,
                "manifest_revision": str(lineage["manifest_revision"]),
                "sector_revision": str(lineage["sector_revision"]),
            },
        )
    return origins


__all__ = [
    "PanelArtifactCompositionOwner",
    "PanelColumnExtension",
    "PanelCompositionBinding",
    "PanelCompositionSession",
    "PanelPartitionOriginRecord",
    "PreparedPanelChunk",
    "PreparedPanelComposition",
    "manifest_composition_binding",
    "manifest_partition_origins",
    "manifest_row_identity_basis",
    "panel_content_identity",
    "prepared_chunk",
]
