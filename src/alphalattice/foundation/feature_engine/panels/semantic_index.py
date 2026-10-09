"""Content-addressed row-hash index for zero-value-read Factor Research preflight."""

from __future__ import annotations

from datetime import date

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.contracts import panel_member_counts
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.kernel.shared_kernel.identity import canonical_hash

SLICE_ALGORITHM_IDENTITY = "ordered-session-row-hash-digest"


class FeaturePanelSessionSemantic(BaseModel):
    """Bind a session to its row count and ordered row-hash digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_date: date
    row_count: int = Field(ge=1)
    ordered_row_hash_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class FeaturePanelSemanticIndex(BaseModel):
    """Bind Panel content and session row hashes for bounded slice checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = "FeaturePanelSemanticIndex"
    panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    panel_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    listing_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    active_listing_count: int = Field(ge=1)
    calendar_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    slice_algorithm_identity: str = SLICE_ALGORITHM_IDENTITY
    sessions: tuple[FeaturePanelSessionSemantic, ...] = Field(min_length=1)
    index_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def slice_hash(self, sessions: tuple[date, ...]) -> str:
        """Hash the selected sessions under this Panel semantic index."""
        by_date = {item.session_date: item for item in self.sessions}
        try:
            selected = tuple(by_date[value] for value in sessions)
        except KeyError as exc:
            raise ValueError("Factor Research slice is outside the semantic index") from exc
        return canonical_hash(
            {
                "slice_algorithm_identity": self.slice_algorithm_identity,
                "catalog_hash": self.catalog_hash,
                "listing_set_hash": self.listing_set_hash,
                "sessions": [
                    {
                        "session_date": item.session_date,
                        "row_count": item.row_count,
                        "ordered_row_hash_digest": item.ordered_row_hash_digest,
                    }
                    for item in selected
                ],
            }
        )


class FeaturePanelSemanticIndexService:
    """Backfill once from immutable row hashes, then serve manifest-only preflight."""

    def __init__(self, resolver: ArtifactResolver) -> None:
        """Use the resolver and Panel reader for index backfill."""
        self.resolver = resolver
        self.reader = FeaturePanelReader(resolver)

    def obtain(self, panel_manifest_ref: str) -> tuple[FeaturePanelSemanticIndex, str, bool]:
        """Read an existing semantic index or build it from immutable row hashes."""
        manifest = self.resolver.load_feature_panel_manifest(panel_manifest_ref)
        snapshot_hash = str(manifest["snapshot_hash"])
        found = self.resolver.find_feature_panel_semantic_index(panel_snapshot_hash=snapshot_hash)
        if found is not None:
            payload, uri = found
            return FeaturePanelSemanticIndex.model_validate(payload), uri, False
        batches = list(self.reader.identity_batches(panel_manifest_ref))
        sessions: tuple[FeaturePanelSessionSemantic, ...] = ()
        if batches:
            # Each session's row hashes in listing order (ties by row hash), as one sort:
            # Arrow orders strings by their UTF-8 bytes, which is their code-point order.
            table = pa.Table.from_batches(batches).sort_by(
                [(name, "ascending") for name in ("session_date", "listing_id", "row_hash")]
            )
            days = table["session_date"].combine_chunks()
            hashes = [str(value) for value in table["row_hash"].to_pylist()]
            ordinals = np.asarray(days.to_numpy(zero_copy_only=False))
            cuts = (np.flatnonzero(ordinals[1:] != ordinals[:-1]) + 1).tolist()
            starts, stops = [0, *cuts], [*cuts, len(hashes)]
            sessions = tuple(
                FeaturePanelSessionSemantic(
                    session_date=session,
                    row_count=stop - start,
                    ordered_row_hash_digest=canonical_hash(hashes[start:stop]),
                )
                for session, start, stop in zip(
                    days.take(starts).to_pylist(), starts, stops, strict=True
                )
            )
        summary = manifest.get("safe_summary")
        lineage = summary.get("lineage") if isinstance(summary, dict) else None
        if not isinstance(lineage, dict):
            raise ValueError("feature panel semantic index has no catalog lineage")
        # A session holds one row per member. A dense Panel's members are its
        # whole axis; one with per-session membership records, per epoch, how
        # many members each session has, and its rows must say the same.
        expected_rows = panel_member_counts(manifest, tuple(item.session_date for item in sessions))
        if not sessions or any(
            item.row_count != expected_rows[item.session_date] for item in sessions
        ):
            raise ValueError("feature panel semantic index has incomplete sessions")
        identity = {
            "kind": "FeaturePanelSemanticIndex",
            "panel_snapshot_hash": snapshot_hash,
            "panel_content_hash": str(manifest["panel_content_hash"]),
            "catalog_hash": str(lineage["catalog_hash"]),
            "listing_set_hash": str(manifest["listing_set_hash"]),
            "active_listing_count": int(manifest["active_listing_count"]),
            "calendar_hash": canonical_hash([item.session_date for item in sessions]),
            "slice_algorithm_identity": SLICE_ALGORITHM_IDENTITY,
            "sessions": [item.model_dump(mode="json") for item in sessions],
        }
        index = FeaturePanelSemanticIndex(**identity, index_hash=canonical_hash(identity))
        descriptor = self.resolver.publish_feature_panel_semantic_index(
            payload=index.model_dump(mode="json"), index_hash=index.index_hash
        )
        return index, descriptor.uri, True


__all__ = [
    "SLICE_ALGORITHM_IDENTITY",
    "FeaturePanelSemanticIndex",
    "FeaturePanelSemanticIndexService",
    "FeaturePanelSessionSemantic",
]
