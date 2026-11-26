"""The development feature input: a study's reader over a sealed overlay and its base Panel.

What a Factor or Alpha study reads when its input carries development-only columns: the
base Panel's reader and the overlay's sealed values, joined row for row, and the composed
identities that bind both by content. Nothing here computes a Factor: the overlay is the
Feature engine's sealed result (``development_overlay``), and a study binds its content,
so a factor's formula is no entry of a study's identity (UC, LAWS.md ID8).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.reader import (
    FeaturePanelReader,
    FeaturePanelReadRequest,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

DEVELOPMENT_FEATURE_OVERLAY_CATEGORY = "development/feature-overlay"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


def development_feature_snapshot_hash(
    *,
    base_panel_snapshot_hash: str,
    overlay_hash: str,
    control_factor_ids: tuple[str, ...],
) -> str:
    """The composite development surface identity, derivable without a reader."""
    return str(
        canonical_hash(
            {
                "base_panel_snapshot_hash": base_panel_snapshot_hash,
                "overlay_hash": overlay_hash,
                "control_factor_ids": control_factor_ids,
            }
        )
    )


class BaseFeatureReader(Protocol):
    """What a composed input reads its base Panel through."""

    def batches(self, request: FeaturePanelReadRequest) -> Iterator[pa.RecordBatch]:
        """The requested rows and columns, in record batches."""
        ...

    def available_sessions(self, manifest_ref: str) -> Sequence[date]:
        """The sessions the manifest covers, in order."""
        ...

    def listing_ids(self, manifest_ref: str) -> tuple[str, ...]:
        """The manifest's listing axis, in order."""
        ...


def sha256_file(path: Path) -> str:
    """The SHA-256 of a file's bytes, read in blocks."""
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DevelopmentFeatureOverlayColumn(_Contract):
    """One development-only column of an overlay and the identities that produced it."""

    factor_id: str
    formula_specification_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    admission_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_recipe_id: str
    preprocessing_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_implementation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_child_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_child_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    market_state_child_identity: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class DevelopmentFeatureOverlayManifest(_Contract):
    """A sealed overlay: its base Panel, its axes, its columns and the parquet that holds them."""

    kind: Literal["DevelopmentFeatureOverlayManifest"] = "DevelopmentFeatureOverlayManifest"
    scope: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    base_panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_panel_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    installed_kernel_capability_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    factor_catalog_revision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_session_axis: tuple[str, ...] = Field(min_length=1)
    ordered_listing_axis: tuple[str, ...] = Field(min_length=1)
    ordered_candidate_axis: tuple[str, ...] = Field(min_length=1)
    columns: tuple[DevelopmentFeatureOverlayColumn, ...] = Field(min_length=1)
    parquet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parquet_relative_path: str
    raw_parquet_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    raw_parquet_relative_path: str | None = None
    raw_column_references: tuple[tuple[str, str], ...] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    preparation_admission_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda value: value is None
    )
    row_count: int = Field(ge=1)
    overlay_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Refuse an overlay whose axes, references or content hash disagree."""
        if tuple(item.factor_id for item in self.columns) != self.ordered_candidate_axis:
            raise ValueError("FEATURE_OVERLAY_COLUMN_AXIS_MISMATCH")
        if (self.raw_parquet_sha256 is None) != (self.raw_parquet_relative_path is None):
            raise ValueError("FEATURE_OVERLAY_RAW_PARQUET_INCOMPLETE")
        if self.raw_column_references is not None and (
            tuple(name for name, _ in self.raw_column_references) != self.ordered_candidate_axis
            or self.raw_parquet_relative_path is not None
            or self.preparation_admission_hash is None
            or self.parquet_relative_path != "values.parquet"
            or self.row_count != len(self.ordered_session_axis) * len(self.ordered_listing_axis)
        ):
            raise ValueError("FEATURE_OVERLAY_PREPARED_REFERENCES_INVALID")
        if (self.raw_column_references is None) != (self.preparation_admission_hash is None):
            raise ValueError("FEATURE_OVERLAY_PREPARATION_ADMISSION_INCOMPLETE")
        if self.overlay_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"overlay_hash"}, exclude_none=True)
        ):
            raise ValueError("FEATURE_OVERLAY_IDENTITY_INVALID")
        return self


class CompositeDevelopmentFeatureReader:
    """Read the immutable base Panel and overlay as one research-only surface."""

    def __init__(
        self,
        *,
        base_reader: BaseFeatureReader,
        base_manifest_ref: str,
        overlay_manifest: DevelopmentFeatureOverlayManifest,
        overlay_root: Path,
        control_factor_ids: tuple[str, ...],
    ) -> None:
        """Open the overlay's parquet, refusing one whose bytes are not the manifest's."""
        self._base_reader = base_reader
        self._base_manifest_ref = base_manifest_ref
        self._listing_ids: tuple[str, ...] | None = None
        self._overlay = overlay_manifest
        self._control_factor_ids = control_factor_ids
        self._parquet_path = (
            overlay_root
            / DEVELOPMENT_FEATURE_OVERLAY_CATEGORY
            / overlay_manifest.overlay_hash
            / overlay_manifest.parquet_relative_path
        )
        if sha256_file(self._parquet_path) != overlay_manifest.parquet_sha256:
            raise ValueError("FEATURE_OVERLAY_PARQUET_IDENTITY_INVALID")
        self.manifest_ref = (
            f"playpen://feature-engine/{DEVELOPMENT_FEATURE_OVERLAY_CATEGORY}/"
            f"{overlay_manifest.overlay_hash}"
        )
        self.snapshot_hash = development_feature_snapshot_hash(
            base_panel_snapshot_hash=overlay_manifest.base_panel_snapshot_hash,
            overlay_hash=overlay_manifest.overlay_hash,
            control_factor_ids=control_factor_ids,
        )

    @property
    def ordered_factor_ids(self) -> tuple[str, ...]:
        """The inherited columns and the overlay's, sorted."""
        return tuple(sorted((*self._control_factor_ids, *self._overlay.ordered_candidate_axis)))

    def listing_ids(self, manifest_ref: str | None = None) -> tuple[str, ...]:
        """The base Panel's listing axis, which this surface covers row for row.

        What an exploration sample is drawn from (binding plan, B17).
        """
        del manifest_ref
        if self._listing_ids is None:
            self._listing_ids = tuple(self._base_reader.listing_ids(self._base_manifest_ref))
        return self._listing_ids

    def panel_manifest(self) -> dict[str, object]:
        """The composed surface's manifest: its snapshot and development-only columns."""
        return {
            "snapshot_hash": self.snapshot_hash,
            "safe_summary": {
                "factor_catalog_summary": {
                    factor_id: {"development_only": True} for factor_id in self.ordered_factor_ids
                }
            },
        }

    def available_sessions(self, manifest_ref: str) -> tuple[date, ...]:
        """The overlay's sessions, for this surface's manifest reference only."""
        if manifest_ref != self.manifest_ref:
            raise ValueError("FEATURE_OVERLAY_MANIFEST_REF_MISMATCH")
        return tuple(date.fromisoformat(value) for value in self._overlay.ordered_session_axis)

    def batches(self, request: FeaturePanelReadRequest) -> Iterator[pa.RecordBatch]:
        """The base rows joined row for row with the overlay's requested columns."""
        if request.manifest_ref != self.manifest_ref or not set(request.factor_columns).issubset(
            self.ordered_factor_ids
        ):
            raise ValueError("FEATURE_OVERLAY_READ_AUTHORITY_MISMATCH")
        controls = tuple(
            value for value in request.factor_columns if value in self._control_factor_ids
        )
        base_columns = controls or (self._control_factor_ids[0],)
        base_request = FeaturePanelReadRequest(
            manifest_ref=self._base_manifest_ref,
            start_session=request.start_session,
            end_session=request.end_session,
            exact_sessions=request.exact_sessions,
            factor_columns=tuple(sorted(base_columns)),
            include_row_hash=request.include_row_hash,
            batch_size=request.batch_size,
            batch_readahead=request.batch_readahead,
            fragment_readahead=request.fragment_readahead,
            use_threads=request.use_threads,
        )
        if len(controls) == len(request.factor_columns):
            # This consumer selected only inherited columns. Its verified input
            # still binds the overlay, but no join or table conversion is needed.
            yield from self._base_reader.batches(base_request)
            return
        base_batches = tuple(self._base_reader.batches(base_request))
        if not base_batches:
            raise ValueError("FEATURE_OVERLAY_BASE_ROWS_MISSING")
        base = pa.Table.from_batches(base_batches).to_pandas()
        if not controls:
            base = base.drop(columns=[self._control_factor_ids[0]])
        overlay = pd.read_parquet(self._parquet_path)
        overlay["session_date"] = pd.to_datetime(overlay["session_date"]).dt.date
        sessions = set(request.exact_sessions or self.available_sessions(self.manifest_ref))
        overlay = overlay.loc[
            overlay["session_date"].isin(sessions),
            [
                "session_date",
                "listing_id",
                *(
                    value
                    for value in request.factor_columns
                    if value in self._overlay.ordered_candidate_axis
                ),
            ],
        ]
        joined = base.merge(
            overlay,
            on=["session_date", "listing_id"],
            how="left",
            validate="one_to_one",
        )
        ordered_columns = [
            "session_date",
            "listing_id",
            *request.factor_columns,
            *(["row_hash"] if request.include_row_hash else []),
        ]
        table = pa.Table.from_pandas(joined.loc[:, ordered_columns], preserve_index=False)
        yield from table.to_batches(max_chunksize=request.batch_size)


@dataclass(frozen=True)
class ResolvedDevelopmentFeatureInput:
    """A call-local composed source, never a counterfeit physical Panel manifest.

    Host verifies the parent input and preparation before constructing this value.
    Its consumers share one reader and one projection. Composed logical identities
    bind the parent's logical identity plus the exact retained/additional columns;
    they are not published into the normal Panel mapping or current pointer.
    """

    source_handle: str
    input_binding_hash: str
    base_panel_snapshot_hash: str
    manifest_ref: str
    panel_manifest: dict[str, Any]
    logical_panel_hash: str
    logical_semantic_index_hash: str
    reader: BaseFeatureReader
    definition_plan_hash: str | None = None

    @classmethod
    def compose(
        cls,
        *,
        source_handle: str,
        input_binding_hash: str,
        base_manifest: dict[str, Any],
        base_artifact_root: Path,
        base_logical_panel_hash: str,
        base_logical_semantic_index_hash: str,
        inherited_factor_ids: tuple[str, ...],
        overlay: DevelopmentFeatureOverlayManifest | None,
        overlay_root: Path,
        definition_plan_hash: str | None = None,
    ) -> ResolvedDevelopmentFeatureInput:
        """Compose the base Panel and an optional overlay into one verified input."""
        base_hash = str(base_manifest["snapshot_hash"])
        base_ref = ArtifactResolver.feature_panel_manifest_uri(base_hash)
        base_catalog = base_manifest["safe_summary"]["factor_catalog_summary"]
        if inherited_factor_ids != tuple(sorted(set(inherited_factor_ids))) or not set(
            inherited_factor_ids
        ).issubset(base_catalog):
            raise ValueError("FEATURE_RESEARCH_INHERITED_AXIS_INVALID")
        reader: BaseFeatureReader = FeaturePanelReader(ArtifactResolver(base_artifact_root))
        selected = {name: dict(base_catalog[name]) for name in inherited_factor_ids}
        snapshot_hash, manifest_ref = base_hash, base_ref
        if overlay is not None:
            if overlay.base_panel_snapshot_hash != base_hash or set(
                overlay.ordered_candidate_axis
            ) & set(inherited_factor_ids):
                raise ValueError("FEATURE_RESEARCH_OVERLAY_BASE_MISMATCH")
            composite = CompositeDevelopmentFeatureReader(
                base_reader=reader,
                base_manifest_ref=base_ref,
                overlay_manifest=overlay,
                overlay_root=overlay_root,
                control_factor_ids=inherited_factor_ids,
            )
            reader, snapshot_hash, manifest_ref = (
                composite,
                composite.snapshot_hash,
                composite.manifest_ref,
            )
            for column in overlay.columns:
                selected[column.factor_id] = {
                    "implementation_hash": column.implementation_hash,
                    "methodology_hash": canonical_hash(
                        {
                            "formula_specification_hash": column.formula_specification_hash,
                            "preprocessing_recipe_hash": column.preprocessing_recipe_hash,
                            "preprocessing_implementation_hash": (
                                column.preprocessing_implementation_hash
                            ),
                        }
                    ),
                    "development_only": True,
                }
        if not selected:
            raise ValueError("FEATURE_RESEARCH_EMPTY_AXIS")
        composed = overlay is not None or set(selected) != set(base_catalog)
        identity = {
            "inherited_factor_ids": inherited_factor_ids,
            "overlay_hash": None if overlay is None else overlay.overlay_hash,
        }
        if overlay is None and composed:
            snapshot_hash = str(canonical_hash({"base_snapshot": base_hash, **identity}))
        logical = (
            str(canonical_hash({"base_logical_panel": base_logical_panel_hash, **identity}))
            if composed
            else base_logical_panel_hash
        )
        semantic = (
            str(
                canonical_hash(
                    {"base_semantic_index": base_logical_semantic_index_hash, **identity}
                )
            )
            if composed
            else base_logical_semantic_index_hash
        )
        # Copy only descriptive lineage, never the base Panel's physical parts.
        panel = {
            "kind": "DevelopmentFeatureInput",
            "snapshot_hash": snapshot_hash,
            "base_panel_snapshot_hash": base_hash,
            "panel_content_hash": logical,
            "temporal_identity_hash": semantic,
            **{
                key: base_manifest[key]
                for key in (
                    "listing_set_hash",
                    "knowledge_cutoff_at",
                    "history_start",
                    "as_of_session",
                )
                if key in base_manifest
            },
            "safe_summary": {
                **{
                    key: base_manifest["safe_summary"][key]
                    for key in ("lineage", "membership")
                    if key in base_manifest["safe_summary"]
                },
                "factor_catalog_summary": dict(sorted(selected.items())),
            },
        }
        return cls(
            source_handle,
            input_binding_hash,
            base_hash,
            manifest_ref,
            panel,
            logical,
            semantic,
            reader,
            definition_plan_hash,
        )


__all__ = [
    "DEVELOPMENT_FEATURE_OVERLAY_CATEGORY",
    "BaseFeatureReader",
    "CompositeDevelopmentFeatureReader",
    "DevelopmentFeatureOverlayColumn",
    "DevelopmentFeatureOverlayManifest",
    "ResolvedDevelopmentFeatureInput",
    "development_feature_snapshot_hash",
    "sha256_file",
]
