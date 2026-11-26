"""Content-addressed Dynamic Panel blocks, fold context and dossiers."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.alpha_research.inputs.dynamic_panel import (
    DynamicPanelFoldDossier,
    DynamicPanelScaleReceipt,
    dynamic_panel_array_identity,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .dynamic_panel_campaign import (
    DynamicPanelModelCampaignDossier,
    DynamicPanelModelProgram,
)

_HASH = r"^[0-9a-f]{64}$"
_SURFACE_CATEGORY = "development/dynamic-panel/surfaces"
_SCALE_CATEGORY = "development/dynamic-panel/scale-receipts"
_DOSSIER_CATEGORY = "development/dynamic-panel/fold-dossiers"
_FOLD_CONTEXT_CATEGORY = "development/dynamic-panel/fold-context"
_MODEL_PROGRAM_CATEGORY = "development/dynamic-panel/model/programs"
_MODEL_DOSSIER_CATEGORY = "development/dynamic-panel/model/dossiers"


class DynamicPanelArtifactError(ValueError):
    """A durable Dynamic Panel child cannot be proven exact."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


def _contract[T: _Contract](model: type[T], values: dict[str, object], field: str) -> T:
    provisional = model.model_construct(**values, **{field: "0" * 64})
    identity = provisional.model_dump(mode="json", exclude={field})
    return model(**values, **{field: str(canonical_hash(identity))})


class DynamicPanelBlockArtifact(_Contract):
    kind: Literal["DynamicPanelBlockArtifact"] = "DynamicPanelBlockArtifact"
    block_id: str
    storage_mode: Literal[
        "SOURCE_REFERENCE",
        "PROCESSED_CONTINUOUS_PARQUET",
        "RAW_CONTEXT_PARQUET",
        "DISCRETE_PARQUET",
    ]
    ordered_feature_ids: tuple[str, ...] = Field(min_length=1)
    array_identity: str = Field(pattern=_HASH)
    parquet_relative_path: str | None = None
    parquet_sha256: str | None = Field(default=None, pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_storage(self) -> Self:
        if (self.storage_mode == "SOURCE_REFERENCE") != (
            self.parquet_relative_path is None and self.parquet_sha256 is None
        ):
            raise DynamicPanelArtifactError("alpha_research.dynamic_panel_block_storage_invalid")
        return self


class DynamicPanelSurfaceManifest(_Contract):
    """One global role block surface; fold-only context remains a child."""

    kind: Literal["DynamicPanelSurfaceManifest"] = "DynamicPanelSurfaceManifest"
    scope: Literal["DEVELOPMENT_ONLY_NO_HOLDOUT"] = "DEVELOPMENT_ONLY_NO_HOLDOUT"
    catalog_hash: str = Field(pattern=_HASH)
    implementation_source_closure_hash: str = Field(pattern=_HASH)
    relative_surface_hash: str = Field(pattern=_HASH)
    raw_formula_surface_hash: str = Field(pattern=_HASH)
    total_return_target_evidence_hash: str = Field(pattern=_HASH)
    outcome_method_binding_hash: str = Field(pattern=_HASH)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=2)
    ordered_factor_ids: tuple[str, ...] = Field(min_length=1)
    ordered_relative_factor_ids: tuple[str, ...] = Field(min_length=1)
    ordered_blocks: tuple[DynamicPanelBlockArtifact, ...] = Field(min_length=8)
    scale_receipt_hashes: tuple[str, ...] = Field(min_length=4)
    row_count: int = Field(ge=1)
    manifest_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        return _contract(cls, dict(values), "manifest_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if (
            self.row_count != len(self.formation_sessions) * len(self.ordered_listing_ids)
            or tuple(value.block_id for value in self.ordered_blocks)
            != tuple(dict.fromkeys(value.block_id for value in self.ordered_blocks))
            or self.manifest_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"manifest_hash"}))
        ):
            raise DynamicPanelArtifactError("alpha_research.dynamic_panel_surface_manifest_invalid")
        return self


class DynamicPanelFoldContextManifest(_Contract):
    """Processed fold-fitted context arrays and their common view axes."""

    kind: Literal["DynamicPanelFoldContextManifest"] = "DynamicPanelFoldContextManifest"
    scope: Literal["DEVELOPMENT_ONLY_NO_HOLDOUT"] = "DEVELOPMENT_ONLY_NO_HOLDOUT"
    surface_manifest_hash: str = Field(pattern=_HASH)
    catalog_hash: str = Field(pattern=_HASH)
    fold_index: int = Field(ge=0)
    training_sessions: tuple[date, ...] = Field(min_length=1)
    validation_sessions: tuple[date, ...] = Field(min_length=1)
    common_training_row_axis_hash: str = Field(pattern=_HASH)
    common_validation_row_axis_hash: str = Field(pattern=_HASH)
    ordered_context_blocks: tuple[DynamicPanelBlockArtifact, ...] = Field(
        min_length=3, max_length=3
    )
    scale_receipt_hashes: tuple[str, ...] = Field(min_length=3, max_length=3)
    dossier_hashes: tuple[str, ...] = Field(min_length=4, max_length=4)
    manifest_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        return _contract(cls, dict(values), "manifest_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if (
            set(self.training_sessions) & set(self.validation_sessions)
            or tuple(value.block_id for value in self.ordered_context_blocks)
            != (
                "sector_context",
                "market_context",
                "absolute_market_states",
            )
            or self.manifest_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"manifest_hash"}))
        ):
            raise DynamicPanelArtifactError(
                "alpha_research.dynamic_panel_fold_context_manifest_invalid"
            )
        return self


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _table_values(table: pa.Table, feature_ids: tuple[str, ...]) -> np.ndarray:
    values = np.column_stack(
        [
            np.asarray(table[feature_id].to_numpy(zero_copy_only=False), dtype=np.float64)
            for feature_id in feature_ids
        ]
    )
    return np.ascontiguousarray(values, dtype=np.float64)


class DynamicPanelArtifactStore:
    """Marker-last publication and exact readback for Dynamic Panel evidence."""

    _PREFIX = "playpen://alpha-research/"

    def __init__(self, artifact_root: Path) -> None:
        self.root = artifact_root.resolve() / "alpha-research"
        self._surface_cache: dict[str, DynamicPanelSurfaceManifest] = {}

    @classmethod
    def uri(cls, category: str, content_hash: str) -> str:
        return f"{cls._PREFIX}{category}/{content_hash}"

    @staticmethod
    def _require_hash(value: str) -> None:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise DynamicPanelArtifactError("alpha_research.dynamic_panel_artifact_hash_invalid")

    def _json_path(self, category: str, content_hash: str) -> Path:
        self._require_hash(content_hash)
        return self.root / category / f"{content_hash}.json"

    def _load_json(
        self,
        *,
        category: str,
        content_hash: str,
        identity_field: str,
        model: type[BaseModel],
    ) -> BaseModel:
        path = self._json_path(category, content_hash)
        if not path.is_file():
            raise FileNotFoundError("Dynamic Panel JSON artifact is missing")
        value = model.model_validate_json(path.read_text(encoding="utf-8"))
        if getattr(value, identity_field) != content_hash:
            raise DynamicPanelArtifactError("alpha_research.dynamic_panel_json_identity_invalid")
        return value

    def load_scale_receipt(self, content_hash: str) -> DynamicPanelScaleReceipt:
        return cast(
            DynamicPanelScaleReceipt,
            self._load_json(
                category=_SCALE_CATEGORY,
                content_hash=content_hash,
                identity_field="receipt_hash",
                model=DynamicPanelScaleReceipt,
            ),
        )

    def load_dossier(self, content_hash: str) -> DynamicPanelFoldDossier:
        return cast(
            DynamicPanelFoldDossier,
            self._load_json(
                category=_DOSSIER_CATEGORY,
                content_hash=content_hash,
                identity_field="dossier_hash",
                model=DynamicPanelFoldDossier,
            ),
        )

    def load_surface_manifest(self, content_hash: str) -> DynamicPanelSurfaceManifest:
        self._require_hash(content_hash)
        cached = self._surface_cache.get(content_hash)
        if cached is not None:
            return cached
        directory = self.root / _SURFACE_CATEGORY / content_hash
        path = directory / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError("Dynamic Panel surface manifest is missing")
        manifest = DynamicPanelSurfaceManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.manifest_hash != content_hash:
            raise DynamicPanelArtifactError("alpha_research.dynamic_panel_surface_identity_invalid")
        for artifact in manifest.ordered_blocks:
            if artifact.storage_mode == "SOURCE_REFERENCE":
                continue
            assert artifact.parquet_relative_path is not None
            assert artifact.parquet_sha256 is not None
            parquet = directory / artifact.parquet_relative_path
            if not parquet.is_file() or _sha256_file(parquet) != artifact.parquet_sha256:
                raise DynamicPanelArtifactError(
                    "alpha_research.dynamic_panel_block_parquet_invalid"
                )
            table = pq.read_table(parquet)
            values = _table_values(table, artifact.ordered_feature_ids).reshape(
                len(manifest.formation_sessions),
                len(manifest.ordered_listing_ids),
                len(artifact.ordered_feature_ids),
            )
            if dynamic_panel_array_identity(values) != artifact.array_identity:
                raise DynamicPanelArtifactError("alpha_research.dynamic_panel_block_array_invalid")
        self._surface_cache[content_hash] = manifest
        return manifest

    def load_fold_context_manifest(self, content_hash: str) -> DynamicPanelFoldContextManifest:
        self._require_hash(content_hash)
        directory = self.root / _FOLD_CONTEXT_CATEGORY / content_hash
        path = directory / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError("Dynamic Panel fold context is missing")
        manifest = DynamicPanelFoldContextManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if manifest.manifest_hash != content_hash:
            raise DynamicPanelArtifactError(
                "alpha_research.dynamic_panel_fold_context_identity_invalid"
            )
        session_count = len(manifest.training_sessions) + len(manifest.validation_sessions)
        surface = self.load_surface_manifest(manifest.surface_manifest_hash)
        for artifact in manifest.ordered_context_blocks:
            assert artifact.parquet_relative_path is not None
            assert artifact.parquet_sha256 is not None
            parquet = directory / artifact.parquet_relative_path
            if not parquet.is_file() or _sha256_file(parquet) != artifact.parquet_sha256:
                raise DynamicPanelArtifactError(
                    "alpha_research.dynamic_panel_fold_context_parquet_invalid"
                )
            values = _table_values(pq.read_table(parquet), artifact.ordered_feature_ids).reshape(
                session_count,
                len(surface.ordered_listing_ids),
                len(artifact.ordered_feature_ids),
            )
            if dynamic_panel_array_identity(values) != artifact.array_identity:
                raise DynamicPanelArtifactError(
                    "alpha_research.dynamic_panel_fold_context_array_invalid"
                )
        for receipt_hash in manifest.scale_receipt_hashes:
            self.load_scale_receipt(receipt_hash)
        for dossier_hash in manifest.dossier_hashes:
            self.load_dossier(dossier_hash)
        return manifest

    def load_model_program(self, content_hash: str) -> DynamicPanelModelProgram:
        return cast(
            DynamicPanelModelProgram,
            self._load_json(
                category=_MODEL_PROGRAM_CATEGORY,
                content_hash=content_hash,
                identity_field="program_hash",
                model=DynamicPanelModelProgram,
            ),
        )

    def load_model_dossier(self, content_hash: str) -> DynamicPanelModelCampaignDossier:
        return cast(
            DynamicPanelModelCampaignDossier,
            self._load_json(
                category=_MODEL_DOSSIER_CATEGORY,
                content_hash=content_hash,
                identity_field="dossier_hash",
                model=DynamicPanelModelCampaignDossier,
            ),
        )


__all__ = [
    "DynamicPanelArtifactError",
    "DynamicPanelArtifactStore",
    "DynamicPanelBlockArtifact",
    "DynamicPanelFoldContextManifest",
    "DynamicPanelSurfaceManifest",
]
