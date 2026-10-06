"""Selective content-addressed Feature overlays for development-only candidates."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, Self, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.catalog.research_values import ResearchFormulaValues
from alphalattice.foundation.feature_engine.contracts import FeaturePanelBinding
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.development_input import (
    DEVELOPMENT_FEATURE_OVERLAY_CATEGORY,
    DevelopmentFeatureOverlayColumn,
    DevelopmentFeatureOverlayManifest,
    sha256_file,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
    extension_factor_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.interactions import (
    interaction_market_state_children,
)
from alphalattice.foundation.feature_engine.producers.factors.specifications import (
    FactorDevelopmentAdmissionReceipt,
    FactorFormulaSpecification,
    admit_factor_development_capabilities,
    build_installed_factor_development_capabilities,
    build_installed_factor_formula_specifications,
    build_research_formula_specification,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.adapters import (
    PanelPreprocessingAdapter,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.catalog import (
    build_installed_panel_preprocessing_catalog,
)
from alphalattice.kernel.quant.factor_contracts import FactorSpec
from alphalattice.kernel.quant.sector_history import reclassification_payload, sector_slices
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.persistence import DURABLE_REPLACE_DELAYS, replace_with_retry

DEVELOPMENT_RAW_FORMULA_SURFACE_CATEGORY = "development/raw-formula-surface"
DEVELOPMENT_METHODOLOGY_SURFACE_CATEGORY = "development/feature-methodology-surface"


def prepared_overlay_source_identity(
    *,
    input_binding_hash: str,
    base_panel_snapshot_hash: str,
    raw_column_references: tuple[tuple[str, str], ...],
    sessions: tuple[date, ...],
    sector_by_listing_id: Mapping[str, str],
    members_by_session: Mapping[date, Sequence[str]],
    source_exclusions_by_session: Mapping[date, Sequence[str]],
) -> str:
    """The exact numerical source context, separately from local names and notes."""
    return str(
        canonical_hash(
            {
                "input_binding_hash": input_binding_hash,
                "raw_columns": sorted({identity for _, identity in raw_column_references}),
                "base_panel_snapshot_hash": base_panel_snapshot_hash,
                "sector_labels": dict(sector_by_listing_id),
                **reclassification_payload(sector_by_listing_id),
                "members": [(str(d), tuple(members_by_session[d])) for d in sessions],
                "excluded_sources": [
                    (str(d), tuple(source_exclusions_by_session.get(d, ()))) for d in sessions
                ],
            }
        )
    )


def _array_identity(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64)
    return str(
        canonical_hash(
            {
                "shape": list(array.shape),
                "values": [
                    None if not np.isfinite(value) else float(value).hex() for value in array
                ],
            }
        )
    )


def _raw_formula_rows(
    source_rows: pd.DataFrame,
    base_rows: pd.DataFrame,
    recipes: tuple[FactorSpec, ...],
    compute: Callable[[pd.DataFrame, FactorSpec], pd.Series],
    *,
    missing_code: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the same Formula inputs for raw checkpoints and campaign overlays."""
    source = source_rows.copy()
    join = ["session_date", "listing_id"]
    if "rev_5" not in source and "rev_5" in base_rows:
        source = source.merge(
            base_rows.loc[:, [*join, "rev_5"]], on=join, how="left", validate="one_to_one"
        )
    missing = sorted(
        {field for recipe in recipes for field in recipe.required_fields} - set(source.columns)
    )
    if missing:
        raise ValueError(f"{missing_code}:{','.join(missing)}")
    raw = source.loc[:, join].copy()
    for recipe in recipes:
        raw[recipe.factor_id] = compute(source, recipe).to_numpy(dtype=float)
    return source, raw


def _materialize_by_run(
    adapter: PanelPreprocessingAdapter,
    *,
    feature_rows: pd.DataFrame,
    sessions: tuple[date, ...],
    sector_by_listing_id: Mapping[str, str],
    members_by_session: Mapping[date, Sequence[str]] | None = None,
    source_exclusions_by_session: Mapping[date, Sequence[str]] | None = None,
    **arguments: Any,
) -> tuple[pd.DataFrame, str]:
    """Preprocess the rows one run of sessions at a time, each with the map it reads (V346).

    Preprocessing is cross-sectional, one session at a time, so a run is computed as the whole
    axis was. One run is one call over every row, as before; several are one call each, their
    rows concatenated and their child identities hashed in order.

    Args:
        adapter: The role's installed preprocessing.
        feature_rows: The raw rows, `session_date` and `listing_id` keyed.
        sessions: The rows' session axis, ascending.
        sector_by_listing_id: A `SectorHistory`, or a map read by every session.
        members_by_session: Each session's members, when the caller names them.
        source_exclusions_by_session: Each session's excluded sources, when named.
        **arguments: The adapter's other arguments, passed through.

    Returns:
        The transformed rows and their child identity.
    """
    runs = sector_slices(sector_by_listing_id, sessions)
    if len(runs) == 1:
        result = adapter.materialize(
            feature_rows=feature_rows,
            sector_by_listing_id=dict(runs[0][1]),
            members_by_session=members_by_session,
            source_exclusions_by_session=source_exclusions_by_session,
            **arguments,
        )
        return result.rows, result.transformed_identity
    days = pd.to_datetime(feature_rows["session_date"]).dt.date
    parts: list[pd.DataFrame] = []
    identities: list[str] = []
    for rows, mapping in runs:
        run_sessions = sessions[rows]
        result = adapter.materialize(
            feature_rows=feature_rows.loc[days.isin(set(run_sessions))].reset_index(drop=True),
            sector_by_listing_id=dict(mapping),
            members_by_session=(
                None
                if members_by_session is None
                else {day: members_by_session[day] for day in run_sessions}
            ),
            source_exclusions_by_session=(
                None
                if source_exclusions_by_session is None
                else {day: source_exclusions_by_session.get(day, ()) for day in run_sessions}
            ),
            **arguments,
        )
        parts.append(pd.DataFrame(result.rows))
        identities.append(result.transformed_identity)
    identity = str(canonical_hash({"kind": "SectorRunPreprocessing", "runs": identities}))
    return pd.concat(parts, ignore_index=True), identity


def _merge_preprocessing(
    transformed: pd.DataFrame,
    rows: pd.DataFrame,
    factor_ids: Sequence[str],
    *,
    normalize_base_dates: bool = False,
) -> pd.DataFrame:
    join = ["session_date", "listing_id"]
    values = rows.loc[:, [*join, *factor_ids]].copy()
    values["session_date"] = pd.to_datetime(values["session_date"]).dt.date
    if normalize_base_dates:
        transformed["session_date"] = pd.to_datetime(transformed["session_date"]).dt.date
    return transformed.merge(values, on=join, how="left", validate="one_to_one")


def _write_parquet(path: Path, rows: pd.DataFrame) -> str:
    staged = path.with_name(f".{path.name}.tmp")
    rows.to_parquet(staged, index=False, compression="zstd")
    os.replace(staged, path)
    return sha256_file(path)


def _write_manifest(path: Path, manifest: BaseModel) -> None:
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(staged, path)


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


def _overlay_column(
    *,
    factor_id: str,
    formula_specification_hash: str,
    implementation_hash: str,
    admission_receipt_hash: str,
    preprocessing_recipe_id: str,
    preprocessing_recipe_hash: str,
    preprocessing_implementation_hash: str,
    raw_values: np.ndarray,
    preprocessing_child_identity: str,
    market_state_values: np.ndarray | None = None,
) -> DevelopmentFeatureOverlayColumn:
    """Seal common column evidence; each caller owns its admission and raw source."""
    return DevelopmentFeatureOverlayColumn(
        factor_id=factor_id,
        formula_specification_hash=formula_specification_hash,
        implementation_hash=implementation_hash,
        admission_receipt_hash=admission_receipt_hash,
        preprocessing_recipe_id=preprocessing_recipe_id,
        preprocessing_recipe_hash=preprocessing_recipe_hash,
        preprocessing_implementation_hash=preprocessing_implementation_hash,
        raw_child_identity=_array_identity(raw_values),
        preprocessing_child_identity=preprocessing_child_identity,
        market_state_child_identity=(
            _array_identity(market_state_values) if market_state_values is not None else None
        ),
    )


def _overlay_manifest(
    *,
    base_panel_snapshot_hash: str,
    panel_binding: FeaturePanelBinding,
    source_identity: str,
    installed_kernel_capability_hash: str,
    factor_catalog_revision_hash: str,
    preprocessing_catalog_hash: str,
    sessions: Sequence[str],
    listing_ids: Sequence[str],
    candidate_ids: Sequence[str],
    columns: Sequence[DevelopmentFeatureOverlayColumn],
    parquet_sha256: str,
    row_count: int,
    raw_parquet_sha256: str | None = None,
    raw_column_references: tuple[tuple[str, str], ...] | None = None,
    preparation_admission_hash: str | None = None,
) -> DevelopmentFeatureOverlayManifest:
    """Keep common overlay identity fields in one owner; publication stays caller-specific."""
    values: dict[str, Any] = {
        "kind": "DevelopmentFeatureOverlayManifest",
        "scope": "DEVELOPMENT_ONLY",
        "base_panel_snapshot_hash": base_panel_snapshot_hash,
        "base_panel_binding_hash": panel_binding.panel_binding_hash,
        "source_identity": source_identity,
        "installed_kernel_capability_hash": installed_kernel_capability_hash,
        "factor_catalog_revision_hash": factor_catalog_revision_hash,
        "preprocessing_catalog_hash": preprocessing_catalog_hash,
        "ordered_session_axis": list(sessions),
        "ordered_listing_axis": list(listing_ids),
        "ordered_candidate_axis": list(candidate_ids),
        "columns": [
            item.model_dump(mode="json", exclude_none=True)
            for item in sorted(columns, key=lambda item: item.factor_id)
        ],
        "parquet_sha256": parquet_sha256,
        "parquet_relative_path": "values.parquet",
        "row_count": row_count,
    }
    if raw_parquet_sha256 is not None:
        values.update(
            raw_parquet_sha256=raw_parquet_sha256,
            raw_parquet_relative_path="raw_values.parquet",
        )
    if raw_column_references is not None:
        values.update(
            raw_column_references=raw_column_references,
            preparation_admission_hash=preparation_admission_hash,
        )
    return DevelopmentFeatureOverlayManifest(**values, overlay_hash=str(canonical_hash(values)))


class PreparedFeatureOverlayAdmission(_Contract):
    """The exact local definitions and verified raw children admitted to preprocessing."""

    kind: Literal["PreparedFeatureOverlayAdmission"] = "PreparedFeatureOverlayAdmission"
    scope: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    input_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_definitions: tuple[tuple[str, str], ...] = Field(min_length=1)
    specifications: tuple[FactorFormulaSpecification, ...] = Field(min_length=1)
    raw_column_references: tuple[tuple[str, str], ...] = Field(min_length=1)
    admission_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_admission(self) -> Self:
        axis = tuple(name for name, _ in self.numerical_definitions)
        if (
            axis != tuple(sorted(set(axis)))
            or axis != tuple(item.factor_id for item in self.specifications)
            or axis != tuple(name for name, _ in self.raw_column_references)
            or self.admission_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"admission_hash"}))
        ):
            raise ValueError("FEATURE_OVERLAY_PREPARED_ADMISSION_INVALID")
        return self


def read_prepared_feature_overlay_admission(
    output_root: Path, manifest: DevelopmentFeatureOverlayManifest
) -> PreparedFeatureOverlayAdmission:
    """Load and verify the prepared admission bound to an overlay."""
    store = PanelClosureArtifactStore(ArtifactResolver(output_root))
    if manifest.preparation_admission_hash is None:
        raise ValueError("FEATURE_OVERLAY_PREPARED_ADMISSION_MISSING")
    receipt = PreparedFeatureOverlayAdmission.model_validate_json(
        json.dumps(
            store.load_json(
                category="research-feature-preprocessing",
                content_hash=manifest.preparation_admission_hash,
            )
        )
    )
    if (
        receipt.admission_hash != manifest.preparation_admission_hash
        or receipt.source_identity != manifest.source_identity
        or receipt.raw_column_references != manifest.raw_column_references
    ):
        raise ValueError("FEATURE_OVERLAY_PREPARED_ADMISSION_MISMATCH")
    return cast(PreparedFeatureOverlayAdmission, receipt)


def _prepared_raw_columns(
    output_root: Path, manifest: DevelopmentFeatureOverlayManifest
) -> dict[str, npt.NDArray[np.float64]]:
    receipt = read_prepared_feature_overlay_admission(output_root, manifest)
    owner = ResearchFormulaValues(PanelClosureArtifactStore(ArtifactResolver(output_root)))
    sessions = tuple(date.fromisoformat(v) for v in manifest.ordered_session_axis)
    listings = manifest.ordered_listing_axis
    data: dict[str, npt.NDArray[np.float64]] = {}
    specs = {v.factor_id: v for v in receipt.specifications}
    numerical = dict(receipt.numerical_definitions)
    for name, identity in receipt.raw_column_references:
        column, values = owner.open(identity)
        recorded = next(v for v in manifest.columns if v.factor_id == name)
        if (
            column.input_binding_hash != receipt.input_binding_hash
            or column.numerical_spec_hash != numerical[name]
            or column.sessions != sessions
            or column.listing_ids != listings
            or column.implementation_hash != specs[name].implementation_hash
            or recorded.admission_receipt_hash != receipt.admission_hash
            or recorded.formula_specification_hash != specs[name].specification_hash
            or recorded.raw_child_identity != _array_identity(values.reshape(-1))
        ):
            raise ValueError("FEATURE_OVERLAY_PREPARED_RAW_BINDING_MISMATCH")
        data[name] = values.reshape(-1)
    return data


class NoAdmittedCandidatesReceipt(_Contract):
    """Seal the absence of admitted development candidates."""

    kind: Literal["NoAdmittedCandidatesReceipt"] = "NoAdmittedCandidatesReceipt"
    disposition: Literal["NO_ADMITTED_CANDIDATES"] = "NO_ADMITTED_CANDIDATES"
    admission_receipt_hashes: tuple[str, ...]
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Check the empty admission receipt content hash."""
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise ValueError("FEATURE_OVERLAY_EMPTY_RECEIPT_INVALID")
        return self


class DevelopmentMethodologySurfaceManifest(_Contract):
    """One full development axis reprocessed from durable raw formula outputs."""

    kind: Literal["DevelopmentMethodologySurfaceManifest"] = "DevelopmentMethodologySurfaceManifest"
    scope: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    base_panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_panel_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_session_axis: tuple[str, ...] = Field(min_length=1)
    ordered_listing_axis: tuple[str, ...] = Field(min_length=1)
    ordered_factor_axis: tuple[str, ...] = Field(min_length=1)
    raw_formula_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formula_authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_by_factor: tuple[tuple[str, str, str, str], ...] = Field(min_length=1)
    """Factor, method id, recipe hash, and exact implementation binding hash."""
    transformed_parquet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformed_parquet_relative_path: Literal["values.parquet"] = "values.parquet"
    row_count: int = Field(ge=1)
    surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_surface(self) -> Self:
        """Check the methodology axis and surface content hash."""
        if tuple(
            value[0] for value in self.method_by_factor
        ) != self.ordered_factor_axis or self.surface_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"surface_hash"})
        ):
            raise ValueError("FEATURE_METHODOLOGY_SURFACE_IDENTITY_INVALID")
        return self


class DevelopmentRawFormulaSurfaceManifest(_Contract):
    """One reusable raw Formula axis, sealed before any preprocessing runs."""

    kind: Literal["DevelopmentRawFormulaSurfaceManifest"] = "DevelopmentRawFormulaSurfaceManifest"
    scope: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    base_panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_panel_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    formula_authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_session_axis: tuple[str, ...] = Field(min_length=1)
    ordered_listing_axis: tuple[str, ...] = Field(min_length=1)
    ordered_factor_axis: tuple[str, ...] = Field(min_length=1)
    ordered_context_axis: tuple[str, ...] = ()
    parquet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parquet_relative_path: Literal["raw_values.parquet"] = "raw_values.parquet"
    row_count: int = Field(ge=1)
    surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_surface(self) -> Self:
        """Check disjoint factor and context axes and surface identity."""
        if set(self.ordered_factor_axis) & set(self.ordered_context_axis):
            raise ValueError("FEATURE_RAW_FORMULA_CONTEXT_AXIS_OVERLAPS_FACTOR_AXIS")
        if self.surface_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"surface_hash"})
        ):
            raise ValueError("FEATURE_RAW_FORMULA_SURFACE_IDENTITY_INVALID")
        return self


def load_development_feature_overlay_manifest(
    *, output_root: Path, overlay_hash: str
) -> DevelopmentFeatureOverlayManifest:
    """Resolve one published overlay manifest by hash and verify its parquet."""
    return _read_verified_overlay(output_root=output_root, overlay_hash=overlay_hash)[0]


def _read_verified_overlay(
    *, output_root: Path, overlay_hash: str
) -> tuple[DevelopmentFeatureOverlayManifest, dict[str, npt.NDArray[np.float64]] | None]:
    """One read scope; return verified raw arrays only to this call's consumer."""
    manifest_path = (
        output_root / DEVELOPMENT_FEATURE_OVERLAY_CATEGORY / overlay_hash / "manifest.json"
    )
    if not manifest_path.is_file():
        raise ValueError("FEATURE_OVERLAY_MANIFEST_MISSING")
    manifest = DevelopmentFeatureOverlayManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.overlay_hash != overlay_hash:
        raise ValueError("FEATURE_OVERLAY_IDENTITY_INVALID")
    parquet_path = manifest_path.parent / manifest.parquet_relative_path
    if not parquet_path.is_file() or sha256_file(parquet_path) != manifest.parquet_sha256:
        raise ValueError("FEATURE_OVERLAY_PARQUET_IDENTITY_INVALID")
    if manifest.raw_parquet_relative_path is not None:
        raw_path = manifest_path.parent / manifest.raw_parquet_relative_path
        if (
            not raw_path.is_file()
            or manifest.raw_parquet_sha256 is None
            or sha256_file(raw_path) != manifest.raw_parquet_sha256
        ):
            raise ValueError("FEATURE_OVERLAY_RAW_PARQUET_IDENTITY_INVALID")
    raw = (
        _prepared_raw_columns(output_root, manifest)
        if manifest.raw_column_references is not None
        else None
    )
    return cast(DevelopmentFeatureOverlayManifest, manifest), raw


class DevelopmentFeatureOverlayService:
    """One Feature-owned build over all independently admitted candidates."""

    def materialize_prepared(
        self,
        *,
        output_root: Path,
        input_binding_hash: str,
        base_panel_snapshot_hash: str,
        panel_binding: FeaturePanelBinding,
        definitions: tuple[FactorSpec, ...],
        raw_column_references: tuple[tuple[str, str], ...],
        sessions: tuple[date, ...],
        listing_ids: tuple[str, ...],
        sector_by_listing_id: Mapping[str, str],
        members_by_session: Mapping[date, Sequence[str]],
        source_exclusions_by_session: Mapping[date, Sequence[str]],
        capacity: Callable[[int], None],
        cancelled: Callable[[], bool] = lambda: False,
        preprocessing_recipes: Mapping[str, str] | None = None,
    ) -> tuple[DevelopmentFeatureOverlayManifest, int]:
        """Transform already verified local values; preserve base bytes and raw references.

        This entry adds no arithmetic implementation or default admission for an
        unknown formula. The specification owner resolves each local recipe, and
        the same installed preprocessing adapters as ``materialize`` perform it.
        The returned call count belongs to this attempt, not an old publication.
        """
        from alphalattice.foundation.feature_engine.producers.factors.core_bundle import (
            numerical_spec_hash,
        )

        def checkpoint() -> None:
            if cancelled():
                raise ValueError("feature_research.cancelled_at_safe_checkpoint")

        checkpoint()
        recipes = tuple(sorted(definitions, key=lambda v: v.factor_id))
        if (
            not sessions
            or sessions != tuple(sorted(set(sessions)))
            or not listing_ids
            or listing_ids != tuple(sorted(set(listing_ids)))
            or set(members_by_session) != set(sessions)
        ):
            raise ValueError("FEATURE_OVERLAY_PREPARED_SCOPE_INVALID")
        axis = tuple(v.factor_id for v in recipes)
        if (
            not axis
            or len(set(axis)) != len(axis)
            or tuple(n for n, _ in raw_column_references) != axis
        ):
            raise ValueError("FEATURE_OVERLAY_PREPARED_AXIS_INVALID")
        if len(sessions) * len(listing_ids) * len(axis) * 8 > 512 * 1024 * 1024:
            raise ValueError("FEATURE_OVERLAY_PREPARED_BUFFER_BUDGET_EXCEEDED")
        specs = tuple(
            build_research_formula_specification(
                v,
                source_session_count=len(sessions),
                preprocessing_recipe=(preprocessing_recipes or {}).get(v.factor_id),
            )
            for v in recipes
        )
        preprocessing = build_installed_panel_preprocessing_catalog()
        roles: dict[str, list[str]] = {}
        for specification in specs:
            role: str = specification.preprocessing_role
            preprocessing.admit_for_development_overlay(role)
            if role == "STATE_INTERACTION_BLOCK":
                raise ValueError("FEATURE_OVERLAY_PREPARED_CONTEXT_REQUIRED")
            roles.setdefault(role, []).append(specification.factor_id)
        source_identity = prepared_overlay_source_identity(
            input_binding_hash=input_binding_hash,
            base_panel_snapshot_hash=base_panel_snapshot_hash,
            raw_column_references=raw_column_references,
            sessions=sessions,
            sector_by_listing_id=sector_by_listing_id,
            members_by_session=members_by_session,
            source_exclusions_by_session=source_exclusions_by_session,
        )
        admission_values = dict(
            kind="PreparedFeatureOverlayAdmission",
            scope="DEVELOPMENT_ONLY",
            input_binding_hash=input_binding_hash,
            source_identity=source_identity,
            numerical_definitions=tuple((v.factor_id, numerical_spec_hash(v)) for v in recipes),
            specifications=specs,
            raw_column_references=raw_column_references,
        )
        draft = PreparedFeatureOverlayAdmission.model_construct(
            **admission_values, admission_hash="0" * 64
        )
        admission = PreparedFeatureOverlayAdmission(
            **admission_values,
            admission_hash=canonical_hash(
                draft.model_dump(mode="json", exclude={"admission_hash"})
            ),
        )
        catalog_hash = str(
            canonical_hash(
                [(v.factor_id, v.specification_hash, v.implementation_hash) for v in specs]
            )
        )
        root = output_root / DEVELOPMENT_FEATURE_OVERLAY_CATEGORY
        reusable: DevelopmentFeatureOverlayManifest | None = None
        reused_names: dict[str, str] = {}
        best_match = (0, "")
        role_bindings = {
            role: preprocessing.resolve_executable(role)[0].recipe_hash for role in roles
        }
        for path in sorted(root.glob("*/manifest.json")):
            prior = DevelopmentFeatureOverlayManifest.model_validate_json(path.read_bytes())
            if (
                prior.base_panel_snapshot_hash != base_panel_snapshot_hash
                or prior.base_panel_binding_hash != panel_binding.panel_binding_hash
                or prior.ordered_listing_axis != listing_ids
                or prior.ordered_session_axis != tuple(str(d) for d in sessions)
                or prior.raw_column_references is None
                or prior.source_identity
                != prepared_overlay_source_identity(
                    input_binding_hash=input_binding_hash,
                    base_panel_snapshot_hash=base_panel_snapshot_hash,
                    raw_column_references=prior.raw_column_references,
                    sessions=sessions,
                    sector_by_listing_id=sector_by_listing_id,
                    members_by_session=members_by_session,
                    source_exclusions_by_session=source_exclusions_by_session,
                )
            ):
                continue
            if (
                prior.preparation_admission_hash == admission.admission_hash
                and preprocessing.catalog_current(prior.preprocessing_catalog_hash)
            ):
                return load_development_feature_overlay_manifest(
                    output_root=output_root, overlay_hash=prior.overlay_hash
                ), 0
            old_columns = {v.factor_id: v for v in prior.columns}
            old_raw: dict[str, list[str]] = {}
            for name, identity in prior.raw_column_references:
                old_raw.setdefault(identity, []).append(name)
            matched: dict[str, str] = {}
            for specification, (name, identity) in zip(specs, raw_column_references, strict=True):
                role = specification.preprocessing_role
                recipe_hash = role_bindings[role]
                for old_name in old_raw.get(identity, ()):
                    old = old_columns[old_name]
                    if (
                        old.preprocessing_recipe_id == role
                        and old.preprocessing_recipe_hash == recipe_hash
                        and preprocessing.implementation_current(
                            role, old.preprocessing_implementation_hash
                        )
                        and old.implementation_hash == specification.implementation_hash
                    ):
                        matched[name] = old_name
                        break
            rank = (len(matched), prior.overlay_hash)
            if matched and rank > best_match:
                best_match = rank
                reusable = prior
                reused_names = matched
        if reusable is not None:
            reusable = load_development_feature_overlay_manifest(
                output_root=output_root, overlay_hash=reusable.overlay_hash
            )
        owner = ResearchFormulaValues(PanelClosureArtifactStore(ArtifactResolver(output_root)))
        raw = pd.DataFrame(
            {
                "session_date": np.repeat(sessions, len(listing_ids)),
                "listing_id": np.tile(listing_ids, len(sessions)),
            }
        )
        for definition, specification, (_, identity) in zip(
            recipes, specs, raw_column_references, strict=True
        ):
            checkpoint()
            column, array = owner.open(identity)
            if (
                column.input_binding_hash != input_binding_hash
                or column.sessions != sessions
                or column.listing_ids != listing_ids
                or column.numerical_spec_hash != numerical_spec_hash(definition)
                or column.implementation_hash != specification.implementation_hash
            ):
                raise ValueError("FEATURE_OVERLAY_PREPARED_RAW_BINDING_MISMATCH")
            raw[definition.factor_id] = array.reshape(-1)
        join = ["session_date", "listing_id"]
        transformed = raw.loc[:, join].copy()
        columns = []
        if reusable is not None:
            retained = pd.read_parquet(
                root / reusable.overlay_hash / reusable.parquet_relative_path
            )
            retained["session_date"] = pd.to_datetime(retained["session_date"]).dt.date
            old_columns = {v.factor_id: v for v in reusable.columns}
            if not retained[join].equals(transformed[join]):
                raise ValueError("FEATURE_OVERLAY_PREPARED_REUSE_AXIS_MISMATCH")
            for specification in specs:
                name = specification.factor_id
                if name not in reused_names:
                    continue
                old_name = reused_names[name]
                transformed[name] = retained[old_name].to_numpy()
                columns.append(
                    old_columns[old_name].model_copy(
                        update={
                            "factor_id": name,
                            "formula_specification_hash": specification.specification_hash,
                            "admission_receipt_hash": admission.admission_hash,
                        }
                    )
                )
        pending_roles = {
            role: [name for name in factor_ids if name not in reused_names]
            for role, factor_ids in roles.items()
        }
        for role, factor_ids in sorted(pending_roles.items()):
            if not factor_ids:
                continue
            checkpoint()
            recipe, adapter = preprocessing.resolve_executable(role)
            rows, child_identity = _materialize_by_run(
                adapter,
                feature_rows=raw,
                sessions=sessions,
                active_listing_ids=listing_ids,
                sector_by_listing_id=sector_by_listing_id,
                factor_ids=tuple(factor_ids),
                binding=panel_binding,
                members_by_session=members_by_session,
                source_exclusions_by_session=source_exclusions_by_session,
            )
            transformed = _merge_preprocessing(transformed, rows, factor_ids)
            for name in factor_ids:
                specification = next(v for v in specs if v.factor_id == name)
                columns.append(
                    _overlay_column(
                        factor_id=name,
                        formula_specification_hash=specification.specification_hash,
                        implementation_hash=specification.implementation_hash,
                        admission_receipt_hash=admission.admission_hash,
                        preprocessing_recipe_id=role,
                        preprocessing_recipe_hash=recipe.recipe_hash,
                        preprocessing_implementation_hash=preprocessing.implementation_binding(
                            role
                        ).implementation_binding_hash,
                        raw_values=raw[name].to_numpy(float),
                        preprocessing_child_identity=child_identity,
                    )
                )
        sink = pa.BufferOutputStream()
        pq.write_table(
            pa.Table.from_pandas(transformed.loc[:, [*join, *axis]], preserve_index=False),
            sink,
            compression="zstd",
        )
        content = sink.getvalue().to_pybytes()
        manifest = _overlay_manifest(
            base_panel_snapshot_hash=base_panel_snapshot_hash,
            panel_binding=panel_binding,
            source_identity=source_identity,
            installed_kernel_capability_hash=default_extension_kernel_registry().installed_capability_hash,
            factor_catalog_revision_hash=catalog_hash,
            preprocessing_catalog_hash=preprocessing.catalog_hash,
            sessions=tuple(str(d) for d in sessions),
            listing_ids=listing_ids,
            candidate_ids=axis,
            columns=columns,
            parquet_sha256=sha256(content).hexdigest(),
            raw_column_references=raw_column_references,
            preparation_admission_hash=admission.admission_hash,
            row_count=len(transformed),
        )
        encoded = json.dumps(
            manifest.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        target = root / manifest.overlay_hash
        closure = PanelClosureArtifactStore(ArtifactResolver(output_root), capacity=capacity)
        admission_bytes = json.dumps(
            admission.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        additions = [
            (target / "values.parquet", content),
            (target / "manifest.json", encoded),
            (
                closure.root
                / "research-feature-preprocessing"
                / f"{admission.admission_hash}.json",
                admission_bytes,
            ),
        ]
        for destination, data in additions:
            if destination.exists() and destination.read_bytes() != data:
                raise ValueError("FEATURE_OVERLAY_PREPARED_PUBLICATION_CONFLICT")
        capacity(sum(len(data) for destination, data in additions if not destination.exists()))
        checkpoint()
        target.mkdir(parents=True, exist_ok=True)
        closure.publish_json(
            category="research-feature-preprocessing",
            content_hash=admission.admission_hash,
            payload=admission.model_dump(mode="json"),
        )
        with TemporaryDirectory(dir=target, prefix=".publish-") as temporary:
            for name, data in (("values.parquet", content), ("manifest.json", encoded)):
                destination = target / name
                if destination.exists():
                    if destination.read_bytes() != data:
                        raise ValueError("FEATURE_OVERLAY_PREPARED_PUBLICATION_CONFLICT")
                    continue
                staged = Path(temporary) / name
                staged.write_bytes(data)
                replace_with_retry(staged, destination, delays=DURABLE_REPLACE_DELAYS)
        return load_development_feature_overlay_manifest(
            output_root=output_root, overlay_hash=manifest.overlay_hash
        ), sum(bool(factor_ids) for factor_ids in pending_roles.values())

    def find_exact(
        self,
        *,
        output_root: Path,
        base_panel_snapshot_hash: str,
        source_identity: str,
        panel_binding: FeaturePanelBinding,
        admission_receipts: tuple[FactorDevelopmentAdmissionReceipt, ...],
    ) -> DevelopmentFeatureOverlayManifest | None:
        """Resolve only a fully published overlay bound to the current authorities."""
        admitted = tuple(
            sorted(item.factor_id for item in admission_receipts if item.disposition == "ADMITTED")
        )
        capabilities = {
            item.factor_id: item for item in build_installed_factor_development_capabilities()
        }
        specifications = build_installed_factor_formula_specifications()
        factor_catalog_revision_hash = str(
            canonical_hash(
                tuple(
                    (
                        factor_id,
                        specifications.resolve(factor_id).specification_hash,
                        capabilities[factor_id].implementation_hash,
                    )
                    for factor_id in admitted
                )
            )
        )
        kernels = default_extension_kernel_registry()
        preprocessing = build_installed_panel_preprocessing_catalog()
        root = output_root / DEVELOPMENT_FEATURE_OVERLAY_CATEGORY
        for path in sorted(root.glob("*/manifest.json")) if root.is_dir() else ():
            manifest = cast(
                DevelopmentFeatureOverlayManifest,
                DevelopmentFeatureOverlayManifest.model_validate_json(
                    path.read_text(encoding="utf-8")
                ),
            )
            parquet = path.parent / manifest.parquet_relative_path
            if (
                manifest.base_panel_snapshot_hash == base_panel_snapshot_hash
                and manifest.base_panel_binding_hash == panel_binding.panel_binding_hash
                and manifest.source_identity == source_identity
                and manifest.ordered_candidate_axis == admitted
                and manifest.installed_kernel_capability_hash == kernels.installed_capability_hash
                and manifest.factor_catalog_revision_hash == factor_catalog_revision_hash
                and preprocessing.catalog_current(manifest.preprocessing_catalog_hash)
                and parquet.is_file()
                and sha256_file(parquet) == manifest.parquet_sha256
            ):
                return manifest
        return None

    def find_reusable(
        self,
        *,
        output_root: Path,
        base_panel_snapshot_hash: str,
        source_identity: str,
        panel_binding: FeaturePanelBinding,
    ) -> DevelopmentFeatureOverlayManifest | None:
        """Find a published overlay whose unrotated columns can seed a rebuild.

        Unlike :meth:`find_exact` this does not require the whole installed
        catalog to match -- only the same immutable base Panel, binding and
        source identity, so column values whose formula and implementation
        identities still hold can be copied instead of recomputed. Among
        eligible manifests the one sharing the most currently-installed column
        identities wins; ties resolve to the greatest overlay hash.
        """
        capabilities = {
            item.factor_id: item for item in build_installed_factor_development_capabilities()
        }
        best: tuple[int, str] | None = None
        chosen: DevelopmentFeatureOverlayManifest | None = None
        root = output_root / DEVELOPMENT_FEATURE_OVERLAY_CATEGORY
        for path in sorted(root.glob("*/manifest.json")) if root.is_dir() else ():
            manifest = cast(
                DevelopmentFeatureOverlayManifest,
                DevelopmentFeatureOverlayManifest.model_validate_json(
                    path.read_text(encoding="utf-8")
                ),
            )
            parquet = path.parent / manifest.parquet_relative_path
            if (
                manifest.base_panel_snapshot_hash != base_panel_snapshot_hash
                or manifest.base_panel_binding_hash != panel_binding.panel_binding_hash
                or manifest.source_identity != source_identity
                or not parquet.is_file()
                or sha256_file(parquet) != manifest.parquet_sha256
            ):
                continue
            matches = sum(
                1
                for column in manifest.columns
                if column.factor_id in capabilities
                and column.formula_specification_hash
                == capabilities[column.factor_id].formula_specification_hash
                and column.implementation_hash == capabilities[column.factor_id].implementation_hash
            )
            key = (matches, manifest.overlay_hash)
            if matches and (best is None or key > best):
                best = key
                chosen = manifest
        return chosen

    def materialize(
        self,
        *,
        source_rows: pd.DataFrame,
        base_panel_rows: pd.DataFrame,
        base_panel_snapshot_hash: str,
        source_identity: str,
        panel_binding: FeaturePanelBinding,
        active_listing_ids: tuple[str, ...],
        sector_by_listing_id: Mapping[str, str],
        output_root: Path,
        admission_receipts: tuple[FactorDevelopmentAdmissionReceipt, ...] | None = None,
        previous_overlay: DevelopmentFeatureOverlayManifest | None = None,
    ) -> DevelopmentFeatureOverlayManifest | NoAdmittedCandidatesReceipt:
        """Materialize an admitted development overlay on the base Panel."""
        receipts = admission_receipts or admit_factor_development_capabilities()
        capabilities = {
            item.factor_id: item for item in build_installed_factor_development_capabilities()
        }
        if len(receipts) != len({item.factor_id for item in receipts}):
            raise ValueError("FEATURE_OVERLAY_ADMISSION_RECEIPT_DUPLICATED")
        for receipt in receipts:
            capability = capabilities.get(receipt.factor_id)
            if capability is None or receipt.capability_hash != capability.capability_hash:
                raise ValueError("FEATURE_OVERLAY_ADMISSION_AUTHORITY_MISMATCH")
        admitted = {item.factor_id: item for item in receipts if item.disposition == "ADMITTED"}
        if not admitted:
            payload = {
                "kind": "NoAdmittedCandidatesReceipt",
                "disposition": "NO_ADMITTED_CANDIDATES",
                "admission_receipt_hashes": [item.receipt_hash for item in receipts],
            }
            return NoAdmittedCandidatesReceipt(
                admission_receipt_hashes=tuple(item.receipt_hash for item in receipts),
                receipt_hash=str(canonical_hash(payload)),
            )
        recipes = tuple(item for item in extension_factor_specs() if item.factor_id in admitted)
        specifications = build_installed_factor_formula_specifications()
        kernels = default_extension_kernel_registry()
        preprocessing = build_installed_panel_preprocessing_catalog()
        factor_catalog_revision_hash = str(
            canonical_hash(
                tuple(
                    (
                        item.factor_id,
                        specifications.resolve(item.factor_id).specification_hash,
                        capabilities[item.factor_id].implementation_hash,
                    )
                    for item in recipes
                )
            )
        )
        reused_columns: dict[str, DevelopmentFeatureOverlayColumn] = {}
        reused_values: pd.DataFrame | None = None
        reused_raw_values: pd.DataFrame | None = None
        if previous_overlay is not None:
            previous_parquet = (
                output_root
                / DEVELOPMENT_FEATURE_OVERLAY_CATEGORY
                / previous_overlay.overlay_hash
                / previous_overlay.parquet_relative_path
            )
            if (
                previous_overlay.base_panel_snapshot_hash != base_panel_snapshot_hash
                or previous_overlay.base_panel_binding_hash != panel_binding.panel_binding_hash
                or previous_overlay.source_identity != source_identity
                or previous_overlay.ordered_listing_axis != active_listing_ids
                or not previous_parquet.is_file()
                or sha256_file(previous_parquet) != previous_overlay.parquet_sha256
            ):
                raise ValueError("FEATURE_OVERLAY_PREVIOUS_AUTHORITY_MISMATCH")
            for column in previous_overlay.columns:
                if column.factor_id not in admitted:
                    continue
                current = specifications.resolve(column.factor_id)
                role: str = current.preprocessing_role
                preprocessing.admit_for_development_overlay(role)
                current_recipe, _adapter = preprocessing.resolve_executable(role)
                if (
                    column.formula_specification_hash == current.specification_hash
                    and column.implementation_hash
                    == capabilities[column.factor_id].implementation_hash
                    and column.preprocessing_recipe_id == role
                    and column.preprocessing_recipe_hash == current_recipe.recipe_hash
                    and preprocessing.implementation_current(
                        role, column.preprocessing_implementation_hash
                    )
                ):
                    # Values are copied verbatim below; the admission receipt is
                    # this run's, because admission is a fresh Host decision.
                    reused_columns[column.factor_id] = column.model_copy(
                        update={"admission_receipt_hash": admitted[column.factor_id].receipt_hash}
                    )
            if reused_columns:
                reused_values = pd.read_parquet(previous_parquet).loc[
                    :, ["session_date", "listing_id", *sorted(reused_columns)]
                ]
                if previous_overlay.raw_parquet_relative_path is not None:
                    previous_raw = (
                        previous_parquet.parent / previous_overlay.raw_parquet_relative_path
                    )
                    if (
                        previous_overlay.raw_parquet_sha256 is None
                        or not previous_raw.is_file()
                        or sha256_file(previous_raw) != previous_overlay.raw_parquet_sha256
                    ):
                        raise ValueError("FEATURE_OVERLAY_PREVIOUS_RAW_AUTHORITY_MISMATCH")
                    reused_raw_values = pd.read_parquet(previous_raw).loc[
                        :, ["session_date", "listing_id", *sorted(reused_columns)]
                    ]
                else:
                    # Old transformed-only overlays remain readable, but they cannot seed
                    # a new methodology whose whole purpose is to reprocess raw formula output.
                    reused_columns.clear()
                    reused_values = None
        compute_recipes = tuple(item for item in recipes if item.factor_id not in reused_columns)
        source, raw = _raw_formula_rows(
            source_rows,
            base_panel_rows,
            compute_recipes,
            kernels.compute,
            missing_code="FEATURE_OVERLAY_SOURCE_COLUMNS_MISSING",
        )
        join_columns = ["session_date", "listing_id"]
        interaction_ids = tuple(
            item.factor_id
            for item in compute_recipes
            if specifications.resolve(item.factor_id).preprocessing_role
            == "STATE_INTERACTION_BLOCK"
        )
        if interaction_ids:
            state_children = interaction_market_state_children(source)
            for factor_id in interaction_ids:
                raw[f"{factor_id}__state"] = state_children[f"{factor_id}__state"].to_numpy(float)
        role_groups: dict[str, list[str]] = {}
        for recipe in compute_recipes:
            role_groups.setdefault(
                specifications.resolve(recipe.factor_id).preprocessing_role, []
            ).append(recipe.factor_id)
        transformed = raw.loc[:, join_columns].copy()
        columns: list[DevelopmentFeatureOverlayColumn] = []
        for role in sorted(role_groups):
            factor_ids = tuple(sorted(role_groups[role]))
            preprocessing.admit_for_development_overlay(role)
            recipe, adapter = preprocessing.resolve_executable(role)
            rows, child_identity = _materialize_by_run(
                adapter,
                feature_rows=raw,
                sessions=tuple(sorted(set(pd.to_datetime(raw["session_date"]).dt.date))),
                active_listing_ids=active_listing_ids,
                sector_by_listing_id=sector_by_listing_id,
                factor_ids=factor_ids,
                binding=panel_binding,
            )
            transformed = _merge_preprocessing(
                transformed, rows, factor_ids, normalize_base_dates=True
            )
            implementation = preprocessing.implementation_binding(role)
            for factor_id in factor_ids:
                columns.append(
                    _overlay_column(
                        factor_id=factor_id,
                        formula_specification_hash=specifications.resolve(
                            factor_id
                        ).specification_hash,
                        implementation_hash=capabilities[factor_id].implementation_hash,
                        admission_receipt_hash=admitted[factor_id].receipt_hash,
                        preprocessing_recipe_id=role,
                        preprocessing_recipe_hash=recipe.recipe_hash,
                        preprocessing_implementation_hash=(
                            implementation.implementation_binding_hash
                        ),
                        raw_values=raw[factor_id].to_numpy(float),
                        preprocessing_child_identity=child_identity,
                        market_state_values=(
                            raw[f"{factor_id}__state"].to_numpy(float)
                            if role == "STATE_INTERACTION_BLOCK"
                            else None
                        ),
                    )
                )
        if reused_values is not None:
            reused_values = reused_values.copy()
            reused_values["session_date"] = pd.to_datetime(reused_values["session_date"]).dt.date
            transformed["session_date"] = pd.to_datetime(transformed["session_date"]).dt.date
            transformed = transformed.merge(
                reused_values, on=join_columns, how="left", validate="one_to_one"
            )
            columns.extend(reused_columns.values())
        if reused_raw_values is not None:
            reused_raw_values = reused_raw_values.copy()
            reused_raw_values["session_date"] = pd.to_datetime(
                reused_raw_values["session_date"]
            ).dt.date
            raw["session_date"] = pd.to_datetime(raw["session_date"]).dt.date
            raw = raw.merge(reused_raw_values, on=join_columns, how="left", validate="one_to_one")
        ordered_candidates = tuple(sorted(admitted))
        transformed = transformed.loc[:, [*join_columns, *ordered_candidates]].sort_values(
            join_columns, kind="mergesort"
        )
        sessions = tuple(
            item.isoformat()
            for item in sorted(pd.to_datetime(transformed["session_date"]).dt.date.unique())
        )
        root = output_root / DEVELOPMENT_FEATURE_OVERLAY_CATEGORY
        root.mkdir(parents=True, exist_ok=True)
        provisional_hash = str(
            canonical_hash(
                {
                    "base_panel_snapshot_hash": base_panel_snapshot_hash,
                    "candidates": list(ordered_candidates),
                    "sessions": list(sessions),
                    "listings": list(active_listing_ids),
                }
            )
        )
        directory = root / provisional_hash
        directory.mkdir(parents=True, exist_ok=True)
        parquet_path = directory / "values.parquet"
        parquet_hash = _write_parquet(parquet_path, transformed)
        raw_output = raw.loc[:, [*join_columns, *ordered_candidates]].sort_values(
            join_columns, kind="mergesort"
        )
        raw_parquet_path = directory / "raw_values.parquet"
        raw_parquet_hash = _write_parquet(raw_parquet_path, raw_output)
        manifest = _overlay_manifest(
            base_panel_snapshot_hash=base_panel_snapshot_hash,
            panel_binding=panel_binding,
            source_identity=source_identity,
            installed_kernel_capability_hash=kernels.installed_capability_hash,
            factor_catalog_revision_hash=factor_catalog_revision_hash,
            preprocessing_catalog_hash=preprocessing.catalog_hash,
            sessions=sessions,
            listing_ids=active_listing_ids,
            candidate_ids=ordered_candidates,
            columns=columns,
            parquet_sha256=parquet_hash,
            raw_parquet_sha256=raw_parquet_hash,
            row_count=len(transformed),
        )
        final_directory = root / manifest.overlay_hash
        if final_directory != directory:
            if final_directory.exists():
                raise ValueError("FEATURE_OVERLAY_ALREADY_EXISTS")
            os.replace(directory, final_directory)
            directory = final_directory
        _write_manifest(directory / "manifest.json", manifest)
        return manifest


def read_raw_development_feature_overlay(
    *, output_root: Path, overlay_hash: str
) -> tuple[DevelopmentFeatureOverlayManifest, pd.DataFrame]:
    """Read the successor raw formula surface; transformed-only legacy fails closed."""
    manifest, raw = _read_verified_overlay(output_root=output_root, overlay_hash=overlay_hash)
    if raw is not None:
        return manifest, pd.DataFrame(
            {
                "session_date": np.repeat(
                    tuple(date.fromisoformat(v) for v in manifest.ordered_session_axis),
                    len(manifest.ordered_listing_axis),
                ),
                "listing_id": np.tile(
                    manifest.ordered_listing_axis, len(manifest.ordered_session_axis)
                ),
                **raw,
            }
        )
    if manifest.raw_parquet_relative_path is None or manifest.raw_parquet_sha256 is None:
        raise ValueError("FEATURE_OVERLAY_RAW_VALUES_UNAVAILABLE")
    path = (
        output_root
        / DEVELOPMENT_FEATURE_OVERLAY_CATEGORY
        / manifest.overlay_hash
        / manifest.raw_parquet_relative_path
    )
    if sha256_file(path) != manifest.raw_parquet_sha256:
        raise ValueError("FEATURE_OVERLAY_RAW_PARQUET_IDENTITY_INVALID")
    return manifest, pd.read_parquet(path)


def load_development_raw_formula_surface(
    *, output_root: Path, surface_hash: str
) -> tuple[DevelopmentRawFormulaSurfaceManifest, pd.DataFrame]:
    """Load a raw formula surface after verifying its manifest and bytes."""
    directory = output_root / DEVELOPMENT_RAW_FORMULA_SURFACE_CATEGORY / surface_hash
    marker = directory / "manifest.json"
    if not marker.is_file():
        raise ValueError("FEATURE_RAW_FORMULA_SURFACE_MISSING")
    manifest = DevelopmentRawFormulaSurfaceManifest.model_validate_json(
        marker.read_text(encoding="utf-8")
    )
    path = directory / manifest.parquet_relative_path
    if (
        manifest.surface_hash != surface_hash
        or not path.is_file()
        or sha256_file(path) != manifest.parquet_sha256
    ):
        raise ValueError("FEATURE_RAW_FORMULA_SURFACE_TAMPERED")
    return cast(DevelopmentRawFormulaSurfaceManifest, manifest), pd.read_parquet(path)


def load_development_methodology_surface_manifest(
    *, output_root: Path, surface_hash: str
) -> DevelopmentMethodologySurfaceManifest:
    """Load and verify a development methodology surface manifest."""
    directory = output_root / DEVELOPMENT_METHODOLOGY_SURFACE_CATEGORY / surface_hash
    path = directory / "manifest.json"
    if not path.is_file():
        raise ValueError("FEATURE_METHODOLOGY_SURFACE_MISSING")
    manifest = DevelopmentMethodologySurfaceManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if (
        manifest.surface_hash != surface_hash
        or sha256_file(directory / manifest.transformed_parquet_relative_path)
        != manifest.transformed_parquet_sha256
    ):
        raise ValueError("FEATURE_METHODOLOGY_SURFACE_TAMPERED")
    raw_surface, _rows = load_development_raw_formula_surface(
        output_root=output_root, surface_hash=manifest.raw_formula_surface_hash
    )
    if raw_surface.formula_authority_hash != manifest.formula_authority_hash:
        raise ValueError("FEATURE_METHODOLOGY_RAW_FORMULA_AUTHORITY_MISMATCH")
    return cast(DevelopmentMethodologySurfaceManifest, manifest)


__all__ = [
    "DEVELOPMENT_RAW_FORMULA_SURFACE_CATEGORY",
    "DevelopmentFeatureOverlayService",
    "DevelopmentMethodologySurfaceManifest",
    "DevelopmentRawFormulaSurfaceManifest",
    "NoAdmittedCandidatesReceipt",
    "load_development_feature_overlay_manifest",
    "load_development_methodology_surface_manifest",
    "load_development_raw_formula_surface",
    "read_prepared_feature_overlay_admission",
    "read_raw_development_feature_overlay",
]
