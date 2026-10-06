"""Host resolution of the sources a historical h1 Dynamic Panel run read.

The catalog, its compile and fold materialization and the publication retired with
V19's retained retirement rationale."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Self, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.factor_research.experiments.campaign import (
    CuratedFactorCheckpoint,
    FactorCampaignArtifactReader,
)
from alphalattice.foundation.feature_engine.panels.development_overlay import (
    DEVELOPMENT_METHODOLOGY_SURFACE_CATEGORY,
    load_development_methodology_surface_manifest,
    load_development_raw_formula_surface,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.catalog import (
    JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z,
)
from alphalattice.investment.alpha_research.inputs.dynamic_panel import (
    DynamicPanelSourceArrays,
)
from alphalattice.investment.alpha_research.inputs.workspace import (
    load_sector_history,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .development_artifacts import AlphaDevelopmentArtifactStore

_HASH = r"^[0-9a-f]{64}$"


class DynamicPanelDevelopmentError(ValueError):
    """Stable refusal while resolving immutable Dynamic Panel authorities."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class DynamicPanelDevelopmentRequest(_Contract):
    """Handles only; roots belong to the Host service, never to the Program."""

    panel_snapshot_hash: str = Field(pattern=_HASH)
    methodology_surface_hash: str = Field(pattern=_HASH)
    curated_factor_checkpoint_hash: str = Field(pattern=_HASH)
    total_return_target_evidence_hash: str = Field(pattern=_HASH)


class DynamicPanelSourceResolution(_Contract):
    kind: str = "DynamicPanelSourceResolution"
    panel_snapshot_hash: str = Field(pattern=_HASH)
    panel_binding_hash: str = Field(pattern=_HASH)
    sector_revision: str = Field(pattern=_HASH)
    methodology_surface_hash: str = Field(pattern=_HASH)
    raw_formula_surface_hash: str = Field(pattern=_HASH)
    curated_factor_checkpoint_hash: str = Field(pattern=_HASH)
    total_return_target_evidence_hash: str = Field(pattern=_HASH)
    total_return_target_binding_hash: str = Field(pattern=_HASH)
    outcome_method_binding_hash: str = Field(pattern=_HASH)
    ordered_session_axis_hash: str = Field(pattern=_HASH)
    ordered_listing_axis_hash: str = Field(pattern=_HASH)
    ordered_factor_axis_hash: str = Field(pattern=_HASH)
    ordered_relative_factor_axis_hash: str = Field(pattern=_HASH)
    methodology_surface_action: str
    raw_formula_surface_action: str
    target_surface_action: str
    network_call_count: int = Field(ge=0)
    holdout_read_count: int = Field(ge=0)
    pointer_mutation_count: int = Field(ge=0)
    source_write_count: int = Field(ge=0)
    resolution_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, resolution_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"resolution_hash"})
        return cls(**values, resolution_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if any(
            value != 0
            for value in (
                self.network_call_count,
                self.holdout_read_count,
                self.pointer_mutation_count,
                self.source_write_count,
            )
        ) or self.resolution_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"resolution_hash"})
        ):
            raise DynamicPanelDevelopmentError(
                "alpha_research.dynamic_panel_source_resolution_invalid"
            )
        return self


class DynamicPanelAlphaDevelopmentService:
    """Resolve the immutable development children a historical Dynamic Panel run read."""

    def __init__(
        self,
        *,
        panel_artifact_root: Path,
        feature_artifact_root: Path,
        factor_artifact_reader: FactorCampaignArtifactReader,
        target_artifact_root: Path,
    ) -> None:
        self._panel_artifact_root = panel_artifact_root.resolve()
        self._feature_artifact_root = feature_artifact_root.resolve()
        self._factor_artifact_reader = factor_artifact_reader
        self._target_artifact_root = target_artifact_root.resolve()

    def _curated_checkpoint(self, content_hash: str) -> CuratedFactorCheckpoint:
        payload = self._factor_artifact_reader.load_pipeline_contract(
            artifact_kind="curated-checkpoint",
            uri=self._factor_artifact_reader.uri("campaign/curated-checkpoints", content_hash),
        )
        return cast(CuratedFactorCheckpoint, CuratedFactorCheckpoint.model_validate(payload))

    @staticmethod
    def _ordered_frame(frame: pd.DataFrame, *, session_column: str) -> pd.DataFrame:
        result = frame.copy()
        result[session_column] = pd.to_datetime(result[session_column]).dt.date
        result["listing_id"] = result["listing_id"].astype(str)
        return cast(
            pd.DataFrame,
            result.sort_values([session_column, "listing_id"], kind="mergesort").reset_index(
                drop=True
            ),
        )

    def resolve_source(
        self, request: DynamicPanelDevelopmentRequest
    ) -> tuple[DynamicPanelSourceArrays, DynamicPanelSourceResolution, datetime]:
        """Load only immutable development children and reconcile all row axes."""

        checkpoint = self._curated_checkpoint(request.curated_factor_checkpoint_hash)
        methodology = load_development_methodology_surface_manifest(
            output_root=self._feature_artifact_root,
            surface_hash=request.methodology_surface_hash,
        )
        if (
            checkpoint.feature_source_hash != methodology.surface_hash
            or not set(checkpoint.ordered_factor_ids).issubset(methodology.ordered_factor_axis)
            or methodology.base_panel_snapshot_hash != request.panel_snapshot_hash
        ):
            raise DynamicPanelDevelopmentError(
                "alpha_research.dynamic_panel_factor_authority_mismatch"
            )
        raw_manifest, raw_rows = load_development_raw_formula_surface(
            output_root=self._feature_artifact_root,
            surface_hash=methodology.raw_formula_surface_hash,
        )
        if (
            raw_manifest.ordered_factor_axis != methodology.ordered_factor_axis
            or raw_manifest.ordered_session_axis != methodology.ordered_session_axis
            or raw_manifest.ordered_listing_axis != methodology.ordered_listing_axis
        ):
            raise DynamicPanelDevelopmentError(
                "alpha_research.dynamic_panel_raw_formula_authority_mismatch"
            )

        target_store = AlphaDevelopmentArtifactStore(self._target_artifact_root)
        evidence, targets, _dispersion, _bounds = target_store.load_total_return_target_surface(
            request.total_return_target_evidence_hash
        )
        binding = target_store.load_total_return_target_binding(evidence.recipe_binding_hash)
        if evidence.ordered_listing_ids != tuple(raw_manifest.ordered_listing_axis):
            raise DynamicPanelDevelopmentError(
                "alpha_research.dynamic_panel_target_listing_axis_mismatch"
            )

        resolver = ArtifactResolver(self._panel_artifact_root)
        panel_ref = resolver.feature_panel_manifest_uri(request.panel_snapshot_hash)
        panel = resolver.load_feature_panel_manifest(panel_ref)
        lineage = panel["safe_summary"]["lineage"]
        sector_revision = str(lineage["sector_revision"])
        sector_map = load_sector_history(resolver=resolver, sector_revision=sector_revision)
        if set(evidence.ordered_listing_ids) - set(sector_map):
            raise DynamicPanelDevelopmentError(
                "alpha_research.dynamic_panel_sector_authority_incomplete"
            )
        sector_map = {value: sector_map[value] for value in evidence.ordered_listing_ids}

        target_frame = self._ordered_frame(targets.to_pandas(), session_column="formation_session")
        sessions = evidence.formation_sessions
        session_set = set(sessions)
        factor_ids = checkpoint.ordered_factor_ids
        method_by_factor = {
            factor_id: method_id
            for factor_id, method_id, _recipe_hash, _binding_hash in (methodology.method_by_factor)
        }
        if set(factor_ids) - set(method_by_factor):
            raise DynamicPanelDevelopmentError(
                "alpha_research.dynamic_panel_factor_role_authority_incomplete"
            )
        relative_factor_ids = tuple(
            factor_id
            for factor_id in factor_ids
            if method_by_factor[factor_id] == JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z
        )
        if not relative_factor_ids:
            raise DynamicPanelDevelopmentError(
                "alpha_research.dynamic_panel_relative_axis_unavailable"
            )
        methodology_path = (
            self._feature_artifact_root
            / DEVELOPMENT_METHODOLOGY_SURFACE_CATEGORY
            / methodology.surface_hash
            / methodology.transformed_parquet_relative_path
        )
        relative_frame = self._ordered_frame(
            pq.read_table(
                methodology_path,
                columns=[
                    "session_date",
                    "listing_id",
                    *relative_factor_ids,
                ],
            ).to_pandas(),
            session_column="session_date",
        )
        relative_frame = relative_frame.loc[
            relative_frame["session_date"].isin(session_set)
        ].reset_index(drop=True)
        raw_frame = self._ordered_frame(raw_rows, session_column="session_date")
        raw_frame = raw_frame.loc[
            raw_frame["session_date"].isin(session_set),
            [
                "session_date",
                "listing_id",
                *factor_ids,
                *raw_manifest.ordered_context_axis,
            ],
        ].reset_index(drop=True)
        target_keys = tuple(
            zip(
                target_frame["formation_session"],
                target_frame["listing_id"],
                strict=True,
            )
        )
        relative_keys = tuple(
            zip(
                relative_frame["session_date"],
                relative_frame["listing_id"],
                strict=True,
            )
        )
        raw_keys = tuple(zip(raw_frame["session_date"], raw_frame["listing_id"], strict=True))
        if target_keys != relative_keys or target_keys != raw_keys:
            raise DynamicPanelDevelopmentError("alpha_research.dynamic_panel_row_axis_mismatch")
        context_axis = (
            "market_drawdown_x_momentum__state",
            "market_vol_ratio_x_reversal__state",
        )
        if not set(context_axis).issubset(raw_manifest.ordered_context_axis):
            raise DynamicPanelDevelopmentError(
                "alpha_research.dynamic_panel_absolute_state_unavailable"
            )
        session_holding = target_frame.groupby("formation_session", sort=True)[
            "holding_end_session"
        ]
        if bool((session_holding.nunique() != 1).any()):
            raise DynamicPanelDevelopmentError(
                "alpha_research.dynamic_panel_holding_end_axis_ambiguous"
            )
        holding_end_sessions = tuple(
            pd.to_datetime(value).date() for value in session_holding.first().tolist()
        )
        shape = (len(sessions), len(evidence.ordered_listing_ids))
        source = DynamicPanelSourceArrays(
            formation_sessions=sessions,
            holding_end_sessions=holding_end_sessions,
            ordered_listing_ids=evidence.ordered_listing_ids,
            ordered_factor_ids=factor_ids,
            ordered_relative_factor_ids=relative_factor_ids,
            sector_by_listing_id=sector_map,
            relative_factor_values=np.ascontiguousarray(
                relative_frame.loc[:, list(relative_factor_ids)]
                .to_numpy(dtype=np.float64)
                .reshape(*shape, len(relative_factor_ids))
            ),
            raw_factor_values=np.ascontiguousarray(
                raw_frame.loc[:, list(factor_ids)]
                .to_numpy(dtype=np.float64)
                .reshape(*shape, len(factor_ids))
            ),
            target_z_values=np.ascontiguousarray(
                target_frame["fit_target"].to_numpy(dtype=np.float64).reshape(shape)
            ),
            raw_log_returns=np.ascontiguousarray(
                target_frame["raw_log_execution_return"].to_numpy(dtype=np.float64).reshape(shape)
            ),
            simple_economic_returns=np.ascontiguousarray(
                target_frame["simple_economic_return"].to_numpy(dtype=np.float64).reshape(shape)
            ),
            raw_absolute_state_values=np.ascontiguousarray(
                raw_frame.loc[:, list(context_axis)]
                .to_numpy(dtype=np.float64)
                .reshape(*shape, len(context_axis))
            ),
            relative_surface_hash=methodology.surface_hash,
            raw_formula_surface_hash=raw_manifest.surface_hash,
            total_return_target_evidence_hash=evidence.evidence_hash,
            outcome_method_binding_hash=binding.outcome_method_binding_hash,
        )
        resolution = DynamicPanelSourceResolution.create(
            panel_snapshot_hash=request.panel_snapshot_hash,
            panel_binding_hash=str(lineage["panel_binding_hash"]),
            sector_revision=sector_revision,
            methodology_surface_hash=methodology.surface_hash,
            raw_formula_surface_hash=raw_manifest.surface_hash,
            curated_factor_checkpoint_hash=checkpoint.checkpoint_hash,
            total_return_target_evidence_hash=evidence.evidence_hash,
            total_return_target_binding_hash=binding.binding_hash,
            outcome_method_binding_hash=binding.outcome_method_binding_hash,
            ordered_session_axis_hash=str(
                canonical_hash([value.isoformat() for value in sessions])
            ),
            ordered_listing_axis_hash=str(canonical_hash(list(evidence.ordered_listing_ids))),
            ordered_factor_axis_hash=str(canonical_hash(list(factor_ids))),
            ordered_relative_factor_axis_hash=str(canonical_hash(list(relative_factor_ids))),
            methodology_surface_action="REUSED_EXACT_CONTENT_ADDRESSED_CHILD",
            raw_formula_surface_action="REUSED_EXACT_CONTENT_ADDRESSED_CHILD",
            target_surface_action="REUSED_EXACT_CONTENT_ADDRESSED_CHILD",
            network_call_count=0,
            holdout_read_count=0,
            pointer_mutation_count=0,
            source_write_count=0,
        )
        raw_frozen = panel["knowledge_cutoff_at"]
        frozen_at = (
            datetime.fromisoformat(raw_frozen)
            if isinstance(raw_frozen, str)
            else cast(datetime, raw_frozen)
        )
        return source, resolution, frozen_at


__all__ = [
    "DynamicPanelAlphaDevelopmentService",
    "DynamicPanelDevelopmentError",
    "DynamicPanelDevelopmentRequest",
    "DynamicPanelSourceResolution",
]
