"""The h1 Dynamic Panel's historical records and the source arrays they were built from.

A historical run sealed a fold dossier and a scale receipt for each block; the paired panel
route (V1) and the monthly refit research read that failed baseline back, and resolve its
source arrays. The catalog, its compile and fold materialization retired with V19
(V19's retained retirement rationale).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type DynamicPanelViewId = Literal[
    "RELATIVE_FACTOR_CONTROL",
    "DYNAMIC_RELATIVE_PANEL",
    "DYNAMIC_CONTEXT_PANEL",
    "DYNAMIC_JOINT_PANEL",
]
type DynamicPanelRole = Literal[
    "RELATIVE_STOCK",
    "ABSOLUTE_STOCK",
    "STOCK_LAGGED_RETURN",
    "MARKET_CONTEXT",
    "SECTOR_CONTEXT",
    "SECTOR_CATEGORY",
]

_HASH = r"^[0-9a-f]{64}$"


class DynamicPanelBoundaryError(ValueError):
    """Fail-closed boundary for Dynamic Panel compilation and replay."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


def dynamic_panel_array_identity(values: npt.NDArray[np.generic]) -> str:
    """Return a shape-, dtype- and NaN-stable numerical identity."""
    array = np.ascontiguousarray(values)
    finite = np.isfinite(array)
    return str(
        canonical_hash(
            {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "finite_mask": sha256(finite.tobytes()).hexdigest(),
                "finite_values": sha256(
                    np.ascontiguousarray(np.where(finite, array, 0.0), dtype=array.dtype).tobytes()
                ).hexdigest(),
            }
        )
    )


def _contract[T: _Contract](model: type[T], values: dict[str, object], field: str) -> T:
    provisional = model.model_construct(**values, **{field: "0" * 64})
    identity = provisional.model_dump(mode="json", exclude={field})
    return model(**values, **{field: str(canonical_hash(identity))})


class DynamicPanelScaleReceipt(_Contract):
    """Realized final-scale identity and clipping evidence for one block."""

    kind: Literal["DynamicPanelScaleReceipt"] = "DynamicPanelScaleReceipt"
    block_id: str
    role: DynamicPanelRole
    final_scale_policy_id: str
    ordered_feature_ids: tuple[str, ...] = Field(min_length=1)
    fit_session_axis_hash: str | None = Field(default=None, pattern=_HASH)
    scaler_parameter_hash: str | None = Field(default=None, pattern=_HASH)
    scaler_parameters: dict[str, tuple[float | None, ...]] | None = None
    input_array_identity: str = Field(pattern=_HASH)
    output_array_identity: str = Field(pattern=_HASH)
    per_feature_finite_count: tuple[int, ...] = Field(min_length=1)
    per_feature_clip_count: tuple[int, ...] = Field(min_length=1)
    per_feature_session_clip_counts: tuple[tuple[int, ...], ...] = Field(min_length=1)
    future_fit_violation_count: int = Field(ge=0)
    receipt_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the declared causal dynamic-panel scaling receipt.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical receipt_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return _contract(cls, dict(values), "receipt_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require aligned diagnostics, causal scaling and exact scaler/receipt identities.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            DynamicPanelBoundaryError: Feature diagnostic lengths differ, a future-fit violation
                exists, scaler authority is partial or scaler/receipt identity is inconsistent.
        """
        feature_count = len(self.ordered_feature_ids)
        if (
            len(self.per_feature_finite_count) != feature_count
            or len(self.per_feature_clip_count) != feature_count
            or len(self.per_feature_session_clip_counts) != feature_count
            or self.future_fit_violation_count != 0
            or bool(self.scaler_parameter_hash) != bool(self.scaler_parameters)
            or (
                self.scaler_parameters is not None
                and self.scaler_parameter_hash != canonical_hash(self.scaler_parameters)
            )
            or self.receipt_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"receipt_hash"}))
        ):
            raise DynamicPanelBoundaryError("alpha_research.dynamic_panel_scale_receipt_invalid")
        return self


@dataclass(frozen=True, slots=True)
class DynamicPanelSourceArrays:
    """Retain dated listing-by-feature Panel values and causal target/source identities.

    Relative and raw factor tensors use declared feature axes. Target/raw/simple-return lanes use
    session-by-listing rows; absolute-state values retain two cells per row. Classification covers
    every listing and holding ends follow formations.
    """

    formation_sessions: tuple[date, ...]
    holding_end_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    ordered_factor_ids: tuple[str, ...]
    ordered_relative_factor_ids: tuple[str, ...]
    sector_by_listing_id: Mapping[str, str]
    relative_factor_values: FloatArray
    raw_factor_values: FloatArray
    target_z_values: FloatArray
    raw_log_returns: FloatArray
    simple_economic_returns: FloatArray
    raw_absolute_state_values: FloatArray
    relative_surface_hash: str
    raw_formula_surface_hash: str
    total_return_target_evidence_hash: str
    outcome_method_binding_hash: str

    def __post_init__(self) -> None:
        """Require canonical aligned Panel axes, source tensor shapes and future holding-end clocks.

        Raises:
            DynamicPanelBoundaryError: Session/listing/factor/classification axes, source shapes or
                holding-end ordering violate the declared Panel contract.
        """
        absolute_shape = (
            len(self.formation_sessions),
            len(self.ordered_listing_ids),
            len(self.ordered_factor_ids),
        )
        relative_shape = (
            len(self.formation_sessions),
            len(self.ordered_listing_ids),
            len(self.ordered_relative_factor_ids),
        )
        row_shape = absolute_shape[:2]
        if (
            self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or len(self.holding_end_sessions) != len(self.formation_sessions)
            or self.ordered_listing_ids != tuple(sorted(set(self.ordered_listing_ids)))
            or self.ordered_factor_ids != tuple(dict.fromkeys(self.ordered_factor_ids))
            or self.ordered_relative_factor_ids
            != tuple(
                value
                for value in self.ordered_factor_ids
                if value in set(self.ordered_relative_factor_ids)
            )
            or not self.ordered_relative_factor_ids
            or set(self.sector_by_listing_id) != set(self.ordered_listing_ids)
            or self.relative_factor_values.shape != relative_shape
            or self.raw_factor_values.shape != absolute_shape
            or self.target_z_values.shape != row_shape
            or self.raw_log_returns.shape != row_shape
            or self.simple_economic_returns.shape != row_shape
            or self.raw_absolute_state_values.shape != (*row_shape, 2)
        ):
            raise DynamicPanelBoundaryError("alpha_research.dynamic_panel_source_axis_invalid")
        if any(
            holding_end <= formation
            for formation, holding_end in zip(
                self.formation_sessions, self.holding_end_sessions, strict=True
            )
        ):
            raise DynamicPanelBoundaryError("alpha_research.dynamic_panel_outcome_clock_invalid")


class DynamicPanelFoldDossier(_Contract):
    """Qualification evidence for one view and outer fold."""

    kind: Literal["DynamicPanelFoldDossier"] = "DynamicPanelFoldDossier"
    view_id: DynamicPanelViewId
    view_recipe_hash: str = Field(pattern=_HASH)
    catalog_hash: str = Field(pattern=_HASH)
    fold_index: int = Field(ge=0)
    training_sessions: tuple[date, ...] = Field(min_length=1)
    validation_sessions: tuple[date, ...] = Field(min_length=1)
    ordered_feature_ids: tuple[str, ...] = Field(min_length=1, max_length=512)
    ordered_training_row_axis_hash: str = Field(pattern=_HASH)
    ordered_validation_row_axis_hash: str = Field(pattern=_HASH)
    training_matrix_identity: str = Field(pattern=_HASH)
    validation_matrix_identity: str = Field(pattern=_HASH)
    target_identity: str = Field(pattern=_HASH)
    raw_economic_return_identity: str = Field(pattern=_HASH)
    scale_receipt_hashes: tuple[str, ...] = Field(min_length=1)
    finite_input_count: int = Field(ge=1)
    nonfinite_model_input_count: Literal[0] = 0
    nonfinite_target_count: Literal[0] = 0
    future_fit_violation_count: Literal[0] = 0
    same_shape_wrong_lane_observed: Literal[False] = False
    receipt_array_reconciled: Literal[True] = True
    relative_sector_mean_max_abs: float = Field(ge=0.0)
    relative_sector_mean_tolerance: float = Field(gt=0.0)
    target_tail_squared_loss_share_above_3: float = Field(ge=0.0, le=1.0)
    constant_feature_ids: tuple[str, ...]
    near_constant_feature_ids: tuple[str, ...]
    train_validation_mean_drift_max_abs: float = Field(ge=0.0)
    train_p001: float
    train_p01: float
    train_p50: float
    train_p99: float
    train_p999: float
    train_mean: float
    train_std: float = Field(gt=0.0)
    train_min: float
    train_max: float
    absolute_input_above_3_count: int = Field(ge=0)
    absolute_input_above_4_count: int = Field(ge=0)
    absolute_input_above_5_count: int = Field(ge=0)
    dossier_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the declared dynamic-panel fold dossier.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical dossier_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return _contract(cls, dict(values), "dossier_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require relative-sector neutrality within tolerance and exact fold dossier identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            DynamicPanelBoundaryError: The maximum absolute sector mean exceeds tolerance or
                dossier_hash is inconsistent.
        """
        if (
            self.relative_sector_mean_max_abs > self.relative_sector_mean_tolerance
            or self.dossier_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"dossier_hash"}))
        ):
            raise DynamicPanelBoundaryError("alpha_research.dynamic_panel_fold_dossier_invalid")
        return self


__all__ = [
    "DynamicPanelBoundaryError",
    "DynamicPanelFoldDossier",
    "DynamicPanelScaleReceipt",
    "DynamicPanelSourceArrays",
    "DynamicPanelViewId",
    "dynamic_panel_array_identity",
]
