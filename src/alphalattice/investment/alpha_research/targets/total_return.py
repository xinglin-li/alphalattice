"""Installed whole-Universe total-return target for development research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from typing import Literal, Self, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.quant.cross_section import (
    median_mad_winsor,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .execution_outcome import AlphaTargetBoundaryError

type FloatArray = npt.NDArray[np.float64]

TOTAL_RETURN_ALPHA_TARGET_RECIPE_ID = "CROSS_SECTIONAL_TOTAL_RETURN_STD_Z"
TOTAL_RETURN_OUTLIER_POLICY_ID = "UNIVERSE_RETURN_NORMALIZED_MAD_WINSOR_3_5"
TOTAL_RETURN_CENTERING_ID = "WHOLE_UNIVERSE_CENTER_ONCE_AFTER_BOUNDING"
TOTAL_RETURN_STANDARDIZATION_ID = "CROSS_SECTIONAL_STD_Z"
TOTAL_RETURN_CROSS_SECTIONAL_AUTHORITY_ID = "WHOLE_ACTIVE_UNIVERSE"

_REQUIRED_COLUMNS = frozenset(
    {
        "formation_session",
        "holding_end_session",
        "listing_id",
        "fit_target",
        "simple_economic_return",
    }
)
_HASH = r"^[0-9a-f]{64}$"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class TotalReturnAlphaTargetRecipe(_Contract):
    """Identity of the no-Sector total-return transformation."""

    kind: Literal["TotalReturnAlphaTargetRecipe"] = "TotalReturnAlphaTargetRecipe"
    target_recipe_id: Literal["CROSS_SECTIONAL_TOTAL_RETURN_STD_Z"] = (
        "CROSS_SECTIONAL_TOTAL_RETURN_STD_Z"
    )
    execution_outcome_recipe_id: str = Field(min_length=1, max_length=96)
    outlier_policy_id: Literal["UNIVERSE_RETURN_NORMALIZED_MAD_WINSOR_3_5"] = (
        "UNIVERSE_RETURN_NORMALIZED_MAD_WINSOR_3_5"
    )
    centering_id: Literal["WHOLE_UNIVERSE_CENTER_ONCE_AFTER_BOUNDING"] = (
        "WHOLE_UNIVERSE_CENTER_ONCE_AFTER_BOUNDING"
    )
    standardization_id: Literal["CROSS_SECTIONAL_STD_Z"] = "CROSS_SECTIONAL_STD_Z"
    cross_sectional_authority_id: Literal["WHOLE_ACTIVE_UNIVERSE"] = "WHOLE_ACTIVE_UNIVERSE"
    source_semantics: Literal["LOG_EXECUTION_RETURN"] = "LOG_EXECUTION_RETURN"
    economic_return_semantics: Literal["SIMPLE_EXECUTION_RETURN"] = "SIMPLE_EXECUTION_RETURN"
    sequence: tuple[
        Literal[
            "universe_return_normalized_mad_winsor_3_5",
            "whole_universe_center_once",
            "cross_sectional_sample_std_z_ddof_1",
        ],
        ...,
    ]
    winsor_multiplier: float = Field(default=3.5, ge=3.5, le=3.5)
    mad_scale: float = Field(default=1.4826, ge=1.4826, le=1.4826)
    dispersion_ddof: Literal[1] = 1
    minimum_coverage: float = Field(gt=0.0, le=1.0)
    minimum_eligible_rows: int = Field(ge=2)
    coverage_transform: Literal["REFERENCE_FIT_COVERAGE_APPLY"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    recipe_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the installed whole-universe sequence and exact recipe identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaTargetBoundaryError: The bounded/centered/sample-z sequence, winsor/MAD constants
                or canonical recipe hash differs.
        """
        if (
            self.sequence
            != (
                "universe_return_normalized_mad_winsor_3_5",
                "whole_universe_center_once",
                "cross_sectional_sample_std_z_ddof_1",
            )
            or self.winsor_multiplier != 3.5
            or self.mad_scale != 1.4826
        ):
            raise AlphaTargetBoundaryError("alpha_research.total_return_target_sequence_invalid")
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise AlphaTargetBoundaryError(
                "alpha_research.total_return_target_recipe_identity_invalid"
            )
        return self


def build_total_return_alpha_target_recipe(
    *,
    execution_outcome_recipe_id: str,
    minimum_coverage: float = 0.98,
    reference_coverage: bool = False,
) -> TotalReturnAlphaTargetRecipe:
    """Seal the installed whole-universe total-return target composition.

    Args:
        execution_outcome_recipe_id: Exact execution outcome source recipe.
        minimum_coverage: Declared required formation coverage.
        reference_coverage: Select reference-fit/coverage-apply population semantics.

    Returns:
        Validated target recipe with one centering step, sample dispersion and canonical identity.
    """
    values: dict[str, object] = {
        "kind": "TotalReturnAlphaTargetRecipe",
        "target_recipe_id": TOTAL_RETURN_ALPHA_TARGET_RECIPE_ID,
        "execution_outcome_recipe_id": execution_outcome_recipe_id,
        "outlier_policy_id": TOTAL_RETURN_OUTLIER_POLICY_ID,
        "centering_id": TOTAL_RETURN_CENTERING_ID,
        "standardization_id": TOTAL_RETURN_STANDARDIZATION_ID,
        "cross_sectional_authority_id": TOTAL_RETURN_CROSS_SECTIONAL_AUTHORITY_ID,
        "source_semantics": "LOG_EXECUTION_RETURN",
        "economic_return_semantics": "SIMPLE_EXECUTION_RETURN",
        "sequence": (
            "universe_return_normalized_mad_winsor_3_5",
            "whole_universe_center_once",
            "cross_sectional_sample_std_z_ddof_1",
        ),
        "winsor_multiplier": 3.5,
        "mad_scale": 1.4826,
        "dispersion_ddof": 1,
        "minimum_coverage": minimum_coverage,
        "minimum_eligible_rows": 2,
    }
    if reference_coverage:
        values["coverage_transform"] = "REFERENCE_FIT_COVERAGE_APPLY"
    return TotalReturnAlphaTargetRecipe(**values, recipe_hash=str(canonical_hash(values)))


def total_return_array_identity(values: FloatArray) -> str:
    """Hash exact normalized float64 target-array dtype, shape and content.

    Args:
        values: Target values converted to contiguous float64 storage.

    Returns:
        Canonical metadata identity containing the SHA-256 of normalized array bytes.
    """
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return str(
        canonical_hash(
            {
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "content": sha256(contiguous.tobytes()).hexdigest(),
            }
        )
    )


class TotalReturnTargetLaneIdentity(_Contract):
    """Bind raw, bounded, centered and standardized return lanes to exact axes.

    Economic returns, dispersion and clipping boundaries retain distinct identities. Holding-end
    dates and ordered session/listing axes are bound; reference-population authority is explicit
    when present.
    """

    kind: Literal["TotalReturnTargetLaneIdentity"] = "TotalReturnTargetLaneIdentity"
    raw_log_execution_return_identity: str = Field(pattern=_HASH)
    simple_economic_return_identity: str = Field(pattern=_HASH)
    bounded_log_execution_return_identity: str = Field(pattern=_HASH)
    universe_centered_return_identity: str = Field(pattern=_HASH)
    cross_sectional_dispersion_identity: str = Field(pattern=_HASH)
    fit_target_identity: str = Field(pattern=_HASH)
    clipping_boundary_identity: str = Field(pattern=_HASH)
    holding_end_sessions_hash: str = Field(pattern=_HASH)
    ordered_sessions_hash: str = Field(pattern=_HASH)
    ordered_listing_ids_hash: str = Field(pattern=_HASH)
    reference_population_hash: str | None = Field(
        default=None, pattern=_HASH, exclude_if=lambda value: value is None
    )
    lane_identity_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact total-return lane and axis identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaTargetBoundaryError: The canonical declared lane payload differs from its identity
                hash.
        """
        if self.lane_identity_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"lane_identity_hash"})
        ):
            raise AlphaTargetBoundaryError(
                "alpha_research.total_return_target_lane_identity_invalid"
            )
        return self


class TotalReturnAlphaTargetEvidence(_Contract):
    """Realized target, clipping and scale evidence for one durable surface."""

    kind: Literal["TotalReturnAlphaTargetEvidence"] = "TotalReturnAlphaTargetEvidence"
    identity_class: Literal["DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"] = (
        "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"
    )
    recipe_binding_hash: str = Field(pattern=_HASH)
    lane_identity_hash: str = Field(pattern=_HASH)
    raw_log_execution_return_identity: str = Field(pattern=_HASH)
    simple_economic_return_identity: str = Field(pattern=_HASH)
    bounded_log_execution_return_identity: str = Field(pattern=_HASH)
    universe_centered_return_identity: str = Field(pattern=_HASH)
    cross_sectional_dispersion_identity: str = Field(pattern=_HASH)
    fit_target_identity: str = Field(pattern=_HASH)
    clipping_boundary_identity: str = Field(pattern=_HASH)
    holding_end_sessions_hash: str = Field(pattern=_HASH)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=2)
    cross_sectional_dispersion: tuple[float | None, ...] = Field(min_length=1)
    clipped_listing_count: tuple[int, ...] = Field(min_length=1)
    clipped_fraction: tuple[float, ...] = Field(min_length=1)
    absolute_z_above_3_count: int = Field(ge=0)
    absolute_z_above_4_count: int = Field(ge=0)
    absolute_z_above_5_count: int = Field(ge=0)
    tail_squared_loss_share_above_3: float = Field(ge=0.0, le=1.0)
    evidence_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal declared total-return target evidence.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical evidence_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = dict(values)
        draft.pop("evidence_hash", None)
        provisional = cls.model_construct(**draft, evidence_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"evidence_hash"})
        return cls(**draft, evidence_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require aligned formation evidence and nested absolute-z exceedance counts.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaTargetBoundaryError: Formation order/evidence lengths, nested exceedance counts or
                the canonical evidence hash differs.
        """
        session_count = len(self.formation_sessions)
        if (
            self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or len(self.cross_sectional_dispersion) != session_count
            or len(self.clipped_listing_count) != session_count
            or len(self.clipped_fraction) != session_count
            or not (
                self.absolute_z_above_5_count
                <= self.absolute_z_above_4_count
                <= self.absolute_z_above_3_count
            )
        ):
            raise AlphaTargetBoundaryError(
                "alpha_research.total_return_target_evidence_axis_invalid"
            )
        if self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise AlphaTargetBoundaryError(
                "alpha_research.total_return_target_evidence_identity_invalid"
            )
        return self


@dataclass(frozen=True, slots=True)
class TotalReturnAlphaTargetSurface:
    """Retain compiled fit targets, dispersion, bounds and total-return lane authority."""

    recipe: TotalReturnAlphaTargetRecipe
    targets: pa.Table
    dispersion: pa.Table
    bounds: pa.Table
    lane_identity: TotalReturnTargetLaneIdentity
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]


def _return_reference_statistics(
    raw: FloatArray, recipe: TotalReturnAlphaTargetRecipe
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    bounded, _median, mad, lower, upper = median_mad_winsor(
        raw, multiplier=recipe.winsor_multiplier, mad_scale=recipe.mad_scale
    )
    with np.errstate(invalid="ignore"):
        center = np.nanmean(bounded, axis=1)
    centered = bounded - center[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        dispersion = np.nanstd(centered, axis=1, ddof=recipe.dispersion_ddof)
    return bounded, center, dispersion, mad, lower, upper


def compile_total_return_alpha_target_surface(
    *,
    source_table: pa.Table,
    recipe: TotalReturnAlphaTargetRecipe,
    reference_eligible: npt.NDArray[np.bool_] | None = None,
    nominal_member_count: npt.NDArray[np.int64] | None = None,
) -> TotalReturnAlphaTargetSurface:
    """Compile bounded total-return Z; masks align to sorted session/listing axes.

    The explicit coverage extension fits the dated reference and applies those
    exact bounds/center/scale to other covered rows. It grants no training
    membership. Nominal counts remain the coverage denominator before exclusions.
    """
    recipe = TotalReturnAlphaTargetRecipe.model_validate_json(recipe.model_dump_json())
    missing = _REQUIRED_COLUMNS - set(source_table.column_names)
    if missing:
        raise AlphaTargetBoundaryError("alpha_research.total_return_target_source_columns_missing")
    ordered = source_table.sort_by(
        [("formation_session", "ascending"), ("listing_id", "ascending")]
    ).combine_chunks()
    session_values = tuple(cast(list[date], ordered["formation_session"].to_pylist()))
    holding_end_values = tuple(cast(list[date], ordered["holding_end_session"].to_pylist()))
    listing_values = tuple(str(value) for value in ordered["listing_id"].to_pylist())
    if len(set(zip(session_values, listing_values, strict=True))) != ordered.num_rows or any(
        holding_end <= formation
        for formation, holding_end in zip(session_values, holding_end_values, strict=True)
    ):
        raise AlphaTargetBoundaryError("alpha_research.total_return_target_source_duplicated")
    sessions = tuple(sorted(set(session_values)))
    listings = tuple(sorted(set(listing_values)))
    if ordered.num_rows != len(sessions) * len(listings):
        raise AlphaTargetBoundaryError("alpha_research.total_return_target_surface_incomplete")
    raw = np.asarray(
        ordered["fit_target"].to_numpy(zero_copy_only=False), dtype=np.float64
    ).reshape(len(sessions), len(listings))
    simple = np.asarray(
        ordered["simple_economic_return"].to_numpy(zero_copy_only=False),
        dtype=np.float64,
    ).reshape(len(sessions), len(listings))
    if bool(np.isinf(raw).any()) or bool(np.isinf(simple).any()):
        raise AlphaTargetBoundaryError("alpha_research.total_return_target_source_nonfinite")

    if (reference_eligible is not None) != (recipe.coverage_transform is not None) or (
        nominal_member_count is not None and reference_eligible is None
    ):
        raise AlphaTargetBoundaryError("alpha_research.total_return_reference_contract_mismatch")
    reference_identity = None
    if reference_eligible is None:
        bounded, center, dispersion, mad, lower, upper = _return_reference_statistics(raw, recipe)
        finite_count = np.isfinite(raw).sum(axis=1)
        nominal: npt.NDArray[np.int64] = np.full(len(sessions), len(listings), dtype=np.int64)
    else:
        if reference_eligible.shape != raw.shape or reference_eligible.dtype != np.bool_:
            raise AlphaTargetBoundaryError("alpha_research.total_return_reference_axis_invalid")
        nominal = (
            reference_eligible.sum(axis=1) if nominal_member_count is None else nominal_member_count
        )
        if (
            nominal.shape != (len(sessions),)
            or nominal.dtype != np.int64
            or np.any(nominal < reference_eligible.sum(axis=1))
            or np.any(nominal > len(listings))
            or np.any(nominal < 1)
        ):
            raise AlphaTargetBoundaryError("alpha_research.total_return_nominal_population_invalid")
        center, dispersion, mad, lower, upper = (
            np.full(len(sessions), np.nan, dtype=np.float64) for _ in range(5)
        )
        masks, inverse = np.unique(reference_eligible, axis=0, return_inverse=True)
        for index, mask in enumerate(masks):
            days, members = np.flatnonzero(inverse == index), np.flatnonzero(mask)
            if not len(members):
                continue
            _, c, d, m, lo, hi = _return_reference_statistics(
                np.ascontiguousarray(raw[np.ix_(days, members)]), recipe
            )
            center[days], dispersion[days], mad[days], lower[days], upper[days] = c, d, m, lo, hi
        bounded = np.clip(raw, lower[:, None], upper[:, None])
        finite_count = (reference_eligible & np.isfinite(raw)).sum(axis=1)
        reference_identity = str(
            canonical_hash(
                {
                    "eligible": total_return_array_identity(reference_eligible.astype(np.float64)),
                    "nominal": total_return_array_identity(nominal.astype(np.float64)),
                    "coverage_transform": recipe.coverage_transform,
                }
            )
        )
    centered = bounded - center[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        target = centered / dispersion[:, None]
    coverage = finite_count / nominal
    available = (
        (coverage >= recipe.minimum_coverage)
        & (finite_count >= recipe.minimum_eligible_rows)
        & np.isfinite(mad)
        & (mad != 0.0)
        & np.isfinite(dispersion)
        & (dispersion != 0.0)
    )
    admitted = np.where(available[:, None] & np.isfinite(target), target, np.nan)
    admitted_dispersion = np.where(available, dispersion, np.nan)
    clipped = np.isfinite(raw) & ((raw < lower[:, None]) | (raw > upper[:, None]))
    if reference_eligible is not None:
        clipped &= reference_eligible
    reasons = tuple(
        None
        if available[index]
        else (
            "COVERAGE_BELOW_MINIMUM"
            if coverage[index] < recipe.minimum_coverage
            else (
                "ELIGIBLE_ROWS_BELOW_MINIMUM"
                if finite_count[index] < recipe.minimum_eligible_rows
                else (
                    "SOURCE_MAD_INVALID"
                    if not np.isfinite(mad[index]) or mad[index] == 0.0
                    else "CROSS_SECTIONAL_DISPERSION_INVALID"
                )
            )
        )
        for index in range(len(sessions))
    )
    targets = pa.table(
        {
            "formation_session": ordered["formation_session"],
            "holding_end_session": ordered["holding_end_session"],
            "listing_id": ordered["listing_id"],
            "fit_target": pa.array(admitted.reshape(-1), type=pa.float64()),
            "raw_log_execution_return": ordered["fit_target"],
            "simple_economic_return": ordered["simple_economic_return"],
            "bounded_log_execution_return": pa.array(bounded.reshape(-1), type=pa.float64()),
            "universe_centered_return": pa.array(centered.reshape(-1), type=pa.float64()),
        }
    ).combine_chunks()
    dispersion_table = pa.table(
        {
            "formation_session": pa.array(sessions, type=pa.date32()),
            "cross_sectional_dispersion": pa.array(
                admitted_dispersion, mask=~available, type=pa.float64()
            ),
            "eligible_listing_count": pa.array(finite_count, type=pa.int64()),
            "clipped_listing_count": pa.array(clipped.sum(axis=1), type=pa.int64()),
            "clipped_fraction": pa.array(
                clipped.sum(axis=1) / np.maximum(finite_count, 1), type=pa.float64()
            ),
            "available": pa.array(available, type=pa.bool_()),
            "unavailable_reason": pa.array(reasons, type=pa.string()),
            **(
                {"nominal_listing_count": pa.array(nominal), "coverage": pa.array(coverage)}
                if reference_eligible is not None
                else {}
            ),
        }
    ).combine_chunks()
    bounds = pa.table(
        {
            "formation_session": pa.array(sessions, type=pa.date32()),
            "lower_bound": pa.array(lower, type=pa.float64()),
            "upper_bound": pa.array(upper, type=pa.float64()),
        }
    ).combine_chunks()
    boundary_identity = str(
        canonical_hash(
            [(float(left), float(right)) for left, right in zip(lower, upper, strict=True)]
        )
    )
    lane_values: dict[str, object] = {
        "kind": "TotalReturnTargetLaneIdentity",
        "raw_log_execution_return_identity": total_return_array_identity(raw),
        "simple_economic_return_identity": total_return_array_identity(simple),
        "bounded_log_execution_return_identity": total_return_array_identity(bounded),
        "universe_centered_return_identity": total_return_array_identity(centered),
        "cross_sectional_dispersion_identity": total_return_array_identity(admitted_dispersion),
        "fit_target_identity": total_return_array_identity(admitted),
        "clipping_boundary_identity": boundary_identity,
        "holding_end_sessions_hash": str(
            canonical_hash([value.isoformat() for value in holding_end_values])
        ),
        "ordered_sessions_hash": str(canonical_hash([value.isoformat() for value in sessions])),
        "ordered_listing_ids_hash": str(canonical_hash(list(listings))),
    }
    if reference_identity is not None:
        lane_values["reference_population_hash"] = reference_identity
    return TotalReturnAlphaTargetSurface(
        recipe=recipe,
        targets=targets,
        dispersion=dispersion_table,
        bounds=bounds,
        lane_identity=TotalReturnTargetLaneIdentity(
            **lane_values, lane_identity_hash=str(canonical_hash(lane_values))
        ),
        formation_sessions=sessions,
        ordered_listing_ids=listings,
    )


__all__ = [
    "TOTAL_RETURN_ALPHA_TARGET_RECIPE_ID",
    "TOTAL_RETURN_CENTERING_ID",
    "TOTAL_RETURN_CROSS_SECTIONAL_AUTHORITY_ID",
    "TOTAL_RETURN_OUTLIER_POLICY_ID",
    "TOTAL_RETURN_STANDARDIZATION_ID",
    "TotalReturnAlphaTargetEvidence",
    "TotalReturnAlphaTargetRecipe",
    "TotalReturnAlphaTargetSurface",
    "TotalReturnTargetLaneIdentity",
    "build_total_return_alpha_target_recipe",
    "compile_total_return_alpha_target_surface",
    "total_return_array_identity",
]
