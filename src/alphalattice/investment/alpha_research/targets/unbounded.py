"""The canonical composition with the bounding step removed, for sensitivity only.

This exists to answer one question -- what does bounding the residual actually
change -- and it is a control, never a candidate. It is not eligible for G4, not
eligible for current, and the return-signal owner refuses it structurally by
requiring the canonical recipe id rather than by consulting a list.

It is a separate recipe type with its own hash rather than a flag on the
canonical recipe. A boolean would make two different methods share one identity,
so a surface compiled without bounding would be indistinguishable from one
compiled with it wherever the flag was not also carried -- which is every place
that stores a recipe hash.

The composition it runs:

    raw constituent log execution returns
    -> equal-weight raw Sector centre
    -> raw Sector residual
    -> equal-Sector re-demean, verified to be a no-op
    -> global ordinary cross-sectional standard deviation, ddof = 1
    -> standardized target Z

The fourth step is where the two methods differ in kind rather than in degree.
In the canonical target the re-demeaning is a genuine repair, because clipping an
asymmetric tail reintroduces a Sector mean. Here nothing was clipped, so the
residual is already Sector-neutral and the second demeaning must be arithmetically
idle. Running it anyway and *requiring* it to be idle is what turns "we did not
need this step" into a checked property instead of an assumption.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from typing import Literal, Self, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.quant.sector_history import sector_positions
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .catalog import AlphaTargetCatalog, build_installed_alpha_target_catalog
from .execution_outcome import AlphaTargetBoundaryError
from .sector_runs import below_sector_sample, demean_by_run
from .standardization import (
    CROSS_SECTIONAL_STD_Z_DDOF,
    CROSS_SECTIONAL_STD_Z_STANDARDIZATION_ID,
)

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]

UNBOUNDED_SENSITIVITY_TARGET_RECIPE_ID = "SECTOR_RESIDUAL_UNBOUNDED_CROSS_SECTIONAL_STD_Z"

#: How far the verified-idle re-demeaning may move a value before the surface is
#: refused. Not zero: the two demeanings are separate float64 reductions over the
#: same numbers, so exact equality would fail on rounding rather than on method.
REDEMEAN_INVARIANCE_TOLERANCE = 1e-12

_REQUIRED_COLUMNS = frozenset(
    {"formation_session", "listing_id", "fit_target", "simple_economic_return"}
)

type UnboundedTargetUnavailableReason = Literal[
    "SOURCE_TARGET_MISSING",
    "COVERAGE_BELOW_MINIMUM",
    "SECTOR_SAMPLE_BELOW_MINIMUM",
    "ELIGIBLE_ROWS_BELOW_MINIMUM",
    "RESIDUAL_DISPERSION_ZERO",
    "RESIDUAL_DISPERSION_NONFINITE",
]


class UnboundedSensitivityAlphaTargetRecipe(BaseModel):  # type: ignore[misc]
    """The complete unbounded method identity.

    It deliberately carries no ``winsor_multiplier`` and no outlier policy id.
    Carrying them as nulls would let a reader believe a bound was configured and
    happened not to bind, when in fact this method has no bounding step at all.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["UnboundedSensitivityAlphaTargetRecipe"] = "UnboundedSensitivityAlphaTargetRecipe"
    target_recipe_id: Literal["SECTOR_RESIDUAL_UNBOUNDED_CROSS_SECTIONAL_STD_Z"] = (
        "SECTOR_RESIDUAL_UNBOUNDED_CROSS_SECTIONAL_STD_Z"
    )
    execution_outcome_recipe_id: str = Field(min_length=1, max_length=96)
    outlier_policy_id: Literal["NONE_RESIDUAL_UNBOUNDED"] = "NONE_RESIDUAL_UNBOUNDED"
    neutralization_id: Literal["EQUAL_SECTOR_DEMEAN_VERIFIED_IDEMPOTENT"] = (
        "EQUAL_SECTOR_DEMEAN_VERIFIED_IDEMPOTENT"
    )
    standardization_id: Literal["CROSS_SECTIONAL_STD_Z"] = "CROSS_SECTIONAL_STD_Z"
    source_semantics: Literal["LOG_EXECUTION_RETURN"] = "LOG_EXECUTION_RETURN"
    economic_return_semantics: Literal["SIMPLE_EXECUTION_RETURN"] = "SIMPLE_EXECUTION_RETURN"
    research_role: Literal["SENSITIVITY_CONTROL_NOT_CANONICAL"] = (
        "SENSITIVITY_CONTROL_NOT_CANONICAL"
    )
    """Stated inside the identity, so a surface compiled by this method cannot be
    presented downstream as a candidate without the claim travelling with it."""

    sequence: tuple[str, ...] = Field(min_length=1)
    redemean_invariance_tolerance: float = Field(gt=0.0)
    dispersion_ddof: int = Field(ge=0, le=1)
    minimum_coverage: float = Field(gt=0.0, le=1.0)
    minimum_sector_sample: int = Field(ge=1)
    minimum_eligible_rows: int = Field(ge=2)
    sector_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    sector_source: str = Field(min_length=1, max_length=96)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the installed unbounded sensitivity sequence and dispersion convention.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaTargetBoundaryError: Sequence, dispersion ddof or canonical recipe identity
                differs.
        """
        if self.sequence != (
            "equal_sector_center",
            "raw_sector_residual",
            "equal_sector_redemean_verified_idle",
            "cross_sectional_std_z",
        ):
            raise AlphaTargetBoundaryError("alpha_research.unbounded_target_sequence_invalid")
        if self.dispersion_ddof != CROSS_SECTIONAL_STD_Z_DDOF:
            raise AlphaTargetBoundaryError("alpha_research.unbounded_target_ddof_invalid")
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise AlphaTargetBoundaryError(
                "alpha_research.unbounded_target_recipe_identity_invalid"
            )
        return self


def build_unbounded_sensitivity_target_recipe(
    *,
    execution_outcome_recipe_id: str,
    sector_revision: str,
    sector_source: str = "YAHOO_CURRENT_SECTOR",
    minimum_coverage: float = 0.98,
    minimum_sector_sample: int = 5,
) -> UnboundedSensitivityAlphaTargetRecipe:
    """Seal the installed sensitivity control against one outcome method and revision.

    The availability constants default to the canonical values on purpose: a
    paired comparison in which the two arms admitted different formations would
    be measuring eligibility, not bounding.
    """
    values: dict[str, object] = {
        "kind": "UnboundedSensitivityAlphaTargetRecipe",
        "target_recipe_id": UNBOUNDED_SENSITIVITY_TARGET_RECIPE_ID,
        "execution_outcome_recipe_id": execution_outcome_recipe_id,
        "outlier_policy_id": "NONE_RESIDUAL_UNBOUNDED",
        "neutralization_id": "EQUAL_SECTOR_DEMEAN_VERIFIED_IDEMPOTENT",
        "standardization_id": CROSS_SECTIONAL_STD_Z_STANDARDIZATION_ID,
        "source_semantics": "LOG_EXECUTION_RETURN",
        "economic_return_semantics": "SIMPLE_EXECUTION_RETURN",
        "research_role": "SENSITIVITY_CONTROL_NOT_CANONICAL",
        "sequence": [
            "equal_sector_center",
            "raw_sector_residual",
            "equal_sector_redemean_verified_idle",
            "cross_sectional_std_z",
        ],
        "redemean_invariance_tolerance": REDEMEAN_INVARIANCE_TOLERANCE,
        "dispersion_ddof": CROSS_SECTIONAL_STD_Z_DDOF,
        "minimum_coverage": minimum_coverage,
        "minimum_sector_sample": minimum_sector_sample,
        "minimum_eligible_rows": 2,
        "sector_revision": sector_revision,
        "sector_source": sector_source,
    }
    return UnboundedSensitivityAlphaTargetRecipe(**values, recipe_hash=canonical_hash(values))


def _lane_identity(values: FloatArray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return str(
        canonical_hash(
            {
                "dtype": str(contiguous.dtype),
                "shape": [int(value) for value in contiguous.shape],
                "content": sha256(contiguous.tobytes()).hexdigest(),
            }
        )
    )


class UnboundedSensitivityLaneIdentity(BaseModel):  # type: ignore[misc]
    """Each retained lane of the unbounded composition, identified separately."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["UnboundedSensitivityLaneIdentity"] = "UnboundedSensitivityLaneIdentity"
    raw_log_execution_return_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    simple_economic_return_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    sector_center_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_residual_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    redemeaned_residual_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    cross_sectional_dispersion_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_target_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximum_redemean_correction: float = Field(ge=0.0)
    """The largest absolute change the idle re-demeaning made, retained as
    evidence that it was idle rather than as a claim that it was."""

    ordered_sessions_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    lane_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact unbounded sensitivity lane identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaTargetBoundaryError: The canonical lane payload differs from its declared hash.
        """
        if self.lane_identity_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"lane_identity_hash"})
        ):
            raise AlphaTargetBoundaryError("alpha_research.unbounded_target_lane_identity_invalid")
        return self


@dataclass(frozen=True, slots=True)
class UnboundedSensitivityTargetSurface:
    """One compiled unbounded surface and everything it retained."""

    recipe: UnboundedSensitivityAlphaTargetRecipe
    targets: pa.Table
    dispersion: pa.Table
    lane_identity: UnboundedSensitivityLaneIdentity
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]


def _ordered(source_table: pa.Table) -> pa.Table:
    missing = _REQUIRED_COLUMNS - set(source_table.column_names)
    if missing:
        raise AlphaTargetBoundaryError("alpha_research.unbounded_target_source_columns_missing")
    ordered = source_table.sort_by(
        [("formation_session", "ascending"), ("listing_id", "ascending")]
    ).combine_chunks()
    keys = pa.table(
        {"formation_session": ordered["formation_session"], "listing_id": ordered["listing_id"]}
    )
    if keys.num_rows != keys.group_by(["formation_session", "listing_id"]).aggregate([]).num_rows:
        raise AlphaTargetBoundaryError("alpha_research.unbounded_target_source_duplicated")
    return ordered


def compile_unbounded_sensitivity_target_surface(
    *,
    source_table: pa.Table,
    recipe: UnboundedSensitivityAlphaTargetRecipe,
    sector_by_listing_id: Mapping[str, str],
    standardizations: AlphaTargetCatalog | None = None,
) -> UnboundedSensitivityTargetSurface:
    """Compile the sensitivity control, retaining every lane it passes through."""
    recipe = UnboundedSensitivityAlphaTargetRecipe.model_validate(recipe.model_dump(mode="json"))
    catalog = standardizations or build_installed_alpha_target_catalog()
    standardization = catalog.resolve(recipe.standardization_id)

    ordered = _ordered(source_table)
    session_values = tuple(cast(list[date], ordered["formation_session"].to_pylist()))
    listing_values = tuple(str(value) for value in ordered["listing_id"].to_pylist())
    sessions = tuple(sorted(set(session_values)))
    listings = tuple(sorted(set(listing_values)))
    if ordered.num_rows != len(sessions) * len(listings):
        raise AlphaTargetBoundaryError("alpha_research.unbounded_target_surface_incomplete")
    if set(listings) - set(sector_by_listing_id):
        raise AlphaTargetBoundaryError("alpha_research.unbounded_target_sector_map_incomplete")

    source = np.asarray(
        ordered["fit_target"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64
    ).reshape(len(sessions), len(listings))
    simple = np.asarray(
        ordered["simple_economic_return"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    ).reshape(len(sessions), len(listings))
    if bool(np.any(np.isinf(source))) or bool(np.any(np.isinf(simple))):
        raise AlphaTargetBoundaryError("alpha_research.unbounded_target_source_nonfinite")

    # Each run of formations reads the Sector map in force at it (V346): one run while no
    # reclassification falls inside the window, the one map every formation read before.
    runs = sector_positions(sector_by_listing_id, sessions, listings)

    # --- the composition, bounding step absent by construction -------------------------
    raw_residual = demean_by_run(source, runs)
    sector_center = source - raw_residual
    redemeaned_residual = demean_by_run(raw_residual, runs)
    correction = np.abs(redemeaned_residual - raw_residual)
    maximum_correction = float(np.nanmax(correction)) if correction.size else 0.0
    if (
        np.isfinite(maximum_correction)
        and maximum_correction > recipe.redemean_invariance_tolerance
    ):
        # Not a numerical nuisance. If demeaning an already-demeaned residual
        # moves it, then one of the two reductions is not the operation this
        # method claims to run, and the surface would be evidence about
        # something other than the absence of bounding.
        raise AlphaTargetBoundaryError("alpha_research.unbounded_target_redemean_not_idle")
    lane = standardization.standardize(redemeaned_residual, mad_scale=1.4826)
    fit_values = lane.values
    dispersion = lane.dispersion

    # --- availability, fail-closed and typed, matching the canonical arm ---------------
    finite = np.isfinite(source)
    finite_counts = finite.sum(axis=1)
    coverage = finite_counts / len(listings)
    eligible_counts = np.isfinite(redemeaned_residual).sum(axis=1)
    reason_masks: tuple[tuple[UnboundedTargetUnavailableReason, BoolArray], ...] = (
        ("SOURCE_TARGET_MISSING", cast(BoolArray, finite_counts == 0)),
        ("COVERAGE_BELOW_MINIMUM", cast(BoolArray, coverage < recipe.minimum_coverage)),
        (
            "SECTOR_SAMPLE_BELOW_MINIMUM",
            below_sector_sample(finite, runs, recipe.minimum_sector_sample),
        ),
        (
            "ELIGIBLE_ROWS_BELOW_MINIMUM",
            cast(BoolArray, eligible_counts < recipe.minimum_eligible_rows),
        ),
        ("RESIDUAL_DISPERSION_NONFINITE", cast(BoolArray, ~np.isfinite(dispersion))),
        ("RESIDUAL_DISPERSION_ZERO", cast(BoolArray, dispersion == 0.0)),
    )
    assigned: BoolArray = np.zeros(len(sessions), dtype=np.bool_)
    reasons: list[str | None] = [None] * len(sessions)
    for reason, raw_mask in reason_masks:
        mask = np.asarray(raw_mask & ~assigned, dtype=np.bool_)
        for index in np.flatnonzero(mask):
            reasons[int(index)] = reason
        assigned |= mask

    available = ~assigned
    admitted = np.where(available[:, None] & finite & np.isfinite(fit_values), fit_values, np.nan)
    valid = np.isfinite(admitted)
    admitted_dispersion = np.where(available, dispersion, np.nan)

    targets = pa.table(
        {
            "formation_session": ordered["formation_session"],
            "listing_id": ordered["listing_id"],
            "fit_target": pa.array(
                admitted.reshape(-1), mask=~valid.reshape(-1), type=pa.float64()
            ),
            "raw_log_execution_return": ordered["fit_target"],
            "simple_economic_return": ordered["simple_economic_return"],
            "sector_center": pa.array(sector_center.reshape(-1), type=pa.float64()),
            "raw_residual": pa.array(raw_residual.reshape(-1), type=pa.float64()),
            "redemeaned_residual": pa.array(redemeaned_residual.reshape(-1), type=pa.float64()),
        }
    ).combine_chunks()
    dispersion_table = pa.table(
        {
            "formation_session": pa.array(list(sessions), type=pa.date32()),
            "cross_sectional_dispersion": pa.array(
                admitted_dispersion, mask=~available, type=pa.float64()
            ),
            "eligible_listing_count": pa.array(
                [int(value) for value in eligible_counts], type=pa.int64()
            ),
            "available": pa.array([bool(value) for value in available], type=pa.bool_()),
            "unavailable_reason": pa.array(reasons, type=pa.string()),
        }
    ).combine_chunks()

    lane_values: dict[str, object] = {
        "kind": "UnboundedSensitivityLaneIdentity",
        "raw_log_execution_return_identity": _lane_identity(source),
        "simple_economic_return_identity": _lane_identity(simple),
        "sector_center_identity": _lane_identity(sector_center),
        "raw_residual_identity": _lane_identity(raw_residual),
        "redemeaned_residual_identity": _lane_identity(redemeaned_residual),
        "cross_sectional_dispersion_identity": _lane_identity(admitted_dispersion),
        "fit_target_identity": _lane_identity(cast(FloatArray, admitted)),
        "maximum_redemean_correction": maximum_correction,
        "ordered_sessions_hash": str(canonical_hash([value.isoformat() for value in sessions])),
        "ordered_listing_ids_hash": str(canonical_hash(list(listings))),
    }
    return UnboundedSensitivityTargetSurface(
        recipe=recipe,
        targets=targets,
        dispersion=dispersion_table,
        lane_identity=UnboundedSensitivityLaneIdentity(
            **lane_values, lane_identity_hash=canonical_hash(lane_values)
        ),
        formation_sessions=sessions,
        ordered_listing_ids=listings,
    )


__all__ = [
    "REDEMEAN_INVARIANCE_TOLERANCE",
    "UNBOUNDED_SENSITIVITY_TARGET_RECIPE_ID",
    "UnboundedSensitivityAlphaTargetRecipe",
    "UnboundedSensitivityLaneIdentity",
    "UnboundedSensitivityTargetSurface",
    "build_unbounded_sensitivity_target_recipe",
    "compile_unbounded_sensitivity_target_surface",
]
