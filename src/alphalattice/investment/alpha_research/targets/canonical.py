"""The canonical Alpha target: sector residual first, bounded, re-demeaned, standardized.

The historical lanes bound the *raw return* and then took a Sector residual once.
This module runs the composition the methodology actually specifies, which is
neither a reordering nor a restyling of that:

    raw constituent log execution returns
    -> equal-weight raw Sector center
    -> raw Sector residual
    -> global residual median-MAD bound at 5 MAD
    -> equal-Sector re-demean of the bounded residual
    -> global ordinary cross-sectional standard deviation, ddof = 1
    -> standardized target Z

Two differences carry the weight. Bounding the residual rather than the raw
return means the clip is applied to the quantity being modelled, so a stock is
not clipped for its Sector's move. And the second demeaning is mandatory rather
than tidy: the bound is nonlinear, so clipping an asymmetric tail reintroduces a
Sector mean that the first demeaning had removed, and without the repair the
"Sector-neutral" target is not Sector-neutral.

The scale that performs the final division is kept, not discarded. It is the
quantity that converts a predicted ``z`` back into return units downstream, so a
consumer that needs it must read the number this compiler produced rather than
compute something similar from a nearby lane.

This is a successor recipe type, not another value in the frozen four-lane
``AlphaTargetLane`` enum. The historical lanes keep their policies, their hashes
and their published artifacts untouched.
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

from alphalattice.kernel.quant.cross_section import (
    median_mad_winsor,
)
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

CANONICAL_ALPHA_TARGET_RECIPE_ID = "SECTOR_RESIDUAL_CROSS_SECTIONAL_STD_Z"
JOINT_PRIMARY_ALPHA_TARGET_RECIPE_ID = "UNIVERSE_BOUND_SECTOR_RESIDUAL_STD_Z"
GLOBAL_RESIDUAL_WINSOR_POLICY_ID = "GLOBAL_RESIDUAL_MEDIAN_MAD_WINSOR_5"
UNIVERSE_RETURN_WINSOR_POLICY_ID = "UNIVERSE_RETURN_NORMALIZED_MAD_WINSOR_3_5"
REDEMEAN_AFTER_BOUNDING_NEUTRALIZATION_ID = "EQUAL_SECTOR_DEMEAN_THEN_REDEMEAN_AFTER_BOUNDING"
SINGLE_SECTOR_DEMEAN_NEUTRALIZATION_ID = "EQUAL_SECTOR_DEMEAN_ONCE_AFTER_BOUNDING"

_REQUIRED_COLUMNS = frozenset(
    {"formation_session", "listing_id", "fit_target", "simple_economic_return"}
)

type CanonicalTargetUnavailableReason = Literal[
    "SOURCE_TARGET_MISSING",
    "COVERAGE_BELOW_MINIMUM",
    "SECTOR_SAMPLE_BELOW_MINIMUM",
    "ELIGIBLE_ROWS_BELOW_MINIMUM",
    "RESIDUAL_DISPERSION_ZERO",
    "RESIDUAL_DISPERSION_NONFINITE",
]


class CanonicalAlphaTargetRecipe(BaseModel):  # type: ignore[misc]
    """The complete method identity: composition, constants and Sector authority.

    Every constant that can move a number is a field, because a recipe that
    names a composition but leaves its multiplier implicit cannot distinguish
    two runs that clipped at different widths.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["CanonicalAlphaTargetRecipe"] = "CanonicalAlphaTargetRecipe"
    target_recipe_id: Literal[
        "SECTOR_RESIDUAL_CROSS_SECTIONAL_STD_Z",
        "UNIVERSE_BOUND_SECTOR_RESIDUAL_STD_Z",
    ] = "SECTOR_RESIDUAL_CROSS_SECTIONAL_STD_Z"
    execution_outcome_recipe_id: str = Field(min_length=1, max_length=96)
    """Named, never assumed. The target is a transformation *of* a causal outcome,
    so the outcome method it was compiled against is part of what it is."""

    outlier_policy_id: Literal[
        "GLOBAL_RESIDUAL_MEDIAN_MAD_WINSOR_5",
        "UNIVERSE_RETURN_NORMALIZED_MAD_WINSOR_3_5",
    ] = "GLOBAL_RESIDUAL_MEDIAN_MAD_WINSOR_5"
    neutralization_id: Literal[
        "EQUAL_SECTOR_DEMEAN_THEN_REDEMEAN_AFTER_BOUNDING",
        "EQUAL_SECTOR_DEMEAN_ONCE_AFTER_BOUNDING",
    ] = "EQUAL_SECTOR_DEMEAN_THEN_REDEMEAN_AFTER_BOUNDING"
    standardization_id: Literal["CROSS_SECTIONAL_STD_Z"] = "CROSS_SECTIONAL_STD_Z"
    source_semantics: Literal["LOG_EXECUTION_RETURN"] = "LOG_EXECUTION_RETURN"
    economic_return_semantics: Literal["SIMPLE_EXECUTION_RETURN"] = "SIMPLE_EXECUTION_RETURN"
    sequence: tuple[str, ...] = Field(min_length=1)
    winsor_multiplier: float = Field(gt=0.0)
    mad_scale: float = Field(gt=0.0)
    dispersion_ddof: int = Field(ge=0, le=1)
    minimum_coverage: float = Field(gt=0.0, le=1.0)
    minimum_sector_sample: int = Field(ge=1)
    minimum_eligible_rows: int = Field(ge=2)
    """Below two rows a sample deviation does not exist; the formation is
    unavailable rather than assigned an invented scale."""

    sector_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    sector_source: str = Field(min_length=1, max_length=96)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the installed canonical sequence, policy constants and recipe identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaTargetBoundaryError: Sequence, recipe-specific outlier/neutralization/winsor
                policy, dispersion ddof or recipe hash differs.
        """
        expected_sequence = (
            (
                "equal_sector_center",
                "raw_sector_residual",
                "global_residual_median_mad_winsor",
                "equal_sector_redemean",
                "cross_sectional_std_z",
            )
            if self.target_recipe_id == CANONICAL_ALPHA_TARGET_RECIPE_ID
            else (
                "universe_return_normalized_mad_winsor",
                "equal_sector_demean",
                "cross_sectional_std_z",
            )
        )
        if self.sequence != expected_sequence:
            raise AlphaTargetBoundaryError("alpha_research.canonical_target_sequence_invalid")
        if self.target_recipe_id == CANONICAL_ALPHA_TARGET_RECIPE_ID:
            if (
                self.outlier_policy_id != GLOBAL_RESIDUAL_WINSOR_POLICY_ID
                or self.neutralization_id != REDEMEAN_AFTER_BOUNDING_NEUTRALIZATION_ID
                or self.winsor_multiplier != 5.0
            ):
                raise AlphaTargetBoundaryError("alpha_research.canonical_target_policy_invalid")
        elif (
            self.outlier_policy_id != UNIVERSE_RETURN_WINSOR_POLICY_ID
            or self.neutralization_id != SINGLE_SECTOR_DEMEAN_NEUTRALIZATION_ID
            or self.winsor_multiplier != 3.5
        ):
            raise AlphaTargetBoundaryError("alpha_research.joint_primary_target_policy_invalid")
        if self.dispersion_ddof != CROSS_SECTIONAL_STD_Z_DDOF:
            raise AlphaTargetBoundaryError("alpha_research.canonical_target_ddof_invalid")
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise AlphaTargetBoundaryError(
                "alpha_research.canonical_target_recipe_identity_invalid"
            )
        return self


def build_canonical_alpha_target_recipe(
    *,
    execution_outcome_recipe_id: str,
    sector_revision: str,
    sector_source: str = "YAHOO_CURRENT_SECTOR",
    winsor_multiplier: float = 5.0,
    mad_scale: float = 1.4826,
    minimum_coverage: float = 0.98,
    minimum_sector_sample: int = 5,
) -> CanonicalAlphaTargetRecipe:
    """Seal the installed canonical recipe against one outcome method and Sector revision."""
    values: dict[str, object] = {
        "kind": "CanonicalAlphaTargetRecipe",
        "target_recipe_id": CANONICAL_ALPHA_TARGET_RECIPE_ID,
        "execution_outcome_recipe_id": execution_outcome_recipe_id,
        "outlier_policy_id": GLOBAL_RESIDUAL_WINSOR_POLICY_ID,
        "neutralization_id": REDEMEAN_AFTER_BOUNDING_NEUTRALIZATION_ID,
        "standardization_id": CROSS_SECTIONAL_STD_Z_STANDARDIZATION_ID,
        "source_semantics": "LOG_EXECUTION_RETURN",
        "economic_return_semantics": "SIMPLE_EXECUTION_RETURN",
        "sequence": [
            "equal_sector_center",
            "raw_sector_residual",
            "global_residual_median_mad_winsor",
            "equal_sector_redemean",
            "cross_sectional_std_z",
        ],
        "winsor_multiplier": winsor_multiplier,
        "mad_scale": mad_scale,
        "dispersion_ddof": CROSS_SECTIONAL_STD_Z_DDOF,
        "minimum_coverage": minimum_coverage,
        "minimum_sector_sample": minimum_sector_sample,
        "minimum_eligible_rows": 2,
        "sector_revision": sector_revision,
        "sector_source": sector_source,
    }
    return CanonicalAlphaTargetRecipe(**values, recipe_hash=canonical_hash(values))


def build_joint_primary_alpha_target_recipe(
    *,
    execution_outcome_recipe_id: str,
    sector_revision: str,
    sector_source: str = "YAHOO_CURRENT_SECTOR",
    minimum_coverage: float = 0.98,
    minimum_sector_sample: int = 5,
) -> CanonicalAlphaTargetRecipe:
    """Seal the one-demean Joint Primary target without relabelling the Control."""
    values: dict[str, object] = {
        "kind": "CanonicalAlphaTargetRecipe",
        "target_recipe_id": JOINT_PRIMARY_ALPHA_TARGET_RECIPE_ID,
        "execution_outcome_recipe_id": execution_outcome_recipe_id,
        "outlier_policy_id": UNIVERSE_RETURN_WINSOR_POLICY_ID,
        "neutralization_id": SINGLE_SECTOR_DEMEAN_NEUTRALIZATION_ID,
        "standardization_id": CROSS_SECTIONAL_STD_Z_STANDARDIZATION_ID,
        "source_semantics": "LOG_EXECUTION_RETURN",
        "economic_return_semantics": "SIMPLE_EXECUTION_RETURN",
        "sequence": [
            "universe_return_normalized_mad_winsor",
            "equal_sector_demean",
            "cross_sectional_std_z",
        ],
        "winsor_multiplier": 3.5,
        "mad_scale": 1.4826,
        "dispersion_ddof": CROSS_SECTIONAL_STD_Z_DDOF,
        "minimum_coverage": minimum_coverage,
        "minimum_sector_sample": minimum_sector_sample,
        "minimum_eligible_rows": 2,
        "sector_revision": sector_revision,
        "sector_source": sector_source,
    }
    return CanonicalAlphaTargetRecipe(**values, recipe_hash=canonical_hash(values))


def _lane_identity(values: FloatArray) -> str:
    """dtype, shape and content as one inseparable claim about an array.

    Bytes alone are ambiguous -- the same buffer read at a different dtype or
    reshaped is a different array with the same digest -- so all three travel
    together, matching the convention the development executor already uses.
    """

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


class CanonicalAlphaTargetLaneIdentity(BaseModel):  # type: ignore[misc]
    """Each retained lane, identified separately.

    Separately rather than as one digest over the whole surface, because the
    lanes answer different questions: a consumer proving it read the raw
    economic return must be able to say so without also claiming anything about
    the fit lane, and a scale consumer binds the dispersion alone.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["CanonicalAlphaTargetLaneIdentity"] = "CanonicalAlphaTargetLaneIdentity"
    raw_log_execution_return_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    simple_economic_return_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    sector_center_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_residual_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    bounded_residual_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    redemeaned_residual_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    cross_sectional_dispersion_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_target_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    bounded_raw_return_identity: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sector_residual_identity: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ordered_sessions_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    lane_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require canonical target lane identity with absent optional fields excluded.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaTargetBoundaryError: The canonical lane payload differs from its declared identity.
        """
        if self.lane_identity_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"lane_identity_hash"}, exclude_none=True)
        ):
            raise AlphaTargetBoundaryError("alpha_research.canonical_target_lane_identity_invalid")
        return self


@dataclass(frozen=True, slots=True)
class CanonicalAlphaTargetSurface:
    """One compiled canonical target surface and everything it retained.

    ``dispersion`` is a separate per-session table rather than a column on the
    long target table: it has one value per formation, not one per row, and
    broadcasting it across listings would invite a consumer to treat a common
    scale as a per-stock quantity.
    """

    recipe: CanonicalAlphaTargetRecipe
    targets: pa.Table
    dispersion: pa.Table
    lane_identity: CanonicalAlphaTargetLaneIdentity
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    residual_bounds: pa.Table
    """The interval the bound actually enforced, one row per formation.

    Retained rather than discarded because a consumer describing what the bound
    did needs the bound. Reconstructing it afterwards from the values that
    changed can only recover an edge the data happened to reach: a formation
    clipped on one side alone leaves the other edge unobserved, and any guess at
    it would be a statement about the method that the method never made.
    """


def _ordered(source_table: pa.Table) -> pa.Table:
    missing = _REQUIRED_COLUMNS - set(source_table.column_names)
    if missing:
        raise AlphaTargetBoundaryError("alpha_research.canonical_target_source_columns_missing")
    ordered = source_table.sort_by(
        [("formation_session", "ascending"), ("listing_id", "ascending")]
    ).combine_chunks()
    keys = pa.table(
        {"formation_session": ordered["formation_session"], "listing_id": ordered["listing_id"]}
    )
    if keys.num_rows != keys.group_by(["formation_session", "listing_id"]).aggregate([]).num_rows:
        raise AlphaTargetBoundaryError("alpha_research.canonical_target_source_duplicated")
    return ordered


def compile_canonical_alpha_target_surface(
    *,
    source_table: pa.Table,
    recipe: CanonicalAlphaTargetRecipe,
    sector_by_listing_id: Mapping[str, str],
    standardizations: AlphaTargetCatalog | None = None,
) -> CanonicalAlphaTargetSurface:
    """Compile the canonical target, retaining every lane the composition passes through.

    The standardization is resolved from the installed catalog rather than
    called directly, so a recipe naming a method this build does not install
    fails before any arithmetic runs instead of silently falling back.
    """
    recipe = CanonicalAlphaTargetRecipe.model_validate(recipe.model_dump(mode="json"))
    catalog = standardizations or build_installed_alpha_target_catalog()
    standardization = catalog.resolve(recipe.standardization_id)

    ordered = _ordered(source_table)
    session_values = tuple(cast(list[date], ordered["formation_session"].to_pylist()))
    listing_values = tuple(str(value) for value in ordered["listing_id"].to_pylist())
    sessions = tuple(sorted(set(session_values)))
    listings = tuple(sorted(set(listing_values)))
    if ordered.num_rows != len(sessions) * len(listings):
        raise AlphaTargetBoundaryError("alpha_research.canonical_target_surface_incomplete")
    if set(listings) - set(sector_by_listing_id):
        raise AlphaTargetBoundaryError("alpha_research.canonical_target_sector_map_incomplete")

    source = np.asarray(
        ordered["fit_target"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64
    ).reshape(len(sessions), len(listings))
    simple = np.asarray(
        ordered["simple_economic_return"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    ).reshape(len(sessions), len(listings))
    if bool(np.any(np.isinf(source))) or bool(np.any(np.isinf(simple))):
        raise AlphaTargetBoundaryError("alpha_research.canonical_target_source_nonfinite")

    # Each run of formations reads the Sector map in force at it (V346): one run while no
    # reclassification falls inside the window, the one map every formation read before.
    runs = sector_positions(sector_by_listing_id, sessions, listings)

    # --- the composition, in the one order the installed method permits ----------------
    bounded_raw_return: FloatArray | None = None
    if recipe.target_recipe_id == CANONICAL_ALPHA_TARGET_RECIPE_ID:
        raw_residual = demean_by_run(source, runs)
        sector_center = source - raw_residual
        bounded_residual, _median, residual_mad, lower_bound, upper_bound = median_mad_winsor(
            raw_residual, multiplier=recipe.winsor_multiplier, mad_scale=recipe.mad_scale
        )
        redemeaned_residual = demean_by_run(np.asarray(bounded_residual, dtype=np.float64), runs)
    else:
        bounded_raw_return, _median, residual_mad, lower_bound, upper_bound = median_mad_winsor(
            source, multiplier=recipe.winsor_multiplier, mad_scale=recipe.mad_scale
        )
        redemeaned_residual = demean_by_run(np.asarray(bounded_raw_return, dtype=np.float64), runs)
        raw_residual = redemeaned_residual
        bounded_residual = bounded_raw_return
        sector_center = bounded_raw_return - redemeaned_residual
    lane = standardization.standardize(redemeaned_residual, mad_scale=recipe.mad_scale)
    fit_values = lane.values
    dispersion = lane.dispersion

    # --- availability, fail-closed and typed ------------------------------------------
    finite = np.isfinite(source)
    finite_counts = finite.sum(axis=1)
    coverage = finite_counts / len(listings)
    eligible_counts = np.isfinite(redemeaned_residual).sum(axis=1)
    reason_masks: tuple[tuple[CanonicalTargetUnavailableReason, BoolArray], ...] = (
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
    # An unavailable formation carries no scale rather than a zero: a consumer
    # multiplying by zero would silently mute a signal it should have refused.
    admitted_dispersion = np.where(available, dispersion, np.nan)

    target_columns: dict[str, pa.Array | pa.ChunkedArray] = {
        "formation_session": ordered["formation_session"],
        "listing_id": ordered["listing_id"],
        "fit_target": pa.array(admitted.reshape(-1), mask=~valid.reshape(-1), type=pa.float64()),
        "raw_log_execution_return": ordered["fit_target"],
        "simple_economic_return": ordered["simple_economic_return"],
        "sector_center": pa.array(sector_center.reshape(-1), type=pa.float64()),
        "raw_residual": pa.array(raw_residual.reshape(-1), type=pa.float64()),
        "bounded_residual": pa.array(
            np.asarray(bounded_residual, dtype=np.float64).reshape(-1), type=pa.float64()
        ),
        "redemeaned_residual": pa.array(redemeaned_residual.reshape(-1), type=pa.float64()),
    }
    if bounded_raw_return is not None:
        target_columns["bounded_raw_log_execution_return"] = pa.array(
            bounded_raw_return.reshape(-1), type=pa.float64()
        )
        target_columns["sector_residual"] = pa.array(
            redemeaned_residual.reshape(-1), type=pa.float64()
        )
    targets = pa.table(target_columns).combine_chunks()
    dispersion_table = pa.table(
        {
            "formation_session": pa.array(list(sessions), type=pa.date32()),
            # Null, not NaN. An unavailable formation must read as *absent* in
            # every representation of this surface -- the in-memory table and the
            # durable evidence alike -- or a consumer that checks one convention
            # will silently carry a missing scale from the other.
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
        "kind": "CanonicalAlphaTargetLaneIdentity",
        "raw_log_execution_return_identity": _lane_identity(source),
        "simple_economic_return_identity": _lane_identity(simple),
        "sector_center_identity": _lane_identity(sector_center),
        "raw_residual_identity": _lane_identity(raw_residual),
        "bounded_residual_identity": _lane_identity(np.asarray(bounded_residual, dtype=np.float64)),
        "redemeaned_residual_identity": _lane_identity(redemeaned_residual),
        "cross_sectional_dispersion_identity": _lane_identity(admitted_dispersion),
        "fit_target_identity": _lane_identity(cast(FloatArray, admitted)),
        "ordered_sessions_hash": str(canonical_hash([value.isoformat() for value in sessions])),
        "ordered_listing_ids_hash": str(canonical_hash(list(listings))),
    }
    if bounded_raw_return is not None:
        lane_values["bounded_raw_return_identity"] = _lane_identity(bounded_raw_return)
        lane_values["sector_residual_identity"] = _lane_identity(redemeaned_residual)
    del residual_mad  # the residual MAD is an intermediate of the bound, not a retained lane
    bounds_table = pa.table(
        {
            "formation_session": pa.array(list(sessions), type=pa.date32()),
            "lower_bound": pa.array(np.asarray(lower_bound, dtype=np.float64), type=pa.float64()),
            "upper_bound": pa.array(np.asarray(upper_bound, dtype=np.float64), type=pa.float64()),
        }
    ).combine_chunks()
    return CanonicalAlphaTargetSurface(
        recipe=recipe,
        targets=targets,
        dispersion=dispersion_table,
        residual_bounds=bounds_table,
        lane_identity=CanonicalAlphaTargetLaneIdentity(
            **lane_values, lane_identity_hash=canonical_hash(lane_values)
        ),
        formation_sessions=sessions,
        ordered_listing_ids=listings,
    )


class CanonicalAlphaTargetRecipeBinding(BaseModel):  # type: ignore[misc]
    """The canonical recipe, bound to the causal outcome it was compiled against.

    A recipe on its own says how to transform; it does not say *what* was
    transformed. Two surfaces compiled by the same recipe against different
    outcome snapshots are different evidence, so the outcome method seal travels
    with the recipe rather than beside it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["CanonicalAlphaTargetRecipeBinding"] = "CanonicalAlphaTargetRecipeBinding"
    target_recipe_id: str = Field(min_length=1, max_length=96)
    target_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    standardization_id: str = Field(min_length=1, max_length=128)
    target_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    causal_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_method_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    maturity_lag_sessions: int = Field(ge=2)
    """Carried from the outcome method, never chosen here. The scale consumer
    reads its causal lag from this rather than branching on a horizon."""

    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        recipe: CanonicalAlphaTargetRecipe,
        target_catalog_hash: str,
        causal_outcome_snapshot_hash: str,
        outcome_method_binding_hash: str,
        maturity_lag_sessions: int,
    ) -> CanonicalAlphaTargetRecipeBinding:
        """Seal canonical target recipe, catalog and matured outcome-source authority.

        Args:
            recipe: Declared canonical target recipe.
            target_catalog_hash: Exact target standardization catalog.
            causal_outcome_snapshot_hash: Exact causal source outcome snapshot.
            outcome_method_binding_hash: Exact outcome method binding.
            maturity_lag_sessions: Declared label maturity lag converted to an integer.

        Returns:
            Validated canonical binding and its derived content hash.
        """
        values: dict[str, object] = {
            "kind": "CanonicalAlphaTargetRecipeBinding",
            "target_recipe_id": recipe.target_recipe_id,
            "target_recipe_hash": recipe.recipe_hash,
            "standardization_id": recipe.standardization_id,
            "target_catalog_hash": target_catalog_hash,
            "causal_outcome_snapshot_hash": causal_outcome_snapshot_hash,
            "outcome_method_binding_hash": outcome_method_binding_hash,
            "maturity_lag_sessions": int(maturity_lag_sessions),
        }
        return cls(**values, binding_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact canonical target recipe/source binding identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaTargetBoundaryError: The canonical binding payload differs from its hash.
        """
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise AlphaTargetBoundaryError(
                "alpha_research.canonical_target_binding_identity_invalid"
            )
        return self


class CanonicalAlphaTargetEvidence(BaseModel):  # type: ignore[misc]
    """The durable record of one compiled canonical surface.

    Carries the realized cross-sectional scale per formation, because that lane
    is consumed by another owner. Publishing it here -- rather than letting the
    scale owner rebuild a residual of its own from the raw outcome -- is what
    keeps ``sigma_XS * z`` an exact inverse instead of two similar numbers
    computed twice.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["CanonicalAlphaTargetEvidence"] = "CanonicalAlphaTargetEvidence"
    recipe_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    lane_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_sessions_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    cross_sectional_dispersion: tuple[float | None, ...] = Field(min_length=1)
    """``None`` where the formation was unavailable. Never zero-filled: a zero
    scale would mute a signal silently where an absence refuses it."""

    available: tuple[bool, ...] = Field(min_length=1)
    unavailable_reasons: tuple[str | None, ...] = Field(min_length=1)
    cross_sectional_dispersion_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    redemeaned_residual_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_log_execution_return_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_target_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require aligned ordered formation evidence and available positive dispersion.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaTargetBoundaryError: Formation/evidence axes disagree or repeat,
                availability/dispersion contradicts, or the evidence identity differs.
        """
        axes = (
            len(self.formation_sessions),
            len(self.cross_sectional_dispersion),
            len(self.available),
            len(self.unavailable_reasons),
        )
        if len(set(axes)) != 1:
            raise AlphaTargetBoundaryError("alpha_research.canonical_target_evidence_axis_mismatch")
        if tuple(sorted(set(self.formation_sessions))) != self.formation_sessions:
            raise AlphaTargetBoundaryError(
                "alpha_research.canonical_target_evidence_axis_unordered"
            )
        for scale, available in zip(self.cross_sectional_dispersion, self.available, strict=True):
            if available and (scale is None or not (scale > 0.0)):
                raise AlphaTargetBoundaryError(
                    "alpha_research.canonical_target_evidence_scale_invalid"
                )
            if not available and scale is not None:
                raise AlphaTargetBoundaryError(
                    "alpha_research.canonical_target_evidence_scale_unexpected"
                )
        if self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise AlphaTargetBoundaryError(
                "alpha_research.canonical_target_evidence_identity_invalid"
            )
        return self


def seal_canonical_alpha_target_evidence(
    *,
    surface: CanonicalAlphaTargetSurface,
    recipe_binding: CanonicalAlphaTargetRecipeBinding,
) -> CanonicalAlphaTargetEvidence:
    """Seal the compiled surface into its durable record, from the surface itself."""
    if recipe_binding.target_recipe_hash != surface.recipe.recipe_hash:
        raise AlphaTargetBoundaryError("alpha_research.canonical_target_evidence_recipe_mismatch")
    table = surface.dispersion
    available = tuple(bool(value) for value in cast(list[bool], table["available"].to_pylist()))
    raw_scales = cast(list[float | None], table["cross_sectional_dispersion"].to_pylist())
    scales = tuple(
        float(value) if flag and value is not None else None
        for value, flag in zip(raw_scales, available, strict=True)
    )
    identity = surface.lane_identity
    values: dict[str, object] = {
        "kind": "CanonicalAlphaTargetEvidence",
        "recipe_binding_hash": recipe_binding.binding_hash,
        "lane_identity_hash": identity.lane_identity_hash,
        "ordered_sessions_hash": identity.ordered_sessions_hash,
        "ordered_listing_ids_hash": identity.ordered_listing_ids_hash,
        "ordered_listing_ids": list(surface.ordered_listing_ids),
        "formation_sessions": [value.isoformat() for value in surface.formation_sessions],
        "cross_sectional_dispersion": list(scales),
        "available": list(available),
        "unavailable_reasons": cast(list[str | None], table["unavailable_reason"].to_pylist()),
        "cross_sectional_dispersion_identity": identity.cross_sectional_dispersion_identity,
        "redemeaned_residual_identity": identity.redemeaned_residual_identity,
        "raw_log_execution_return_identity": identity.raw_log_execution_return_identity,
        "fit_target_identity": identity.fit_target_identity,
    }
    return CanonicalAlphaTargetEvidence(
        recipe_binding_hash=recipe_binding.binding_hash,
        lane_identity_hash=identity.lane_identity_hash,
        ordered_sessions_hash=identity.ordered_sessions_hash,
        ordered_listing_ids_hash=identity.ordered_listing_ids_hash,
        ordered_listing_ids=surface.ordered_listing_ids,
        formation_sessions=surface.formation_sessions,
        cross_sectional_dispersion=scales,
        available=available,
        unavailable_reasons=tuple(cast(list[str | None], table["unavailable_reason"].to_pylist())),
        cross_sectional_dispersion_identity=identity.cross_sectional_dispersion_identity,
        redemeaned_residual_identity=identity.redemeaned_residual_identity,
        raw_log_execution_return_identity=identity.raw_log_execution_return_identity,
        fit_target_identity=identity.fit_target_identity,
        evidence_hash=canonical_hash(values),
    )


class CanonicalAlphaScoreBinding(BaseModel):  # type: ignore[misc]
    """Alpha's claim that one score surface was fitted to one canonical target.

    Nothing about a prediction array reveals which target produced it. A model
    fitted to the historical rank-Gauss lane emits float64 of the same shape,
    same dtype and plausible magnitude as one fitted to the canonical std-Z
    target, and a consumer that decides by reading a recipe *string* is trusting
    a label rather than checking a fact.

    This binding is that fact, and it is Alpha-owned because Alpha is the only
    Desk that knows what it fitted. A Stage 1 canonical score writer publishes
    it beside the surface; until one exists, no score surface can satisfy a
    consumer that requires it, which is the intended outcome rather than a gap
    to be worked around.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["CanonicalAlphaScoreBinding"] = "CanonicalAlphaScoreBinding"
    alpha_score_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str = Field(min_length=1, max_length=128)
    target_recipe_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_sessions_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        alpha_score_surface_hash: str,
        candidate_id: str,
        target_recipe_binding_hash: str,
        target_evidence_hash: str,
        ordered_listing_ids_hash: str,
        formation_sessions_hash: str,
    ) -> CanonicalAlphaScoreBinding:
        """Seal Alpha score, candidate, target and ordered-axis lineage.

        Args:
            alpha_score_surface_hash: Exact retained score surface.
            candidate_id: Candidate producing that score.
            target_recipe_binding_hash: Exact target recipe/source binding.
            target_evidence_hash: Exact target evidence identity.
            ordered_listing_ids_hash: Declared ordered listing axis.
            formation_sessions_hash: Declared formation-session axis.

        Returns:
            Validated score binding and derived canonical identity.
        """
        values: dict[str, object] = {
            "kind": "CanonicalAlphaScoreBinding",
            "alpha_score_surface_hash": alpha_score_surface_hash,
            "candidate_id": candidate_id,
            "target_recipe_binding_hash": target_recipe_binding_hash,
            "target_evidence_hash": target_evidence_hash,
            "ordered_listing_ids_hash": ordered_listing_ids_hash,
            "formation_sessions_hash": formation_sessions_hash,
        }
        return cls(**values, binding_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact score/candidate/target lineage identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaTargetBoundaryError: The canonical score-binding payload differs from its hash.
        """
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise AlphaTargetBoundaryError(
                "alpha_research.canonical_score_binding_identity_invalid"
            )
        return self


__all__ = [
    "CANONICAL_ALPHA_TARGET_RECIPE_ID",
    "GLOBAL_RESIDUAL_WINSOR_POLICY_ID",
    "JOINT_PRIMARY_ALPHA_TARGET_RECIPE_ID",
    "REDEMEAN_AFTER_BOUNDING_NEUTRALIZATION_ID",
    "CanonicalAlphaScoreBinding",
    "CanonicalAlphaTargetEvidence",
    "CanonicalAlphaTargetLaneIdentity",
    "CanonicalAlphaTargetRecipe",
    "CanonicalAlphaTargetRecipeBinding",
    "CanonicalAlphaTargetSurface",
    "build_canonical_alpha_target_recipe",
    "build_joint_primary_alpha_target_recipe",
    "compile_canonical_alpha_target_surface",
    "seal_canonical_alpha_target_evidence",
]
