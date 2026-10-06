"""Alpha fit-target lanes derived from the verified one-session execution outcome."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from enum import StrEnum
from typing import Literal, Self, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.quant.cross_section import (
    MAD_SCALE,
    MIN_COVERAGE,
    MIN_SECTOR_SAMPLE,
    WINSOR_MULTIPLIER,
    median_mad_winsor,
)
from alphalattice.kernel.quant.sector_history import sector_positions
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sector_treatment import (
    SECTOR_HISTORY_BACKFILLED,
    SectorHistoryTreatment,
)

from .catalog import AlphaTargetCatalog, build_installed_alpha_target_catalog
from .sector_runs import below_sector_sample, demean_by_run
from .standardization import RANK_GAUSS_STANDARDIZATION_ID, ROBUST_Z_STANDARDIZATION_ID

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]

_REQUIRED_COLUMNS = frozenset(
    {
        "formation_session",
        "listing_id",
        "fit_target",
        "simple_economic_return",
    }
)


class AlphaTargetBoundaryError(ValueError):
    """Stable failure raised before an Alpha target lane is admitted."""


class AlphaTargetLane(StrEnum):
    """Declare installed execution-return target lanes and their standardization routes."""

    SECTOR_RESIDUAL_RANK_GAUSS = "SECTOR_RESIDUAL_RANK_GAUSS"
    SECTOR_RESIDUAL_ROBUST_Z = "SECTOR_RESIDUAL_ROBUST_Z"
    LOG_RETURN_RANK_GAUSS = "LOG_RETURN_RANK_GAUSS"
    LOG_RETURN_ROBUST_Z = "LOG_RETURN_ROBUST_Z"


CURRENT_ALPHA_TARGET_LANES: tuple[AlphaTargetLane, AlphaTargetLane] = (
    AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS,
    AlphaTargetLane.SECTOR_RESIDUAL_ROBUST_Z,
)


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlphaTargetPolicy(_Contract):
    """Bind one-session execution returns to explicit lane, sector and outlier policy.

    Log execution returns form the target source; simple execution returns retain economic
    semantics. Current sector classification is backfilled and explicitly not point-in-time
    qualified. The lane decides whether equal-sector demeaning is used.
    """

    kind: Literal["AlphaTargetPolicy"] = "AlphaTargetPolicy"
    lane: AlphaTargetLane
    prediction_horizon_sessions: Literal[1] = 1
    source_semantics: Literal["LOG_EXECUTION_RETURN"] = "LOG_EXECUTION_RETURN"
    economic_return_semantics: Literal["SIMPLE_EXECUTION_RETURN"] = "SIMPLE_EXECUTION_RETURN"
    sequence: tuple[str, ...] = (
        "median_mad_winsor",
        "equal_sector_demean",
        "lane_standardization",
    )
    neutralization: Literal["EQUAL_SECTOR_DEMEAN", "NONE"] = "EQUAL_SECTOR_DEMEAN"
    sector_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    sector_source: Literal["YAHOO_CURRENT_SECTOR"] = "YAHOO_CURRENT_SECTOR"
    sector_history_treatment: SectorHistoryTreatment = "CURRENT_CLASSIFICATION_BACKFILLED"
    sector_point_in_time_qualified: Literal[False] = False
    winsor_multiplier: float = WINSOR_MULTIPLIER
    mad_scale: float = MAD_SCALE
    minimum_coverage: float = MIN_COVERAGE
    minimum_sector_sample: int = MIN_SECTOR_SAMPLE
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def standardization_id(self) -> str:
        """Declared routing key for the installed lane standardization.

        A property, not a field: the frozen `policy_hash` of every published
        target artifact stays exactly as it was.
        """
        return (
            RANK_GAUSS_STANDARDIZATION_ID
            if self.lane
            in {
                AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS,
                AlphaTargetLane.LOG_RETURN_RANK_GAUSS,
            }
            else ROBUST_Z_STANDARDIZATION_ID
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_policy(self) -> Self:
        """Require lane-specific neutralization/sequence and admitted policy identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The lane-specific sequence/neutralization differs or neither current nor
                compatible historical serialization matches the policy hash.
        """
        expected_neutralization = (
            "EQUAL_SECTOR_DEMEAN"
            if self.lane
            in {
                AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS,
                AlphaTargetLane.SECTOR_RESIDUAL_ROBUST_Z,
            }
            else "NONE"
        )
        expected_sequence = (
            ("median_mad_winsor", "equal_sector_demean", "lane_standardization")
            if expected_neutralization == "EQUAL_SECTOR_DEMEAN"
            else ("median_mad_winsor", "lane_standardization")
        )
        current = canonical_hash(self.model_dump(mode="json", exclude={"policy_hash"}))
        historical = canonical_hash(
            self.model_dump(mode="json", exclude={"policy_hash", "neutralization"})
        )
        if (
            self.neutralization != expected_neutralization
            or self.sequence != expected_sequence
            or self.policy_hash not in {current, historical}
        ):
            raise ValueError("Alpha target policy hash is invalid")
        return self


def _seal[ContractT: _Contract](
    contract: type[ContractT], field: str, **values: object
) -> ContractT:
    provisional = contract.model_construct(**values, **{field: "0" * 64})
    identity = provisional.model_dump(mode="json", exclude={field})
    return contract(**values, **{field: canonical_hash(identity)})


def build_alpha_target_policy(
    *,
    lane: AlphaTargetLane,
    sector_revision: str,
    sector_history_treatment: SectorHistoryTreatment = SECTOR_HISTORY_BACKFILLED,
) -> AlphaTargetPolicy:
    """Build the installed lane policy with explicit current-sector revision.

    Sector-residual lanes retain compatible historical hashing without the neutralization field.
    Log-return lanes use no neutralization and hash the current declared sequence. The Sector
    treatment is what its sessions read (V346); the backfill keeps every policy sealed before
    the forward rule as it was.

    Args:
        lane: Installed target lane.
        sector_revision: Explicit current-sector classification revision.
        sector_history_treatment: What the sessions read of the classification.

    Returns:
        Validated installed lane policy with its admitted identity.
    """
    if lane in {
        AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS,
        AlphaTargetLane.SECTOR_RESIDUAL_ROBUST_Z,
    }:
        values: dict[str, object] = {
            "lane": lane,
            "sector_revision": sector_revision,
            "sector_history_treatment": sector_history_treatment,
        }
        draft = AlphaTargetPolicy.model_construct(**values, policy_hash="0" * 64)
        legacy_identity = draft.model_dump(mode="json", exclude={"policy_hash", "neutralization"})
        return cast(
            AlphaTargetPolicy,
            AlphaTargetPolicy.model_validate(
                {**values, "policy_hash": canonical_hash(legacy_identity)}
            ),
        )
    return _seal(
        AlphaTargetPolicy,
        "policy_hash",
        lane=lane,
        sector_revision=sector_revision,
        sector_history_treatment=sector_history_treatment,
        neutralization="NONE",
        sequence=("median_mad_winsor", "lane_standardization"),
    )


def _ordered(source: pa.Table) -> pa.Table:
    missing = sorted(_REQUIRED_COLUMNS - set(source.schema.names))
    if missing:
        raise AlphaTargetBoundaryError(
            f"alpha_research.target_required_column_missing:{','.join(missing)}"
        )
    if source.num_rows == 0:
        raise AlphaTargetBoundaryError("alpha_research.target_surface_empty")
    ordered = source.take(
        pc.sort_indices(
            source,
            sort_keys=[("formation_session", "ascending"), ("listing_id", "ascending")],
        )
    ).combine_chunks()
    if ordered.num_rows > 1:
        same_session = pc.equal(
            ordered["formation_session"].slice(1),
            ordered["formation_session"].slice(0, ordered.num_rows - 1),
        )
        same_listing = pc.equal(
            ordered["listing_id"].slice(1),
            ordered["listing_id"].slice(0, ordered.num_rows - 1),
        )
        if bool(pc.any(pc.and_(same_session, same_listing)).as_py()):
            raise AlphaTargetBoundaryError("alpha_research.target_duplicate_row")
    return ordered


def compile_alpha_target_surface(
    *,
    source_table: pa.Table,
    policy: AlphaTargetPolicy,
    sector_by_listing_id: Mapping[str, str],
    standardizations: AlphaTargetCatalog | None = None,
    standardization_id: str | None = None,
) -> pa.Table:
    """Compile one immutable fit lane while retaining raw economic returns.

    ``standardization_id`` overrides the routing key the policy derives from its
    lane. It exists because that derivation reads a *closed four-lane
    enumeration*: a standardization outside it cannot be named by any policy, so
    the only way a third method could reach this function was to relabel itself
    as one of the two installed ones -- which is not an extension, it is a
    disguise, and it makes the resulting evidence describe the wrong method.

    Defaulted to the policy's own derived key, so every frozen lane routes
    exactly as before and no published target artifact's ``policy_hash`` moves.
    """
    policy = AlphaTargetPolicy.model_validate(policy)
    catalog = standardizations or build_installed_alpha_target_catalog()
    standardization = catalog.resolve(standardization_id or policy.standardization_id)
    ordered = _ordered(source_table)
    session_values = tuple(cast(list[date], ordered["formation_session"].to_pylist()))
    listing_values = tuple(str(value) for value in ordered["listing_id"].to_pylist())
    sessions = tuple(sorted(set(session_values)))
    listings = tuple(sorted(set(listing_values)))
    if ordered.num_rows != len(sessions) * len(listings):
        raise AlphaTargetBoundaryError("alpha_research.target_surface_incomplete")
    if set(listings) - set(sector_by_listing_id):
        raise AlphaTargetBoundaryError("alpha_research.target_sector_map_incomplete")

    source = np.asarray(
        ordered["fit_target"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    ).reshape(len(sessions), len(listings))
    simple = np.asarray(
        ordered["simple_economic_return"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    ).reshape(len(sessions), len(listings))
    if bool(np.any(np.isinf(source))) or bool(np.any(np.isinf(simple))):
        raise AlphaTargetBoundaryError("alpha_research.target_source_nonfinite")
    # Each run of formations reads the Sector map in force at it (V346): one run while no
    # reclassification falls inside the window, the one map every formation read before.
    runs = sector_positions(sector_by_listing_id, sessions, listings)
    finite = np.isfinite(source)
    finite_counts = finite.sum(axis=1)
    coverage = finite_counts / len(listings)
    winsor, _median, source_mad, _lower, _upper = median_mad_winsor(
        source,
        multiplier=policy.winsor_multiplier,
        mad_scale=policy.mad_scale,
    )
    transformed = (
        demean_by_run(np.asarray(winsor, dtype=np.float64), runs)
        if policy.neutralization == "EQUAL_SECTOR_DEMEAN"
        else winsor
    )
    lane = standardization.standardize(cast(FloatArray, transformed), mad_scale=policy.mad_scale)
    lane_values = lane.values
    lane_dispersion = lane.dispersion

    reason_masks = (
        ("SOURCE_TARGET_MISSING", finite_counts == 0),
        ("COVERAGE_BELOW_MINIMUM", coverage < policy.minimum_coverage),
        (
            "SECTOR_SAMPLE_BELOW_MINIMUM",
            (
                below_sector_sample(finite, runs, policy.minimum_sector_sample)
                if policy.neutralization == "EQUAL_SECTOR_DEMEAN"
                else np.zeros(len(sessions), dtype=np.bool_)
            ),
        ),
        ("SOURCE_DISPERSION_ZERO", ~np.isfinite(source_mad) | (source_mad == 0.0)),
        (
            "LANE_DISPERSION_ZERO",
            ~np.isfinite(lane_dispersion) | (lane_dispersion == 0.0),
        ),
    )
    assigned: BoolArray = np.zeros(len(sessions), dtype=np.bool_)
    for _reason, raw_mask in reason_masks:
        mask = np.asarray(raw_mask & ~assigned, dtype=np.bool_)
        assigned |= mask
    admitted = np.where(
        (~assigned)[:, None] & finite & np.isfinite(lane_values), lane_values, np.nan
    )
    valid = np.isfinite(admitted)
    target_table = pa.table(
        {
            "formation_session": ordered["formation_session"],
            "listing_id": ordered["listing_id"],
            "fit_target": pa.array(
                admitted.reshape(-1), mask=~valid.reshape(-1), type=pa.float64()
            ),
            "raw_log_execution_return": ordered["fit_target"],
            "simple_economic_return": ordered["simple_economic_return"],
        }
    ).combine_chunks()
    return target_table


__all__ = [
    "CURRENT_ALPHA_TARGET_LANES",
    "AlphaTargetBoundaryError",
    "AlphaTargetLane",
    "AlphaTargetPolicy",
    "build_alpha_target_policy",
    "compile_alpha_target_surface",
]
