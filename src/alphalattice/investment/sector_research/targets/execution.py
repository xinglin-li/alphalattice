"""The clean Sector target: one session's equal-weight sector mean of log returns.

``ONE_SESSION_SECTOR_MEAN_CONSTITUENT_LOG_RETURN`` is deliberately the shortest
composition that is still a target: group the constituent log execution returns
of one formation by sector and take the equal-weight mean. No winsor, no rank,
no market demean, and no reuse of the legacy excess-EMA target -- that method
models market-*excess* returns, and publishing its transformation here would
make the clean catalog inherit the very entanglement this Desk exists to end.

What carries the weight is not the arithmetic but the binding around it. The
sealed evidence names the causal outcome snapshot and its terminal method seal,
the sector membership revision with its history treatment, the exact ordered
session/sector/listing axes, and the formation, entry, exit and availability
clocks per formation. A same-shaped surface with one wrong date, a relabelled
membership, a different outcome method or a reordered axis fails at the
contract, before any model is called.

The availability clock is the causal core. A formation's label becomes
observable when its exit open has printed, which is the ``holding_end_session``
the outcome rows already carry; forecast formation happens at that session's
official close, after the open. Training consumes labels by comparing that clock
against a forecast formation date -- never by slicing a session index, because
index arithmetic silently admits a label whose outcome had not finished.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal, Self, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.quant.sector_history import sector_positions as sector_runs
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sector_treatment import (
    SECTOR_HISTORY_BACKFILLED,
    SectorHistoryTreatment,
)

from ..contracts import (
    FloatArray,
    SectorResearchError,
    grid_to_matrix,
    sector_array_identity,
)

type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]

SECTOR_CLEAN_TARGET_RECIPE_ID = "ONE_SESSION_SECTOR_MEAN_CONSTITUENT_LOG_RETURN"
EQUAL_WEIGHT_SECTOR_MEAN_AGGREGATION_ID = "EQUAL_WEIGHT_SECTOR_MEAN"

type SectorTargetUnavailableReason = Literal[
    "SOURCE_TARGET_MISSING",
    "SECTOR_SAMPLE_BELOW_MINIMUM",
]

_REQUIRED_COLUMNS = frozenset(
    {
        "formation_session",
        "listing_id",
        "fit_target",
        "simple_economic_return",
        "entry_session",
        "holding_end_session",
    }
)


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class SectorTargetRecipe(_Contract):
    """The complete clean-target method identity.

    ``sequence`` is a single step and the validator refuses anything longer: a
    recipe that quietly grew a winsor or a demean stage would still parse as "a
    sector target" everywhere it is compared by id, which is exactly the
    disguise the explicit sequence exists to make impossible.
    """

    kind: Literal["SectorTargetRecipe"] = "SectorTargetRecipe"
    target_recipe_id: Literal["ONE_SESSION_SECTOR_MEAN_CONSTITUENT_LOG_RETURN"] = (
        "ONE_SESSION_SECTOR_MEAN_CONSTITUENT_LOG_RETURN"
    )
    execution_outcome_recipe_id: str = Field(min_length=1, max_length=96)
    """Named, never assumed: the target is a transformation *of* a causal
    outcome, so the outcome method it was compiled against is part of what it is."""

    aggregation_id: Literal["EQUAL_WEIGHT_SECTOR_MEAN"] = "EQUAL_WEIGHT_SECTOR_MEAN"
    source_semantics: Literal["LOG_EXECUTION_RETURN"] = "LOG_EXECUTION_RETURN"
    economic_return_semantics: Literal["SIMPLE_EXECUTION_RETURN"] = "SIMPLE_EXECUTION_RETURN"
    sequence: tuple[str, ...] = Field(min_length=1)
    minimum_sector_sample: int = Field(ge=1)
    """Below this many finite constituents a sector-session is unavailable
    rather than a mean over a sample too small to call a sector."""

    sector_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    sector_source: str = Field(min_length=1, max_length=96)
    sector_history_treatment: SectorHistoryTreatment = "CURRENT_CLASSIFICATION_BACKFILLED"
    sector_point_in_time_qualified: Literal[False] = False
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the admitted equal-weight target sequence and exact recipe identity.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            SectorResearchError: The target sequence or recipe_hash differs from the declared target
                contract.
        """
        if self.sequence != ("equal_weight_sector_mean",):
            raise SectorResearchError("sector_research.target_sequence_invalid")
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise SectorResearchError("sector_research.target_recipe_identity_invalid")
        return self


def build_sector_target_recipe(
    *,
    execution_outcome_recipe_id: str,
    sector_revision: str,
    sector_source: str = "YAHOO_CURRENT_SECTOR",
    minimum_sector_sample: int = 5,
    sector_history_treatment: SectorHistoryTreatment = SECTOR_HISTORY_BACKFILLED,
) -> SectorTargetRecipe:
    """Seal the installed clean-target recipe against one outcome method and revision."""
    values: dict[str, object] = {
        "kind": "SectorTargetRecipe",
        "target_recipe_id": SECTOR_CLEAN_TARGET_RECIPE_ID,
        "execution_outcome_recipe_id": execution_outcome_recipe_id,
        "aggregation_id": EQUAL_WEIGHT_SECTOR_MEAN_AGGREGATION_ID,
        "source_semantics": "LOG_EXECUTION_RETURN",
        "economic_return_semantics": "SIMPLE_EXECUTION_RETURN",
        "sequence": ["equal_weight_sector_mean"],
        "minimum_sector_sample": minimum_sector_sample,
        "sector_revision": sector_revision,
        "sector_source": sector_source,
        "sector_history_treatment": sector_history_treatment,
        "sector_point_in_time_qualified": False,
    }
    return SectorTargetRecipe(**values, recipe_hash=canonical_hash(values))


@dataclass(frozen=True, slots=True)
class SectorTargetSurface:
    """One compiled clean-target surface: values, clocks and axes, session-major.

    ``sector_target`` and ``economic_diagnostic`` are sessions-by-sectors
    float64 matrices with ``NaN`` where a cell is unavailable; the per-cell
    reason travels beside them rather than being inferred from the ``NaN``.
    """

    recipe: SectorTargetRecipe
    formation_sessions: tuple[date, ...]
    entry_sessions: tuple[date, ...]
    exit_sessions: tuple[date, ...]
    target_available_sessions: tuple[date, ...]
    ordered_sectors: tuple[str, ...]
    ordered_listing_ids: tuple[str, ...]
    sector_target: FloatArray
    economic_diagnostic: FloatArray
    available: tuple[tuple[bool, ...], ...]
    unavailable_reasons: tuple[tuple[str | None, ...], ...]


def _ordered(source_table: pa.Table) -> pa.Table:
    missing = _REQUIRED_COLUMNS - set(source_table.column_names)
    if missing:
        raise SectorResearchError("sector_research.target_source_columns_missing")
    if source_table.num_rows == 0:
        raise SectorResearchError("sector_research.target_source_empty")
    ordered = source_table.sort_by(
        [("formation_session", "ascending"), ("listing_id", "ascending")]
    ).combine_chunks()
    keys = pa.table(
        {"formation_session": ordered["formation_session"], "listing_id": ordered["listing_id"]}
    )
    if keys.num_rows != keys.group_by(["formation_session", "listing_id"]).aggregate([]).num_rows:
        raise SectorResearchError("sector_research.target_source_duplicated")
    return ordered


def compile_sector_target_surface(
    *,
    source_table: pa.Table,
    recipe: SectorTargetRecipe,
    sector_by_listing_id: Mapping[str, str],
) -> SectorTargetSurface:
    """Compile the clean Sector target over one complete formation-by-listing grid.

    The schedule clocks are read from the rows and required to be unique per
    formation: two listings of one formation disagreeing about entry or exit
    would mean the source is not the single-schedule outcome seam this target
    is defined over, and averaging across that disagreement would hide it.
    """
    recipe = SectorTargetRecipe.model_validate(recipe.model_dump(mode="json"))
    ordered = _ordered(source_table)
    session_values = tuple(cast(list[date], ordered["formation_session"].to_pylist()))
    listing_values = tuple(str(value) for value in ordered["listing_id"].to_pylist())
    sessions = tuple(sorted(set(session_values)))
    listings = tuple(sorted(set(listing_values)))
    if ordered.num_rows != len(sessions) * len(listings):
        raise SectorResearchError("sector_research.target_surface_incomplete")
    if set(listings) - set(sector_by_listing_id):
        raise SectorResearchError("sector_research.target_sector_membership_incomplete")

    source = np.asarray(
        ordered["fit_target"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64
    ).reshape(len(sessions), len(listings))
    simple = np.asarray(
        ordered["simple_economic_return"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    ).reshape(len(sessions), len(listings))
    if bool(np.any(np.isinf(source))) or bool(np.any(np.isinf(simple))):
        raise SectorResearchError("sector_research.target_source_nonfinite")
    # The factor compiler masks both lanes with one validity mask; a row where
    # they disagree is not that seam and refusing is better than averaging it.
    if not bool(np.array_equal(np.isfinite(source), np.isfinite(simple))):
        raise SectorResearchError("sector_research.target_lane_mask_mismatch")

    entry_values = tuple(cast(list[date], ordered["entry_session"].to_pylist()))
    exit_values = tuple(cast(list[date], ordered["holding_end_session"].to_pylist()))
    entry_by_session: dict[date, date] = {}
    exit_by_session: dict[date, date] = {}
    for formation, entry, exit_session in zip(
        session_values, entry_values, exit_values, strict=True
    ):
        if entry_by_session.setdefault(formation, entry) != entry:
            raise SectorResearchError("sector_research.target_schedule_axis_mismatch")
        if exit_by_session.setdefault(formation, exit_session) != exit_session:
            raise SectorResearchError("sector_research.target_schedule_axis_mismatch")
    entries = tuple(entry_by_session[value] for value in sessions)
    exits = tuple(exit_by_session[value] for value in sessions)
    for formation, entry, exit_session in zip(sessions, entries, exits, strict=True):
        if not formation < entry < exit_session:
            raise SectorResearchError("sector_research.target_schedule_clock_invalid")

    # Each run of formations aggregates the Sectors in force there; the Sector axis is
    # every Sector some formation reads, one run's while no reclassification falls inside it.
    runs = sector_runs(sector_by_listing_id, sessions, listings)
    sectors = tuple(
        sorted({sector for _rows, run_sectors, _members in runs for sector in run_sectors})
    )
    empty: IntArray = np.asarray([], dtype=np.int64)
    positions_by_row: list[tuple[IntArray, ...]] = []
    for rows, run_sectors, run_positions in runs:
        held = dict(zip(run_sectors, run_positions, strict=True))
        aligned = tuple(held.get(sector, empty) for sector in sectors)
        positions_by_row.extend(aligned for _ in range(rows.stop - rows.start))

    finite = np.isfinite(source)
    sector_target: FloatArray = np.full((len(sessions), len(sectors)), np.nan, dtype=np.float64)
    economic: FloatArray = np.full((len(sessions), len(sectors)), np.nan, dtype=np.float64)
    available_rows: list[tuple[bool, ...]] = []
    reason_rows: list[tuple[str | None, ...]] = []
    for row in range(len(sessions)):
        positions = positions_by_row[row]
        row_available: list[bool] = []
        row_reasons: list[str | None] = []
        for column, sector_positions in enumerate(positions):
            member_finite = finite[row, sector_positions]
            count = int(member_finite.sum())
            if count == 0:
                row_available.append(False)
                row_reasons.append("SOURCE_TARGET_MISSING")
                continue
            if count < recipe.minimum_sector_sample:
                row_available.append(False)
                row_reasons.append("SECTOR_SAMPLE_BELOW_MINIMUM")
                continue
            selected = sector_positions[member_finite]
            sector_target[row, column] = float(np.mean(source[row, selected]))
            economic[row, column] = float(np.mean(simple[row, selected]))
            row_available.append(True)
            row_reasons.append(None)
        available_rows.append(tuple(row_available))
        reason_rows.append(tuple(row_reasons))

    return SectorTargetSurface(
        recipe=recipe,
        formation_sessions=sessions,
        entry_sessions=entries,
        exit_sessions=exits,
        # The label is observable once the exit open has printed; formation is
        # at the official close of a session, after its open, so a same-date
        # comparison stays causal.
        target_available_sessions=exits,
        ordered_sectors=sectors,
        ordered_listing_ids=listings,
        sector_target=sector_target,
        economic_diagnostic=economic,
        available=tuple(available_rows),
        unavailable_reasons=tuple(reason_rows),
    )


def _grid(values: FloatArray) -> tuple[tuple[float | None, ...], ...]:
    return tuple(
        tuple(None if not np.isfinite(value) else float(value) for value in row) for row in values
    )


class SectorTargetEvidence(_Contract):
    """The durable record of one compiled clean-target surface.

    Values travel inline rather than as a sidecar chunk: the surface is
    sessions-by-sectors, small enough that a second artifact would add a
    dangling-reference failure mode without removing any real cost. The lane
    identities are recomputed from the inline values on every parse, so a
    tampered cell fails to load rather than verifying against itself.
    """

    kind: Literal["SectorTargetEvidence"] = "SectorTargetEvidence"
    recipe: SectorTargetRecipe
    """Embedded, not referenced: the recipe is small, has no second consumer,
    and embedding it makes the evidence revalidate its method on read."""

    causal_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_method_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    maturity_lag_sessions: int = Field(ge=2)
    actual_session_span: int = Field(ge=2)
    forecast_horizon_sessions: int = Field(ge=1)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    entry_sessions: tuple[date, ...] = Field(min_length=1)
    exit_sessions: tuple[date, ...] = Field(min_length=1)
    target_available_sessions: tuple[date, ...] = Field(min_length=1)
    ordered_sectors: tuple[str, ...] = Field(min_length=1)
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_sessions_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_sectors_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sector_target_values: tuple[tuple[float | None, ...], ...] = Field(min_length=1)
    economic_diagnostic_values: tuple[tuple[float | None, ...], ...] = Field(min_length=1)
    available: tuple[tuple[bool, ...], ...] = Field(min_length=1)
    unavailable_reasons: tuple[tuple[str | None, ...], ...] = Field(min_length=1)
    sector_target_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    economic_diagnostic_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require causal target clocks, aligned cells and exact numerical/evidence identities.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            SectorResearchError: Session/sector axes, formation-entry-exit clocks, availability
                cells, array identities or evidence_hash are inconsistent.
        """
        rows = len(self.formation_sessions)
        columns = len(self.ordered_sectors)
        clocks = (self.entry_sessions, self.exit_sessions, self.target_available_sessions)
        if any(len(value) != rows for value in clocks):
            raise SectorResearchError("sector_research.target_evidence_clock_axis_mismatch")
        if tuple(sorted(set(self.formation_sessions))) != self.formation_sessions:
            raise SectorResearchError("sector_research.target_evidence_axis_unordered")
        if tuple(sorted(set(self.ordered_sectors))) != self.ordered_sectors:
            raise SectorResearchError("sector_research.target_evidence_sector_axis_unordered")
        for formation, entry, exit_session, available_at in zip(
            self.formation_sessions,
            self.entry_sessions,
            self.exit_sessions,
            self.target_available_sessions,
            strict=True,
        ):
            if not formation < entry < exit_session or available_at != exit_session:
                raise SectorResearchError("sector_research.target_evidence_clock_invalid")
        grids: tuple[tuple[tuple[object, ...], ...], ...] = (
            self.sector_target_values,
            self.economic_diagnostic_values,
            self.available,
            self.unavailable_reasons,
        )
        if any(len(grid) != rows or any(len(row) != columns for row in grid) for grid in grids):
            raise SectorResearchError("sector_research.target_evidence_grid_shape_invalid")
        for value_row, economic_row, available_row, reason_row in zip(
            self.sector_target_values,
            self.economic_diagnostic_values,
            self.available,
            self.unavailable_reasons,
            strict=True,
        ):
            for value, economic, cell_available, reason in zip(
                value_row, economic_row, available_row, reason_row, strict=True
            ):
                populated = value is not None and economic is not None and reason is None
                absent = value is None and economic is None and reason is not None
                if not (populated or absent) or cell_available != populated:
                    raise SectorResearchError("sector_research.target_evidence_cell_invalid")
        if self.sector_target_identity != sector_array_identity(
            grid_to_matrix(self.sector_target_values)
        ) or self.economic_diagnostic_identity != sector_array_identity(
            grid_to_matrix(self.economic_diagnostic_values)
        ):
            raise SectorResearchError("sector_research.target_evidence_lane_identity_invalid")
        if self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise SectorResearchError("sector_research.target_evidence_identity_invalid")
        return self


def seal_sector_target_evidence(
    *,
    surface: SectorTargetSurface,
    causal_outcome_snapshot_hash: str,
    outcome_method_binding_hash: str,
    maturity_lag_sessions: int,
    actual_session_span: int,
    forecast_horizon_sessions: int,
) -> SectorTargetEvidence:
    """Seal the compiled surface into its durable record, from the surface itself."""
    values: dict[str, object] = {
        "kind": "SectorTargetEvidence",
        "recipe": surface.recipe.model_dump(mode="json"),
        "causal_outcome_snapshot_hash": causal_outcome_snapshot_hash,
        "outcome_method_binding_hash": outcome_method_binding_hash,
        "maturity_lag_sessions": int(maturity_lag_sessions),
        "actual_session_span": int(actual_session_span),
        "forecast_horizon_sessions": int(forecast_horizon_sessions),
        "formation_sessions": [value.isoformat() for value in surface.formation_sessions],
        "entry_sessions": [value.isoformat() for value in surface.entry_sessions],
        "exit_sessions": [value.isoformat() for value in surface.exit_sessions],
        "target_available_sessions": [
            value.isoformat() for value in surface.target_available_sessions
        ],
        "ordered_sectors": list(surface.ordered_sectors),
        "ordered_listing_ids_hash": str(canonical_hash(list(surface.ordered_listing_ids))),
        "ordered_sessions_hash": str(
            canonical_hash([value.isoformat() for value in surface.formation_sessions])
        ),
        "ordered_sectors_hash": str(canonical_hash(list(surface.ordered_sectors))),
        "sector_target_values": [list(row) for row in _grid(surface.sector_target)],
        "economic_diagnostic_values": [list(row) for row in _grid(surface.economic_diagnostic)],
        "available": [list(row) for row in surface.available],
        "unavailable_reasons": [list(row) for row in surface.unavailable_reasons],
        "sector_target_identity": sector_array_identity(surface.sector_target),
        "economic_diagnostic_identity": sector_array_identity(surface.economic_diagnostic),
    }
    return cast(
        SectorTargetEvidence,
        SectorTargetEvidence.model_validate({**values, "evidence_hash": canonical_hash(values)}),
    )


__all__ = [
    "EQUAL_WEIGHT_SECTOR_MEAN_AGGREGATION_ID",
    "SECTOR_CLEAN_TARGET_RECIPE_ID",
    "SectorTargetEvidence",
    "SectorTargetRecipe",
    "SectorTargetSurface",
    "build_sector_target_recipe",
    "compile_sector_target_surface",
    "seal_sector_target_evidence",
]
