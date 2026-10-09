"""Resolve a Sector forecast selection into verified values, owned by this Desk.

A consumer persists a *selection* -- the declared ZERO control, or a Sector
experiment evidence handle -- and never a copy of the forecast array. Values
exist at runtime only, resolved here: the evidence graph is verified down to
the outcome method seal, the per-sector surface is mapped onto the consumer's
listing axis through the same membership revision the forecast was built
against, and the result is handed over read-only.

The ZERO control is resolved by this owner too, against the requested axis. A
consumer that synthesized a zero array from a method id would be computing a
Sector number outside Sector Research -- harmless for zeros, and exactly the
habit that stops being harmless with the first real method.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.kernel.quant.sector_history import sector_slices
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import FloatArray, SectorResearchError
from ..inputs.membership import load_sector_membership
from ..models.catalog import build_installed_sector_forecast_catalog
from ..models.contracts import SectorForecastCatalogBinding
from ..models.zero import ZERO_SECTOR_FORECAST_METHOD_ID
from .development_artifacts import SectorDevelopmentArtifactStore
from .verification import SectorEvidenceVerifier


class SectorForecastSelection(BaseModel):  # type: ignore[misc]
    """The durable half: which Sector forecast a consumer chose.

    Either the declared ZERO control -- a method id with no evidence, legal for
    exactly that one method -- or an experiment evidence handle whose graph the
    Sector verifier can walk. This is the only Sector object a consumer may
    bind into its own durable evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["SectorForecastSelection"] = "SectorForecastSelection"
    method_id: str = Field(min_length=1, max_length=96)
    sector_experiment_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    selection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def control_zero(cls) -> Self:
        values: dict[str, object] = {
            "kind": "SectorForecastSelection",
            "method_id": ZERO_SECTOR_FORECAST_METHOD_ID,
            "sector_experiment_hash": None,
        }
        return cls(**values, selection_hash=canonical_hash(values))

    @classmethod
    def from_experiment(cls, *, method_id: str, sector_experiment_hash: str) -> Self:
        values: dict[str, object] = {
            "kind": "SectorForecastSelection",
            "method_id": method_id,
            "sector_experiment_hash": sector_experiment_hash,
        }
        return cls(**values, selection_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.sector_experiment_hash is None and self.method_id != ZERO_SECTOR_FORECAST_METHOD_ID:
            # Only the declared control may stand without evidence. Any other
            # method id here would be a forecast nobody published.
            raise SectorResearchError("sector_research.selection_evidence_required")
        if self.selection_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"selection_hash"})
        ):
            raise SectorResearchError("sector_research.selection_identity_invalid")
        return self


@dataclass(frozen=True, slots=True)
class ResolvedSectorForecast:
    """The runtime half: verified values on the consumer's axis, plus lineage.

    Never persisted by a consumer. ``expected_return_by_listing`` is sessions
    by listings, read-only, with ``NaN`` where the underlying sector forecast
    was unavailable -- an absence a consumer must carry as an absence rather
    than mute into a zero.
    """

    selection: SectorForecastSelection
    method_id: str
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    expected_return_by_listing: FloatArray
    sector_experiment_hash: str | None
    sector_target_evidence_hash: str | None
    outcome_method_binding_hash: str | None


class SectorForecastResolver:
    """The one seam through which a consumer obtains Sector forecast values."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        outcome_reader: CausalExecutionOutcomeDevelopmentReader,
        resolver: ArtifactResolver,
        catalog_binding: SectorForecastCatalogBinding | None = None,
    ) -> None:
        self._store = SectorDevelopmentArtifactStore(artifact_root)
        self._outcome_reader = outcome_reader
        self._resolver = resolver
        self._catalog_binding = catalog_binding or build_installed_sector_forecast_catalog().binding

    def resolve(
        self,
        *,
        selection: SectorForecastSelection,
        formation_sessions: tuple[date, ...],
        ordered_listing_ids: tuple[str, ...],
    ) -> ResolvedSectorForecast:
        if (
            not formation_sessions
            or not ordered_listing_ids
            or tuple(sorted(set(formation_sessions))) != formation_sessions
            or tuple(sorted(set(ordered_listing_ids))) != ordered_listing_ids
        ):
            raise SectorResearchError("sector_research.resolution_axis_invalid")
        if selection.sector_experiment_hash is None:
            values: FloatArray = np.zeros(
                (len(formation_sessions), len(ordered_listing_ids)), dtype=np.float64
            )
            values.setflags(write=False)
            return ResolvedSectorForecast(
                selection=selection,
                method_id=selection.method_id,
                formation_sessions=formation_sessions,
                ordered_listing_ids=ordered_listing_ids,
                expected_return_by_listing=values,
                sector_experiment_hash=None,
                sector_target_evidence_hash=None,
                outcome_method_binding_hash=None,
            )

        lineage = SectorEvidenceVerifier(
            store=self._store,
            outcome_reader=self._outcome_reader,
            catalog_binding=self._catalog_binding,
        ).verify(experiment_hash=selection.sector_experiment_hash)
        surface = lineage.surface
        if surface.recipe.method_id != selection.method_id:
            raise SectorResearchError("sector_research.resolution_method_mismatch")

        surface_rows = {
            value: index for index, value in enumerate(surface.forecast_formation_sessions)
        }
        missing_sessions = [value for value in formation_sessions if value not in surface_rows]
        if missing_sessions:
            raise SectorResearchError("sector_research.resolution_session_axis_mismatch")

        membership = load_sector_membership(
            resolver=self._resolver,
            sector_revision=lineage.target_evidence.recipe.sector_revision,
        )
        uncovered = set(ordered_listing_ids) - set(membership)
        if uncovered:
            raise SectorResearchError("sector_research.resolution_membership_incomplete")
        sector_columns = {value: index for index, value in enumerate(surface.ordered_sectors)}
        # Each formation broadcasts its Sector's forecast to the listings in it then.
        columns_by_row: list[list[int]] = []
        for rows, mapping in sector_slices(membership, formation_sessions):
            columns: list[int] = []
            for listing in ordered_listing_ids:
                column = sector_columns.get(mapping[listing])
                if column is None:
                    raise SectorResearchError("sector_research.resolution_sector_axis_mismatch")
                columns.append(column)
            columns_by_row.extend(columns for _ in range(rows.stop - rows.start))

        values = np.full(
            (len(formation_sessions), len(ordered_listing_ids)), np.nan, dtype=np.float64
        )
        for row, session in enumerate(formation_sessions):
            source_row = surface.values[surface_rows[session]]
            for target_column, source_column in enumerate(columns_by_row[row]):
                cell = source_row[source_column]
                if cell is not None:
                    values[row, target_column] = float(cell)
        values.setflags(write=False)
        return ResolvedSectorForecast(
            selection=selection,
            method_id=selection.method_id,
            formation_sessions=formation_sessions,
            ordered_listing_ids=ordered_listing_ids,
            expected_return_by_listing=values,
            sector_experiment_hash=selection.sector_experiment_hash,
            sector_target_evidence_hash=lineage.target_evidence.evidence_hash,
            outcome_method_binding_hash=lineage.outcome_method_binding_hash,
        )


__all__ = [
    "ResolvedSectorForecast",
    "SectorForecastResolver",
    "SectorForecastSelection",
]
