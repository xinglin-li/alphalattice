"""The minimal Sector development evidence graph and its content-addressed store.

Four categories, one graph::

    SectorTargetEvidence -> SectorForecastSurface -> SectorForecastEvaluation
                                                  -> SectorExperimentEvidence

The experiment evidence is the root and is published last, after every child it
names exists -- the marker-last rule this program applies everywhere. Small
bindings do not become categories: the target recipe embeds in its evidence and
the program binding embeds in its surface, because a category with a single
consumer is a dangling-reference failure mode purchased for nothing.

Development artifacts only. Nothing in this module can write a current pointer,
and there is no admission or activation path from here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.source_identity import (
    switched_source_identity,
)

from ..contracts import SectorResearchError, grid_to_matrix, sector_array_identity
from ..models.contracts import SectorForecastRecipe
from ..targets.execution import SectorTargetEvidence

SECTOR_TARGET_EVIDENCE_CATEGORY = "development/sector-target-evidence"
SECTOR_FORECAST_SURFACE_CATEGORY = "development/sector-forecast-surfaces"
SECTOR_FORECAST_EVALUATION_CATEGORY = "development/sector-forecast-evaluations"
SECTOR_EXPERIMENT_EVIDENCE_CATEGORY = "development/sector-experiment-evidence"
SECTOR_SHRINK_CALIBRATION_CATEGORY = "development/sector-shrink-calibrations"
SECTOR_METHOD_COMPARISON_CATEGORY = "development/sector-method-comparisons"
SECTOR_CAMPAIGN_DOSSIER_CATEGORY = "development/sector-campaign-dossiers"
SECTOR_CAMPAIGN_DECISION_CATEGORY = "development/sector-campaign-decisions"
SECTOR_CAMPAIGN_REPLAY_CATEGORY = "development/sector-campaign-replays"

_PREFIX = "playpen://sector-research/"


def sector_artifact_uri(category: str, content_hash: str) -> str:
    """The one spelling of where a Sector development artifact lives."""

    return f"{_PREFIX}{category}/{content_hash}"


def sector_development_source_closure_hash() -> str:
    """Byte identity of the Host execution path that produced a development run.

    The adapter's numerical binding covers only the method's own arithmetic.
    What it does not cover is everything the Host does around it: the clean
    target compiler, the membership resolution, the causal training selection
    and evaluation in this module, and the refit loop in the service. Any of
    those can move a number while every recorded hash stays internally valid --
    so their exact bytes enter the program identity, keyed by semantic
    component id via the shared kernel so the hash does not move when the
    repository is checked out elsewhere.

    Hashed as file bytes, deliberately not imported: the verifier compares this
    closure without ever gaining an import path to the service or a model.
    """

    here = Path(__file__)
    package = here.parents[1]
    return switched_source_identity(
        {
            "sector_research.contracts": package / "contracts.py",
            "sector_research.targets.execution": package / "targets" / "execution.py",
            "sector_research.inputs.membership": package / "inputs" / "membership.py",
            "sector_research.experiments.development_artifacts": here,
            "sector_research.experiments.service": here.with_name("service.py"),
        },
        semantic_owner="sector_research",
        numerical_role="SECTOR_DEVELOPMENT_EXECUTION",
    )


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class SectorForecastProgramBinding(_Contract):
    """What one forecast run bound, every boundary a typed field.

    Training start and end, the forecast formation range, horizon, maturity
    lag, refit cadence, minimum history and the ordered forecast sessions are
    all here rather than in code constants, so two runs that differ in any of
    them are different programs by hash and not by archaeology.
    """

    kind: Literal["SectorForecastProgramBinding"] = "SectorForecastProgramBinding"
    target_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_id: str = Field(min_length=1, max_length=96)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_start: date
    training_end: date
    forecast_start: date
    forecast_end: date
    forecast_horizon_sessions: int = Field(ge=1)
    maturity_lag_sessions: int = Field(ge=2)
    refit_every_sessions: int = Field(ge=0)
    """Sessions between adapter refits; ``0`` for a stateless method that is
    evaluated fresh at every formation."""

    minimum_history_sessions: int = Field(ge=0)
    forecast_formation_sessions: tuple[date, ...] = Field(min_length=1)
    development_source_closure_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The Host execution path's byte identity at run time. Two runs under
    different compiler, selection, evaluation or service code are different
    programs by hash, and the verifier refuses a closure that is no longer the
    installed one -- the same drift rule the per-method numerical binding
    already enforces for the adapter."""

    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.training_start > self.training_end or self.forecast_start > self.forecast_end:
            raise SectorResearchError("sector_research.program_range_invalid")
        if tuple(sorted(set(self.forecast_formation_sessions))) != (
            self.forecast_formation_sessions
        ):
            raise SectorResearchError("sector_research.program_forecast_axis_unordered")
        if any(
            not self.forecast_start <= value <= self.forecast_end
            for value in self.forecast_formation_sessions
        ):
            raise SectorResearchError("sector_research.program_forecast_axis_outside_range")
        if self.program_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"program_hash"})
        ):
            raise SectorResearchError("sector_research.program_identity_invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        draft = dict(values)
        draft.pop("program_hash", None)
        provisional = cls.model_construct(**draft, program_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"program_hash"})
        return cast(
            Self, cls.model_validate({**identity, "program_hash": canonical_hash(identity)})
        )


class SectorForecastSurface(_Contract):
    """One bounded development forecast run, produced by any installed method.

    The method is described by the embedded recipe and the program binding
    rather than by typed per-method fields, so a second capability needs no new
    surface contract. ``training_row_counts`` records how many matured training
    rows the bound input carried at each formation's refit -- the number the
    verifier re-derives from the target evidence clocks, which is what makes
    "the run honoured the causal boundary" checkable after the fact.
    """

    kind: Literal["SectorForecastSurface"] = "SectorForecastSurface"
    identity_class: Literal["DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"] = (
        "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"
    )
    target_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipe: SectorForecastRecipe
    program_binding: SectorForecastProgramBinding
    ordered_sectors: tuple[str, ...] = Field(min_length=1)
    forecast_formation_sessions: tuple[date, ...] = Field(min_length=1)
    values: tuple[tuple[float | None, ...], ...] = Field(min_length=1)
    unavailable_reasons: tuple[tuple[str | None, ...], ...] = Field(min_length=1)
    training_row_counts: tuple[int, ...] = Field(min_length=1)
    values_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        binding = self.program_binding
        if (
            self.recipe.method_id != binding.method_id
            or self.recipe.recipe_hash != binding.recipe_hash
            or self.target_evidence_hash != binding.target_evidence_hash
            or self.forecast_formation_sessions != binding.forecast_formation_sessions
        ):
            raise SectorResearchError("sector_research.surface_not_its_program")
        rows = len(self.forecast_formation_sessions)
        columns = len(self.ordered_sectors)
        if (
            len(self.values) != rows
            or len(self.unavailable_reasons) != rows
            or len(self.training_row_counts) != rows
            or any(len(row) != columns for row in self.values)
            or any(len(row) != columns for row in self.unavailable_reasons)
            or any(count < 0 for count in self.training_row_counts)
        ):
            raise SectorResearchError("sector_research.surface_grid_shape_invalid")
        for value_row, reason_row in zip(self.values, self.unavailable_reasons, strict=True):
            for value, reason in zip(value_row, reason_row, strict=True):
                if (value is None) == (reason is None):
                    raise SectorResearchError("sector_research.surface_cell_invalid")
                if value is not None and not np.isfinite(value):
                    raise SectorResearchError("sector_research.surface_value_nonfinite")
        if self.values_identity != sector_array_identity(grid_to_matrix(self.values)):
            raise SectorResearchError("sector_research.surface_values_identity_invalid")
        if self.surface_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"surface_hash"})
        ):
            raise SectorResearchError("sector_research.surface_identity_invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        draft = dict(values)
        draft.pop("surface_hash", None)
        provisional = cls.model_construct(**draft, surface_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"surface_hash"})
        return cast(
            Self, cls.model_validate({**identity, "surface_hash": canonical_hash(identity)})
        )


class SectorForecastEvaluation(_Contract):
    """Realized-versus-forecast summary per sector, over cells where both exist.

    Retrospective by construction: every realized value in the development
    target evidence had matured before it was sealed, so comparing a forecast
    formed at ``F`` against the label realized for ``F`` is bookkeeping about
    the past, not information leaking into the model.
    """

    kind: Literal["SectorForecastEvaluation"] = "SectorForecastEvaluation"
    surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    realized_lane_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_sectors: tuple[str, ...] = Field(min_length=1)
    evaluated_counts: tuple[int, ...] = Field(min_length=1)
    mean_errors: tuple[float | None, ...] = Field(min_length=1)
    mean_squared_errors: tuple[float | None, ...] = Field(min_length=1)
    evaluation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        columns = len(self.ordered_sectors)
        if (
            len(self.evaluated_counts) != columns
            or len(self.mean_errors) != columns
            or len(self.mean_squared_errors) != columns
            or any(count < 0 for count in self.evaluated_counts)
        ):
            raise SectorResearchError("sector_research.evaluation_axis_invalid")
        for count, mean_error, mean_squared in zip(
            self.evaluated_counts, self.mean_errors, self.mean_squared_errors, strict=True
        ):
            if (count == 0) != (mean_error is None) or (count == 0) != (mean_squared is None):
                raise SectorResearchError("sector_research.evaluation_cell_invalid")
        if self.evaluation_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evaluation_hash"})
        ):
            raise SectorResearchError("sector_research.evaluation_identity_invalid")
        return self


def evaluate_sector_forecast_surface(
    *,
    evidence: SectorTargetEvidence,
    surface: SectorForecastSurface,
) -> SectorForecastEvaluation:
    """Derive the evaluation from the two artifacts it relates, and nothing else.

    Pure and adapter-free, so the verifier can recompute it byte-for-byte from
    the published children: an evaluation that cannot be re-derived is not
    evidence of anything.
    """

    if surface.target_evidence_hash != evidence.evidence_hash:
        raise SectorResearchError("sector_research.evaluation_target_mismatch")
    if surface.ordered_sectors != evidence.ordered_sectors:
        raise SectorResearchError("sector_research.evaluation_sector_axis_mismatch")
    session_index = {value: index for index, value in enumerate(evidence.formation_sessions)}
    errors: list[list[float]] = [[] for _ in surface.ordered_sectors]
    for row, formation in enumerate(surface.forecast_formation_sessions):
        target_row = session_index.get(formation)
        if target_row is None:
            raise SectorResearchError("sector_research.evaluation_session_not_in_target")
        for column in range(len(surface.ordered_sectors)):
            forecast = surface.values[row][column]
            realized = evidence.sector_target_values[target_row][column]
            if forecast is None or realized is None:
                continue
            errors[column].append(float(forecast) - float(realized))
    counts = tuple(len(value) for value in errors)
    mean_errors = tuple(
        float(np.mean(np.asarray(value, dtype=np.float64))) if value else None for value in errors
    )
    mean_squared = tuple(
        float(np.mean(np.square(np.asarray(value, dtype=np.float64)))) if value else None
        for value in errors
    )
    values: dict[str, object] = {
        "kind": "SectorForecastEvaluation",
        "surface_hash": surface.surface_hash,
        "target_evidence_hash": evidence.evidence_hash,
        "realized_lane_identity": evidence.sector_target_identity,
        "ordered_sectors": list(surface.ordered_sectors),
        "evaluated_counts": list(counts),
        "mean_errors": list(mean_errors),
        "mean_squared_errors": list(mean_squared),
    }
    return cast(
        SectorForecastEvaluation,
        SectorForecastEvaluation.model_validate(
            {**values, "evaluation_hash": canonical_hash(values)}
        ),
    )


def causal_training_selection(
    *,
    evidence: SectorTargetEvidence,
    program: SectorForecastProgramBinding,
    forecast_row: int,
) -> tuple[int, ...]:
    """Indices of the evidence formations one forecast row may train on.

    One derivation, shared by the executor that builds the bound input and the
    verifier that re-checks it -- two copies of this rule is how they drift.
    The selection is by clocks, never by index arithmetic: a formation trains a
    forecast only if its label's availability date is on or before the refit
    formation date, and the refit formation is the cadence-aligned session at
    or before the forecast row.
    """

    sessions = program.forecast_formation_sessions
    if not 0 <= forecast_row < len(sessions):
        raise SectorResearchError("sector_research.training_selection_row_invalid")
    cadence = program.refit_every_sessions
    refit_row = forecast_row - (forecast_row % cadence) if cadence > 0 else forecast_row
    refit_formation = sessions[refit_row]
    return tuple(
        index
        for index, (formation, available_at) in enumerate(
            zip(evidence.formation_sessions, evidence.target_available_sessions, strict=True)
        )
        if program.training_start <= formation <= program.training_end
        and available_at <= refit_formation
    )


class SectorExperimentEvidence(_Contract):
    """The root record of one Sector development experiment.

    ``ordered_child_uris`` is validated against the child hashes and the
    category constants, in publication order. Completeness and order are one
    claim: a graph missing a child is incomplete, and a reordered one describes
    a publication sequence that never happened.
    """

    kind: Literal["SectorExperimentEvidence"] = "SectorExperimentEvidence"
    causal_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sector_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_id: str = Field(min_length=1, max_length=96)
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    forecast_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_child_uris: tuple[str, ...] = Field(min_length=3, max_length=3)
    experiment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        expected = (
            sector_artifact_uri(SECTOR_TARGET_EVIDENCE_CATEGORY, self.target_evidence_hash),
            sector_artifact_uri(SECTOR_FORECAST_SURFACE_CATEGORY, self.forecast_surface_hash),
            sector_artifact_uri(SECTOR_FORECAST_EVALUATION_CATEGORY, self.evaluation_hash),
        )
        if self.ordered_child_uris != expected:
            raise SectorResearchError("sector_research.experiment_children_invalid")
        if self.experiment_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"experiment_hash"})
        ):
            raise SectorResearchError("sector_research.experiment_identity_invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        draft = dict(values)
        draft.pop("experiment_hash", None)
        provisional = cls.model_construct(**draft, experiment_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"experiment_hash"})
        return cast(
            Self, cls.model_validate({**identity, "experiment_hash": canonical_hash(identity)})
        )


class SectorArtifactReadbackError(ValueError):
    """A stored Sector development artifact exists but does not verify."""


class SectorDevelopmentArtifactStore:
    """Immutable content-addressed JSON evidence under ``sector-research/``.

    Absent and invalid stay distinct on the way out: a hash that was never
    published raises ``FileNotFoundError``; content that exists but fails its
    own identity raises ``SectorArtifactReadbackError``. Collapsing the two is
    how a corrupt artifact once got reported as merely waiting.
    """

    def __init__(self, artifact_root: Path) -> None:
        self.root = artifact_root.resolve() / "sector-research"

    @staticmethod
    def _require_hash(value: str) -> None:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise SectorResearchError("sector_research.artifact_identity_not_sha256")

    def _path(self, category: str, content_hash: str) -> Path:
        self._require_hash(content_hash)
        return self.root / category / f"{content_hash}.json"

    @staticmethod
    def _json_bytes(payload: Mapping[str, object]) -> bytes:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")

    @staticmethod
    def _atomic_write(target: Path, content: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        staged.write_bytes(content)
        os.replace(staged, target)
        staged.unlink(missing_ok=True)

    def _publish(self, *, category: str, value: BaseModel, identity_field: str) -> str:
        payload = value.model_dump(mode="json")
        content_hash = str(payload[identity_field])
        self._require_hash(content_hash)
        identity = dict(payload)
        identity.pop(identity_field)
        if canonical_hash(identity) != content_hash:
            raise SectorResearchError("sector_research.artifact_identity_field_invalid")
        serialized = self._json_bytes(payload)
        target = self._path(category, content_hash)
        if target.exists():
            if target.read_bytes() != serialized:
                raise SectorResearchError("sector_research.artifact_identity_reused")
        else:
            self._atomic_write(target, serialized)
        return sector_artifact_uri(category, content_hash)

    def _load(self, *, category: str, content_hash: str, identity_field: str) -> dict[str, object]:
        target = self._path(category, content_hash)
        if not target.is_file():
            raise FileNotFoundError(f"sector artifact is missing: {category}/{content_hash}")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get(identity_field) != content_hash:
            raise SectorArtifactReadbackError("sector artifact requested identity is invalid")
        identity = dict(payload)
        identity.pop(identity_field)
        if canonical_hash(identity) != content_hash:
            raise SectorArtifactReadbackError("sector artifact payload hash is invalid")
        return cast(dict[str, object], payload)

    # ------------------------------------------------------------------ publish
    def publish_target_evidence(self, value: SectorTargetEvidence) -> str:
        return self._publish(
            category=SECTOR_TARGET_EVIDENCE_CATEGORY, value=value, identity_field="evidence_hash"
        )

    def publish_forecast_surface(self, value: SectorForecastSurface) -> str:
        return self._publish(
            category=SECTOR_FORECAST_SURFACE_CATEGORY, value=value, identity_field="surface_hash"
        )

    def publish_evaluation(self, value: SectorForecastEvaluation) -> str:
        return self._publish(
            category=SECTOR_FORECAST_EVALUATION_CATEGORY,
            value=value,
            identity_field="evaluation_hash",
        )

    def publish_experiment_evidence(self, value: SectorExperimentEvidence) -> str:
        return self._publish(
            category=SECTOR_EXPERIMENT_EVIDENCE_CATEGORY,
            value=value,
            identity_field="experiment_hash",
        )

    # -------------------------------------------------------------------- load
    def load_target_evidence(self, content_hash: str) -> SectorTargetEvidence:
        return cast(
            SectorTargetEvidence,
            SectorTargetEvidence.model_validate(
                self._load(
                    category=SECTOR_TARGET_EVIDENCE_CATEGORY,
                    content_hash=content_hash,
                    identity_field="evidence_hash",
                )
            ),
        )

    def load_forecast_surface(self, content_hash: str) -> SectorForecastSurface:
        return cast(
            SectorForecastSurface,
            SectorForecastSurface.model_validate(
                self._load(
                    category=SECTOR_FORECAST_SURFACE_CATEGORY,
                    content_hash=content_hash,
                    identity_field="surface_hash",
                )
            ),
        )

    def load_evaluation(self, content_hash: str) -> SectorForecastEvaluation:
        return cast(
            SectorForecastEvaluation,
            SectorForecastEvaluation.model_validate(
                self._load(
                    category=SECTOR_FORECAST_EVALUATION_CATEGORY,
                    content_hash=content_hash,
                    identity_field="evaluation_hash",
                )
            ),
        )

    def load_experiment_evidence(self, content_hash: str) -> SectorExperimentEvidence:
        return cast(
            SectorExperimentEvidence,
            SectorExperimentEvidence.model_validate(
                self._load(
                    category=SECTOR_EXPERIMENT_EVIDENCE_CATEGORY,
                    content_hash=content_hash,
                    identity_field="experiment_hash",
                )
            ),
        )

    def list_experiment_evidence(self) -> tuple[str, ...]:
        """Published experiment hashes in canonical name order, without loading them."""

        return self.list_category(SECTOR_EXPERIMENT_EVIDENCE_CATEGORY)

    def list_category(self, category: str) -> tuple[str, ...]:
        """Published hashes in one category, canonical name order, unloaded."""

        root = self.root / category
        if not root.is_dir():
            return ()
        return tuple(sorted(path.stem for path in root.glob("*.json")))

    # --------------------------------------------------------------- Campaign
    # The Campaign categories are published and read through the same identity
    # rules as the four above, generically rather than through five near-identical
    # pairs. Typing stays with the caller that names the contract: the store
    # deliberately does not import the calibration, comparison or decision
    # modules, because those import the store.
    def publish_contract(self, *, category: str, value: BaseModel, identity_field: str) -> str:
        return self._publish(category=category, value=value, identity_field=identity_field)

    def load_contract[ContractT: BaseModel](
        self,
        *,
        category: str,
        content_hash: str,
        identity_field: str,
        model: type[ContractT],
    ) -> ContractT:
        return cast(
            ContractT,
            model.model_validate(
                self._load(
                    category=category, content_hash=content_hash, identity_field=identity_field
                )
            ),
        )


__all__ = [
    "SECTOR_CAMPAIGN_DECISION_CATEGORY",
    "SECTOR_CAMPAIGN_DOSSIER_CATEGORY",
    "SECTOR_CAMPAIGN_REPLAY_CATEGORY",
    "SECTOR_EXPERIMENT_EVIDENCE_CATEGORY",
    "SECTOR_FORECAST_EVALUATION_CATEGORY",
    "SECTOR_FORECAST_SURFACE_CATEGORY",
    "SECTOR_METHOD_COMPARISON_CATEGORY",
    "SECTOR_SHRINK_CALIBRATION_CATEGORY",
    "SECTOR_TARGET_EVIDENCE_CATEGORY",
    "SectorArtifactReadbackError",
    "SectorDevelopmentArtifactStore",
    "SectorExperimentEvidence",
    "SectorForecastEvaluation",
    "SectorForecastProgramBinding",
    "SectorForecastSurface",
    "causal_training_selection",
    "evaluate_sector_forecast_surface",
    "sector_artifact_uri",
    "sector_development_source_closure_hash",
]
