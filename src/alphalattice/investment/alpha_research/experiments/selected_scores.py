"""The canonical fold-selected predicted-Z projection of one Stage 3 decision.

Stage 3 selects **per fold**. Its decision names a set of admitted methods, and
each outer fold's inner search picked one of them; the dossier records which. The
durable graph therefore contains everything needed to say "these are the
predictions the Stage 3 decision selected", but it did not previously expose that
as one object, so a consumer had to reconstruct it -- and the Portfolio Desk,
reconstructing it, reached for whichever candidate happened to be first in an
unrelated score surface instead.

This is that object, and it is deliberately narrow: it resolves, it verifies, and
it concatenates. It fits nothing, predicts nothing, and re-interprets no
selection. A method that the decision did not admit, a trial no inner-selection
record chose, or two folds claiming the same formation are all refusals.

The values are the model's **predicted residual Z**, in the space of the
canonical target. They are not an expected return and must not be handed to an
optimizer as one -- converting them takes the Alpha calibration and the
cross-sectional dispersion forecast, which are separate owners.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]

_HASH = r"^[0-9a-f]{64}$"

ALPHA_SELECTED_SCORE_SEMANTICS = "PREDICTED_RESIDUAL_Z_CANONICAL_TARGET"
"""What the values are. Named so a consumer cannot mistake them for a return."""


class AlphaSelectedScoreError(ValueError):
    """Stable fail-closed boundary for the selected-score projection."""


class AlphaSelectedFold(BaseModel):  # type: ignore[misc]
    """One outer fold, and the trial its own inner search selected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["AlphaSelectedFold"] = "AlphaSelectedFold"
    fold_index: int = Field(ge=0)
    horizon_sessions: int = Field(ge=1)
    selected_method_id: str = Field(min_length=1, max_length=96)
    selected_trial_hash: str = Field(pattern=_HASH)
    inner_selection_record_hash: str = Field(pattern=_HASH)
    prediction_artifact_hash: str = Field(pattern=_HASH)
    prediction_value_hash: str = Field(pattern=_HASH)
    formation_session_count: int = Field(ge=1)


class AlphaSelectedScoreProjection(BaseModel):  # type: ignore[misc]
    """The Stage 3 decision's own predictions, on one session-by-listing axis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["AlphaSelectedScoreProjection"] = "AlphaSelectedScoreProjection"
    identity_class: Literal["DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"] = (
        "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"
    )
    value_semantics: Literal["PREDICTED_RESIDUAL_Z_CANONICAL_TARGET"] = (
        "PREDICTED_RESIDUAL_Z_CANONICAL_TARGET"
    )
    """Not a return. A consumer that wants return units must calibrate and scale."""

    dossier_hash: str = Field(pattern=_HASH)
    decision_receipt_hash: str = Field(pattern=_HASH)
    program_hash: str = Field(pattern=_HASH)
    target_evidence_hash: str = Field(pattern=_HASH)
    target_recipe_binding_hash: str = Field(pattern=_HASH)
    outcome_method_binding_hash: str = Field(pattern=_HASH)
    horizon_sessions: int = Field(ge=1)
    admitted_method_ids: tuple[str, ...] = Field(min_length=1)
    """Every method the decision admitted, kept plural.

    Stage 3 chose per fold; there is no global winner, and manufacturing one here
    would be inventing a decision nobody made.
    """

    ordered_folds: tuple[AlphaSelectedFold, ...] = Field(min_length=1)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    resolved_cell_count: int = Field(ge=1)
    unresolved_cell_count: int = Field(ge=0)
    predicted_z_values_hash: str = Field(pattern=_HASH)
    projection_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        draft = dict(values)
        draft.pop("projection_hash", None)
        provisional = cls.model_construct(**draft, projection_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"projection_hash"})
        return cls(**draft, projection_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.formation_sessions != tuple(sorted(set(self.formation_sessions))):
            raise AlphaSelectedScoreError("alpha_research.selected_scores_session_axis_unordered")
        if self.ordered_listing_ids != tuple(sorted(set(self.ordered_listing_ids))):
            raise AlphaSelectedScoreError("alpha_research.selected_scores_listing_axis_unordered")
        folds = tuple(value.fold_index for value in self.ordered_folds)
        if len(set(folds)) != len(folds):
            raise AlphaSelectedScoreError("alpha_research.selected_scores_fold_duplicated")
        if any(value.horizon_sessions != self.horizon_sessions for value in self.ordered_folds):
            raise AlphaSelectedScoreError("alpha_research.selected_scores_horizon_mixed")
        if any(
            value.selected_method_id not in self.admitted_method_ids for value in self.ordered_folds
        ):
            raise AlphaSelectedScoreError("alpha_research.selected_scores_method_not_admitted")
        cells = len(self.formation_sessions) * len(self.ordered_listing_ids)
        if self.resolved_cell_count + self.unresolved_cell_count != cells:
            raise AlphaSelectedScoreError("alpha_research.selected_scores_cell_count_invalid")
        if self.projection_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"projection_hash"})
        ):
            raise AlphaSelectedScoreError("alpha_research.selected_scores_identity_invalid")
        return self


@dataclass(frozen=True, slots=True)
class AlphaSelectedRowAxis:
    """The exact ordered row axis Stage 3 calibrated on, rebuilt from its graph.

    A row axis rather than a matrix, because the calibration is a fit over rows
    and its cross-fitting split is expressed in row positions. The matrix
    projection beside it answers a different question -- "what is the signal at
    this formation for this listing" -- and neither can be derived from the other
    without inventing an ordering.
    """

    ordered_folds: tuple[AlphaSelectedFold, ...]
    row_sessions: tuple[date, ...]
    row_listing_ids: tuple[str, ...]
    predicted_z: FloatArray
    ordered_fold_row_counts: tuple[int, ...]
    row_axis_hash: str

    @property
    def row_count(self) -> int:
        return len(self.row_sessions)


def predicted_z_values_hash(values: FloatArray) -> str:
    """Byte identity of the predicted-Z matrix, NaNs included."""

    import hashlib

    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _resolve_selected_folds(
    *,
    dossier: object,
    decision: object,
    prediction_surface_root: Path,
    horizon_sessions: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[str, ...],
    dict[str, Any],
    list[AlphaSelectedFold],
    list[tuple[tuple[str, ...], tuple[float, ...]]],
]:
    """Resolve which trial each outer fold selected, and read its predictions.

    Shared by both projections so "which predictions did this decision select"
    is answered once. The per-fold rows are returned **in the parquet's own
    order**, not as a mapping: the row axis downstream is that order, and
    rebuilding it from a dict would make it depend on insertion order instead.
    """

    dossier_payload = dossier.model_dump(mode="json")  # type: ignore[attr-defined]
    decision_payload = decision.model_dump(mode="json")  # type: ignore[attr-defined]
    submission = decision_payload["submission"]
    if submission["dossier_hash"] != dossier_payload["dossier_hash"]:
        raise AlphaSelectedScoreError("alpha_research.selected_scores_decision_not_this_dossier")
    admitted = tuple(submission["selected_method_ids"])
    if not admitted:
        raise AlphaSelectedScoreError("alpha_research.selected_scores_decision_selected_nothing")

    # Keyed by the inner-selection record, not by ``trial_hash``. A trial hash
    # identifies a *configuration* -- method plus parameters -- and the same
    # configuration legitimately wins in more than one fold, so the dossier
    # carries it more than once with different predictions each time. Indexing by
    # it silently keeps whichever entry came last and hands one fold another
    # fold's predictions.
    trials = {
        (str(value["inner_selection_record_hash"]), str(value["trial_hash"])): value
        for value in dossier_payload["trial_evidence"]
    }
    horizon = next(
        value
        for value in dossier_payload["program"]["horizons"]
        if int(value["horizon_sessions"]) == horizon_sessions
    )
    folds: list[AlphaSelectedFold] = []
    per_fold: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
    for record in sorted(
        (
            value
            for value in dossier_payload["inner_selection_records"]
            if int(value["horizon_sessions"]) == horizon_sessions
        ),
        key=lambda value: int(value["fold_index"]),
    ):
        if record["selected_method_id"] not in admitted:
            raise AlphaSelectedScoreError("alpha_research.selected_scores_method_not_admitted")
        trial = trials.get((str(record["record_hash"]), str(record["selected_trial_hash"])))
        if trial is None:
            raise AlphaSelectedScoreError("alpha_research.selected_scores_trial_unresolved")
        if (
            int(trial["fold_index"]) != int(record["fold_index"])
            or int(trial["horizon_sessions"]) != horizon_sessions
        ):
            raise AlphaSelectedScoreError("alpha_research.selected_scores_trial_not_this_fold")
        path = prediction_surface_root / f"{trial['prediction_artifact_hash']}.parquet"
        if not path.is_file():
            raise AlphaSelectedScoreError("alpha_research.selected_scores_prediction_missing")
        table = pq.read_table(path)
        if set(table.schema.names) != {"row_id", "prediction"}:
            raise AlphaSelectedScoreError(
                "alpha_research.selected_scores_prediction_schema_invalid"
            )
        row_ids = tuple(str(value) for value in table.column("row_id").to_pylist())
        predictions = tuple(float(value) for value in table.column("prediction").to_pylist())
        if len(row_ids) != len(predictions):
            raise AlphaSelectedScoreError("alpha_research.selected_scores_prediction_axis_invalid")
        if len(set(row_ids)) != len(row_ids):
            raise AlphaSelectedScoreError("alpha_research.selected_scores_row_duplicated")
        per_fold.append((row_ids, predictions))
        folds.append(
            AlphaSelectedFold(
                fold_index=int(record["fold_index"]),
                horizon_sessions=horizon_sessions,
                selected_method_id=str(record["selected_method_id"]),
                selected_trial_hash=str(record["selected_trial_hash"]),
                inner_selection_record_hash=str(record["record_hash"]),
                prediction_artifact_hash=str(trial["prediction_artifact_hash"]),
                prediction_value_hash=str(trial["prediction_value_hash"]),
                formation_session_count=len({key.split("|", 1)[0] for key in row_ids}),
            )
        )
    if not folds:
        raise AlphaSelectedScoreError("alpha_research.selected_scores_horizon_absent")

    claimed: dict[str, int] = {}
    for index, (row_ids, _predictions) in enumerate(per_fold):
        for key in row_ids:
            session = key.split("|", 1)[0]
            owner = claimed.setdefault(session, index)
            if owner != index:
                raise AlphaSelectedScoreError("alpha_research.selected_scores_fold_overlap")
    return dossier_payload, decision_payload, admitted, horizon, folds, per_fold


def project_alpha_selected_scores(
    *,
    dossier: object,
    decision: object,
    prediction_surface_root: Path,
    horizon_sessions: int,
) -> tuple[AlphaSelectedScoreProjection, FloatArray]:
    """Assemble the fold-selected predictions of one Stage 3 decision.

    The caller has already verified the decision against the dossier through the
    Alpha owner; this resolves what that decision selected. Folds are walked in
    index order and a formation may be claimed by exactly one of them -- outer
    folds do not overlap, and a duplicate would mean two different models
    predicting the same cell with no rule for which wins.
    """

    dossier_payload, decision_payload, admitted, horizon, folds, per_fold = _resolve_selected_folds(
        dossier=dossier,
        decision=decision,
        prediction_surface_root=prediction_surface_root,
        horizon_sessions=horizon_sessions,
    )
    sessions = tuple(
        sorted({date.fromisoformat(key.split("|", 1)[0]) for keys, _ in per_fold for key in keys})
    )
    listings = tuple(sorted({key.split("|", 1)[1] for keys, _ in per_fold for key in keys}))
    session_position = {value: index for index, value in enumerate(sessions)}
    listing_position = {value: index for index, value in enumerate(listings)}
    matrix: FloatArray = np.full((len(sessions), len(listings)), np.nan, dtype=np.float64)
    for row_ids, predictions in per_fold:
        for key, prediction in zip(row_ids, predictions, strict=True):
            session, listing = key.split("|", 1)
            matrix[session_position[date.fromisoformat(session)], listing_position[listing]] = (
                prediction
            )
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    resolved = int(np.isfinite(matrix).sum())
    projection = AlphaSelectedScoreProjection.create(
        dossier_hash=dossier_payload["dossier_hash"],
        decision_receipt_hash=decision_payload["receipt_hash"],
        program_hash=dossier_payload["program"]["program_hash"],
        target_evidence_hash=horizon["target_evidence_hash"],
        target_recipe_binding_hash=horizon["target_recipe_binding_hash"],
        outcome_method_binding_hash=horizon["outcome_method_binding_hash"],
        horizon_sessions=horizon_sessions,
        admitted_method_ids=admitted,
        ordered_folds=tuple(folds),
        formation_sessions=sessions,
        ordered_listing_ids=listings,
        resolved_cell_count=resolved,
        unresolved_cell_count=matrix.size - resolved,
        predicted_z_values_hash=predicted_z_values_hash(matrix),
    )
    matrix.setflags(write=False)
    return projection, matrix


def project_alpha_selected_rows(
    *,
    dossier: object,
    decision: object,
    prediction_surface_root: Path,
    horizon_sessions: int,
) -> AlphaSelectedRowAxis:
    """Rebuild the exact ordered row axis Stage 3's calibration consumed.

    The rows are the outer folds' validation rows, concatenated in fold-index
    order and in each fold's own stored order. That concatenation *is* Stage 3's
    calibration axis: the campaign built it from the same surfaces and handed it
    straight to the calibration owner, so reproducing it here is reconstruction
    rather than reinterpretation -- and the published calibration evidence
    carries its per-row session list, which a consumer can hold this against.
    """

    _dossier_payload, _decision_payload, _admitted, _horizon, folds, per_fold = (
        _resolve_selected_folds(
            dossier=dossier,
            decision=decision,
            prediction_surface_root=prediction_surface_root,
            horizon_sessions=horizon_sessions,
        )
    )
    sessions: list[date] = []
    listings: list[str] = []
    values: list[float] = []
    for row_ids, predictions in per_fold:
        for key, prediction in zip(row_ids, predictions, strict=True):
            session, listing = key.split("|", 1)
            sessions.append(date.fromisoformat(session))
            listings.append(listing)
            values.append(prediction)
    if not sessions:
        raise AlphaSelectedScoreError("alpha_research.selected_scores_row_axis_empty")
    if len(set(zip(sessions, listings, strict=True))) != len(sessions):
        raise AlphaSelectedScoreError("alpha_research.selected_scores_row_duplicated")
    predicted: FloatArray = np.ascontiguousarray(values, dtype=np.float64)
    if not bool(np.isfinite(predicted).all()):
        raise AlphaSelectedScoreError("alpha_research.selected_scores_row_value_nonfinite")
    predicted.setflags(write=False)
    counts = tuple(len(row_ids) for row_ids, _ in per_fold)
    return AlphaSelectedRowAxis(
        ordered_folds=tuple(folds),
        row_sessions=tuple(sessions),
        row_listing_ids=tuple(listings),
        predicted_z=predicted,
        ordered_fold_row_counts=counts,
        row_axis_hash=str(
            canonical_hash(
                {
                    "kind": "AlphaSelectedRowAxis",
                    "horizon_sessions": horizon_sessions,
                    "ordered_fold_row_counts": list(counts),
                    "row_ids": canonical_hash(
                        [
                            f"{session.isoformat()}|{listing}"
                            for session, listing in zip(sessions, listings, strict=True)
                        ]
                    ),
                }
            )
        ),
    )


__all__ = [
    "ALPHA_SELECTED_SCORE_SEMANTICS",
    "AlphaSelectedFold",
    "AlphaSelectedRowAxis",
    "AlphaSelectedScoreError",
    "AlphaSelectedScoreProjection",
    "predicted_z_values_hash",
    "project_alpha_selected_rows",
    "project_alpha_selected_scores",
]
