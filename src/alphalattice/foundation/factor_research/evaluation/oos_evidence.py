"""Complete-fold out-of-sample evidence for one-session Factor Research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.factor_research.evaluation.rank_correlation import (
    average_ranks_over_rows,
    average_ranks_vectorized,
    centered_ranks,
    correlation_from_centered_ranks,
    decile_spread_from_ranks,
    ordinal_percentile_columns,
    stable_column_orders,
)
from alphalattice.foundation.factor_research.inputs.execution_target import FactorTargetSurface
from alphalattice.foundation.factor_research.programs.sealed import seal_contract
from alphalattice.foundation.factor_research.programs.walk_forward import FactorWalkForwardPlan
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.validation.screening_statistics import (
    benjamini_yekutieli,
    newey_west_mean_test,
)

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.intp]


class FactorEvidenceBoundaryError(ValueError):
    """Stable failure raised when formal OOS evidence cannot be computed."""


class FactorEvidenceClassification(StrEnum):
    """Classification of a factor's validation-only evidence."""

    POSITIVE_OOS_EVIDENCE = "POSITIVE_OOS_EVIDENCE"
    MIXED_OOS_EVIDENCE = "MIXED_OOS_EVIDENCE"
    NO_DETECTABLE_EFFECT = "NO_DETECTABLE_EFFECT"
    NEGATIVE_OOS_EVIDENCE = "NEGATIVE_OOS_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class FactorEvidencePolicy(_Contract):
    """Sealed rules for orientation, coverage and OOS classification."""

    kind: Literal["FactorEvidencePolicy"] = "FactorEvidencePolicy"
    prediction_horizon_sessions: Literal[1] = 1
    orientation_source: Literal["TRAINING_MEAN_DAILY_SPEARMAN"] = "TRAINING_MEAN_DAILY_SPEARMAN"
    evaluation_source: Literal["VALIDATION_ONLY"] = "VALIDATION_ONLY"
    fdr_method: Literal["BENJAMINI_YEKUTIELI"] = "BENJAMINI_YEKUTIELI"
    classification_rule: Literal["DIRECTIONAL_RESEARCH_EVIDENCE"] = "DIRECTIONAL_RESEARCH_EVIDENCE"
    fdr_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    minimum_pair_coverage: float = Field(default=0.8, gt=0.0, le=1.0)
    minimum_cross_section_observations: int = Field(default=100, ge=2)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_policy(self) -> FactorEvidencePolicy:
        """Verify the policy's content hash.

        Returns:
            The validated policy.

        Raises:
            ValueError: If the policy hash does not match its fields.
        """
        expected = canonical_hash(self.model_dump(mode="json", exclude={"policy_hash"}))
        if self.policy_hash != expected:
            raise ValueError("Factor evidence policy hash is invalid")
        return self


class FactorFoldEvidence(_Contract):
    """Sealed training orientation and validation evidence for one fold."""

    kind: Literal["FactorFoldEvidence"] = "FactorFoldEvidence"
    factor_id: str = Field(min_length=1)
    fold_index: int = Field(ge=0)
    orientation: Literal[-1, 1] | None
    training_period_count: int = Field(ge=0)
    validation_period_count: int = Field(ge=0)
    training_mean_rank_ic: float | None
    validation_oriented_mean_rank_ic: float | None
    validation_oriented_mean_simple_spread: float | None
    validation_pair_coverage_mean: float
    validation_rank_turnover_mean: float | None
    fold_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_evidence(self) -> FactorFoldEvidence:
        """Verify a fold's coverage, finite values and content hash.

        Returns:
            The validated fold evidence.

        Raises:
            ValueError: If a coverage, value or hash is invalid.
        """
        if not 0.0 <= self.validation_pair_coverage_mean <= 1.0:
            raise ValueError("Factor fold coverage is invalid")
        for value in (
            self.training_mean_rank_ic,
            self.validation_oriented_mean_rank_ic,
            self.validation_oriented_mean_simple_spread,
            self.validation_rank_turnover_mean,
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError("Factor fold evidence is non-finite")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"fold_hash"}))
        if self.fold_hash != expected:
            raise ValueError("Factor fold evidence hash is invalid")
        return self


class FactorOosEvidenceItem(_Contract):
    """Sealed cross-fold validation evidence for one factor."""

    kind: Literal["FactorOosEvidenceItem"] = "FactorOosEvidenceItem"
    factor_id: str = Field(min_length=1)
    fold_evidence: tuple[FactorFoldEvidence, ...] = Field(min_length=1)
    validation_period_count: int = Field(ge=0)
    mean_oriented_rank_ic: float | None
    rank_ic_volatility: float | None
    rank_ic_ir: float | None
    rank_ic_positive_hit_rate: float | None
    rank_ic_hac_lag: int = Field(ge=0)
    rank_ic_hac_t_stat: float | None
    rank_ic_raw_p_value: float
    rank_ic_by_q_value: float
    mean_oriented_simple_spread: float | None
    simple_spread_hac_lag: int = Field(ge=0)
    simple_spread_hac_t_stat: float | None
    simple_spread_raw_p_value: float
    validation_pair_coverage_mean: float
    validation_rank_turnover_mean: float | None
    classification: FactorEvidenceClassification
    reason_codes: tuple[str, ...]
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_evidence(self) -> FactorOosEvidenceItem:
        """Verify an item's probability, coverage and content hash.

        Returns:
            The validated factor evidence.

        Raises:
            ValueError: If a probability, coverage or hash is invalid.
        """
        probabilities = (
            self.rank_ic_raw_p_value,
            self.rank_ic_by_q_value,
            self.simple_spread_raw_p_value,
            self.validation_pair_coverage_mean,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Factor OOS probability or coverage is invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"evidence_hash"}))
        if self.evidence_hash != expected:
            raise ValueError("Factor OOS evidence hash is invalid")
        return self


class FactorOosEvidenceReport(_Contract):
    """Sealed factor-axis report of validation-only evidence."""

    kind: Literal["FactorOosEvidenceReport"] = "FactorOosEvidenceReport"
    feature_panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_panel_manifest_ref: str = Field(min_length=1)
    target_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    walk_forward_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    hypothesis_count: int = Field(ge=1)
    factor_ids: tuple[str, ...] = Field(min_length=1)
    items: tuple[FactorOosEvidenceItem, ...] = Field(min_length=1)
    classification_counts: dict[FactorEvidenceClassification, int]
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_report(self) -> FactorOosEvidenceReport:
        """Verify the factor axis, classification counts and report hash.

        Returns:
            The validated evidence report.

        Raises:
            ValueError: If the report's axis, counts or hash are inconsistent.
        """
        if self.factor_ids != tuple(sorted(set(self.factor_ids))):
            raise ValueError("Factor OOS factor axis is not canonical")
        if tuple(item.factor_id for item in self.items) != self.factor_ids:
            raise ValueError("Factor OOS item axis is inconsistent")
        if self.hypothesis_count != len(self.factor_ids):
            raise ValueError("Factor OOS hypothesis family is incomplete")
        expected_counts = {
            classification: sum(item.classification is classification for item in self.items)
            for classification in FactorEvidenceClassification
        }
        if self.classification_counts != expected_counts:
            raise ValueError("Factor OOS classification counts are invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"report_hash"}))
        if self.report_hash != expected:
            raise ValueError("Factor OOS report hash is invalid")
        return self


@dataclass(frozen=True, slots=True)
class _EvidenceSurface:
    sessions: tuple[date, ...]
    listings: tuple[str, ...]
    factors: tuple[str, ...]
    features: FloatArray
    fit_targets: FloatArray
    simple_returns: FloatArray
    members: BoolArray


@dataclass(frozen=True, slots=True)
class _FactorComputation:
    folds: tuple[FactorFoldEvidence, ...]
    validation_ics: tuple[float, ...]
    validation_spreads: tuple[float, ...]
    validation_coverages: tuple[float, ...]
    validation_turnovers: tuple[float, ...]


def build_factor_evidence_policy(
    *, minimum_cross_section_observations: int = 100
) -> FactorEvidencePolicy:
    """Seal the fixed OOS evidence rules with the requested observation floor.

    Args:
        minimum_cross_section_observations: Minimum paired listings in a period.

    Returns:
        The content-bound evidence policy.
    """
    return seal_contract(
        FactorEvidencePolicy,
        "policy_hash",
        minimum_cross_section_observations=minimum_cross_section_observations,
    )


def _positions(
    *,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
    session_column: pa.ChunkedArray,
    listing_column: pa.ChunkedArray,
) -> IntArray:
    """Row positions on the dense ``sessions x listings`` grid.

    One hash lookup per row inside Arrow instead of a Python dictionary per
    row; a session or listing off the axis answers null and is refused as
    before.
    """
    session_positions = pc.index_in(
        session_column, value_set=pa.array(sessions, type=session_column.type)
    )
    listing_positions = pc.index_in(
        listing_column, value_set=pa.array(listings, type=listing_column.type)
    )
    if session_positions.null_count or listing_positions.null_count:
        raise FactorEvidenceBoundaryError("factor_research.evidence_axis_mismatch")
    return cast(
        IntArray,
        np.asarray(session_positions.to_numpy(), dtype=np.intp) * len(listings)
        + np.asarray(listing_positions.to_numpy(), dtype=np.intp),
    )


def _float_values(column: pa.ChunkedArray) -> FloatArray:
    return cast(
        FloatArray,
        np.asarray(column.combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64),
    )


def _surface(
    *,
    feature_table: pa.Table,
    target_surface: FactorTargetSurface,
    plan: FactorWalkForwardPlan,
    factor_ids: tuple[str, ...],
) -> _EvidenceSurface:
    required = {"session_date", "listing_id", *factor_ids}
    missing = sorted(required - set(feature_table.schema.names))
    if missing:
        raise FactorEvidenceBoundaryError(
            f"factor_research.evidence_feature_column_missing:{','.join(missing)}"
        )
    if factor_ids != tuple(sorted(set(factor_ids))):
        raise FactorEvidenceBoundaryError("factor_research.evidence_factor_axis_invalid")
    sessions = tuple(
        sorted(
            {
                session
                for window in plan.formal_split.windows
                for session in (*window.train_sessions, *window.validation_sessions)
            }
        )
    )
    listings = tuple(
        sorted(str(value) for value in pc.unique(target_surface.table["listing_id"]).to_pylist())
    )
    row_count = len(sessions) * len(listings)
    factor_count = len(factor_ids)
    feature_values: FloatArray = np.full((row_count, factor_count), np.nan, dtype=np.float64)
    feature_present: BoolArray = np.zeros(row_count, dtype=np.bool_)
    ordered_features = feature_table.take(
        pc.sort_indices(
            feature_table,
            sort_keys=[("session_date", "ascending"), ("listing_id", "ascending")],
        )
    ).combine_chunks()
    feature_positions = _positions(
        sessions=sessions,
        listings=listings,
        session_column=ordered_features["session_date"],
        listing_column=ordered_features["listing_id"],
    )
    if np.unique(feature_positions).size != feature_positions.size:
        raise FactorEvidenceBoundaryError("factor_research.evidence_duplicate_feature_row")
    for factor_index, factor_id in enumerate(factor_ids):
        feature_values[feature_positions, factor_index] = _float_values(ordered_features[factor_id])
    feature_present[feature_positions] = True
    observed_sessions = tuple(sorted(pc.unique(ordered_features["session_date"]).to_pylist()))
    if observed_sessions != sessions:
        raise FactorEvidenceBoundaryError("factor_research.evidence_feature_sessions_incomplete")

    fit_targets: FloatArray = np.full(row_count, np.nan, dtype=np.float64)
    simple_returns: FloatArray = np.full(row_count, np.nan, dtype=np.float64)
    target_table = target_surface.table
    target_positions = _positions(
        sessions=sessions,
        listings=listings,
        session_column=target_table["formation_session"],
        listing_column=target_table["listing_id"],
    )
    if np.unique(target_positions).size != target_positions.size:
        raise FactorEvidenceBoundaryError("factor_research.evidence_duplicate_target_row")
    fit_targets[target_positions] = _float_values(target_table["fit_target"])
    simple_returns[target_positions] = _float_values(target_table["simple_economic_return"])
    feature_values.setflags(write=False)
    feature_present.setflags(write=False)
    fit_targets.setflags(write=False)
    simple_returns.setflags(write=False)
    return _EvidenceSurface(
        sessions=sessions,
        listings=listings,
        factors=factor_ids,
        features=feature_values.reshape(len(sessions), len(listings), factor_count),
        fit_targets=fit_targets.reshape(len(sessions), len(listings)),
        simple_returns=simple_returns.reshape(len(sessions), len(listings)),
        members=feature_present.reshape(len(sessions), len(listings)),
    )


class _SessionRanks:
    """The rank material of one session, computed once for every factor.

    Every statistic below ranks a factor's scores over the rows where the
    factor and its target are both finite. The session's columns are sorted
    once (``stable_column_orders``) and every factor's ranks over its rows
    are read off that order (``average_ranks_over_rows``), each handed on as
    the same contiguous operand the one-factor path built. The target is
    ranked once per row mask. Oriented validation ranks are the exact mirror
    ``(count + 1) - rank`` of the unoriented ones: the tie groups average the
    same positions from the other end, and the values are exact half-integers.
    """

    __slots__ = ("_finite", "_keys", "_ranks", "_target_by_mask")

    def __init__(self, block: FloatArray, fit_targets: FloatArray) -> None:
        self._finite = np.isfinite(block) & np.isfinite(fit_targets)[:, None]
        groups: dict[bytes, list[int]] = {}
        packed = np.packbits(self._finite, axis=0)
        for column in range(block.shape[1]):
            groups.setdefault(packed[:, column].tobytes(), []).append(column)
        self._keys = {member: key for key, members in groups.items() for member in members}
        self._ranks: dict[int, FloatArray] = {}
        self._target_by_mask: dict[bytes, tuple[FloatArray, float]] = {}
        orders = stable_column_orders(block)
        for key, members in groups.items():
            mask = self._finite[:, members[0]]
            if not mask.any():
                continue
            ranked = average_ranks_over_rows(block, orders, mask, members)
            for position, member in enumerate(members):
                self._ranks[member] = ranked[position]
            self._target_by_mask[key] = centered_ranks(average_ranks_vectorized(fit_targets[mask]))

    def mask(self, factor: int) -> BoolArray:
        return cast(BoolArray, self._finite[:, factor])

    def count(self, factor: int) -> int:
        return int(self._finite[:, factor].sum())

    def ranks(self, factor: int, orientation: int | None) -> FloatArray:
        ranks = self._ranks[factor]
        if orientation == -1:
            ranks = (ranks.size + 1) - ranks
        return ranks

    def rank_ic(self, factor: int, orientation: int | None, minimum: int) -> float | None:
        if self.count(factor) < minimum:
            return None
        target_centered, target_norm = self._target_by_mask[self._keys[factor]]
        scores_centered, scores_norm = centered_ranks(self.ranks(factor, orientation))
        return correlation_from_centered_ranks(
            scores_centered, scores_norm, target_centered, target_norm
        )


def _fold_orientation(training_ics: tuple[float, ...]) -> Literal[-1, 1] | None:
    training_mean = float(np.mean(training_ics)) if training_ics else None
    if training_mean is not None and math.isfinite(training_mean):
        return 1 if training_mean >= 0.0 else -1
    return None


@dataclass(slots=True)
class _FoldAccumulator:
    """One fold's per-factor validation series while its sessions stream by."""

    window: object
    train_indices: tuple[int, ...]
    orientations: tuple[Literal[-1, 1] | None, ...]
    orientation_row: FloatArray
    training_ics: tuple[tuple[float, ...], ...]
    validation_ics: list[list[float]]
    spreads: list[list[float]]
    coverages: list[list[float]]
    turnovers: list[list[float]]
    previous_percentiles: tuple[FloatArray, BoolArray] | None = None


def _compute_factors(
    *,
    surface: _EvidenceSurface,
    plan: FactorWalkForwardPlan,
    policy: FactorEvidencePolicy,
) -> tuple[_FactorComputation, ...]:
    """Every factor's fold evidence, in one pass over the sessions.

    The statistics are the one-factor ones -- a training rank IC per training
    session, then per validation session the oriented rank IC, the decile
    spread on simple returns, the pair coverage and the rank turnover against
    the previous validation session of the fold -- computed for all factors of
    a session together, so each session is ranked once whether it trains one
    fold, validates another, or both. A fold's orientation is fixed from its
    training ICs when its first validation session arrives (every training
    session of a rolling fold precedes its validation window; one that did
    not is ranked on demand), and the series are assembled per factor in the
    fold order and session order the one-factor loop produced.
    """
    session_index = {value: index for index, value in enumerate(surface.sessions)}
    factor_count = len(surface.factors)
    minimum = policy.minimum_cross_section_observations
    fold_indices = tuple(
        (
            window,
            tuple(session_index[value] for value in window.train_sessions),
            tuple(session_index[value] for value in window.validation_sessions),
        )
        for window in plan.formal_split.windows
    )
    training_sessions = {index for _, train, _ in fold_indices for index in train}
    validation_fold: dict[int, int] = {}
    for fold_position, (_window, _train, validation) in enumerate(fold_indices):
        for index in validation:
            validation_fold[index] = fold_position
    training_ic: dict[int, list[float | None]] = {}

    def training_ics_of(index: int) -> list[float | None]:
        if index not in training_ic:
            session = _SessionRanks(surface.features[index], surface.fit_targets[index])
            training_ic[index] = [
                session.rank_ic(factor, None, minimum) for factor in range(factor_count)
            ]
        return training_ic[index]

    accumulators: dict[int, _FoldAccumulator] = {}
    for index in range(len(surface.sessions)):
        fold: int | None = validation_fold.get(index)
        if fold is None and index not in training_sessions:
            continue
        scores = surface.features[index]
        session = _SessionRanks(scores, surface.fit_targets[index])
        if index in training_sessions:
            training_ic[index] = [
                session.rank_ic(factor, None, minimum) for factor in range(factor_count)
            ]
        if fold is None:
            continue
        accumulator = accumulators.get(fold)
        if accumulator is None:
            window, train_indices, _validation = fold_indices[fold]
            training_ics = tuple(
                tuple(
                    value
                    for train_index in train_indices
                    if (value := training_ics_of(train_index)[factor]) is not None
                )
                for factor in range(factor_count)
            )
            orientations = tuple(_fold_orientation(values) for values in training_ics)
            accumulator = accumulators[fold] = _FoldAccumulator(
                window=window,
                train_indices=train_indices,
                orientations=orientations,
                orientation_row=np.asarray(
                    [1.0 if value is None else float(value) for value in orientations],
                    dtype=np.float64,
                ),
                training_ics=training_ics,
                validation_ics=[[] for _ in range(factor_count)],
                spreads=[[] for _ in range(factor_count)],
                coverages=[[] for _ in range(factor_count)],
                turnovers=[[] for _ in range(factor_count)],
            )
        simple_return = surface.simple_returns[index]
        population = int(surface.members[index].sum())
        if not population:
            raise FactorEvidenceBoundaryError("factor_research.evidence_member_population_empty")
        finite_simple = np.isfinite(simple_return)
        oriented = scores * accumulator.orientation_row
        percentiles, percentile_finite = ordinal_percentile_columns(oriented)
        for factor in range(factor_count):
            orientation = accumulator.orientations[factor]
            mask = session.mask(factor)
            accumulator.coverages[factor].append(float(mask.sum() / population))
            rank_ic = session.rank_ic(factor, orientation, minimum)
            if rank_ic is not None:
                accumulator.validation_ics[factor].append(rank_ic)
            spread_mask = np.isfinite(oriented[:, factor]) & finite_simple
            if int(spread_mask.sum()) >= minimum:
                ranks = (
                    session.ranks(factor, orientation)
                    if np.array_equal(spread_mask, mask)
                    else average_ranks_vectorized(oriented[spread_mask, factor])
                )
                spread = decile_spread_from_ranks(ranks, simple_return[spread_mask])
                if spread is not None:
                    accumulator.spreads[factor].append(spread)
            if accumulator.previous_percentiles is not None:
                left, left_finite = accumulator.previous_percentiles
                common = left_finite[:, factor] & percentile_finite[:, factor]
                if int(common.sum()) >= 2:
                    accumulator.turnovers[factor].append(
                        float(np.mean(np.abs(percentiles[common, factor] - left[common, factor])))
                    )
        accumulator.previous_percentiles = (percentiles, percentile_finite)

    folds: list[list[FactorFoldEvidence]] = [[] for _ in range(factor_count)]
    all_ics: list[list[float]] = [[] for _ in range(factor_count)]
    all_spreads: list[list[float]] = [[] for _ in range(factor_count)]
    all_coverages: list[list[float]] = [[] for _ in range(factor_count)]
    all_turnovers: list[list[float]] = [[] for _ in range(factor_count)]
    for fold_position, (window, _train, _validation) in enumerate(fold_indices):
        accumulator = accumulators[fold_position]
        for factor in range(factor_count):
            factor_training_ics = accumulator.training_ics[factor]
            validation_ics = accumulator.validation_ics[factor]
            spreads = accumulator.spreads[factor]
            coverages = accumulator.coverages[factor]
            turnovers = accumulator.turnovers[factor]
            folds[factor].append(
                seal_contract(
                    FactorFoldEvidence,
                    "fold_hash",
                    factor_id=surface.factors[factor],
                    fold_index=window.fold_index,
                    orientation=accumulator.orientations[factor],
                    training_period_count=len(factor_training_ics),
                    validation_period_count=len(validation_ics),
                    training_mean_rank_ic=(
                        float(np.mean(factor_training_ics)) if factor_training_ics else None
                    ),
                    validation_oriented_mean_rank_ic=(
                        float(np.mean(validation_ics)) if validation_ics else None
                    ),
                    validation_oriented_mean_simple_spread=(
                        float(np.mean(spreads)) if spreads else None
                    ),
                    validation_pair_coverage_mean=(float(np.mean(coverages)) if coverages else 0.0),
                    validation_rank_turnover_mean=(
                        float(np.mean(turnovers)) if turnovers else None
                    ),
                )
            )
            all_ics[factor].extend(validation_ics)
            all_spreads[factor].extend(spreads)
            all_coverages[factor].extend(coverages)
            all_turnovers[factor].extend(turnovers)
    return tuple(
        _FactorComputation(
            folds=tuple(folds[factor]),
            validation_ics=tuple(all_ics[factor]),
            validation_spreads=tuple(all_spreads[factor]),
            validation_coverages=tuple(all_coverages[factor]),
            validation_turnovers=tuple(all_turnovers[factor]),
        )
        for factor in range(factor_count)
    )


def _classification(
    *,
    computation: _FactorComputation,
    mean_ic: float | None,
    mean_spread: float | None,
    q_value: float,
    coverage: float,
    policy: FactorEvidencePolicy,
) -> tuple[FactorEvidenceClassification, tuple[str, ...]]:
    reasons: list[str] = []
    if any(fold.orientation is None for fold in computation.folds):
        reasons.append("TRAINING_ORIENTATION_UNAVAILABLE")
    if not computation.validation_ics:
        reasons.append("VALIDATION_RANK_IC_UNAVAILABLE")
    if not computation.validation_spreads:
        reasons.append("VALIDATION_SIMPLE_SPREAD_UNAVAILABLE")
    if coverage < policy.minimum_pair_coverage:
        reasons.append("VALIDATION_PAIR_COVERAGE_INSUFFICIENT")
    if reasons:
        return FactorEvidenceClassification.INSUFFICIENT_EVIDENCE, tuple(reasons)
    assert mean_ic is not None and mean_spread is not None
    if mean_ic > 0.0 and mean_spread > 0.0:
        reason = (
            "DIRECTIONALLY_POSITIVE_BY_CONFIRMED"
            if q_value <= policy.fdr_alpha
            else "DIRECTIONALLY_POSITIVE_NOT_BY_CONFIRMED"
        )
        return FactorEvidenceClassification.POSITIVE_OOS_EVIDENCE, (reason,)
    if mean_ic <= 0.0 and mean_spread <= 0.0:
        if mean_ic < 0.0 and mean_spread < 0.0 and q_value <= policy.fdr_alpha:
            return FactorEvidenceClassification.NEGATIVE_OOS_EVIDENCE, (
                "DIRECTIONALLY_NEGATIVE_BY_CONFIRMED",
            )
        return FactorEvidenceClassification.NO_DETECTABLE_EFFECT, (
            "NONPOSITIVE_DIRECTION_NOT_BY_CONFIRMED",
        )
    return FactorEvidenceClassification.MIXED_OOS_EVIDENCE, ("MIXED_OOS_DIRECTION",)


def compute_factor_oos_evidence(
    *,
    feature_table: pa.Table,
    feature_panel_snapshot_hash: str,
    feature_panel_manifest_ref: str,
    target_surface: FactorTargetSurface,
    walk_forward_plan: FactorWalkForwardPlan,
    factor_ids: tuple[str, ...],
    policy: FactorEvidencePolicy,
) -> FactorOosEvidenceReport:
    """Compute complete validation-only evidence and one 53-item BY family."""
    policy = FactorEvidencePolicy.model_validate(policy)
    plan = FactorWalkForwardPlan.model_validate(walk_forward_plan)
    evidence_surface = _surface(
        feature_table=feature_table,
        target_surface=target_surface,
        plan=plan,
        factor_ids=factor_ids,
    )
    computations = _compute_factors(surface=evidence_surface, plan=plan, policy=policy)
    raw_p_values: list[tuple[str, float]] = []
    ic_tests = []
    spread_tests = []
    for factor_id, computation in zip(factor_ids, computations, strict=True):
        ic_values: FloatArray = np.asarray(computation.validation_ics, dtype=np.float64)
        spread_values: FloatArray = np.asarray(computation.validation_spreads, dtype=np.float64)
        ic_test = newey_west_mean_test(ic_values) if ic_values.size else None
        spread_test = newey_west_mean_test(spread_values) if spread_values.size else None
        ic_tests.append(ic_test)
        spread_tests.append(spread_test)
        raw_p_values.append((factor_id, ic_test.p_value if ic_test is not None else 1.0))
    q_values = benjamini_yekutieli(tuple(raw_p_values))
    items: list[FactorOosEvidenceItem] = []
    for factor_id, computation, ic_test, spread_test in zip(
        factor_ids, computations, ic_tests, spread_tests, strict=True
    ):
        mean_ic = ic_test.mean if ic_test is not None else None
        mean_spread = spread_test.mean if spread_test is not None else None
        coverage = (
            float(np.mean(computation.validation_coverages))
            if computation.validation_coverages
            else 0.0
        )
        classification, reason_codes = _classification(
            computation=computation,
            mean_ic=mean_ic,
            mean_spread=mean_spread,
            q_value=q_values[factor_id],
            coverage=coverage,
            policy=policy,
        )
        ic_values = np.asarray(computation.validation_ics, dtype=np.float64)
        volatility = float(np.std(ic_values, ddof=1)) if ic_values.size > 1 else None
        hit_rate = float(np.mean(ic_values > 0.0)) if ic_values.size else None
        items.append(
            seal_contract(
                FactorOosEvidenceItem,
                "evidence_hash",
                factor_id=factor_id,
                fold_evidence=computation.folds,
                validation_period_count=len(computation.validation_ics),
                mean_oriented_rank_ic=mean_ic,
                rank_ic_volatility=volatility,
                rank_ic_ir=ic_test.icir if ic_test is not None else None,
                rank_ic_positive_hit_rate=hit_rate,
                rank_ic_hac_lag=ic_test.lag if ic_test is not None else 0,
                rank_ic_hac_t_stat=ic_test.t_stat if ic_test is not None else None,
                rank_ic_raw_p_value=ic_test.p_value if ic_test is not None else 1.0,
                rank_ic_by_q_value=q_values[factor_id],
                mean_oriented_simple_spread=mean_spread,
                simple_spread_hac_lag=spread_test.lag if spread_test is not None else 0,
                simple_spread_hac_t_stat=spread_test.t_stat if spread_test is not None else None,
                simple_spread_raw_p_value=spread_test.p_value if spread_test is not None else 1.0,
                validation_pair_coverage_mean=coverage,
                validation_rank_turnover_mean=(
                    float(np.mean(computation.validation_turnovers))
                    if computation.validation_turnovers
                    else None
                ),
                classification=classification,
                reason_codes=reason_codes,
            )
        )
    counts = {
        classification: sum(item.classification is classification for item in items)
        for classification in FactorEvidenceClassification
    }
    return seal_contract(
        FactorOosEvidenceReport,
        "report_hash",
        feature_panel_snapshot_hash=feature_panel_snapshot_hash,
        feature_panel_manifest_ref=feature_panel_manifest_ref,
        target_surface_hash=target_surface.manifest.surface_hash,
        walk_forward_plan_hash=plan.plan_hash,
        evidence_policy_hash=policy.policy_hash,
        hypothesis_count=len(factor_ids),
        factor_ids=factor_ids,
        items=tuple(items),
        classification_counts=counts,
    )


__all__ = [
    "FactorEvidenceBoundaryError",
    "FactorEvidenceClassification",
    "FactorEvidencePolicy",
    "FactorFoldEvidence",
    "FactorOosEvidenceItem",
    "FactorOosEvidenceReport",
    "build_factor_evidence_policy",
    "compute_factor_oos_evidence",
]
