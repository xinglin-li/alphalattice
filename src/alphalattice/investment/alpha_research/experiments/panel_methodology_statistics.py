"""Installed long-only and paired statistics for Panel research methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from math import ceil, sqrt
from types import MappingProxyType
from typing import Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
    session_grouped_rank_ic,
)
from alphalattice.capabilities.portfolio_backtesting.metrics import (
    annualized_volatility,
    break_even_cost_bps,
    maximum_drawdown,
    sortino_ratio,
)
from alphalattice.investment.alpha_research.scores.score_filters import (
    RAW_SCORE_INPUT_MODE,
    RAW_SCORE_TEMPORAL_FILTER_METHOD_ID,
    AlphaScoreFilterSpec,
    apply_alpha_score_filter,
)
from alphalattice.investment.alpha_research.scores.temporal_aggregation import (
    row_axis_hash,
    score_value_hash,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.validation.screening_statistics import gross_decile_spread

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type Disposition = Literal[
    "SUPERIOR_PAIRED_EVIDENCE",
    "INFERIOR_PAIRED_EVIDENCE",
    "INCONCLUSIVE_PAIRED_EVIDENCE",
]

_HASH = r"^[0-9a-f]{64}$"
_BOOTSTRAP_SEED = 1729
_BOOTSTRAP_RESAMPLES = 2_000


class PanelStatisticError(ValueError):
    """Stable refusal for misaligned or nonfinite scientific evidence."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class LongOnlySessionEvidence(_Contract):
    candidate_id: str
    span_sessions: int = Field(ge=1)
    sessions: tuple[date, ...] = Field(min_length=1)
    gross_active_returns: tuple[float, ...] = Field(min_length=1)
    net_active_returns_5bps: tuple[float, ...] = Field(min_length=1)
    one_way_turnovers: tuple[float, ...] = Field(min_length=1)
    session_rank_ics: tuple[float | None, ...] = Field(min_length=1)
    session_long_short_spreads: tuple[float | None, ...] = Field(min_length=1)
    insufficient_rank_session_count: int = Field(ge=0)
    constant_rank_session_count: int = Field(ge=0)
    mean_gross_active_return: float
    mean_net_active_return_5bps: float
    mean_turnover: float = Field(ge=0.0)
    mean_rank_ic: float | None
    early_half_net_5bps: float
    late_half_net_5bps: float
    evidence_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, evidence_hash="0" * 64)
        return cls(
            **values,
            evidence_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"evidence_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_axis(self) -> Self:
        length = len(self.sessions)
        if (
            any(
                len(value) != length
                for value in (
                    self.gross_active_returns,
                    self.net_active_returns_5bps,
                    self.one_way_turnovers,
                    self.session_rank_ics,
                    self.session_long_short_spreads,
                )
            )
            or self.sessions != tuple(sorted(set(self.sessions)))
            or self.evidence_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"evidence_hash"}))
        ):
            raise PanelStatisticError("alpha_research.long_only_evidence_invalid")
        return self


class CausalCandidateSelection(_Contract):
    fold_index: int = Field(ge=0)
    candidate_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    inner_best_evidence_hash: str = Field(pattern=_HASH)
    statistically_indistinguishable_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    selected_evidence_hash: str = Field(pattern=_HASH)
    selected_candidate_id: str
    selected_span_sessions: int = Field(ge=1)
    tie_break_order: tuple[str, ...] = (
        "LOWER_TURNOVER",
        "FEWER_TRANSFORMS",
        "SIMPLER_MODEL",
        "LONGER_SPAN",
        "CANONICAL_IDENTITY",
    )
    dependence_block_length: int = Field(ge=1)
    selection_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, selection_hash="0" * 64)
        return cls(
            **values,
            selection_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"selection_hash"}))
            ),
        )


class CausalModelSpecSelection(_Contract):
    """Stage-one model/view selection at the unaggregated score surface."""

    fold_index: int = Field(ge=0)
    selection_method_id: Literal["SESSION_RANK_IC_PAIRED"] = "SESSION_RANK_IC_PAIRED"
    fixed_span_sessions: Literal[1] = 1
    candidate_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    common_rank_ic_sessions: tuple[date, ...] = Field(min_length=2)
    inner_best_evidence_hash: str = Field(pattern=_HASH)
    statistically_indistinguishable_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    selected_evidence_hash: str = Field(pattern=_HASH)
    selected_candidate_id: str
    tie_break_order: tuple[str, ...] = (
        "FEWER_TRANSFORMS",
        "SIMPLER_MODEL",
        "CANONICAL_IDENTITY",
    )
    dependence_block_length: int = Field(ge=1)
    selection_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, selection_hash="0" * 64)
        return cls(
            **values,
            selection_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"selection_hash"}))
            ),
        )


class CausalScoreAggregationSelection(_Contract):
    """Stage-two span selection for one already-selected model specification."""

    fold_index: int = Field(ge=0)
    selection_method_id: Literal["CAUSAL_NET_ACTIVE_RETURN_5BPS"] = "CAUSAL_NET_ACTIVE_RETURN_5BPS"
    selected_model_candidate_id: str
    candidate_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    inner_best_evidence_hash: str = Field(pattern=_HASH)
    statistically_indistinguishable_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    selected_evidence_hash: str = Field(pattern=_HASH)
    selected_span_sessions: int = Field(ge=1)
    fold_zero_unaggregated_control: bool
    tie_break_order: tuple[str, ...] = (
        "LOWER_TURNOVER",
        "LONGER_SPAN",
        "CANONICAL_IDENTITY",
    )
    dependence_block_length: int = Field(ge=1)
    selection_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, selection_hash="0" * 64)
        return cls(
            **values,
            selection_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"selection_hash"}))
            ),
        )


@dataclass(frozen=True, slots=True)
class CandidateComplexity:
    transform_count: int
    model_order: int


class PanelScientificSessionEvidence(_Contract):
    """Raw span-one Alpha evidence with no Portfolio/cost fields."""

    kind: Literal["PanelScientificSessionEvidence"] = "PanelScientificSessionEvidence"
    fold_index: int = Field(ge=0)
    spec_id: str
    view_id: str
    model_family_id: str
    recipe_hash: str = Field(pattern=_HASH)
    sessions: tuple[date, ...] = Field(min_length=2)
    session_rank_ics: tuple[float | None, ...] = Field(min_length=2)
    session_gross_long_short_spreads: tuple[float | None, ...] = Field(min_length=2)
    insufficient_session_count: int = Field(ge=0)
    constant_session_count: int = Field(ge=0)
    mean_rank_ic: float | None
    rank_ic_sample_std: float | None = Field(default=None, ge=0.0)
    rank_ic_information_ratio: float | None
    mean_gross_long_short_spread: float | None
    evidence_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, evidence_hash="0" * 64)
        return cls(
            **values,
            evidence_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"evidence_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_axis(self) -> Self:
        if (
            len(self.sessions) != len(self.session_rank_ics)
            or len(self.sessions) != len(self.session_gross_long_short_spreads)
            or self.sessions != tuple(sorted(set(self.sessions)))
            or self.evidence_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"evidence_hash"}))
        ):
            raise PanelStatisticError("alpha_research.panel_scientific_evidence_invalid")
        return self


class PanelModelScientificSelection(_Contract):
    """Scientific conclusion and operational pipeline choice are distinct."""

    kind: Literal["PanelModelScientificSelection"] = "PanelModelScientificSelection"
    fold_index: int = Field(ge=0)
    selection_method_id: Literal["SESSION_RANK_IC_PAIRED"] = "SESSION_RANK_IC_PAIRED"
    fixed_span_sessions: Literal[1] = 1
    candidate_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    common_rank_ic_sessions: tuple[date, ...] = Field(min_length=2)
    inner_best_evidence_hash: str = Field(pattern=_HASH)
    inner_best_spec_id: str
    statistically_indistinguishable_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    scientific_disposition: Literal["SUPERIOR_PAIRED_EVIDENCE", "INCONCLUSIVE_PAIRED_EVIDENCE"]
    scientific_winner_spec_id: str | None
    operational_fallback_evidence_hash: str = Field(pattern=_HASH)
    operational_fallback_spec_id: str
    operational_fallback_reason: Literal[
        "SCIENTIFIC_WINNER", "FEWER_TRANSFORMS", "SIMPLER_MODEL", "CANONICAL_IDENTITY"
    ]
    tie_break_order: tuple[str, ...] = (
        "FEWER_TRANSFORMS",
        "SIMPLER_MODEL",
        "CANONICAL_IDENTITY",
    )
    dependence_block_length: int = Field(ge=1)
    selection_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, selection_hash="0" * 64)
        return cls(
            **values,
            selection_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"selection_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_conclusion(self) -> Self:
        winner_present = self.scientific_winner_spec_id is not None
        if winner_present != (
            self.scientific_disposition == "SUPERIOR_PAIRED_EVIDENCE"
        ) or self.selection_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"selection_hash"})
        ):
            raise PanelStatisticError("alpha_research.model_scientific_selection_invalid")
        return self


class PanelScorePersistencePoint(_Contract):
    lag_sessions: int = Field(ge=0)
    forward_horizon_sessions: int = Field(ge=1)
    score_autocorrelation: float | None
    score_autocorrelation_session_count: int = Field(ge=0)
    stale_score_predictive_ic: float | None
    stale_score_predictive_session_count: int = Field(ge=0)
    forward_cumulative_return_ic: float | None
    forward_cumulative_return_session_count: int = Field(ge=0)


class PanelScoreFilterEvidence(_Contract):
    """Quality and persistence evidence for one declared score filter."""

    kind: Literal["PanelScoreFilterEvidence"] = "PanelScoreFilterEvidence"
    upstream_raw_score_artifact_hashes: tuple[str, ...] = ()
    model_spec_ids_by_fold: tuple[str, ...] = Field(min_length=1)
    filter_spec: AlphaScoreFilterSpec
    row_count: int = Field(ge=1)
    row_axis_hash: str = Field(pattern=_HASH)
    score_value_hash: str = Field(pattern=_HASH)
    sessions: tuple[date, ...] = Field(min_length=2)
    fold_mean_rank_ics: tuple[float | None, ...] = Field(min_length=1)
    session_rank_ics: tuple[float | None, ...] = Field(min_length=2)
    session_gross_long_short_spreads: tuple[float | None, ...] = Field(min_length=2)
    insufficient_session_count: int = Field(ge=0)
    constant_session_count: int = Field(ge=0)
    mean_rank_ic: float | None
    rank_ic_sample_std: float | None = Field(default=None, ge=0.0)
    rank_ic_information_ratio: float | None
    mean_gross_long_short_spread: float | None
    persistence: tuple[PanelScorePersistencePoint, ...] = Field(min_length=1)
    evidence_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, evidence_hash="0" * 64)
        return cls(
            **values,
            evidence_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"evidence_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_axis(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"evidence_hash"})
        admitted_hashes = {str(canonical_hash(payload))}
        if not self.upstream_raw_score_artifact_hashes:
            legacy = dict(payload)
            legacy.pop("upstream_raw_score_artifact_hashes", None)
            admitted_hashes.add(str(canonical_hash(legacy)))
        if (
            len(self.sessions) != len(self.session_rank_ics)
            or len(self.sessions) != len(self.session_gross_long_short_spreads)
            or self.sessions != tuple(sorted(set(self.sessions)))
            or tuple(value.lag_sessions for value in self.persistence)
            != tuple(sorted({value.lag_sessions for value in self.persistence}))
            or (
                bool(self.upstream_raw_score_artifact_hashes)
                and len(self.upstream_raw_score_artifact_hashes) != len(self.model_spec_ids_by_fold)
            )
            or any(len(value) != 64 for value in self.upstream_raw_score_artifact_hashes)
            or self.evidence_hash not in admitted_hashes
        ):
            raise PanelStatisticError("alpha_research.score_filter_evidence_invalid")
        return self


class PanelScoreFilterCandidateCriterion(_Contract):
    spec_id: str
    evidence_hash: str = Field(pattern=_HASH)
    matured_prior_session_count: int = Field(ge=0)
    mean_rank_ic: float | None
    best_minus_candidate_interval: tuple[float, float] | None


class PanelScoreFilterSelection(_Contract):
    kind: Literal["PanelScoreFilterSelection"] = "PanelScoreFilterSelection"
    fold_index: int = Field(ge=0)
    selection_method_id: Literal["MATURED_PRIOR_FOLD_SESSION_RANK_IC_PAIRED"] = (
        "MATURED_PRIOR_FOLD_SESSION_RANK_IC_PAIRED"
    )
    current_fold_first_formation_session: date
    candidate_criteria: tuple[PanelScoreFilterCandidateCriterion, ...] = Field(min_length=1)
    common_matured_prior_sessions: tuple[date, ...]
    scientific_disposition: Literal["SUPERIOR_PAIRED_EVIDENCE", "INCONCLUSIVE_PAIRED_EVIDENCE"]
    scientific_winner_spec_id: str | None
    operational_fallback_spec_id: str
    operational_fallback_reason: Literal[
        "SCIENTIFIC_WINNER", "DECLARED_RAW_CONTROL", "SMALLEST_SPAN", "SIMPLER_FILTER"
    ]
    statistically_indistinguishable_spec_ids: tuple[str, ...] = Field(min_length=1)
    tie_break_order: tuple[str, ...] = (
        "SMALLEST_SPAN",
        "SIMPLER_FILTER",
        "CANONICAL_IDENTITY",
    )
    dependence_block_length: int = Field(ge=1)
    selection_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, selection_hash="0" * 64)
        return cls(
            **values,
            selection_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"selection_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_conclusion(self) -> Self:
        if (self.scientific_winner_spec_id is not None) != (
            self.scientific_disposition == "SUPERIOR_PAIRED_EVIDENCE"
        ) or self.selection_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"selection_hash"})
        ):
            raise PanelStatisticError("alpha_research.score_filter_selection_invalid")
        return self


@dataclass(frozen=True, slots=True)
class PanelScoreFilterEvaluationResult:
    evidence: tuple[PanelScoreFilterEvidence, ...]
    selections: tuple[PanelScoreFilterSelection, ...]
    filtered_scores_by_spec_id: Mapping[str, FloatArray]
    selected_scores: FloatArray


class AlphaEconomicPathMetrics(_Contract):
    """One raw-simple-return path under the fixed Alpha diagnostic book."""

    mean_return: float
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float = Field(ge=0.0)
    sharpe: float
    sortino: float
    maximum_drawdown: float = Field(ge=0.0, le=1.0)


class AlphaActivePathMetrics(_Contract):
    """Same-session eligible-Universe-relative diagnostics."""

    mean_active_return: float
    tracking_error: float = Field(ge=0.0)
    information_ratio: float
    active_sharpe: float
    active_sortino: float


class AlphaEconomicCandidateEvidence(_Contract):
    """Fixed-book economics for one OOF model recipe; never a Portfolio policy."""

    kind: Literal["AlphaEconomicCandidateEvidence"] = "AlphaEconomicCandidateEvidence"
    model_recipe_id: str
    ordered_score_surface_hashes: tuple[str, ...] = Field(min_length=1)
    sessions: tuple[date, ...] = Field(min_length=2)
    book_construction_id: Literal[
        "SESSION_DECILE_EQUAL_WEIGHT_LONG_ONLY_AND_DOLLAR_NEUTRAL_LONG_SHORT"
    ] = "SESSION_DECILE_EQUAL_WEIGHT_LONG_ONLY_AND_DOLLAR_NEUTRAL_LONG_SHORT"
    decile_rule_id: Literal["CEIL_N_OVER_10_SCORE_DESC_LISTING_ASC_TIE_BREAK"] = (
        "CEIL_N_OVER_10_SCORE_DESC_LISTING_ASC_TIE_BREAK"
    )
    long_only_returns: tuple[float, ...] = Field(min_length=2)
    eligible_universe_returns: tuple[float, ...] = Field(min_length=2)
    active_returns: tuple[float, ...] = Field(min_length=2)
    long_short_returns: tuple[float, ...] = Field(min_length=2)
    top_decile_returns: tuple[float, ...] = Field(min_length=2)
    bottom_decile_returns: tuple[float, ...] = Field(min_length=2)
    short_contributions: tuple[float, ...] = Field(min_length=2)
    session_rank_ics: tuple[float, ...] = Field(min_length=2)
    long_only_metrics: AlphaEconomicPathMetrics
    active_metrics: AlphaActivePathMetrics
    long_short_metrics: AlphaEconomicPathMetrics
    mean_top_decile_return: float
    mean_top_decile_active_contribution: float
    mean_bottom_decile_return: float
    mean_short_contribution: float
    leg_contribution_share_policy: Literal["ABSOLUTE_MEAN_LEG_CONTRIBUTIONS"] = (
        "ABSOLUTE_MEAN_LEG_CONTRIBUTIONS"
    )
    long_contribution_share: float = Field(ge=0.0, le=1.0)
    short_contribution_share: float = Field(ge=0.0, le=1.0)
    mean_session_rank_ic: float
    rank_ic_information_ratio: float
    tail_rank_ic: float
    score_autocorrelation_lag_1: float | None
    stale_score_ic_decay: tuple[PanelScorePersistencePoint, ...] = Field(min_length=1)
    mean_one_way_turnover_diagnostic: float = Field(ge=0.0)
    break_even_cost_bps_diagnostic: float | None
    evidence_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, evidence_hash="0" * 64)
        return cls(
            **values,
            evidence_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"evidence_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_common_axis(self) -> Self:
        length = len(self.sessions)
        lanes = (
            self.long_only_returns,
            self.eligible_universe_returns,
            self.active_returns,
            self.long_short_returns,
            self.top_decile_returns,
            self.bottom_decile_returns,
            self.short_contributions,
            self.session_rank_ics,
        )
        if (
            any(len(value) != length for value in lanes)
            or self.sessions != tuple(sorted(set(self.sessions)))
            or self.evidence_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"evidence_hash"}))
        ):
            raise PanelStatisticError("alpha_research.economic_candidate_axis_invalid")
        return self


type _AlphaEconomicEndpointId = Literal[
    "MEAN_ACTIVE_RETURN",
    "ACTIVE_SHARPE",
    "ACTIVE_SORTINO",
    "LONG_ONLY_ABSOLUTE_MEAN_RETURN",
    "LONG_ONLY_ABSOLUTE_SHARPE",
    "LONG_ONLY_ABSOLUTE_SORTINO",
    "LONG_ONLY_ACTIVE_MEAN_RETURN",
    "LONG_ONLY_ACTIVE_SHARPE",
    "LONG_ONLY_ACTIVE_SORTINO",
    "LONG_SHORT_MEAN_RETURN",
    "LONG_SHORT_SHARPE",
    "LONG_SHORT_SORTINO",
    "LONG_LEG_MEAN_RETURN",
    "SHORT_LEG_MEAN_CONTRIBUTION",
    "MEAN_SESSION_RANK_IC",
]
type _AlphaEconomicLaneId = Literal[
    "LONG_ONLY_RETURNS",
    "ACTIVE_RETURNS",
    "LONG_SHORT_RETURNS",
    "TOP_DECILE_RETURNS",
    "SHORT_CONTRIBUTIONS",
    "SESSION_RANK_ICS",
]
type _AlphaEconomicStatisticId = Literal["MEAN", "SHARPE", "SORTINO"]

_FIXED_ECONOMIC_ENDPOINT_DECLARATIONS: tuple[
    tuple[_AlphaEconomicEndpointId, _AlphaEconomicLaneId, _AlphaEconomicStatisticId], ...
] = (
    ("LONG_ONLY_ABSOLUTE_MEAN_RETURN", "LONG_ONLY_RETURNS", "MEAN"),
    ("LONG_ONLY_ABSOLUTE_SHARPE", "LONG_ONLY_RETURNS", "SHARPE"),
    ("LONG_ONLY_ABSOLUTE_SORTINO", "LONG_ONLY_RETURNS", "SORTINO"),
    ("LONG_ONLY_ACTIVE_MEAN_RETURN", "ACTIVE_RETURNS", "MEAN"),
    ("LONG_ONLY_ACTIVE_SHARPE", "ACTIVE_RETURNS", "SHARPE"),
    ("LONG_ONLY_ACTIVE_SORTINO", "ACTIVE_RETURNS", "SORTINO"),
    ("LONG_SHORT_MEAN_RETURN", "LONG_SHORT_RETURNS", "MEAN"),
    ("LONG_SHORT_SHARPE", "LONG_SHORT_RETURNS", "SHARPE"),
    ("LONG_SHORT_SORTINO", "LONG_SHORT_RETURNS", "SORTINO"),
    ("LONG_LEG_MEAN_RETURN", "TOP_DECILE_RETURNS", "MEAN"),
    ("SHORT_LEG_MEAN_CONTRIBUTION", "SHORT_CONTRIBUTIONS", "MEAN"),
    ("MEAN_SESSION_RANK_IC", "SESSION_RANK_ICS", "MEAN"),
)
_FIXED_ECONOMIC_RECIPE_IDS = (
    "RELATIVE_CONTROL_RIDGE",
    "SPARSE_SESSION_AMPLITUDE_RIDGE",
    "SPARSE_SESSION_AMPLITUDE_LIGHTGBM",
)
_FIXED_ECONOMIC_COMPARISON_DECLARATIONS = (
    ("FEATURE_LIFT", "SPARSE_SESSION_AMPLITUDE_RIDGE", "RELATIVE_CONTROL_RIDGE"),
    ("MODEL_LIFT", "SPARSE_SESSION_AMPLITUDE_LIGHTGBM", "SPARSE_SESSION_AMPLITUDE_RIDGE"),
    ("END_TO_END_ALPHA_LIFT", "SPARSE_SESSION_AMPLITUDE_LIGHTGBM", "RELATIVE_CONTROL_RIDGE"),
)
# Compatibility-only shape for the admitted qualification artifact that owns the
# recommended fixed-195 numerical child. Active evaluation cannot author this
# completed experiment again.
_HISTORICAL_TRANSFORM_QUALIFICATION_RECIPE_IDS = (
    "RELATIVE_CONTROL_RIDGE",
    "SPARSE_SESSION_AMPLITUDE_RIDGE",
    "SPARSE_SESSION_AMPLITUDE_STOCK_POSITIVE_IDENTITY_RIDGE",
    "SPARSE_SESSION_AMPLITUDE_LIGHTGBM",
    "SPARSE_SESSION_AMPLITUDE_RELATIVE_POSITIVE_IDENTITY_LIGHTGBM",
    "SPARSE_SESSION_AMPLITUDE_NON_NEUTRAL_POSITIVE_IDENTITY_LIGHTGBM",
    "SPARSE_SESSION_AMPLITUDE_STOCK_POSITIVE_IDENTITY_LIGHTGBM",
)
_HISTORICAL_TRANSFORM_QUALIFICATION_COMPARISON_DECLARATIONS = (
    (
        "RELATIVE_STOCK_IDENTITY_LIFT",
        "SPARSE_SESSION_AMPLITUDE_RELATIVE_POSITIVE_IDENTITY_LIGHTGBM",
        "SPARSE_SESSION_AMPLITUDE_LIGHTGBM",
    ),
    (
        "NON_NEUTRAL_STOCK_IDENTITY_LIFT",
        "SPARSE_SESSION_AMPLITUDE_NON_NEUTRAL_POSITIVE_IDENTITY_LIGHTGBM",
        "SPARSE_SESSION_AMPLITUDE_LIGHTGBM",
    ),
    (
        "COMBINED_STOCK_IDENTITY_LIFT",
        "SPARSE_SESSION_AMPLITUDE_STOCK_POSITIVE_IDENTITY_LIGHTGBM",
        "SPARSE_SESSION_AMPLITUDE_LIGHTGBM",
    ),
    (
        "LINEAR_STOCK_IDENTITY_LIFT",
        "SPARSE_SESSION_AMPLITUDE_STOCK_POSITIVE_IDENTITY_RIDGE",
        "SPARSE_SESSION_AMPLITUDE_RIDGE",
    ),
    ("MODEL_LIFT", "SPARSE_SESSION_AMPLITUDE_LIGHTGBM", "SPARSE_SESSION_AMPLITUDE_RIDGE"),
)
_LEGACY_ECONOMIC_ENDPOINT_IDS = frozenset(
    {
        "MEAN_ACTIVE_RETURN",
        "LONG_SHORT_MEAN_RETURN",
        "ACTIVE_SHARPE",
        "ACTIVE_SORTINO",
        "MEAN_SESSION_RANK_IC",
    }
)


def _fixed_economic_relationship(
    model_recipe_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]] | None:
    recipe_set = set(model_recipe_ids)
    if len(model_recipe_ids) == len(_FIXED_ECONOMIC_RECIPE_IDS) and recipe_set == set(
        _FIXED_ECONOMIC_RECIPE_IDS
    ):
        return _FIXED_ECONOMIC_RECIPE_IDS, _FIXED_ECONOMIC_COMPARISON_DECLARATIONS
    return None


def _historical_economic_relationship(
    model_recipe_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]] | None:
    recipe_set = set(model_recipe_ids)
    if len(model_recipe_ids) == len(
        _HISTORICAL_TRANSFORM_QUALIFICATION_RECIPE_IDS
    ) and recipe_set == set(_HISTORICAL_TRANSFORM_QUALIFICATION_RECIPE_IDS):
        return (
            _HISTORICAL_TRANSFORM_QUALIFICATION_RECIPE_IDS,
            _HISTORICAL_TRANSFORM_QUALIFICATION_COMPARISON_DECLARATIONS,
        )
    return None


def fixed_recipe_economic_metric_call_count(model_recipe_ids: tuple[str, ...]) -> int:
    """Price the exact fixed economic relationship evaluated by this owner."""

    relationship = _fixed_economic_relationship(model_recipe_ids)
    if relationship is None:
        return 0
    recipes, comparisons = relationship
    return len(recipes) + len(comparisons) * len(_FIXED_ECONOMIC_ENDPOINT_DECLARATIONS)


class AlphaEconomicPairedEndpoint(_Contract):
    endpoint_id: _AlphaEconomicEndpointId
    lane_id: _AlphaEconomicLaneId | None = None
    statistic_id: _AlphaEconomicStatisticId | None = None
    point_difference: float
    paired_interval_95: tuple[float, float]
    common_session_count: int = Field(ge=2)
    disposition: Disposition

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_interval(self) -> Self:
        lower, upper = self.paired_interval_95
        declaration_by_id = {
            endpoint_id: (lane_id, statistic_id)
            for endpoint_id, lane_id, statistic_id in _FIXED_ECONOMIC_ENDPOINT_DECLARATIONS
        }
        mapping = (self.lane_id, self.statistic_id)
        mapping_is_historical = mapping == (None, None)
        expected: Disposition = (
            "SUPERIOR_PAIRED_EVIDENCE"
            if lower > 0.0
            else ("INFERIOR_PAIRED_EVIDENCE" if upper < 0.0 else "INCONCLUSIVE_PAIRED_EVIDENCE")
        )
        if (
            not np.isfinite((self.point_difference, lower, upper)).all()
            or lower > upper
            or self.disposition != expected
            or (mapping_is_historical and self.endpoint_id not in _LEGACY_ECONOMIC_ENDPOINT_IDS)
            or (not mapping_is_historical and declaration_by_id.get(self.endpoint_id) != mapping)
        ):
            raise PanelStatisticError("alpha_research.economic_paired_interval_invalid")
        return self


class AlphaEconomicPairedComparison(_Contract):
    comparison_id: Literal[
        "FEATURE_LIFT",
        "MODEL_LIFT",
        "END_TO_END_ALPHA_LIFT",
        "RELATIVE_STOCK_IDENTITY_LIFT",
        "NON_NEUTRAL_STOCK_IDENTITY_LIFT",
        "COMBINED_STOCK_IDENTITY_LIFT",
        "LINEAR_STOCK_IDENTITY_LIFT",
    ]
    treatment_recipe_id: str
    control_recipe_id: str
    endpoints: tuple[AlphaEconomicPairedEndpoint, ...] = Field(min_length=5, max_length=12)
    comparison_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, comparison_hash="0" * 64)
        return cls(
            **values,
            comparison_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"comparison_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        endpoint_bindings = tuple(
            (value.endpoint_id, value.lane_id, value.statistic_id) for value in self.endpoints
        )
        current_identity = self.model_dump(mode="json", exclude={"comparison_hash"})
        valid_hashes = {canonical_hash(current_identity)}
        if all(value.lane_id is None and value.statistic_id is None for value in self.endpoints):
            historical_identity = dict(current_identity)
            historical_identity["endpoints"] = [
                {key: item for key, item in value.items() if key not in {"lane_id", "statistic_id"}}
                for value in cast(list[dict[str, object]], current_identity["endpoints"])
            ]
            valid_hashes.add(canonical_hash(historical_identity))
            valid_shape = (
                len(self.endpoints) == len(_LEGACY_ECONOMIC_ENDPOINT_IDS)
                and {value.endpoint_id for value in self.endpoints} == _LEGACY_ECONOMIC_ENDPOINT_IDS
            )
        else:
            valid_shape = endpoint_bindings == _FIXED_ECONOMIC_ENDPOINT_DECLARATIONS
        if not valid_shape or self.comparison_hash not in valid_hashes:
            raise PanelStatisticError("alpha_research.economic_comparison_invalid")
        return self


class PanelAlphaEconomicEvaluation(_Contract):
    """One fixed candidate relationship on a common raw-return support."""

    kind: Literal["PanelAlphaEconomicEvaluation"] = "PanelAlphaEconomicEvaluation"
    candidates: tuple[AlphaEconomicCandidateEvidence, ...] = Field(min_length=3, max_length=7)
    comparisons: tuple[AlphaEconomicPairedComparison, ...] = Field(min_length=3, max_length=5)
    common_row_axis_hash: str = Field(pattern=_HASH)
    raw_simple_return_value_hash: str = Field(pattern=_HASH)
    ordered_common_row_axis_hashes: tuple[str, ...] = Field(min_length=1)
    ordered_raw_simple_return_lane_hashes: tuple[str, ...] = Field(min_length=1)
    economic_return_lane_id: Literal["RAW_SIMPLE_EXECUTION_RETURN"] = "RAW_SIMPLE_EXECUTION_RETURN"
    annualization_sessions: Literal[252] = 252
    sortino_mar: Literal[0] = 0
    downside_deviation_policy: Literal["ROOT_MEAN_SQUARED_NEGATIVE_SIMPLE_RETURN"] = (
        "ROOT_MEAN_SQUARED_NEGATIVE_SIMPLE_RETURN"
    )
    zero_variance_policy: Literal["TYPED_REFUSAL"] = "TYPED_REFUSAL"
    resampling_method_id: Literal["PAIRED_CIRCULAR_BLOCK_COMMON_SESSION_INDICES"] = (
        "PAIRED_CIRCULAR_BLOCK_COMMON_SESSION_INDICES"
    )
    dependence_block_length: int = Field(ge=1)
    evaluation_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        candidates = cast(tuple[AlphaEconomicCandidateEvidence, ...], values["candidates"])
        if (
            _fixed_economic_relationship(tuple(value.model_recipe_id for value in candidates))
            is None
        ):
            raise PanelStatisticError("alpha_research.fixed_economic_relationship_not_writable")
        provisional = cls.model_construct(**values, evaluation_hash="0" * 64)
        return cls(
            **values,
            evaluation_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"evaluation_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_fixed_relationships(self) -> Self:
        model_recipe_ids = tuple(value.model_recipe_id for value in self.candidates)
        relationship = _fixed_economic_relationship(model_recipe_ids)
        if relationship is None:
            relationship = _historical_economic_relationship(model_recipe_ids)
        if relationship is None:
            raise PanelStatisticError("alpha_research.fixed_economic_relationship_invalid")
        recipe_ids, comparison_declarations = relationship
        candidate_ids = set(recipe_ids)
        comparison_bindings = set(comparison_declarations)
        current_endpoint_bindings = tuple(_FIXED_ECONOMIC_ENDPOINT_DECLARATIONS)
        endpoint_bindings = tuple(
            tuple((item.endpoint_id, item.lane_id, item.statistic_id) for item in value.endpoints)
            for value in self.comparisons
        )
        historical_endpoints = all(
            len(value.endpoints) == len(_LEGACY_ECONOMIC_ENDPOINT_IDS)
            and {item.endpoint_id for item in value.endpoints} == _LEGACY_ECONOMIC_ENDPOINT_IDS
            and all(item.lane_id is None and item.statistic_id is None for item in value.endpoints)
            for value in self.comparisons
        )
        current_endpoints = all(value == current_endpoint_bindings for value in endpoint_bindings)
        current_identity = self.model_dump(mode="json", exclude={"evaluation_hash"})
        valid_hashes = {canonical_hash(current_identity)}
        if historical_endpoints:
            historical_identity = dict(current_identity)
            historical_comparisons: list[dict[str, object]] = []
            for comparison in cast(list[dict[str, object]], current_identity["comparisons"]):
                historical_comparison = dict(comparison)
                historical_comparison["endpoints"] = [
                    {
                        key: item
                        for key, item in endpoint.items()
                        if key not in {"lane_id", "statistic_id"}
                    }
                    for endpoint in cast(list[dict[str, object]], comparison["endpoints"])
                ]
                historical_comparisons.append(historical_comparison)
            historical_identity["comparisons"] = historical_comparisons
            valid_hashes.add(canonical_hash(historical_identity))
        if (
            {value.model_recipe_id for value in self.candidates} != candidate_ids
            or {
                (value.comparison_id, value.treatment_recipe_id, value.control_recipe_id)
                for value in self.comparisons
            }
            != comparison_bindings
            or not (historical_endpoints or current_endpoints)
            or any(value.sessions != self.candidates[0].sessions for value in self.candidates[1:])
            or any(
                item.common_session_count != len(self.candidates[0].sessions)
                for value in self.comparisons
                for item in value.endpoints
            )
            or any(
                len(value.ordered_score_surface_hashes) != len(self.ordered_common_row_axis_hashes)
                for value in self.candidates
            )
            or len(self.ordered_raw_simple_return_lane_hashes)
            != len(self.ordered_common_row_axis_hashes)
            or self.evaluation_hash not in valid_hashes
        ):
            raise PanelStatisticError("alpha_research.fixed_economic_relationship_invalid")
        return self


@dataclass(frozen=True, slots=True)
class AlphaEconomicCandidateInput:
    model_recipe_id: str
    ordered_score_surface_hashes: tuple[str, ...]
    ordered_row_axis_hashes: tuple[str, ...]
    ordered_raw_simple_return_lane_hashes: tuple[str, ...]
    sessions: tuple[date, ...]
    listings: tuple[str, ...]
    scores: FloatArray
    raw_simple_returns: FloatArray


def _economic_path_metrics(values: FloatArray) -> AlphaEconomicPathMetrics:
    path = np.asarray(values, dtype=np.float64)
    if path.ndim != 1 or path.size < 2 or not np.isfinite(path).all() or bool(np.any(path <= -1.0)):
        raise PanelStatisticError("alpha_research.economic_path_invalid")
    deviation = float(np.std(path, ddof=1))
    if deviation <= 0.0 or not np.isfinite(deviation):
        raise PanelStatisticError("alpha_research.economic_path_variance_zero")
    downside = sortino_ratio(path)
    if not np.isfinite(downside):
        raise PanelStatisticError("alpha_research.economic_path_downside_variance_zero")
    log_path = np.log1p(path)
    return AlphaEconomicPathMetrics(
        mean_return=float(np.mean(path)),
        cumulative_return=float(np.expm1(np.sum(log_path))),
        annualized_return=float(np.expm1(np.sum(log_path) * 252.0 / path.size)),
        annualized_volatility=annualized_volatility(path),
        sharpe=float(np.mean(path) / deviation * sqrt(252.0)),
        sortino=downside,
        maximum_drawdown=maximum_drawdown(log_path),
    )


def _active_path_metrics(values: FloatArray) -> AlphaActivePathMetrics:
    active = np.asarray(values, dtype=np.float64)
    if active.ndim != 1 or active.size < 2 or not np.isfinite(active).all():
        raise PanelStatisticError("alpha_research.active_path_invalid")
    deviation = float(np.std(active, ddof=1))
    tracking = annualized_volatility(active)
    if deviation <= 0.0 or tracking <= 0.0:
        raise PanelStatisticError("alpha_research.active_path_variance_zero")
    sortino = sortino_ratio(active)
    if not np.isfinite(sortino):
        raise PanelStatisticError("alpha_research.active_path_downside_variance_zero")
    return AlphaActivePathMetrics(
        mean_active_return=float(np.mean(active)),
        tracking_error=tracking,
        information_ratio=float(np.mean(active) * 252.0 / tracking),
        active_sharpe=float(np.mean(active) / deviation * sqrt(252.0)),
        active_sortino=sortino,
    )


def _paired_circular_interval(
    *,
    left: FloatArray,
    right: FloatArray,
    resample_indices: IntArray,
    statistic: _AlphaEconomicStatisticId,
) -> tuple[float, tuple[float, float]]:
    if (
        left.shape != right.shape
        or left.ndim != 1
        or left.size < 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or resample_indices.shape != (_BOOTSTRAP_RESAMPLES, left.size)
        or np.any(resample_indices < 0)
        or np.any(resample_indices >= left.size)
    ):
        raise PanelStatisticError("alpha_research.economic_paired_axis_invalid")

    def evaluate(values: FloatArray) -> float:
        if statistic == "MEAN":
            return float(np.mean(values))
        if statistic == "SHARPE":
            deviation = float(np.std(values, ddof=1))
            if deviation <= 0.0:
                raise PanelStatisticError("alpha_research.economic_bootstrap_variance_zero")
            return float(np.mean(values) / deviation * sqrt(252.0))
        result = sortino_ratio(values)
        if not np.isfinite(result):
            raise PanelStatisticError("alpha_research.economic_bootstrap_downside_variance_zero")
        return result

    point = evaluate(left) - evaluate(right)
    estimates = np.empty(_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for position, indices in enumerate(resample_indices):
        estimates[position] = evaluate(left[indices]) - evaluate(right[indices])
    lower, upper = np.percentile(estimates, (2.5, 97.5), method="linear")
    return point, (float(lower), float(upper))


def _paired_circular_resample_indices(*, size: int, block_length: int) -> IntArray:
    if size < 2 or block_length < 1:
        raise PanelStatisticError("alpha_research.economic_resampling_input_invalid")
    blocks = ceil(size / block_length)
    offsets: IntArray = np.arange(block_length, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(_BOOTSTRAP_SEED))
    indices = np.empty((_BOOTSTRAP_RESAMPLES, size), dtype=np.int64)
    for position in range(_BOOTSTRAP_RESAMPLES):
        starts = rng.integers(0, size, size=blocks, dtype=np.int64)
        indices[position] = ((starts[:, None] + offsets[None, :]) % size).reshape(-1)[:size]
    indices.setflags(write=False)
    return indices


def _paired_endpoint(
    *,
    endpoint_id: _AlphaEconomicEndpointId,
    lane_id: _AlphaEconomicLaneId,
    statistic_id: _AlphaEconomicStatisticId,
    left: FloatArray,
    right: FloatArray,
    resample_indices: IntArray,
) -> AlphaEconomicPairedEndpoint:
    point, interval = _paired_circular_interval(
        left=left,
        right=right,
        resample_indices=resample_indices,
        statistic=statistic_id,
    )
    disposition: Disposition = (
        "SUPERIOR_PAIRED_EVIDENCE"
        if interval[0] > 0.0
        else ("INFERIOR_PAIRED_EVIDENCE" if interval[1] < 0.0 else "INCONCLUSIVE_PAIRED_EVIDENCE")
    )
    return AlphaEconomicPairedEndpoint(
        endpoint_id=endpoint_id,
        lane_id=lane_id,
        statistic_id=statistic_id,
        point_difference=point,
        paired_interval_95=interval,
        common_session_count=int(left.size),
        disposition=disposition,
    )


def _candidate_economic_evidence(
    value: AlphaEconomicCandidateInput,
    *,
    minimum_paired_rows: int,
    persistence_lags: tuple[int, ...],
) -> AlphaEconomicCandidateEvidence:
    count = len(value.sessions)
    if (
        count == 0
        or len(value.listings) != count
        or value.scores.shape != (count,)
        or value.raw_simple_returns.shape != (count,)
        or not np.isfinite(value.scores).all()
        or not np.isfinite(value.raw_simple_returns).all()
        or value.scores.flags.writeable
        or value.raw_simple_returns.flags.writeable
    ):
        raise PanelStatisticError("alpha_research.economic_candidate_input_invalid")
    session_axis, positions_by_session = _session_positions(value.sessions)
    top: list[float] = []
    bottom: list[float] = []
    benchmark: list[float] = []
    rank_ics: list[float] = []
    tail_ics: list[float] = []
    turnovers: list[float] = []
    previous_top: frozenset[str] = frozenset()
    for session in session_axis:
        positions = positions_by_session[session]
        if positions.size < minimum_paired_rows:
            raise PanelStatisticError("alpha_research.economic_session_rows_insufficient")
        scores = value.scores[positions]
        returns = value.raw_simple_returns[positions]
        listing_values = np.asarray([value.listings[int(item)] for item in positions])
        order = np.lexsort((listing_values, -scores))
        decile_count = max(1, ceil(positions.size / 10.0))
        top_local = order[:decile_count]
        bottom_local = order[-decile_count:]
        top.append(float(np.mean(returns[top_local])))
        bottom.append(float(np.mean(returns[bottom_local])))
        benchmark.append(float(np.mean(returns)))
        rank_ic, insufficient, constant = session_grouped_rank_ic(
            np.ascontiguousarray(scores, dtype=np.float64),
            np.ascontiguousarray(returns, dtype=np.float64),
            np.zeros(positions.size, dtype=np.int64),
            minimum_paired_rows=minimum_paired_rows,
        )
        if insufficient or constant or rank_ic is None:
            raise PanelStatisticError("alpha_research.economic_rank_ic_session_invalid")
        rank_ics.append(float(rank_ic))
        tail = np.concatenate((top_local, bottom_local))
        tail_ic, tail_insufficient, tail_constant = session_grouped_rank_ic(
            np.ascontiguousarray(scores[tail], dtype=np.float64),
            np.ascontiguousarray(returns[tail], dtype=np.float64),
            np.zeros(tail.size, dtype=np.int64),
            minimum_paired_rows=max(2, min(minimum_paired_rows, int(tail.size))),
        )
        if tail_insufficient or tail_constant or tail_ic is None:
            raise PanelStatisticError("alpha_research.economic_tail_rank_ic_session_invalid")
        tail_ics.append(float(tail_ic))
        current_top = frozenset(str(listing_values[item]) for item in top_local)
        turnovers.append(
            1.0
            if not previous_top
            else 1.0 - len(current_top & previous_top) / float(len(current_top))
        )
        previous_top = current_top

    top_values = np.asarray(top, dtype=np.float64)
    bottom_values = np.asarray(bottom, dtype=np.float64)
    benchmark_values = np.asarray(benchmark, dtype=np.float64)
    active_values = top_values - benchmark_values
    long_short_values = top_values - bottom_values
    short_values = -bottom_values
    turnover_values = np.asarray(turnovers, dtype=np.float64)
    for lane in (
        top_values,
        bottom_values,
        benchmark_values,
        active_values,
        long_short_values,
        short_values,
        turnover_values,
    ):
        lane.setflags(write=False)
    session_position = {item: index for index, item in enumerate(session_axis)}
    listing_axis = tuple(sorted(set(value.listings)))
    listing_position = {item: index for index, item in enumerate(listing_axis)}
    session_codes: IntArray = np.fromiter(
        (session_position[item] for item in value.sessions), dtype=np.int64, count=count
    )
    listing_codes: IntArray = np.fromiter(
        (listing_position[item] for item in value.listings), dtype=np.int64, count=count
    )
    row_panel: IntArray = np.full((len(session_axis), len(listing_axis)), -1, dtype=np.int64)
    row_panel[session_codes, listing_codes] = np.arange(count, dtype=np.int64)
    persistence = tuple(
        _persistence_point(
            lag_sessions=lag,
            scores=value.scores,
            raw_simple_returns=value.raw_simple_returns,
            minimum_paired_rows=minimum_paired_rows,
            session_codes=session_codes,
            listing_codes=listing_codes,
            row_index_panel=row_panel,
        )
        for lag in persistence_lags
    )
    rank_array = np.asarray(rank_ics, dtype=np.float64)
    rank_std = float(np.std(rank_array, ddof=1))
    if rank_std <= 0.0:
        raise PanelStatisticError("alpha_research.economic_rank_ic_variance_zero")
    mean_long = float(np.mean(top_values))
    mean_short = float(np.mean(short_values))
    contribution_denominator = abs(mean_long) + abs(mean_short)
    long_share = abs(mean_long) / contribution_denominator if contribution_denominator else 0.5
    return AlphaEconomicCandidateEvidence.create(
        model_recipe_id=value.model_recipe_id,
        ordered_score_surface_hashes=value.ordered_score_surface_hashes,
        sessions=session_axis,
        long_only_returns=tuple(float(item) for item in top_values),
        eligible_universe_returns=tuple(float(item) for item in benchmark_values),
        active_returns=tuple(float(item) for item in active_values),
        long_short_returns=tuple(float(item) for item in long_short_values),
        top_decile_returns=tuple(float(item) for item in top_values),
        bottom_decile_returns=tuple(float(item) for item in bottom_values),
        short_contributions=tuple(float(item) for item in short_values),
        session_rank_ics=tuple(rank_ics),
        long_only_metrics=_economic_path_metrics(top_values),
        active_metrics=_active_path_metrics(active_values),
        long_short_metrics=_economic_path_metrics(long_short_values),
        mean_top_decile_return=mean_long,
        mean_top_decile_active_contribution=float(np.mean(active_values)),
        mean_bottom_decile_return=float(np.mean(bottom_values)),
        mean_short_contribution=mean_short,
        long_contribution_share=long_share,
        short_contribution_share=1.0 - long_share,
        mean_session_rank_ic=float(np.mean(rank_array)),
        rank_ic_information_ratio=float(np.mean(rank_array) / rank_std),
        tail_rank_ic=float(np.mean(tail_ics)),
        score_autocorrelation_lag_1=next(
            item.score_autocorrelation for item in persistence if item.lag_sessions == 1
        ),
        stale_score_ic_decay=persistence,
        mean_one_way_turnover_diagnostic=float(np.mean(turnover_values)),
        break_even_cost_bps_diagnostic=break_even_cost_bps(
            gross_simple_returns=top_values, one_way_turnovers=turnover_values
        ),
    )


def evaluate_fixed_recipe_alpha_economics(
    *,
    candidates: tuple[AlphaEconomicCandidateInput, ...],
    holding_horizon_sessions: int,
    minimum_paired_rows: int = 20,
    persistence_lags: tuple[int, ...] = (0, 1, 2, 5, 10, 21),
) -> PanelAlphaEconomicEvaluation:
    """Evaluate one installed fixed-recipe OOF relationship without selection feedback."""

    relationship = _fixed_economic_relationship(
        tuple(value.model_recipe_id for value in candidates)
    )
    if (
        relationship is None
        or tuple(sorted(set(persistence_lags))) != persistence_lags
        or not {0, 1}.issubset(persistence_lags)
    ):
        raise PanelStatisticError("alpha_research.fixed_economic_recipe_set_invalid")
    common_input = candidates[0]
    if any(
        value.sessions != common_input.sessions
        or value.listings != common_input.listings
        or value.ordered_row_axis_hashes != common_input.ordered_row_axis_hashes
        or value.ordered_raw_simple_return_lane_hashes
        != common_input.ordered_raw_simple_return_lane_hashes
        or value.raw_simple_returns.shape != common_input.raw_simple_returns.shape
        or not np.array_equal(value.raw_simple_returns, common_input.raw_simple_returns)
        for value in candidates[1:]
    ) or not (
        len(common_input.ordered_score_surface_hashes)
        == len(common_input.ordered_row_axis_hashes)
        == len(common_input.ordered_raw_simple_return_lane_hashes)
    ):
        raise PanelStatisticError("alpha_research.fixed_economic_common_axis_invalid")
    common_row_hash = row_axis_hash(common_input.sessions, common_input.listings)
    raw_return_hash = score_value_hash(common_input.raw_simple_returns)
    evidence = tuple(
        _candidate_economic_evidence(
            value,
            minimum_paired_rows=minimum_paired_rows,
            persistence_lags=persistence_lags,
        )
        for value in sorted(candidates, key=lambda item: item.model_recipe_id)
    )
    by_id = {value.model_recipe_id: value for value in evidence}
    common_axis = next(iter(by_id.values())).sessions
    if any(value.sessions != common_axis for value in evidence):
        raise PanelStatisticError("alpha_research.fixed_economic_common_axis_invalid")
    block = dependence_aware_block_length(
        spans=(1,), holding_horizon_sessions=holding_horizon_sessions
    )
    resample_indices = _paired_circular_resample_indices(size=len(common_axis), block_length=block)
    comparisons: list[AlphaEconomicPairedComparison] = []
    _recipe_ids, comparison_declarations = relationship
    for comparison_id, treatment_id, control_id in comparison_declarations:
        treatment = by_id[treatment_id]
        control = by_id[control_id]
        lanes_by_id: Mapping[_AlphaEconomicLaneId, tuple[float, ...]] = {
            "LONG_ONLY_RETURNS": treatment.long_only_returns,
            "ACTIVE_RETURNS": treatment.active_returns,
            "LONG_SHORT_RETURNS": treatment.long_short_returns,
            "TOP_DECILE_RETURNS": treatment.top_decile_returns,
            "SHORT_CONTRIBUTIONS": treatment.short_contributions,
            "SESSION_RANK_ICS": treatment.session_rank_ics,
        }
        control_lanes_by_id: Mapping[_AlphaEconomicLaneId, tuple[float, ...]] = {
            "LONG_ONLY_RETURNS": control.long_only_returns,
            "ACTIVE_RETURNS": control.active_returns,
            "LONG_SHORT_RETURNS": control.long_short_returns,
            "TOP_DECILE_RETURNS": control.top_decile_returns,
            "SHORT_CONTRIBUTIONS": control.short_contributions,
            "SESSION_RANK_ICS": control.session_rank_ics,
        }
        endpoints = tuple(
            _paired_endpoint(
                endpoint_id=endpoint_id,
                lane_id=lane_id,
                statistic_id=statistic_id,
                left=np.asarray(lanes_by_id[lane_id], dtype=np.float64),
                right=np.asarray(control_lanes_by_id[lane_id], dtype=np.float64),
                resample_indices=resample_indices,
            )
            for endpoint_id, lane_id, statistic_id in _FIXED_ECONOMIC_ENDPOINT_DECLARATIONS
        )
        comparisons.append(
            AlphaEconomicPairedComparison.create(
                comparison_id=comparison_id,
                treatment_recipe_id=treatment_id,
                control_recipe_id=control_id,
                endpoints=endpoints,
            )
        )
    return PanelAlphaEconomicEvaluation.create(
        candidates=evidence,
        comparisons=tuple(comparisons),
        common_row_axis_hash=common_row_hash,
        raw_simple_return_value_hash=raw_return_hash,
        ordered_common_row_axis_hashes=common_input.ordered_row_axis_hashes,
        ordered_raw_simple_return_lane_hashes=(common_input.ordered_raw_simple_return_lane_hashes),
        dependence_block_length=block,
    )


def dependence_aware_block_length(*, spans: tuple[int, ...], holding_horizon_sessions: int) -> int:
    if not spans or any(value < 1 for value in spans) or holding_horizon_sessions < 1:
        raise PanelStatisticError("alpha_research.dependence_policy_input_invalid")
    return max(21, *spans, holding_horizon_sessions)


def circular_block_interval(
    observations: tuple[float, ...], *, block_length: int
) -> tuple[float, float]:
    values: FloatArray = np.asarray(observations, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all() or block_length < 1:
        raise PanelStatisticError("alpha_research.paired_block_input_invalid")
    blocks = ceil(values.size / block_length)
    offsets: IntArray = np.arange(block_length, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(_BOOTSTRAP_SEED))
    estimates: FloatArray = np.empty(_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for position in range(_BOOTSTRAP_RESAMPLES):
        starts = rng.integers(0, values.size, size=blocks, dtype=np.int64)
        indices = ((starts[:, None] + offsets[None, :]) % values.size).reshape(-1)
        estimates[position] = float(np.mean(values[indices[: values.size]]))
    lower, upper = np.percentile(estimates, (2.5, 97.5), method="linear")
    return float(lower), float(upper)


def _mean_std_ir(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, None, None
    sample_std = float(np.std(values, ddof=1))
    information_ratio = mean / sample_std if sample_std > 0.0 else None
    return mean, sample_std, information_ratio


def panel_scientific_session_evidence(
    *,
    fold_index: int,
    spec_id: str,
    view_id: str,
    model_family_id: str,
    recipe_hash: str,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
    scores: FloatArray,
    raw_simple_returns: FloatArray,
    evaluation_sessions: tuple[date, ...],
    minimum_paired_rows: int = 20,
) -> PanelScientificSessionEvidence:
    """Evaluate a raw model surface without constructing a Portfolio book."""

    count = len(sessions)
    if (
        len(listings) != count
        or any(value.shape != (count,) for value in (scores, raw_simple_returns))
        or any(value.flags.writeable for value in (scores, raw_simple_returns))
        or not all(np.isfinite(value).all() for value in (scores, raw_simple_returns))
        or evaluation_sessions != tuple(sorted(set(evaluation_sessions)))
        or minimum_paired_rows < 2
    ):
        raise PanelStatisticError("alpha_research.panel_scientific_input_invalid")
    rank_values: list[float | None] = []
    spreads: list[float | None] = []
    insufficient = 0
    constant = 0
    retained: list[date] = []
    _session_axis, positions_by_session = _session_positions(sessions)
    empty: IntArray = np.asarray((), dtype=np.int64)
    empty.setflags(write=False)
    for session in evaluation_sessions:
        positions = positions_by_session.get(session, empty)
        if positions.size < minimum_paired_rows:
            insufficient += 1
            rank_values.append(None)
            spreads.append(None)
            retained.append(session)
            continue
        session_scores = np.ascontiguousarray(scores[positions], dtype=np.float64)
        session_outcomes = np.ascontiguousarray(raw_simple_returns[positions], dtype=np.float64)
        rank_ic, rank_insufficient, rank_constant = session_grouped_rank_ic(
            session_scores,
            session_outcomes,
            np.zeros(positions.size, dtype=np.int64),
            minimum_paired_rows=minimum_paired_rows,
        )
        insufficient += rank_insufficient
        constant += rank_constant
        rank_values.append(rank_ic if not (rank_insufficient or rank_constant) else None)
        spreads.append(
            None
            if np.unique(session_scores).size < 2
            else gross_decile_spread(session_scores, raw_simple_returns[positions])
        )
        retained.append(session)
    finite_rank = [float(value) for value in rank_values if value is not None]
    finite_spread = [float(value) for value in spreads if value is not None]
    mean, sample_std, information_ratio = _mean_std_ir(finite_rank)
    return PanelScientificSessionEvidence.create(
        fold_index=fold_index,
        spec_id=spec_id,
        view_id=view_id,
        model_family_id=model_family_id,
        recipe_hash=recipe_hash,
        sessions=tuple(retained),
        session_rank_ics=tuple(rank_values),
        session_gross_long_short_spreads=tuple(spreads),
        insufficient_session_count=insufficient,
        constant_session_count=constant,
        mean_rank_ic=mean,
        rank_ic_sample_std=sample_std,
        rank_ic_information_ratio=information_ratio,
        mean_gross_long_short_spread=(float(np.mean(finite_spread)) if finite_spread else None),
    )


def select_panel_model_scientifically(
    *,
    fold_index: int,
    evidence: tuple[PanelScientificSessionEvidence, ...],
    complexities: Mapping[str, CandidateComplexity],
    holding_horizon_sessions: int,
) -> PanelModelScientificSelection:
    """Authorize a winner only when paired Rank-IC evidence separates it."""

    if (
        not evidence
        or len({value.spec_id for value in evidence}) != len(evidence)
        or any(value.spec_id not in complexities for value in evidence)
    ):
        raise PanelStatisticError("alpha_research.model_scientific_selection_input_invalid")
    rank_by_spec = {
        value.spec_id: {
            session: float(rank)
            for session, rank in zip(value.sessions, value.session_rank_ics, strict=True)
            if rank is not None
        }
        for value in evidence
    }
    common = tuple(sorted(set.intersection(*(set(value) for value in rank_by_spec.values()))))
    if len(common) < 2:
        raise PanelStatisticError("alpha_research.model_spec_rank_common_axis_insufficient")
    mean_rank = {
        spec_id: float(np.mean([values[session] for session in common]))
        for spec_id, values in rank_by_spec.items()
    }
    best = min(
        evidence,
        key=lambda value: (-mean_rank[value.spec_id], value.spec_id, value.evidence_hash),
    )
    block = dependence_aware_block_length(
        spans=(1,), holding_horizon_sessions=holding_horizon_sessions
    )
    best_values = rank_by_spec[best.spec_id]
    indistinguishable: list[PanelScientificSessionEvidence] = []
    for candidate in evidence:
        candidate_values = rank_by_spec[candidate.spec_id]
        interval = circular_block_interval(
            tuple(best_values[session] - candidate_values[session] for session in common),
            block_length=block,
        )
        if interval[0] <= 0.0 <= interval[1]:
            indistinguishable.append(candidate)
    fallback = min(
        indistinguishable,
        key=lambda value: (
            complexities[value.spec_id].transform_count,
            complexities[value.spec_id].model_order,
            value.spec_id,
            value.evidence_hash,
        ),
    )
    scientific_winner = best if len(evidence) > 1 and len(indistinguishable) == 1 else None
    if scientific_winner is not None:
        reason: Literal[
            "SCIENTIFIC_WINNER", "FEWER_TRANSFORMS", "SIMPLER_MODEL", "CANONICAL_IDENTITY"
        ] = "SCIENTIFIC_WINNER"
    elif (
        complexities[fallback.spec_id].transform_count < complexities[best.spec_id].transform_count
    ):
        reason = "FEWER_TRANSFORMS"
    elif complexities[fallback.spec_id].model_order < complexities[best.spec_id].model_order:
        reason = "SIMPLER_MODEL"
    else:
        reason = "CANONICAL_IDENTITY"
    return PanelModelScientificSelection.create(
        fold_index=fold_index,
        candidate_evidence_hashes=tuple(value.evidence_hash for value in evidence),
        common_rank_ic_sessions=common,
        inner_best_evidence_hash=best.evidence_hash,
        inner_best_spec_id=best.spec_id,
        statistically_indistinguishable_evidence_hashes=tuple(
            value.evidence_hash for value in indistinguishable
        ),
        scientific_disposition=(
            "SUPERIOR_PAIRED_EVIDENCE"
            if scientific_winner is not None
            else "INCONCLUSIVE_PAIRED_EVIDENCE"
        ),
        scientific_winner_spec_id=(
            scientific_winner.spec_id if scientific_winner is not None else None
        ),
        operational_fallback_evidence_hash=fallback.evidence_hash,
        operational_fallback_spec_id=fallback.spec_id,
        operational_fallback_reason=reason,
        dependence_block_length=block,
    )


def _session_positions(
    sessions: tuple[date, ...],
) -> tuple[tuple[date, ...], Mapping[date, IntArray]]:
    grouped: dict[date, list[int]] = {}
    for position, session in enumerate(sessions):
        grouped.setdefault(session, []).append(position)
    session_axis = tuple(sorted(grouped))
    positions: dict[date, IntArray] = {}
    for session in session_axis:
        values: IntArray = np.asarray(grouped[session], dtype=np.int64)
        values.setflags(write=False)
        positions[session] = values
    return session_axis, MappingProxyType(positions)


def _equal_session_rank_correlation(
    *,
    left: FloatArray,
    right: FloatArray,
    session_codes: IntArray,
    minimum_paired_rows: int,
) -> tuple[float | None, int]:
    if not left.size:
        return None, 0
    value, insufficient, constant = session_grouped_rank_ic(
        left,
        right,
        session_codes,
        minimum_paired_rows=minimum_paired_rows,
    )
    # `np.unique` rather than a set over `.tolist()`: this runs three times per
    # persistence lag on the full row axis, and the list form boxed a million
    # Python integers on every call.
    distinct = int(np.unique(session_codes).size)
    return (value if not (insufficient or constant) else None), int(
        distinct - insufficient - constant
    )


def _persistence_point(
    *,
    lag_sessions: int,
    scores: FloatArray,
    raw_simple_returns: FloatArray,
    minimum_paired_rows: int,
    session_codes: IntArray,
    listing_codes: IntArray,
    row_index_panel: IntArray,
) -> PanelScorePersistencePoint:
    """Stale-score and forward-return persistence for one lag.

    The row axis is addressed through an integer panel rather than a
    ``(session, listing)`` dictionary. The dictionary form walked every row in
    Python and, for the forward leg, walked it again once per session of the
    horizon: at panel scale and the installed lag set that is roughly forty
    million tuple-keyed lookups per candidate. The panel gives the same answers
    from array indexing, and the masks preserve row order so that tied ranks
    break exactly as they did before.
    """

    horizon = max(1, lag_sessions)
    session_count = int(row_index_panel.shape[0])
    stale_positions = session_codes - lag_sessions
    has_stale = stale_positions >= 0
    stale_rows: IntArray = np.full(session_codes.shape, -1, dtype=np.int64)
    stale_rows[has_stale] = row_index_panel[stale_positions[has_stale], listing_codes[has_stale]]
    paired = has_stale & (stale_rows >= 0)
    paired_stale = stale_rows[paired]

    product: FloatArray = np.ones(session_codes.shape, dtype=np.float64)
    alive = (session_codes + horizon) <= session_count
    for offset in range(horizon):
        forward_rows: IntArray = np.full(session_codes.shape, -1, dtype=np.int64)
        forward_rows[alive] = row_index_panel[session_codes[alive] + offset, listing_codes[alive]]
        alive &= forward_rows >= 0
        product[alive] *= 1.0 + raw_simple_returns[forward_rows[alive]]

    autocorrelation, autocorrelation_count = _equal_session_rank_correlation(
        left=scores[paired],
        right=scores[paired_stale],
        session_codes=session_codes[paired],
        minimum_paired_rows=minimum_paired_rows,
    )
    stale_predictive, stale_predictive_count = _equal_session_rank_correlation(
        left=scores[paired_stale],
        right=raw_simple_returns[paired],
        session_codes=session_codes[paired],
        minimum_paired_rows=minimum_paired_rows,
    )
    forward_value, forward_count = _equal_session_rank_correlation(
        left=scores[alive],
        right=product[alive] - 1.0,
        session_codes=session_codes[alive],
        minimum_paired_rows=minimum_paired_rows,
    )
    return PanelScorePersistencePoint(
        lag_sessions=lag_sessions,
        forward_horizon_sessions=horizon,
        score_autocorrelation=autocorrelation,
        score_autocorrelation_session_count=autocorrelation_count,
        stale_score_predictive_ic=stale_predictive,
        stale_score_predictive_session_count=stale_predictive_count,
        forward_cumulative_return_ic=forward_value,
        forward_cumulative_return_session_count=forward_count,
    )


def _score_filter_evidence(
    *,
    spec: AlphaScoreFilterSpec,
    upstream_raw_score_artifact_hashes: tuple[str, ...],
    model_spec_ids_by_fold: tuple[str, ...],
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
    fold_indices: IntArray,
    scores: FloatArray,
    raw_simple_returns: FloatArray,
    persistence_lags: tuple[int, ...],
    minimum_paired_rows: int,
    persistence_session_codes: IntArray,
    persistence_listing_codes: IntArray,
    persistence_row_index_panel: IntArray,
) -> PanelScoreFilterEvidence:
    session_axis, positions_by_session = _session_positions(sessions)
    rank_values: list[float | None] = []
    spreads: list[float | None] = []
    insufficient = 0
    constant = 0
    for session in session_axis:
        positions = positions_by_session[session]
        if positions.size < minimum_paired_rows:
            insufficient += 1
            rank_values.append(None)
            spreads.append(None)
            continue
        session_scores = np.ascontiguousarray(scores[positions], dtype=np.float64)
        session_outcomes = np.ascontiguousarray(raw_simple_returns[positions], dtype=np.float64)
        rank_ic, rank_insufficient, rank_constant = session_grouped_rank_ic(
            session_scores,
            session_outcomes,
            np.zeros(positions.size, dtype=np.int64),
            minimum_paired_rows=minimum_paired_rows,
        )
        insufficient += rank_insufficient
        constant += rank_constant
        rank_values.append(rank_ic if not (rank_insufficient or rank_constant) else None)
        spreads.append(
            None
            if np.unique(session_scores).size < 2
            else gross_decile_spread(session_scores, raw_simple_returns[positions])
        )
    rank_by_session = dict(zip(session_axis, rank_values, strict=True))
    fold_mean_rank_ics: list[float | None] = []
    for fold_index in range(len(model_spec_ids_by_fold)):
        fold_sessions = {
            session
            for session, row_fold in zip(sessions, fold_indices, strict=True)
            if int(row_fold) == fold_index
        }
        values = [
            cast(float, rank_by_session[session])
            for session in sorted(fold_sessions)
            if rank_by_session[session] is not None
        ]
        fold_mean_rank_ics.append(float(np.mean(values)) if values else None)
    finite_rank = [float(value) for value in rank_values if value is not None]
    finite_spreads = [float(value) for value in spreads if value is not None]
    mean, sample_std, information_ratio = _mean_std_ir(finite_rank)
    return PanelScoreFilterEvidence.create(
        upstream_raw_score_artifact_hashes=upstream_raw_score_artifact_hashes,
        model_spec_ids_by_fold=model_spec_ids_by_fold,
        filter_spec=spec,
        row_count=len(sessions),
        row_axis_hash=row_axis_hash(sessions, listings),
        score_value_hash=score_value_hash(scores),
        sessions=session_axis,
        fold_mean_rank_ics=tuple(fold_mean_rank_ics),
        session_rank_ics=tuple(rank_values),
        session_gross_long_short_spreads=tuple(spreads),
        insufficient_session_count=insufficient,
        constant_session_count=constant,
        mean_rank_ic=mean,
        rank_ic_sample_std=sample_std,
        rank_ic_information_ratio=information_ratio,
        mean_gross_long_short_spread=(float(np.mean(finite_spreads)) if finite_spreads else None),
        persistence=tuple(
            _persistence_point(
                lag_sessions=lag,
                scores=scores,
                raw_simple_returns=raw_simple_returns,
                minimum_paired_rows=minimum_paired_rows,
                session_codes=persistence_session_codes,
                listing_codes=persistence_listing_codes,
                row_index_panel=persistence_row_index_panel,
            )
            for lag in persistence_lags
        ),
    )


def _select_score_filter_for_fold(
    *,
    fold_index: int,
    first_formation_session: date,
    sessions: tuple[date, ...],
    holding_end_sessions: tuple[date, ...],
    fold_indices: IntArray,
    evidence: tuple[PanelScoreFilterEvidence, ...],
    holding_horizon_sessions: int,
) -> PanelScoreFilterSelection:
    raw = next(
        (
            value
            for value in evidence
            if (
                value.filter_spec.method_id,
                value.filter_spec.span_sessions,
                value.filter_spec.input_mode,
            )
            == (RAW_SCORE_TEMPORAL_FILTER_METHOD_ID, 1, RAW_SCORE_INPUT_MODE)
        ),
        None,
    )
    if raw is None:
        raise PanelStatisticError("alpha_research.raw_score_filter_control_absent")
    # A session is eligible only when every admitted row outcome is mature and
    # its OOF fold precedes the fold now being selected.
    eligibility: dict[date, list[bool]] = {}
    for session, holding_end, row_fold in zip(
        sessions, holding_end_sessions, fold_indices, strict=True
    ):
        eligibility.setdefault(session, []).append(
            int(row_fold) < fold_index and holding_end < first_formation_session
        )
    matured_sessions = tuple(
        sorted(session for session, flags in eligibility.items() if flags and all(flags))
    )
    rank_by_spec = {
        value.filter_spec.spec_id: dict(zip(value.sessions, value.session_rank_ics, strict=True))
        for value in evidence
    }
    common = tuple(
        session
        for session in matured_sessions
        if all(rank_by_spec[spec_id].get(session) is not None for spec_id in rank_by_spec)
    )
    block = dependence_aware_block_length(
        spans=tuple(value.filter_spec.span_sessions for value in evidence),
        holding_horizon_sessions=holding_horizon_sessions,
    )
    if len(common) < 2:
        criteria = tuple(
            PanelScoreFilterCandidateCriterion(
                spec_id=value.filter_spec.spec_id,
                evidence_hash=value.evidence_hash,
                matured_prior_session_count=len(common),
                mean_rank_ic=None,
                best_minus_candidate_interval=None,
            )
            for value in evidence
        )
        return PanelScoreFilterSelection.create(
            fold_index=fold_index,
            current_fold_first_formation_session=first_formation_session,
            candidate_criteria=criteria,
            common_matured_prior_sessions=(),
            scientific_disposition="INCONCLUSIVE_PAIRED_EVIDENCE",
            scientific_winner_spec_id=None,
            operational_fallback_spec_id=raw.filter_spec.spec_id,
            operational_fallback_reason="DECLARED_RAW_CONTROL",
            statistically_indistinguishable_spec_ids=tuple(
                value.filter_spec.spec_id for value in evidence
            ),
            dependence_block_length=block,
        )
    mean_by_spec = {
        spec_id: float(np.mean([cast(float, rank_by_spec[spec_id][session]) for session in common]))
        for spec_id in rank_by_spec
    }
    best = min(
        evidence,
        key=lambda value: (
            -mean_by_spec[value.filter_spec.spec_id],
            value.filter_spec.spec_id,
            value.evidence_hash,
        ),
    )
    best_values = rank_by_spec[best.filter_spec.spec_id]
    intervals: dict[str, tuple[float, float]] = {}
    indistinguishable: list[PanelScoreFilterEvidence] = []
    for candidate in evidence:
        candidate_values = rank_by_spec[candidate.filter_spec.spec_id]
        interval = circular_block_interval(
            tuple(
                cast(float, best_values[session]) - cast(float, candidate_values[session])
                for session in common
            ),
            block_length=block,
        )
        intervals[candidate.filter_spec.spec_id] = interval
        if interval[0] <= 0.0 <= interval[1]:
            indistinguishable.append(candidate)
    method_order = {
        RAW_SCORE_TEMPORAL_FILTER_METHOD_ID: 0,
        "TRAILING_MEAN_FORMATION_SCORE": 1,
        "EWMA_FORMATION_SCORE": 2,
    }
    input_mode_order = {RAW_SCORE_INPUT_MODE: 0, "CROSS_SECTION_STANDARDIZED_SCORE": 1}
    fallback = min(
        indistinguishable,
        key=lambda value: (
            value.filter_spec.span_sessions,
            method_order[value.filter_spec.method_id],
            input_mode_order[value.filter_spec.input_mode],
            value.filter_spec.spec_id,
            value.evidence_hash,
        ),
    )
    winner = best if len(evidence) > 1 and len(indistinguishable) == 1 else None
    if winner is not None:
        fallback = winner
        reason: Literal[
            "SCIENTIFIC_WINNER", "DECLARED_RAW_CONTROL", "SMALLEST_SPAN", "SIMPLER_FILTER"
        ] = "SCIENTIFIC_WINNER"
    elif fallback.filter_spec.span_sessions < best.filter_spec.span_sessions:
        reason = "SMALLEST_SPAN"
    elif method_order[fallback.filter_spec.method_id] < method_order[best.filter_spec.method_id]:
        reason = "SIMPLER_FILTER"
    else:
        reason = "DECLARED_RAW_CONTROL"
    criteria = tuple(
        PanelScoreFilterCandidateCriterion(
            spec_id=value.filter_spec.spec_id,
            evidence_hash=value.evidence_hash,
            matured_prior_session_count=len(common),
            mean_rank_ic=mean_by_spec[value.filter_spec.spec_id],
            best_minus_candidate_interval=intervals[value.filter_spec.spec_id],
        )
        for value in evidence
    )
    return PanelScoreFilterSelection.create(
        fold_index=fold_index,
        current_fold_first_formation_session=first_formation_session,
        candidate_criteria=criteria,
        common_matured_prior_sessions=common,
        scientific_disposition=(
            "SUPERIOR_PAIRED_EVIDENCE" if winner is not None else "INCONCLUSIVE_PAIRED_EVIDENCE"
        ),
        scientific_winner_spec_id=(winner.filter_spec.spec_id if winner is not None else None),
        operational_fallback_spec_id=fallback.filter_spec.spec_id,
        operational_fallback_reason=reason,
        statistically_indistinguishable_spec_ids=tuple(
            value.filter_spec.spec_id for value in indistinguishable
        ),
        dependence_block_length=block,
    )


def evaluate_panel_score_filters(
    *,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
    fold_indices: IntArray,
    holding_end_sessions: tuple[date, ...],
    raw_scores: FloatArray,
    raw_simple_returns: FloatArray,
    upstream_raw_score_artifact_hashes: tuple[str, ...],
    model_spec_ids_by_fold: tuple[str, ...],
    filter_specs: tuple[AlphaScoreFilterSpec, ...],
    persistence_lags: tuple[int, ...] = (0, 1, 2, 5, 10, 21),
    holding_horizon_sessions: int = 1,
    minimum_paired_rows: int = 20,
) -> PanelScoreFilterEvaluationResult:
    """Evaluate declared filters and select causally without Portfolio feedback."""

    count = len(sessions)
    if (
        count == 0
        or len(listings) != count
        or len(holding_end_sessions) != count
        or fold_indices.shape != (count,)
        or any(value.shape != (count,) for value in (raw_scores, raw_simple_returns))
        or any(value.flags.writeable for value in (raw_scores, raw_simple_returns))
        or not all(np.isfinite(value).all() for value in (raw_scores, raw_simple_returns))
        or not filter_specs
        or len(upstream_raw_score_artifact_hashes) != len(model_spec_ids_by_fold)
        or any(len(value) != 64 for value in upstream_raw_score_artifact_hashes)
        or len({value.spec_id for value in filter_specs}) != len(filter_specs)
        or tuple(sorted(set(persistence_lags))) != persistence_lags
        or not persistence_lags
        or persistence_lags[0] != 0
        or minimum_paired_rows < 2
    ):
        raise PanelStatisticError("alpha_research.score_filter_evaluation_input_invalid")
    folds = tuple(sorted(set(int(value) for value in fold_indices.tolist())))
    if folds != tuple(range(len(model_spec_ids_by_fold))):
        raise PanelStatisticError("alpha_research.score_filter_fold_axis_invalid")
    session_axis = tuple(sorted(set(sessions)))
    listing_axis = tuple(sorted(set(listings)))
    session_position = {value: index for index, value in enumerate(session_axis)}
    listing_position = {value: index for index, value in enumerate(listing_axis)}
    persistence_session_codes: IntArray = np.fromiter(
        (session_position[value] for value in sessions), dtype=np.int64, count=count
    )
    persistence_listing_codes: IntArray = np.fromiter(
        (listing_position[value] for value in listings), dtype=np.int64, count=count
    )
    persistence_row_index_panel: IntArray = np.full(
        (len(session_axis), len(listing_axis)), -1, dtype=np.int64
    )
    persistence_row_index_panel[persistence_session_codes, persistence_listing_codes] = np.arange(
        count, dtype=np.int64
    )
    if int((persistence_row_index_panel >= 0).sum()) != count:
        raise PanelStatisticError("alpha_research.score_filter_row_duplicated")
    filtered: dict[str, FloatArray] = {}
    evidence: list[PanelScoreFilterEvidence] = []
    for spec in filter_specs:
        values, _realized_depth = apply_alpha_score_filter(
            spec=spec,
            row_sessions=sessions,
            row_listing_ids=listings,
            scores=raw_scores,
        )
        filtered[spec.spec_id] = values
        evidence.append(
            _score_filter_evidence(
                spec=spec,
                upstream_raw_score_artifact_hashes=upstream_raw_score_artifact_hashes,
                model_spec_ids_by_fold=model_spec_ids_by_fold,
                sessions=sessions,
                listings=listings,
                fold_indices=fold_indices,
                scores=values,
                raw_simple_returns=raw_simple_returns,
                persistence_lags=persistence_lags,
                minimum_paired_rows=minimum_paired_rows,
                persistence_session_codes=persistence_session_codes,
                persistence_listing_codes=persistence_listing_codes,
                persistence_row_index_panel=persistence_row_index_panel,
            )
        )
    frozen_evidence = tuple(evidence)
    selections: list[PanelScoreFilterSelection] = []
    for fold_index in folds:
        fold_sessions = tuple(
            session
            for session, row_fold in zip(sessions, fold_indices, strict=True)
            if int(row_fold) == fold_index
        )
        if not fold_sessions:
            raise PanelStatisticError("alpha_research.score_filter_fold_empty")
        selections.append(
            _select_score_filter_for_fold(
                fold_index=fold_index,
                first_formation_session=min(fold_sessions),
                sessions=sessions,
                holding_end_sessions=holding_end_sessions,
                fold_indices=fold_indices,
                evidence=frozen_evidence,
                holding_horizon_sessions=holding_horizon_sessions,
            )
        )
    selection_by_fold = {value.fold_index: value for value in selections}
    selected: FloatArray = np.empty(count, dtype=np.float64)
    for position, row_fold in enumerate(fold_indices):
        selected[position] = filtered[
            selection_by_fold[int(row_fold)].operational_fallback_spec_id
        ][position]
    selected = np.ascontiguousarray(selected, dtype=np.float64)
    selected.setflags(write=False)
    return PanelScoreFilterEvaluationResult(
        evidence=frozen_evidence,
        selections=tuple(selections),
        filtered_scores_by_spec_id=MappingProxyType(dict(filtered)),
        selected_scores=selected,
    )


__all__ = [
    "AlphaActivePathMetrics",
    "AlphaEconomicCandidateEvidence",
    "AlphaEconomicCandidateInput",
    "AlphaEconomicPairedComparison",
    "AlphaEconomicPairedEndpoint",
    "AlphaEconomicPathMetrics",
    "CandidateComplexity",
    "CausalCandidateSelection",
    "CausalModelSpecSelection",
    "CausalScoreAggregationSelection",
    "Disposition",
    "LongOnlySessionEvidence",
    "PanelAlphaEconomicEvaluation",
    "PanelModelScientificSelection",
    "PanelScientificSessionEvidence",
    "PanelScoreFilterCandidateCriterion",
    "PanelScoreFilterEvaluationResult",
    "PanelScoreFilterEvidence",
    "PanelScoreFilterSelection",
    "PanelScorePersistencePoint",
    "PanelStatisticError",
    "circular_block_interval",
    "dependence_aware_block_length",
    "evaluate_fixed_recipe_alpha_economics",
    "evaluate_panel_score_filters",
    "fixed_recipe_economic_metric_call_count",
    "panel_scientific_session_evidence",
    "select_panel_model_scientifically",
]
