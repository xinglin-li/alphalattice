"""Durable evidence contracts for paired Alpha Portfolio research."""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

from pydantic import Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioStateTransitionBinding,
    RebalanceClockBinding,
)
from alphalattice.capabilities.portfolio_backtesting.metrics import (
    PortfolioEconomicMetricSet,
)
from alphalattice.investment.portfolio_strategy_lab.contracts import (
    PortfolioLabContract,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_HASH = r"^[0-9a-f]{64}$"

type PortfolioEffectDimension = Literal[
    "R0_HYPERPARAMETER_SELECTION",
    "R1_BEST_ACHIEVABLE_DIAGNOSTIC",
    "POLICY_EFFECT",
    "RISK_EFFECT",
    "SECTOR_CAPACITY_EFFECT",
    "SECONDARY_INTERACTION",
    "DYNAMIC_CONTROL",
    "SIGNAL_EFFECT",
    "SCORE_FILTER_EFFECT",
    "RANK_BUFFERED_TURNOVER_SELECTION",
]


class PairedPortfolioError(ValueError):
    """Stable fail-closed boundary for installed paired Portfolio research."""


def _canonical_model_identity(
    value: PortfolioLabContract,
    *,
    identity_field: str,
    exclude_none: bool = False,
) -> str:
    """Hash the exact canonical payload used by the durable constructor."""

    return str(
        canonical_hash(
            value.model_dump(
                mode="json",
                exclude={identity_field},
                exclude_none=exclude_none,
            )
        )
    )


def _trial_identity(value: PairedPortfolioTrialEvidence) -> str:
    """Preserve historical trial hashes while current cadence bindings seal."""

    payload = value.model_dump(mode="json", exclude={"evidence_hash"})
    if "rebalance_clock_binding" not in value.model_fields_set:
        payload.pop("rebalance_clock_binding", None)
    return str(canonical_hash(payload))


class PortfolioObjectiveAuditSummary(PortfolioLabContract):
    """Retain objective means, sector displacement and session audit identity.

    Retain mean objective terms, hard-sector displacement and packed per-session audit identity.
    """

    alpha_utility_term_mean: float
    risk_penalty_term_mean: float = Field(ge=0.0)
    transaction_cost_term_mean: float = Field(ge=0.0)
    turnover_regularization_term_mean: float = Field(ge=0.0)
    sector_penalty_status: Literal["NOT_APPLICABLE", "SOFT_PENALTY_INSTALLED"]
    sector_penalty_term_mean: float | None = Field(default=None, ge=0.0)
    reference_to_sector_unconstrained_l1_mean: float = Field(ge=0.0)
    reference_to_final_l1_mean: float = Field(ge=0.0)
    sector_unconstrained_to_final_l1_mean: float = Field(ge=0.0)
    binding_session_count: int = Field(ge=0)
    binding_session_frequency: float = Field(ge=0.0, le=1.0)
    minimum_sector_slack: float | None
    mean_absolute_sector_dual: float | None = Field(default=None, ge=0.0)
    mean_displaced_name_count: float = Field(ge=0.0)
    mean_displaced_weight_mass: float = Field(ge=0.0)
    pre_constraint_score_utility_mean: float
    post_constraint_score_utility_mean: float
    pre_constraint_predicted_risk: float = Field(ge=0.0)
    post_constraint_predicted_risk: float = Field(ge=0.0)
    mean_realized_return_impact: float
    packed_session_audit_hash: str = Field(pattern=_HASH)


class PairedPortfolioTrialEvidence(PortfolioLabContract):
    """Bind paired trial axes, economic metrics, allocation lanes and exact numerical authority."""

    kind: Literal["PairedPortfolioTrialEvidence"] = "PairedPortfolioTrialEvidence"
    program_hash: str = Field(pattern=_HASH)
    phase: Literal["INNER_RISK_SELECTION", "OUTER_DEVELOPMENT_EVALUATION"]
    arm_id: str = Field(min_length=1)
    effect_dimension: PortfolioEffectDimension
    score_surface_hash: str = Field(pattern=_HASH)
    risk_method_id: Literal["R0", "R1"]
    risk_surface_hash: str = Field(pattern=_HASH)
    policy_id: str
    policy_recipe_hash: str = Field(pattern=_HASH)
    rebalance_clock_binding: RebalanceClockBinding | None = None
    alpha_utility_units_admission: Literal[
        "NOT_APPLICABLE", "DIMENSIONLESS_SCORE_UTILITY_IDENTITY_BOUND"
    ]
    alpha_utility_admission_hash: str | None = Field(default=None, pattern=_HASH)
    sector_capacity: float | None = Field(default=None, gt=0.0, le=1.0)
    executed_sessions: tuple[date, ...] = Field(min_length=2)
    evaluated_sessions: tuple[date, ...] = Field(min_length=2)
    decision_session_count: int = Field(ge=2)
    economic_session_count: int = Field(ge=2)
    passive_hold_sessions: tuple[date, ...] = ()
    passive_optimizer_call_count: Literal[0] = 0
    passive_target_decision_count: Literal[0] = 0
    cost_metrics: tuple[PortfolioEconomicMetricSet, ...] = Field(min_length=5, max_length=5)
    predicted_risk: float = Field(ge=0.0)
    realized_risk: float = Field(ge=0.0)
    realized_to_predicted_variance_ratio: float | None = Field(default=None, ge=0.0)
    variance_calibration_support_count: int = Field(ge=0)
    variance_calibration_excluded_count: int = Field(ge=0)
    mean_one_way_turnover: float = Field(ge=0.0)
    break_even_cost_bps: float | None
    anchor_active_log_wealth_5bps: float
    fold_stability_net_5bps: tuple[tuple[int, float], ...]
    half_stability_net_5bps: tuple[float, float]
    missing_execution_count: int = Field(ge=0)
    solver_call_count: int = Field(ge=0)
    objective_audit: PortfolioObjectiveAuditSummary | None
    target_weight_lane_hash: str = Field(pattern=_HASH)
    reference_weight_lane_hash: str | None = Field(default=None, pattern=_HASH)
    path_lane_hash: str = Field(pattern=_HASH)
    evidence_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal paired trial evidence through its declared trial-identity projection.

        Args:
            values: Explicit trial fields excluding the generated evidence_hash.

        Returns:
            Validated paired trial evidence with generated identity; compatibility follows the trial
            identity owner.
        """
        provisional = cls.model_construct(**values, evidence_hash="0" * 64)
        return cls(
            **values,
            evidence_hash=_trial_identity(provisional),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_axes(self) -> Self:
        """Require reconciled decision/economic/passive and variance-calibration support.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PairedPortfolioError: Axis counts/passive membership differ, or a trial carrying
                reference-weight identity has a mismatched evidence_hash.
        """
        if (
            self.economic_session_count != len(self.executed_sessions)
            or self.decision_session_count + len(self.passive_hold_sessions)
            != self.economic_session_count
            or not set(self.passive_hold_sessions) <= set(self.executed_sessions)
            or self.variance_calibration_support_count + self.variance_calibration_excluded_count
            != len(self.evaluated_sessions)
        ):
            raise PairedPortfolioError("portfolio_strategy_lab.trial_economic_axis_invalid")
        if self.reference_weight_lane_hash is not None and self.evidence_hash != _trial_identity(
            self
        ):
            raise PairedPortfolioError("portfolio_strategy_lab.trial_identity_invalid")
        return self


class PortfolioGoldenOracleEvidence(PortfolioLabContract):
    """Bind solver agreement to weight, objective and feasibility tolerances.

    Bind direct/independent solver agreement to declared weight, objective and feasibility
    tolerances.
    """

    kind: Literal["PortfolioGoldenOracleEvidence"] = "PortfolioGoldenOracleEvidence"
    program_hash: str = Field(pattern=_HASH)
    production_owner: Literal["DIRECT_OSQP"] = "DIRECT_OSQP"
    oracle_owner: Literal["CVXPY_OSQP"] = "CVXPY_OSQP"
    calibration_sessions: tuple[date, ...] = Field(min_length=1)
    validation_sessions: tuple[date, ...] = Field(min_length=1)
    maximum_weight_tolerance: float = Field(gt=0.0)
    objective_tolerance: float = Field(gt=0.0)
    feasibility_tolerance: float = Field(gt=0.0)
    validation_maximum_weight_error: float = Field(ge=0.0)
    validation_maximum_objective_error: float = Field(ge=0.0)
    validation_maximum_feasibility_residual: float = Field(ge=0.0)
    direct_osqp_call_count: int = Field(ge=5, le=10)
    cvxpy_osqp_call_count: Literal[5] = 5
    admitted: bool
    evidence_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one golden solver oracle evidence record.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical evidence_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        provisional = cls.model_construct(**values, evidence_hash="0" * 64)
        return cls(
            **values,
            evidence_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"evidence_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_admission(self) -> Self:
        """Require oracle admission to match all three declared error tolerances.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PairedPortfolioError: admitted differs from weight/objective/feasibility tolerance
                comparison.
        """
        expected = (
            self.validation_maximum_weight_error <= self.maximum_weight_tolerance
            and self.validation_maximum_objective_error <= self.objective_tolerance
            and self.validation_maximum_feasibility_residual <= self.feasibility_tolerance
        )
        if self.admitted != expected:
            raise PairedPortfolioError("portfolio_strategy_lab.golden_admission_invalid")
        return self


class ControlledCovarianceRegressionEvidence(PortfolioLabContract):
    """Bind a controlled covariance change to nonzero allocation/objective movement."""

    kind: Literal["ControlledCovarianceRegressionEvidence"] = (
        "ControlledCovarianceRegressionEvidence"
    )
    program_hash: str = Field(pattern=_HASH)
    session: date
    source_covariance_hash: str = Field(pattern=_HASH)
    alternate_covariance_hash: str = Field(pattern=_HASH)
    objective_difference: float = Field(gt=0.0)
    maximum_weight_difference: float = Field(gt=0.0)
    direct_osqp_call_count: int = Field(ge=2, le=4)
    covariance_consumed: Literal[True] = True
    evidence_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one controlled covariance regression.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical evidence_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        provisional = cls.model_construct(**values, evidence_hash="0" * 64)
        return cls(
            **values,
            evidence_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"evidence_hash"}))
            ),
        )


class PortfolioPairedDecisionEvidence(PortfolioLabContract):
    """Bind aligned trial differences, paired interval and temporal stability disposition."""

    comparison_id: str = Field(min_length=1)
    left_trial_hash: str = Field(pattern=_HASH)
    right_trial_hash: str = Field(pattern=_HASH)
    common_sessions: tuple[date, ...] = Field(min_length=2)
    point_difference: float
    paired_interval: tuple[float, float]
    block_length: int = Field(ge=1)
    fraction_sessions_left_won: float = Field(ge=0.0, le=1.0)
    fold_stability: tuple[float, ...]
    half_stability: tuple[float, float]
    disposition: Literal[
        "SUPERIOR_PAIRED_EVIDENCE",
        "INFERIOR_PAIRED_EVIDENCE",
        "INCONCLUSIVE_PAIRED_EVIDENCE",
    ]
    decision_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one paired trial decision.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical decision_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        provisional = cls.model_construct(**values, decision_hash="0" * 64)
        return cls(
            **values,
            decision_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"decision_hash"}))
            ),
        )


class PairedAlphaPortfolioResearchRoot(PortfolioLabContract):
    """Bind development-only score/Risk experiments and ordered evidence.

    Bind development-only score/Risk experiments and ordered evidence without public pointer
    authority.
    """

    kind: Literal["PairedAlphaPortfolioResearchRoot"] = "PairedAlphaPortfolioResearchRoot"
    identity_class: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    program_hash: str = Field(pattern=_HASH)
    input_binding_hash: str = Field(pattern=_HASH)
    dynamic_score_surface_hash: str = Field(pattern=_HASH)
    control_score_surface_hash: str | None = Field(default=None, pattern=_HASH)
    raw_dynamic_score_surface_hash: str | None = Field(default=None, pattern=_HASH)
    fixed_simple_score_surface_hash: str | None = Field(default=None, pattern=_HASH)
    fixed_simple_score_binding_hash: str | None = Field(default=None, pattern=_HASH)
    declared_filter_surface_hashes: tuple[str, ...] = ()
    r0_surface_hash: str = Field(pattern=_HASH)
    r1_surface_hash: str | None = Field(default=None, pattern=_HASH)
    benchmark_surface_hash: str = Field(pattern=_HASH)
    # Absent only on historical roots published before common temporal
    # admission. Every newly created paired root supplies all three.
    alpha_score_temporal_handoff_hash: str | None = Field(default=None, pattern=_HASH)
    strategy_schedule_hash: str | None = Field(default=None, pattern=_HASH)
    causal_admission_hash: str | None = Field(default=None, pattern=_HASH)
    state_transition: PortfolioStateTransitionBinding | None = None
    execution_events_hash: str | None = Field(default=None, pattern=_HASH)
    inner_selection_stop: int = Field(ge=1)
    maturity_purge_sessions: Literal[1] = 1
    evaluation_start: int = Field(ge=0)
    selected_r0_policy_recipe_hash: str = Field(pattern=_HASH)
    selected_r1_best_achievable_recipe_hash: str | None = Field(default=None, pattern=_HASH)
    selection_study_type: Literal["RANK_BUFFERED_R0_INNER"] | None = None
    ordered_trial_hashes: tuple[str, ...] = Field(min_length=1)
    golden_oracle_evidence_hash: str = Field(pattern=_HASH)
    controlled_covariance_evidence_hash: str = Field(pattern=_HASH)
    ordered_paired_decision_hashes: tuple[str, ...] = Field(min_length=1)
    spy_metrics: PortfolioEconomicMetricSet
    risk_grid_distinct_behavior_count: int = Field(ge=1)
    risk_grid_behavior_disposition: Literal["BEHAVIORALLY_DISTINCT", "CONSTRAINT_DEGENERATE"]
    trial_direct_osqp_call_count: int = Field(gt=0)
    direct_osqp_call_count: int = Field(gt=0)
    solver_call_count: int = Field(gt=0)
    cvxpy_oracle_call_count: int = Field(gt=0)
    network_access_count: Literal[0] = 0
    provider_access_count: Literal[0] = 0
    holdout_access_count: Literal[0] = 0
    current_or_production_pointer_read_count: Literal[0] = 0
    pointer_mutation_count: Literal[0] = 0
    source_workspace_write_count: Literal[0] = 0
    disposition: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    root_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal paired development root with absent optional fields omitted from identity.

        Args:
            values: Explicit root fields excluding the generated root_hash.

        Returns:
            Validated development root through its canonical identity projection.
        """
        provisional = cls.model_construct(**values, root_hash="0" * 64)
        return cls(
            **values,
            root_hash=_canonical_model_identity(
                provisional,
                identity_field="root_hash",
                exclude_none=True,
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_authority_group(self) -> Self:
        """Require complete temporal/transition groups and coherent paired Risk study authority.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PairedPortfolioError: Optional authority groups are partial, rank-buffered/R1 study
                disposition differs, or a transition-bound root identity differs.
        """
        temporal = (
            self.alpha_score_temporal_handoff_hash,
            self.strategy_schedule_hash,
            self.causal_admission_hash,
        )
        transition = (self.state_transition, self.execution_events_hash)
        if any(value is None for value in temporal) != all(
            value is None for value in temporal
        ) or any(value is None for value in transition) != all(
            value is None for value in transition
        ):
            raise PairedPortfolioError(
                "portfolio_strategy_lab.paired_root_authority_group_incomplete"
            )
        rank_buffered_inner = self.selection_study_type == "RANK_BUFFERED_R0_INNER"
        if rank_buffered_inner != (
            self.r1_surface_hash is None and self.selected_r1_best_achievable_recipe_hash is None
        ):
            raise PairedPortfolioError("portfolio_strategy_lab.paired_root_risk_study_invalid")
        if self.state_transition is not None and self.root_hash != _canonical_model_identity(
            self,
            identity_field="root_hash",
            exclude_none=True,
        ):
            raise PairedPortfolioError("portfolio_strategy_lab.paired_root_identity_invalid")
        return self


class PairedAlphaPortfolioReplayReceipt(PortfolioLabContract):
    """Strong target-readback replay over the installed causal state path."""

    kind: Literal["PairedAlphaPortfolioReplayReceipt"] = "PairedAlphaPortfolioReplayReceipt"
    identity_class: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    program_hash: str = Field(pattern=_HASH)
    root_hash: str = Field(pattern=_HASH)
    input_binding_hash: str = Field(pattern=_HASH)
    state_transition_binding_hash: str = Field(pattern=_HASH)
    session_mark_surface_hash: str = Field(pattern=_HASH)
    execution_events_hash: str = Field(pattern=_HASH)
    verified_trial_count: int = Field(ge=1)
    rederived_decision_count: int = Field(ge=1)
    rederived_economic_record_count: int = Field(ge=1)
    rederived_metric_set_count: int = Field(ge=1)
    fit_call_count: Literal[0] = 0
    predict_call_count: Literal[0] = 0
    optimizer_call_count: Literal[0] = 0
    solver_call_count: Literal[0] = 0
    risk_estimate_call_count: Literal[0] = 0
    disposition: Literal["NUMERICAL_PATH_REDERIVED_SOLVER_DECISION_READBACK"] = (
        "NUMERICAL_PATH_REDERIVED_SOLVER_DECISION_READBACK"
    )
    receipt_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal paired graph replay evidence through the receipt identity owner.

        Args:
            values: Explicit receipt fields excluding generated receipt_hash.

        Returns:
            Validated replay receipt with declared canonical identity projection.
        """
        provisional = cls.model_construct(**values, receipt_hash="0" * 64)
        return cls(
            **values,
            receipt_hash=_canonical_model_identity(
                provisional,
                identity_field="receipt_hash",
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact paired replay receipt identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PairedPortfolioError: receipt_hash differs.
        """
        if self.receipt_hash != _canonical_model_identity(
            self,
            identity_field="receipt_hash",
        ):
            raise PairedPortfolioError("portfolio_strategy_lab.paired_replay_identity_invalid")
        return self


__all__ = [
    "ControlledCovarianceRegressionEvidence",
    "PairedAlphaPortfolioReplayReceipt",
    "PairedAlphaPortfolioResearchRoot",
    "PairedPortfolioError",
    "PairedPortfolioTrialEvidence",
    "PortfolioEffectDimension",
    "PortfolioGoldenOracleEvidence",
    "PortfolioObjectiveAuditSummary",
    "PortfolioPairedDecisionEvidence",
]
