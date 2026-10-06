"""Typed scientific and publication contracts for Portfolio development research."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import PortfolioTrialMetrics
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model_validated, validate_hash_compatible

_HASH = r"^[0-9a-f]{64}$"
_DEVELOPMENT_COUNT = 494


class PortfolioLabContract(BaseModel):  # type: ignore[misc]
    """Provide immutable portfolio research contracts that forbid undeclared fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PortfolioPolicyFamily(StrEnum):
    """Name the installed equal-weight, variance, score/Risk/cost and sector policy families."""

    TOP_K_EQUAL_WEIGHT = "TOP_K_EQUAL_WEIGHT"
    TOP_K_MINIMUM_VARIANCE = "TOP_K_MINIMUM_VARIANCE"
    TOP_K_SCORE_RISK_COST = "TOP_K_SCORE_RISK_COST"
    SECTOR_DEVIATION_PENALTY = "SECTOR_DEVIATION_PENALTY"


class PortfolioScoreMode(StrEnum):
    """Declare stock scores alone or stock scores with the sector component."""

    STOCK_ONLY = "STOCK_ONLY"
    STOCK_PLUS_SECTOR_COMPONENT = "STOCK_PLUS_SECTOR_COMPONENT"


class TrialState(StrEnum):
    """Record asked, duplicate, invalid, failed or completed trial disposition."""

    ASKED = "ASKED"
    DUPLICATE = "DUPLICATE"
    INVALID = "INVALID"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class PortfolioResearchOutcome(StrEnum):
    """Declare candidate acceptance, structural experiment, owner routing or scientific stop."""

    ACCEPT_POLICY_CANDIDATES = "ACCEPT_POLICY_CANDIDATES"
    RUN_SECTOR_DEVIATION_EXPERIMENT = "RUN_SECTOR_DEVIATION_EXPERIMENT"
    ROUTE_TO_ALPHA = "ROUTE_TO_ALPHA"
    ROUTE_TO_RISK = "ROUTE_TO_RISK"
    REQUEST_NEW_MANDATE = "REQUEST_NEW_MANDATE"
    STOP_WITH_EVIDENCE = "STOP_WITH_EVIDENCE"


class TopKEqualWeightPolicy(PortfolioLabContract):
    """Declare top-k equal weighting with an explicit maximum single-name weight."""

    family: Literal[PortfolioPolicyFamily.TOP_K_EQUAL_WEIGHT] = Field(
        default=PortfolioPolicyFamily.TOP_K_EQUAL_WEIGHT, description="Top-k names at equal weight."
    )
    top_k: int = Field(
        ge=1, description="How many of the best-scored eligible names the book holds."
    )
    maximum_weight: float = Field(
        gt=0.0, le=1.0, description="The largest weight one name may hold."
    )

    @property
    def policy_id(self) -> str:
        """Read the installed policy family identity.

        Returns:
            String value of the retained family enum.
        """
        return str(self.family.value)


class TopKMinimumVariancePolicy(PortfolioLabContract):
    """Declare top-k covariance minimization with an explicit maximum single-name weight."""

    family: Literal[PortfolioPolicyFamily.TOP_K_MINIMUM_VARIANCE] = Field(
        default=PortfolioPolicyFamily.TOP_K_MINIMUM_VARIANCE,
        description="Top-k names at the weights of least predicted variance.",
    )
    top_k: int = Field(
        ge=1, description="How many of the best-scored eligible names the book holds."
    )
    maximum_weight: float = Field(
        gt=0.0, le=1.0, description="The largest weight one name may hold."
    )

    @property
    def policy_id(self) -> str:
        """Read the installed policy family identity.

        Returns:
            String value of the retained family enum.
        """
        return str(self.family.value)


class TopKScoreRiskCostPolicy(PortfolioLabContract):
    """Declare top-k score utility with positive Risk aversion and turnover regularization."""

    family: Literal[PortfolioPolicyFamily.TOP_K_SCORE_RISK_COST] = Field(
        default=PortfolioPolicyFamily.TOP_K_SCORE_RISK_COST,
        description="Top-k names weighed by score against predicted risk and cost.",
    )
    top_k: int = Field(
        ge=1, description="How many of the best-scored eligible names the book holds."
    )
    maximum_weight: float = Field(
        gt=0.0, le=1.0, description="The largest weight one name may hold."
    )
    risk_aversion: float = Field(
        gt=0.0, description="What the objective charges for predicted variance."
    )
    turnover_regularization: float = Field(
        ge=0.0, description="What the objective charges for turning over the held book."
    )

    @property
    def policy_id(self) -> str:
        """Read the installed policy family identity.

        Returns:
            String value of the retained family enum.
        """
        return str(self.family.value)


class SectorDeviationPenaltyPolicy(PortfolioLabContract):
    """Declare score/Risk/cost optimization with a positive sector deviation penalty."""

    family: Literal[PortfolioPolicyFamily.SECTOR_DEVIATION_PENALTY] = Field(
        default=PortfolioPolicyFamily.SECTOR_DEVIATION_PENALTY,
        description="Score, risk and cost, with sector deviations charged.",
    )
    top_k: int = Field(
        ge=1, description="How many of the best-scored eligible names the book holds."
    )
    maximum_weight: float = Field(
        gt=0.0, le=1.0, description="The largest weight one name may hold."
    )
    risk_aversion: float = Field(
        gt=0.0, description="What the objective charges for predicted variance."
    )
    turnover_regularization: float = Field(
        ge=0.0, description="What the objective charges for turning over the held book."
    )
    sector_deviation_penalty: float = Field(
        gt=0.0,
        description="What the objective charges for sector weights away from the equal-weight "
        "book's.",
    )

    @property
    def policy_id(self) -> str:
        """Read the installed policy family identity.

        Returns:
            String value of the retained family enum.
        """
        return str(self.family.value)


PortfolioPolicySpec = Annotated[
    TopKEqualWeightPolicy
    | TopKMinimumVariancePolicy
    | TopKScoreRiskCostPolicy
    | SectorDeviationPenaltyPolicy,
    Field(discriminator="family"),
]


class PortfolioStrategyLabMandate(PortfolioLabContract):
    """Frozen development-only authority and search budget."""

    kind: Literal["PortfolioStrategyLabMandate"] = "PortfolioStrategyLabMandate"
    portfolio_input_bundle_hash: str = Field(pattern=_HASH)
    portfolio_development_mandate_hash: str = Field(pattern=_HASH)
    alpha_score_surface_hash: str = Field(pattern=_HASH)
    risk_surface_hash: str = Field(pattern=_HASH)
    tradability_bundle_hash: str = Field(pattern=_HASH)
    universe_epoch_hash: str = Field(pattern=_HASH)
    ordered_listing_ids_hash: str = Field(pattern=_HASH)
    ordered_candidate_ids: tuple[str, ...] = Field(min_length=1)
    score_modes: tuple[PortfolioScoreMode, ...] = Field(min_length=1)
    development_formation_sessions: tuple[date, ...] = Field(
        min_length=_DEVELOPMENT_COUNT, max_length=_DEVELOPMENT_COUNT
    )
    policy_oos_state: Literal["LOCKED"] = "LOCKED"
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    formation_semantics: Literal["T_CLOSE"] = "T_CLOSE"
    intended_execution_semantics: Literal["NEXT_COMMON_SESSION_OPEN"] = "NEXT_COMMON_SESSION_OPEN"
    holding_end_semantics: Literal["FOLLOWING_COMMON_SESSION_OPEN"] = (
        "FOLLOWING_COMMON_SESSION_OPEN"
    )
    long_only: Literal[True] = True
    fully_invested_target: Literal[True] = True
    baseline_top_k: Literal[50] = 50
    search_top_k: tuple[Literal[20, 30, 50, 75, 100], ...] = (20, 30, 50, 75, 100)
    selection_cost_bps: Literal[10] = 10
    reporting_cost_bps: tuple[Literal[5, 10, 20], ...] = (5, 10, 20)
    initial_search_trials_per_stratum: Literal[24] = 24
    structural_trials_per_stratum: Literal[12] = 12
    bootstrap_resamples: Literal[2000] = 2000
    sampler_seed: Literal[20260811] = 20260811
    limitations: tuple[str, ...] = Field(min_length=1)
    mandate_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_mandate(self) -> Self:
        """Require ordered development support, unique candidates and the declared search design.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Development dates, candidate ordering, admitted score-mode sequence, fixed
                top-k grid or mandate_hash differs.
        """
        if self.development_formation_sessions != tuple(
            sorted(set(self.development_formation_sessions))
        ):
            raise ValueError("portfolio_strategy_lab.development_axis_invalid")
        if self.ordered_candidate_ids != tuple(dict.fromkeys(self.ordered_candidate_ids)):
            raise ValueError("portfolio_strategy_lab.candidate_axis_invalid")
        if self.score_modes not in {
            (PortfolioScoreMode.STOCK_ONLY,),
            (
                PortfolioScoreMode.STOCK_ONLY,
                PortfolioScoreMode.STOCK_PLUS_SECTOR_COMPONENT,
            ),
        }:
            raise ValueError("portfolio_strategy_lab.score_mode_axis_invalid")
        if self.search_top_k != (20, 30, 50, 75, 100):
            raise ValueError("portfolio_strategy_lab.search_domain_invalid")
        _validate_hash(self, "mandate_hash")
        return self


class PortfolioExperimentStratum(PortfolioLabContract):
    """Identify one candidate and score-mode stratum in the experiment program."""

    candidate_id: str = Field(min_length=1)
    score_mode: PortfolioScoreMode
    stratum_id: str = Field(min_length=1)


class PortfolioExperimentProgram(PortfolioLabContract):
    """Bind ordered strata, baselines, attempt budgets and exact numerical execution authority."""

    kind: Literal["PortfolioExperimentProgram"] = "PortfolioExperimentProgram"
    mandate_hash: str = Field(pattern=_HASH)
    strata: tuple[PortfolioExperimentStratum, ...] = Field(min_length=1)
    baseline_specs: tuple[PortfolioPolicySpec, ...] = Field(min_length=3)
    initial_attempt_budget: int = Field(ge=1)
    structural_attempt_budget: int = Field(ge=1)
    execution_binding_hash: str = Field(pattern=_HASH)
    numerical_environment_hash: str = Field(pattern=_HASH)
    program_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_program(self) -> Self:
        """Require unique strata and exact initial/structural budgets before sealing.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Stratum identity repeats, budget is not respectively 24/12 attempts per
                stratum or program_hash differs.
        """
        if len({value.stratum_id for value in self.strata}) != len(self.strata):
            raise ValueError("portfolio_strategy_lab.stratum_axis_invalid")
        if self.initial_attempt_budget != 24 * len(self.strata):
            raise ValueError("portfolio_strategy_lab.initial_budget_invalid")
        if self.structural_attempt_budget != 12 * len(self.strata):
            raise ValueError("portfolio_strategy_lab.structural_budget_invalid")
        _validate_hash(self, "program_hash")
        return self


class PortfolioSolverRuntimeEvidence(PortfolioLabContract):
    """Bind solver/oracle agreement and declared latency admission evidence to a program."""

    kind: Literal["PortfolioSolverRuntimeEvidence"] = "PortfolioSolverRuntimeEvidence"
    program_hash: str = Field(pattern=_HASH)
    oracle_maximum_weight_difference: float = Field(ge=0.0)
    oracle_objective_difference: float = Field(ge=0.0)
    k100_solve_p95_seconds: float = Field(gt=0.0)
    projected_trial_seconds: float = Field(gt=0.0)
    production_owner: Literal["DIRECT_OSQP"] = "DIRECT_OSQP"
    oracle_owner: Literal["CVXPY_OSQP"] = "CVXPY_OSQP"
    admitted: bool
    evidence_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_evidence(self) -> Self:
        """Require admission to match solver agreement and latency thresholds.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Admission contradicts both 1e-8 oracle tolerances, 0.030-second solve p95 or
                20-second projected trial limit, or evidence_hash differs.
        """
        expected = (
            self.oracle_maximum_weight_difference <= 1e-8
            and self.oracle_objective_difference <= 1e-8
            and self.k100_solve_p95_seconds <= 0.030
            and self.projected_trial_seconds <= 20.0
        )
        if self.admitted != expected:
            raise ValueError("portfolio_strategy_lab.solver_runtime_admission_invalid")
        _validate_hash(self, "evidence_hash")
        return self


class PortfolioTrialEvidence(PortfolioLabContract):
    """Bind trial disposition, completed metrics/return paths and failure evidence."""

    kind: Literal["PortfolioTrialEvidence"] = "PortfolioTrialEvidence"
    program_hash: str = Field(pattern=_HASH)
    stratum_id: str = Field(min_length=1)
    trial_number: int = Field(ge=0)
    policy: PortfolioPolicySpec
    search_parameters: dict[str, float | int] = Field(default_factory=dict)
    state: TrialState
    failure_code: str | None = None
    metrics: PortfolioTrialMetrics | None = None
    gross_log_returns: tuple[float, ...] = ()
    net_log_returns_5bps: tuple[float, ...] = ()
    net_log_returns_10bps: tuple[float, ...] = ()
    net_log_returns_20bps: tuple[float, ...] = ()
    annual_net_log_returns: tuple[tuple[int, float], ...] = ()
    chronological_block_net_log_returns: tuple[float, ...] = ()
    bootstrap_probability_net_positive: float | None = Field(default=None, ge=0.0, le=1.0)
    sector_exposure_summary: dict[str, float] = Field(default_factory=dict)
    evidence_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_evidence(self) -> Self:
        """Require completed metrics and full return paths or an explicit failure.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Completion differs from metrics/full declared development paths, completed
                failure is present, noncompleted failure is absent or evidence_hash differs.
        """
        completed = self.state is TrialState.COMPLETED
        paths = (
            self.gross_log_returns,
            self.net_log_returns_5bps,
            self.net_log_returns_10bps,
            self.net_log_returns_20bps,
        )
        if completed != (self.metrics is not None) or completed != all(
            len(value) == _DEVELOPMENT_COUNT for value in paths
        ):
            raise ValueError("portfolio_strategy_lab.trial_evidence_incomplete")
        if completed and self.failure_code is not None:
            raise ValueError("portfolio_strategy_lab.trial_failure_invalid")
        if not completed and self.failure_code is None:
            raise ValueError("portfolio_strategy_lab.trial_failure_missing")
        _validate_hash(self, "evidence_hash")
        return self


class PortfolioTrialLedgerEntry(PortfolioLabContract):
    """Record one asked policy and its completion objectives or failure disposition."""

    stratum_id: str = Field(min_length=1)
    trial_number: int = Field(ge=0)
    policy: PortfolioPolicySpec
    search_parameters: dict[str, float | int] = Field(default_factory=dict)
    state: TrialState
    evidence_hash: str | None = Field(default=None, pattern=_HASH)
    failure_code: str | None = None
    objective_net_log_wealth: float | None = None
    objective_annualized_volatility: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_entry(self) -> Self:
        """Require completed objectives and evidence without a failure code.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Completion differs from presence of all three evidence/objective fields or a
                completed entry retains failure.
        """
        complete = self.state is TrialState.COMPLETED
        objectives = (
            self.evidence_hash,
            self.objective_net_log_wealth,
            self.objective_annualized_volatility,
        )
        if complete != all(value is not None for value in objectives):
            raise ValueError("portfolio_strategy_lab.ledger_entry_incomplete")
        if complete and self.failure_code is not None:
            raise ValueError("portfolio_strategy_lab.ledger_failure_invalid")
        return self


class PortfolioTrialLedger(PortfolioLabContract):
    """Bind unique stratum/trial entries to an explicit phase and attempt budget."""

    kind: Literal["PortfolioTrialLedger"] = "PortfolioTrialLedger"
    program_hash: str = Field(pattern=_HASH)
    phase: Literal["INITIAL_SEARCH", "SECTOR_DEVIATION_EXPERIMENT", "COMPLETE_PROGRAM"]
    attempt_budget: int = Field(ge=1)
    entries: tuple[PortfolioTrialLedgerEntry, ...]
    ledger_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_ledger(self) -> Self:
        """Require unique trial keys within budget and complete terminal programs.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Trial keys repeat/exceed budget, terminal count is incomplete or ledger_hash
                differs.
        """
        keys = tuple((value.stratum_id, value.trial_number) for value in self.entries)
        if keys != tuple(dict.fromkeys(keys)) or len(keys) > self.attempt_budget:
            raise ValueError("portfolio_strategy_lab.trial_ledger_invalid")
        if self.phase == "COMPLETE_PROGRAM" and len(keys) != self.attempt_budget:
            raise ValueError("portfolio_strategy_lab.terminal_ledger_incomplete")
        _validate_hash(self, "ledger_hash")
        return self


class PortfolioBaselineSlate(PortfolioLabContract):
    """Bind unique baseline evidence identities and control summaries to a program."""

    kind: Literal["PortfolioBaselineSlate"] = "PortfolioBaselineSlate"
    program_hash: str = Field(pattern=_HASH)
    evidence_hashes: tuple[str, ...] = Field(min_length=3)
    control_summaries: tuple[str, ...] = Field(min_length=2)
    slate_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_slate(self) -> Self:
        """Require unique baseline evidence and exact slate identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Baseline evidence repeats or slate_hash differs.
        """
        if len(set(self.evidence_hashes)) != len(self.evidence_hashes):
            raise ValueError("portfolio_strategy_lab.baseline_slate_duplicate")
        _validate_hash(self, "slate_hash")
        return self


class PortfolioEvidenceDossier(PortfolioLabContract):
    """Bind attempt counts, region evidence, contradictions, routing and explicit limitations."""

    kind: Literal["PortfolioEvidenceDossier"] = "PortfolioEvidenceDossier"
    program_hash: str = Field(pattern=_HASH)
    baseline_count: int = Field(ge=3)
    attempted_count: int = Field(ge=1)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    pareto_region_count: int = Field(ge=0)
    robust_region_count: int = Field(ge=0)
    baseline_summary: tuple[str, ...]
    stratum_summaries: tuple[str, ...]
    contradictions: tuple[str, ...]
    falsification_ledger: tuple[str, ...]
    owner_routing_evidence: tuple[str, ...]
    limitations: tuple[str, ...] = Field(min_length=1)
    dossier_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_dossier(self) -> Self:
        """Reconcile completed and failed attempts before checking dossier identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Completed plus failed count differs from attempted count or dossier_hash
                differs.
        """
        if self.completed_count + self.failed_count != self.attempted_count:
            raise ValueError("portfolio_strategy_lab.dossier_attempts_invalid")
        _validate_hash(self, "dossier_hash")
        return self


class PortfolioResearchRFC(PortfolioLabContract):
    """Bind proposed research routing and acknowledged limitations to the evidence dossier."""

    kind: Literal["PortfolioResearchRFC"] = "PortfolioResearchRFC"
    dossier_hash: str = Field(pattern=_HASH)
    competing_hypotheses: tuple[str, ...] = Field(min_length=1, max_length=6)
    observed_anomaly_signature: str = Field(min_length=1, max_length=1200)
    proposed_outcome: PortfolioResearchOutcome
    expected_uncertainty_reduction: str = Field(min_length=1, max_length=600)
    estimated_compute_budget: str = Field(min_length=1, max_length=300)
    posterior_branching_rule: str = Field(min_length=1, max_length=1200)
    limitations_acknowledged: bool
    origin: Literal["AGENT", "HOST_FALLBACK"] = "AGENT"
    rfc_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_rfc(self) -> Self:
        """Require acknowledged limitations and compatible RFC identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Limitations are not acknowledged or rfc_hash differs.
        """
        if not self.limitations_acknowledged:
            raise ValueError("portfolio_strategy_lab.rfc_limitations_unacknowledged")
        validate_hash_compatible(self, "rfc_hash", code="portfolio_strategy_lab.identity_invalid")
        return self


class PortfolioPolicyCandidate(PortfolioLabContract):
    """Identify one evidence-backed policy with local support and Pareto tolerance."""

    stratum_id: str = Field(min_length=1)
    policy: PortfolioPolicySpec
    trial_evidence_hash: str = Field(pattern=_HASH)
    pareto_epsilon_fraction: float = Field(ge=0.0, le=0.05)
    neighborhood_support_count: int = Field(ge=1)


class PortfolioPolicyCandidateSet(PortfolioLabContract):
    """Bind unique candidate evidence to review authority and locked validation states."""

    kind: Literal["PortfolioPolicyCandidateSet"] = "PortfolioPolicyCandidateSet"
    program_hash: str = Field(pattern=_HASH)
    dossier_hash: str = Field(pattern=_HASH)
    rfc_hash: str = Field(pattern=_HASH)
    review_hash: str = Field(pattern=_HASH)
    candidates: tuple[PortfolioPolicyCandidate, ...] = Field(min_length=1)
    policy_oos_state: Literal["LOCKED"] = "LOCKED"
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    candidate_set_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_candidates(self) -> Self:
        """Require unique stratum/evidence candidate pairs and exact set identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Candidate pair repeats or candidate_set_hash differs.
        """
        keys = tuple((value.stratum_id, value.trial_evidence_hash) for value in self.candidates)
        if keys != tuple(dict.fromkeys(keys)):
            raise ValueError("portfolio_strategy_lab.candidate_set_duplicate")
        _validate_hash(self, "candidate_set_hash")
        return self


class PortfolioScientificStop(PortfolioLabContract):
    """Bind a nonacceptance research outcome and reason to reviewed evidence."""

    kind: Literal["PortfolioScientificStop"] = "PortfolioScientificStop"
    program_hash: str = Field(pattern=_HASH)
    dossier_hash: str = Field(pattern=_HASH)
    rfc_hash: str = Field(pattern=_HASH)
    review_hash: str = Field(pattern=_HASH)
    outcome: PortfolioResearchOutcome
    reason: str = Field(min_length=1, max_length=1600)
    policy_oos_state: Literal["LOCKED"] = "LOCKED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    stop_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_stop(self) -> Self:
        """Require a scientific-stop outcome distinct from candidate acceptance.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Outcome accepts policy candidates or stop_hash differs.
        """
        if self.outcome is PortfolioResearchOutcome.ACCEPT_POLICY_CANDIDATES:
            raise ValueError("portfolio_strategy_lab.scientific_stop_outcome_invalid")
        _validate_hash(self, "stop_hash")
        return self


class PortfolioResearchReview(PortfolioLabContract):
    """Bind observed/counterfactual routing, structural attempts and review completion evidence."""

    kind: Literal["PortfolioResearchReview"] = "PortfolioResearchReview"
    review_binding_hash: str | None = Field(default=None, pattern=_HASH)
    dossier_hash: str = Field(pattern=_HASH)
    rfc_hash: str = Field(pattern=_HASH)
    terminal_outcome: PortfolioResearchOutcome
    counterfactual_outcome: PortfolioResearchOutcome
    agent_changed_route: bool
    structural_experiment_executed: bool
    additional_attempt_count: int = Field(ge=0, le=72)
    summary: str = Field(min_length=1, max_length=1600)
    model_calls: int = Field(ge=0, le=8)
    executed_tool_names: tuple[str, ...]
    completion_status: Literal["COMPLETE", "AGENT_REVIEW_INCOMPLETE"] = "COMPLETE"
    reviewed_at: datetime
    review_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_review(self) -> Self:
        """Require aware review clocks and coherent route, attempt and completion evidence.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Clock, route-change flag, structural-attempt flag, current completion
                binding/tool sequence, incomplete fallback route or compatible review_hash differs.
        """
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("portfolio_strategy_lab.review_clock_invalid")
        if self.agent_changed_route != (self.terminal_outcome is not self.counterfactual_outcome):
            raise ValueError("portfolio_strategy_lab.counterfactual_comparison_invalid")
        if self.structural_experiment_executed != (self.additional_attempt_count > 0):
            raise ValueError("portfolio_strategy_lab.structural_attempt_count_invalid")
        if self.completion_status == "COMPLETE":
            legacy_review = "review_binding_hash" not in self.model_fields_set
            current_review_missing_binding = not legacy_review and self.review_binding_hash is None
            if current_review_missing_binding or self.executed_tool_names != (
                "submit_portfolio_research_rfc",
                "submit_portfolio_research_outcome",
            ):
                raise ValueError("portfolio_strategy_lab.complete_review_invalid")
        elif self.agent_changed_route or self.terminal_outcome is not self.counterfactual_outcome:
            raise ValueError("portfolio_strategy_lab.incomplete_review_fallback_invalid")
        validate_hash_compatible(
            self, "review_hash", code="portfolio_strategy_lab.identity_invalid"
        )
        return self


class PortfolioResearchProjectionBundle(PortfolioLabContract):
    """Bind terminal research artifacts and counts to coherent validation-readiness status."""

    kind: Literal["PortfolioResearchProjectionBundle"] = "PortfolioResearchProjectionBundle"
    program_hash: str = Field(pattern=_HASH)
    ledger_hash: str = Field(pattern=_HASH)
    dossier_hash: str = Field(pattern=_HASH)
    review_hash: str = Field(pattern=_HASH)
    terminal_artifact_hash: str = Field(pattern=_HASH)
    terminal_kind: Literal["CANDIDATE_SET", "SCIENTIFIC_STOP"]
    status: Literal[
        "PORTFOLIO_POLICY_CANDIDATES_READY_FOR_VALIDATION",
        "NO_PORTFOLIO_POLICY_READY_FOR_VALIDATION",
    ]
    attempted_count: int = Field(ge=144)
    completed_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0, le=8)
    policy_oos_state: Literal["LOCKED"] = "LOCKED"
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    limitations: tuple[str, ...] = Field(min_length=1)
    bundle_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_bundle(self) -> Self:
        """Require candidate-readiness status to match candidate count and terminal kind.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Readiness differs from positive candidate count/candidate-set kind or
                bundle_hash differs.
        """
        ready = self.status == "PORTFOLIO_POLICY_CANDIDATES_READY_FOR_VALIDATION"
        if ready != (self.candidate_count > 0) or ready != (self.terminal_kind == "CANDIDATE_SET"):
            raise ValueError("portfolio_strategy_lab.projection_status_invalid")
        _validate_hash(self, "bundle_hash")
        return self

    def model_view(self) -> dict[str, object]:
        """Return model-safe semantics without hashes or storage references."""
        return {
            "kind": self.kind,
            "status": self.status,
            "attempted_count": self.attempted_count,
            "completed_count": self.completed_count,
            "candidate_count": self.candidate_count,
            "model_call_count": self.model_call_count,
            "policy_oos_state": self.policy_oos_state,
            "policy_holdout_state": self.policy_holdout_state,
            "system_holdout_state": self.system_holdout_state,
            "limitations": self.limitations,
        }


class CurrentPortfolioResearchMarker(PortfolioLabContract):
    """Bind the published research bundle and status to an aware publication clock."""

    kind: Literal["CurrentPortfolioResearchMarker"] = "CurrentPortfolioResearchMarker"
    program_hash: str = Field(pattern=_HASH)
    ledger_hash: str = Field(pattern=_HASH)
    bundle_hash: str = Field(pattern=_HASH)
    status: Literal[
        "PORTFOLIO_POLICY_CANDIDATES_READY_FOR_VALIDATION",
        "NO_PORTFOLIO_POLICY_READY_FOR_VALIDATION",
    ]
    published_at: datetime
    marker_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_marker(self) -> Self:
        """Require an aware publication clock and exact marker identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Publication clock is naive or marker_hash differs.
        """
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            raise ValueError("portfolio_strategy_lab.publication_clock_invalid")
        _validate_hash(self, "marker_hash")
        return self


class CurrentPortfolioResearchPointer(PortfolioLabContract):
    """Bind one current research marker through its own canonical pointer identity."""

    kind: Literal["CurrentPortfolioResearchPointer"] = "CurrentPortfolioResearchPointer"
    marker_hash: str = Field(pattern=_HASH)
    pointer_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_pointer(self) -> Self:
        """Require exact current-marker pointer identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: pointer_hash differs from its canonical payload.
        """
        _validate_hash(self, "pointer_hash")
        return self


def seal_contract[ContractT: PortfolioLabContract](
    model: type[ContractT], identity_field: str, /, **values: object
) -> ContractT:
    """Construct and validate a concrete portfolio contract with its generated identity.

    Args:
        model: Concrete portfolio contract class.
        identity_field: Name of its generated self identity.
        values: Explicit contract fields supplied for validation.

    Returns:
        Validated concrete model sealed through the shared identity owner.

    Raises:
        pydantic.ValidationError: Supplied fields or concrete consistency checks fail.
    """
    return seal_model_validated(model, identity_field, **values)


def _validate_hash(value: PortfolioLabContract, field: str) -> None:
    if getattr(value, field) != canonical_hash(value.model_dump(mode="json", exclude={field})):
        raise ValueError("portfolio_strategy_lab.identity_invalid")


__all__ = [
    "CurrentPortfolioResearchMarker",
    "CurrentPortfolioResearchPointer",
    "PortfolioBaselineSlate",
    "PortfolioEvidenceDossier",
    "PortfolioExperimentProgram",
    "PortfolioExperimentStratum",
    "PortfolioLabContract",
    "PortfolioPolicyCandidate",
    "PortfolioPolicyCandidateSet",
    "PortfolioPolicyFamily",
    "PortfolioPolicySpec",
    "PortfolioResearchOutcome",
    "PortfolioResearchProjectionBundle",
    "PortfolioResearchRFC",
    "PortfolioResearchReview",
    "PortfolioScientificStop",
    "PortfolioScoreMode",
    "PortfolioSolverRuntimeEvidence",
    "PortfolioStrategyLabMandate",
    "PortfolioTrialEvidence",
    "PortfolioTrialLedger",
    "PortfolioTrialLedgerEntry",
    "PortfolioTrialMetrics",
    "SectorDeviationPenaltyPolicy",
    "TopKEqualWeightPolicy",
    "TopKMinimumVariancePolicy",
    "TopKScoreRiskCostPolicy",
    "TrialState",
    "seal_contract",
]
