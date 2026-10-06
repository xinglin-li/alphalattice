"""Typed evidence, fold, Agent-board, and validation-slate contracts."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import (
    PortfolioLabContract,
    PortfolioPolicySpec,
)
from .split_policy import (
    PORTFOLIO_ANCHORED_THREE_FOLD,
    RESEARCH_SESSION_COUNT,
    anchored_fold_geometry,
)

_HASH = r"^[0-9a-f]{64}$"
_RESEARCH_COUNT = RESEARCH_SESSION_COUNT


def _identity(value: PortfolioLabContract, field: str) -> None:
    if getattr(value, field) != canonical_hash(value.model_dump(mode="json", exclude={field})):
        raise ValueError("portfolio_strategy_lab.regularization_identity_invalid")


class PortfolioResearchAlphaSource(StrEnum):
    """Identify the current or sector-aware Alpha source in strategy research."""

    CURRENT_ALPHA = "CURRENT_ALPHA"
    SECTOR_AWARE_ALPHA = "SECTOR_AWARE_ALPHA"


class PortfolioHypothesisStatus(StrEnum):
    """Declare active, supported, weakened, falsified or unresolved research hypotheses."""

    ACTIVE = "ACTIVE"
    SUPPORTED = "SUPPORTED"
    WEAKENED = "WEAKENED"
    FALSIFIED = "FALSIFIED"
    UNRESOLVED = "UNRESOLVED"


class PortfolioResearchBoardState(StrEnum):
    """Declare bounded hypothesis, experiment, evidence-update and terminal workflow stages."""

    FORM_COMPETING_HYPOTHESES = "FORM_COMPETING_HYPOTHESES"
    REQUEST_EXPERIMENT_OR_TERMINATE = "REQUEST_EXPERIMENT_OR_TERMINATE"
    HOST_EXECUTES_EXPERIMENT = "HOST_EXECUTES_EXPERIMENT"
    UPDATE_FROM_EVIDENCE_DELTA = "UPDATE_FROM_EVIDENCE_DELTA"
    REQUEST_SECOND_EXPERIMENT_OR_TERMINATE = "REQUEST_SECOND_EXPERIMENT_OR_TERMINATE"
    TERMINAL = "TERMINAL"


class PortfolioResearchExperimentKind(StrEnum):
    """Name evidence-only attribution and declared sector/concentration/turnover/Risk ablations."""

    EVIDENCE_ONLY_ATTRIBUTION = "EVIDENCE_ONLY_ATTRIBUTION"
    SECTOR_CONTROL_ABLATION = "SECTOR_CONTROL_ABLATION"
    CONCENTRATION_ABLATION = "CONCENTRATION_ABLATION"
    TURNOVER_CONTROL_ABLATION = "TURNOVER_CONTROL_ABLATION"
    RISK_PENALTY_ABLATION = "RISK_PENALTY_ABLATION"


class PortfolioResearchTerminalRoute(StrEnum):
    """Declare candidate acceptance, Alpha/Risk routing, a new mandate or evidence-backed stop."""

    ACCEPT_POLICY_CANDIDATES = "ACCEPT_POLICY_CANDIDATES"
    ROUTE_TO_ALPHA = "ROUTE_TO_ALPHA"
    ROUTE_TO_RISK = "ROUTE_TO_RISK"
    REQUEST_NEW_MANDATE = "REQUEST_NEW_MANDATE"
    STOP_WITH_EVIDENCE = "STOP_WITH_EVIDENCE"


class PortfolioAgentValueClassification(StrEnum):
    """Classify routing value, compute avoidance, equivalence or review failure.

    Classify observed routing/compute value, equivalence, unsupported harm or incomplete review.
    """

    MATERIAL_RESEARCH_ROUTE_VALUE = "MATERIAL_RESEARCH_ROUTE_VALUE"
    COMPUTE_AVOIDANCE_VALUE = "COMPUTE_AVOIDANCE_VALUE"
    EQUIVALENT_TO_DETERMINISTIC_ROUTER = "EQUIVALENT_TO_DETERMINISTIC_ROUTER"
    HARMFUL_OR_UNSUPPORTED = "HARMFUL_OR_UNSUPPORTED"
    AGENT_EVALUATION_INCOMPLETE = "AGENT_EVALUATION_INCOMPLETE"


class PortfolioEvidenceReclassificationReceipt(PortfolioLabContract):
    """Bind authorized reclassification and its retired validation claim.

    Bind authorized evidence reclassification and retirement of its independent-validation claim.
    """

    kind: Literal["PortfolioEvidenceReclassificationReceipt"] = (
        "PortfolioEvidenceReclassificationReceipt"
    )
    original_development_mandate_hash: str = Field(pattern=_HASH)
    original_development_attempt_count: Literal[216] = 216
    sector_aware_attempt_count: Literal[108] = 108
    consumed_oos_slate_hash: str = Field(pattern=_HASH)
    consumed_oos_result_hash: str = Field(pattern=_HASH)
    consumed_oos_evidence_hashes: tuple[str, ...] = Field(min_length=1, max_length=6)
    original_oos_status: Literal["NO_PORTFOLIO_POLICY_ADMITTED"] = "NO_PORTFOLIO_POLICY_ADMITTED"
    new_evidence_classification: Literal["STRATEGY_LAB_RESEARCH_AND_VALIDATION_EVIDENCE"] = (
        "STRATEGY_LAB_RESEARCH_AND_VALIDATION_EVIDENCE"
    )
    independent_oos_claim_retired: Literal[True] = True
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    typed_authority: Literal["USER_AUTHORIZED_RECLASSIFICATION_AND_CONDITIONAL_POLICY_HOLDOUT"]
    receipt_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_receipt(self) -> Self:
        """Require exact authorized reclassification receipt identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: receipt_hash differs.
        """
        _identity(self, "receipt_hash")
        return self


class PortfolioResearchAlphaRecipe(PortfolioLabContract):
    """Bind one research Alpha candidate to exact program, registry and development scores."""

    source: PortfolioResearchAlphaSource
    candidate_id: str = Field(min_length=1)
    alpha_program_hash: str = Field(pattern=_HASH)
    alpha_registry_hash: str = Field(pattern=_HASH)
    development_score_surface_hash: str = Field(pattern=_HASH)


class PortfolioResearchFold(PortfolioLabContract):
    """Declare anchored training, embargo and validation in both index spaces.

    Declare anchored training, embargo and validation geometry in global and observation indices.
    """

    fold_index: Literal[1, 2, 3]
    training_global_start: Literal[0] = 0
    training_global_end: int = Field(ge=1, le=620)
    training_local_stop: int = Field(ge=1, le=_RESEARCH_COUNT)
    embargo_global_index: Literal[368, 494, 621]
    embargo_session: date
    validation_global_start: Literal[369, 495, 622]
    validation_global_end: Literal[493, 620, 746]
    validation_local_start: int = Field(ge=1, le=_RESEARCH_COUNT)
    validation_local_stop: int = Field(ge=1, le=_RESEARCH_COUNT)
    validation_formation_sessions: tuple[date, ...] = Field(min_length=125, max_length=126)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_fold(self) -> Self:
        # Read from the split-policy owner rather than restated here. The literal
        # dict this replaced was the second of three copies of one geometry, and
        # a contract that validates against its own copy cannot catch the runtime
        # drifting away from it.
        """Require exact geometry from the installed split-policy owner.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Global/local boundaries, expected validation length or sorted unique
                validation sessions differ.
        """
        geometry = {
            value.fold_index: value
            for value in anchored_fold_geometry(PORTFOLIO_ANCHORED_THREE_FOLD)
        }[self.fold_index]
        expected = (
            geometry.training_global_end,
            geometry.embargo_global_index,
            geometry.validation_global_start,
            geometry.validation_global_end,
            geometry.training_local_stop,
            geometry.validation_local_start,
            geometry.validation_local_stop,
            geometry.validation_length,
        )
        observed = (
            self.training_global_end,
            self.embargo_global_index,
            self.validation_global_start,
            self.validation_global_end,
            self.training_local_stop,
            self.validation_local_start,
            self.validation_local_stop,
            len(self.validation_formation_sessions),
        )
        if observed != expected or self.validation_formation_sessions != tuple(
            sorted(set(self.validation_formation_sessions))
        ):
            raise ValueError("portfolio_strategy_lab.regularization_fold_invalid")
        return self


class PortfolioStrategyResearchMandate(PortfolioLabContract):
    """Bind the declared source pair, anchored folds, numerical environment and research budgets."""

    kind: Literal["PortfolioStrategyResearchMandate"] = "PortfolioStrategyResearchMandate"
    reclassification_receipt_hash: str = Field(pattern=_HASH)
    consumed_oos_result_hash: str = Field(pattern=_HASH)
    recipes: tuple[PortfolioResearchAlphaRecipe, PortfolioResearchAlphaRecipe]
    research_global_indices: tuple[int, ...] = Field(
        min_length=_RESEARCH_COUNT, max_length=_RESEARCH_COUNT
    )
    research_formation_sessions: tuple[date, ...] = Field(
        min_length=_RESEARCH_COUNT, max_length=_RESEARCH_COUNT
    )
    folds: tuple[PortfolioResearchFold, PortfolioResearchFold, PortfolioResearchFold]
    policy_holdout_embargo_session: date
    policy_holdout_formation_sessions: tuple[date, ...] = Field(min_length=252, max_length=252)
    risk_surface_hash: str = Field(pattern=_HASH)
    tradability_bundle_hash: str = Field(pattern=_HASH)
    benchmark_identity_hash: str = Field(pattern=_HASH)
    universe_epoch_hash: str = Field(pattern=_HASH)
    ordered_listing_ids_hash: str = Field(pattern=_HASH)
    selection_cost_bps: Literal[10] = 10
    reporting_cost_bps: tuple[Literal[5, 10, 20], ...] = (5, 10, 20)
    search_trials_per_fold_stratum: Literal[24] = 24
    maximum_experiment_generations: Literal[2] = 2
    experiment_trials_per_stratum: Literal[12] = 12
    bootstrap_resamples: Literal[2000] = 2000
    bootstrap_block_size: Literal[21] = 21
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    numerical_execution_binding_hash: str = Field(pattern=_HASH)
    numerical_environment_hash: str = Field(pattern=_HASH)
    limitations: tuple[str, ...] = Field(min_length=1)
    mandate_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_mandate(self) -> Self:
        """Require declared research indices, ordered support, fold order and Alpha source pair.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Declared index geometry, sorted unique axes, folds 1/2/3, source order or
                mandate_hash differs.
        """
        expected_indices = (*range(0, 494), *range(495, 747))
        if (
            self.research_global_indices != expected_indices
            or self.research_formation_sessions
            != tuple(sorted(set(self.research_formation_sessions)))
            or tuple(value.fold_index for value in self.folds) != (1, 2, 3)
            or tuple(value.source for value in self.recipes)
            != (
                PortfolioResearchAlphaSource.CURRENT_ALPHA,
                PortfolioResearchAlphaSource.SECTOR_AWARE_ALPHA,
            )
            or self.policy_holdout_formation_sessions
            != tuple(sorted(set(self.policy_holdout_formation_sessions)))
        ):
            raise ValueError("portfolio_strategy_lab.regularization_mandate_invalid")
        _identity(self, "mandate_hash")
        return self


class PortfolioCrossFoldPolicyCandidate(PortfolioLabContract):
    """Retain supported cross-fold policy evidence and conservative region diagnostics."""

    source: PortfolioResearchAlphaSource
    candidate_id: str = Field(min_length=1)
    policy: PortfolioPolicySpec
    supporting_fold_evidence_hashes: tuple[str, ...] = Field(min_length=2, max_length=3)
    fold_support_count: int = Field(ge=2, le=3)
    median_active_log_wealth: float = Field(gt=0.0)
    median_sortino: float
    worst_fold_drawdown: float = Field(ge=0.0, le=1.0)
    median_turnover: float = Field(ge=0.0)
    median_hhi: float = Field(ge=0.0, le=1.0)
    stable_policy_key: str = Field(min_length=1)


class PortfolioCrossFoldEvidenceDossier(PortfolioLabContract):
    """Bind six fold ledgers, candidate support, contradictions and limitations."""

    kind: Literal["PortfolioCrossFoldEvidenceDossier"] = "PortfolioCrossFoldEvidenceDossier"
    mandate_hash: str = Field(pattern=_HASH)
    fold_ledger_hashes: tuple[str, ...] = Field(min_length=6, max_length=6)
    control_evidence: tuple[str, ...] = Field(min_length=24)
    fold_validation_evidence_hashes: tuple[str, ...] = Field(max_length=6)
    stable_candidates: tuple[PortfolioCrossFoldPolicyCandidate, ...] = Field(max_length=3)
    contradictions: tuple[str, ...]
    owner_attribution: tuple[str, ...]
    falsification_ledger: tuple[str, ...]
    limitations: tuple[str, ...] = Field(min_length=1)
    dossier_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_dossier(self) -> Self:
        """Require six distinct fold ledgers and exact dossier identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Fold ledger uniqueness/count or dossier_hash differs.
        """
        if len(set(self.fold_ledger_hashes)) != 6:
            raise ValueError("portfolio_strategy_lab.regularization_dossier_invalid")
        _identity(self, "dossier_hash")
        return self


class PortfolioHypothesis(PortfolioLabContract):
    """Declare a testable claim, prediction and falsification condition.

    Declare a testable research claim, expected observation and explicit falsification condition.
    """

    handle: str = Field(pattern=r"^H[1-6]$")
    statement: str = Field(min_length=1, max_length=1000)
    predicted_observation: str = Field(min_length=1, max_length=1000)
    falsification_condition: str = Field(min_length=1, max_length=1000)
    status: PortfolioHypothesisStatus


class PortfolioResearchEvidenceDelta(PortfolioLabContract):
    """Bind one bounded experiment generation, changed hypotheses and evidence findings."""

    generation: Literal[1, 2]
    experiment_kind: PortfolioResearchExperimentKind
    semantic_region_handle: str = Field(min_length=1, max_length=80)
    attempt_count: int = Field(ge=0, le=24)
    changed_hypothesis_handles: tuple[str, ...] = Field(max_length=6)
    findings: tuple[str, ...] = Field(min_length=1)
    delta_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_delta(self) -> Self:
        """Require hypothesis progress in a second experiment generation.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Generation two changes no hypothesis or delta_hash differs.
        """
        if self.generation == 2 and not self.changed_hypothesis_handles:
            raise ValueError("portfolio_strategy_lab.research_delta_nonprogress")
        _identity(self, "delta_hash")
        return self


class PortfolioResearchBoard(PortfolioLabContract):
    """Bind hypotheses, experiments, evidence deltas and remaining budgets.

    Bind hypothesis state, accepted experiments, evidence deltas and remaining host budgets.
    """

    kind: Literal["PortfolioResearchBoard"] = "PortfolioResearchBoard"
    dossier_hash: str = Field(pattern=_HASH)
    state: PortfolioResearchBoardState
    hypotheses: tuple[PortfolioHypothesis, ...] = Field(min_length=1, max_length=6)
    accepted_rfc_generations: tuple[int, ...] = Field(max_length=2)
    evidence_deltas: tuple[PortfolioResearchEvidenceDelta, ...] = Field(max_length=2)
    remaining_model_calls: int = Field(ge=0, le=12)
    remaining_experiment_generations: int = Field(ge=0, le=2)
    remaining_compute_attempts: int = Field(ge=0, le=48)
    prior_typed_corrections: tuple[str, ...]
    board_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_board(self) -> Self:
        """Require unique ordered hypothesis handles and exact board identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Hypothesis handles repeat or board_hash differs.
        """
        handles = tuple(value.handle for value in self.hypotheses)
        if handles != tuple(dict.fromkeys(handles)):
            raise ValueError("portfolio_strategy_lab.research_board_duplicate")
        _identity(self, "board_hash")
        return self


class PortfolioResearchTerminalOutcome(PortfolioLabContract):
    """Record bounded terminal routing, candidate keys and acknowledged evidence limitations."""

    route: PortfolioResearchTerminalRoute
    accepted_candidate_keys: tuple[str, ...] = Field(max_length=3)
    hypothesis_updates: dict[str, PortfolioHypothesisStatus]
    summary: str = Field(min_length=1, max_length=1600)
    limitations_acknowledged: Literal[True] = True


class PortfolioAgentValueReport(PortfolioLabContract):
    """Bind observed versus deterministic routing and bounded research/compute value evidence."""

    kind: Literal["PortfolioAgentValueReport"] = "PortfolioAgentValueReport"
    dossier_hash: str = Field(pattern=_HASH)
    board_hash: str = Field(pattern=_HASH)
    classification: PortfolioAgentValueClassification
    agent_route: PortfolioResearchTerminalRoute
    deterministic_route: PortfolioResearchTerminalRoute
    changed_route: bool
    differentiated_hypothesis_count: int = Field(ge=0, le=6)
    falsified_hypothesis_count: int = Field(ge=0, le=6)
    repeated_falsification_count: int = Field(ge=0)
    compute_attempts_used: int = Field(ge=0, le=48)
    compute_attempts_avoided: int = Field(ge=0, le=48)
    unsupported_request_count: int = Field(ge=0)
    typed_repair_count: int = Field(ge=0)
    resisted_isolated_best_trial: bool
    completion_status: Literal["COMPLETE", "AGENT_REVIEW_INCOMPLETE"]
    model_calls: int = Field(ge=0, le=12)
    report_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_report(self) -> Self:
        """Require route-change evidence to match the compared routes.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: changed_route contradicts route comparison or report_hash differs.
        """
        if self.changed_route != (self.agent_route != self.deterministic_route):
            raise ValueError("portfolio_strategy_lab.agent_value_route_invalid")
        _identity(self, "report_hash")
        return self


__all__ = [
    "PortfolioAgentValueClassification",
    "PortfolioAgentValueReport",
    "PortfolioCrossFoldEvidenceDossier",
    "PortfolioCrossFoldPolicyCandidate",
    "PortfolioEvidenceReclassificationReceipt",
    "PortfolioHypothesis",
    "PortfolioHypothesisStatus",
    "PortfolioResearchAlphaRecipe",
    "PortfolioResearchAlphaSource",
    "PortfolioResearchBoard",
    "PortfolioResearchBoardState",
    "PortfolioResearchEvidenceDelta",
    "PortfolioResearchExperimentKind",
    "PortfolioResearchFold",
    "PortfolioResearchTerminalOutcome",
    "PortfolioResearchTerminalRoute",
    "PortfolioStrategyResearchMandate",
]
