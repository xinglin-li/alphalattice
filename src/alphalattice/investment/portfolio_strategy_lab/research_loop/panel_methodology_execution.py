"""Execute an installed paired Panel methodology through its real Desk owners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from secrets import token_hex
from time import perf_counter
from types import MappingProxyType
from typing import Any, Literal, Protocol, Self, cast

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed, parallel_config  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.causal_inputs.admission import admit_decision_input_set
from alphalattice.capabilities.causal_inputs.contracts import (
    AdmittedInformationSet,
    CausalInputAuthority,
)
from alphalattice.capabilities.causal_inputs.schedules import (
    resolve_installed_schedule_handle,
    resolve_strategy_decision_schedule,
)
from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioStateTransitionBinding,
)
from alphalattice.capabilities.portfolio_inputs.session_marks import (
    session_mark_input_authority,
)
from alphalattice.capabilities.portfolio_inputs.signed_score.contracts import (
    PortfolioExecutionEvents,
    ScoreObservationAuthority,
    score_input_authority,
)
from alphalattice.capabilities.portfolio_inputs.tradability.contracts import (
    tradability_input_authority,
)
from alphalattice.control.observation_runtime.telemetry.process_metrics import (
    ProcessResourceMonitor,
    ProcessResourceUsage,
)
from alphalattice.foundation.causal_outcomes.execution.methods import (
    build_installed_execution_outcome_method_catalog,
)
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactStore,
    PanelScoreSurfaceArtifact,
)
from alphalattice.investment.alpha_research.experiments.panel_alpha_fold_execution import (
    PanelAlphaFold,
    PanelAlphaFoldWorkerRequest,
    PanelAlphaFoldWorkerResult,
    PanelFeatureSourcePayload,
    execute_panel_alpha_fold_worker,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_authoring import (
    PANEL_ALPHA_FOLD_RSS_FUSE_BYTES,
    PANEL_ALPHA_PARENT_RSS_RESERVATION_BYTES,
    PanelResearchMethodologyRequest,
    PanelResearchPreflight,
    PanelResearchRuntimeCaps,
    alpha_model_call_upper_bounds,
    methodology_section,
    panel_alpha_configuration_hash,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_models import (
    PanelModelFoldEvidence,
    PanelModelTrialEvidence,
    PanelScoreSurface,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_statistics import (
    AlphaEconomicCandidateEvidence,
    AlphaEconomicCandidateInput,
    PanelAlphaEconomicEvaluation,
    PanelModelScientificSelection,
    PanelScoreFilterEvaluationResult,
    PanelScoreFilterEvidence,
    PanelScoreFilterSelection,
    evaluate_fixed_recipe_alpha_economics,
    evaluate_panel_score_filters,
    fixed_recipe_economic_metric_call_count,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (
    PanelFeaturePlan,
)
from alphalattice.investment.alpha_research.scores.temporal_aggregation import (
    SEALED_PANEL_ALPHA_MODEL_RECIPE_ID,
    SEALED_PANEL_ALPHA_PROGRAM_HASH,
    SEALED_PANEL_ALPHA_ROOT_HASH,
    SEALED_PANEL_ALPHA_VIEW_ID,
    AlphaPanelSourceIdentity,
    FixedAlphaRowAxisReceipt,
    PanelScoreFoldIdentity,
    PanelScoreFormationReadinessReceipt,
    PanelScoreProducerIdentity,
    resolve_panel_score_producer_identity,
    score_value_hash,
)
from alphalattice.investment.alpha_research.simple_signal.authority import (
    WorkspaceSimpleSignalMaterializationAuthority,
    WorkspaceSimpleSignalSource,
)
from alphalattice.investment.alpha_research.simple_signal.contracts import (
    FIXED_STUDY_SIMPLE_SCORE_METHOD_ID,
    SimpleSignedScoreBinding,
)
from alphalattice.investment.alpha_research.simple_signal.standardize import (
    score_values_identity as simple_score_values_identity,
)
from alphalattice.investment.portfolio_strategy_lab.campaign.authority import (
    ResolvedReferenceMark,
    ResolvedStageSixCovariance,
    project_reference_mark_lane,
)
from alphalattice.investment.portfolio_strategy_lab.inputs.development import ExecutionClockContext
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioPublicationError,
    PortfolioResearchArtifactStore,
)
from alphalattice.investment.portfolio_strategy_lab.research_loop.paired_alpha_portfolio import (
    DeclaredFilteredScore,
    ImmutablePortfolioBenchmark,
    ImmutablePortfolioMarketInputs,
    PairedAlphaPortfolioInputs,
    replay_paired_alpha_portfolio_research,
    run_paired_alpha_portfolio_research,
    verify_paired_alpha_portfolio_graph,
)
from alphalattice.investment.risk_research.experiments.temporal import (
    RISK_READINESS_BUDGET,
    risk_covariance_authority,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskExecutionResult,
    DeskExperimentCompiler,
    NumericalCallRecorder,
    ResearchExecutionEvidence,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int64]


class PortfolioAxisAuthority(Protocol):
    """The exact axes needed to intersect Alpha, Risk, and Market authority."""

    @property
    def formation_sessions(self) -> tuple[date, ...]:
        """Expose the exact available portfolio formation axis.

        Returns:
            Ordered admitted formation sessions.
        """
        ...

    @property
    def ordered_listing_ids(self) -> tuple[str, ...]:
        """Expose the exact available portfolio listing axis.

        Returns:
            Ordered admitted listing identities.
        """
        ...


class SessionMarkTemporalAuthority(Protocol):
    """The Market fields consumed by common causal admission."""

    availability_policy_id: str
    availability_policy_hash: str
    surface_hash: str
    epoch: Any
    observed_through_offset_sessions: int
    observed_through_event: str


_HASH = r"^[0-9a-f]{64}$"
PANEL_METHODOLOGY_ROOT_CATEGORY = "development/paired-panel-research/roots"
_ALPHA_ECONOMIC_CONTENT_CATEGORY = "development/paired-panel-research/alpha-economic-evaluations"
_ALPHA_ECONOMIC_ATTESTATION_CATEGORY = (
    "development/paired-panel-research/alpha-economic-publication-attestations"
)
_PORTFOLIO_PREFLIGHT_CATEGORY = "development/paired-panel-research/portfolio-preflights"
_PORTFOLIO_PREFLIGHT_BINDING_CATEGORY = (
    "development/paired-panel-research/portfolio-preflight-bindings"
)


class PanelMethodologyExecutionError(ValueError):
    """Stable refusal at the installed methodology execution boundary."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class PanelScoreTemporalHandoff(_Contract):
    """Portfolio's strategy-bound consumption of a factual Alpha producer."""

    kind: Literal["PanelScoreTemporalHandoff"] = "PanelScoreTemporalHandoff"
    producer: PanelScoreProducerIdentity
    observation: ScoreObservationAuthority
    causal_input: CausalInputAuthority
    point_in_time_disposition: Literal["CURRENT_MEMBERSHIP_BACKFILLED"] = (
        "CURRENT_MEMBERSHIP_BACKFILLED"
    )
    publication_scope: Literal["DEVELOPMENT_EVIDENCE_ONLY"] = "DEVELOPMENT_EVIDENCE_ONLY"
    handoff_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, handoff_hash="0" * 64)
        return cls(
            **values,
            handoff_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"handoff_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_authority_chain(self) -> Self:
        causal = self.causal_input
        if (
            self.observation.methodology_identity != self.producer.producer_hash
            or causal.surface_hash != self.producer.score_surface_hash
            or causal.owner_identity_hash != self.observation.authority_hash
            or causal.observed_through.anchor.offset_sessions != 0
            or causal.observed_through.anchor.event != "OFFICIAL_CLOSE"
            or causal.source_available.anchor.offset_sessions != 0
            or causal.source_available.anchor.event != "OFFICIAL_CLOSE"
            or causal.derived_ready is None
            or causal.derived_ready.anchor.offset_sessions != 0
            or causal.derived_ready.anchor.event != "OFFICIAL_CLOSE"
            or causal.point_in_time_disposition != "CURRENT_MEMBERSHIP_BACKFILLED"
            or self.handoff_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"handoff_hash"}))
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.panel_score_temporal_handoff_invalid"
            )
        return self


@dataclass(frozen=True, slots=True)
class PanelMethodologyOuterFold:
    """Retain one outer fold's training/validation support and source manifest authority."""

    fold_index: int
    training_sessions: tuple[date, ...]
    validation_sessions: tuple[date, ...]
    source_manifest_hash: str


class ResolvedCandidateScoreSurface(_Contract):
    """One exact numerical OOF child; temporal authority travels beside the set."""

    kind: Literal["ResolvedCandidateScoreSurface"] = "ResolvedCandidateScoreSurface"
    root_hash: str = Field(pattern=_HASH)
    model_recipe_id: str
    fold_index: int = Field(ge=0)
    artifact_hash: str = Field(pattern=_HASH)
    source_surface_hash: str = Field(pattern=_HASH)
    formation_start: date
    formation_end: date
    validation_session_count: int = Field(ge=1)
    validation_session_axis_hash: str = Field(pattern=_HASH)
    row_count: int = Field(ge=1)
    row_axis_hash: str = Field(pattern=_HASH)
    score_value_hash: str = Field(pattern=_HASH)
    target_z_lane_hash: str = Field(pattern=_HASH)
    raw_simple_return_lane_hash: str = Field(pattern=_HASH)
    handoff_disposition: Literal["NUMERICAL_HANDOFF_READY"] = "NUMERICAL_HANDOFF_READY"
    resolution_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one resolved candidate score surface.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical resolution_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        provisional = cls.model_construct(**values, resolution_hash="0" * 64)
        return cls(
            **values,
            resolution_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"resolution_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require ordered score support endpoints and exact source resolution identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PanelMethodologyExecutionError: Formation end precedes start or resolution_hash differs.
        """
        if self.formation_end < self.formation_start or self.resolution_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"resolution_hash"})
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_candidate_score_resolution_invalid"
            )
        return self


class _ExecutionSessionClockReceipt(_Contract):
    session: date
    session_open_timestamp: datetime | None = None
    session_close_timestamp: datetime | None = None

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_clock(self) -> Self:
        if self.session_open_timestamp is None and self.session_close_timestamp is None:
            raise PanelMethodologyExecutionError(
                "alpha_research.portfolio_preflight_execution_clock_invalid"
            )
        return self


def resolve_portfolio_state_transition_binding(
    payload: Mapping[str, object] | PortfolioStateTransitionBinding,
) -> PortfolioStateTransitionBinding:
    """Reopen a host-supplied state-transition payload at its contract owner."""
    return cast(
        PortfolioStateTransitionBinding, PortfolioStateTransitionBinding.model_validate(payload)
    )


def panel_portfolio_preflight_source_hash(
    *,
    r0_surface_hash: str,
    r1_surface_hash: str | None,
    alpha_panel_source_identity: AlphaPanelSourceIdentity,
    total_return_target_evidence_hash: str | None,
    outcome_method_binding_hash: str | None,
    panel_derivation_recipe_hash: str | None,
    risk_input_binding_hash: str | None,
    risk_return_surface_hash: str | None,
    market_tradability_bundle_hash: str,
    market_universe_epoch_hash: str,
    market_sector_revision: str,
    market_source_surface_hash: str,
    benchmark_surface_hash: str,
    state_transition: PortfolioStateTransitionBinding | None,
) -> str:
    """Return the durable identity of compact Portfolio preflight inputs."""
    return str(
        canonical_hash(
            {
                "r0_surface_hash": r0_surface_hash,
                "r1_surface_hash": r1_surface_hash,
                "alpha_panel_source_identity": alpha_panel_source_identity.model_dump(mode="json"),
                "total_return_target_evidence_hash": total_return_target_evidence_hash,
                "outcome_method_binding_hash": outcome_method_binding_hash,
                "panel_derivation_recipe_hash": panel_derivation_recipe_hash,
                "risk_input_binding_hash": risk_input_binding_hash,
                "risk_return_surface_hash": risk_return_surface_hash,
                "market_tradability_bundle_hash": market_tradability_bundle_hash,
                "market_universe_epoch_hash": market_universe_epoch_hash,
                "market_sector_revision": market_sector_revision,
                "market_source_surface_hash": market_source_surface_hash,
                "benchmark_surface_hash": benchmark_surface_hash,
                "state_transition": (
                    state_transition.model_dump(mode="json")
                    if state_transition is not None
                    else None
                ),
            }
        )
    )


class PanelPortfolioPreflightReceipt(_Contract):
    """Program-bound prepared inputs for one source-free Portfolio run."""

    kind: Literal["PanelPortfolioPreflightReceipt"] = "PanelPortfolioPreflightReceipt"
    identity_class: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    program_hash: str = Field(pattern=_HASH)
    authority_hash: str = Field(pattern=_HASH)
    preflight: PanelResearchPreflight
    feature_count_by_method: tuple[tuple[str, int], ...] = Field(min_length=1)
    feature_axis_hash_by_method: tuple[tuple[str, str], ...] = Field(min_length=1)
    feature_view_binding_hash_by_method: tuple[tuple[str, str], ...] = Field(min_length=1)
    fixed_alpha_root_hash: str = Field(pattern=_HASH)
    fixed_alpha_row_axis_receipt_hash: str = Field(pattern=_HASH)
    ordered_fold_hashes: tuple[str, ...] = Field(min_length=1)
    r0_surface_hash: str = Field(pattern=_HASH)
    r1_surface_hash: str | None = Field(default=None, pattern=_HASH)
    alpha_panel_source_identity: AlphaPanelSourceIdentity
    total_return_target_evidence_hash: str | None = Field(default=None, pattern=_HASH)
    outcome_method_binding_hash: str | None = Field(default=None, pattern=_HASH)
    panel_derivation_recipe_hash: str | None = Field(default=None, pattern=_HASH)
    risk_input_binding_hash: str | None = Field(default=None, pattern=_HASH)
    risk_return_surface_hash: str | None = Field(default=None, pattern=_HASH)
    market_tradability_bundle_hash: str = Field(pattern=_HASH)
    market_universe_epoch_hash: str = Field(pattern=_HASH)
    market_sector_revision: str = Field(pattern=_HASH)
    market_source_surface_hash: str = Field(pattern=_HASH)
    benchmark_surface_hash: str = Field(pattern=_HASH)
    state_transition: PortfolioStateTransitionBinding | None = None
    preflight_source_hash: str | None = Field(default=None, pattern=_HASH)
    portfolio_formation_sessions: tuple[date, ...] = Field(min_length=1)
    portfolio_economic_formation_sessions: tuple[date, ...] = Field(min_length=1)
    portfolio_decision_ranges: tuple[tuple[int, int], ...] = Field(min_length=1)
    portfolio_passive_sessions: tuple[date, ...]
    portfolio_ordered_listing_ids: tuple[str, ...] = Field(min_length=2)
    portfolio_session_axis_hash: str = Field(pattern=_HASH)
    portfolio_economic_session_axis_hash: str = Field(pattern=_HASH)
    portfolio_listing_axis_hash: str = Field(pattern=_HASH)
    portfolio_axis_hash: str = Field(pattern=_HASH)
    simple_score_binding_hash: str | None = Field(default=None, pattern=_HASH)
    simple_score_value_artifact_hash: str | None = Field(default=None, pattern=_HASH)
    feature_materialization_count: Literal[0] = 0
    feature_materialization_seconds: float = Field(ge=0.0)
    simple_score_materialization_count: int = Field(ge=0, le=1)
    simple_score_materialization_seconds: float = Field(ge=0.0)
    owner_resolution_seconds: float = Field(ge=0.0)
    alpha_verification_seconds: float = Field(ge=0.0)
    receipt_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        source_handle_group = (
            values["total_return_target_evidence_hash"],
            values["outcome_method_binding_hash"],
            values["panel_derivation_recipe_hash"],
            values["risk_input_binding_hash"],
            values["risk_return_surface_hash"],
            values["state_transition"],
        )
        source_hash = (
            panel_portfolio_preflight_source_hash(
                r0_surface_hash=cast(str, values["r0_surface_hash"]),
                r1_surface_hash=cast(str | None, values["r1_surface_hash"]),
                alpha_panel_source_identity=cast(
                    AlphaPanelSourceIdentity, values["alpha_panel_source_identity"]
                ),
                total_return_target_evidence_hash=cast(
                    str | None, values["total_return_target_evidence_hash"]
                ),
                outcome_method_binding_hash=cast(str | None, values["outcome_method_binding_hash"]),
                panel_derivation_recipe_hash=cast(
                    str | None, values["panel_derivation_recipe_hash"]
                ),
                risk_input_binding_hash=cast(str | None, values["risk_input_binding_hash"]),
                risk_return_surface_hash=cast(str | None, values["risk_return_surface_hash"]),
                market_tradability_bundle_hash=cast(str, values["market_tradability_bundle_hash"]),
                market_universe_epoch_hash=cast(str, values["market_universe_epoch_hash"]),
                market_sector_revision=cast(str, values["market_sector_revision"]),
                market_source_surface_hash=cast(str, values["market_source_surface_hash"]),
                benchmark_surface_hash=cast(str, values["benchmark_surface_hash"]),
                state_transition=cast(
                    PortfolioStateTransitionBinding | None, values["state_transition"]
                ),
            )
            if all(value is not None for value in source_handle_group)
            else None
        )
        declared_source_hash = values.get("preflight_source_hash")
        if declared_source_hash is not None and declared_source_hash != source_hash:
            raise PanelMethodologyExecutionError(
                "alpha_research.portfolio_preflight_source_identity_invalid"
            )
        materialized = {**values, "preflight_source_hash": source_hash}
        provisional = cls.model_construct(**materialized, receipt_hash="0" * 64)
        return cls(
            **materialized,
            receipt_hash=str(
                canonical_hash(
                    provisional.model_dump(mode="json", exclude={"receipt_hash"}, exclude_none=True)
                )
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        feature_methods = tuple(value[0] for value in self.feature_count_by_method)
        simple_score_group = (
            self.simple_score_binding_hash,
            self.simple_score_value_artifact_hash,
        )
        source_handle_group = (
            self.total_return_target_evidence_hash,
            self.outcome_method_binding_hash,
            self.panel_derivation_recipe_hash,
            self.risk_input_binding_hash,
            self.risk_return_surface_hash,
        )
        if (
            self.fixed_alpha_root_hash != SEALED_PANEL_ALPHA_ROOT_HASH
            or len(set(feature_methods)) != len(feature_methods)
            or tuple(value[0] for value in self.feature_axis_hash_by_method) != feature_methods
            or tuple(value[0] for value in self.feature_view_binding_hash_by_method)
            != feature_methods
            or any(value is None for value in simple_score_group)
            != all(value is None for value in simple_score_group)
            or any(value is None for value in source_handle_group)
            != all(value is None for value in source_handle_group)
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.portfolio_preflight_receipt_invalid"
            )
        if (
            len(self.portfolio_formation_sessions)
            != self.preflight.actual_portfolio_formation_count
            or len(self.portfolio_economic_formation_sessions)
            != self.preflight.actual_portfolio_economic_formation_count
            or len(self.portfolio_ordered_listing_ids)
            != self.preflight.actual_portfolio_listing_count
            or self.portfolio_session_axis_hash
            != canonical_hash([value.isoformat() for value in self.portfolio_formation_sessions])
            or self.portfolio_economic_session_axis_hash
            != canonical_hash(
                [value.isoformat() for value in self.portfolio_economic_formation_sessions]
            )
            or self.portfolio_listing_axis_hash
            != canonical_hash(list(self.portfolio_ordered_listing_ids))
            or tuple(
                value
                for value in self.portfolio_economic_formation_sessions
                if value not in set(self.portfolio_formation_sessions)
            )
            != self.portfolio_passive_sessions
            or self.portfolio_axis_hash
            != canonical_hash(
                {
                    "sessions": self.portfolio_session_axis_hash,
                    "economic_sessions": self.portfolio_economic_session_axis_hash,
                    "decision_ranges": self.portfolio_decision_ranges,
                    "passive_sessions": [
                        value.isoformat() for value in self.portfolio_passive_sessions
                    ],
                    "listings": self.portfolio_listing_axis_hash,
                    "score_filter_row_axis_policy": "INPUT_ROW_AXIS_INVARIANT",
                }
            )
            or (
                self.preflight_source_hash is not None
                and self.preflight_source_hash
                != panel_portfolio_preflight_source_hash(
                    r0_surface_hash=self.r0_surface_hash,
                    r1_surface_hash=self.r1_surface_hash,
                    alpha_panel_source_identity=self.alpha_panel_source_identity,
                    total_return_target_evidence_hash=self.total_return_target_evidence_hash,
                    outcome_method_binding_hash=self.outcome_method_binding_hash,
                    panel_derivation_recipe_hash=self.panel_derivation_recipe_hash,
                    risk_input_binding_hash=self.risk_input_binding_hash,
                    risk_return_surface_hash=self.risk_return_surface_hash,
                    market_tradability_bundle_hash=self.market_tradability_bundle_hash,
                    market_universe_epoch_hash=self.market_universe_epoch_hash,
                    market_sector_revision=self.market_sector_revision,
                    market_source_surface_hash=self.market_source_surface_hash,
                    benchmark_surface_hash=self.benchmark_surface_hash,
                    state_transition=self.state_transition,
                )
            )
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.portfolio_preflight_receipt_invalid"
            )
        receipt_payload = self.model_dump(mode="json", exclude={"receipt_hash"}, exclude_none=True)
        admitted_hashes = {canonical_hash(receipt_payload)}
        if self.preflight.has_pre_allocation_runtime_plan_payload:
            legacy_runtime_plan_payload = dict(receipt_payload)
            legacy_runtime_plan_payload["preflight"] = PanelResearchPreflight._identity_payload(
                self.preflight,
                include_operational_capacity=True,
                include_legacy_runtime_plan=True,
                include_preflight_hash=True,
            )
            admitted_hashes.add(canonical_hash(legacy_runtime_plan_payload))
        if self.receipt_hash not in admitted_hashes:
            raise PanelMethodologyExecutionError(
                "alpha_research.portfolio_preflight_receipt_invalid"
            )
        return self


class _HistoricalPanelPortfolioPreflightReceipt(PanelPortfolioPreflightReceipt):
    """Hash-verified binding-only predecessor; never admitted for a strong run."""

    execution_events: PortfolioExecutionEvents
    execution_session_clocks: tuple[_ExecutionSessionClockReceipt, ...] = Field(min_length=1)
    entry_sessions_by_formation: tuple[tuple[date, date], ...] = Field(min_length=1)
    market_sector_ids: tuple[str, ...] = Field(min_length=1)
    market_sector_exposure_lane_hash: str = Field(pattern=_HASH)
    market_equal_sector_exposure_lane_hash: str = Field(pattern=_HASH)
    market_decision_eligible_lane_hash: str = Field(pattern=_HASH)
    market_execution_available_lane_hash: str = Field(pattern=_HASH)
    market_realized_return_lane_hash: str = Field(pattern=_HASH)
    market_passive_return_lane_hash: str = Field(pattern=_HASH)
    market_adv20_lane_hash: str = Field(pattern=_HASH)
    benchmark_log_returns: tuple[float, ...] = Field(min_length=1)
    benchmark_economic_log_returns: tuple[float, ...] = Field(min_length=1)
    simple_score_binding: SimpleSignedScoreBinding
    simple_score_lane_hash: str = Field(pattern=_HASH)


class PanelPortfolioPreflightBinding(_Contract):
    """Exact Program-to-receipt index; Program identity is the lookup key."""

    kind: Literal["PanelPortfolioPreflightBinding"] = "PanelPortfolioPreflightBinding"
    program_hash: str = Field(pattern=_HASH)
    receipt_hash: str = Field(pattern=_HASH)
    binding_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, binding_hash="0" * 64)
        return cls(
            **values,
            binding_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"binding_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.portfolio_preflight_binding_invalid"
            )
        return self


class _PanelFoldResourceReceipt(_Contract):
    kind: Literal["PanelFoldResourceReceipt"] = "PanelFoldResourceReceipt"
    fold_index: int = Field(ge=0)
    # Optional only so historical roots remain readable.  Every newly
    # attested publication requires the complete fold/call binding.
    fold_hash: str | None = Field(default=None, pattern=_HASH)
    fit_call_count: int | None = Field(default=None, ge=0)
    predict_call_count: int | None = Field(default=None, ge=0)
    metric_call_count: int | None = Field(default=None, ge=0)
    solver_call_count: int | None = Field(default=None, ge=0)
    measurement_scope: Literal["WORKER_PROCESS_LIFETIME"] = "WORKER_PROCESS_LIFETIME"
    wall_seconds: float = Field(gt=0.0)
    average_machine_cpu_percent: float = Field(ge=0.0, le=50.0)
    peak_machine_cpu_percent: float = Field(ge=0.0, le=50.0)
    peak_rss_bytes: int = Field(gt=0, le=PANEL_ALPHA_FOLD_RSS_FUSE_BYTES)
    receipt_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, receipt_hash="0" * 64)
        return cls(
            **values,
            receipt_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"receipt_hash"}))
            ),
        )


class _AlphaEconomicPublicationAttestation(_Contract):
    """Run-authority seal over reusable Alpha-economic numerical content."""

    kind: Literal["AlphaEconomicPublicationAttestation"] = "AlphaEconomicPublicationAttestation"
    identity_class: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    disposition: Literal["ADMITTED"] = "ADMITTED"
    program_hash: str = Field(pattern=_HASH)
    desk_program_hash: str = Field(pattern=_HASH)
    method_binding_hash: str = Field(pattern=_HASH)
    authority_hash: str = Field(pattern=_HASH)
    request_hash: str = Field(pattern=_HASH)
    preflight_hash: str = Field(pattern=_HASH)
    source_resolution_hash: str = Field(pattern=_HASH)
    feature_preflight_hash: str = Field(pattern=_HASH)
    economic_content_hash: str = Field(pattern=_HASH)
    # Numerical bytes are reusable; publication authority belongs to one
    # concrete executor attempt. Historical seals without this field stay
    # parseable for diagnosis but cannot satisfy active resolution.
    run_publication_attempt_hash: str | None = Field(default=None, pattern=_HASH)
    attempt_executed_fold_count: int | None = Field(default=None, ge=0)
    attempt_reused_fold_count: int | None = Field(default=None, ge=0)
    attempt_fit_call_count: int | None = Field(default=None, ge=0)
    attempt_predict_call_count: int | None = Field(default=None, ge=0)
    attempt_metric_call_count: int | None = Field(default=None, ge=0)
    attempt_solver_call_count: int | None = Field(default=None, ge=0)
    ordered_fold_hashes: tuple[str, ...] = Field(min_length=1)
    actual_fit_call_count: int = Field(ge=0)
    actual_predict_call_count: int = Field(ge=0)
    actual_metric_call_count: int = Field(ge=0)
    actual_solver_call_count: int = Field(ge=0)
    admitted_fit_call_count: int = Field(ge=0)
    admitted_predict_call_count: int = Field(ge=0)
    admitted_metric_call_count: int = Field(ge=0)
    admitted_solver_call_count: int = Field(ge=0)
    attestation_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        attempt_fields = (
            "run_publication_attempt_hash",
            "attempt_executed_fold_count",
            "attempt_reused_fold_count",
            "attempt_fit_call_count",
            "attempt_predict_call_count",
            "attempt_metric_call_count",
            "attempt_solver_call_count",
        )
        if any(values.get(field) is None for field in attempt_fields):
            raise PanelMethodologyExecutionError(
                "alpha_research.alpha_economic_run_publication_attempt_missing"
            )
        provisional = cls.model_construct(**values, attestation_hash="0" * 64)
        return cls(
            **values,
            attestation_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"attestation_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_admission(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"attestation_hash"}, exclude_none=True)
        attempt_counts = (
            self.attempt_executed_fold_count,
            self.attempt_reused_fold_count,
            self.attempt_fit_call_count,
            self.attempt_predict_call_count,
            self.attempt_metric_call_count,
            self.attempt_solver_call_count,
        )
        has_run_attempt = self.run_publication_attempt_hash is not None
        attempt_fields_complete = all(value is not None for value in attempt_counts)
        attempt_fields_absent = all(value is None for value in attempt_counts)
        if has_run_attempt:
            if not attempt_fields_complete:
                raise PanelMethodologyExecutionError(
                    "alpha_research.alpha_economic_publication_attestation_invalid"
                )
            executed, reused, fit, predict, metric, solver = cast(
                tuple[int, int, int, int, int, int], attempt_counts
            )
            if (
                executed + reused != len(self.ordered_fold_hashes)
                or fit > self.actual_fit_call_count
                or predict > self.actual_predict_call_count
                or metric > self.actual_metric_call_count
                or solver > self.actual_solver_call_count
            ):
                raise PanelMethodologyExecutionError(
                    "alpha_research.alpha_economic_publication_attestation_invalid"
                )
        elif not attempt_fields_absent:
            raise PanelMethodologyExecutionError(
                "alpha_research.alpha_economic_publication_attestation_invalid"
            )
        if (
            self.actual_fit_call_count > self.admitted_fit_call_count
            or self.actual_predict_call_count > self.admitted_predict_call_count
            or self.actual_metric_call_count > self.admitted_metric_call_count
            or self.actual_solver_call_count > self.admitted_solver_call_count
            or self.attestation_hash != canonical_hash(payload)
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.alpha_economic_publication_attestation_invalid"
            )
        return self


class PanelMethodologyExecutionRoot(_Contract):
    """Bind methodology lineage and scoped resource receipts.

    Bind development-only methodology evidence, numerical lineage and scoped resource receipts.
    """

    kind: str = "PanelMethodologyExecutionRoot"
    identity_class: str = "DEVELOPMENT_ONLY"
    program_hash: str = Field(pattern=_HASH)
    desk_program_hash: str = Field(pattern=_HASH)
    method_binding_hash: str = Field(pattern=_HASH)
    authority_hash: str = Field(pattern=_HASH)
    desk_input_binding_hash: str = Field(pattern=_HASH)
    source_resolution_hash: str = Field(pattern=_HASH)
    request_hash: str = Field(pattern=_HASH)
    alpha_configuration_hash: str | None = Field(default=None, pattern=_HASH)
    upstream_alpha_root_hash: str | None = Field(default=None, pattern=_HASH)
    upstream_score_filter_root_hash: str | None = Field(default=None, pattern=_HASH)
    preflight_hash: str = Field(pattern=_HASH)
    feature_preflight_hash: str = Field(pattern=_HASH)
    ordered_fold_hashes: tuple[str, ...] = Field(min_length=1)
    ordered_scale_receipt_hashes: tuple[str, ...] = Field(min_length=1)
    ordered_inner_trial_hashes: tuple[str, ...] = Field(min_length=1)
    ordered_outer_trial_hashes: tuple[str, ...] = Field(min_length=1)
    ordered_score_surface_artifact_hashes: tuple[str, ...] = Field(min_length=2)
    alpha_score_surface_hash: str | None = Field(default=None, pattern=_HASH)
    alpha_row_axis_receipt_hash: str | None = Field(default=None, pattern=_HASH)
    alpha_simple_score_binding_hash: str | None = Field(default=None, pattern=_HASH)
    alpha_simple_score_value_artifact_hash: str | None = Field(default=None, pattern=_HASH)
    alpha_formation_readiness_receipt_hash: str | None = Field(default=None, pattern=_HASH)
    # Legacy roots bind naked numerical content and remain loadable only. New
    # roots bind the run-authority attestation below instead.
    alpha_economic_evaluation_hash: str | None = Field(default=None, pattern=_HASH)
    alpha_economic_publication_attestation_hash: str | None = Field(default=None, pattern=_HASH)
    ordered_score_filter_evidence_hashes: tuple[str, ...] = ()
    ordered_score_filter_selection_hashes: tuple[str, ...] = ()
    score_filter_frontier_hash: str | None = Field(default=None, pattern=_HASH)
    fixed_simple_score_artifact_hash: str | None = Field(default=None, pattern=_HASH)
    portfolio_root_hash: str | None = Field(default=None, pattern=_HASH)
    portfolio_replay_receipt_hash: str | None = Field(default=None, pattern=_HASH)
    common_formation_sessions: tuple[date, ...] = ()
    portfolio_common_sessions: tuple[date, ...] = ()
    portfolio_economic_sessions: tuple[date, ...] = ()
    portfolio_ordered_listing_ids_hash: str | None = Field(default=None, pattern=_HASH)
    fit_call_count: int = Field(ge=0)
    predict_call_count: int = Field(ge=0)
    metric_call_count: int = Field(ge=0)
    solver_call_count: int = Field(ge=0)
    executed_fold_count: int = Field(default=0, ge=0)
    reused_fold_count: int = Field(default=0, ge=0)
    wall_seconds: float = Field(gt=0.0)
    average_machine_cpu_percent: float = Field(ge=0.0, le=50.0)
    peak_machine_cpu_percent: float = Field(ge=0.0, le=50.0)
    peak_rss_bytes: int = Field(gt=0)
    resource_measurement_scope: Literal[
        "LEGACY_UNSCOPED",
        "PARENT_PROCESS_ONLY",
        "TIME_ALIGNED_PROCESS_TREE",
    ] = "LEGACY_UNSCOPED"
    worker_capacity_reservation_upper_bound_bytes: int = Field(default=0, ge=0)
    process_tree_peak_cpu_measurement: Literal[
        "NOT_MEASURED", "TIME_ALIGNED_PARENT_AND_LIVE_DESCENDANTS"
    ] = "NOT_MEASURED"
    process_tree_peak_rss_measurement: Literal[
        "NOT_MEASURED", "TIME_ALIGNED_PARENT_AND_LIVE_DESCENDANTS"
    ] = "NOT_MEASURED"
    process_tree_peak_rss_bytes: int | None = Field(default=None, gt=0)
    process_tree_peak_live_descendant_count: int | None = Field(default=None, ge=0)
    process_tree_sample_interval_seconds: float | None = Field(default=None, gt=0.0)
    process_tree_instrumentation_limitation: str | None = None
    runtime_plan_hash: str | None = Field(default=None, pattern=_HASH)
    runtime_workload: (
        Literal[
            "METADATA_PREFLIGHT",
            "ALPHA_FOLDS",
            "SCORE_FILTER_NUMERICAL",
            "PORTFOLIO_NUMERICAL",
        ]
        | None
    ) = None
    runtime_profile: Literal["AUTO", "LOW_MEMORY", "BALANCED", "THROUGHPUT"] | None = None
    runtime_profile_resolution: Literal["LOW_MEMORY", "BALANCED", "THROUGHPUT"] | None = None
    runtime_process_logical_processor_limit: int | None = Field(default=None, ge=1)
    runtime_applied_logical_processor_ids: tuple[int, ...] = ()
    runtime_fold_workers: int | None = Field(default=None, ge=0)
    runtime_lightgbm_threads_per_fit: int | None = Field(default=None, ge=0, le=1)
    runtime_duckdb_threads: int | None = Field(default=None, ge=1)
    runtime_blas_threads: Literal[1] | None = None
    worker_resource_receipts: tuple[_PanelFoldResourceReceipt, ...] = ()
    network_access_count: int = 0
    provider_access_count: int = 0
    holdout_access_count: int = 0
    current_or_production_pointer_read_count: int = 0
    pointer_mutation_count: int = 0
    source_workspace_write_count: int = 0
    root_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the methodology root with exact economic worker-call authority.

        Seal methodology root and require full worker-call authority for new economic attestation.

        Args:
            values: Explicit root fields excluding generated root_hash.

        Returns:
            Validated root with absent optional fields omitted from canonical identity.

        Raises:
            PanelMethodologyExecutionError: An economic publication attestation lacks complete
                worker-call authority.
        """
        provisional = cls.model_construct(**values, root_hash="0" * 64)
        root = cls(
            **values,
            root_hash=str(
                canonical_hash(
                    provisional.model_dump(mode="json", exclude={"root_hash"}, exclude_none=True)
                )
            ),
        )
        if (
            root.alpha_economic_publication_attestation_hash is not None
            and not root.has_complete_worker_call_authority()
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.alpha_economic_publication_attempt_lineage_invalid"
            )
        return root

    def has_complete_worker_call_authority(self) -> bool:
        """Return whether every executed worker is bound to its fold and calls."""
        return (
            self.resource_measurement_scope in {"PARENT_PROCESS_ONLY", "TIME_ALIGNED_PROCESS_TREE"}
            and len(self.worker_resource_receipts) == self.executed_fold_count
            and all(
                receipt.fold_hash is not None
                and receipt.fit_call_count is not None
                and receipt.predict_call_count is not None
                and receipt.metric_call_count is not None
                and receipt.solver_call_count is not None
                for receipt in self.worker_resource_receipts
            )
        )

    def has_uniform_legacy_worker_receipts(self) -> bool:
        """Recognize hash-bound receipts from before call authority was recorded."""
        return (
            self.resource_measurement_scope in {"PARENT_PROCESS_ONLY", "TIME_ALIGNED_PROCESS_TREE"}
            and self.executed_fold_count > 0
            and len(self.worker_resource_receipts) == self.executed_fold_count
            and all(
                receipt.fold_hash is None
                and receipt.fit_call_count is None
                and receipt.predict_call_count is None
                and receipt.metric_call_count is None
                and receipt.solver_call_count is None
                for receipt in self.worker_resource_receipts
            )
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_zero_access(self) -> Self:
        """Require zero protected/external access and exact supported root/resource authority.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PanelMethodologyExecutionError: Any access/pointer/source-write counter is nonzero,
                resource/worker receipts are inconsistent, Alpha authority is partial, economic
                evidence claims conflict or root identity is not admitted.
        """
        payload = self.model_dump(mode="json", exclude={"root_hash"}, exclude_none=True)
        admitted_hashes = {str(canonical_hash(payload))}
        legacy_payload = dict(payload)
        if self.resource_measurement_scope == "LEGACY_UNSCOPED":
            for field in (
                "resource_measurement_scope",
                "worker_capacity_reservation_upper_bound_bytes",
                "process_tree_peak_cpu_measurement",
                "process_tree_peak_rss_measurement",
                "worker_resource_receipts",
            ):
                legacy_payload.pop(field, None)
            admitted_hashes.add(str(canonical_hash(legacy_payload)))
        if (
            not self.ordered_score_filter_evidence_hashes
            and not self.ordered_score_filter_selection_hashes
            and self.score_filter_frontier_hash is None
        ):
            legacy_payload.pop("ordered_score_filter_evidence_hashes", None)
            legacy_payload.pop("ordered_score_filter_selection_hashes", None)
            legacy_payload.pop("score_filter_frontier_hash", None)
            admitted_hashes.add(str(canonical_hash(legacy_payload)))
        if (
            self.alpha_configuration_hash is None
            and self.upstream_alpha_root_hash is None
            and self.upstream_score_filter_root_hash is None
        ):
            legacy_payload.pop("alpha_configuration_hash", None)
            legacy_payload.pop("upstream_alpha_root_hash", None)
            legacy_payload.pop("upstream_score_filter_root_hash", None)
            admitted_hashes.add(str(canonical_hash(legacy_payload)))
        if self.executed_fold_count == 0 and self.reused_fold_count == 0:
            legacy_payload.pop("executed_fold_count", None)
            legacy_payload.pop("reused_fold_count", None)
            admitted_hashes.add(str(canonical_hash(legacy_payload)))
        if not self.common_formation_sessions:
            legacy_payload.pop("common_formation_sessions", None)
            admitted_hashes.add(str(canonical_hash(legacy_payload)))
        if "portfolio_economic_sessions" not in self.model_fields_set:
            economic_axis_legacy_payload = dict(payload)
            economic_axis_legacy_payload.pop("portfolio_economic_sessions", None)
            admitted_hashes.add(str(canonical_hash(economic_axis_legacy_payload)))
        # ``runtime_applied_logical_processor_ids`` was added after the fixed
        # Alpha successor was sealed.  It is intentionally an operational,
        # rather than scientific, assertion, but an empty tuple is serialized
        # even when no runtime plan exists.  Admit the prior exact payload only
        # for a wholly absent runtime plan; a populated plan remains bound to
        # every allocation field and cannot take this compatibility path.
        runtime_plan_absent = (
            self.runtime_plan_hash is None
            and self.runtime_workload is None
            and self.runtime_profile is None
            and self.runtime_profile_resolution is None
            and self.runtime_process_logical_processor_limit is None
            and not self.runtime_applied_logical_processor_ids
            and self.runtime_fold_workers is None
            and self.runtime_lightgbm_threads_per_fit is None
            and self.runtime_duckdb_threads is None
            and self.runtime_blas_threads is None
        )
        if runtime_plan_absent:
            for compatibility_payload in (payload, legacy_payload):
                pre_runtime_plan_payload = dict(compatibility_payload)
                pre_runtime_plan_payload.pop("runtime_applied_logical_processor_ids", None)
                admitted_hashes.add(str(canonical_hash(pre_runtime_plan_payload)))
        tree_measurement_complete = (
            self.resource_measurement_scope == "TIME_ALIGNED_PROCESS_TREE"
            and self.process_tree_peak_cpu_measurement == "TIME_ALIGNED_PARENT_AND_LIVE_DESCENDANTS"
            and self.process_tree_peak_rss_measurement == "TIME_ALIGNED_PARENT_AND_LIVE_DESCENDANTS"
            and self.process_tree_peak_rss_bytes is not None
            and self.process_tree_peak_rss_bytes >= self.peak_rss_bytes
            and self.process_tree_peak_live_descendant_count is not None
            and self.process_tree_sample_interval_seconds is not None
            and self.process_tree_instrumentation_limitation is None
            and self.runtime_plan_hash is not None
            and self.runtime_workload is not None
            and self.runtime_profile is not None
            and self.runtime_profile_resolution is not None
            and self.runtime_process_logical_processor_limit is not None
            and len(self.runtime_applied_logical_processor_ids)
            == self.runtime_process_logical_processor_limit
            and len(set(self.runtime_applied_logical_processor_ids))
            == len(self.runtime_applied_logical_processor_ids)
            and self.runtime_fold_workers is not None
            and self.runtime_lightgbm_threads_per_fit is not None
            and self.runtime_duckdb_threads is not None
            and self.runtime_blas_threads is not None
            and self.runtime_fold_workers * self.runtime_lightgbm_threads_per_fit
            + self.runtime_duckdb_threads
            <= self.runtime_process_logical_processor_limit
            and (
                self.runtime_workload == "ALPHA_FOLDS"
                or (self.runtime_fold_workers == 0 and self.runtime_lightgbm_threads_per_fit == 0)
            )
        )
        resource_receipts_valid = (
            (
                self.resource_measurement_scope == "LEGACY_UNSCOPED"
                and not self.worker_resource_receipts
                and self.worker_capacity_reservation_upper_bound_bytes == 0
            )
            or (
                self.resource_measurement_scope == "PARENT_PROCESS_ONLY"
                and len(self.worker_resource_receipts) == self.executed_fold_count
                and tuple(value.fold_index for value in self.worker_resource_receipts)
                == tuple(sorted({value.fold_index for value in self.worker_resource_receipts}))
                and (
                    self.executed_fold_count == 0
                    or self.worker_capacity_reservation_upper_bound_bytes > 0
                )
            )
            or (
                tree_measurement_complete
                and len(self.worker_resource_receipts) == self.executed_fold_count
                and tuple(value.fold_index for value in self.worker_resource_receipts)
                == tuple(sorted({value.fold_index for value in self.worker_resource_receipts}))
                and (
                    self.executed_fold_count == 0
                    or self.worker_capacity_reservation_upper_bound_bytes > 0
                )
            )
        )
        attested_receipts_readable = (
            self.alpha_economic_publication_attestation_hash is None
            or self.has_complete_worker_call_authority()
            or self.has_uniform_legacy_worker_receipts()
        )
        alpha_authority_group = (
            self.alpha_score_surface_hash,
            self.alpha_row_axis_receipt_hash,
            self.alpha_simple_score_binding_hash,
            self.alpha_simple_score_value_artifact_hash,
            self.alpha_formation_readiness_receipt_hash,
        )
        if (
            any(
                value != 0
                for value in (
                    self.network_access_count,
                    self.provider_access_count,
                    self.holdout_access_count,
                    self.current_or_production_pointer_read_count,
                    self.pointer_mutation_count,
                    self.source_workspace_write_count,
                )
            )
            or not resource_receipts_valid
            or not attested_receipts_readable
            or any(value is not None for value in alpha_authority_group)
            != all(value is not None for value in alpha_authority_group)
            or (
                self.alpha_economic_evaluation_hash is not None
                and self.alpha_economic_publication_attestation_hash is not None
            )
            or self.root_hash not in admitted_hashes
        ):
            raise PanelMethodologyExecutionError("alpha_research.panel_methodology_root_invalid")
        return self


@dataclass(frozen=True, slots=True)
class PanelMethodologyExecutionInputs:
    """Retain resolved numerical inputs, preflight evidence and timings.

    Retain resolved Alpha/Portfolio numerical inputs, immutable preflight evidence and operational
    timings.
    """

    plan: PanelFeaturePlan | None
    source_resolution_hash: str
    feature_preflight_hash: str
    outer_folds: tuple[PanelMethodologyOuterFold, ...]
    fixed_feature_column_count: int | None = None
    fixed_feature_axis_hash: str | None = None
    fixed_feature_count_by_method: tuple[tuple[str, int], ...] = ()
    fixed_feature_axis_hash_by_method: tuple[tuple[str, str], ...] = ()
    fixed_feature_view_binding_hash_by_method: tuple[tuple[str, str], ...] = ()
    fixed_alpha_row_axis_receipt: FixedAlphaRowAxisReceipt | None = None
    fixed_alpha_row_axes: tuple[tuple[tuple[date, ...], tuple[str, ...]], ...] = ()
    portfolio_market: ImmutablePortfolioMarketInputs | None = None
    portfolio_axis_authority: PortfolioAxisAuthority | None = None
    portfolio_benchmark: ImmutablePortfolioBenchmark | None = None
    portfolio_fold_indices: IntArray | None = None
    r0_covariance: ResolvedStageSixCovariance | None = None
    r1_covariance: ResolvedStageSixCovariance | None = None
    portfolio_axis: PanelPortfolioAxis | None = None
    simple_score_authority: WorkspaceSimpleSignalMaterializationAuthority | None = None
    simple_score_source: WorkspaceSimpleSignalSource | None = None
    fixed_simple_score_binding_hash: str | None = None
    fixed_simple_score_value_artifact_hash: str | None = None
    total_return_target_evidence_hash: str | None = None
    outcome_method_binding_hash: str | None = None
    panel_derivation_recipe_hash: str | None = None
    risk_input_binding_hash: str | None = None
    risk_return_surface_hash: str | None = None
    benchmark_surface_hash: str | None = None
    market_tradability_bundle_hash: str | None = None
    market_universe_epoch_hash: str | None = None
    market_sector_revision: str | None = None
    market_source_surface_hash: str | None = None
    portfolio_state_transition: PortfolioStateTransitionBinding | None = None
    portfolio_preflight_source_hash: str | None = None
    fixed_simple_score_binding: SimpleSignedScoreBinding | None = None
    fixed_simple_score_values: FloatArray | None = None
    alpha_panel_source_identity: AlphaPanelSourceIdentity | None = None
    panel_score_readiness: PanelScoreFormationReadinessReceipt | None = None
    execution_clock: ExecutionClockContext | None = None
    reference_marks: ResolvedReferenceMark | None = None
    feature_materialization_count: int = 0
    feature_materialization_seconds: float = 0.0
    owner_resolution_seconds: float = 0.0
    alpha_verification_seconds: float = 0.0
    sealed_receipt_load_seconds: float = 0.0
    sealed_preflight_program_hash: str | None = None
    sealed_preflight_receipt: PanelPortfolioPreflightReceipt | None = None
    simple_score_materialization_count: int = 0
    simple_score_materialization_seconds: float = 0.0
    owner_resolution_metrics: tuple[tuple[str, float, int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class _CompletedPanelAlphaFold:
    evidence: PanelModelFoldEvidence
    raw_surface: PanelScoreSurface
    control_surface: PanelScoreSurface
    raw_artifact: PanelScoreSurfaceArtifact
    control_artifact: PanelScoreSurfaceArtifact
    model_spec_id: str
    fixed_surfaces: tuple[PanelScoreSurface, ...] = ()
    fixed_artifacts: tuple[PanelScoreSurfaceArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class _MonitoredPanelAlphaFoldWorkerResult:
    result: PanelAlphaFoldWorkerResult
    resource_usage: ProcessResourceUsage


def _execute_monitored_panel_alpha_fold_worker(
    request: PanelAlphaFoldWorkerRequest,
    *,
    owns_process: bool = False,
) -> _MonitoredPanelAlphaFoldWorkerResult:
    """Measure one process worker without adding telemetry to the Alpha owner."""

    monitor = ProcessResourceMonitor()
    monitor.start()
    try:
        result = execute_panel_alpha_fold_worker(request, owns_process=owns_process)
    except BaseException:
        monitor.finish()
        raise
    return _MonitoredPanelAlphaFoldWorkerResult(
        result=result,
        resource_usage=monitor.finish(),
    )


def _fold_resource_receipt(
    worker: _MonitoredPanelAlphaFoldWorkerResult,
) -> _PanelFoldResourceReceipt:
    usage = worker.resource_usage
    fold = worker.result.fold.fold_evidence
    return _PanelFoldResourceReceipt.create(
        fold_index=fold.fold_index,
        fold_hash=fold.fold_hash,
        fit_call_count=fold.fit_call_count,
        predict_call_count=fold.predict_call_count,
        metric_call_count=fold.metric_call_count,
        solver_call_count=worker.result.solver_call_count,
        wall_seconds=usage.wall_seconds,
        average_machine_cpu_percent=usage.average_machine_cpu_percent,
        peak_machine_cpu_percent=usage.peak_machine_cpu_percent,
        peak_rss_bytes=usage.peak_rss_bytes,
    )


def _admit_preflight_call_counts(
    *,
    preflight: PanelResearchPreflight,
    fit_call_count: int,
    predict_call_count: int,
    metric_call_count: int,
    solver_call_count: int,
) -> None:
    if (
        fit_call_count > preflight.actual_fit_calls
        or predict_call_count > preflight.actual_predict_calls
        or metric_call_count > preflight.actual_metric_calls
        or solver_call_count > preflight.actual_solver_calls
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.panel_methodology_preflight_call_bound_exceeded"
        )


def _publish_admitted_alpha_economic_evaluation(
    *,
    store: PortfolioResearchArtifactStore,
    economic: PanelAlphaEconomicEvaluation,
    program: SealedResearchProgram,
    preflight: PanelResearchPreflight,
    source_resolution_hash: str,
    feature_preflight_hash: str,
    run_publication_attempt_hash: str,
    attempt_executed_fold_count: int,
    attempt_reused_fold_count: int,
    attempt_fit_call_count: int,
    attempt_predict_call_count: int,
    attempt_metric_call_count: int,
    attempt_solver_call_count: int,
    ordered_fold_hashes: tuple[str, ...],
    fit_call_count: int,
    predict_call_count: int,
    metric_call_count: int,
    solver_call_count: int,
) -> _AlphaEconomicPublicationAttestation:
    """Publish reusable bytes only with this Program's aggregate admission."""

    _admit_preflight_call_counts(
        preflight=preflight,
        fit_call_count=fit_call_count,
        predict_call_count=predict_call_count,
        metric_call_count=metric_call_count,
        solver_call_count=solver_call_count,
    )
    store.publish(
        category=_ALPHA_ECONOMIC_CONTENT_CATEGORY,
        value=economic,
        identity_field="evaluation_hash",
    )
    attestation = _AlphaEconomicPublicationAttestation.create(
        program_hash=program.program_hash,
        desk_program_hash=program.desk_program_hash,
        method_binding_hash=program.method_binding_hash,
        authority_hash=program.authority_hash,
        request_hash=preflight.request_hash,
        preflight_hash=preflight.preflight_hash,
        source_resolution_hash=source_resolution_hash,
        feature_preflight_hash=feature_preflight_hash,
        economic_content_hash=economic.evaluation_hash,
        run_publication_attempt_hash=run_publication_attempt_hash,
        attempt_executed_fold_count=attempt_executed_fold_count,
        attempt_reused_fold_count=attempt_reused_fold_count,
        attempt_fit_call_count=attempt_fit_call_count,
        attempt_predict_call_count=attempt_predict_call_count,
        attempt_metric_call_count=attempt_metric_call_count,
        attempt_solver_call_count=attempt_solver_call_count,
        ordered_fold_hashes=ordered_fold_hashes,
        actual_fit_call_count=fit_call_count,
        actual_predict_call_count=predict_call_count,
        actual_metric_call_count=metric_call_count,
        actual_solver_call_count=solver_call_count,
        admitted_fit_call_count=preflight.actual_fit_calls,
        admitted_predict_call_count=preflight.actual_predict_calls,
        admitted_metric_call_count=preflight.actual_metric_calls,
        admitted_solver_call_count=preflight.actual_solver_calls,
    )
    store.publish(
        category=_ALPHA_ECONOMIC_ATTESTATION_CATEGORY,
        value=attestation,
        identity_field="attestation_hash",
    )
    return attestation


def _alpha_economic_metric_call_count(economic: PanelAlphaEconomicEvaluation) -> int:
    """Count the fixed-book metrics represented by immutable economic content."""

    return len(economic.candidates) + sum(
        len(comparison.endpoints) for comparison in economic.comparisons
    )


def _load_admitted_alpha_economic_publication(
    *,
    store: PortfolioResearchArtifactStore,
    root: PanelMethodologyExecutionRoot,
) -> tuple[_AlphaEconomicPublicationAttestation, PanelAlphaEconomicEvaluation] | None:
    """Resolve run authority; legacy naked content remains load-only."""

    attestation_hash = root.alpha_economic_publication_attestation_hash
    if attestation_hash is None:
        if root.alpha_economic_evaluation_hash is not None:
            raise PanelMethodologyExecutionError(
                "alpha_research.alpha_economic_publication_attestation_missing"
            )
        return None
    if root.alpha_economic_evaluation_hash is not None:
        raise PanelMethodologyExecutionError(
            "alpha_research.alpha_economic_publication_lineage_invalid"
        )
    if not root.has_complete_worker_call_authority():
        raise PanelMethodologyExecutionError(
            "alpha_research.alpha_economic_publication_attempt_lineage_invalid"
        )
    attestation = store.load(
        category=_ALPHA_ECONOMIC_ATTESTATION_CATEGORY,
        content_hash=attestation_hash,
        model=_AlphaEconomicPublicationAttestation,
        identity_field="attestation_hash",
    )
    if attestation.run_publication_attempt_hash is None:
        raise PanelMethodologyExecutionError(
            "alpha_research.alpha_economic_run_publication_attestation_missing"
        )
    economic = store.load(
        category=_ALPHA_ECONOMIC_CONTENT_CATEGORY,
        content_hash=attestation.economic_content_hash,
        model=PanelAlphaEconomicEvaluation,
        identity_field="evaluation_hash",
    )
    folds = tuple(
        store.load(
            category="development/paired-panel-research/folds",
            content_hash=fold_hash,
            model=PanelModelFoldEvidence,
            identity_field="fold_hash",
        )
        for fold_hash in root.ordered_fold_hashes
    )
    if tuple(fold.fold_index for fold in folds) != tuple(range(len(folds))) or any(
        fold.program_hash != root.program_hash for fold in folds
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.alpha_economic_publication_lineage_invalid"
        )
    economic_metric_calls = _alpha_economic_metric_call_count(economic)
    graph_calls = (
        sum(fold.fit_call_count for fold in folds),
        sum(fold.predict_call_count for fold in folds),
        sum(fold.metric_call_count for fold in folds) + economic_metric_calls,
        0,
    )
    fold_by_hash = {fold.fold_hash: fold for fold in folds}
    attempt_calls = [0, 0, economic_metric_calls, 0]
    executed_fold_hashes: set[str] = set()
    for receipt in root.worker_resource_receipts:
        receipt_counts = (
            receipt.fit_call_count,
            receipt.predict_call_count,
            receipt.metric_call_count,
            receipt.solver_call_count,
        )
        if receipt.fold_hash is None or any(value is None for value in receipt_counts):
            raise PanelMethodologyExecutionError(
                "alpha_research.alpha_economic_publication_attempt_lineage_invalid"
            )
        fold = fold_by_hash.get(receipt.fold_hash)
        if (
            fold is None
            or receipt.fold_hash in executed_fold_hashes
            or fold.fold_index != receipt.fold_index
            or tuple(cast(tuple[int, int, int, int], receipt_counts))
            != (fold.fit_call_count, fold.predict_call_count, fold.metric_call_count, 0)
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.alpha_economic_publication_attempt_lineage_invalid"
            )
        executed_fold_hashes.add(receipt.fold_hash)
        for index, value in enumerate(cast(tuple[int, int, int, int], receipt_counts)):
            attempt_calls[index] += value
    attempt_expected = (
        attestation.attempt_executed_fold_count == root.executed_fold_count,
        attestation.attempt_reused_fold_count == root.reused_fold_count,
        root.executed_fold_count == len(executed_fold_hashes),
        root.reused_fold_count == len(folds) - len(executed_fold_hashes),
        tuple(attempt_calls)
        == (
            attestation.attempt_fit_call_count,
            attestation.attempt_predict_call_count,
            attestation.attempt_metric_call_count,
            attestation.attempt_solver_call_count,
        ),
    )
    if not all(attempt_expected):
        raise PanelMethodologyExecutionError(
            "alpha_research.alpha_economic_publication_attempt_lineage_invalid"
        )
    expected = (
        attestation.program_hash == root.program_hash,
        attestation.desk_program_hash == root.desk_program_hash,
        attestation.method_binding_hash == root.method_binding_hash,
        attestation.authority_hash == root.authority_hash,
        attestation.request_hash == root.request_hash,
        attestation.preflight_hash == root.preflight_hash,
        attestation.source_resolution_hash == root.source_resolution_hash,
        attestation.feature_preflight_hash == root.feature_preflight_hash,
        attestation.ordered_fold_hashes == root.ordered_fold_hashes,
        attestation.actual_fit_call_count == root.fit_call_count,
        attestation.actual_predict_call_count == root.predict_call_count,
        attestation.actual_metric_call_count == root.metric_call_count,
        attestation.actual_solver_call_count == root.solver_call_count,
        graph_calls
        == (
            root.fit_call_count,
            root.predict_call_count,
            root.metric_call_count,
            root.solver_call_count,
        ),
        graph_calls
        == (
            attestation.actual_fit_call_count,
            attestation.actual_predict_call_count,
            attestation.actual_metric_call_count,
            attestation.actual_solver_call_count,
        ),
    )
    if not all(expected):
        raise PanelMethodologyExecutionError(
            "alpha_research.alpha_economic_publication_lineage_invalid"
        )
    return attestation, economic


def _load_fixed_candidate_graph(
    *,
    output_workspace: Path,
    root_hash: str,
    model_recipe_id: str,
) -> tuple[
    PortfolioResearchArtifactStore,
    PanelMethodologyExecutionRoot,
    PanelAlphaEconomicEvaluation,
    AlphaEconomicCandidateEvidence,
    tuple[PanelModelFoldEvidence, ...],
    tuple[PanelModelTrialEvidence, ...],
    tuple[PanelScoreSurfaceArtifact, ...],
]:
    """Open the exact admitted Alpha graph once for downstream resolvers."""

    store = PortfolioResearchArtifactStore(Path(output_workspace) / "portfolio-development")
    root = store.load(
        category=PANEL_METHODOLOGY_ROOT_CATEGORY,
        content_hash=root_hash,
        model=PanelMethodologyExecutionRoot,
        identity_field="root_hash",
    )
    publication = _load_admitted_alpha_economic_publication(store=store, root=root)
    if publication is None or root.identity_class != "DEVELOPMENT_ONLY":
        raise PanelMethodologyExecutionError("alpha_research.fixed_candidate_root_not_admitted")
    _attestation, economic = publication
    candidate = next(
        (value for value in economic.candidates if value.model_recipe_id == model_recipe_id),
        None,
    )
    if candidate is None:
        raise PanelMethodologyExecutionError("alpha_research.fixed_candidate_recipe_not_in_root")
    folds = tuple(
        store.load(
            category="development/paired-panel-research/folds",
            content_hash=value,
            model=PanelModelFoldEvidence,
            identity_field="fold_hash",
        )
        for value in root.ordered_fold_hashes
    )
    if tuple(value.fold_index for value in folds) != tuple(range(len(folds))) or any(
        value.program_hash != root.program_hash for value in folds
    ):
        raise PanelMethodologyExecutionError("alpha_research.fixed_candidate_fold_axis_invalid")
    outer_trials = tuple(
        store.load(
            category="development/paired-panel-research/outer-trials",
            content_hash=value,
            model=PanelModelTrialEvidence,
            identity_field="evidence_hash",
        )
        for value in root.ordered_outer_trial_hashes
    )
    artifacts = tuple(
        store.load(
            category="development/paired-panel-research/score-surfaces",
            content_hash=value,
            model=PanelScoreSurfaceArtifact,
            identity_field="artifact_hash",
        )
        for value in root.ordered_score_surface_artifact_hashes
    )
    return store, root, economic, candidate, folds, outer_trials, artifacts


def resolve_fixed_candidate_score_surfaces(
    *,
    output_workspace: Path,
    root_hash: str,
    model_recipe_id: str,
) -> tuple[ResolvedCandidateScoreSurface, ...]:
    """Resolve one exact fixed recipe's OOF children without opening source data.

    ``output_workspace`` is Host runtime routing.  The durable scientific handle
    remains exactly ``root_hash + model_recipe_id``; no child path or five-hash
    list crosses the authoring boundary.
    """
    store, root, economic, candidate, folds, outer_trials, artifacts = _load_fixed_candidate_graph(
        output_workspace=output_workspace,
        root_hash=root_hash,
        model_recipe_id=model_recipe_id,
    )
    resolved: list[ResolvedCandidateScoreSurface] = []
    for fold in folds:
        matches = tuple(
            value
            for value in artifacts
            if value.fold_index == fold.fold_index
            and value.model_recipe_id == model_recipe_id
            and value.producer_spec_id is None
        )
        trials = tuple(
            value
            for value in outer_trials
            if value.fold_index == fold.fold_index
            and value.recipe_id == model_recipe_id
            and value.phase == "OUTER_VALIDATION"
        )
        if len(matches) != 1 or len(trials) != 1:
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_candidate_fold_child_not_unique"
            )
        artifact = matches[0]
        trial = trials[0]
        scientific = trial.scientific_evidence
        if scientific is None or not scientific.sessions:
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_candidate_oof_validation_missing"
            )
        sessions = tuple(scientific.sessions)
        score_payload = store.load_packed_bytes(
            category="development/paired-panel-research/score-lanes",
            content_hash=artifact.score_lane_hash,
        )
        target_payload = store.load_packed_bytes(
            category="development/paired-panel-research/target-z-lanes",
            content_hash=artifact.target_z_lane_hash,
        )
        raw_return_payload = store.load_packed_bytes(
            category="development/paired-panel-research/raw-simple-return-lanes",
            content_hash=artifact.raw_simple_return_lane_hash,
        )
        lane_row_counts = tuple(
            len(payload) // np.dtype(np.float64).itemsize
            for payload in (score_payload, target_payload, raw_return_payload)
        )
        lane_bytes_valid = all(
            len(payload) % np.dtype(np.float64).itemsize == 0
            for payload in (score_payload, target_payload, raw_return_payload)
        )
        try:
            economic_position = candidate.ordered_score_surface_hashes.index(artifact.artifact_hash)
        except ValueError as error:
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_candidate_economic_lineage_missing"
            ) from error
        if (
            artifact.source_surface_hash not in fold.fixed_candidate_surface_hashes
            or artifact.artifact_hash not in root.ordered_score_surface_artifact_hashes
            or not lane_bytes_valid
            or lane_row_counts != (artifact.row_count,) * 3
            or artifact.row_axis_hash != economic.ordered_common_row_axis_hashes[economic_position]
            or artifact.raw_simple_return_lane_hash
            != economic.ordered_raw_simple_return_lane_hashes[economic_position]
            or sessions != tuple(sorted(set(sessions)))
            or not set(sessions).issubset(root.common_formation_sessions)
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_candidate_score_lineage_invalid"
            )
        resolved.append(
            ResolvedCandidateScoreSurface.create(
                root_hash=root.root_hash,
                model_recipe_id=model_recipe_id,
                fold_index=fold.fold_index,
                artifact_hash=artifact.artifact_hash,
                source_surface_hash=artifact.source_surface_hash,
                formation_start=sessions[0],
                formation_end=sessions[-1],
                validation_session_count=len(sessions),
                validation_session_axis_hash=str(
                    canonical_hash([value.isoformat() for value in sessions])
                ),
                row_count=artifact.row_count,
                row_axis_hash=artifact.row_axis_hash,
                score_value_hash=artifact.score_lane_hash,
                target_z_lane_hash=artifact.target_z_lane_hash,
                raw_simple_return_lane_hash=artifact.raw_simple_return_lane_hash,
            )
        )
    if (
        len(resolved) != len(folds)
        or tuple(value.artifact_hash for value in resolved)
        != candidate.ordered_score_surface_hashes
    ):
        raise PanelMethodologyExecutionError("alpha_research.fixed_candidate_score_set_incomplete")
    return tuple(resolved)


def resolve_fixed_candidate_decision_folds(
    *, output_workspace: Path, root_hash: str, model_recipe_id: str
) -> tuple[PanelMethodologyOuterFold, ...]:
    """Derive Portfolio decision folds from the exact sealed Alpha graph."""
    _store, root, _economic, _candidate, folds, outer_trials, _artifacts = (
        _load_fixed_candidate_graph(
            output_workspace=output_workspace,
            root_hash=root_hash,
            model_recipe_id=model_recipe_id,
        )
    )
    resolved: list[PanelMethodologyOuterFold] = []
    for fold in folds:
        trials = tuple(
            value
            for value in outer_trials
            if value.fold_index == fold.fold_index
            and value.recipe_id == model_recipe_id
            and value.phase == "OUTER_VALIDATION"
        )
        if len(trials) != 1:
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_candidate_fold_child_not_unique"
            )
        scientific = trials[0].scientific_evidence
        if scientific is None or not scientific.sessions:
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_candidate_oof_validation_missing"
            )
        sessions = tuple(scientific.sessions)
        if sessions != tuple(sorted(set(sessions))) or not set(sessions).issubset(
            root.common_formation_sessions
        ):
            raise PanelMethodologyExecutionError("alpha_research.fixed_candidate_fold_axis_invalid")
        resolved.append(
            PanelMethodologyOuterFold(
                fold_index=fold.fold_index,
                # Portfolio never trains. Its fold owner is the immutable OOF
                # validation child, so no synthetic training axis is copied
                # into this downstream consumer view.
                training_sessions=(),
                validation_sessions=sessions,
                source_manifest_hash=fold.fold_hash,
            )
        )
    if (
        len(resolved) != len(folds)
        or tuple(sorted(session for value in resolved for session in value.validation_sessions))
        != root.common_formation_sessions
    ):
        raise PanelMethodologyExecutionError("alpha_research.fixed_candidate_fold_axis_invalid")
    return tuple(resolved)


def load_fixed_alpha_row_axes(
    *,
    output_workspace: Path,
    root_hash: str = SEALED_PANEL_ALPHA_ROOT_HASH,
    receipt_hash: str | None = None,
    panel_snapshot_hash: str | None = None,
) -> tuple[
    FixedAlphaRowAxisReceipt,
    tuple[tuple[tuple[date, ...], tuple[str, ...]], ...],
]:
    """Open the exact compact row-axis sidecar; never fall back to Feature."""
    if receipt_hash is None:
        raise PanelMethodologyExecutionError(
            "alpha_research.fixed_alpha_row_axis_receipt_handle_required"
        )
    alpha_store = AlphaDevelopmentArtifactStore(Path(output_workspace) / "portfolio-development")
    receipt = alpha_store.load_fixed_alpha_row_axis_receipt(receipt_hash)
    if panel_snapshot_hash is None or receipt.panel_snapshot_hash != panel_snapshot_hash:
        raise PanelMethodologyExecutionError(
            "alpha_research.fixed_alpha_row_axis_receipt_unavailable"
        )
    _store, root, _economic, _candidate, _folds, _trials, artifacts = _load_fixed_candidate_graph(
        output_workspace=output_workspace,
        root_hash=root_hash,
        model_recipe_id=receipt.fixed_model_recipe_id,
    )
    if (
        receipt.program_hash != root.program_hash
        or receipt.source_resolution_hash != root.source_resolution_hash
        or (receipt.root_hash is not None and receipt.root_hash != root_hash)
        or (receipt.root_hash is None and root.alpha_row_axis_receipt_hash != receipt.receipt_hash)
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.fixed_alpha_row_axis_receipt_unavailable"
        )
    axes: list[tuple[tuple[date, ...], tuple[str, ...]]] = []
    for fold in receipt.folds:
        fixed = tuple(
            value
            for value in artifacts
            if value.fold_index == fold.fold_index
            and value.model_recipe_id == receipt.fixed_model_recipe_id
            and value.producer_spec_id is None
        )
        control = tuple(
            value
            for value in artifacts
            if value.fold_index == fold.fold_index
            and value.model_recipe_id == receipt.control_model_recipe_id
            and value.producer_spec_id is None
        )
        if (
            len(fixed) != 1
            or len(control) != 1
            or fixed[0].artifact_hash != fold.fixed_score_artifact_hash
            or control[0].artifact_hash != fold.control_score_artifact_hash
            or fixed[0].row_axis_hash != fold.row_axis_hash
            or control[0].row_axis_hash != fold.row_axis_hash
            or fixed[0].score_lane_hash != fold.fixed_score_value_hash
            or control[0].score_lane_hash != fold.control_score_value_hash
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_alpha_row_axis_artifact_binding_invalid"
            )
        payload = alpha_store.load_fixed_alpha_row_axis_lane(fold.listing_position_lane_hash)
        position_dtype = np.dtype(fold.listing_position_dtype)
        if len(payload) != fold.row_count * position_dtype.itemsize:
            raise PanelMethodologyExecutionError("alpha_research.fixed_alpha_row_axis_lane_invalid")
        positions: npt.NDArray[np.uint16] = np.frombuffer(payload, dtype=position_dtype)
        if bool(np.any(positions >= len(receipt.ordered_listing_ids))):
            raise PanelMethodologyExecutionError("alpha_research.fixed_alpha_row_axis_lane_invalid")
        sessions = tuple(
            session
            for session, count in zip(
                fold.validation_sessions, fold.row_count_by_session, strict=True
            )
            for _row in range(count)
        )
        listings = tuple(receipt.ordered_listing_ids[int(value)] for value in positions)
        if (
            canonical_hash(
                [
                    (session.isoformat(), listing)
                    for session, listing in zip(sessions, listings, strict=True)
                ]
            )
            != fold.row_axis_hash
        ):
            raise PanelMethodologyExecutionError("alpha_research.fixed_alpha_row_axis_lane_invalid")
        axes.append((sessions, listings))
    return receipt, tuple(axes)


@dataclass(frozen=True, slots=True)
class FixedAlphaFeatureBinding:
    source_resolution_hash: str
    feature_preflight_hash: str
    feature_count_by_method: MappingProxyType[str, int]
    feature_axis_hash_by_method: MappingProxyType[str, str]
    view_binding_hash_by_method: MappingProxyType[str, str]


def resolve_fixed_alpha_feature_binding(
    *,
    output_workspace: Path,
    root_hash: str = SEALED_PANEL_ALPHA_ROOT_HASH,
) -> FixedAlphaFeatureBinding:
    """Derive Feature identity from fixed trials/estimators, never Feature values."""
    from alphalattice.capabilities.alpha_modeling.contracts import AlphaEstimatorContent

    store, root, _economic, _candidate, folds, trials, _artifacts = _load_fixed_candidate_graph(
        output_workspace=output_workspace,
        root_hash=root_hash,
        model_recipe_id=SEALED_PANEL_ALPHA_MODEL_RECIPE_ID,
    )
    recipes = {
        "RELATIVE_CONTROL": "RELATIVE_CONTROL_RIDGE",
        "SPARSE_SESSION_AMPLITUDE": SEALED_PANEL_ALPHA_MODEL_RECIPE_ID,
    }
    counts: dict[str, int] = {}
    axes: dict[str, str] = {}
    bindings: dict[str, str] = {}
    for method_id, recipe_id in recipes.items():
        selected = tuple(
            value
            for value in trials
            if value.phase == "OUTER_VALIDATION" and value.recipe_id == recipe_id
        )
        if tuple(value.fold_index for value in selected) != tuple(range(len(folds))):
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_alpha_feature_receipt_incomplete"
            )
        estimator_axes: list[tuple[str, ...]] = []
        for trial in selected:
            estimator = store.load(
                category="development/paired-panel-research/estimators",
                content_hash=trial.estimator_content_hash,
                model=AlphaEstimatorContent,
                identity_field="content_hash",
            )
            if canonical_hash(list(estimator.ordered_feature_ids)) != trial.feature_axis_hash:
                raise PanelMethodologyExecutionError(
                    "alpha_research.fixed_alpha_feature_receipt_invalid"
                )
            estimator_axes.append(estimator.ordered_feature_ids)
        if len(set(estimator_axes)) != 1:
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_alpha_feature_receipt_invalid"
            )
        ordered = estimator_axes[0]
        counts[method_id] = len(ordered)
        axes[method_id] = str(canonical_hash(list(ordered)))
        bindings[method_id] = str(
            canonical_hash(
                {
                    "method_id": method_id,
                    "model_recipe_id": recipe_id,
                    "feature_preflight_hash": root.feature_preflight_hash,
                    "feature_count": len(ordered),
                    "feature_axis_hash": axes[method_id],
                }
            )
        )
    if counts.get("SPARSE_SESSION_AMPLITUDE") != 195:
        raise PanelMethodologyExecutionError("alpha_research.fixed_alpha_feature_receipt_invalid")
    return FixedAlphaFeatureBinding(
        source_resolution_hash=root.source_resolution_hash,
        feature_preflight_hash=root.feature_preflight_hash,
        feature_count_by_method=MappingProxyType(counts),
        feature_axis_hash_by_method=MappingProxyType(axes),
        view_binding_hash_by_method=MappingProxyType(bindings),
    )


def publish_panel_portfolio_preflight_receipt(
    *,
    output_workspace: Path,
    program: SealedResearchProgram,
    request: PanelResearchMethodologyRequest,
    preflight: PanelResearchPreflight,
    inputs: PanelMethodologyExecutionInputs,
) -> PanelPortfolioPreflightReceipt:
    """Seal prepared Portfolio inputs once; no Alpha Feature view is opened here."""
    source_axis = inputs.portfolio_axis_authority
    axis = inputs.portfolio_axis
    row_receipt = inputs.fixed_alpha_row_axis_receipt
    r0 = inputs.r0_covariance
    r1 = inputs.r1_covariance
    source_handles = (
        inputs.total_return_target_evidence_hash,
        inputs.outcome_method_binding_hash,
        inputs.panel_derivation_recipe_hash,
        inputs.risk_input_binding_hash,
        inputs.risk_return_surface_hash,
        inputs.benchmark_surface_hash,
    )
    if (
        request.execution_stage != "PORTFOLIO_ONLY"
        or source_axis is None
        or axis is None
        or row_receipt is None
        or r0 is None
        or inputs.alpha_panel_source_identity is None
        or request.fixed_score_method_id is None
        or request.top_k is None
        or inputs.feature_materialization_count != 0
        or any(value is None for value in source_handles)
        or inputs.market_tradability_bundle_hash is None
        or inputs.market_universe_epoch_hash is None
        or inputs.market_sector_revision is None
        or inputs.market_source_surface_hash is None
        or inputs.portfolio_state_transition is None
        or inputs.portfolio_preflight_source_hash is None
    ):
        raise PanelMethodologyExecutionError("alpha_research.portfolio_preflight_inputs_incomplete")
    if (
        row_receipt.source_resolution_hash
        != inputs.alpha_panel_source_identity.source_resolution_hash
        or row_receipt.panel_snapshot_hash != inputs.alpha_panel_source_identity.panel_snapshot_hash
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.fixed_alpha_row_axis_receipt_unavailable"
        )
    store = PortfolioResearchArtifactStore(output_workspace / "portfolio-development")
    started = perf_counter()
    if axis.economic_formation_sessions != tuple(
        value
        for value in source_axis.formation_sessions
        if axis.economic_formation_sessions[0] <= value <= axis.economic_formation_sessions[-1]
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_preflight_owner_axis_mismatch"
        )
    simple_artifact_hash = inputs.fixed_simple_score_value_artifact_hash
    simple_binding_hash = inputs.fixed_simple_score_binding_hash
    binding = inputs.fixed_simple_score_binding
    if binding is None or simple_artifact_hash is None or simple_binding_hash is None:
        raise PanelMethodologyExecutionError(
            "alpha_research.simple_signal_value_artifact_handle_required"
        )
    portfolio_listings = frozenset(axis.ordered_listing_ids)
    projected_simple_listing_ids = tuple(
        listing_id for listing_id in binding.ordered_listing_ids if listing_id in portfolio_listings
    )
    if (
        binding.binding_hash != simple_binding_hash
        or tuple(
            value
            for value in binding.ordered_formation_sessions
            if value in set(axis.formation_sessions)
        )
        != axis.formation_sessions
        or binding.ordered_listing_ids != row_receipt.ordered_listing_ids
        or projected_simple_listing_ids != axis.ordered_listing_ids
        or binding.minimum_finite_listings_observed < request.top_k
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.fixed_simple_score_common_axis_incomplete"
        )
    if inputs.portfolio_preflight_source_hash != panel_portfolio_preflight_source_hash(
        r0_surface_hash=r0.surface_hash,
        r1_surface_hash=None if r1 is None else r1.surface_hash,
        alpha_panel_source_identity=inputs.alpha_panel_source_identity,
        total_return_target_evidence_hash=inputs.total_return_target_evidence_hash,
        outcome_method_binding_hash=inputs.outcome_method_binding_hash,
        panel_derivation_recipe_hash=inputs.panel_derivation_recipe_hash,
        risk_input_binding_hash=inputs.risk_input_binding_hash,
        risk_return_surface_hash=inputs.risk_return_surface_hash,
        market_tradability_bundle_hash=inputs.market_tradability_bundle_hash,
        market_universe_epoch_hash=inputs.market_universe_epoch_hash,
        market_sector_revision=inputs.market_sector_revision,
        market_source_surface_hash=inputs.market_source_surface_hash,
        benchmark_surface_hash=cast(str, inputs.benchmark_surface_hash),
        state_transition=inputs.portfolio_state_transition,
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_preflight_source_identity_invalid"
        )

    simple_resolution_seconds = perf_counter() - started
    receipt = PanelPortfolioPreflightReceipt.create(
        program_hash=program.program_hash,
        authority_hash=program.authority_hash,
        preflight=preflight,
        feature_count_by_method=inputs.fixed_feature_count_by_method,
        feature_axis_hash_by_method=inputs.fixed_feature_axis_hash_by_method,
        feature_view_binding_hash_by_method=(inputs.fixed_feature_view_binding_hash_by_method),
        fixed_alpha_root_hash=SEALED_PANEL_ALPHA_ROOT_HASH,
        fixed_alpha_row_axis_receipt_hash=row_receipt.receipt_hash,
        ordered_fold_hashes=tuple(value.source_manifest_hash for value in inputs.outer_folds),
        r0_surface_hash=r0.surface_hash,
        r1_surface_hash=None if r1 is None else r1.surface_hash,
        alpha_panel_source_identity=inputs.alpha_panel_source_identity,
        total_return_target_evidence_hash=inputs.total_return_target_evidence_hash,
        outcome_method_binding_hash=inputs.outcome_method_binding_hash,
        panel_derivation_recipe_hash=inputs.panel_derivation_recipe_hash,
        risk_input_binding_hash=inputs.risk_input_binding_hash,
        risk_return_surface_hash=inputs.risk_return_surface_hash,
        market_tradability_bundle_hash=inputs.market_tradability_bundle_hash,
        market_universe_epoch_hash=inputs.market_universe_epoch_hash,
        market_sector_revision=inputs.market_sector_revision,
        market_source_surface_hash=inputs.market_source_surface_hash,
        benchmark_surface_hash=inputs.benchmark_surface_hash,
        state_transition=inputs.portfolio_state_transition,
        preflight_source_hash=inputs.portfolio_preflight_source_hash,
        portfolio_formation_sessions=axis.formation_sessions,
        portfolio_economic_formation_sessions=axis.economic_formation_sessions,
        portfolio_decision_ranges=axis.decision_ranges,
        portfolio_passive_sessions=axis.passive_sessions,
        portfolio_ordered_listing_ids=axis.ordered_listing_ids,
        portfolio_session_axis_hash=axis.session_axis_hash,
        portfolio_economic_session_axis_hash=axis.economic_session_axis_hash,
        portfolio_listing_axis_hash=axis.listing_axis_hash,
        portfolio_axis_hash=axis.axis_hash,
        simple_score_binding_hash=binding.binding_hash,
        simple_score_value_artifact_hash=simple_artifact_hash,
        feature_materialization_count=0,
        feature_materialization_seconds=inputs.feature_materialization_seconds,
        simple_score_materialization_count=0,
        simple_score_materialization_seconds=0.0,
        owner_resolution_seconds=inputs.owner_resolution_seconds + simple_resolution_seconds,
        alpha_verification_seconds=inputs.alpha_verification_seconds,
    )
    store.publish(
        category=_PORTFOLIO_PREFLIGHT_CATEGORY,
        value=receipt,
        identity_field="receipt_hash",
    )
    binding_index = PanelPortfolioPreflightBinding.create(
        program_hash=program.program_hash,
        receipt_hash=receipt.receipt_hash,
    )
    store.publish(
        category=_PORTFOLIO_PREFLIGHT_BINDING_CATEGORY,
        value=binding_index,
        identity_field="program_hash",
    )
    return receipt


def load_panel_portfolio_preflight_receipt(
    *, output_workspace: Path, program_hash: str
) -> PanelPortfolioPreflightReceipt:
    """Load one Program's exact preflight receipt without reopening source owners."""
    store = PortfolioResearchArtifactStore(output_workspace / "portfolio-development")
    binding = store.load(
        category=_PORTFOLIO_PREFLIGHT_BINDING_CATEGORY,
        content_hash=program_hash,
        model=PanelPortfolioPreflightBinding,
        identity_field="program_hash",
    )
    if binding.program_hash != program_hash:
        raise PanelMethodologyExecutionError("alpha_research.portfolio_preflight_program_mismatch")
    try:
        receipt = store.load(
            category=_PORTFOLIO_PREFLIGHT_CATEGORY,
            content_hash=binding.receipt_hash,
            model=PanelPortfolioPreflightReceipt,
            identity_field="receipt_hash",
        )
    except PortfolioPublicationError:
        receipt = store.load(
            category=_PORTFOLIO_PREFLIGHT_CATEGORY,
            content_hash=binding.receipt_hash,
            model=_HistoricalPanelPortfolioPreflightReceipt,
            identity_field="receipt_hash",
        )
    if receipt.program_hash != program_hash:
        raise PanelMethodologyExecutionError("alpha_research.portfolio_preflight_program_mismatch")
    return receipt


def inputs_from_panel_portfolio_preflight_receipt(
    *,
    output_workspace: Path,
    receipt: PanelPortfolioPreflightReceipt,
    r0_covariance: ResolvedStageSixCovariance,
    r1_covariance: ResolvedStageSixCovariance | None,
    simple_score_source: WorkspaceSimpleSignalSource,
    portfolio_market: ImmutablePortfolioMarketInputs,
    portfolio_benchmark: ImmutablePortfolioBenchmark,
    reference_marks: ResolvedReferenceMark,
    execution_clock: ExecutionClockContext,
    alpha_panel_source_identity: AlphaPanelSourceIdentity,
    panel_score_readiness: PanelScoreFormationReadinessReceipt,
    receipt_load_seconds: float,
) -> PanelMethodologyExecutionInputs:
    """Rehydrate only bytes named by a sealed preflight receipt."""
    if (
        r0_covariance.surface_hash != receipt.r0_surface_hash
        or (
            (r1_covariance is None) != (receipt.r1_surface_hash is None)
            or (r1_covariance is not None and r1_covariance.surface_hash != receipt.r1_surface_hash)
        )
        or alpha_panel_source_identity != receipt.alpha_panel_source_identity
        or portfolio_market.tradability_bundle_hash != receipt.market_tradability_bundle_hash
        or portfolio_market.universe_epoch_hash != receipt.market_universe_epoch_hash
        or portfolio_market.sector_revision != receipt.market_sector_revision
        or portfolio_market.source_surface_hash != receipt.market_source_surface_hash
        or portfolio_benchmark.surface_hash != receipt.benchmark_surface_hash
        or receipt.state_transition is None
        or reference_marks.binding.binding_hash != receipt.state_transition.binding_hash
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_preflight_risk_identity_mismatch"
        )
    row_receipt, row_axes = load_fixed_alpha_row_axes(
        output_workspace=output_workspace,
        root_hash=receipt.fixed_alpha_root_hash,
        receipt_hash=receipt.fixed_alpha_row_axis_receipt_hash,
        panel_snapshot_hash=receipt.alpha_panel_source_identity.panel_snapshot_hash,
    )
    if len(row_receipt.folds) != len(receipt.ordered_fold_hashes):
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_preflight_fold_identity_mismatch"
        )
    folds = tuple(
        PanelMethodologyOuterFold(
            fold_index=value.fold_index,
            training_sessions=(),
            validation_sessions=value.validation_sessions,
            source_manifest_hash=fold_hash,
        )
        for value, fold_hash in zip(row_receipt.folds, receipt.ordered_fold_hashes, strict=True)
    )
    fold_by_session = {
        session: value.fold_index
        for value in row_receipt.folds
        for session in value.validation_sessions
    }
    try:
        portfolio_fold_indices = tuple(
            fold_by_session[value] for value in receipt.portfolio_formation_sessions
        )
    except KeyError as error:
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_preflight_fold_identity_mismatch"
        ) from error
    market_positions = {
        value: index for index, value in enumerate(portfolio_market.ordered_listing_ids)
    }
    r0_positions = {value: index for index, value in enumerate(r0_covariance.ordered_listing_ids)}
    r1_positions = (
        {value: index for index, value in enumerate(r1_covariance.ordered_listing_ids)}
        if r1_covariance is not None
        else None
    )
    try:
        axis = PanelPortfolioAxis(
            formation_sessions=receipt.portfolio_formation_sessions,
            economic_formation_sessions=receipt.portfolio_economic_formation_sessions,
            decision_ranges=receipt.portfolio_decision_ranges,
            passive_sessions=receipt.portfolio_passive_sessions,
            ordered_listing_ids=receipt.portfolio_ordered_listing_ids,
            market_listing_positions=cast(
                IntArray,
                _readonly(
                    np.asarray(
                        [
                            market_positions[value]
                            for value in receipt.portfolio_ordered_listing_ids
                        ],
                        dtype=np.int64,
                    ),
                    dtype=np.dtype(np.int64),
                ),
            ),
            r0_listing_positions=cast(
                IntArray,
                _readonly(
                    np.asarray(
                        [r0_positions[value] for value in receipt.portfolio_ordered_listing_ids],
                        dtype=np.int64,
                    ),
                    dtype=np.dtype(np.int64),
                ),
            ),
            r1_listing_positions=(
                cast(
                    IntArray,
                    _readonly(
                        np.asarray(
                            [
                                r1_positions[value]
                                for value in receipt.portfolio_ordered_listing_ids
                            ],
                            dtype=np.int64,
                        ),
                        dtype=np.dtype(np.int64),
                    ),
                )
                if r1_positions is not None
                else None
            ),
            session_axis_hash=receipt.portfolio_session_axis_hash,
            economic_session_axis_hash=receipt.portfolio_economic_session_axis_hash,
            listing_axis_hash=receipt.portfolio_listing_axis_hash,
            axis_hash=receipt.portfolio_axis_hash,
        )
    except KeyError as error:
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_preflight_listing_axis_mismatch"
        ) from error
    market, benchmark, selected_fold_indices = _subset_market(
        market=portfolio_market,
        benchmark=portfolio_benchmark,
        fold_indices=cast(
            IntArray,
            _readonly(
                np.asarray(portfolio_fold_indices, dtype=np.int64),
                dtype=np.dtype(np.int64),
            ),
        ),
        sessions=axis.formation_sessions,
        economic_sessions=axis.economic_formation_sessions,
        passive_sessions=axis.passive_sessions,
        listings=axis.ordered_listing_ids,
        listing_positions=axis.market_listing_positions,
    )
    if (
        receipt.simple_score_binding_hash is None
        or receipt.simple_score_value_artifact_hash is None
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.simple_signal_value_artifact_handle_required"
        )
    durable_simple = simple_score_source.load_durable_projection(
        artifact_hash=receipt.simple_score_value_artifact_hash,
        binding_hash=receipt.simple_score_binding_hash,
        formation_sessions=receipt.portfolio_formation_sessions,
        ordered_listing_ids=receipt.portfolio_ordered_listing_ids,
    )
    simple_values = durable_simple.standardized_values
    if (
        simple_values.shape
        != (
            len(receipt.portfolio_formation_sessions),
            len(receipt.portfolio_ordered_listing_ids),
        )
        or not np.isfinite(simple_values).all()
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_preflight_simple_score_identity_mismatch"
        )
    counts = dict(receipt.feature_count_by_method)
    axes = dict(receipt.feature_axis_hash_by_method)
    return PanelMethodologyExecutionInputs(
        plan=None,
        source_resolution_hash=receipt.preflight.source_resolution_hash,
        feature_preflight_hash=receipt.preflight.feature_preflight_hash,
        outer_folds=folds,
        fixed_feature_column_count=counts.get("SPARSE_SESSION_AMPLITUDE"),
        fixed_feature_axis_hash=axes.get("SPARSE_SESSION_AMPLITUDE"),
        fixed_feature_count_by_method=receipt.feature_count_by_method,
        fixed_feature_axis_hash_by_method=receipt.feature_axis_hash_by_method,
        fixed_feature_view_binding_hash_by_method=(receipt.feature_view_binding_hash_by_method),
        fixed_alpha_row_axis_receipt=row_receipt,
        fixed_alpha_row_axes=row_axes,
        portfolio_market=market,
        portfolio_axis_authority=market,
        portfolio_benchmark=benchmark,
        portfolio_fold_indices=selected_fold_indices,
        r0_covariance=r0_covariance,
        r1_covariance=r1_covariance,
        portfolio_axis=axis,
        simple_score_authority=None,
        simple_score_source=simple_score_source,
        fixed_simple_score_binding_hash=receipt.simple_score_binding_hash,
        fixed_simple_score_value_artifact_hash=receipt.simple_score_value_artifact_hash,
        total_return_target_evidence_hash=receipt.total_return_target_evidence_hash,
        outcome_method_binding_hash=receipt.outcome_method_binding_hash,
        panel_derivation_recipe_hash=receipt.panel_derivation_recipe_hash,
        risk_input_binding_hash=receipt.risk_input_binding_hash,
        risk_return_surface_hash=receipt.risk_return_surface_hash,
        benchmark_surface_hash=receipt.benchmark_surface_hash,
        market_tradability_bundle_hash=receipt.market_tradability_bundle_hash,
        market_universe_epoch_hash=receipt.market_universe_epoch_hash,
        market_sector_revision=receipt.market_sector_revision,
        market_source_surface_hash=receipt.market_source_surface_hash,
        portfolio_state_transition=receipt.state_transition,
        portfolio_preflight_source_hash=receipt.preflight_source_hash,
        fixed_simple_score_binding=durable_simple.binding,
        fixed_simple_score_values=simple_values,
        alpha_panel_source_identity=alpha_panel_source_identity,
        panel_score_readiness=panel_score_readiness,
        execution_clock=execution_clock,
        reference_marks=reference_marks,
        feature_materialization_count=0,
        feature_materialization_seconds=0.0,
        owner_resolution_seconds=0.0,
        alpha_verification_seconds=0.0,
        sealed_receipt_load_seconds=receipt_load_seconds,
        sealed_preflight_program_hash=receipt.program_hash,
        simple_score_materialization_count=0,
        simple_score_materialization_seconds=0.0,
    )


def _admit_parent_resource_usage(usage: ProcessResourceUsage) -> None:
    if (
        usage.average_machine_cpu_percent > 50.0
        or usage.peak_machine_cpu_percent > 50.0
        or usage.peak_rss_bytes > PANEL_ALPHA_PARENT_RSS_RESERVATION_BYTES
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.panel_methodology_parent_capacity_exceeded"
        )


def _admit_execution_resource_usage(
    *,
    runtime_caps: PanelResearchRuntimeCaps,
    usage: ProcessResourceUsage,
) -> None:
    """Refuse a new planned execution unless its full process tree was observed."""

    _admit_parent_resource_usage(usage)
    if runtime_caps.runtime_plan_hash is None:
        return
    if (
        usage.measurement_scope != "PARENT_AND_LIVE_DESCENDANTS"
        or usage.instrumentation_limitation is not None
        or usage.process_tree_peak_rss_bytes <= 0
        or usage.process_tree_peak_rss_bytes < usage.peak_rss_bytes
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.panel_methodology_process_tree_measurement_required"
        )
    if usage.process_tree_peak_rss_bytes > runtime_caps.maximum_estimated_peak_memory_bytes:
        raise PanelMethodologyExecutionError(
            "alpha_research.panel_methodology_process_tree_capacity_exceeded"
        )


def _execution_resource_receipt_fields(
    *,
    runtime_caps: PanelResearchRuntimeCaps,
    usage: ProcessResourceUsage,
    worker_capacity_reservation_upper_bound_bytes: int = 0,
    worker_resource_receipts: tuple[_PanelFoldResourceReceipt, ...] = (),
) -> dict[str, object]:
    """Bind applied operational limits to the root without making them science."""

    caps = runtime_caps
    if caps.runtime_plan_hash is None:
        return {
            "resource_measurement_scope": "PARENT_PROCESS_ONLY",
            "worker_capacity_reservation_upper_bound_bytes": (
                worker_capacity_reservation_upper_bound_bytes
            ),
            "worker_resource_receipts": worker_resource_receipts,
        }
    return {
        "resource_measurement_scope": "TIME_ALIGNED_PROCESS_TREE",
        "worker_capacity_reservation_upper_bound_bytes": (
            worker_capacity_reservation_upper_bound_bytes
        ),
        "process_tree_peak_cpu_measurement": "TIME_ALIGNED_PARENT_AND_LIVE_DESCENDANTS",
        "process_tree_peak_rss_measurement": "TIME_ALIGNED_PARENT_AND_LIVE_DESCENDANTS",
        "process_tree_peak_rss_bytes": usage.process_tree_peak_rss_bytes,
        "process_tree_peak_live_descendant_count": (usage.process_tree_peak_live_descendant_count),
        "process_tree_sample_interval_seconds": usage.sample_interval_seconds,
        "process_tree_instrumentation_limitation": usage.instrumentation_limitation,
        "runtime_plan_hash": caps.runtime_plan_hash,
        "runtime_workload": caps.runtime_workload,
        "runtime_profile": caps.runtime_profile,
        "runtime_profile_resolution": caps.runtime_profile_resolution,
        "runtime_process_logical_processor_limit": caps.process_logical_processor_limit,
        "runtime_applied_logical_processor_ids": caps.applied_logical_processor_ids,
        "runtime_fold_workers": caps.maximum_fold_workers,
        "runtime_lightgbm_threads_per_fit": caps.lightgbm_threads_per_fit,
        "runtime_duckdb_threads": caps.duckdb_threads,
        "runtime_blas_threads": caps.blas_threads,
        "worker_resource_receipts": worker_resource_receipts,
    }


def _admit_fold_before_publication(
    *,
    worker: _MonitoredPanelAlphaFoldWorkerResult,
    request: PanelResearchMethodologyRequest,
) -> _PanelFoldResourceReceipt:
    """Validate one worker's sealed resource and numerical budgets before writes."""

    usage = worker.resource_usage
    if (
        usage.average_machine_cpu_percent > 50.0
        or usage.peak_machine_cpu_percent > 50.0
        or usage.peak_rss_bytes > PANEL_ALPHA_FOLD_RSS_FUSE_BYTES
    ):
        raise PanelMethodologyExecutionError("alpha_research.panel_fold_worker_capacity_exceeded")
    receipt = _fold_resource_receipt(worker)
    _candidate_count, fit_limit, predict_limit, metric_limit = alpha_model_call_upper_bounds(
        request, fold_count=1
    )
    fold = worker.result.fold.fold_evidence
    if (
        fold.fit_call_count > fit_limit
        or fold.predict_call_count > predict_limit
        or fold.metric_call_count > metric_limit
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.panel_fold_preflight_call_bound_exceeded"
        )
    return receipt


@dataclass(frozen=True, slots=True)
class PanelPortfolioAxis:
    """Bind common decision/economic/listing axes and immutable source position projections."""

    formation_sessions: tuple[date, ...]
    economic_formation_sessions: tuple[date, ...]
    decision_ranges: tuple[tuple[int, int], ...]
    passive_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    market_listing_positions: IntArray
    r0_listing_positions: IntArray
    r1_listing_positions: IntArray | None
    session_axis_hash: str
    economic_session_axis_hash: str
    listing_axis_hash: str
    axis_hash: str


def _readonly(values: npt.NDArray[Any], *, dtype: np.dtype[Any]) -> npt.NDArray[Any]:
    result = np.ascontiguousarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _publish_score_surface(
    *,
    store: PortfolioResearchArtifactStore,
    surface: PanelScoreSurface,
    producer_spec_id: str | None = None,
    upstream_raw_score_artifact_hash: str | None = None,
    filter_spec_hash: str | None = None,
    filter_evidence_hash: str | None = None,
    fold_selection_hash: str | None = None,
) -> PanelScoreSurfaceArtifact:
    def publish(name: str, values: FloatArray) -> str:
        return str(
            store.publish_array(
                category=f"development/paired-panel-research/{name}",
                values=np.ascontiguousarray(values, dtype=np.float64),
            )
        )

    score_lane_hash = publish("score-lanes", surface.scores)
    artifact = PanelScoreSurfaceArtifact.create(
        source_surface_hash=surface.surface_hash,
        fold_index=surface.fold_index,
        surface_id=surface.surface_id,
        span_sessions=surface.span_sessions,
        row_count=len(surface.row_sessions),
        row_axis_hash=str(
            canonical_hash(
                [
                    (session.isoformat(), listing)
                    for session, listing in zip(
                        surface.row_sessions, surface.row_listing_ids, strict=True
                    )
                ]
            )
        ),
        score_lane_hash=score_lane_hash,
        target_z_lane_hash=publish("target-z-lanes", surface.target_z),
        raw_simple_return_lane_hash=publish("raw-simple-return-lanes", surface.raw_simple_returns),
        model_recipe_id=surface.model_recipe_id,
        producer_spec_id=producer_spec_id,
        upstream_raw_score_artifact_hash=upstream_raw_score_artifact_hash,
        filter_spec_hash=filter_spec_hash,
        filter_evidence_hash=filter_evidence_hash,
        fold_selection_hash=fold_selection_hash,
        output_score_value_hash=(score_lane_hash if producer_spec_id is not None else None),
    )
    store.publish(
        category="development/paired-panel-research/score-surfaces",
        value=artifact,
        identity_field="artifact_hash",
    )
    return artifact


def _publish_alpha_fold_worker_result(
    *, store: PortfolioResearchArtifactStore, worker: PanelAlphaFoldWorkerResult
) -> tuple[PanelScoreSurfaceArtifact, ...]:
    """Publish one complete fold atomically enough for fold-level resume.

    The fold receipt is deliberately written last. Its presence therefore means
    every child needed for resume was already content-addressed by its real owner.
    An interruption before that final write leaves harmless orphan children, not
    a falsely complete checkpoint.
    """

    result = worker.fold
    for receipt in worker.scale_receipts:
        store.publish(
            category="development/paired-panel-research/scale-receipts",
            value=receipt,
            identity_field="receipt_hash",
        )
    for evidence in result.inner_evidence:
        store.publish(
            category="development/paired-panel-research/inner-trials",
            value=evidence,
            identity_field="evidence_hash",
        )
    for selection in result.group_selections:
        store.publish(
            category="development/paired-panel-research/selections",
            value=selection,
            identity_field="selection_hash",
        )
    for selection in (result.deployable_selection, result.dynamic_challenger_selection):
        store.publish(
            category="development/paired-panel-research/selections",
            value=selection,
            identity_field="selection_hash",
        )
    if result.score_aggregation_selection is not None:
        store.publish(
            category="development/paired-panel-research/selections",
            value=result.score_aggregation_selection,
            identity_field="selection_hash",
        )
    for evidence in result.outer_evidence:
        store.publish(
            category="development/paired-panel-research/outer-trials",
            value=evidence,
            identity_field="evidence_hash",
        )
    for content in result.model_contents:
        store.publish(
            category="development/paired-panel-research/estimators",
            value=content.estimator,
            identity_field="content_hash",
        )
        store.publish(
            category="development/paired-panel-research/state-projections",
            value=content.execution.fit.state_projection,
            identity_field="projection_hash",
        )
        store.publish(
            category="development/paired-panel-research/provenance",
            value=content.execution.provenance,
            identity_field="provenance_hash",
        )
        store.publish(
            category="development/paired-panel-research/environments",
            value=content.execution.numerical_environment,
            identity_field="environment_hash",
        )
    fixed = tuple(
        _publish_score_surface(store=store, surface=surface)
        for surface in result.fixed_candidate_surfaces
    )
    if result.raw_dynamic_surface.surface_hash not in {
        value.source_surface_hash for value in fixed
    } or result.common_span_control_surface.surface_hash not in {
        value.source_surface_hash for value in fixed
    }:
        raise PanelMethodologyExecutionError(
            "alpha_research.fixed_candidate_publication_incomplete"
        )
    store.publish(
        category="development/paired-panel-research/folds",
        value=result.fold_evidence,
        identity_field="fold_hash",
    )
    return fixed


def _verify_alpha_fold_checkpoint(
    *, store: PortfolioResearchArtifactStore, fold: PanelModelFoldEvidence
) -> None:
    """Read every child that makes the fold receipt eligible for resume."""

    from alphalattice.capabilities.alpha_modeling.contracts import (
        AlphaEstimatorContent,
        AlphaFitProvenanceReceipt,
        AlphaModelStateProjection,
    )
    from alphalattice.capabilities.alpha_modeling.runtime.numerical_environment import (
        AlphaModelNumericalEnvironment,
    )
    from alphalattice.investment.alpha_research.experiments.panel_methodology_statistics import (
        CausalCandidateSelection,
        CausalModelSpecSelection,
        CausalScoreAggregationSelection,
    )
    from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
        FoldScaleReceipt,
    )

    selection_model = (
        PanelModelScientificSelection
        if fold.selection_contract_kind == "PanelModelScientificSelection"
        else (
            CausalModelSpecSelection
            if fold.score_aggregation_selection_hash is not None
            else CausalCandidateSelection
        )
    )
    for selection_hash in (
        *fold.group_selection_hashes,
        fold.deployable_selection_hash,
        fold.dynamic_challenger_selection_hash,
    ):
        selection = store.load(
            category="development/paired-panel-research/selections",
            content_hash=selection_hash,
            model=selection_model,
            identity_field="selection_hash",
        )
        if selection.fold_index != fold.fold_index:
            raise PanelMethodologyExecutionError(
                "alpha_research.alpha_fold_selection_identity_mismatch"
            )
    if fold.score_aggregation_selection_hash is not None:
        aggregation = store.load(
            category="development/paired-panel-research/selections",
            content_hash=fold.score_aggregation_selection_hash,
            model=CausalScoreAggregationSelection,
            identity_field="selection_hash",
        )
        if aggregation.fold_index != fold.fold_index:
            raise PanelMethodologyExecutionError(
                "alpha_research.alpha_fold_selection_identity_mismatch"
            )
    for category, hashes in (
        ("development/paired-panel-research/inner-trials", fold.inner_trial_evidence_hashes),
        ("development/paired-panel-research/outer-trials", fold.outer_trial_evidence_hashes),
    ):
        for trial_hash in hashes:
            trial = store.load(
                category=category,
                content_hash=trial_hash,
                model=PanelModelTrialEvidence,
                identity_field="evidence_hash",
            )
            if trial.program_hash != fold.program_hash or trial.fold_index != fold.fold_index:
                raise PanelMethodologyExecutionError(
                    "alpha_research.alpha_fold_trial_identity_mismatch"
                )
            for receipt_hash in trial.scale_receipt_hashes:
                store.load(
                    category="development/paired-panel-research/scale-receipts",
                    content_hash=receipt_hash,
                    model=FoldScaleReceipt,
                    identity_field="receipt_hash",
                )
            estimator = store.load(
                category="development/paired-panel-research/estimators",
                content_hash=trial.estimator_content_hash,
                model=AlphaEstimatorContent,
                identity_field="content_hash",
            )
            store.load(
                category="development/paired-panel-research/state-projections",
                content_hash=trial.state_projection_hash,
                model=AlphaModelStateProjection,
                identity_field="projection_hash",
            )
            provenance = store.load(
                category="development/paired-panel-research/provenance",
                content_hash=trial.provenance_hash,
                model=AlphaFitProvenanceReceipt,
                identity_field="provenance_hash",
            )
            environment = store.load(
                category="development/paired-panel-research/environments",
                content_hash=trial.numerical_environment_hash,
                model=AlphaModelNumericalEnvironment,
                identity_field="environment_hash",
            )
            if (
                estimator.content_hash != provenance.estimator_content_hash
                or provenance.numerical_environment_hash != environment.environment_hash
                or provenance.recipe_hash != trial.recipe_hash
                or provenance.training_binding_hash != trial.training_binding_hash
            ):
                raise PanelMethodologyExecutionError(
                    "alpha_research.alpha_fold_model_lineage_mismatch"
                )


def _resolve_compatible_stage_root(
    *,
    store: PortfolioResearchArtifactStore,
    request: PanelResearchMethodologyRequest,
    source_resolution_hash: str,
    feature_preflight_hash: str,
    stage: Literal["ALPHA_ONLY", "SCORE_FILTER_ONLY"],
) -> PanelMethodologyExecutionRoot:
    """Resolve one symbolic upstream handle without a pointer or caller hash."""

    if stage == "ALPHA_ONLY" and request.execution_stage == "PORTFOLIO_ONLY":
        root = store.load(
            category=PANEL_METHODOLOGY_ROOT_CATEGORY,
            content_hash=SEALED_PANEL_ALPHA_ROOT_HASH,
            model=PanelMethodologyExecutionRoot,
            identity_field="root_hash",
        )
        publication = _load_admitted_alpha_economic_publication(store=store, root=root)
        if (
            root.program_hash != SEALED_PANEL_ALPHA_PROGRAM_HASH
            or root.source_resolution_hash != source_resolution_hash
            or root.identity_class != "DEVELOPMENT_ONLY"
            or root.upstream_alpha_root_hash is not None
            or root.ordered_score_filter_evidence_hashes
            or root.portfolio_root_hash is not None
            or publication is None
            or root.alpha_row_axis_receipt_hash is None
            or root.alpha_simple_score_value_artifact_hash is None
            or root.alpha_formation_readiness_receipt_hash is None
        ):
            raise PanelMethodologyExecutionError("alpha_research.fixed_candidate_root_not_admitted")
        return root

    category_root = store.root / PANEL_METHODOLOGY_ROOT_CATEGORY
    matches: list[PanelMethodologyExecutionRoot] = []
    if category_root.is_dir():
        for path in sorted(category_root.glob("*.json")):
            root = store.load(
                category=PANEL_METHODOLOGY_ROOT_CATEGORY,
                content_hash=path.stem,
                model=PanelMethodologyExecutionRoot,
                identity_field="root_hash",
            )
            is_alpha = (
                root.upstream_alpha_root_hash is None
                and not root.ordered_score_filter_evidence_hashes
                and root.portfolio_root_hash is None
                and root.alpha_row_axis_receipt_hash is not None
                and root.alpha_formation_readiness_receipt_hash is not None
            )
            is_filter = (
                root.upstream_alpha_root_hash is not None
                and bool(root.ordered_score_filter_evidence_hashes)
                and root.portfolio_root_hash is None
            )
            if (
                root.source_resolution_hash == source_resolution_hash
                and root.feature_preflight_hash == feature_preflight_hash
                and root.alpha_configuration_hash == panel_alpha_configuration_hash(request)
                and (
                    (stage == "ALPHA_ONLY" and is_alpha)
                    or (stage == "SCORE_FILTER_ONLY" and is_filter)
                )
            ):
                if (
                    stage == "ALPHA_ONLY"
                    and root.alpha_economic_evaluation_hash is not None
                    and root.alpha_economic_publication_attestation_hash is None
                ):
                    continue
                if root.alpha_economic_publication_attestation_hash is not None:
                    try:
                        _load_admitted_alpha_economic_publication(store=store, root=root)
                    except PanelMethodologyExecutionError as error:
                        if (
                            str(error)
                            == "alpha_research.alpha_economic_run_publication_attestation_missing"
                        ):
                            continue
                        raise
                matches.append(root)
    if len(matches) != 1:
        raise PanelMethodologyExecutionError("alpha_research.symbolic_upstream_handle_not_unique")
    return matches[0]


def _fold_row_axis(
    *, plan: PanelFeaturePlan, fold: PanelMethodologyOuterFold
) -> tuple[tuple[date, ...], tuple[str, ...], str]:
    position_by_session = {
        value: index for index, value in enumerate(plan.source.formation_sessions)
    }
    sessions: list[date] = []
    listings: list[str] = []
    for session in fold.validation_sessions:
        try:
            mask = plan.common_row_mask[position_by_session[session]]
        except KeyError as error:
            raise PanelMethodologyExecutionError(
                "alpha_research.upstream_score_session_not_in_source"
            ) from error
        for listing, admitted in zip(plan.source.ordered_listing_ids, mask, strict=True):
            if bool(admitted):
                sessions.append(session)
                listings.append(listing)
    row_sessions = tuple(sessions)
    row_listings = tuple(listings)
    return (
        row_sessions,
        row_listings,
        str(
            canonical_hash(
                [
                    (session.isoformat(), listing)
                    for session, listing in zip(row_sessions, row_listings, strict=True)
                ]
            )
        ),
    )


def _load_score_surface(
    *,
    store: PortfolioResearchArtifactStore,
    artifact: PanelScoreSurfaceArtifact,
    plan: PanelFeaturePlan | None = None,
    fold: PanelMethodologyOuterFold | None = None,
    row_axis: tuple[tuple[date, ...], tuple[str, ...]] | None = None,
) -> PanelScoreSurface:
    if row_axis is None:
        if plan is None or fold is None:
            raise PanelMethodologyExecutionError("alpha_research.upstream_score_axis_missing")
        sessions, listings, axis_hash = _fold_row_axis(plan=plan, fold=fold)
    else:
        if plan is not None or fold is None:
            raise PanelMethodologyExecutionError("alpha_research.upstream_score_axis_ambiguous")
        sessions, listings = row_axis
        axis_hash = str(
            canonical_hash(
                [
                    (session.isoformat(), listing)
                    for session, listing in zip(sessions, listings, strict=True)
                ]
            )
        )
    if (
        artifact.fold_index != fold.fold_index
        or artifact.row_count != len(sessions)
        or artifact.row_axis_hash != axis_hash
    ):
        raise PanelMethodologyExecutionError("alpha_research.upstream_score_axis_mismatch")

    def lane(category: str, content_hash: str) -> FloatArray:
        values: FloatArray = cast(
            FloatArray,
            np.frombuffer(
                store.load_packed_bytes(
                    category=f"development/paired-panel-research/{category}",
                    content_hash=content_hash,
                ),
                dtype=np.float64,
            ).copy(),
        )
        values.setflags(write=False)
        if values.shape != (artifact.row_count,):
            raise PanelMethodologyExecutionError("alpha_research.upstream_score_lane_shape_invalid")
        return cast(FloatArray, values)

    return PanelScoreSurface(
        surface_id=artifact.surface_id,
        surface_hash=artifact.source_surface_hash,
        fold_index=artifact.fold_index,
        span_sessions=artifact.span_sessions,
        row_sessions=sessions,
        row_listing_ids=listings,
        scores=lane("score-lanes", artifact.score_lane_hash),
        target_z=lane("target-z-lanes", artifact.target_z_lane_hash),
        raw_simple_returns=lane("raw-simple-return-lanes", artifact.raw_simple_return_lane_hash),
        model_recipe_id=artifact.model_recipe_id,
    )


def _verify_filtered_score_surface_relationship(
    *,
    store: PortfolioResearchArtifactStore,
    artifact: PanelScoreSurfaceArtifact,
    upstream_alpha_root: PanelMethodologyExecutionRoot,
) -> None:
    """Verify durable filtered-score lineage without recomputing filter arithmetic."""

    if (
        artifact.producer_spec_id is None
        or artifact.upstream_raw_score_artifact_hash is None
        or artifact.filter_spec_hash is None
        or artifact.filter_evidence_hash is None
        or artifact.fold_selection_hash is None
        or artifact.output_score_value_hash is None
        or artifact.upstream_raw_score_artifact_hash
        not in upstream_alpha_root.ordered_score_surface_artifact_hashes
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.score_filter_surface_lineage_incomplete"
        )
    raw = store.load(
        category="development/paired-panel-research/score-surfaces",
        content_hash=artifact.upstream_raw_score_artifact_hash,
        model=PanelScoreSurfaceArtifact,
        identity_field="artifact_hash",
    )
    evidence = store.load(
        category="development/paired-panel-research/score-filter-evidence",
        content_hash=artifact.filter_evidence_hash,
        model=PanelScoreFilterEvidence,
        identity_field="evidence_hash",
    )
    selection = store.load(
        category="development/paired-panel-research/score-filter-selections",
        content_hash=artifact.fold_selection_hash,
        model=PanelScoreFilterSelection,
        identity_field="selection_hash",
    )
    candidate = next(
        (
            value
            for value in selection.candidate_criteria
            if value.spec_id == artifact.producer_spec_id
        ),
        None,
    )
    score_payload = store.load_packed_bytes(
        category="development/paired-panel-research/score-lanes",
        content_hash=artifact.score_lane_hash,
    )
    score_hash = score_value_hash(np.frombuffer(score_payload, dtype=np.float64))
    expected_surface_hash = str(
        canonical_hash(
            {
                "source_surface_hash": raw.source_surface_hash,
                "filter_spec_hash": artifact.filter_spec_hash,
                "score_value_hash": score_hash,
            }
        )
    )
    if (
        raw.producer_spec_id is not None
        or evidence.filter_spec.spec_id != artifact.producer_spec_id
        or evidence.filter_spec.spec_hash != artifact.filter_spec_hash
        or artifact.upstream_raw_score_artifact_hash
        not in evidence.upstream_raw_score_artifact_hashes
        or selection.fold_index != artifact.fold_index
        or candidate is None
        or candidate.evidence_hash != evidence.evidence_hash
        or artifact.row_count != raw.row_count
        or artifact.row_axis_hash != raw.row_axis_hash
        or artifact.target_z_lane_hash != raw.target_z_lane_hash
        or artifact.raw_simple_return_lane_hash != raw.raw_simple_return_lane_hash
        or artifact.output_score_value_hash != score_hash
        or artifact.score_lane_hash != score_hash
        or artifact.source_surface_hash != expected_surface_hash
    ):
        raise PanelMethodologyExecutionError("alpha_research.score_filter_surface_lineage_invalid")


def _verify_filtered_score_surface_group(
    *,
    store: PortfolioResearchArtifactStore,
    artifacts: tuple[PanelScoreSurfaceArtifact, ...],
    upstream_alpha_root: PanelMethodologyExecutionRoot,
) -> None:
    """Verify complete per-spec OOF bytes against the durable filter evidence."""

    by_evidence: dict[str, list[PanelScoreSurfaceArtifact]] = {}
    for artifact in artifacts:
        _verify_filtered_score_surface_relationship(
            store=store,
            artifact=artifact,
            upstream_alpha_root=upstream_alpha_root,
        )
        assert artifact.filter_evidence_hash is not None
        by_evidence.setdefault(artifact.filter_evidence_hash, []).append(artifact)
    for evidence_hash, values in by_evidence.items():
        evidence = store.load(
            category="development/paired-panel-research/score-filter-evidence",
            content_hash=evidence_hash,
            model=PanelScoreFilterEvidence,
            identity_field="evidence_hash",
        )
        ordered = tuple(sorted(values, key=lambda value: value.fold_index))
        raw_hashes = tuple(value.upstream_raw_score_artifact_hash for value in ordered)
        score_arrays: tuple[FloatArray, ...] = tuple(
            cast(
                FloatArray,
                np.frombuffer(
                    store.load_packed_bytes(
                        category="development/paired-panel-research/score-lanes",
                        content_hash=value.score_lane_hash,
                    ),
                    dtype=np.float64,
                ),
            )
            for value in ordered
        )
        if (
            len({value.fold_index for value in ordered}) != len(ordered)
            or raw_hashes != evidence.upstream_raw_score_artifact_hashes
            or len(ordered) != len(evidence.model_spec_ids_by_fold)
            or score_value_hash(np.concatenate(score_arrays)) != evidence.score_value_hash
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.score_filter_surface_group_lineage_invalid"
            )


def _load_completed_alpha_folds(
    *,
    store: PortfolioResearchArtifactStore,
    plan: PanelFeaturePlan | None = None,
    fixed_row_axes: tuple[tuple[tuple[date, ...], tuple[str, ...]], ...] = (),
    outer_folds: tuple[PanelMethodologyOuterFold, ...],
    program_hash: str | None = None,
    ordered_fold_hashes: tuple[str, ...] | None = None,
    ordered_score_surface_artifact_hashes: tuple[str, ...] | None = None,
) -> tuple[_CompletedPanelAlphaFold, ...]:
    """Load only fully published Alpha folds, either for resume or downstream use."""

    if (program_hash is None) == (ordered_fold_hashes is None):
        raise PanelMethodologyExecutionError("alpha_research.alpha_fold_lookup_invalid")
    if (plan is None) == (not fixed_row_axes):
        raise PanelMethodologyExecutionError("alpha_research.upstream_score_axis_ambiguous")
    row_axis_by_fold = (
        {}
        if plan is not None
        else {fold.fold_index: axis for fold, axis in zip(outer_folds, fixed_row_axes, strict=True)}
    )

    artifact_by_source: dict[str, tuple[PanelScoreSurfaceArtifact, str]] = {}
    if ordered_score_surface_artifact_hashes is None:
        artifact_root = store.root / "development/paired-panel-research/score-surfaces"
        artifact_hashes = tuple(
            path.stem
            for path in (sorted(artifact_root.glob("*.json")) if artifact_root.is_dir() else ())
        )
    else:
        artifact_hashes = ordered_score_surface_artifact_hashes
    for artifact_hash in artifact_hashes:
        artifact = store.load(
            category="development/paired-panel-research/score-surfaces",
            content_hash=artifact_hash,
            model=PanelScoreSurfaceArtifact,
            identity_field="artifact_hash",
        )
        artifact_by_source[artifact.source_surface_hash] = (artifact, artifact_hash)
    fold_by_index = {value.fold_index: value for value in outer_folds}
    if ordered_fold_hashes is None:
        fold_hashes: list[str] = []
        fold_root = store.root / "development/paired-panel-research/folds"
        for path in sorted(fold_root.glob("*.json")) if fold_root.is_dir() else ():
            evidence = store.load(
                category="development/paired-panel-research/folds",
                content_hash=path.stem,
                model=PanelModelFoldEvidence,
                identity_field="fold_hash",
            )
            if evidence.program_hash == program_hash:
                fold_hashes.append(path.stem)
    else:
        fold_hashes = list(ordered_fold_hashes)
    completed: list[_CompletedPanelAlphaFold] = []
    seen_indices: set[int] = set()
    for fold_hash in fold_hashes:
        evidence = store.load(
            category="development/paired-panel-research/folds",
            content_hash=fold_hash,
            model=PanelModelFoldEvidence,
            identity_field="fold_hash",
        )
        if program_hash is not None and evidence.program_hash != program_hash:
            raise PanelMethodologyExecutionError("alpha_research.alpha_fold_program_mismatch")
        if evidence.fold_index in seen_indices:
            raise PanelMethodologyExecutionError("alpha_research.alpha_fold_checkpoint_conflict")
        seen_indices.add(evidence.fold_index)
        _verify_alpha_fold_checkpoint(store=store, fold=evidence)
        try:
            fold = fold_by_index[evidence.fold_index]
            raw_artifact, _raw_artifact_hash = artifact_by_source[
                cast(str, evidence.raw_dynamic_surface_hash)
            ]
            control_artifact, _control_artifact_hash = artifact_by_source[
                evidence.common_span_control_surface_hash
            ]
            fixed_artifacts = tuple(
                artifact_by_source[value][0] for value in evidence.fixed_candidate_surface_hashes
            )
        except KeyError as error:
            raise PanelMethodologyExecutionError(
                "alpha_research.upstream_alpha_child_missing"
            ) from error
        selection = store.load(
            category="development/paired-panel-research/selections",
            content_hash=evidence.dynamic_challenger_selection_hash,
            model=PanelModelScientificSelection,
            identity_field="selection_hash",
        )
        completed.append(
            _CompletedPanelAlphaFold(
                evidence=evidence,
                raw_surface=_load_score_surface(
                    store=store,
                    artifact=raw_artifact,
                    plan=plan,
                    fold=fold,
                    row_axis=(row_axis_by_fold.get(fold.fold_index) if plan is None else None),
                ),
                control_surface=_load_score_surface(
                    store=store,
                    artifact=control_artifact,
                    plan=plan,
                    fold=fold,
                    row_axis=(row_axis_by_fold.get(fold.fold_index) if plan is None else None),
                ),
                raw_artifact=raw_artifact,
                control_artifact=control_artifact,
                model_spec_id=selection.operational_fallback_spec_id,
                fixed_surfaces=tuple(
                    _load_score_surface(
                        store=store,
                        artifact=artifact,
                        plan=plan,
                        fold=fold,
                        row_axis=(row_axis_by_fold.get(fold.fold_index) if plan is None else None),
                    )
                    for artifact in fixed_artifacts
                ),
                fixed_artifacts=fixed_artifacts,
            )
        )
    return tuple(sorted(completed, key=lambda value: value.evidence.fold_index))


def _fixed_panel_score_folds(
    completed_folds: tuple[_CompletedPanelAlphaFold, ...],
) -> tuple[
    tuple[
        PanelScoreFoldIdentity,
        PanelScoreSurface,
        PanelScoreSurfaceArtifact,
        PanelScoreSurfaceArtifact,
    ],
    ...,
]:
    resolved = []
    for completed in completed_folds:
        pairs = tuple(zip(completed.fixed_surfaces, completed.fixed_artifacts, strict=True))
        fixed = tuple(
            pair for pair in pairs if pair[0].model_recipe_id == SEALED_PANEL_ALPHA_MODEL_RECIPE_ID
        )
        controls = tuple(
            pair for pair in pairs if pair[0].model_recipe_id == "RELATIVE_CONTROL_RIDGE"
        )
        if len(fixed) != 1 or len(controls) != 1:
            raise PanelMethodologyExecutionError("alpha_research.fixed_panel_score_fold_not_unique")
        surface, artifact = fixed[0]
        control_surface, control_artifact = controls[0]
        sessions = tuple(dict.fromkeys(surface.row_sessions))
        row_hash = str(
            canonical_hash(
                [
                    (session.isoformat(), listing)
                    for session, listing in zip(
                        surface.row_sessions, surface.row_listing_ids, strict=True
                    )
                ]
            )
        )
        if (
            sessions != tuple(sorted(set(sessions)))
            or artifact.source_surface_hash != surface.surface_hash
            or artifact.row_count != len(surface.row_sessions)
            or artifact.row_axis_hash != row_hash
            or control_surface.row_sessions != surface.row_sessions
            or control_surface.row_listing_ids != surface.row_listing_ids
            or control_artifact.row_axis_hash != row_hash
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_panel_score_fold_axis_invalid"
            )
        identity = PanelScoreFoldIdentity.create(
            fold_index=completed.evidence.fold_index,
            artifact_hash=artifact.artifact_hash,
            source_surface_hash=artifact.source_surface_hash,
            validation_session_count=len(sessions),
            validation_session_axis_hash=str(
                canonical_hash([value.isoformat() for value in sessions])
            ),
            row_count=artifact.row_count,
            row_axis_hash=artifact.row_axis_hash,
            score_value_hash=artifact.score_lane_hash,
        )
        resolved.append((identity, surface, artifact, control_artifact))
    return tuple(resolved)


def _publish_fixed_alpha_successor_authority(
    *,
    output_workspace: Path,
    program_hash: str,
    completed_folds: tuple[_CompletedPanelAlphaFold, ...],
    inputs: PanelMethodologyExecutionInputs,
) -> tuple[str, str, str, str, str, float]:
    source = inputs.alpha_panel_source_identity
    simple_authority = inputs.simple_score_authority
    plan = inputs.plan
    if source is None or simple_authority is None or plan is None:
        raise PanelMethodologyExecutionError(
            "alpha_research.panel_score_successor_authority_not_installed"
        )
    _store, _root, _economic, _candidate, _folds, _trials, historical = _load_fixed_candidate_graph(
        output_workspace=output_workspace,
        root_hash=SEALED_PANEL_ALPHA_ROOT_HASH,
        model_recipe_id=SEALED_PANEL_ALPHA_MODEL_RECIPE_ID,
    )
    fixed = _fixed_panel_score_folds(completed_folds)

    def materialize_simple_score(
        sessions: tuple[date, ...], listings: tuple[str, ...]
    ) -> tuple[SimpleSignedScoreBinding, FloatArray]:
        materialized = simple_authority.materialize(
            method_id=FIXED_STUDY_SIMPLE_SCORE_METHOD_ID,
            ordered_formation_sessions=sessions,
            ordered_listing_ids=listings,
        )
        return materialized.binding, materialized.standardized_values

    try:
        return AlphaDevelopmentArtifactStore(
            output_workspace / "portfolio-development"
        ).publish_panel_score_successor_authority(
            program_hash=program_hash,
            source=source,
            feature_column_count=plan.preflight.feature_count_by_method.get(
                SEALED_PANEL_ALPHA_VIEW_ID, 0
            ),
            folds=tuple(
                (
                    identity.fold_index,
                    surface.row_sessions,
                    surface.row_listing_ids,
                    artifact,
                    control,
                )
                for identity, surface, artifact, control in fixed
            ),
            historical_artifacts=historical,
            simple_score_materializer=materialize_simple_score,
        )
    except ValueError as error:
        raise PanelMethodologyExecutionError(str(error)) from error


def _fixed_panel_score_temporal_handoff(
    *,
    root: PanelMethodologyExecutionRoot,
    completed_folds: tuple[_CompletedPanelAlphaFold, ...],
    inputs: PanelMethodologyExecutionInputs,
) -> PanelScoreTemporalHandoff:
    """Re-derive the exact fixed model's score clock from durable owners."""

    source = inputs.alpha_panel_source_identity
    if source is None or root.source_resolution_hash != source.source_resolution_hash:
        raise PanelMethodologyExecutionError(
            "alpha_research.panel_score_source_authority_not_installed"
        )
    folds = tuple(value[0] for value in _fixed_panel_score_folds(completed_folds))
    feature_axis_hash = inputs.fixed_feature_axis_hash
    feature_count = inputs.fixed_feature_column_count
    if inputs.plan is not None:
        feature_axis_hash = inputs.plan.preflight.ordered_feature_axis_hash_by_method.get(
            "SPARSE_SESSION_AMPLITUDE"
        )
        feature_count = inputs.plan.preflight.feature_count_by_method.get(
            "SPARSE_SESSION_AMPLITUDE"
        )
    if feature_axis_hash is None or feature_count is None:
        raise PanelMethodologyExecutionError("alpha_research.fixed_panel_score_feature_axis_absent")
    return _panel_score_temporal_handoff(
        root=root,
        source=source,
        folds=folds,
        feature_axis_hash=feature_axis_hash,
        feature_count=feature_count,
        readiness=inputs.panel_score_readiness,
    )


def _panel_score_temporal_handoff(
    *,
    root: PanelMethodologyExecutionRoot,
    source: AlphaPanelSourceIdentity,
    folds: tuple[PanelScoreFoldIdentity, ...],
    feature_axis_hash: str,
    feature_count: int,
    readiness: PanelScoreFormationReadinessReceipt | None,
) -> PanelScoreTemporalHandoff:
    producer = resolve_panel_score_producer_identity(
        program_hash=root.program_hash,
        root_hash=root.root_hash,
        model_recipe_id=SEALED_PANEL_ALPHA_MODEL_RECIPE_ID,
        source=source,
        feature_preflight_hash=root.feature_preflight_hash,
        feature_column_count=feature_count,
        feature_axis_hash=feature_axis_hash,
        ordered_formation_sessions=root.common_formation_sessions,
        folds=folds,
        fit_call_count=root.fit_call_count,
        predict_call_count=root.predict_call_count,
        metric_call_count=root.metric_call_count,
        solver_call_count=root.solver_call_count,
        resource_measurement_scope=root.resource_measurement_scope,
        worker_resource_receipt_hashes=tuple(
            value.receipt_hash for value in root.worker_resource_receipts
        ),
    )
    if (
        readiness is None
        or readiness.source_identity_hash != source.identity_hash
        or readiness.score_surface_hash != producer.score_surface_hash
        or readiness.formation_count != len(root.common_formation_sessions)
        or readiness.formation_axis_hash != producer.decision_session_axis_hash
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.panel_score_formation_readiness_unverified"
        )
    observation = ScoreObservationAuthority.create(
        observation_session_offset_sessions=0,
        availability_delay_sessions=0,
        availability_policy_id=source.governing_availability_policy_id,
        availability_policy_hash=source.governing_availability_policy_hash,
        observation_clock_hash=source.source_authority_binding_hash,
        formula_observation_semantics=(
            "Panel Formula observations and installed source facts through close(T)"
        ),
        source_authority_id="alpha_research.panel_score_producer",
        methodology_identity=producer.producer_hash,
        strategy_scope="CORRECTED_STRATEGY_SIGNAL",
    )
    causal = score_input_authority(
        observation,
        surface_hash=producer.score_surface_hash,
        input_id="fixed_panel_alpha_score",
        derived_ready=readiness.derived_ready,
    )
    if (
        causal.observed_through != readiness.observed_through
        or causal.source_available != readiness.source_available
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.panel_score_formation_readiness_unverified"
        )
    return PanelScoreTemporalHandoff.create(
        producer=producer,
        observation=observation,
        causal_input=causal,
    )


def resolve_fixed_panel_score_temporal_handoff(
    *,
    output_workspace: Path,
    root: PanelMethodologyExecutionRoot,
    row_axis_receipt: FixedAlphaRowAxisReceipt,
    source: AlphaPanelSourceIdentity,
    feature_axis_hash: str,
    feature_count: int,
    readiness: PanelScoreFormationReadinessReceipt,
) -> PanelScoreTemporalHandoff:
    """Re-derive the fixed producer from exact manifests, never score values."""
    store = PortfolioResearchArtifactStore(output_workspace / "portfolio-development")
    folds: list[PanelScoreFoldIdentity] = []
    for row_fold in row_axis_receipt.folds:
        artifact = store.load(
            category="development/paired-panel-research/score-surfaces",
            content_hash=row_fold.fixed_score_artifact_hash,
            model=PanelScoreSurfaceArtifact,
            identity_field="artifact_hash",
        )
        if (
            artifact.fold_index != row_fold.fold_index
            or artifact.model_recipe_id != SEALED_PANEL_ALPHA_MODEL_RECIPE_ID
            or artifact.producer_spec_id is not None
            or artifact.row_count != row_fold.row_count
            or artifact.row_axis_hash != row_fold.row_axis_hash
            or artifact.score_lane_hash != row_fold.fixed_score_value_hash
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_panel_score_manifest_not_this_axis"
            )
        folds.append(
            PanelScoreFoldIdentity.create(
                fold_index=row_fold.fold_index,
                artifact_hash=artifact.artifact_hash,
                source_surface_hash=artifact.source_surface_hash,
                validation_session_count=len(row_fold.validation_sessions),
                validation_session_axis_hash=str(
                    canonical_hash([value.isoformat() for value in row_fold.validation_sessions])
                ),
                row_count=row_fold.row_count,
                row_axis_hash=row_fold.row_axis_hash,
                score_value_hash=row_fold.fixed_score_value_hash,
            )
        )
    handoff = _panel_score_temporal_handoff(
        root=root,
        source=source,
        folds=tuple(folds),
        feature_axis_hash=feature_axis_hash,
        feature_count=feature_count,
        readiness=readiness,
    )
    if (
        root.alpha_row_axis_receipt_hash != row_axis_receipt.receipt_hash
        or root.alpha_score_surface_hash != handoff.producer.score_surface_hash
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.fixed_panel_score_manifest_not_this_root"
        )
    return handoff


def _admit_fixed_panel_portfolio_inputs(
    *,
    handoff: PanelScoreTemporalHandoff,
    inputs: PanelMethodologyExecutionInputs,
    axis: PanelPortfolioAxis,
    market: ImmutablePortfolioMarketInputs,
) -> AdmittedInformationSet:
    """Meet Alpha and Risk facts with the execution-owned strategy clock once."""

    execution = inputs.execution_clock
    r0 = inputs.r0_covariance
    r1 = inputs.r1_covariance
    reference_marks = inputs.reference_marks
    if execution is None or r0 is None or reference_marks is None:
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_temporal_owner_not_installed"
        )
    return admit_fixed_panel_portfolio_authorities(
        handoff=handoff,
        execution=execution,
        r0_covariance=r0,
        r1_covariance=r1,
        mark_surface=reference_marks.surface,
        marked_sessions=reference_marks.marked_sessions,
        tradability_bundle_hash=market.tradability_bundle_hash,
        universe_epoch_hash=market.universe_epoch_hash,
        axis=axis,
    )


def admit_fixed_panel_portfolio_authorities(
    *,
    handoff: PanelScoreTemporalHandoff,
    execution: ExecutionClockContext,
    r0_covariance: ResolvedStageSixCovariance,
    r1_covariance: ResolvedStageSixCovariance | None,
    mark_surface: SessionMarkTemporalAuthority,
    marked_sessions: tuple[date, ...],
    tradability_bundle_hash: str,
    universe_epoch_hash: str,
    axis: PanelPortfolioAxis,
) -> AdmittedInformationSet:
    """Admit exact clocks and manifests before any Portfolio value lane opens."""
    events = execution.events
    if (
        events.method_seal_disposition != "METHOD_BOUND"
        or events.execution_recipe_id is None
        or events.execution_recipe_hash is None
    ):
        raise PanelMethodologyExecutionError("alpha_research.portfolio_execution_method_not_bound")
    catalog = build_installed_execution_outcome_method_catalog()
    recipe = catalog.resolve(events.execution_recipe_id)
    if recipe.recipe_hash != events.execution_recipe_hash:
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_execution_recipe_not_this_graph"
        )
    schedule = resolve_strategy_decision_schedule(
        schedule_handle=resolve_installed_schedule_handle(recipe.recipe_id),
        recipe=recipe,
        method_binding_hash=events.method_binding_hash,
    )
    if (
        tuple(
            value
            for value in handoff.producer.ordered_formation_sessions
            if value in set(axis.formation_sessions)
        )
        != axis.formation_sessions
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_score_decision_axis_mismatch"
        )
    if tuple(value for value in marked_sessions if value in set(axis.formation_sessions)) != (
        axis.formation_sessions
    ):
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_state_transition_not_this_axis"
        )
    admitted = admit_decision_input_set(
        authorities=(
            handoff.causal_input,
            risk_covariance_authority(
                surface_hash=r0_covariance.surface_hash,
                owner_identity_hash=r0_covariance.surface_hash,
                input_id="risk_covariance_r0",
                readiness_handle=RISK_READINESS_BUDGET,
            ),
            *(
                (
                    risk_covariance_authority(
                        surface_hash=r1_covariance.surface_hash,
                        owner_identity_hash=r1_covariance.surface_hash,
                        input_id="risk_covariance_r1",
                        readiness_handle=RISK_READINESS_BUDGET,
                    ),
                )
                if r1_covariance is not None
                else ()
            ),
            tradability_input_authority(
                bundle_hash=tradability_bundle_hash,
                universe_epoch_hash=universe_epoch_hash,
            ),
            session_mark_input_authority(cast(Any, mark_surface)),
        ),
        schedule=schedule,
        formation_sessions=axis.formation_sessions,
        ordered_axis=tuple(sorted(execution.session_clocks)),
        session_clocks=execution.session_clocks,
    )
    if admitted.disposition != "ADMITTED":
        refused = next(value for value in admitted.results if value.disposition == "REFUSED")
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_temporal_admission_refused:"
            f"{refused.input_id}:{refused.refused_relation}"
        )
    return admitted


def _scale_receipt_hashes_for_folds(
    *,
    store: PortfolioResearchArtifactStore,
    folds: tuple[PanelModelFoldEvidence, ...],
) -> tuple[str, ...]:
    hashes: list[str] = []
    for fold in folds:
        for category, trial_hashes in (
            ("development/paired-panel-research/inner-trials", fold.inner_trial_evidence_hashes),
            ("development/paired-panel-research/outer-trials", fold.outer_trial_evidence_hashes),
        ):
            for trial_hash in trial_hashes:
                trial = store.load(
                    category=category,
                    content_hash=trial_hash,
                    model=PanelModelTrialEvidence,
                    identity_field="evidence_hash",
                )
                if trial.program_hash != fold.program_hash or trial.fold_index != fold.fold_index:
                    raise PanelMethodologyExecutionError(
                        "alpha_research.alpha_fold_trial_identity_mismatch"
                    )
                hashes.extend(trial.scale_receipt_hashes)
    return tuple(dict.fromkeys(hashes))


def _evaluate_fixed_alpha_economics(
    completed_folds: tuple[_CompletedPanelAlphaFold, ...],
) -> PanelAlphaEconomicEvaluation:
    """Pool exact named OOF candidates and delegate fixed-book metrics to Alpha."""

    grouped: dict[str, list[tuple[PanelScoreSurface, PanelScoreSurfaceArtifact]]] = {}
    for fold in completed_folds:
        if len(fold.fixed_surfaces) != len(fold.fixed_artifacts):
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_candidate_artifact_set_invalid"
            )
        for surface, artifact in zip(fold.fixed_surfaces, fold.fixed_artifacts, strict=True):
            if (
                surface.model_recipe_id is None
                or artifact.model_recipe_id != surface.model_recipe_id
                or artifact.source_surface_hash != surface.surface_hash
            ):
                raise PanelMethodologyExecutionError(
                    "alpha_research.fixed_candidate_recipe_lineage_invalid"
                )
            grouped.setdefault(surface.model_recipe_id, []).append((surface, artifact))
    inputs: list[AlphaEconomicCandidateInput] = []
    for recipe_id, pairs in sorted(grouped.items()):
        ordered = sorted(pairs, key=lambda item: item[0].fold_index)
        scores = np.ascontiguousarray(
            np.concatenate([value.scores for value, _artifact in ordered]), dtype=np.float64
        )
        raw = np.ascontiguousarray(
            np.concatenate([value.raw_simple_returns for value, _artifact in ordered]),
            dtype=np.float64,
        )
        scores.setflags(write=False)
        raw.setflags(write=False)
        inputs.append(
            AlphaEconomicCandidateInput(
                model_recipe_id=recipe_id,
                ordered_score_surface_hashes=tuple(
                    artifact.artifact_hash for _value, artifact in ordered
                ),
                ordered_row_axis_hashes=tuple(
                    artifact.row_axis_hash for _value, artifact in ordered
                ),
                ordered_raw_simple_return_lane_hashes=tuple(
                    artifact.raw_simple_return_lane_hash for _value, artifact in ordered
                ),
                sessions=tuple(
                    session for value, _artifact in ordered for session in value.row_sessions
                ),
                listings=tuple(
                    listing for value, _artifact in ordered for listing in value.row_listing_ids
                ),
                scores=scores,
                raw_simple_returns=raw,
            )
        )
    return evaluate_fixed_recipe_alpha_economics(
        candidates=tuple(inputs), holding_horizon_sessions=1
    )


def _load_upstream_filter_surfaces(
    *,
    store: PortfolioResearchArtifactStore,
    root: PanelMethodologyExecutionRoot,
    upstream_alpha_root: PanelMethodologyExecutionRoot,
    plan: PanelFeaturePlan,
    outer_folds: tuple[PanelMethodologyOuterFold, ...],
) -> tuple[
    Mapping[str, tuple[PanelScoreSurface, ...]],
    tuple[PanelScoreSurface, ...],
    tuple[PanelScoreSurfaceArtifact, ...],
]:
    fold_by_index = {value.fold_index: value for value in outer_folds}
    selections: dict[int, PanelScoreFilterSelection] = {}
    for selection_hash in root.ordered_score_filter_selection_hashes:
        selection = store.load(
            category="development/paired-panel-research/score-filter-selections",
            content_hash=selection_hash,
            model=PanelScoreFilterSelection,
            identity_field="selection_hash",
        )
        selections[selection.fold_index] = selection
    artifacts: list[PanelScoreSurfaceArtifact] = []
    surfaces: dict[str, list[PanelScoreSurface]] = {}
    selected: dict[int, PanelScoreSurface] = {}
    for artifact_hash in root.ordered_score_surface_artifact_hashes:
        artifact = store.load(
            category="development/paired-panel-research/score-surfaces",
            content_hash=artifact_hash,
            model=PanelScoreSurfaceArtifact,
            identity_field="artifact_hash",
        )
        if artifact.producer_spec_id is None:
            continue
        try:
            fold = fold_by_index[artifact.fold_index]
            selection = selections[artifact.fold_index]
        except KeyError as error:
            raise PanelMethodologyExecutionError(
                "alpha_research.upstream_filter_child_missing"
            ) from error
        surface = _load_score_surface(store=store, artifact=artifact, plan=plan, fold=fold)
        artifacts.append(artifact)
        surfaces.setdefault(artifact.producer_spec_id, []).append(surface)
        if artifact.producer_spec_id == selection.operational_fallback_spec_id:
            if artifact.fold_index in selected:
                raise PanelMethodologyExecutionError(
                    "alpha_research.upstream_filter_selection_duplicated"
                )
            selected[artifact.fold_index] = surface
    _verify_filtered_score_surface_group(
        store=store,
        artifacts=tuple(artifacts),
        upstream_alpha_root=upstream_alpha_root,
    )
    expected_folds = tuple(value.fold_index for value in outer_folds)
    if tuple(sorted(selections)) != expected_folds or tuple(sorted(selected)) != expected_folds:
        raise PanelMethodologyExecutionError("alpha_research.upstream_filter_axis_incomplete")
    ordered_by_spec = MappingProxyType(
        {
            spec_id: tuple(sorted(values, key=lambda value: value.fold_index))
            for spec_id, values in sorted(surfaces.items())
        }
    )
    return (
        ordered_by_spec,
        tuple(selected[index] for index in expected_folds),
        tuple(artifacts),
    )


def _evaluate_oof_score_filters(
    *,
    raw_surfaces: tuple[PanelScoreSurface, ...],
    model_spec_ids_by_fold: tuple[str, ...],
    upstream_raw_score_artifact_hashes: tuple[str, ...],
    request: PanelResearchMethodologyRequest,
    plan: PanelFeaturePlan,
) -> tuple[
    PanelScoreFilterEvaluationResult,
    Mapping[str, tuple[PanelScoreSurface, ...]],
    tuple[PanelScoreSurface, ...],
]:
    sessions = tuple(session for surface in raw_surfaces for session in surface.row_sessions)
    listings = tuple(listing for surface in raw_surfaces for listing in surface.row_listing_ids)
    scores = cast(
        FloatArray,
        _readonly(
            np.concatenate(tuple(surface.scores for surface in raw_surfaces)),
            dtype=np.dtype(np.float64),
        ),
    )
    raw_returns = cast(
        FloatArray,
        _readonly(
            np.concatenate(tuple(surface.raw_simple_returns for surface in raw_surfaces)),
            dtype=np.dtype(np.float64),
        ),
    )
    row_folds = cast(
        IntArray,
        _readonly(
            np.concatenate(
                tuple(
                    np.full(len(surface.row_sessions), surface.fold_index, dtype=np.int64)
                    for surface in raw_surfaces
                )
            ),
            dtype=np.dtype(np.int64),
        ),
    )
    holding_end_by_session = dict(
        zip(
            plan.source.formation_sessions,
            plan.source.holding_end_sessions,
            strict=True,
        )
    )
    result = evaluate_panel_score_filters(
        sessions=sessions,
        listings=listings,
        fold_indices=row_folds,
        holding_end_sessions=tuple(holding_end_by_session[value] for value in sessions),
        raw_scores=scores,
        raw_simple_returns=raw_returns,
        upstream_raw_score_artifact_hashes=upstream_raw_score_artifact_hashes,
        model_spec_ids_by_fold=model_spec_ids_by_fold,
        filter_specs=request.score_filter_candidates,
        persistence_lags=request.persistence_lags,
        holding_horizon_sessions=1,
    )
    selection_by_fold = {value.fold_index: value for value in result.selections}
    candidates: dict[str, list[PanelScoreSurface]] = {
        value.spec_id: [] for value in request.score_filter_candidates
    }
    selected: list[PanelScoreSurface] = []
    offset = 0
    for raw_surface in raw_surfaces:
        count = len(raw_surface.row_sessions)
        selection = selection_by_fold[raw_surface.fold_index]
        for spec in request.score_filter_candidates:
            values = result.filtered_scores_by_spec_id[spec.spec_id][offset : offset + count]
            values = cast(
                FloatArray,
                _readonly(values, dtype=np.dtype(np.float64)),
            )
            surface = PanelScoreSurface(
                surface_id=f"{spec.spec_id}::{raw_surface.surface_id}",
                surface_hash=str(
                    canonical_hash(
                        {
                            "source_surface_hash": raw_surface.surface_hash,
                            "filter_spec_hash": spec.spec_hash,
                            "score_value_hash": score_value_hash(values),
                        }
                    )
                ),
                fold_index=raw_surface.fold_index,
                span_sessions=spec.span_sessions,
                row_sessions=raw_surface.row_sessions,
                row_listing_ids=raw_surface.row_listing_ids,
                scores=values,
                target_z=raw_surface.target_z,
                raw_simple_returns=raw_surface.raw_simple_returns,
                model_recipe_id=raw_surface.model_recipe_id,
            )
            candidates[spec.spec_id].append(surface)
            if spec.spec_id == selection.operational_fallback_spec_id:
                selected.append(surface)
        offset += count
    if offset != len(sessions) or len(selected) != len(raw_surfaces):
        raise PanelMethodologyExecutionError("alpha_research.score_filter_surface_invalid")
    return (
        result,
        MappingProxyType({spec_id: tuple(surfaces) for spec_id, surfaces in candidates.items()}),
        tuple(selected),
    )


def preflight_panel_portfolio_axis(
    *,
    plan: PanelFeaturePlan | None = None,
    fixed_row_axes: tuple[tuple[tuple[date, ...], tuple[str, ...]], ...] = (),
    ordered_score_listing_ids: tuple[str, ...] = (),
    outer_folds: tuple[PanelMethodologyOuterFold, ...],
    portfolio_market: PortfolioAxisAuthority,
    r0_covariance: ResolvedStageSixCovariance,
    r1_covariance: ResolvedStageSixCovariance | None,
    maximum_formation_count: int | None = None,
) -> PanelPortfolioAxis:
    """Admit common stable listing support and explicit decision/passive economic geometry.

    Args:
        plan: Optional dynamic feature plan; mutually exclusive with fixed_row_axes.
        fixed_row_axes: Durable fixed Alpha row axes for every outer fold.
        ordered_score_listing_ids: Required fixed score listing order.
        outer_folds: Ordered declared outer validation folds.
        portfolio_market: Owner of market session/listing support.
        r0_covariance: Required admitted baseline Risk support.
        r1_covariance: Optional paired Risk support.
        maximum_formation_count: Optional bounded prefix of at least 100 formations.

    Returns:
        Common support with at least 100 listings, read-only source position arrays and exact axis
        hashes; only single passive gaps are admitted.

    Raises:
        PanelMethodologyExecutionError: Score-axis source is ambiguous, fixed row receipts/common
            support are invalid, count is insufficient or economic gap geometry is unsupported.
    """
    fold_sessions = tuple(session for fold in outer_folds for session in fold.validation_sessions)
    available = set(portfolio_market.formation_sessions) & set(r0_covariance.formation_sessions)
    if r1_covariance is not None:
        available &= set(r1_covariance.formation_sessions)
    full_sessions = tuple(value for value in fold_sessions if value in available)
    if full_sessions != tuple(sorted(set(full_sessions))) or len(full_sessions) < 100:
        raise PanelMethodologyExecutionError("alpha_research.portfolio_common_session_axis_invalid")
    sessions = full_sessions
    if maximum_formation_count is not None:
        if not 100 <= maximum_formation_count <= len(sessions):
            raise PanelMethodologyExecutionError(
                "alpha_research.portfolio_study_axis_limit_invalid"
            )
        sessions = sessions[:maximum_formation_count]
    if (plan is None) == (not fixed_row_axes):
        raise PanelMethodologyExecutionError("alpha_research.portfolio_score_axis_source_ambiguous")
    if plan is not None:
        source_positions = {
            value: index for index, value in enumerate(plan.source.formation_sessions)
        }
        rows: IntArray = np.asarray([source_positions[value] for value in sessions], dtype=np.int64)
        stable = np.all(plan.common_row_mask[rows], axis=0)
        source_listing_ids = plan.source.ordered_listing_ids
    else:
        if len(fixed_row_axes) != len(outer_folds) or not ordered_score_listing_ids:
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_alpha_row_axis_receipt_invalid"
            )
        # The durable rows cover the complete fixed Alpha fold graph.  Validate
        # that immutable graph before projecting a bounded Portfolio study
        # prefix; otherwise valid later-fold rows look like foreign rows merely
        # because the requested Portfolio axis is shorter.
        full_session_set = set(full_sessions)
        score_listing_positions = {
            value: index for index, value in enumerate(ordered_score_listing_ids)
        }
        present_by_session: dict[date, set[str]] = {value: set() for value in sessions}
        for (row_sessions, row_listings), fold in zip(fixed_row_axes, outer_folds, strict=True):
            if len(row_sessions) != len(row_listings) or not set(fold.validation_sessions).issubset(
                full_session_set
            ):
                raise PanelMethodologyExecutionError(
                    "alpha_research.fixed_alpha_row_axis_receipt_invalid"
                )
            row_position = 0
            for session in fold.validation_sessions:
                lane_start = row_position
                while row_position < len(row_sessions) and row_sessions[row_position] == session:
                    row_position += 1
                session_listings = row_listings[lane_start:row_position]
                try:
                    positions = tuple(score_listing_positions[value] for value in session_listings)
                except KeyError as error:
                    raise PanelMethodologyExecutionError(
                        "alpha_research.fixed_alpha_row_axis_receipt_invalid"
                    ) from error
                if (
                    not session_listings
                    or positions != tuple(sorted(positions))
                    or len(positions) != len(set(positions))
                ):
                    raise PanelMethodologyExecutionError(
                        "alpha_research.fixed_alpha_row_axis_receipt_invalid"
                    )
                if session in present_by_session:
                    present_by_session[session].update(session_listings)
            if row_position != len(row_sessions) or any(
                session not in full_session_set for session in row_sessions
            ):
                raise PanelMethodologyExecutionError(
                    "alpha_research.fixed_alpha_row_axis_receipt_invalid"
                )
        stable_listings = set(ordered_score_listing_ids)
        for session in sessions:
            stable_listings &= present_by_session[session]
        source_listing_ids = ordered_score_listing_ids
        stable = np.asarray(
            [value in stable_listings for value in source_listing_ids], dtype=np.bool_
        )
    market_listing_position = {
        value: index for index, value in enumerate(portfolio_market.ordered_listing_ids)
    }
    r0_position = {value: index for index, value in enumerate(r0_covariance.ordered_listing_ids)}
    r1_position = (
        {value: index for index, value in enumerate(r1_covariance.ordered_listing_ids)}
        if r1_covariance is not None
        else None
    )
    listings = tuple(
        listing
        for index, listing in enumerate(source_listing_ids)
        if stable[index]
        and listing in market_listing_position
        and listing in r0_position
        and (r1_position is None or listing in r1_position)
    )
    if len(listings) < 100:
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_common_listing_axis_insufficient"
        )
    session_axis_hash = str(canonical_hash([value.isoformat() for value in sessions]))
    economic_sessions = tuple(
        value
        for value in portfolio_market.formation_sessions
        if sessions[0] <= value <= sessions[-1] and value in available
    )
    economic_position = {value: index for index, value in enumerate(economic_sessions)}
    ranges: list[tuple[int, int]] = []
    passive_sessions: list[date] = []
    range_start = 0
    for position in range(1, len(sessions)):
        left = economic_position[sessions[position - 1]]
        right = economic_position[sessions[position]]
        if right == left + 1:
            continue
        if right != left + 2:
            raise PanelMethodologyExecutionError(
                "alpha_research.portfolio_state_axis_gap_unsupported"
            )
        passive = economic_sessions[left + 1]
        if passive in set(sessions):
            raise PanelMethodologyExecutionError(
                "alpha_research.portfolio_passive_session_is_decision"
            )
        ranges.append((range_start, position))
        passive_sessions.append(passive)
        range_start = position
    ranges.append((range_start, len(sessions)))
    if len(economic_sessions) != len(sessions) + len(passive_sessions):
        raise PanelMethodologyExecutionError("alpha_research.portfolio_economic_axis_invalid")
    economic_session_axis_hash = str(
        canonical_hash([value.isoformat() for value in economic_sessions])
    )
    listing_axis_hash = str(canonical_hash(list(listings)))
    return PanelPortfolioAxis(
        formation_sessions=sessions,
        economic_formation_sessions=economic_sessions,
        decision_ranges=tuple(ranges),
        passive_sessions=tuple(passive_sessions),
        ordered_listing_ids=listings,
        market_listing_positions=cast(
            IntArray,
            _readonly(
                np.asarray([market_listing_position[value] for value in listings], dtype=np.int64),
                dtype=np.dtype(np.int64),
            ),
        ),
        r0_listing_positions=cast(
            IntArray,
            _readonly(
                np.asarray([r0_position[value] for value in listings], dtype=np.int64),
                dtype=np.dtype(np.int64),
            ),
        ),
        r1_listing_positions=(
            cast(
                IntArray,
                _readonly(
                    np.asarray([r1_position[value] for value in listings], dtype=np.int64),
                    dtype=np.dtype(np.int64),
                ),
            )
            if r1_position is not None
            else None
        ),
        session_axis_hash=session_axis_hash,
        economic_session_axis_hash=economic_session_axis_hash,
        listing_axis_hash=listing_axis_hash,
        axis_hash=str(
            canonical_hash(
                {
                    "sessions": session_axis_hash,
                    "economic_sessions": economic_session_axis_hash,
                    "decision_ranges": ranges,
                    "passive_sessions": [value.isoformat() for value in passive_sessions],
                    "listings": listing_axis_hash,
                    "score_filter_row_axis_policy": "INPUT_ROW_AXIS_INVARIANT",
                }
            )
        ),
    )


def _score_matrix(
    *,
    surfaces: tuple[PanelScoreSurface, ...],
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
) -> FloatArray:
    session_position = {value: index for index, value in enumerate(sessions)}
    listing_position = {value: index for index, value in enumerate(listings)}
    values: FloatArray = np.full((len(sessions), len(listings)), np.nan, dtype=np.float64)
    for surface in surfaces:
        for session, listing, score in zip(
            surface.row_sessions, surface.row_listing_ids, surface.scores, strict=True
        ):
            row = session_position.get(session)
            column = listing_position.get(listing)
            if row is None or column is None:
                continue
            if np.isfinite(values[row, column]):
                raise PanelMethodologyExecutionError(
                    "alpha_research.portfolio_score_row_duplicated"
                )
            values[row, column] = score
    if not np.isfinite(values).all():
        raise PanelMethodologyExecutionError(
            "alpha_research.portfolio_score_common_axis_incomplete"
        )
    return cast(
        FloatArray,
        _readonly(values, dtype=np.dtype(np.float64)),
    )


def _subset_market(
    *,
    market: ImmutablePortfolioMarketInputs,
    benchmark: ImmutablePortfolioBenchmark,
    fold_indices: IntArray,
    sessions: tuple[date, ...],
    economic_sessions: tuple[date, ...],
    passive_sessions: tuple[date, ...],
    listings: tuple[str, ...],
    listing_positions: IntArray,
) -> tuple[ImmutablePortfolioMarketInputs, ImmutablePortfolioBenchmark, IntArray]:
    source_sessions = {value: index for index, value in enumerate(market.formation_sessions)}
    rows: IntArray = np.asarray([source_sessions[value] for value in sessions], dtype=np.int64)
    if len(fold_indices) == len(sessions):
        selected_fold_indices = fold_indices
    elif len(fold_indices) == len(market.formation_sessions):
        selected_fold_indices = fold_indices[rows]
    else:
        raise PanelMethodologyExecutionError("alpha_research.portfolio_fold_index_axis_invalid")
    decision = np.asarray(market.decision_eligible[np.ix_(rows, listing_positions)], dtype=np.bool_)
    # A Sector history's per-session exposure keeps the subset's sessions.
    sector_matrix = np.asarray(
        market.sector_exposure_matrix[rows][:, :, listing_positions]
        if market.sector_exposure_matrix.ndim == 3
        else market.sector_exposure_matrix[:, listing_positions],
        dtype=np.float64,
    )
    anchor: FloatArray = np.zeros((len(sessions), len(market.sector_ids)), dtype=np.float64)
    for row in range(len(sessions)):
        eligible = decision[row]
        if not bool(np.any(eligible)):
            raise PanelMethodologyExecutionError("alpha_research.anchor_axis_empty")
        session_matrix = sector_matrix[row] if sector_matrix.ndim == 3 else sector_matrix
        anchor[row] = session_matrix[:, eligible] @ np.full(
            int(np.sum(eligible)), 1.0 / int(np.sum(eligible)), dtype=np.float64
        )
    arrays = (
        decision,
        market.execution_available[np.ix_(rows, listing_positions)],
        market.realized_simple_returns[np.ix_(rows, listing_positions)],
        market.causal_adv20[np.ix_(rows, listing_positions)],
        sector_matrix,
        anchor,
    )
    immutable = tuple(
        _readonly(value, dtype=np.dtype(np.bool_ if index < 2 else np.float64))
        for index, value in enumerate(arrays)
    )
    subset = ImmutablePortfolioMarketInputs(
        formation_sessions=sessions,
        economic_formation_sessions=economic_sessions,
        ordered_listing_ids=listings,
        sector_ids=market.sector_ids,
        sector_exposure_matrix=cast(FloatArray, immutable[4]),
        equal_weight_sector_exposure=cast(FloatArray, immutable[5]),
        decision_eligible=cast(BoolArray, immutable[0]),
        execution_available=cast(BoolArray, immutable[1]),
        realized_simple_returns=cast(FloatArray, immutable[2]),
        passive_returns_by_session={
            session: cast(
                FloatArray,
                _readonly(
                    market.realized_simple_returns[source_sessions[session], listing_positions],
                    dtype=np.dtype(np.float64),
                ),
            )
            for session in passive_sessions
        },
        causal_adv20=cast(FloatArray, immutable[3]),
        tradability_bundle_hash=market.tradability_bundle_hash,
        universe_epoch_hash=market.universe_epoch_hash,
        sector_revision=market.sector_revision,
        source_surface_hash=market.source_surface_hash,
    )
    benchmark_position = {value: index for index, value in enumerate(benchmark.formation_sessions)}
    benchmark_rows = [benchmark_position[value] for value in sessions]
    benchmark_economic_rows = [benchmark_position[value] for value in economic_sessions]
    selected_benchmark = ImmutablePortfolioBenchmark(
        formation_sessions=sessions,
        economic_formation_sessions=economic_sessions,
        simple_returns=tuple(benchmark.simple_returns[value] for value in benchmark_rows),
        log_returns=tuple(benchmark.log_returns[value] for value in benchmark_rows),
        economic_simple_returns=tuple(
            benchmark.simple_returns[value] for value in benchmark_economic_rows
        ),
        economic_log_returns=tuple(
            benchmark.log_returns[value] for value in benchmark_economic_rows
        ),
        surface_hash=benchmark.surface_hash,
    )
    return (
        subset,
        selected_benchmark,
        cast(
            IntArray,
            _readonly(selected_fold_indices, dtype=np.dtype(np.int64)),
        ),
    )


def _economic_fold_indices(
    *,
    decision_sessions: tuple[date, ...],
    decision_fold_indices: IntArray,
    economic_sessions: tuple[date, ...],
) -> IntArray:
    """Assign each passive carry to the fold whose preceding decision owns it."""

    if (
        len(decision_sessions) != len(decision_fold_indices)
        or decision_sessions != tuple(sorted(set(decision_sessions)))
        or economic_sessions != tuple(sorted(set(economic_sessions)))
        or not set(decision_sessions).issubset(economic_sessions)
    ):
        raise PanelMethodologyExecutionError("alpha_research.portfolio_fold_axis_invalid")
    decision_position = 0
    carried_fold: int | None = None
    values: list[int] = []
    for session in economic_sessions:
        if (
            decision_position < len(decision_sessions)
            and session == decision_sessions[decision_position]
        ):
            carried_fold = int(decision_fold_indices[decision_position])
            decision_position += 1
        elif carried_fold is None or (
            decision_position < len(decision_sessions)
            and session > decision_sessions[decision_position]
        ):
            raise PanelMethodologyExecutionError("alpha_research.portfolio_fold_axis_invalid")
        values.append(carried_fold)
    if decision_position != len(decision_sessions):
        raise PanelMethodologyExecutionError("alpha_research.portfolio_fold_axis_invalid")
    return cast(
        IntArray,
        _readonly(np.asarray(values, dtype=np.int64), dtype=np.dtype(np.int64)),
    )


class PanelResearchMethodologyExecutor:
    """Portfolio-owned cross-Desk loop; numerical owners remain injected capabilities."""

    kind = "alpha.model-development"

    def __init__(
        self,
        *,
        compiler: DeskExperimentCompiler,
        request: PanelResearchMethodologyRequest,
        preflight: PanelResearchPreflight,
        inputs: PanelMethodologyExecutionInputs,
        runtime_caps: PanelResearchRuntimeCaps | None = None,
    ) -> None:
        """Bind sealed methodology request/preflight, resolved inputs and operational resource caps.

        Args:
            compiler: Deterministic desk experiment compilation owner.
            request: Admitted methodology request.
            preflight: Immutable admitted metadata preflight.
            inputs: Resolved owned numerical inputs and evidence.
            runtime_caps: Optional CLI-applied operational caps; otherwise use preflight caps.
        """
        self.compiler = compiler
        self.request = request
        self.preflight = preflight
        self.inputs = inputs
        # A sealed Portfolio run reuses the immutable metadata preflight for
        # science but records the current CLI-applied operational plan on its
        # execution root.  Alpha paths use their preflight caps unchanged.
        self.runtime_caps = runtime_caps or preflight.caps
        self.stage_telemetry: dict[str, object] = {
            "feature_materialization_count": inputs.feature_materialization_count,
            "feature_materialization_seconds": inputs.feature_materialization_seconds,
            "owner_resolution_seconds": inputs.owner_resolution_seconds,
            "alpha_verification_seconds": inputs.alpha_verification_seconds,
            "sealed_receipt_load_seconds": inputs.sealed_receipt_load_seconds,
            "simple_score_materialization_count": (inputs.simple_score_materialization_count),
            "simple_score_materialization_seconds": (inputs.simple_score_materialization_seconds),
            "portfolio_numerical_path_seconds": 0.0,
        }
        if inputs.owner_resolution_metrics:
            self.stage_telemetry.update(
                {
                    "owner_resolution_metrics": [
                        {
                            "owner": owner,
                            "seconds": seconds,
                            "call_count": calls,
                            "bytes_read": bytes_read,
                        }
                        for owner, seconds, calls, bytes_read in inputs.owner_resolution_metrics
                    ],
                    "target_value_lane_read_count": 0,
                    "tradability_value_lane_read_count": 0,
                    "risk_return_chunk_read_count": 0,
                    "mark_value_lane_read_count": 0,
                    "covariance_projection_count": 0,
                    "optimizer_construction_count": 0,
                    "solver_call_count": 0,
                    "preflight_publication_count": 0,
                }
            )

    def seal_preflight_receipt(
        self, *, program: SealedResearchProgram, output_workspace: Path
    ) -> PanelPortfolioPreflightReceipt | None:
        """Publish the prepared Portfolio source receipt after Program sealing."""
        if self.request.execution_stage != "PORTFOLIO_ONLY":
            return None
        if self.inputs.sealed_preflight_receipt is not None:
            receipt = self.inputs.sealed_preflight_receipt
            if (
                receipt.program_hash != program.program_hash
                or receipt.preflight != self.preflight
                or self.inputs.sealed_preflight_program_hash != program.program_hash
            ):
                raise PanelMethodologyExecutionError(
                    "alpha_research.portfolio_preflight_program_mismatch"
                )
            return receipt
        receipt = publish_panel_portfolio_preflight_receipt(
            output_workspace=output_workspace,
            program=program,
            request=self.request,
            preflight=self.preflight,
            inputs=self.inputs,
        )
        self.stage_telemetry.update(
            {
                "feature_materialization_count": receipt.feature_materialization_count,
                "feature_materialization_seconds": receipt.feature_materialization_seconds,
                "owner_resolution_seconds": receipt.owner_resolution_seconds,
                "alpha_verification_seconds": receipt.alpha_verification_seconds,
                "simple_score_materialization_count": (receipt.simple_score_materialization_count),
                "simple_score_materialization_seconds": (
                    receipt.simple_score_materialization_seconds
                ),
                "portfolio_numerical_path_seconds": 0.0,
                "preflight_publication_count": 1,
            }
        )
        return receipt

    def execute(
        self,
        *,
        program: SealedResearchProgram,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
        output_workspace: Path,
        recorder: NumericalCallRecorder | None = None,
    ) -> DeskExecutionResult:
        """Execute the exact program under a distinct publication attempt and process-tree monitor.

        Args:
            program: Exact sealed research program.
            document: Declared experiment document.
            authority: Exact resolved research authority.
            output_workspace: Caller-owned output workspace copy.
            recorder: Optional numerical-call recorder.

        Returns:
            Desk execution result from deterministic monitored dispatch.

        Raises:
            PanelMethodologyExecutionError: Retained sealed preflight belongs to another program;
                execution refusals propagate after monitor cleanup.
        """
        if (
            self.inputs.sealed_preflight_program_hash is not None
            and self.inputs.sealed_preflight_program_hash != program.program_hash
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.portfolio_preflight_program_mismatch"
            )
        run_publication_attempt_hash = str(
            canonical_hash(
                {
                    "program_hash": program.program_hash,
                    "nonce": token_hex(32),
                }
            )
        )
        resource_monitor = ProcessResourceMonitor(include_live_descendants=True)
        resource_monitor.start()
        try:
            return self._execute_monitored(
                program=program,
                document=document,
                authority=authority,
                output_workspace=output_workspace,
                recorder=recorder,
                resource_monitor=resource_monitor,
                run_publication_attempt_hash=run_publication_attempt_hash,
            )
        except BaseException:
            resource_monitor.finish()
            raise

    def _run_alpha_folds(
        self, *, program_hash: str, store: PortfolioResearchArtifactStore
    ) -> tuple[
        tuple[_CompletedPanelAlphaFold, ...],
        tuple[_MonitoredPanelAlphaFoldWorkerResult, ...],
        int,
        int,
    ]:
        """Run missing folds in bounded workers and publish each completed fold."""

        plan = self.inputs.plan
        if plan is None:
            raise PanelMethodologyExecutionError(
                "alpha_research.feature_plan_not_installed_for_alpha_stage"
            )
        payload = PanelFeatureSourcePayload.from_source(plan.source)
        previously_completed = _load_completed_alpha_folds(
            store=store,
            plan=plan,
            outer_folds=self.inputs.outer_folds,
            program_hash=program_hash,
        )
        completed_indices = {value.evidence.fold_index for value in previously_completed}
        requests = tuple(
            PanelAlphaFoldWorkerRequest(
                program_hash=program_hash,
                expected_feature_preflight_hash=self.inputs.feature_preflight_hash,
                source=payload,
                fold=PanelAlphaFold(
                    fold_index=value.fold_index,
                    training_sessions=value.training_sessions,
                    validation_sessions=value.validation_sessions,
                    source_manifest_hash=value.source_manifest_hash,
                ),
                methodology=self.request,
                maximum_aggregation_span=1,
                numerical_thread_count=self.preflight.caps.lightgbm_threads_per_fit,
            )
            for value in self.inputs.outer_folds
            if value.fold_index not in completed_indices
        )
        if not requests:
            return previously_completed, (), len(previously_completed), 0
        worker_count = min(self.preflight.caps.maximum_fold_workers, len(requests))
        if worker_count < 1:
            raise PanelMethodologyExecutionError("alpha_research.panel_runtime_plan_not_admitted")
        worker_capacity_reservation = worker_count * PANEL_ALPHA_FOLD_RSS_FUSE_BYTES
        total_capacity_reservation = (
            PANEL_ALPHA_PARENT_RSS_RESERVATION_BYTES + worker_capacity_reservation
        )
        if (
            total_capacity_reservation > self.preflight.estimated_peak_memory_bytes
            or total_capacity_reservation > self.preflight.caps.maximum_estimated_peak_memory_bytes
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.panel_fold_worker_memory_budget_exceeded"
            )
        executed: list[_MonitoredPanelAlphaFoldWorkerResult] = []

        def admit_and_publish(worker: _MonitoredPanelAlphaFoldWorkerResult) -> None:
            result = worker.result
            if (
                result.feature_preflight_hash != self.inputs.feature_preflight_hash
                or result.numerical_thread_count != self.preflight.caps.lightgbm_threads_per_fit
                or result.covariance_open_count != 0
                or result.optimizer_call_count != 0
                or result.solver_call_count != 0
            ):
                raise PanelMethodologyExecutionError(
                    "alpha_research.panel_fold_worker_capacity_or_identity_invalid"
                )
            _admit_fold_before_publication(worker=worker, request=self.request)
            _publish_alpha_fold_worker_result(store=store, worker=result)
            executed.append(worker)

        if worker_count == 1:
            for request in requests:
                admit_and_publish(_execute_monitored_panel_alpha_fold_worker(request))
        else:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                computed = Parallel(
                    n_jobs=worker_count,
                    pre_dispatch=worker_count,
                    max_nbytes="64M",
                    mmap_mode="r",
                    return_as="generator_unordered",
                )(
                    delayed(_execute_monitored_panel_alpha_fold_worker)(value, owns_process=True)
                    for value in requests
                )
                for worker in computed:
                    admit_and_publish(cast(_MonitoredPanelAlphaFoldWorkerResult, worker))
        ordered = tuple(
            sorted(executed, key=lambda value: value.result.fold.fold_evidence.fold_index)
        )
        completed = _load_completed_alpha_folds(
            store=store,
            plan=plan,
            outer_folds=self.inputs.outer_folds,
            program_hash=program_hash,
        )
        if tuple(value.evidence.fold_index for value in completed) != tuple(
            value.fold_index for value in self.inputs.outer_folds
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.panel_fold_worker_capacity_or_identity_invalid"
            )
        return completed, ordered, len(previously_completed), worker_capacity_reservation

    def _execute_monitored(
        self,
        *,
        program: SealedResearchProgram,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
        output_workspace: Path,
        recorder: NumericalCallRecorder | None,
        resource_monitor: ProcessResourceMonitor,
        run_publication_attempt_hash: str,
    ) -> DeskExecutionResult:
        section = methodology_section(document)
        if section is None:
            raise AuthoringError("alpha_research.methodology_not_selected")
        envelope = document.get("experiment")
        if not isinstance(envelope, Mapping):
            raise AuthoringError("research_authoring.experiment_section_missing")
        from alphalattice.protocols.research_authoring.contracts import (
            ResearchExperimentEnvelope,
        )

        compiled = self.compiler.compile_desk_program(
            envelope=ResearchExperimentEnvelope.create(**envelope),
            document=document,
            authority=authority,
        )
        if compiled.desk_program_hash != program.desk_program_hash:
            raise AuthoringError("research_authoring.replay_identity_mismatch")
        store = PortfolioResearchArtifactStore(output_workspace / "portfolio-development")
        alpha_root: PanelMethodologyExecutionRoot | None = None
        worker_results: tuple[_MonitoredPanelAlphaFoldWorkerResult, ...] = ()
        reused_fold_count = 0
        worker_capacity_reservation = 0
        completed_folds: tuple[_CompletedPanelAlphaFold, ...]
        plan = self.inputs.plan
        if self.request.execution_stage in {"ALPHA_ONLY", "END_TO_END"}:
            (
                completed_folds,
                worker_results,
                reused_fold_count,
                worker_capacity_reservation,
            ) = self._run_alpha_folds(program_hash=program.program_hash, store=store)
        else:
            alpha_root = _resolve_compatible_stage_root(
                store=store,
                request=self.request,
                source_resolution_hash=self.inputs.source_resolution_hash,
                feature_preflight_hash=self.inputs.feature_preflight_hash,
                stage="ALPHA_ONLY",
            )
            if self.request.execution_stage == "PORTFOLIO_ONLY":
                receipt = self.inputs.fixed_alpha_row_axis_receipt
                if (
                    receipt is None
                    or receipt.receipt_hash != alpha_root.alpha_row_axis_receipt_hash
                    or receipt.receipt_hash == ""
                ):
                    raise PanelMethodologyExecutionError(
                        "alpha_research.fixed_alpha_row_axis_receipt_unavailable"
                    )
                completed_folds = _load_completed_alpha_folds(
                    store=store,
                    fixed_row_axes=self.inputs.fixed_alpha_row_axes,
                    outer_folds=self.inputs.outer_folds,
                    ordered_fold_hashes=alpha_root.ordered_fold_hashes,
                    ordered_score_surface_artifact_hashes=(
                        alpha_root.ordered_score_surface_artifact_hashes
                    ),
                )
            else:
                if plan is None:
                    raise PanelMethodologyExecutionError(
                        "alpha_research.feature_plan_not_installed_for_filter_stage"
                    )
                completed_folds = _load_completed_alpha_folds(
                    store=store,
                    plan=plan,
                    outer_folds=self.inputs.outer_folds,
                    ordered_fold_hashes=alpha_root.ordered_fold_hashes,
                )
        fold_evidence = tuple(value.evidence for value in completed_folds)
        alpha_fit_call_count = sum(value.fit_call_count for value in fold_evidence)
        alpha_predict_call_count = sum(value.predict_call_count for value in fold_evidence)
        alpha_metric_call_count = sum(value.metric_call_count for value in fold_evidence)
        attempted_fold_evidence = tuple(value.result.fold.fold_evidence for value in worker_results)
        attempt_fit_call_count = sum(value.fit_call_count for value in attempted_fold_evidence)
        attempt_predict_call_count = sum(
            value.predict_call_count for value in attempted_fold_evidence
        )
        attempt_metric_call_count = sum(
            value.metric_call_count for value in attempted_fold_evidence
        )
        worker_resource_receipts = tuple(
            _fold_resource_receipt(value)
            for value in sorted(
                worker_results,
                key=lambda item: item.result.fold.fold_evidence.fold_index,
            )
        )
        temporal_handoff: PanelScoreTemporalHandoff | None = None
        if self.request.execution_stage == "PORTFOLIO_ONLY":
            if alpha_root is None or alpha_root.root_hash != SEALED_PANEL_ALPHA_ROOT_HASH:
                raise PanelMethodologyExecutionError(
                    "alpha_research.fixed_panel_alpha_root_not_selected"
                )
            temporal_handoff = _fixed_panel_score_temporal_handoff(
                root=alpha_root,
                completed_folds=completed_folds,
                inputs=self.inputs,
            )
            raw_surfaces = tuple(
                next(
                    surface
                    for surface in value.fixed_surfaces
                    if surface.model_recipe_id == SEALED_PANEL_ALPHA_MODEL_RECIPE_ID
                )
                for value in completed_folds
            )
        else:
            raw_surfaces = tuple(value.raw_surface for value in completed_folds)
        control_surfaces = tuple(value.control_surface for value in completed_folds)
        raw_artifact_hashes = tuple(value.raw_artifact.artifact_hash for value in completed_folds)
        model_spec_ids_by_fold = tuple(value.model_spec_id for value in completed_folds)
        score_artifacts: list[PanelScoreSurfaceArtifact] = [
            artifact
            for value in completed_folds
            for artifact in (
                value.fixed_artifacts
                if value.fixed_artifacts
                else (value.raw_artifact, value.control_artifact)
            )
        ]
        scale_hashes = _scale_receipt_hashes_for_folds(
            store=store,
            folds=fold_evidence,
        )

        if self.request.execution_stage == "ALPHA_ONLY":
            economic = (
                _evaluate_fixed_alpha_economics(completed_folds)
                if fixed_recipe_economic_metric_call_count(
                    tuple(value.recipe_id for value in self.request.model_recipes)
                )
                > 0
                else None
            )
            if economic is not None:
                economic_metric_call_count = _alpha_economic_metric_call_count(economic)
                alpha_metric_call_count += economic_metric_call_count
                attempt_metric_call_count += economic_metric_call_count
            alpha_authority_handles = (
                _publish_fixed_alpha_successor_authority(
                    output_workspace=output_workspace,
                    program_hash=program.program_hash,
                    completed_folds=completed_folds,
                    inputs=self.inputs,
                )
                if SEALED_PANEL_ALPHA_MODEL_RECIPE_ID
                in {value.recipe_id for value in self.request.model_recipes}
                else None
            )
            if alpha_authority_handles is not None:
                self.stage_telemetry["simple_score_materialization_count"] = 1
                self.stage_telemetry["simple_score_materialization_seconds"] = (
                    alpha_authority_handles[5]
                )
            resource_usage = resource_monitor.finish()
            _admit_execution_resource_usage(
                runtime_caps=self.runtime_caps,
                usage=resource_usage,
            )
            if economic is None:
                _admit_preflight_call_counts(
                    preflight=self.preflight,
                    fit_call_count=alpha_fit_call_count,
                    predict_call_count=alpha_predict_call_count,
                    metric_call_count=alpha_metric_call_count,
                    solver_call_count=0,
                )
                economic_attestation = None
            else:
                economic_attestation = _publish_admitted_alpha_economic_evaluation(
                    store=store,
                    economic=economic,
                    program=program,
                    preflight=self.preflight,
                    source_resolution_hash=self.inputs.source_resolution_hash,
                    feature_preflight_hash=self.inputs.feature_preflight_hash,
                    run_publication_attempt_hash=run_publication_attempt_hash,
                    attempt_executed_fold_count=len(worker_results),
                    attempt_reused_fold_count=reused_fold_count,
                    attempt_fit_call_count=attempt_fit_call_count,
                    attempt_predict_call_count=attempt_predict_call_count,
                    attempt_metric_call_count=attempt_metric_call_count,
                    attempt_solver_call_count=0,
                    ordered_fold_hashes=tuple(value.fold_hash for value in fold_evidence),
                    fit_call_count=alpha_fit_call_count,
                    predict_call_count=alpha_predict_call_count,
                    metric_call_count=alpha_metric_call_count,
                    solver_call_count=0,
                )
            sessions = tuple(
                sorted({session for value in raw_surfaces for session in value.row_sessions})
            )
            input_binding_hash = str(
                canonical_hash(
                    {
                        "stage": "ALPHA_ONLY",
                        "program_hash": program.program_hash,
                        "source_resolution_hash": self.inputs.source_resolution_hash,
                        "ordered_fold_hashes": [value.fold_hash for value in fold_evidence],
                        "ordered_score_surface_hashes": [
                            value.source_surface_hash for value in score_artifacts
                        ],
                    }
                )
            )
            alpha_root_handles = alpha_authority_handles or (None,) * 6
            root = PanelMethodologyExecutionRoot.create(
                program_hash=program.program_hash,
                desk_program_hash=program.desk_program_hash,
                method_binding_hash=program.method_binding_hash,
                authority_hash=program.authority_hash,
                desk_input_binding_hash=input_binding_hash,
                source_resolution_hash=self.inputs.source_resolution_hash,
                request_hash=self.request.request_hash,
                alpha_configuration_hash=panel_alpha_configuration_hash(self.request),
                preflight_hash=self.preflight.preflight_hash,
                feature_preflight_hash=self.inputs.feature_preflight_hash,
                ordered_fold_hashes=tuple(value.fold_hash for value in fold_evidence),
                ordered_scale_receipt_hashes=scale_hashes,
                ordered_inner_trial_hashes=tuple(
                    value for fold in fold_evidence for value in fold.inner_trial_evidence_hashes
                ),
                ordered_outer_trial_hashes=tuple(
                    value for fold in fold_evidence for value in fold.outer_trial_evidence_hashes
                ),
                ordered_score_surface_artifact_hashes=tuple(
                    value.artifact_hash for value in score_artifacts
                ),
                alpha_score_surface_hash=alpha_root_handles[0],
                alpha_row_axis_receipt_hash=alpha_root_handles[1],
                alpha_simple_score_binding_hash=alpha_root_handles[2],
                alpha_simple_score_value_artifact_hash=alpha_root_handles[3],
                alpha_formation_readiness_receipt_hash=alpha_root_handles[4],
                alpha_economic_publication_attestation_hash=(
                    economic_attestation.attestation_hash
                    if economic_attestation is not None
                    else None
                ),
                common_formation_sessions=sessions,
                fit_call_count=alpha_fit_call_count,
                predict_call_count=alpha_predict_call_count,
                metric_call_count=alpha_metric_call_count,
                solver_call_count=0,
                executed_fold_count=len(worker_results),
                reused_fold_count=reused_fold_count,
                wall_seconds=resource_usage.wall_seconds,
                average_machine_cpu_percent=resource_usage.average_machine_cpu_percent,
                peak_machine_cpu_percent=resource_usage.peak_machine_cpu_percent,
                peak_rss_bytes=resource_usage.peak_rss_bytes,
                **_execution_resource_receipt_fields(
                    runtime_caps=self.runtime_caps,
                    usage=resource_usage,
                    worker_capacity_reservation_upper_bound_bytes=worker_capacity_reservation,
                    worker_resource_receipts=worker_resource_receipts,
                ),
            )
            store.publish(
                category=PANEL_METHODOLOGY_ROOT_CATEGORY,
                value=root,
                identity_field="root_hash",
            )
            if recorder is not None:
                recorder.record(capability=program.method_binding_hash)
            return DeskExecutionResult(
                disposition="COMPUTED",
                artifact_uris=(
                    f"playpen://portfolio-strategy-lab/{PANEL_METHODOLOGY_ROOT_CATEGORY}/"
                    f"{root.root_hash}",
                ),
                formation_sessions=sessions,
                numerical_call_count=(
                    root.fit_call_count + root.predict_call_count + root.metric_call_count
                ),
                desk_input_binding_hash=input_binding_hash,
            )

        filter_result: PanelScoreFilterEvaluationResult | None = None
        filter_root: PanelMethodologyExecutionRoot | None = None
        filter_surfaces_by_spec: Mapping[str, tuple[PanelScoreSurface, ...]]
        selected_filter_surfaces: tuple[PanelScoreSurface, ...]
        filter_evidence_hashes: tuple[str, ...]
        filter_selection_hashes: tuple[str, ...]
        filter_frontier_hash: str | None
        if self.request.execution_stage == "PORTFOLIO_ONLY":
            if self.request.upstream_score_filter_handle is None:
                filter_surfaces_by_spec = MappingProxyType({})
                selected_filter_surfaces = raw_surfaces
                filter_evidence_hashes = ()
                filter_selection_hashes = ()
                filter_frontier_hash = None
            else:
                filter_root = _resolve_compatible_stage_root(
                    store=store,
                    request=self.request,
                    source_resolution_hash=self.inputs.source_resolution_hash,
                    feature_preflight_hash=self.inputs.feature_preflight_hash,
                    stage="SCORE_FILTER_ONLY",
                )
                if (
                    alpha_root is None
                    or filter_root.upstream_alpha_root_hash != alpha_root.root_hash
                ):
                    raise PanelMethodologyExecutionError(
                        "alpha_research.upstream_filter_alpha_lineage_mismatch"
                    )
                if plan is None:
                    raise PanelMethodologyExecutionError(
                        "alpha_research.fixed_filter_row_axis_receipt_unavailable"
                    )
                (
                    filter_surfaces_by_spec,
                    selected_filter_surfaces,
                    filter_artifacts,
                ) = _load_upstream_filter_surfaces(
                    store=store,
                    root=filter_root,
                    upstream_alpha_root=alpha_root,
                    plan=plan,
                    outer_folds=self.inputs.outer_folds,
                )
                score_artifacts.extend(filter_artifacts)
                filter_evidence_hashes = filter_root.ordered_score_filter_evidence_hashes
                filter_selection_hashes = filter_root.ordered_score_filter_selection_hashes
                filter_frontier_hash = filter_root.score_filter_frontier_hash
        else:
            if plan is None:
                raise PanelMethodologyExecutionError(
                    "alpha_research.feature_plan_not_installed_for_filter_stage"
                )
            filter_result, filter_surfaces_by_spec, selected_filter_surfaces = (
                _evaluate_oof_score_filters(
                    raw_surfaces=raw_surfaces,
                    model_spec_ids_by_fold=model_spec_ids_by_fold,
                    upstream_raw_score_artifact_hashes=raw_artifact_hashes,
                    request=self.request,
                    plan=plan,
                )
            )
            for evidence in filter_result.evidence:
                store.publish(
                    category="development/paired-panel-research/score-filter-evidence",
                    value=evidence,
                    identity_field="evidence_hash",
                )
            for selection in filter_result.selections:
                store.publish(
                    category="development/paired-panel-research/score-filter-selections",
                    value=selection,
                    identity_field="selection_hash",
                )
            evidence_by_spec = {
                value.filter_spec.spec_id: value for value in filter_result.evidence
            }
            selection_by_fold = {value.fold_index: value for value in filter_result.selections}
            raw_artifact_by_fold = {
                value.evidence.fold_index: value.raw_artifact for value in completed_folds
            }
            for spec_id, surfaces in filter_surfaces_by_spec.items():
                evidence = evidence_by_spec[spec_id]
                score_artifacts.extend(
                    _publish_score_surface(
                        store=store,
                        surface=surface,
                        producer_spec_id=spec_id,
                        upstream_raw_score_artifact_hash=raw_artifact_by_fold[
                            surface.fold_index
                        ].artifact_hash,
                        filter_spec_hash=evidence.filter_spec.spec_hash,
                        filter_evidence_hash=evidence.evidence_hash,
                        fold_selection_hash=selection_by_fold[surface.fold_index].selection_hash,
                    )
                    for surface in surfaces
                )
            filter_evidence_hashes = tuple(value.evidence_hash for value in filter_result.evidence)
            filter_selection_hashes = tuple(
                value.selection_hash for value in filter_result.selections
            )
            filter_frontier_hash = None

        if self.request.execution_stage == "SCORE_FILTER_ONLY":
            if alpha_root is None or filter_result is None:
                raise PanelMethodologyExecutionError(
                    "alpha_research.upstream_alpha_root_not_resolved"
                )
            resource_usage = resource_monitor.finish()
            _admit_execution_resource_usage(
                runtime_caps=self.runtime_caps,
                usage=resource_usage,
            )
            sessions = tuple(
                sorted({session for surface in raw_surfaces for session in surface.row_sessions})
            )
            input_binding_hash = str(
                canonical_hash(
                    {
                        "stage": "SCORE_FILTER_ONLY",
                        "program_hash": program.program_hash,
                        "upstream_alpha_root_hash": alpha_root.root_hash,
                        "score_filter_spec_hashes": [
                            value.spec_hash for value in self.request.score_filter_candidates
                        ],
                        "ordered_score_surface_artifact_hashes": [
                            value.artifact_hash for value in score_artifacts
                        ],
                    }
                )
            )
            metric_call_count = len(filter_result.evidence) * (
                1 + 3 * len(self.request.persistence_lags)
            )
            _admit_preflight_call_counts(
                preflight=self.preflight,
                fit_call_count=0,
                predict_call_count=0,
                metric_call_count=metric_call_count,
                solver_call_count=0,
            )
            root = PanelMethodologyExecutionRoot.create(
                program_hash=program.program_hash,
                desk_program_hash=program.desk_program_hash,
                method_binding_hash=program.method_binding_hash,
                authority_hash=program.authority_hash,
                desk_input_binding_hash=input_binding_hash,
                source_resolution_hash=self.inputs.source_resolution_hash,
                request_hash=self.request.request_hash,
                alpha_configuration_hash=panel_alpha_configuration_hash(self.request),
                upstream_alpha_root_hash=alpha_root.root_hash,
                preflight_hash=self.preflight.preflight_hash,
                feature_preflight_hash=self.inputs.feature_preflight_hash,
                ordered_fold_hashes=alpha_root.ordered_fold_hashes,
                ordered_scale_receipt_hashes=alpha_root.ordered_scale_receipt_hashes,
                ordered_inner_trial_hashes=alpha_root.ordered_inner_trial_hashes,
                ordered_outer_trial_hashes=alpha_root.ordered_outer_trial_hashes,
                ordered_score_surface_artifact_hashes=tuple(
                    value.artifact_hash for value in score_artifacts
                ),
                ordered_score_filter_evidence_hashes=tuple(filter_evidence_hashes),
                ordered_score_filter_selection_hashes=tuple(filter_selection_hashes),
                score_filter_frontier_hash=filter_frontier_hash,
                common_formation_sessions=sessions,
                fit_call_count=0,
                predict_call_count=0,
                metric_call_count=metric_call_count,
                solver_call_count=0,
                executed_fold_count=0,
                reused_fold_count=len(completed_folds),
                wall_seconds=resource_usage.wall_seconds,
                average_machine_cpu_percent=resource_usage.average_machine_cpu_percent,
                peak_machine_cpu_percent=resource_usage.peak_machine_cpu_percent,
                peak_rss_bytes=resource_usage.peak_rss_bytes,
                **_execution_resource_receipt_fields(
                    runtime_caps=self.runtime_caps,
                    usage=resource_usage,
                ),
            )
            store.publish(
                category=PANEL_METHODOLOGY_ROOT_CATEGORY,
                value=root,
                identity_field="root_hash",
            )
            return DeskExecutionResult(
                disposition="COMPUTED",
                artifact_uris=(
                    f"playpen://portfolio-strategy-lab/{PANEL_METHODOLOGY_ROOT_CATEGORY}/"
                    f"{root.root_hash}",
                ),
                formation_sessions=sessions,
                numerical_call_count=metric_call_count,
                desk_input_binding_hash=input_binding_hash,
            )

        axis = self.inputs.portfolio_axis
        r1_required = self.request.portfolio_policy_ids != ("RANK_BUFFERED_SCORE_RISK_COST",)
        if (
            axis is None
            or self.inputs.portfolio_market is None
            or self.inputs.portfolio_benchmark is None
            or self.inputs.portfolio_fold_indices is None
            or self.inputs.r0_covariance is None
            or (r1_required and self.inputs.r1_covariance is None)
            or (
                self.inputs.simple_score_authority is None
                and (
                    self.inputs.fixed_simple_score_binding is None
                    or self.inputs.fixed_simple_score_values is None
                )
            )
            or self.request.fixed_score_method_id is None
            or self.request.top_k is None
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.panel_methodology_portfolio_inputs_not_installed"
            )
        sessions = axis.formation_sessions
        listings = axis.ordered_listing_ids
        if self.inputs.sealed_preflight_program_hash is not None:
            market = self.inputs.portfolio_market
            benchmark = self.inputs.portfolio_benchmark
            fold_indices = self.inputs.portfolio_fold_indices
            if (
                market.formation_sessions != sessions
                or market.economic_formation_sessions != axis.economic_formation_sessions
                or market.ordered_listing_ids != listings
                or benchmark.formation_sessions != sessions
                or benchmark.economic_formation_sessions != axis.economic_formation_sessions
                or fold_indices.shape != (len(sessions),)
            ):
                raise PanelMethodologyExecutionError(
                    "alpha_research.portfolio_preflight_prepared_axis_mismatch"
                )
        else:
            market, benchmark, fold_indices = _subset_market(
                market=self.inputs.portfolio_market,
                benchmark=self.inputs.portfolio_benchmark,
                fold_indices=self.inputs.portfolio_fold_indices,
                sessions=sessions,
                economic_sessions=axis.economic_formation_sessions,
                passive_sessions=axis.passive_sessions,
                listings=listings,
                listing_positions=axis.market_listing_positions,
            )
        if temporal_handoff is None:
            raise PanelMethodologyExecutionError(
                "alpha_research.panel_score_temporal_authority_not_installed"
            )
        temporal_admission = _admit_fixed_panel_portfolio_inputs(
            handoff=temporal_handoff,
            inputs=self.inputs,
            axis=axis,
            market=market,
        )
        dynamic_surfaces = selected_filter_surfaces
        raw_dynamic_surfaces = raw_surfaces
        dynamic_scores = _score_matrix(
            surfaces=dynamic_surfaces, sessions=sessions, listings=listings
        )
        raw_dynamic_scores = _score_matrix(
            surfaces=raw_dynamic_surfaces, sessions=sessions, listings=listings
        )
        control_scores = _score_matrix(
            surfaces=control_surfaces, sessions=sessions, listings=listings
        )
        declared_filtered_scores = tuple(
            DeclaredFilteredScore(
                spec_id=spec_id,
                surface_hash=str(canonical_hash([value.surface_hash for value in surfaces])),
                values=_score_matrix(
                    surfaces=surfaces,
                    sessions=sessions,
                    listings=listings,
                ),
            )
            for spec_id, surfaces in sorted(filter_surfaces_by_spec.items())
        )
        fixed_simple_score_is_durable = self.inputs.fixed_simple_score_binding is not None
        if fixed_simple_score_is_durable:
            simple_binding = self.inputs.fixed_simple_score_binding
            simple_values = self.inputs.fixed_simple_score_values
        else:
            assert self.inputs.simple_score_authority is not None
            simple_score = self.inputs.simple_score_authority.materialize(
                method_id=self.request.fixed_score_method_id,
                ordered_formation_sessions=sessions,
                ordered_listing_ids=listings,
            )
            simple_binding = simple_score.binding
            simple_values = simple_score.standardized_values
        assert simple_binding is not None
        if (
            simple_values is None
            or simple_values.shape != (len(sessions), len(listings))
            or not np.isfinite(simple_values).all()
            or simple_binding.minimum_finite_listings_observed < self.request.top_k
            or (
                not fixed_simple_score_is_durable
                and simple_score_values_identity(simple_values) != simple_binding.values_identity
            )
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.fixed_simple_score_common_axis_incomplete"
            )
        fixed_simple_artifact_hash = self.inputs.fixed_simple_score_value_artifact_hash
        if (
            fixed_simple_artifact_hash is None
            or self.inputs.fixed_simple_score_binding_hash != simple_binding.binding_hash
        ):
            raise PanelMethodologyExecutionError(
                "alpha_research.simple_signal_value_artifact_handle_required"
            )
        r0_covariance_projection = self.inputs.r0_covariance.project(
            sessions=sessions, positions=axis.r0_listing_positions
        )
        r1_covariance_projection = (
            self.inputs.r1_covariance.project(
                sessions=sessions,
                positions=cast(IntArray, axis.r1_listing_positions),
            )
            if self.inputs.r1_covariance is not None and axis.r1_listing_positions is not None
            else None
        )
        r0_covariance = r0_covariance_projection.values
        r1_covariance = (
            None if r1_covariance_projection is None else r1_covariance_projection.values
        )
        if self.inputs.reference_marks is None or self.inputs.execution_clock is None:
            raise PanelMethodologyExecutionError(
                "alpha_research.portfolio_state_transition_not_installed"
            )
        reference_marks = project_reference_mark_lane(
            resolved=self.inputs.reference_marks,
            formation_sessions=axis.formation_sessions,
            ordered_listing_ids=axis.ordered_listing_ids,
        )
        dynamic_hash = str(canonical_hash([value.surface_hash for value in dynamic_surfaces]))
        raw_dynamic_hash = str(
            canonical_hash([value.surface_hash for value in raw_dynamic_surfaces])
        )
        control_hash = str(canonical_hash([value.surface_hash for value in control_surfaces]))
        input_binding_hash = str(
            canonical_hash(
                {
                    "program_hash": program.program_hash,
                    "source_resolution_hash": self.inputs.source_resolution_hash,
                    "upstream_alpha_root_hash": (
                        alpha_root.root_hash if alpha_root is not None else None
                    ),
                    "upstream_score_filter_root_hash": (
                        filter_root.root_hash if filter_root is not None else None
                    ),
                    "dynamic_score_surface_hash": dynamic_hash,
                    "raw_dynamic_score_surface_hash": raw_dynamic_hash,
                    "control_score_surface_hash": control_hash,
                    "fixed_simple_score_surface_hash": simple_binding.values_identity,
                    "fixed_simple_score_binding_hash": simple_binding.binding_hash,
                    "declared_filter_surface_hashes": {
                        value.spec_id: value.surface_hash for value in declared_filtered_scores
                    },
                    "r0_surface_hash": self.inputs.r0_covariance.surface_hash,
                    "r1_surface_hash": (
                        self.inputs.r1_covariance.surface_hash
                        if self.inputs.r1_covariance is not None
                        else None
                    ),
                    "market_surface_hash": market.source_surface_hash,
                    "benchmark_surface_hash": benchmark.surface_hash,
                    "state_transition_binding_hash": reference_marks.binding.binding_hash,
                    "execution_events_hash": self.inputs.execution_clock.events.events_hash,
                    "alpha_score_temporal_handoff_hash": temporal_handoff.handoff_hash,
                    "strategy_schedule_hash": temporal_admission.strategy_schedule_hash,
                    "causal_admission_hash": temporal_admission.admission_hash,
                    "sessions": [value.isoformat() for value in sessions],
                    "economic_sessions": [
                        value.isoformat() for value in axis.economic_formation_sessions
                    ],
                    "decision_ranges": axis.decision_ranges,
                    "passive_sessions": [value.isoformat() for value in axis.passive_sessions],
                    "listings": list(listings),
                }
            )
        )
        portfolio_inputs = PairedAlphaPortfolioInputs(
            formation_sessions=sessions,
            ordered_listing_ids=listings,
            fold_indices=fold_indices,
            economic_fold_indices=_economic_fold_indices(
                decision_sessions=sessions,
                decision_fold_indices=fold_indices,
                economic_sessions=axis.economic_formation_sessions,
            ),
            decision_ranges=axis.decision_ranges,
            passive_sessions=axis.passive_sessions,
            dynamic_scores=dynamic_scores,
            control_scores=control_scores,
            raw_dynamic_scores=raw_dynamic_scores,
            fixed_simple_scores=simple_values,
            declared_filtered_scores=declared_filtered_scores,
            dynamic_score_surface_hash=dynamic_hash,
            control_score_surface_hash=control_hash,
            raw_dynamic_score_surface_hash=raw_dynamic_hash,
            fixed_simple_score_surface_hash=simple_binding.values_identity,
            fixed_simple_score_binding_hash=simple_binding.binding_hash,
            selected_aggregation_span=max(value.span_sessions for value in dynamic_surfaces),
            r0_covariances=r0_covariance,
            r0_covariance_validation=r0_covariance_projection,
            r1_covariances=r1_covariance,
            r1_covariance_validation=r1_covariance_projection,
            r0_surface_hash=self.inputs.r0_covariance.surface_hash,
            r1_surface_hash=(
                self.inputs.r1_covariance.surface_hash
                if self.inputs.r1_covariance is not None
                else None
            ),
            market=market,
            benchmark=benchmark,
            state_transition=reference_marks.binding,
            reference_mark=reference_marks.lane,
            execution_events_hash=self.inputs.execution_clock.events.events_hash,
            alpha_score_temporal_handoff_hash=temporal_handoff.handoff_hash,
            strategy_schedule_hash=temporal_admission.strategy_schedule_hash,
            causal_admission_hash=temporal_admission.admission_hash,
            input_binding_hash=input_binding_hash,
        )
        numerical_started = perf_counter()
        try:
            portfolio = run_paired_alpha_portfolio_research(
                program_hash=program.program_hash,
                inputs=portfolio_inputs,
                store=store,
                portfolio_policy_ids=self.request.portfolio_policy_ids,
            )
        finally:
            self.stage_telemetry["portfolio_numerical_path_seconds"] = (
                perf_counter() - numerical_started
            )
        replay = replay_paired_alpha_portfolio_research(
            program_hash=program.program_hash,
            inputs=portfolio_inputs,
            root=portfolio.root,
            store=store,
        )
        resource_usage = resource_monitor.finish()
        _admit_execution_resource_usage(
            runtime_caps=self.runtime_caps,
            usage=resource_usage,
        )
        filter_metric_call_count = (
            len(filter_result.evidence) * (1 + 3 * len(self.request.persistence_lags))
            if filter_result is not None
            else 0
        )
        stage_alpha_fit_calls = (
            alpha_fit_call_count if self.request.execution_stage == "END_TO_END" else 0
        )
        stage_alpha_predict_calls = (
            alpha_predict_call_count if self.request.execution_stage == "END_TO_END" else 0
        )
        stage_alpha_metric_calls = (
            alpha_metric_call_count if self.request.execution_stage == "END_TO_END" else 0
        )
        _admit_preflight_call_counts(
            preflight=self.preflight,
            fit_call_count=stage_alpha_fit_calls,
            predict_call_count=stage_alpha_predict_calls,
            metric_call_count=stage_alpha_metric_calls + filter_metric_call_count,
            solver_call_count=portfolio.root.solver_call_count,
        )
        root = PanelMethodologyExecutionRoot.create(
            program_hash=program.program_hash,
            desk_program_hash=program.desk_program_hash,
            method_binding_hash=program.method_binding_hash,
            authority_hash=program.authority_hash,
            desk_input_binding_hash=input_binding_hash,
            source_resolution_hash=self.inputs.source_resolution_hash,
            request_hash=self.request.request_hash,
            alpha_configuration_hash=panel_alpha_configuration_hash(self.request),
            upstream_alpha_root_hash=alpha_root.root_hash if alpha_root is not None else None,
            upstream_score_filter_root_hash=(
                filter_root.root_hash if filter_root is not None else None
            ),
            preflight_hash=self.preflight.preflight_hash,
            feature_preflight_hash=self.inputs.feature_preflight_hash,
            ordered_fold_hashes=tuple(value.fold_hash for value in fold_evidence),
            ordered_scale_receipt_hashes=scale_hashes,
            ordered_inner_trial_hashes=tuple(
                value for fold in fold_evidence for value in fold.inner_trial_evidence_hashes
            ),
            ordered_outer_trial_hashes=tuple(
                value for fold in fold_evidence for value in fold.outer_trial_evidence_hashes
            ),
            ordered_score_surface_artifact_hashes=tuple(
                value.artifact_hash for value in score_artifacts
            ),
            ordered_score_filter_evidence_hashes=filter_evidence_hashes,
            ordered_score_filter_selection_hashes=filter_selection_hashes,
            score_filter_frontier_hash=filter_frontier_hash,
            fixed_simple_score_artifact_hash=fixed_simple_artifact_hash,
            portfolio_root_hash=portfolio.root.root_hash,
            portfolio_replay_receipt_hash=replay.receipt_hash,
            common_formation_sessions=sessions,
            portfolio_common_sessions=sessions,
            portfolio_economic_sessions=axis.economic_formation_sessions,
            portfolio_ordered_listing_ids_hash=str(canonical_hash(list(listings))),
            fit_call_count=stage_alpha_fit_calls,
            predict_call_count=stage_alpha_predict_calls,
            metric_call_count=stage_alpha_metric_calls + filter_metric_call_count,
            solver_call_count=portfolio.root.solver_call_count,
            executed_fold_count=len(worker_results),
            reused_fold_count=(len(completed_folds) - len(worker_results)),
            wall_seconds=resource_usage.wall_seconds,
            average_machine_cpu_percent=resource_usage.average_machine_cpu_percent,
            peak_machine_cpu_percent=resource_usage.peak_machine_cpu_percent,
            peak_rss_bytes=resource_usage.peak_rss_bytes,
            **_execution_resource_receipt_fields(
                runtime_caps=self.runtime_caps,
                usage=resource_usage,
                worker_capacity_reservation_upper_bound_bytes=worker_capacity_reservation,
                worker_resource_receipts=worker_resource_receipts,
            ),
        )
        store.publish(
            category=PANEL_METHODOLOGY_ROOT_CATEGORY,
            value=root,
            identity_field="root_hash",
        )
        if recorder is not None:
            recorder.record(capability=program.method_binding_hash)
        return DeskExecutionResult(
            disposition="COMPUTED",
            artifact_uris=(
                f"playpen://portfolio-strategy-lab/{PANEL_METHODOLOGY_ROOT_CATEGORY}/"
                f"{root.root_hash}",
            ),
            formation_sessions=sessions,
            numerical_call_count=(
                root.fit_call_count
                + root.predict_call_count
                + root.metric_call_count
                + root.solver_call_count
            ),
            desk_input_binding_hash=input_binding_hash,
        )


def load_panel_methodology_root(
    *, output_workspace: Path, root_hash: str
) -> PanelMethodologyExecutionRoot:
    """Read one exact root manifest through its publishing owner."""
    return PortfolioResearchArtifactStore(output_workspace / "portfolio-development").load(
        category=PANEL_METHODOLOGY_ROOT_CATEGORY,
        content_hash=root_hash,
        model=PanelMethodologyExecutionRoot,
        identity_field="root_hash",
    )


def verify_panel_methodology_graph(
    *,
    output_workspace: Path,
    program: SealedResearchProgram | None,
    root_hash: str,
    _visited_root_hashes: frozenset[str] = frozenset(),
) -> PanelMethodologyExecutionRoot:
    """Read every durable child through its publishing owner; never compute."""
    if root_hash in _visited_root_hashes:
        raise AuthoringError("alpha_research.panel_methodology_root_cycle")
    store = PortfolioResearchArtifactStore(output_workspace / "portfolio-development")
    root = load_panel_methodology_root(
        output_workspace=output_workspace,
        root_hash=root_hash,
    )
    if program is not None and (
        root.program_hash != program.program_hash
        or root.desk_program_hash != program.desk_program_hash
        or root.method_binding_hash != program.method_binding_hash
        or root.authority_hash != program.authority_hash
    ):
        raise AuthoringError("alpha_research.panel_methodology_root_not_this_program")
    if root.alpha_row_axis_receipt_hash is not None:
        alpha_store = AlphaDevelopmentArtifactStore(output_workspace / "portfolio-development")
        row_receipt = alpha_store.load_fixed_alpha_row_axis_receipt(
            root.alpha_row_axis_receipt_hash
        )
        loaded_receipt, _row_axes = load_fixed_alpha_row_axes(
            output_workspace=output_workspace,
            root_hash=root.root_hash,
            receipt_hash=row_receipt.receipt_hash,
            panel_snapshot_hash=row_receipt.panel_snapshot_hash,
        )
        binding, _artifact, _values = alpha_store.load_simple_signed_score_values(
            artifact_hash=cast(str, root.alpha_simple_score_value_artifact_hash),
            binding_hash=cast(str, root.alpha_simple_score_binding_hash),
        )
        readiness = alpha_store.load_panel_score_formation_readiness(
            cast(str, root.alpha_formation_readiness_receipt_hash),
            program_hash=root.program_hash,
            score_surface_hash=cast(str, root.alpha_score_surface_hash),
            formation_sessions=root.common_formation_sessions,
        )
        if (
            loaded_receipt.program_hash != root.program_hash
            or binding.ordered_formation_sessions != root.common_formation_sessions
            or binding.ordered_listing_ids != loaded_receipt.ordered_listing_ids
            or readiness.program_hash != root.program_hash
        ):
            raise AuthoringError("alpha_research.panel_score_successor_graph_invalid")
    economic_publication = _load_admitted_alpha_economic_publication(
        store=store,
        root=root,
    )
    visited = _visited_root_hashes | {root_hash}
    upstream_roots: dict[str, PanelMethodologyExecutionRoot] = {}
    for role, upstream_hash in (
        ("ALPHA", root.upstream_alpha_root_hash),
        ("SCORE_FILTER", root.upstream_score_filter_root_hash),
    ):
        if upstream_hash is None:
            continue
        upstream = verify_panel_methodology_graph(
            output_workspace=output_workspace,
            program=None,
            root_hash=upstream_hash,
            _visited_root_hashes=visited,
        )
        if (
            upstream.source_resolution_hash != root.source_resolution_hash
            or upstream.feature_preflight_hash != root.feature_preflight_hash
            or upstream.alpha_configuration_hash != root.alpha_configuration_hash
        ):
            raise AuthoringError("alpha_research.panel_methodology_upstream_identity_mismatch")
        upstream_roots[role] = upstream
    if (
        filter_root := upstream_roots.get("SCORE_FILTER")
    ) is not None and filter_root.upstream_alpha_root_hash != root.upstream_alpha_root_hash:
        raise AuthoringError("alpha_research.panel_methodology_filter_lineage_mismatch")
    from alphalattice.capabilities.alpha_modeling.contracts import (
        AlphaEstimatorContent,
        AlphaFitProvenanceReceipt,
        AlphaModelStateProjection,
    )
    from alphalattice.capabilities.alpha_modeling.runtime.numerical_environment import (
        AlphaModelNumericalEnvironment,
    )
    from alphalattice.investment.alpha_research.experiments.panel_methodology_models import (
        PanelModelFoldEvidence,
        PanelModelTrialEvidence,
    )
    from alphalattice.investment.alpha_research.experiments.panel_methodology_statistics import (
        CausalCandidateSelection,
        CausalModelSpecSelection,
        CausalScoreAggregationSelection,
        PanelModelScientificSelection,
        PanelScoreFilterEvidence,
        PanelScoreFilterSelection,
    )
    from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
        FoldScaleReceipt,
    )

    for value in root.ordered_fold_hashes:
        fold = store.load(
            category="development/paired-panel-research/folds",
            content_hash=value,
            model=PanelModelFoldEvidence,
            identity_field="fold_hash",
        )
        for selection_hash in (
            *fold.group_selection_hashes,
            fold.deployable_selection_hash,
            fold.dynamic_challenger_selection_hash,
        ):
            selection = store.load(
                category="development/paired-panel-research/selections",
                content_hash=selection_hash,
                model=(
                    PanelModelScientificSelection
                    if fold.selection_contract_kind == "PanelModelScientificSelection"
                    else (
                        CausalModelSpecSelection
                        if fold.score_aggregation_selection_hash is not None
                        else CausalCandidateSelection
                    )
                ),
                identity_field="selection_hash",
            )
            if selection.fold_index != fold.fold_index:
                raise AuthoringError("alpha_research.panel_selection_fold_identity_mismatch")
        if fold.score_aggregation_selection_hash is not None:
            aggregation_selection = store.load(
                category="development/paired-panel-research/selections",
                content_hash=fold.score_aggregation_selection_hash,
                model=CausalScoreAggregationSelection,
                identity_field="selection_hash",
            )
            if aggregation_selection.fold_index != fold.fold_index:
                raise AuthoringError("alpha_research.panel_selection_fold_identity_mismatch")
    for value in root.ordered_scale_receipt_hashes:
        store.load(
            category="development/paired-panel-research/scale-receipts",
            content_hash=value,
            model=FoldScaleReceipt,
            identity_field="receipt_hash",
        )
    for category, hashes in (
        ("development/paired-panel-research/inner-trials", root.ordered_inner_trial_hashes),
        ("development/paired-panel-research/outer-trials", root.ordered_outer_trial_hashes),
    ):
        for value in hashes:
            trial = store.load(
                category=category,
                content_hash=value,
                model=PanelModelTrialEvidence,
                identity_field="evidence_hash",
            )
            if not set(trial.scale_receipt_hashes).issubset(root.ordered_scale_receipt_hashes):
                raise AuthoringError("alpha_research.panel_trial_scale_lineage_mismatch")
            estimator = store.load(
                category="development/paired-panel-research/estimators",
                content_hash=trial.estimator_content_hash,
                model=AlphaEstimatorContent,
                identity_field="content_hash",
            )
            store.load(
                category="development/paired-panel-research/state-projections",
                content_hash=trial.state_projection_hash,
                model=AlphaModelStateProjection,
                identity_field="projection_hash",
            )
            provenance = store.load(
                category="development/paired-panel-research/provenance",
                content_hash=trial.provenance_hash,
                model=AlphaFitProvenanceReceipt,
                identity_field="provenance_hash",
            )
            environment = store.load(
                category="development/paired-panel-research/environments",
                content_hash=trial.numerical_environment_hash,
                model=AlphaModelNumericalEnvironment,
                identity_field="environment_hash",
            )
            if (
                estimator.content_hash != provenance.estimator_content_hash
                or provenance.numerical_environment_hash != environment.environment_hash
                or provenance.recipe_hash != trial.recipe_hash
                or provenance.training_binding_hash != trial.training_binding_hash
            ):
                raise AuthoringError("alpha_research.panel_trial_model_lineage_mismatch")
    for value in root.ordered_score_filter_evidence_hashes:
        filter_evidence = store.load(
            category="development/paired-panel-research/score-filter-evidence",
            content_hash=value,
            model=PanelScoreFilterEvidence,
            identity_field="evidence_hash",
        )
        if not set(filter_evidence.upstream_raw_score_artifact_hashes).issubset(
            root.ordered_score_surface_artifact_hashes
        ):
            raise AuthoringError("alpha_research.score_filter_upstream_lineage_missing")
    for value in root.ordered_score_filter_selection_hashes:
        store.load(
            category="development/paired-panel-research/score-filter-selections",
            content_hash=value,
            model=PanelScoreFilterSelection,
            identity_field="selection_hash",
        )
    if root.score_filter_frontier_hash is not None:
        raise AuthoringError("alpha_research.retired_score_filter_frontier_not_executable")
    filtered_surface_artifacts: list[PanelScoreSurfaceArtifact] = []
    score_surface_by_hash: dict[str, PanelScoreSurfaceArtifact] = {}
    for value in root.ordered_score_surface_artifact_hashes:
        surface = store.load(
            category="development/paired-panel-research/score-surfaces",
            content_hash=value,
            model=PanelScoreSurfaceArtifact,
            identity_field="artifact_hash",
        )
        score_surface_by_hash[surface.artifact_hash] = surface
        if surface.producer_spec_id is not None:
            alpha_root = upstream_roots.get("ALPHA")
            if alpha_root is None:
                raise AuthoringError("alpha_research.score_filter_alpha_lineage_missing")
            filtered_surface_artifacts.append(surface)
        for category, lane_hash in (
            ("score-lanes", surface.score_lane_hash),
            ("target-z-lanes", surface.target_z_lane_hash),
            ("raw-simple-return-lanes", surface.raw_simple_return_lane_hash),
        ):
            store.load_packed_bytes(
                category=f"development/paired-panel-research/{category}",
                content_hash=lane_hash,
            )
    if filtered_surface_artifacts:
        alpha_root = upstream_roots.get("ALPHA")
        if alpha_root is None:
            raise AuthoringError("alpha_research.score_filter_alpha_lineage_missing")
        _verify_filtered_score_surface_group(
            store=store,
            artifacts=tuple(filtered_surface_artifacts),
            upstream_alpha_root=alpha_root,
        )
    if economic_publication is not None:
        _, economic = economic_publication
        for candidate in economic.candidates:
            try:
                artifacts = tuple(
                    score_surface_by_hash[value] for value in candidate.ordered_score_surface_hashes
                )
            except KeyError as error:
                raise AuthoringError("alpha_research.economic_score_lineage_mismatch") from error
            if (
                any(value.model_recipe_id != candidate.model_recipe_id for value in artifacts)
                or tuple(value.row_axis_hash for value in artifacts)
                != economic.ordered_common_row_axis_hashes
                or tuple(value.raw_simple_return_lane_hash for value in artifacts)
                != economic.ordered_raw_simple_return_lane_hashes
            ):
                raise AuthoringError("alpha_research.economic_score_lineage_mismatch")
        raw_values = np.concatenate(
            [
                np.frombuffer(
                    store.load_packed_bytes(
                        category="development/paired-panel-research/raw-simple-return-lanes",
                        content_hash=value,
                    ),
                    dtype=np.float64,
                )
                for value in economic.ordered_raw_simple_return_lane_hashes
            ]
        )
        if score_value_hash(raw_values) != economic.raw_simple_return_value_hash:
            raise AuthoringError("alpha_research.economic_score_lineage_mismatch")
    if root.portfolio_root_hash is None:
        return root
    try:
        verify_paired_alpha_portfolio_graph(
            output_workspace=output_workspace,
            store=store,
            root_hash=root.portfolio_root_hash,
            fixed_simple_score_artifact_hash=root.fixed_simple_score_artifact_hash,
            formation_sessions=root.portfolio_common_sessions,
            replay_receipt_hash=root.portfolio_replay_receipt_hash,
        )
    except (FileNotFoundError, ValueError) as error:
        raise AuthoringError("alpha_research.paired_portfolio_graph_invalid") from error
    return root


class PanelMethodologyEvidenceVerifier:
    """Adapt the Portfolio-owned graph verifier to the Alpha Desk replay seam."""

    replay_disposition = "VERIFIED_DURABLE_GRAPH_READBACK"

    def applies(self, evidence: ResearchExecutionEvidence) -> bool:
        """Recognize methodology evidence by its registered root category URI.

        Args:
            evidence: Execution evidence artifact URI declaration.

        Returns:
            True when a methodology-root category URI is present.
        """
        return any(f"/{PANEL_METHODOLOGY_ROOT_CATEGORY}/" in uri for uri in evidence.artifact_uris)

    def verify(
        self,
        *,
        program: SealedResearchProgram,
        evidence: ResearchExecutionEvidence,
        authority: ResolvedResearchAuthority | None,
        output_workspace: Path,
    ) -> None:
        """Reopen one methodology graph and reconcile exact authority, support and input evidence.

        Args:
            program: Exact sealed program to verify.
            evidence: Execution evidence declaring exactly one methodology root.
            authority: Optional exact expected research authority.
            output_workspace: Caller-owned output workspace copy.

        Raises:
            AuthoringError: Root is not unique or reopened authority/support/input binding differs;
                graph validation refusals also propagate.
        """
        methodology_roots = tuple(
            uri for uri in evidence.artifact_uris if f"/{PANEL_METHODOLOGY_ROOT_CATEGORY}/" in uri
        )
        if len(methodology_roots) != 1 or len(evidence.artifact_uris) != 1:
            raise AuthoringError("alpha_research.panel_methodology_root_not_unique")
        root = verify_panel_methodology_graph(
            output_workspace=output_workspace,
            program=program,
            root_hash=methodology_roots[0].rsplit("/", 1)[-1],
        )
        if (
            (authority is not None and root.authority_hash != authority.authority_hash)
            or (root.common_formation_sessions or root.portfolio_common_sessions)
            != evidence.formation_sessions
            or root.desk_input_binding_hash != evidence.desk_input_binding_hash
        ):
            raise AuthoringError("alpha_research.panel_methodology_replay_mismatch")


__all__ = [
    "PANEL_METHODOLOGY_ROOT_CATEGORY",
    "PanelMethodologyEvidenceVerifier",
    "PanelMethodologyExecutionError",
    "PanelMethodologyExecutionInputs",
    "PanelMethodologyExecutionRoot",
    "PanelMethodologyOuterFold",
    "PanelPortfolioAxis",
    "PanelResearchMethodologyExecutor",
    "PanelScoreSurfaceArtifact",
    "PortfolioAxisAuthority",
    "ResolvedCandidateScoreSurface",
    "admit_fixed_panel_portfolio_authorities",
    "preflight_panel_portfolio_axis",
    "resolve_fixed_candidate_decision_folds",
    "resolve_fixed_candidate_score_surfaces",
    "resolve_fixed_panel_score_temporal_handoff",
    "verify_panel_methodology_graph",
]
