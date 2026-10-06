"""Durable generic evidence and actor-neutral decision for Alpha development."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any, Final, Literal, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from pydantic_core.core_schema import SerializerFunctionWrapHandler

from alphalattice.investment.alpha_research.calibration.stock_returns import (
    StockCalibrationEvidence,
)
from alphalattice.investment.alpha_research.scaling.evaluation import (
    CrossSectionalScaleComparisonEvidence,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash
from alphalattice.protocols.actor_execution import (
    ActorSubmissionBinding,
)

from .campaign import (
    INNER_DISCRIMINATION_FLOOR,
    AlphaConditionalMethodPlan,
    AlphaDevelopmentProgram,
    AlphaInnerSelectionRuleId,
    AlphaTrialPlan,
)

_INSTALLED_PLAYPEN_ROOT: Final = resolve_playpen_root(Path(__file__))


class AlphaCampaignDecisionAuthorityError(ValueError):
    """A Campaign decision failed deterministic Host authority validation."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class _PopulatedContract(_Contract):
    """Identity over what the artifact asserts, so a later member cannot move it.

    Deliberately shallow: nested members of this family run their own
    serializer, and no member here embeds a foreign sealed binding.
    """

    @model_serializer(mode="wrap")  # type: ignore[untyped-decorator]
    def _serialize_populated(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        serialized: dict[str, object] = handler(self)
        return {key: value for key, value in serialized.items() if value is not None}


class AlphaInnerTrialScore(_PopulatedContract):
    trial_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_id: str
    inner_validation_mse: float = Field(ge=0.0, allow_inf_nan=False)
    inner_validation_rank_ic: float | None = Field(default=None, ge=-1.0, le=1.0)
    inner_residual_excess_kurtosis: float | None = Field(default=None, allow_inf_nan=False)
    inner_mean_daily_rank_ic: float | None = Field(default=None, ge=-1.0, le=1.0)
    inner_mean_one_way_turnover: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    inner_net_daily_rank_ic: float | None = Field(default=None, allow_inf_nan=False)
    inner_discriminating_session_count: int | None = Field(default=None, ge=0)
    inner_constant_session_count: int | None = Field(default=None, ge=0)


class AlphaInnerSelectionRecord(_PopulatedContract):
    """One outer fold's complete chronological inner search and its choice."""

    kind: Literal["AlphaInnerSelectionRecord"] = "AlphaInnerSelectionRecord"
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    horizon_sessions: Literal[1, 5]
    fold_index: int = Field(ge=0)
    inner_training_row_count: int = Field(ge=40)
    inner_purge_row_count: int = Field(ge=0)
    inner_validation_row_count: int = Field(ge=20)
    inner_training_row_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    inner_validation_row_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    trial_scores: tuple[AlphaInnerTrialScore, ...] = Field(min_length=1)
    selected_method_id: str
    selected_trial_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_rule_id: AlphaInnerSelectionRuleId
    fit_call_count: int = Field(ge=1)
    predict_call_count: int = Field(ge=1)
    metric_call_count: int = Field(ge=1)
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        payload = {"kind": "AlphaInnerSelectionRecord", **values}
        identity = cls.model_construct(**payload, record_hash="0" * 64).model_dump(
            mode="json", exclude={"record_hash"}
        )
        return cls(**payload, record_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        hashes = tuple(value.trial_hash for value in self.trial_scores)
        if len(set(hashes)) != len(hashes) or self.selected_trial_hash not in set(hashes):
            raise ValueError("ALPHA_INNER_SELECTION_RECORD_SCORES_INVALID")
        if self.record_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"record_hash"})
        ):
            raise ValueError("ALPHA_INNER_SELECTION_RECORD_INVALID")
        return self


class AlphaDevelopmentTrialEvidence(_Contract):
    """One outer fold's single selected-method refit and honest prediction."""

    kind: Literal["AlphaDevelopmentTrialEvidence"] = "AlphaDevelopmentTrialEvidence"
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    horizon_sessions: Literal[1, 5]
    trial_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_index: int = Field(ge=0)
    inner_selection_record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimator_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_value_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_row_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_mse: float = Field(ge=0.0, allow_inf_nan=False)
    validation_rank_ic: float | None = Field(default=None, ge=-1.0, le=1.0)
    raw_economic_return_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    validation_residual_excess_kurtosis: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )
    fit_call_count: int = Field(ge=1)
    predict_call_count: int = Field(ge=1)
    metric_call_count: int = Field(ge=1)
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        payload = {"kind": "AlphaDevelopmentTrialEvidence", **values}
        identity = cls.model_construct(**payload, evidence_hash="0" * 64).model_dump(
            mode="json", exclude={"evidence_hash"}
        )
        return cls(**payload, evidence_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise ValueError("ALPHA_DEVELOPMENT_TRIAL_EVIDENCE_INVALID")
        return self


class AlphaDevelopmentMethodEvidence(_Contract):
    kind: Literal["AlphaDevelopmentMethodEvidence"] = "AlphaDevelopmentMethodEvidence"
    method_id: str
    horizon_sessions: Literal[1, 5]
    disposition: Literal["EVALUATED", "NOT_TRIGGERED"]
    trigger_reason: str | None = None
    inner_selection_record_hashes: tuple[str, ...]
    mean_inner_validation_mse: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    selected_outer_fold_count: int = Field(ge=0)
    outer_trial_evidence_hashes: tuple[str, ...]
    mean_outer_validation_mse: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    mean_outer_validation_rank_ic: float | None = Field(default=None, ge=-1.0, le=1.0)
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        payload = {"kind": "AlphaDevelopmentMethodEvidence", **values}
        identity = cls.model_construct(**payload, evidence_hash="0" * 64).model_dump(
            mode="json", exclude={"evidence_hash"}
        )
        return cls(**payload, evidence_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_method(self) -> Self:
        if self.disposition == "EVALUATED":
            if not self.inner_selection_record_hashes or self.mean_inner_validation_mse is None:
                raise ValueError("ALPHA_DEVELOPMENT_METHOD_EVIDENCE_INCOMPLETE")
        elif (
            self.inner_selection_record_hashes
            or self.outer_trial_evidence_hashes
            or self.selected_outer_fold_count
            or not self.trigger_reason
        ):
            raise ValueError("ALPHA_DEVELOPMENT_METHOD_TRIGGER_EVIDENCE_INVALID")
        if self.selected_outer_fold_count != len(self.outer_trial_evidence_hashes):
            raise ValueError("ALPHA_DEVELOPMENT_METHOD_EVIDENCE_FOLDS_INVALID")
        if self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise ValueError("ALPHA_DEVELOPMENT_METHOD_EVIDENCE_INVALID")
        return self


class AlphaRawEconomicEvaluationEvidence(_Contract):
    kind: Literal["AlphaRawEconomicEvaluationEvidence"] = "AlphaRawEconomicEvaluationEvidence"
    horizon_sessions: Literal[1, 5]
    trial_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    mean_raw_economic_return_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        payload = {"kind": "AlphaRawEconomicEvaluationEvidence", **values}
        identity = cls.model_construct(**payload, evidence_hash="0" * 64).model_dump(
            mode="json", exclude={"evidence_hash"}
        )
        return cls(**payload, evidence_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise ValueError("ALPHA_RAW_ECONOMIC_EVIDENCE_INVALID")
        return self


class AlphaDevelopmentDecisionDossier(_Contract):
    kind: Literal["AlphaDevelopmentDecisionDossier"] = "AlphaDevelopmentDecisionDossier"
    program: AlphaDevelopmentProgram
    inner_selection_records: tuple[AlphaInnerSelectionRecord, ...] = Field(min_length=2)
    trial_evidence: tuple[AlphaDevelopmentTrialEvidence, ...] = Field(min_length=2)
    method_evidence: tuple[AlphaDevelopmentMethodEvidence, ...] = Field(min_length=1)
    scale_forecast_hashes: tuple[str, ...] = Field(min_length=3)
    scale_comparison_evidence: tuple[
        CrossSectionalScaleComparisonEvidence,
        CrossSectionalScaleComparisonEvidence,
    ]
    scale_trigger_dispositions: dict[str, Literal["EVALUATED", "NOT_TRIGGERED"]]
    calibration_evidence: tuple[StockCalibrationEvidence, StockCalibrationEvidence]
    fast_slow_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_economic_evidence: tuple[
        AlphaRawEconomicEvaluationEvidence,
        AlphaRawEconomicEvaluationEvidence,
    ]
    fit_call_count: int = Field(ge=0)
    predict_call_count: int = Field(ge=0)
    metric_call_count: int = Field(ge=0)
    factor_numerical_call_count: int = Field(ge=0)
    limitations: tuple[str, ...] = Field(min_length=1)
    dossier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_dossier(self) -> Self:
        records = {
            (value.horizon_sessions, value.fold_index): value
            for value in self.inner_selection_records
        }
        outer = {(value.horizon_sessions, value.fold_index): value for value in self.trial_evidence}
        admitted_plan_hashes = {
            *(value.trial_hash for value in self.program.ordered_trials),
            *(value.plan_hash for value in self.program.conditional_method_plans),
        }
        if (
            len(records) != len(self.inner_selection_records)
            or len(outer) != len(self.trial_evidence)
            or set(records) != set(outer)
        ):
            raise ValueError("ALPHA_DEVELOPMENT_DOSSIER_FOLD_AXIS_INVALID")
        if any(
            value.program_hash != self.program.program_hash
            for value in (*self.inner_selection_records, *self.trial_evidence)
        ):
            raise ValueError("ALPHA_DEVELOPMENT_DOSSIER_PROGRAM_MISMATCH")
        for key, evidence in outer.items():
            record = records[key]
            if (
                evidence.trial_hash != record.selected_trial_hash
                or evidence.inner_selection_record_hash != record.record_hash
                or evidence.trial_hash not in admitted_plan_hashes
            ):
                raise ValueError("ALPHA_DEVELOPMENT_DOSSIER_SELECTION_MISMATCH")
        record_hashes = {value.record_hash for value in self.inner_selection_records}
        outer_hashes = {value.evidence_hash for value in self.trial_evidence}
        if any(
            child not in record_hashes
            for method in self.method_evidence
            for child in method.inner_selection_record_hashes
        ) or any(
            child not in outer_hashes
            for method in self.method_evidence
            for child in method.outer_trial_evidence_hashes
        ):
            raise ValueError("ALPHA_DEVELOPMENT_DOSSIER_CHILD_MISSING")
        if (
            self.fit_call_count
            != sum(value.fit_call_count for value in self.inner_selection_records)
            + sum(value.fit_call_count for value in self.trial_evidence)
            or self.predict_call_count
            != sum(value.predict_call_count for value in self.inner_selection_records)
            + sum(value.predict_call_count for value in self.trial_evidence)
            or self.metric_call_count
            != sum(value.metric_call_count for value in self.inner_selection_records)
            + sum(value.metric_call_count for value in self.trial_evidence)
            or tuple(value.horizon_sessions for value in self.scale_comparison_evidence) != (1, 5)
            or tuple(value.horizon_sessions for value in self.calibration_evidence) != (1, 5)
            or tuple(value.horizon_sessions for value in self.raw_economic_evidence) != (1, 5)
            or self.dossier_hash != canonical_hash(_alpha_dossier_identity(self))
        ):
            raise ValueError("ALPHA_DEVELOPMENT_DOSSIER_INVALID")
        return self


def _alpha_dossier_identity(dossier: AlphaDevelopmentDecisionDossier) -> dict[str, Any]:
    """Preserve frozen dossiers written before methodology fields existed."""

    identity = dossier.model_dump(mode="json", exclude={"dossier_hash"})
    program = cast(dict[str, Any], identity["program"])
    for field in ("feature_preprocessing_method_id", "target_method_id"):
        if program.get(field) is None:
            program.pop(field, None)
    return cast(dict[str, Any], identity)


class AlphaCampaignDecisionSubmission(_Contract):
    """Typed Alpha-domain submission; carries selections, never authority."""

    kind: Literal["AlphaCampaignDecisionSubmission"] = "AlphaCampaignDecisionSubmission"
    dossier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_method_ids: tuple[str, ...]
    selected_scale_method_ids: tuple[str, ...]
    selected_blend_recipe_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    rationale: str = Field(min_length=1)
    submission_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        payload = {"kind": "AlphaCampaignDecisionSubmission", **values}
        return cls(**payload, submission_hash=canonical_hash(payload))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.submission_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"submission_hash"})
        ):
            raise ValueError("ALPHA_CAMPAIGN_DECISION_SUBMISSION_INVALID")
        return self


class AlphaCampaignDecisionReceipt(_Contract):
    """Host-sealed development decision with shared actor provenance.

    The actor is recorded through ``ActorSubmissionBinding`` -- Agent execution
    evidence is admissible exactly for ``INSTALLED_AGENT`` -- and the decision
    policy identity is derived from the installed Host source, never named by
    any actor.
    """

    kind: Literal["AlphaCampaignDecisionReceipt"] = "AlphaCampaignDecisionReceipt"
    submission: AlphaCampaignDecisionSubmission
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    actor_submission: ActorSubmissionBinding
    decision_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: Literal["DEVELOPMENT_SELECTED", "DEVELOPMENT_NOT_SELECTED"]
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.actor_submission.submission_hash != self.submission.submission_hash:
            raise ValueError("ALPHA_CAMPAIGN_DECISION_ACTOR_BINDING_MISMATCH")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise ValueError("ALPHA_CAMPAIGN_DECISION_RECEIPT_INVALID")
        return self


class AlphaRecursiveReplayReceipt(_Contract):
    kind: Literal["AlphaRecursiveReplayReceipt"] = "AlphaRecursiveReplayReceipt"
    dossier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified_child_count: int = Field(ge=1)
    disposition: Literal["REUSED_EXACT"] = "REUSED_EXACT"
    fit_call_count: Literal[0] = 0
    predict_call_count: Literal[0] = 0
    metric_call_count: Literal[0] = 0
    factor_numerical_call_count: Literal[0] = 0
    pointer_mutation_count: Literal[0] = 0
    replay_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.replay_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"replay_hash"})
        ):
            raise ValueError("ALPHA_RECURSIVE_REPLAY_RECEIPT_INVALID")
        return self


CAMPAIGN_DECISION_POLICY_ROLE = "alpha_research.campaign_decision_policy"
"""The role this policy is recorded under in the identity successors."""


@cache
def _policy_hash_for(root: Path) -> str:
    return cast(
        str,
        canonical_hash(
            {
                "sources": source_rule_closure_hash(
                    root=root,
                    tracked_paths=(
                        "src/alphalattice/investment/alpha_research/experiments/campaign.py",
                        (
                            "src/alphalattice/investment/alpha_research/experiments/"
                            "campaign_evidence.py"
                        ),
                        "src/alphalattice/protocols/actor_execution/contracts.py",
                    ),
                    semantic_owner="alpha_research.campaign_decision",
                    numerical_role="HOST_DECISION_POLICY",
                ),
            }
        ),
    )


def alpha_campaign_decision_policy_hash(playpen_root: Path | None = None) -> str:
    """The installed Host decision policy identity; no caller can name it.

    Measured from the installed module tree exactly as the Factor curation
    policy is. Actor identity, profile, or model choice can never reach it.
    """

    return _policy_hash_for(
        _INSTALLED_PLAYPEN_ROOT if playpen_root is None else Path(playpen_root).resolve()
    )


def _plan_index(
    program: AlphaDevelopmentProgram,
) -> dict[str, AlphaTrialPlan | AlphaConditionalMethodPlan]:
    return {
        **{value.trial_hash: value for value in program.ordered_trials},
        **{value.plan_hash: value for value in program.conditional_method_plans},
    }


def _admissible_net_rank_ic(value: AlphaInnerTrialScore) -> float | None:
    """The successor rule's score, or ``None`` where the trial cannot be judged.

    A configuration that produced a constant score on more than a fifth of the
    scored sessions is refused outright: the turnover penalty rewards exactly
    that degeneracy, so the floor is what stops a shrunk-to-nothing fit winning
    on cheapness.
    """

    admitted = value.inner_discriminating_session_count
    constant = value.inner_constant_session_count
    if value.inner_net_daily_rank_ic is None or admitted is None or constant is None:
        return None
    if admitted < INNER_DISCRIMINATION_FLOOR * constant:
        return None
    return value.inner_net_daily_rank_ic


def _lightgbm_configuration_selection(
    program: AlphaDevelopmentProgram,
    scores: Sequence[AlphaInnerTrialScore],
    selection_rule_id: AlphaInnerSelectionRuleId,
) -> tuple[float, str]:
    """One-standard-error-then-complexity over configuration seed means.

    Seeds are averaged before anything is compared, so the search never picks a
    configuration for having drawn a lucky bin construction. The tolerance runs
    below the best mean under the loss rule and above it under the ordering
    rule; the complexity vector is shared, because "simpler and more strongly
    regularized" means the same thing whichever score is being tolerated.
    """

    maximize = selection_rule_id == "MAXIMUM_INNER_NET_DAILY_RANK_IC_THEN_METHOD_ID"
    plans = {value.trial_hash: value for value in program.ordered_trials}
    grouped: dict[tuple[object, ...], list[float]] = {}
    candidates: dict[tuple[object, ...], dict[int, str]] = {}
    complexity: dict[tuple[object, ...], tuple[object, ...]] = {}
    for score in scores:
        observation = _admissible_net_rank_ic(score) if maximize else score.inner_validation_mse
        if observation is None:
            continue
        parameters = plans[score.trial_hash].recipe.parameters
        key = (
            parameters["num_leaves"],
            parameters["learning_rate"],
            parameters["max_depth"],
            parameters["min_child_samples"],
            parameters.get("lambda_l1", 0.0),
            parameters.get("lambda_l2", 0.0),
            parameters.get("feature_fraction", 1.0),
            parameters.get("bagging_fraction", 1.0),
            parameters.get("bagging_freq", 0),
        )
        grouped.setdefault(key, []).append(observation)
        candidates.setdefault(key, {})[int(parameters["seed"])] = score.trial_hash
        complexity[key] = (
            parameters["num_leaves"],
            parameters["max_depth"],
            -int(parameters["min_child_samples"]),
            -float(parameters.get("lambda_l1", 0.0)) - float(parameters.get("lambda_l2", 0.0)),
            float(parameters.get("feature_fraction", 1.0)),
            float(parameters.get("bagging_fraction", 1.0)),
            parameters["learning_rate"],
        )
    if not grouped:
        raise ValueError("ALPHA_INNER_SELECTION_NO_DISCRIMINATING_CONFIGURATION")
    if any(not candidates.get(key) for key in grouped):
        raise ValueError("ALPHA_INNER_SELECTION_LIGHTGBM_SCORES_INCOMPLETE")
    # The lowest admitted seed represents its configuration. Naming one fixed
    # seed would leave a group unrepresented whenever that seed is the member
    # the discrimination floor refused, which is exactly when the group matters.
    representative = {key: value[min(value)] for key, value in candidates.items()}
    summaries = {
        key: (
            float(np.mean(values)),
            float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        )
        for key, values in grouped.items()
    }
    if maximize:
        best_key = max(summaries, key=lambda key: (summaries[key][0], complexity[key]))
        tolerated = summaries[best_key][0] - summaries[best_key][1]
        admitted = (key for key, summary in summaries.items() if summary[0] >= tolerated)
    else:
        best_key = min(summaries, key=lambda key: (summaries[key][0], complexity[key]))
        tolerated = summaries[best_key][0] + summaries[best_key][1]
        admitted = (key for key, summary in summaries.items() if summary[0] <= tolerated)
    selected_key = min(admitted, key=lambda key: complexity[key])
    return summaries[selected_key][0], representative[selected_key]


def derive_alpha_inner_selection(
    *,
    program: AlphaDevelopmentProgram,
    trial_scores: Sequence[AlphaInnerTrialScore],
    selection_rule_id: AlphaInnerSelectionRuleId,
) -> tuple[str, str]:
    """Re-derivable per-fold selection over inner evidence only.

    Under the loss rule the best configuration per method is the minimum inner
    MSE and the minimum wins across methods; under the ordering rule both
    directions flip to the maximum turnover-net daily rank IC. LightGBM uses the
    registered one-standard-error rule over its seed means either way, and the
    method id is the deterministic tie-break in both. Nothing here reads an
    outer validation value.
    """

    maximize = selection_rule_id == "MAXIMUM_INNER_NET_DAILY_RANK_IC_THEN_METHOD_ID"
    plans = _plan_index(program)
    if len({value.trial_hash for value in trial_scores}) != len(trial_scores) or any(
        value.trial_hash not in plans or plans[value.trial_hash].method_id != value.method_id
        for value in trial_scores
    ):
        raise ValueError("ALPHA_INNER_SELECTION_SCORES_INVALID")
    by_method: dict[str, list[AlphaInnerTrialScore]] = {}
    for value in trial_scores:
        by_method.setdefault(value.method_id, []).append(value)
    method_best: dict[str, tuple[float, str]] = {}
    for method_id, scores in by_method.items():
        if method_id in {"LIGHTGBM", "LIGHTGBM_REGULARIZED"}:
            try:
                method_best[method_id] = _lightgbm_configuration_selection(
                    program, scores, selection_rule_id
                )
            except ValueError as error:
                if str(error) != "ALPHA_INNER_SELECTION_NO_DISCRIMINATING_CONFIGURATION":
                    raise
            continue
        if maximize:
            admissible = [
                (value, _admissible_net_rank_ic(value))
                for value in scores
                if _admissible_net_rank_ic(value) is not None
            ]
            if not admissible:
                continue
            chosen = min(admissible, key=lambda item: (-cast(float, item[1]), item[0].trial_hash))
            method_best[method_id] = (cast(float, chosen[1]), chosen[0].trial_hash)
            continue
        best = min(scores, key=lambda value: (value.inner_validation_mse, value.trial_hash))
        method_best[method_id] = (best.inner_validation_mse, best.trial_hash)
    if not method_best:
        raise ValueError("ALPHA_INNER_SELECTION_NO_DISCRIMINATING_CONFIGURATION")
    if maximize:
        selected_method = min(method_best, key=lambda key: (-method_best[key][0], key))
    else:
        selected_method = min(method_best, key=lambda key: (method_best[key][0], key))
    return selected_method, method_best[selected_method][1]


def _derive_decision_membership(
    dossier: AlphaDevelopmentDecisionDossier,
) -> tuple[frozenset[str], frozenset[str]]:
    """The Host-derived admissible selections: fold-winning methods and scales."""

    winners = frozenset(
        value.method_id
        for value in dossier.method_evidence
        if value.disposition == "EVALUATED" and value.selected_outer_fold_count > 0
    )
    scales = frozenset(
        method
        for method, disposition in dossier.scale_trigger_dispositions.items()
        if disposition == "EVALUATED"
    )
    return winners, scales


def verify_alpha_campaign_decision(
    *,
    decision: AlphaCampaignDecisionReceipt,
    dossier: AlphaDevelopmentDecisionDossier,
) -> None:
    """Re-derive a persisted decision from the dossier and the installed policy."""

    if (
        decision.submission.dossier_hash != dossier.dossier_hash
        or decision.program_hash != dossier.program.program_hash
    ):
        raise AlphaCampaignDecisionAuthorityError("ALPHA_CAMPAIGN_DECISION_DOSSIER_MISMATCH")
    winners, scales = _derive_decision_membership(dossier)
    if not set(decision.submission.selected_method_ids).issubset(winners) or not set(
        decision.submission.selected_scale_method_ids
    ).issubset(scales):
        raise AlphaCampaignDecisionAuthorityError("ALPHA_CAMPAIGN_DECISION_SELECTION_NOT_EVALUATED")
    expected_disposition = (
        "DEVELOPMENT_SELECTED"
        if decision.submission.selected_method_ids
        else "DEVELOPMENT_NOT_SELECTED"
    )
    if decision.disposition != expected_disposition:
        raise AlphaCampaignDecisionAuthorityError("ALPHA_CAMPAIGN_DECISION_DISPOSITION_INVALID")
    if not is_current(
        CAMPAIGN_DECISION_POLICY_ROLE,
        decision.decision_policy_hash,
        alpha_campaign_decision_policy_hash(),
    ):
        raise AlphaCampaignDecisionAuthorityError("ALPHA_CAMPAIGN_DECISION_POLICY_NOT_INSTALLED")
    if decision.actor_submission.submission_hash != decision.submission.submission_hash:
        raise AlphaCampaignDecisionAuthorityError("ALPHA_CAMPAIGN_DECISION_ACTOR_MISMATCH")


__all__ = [
    "AlphaCampaignDecisionAuthorityError",
    "AlphaCampaignDecisionReceipt",
    "AlphaCampaignDecisionSubmission",
    "AlphaDevelopmentDecisionDossier",
    "AlphaDevelopmentMethodEvidence",
    "AlphaDevelopmentTrialEvidence",
    "AlphaInnerSelectionRecord",
    "AlphaInnerTrialScore",
    "AlphaRawEconomicEvaluationEvidence",
    "AlphaRecursiveReplayReceipt",
    "alpha_campaign_decision_policy_hash",
    "derive_alpha_inner_selection",
    "verify_alpha_campaign_decision",
]
