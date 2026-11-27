"""Typed authority and evidence for deterministic Alpha model experiments."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.capabilities.domain_contracts.sealing import seal_model_from_dump
from alphalattice.kernel.identity import BLAKE3_PATTERN, blake3_hex, is_blake3

from ..evaluation.contracts import AlphaMetricPolicy as _AlphaMetricPolicy
from ..targets.execution_outcome import AlphaTargetLane, AlphaTargetPolicy
from .mandate import AlphaResearchModelRecipe

Hash = str


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


def seal_contract[ContractT: BaseModel](
    model: type[ContractT], values: dict[str, Any], hash_field: str
) -> ContractT:
    return seal_model_from_dump(model, values, field=hash_field)


def _validate_hash(model: BaseModel, field: str) -> None:
    expected = canonical_hash(model.model_dump(mode="json", exclude={field}))
    if getattr(model, field) != expected:
        raise ValueError(f"{model.__class__.__name__} identity is invalid")


def _validate_hash_compatible(
    model: BaseModel, field: str, *, optional_fields: tuple[str, ...]
) -> None:
    identity = model.model_dump(mode="json", exclude={field})
    claimed = getattr(model, field)
    if claimed == canonical_hash(identity):
        return
    legacy = dict(identity)
    for name in optional_fields:
        if legacy.get(name) is None:
            legacy.pop(name, None)
            if claimed == canonical_hash(legacy):
                return
    raise ValueError(f"{model.__class__.__name__} identity is invalid")


class RidgeModelSpec(_Contract):
    family: Literal["ridge"] = "ridge"
    alpha: float = Field(ge=0.1, le=100.0, allow_inf_nan=False)
    target_lane: AlphaTargetLane | None = None


class LassoModelSpec(_Contract):
    family: Literal["lasso"] = "lasso"
    alpha_max_multiplier: float = Field(ge=0.01, le=1.0, allow_inf_nan=False)
    target_lane: AlphaTargetLane | None = None


class ElasticNetModelSpec(_Contract):
    family: Literal["elastic_net"] = "elastic_net"
    alpha_max_multiplier: float = Field(ge=0.01, le=1.0, allow_inf_nan=False)
    l1_ratio: float = Field(ge=0.25, le=0.75, allow_inf_nan=False)
    target_lane: AlphaTargetLane | None = None


type LegacyModelSpec = Annotated[
    RidgeModelSpec | LassoModelSpec | ElasticNetModelSpec,
    Field(discriminator="family"),
]
MODEL_SPEC_ADAPTER: TypeAdapter[LegacyModelSpec] = TypeAdapter(LegacyModelSpec)


class ResolvedRegularizedLinearSpec(_Contract):
    """Frozen readback codec for pre-Mandate Goal Research recipes."""

    kind: Literal["ResolvedRegularizedLinearSpec"] = "ResolvedRegularizedLinearSpec"
    family: Literal["ridge", "lasso", "elastic_net"]
    alpha: float | None = Field(default=None, ge=0.1, le=100.0, allow_inf_nan=False)
    alpha_max_multiplier: float | None = Field(default=None, ge=0.01, le=1.0, allow_inf_nan=False)
    l1_ratio: float | None = Field(default=None, ge=0.25, le=0.75, allow_inf_nan=False)
    target_lane: AlphaTargetLane | None = None
    spec_hash: Hash = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_spec(self) -> Self:
        if self.family == "ridge":
            valid = (
                self.alpha is not None
                and self.alpha_max_multiplier is None
                and self.l1_ratio is None
            )
        elif self.family == "lasso":
            valid = (
                self.alpha is None
                and self.alpha_max_multiplier is not None
                and self.l1_ratio is None
            )
        else:
            valid = (
                self.alpha is None
                and self.alpha_max_multiplier is not None
                and self.l1_ratio is not None
            )
        if not valid:
            raise ValueError("Alpha regularized-linear spec shape is invalid")
        _validate_hash_compatible(self, "spec_hash", optional_fields=("target_lane",))
        return self


def resolve_model_spec(
    spec: LegacyModelSpec | dict[str, object],
) -> ResolvedRegularizedLinearSpec:
    admitted = MODEL_SPEC_ADAPTER.validate_python(spec)
    values: dict[str, object] = {
        "kind": "ResolvedRegularizedLinearSpec",
        "family": admitted.family,
        "alpha": admitted.alpha if isinstance(admitted, RidgeModelSpec) else None,
        "alpha_max_multiplier": (
            admitted.alpha_max_multiplier
            if isinstance(admitted, LassoModelSpec | ElasticNetModelSpec)
            else None
        ),
        "l1_ratio": admitted.l1_ratio if isinstance(admitted, ElasticNetModelSpec) else None,
        "target_lane": admitted.target_lane,
    }
    if admitted.target_lane is None:
        identity = dict(values)
        identity.pop("target_lane")
        return cast(
            ResolvedRegularizedLinearSpec,
            ResolvedRegularizedLinearSpec.model_validate(
                {**values, "spec_hash": canonical_hash(identity)}
            ),
        )
    return seal_contract(ResolvedRegularizedLinearSpec, values, "spec_hash")


type StoredModelSpec = Annotated[
    ResolvedRegularizedLinearSpec | AlphaResearchModelRecipe,
    Field(discriminator="kind"),
]


def candidate_id_for_spec(spec: StoredModelSpec) -> str:
    """Derive a candidate name from its model family and the spec digest."""
    prefix = "alpha-candidate" if isinstance(spec, AlphaResearchModelRecipe) else "agent-linear"
    return f"{prefix}-{blake3_hex(spec.spec_hash)[:16]}"


class AlphaResearchProgram(_Contract):
    """Bind the research program to its foundation, target lanes and evaluation policies."""

    kind: Literal["AlphaResearchProgram"] = "AlphaResearchProgram"
    foundation_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    logical_panel_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    logical_semantic_index_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    causal_outcome_snapshot_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    pm_plan_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    ordered_listing_ids_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    ordered_factor_ids_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    split_policy_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    metric_policy_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    package_identity_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    stability_policy_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    goal_criteria_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    user_authorization_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    research_goal_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    model_mandate_hash: Hash | None = Field(default=None, pattern=BLAKE3_PATTERN)
    model_catalog_hash: Hash | None = Field(default=None, pattern=BLAKE3_PATTERN)
    target_policy_hashes: tuple[Hash, ...] | None = None
    sector_ema_policy_hash: Hash | None = Field(default=None, pattern=BLAKE3_PATTERN)
    sector_context_policy_hash: Hash | None = Field(default=None, pattern=BLAKE3_PATTERN)
    program_hash: Hash = Field(pattern=BLAKE3_PATTERN)

    @property
    def ordered_feature_ids_hash(self) -> str:
        """Expose the generic feature-axis name without changing frozen payloads."""

        return self.ordered_factor_ids_hash

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_program(self) -> Self:
        """Verify that the program's target lanes, comparison policies and identity agree."""

        target_count = len(self.target_policy_hashes or ())
        if (
            (
                target_count == 0
                and (
                    self.sector_ema_policy_hash is not None
                    or self.sector_context_policy_hash is not None
                )
            )
            or (
                target_count == 2
                and (
                    self.sector_ema_policy_hash is None
                    or self.sector_context_policy_hash is not None
                )
            )
            or (target_count == 4 and self.sector_context_policy_hash is None)
        ):
            raise ValueError("Alpha research target authority is incomplete")
        # One lane is a qualification of a development question's lane (GR3); two and four are
        # the goal loop's lane sets its Programs sealed.
        if self.target_policy_hashes is not None and (
            len(self.target_policy_hashes) not in {1, 2, 4}
            or len(set(self.target_policy_hashes)) != len(self.target_policy_hashes)
            or any(not is_blake3(value) for value in self.target_policy_hashes)
        ):
            raise ValueError("Alpha research target policy axis is invalid")
        _validate_hash_compatible(
            self,
            "program_hash",
            optional_fields=(
                "target_policy_hashes",
                "sector_ema_policy_hash",
                "sector_context_policy_hash",
                "model_mandate_hash",
                "model_catalog_hash",
            ),
        )
        return self


class AlphaExperimentBatch(_Contract):
    """Bind an ordered batch of candidate specifications to its program and predecessor."""

    kind: Literal["AlphaExperimentBatch"] = "AlphaExperimentBatch"
    program_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    batch_index: int = Field(ge=1)
    specs: tuple[StoredModelSpec, ...] = Field(min_length=1)
    predecessor_batch_hash: Hash | None = Field(default=None, pattern=BLAKE3_PATTERN)
    batch_hash: Hash = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_batch(self) -> Self:
        if (self.batch_index == 1) != (self.predecessor_batch_hash is None):
            raise ValueError("Alpha research batch predecessor boundary is invalid")
        hashes = tuple(item.spec_hash for item in self.specs)
        if len(set(hashes)) != len(hashes):
            raise ValueError("Alpha research batch repeats a model spec")
        _validate_hash(self, "batch_hash")
        return self


class AlphaExperimentCandidateResult(_Contract):
    """Record a candidate's measured fold outcomes and referenced numerical artifacts."""

    kind: Literal["AlphaExperimentCandidateResult"] = "AlphaExperimentCandidateResult"
    candidate_id: str = Field(pattern=r"^(?:agent-linear|alpha-candidate)-[0-9a-f]{16}$")
    spec_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    development_report_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    inference_evidence_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    numerical_result_hashes: tuple[Hash, ...] = Field(min_length=1)
    estimator_state_hashes: tuple[Hash, ...] = Field(min_length=1)
    pooled_oos_r2: float | None = Field(default=None, allow_inf_nan=False)
    mean_rank_ic: float | None = Field(default=None, allow_inf_nan=False)
    mean_gross_decile_spread: float | None = Field(default=None, allow_inf_nan=False)
    fold_coverage_mean: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    target_lane: AlphaTargetLane | None = None
    development_surface_hash: Hash | None = Field(default=None, pattern=BLAKE3_PATTERN)
    development_surface_binding_hash: Hash | None = Field(default=None, pattern=BLAKE3_PATTERN)
    fold_surface_hashes: tuple[Hash, ...] | None = None
    status: Literal["DEVELOPMENT_EVALUATED", "DEVELOPMENT_FAILED"]
    failure_codes: tuple[str, ...] = ()
    result_hash: Hash = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_result(self) -> Self:
        if self.status == "DEVELOPMENT_EVALUATED" and self.failure_codes:
            raise ValueError("successful Alpha experiment result has failures")
        if self.status == "DEVELOPMENT_FAILED" and not self.failure_codes:
            raise ValueError("failed Alpha experiment result lacks failure evidence")
        lane_fields = (
            self.development_surface_hash,
            self.development_surface_binding_hash,
            self.fold_surface_hashes,
        )
        if self.target_lane is not None and any(value is None for value in lane_fields):
            raise ValueError("Alpha target-lane result lacks its development surface")
        _validate_hash_compatible(
            self,
            "result_hash",
            optional_fields=(
                "target_lane",
                "development_surface_hash",
                "development_surface_binding_hash",
                "fold_surface_hashes",
            ),
        )
        return self


class AlphaExperimentBatchResult(_Contract):
    """Collect candidate results and work counts for one development surface and batch."""

    kind: Literal["AlphaExperimentBatchResult"] = "AlphaExperimentBatchResult"
    program_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    batch_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    development_surface_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    development_surface_binding_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    fold_surface_hashes: tuple[Hash, ...] = Field(min_length=1)
    candidates: tuple[AlphaExperimentCandidateResult, ...] = Field(min_length=1)
    fit_call_count: int = Field(ge=0)
    predict_call_count: int = Field(ge=0)
    metric_call_count: int = Field(ge=0)
    result_hash: Hash = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_batch_result(self) -> Self:
        candidate_ids = tuple(value.candidate_id for value in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Alpha experiment batch result repeats a candidate")
        _validate_hash(self, "result_hash")
        return self


class AlphaDevelopmentFitEvidence(_Contract):
    """Model-neutral binding from one fold operation to estimator evidence."""

    kind: Literal["AlphaDevelopmentFitEvidence"] = "AlphaDevelopmentFitEvidence"
    execution_binding_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    candidate_id: str
    fold_index: int = Field(ge=0)
    adapter_id: str = Field(min_length=1, max_length=96)
    recipe_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    training_binding_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    fit_plan_hash: Hash | None = Field(
        default=None,
        pattern=BLAKE3_PATTERN,
        exclude_if=lambda value: value is None,
    )
    numerical_binding_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    numerical_environment_hash: Hash | None = Field(
        default=None,
        pattern=BLAKE3_PATTERN,
        exclude_if=lambda value: value is None,
    )
    estimator_content_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    fit_provenance_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    state_projection_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    score_evidence_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    fit_call_count: int = Field(ge=0)
    predict_call_count: int = Field(ge=0)
    evidence_hash: Hash = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        _validate_hash_compatible(
            self,
            "evidence_hash",
            optional_fields=("fit_plan_hash", "numerical_environment_hash"),
        )
        return self


class AlphaCandidateRole(StrEnum):
    BENCHMARK = "BENCHMARK"
    REGULARIZED_ALPHA = "REGULARIZED_ALPHA"
    MODEL_ALPHA = "MODEL_ALPHA"
    DIAGNOSTIC_ONLY_BOUNDED = "DIAGNOSTIC_ONLY_BOUNDED"
    NOT_ADMITTED = "NOT_ADMITTED"


class AlphaCandidateStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class AlphaFailureStage(StrEnum):
    ADMISSION = "ADMISSION"
    ARRAY_CONSTRUCTION = "ARRAY_CONSTRUCTION"
    FIT = "FIT"
    PREDICT = "PREDICT"
    METRICS = "METRICS"
    PUBLICATION = "PUBLICATION"
    READBACK = "READBACK"
    REVIEW = "REVIEW"


class AlphaExperimentCard(_Contract):
    """Describe one candidate's feature shape, model family and training settings."""

    kind: Literal["AlphaExperimentCard"] = "AlphaExperimentCard"
    candidate_id: str = Field(min_length=1, max_length=120)
    source_card_hash: str = Field(pattern=BLAKE3_PATTERN)
    family_id: Literal[
        "zero_forecast",
        "historical_mean",
        "quantile_regression",
        "ols_3",
        "ridge",
        "lasso",
        "elastic_net",
        "lightgbm",
    ]
    role: AlphaCandidateRole
    feature_shape: Literal["NONE", "FROZEN_ORDERED"]
    factor_count: int = Field(ge=0, le=55)
    alpha: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    alpha_multiplier: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    l1_ratio: float | None = Field(default=None, gt=0, lt=1, allow_inf_nan=False)
    quantile_tau: float | None = Field(default=None, gt=0, lt=1, allow_inf_nan=False)
    num_leaves: int | None = Field(default=None, ge=2, le=256)
    learning_rate: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    maximum_iterations: int | None = Field(default=None, ge=1)
    early_stopping_rounds: int | None = Field(default=None, ge=1)
    required_feature_ids: tuple[str, ...] = ()
    admission_status: Literal["ADMITTED", "NOT_ADMITTED_MISSING_FACTORS"] = "ADMITTED"
    selection_eligible: bool | None = None
    fit_intercept: bool | None = None
    solver: Literal["svd"] | None = None
    tolerance: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    deterministic_seed: int | None = None
    secondary_feature_preprocessing: Literal[False] = False
    package_identity: str = Field(min_length=1, max_length=160)
    card_hash: str = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_card(self) -> AlphaExperimentCard:
        learned = {"ridge", "lasso", "elastic_net", "lightgbm"}
        parameter_values = (
            self.alpha_multiplier,
            self.l1_ratio,
            self.quantile_tau,
            self.num_leaves,
            self.learning_rate,
            self.maximum_iterations,
            self.early_stopping_rounds,
        )
        if self.family_id in {"zero_forecast", "historical_mean"}:
            expected_id = f"benchmark.{self.family_id.replace('_', '-')}"
            if self.family_id == "historical_mean":
                expected_id = "benchmark.historical-mean"
            if (
                self.candidate_id != expected_id
                or self.role is not AlphaCandidateRole.BENCHMARK
                or self.feature_shape != "NONE"
                or self.factor_count != 0
                or self.alpha is not None
                or any(value is not None for value in parameter_values)
                or self.fit_intercept is not None
                or self.selection_eligible is True
                or self.package_identity != "alpha-research-deterministic"
            ):
                raise ValueError("registered benchmark card parameters are invalid")
        elif self.family_id == "quantile_regression":
            expected = {
                "diagnostic.quantile.tau-0.1": 0.1,
                "diagnostic.quantile.tau-0.5": 0.5,
                "diagnostic.quantile.tau-0.9": 0.9,
            }.get(self.candidate_id)
            if (
                expected is None
                or self.quantile_tau != expected
                or self.role is not AlphaCandidateRole.DIAGNOSTIC_ONLY_BOUNDED
                or self.feature_shape != "FROZEN_ORDERED"
                or self.factor_count < 1
                or self.maximum_iterations != 10000
                or self.selection_eligible
                or self.package_identity != "statsmodels==0.14.6"
            ):
                raise ValueError("registered quantile diagnostic card is invalid")
        elif self.family_id == "ols_3":
            if (
                self.candidate_id != "model.ols-3"
                or self.required_feature_ids != ("mom_252_21", "rev_21", "vol_63")
                or self.admission_status != "NOT_ADMITTED_MISSING_FACTORS"
                or self.role is not AlphaCandidateRole.NOT_ADMITTED
                or self.selection_eligible
                or self.package_identity != "statsmodels==0.14.6"
            ):
                raise ValueError("registered OLS-3 non-admission is invalid")
        elif self.family_id in learned:
            if (
                self.role is not AlphaCandidateRole.REGULARIZED_ALPHA
                or self.feature_shape != "FROZEN_ORDERED"
                or self.factor_count < 1
                or self.admission_status != "ADMITTED"
                or self.selection_eligible is False
            ):
                raise ValueError("registered selectable Alpha card boundary is invalid")
            if self.family_id == "ridge" and (
                self.alpha not in {0.1, 1.0, 10.0, 100.0}
                or self.solver != "svd"
                or self.tolerance != 1e-8
                or self.fit_intercept is not True
                or self.package_identity != "scikit-learn==1.9.0"
            ):
                raise ValueError("registered Ridge card parameters are invalid")
            if self.family_id in {"lasso", "elastic_net"} and (
                self.alpha_multiplier not in {0.01, 0.03, 0.1, 0.3, 1.0}
                or self.maximum_iterations != 10000
                or self.tolerance != 1e-8
                or self.fit_intercept is not True
                or (self.family_id == "lasso" and self.l1_ratio is not None)
                or (self.family_id == "elastic_net" and self.l1_ratio not in {0.25, 0.5, 0.75})
                or self.package_identity != "scikit-learn==1.9.0"
            ):
                raise ValueError("registered sparse-linear card parameters are invalid")
            if self.family_id == "lightgbm" and (
                self.num_leaves not in {15, 31}
                or self.learning_rate not in {0.03, 0.05, 0.1}
                or self.maximum_iterations != 500
                or self.early_stopping_rounds != 50
                or self.fit_intercept is not None
                or self.package_identity != "lightgbm==4.7.0"
            ):
                raise ValueError("registered LightGBM card parameters are invalid")
        if self.card_hash != canonical_hash(self.model_dump(mode="json", exclude={"card_hash"})):
            raise ValueError("Alpha experiment card hash is invalid")
        return self


class AlphaSplitPolicy(_Contract):
    """Declare the training, purge, validation and embargo windows for an Alpha evaluation."""

    kind: Literal["AlphaSplitPolicy"] = "AlphaSplitPolicy"
    frequency: Literal["DAILY"] = "DAILY"
    mode: Literal["ROLLING"] = "ROLLING"
    train_sessions: Literal[756] = 756
    purge_sessions: Literal[1] = 1
    validation_sessions: Literal[252] = 252
    step_sessions: Literal[252] = 252
    embargo_sessions: Literal[0] = 0
    sealed_holdout_sessions: Literal[252] = 252
    minimum_folds: Literal[3] = 3
    expected_complete_folds: Literal[5] = 5
    split_owner: Literal["alphalattice.kernel.validation.splitting.build_research_split"] = (
        "alphalattice.kernel.validation.splitting.build_research_split"
    )
    policy_hash: str = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaSplitPolicy:
        if self.policy_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"policy_hash"})
        ):
            raise ValueError("Alpha split policy hash is invalid")
        return self


class AlphaDevelopmentSplitPolicy(_Contract):
    """Exact split geometry sealed for one standalone development program."""

    kind: Literal["AlphaDevelopmentSplitPolicy"] = "AlphaDevelopmentSplitPolicy"
    frequency: Literal["DAILY"] = "DAILY"
    mode: Literal["EXPANDING", "ROLLING"]
    train_sessions: int = Field(ge=1)
    purge_sessions: int = Field(ge=0)
    validation_sessions: int = Field(ge=1)
    step_sessions: int = Field(ge=1)
    embargo_sessions: int = Field(ge=0)
    sealed_holdout_sessions: int = Field(ge=0)
    minimum_folds: int = Field(ge=1)
    expected_complete_folds: int = Field(ge=1)
    split_owner: Literal["alphalattice.kernel.validation.splitting.build_research_split"] = (
        "alphalattice.kernel.validation.splitting.build_research_split"
    )
    policy_hash: Hash = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.minimum_folds > self.expected_complete_folds or self.policy_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"policy_hash"})
        ):
            raise ValueError("ALPHA_DEVELOPMENT_SPLIT_POLICY_IDENTITY_INVALID")
        return self


class AlphaPackageIdentity(_Contract):
    """Identify the numerical library versions and float representation used by a candidate."""

    numpy: Literal["numpy==2.5.1"] = "numpy==2.5.1"
    pyarrow: Literal["pyarrow==25.0.0"] = "pyarrow==25.0.0"
    scikit_learn: Literal["scikit-learn==1.9.0"] = "scikit-learn==1.9.0"
    float_identity: Literal["IEEE754_FLOAT64"] = "IEEE754_FLOAT64"
    package_hash: str = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaPackageIdentity:
        if self.package_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"package_hash"})
        ):
            raise ValueError("Alpha package identity hash is invalid")
        return self


class AlphaDevelopmentProgram(_Contract):
    """Standalone deterministic development authority without Goal/current policy."""

    kind: Literal["AlphaDevelopmentProgram"] = "AlphaDevelopmentProgram"
    foundation_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    logical_panel_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    logical_semantic_index_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    causal_outcome_snapshot_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    ordered_listing_ids_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    ordered_feature_ids_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    split_policy: AlphaDevelopmentSplitPolicy
    split_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    fold_commitment_hashes: tuple[Hash, ...] = Field(min_length=1)
    target_policy: AlphaTargetPolicy | None = None
    target_method_hash: Hash | None = Field(default=None, pattern=BLAKE3_PATTERN)
    """The resolved target method's own identity.

    ``target_policy`` cannot stand in for it. A method whose composition is not
    one of the four frozen lanes has no policy, so two arms of a paired study --
    the canonical bounded target and its unbounded control -- would both seal
    ``target_policy: None`` and produce the *same* Program identity while
    computing different targets. Every hash in the chain would agree and the
    numbers would have come from different science.
    """

    metric_policy: _AlphaMetricPolicy
    package_identity: AlphaPackageIdentity
    model_mandate_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    model_catalog_hash: Hash = Field(pattern=BLAKE3_PATTERN)
    program_hash: Hash = Field(pattern=BLAKE3_PATTERN)

    @property
    def split_policy_hash(self) -> str:
        return self.split_policy.policy_hash

    @property
    def metric_policy_hash(self) -> str:
        return self.metric_policy.policy_hash

    @property
    def package_identity_hash(self) -> str:
        return self.package_identity.package_hash

    @property
    def target_policy_hashes(self) -> tuple[str, ...] | None:
        if self.target_policy is None:
            return None
        return (self.target_policy.policy_hash,)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if len(set(self.fold_commitment_hashes)) != len(
            self.fold_commitment_hashes
        ) or self.program_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"program_hash"})
        ):
            raise ValueError("ALPHA_DEVELOPMENT_PROGRAM_IDENTITY_INVALID")
        return self


class AlphaDevelopmentProgramAuthority(Protocol):
    """Fields consumed by model-neutral development execution."""

    @property
    def foundation_hash(self) -> str: ...

    @property
    def logical_panel_hash(self) -> str: ...

    @property
    def logical_semantic_index_hash(self) -> str: ...

    @property
    def causal_outcome_snapshot_hash(self) -> str: ...

    @property
    def ordered_listing_ids_hash(self) -> str: ...

    @property
    def ordered_feature_ids_hash(self) -> str: ...

    @property
    def split_policy_hash(self) -> str: ...

    @property
    def metric_policy_hash(self) -> str: ...

    @property
    def package_identity_hash(self) -> str: ...

    @property
    def model_mandate_hash(self) -> str | None: ...

    @property
    def model_catalog_hash(self) -> str | None: ...

    @property
    def target_policy_hashes(self) -> tuple[str, ...] | None: ...

    @property
    def program_hash(self) -> str: ...


class AlphaExperimentRequest(_Contract):
    """Bind the candidate inventory and split policy to the requested research inputs."""

    kind: Literal["AlphaExperimentRequest"] = "AlphaExperimentRequest"
    foundation_hash: str = Field(pattern=BLAKE3_PATTERN)
    feature_panel_snapshot_hash: str = Field(pattern=BLAKE3_PATTERN)
    logical_panel_hash: str = Field(pattern=BLAKE3_PATTERN)
    logical_semantic_index_hash: str = Field(pattern=BLAKE3_PATTERN)
    causal_outcome_snapshot_hash: str = Field(pattern=BLAKE3_PATTERN)
    listing_set_hash: str = Field(pattern=BLAKE3_PATTERN)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    ordered_factor_ids: tuple[str, ...] = Field(min_length=1, max_length=55)
    inventory_hash: str = Field(pattern=BLAKE3_PATTERN)
    candidate_ids: tuple[str, ...] = Field(min_length=1)
    candidate_card_hashes: tuple[str, ...] = Field(min_length=1)
    split_policy: AlphaSplitPolicy
    metric_policy: _AlphaMetricPolicy
    package_identity: AlphaPackageIdentity
    request_hash: str = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaExperimentRequest:
        if self.ordered_listing_ids != tuple(sorted(set(self.ordered_listing_ids))):
            raise ValueError("Alpha request listing order is not canonical")
        if self.listing_set_hash != canonical_hash(self.ordered_listing_ids):
            raise ValueError("Alpha request listing-set hash is invalid")
        if self.ordered_factor_ids != tuple(dict.fromkeys(self.ordered_factor_ids)):
            raise ValueError("Alpha request factor order is not unique")
        if self.candidate_ids != tuple(dict.fromkeys(self.candidate_ids)):
            raise ValueError("Alpha request candidate inventory is not ordered and unique")
        if len(self.candidate_card_hashes) != len(self.candidate_ids):
            raise ValueError("Alpha request candidate/card counts differ")
        if len(set(self.candidate_card_hashes)) != len(self.candidate_card_hashes):
            raise ValueError("Alpha request candidate card hashes are not unique")
        if self.request_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"request_hash"})
        ):
            raise ValueError("Alpha numerical request hash is invalid")
        return self


class AlphaFoldCommitment(_Contract):
    """Seal the exact training and validation sessions of an admitted fold."""

    fold_index: int = Field(ge=0)
    train_first: date
    train_last: date
    validation_first: date
    validation_last: date
    train_session_count: int = Field(ge=1)
    validation_session_count: int = Field(ge=1)
    train_sessions_hash: str = Field(pattern=BLAKE3_PATTERN)
    validation_sessions_hash: str = Field(pattern=BLAKE3_PATTERN)
    commitment_hash: str = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaFoldCommitment:
        if self.train_first > self.train_last or self.validation_first > self.validation_last:
            raise ValueError("Alpha fold date boundaries are reversed")
        if self.train_last >= self.validation_first:
            raise ValueError("Alpha fold train/validation surfaces overlap")
        if self.commitment_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"commitment_hash"})
        ):
            raise ValueError("Alpha fold commitment hash is invalid")
        return self


class AlphaFitLedgerEntry(_Contract):
    """Record the resulting estimator or named failure for one candidate fold."""

    candidate_id: str
    fold_index: int = Field(ge=0)
    training_row_count: int = Field(ge=0)
    prediction_row_count: int = Field(ge=0)
    estimator_state_hash: str | None = Field(default=None, pattern=BLAKE3_PATTERN)
    status: Literal["COMPLETED", "FAILED", "NOT_APPLICABLE"]
    failure_code: str | None = Field(default=None, max_length=120)
    ledger_hash: str = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaFitLedgerEntry:
        if (self.status == "FAILED") != (self.failure_code is not None):
            raise ValueError("Alpha fit ledger failure state is inconsistent")
        if self.ledger_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"ledger_hash"})
        ):
            raise ValueError("Alpha fit ledger hash is invalid")
        return self


class AlphaCandidateFailure(_Contract):
    """Record a candidate failure by stage, stable code and affected fold."""

    candidate_id: str
    stage: AlphaFailureStage
    code: str = Field(min_length=1, max_length=120)
    safe_detail: str = Field(min_length=1, max_length=500)
    fold_index: int | None = Field(default=None, ge=0)
    failure_hash: str = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCandidateFailure:
        if self.failure_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"failure_hash"})
        ):
            raise ValueError("Alpha candidate failure hash is invalid")
        return self


__all__ = [
    "MODEL_SPEC_ADAPTER",
    "AlphaCandidateFailure",
    "AlphaCandidateRole",
    "AlphaCandidateStatus",
    "AlphaDevelopmentFitEvidence",
    "AlphaDevelopmentProgram",
    "AlphaDevelopmentProgramAuthority",
    "AlphaDevelopmentSplitPolicy",
    "AlphaExperimentBatch",
    "AlphaExperimentBatchResult",
    "AlphaExperimentCandidateResult",
    "AlphaExperimentCard",
    "AlphaExperimentRequest",
    "AlphaFailureStage",
    "AlphaFitLedgerEntry",
    "AlphaFoldCommitment",
    "AlphaPackageIdentity",
    "AlphaResearchProgram",
    "AlphaSplitPolicy",
    "ElasticNetModelSpec",
    "LassoModelSpec",
    "LegacyModelSpec",
    "ResolvedRegularizedLinearSpec",
    "RidgeModelSpec",
    "StoredModelSpec",
    "candidate_id_for_spec",
    "resolve_model_spec",
    "seal_contract",
]
