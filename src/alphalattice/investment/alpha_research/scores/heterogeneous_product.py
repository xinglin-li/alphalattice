"""Frozen heterogeneous Alpha successor: exact recipes and one lifecycle.

The predecessor public recipe remains installed and readable.  This module
installs a *successor* identity made from four complete Alpha children.  It
does not interpret Gate I score arrays as a model recipe: the three specialist
children restate the exact recipe manifests that produced them, while G0 names
the already-installed IW184 recipe by hash.

Scoring still belongs to the existing Alpha lifecycle.  The coordinator below
calls :func:`resolve_formation_lifecycle` once per child and only hands the
formation to a caller-supplied score owner after all four twelve-model sets have
been verified.  There is no G0-only fallback and no partial-component
renormalisation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import date
from typing import ClassVar, Final, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.alpha_research.scores.product_recipe import (
    INSTALLED_ALPHA_PRODUCT_RECIPE,
    PRODUCT_ESTIMATOR_POINT,
    AlphaEstimatorParameterPoint,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.recipe_identity import recipe_identity, recipe_seal_holds

HETEROGENEOUS_STRATEGY_ID: Final = "FOUR_COMPONENT_BOOK"
_RECIPE_SEAL: Final = frozenset({"recipe_hash"})
_STRATEGY_SEAL: Final = frozenset({"strategy_hash"})
SUCCESSOR_PACKAGE_HASH: Final = "89a9e84b2035d36b59b0f76a087c777fe8fd997973fd67aee1a7e28d3f4aafb7"
SUCCESSOR_REPLAY_RESULT_HASH: Final = (
    "956e0932d9d1c0c07eb538adeaca46e7ddba9ea37a6c288245461fafb5f71df5"
)
SUCCESSOR_REPORT_CONTRACT_HASH: Final = (
    "dc56d3bcfdaa464cbd295f48cb3163ee9530e62f5e87c276add6cfcb146a53ed"
)
SUCCESSOR_VALIDATION_CONTRACT_HASH: Final = (
    "c747925f2c5b6f6fd6b248bef8cb75f79a29968c799abf290e4692397dcaaea0"
)

ComponentId = Literal[
    "G0_IW184",
    "G2_R0_TREND",
    "G6_R0_FAST_REBOUND",
    "G7_R1_CONTEXTUAL_MOMENTUM",
]

ScoreAggregationSemantics = Literal[
    "PER_MODEL_PERCENTILE_0_1_THEN_EQUAL_SEED_MEAN_THEN_WEIGHTED_VINTAGE_MEAN",
    "EQUAL_RAW_SEED_MEAN_THEN_CANDIDATE_PERCENTILE_1_N_TO_1_THEN_WEIGHTED_VINTAGE_MEAN_WITH_OUTSIDER_MINUS_ONE",
]

COMPONENT_IDS: Final[tuple[ComponentId, ...]] = (
    "G0_IW184",
    "G2_R0_TREND",
    "G6_R0_FAST_REBOUND",
    "G7_R1_CONTEXTUAL_MOMENTUM",
)


class HeterogeneousLiveScoreClosureReceipt(BaseModel):  # type: ignore[misc]
    """Exact current Feature -> score -> Portfolio-consumer closure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["HeterogeneousLiveScoreClosureReceipt"] = "HeterogeneousLiveScoreClosureReceipt"
    formation_session: date
    strategy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    book_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    component_model_set_hashes: dict[ComponentId, str]
    component_projection_hashes: dict[ComponentId, str]
    raw_prediction_maximum_gaps: dict[ComponentId, float]
    aggregate_score_maximum_gaps: dict[ComponentId, float]
    maximum_raw_prediction_gap: float = Field(ge=0.0)
    maximum_aggregate_score_gap: float = Field(ge=0.0)
    decision_eligible_count: int = Field(ge=1)
    target_name_count: int = Field(ge=1)
    target_weight_sum: float
    target_weights_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    consumed_score_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_count: Literal[0] = 0
    prediction_call_count: Literal[96] = 96
    portfolio_decision_count: Literal[1] = 1
    provider_call_count: Literal[0] = 0
    protected_evaluation_count: Literal[0] = 0
    pointer_mutation_count: Literal[0] = 0
    source_readback_unchanged: Literal[True] = True
    disposition: Literal["READY"] = "READY"
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def seal(cls, values: Mapping[str, object]) -> Self:
        """Seal an unsealed live-score closure receipt from declared replay evidence.

        Args:
            values: Declared closure fields excluding receipt_hash.

        Returns:
            Validated receipt with the canonical complete closure identity.

        Raises:
            HeterogeneousAlphaError: A self identity was already supplied or closure consistency
                fails.
        """
        if "receipt_hash" in values:
            raise HeterogeneousAlphaError(
                "alpha_research.heterogeneous_live_score_receipt_already_sealed"
            )
        identity = cls.model_construct(**values, receipt_hash="0" * 64).model_dump(
            mode="json", exclude={"receipt_hash"}
        )
        return cls(**identity, receipt_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_closure(self) -> Self:
        """Require complete installed component axes, exact zero replay gaps and unit weight sum.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            HeterogeneousAlphaError: Component axes/identities, installed strategy, maximum gaps or
                target weight sum disagree, or receipt_hash is inconsistent.
        """
        if (
            tuple(self.component_model_set_hashes) != COMPONENT_IDS
            or tuple(self.component_projection_hashes) != COMPONENT_IDS
            or tuple(self.raw_prediction_maximum_gaps) != COMPONENT_IDS
            or tuple(self.aggregate_score_maximum_gaps) != COMPONENT_IDS
            or any(
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
                for value in (
                    *self.component_model_set_hashes.values(),
                    *self.component_projection_hashes.values(),
                )
            )
            or self.strategy_hash != INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.strategy_hash
            or self.maximum_raw_prediction_gap != max(self.raw_prediction_maximum_gaps.values())
            or self.maximum_aggregate_score_gap != max(self.aggregate_score_maximum_gaps.values())
            or self.maximum_raw_prediction_gap != 0.0
            or self.maximum_aggregate_score_gap != 0.0
            or self.target_weight_sum != 1.0
        ):
            raise HeterogeneousAlphaError("alpha_research.heterogeneous_live_score_closure_invalid")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise HeterogeneousAlphaError(
                "alpha_research.heterogeneous_live_score_receipt_identity_invalid"
            )
        return self


_TREND_CANDIDATE_FEATURES: Final = (
    "RELATIVE_STOCK_CROSS_SECTION::dist_52w_high::current",
    "RELATIVE_STOCK_CROSS_SECTION::momentum_consistency_252::current",
    "RELATIVE_STOCK_CROSS_SECTION::residual_mom_252_21::current",
    "RELATIVE_STOCK_CROSS_SECTION::trend_r2_252::current",
    "NON_NEUTRAL_STOCK_CROSS_SECTION::at_own_high_share_63::current",
    "NON_NEUTRAL_STOCK_CROSS_SECTION::high_extension::current",
    "NON_NEUTRAL_STOCK_CROSS_SECTION::return_run_length_21::current",
    "NON_NEUTRAL_STOCK_CROSS_SECTION::sessions_since_252_high::current",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::stock_sharpe_21",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::stock_sharpe_63",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::ppo_12_26",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::ppo_signal_gap_9",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::mean_return_acceleration_5_21",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::recovery_from_21d_low",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::momentum_sign_consistency_21",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::momentum_sign_consistency_63",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::trend_r2_21",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::trend_r2_63",
)

_FAST_REBOUND_FEATURES: Final = (
    "ALPHA_DEVELOPMENT_CANDIDATE::FAST_ATOMS_NON_NEUTRAL::price_efficiency_3",
    "ALPHA_DEVELOPMENT_CANDIDATE::FAST_ATOMS_NON_NEUTRAL::relative_volume_5",
    "ALPHA_DEVELOPMENT_CANDIDATE::FAST_ATOMS_NON_NEUTRAL::clv",
    "ALPHA_DEVELOPMENT_CANDIDATE::FAST_ATOMS_NON_NEUTRAL::fast_slope_2",
    "ALPHA_DEVELOPMENT_CANDIDATE::FAST_ATOMS_NON_NEUTRAL::prior_slope_3",
    "ALPHA_DEVELOPMENT_CANDIDATE::FAST_ATOMS_NON_NEUTRAL::natr_5",
    "ALPHA_DEVELOPMENT_CANDIDATE::FAST_ATOMS_NON_NEUTRAL::direction_balance_3",
    "ALPHA_DEVELOPMENT_CANDIDATE::FAST_COMPOSITES_NON_NEUTRAL::iie_3",
    "ALPHA_DEVELOPMENT_CANDIDATE::FAST_COMPOSITES_NON_NEUTRAL::tcd_vol",
    "ALPHA_DEVELOPMENT_CANDIDATE::FAST_COMPOSITES_NON_NEUTRAL::rca_5",
    "ALPHA_DEVELOPMENT_CANDIDATE::FAST_COMPOSITES_NON_NEUTRAL::dmi_3",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::ppo_12_26",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::ppo_signal_gap_9",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::mean_return_acceleration_5_21",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::recovery_from_21d_low",
    "MARKET_CONTEXT::market_drawdown_252::lag1",
    "MARKET_CONTEXT::observed_breadth_positive_share::current",
    "MARKET_CONTEXT::observed_new_high_low_share::current",
    "SECTOR_CONTEXT::sector_trend_20::lag1",
    "ALPHA_DEVELOPMENT_CANDIDATE::C::sector_recovery_from_low_21",
)

_CONTEXTUAL_MOMENTUM_FEATURES: Final = (
    "RELATIVE_STOCK_CROSS_SECTION::mom_126_21::current",
    "RELATIVE_STOCK_CROSS_SECTION::mom_252_21::current",
    "RELATIVE_STOCK_CROSS_SECTION::residual_mom_252_21::current",
    "RELATIVE_STOCK_CROSS_SECTION::momentum_consistency_252::current",
    "RELATIVE_STOCK_CROSS_SECTION::trend_r2_252::current",
    "RELATIVE_STOCK_CROSS_SECTION::directional_strength_14::current",
    "RELATIVE_STOCK_CROSS_SECTION::dist_52w_high::current",
    "RELATIVE_STOCK_CROSS_SECTION::ma_gap_50_200::current",
    "RELATIVE_STOCK_CROSS_SECTION::price_to_ma_200::current",
    "RELATIVE_STOCK_CROSS_SECTION::return_autocorr_21::current",
    "RELATIVE_STOCK_CROSS_SECTION::obv_slope_63::current",
    "RELATIVE_STOCK_CROSS_SECTION::cmf_21::current",
    "NON_NEUTRAL_STOCK_CROSS_SECTION::at_own_high_share_63::current",
    "NON_NEUTRAL_STOCK_CROSS_SECTION::high_extension::current",
    "NON_NEUTRAL_STOCK_CROSS_SECTION::return_run_length_21::current",
    "NON_NEUTRAL_STOCK_CROSS_SECTION::sessions_since_252_high::current",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::ppo_12_26",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::ppo_signal_gap_9",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::mean_return_acceleration_5_21",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::momentum_sign_consistency_21",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::momentum_sign_consistency_63",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::trend_r2_21",
    "ALPHA_DEVELOPMENT_CANDIDATE::R_NON_NEUTRAL::trend_r2_63",
    "MARKET_CONTEXT::market_raw_log_return::lag5",
    "MARKET_CONTEXT::market_raw_log_return::mean21",
    "MARKET_CONTEXT::market_raw_log_return::mean63",
    "MARKET_CONTEXT::market_volatility_21::lag1",
    "MARKET_CONTEXT::market_volatility_21::mean21",
    "MARKET_CONTEXT::market_drawdown_252::lag1",
    "MARKET_CONTEXT::market_drawdown_252::mean21",
    "MARKET_CONTEXT::observed_breadth_positive_share::current",
    "MARKET_CONTEXT::observed_breadth_positive_share::lag4",
    "MARKET_CONTEXT::observed_new_high_low_share::current",
    "MARKET_CONTEXT::observed_new_high_low_share::lag4",
    "MARKET_CONTEXT::observed_up_volume_share::current",
    "MARKET_CONTEXT::observed_up_volume_share::lag4",
    "MARKET_CONTEXT::observed_rotation_ratio::current",
    "MARKET_CONTEXT::observed_rotation_ratio::lag4",
    "SECTOR_CONTEXT::sector_equal_weight_raw_simple_return::lag1",
    "SECTOR_CONTEXT::sector_equal_weight_raw_simple_return::mean21",
    "SECTOR_CONTEXT::sector_trend_20::lag1",
    "SECTOR_CONTEXT::sector_trend_20::mean21",
    "SECTOR_CONTEXT::sector_surprise_0::lag1",
    "SECTOR_CONTEXT::sector_surprise_1::lag1",
)


class HeterogeneousAlphaError(ValueError):
    """Stable refusal for successor recipe, lifecycle or component failure."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class HeterogeneousAlphaComponentRecipe(_Contract):
    """One complete Alpha child, compatible with the existing lifecycle view."""

    LABELS: ClassVar[frozenset[str]] = frozenset(
        {
            "component_id",
            "family_id",
            "feature_axis_id",
            "target_cross_section",
            "score_consumer",
            "objective",
            "evidence_disposition",
        }
    )
    """Names and words the recipe hash leaves out: what they describe is bound by the hashes,
    the target method and the parameters beside them (ID10)."""

    kind: Literal["HeterogeneousAlphaComponentRecipe"] = "HeterogeneousAlphaComponentRecipe"
    component_id: ComponentId
    family_id: str = Field(min_length=2)
    research_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_recipe_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    feature_axis_id: str = Field(min_length=1)
    feature_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_count: int = Field(ge=1)
    ordered_feature_ids: tuple[str, ...] = ()
    source_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_recipe: str = Field(min_length=1)
    target_cross_section: str = Field(min_length=1)
    candidate_semantics: str = Field(min_length=1)
    score_consumer: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    estimator_point: AlphaEstimatorParameterPoint
    training_window_sessions: int = Field(ge=1)
    purge_sessions: int = Field(ge=0)
    refit_quarter_start_months: tuple[int, ...] = (1, 4, 7, 10)
    seeds: tuple[int, ...] = (1729, 2718, 31415)
    vintage_count: Literal[4] = 4
    vintage_weights: tuple[int, ...] = (4, 3, 2, 1)
    live_model_count: Literal[12] = 12
    score_aggregation: ScoreAggregationSemantics
    evidence_disposition: Literal["FROZEN_RESEARCH_PACKAGE_ADMITTED_AS_SUCCESSOR"] = (
        "FROZEN_RESEARCH_PACKAGE_ADMITTED_AS_SUCCESSOR"
    )
    evidence_daily_parquet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal a component recipe after validating its typed estimator parameter point.

        Args:
            values: Explicit component fields excluding recipe_hash; estimator_point is
                model-validated.

        Returns:
            Validated component recipe and canonical recipe_hash.

        Raises:
            pydantic.ValidationError: Component fields, estimator point or recipe consistency
                violate the model.
        """
        constructed = dict(values)
        constructed["estimator_point"] = AlphaEstimatorParameterPoint.model_validate(
            constructed["estimator_point"]
        )
        draft = cls.model_construct(**constructed, recipe_hash="0" * 64)
        identity = draft.model_dump(mode="json", exclude={"recipe_hash"})
        return cls(**identity, recipe_hash=recipe_identity(draft, exclude=_RECIPE_SEAL))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require feature-axis identity, three distinct seeds and exact live-model count.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            HeterogeneousAlphaError: Feature count/hash, seed uniqueness, vintage-by-seed model
                count or recipe_hash is inconsistent.
        """
        if self.ordered_feature_ids:
            if len(self.ordered_feature_ids) != self.feature_count:
                raise HeterogeneousAlphaError("alpha_research.heterogeneous_feature_count_invalid")
            if canonical_hash(list(self.ordered_feature_ids)) != self.feature_axis_hash:
                raise HeterogeneousAlphaError("alpha_research.heterogeneous_feature_axis_invalid")
        if (
            len(set(self.seeds)) != 3
            or self.live_model_count != len(self.seeds) * self.vintage_count
        ):
            raise HeterogeneousAlphaError("alpha_research.heterogeneous_model_set_invalid")
        if not recipe_seal_holds(self, "recipe_hash"):
            raise HeterogeneousAlphaError("alpha_research.heterogeneous_recipe_identity_invalid")
        return self


def _component_recipes() -> tuple[HeterogeneousAlphaComponentRecipe, ...]:
    g0 = INSTALLED_ALPHA_PRODUCT_RECIPE
    common: dict[str, object] = {
        "estimator_point": PRODUCT_ESTIMATOR_POINT,
        "refit_quarter_start_months": (1, 4, 7, 10),
        "seeds": (1729, 2718, 31415),
        "vintage_count": 4,
        "vintage_weights": (4, 3, 2, 1),
        "live_model_count": 12,
        "evidence_disposition": "FROZEN_RESEARCH_PACKAGE_ADMITTED_AS_SUCCESSOR",
    }
    return (
        HeterogeneousAlphaComponentRecipe.create(
            component_id="G0_IW184",
            family_id="BROAD_FEATURE_MODEL",
            research_recipe_hash=g0.recipe_hash,
            parent_recipe_hash=g0.recipe_hash,
            feature_axis_id=g0.feature_axis_id,
            feature_axis_hash=g0.feature_axis_hash,
            feature_count=g0.feature_count,
            source_binding_hash=g0.source_candidate_manifest_sha256,
            target_recipe="EXACT_FROZEN_G0_H1_WHOLE_UNIVERSE_TARGET",
            target_cross_section="UNIVERSE_RETURN_NORMALIZATION",
            candidate_semantics="ALL_ELIGIBLE_FULL_UNIVERSE",
            score_consumer="STANDALONE_SCORE_SORT",
            objective="BROAD_SQUARED_ERROR",
            training_window_sessions=g0.training_window_sessions,
            purge_sessions=g0.purge_sessions,
            score_aggregation=(
                "PER_MODEL_PERCENTILE_0_1_THEN_EQUAL_SEED_MEAN_THEN_WEIGHTED_VINTAGE_MEAN"
            ),
            evidence_daily_parquet_sha256="c8a02ce9637a2799acb7faf64153450e7fe934ef06abeb601b659f28efafcd87",
            **common,
        ),
        HeterogeneousAlphaComponentRecipe.create(
            component_id="G2_R0_TREND",
            family_id="TREND_REBOUND_LGBM",
            research_recipe_hash="145f2cdc492e2c12dae67426eb5753f8d73a4295681ae5bc474ab9aa17ae0e28",
            feature_axis_id="TREND_FEATURE_AXIS",
            feature_axis_hash=canonical_hash(list(_TREND_CANDIDATE_FEATURES)),
            feature_count=len(_TREND_CANDIDATE_FEATURES),
            ordered_feature_ids=_TREND_CANDIDATE_FEATURES,
            source_binding_hash="bb06155d2282cd5992988d134c2d108005f9b7f23e9410f1a1af9ac537a89392",
            target_recipe="T1_H3_PURE_TOTAL_RETURN_Z",
            target_cross_section="WHOLE_UNIVERSE_MAD_3_5_MEAN_CENTER_SAMPLE_STD_Z",
            candidate_semantics="FORMATION_CLOSE_RAW_12_1_TOP100_WITH_BOUNDARY_TIES_MINIMUM_100",
            score_consumer="CANDIDATE_PERCENTILE_WITH_ELIGIBLE_OUTSIDER_SENTINEL_MINUS_ONE",
            objective="EXACT_GATE_B_L2",
            training_window_sessions=1260,
            purge_sessions=5,
            score_aggregation=(
                "EQUAL_RAW_SEED_MEAN_THEN_CANDIDATE_PERCENTILE_1_N_TO_1_THEN_"
                "WEIGHTED_VINTAGE_MEAN_WITH_OUTSIDER_MINUS_ONE"
            ),
            evidence_daily_parquet_sha256="9649fb62fc01793ad9b5fc51b64b9567fdfd33c4b4acd57b6b619c542ce589f6",
            **common,
        ),
        HeterogeneousAlphaComponentRecipe.create(
            component_id="G6_R0_FAST_REBOUND",
            family_id="FAST_REBOUND_RECAPTURE",
            research_recipe_hash="a2896193b0575e37d4ad99ed5b2bdbdd29bab9a333298e9bddaaaa5f251ee61f",
            feature_axis_id="REBOUND_CONTEXT_AXIS",
            feature_axis_hash=canonical_hash(list(_FAST_REBOUND_FEATURES)),
            feature_count=len(_FAST_REBOUND_FEATURES),
            ordered_feature_ids=_FAST_REBOUND_FEATURES,
            source_binding_hash="bb06155d2282cd5992988d134c2d108005f9b7f23e9410f1a1af9ac537a89392",
            target_recipe="EXACT_FROZEN_G0_H1_WHOLE_UNIVERSE_TARGET",
            target_cross_section="UNIVERSE_RETURN_NORMALIZATION",
            candidate_semantics="ALL_ELIGIBLE_FULL_UNIVERSE",
            score_consumer="FULL_FINITE_SCORE_SURFACE_EXACT_BATCH81_BOOK",
            objective="ROW_BALANCED_L2",
            training_window_sessions=1260,
            purge_sessions=5,
            score_aggregation=(
                "EQUAL_RAW_SEED_MEAN_THEN_CANDIDATE_PERCENTILE_1_N_TO_1_THEN_"
                "WEIGHTED_VINTAGE_MEAN_WITH_OUTSIDER_MINUS_ONE"
            ),
            evidence_daily_parquet_sha256="d489c43ba240903ed214776deaedff3260b2907cf315f2c693be107fae64eb7e",
            **common,
        ),
        HeterogeneousAlphaComponentRecipe.create(
            component_id="G7_R1_CONTEXTUAL_MOMENTUM",
            family_id="CONTEXTUAL_MOMENTUM_LGBM",
            research_recipe_hash="df313e2cc6aaa5c6ea10bb91c8c537d0fd83dd1c436db34cf57311b85aad02b3",
            feature_axis_id="MOMENTUM_CONTEXT_AXIS",
            feature_axis_hash=canonical_hash(list(_CONTEXTUAL_MOMENTUM_FEATURES)),
            feature_count=len(_CONTEXTUAL_MOMENTUM_FEATURES),
            ordered_feature_ids=_CONTEXTUAL_MOMENTUM_FEATURES,
            source_binding_hash="bb06155d2282cd5992988d134c2d108005f9b7f23e9410f1a1af9ac537a89392",
            target_recipe="MOMENTUM_CONTINUATION_H1_Z",
            target_cross_section="CANDIDATE_ONLY_MAD_3_5_MEAN_CENTER_SAMPLE_STD_Z",
            candidate_semantics=(
                "FORMATION_CLOSE_MIDRANK_PERCENTILE_GTE_0_50_WITH_BOUNDARY_TIES_MINIMUM_70"
            ),
            score_consumer="CANDIDATE_PERCENTILE_WITH_ELIGIBLE_OUTSIDER_SENTINEL_MINUS_ONE",
            objective="CONTEXTUAL_MOMENTUM_L2",
            training_window_sessions=1260,
            purge_sessions=5,
            score_aggregation=(
                "EQUAL_RAW_SEED_MEAN_THEN_CANDIDATE_PERCENTILE_1_N_TO_1_THEN_"
                "WEIGHTED_VINTAGE_MEAN_WITH_OUTSIDER_MINUS_ONE"
            ),
            evidence_daily_parquet_sha256="f908b81b103cca2c2d1ff44da59f13b2cd4084437835d3eaca3d0f761d88f045",
            **common,
        ),
    )


class HeterogeneousAlphaStrategyRecipe(_Contract):
    """The four Alpha children and the exact Portfolio handoff they feed."""

    LABELS: ClassVar[frozenset[str]] = frozenset({"strategy_id"})
    """The strategy's name, which its hash leaves out (ID10)."""

    kind: Literal["HeterogeneousAlphaStrategyRecipe"] = "HeterogeneousAlphaStrategyRecipe"
    strategy_id: Literal["FOUR_COMPONENT_BOOK"] = HETEROGENEOUS_STRATEGY_ID
    predecessor_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    successor_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    successor_replay_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    components: tuple[HeterogeneousAlphaComponentRecipe, ...]
    allocation_basis_points: tuple[int, ...] = (2500, 2500, 2500, 2500)
    component_consumer: Literal["FOUR_INDEPENDENT_TOP35_EXIT70_TRANCHE3_BOOKS"] = (
        "FOUR_INDEPENDENT_TOP35_EXIT70_TRANCHE3_BOOKS"
    )
    merge_semantics: Literal["POST_TRADE_LISTING_WEIGHTS_THEN_RECOMPUTE_ECONOMICS"] = (
        "POST_TRADE_LISTING_WEIGHTS_THEN_RECOMPUTE_ECONOMICS"
    )
    weighting_rule: Literal["ew"] = "ew"
    hazard_overlay: Literal["NONE"] = "NONE"
    risk_for_weights: Literal["NONE_EQUAL_WEIGHT_COMPONENT_BOOKS"] = (
        "NONE_EQUAL_WEIGHT_COMPONENT_BOOKS"
    )
    report_contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def installed(cls) -> Self:
        """Seal the installed four-component equal-budget strategy and its frozen receipts.

        Each component receives 2500 basis points and maintains an independent equal-weight book.
        Post-trade listing weights merge before economic recalculation; the installed declaration
        carries no hazard or Risk weighting overlay.

        Returns:
            Installed strategy recipe with exact component recipes, consumer policy and canonical
            identity.
        """
        values: dict[str, object] = {
            "kind": "HeterogeneousAlphaStrategyRecipe",
            "strategy_id": HETEROGENEOUS_STRATEGY_ID,
            "predecessor_recipe_hash": INSTALLED_ALPHA_PRODUCT_RECIPE.recipe_hash,
            "successor_package_hash": SUCCESSOR_PACKAGE_HASH,
            "successor_replay_result_hash": SUCCESSOR_REPLAY_RESULT_HASH,
            "components": [value.model_dump(mode="json") for value in _component_recipes()],
            "allocation_basis_points": [2500, 2500, 2500, 2500],
            "component_consumer": "FOUR_INDEPENDENT_TOP35_EXIT70_TRANCHE3_BOOKS",
            "merge_semantics": "POST_TRADE_LISTING_WEIGHTS_THEN_RECOMPUTE_ECONOMICS",
            "weighting_rule": "ew",
            "hazard_overlay": "NONE",
            "risk_for_weights": "NONE_EQUAL_WEIGHT_COMPONENT_BOOKS",
            "report_contract_hash": SUCCESSOR_REPORT_CONTRACT_HASH,
            "validation_contract_hash": SUCCESSOR_VALIDATION_CONTRACT_HASH,
        }
        draft = cls.model_construct(**values, strategy_hash="0" * 64)
        return cls(**values, strategy_hash=recipe_identity(draft, exclude=_STRATEGY_SEAL))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the installed component order and four equal 2500-basis-point budgets.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            HeterogeneousAlphaError: Component order, allocation size/sum/equality or strategy_hash
                is inconsistent.
        """
        if tuple(value.component_id for value in self.components) != COMPONENT_IDS:
            raise HeterogeneousAlphaError("alpha_research.heterogeneous_component_set_invalid")
        if len(self.allocation_basis_points) != 4 or sum(self.allocation_basis_points) != 10_000:
            raise HeterogeneousAlphaError("alpha_research.heterogeneous_allocation_invalid")
        if any(value != 2500 for value in self.allocation_basis_points):
            raise HeterogeneousAlphaError("alpha_research.heterogeneous_allocation_invalid")
        if not recipe_seal_holds(self, "strategy_hash"):
            raise HeterogeneousAlphaError("alpha_research.heterogeneous_strategy_identity_invalid")
        return self

    def component(self, component_id: ComponentId) -> HeterogeneousAlphaComponentRecipe:
        """Resolve one component recipe from the installed ordered strategy.

        Args:
            component_id: Exact requested component identifier.

        Returns:
            Matching installed component recipe.

        Raises:
            HeterogeneousAlphaError: The component is absent from the strategy.
        """
        for recipe in self.components:
            if recipe.component_id == component_id:
                return recipe
        raise HeterogeneousAlphaError("alpha_research.heterogeneous_component_absent")


INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY: Final = HeterogeneousAlphaStrategyRecipe.installed()

STRATEGY_ROLE: Final = "alpha_research.strategy_recipe"
"""The readout role of the installed strategy hash: a recorded hash is current through its moves."""

_COMPONENT_ROLE_KEYS: Final[dict[ComponentId, str]] = {
    "G0_IW184": "broad_feature",
    "G2_R0_TREND": "trend_candidate",
    "G6_R0_FAST_REBOUND": "fast_rebound",
    "G7_R1_CONTEXTUAL_MOMENTUM": "contextual_momentum",
}
"""Each component's readout role, named by what it scores: the ids workspaces store keep their
spelling until the release's corpora are prepared fresh (NM2, V451)."""


def component_recipe_role(component_id: str) -> str:
    """The readout role of one installed component recipe (NM1)."""
    return (
        f"alpha_research.component_recipe.{_COMPONENT_ROLE_KEYS[cast(ComponentId, component_id)]}"
    )


def installed_component_recipe_hash(component: str) -> str:
    """One installed component recipe's hash, by its role's key: the role's value (NM1)."""
    for component_id, key in _COMPONENT_ROLE_KEYS.items():
        if key == component:
            return INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component_id).recipe_hash
    raise HeterogeneousAlphaError("alpha_research.heterogeneous_component_absent")


def installed_strategy_hash() -> str:
    """The installed strategy hash: its readout role's value (NM1)."""
    return INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.strategy_hash


def is_installed_component(component: HeterogeneousAlphaComponentRecipe) -> bool:
    """Whether a stored component recipe is the installed one: its content's identity (ID10)."""
    installed = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component.component_id)
    return recipe_identity(component, exclude=_RECIPE_SEAL) == installed.recipe_hash


def component_recipe_is_current(component_id: str, recorded: str, installed: str) -> bool:
    """Whether two recorded component recipe hashes name one recipe.

    Equal, or one reaches the other through the role's recorded moves: a value written before a
    move against one written after.
    """
    role = component_recipe_role(component_id)
    return is_current(role, recorded, installed) or is_current(role, installed, recorded)


def strategy_recipe_is_current(recorded: str, installed: str) -> bool:
    """Whether two recorded strategy hashes name one strategy, as `component_recipe_is_current`.

    A sealed closure keeps the strategy hash it was sealed with; one sealed before a recorded
    move of the strategy's role still names the installed strategy (NM1's move, V451).
    """
    return is_current(STRATEGY_ROLE, recorded, installed) or is_current(
        STRATEGY_ROLE, installed, recorded
    )


LIVE_MODEL_VINTAGES: Final[tuple[str, ...]] = ("2026-07", "2026-04", "2026-01", "2025-10")
"""The four live vintages, newest first, of the admitted child model set."""

LIVE_SCORE_CLOSURE_RECEIPT_HASH: Final = (
    "e7a58ee330567147ac95f87119c0514e6ba9b47b39d8f433e89746b05cb580b5"
)
"""The Gate M closure that reopened all 48 children and scored them at zero gap."""

LIVE_SOURCE_READBACK_SHA256: Final = (
    "ecb5e2b679d7779f723ee7b41510b9d78338c6421f7badc1a5b0fe2256f9617b"
)
"""The Gate M source readback: every file the closure was allowed to derive from."""

LIVE_MODEL_CLOSURE_MANIFEST_HASH: Final = (
    "570780576859cc649f1d19dce48382dc74b9231a07be93e3b656998ccd2b5b42"
)
"""The Gate L manifest that reconstructed and admitted the twelve G0 children."""

LIVE_COMPONENT_MODEL_SET_HASHES: Final[dict[ComponentId, str]] = {
    "G0_IW184": "fdf50ae203586a2ab314ddfe5d563ae863692041eb68ea9486e778a0c3ea2132",
    "G2_R0_TREND": "ebd48211e4c2d7ec70b3b8dc15ed6fcd8f60f56e7ffe5171afee9039ed808eb1",
    "G6_R0_FAST_REBOUND": "8cadf2a8344b4f86d12644bb71778a19216357a985869172aa7fb4110c7111d8",
    "G7_R1_CONTEXTUAL_MOMENTUM": "b2d982a99c072059f32ed5805dd23c83398c01b9f9de883bc0b8c92b12bf599b",
}
"""The four live model-set identities, as `HeterogeneousLiveModelSet` over content hashes.

The three specialists' are the values in the Gate M receipt. G0's is the same
formula over the twelve Gate L children; the receipt names G0 by its Gate L
manifest hash instead, and the sealed closure manifest binds both.
"""


def array_value_hash(values: npt.ArrayLike) -> str:
    """The identity of one float lane: sha256 over its little-endian float64 bytes."""
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


class HeterogeneousChildModelIdentity(_Contract):
    """One admitted child: which vintage and seed, and exactly which model."""

    kind: Literal["HeterogeneousChildModelIdentity"] = "HeterogeneousChildModelIdentity"
    vintage: str = Field(pattern=r"^\d{4}-\d{2}$")
    seed: int
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The estimator payload's own content identity, the thing a reopen proves."""

    lineage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The fit and prediction receipt that produced the payload."""


class HeterogeneousComponentModelSet(_Contract):
    """The exact twelve children one component scores with, as its scoring recipe view.

    The science fields are pinned to the installed component recipe by the
    validator. The Feature axis is bound by hash: every payload carries its axis
    and the scoring owner refuses a model or a surface whose axis does not hash
    here, which is how the installed children were scored. An authority whose
    axis is not the installed recipe's lists it, and its hash moves.
    """

    kind: Literal["HeterogeneousComponentModelSet"] = "HeterogeneousComponentModelSet"
    component_id: ComponentId
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_count: int = Field(ge=1)
    ordered_feature_ids: tuple[str, ...] = ()
    seeds: tuple[int, ...]
    vintage_count: int
    vintage_weights: tuple[int, ...]
    candidate_semantics: str = Field(min_length=1)
    score_aggregation: ScoreAggregationSemantics
    children: tuple[HeterogeneousChildModelIdentity, ...]
    model_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The live model-set identity Gate M sealed: recipe plus content hashes, in order."""

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal a live model set using recipe and ordered child content identities.

        model_set_hash binds recipe_hash and the ordered content_hash values. Lineage, environment
        and other record fields remain declared evidence and are separately checked by model-set
        validation.

        Args:
            values: Explicit model-set fields excluding model_set_hash.

        Returns:
            Validated model set whose semantic identity is derived by live_model_set_hash.

        Raises:
            pydantic.ValidationError: Model-set fields or closure consistency violate the model.
        """
        constructed = cls.model_construct(**values, model_set_hash="0" * 64)
        identity = constructed.model_dump(mode="json", exclude={"model_set_hash"})
        return cls(
            **identity,
            model_set_hash=live_model_set_hash(
                recipe_hash=constructed.recipe_hash,
                content_hashes=tuple(value.content_hash for value in constructed.children),
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_closure(self) -> Self:
        """Require installed recipe/vintage/seed closure, unique children and exact model identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            HeterogeneousAlphaError: Recipe semantics, ordered child population, feature-axis
                authority or recipe-plus-child-content model_set_hash differs from the installed
                declaration.
        """
        recipe = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(self.component_id)
        vintages = tuple(dict.fromkeys(value.vintage for value in self.children))
        expected = tuple((vintage, seed) for vintage in vintages for seed in recipe.seeds)
        if (
            self.recipe_hash != recipe.recipe_hash
            or self.seeds != recipe.seeds
            or self.vintage_count != recipe.vintage_count
            or self.vintage_weights != recipe.vintage_weights
            or self.candidate_semantics != recipe.candidate_semantics
            or self.score_aggregation != recipe.score_aggregation
            or len(self.children) != recipe.live_model_count
            or len(vintages) != recipe.vintage_count
            or tuple((value.vintage, value.seed) for value in self.children) != expected
            or len({value.content_hash for value in self.children}) != len(self.children)
        ):
            raise HeterogeneousAlphaError("alpha_research.heterogeneous_model_set_closure_invalid")
        if self.ordered_feature_ids:
            axis_valid = (
                len(self.ordered_feature_ids) == self.feature_count
                and len(set(self.ordered_feature_ids)) == self.feature_count
                and canonical_hash(list(self.ordered_feature_ids)) == self.feature_axis_hash
            )
        else:
            axis_valid = (
                self.feature_axis_hash == recipe.feature_axis_hash
                and self.feature_count == recipe.feature_count
            )
        if not axis_valid:
            raise HeterogeneousAlphaError("alpha_research.heterogeneous_model_set_axis_invalid")
        if self.model_set_hash != live_model_set_hash(
            recipe_hash=self.recipe_hash,
            content_hashes=tuple(value.content_hash for value in self.children),
        ):
            raise HeterogeneousAlphaError("alpha_research.heterogeneous_model_set_identity_invalid")
        return self

    @property
    def vintages(self) -> tuple[str, ...]:
        """Read unique model vintages in first-child encounter order.

        Returns:
            Ordered distinct vintage identifiers from the child sequence.
        """
        return tuple(dict.fromkeys(value.vintage for value in self.children))


def live_model_set_hash(*, recipe_hash: str, content_hashes: tuple[str, ...]) -> str:
    """Gate M's live model-set identity: the component recipe plus its payloads, in order."""
    return str(
        canonical_hash(
            {
                "kind": "HeterogeneousLiveModelSet",
                "component_recipe_hash": recipe_hash,
                "models": list(content_hashes),
            }
        )
    )


class HeterogeneousSurfaceBinding(_Contract):
    """What one vintage's Feature surface for one formation must be, to be scored."""

    kind: Literal["HeterogeneousSurfaceBinding"] = "HeterogeneousSurfaceBinding"
    component_id: ComponentId
    vintage: str = Field(pattern=r"^\d{4}-\d{2}$")
    formation_session: date
    source_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Where the values came from: the matrix or projection identity of that vintage."""

    feature_values_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The exact values, as the live surface itself hashes them."""


class HeterogeneousLaneValueBinding(_Contract):
    """One formation's row of a scored lane, by value identity."""

    kind: Literal["HeterogeneousLaneValueBinding"] = "HeterogeneousLaneValueBinding"
    formation_session: date
    value_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class HeterogeneousModelSetAuthority(_Contract):
    """The exact current-scoring authority, derived from one sealed closure manifest.

    This is what a Program binds when the successor scores on the current axis.
    A count of models is not an identity; a complete set of forty-eight children
    that are not these forty-eight is a different strategy, and so is the same
    forty-eight scored over different Feature values or a different momentum
    lane. Everything a prediction depends on is named here -- the children, the
    formation and listing axes, every surface's source and value identity, the
    momentum lane, the numerical environment, and the manifest that sealed them
    -- so that any correction rotates the authority hash and, with it, every
    PLAN, Program, ledger, report and result sealed under it.

    It is never written by hand. The closure reader constructs it from a
    verified manifest; a fixture constructs it from the closure it stands for.
    """

    kind: Literal["HeterogeneousModelSetAuthority"] = "HeterogeneousModelSetAuthority"
    strategy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_source_mode: Literal["CURRENT_MODEL_SCORING"] = "CURRENT_MODEL_SCORING"
    closure_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    closure_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The Gate M receipt that reopened this exact child set and reported zero gap."""

    numerical_environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    live_vintages: tuple[str, ...]
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    listing_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    components: tuple[HeterogeneousComponentModelSet, ...]
    surface_bindings: tuple[HeterogeneousSurfaceBinding, ...] = Field(min_length=1)
    momentum_source_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    momentum_value_bindings: tuple[HeterogeneousLaneValueBinding, ...] = Field(min_length=1)
    authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal declared heterogeneous model, formation and input-lane authority.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical authority_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, authority_hash="0" * 64).model_dump(
            mode="json", exclude={"authority_hash"}
        )
        return cls(**identity, authority_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require complete installed component/vintage/session input authority.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            HeterogeneousAlphaError: Strategy/component/vintage/session axes, exhaustive
                feature-surface coverage, momentum axis or authority_hash is inconsistent.
        """
        sessions = self.formation_sessions
        if (
            self.strategy_hash != INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.strategy_hash
            or tuple(value.component_id for value in self.components) != COMPONENT_IDS
            or len(self.live_vintages) != 4
            or any(value.vintages != self.live_vintages for value in self.components)
            or tuple(sorted(set(sessions))) != sessions
        ):
            raise HeterogeneousAlphaError("alpha_research.heterogeneous_authority_invalid")
        expected_surfaces = {
            (component, vintage, session)
            for component in COMPONENT_IDS
            for vintage in self.live_vintages
            for session in sessions
        }
        declared_surfaces = [
            (value.component_id, value.vintage, value.formation_session)
            for value in self.surface_bindings
        ]
        declared_momentum = [value.formation_session for value in self.momentum_value_bindings]
        if (
            len(declared_surfaces) != len(expected_surfaces)
            or set(declared_surfaces) != expected_surfaces
            or tuple(declared_momentum) != sessions
        ):
            raise HeterogeneousAlphaError("alpha_research.heterogeneous_authority_lanes_invalid")
        if self.authority_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"authority_hash"})
        ):
            raise HeterogeneousAlphaError("alpha_research.heterogeneous_authority_identity_invalid")
        return self

    def component(self, component_id: ComponentId) -> HeterogeneousComponentModelSet:
        """Resolve one declared component model set from this authority.

        Args:
            component_id: Exact component whose live model set is requested.

        Returns:
            Matching admitted component model set.

        Raises:
            HeterogeneousAlphaError: The component is absent.
        """
        for value in self.components:
            if value.component_id == component_id:
                return value
        raise HeterogeneousAlphaError("alpha_research.heterogeneous_component_absent")

    def surface(
        self, *, component_id: ComponentId, vintage: str, formation_session: date
    ) -> HeterogeneousSurfaceBinding | None:
        """Find the exact component, vintage and formation feature-surface binding.

        Args:
            component_id: Requested component.
            vintage: Requested model vintage.
            formation_session: Requested source formation.

        Returns:
            Matching declared binding, or None when the exact tuple is absent.
        """
        for value in self.surface_bindings:
            if (
                value.component_id == component_id
                and value.vintage == vintage
                and value.formation_session == formation_session
            ):
                return value
        return None

    def momentum_value_hash(self, formation_session: date) -> str | None:
        """Read the declared raw momentum value identity for one formation.

        Args:
            formation_session: Requested source formation.

        Returns:
            Matching value identity, or None when no binding names that formation.
        """
        for value in self.momentum_value_bindings:
            if value.formation_session == formation_session:
                return value.value_hash
        return None


def heterogeneous_model_set_authority(
    children: Mapping[ComponentId, tuple[tuple[str, int, str, str, str], ...]],
    *,
    live_vintages: tuple[str, ...],
    closure_receipt_hash: str,
    closure_manifest_hash: str,
    numerical_environment_hash: str,
    formation_sessions: tuple[date, ...],
    ordered_listing_ids: tuple[str, ...],
    surface_bindings: tuple[HeterogeneousSurfaceBinding, ...],
    momentum_source_identity_hash: str,
    momentum_value_bindings: tuple[HeterogeneousLaneValueBinding, ...],
    feature_axes: Mapping[ComponentId, tuple[str, ...]] | None = None,
) -> HeterogeneousModelSetAuthority:
    """Seal one authority from `(vintage, seed, recipe, content, lineage)` rows and its lanes.

    The science fields come from the installed component recipes and cannot be
    supplied. The Feature axes default to the recipes' own; a caller declaring a
    different closure names them explicitly, and the authority hash moves.
    """
    strategy = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY

    def _axis(component_id: ComponentId) -> dict[str, object]:
        recipe = strategy.component(component_id)
        if feature_axes is None:
            return {
                "feature_axis_hash": recipe.feature_axis_hash,
                "feature_count": recipe.feature_count,
            }
        axis = tuple(feature_axes[component_id])
        return {
            "feature_axis_hash": canonical_hash(list(axis)),
            "feature_count": len(axis),
            "ordered_feature_ids": axis,
        }

    return HeterogeneousModelSetAuthority.create(
        strategy_hash=strategy.strategy_hash,
        closure_manifest_hash=closure_manifest_hash,
        closure_receipt_hash=closure_receipt_hash,
        numerical_environment_hash=numerical_environment_hash,
        live_vintages=live_vintages,
        formation_sessions=formation_sessions,
        listing_axis_hash=canonical_hash(list(ordered_listing_ids)),
        components=tuple(
            HeterogeneousComponentModelSet.create(
                component_id=component_id,
                recipe_hash=strategy.component(component_id).recipe_hash,
                **_axis(component_id),
                seeds=strategy.component(component_id).seeds,
                vintage_count=strategy.component(component_id).vintage_count,
                vintage_weights=strategy.component(component_id).vintage_weights,
                candidate_semantics=strategy.component(component_id).candidate_semantics,
                score_aggregation=strategy.component(component_id).score_aggregation,
                children=tuple(
                    HeterogeneousChildModelIdentity(
                        vintage=vintage,
                        seed=seed,
                        recipe_hash=recipe_hash,
                        content_hash=content_hash,
                        lineage_hash=lineage_hash,
                    )
                    for vintage, seed, recipe_hash, content_hash, lineage_hash in children[
                        component_id
                    ]
                ),
            )
            for component_id in COMPONENT_IDS
        ),
        surface_bindings=surface_bindings,
        momentum_source_identity_hash=momentum_source_identity_hash,
        momentum_value_bindings=momentum_value_bindings,
    )


__all__ = [
    "COMPONENT_IDS",
    "HETEROGENEOUS_STRATEGY_ID",
    "INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY",
    "LIVE_COMPONENT_MODEL_SET_HASHES",
    "LIVE_MODEL_CLOSURE_MANIFEST_HASH",
    "LIVE_MODEL_VINTAGES",
    "LIVE_SCORE_CLOSURE_RECEIPT_HASH",
    "LIVE_SOURCE_READBACK_SHA256",
    "STRATEGY_ROLE",
    "SUCCESSOR_PACKAGE_HASH",
    "ComponentId",
    "HeterogeneousAlphaComponentRecipe",
    "HeterogeneousAlphaError",
    "HeterogeneousAlphaStrategyRecipe",
    "HeterogeneousChildModelIdentity",
    "HeterogeneousComponentModelSet",
    "HeterogeneousLaneValueBinding",
    "HeterogeneousLiveScoreClosureReceipt",
    "HeterogeneousModelSetAuthority",
    "HeterogeneousSurfaceBinding",
    "ScoreAggregationSemantics",
    "array_value_hash",
    "component_recipe_is_current",
    "component_recipe_role",
    "heterogeneous_model_set_authority",
    "is_installed_component",
    "live_model_set_hash",
    "strategy_recipe_is_current",
]
