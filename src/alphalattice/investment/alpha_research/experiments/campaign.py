"""Generic development Program and evidence for the Factor-to-Alpha Campaign."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from alphalattice.capabilities.alpha_modeling.adapters.hist_gradient_boosting import (
    HistGradientBoostingParameters,
    build_hist_gradient_boosting_recipe,
    build_hist_gradient_boosting_search_domain,
)
from alphalattice.capabilities.alpha_modeling.adapters.huber import (
    HuberParameters,
    build_huber_recipe,
    build_huber_search_domain,
)
from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_chronological import (
    ChronologicalLightGBMParameters,
    RegularizedChronologicalLightGBMParameters,
    build_chronological_lightgbm_recipe,
    build_chronological_lightgbm_search_domain,
    build_regularized_chronological_lightgbm_recipe,
    build_regularized_chronological_lightgbm_search_domain,
)
from alphalattice.capabilities.alpha_modeling.adapters.rank_composite import (
    RankCompositeParameters,
    build_rank_composite_recipe,
    build_rank_composite_search_domain,
)
from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear import (
    RegularizedLinearParameters,
    build_regularized_linear_recipe,
    build_regularized_linear_search_domain,
)
from alphalattice.capabilities.alpha_modeling.catalog import (
    AlphaModelCatalog,
    AlphaModelCatalogBinding,
)
from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
)
from alphalattice.investment.alpha_research.scaling.catalog import CrossSectionalScaleCatalog
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import AuthoringError
from alphalattice.protocols.research_authoring.selection import (
    load_safe_yaml_document,
    require_selection_only_document,
)

from ..evaluation.metrics import TURNOVER_IC_COST

type ModelMethodId = Literal[
    "EQUAL_WEIGHT_RANK_COMPOSITE",
    "RIDGE",
    "LASSO",
    "ELASTIC_NET",
    "STATE_INTERACTION_LINEAR",
    "LIGHTGBM",
    "LIGHTGBM_REGULARIZED",
    "HIST_GRADIENT_BOOSTING",
    "HUBER_LINEAR",
]
type ScaleMethodId = Literal[
    "LAGGED_XS_DISPERSION",
    "EWMA_XS_DISPERSION",
    "ASYMMETRIC_EWMA_XS_DISPERSION",
    "HAR_XS_DISPERSION",
]


class AlphaCampaignBoundaryError(ValueError):
    """Stable pre-fit refusal for wrong Campaign authority."""


PRIMARY_MODEL_METHOD_IDS: Final[tuple[ModelMethodId, ...]] = (
    # The null control leads the canonical order because it is the incumbent the
    # rest are measured against, not the weakest of several competitors.
    "EQUAL_WEIGHT_RANK_COMPOSITE",
    "RIDGE",
    "LASSO",
    "ELASTIC_NET",
    "STATE_INTERACTION_LINEAR",
    "LIGHTGBM",
    "LIGHTGBM_REGULARIZED",
)
"""Installed primary methods, in the canonical order trials are indexed in."""

CONDITIONAL_MODEL_METHOD_IDS: Final[tuple[ModelMethodId, ...]] = (
    "HIST_GRADIENT_BOOSTING",
    "HUBER_LINEAR",
)

MODEL_METHOD_SEARCH_DOMAIN_IDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "EQUAL_WEIGHT_RANK_COMPOSITE": "RANK_COMPOSITE_BOUNDED_GRID",
        "RIDGE": "REGULARIZED_LINEAR_BOUNDED_GRID",
        "LASSO": "REGULARIZED_LINEAR_BOUNDED_GRID",
        "ELASTIC_NET": "REGULARIZED_LINEAR_BOUNDED_GRID",
        "STATE_INTERACTION_LINEAR": "REGULARIZED_LINEAR_BOUNDED_GRID",
        "LIGHTGBM": "CHRONOLOGICAL_LIGHTGBM_BOUNDED_GRID",
        "LIGHTGBM_REGULARIZED": "CHRONOLOGICAL_LIGHTGBM_REGULARIZED_GRID",
        "HIST_GRADIENT_BOOSTING": "HIST_GRADIENT_BOOSTING_CONDITIONAL_GRID",
        "HUBER_LINEAR": "HUBER_CONDITIONAL_GRID",
    }
)
"""Which installed search domain each installed method requires.

A selection must name exactly the domains its methods need: a missing domain
would let a method run unbounded, and an unrelated one would seal authority for
a method the experiment never selected.
"""

SCALE_METHOD_SEARCH_DOMAIN_IDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "LAGGED_XS_DISPERSION": "LAGGED_CONTROL",
        "EWMA_XS_DISPERSION": "EWMA_HALF_LIVES_10_21_42",
        "ASYMMETRIC_EWMA_XS_DISPERSION": "ASYMMETRIC_EWMA_RISE_10_DECAY_42",
        "HAR_XS_DISPERSION": "HAR_MINIMUM_63_RIDGE_RATIO_1E_6",
    }
)

REQUIRED_SCALE_METHOD_IDS: Final[frozenset[str]] = frozenset(
    {
        "LAGGED_XS_DISPERSION",
        "EWMA_XS_DISPERSION",
        "ASYMMETRIC_EWMA_XS_DISPERSION",
    }
)
"""The scientific minimum the installed scale comparison itself requires.

``compare_cross_sectional_scale_methods`` refuses a comparison that is missing
any of these three primaries, so a selection without them could not produce
scale evidence at all. HAR remains genuinely optional and conditional.
"""

STATE_INTERACTION_FACTOR_IDS: Final[tuple[str, ...]] = (
    "market_drawdown_x_momentum",
    "market_vol_ratio_x_reversal",
)
"""The Factors whose presence makes a state-interaction view possible."""


def derive_state_interaction_axis(
    ordered_factor_ids: tuple[str, ...], *, selected_method_ids: Sequence[str]
) -> tuple[str, ...]:
    """The interaction feature view this selection and this axis actually admit.

    Empty when the method was not selected, and equally empty when it was
    selected but curation admitted no interaction Factor -- the view is a
    property of the data, not of the request. Shared by the compiler and by
    replay so both derive one axis rather than agreeing by coincidence.
    """

    if "STATE_INTERACTION_LINEAR" not in set(selected_method_ids):
        return ()
    admitted = set(STATE_INTERACTION_FACTOR_IDS)
    return tuple(value for value in ordered_factor_ids if value in admitted)


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


type AlphaInnerSelectionRuleId = Literal[
    "MINIMUM_INNER_MSE_THEN_METHOD_ID",
    "MAXIMUM_INNER_NET_DAILY_RANK_IC_THEN_METHOD_ID",
]
"""The installed per-fold inner selection rules.

``MINIMUM_INNER_MSE_THEN_METHOD_ID`` is the original. On a per-day standardized
target ``MSE = E[y^2](1 - rho^2) + (c - rho)^2``, so at the correlations this
programme sees its differences are dominated by the scale term -- which the
downstream calibration re-fits and discards. The successor rule judges ordering
directly, net of what holding the resulting book costs.
"""

INNER_DISCRIMINATION_FLOOR: Final = 4
"""Admissible only when non-constant on at least four fifths of scored sessions.

The turnover penalty can exceed the signal, so a heavily shrunk near-constant
predictor could otherwise win the successor rule on low turnover alone.
"""


class AlphaDevelopmentRequest(_Contract):
    """Actor intent only: installed methods and bounded scientific scope."""

    kind: Literal["AlphaDevelopmentRequest"] = "AlphaDevelopmentRequest"
    feature_preprocessing_method_id: Literal[
        "ROBUST_SECTOR_NEUTRAL_Z",
        "JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z",
    ] = "ROBUST_SECTOR_NEUTRAL_Z"
    target_method_id: Literal[
        "SECTOR_RESIDUAL_CROSS_SECTIONAL_STD_Z",
        "UNIVERSE_BOUND_SECTOR_RESIDUAL_STD_Z",
    ] = "SECTOR_RESIDUAL_CROSS_SECTIONAL_STD_Z"
    context_axis_method_id: Literal["MARKET_REGIME_STATE_CONTEXT"] | None = None
    model_method_ids: tuple[ModelMethodId, ...] = Field(min_length=1)
    model_search_domain_ids: tuple[
        Literal[
            "RANK_COMPOSITE_BOUNDED_GRID",
            "REGULARIZED_LINEAR_BOUNDED_GRID",
            "CHRONOLOGICAL_LIGHTGBM_BOUNDED_GRID",
            "CHRONOLOGICAL_LIGHTGBM_REGULARIZED_GRID",
            "HIST_GRADIENT_BOOSTING_CONDITIONAL_GRID",
            "HUBER_CONDITIONAL_GRID",
        ],
        ...,
    ] = Field(min_length=1)
    scale_method_ids: tuple[ScaleMethodId, ...] = Field(min_length=3)
    scale_search_domain_ids: tuple[
        Literal[
            "LAGGED_CONTROL",
            "EWMA_HALF_LIVES_10_21_42",
            "ASYMMETRIC_EWMA_RISE_10_DECAY_42",
            "HAR_MINIMUM_63_RIDGE_RATIO_1E_6",
        ],
        ...,
    ] = Field(min_length=3)
    target_horizon_sessions: tuple[Literal[1, 5], ...] = (1, 5)
    calibration_method_id: Literal["NORMALIZED_MOMENT_NONNEGATIVE_SLOPE"] = (
        "NORMALIZED_MOMENT_NONNEGATIVE_SLOPE"
    )
    blend_method_id: Literal["FAST_SLOW_STOCK_SIGNAL_BLEND"] = "FAST_SLOW_STOCK_SIGNAL_BLEND"
    blend_fast_weight_domain: tuple[float, ...] = (0.25, 0.5, 0.75)
    inner_selection_rule_id: AlphaInnerSelectionRuleId = "MINIMUM_INNER_MSE_THEN_METHOD_ID"

    @property
    def selected_model_method_ids(self) -> tuple[ModelMethodId, ...]:
        """The selection in canonical installed order, primaries before conditionals."""

        selected = set(self.model_method_ids)
        return tuple(
            value
            for value in (*PRIMARY_MODEL_METHOD_IDS, *CONDITIONAL_MODEL_METHOD_IDS)
            if value in selected
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_request(self) -> Self:
        """Admit any installed subset whose declared domains exactly match it.

        A selection is a scientific statement, so the domains it names must be
        exactly the ones its methods need. A missing domain would let a method
        run against no declared bound; an unrelated one would seal authority for
        a method this experiment never selected. Scale selection is a subset too,
        but the installed comparison itself refuses fewer than its three
        primaries, so those stay required and only HAR is optional.
        """

        required_model_domains = {
            MODEL_METHOD_SEARCH_DOMAIN_IDS[value] for value in self.model_method_ids
        }
        required_scale_domains = {
            SCALE_METHOD_SEARCH_DOMAIN_IDS[value] for value in self.scale_method_ids
        }
        if (
            self.model_method_ids != tuple(dict.fromkeys(self.model_method_ids))
            or self.model_search_domain_ids != tuple(dict.fromkeys(self.model_search_domain_ids))
            or self.scale_method_ids != tuple(dict.fromkeys(self.scale_method_ids))
            or self.scale_search_domain_ids != tuple(dict.fromkeys(self.scale_search_domain_ids))
            or self.target_horizon_sessions != (1, 5)
            or set(self.model_search_domain_ids) != required_model_domains
            or set(self.scale_search_domain_ids) != required_scale_domains
            or not REQUIRED_SCALE_METHOD_IDS.issubset(self.scale_method_ids)
        ):
            raise AlphaCampaignBoundaryError("ALPHA_DEVELOPMENT_REQUEST_INVALID")
        return self

    @classmethod
    def from_yaml(cls, source: str) -> Self:
        """Parse selections only; authority-bearing keys are refused before validation.

        YAML selects installed methods, domains and scope. It must never carry
        trusted hashes, artifact URIs, Python import paths, executors, or
        publication authority -- the Host resolves every identity from the
        installed catalogs after this boundary.
        """

        # The one declaration loader: a key written twice is refused.
        loaded = load_safe_yaml_document(source)
        if not isinstance(loaded, dict):
            raise AlphaCampaignBoundaryError("ALPHA_DEVELOPMENT_REQUEST_YAML_INVALID")
        # One owner, not a set kept here. The local set scanned top-level keys
        # only, which is a real gap rather than a stylistic one: nesting an
        # identity one level down under a list defeats it entirely, and that is
        # how six of them survived in a committed Portfolio request. The shared
        # rule walks mappings and sequences at every depth and checks values as
        # well as names.
        try:
            require_selection_only_document(loaded)
        except AuthoringError as error:
            raise AlphaCampaignBoundaryError(
                f"ALPHA_DEVELOPMENT_REQUEST_AUTHORITY_FORBIDDEN:{error}"
            ) from error
        return cast(Self, cls.model_validate(loaded))


class AlphaHorizonAuthority(_Contract):
    horizon_sessions: Literal[1, 5]
    target_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_recipe_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_method_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    maturity_lag_sessions: int = Field(ge=2)
    embargo_sessions: int = Field(ge=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_clock(self) -> Self:
        expected_lag = 2 if self.horizon_sessions == 1 else 6
        if self.maturity_lag_sessions != expected_lag or self.embargo_sessions != expected_lag - 1:
            raise AlphaCampaignBoundaryError("ALPHA_DEVELOPMENT_HORIZON_CLOCK_INVALID")
        return self


class AlphaTrialPlan(_Contract):
    trial_index: int = Field(ge=0)
    method_id: str
    feature_view: Literal["FULL_AXIS", "STATE_INTERACTION_VIEW", "FACTOR_AND_REGIME_CONTEXT"] = (
        "FULL_AXIS"
    )
    ordered_feature_ids: tuple[str, ...] = Field(min_length=1)
    ordered_feature_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipe: AlphaModelRecipeEnvelope
    search_domain: AlphaModelSearchDomainEnvelope
    trigger_disposition: Literal["REQUIRED", "TRIGGERED", "NOT_TRIGGERED"] = "REQUIRED"
    trial_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> Self:
        payload = {
            "feature_view": "FULL_AXIS",
            "trigger_disposition": "REQUIRED",
            **values,
        }
        identity = cls.model_construct(**payload, trial_hash="0" * 64).model_dump(
            mode="json", exclude={"trial_hash"}
        )
        return cls(**payload, trial_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        identity = self.model_dump(mode="json", exclude={"trial_hash"})
        if self.ordered_feature_axis_hash != canonical_hash(
            self.ordered_feature_ids
        ) or self.trial_hash != canonical_hash(identity):
            raise AlphaCampaignBoundaryError("ALPHA_DEVELOPMENT_TRIAL_IDENTITY_INVALID")
        return self


class AlphaConditionalMethodPlan(_Contract):
    method_id: Literal["HIST_GRADIENT_BOOSTING", "HUBER_LINEAR"]
    ordered_feature_ids: tuple[str, ...] = Field(min_length=1)
    ordered_feature_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipe: AlphaModelRecipeEnvelope
    search_domain: AlphaModelSearchDomainEnvelope
    trigger_rule: str = Field(min_length=1)
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> Self:
        identity = cls.model_construct(**values, plan_hash="0" * 64).model_dump(
            mode="json", exclude={"plan_hash"}
        )
        return cls(**values, plan_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.ordered_feature_axis_hash != canonical_hash(
            self.ordered_feature_ids
        ) or self.plan_hash != canonical_hash(self.model_dump(mode="json", exclude={"plan_hash"})):
            raise AlphaCampaignBoundaryError("ALPHA_CONDITIONAL_METHOD_PLAN_INVALID")
        return self


class AlphaDevelopmentProgram(_Contract):
    """Host-sealed successor Program; frozen 30-trial contracts remain unchanged."""

    @model_serializer(mode="wrap")  # type: ignore[untyped-decorator]
    def _serialize_populated(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Serialize what the Program asserts, matching how it seals itself.

        ``program_hash`` has always been taken over the populated payload, so a
        dump that carried the nulls would not re-derive it. That only became
        observable once a member existed that a Program may legitimately leave
        unset -- the durable store re-derives identity from the serialized form,
        and would have refused every Program that declined the context lane.
        """

        serialized: dict[str, object] = handler(self)
        return {key: value for key, value in serialized.items() if value is not None}

    kind: Literal["AlphaDevelopmentCampaignProgram"] = "AlphaDevelopmentCampaignProgram"
    factor_checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_preprocessing_method_id: str | None = None
    target_method_id: str | None = None
    ordered_factor_ids: tuple[str, ...] = Field(min_length=1)
    ordered_factor_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_axis_method_id: str | None = None
    ordered_context_ids: tuple[str, ...] | None = None
    context_source_hash: str | None = None
    horizons: tuple[AlphaHorizonAuthority, AlphaHorizonAuthority]
    catalog_binding: AlphaModelCatalogBinding
    model_method_ids: tuple[str, ...] = Field(min_length=1)
    scale_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scale_method_ids: tuple[str, ...] = Field(min_length=3)
    scale_parameter_domain: dict[str, object]
    ordered_trials: tuple[AlphaTrialPlan, ...] = Field(min_length=1)
    inner_selection_policy: dict[str, object]
    conditional_method_plans: tuple[AlphaConditionalMethodPlan, ...] = Field(max_length=2)
    conditional_method_triggers: dict[str, str]
    calibration_method_id: str
    blend_method_id: str
    blend_fast_weight_domain: tuple[float, ...]
    refit_policy_id: Literal["EVERY_21_FORMATIONS"] = "EVERY_21_FORMATIONS"
    network_policy: Literal["DISABLED"] = "DISABLED"
    evidence_scope: Literal["DEVELOPMENT_ONLY_NO_HOLDOUT"] = "DEVELOPMENT_ONLY_NO_HOLDOUT"
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """The sealed surface must be exactly the selection, at its approved size.

        Trial counts are derived from the selection rather than asserted as one
        total, so a reduced experiment is admissible and a *silently* reduced
        one is not: every selected primary must contribute its full approved
        grid, and no method outside the selection may contribute a plan at all.
        """

        method_counts: dict[str, int] = {}
        for value in self.ordered_trials:
            method_counts[value.method_id] = method_counts.get(value.method_id, 0) + 1
        selected = tuple(
            value
            for value in (*PRIMARY_MODEL_METHOD_IDS, *CONDITIONAL_MODEL_METHOD_IDS)
            if value in set(self.model_method_ids)
        )
        expected_counts = {
            method_id: APPROVED_TRIAL_COUNTS[method_id]
            for method_id in selected
            if method_id in APPROVED_TRIAL_COUNTS
        }
        if (
            tuple(value.horizon_sessions for value in self.horizons) != (1, 5)
            or self.model_method_ids != selected
            or tuple(value.trial_index for value in self.ordered_trials)
            != tuple(range(len(self.ordered_trials)))
            or method_counts != expected_counts
            or tuple(value.method_id for value in self.conditional_method_plans)
            != tuple(value for value in CONDITIONAL_MODEL_METHOD_IDS if value in selected)
            or self.ordered_factor_axis_hash != canonical_hash(self.ordered_factor_ids)
            or len(
                {
                    self.context_axis_method_id is None,
                    self.ordered_context_ids is None,
                    self.context_source_hash is None,
                }
            )
            != 1
            or bool(set(self.ordered_context_ids or ()) & set(self.ordered_factor_ids))
            or (self.ordered_context_ids is not None and not self.ordered_context_ids)
            or self.program_hash
            != canonical_hash(
                self.model_dump(mode="json", exclude={"program_hash"}, exclude_none=True)
            )
        ):
            raise AlphaCampaignBoundaryError("ALPHA_DEVELOPMENT_PROGRAM_INVALID")
        return self


APPROVED_RIDGE_ALPHAS: Final = (0.1, 1.0, 10.0, 100.0)
APPROVED_SPARSE_ALPHA_MAX_MULTIPLIERS: Final = (0.02, 0.05, 0.2, 0.5)
APPROVED_ELASTIC_NET_L1_RATIOS: Final = (0.35, 0.5, 0.65)
APPROVED_LIGHTGBM_SEEDS: Final = (1729, 2718, 31415)
APPROVED_LIGHTGBM_CONFIGURATIONS: Final = (
    # The full installed lattice over the two slower learning rates ...
    *(
        (num_leaves, learning_rate, max_depth, min_child_samples)
        for num_leaves in (15, 31)
        for learning_rate in (0.03, 0.05)
        for max_depth in (3, 5)
        for min_child_samples in (20, 50)
    ),
    # ... plus the four fast-rate corners at the legacy-precedent 0.1 rate.
    *((num_leaves, 0.1, max_depth, 20) for num_leaves in (15, 31) for max_depth in (3, 5)),
)
APPROVED_REGULARIZED_CAPACITY_PROFILES: Final = ((3, 8), (5, 31))
"""The two genuinely distinct tree capacities, as ``(max_depth, num_leaves)``.

Paired rather than crossed because ``max_depth=3`` caps a tree at eight leaves,
so a crossed grid spent a quarter of its budget re-fitting the same model under
two different leaf counts. Spelling the cap here makes the de-duplication a
property of the declared domain rather than a convention LightGBM applies
silently.
"""

APPROVED_REGULARIZED_FEATURE_FRACTIONS: Final = (0.5, 1.0)
APPROVED_REGULARIZED_BAGGING_PROFILES: Final = ((1.0, 0), (0.5, 1))
"""``(bagging_fraction, bagging_freq)``, paired: a fraction at frequency zero is
a no-op, which is exactly the defect this axis exists to remove."""

APPROVED_REGULARIZED_LIGHTGBM_PROFILES: Final = (
    (0.0, 0.0),
    (0.0, 1000.0),
    (0.0, 10000.0),
)
"""``(lambda_l1, lambda_l2)`` scaled to the leaf hessian mass this data produces.

Under L2 loss each row contributes a hessian of one, so a leaf output is
``-G / (n_leaf + lambda_l2)``. At roughly nine thousand rows per leaf the former
grid's ``lambda_l2 <= 10`` shrank by a tenth of a percent; these give nothing,
a tenth, and a half. ``lambda_l1`` soft-thresholds ``G``, whose scale is
``O(sqrt(n_leaf)) ~ 95``, so the former ``<= 10`` was noise there too and it is
pinned off rather than left as decorative variation.
"""
"""The approved 20 deterministic LightGBM configurations, each run at 3 seeds."""

APPROVED_RANK_COMPOSITE_SELECTION_COUNTS: Final = (3, 5, 10, 20)
"""How many Factors the null control averages.

A count is searched because the evidence says the choice matters and the model
does not make it: equal-weighting every Factor scores `-0.006121` net rank IC
out of sample where the best three score `+0.003392`. Searching it keeps the
control honest without letting anything be *fitted*.
"""

APPROVED_TRIAL_COUNTS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "EQUAL_WEIGHT_RANK_COMPOSITE": len(APPROVED_RANK_COMPOSITE_SELECTION_COUNTS),
        "RIDGE": len(APPROVED_RIDGE_ALPHAS),
        "LASSO": len(APPROVED_SPARSE_ALPHA_MAX_MULTIPLIERS),
        "ELASTIC_NET": (
            len(APPROVED_SPARSE_ALPHA_MAX_MULTIPLIERS) * len(APPROVED_ELASTIC_NET_L1_RATIOS)
        ),
        "STATE_INTERACTION_LINEAR": 1,
        "LIGHTGBM": len(APPROVED_LIGHTGBM_CONFIGURATIONS) * len(APPROVED_LIGHTGBM_SEEDS),
        "LIGHTGBM_REGULARIZED": (
            len(APPROVED_REGULARIZED_CAPACITY_PROFILES)
            * len(APPROVED_REGULARIZED_FEATURE_FRACTIONS)
            * len(APPROVED_REGULARIZED_BAGGING_PROFILES)
            * len(APPROVED_REGULARIZED_LIGHTGBM_PROFILES)
            * len(APPROVED_LIGHTGBM_SEEDS)
        ),
    }
)
"""How many trials each selected primary method must contribute, in full."""


def _trial_plans(
    catalog: AlphaModelCatalog,
    *,
    ordered_factor_ids: tuple[str, ...],
    selected_method_ids: tuple[str, ...],
    state_interaction_factor_ids: tuple[str, ...],
    ordered_context_ids: tuple[str, ...] = (),
) -> tuple[AlphaTrialPlan, ...]:
    """Seal the approved surface for the selected methods, never a silent cut.

    A method the experiment did not select contributes nothing; a method it did
    select contributes its complete approved grid -- Ridge over its four
    approved alphas, Lasso over the four approved regularization multipliers,
    Elastic Net over those multipliers crossed with the three approved
    ``l1_ratio`` values, the state-interaction linear candidate when its feature
    view exists, and the twenty approved deterministic LightGBM configurations
    at three seeds each.
    """

    selected = set(selected_method_ids)
    plans: list[AlphaTrialPlan] = []
    if "EQUAL_WEIGHT_RANK_COMPOSITE" in selected:
        composite_domain = build_rank_composite_search_domain()
        for count in APPROVED_RANK_COMPOSITE_SELECTION_COUNTS:
            recipe = build_rank_composite_recipe(
                RankCompositeParameters(selected_factor_count=count)
            )
            catalog.admit_recipe(recipe=recipe, domain=composite_domain)
            plans.append(
                AlphaTrialPlan.create(
                    trial_index=len(plans),
                    method_id="EQUAL_WEIGHT_RANK_COMPOSITE",
                    ordered_feature_ids=ordered_factor_ids,
                    ordered_feature_axis_hash=canonical_hash(ordered_factor_ids),
                    recipe=recipe,
                    search_domain=composite_domain,
                )
            )
    linear_domain = build_regularized_linear_search_domain()
    linear: list[tuple[str, str, RegularizedLinearParameters]] = []
    if "RIDGE" in selected:
        linear.extend(
            ("RIDGE", "FULL_AXIS", RegularizedLinearParameters(family="ridge", alpha=alpha))
            for alpha in APPROVED_RIDGE_ALPHAS
        )
    if "LASSO" in selected:
        linear.extend(
            (
                "LASSO",
                "FULL_AXIS",
                RegularizedLinearParameters(family="lasso", alpha_max_multiplier=multiplier),
            )
            for multiplier in APPROVED_SPARSE_ALPHA_MAX_MULTIPLIERS
        )
    if "ELASTIC_NET" in selected:
        linear.extend(
            (
                "ELASTIC_NET",
                "FULL_AXIS",
                RegularizedLinearParameters(
                    family="elastic_net", alpha_max_multiplier=multiplier, l1_ratio=l1_ratio
                ),
            )
            for multiplier in APPROVED_SPARSE_ALPHA_MAX_MULTIPLIERS
            for l1_ratio in APPROVED_ELASTIC_NET_L1_RATIOS
        )
    if "STATE_INTERACTION_LINEAR" in selected and state_interaction_factor_ids:
        linear.append(
            (
                "STATE_INTERACTION_LINEAR",
                "STATE_INTERACTION_VIEW",
                RegularizedLinearParameters(family="ridge", alpha=10.0),
            )
        )
    # Day-constant state cannot move a linear score's within-day ordering: on any
    # session it contributes the same amount to every listing, which a ranking
    # discards. Only a method that can *split* on it -- change how the Factors are
    # read once the state crosses a threshold -- can use it at all, so the axis is
    # offered to the tree methods and withheld from the linear ones rather than
    # spending coefficients to prove the point.
    tree_axis = (*ordered_factor_ids, *ordered_context_ids)
    tree_view = "FACTOR_AND_REGIME_CONTEXT" if ordered_context_ids else "FULL_AXIS"
    for method, feature_view, parameters in linear:
        recipe = build_regularized_linear_recipe(parameters)
        catalog.admit_recipe(recipe=recipe, domain=linear_domain)
        trial_axis = (
            ordered_factor_ids if feature_view == "FULL_AXIS" else state_interaction_factor_ids
        )
        plans.append(
            AlphaTrialPlan.create(
                trial_index=len(plans),
                method_id=method,
                feature_view=feature_view,
                ordered_feature_ids=trial_axis,
                ordered_feature_axis_hash=canonical_hash(trial_axis),
                recipe=recipe,
                search_domain=linear_domain,
            )
        )
    if "LIGHTGBM" not in selected and "LIGHTGBM_REGULARIZED" not in selected:
        return tuple(plans)
    if "LIGHTGBM" in selected:
        nonlinear_domain = build_chronological_lightgbm_search_domain()
        combinations = tuple(
            ChronologicalLightGBMParameters(
                seed=seed,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_samples=min_child_samples,
            )
            for num_leaves, learning_rate, max_depth, min_child_samples in (
                APPROVED_LIGHTGBM_CONFIGURATIONS
            )
            for seed in APPROVED_LIGHTGBM_SEEDS
        )
        for parameters in combinations:
            recipe = build_chronological_lightgbm_recipe(parameters)
            catalog.admit_recipe(recipe=recipe, domain=nonlinear_domain)
            plans.append(
                AlphaTrialPlan.create(
                    trial_index=len(plans),
                    method_id="LIGHTGBM",
                    feature_view=tree_view,
                    ordered_feature_ids=tree_axis,
                    ordered_feature_axis_hash=canonical_hash(tree_axis),
                    recipe=recipe,
                    search_domain=nonlinear_domain,
                )
            )
    if "LIGHTGBM_REGULARIZED" in selected:
        regularized_domain = build_regularized_chronological_lightgbm_search_domain()
        regularized = tuple(
            RegularizedChronologicalLightGBMParameters(
                seed=seed,
                num_leaves=num_leaves,
                learning_rate=0.03,
                max_depth=max_depth,
                min_child_samples=50,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                feature_fraction=feature_fraction,
                bagging_fraction=bagging_fraction,
                bagging_freq=bagging_freq,
            )
            for max_depth, num_leaves in APPROVED_REGULARIZED_CAPACITY_PROFILES
            for feature_fraction in APPROVED_REGULARIZED_FEATURE_FRACTIONS
            for bagging_fraction, bagging_freq in APPROVED_REGULARIZED_BAGGING_PROFILES
            for lambda_l1, lambda_l2 in APPROVED_REGULARIZED_LIGHTGBM_PROFILES
            for seed in APPROVED_LIGHTGBM_SEEDS
        )
        for parameters in regularized:
            recipe = build_regularized_chronological_lightgbm_recipe(parameters)
            catalog.admit_recipe(recipe=recipe, domain=regularized_domain)
            plans.append(
                AlphaTrialPlan.create(
                    trial_index=len(plans),
                    method_id="LIGHTGBM_REGULARIZED",
                    feature_view=tree_view,
                    ordered_feature_ids=tree_axis,
                    ordered_feature_axis_hash=canonical_hash(tree_axis),
                    recipe=recipe,
                    search_domain=regularized_domain,
                )
            )
    return tuple(plans)


def compile_alpha_development_program(
    *,
    request: AlphaDevelopmentRequest,
    factor_checkpoint_hash: str,
    feature_source_hash: str,
    ordered_factor_ids: tuple[str, ...],
    horizon_authority: Mapping[int, AlphaHorizonAuthority],
    model_catalog: AlphaModelCatalog,
    scale_catalog: CrossSectionalScaleCatalog,
    ordered_context_ids: tuple[str, ...] = (),
    context_source_hash: str | None = None,
) -> AlphaDevelopmentProgram:
    """Resolve all caller intent against installed Host authority before fitting.

    The compiled Program contains only the selected installed methods and their
    domains. ``STATE_INTERACTION_LINEAR`` additionally requires a feature view
    the curated axis actually provides, so a selection naming it over an axis
    without interaction Factors compiles without it and records the effective
    selection rather than claiming a method it never planned.
    """

    horizons = tuple(horizon_authority[value] for value in request.target_horizon_sessions)
    if len(horizons) != 2 or any(
        value.horizon_sessions != key for key, value in zip((1, 5), horizons, strict=True)
    ):
        raise AlphaCampaignBoundaryError("ALPHA_DEVELOPMENT_TARGET_AUTHORITY_MISMATCH")
    for method in request.scale_method_ids:
        scale_catalog.resolve(method)
    requested = request.selected_model_method_ids
    state_interaction_factor_ids = derive_state_interaction_axis(
        ordered_factor_ids, selected_method_ids=requested
    )
    effective = tuple(
        value
        for value in requested
        if value != "STATE_INTERACTION_LINEAR" or state_interaction_factor_ids
    )
    if not effective:
        raise AlphaCampaignBoundaryError("ALPHA_DEVELOPMENT_MODEL_SELECTION_EMPTY")
    if bool(request.context_axis_method_id) != bool(ordered_context_ids) or (
        ordered_context_ids and context_source_hash is None
    ):
        raise AlphaCampaignBoundaryError("ALPHA_DEVELOPMENT_CONTEXT_AXIS_UNRESOLVED")
    trials = _trial_plans(
        model_catalog,
        ordered_factor_ids=ordered_factor_ids,
        selected_method_ids=effective,
        state_interaction_factor_ids=state_interaction_factor_ids,
        ordered_context_ids=ordered_context_ids,
    )
    conditional_plans: list[AlphaConditionalMethodPlan] = []
    if "HIST_GRADIENT_BOOSTING" in effective:
        hgb_domain = build_hist_gradient_boosting_search_domain()
        hgb_recipe = build_hist_gradient_boosting_recipe(
            HistGradientBoostingParameters(
                learning_rate=0.03,
                max_leaf_nodes=15,
                l2_regularization=1.0,
            )
        )
        model_catalog.admit_recipe(recipe=hgb_recipe, domain=hgb_domain)
        conditional_plans.append(
            AlphaConditionalMethodPlan.create(
                method_id="HIST_GRADIENT_BOOSTING",
                ordered_feature_ids=ordered_factor_ids,
                ordered_feature_axis_hash=canonical_hash(ordered_factor_ids),
                recipe=hgb_recipe,
                search_domain=hgb_domain,
                trigger_rule=("LIGHTGBM_MEAN_INNER_RANK_IC_GT_0_AND_NONNEGATIVE_SHARE_GE_0_60"),
            )
        )
    if "HUBER_LINEAR" in effective:
        huber_domain = build_huber_search_domain()
        huber_recipe = build_huber_recipe(HuberParameters(epsilon=1.35, alpha=1e-4))
        model_catalog.admit_recipe(recipe=huber_recipe, domain=huber_domain)
        conditional_plans.append(
            AlphaConditionalMethodPlan.create(
                method_id="HUBER_LINEAR",
                ordered_feature_ids=ordered_factor_ids,
                ordered_feature_axis_hash=canonical_hash(ordered_factor_ids),
                recipe=huber_recipe,
                search_domain=huber_domain,
                trigger_rule="RIDGE_MEAN_INNER_RESIDUAL_EXCESS_KURTOSIS_GT_3",
            )
        )
    values: dict[str, object] = {
        "kind": "AlphaDevelopmentCampaignProgram",
        "factor_checkpoint_hash": factor_checkpoint_hash,
        "feature_source_hash": feature_source_hash,
        "feature_preprocessing_method_id": request.feature_preprocessing_method_id,
        "target_method_id": request.target_method_id,
        "ordered_factor_ids": ordered_factor_ids,
        "ordered_factor_axis_hash": canonical_hash(ordered_factor_ids),
        **(
            {
                "context_axis_method_id": request.context_axis_method_id,
                "ordered_context_ids": ordered_context_ids,
                "context_source_hash": context_source_hash,
            }
            if ordered_context_ids
            else {}
        ),
        "horizons": horizons,
        "catalog_binding": model_catalog.binding,
        "model_method_ids": effective,
        "scale_catalog_hash": scale_catalog.catalog_hash,
        "scale_method_ids": request.scale_method_ids,
        "scale_parameter_domain": {
            method_id: domain
            for method_id, domain in (
                ("EWMA_XS_DISPERSION", {"half_life_sessions": (10, 21, 42)}),
                (
                    "ASYMMETRIC_EWMA_XS_DISPERSION",
                    {"rise_half_life_sessions": 10, "decay_half_life_sessions": 42},
                ),
                (
                    "HAR_XS_DISPERSION",
                    {
                        "minimum_fit_rows": 63,
                        "ridge_ratio": 1e-6,
                        "trigger": "BEST_PRIMARY_NORMALIZED_MAE_GT_0_20_AND_N_GE_252",
                    },
                ),
            )
            if method_id in request.scale_method_ids
        },
        "ordered_trials": trials,
        "inner_selection_policy": {
            "selection_evidence": "CHRONOLOGICAL_INNER_VALIDATION_ONLY",
            "inner_validation_sessions": "max(1, outer_training_sessions // 5) grown to >= 20 rows",
            "inner_purge_sessions": "horizon_sessions",
            "session_boundary_rule": "COMPLETE_SESSIONS_ONLY_NEVER_SPLIT_A_SESSION",
            "selection_rule_id": request.inner_selection_rule_id,
            "selection_metric_id": (
                "INNER_VALIDATION_MSE"
                if request.inner_selection_rule_id == "MINIMUM_INNER_MSE_THEN_METHOD_ID"
                else "INNER_TURNOVER_NET_DAILY_RANK_IC"
            ),
            "turnover_ic_cost_by_horizon": {
                str(horizon): cost for horizon, cost in sorted(TURNOVER_IC_COST.items())
            },
            "inner_discrimination_floor": (
                f"NON_CONSTANT_ON_AT_LEAST_{INNER_DISCRIMINATION_FLOOR}_OF_"
                f"{INNER_DISCRIMINATION_FLOOR + 1}_SCORED_SESSIONS"
            ),
            "lightgbm_early_stopping_split": "SESSION_CAUSAL_WITHIN_TRAINING_PORTION",
            "lightgbm_configuration_rule": "ONE_STANDARD_ERROR_THEN_COMPLEXITY_OVER_SEED_MEANS",
            "cross_method_rule": request.inner_selection_rule_id,
            "outer_refit": "SELECTED_METHOD_REFIT_ON_COMPLETE_OUTER_TRAINING",
            "outer_validation_predictions_per_fold": 1,
        },
        "conditional_method_plans": tuple(conditional_plans),
        "conditional_method_triggers": {
            method_id: rule
            for method_id, rule in (
                (
                    "HIST_GRADIENT_BOOSTING",
                    "LIGHTGBM_MEAN_INNER_RANK_IC_GT_0_AND_NONNEGATIVE_SHARE_GE_0_60",
                ),
                ("HUBER_LINEAR", "RIDGE_MEAN_INNER_RESIDUAL_EXCESS_KURTOSIS_GT_3"),
                ("HAR_XS_DISPERSION", "BEST_PRIMARY_NORMALIZED_MAE_GT_0_20_AND_N_GE_252"),
            )
            if method_id in effective or method_id in request.scale_method_ids
        },
        "calibration_method_id": request.calibration_method_id,
        "blend_method_id": request.blend_method_id,
        "blend_fast_weight_domain": request.blend_fast_weight_domain,
        "refit_policy_id": "EVERY_21_FORMATIONS",
        "network_policy": "DISABLED",
        "evidence_scope": "DEVELOPMENT_ONLY_NO_HOLDOUT",
    }
    identity = AlphaDevelopmentProgram.model_construct(**values, program_hash="0" * 64).model_dump(
        mode="json", exclude={"program_hash"}, exclude_none=True
    )
    return AlphaDevelopmentProgram(**values, program_hash=canonical_hash(identity))


__all__ = [
    "APPROVED_ELASTIC_NET_L1_RATIOS",
    "APPROVED_LIGHTGBM_CONFIGURATIONS",
    "APPROVED_LIGHTGBM_SEEDS",
    "APPROVED_RIDGE_ALPHAS",
    "APPROVED_SPARSE_ALPHA_MAX_MULTIPLIERS",
    "APPROVED_TRIAL_COUNTS",
    "CONDITIONAL_MODEL_METHOD_IDS",
    "MODEL_METHOD_SEARCH_DOMAIN_IDS",
    "PRIMARY_MODEL_METHOD_IDS",
    "REQUIRED_SCALE_METHOD_IDS",
    "SCALE_METHOD_SEARCH_DOMAIN_IDS",
    "STATE_INTERACTION_FACTOR_IDS",
    "AlphaCampaignBoundaryError",
    "AlphaConditionalMethodPlan",
    "AlphaDevelopmentProgram",
    "AlphaDevelopmentRequest",
    "AlphaHorizonAuthority",
    "AlphaTrialPlan",
    "compile_alpha_development_program",
    "derive_state_interaction_axis",
]
