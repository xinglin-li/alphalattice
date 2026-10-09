"""Generic authoring for installed paired Panel research methodologies.

This extends the existing ``alpha.model-development`` Desk kind.  The Dynamic
Panel experiment is one installed Feature recipe selection, not a request type
or Host branch.  Compilation seals descriptors, domains, authority and an
already-resolved common-axis preflight; it performs no fit, predict, metric or
solver call.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
    build_dynamic_panel_lightgbm_search_domain,
)
from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear_dynamic_panel import (
    build_dynamic_panel_regularized_linear_search_domain,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_statistics import (
    fixed_recipe_economic_metric_call_count,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (
    PanelFeaturePlan,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
    INSTALLED_PANEL_VIEW_IDS,
    SPARSE_SESSION_AMPLITUDE_VIEW_IDS,
    InstalledPanelViewId,
)
from alphalattice.investment.alpha_research.scores.product_lifecycle import (
    lifecycle_implementation_hash,
)
from alphalattice.investment.alpha_research.scores.score_filters import (
    CROSS_SECTION_STANDARDIZED_SCORE_INPUT_MODE,
    RAW_SCORE_INPUT_MODE,
    RAW_SCORE_TEMPORAL_FILTER_METHOD_ID,
    AlphaScoreFilterSpec,
)
from alphalattice.investment.portfolio_management.mandates.paired_panel_grid import (
    PAIRED_PANEL_GRID_FIELDS,
    PAIRED_PANEL_PORTFOLIO_GRID,
    RANK_BUFFERED_TURNOVER_SUCCESSOR_GRID,
    PairedPanelPortfolioGrid,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.source_identity import (
    switched_source_identity,
)
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskProgramCompilation,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
)

type PanelMethodologyId = Literal[
    "SINGLE_MODEL_FIXED_AXIS",
    "PAIRED_PANEL_PORTFOLIO_RESEARCH",
    "MODEL_LIFECYCLE_REPLAY",
]
type PanelExecutionStage = Literal[
    "ALPHA_ONLY",
    "SCORE_FILTER_ONLY",
    "PORTFOLIO_ONLY",
    "END_TO_END",
]

_HASH = r"^[0-9a-f]{64}$"
type PanelModelFamilyId = Literal["RIDGE", "ELASTIC_NET", "LIGHTGBM"]
"""The estimator families this Desk installs.

Declared once so the request field and the admission set cannot drift apart.
`recipe_scope` is deliberately absent from it: scope states Feature-view breadth
and the estimator family is validated against this install declaration, so the
two combine rather than one narrowing the other.
"""

INSTALLED_PANEL_MODEL_FAMILIES: Final[frozenset[str]] = frozenset(
    get_args(PanelModelFamilyId.__value__)
)
INSTALLED_PANEL_MODEL_GRID_SIZES: Final[Mapping[str, int]] = {
    "RIDGE": 4,
    "ELASTIC_NET": 12,
    "LIGHTGBM": 720,
}
"""How many trials each family's installed grid holds, per view.

Named here because the budget has to price a grid it never builds, and because a
reader asking "what does naming a family cost" should find the answer beside the
families rather than inside an arithmetic expression.
"""

_CONTROL_VIEW_ID: Final[InstalledPanelViewId] = "RELATIVE_CONTROL"
_LIGHTGBM_EARLY_STOPPING_GRID_SHARE: Final = 240
"""Grid trials that early-stop, and so pay one extra native fit each."""

PANEL_ALPHA_FOLD_RSS_FUSE_BYTES: Final = 12 * 1024**3
PANEL_ALPHA_PARENT_RSS_RESERVATION_BYTES: Final = 6 * 1024**3
_RUNTIME_ALLOCATION_CAP_FIELDS: Final[tuple[str, ...]] = (
    "runtime_workload",
    "runtime_profile",
    "runtime_profile_resolution",
    "process_logical_processor_limit",
    "applied_logical_processor_ids",
    "duckdb_threads",
    "blas_threads",
)
_ADMITTED_VIEWS: Final[frozenset[InstalledPanelViewId]] = frozenset(INSTALLED_PANEL_VIEW_IDS)
"""Every installed Feature view, admitted regardless of `recipe_scope`.

`recipe_scope` used to carry two tuples and hand one of them to the design
check, so declaring `LEAN` did not just size the run -- it decided which Features
existed. A scope is a resource statement: narrower views, fewer candidates, a
smaller budget, Alpha only. Which Features a study reads is the study's own
declaration, and `feature_view_ids` already carries it. The scope conditions that
survive are the scientific ones a few lines below.
"""


def _paired_methodology_implementation_hash() -> str:
    root = Path(__file__).resolve().parents[3]

    def component_id(value: str) -> str:
        relative = Path(value).with_suffix("").as_posix().replace("/", ".")
        return f"alpha_research.paired_panel.{relative}"

    relative_sources = (
        "control/observation_runtime/telemetry/process_metrics.py",
        "control/product_host/research_authoring/authority.py",
        "control/product_host/research_authoring/execution.py",
        "control/product_host/research_authoring/panel_methodology_sources.py",
        "control/research_program/authoring/workflow.py",
        "foundation/feature_engine/producers/preprocessing/joint_primary.py",
        "capabilities/alpha_modeling/contracts.py",
        "capabilities/alpha_modeling/adapters/lightgbm_chronological.py",
        "capabilities/alpha_modeling/adapters/lightgbm_dynamic_panel.py",
        "capabilities/alpha_modeling/adapters/regularized_linear_dynamic_panel.py",
        "capabilities/portfolio_backtesting/execution.py",
        "capabilities/portfolio_backtesting/contracts.py",
        "capabilities/portfolio_backtesting/engine.py",
        "capabilities/portfolio_backtesting/metrics.py",
        "capabilities/portfolio_backtesting/segments.py",
        "investment/alpha_research/experiments/authoring.py",
        "investment/alpha_research/experiments/panel_alpha_fold_execution.py",
        "investment/alpha_research/experiments/panel_methodology_authoring.py",
        "investment/portfolio_strategy_lab/research_loop/panel_methodology_execution.py",
        "investment/alpha_research/experiments/panel_methodology_models.py",
        "investment/alpha_research/experiments/panel_methodology_statistics.py",
        "investment/alpha_research/experiments/verification.py",
        "investment/alpha_research/inputs/panel_feature_materialization.py",
        "investment/alpha_research/inputs/panel_feature_views.py",
        "investment/alpha_research/scores/score_filters.py",
        "investment/alpha_research/scores/temporal_aggregation.py",
        "investment/alpha_research/targets/total_return.py",
        "investment/portfolio_strategy_lab/evaluation/walk_forward.py",
        "investment/portfolio_strategy_lab/campaign/authority.py",
        "investment/portfolio_strategy_lab/optimizer/service.py",
        "investment/portfolio_strategy_lab/policies/minimum_variance.py",
        "investment/portfolio_strategy_lab/policies/score_risk_cost.py",
        "investment/portfolio_strategy_lab/research_loop/paired_alpha_portfolio.py",
        "investment/portfolio_strategy_lab/research_loop/paired_alpha_portfolio_evidence.py",
        "investment/portfolio_strategy_lab/publication/artifacts.py",
        "investment/portfolio_management/mandates/paired_panel_grid.py",
    )
    return str(
        switched_source_identity(
            {component_id(value): root / value for value in relative_sources},
            semantic_owner="alpha_research",
            numerical_role="PAIRED_PANEL_PORTFOLIO_RESEARCH",
        )
    )


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


type PanelModelParameterValue = float | int | str | bool | None


class PanelModelRecipeDeclaration(_Contract):
    """One estimator configuration a request names, instead of a family it expands.

    Naming a family is a request to explore that family's installed grid, which
    for LightGBM is seven hundred and twenty trials per view. That is the right
    answer to "which hyperparameters win" and the wrong answer to every other
    question -- fitting a configuration that is already known, reproducing a
    result, or checking that a lane executes. A researcher had no way to ask the
    second kind of question from a document: the only lever was the family, and
    the platform expanded it.

    `parameters` is passed to the family's own adapter, which validates it
    against the bounds its search domain declares. Nothing here interprets a
    parameter, so installing a new estimator parameter does not touch this file.
    """

    kind: Literal["PanelModelRecipeDeclaration"] = "PanelModelRecipeDeclaration"
    recipe_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    family: PanelModelFamilyId
    parameters: Mapping[str, PanelModelParameterValue]
    view_ids: tuple[InstalledPanelViewId, ...] = ()
    """Which declared Feature views this recipe fits, or every treatment view.

    Named because "the wider estimators run on the treatment views" is the right
    default and the wrong answer for an ablation design: a request declaring the
    control plus ten `DYNAMIC_JOINT_WITHOUT_*` views wants Ridge on all eleven
    and the tree on the one full view, not seven thousand tree fits. Which of the
    declared views a recipe belongs to is a property of the experiment, so the
    experiment says it.
    """

    declaration_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(
        cls,
        *,
        recipe_id: str,
        family: str,
        parameters: Mapping[str, Any],
        view_ids: tuple[str, ...] = (),
    ) -> Self:
        if family not in INSTALLED_PANEL_MODEL_FAMILIES:
            raise AuthoringError("alpha_research.panel_model_recipe_family_not_installed")
        if not set(view_ids).issubset(INSTALLED_PANEL_VIEW_IDS) or len(set(view_ids)) != len(
            view_ids
        ):
            raise AuthoringError("alpha_research.panel_model_recipe_view_not_installed")
        frozen: dict[str, PanelModelParameterValue] = {}
        for key, value in parameters.items():
            if isinstance(value, (list, tuple, dict)):
                raise AuthoringError("alpha_research.panel_model_recipe_not_exact")
            frozen[str(key)] = cast(PanelModelParameterValue, value)
        draft = {
            "recipe_id": recipe_id,
            "family": family,
            "parameters": frozen,
            "view_ids": tuple(view_ids),
        }
        return cls(**draft, declaration_hash=str(canonical_hash(draft)))


class AlphaResearchMethodologyDescriptor(_Contract):
    method_id: PanelMethodologyId
    owner: Literal["alpha_research.experiments"] = "alpha_research.experiments"
    description: str
    lifecycle_commands: tuple[str, ...]
    implementation_hash: str = Field(pattern=_HASH)
    descriptor_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, descriptor_hash="0" * 64)
        return cls(
            **values,
            descriptor_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"descriptor_hash"}))
            ),
        )


class AlphaResearchMethodologyCatalog:
    """Descriptor inventory; selected identities do not bind inventory peers."""

    def __init__(self) -> None:
        lifecycle = (
            "list-methods",
            "inspect-method",
            "preflight",
            "alpha-run",
            "alpha-resume",
            "score-filter",
            "portfolio-run",
            "portfolio-resume",
            "inspect",
            "replay",
        )
        descriptors = (
            AlphaResearchMethodologyDescriptor.create(
                method_id="SINGLE_MODEL_FIXED_AXIS",
                description="Existing fixed-axis single-model Alpha development method.",
                lifecycle_commands=lifecycle,
                implementation_hash=str(
                    canonical_hash("alpha_research.single_model_fixed_axis.installed")
                ),
            ),
            AlphaResearchMethodologyDescriptor.create(
                method_id="PAIRED_PANEL_PORTFOLIO_RESEARCH",
                description=(
                    "Paired Feature/view/model/span research followed by installed Risk, "
                    "Portfolio optimizer and OOS evaluation."
                ),
                lifecycle_commands=lifecycle,
                implementation_hash=_paired_methodology_implementation_hash(),
            ),
            AlphaResearchMethodologyDescriptor.create(
                method_id="MODEL_LIFECYCLE_REPLAY",
                description=(
                    "Declared model renewal and ensemble replay over admitted local "
                    "training observations; development evidence only."
                ),
                lifecycle_commands=(
                    "validate",
                    "freeze",
                    "preflight",
                    "run",
                    "resume",
                    "inspect",
                    "replay",
                ),
                implementation_hash=lifecycle_implementation_hash(),
            ),
        )
        self._descriptors = {value.method_id: value for value in descriptors}
        self.inventory_hash = str(
            canonical_hash(
                [value.descriptor_hash for value in sorted(descriptors, key=lambda x: x.method_id)]
            )
        )

    @property
    def descriptors(self) -> tuple[AlphaResearchMethodologyDescriptor, ...]:
        return tuple(self._descriptors[key] for key in sorted(self._descriptors))

    def resolve(self, method_id: str) -> AlphaResearchMethodologyDescriptor:
        try:
            return self._descriptors[cast(PanelMethodologyId, method_id)]
        except KeyError as error:
            raise AuthoringError("alpha_research.methodology_not_installed") from error


class PanelResearchRuntimeCaps(_Contract):
    maximum_feature_count: int = Field(ge=1)
    maximum_candidate_count: int = Field(ge=1, le=4096)
    maximum_fit_calls: int = Field(ge=1, le=20_000)
    maximum_predict_calls: int = Field(ge=1)
    maximum_metric_calls: int = Field(ge=1)
    maximum_solver_calls: int = Field(ge=1)
    maximum_estimated_wall_seconds: int = Field(ge=1, le=64_800)
    maximum_estimated_peak_memory_bytes: int = Field(ge=1)
    maximum_fold_workers: int = Field(ge=0, le=4)
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
    process_logical_processor_limit: int | None = Field(default=None, ge=1)
    applied_logical_processor_ids: tuple[int, ...] = ()
    duckdb_threads: int | None = Field(default=None, ge=1)
    blas_threads: Literal[1] | None = None
    lightgbm_threads_per_fit: int = Field(default=1, ge=0, le=1)


class PanelResearchMethodologyRequest(_Contract):
    methodology_id: Literal["PAIRED_PANEL_PORTFOLIO_RESEARCH"]
    execution_stage: PanelExecutionStage = "END_TO_END"
    upstream_alpha_handle: Literal["SEALED_ALPHA_OUTPUT"] | None = None
    upstream_score_filter_handle: Literal["SEALED_SCORE_FILTER_OUTPUT"] | None = None
    recipe_scope: Literal["PRIMARY", "LEAN"]
    search_intent: Literal["HYPERPARAMETER_DISCOVERY"] | None = None
    input_method_id: Literal["FEATURE_T_PANEL_TOTAL_RETURN_INPUTS"]
    development_overlay_method_id: Literal["SESSION_OBSERVATION_FORMULA_OVERLAY"] | None = None
    target_method_id: Literal["CROSS_SECTIONAL_TOTAL_RETURN_STD_Z"]
    feature_view_ids: tuple[InstalledPanelViewId, ...]
    model_family_ids: tuple[PanelModelFamilyId, ...] = ()
    model_recipes: tuple[PanelModelRecipeDeclaration, ...] = ()
    model_selection_method_id: Literal["SESSION_RANK_IC_PAIRED"]
    score_filter_candidates: tuple[AlphaScoreFilterSpec, ...] = ()
    score_filter_selection_method_id: (
        Literal["MATURED_PRIOR_FOLD_SESSION_RANK_IC_PAIRED"] | None
    ) = None
    persistence_lags: tuple[int, ...] = ()
    fixed_score_method_id: Literal["MOMENTUM_252_21_SIGNED_CROSS_SECTION"] | None = None
    dependence_method_id: Literal["SPAN_HORIZON_AWARE_CIRCULAR_BLOCK"]
    risk_method_ids: tuple[Literal["R0", "R1"], ...] = ()
    portfolio_policy_ids: tuple[
        Literal[
            "TOP_K_EQUAL_WEIGHT",
            "MINIMUM_VARIANCE",
            "TOP_K_SCORE_RISK_COST",
            "RANK_BUFFERED_SCORE_RISK_COST",
        ],
        ...,
    ] = ()
    sector_capacities: tuple[float | None, ...] = ()
    cost_stress_bps: tuple[int, ...] = ()
    top_k: int | None = None
    name_cap: float | None = None
    risk_aversion_grid: tuple[float, ...] = ()
    turnover_regularization_grid: tuple[float, ...] = ()
    alpha_utility_method_id: Literal["DIMENSIONLESS_SCORE_UTILITY"] | None = None
    request_hash: str = Field(pattern=_HASH)

    @property
    def model_families(self) -> frozenset[str]:
        """The estimator families this request uses, however it declared them.

        `model_family_ids` names a family and asks for its installed grid;
        `model_recipes` names configurations and asks for exactly those. Both
        answer "which families run", and every caller downstream of that question
        should not have to know which form was used.
        """

        return frozenset(self.model_family_ids) | frozenset(
            value.family for value in self.model_recipes
        )

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any]) -> Self:
        forbidden = (
            "hash",
            "path",
            "uri",
            "matrix",
            "callback",
            "authority",
            "permission",
            "publication",
            "admission",
        )
        if any(
            any(token in str(key).lower() for token in forbidden)
            for key in section
            if key != "methodology_id"
        ):
            raise AuthoringError("alpha_research.methodology_authority_forbidden")
        draft = dict(section)
        draft.pop("request_hash", None)
        raw_recipes = draft.get("model_recipes", [])
        if not isinstance(raw_recipes, list):
            raise AuthoringError("alpha_research.panel_model_recipes_invalid")
        try:
            draft["model_recipes"] = tuple(
                PanelModelRecipeDeclaration.create(
                    recipe_id=str(value["recipe_id"]),
                    family=str(value["family"]),
                    parameters=dict(value.get("parameters", {})),
                    view_ids=tuple(value.get("view_ids", ())),
                )
                for value in raw_recipes
                if isinstance(value, Mapping)
            )
        except AuthoringError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise AuthoringError("alpha_research.panel_model_recipes_invalid") from error
        if len(draft["model_recipes"]) != len(raw_recipes):
            raise AuthoringError("alpha_research.panel_model_recipes_invalid")
        raw_filter_candidates = draft.get("score_filter_candidates", [])
        if not isinstance(raw_filter_candidates, list):
            raise AuthoringError("alpha_research.score_filter_candidates_invalid")
        try:
            draft["score_filter_candidates"] = tuple(
                AlphaScoreFilterSpec.create(
                    method_id=str(value["method_id"]),
                    span_sessions=int(value["span_sessions"]),
                    input_mode=str(value.get("input_mode", "RAW_SCORE")),
                )
                for value in raw_filter_candidates
                if isinstance(value, Mapping)
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AuthoringError("alpha_research.score_filter_candidates_invalid") from error
        if len(draft["score_filter_candidates"]) != len(raw_filter_candidates):
            raise AuthoringError("alpha_research.score_filter_candidates_invalid")
        for field in (
            "feature_view_ids",
            "model_family_ids",
            "persistence_lags",
            "risk_method_ids",
            "portfolio_policy_ids",
            "sector_capacities",
            "cost_stress_bps",
            "risk_aversion_grid",
            "turnover_regularization_grid",
        ):
            if field in draft:
                draft[field] = tuple(draft[field])
        for field in ("risk_aversion_grid", "turnover_regularization_grid"):
            if field in draft:
                draft[field] = tuple(float(value) for value in draft[field])
        declared_scope = draft.get("recipe_scope")
        declared_views = tuple(draft.get("feature_view_ids", ()))
        # Refused here as well as in the validator so the caller sees the stable
        # authoring error rather than a wrapped one, which is how every sibling
        # rule in this method reports.
        if bool(draft.get("model_family_ids")) == bool(draft["model_recipes"]):
            raise AuthoringError("alpha_research.panel_model_declaration_ambiguous")
        declared_recipe_ids = tuple(value.recipe_id for value in draft["model_recipes"])
        if len(set(declared_recipe_ids)) != len(declared_recipe_ids):
            raise AuthoringError("alpha_research.panel_model_recipe_id_duplicated")
        if bool(draft.get("model_family_ids")) != (
            draft.get("search_intent") == "HYPERPARAMETER_DISCOVERY"
        ):
            raise AuthoringError("alpha_research.panel_model_search_intent_invalid")
        if any(not value.parameters or not value.view_ids for value in draft["model_recipes"]):
            raise AuthoringError("alpha_research.panel_model_recipe_not_exact")
        if any(
            not set(value.view_ids).issubset(declared_views) for value in draft["model_recipes"]
        ):
            raise AuthoringError("alpha_research.panel_model_recipe_view_not_declared")
        declared_families = frozenset(draft.get("model_family_ids", ())) | frozenset(
            value.family for value in draft["model_recipes"]
        )
        # `recipe_scope` states the Feature view's breadth; the estimator is
        # stated either as a family to expand or as named configurations. All
        # three are validated against their own install catalogs and then
        # combined, so a scope may not decide which models are admissible. The
        # conditions that remain are scientific: a declared control has to be
        # present, the control view has to be joined by at least one treatment
        # view, and a PRIMARY-scope run of the wider estimators has to include
        # the wider view it is meant to exercise.
        if declared_scope in {"PRIMARY", "LEAN"} and (
            "RIDGE" not in declared_families
            or "RELATIVE_CONTROL" not in declared_views
            or not any(value != "RELATIVE_CONTROL" for value in declared_views)
            or (
                declared_scope == "PRIMARY"
                and {"ELASTIC_NET", "LIGHTGBM"}.intersection(declared_families)
                and "DYNAMIC_JOINT_PRIMARY" not in declared_views
            )
        ):
            raise AuthoringError("alpha_research.panel_model_family_view_not_installed")
        declared_stage = draft.get("execution_stage", "END_TO_END")
        declared_filters = tuple(draft["score_filter_candidates"])
        if (
            declared_stage in {"SCORE_FILTER_ONLY", "END_TO_END"}
            and sum(
                (
                    value.method_id,
                    value.span_sessions,
                    value.input_mode,
                )
                == (RAW_SCORE_TEMPORAL_FILTER_METHOD_ID, 1, RAW_SCORE_INPUT_MODE)
                for value in declared_filters
            )
            != 1
        ):
            raise AuthoringError("alpha_research.score_filter_raw_control_invalid")
        provisional = cls.model_construct(**draft, request_hash="0" * 64)
        return cls(
            **draft,
            request_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"request_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_design(self) -> Self:
        admitted_views = _ADMITTED_VIEWS
        # Estimator families come from the install declaration, never from the
        # scope: `recipe_scope` states Feature-view breadth and may not decide
        # which models are admissible.
        admitted_models = INSTALLED_PANEL_MODEL_FAMILIES
        admitted_filters = {
            (RAW_SCORE_TEMPORAL_FILTER_METHOD_ID, 1, RAW_SCORE_INPUT_MODE),
            (
                RAW_SCORE_TEMPORAL_FILTER_METHOD_ID,
                1,
                CROSS_SECTION_STANDARDIZED_SCORE_INPUT_MODE,
            ),
            ("TRAILING_MEAN_FORMATION_SCORE", 21, RAW_SCORE_INPUT_MODE),
            ("TRAILING_MEAN_FORMATION_SCORE", 63, RAW_SCORE_INPUT_MODE),
            (
                "TRAILING_MEAN_FORMATION_SCORE",
                21,
                CROSS_SECTION_STANDARDIZED_SCORE_INPUT_MODE,
            ),
            ("EWMA_FORMATION_SCORE", 21, RAW_SCORE_INPUT_MODE),
            ("EWMA_FORMATION_SCORE", 63, RAW_SCORE_INPUT_MODE),
            ("EWMA_FORMATION_SCORE", 21, CROSS_SECTION_STANDARDIZED_SCORE_INPUT_MODE),
        }
        selected_filters = {
            (value.method_id, value.span_sessions, value.input_mode)
            for value in self.score_filter_candidates
        }
        raw_controls = tuple(
            value
            for value in self.score_filter_candidates
            if (
                value.method_id,
                value.span_sessions,
                value.input_mode,
            )
            == (RAW_SCORE_TEMPORAL_FILTER_METHOD_ID, 1, RAW_SCORE_INPUT_MODE)
        )
        stage_uses_filters = self.execution_stage in {"SCORE_FILTER_ONLY", "END_TO_END"}
        stage_uses_portfolio = self.execution_stage in {"PORTFOLIO_ONLY", "END_TO_END"}
        stage_uses_alpha = self.execution_stage in {"ALPHA_ONLY", "END_TO_END"}
        if not self.feature_view_ids or not set(self.feature_view_ids).issubset(admitted_views):
            raise AuthoringError("alpha_research.panel_methodology_design_invalid")
        if len(set(self.feature_view_ids)) != len(self.feature_view_ids):
            raise AuthoringError("alpha_research.panel_methodology_design_invalid")
        # Exactly one of the two forms. Allowing both would leave "does LIGHTGBM
        # mean these three points or all seven hundred and twenty" answerable
        # only by reading the resolver.
        if bool(self.model_family_ids) == bool(self.model_recipes):
            raise AuthoringError("alpha_research.panel_model_declaration_ambiguous")
        if not self.model_families.issubset(admitted_models):
            raise AuthoringError("alpha_research.panel_methodology_design_invalid")
        if len(set(self.model_family_ids)) != len(self.model_family_ids):
            raise AuthoringError("alpha_research.panel_methodology_design_invalid")
        recipe_ids = tuple(value.recipe_id for value in self.model_recipes)
        if len(set(recipe_ids)) != len(recipe_ids):
            raise AuthoringError("alpha_research.panel_model_recipe_id_duplicated")
        if bool(self.model_family_ids) != (self.search_intent == "HYPERPARAMETER_DISCOVERY"):
            raise AuthoringError("alpha_research.panel_model_search_intent_invalid")
        if any(not value.parameters or not value.view_ids for value in self.model_recipes):
            raise AuthoringError("alpha_research.panel_model_recipe_not_exact")
        if any(
            not set(value.view_ids).issubset(self.feature_view_ids) for value in self.model_recipes
        ):
            raise AuthoringError("alpha_research.panel_model_recipe_view_not_declared")
        if bool(set(self.feature_view_ids).intersection(SPARSE_SESSION_AMPLITUDE_VIEW_IDS)) != (
            self.development_overlay_method_id == "SESSION_OBSERVATION_FORMULA_OVERLAY"
        ):
            raise AuthoringError("alpha_research.panel_development_overlay_declaration_invalid")
        if (
            "RIDGE" not in self.model_families
            or "RELATIVE_CONTROL" not in self.feature_view_ids
            or not any(value != "RELATIVE_CONTROL" for value in self.feature_view_ids)
            or (
                self.recipe_scope == "PRIMARY"
                and {"ELASTIC_NET", "LIGHTGBM"}.intersection(self.model_families)
                and "DYNAMIC_JOINT_PRIMARY" not in self.feature_view_ids
            )
        ):
            raise AuthoringError("alpha_research.panel_model_family_view_not_installed")
        if stage_uses_filters != bool(self.score_filter_candidates):
            raise AuthoringError("alpha_research.panel_methodology_stage_filter_invalid")
        if stage_uses_filters and len(raw_controls) != 1:
            raise AuthoringError("alpha_research.score_filter_raw_control_invalid")
        if selected_filters - admitted_filters or len(
            {value.spec_id for value in self.score_filter_candidates}
        ) != len(self.score_filter_candidates):
            raise AuthoringError("alpha_research.panel_methodology_design_invalid")
        if self.score_filter_selection_method_id != (
            "MATURED_PRIOR_FOLD_SESSION_RANK_IC_PAIRED" if stage_uses_filters else None
        ):
            raise AuthoringError("alpha_research.panel_methodology_stage_filter_invalid")
        if self.persistence_lags != ((0, 1, 2, 5, 10, 21) if stage_uses_filters else ()):
            raise AuthoringError("alpha_research.panel_methodology_stage_filter_invalid")
        if self.upstream_alpha_handle != (None if stage_uses_alpha else "SEALED_ALPHA_OUTPUT"):
            raise AuthoringError("alpha_research.panel_methodology_upstream_alpha_handle_invalid")
        if self.upstream_score_filter_handle is not None and not stage_uses_portfolio:
            raise AuthoringError("alpha_research.panel_methodology_upstream_filter_handle_invalid")
        # The Portfolio lab runs these grids; one definition both read (OW10).
        stage = PairedPanelPortfolioGrid(
            **{name: getattr(self, name) for name in PAIRED_PANEL_GRID_FIELDS}
        )
        portfolio_fields_valid = stage == PAIRED_PANEL_PORTFOLIO_GRID
        turnover_successor_fields_valid = stage == RANK_BUFFERED_TURNOVER_SUCCESSOR_GRID
        portfolio_fields_empty = (
            not self.risk_method_ids
            and not self.portfolio_policy_ids
            and not self.sector_capacities
            and not self.cost_stress_bps
            and self.top_k is None
            and self.name_cap is None
            and not self.risk_aversion_grid
            and not self.turnover_regularization_grid
            and self.alpha_utility_method_id is None
            and self.fixed_score_method_id is None
        )
        if stage_uses_portfolio != (portfolio_fields_valid or turnover_successor_fields_valid) or (
            not stage_uses_portfolio and not portfolio_fields_empty
        ):
            raise AuthoringError("alpha_research.panel_methodology_stage_portfolio_invalid")
        if self.request_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"request_hash"})
        ):
            raise AuthoringError("alpha_research.panel_methodology_design_invalid")
        return self


def panel_alpha_configuration_hash(request: PanelResearchMethodologyRequest) -> str:
    """Identity shared by Alpha production and its downstream stage consumers."""

    return str(
        canonical_hash(
            {
                "recipe_scope": request.recipe_scope,
                "search_intent": request.search_intent,
                "input_method_id": request.input_method_id,
                "development_overlay_method_id": request.development_overlay_method_id,
                "target_method_id": request.target_method_id,
                "feature_view_ids": list(request.feature_view_ids),
                "model_family_ids": sorted(request.model_family_ids),
                "model_recipes": [
                    value.model_dump(mode="json", exclude={"declaration_hash"})
                    for value in sorted(request.model_recipes, key=lambda item: item.recipe_id)
                ],
                "model_selection_method_id": request.model_selection_method_id,
                "dependence_method_id": request.dependence_method_id,
            }
        )
    )


class PanelResearchPreflight(_Contract):
    kind: str = "PanelResearchPreflight"
    request_hash: str = Field(pattern=_HASH)
    source_resolution_hash: str = Field(pattern=_HASH)
    feature_preflight_hash: str = Field(pattern=_HASH)
    actual_feature_count: int = Field(ge=1)
    actual_candidate_count: int = Field(ge=0)
    actual_fit_calls: int = Field(ge=0)
    actual_predict_calls: int = Field(ge=0)
    actual_metric_calls: int = Field(ge=0)
    actual_solver_calls: int = Field(ge=0)
    actual_portfolio_formation_count: int = Field(ge=0)
    actual_portfolio_economic_formation_count: int = Field(default=0, ge=0)
    actual_portfolio_listing_count: int = Field(ge=0)
    portfolio_common_session_axis_hash: str | None = Field(default=None, pattern=_HASH)
    portfolio_economic_session_axis_hash: str | None = Field(default=None, pattern=_HASH)
    portfolio_common_listing_axis_hash: str | None = Field(default=None, pattern=_HASH)
    portfolio_common_axis_hash: str | None = Field(default=None, pattern=_HASH)
    estimated_wall_seconds: int = Field(ge=1)
    estimated_peak_memory_bytes: int = Field(ge=1)
    reuse_actions: tuple[str, ...]
    recompute_actions: tuple[str, ...]
    caps: PanelResearchRuntimeCaps
    preflight_hash: str = Field(pattern=_HASH)

    @staticmethod
    def _identity_payload(
        value: PanelResearchPreflight,
        *,
        include_operational_capacity: bool = False,
        include_legacy_runtime_plan: bool = False,
        include_preflight_hash: bool = False,
    ) -> dict[str, object]:
        """Use one legacy-compatible canonicalization for create and readback."""

        payload = value.model_dump(
            mode="json",
            exclude=None if include_preflight_hash else {"preflight_hash"},
        )
        # Preflights sealed before the Portfolio state/economic split did not
        # carry these fields. Preserve their immutable readback identity while
        # requiring every newly created receipt to bind both axes.
        if "actual_portfolio_economic_formation_count" not in value.model_fields_set:
            payload.pop("actual_portfolio_economic_formation_count", None)
        if "portfolio_economic_session_axis_hash" not in value.model_fields_set:
            payload.pop("portfolio_economic_session_axis_hash", None)
        # Capacity is operational evidence, not scientific identity.  The
        # exact plan is recorded on execution roots, while a Program remains
        # reproducible across operational profiles and host CPU assignments.
        caps_payload = cast(dict[str, object], payload["caps"])
        runtime_fields = ("runtime_plan_hash", *_RUNTIME_ALLOCATION_CAP_FIELDS)
        if include_legacy_runtime_plan:
            # Gate 4 receipts recorded the predecessor's opaque plan handle
            # but predate the workload/allocation fields below.  Keep exactly
            # that older payload readable; writers continue using the current
            # non-operational identity.
            runtime_fields = runtime_fields[1:]
        for field in runtime_fields:
            caps_payload.pop(field, None)
        if not include_operational_capacity:
            # These estimates are derived from the operational fold allocation
            # and memory reservation above.  They remain enforced admission
            # evidence, but cannot make an otherwise identical scientific
            # Program host-bound.
            caps_payload.pop("maximum_estimated_peak_memory_bytes", None)
            caps_payload.pop("maximum_fold_workers", None)
            payload.pop("estimated_wall_seconds", None)
            payload.pop("estimated_peak_memory_bytes", None)
        return payload

    @property
    def has_pre_allocation_runtime_plan_payload(self) -> bool:
        """Recognize the pre-allocation plan shape for hash-verified readback."""

        return self.caps.runtime_plan_hash is not None and not set(
            _RUNTIME_ALLOCATION_CAP_FIELDS
        ).intersection(self.caps.model_fields_set)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, preflight_hash="0" * 64)
        return cls(
            **values,
            preflight_hash=str(canonical_hash(cls._identity_payload(provisional))),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_budget(self) -> Self:
        failures = (
            self.actual_feature_count > self.caps.maximum_feature_count,
            self.actual_candidate_count > self.caps.maximum_candidate_count,
            self.actual_fit_calls > self.caps.maximum_fit_calls,
            self.actual_predict_calls > self.caps.maximum_predict_calls,
            self.actual_metric_calls > self.caps.maximum_metric_calls,
            self.actual_solver_calls > self.caps.maximum_solver_calls,
            self.estimated_wall_seconds > self.caps.maximum_estimated_wall_seconds,
            self.estimated_peak_memory_bytes > self.caps.maximum_estimated_peak_memory_bytes,
        )
        if any(failures):
            raise AuthoringError("alpha_research.panel_methodology_budget_exceeded")
        admitted_hashes = {
            canonical_hash(self._identity_payload(self)),
            # Read receipts sealed before operational capacity stopped rotating
            # Program identity, without publishing that canonicalization again.
            canonical_hash(self._identity_payload(self, include_operational_capacity=True)),
        }
        if self.has_pre_allocation_runtime_plan_payload:
            admitted_hashes.add(
                canonical_hash(
                    self._identity_payload(
                        self,
                        include_operational_capacity=True,
                        include_legacy_runtime_plan=True,
                    )
                )
            )
        if self.preflight_hash not in admitted_hashes:
            raise AuthoringError("alpha_research.panel_methodology_preflight_invalid")
        return self


def panel_model_recipe_points(
    declaration: PanelModelRecipeDeclaration,
) -> tuple[dict[str, object], ...]:
    """Return the one exact parameter point bound by a named recipe."""

    if not declaration.parameters:
        raise AuthoringError("alpha_research.panel_model_recipe_not_exact")
    return (dict(declaration.parameters),)


def alpha_model_call_upper_bounds(
    request: PanelResearchMethodologyRequest, *, fold_count: int
) -> tuple[int, int, int, int]:
    """Return candidate and fit/predict/metric upper bounds for Alpha execution."""

    treatment_views = sum(1 for value in request.feature_view_ids if value != _CONTROL_VIEW_ID)
    declared = request.model_families
    ridge_groups = len(request.feature_view_ids) if "RIDGE" in declared else 0
    elastic_groups = treatment_views if "ELASTIC_NET" in declared else 0
    lightgbm_groups = treatment_views if "LIGHTGBM" in declared else 0
    if request.model_recipes:
        inner_trials_per_fold = 0
        early_stopping_lightgbm_specs = 0
        group_keys: set[tuple[str, str]] = set()
        for declaration in request.model_recipes:
            points = panel_model_recipe_points(declaration)
            inner_trials_per_fold += len(declaration.view_ids) * len(points)
            group_keys.update((view_id, declaration.family) for view_id in declaration.view_ids)
            if declaration.family == "LIGHTGBM":
                early_stopping_lightgbm_specs += len(declaration.view_ids) * sum(
                    point.get("training_policy") != "FIXED_ITERATION" for point in points
                )
        ridge_groups = sum(family == "RIDGE" for _view, family in group_keys)
        elastic_groups = sum(family == "ELASTIC_NET" for _view, family in group_keys)
        lightgbm_groups = sum(family == "LIGHTGBM" for _view, family in group_keys)
        outer_selected_specs = inner_trials_per_fold
        nested_outer_specs = early_stopping_lightgbm_specs
    else:
        inner_trials_per_fold = (
            ridge_groups * INSTALLED_PANEL_MODEL_GRID_SIZES["RIDGE"]
            + elastic_groups * INSTALLED_PANEL_MODEL_GRID_SIZES["ELASTIC_NET"]
            + lightgbm_groups * INSTALLED_PANEL_MODEL_GRID_SIZES["LIGHTGBM"]
        )
        early_stopping_lightgbm_specs = lightgbm_groups * _LIGHTGBM_EARLY_STOPPING_GRID_SHARE
        outer_groups = ridge_groups + elastic_groups + lightgbm_groups
        outer_selected_specs = min(inner_trials_per_fold, outer_groups + 2)
        nested_outer_specs = (
            min(3, outer_selected_specs - ridge_groups - elastic_groups) if lightgbm_groups else 0
        )
    inner_native_fit_calls = inner_trials_per_fold + early_stopping_lightgbm_specs
    outer_native_fit_calls = outer_selected_specs + nested_outer_specs
    inner_predict_calls = 2 * inner_trials_per_fold + early_stopping_lightgbm_specs
    outer_predict_calls = 2 * outer_selected_specs + nested_outer_specs
    return (
        inner_trials_per_fold,
        fold_count * (inner_native_fit_calls + outer_native_fit_calls),
        fold_count * (inner_predict_calls + outer_predict_calls),
        fold_count * (inner_trials_per_fold + outer_selected_specs),
    )


def build_panel_research_preflight(
    *,
    request: PanelResearchMethodologyRequest,
    plan: PanelFeaturePlan | None,
    fixed_feature_preflight_hash: str | None = None,
    fixed_feature_count_by_method: Mapping[str, int] | None = None,
    source_resolution_hash: str,
    fold_count: int,
    maximum_solver_calls_for_formation_count: Callable[[int], int],
    caps: PanelResearchRuntimeCaps,
    formation_count: int = 0,
    economic_formation_count: int = 0,
    portfolio_listing_count: int = 0,
    portfolio_session_axis_hash: str | None = None,
    portfolio_economic_session_axis_hash: str | None = None,
    portfolio_listing_axis_hash: str | None = None,
    portfolio_axis_hash: str | None = None,
) -> PanelResearchPreflight:
    alpha_stage = request.execution_stage in {"ALPHA_ONLY", "END_TO_END"}
    filter_stage = request.execution_stage in {"SCORE_FILTER_ONLY", "END_TO_END"}
    portfolio_stage = request.execution_stage in {"PORTFOLIO_ONLY", "END_TO_END"}
    (
        inner_trials_per_fold,
        alpha_fit_calls,
        alpha_predict_calls,
        alpha_metric_calls,
    ) = alpha_model_call_upper_bounds(request, fold_count=fold_count)
    actual_fit_calls = alpha_fit_calls if alpha_stage else 0
    actual_predict_calls = alpha_predict_calls if alpha_stage else 0
    filter_metric_calls = (
        len(request.score_filter_candidates) * (1 + 3 * len(request.persistence_lags))
        if filter_stage
        else 0
    )
    economic_metric_calls = (
        fixed_recipe_economic_metric_call_count(
            tuple(value.recipe_id for value in request.model_recipes)
        )
        if alpha_stage
        else 0
    )
    actual_metric_calls = (
        (alpha_metric_calls if alpha_stage else 0) + filter_metric_calls + economic_metric_calls
    )
    scientific_candidates = (inner_trials_per_fold if alpha_stage else 0) + (
        len(request.score_filter_candidates) if filter_stage else 0
    )
    # Portfolio owns its exact arm expansion and Direct-OSQP retry policy.  The
    # Alpha compiler consumes that owner budget rather than duplicating it.
    actual_solver_calls = (
        maximum_solver_calls_for_formation_count(formation_count) if portfolio_stage else 0
    )
    if plan is not None:
        feature_preflight_hash = plan.preflight.preflight_hash
        feature_count_by_method = plan.preflight.feature_count_by_method
    else:
        feature_preflight_hash = fixed_feature_preflight_hash
        feature_count_by_method = fixed_feature_count_by_method
    if feature_preflight_hash is None or not feature_count_by_method:
        raise AuthoringError("alpha_research.panel_feature_handoff_missing")
    feature_count = max(feature_count_by_method.values())
    # The review-remediation run measured a 10.49-GiB worker lifetime peak.  The
    # admitted reservation is therefore 12 GiB per worker plus 6 GiB for the
    # parent/source process, not the former 10-GiB assumption presented as a
    # measurement.  It remains below both the 45% preflight fuse and 50% machine
    # ceiling with four workers.
    worker_count = max(1, min(fold_count, caps.maximum_fold_workers))
    estimated_peak_memory = (
        PANEL_ALPHA_PARENT_RSS_RESERVATION_BYTES + PANEL_ALPHA_FOLD_RSS_FUSE_BYTES * worker_count
        if alpha_stage
        else PANEL_ALPHA_PARENT_RSS_RESERVATION_BYTES
    )
    ridge_only_lean = (
        alpha_stage and request.recipe_scope == "LEAN" and request.model_family_ids == ("RIDGE",)
    )
    # The accepted five-fold Lean Ridge run completed in 513.580 seconds over
    # two bounded worker waves.  Reserve 20% planning margin per wave for
    # admission; this is deliberately not a host-independent runtime promise.
    accepted_lean_wave_seconds = 513.580 / 2.0
    # Apply the same planning-reservation meaning to the measured filter stage.
    accepted_score_filter_seconds = 292.639
    fixed_economic_recipe = (
        alpha_stage
        and fixed_recipe_economic_metric_call_count(
            tuple(value.recipe_id for value in request.model_recipes)
        )
        > 0
    )
    if fixed_economic_recipe:
        # Planning reservation, not a host-independent guarantee: retain the
        # accepted 513.580-second two-view Ridge run, add two bounded waves for
        # the fixed 300-iteration tree candidate, then round up materially.
        estimated_wall = 1_200
    elif ridge_only_lean:
        estimated_wall = math.ceil(
            accepted_lean_wave_seconds * 1.20 * ((fold_count + worker_count - 1) // worker_count)
        )
    elif request.execution_stage == "SCORE_FILTER_ONLY":
        estimated_wall = math.ceil(accepted_score_filter_seconds * 1.20)
    else:
        estimated_wall = max(
            1,
            int(actual_fit_calls * 11.0 / worker_count + actual_solver_calls * 0.16),
        )
    return PanelResearchPreflight.create(
        request_hash=request.request_hash,
        source_resolution_hash=source_resolution_hash,
        feature_preflight_hash=feature_preflight_hash,
        actual_feature_count=feature_count,
        actual_candidate_count=scientific_candidates,
        actual_fit_calls=actual_fit_calls,
        actual_predict_calls=actual_predict_calls,
        actual_metric_calls=actual_metric_calls,
        actual_solver_calls=actual_solver_calls,
        actual_portfolio_formation_count=formation_count,
        actual_portfolio_economic_formation_count=economic_formation_count,
        actual_portfolio_listing_count=portfolio_listing_count,
        portfolio_common_session_axis_hash=portfolio_session_axis_hash,
        portfolio_economic_session_axis_hash=portfolio_economic_session_axis_hash,
        portfolio_common_listing_axis_hash=portfolio_listing_axis_hash,
        portfolio_common_axis_hash=portfolio_axis_hash,
        estimated_wall_seconds=estimated_wall,
        estimated_peak_memory_bytes=estimated_peak_memory,
        reuse_actions=(
            *(
                ("SEALED_ALPHA_ROOT_SCORE_AND_ROW_AXIS_HANDOFF",)
                if request.execution_stage == "PORTFOLIO_ONLY"
                else ("FEATURE_T_BASE_FORMULA_CONTENT_ADDRESSED_CLOSURE",)
            ),
            "E2F43C4_FACTOR_INDEPENDENT_TOTAL_RETURN_TARGET_AND_OUTCOME_LANES",
            *(
                ("INSTALLED_RISK_METHOD_CONTENT_ADDRESSED_COVARIANCE_SURFACES",)
                if portfolio_stage
                else ()
            ),
        ),
        recompute_actions=(
            *(
                (
                    "CORRECTED_SECTOR_AND_MARKET_CONTEXT_SURFACES",
                    "ROLE_CORRECT_FEATURE_PROJECTIONS",
                    "INNER_AND_OUTER_SCALE_RECEIPTS",
                    "ALPHA_MODEL_EVIDENCE",
                )
                if alpha_stage
                else ()
            ),
            *(("DECLARED_SCORE_FILTER_AND_PERSISTENCE_EVIDENCE",) if filter_stage else ()),
            *(
                (
                    "INSTALLED_TARGET_FREE_FIXED_SCORE_EVIDENCE",
                    "INSTALLED_PORTFOLIO_AND_OOS_EVIDENCE",
                )
                if portfolio_stage
                else ()
            ),
        ),
        caps=caps,
    )


class PanelResearchMethodologyCompiler:
    """Compile the installed methodology under the existing Alpha Desk kind."""

    def __init__(
        self,
        *,
        plan: PanelFeaturePlan | None,
        source_resolution_hash: str,
        preflight: PanelResearchPreflight,
        portfolio_policy_recipe_hash: str | None,
        portfolio_preflight_source_hash: str | None = None,
        fixed_feature_view_binding_hashes: Mapping[str, str] | None = None,
        catalog: AlphaResearchMethodologyCatalog | None = None,
    ) -> None:
        self.plan = plan
        self.source_resolution_hash = source_resolution_hash
        self.preflight = preflight
        self.portfolio_policy_recipe_hash = portfolio_policy_recipe_hash
        self.portfolio_preflight_source_hash = portfolio_preflight_source_hash
        self.fixed_feature_view_binding_hashes = dict(fixed_feature_view_binding_hashes or {})
        self.catalog = catalog or AlphaResearchMethodologyCatalog()

    def compile(
        self,
        *,
        envelope: ResearchExperimentEnvelope,
        section: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
    ) -> DeskProgramCompilation:
        request = PanelResearchMethodologyRequest.from_mapping(section)
        descriptor = self.catalog.resolve(request.methodology_id)
        if request.request_hash != self.preflight.request_hash:
            raise AuthoringError("alpha_research.panel_methodology_preflight_request_mismatch")
        if self.plan is not None:
            selected_view_binding_hashes = tuple(
                self.plan.catalog.resolve(value).recipe_hash for value in request.feature_view_ids
            )
            feature_preflight_hash = self.plan.preflight.preflight_hash
        else:
            try:
                selected_view_binding_hashes = tuple(
                    self.fixed_feature_view_binding_hashes[value]
                    for value in request.feature_view_ids
                )
            except KeyError as error:
                raise AuthoringError(
                    "alpha_research.panel_methodology_view_resolution_mismatch"
                ) from error
            feature_preflight_hash = self.preflight.feature_preflight_hash
        linear_domain = build_dynamic_panel_regularized_linear_search_domain()
        lightgbm_domain = build_dynamic_panel_lightgbm_search_domain()
        parameter_domain_hash = str(
            canonical_hash(
                {
                    "selected_view_recipe_hashes": list(selected_view_binding_hashes),
                    "linear_search_domain_hash": linear_domain.search_domain_hash,
                    "lightgbm_search_domain_hash": lightgbm_domain.search_domain_hash,
                    "score_filter_spec_hashes": [
                        value.spec_hash for value in request.score_filter_candidates
                    ],
                    "score_filter_selection_method_id": (request.score_filter_selection_method_id),
                    "persistence_lags": list(request.persistence_lags),
                    "portfolio_recipe_hash": (
                        self.portfolio_policy_recipe_hash
                        if request.execution_stage in {"PORTFOLIO_ONLY", "END_TO_END"}
                        else None
                    ),
                    "portfolio_preflight_source_hash": (
                        self.portfolio_preflight_source_hash
                        if request.execution_stage == "PORTFOLIO_ONLY"
                        else None
                    ),
                    "risk_aversion_grid": list(request.risk_aversion_grid),
                    "turnover_regularization_grid": list(request.turnover_regularization_grid),
                    "sector_capacities": list(request.sector_capacities),
                    "cost_stress_bps": list(request.cost_stress_bps),
                }
            )
        )
        method_binding_hash = str(
            canonical_hash(
                {
                    "methodology_descriptor_hash": descriptor.descriptor_hash,
                    "request_hash": request.request_hash,
                    "authority_hash": authority.authority_hash,
                    "source_resolution_hash": self.source_resolution_hash,
                    "feature_preflight_hash": feature_preflight_hash,
                    "preflight_hash": self.preflight.preflight_hash,
                    "parameter_domain_hash": parameter_domain_hash,
                }
            )
        )
        # Per-method catalog identity deliberately excludes unselected peers.
        # Installing method B therefore leaves method A's Program unchanged.
        catalog_hash = str(
            canonical_hash(
                {
                    "methodology_descriptor_hash": descriptor.descriptor_hash,
                    "selected_view_recipe_hashes": list(selected_view_binding_hashes),
                }
            )
        )
        desk_program_hash = str(
            canonical_hash(
                {
                    "kind": "alpha.model-development",
                    "envelope_hash": envelope.envelope_hash,
                    "authority_hash": authority.authority_hash,
                    "method_binding_hash": method_binding_hash,
                    "catalog_hash": catalog_hash,
                }
            )
        )
        return DeskProgramCompilation(
            desk_program_hash=desk_program_hash,
            catalog_hash=catalog_hash,
            method_binding_hash=method_binding_hash,
            parameter_domain_hash=parameter_domain_hash,
        )


def methodology_section(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    section = document.get("alpha")
    if not isinstance(section, Mapping):
        return None
    method_id = section.get("methodology_id")
    return section if method_id == "PAIRED_PANEL_PORTFOLIO_RESEARCH" else None


__all__ = [
    "AlphaResearchMethodologyCatalog",
    "AlphaResearchMethodologyDescriptor",
    "PanelResearchMethodologyCompiler",
    "PanelResearchMethodologyRequest",
    "PanelResearchPreflight",
    "PanelResearchRuntimeCaps",
    "build_panel_research_preflight",
    "methodology_section",
    "panel_alpha_configuration_hash",
]
