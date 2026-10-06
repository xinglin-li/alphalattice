"""The installed frozen strategies: declarations, book policies and score sources.

This is the one production module that knows a strategy by name. It declares
the two installed packages through the generic contract, owns the Portfolio
half of their book policies, and installs the score sources that make each
declared capability runnable. Product Host composes what `install_frozen_strategies`
returns and never learns which package answered.

The predecessor leaves its holdings controls where the shared catalog admits
them; the successor freezes `Top35 / Exit70 / Tranche3 / EW` and carries four
component books merged post-trade at `25/25/25/25`, obtained either from the
sealed Gate I score arrays or by scoring the exact admitted forty-eight child
models on the current axis.

The two book recipes keep their published `kind` strings and therefore their
published `recipe_hash`. They moved module, not identity: a frozen policy that
changed hash because its file moved would invalidate every artifact sealed
under it for no scientific reason.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, Protocol, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioPerSideCostAssumption,
)
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    COMPONENT_IDS,
    HETEROGENEOUS_STRATEGY_ID,
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
    SUCCESSOR_PACKAGE_HASH,
    ComponentId,
    HeterogeneousChildModelIdentity,
    HeterogeneousModelSetAuthority,
    array_value_hash,
)
from alphalattice.investment.alpha_research.scores.heterogeneous_replay import (
    HeterogeneousReplayError,
    admit_heterogeneous_current_closure,
    admit_successor_evidence,
)
from alphalattice.investment.alpha_research.scores.product_recipe import (
    INSTALLED_ALPHA_PRODUCT_RECIPE,
)
from alphalattice.investment.alpha_research.scores.product_replay import (
    INSTALLED_EVIDENCE_MANIFEST_SHA256,
    AdmittedProductEvidence,
    AlphaProductScoreProjection,
    HeterogeneousFormationScoreInput,
    HeterogeneousLiveModel,
    HeterogeneousPredictionOwner,
    HeterogeneousScoringError,
    HeterogeneousVintageFeatureSurface,
    live_vintages,
    replay_formation_scores,
    score_heterogeneous_component,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioResearchSpec,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    ComponentPlanEntry,
    FrozenStrategyPackage,
    InstalledPackageBinding,
    PackageControlSurface,
    PackageFrozenControl,
    PackagePolicyIdentity,
    ScoreSourceCapability,
    ScoreSourceMode,
    ScoreSupport,
    SharedPortfolioInputs,
    StrategyComponentResolution,
    StrategyPackageError,
    StrategyScoreSource,
    tranche_recipe_of,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    CappedSleeveComponent,
    ComponentBookFactory,
    TrancheBookComponent,
    TrancheExecutionError,
    TrancheFormationInputs,
)
from alphalattice.investment.portfolio_strategy_lab.policies.buffered_rank_return import (
    build_causal_rank_return_curve,
)
from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
    CausalRankReturnCurveSlice,
)
from alphalattice.investment.portfolio_strategy_lab.policies.lifecycle_research import (
    ARTIFACT_KEY as LIFECYCLE_RESEARCH_ARTIFACT_KEY,
)
from alphalattice.investment.portfolio_strategy_lab.policies.lifecycle_research import (
    LifecycleResearchScoreSource,
)
from alphalattice.investment.portfolio_strategy_lab.policies.post_observed_authority import (
    FAST_REBOUND_COMPONENT_ID,
    TREND_CANDIDATE_COMPONENT_ID,
    AdmittedPostObservedStrategyAuthority,
    FrozenHistoricalBookRecipe,
    FrozenHistoricalComponentRecipe,
    PostObservedAuthorityError,
    PostObservedHistoricalScoreSource,
    admit_post_observed_strategy_authority,
)
from alphalattice.investment.risk_research.surfaces.decomposition import (
    RiskAllocationProjection,
    RiskAttributionProjection,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.recipe_identity import recipe_identity, recipe_seal_holds
from alphalattice.kernel.shared_kernel.source_identity import (
    switched_source_identity,
)

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]

BROAD_FEATURE_STRATEGY_ID: Final = "BROAD_FEATURE_BOOK"
PRODUCT_COMPONENT_ID: Final = "BROAD_ENSEMBLE"

HETEROGENEOUS_CONTROLS_FROZEN: Final = "portfolio_application.heterogeneous_controls_frozen"
"""The successor's own refusal code, unchanged from the predecessor Gate."""

POST_OBSERVED_CONTROLS_FROZEN: Final = "portfolio_application.post_observed_controls_frozen"
"""One refusal shared by the two immutable post-observed package surfaces."""

REBOUND_RETURN_STRATEGY_ID: Final = "RETURN_G6_MU_ONLY"
TREND_REBOUND_STRATEGY_ID: Final = "BALANCED_G2_G6_EQUAL_CAPITAL"
_RECIPE_SEAL: Final = frozenset({"recipe_hash"})
TREND_CANDIDATE_RESEARCH_RECIPE_HASH: Final = (
    "145f2cdc492e2c12dae67426eb5753f8d73a4295681ae5bc474ab9aa17ae0e28"
)
FAST_REBOUND_RESEARCH_RECIPE_HASH: Final = (
    "a2896193b0575e37d4ad99ed5b2bdbdd29bab9a333298e9bddaaaa5f251ee61f"
)

SHARED_CONTROL_IDS: Final[tuple[str, ...]] = (
    "cost_bps_per_side",
    "secondary_benchmark_view",
    "report_unit",
    "study_start",
    "study_end",
)
"""Cost, benchmark, report and window: available under every installed package."""

HOLDINGS_CONTROL_IDS: Final[tuple[str, ...]] = (
    "top_k",
    "tranches",
    "exit_rank",
    "weight_rule",
)


class InstalledStrategyError(ValueError):
    """Stable refusal for an installed book recipe identity failure."""


# ------------------------------------------------------------- book policies


class ComponentBookRecipe(BaseModel):  # type: ignore[misc]
    """Exact closed-form component book frozen by the successor's Gate I package.

    The predecessor public recipe resets every reviewed sleeve to ``1/T``.
    This one deliberately does not: the reviewed sleeve keeps the capital share
    it reached through the shared engine's drift. Keeping it a distinct typed
    recipe prevents a successor call site from silently changing the predecessor
    policy, and makes the historical cap activation explicit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ComponentBookRecipe"] = "ComponentBookRecipe"
    top_k: Literal[35] = 35
    exit_rank: Literal[70] = 70
    tranches: Literal[3] = 3
    weight_rule: Literal["ew"] = "ew"
    schedule_offset: Literal[0] = 0
    true_up: Literal[False] = False
    sleeve_share_policy: Literal["PRESERVE_DRIFTED_SLEEVE_NOTIONAL"] = (
        "PRESERVE_DRIFTED_SLEEVE_NOTIONAL"
    )
    aggregate_name_cap: float = Field(default=0.06, gt=0.0, le=1.0)
    aggregate_cap_start_formation: Literal[525] = 525
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def installed(cls) -> Self:
        """Seal the installed three-sleeve equal-weight component recipe.

        Returns:
            Canonical recipe for 35-name selection, 70 exit rank, preserved drifted sleeve notional
            and the frozen aggregate cap.
        """
        values: dict[str, object] = {
            "kind": "ComponentBookRecipe",
            "top_k": 35,
            "exit_rank": 70,
            "tranches": 3,
            "weight_rule": "ew",
            "schedule_offset": 0,
            "true_up": False,
            "sleeve_share_policy": "PRESERVE_DRIFTED_SLEEVE_NOTIONAL",
            "aggregate_name_cap": 0.06,
            "aggregate_cap_start_formation": 525,
        }
        return cls(**values, recipe_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the frozen component aggregate cap and exact recipe identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            InstalledStrategyError: Aggregate cap differs from 0.06 or recipe_hash differs.
        """
        if self.aggregate_name_cap != 0.06:
            raise InstalledStrategyError("portfolio_strategy_lab.component_book_recipe_invalid")
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise InstalledStrategyError("portfolio_strategy_lab.component_book_recipe_invalid")
        return self


class HeterogeneousBookRecipe(BaseModel):  # type: ignore[misc]
    """The Portfolio-owned half of the frozen heterogeneous successor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    LABELS: ClassVar[frozenset[str]] = frozenset({"component_ids"})
    """The components' names; the strategy hash binds the components in their order (ID10)."""

    kind: Literal["HeterogeneousBookRecipe"] = "HeterogeneousBookRecipe"
    strategy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    component_ids: tuple[ComponentId, ...] = COMPONENT_IDS
    allocation_basis_points: tuple[int, ...] = (2500, 2500, 2500, 2500)
    component_book_recipe: ComponentBookRecipe
    per_side_cost_assumption: PortfolioPerSideCostAssumption
    merge_semantics: Literal["POST_TRADE_LISTING_WEIGHTS_THEN_RECOMPUTE_ECONOMICS"] = (
        "POST_TRADE_LISTING_WEIGHTS_THEN_RECOMPUTE_ECONOMICS"
    )
    risk_for_weights: Literal["NONE_EQUAL_WEIGHT_COMPONENT_BOOKS"] = (
        "NONE_EQUAL_WEIGHT_COMPONENT_BOOKS"
    )
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def installed(cls) -> Self:
        """Seal the installed four-component book, allocation, costs and merge semantics.

        Returns:
            Canonical recipe with equal component allocations, installed component book and
            five-bps-per-side cost declaration.
        """
        values: dict[str, object] = {
            "kind": "HeterogeneousBookRecipe",
            "strategy_hash": INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.strategy_hash,
            "component_ids": list(COMPONENT_IDS),
            "allocation_basis_points": [2500, 2500, 2500, 2500],
            "component_book_recipe": ComponentBookRecipe.installed().model_dump(mode="json"),
            "per_side_cost_assumption": PortfolioPerSideCostAssumption.from_bps_per_side(
                "5"
            ).model_dump(mode="json"),
            "merge_semantics": "POST_TRADE_LISTING_WEIGHTS_THEN_RECOMPUTE_ECONOMICS",
            "risk_for_weights": "NONE_EQUAL_WEIGHT_COMPONENT_BOOKS",
        }
        draft = cls.model_construct(**values, recipe_hash="0" * 64)
        return cls(**values, recipe_hash=recipe_identity(draft, exclude=_RECIPE_SEAL))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact installed components, allocation, sleeve controls, costs and identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            InstalledStrategyError: Component order, equal allocations, declared book controls, cost
                or recipe_hash differs.
        """
        if self.component_ids != COMPONENT_IDS:
            raise InstalledStrategyError("portfolio_strategy_lab.heterogeneous_components_invalid")
        if self.allocation_basis_points != (2500, 2500, 2500, 2500):
            raise InstalledStrategyError("portfolio_strategy_lab.heterogeneous_allocation_invalid")
        book = self.component_book_recipe
        if (
            book.top_k != 35
            or book.exit_rank != 70
            or book.tranches != 3
            or book.weight_rule != "ew"
            or book.true_up
            or book.sleeve_share_policy != "PRESERVE_DRIFTED_SLEEVE_NOTIONAL"
            or book.aggregate_cap_start_formation != 525
        ):
            raise InstalledStrategyError("portfolio_strategy_lab.heterogeneous_book_invalid")
        if self.per_side_cost_assumption.cost_bps_per_side_tenths != 50:
            raise InstalledStrategyError("portfolio_strategy_lab.heterogeneous_cost_invalid")
        if not recipe_seal_holds(self, "recipe_hash"):
            raise InstalledStrategyError("portfolio_strategy_lab.heterogeneous_identity_invalid")
        return self


INSTALLED_HETEROGENEOUS_BOOK_RECIPE: Final = HeterogeneousBookRecipe.installed()

HETEROGENEOUS_POLICY_ADAPTER_BINDING_HASH: Final = canonical_hash(
    {
        "kind": "HeterogeneousTranchePolicyAdapterBinding",
        "recipe_hash": INSTALLED_HETEROGENEOUS_BOOK_RECIPE.recipe_hash,
        "decision_owner": "MergedComponentBookProvider",
        "component_owner": "CappedSleeveBookProvider",
        "mechanics_owner": "SHARED_PORTFOLIO_BACKTESTING",
        "solver_backed": False,
    }
)
"""The successor's policy binding, naming the owners that actually decide."""

HETEROGENEOUS_POLICY_CATALOG_HASH: Final = canonical_hash(
    {
        "kind": "InstalledHeterogeneousPortfolioPolicyCatalog",
        "adapter_binding_hash": HETEROGENEOUS_POLICY_ADAPTER_BINDING_HASH,
    }
)


# -------------------------------------------------------------- declarations

_SHARED_LANES: Final[tuple[str, ...]] = (
    "MARKET_REALIZED_RETURN",
    "TRADABILITY_DECISION_ELIGIBILITY",
    "CAUSAL_EXECUTION_OUTCOME",
    "SECTOR_CLASSIFICATION",
    "ELIGIBLE_UNIVERSE_EQUAL_WEIGHT_BENCHMARK",
)

BROAD_FEATURE_PACKAGE: Final = FrozenStrategyPackage.create(
    strategy_id=BROAD_FEATURE_STRATEGY_ID,
    alpha_recipe_hash=INSTALLED_ALPHA_PRODUCT_RECIPE.recipe_hash,
    policy_binding="REQUEST_SELECTED",
    frozen_policy=None,
    component_plan=(
        ComponentPlanEntry(
            component_id=PRODUCT_COMPONENT_ID,
            allocation_basis_points=10_000,
            family_id="BROAD_ENSEMBLE_MODEL",
            recipe_hash=INSTALLED_ALPHA_PRODUCT_RECIPE.recipe_hash,
            target_recipe="HISTORICAL_ENSEMBLE_TARGET",
            objective="ONE_AGGREGATED_PRODUCT_SCORE_PER_FORMATION",
        ),
    ),
    merge_semantics="SINGLE_COMPONENT_BOOK",
    score_sources=(
        ScoreSourceCapability(
            mode="HISTORICAL_ARRAY_REPLAY",
            evidence_identity_hash=INSTALLED_EVIDENCE_MANIFEST_SHA256,
            description="Twelve admitted broad-ensemble models replayed per formation",
        ),
    ),
    default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
    required_shared_input_lanes=(
        *_SHARED_LANES,
        "RISK_ALLOCATION_AND_ATTRIBUTION",
        "CAUSAL_RANK_RETURN_CURVE",
    ),
    risk_disposition="POLICY_CONSUMED_RISK_ALLOCATION_PLUS_REPORT_ONLY_DFD_PRIME_PLUS_E",
    controls=PackageControlSurface(
        frozen=(),
        admitted=HOLDINGS_CONTROL_IDS,
        shared=SHARED_CONTROL_IDS,
    ),
    claim_limits=(
        "NO_INDEPENDENT_HOLDOUT_CLAIM",
        "NO_CAPACITY_ESTIMATE",
        "NO_CURRENT_OR_PRODUCTION_POINTER_MUTATION",
    ),
    provenance=(
        "ALPHA_SCORES_REPLAYED_FROM_THE_ADMITTED_ENSEMBLE_EVIDENCE_MANIFEST",
        "RISK_DECOMPOSITION_PRODUCED_PER_FORMATION_FROM_THE_CAUSAL_RETURN_SURFACE",
    ),
)


def heterogeneous_package(
    *,
    historical: bool = True,
    current_authority: HeterogeneousModelSetAuthority | None = None,
) -> FrozenStrategyPackage:
    """Declare the successor over exactly the capabilities this desktop installs.

    A declaration is hashed as a whole, so a package that advertised current
    model scoring on a machine that cannot run it would let one package hash
    describe two different runnable surfaces. The capabilities are therefore
    arguments: the historical arrays, and the exact model-set authority current
    scoring binds -- never a mode without the authority behind it.
    """
    sources: list[ScoreSourceCapability] = []
    if historical:
        sources.append(
            ScoreSourceCapability(
                mode="HISTORICAL_ARRAY_REPLAY",
                evidence_identity_hash=SUCCESSOR_PACKAGE_HASH,
                description="Four sealed component score arrays, reopened exactly",
            )
        )
    if current_authority is not None:
        sources.append(
            ScoreSourceCapability(
                mode="CURRENT_MODEL_SCORING",
                evidence_identity_hash=current_authority.authority_hash,
                description=(
                    "The exact admitted forty-eight child models, scored on the current axis"
                ),
            )
        )
    if not sources:
        raise StrategyPackageError("portfolio_application.package_no_capability_installed")
    return FrozenStrategyPackage.create(
        strategy_id=HETEROGENEOUS_STRATEGY_ID,
        alpha_recipe_hash=INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.strategy_hash,
        policy_binding="PACKAGE_FROZEN",
        frozen_policy=PackagePolicyIdentity(
            policy_recipe_hash=INSTALLED_HETEROGENEOUS_BOOK_RECIPE.recipe_hash,
            policy_catalog_hash=HETEROGENEOUS_POLICY_CATALOG_HASH,
            policy_adapter_binding_hash=HETEROGENEOUS_POLICY_ADAPTER_BINDING_HASH,
        ),
        component_plan=tuple(
            ComponentPlanEntry(
                component_id=component.component_id,
                allocation_basis_points=2500,
                family_id=component.family_id,
                recipe_hash=component.recipe_hash,
                target_recipe=component.target_recipe,
                objective=component.objective,
            )
            for component in INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.components
        ),
        merge_semantics="POST_TRADE_LISTING_WEIGHTS_THEN_RECOMPUTE_ECONOMICS",
        score_sources=tuple(sources),
        default_score_source_mode=sources[0].mode,
        required_shared_input_lanes=(*_SHARED_LANES, "RISK_ATTRIBUTION_REPORT_ONLY"),
        risk_disposition="REPORT_ONLY_DFD_PRIME_PLUS_E_NEVER_USED_FOR_WEIGHTS",
        controls=PackageControlSurface(
            frozen=(
                PackageFrozenControl(
                    control_id="top_k",
                    frozen_display="35",
                    refusal_code=HETEROGENEOUS_CONTROLS_FROZEN,
                    reason="Each component book holds exactly 35 names",
                ),
                PackageFrozenControl(
                    control_id="exit_rank",
                    frozen_display="70",
                    refusal_code=HETEROGENEOUS_CONTROLS_FROZEN,
                    reason="Hysteresis exit is frozen at rank 70",
                ),
                PackageFrozenControl(
                    control_id="tranches",
                    frozen_display="3",
                    refusal_code=HETEROGENEOUS_CONTROLS_FROZEN,
                    reason="Three sleeves per component book",
                ),
                PackageFrozenControl(
                    control_id="weight_rule",
                    frozen_display="ew",
                    refusal_code=HETEROGENEOUS_CONTROLS_FROZEN,
                    reason="Component books are equal weight and read no risk lane",
                ),
            ),
            admitted=(),
            shared=SHARED_CONTROL_IDS,
        ),
        claim_limits=(
            "NO_INDEPENDENT_HOLDOUT_CLAIM",
            "NO_DOMINANCE_CLAIM",
            "NO_CAPACITY_ESTIMATE",
            "NO_CURRENT_OR_PRODUCTION_POINTER_MUTATION",
            "WEIGHT_PARITY_ASSUMES_FULL_EXECUTION_AVAILABILITY",
        ),
        provenance=(
            "FOUR_INDEPENDENT_COMPONENT_BOOKS_MERGED_POST_TRADE_AT_TWENTY_FIVE_PERCENT_EACH",
            "COMPONENT_SCORES_NEVER_BLENDED_BEFORE_SELECTION",
        ),
    )


REBOUND_RETURN_BOOK_RECIPE: Final = FrozenHistoricalBookRecipe(
    strategy_id=REBOUND_RETURN_STRATEGY_ID,
    components=(
        FrozenHistoricalComponentRecipe(
            component_id=FAST_REBOUND_COMPONENT_ID,
            allocation_basis_points=10_000,
            weight_rule="mu.iv0",
        ),
    ),
)

TREND_REBOUND_BOOK_RECIPE: Final = FrozenHistoricalBookRecipe(
    strategy_id=TREND_REBOUND_STRATEGY_ID,
    components=(
        FrozenHistoricalComponentRecipe(
            component_id=TREND_CANDIDATE_COMPONENT_ID,
            allocation_basis_points=5_000,
            weight_rule="ew",
            outsider_sentinel=-1.0,
        ),
        FrozenHistoricalComponentRecipe(
            component_id=FAST_REBOUND_COMPONENT_ID,
            allocation_basis_points=5_000,
            weight_rule="ew",
        ),
    ),
)

FROZEN_RESEARCH_BOOK_RECIPES: Final = (REBOUND_RETURN_BOOK_RECIPE, TREND_REBOUND_BOOK_RECIPE)


_BOOK_ROLE_KEYS: Final[dict[str, str]] = {
    HETEROGENEOUS_STRATEGY_ID: "four_component",
    REBOUND_RETURN_STRATEGY_ID: "rebound_return",
    TREND_REBOUND_STRATEGY_ID: "trend_rebound",
}
"""Each frozen book's readout roles, named by what the book holds: the strategy ids workspaces
store keep their spelling until the release's corpora are prepared fresh (NM2, V451)."""

BOOK_PARTS: Final = ("book_recipe", "policy_binding", "policy_catalog", "book_alpha_recipe")
"""What each frozen book's roles read: its recipe, its policy adapter binding and catalog, and
for a historical book the Alpha recipe its components score with."""


def book_part_role(strategy_id: str, part: str) -> str:
    """The readout role of one part of a frozen book (NM1)."""
    return f"portfolio_strategy_lab.{part}.{_BOOK_ROLE_KEYS[strategy_id]}"


def installed_book_part_hash(book: str, part: str) -> str:
    """One part of a frozen book this build installs, by its role's key: the role's value (NM1)."""
    strategy_id = next((value for value, key in _BOOK_ROLE_KEYS.items() if key == book), None)
    values: dict[str, str]
    if strategy_id == HETEROGENEOUS_STRATEGY_ID:
        values = {
            "book_recipe": INSTALLED_HETEROGENEOUS_BOOK_RECIPE.recipe_hash,
            "policy_binding": HETEROGENEOUS_POLICY_ADAPTER_BINDING_HASH,
            "policy_catalog": HETEROGENEOUS_POLICY_CATALOG_HASH,
        }
    else:
        recipe = next(
            (value for value in FROZEN_RESEARCH_BOOK_RECIPES if value.strategy_id == strategy_id),
            None,
        )
        if recipe is None:
            raise InstalledStrategyError("portfolio_strategy_lab.book_role_unknown")
        alpha, binding, catalog = _historical_book_identities(recipe)
        values = {
            "book_recipe": recipe.recipe_hash,
            "policy_binding": binding,
            "policy_catalog": catalog,
            "book_alpha_recipe": alpha,
        }
    if part not in values:
        raise InstalledStrategyError("portfolio_strategy_lab.book_role_unknown")
    return values[part]


_POST_OBSERVED_COMPONENT_ALPHA: Final = {
    TREND_CANDIDATE_COMPONENT_ID: TREND_CANDIDATE_RESEARCH_RECIPE_HASH,
    FAST_REBOUND_COMPONENT_ID: FAST_REBOUND_RESEARCH_RECIPE_HASH,
}


def _post_observed_package(
    *,
    authority: AdmittedPostObservedStrategyAuthority,
    recipe: FrozenHistoricalBookRecipe,
) -> FrozenStrategyPackage:
    """Declare one historical package without teaching the Host its name."""
    return frozen_book_package(recipe=recipe, evidence_identity_hash=authority.authority_hash)


def _historical_book_identities(recipe: FrozenHistoricalBookRecipe) -> tuple[str, str, str]:
    """A historical book's Alpha recipe, policy adapter binding and policy catalog hashes."""
    alpha_recipe_hash = canonical_hash(
        {
            "kind": "PostObservedHistoricalAlphaRecipe",
            # Each component by its research recipe, in order; its name stays out (ID10).
            "components": [
                {
                    "allocation_basis_points": value.allocation_basis_points,
                    "recipe_hash": _POST_OBSERVED_COMPONENT_ALPHA[value.component_id],
                }
                for value in recipe.components
            ],
        }
    )
    policy_adapter_hash = canonical_hash(
        {
            "kind": "PostObservedFrozenBookPolicyAdapterBinding",
            "recipe_hash": recipe.recipe_hash,
            "component_owner": "CappedSleeveBookProvider",
            "merge_owner": "MergedComponentBookProvider",
            "risk_for_weights": "NONE",
        }
    )
    policy_catalog_hash = canonical_hash(
        {
            "kind": "PostObservedFrozenBookPolicyCatalog",
            "adapter_binding_hash": policy_adapter_hash,
        }
    )
    return alpha_recipe_hash, policy_adapter_hash, policy_catalog_hash


def frozen_book_package(
    *,
    recipe: FrozenHistoricalBookRecipe,
    evidence_identity_hash: str,
    source_description: str = (
        "Post-observed trend-candidate and fast-rebound historical score arrays under the sealed "
        "post-observed authority"
    ),
    extra_claim_limits: tuple[str, ...] = (),
) -> FrozenStrategyPackage:
    """One frozen policy/control declaration; provenance belongs to the score source."""
    component_plan = tuple(
        ComponentPlanEntry(
            component_id=value.component_id,
            allocation_basis_points=value.allocation_basis_points,
            family_id=value.component_id,
            recipe_hash=_POST_OBSERVED_COMPONENT_ALPHA[value.component_id],
            target_recipe=(
                "T1_H3_PURE_TOTAL_RETURN_Z"
                if value.component_id == TREND_CANDIDATE_COMPONENT_ID
                else "EXACT_FROZEN_G0_H1_WHOLE_UNIVERSE_TARGET"
            ),
            objective=(
                "FORMATION_CLOSE_RAW_12_1_TOP100_CANDIDATE_PERCENTILE"
                if value.component_id == TREND_CANDIDATE_COMPONENT_ID
                else "FULL_UNIVERSE_FAST_REBOUND_SCORE"
            ),
        )
        for value in recipe.components
    )
    alpha_recipe_hash, policy_adapter_hash, policy_catalog_hash = _historical_book_identities(
        recipe
    )
    weight_display = recipe.components[0].weight_rule
    return FrozenStrategyPackage.create(
        strategy_id=recipe.strategy_id,
        alpha_recipe_hash=alpha_recipe_hash,
        policy_binding="PACKAGE_FROZEN",
        frozen_policy=PackagePolicyIdentity(
            policy_recipe_hash=recipe.recipe_hash,
            policy_catalog_hash=policy_catalog_hash,
            policy_adapter_binding_hash=policy_adapter_hash,
        ),
        component_plan=component_plan,
        merge_semantics=(
            "SINGLE_COMPONENT_BOOK"
            if len(component_plan) == 1
            else "POST_TRADE_LISTING_WEIGHTS_THEN_RECOMPUTE_ECONOMICS"
        ),
        score_sources=(
            ScoreSourceCapability(
                mode="HISTORICAL_ARRAY_REPLAY",
                evidence_identity_hash=evidence_identity_hash,
                description=source_description,
            ),
        ),
        default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
        required_shared_input_lanes=(*_SHARED_LANES, "RISK_ATTRIBUTION_REPORT_ONLY"),
        risk_disposition="REPORT_ONLY_DFD_PRIME_PLUS_E_NEVER_USED_FOR_WEIGHTS",
        controls=PackageControlSurface(
            frozen=(
                PackageFrozenControl(
                    control_id="top_k",
                    frozen_display="35",
                    refusal_code=POST_OBSERVED_CONTROLS_FROZEN,
                    reason="The frozen component book holds 35 names",
                ),
                PackageFrozenControl(
                    control_id="exit_rank",
                    frozen_display="70",
                    refusal_code=POST_OBSERVED_CONTROLS_FROZEN,
                    reason="The frozen hysteresis exit rank is 70",
                ),
                PackageFrozenControl(
                    control_id="tranches",
                    frozen_display="3",
                    refusal_code=POST_OBSERVED_CONTROLS_FROZEN,
                    reason="The frozen stateful book has three sleeves",
                ),
                PackageFrozenControl(
                    control_id="weight_rule",
                    frozen_display=weight_display,
                    refusal_code=POST_OBSERVED_CONTROLS_FROZEN,
                    reason=(
                        "The strategy's stock sizing rule is frozen by its post-observed authority"
                    ),
                ),
            ),
            admitted=(),
            shared=SHARED_CONTROL_IDS,
        ),
        claim_limits=(
            "POST_OBSERVED_NOT_PROSPECTIVE_VALIDATION",
            "CURRENT_UNIVERSE_RESEARCH_COUNTERFACTUAL",
            "NO_PIT_UNIVERSE_OR_DELISTING_RETURN_AUTHORITY",
            "NO_CAPACITY_OR_IMPACT_ESTIMATE_BEYOND_FIXED_5_10_BPS",
            "NO_CURRENT_SCORING_OR_PRODUCTION_POINTER_MUTATION",
            *extra_claim_limits,
        ),
        provenance=(
            "DECISION_AT_CLOSE_T_EXECUTION_AT_OPEN_T_PLUS_1",
            "WEIGHTS_ARE_PARITY_ORACLES_NOT_RUNTIME_INPUTS",
            "POST_TRADE_LISTING_WEIGHTS_MERGED_BEFORE_ONE_COST_APPLICATION",
        ),
    )


# ---------------------------------------------------- the predecessor's lane


def _alpha_support(evidence: AdmittedProductEvidence) -> tuple[tuple[date, ...], tuple[str, ...]]:
    manifest = evidence.manifest()
    sessions_by_vintage: dict[str, set[date]] = {}
    listings_by_vintage: dict[str, set[str]] = {}
    for vintage in manifest.model_lineage:
        axis = evidence.prediction_axis(vintage)
        sessions_by_vintage[vintage] = set(axis.row_sessions)
        listings_by_vintage[vintage] = set(axis.row_listing_ids)
    candidates = sorted(set().union(*sessions_by_vintage.values()))
    count = INSTALLED_ALPHA_PRODUCT_RECIPE.vintage_count
    sessions = tuple(
        session
        for session in candidates
        if all(
            vintage in sessions_by_vintage and session in sessions_by_vintage[vintage]
            for vintage in live_vintages(session, count=count)
        )
    )
    if not sessions:
        raise StrategyPackageError("portfolio_application.alpha_support_empty")
    required_vintages = {
        vintage for session in sessions for vintage in live_vintages(session, count=count)
    }
    listings = tuple(sorted(set.intersection(*(listings_by_vintage[v] for v in required_vintages))))
    if len(listings) < 80:
        raise StrategyPackageError("portfolio_application.alpha_support_too_small")
    return sessions, listings


def build_replayed_formation_inputs(
    *,
    evidence: AdmittedProductEvidence,
    formation_sessions: tuple[date, ...],
    ordered_listing_ids: tuple[str, ...],
    decision_eligible: BoolArray,
    risk_by_session: Mapping[date, RiskAllocationProjection] | None,
    risk_attribution_by_session: Mapping[date, RiskAttributionProjection],
    curve_by_session: Mapping[date, CausalRankReturnCurveSlice] | None,
    projections: tuple[AlphaProductScoreProjection, ...] | None = None,
) -> tuple[tuple[TrancheFormationInputs, ...], tuple[AlphaProductScoreProjection, ...]]:
    """Assemble the predecessor's formation inputs from admitted evidence.

    This is the production route from the frozen Alpha recipe to the book: the
    replay owner aggregates the twelve live models into one score per formation,
    and those scores become ``TrancheFormationInputs.scores``. The Risk and
    causal-`mu` lanes arrive through their own typed owners and are passed
    through untouched -- this function composes, and estimates nothing.

    Eligibility handed to the policy is the **admitted live set**, so a name no
    admitted model scored cannot be selected. The policy's own finite-score check
    then sees the same answer from the other direction, which is deliberate: the
    two disagree only if something upstream is wrong.

    ``projections`` lets a caller that has already replayed these formations hand
    them in rather than paying for a second pass. The score source does exactly
    that, because it replays the full common axis once to build the rank curve
    and then forms books over the executable suffix of the same result.
    """
    if decision_eligible.shape != (len(formation_sessions), len(ordered_listing_ids)):
        raise TrancheExecutionError(
            "portfolio_strategy_lab.tranche_execution_eligibility_axis_invalid"
        )
    if not formation_sessions:
        raise TrancheExecutionError("portfolio_strategy_lab.tranche_execution_no_formations")
    if projections is not None and len(projections) != len(formation_sessions):
        raise StrategyPackageError("portfolio_application.replayed_projection_axis_invalid")

    inputs: list[TrancheFormationInputs] = []
    replayed: list[AlphaProductScoreProjection] = []
    for index, session in enumerate(formation_sessions):
        projection = (
            projections[index]
            if projections is not None
            else replay_formation_scores(
                evidence=evidence,
                formation_session=session,
                ordered_listing_ids=ordered_listing_ids,
                decision_eligible=np.asarray(decision_eligible[index], dtype=np.bool_),
            )
        )
        # `None` for a whole lane means the selected rule does not read it, which
        # is different from a lane that is missing a session. The first is a
        # legitimate configuration; the second is a resolution defect.
        risk = None if risk_by_session is None else risk_by_session.get(session)
        risk_attribution = risk_attribution_by_session.get(session)
        curve = None if curve_by_session is None else curve_by_session.get(session)
        if risk_by_session is not None and risk is None:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_risk_projection_absent"
            )
        if risk_attribution is None:
            raise TrancheExecutionError(
                "portfolio_strategy_lab.tranche_execution_risk_attribution_absent"
            )
        if curve_by_session is not None and curve is None:
            raise TrancheExecutionError("portfolio_strategy_lab.tranche_execution_curve_absent")
        inputs.append(
            TrancheFormationInputs(
                formation_session=session,
                scores=projection.scores,
                decision_eligible=np.asarray(projection.live, dtype=np.bool_),
                risk_allocation=risk,
                risk_attribution=risk_attribution,
                causal_rank_return_curve=curve,
                score_projection=projection,
            )
        )
        replayed.append(projection)
    return tuple(inputs), tuple(replayed)


class ProductScoreSource:
    """The predecessor's own score lane: twelve admitted models per formation.

    It owns the causal rank curve too, because the curve is built *from these
    scores*: a shared resolver that produced it would have to replay a strategy
    nobody selected.
    """

    def __init__(self, *, alpha_evidence_root: Path) -> None:
        """Bind installed historical replay to a caller-supplied evidence copy.

        Args:
            alpha_evidence_root: Root of the exact replay evidence copy.
        """
        self._root = alpha_evidence_root.resolve()

    @property
    def mode(self) -> ScoreSourceMode:
        """Declare historical array replay for the installed score source.

        Returns:
            HISTORICAL_ARRAY_REPLAY.
        """
        return "HISTORICAL_ARRAY_REPLAY"

    @property
    def evidence_identity_hash(self) -> str:
        """Read the installed replay evidence manifest identity.

        Returns:
            Exact installed manifest SHA256.
        """
        return str(INSTALLED_EVIDENCE_MANIFEST_SHA256)

    def evidence(self) -> AdmittedProductEvidence:
        """The admitted predecessor evidence, opened from the installed root."""
        return AdmittedProductEvidence(
            root=self._root,
            recipe=INSTALLED_ALPHA_PRODUCT_RECIPE,
            expected_manifest_sha256=INSTALLED_EVIDENCE_MANIFEST_SHA256,
        )

    def support(self, *, spec: PortfolioResearchSpec) -> ScoreSupport:
        """Declare installed replay support with causal rank-mu maturity required.

        Args:
            spec: Research controls unused by this fixed support declaration.

        Returns:
            Replay formation/listing support with unconditional causal-curve maturity admission.
        """
        del spec
        sessions, listings = _alpha_support(self.evidence())
        return ScoreSupport(
            owner_id="alpha_product_replay",
            lane="BROAD_ENSEMBLE_DEVELOPMENT_REPLAY_SCORE",
            identity_hash=INSTALLED_EVIDENCE_MANIFEST_SHA256,
            formation_sessions=sessions,
            ordered_listing_ids=listings,
            # Unconditional, as it has always been for this package: the rank-`mu`
            # default reads the curve, and moving the axis with the weight rule
            # would silently re-run every non-`mu` configuration over a longer
            # book than the one its published results were measured on.
            requires_causal_rank_mu_maturity=True,
        )

    def resolve_components(
        self, *, shared: SharedPortfolioInputs, spec: PortfolioResearchSpec
    ) -> StrategyComponentResolution:
        """Replay once, admit the causal mature suffix and supply only recipe-consumed lanes.

        Args:
            shared: Exact common portfolio source, listing and execution support.
            spec: Admitted research controls selecting declared input consumption.

        Returns:
            One tranche component and exact active score receipts; replay projections also supply
            the causal calibration pass.

        Raises:
            StrategyPackageError: Execution events, mature suffix or shared formation support is
                absent/inconsistent.
        """
        evidence = self.evidence()
        listings = shared.ordered_listing_ids
        projections = tuple(
            replay_formation_scores(
                evidence=evidence,
                formation_session=session,
                ordered_listing_ids=listings,
                decision_eligible=np.asarray(
                    shared.common_decision_eligible[index], dtype=np.bool_
                ),
            )
            for index, session in enumerate(shared.common_sessions)
        )
        scores: FloatArray = np.vstack([value.scores for value in projections])
        live: BoolArray = np.vstack([value.live for value in projections])
        if shared.execution_events is None:
            raise StrategyPackageError("portfolio_application.execution_events_absent")
        curve = build_causal_rank_return_curve(
            formation_sessions=shared.common_sessions,
            ordered_listing_ids=listings,
            scores=scores,
            score_authority_identity=canonical_hash(
                tuple(value.projection_hash for value in projections)
            ),
            decision_eligible=live,
            realized_simple_returns=shared.common_realized_simple_returns,
            execution_events=shared.execution_events,
        )
        dispositions = tuple(
            curve.at(index).disposition for index in range(len(shared.common_sessions))
        )
        last_unavailable = max(
            (index for index, value in enumerate(dispositions) if value != "AVAILABLE"),
            default=-1,
        )
        start = last_unavailable + 1
        if start >= len(shared.common_sessions):
            raise StrategyPackageError("portfolio_application.rank_mu_support_absent")
        if shared.common_sessions[start:] != shared.formation_sessions:
            raise StrategyPackageError("portfolio_application.causal_rank_mu_support_mismatch")
        curves = {
            session: CausalRankReturnCurveSlice(
                formation_index=local,
                formation_session=session,
                bucket_means=curve.at(original).bucket_means,
                bucket_support_counts=curve.at(original).bucket_support_counts,
                admitted_formation_count=curve.at(original).admitted_formation_count,
                disposition="AVAILABLE",
                curve_hash=curve.curve_hash,
            )
            for local, (original, session) in enumerate(
                zip(shared.active_rows, shared.formation_sessions, strict=True)
            )
        }
        # Only the lanes the selected rule declares. `ew` reads no risk and no
        # `mu`; `iv*` reads risk but no `mu`. Supplying an unread lane is refused
        # by the policy, and rightly: it would advertise a dependency the book
        # does not have.
        recipe = tranche_recipe_of(spec)
        # The same projections, not a second replay: this is the executable
        # suffix of the pass that built the curve above.
        active_projections = tuple(projections[index] for index in shared.active_rows)
        formations, _ = build_replayed_formation_inputs(
            evidence=evidence,
            formation_sessions=shared.formation_sessions,
            ordered_listing_ids=listings,
            decision_eligible=live[shared.active_rows],
            risk_by_session=(shared.risk_allocation_by_session if recipe.consumes_risk else None),
            risk_attribution_by_session=shared.risk_attribution_by_session,
            curve_by_session=curves if recipe.consumes_mu else None,
            projections=active_projections,
        )
        component = TrancheBookComponent(
            component_id=PRODUCT_COMPONENT_ID,
            allocation_basis_points=10_000,
            recipe=recipe,
            formations=formations,
            ordered_listing_ids=listings,
            sector_exposure_matrix=np.asarray(
                shared.workspace.sector_exposure_matrix, dtype=np.float64
            ),
            equal_weight_sector_exposure=np.asarray(
                shared.workspace.equal_weight_sector_exposure, dtype=np.float64
            ),
        )
        return StrategyComponentResolution(
            components=(component,),
            score_receipt_hashes=tuple(value.projection_hash for value in active_projections),
        )


def product_evidence_closure_reader(
    *, recipe: object = INSTALLED_ALPHA_PRODUCT_RECIPE
) -> Callable[..., tuple[str, str, int] | None]:
    """Verify one predecessor evidence manifest and report what it closes over.

    Handed to the finalization composition rather than imported by it. Which
    Alpha evidence format a release verifies is a property of the installed
    package, and a protected release path that knew the answer would have to
    grow a branch for the next one.
    """

    def _read(*, root: Path, expected_manifest_sha256: str) -> tuple[str, str, int] | None:
        try:
            closure = AdmittedProductEvidence(
                root=root,
                recipe=cast(Any, recipe),
                expected_manifest_sha256=expected_manifest_sha256,
            ).verify_closure()
        except Exception:
            return None
        return (
            str(closure.manifest_sha256),
            str(closure.closure_hash),
            int(closure.child_count),
        )

    return _read


# ----------------------------------------------------- the successor's lanes


def component_formation_receipt(
    *,
    formation_session: date,
    component_rows: tuple[FloatArray, ...],
    strategy_hash: str,
    evidence_manifest_hash: str,
) -> str:
    """One receipt over every component row a formation consumed."""
    row_hashes = [
        hashlib.sha256(np.ascontiguousarray(row, dtype="<f8").tobytes()).hexdigest()
        for row in component_rows
    ]
    return str(
        canonical_hash(
            {
                "strategy_hash": strategy_hash,
                "evidence_manifest_hash": evidence_manifest_hash,
                "formation_session": formation_session.isoformat(),
                "component_row_hashes": row_hashes,
            }
        )
    )


def _component_resolution(
    *,
    rows_by_component: Mapping[ComponentId, FloatArray],
    shared: SharedPortfolioInputs,
    evidence_manifest_hash: str,
    projections_by_component: Mapping[ComponentId, tuple[AlphaProductScoreProjection, ...]]
    | None = None,
) -> StrategyComponentResolution:
    """Four equal-weight component books over one eligibility and one axis."""

    book = INSTALLED_HETEROGENEOUS_BOOK_RECIPE.component_book_recipe
    listings = shared.ordered_listing_ids
    eligible = np.asarray(shared.decision_eligible, dtype=np.bool_)
    components: list[ComponentBookFactory] = []
    for component in COMPONENT_IDS:
        rows = rows_by_component[component]
        if rows.shape != (len(shared.formation_sessions), len(listings)):
            raise StrategyPackageError("portfolio_application.package_component_axis_invalid")
        projections = (
            None if projections_by_component is None else projections_by_component[component]
        )
        formations = tuple(
            TrancheFormationInputs(
                formation_session=session,
                scores=np.asarray(rows[index], dtype=np.float64),
                decision_eligible=np.asarray(
                    eligible[index] & np.isfinite(rows[index]), dtype=np.bool_
                ),
                risk_allocation=None,
                risk_attribution=None,
                causal_rank_return_curve=None,
                score_projection=None if projections is None else projections[index],
            )
            for index, session in enumerate(shared.formation_sessions)
        )
        components.append(
            CappedSleeveComponent(
                component_id=component,
                allocation_basis_points=2_500,
                top_k=book.top_k,
                exit_rank=book.exit_rank,
                tranches=book.tranches,
                aggregate_name_cap=book.aggregate_name_cap,
                aggregate_cap_start_formation=book.aggregate_cap_start_formation,
                formations=formations,
                ordered_listing_ids=listings,
            )
        )
    receipts = tuple(
        component_formation_receipt(
            formation_session=session,
            component_rows=tuple(
                np.asarray(rows_by_component[component][index], dtype=np.float64)
                for component in COMPONENT_IDS
            ),
            strategy_hash=INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.strategy_hash,
            evidence_manifest_hash=evidence_manifest_hash,
        )
        for index, session in enumerate(shared.formation_sessions)
    )
    return StrategyComponentResolution(components=tuple(components), score_receipt_hashes=receipts)


class SuccessorComponentScoreSource:
    """The successor's historical mode: four sealed score arrays, reopened exactly."""

    def __init__(
        self, *, successor_root: Path, component_score_paths: Mapping[ComponentId, Path]
    ) -> None:
        """Bind the exact ordered historical component set to evidence-copy paths.

        Args:
            successor_root: Root of the exact component evidence copy.
            component_score_paths: Paths keyed in installed component order.

        Raises:
            StrategyPackageError: Component mapping order/set differs from the installed
                declaration.
        """
        if tuple(component_score_paths) != COMPONENT_IDS:
            raise StrategyPackageError("portfolio_application.package_component_set_invalid")
        self._root = successor_root.resolve()
        self._paths = {
            component: component_score_paths[component].resolve() for component in COMPONENT_IDS
        }

    @property
    def mode(self) -> ScoreSourceMode:
        """Declare historical array replay for component score arrays.

        Returns:
            HISTORICAL_ARRAY_REPLAY.
        """
        return "HISTORICAL_ARRAY_REPLAY"

    @property
    def evidence_identity_hash(self) -> str:
        """Read the installed historical component package identity.

        Returns:
            Exact package hash.
        """
        return str(SUCCESSOR_PACKAGE_HASH)

    def support(self, *, spec: PortfolioResearchSpec) -> ScoreSupport:
        """Declare the historical component-array lane without inventing support axes.

        Args:
            spec: Research controls unused by this fixed lane declaration.

        Returns:
            Component-array score support bound to the installed package identity.
        """
        del spec
        return ScoreSupport(
            owner_id="heterogeneous_component_arrays",
            lane="COMPONENT_SCORE_ARRAY",
            identity_hash=SUCCESSOR_PACKAGE_HASH,
        )

    def resolve_components(
        self, *, shared: SharedPortfolioInputs, spec: PortfolioResearchSpec
    ) -> StrategyComponentResolution:
        """Admit exact historical component arrays against shared formation/listing support.

        Args:
            shared: Exact common portfolio source, listing and execution support.
            spec: Admitted research controls selecting declared input consumption.

        Returns:
            Component resolution under the admitted package evidence identity.
        """
        del spec
        evidence = admit_successor_evidence(
            self._root,
            formation_sessions=shared.formation_sessions,
            ordered_listing_ids=shared.ordered_listing_ids,
            component_score_paths=self._paths,
        )
        return _component_resolution(
            rows_by_component={
                component: evidence.component(component) for component in COMPONENT_IDS
            },
            shared=shared,
            evidence_manifest_hash=SUCCESSOR_PACKAGE_HASH,
        )


class HeterogeneousCurrentClosure(Protocol):
    """A product-readable closure over the admitted children and their Features.

    What the current mode needs from an installation and nothing it does not:
    the live models per component, the vintage Feature surfaces for a formation,
    the raw momentum lane the candidate rules read, and the prediction owner
    that reopens a payload. It is a protocol because the exact payload reopening
    belongs to the installed Alpha runtime, not to this module.
    """

    @property
    def formation_sessions(self) -> tuple[date, ...]:
        """Every formation this closure can supply Feature surfaces for."""
        ...

    @property
    def ordered_listing_ids(self) -> tuple[str, ...]:
        """Read the declared current-model listing axis.

        Returns:
            Ordered listing identities.
        """
        ...

    @property
    def models_by_component(self) -> Mapping[ComponentId, tuple[HeterogeneousLiveModel, ...]]:
        """Read registered live models grouped by installed component.

        Returns:
            Component mapping to ordered live model tuples.
        """
        ...

    @property
    def prediction_owner(self) -> HeterogeneousPredictionOwner:
        """Read the deterministic prediction owner for current-model scoring.

        Returns:
            Owner that validates and computes predictions.
        """
        ...

    def feature_surfaces(
        self,
        *,
        component: ComponentId,
        formation_session: date,
        ordered_listing_ids: tuple[str, ...],
    ) -> tuple[HeterogeneousVintageFeatureSurface, ...]:
        """Resolve this component's exact vintage feature surfaces for one formation.

        Args:
            component: Installed component identity.
            formation_session: Exact score formation.
            ordered_listing_ids: Admitted listing axis.

        Returns:
            Ordered vintage feature surfaces for deterministic prediction admission.
        """
        ...

    def raw_12_1_momentum(
        self, *, formation_session: date, ordered_listing_ids: tuple[str, ...]
    ) -> FloatArray | None:
        """Resolve the declared raw momentum lane on an exact formation/listing axis.

        Args:
            formation_session: Exact score formation.
            ordered_listing_ids: Admitted listing axis.

        Returns:
            Raw momentum array when available, otherwise None.
        """
        ...


class HeterogeneousCurrentScoreSource:
    """The successor's current mode: score the exact admitted child model set.

    Admission is by identity, not by count, and it covers everything a
    prediction depends on. Before the first prediction every child the closure
    holds is compared with the authority -- order, vintage, seed, recipe,
    content and lineage -- and so is every Feature surface it supplies (source
    and value identity, per component, vintage and formation), every row of the
    momentum lane the candidate rules read, and the listing axis. A complete but
    substituted set, a stale surface, a re-derived momentum lane: each refuses
    with zero predictions and zero Portfolio work. There is no fallback to the
    leading child and no substitution of a historical array.
    """

    def __init__(
        self, *, authority: HeterogeneousModelSetAuthority, closure: HeterogeneousCurrentClosure
    ) -> None:
        """Bind exact model-set authority to deterministic feature/prediction ownership.

        Args:
            authority: Admitted ordered child model authority.
            closure: Owner of feature surfaces, momentum and numerical prediction.
        """
        self._authority = authority
        self._closure = closure

    @property
    def mode(self) -> ScoreSourceMode:
        """Declare scoring under the installed current-model closure.

        Returns:
            CURRENT_MODEL_SCORING.
        """
        return "CURRENT_MODEL_SCORING"

    @property
    def authority(self) -> HeterogeneousModelSetAuthority:
        """Read the retained model-set authority.

        Returns:
            Exact authority supplied at construction.
        """
        return self._authority

    @property
    def closure(self) -> HeterogeneousCurrentClosure:
        """Read the retained deterministic scoring closure.

        Returns:
            Exact closure supplied at construction.
        """
        return self._closure

    @property
    def evidence_identity_hash(self) -> str:
        """Read the retained model-set evidence identity.

        Returns:
            The authority authority_hash.
        """
        return str(self._authority.authority_hash)

    def support(self, *, spec: PortfolioResearchSpec) -> ScoreSupport:
        """The formations the authority names, on the listing axis it names.

        Declared from the authority alone -- nothing is opened -- so `PLAN`
        answers, and refuses an empty intersection, before any array is read.
        """
        del spec
        sessions = tuple(
            session
            for session in self._closure.formation_sessions
            if session in self._authority.formation_sessions
        )
        if not sessions:
            raise StrategyPackageError("portfolio_application.package_current_support_empty")
        return ScoreSupport(
            owner_id="heterogeneous_current_models",
            lane="CURRENT_CHILD_MODEL_SCORE",
            identity_hash=self._authority.authority_hash,
            formation_sessions=sessions,
            ordered_listing_ids=tuple(self._closure.ordered_listing_ids),
        )

    def _admit_children(self) -> Mapping[ComponentId, tuple[HeterogeneousLiveModel, ...]]:
        installed = self._closure.models_by_component
        if tuple(installed) != COMPONENT_IDS:
            raise StrategyPackageError("portfolio_application.package_current_model_set_incomplete")
        for component in COMPONENT_IDS:
            declared = self._authority.component(component).children
            held = tuple(
                HeterogeneousChildModelIdentity(
                    vintage=model.vintage,
                    seed=model.seed,
                    recipe_hash=model.recipe_hash,
                    content_hash=model.estimator.content_hash,
                    lineage_hash=model.lineage_hash,
                )
                for model in installed[component]
            )
            if len(held) != len(declared):
                raise StrategyPackageError(
                    f"portfolio_application.package_current_model_set_incomplete:{component}"
                )
            if held != declared:
                raise StrategyPackageError(
                    f"portfolio_application.package_current_model_set_invalid:{component}"
                )
        return installed

    def _admit_lanes(
        self, *, shared: SharedPortfolioInputs
    ) -> tuple[
        dict[date, FloatArray | None],
        dict[tuple[ComponentId, date], tuple[HeterogeneousVintageFeatureSurface, ...]],
    ]:
        """Read and verify every lane for every formation before anything is scored."""

        listings = shared.ordered_listing_ids
        if canonical_hash(list(listings)) != self._authority.listing_axis_hash:
            raise StrategyPackageError("portfolio_application.package_current_listing_axis_invalid")
        momentum: dict[date, FloatArray | None] = {}
        surfaces: dict[
            tuple[ComponentId, date], tuple[HeterogeneousVintageFeatureSurface, ...]
        ] = {}
        for session in shared.formation_sessions:
            expected_momentum = self._authority.momentum_value_hash(session)
            if (
                expected_momentum is None
                or live_vintages(session, count=len(self._authority.live_vintages))
                != self._authority.live_vintages
            ):
                raise StrategyPackageError(
                    "portfolio_application.package_current_formation_unsupported"
                )
            try:
                lane = self._closure.raw_12_1_momentum(
                    formation_session=session, ordered_listing_ids=listings
                )
            except HeterogeneousReplayError as error:
                raise StrategyPackageError(
                    f"portfolio_application.package_current_lane_unreadable:{error}"
                ) from error
            if lane is None or array_value_hash(lane) != expected_momentum:
                raise StrategyPackageError("portfolio_application.package_current_momentum_invalid")
            momentum[session] = lane
            for component in COMPONENT_IDS:
                try:
                    supplied = self._closure.feature_surfaces(
                        component=component, formation_session=session, ordered_listing_ids=listings
                    )
                except HeterogeneousReplayError as error:
                    raise StrategyPackageError(
                        f"portfolio_application.package_current_lane_unreadable:{error}"
                    ) from error
                if tuple(value.vintage for value in supplied) != self._authority.live_vintages:
                    raise StrategyPackageError(
                        f"portfolio_application.package_current_surface_invalid:{component}"
                    )
                for surface in supplied:
                    expected = self._authority.surface(
                        component_id=component, vintage=surface.vintage, formation_session=session
                    )
                    if (
                        expected is None
                        or surface.source_binding_hash != expected.source_binding_hash
                        or surface.feature_values_hash != expected.feature_values_hash
                    ):
                        raise StrategyPackageError(
                            "portfolio_application.package_current_surface_invalid:"
                            f"{component}:{surface.vintage}"
                        )
                surfaces[(component, session)] = supplied
        return momentum, surfaces

    def resolve_components(
        self, *, shared: SharedPortfolioInputs, spec: PortfolioResearchSpec
    ) -> StrategyComponentResolution:
        """Admit child models/input lanes and score every declared component formation.

        Args:
            shared: Exact common portfolio source, listing and execution support.
            spec: Admitted research controls selecting declared input consumption.

        Returns:
            Component arrays and exact per-formation score receipts bound to the model-set
            authority.
        """
        del spec
        models = self._admit_children()
        momentum, surfaces = self._admit_lanes(shared=shared)
        listings = shared.ordered_listing_ids
        rows: dict[ComponentId, FloatArray] = {
            component: np.full((len(shared.formation_sessions), len(listings)), np.nan)
            for component in COMPONENT_IDS
        }
        projections: dict[ComponentId, list[AlphaProductScoreProjection]] = {
            component: [] for component in COMPONENT_IDS
        }
        for index, session in enumerate(shared.formation_sessions):
            for component in COMPONENT_IDS:
                projection = self._score(
                    component=component,
                    session=session,
                    listings=listings,
                    eligible=np.asarray(shared.decision_eligible[index], dtype=np.bool_),
                    momentum=momentum[session],
                    surfaces=surfaces[(component, session)],
                    models=models[component],
                )
                rows[component][index] = projection.scores
                projections[component].append(projection)
        return _component_resolution(
            rows_by_component=rows,
            shared=shared,
            evidence_manifest_hash=self._authority.authority_hash,
            projections_by_component={
                component: tuple(values) for component, values in projections.items()
            },
        )

    def _score(
        self,
        *,
        component: ComponentId,
        session: date,
        listings: tuple[str, ...],
        eligible: BoolArray,
        momentum: FloatArray | None,
        surfaces: tuple[HeterogeneousVintageFeatureSurface, ...],
        models: tuple[HeterogeneousLiveModel, ...],
    ) -> AlphaProductScoreProjection:
        try:
            inputs = HeterogeneousFormationScoreInput.create(
                formation_session=session,
                ordered_listing_ids=listings,
                decision_eligible=eligible,
                raw_12_1_momentum=momentum,
                feature_surfaces=surfaces,
                models=models,
                model_set_manifest_hash=self._authority.component(component).model_set_hash,
            )
            # The authority is the scoring recipe view: the installed science
            # fields plus the exact Feature axis the children were scored over.
            return score_heterogeneous_component(
                component=self._authority.component(component),
                inputs=inputs,
                prediction_owner=self._closure.prediction_owner,
            )
        except HeterogeneousScoringError as error:
            # A child that did not answer, an axis that did not match, a
            # nonfinite prediction: all refused by name, none skipped.
            raise StrategyPackageError(
                f"portfolio_application.package_current_score_refused:{error}"
            ) from error


# ---------------------------------------------------------------- installation

_ROOT: Final = Path(__file__).resolve().parents[4]

INSTALLED_STRATEGY_SOURCE_HASH: Final = str(
    switched_source_identity(
        {
            value.removesuffix(".py").replace("/", "."): _ROOT / "alphalattice" / value
            for value in (
                "investment/alpha_research/scores/heterogeneous_product.py",
                "investment/alpha_research/scores/heterogeneous_replay.py",
                "investment/portfolio_strategy_lab/policies/installed_strategies.py",
                "investment/portfolio_strategy_lab/policies/post_observed_authority.py",
                "investment/portfolio_strategy_lab/policies/lifecycle_research.py",
            )
        },
        semantic_owner="installed_frozen_strategies",
        numerical_role="score_source_and_book_policy",
    )
)
"""The implementation bytes behind the installed score sources and book policies.

Bound into every Program a package seals, beside the generic path's own source
identity, so an edit to a strategy's scoring code rotates the Programs it
produced -- and an edit to the generic path does not need to know which
strategies exist to do the same.
"""

PRODUCT_EVIDENCE_ROOT_KEY: Final = "BROAD_ENSEMBLE_EVIDENCE_ROOT"
HETEROGENEOUS_SUCCESSOR_PACKAGE_ROOT_KEY: Final = "HETEROGENEOUS_PACKAGE_ROOT"
HETEROGENEOUS_SUCCESSOR_SCORE_KEY_PREFIX: Final = "HETEROGENEOUS_COMPONENT_SCORE:"
HETEROGENEOUS_CURRENT_CLOSURE_ROOT_KEY: Final = "HETEROGENEOUS_CURRENT_CLOSURE_ROOT"
POST_OBSERVED_STRATEGY_AUTHORITY_ROOT_KEY: Final = "POST_OBSERVED_STRATEGY_AUTHORITY_ROOT"
STRATEGY_ARTIFACT_KEYS: Final[tuple[str, ...]] = (
    PRODUCT_EVIDENCE_ROOT_KEY,
    HETEROGENEOUS_SUCCESSOR_PACKAGE_ROOT_KEY,
    *(f"{HETEROGENEOUS_SUCCESSOR_SCORE_KEY_PREFIX}{component}" for component in COMPONENT_IDS),
    HETEROGENEOUS_CURRENT_CLOSURE_ROOT_KEY,
    POST_OBSERVED_STRATEGY_AUTHORITY_ROOT_KEY,
    LIFECYCLE_RESEARCH_ARTIFACT_KEY,
)
"""Every artifact key an installation may name. Anything else is refused."""


def install_frozen_strategies(
    *,
    artifacts: Mapping[str, Path],
) -> tuple[InstalledPackageBinding, ...]:
    """Register the installed packages over the artifacts this desktop has.

    The predecessor is installed from its admitted evidence root. The successor
    is installed only when at least one of its capabilities is: the historical
    arrays from a Gate I package root plus one sealed score array per component,
    and current scoring only from a sealed local closure root whose manifest the
    reader verifies completely -- every pin, every file, every payload -- before
    the mode is declared. No closure root means no such capability; a named
    root that does not verify is a configuration refusal, not a downgrade.
    """
    unknown = sorted(set(artifacts) - set(STRATEGY_ARTIFACT_KEYS))
    if unknown:
        raise StrategyPackageError(
            "portfolio_application.installed_strategy_artifact_unknown:" + ",".join(unknown)
        )
    evidence_root = artifacts.get(PRODUCT_EVIDENCE_ROOT_KEY)
    local_research = artifacts.get(LIFECYCLE_RESEARCH_ARTIFACT_KEY)
    if evidence_root is None and local_research is None:
        raise StrategyPackageError(
            f"portfolio_application.installed_strategy_artifact_absent:{PRODUCT_EVIDENCE_ROOT_KEY}"
        )
    bindings = (
        []
        if evidence_root is None
        else [
            InstalledPackageBinding(
                package=BROAD_FEATURE_PACKAGE,
                sources={
                    "HISTORICAL_ARRAY_REPLAY": ProductScoreSource(alpha_evidence_root=evidence_root)
                },
                source_identity_hash=INSTALLED_STRATEGY_SOURCE_HASH,
            )
        ]
    )
    successor: dict[ScoreSourceMode, StrategyScoreSource] = {}
    successor_root = artifacts.get(HETEROGENEOUS_SUCCESSOR_PACKAGE_ROOT_KEY)
    if successor_root is not None:
        score_paths: dict[ComponentId, Path] = {}
        for component in COMPONENT_IDS:
            path = artifacts.get(f"{HETEROGENEOUS_SUCCESSOR_SCORE_KEY_PREFIX}{component}")
            if path is None:
                raise StrategyPackageError(
                    "portfolio_application.installed_strategy_artifact_absent:"
                    f"{HETEROGENEOUS_SUCCESSOR_SCORE_KEY_PREFIX}{component}"
                )
            score_paths[component] = path
        successor["HISTORICAL_ARRAY_REPLAY"] = SuccessorComponentScoreSource(
            successor_root=successor_root, component_score_paths=score_paths
        )
    current_authority: HeterogeneousModelSetAuthority | None = None
    closure_root = artifacts.get(HETEROGENEOUS_CURRENT_CLOSURE_ROOT_KEY)
    if closure_root is not None:
        try:
            closure = admit_heterogeneous_current_closure(closure_root)
        except HeterogeneousReplayError as error:
            raise StrategyPackageError(
                f"portfolio_application.installed_strategy_closure_refused:{error}"
            ) from error
        current_authority = closure.authority
        successor["CURRENT_MODEL_SCORING"] = HeterogeneousCurrentScoreSource(
            authority=current_authority, closure=closure
        )
    if successor:
        bindings.append(
            InstalledPackageBinding(
                package=heterogeneous_package(
                    historical="HISTORICAL_ARRAY_REPLAY" in successor,
                    current_authority=current_authority,
                ),
                sources=successor,
                source_identity_hash=INSTALLED_STRATEGY_SOURCE_HASH,
            )
        )
    post_observed_root = artifacts.get(POST_OBSERVED_STRATEGY_AUTHORITY_ROOT_KEY)
    if post_observed_root is not None and local_research is not None:
        raise StrategyPackageError("portfolio_application.historical_source_selection_required")
    if post_observed_root is not None:
        try:
            post_observed = admit_post_observed_strategy_authority(post_observed_root)
        except PostObservedAuthorityError as error:
            raise StrategyPackageError(
                f"portfolio_application.installed_strategy_closure_refused:{error}"
            ) from error
        for recipe in FROZEN_RESEARCH_BOOK_RECIPES:
            bindings.append(
                InstalledPackageBinding(
                    package=_post_observed_package(authority=post_observed, recipe=recipe),
                    sources={
                        "HISTORICAL_ARRAY_REPLAY": PostObservedHistoricalScoreSource(
                            authority=post_observed, recipe=recipe
                        )
                    },
                    source_identity_hash=INSTALLED_STRATEGY_SOURCE_HASH,
                )
            )
    if local_research is not None:
        for recipe in FROZEN_RESEARCH_BOOK_RECIPES:
            source = LifecycleResearchScoreSource(manifest_path=local_research, recipe=recipe)
            if source.manifest.qa_outcome is None:
                raise StrategyPackageError("portfolio_application.full_local_qa_outcomes_required")
            source.verify()
            bindings.append(
                InstalledPackageBinding(
                    package=frozen_book_package(
                        recipe=recipe,
                        evidence_identity_hash=source.evidence_identity_hash,
                        source_description=(
                            "Local frozen-recipe reconstruction from verified lifecycle "
                            "executions; not the original research results"
                        ),
                        extra_claim_limits=(
                            "LOCAL_FROZEN_RECIPE_RECONSTRUCTION_NOT_ORIGINAL_RESEARCH_RESULTS",
                            "POST_OBSERVED_LOCAL_QA_OUTCOMES_NOT_HOLDOUT_RELEASE",
                        ),
                    ),
                    sources={"HISTORICAL_ARRAY_REPLAY": source},
                    source_identity_hash=INSTALLED_STRATEGY_SOURCE_HASH,
                )
            )
    return tuple(bindings)


__all__ = [
    "BOOK_PARTS",
    "BROAD_FEATURE_PACKAGE",
    "BROAD_FEATURE_STRATEGY_ID",
    "HETEROGENEOUS_CONTROLS_FROZEN",
    "HETEROGENEOUS_CURRENT_CLOSURE_ROOT_KEY",
    "HETEROGENEOUS_POLICY_ADAPTER_BINDING_HASH",
    "HETEROGENEOUS_POLICY_CATALOG_HASH",
    "HETEROGENEOUS_SUCCESSOR_PACKAGE_ROOT_KEY",
    "HETEROGENEOUS_SUCCESSOR_SCORE_KEY_PREFIX",
    "HOLDINGS_CONTROL_IDS",
    "INSTALLED_HETEROGENEOUS_BOOK_RECIPE",
    "INSTALLED_STRATEGY_SOURCE_HASH",
    "POST_OBSERVED_CONTROLS_FROZEN",
    "POST_OBSERVED_STRATEGY_AUTHORITY_ROOT_KEY",
    "PRODUCT_COMPONENT_ID",
    "PRODUCT_EVIDENCE_ROOT_KEY",
    "REBOUND_RETURN_BOOK_RECIPE",
    "REBOUND_RETURN_STRATEGY_ID",
    "SHARED_CONTROL_IDS",
    "STRATEGY_ARTIFACT_KEYS",
    "TREND_REBOUND_BOOK_RECIPE",
    "TREND_REBOUND_STRATEGY_ID",
    "ComponentBookRecipe",
    "HeterogeneousBookRecipe",
    "HeterogeneousCurrentClosure",
    "HeterogeneousCurrentScoreSource",
    "InstalledStrategyError",
    "ProductScoreSource",
    "SuccessorComponentScoreSource",
    "book_part_role",
    "build_replayed_formation_inputs",
    "component_formation_receipt",
    "heterogeneous_package",
    "install_frozen_strategies",
    "installed_book_part_hash",
    "product_evidence_closure_reader",
]
