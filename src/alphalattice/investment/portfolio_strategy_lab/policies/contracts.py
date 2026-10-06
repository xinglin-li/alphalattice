"""Typed boundary for Strategy Lab portfolio policies.

A policy owns how one formation session's target weights are produced. The
shared `portfolio_backtesting` owner keeps the clock, state, execution, costs,
segments, benchmark alignment, and path metrics; the Strategy Lab keeps
research, candidate freeze, and evidence. A policy adapter sits between them and
owns nothing else.

Routing uses an adapter-declared `policy_id` string rather than the frozen
`PortfolioPolicyFamily` enum, so an additional policy can be installed without
editing the enum or the frozen `PortfolioPolicySpec` union that published
artifacts still decode.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    BoolArray,
    FloatArray,
    PortfolioTargetDecision,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash, successor_identity_payload
from alphalattice.kernel.shared_kernel.source_identity import (
    source_component_id,
    switched_source_identity,
)

if TYPE_CHECKING:
    # Type-level only, and deliberately so. Both names below are used purely as
    # annotations, this module has ``from __future__ import annotations``, and
    # ``BoundPolicyDecisionInput`` is a plain dataclass rather than a Pydantic
    # model -- so nothing here is evaluated at runtime.
    #
    # What that buys is the first-release rule: importing the public policy
    # contract must not import ``optimizer/service.py``, CVXPY or OSQP. Before
    # this block, importing this module loaded all three, which put a solver in
    # the desktop process for two annotations that never run.
    #
    # This hides no architectural edge. The repository's dependency discoverer
    # reads the full AST and counts ``TYPE_CHECKING`` imports as real
    # dependencies by design, so the relationship stays declared and guarded;
    # only its execution goes away.
    from alphalattice.investment.portfolio_strategy_lab.optimizer.service import (
        CovarianceValidationProof,
        PortfolioOptimizer,
    )


def portfolio_adapter_implementation_hash(*sources: tuple[str, Path]) -> str:
    """Content identity of the modules that will actually decide weights.

    ``policy_id`` is a name, and a name is not an identity: the same string
    denotes whatever the adapter currently contains, so an adapter could be
    rewritten completely -- different objective, different constraints, different
    numbers -- while its id, its ``solver_backed`` flag and therefore the catalog
    binding all stood still.

    Neither the class name nor ``solver_backed`` is used, because both are
    declarations *about* an implementation rather than the implementation. The
    bytes are. Each adapter passes its own sources, so an adapter installed from
    outside this package states its content exactly as a built-in one does, and a
    solver-backed adapter includes the optimizer it delegates to -- that is where
    its numbers come from, and a change there moving no adapter binding would be
    the same defect one level down.

    Each source is given as ``(package_id, path)``. The package is stated per
    source rather than once for the call, because a closure genuinely spans
    packages: an adapter installed from outside this Desk that delegates to the
    Desk optimizer owns one of those files and not the other. Naming the package
    is what makes the component id survive a repository move, and inferring it
    from the path would be guessing at exactly the point the identity must be
    exact.
    """
    if not sources:
        raise ValueError("PORTFOLIO_POLICY_ADAPTER_SOURCES_EMPTY")
    return switched_source_identity(
        {
            source_component_id(package_id=package_id, source_path=path): path
            for package_id, path in sources
        },
        semantic_owner="portfolio_strategy_lab",
        numerical_role="portfolio-policy-adapter",
    )


PORTFOLIO_POLICY_PACKAGE = "portfolio_strategy_lab"
"""The package a built-in adapter's own sources belong to."""


class PortfolioSelectionAllocationSemantics(BaseModel):  # type: ignore[misc]
    """The separate score-admission and optimizer-allocation claims of one policy.

    ``top_k`` is a recipe parameter, but it is not evidence that an optimizer
    selected the global universe. This immutable statement names the two stages;
    the recipe and path still own the exact count, listing axis, and lane hashes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["PortfolioSelectionAllocationSemantics"] = "PortfolioSelectionAllocationSemantics"
    score_preselection_method: Literal[
        "STABLE_FINITE_SCORE_TOP_K",
        "STABLE_FINITE_SCORE_TOP_K_WITH_PREVIOUS_TARGET_EXIT_BUFFER",
        "STABLE_FINITE_SCORE_TOP_K_WITH_PREVIOUS_TARGET_EXIT_RANK",
    ]
    score_preselection_axis: Literal["DECISION_ELIGIBLE_FINITE_SCORE_LISTING_AXIS"] = (
        "DECISION_ELIGIBLE_FINITE_SCORE_LISTING_AXIS"
    )
    selection_count_source: Literal["RECIPE_TOP_K"] = "RECIPE_TOP_K"
    allocation_method: Literal[
        "CONVEX_SCORE_RISK_COST",
        "CLOSED_FORM_EQUAL_WEIGHT_SELECTED_BOOK",
        "CLOSED_FORM_INVERSE_VOLATILITY_SELECTED_BOOK",
        "CLOSED_FORM_DIAGONAL_RANK_MU_TILT_SELECTED_BOOK",
        "CLOSED_FORM_TRANCHE_SLEEVE_SCHEDULE_BOOK",
    ] = "CONVEX_SCORE_RISK_COST"
    allocation_axis: Literal[
        "PRESELECTED_NAMES_PLUS_CLOSE_MARKED_REFERENCE_AND_FROZEN_CARRY",
        "SELECTED_BOOK_PLUS_FROZEN_UNTRADABLE_CARRY",
        "SCHEDULED_SLEEVE_BOOKS_PLUS_FROZEN_UNTRADABLE_CARRY",
    ] = "PRESELECTED_NAMES_PLUS_CLOSE_MARKED_REFERENCE_AND_FROZEN_CARRY"
    optimizer_selection_claim: Literal[
        "OPTIMIZER_DOES_NOT_SELECT_GLOBAL_TOP_K",
        "NO_OPTIMIZER_SELECTS_OR_ALLOCATES_THE_BOOK",
    ] = "OPTIMIZER_DOES_NOT_SELECT_GLOBAL_TOP_K"
    semantics_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        score_preselection_method: Literal[
            "STABLE_FINITE_SCORE_TOP_K",
            "STABLE_FINITE_SCORE_TOP_K_WITH_PREVIOUS_TARGET_EXIT_BUFFER",
            "STABLE_FINITE_SCORE_TOP_K_WITH_PREVIOUS_TARGET_EXIT_RANK",
        ],
    ) -> Self:
        """Seal declared score preselection separately from convex allocation semantics.

        Args:
            score_preselection_method: Admitted stable top-k or prior-target hysteresis selection
                method.

        Returns:
            Exact binding for finite-score eligible selection and allocation with reference/frozen
            carry; global selection is not delegated to the optimizer.
        """
        values = {
            "kind": "PortfolioSelectionAllocationSemantics",
            "score_preselection_method": score_preselection_method,
            "score_preselection_axis": "DECISION_ELIGIBLE_FINITE_SCORE_LISTING_AXIS",
            "selection_count_source": "RECIPE_TOP_K",
            "allocation_method": "CONVEX_SCORE_RISK_COST",
            "allocation_axis": "PRESELECTED_NAMES_PLUS_CLOSE_MARKED_REFERENCE_AND_FROZEN_CARRY",
            "optimizer_selection_claim": "OPTIMIZER_DOES_NOT_SELECT_GLOBAL_TOP_K",
        }
        return cls(**values, semantics_hash=canonical_hash(values))

    @classmethod
    def closed_form_hysteresis_equal_weight(cls) -> Self:
        """The C1 direct-selection/direct-allocation statement."""
        values = {
            "kind": "PortfolioSelectionAllocationSemantics",
            "score_preselection_method": "STABLE_FINITE_SCORE_TOP_K_WITH_PREVIOUS_TARGET_EXIT_RANK",
            "score_preselection_axis": "DECISION_ELIGIBLE_FINITE_SCORE_LISTING_AXIS",
            "selection_count_source": "RECIPE_TOP_K",
            "allocation_method": "CLOSED_FORM_EQUAL_WEIGHT_SELECTED_BOOK",
            "allocation_axis": "SELECTED_BOOK_PLUS_FROZEN_UNTRADABLE_CARRY",
            "optimizer_selection_claim": "NO_OPTIMIZER_SELECTS_OR_ALLOCATES_THE_BOOK",
        }
        return cls(**values, semantics_hash=canonical_hash(values))

    @classmethod
    def closed_form_hysteresis_inverse_volatility(cls) -> Self:
        """The C2 direct-selection/inverse-volatility-allocation statement."""
        values = {
            "kind": "PortfolioSelectionAllocationSemantics",
            "score_preselection_method": "STABLE_FINITE_SCORE_TOP_K_WITH_PREVIOUS_TARGET_EXIT_RANK",
            "score_preselection_axis": "DECISION_ELIGIBLE_FINITE_SCORE_LISTING_AXIS",
            "selection_count_source": "RECIPE_TOP_K",
            "allocation_method": "CLOSED_FORM_INVERSE_VOLATILITY_SELECTED_BOOK",
            "allocation_axis": "SELECTED_BOOK_PLUS_FROZEN_UNTRADABLE_CARRY",
            "optimizer_selection_claim": "NO_OPTIMIZER_SELECTS_OR_ALLOCATES_THE_BOOK",
        }
        return cls(**values, semantics_hash=canonical_hash(values))

    @classmethod
    def closed_form_tranche_sleeve_schedule(cls) -> Self:
        """The tranche book's statement: same selection rule, per-sleeve and scheduled.

        The preselection method is the same hysteresis every closed-form policy
        here declares, because it is the same rule -- it is simply applied to one
        sleeve rather than to the whole book. What differs is the allocation:
        several sleeve books, only one of them reviewed at a formation, assembled
        under an aggregate name cap.
        """
        values = {
            "kind": "PortfolioSelectionAllocationSemantics",
            "score_preselection_method": "STABLE_FINITE_SCORE_TOP_K_WITH_PREVIOUS_TARGET_EXIT_RANK",
            "score_preselection_axis": "DECISION_ELIGIBLE_FINITE_SCORE_LISTING_AXIS",
            "selection_count_source": "RECIPE_TOP_K",
            "allocation_method": "CLOSED_FORM_TRANCHE_SLEEVE_SCHEDULE_BOOK",
            "allocation_axis": "SCHEDULED_SLEEVE_BOOKS_PLUS_FROZEN_UNTRADABLE_CARRY",
            "optimizer_selection_claim": "NO_OPTIMIZER_SELECTS_OR_ALLOCATES_THE_BOOK",
        }
        return cls(**values, semantics_hash=canonical_hash(values))

    @classmethod
    def closed_form_hysteresis_diagonal_rank_mu_tilt(cls) -> Self:
        """The C6 direct-selection/causal-rank-mu allocation statement."""
        values = {
            "kind": "PortfolioSelectionAllocationSemantics",
            "score_preselection_method": "STABLE_FINITE_SCORE_TOP_K_WITH_PREVIOUS_TARGET_EXIT_RANK",
            "score_preselection_axis": "DECISION_ELIGIBLE_FINITE_SCORE_LISTING_AXIS",
            "selection_count_source": "RECIPE_TOP_K",
            "allocation_method": "CLOSED_FORM_DIAGONAL_RANK_MU_TILT_SELECTED_BOOK",
            "allocation_axis": "SELECTED_BOOK_PLUS_FROZEN_UNTRADABLE_CARRY",
            "optimizer_selection_claim": "NO_OPTIMIZER_SELECTS_OR_ALLOCATES_THE_BOOK",
        }
        return cls(**values, semantics_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact preselection/allocation semantics identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: semantics_hash differs.
        """
        if self.semantics_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"semantics_hash"})
        ):
            raise ValueError("portfolio_strategy_lab.selection_allocation_semantics_invalid")
        return self

    def researcher_payload(self) -> dict[str, object]:
        """Project the complete selection/allocation declaration as JSON-compatible fields.

        Returns:
            Model JSON mapping including semantics identity.
        """
        return cast(dict[str, object], self.model_dump(mode="json"))


ADAPTER_BINDING_SUCCESSOR_FIELDS = ("selection_allocation_semantics",)
"""Legacy adapter receipts predate the explicit score/admission statement."""

ADAPTER_INPUT_CONSUMPTION_SUCCESSOR_FIELDS = ("input_consumption_semantics",)
"""Legacy adapter receipts predate explicit risk/optimizer consumption semantics."""


def adapter_binding_identity(payload: dict[str, Any]) -> dict[str, Any]:
    """Identity payload with the standalone selection/allocation successor rule."""
    return dict(
        successor_identity_payload(
            successor_identity_payload(dict(payload), ADAPTER_BINDING_SUCCESSOR_FIELDS),
            ADAPTER_INPUT_CONSUMPTION_SUCCESSOR_FIELDS,
        )
    )


class PortfolioPolicyAdapterBinding(BaseModel):  # type: ignore[misc]
    """Content identity of one adapter's deterministic decision behaviour.

    Development-only, and deliberately **not** folded into
    ``PortfolioPolicyCatalogBinding``: that contract may serve frozen artifact
    readback, and widening it would move identities published evidence already
    decodes. This is the narrow statement the development path needs, and it is
    gathered nowhere else.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["PortfolioPolicyAdapterBinding"] = "PortfolioPolicyAdapterBinding"
    policy_id: str = Field(min_length=1, max_length=128)
    adapter_implementation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipe_schema_id: str = Field(min_length=1, max_length=128)
    """Which recipe shape this adapter decodes, declared rather than inferred."""

    solver_semantics: str = Field(min_length=1, max_length=64)
    deterministic_policy: dict[str, Any]
    selection_allocation_semantics: PortfolioSelectionAllocationSemantics | None = None
    input_consumption_semantics: (
        Literal[
            "RISK_FORECAST_AND_OPTIMIZER_REQUIRED",
            "RISK_FORECAST_REQUIRED_NO_OPTIMIZER",
            "PER_NAME_RISK_SCALE_REQUIRED_NO_FORECAST_OR_OPTIMIZER",
            "NO_RISK_FORECAST_OR_OPTIMIZER",
        ]
        | None
    ) = None
    """What the adapter actually reads to form one target.

    ``None`` is historical readback and means the pre-successor conservative
    contract: a Risk forecast and optimizer are required. New bindings state
    this explicitly so a closed-form policy cannot be treated as a failed
    optimizer path merely because it has no covariance forecast.
    """
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        policy_id: str,
        adapter_implementation_hash: str,
        recipe_schema_id: str,
        solver_semantics: str,
        deterministic_policy: Mapping[str, Any],
        selection_allocation_semantics: PortfolioSelectionAllocationSemantics | None = None,
        input_consumption_semantics: Literal[
            "RISK_FORECAST_AND_OPTIMIZER_REQUIRED",
            "RISK_FORECAST_REQUIRED_NO_OPTIMIZER",
            "PER_NAME_RISK_SCALE_REQUIRED_NO_FORECAST_OR_OPTIMIZER",
            "NO_RISK_FORECAST_OR_OPTIMIZER",
        ] = "RISK_FORECAST_AND_OPTIMIZER_REQUIRED",
    ) -> Self:
        """Seal implementation, recipe schema, solver and input-consumption declarations.

        Args:
            policy_id: Installed deterministic policy identity.
            adapter_implementation_hash: Exact implementation closure identity.
            recipe_schema_id: Declared recipe contract identity.
            solver_semantics: Explicit allocation/solver semantics.
            deterministic_policy: Declared numerical policy parameters.
            selection_allocation_semantics: Optional explicit preselection/allocation binding.
            input_consumption_semantics: Declared Risk forecast/scale and optimizer dependencies.

        Returns:
            Validated canonical adapter binding; a missing optional declaration is not invented.
        """
        values = {
            "kind": "PortfolioPolicyAdapterBinding",
            "policy_id": policy_id,
            "adapter_implementation_hash": adapter_implementation_hash,
            "recipe_schema_id": recipe_schema_id,
            "solver_semantics": solver_semantics,
            "deterministic_policy": dict(deterministic_policy),
            "input_consumption_semantics": input_consumption_semantics,
        }
        if selection_allocation_semantics is not None:
            values["selection_allocation_semantics"] = selection_allocation_semantics.model_dump(
                mode="json"
            )
        return cls(
            policy_id=policy_id,
            adapter_implementation_hash=adapter_implementation_hash,
            recipe_schema_id=recipe_schema_id,
            solver_semantics=solver_semantics,
            deterministic_policy=dict(deterministic_policy),
            selection_allocation_semantics=selection_allocation_semantics,
            input_consumption_semantics=input_consumption_semantics,
            binding_hash=str(canonical_hash(values)),
        )

    @property
    def requires_risk_forecast(self) -> bool:
        """Read whether declared input consumption requires a Risk forecast.

        Returns:
            True for legacy unspecified, Risk-and-optimizer or Risk-without-optimizer semantics.
        """
        return self.input_consumption_semantics in (
            None,
            "RISK_FORECAST_AND_OPTIMIZER_REQUIRED",
            "RISK_FORECAST_REQUIRED_NO_OPTIMIZER",
        )

    @property
    def requires_optimizer(self) -> bool:
        """Read whether declared input consumption requires an optimizer.

        Returns:
            True for legacy unspecified or Risk-and-optimizer semantics.
        """
        return self.input_consumption_semantics in (
            None,
            "RISK_FORECAST_AND_OPTIMIZER_REQUIRED",
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require canonical admitted adapter-binding identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: binding_hash differs from adapter_binding_identity.
        """
        if self.binding_hash != canonical_hash(
            adapter_binding_identity(self.model_dump(mode="json", exclude={"binding_hash"}))
        ):
            raise ValueError("portfolio_strategy_lab.policy_adapter_binding_invalid")
        return self


class PortfolioPolicyRecipe(Protocol):
    """Structural view of one frozen policy specification."""

    top_k: int
    maximum_weight: float

    @property
    def policy_id(self) -> str:
        """Expose the installed recipe's deterministic policy identity.

        Returns:
            Explicit installed policy identity.
        """
        ...


@dataclass(frozen=True, slots=True)
class CausalRankReturnCurveSlice:
    """One formation's C6 curve input, supplied only by its causal owner."""

    formation_index: int
    formation_session: date
    bucket_means: FloatArray
    bucket_support_counts: tuple[int, ...]
    admitted_formation_count: int
    disposition: Literal[
        "AVAILABLE",
        "INSUFFICIENT_MATURED_FORMATION_HISTORY",
        "PARTIAL_BUCKET_SUPPORT",
    ]
    curve_hash: str


class PerNameRiskScale(Protocol):
    """The only Risk surface a closed-form policy is allowed to consume.

    Structural rather than a concrete import, and that is the point: this shared
    contract is used by every installed policy, so typing the field as
    ``RiskAllocationProjection`` would give every research policy a Risk import
    it never asked for. The public executor owns that edge; the contract states
    only the shape.

    Deliberately a per-name lane and not a matrix. A closed-form book weights on
    ``1 / sigma^p``, so an off-diagonal it cannot read would be an input nothing
    validates and nothing consumes -- and handing one over is how a policy ends
    up quietly depending on a covariance the public path never admitted.
    """

    @property
    def formation_session(self) -> date:
        """Read the Risk scale formation session.

        Returns:
            Exact declared formation session.
        """
        ...

    @property
    def ordered_listing_ids(self) -> tuple[str, ...]:
        """Read the Risk scale listing axis.

        Returns:
            Ordered listing identities.
        """
        ...

    @property
    def per_name_volatility(self) -> FloatArray:
        """Read per-name volatility on the declared axis.

        Returns:
            Owner-admitted volatility array.
        """
        ...

    @property
    def recipe_hash(self) -> str:
        """Read the deterministic Risk scale recipe identity.

        Returns:
            Exact recipe hash.
        """
        ...

    @property
    def projection_hash(self) -> str:
        """Read this formation's Risk scale projection identity.

        Returns:
            Exact projection hash.
        """
        ...

    def verify_content(self) -> None:
        """Recompute the lane identity before a policy consumes it."""
        ...


@dataclass(frozen=True, slots=True)
class BoundPolicyDecisionInput:
    """Host-bound decision surface for exactly one formation session."""

    scores: FloatArray
    covariance: FloatArray | None
    decision_eligible: BoolArray
    reference_weights: FloatArray
    sector_exposure_matrix: FloatArray
    equal_weight_sector_exposure: FloatArray
    covariance_validation: CovarianceValidationProof | None = None
    market_exposure: FloatArray | None = None
    """Raw per-listing SPY beta for this formation, or absent.

    Optional because the installed policies that predate the guardrail do not
    read it, and defaulting it to ones would hand them a fabricated exposure they
    never asked for. A policy that needs it refuses when it is absent rather than
    enforcing a guardrail against numbers nobody estimated.
    """

    previous_target_weights: FloatArray | None = None
    """The last target this policy *intended*, or ``None`` on its first formation.

    Distinct from ``reference_weights``, which is what actually got filled and
    then held. A no-trade band or a hysteresis rule compares against the
    intention; comparing against the executed book would re-open a position the
    policy already decided to keep, every time a fill came up short.
    """

    ordered_listing_ids: tuple[str, ...] | None = None
    """Decision-axis listing identities for policies whose tie break names them."""

    formation_index: int | None = None
    """Position on *this segment's* arrays: scores, curve slice, availability.

    Local addressing only. It is not the schedule position, and conflating the
    two is what let a continuation restage every sleeve: the protected segment
    starts at local index 0, and a schedule keyed on that index reads index 0 as
    "this book has never traded".
    """

    schedule_position: int | None = None
    """How many formations this *policy* has already decided, plus this one.

    Persistent across segments, so a book continued after a development prefix
    reviews the sleeve that is genuinely next rather than restaging all of them.
    `None` means "same as `formation_index`", which is exactly true for a path
    that starts from nothing and is what keeps every existing identity intact.
    """

    formation_session: date | None = None
    """Formation session, required by formation-bound Risk and mu inputs."""

    causal_rank_return_curve: CausalRankReturnCurveSlice | None = None
    """The owner-derived C6 rank-return curve slice, never a caller-made mu lane."""

    risk_allocation: PerNameRiskScale | None = None
    """The admitted per-name risk lane, for policies that weight on it.

    Separate from ``covariance`` rather than derived from it. The existing field
    carries a full matrix for the solver-backed policies that genuinely need one;
    a closed-form book needs a per-name scale with an identity, and taking the
    diagonal of a matrix would give it the numbers without the identity -- no
    projection hash to bind, no axis to check, and no way to tell an admitted
    lane from a matrix somebody happened to pass.
    """

    previous_sleeve_weights: tuple[FloatArray, ...] | None = None
    """Each sleeve's own book-scale weights, for a scheduled multi-sleeve policy.

    A tranche book cannot recover this from ``previous_target_weights``. Sleeves
    overlap -- the same name sits in more than one, which is why the book holds
    fewer distinct names than ``top_k * tranches`` -- so the assembled vector
    cannot be partitioned back into the sleeves that produced it.

    Vectors are book-scale rather than normalised, so each one's sum is that
    sleeve's current share of the book. That keeps the share an explicit input
    rather than a constant the policy assumes: whether a reviewed sleeve keeps
    its accumulated share or is reset to an equal one is a scientific choice,
    and a policy that hard-coded either would make that choice invisible.

    ``None`` on the first formation, where every sleeve is staged at once.
    """

    def require_covariance(self) -> FloatArray:
        """Return the owner-validated covariance only for a policy that declared it."""
        if self.covariance is None:
            raise ValueError("portfolio_strategy_lab.policy_covariance_not_declared")
        return self.covariance


class PortfolioPolicyAdapter(Protocol):
    """Desk-owned deterministic policy seam consumed by the walk-forward engine."""

    policy_id: str
    solver_backed: bool

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        """State this adapter's own content identity.

        Declared by the adapter rather than derived by the Host, for the same
        reason a Risk estimator declares its numerical binding: only the adapter
        knows which modules its numbers actually depend on, and an adapter
        installed from outside this package has no other way to say.
        """
        ...

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: PortfolioOptimizer,
    ) -> PortfolioTargetDecision:
        """Produce deterministic targets from an admitted recipe and exact input lanes.

        Args:
            policy: Concrete admitted recipe for this installed adapter.
            inputs: Bound scores, eligibility, reference and declared auxiliary lanes.
            optimizer: Deterministic numerical owner; closed-form adapters do not use it.

        Returns:
            Target weights and declared forecast/solver diagnostics; the adapter owns numerical
            allocation.
        """
        ...


__all__ = [
    "ADAPTER_BINDING_SUCCESSOR_FIELDS",
    "PORTFOLIO_POLICY_PACKAGE",
    "BoundPolicyDecisionInput",
    "CausalRankReturnCurveSlice",
    "PerNameRiskScale",
    "PortfolioPolicyAdapter",
    "PortfolioPolicyAdapterBinding",
    "PortfolioPolicyRecipe",
    "PortfolioSelectionAllocationSemantics",
    "adapter_binding_identity",
    "portfolio_adapter_implementation_hash",
]
