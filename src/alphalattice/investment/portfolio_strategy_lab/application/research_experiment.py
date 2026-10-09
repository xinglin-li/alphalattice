"""Authored EW research over verified Alpha evidence and the shared path engine.

This is a Desk adapter, not a Task runtime or an installed strategy. Segment
receipts encode the existing engine state; no holding/return arithmetic lives here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, Self

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from alphalattice.capabilities.portfolio_backtesting.clocks import ScoredFormationClock
from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioCostPolicy,
    PortfolioPerSideCostAssumption,
    PortfolioStateTransitionBinding,
    PortfolioWalkForwardSegmentResult,
    PortfolioWalkForwardState,
)
from alphalattice.capabilities.portfolio_backtesting.metrics import (
    PortfolioEconomicMetricSet,
    evaluate_raw_simple_return_path,
)
from alphalattice.capabilities.portfolio_backtesting.reference_marks import ReferenceMarkLane
from alphalattice.capabilities.portfolio_backtesting.segments import (
    run_portfolio_walk_forward_segment,
)
from alphalattice.investment.alpha_research.experiments.development_contracts import (
    canonical_score_value_identity,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    TrancheBookDecisionProvider,
    TrancheFormationInputs,
)
from alphalattice.investment.portfolio_strategy_lab.contracts import (
    PortfolioPolicySpec,
    PortfolioScoreMode,
)
from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
    build_public_portfolio_policy_catalog,
)
from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
    PortfolioPolicyRecipe,
)
from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import TrancheBookRecipe
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioResearchArtifactStore,
)
from alphalattice.investment.risk_research.surfaces.decomposition import RiskAllocationProjection
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskExecutionResult,
    DeskProgramCompilation,
    NumericalCallRecorder,
    ResearchExecutionEvidence,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)

if TYPE_CHECKING:
    # Types only: the campaign authority and the walk-forward provider load with the policy
    # path that reads them, not with every Portfolio read (LAWS.md ID8).
    from alphalattice.investment.portfolio_strategy_lab.campaign.authority import (
        CampaignValidatedCovarianceLane,
    )
    from alphalattice.investment.portfolio_strategy_lab.evaluation.walk_forward import (
        PortfolioPolicyDecisionProvider,
    )

KIND = "portfolio.policy-development"
RECEIPTS = "development/authored-replay"
SEGMENTS = "development/authored-segments"
LANES = "development/authored-lanes"
_HASH = r"^[0-9a-f]{64}$"
_ROOT = resolve_playpen_root(Path(__file__))
type FloatArray = npt.NDArray[np.float64]


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


TRANCHE_FIELDS = ("top_k", "tranches", "exit_rank", "weight_rule")
"""The tranche book's own fields, which a declared policy replaces."""


class PortfolioExperimentSpec(_Contract):
    """Declare an Alpha-linked development book, optional Risk route and return-gap policy.

    The tranche book declares names, sleeves, exit rank, sizing and per-side costs. An installed
    policy can replace the tranche route with explicit linked Risk authority. Data-quality
    quarantine is a declared retrospective universe choice, not point-in-time screening.
    """

    alpha_task_id: str = Field(
        description="The published Alpha development study whose scores the book trades."
    )
    candidate_id: str = Field(
        min_length=1, description="The Alpha study's candidate whose scores the book trades."
    )
    top_k: Annotated[int, Field(description="The tranche book's names per sleeve.")] = 35
    tranches: Annotated[
        int, Field(description="The tranche book's sleeves, each rebalanced in its turn.")
    ] = 3
    exit_rank: Annotated[
        int, Field(description="The score rank past which the tranche book sells a held name.")
    ] = 70
    weight_rule: Literal["ew", "iv1", "iv2"] = Field(
        default="ew",
        description="How the tranche book weighs its names: `ew` equally; `iv1` by inverse "
        "volatility and `iv2` by inverse variance, from the Risk study `risk_task_id` names.",
    )
    cost_bps_per_side: str = Field(
        default="5", description="The cost of each side of a trade, in basis points."
    )
    risk_task_id: str | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="A completed Risk study on the same research input that sizes the book: "
        "an `iv` rule's per-name volatility, or a `policy`'s covariance.",
    )
    unavailable_return_policy: Literal["require_complete", "quarantine_listings"] = Field(
        default="require_complete",
        exclude_if=lambda v: v == "require_complete",
        description="`require_complete` refuses a name whose return is unavailable; "
        "`quarantine_listings` excludes it for this window and reports the smaller universe.",
    )
    policy: PortfolioPolicySpec | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="A catalog policy the book runs in place of the tranche book, on the "
        "linked Risk study's covariance: equal weight, minimum variance, score/risk/cost or "
        "sector deviation; the tranche fields stay at their defaults.",
    )

    @field_validator("cost_bps_per_side")  # type: ignore[untyped-decorator]
    @classmethod
    def normalize_cost(cls, value: str) -> str:
        """Normalize an admitted per-side cost through the registered cost assumption.

        Args:
            value: Declared per-side cost in basis points.

        Returns:
            Canonical string form of the admitted per-side cost.

        Raises:
            AuthoringError: The cost assumption rejects arithmetic or value admission.
        """
        try:
            return str(PortfolioPerSideCostAssumption.from_bps_per_side(value).cost_bps_per_side)
        except (ArithmeticError, ValueError) as error:
            raise AuthoringError("portfolio_research.cost_invalid") from error

    @property
    def recipe(self) -> TrancheBookRecipe:
        """Seal the declared top-k, tranche, exit-rank and sizing controls.

        Returns:
            Validated TrancheBookRecipe for this specification.
        """
        return TrancheBookRecipe.create(
            top_k=self.top_k,
            tranches=self.tranches,
            exit_rank=self.exit_rank,
            weight_rule=self.weight_rule,
        )

    @property
    def book(self) -> PortfolioPolicyRecipe:
        """The policy the book runs: the declared catalog policy, else the tranche book."""
        return self.recipe if self.policy is None else self.policy

    @property
    def cost(self) -> PortfolioPerSideCostAssumption:
        """Resolve the declared per-side cost into its registered platform assumption.

        Returns:
            Validated per-side cost assumption and platform cost conversion.
        """
        return PortfolioPerSideCostAssumption.from_bps_per_side(self.cost_bps_per_side)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_policy(self) -> Self:
        """Require declared tranche controls to form an admitted installed recipe.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The declared tranche recipe cannot be admitted.
        """
        _ = self.recipe
        return self

    @model_serializer(mode="wrap")  # type: ignore[untyped-decorator]
    def _policy_replaces_the_book(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # A declared policy runs in place of the tranche book, whose fields then hold
        # their defaults and decide nothing: the document leaves them out.
        values: dict[str, Any] = handler(self)
        if self.policy is not None:
            for name in TRANCHE_FIELDS:
                values.pop(name, None)
        return values


class PortfolioExperimentSource(_Contract):
    """Bind published Alpha scores and optional Risk authority to exact research input axes.

    Program/receipt/candidate/target, Panel/outcome/universe and score-value identities select the
    source. Formation and scored-session axes remain distinct so unscored gaps cannot invent
    predictions.
    """

    alpha_task_id: str
    alpha_program_hash: str = Field(pattern=_HASH)
    alpha_receipt_hash: str = Field(pattern=_HASH)
    candidate_id: str
    target_recipe_id: str
    target_binding_hash: str = Field(pattern=_HASH)
    foundation_admission_hash: str | None = None
    input_binding_hash: str = Field(pattern=_HASH)
    panel_snapshot_hash: str = Field(pattern=_HASH)
    outcome_snapshot_hash: str = Field(pattern=_HASH)
    universe_revision: str = Field(pattern=_HASH)
    ordered_listing_ids: tuple[str, ...]
    formation_sessions: tuple[date, ...]
    score_sessions: tuple[date, ...]
    # Numerical result and validation surface, in the receipt's fold order.
    score_refs: tuple[tuple[str, str], ...]
    score_value_hash: str = Field(pattern=_HASH)
    # The linked Risk study, bound as the Alpha source is; absent for equal weight, and
    # then no hash sealed before it moves.
    risk_task_id: str | None = Field(default=None, exclude_if=lambda v: v is None)
    risk_program_hash: str | None = Field(
        default=None, pattern=_HASH, exclude_if=lambda v: v is None
    )
    risk_surface_hash: str | None = Field(
        default=None, pattern=_HASH, exclude_if=lambda v: v is None
    )
    source_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal one declared portfolio experiment source.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical source_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = cls.model_construct(**values, source_hash="")
        return cls(
            **values,
            source_hash=canonical_hash(draft.model_dump(mode="json", exclude={"source_hash"})),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_source(self) -> Self:
        """Require canonical source axes and scored coverage of both formation endpoints.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AuthoringError: Source axes are empty/repeated/unordered, scored sessions leave
                formation support or omit its endpoints, or source_hash differs.
        """
        if (
            not self.score_sessions
            or not self.formation_sessions
            or not self.ordered_listing_ids
            or self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or self.score_sessions != tuple(sorted(set(self.score_sessions)))
            or not set(self.score_sessions) <= set(self.formation_sessions)
            or self.score_sessions[0] != self.formation_sessions[0]
            or self.score_sessions[-1] != self.formation_sessions[-1]
            or self.ordered_listing_ids != tuple(sorted(set(self.ordered_listing_ids)))
            or self.source_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"source_hash"}))
        ):
            raise AuthoringError("portfolio_research.source_identity_invalid")
        return self

    @property
    def clock(self) -> ScoredFormationClock:
        """Project scored sessions onto their exact formation-axis indices.

        Returns:
            ScoredFormationClock selecting only declared scored formations.
        """
        scored = set(self.score_sessions)
        return ScoredFormationClock(
            tuple(i for i, s in enumerate(self.formation_sessions) if s in scored)
        )


def implementation_hash() -> str:
    """Hash the Portfolio execution implementation and numerical library environment.

    The code, and the numerical library version, that turn a Portfolio Program into
    numbers. A study plan binds it and records its moves; the Program does not (binding
    plan, P). The Alpha scores it reads are bound by their content (the source's
    ``score_value_hash``), not by the Alpha owner's reader (B12).
    """
    paths = (
        "investment/portfolio_strategy_lab/application/research_experiment.py",
        "investment/portfolio_strategy_lab/application/tranche_book_execution.py",
        "investment/portfolio_strategy_lab/policies/tranche_book.py",
        "investment/portfolio_strategy_lab/policies/buffered_equal_weight.py",
        "investment/portfolio_strategy_lab/publication/artifacts.py",
        "investment/portfolio_strategy_lab/campaign/authority.py",
        # A catalog policy's decisions.
        "investment/portfolio_strategy_lab/contracts.py",
        "investment/portfolio_strategy_lab/evaluation/walk_forward.py",
        "investment/portfolio_strategy_lab/policies/catalog.py",
        "investment/portfolio_strategy_lab/policies/contracts.py",
        "investment/portfolio_strategy_lab/policies/top_k_equal_weight.py",
        "investment/portfolio_strategy_lab/policies/minimum_variance.py",
        "investment/portfolio_strategy_lab/policies/score_risk_cost.py",
        "investment/portfolio_strategy_lab/optimizer/service.py",
        "control/product_host/research_authoring/portfolio_handoff.py",
        "capabilities/portfolio_inputs/tradability/surface.py",
        "capabilities/portfolio_inputs/tradability/readback.py",
        "foundation/market_data_ops/publication/session_marks.py",
        *tuple(
            f"capabilities/portfolio_backtesting/{v}.py"
            for v in (
                "contracts",
                "clocks",
                "segments",
                "execution",
                "state",
                "metrics",
                "reference_marks",
            )
        ),
    )
    # The numerics this code runs on are the environment, provenance and never this
    # identity (LAWS.md ID6).
    return source_rule_closure_hash(
        root=_ROOT,
        tracked_paths=tuple("src/alphalattice/" + p for p in paths),
        semantic_owner="portfolio_strategy_lab",
        numerical_role="AUTHORED_EW_REPLAY",
    )


class PortfolioExperimentCompiler:
    """Own deterministic source/specification admission and desk program composition."""

    kind = KIND

    def __init__(self, source: PortfolioExperimentSource):
        """Bind portfolio compilation to an exact published research source.

        Args:
            source: Sealed Alpha-linked portfolio experiment source.
        """
        self.source = source

    def method_identity(self, spec: PortfolioExperimentSpec) -> dict[str, Any]:
        # The tranche book resolves in the public catalog; a declared policy in the installed
        # one, which holds the solver-backed policies.
        """Describe selected policy semantics, source and scored clock without adapter code bytes.

        Args:
            spec: Declared tranche book or installed policy specification.

        Returns:
            Method payload retaining spec/source/policy, semantic adapter binding and clock;
            implementation/hash fields are excluded from the adapter projection.
        """
        if spec.policy is None:
            catalog = build_public_portfolio_policy_catalog()
            policy = spec.recipe.recipe_hash
        else:
            from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
                build_installed_portfolio_policy_catalog,
            )

            catalog = build_installed_portfolio_policy_catalog()
            policy = canonical_hash(spec.policy.model_dump(mode="json"))
        return {
            "spec": spec.model_dump(mode="json"),
            "source": self.source.source_hash,
            "policy": policy,
            # What the adapter declares, not its code: the code is the plan's implementation,
            # whose moves are recorded (P, LAWS.md ID1), and its bytes here moved every
            # Portfolio Program for a docstring.
            "adapter": catalog.resolve(spec.book)
            .describe_adapter_binding()
            .model_dump(mode="json", exclude={"adapter_implementation_hash", "binding_hash"}),
            "clock": self.source.clock.binding.model_dump(mode="json"),
        }

    def compile_desk_program(
        self,
        *,
        envelope: ResearchExperimentEnvelope,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
    ) -> DeskProgramCompilation:
        """Admit exact source/authority, work budget and feasible universe controls.

        Args:
            envelope: Sealed research experiment envelope and declared budget.
            document: Exact experiment/portfolio document sections.
            authority: Resolved Panel, universe, listing and formation authority.

        Returns:
            Desk compilation binding authority/method, selected adapter semantics and recipe-schema
            route.

        Raises:
            AuthoringError: Document/source/authority differs, budget is insufficient, exit/top-k
                exceeds the scored universe or policy cap cannot support unit mass.
        """
        if envelope.kind != KIND or set(document) != {"experiment", "portfolio"}:
            raise AuthoringError("portfolio_research.document_sections_invalid")
        spec = PortfolioExperimentSpec.model_validate(document["portfolio"])
        source = self.source
        if (
            spec.alpha_task_id != source.alpha_task_id
            or spec.candidate_id != source.candidate_id
            or authority.panel_snapshot_hash != source.panel_snapshot_hash
            or authority.universe_revision_sha256 != source.universe_revision
            or authority.ordered_listing_ids != source.ordered_listing_ids
            or authority.sessions != source.formation_sessions
        ):
            raise AuthoringError("portfolio_research.source_authority_mismatch")
        if envelope.budget.maximum_numerical_calls < len(source.formation_sessions) + 1:
            raise AuthoringError("portfolio_research.work_budget_exceeded")
        if spec.exit_rank > len(source.ordered_listing_ids):
            raise AuthoringError("portfolio_research.exit_rank_exceeds_universe")
        if spec.policy is not None:
            if spec.policy.top_k > len(source.ordered_listing_ids):
                # The bound is the listings the Alpha study scored.
                raise AuthoringError(
                    "portfolio_research.top_k_exceeds_universe",
                    expected={
                        "portfolio.policy.top_k": {"maximum": len(source.ordered_listing_ids)}
                    },
                )
            if spec.policy.top_k * spec.policy.maximum_weight < 1.0:
                raise AuthoringError("portfolio_research.policy_cap_infeasible")
        identity = self.method_identity(spec)
        method = canonical_hash(identity)
        return DeskProgramCompilation(
            desk_program_hash=canonical_hash(
                {
                    "kind": KIND,
                    # The envelope is bound once, by the sealed Program itself (B6).
                    "authority": authority.authority_hash,
                    "method": method,
                }
            ),
            # The policy this Program runs, not the public catalog around it, and the
            # recipe schema it names, not the spec's JSON schema, which a contract
            # docstring moves: neither the menu nor the schema decides a number.
            catalog_hash=canonical_hash({"adapter": identity["adapter"]}),
            method_binding_hash=method,
            parameter_domain_hash=canonical_hash(
                {
                    "policy_id": identity["adapter"]["policy_id"],
                    "recipe_schema_id": identity["adapter"]["recipe_schema_id"],
                }
            ),
        )


@dataclass(frozen=True)
class _ResearchMetrics:
    bootstrap_resamples: int = 0


class PortfolioDataExclusion(_Contract):
    """Record retrospective listing return unavailability on declared affected sessions."""

    listing_id: str = Field(min_length=1)
    affected_sessions: tuple[date, ...] = Field(min_length=1)
    reason: Literal["EXECUTION_RETURN_UNAVAILABLE"] = "EXECUTION_RETURN_UNAVAILABLE"


class ResearchUniversePolicy(Protocol):
    """Expose the book population size and declared unavailable-return treatment."""

    @property
    def top_k(self) -> int:
        """Read the declared number of selected portfolio names.

        Returns:
            Admitted top-k population size.
        """
        ...

    @property
    def unavailable_return_policy(self) -> Literal["require_complete", "quarantine_listings"]:
        """Read explicit complete-return or retrospective data-quarantine treatment.

        Returns:
            require_complete or quarantine_listings as declared by the deterministic owner.
        """
        ...


def resolve_research_universe(
    *,
    spec: ResearchUniversePolicy,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
    decision: npt.NDArray[np.bool_],
    realized: FloatArray,
) -> tuple[npt.NDArray[np.bool_], tuple[PortfolioDataExclusion, ...]]:
    """An observed-data quarantine, not a new valuation or a prospective universe."""
    if decision.shape != realized.shape or realized.shape != (len(sessions), len(listings)):
        raise AuthoringError("portfolio_research.input_axis_invalid")
    exclusions: tuple[PortfolioDataExclusion, ...] = ()
    if spec.unavailable_return_policy == "quarantine_listings":
        # A name admitted at any formation may remain held at a later one. Check
        # its whole required return path, not only dates on which it can enter.
        missing = ~np.isfinite(realized) & decision.any(axis=0)[None, :]
        indices = np.flatnonzero(missing.any(axis=0))
        exclusions = tuple(
            PortfolioDataExclusion(
                listing_id=listings[j],
                affected_sessions=tuple(sessions[i] for i in np.flatnonzero(missing[:, j])),
            )
            for j in indices
        )
        if exclusions:
            decision = decision.copy()
            decision[:, indices] = False
            decision.setflags(write=False)
            if int(np.count_nonzero(decision.any(axis=0))) < spec.top_k:
                raise AuthoringError("portfolio_research.universe_too_small_after_quarantine")
    empty = ~decision.any(axis=1)
    unavailable = decision & ~np.isfinite(realized)
    if empty.any() or unavailable.any():
        raise AuthoringError(_support_absence(sessions, empty, unavailable))
    return decision, exclusions


def _support_absence(
    sessions: tuple[date, ...],
    empty: npt.NDArray[np.bool_],
    unavailable: npt.NDArray[np.bool_],
) -> str:
    """The support refusal, its subject naming the cause and where it holds.

    A formation session with no eligible name comes first: no unavailable-return policy repairs
    it. Else the eligible names without their realized return, which `require_complete` refuses
    and the declared quarantine excludes. The subject counts names and sessions and spans the
    first to the last affected session, within the code's 120 characters.

    Args:
        sessions: The formation sessions, in order.
        empty: The sessions with no eligible name.
        unavailable: The eligible (session, name) cells without a finite realized return.

    Returns:
        The code with its subject.
    """
    code = "portfolio_research.benchmark_support_absent"
    rows = np.flatnonzero(empty if empty.any() else unavailable.any(axis=1))
    span = f"{rows.size} sessions,{sessions[rows[0]]}..{sessions[rows[-1]]}"
    if empty.any():
        return f"{code}:no_eligible_name:{span}"
    return (
        f"{code}:returns_unavailable:{int(np.count_nonzero(unavailable.any(axis=0)))} names,{span}"
    )


@dataclass(frozen=True)
class PortfolioExperimentInputs:
    """Retain aligned score, execution, return, capacity and sector inputs for replay.

    Formation/listing axes govern the array lanes. Passive-return marks and reference carry remain
    explicit; optional exclusion, label, transition and tradability bindings describe source
    admission rather than authorizing a new numerical method.
    """

    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    scores: FloatArray
    decision_eligible: npt.NDArray[np.bool_]
    execution_available: npt.NDArray[np.bool_]
    realized_simple_returns: FloatArray
    causal_adv20: FloatArray
    sector_exposure_matrix: FloatArray
    equal_weight_sector_exposure: FloatArray
    passive_returns_by_session: dict[date, FloatArray]
    reference_mark: ReferenceMarkLane
    market_binding: dict[str, str]
    mandate: _ResearchMetrics = field(default_factory=_ResearchMetrics)
    listing_labels: tuple[str, ...] | None = None
    data_exclusions: tuple[PortfolioDataExclusion, ...] = ()
    market_decision_eligible: npt.NDArray[np.bool_] | None = None
    transition_binding: PortfolioStateTransitionBinding | None = None
    tradability_decision_hash: str | None = None


class PortfolioReplaySegment(_Contract):
    """Seal one bounded replay segment with exact numerical lane and carry lineage.

    Start/stop, asset/sleeve counts and previous segment identity select the segment. Stored
    executed/target/pretrade/reference/sleeve lanes accompany return/turnover/mode,
    concentration/support/capacity/sector diagnostics and cash carry.
    """

    program_hash: str = Field(pattern=_HASH)
    source_hash: str = Field(pattern=_HASH)
    start_index: int = Field(ge=0)
    stop_index: int = Field(gt=0)
    asset_count: int = Field(gt=0)
    sleeve_count: int = Field(gt=0)
    previous_segment_hash: str | None = None
    gross_simple_returns: tuple[float, ...]
    one_way_turnovers: tuple[float, ...]
    decision_modes: tuple[Literal["REBALANCE", "HOLD"], ...]
    hhi: tuple[float, ...]
    holding_counts: tuple[float, ...]
    weighted_adv20: tuple[float, ...]
    maximum_absolute_sector_deviations: tuple[float, ...]
    missed_execution_count: int
    pretrade_cash: float
    optimizer_reference_cash: float
    cash_path: tuple[float, ...]
    lanes: dict[str, str]
    segment_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_segment(self) -> Self:
        """Require a positive replay interval, complete equal-width diagnostics and exact lanes.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AuthoringError: Interval width, retained diagnostic/cash lengths, the five numerical
                lane names or segment_hash differs.
        """
        width = self.stop_index - self.start_index
        if (
            width <= 0
            or any(
                len(getattr(self, name)) != width
                for name in (
                    "gross_simple_returns",
                    "one_way_turnovers",
                    "decision_modes",
                    "hhi",
                    "holding_counts",
                    "weighted_adv20",
                    "maximum_absolute_sector_deviations",
                    "cash_path",
                )
            )
            or set(self.lanes) != {"executed", "targets", "pretrade", "reference", "sleeves"}
            or self.segment_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"segment_hash"}))
        ):
            raise AuthoringError("portfolio_research.segment_identity_invalid")
        return self


class PortfolioReplayReceipt(_Contract):
    """Bind replay segments and economics to exact source, method and market authority.

    Per-formation net/benchmark/cost lanes, declared metric set, optional labels and retrospective
    exclusions retain their own source axes and identities.
    """

    source: PortfolioExperimentSource
    spec: PortfolioExperimentSpec
    program_hash: str = Field(pattern=_HASH)
    method_binding_hash: str = Field(pattern=_HASH)
    authority_hash: str = Field(pattern=_HASH)
    market_binding: dict[str, str]
    segments: tuple[str, ...]
    result: dict[str, float | None]
    net_simple_returns: tuple[float, ...]
    benchmark_simple_returns: tuple[float, ...]
    cost_fractions: tuple[float, ...]
    method_identity: dict[str, Any]
    listing_labels: tuple[str, ...] | None = Field(default=None, exclude_if=lambda v: v is None)
    data_exclusions: tuple[PortfolioDataExclusion, ...] = Field(
        default=(), exclude_if=lambda v: not v
    )
    receipt_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require valid exclusion/label axes and exact source/method/economic receipt lineage.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AuthoringError: Excluded listings/sessions repeat or leave source support, quarantine
                policy/binding is absent, label/return/cost axes differ, metric fields or method
                spec/source/hash disagree, or receipt_hash differs.
        """
        excluded = tuple(v.listing_id for v in self.data_exclusions)
        if (
            len(set(excluded)) != len(excluded)
            or not set(excluded) <= set(self.source.ordered_listing_ids)
            or any(
                v.affected_sessions != tuple(sorted(set(v.affected_sessions)))
                or not set(v.affected_sessions) <= set(self.source.formation_sessions)
                for v in self.data_exclusions
            )
            or (
                bool(excluded)
                and (
                    self.spec.unavailable_return_policy != "quarantine_listings"
                    or self.market_binding.get("data_quarantine")
                    != canonical_hash([v.model_dump(mode="json") for v in self.data_exclusions])
                )
            )
            or (
                self.listing_labels is not None
                and len(self.listing_labels) != len(self.source.ordered_listing_ids)
            )
            or set(self.result) != set(PortfolioEconomicMetricSet.model_fields)
            or self.method_identity.get("spec") != self.spec.model_dump(mode="json")
            or self.method_identity.get("source") != self.source.source_hash
            or canonical_hash(self.method_identity) != self.method_binding_hash
            or any(
                len(v) != len(self.source.formation_sessions)
                for v in (
                    self.net_simple_returns,
                    self.benchmark_simple_returns,
                    self.cost_fractions,
                )
            )
            or self.receipt_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"receipt_hash"}))
        ):
            raise AuthoringError("portfolio_research.receipt_identity_invalid")
        return self

    @property
    def data_quality(self) -> dict[str, Any]:
        """Project source/effective population counts and declared retrospective exclusions.

        Returns:
            Data-quality read model with policy, counts, retained exclusion labels and a notice when
            exclusions exist.
        """
        labels = dict(
            zip(
                self.source.ordered_listing_ids,
                self.listing_labels or self.source.ordered_listing_ids,
                strict=True,
            )
        )
        total, count = len(self.source.ordered_listing_ids), len(self.data_exclusions)
        return {
            "policy": self.spec.unavailable_return_policy,
            "source_listing_count": total,
            "effective_listing_count": total - count,
            "excluded": [
                {**v.model_dump(mode="json"), "label": labels[v.listing_id]}
                for v in self.data_exclusions
            ],
            "notice": (
                f"Retrospective data-quality quarantine: {count} of {total} names excluded; "
                f"effective Portfolio universe {total - count}. "
                + ", ".join(labels[v.listing_id] for v in self.data_exclusions)
                + ". Original Alpha/input history retained; this is not point-in-time screening."
            )
            if count
            else None,
        }


def _lane(store: PortfolioResearchArtifactStore, digest: str, shape: tuple[int, ...]) -> FloatArray:
    data = store.load_packed_bytes(category=LANES, content_hash=digest)
    if len(data) != int(np.prod(shape)) * 8:
        raise AuthoringError("portfolio_research.lane_shape_invalid")
    return np.frombuffer(data, dtype="<f8").reshape(shape)


def _reopen_segment(
    store: PortfolioResearchArtifactStore, digest: str
) -> tuple[PortfolioReplaySegment, PortfolioWalkForwardSegmentResult, FloatArray, FloatArray]:
    meta = store.load(
        category=SEGMENTS,
        content_hash=digest,
        model=PortfolioReplaySegment,
        identity_field="segment_hash",
    )
    count, assets = meta.stop_index - meta.start_index, meta.asset_count
    values = {
        name: _lane(store, meta.lanes[name], shape)
        for name, shape in (
            ("executed", (count, assets)),
            ("targets", (count, assets)),
            ("pretrade", (assets,)),
            ("reference", (assets,)),
            ("sleeves", (meta.sleeve_count, assets)),
        )
    }
    state = PortfolioWalkForwardState(
        pretrade_weights=values["pretrade"],
        pretrade_cash=meta.pretrade_cash,
        optimizer_reference=values["reference"],
        optimizer_reference_cash=meta.optimizer_reference_cash,
    )
    result = PortfolioWalkForwardSegmentResult(
        **{
            k: getattr(meta, k)
            for k in (
                "start_index",
                "stop_index",
                "gross_simple_returns",
                "one_way_turnovers",
                "decision_modes",
                "hhi",
                "holding_counts",
                "weighted_adv20",
                "maximum_absolute_sector_deviations",
                "missed_execution_count",
            )
        },
        final_state=state,
        executed_weights=tuple(map(tuple, values["executed"])),
        predicted_variances=(None,) * count,
        risk_forecast_required=(False,) * count,
    )
    return meta, result, values["sleeves"], values["targets"]


class PortfolioExperimentCancelled(ValueError):
    """Signal a requested development replay cancellation at a durable segment boundary."""

    pass


class PortfolioExperimentExecutor:
    """Own admitted segmented development replay and retained economic publication."""

    kind = KIND

    def __init__(
        self,
        source: PortfolioExperimentSource,
        inputs: Callable[[Path], PortfolioExperimentInputs],
        cancellation: Callable[[], bool] = lambda: False,
        risk_allocation: Callable[[], tuple[RiskAllocationProjection, ...]] | None = None,
        risk_covariance: Callable[[tuple[date, ...]], CampaignValidatedCovarianceLane]
        | None = None,
    ):
        """Bind replay input loading, boundary cancellation and optional linked Risk readers.

        Args:
            source: Exact published Alpha source authority.
            inputs: Caller-owned output-workspace input loader.
            cancellation: Explicit cancellation predicate checked at segment boundaries.
            risk_allocation: Optional linked Risk volatility-lane reader.
            risk_covariance: Optional linked Risk covariance reader at requested formations.
        """
        self.source, self.inputs, self.cancellation = source, inputs, cancellation
        self.compiler = PortfolioExperimentCompiler(source)
        self.risk_allocation = risk_allocation
        """Each formation's volatility lane from the linked Risk study, loaded by the Host."""
        self.risk_covariance = risk_covariance
        """The linked Risk study's covariance at the named formations, loaded by the Host."""

    def execute(
        self,
        *,
        program: SealedResearchProgram,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
        output_workspace: Path,
        recorder: NumericalCallRecorder | None = None,
    ) -> DeskExecutionResult:
        """Verify exact program/input authority before reusing or computing replay segments.

        Unscored gaps cannot carry finite invented scores or active eligibility. Existing segments
        must match ordered source/parent/shape lineage; new segments retain numerical lanes and
        carry, with cancellation checked at boundaries. Reported work counts new segment formations
        plus one economic evaluation.

        Args:
            program: Exact sealed research program.
            document: Authored experiment/portfolio sections.
            authority: Resolved research input authority.
            output_workspace: Caller-owned replay artifact destination.
            recorder: Optional numerical capability recorder notified on completed publication.

        Returns:
            COMPUTED desk result selecting the retained replay receipt, formation axis and reported
            work.

        Raises:
            AuthoringError: Program/input/source/gap/quarantine authority or retained segment
                lineage differs.
            PortfolioExperimentCancelled: Cancellation is requested at a segment boundary.
        """
        compiled = self.compiler.compile_desk_program(
            envelope=ResearchExperimentEnvelope.create(**document["experiment"]),
            document=document,
            authority=authority,
        )
        if compiled.desk_program_hash != program.desk_program_hash:
            raise AuthoringError("portfolio_research.program_changed")
        spec = PortfolioExperimentSpec.model_validate(document["portfolio"])
        inputs = self.inputs(output_workspace)
        source = self.source
        shape = (len(source.formation_sessions), len(source.ordered_listing_ids))
        if (
            inputs.formation_sessions != source.formation_sessions
            or inputs.ordered_listing_ids != source.ordered_listing_ids
            or any(
                a.shape != shape
                for a in (
                    inputs.scores,
                    inputs.decision_eligible,
                    inputs.execution_available,
                    inputs.realized_simple_returns,
                    inputs.causal_adv20,
                )
            )
        ):
            raise AuthoringError("portfolio_research.input_axis_invalid")
        scored_indices = source.clock.formation_indices
        if (
            np.isinf(inputs.scores).any()
            or canonical_score_value_identity(inputs.scores[list(scored_indices)])
            != source.score_value_hash
            or inputs.market_binding.get("source") != source.input_binding_hash
        ):
            raise AuthoringError("portfolio_research.input_values_changed")
        gaps = tuple(i for i in range(shape[0]) if i not in scored_indices)
        if gaps and (
            np.isfinite(inputs.scores[list(gaps)]).any()
            or inputs.decision_eligible[list(gaps)].any()
        ):
            raise AuthoringError("portfolio_research.gap_contains_invented_scores")
        # The handoff held each formation its candidate under-scored. One whose tradable, scored
        # names are still fewer than a rebalance selects refuses here, before any segment is
        # written, by its session, never in the middle of the walk.
        selected = spec.top_k if spec.policy is None else spec.policy.top_k
        short = next(
            (
                source.formation_sessions[i]
                for i in scored_indices
                if np.count_nonzero(inputs.decision_eligible[i] & np.isfinite(inputs.scores[i]))
                < selected
            ),
            None,
        )
        if short is not None:
            raise AuthoringError(f"portfolio_research.eligible_pool_short:{short}")
        if inputs.data_exclusions and (
            spec.unavailable_return_policy != "quarantine_listings"
            or any(
                v.listing_id not in source.ordered_listing_ids
                or inputs.decision_eligible[:, source.ordered_listing_ids.index(v.listing_id)].any()
                for v in inputs.data_exclusions
            )
        ):
            raise AuthoringError("portfolio_research.quarantine_not_applied")
        inputs.reference_mark.require_carry_segment(source.formation_sessions)
        risk_lanes = self._risk_lanes(spec)
        store = PortfolioResearchArtifactStore(output_workspace)
        prior: dict[int, PortfolioReplaySegment] = {}
        for path in (store.root / SEGMENTS).glob("*.json"):
            meta = PortfolioReplaySegment.model_validate_json(path.read_bytes())
            if meta.segment_hash != path.stem:
                raise AuthoringError("portfolio_research.segment_misfiled")
            if meta.program_hash == program.program_hash:
                if meta.start_index in prior:
                    raise AuthoringError("portfolio_research.segment_ambiguous")
                prior[meta.start_index] = meta
        state = None
        sleeves = None
        # A catalog policy's book is one sleeve: its last intended target, which the next
        # segment's decisions start from.
        sleeve_count = spec.tranches if spec.policy is None else 1
        segments = []
        hashes: list[str] = []
        calls = 0
        for start in range(0, shape[0], 21):
            stop = min(start + 21, shape[0])
            if self.cancellation():
                raise PortfolioExperimentCancelled(
                    "portfolio_research.cancelled_at_segment_boundary"
                )
            if start in prior:
                meta, segment, sleeves, _targets = _reopen_segment(store, prior[start].segment_hash)
                if (
                    meta.source_hash != source.source_hash
                    or meta.stop_index != stop
                    or meta.previous_segment_hash != (hashes[-1] if hashes else None)
                    or meta.asset_count != shape[1]
                    or meta.sleeve_count != sleeve_count
                ):
                    raise AuthoringError("portfolio_research.segment_lineage_invalid")
            else:
                provider: TrancheBookDecisionProvider | PortfolioPolicyDecisionProvider
                if spec.policy is None:
                    formations = tuple(
                        TrancheFormationInputs(
                            formation_session=session,
                            scores=inputs.scores[i],
                            decision_eligible=inputs.decision_eligible[i],
                            risk_allocation=risk_lanes[i],
                            causal_rank_return_curve=None,
                        )
                        for i, session in enumerate(source.formation_sessions)
                    )
                    provider = TrancheBookDecisionProvider(
                        recipe=spec.recipe,
                        formations=formations,
                        ordered_listing_ids=source.ordered_listing_ids,
                        sector_exposure_matrix=inputs.sector_exposure_matrix,
                        equal_weight_sector_exposure=inputs.equal_weight_sector_exposure,
                        initial_sleeve_weights=sleeves,
                        # A development covariance has no factor block to attribute.
                        attribution_required=False,
                    )
                else:
                    provider = self._policy_provider(
                        spec.policy,
                        inputs,
                        start=start,
                        stop=stop,
                        previous=None if sleeves is None else sleeves[0],
                    )
                cash_path: list[float] = []

                def capture_cash(
                    i: int, w: FloatArray, cash: float, *, values: list[float] = cash_path
                ) -> None:
                    values.append(cash)

                segment = run_portfolio_walk_forward_segment(
                    workspace=inputs,
                    decision_provider=provider,
                    start_index=start,
                    stop_index=stop,
                    initial_state=state,
                    rebalance_clock=source.clock,
                    reference_mark=inputs.reference_mark,
                    entry_observer=capture_cash,
                )
                sleeves, targets = _book_state(provider)
                lanes = {
                    name: store.publish_array(
                        category=LANES, values=np.ascontiguousarray(value, dtype="<f8")
                    )
                    for name, value in {
                        "executed": segment.executed_weights,
                        "targets": targets,
                        "sleeves": sleeves,
                        "pretrade": segment.final_state.pretrade_weights,
                        "reference": segment.final_state.optimizer_reference,
                    }.items()
                }
                values = dict(
                    program_hash=program.program_hash,
                    source_hash=source.source_hash,
                    asset_count=shape[1],
                    sleeve_count=sleeve_count,
                    previous_segment_hash=hashes[-1] if hashes else None,
                    pretrade_cash=segment.final_state.pretrade_cash,
                    optimizer_reference_cash=segment.final_state.optimizer_reference_cash,
                    lanes=lanes,
                    cash_path=tuple(cash_path),
                    **{
                        k: getattr(segment, k)
                        for k in (
                            "start_index",
                            "stop_index",
                            "gross_simple_returns",
                            "one_way_turnovers",
                            "decision_modes",
                            "hhi",
                            "holding_counts",
                            "weighted_adv20",
                            "maximum_absolute_sector_deviations",
                            "missed_execution_count",
                        )
                    },
                )
                draft = PortfolioReplaySegment.model_construct(**values, segment_hash="")
                meta = PortfolioReplaySegment(
                    **values,
                    segment_hash=canonical_hash(
                        draft.model_dump(mode="json", exclude={"segment_hash"})
                    ),
                )
                store.publish(category=SEGMENTS, value=meta, identity_field="segment_hash")
                calls += stop - start
            state = segment.final_state
            hashes.append(meta.segment_hash)
            segments.append(segment)
        cost = int(spec.cost.platform_one_way_cost_bps)
        gross: FloatArray = np.asarray(
            [v for segment in segments for v in segment.gross_simple_returns], dtype=np.float64
        )
        turnover: FloatArray = np.asarray(
            [v for segment in segments for v in segment.one_way_turnovers], dtype=np.float64
        )
        benchmark: FloatArray = np.asarray(
            [
                float(np.mean(inputs.passive_returns_by_session[s]))
                for s in source.formation_sessions
            ],
            dtype=np.float64,
        )
        cost_policy = PortfolioCostPolicy(reporting_bps=(cost,), selection_bps=cost)
        measured = evaluate_raw_simple_return_path(
            gross_simple_returns=gross,
            one_way_turnovers=turnover,
            benchmark_simple_returns=benchmark,
            cost_bps=cost,
            cost_policy=cost_policy,
        )
        result = {
            k: float(v) if v is not None and np.isfinite(v) else None
            for k, v in measured.model_dump().items()
        }
        method_identity = self.compiler.method_identity(spec)
        values = dict(
            source=source,
            spec=spec,
            program_hash=program.program_hash,
            method_binding_hash=program.method_binding_hash,
            authority_hash=authority.authority_hash,
            market_binding=inputs.market_binding,
            segments=tuple(hashes),
            result=result,
            net_simple_returns=tuple(
                cost_policy.net_simple_returns(
                    gross_simple_returns=gross, one_way_turnovers=turnover, cost_bps=cost
                )
            ),
            benchmark_simple_returns=tuple(benchmark),
            cost_fractions=tuple(
                cost_policy.cost_fraction(one_way_turnover=turnover, cost_bps=cost)
            ),
            method_identity=method_identity,
            listing_labels=inputs.listing_labels,
            data_exclusions=inputs.data_exclusions,
        )
        draft = PortfolioReplayReceipt.model_construct(**values, receipt_hash="")
        receipt = PortfolioReplayReceipt(
            **values,
            receipt_hash=canonical_hash(draft.model_dump(mode="json", exclude={"receipt_hash"})),
        )
        store.publish(category=RECEIPTS, value=receipt, identity_field="receipt_hash")
        if recorder is not None:
            recorder.record(capability=program.desk_program_hash)
        return DeskExecutionResult(
            disposition="COMPUTED",
            artifact_uris=(f"playpen://portfolio-strategy-lab/{RECEIPTS}/{receipt.receipt_hash}",),
            formation_sessions=source.formation_sessions,
            numerical_call_count=calls + 1,
            desk_input_binding_hash=source.source_hash,
        )

    def _policy_provider(
        self,
        policy: PortfolioPolicySpec,
        inputs: PortfolioExperimentInputs,
        *,
        start: int,
        stop: int,
        previous: FloatArray | None,
    ) -> PortfolioPolicyDecisionProvider:
        """A catalog policy's decisions over formations ``start`` to ``stop``, on the linked Risk
        study's covariance projected for those formations alone."""
        from alphalattice.investment.portfolio_strategy_lab.evaluation.walk_forward import (
            PortfolioPolicyDecisionProvider,
        )
        from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
            build_installed_portfolio_policy_catalog,
        )

        source = self.source
        if self.risk_covariance is None or source.risk_surface_hash is None:
            raise AuthoringError("portfolio_research.risk_study_required")
        if start and previous is None:
            raise AuthoringError("portfolio_research.policy_state_absent")
        lane = self.risk_covariance(source.formation_sessions[start:stop])
        width = len(source.ordered_listing_ids)
        if lane.values.shape != (stop - start, width, width):
            raise AuthoringError("portfolio_research.risk_axis_mismatch")
        return PortfolioPolicyDecisionProvider(
            workspace=_PolicyWorkspace(
                mandate=inputs.mandate,
                formation_sessions=inputs.formation_sessions,
                ordered_listing_ids=inputs.ordered_listing_ids,
                scores={(source.candidate_id, PortfolioScoreMode.STOCK_ONLY): inputs.scores},
                covariances=lane.values,
                covariance_validation=lane,
                decision_eligible=inputs.decision_eligible,
                execution_available=inputs.execution_available,
                realized_simple_returns=inputs.realized_simple_returns,
                causal_adv20=inputs.causal_adv20,
                sector_exposure_matrix=inputs.sector_exposure_matrix,
                equal_weight_sector_exposure=inputs.equal_weight_sector_exposure,
                passive_returns_by_session=inputs.passive_returns_by_session,
            ),
            candidate_id=source.candidate_id,
            score_mode=PortfolioScoreMode.STOCK_ONLY,
            policy=policy,
            policies=build_installed_portfolio_policy_catalog(),
            record_targets=True,
            covariance_offset=start,
            previous_target_weights=previous,
        )

    def _risk_lanes(
        self, spec: PortfolioExperimentSpec
    ) -> tuple[RiskAllocationProjection | None, ...]:
        """Each formation's volatility lane for an ``iv`` rule, none for equal weight."""

        source = self.source
        if spec.weight_rule == "ew":
            return (None,) * len(source.formation_sessions)
        if self.risk_allocation is None or source.risk_surface_hash is None:
            raise AuthoringError("portfolio_research.risk_study_required")
        lanes = self.risk_allocation()
        if tuple(v.formation_session for v in lanes) != source.formation_sessions or any(
            tuple(v.ordered_listing_ids) != source.ordered_listing_ids
            or v.surface_hash != source.risk_surface_hash
            for v in lanes
        ):
            raise AuthoringError("portfolio_research.risk_axis_mismatch")
        return lanes


@dataclass(frozen=True)
class _PolicyWorkspace:
    """The experiment's inputs as a catalog policy reads them: its scores by candidate and one
    segment's covariances from the linked Risk study."""

    mandate: _ResearchMetrics
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    scores: dict[tuple[str, PortfolioScoreMode], FloatArray]
    covariances: FloatArray
    covariance_validation: CampaignValidatedCovarianceLane
    decision_eligible: npt.NDArray[np.bool_]
    execution_available: npt.NDArray[np.bool_]
    realized_simple_returns: FloatArray
    causal_adv20: FloatArray
    sector_exposure_matrix: FloatArray
    equal_weight_sector_exposure: FloatArray
    passive_returns_by_session: dict[date, FloatArray]


def _book_state(
    provider: TrancheBookDecisionProvider | PortfolioPolicyDecisionProvider,
) -> tuple[FloatArray, FloatArray]:
    """A segment's carried sleeves and its targets by formation, whichever book decided them."""
    if isinstance(provider, TrancheBookDecisionProvider):
        sleeves = provider.sleeve_state
        if sleeves is None:
            raise AuthoringError("portfolio_research.policy_state_absent")
        return sleeves, np.asarray(provider.target_weights_by_formation, dtype=np.float64)
    carried = provider.previous_target_weights
    if carried is None:
        raise AuthoringError("portfolio_research.policy_state_absent")
    return (
        np.asarray(carried, dtype=np.float64)[None, :],
        np.vstack(provider.recorded_targets).astype(np.float64),
    )


def read_portfolio_experiment(
    output: Path, handle: str
) -> tuple[PortfolioReplayReceipt, list[dict[str, Any]]]:
    """Reopen exact replay receipt and ordered segment rows without recomputing decisions.

    Args:
        output: Caller-owned replay artifact root.
        handle: Exact retained receipt URI.

    Returns:
        Validated receipt and formation-ordered position/economic rows.

    Raises:
        AuthoringError: Handle category, segment parent/source/program/position lineage or complete
            formation coverage differs.
    """
    prefix = f"playpen://portfolio-strategy-lab/{RECEIPTS}/"
    if not handle.startswith(prefix):
        raise AuthoringError("portfolio_research.receipt_handle_invalid")
    store = PortfolioResearchArtifactStore(output)
    receipt = store.load(
        category=RECEIPTS,
        content_hash=handle[len(prefix) :],
        model=PortfolioReplayReceipt,
        identity_field="receipt_hash",
    )
    rows: list[dict[str, Any]] = []
    previous = None
    net = receipt.net_simple_returns
    for digest in receipt.segments:
        meta, segment, _sleeves, targets = _reopen_segment(store, digest)
        if (
            meta.start_index != len(rows)
            or meta.previous_segment_hash != previous
            or meta.source_hash != receipt.source.source_hash
            or meta.program_hash != receipt.program_hash
        ):
            raise AuthoringError("portfolio_research.segment_lineage_invalid")
        for local, weights in enumerate(segment.executed_weights):
            index = meta.start_index + local
            rows.append(
                dict(
                    session=str(receipt.source.formation_sessions[index]),
                    decision_mode=segment.decision_modes[local],
                    targets=list(map(float, targets[local])),
                    weights=list(weights),
                    cash=meta.cash_path[local],
                    gross_simple_return=segment.gross_simple_returns[local],
                    net_simple_return=net[index],
                    benchmark_simple_return=receipt.benchmark_simple_returns[index],
                    cost_fraction=receipt.cost_fractions[index],
                    one_way_turnover=segment.one_way_turnovers[local],
                    hhi=segment.hhi[local],
                    holding_count=segment.holding_counts[local],
                )
            )
        previous = digest
    if len(rows) != len(receipt.source.formation_sessions):
        raise AuthoringError("portfolio_research.segment_axis_incomplete")
    return receipt, rows


class PortfolioExperimentVerifier:
    """Own exact retained replay verification and explicit historical method standing."""

    kind = KIND

    def __init__(self) -> None:
        """Initialize empty method-standing, refusal and request-local verified readback state."""
        self.method_standing: Literal["INSTALLED", "NOT_CURRENT"] | None = None
        self.method_refusal: str | None = None
        self.verified_readback: tuple[PortfolioReplayReceipt, list[dict[str, Any]]] | None = None
        """The receipt and rows the most recent ``verify`` reopened and proved, for
        the caller of that verification to project inside the same request. A
        value the read produced, not a substitute for it: the next ``verify``
        replaces it, and it says nothing about the artifacts after the read."""

    def verify(
        self,
        *,
        program: SealedResearchProgram,
        evidence: ResearchExecutionEvidence,
        authority: ResolvedResearchAuthority | None,
        output_workspace: Path,
    ) -> None:
        """Reopen one receipt and match its program, evidence and optional source authority.

        Each call clears earlier verification state. Compatible historical implementation-bearing
        schemes remain readable and receive NOT_CURRENT standing; the verified receipt/rows belong
        only to this read request.

        Args:
            program: Exact sealed program whose receipt is checked.
            evidence: One declared replay receipt and its source/formation binding.
            authority: Optional current resolved research authority.
            output_workspace: Caller-owned retained replay artifact root.

        Raises:
            AuthoringError: Receipt count, program/method/authority, evidence kind/source axis or
                current authority differs.
        """
        self.verified_readback = None
        self.method_standing = None
        self.method_refusal = None
        if len(evidence.artifact_uris) != 1:
            raise AuthoringError("portfolio_research.receipt_required")
        receipt, rows = read_portfolio_experiment(output_workspace, evidence.artifact_uris[0])
        if (
            receipt.program_hash != program.program_hash
            or receipt.method_binding_hash != program.method_binding_hash
            or receipt.authority_hash != program.authority_hash
            or evidence.kind != KIND
            or evidence.desk_input_binding_hash != receipt.source.source_hash
            or evidence.formation_sessions != receipt.source.formation_sessions
            or (
                authority is not None
                and (
                    authority.authority_hash != receipt.authority_hash
                    or authority.ordered_listing_ids != receipt.source.ordered_listing_ids
                    or authority.sessions != receipt.source.formation_sessions
                )
            )
        ):
            raise AuthoringError("portfolio_research.receipt_program_mismatch")
        # A Program sealed before P also bound the implementation, and one sealed before
        # its adapter's code; either verifies and reads back as recorded, and is
        # historical.
        earlier = "implementation" in receipt.method_identity or (
            "adapter_implementation_hash" in receipt.method_identity.get("adapter", {})
        )
        self.method_standing = "NOT_CURRENT" if earlier else "INSTALLED"
        self.method_refusal = "portfolio_research.program_scheme_superseded" if earlier else None
        self.verified_readback = (receipt, rows)
