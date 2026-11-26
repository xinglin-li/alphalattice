"""One installed frozen strategy declaration, and the catalog that owns them.

A package is a *declaration*: identities, a component plan, the score sources it
installs, the controls it freezes and the lanes it needs. It carries no store, no
workspace path, no credential, no callback and no import target, because the
whole point of the seam is that adding a strategy is a declaration plus a
registration rather than another resolver, adapter or Host branch.

The runtime half is one class and two protocols. `InstalledPackageBinding` is
the installed owner of one package -- it holds the declaration to what is
actually installed, admits a request against the package's own control surface,
names the policy identity that request compiles to, and hands back the score
source for a selected mode. `StrategyScoreSource` resolves that package's own
score lane over shared inputs somebody else resolved, and `SharedInputOwner` is
whoever resolved them. None of them is stored inside the declaration, and the
catalog hashes only declarations, so a runtime object can never be mistaken for
an identity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from functools import cache
from pathlib import Path
from typing import ClassVar, Literal, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    BoolArray,
    FloatArray,
    PortfolioBacktestWorkspace,
)
from alphalattice.capabilities.portfolio_backtesting.reference_marks import ReferenceMarkLane
from alphalattice.capabilities.portfolio_inputs.signed_score.contracts import (
    PortfolioExecutionEvents,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioResearchSpec,
    PortfolioSupportCoverage,
    ScoreSourceMode,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    ComponentBookFactory,
)
from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
    build_public_portfolio_policy_catalog,
)
from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import (
    ADMITTED_WEIGHT_RULES,
    TrancheBookRecipe,
)
from alphalattice.investment.risk_research.surfaces.decomposition import (
    RiskAllocationProjection,
    RiskAttributionProjection,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import recorded_moves
from alphalattice.kernel.shared_kernel.recipe_identity import recipe_identity, recipe_seal_holds

type MergeSemantics = Literal[
    "SINGLE_COMPONENT_BOOK",
    "POST_TRADE_LISTING_WEIGHTS_THEN_RECOMPUTE_ECONOMICS",
]

type RiskDisposition = Literal[
    "REPORT_ONLY_DFD_PRIME_PLUS_E_NEVER_USED_FOR_WEIGHTS",
    "POLICY_CONSUMED_RISK_ALLOCATION_PLUS_REPORT_ONLY_DFD_PRIME_PLUS_E",
]

type PolicyBinding = Literal["PACKAGE_FROZEN", "REQUEST_SELECTED"]
"""Whether the package fixes its policy identity or derives it from the request.

The successor freezes `Top35 / Exit70 / Tranche3 / EW`; the predecessor lets a
researcher move them inside the shared catalog's admitted bands. Both are the
same field on the same contract, so the compiler asks one question.
"""

ALLOCATION_BASIS_POINTS_TOTAL: int = 10_000


_PACKAGE_SEAL = frozenset({"package_hash"})

VALUE_ROLE_PREFIXES: tuple[str, ...] = (
    "alpha_research.product_recipe",
    "alpha_research.strategy_recipe",
    "alpha_research.component_recipe.",
    "portfolio_strategy_lab.book_recipe.",
    "portfolio_strategy_lab.policy_binding.",
    "portfolio_strategy_lab.policy_catalog.",
    "portfolio_strategy_lab.book_alpha_recipe.",
)
"""The readout roles a package's parts move under: its Alpha recipe and components, its book
recipe and its policy binding and catalog (NM1). Their recorded moves are what a value a workspace
stored before a move is read through."""


@cache
def _value_moves() -> dict[str, str]:
    """Each value a value role moved through, by the value its moves started from."""
    previous: dict[str, str] = {}
    for move in recorded_moves():
        if move.role.startswith(VALUE_ROLE_PREFIXES):
            previous[move.successor] = move.predecessor
    roots: dict[str, str] = {}
    for value in {*previous, *previous.values()}:
        root, seen = value, {value}
        while root in previous and previous[root] not in seen:
            root = previous[root]
            seen.add(root)
        roots[value] = root
    return roots


def recipe_hashes_match(first: str, second: str) -> bool:
    """Whether two recorded recipe hashes name one recipe.

    Equal, or one moved to the other through a value role's recorded moves: a hash written before
    NM1 against one written after.
    """
    if first == second:
        return True
    roots = _value_moves()
    return first in roots and roots[first] == roots.get(second)


class StrategyPackageError(ValueError):
    """Stable refusal for a package declaration, catalog or selection failure."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentPlanEntry(_Contract):
    """One component book and the share of the merged book it carries."""

    LABELS: ClassVar[frozenset[str]] = frozenset(
        {"component_id", "family_id", "target_recipe", "objective"}
    )
    """The component's names, which the package hash leaves out: its recipe hash binds them."""

    kind: Literal["ComponentPlanEntry"] = "ComponentPlanEntry"
    component_id: str = Field(min_length=1)
    allocation_basis_points: int = Field(gt=0, le=ALLOCATION_BASIS_POINTS_TOTAL)
    family_id: str = Field(min_length=1)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_recipe: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    model_lifecycle_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda value: value is None
    )


class ScoreSourceCapability(_Contract):
    """One installed way to obtain this package's scores, named by its authority."""

    LABELS: ClassVar[frozenset[str]] = frozenset({"description"})

    kind: Literal["ScoreSourceCapability"] = "ScoreSourceCapability"
    mode: ScoreSourceMode
    evidence_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The exact evidence or model-set identity this mode reopens.

    It is the Program's `alpha_evidence_manifest_hash` for a run in this mode, so
    two modes of one strategy produce two Program identities and never share a
    ledger.
    """

    description: str = Field(min_length=1)


class PackageFrozenControl(_Contract):
    """A control the package fixes: a fact to display, never a widget to offer."""

    LABELS: ClassVar[frozenset[str]] = frozenset({"frozen_display", "refusal_code", "reason"})
    """What a reader is shown and told; the frozen value itself is the package's policy."""

    kind: Literal["PackageFrozenControl"] = "PackageFrozenControl"
    control_id: str = Field(min_length=1)
    frozen_display: str = Field(min_length=1)
    refusal_code: str = Field(min_length=1)
    """Raised when a request moves this control anyway.

    Declared beside the frozen value rather than raised by a caller, so the page
    that renders the fact and the compiler that refuses the request quote the
    same code.
    """

    reason: str = Field(min_length=1)


class PackageControlSurface(_Contract):
    """What the selected package lets a researcher move, and what it does not.

    Four disjoint sets. A browser that renders this cannot offer a combination
    the compiler must then reject, which was the concrete defect one installed
    strategy hit: its page advertised another package's weighting default and
    every holdings control beside it, all of which the compiler then refused.
    """

    kind: Literal["PackageControlSurface"] = "PackageControlSurface"
    frozen: tuple[PackageFrozenControl, ...] = ()
    admitted: tuple[str, ...] = ()
    shared: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_disjoint(self) -> Self:
        """Require unique disjoint frozen, admitted and shared control groups.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            StrategyPackageError: A control repeats within a group or appears in more than one
                group.
        """
        groups = (
            tuple(value.control_id for value in self.frozen),
            self.admitted,
            self.shared,
        )
        seen: set[str] = set()
        for group in groups:
            if len(set(group)) != len(group) or seen & set(group):
                raise StrategyPackageError("portfolio_application.package_control_surface_overlaps")
            seen |= set(group)
        return self

    def frozen_control(self, control_id: str) -> PackageFrozenControl | None:
        """Find one explicitly frozen package control.

        Args:
            control_id: Exact frozen control identity.

        Returns:
            Matching frozen control, otherwise None.
        """
        for value in self.frozen:
            if value.control_id == control_id:
                return value
        return None


class PackagePolicyIdentity(_Contract):
    """The three policy identities one request compiles to under one package."""

    kind: Literal["PackagePolicyIdentity"] = "PackagePolicyIdentity"
    policy_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_adapter_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class FrozenStrategyPackage(_Contract):
    """One immutable installed strategy, declared once and hashed as a whole."""

    LABELS: ClassVar[frozenset[str]] = frozenset({"strategy_id", "claim_limits", "provenance"})
    """The package's name and words, which its hash leaves out: its identity is the recipe it
    builds a book from, at every depth (ID10)."""

    kind: Literal["FrozenStrategyPackage"] = "FrozenStrategyPackage"
    strategy_id: str = Field(min_length=1)
    alpha_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The Alpha formation/score authority this package forms books from.

    The *Alpha* recipe, and only that. Whole-strategy identity is
    `package_hash`, which is why nothing downstream has to read this field to
    work out which product it is looking at.
    """

    policy_binding: PolicyBinding
    frozen_policy: PackagePolicyIdentity | None = None
    """Present exactly when `policy_binding` is `PACKAGE_FROZEN`."""

    component_plan: tuple[ComponentPlanEntry, ...] = Field(min_length=1)
    merge_semantics: MergeSemantics
    score_sources: tuple[ScoreSourceCapability, ...] = Field(min_length=1)
    default_score_source_mode: ScoreSourceMode
    required_shared_input_lanes: tuple[str, ...] = Field(min_length=1)
    """The shared lanes this package reads, checked against what it was handed.

    A declaration nothing verifies is decoration, so `SharedPortfolioInputs`
    names the lanes it carries and a package asking for one that is absent is
    refused before it scores anything.
    """

    risk_disposition: RiskDisposition
    controls: PackageControlSurface
    claim_limits: tuple[str, ...] = Field(min_length=1)
    provenance: tuple[str, ...] = Field(min_length=1)
    package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared installed strategy package.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical package_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = cls.model_construct(**values, package_hash="0" * 64)
        identity = draft.model_dump(mode="json", exclude={"package_hash"})
        return cls(**identity, package_hash=recipe_identity(draft, exclude=_PACKAGE_SEAL))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require complete component allocation, coherent merge/mode/policy declarations.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            StrategyPackageError: Component IDs or score modes repeat, allocation total is
                incomplete, single/merged semantics disagree, default mode is absent, frozen
                policy/binding contradicts, or package_hash differs.
        """
        ids = tuple(value.component_id for value in self.component_plan)
        if len(set(ids)) != len(ids):
            raise StrategyPackageError("portfolio_application.package_component_duplicated")
        total = sum(value.allocation_basis_points for value in self.component_plan)
        if total != ALLOCATION_BASIS_POINTS_TOTAL:
            raise StrategyPackageError("portfolio_application.package_allocation_incomplete")
        single = len(self.component_plan) == 1
        if single != (self.merge_semantics == "SINGLE_COMPONENT_BOOK"):
            # A one-component plan that claims a merge would describe a merge the
            # executor never performs, and a multi-component plan that claims a
            # single book would hide the one it does.
            raise StrategyPackageError("portfolio_application.package_merge_semantics_invalid")
        modes = tuple(value.mode for value in self.score_sources)
        if len(set(modes)) != len(modes):
            raise StrategyPackageError("portfolio_application.package_score_source_duplicated")
        if self.default_score_source_mode not in modes:
            raise StrategyPackageError("portfolio_application.package_default_mode_not_installed")
        frozen = self.frozen_policy is not None
        if frozen != (self.policy_binding == "PACKAGE_FROZEN"):
            raise StrategyPackageError("portfolio_application.package_policy_binding_invalid")
        if not recipe_seal_holds(self, "package_hash"):
            raise StrategyPackageError("portfolio_application.package_identity_invalid")
        return self

    @property
    def component_ids(self) -> tuple[str, ...]:
        """Read component identities in declared package-plan order.

        Returns:
            Ordered component IDs without sorting.
        """
        return tuple(value.component_id for value in self.component_plan)

    @property
    def installed_modes(self) -> tuple[ScoreSourceMode, ...]:
        """Read score-source modes in declared package order.

        Returns:
            Ordered installed score-source modes.
        """
        return tuple(value.mode for value in self.score_sources)

    def admit_control_values(self, values: Mapping[str, str]) -> None:
        """Refuse a request that moved a control this package froze.

        The caller renders its request as control displays and the package
        compares them to its own declaration, which keeps the frozen fact and
        the refusal in one place instead of restating the values at every entry
        point that could violate them.
        """
        for control in self.controls.frozen:
            selected = values.get(control.control_id)
            if selected is not None and selected != control.frozen_display:
                raise StrategyPackageError(control.refusal_code)

    def score_source(self, mode: ScoreSourceMode) -> ScoreSourceCapability:
        """Resolve one score-source capability declared by this package.

        Args:
            mode: Explicit installed score-source mode.

        Returns:
            Matching declared capability.

        Raises:
            StrategyPackageError: This package does not declare the requested mode.
        """
        for value in self.score_sources:
            if value.mode == mode:
                return value
        raise StrategyPackageError(
            f"portfolio_application.package_score_source_not_installed:{mode}"
        )


class StrategyDisclosure(_Contract):
    """What the report and the readback say about the strategy that produced it.

    One projection for every package. The renderer reads these fields; it does
    not know which strategy it is rendering, which is the property that keeps a
    third package from needing a renderer change.
    """

    kind: Literal["StrategyDisclosure"] = "StrategyDisclosure"
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy_id: str = Field(min_length=1)
    package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    alpha_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_source_mode: ScoreSourceMode
    score_source_authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_source_description: str = Field(min_length=1)
    component_plan: tuple[ComponentPlanEntry, ...] = Field(min_length=1)
    merge_semantics: MergeSemantics
    risk_disposition: RiskDisposition
    claim_limits: tuple[str, ...] = Field(min_length=1)
    provenance: tuple[str, ...] = Field(min_length=1)
    disclosure_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def compile(
        cls,
        *,
        package: FrozenStrategyPackage,
        mode: ScoreSourceMode,
        report_hash: str,
        program_hash: str,
    ) -> Self:
        """Seal report/program disclosure from the exact package and selected source mode.

        Args:
            package: Installed frozen package declaration.
            mode: Declared source mode selecting evidence identity/description.
            report_hash: Exact report being disclosed.
            program_hash: Exact execution program being disclosed.

        Returns:
            Validated disclosure retaining package/component/merge, Risk disposition, provenance and
            claim limits.

        Raises:
            StrategyPackageError: The source mode is not declared or disclosure consistency fails.
        """
        capability = package.score_source(mode)
        identity: dict[str, object] = {
            "kind": "StrategyDisclosure",
            "report_hash": report_hash,
            "program_hash": program_hash,
            "strategy_id": package.strategy_id,
            "package_hash": package.package_hash,
            "alpha_recipe_hash": package.alpha_recipe_hash,
            "score_source_mode": mode,
            "score_source_authority_hash": capability.evidence_identity_hash,
            "score_source_description": capability.description,
            "component_plan": [value.model_dump(mode="json") for value in package.component_plan],
            "merge_semantics": package.merge_semantics,
            "risk_disposition": package.risk_disposition,
            "claim_limits": list(package.claim_limits),
            "provenance": list(package.provenance),
        }
        return cls(**identity, disclosure_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact declared strategy disclosure identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            StrategyPackageError: The canonical disclosure payload differs from its hash.
        """
        if self.disclosure_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"disclosure_hash"})
        ):
            raise StrategyPackageError("portfolio_application.strategy_disclosure_identity_invalid")
        return self


@dataclass(frozen=True, slots=True)
class SharedPortfolioInputs:
    """Every lane the product resolves for any package, and nothing a package owns.

    Market, Tradability, Outcome, classification, benchmark and the report-only
    Risk decomposition. No score, no candidate semantics, no policy binding: a
    package reads this and supplies the rest, which is what stops the shared
    resolver from replaying an unselected strategy's models in order to throw the
    result away.
    """

    workspace: PortfolioBacktestWorkspace
    formation_sessions: tuple[date, ...]
    """The executable axis: what the walk will actually decide over."""

    ordered_listing_ids: tuple[str, ...]
    decision_eligible: BoolArray
    """`(formations, listings)` eligibility from the Tradability decision lane."""

    common_sessions: tuple[date, ...]
    """Every session the shared lanes agree on, before any policy maturity cut.

    A package whose policy averages over matured prior formations needs history
    the walk itself never decides over, so both axes travel together and
    `active_rows` says how they line up.
    """

    active_rows: npt.NDArray[np.int64]
    common_decision_eligible: BoolArray
    common_realized_simple_returns: FloatArray
    execution_events: PortfolioExecutionEvents | None
    """Execution clock when the selected score source consumes one.

    A sealed historical package can carry a close/open convention and exact
    realized lanes without claiming that its post-observed rows are part of the
    current development outcome schedule.  Such a package leaves this absent;
    a score source that needs the schedule must refuse the absence itself.
    """
    risk_allocation_by_session: Mapping[date, RiskAllocationProjection]
    risk_attribution_by_session: Mapping[date, RiskAttributionProjection]
    report_risk_attributions: tuple[RiskAttributionProjection, ...]
    """One per executable formation, in axis order, for the report-only lane."""

    risk_recipe_hash: str
    risk_return_surface_hash: str
    sector_map_hash: str
    tradability_decision_hash: str
    execution_outcome_manifest_hash: str
    secondary_benchmark_id: str | None
    secondary_benchmark_returns: tuple[float, ...] | None
    secondary_benchmark_disposition: str
    provided_lanes: frozenset[str]
    """What these inputs actually carry, for the selected package to check."""
    reference_mark: ReferenceMarkLane | None = None
    """Exact research close-to-entry marks; absent preserves legacy open-proxy behavior."""
    sector_history_hash: str | None = None
    """The Sector history's identity while a reclassification is in force."""

    def require_lanes(self, lanes: tuple[str, ...]) -> None:
        """Refuse a package whose declared lanes this resolution does not carry."""
        missing = tuple(sorted(set(lanes) - self.provided_lanes))
        if missing:
            raise StrategyPackageError(
                "portfolio_application.package_shared_lane_absent:" + ",".join(missing)
            )


@dataclass(frozen=True, slots=True)
class SharedPortfolioAuthorities:
    """The shared half of a PLAN receipt, before a package names its own."""

    coverage: PortfolioSupportCoverage
    candidate_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    risk_recipe_hash: str
    risk_return_surface_hash: str
    sector_map_hash: str
    tradability_decision_hash: str
    execution_outcome_manifest_hash: str
    sector_history_hash: str | None = None
    """The Sector history's identity while a reclassification is in force."""


@dataclass(frozen=True, slots=True)
class FrozenSharedMarketInputs:
    """Verified historical Market/Tradability/Outcome lanes from one authority."""

    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    decision_eligible: BoolArray
    execution_available: BoolArray
    realized_simple_returns: FloatArray
    causal_adv20: FloatArray | None
    reference_mark: ReferenceMarkLane | None = None


@runtime_checkable
class FrozenSharedMarketSource(Protocol):
    """Optional package-bound history for shared lanes current readers cannot claim."""

    @property
    def formation_sessions(self) -> tuple[date, ...]:
        """Read the frozen shared market formation axis.

        Returns:
            Declared formation sessions in source order.
        """
        ...

    @property
    def ordered_listing_ids(self) -> tuple[str, ...]:
        """Read the frozen shared market listing axis.

        Returns:
            Declared ordered listing identities.
        """
        ...

    @property
    def tradability_decision_hash(self) -> str:
        """Read the shared market tradability decision identity.

        Returns:
            Exact declared tradability decision hash.
        """
        ...

    @property
    def execution_outcome_manifest_hash(self) -> str:
        """Read the shared market execution-outcome source identity.

        Returns:
            Exact declared execution outcome manifest hash.
        """
        ...

    def resolve(self) -> FrozenSharedMarketInputs:
        """Resolve the declared frozen shared market numerical and source inputs.

        Returns:
            Frozen shared market inputs supplied by their deterministic owner.
        """
        ...


@dataclass(frozen=True, slots=True)
class SharedPortfolioArtifactBinding:
    """Exact immutable report-lane references supplied by a research score authority."""

    artifact_root: Path
    risk_return_surface_hash: str
    sector_map_hash: str


@dataclass(frozen=True, slots=True)
class ScoreSupport:
    """What the selected package's score lane can answer for, before it answers.

    Declared cheaply from manifests so `PLAN` can intersect it with the shared
    lanes without replaying a single model. `formation_sessions` is `None` when
    the source imposes no independent bound and simply answers wherever the
    shared axis lands; `ordered_listing_ids` is `None` for the same reason.
    """

    owner_id: str
    lane: str
    identity_hash: str
    formation_sessions: tuple[date, ...] | None = None
    ordered_listing_ids: tuple[str, ...] | None = None
    frozen_shared_market_source: FrozenSharedMarketSource | None = None
    """Historical shared lanes admitted with this score authority, if required.

    This is a runtime owner, not an identity.  Its two public hashes enter the
    shared PLAN receipt and its bytes are reverified by ``resolve``.
    """

    requires_causal_rank_mu_maturity: bool = False
    """Whether the executable axis must wait for the rank-`mu` holding maturity.

    Declared by the package rather than assumed, because it is a property of the
    policy that reads the curve. Inheriting it everywhere is how the successor --
    which reads no curve at all -- ended up truncated by the predecessor's
    lookback.
    """
    shared_artifacts: SharedPortfolioArtifactBinding | None = None


@dataclass(frozen=True, slots=True)
class StrategyComponentResolution:
    """What one package's score lane produced: books to open, and their receipt."""

    components: tuple[ComponentBookFactory, ...]
    score_receipt_hashes: tuple[str, ...]
    """One receipt per formation over every component row consumed at it."""


@runtime_checkable
class StrategyScoreSource(Protocol):
    """One package's own score lane, over shared inputs it did not resolve."""

    @property
    def mode(self) -> ScoreSourceMode:
        """Read the installed score-source capability mode.

        Returns:
            Declared source mode.
        """
        ...

    @property
    def evidence_identity_hash(self) -> str:
        """Read the exact evidence identity selected by this installed score source.

        Returns:
            Declared source evidence identity.
        """
        ...

    def support(self, *, spec: PortfolioResearchSpec) -> ScoreSupport:
        """Resolve score-source support admitted by the declared research controls.

        Args:
            spec: Admitted research controls and selected score-source mode.

        Returns:
            Declared score support available to shared input resolution.
        """
        ...

    def resolve_components(
        self, *, shared: SharedPortfolioInputs, spec: PortfolioResearchSpec
    ) -> StrategyComponentResolution:
        """Resolve ordered strategy component scores against shared source authority.

        Args:
            shared: Resolved shared portfolio input authority.
            spec: Admitted strategy/mode and research controls.

        Returns:
            Component resolution with declared score/evidence coverage.
        """
        ...


@runtime_checkable
class SharedInputOwner(Protocol):
    """Whoever resolves the lanes every package shares, however it gets them.

    A protocol rather than the artifact resolver itself, so an acceptance test
    can supply recorded lanes and still drive the real compiler, executor,
    ledger, report and readback -- which is the only way to prove the generic
    path without the artifacts a desktop installation has and a case study does
    not.
    """

    def authorities(self, *, workspace: Path, support: ScoreSupport) -> SharedPortfolioAuthorities:
        """Resolve shared input authority on admitted score-source support.

        Args:
            workspace: Caller-owned workspace root.
            support: Declared selected score-source support.

        Returns:
            Shared common coverage and exact source/Risk/market authority.
        """
        ...

    def resolve(
        self, *, workspace: Path, spec: PortfolioResearchSpec, support: ScoreSupport
    ) -> SharedPortfolioInputs:
        """Resolve shared portfolio inputs under explicit controls and score-source support.

        Args:
            workspace: Caller-owned workspace root.
            spec: Admitted research controls.
            support: Declared score-source support.

        Returns:
            Shared execution, Risk, source and reporting inputs.
        """
        ...


def control_values_of(spec: PortfolioResearchSpec) -> dict[str, str]:
    """Render one request as the control displays a package declares over."""
    return {
        "top_k": str(spec.top_k),
        "tranches": str(spec.tranches),
        "exit_rank": str(spec.exit_rank),
        "weight_rule": str(spec.weight_rule),
    }


def tranche_recipe_of(spec: PortfolioResearchSpec) -> TrancheBookRecipe:
    """The shared tranche-book recipe one holdings request names."""
    if spec.weight_rule not in ADMITTED_WEIGHT_RULES:
        raise StrategyPackageError("portfolio_application.request_policy_weight_rule_not_admitted")

    return TrancheBookRecipe.create(
        top_k=spec.top_k,
        tranches=spec.tranches,
        exit_rank=spec.exit_rank,
        weight_rule=spec.weight_rule,
    )


def _hex64(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class InstalledPackageBinding:
    """One installed package, its score sources, and how it admits a request.

    There is one of these, not one per strategy. Whether a package freezes its
    policy or derives it from the request is a field on the declaration, so
    installing another strategy is a registration rather than another binding
    class -- which is the extension budget stated as code instead of prose.

    It is also where a declaration is held to what is actually installed. Every
    capability the package declares must arrive as a source of that mode and
    that authority, and no source may arrive for a capability the package does
    not declare. A package hash therefore describes exactly one runnable surface,
    and a mode the desktop cannot run is refused here -- at configuration, before
    any service starts -- rather than offered and failed later.

    `source_identity_hash` binds the implementation bytes behind the sources, so
    an edit to a strategy's own scoring code rotates every Program that strategy
    seals, the way the generic path's bytes already do.
    """

    __slots__ = ("_package", "_source_identity_hash", "_sources")

    def __init__(
        self,
        *,
        package: FrozenStrategyPackage,
        sources: Mapping[ScoreSourceMode, StrategyScoreSource],
        source_identity_hash: str,
    ) -> None:
        """Bind exact declared source modes/evidence to an installed strategy package.

        Args:
            package: Frozen installed package declaration.
            sources: Installed owners for exactly the declared source modes.
            source_identity_hash: Explicit lowercase hexadecimal installed source identity.

        Raises:
            StrategyPackageError: Source modes are missing/extra, owner mode/evidence differs from
                declaration or source hash syntax is invalid.
        """
        declared = set(package.installed_modes)
        if set(sources) != declared:
            absent = sorted((declared - set(sources)) | (set(sources) - declared))
            raise StrategyPackageError(
                "portfolio_application.package_capability_not_installed:" + ",".join(absent)
            )
        for mode, source in sources.items():
            if source.mode != mode:
                raise StrategyPackageError(
                    "portfolio_application.package_score_source_mode_mismatch"
                )
            if source.evidence_identity_hash != package.score_source(mode).evidence_identity_hash:
                # The declaration and the installed owner have to name the same
                # evidence, or the Program would bind an authority the run never
                # opened.
                raise StrategyPackageError(
                    "portfolio_application.package_score_source_authority_mismatch"
                )
        if not _hex64(source_identity_hash):
            raise StrategyPackageError("portfolio_application.package_source_identity_invalid")
        self._package = package
        self._sources: Mapping[ScoreSourceMode, StrategyScoreSource] = dict(sources)
        self._source_identity_hash = source_identity_hash

    @property
    def package(self) -> FrozenStrategyPackage:
        """Read the retained installed frozen package declaration.

        Returns:
            The retained package.
        """
        return self._package

    @property
    def source_identity_hash(self) -> str:
        """Read the explicit installed source identity supplied at binding.

        Returns:
            The validated source_identity_hash.
        """
        return self._source_identity_hash

    @property
    def installed_modes(self) -> tuple[ScoreSourceMode, ...]:
        """Read installed owner modes in binding registration order.

        Returns:
            Ordered source-mode mapping keys.
        """
        return tuple(self._sources)

    def admit_request(self, *, spec: PortfolioResearchSpec) -> None:
        """Refuse a request the package's control surface does not admit."""
        self._package.admit_control_values(control_values_of(spec))
        if self._package.policy_binding == "REQUEST_SELECTED" and (
            spec.weight_rule not in ADMITTED_WEIGHT_RULES
        ):
            raise StrategyPackageError(
                "portfolio_application.request_policy_weight_rule_not_admitted"
            )

    def policy_identity(self, *, spec: PortfolioResearchSpec) -> PackagePolicyIdentity:
        """Resolve the frozen package policy or admitted public request-selected recipe.

        Args:
            spec: Admitted research controls used only for request-selected policy.

        Returns:
            Exact frozen policy identity or public recipe/catalog/adapter binding.

        Raises:
            StrategyPackageError: Request-selected public policy requires an optimizer.
        """
        frozen = self._package.frozen_policy
        if frozen is not None:
            return frozen
        recipe = tranche_recipe_of(spec)
        catalog = build_public_portfolio_policy_catalog()
        adapter = catalog.resolve(recipe)
        binding = adapter.describe_adapter_binding()
        if adapter.solver_backed or binding.requires_optimizer:
            raise StrategyPackageError("portfolio_application.public_policy_requires_optimizer")
        return PackagePolicyIdentity(
            policy_recipe_hash=recipe.recipe_hash,
            policy_catalog_hash=catalog.binding.catalog_hash,
            policy_adapter_binding_hash=binding.binding_hash,
        )

    def score_source(self, mode: ScoreSourceMode) -> StrategyScoreSource:
        """Resolve the installed deterministic owner for one declared mode.

        Args:
            mode: Explicit installed score-source mode.

        Returns:
            Matching score-source owner.

        Raises:
            StrategyPackageError: The mode has no installed owner.
        """
        source = self._sources.get(mode)
        if source is None:
            raise StrategyPackageError(
                f"portfolio_application.package_score_source_not_installed:{mode}"
            )
        return source


class SelectedStrategy:
    """One package, one score-source mode, resolved together and carried as one.

    Selection is a pair, not a name: the successor's historical arrays and its
    current model scoring are the same strategy with different evidence, and
    every identity downstream has to say which of the two produced a number.
    """

    __slots__ = ("_binding", "_mode")

    def __init__(self, *, binding: InstalledPackageBinding, mode: ScoreSourceMode) -> None:
        """Select one admitted source mode together with its installed package binding.

        Args:
            binding: Exact installed package/source binding.
            mode: Mode admitted by the binding.

        Raises:
            StrategyPackageError: The requested mode is not installed.
        """
        if mode not in binding.installed_modes:
            raise StrategyPackageError(
                f"portfolio_application.package_score_source_not_installed:{mode}"
            )
        self._binding = binding
        self._mode = mode

    @property
    def package(self) -> FrozenStrategyPackage:
        """Read the selected installed frozen package.

        Returns:
            Package declaration retained by the installed binding.
        """
        return self._binding.package

    @property
    def mode(self) -> ScoreSourceMode:
        """Read the explicitly selected score-source mode.

        Returns:
            The retained source mode.
        """
        return self._mode

    @property
    def alpha_recipe_hash(self) -> str:
        """Read the selected package Alpha recipe identity.

        Returns:
            The package alpha_recipe_hash.
        """
        return self.package.alpha_recipe_hash

    @property
    def alpha_evidence_manifest_hash(self) -> str:
        """Read the evidence identity declared for the selected source mode.

        Returns:
            Selected score-source evidence_identity_hash.
        """
        return self.package.score_source(self._mode).evidence_identity_hash

    @property
    def source_identity_hash(self) -> str:
        """Read the installed source identity selecting this strategy.

        Returns:
            Source identity retained by the installed binding.
        """
        return self._binding.source_identity_hash

    def admit_request(self, *, spec: PortfolioResearchSpec) -> None:
        """Admit selected research controls through the package control-surface owner.

        Args:
            spec: Request controls to compare with frozen/admitted package declarations.

        Raises:
            StrategyPackageError: Controls or a request-selected weight rule are not admitted.
        """
        self._binding.admit_request(spec=spec)

    def policy_identity(self, *, spec: PortfolioResearchSpec) -> PackagePolicyIdentity:
        """Resolve policy identity through the selected installed package binding.

        Args:
            spec: Admitted research controls.

        Returns:
            Frozen or request-selected policy recipe/catalog/adapter identity.
        """
        return self._binding.policy_identity(spec=spec)

    def score_source(self) -> StrategyScoreSource:
        """Read the deterministic score-source owner for the selected mode.

        Returns:
            Installed owner selected by the retained package/mode pair.
        """
        return self._binding.score_source(self._mode)


class InstalledFrozenStrategyCatalog:
    """The finite installed strategy surface, with one identity over declarations.

    Not a loader. It receives already-constructed installed owners and never
    imports, scans or executes anything to find one; its hash covers the package
    declarations alone, so registering the same packages behind different roots
    produces the same catalog identity.
    """

    __slots__ = ("_bindings", "_catalog_hash", "_default")

    def __init__(
        self,
        bindings: Iterable[InstalledPackageBinding],
        *,
        default_strategy_id: str | None,
    ) -> None:
        """Index a nonempty finite strategy catalog and seal sorted package declarations.

        Args:
            bindings: Explicit already-constructed installed package bindings.
            default_strategy_id: Optional admitted default strategy identity.

        Raises:
            StrategyPackageError: Catalog is empty, strategy IDs repeat or the declared default is
                not installed.
        """
        ordered = tuple(bindings)
        if not ordered:
            raise StrategyPackageError("portfolio_application.strategy_catalog_empty")
        by_id: dict[str, InstalledPackageBinding] = {}
        for binding in ordered:
            strategy_id = binding.package.strategy_id
            if strategy_id in by_id:
                raise StrategyPackageError("portfolio_application.strategy_catalog_duplicated")
            by_id[strategy_id] = binding
        if default_strategy_id is not None and default_strategy_id not in by_id:
            raise StrategyPackageError(
                f"portfolio_application.strategy_not_installed:{default_strategy_id}"
            )
        self._bindings: Mapping[str, InstalledPackageBinding] = by_id
        self._default = default_strategy_id
        self._catalog_hash = str(
            canonical_hash(
                {
                    "kind": "InstalledFrozenStrategyCatalog",
                    "default_strategy_id": default_strategy_id,
                    "packages": [
                        by_id[value].package.model_dump(mode="json") for value in sorted(by_id)
                    ],
                }
            )
        )

    @property
    def catalog_hash(self) -> str:
        """Read the installed catalog identity over sorted package declarations and default.

        Returns:
            The retained catalog hash; runtime roots are not part of this declaration.
        """
        return self._catalog_hash

    @property
    def default_strategy_id(self) -> str | None:
        """Read the declared default strategy when one is installed.

        Returns:
            Declared default identity or None.
        """
        return self._default

    def packages(self) -> Mapping[str, FrozenStrategyPackage]:
        """Installed declarations keyed by package hash for execution/readback."""
        return {
            binding.package.package_hash: binding.package for binding in self._bindings.values()
        }

    def packages_by_strategy_id(self) -> Mapping[str, FrozenStrategyPackage]:
        """Installed declarations keyed by the request-facing strategy id."""
        return {strategy_id: binding.package for strategy_id, binding in self._bindings.items()}

    def with_default(self, strategy_id: str) -> InstalledFrozenStrategyCatalog:
        """The same installed owners with a manifest-selected default."""
        return InstalledFrozenStrategyCatalog(
            self._bindings.values(), default_strategy_id=strategy_id
        )

    def _binding(self, strategy_id: str) -> InstalledPackageBinding:
        binding = self._bindings.get(strategy_id)
        if binding is None:
            raise StrategyPackageError(
                f"portfolio_application.strategy_not_installed:{strategy_id}"
            )
        return binding

    def select(
        self,
        *,
        strategy_id: str | None = None,
        score_source_mode: ScoreSourceMode | None = None,
    ) -> SelectedStrategy:
        """Resolve one selection, falling back to the installed defaults only.

        An unknown strategy or an uninstalled mode refuses by name. There is no
        nearest match and no silent downgrade: a run that asked for current
        scoring and got a historical array back would be a different experiment
        wearing the same label.
        """
        selected = self._default if strategy_id is None else strategy_id
        if selected is None:
            raise StrategyPackageError("portfolio_application.explicit_strategy_selection_required")
        binding = self._binding(selected)
        mode = (
            binding.package.default_score_source_mode
            if score_source_mode is None
            else score_source_mode
        )
        return SelectedStrategy(binding=binding, mode=mode)


__all__ = [
    "ALLOCATION_BASIS_POINTS_TOTAL",
    "VALUE_ROLE_PREFIXES",
    "ComponentPlanEntry",
    "FrozenSharedMarketInputs",
    "FrozenSharedMarketSource",
    "FrozenStrategyPackage",
    "InstalledFrozenStrategyCatalog",
    "InstalledPackageBinding",
    "MergeSemantics",
    "PackageControlSurface",
    "PackageFrozenControl",
    "PackagePolicyIdentity",
    "PolicyBinding",
    "RiskDisposition",
    "ScoreSourceCapability",
    "ScoreSourceMode",
    "ScoreSupport",
    "SelectedStrategy",
    "SharedInputOwner",
    "SharedPortfolioAuthorities",
    "SharedPortfolioInputs",
    "StrategyComponentResolution",
    "StrategyDisclosure",
    "StrategyPackageError",
    "StrategyScoreSource",
    "control_values_of",
    "recipe_hashes_match",
    "tranche_recipe_of",
]
