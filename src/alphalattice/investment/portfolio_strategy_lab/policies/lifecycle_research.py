"""Frozen book recipes over independently materialized local lifecycle research.

This is a historical score source, never a fit owner or current-scoring adapter.
The Host verifies the Alpha executions and materializes their complete scores and
actual market lanes here; the ordinary Portfolio resolver/executor consumes them.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal, Self
from uuid import UUID

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    MARKED_TO_MARKET_AT_CLOSE_T,
    PortfolioStateTransitionBinding,
)
from alphalattice.capabilities.portfolio_backtesting.reference_marks import ReferenceMarkLane
from alphalattice.foundation.causal_outcomes.execution.methods import build_one_session_recipe
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
)
from alphalattice.investment.alpha_research.scores.product_lifecycle import (
    AlphaModelLifecycleRecipe,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioResearchSpec,
)
from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
    PortfolioDataExclusion,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    FrozenSharedMarketInputs,
    ScoreSupport,
    SharedPortfolioArtifactBinding,
    SharedPortfolioInputs,
    StrategyComponentResolution,
    StrategyPackageError,
)
from alphalattice.investment.portfolio_strategy_lab.policies.post_observed_authority import (
    FrozenHistoricalBookRecipe,
    resolve_frozen_book_components,
)
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioResearchArtifactStore,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

CATEGORY = "lifecycle-research-authorities"
LANES = "lifecycle-research-lanes"
ARTIFACT_KEY = "LIFECYCLE_RESEARCH_STRATEGY_AUTHORITY"
_HASH = r"^[0-9a-f]{64}$"
type FloatArray = npt.NDArray[np.float64]


class LifecycleScoreEvidence(BaseModel):  # type: ignore[misc]
    """Bind one local lifecycle component recipe, score axis and numerical evidence identities."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    component_id: str
    task_id: UUID
    program_hash: str = Field(pattern=_HASH)
    receipt_hash: str = Field(pattern=_HASH)
    component_recipe_hash: str = Field(pattern=_HASH)
    lifecycle_hash: str = Field(pattern=_HASH)
    formation_sessions: tuple[date, ...]
    scores_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_frozen_recipe(self) -> Self:
        """Require installed component/lifecycle recipes and nonempty ordered score support.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            StrategyPackageError: Component/lifecycle recipe hash or sorted unique formation support
                differs.
        """
        recipe = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(self.component_id)
        if (
            recipe.recipe_hash != self.component_recipe_hash
            or AlphaModelLifecycleRecipe.from_component(recipe).content_hash != self.lifecycle_hash
            or not self.formation_sessions
            or self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
        ):
            raise StrategyPackageError("portfolio_application.lifecycle_recipe_or_axis_mismatch")
        return self


class LocalQAOutcomeBinding(BaseModel):  # type: ignore[misc]
    """Consumed rows of Foundation's admitted QA reader, never a holdout release."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["LocalQAOutcomeBinding"] = "LocalQAOutcomeBinding"
    purpose: Literal["LOCAL_QA_OBSERVATIONS_NOT_HOLDOUT_RELEASE"] = (
        "LOCAL_QA_OBSERVATIONS_NOT_HOLDOUT_RELEASE"
    )
    input_binding_hash: str = Field(pattern=_HASH)
    method_recipe_hash: str = Field(pattern=_HASH)
    source_rows_hash: str = Field(pattern=_HASH)
    returns_hash: str = Field(pattern=_HASH)
    formation_sessions: tuple[date, ...]
    listing_axis_hash: str = Field(pattern=_HASH)
    through: date
    content_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal one local QA outcome binding.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical content_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = cls.model_construct(**values, content_hash="0" * 64)
        return cls(
            **values,
            content_hash=canonical_hash(draft.model_dump(mode="json", exclude={"content_hash"})),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def verify(self) -> Self:
        """Require exact QA outcome identity, causal support and installed return recipe.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            StrategyPackageError: Canonical identity, nonempty ordered support before through, or
                one-session method recipe differs.
        """
        if (
            self.content_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"content_hash"}))
            or not self.formation_sessions
            or self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or self.formation_sessions[-1] >= self.through
            or self.method_recipe_hash != build_one_session_recipe().recipe_hash
        ):
            raise StrategyPackageError("portfolio_application.local_qa_outcome_binding_invalid")
        return self


class LifecyclePortfolioAuthority(BaseModel):  # type: ignore[misc]
    """Bind local lifecycle scores, market execution, calibration and explicit data disposition."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["LocalLifecyclePortfolioAuthority"] = "LocalLifecyclePortfolioAuthority"
    input_binding_hash: str = Field(pattern=_HASH)
    preparation_request_hash: str = Field(pattern=_HASH)
    preparation_implementation_hash: str = Field(pattern=_HASH)
    components: tuple[LifecycleScoreEvidence, ...]
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    decision_hash: str = Field(pattern=_HASH)
    execution_hash: str = Field(pattern=_HASH)
    returns_hash: str = Field(pattern=_HASH)
    adv_hash: str = Field(pattern=_HASH)
    marks_hash: str = Field(pattern=_HASH)
    entry_sessions: tuple[date, ...]
    holding_end_sessions: tuple[date, ...]
    transition: PortfolioStateTransitionBinding
    tradability_decision_hash: str = Field(pattern=_HASH)
    outcome_snapshot_hash: str = Field(pattern=_HASH)
    risk_return_surface_hash: str = Field(pattern=_HASH)
    sector_map_hash: str = Field(pattern=_HASH)
    curve_hash: str = Field(pattern=_HASH)
    latest_calibration_hash: str = Field(pattern=_HASH)
    unavailable_return_policy: Literal["require_complete", "quarantine_listings"]
    data_exclusions: tuple[PortfolioDataExclusion, ...] = ()
    qa_outcome: LocalQAOutcomeBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    authority_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal one local lifecycle portfolio authority.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical authority_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = cls.model_construct(**values, authority_hash="0" * 64)
        return cls(
            **values,
            authority_hash=canonical_hash(
                draft.model_dump(mode="json", exclude={"authority_hash"})
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact lifecycle authority, common axes, causal execution and QA bindings.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            StrategyPackageError: Identity, sorted unique/nonempty axes, entry/end timing,
                close-mark transition, unique components, exclusions or QA input/return/axis/through
                bindings differ.
        """
        if (
            self.authority_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"authority_hash"}))
            or not self.formation_sessions
            or self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or not self.ordered_listing_ids
            or self.ordered_listing_ids != tuple(sorted(set(self.ordered_listing_ids)))
            or len(self.entry_sessions) != len(self.formation_sessions)
            or len(self.holding_end_sessions) != len(self.formation_sessions)
            or any(
                entry <= day
                for day, entry in zip(self.formation_sessions, self.entry_sessions, strict=True)
            )
            or self.transition.reference_mark_method != MARKED_TO_MARKET_AT_CLOSE_T
            or any(
                end <= entry
                for entry, end in zip(self.entry_sessions, self.holding_end_sessions, strict=True)
            )
            or not self.components
            or len({v.component_id for v in self.components}) != len(self.components)
            or (self.unavailable_return_policy == "require_complete" and self.data_exclusions)
            or not {v.listing_id for v in self.data_exclusions} <= set(self.ordered_listing_ids)
        ):
            raise StrategyPackageError("portfolio_application.lifecycle_authority_invalid")
        qa = self.qa_outcome
        if qa is not None and (
            qa.input_binding_hash != self.input_binding_hash
            or qa.formation_sessions != self.formation_sessions
            or qa.listing_axis_hash != canonical_hash(self.ordered_listing_ids)
            or qa.returns_hash != self.returns_hash
            or qa.content_hash != self.outcome_snapshot_hash
            or any(end > qa.through for end in self.holding_end_sessions)
        ):
            raise StrategyPackageError("portfolio_application.local_qa_outcome_axis_mismatch")
        return self


class LifecycleResearchScoreSource:
    """One parameterized historical source for any admitted frozen component book."""

    mode: Literal["HISTORICAL_ARRAY_REPLAY"] = "HISTORICAL_ARRAY_REPLAY"

    def __init__(self, *, manifest_path: Path, recipe: FrozenHistoricalBookRecipe) -> None:
        """Reopen a sealed lifecycle manifest and admit the common component support.

        Args:
            manifest_path: Exact content-addressed lifecycle authority manifest in its registered
                category.
            recipe: Declared frozen historical component book recipe.

        Raises:
            StrategyPackageError: Manifest location, required component or common score support is
                invalid.
        """
        path = manifest_path.resolve()
        if path.parent.name != CATEGORY or path.parent.parent.name != "portfolio-strategy-lab":
            raise StrategyPackageError("portfolio_application.lifecycle_manifest_path_invalid")
        self.store = PortfolioResearchArtifactStore(path.parents[2])
        self.recipe = recipe
        self.manifest = self.store.load(
            category=CATEGORY,
            content_hash=path.stem,
            model=LifecyclePortfolioAuthority,
            identity_field="authority_hash",
        )
        self._components = {v.component_id: v for v in self.manifest.components}
        if any(v.component_id not in self._components for v in recipe.components):
            raise StrategyPackageError("portfolio_application.lifecycle_component_absent")
        starts = [self._components[v.component_id].formation_sessions[0] for v in recipe.components]
        ends = [self._components[v.component_id].formation_sessions[-1] for v in recipe.components]
        self.formation_sessions = tuple(
            day for day in self.manifest.formation_sessions if max(starts) <= day <= min(ends)
        )
        if not self.formation_sessions or any(
            not set(self.formation_sessions)
            <= set(self._components[v.component_id].formation_sessions)
            for v in recipe.components
        ):
            raise StrategyPackageError("portfolio_application.lifecycle_interior_score_gap")
        self.ordered_listing_ids = self.manifest.ordered_listing_ids

    @property
    def evidence_identity_hash(self) -> str:
        """Read the admitted lifecycle authority identity.

        Returns:
            Manifest authority_hash.
        """
        return self.manifest.authority_hash

    @property
    def tradability_decision_hash(self) -> str:
        """Read the declared lifecycle tradability decision identity.

        Returns:
            Manifest tradability_decision_hash.
        """
        return self.manifest.tradability_decision_hash

    @property
    def execution_outcome_manifest_hash(self) -> str:
        """Read the declared lifecycle execution outcome identity.

        Returns:
            Manifest outcome_snapshot_hash.
        """
        return self.manifest.outcome_snapshot_hash

    def support(self, *, spec: PortfolioResearchSpec) -> ScoreSupport:
        """Verify manifest bindings and expose frozen shared market/artifact support.

        Args:
            spec: Research controls unused by this frozen support declaration.

        Returns:
            Verified lifecycle score axes and shared market/Risk/sector artifact bindings.
        """
        del spec
        self._verify_manifest()
        return ScoreSupport(
            owner_id="local_lifecycle_research",
            lane="VERIFIED_LOCAL_LIFECYCLE_SCORES",
            identity_hash=self.evidence_identity_hash,
            formation_sessions=self.formation_sessions,
            ordered_listing_ids=self.ordered_listing_ids,
            frozen_shared_market_source=self,
            shared_artifacts=SharedPortfolioArtifactBinding(
                artifact_root=self.store.root.parent,
                risk_return_surface_hash=self.manifest.risk_return_surface_hash,
                sector_map_hash=self.manifest.sector_map_hash,
            ),
        )

    def _lane(self, identity: str, shape: tuple[int, ...], dtype: str = "<f8") -> npt.NDArray[Any]:
        payload = self.store.load_packed_bytes(category=LANES, content_hash=identity)
        if len(payload) != int(np.prod(shape)) * np.dtype(dtype).itemsize:
            raise StrategyPackageError("portfolio_application.lifecycle_lane_axis_invalid")
        return np.frombuffer(payload, dtype=dtype).reshape(shape)

    def _verify_manifest(self) -> None:
        self.store.load(
            category=CATEGORY,
            content_hash=self.manifest.authority_hash,
            model=LifecyclePortfolioAuthority,
            identity_field="authority_hash",
        )

    def _rows(self, axis: tuple[date, ...]) -> list[int]:
        positions = {day: i for i, day in enumerate(axis)}
        return [positions[day] for day in self.formation_sessions]

    def verify(self) -> None:
        """Prove packed lanes and causal curve admissibility without executing a book."""
        self.resolve()
        for component in self.manifest.components:
            scores = self._lane(
                component.scores_hash,
                (len(component.formation_sessions), len(self.ordered_listing_ids)),
            )
            if np.isinf(scores).any():
                raise StrategyPackageError("portfolio_application.lifecycle_score_values_invalid")
        count = len(self.manifest.formation_sessions)
        curve = self._lane(self.manifest.curve_hash, (count, 20))
        latest = self._lane(self.manifest.latest_calibration_hash, (count,), "<i8")
        rows = self._rows(self.manifest.formation_sessions)
        if any(c.weight_rule == "mu.iv0" for c in self.recipe.components):
            for local, row in enumerate(rows):
                if local < self.recipe.sizing_activation_formation:
                    continue
                last = int(latest[row])
                if (
                    not np.isfinite(curve[row]).all()
                    or not 0 <= last < row
                    or self.manifest.holding_end_sessions[last]
                    > self.manifest.formation_sessions[row]
                ):
                    raise StrategyPackageError("portfolio_application.lifecycle_curve_not_causal")

    def resolve(self) -> FrozenSharedMarketInputs:
        """The same score authority also binds its immutable observed market lanes."""
        self._verify_manifest()
        m = self.manifest
        shape = (len(m.formation_sessions), len(m.ordered_listing_ids))
        rows = self._rows(m.formation_sessions)
        decision = self._lane(m.decision_hash, shape, "?")[rows]
        execution = self._lane(m.execution_hash, shape, "?")[rows]
        returns = self._lane(m.returns_hash, shape)[rows]
        marks = self._lane(m.marks_hash, shape)[rows]
        if np.isinf(returns).any() or not np.isfinite(returns[decision]).all():
            raise StrategyPackageError("portfolio_application.lifecycle_market_values_invalid")
        return FrozenSharedMarketInputs(
            formation_sessions=self.formation_sessions,
            ordered_listing_ids=self.ordered_listing_ids,
            decision_eligible=decision,
            execution_available=execution,
            realized_simple_returns=returns,
            causal_adv20=self._lane(m.adv_hash, shape)[rows],
            reference_mark=ReferenceMarkLane(
                method=m.transition.reference_mark_method,
                marks_by_session=dict(zip(self.formation_sessions, marks, strict=True)),
                entry_session_by_formation={
                    day: m.entry_sessions[index]
                    for day, index in zip(self.formation_sessions, rows, strict=True)
                },
                state_transition_binding_hash=m.transition.binding_hash,
            ),
        )

    def resolve_components(
        self, *, shared: SharedPortfolioInputs, spec: PortfolioResearchSpec
    ) -> StrategyComponentResolution:
        """Verify lifecycle authority and align component/calibration arrays.

        Verify lifecycle authority and resolve exact component/calibration arrays on shared support.

        Args:
            shared: Exact common portfolio source, listing and execution support.
            spec: Admitted research controls selecting declared input consumption.

        Returns:
            Frozen component resolution from content-addressed score, curve and calibration lanes.

        Raises:
            StrategyPackageError: Shared formation support differs or retained manifest/array
                verification fails.
        """
        del spec
        self.verify()
        if shared.formation_sessions != self.formation_sessions:
            raise StrategyPackageError("portfolio_application.lifecycle_book_support_changed")
        scores = {}
        for recipe_component in self.recipe.components:
            component = self._components[recipe_component.component_id]
            scores[component.component_id] = self._lane(
                component.scores_hash,
                (len(component.formation_sessions), len(self.ordered_listing_ids)),
            )[self._rows(component.formation_sessions)]
        rows = self._rows(self.manifest.formation_sessions)
        return resolve_frozen_book_components(
            recipe=self.recipe,
            authority_hash=self.evidence_identity_hash,
            authority_sessions=self.formation_sessions,
            ordered_listing_ids=self.ordered_listing_ids,
            score_arrays=scores,
            shared=shared,
            curve=self._lane(self.manifest.curve_hash, (len(self.manifest.formation_sessions), 20))[
                rows
            ],
            latest=self._lane(
                self.manifest.latest_calibration_hash,
                (len(self.manifest.formation_sessions),),
                "<i8",
            )[rows],
        )
