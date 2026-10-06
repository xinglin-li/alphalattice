"""The public-product Alpha lifecycle seam: fit-or-score, and what proves it ran.

Replay reads sealed artifacts and never fits; the current refit owner fits and
knows nothing about a product calendar. Neither of them can answer the question a
daily lifecycle actually asks -- *for this formation, is a fit due, and are the
twelve live models really here* -- so this seam asks it, and delegates both
halves rather than reimplementing either.

Two rules give the seam its shape.

**The order is fit, then verify, then score.** Whether a quarter is due is
decided *before* any artifact is read, from the recipe calendar and the
fitted-vintage authority alone -- so a formation whose current vintage has no
receipt yet is a formation that needs the declared fit, not one that is short of
history. Reading first and refusing on the absence would make the quarterly refit
unreachable: the receipt it is supposed to produce is exactly what is missing.

**A vintage string is not fit authority.** After the fit, the seam rereads all
twelve receipts and each one must bind its vintage, its seed, the exact frozen
training axis, the purge, the Feature axis, the estimator identity and its own
output digest. Eleven receipts is not "mostly ready": it is
`INSUFFICIENT_HISTORY`, whether the twelfth was never due or the declared fit
failed to produce it. Nothing scores with fewer models and renormalised weights.

**The calendar is the recipe's.** Due-ness comes from
``refit_quarter_start_months`` and from which vintages have already been fitted --
never from how much time has passed, how busy the machine is, or how many
sessions were missed. That is what makes a catch-up fit one quarter once.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.alpha_research.scores.product_replay import (
    AlphaProductRecipeView,
    calendar_model_vintages,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash

AlphaLifecycleDisposition = Literal[
    "SCORE_ONLY",
    "FIT_THEN_VERIFY_THEN_SCORE",
    "INSUFFICIENT_HISTORY",
]


def lifecycle_implementation_hash() -> str:
    return str(
        source_rule_closure_hash(
            root=resolve_playpen_root(Path(__file__)),
            semantic_owner="alpha_research",
            numerical_role="MODEL_LIFECYCLE_REPLAY",
            tracked_paths=tuple(
                f"src/alphalattice/investment/alpha_research/{path}"
                for path in (
                    "experiments/lifecycle_authoring.py",
                    "scores/product_lifecycle.py",
                    "scores/model_renewal.py",
                    "targets/component_training.py",
                    "targets/total_return.py",
                    "scores/frozen_inference.py",
                    "inputs/frozen_price_volume.py",
                    "inputs/panel_feature_materialization.py",
                    "inputs/panel_feature_views.py",
                    "scores/product_replay.py",
                    "experiments/development_artifacts.py",
                    "publication/artifacts.py",
                    "publication/contracts.py",
                    "inputs/preprocessing/sector_context.py",
                )
            )
            + tuple(
                f"src/alphalattice/capabilities/alpha_modeling/{path}"
                for path in (
                    "adapters/lightgbm_dynamic_panel.py",
                    "adapters/lightgbm_chronological.py",
                    "runtime/service.py",
                    "runtime/numerical_environment.py",
                )
            )
            + ("src/alphalattice/kernel/quant/cross_section.py",),
        )
    )


class AlphaLifecycleError(ValueError):
    """Stable refusal for a product lifecycle identity or authority failure."""


class _LifecycleContract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        payload = cls.model_construct(**values, content_hash="0" * 64).model_dump(
            mode="json", exclude={"content_hash"}
        )
        return cls(**payload, content_hash=canonical_hash(payload))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def verify_identity(self) -> Self:
        if self.content_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"content_hash"})
        ):
            raise AlphaLifecycleError("alpha_research.model_lifecycle_identity_invalid")
        return self


class AlphaModelLifecycleRecipe(_LifecycleContract):
    """Research-selectable deployment rules, not model or execution authority.

    Evaluation folds remain in their existing owner. The execution envelope's
    random seed is not this explicitly declared estimator ensemble.
    """

    month_interval: Literal[1, 3, 12] = 3
    anchor_month: int = Field(default=1, ge=1, le=12)
    training_window_sessions: int = Field(ge=1)
    purge_sessions: int = Field(ge=0)
    seeds: tuple[int, ...] = Field(min_length=1)
    vintage_weights: tuple[int, ...] = Field(min_length=1)
    score_aggregation: Literal[
        "PER_MODEL_PERCENTILE_0_1_THEN_EQUAL_SEED_MEAN_THEN_WEIGHTED_VINTAGE_MEAN",
        "EQUAL_RAW_SEED_MEAN_THEN_CANDIDATE_PERCENTILE_1_N_TO_1_THEN_"
        "WEIGHTED_VINTAGE_MEAN_WITH_OUTSIDER_MINUS_ONE",
    ]
    missing_model_policy: Literal["REQUIRE_COMPLETE_SET"] = "REQUIRE_COMPLETE_SET"

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_rules(self) -> Self:
        if len(set(self.seeds)) != len(self.seeds) or any(
            weight <= 0 for weight in self.vintage_weights
        ):
            raise AlphaLifecycleError("alpha_research.model_lifecycle_rules_invalid")
        return self

    @property
    def vintage_count(self) -> int:
        return len(self.vintage_weights)

    def vintages(self, formation: date) -> tuple[str, ...]:
        return tuple(
            calendar_model_vintages(
                formation,
                count=self.vintage_count,
                month_interval=self.month_interval,
                anchor_month=self.anchor_month,
            )
        )

    @classmethod
    def from_component(cls, recipe: AlphaProductRecipeView) -> Self:
        """Project the frozen declaration without serializing new fields into it."""
        if tuple(recipe.refit_quarter_start_months) != (1, 4, 7, 10):
            raise AlphaLifecycleError("alpha_research.model_lifecycle_calendar_not_installed")
        return cls.create(
            training_window_sessions=recipe.training_window_sessions,
            purge_sessions=recipe.purge_sessions,
            seeds=recipe.seeds,
            vintage_weights=recipe.vintage_weights,
            score_aggregation=recipe.score_aggregation,
        )


class ResolvedAlphaRefitPlan(_LifecycleContract):
    """One immutable, cutoff-bound period; inputs must prove these exact axes."""

    lifecycle: AlphaModelLifecycleRecipe
    component_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    vintage: str = Field(pattern=r"^\d{4}-\d{2}$")
    first_formation: date
    training_sessions: tuple[date, ...]
    purge_sessions: tuple[date, ...]
    source_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_listing_ids: tuple[str, ...] = Field(min_length=2)
    ordered_feature_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def verify_axes(self) -> Self:
        sessions = (*self.training_sessions, *self.purge_sessions, self.first_formation)
        if (
            len(self.training_sessions) != self.lifecycle.training_window_sessions
            or len(self.purge_sessions) != self.lifecycle.purge_sessions
            or sessions != tuple(sorted(set(sessions)))
            or self.lifecycle.vintages(self.first_formation)[0] != self.vintage
            or len(set(self.ordered_listing_ids)) != len(self.ordered_listing_ids)
            or len(set(self.ordered_feature_ids)) != len(self.ordered_feature_ids)
        ):
            raise AlphaLifecycleError("alpha_research.refit_plan_axis_invalid")
        return self


def resolve_alpha_refit_plan(
    *,
    lifecycle: AlphaModelLifecycleRecipe,
    vintage: str,
    sessions: tuple[date, ...],
    component_recipe_hash: str,
    source_binding_hash: str,
    ordered_listing_ids: tuple[str, ...],
    ordered_feature_ids: tuple[str, ...],
) -> ResolvedAlphaRefitPlan:
    """Resolve calendar periods against a complete admitted trading-session axis."""
    if sessions != tuple(sorted(set(sessions))):
        raise AlphaLifecycleError("alpha_research.refit_source_axis_invalid")
    boundary = date.fromisoformat(vintage + "-01")
    first = next((i for i, day in enumerate(sessions) if day >= boundary), None)
    if first is None or lifecycle.vintages(sessions[first])[0] != vintage:
        raise AlphaLifecycleError("alpha_research.refit_period_unavailable")
    stop = first - lifecycle.purge_sessions
    start = stop - lifecycle.training_window_sessions
    if start < 0:
        raise AlphaLifecycleError("alpha_research.refit_history_insufficient")
    return ResolvedAlphaRefitPlan.create(
        lifecycle=lifecycle,
        component_recipe_hash=component_recipe_hash,
        vintage=vintage,
        first_formation=sessions[first],
        training_sessions=sessions[start:stop],
        purge_sessions=sessions[stop:first],
        source_binding_hash=source_binding_hash,
        ordered_listing_ids=ordered_listing_ids,
        ordered_feature_ids=ordered_feature_ids,
    )


__all__ = ["AlphaLifecycleDisposition", "AlphaLifecycleError"]
