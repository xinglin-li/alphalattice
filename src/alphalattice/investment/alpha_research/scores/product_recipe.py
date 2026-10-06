"""The frozen public Alpha product recipe: one scientific identity, twelve live models.

This owner exists because nothing in the tree owned a *vintage set*. The
estimator adapter owns a bounded hyperparameter space, ``temporal_aggregation``
owns trailing-mean spans over an existing surface, and ``refit`` owns executing a
current fit -- but the thing the public desktop actually ships is a recipe that
says which twelve models are live at a formation, how their seeds and vintages
combine, and which ordered Feature axis they were fitted on. That statement had
no home, so a product could only be described in prose.

The distinction against the adapter is the point and not a technicality. The
installed ``DynamicPanelLightGBMAdapter`` admits any configuration inside its
declared bounds, deliberately, so that stating a configuration is a document
change rather than a source change. A *product* recipe is the opposite: it pins
one point and refuses to be a range. Both are correct, and they are different
objects, so this module states the point explicitly rather than pointing at the
space that contains it.

What this module does not do: fit, predict, score, aggregate, cache, or resolve
an artifact root. It resolves into the installed adapter's own parameter type and
stops there.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
    DynamicPanelLightGBMParameters,
)
from alphalattice.kernel.shared_kernel.recipe_identity import recipe_identity, recipe_seal_holds

_SEAL: Final = frozenset({"recipe_hash"})

PRODUCT_RECIPE_ID: Final = "IW184_IMPLIED_PLUS_WITHIN_FIXED1260_QUARTERLY_SEED3_VINTAGE4"
"""Stable domain name. It states the axis and the four identity-bearing choices."""

PRODUCT_FEATURE_AXIS_ID: Final = "IW184_IMPLIED_PLUS_WITHIN"
PRODUCT_FEATURE_AXIS_HASH: Final = (
    "7d307c7637b046da4fca9eb05e5e19946635ab58e4a8904c115d6b7b7091d976"
)
PRODUCT_FEATURE_COUNT: Final = 184
PRODUCT_SOURCE_CANDIDATE_MANIFEST_SHA256: Final = (
    "7e8618accd8efd59a6c963f717d8e472a4c5527d706a7366ddf3eb97da62b10d"
)

PRODUCT_TRAINING_WINDOW_SESSIONS: Final = 1_260
"""Frozen scientific identity, bound only here.

It is not a Portfolio control, not a required score-output length, and not a
constraint on any ledger, study window or fold. Changing it creates a different
Alpha recipe, fit and score lineage rather than a variant of this one.
"""

PRODUCT_PURGE_SESSIONS: Final = 1
PRODUCT_REFIT_QUARTER_START_MONTHS: Final = (1, 4, 7, 10)
PRODUCT_SEEDS: Final = (1729, 2718, 31415)
PRODUCT_VINTAGE_COUNT: Final = 4
PRODUCT_VINTAGE_WEIGHTS: Final = (4, 3, 2, 1)
PRODUCT_LIVE_MODEL_COUNT: Final = 12

PRODUCT_EVIDENCE_DISPOSITION: Final = "DEVELOPMENT_DIAGNOSTIC_CANDIDATE_NOT_INSTALLED"
"""The evidence package's own word for itself, carried rather than upgraded.

Installing a recipe that cites a diagnostic package does not promote the package.
A product claim needs its own admission; this field exists so that reading the
recipe cannot leave anyone with the opposite impression.
"""

PRODUCT_EVIDENCE_DAILY_PARQUET_SHA256: Final = (
    "fc61bc9cd67eabdd562cd308fb2f091d8984d040ab41dd477998dcf63256532e"
)
PRODUCT_EVIDENCE_SUPPORT_FIRST: Final = date(2022, 7, 1)
PRODUCT_EVIDENCE_SUPPORT_LAST: Final = date(2024, 8, 12)
PRODUCT_EVIDENCE_SUPPORT_SESSION_COUNT: Final = 529


class AlphaProductRecipeError(ValueError):
    """Stable refusal for a product recipe identity or consistency failure."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlphaEstimatorParameterPoint(_Contract):
    """One pinned point in the installed adapter's bounded space.

    Field names match ``DynamicPanelLightGBMParameters`` so that ``resolve``
    below is a rename-free construction and a reader can diff the two by eye.
    The values live here and not in a research script, which is the whole reason
    this type exists: the point was previously reachable only through a private
    function in an ignored workspace module, loaded by path.
    """

    kind: Literal["AlphaEstimatorParameterPoint"] = "AlphaEstimatorParameterPoint"
    adapter_id: Literal["dynamic_panel_lightgbm"] = "dynamic_panel_lightgbm"
    max_depth: int = Field(ge=1)
    num_leaves: int = Field(ge=2)
    min_child_samples: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    lambda_l1: float = Field(ge=0.0)
    lambda_l2: float = Field(ge=0.0)
    min_gain_to_split: float = Field(ge=0.0)
    feature_fraction: float = Field(gt=0.0, le=1.0)
    bagging_fraction: float = Field(gt=0.0, le=1.0)
    bagging_freq: int = Field(ge=0)
    training_policy: Literal["FIXED_ITERATION"] = "FIXED_ITERATION"
    fixed_iterations: int = Field(ge=1)

    def resolve(self, *, seed: int) -> DynamicPanelLightGBMParameters:
        """Hand this point to the installed adapter's own parameter type.

        The adapter validates bounds on construction, so a point that drifts
        outside the installed space fails here rather than at fit time.
        """
        return DynamicPanelLightGBMParameters(
            seed=seed,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            learning_rate=self.learning_rate,
            lambda_l1=self.lambda_l1,
            lambda_l2=self.lambda_l2,
            min_gain_to_split=self.min_gain_to_split,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            training_policy=self.training_policy,
            fixed_iterations=self.fixed_iterations,
        )


PRODUCT_ESTIMATOR_POINT: Final = AlphaEstimatorParameterPoint(
    max_depth=5,
    num_leaves=31,
    min_child_samples=50,
    learning_rate=0.05,
    lambda_l1=10.0,
    lambda_l2=100.0,
    min_gain_to_split=0.01,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=1,
    fixed_iterations=300,
)


class AlphaProductRecipe(_Contract):
    """The complete frozen statement of what the public desktop scores with.

    Every field here is scientific identity. Operational choices -- threads,
    memory, chunking -- are deliberately absent, because they do not belong in a
    recipe hash and a profile that changed one must not look like a new model.
    """

    LABELS: ClassVar[frozenset[str]] = frozenset(
        {"recipe_id", "feature_axis_id", "evidence_disposition"}
    )
    """Names and words the recipe hash leaves out: the axis is bound by its hash (ID10)."""

    kind: Literal["AlphaProductRecipe"] = "AlphaProductRecipe"
    recipe_id: Literal["IW184_IMPLIED_PLUS_WITHIN_FIXED1260_QUARTERLY_SEED3_VINTAGE4"] = (
        PRODUCT_RECIPE_ID
    )

    feature_axis_id: Literal["IW184_IMPLIED_PLUS_WITHIN"] = PRODUCT_FEATURE_AXIS_ID
    feature_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_count: Literal[184] = PRODUCT_FEATURE_COUNT
    source_candidate_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    estimator_point: AlphaEstimatorParameterPoint

    training_window_sessions: Literal[1260] = PRODUCT_TRAINING_WINDOW_SESSIONS
    purge_sessions: Literal[1] = PRODUCT_PURGE_SESSIONS
    refit_quarter_start_months: tuple[int, ...] = PRODUCT_REFIT_QUARTER_START_MONTHS
    seeds: tuple[int, ...] = PRODUCT_SEEDS
    vintage_count: Literal[4] = PRODUCT_VINTAGE_COUNT
    vintage_weights: tuple[int, ...] = PRODUCT_VINTAGE_WEIGHTS
    live_model_count: Literal[12] = PRODUCT_LIVE_MODEL_COUNT

    seed_aggregation: Literal["EQUAL_MEAN_WITHIN_VINTAGE"] = "EQUAL_MEAN_WITHIN_VINTAGE"
    vintage_aggregation: Literal["DECLARED_WEIGHT_MEAN_ACROSS_VINTAGES"] = (
        "DECLARED_WEIGHT_MEAN_ACROSS_VINTAGES"
    )
    score_aggregation: Literal["WITHIN_SESSION_PERCENTILE_RANK"] = "WITHIN_SESSION_PERCENTILE_RANK"

    evidence_disposition: Literal["DEVELOPMENT_DIAGNOSTIC_CANDIDATE_NOT_INSTALLED"] = (
        PRODUCT_EVIDENCE_DISPOSITION
    )
    evidence_daily_parquet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_support_first: date = PRODUCT_EVIDENCE_SUPPORT_FIRST
    evidence_support_last: date = PRODUCT_EVIDENCE_SUPPORT_LAST
    evidence_support_session_count: Literal[529] = PRODUCT_EVIDENCE_SUPPORT_SESSION_COUNT

    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def installed(cls) -> Self:
        """The one frozen public recipe. There is no variant constructor."""
        values: dict[str, object] = {
            "kind": "AlphaProductRecipe",
            "recipe_id": PRODUCT_RECIPE_ID,
            "feature_axis_id": PRODUCT_FEATURE_AXIS_ID,
            "feature_axis_hash": PRODUCT_FEATURE_AXIS_HASH,
            "feature_count": PRODUCT_FEATURE_COUNT,
            "source_candidate_manifest_sha256": PRODUCT_SOURCE_CANDIDATE_MANIFEST_SHA256,
            "estimator_point": PRODUCT_ESTIMATOR_POINT.model_dump(mode="json"),
            "training_window_sessions": PRODUCT_TRAINING_WINDOW_SESSIONS,
            "purge_sessions": PRODUCT_PURGE_SESSIONS,
            "refit_quarter_start_months": list(PRODUCT_REFIT_QUARTER_START_MONTHS),
            "seeds": list(PRODUCT_SEEDS),
            "vintage_count": PRODUCT_VINTAGE_COUNT,
            "vintage_weights": list(PRODUCT_VINTAGE_WEIGHTS),
            "live_model_count": PRODUCT_LIVE_MODEL_COUNT,
            "seed_aggregation": "EQUAL_MEAN_WITHIN_VINTAGE",
            "vintage_aggregation": "DECLARED_WEIGHT_MEAN_ACROSS_VINTAGES",
            "score_aggregation": "WITHIN_SESSION_PERCENTILE_RANK",
            "evidence_disposition": PRODUCT_EVIDENCE_DISPOSITION,
            "evidence_daily_parquet_sha256": PRODUCT_EVIDENCE_DAILY_PARQUET_SHA256,
            "evidence_support_first": PRODUCT_EVIDENCE_SUPPORT_FIRST.isoformat(),
            "evidence_support_last": PRODUCT_EVIDENCE_SUPPORT_LAST.isoformat(),
            "evidence_support_session_count": PRODUCT_EVIDENCE_SUPPORT_SESSION_COUNT,
        }
        draft = cls.model_construct(**values, recipe_hash="0" * 64)
        return cls(**values, recipe_hash=recipe_identity(draft, exclude=_SEAL))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_recipe(self) -> Self:
        """Verify seed, vintage, calendar, support and canonical recipe consistency.

        Returns:
            This recipe after its declared model count and content hash agree.

        Raises:
            AlphaProductRecipeError: A declared axis, weight, support or recipe
                identity violates the installed consistency rules.
        """
        if len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise AlphaProductRecipeError("alpha_research.product_recipe_seeds_invalid")
        if len(self.vintage_weights) != self.vintage_count or any(
            weight <= 0 for weight in self.vintage_weights
        ):
            raise AlphaProductRecipeError("alpha_research.product_recipe_vintage_weights_invalid")
        if sorted(self.vintage_weights, reverse=True) != list(self.vintage_weights):
            raise AlphaProductRecipeError("alpha_research.product_recipe_vintage_order_invalid")
        if len(set(self.refit_quarter_start_months)) != 4 or any(
            month < 1 or month > 12 for month in self.refit_quarter_start_months
        ):
            raise AlphaProductRecipeError("alpha_research.product_recipe_refit_calendar_invalid")
        # The live model count is not an independent number. Stating it and
        # deriving it separately is how a recipe ends up describing eleven models
        # while claiming twelve.
        if self.live_model_count != len(self.seeds) * self.vintage_count:
            raise AlphaProductRecipeError("alpha_research.product_recipe_model_count_invalid")
        if self.evidence_support_last <= self.evidence_support_first:
            raise AlphaProductRecipeError("alpha_research.product_recipe_support_invalid")
        if not recipe_seal_holds(self, "recipe_hash"):
            raise AlphaProductRecipeError("alpha_research.product_recipe_identity_invalid")
        return self

    def resolve_estimator_parameters(self) -> tuple[DynamicPanelLightGBMParameters, ...]:
        """One parameter object per seed, in declared seed order.

        These are what a vintage fit consumes. The recipe hands over exactly the
        installed adapter's type, so nothing downstream needs to know that a
        product recipe was involved.
        """
        return tuple(self.estimator_point.resolve(seed=seed) for seed in self.seeds)

    def vintage_weight_shares(self) -> tuple[float, ...]:
        """The declared weights normalised, newest vintage first.

        Kept as a derivation rather than a stored field so that the ratio and
        the shares cannot disagree.
        """
        total = float(sum(self.vintage_weights))
        return tuple(weight / total for weight in self.vintage_weights)

    def is_refit_month(self, month: int) -> bool:
        """Check whether a month belongs to the declared quarterly refit calendar.

        Args:
            month: Calendar month number to test for membership.

        Returns:
            Whether the declared quarter-start months contain the supplied number.
        """
        return month in self.refit_quarter_start_months

    def researcher_payload(self) -> dict[str, object]:
        """Provenance for a report appendix, including the limitation."""
        return {
            "recipe_id": self.recipe_id,
            "feature_axis_id": self.feature_axis_id,
            "feature_axis_hash": self.feature_axis_hash,
            "feature_count": self.feature_count,
            "training_window_sessions": self.training_window_sessions,
            "purge_sessions": self.purge_sessions,
            "refit_quarter_start_months": list(self.refit_quarter_start_months),
            "seeds": list(self.seeds),
            "vintage_weights": list(self.vintage_weights),
            "live_model_count": self.live_model_count,
            "score_aggregation": self.score_aggregation,
            "evidence_disposition": self.evidence_disposition,
            "evidence_support": (
                f"{self.evidence_support_first.isoformat()} to "
                f"{self.evidence_support_last.isoformat()}"
            ),
            "recipe_hash": self.recipe_hash,
        }


INSTALLED_ALPHA_PRODUCT_RECIPE: Final = AlphaProductRecipe.installed()
"""The single public Alpha identity. Composition injects it; nothing discovers it."""

PRODUCT_RECIPE_ROLE: Final = "alpha_research.product_recipe"
"""The readout role of the installed recipe hash: a recorded hash is current through its moves."""


def installed_product_recipe_hash() -> str:
    """The recipe hash this build installs: its readout role's value (NM1)."""
    return INSTALLED_ALPHA_PRODUCT_RECIPE.recipe_hash


__all__ = [
    "INSTALLED_ALPHA_PRODUCT_RECIPE",
    "PRODUCT_ESTIMATOR_POINT",
    "PRODUCT_FEATURE_AXIS_HASH",
    "PRODUCT_FEATURE_AXIS_ID",
    "PRODUCT_LIVE_MODEL_COUNT",
    "PRODUCT_RECIPE_ID",
    "PRODUCT_RECIPE_ROLE",
    "PRODUCT_SEEDS",
    "PRODUCT_TRAINING_WINDOW_SESSIONS",
    "PRODUCT_VINTAGE_WEIGHTS",
    "AlphaEstimatorParameterPoint",
    "AlphaProductRecipe",
    "AlphaProductRecipeError",
]
