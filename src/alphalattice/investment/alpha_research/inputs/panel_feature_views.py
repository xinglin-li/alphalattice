"""Installed multi-role Panel feature views with fold-owned final scaling.

This module is the one small seam the existing Alpha input owners did not have:
composition of multiple, causally transformed Feature roles into a common model
matrix.  Formula keeps ownership of factor-specific pretransforms, Sector
Research keeps ownership of Sector state, and Alpha owns only temporal
transformation plus the fold boundary at which final model scale is learned.

The catalog is not Dynamic-specific.  Dynamic Panel is four installed view
recipes in it, and a lean recipe is a second real consumer.  Registration of a
new view changes the catalog inventory but does not rotate existing recipe
identity, because each recipe binds only its own role selections.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from types import MappingProxyType
from typing import Final, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.quant.sector_history import sector_ids, sector_slices
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int64]
type FeatureRoleId = Literal[
    "RELATIVE_STOCK_CROSS_SECTION",
    "NON_NEUTRAL_STOCK_CROSS_SECTION",
    "ABSOLUTE_STOCK_TIME_SERIES_STATE",
    "MATURED_STOCK_OUTCOME_HISTORY",
    "SECTOR_CONTEXT",
    "MARKET_CONTEXT",
    "SECTOR_CATEGORY",
]
type FeatureTransformId = Literal[
    "current",
    "lag1",
    "lag2",
    "lag3",
    "lag4",
    "lag5",
    "lag21",
    "lag63",
    "mean3",
    "mean5",
    "mean10",
    "mean21",
    "mean63",
    "max5",
    "max10",
    "min5",
    "min10",
    "std10",
    "delta1",
    "delta5",
    "delta21",
    "EWMA21",
    "EWMA63",
]
type InstalledPanelViewId = Literal[
    "RELATIVE_CONTROL",
    "DYNAMIC_RELATIVE",
    "DYNAMIC_CONTEXT",
    "DYNAMIC_JOINT_PRIMARY",
    "DYNAMIC_JOINT_LEAN",
    "SPARSE_SESSION_AMPLITUDE",
    "SPARSE_HISTORICAL_SEMANTIC_INTERSECTION",
    "DYNAMIC_JOINT_WITHOUT_RELATIVE_STOCK_CROSS_SECTION",
    "DYNAMIC_JOINT_WITHOUT_NON_NEUTRAL_STOCK_CROSS_SECTION",
    "DYNAMIC_JOINT_WITHOUT_ABSOLUTE_STOCK_TIME_SERIES_STATE",
    "DYNAMIC_JOINT_WITHOUT_MATURED_STOCK_OUTCOME_HISTORY",
    "DYNAMIC_JOINT_WITHOUT_SECTOR_CONTEXT",
    "DYNAMIC_JOINT_WITHOUT_MARKET_CONTEXT",
    "DYNAMIC_JOINT_WITHOUT_SECTOR_CATEGORY",
]
type HistoricalPanelEvidenceViewId = Literal[
    "SPARSE_SESSION_AMPLITUDE_RELATIVE_POSITIVE_IDENTITY",
    "SPARSE_SESSION_AMPLITUDE_NON_NEUTRAL_POSITIVE_IDENTITY",
    "SPARSE_SESSION_AMPLITUDE_STOCK_POSITIVE_IDENTITY",
]
type ScaleState = Literal["INVALID_NONFINITE_SCALE", "CONSTANT_TRAINING_FEATURE", "VALID_SCALE"]

_HASH = r"^[0-9a-f]{64}$"
_NEAR_CONSTANT_THRESHOLD: Final = 1e-8
INSTALLED_PANEL_VIEW_IDS: Final[tuple[InstalledPanelViewId, ...]] = (
    "RELATIVE_CONTROL",
    "DYNAMIC_RELATIVE",
    "DYNAMIC_CONTEXT",
    "DYNAMIC_JOINT_PRIMARY",
    "DYNAMIC_JOINT_LEAN",
    "SPARSE_SESSION_AMPLITUDE",
    "SPARSE_HISTORICAL_SEMANTIC_INTERSECTION",
    "DYNAMIC_JOINT_WITHOUT_RELATIVE_STOCK_CROSS_SECTION",
    "DYNAMIC_JOINT_WITHOUT_NON_NEUTRAL_STOCK_CROSS_SECTION",
    "DYNAMIC_JOINT_WITHOUT_ABSOLUTE_STOCK_TIME_SERIES_STATE",
    "DYNAMIC_JOINT_WITHOUT_MATURED_STOCK_OUTCOME_HISTORY",
    "DYNAMIC_JOINT_WITHOUT_SECTOR_CONTEXT",
    "DYNAMIC_JOINT_WITHOUT_MARKET_CONTEXT",
    "DYNAMIC_JOINT_WITHOUT_SECTOR_CATEGORY",
)
SPARSE_SESSION_AMPLITUDE_VIEW_IDS: Final[tuple[InstalledPanelViewId, ...]] = (
    "SPARSE_SESSION_AMPLITUDE",
    "SPARSE_HISTORICAL_SEMANTIC_INTERSECTION",
)
_ROLE_ORDER: Final[tuple[FeatureRoleId, ...]] = (
    "RELATIVE_STOCK_CROSS_SECTION",
    "NON_NEUTRAL_STOCK_CROSS_SECTION",
    "ABSOLUTE_STOCK_TIME_SERIES_STATE",
    "MATURED_STOCK_OUTCOME_HISTORY",
    "SECTOR_CONTEXT",
    "MARKET_CONTEXT",
    "SECTOR_CATEGORY",
)
SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS: Final[tuple[str, ...]] = (
    "at_own_high_share_63",
    "close_to_close",
    "gap",
    "gap_amplitude",
    # Preserve the fixed recipe's semantic slots across the three corrected
    # Formula names.  LightGBM feature sampling is positional, so lexically
    # regrouping renamed successors would change the installed recipe.
    "previous_close_excursion_intraday_adjusted_square",
    "high_extension",
    "intraday",
    "intraday_amplitude",
    "low_extension",
    "previous_close_excursion_scaled_span_square",
    "range_position",
    "return_run_length_21",
    "previous_close_excursion_intraday_cross",
    "session_dollar_volume",
    "session_span",
    "sessions_since_252_high",
)
SECTOR_CONTEXT_SOURCE_IDS: Final[tuple[str, ...]] = (
    "sector_equal_weight_raw_simple_return",
    "sector_trend_20",
    "sector_surprise_0",
    "sector_surprise_1",
    "sector_surprise_5",
)
BASE_MARKET_CONTEXT_SOURCE_IDS: Final[tuple[str, ...]] = (
    "market_raw_log_return",
    "market_volatility_21",
    "market_volatility_63",
    "market_drawdown_252",
    "breadth_positive_share",
    "cross_sectional_dispersion",
    "market_drawdown_x_momentum__state",
    "market_vol_ratio_x_reversal__state",
)


class PanelFeatureBoundaryError(ValueError):
    """Stable fail-closed error at the Alpha Feature-view boundary."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


def panel_array_hash(values: npt.NDArray[np.generic]) -> str:
    """Hash dtype, shape, finite mask and finite values without normalizing NaN."""
    array = np.ascontiguousarray(values)
    finite = np.isfinite(array)
    return str(
        canonical_hash(
            {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "finite_mask": sha256(finite.tobytes()).hexdigest(),
                "finite_values": sha256(
                    np.ascontiguousarray(np.where(finite, array, 0), dtype=array.dtype).tobytes()
                ).hexdigest(),
            }
        )
    )


def _seal[T: _Contract](model: type[T], values: dict[str, object], field: str) -> T:
    provisional = model.model_construct(**values, **{field: "0" * 64})
    return model(
        **values,
        **{field: str(canonical_hash(provisional.model_dump(mode="json", exclude={field})))},
    )


class FeatureTransformDescriptor(_Contract):
    """Seal a causal full-window transform with exact implementation and missingness policy.

    The declared boundary forbids forward filling and partial-window transforms. Lookback and
    implementation identities are part of descriptor_hash.
    """

    kind: Literal["FeatureTransformDescriptor"] = "FeatureTransformDescriptor"
    transform_id: FeatureTransformId
    lookback_sessions: int = Field(ge=0)
    full_window_required: Literal[True] = True
    causal: Literal[True] = True
    missingness_policy_id: Literal["NO_FORWARD_FILL_NO_PARTIAL_WINDOW"] = (
        "NO_FORWARD_FILL_NO_PARTIAL_WINDOW"
    )
    implementation_hash: str = Field(pattern=_HASH)
    descriptor_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the declared causal full-window feature-transform descriptor.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical descriptor_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return _seal(cls, dict(values), "descriptor_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the exact canonical feature-transform descriptor identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PanelFeatureBoundaryError: descriptor_hash differs from the complete descriptor payload.
        """
        if self.descriptor_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"descriptor_hash"})
        ):
            raise PanelFeatureBoundaryError("alpha_research.feature_transform_identity_invalid")
        return self


class FeatureRoleSelection(_Contract):
    """Seal one role source/unit/transform axis, maturity policy and final scaling method.

    Every source has a unit and a nonempty unique transform sequence. The supplied source order is
    retained as role authority.
    """

    kind: Literal["FeatureRoleSelection"] = "FeatureRoleSelection"
    role_id: FeatureRoleId
    source_ids: tuple[str, ...] = Field(min_length=1)
    source_transform_ids: tuple[tuple[FeatureTransformId, ...], ...] = Field(min_length=1)
    source_unit_ids: tuple[str, ...] = Field(min_length=1)
    maturity_policy_id: str
    final_scale_method_id: str
    selection_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared feature role and its source/transform/unit authority.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical selection_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return _seal(cls, dict(values), "selection_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require aligned role source/unit/transform axes and exact selection identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PanelFeatureBoundaryError: Source-axis lengths differ, a transform sequence is
                empty/duplicated or selection_hash is inconsistent.
        """
        if (
            len(self.source_ids) != len(self.source_unit_ids)
            or len(self.source_ids) != len(self.source_transform_ids)
            or any(
                not values or values != tuple(dict.fromkeys(values))
                for values in self.source_transform_ids
            )
            or self.selection_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"selection_hash"}))
        ):
            raise PanelFeatureBoundaryError("alpha_research.feature_role_selection_invalid")
        return self


class PanelFeatureViewRecipe(_Contract):
    """Seal an installed ordered feature-role view with shared normalized model inputs.

    All models consume the same role-normalized matrix; adapter-private preprocessing is forbidden
    by the declared recipe.
    """

    kind: Literal["PanelFeatureViewRecipe"] = "PanelFeatureViewRecipe"
    method_id: InstalledPanelViewId
    roles: tuple[FeatureRoleSelection, ...] = Field(min_length=1)
    matrix_policy_id: Literal["SAME_ROLE_NORMALIZED_MATRIX_ALL_MODELS"] = (
        "SAME_ROLE_NORMALIZED_MATRIX_ALL_MODELS"
    )
    adapter_preprocessing_policy_id: Literal["PRIVATE_PREPROCESSING_FORBIDDEN"] = (
        "PRIVATE_PREPROCESSING_FORBIDDEN"
    )
    recipe_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal an installed ordered Panel feature-view recipe.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical recipe_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return _seal(cls, dict(values), "recipe_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require unique roles in the admitted role order and exact feature-view identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PanelFeatureBoundaryError: The role axis repeats or differs from admitted order, or
                recipe_hash is inconsistent.
        """
        role_ids = tuple(value.role_id for value in self.roles)
        if (
            role_ids != tuple(value for value in _ROLE_ORDER if value in set(role_ids))
            or role_ids != tuple(dict.fromkeys(role_ids))
            or self.recipe_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"recipe_hash"}))
        ):
            raise PanelFeatureBoundaryError("alpha_research.feature_view_recipe_invalid")
        return self


class PanelFeatureViewCatalog:
    """Explicit Alpha-owned view inventory; it owns no Formula or scaler numerics."""

    def __init__(
        self,
        *,
        transforms: tuple[FeatureTransformDescriptor, ...],
        views: tuple[PanelFeatureViewRecipe, ...],
    ) -> None:
        """Bind unique installed transforms and views with exact transform-consumer coverage.

        Args:
            transforms: Installed unique transform descriptors.
            views: Installed unique view recipes whose selected transforms must equal the transform
                inventory.

        Raises:
            PanelFeatureBoundaryError: Transform/view identifiers repeat or selected transform
                consumers differ from the inventory.
        """
        transform_index = {value.transform_id: value for value in transforms}
        view_index = {value.method_id: value for value in views}
        if len(transform_index) != len(transforms) or len(view_index) != len(views):
            raise PanelFeatureBoundaryError("alpha_research.feature_view_catalog_duplicate")
        selected = {
            transform
            for view in views
            for role in view.roles
            for source_transforms in role.source_transform_ids
            for transform in source_transforms
        }
        if selected != set(transform_index):
            raise PanelFeatureBoundaryError("alpha_research.feature_transform_consumer_mismatch")
        self._transforms = MappingProxyType(transform_index)
        self._views = MappingProxyType(view_index)
        self.catalog_hash = str(
            canonical_hash(
                {
                    "transform_descriptor_hashes": [
                        self._transforms[key].descriptor_hash for key in sorted(self._transforms)
                    ],
                    "installed_method_ids": list(sorted(self._views)),
                    "recipe_hashes": [self._views[key].recipe_hash for key in sorted(self._views)],
                }
            )
        )

    @property
    def method_ids(self) -> tuple[str, ...]:
        """Read installed feature-view method identifiers in sorted order.

        Returns:
            Sorted installed method identifiers.
        """
        return tuple(sorted(self._views))

    @property
    def descriptors(self) -> tuple[PanelFeatureViewRecipe, ...]:
        """Read installed feature-view recipes in sorted method order.

        Returns:
            Installed recipe descriptors ordered by method identifier.
        """
        return tuple(self._views[key] for key in sorted(self._views))

    def resolve(self, method_id: str) -> PanelFeatureViewRecipe:
        """Resolve one exact installed feature-view method.

        Args:
            method_id: Installed Panel view identifier to resolve.

        Returns:
            Installed feature-view recipe.

        Raises:
            PanelFeatureBoundaryError: The requested view method is not installed.
        """
        try:
            return self._views[cast(InstalledPanelViewId, method_id)]
        except KeyError as error:
            raise PanelFeatureBoundaryError(
                "alpha_research.feature_view_method_not_installed"
            ) from error

    def transform(self, transform_id: FeatureTransformId) -> FeatureTransformDescriptor:
        """Resolve one exact installed causal transform descriptor.

        Args:
            transform_id: Installed transform identifier.

        Returns:
            Transform descriptor from the immutable catalog.

        Raises:
            KeyError: The transform identifier is absent.
        """
        return self._transforms[transform_id]


def _transform_lookback(transform_id: FeatureTransformId) -> int:
    if transform_id == "current":
        return 0
    digits = "".join(value for value in transform_id if value.isdigit())
    return int(digits)


def _transform_descriptor(transform_id: FeatureTransformId) -> FeatureTransformDescriptor:
    return FeatureTransformDescriptor.create(
        transform_id=transform_id,
        lookback_sessions=_transform_lookback(transform_id),
        implementation_hash=str(
            canonical_hash(
                {
                    "owner": "alpha_research.inputs.panel_feature_views",
                    "transform_id": transform_id,
                    "lag_semantics": "ACCEPTED_TRADING_SESSION",
                    "mean_semantics": "CURRENT_INCLUSIVE_FULL_WINDOW",
                    "delta_semantics": "CURRENT_MINUS_LAG",
                    "ewma_semantics": "ADJUST_FALSE_MIN_PERIODS_SPAN_CONSECUTIVE_FINITE",
                }
            )
        ),
    )


def _role(
    role_id: FeatureRoleId,
    *,
    source_ids: tuple[str, ...],
    source_unit_ids: tuple[str, ...],
    transforms: tuple[FeatureTransformId, ...] | tuple[tuple[FeatureTransformId, ...], ...],
    maturity: str,
    scale: str,
) -> FeatureRoleSelection:
    return FeatureRoleSelection.create(
        role_id=role_id,
        source_ids=source_ids,
        source_unit_ids=source_unit_ids,
        source_transform_ids=(
            cast(tuple[tuple[FeatureTransformId, ...], ...], transforms)
            if transforms and isinstance(transforms[0], tuple)
            else (cast(tuple[FeatureTransformId, ...], transforms),) * len(source_ids)
        ),
        maturity_policy_id=maturity,
        final_scale_method_id=scale,
    )


def build_installed_panel_feature_view_catalog(
    *,
    factor_ids: tuple[str, ...],
    absolute_state_factor_ids: tuple[str, ...],
    sector_ids: tuple[str, ...],
) -> PanelFeatureViewCatalog:
    """Install exactly the primary/lean consumed view and transform methods."""
    if (
        not factor_ids
        or factor_ids != tuple(dict.fromkeys(factor_ids))
        or not absolute_state_factor_ids
        or not set(absolute_state_factor_ids).issubset(factor_ids)
        or sector_ids != tuple(sorted(set(sector_ids)))
    ):
        raise PanelFeatureBoundaryError("alpha_research.feature_view_source_axis_invalid")
    relative_transforms: tuple[FeatureTransformId, ...] = (
        "current",
        "lag1",
        "lag5",
        "mean21",
        "delta1",
        "EWMA21",
    )
    non_neutral_transforms: tuple[FeatureTransformId, ...] = ("current", "lag1", "delta1")
    absolute_transforms: tuple[FeatureTransformId, ...] = (
        "current",
        "lag21",
        "lag63",
        "mean63",
        "delta21",
        "EWMA63",
    )
    outcome_transforms: tuple[FeatureTransformId, ...] = (
        "lag1",
        "lag5",
        "lag21",
        "lag63",
        "mean5",
        "mean21",
        "mean63",
        "delta1",
        "delta5",
        "delta21",
        "EWMA21",
        "EWMA63",
    )
    raw_context_transforms: tuple[FeatureTransformId, ...] = (
        "current",
        *outcome_transforms,
    )
    state_context_transforms: tuple[FeatureTransformId, ...] = (
        "current",
        "lag1",
        "lag5",
        "lag21",
        "mean21",
        "delta1",
        "delta5",
        "EWMA21",
    )
    # SPARSE_SESSION_AMPLITUDE -- the assembled design, named so it can never be
    # read as one of the DYNAMIC_JOINT views.
    #
    # Two properties distinguish it, and the name states both. It is **sparse**:
    # one transform per published Formula, where the joint views broadcast six,
    # because the measured failure mode of this Panel is redundancy rather than
    # capacity. And it is **session-amplitude led**: the blocks that earned their
    # place read the size of one session's move -- the range estimators, the
    # overnight/intraday split, the extremes -- at three to ten sessions, windows
    # the joint views do not carry at all.
    #
    # Every window below is a view transform over a single-session Formula. The
    # Formula owns the economic fact; how many sessions of it a design reads is
    # this layer's decision, which is why installing the design needed no
    # per-window Formula.
    _AMPLITUDE_WINDOWS: tuple[FeatureTransformId, ...] = ("mean3", "mean5", "mean10")
    # Two neutralizations, because the measurement says two. Return, price and
    # amplitude columns want universe centring -- their Sector level is the view
    # this design keeps. Liquidity and the published Factor axis want the Sector
    # demean, where the Sector level is structural rather than a view. Sending
    # the amplitude blocks through the demean is the wrong half of that rule, and
    # it is the first thing this design got wrong.
    universe_centred_transforms: dict[str, tuple[FeatureTransformId, ...]] = {
        "previous_close_excursion_scaled_span_square": _AMPLITUDE_WINDOWS,
        "previous_close_excursion_intraday_cross": _AMPLITUDE_WINDOWS,
        "previous_close_excursion_intraday_adjusted_square": _AMPLITUDE_WINDOWS,
        "gap_amplitude": _AMPLITUDE_WINDOWS,
        "intraday_amplitude": _AMPLITUDE_WINDOWS,
        "session_span": (*_AMPLITUDE_WINDOWS, "std10"),
        # Rolling extremes, not the five session lags. Both exist in the source
        # design and the assembled 151 keeps the extremes: the lagged form sits
        # in the standing base and is excluded from the design's stem, which is
        # the one distinction the block table records and the column ranges
        # confirm. The fifth column of that block is a drawdown from a 21-session
        # high, which is not an installed Formula.
        "close_to_close": ("max5", "min5", "max10", "min10"),
        "range_position": ("current", "mean5", "mean21"),
        "gap": ("current",),
        "intraday": ("current",),
        "high_extension": ("current",),
        "low_extension": ("current",),
        # Persistence. Universe centred, because how long a listing's direction
        # has run and how recently it was at its own high are facts about the
        # listing rather than about its Sector's level.
        "return_run_length_21": ("current",),
        "at_own_high_share_63": ("current",),
        "sessions_since_252_high": ("current",),
    }
    amplitude_ids = tuple(
        value
        for value in SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS
        if value in universe_centred_transforms
    )
    extension_ids = set(SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS)
    control_factor_ids = tuple(value for value in factor_ids if value not in extension_ids)
    sparse_relative_factor_ids = tuple(
        value for value in factor_ids if value not in set(amplitude_ids)
    )
    sparse_relative = _role(
        "RELATIVE_STOCK_CROSS_SECTION",
        source_ids=sparse_relative_factor_ids,
        source_unit_ids=("FORMULA_DECLARED",) * len(sparse_relative_factor_ids),
        transforms=("current",),
        maturity="FORMATION_AVAILABLE_FORMULA_SOURCE",
        scale="SESSION_MAD_3_5_SECTOR_DEMEAN_ONCE_SAMPLE_STD_Z",
    )
    sparse_amplitude = _role(
        "NON_NEUTRAL_STOCK_CROSS_SECTION",
        source_ids=amplitude_ids,
        source_unit_ids=("FORMULA_DECLARED",) * len(amplitude_ids),
        transforms=tuple(universe_centred_transforms[value] for value in amplitude_ids),
        maturity="FORMATION_AVAILABLE_FORMULA_SOURCE",
        scale="SESSION_MAD_3_5_UNIVERSE_CENTER_ONCE_SAMPLE_STD_Z",
    )
    # Short consecutive lags. A per-session count -- breadth, dispersion -- is a
    # fresh observation each session rather than a slow-moving level, so what it
    # carries is the recent sequence and not one point of it. The installed
    # ladder jumps 1 to 5 and cannot express that.
    # Two market speeds, and each source gets the one that suits it.
    #
    # A matured execution aggregate is already three sessions behind formation,
    # so the recent sequence of it carries little the previous value does not;
    # what it carries is the regime, which is the twenty-one and sixty-three
    # session read. An observation aggregate is complete at `close(t)`, so its
    # recent sequence is exactly the fresh state a design wants, and the block
    # built that way was the best measured in the programme. Giving every source
    # both speeds is what took the market axis from forty-seven columns to a
    # hundred and twelve without the evidence for it.
    sparse_execution_market_transforms: tuple[FeatureTransformId, ...] = (
        "lag1",
        "lag5",
        "mean21",
        "mean63",
    )
    sparse_market_transforms: tuple[FeatureTransformId, ...] = (
        "current",
        "lag1",
        "lag2",
        "lag3",
        "lag4",
    )
    sparse_sector_transforms: tuple[FeatureTransformId, ...] = (
        "lag1",
        "lag5",
        "mean21",
        "mean63",
    )
    relative_current = _role(
        "RELATIVE_STOCK_CROSS_SECTION",
        source_ids=control_factor_ids,
        source_unit_ids=("FORMULA_DECLARED",) * len(control_factor_ids),
        transforms=("current",),
        maturity="FORMATION_AVAILABLE_FORMULA_SOURCE",
        scale="SESSION_MAD_3_5_SECTOR_DEMEAN_ONCE_SAMPLE_STD_Z",
    )
    relative_dynamic = relative_current.model_copy(
        update={
            "source_transform_ids": (relative_transforms,) * len(control_factor_ids),
            "selection_hash": "0" * 64,
        }
    )
    relative_dynamic = FeatureRoleSelection.create(
        **relative_dynamic.model_dump(mode="python", exclude={"selection_hash"})
    )
    non_neutral = _role(
        "NON_NEUTRAL_STOCK_CROSS_SECTION",
        source_ids=factor_ids,
        source_unit_ids=("FORMULA_DECLARED",) * len(factor_ids),
        transforms=non_neutral_transforms,
        maturity="FORMATION_AVAILABLE_FORMULA_SOURCE",
        scale="SESSION_MAD_3_5_UNIVERSE_CENTER_ONCE_SAMPLE_STD_Z",
    )
    absolute = _role(
        "ABSOLUTE_STOCK_TIME_SERIES_STATE",
        source_ids=absolute_state_factor_ids,
        source_unit_ids=("FORMULA_DECLARED",) * len(absolute_state_factor_ids),
        transforms=absolute_transforms,
        maturity="FORMATION_AVAILABLE_FORMULA_SOURCE",
        scale="TRAINING_ROWS_MEDIAN_NORMALIZED_MAD_CLIP_3_5_SCALE",
    )
    outcomes = _role(
        "MATURED_STOCK_OUTCOME_HISTORY",
        source_ids=(
            "total_return_target_z_history",
            "raw_log_execution_return_history",
            "sector_residual_log_return_history",
        ),
        source_unit_ids=("DIMENSIONLESS", "LOG_RETURN", "LOG_RETURN"),
        transforms=outcome_transforms,
        maturity="HOLDING_END_STRICTLY_BEFORE_FORMATION",
        scale="LANE_BOUND_SESSION_FINAL_SCALE",
    )
    sector_source_ids = SECTOR_CONTEXT_SOURCE_IDS
    sector_unit_ids: tuple[str, ...] = ("DECIMAL_SIMPLE_RETURN",) + ("FORMULA_DECLARED",) * 4
    sector = _role(
        "SECTOR_CONTEXT",
        source_ids=sector_source_ids,
        source_unit_ids=sector_unit_ids,
        transforms=(
            raw_context_transforms,
            *([state_context_transforms] * (len(sector_source_ids) - 1)),
        ),
        maturity="SOURCE_DESCRIPTOR_CAUSAL_FORMATION",
        scale="PER_SESSION_LISTING_MEAN_POPULATION_STD_Z",
    )
    market_source_ids = BASE_MARKET_CONTEXT_SOURCE_IDS
    market_unit_ids: tuple[str, ...] = (
        "LOG_RETURN",
        "LOG_RETURN_VOLATILITY",
        "LOG_RETURN_VOLATILITY",
        "LOG_RETURN_DRAWDOWN",
        "SHARE",
        "LOG_RETURN_DISPERSION",
        "FORMULA_DECLARED",
        "FORMULA_DECLARED",
    )
    # A Panel carrying the close-to-close observation also carries the two market
    # aggregates of it, which are available at the formation session rather than
    # three sessions behind it.
    observation_market = "close_to_close" in set(factor_ids)
    if observation_market:
        market_source_ids = (*market_source_ids, *OBSERVED_MARKET_SOURCE_IDS)
        market_unit_ids = (
            *market_unit_ids,
            *(OBSERVED_MARKET_UNIT_IDS[value] for value in OBSERVED_MARKET_SOURCE_IDS),
        )
    market = _role(
        "MARKET_CONTEXT",
        source_ids=market_source_ids,
        source_unit_ids=market_unit_ids,
        transforms=(
            raw_context_transforms,
            *([state_context_transforms] * (len(market_source_ids) - 1)),
        ),
        maturity="SOURCE_DESCRIPTOR_CAUSAL_FORMATION",
        scale="TRAINING_NATIVE_SESSION_MEDIAN_NORMALIZED_MAD",
    )
    category = _role(
        "SECTOR_CATEGORY",
        source_ids=sector_ids,
        source_unit_ids=("CATEGORICAL_0_1",) * len(sector_ids),
        transforms=("current",),
        maturity="CURRENT_QUALIFIED_FIXED_DEVELOPMENT_WORLD_NOT_PIT",
        scale="DECLARED_CATEGORICAL_ONE_HOT_NO_SCALE",
    )
    joint_roles = (relative_dynamic, non_neutral, absolute, outcomes, sector, market, category)
    sparse_context_roles = (
        _role(
            "SECTOR_CONTEXT",
            source_ids=sector_source_ids,
            source_unit_ids=sector_unit_ids,
            transforms=sparse_sector_transforms,
            maturity="SOURCE_DESCRIPTOR_CAUSAL_FORMATION",
            scale="PER_SESSION_LISTING_MEAN_POPULATION_STD_Z",
        ),
        _role(
            "MARKET_CONTEXT",
            source_ids=market_source_ids,
            source_unit_ids=market_unit_ids,
            transforms=tuple(
                sparse_market_transforms
                if value in OBSERVED_MARKET_SOURCE_IDS
                else sparse_execution_market_transforms
                for value in market_source_ids
            ),
            maturity="SOURCE_DESCRIPTOR_CAUSAL_FORMATION",
            scale="TRAINING_NATIVE_SESSION_MEDIAN_NORMALIZED_MAD",
        ),
        category,
    )
    sparse_roles = (sparse_relative, sparse_amplitude, *sparse_context_roles)
    # Literal ordered subsequence of SPARSE_SESSION_AMPLITUDE containing only
    # current columns with a named semantic counterpart in the historical 151
    # recipe.  This is deliberately not padded to 151: fifteen approximate stock
    # columns and two Sector-trend proxies formerly used to fill absent slots are
    # excluded.  Current Formula values and current scaling remain authoritative,
    # including the installed non-log ``session_dollar_volume`` definition. The
    # historical Markdown's logged-liquidity description is not calculation
    # authority. This view is an intersection audit, not a reconstruction of
    # that document's preprocessing.
    historical_intersection_non_neutral_transforms: dict[str, tuple[FeatureTransformId, ...]] = {
        "at_own_high_share_63": ("current",),
        "close_to_close": ("max5", "min5", "max10", "min10"),
        "gap": ("current",),
        "previous_close_excursion_intraday_adjusted_square": _AMPLITUDE_WINDOWS,
        "intraday": ("current",),
        "previous_close_excursion_scaled_span_square": _AMPLITUDE_WINDOWS,
        "return_run_length_21": ("current",),
        "previous_close_excursion_intraday_cross": _AMPLITUDE_WINDOWS,
        "sessions_since_252_high": ("current",),
    }
    historical_intersection_non_neutral_ids = tuple(
        value for value in amplitude_ids if value in historical_intersection_non_neutral_transforms
    )
    historical_intersection_non_neutral = _role(
        "NON_NEUTRAL_STOCK_CROSS_SECTION",
        source_ids=historical_intersection_non_neutral_ids,
        source_unit_ids=("FORMULA_DECLARED",) * len(historical_intersection_non_neutral_ids),
        transforms=tuple(
            historical_intersection_non_neutral_transforms[value]
            for value in historical_intersection_non_neutral_ids
        ),
        maturity="FORMATION_AVAILABLE_FORMULA_SOURCE",
        scale="SESSION_MAD_3_5_UNIVERSE_CENTER_ONCE_SAMPLE_STD_Z",
    )
    historical_intersection_sector = _role(
        "SECTOR_CONTEXT",
        source_ids=("sector_equal_weight_raw_simple_return",),
        source_unit_ids=("DECIMAL_SIMPLE_RETURN",),
        transforms=(sparse_sector_transforms,),
        maturity="SOURCE_DESCRIPTOR_CAUSAL_FORMATION",
        scale="PER_SESSION_LISTING_MEAN_POPULATION_STD_Z",
    )
    historical_intersection_market_transforms: dict[str, tuple[FeatureTransformId, ...]] = {
        "market_raw_log_return": ("lag1", "lag5"),
        "market_volatility_21": ("lag1",),
        "market_volatility_63": ("lag1",),
        "market_drawdown_252": ("mean21", "mean63"),
        "breadth_positive_share": ("mean21", "mean63"),
        "cross_sectional_dispersion": ("mean21", "mean63"),
        "market_drawdown_x_momentum__state": ("lag1",),
        "market_vol_ratio_x_reversal__state": ("lag1",),
        "observed_breadth_positive_share": sparse_market_transforms,
        "observed_cross_sectional_dispersion": sparse_market_transforms,
        "observed_implied_correlation": sparse_market_transforms,
        "observed_within_sector_dispersion": ("current",),
        "observed_between_sector_dispersion": ("current",),
        "observed_rotation_ratio": ("current", "lag1", "lag2"),
        "observed_volume_regime": sparse_market_transforms,
        "observed_new_high_low_share": sparse_market_transforms,
        "observed_up_volume_share": sparse_market_transforms,
    }
    historical_intersection_market = _role(
        "MARKET_CONTEXT",
        source_ids=market_source_ids,
        source_unit_ids=market_unit_ids,
        transforms=tuple(
            historical_intersection_market_transforms[value] for value in market_source_ids
        ),
        maturity="SOURCE_DESCRIPTOR_CAUSAL_FORMATION",
        scale="TRAINING_NATIVE_SESSION_MEDIAN_NORMALIZED_MAD",
    )
    historical_intersection_roles = (
        sparse_relative,
        historical_intersection_non_neutral,
        historical_intersection_sector,
        historical_intersection_market,
        category,
    )
    views = (
        PanelFeatureViewRecipe.create(method_id="RELATIVE_CONTROL", roles=(relative_current,)),
        PanelFeatureViewRecipe.create(method_id="DYNAMIC_RELATIVE", roles=(relative_dynamic,)),
        PanelFeatureViewRecipe.create(
            method_id="DYNAMIC_CONTEXT",
            roles=(relative_current, outcomes, sector, market, category),
        ),
        PanelFeatureViewRecipe.create(
            method_id="DYNAMIC_JOINT_PRIMARY",
            roles=joint_roles,
        ),
        PanelFeatureViewRecipe.create(
            method_id="DYNAMIC_JOINT_LEAN",
            roles=(relative_dynamic, outcomes, sector, market, category),
        ),
        PanelFeatureViewRecipe.create(
            method_id="SPARSE_SESSION_AMPLITUDE",
            roles=sparse_roles,
        ),
        PanelFeatureViewRecipe.create(
            method_id="SPARSE_HISTORICAL_SEMANTIC_INTERSECTION",
            roles=historical_intersection_roles,
        ),
        *(
            PanelFeatureViewRecipe.create(
                method_id=cast(
                    InstalledPanelViewId,
                    f"DYNAMIC_JOINT_WITHOUT_{removed.role_id}",
                ),
                roles=tuple(role for role in joint_roles if role.role_id != removed.role_id),
            )
            for removed in joint_roles
        ),
    )
    # Derived from the views rather than transcribed beside them. The catalog
    # refuses a transform no view consumes -- the right invariant -- and a
    # hand-kept list of what is consumed had to be edited in lockstep with every
    # view change, which is a synchronisation the code can do itself.
    consumed = tuple(
        sorted(
            {
                transform
                for view in views
                for role in view.roles
                for source_transforms in role.source_transform_ids
                for transform in source_transforms
            }
        )
    )
    if tuple(value.method_id for value in views) != INSTALLED_PANEL_VIEW_IDS:
        raise PanelFeatureBoundaryError("alpha_research.feature_view_installation_drift")
    return PanelFeatureViewCatalog(
        transforms=tuple(_transform_descriptor(value) for value in consumed), views=views
    )


def _readonly(values: npt.NDArray[np.generic], *, dtype: np.dtype[np.generic]) -> npt.NDArray:
    output = np.ascontiguousarray(values, dtype=dtype)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class PanelFeatureSourceArrays:
    """Host-resolved immutable source lanes; callers never supply these arrays."""

    formation_sessions: tuple[date, ...]
    holding_end_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    ordered_factor_ids: tuple[str, ...]
    absolute_state_factor_ids: tuple[str, ...]
    ordered_sector_ids: tuple[str, ...]
    sector_by_listing_id: Mapping[str, str]
    """Each listing's Sector: a map every session reads, or the Panel's Sector history, whose
    sessions each read the map in force there (V346)."""
    raw_formula_values: FloatArray
    total_return_target_z: FloatArray
    raw_log_execution_returns: FloatArray
    raw_simple_execution_returns: FloatArray
    sector_context_values: FloatArray
    market_context_values: FloatArray
    source_identity_hashes: MappingProxyType[str, str]
    reference_eligible: BoolArray | None = None

    def __post_init__(self) -> None:
        """Require canonical source axes, causal holding-end clocks and non-writeable tensors.

        Raises:
            PanelFeatureBoundaryError: Source/classification/feature/sector axes, tensor/context
                shapes, identity lengths, clocks or optional reference eligibility violate the
                boundary.
        """
        sessions = len(self.formation_sessions)
        listings = len(self.ordered_listing_ids)
        factors = len(self.ordered_factor_ids)
        sectors = len(self.ordered_sector_ids)
        if (
            self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or len(self.holding_end_sessions) != sessions
            or self.ordered_listing_ids != tuple(sorted(set(self.ordered_listing_ids)))
            or self.ordered_factor_ids != tuple(dict.fromkeys(self.ordered_factor_ids))
            or not set(self.absolute_state_factor_ids).issubset(self.ordered_factor_ids)
            or self.ordered_sector_ids != tuple(sorted(set(self.ordered_sector_ids)))
            or set(self.sector_by_listing_id) != set(self.ordered_listing_ids)
            or set(sector_ids(self.sector_by_listing_id)) - set(self.ordered_sector_ids)
            or self.raw_formula_values.shape != (sessions, listings, factors)
            or self.total_return_target_z.shape != (sessions, listings)
            or self.raw_log_execution_returns.shape != (sessions, listings)
            or self.raw_simple_execution_returns.shape != (sessions, listings)
            or self.sector_context_values.shape != (sessions, sectors, 5)
            # Eight execution-derived market columns, and two more when the Panel
            # carries the close-to-close observation those aggregate.
            or self.market_context_values.shape
            not in {(sessions, 8), (sessions, 8 + len(OBSERVED_MARKET_SOURCE_IDS))}
            or any(len(value) != 64 for value in self.source_identity_hashes.values())
        ):
            raise PanelFeatureBoundaryError("alpha_research.panel_feature_source_axis_invalid")
        if any(
            holding_end <= formation
            for formation, holding_end in zip(
                self.formation_sessions, self.holding_end_sessions, strict=True
            )
        ):
            raise PanelFeatureBoundaryError("alpha_research.panel_feature_outcome_clock_invalid")
        for value in (
            self.raw_formula_values,
            self.total_return_target_z,
            self.raw_log_execution_returns,
            self.raw_simple_execution_returns,
            self.sector_context_values,
            self.market_context_values,
        ):
            if value.flags.writeable:
                raise PanelFeatureBoundaryError("alpha_research.panel_feature_source_mutable")
        if self.reference_eligible is not None and (
            self.reference_eligible.shape != (sessions, listings)
            or self.reference_eligible.dtype != np.bool_
            or self.reference_eligible.flags.writeable
        ):
            raise PanelFeatureBoundaryError("alpha_research.panel_feature_reference_invalid")


OBSERVED_MARKET_SOURCE_IDS: Final[tuple[str, ...]] = (
    "observed_breadth_positive_share",
    "observed_cross_sectional_dispersion",
    "observed_implied_correlation",
    "observed_within_sector_dispersion",
    "observed_between_sector_dispersion",
    "observed_rotation_ratio",
    "observed_volume_regime",
    "observed_new_high_low_share",
    "observed_up_volume_share",
)

FORMATION_OBSERVATION_DOLLAR_VOLUME_SOURCE_LANE_ID: Final = "FORMATION_OBSERVATION_DOLLAR_VOLUME"

OBSERVED_MARKET_UNIT_IDS: Final[dict[str, str]] = {
    "observed_breadth_positive_share": "SHARE",
    "observed_cross_sectional_dispersion": "LOG_RETURN_DISPERSION",
    "observed_implied_correlation": "CORRELATION",
    "observed_within_sector_dispersion": "LOG_RETURN_DISPERSION",
    "observed_between_sector_dispersion": "LOG_RETURN_DISPERSION",
    "observed_rotation_ratio": "DIMENSIONLESS",
    "observed_volume_regime": "DIMENSIONLESS",
    "observed_new_high_low_share": "SHARE",
    "observed_up_volume_share": "SHARE",
}
"""The unit each observed market source carries.

Keyed by source rather than positional, because the two lists drifted the moment
a ninth source was added and the mismatch surfaced as a role-identity refusal
several minutes into a run rather than at the declaration.
"""
"""Market aggregates of the observation panel, available at the formation session.

Every other market column aggregates matured execution outcomes and is therefore
read about three sessions behind formation. These read Formulas complete at
`close(t)`, which is when formation happens, so they carry the freshest market
state a causal design can hold -- and short lags of that state were the single
best-measured block in the programme that produced this design.
"""

_NEW_EXTREME_BAND: Final = 0.001
"""How close to its own 252-session extreme a listing counts as at it."""


def _trailing_std(values: FloatArray, window: int) -> FloatArray:
    """Trailing sample standard deviation ending at each row, all-finite windows."""

    source = np.asarray(values, dtype=np.float64)
    rows, columns = source.shape
    result = np.full((rows, columns), np.nan, dtype=np.float64)
    for position in range(window - 1, rows):
        block = source[position - window + 1 : position + 1]
        finite = np.isfinite(block).all(axis=0)
        if not finite.any():
            continue
        result[position, finite] = np.std(block[:, finite], axis=0, ddof=1)
    return cast(FloatArray, result)


def assemble_panel_context_arrays(
    *,
    formation_sessions: tuple[date, ...],
    holding_end_sessions: tuple[date | None, ...],
    ordered_listing_ids: tuple[str, ...],
    ordered_sector_ids: tuple[str, ...],
    sector_by_listing_id: Mapping[str, str],
    raw_log_execution_returns: FloatArray,
    raw_simple_execution_returns: FloatArray,
    sector_state_values: FloatArray,
    market_interaction_state_values: FloatArray,
    observation_returns: FloatArray | None = None,
    observation_volume_state: FloatArray | None = None,
    observation_dollar_volume: FloatArray | None = None,
    observation_high_distance: FloatArray | None = None,
    observation_low_distance: FloatArray | None = None,
    reference_eligible: BoolArray | None = None,
    selected_sector_source_ids: tuple[str, ...] | None = None,
    selected_market_source_ids: tuple[str, ...] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Build the declared causal Sector and Market source lanes.

    Return-derived context ends at the latest observation whose holding clock
    is strictly before formation.  Sector states and Formula-owned market
    interaction states are already formation-available published surfaces and
    remain distinct lanes.  This owner performs no final scaling; each derived
    column is scaled later at its consuming fold boundary.

    Args:
        formation_sessions: Ordered formations on the complete source history.
        holding_end_sessions: Each outcome's maturity, absent while it is unknown.
        ordered_listing_ids: Source and reference listing axis.
        ordered_sector_ids: Sector state and output classification axis.
        sector_by_listing_id: The classification each source session reads.
        raw_log_execution_returns: Source outcome log returns on the declared axes.
        raw_simple_execution_returns: Source outcome simple returns on those axes.
        sector_state_values: Four already admitted causal Sector state lanes.
        market_interaction_state_values: Two formation-available interaction lanes.
        observation_returns: Optional formation-available close-to-close observations.
        observation_volume_state: Optional Formula-owned volume deviations.
        observation_dollar_volume: Optional formation-observed traded values.
        observation_high_distance: Optional distances from each listing's own high.
        observation_low_distance: Optional distances from each listing's own low.
        reference_eligible: Optional dated reference population; absent means all members.
        selected_sector_source_ids: Explicit ordered named output lanes, or the full axis.
        selected_market_source_ids: Explicit ordered named output lanes, or the full axis
            available from the supplied observations. An empty tuple selects no lanes.

    Returns:
        Read-only float64 Sector and Market arrays in the requested source-axis order.
        With both selections absent, these retain the original complete output axes.

    Raises:
        PanelFeatureBoundaryError: Source/reference axes are invalid, a selected source is
            unknown or repeated, or a provided observation array in selected mode has the
            wrong source shape. Default full mode retains its original input admission.
    """
    sessions = len(formation_sessions)
    listings = len(ordered_listing_ids)
    sectors = len(ordered_sector_ids)
    if (
        len(holding_end_sessions) != sessions
        or raw_log_execution_returns.shape != (sessions, listings)
        or raw_simple_execution_returns.shape != (sessions, listings)
        or sector_state_values.shape != (sessions, sectors, 4)
        or market_interaction_state_values.shape != (sessions, 2)
    ):
        raise PanelFeatureBoundaryError("alpha_research.panel_context_source_axis_invalid")
    if reference_eligible is not None and (
        reference_eligible.shape != (sessions, listings) or reference_eligible.dtype != np.bool_
    ):
        raise PanelFeatureBoundaryError("alpha_research.panel_context_reference_axis_invalid")
    available_market = (
        BASE_MARKET_CONTEXT_SOURCE_IDS
        if observation_returns is None
        else (*BASE_MARKET_CONTEXT_SOURCE_IDS, *OBSERVED_MARKET_SOURCE_IDS)
    )
    selected_sector = (
        SECTOR_CONTEXT_SOURCE_IDS
        if selected_sector_source_ids is None
        else selected_sector_source_ids
    )
    selected_market = (
        available_market if selected_market_source_ids is None else selected_market_source_ids
    )
    if len(set(selected_sector)) != len(selected_sector) or len(set(selected_market)) != len(
        selected_market
    ):
        raise PanelFeatureBoundaryError("alpha_research.feature_context_axis_invalid")
    if not set(selected_sector).issubset(SECTOR_CONTEXT_SOURCE_IDS) or not set(
        selected_market
    ).issubset(available_market):
        raise PanelFeatureBoundaryError("alpha_research.feature_context_source_not_installed")
    if selected_sector_source_ids is not None or selected_market_source_ids is not None:
        for values in (
            observation_returns,
            observation_volume_state,
            observation_dollar_volume,
            observation_high_distance,
            observation_low_distance,
        ):
            if values is not None and values.shape != (sessions, listings):
                raise PanelFeatureBoundaryError("alpha_research.feature_context_axis_invalid")
    sector_output = {
        SECTOR_CONTEXT_SOURCE_IDS.index(name): position
        for position, name in enumerate(selected_sector)
    }
    market_output = {
        available_market.index(name): position for position, name in enumerate(selected_market)
    }
    groups: tuple[tuple[IntArray, IntArray], ...] = ()
    if reference_eligible is not None:
        masks, inverse = np.unique(reference_eligible, axis=0, return_inverse=True)
        groups = tuple(
            (np.flatnonzero(inverse == i), np.flatnonzero(mask)) for i, mask in enumerate(masks)
        )

    def reduce_reference(
        values: FloatArray,
        statistic: Callable[[FloatArray], FloatArray],
        positions: IntArray | None = None,
    ) -> FloatArray:
        if reference_eligible is None:
            return statistic(values if positions is None else values[:, positions])
        reduced: FloatArray = np.full(sessions, np.nan, dtype=np.float64)
        for days, members in groups:
            if positions is not None:
                members = members[np.isin(members, positions)]
            if len(members):
                reduced[days] = statistic(np.ascontiguousarray(values[np.ix_(days, members)]))
        return reduced

    def mean(values: FloatArray) -> FloatArray:
        return np.nanmean(values, axis=1)

    def sample_std(values: FloatArray) -> FloatArray:
        return np.nanstd(values, axis=1, ddof=1)

    def population_std(values: FloatArray) -> FloatArray:
        return np.nanstd(values, axis=1)

    def total(values: FloatArray) -> FloatArray:
        return np.sum(values, axis=1)

    latest: IntArray = np.full(sessions, -1, dtype=np.int64)
    known: IntArray = np.asarray(
        [index for index, value in enumerate(holding_end_sessions) if value is not None],
        dtype=np.int64,
    )
    holding = np.asarray([holding_end_sessions[index] for index in known], dtype=object)
    for position, formation in enumerate(formation_sessions):
        eligible = known[np.flatnonzero(holding < formation)]
        if eligible.size:
            latest[position] = int(eligible[-1])

    # Each run of formations reads the Sectors in force there (V346); one run while no
    # reclassification falls inside the window.
    sector_runs: tuple[tuple[slice, dict[str, IntArray]], ...] = tuple(
        (
            rows,
            {
                sector: np.asarray(
                    [
                        position
                        for position, listing in enumerate(ordered_listing_ids)
                        if mapping[listing] == sector
                    ],
                    dtype=np.int64,
                )
                for sector in ordered_sector_ids
            },
        )
        for rows, mapping in sector_slices(sector_by_listing_id, formation_sessions)
    )
    sector_return_source: FloatArray | None = None
    if 0 in sector_output:
        sector_return_source = np.full((sessions, sectors), np.nan, dtype=np.float64)
        for rows, sector_positions in sector_runs:
            for sector_position, sector in enumerate(ordered_sector_ids):
                with np.errstate(invalid="ignore"):
                    sector_return_source[rows, sector_position] = reduce_reference(
                        raw_simple_execution_returns, mean, sector_positions[sector]
                    )[rows]
    # Deliberately not widened by an observed Sector breadth. It was built and
    # measured at two columns and at ten, and both lost: +0.018119 and +0.019066
    # against +0.020944 without it. The Sector broadcast slot is full -- the same
    # value repeated across every member of a Sector competes with the fifteen
    # Sector columns already carrying that dimension.
    sector_context: FloatArray = np.full(
        (sessions, sectors, len(selected_sector)), np.nan, dtype=np.float64
    )
    if 0 in sector_output:
        assert sector_return_source is not None
        for position, source_position in enumerate(latest):
            if source_position >= 0:
                sector_context[position, :, sector_output[0]] = sector_return_source[
                    int(source_position)
                ]
    if selected_sector_source_ids is None:
        sector_context[:, :, 1:] = sector_state_values
    else:
        for column, output in sector_output.items():
            if column:
                sector_context[:, :, output] = sector_state_values[:, :, column - 1]

    market_return_source: FloatArray | None = None
    breadth_source: FloatArray | None = None
    dispersion_source: FloatArray | None = None
    with np.errstate(invalid="ignore", divide="ignore"):
        if any(column in market_output for column in (0, 1, 2, 3)):
            market_return_source = reduce_reference(raw_log_execution_returns, mean)
        if 4 in market_output:
            finite_simple = np.isfinite(raw_simple_execution_returns)
            if reference_eligible is not None:
                finite_simple &= reference_eligible
            positive = finite_simple & (raw_simple_execution_returns > 0.0)
            counts = np.sum(finite_simple, axis=1)
            breadth_source = np.divide(
                np.sum(positive, axis=1),
                counts,
                out=np.full(sessions, np.nan, dtype=np.float64),
                where=counts > 0,
            )
        if 5 in market_output:
            dispersion_source = reduce_reference(raw_log_execution_returns, sample_std)
    cumulative_log: FloatArray | None = None
    if 3 in market_output:
        assert market_return_source is not None
        cumulative_log = np.cumsum(
            np.where(np.isfinite(market_return_source), market_return_source, 0.0)
        )
    # Two market columns read the *observation* panel rather than the execution
    # one, and they are the reason this parameter exists.
    #
    # Every column below it is an aggregate of `raw_*_execution_returns`, which
    # are outcomes: a return realised over `open(t+1)` to `open(t+2)`. An outcome
    # may only be read once its holding period has ended, so `latest[position]`
    # moves those columns about three sessions back from formation. That is
    # correct for an outcome and wrong for an observation. Breadth computed from
    # close-to-close returns is complete at `close(t)`, which is when formation
    # happens, so it is available at the formation session itself -- and the
    # freshest lags are exactly where the measured signal in this block sits.
    width = len(selected_market)
    market_context: FloatArray = np.full((sessions, width), np.nan, dtype=np.float64)
    if observation_returns is not None:
        if observation_returns.shape != (sessions, listings):
            raise PanelFeatureBoundaryError("alpha_research.feature_context_axis_invalid")
        with np.errstate(invalid="ignore", divide="ignore"):
            finite_observed: BoolArray | None = None
            observed_counts: IntArray | None = None
            if any(column in market_output for column in (8, 10, 16)):
                finite_observed = np.isfinite(observation_returns)
                if reference_eligible is not None:
                    finite_observed &= reference_eligible
            if 8 in market_output or 10 in market_output:
                assert finite_observed is not None
                observed_counts = np.sum(finite_observed, axis=1)
            if 8 in market_output:
                assert finite_observed is not None and observed_counts is not None
                market_context[:, market_output[8]] = np.divide(
                    np.sum(finite_observed & (observation_returns > 0.0), axis=1),
                    observed_counts,
                    out=np.full(sessions, np.nan, dtype=np.float64),
                    where=observed_counts > 0,
                )
            if 9 in market_output:
                market_context[:, market_output[9]] = reduce_reference(
                    observation_returns, sample_std
                )

            # Implied average correlation: the index variance a set of stock
            # variances and one realised index variance imply, under equal
            # weights. Session constant, and not a transform of the index return.
            if 10 in market_output:
                assert finite_observed is not None and observed_counts is not None
                safe_counts = np.maximum(observed_counts, 1)
                observed_mean = np.divide(
                    reduce_reference(np.where(finite_observed, observation_returns, 0.0), total),
                    observed_counts,
                    out=np.full(sessions, np.nan, dtype=np.float64),
                    where=observed_counts > 0,
                )
                stock_vol = _trailing_std(observation_returns, 63)
                index_vol = _trailing_std(observed_mean[:, None], 63)[:, 0]
                mean_variance = reduce_reference(stock_vol**2, mean)
                mean_vol = reduce_reference(stock_vol, mean)
                denominator = mean_vol**2 - mean_variance / safe_counts
                market_context[:, market_output[10]] = np.divide(
                    index_vol**2 - mean_variance / safe_counts,
                    denominator,
                    out=np.full(sessions, np.nan, dtype=np.float64),
                    where=np.isfinite(denominator) & (np.abs(denominator) > 0.0),
                )

            # The dispersion split. Between-Sector is the spread of Sector means;
            # within is what is left once each name is measured against its own
            # Sector, and the ratio is the rotation-versus-selection regime in one
            # number.
            if any(column in market_output for column in (11, 12, 13)):
                sector_mean = np.full_like(observation_returns, np.nan)
                for rows, sector_positions in sector_runs:
                    for positions in sector_positions.values():
                        if positions.size:
                            sector_mean[rows, positions] = reduce_reference(
                                observation_returns, mean, positions
                            )[rows][:, None]
                within: FloatArray | None = None
                between: FloatArray | None = None
                if 11 in market_output or 13 in market_output:
                    within = reduce_reference(observation_returns - sector_mean, population_std)
                if 12 in market_output or 13 in market_output:
                    between = reduce_reference(sector_mean, population_std)
                if 11 in market_output:
                    assert within is not None
                    market_context[:, market_output[11]] = within
                if 12 in market_output:
                    assert between is not None
                    market_context[:, market_output[12]] = between
                if 13 in market_output:
                    assert within is not None and between is not None
                    market_context[:, market_output[13]] = np.divide(
                        between,
                        within,
                        out=np.full(sessions, np.nan, dtype=np.float64),
                        where=within > 0.0,
                    )

            if 14 in market_output and observation_volume_state is not None:
                # Universe volume against its own normal, equal weighted. The
                # per-listing Formula is already the deviation of a session's log
                # volume from its own trailing mean, so the cross-section of it is
                # the market's volume regime without a second owner of that rule.
                market_context[:, market_output[14]] = reduce_reference(
                    observation_volume_state, mean
                )
            if (
                15 in market_output
                and observation_high_distance is not None
                and observation_low_distance is not None
            ):
                at_high = np.isfinite(observation_high_distance) & (
                    observation_high_distance >= -_NEW_EXTREME_BAND
                )
                at_low = np.isfinite(observation_low_distance) & (
                    observation_low_distance <= _NEW_EXTREME_BAND
                )
                reach = np.isfinite(observation_high_distance)
                if reference_eligible is not None:
                    at_high &= reference_eligible
                    at_low &= reference_eligible
                    reach &= reference_eligible
                reached = np.sum(reach, axis=1)
                market_context[:, market_output[15]] = np.divide(
                    np.sum(at_high, axis=1) - np.sum(at_low, axis=1),
                    reached,
                    out=np.full(sessions, np.nan, dtype=np.float64),
                    where=reached > 0,
                )
            if 16 in market_output and observation_dollar_volume is not None:
                # The share of a session's traded value that changed hands in
                # names that rose. Breadth counts listings; this weighs them, and
                # the two disagree exactly when a move is narrow and large.
                assert finite_observed is not None
                traded = np.isfinite(observation_dollar_volume) & finite_observed
                total_traded = reduce_reference(
                    np.where(traded, observation_dollar_volume, 0.0), total
                )
                market_context[:, market_output[16]] = np.divide(
                    reduce_reference(
                        np.where(
                            traded & (observation_returns > 0.0),
                            observation_dollar_volume,
                            0.0,
                        ),
                        total,
                    ),
                    total_traded,
                    out=np.full(sessions, np.nan, dtype=np.float64),
                    where=total_traded > 0.0,
                )

    for position, source_position_value in enumerate(latest):
        source_position = int(source_position_value)
        if source_position < 0:
            continue
        if 0 in market_output:
            assert market_return_source is not None
            market_context[position, market_output[0]] = market_return_source[source_position]
        for column, lookback in ((1, 21), (2, 63)):
            if column not in market_output:
                continue
            assert market_return_source is not None
            start = source_position - lookback + 1
            if start >= 0:
                window = market_return_source[start : source_position + 1]
                if np.isfinite(window).all():
                    market_context[position, market_output[column]] = np.std(window, ddof=1)
        if 3 in market_output:
            assert cumulative_log is not None
            drawdown_start = source_position - 252 + 1
            if drawdown_start >= 0:
                window = cumulative_log[drawdown_start : source_position + 1]
                market_context[position, market_output[3]] = cumulative_log[
                    source_position
                ] - np.max(window)
        if 4 in market_output:
            assert breadth_source is not None
            market_context[position, market_output[4]] = breadth_source[source_position]
        if 5 in market_output:
            assert dispersion_source is not None
            market_context[position, market_output[5]] = dispersion_source[source_position]
    if selected_market_source_ids is None:
        market_context[:, 6:8] = market_interaction_state_values
    else:
        for column, output in market_output.items():
            if column in (6, 7):
                market_context[:, output] = market_interaction_state_values[:, column - 6]
    return (
        cast(FloatArray, _readonly(sector_context, dtype=np.dtype(np.float64))),
        cast(FloatArray, _readonly(market_context, dtype=np.dtype(np.float64))),
    )


class FoldScaleReceipt(_Contract):
    """Identity and causality proof for one fold/boundary transformed block."""

    kind: Literal["FoldScaleReceipt"] = "FoldScaleReceipt"
    program_hash: str = Field(pattern=_HASH)
    fold_index: int = Field(ge=0)
    boundary_id: Literal["INNER", "OUTER"]
    role_id: FeatureRoleId
    method_id: InstalledPanelViewId | HistoricalPanelEvidenceViewId
    final_scale_method_id: str
    ordered_feature_ids: tuple[str, ...] = Field(min_length=1)
    fit_sessions: tuple[date, ...] = Field(min_length=1)
    transform_sessions: tuple[date, ...] = Field(min_length=1)
    native_entity_axis_hash: str = Field(pattern=_HASH)
    source_array_hash: str = Field(pattern=_HASH)
    parameter_hash: str = Field(pattern=_HASH)
    fitted_center_scale_hash: str = Field(pattern=_HASH)
    transformed_array_hash: str = Field(pattern=_HASH)
    final_scale_identity: str = Field(pattern=_HASH)
    scale_states: tuple[ScaleState, ...] = Field(min_length=1)
    constant_feature_count: int = Field(ge=0)
    near_constant_feature_count: int = Field(ge=0)
    final_scale_entity_axis_hash: str | None = Field(default=None, pattern=_HASH)
    source_missing_session_feature_count: int | None = Field(default=None, ge=0)
    source_missing_session_feature_axis_hash: str | None = Field(default=None, pattern=_HASH)
    constant_session_feature_count: int | None = Field(default=None, ge=0)
    constant_session_feature_axis_hash: str | None = Field(default=None, pattern=_HASH)
    nondegenerate_session_feature_count: int | None = Field(default=None, ge=0)
    listing_axis_mean_max_abs: float | None = Field(default=None, ge=0.0)
    listing_axis_population_std_max_abs_error: float | None = Field(default=None, ge=0.0)
    future_fit_violation_count: Literal[0] = 0
    receipt_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal a writable installed feature-view scale receipt using its non-null payload.

        Args:
            values: Explicit receipt fields excluding receipt_hash; method_id must be an installed
                writable view.

        Returns:
            Validated scaling receipt with canonical exclude-none receipt_hash.

        Raises:
            PanelFeatureBoundaryError: The method is historical and cannot be written.
            pydantic.ValidationError: Receipt fields or consistency violate the model.
        """
        if values.get("method_id") not in INSTALLED_PANEL_VIEW_IDS:
            raise PanelFeatureBoundaryError("alpha_research.historical_panel_view_not_writable")
        provisional = cls.model_construct(**values, receipt_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"receipt_hash"}, exclude_none=True)
        return cls(**values, receipt_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require aligned scale states, causal fit/transform clocks and exact receipt content.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PanelFeatureBoundaryError: Scale states/counts disagree, session axes are noncanonical
                or noncausal, required listing-scale evidence is absent or exclude-none receipt_hash
                is inconsistent.
        """
        listing_scale_evidence = (
            self.final_scale_entity_axis_hash,
            self.source_missing_session_feature_count,
            self.source_missing_session_feature_axis_hash,
            self.constant_session_feature_count,
            self.constant_session_feature_axis_hash,
            self.nondegenerate_session_feature_count,
            self.listing_axis_mean_max_abs,
            self.listing_axis_population_std_max_abs_error,
        )
        if (
            len(self.scale_states) != len(self.ordered_feature_ids)
            or self.constant_feature_count
            != sum(value == "CONSTANT_TRAINING_FEATURE" for value in self.scale_states)
            or self.fit_sessions != tuple(sorted(set(self.fit_sessions)))
            or self.transform_sessions != tuple(sorted(set(self.transform_sessions)))
            or max(self.fit_sessions) >= min(self.transform_sessions)
            or (
                self.final_scale_method_id == "PER_SESSION_LISTING_MEAN_POPULATION_STD_Z"
                and any(value is None for value in listing_scale_evidence)
            )
            or self.receipt_hash
            != canonical_hash(
                self.model_dump(mode="json", exclude={"receipt_hash"}, exclude_none=True)
            )
        ):
            raise PanelFeatureBoundaryError("alpha_research.fold_scale_receipt_invalid")
        return self


class ViewRetentionEvidence(_Contract):
    """Record retained-row fractions and role-specific exclusions for one installed view."""

    method_id: InstalledPanelViewId
    total_row_count: int = Field(ge=1)
    retained_row_count: int = Field(ge=0)
    retained_fraction: float = Field(ge=0.0, le=1.0)
    excluded_by_role: dict[str, int]
    excluded_session_count_by_role: dict[str, int]
    row_axis_hash: str = Field(pattern=_HASH)


class PanelFeaturePreflight(_Contract):
    """Seal selected view axes, common row/session retention and lookback/aggregation bounds."""

    kind: Literal["PanelFeaturePreflight"] = "PanelFeaturePreflight"
    selected_method_ids: tuple[InstalledPanelViewId, ...] = Field(min_length=1)
    feature_count_by_method: dict[str, int]
    ordered_feature_axis_hash_by_method: dict[str, str] = Field(default_factory=dict)
    common_row_retention_by_method: tuple[ViewRetentionEvidence, ...] = Field(min_length=1)
    common_row_axis_hash: str = Field(pattern=_HASH)
    common_session_axis_hash: str = Field(pattern=_HASH)
    maximum_lookback_sessions: int = Field(ge=0)
    maximum_aggregation_span: int = Field(ge=1)
    preflight_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the declared multi-view common-row preflight evidence.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical preflight_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return _seal(cls, dict(values), "preflight_hash")


__all__ = [
    "BASE_MARKET_CONTEXT_SOURCE_IDS",
    "FORMATION_OBSERVATION_DOLLAR_VOLUME_SOURCE_LANE_ID",
    "INSTALLED_PANEL_VIEW_IDS",
    "SECTOR_CONTEXT_SOURCE_IDS",
    "SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS",
    "SPARSE_SESSION_AMPLITUDE_VIEW_IDS",
    "FeatureRoleId",
    "FeatureRoleSelection",
    "FeatureTransformDescriptor",
    "FeatureTransformId",
    "FoldScaleReceipt",
    "InstalledPanelViewId",
    "PanelFeatureBoundaryError",
    "PanelFeaturePreflight",
    "PanelFeatureSourceArrays",
    "PanelFeatureViewCatalog",
    "PanelFeatureViewRecipe",
    "ScaleState",
    "ViewRetentionEvidence",
    "assemble_panel_context_arrays",
    "build_installed_panel_feature_view_catalog",
    "panel_array_hash",
]
