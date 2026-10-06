"""Independent desktop-cost-qualified feature catalog."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, cast

from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    FEATURE_OBSERVATION_CLOCK_POLICY_ID,
    LEGACY_SKIP_WIRE_FIELD,
    FeatureAvailabilityPolicy,
    FeatureObservationClock,
    SourceAvailabilityCatalog,
    formula_observation_policy_hash,
    formula_skip_sessions,
    formula_source_authority_binding_hash,
    installed_feature_availability_policy,
    installed_source_availability_catalog,
    observation_clock_for,
)
from alphalattice.foundation.feature_engine.contracts import FeatureCatalogBinding, canonical_hash
from alphalattice.foundation.feature_engine.producers.arithmetic_identity import (
    control_arithmetic_content_hash,
)
from alphalattice.kernel.quant.factor_contracts import FactorSpec
from alphalattice.kernel.shared_kernel.identity_successors import is_current, recorded_origin

DESKTOP_FEATURE_CATALOG_ID = "feature-catalog.desktop-core-equity"
SOURCE_AVAILABILITY_ROLE = "feature_engine.source_availability_catalog"
"""The source-availability catalog's role in ``config/identity-roles.json`` (V345)."""
_RESOURCE_NAME = "resources/desktop-feature-catalog.json"


@lru_cache(maxsize=1)
def _resource_payload() -> dict[str, Any]:
    raw = (
        files("alphalattice.foundation.feature_engine")
        .joinpath(_RESOURCE_NAME)
        .read_text(encoding="utf-8")
    )
    payload = cast(dict[str, Any], json.loads(raw))
    if set(payload) != {"stable_id", "observation_clock", "factors", "maintenance_policy"}:
        raise ValueError("desktop feature catalog resource shape is invalid")
    if payload["stable_id"] != DESKTOP_FEATURE_CATALOG_ID:
        raise ValueError("desktop feature catalog stable ID is invalid")
    return payload


MARKET_DEPENDENT_FACTOR_IDS = frozenset(
    str(value) for value in _resource_payload()["maintenance_policy"]["market_dependent_factor_ids"]
)


@dataclass(frozen=True)
class DesktopCoreFeatureBundle:
    """Content-addressed activation manifest for the current base Panel.

    Its factor axis is a governed selection from the installed Formula Registry,
    not an architectural count and not the set of every Formula this build knows.
    A later catalog revision may explicitly activate a different axis.
    """

    bundle_id: str
    factor_ids: tuple[str, ...]
    formula_refs: tuple[tuple[str, str], ...]
    numerical_spec_hashes: tuple[tuple[str, str], ...]
    bundle_hash: str


def desktop_core_feature_bundle() -> DesktopCoreFeatureBundle:
    """Describe the installed base Panel's activated factor recipes.

    Returns:
        Ordered factor axis, formula references, numerical specifications, and
        the digest binding catalog content to its declared observation clock.

    Raises:
        ValueError: The installed resource has an invalid shape or stable ID.
    """
    payload = _resource_payload()
    factor_entries = tuple(payload["factors"])
    observation_clock = dict(payload["observation_clock"])
    factor_ids = tuple(str(item["factor_id"]) for item in factor_entries)
    formula_refs = tuple(
        (str(item["factor_id"]), str(item["formula_ref"])) for item in factor_entries
    )
    numerical_spec_hashes = tuple(
        (
            str(item["factor_id"]),
            canonical_hash(
                {
                    key: item[key]
                    for key in (
                        "formula",
                        "formula_ref",
                        "lag_sessions",
                        "minimum_observations",
                        "required_fields",
                        "return_convention",
                        "window_sessions",
                    )
                }
            ),
        )
        for item in factor_entries
    )
    return DesktopCoreFeatureBundle(
        bundle_id="desktop-core-features.2026-08-07",
        factor_ids=factor_ids,
        formula_refs=formula_refs,
        numerical_spec_hashes=numerical_spec_hashes,
        bundle_hash=canonical_hash(
            {
                "bundle_id": "desktop-core-features.2026-08-07",
                "factor_ids": factor_ids,
                "formula_refs": formula_refs,
                "numerical_spec_hashes": numerical_spec_hashes,
                "catalog_content": factor_entries,
                # The declared clock rides with the bundle a control's identity
                # is built from. Without it a bundle whose skips mean "economic
                # skip" and one whose identical numbers meant "safety lag" would
                # publish the same identity over different values.
                "observation_clock": observation_clock,
            }
        ),
    )


@dataclass(frozen=True)
class FeatureMaintenanceContract:
    """Finite input and invalidation contract for one desktop feature.

    Attributes:
        factor_id: Recipe whose maintenance this contract governs.
        source_basis: Provider source families read by the recipe.
        return_transform: Transformation applied before return aggregation.
        required_fields: Source fields required to recompute the feature.
        formula_skip_sessions: Economic skip between the observation and source window.
        lookback_sessions: Sessions in the recipe's source window.
        maximum_invalidation_sessions: Maximum history affected by a source correction.
        market_reference: Required market reference for market-dependent recipes.
        adjustment_transformation: Rebase rule for source prices or volume.
        bounded_state_proof: Qualified finite-window basis for local recomputation.
        research_role: Recipe family used in research.
        maintenance_cost: Declared maintenance-cost class.
    """

    factor_id: str
    source_basis: str
    return_transform: str
    required_fields: tuple[str, ...]
    formula_skip_sessions: int
    lookback_sessions: int
    maximum_invalidation_sessions: int
    market_reference: str | None
    adjustment_transformation: str
    bounded_state_proof: str
    research_role: str
    maintenance_cost: str

    def __post_init__(self) -> None:
        """Refuse unbounded maintenance state and invalid session bounds.

        Raises:
            ValueError: Windows are nonpositive, the economic skip is negative,
                invalidation exceeds the qualified budget, or the state proof is unknown.
        """
        if self.lookback_sessions < 1 or self.maximum_invalidation_sessions < 1:
            raise ValueError("feature maintenance windows must be positive")
        if self.formula_skip_sessions < 0:
            raise ValueError("feature formula skip must be non-negative")
        if self.maximum_invalidation_sessions > 276:
            raise ValueError("desktop feature exceeds the qualified invalidation budget")
        if self.bounded_state_proof not in {
            "finite_rolling_window",
            "finite_calendar_window",
            "window_local_cumulative_transform",
        }:
            raise ValueError("desktop feature has no bounded-state proof")


def _maintenance_contract(
    spec: FactorSpec,
    *,
    policy: dict[str, Any],
) -> FeatureMaintenanceContract:
    fields = tuple(str(item) for item in spec.required_fields)
    has_adjusted = "provider_adjusted_close" in fields
    has_ohlc = any(item.endswith("_split_adjusted") for item in fields)
    has_volume = "volume_raw" in fields
    if has_adjusted and (has_ohlc or has_volume):
        source_basis = "PROVIDER_ADJUSTED_CLOSE+PROVIDER_OHLCV"
    elif has_adjusted:
        source_basis = "PROVIDER_ADJUSTED_CLOSE"
    elif has_ohlc and has_volume:
        source_basis = "PROVIDER_OHLCV"
    elif has_ohlc:
        source_basis = "PROVIDER_OHLC"
    elif has_volume:
        source_basis = "PROVIDER_VOLUME"
    else:
        source_basis = "PROVIDER_OHLCV"
    finite_calendar = frozenset(str(item) for item in policy["finite_calendar_factor_ids"])
    local_cumulative = frozenset(str(item) for item in policy["window_local_cumulative_factor_ids"])
    market_dependent = frozenset(str(item) for item in policy["market_dependent_factor_ids"])
    overrides = {
        str(key): int(value)
        for key, value in dict(policy["maximum_invalidation_overrides"]).items()
    }
    if spec.factor_id in finite_calendar:
        proof = "finite_calendar_window"
    elif spec.factor_id in local_cumulative:
        proof = "window_local_cumulative_transform"
    else:
        proof = "finite_rolling_window"
    transformation = (
        "ADJUSTED_CLOSE_UNIFORM_SCALE"
        if has_adjusted and not (has_ohlc or has_volume)
        else "SPLIT_REBASE_OHLC_X_K_VOLUME_DIV_K"
        if has_ohlc or has_volume
        else "PROVIDER_VALUE_STABLE"
    )
    return FeatureMaintenanceContract(
        factor_id=spec.factor_id,
        source_basis=source_basis,
        return_transform="LOG_RETURN" if has_adjusted else "NONE",
        required_fields=fields,
        formula_skip_sessions=formula_skip_sessions(spec),
        lookback_sessions=spec.window_sessions,
        maximum_invalidation_sessions=overrides.get(spec.factor_id, spec.window_sessions),
        market_reference=(
            str(policy["market_reference"]) if spec.factor_id in market_dependent else None
        ),
        adjustment_transformation=transformation,
        bounded_state_proof=proof,
        research_role=str(spec.family.value),
        maintenance_cost="BOUNDED_LOCAL",
    )


_LAGGED_PROJECTION = re.compile(r"^(?P<stem>.+?)_(?:lag|lagged|shift|shifted)_(?P<offset>\d+)$")


def _refuse_lagged_projections(factor_ids: tuple[str, ...]) -> None:
    """Refuse a canonical axis that names ``shift(existing_feature, k)`` as a Formula.

    The canonical store holds one thing: ``Feature_i(T)``, the raw output of a
    Formula on its own observation session. ``Feature_i_lag_1(T)`` is not a
    second Formula -- it is the same value read from a different row, and it is a
    model-input projection the consuming Desk compiles against an authoritative
    trading-session axis, cached as a content-addressed experiment artifact if it
    is worth caching at all.

    Writing one back here would put a strategy's choice of history depth into the
    Data layer, duplicate every value it shifts, and give a projection a Feature's
    identity and lineage. A genuine economic Formula -- an EWMA, a slope, an
    autocorrelation, a twelve-minus-one momentum -- states its own mathematics and
    is welcome; a bare shift of an installed Formula is not, and the name is the
    only place that distinction is visible from here.
    """
    installed = set(factor_ids)
    for factor_id in factor_ids:
        matched = _LAGGED_PROJECTION.match(factor_id)
        if matched is not None and matched.group("stem") in installed:
            raise ValueError(
                f"desktop feature catalog names a lagged projection of another Formula: {factor_id}"
            )


@dataclass(frozen=True)
class FeatureCatalog:
    """Bind qualified factor recipes, maintenance policy, and observation clocks.

    Attributes:
        binding: Content and implementation identities governing the catalog.
        factors: Activated recipes in their admitted order.
        maintenance_contracts: Finite maintenance contract for each recipe.
        maintenance_policy: Catalog policy used to derive maintenance contracts.
        observation_clock_policy: Declared clock policy validated against installed owners.
        observation_clocks: Per-recipe source intervals and economic skips.
    """

    binding: FeatureCatalogBinding
    factors: tuple[FactorSpec, ...]
    maintenance_contracts: tuple[FeatureMaintenanceContract, ...]
    maintenance_policy: dict[str, Any]
    observation_clock_policy: dict[str, Any]
    observation_clocks: tuple[FeatureObservationClock, ...]
    source_availability: FeatureAvailabilityPolicy
    """The governing availability event, derived from the owners below.

    Held beside the per-Formula clocks rather than inside them: when a value may
    be read is not part of what the Formula computes.
    """

    source_authorities: SourceAvailabilityCatalog
    """Every installed source-availability owner, and the field map that resolves them."""

    formula_source_authorities: Mapping[str, tuple[str, ...]]
    """Which owners each Formula depends on, derived from its ``required_fields``.

    A Formula reading ``sector_membership_asof`` depends on a map that is a
    current classification backfilled across history; one reading ``rev_5``
    depends on another installed Formula. Neither is the price feed, and before
    this the catalog had no way to say so.
    """

    @classmethod
    def load(cls) -> FeatureCatalog:
        """Load and qualify the installed desktop catalog resource.

        Returns:
            Catalog with admitted recipes, maintenance bounds, clocks, and bindings.

        Raises:
            ValueError: Resource, clock, factor axis, or maintenance policy is invalid.
        """
        return cls.from_payload(_resource_payload())

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FeatureCatalog:
        """Build one governed catalog revision from a complete typed payload.

        Args:
            payload: Stable ID, observation clock, factor recipes, and maintenance policy.

        Returns:
            Qualified catalog with derived contracts, source authorities, and bindings.

        Raises:
            ValueError: Shape, clock authority, factor axis, recipe, or maintenance
                bounds fail qualification; bare shifts of installed factors are refused.
        """
        if set(payload) != {
            "stable_id",
            "observation_clock",
            "factors",
            "maintenance_policy",
        }:
            raise ValueError("desktop feature catalog payload shape is invalid")
        clock_policy = cast(dict[str, Any], payload["observation_clock"])
        source_authorities = installed_source_availability_catalog()
        availability = installed_feature_availability_policy()
        # A payload that does not declare the installed clock is refused rather
        # than defaulted. Every pre-successor catalog states no clock at all, so
        # it cannot acquire observation-session authority by being loaded here.
        if (
            str(clock_policy.get("policy_id")) != FEATURE_OBSERVATION_CLOCK_POLICY_ID
            or str(clock_policy.get("observation_session")) != "FORMULA_FORMATION_SESSION"
            or str(clock_policy.get("formula_skip_semantics")) != "ECONOMIC_SKIP_ONLY"
            or str(clock_policy.get("legacy_skip_wire_field")) != LEGACY_SKIP_WIRE_FIELD
            or str(clock_policy.get("source_availability_policy_id")) != availability.policy_id
        ):
            raise ValueError("desktop feature catalog observation clock is not the installed one")
        factors = tuple(
            FactorSpec.model_validate_json(json.dumps(item, separators=(",", ":")))
            for item in payload["factors"]
        )
        factor_ids = tuple(item.factor_id for item in factors)
        if (
            not factors
            or len(set(factor_ids)) != len(factors)
            or factor_ids != tuple(sorted(factor_ids))
        ):
            raise ValueError("desktop feature catalog requires sorted unique factors")
        _refuse_lagged_projections(factor_ids)
        policy = cast(dict[str, Any], payload["maintenance_policy"])
        policy_factor_ids = {
            *(str(value) for value in policy["market_dependent_factor_ids"]),
            *(str(value) for value in policy["finite_calendar_factor_ids"]),
            *(str(value) for value in policy["window_local_cumulative_factor_ids"]),
            *(str(value) for value in policy["maximum_invalidation_overrides"]),
        }
        if not policy_factor_ids.issubset(factor_ids):
            raise ValueError("catalog maintenance policy references inactive factors")
        market_dependent = frozenset(str(value) for value in policy["market_dependent_factor_ids"])
        if not market_dependent.issubset(factor_ids):
            raise ValueError("catalog market-dependent factors are outside the factor axis")
        contracts = tuple(_maintenance_contract(spec, policy=policy) for spec in factors)
        # The interval *kind* is the one thing a recipe cannot state, and the
        # maintenance policy already names the calendar-driven Formulas. Reusing
        # that declaration keeps the classification in one place instead of
        # opening a second registry beside it.
        calendar_ids = frozenset(str(value) for value in policy["finite_calendar_factor_ids"])
        clocks = tuple(
            observation_clock_for(
                spec,
                source_interval_kind=(
                    "CALENDAR_MONTH_SELECTION"
                    if spec.factor_id in calendar_ids
                    else "CONTIGUOUS_SESSION_WINDOW"
                ),
            )
            for spec in factors
        )
        # Derived from each recipe's declared fields, so a Formula that starts
        # reading a new kind of source cannot keep the old authority set by
        # forgetting to update a second declaration of it.
        authorities = {
            spec.factor_id: source_authorities.authorities_for(spec.required_fields)
            for spec in factors
        }
        semantic_payload = {
            "stable_id": payload["stable_id"],
            "observation_clock": clock_policy,
            "factors": [factor.model_dump(mode="json") for factor in factors],
            "maintenance_policy": policy,
        }
        catalog_content_hash = canonical_hash(semantic_payload)
        # The measured bytes of the arithmetic, beside the description of it.
        # The description alone survived an arbitrary rewrite of the engine, so a
        # catalog binding built from it could not answer whether the code that
        # produced a Panel's values had changed -- and one consumer took equal
        # catalog hashes as proof that it had not.
        formula_implementation_hash = canonical_hash(
            {
                "arithmetic_content": control_arithmetic_content_hash(),
                "engine": "playpen.feature_engine.desktop_finite_window_materializer",
                "numeric_domain": "float64",
                "rolling_evaluation": "direct_finite_source_window_recompute",
                "adjusted_return_transform": "log(price_t/price_t_minus_1)",
                "return_aggregation": "sum_log_returns",
                "residual_momentum_estimation": (
                    "intercept_beta_ols_252_ending_at_lag21_then_sum_terminal_231_residuals"
                ),
                "seasonality": "prior_year_same_calendar_month_log_return",
                "factor_formula_refs": tuple(
                    (factor.factor_id, factor.formula_ref) for factor in factors
                ),
            }
        )
        materializer_policy_hash = canonical_hash(
            {
                "engine": "playpen.feature_engine.desktop_bounded_materializer",
                "stable_id": str(payload["stable_id"]),
                "contracts": [asdict(item) for item in contracts],
                **policy,
            }
        )
        return cls(
            binding=FeatureCatalogBinding.create(
                stable_id=str(payload["stable_id"]),
                catalog_content_hash=catalog_content_hash,
                formula_implementation_hash=formula_implementation_hash,
                materializer_policy_hash=materializer_policy_hash,
                formula_observation_policy_hash=formula_observation_policy_hash(clocks),
                source_availability_policy_hash=source_availability_binding(
                    source_authorities.catalog_hash
                ),
                source_authority_binding_hash=formula_source_authority_binding_hash(authorities),
            ),
            factors=factors,
            maintenance_contracts=contracts,
            maintenance_policy=json.loads(json.dumps(policy)),
            observation_clock_policy=json.loads(json.dumps(clock_policy)),
            source_availability=availability,
            source_authorities=source_authorities,
            formula_source_authorities=authorities,
            observation_clocks=clocks,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a detached payload suitable for a governed catalog edit.

        Returns:
            Stable ID, clock declaration, recipes, and copied maintenance policy.
        """
        return {
            "stable_id": self.binding.stable_id,
            "observation_clock": json.loads(json.dumps(self.observation_clock_policy)),
            "factors": [factor.model_dump(mode="json") for factor in self.factors],
            "maintenance_policy": json.loads(json.dumps(self.maintenance_policy)),
        }

    @property
    def factor_ids(self) -> tuple[str, ...]:
        """Return the activated factor IDs in catalog order."""
        return tuple(factor.factor_id for factor in self.factors)

    @property
    def clocks_by_factor(self) -> dict[str, FeatureObservationClock]:
        """Return each activated factor's observation clock keyed by its ID."""
        return {item.factor_id: item for item in self.observation_clocks}

    @property
    def calendar_factor_ids(self) -> frozenset[str]:
        """Return formulas whose newest source event is a calendar selection."""
        return frozenset(
            item.factor_id
            for item in self.observation_clocks
            if item.source_interval.kind == "CALENDAR_MONTH_SELECTION"
        )

    @property
    def contracts_by_factor(self) -> dict[str, FeatureMaintenanceContract]:
        """Return the finite maintenance contracts keyed by factor ID."""
        return {item.factor_id: item for item in self.maintenance_contracts}

    @property
    def market_dependent(self) -> tuple[FactorSpec, ...]:
        """Return catalog recipes whose maintenance needs a market reference."""
        market_ids = {
            contract.factor_id
            for contract in self.maintenance_contracts
            if contract.market_reference is not None
        }
        return tuple(factor for factor in self.factors if factor.factor_id in market_ids)

    @property
    def non_market_dependent(self) -> tuple[FactorSpec, ...]:
        """Return catalog recipes whose maintenance needs no market reference."""
        market_ids = {factor.factor_id for factor in self.market_dependent}
        return tuple(factor for factor in self.factors if factor.factor_id not in market_ids)


def source_availability_binding(catalog_hash: str) -> str:
    """The value a Feature catalog binds for a source-availability catalog (LAWS.md ID1, V345).

    Every catalog binding, Panel lineage and Alpha source identity compares it by equality, so
    it is held at its recorded origin: a move recorded as keeping when every existing field may
    be read (a field or a mark added for a new source) leaves every Panel's binding where it
    was, and an unrecorded move is a new binding.

    Args:
        catalog_hash: A source-availability catalog's own hash, installed or recorded.

    Returns:
        The value its recorded moves lead from.
    """
    return recorded_origin(SOURCE_AVAILABILITY_ROLE, catalog_hash)


def source_availability_is_current(recorded: str, installed: str) -> bool:
    """Whether a recorded source-availability catalog names the installed one (V345).

    Args:
        recorded: The catalog hash a Panel's lineage recorded.
        installed: The installed catalog's hash.

    Returns:
        True when they are equal or recorded moves lead from the first to the second.
    """
    return is_current(SOURCE_AVAILABILITY_ROLE, recorded, installed)


__all__ = [
    "DESKTOP_FEATURE_CATALOG_ID",
    "MARKET_DEPENDENT_FACTOR_IDS",
    "SOURCE_AVAILABILITY_ROLE",
    "DesktopCoreFeatureBundle",
    "FeatureCatalog",
    "FeatureMaintenanceContract",
    "desktop_core_feature_bundle",
    "source_availability_binding",
    "source_availability_is_current",
]
