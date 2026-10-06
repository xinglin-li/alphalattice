"""Complete Formula Specifications and independent Factor development admission."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.feature_engine.catalog.contracts import desktop_core_feature_bundle
from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    FeatureObservationClock,
    formula_skip_sessions,
    installed_source_availability_catalog,
    observation_clock_for,
)
from alphalattice.kernel.quant.factor_contracts import FactorSpec
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .catalog import default_extension_kernel_registry, extension_factor_specs
from .core_bundle import numerical_spec_hash
from .formula import (
    FORMULA_IDS,
    FORMULA_POINT_IN_TIME_IDS,
    formula_goldens,
    formula_preprocessing,
    formula_specification,
)
from .formula_language import FormulaError
from .open_intraday import MEAN_ADJUSTED_RETURN_ID
from .registry import FeatureKernelRegistry

ROBUST_SECTOR_NEUTRAL_Z_ROLE = "ROBUST_SECTOR_NEUTRAL_Z"
TIME_SERIES_ABSOLUTE_STATE_ROBUST_ROLE = "TIME_SERIES_ABSOLUTE_STATE_ROBUST"
STATE_INTERACTION_BLOCK_ROLE = "STATE_INTERACTION_BLOCK"


class FeatureFormulaSpecificationError(ValueError):
    """Stable refusal raised before a Factor reaches development execution."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class FactorFormulaGoldenSeries(_Contract):
    """Name one ordered source column used by a formula's numerical golden example.

    Attributes:
        field: Source field whose values the example supplies.
        values: Ordered observations read by the example.
    """

    field: str = Field(min_length=1, max_length=96)
    values: tuple[float, ...]


class FactorFormulaGoldenExample(_Contract):
    """Bind an explained source fixture to its expected formula result.

    Attributes:
        label: Stable name of the numerical boundary or behavior illustrated.
        rationale: Scientific reason for retaining this example.
        ordered_inputs: Source columns in the qualified field order, with equal row counts.
        expected_value: Expected scalar result, or None when the result should be missing.
        listing_count: Population used when constructing the example's input frame.
    """

    label: str = Field(min_length=1, max_length=96)
    rationale: str = Field(min_length=1)
    ordered_inputs: tuple[FactorFormulaGoldenSeries, ...] = Field(min_length=1)
    expected_value: float | None
    listing_count: int = Field(default=1, ge=1)

    @property
    def input_fields(self) -> tuple[str, ...]:
        """Return the ordered source field names used by this example."""
        return tuple(series.field for series in self.ordered_inputs)

    @property
    def row_count(self) -> int:
        """Return the number of source rows in the example."""
        return len(self.ordered_inputs[0].values)

    def column(self, field: str) -> tuple[float, ...]:
        """Return the example values for a named source field."""
        return next(item.values for item in self.ordered_inputs if item.field == field)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_example(self) -> Self:
        """Reject duplicate fields or columns with unequal row counts."""
        if self.input_fields != tuple(sorted(set(self.input_fields))):
            raise ValueError("feature_engine.formula_specification_golden_invalid")
        if len({len(series.values) for series in self.ordered_inputs}) != 1:
            raise ValueError("feature_engine.formula_specification_golden_invalid")
        return self


class FactorFormulaClock(_Contract):
    """The Formula's own intervals, stated beside the clock they must agree with.

    ``observation_clock`` is the derived authority -- observation session, economic
    skip, availability -- and the interval strings here describe it in the
    Formula's own vocabulary. A specification whose prose and clock disagree is
    refused rather than published.
    """

    formation_row: str = "t"
    observation_clock: FeatureObservationClock
    source_interval: str = Field(min_length=1)
    estimation_interval: str | None = None
    event_interval: str | None = None
    recovery_interval: str | None = None
    minimum_ordered_source_rows: int = Field(ge=1)
    one_row_short_is_missing: bool = True


class FactorFormulaSpecification(_Contract):
    """Pure, complete formula meaning; never an admission decision."""

    kind: Literal["FactorFormulaSpecification"] = "FactorFormulaSpecification"
    status: Literal["COMPLETE"] = "COMPLETE"
    factor_id: str = Field(min_length=1, max_length=128)
    scientific_question: str = Field(min_length=1)
    formula_ref: str = Field(min_length=1, max_length=128)
    implementation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_fields: tuple[str, ...] = Field(min_length=1)
    price_basis: str = Field(min_length=1)
    formation_cutoff: str = Field(min_length=1)
    clock: FactorFormulaClock
    formula: str = Field(min_length=1)
    executable_definition: str | None = Field(default=None, exclude_if=lambda value: value is None)
    sign_convention: str = Field(min_length=1)
    expected_units: str = Field(min_length=1)
    missing_value_behavior: tuple[str, ...] = Field(min_length=1)
    finite_policy: str = Field(min_length=1)
    preprocessing_role: Literal[
        "ROBUST_SECTOR_NEUTRAL_Z",
        "ROBUST_UNIVERSE_Z",
        "TIME_SERIES_ABSOLUTE_STATE_ROBUST",
        "STATE_INTERACTION_BLOCK",
    ]
    golden_examples: tuple[FactorFormulaGoldenExample, ...] = Field(min_length=2)
    absolute_tolerance: float = Field(gt=0.0)
    relative_tolerance: float = Field(gt=0.0)
    specification_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal a complete Formula Specification with its canonical identity."""
        payload = {"kind": "FactorFormulaSpecification", "status": "COMPLETE", **values}
        identity = cls.model_construct(**payload, specification_hash="0" * 64).model_dump(
            mode="json", exclude={"specification_hash"}
        )
        return cls(**payload, specification_hash=str(canonical_hash(identity)))

    @property
    def minimum_ordered_source_rows(self) -> int:
        """Return the source-row minimum required by the Formula clock."""
        return self.clock.minimum_ordered_source_rows

    @property
    def source_authorities(self) -> tuple[str, ...]:
        """Which source owners this Formula depends on, derived from its fields.

        A property rather than a field, and deliberately outside
        ``specification_hash``. ``source_fields`` already states the dependency;
        storing the authority set beside it would be a second declaration of one
        fact, and putting it in the identity would make a Provider's publication
        schedule look like a change to the Formula's mathematics.
        """
        return installed_source_availability_catalog().authorities_for(self.source_fields)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_specification(self) -> Self:
        """Reject a Formula Specification whose sources, examples or identity disagree."""
        if self.source_fields != tuple(sorted(set(self.source_fields))):
            raise ValueError("feature_engine.formula_specification_source_fields_invalid")
        if any(example.input_fields != self.source_fields for example in self.golden_examples):
            raise ValueError("feature_engine.formula_specification_golden_fields_mismatch")
        labels = {example.label: example for example in self.golden_examples}
        short = labels.get("one-row-short-is-missing")
        first = labels.get("first-valid-source-boundary")
        if short is None or first is None:
            raise ValueError("feature_engine.formula_specification_boundary_goldens_missing")
        if (
            short.row_count != self.minimum_ordered_source_rows - 1
            or short.expected_value is not None
        ):
            raise ValueError("feature_engine.formula_specification_short_boundary_invalid")
        if first.row_count != self.minimum_ordered_source_rows:
            raise ValueError("feature_engine.formula_specification_first_boundary_invalid")
        if self.specification_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"specification_hash"})
        ):
            raise ValueError("feature_engine.formula_specification_identity_invalid")
        return self


class FactorDevelopmentCapability(_Contract):
    """Bind formula meaning to executable content and its development evidence method.

    Attributes:
        kind: Capability discriminator.
        factor_id: Factor whose development this capability governs.
        formula_specification_hash: Qualified formula meaning and golden-example identity.
        implementation_hash: Measured identity of the formula's numerical implementation.
        preprocessing_role: Transformation role used by the candidate.
        nearest_neighbor_factor_ids: Ordered comparison factors for incremental evidence.
        incremental_evidence_method: Declared training-only ridge residualization method.
        capability_hash: Canonical identity of these capability fields.
    """

    kind: Literal["FactorDevelopmentCapability"] = "FactorDevelopmentCapability"
    factor_id: str
    formula_specification_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_role: str
    nearest_neighbor_factor_ids: tuple[str, ...] = Field(min_length=1)
    incremental_evidence_method: Literal["TRAINING_RIDGE_RESIDUALIZATION_LAMBDA_1E_6"] = (
        "TRAINING_RIDGE_RESIDUALIZATION_LAMBDA_1E_6"
    )
    capability_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal a development capability with ordered neighbours and its hash."""
        payload = {"kind": "FactorDevelopmentCapability", **values}
        payload["nearest_neighbor_factor_ids"] = tuple(
            sorted(payload["nearest_neighbor_factor_ids"])
        )
        identity = cls.model_construct(**payload, capability_hash="0" * 64).model_dump(
            mode="json", exclude={"capability_hash"}
        )
        return cls(**payload, capability_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Reject unordered neighbours or a capability hash that does not match."""
        if self.nearest_neighbor_factor_ids != tuple(sorted(set(self.nearest_neighbor_factor_ids))):
            raise ValueError("feature_engine.factor_development_neighbours_invalid")
        if self.capability_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"capability_hash"})
        ):
            raise ValueError("feature_engine.factor_development_capability_identity_invalid")
        return self


class FactorDevelopmentAdmissionReceipt(_Contract):
    """Record the Host's development admission or refusal for a bound capability.

    Attributes:
        kind: Admission-receipt discriminator.
        factor_id: Factor assessed for development use.
        capability_hash: Qualified capability whose admission is decided.
        disposition: Admission or refusal outcome.
        reason: Recorded cause of that outcome.
        receipt_hash: Canonical identity of the outcome and its binding.
    """

    kind: Literal["FactorDevelopmentAdmissionReceipt"] = "FactorDevelopmentAdmissionReceipt"
    factor_id: str
    capability_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: Literal["ADMITTED", "REFUSED"]
    reason: str
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal the Host's development-admission receipt with its hash."""
        payload = {"kind": "FactorDevelopmentAdmissionReceipt", **values}
        return cls(**payload, receipt_hash=str(canonical_hash(payload)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Reject an admission receipt whose sealed hash does not match."""
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise ValueError("feature_engine.factor_development_receipt_identity_invalid")
        return self


def _series(field: str, rows: int) -> FactorFormulaGoldenSeries:
    market_returns = tuple(
        0.004 * math.sin(index / 3) + 0.001 * ((index % 5) - 2) for index in range(rows)
    )
    prices = tuple(
        100.0 * math.exp(0.0002 * index + 0.03 * math.sin(index / 13)) for index in range(rows)
    )
    if field == "market_return_log":
        values = market_returns
    elif field == "sector_return_log":
        values = tuple(0.4 * value for value in market_returns)
    elif field == "market_provider_adjusted_close":
        cumulative = 0.0
        output = []
        for value in market_returns:
            cumulative += value
            output.append(300.0 * math.exp(cumulative))
        values = tuple(output)
    elif field == "open_split_adjusted":
        values = tuple(
            price * (1.0 + 0.002 * math.sin(index)) for index, price in enumerate(prices)
        )
    elif field == "high_split_adjusted":
        values = tuple(1.01 * value for value in prices)
    elif field == "low_split_adjusted":
        values = tuple(0.99 * value for value in prices)
    elif field == "rev_5":
        values = tuple(math.sin(index / 7) for index in range(rows))
    elif field == "sector_membership_asof":
        values = (1.0,) * rows
    elif field == "volume_raw":
        values = (1_000_000.0,) * rows
    else:
        values = prices
    return FactorFormulaGoldenSeries(field=field, values=values)


_GOLDEN_EXPECTED: Mapping[str, float] = MappingProxyType(
    {
        "absolute_momentum_20": 0.03398431828834257,
        "at_own_high_share_63": 0.2857142857142857,
        "beta_asymmetry_252": -0.08695193532066718,
        "close_to_close": 0.002505417150241934,
        "down_day_absorption_63": 0.013586956336952238,
        "drawdown_from_63d_high": -0.050415107750213184,
        "gap": 0.004186944559880442,
        "gap_absorption_63": 0.4589809369087034,
        "gap_amplitude": 0.004186944559880442,
        "high_extension": 0.012455748003410094,
        "intraday": 0.0,
        # The boundary frame for a one-row candidate is session index 0, where the
        # declared open series `price * (1 + 0.002 * sin(index))` has `sin(0) == 0`,
        # so open equals close exactly and `|log(close/open)|` is exactly zero. The
        # value recorded here until 2026-09-02, 0.0016815274096384609, is bit-exactly
        # what this same formula yields at session index 1, so it was transcribed
        # from a two-row frame. It is pinned at that row by the "second session"
        # regression in the researcher-methodology-surface case study; the formula
        # is unchanged.
        "intraday_amplitude": 0.0,
        "intraday_reversal_5": 0.00034622919223982495,
        "low_extension": -0.007544918703259362,
        "market_drawdown_x_momentum": -0.023279795354953527,
        "market_vol_ratio_x_reversal": -0.9917788534431158,
        "overnight_intraday_tug_of_war_63": -0.17271462854657055,
        "overnight_reversal_5": -0.012602308633517672,
        "overnight_return_21": 0.0018008546474423657,
        "price_vs_sma20_atr": 0.6149813581447421,
        "price_vs_sma60_atr": -1.5657225979190186,
        "previous_close_excursion_intraday_adjusted_square": 0.00019892107374970178,
        "previous_close_excursion_intraday_cross": 0.00020381376249098835,
        "previous_close_excursion_scaled_span_square": 0.00014427912279327254,
        "range_position": 0.37723335996311663,
        "recovery_ratio_252": 1.2486071746738832,
        "residual_reversal_vol_scaled_5": 1.6326221777621965,
        "return_run_length_21": 21.0,
        "sector_leader_lag_5": 0.0,
        "session_dollar_volume": 100_000_000.0,
        "session_span": 0.020000666706669456,
        "sessions_since_252_high": 66.0,
        "tail_resilience_252": -0.1895608748818209,
    }
)


def _goldens(recipe: FactorSpec) -> tuple[FactorFormulaGoldenExample, ...]:
    exact = int(recipe.minimum_observations)
    listing_count = 12 if recipe.factor_id == "sector_leader_lag_5" else 1
    return (
        FactorFormulaGoldenExample(
            label="first-valid-source-boundary",
            rationale="Exactly the mechanically derived causal source interval.",
            ordered_inputs=tuple(_series(field, exact) for field in recipe.required_fields),
            expected_value=_GOLDEN_EXPECTED[recipe.factor_id],
            listing_count=listing_count,
        ),
        FactorFormulaGoldenExample(
            label="one-row-short-is-missing",
            rationale=(
                "The earliest required row is absent, so the strict causal window is incomplete."
            ),
            ordered_inputs=tuple(_series(field, exact - 1) for field in recipe.required_fields),
            expected_value=None,
            listing_count=listing_count,
        ),
    )


_ROLE_BY_FACTOR: Mapping[str, str] = MappingProxyType(
    {
        "absolute_momentum_20": TIME_SERIES_ABSOLUTE_STATE_ROBUST_ROLE,
        "drawdown_from_63d_high": TIME_SERIES_ABSOLUTE_STATE_ROBUST_ROLE,
        "market_drawdown_x_momentum": STATE_INTERACTION_BLOCK_ROLE,
        "market_vol_ratio_x_reversal": STATE_INTERACTION_BLOCK_ROLE,
        "price_vs_sma20_atr": TIME_SERIES_ABSOLUTE_STATE_ROBUST_ROLE,
        "price_vs_sma60_atr": TIME_SERIES_ABSOLUTE_STATE_ROBUST_ROLE,
        "recovery_ratio_252": TIME_SERIES_ABSOLUTE_STATE_ROBUST_ROLE,
    }
)

_NEIGHBOURS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "absolute_momentum_20": ("mom_126_21", "mom_252_21"),
        "at_own_high_share_63": ("dist_52w_high", "max_21"),
        "beta_asymmetry_252": ("beta_252", "downside_beta_252"),
        "close_to_close": ("rev_5", "rev_21"),
        "down_day_absorption_63": ("downside_beta_252", "downside_vol_63"),
        "drawdown_from_63d_high": ("dist_52w_high", "max_21"),
        "gap_absorption_63": ("rev_5", "rev_21"),
        "gap": ("rev_5", "rev_21"),
        "gap_amplitude": ("vol_21", "vol_63"),
        "high_extension": ("dist_52w_high", "vol_21"),
        "intraday": ("rev_5", "rev_21"),
        "intraday_amplitude": ("vol_21", "vol_63"),
        "intraday_reversal_5": ("rev_5", "vol_21"),
        "market_drawdown_x_momentum": ("mom_126_21", "mom_252_21"),
        "market_vol_ratio_x_reversal": ("rev_5", "vol_252"),
        "low_extension": ("dist_52w_low", "vol_21"),
        "overnight_intraday_tug_of_war_63": ("mom_126_21", "rev_21"),
        "overnight_reversal_5": ("rev_5", "rev_21"),
        "overnight_return_21": ("mom_126_21", "rev_21"),
        "price_vs_sma20_atr": ("bollinger_pctb_20_2", "price_to_ma_200"),
        "price_vs_sma60_atr": ("dist_52w_high", "mom_126_21"),
        "previous_close_excursion_intraday_adjusted_square": ("vol_21", "vol_63"),
        "previous_close_excursion_intraday_cross": ("vol_21", "vol_63"),
        "previous_close_excursion_scaled_span_square": ("vol_21", "vol_63"),
        "range_position": ("bollinger_pctb_20_2", "stoch_k_14"),
        "recovery_ratio_252": ("dist_52w_high", "drawdown_252"),
        "residual_reversal_vol_scaled_5": ("rev_5", "vol_63"),
        "return_run_length_21": ("mom_126_21", "rev_21"),
        "sector_leader_lag_5": ("dollar_volume_21", "mom_126_21"),
        "session_dollar_volume": ("dollar_volume_21", "dollar_volume_252"),
        "session_span": ("vol_21", "vol_63"),
        "sessions_since_252_high": ("dist_52w_high", "drawdown_252"),
        "tail_resilience_252": ("beta_252", "idio_vol_252"),
    }
)


def _clock(recipe: FactorSpec) -> FactorFormulaClock:
    # Every installed extension recipe reads a contiguous session window; the one
    # calendar-selected Formula lives in the shipped catalog, not here.
    observation_clock = observation_clock_for(recipe)
    if recipe.factor_id == "tail_resilience_252":
        return FactorFormulaClock(
            observation_clock=observation_clock,
            source_interval="[t-252,t]",
            estimation_interval="OLS returns [t-251,t]",
            event_interval="worst 25 Market sessions in [t-251,t-5], date tie-break",
            recovery_interval="[s,s+5], latest end t",
            minimum_ordered_source_rows=int(recipe.minimum_observations),
        )
    if recipe.factor_id == "sector_leader_lag_5":
        return FactorFormulaClock(
            observation_clock=observation_clock,
            source_interval="[t-25,t]",
            estimation_interval=(
                "ADV21 [t-25,t-5] inclusive; leader formation ell=t-5; five-session return (t-5,t]"
            ),
            minimum_ordered_source_rows=int(recipe.minimum_observations),
        )
    return FactorFormulaClock(
        observation_clock=observation_clock,
        source_interval=(
            f"{recipe.minimum_observations} ordered rows, window "
            f"{observation_clock.source_interval}"
        ),
        minimum_ordered_source_rows=int(recipe.minimum_observations),
    )


def _formation_cutoff(recipe: FactorSpec) -> str:
    """State the observation clock in one sentence, derived rather than asserted.

    The previous constant read "Formation t consumes no observation later than
    t-1" for every Formula, which was true of the implementation and wrong about
    the strategy: a Feature named ``current`` was one session stale, and the
    sentence made that look intended.
    """
    clock = observation_clock_for(recipe)
    skip = clock.formula_skip_sessions
    latest = "t" if not skip else f"t-{skip}"
    reason = (
        "no economic skip" if not skip else f"the Formula's declared {skip}-session economic skip"
    )
    return (
        f"Observation session t consumes no source row later than {latest} ({reason}). "
        f"When that value may be read is the source availability policy's answer, not "
        f"this Formula's; entry and exit belong to the execution recipe."
    )


def _specification(
    recipe: FactorSpec,
    registry: FeatureKernelRegistry,
    *,
    goldens: tuple[FactorFormulaGoldenExample, ...] | None = None,
) -> FactorFormulaSpecification:
    return FactorFormulaSpecification.create(
        factor_id=recipe.factor_id,
        scientific_question=f"Does {recipe.factor_id} add causal incremental evidence?",
        formula_ref=recipe.formula_ref,
        implementation_hash=str(
            registry.implementation_hash(recipe, core_bundle=desktop_core_feature_bundle())
        ),
        source_fields=tuple(recipe.required_fields),
        price_basis=recipe.return_convention,
        formation_cutoff=_formation_cutoff(recipe),
        clock=_clock(recipe),
        formula=recipe.formula,
        sign_convention="The sign is the literal sign stated by the formula.",
        expected_units="Dimensionless return, ratio, or beta difference as stated by the formula.",
        missing_value_behavior=(
            "Any required non-finite input makes the strict window missing.",
            "Non-positive log-price legs and non-positive denominators are missing.",
            "One source row below the mechanical minimum is missing.",
        ),
        finite_policy="Every output is finite or missing; infinities are refused.",
        preprocessing_role=_ROLE_BY_FACTOR.get(recipe.factor_id, ROBUST_SECTOR_NEUTRAL_Z_ROLE),
        golden_examples=_goldens(recipe) if goldens is None else goldens,
        absolute_tolerance=max(float(recipe.absolute_tolerance), 1e-12),
        relative_tolerance=max(float(recipe.relative_tolerance), 1e-12),
    )


class FactorFormulaSpecificationCatalog:
    """Resolve installed Formula Specifications by stable factor identifier."""

    def __init__(self, specifications: Iterable[FactorFormulaSpecification]) -> None:
        """Index installed Formula Specifications and reject duplicate factor identifiers."""
        ordered = tuple(specifications)
        indexed = {item.factor_id: item for item in ordered}
        if len(indexed) != len(ordered):
            raise FeatureFormulaSpecificationError(
                "feature_engine.formula_specification_duplicated"
            )
        self._by_factor = MappingProxyType(indexed)

    @property
    def factor_ids(self) -> tuple[str, ...]:
        """Return the installed factor identifiers in stable order."""
        return tuple(sorted(self._by_factor))

    def resolve(self, factor_id: str) -> FactorFormulaSpecification:
        """Return an installed Formula Specification or reject an unknown factor."""
        try:
            return self._by_factor[factor_id]
        except KeyError as error:
            raise FeatureFormulaSpecificationError(
                "feature_engine.formula_specification_not_installed"
            ) from error

    def validate_registration(
        self,
        recipe: FactorSpec,
        *,
        registry: FeatureKernelRegistry,
    ) -> FactorFormulaSpecification:
        """Re-derive code and recipe authority for one installed formula.

        This validates a Formula Specification and executable registration; it
        deliberately does not make the later development-admission decision.
        """
        specification = self.resolve(recipe.factor_id)
        measured = registry.implementation_hash(recipe, core_bundle=desktop_core_feature_bundle())
        if (
            specification.status != "COMPLETE"
            or specification.factor_id != recipe.factor_id
            or specification.formula_ref != recipe.formula_ref
            or specification.formula != recipe.formula
            or specification.source_fields != tuple(recipe.required_fields)
            or specification.minimum_ordered_source_rows != recipe.minimum_observations
            or specification.implementation_hash != measured
        ):
            raise FeatureFormulaSpecificationError(
                "feature_engine.formula_specification_registration_mismatch"
            )
        registry.resolve(recipe.formula_ref)
        return specification

    def admit(
        self,
        recipe: FactorSpec,
        *,
        registry: FeatureKernelRegistry,
    ) -> FactorFormulaSpecification:
        """Compatibility alias for formula-registration validation only."""
        return self.validate_registration(recipe, registry=registry)


def build_installed_factor_formula_specifications() -> FactorFormulaSpecificationCatalog:
    """Build the catalog from installed extension recipes and kernel registrations."""
    registry = default_extension_kernel_registry()
    return FactorFormulaSpecificationCatalog(
        _specification(recipe, registry) for recipe in extension_factor_specs()
    )


def build_research_formula_specification(
    recipe: FactorSpec, *, source_session_count: int, preprocessing_recipe: str | None = None
) -> FactorFormulaSpecification:
    """Resolve local meaning without promoting an arbitrary registered kernel.

    Exact installed recipes (including a local alias) retain their complete
    specification and preprocessing role. The existing mean-return kernel has
    a parameterized, independently checkable geometric-price example. Other
    numerical variants need their own complete specification; registration or
    a successful raw calculation is not that authority.
    """
    if recipe.formula_ref in FORMULA_IDS:
        return _formula_research_specification(
            recipe,
            source_session_count=source_session_count,
            preprocessing_recipe=preprocessing_recipe,
        )
    if recipe.minimum_observations > source_session_count:
        raise FeatureFormulaSpecificationError("feature_research.formula_source_window_unavailable")
    registry = default_extension_kernel_registry()
    measured = registry.implementation_hash(recipe, core_bundle=desktop_core_feature_bundle())
    for installed in extension_factor_specs():
        if numerical_spec_hash(installed) == numerical_spec_hash(recipe):
            original = _specification(installed, registry)
            if (
                original.absolute_tolerance != recipe.absolute_tolerance
                or original.relative_tolerance != recipe.relative_tolerance
            ):
                raise FeatureFormulaSpecificationError(
                    "feature_research.formula_validation_policy_change_requires_review"
                )
            values = original.model_dump(mode="python", exclude={"specification_hash"})
            values.update(
                factor_id=recipe.factor_id,
                implementation_hash=measured,
                scientific_question=f"Does {recipe.factor_id} add causal incremental evidence?",
                golden_examples=original.golden_examples,
                clock=FactorFormulaClock(
                    **{
                        **original.clock.model_dump(mode="python"),
                        "observation_clock": observation_clock_for(recipe),
                    }
                ),
            )
            return FactorFormulaSpecification.create(**values)
    if (
        recipe.formula_ref != MEAN_ADJUSTED_RETURN_ID
        or recipe.required_fields != ("provider_adjusted_close",)
        or recipe.return_convention != "provider_adjusted_log_return"
        or recipe.minimum_observations != recipe.window_sessions + recipe.lag_sessions + 1
        or recipe.absolute_tolerance <= 0
        or recipe.relative_tolerance <= 0
    ):
        raise FeatureFormulaSpecificationError(
            "feature_research.complete_formula_specification_required"
        )
    # Exact geometric ratios: every required log return is log(1.001). The
    # expected value is stated analytically, never measured from this kernel.
    rows = recipe.minimum_observations
    prices = tuple(math.exp(index * math.log1p(0.001)) for index in range(rows))
    examples = (
        FactorFormulaGoldenExample(
            label="first-valid-source-boundary",
            rationale=(
                "Geometric adjusted prices have a constant log return across "
                "the declared window and economic lag."
            ),
            ordered_inputs=(
                FactorFormulaGoldenSeries(field="provider_adjusted_close", values=prices),
            ),
            expected_value=math.log1p(0.001),
        ),
        FactorFormulaGoldenExample(
            label="one-row-short-is-missing",
            rationale="One missing earliest price leaves the lagged return window incomplete.",
            ordered_inputs=(
                FactorFormulaGoldenSeries(field="provider_adjusted_close", values=prices[:-1]),
            ),
            expected_value=None,
        ),
    )
    result = _specification(recipe, registry, goldens=examples)
    # Unlike historical installed-spec construction, a new local declaration
    # must not have its stated tolerance relaxed by a display-oriented floor.
    values = result.model_dump(mode="python", exclude={"specification_hash"})
    values.update(
        absolute_tolerance=recipe.absolute_tolerance,
        relative_tolerance=recipe.relative_tolerance,
        clock=result.clock,
        golden_examples=result.golden_examples,
        executable_definition=(
            "mean(log(provider_adjusted_close[t-lag-j] / "
            "provider_adjusted_close[t-lag-j-1]), j=0..window-1); "
            f"window={recipe.window_sessions}, lag={recipe.lag_sessions}"
        ),
        sign_convention="Positive for rising adjusted prices; defined by the installed kernel.",
        expected_units="Mean provider-adjusted log return per session.",
    )
    return FactorFormulaSpecification.create(**values)


def _formula_research_specification(
    recipe: FactorSpec, *, source_session_count: int, preprocessing_recipe: str | None
) -> FactorFormulaSpecification:
    """A formula factor's complete specification (EX, V88).

    The formula decides it: the kept spec is the one the plan derived, and the goldens state the
    language's reference evaluation, never the kernel's, on deterministic source rows.
    """
    try:
        derived = formula_specification(recipe)
        role = formula_preprocessing(preprocessing_recipe)
    except FormulaError as error:
        raise FeatureFormulaSpecificationError(str(error)) from error
    if derived != recipe:
        raise FeatureFormulaSpecificationError("feature_research.formula_specification_not_derived")
    if recipe.absolute_tolerance <= 0 or recipe.relative_tolerance <= 0:
        raise FeatureFormulaSpecificationError(
            "feature_research.complete_formula_specification_required"
        )
    if recipe.minimum_observations > source_session_count:
        raise FeatureFormulaSpecificationError("feature_research.formula_source_window_unavailable")
    examples = tuple(
        FactorFormulaGoldenExample(
            label=label,
            rationale=rationale,
            ordered_inputs=tuple(
                FactorFormulaGoldenSeries(field=field, values=values)
                for field, values in source.items()
            ),
            expected_value=expected,
        )
        for label, rationale, source, expected in formula_goldens(recipe)
    )
    result = _specification(recipe, default_extension_kernel_registry(), goldens=examples)
    values = result.model_dump(mode="python", exclude={"specification_hash"})
    values.update(
        absolute_tolerance=recipe.absolute_tolerance,
        relative_tolerance=recipe.relative_tolerance,
        clock=result.clock,
        golden_examples=result.golden_examples,
        # The recipe the formula's declaration chose, never a default (EX; which factors want
        # the Sector demean is a question of shape, the research note's).
        preprocessing_role=role,
        executable_definition=f"{recipe.formula}; skip={formula_skip_sessions(recipe)}",
        sign_convention="The sign the formula states.",
        expected_units=(
            "None: price and volume at equal exponents and the adjusted close at none, so no "
            "later split or dividend moves a value."
            if recipe.formula_ref not in FORMULA_POINT_IN_TIME_IDS
            else "As the formula states: its point-in-time leaves read each session's values as "
            "they traded, re-based to the session a value is for, so no later split moves a "
            "value; any split-basis leaf it reads holds price and volume at equal exponents and "
            "the adjusted close at none."
        ),
        missing_value_behavior=(
            "A window missing any row is missing; so are x / 0, log or sqrt outside its domain "
            "and a statistic whose deviation is 0.",
            "Missing propagates; a comparison or logic operand that is missing makes it missing.",
            "One source row below the formula's window and skip is missing.",
        ),
    )
    return FactorFormulaSpecification.create(**values)


def build_installed_factor_development_capabilities() -> tuple[FactorDevelopmentCapability, ...]:
    """Build the installed extension factors' declared development capabilities."""
    specifications = build_installed_factor_formula_specifications()
    capabilities = tuple(
        FactorDevelopmentCapability.create(
            factor_id=recipe.factor_id,
            formula_specification_hash=specifications.resolve(recipe.factor_id).specification_hash,
            implementation_hash=specifications.resolve(recipe.factor_id).implementation_hash,
            preprocessing_role=specifications.resolve(recipe.factor_id).preprocessing_role,
            nearest_neighbor_factor_ids=_NEIGHBOURS[recipe.factor_id],
        )
        for recipe in extension_factor_specs()
    )
    controls = set(desktop_core_feature_bundle().factor_ids)
    unknown = {
        item.factor_id: tuple(sorted(set(item.nearest_neighbor_factor_ids) - controls))
        for item in capabilities
        if set(item.nearest_neighbor_factor_ids) - controls
    }
    if unknown:
        raise FeatureFormulaSpecificationError(
            "feature_engine.factor_development_neighbour_not_installed"
        )
    return capabilities


def admit_factor_development_capabilities(
    *,
    lagged_sector_membership_available: bool = False,
    admitted_preprocessing_roles: tuple[str, ...] = (
        ROBUST_SECTOR_NEUTRAL_Z_ROLE,
        STATE_INTERACTION_BLOCK_ROLE,
        TIME_SERIES_ABSOLUTE_STATE_ROBUST_ROLE,
    ),
) -> tuple[FactorDevelopmentAdmissionReceipt, ...]:
    """Decide which installed factor capabilities development may use.

    The Host checks lagged sector membership and admitted preprocessing roles
    before sealing one admission receipt per capability.
    """
    specifications = build_installed_factor_formula_specifications()
    registry = default_extension_kernel_registry()
    recipes = {item.factor_id: item for item in extension_factor_specs()}
    receipts = []
    for capability in build_installed_factor_development_capabilities():
        specifications.validate_registration(recipes[capability.factor_id], registry=registry)
        if capability.factor_id == "sector_leader_lag_5" and not lagged_sector_membership_available:
            disposition, reason = "REFUSED", "LAGGED_SECTOR_MEMBERSHIP_UNAVAILABLE"
        elif capability.preprocessing_role not in admitted_preprocessing_roles:
            disposition, reason = "REFUSED", "PREPROCESSING_CAPABILITY_NOT_ADMITTED"
        else:
            disposition, reason = "ADMITTED", "FORMULA_IMPLEMENTATION_AND_PREPROCESSING_VERIFIED"
        receipts.append(
            FactorDevelopmentAdmissionReceipt.create(
                factor_id=capability.factor_id,
                capability_hash=capability.capability_hash,
                disposition=disposition,
                reason=reason,
            )
        )
    return tuple(receipts)


def admitted_extension_factor_specs(
    *,
    receipts: tuple[FactorDevelopmentAdmissionReceipt, ...] | None = None,
    catalog: FactorFormulaSpecificationCatalog | None = None,
    registry: FeatureKernelRegistry | None = None,
    recipes: tuple[FactorSpec, ...] | None = None,
) -> tuple[FactorSpec, ...]:
    """Return extension recipes admitted by their capability receipts.

    When a caller supplies explicit catalog, registry and recipes, validate that
    set together; otherwise use the installed set.
    """
    if any(value is not None for value in (catalog, registry, recipes)):
        if catalog is None or registry is None or recipes is None or receipts is not None:
            raise FeatureFormulaSpecificationError(
                "feature_engine.external_formula_admission_arguments_invalid"
            )
        return tuple(
            recipe for recipe in recipes if catalog.validate_registration(recipe, registry=registry)
        )
    decisions = receipts or admit_factor_development_capabilities()
    admitted = {item.factor_id for item in decisions if item.disposition == "ADMITTED"}
    return tuple(recipe for recipe in extension_factor_specs() if recipe.factor_id in admitted)


def golden_example_frame(
    example: FactorFormulaGoldenExample, *, listing_id: str = "GOLDEN"
) -> pd.DataFrame:
    """Build the deterministic source frame for a Formula golden example.

    Give each listing a stable suffix under the requested identifier prefix.
    """
    rows = example.row_count
    frames: list[pd.DataFrame] = []
    for position in range(example.listing_count):
        columns: dict[str, object] = {
            "listing_id": [f"{listing_id}{position:02d}"] * rows,
            "session_date": pd.date_range("2024-01-01", periods=rows, freq="D").date,
        }
        for series in example.ordered_inputs:
            values: np.ndarray = np.asarray(series.values, dtype=float)
            if series.field == "volume_raw":
                values = values * (position + 1)
            elif series.field in {
                "close_raw",
                "provider_adjusted_close",
            }:
                values = values * (1.0 + 0.001 * position)
            columns[series.field] = values.tolist()
        frames.append(pd.DataFrame(columns))
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "ROBUST_SECTOR_NEUTRAL_Z_ROLE",
    "STATE_INTERACTION_BLOCK_ROLE",
    "TIME_SERIES_ABSOLUTE_STATE_ROBUST_ROLE",
    "FactorDevelopmentAdmissionReceipt",
    "FactorDevelopmentCapability",
    "FactorFormulaClock",
    "FactorFormulaGoldenExample",
    "FactorFormulaGoldenSeries",
    "FactorFormulaSpecification",
    "FactorFormulaSpecificationCatalog",
    "FeatureFormulaSpecificationError",
    "admit_factor_development_capabilities",
    "admitted_extension_factor_specs",
    "build_installed_factor_development_capabilities",
    "build_installed_factor_formula_specifications",
    "build_research_formula_specification",
    "extension_factor_specs",
    "golden_example_frame",
]
