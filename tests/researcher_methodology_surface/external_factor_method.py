"""One Factor method defined entirely outside the product package.

This is the external consumer the Factor Research platform claims to support: a
formula, a typed recipe, a Formula Specification and a kernel registration, all
of them written here, none of them requiring an edit to ``alphalattice``. If a
method family can only be added by someone who can commit to the product tree,
the extension surface is a description of the product rather than a platform.

Everything is injected and nothing is installed. The product's own kernel
registry does not resolve ``factor.case-study.external.range_position.v1``, its
Formula Specification catalog does not contain ``zz_case_study_range_position_5``,
and its authoring report never mentions either. The acceptance beside this module
asserts exactly that, because "installed in the product" is the one claim a case
study must never be able to make by accident.

The method is deliberately trivial as science and deliberately awkward as
plumbing. It reads three columns rather than two, so it exercises a golden schema
that used to be hard-coded to an open/close pair; and its observation is a single
session rather than a gap, so it exercises the row arithmetic that used to assume
every observation spanned two. Those are the two places the specification
contract fitted exactly one Factor, and a tiny external method is what makes the
difference visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.catalog.contracts import (
    FeatureCatalog,
    desktop_core_feature_bundle,
)
from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    observation_clock_for,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import (
    FeatureKernelRegistry,
    RegisteredFeatureKernel,
)
from alphalattice.foundation.feature_engine.producers.factors.specifications import (
    ROBUST_SECTOR_NEUTRAL_Z_ROLE,
    FactorFormulaClock,
    FactorFormulaGoldenExample,
    FactorFormulaGoldenSeries,
    FactorFormulaSpecification,
    FactorFormulaSpecificationCatalog,
    build_installed_factor_formula_specifications,
)
from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack
from alphalattice.kernel.shared_kernel.identity import canonical_hash

EXTERNAL_METHOD_FAMILY = "CASE_STUDY_SESSION_RANGE"
EXTERNAL_METHOD_FAMILY_OWNERS = (__name__,)
"""This module, named the way the import system names it.

The identity owner resolves owners through ``find_spec``, so an external family
measures its implementation closure exactly the way a product family does. That
is the property worth proving: an outside method gets a *measured* identity, not
a declared one it could keep through an arbitrary rewrite.
"""

EXTERNAL_IMPLEMENTATION_ID = "factor.case-study.external.range_position.v1"
EXTERNAL_FACTOR_ID = "zz_case_study_range_position_5"

EXTERNAL_REQUIRED_FIELDS = (
    "close_split_adjusted",
    "high_split_adjusted",
    "low_split_adjusted",
)

EXTERNAL_WINDOW_SESSIONS = 5
EXTERNAL_FORMULA_SKIP_SESSIONS = 1
"""This external method's declared *economic* skip, kept deliberately non-zero.

Every product Formula declares no skip, so a fixture that also declared none
would leave the platform's ability to carry one untested. Here the consumer's
own methodology excludes the formation session's range on purpose, and the
installed observation clock has to represent that without turning it back into
the execution-safety lag the successor removed.
"""

EXTERNAL_OBSERVATION_SPANS_SESSIONS = 1
EXTERNAL_MINIMUM_ORDERED_SOURCE_ROWS = (
    EXTERNAL_WINDOW_SESSIONS
    + EXTERNAL_FORMULA_SKIP_SESSIONS
    + EXTERNAL_OBSERVATION_SPANS_SESSIONS
    - 1
)

_DECLARATION = {
    "implementation_id": EXTERNAL_IMPLEMENTATION_ID,
    "algorithm": "mean(((close - low) / (high - low)).shift(skip), window=spec.window_sessions)",
    "degenerate_or_non_finite_range": "nan",
    "numeric_domain": "float64",
    "grouping": "listing_id",
    "ordering": "session_date",
    "price_basis": "split_adjusted",
}


def range_position(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Where the close sat inside its own session range, averaged over the window.

    Zero and one are the extremes of a real session, so the quantity is bounded
    by construction and a value outside ``[0, 1]`` would mean the inputs were not
    one session's high, low and close. A session whose range is zero or not
    finite has no position to report -- the close is simultaneously at the high
    and at the low -- so it fails to missing rather than to an arbitrary 0.5.
    """

    ordered = source.assign(_source_position=np.arange(len(source), dtype=int)).sort_values(
        ["listing_id", "session_date"]
    )
    close = pd.to_numeric(ordered["close_split_adjusted"], errors="coerce").astype(float)
    high = pd.to_numeric(ordered["high_split_adjusted"], errors="coerce").astype(float)
    low = pd.to_numeric(ordered["low_split_adjusted"], errors="coerce").astype(float)
    span = high - low
    usable = (
        np.isfinite(close) & np.isfinite(high) & np.isfinite(low) & np.isfinite(span) & (span > 0.0)
    )
    position = ((close - low) / span).where(usable)
    values = position.groupby(ordered["listing_id"], sort=False).transform(
        lambda item: (
            item.shift(specification.lag_sessions)
            .rolling(
                specification.window_sessions,
                min_periods=specification.window_sessions,
            )
            .mean()
        )
    )
    ordered["_computed"] = values.to_numpy(dtype=float)
    restored = ordered.sort_values("_source_position")
    return pd.Series(restored["_computed"].to_numpy(dtype=float), index=source.index)


def external_factor_spec() -> FactorSpec:
    """The typed recipe an external catalog revision installs."""

    return FactorSpec(
        factor_id=EXTERNAL_FACTOR_ID,
        family=FactorFamily.TECHNICAL,
        formula_ref=EXTERNAL_IMPLEMENTATION_ID,
        formula=(
            "mean((close_t - low_t)/(high_t - low_t)) over 5 sessions ending at t-1; "
            "the formation session's own range is excluded by this method's "
            "declared 1-session economic skip"
        ),
        window_sessions=EXTERNAL_WINDOW_SESSIONS,
        lag_sessions=EXTERNAL_FORMULA_SKIP_SESSIONS,
        return_convention="split_adjusted_session_range_position",
        required_fields=EXTERNAL_REQUIRED_FIELDS,
        literature_sources=("case-study://researcher-methodology-surface/external-consumer",),
        minimum_observations=EXTERNAL_MINIMUM_ORDERED_SOURCE_ROWS,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        track=FactorTrack.MODEL,
        core_anchor=False,
    )


def external_kernel_registry() -> FeatureKernelRegistry:
    """The product's installed kernels plus this one, composed by the consumer.

    Composed rather than replaced: an external method is an addition to a build,
    and a registry that dropped the product's own kernels would be testing a
    different product. The product's ``default_extension_kernel_registry`` is
    untouched and still resolves neither this implementation id nor this factor.
    """

    product = default_extension_kernel_registry()
    installed = tuple(
        product.resolve(implementation_id)
        for implementation_id in (
            "factor.desktop.experimental.mean_adjusted_return.v1",
            "factor.desktop.experimental.overnight_return.v1",
        )
    )
    return FeatureKernelRegistry(
        (
            *installed,
            RegisteredFeatureKernel(
                implementation_id=EXTERNAL_IMPLEMENTATION_ID,
                method_family=EXTERNAL_METHOD_FAMILY,
                method_family_owners=EXTERNAL_METHOD_FAMILY_OWNERS,
                required_fields=EXTERNAL_REQUIRED_FIELDS,
                implementation_hash=canonical_hash(dict(_DECLARATION)),
                compute=range_position,
            ),
        )
    )


def external_formula_specification() -> FactorFormulaSpecification:
    """The consumer's own frozen methodology, admitted in the consumer's catalog.

    ``ADMITTED`` here and nowhere else. Admission is a property of the catalog
    that grants it, and the catalog granting it is the one this module builds --
    the product's installed specification catalog neither contains this Factor
    nor could be made to by anything written here. The distinction is the whole
    point of installed-versus-admitted, and it is what lets a case study prove
    the gate works without the case study becoming an admission.

    The goldens are analytic. A session whose high, low and close are 110, 90 and
    100 has its close exactly halfway up its range, so a window of identical
    sessions averages to 0.5 whatever the code does.
    """

    recipe = external_factor_spec()
    registry = external_kernel_registry()
    flat = _golden_series(
        rows=EXTERNAL_MINIMUM_ORDERED_SOURCE_ROWS, close=100.0, high=110.0, low=90.0
    )
    return FactorFormulaSpecification.create(
        factor_id=recipe.factor_id,
        scientific_question=(
            "Does where a listing closes inside its own session range, averaged over five "
            "lagged sessions, carry cross-sectional information the installed momentum and "
            "reversal factors do not already carry?"
        ),
        formula_ref=recipe.formula_ref,
        implementation_hash=str(
            registry.implementation_hash(recipe, core_bundle=desktop_core_feature_bundle())
        ),
        source_fields=tuple(recipe.required_fields),
        price_basis=recipe.return_convention,
        formation_cutoff=(
            "Observation session t consumes no source row later than t-1 (the Formula's "
            "declared 1-session economic skip). When that value may be read is the source "
            "availability policy's answer, not this Formula's; entry and exit belong to "
            "the execution recipe."
        ),
        clock=FactorFormulaClock(
            observation_clock=observation_clock_for(recipe),
            source_interval="[t-5,t-1]",
            estimation_interval="five one-session range observations [t-5,t-1]",
            minimum_ordered_source_rows=int(recipe.minimum_observations),
        ),
        missing_value_behavior=(
            "A session whose high, low or close is not finite yields a missing position.",
            "A session whose range is zero yields a missing position rather than a "
            "midpoint: the close is at the high and at the low at once, and 0.5 would be "
            "an invented answer.",
            "A window with any missing position yields a missing factor value.",
            "A listing with fewer than 6 ordered source rows has no value at all: five "
            "observations plus the one session the declared skip excludes.",
        ),
        formula=recipe.formula,
        sign_convention=(
            "Higher means the listing closed nearer the top of its own range on average."
        ),
        expected_units="Fraction of the session range, dimensionless, bounded to [0, 1].",
        finite_policy=(
            "Every produced value is finite or missing. Enforced by screening the range "
            "before the division rather than by assuming a positive range."
        ),
        preprocessing_role=ROBUST_SECTOR_NEUTRAL_Z_ROLE,
        golden_examples=(
            FactorFormulaGoldenExample(
                label="first-valid-source-boundary",
                rationale=(
                    "Every session closes exactly halfway up a range of 20, so the windowed "
                    "mean is 0.5 analytically. Exactly six ordered rows, which is the first "
                    "position at which any value exists."
                ),
                ordered_inputs=flat,
                expected_value=0.5,
            ),
            FactorFormulaGoldenExample(
                label="zero-range-session-is-missing",
                rationale=(
                    "One session with high equal to low sits inside the skipped window. Its "
                    "position is undefined rather than 0.5, so the window that reads it must "
                    "yield a missing value."
                ),
                ordered_inputs=_golden_series(
                    rows=EXTERNAL_MINIMUM_ORDERED_SOURCE_ROWS,
                    close=100.0,
                    high=110.0,
                    low=90.0,
                    degenerate_index=2,
                ),
                expected_value=None,
            ),
            FactorFormulaGoldenExample(
                label="one-row-short-is-missing",
                rationale=(
                    "Five ordered rows: the declared skip excludes one, leaving four "
                    "positions against the five the window requires. One row short of the "
                    "six the identity 5 + 1 + (1 - 1) demands."
                ),
                ordered_inputs=_golden_series(
                    rows=EXTERNAL_MINIMUM_ORDERED_SOURCE_ROWS - 1,
                    close=100.0,
                    high=110.0,
                    low=90.0,
                ),
                expected_value=None,
            ),
        ),
        absolute_tolerance=float(recipe.absolute_tolerance),
        relative_tolerance=float(recipe.relative_tolerance),
    )


def _golden_series(
    *,
    rows: int,
    close: float,
    high: float,
    low: float,
    degenerate_index: int | None = None,
) -> tuple[FactorFormulaGoldenSeries, ...]:
    """Three ordered columns, optionally with one session collapsed to a point."""

    highs = [high] * rows
    lows = [low] * rows
    closes = [close] * rows
    if degenerate_index is not None:
        highs[degenerate_index] = close
        lows[degenerate_index] = close
    return (
        FactorFormulaGoldenSeries(field="close_split_adjusted", values=tuple(closes)),
        FactorFormulaGoldenSeries(field="high_split_adjusted", values=tuple(highs)),
        FactorFormulaGoldenSeries(field="low_split_adjusted", values=tuple(lows)),
    )


def external_specification_catalog() -> FactorFormulaSpecificationCatalog:
    """The product's installed specifications plus the consumer's own.

    Built by composing the product catalog rather than by replacing it, so the
    admission gate is exercised against a catalog that still contains the
    product's own pending Factor. A catalog holding only the external method
    would prove the gate admits things and not that it still refuses.
    """

    product = build_installed_factor_formula_specifications()
    installed = tuple(product.resolve(factor_id) for factor_id in product.factor_ids)
    return FactorFormulaSpecificationCatalog((*installed, external_formula_specification()))


def external_feature_catalog() -> FeatureCatalog:
    """The current base activations plus this one external method, sorted.

    The shipped ``desktop-feature-catalog.json`` is not written to. This revision
    exists only in the workspace that installs it, which is why no published
    manifest and no current pointer moves.
    """

    shipped = FeatureCatalog.load()
    payload = shipped.to_payload()
    payload["factors"] = sorted(
        [*payload["factors"], external_factor_spec().model_dump(mode="json")],
        key=lambda factor: str(factor["factor_id"]),
    )
    return FeatureCatalog.from_payload(payload)


__all__ = [
    "EXTERNAL_FACTOR_ID",
    "EXTERNAL_FORMULA_SKIP_SESSIONS",
    "EXTERNAL_IMPLEMENTATION_ID",
    "EXTERNAL_METHOD_FAMILY",
    "EXTERNAL_METHOD_FAMILY_OWNERS",
    "EXTERNAL_MINIMUM_ORDERED_SOURCE_ROWS",
    "external_factor_spec",
    "external_feature_catalog",
    "external_formula_specification",
    "external_kernel_registry",
    "external_specification_catalog",
    "range_position",
]
