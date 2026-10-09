"""Prove the Risk estimator seam is the real execution path, not a parallel API.

A researcher adding a Risk estimator should write one method implementation, a
typed recipe envelope, and a catalog installation. This case study checks that
the active historical executor resolves through that seam, that the covariance
numbers are unchanged by the seam, and that a second adapter reaches an estimate
without editing any module under `risk_research`.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from alphalattice.investment.risk_research.contracts import (
    CausalRiskReturnChunk,
    CausalRiskReturnSurface,
    CovarianceDiagnostics,
    RiskUniverseEpoch,
    default_covariance_recipe,
    seal_contract,
)
from alphalattice.investment.risk_research.estimators.catalog import (
    RiskEstimatorCatalog,
    build_installed_risk_estimator_catalog,
)
from alphalattice.investment.risk_research.estimators.contracts import (
    SINGLE_THREAD_NUMERICAL_CAPABILITY,
    BoundRiskReturnInput,
    EstimatedCovariance,
    RiskEstimatorNumericalBinding,
    RiskEstimatorRecipeEnvelope,
    implementation_content_hash,
)
from alphalattice.investment.risk_research.estimators.covariance import (
    COVARIANCE_ADAPTER_ID,
    COVARIANCE_RECIPE_SCHEMA_ID,
    estimate_dynamic_covariance,
)
from alphalattice.investment.risk_research.estimators.matrix_identity import matrix_content_hash

type FloatArray = NDArray[np.float64]

_SESSION_COUNT = 337
_LISTING_IDS = ("listing-a", "listing-b", "listing-c")
_SECTORS = {"listing-a": "sector-1", "listing-b": "sector-1", "listing-c": "sector-2"}


class _FixtureReturnReader:
    def __init__(self, sessions: tuple[date, ...], values: FloatArray) -> None:
        self.sessions = sessions
        self.values = values

    def available_sessions(self, _surface: CausalRiskReturnSurface) -> tuple[date, ...]:
        return self.sessions

    def read_sessions(
        self,
        _surface: CausalRiskReturnSurface,
        sessions: tuple[date, ...],
    ) -> FloatArray:
        positions = {session: index for index, session in enumerate(self.sessions)}
        output = self.values[[positions[session] for session in sessions]]
        output.setflags(write=False)
        return output


def _fixture_surface() -> tuple[CausalRiskReturnSurface, _FixtureReturnReader, FloatArray]:
    sessions = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(_SESSION_COUNT))
    epoch = seal_contract(
        RiskUniverseEpoch,
        "epoch_hash",
        market_profile_id="test-us",
        universe_manifest_revision="a" * 64,
        panel_snapshot_hash="b" * 64,
        ordered_listing_ids=_LISTING_IDS,
    )
    chunk = CausalRiskReturnChunk(
        year=2024,
        row_count=len(sessions) * len(_LISTING_IDS),
        first_formation_session=sessions[0],
        last_formation_session=sessions[-1],
        content_hash="c" * 64,
        metadata_hash="d" * 64,
        uri="fixture://risk-returns",
    )
    surface = seal_contract(
        CausalRiskReturnSurface,
        "surface_hash",
        epoch=epoch,
        first_formation_session=sessions[0],
        last_formation_session=sessions[-1],
        formation_count=len(sessions),
        source_watermark_hash="e" * 64,
        chunks=(chunk,),
        limitations=("fixture",),
    )
    rng = np.random.default_rng(19)
    values: FloatArray = rng.normal(0.0, 0.02, size=(len(sessions), len(_LISTING_IDS)))
    return surface, _FixtureReturnReader(sessions, values), values


class _CountingCatalog(RiskEstimatorCatalog):
    """Observe that the active executor really resolves through the catalog."""

    def __init__(self) -> None:
        from alphalattice.investment.risk_research.estimators.covariance import (
            CovarianceEstimatorAdapter,
        )

        super().__init__((CovarianceEstimatorAdapter(),))
        self.resolved_adapter_ids: list[str] = []

    def resolve(self, recipe: RiskEstimatorRecipeEnvelope):  # type: ignore[no-untyped-def]
        adapter = super().resolve(recipe)
        self.resolved_adapter_ids.append(adapter.adapter_id)
        return adapter


class _DiagonalVarianceAdapter:
    """A second, deliberately simple estimator installed only by this case study."""

    adapter_id = "case-study-diagonal-variance"
    recipe_schema_id = "CASE_STUDY_DIAGONAL_VARIANCE"

    def describe_numerical_binding(self) -> RiskEstimatorNumericalBinding:
        return RiskEstimatorNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimate_content_format_id="risk-covariance-dense-symmetric-float64",
            implementation_owners=("case-study.researcher-methodology-surface",),
            # A capability installed from outside the package states its content
            # exactly as the built-in one does: the bytes of its own module.
            implementation_content_hash=implementation_content_hash(Path(__file__)),
            deterministic_policy={"estimator": "sample-diagonal-variance"},
            required_runtime_capabilities=(SINGLE_THREAD_NUMERICAL_CAPABILITY,),
        )

    def validate_recipe(self, recipe: RiskEstimatorRecipeEnvelope) -> dict[str, object]:
        if recipe.adapter_id != self.adapter_id:
            raise ValueError("CASE_STUDY_ESTIMATOR_ROUTE_INVALID")
        if recipe.recipe_schema_id != self.recipe_schema_id:
            raise ValueError("CASE_STUDY_ESTIMATOR_SCHEMA_INVALID")
        return dict(recipe.parameters)

    def estimate(
        self,
        *,
        recipe: RiskEstimatorRecipeEnvelope,
        inputs: BoundRiskReturnInput,
    ) -> EstimatedCovariance:
        self.validate_recipe(recipe)
        values = np.asarray(inputs.returns, dtype=np.float64)
        variance: FloatArray = np.maximum(np.var(values, axis=0, ddof=1), 1e-10)
        covariance: FloatArray = np.diag(variance)
        volatility: FloatArray = np.sqrt(variance)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        annualized = volatility * np.sqrt(252.0)
        diagnostics = CovarianceDiagnostics(
            formation_session=inputs.formation_session,
            asset_count=len(inputs.ordered_listing_ids),
            shrinkage=0.0,
            minimum_eigenvalue=float(eigenvalues[0]),
            maximum_eigenvalue=float(eigenvalues[-1]),
            condition_number=float(eigenvalues[-1] / eigenvalues[0]),
            trace=float(np.trace(covariance)),
            average_correlation=0.0,
            top_one_eigenvalue_share=float(eigenvalues[-1] / eigenvalues.sum()),
            top_five_eigenvalue_share=float(
                eigenvalues[-min(5, len(eigenvalues)) :].sum() / eigenvalues.sum()
            ),
            annualized_volatility_minimum=float(annualized.min()),
            annualized_volatility_median=float(np.median(annualized)),
            annualized_volatility_maximum=float(annualized.max()),
            numerical_environment_hash="0" * 64,
            matrix_hash=matrix_content_hash(covariance),
        )
        for array in (covariance, volatility, eigenvalues, eigenvectors):
            array.setflags(write=False)
        return EstimatedCovariance(
            matrix=covariance,
            forecast_volatility=volatility,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            diagnostics=diagnostics,
        )


def test_a_second_estimator_reaches_an_estimate_without_editing_risk_runtime() -> None:
    """Installing a new adapter is the only step needed to run a new method."""

    catalog = RiskEstimatorCatalog((_DiagonalVarianceAdapter(),))
    envelope = catalog.seal_recipe(
        recipe_schema_id=_DiagonalVarianceAdapter.recipe_schema_id,
        parameters={"variance_floor": 1e-10},
    )
    adapter = catalog.resolve(envelope)
    rng = np.random.default_rng(3)
    returns: FloatArray = rng.normal(0.0, 0.02, size=(315, len(_LISTING_IDS)))
    estimate = adapter.estimate(
        recipe=envelope,
        inputs=BoundRiskReturnInput.create(
            return_surface_hash="a" * 64,
            ordered_listing_ids=_LISTING_IDS,
            formation_session=date(2024, 6, 1),
            returns=returns,
        ),
    )

    assert estimate.matrix.shape == (len(_LISTING_IDS), len(_LISTING_IDS))
    assert np.array_equal(estimate.matrix, estimate.matrix.T)
    assert float(estimate.eigenvalues[0]) > 0.0
    assert estimate.diagnostics.matrix_hash == matrix_content_hash(estimate.matrix)

    # The Host catalog installs its own two real capabilities and neither of
    # them is this one: a case-study adapter never becomes reachable from a
    # production composition by existing.
    installed = build_installed_risk_estimator_catalog().adapter_ids
    assert COVARIANCE_ADAPTER_ID in installed
    assert _DiagonalVarianceAdapter.adapter_id not in installed


def test_catalog_identity_covers_every_installed_capability() -> None:
    from alphalattice.investment.risk_research.estimators.covariance import (
        CovarianceEstimatorAdapter,
    )
    from alphalattice.investment.risk_research.estimators.diagonal import (
        DIAGONAL_ADAPTER_ID,
        DIAGONAL_RECIPE_SCHEMA_ID,
    )
    from alphalattice.investment.risk_research.estimators.fast_slow import (
        FAST_SLOW_ADAPTER_ID,
        FAST_SLOW_RECIPE_SCHEMA_ID,
    )

    binding = build_installed_risk_estimator_catalog().binding
    by_adapter = {value.adapter_id: value for value in binding.ordered_capabilities}
    assert set(by_adapter) == {
        COVARIANCE_ADAPTER_ID,
        FAST_SLOW_ADAPTER_ID,
        DIAGONAL_ADAPTER_ID,
    }
    assert by_adapter[COVARIANCE_ADAPTER_ID].recipe_schema_id == COVARIANCE_RECIPE_SCHEMA_ID
    assert by_adapter[FAST_SLOW_ADAPTER_ID].recipe_schema_id == FAST_SLOW_RECIPE_SCHEMA_ID
    assert by_adapter[DIAGONAL_ADAPTER_ID].recipe_schema_id == DIAGONAL_RECIPE_SCHEMA_ID

    # Installing one more capability must move the catalog identity.
    extended = RiskEstimatorCatalog(
        (
            CovarianceEstimatorAdapter(),
            build_installed_risk_estimator_catalog().adapters[1],
            _DiagonalVarianceAdapter(),
        )
    )
    assert extended.binding.catalog_hash != binding.catalog_hash


def test_estimator_seam_fails_closed_before_any_numerical_call() -> None:
    catalog = build_installed_risk_estimator_catalog()

    with pytest.raises(ValueError, match="RISK_ESTIMATOR_RECIPE_SCHEMA_NOT_INSTALLED"):
        catalog.seal_recipe(recipe_schema_id="NOT_INSTALLED", parameters={})

    unknown = RiskEstimatorRecipeEnvelope.create(
        adapter_id="not-installed",
        recipe_schema_id=COVARIANCE_RECIPE_SCHEMA_ID,
        parameters={},
    )
    with pytest.raises(ValueError, match="RISK_ESTIMATOR_ADAPTER_NOT_INSTALLED"):
        catalog.resolve(unknown)

    # A same-shaped window on the wrong listing axis width is rejected by the
    # bound input, before an adapter is ever asked to estimate.
    rng = np.random.default_rng(11)
    with pytest.raises(ValueError, match="RISK_BOUND_RETURN_INPUT_INVALID"):
        BoundRiskReturnInput.create(
            return_surface_hash="a" * 64,
            ordered_listing_ids=_LISTING_IDS,
            formation_session=date(2024, 6, 1),
            returns=rng.normal(0.0, 0.02, size=(315, len(_LISTING_IDS) + 1)),
        )


# ------------------------------------------------------- the installed R1 method


def _fast_slow_returns(*, assets: int = 8, seed: int = 19) -> FloatArray:
    """A 315-session window with genuine cross-asset structure.

    A common factor plus idiosyncratic noise, so the Ledoit-Wolf shrinkage is
    non-trivial and the fast and slow windows disagree -- which is the only
    condition under which a blend weight is observable at all.
    """

    generator = np.random.default_rng(seed)
    common = generator.normal(0.0, 0.010, size=(315, 1))
    values: FloatArray = 0.6 * common + generator.normal(0.0, 0.012, size=(315, assets))
    return np.ascontiguousarray(values)


def _hand_built_fast_slow(returns: FloatArray, *, eta: float, decay: float) -> FloatArray:
    """The Formula Specification, rebuilt here from the documented steps.

    Written out rather than imported so the standardization recursion, the two
    Ledoit-Wolf windows, the convex blend, the normalization and the single
    volatility diagonal are each pinned by something that would not move with
    the implementation.
    """

    from sklearn.covariance import LedoitWolf

    variance = np.var(returns[:63], axis=0, ddof=1, dtype=np.float64)
    residuals: FloatArray = np.empty((252, returns.shape[1]), dtype=np.float64)
    for index, row in enumerate(returns[63:]):
        residuals[index] = row / np.sqrt(variance)
        variance = decay * variance + (1.0 - decay) * np.square(row)

    def _correlation(window: FloatArray) -> FloatArray:
        fitted = LedoitWolf(store_precision=False, assume_centered=False).fit(window)
        covariance = np.asarray(fitted.covariance_, dtype=np.float64)
        scale = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(scale, scale)
        return np.asarray((correlation + correlation.T) * 0.5, dtype=np.float64)

    blended = (1.0 - eta) * _correlation(residuals) + eta * _correlation(residuals[-63:])
    blend_scale = np.sqrt(np.diag(blended))
    correlation = blended / np.outer(blend_scale, blend_scale)
    correlation = (correlation + correlation.T) * 0.5
    volatility = np.sqrt(variance)
    covariance = correlation * np.outer(volatility, volatility)
    return np.asarray((covariance + covariance.T) * 0.5, dtype=np.float64)


@pytest.mark.parametrize("eta", (0.25, 0.50, 0.75))
def test_fast_slow_matches_its_formula_and_reports_both_shrinkages(eta: float) -> None:
    """The fast and slow covariance estimator matches its formula and reports both shrinkages."""

    from alphalattice.investment.risk_research.estimators.fast_slow import (
        FAST_SLOW_RECIPE_SCHEMA_ID,
    )

    returns = _fast_slow_returns()
    listings = tuple(f"listing-{index}" for index in range(returns.shape[1]))
    formation = date(2024, 3, 1)
    catalog = build_installed_risk_estimator_catalog()
    admission = catalog.admit_recipe(
        capability_handle=FAST_SLOW_RECIPE_SCHEMA_ID,
        parameters={"blend_weight": eta},
        seed=0,
    )
    estimate = catalog.resolve(admission.recipe).estimate(
        recipe=admission.recipe,
        inputs=BoundRiskReturnInput.create(
            return_surface_hash="a" * 64,
            ordered_listing_ids=listings,
            formation_session=formation,
            returns=returns,
        ),
    )

    expected = _hand_built_fast_slow(returns, eta=eta, decay=0.94)
    assert np.array_equal(estimate.matrix, expected)
    assert estimate.diagnostics.matrix_hash == matrix_content_hash(expected)

    components = dict(estimate.component_shrinkages)
    assert set(components) == {"slow_ledoit_wolf_shrinkage", "fast_ledoit_wolf_shrinkage"}
    # The slow component is the control's own fit on the control's own
    # residuals, so its intensity is exactly the control's shrinkage. A real
    # oracle rather than a restatement: it is what "differ only in correlation
    # lookback" means, and it would break the moment the standardization or the
    # Ledoit-Wolf settings diverged.
    control = estimate_dynamic_covariance(
        returns=returns,
        ordered_listing_ids=listings,
        formation_session=formation,
        recipe=default_covariance_recipe(),
    )
    assert components["slow_ledoit_wolf_shrinkage"] == control.diagnostics.shrinkage
    # The shorter window shrinks harder; that is the method's whole risk.
    assert components["fast_ledoit_wolf_shrinkage"] > components["slow_ledoit_wolf_shrinkage"]
    # The reported summary is the blend of the parts, weighted as they were.
    assert estimate.diagnostics.shrinkage == pytest.approx(
        (1.0 - eta) * components["slow_ledoit_wolf_shrinkage"]
        + eta * components["fast_ledoit_wolf_shrinkage"],
        rel=0.0,
        abs=1e-15,
    )
    # One volatility diagonal, applied once, shared with the control.
    assert np.array_equal(estimate.forecast_volatility, control.forecast_volatility)
    assert np.allclose(np.diag(estimate.matrix), np.square(estimate.forecast_volatility))
    assert estimate.diagnostics.numerical_environment_hash == (
        control.diagnostics.numerical_environment_hash
    )


def test_fast_slow_refuses_same_shaped_wrong_inputs_before_estimating() -> None:
    """The fast and slow covariance estimator refuses wrong inputs before estimating even when
    their shapes match."""

    from alphalattice.investment.risk_research.estimators.capability import RiskCapabilityError
    from alphalattice.investment.risk_research.estimators.covariance import RiskNumericalError
    from alphalattice.investment.risk_research.estimators.fast_slow import (
        FAST_SLOW_RECIPE_SCHEMA_ID,
        FastSlowCovarianceAdapter,
    )

    catalog = build_installed_risk_estimator_catalog()

    # A blend weight the declared domain does not admit.
    with pytest.raises(RiskCapabilityError):
        catalog.admit_recipe(
            capability_handle=FAST_SLOW_RECIPE_SCHEMA_ID,
            parameters={"blend_weight": 0.40},
            seed=0,
        )
    # An axis the domain never declared.
    with pytest.raises(RiskCapabilityError):
        catalog.admit_recipe(
            capability_handle=FAST_SLOW_RECIPE_SCHEMA_ID,
            parameters={"fast_correlation_weight": 0.25},
            seed=0,
        )
    # A seed a deterministic capability cannot consume.
    with pytest.raises(RiskCapabilityError):
        catalog.admit_recipe(
            capability_handle=FAST_SLOW_RECIPE_SCHEMA_ID,
            parameters={"blend_weight": 0.25},
            seed=7,
        )

    admission = catalog.admit_recipe(
        capability_handle=FAST_SLOW_RECIPE_SCHEMA_ID,
        parameters={"blend_weight": 0.25},
        seed=0,
    )
    adapter = FastSlowCovarianceAdapter()
    # A same-shaped envelope routed to the wrong adapter.
    forged = RiskEstimatorRecipeEnvelope.create(
        adapter_id=COVARIANCE_ADAPTER_ID,
        recipe_schema_id=FAST_SLOW_RECIPE_SCHEMA_ID,
        parameters=dict(admission.recipe.parameters),
    )
    with pytest.raises(RiskNumericalError):
        adapter.validate_recipe(forged)

    # A window of the wrong length, same-shaped in every other respect.
    short = _fast_slow_returns()[:314]
    listings = tuple(f"listing-{index}" for index in range(short.shape[1]))
    with pytest.raises(RiskNumericalError):
        adapter.estimate(
            recipe=admission.recipe,
            inputs=BoundRiskReturnInput.create(
                return_surface_hash="b" * 64,
                ordered_listing_ids=listings,
                formation_session=date(2024, 3, 1),
                returns=short,
            ),
        )


def test_installing_r1_leaves_the_control_method_identity_unmoved() -> None:
    """Installing an additional risk method leaves the control method's numerical identity
    unchanged."""

    from alphalattice.investment.risk_research.estimators.covariance import (
        CovarianceEstimatorAdapter,
    )
    from alphalattice.investment.risk_research.estimators.diagonal import (
        DiagonalShrunkCovarianceAdapter,
    )
    from alphalattice.investment.risk_research.estimators.fast_slow import (
        FastSlowCovarianceAdapter,
    )

    catalog = build_installed_risk_estimator_catalog()
    installed = {value.adapter_id: value for value in catalog.binding.ordered_capabilities}
    assert set(installed) == {
        COVARIANCE_ADAPTER_ID,
        FastSlowCovarianceAdapter().adapter_id,
        DiagonalShrunkCovarianceAdapter().adapter_id,
    }
    control_binding = CovarianceEstimatorAdapter().describe_numerical_binding()
    # R2 is the third install and the claim is unchanged by it: a method appended
    # to the catalog moves ``catalog_hash`` and nothing else. Its own content
    # hash differs from both predecessors, and the control's does not move.
    third_binding = DiagonalShrunkCovarianceAdapter().describe_numerical_binding()
    assert installed[DiagonalShrunkCovarianceAdapter().adapter_id].numerical_binding_hash == (
        third_binding.numerical_binding_hash
    )
    assert third_binding.implementation_content_hash != control_binding.implementation_content_hash
    # No binding folds the environment (LAWS.md ID6): each estimate records its own.
    assert third_binding.numerical_environment_hash is None
    assert installed[COVARIANCE_ADAPTER_ID].numerical_binding_hash == (
        control_binding.numerical_binding_hash
    )
    # The difference between the methods is code, and their content hashes say so.
    challenger_binding = FastSlowCovarianceAdapter().describe_numerical_binding()
    assert challenger_binding.numerical_environment_hash is None
    assert control_binding.numerical_environment_hash is None
    assert (
        challenger_binding.implementation_content_hash
        != control_binding.implementation_content_hash
    )
    assert SINGLE_THREAD_NUMERICAL_CAPABILITY in challenger_binding.required_runtime_capabilities
