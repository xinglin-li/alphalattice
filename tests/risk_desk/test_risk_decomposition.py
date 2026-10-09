"""The public Risk decomposition: one surface, two consumers, one identity each.

Two claims carry this slice. The reconstruction check proves the policy's
per-name scale and the report's factor structure are readings of the same
object; and a report-only change to that factor structure must rotate the
attribution identity while leaving the allocation identity exactly where it was.
The second is what lets a Risk report be revised without invalidating a sealed
holdings ledger, so it is asserted on hashes rather than on values.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from alphalattice.foundation.feature_engine.panels.closure_contracts import (
    SectorRevisionEntry,
    SectorRevisionMap,
)
from alphalattice.investment.risk_research.surfaces.decomposition import (
    INSTALLED_RISK_DECOMPOSITION_RECIPE,
    RISK_IMPLEMENTATION_SOURCE_BLOB,
    RISK_IMPLEMENTATION_SOURCE_COMMIT,
    FactorIdiosyncraticRiskSurface,
    RiskAllocationProjection,
    RiskAttributionProjection,
    RiskDecompositionError,
    RiskDecompositionRecipe,
)
from alphalattice.investment.risk_research.surfaces.producer import (
    RiskDecompositionInputs,
    RiskSurfaceProducer,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

NAMES = 40
FACTORS = 5
SESSION = date(2024, 1, 3)


def _blocks(seed: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    exposures = np.zeros((NAMES, FACTORS), dtype=np.float64)
    exposures[np.arange(NAMES), rng.integers(0, FACTORS, NAMES)] = 1.0
    root = rng.normal(size=(FACTORS, FACTORS))
    factor_covariance = root @ root.T * 0.0004
    idiosyncratic = rng.uniform(1e-4, 9e-4, NAMES)
    volatility = rng.uniform(0.01, 0.04, NAMES)
    return exposures, factor_covariance, idiosyncratic, volatility


def _surface(
    *,
    exposures: np.ndarray | None = None,
    factor_covariance: np.ndarray | None = None,
    idiosyncratic: np.ndarray | None = None,
    volatility: np.ndarray | None = None,
) -> FactorIdiosyncraticRiskSurface:
    base_b, base_f, base_e, base_v = _blocks()
    return FactorIdiosyncraticRiskSurface.create(
        recipe=INSTALLED_RISK_DECOMPOSITION_RECIPE,
        formation_session=SESSION,
        ordered_listing_ids=tuple(f"L{index:03d}" for index in range(NAMES)),
        ordered_factor_ids=tuple(f"S{index}" for index in range(FACTORS)),
        exposures=base_b if exposures is None else exposures,
        factor_covariance=base_f if factor_covariance is None else factor_covariance,
        idiosyncratic_variance=base_e if idiosyncratic is None else idiosyncratic,
        conditional_volatility=base_v if volatility is None else volatility,
        producer_identity="TEST_PRODUCER",
    )


def _book() -> np.ndarray:
    weights = np.zeros(NAMES, dtype=np.float64)
    weights[:10] = 0.1
    return weights


def test_the_public_diagonal_is_exactly_the_conditional_volatility() -> None:
    """The public diagonal is exactly the conditional volatility."""

    surface = _surface()
    _, _, _, volatility = _blocks()
    assert np.sqrt(surface.per_name_variance()) == pytest.approx(volatility)


def test_a_rescaling_inconsistent_with_its_own_factor_block_is_refused() -> None:
    """The check must be able to fail, before any weight is decided."""

    surface = _surface()
    tampered = replace(surface, rescaling=surface.rescaling * 1.05)
    with pytest.raises(RiskDecompositionError, match="decomposition_reconstruction_failed"):
        tampered.verify_reconstruction()


def test_a_report_only_factor_change_leaves_the_allocation_identity_alone() -> None:
    """A report only factor change leaves the allocation identity alone."""

    _, factor_covariance, idiosyncratic, _ = _blocks()
    base = _surface()
    revised_factors = _surface(factor_covariance=factor_covariance * 1.7)
    revised_specific = _surface(idiosyncratic=idiosyncratic * 2.0)
    family = (base, revised_factors, revised_specific)

    assert len({item.surface_hash for item in family}) == 3
    assert len({RiskAttributionProjection.of(item).projection_hash for item in family}) == 3
    assert len({RiskAllocationProjection.of(item).projection_hash for item in family}) == 1


def test_a_changed_volatility_lane_does_rotate_the_allocation_identity() -> None:
    """The other half of the same claim: what the policy consumes must invalidate it."""

    _, _, _, volatility = _blocks()
    base = RiskAllocationProjection.of(_surface())
    moved = RiskAllocationProjection.of(_surface(volatility=volatility * 1.01))
    assert base.projection_hash != moved.projection_hash


def test_an_allocation_projection_with_values_outside_its_identity_is_refused() -> None:
    """A stored hash is evidence only after the consumed bytes are re-read."""

    projection = RiskAllocationProjection.of(_surface())
    tampered = replace(
        projection,
        per_name_volatility=np.asarray(projection.per_name_volatility) * 1.01,
    )
    with pytest.raises(RiskDecompositionError, match="allocation_projection_content_mismatch"):
        tampered.verify_content()


def test_the_allocation_projection_carries_scale_and_nothing_else() -> None:
    """A projection carrying the factor block would re-create the coupling."""

    projection = RiskAllocationProjection.of(_surface())
    fields = set(projection.__slots__)
    assert "exposures" not in fields
    assert "factor_covariance" not in fields
    assert "idiosyncratic_variance" not in fields
    assert projection.diagonal_covariance() == pytest.approx(
        np.square(projection.per_name_volatility)
    )


def test_the_book_variance_matches_the_dense_form_it_never_builds() -> None:
    """The book variance matches the dense form it never builds."""

    surface = _surface()
    scaled = surface.scaled_exposures
    dense = scaled @ surface.factor_covariance @ scaled.T + np.diag(
        np.square(surface.rescaling) * surface.idiosyncratic_variance
    )
    weights = _book()
    assert surface.book_variance(weights) == pytest.approx(float(weights @ dense @ weights))
    assert surface.per_name_variance() == pytest.approx(np.diag(dense))
    shares = surface.variance_shares(weights)
    assert shares.sum() == pytest.approx(1.0)
    assert shares == pytest.approx(weights * (dense @ weights) / surface.book_variance(weights))


def test_the_variance_split_accounts_for_the_whole_book_variance() -> None:
    surface = _surface()
    weights = _book()
    systematic, specific = surface.variance_split(weights)
    assert systematic >= 0.0 and specific >= 0.0
    assert systematic + specific == pytest.approx(surface.book_variance(weights))


def test_industry_exposure_is_the_books_weight_on_each_admitted_factor() -> None:
    surface = _surface()
    weights = _book()
    exposure = surface.industry_exposure(weights)
    assert exposure.shape == (FACTORS,)
    # The exposures are industry dummies, so a long-only book's exposures sum to
    # its invested weight.
    assert float(exposure.sum()) == pytest.approx(float(weights.sum()))


def test_an_asymmetric_factor_covariance_is_refused() -> None:
    _, factor_covariance, _, _ = _blocks()
    broken = factor_covariance.copy()
    broken[0, 1] += 0.001
    with pytest.raises(RiskDecompositionError, match="factor_covariance_asymmetric"):
        _surface(factor_covariance=broken)


def test_a_negative_idiosyncratic_variance_is_refused() -> None:
    _, _, idiosyncratic, _ = _blocks()
    broken = idiosyncratic.copy()
    broken[3] = -1e-5
    with pytest.raises(RiskDecompositionError):
        _surface(idiosyncratic=broken)


def test_a_degenerate_base_variance_is_refused_rather_than_divided_by() -> None:
    """No exposure and no specific risk leaves nothing to rescale onto."""

    with pytest.raises(RiskDecompositionError, match="base_variance_degenerate"):
        _surface(
            exposures=np.zeros((NAMES, FACTORS)),
            idiosyncratic=np.zeros(NAMES),
        )


def test_weights_off_the_listing_axis_are_refused() -> None:
    surface = _surface()
    with pytest.raises(RiskDecompositionError, match="weight_axis_invalid"):
        surface.book_variance(np.zeros(NAMES + 1))


def test_the_recipe_states_its_semantics_and_is_tamper_evident() -> None:
    recipe = INSTALLED_RISK_DECOMPOSITION_RECIPE
    assert recipe.conditional_volatility_decay == 0.94
    assert recipe.conditional_volatility_initialization_sessions == 63
    assert recipe.factor_fit_sessions == 504
    assert recipe.return_unit == "ONE_SESSION_OPEN_TO_OPEN_LOG_RETURN"
    assert recipe.dense_materialization == "NOT_REQUIRED"
    # Industry membership is Foundation authority that Risk reads, not a Sector
    # Forecast capability this recipe implies.
    assert recipe.classification_authority == "FOUNDATION_SECTOR_REVISION_MAP"

    payload = recipe.model_dump(mode="json")
    payload["factor_fit_sessions"] = 252
    with pytest.raises(Exception, match="recipe_identity_invalid"):
        RiskDecompositionRecipe.model_validate(payload)


def test_the_surface_records_the_producer_it_could_not_have_estimated() -> None:
    """A risk surface carries its actual producer's identity while its readback owner performs no
    estimation."""

    assert _surface().producer_identity == "TEST_PRODUCER"


def test_two_surfaces_from_the_same_inputs_share_one_identity() -> None:
    assert _surface().surface_hash == _surface().surface_hash


def test_an_indefinite_factor_covariance_is_refused() -> None:
    """An indefinite factor covariance is refused."""

    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert float(np.linalg.eigvalsh(indefinite).min()) < 0.0
    with pytest.raises(RiskDecompositionError, match="not_positive_semidefinite"):
        FactorIdiosyncraticRiskSurface.create(
            recipe=INSTALLED_RISK_DECOMPOSITION_RECIPE,
            formation_session=SESSION,
            ordered_listing_ids=("A", "B"),
            ordered_factor_ids=("F1", "F2"),
            exposures=np.eye(2),
            factor_covariance=indefinite,
            idiosyncratic_variance=np.array([0.5, 0.5]),
            conditional_volatility=np.array([0.2, 0.2]),
            producer_identity="PROBE",
        )


def test_a_negative_book_variance_is_refused_rather_than_floored() -> None:
    """A negative book variance is refused rather than floored."""

    surface = _surface()
    broken = replace(surface, idiosyncratic_variance=np.full(NAMES, -1e6))
    with pytest.raises(RiskDecompositionError, match="book_variance_negative"):
        broken.book_variance(_book())


def test_the_surface_owns_its_bytes_and_freezes_them() -> None:
    """The surface owns its bytes and freezes them."""

    exposures, factor_covariance, idiosyncratic, volatility = _blocks()
    mutable = exposures.copy()
    surface = FactorIdiosyncraticRiskSurface.create(
        recipe=INSTALLED_RISK_DECOMPOSITION_RECIPE,
        formation_session=SESSION,
        ordered_listing_ids=tuple(f"L{index:03d}" for index in range(NAMES)),
        ordered_factor_ids=tuple(f"S{index}" for index in range(FACTORS)),
        exposures=mutable,
        factor_covariance=factor_covariance,
        idiosyncratic_variance=idiosyncratic,
        conditional_volatility=volatility,
        producer_identity="TEST_PRODUCER",
    )
    before = surface.exposures.copy()

    mutable[0, 0] = 99.0
    assert surface.exposures == pytest.approx(before)
    assert not surface.exposures.flags.writeable
    with pytest.raises(ValueError):
        surface.exposures[0, 0] = 42.0
    surface.verify_content()


def test_content_verification_catches_a_surface_its_hash_does_not_describe() -> None:
    surface = _surface()
    surface.verify_content()
    forged = replace(surface, surface_hash="0" * 64)
    with pytest.raises(RiskDecompositionError, match="content_hash_mismatch"):
        forged.verify_content()


def test_the_frozen_arrays_cannot_be_thawed_by_their_holder() -> None:
    """The frozen arrays cannot be thawed by their holder."""

    surface = _surface()
    for values in (
        surface.exposures,
        surface.factor_covariance,
        surface.idiosyncratic_variance,
        surface.conditional_volatility,
        surface.rescaling,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)


def test_a_replaced_rescaling_is_caught_by_readback() -> None:
    """A replaced rescaling is caught by readback."""

    surface = _surface()
    forged = replace(surface, rescaling=np.full(NAMES, 9.0))
    with pytest.raises(RiskDecompositionError, match="content_hash_mismatch"):
        forged.verify_content()


def test_readback_rebuilds_the_derived_rescaling_rather_than_trusting_it() -> None:
    """A hash proves the bytes; it does not prove they are mutually consistent."""

    surface = _surface()
    scaled = replace(
        surface,
        rescaling=surface.rescaling * 1.5,
        surface_hash=surface.surface_hash,
    )
    with pytest.raises(RiskDecompositionError):
        scaled.verify_content()
    surface.verify_content()


def test_the_freeze_cannot_be_undone_through_the_base_array() -> None:
    """The freeze cannot be undone through the base array."""

    surface = _surface()
    values = surface.exposures
    with pytest.raises(ValueError, match="WRITEABLE"):
        values.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        values.base.setflags(write=True)

    terminal = values.base
    while hasattr(terminal, "base") and terminal.base is not None:
        terminal = terminal.base
    assert isinstance(terminal, bytes)
    assert not hasattr(terminal, "setflags")


def test_every_surface_block_is_frozen_the_same_way() -> None:
    surface = _surface()
    for values in (
        surface.exposures,
        surface.factor_covariance,
        surface.idiosyncratic_variance,
        surface.conditional_volatility,
        surface.rescaling,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)
        with pytest.raises(ValueError):
            values.base.setflags(write=True)


def _sector_map(listings: tuple[str, ...]) -> SectorRevisionMap:
    entries = tuple(
        SectorRevisionEntry(
            listing_id=listing_id,
            provider="TEST",
            provider_symbol=listing_id,
            sector_name=f"Sector {index % 2}",
            sector_key=f"sector-{index % 2}",
            payload_hash=f"{index + 1:064x}",
            evidence_hash=f"{index + 101:064x}",
        )
        for index, listing_id in enumerate(listings)
    )
    identity = {
        "kind": "SectorRevisionMap",
        "manifest_revision": "1" * 64,
        "sector_revision": "2" * 64,
        "entries": [entry.model_dump(mode="json") for entry in entries],
    }
    return SectorRevisionMap(**identity, map_hash=canonical_hash(identity))


def _producer_inputs(*, omit_last_classification: bool = False) -> RiskDecompositionInputs:
    listings = tuple(f"P{index:02d}" for index in range(6))
    sessions = tuple(date.fromordinal(730_000 + index) for index in range(567))
    rng = np.random.default_rng(8721)
    returns = rng.normal(0.0, 0.012, (567, len(listings))).astype(np.float64)
    returns[71, 2] = np.nan
    returns[430, 5] = np.nan
    classification_axis = listings[:-1] if omit_last_classification else listings
    return RiskDecompositionInputs(
        formation_session=date.fromordinal(730_568),
        history_sessions=sessions,
        ordered_listing_ids=listings,
        open_to_open_log_returns=returns,
        classification=_sector_map(classification_axis),
        return_surface_hash="3" * 64,
    )


def test_the_promoted_risk_producer_matches_an_independent_reference() -> None:
    inputs = _producer_inputs()
    produced = RiskSurfaceProducer().produce(inputs)
    recipe = INSTALLED_RISK_DECOMPOSITION_RECIPE
    returns = inputs.open_to_open_log_returns

    state = np.nanvar(returns[:63], axis=0, ddof=1)
    historical_sigma = np.full(returns.shape, np.nan, dtype=np.float64)
    for row in range(63, len(returns)):
        historical_sigma[row] = np.sqrt(np.clip(state, 1e-16, None))
        finite = np.isfinite(returns[row])
        squared = np.square(np.nan_to_num(returns[row], nan=0.0))
        state = np.where(finite, 0.94 * state + 0.06 * squared, state)

    standardized = np.nan_to_num(
        returns[-504:] / historical_sigma[-504:],
        nan=0.0,
    )
    standardized = np.clip(standardized, -10.0, 10.0)
    standardized -= standardized.mean(axis=0)
    exposures = np.zeros((6, 2), dtype=np.float64)
    exposures[np.arange(6), np.arange(6) % 2] = 1.0
    factor_returns = standardized @ exposures / exposures.sum(axis=0)
    residuals = standardized - factor_returns @ exposures.T
    expected_f = np.cov(factor_returns, rowvar=False, ddof=1)
    expected_e = np.square(residuals).sum(axis=0) / (504 - 2)

    assert produced.surface.factor_covariance == pytest.approx(expected_f)
    assert produced.surface.idiosyncratic_variance == pytest.approx(expected_e)
    assert produced.surface.conditional_volatility == pytest.approx(
        np.sqrt(np.clip(state, recipe.variance_floor, None))
    )
    assert produced.surface.per_name_variance() == pytest.approx(
        np.square(produced.allocation.per_name_volatility)
    )
    assert produced.attribution.surface_hash == produced.surface.surface_hash
    produced.surface.verify_content()


def test_the_risk_producer_refuses_an_incomplete_foundation_classification() -> None:
    with pytest.raises(RiskDecompositionError, match="classification_incomplete"):
        RiskSurfaceProducer().produce(_producer_inputs(omit_last_classification=True))


def test_the_installed_recipe_binds_the_promoted_implementation_bytes() -> None:
    recipe = INSTALLED_RISK_DECOMPOSITION_RECIPE
    assert recipe.implementation_source_commit == RISK_IMPLEMENTATION_SOURCE_COMMIT
    assert recipe.implementation_source_blob == RISK_IMPLEMENTATION_SOURCE_BLOB
