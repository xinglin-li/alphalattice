from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)
from alphalattice.investment.risk_research.contracts import (
    CausalRiskReturnSurface,
    RiskHistoricalDiagnostics,
    RiskUniverseEpoch,
    default_covariance_recipe,
    seal_contract,
)
from alphalattice.investment.risk_research.estimators.covariance import (
    RISK_NUMERICAL_THREAD_LIMIT,
    RiskNumericalError,
    estimate_dynamic_covariance,
    matrix_content_hash,
    numerical_environment,
    numerical_environment_hash,
    risk_numerical_thread_policy,
)
from alphalattice.investment.risk_research.surfaces.artifacts import (
    HistoricalCovarianceReader,
    RiskArtifactError,
    RiskArtifactStore,
)
from alphalattice.investment.risk_research.surfaces.returns import (
    CausalRiskReturnReader,
    RiskReturnArtifactStore,
    RiskReturnSurfaceError,
    derive_risk_return_rows,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

PLAYPEN_ROOT = Path(__file__).resolve().parents[2]


def _manifest(*, daily_price_basis: str = "split_adjusted") -> UniverseManifest:
    profile = MarketProfile(
        market_profile_id="test-us",
        display_name="Test",
        market="US",
        currency="USD",
        calendar_id="XNYS",
        provider="test",
        daily_price_basis=daily_price_basis,
        manifest_as_of=date(2026, 1, 3),
        data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
    )
    listing = ManifestListing(
        listing_id="listing-a", symbol="AAA", mic="XNYS", provider_symbol="AAA"
    )
    return UniverseManifest(
        manifest_id="test",
        profile=profile,
        listings=(listing,),
        revision_sha256="a" * 64,
    )


def _bar(session: date, open_price: float) -> RawDailyBar:
    return RawDailyBar(
        listing_id="listing-a",
        provider="test",
        session_date=session,
        open=open_price,
        high=open_price,
        low=open_price,
        close=open_price,
        volume=100,
    )


def test_split_adjusted_reverse_split_does_not_create_a_false_return() -> None:
    first = date(2026, 1, 2)
    second = date(2026, 1, 3)
    rows = derive_risk_return_rows(
        manifest=_manifest(),
        listing_id="listing-a",
        symbol="AAA",
        bars=(_bar(first, 100.0), _bar(second, 100.0)),
        actions=(
            CorporateActionEvent(
                listing_id="listing-a",
                provider="test",
                effective_date=second,
                action_kind="SPLIT",
                new_shares_per_old_share=0.1,
            ),
        ),
        required_sessions=(first, second),
    )
    assert len(rows) == 1
    assert rows[0]["open_total_return_log"] == 0.0


def test_dividend_aware_log_return_and_missing_session_fail_closed() -> None:
    first = date(2026, 1, 2)
    second = date(2026, 1, 3)
    rows = derive_risk_return_rows(
        manifest=_manifest(),
        listing_id="listing-a",
        symbol="AAA",
        bars=(_bar(first, 100.0), _bar(second, 101.0)),
        actions=(
            CorporateActionEvent(
                listing_id="listing-a",
                provider="test",
                effective_date=second,
                action_kind="CASH_DIVIDEND",
                cash_amount=1.0,
            ),
        ),
        required_sessions=(first, second),
    )
    assert float(rows[0]["open_total_return_log"]) == pytest.approx(np.log(1.02))
    with pytest.raises(
        RiskReturnSurfaceError, match=r"risk_research\.return_source_session_missing"
    ):
        derive_risk_return_rows(
            manifest=_manifest(),
            listing_id="listing-a",
            symbol="AAA",
            bars=(_bar(first, 100.0),),
            actions=(),
            required_sessions=(first, second),
        )


def _returns(seed: int = 7, assets: int = 8) -> np.ndarray:
    generator = np.random.default_rng(seed)
    market = generator.normal(0.0, 0.009, size=(315, 1))
    idiosyncratic = generator.normal(0.0, 0.014, size=(315, assets))
    return market + idiosyncratic


def test_ewma_standardized_ledoit_wolf_covariance_golden() -> None:
    returns = _returns()
    listing_ids = tuple(f"listing-{index:02d}" for index in range(returns.shape[1]))
    result = estimate_dynamic_covariance(
        returns=returns,
        ordered_listing_ids=listing_ids,
        formation_session=date(2026, 1, 3),
    )
    expected_variance = np.var(returns[:63], axis=0, ddof=1)
    for row in returns[63:]:
        expected_variance = 0.94 * expected_variance + 0.06 * np.square(row)
    np.testing.assert_allclose(np.diag(result.matrix), expected_variance, rtol=1e-12)
    np.testing.assert_allclose(result.matrix, result.matrix.T, rtol=0.0, atol=0.0)
    scale = np.sqrt(np.diag(result.matrix))
    correlation = result.matrix / np.outer(scale, scale)
    np.testing.assert_allclose(np.diag(correlation), 1.0, rtol=0.0, atol=1e-12)
    assert result.diagnostics.minimum_eigenvalue > 0.0
    assert result.diagnostics.condition_number <= 1e12
    assert not result.matrix.flags.writeable
    assert result.diagnostics.matrix_hash == matrix_content_hash(result.matrix)


def test_covariance_axis_permutation_and_invalid_inputs() -> None:
    returns = _returns()
    listing_ids = tuple(f"listing-{index:02d}" for index in range(returns.shape[1]))
    baseline = estimate_dynamic_covariance(
        returns=returns,
        ordered_listing_ids=listing_ids,
        formation_session=date(2026, 1, 3),
    )
    permutation = np.asarray([3, 0, 7, 2, 1, 6, 5, 4])
    permuted = estimate_dynamic_covariance(
        returns=returns[:, permutation],
        ordered_listing_ids=tuple(listing_ids[index] for index in permutation),
        formation_session=date(2026, 1, 3),
    )
    np.testing.assert_allclose(
        permuted.matrix, baseline.matrix[np.ix_(permutation, permutation)], rtol=1e-12
    )
    with pytest.raises(RiskNumericalError, match="input_shape_invalid"):
        estimate_dynamic_covariance(
            returns=returns[:-1],
            ordered_listing_ids=listing_ids,
            formation_session=date(2026, 1, 3),
        )
    contaminated = returns.copy()
    contaminated[0, 0] = np.nan
    with pytest.raises(RiskNumericalError, match="input_non_finite"):
        estimate_dynamic_covariance(
            returns=contaminated,
            ordered_listing_ids=listing_ids,
            formation_session=date(2026, 1, 3),
        )


def test_risk_numerical_thread_policy_is_fixed_and_path_free() -> None:
    returns = _returns()
    listing_ids = tuple(f"listing-{index:02d}" for index in range(returns.shape[1]))
    outputs = []
    for ambient_limit in (1, 4):
        with threadpool_limits(limits=ambient_limit), risk_numerical_thread_policy():
            assert all(
                int(pool["num_threads"]) <= RISK_NUMERICAL_THREAD_LIMIT
                for pool in threadpool_info()
                if pool.get("num_threads") is not None
            )
            outputs.append(
                estimate_dynamic_covariance(
                    returns=returns,
                    ordered_listing_ids=listing_ids,
                    formation_session=date(2026, 1, 3),
                )
            )
    np.testing.assert_array_equal(outputs[0].matrix, outputs[1].matrix)
    assert outputs[0].diagnostics == outputs[1].diagnostics
    assert outputs[0].diagnostics.numerical_environment_hash == numerical_environment_hash()
    environment = numerical_environment()
    assert environment["thread_limit"] == 1
    assert "filepath" not in str(environment)
    assert "site-packages" not in str(environment)


@pytest.mark.parametrize(
    ("module_name", "function_name", "parameters"),
    (
        ("covariance", "estimate_dynamic_covariance", None),
        ("fast_slow", "estimate_fast_slow_covariance", {"blend_weight": 0.25}),
        ("diagonal", "estimate_diagonal_shrunk_covariance", {"correlation_shrinkage": 0.0}),
    ),
)
def test_the_live_estimate_runs_single_threaded_whatever_the_ambient_threads(
    monkeypatch: pytest.MonkeyPatch, module_name, function_name, parameters
) -> None:
    """regression (INV FINDING 2026-10-08): each installed adapter's estimate enters the policy.

    The policy existed and was tested, but no production path entered it: the live estimates
    ran at the machine's default threads, and their eigen diagnostics' last bits moved with them.
    """
    from importlib import import_module

    from alphalattice.investment.risk_research.contracts import default_covariance_recipe
    from alphalattice.investment.risk_research.estimators.catalog import (
        build_installed_risk_estimator_catalog,
    )
    from alphalattice.investment.risk_research.estimators.contracts import BoundRiskReturnInput

    owner = import_module(f"alphalattice.investment.risk_research.estimators.{module_name}")
    catalog = build_installed_risk_estimator_catalog()
    if parameters is None:
        envelope = catalog.seal_recipe(
            recipe_schema_id=owner.COVARIANCE_RECIPE_SCHEMA_ID,
            parameters=default_covariance_recipe().model_dump(mode="json"),
        )
    else:
        schema = next(
            value
            for name, value in vars(owner).items()
            if name.endswith("_RECIPE_SCHEMA_ID") and not name.startswith("COVARIANCE")
        )
        envelope = catalog.admit_recipe(
            capability_handle=schema, parameters=parameters, seed=0
        ).recipe
    adapter = catalog.resolve(envelope)
    returns = _returns(assets=120)
    listing_ids = tuple(f"listing-{index:03d}" for index in range(returns.shape[1]))
    observed: list[int] = []
    inner = getattr(owner, function_name)

    def watched(**kwargs):
        observed.append(
            max(
                int(pool["num_threads"])
                for pool in threadpool_info()
                if pool.get("num_threads") is not None
            )
        )
        return inner(**kwargs)

    monkeypatch.setattr(owner, function_name, watched)
    outputs = []
    for ambient_limit in (4, 2):
        with threadpool_limits(limits=ambient_limit):
            outputs.append(
                adapter.estimate(
                    recipe=envelope,
                    inputs=BoundRiskReturnInput.create(
                        return_surface_hash="a" * 64,
                        ordered_listing_ids=listing_ids,
                        formation_session=date(2026, 1, 3),
                        returns=returns,
                    ),
                )
            )
    assert observed == [RISK_NUMERICAL_THREAD_LIMIT, RISK_NUMERICAL_THREAD_LIMIT]
    assert outputs[0].eigenvalues.tobytes() == outputs[1].eigenvalues.tobytes()
    assert outputs[0].diagnostics == outputs[1].diagnostics


def test_packed_covariance_chunk_round_trip_and_single_lease(tmp_path: Path) -> None:
    returns = _returns(assets=5)
    listing_ids = tuple(f"listing-{index:02d}" for index in range(returns.shape[1]))
    first = estimate_dynamic_covariance(
        returns=returns,
        ordered_listing_ids=listing_ids,
        formation_session=date(2026, 1, 2),
        recipe=default_covariance_recipe(),
    )
    second = estimate_dynamic_covariance(
        returns=np.roll(returns, -1, axis=0),
        ordered_listing_ids=listing_ids,
        formation_session=date(2026, 1, 3),
    )
    store = RiskArtifactStore(tmp_path)
    chunk = store.publish_covariance_chunk(
        formation_sessions=(date(2026, 1, 2), date(2026, 1, 3)),
        matrices=(first.matrix, second.matrix),
        matrix_hashes=(first.diagnostics.matrix_hash, second.diagnostics.matrix_hash),
    )
    reader = HistoricalCovarianceReader(tmp_path)
    with reader.lease(chunk) as lease:
        first_read = lease.matrix(date(2026, 1, 2))
        second_read = lease.matrix(date(2026, 1, 3))
        np.testing.assert_array_equal(first_read, first.matrix)
        np.testing.assert_array_equal(second_read, second.matrix)
        assert first_read is not second_read
        assert not first_read.flags.writeable
        assert not second_read.flags.writeable
        with pytest.raises(RiskArtifactError, match="already_active"), reader.lease(chunk):
            pass
    with pytest.raises(RiskArtifactError, match="lease_closed"):
        lease.matrix(date(2026, 1, 2))

    path = store.chunk_path(chunk)
    damaged = bytearray(path.read_bytes())
    damaged[0] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(RiskArtifactError, match="chunk_tampered"), reader.lease(chunk):
        pass


class _FixtureReturnReader:
    def __init__(self, sessions: tuple[date, ...], values: np.ndarray) -> None:
        self.sessions = sessions
        self.values = values

    def available_sessions(self, _surface: CausalRiskReturnSurface) -> tuple[date, ...]:
        return self.sessions

    def read_sessions(
        self,
        _surface: CausalRiskReturnSurface,
        sessions: tuple[date, ...],
    ) -> np.ndarray:
        positions = {session: index for index, session in enumerate(self.sessions)}
        output = self.values[[positions[session] for session in sessions]]
        output.setflags(write=False)
        return output


def test_unadjusted_risk_return_basis_fails_closed() -> None:
    first = date(2026, 1, 2)
    second = first + timedelta(days=1)
    with pytest.raises(RiskReturnSurfaceError, match="return_price_basis_unsupported"):
        derive_risk_return_rows(
            manifest=_manifest(daily_price_basis="unadjusted"),
            listing_id="listing-a",
            symbol="AAA",
            bars=(_bar(first, 10.0), _bar(second, 100.0)),
            actions=(
                CorporateActionEvent(
                    listing_id="listing-a",
                    provider="test",
                    effective_date=second,
                    action_kind="SPLIT",
                    new_shares_per_old_share=0.1,
                ),
            ),
            required_sessions=(first, second),
        )


def test_return_chunk_rejects_value_tamper_with_preserved_footer(
    tmp_path: Path,
) -> None:
    first = date(2026, 1, 2)
    second = first + timedelta(days=1)
    rows = derive_risk_return_rows(
        manifest=_manifest(),
        listing_id="listing-a",
        symbol="AAA",
        bars=(_bar(first, 100.0), _bar(second, 101.0)),
        actions=(),
        required_sessions=(first, second),
    )
    store = RiskReturnArtifactStore(tmp_path)
    chunk = store.publish_chunk(pa.Table.from_pylist(rows))
    target = store.resolve_chunk(chunk)
    table = pq.read_table(target).combine_chunks()
    column = table.schema.get_field_index("open_total_return_log")
    tampered = table.set_column(
        column,
        "open_total_return_log",
        pa.array([float(table.column(column)[0].as_py()) + 0.1], type=pa.float64()),
    )
    pq.write_table(tampered, target, compression="zstd")
    with pytest.raises(RiskReturnSurfaceError, match="return_chunk_tampered"):
        store.resolve_chunk(chunk)

    epoch = seal_contract(
        RiskUniverseEpoch,
        "epoch_hash",
        market_profile_id="test-us",
        universe_manifest_revision="a" * 64,
        panel_snapshot_hash="b" * 64,
        ordered_listing_ids=("listing-a",),
    )
    surface = CausalRiskReturnSurface.model_construct(
        kind="CausalRiskReturnSurface",
        epoch=epoch,
        first_formation_session=second,
        last_formation_session=second,
        formation_count=1,
        source_watermark_hash="c" * 64,
        chunks=(chunk,),
        limitations=("fixture",),
        surface_hash="d" * 64,
    )
    with pytest.raises(RiskReturnSurfaceError, match="return_chunk_tampered"):
        CausalRiskReturnReader(tmp_path).read_sessions(surface, (second,))


def test_historical_diagnostics_accepts_only_complete_or_legacy_metric_group() -> None:
    evaluation = {
        "formation_session": "2026-01-02",
        "next_session": "2026-01-03",
        "matrix_hash": "a" * 64,
        "shrinkage": 0.2,
        "minimum_eigenvalue": 0.001,
        "maximum_eigenvalue": 0.02,
        "condition_number": 20.0,
        "trace": 0.1,
        "average_correlation": 0.2,
        "top_one_eigenvalue_share": 0.3,
        "top_five_eigenvalue_share": 0.6,
        "gaussian_log_score_per_asset": 1.0,
        "equal_weight_predicted_variance": 0.001,
        "equal_weight_realized_squared_return": 0.002,
        "sector_balanced_predicted_variance": 0.001,
        "sector_balanced_realized_squared_return": 0.002,
        "trace_ratio": None,
        "frobenius_delta_ratio": None,
        "leading_eigenvalue_delta_ratio": None,
    }
    payload = {
        "kind": "RiskHistoricalDiagnostics",
        "epoch_hash": "b" * 64,
        "recipe_hash": "c" * 64,
        "evaluations": [evaluation],
    }
    payload["diagnostics_hash"] = canonical_hash(payload)
    historical = RiskHistoricalDiagnostics.model_validate(payload)
    assert historical.evaluations[0].annualized_volatility_median is None

    partial = dict(evaluation, annualized_volatility_minimum=0.1)
    with pytest.raises(ValueError, match="formation_diagnostics_incomplete"):
        RiskHistoricalDiagnostics.model_validate(
            {
                **payload,
                "evaluations": [partial],
                "diagnostics_hash": "d" * 64,
            }
        )
