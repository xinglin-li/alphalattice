"""Focused contracts for the program-scoped Alpha array surface."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from alphalattice.capabilities.alpha_modeling.adapters import (
    regularized_linear as regularized_linear_module,
)
from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear import (
    RegularizedLinearParameters,
    fit_regularized_linear,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.investment.alpha_research.inputs import surfaces as surface_module
from alphalattice.investment.alpha_research.inputs.folds import (
    AlphaArrayBoundaryError,
    AlphaCurrentRefitArrays,
)
from alphalattice.investment.alpha_research.inputs.observations import (
    AlphaArrayReadLedger,
    AlphaArrayReadObservation,
)
from alphalattice.investment.alpha_research.inputs.surfaces import AlphaArraySurfaceBudget
from alphalattice.investment.alpha_research.scores import refit as current_refit_module

GIB = 1024**3
MIB = 1024**2


@pytest.mark.parametrize(
    ("parameters", "estimator_name"),
    (
        (
            RegularizedLinearParameters(family="lasso", alpha_max_multiplier=0.2),
            "Lasso",
        ),
        (
            RegularizedLinearParameters(
                family="elastic_net",
                alpha_max_multiplier=0.05,
                l1_ratio=0.35,
            ),
            "ElasticNet",
        ),
    ),
)
def test_sparse_regularized_linear_uses_precomputed_gram(
    monkeypatch: pytest.MonkeyPatch,
    parameters: RegularizedLinearParameters,
    estimator_name: str,
) -> None:
    observed: dict[str, object] = {}
    estimator = getattr(regularized_linear_module, estimator_name)

    def factory(**kwargs):
        observed.update(kwargs)
        return estimator(**kwargs)

    monkeypatch.setattr(regularized_linear_module, estimator_name, factory)
    features = np.arange(120, dtype=np.float64).reshape(40, 3) / 100.0
    targets = np.sin(np.arange(40, dtype=np.float64))
    fit = fit_regularized_linear(
        parameters=parameters,
        training_features=features,
        training_targets=targets,
        prediction_features=features[:2],
    )

    assert observed["precompute"] is True
    assert fit.predictions.shape == (2,)


@pytest.mark.parametrize(
    ("physical_memory", "cache_cap", "peak_limit", "minimum", "preferred"),
    (
        (8 * GIB, int(8 * GIB * 0.08), 4 * GIB, False, False),
        (16 * GIB, int(16 * GIB * 0.08), 4 * GIB, True, False),
        (32 * GIB, 2 * GIB, 6 * GIB, True, True),
        (64 * GIB, 2 * GIB, 6 * GIB, True, True),
    ),
)
def test_budget_balances_desktop_memory_and_speed(
    physical_memory: int,
    cache_cap: int,
    peak_limit: int,
    minimum: bool,
    preferred: bool,
) -> None:
    budget = AlphaArraySurfaceBudget.derive(
        session_count=2_264,
        listing_count=466,
        factor_count=31,
        total_physical_memory_bytes=physical_memory,
    )
    assert budget.cache_cap_bytes == cache_cap
    assert budget.peak_rss_limit_bytes == peak_limit
    assert budget.minimum_recommended_physical_memory_bytes == 16 * GIB
    assert budget.preferred_physical_memory_bytes == 32 * GIB
    assert budget.meets_minimum_recommendation is minimum
    assert budget.meets_preferred_recommendation is preferred


def test_budget_falls_back_when_dense_surface_exceeds_cap() -> None:
    budget = AlphaArraySurfaceBudget.derive(
        session_count=20_000,
        listing_count=3_000,
        factor_count=55,
        total_physical_memory_bytes=16 * GIB,
    )
    assert budget.cache_cap_bytes > 512 * MIB
    assert budget.estimated_surface_bytes > budget.cache_cap_bytes
    assert budget.mode == "BOUNDED_FOLD"
    budget.assert_live_surface_admitted(
        session_count=756,
        listing_count=466,
        factor_count=31,
    )
    with pytest.raises(
        AlphaArrayBoundaryError,
        match="array_surface_resource_limit_exceeded",
    ):
        budget.assert_live_surface_admitted(
            session_count=10_000,
            listing_count=3_000,
            factor_count=55,
        )


def test_budget_accounts_for_retained_and_live_fold_arrays() -> None:
    budget = AlphaArraySurfaceBudget.derive(
        session_count=2_264,
        listing_count=466,
        factor_count=31,
        retained_fold_row_counts=(1_000, 2_000),
        worst_live_fold_row_count=3_000,
        total_physical_memory_bytes=32 * GIB,
    )
    assert budget.estimated_retained_derived_bytes > 0
    assert budget.estimated_worst_live_fold_bytes > 0
    assert budget.estimated_peak_working_set_bytes == (
        budget.estimated_surface_bytes
        + budget.estimated_retained_derived_bytes
        + budget.estimated_worst_live_fold_bytes
    )
    bytes_per_row = budget.estimate_fold_bytes(row_count=1, factor_count=31)
    rows_beyond_peak_limit = budget.peak_rss_limit_bytes // bytes_per_row + 1
    with pytest.raises(
        AlphaArrayBoundaryError,
        match="array_surface_resource_limit_exceeded",
    ):
        budget.assert_bounded_operation_admitted(
            session_count=100,
            listing_count=100,
            factor_count=31,
            live_row_count=rows_beyond_peak_limit,
        )


def test_read_ledger_reports_observed_full_reads_instead_of_a_constant() -> None:
    ledger = AlphaArrayReadLedger()
    ledger.observe(
        AlphaArrayReadObservation(
            operation="read_development",
            row_count=100,
            chunk_count=2,
            byte_count=8_192,
        )
    )
    assert ledger.count("read_development") == 1
    assert ledger.total("row_count") == 100


def test_full_development_reader_is_not_an_active_api() -> None:
    assert not hasattr(CausalExecutionOutcomeDevelopmentReader, "read_development")
    playpen_root = Path(__file__).resolve().parents[2]
    active_sources = tuple((playpen_root / "src").rglob("*.py")) + tuple(
        (playpen_root / "scripts").rglob("*.py")
    )
    call = b".read_development("
    assert not any(call in path.read_bytes() for path in active_sources)


def test_arrow_float_conversion_preserves_signed_zero_and_nan_payload() -> None:
    expected_bits = np.asarray(
        (0x0000000000000000, 0x8000000000000000, 0x7FF8000000000042),
        dtype=np.uint64,
    )
    source = expected_bits.view(np.float64)
    observed = surface_module._float_values(pa.array(source))
    assert np.array_equal(observed.view(np.uint64), expected_bits)
    assert not observed.flags.writeable


def test_fixed_width_hash_encoding_never_silently_truncates() -> None:
    accepted = surface_module._encoded_hashes(pa.array(["a", "b" * 64]))
    assert bytes(accepted[0]).decode("ascii") == "a"
    assert bytes(accepted[1]).decode("ascii") == "b" * 64
    with pytest.raises(AlphaArrayBoundaryError, match="array_surface_axis_mismatch"):
        surface_module._encoded_hashes(pa.array(["c" * 65]))
    with pytest.raises(AlphaArrayBoundaryError, match="array_surface_axis_mismatch"):
        surface_module._encoded_hashes(pa.array(["dé"]))


def test_hash_decoding_can_be_omitted_for_numerical_only_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = date(2026, 1, 2)
    feature = surface_module._FeatureSurface(
        sessions=(session,),
        listings=("A",),
        session_index={session: 0},
        listing_index={"A": 0},
        values=np.asarray(((1.0,),), dtype=np.float64),
        present=np.asarray((True,), dtype=np.bool_),
        row_hashes=np.asarray((b"a",), dtype="S64"),
        read_count=1,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("row hashes must not be decoded")

    monkeypatch.setattr(surface_module, "_decode_hashes", forbidden)
    rows = surface_module._matrix_for_keys(
        ((session, "A"),),
        feature,
        None,
        factor_count=1,
        include_hashes=False,
    )

    assert rows[4] == rows[5] == ()


def test_current_state_computes_score_statistics_once(monkeypatch) -> None:
    arrays = AlphaCurrentRefitArrays(
        ordered_factor_ids=("factor",),
        ordered_listing_ids=("listing",),
        training_sessions=(date(2026, 1, 2),),
        training_cutoff=date(2026, 1, 2),
        formation_session=date(2026, 1, 3),
        training_features=np.asarray(((1.0,),), dtype=np.float64),
        training_targets=np.asarray((0.01,), dtype=np.float64),
        training_mask=np.asarray((True,), dtype=np.bool_),
        current_features=np.asarray(((2.0,),), dtype=np.float64),
        current_feature_complete=np.asarray((True,), dtype=np.bool_),
    )
    calls = 0
    original = current_refit_module._score_statistics

    def observed(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(current_refit_module, "_score_statistics", observed)
    current_refit_module._linear_state(
        request_hash="a" * 64,
        candidate_id="model.ridge.test",
        family_id="ridge",
        arrays=arrays,
        coefficients=np.asarray((0.5,), dtype=np.float64),
        intercept=0.0,
        training_mse=0.01,
        scores=np.asarray((1.0,), dtype=np.float64),
    )
    assert calls == 1
