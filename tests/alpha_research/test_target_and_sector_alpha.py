"""Focused contracts for Alpha target lanes and the causal sector EMA component."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pyarrow as pa
import pytest

from alphalattice.investment.alpha_research.candidates.artifacts import (
    AlphaGoalResearchArtifactStore,
)
from alphalattice.investment.alpha_research.targets.execution_outcome import (
    AlphaTargetBoundaryError,
    AlphaTargetLane,
    build_alpha_target_policy,
    compile_alpha_target_surface,
)
from alphalattice.investment.sector_research.inputs.storage import SectorContextStore
from alphalattice.investment.sector_research.inputs.surface import (
    SectorContextBoundaryError,
    build_sector_context_policy,
    compile_sector_context_surface,
)
from alphalattice.investment.sector_research.models.legacy_excess_ema import (
    SectorEmaAlphaManifest,
    SectorEmaAlphaSurface,
    sector_ema_table_content_hash,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _sessions(count: int) -> tuple[date, ...]:
    result: list[date] = []
    current = date(2025, 1, 2)
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return tuple(result)


def _source(count: int = 45) -> tuple[pa.Table, dict[str, str]]:
    sessions = _sessions(count)
    listings = tuple(f"listing-{index:02d}" for index in range(10))
    sectors = {
        listing: ("sector-a" if index < 5 else "sector-b") for index, listing in enumerate(listings)
    }
    rows = []
    for session_index, session in enumerate(sessions):
        close = datetime.combine(session, datetime.min.time(), tzinfo=UTC) + timedelta(hours=21)
        available = close + timedelta(hours=41)
        for listing_index, listing in enumerate(listings):
            log_return = (
                (0.001 if listing_index < 5 else -0.001)
                + (listing_index - 4.5) * 0.0001
                + session_index * 0.00001
            )
            rows.append(
                {
                    "formation_session": session,
                    "formation_close_at": close,
                    "holding_end_open_at": available,
                    "listing_id": listing,
                    "fit_target": log_return,
                    "simple_economic_return": float(np.expm1(log_return)),
                }
            )
    return pa.Table.from_pylist(rows), sectors


@pytest.mark.parametrize("lane", tuple(AlphaTargetLane))
def test_alpha_target_lanes_share_sector_treatment_and_preserve_economic_return(
    lane: AlphaTargetLane,
) -> None:
    source, sectors = _source()
    policy = build_alpha_target_policy(lane=lane, sector_revision="a" * 64)
    surface = compile_alpha_target_surface(
        source_table=source,
        policy=policy,
        sector_by_listing_id=sectors,
    )
    assert policy.sector_point_in_time_qualified is False
    assert policy.sector_history_treatment == "CURRENT_CLASSIFICATION_BACKFILLED"
    np.testing.assert_array_equal(
        surface["simple_economic_return"].to_numpy(),
        source["simple_economic_return"].to_numpy(),
    )
    values = surface["fit_target"].to_numpy().reshape(45, 10)
    assert np.all(np.isfinite(values))
    if lane is AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS:
        assert float(np.max(np.abs(np.mean(values, axis=1)))) < 0.02
    else:
        np.testing.assert_allclose(np.median(values, axis=1), 0.0, atol=1e-15)


def test_extreme_outlier_is_clipped_in_the_fit_lane_and_never_in_the_raw_lanes() -> None:
    """Extreme outlier is clipped in the fit lane and never in the raw lanes."""

    source, sectors = _source()
    outlier_index = 4  # session 0, listing-04
    fit = source["fit_target"].to_pylist()
    simple = source["simple_economic_return"].to_pylist()
    fit[outlier_index] = 5.0
    simple[outlier_index] = float(np.expm1(5.0))
    outlier_source = source.set_column(
        source.schema.get_field_index("fit_target"),
        "fit_target",
        pa.array(fit, type=pa.float64()),
    ).set_column(
        source.schema.get_field_index("simple_economic_return"),
        "simple_economic_return",
        pa.array(simple, type=pa.float64()),
    )

    policy = build_alpha_target_policy(
        lane=AlphaTargetLane.SECTOR_RESIDUAL_ROBUST_Z, sector_revision="a" * 64
    )
    surface = compile_alpha_target_surface(
        source_table=outlier_source,
        policy=policy,
        sector_by_listing_id=sectors,
    )

    np.testing.assert_array_equal(
        surface["raw_log_execution_return"].to_numpy(),
        outlier_source["fit_target"].to_numpy(),
    )
    np.testing.assert_array_equal(
        surface["simple_economic_return"].to_numpy(),
        outlier_source["simple_economic_return"].to_numpy(),
    )

    session_values = np.asarray(fit[:10], dtype=np.float64)
    session_median = float(np.median(session_values))
    session_mad = float(np.median(np.abs(session_values - session_median)))
    unclipped_scale = (5.0 - session_median) / (1.4826 * session_mad)
    assert unclipped_scale > 1_000.0

    lane_values = surface["fit_target"].to_numpy().reshape(45, 10)
    assert np.all(np.isfinite(lane_values))
    assert abs(float(lane_values[0, outlier_index])) < 50.0


def test_alpha_target_fails_closed_for_incomplete_sector_authority() -> None:
    source, sectors = _source()
    sectors.pop("listing-09")
    with pytest.raises(AlphaTargetBoundaryError, match="sector_map_incomplete"):
        compile_alpha_target_surface(
            source_table=source,
            policy=build_alpha_target_policy(
                lane=AlphaTargetLane.SECTOR_RESIDUAL_ROBUST_Z,
                sector_revision="2" * 64,
            ),
            sector_by_listing_id=sectors,
        )


def _sector_ema_surface() -> SectorEmaAlphaSurface:
    """A sealed surface of the retired builder's shape, for the readback that stays."""

    table = pa.table(
        {
            "formation_session": pa.array([date(2025, 1, 2)] * 2, type=pa.date32()),
            "sector_id": pa.array(["sector-a", "sector-b"]),
            "ema_excess_log_return": pa.array([0.01, -0.01], type=pa.float64()),
            "ordinal_sector_score": pa.array([0.25, -0.25], type=pa.float64()),
        }
    )
    values = {
        "policy_hash": "8" * 64,
        "source_surface_hash": "7" * 64,
        "session_count": 1,
        "sector_ids": ("sector-a", "sector-b"),
        "available_value_count": 2,
        "table_content_hash": sector_ema_table_content_hash(table),
    }
    draft = SectorEmaAlphaManifest.model_construct(**values, manifest_hash="0" * 64)
    manifest = SectorEmaAlphaManifest(
        **values,
        manifest_hash=canonical_hash(draft.model_dump(mode="json", exclude={"manifest_hash"})),
    )
    return SectorEmaAlphaSurface(table=table, manifest=manifest)


def test_sector_ema_artifact_readback_rejects_tampered_table(tmp_path) -> None:
    surface = _sector_ema_surface()
    store = AlphaGoalResearchArtifactStore(tmp_path)
    store.publish_sector_ema_surface(surface)
    assert store.load_sector_ema_manifest(surface.manifest.manifest_hash) == surface.manifest

    table_path = (
        tmp_path
        / "alpha-research"
        / "goal-driven"
        / "sector-ema-surfaces"
        / f"{surface.manifest.manifest_hash}.parquet"
    )
    table_path.write_bytes(b"tampered")
    with pytest.raises(pa.ArrowInvalid):
        store.load_sector_ema_manifest(surface.manifest.manifest_hash)


def test_sector_context_is_causal_and_uses_four_shared_features(tmp_path) -> None:
    source, sectors = _source(60)
    policy = build_sector_context_policy(sector_revision="c" * 64)
    surface, _table, payload = compile_sector_context_surface(
        source_table=source,
        source_surface_hash="d" * 64,
        policy=policy,
        sector_by_listing_id=sectors,
    )
    assert surface.values.shape == (60, 2, 4)
    assert surface.values.flags.writeable is False
    assert np.isfinite(surface.values).sum() > 0

    changed = source.set_column(
        source.schema.get_field_index("fit_target"),
        "fit_target",
        pa.array(
            [
                (10.0 if index >= source.num_rows - 10 else value.as_py())
                for index, value in enumerate(source["fit_target"])
            ],
            type=pa.float64(),
        ),
    )
    future, _future_table, _future_payload = compile_sector_context_surface(
        source_table=changed,
        source_surface_hash="e" * 64,
        policy=policy,
        sector_by_listing_id=sectors,
    )
    np.testing.assert_array_equal(surface.values[:-1], future.values[:-1])

    store = SectorContextStore(tmp_path)
    store.publish(manifest=surface.manifest, payload=payload)
    np.testing.assert_array_equal(
        store.load(surface.manifest.manifest_hash).values,
        surface.values,
    )


@pytest.mark.parametrize(
    ("column", "edit", "code"),
    [
        ("holding_end_open_at", lambda values: [*values[:5], None, *values[6:]], "clock_invalid"),
        (
            "formation_close_at",
            lambda values: [*values[:13], values[13] + timedelta(seconds=1), *values[14:]],
            "clock_mismatch",
        ),
        (
            "holding_end_open_at",
            lambda values: [v - timedelta(hours=41) for v in values],
            "causal_order_invalid",
        ),
    ],
)
def test_sector_context_refuses_each_unusable_clock(column, edit, code) -> None:
    """regression: the clocks are checked as arrays, not one value at a time; a
    missing clock, one listing's close apart from its session's, and an availability no later
    than the close each still refuse by name."""
    source, sectors = _source(30)
    values = edit(source[column].to_pylist())
    index = source.schema.get_field_index(column)
    source = source.set_column(index, column, pa.array(values, type=source[column].type))
    with pytest.raises(SectorContextBoundaryError, match=f"sector_context_{code}$"):
        compile_sector_context_surface(
            source_table=source,
            source_surface_hash="d" * 64,
            policy=build_sector_context_policy(sector_revision="c" * 64),
            sector_by_listing_id=sectors,
        )


def _canonical_source(
    sessions: tuple[date, ...], listings: tuple[str, ...], *, outlier: float | None = None
) -> pa.Table:
    generator = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for session in sessions:
        for listing in listings:
            value = float(generator.normal(0.0, 0.02))
            rows.append(
                {
                    "formation_session": session,
                    "holding_end_session": session + timedelta(days=1),
                    "listing_id": listing,
                    "fit_target": value,
                    "simple_economic_return": float(np.expm1(value)),
                }
            )
    if outlier is not None:
        rows[3]["fit_target"] = outlier
        rows[3]["simple_economic_return"] = float(np.expm1(outlier))
    return pa.Table.from_pylist(rows)


_CANONICAL_LISTINGS = tuple(f"L{index:02d}" for index in range(12))
_CANONICAL_SECTORS = {
    listing: ("TECH" if index < 6 else "FIN") for index, listing in enumerate(_CANONICAL_LISTINGS)
}


def test_canonical_target_bounds_the_residual_then_repairs_the_sector_mean() -> None:
    """Canonical target bounds the residual then repairs the sector mean."""

    from alphalattice.investment.alpha_research.targets.canonical import (
        build_canonical_alpha_target_recipe,
        compile_canonical_alpha_target_surface,
    )

    sessions = (date(2026, 1, 5), date(2026, 1, 6))
    source = _canonical_source(sessions, _CANONICAL_LISTINGS, outlier=0.90)
    recipe = build_canonical_alpha_target_recipe(
        execution_outcome_recipe_id="NEXT_OPEN_TO_OPEN_ONE_SESSION",
        sector_revision="a" * 64,
    )
    surface = compile_canonical_alpha_target_surface(
        source_table=source, recipe=recipe, sector_by_listing_id=_CANONICAL_SECTORS
    )
    shape = (len(sessions), len(_CANONICAL_LISTINGS))
    columns = {
        name: np.asarray(
            surface.targets[name].to_numpy(zero_copy_only=False), dtype=np.float64
        ).reshape(shape)
        for name in (
            "raw_log_execution_return",
            "sector_center",
            "raw_residual",
            "bounded_residual",
            "redemeaned_residual",
            "fit_target",
        )
    }

    # The residual is the raw return minus its own Sector centre, before any bound.
    assert np.allclose(
        columns["raw_residual"], columns["raw_log_execution_return"] - columns["sector_center"]
    )
    # The bound acted on the residual, not the raw return.
    assert not np.allclose(columns["raw_residual"], columns["bounded_residual"])
    assert np.nanmax(np.abs(columns["bounded_residual"])) < np.nanmax(
        np.abs(columns["raw_residual"])
    )

    positions = {
        sector: [
            index
            for index, listing in enumerate(_CANONICAL_LISTINGS)
            if _CANONICAL_SECTORS[listing] == sector
        ]
        for sector in ("TECH", "FIN")
    }
    # The outlier sits in TECH on the first session, so that is the one Sector the
    # bound distorts. Naming it exactly -- rather than asserting every Sector is
    # distorted -- is the difference between testing the mechanism and testing the
    # fixture.
    assert abs(float(np.mean(columns["bounded_residual"][0, positions["TECH"]]))) > 1e-6
    for row in range(len(sessions)):
        for members in positions.values():
            # Wherever the bound left a Sector mean behind, the second demeaning
            # removes it; where it left none, the repair is a no-op.
            assert float(np.mean(columns["redemeaned_residual"][row, members])) == pytest.approx(
                0.0, abs=1e-12
            )
        scale = float(np.std(columns["redemeaned_residual"][row], ddof=1))
        assert scale == pytest.approx(
            float(surface.dispersion["cross_sectional_dispersion"][row].as_py())
        )
        # A population deviation would be a different number; the method fixes ddof=1.
        assert scale != pytest.approx(float(np.std(columns["redemeaned_residual"][row], ddof=0)))
        assert float(np.std(columns["fit_target"][row], ddof=1)) == pytest.approx(1.0)

    # The clip bound applies before re-demeaning, so the final lane may exceed it.
    assert np.nanmax(np.abs(columns["redemeaned_residual"])) > np.nanmax(
        np.abs(columns["bounded_residual"])
    )


def test_joint_primary_target_bounds_once_then_demeans_once_and_standardizes() -> None:
    """requirement: the Joint target follows its declared three-step composition exactly."""

    from alphalattice.foundation.feature_engine.producers.preprocessing import (
        robust_cross_section,
    )
    from alphalattice.investment.alpha_research.targets.canonical import (
        build_joint_primary_alpha_target_recipe,
        compile_canonical_alpha_target_surface,
    )

    sessions = (date(2026, 1, 5), date(2026, 1, 6))
    source = _canonical_source(sessions, _CANONICAL_LISTINGS, outlier=0.90)
    recipe = build_joint_primary_alpha_target_recipe(
        execution_outcome_recipe_id="NEXT_OPEN_TO_OPEN_ONE_SESSION",
        sector_revision="a" * 64,
    )
    surface = compile_canonical_alpha_target_surface(
        source_table=source, recipe=recipe, sector_by_listing_id=_CANONICAL_SECTORS
    )
    shape = (len(sessions), len(_CANONICAL_LISTINGS))
    raw = np.asarray(
        surface.targets["raw_log_execution_return"].to_numpy(zero_copy_only=False),
        dtype=np.float64,
    ).reshape(shape)
    bounded, _median, _mad, _lower, _upper = robust_cross_section.median_mad_winsor(
        raw, multiplier=3.5, mad_scale=1.4826
    )
    sector_positions = tuple(
        np.asarray(
            [
                index
                for index, listing in enumerate(_CANONICAL_LISTINGS)
                if _CANONICAL_SECTORS[listing] == sector
            ],
            dtype=np.int64,
        )
        for sector in ("FIN", "TECH")
    )
    residual = robust_cross_section.equal_sector_demean(bounded, sector_positions=sector_positions)
    expected = residual / np.std(residual, axis=1, ddof=1)[:, None]
    actual = np.asarray(
        surface.targets["fit_target"].to_numpy(zero_copy_only=False), dtype=np.float64
    ).reshape(shape)

    assert np.allclose(actual, expected, rtol=0.0, atol=1e-15)
    assert np.allclose(
        np.asarray(
            surface.targets["bounded_raw_log_execution_return"].to_numpy(zero_copy_only=False),
            dtype=np.float64,
        ).reshape(shape),
        bounded,
        rtol=0.0,
        atol=0.0,
    )
    for positions in sector_positions:
        assert np.allclose(np.mean(residual[:, positions], axis=1), 0.0, atol=1e-15)
    assert np.allclose(np.std(actual, axis=1, ddof=1), 1.0, atol=1e-15)


def test_total_return_target_bounds_and_centers_universe_without_sector_demean() -> None:
    """requirement: preserve cross-Sector ordering while scaling the whole Universe."""

    from alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section import (  # noqa: E501
        median_mad_winsor,
    )
    from alphalattice.investment.alpha_research.targets.total_return import (
        build_total_return_alpha_target_recipe,
        compile_total_return_alpha_target_surface,
    )

    sessions = (date(2026, 1, 5), date(2026, 1, 6))
    base = _canonical_source(sessions, _CANONICAL_LISTINGS, outlier=0.90)
    payload = base.to_pylist()
    for row in payload:
        if _CANONICAL_SECTORS[str(row["listing_id"])] == "TECH":
            row["fit_target"] = float(row["fit_target"]) + 0.04
            row["simple_economic_return"] = float(np.expm1(float(row["fit_target"])))
    source = pa.Table.from_pylist(payload)
    recipe = build_total_return_alpha_target_recipe(
        execution_outcome_recipe_id="NEXT_OPEN_TO_OPEN_ONE_SESSION"
    )
    surface = compile_total_return_alpha_target_surface(source_table=source, recipe=recipe)
    shape = (len(sessions), len(_CANONICAL_LISTINGS))
    raw = np.asarray(
        surface.targets["raw_log_execution_return"].to_numpy(), dtype=np.float64
    ).reshape(shape)
    bounded, _median, _mad, _lower, _upper = median_mad_winsor(
        raw, multiplier=3.5, mad_scale=1.4826
    )
    centered = bounded - np.mean(bounded, axis=1)[:, None]
    expected = centered / np.std(centered, axis=1, ddof=1)[:, None]
    actual = np.asarray(surface.targets["fit_target"].to_numpy(), dtype=np.float64).reshape(shape)

    assert np.allclose(actual, expected, rtol=0.0, atol=1e-15)
    assert np.allclose(np.mean(actual, axis=1), 0.0, atol=1e-15)
    assert np.allclose(np.std(actual, axis=1, ddof=1), 1.0, atol=1e-15)
    assert surface.targets["holding_end_session"].to_pylist() == [
        session + timedelta(days=1) for session in sessions for _listing in _CANONICAL_LISTINGS
    ]
    assert (
        surface.targets["simple_economic_return"].to_numpy().tobytes()
        == source.sort_by([("formation_session", "ascending"), ("listing_id", "ascending")])[
            "simple_economic_return"
        ]
        .to_numpy()
        .tobytes()
    )
    tech = np.asarray(
        [
            index
            for index, listing in enumerate(_CANONICAL_LISTINGS)
            if _CANONICAL_SECTORS[listing] == "TECH"
        ],
        dtype=np.int64,
    )
    fin = np.asarray(
        [
            index
            for index, listing in enumerate(_CANONICAL_LISTINGS)
            if _CANONICAL_SECTORS[listing] == "FIN"
        ],
        dtype=np.int64,
    )
    assert np.all(np.mean(actual[:, tech], axis=1) > np.mean(actual[:, fin], axis=1))
    assert int(np.sum(surface.dispersion["clipped_listing_count"].to_numpy())) > 0


def test_reference_targets_transform_entrants_without_refitting_or_shrinking_coverage():
    from alphalattice.investment.alpha_research.targets.total_return import (
        build_total_return_alpha_target_recipe,
        compile_total_return_alpha_target_surface,
    )

    sessions = tuple(date(2026, 1, 5) + timedelta(days=i) for i in range(3))
    original = np.random.default_rng(87).normal(0, 0.01, (3, 100))
    recipe = build_total_return_alpha_target_recipe(
        execution_outcome_recipe_id="NEXT_OPEN_TO_OPEN_ONE_SESSION",
        reference_coverage=True,
    )
    legacy = build_total_return_alpha_target_recipe(
        execution_outcome_recipe_id="NEXT_OPEN_TO_OPEN_ONE_SESSION"
    )
    assert "coverage_transform" not in legacy.model_dump(mode="json")
    assert recipe.recipe_hash != legacy.recipe_hash

    def compile(values, mask, nominal):
        width = values.shape[1]
        table = pa.table(
            {
                "formation_session": [day for day in sessions for _ in range(width)],
                "holding_end_session": [
                    day + timedelta(days=1) for day in sessions for _ in range(width)
                ],
                "listing_id": [f"L{i:03d}" for _ in sessions for i in range(width)],
                "fit_target": values.reshape(-1),
                "simple_economic_return": np.expm1(values).reshape(-1),
            }
        )
        return compile_total_return_alpha_target_surface(
            source_table=table,
            recipe=recipe,
            reference_eligible=mask,
            nominal_member_count=np.asarray(nominal, dtype=np.int64),
        )

    baseline = compile(original, np.ones(original.shape, dtype=np.bool_), [100] * 3)
    covered = np.column_stack((original, np.ones(3)))
    mask = np.ones(covered.shape, dtype=np.bool_)
    mask[:2, -1] = False
    result = compile(covered, mask, [100, 100, 101])
    before = baseline.targets["fit_target"].to_numpy().reshape(3, 100)
    after = result.targets["fit_target"].to_numpy().reshape(3, 101)
    np.testing.assert_array_equal(after[:2, :100].view("u8"), before[:2].view("u8"))
    assert result.bounds.slice(0, 2).equals(baseline.bounds.slice(0, 2))
    assert np.isfinite(after[:2, -1]).all()
    covered[:2, -1] *= 2
    repeated = compile(covered, mask, [100, 100, 101])
    np.testing.assert_array_equal(repeated.targets["fit_target"].to_numpy(), after.reshape(-1))
    assert result.lane_identity.reference_population_hash is not None
    assert not np.array_equal(after[-1, :100], before[-1])
    excluded = np.ones(original.shape, dtype=np.bool_)
    excluded[0, :3] = False
    shortage = compile(original, excluded, [100] * 3)
    assert shortage.dispersion["coverage"][0].as_py() == 0.97
    assert shortage.dispersion["unavailable_reason"][0].as_py() == "COVERAGE_BELOW_MINIMUM"
    assert np.isnan(shortage.targets["fit_target"].to_numpy()[:100]).all()
    with pytest.raises(ValueError, match="nominal_population_invalid"):
        compile(original, np.ones(original.shape, dtype=np.bool_), [99] * 3)


def test_total_return_target_is_installed_with_whole_universe_not_sector_authority() -> None:
    from alphalattice.investment.alpha_research.targets.authority import (
        TotalReturnTargetMethod,
        WholeUniverseAlphaTargetMethodBinding,
        installed_alpha_target_methods,
    )
    from alphalattice.investment.alpha_research.targets.total_return import (
        TOTAL_RETURN_ALPHA_TARGET_RECIPE_ID,
    )

    method = installed_alpha_target_methods(
        sector_revision="a" * 64,
        execution_outcome_recipe_id="NEXT_OPEN_TO_OPEN_ONE_SESSION",
    ).resolve(TOTAL_RETURN_ALPHA_TARGET_RECIPE_ID)
    assert isinstance(method, TotalReturnTargetMethod)
    binding = WholeUniverseAlphaTargetMethodBinding.create(
        method=method,
        listing_set_hash="b" * 64,
        execution_outcome_recipe_id="NEXT_OPEN_TO_OPEN_ONE_SESSION",
        causal_outcome_snapshot_hash="c" * 64,
        outcome_method_binding_hash="d" * 64,
        maturity_lag_sessions=2,
    )

    assert binding.cross_sectional_authority_id == "WHOLE_ACTIVE_UNIVERSE"
    assert binding.listing_set_hash == "b" * 64
    assert "sector_revision" not in binding.model_dump(mode="json")


def test_dispersion_forecast_takes_its_lag_from_the_sealed_outcome_method() -> None:
    """Dispersion forecast takes its lag from the sealed outcome method."""

    from alphalattice.investment.alpha_research.scaling.execution import (
        compile_cross_sectional_dispersion_forecast,
    )
    from alphalattice.investment.alpha_research.targets.canonical import (
        CanonicalAlphaTargetRecipeBinding,
        build_canonical_alpha_target_recipe,
        compile_canonical_alpha_target_surface,
        seal_canonical_alpha_target_evidence,
    )
    from alphalattice.investment.alpha_research.targets.catalog import (
        build_installed_alpha_target_catalog,
    )

    sessions = tuple(date(2026, 1, day) for day in (5, 6, 7, 8, 9, 12, 13, 14))
    recipe = build_canonical_alpha_target_recipe(
        execution_outcome_recipe_id="NEXT_OPEN_TO_OPEN_ONE_SESSION",
        sector_revision="a" * 64,
    )
    surface = compile_canonical_alpha_target_surface(
        source_table=_canonical_source(sessions, _CANONICAL_LISTINGS),
        recipe=recipe,
        sector_by_listing_id=_CANONICAL_SECTORS,
    )
    catalog_hash = build_installed_alpha_target_catalog().binding.catalog_hash

    for lag in (2, 6):
        binding = CanonicalAlphaTargetRecipeBinding.create(
            recipe=recipe,
            target_catalog_hash=catalog_hash,
            causal_outcome_snapshot_hash="b" * 64,
            outcome_method_binding_hash="c" * 64,
            maturity_lag_sessions=lag,
        )
        evidence = seal_canonical_alpha_target_evidence(surface=surface, recipe_binding=binding)
        forecast = compile_cross_sectional_dispersion_forecast(
            target_evidence=evidence,
            target_recipe_binding=binding,
            outcome_method_binding_hash="c" * 64,
            maturity_lag_sessions=lag,
        )
        realized = evidence.cross_sectional_dispersion
        assert forecast.forecast_values[lag:] == realized[: len(sessions) - lag]
        assert all(value is None for value in forecast.forecast_values[:lag])
        assert forecast.source_sessions[lag:] == sessions[: len(sessions) - lag]
        assert forecast.maturity_lag_sessions == lag


def test_dispersion_forecast_refuses_mismatched_authority_before_computing() -> None:
    """Every same-shaped substitution is refused, and refused early."""

    from alphalattice.investment.alpha_research.scaling.contracts import (
        CrossSectionalScalingError,
    )
    from alphalattice.investment.alpha_research.scaling.execution import (
        compile_cross_sectional_dispersion_forecast,
    )
    from alphalattice.investment.alpha_research.targets.canonical import (
        CanonicalAlphaTargetRecipeBinding,
        build_canonical_alpha_target_recipe,
        compile_canonical_alpha_target_surface,
        seal_canonical_alpha_target_evidence,
    )
    from alphalattice.investment.alpha_research.targets.catalog import (
        build_installed_alpha_target_catalog,
    )

    sessions = tuple(date(2026, 1, day) for day in (5, 6, 7, 8))
    recipe = build_canonical_alpha_target_recipe(
        execution_outcome_recipe_id="NEXT_OPEN_TO_OPEN_ONE_SESSION",
        sector_revision="a" * 64,
    )
    surface = compile_canonical_alpha_target_surface(
        source_table=_canonical_source(sessions, _CANONICAL_LISTINGS),
        recipe=recipe,
        sector_by_listing_id=_CANONICAL_SECTORS,
    )
    binding = CanonicalAlphaTargetRecipeBinding.create(
        recipe=recipe,
        target_catalog_hash=build_installed_alpha_target_catalog().binding.catalog_hash,
        causal_outcome_snapshot_hash="b" * 64,
        outcome_method_binding_hash="c" * 64,
        maturity_lag_sessions=2,
    )
    evidence = seal_canonical_alpha_target_evidence(surface=surface, recipe_binding=binding)
    baseline = {
        "target_evidence": evidence,
        "target_recipe_binding": binding,
        "outcome_method_binding_hash": "c" * 64,
        "maturity_lag_sessions": 2,
    }

    for override, expected in (
        ({"maturity_lag_sessions": 6}, "SCALING_MATURITY_LAG_MISMATCH"),
        ({"outcome_method_binding_hash": "d" * 64}, "SCALING_OUTCOME_METHOD_MISMATCH"),
        ({"recipe_id": "EWMA_XS_DISPERSION"}, "SCALING_RECIPE_NOT_INSTALLED"),
    ):
        with pytest.raises(CrossSectionalScalingError, match=expected):
            compile_cross_sectional_dispersion_forecast(**{**baseline, **override})

    # Evidence that belongs to a different binding is refused even though both
    # documents are individually well formed.
    other = CanonicalAlphaTargetRecipeBinding.create(
        recipe=recipe,
        target_catalog_hash=build_installed_alpha_target_catalog().binding.catalog_hash,
        causal_outcome_snapshot_hash="e" * 64,
        outcome_method_binding_hash="c" * 64,
        maturity_lag_sessions=2,
    )
    with pytest.raises(
        CrossSectionalScalingError, match="SCALING_TARGET_EVIDENCE_BINDING_MISMATCH"
    ):
        compile_cross_sectional_dispersion_forecast(**{**baseline, "target_recipe_binding": other})
