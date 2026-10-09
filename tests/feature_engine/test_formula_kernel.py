"""The factor formula kernel: the user's scale-invariant rule, the kernel against the independent
reference, a formula factor kept, specified and computed by the research path,
and the point-in-time leaves."""

from __future__ import annotations

import math
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    PointInTimeMark,
    SourceAvailabilityCatalog,
    installed_source_availability_catalog,
)
from alphalattice.foundation.feature_engine.catalog.research import (
    ResearchFeatureChange,
    plan_research_feature_change,
    research_feature_controls,
)
from alphalattice.foundation.feature_engine.catalog.service import (
    FeatureCatalogCrudError,
    FeatureCatalogCrudPlanner,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.formula import (
    FORMULA_ID,
    FORMULA_LEAVES,
    FORMULA_POINT_IN_TIME_ID,
    FORMULA_POINT_IN_TIME_REQUIRED_FIELDS,
    FORMULA_REQUIRED_FIELDS,
    formula_required_fields,
    formula_specification,
    formula_tree,
    reference_value,
)
from alphalattice.foundation.feature_engine.producers.factors.formula_language import FormulaError
from alphalattice.foundation.feature_engine.producers.factors.specifications import (
    FeatureFormulaSpecificationError,
    build_research_formula_specification,
    golden_example_frame,
)
from alphalattice.foundation.market_data_ops.publication.projection import (
    is_share_split,
    project_as_traded_series,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)
from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

FORMULAS = (
    "close / lag(close, 20) - 1",
    "ts_corr(close / lag(close, 1), volume / lag(volume, 1), 20)",
    "close * volume / ts_mean(close * volume, 21)",
    "where(close > open, high / low, low / high)",
    "ts_rank(adjusted_close / lag(adjusted_close, 5), 60)",
    "ewm(close / lag(close, 1) - 1, 3.5) / ts_std(close / lag(close, 1), 30)",
    "ts_beta(close / lag(close, 1), open / lag(open, 1), 40)",
    "sqrt(ts_cov(high / low, high / low, 10))",
    "not close > open or high / low > 1.01",
    "sign(delta(close / open, 3)) * abs(ts_min(low / close, 7))",
    "clip(ts_sum(close / lag(close, 1) - 1, 5), -0.1, 0.1)",
    "log(ts_max(high, 10) / ts_min(low, 10))",
)


def _declared(formula: str, *, skip: int = 0, factor_id: str = "formula_probe") -> FactorSpec:
    return FactorSpec(
        factor_id=factor_id,
        family=FactorFamily.PRICE_LEVEL_TREND,
        formula_ref=FORMULA_ID,
        formula=formula,
        window_sessions=1,
        lag_sessions=skip,
        return_convention="as declared",
        required_fields=("close_split_adjusted",),
        literature_sources=("research-local note",),
        minimum_observations=1,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        track=FactorTrack.MODEL,
    )


def _panel(seed: int = 7, listings: int = 4, rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for listing in range(listings):
        start = int(rng.integers(0, 60))
        count = rows - start
        close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, count)))
        opening = close * np.exp(rng.normal(0.0, 0.01, count))
        frame = pd.DataFrame(
            {
                "listing_id": f"L{listing}",
                "session_date": pd.bdate_range("2020-01-01", periods=rows)[start:].date,
                "close_split_adjusted": close,
                "open_split_adjusted": opening,
                "high_split_adjusted": np.maximum(opening, close) * 1.01,
                "low_split_adjusted": np.minimum(opening, close) * 0.99,
                "provider_adjusted_close": close * 0.97,
                "volume_raw": rng.uniform(1e5, 2e6, count),
            }
        )
        frame.loc[rng.random(count) < 0.02, "close_split_adjusted"] = np.nan
        parts.append(frame)
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=1)


@pytest.mark.parametrize(
    ("formula", "skip", "code"),
    [
        ("log(close)", 0, "factor_formula.level_use:log"),
        ("clip(close, 1, 2)", 0, "factor_formula.level_use:clip"),
        ("close > 5", 0, "factor_formula.units_differ:>"),
        ("close - open", 0, "factor_formula.not_scale_invariant"),
        ("close / lag(volume, 1)", 0, "factor_formula.not_scale_invariant"),
        ("adjusted_close / close", 0, "factor_formula.not_scale_invariant"),
        ("rank(close / open)", 0, "factor_formula.section_is_preprocessing:rank"),
        ("1 + 2", 0, "factor_formula.source_absent"),
        ("sector_label", 0, "factor_formula.leaf_unknown:sector_label"),
        ("ts_mean(close / lag(close, 1), 276)", 0, "factor_formula.lookback_exceeds_floor:277>276"),
        ("ts_mean(close / lag(close, 1), 275)", 1, "factor_formula.lookback_exceeds_floor:276>275"),
    ],
)
def test_a_formula_reads_the_leaves_only_where_no_later_split_or_dividend_moves_it(
    formula: str, skip: int, code: str
) -> None:
    """requirement (the user's scale-invariant rule): a level, a mixed scale, the cross-section,
    a formula of no source and a lookback past the budget, the skip counted, are refused by
    name."""

    with pytest.raises(FormulaError, match=f"^{re.escape(code)}$"):
        formula_tree(formula, skip=skip)


def test_a_later_split_or_dividend_leaves_every_admitted_formula_unchanged() -> None:
    """requirement (the user's scale-invariant rule): rescaling a listing's whole history as a
    later split (prices by 1/r, volume by r) and a later dividend (the adjusted close by c) do
    moves no admitted formula's value."""

    registry = default_extension_kernel_registry()
    source = _panel()
    rescaled = source.copy()
    for field in ("open_split_adjusted", "high_split_adjusted", "low_split_adjusted"):
        rescaled[field] = rescaled[field] / 4.0
    rescaled["close_split_adjusted"] = rescaled["close_split_adjusted"] / 4.0
    rescaled["volume_raw"] = rescaled["volume_raw"] * 4.0
    rescaled["provider_adjusted_close"] = rescaled["provider_adjusted_close"] * 0.93
    for formula in FORMULAS:
        kept = formula_specification(_declared(formula))
        before = registry.compute(source, kept)
        after = registry.compute(rescaled, kept)
        assert before.notna().any(), formula
        np.testing.assert_allclose(after, before, rtol=1e-9, atol=1e-12, err_msg=formula)


@pytest.mark.parametrize("skip", [0, 2])
def test_the_kernel_computes_what_the_independent_reference_states(skip: int) -> None:
    """requirement (EX): on shuffled rows, late starts and missing closes, each listing's own
    rows give the values the plain-Python reference states, missing where it states none."""

    registry = default_extension_kernel_registry()
    source = _panel()
    for formula in FORMULAS:
        kept = formula_specification(_declared(formula, skip=skip))
        values = registry.compute(source, kept)
        tree = formula_tree(kept.formula, skip=skip)
        for _listing, rows in source.groupby("listing_id"):
            ordered = rows.sort_values("session_date")
            ends = (len(ordered) - 1, len(ordered) // 2, kept.minimum_observations - 2)
            for end in (end for end in ends if end >= 0):
                prefix = ordered.iloc[: end + 1]
                expected = reference_value(
                    tree,
                    {
                        leaf: prefix[field].to_numpy(dtype=float).tolist()
                        for leaf, field in FORMULA_LEAVES.items()
                        if field in prefix
                    },
                    skip=skip,
                )
                got = float(values.loc[prefix.index[-1]])
                if expected is None:
                    assert math.isnan(got), (formula, end)
                else:
                    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12), (formula, end)


def test_the_research_plan_keeps_a_formula_factor_as_its_kernel_derives_it() -> None:
    """requirement (EX): the plan keeps the canonical formula and derives the fields, the window
    and the minimum rows; its research specification states goldens from the reference, the
    registry computes them, and a spec the plan did not derive is refused."""

    catalog = FeatureCatalog.load()
    registry = default_extension_kernel_registry()
    revision = FeatureCatalogCrudPlanner(registry).revision(catalog)
    declared = _declared(
        "ts_corr(close/lag(close,1), volume/delay(volume,1), n=20)",
        skip=1,
        factor_id="volume_price_corr_20",
    )
    request = ResearchFeatureChange(
        input_binding_hash="a" * 64,
        base_revision_hash=revision.revision_hash,
        edits=[
            {
                "operation": "CREATE",
                "factor_id": declared.factor_id,
                "specification": declared,
                "preprocessing_recipe": "TIME_SERIES_ABSOLUTE_STATE_ROBUST",
            }
        ],
        reason="a formula factor",
    )
    plan = plan_research_feature_change(
        catalog=catalog, base=revision, source_panel_snapshot_hash="b" * 64, request=request
    )
    kept = next(
        v.specification for v in plan.candidate.features if v.factor_id == "volume_price_corr_20"
    )
    assert kept.formula == "ts_corr(close / lag(close, 1), volume / lag(volume, 1), 20)"
    assert (kept.required_fields, kept.window_sessions, kept.minimum_observations) == (
        FORMULA_REQUIRED_FIELDS,
        21,
        22,
    )
    assert plan.work.base_compute_factor_ids == ("volume_price_corr_20",)
    assert plan.preprocessing_recipes == {
        "volume_price_corr_20": "TIME_SERIES_ABSOLUTE_STATE_ROBUST"
    }
    specification = build_research_formula_specification(
        kept, source_session_count=500, preprocessing_recipe="TIME_SERIES_ABSOLUTE_STATE_ROBUST"
    )
    assert specification.minimum_ordered_source_rows == 22
    assert specification.preprocessing_role == "TIME_SERIES_ABSOLUTE_STATE_ROBUST"
    for example in specification.golden_examples:
        computed = registry.compute(golden_example_frame(example), kept)
        if example.expected_value is None:
            assert math.isnan(computed.iloc[-1])
        else:
            assert computed.iloc[-1] == pytest.approx(example.expected_value, rel=1e-12)
    with pytest.raises(FeatureFormulaSpecificationError, match="formula_specification_not_derived"):
        build_research_formula_specification(
            declared, source_session_count=500, preprocessing_recipe="ROBUST_SECTOR_NEUTRAL_Z"
        )
    with pytest.raises(FeatureFormulaSpecificationError, match="preprocessing_recipe_required"):
        build_research_formula_specification(kept, source_session_count=500)
    refused = request.edits[0].model_copy(
        update={"specification": declared.model_copy(update={"formula": "close - open"})}
    )
    with pytest.raises(FeatureCatalogCrudError, match="`feature controls`") as caught:
        plan_research_feature_change(
            catalog=catalog,
            base=revision,
            source_panel_snapshot_hash="b" * 64,
            request=request.model_copy(update={"edits": (refused,)}),
        )
    assert caught.value.failure_code == "factor_formula.not_scale_invariant"
    language = research_feature_controls()["formula_language"]
    assert language["formula_ref"] == FORMULA_ID
    assert "rank(x)" not in language["operators"]


def test_a_formula_factor_names_its_preprocessing_recipe_and_no_other_factor_does() -> None:
    """requirement (the user, 2026-09-30): a formula's declaration chooses its recipe from the
    catalog, never a default; a missing or unadmitted recipe and a recipe on an installed
    kernel's factor are refused by name; a rename carries the recipe and a retirement drops it."""

    catalog = FeatureCatalog.load()
    revision = FeatureCatalogCrudPlanner(default_extension_kernel_registry()).revision(catalog)
    declared = _declared("close / lag(close, 5) - 1", factor_id="reversal_5")

    def change(*edits: dict[str, object]) -> ResearchFeatureChange:
        return ResearchFeatureChange(
            input_binding_hash="a" * 64,
            base_revision_hash=revision.revision_hash,
            edits=list(edits),
            reason="recipes",
        )

    create = {"operation": "CREATE", "factor_id": "reversal_5", "specification": declared}
    for recipe, code in (
        (None, "factor_formula.preprocessing_recipe_required:ROBUST_SECTOR_NEUTRAL_Z,"),
        ("JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z", "preprocessing_recipe_not_admitted"),
    ):
        edit = create if recipe is None else {**create, "preprocessing_recipe": recipe}
        with pytest.raises(FeatureCatalogCrudError) as caught:
            plan_research_feature_change(
                catalog=catalog,
                base=revision,
                source_panel_snapshot_hash="b" * 64,
                request=change(edit),
            )
        assert code in caught.value.failure_code
    installed = catalog.factors[0].model_copy(update={"factor_id": "zz_copy", "core_anchor": False})
    with pytest.raises(FeatureCatalogCrudError) as caught:
        plan_research_feature_change(
            catalog=catalog,
            base=revision,
            source_panel_snapshot_hash="b" * 64,
            request=change(
                {
                    "operation": "CREATE",
                    "factor_id": "zz_copy",
                    "specification": installed,
                    "preprocessing_recipe": "ROBUST_SECTOR_NEUTRAL_Z",
                }
            ),
        )
    assert caught.value.failure_code == "feature_research.preprocessing_recipe_formula_only"
    first = plan_research_feature_change(
        catalog=catalog,
        base=revision,
        source_panel_snapshot_hash="b" * 64,
        request=change({**create, "preprocessing_recipe": "ROBUST_SECTOR_NEUTRAL_Z"}),
    )
    renamed = plan_research_feature_change(
        catalog=FeatureCatalog.from_payload(first.candidate_payload),
        base=first.candidate,
        source_panel_snapshot_hash="b" * 64,
        request=ResearchFeatureChange(
            input_binding_hash="a" * 64,
            base_revision_hash=first.candidate.revision_hash,
            parent_plan_hash=first.plan_hash,
            edits=[
                {
                    "operation": "RENAME",
                    "factor_id": "reversal_5",
                    "specification": declared.model_copy(update={"factor_id": "reversal_five"}),
                    "preprocessing_recipe": "TIME_SERIES_ABSOLUTE_STATE_ROBUST",
                }
            ],
            reason="rename",
        ),
        preprocessing_recipes=first.preprocessing_recipes,
    )
    assert renamed.preprocessing_recipes == {"reversal_five": "TIME_SERIES_ABSOLUTE_STATE_ROBUST"}
    retired = plan_research_feature_change(
        catalog=FeatureCatalog.from_payload(renamed.candidate_payload),
        base=renamed.candidate,
        source_panel_snapshot_hash="b" * 64,
        request=ResearchFeatureChange(
            input_binding_hash="a" * 64,
            base_revision_hash=renamed.candidate.revision_hash,
            parent_plan_hash=renamed.plan_hash,
            edits=[{"operation": "RETIRE", "factor_id": "reversal_five"}],
            reason="retire",
        ),
        preprocessing_recipes=renamed.preprocessing_recipes,
    )
    assert retired.preprocessing_recipes == {}
    assert "preprocessing_recipes" not in retired.model_dump(mode="json")


def test_the_sector_leaf_is_a_field_only_a_formula_reading_it_names() -> None:
    """The sector leaf is a field only a formula reading it names."""

    from alphalattice.foundation.feature_engine.catalog.contracts import (
        desktop_core_feature_bundle,
    )
    from alphalattice.foundation.feature_engine.producers.factors.formula import (
        FORMULA_SECTOR_FIELD,
        FORMULA_SECTOR_ID,
    )

    assert FORMULA_LEAVES["sector_return"] == FORMULA_SECTOR_FIELD == "sector_return_log"
    assert formula_required_fields(formula_tree("close / lag(close, 20)")) == (
        FORMULA_REQUIRED_FIELDS
    )
    reads = "ts_sum(log(adjusted_close / lag(adjusted_close, 1)) - sector_return, 5)"
    assert formula_required_fields(formula_tree(reads)) == tuple(
        sorted({*FORMULA_REQUIRED_FIELDS, FORMULA_SECTOR_FIELD})
    )
    assert formula_specification(_declared("close / lag(close, 20)")).formula_ref == FORMULA_ID
    derived = formula_specification(_declared(reads, factor_id="formula_sector_gap_5"))
    assert derived.formula_ref == FORMULA_SECTOR_ID
    assert formula_specification(derived) == derived
    built = build_research_formula_specification(
        derived, source_session_count=300, preprocessing_recipe="ROBUST_UNIVERSE_Z"
    )
    assert built.preprocessing_role == "ROBUST_UNIVERSE_Z"
    registry = default_extension_kernel_registry()
    for example in built.golden_examples:
        assert FORMULA_SECTOR_FIELD in example.input_fields
        computed = float(registry.compute(golden_example_frame(example), derived).iloc[-1])
        if example.expected_value is None:
            assert math.isnan(computed)
        else:
            assert computed == pytest.approx(example.expected_value, rel=1e-10, abs=1e-10)
    stray = derived.model_copy(
        update={"required_fields": tuple(sorted({*derived.required_fields, "market_return_log"}))}
    )
    with pytest.raises(ValueError, match=r"required fields differ from its kernel"):
        registry.implementation_hash(stray, core_bundle=desktop_core_feature_bundle())
    priced = derived.model_copy(update={"formula_ref": FORMULA_ID})
    with pytest.raises(ValueError, match=r"required fields differ from its kernel"):
        registry.implementation_hash(priced, core_bundle=desktop_core_feature_bundle())


def _stored(
    sessions: list[date], traded: np.ndarray, shares: np.ndarray, ratios: list[tuple[date, float]]
) -> list[RawDailyBar]:
    """The bars a provider stores once it recorded `ratios`: each session's prices divided and its
    volume multiplied by every ratio after it, as its split mechanics do."""

    bars = []
    for at, price, volume in zip(sessions, traded, shares, strict=True):
        later = math.prod(r for when, r in ratios if when > at)
        bars.append(
            RawDailyBar(
                listing_id="L1",
                provider="yfinance",
                session_date=at,
                open=price / later,
                high=price * 1.01 / later,
                low=price * 0.99 / later,
                close=price / later,
                volume=round(volume * later),
            )
        )
    return bars


def _as_of_download(bars: list[RawDailyBar], ratios: list[tuple[date, float]]) -> pd.DataFrame:
    actions = [
        CorporateActionEvent(
            listing_id="L1",
            provider="yfinance",
            effective_date=when,
            action_kind="SPLIT",
            new_shares_per_old_share=ratio,
        )
        for when, ratio in ratios
    ]
    traded = project_as_traded_series(bars, actions, daily_price_basis="split_adjusted")
    return pd.DataFrame(
        [
            {
                "listing_id": bar.listing_id,
                "session_date": bar.session_date,
                "open_split_adjusted": bar.open,
                "high_split_adjusted": bar.high,
                "low_split_adjusted": bar.low,
                "close_split_adjusted": bar.close,
                "volume_raw": float(bar.volume),
                "provider_adjusted_close": bar.close,
                **{k: v for k, v in row.__dict__.items() if k != "session_date"},
            }
            for bar, row in zip(bars, traded, strict=True)
        ]
    )


@pytest.mark.parametrize("skip", [0, 2])
def test_a_point_in_time_leaf_reads_each_session_as_it_stood_and_admits_a_level(skip: int) -> None:
    """A point in time leaf reads each session as it stood and admits a level."""

    for ratio in (1.589, 1.281, 1.253, 1.128, 1.196):  # RTX, GE twice, DHR, MMM
        assert not is_share_split(ratio), ratio
    for ratio in (2.0, 3.0, 4.0, 10.0, 20.0, 50.0, 1.5, 1.05, 0.125, 0.333333, 0.05):
        assert is_share_split(ratio), ratio
    for formula in ("close_pit > 5", "log(close_pit)", "clip(close_pit, 1, 2)"):
        assert formula_required_fields(formula_tree(formula)) == (
            FORMULA_POINT_IN_TIME_REQUIRED_FIELDS
        )
    with pytest.raises(FormulaError, match=r"^factor_formula\.not_scale_invariant$"):
        formula_tree("close_pit * volume")

    sessions = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
    rng = np.random.default_rng(3)
    traded = 40.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 30)))
    shares = 2_000.0 * rng.integers(1, 50, 30)
    recorded = [(sessions[10], 1.281), (sessions[20], 2.0)]  # a spin-off, then a 2-for-1 split
    later = [*recorded, (date(2024, 3, 1), 3.0)]  # a 3-for-1 split after the last bar
    first = _as_of_download(_stored(sessions, traded, shares, recorded), recorded)
    second = _as_of_download(_stored(sessions, traded, shares, later), later)
    np.testing.assert_allclose(first["close_as_traded"], traded, rtol=1e-12)
    np.testing.assert_allclose(second["volume_as_traded"], shares, rtol=1e-3)

    registry = default_extension_kernel_registry()
    for formula, share_only in (("ts_mean(close_pit, 5)", False), ("ts_sum(volume_pit, 5)", True)):
        kept = formula_specification(_declared(formula, skip=skip))
        assert kept.formula_ref == FORMULA_POINT_IN_TIME_ID
        values = registry.compute(first, kept).to_numpy()
        np.testing.assert_allclose(registry.compute(second, kept), values, rtol=1e-3)
        for end in range(len(sessions)):
            window = range(end - skip - 4, end - skip + 1)
            if window.start < 0:
                assert math.isnan(values[end]), (formula, end)
                continue
            between = [
                math.prod(r for when, r in recorded if sessions[t] < when <= sessions[end])
                if not share_only
                else math.prod(
                    r
                    for when, r in recorded
                    if sessions[t] < when <= sessions[end] and is_share_split(r)
                )
                for t in window
            ]
            expected = (
                sum(shares[t] * k for t, k in zip(window, between, strict=True))
                if share_only
                else sum(traded[t] / k for t, k in zip(window, between, strict=True)) / 5
            )
            assert values[end] == pytest.approx(expected, rel=1e-3), (formula, end)

    derived = formula_specification(_declared("ts_mean(close_pit, 5)", factor_id="price_level_5"))
    built = build_research_formula_specification(
        derived, source_session_count=300, preprocessing_recipe="ROBUST_UNIVERSE_Z"
    )
    for example in built.golden_examples:
        assert set(FORMULA_POINT_IN_TIME_REQUIRED_FIELDS) == set(example.input_fields)
        computed = float(registry.compute(golden_example_frame(example), derived).iloc[-1])
        if example.expected_value is None:
            assert math.isnan(computed)
        else:
            assert computed == pytest.approx(example.expected_value, rel=1e-10, abs=1e-10)


def test_every_source_field_carries_its_point_in_time_mark_and_a_rescaled_leaf_has_no_level() -> (
    None
):
    """Requirement: the availability catalog marks every field it maps; a formula leaf is
    read at a level exactly where no later event rescales its field, and a catalog recorded before
    the marks validates as it was recorded."""

    catalog = installed_source_availability_catalog()
    assert set(catalog.point_in_time) == set(catalog.field_authorities)
    for leaf, field in FORMULA_LEAVES.items():
        if catalog.point_in_time[field] is PointInTimeMark.RESCALED_BY_LATER_EVENTS:
            with pytest.raises(FormulaError):
                formula_tree(f"{leaf} > 1")
        else:
            formula_tree(f"{leaf} > 1")
    before = SourceAvailabilityCatalog.create(
        catalog_id=catalog.catalog_id,
        owners=catalog.owners,
        field_authorities=catalog.field_authorities,
    ).model_dump(mode="json")
    assert "point_in_time" not in before
    assert SourceAvailabilityCatalog.model_validate(before).catalog_hash == before["catalog_hash"]


def test_the_daily_build_carries_the_as_traded_fields_its_catalog_reads() -> None:
    """The daily build carries the as traded fields its catalog reads."""

    from alphalattice.foundation.feature_engine.producers.base_materializer import (
        BaseFeatureMaterializer,
    )
    from alphalattice.foundation.feature_engine.producers.factors.formula import (
        FORMULA_POINT_IN_TIME_FIELDS,
        FORMULA_RESEARCH_IDS,
        FORMULA_SECTOR_POINT_IN_TIME_ID,
    )
    from alphalattice.foundation.feature_engine.runtime.service import catalog_reads_as_traded
    from alphalattice.foundation.market_data_ops.publication.projection import AS_TRADED_FIELDS

    assert set(FORMULA_POINT_IN_TIME_FIELDS) == set(AS_TRADED_FIELDS)
    assert FORMULA_POINT_IN_TIME_ID not in FORMULA_RESEARCH_IDS
    assert FORMULA_SECTOR_POINT_IN_TIME_ID in FORMULA_RESEARCH_IDS
    shipped = FeatureCatalog.load()
    assert not catalog_reads_as_traded(shipped)
    registry = default_extension_kernel_registry()
    revision = FeatureCatalogCrudPlanner(registry).revision(shipped)
    declared = _declared("ts_mean(close_pit, 5)", factor_id="price_level_5")
    plan = plan_research_feature_change(
        catalog=shipped,
        base=revision,
        source_panel_snapshot_hash="b" * 64,
        request=ResearchFeatureChange(
            input_binding_hash="a" * 64,
            base_revision_hash=revision.revision_hash,
            edits=[
                {
                    "operation": "CREATE",
                    "factor_id": "price_level_5",
                    "specification": declared,
                    "preprocessing_recipe": "ROBUST_UNIVERSE_Z",
                }
            ],
            reason="a price level",
        ),
    )
    activated = FeatureCatalog.from_payload(plan.candidate_payload)
    assert catalog_reads_as_traded(activated)

    sessions = [date(2024, 1, 1) + timedelta(days=i) for i in range(40)]
    rng = np.random.default_rng(5)
    traded = 40.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 40)))
    shares = 2_000.0 * rng.integers(1, 50, 40)
    recorded = [(sessions[10], 1.281), (sessions[20], 2.0)]
    frame = _as_of_download(_stored(sessions, traded, shares, recorded), recorded)
    for name in ("open", "high", "low", "close"):
        frame[f"{name}_raw"] = frame[f"{name}_split_adjusted"]
    plain = frame.drop(columns=["listing_id", *AS_TRADED_FIELDS])
    block = BaseFeatureMaterializer(activated, kernel_registry=registry).materialize_listing(
        listing_id="L1", projected_bars=frame.drop(columns=["listing_id"]), market_bars=plain
    )
    kept = next(spec for spec in activated.factors if spec.factor_id == "price_level_5")
    np.testing.assert_allclose(
        block.values["price_level_5"].to_numpy(dtype=float),
        registry.compute(frame, kept).to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    before = BaseFeatureMaterializer(shipped).materialize_listing(
        listing_id="L1", projected_bars=plain, market_bars=plain
    )
    after = BaseFeatureMaterializer(shipped).materialize_listing(
        listing_id="L1", projected_bars=frame.drop(columns=["listing_id"]), market_bars=plain
    )
    pd.testing.assert_frame_equal(after.values, before.values)


def test_a_one_session_formula_reviews_its_empty_boundary_golden() -> None:
    """Regression: a formula needing one session has a boundary golden one
    row short, which is no row at all; the review computed it and raised IndexError, so a
    completed trial's factor could not be reviewed. Its result is the missing value it expects."""

    from alphalattice.control.product_host.research_authoring.feature_extensions import (
        formula_goldens,
    )

    catalog = FeatureCatalog.load()
    revision = FeatureCatalogCrudPlanner(default_extension_kernel_registry()).revision(catalog)
    declared = _declared("close / open - 1", factor_id="open_to_close_return")
    request = ResearchFeatureChange(
        input_binding_hash="a" * 64,
        base_revision_hash=revision.revision_hash,
        edits=[
            {
                "operation": "CREATE",
                "factor_id": declared.factor_id,
                "specification": declared,
                "preprocessing_recipe": "TIME_SERIES_ABSOLUTE_STATE_ROBUST",
            }
        ],
        reason="a one-session formula",
    )
    plan = plan_research_feature_change(
        catalog=catalog, base=revision, source_panel_snapshot_hash="b" * 64, request=request
    )
    planned = {v.factor_id: v.specification for v in plan.candidate.features}
    kept = planned[declared.factor_id]
    specification = build_research_formula_specification(
        kept, source_session_count=500, preprocessing_recipe="TIME_SERIES_ABSOLUTE_STATE_ROBUST"
    )
    short = next(e for e in specification.golden_examples if e.label == "one-row-short-is-missing")
    assert (kept.minimum_observations, short.row_count) == (1, 0)
    goldens = {row["label"]: row for row in formula_goldens(kept, specification)}
    assert goldens["one-row-short-is-missing"]["computed"] is None
    assert all(row["within_tolerance"] for row in goldens.values())


_SMALLEST = {
    "abs(x)": "abs(close / open - 1)",
    "clip(x, lo, hi)": "clip(close / open - 1, -0.1, 0.1)",
    "log(x)": "log(close / open)",
    "max(a, b)": "max(close / open, 1)",
    "min(a, b)": "min(close / open, 1)",
    "sign(x)": "sign(close - open)",
    "sqrt(x)": "sqrt(volume / lag(volume, 1))",
    "where(c, a, b)": "where(close > open, 1, -1)",
    "delta(x, n)": "delta(close, 1) / close",
    "ewm(x, halflife)": "ewm(close / lag(close, 1), 1)",
    "lag(x, n)": "lag(close, 1) / close - 1",
    "ts_beta(y, x, n)": "ts_beta(close / lag(close, 1), volume / lag(volume, 1), 2)",
    "ts_corr(x, y, n)": "ts_corr(close, volume, 2)",
    "ts_cov(x, y, n)": "ts_cov(close, volume, 2)",
    "ts_max(x, n)": "ts_max(high / close, 1)",
    "ts_mean(x, n)": "ts_mean(close / open, 1)",
    "ts_min(x, n)": "ts_min(low / close, 1)",
    "ts_rank(x, n)": "ts_rank(close, 2)",
    "ts_std(x, n)": "ts_std(close / lag(close, 1), 2)",
    "ts_sum(x, n)": "ts_sum(close / lag(close, 1) - 1, 1)",
}
"""Each operator of the formula language in its smallest legal formula: the fewest sessions."""


def test_every_operators_smallest_formula_runs_its_goldens() -> None:
    """Every operator's smallest formula passes its numerical goldens."""

    from alphalattice.control.product_host.research_authoring.feature_extensions import (
        formula_controls,
        formula_goldens,
    )

    assert set(_SMALLEST) == set(formula_controls()["operators"])
    catalog = FeatureCatalog.load()
    revision = FeatureCatalogCrudPlanner(default_extension_kernel_registry()).revision(catalog)
    recipe = "TIME_SERIES_ABSOLUTE_STATE_ROBUST"
    for operator, formula in _SMALLEST.items():
        declared = _declared(formula, factor_id="smallest_formula")
        request = ResearchFeatureChange(
            input_binding_hash="a" * 64,
            base_revision_hash=revision.revision_hash,
            edits=[
                {
                    "operation": "CREATE",
                    "factor_id": declared.factor_id,
                    "specification": declared,
                    "preprocessing_recipe": recipe,
                }
            ],
            reason="the smallest formula of an operator",
        )
        plan = plan_research_feature_change(
            catalog=catalog, base=revision, source_panel_snapshot_hash="b" * 64, request=request
        )
        kept = {v.factor_id: v.specification for v in plan.candidate.features}[declared.factor_id]
        specification = build_research_formula_specification(
            kept, source_session_count=500, preprocessing_recipe=recipe
        )
        goldens = formula_goldens(kept, specification)
        assert goldens and all(row["within_tolerance"] for row in goldens), (operator, goldens)
