"""The research desk's two candidate inventories, the tests' fixtures since: the
four-card prototype and the 36-card role-aware Alpha set, built from the product's inventory
contracts (their loaders left the product with the PM's effort resolution)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from pydantic import BaseModel

from alphalattice.investment.portfolio_management.mandates.research_effort import (
    DesktopAlphaModelCard,
    DesktopPortfolioScenarioCard,
    DesktopRiskCandidateCard,
    ResearchDeskCandidateInventory,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _with_hash[ContractT: BaseModel](
    model: type[ContractT], values: Mapping[str, object], field: str
) -> ContractT:
    provisional = model.model_construct(**values, **{field: "0" * 64})
    identity = provisional.model_dump(mode="json", exclude={field})
    return cast(ContractT, model.model_validate({**values, field: canonical_hash(identity)}))


def _full_alpha_values() -> tuple[tuple[str, str, str, int, int, bool], ...]:
    alpha_values: list[tuple[str, str, str, int, int, bool]] = [
        ("benchmark.historical-mean", "historical_mean", "NONE", 0, 0, False),
        ("benchmark.zero-forecast", "zero_forecast", "NONE", 0, 0, False),
        ("diagnostic.quantile.tau-0.1", "quantile_regression", "DYNAMIC_ORDERED", 1, 55, False),
        ("diagnostic.quantile.tau-0.5", "quantile_regression", "DYNAMIC_ORDERED", 1, 55, False),
        ("diagnostic.quantile.tau-0.9", "quantile_regression", "DYNAMIC_ORDERED", 1, 55, False),
        ("model.ols-3", "ols_3", "DYNAMIC_ORDERED", 3, 3, False),
    ]
    alpha_values.extend(
        (
            f"model.ridge.alpha-{str(alpha_value).replace('.', 'p')}",
            "ridge",
            "DYNAMIC_ORDERED",
            1,
            55,
            False,
        )
        for alpha_value in (0.1, 1.0, 10.0, 100.0)
    )
    for multiplier in (0.01, 0.03, 0.1, 0.3, 1.0):
        token = str(multiplier).replace(".", "p")
        alpha_values.append(
            (
                f"model.lasso.alpha-max-x-{token}",
                "lasso",
                "DYNAMIC_ORDERED",
                1,
                55,
                False,
            )
        )
        alpha_values.extend(
            (
                f"model.elastic-net.alpha-max-x-{token}.l1-{str(ratio).replace('.', 'p')}",
                "elastic_net",
                "DYNAMIC_ORDERED",
                1,
                55,
                False,
            )
            for ratio in (0.25, 0.5, 0.75)
        )
    alpha_values.extend(
        (
            f"model.lightgbm.leaves-{leaves}.rate-{str(rate).replace('.', 'p')}",
            "lightgbm",
            "DYNAMIC_ORDERED",
            1,
            55,
            True,
        )
        for leaves in (15, 31)
        for rate in (0.03, 0.05, 0.1)
    )
    alpha_values = sorted(alpha_values, key=lambda value: value[0])
    if len(alpha_values) != 36:
        raise AssertionError("HIGH Alpha inventory must contain 36 registered cards")
    return tuple(alpha_values)


def _candidate_inventory(
    alpha_values: tuple[tuple[str, str, str, int, int, bool], ...],
) -> ResearchDeskCandidateInventory:
    alpha = tuple(
        _with_hash(
            DesktopAlphaModelCard,
            {
                "candidate_id": candidate_id,
                "family_id": family_id,
                "feature_shape": feature_shape,
                "minimum_factor_count": minimum,
                "maximum_factor_count": maximum,
                "secondary_feature_preprocessing": False,
                "deterministic_seed_required": seed_required,
            },
            "card_hash",
        )
        for candidate_id, family_id, feature_shape, minimum, maximum, seed_required in alpha_values
    )
    risk = tuple(
        _with_hash(DesktopRiskCandidateCard, {"candidate_id": candidate_id}, "card_hash")
        for candidate_id in ("sample", "ledoit_wolf", "vol_scaled_ledoit_wolf")
    )
    portfolios = tuple(
        _with_hash(
            DesktopPortfolioScenarioCard,
            {"scenario_id": scenario_id},
            "card_hash",
        )
        for scenario_id in (
            "risk_only_gmv",
            "forecast_mvo_conservative",
            "forecast_mvo_balanced",
            "forecast_mvo_alpha_tilted",
            "forecast_mvo_low_turnover",
            "black_litterman_mvo",
        )
    )
    values = {
        "alpha_models": alpha,
        "risk_candidates": risk,
        "portfolio_scenarios": portfolios,
    }
    return _with_hash(ResearchDeskCandidateInventory, values, "inventory_hash")


def load_initial_candidate_inventory() -> ResearchDeskCandidateInventory:
    """The accepted four-card prototype inventory."""
    return _candidate_inventory(
        (
            ("benchmark.zero-forecast", "zero_forecast", "NONE", 0, 0, False),
            ("benchmark.historical-mean", "historical_mean", "NONE", 0, 0, False),
            ("model.ridge.alpha-1p0", "ridge", "DYNAMIC_ORDERED", 1, 55, True),
            ("model.ridge.alpha-10p0", "ridge", "DYNAMIC_ORDERED", 1, 55, True),
        )
    )


def load_full_alpha_candidate_inventory() -> ResearchDeskCandidateInventory:
    """The role-aware 36-card Alpha inventory."""
    return _candidate_inventory(_full_alpha_values())


__all__ = ["load_full_alpha_candidate_inventory", "load_initial_candidate_inventory"]
