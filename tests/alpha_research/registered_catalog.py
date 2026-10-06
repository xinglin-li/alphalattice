"""The registered Alpha candidate-card catalog, kept as the tests' fixture since the goal loop
that read it retired (GR3): the numerical tests build their request from it."""

from __future__ import annotations

from alphalattice.investment.alpha_research.experiments.contracts import (
    AlphaCandidateRole,
    AlphaExperimentCard,
    seal_contract,
)
from alphalattice.investment.portfolio_management.mandates.research_effort import (
    ResearchDeskCandidateInventory,
)
from tests.alpha_research.candidate_inventories import (
    load_full_alpha_candidate_inventory,
    load_initial_candidate_inventory,
)

REGISTERED_ALPHA_CANDIDATE_IDS = tuple(
    value.candidate_id for value in load_full_alpha_candidate_inventory().alpha_models
)


class AlphaCatalogAdmissionError(ValueError):
    """Raised when current Research Effort authority differs from the frozen catalog."""


def _card_values(
    *,
    candidate_id: str,
    source_card_hash: str,
    factor_count: int,
    ordered_factor_ids: tuple[str, ...],
) -> dict[str, object]:
    common: dict[str, object] = {
        "kind": "AlphaExperimentCard",
        "candidate_id": candidate_id,
        "source_card_hash": source_card_hash,
        "secondary_feature_preprocessing": False,
    }
    if candidate_id == "benchmark.zero-forecast":
        return {
            **common,
            "family_id": "zero_forecast",
            "role": AlphaCandidateRole.BENCHMARK,
            "feature_shape": "NONE",
            "factor_count": 0,
            "selection_eligible": False,
            "package_identity": "alpha-research-deterministic",
        }
    if candidate_id == "benchmark.historical-mean":
        return {
            **common,
            "family_id": "historical_mean",
            "role": AlphaCandidateRole.BENCHMARK,
            "feature_shape": "NONE",
            "factor_count": 0,
            "selection_eligible": False,
            "package_identity": "alpha-research-deterministic",
        }
    quantile = {
        "diagnostic.quantile.tau-0.1": 0.1,
        "diagnostic.quantile.tau-0.5": 0.5,
        "diagnostic.quantile.tau-0.9": 0.9,
    }.get(candidate_id)
    if quantile is not None:
        return {
            **common,
            "family_id": "quantile_regression",
            "role": AlphaCandidateRole.DIAGNOSTIC_ONLY_BOUNDED,
            "feature_shape": "FROZEN_ORDERED",
            "factor_count": factor_count,
            "quantile_tau": quantile,
            "maximum_iterations": 10000,
            "deterministic_seed": 1729,
            "selection_eligible": False,
            "package_identity": "statsmodels==0.14.6",
        }
    if candidate_id == "model.ols-3":
        missing = tuple(
            value for value in ("mom_252_21", "rev_21", "vol_63") if value not in ordered_factor_ids
        )
        if not missing:
            raise AlphaCatalogAdmissionError("OLS3_CURRENT_FOUNDATION_UNEXPECTEDLY_COMPLETE")
        return {
            **common,
            "family_id": "ols_3",
            "role": AlphaCandidateRole.NOT_ADMITTED,
            "feature_shape": "FROZEN_ORDERED",
            "factor_count": 3,
            "required_feature_ids": ("mom_252_21", "rev_21", "vol_63"),
            "admission_status": "NOT_ADMITTED_MISSING_FACTORS",
            "selection_eligible": False,
            "package_identity": "statsmodels==0.14.6",
        }
    ridge = {
        f"model.ridge.alpha-{str(value).replace('.', 'p')}": value
        for value in (0.1, 1.0, 10.0, 100.0)
    }.get(candidate_id)
    if ridge is not None:
        return {
            **common,
            "family_id": "ridge",
            "role": AlphaCandidateRole.REGULARIZED_ALPHA,
            "feature_shape": "FROZEN_ORDERED",
            "factor_count": factor_count,
            "alpha": ridge,
            "fit_intercept": True,
            "solver": "svd",
            "tolerance": 1e-8,
            "selection_eligible": True,
            "package_identity": "scikit-learn==1.9.0",
        }
    for multiplier in (0.01, 0.03, 0.1, 0.3, 1.0):
        token = str(multiplier).replace(".", "p")
        if candidate_id == f"model.lasso.alpha-max-x-{token}":
            return {
                **common,
                "family_id": "lasso",
                "role": AlphaCandidateRole.REGULARIZED_ALPHA,
                "feature_shape": "FROZEN_ORDERED",
                "factor_count": factor_count,
                "alpha_multiplier": multiplier,
                "fit_intercept": True,
                "tolerance": 1e-8,
                "maximum_iterations": 10000,
                "selection_eligible": True,
                "package_identity": "scikit-learn==1.9.0",
            }
        for ratio in (0.25, 0.5, 0.75):
            ratio_token = str(ratio).replace(".", "p")
            if candidate_id == f"model.elastic-net.alpha-max-x-{token}.l1-{ratio_token}":
                return {
                    **common,
                    "family_id": "elastic_net",
                    "role": AlphaCandidateRole.REGULARIZED_ALPHA,
                    "feature_shape": "FROZEN_ORDERED",
                    "factor_count": factor_count,
                    "alpha_multiplier": multiplier,
                    "l1_ratio": ratio,
                    "fit_intercept": True,
                    "tolerance": 1e-8,
                    "maximum_iterations": 10000,
                    "selection_eligible": True,
                    "package_identity": "scikit-learn==1.9.0",
                }
    for leaves in (15, 31):
        for rate in (0.03, 0.05, 0.1):
            token = str(rate).replace(".", "p")
            if candidate_id == f"model.lightgbm.leaves-{leaves}.rate-{token}":
                return {
                    **common,
                    "family_id": "lightgbm",
                    "role": AlphaCandidateRole.REGULARIZED_ALPHA,
                    "feature_shape": "FROZEN_ORDERED",
                    "factor_count": factor_count,
                    "num_leaves": leaves,
                    "learning_rate": rate,
                    "maximum_iterations": 500,
                    "early_stopping_rounds": 50,
                    "fit_intercept": None,
                    "deterministic_seed": 1729,
                    "selection_eligible": True,
                    "package_identity": "lightgbm==4.7.0",
                }
    raise AlphaCatalogAdmissionError(f"UNREGISTERED_ALPHA_CANDIDATE:{candidate_id}")


def load_registered_alpha_catalog(
    inventory: ResearchDeskCandidateInventory,
    *,
    ordered_factor_ids: tuple[str, ...],
) -> tuple[AlphaExperimentCard, ...]:
    """Derive all role-aware cards from current authority and reject every deviation."""

    authority = load_full_alpha_candidate_inventory()
    legacy = load_initial_candidate_inventory()
    if inventory != authority and inventory != legacy:
        raise AlphaCatalogAdmissionError("ALPHA_INVENTORY_NOT_CURRENT_AUTHORITY")
    source_ids = tuple(value.candidate_id for value in inventory.alpha_models)
    allowed_ids = {
        REGISTERED_ALPHA_CANDIDATE_IDS,
        tuple(value.candidate_id for value in legacy.alpha_models),
    }
    if source_ids not in allowed_ids:
        raise AlphaCatalogAdmissionError("ALPHA_INVENTORY_ORDER_MISMATCH")
    if (
        not ordered_factor_ids
        or len(ordered_factor_ids) > inventory.dynamic_model_factor_ceiling
        or ordered_factor_ids != tuple(dict.fromkeys(ordered_factor_ids))
    ):
        raise AlphaCatalogAdmissionError("ALPHA_FOUNDATION_FACTOR_AUTHORITY_INVALID")
    cards = tuple(
        seal_contract(
            AlphaExperimentCard,
            _card_values(
                candidate_id=source.candidate_id,
                source_card_hash=source.card_hash,
                factor_count=len(ordered_factor_ids),
                ordered_factor_ids=ordered_factor_ids,
            ),
            "card_hash",
        )
        for source in inventory.alpha_models
    )
    if tuple(value.candidate_id for value in cards) != source_ids:
        raise AlphaCatalogAdmissionError("ALPHA_REGISTERED_CARD_ORDER_MISMATCH")
    # A card's package names the implementation it was validated with; the version
    # installed is the environment, recorded beside each fit and never a gate (LAWS.md ID6).
    return cards


__all__ = [
    "REGISTERED_ALPHA_CANDIDATE_IDS",
    "AlphaCatalogAdmissionError",
    "load_registered_alpha_catalog",
]
