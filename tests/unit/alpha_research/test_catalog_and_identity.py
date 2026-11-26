"""Catalog and pure numerical identity acceptance tests."""

from __future__ import annotations

import pytest

from alphalattice.investment.alpha_research.experiments.bindings import (
    AlphaExperimentAdmissionError,
    build_alpha_experiment_request,
    build_candidate_execution_binding,
    build_development_surface_binding,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.alpha_research.candidate_inventories import (
    load_full_alpha_candidate_inventory,
    load_initial_candidate_inventory,
)
from tests.alpha_research.fixtures import FIXTURE_FACTOR_IDS, FIXTURE_LISTING_IDS, synthetic_request
from tests.alpha_research.registered_catalog import (
    load_registered_alpha_catalog,
)


def test_catalog_rejects_reorder_and_forged_source_card() -> None:
    _request, inventory = synthetic_request()
    reordered = inventory.model_copy(
        update={"alpha_models": tuple(reversed(inventory.alpha_models))}
    )
    with pytest.raises(ValueError, match="CURRENT_AUTHORITY"):
        load_registered_alpha_catalog(
            reordered,
            ordered_factor_ids=FIXTURE_FACTOR_IDS,
        )
    forged_card = inventory.alpha_models[0].model_copy(update={"card_hash": "f" * 64})
    forged = inventory.model_copy(
        update={"alpha_models": (forged_card, *inventory.alpha_models[1:])}
    )
    with pytest.raises(ValueError, match="CURRENT_AUTHORITY"):
        load_registered_alpha_catalog(
            forged,
            ordered_factor_ids=FIXTURE_FACTOR_IDS,
        )


def test_full_high_inventory_has_exact_role_aware_36_card_surface() -> None:
    inventory = load_full_alpha_candidate_inventory()
    cards = load_registered_alpha_catalog(
        inventory,
        ordered_factor_ids=FIXTURE_FACTOR_IDS,
    )
    assert len(cards) == 36
    assert tuple(value.candidate_id for value in cards) == tuple(
        sorted(value.candidate_id for value in cards)
    )
    assert sum(value.role.value == "BENCHMARK" for value in cards) == 2
    assert sum(value.role.value == "DIAGNOSTIC_ONLY_BOUNDED" for value in cards) == 3
    assert sum(value.role.value == "NOT_ADMITTED" for value in cards) == 1
    assert sum(value.selection_eligible is True for value in cards) == 30
    ols = next(value for value in cards if value.candidate_id == "model.ols-3")
    assert ols.factor_count == 3
    assert ols.admission_status == "NOT_ADMITTED_MISSING_FACTORS"


def test_adding_sibling_candidate_does_not_change_existing_numerical_child_binding() -> None:
    request, inventory = synthetic_request()
    cards = load_registered_alpha_catalog(
        inventory,
        ordered_factor_ids=FIXTURE_FACTOR_IDS,
    )
    fold_hashes = ("1" * 64, "2" * 64)
    original_surface = build_development_surface_binding(
        request=request,
        split_hash="3" * 64,
        fold_commitment_hashes=fold_hashes,
    )
    provisional_request = request.model_copy(
        update={
            "inventory_hash": "4" * 64,
            "candidate_ids": (*request.candidate_ids, "model.new-sibling"),
            "candidate_card_hashes": (
                *request.candidate_card_hashes,
                "5" * 64,
            ),
            "request_hash": "0" * 64,
        }
    )
    expanded_request = type(request).model_validate(
        provisional_request.model_copy(
            update={
                "request_hash": canonical_hash(
                    provisional_request.model_dump(mode="json", exclude={"request_hash"})
                )
            }
        )
    )
    expanded_surface = build_development_surface_binding(
        request=expanded_request,
        split_hash="3" * 64,
        fold_commitment_hashes=fold_hashes,
    )
    assert expanded_request.request_hash != request.request_hash
    assert expanded_surface == original_surface
    assert build_candidate_execution_binding(
        surface=original_surface,
        card=cards[2],
        fold_commitment_hash=fold_hashes[0],
    ) == build_candidate_execution_binding(
        surface=expanded_surface,
        card=cards[2],
        fold_commitment_hash=fold_hashes[0],
    )


def test_arbitrary_ridge_parameter_is_rejected_before_request_hashing() -> None:
    request, inventory = synthetic_request()
    cards = load_registered_alpha_catalog(
        load_initial_candidate_inventory(),
        ordered_factor_ids=FIXTURE_FACTOR_IDS,
    )
    forged = cards[2].model_copy(update={"alpha": 2.0})
    foundation = type("Foundation", (), {})()
    foundation.foundation_hash = request.foundation_hash
    foundation.feature_panel_snapshot_hash = request.feature_panel_snapshot_hash
    foundation.logical_panel_hash = request.logical_panel_hash
    foundation.logical_semantic_index_hash = request.logical_semantic_index_hash
    foundation.execution_outcome = type("Outcome", (), {})()
    foundation.execution_outcome.snapshot_hash = request.causal_outcome_snapshot_hash
    foundation.ordered_factor_ids = FIXTURE_FACTOR_IDS
    with pytest.raises(AlphaExperimentAdmissionError, match="CARD_CONTRACT_INVALID"):
        build_alpha_experiment_request(
            foundation=foundation,
            inventory=inventory,
            cards=(cards[0], cards[1], forged, cards[3]),
            listing_set_hash=canonical_hash(FIXTURE_LISTING_IDS),
            ordered_listing_ids=FIXTURE_LISTING_IDS,
        )
