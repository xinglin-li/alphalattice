"""Stage 1: five-session authority, the three target arms, clipping evidence, score sealing.

These cases prove the software path on seeded deterministic evidence. They are
mechanics, not science: no case here claims a scientific result, and the
milestone reports `AWAITING_INPUT_EVIDENCE` for real Stage 1 numbers.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear import (
    build_regularized_linear_search_domain,
)
from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaModelRecipeEnvelope,
)
from alphalattice.foundation.causal_outcomes.execution.contracts import (
    DevelopmentOnlyExecutionOutcomeManifest,
)
from alphalattice.foundation.causal_outcomes.execution.methods import (
    FIVE_SESSION_RECIPE_ID,
    ONE_SESSION_RECIPE_ID,
    ExecutionOutcomeMethodError,
    build_development_only_execution_outcome_publication_policy,
    build_installed_execution_outcome_publication_policy,
    build_installed_execution_outcome_publication_policy_catalog,
)
from alphalattice.investment.alpha_research.experiments.policies import (
    AlphaExperimentPolicyError,
    derive_alpha_development_split_policy,
    load_alpha_split_policy,
)
from alphalattice.investment.alpha_research.experiments.target_preprocessing import (
    AlphaComparisonArm,
    AlphaTargetClippingRow,
    AlphaTargetPreprocessingComparisonProgram,
)
from alphalattice.investment.alpha_research.targets.authority import (
    installed_alpha_target_methods,
)
from alphalattice.investment.alpha_research.targets.unbounded import (
    AlphaTargetBoundaryError,
    build_unbounded_sensitivity_target_recipe,
    compile_unbounded_sensitivity_target_surface,
)

CASE_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = CASE_ROOT.parents[1]

_SECTOR_REVISION = "a" * 64


def test_the_one_session_publication_policy_did_not_move() -> None:
    """requirement: admitting a second method must not rotate the first's identity.

    Every seal already on disk carries the one-session policy hash, and the seal
    verifier compares against what this build constructs. Widening the installed
    policy in place -- another admitted id, another field, even a defaulted one --
    would have invalidated all of them at once.
    """

    policy = build_installed_execution_outcome_publication_policy()
    assert policy.admitted_recipe_ids == (ONE_SESSION_RECIPE_ID,)
    assert policy.development_split_semantics == "NON_OVERLAPPING_SINGLE_SESSION_LABEL"
    assert policy.publication_scope == "DEVELOPMENT_AND_SEALED_HOLDOUT"
    assert policy.policy_hash == "ac847376d78540416c725a94defacadb2458ffdb1132caae7430dfb94ba26823"


def test_five_session_admission_is_a_second_policy_scoped_to_development() -> None:
    """requirement: overlapping labels may not be cut at a sealed boundary."""

    development_only = build_development_only_execution_outcome_publication_policy()
    assert development_only.admitted_recipe_ids == (FIVE_SESSION_RECIPE_ID,)
    assert development_only.publication_scope == "DEVELOPMENT_ONLY"

    catalog = build_installed_execution_outcome_publication_policy_catalog()
    assert catalog.resolve(FIVE_SESSION_RECIPE_ID).policy_hash == development_only.policy_hash
    with pytest.raises(ExecutionOutcomeMethodError, match="not_installed_for_recipe"):
        catalog.resolve("NEXT_OPEN_TO_OPEN_THREE_SESSION")


def test_the_development_only_manifest_cannot_carry_a_sealed_chunk() -> None:
    """requirement: a scope that says it holds no Holdout must hold none.

    The frozen contract cannot express a five-session span and must not be
    widened to; this successor can, and refuses the one thing that would make it
    a Holdout writer in disguise.
    """

    with pytest.raises(ValueError, match="carries a sealed chunk"):
        DevelopmentOnlyExecutionOutcomeManifest.model_validate(
            {
                "kind": "DevelopmentOnlyExecutionOutcomeSnapshot",
                "research_cadence": "DAILY",
                "market_as_of": "2026-01-05",
                "listing_ids": ["AAA"],
                "listing_set_hash": "b" * 64,
                "schedule_hash": "c" * 64,
                "ordered_session_triples_hash": "d" * 64,
                "source_rows_semantic_hash": "e" * 64,
                "price_basis": "open_split_adjusted",
                "return_formula_identity": "five-session",
                "corporate_action_identity": "provider-split-adjusted-open-and-period-dividend",
                "data_validity_class": "CURRENT_UNIVERSE_RESEARCH_ONLY",
                "allowed_scope": "DEVELOPMENT_ONLY",
                "development_chunks": [
                    {
                        "split": "SEALED_HOLDOUT",
                        "year": 2026,
                        "row_count": 1,
                        "first_formation_session": "2026-01-05",
                        "last_formation_session": "2026-01-05",
                        "content_hash": "f" * 64,
                        "metadata_hash": "0" * 64,
                        "uri": "playpen://x",
                    }
                ],
                "development_formation_count": 1,
                "maturity_lag_sessions": 6,
                "limitations": [],
                "snapshot_hash": "1" * 64,
            }
        )


@pytest.mark.parametrize(
    ("maturity_lag", "embargo", "step"),
    [(2, 1, 253), (6, 5, 257)],
)
def test_the_embargo_and_step_come_from_the_maturity_clock(
    maturity_lag: int, embargo: int, step: int
) -> None:
    """requirement: the step must cover the embargo, or the exclusion means nothing.

    A rolling step that did not cover the embargo would move the next validation
    window straight over the sessions the embargo had just excluded. The frozen
    geometry steps by its validation length, which is admissible only while the
    embargo is zero -- so a development run steps by ``validation + embargo``.
    """

    geometry = load_alpha_split_policy()
    policy = derive_alpha_development_split_policy(
        geometry=geometry, maturity_lag_sessions=maturity_lag, session_count=2515
    )
    assert policy.embargo_sessions == embargo
    assert policy.step_sessions == step
    assert policy.validation_sessions == geometry.validation_sessions
    assert policy.train_sessions == geometry.train_sessions
    # Predicted here and produced by the splitter; the plan builder requires the
    # two to agree exactly, which is a cross-check rather than a relaxation.
    assert policy.expected_complete_folds == 5
    # The frozen policy is untouched and still says zero.
    assert geometry.embargo_sessions == 0


def test_an_axis_too_short_for_the_embargo_is_refused_before_any_fit() -> None:
    """requirement: fewer folds than the minimum is a refusal, not a smaller study."""

    with pytest.raises(AlphaExperimentPolicyError, match="EMBARGOED_AXIS_TOO_SHORT"):
        derive_alpha_development_split_policy(
            geometry=load_alpha_split_policy(),
            maturity_lag_sessions=6,
            session_count=1400,
        )


def test_canonical_and_unbounded_are_distinct_installed_methods() -> None:
    """requirement: two compositions must never share one identity.

    Same estimator, same parameters, two target compositions with no lane. If the
    methods collided, so would every candidate fitted to them.
    """

    catalog = installed_alpha_target_methods(
        sector_revision=_SECTOR_REVISION, execution_outcome_recipe_id=ONE_SESSION_RECIPE_ID
    )
    canonical = catalog.resolve("SECTOR_RESIDUAL_CROSS_SECTIONAL_STD_Z")
    unbounded = catalog.resolve("SECTOR_RESIDUAL_UNBOUNDED_CROSS_SECTIONAL_STD_Z")
    assert canonical.target_method_hash != unbounded.target_method_hash
    assert canonical.admitted_model_lanes is None
    assert unbounded.admitted_model_lanes is None
    # The two frozen lanes still resolve under their declared ids, which the lane
    # alone could not distinguish -- both are rank-gauss lanes.
    rank = catalog.resolve("SECTOR_RESIDUAL_RANK_GAUSS")
    robust = catalog.resolve("SECTOR_RESIDUAL_ROBUST_Z")
    assert rank.target_method_hash != robust.target_method_hash
    assert rank.admitted_model_lanes == robust.admitted_model_lanes

    # Changing the clock changes the successor methods and leaves the frozen
    # lanes alone: a lane predates the seam and answers for no outcome method.
    five = installed_alpha_target_methods(
        sector_revision=_SECTOR_REVISION, execution_outcome_recipe_id=FIVE_SESSION_RECIPE_ID
    )
    assert five.resolve("SECTOR_RESIDUAL_CROSS_SECTIONAL_STD_Z").target_method_hash != (
        canonical.target_method_hash
    )
    assert five.resolve("SECTOR_RESIDUAL_RANK_GAUSS").target_method_hash == rank.target_method_hash


def _source_table(*, sessions: int = 3, listings: int = 12) -> tuple[pa.Table, dict[str, str]]:
    """A small deterministic surface with one deliberate outlier per session."""

    base = date(2026, 1, 5)
    rows: list[dict[str, object]] = []
    sector_by_listing: dict[str, str] = {}
    for listing_index in range(listings):
        listing = f"L{listing_index:02d}"
        sector_by_listing[listing] = "S0" if listing_index < listings // 2 else "S1"
    for session_index in range(sessions):
        session = base + timedelta(days=session_index)
        for listing_index in range(listings):
            listing = f"L{listing_index:02d}"
            value = 0.001 * (listing_index - listings / 2) + 0.0005 * session_index
            if listing_index == 0:
                value = 0.75  # far enough out to be bound at five MAD
            rows.append(
                {
                    "formation_session": session,
                    "listing_id": listing,
                    "fit_target": float(value),
                    "simple_economic_return": float(np.expm1(value)),
                }
            )
    return pa.Table.from_pylist(rows), sector_by_listing


def test_the_unbounded_control_verifies_its_re_demeaning_is_idle() -> None:
    """requirement: 'this step was unnecessary' must be checked, not assumed.

    With nothing clipped the residual is already Sector-neutral, so demeaning it
    again must move nothing. Running the step and requiring it to be idle turns
    an assumption into a property.
    """

    table, sectors = _source_table()
    recipe = build_unbounded_sensitivity_target_recipe(
        execution_outcome_recipe_id=ONE_SESSION_RECIPE_ID,
        sector_revision=_SECTOR_REVISION,
        minimum_coverage=0.5,
        minimum_sector_sample=2,
    )
    surface = compile_unbounded_sensitivity_target_surface(
        source_table=table, recipe=recipe, sector_by_listing_id=sectors
    )
    assert surface.lane_identity.maximum_redemean_correction <= (
        recipe.redemean_invariance_tolerance
    )
    assert recipe.research_role == "SENSITIVITY_CONTROL_NOT_CANONICAL"
    # No bounding step exists, so the outlier survives into the residual whole.
    raw = np.asarray(
        surface.targets["raw_residual"].to_numpy(zero_copy_only=False), dtype=np.float64
    )
    assert float(np.nanmax(np.abs(raw))) > 0.5


def test_an_unbounded_row_may_not_appear_in_the_bounded_detail_child() -> None:
    """requirement: the detail child counts bounded observations and nothing else."""

    with pytest.raises(ValueError, match="row_not_bounded"):
        AlphaTargetClippingRow(
            formation_session=date(2026, 1, 5),
            listing_id="L00",
            raw_residual=0.1,
            lower_bound=-1.0,
            upper_bound=1.0,
            bounded_residual=0.1,
        )


def _installed_model_recipe() -> tuple[AlphaModelRecipeEnvelope, str]:
    """One complete admitted-shape envelope and its domain, for pure Program cases.

    Built through the adapter's own builders rather than as loose hex, because
    the Program now refuses an envelope whose digest does not re-derive from its
    carried route and parameters -- exactly the property Gate A added.
    """

    domain = build_regularized_linear_search_domain()
    return (
        AlphaModelRecipeEnvelope.create(
            adapter_id=domain.adapter_id,
            recipe_schema_id=domain.recipe_schema_id,
            parameters={"family": "ridge", "alpha": 1.0},
        ),
        str(domain.search_domain_hash),
    )


def _arm(arm_id: str, *, role: str, outcome: str, split: str) -> AlphaComparisonArm:
    return AlphaComparisonArm(
        arm_id=arm_id,
        target_recipe_id="SECTOR_RESIDUAL_CROSS_SECTIONAL_STD_Z",
        target_method_hash="a" * 64,
        execution_outcome_recipe_id=outcome,
        causal_outcome_snapshot_hash="b" * 64,
        outcome_method_binding_hash="c" * 64,
        maturity_lag_sessions=2,
        split_policy_hash=split,
        evaluation_policy_id="ALPHA_DEFAULT",
        role=role,  # type: ignore[arg-type]
    )


def test_the_comparison_program_refuses_a_pair_on_two_clocks() -> None:
    """requirement: a pair on different clocks cannot isolate the bound.

    The whole point of the pairing is that everything except the bounding step is
    held fixed. Two arms on different holding spans differ in the horizon as
    well, and the difference would be attributed to the bound.
    """

    arms = (
        _arm(
            "ref", role="CANONICAL_BOUNDED_REFERENCE", outcome=ONE_SESSION_RECIPE_ID, split="d" * 64
        ),
        _arm(
            "ctl",
            role="UNBOUNDED_SENSITIVITY_CONTROL",
            outcome=FIVE_SESSION_RECIPE_ID,
            split="d" * 64,
        ),
        _arm(
            "cand",
            role="LONGER_HORIZON_CANDIDATE",
            outcome=FIVE_SESSION_RECIPE_ID,
            split="e" * 64,
        ),
    )
    with pytest.raises(ValueError, match="pair_clock_mismatch"):
        AlphaTargetPreprocessingComparisonProgram.create(
            ordered_arms=arms,
            common_authority_hash="f" * 64,
            feature_panel_snapshot_hash="2" * 64,
            factor_development_receipt_hash="3" * 64,
            ordered_feature_ids=("factor.a",),
            ordered_listing_ids_hash="0" * 64,
            model_capability_handle="capability-1",
            model_recipe=_installed_model_recipe()[0],
            model_search_domain_hash=_installed_model_recipe()[1],
            paired_arm_ids=("ref", "ctl"),
        )


def test_the_comparison_program_fixes_arm_order_before_anything_runs() -> None:
    """requirement: the reference must exist before the control that controls for it."""

    arms = (
        _arm(
            "ctl",
            role="UNBOUNDED_SENSITIVITY_CONTROL",
            outcome=ONE_SESSION_RECIPE_ID,
            split="d" * 64,
        ),
        _arm(
            "ref", role="CANONICAL_BOUNDED_REFERENCE", outcome=ONE_SESSION_RECIPE_ID, split="d" * 64
        ),
        _arm(
            "cand", role="LONGER_HORIZON_CANDIDATE", outcome=FIVE_SESSION_RECIPE_ID, split="e" * 64
        ),
    )
    with pytest.raises(ValueError, match="arm_order_invalid"):
        AlphaTargetPreprocessingComparisonProgram.create(
            ordered_arms=arms,
            common_authority_hash="f" * 64,
            feature_panel_snapshot_hash="2" * 64,
            factor_development_receipt_hash="3" * 64,
            ordered_feature_ids=("factor.a",),
            ordered_listing_ids_hash="0" * 64,
            model_capability_handle="capability-1",
            model_recipe=_installed_model_recipe()[0],
            model_search_domain_hash=_installed_model_recipe()[1],
            paired_arm_ids=("ref", "ctl"),
        )


def test_the_unbounded_compiler_refuses_a_source_it_cannot_describe() -> None:
    """requirement: a missing retained column is a refusal, not a silent partial."""

    table, sectors = _source_table()
    recipe = build_unbounded_sensitivity_target_recipe(
        execution_outcome_recipe_id=ONE_SESSION_RECIPE_ID,
        sector_revision=_SECTOR_REVISION,
    )
    with pytest.raises(AlphaTargetBoundaryError, match="source_columns_missing"):
        compile_unbounded_sensitivity_target_surface(
            source_table=table.drop_columns(["simple_economic_return"]),
            recipe=recipe,
            sector_by_listing_id=sectors,
        )
