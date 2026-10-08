"""Requirement tests for the development-only LightGBM capability."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from importlib import import_module
from importlib.metadata import version

import numpy as np
import pytest

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm import (
    LIGHTGBM_ADAPTER_ID,
)
from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_chronological import (
    ChronologicalLightGBMAdapter,
)
from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear_dynamic_panel import (
    DynamicPanelRegularizedLinearAdapter,
    DynamicPanelRegularizedLinearParameters,
    build_dynamic_panel_regularized_linear_recipe,
)
from alphalattice.capabilities.alpha_modeling.catalog import (
    build_installed_alpha_model_catalog,
)
from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaModelFitResult,
    AlphaModelNumericalBinding,
    AlphaModelRecipeEnvelope,
    BoundAlphaModelFitInput,
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
)
from alphalattice.capabilities.alpha_modeling.runtime import (
    lightgbm_threads as lightgbm_threads_module,
)
from alphalattice.capabilities.alpha_modeling.runtime import (
    numerical_environment as numerical_environment_module,
)
from alphalattice.capabilities.alpha_modeling.runtime.lightgbm_threads import (
    LIGHTGBM_THREAD_CANARIES,
    LightGBMThreadCanaryMismatch,
    canary_digest,
    lightgbm_fit_threads,
    lightgbm_threads,
    model_trees,
    sequential_l2,
)
from alphalattice.capabilities.alpha_modeling.runtime.numerical_environment import (
    AlphaNumericalEnvironmentError,
    alpha_model_numerical_scope,
    resolve_alpha_model_numerical_environment,
)
from alphalattice.foundation.research_foundation.contracts import (
    ResearchDeskExecutionOutcomeRef,
    ResearchFoundationBinding,
)
from alphalattice.investment.alpha_research.experiments.contracts import (
    AlphaFoldCommitment,
)
from alphalattice.investment.alpha_research.experiments.mandate import (
    build_current_alpha_research_model_mandate,
)
from alphalattice.investment.alpha_research.inputs.folds import (
    AlphaFoldArrayPlan,
    AlphaFoldArrays,
    build_alpha_fold_commitment,
)
from alphalattice.investment.alpha_research.inputs.training import (
    build_alpha_training_input_binding,
)
from alphalattice.kernel.shared_kernel.domain.enums import DataValidityClass
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.validation.contracts import (
    RebalanceEvidencePoint,
    ResearchSplitSpec,
    ValidationTimeline,
)
from alphalattice.kernel.validation.enums import RebalanceFrequency, SplitMode
from alphalattice.kernel.validation.splitting import build_research_split

FACTOR_IDS = ("nonlinear-a", "nonlinear-b", "size", "trend")
LISTING_IDS = tuple(f"listing-{index}" for index in range(4))


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result


def _foundation() -> ResearchFoundationBinding:
    execution = ResearchDeskExecutionOutcomeRef.model_construct(
        research_cadence=RebalanceFrequency.DAILY,
        snapshot_hash="d" * 64,
        schedule_hash="e" * 64,
        development_content_hash="f" * 64,
        sealed_holdout_content_hash="0" * 64,
        marker_hash="1" * 64,
        market_as_of="2026-07-31",
        data_validity_class=DataValidityClass.CURRENT_UNIVERSE_RESEARCH_ONLY,
    )
    return ResearchFoundationBinding.model_construct(
        research_cadence=RebalanceFrequency.DAILY,
        feature_panel_snapshot_hash="2" * 64,
        logical_panel_hash="a" * 64,
        logical_semantic_index_hash="c" * 64,
        factor_training_outcome_snapshot_hash="3" * 64,
        factor_screening_result_hash="4" * 64,
        factor_candidate_slate_hash="5" * 64,
        research_desk_factor_input_hash="b" * 64,
        execution_outcome=execution,
        ordered_factor_ids=FACTOR_IDS,
        downstream_factor_research_forbidden=True,
        secondary_feature_preprocessing_forbidden=True,
        foundation_hash="9" * 64,
    )


def _matrix(
    keys: tuple[tuple[date, str], ...], first_session: date
) -> tuple[np.ndarray, np.ndarray]:
    features = np.empty((len(keys), len(FACTOR_IDS)), dtype=np.float64)
    targets = np.empty(len(keys), dtype=np.float64)
    for row_index, (session, listing) in enumerate(keys):
        session_index = (session - first_session).days
        listing_index = int(listing.rsplit("-", 1)[1])
        a = np.sin((session_index + 3 * listing_index) / 17.0)
        b = np.cos((2 * session_index - listing_index) / 29.0)
        size = (listing_index - 1.5) / 2.0
        trend = (session_index % 31 - 15) / 15.0
        features[row_index] = (a, b, size, trend)
        targets[row_index] = a * b + 0.4 * size * size - 0.2 * trend
    return _readonly(features), _readonly(targets)


class _ArrayWorkspace:
    def __init__(self, fold: AlphaFoldArrays) -> None:
        self.fold = fold

    @contextmanager
    def fold_lease(self, fold_index: int):  # type: ignore[no-untyped-def]
        assert fold_index == 0
        yield self.fold

    def load_fold(self, fold_index: int) -> AlphaFoldArrays:
        assert fold_index == 0
        return self.fold

    def prepare_current_refit(self):  # type: ignore[no-untyped-def]
        raise AssertionError("current refit is outside this development fixture")


def _plan_and_workspace() -> tuple[AlphaFoldArrayPlan, _ArrayWorkspace]:
    first_session = date(2023, 1, 1)
    sessions = tuple(first_session + timedelta(days=index) for index in range(763))
    frozen_at = datetime(2026, 1, 1, tzinfo=UTC)
    timeline = ValidationTimeline(
        points=tuple(
            RebalanceEvidencePoint(
                session_date=session,
                evidence_available_at=frozen_at,
                benchmark_available_at=frozen_at,
            )
            for session in sessions
        )
    )
    split_plan = build_research_split(
        timeline,
        ResearchSplitSpec(
            mode=SplitMode.ROLLING,
            as_of_timestamp=frozen_at,
            train_sessions=756,
            validation_sessions=2,
            step_sessions=2,
            purge_sessions=1,
            embargo_sessions=0,
            holdout_sessions=2,
            minimum_folds=1,
        ),
    )
    assert len(split_plan.windows) == 1
    window = split_plan.windows[0]
    train_keys = tuple(
        (session, listing) for session in window.train_sessions for listing in LISTING_IDS
    )
    validation_keys = tuple(
        (session, listing) for session in window.validation_sessions for listing in LISTING_IDS
    )
    training_features, training_targets = _matrix(train_keys, first_session)
    validation_features, validation_targets = _matrix(validation_keys, first_session)
    train_mask = _readonly(np.ones(len(train_keys), dtype=np.bool_))
    validation_mask = _readonly(np.ones(len(validation_keys), dtype=np.bool_))
    training_row_sessions = tuple(value[0] for value in train_keys)
    training_listing_ids = tuple(value[1] for value in train_keys)
    validation_row_sessions = tuple(value[0] for value in validation_keys)
    validation_listing_ids = tuple(value[1] for value in validation_keys)
    training_feature_hashes = tuple(canonical_hash(("feature", *value)) for value in train_keys)
    training_outcome_hashes = tuple(canonical_hash(("outcome", *value)) for value in train_keys)
    validation_feature_hashes = tuple(
        canonical_hash(("feature", *value)) for value in validation_keys
    )
    validation_outcome_hashes = tuple(
        canonical_hash(("outcome", *value)) for value in validation_keys
    )
    foundation = _foundation()
    commitment: AlphaFoldCommitment = build_alpha_fold_commitment(window)
    binding = build_alpha_training_input_binding(
        scope="DEVELOPMENT_FOLD",
        foundation_hash=foundation.foundation_hash,
        feature_panel_snapshot_hash=foundation.feature_panel_snapshot_hash,
        causal_outcome_snapshot_hash=foundation.execution_outcome.snapshot_hash,
        target_policy=None,
        ordered_feature_ids=FACTOR_IDS,
        feature_context_hash=None,
        training_row_sessions=training_row_sessions,
        training_row_listing_ids=training_listing_ids,
        prediction_row_sessions=validation_row_sessions,
        prediction_row_listing_ids=validation_listing_ids,
        training_features=training_features,
        training_targets=training_targets,
        training_mask=train_mask,
        prediction_features=validation_features,
        prediction_targets=validation_targets,
        prediction_mask=validation_mask,
        training_feature_row_hashes=training_feature_hashes,
        training_outcome_row_hashes=training_outcome_hashes,
        prediction_feature_row_hashes=validation_feature_hashes,
        prediction_outcome_row_hashes=validation_outcome_hashes,
        training_economic_returns=training_targets,
        prediction_economic_returns=validation_targets,
        training_cutoff=window.train_sessions[-1],
        outcome_maturity_session=window.train_sessions[-1],
        prediction_anchor=window.validation_sessions[0],
        fold_commitment_hash=commitment.commitment_hash,
    )
    fold = AlphaFoldArrays(
        commitment=commitment,
        ordered_factor_ids=FACTOR_IDS,
        training_sessions=tuple(window.train_sessions),
        training_listing_ids=training_listing_ids,
        training_features=training_features,
        training_targets=training_targets,
        training_feature_complete=train_mask,
        training_outcome_complete=train_mask,
        validation_sessions=tuple(window.validation_sessions),
        validation_row_sessions=validation_row_sessions,
        validation_listing_ids=validation_listing_ids,
        validation_features=validation_features,
        validation_targets=validation_targets,
        validation_feature_complete=validation_mask,
        validation_outcome_complete=validation_mask,
        validation_feature_row_hashes=validation_feature_hashes,
        validation_outcome_row_hashes=validation_outcome_hashes,
        training_economic_returns=training_targets,
        validation_economic_returns=validation_targets,
        training_row_sessions=training_row_sessions,
        training_feature_row_hashes=training_feature_hashes,
        training_outcome_row_hashes=training_outcome_hashes,
        training_input_binding=binding,
    )
    plan = AlphaFoldArrayPlan(
        foundation=foundation,
        ordered_listing_ids=LISTING_IDS,
        feature_reader=object(),  # type: ignore[arg-type]
        outcome_reader=object(),  # type: ignore[arg-type]
        feature_panel_manifest_ref="playpen://fixture/lightgbm-panel",
        causal_outcome_manifest_ref="playpen://fixture/lightgbm-outcome",
        split_plan=split_plan,
    )
    return plan, _ArrayWorkspace(fold)


def test_default_catalog_and_current_mandate_install_the_product_lightgbm_only() -> None:
    """The research inventory opens the main product's adapter, not this development grid.

    The linear family stays first so every existing ``capability-1`` declaration
    resolves as before; the Dynamic Panel LightGBM adapter -- the one the
    installed G2/G6 components train and score with -- is ``capability-2``. The
    retired development-only LightGBM adapter's id (Phase 2B's grid, RT R05) is not
    installed.
    """

    catalog = build_installed_alpha_model_catalog()
    mandate = build_current_alpha_research_model_mandate(catalog=catalog)

    assert catalog.adapter_ids == ("regularized_linear", "dynamic_panel_lightgbm")
    assert tuple(value.adapter_id for value in mandate.ordered_search_domains) == (
        "regularized_linear",
        "dynamic_panel_lightgbm",
    )
    assert LIGHTGBM_ADAPTER_ID not in catalog.adapter_ids
    assert mandate.resolve_capability_handle("capability-1").adapter_id == "regularized_linear"
    assert mandate.resolve_capability_handle("capability-2").adapter_id == "dynamic_panel_lightgbm"


def test_dynamic_panel_ridge_batch_reuses_one_eigh_with_predecessor_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """regression: one Gram/eigh serves four alphas within measured SVD tolerances."""

    from sklearn.linear_model import Ridge

    from alphalattice.capabilities.alpha_modeling.adapters import (
        regularized_linear_dynamic_panel as ridge_module,
    )
    from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
        session_grouped_rank_ic,
    )

    generator = np.random.default_rng(1729)
    beta = generator.normal(size=18)
    ordinary = generator.normal(size=(128, 18))
    ordinary_validation = generator.normal(size=(64, 18))
    rank_base = generator.normal(size=(128, 8))
    rank_validation_base = generator.normal(size=(64, 8))
    rank_deficient = np.column_stack((rank_base, rank_base, rank_base[:, :2]))
    rank_deficient_validation = np.column_stack(
        (rank_validation_base, rank_validation_base, rank_validation_base[:, :2])
    )
    large_centered = generator.normal(size=(160, 12))
    large_validation_centered = generator.normal(size=(64, 12))
    large_centered[:, 1] = large_centered[:, 0] + 1e-6 * generator.normal(size=160)
    large_validation_centered[:, 1] = large_validation_centered[:, 0] + 1e-6 * generator.normal(
        size=64
    )
    offsets = np.linspace(1e8, 1e9, 12)
    real_fold = _plan_and_workspace()[1].load_fold(0)
    cases = (
        (
            ordinary,
            ordinary @ beta + 0.05 * generator.normal(size=128),
            ordinary_validation,
            ordinary_validation @ beta + 0.05 * generator.normal(size=64),
            np.repeat(np.arange(4, dtype=np.int64), 16),
        ),
        (
            rank_deficient,
            rank_deficient @ np.resize(beta, rank_deficient.shape[1])
            + 0.05 * generator.normal(size=128),
            rank_deficient_validation,
            rank_deficient_validation @ np.resize(beta, rank_deficient.shape[1])
            + 0.05 * generator.normal(size=64),
            np.repeat(np.arange(4, dtype=np.int64), 16),
        ),
        (
            large_centered + offsets,
            large_centered @ np.resize(beta, 12) + 0.05 * generator.normal(size=160),
            large_validation_centered + offsets,
            large_validation_centered @ np.resize(beta, 12) + 0.05 * generator.normal(size=64),
            np.repeat(np.arange(4, dtype=np.int64), 16),
        ),
        (
            real_fold.training_features,
            real_fold.training_targets,
            real_fold.validation_features,
            real_fold.validation_targets,
            np.asarray(
                [
                    tuple(sorted(set(real_fold.validation_row_sessions))).index(value)
                    for value in real_fold.validation_row_sessions
                ],
                dtype=np.int64,
            ),
        ),
    )
    recipes = tuple(
        build_dynamic_panel_regularized_linear_recipe(
            DynamicPanelRegularizedLinearParameters(family="ridge", alpha=alpha)
        )
        for alpha in (0.1, 1.0, 10.0, 100.0)
    )
    adapter = DynamicPanelRegularizedLinearAdapter()
    assert (
        adapter.describe_numerical_binding().deterministic_policy["ridge_cross_product_block_rows"]
        == 32_768
    )
    original = ridge_module._ridge_gram_spectral
    calls = 0

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ridge_module, "_ridge_gram_spectral", counted)
    maxima = {"coefficient": 0.0, "intercept": 0.0, "prediction": 0.0, "mse": 0.0, "rank_ic": 0.0}
    for case_index, (train_x, train_y, validation_x, validation_y, session_codes) in enumerate(
        cases
    ):
        features = _readonly(np.asarray(train_x, dtype=np.float64))
        targets = _readonly(np.asarray(train_y, dtype=np.float64))
        validation = _readonly(np.asarray(validation_x, dtype=np.float64))
        validation_targets = np.asarray(validation_y, dtype=np.float64)
        feature_ids = tuple(f"feature-{value}" for value in range(features.shape[1]))
        inputs = BoundAlphaTrainingInput(
            training_binding_hash=f"{case_index + 1:x}" * 64,
            ordered_feature_ids=feature_ids,
            features=features,
            targets=targets,
        )
        fit_plan = BoundAlphaModelFitInput(
            fit_plan_hash=f"{case_index + 5:x}" * 64,
            protocol_id="DIRECT_FIT",
            parent_training_binding_hash=inputs.training_binding_hash,
            ordered_feature_ids=feature_ids,
        )
        prediction_input = BoundAlphaPredictionInput(
            training_binding_hash=inputs.training_binding_hash,
            ordered_feature_ids=feature_ids,
            features=validation,
        )
        predecessor = tuple(
            Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(features, targets)
            for alpha in (0.1, 1.0, 10.0, 100.0)
        )
        before = calls
        successor = adapter.fit_ridge_batch(recipes=recipes, inputs=inputs, fit_plan=fit_plan)
        assert calls - before == 1
        predecessor_predictions = tuple(
            np.asarray(value.predict(validation), dtype=np.float64) for value in predecessor
        )
        successor_predictions = tuple(
            adapter.predict(estimator=value.estimator_content, inputs=prediction_input).predictions
            for value in successor
        )
        predecessor_coefficients = tuple(
            np.asarray(value.coef_, dtype=np.float64).reshape(-1) for value in predecessor
        )
        successor_coefficients = tuple(
            np.asarray(
                [
                    float.fromhex(item)
                    for item in value.estimator_content.payload["coefficient_hex"]
                ],
                dtype=np.float64,
            )
            for value in successor
        )
        for old, new in zip(predecessor_coefficients, successor_coefficients, strict=True):
            maxima["coefficient"] = max(maxima["coefficient"], float(np.max(np.abs(new - old))))
            np.testing.assert_allclose(new, old, rtol=2e-8, atol=2e-10)
        predecessor_intercepts = np.asarray(
            [float(value.intercept_) for value in predecessor], dtype=np.float64
        )
        successor_intercepts = np.asarray(
            [
                float.fromhex(str(value.estimator_content.payload["intercept_hex"]))
                for value in successor
            ],
            dtype=np.float64,
        )
        maxima["intercept"] = max(
            maxima["intercept"],
            float(np.max(np.abs(successor_intercepts - predecessor_intercepts))),
        )
        np.testing.assert_allclose(
            successor_intercepts,
            predecessor_intercepts,
            rtol=2e-8,
            atol=2e-8,
        )
        for old, new in zip(predecessor_predictions, successor_predictions, strict=True):
            maxima["prediction"] = max(maxima["prediction"], float(np.max(np.abs(new - old))))
            np.testing.assert_allclose(new, old, rtol=2e-8, atol=2e-6)
        predecessor_mse = np.asarray(
            [
                float(np.mean(np.square(validation_targets - value)))
                for value in predecessor_predictions
            ]
        )
        successor_mse = np.asarray(
            [
                float(np.mean(np.square(validation_targets - value)))
                for value in successor_predictions
            ]
        )
        maxima["mse"] = max(maxima["mse"], float(np.max(np.abs(successor_mse - predecessor_mse))))
        np.testing.assert_allclose(successor_mse, predecessor_mse, rtol=4e-6, atol=1e-7)
        minimum_rows = int(np.min(np.bincount(session_codes)))
        predecessor_ic = np.asarray(
            [
                session_grouped_rank_ic(
                    value,
                    validation_targets,
                    session_codes,
                    minimum_paired_rows=minimum_rows,
                )[0]
                for value in predecessor_predictions
            ]
        )
        successor_ic = np.asarray(
            [
                session_grouped_rank_ic(
                    value,
                    validation_targets,
                    session_codes,
                    minimum_paired_rows=minimum_rows,
                )[0]
                for value in successor_predictions
            ]
        )
        maxima["rank_ic"] = max(
            maxima["rank_ic"], float(np.max(np.abs(successor_ic - predecessor_ic)))
        )
        np.testing.assert_allclose(successor_ic, predecessor_ic, rtol=0.0, atol=2e-12)
        assert (
            np.argsort(-successor_ic, kind="stable").tolist()
            == np.argsort(-predecessor_ic, kind="stable").tolist()
        )
        assert int(np.argmax(successor_ic)) == int(np.argmax(predecessor_ic))
    print(f"RIDGE_NUMERICAL_SUCCESSOR_MAX_ABSOLUTE_DEVIATIONS={maxima}")


def test_a_declared_version_is_recorded_never_enforced_and_a_missing_package_is_operational(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """requirement (LAWS.md ID6, E0): the fit records the installed version beside itself and
    no declared version refuses it; a package the binding needs and the host lacks is an
    operational failure, never scientific evidence."""

    from alphalattice.capabilities.alpha_modeling.runtime import numerical_environment

    binding = AlphaModelNumericalBinding.create(
        adapter_id="lightgbm",
        estimator_content_format_id="test",
        implementation_owners=("test",),
        deterministic_policy={"thread_count": 1},
        required_runtime_capabilities=("lightgbm==0.0.0",),
    )

    recorded = resolve_alpha_model_numerical_environment(binding)
    assert dict(recorded.package_versions)["lightgbm"] not in {"0.0.0", "absent"}

    monkeypatch.setattr(
        numerical_environment,
        "package_versions",
        lambda names: tuple((name, "absent") for name in names),
    )
    with pytest.raises(AlphaNumericalEnvironmentError) as raised:
        resolve_alpha_model_numerical_environment(binding)
    assert raised.value.failure_class == "OPERATIONAL_FAILURE"
    assert raised.value.code == "ALPHA_MODEL_RUNTIME_PACKAGE_MISSING"


def test_numerical_environment_effective_thread_failure_is_operational(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = AlphaModelNumericalBinding.create(
        adapter_id="lightgbm",
        estimator_content_format_id="test",
        implementation_owners=("test",),
        deterministic_policy={"thread_count": 1},
        required_runtime_capabilities=("single-thread",),
    )
    monkeypatch.setattr(
        numerical_environment_module,
        "numerical_thread_pools",
        lambda: [{"num_threads": 2}],
    )

    with (
        pytest.raises(AlphaNumericalEnvironmentError) as raised,
        alpha_model_numerical_scope(binding),
    ):
        pass
    assert raised.value.failure_class == "OPERATIONAL_FAILURE"


def test_dynamic_panel_lightgbm_installs_aligned_regularization_and_training_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """requirement: heavy/light L2, nonzero L1, rank and fixed policies are sealed."""

    from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
        DYNAMIC_PANEL_FIXED_ITERATIONS,
        DYNAMIC_PANEL_LIGHTGBM_MAXIMUM_ITERATIONS,
        DYNAMIC_PANEL_LIGHTGBM_PATIENCE,
        DynamicPanelLightGBMAdapter,
        build_dynamic_panel_lightgbm_recipe,
        build_dynamic_panel_lightgbm_search_domain,
        dynamic_panel_lightgbm_parameters,
    )

    parameters = dynamic_panel_lightgbm_parameters()
    domain = build_dynamic_panel_lightgbm_search_domain()
    adapter = DynamicPanelLightGBMAdapter()
    assert len(parameters) == 720
    assert any(value.lambda_l1 > 0.0 for value in parameters)
    assert {
        (value.lambda_l1, value.lambda_l2, value.min_gain_to_split) for value in parameters
    } == {
        (1.0, 1.0, 0.0),
        (10.0, 10.0, 0.001),
        (0.0, 1000.0, 0.0),
        (50.0, 1000.0, 0.01),
        (100.0, 10000.0, 0.05),
    }
    assert {value.learning_rate for value in parameters} == {0.03, 0.05}
    assert {value.min_child_samples for value in parameters} == {50, 200}
    assert {value.feature_fraction for value in parameters} == {0.7, 1.0}
    assert {value.bagging_fraction for value in parameters} == {0.7, 1.0}
    assert {value.training_policy for value in parameters} == {
        "L2_EARLY_STOPPING",
        "SESSION_RANK_IC_EARLY_STOPPING",
        "FIXED_ITERATION",
    }
    assert {
        value.fixed_iterations for value in parameters if value.training_policy == "FIXED_ITERATION"
    } == set(DYNAMIC_PANEL_FIXED_ITERATIONS)
    assert domain.constraints["maximum_iterations"] == 500
    assert domain.constraints["early_stopping_rounds"] == 50
    assert DYNAMIC_PANEL_LIGHTGBM_MAXIMUM_ITERATIONS == 500
    assert DYNAMIC_PANEL_LIGHTGBM_PATIENCE == 50
    assert all(
        adapter.validate_recipe_for_domain(
            recipe=build_dynamic_panel_lightgbm_recipe(value),
            domain=domain,
        )
        == value
        for value in parameters
    )
    broad_leaf = next(value for value in parameters if value.min_child_samples == 200)
    assert adapter._native_params(broad_leaf)["min_data_in_leaf"] == 200
    assert adapter._native_params(broad_leaf)["num_threads"] == 1
    assert (
        adapter.describe_numerical_binding().deterministic_policy["adapter_preprocessing"] == "NONE"
    )
    assert adapter.describe_numerical_binding().deterministic_policy["thread_count"] == 1
    assert "single-thread" in adapter.describe_numerical_binding().required_runtime_capabilities


@pytest.mark.parametrize(
    ("training_policy", "admitted"),
    [
        ("FIXED_ITERATION", True),
        (["FIXED_ITERATION"], False),
        ({"policy": "FIXED_ITERATION"}, False),
        ("NOT_A_POLICY", False),
    ],
    ids=["string", "list", "mapping", "unknown"],
)
def test_dynamic_panel_training_policy_kind_is_checked_before_its_value(
    training_policy: object, admitted: bool
) -> None:
    """A YAML list or mapping where the policy name belongs is a typed refusal.

    The membership test against the admitted policies would hash the value
    first; an unhashable one used to escape as a TypeError, which the entry
    points cannot name. The parameter owner checks the kind, then the value.
    """

    from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
        DynamicPanelLightGBMAdapter,
        build_dynamic_panel_lightgbm_recipe,
        build_dynamic_panel_lightgbm_search_domain,
    )
    from alphalattice.investment.alpha_research.scores.product_recipe import PRODUCT_ESTIMATOR_POINT

    adapter = DynamicPanelLightGBMAdapter()
    installed = build_dynamic_panel_lightgbm_recipe(PRODUCT_ESTIMATOR_POINT.resolve(seed=1729))
    recipe = AlphaModelRecipeEnvelope.create(
        adapter_id=installed.adapter_id,
        recipe_schema_id=installed.recipe_schema_id,
        parameters={**installed.parameters, "training_policy": training_policy},
    )
    if admitted:
        assert adapter.validate_recipe(recipe).training_policy == training_policy
        assert (
            adapter.validate_recipe_for_domain(
                recipe=recipe, domain=build_dynamic_panel_lightgbm_search_domain()
            ).training_policy
            == training_policy
        )
        return
    with pytest.raises(ValueError, match="ALPHA_DYNAMIC_PANEL_LIGHTGBM_RECIPE_PARAMETERS_INVALID"):
        adapter.validate_recipe(recipe)
    with pytest.raises(ValueError, match="ALPHA_DYNAMIC_PANEL_LIGHTGBM_RECIPE_PARAMETERS_INVALID"):
        adapter.validate_recipe_for_domain(
            recipe=recipe, domain=build_dynamic_panel_lightgbm_search_domain()
        )


def test_dynamic_panel_rank_ic_is_equal_session_weighted_and_fail_closed() -> None:
    """requirement: formation Spearman excludes declared bad sessions only."""

    from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
        session_grouped_rank_ic,
    )

    predictions = np.concatenate(
        (
            np.arange(19),
            np.ones(20),
            np.arange(20),
            np.arange(40),
        )
    ).astype(np.float64)
    targets = np.concatenate(
        (
            np.arange(19),
            np.arange(20),
            np.arange(20),
            np.arange(39, -1, -1),
        )
    ).astype(np.float64)
    codes = np.repeat(np.arange(4), (19, 20, 20, 40)).astype(np.int64)

    value, insufficient, constant = session_grouped_rank_ic(predictions, targets, codes)

    assert value == pytest.approx(0.0, abs=1e-15)
    assert (insufficient, constant) == (1, 1)
    permutation = np.random.default_rng(1729).permutation(len(codes))
    assert session_grouped_rank_ic(
        predictions[permutation], targets[permutation], codes[permutation]
    ) == pytest.approx((value, insufficient, constant), abs=1e-15)
    predictions[0] = np.nan
    with pytest.raises(ValueError, match="FINITE_AXIS_INVALID"):
        session_grouped_rank_ic(predictions, targets, codes)


def test_lightgbm_curve_and_stored_trees_do_not_move_with_the_thread_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binding plan B3: LightGBM's built-in l2 sums in parallel, so the last bits of the curve
    early stopping reads moved with the thread count; the l2 the adapters install reproduces
    the one-thread curve at any count, and the stored model text records no count."""

    lightgbm = import_module("lightgbm")
    train_x = lightgbm_threads_module._canary_values(4_000, 8, 0)
    valid_x = lightgbm_threads_module._canary_values(1_500, 8, 4_000 * 8)

    def fit(threads: int, *, builtin: bool) -> tuple[object, np.ndarray]:
        params = {
            "objective": "l2",
            "metric": "l2" if builtin else "None",
            "num_leaves": 15,
            "learning_rate": 0.1,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": threads,
            "verbosity": -1,
            "seed": 7,
        }
        train_y = lightgbm_threads_module._canary_target(train_x)
        valid_y = lightgbm_threads_module._canary_target(valid_x)
        train = lightgbm.Dataset(train_x, label=train_y, params=params)
        valid = lightgbm.Dataset(valid_x, label=valid_y, reference=train)
        curve: dict[str, dict[str, list[float]]] = {}
        booster = lightgbm.train(
            params,
            train,
            num_boost_round=60,
            valid_sets=[valid],
            feval=None if builtin else sequential_l2,
            callbacks=[lightgbm.record_evaluation(curve)],
        )
        return booster, np.asarray(curve["valid_0"]["l2"])

    one, one_thread_curve = fit(1, builtin=True)
    four, sequential_curve = fit(4, builtin=False)
    trees = model_trees(one.model_to_string())

    assert np.array_equal(sequential_curve, one_thread_curve)
    assert model_trees(four.model_to_string()) == trees and "num_threads" not in trees
    assert np.array_equal(lightgbm.Booster(model_str=trees).predict(valid_x), one.predict(valid_x))
    metric, evaluator = ChronologicalLightGBMAdapter()._evaluation(
        fit_plan=None,  # type: ignore[arg-type]
        parameters=None,  # type: ignore[arg-type]
    )
    assert (metric, evaluator) == ("None", sequential_l2)

    from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
        DYNAMIC_PANEL_FIXED_ITERATIONS,
        DynamicPanelLightGBMAdapter,
        build_dynamic_panel_lightgbm_recipe,
        dynamic_panel_lightgbm_parameters,
    )

    adapter = DynamicPanelLightGBMAdapter()
    parameters = next(
        value
        for value in dynamic_panel_lightgbm_parameters()
        if value.training_policy == "FIXED_ITERATION"
        and value.fixed_iterations == min(DYNAMIC_PANEL_FIXED_ITERATIONS)
        and value.min_child_samples == 50
        and value.lambda_l1 == value.lambda_l2 == 1.0
        and value.feature_fraction == value.bagging_fraction == 1.0
    )
    recipe = build_dynamic_panel_lightgbm_recipe(parameters)
    feature_ids = tuple(f"feature-{index}" for index in range(train_x.shape[1]))
    inputs = BoundAlphaTrainingInput(
        training_binding_hash="a" * 64,
        ordered_feature_ids=feature_ids,
        features=_readonly(train_x),
        targets=_readonly(train_x[:, 0] * train_x[:, 1] - train_x[:, 2]),
    )
    fit_plan = BoundAlphaModelFitInput(
        fit_plan_hash="b" * 64,
        protocol_id="DIRECT_FIT",
        parent_training_binding_hash=inputs.training_binding_hash,
        ordered_feature_ids=feature_ids,
    )
    prediction_calls: list[tuple[tuple[int, ...], object, bytes]] = []
    original_predict = lightgbm.Booster.predict

    def observe_prediction(booster: object, data: np.ndarray, **kwargs: object) -> object:
        result = original_predict(booster, data, **kwargs)
        prediction_calls.append(
            (data.shape, kwargs.get("num_threads"), np.asarray(result, dtype="<f8").tobytes())
        )
        return result

    monkeypatch.setattr(lightgbm.Booster, "predict", observe_prediction)

    def fixed_fit(threads: int) -> tuple[AlphaModelFitResult, bytes]:
        with lightgbm_threads(threads) as admitted:
            assert admitted.threads == threads
            prediction_calls.clear()
            with alpha_model_numerical_scope(adapter.describe_numerical_binding()):
                result = adapter.fit(recipe=recipe, inputs=inputs, fit_plan=fit_plan)
            assert [(shape, width) for shape, width, _ in prediction_calls] == [
                (inputs.features.shape, threads)
            ]
            return result, prediction_calls[0][2]

    one_fit, one_predictions = fixed_fit(1)
    two_fit, two_predictions = fixed_fit(2)
    assert one_predictions == two_predictions
    assert (
        np.asarray([one_fit.training_mse], dtype="<f8").tobytes()
        == np.asarray([two_fit.training_mse], dtype="<f8").tobytes()
    )
    assert one_fit == two_fit
    assert one_fit.estimator_content.content_hash == two_fit.estimator_content.content_hash
    assert one_fit.state_projection.projection_hash == two_fit.state_projection.projection_hash
    assert one_fit.selection_diagnostic is not None and two_fit.selection_diagnostic is not None
    assert (
        one_fit.selection_diagnostic.diagnostic_hash == two_fit.selection_diagnostic.diagnostic_hash
    )
    assert one_fit.fit_call_count == one_fit.predict_call_count == 1


def test_lightgbm_fits_on_one_thread_unless_the_sealed_canary_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proven: list[int] = []
    monkeypatch.setattr(lightgbm_threads_module, "_PROVEN", {1})
    monkeypatch.setattr(
        lightgbm_threads_module, "canary_digest", lambda threads: proven.append(threads) or "0" * 64
    )

    refusal = pytest.raises(LightGBMThreadCanaryMismatch, match="lightgbm_thread_canary_mismatch")
    with refusal, lightgbm_threads(3):
        pass
    assert lightgbm_fit_threads() == 1 and proven == [3]

    # A version with no sealed canary grows it once on one thread; it matches none here.
    with monkeypatch.context() as unsealed:
        unsealed.setattr(lightgbm_threads_module, "LIGHTGBM_THREAD_CANARIES", {})
        unsealed.setattr(lightgbm_threads_module, "version", lambda _name: "0.0.0+unsealed")
        with lightgbm_threads(3) as chosen:
            assert chosen.threads == lightgbm_fit_threads() == 1
            assert "no sealed canary" in chosen.reason
    assert proven == [3, 1]

    monkeypatch.setattr(
        lightgbm_threads_module, "LIGHTGBM_THREAD_CANARIES", {version("lightgbm"): "0" * 64}
    )
    for _ in range(2):
        with lightgbm_threads(3) as chosen:
            assert chosen.threads == lightgbm_fit_threads() == 3
    assert lightgbm_fit_threads() == 1
    assert proven == [3, 1, 3], "a session proves a thread count once"


def test_a_lightgbm_that_grows_a_sealed_canary_keeps_its_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """requirement (PA3, W10): a dependency change keeps LightGBM's threads when the new
    version, with no canary of its own, grows a sealed version's canary on one thread; it is
    then proven against that canary at the requested count, and one that grows another fits on
    one thread. Its fake versions and thread count are its own, so no proof it leaves is
    another test's."""

    grown: list[int] = []
    monkeypatch.setattr(lightgbm_threads_module, "LIGHTGBM_THREAD_CANARIES", {"0.0.1": "a" * 64})
    monkeypatch.setattr(lightgbm_threads_module, "version", lambda _name: "0.0.0+inherits")
    monkeypatch.setattr(
        lightgbm_threads_module, "canary_digest", lambda threads: grown.append(threads) or "a" * 64
    )
    for _ in range(2):
        with lightgbm_threads(13) as chosen:
            assert chosen.threads == lightgbm_fit_threads() == 13
    assert grown == [1, 13], "the one-thread canary is grown once, the count proven once"

    monkeypatch.setattr(lightgbm_threads_module, "version", lambda _name: "0.0.0+grows-another")
    monkeypatch.setattr(lightgbm_threads_module, "canary_digest", lambda threads: "b" * 64)
    with lightgbm_threads(13) as chosen:
        assert chosen.threads == 1 and "no sealed canary" in chosen.reason


def test_the_sealed_lightgbm_canary_holds_at_two_threads_on_this_machine() -> None:
    sealed = LIGHTGBM_THREAD_CANARIES.get(version("lightgbm"))

    assert sealed is not None, "the locked LightGBM has no sealed canary: its fits stay on one"
    assert canary_digest(2) == sealed


def test_dataset_reuse_mutates_every_binning_field_and_preserves_operator_parameters() -> None:
    """regression (V526): generated binning-field mutations miss; operator pacing/logging hit.

    This uses the real Dataset selector with a counting constructor, without fitting a model.
    """
    from copy import deepcopy

    from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_chronological import (
        ChronologicalLightGBMParameters,
        chronological_lightgbm_dataset_cache,
    )

    class Backend:
        calls = 0

        class Dataset:
            def __init__(self, *args, **kwargs):
                Backend.calls += 1

            def construct(self):
                return self

    adapter = ChronologicalLightGBMAdapter()
    parameters = adapter._dataset_params(
        ChronologicalLightGBMParameters(
            seed=1729,
            num_leaves=31,
            learning_rate=0.05,
            max_depth=3,
            min_child_samples=20,
        )
    )
    arguments = {
        "features": np.arange(12, dtype=np.float64).reshape(4, 3),
        "targets": np.arange(4, dtype=np.float64),
        "ordered_feature_ids": ("a", "b", "c"),
        "dataset_params": parameters,
        "reference_key": "training",
    }

    def changed(value):
        if isinstance(value, bool):
            return not value
        if isinstance(value, int | float):
            return value + 1
        return "changed" if value is None else str(value) + " changed"

    with chronological_lightgbm_dataset_cache():
        original, key = adapter._dataset(Backend, **arguments)
        assert adapter._dataset(Backend, **deepcopy(arguments)) == (original, key)
        for field, value in parameters.items():
            altered = deepcopy(arguments)
            altered["dataset_params"][field] = changed(value)
            result, other_key = adapter._dataset(Backend, **altered)
            if field in {"num_threads", "verbosity"}:
                assert result is original and other_key == key, field
            else:
                assert result is not original and other_key != key, field
        for field in ("features", "targets"):
            for index in np.ndindex(arguments[field].shape):
                altered = deepcopy(arguments)
                altered[field][index] += 1
                assert adapter._dataset(Backend, **altered)[1] != key, (field, index)
            altered = deepcopy(arguments)
            altered[field] = altered[field].astype(np.float32)
            assert adapter._dataset(Backend, **altered)[1] != key, field
        for index in range(len(arguments["ordered_feature_ids"])):
            altered = deepcopy(arguments)
            axis = list(altered["ordered_feature_ids"])
            axis[index] = "another"
            altered["ordered_feature_ids"] = tuple(axis)
            assert adapter._dataset(Backend, **altered)[1] != key
        assert adapter._dataset(Backend, **(arguments | {"reference_key": "other"}))[1] != key
    assert Backend.calls > 1


def test_a_booster_cache_scope_parses_each_model_once_and_predicts_the_same_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """requirement (PERF-1 lever 3): a scoring loop reuses each model's Booster by its text."""
    import lightgbm

    from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_chronological import (
        CHRONOLOGICAL_LIGHTGBM_CONTENT_FORMAT_ID,
        chronological_lightgbm_booster_cache,
    )
    from alphalattice.capabilities.alpha_modeling.contracts import AlphaEstimatorContent

    rng = np.random.default_rng(7)
    features = rng.standard_normal((200, 3))
    targets = features @ np.array([0.5, -0.25, 0.1]) + rng.standard_normal(200) * 0.1
    model = lightgbm.train(
        {"objective": "l2", "num_threads": 1, "deterministic": True, "verbosity": -1},
        lightgbm.Dataset(features, targets),
        num_boost_round=20,
    )
    feature_ids = ("a", "b", "c")
    adapter = ChronologicalLightGBMAdapter()
    estimator = AlphaEstimatorContent.create(
        adapter_id=adapter.adapter_id,
        content_format_id=CHRONOLOGICAL_LIGHTGBM_CONTENT_FORMAT_ID,
        ordered_feature_ids=feature_ids,
        payload={"model_text": model.model_to_string(), "best_iteration": 20},
    )
    rows = np.ascontiguousarray(features[:50])
    rows.setflags(write=False)
    inputs = BoundAlphaPredictionInput(
        training_binding_hash="a" * 64, ordered_feature_ids=feature_ids, features=rows
    )
    fresh = adapter.predict(estimator=estimator, inputs=inputs).predictions

    parsed = 0
    booster = lightgbm.Booster

    def counted(*args, **kwargs):
        nonlocal parsed
        parsed += 1
        return booster(*args, **kwargs)

    monkeypatch.setattr(lightgbm, "Booster", counted)
    with chronological_lightgbm_booster_cache():
        reused = [adapter.predict(estimator=estimator, inputs=inputs).predictions for _ in range(3)]
    assert parsed == 1
    assert all(value.tobytes() == fresh.tobytes() for value in reused)
    adapter.predict(estimator=estimator, inputs=inputs)
    assert parsed == 2  # nothing survives the scope
