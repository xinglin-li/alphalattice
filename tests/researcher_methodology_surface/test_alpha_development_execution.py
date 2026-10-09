"""An authored Alpha document reaches the real model runtime, through the Host.

Before this, ``AlphaDevelopmentTargetRecipe`` had exactly one caller: a test that
built one by hand. The recipe was right and unreachable -- no Desk kind, no
compiler, no executor, and nothing in the canonical path that could select it.

These cases drive the real workflow over the real workspace. Authority is
resolved by Product Host composition, not assembled here: the Panel and its
logical identity, the published causal outcomes, and one Factor development
checkpoint loaded by exact hash. The fold arrays come from the real readers and
the fit runs through ``execute_alpha_model_batch``.

The recipe under test is deliberately the one no ``AlphaTargetPolicy`` can name.
Its lane's derived ``standardization_id`` is rank-gauss; the recipe selects
robust-z. If routing still went through the derived property the run would
succeed and produce rank-gauss values under a robust-z name, which is the exact
defect this surface exists to make impossible -- so it is asserted directly
against the compiled surface rather than inferred from the run completing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
    DynamicPanelLightGBMAdapter,
    build_dynamic_panel_lightgbm_recipe,
)
from alphalattice.control.product_host.research_authoring.execution import (
    build_installed_desk_executors,
)
from alphalattice.control.product_host.research_authoring.foundation import (
    assert_distinct_roots,
    build_alpha_development_foundation,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.methods import ONE_SESSION_RECIPE_ID
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.investment.alpha_research.experiments.comparison import (
    compare_saved_alpha_candidates,
)
from alphalattice.investment.alpha_research.experiments.contracts import (
    AlphaExperimentBatchResult,
    AlphaExperimentCandidateResult,
    AlphaFitLedgerEntry,
    seal_contract,
)
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.experiments.development_contracts import (
    AlphaCandidateDevelopmentReport,
    AlphaCandidateFoldEvidence,
    AlphaCandidateInferenceEvidence,
    AlphaDevelopmentChildLineage,
    AlphaDevelopmentExecutionReceipt,
    LegacyAlphaCandidateFoldEvidence,
    seal_current_contract,
)
from alphalattice.investment.alpha_research.experiments.development_evidence import (
    AlphaDevelopmentReceiptReader,
    AlphaDevelopmentVerifiedGraph,
    verify_alpha_development_execution_receipt,
)
from alphalattice.investment.alpha_research.experiments.policies import (
    derive_alpha_development_split_policy,
    load_alpha_split_policy,
)
from alphalattice.investment.alpha_research.inputs.folds import (
    AlphaArrayBoundaryError,
    prepare_alpha_fold_plan,
)
from alphalattice.investment.alpha_research.inputs.surfaces import (
    AlphaProgramArrayWorkspace,
    prepare_alpha_program_array_workspace,
)
from alphalattice.investment.alpha_research.inputs.training import bind_alpha_model_inputs
from alphalattice.investment.alpha_research.scores.product_recipe import PRODUCT_ESTIMATOR_POINT
from alphalattice.investment.alpha_research.targets.authority import (
    AlphaTargetMethodBinding,
    installed_alpha_target_methods,
)
from alphalattice.investment.alpha_research.targets.development import (
    CROSS_STANDARDIZED_DEVELOPMENT_TARGET_RECIPE_ID,
    DEFAULT_DEVELOPMENT_TARGET_RECIPE_ID,
    AlphaDevelopmentTargetFoldRecord,
    AlphaDevelopmentTargetMaterializationBinding,
    compile_alpha_development_target_surface,
    installed_alpha_development_target_recipes,
)
from alphalattice.investment.alpha_research.targets.standardization import (
    RANK_GAUSS_STANDARDIZATION_ID,
    ROBUST_Z_STANDARDIZATION_ID,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.contracts import ActorKind
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExperimentEnvelope,
)
from tests.researcher_methodology_surface.alpha_development_support import (
    _ALPHA_STORE_ROOT,
    _FROZEN_AT,
    _document,
    _workflow,
    alpha_authority_for,
)
from tests.researcher_methodology_surface.real_workspace import (
    RealRiskWorkspace,
)


def test_shared_score_rows_bind_positions_availability_and_unique_cells():

    import pyarrow as pa

    from alphalattice.investment.alpha_research.experiments.score_rows import (
        read_development_score_rows,
    )

    scores = pa.table(
        {
            "row_index": [0, 1],
            "score": [1.0, None],
            "availability": ["SCORED", "FEATURE_INCOMPLETE"],
        }
    )
    validation = pa.table(
        {"row_index": [0, 1], "formation_session": [date(2020, 1, 1)] * 2, "listing_id": ["A", "B"]}
    )
    # The store resolves each chunk once, as part of loading the child that
    # names it, and hands the rows back with the child.
    store = SimpleNamespace(
        read_candidate_numerical_fold_result=lambda h: (
            SimpleNamespace(
                fold_commitment_hash="a", score_chunk=SimpleNamespace(content_hash="b")
            ),
            scores,
        ),
        read_fold_surface=lambda h: (
            SimpleNamespace(fold_commitment_hash="a", validation_chunk="c"),
            validation,
        ),
    )
    assert read_development_score_rows(
        store, numerical_result_hash="n", fold_surface_hash="f"
    ).available.tolist() == [True, False]
    scores = scores.set_column(0, "row_index", pa.array([1, 0]))
    with pytest.raises(ValueError, match="row_index_mismatch"):
        read_development_score_rows(store, numerical_result_hash="n", fold_surface_hash="f")
    scores = scores.set_column(0, "row_index", pa.array([0, 1])).set_column(
        2, "availability", pa.array(["SCORED", "SCORED"])
    )
    with pytest.raises(ValueError, match="availability_mismatch"):
        read_development_score_rows(store, numerical_result_hash="n", fold_surface_hash="f")
    validation = validation.set_column(2, "listing_id", pa.array(["A", "A"]))
    with pytest.raises(ValueError, match="duplicate_row"):
        read_development_score_rows(store, numerical_result_hash="n", fold_surface_hash="f")


def _rebuilt_receipt(
    receipt: AlphaDevelopmentExecutionReceipt,
    *,
    child_lineage: tuple[AlphaDevelopmentChildLineage, ...],
    batch_result_hash: str | None = None,
    fold_surface_hashes: tuple[str, ...] | None = None,
) -> AlphaDevelopmentExecutionReceipt:
    """Reseal a receipt, valid on its own terms.

    Spelled out field by field rather than splatted from a dump, because the two
    target bindings are nested models and a dump would hand back plain mappings.
    """

    return AlphaDevelopmentExecutionReceipt.create(
        program_hash=receipt.program_hash,
        desk_program_hash=receipt.desk_program_hash,
        method_binding_hash=receipt.method_binding_hash,
        desk_input_binding_hash=receipt.desk_input_binding_hash,
        target_recipe_binding=receipt.target_recipe_binding,
        target_materialization_binding=receipt.target_materialization_binding,
        ordered_base_feature_ids=receipt.ordered_base_feature_ids,
        research_recipe_hash=receipt.research_recipe_hash,
        model_adapter_id=receipt.model_adapter_id,
        model_adapter_recipe_hash=receipt.model_adapter_recipe_hash,
        development_program_hash=receipt.development_program_hash,
        split_policy=receipt.split_policy,
        model_method_binding_hash=receipt.model_method_binding_hash,
        batch_hash=receipt.batch_hash,
        batch_result_hash=batch_result_hash or receipt.batch_result_hash,
        development_surface_hash=receipt.development_surface_hash,
        development_surface_binding_hash=receipt.development_surface_binding_hash,
        fold_surface_hashes=fold_surface_hashes or receipt.fold_surface_hashes,
        child_lineage=child_lineage,
    )


@pytest.fixture(scope="module")
def alpha_authority(
    real_risk_workspace: RealRiskWorkspace, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, tuple[str, ...], Path]:
    """One Factor development checkpoint, published where Alpha can be told to look.

    Module-scoped because producing it runs the whole deterministic Factor
    program, and every case here consumes the same checkpoint.
    """

    return alpha_authority_for(real_risk_workspace, tmp_path_factory.mktemp("factor-run"))


def test_an_authored_alpha_document_reaches_the_real_model_runtime(
    real_risk_workspace: RealRiskWorkspace,
    alpha_authority: tuple[str, tuple[str, ...], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An authored alpha document reaches the real model runtime."""

    handle, factor_ids, evidence_root = alpha_authority
    declared = factor_ids[:8]
    document = _document(handle=handle, feature_ids=declared)
    workflow = _workflow(
        real_risk_workspace,
        document,
        evidence_root=evidence_root,
        workspace_root=tmp_path,
    )

    # Every lease of a fold, counted on the workspace itself. The checks and the
    # target digest used to live in a pre-pass that leased every fold before the
    # batch leased them all again -- so under a bounded budget, proving what ran
    # re-read the Feature and Outcome surfaces and re-ran the target
    # transformation for the whole split.
    leased: list[int] = []
    original_lease = AlphaProgramArrayWorkspace.fold_lease

    def _counted(self: AlphaProgramArrayWorkspace, fold_index: int) -> Any:
        leased.append(fold_index)
        return original_lease(self, fold_index)

    monkeypatch.setattr(AlphaProgramArrayWorkspace, "fold_lease", _counted)

    evidence, binding = workflow.run(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")

    assert evidence.disposition == "COMPUTED"
    assert binding.actor_kind is ActorKind.HUMAN
    assert evidence.formation_sessions
    # Real fits, real predictions, real metrics -- not a plan that described them.
    assert evidence.numerical_call_count > 0
    assert evidence.artifact_uris

    reader = AlphaDevelopmentReceiptReader(_ALPHA_STORE_ROOT(tmp_path))
    receipt = reader.load(evidence.artifact_uris[0])
    # Both spellings, because the generic evidence publishes the URI and a
    # researcher copying an identity out of a report has the bare hash. A reader
    # accepting only one of the two refuses the form its own producer emits.
    assert reader.load(receipt.receipt_hash).receipt_hash == receipt.receipt_hash
    verify_alpha_development_execution_receipt(receipt=receipt, evidence=evidence)

    # The declared method, recoverable rather than hashed away.
    assert receipt.target_recipe_binding.standardization_id == ROBUST_Z_STANDARDIZATION_ID
    assert receipt.target_materialization_binding.standardization_id == ROBUST_Z_STANDARDIZATION_ID
    assert receipt.ordered_base_feature_ids == declared

    # Every fold's transformed targets carry their own boundaries, so the value
    # hash is a claim about arrays rather than about one concatenated buffer.
    records = receipt.target_materialization_binding.fold_records
    assert records
    assert all(value.training_dtype == "float64" for value in records)
    assert all(len(value.training_shape) == 1 for value in records)

    # Each fold materialized exactly once, in order: the identity is a by-product
    # of the pass that fits, not a second pass over the same arrays.
    assert leased == list(range(len(records)))

    # And every fitted child is joined to that materialization by name, not by
    # position in a parallel list.
    assert receipt.child_lineage
    for entry in receipt.child_lineage:
        assert entry.ordered_factor_ids == declared
        assert entry.target_materialization_binding_hash == (
            receipt.target_materialization_binding.binding_hash
        )
        assert entry.fold_commitment_hash in {value.fold_commitment_hash for value in records}

    # Projection consumes the same verified children, not a second score read.
    score_reads: list[str] = []
    resolve = AlphaDevelopmentArtifactStore.resolve_numerical_score_chunk

    def counted_score(store: Any, reference: Any) -> Any:
        score_reads.append(reference.content_hash)
        return resolve(store, reference)

    monkeypatch.setattr(
        AlphaDevelopmentArtifactStore, "resolve_numerical_score_chunk", counted_score
    )
    projection = reader.projection(receipt.receipt_hash)
    assert projection["receipt"] == receipt.model_dump(mode="json")
    assert len(score_reads) == len(receipt.child_lineage)
    assert [fold["numerical_result_hash"] for fold in projection["fold_results"]] == [
        entry.numerical_result_hash for entry in receipt.child_lineage
    ]

    # Call-local reuse must not conceal a changed file on this same reader.
    scores = reader.root / "current/numerical-development-score-chunks"
    target = scores / f"{score_reads[0]}.parquet"
    original = target.read_bytes()
    try:
        target.write_bytes((scores / f"{score_reads[1]}.parquet").read_bytes())
        with pytest.raises(AuthoringError, match="child_unavailable"):
            reader.projection(receipt.receipt_hash)
    finally:
        target.write_bytes(original)
    assert reader.projection(receipt.receipt_hash) == projection


def _lightgbm_parameters(seed: int = 1729) -> dict[str, Any]:
    """The installed G6 estimator point, spelled as an author would in YAML."""

    point = PRODUCT_ESTIMATOR_POINT.resolve(seed=seed)
    return dict(build_dynamic_panel_lightgbm_recipe(point).parameters)


def _alpha_executor(
    workspace: RealRiskWorkspace,
    document: dict[str, Any],
    *,
    evidence_root: Path,
    workspace_root: Path,
) -> tuple[Any, Any]:
    """The installed Alpha executor and the sealed submission the Host would run."""

    envelope = ResearchExperimentEnvelope.create(**dict(document["experiment"]))
    executors = build_installed_desk_executors(
        envelope=envelope,
        workspace=workspace.workspace,
        workspace_root=workspace_root,
        document=document,
        factor_evidence_root=evidence_root,
        alpha_split_policy=load_alpha_split_policy(),
    )
    workflow = _workflow(
        workspace, document, evidence_root=evidence_root, workspace_root=workspace_root
    )
    sealed = workflow.prepare(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")
    return executors[0], sealed


def test_a_lightgbm_declaration_fits_the_installed_dynamic_panel_adapter(
    real_risk_workspace: RealRiskWorkspace,
    alpha_authority: tuple[str, tuple[str, ...], Path],
    tmp_path: Path,
) -> None:
    """A LightGBM declaration fits the installed dynamic panel adapter."""

    handle, factor_ids, evidence_root = alpha_authority
    declared = factor_ids[:8]
    parameters = _lightgbm_parameters()
    document = _document(
        handle=handle,
        feature_ids=declared,
        model_capability_handle="capability-2",
        model_parameters=parameters,
    )
    workflow = _workflow(
        real_risk_workspace, document, evidence_root=evidence_root, workspace_root=tmp_path
    )
    evidence, _binding = workflow.run(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")
    assert evidence.disposition == "COMPUTED"

    reader = AlphaDevelopmentReceiptReader(_ALPHA_STORE_ROOT(tmp_path))
    receipt = reader.load(evidence.artifact_uris[0])
    verify_alpha_development_execution_receipt(receipt=receipt, evidence=evidence)
    store = AlphaDevelopmentArtifactStore(_ALPHA_STORE_ROOT(tmp_path))
    adapter = DynamicPanelLightGBMAdapter()
    expected_recipe = build_dynamic_panel_lightgbm_recipe(
        PRODUCT_ESTIMATOR_POINT.resolve(seed=1729)
    )

    # The same fold arrays the executor built: its own preparation of the sealed
    # document, which is what ``execute`` materializes, and the PLAN preview.
    executor, sealed = _alpha_executor(
        real_risk_workspace, document, evidence_root=evidence_root, workspace_root=tmp_path
    )
    prepared = executor.prepare_execution(
        program=sealed.program, document=document, authority=sealed.authority
    )
    preview = prepared.describe()
    assert preview["model_adapter_id"] == adapter.adapter_id
    assert preview["model_recipe_hash"] == expected_recipe.recipe_hash
    assert preview["fit_protocol"] == "DIRECT_FIT"
    assert preview["model_parameters"]["training_policy"] == "FIXED_ITERATION"
    assert preview["numerical_policy"]["thread_count"] == 1
    assert preview["fit_call_upper_bound"] == preview["fold_count"]
    assert preview["expected_numerical_calls"] == 4 * preview["fold_count"]
    assert receipt.child_lineage
    plan = prepared.fold_plan
    with prepare_alpha_program_array_workspace(plan, train_session_count=None) as arrays:
        for entry in receipt.child_lineage:
            numerical = store.load_candidate_numerical_fold_result(entry.numerical_result_hash)
            operation = numerical.execution_binding_hash
            _sidecar, estimator, provenance = store.load_model_fit_sidecar(operation)
            fit_evidence = store.load_development_fit_evidence(entry.fit_evidence_hash)
            fit_plan = store.load_model_fit_plan(fit_evidence.fit_plan_hash)
            # Which adapter, which recipe, which plan.
            assert estimator.adapter_id == adapter.adapter_id
            assert provenance.recipe_hash == expected_recipe.recipe_hash
            assert fit_evidence.adapter_id == adapter.adapter_id
            assert fit_plan.protocol_id == "DIRECT_FIT"
            assert fit_plan.tuning_training_row_axis_hash is None
            assert int(estimator.payload["best_iteration"]) == parameters["fixed_iterations"]
            assert fit_evidence.fit_call_count == 1 and fit_evidence.predict_call_count == 2
            state = store.load_development_estimator_state(numerical.estimator_state_hash)
            assert state.family_id == "lightgbm" and state.ordered_factor_ids == declared
            # The sealed model, reloaded, is the model that scored.
            assert numerical.score_chunk is not None
            published = store.resolve_candidate_score_chunk(numerical.score_chunk)
            scores = published.column("score").to_pylist()
            with arrays.fold_lease(numerical.fold_index) as fold:
                assert fold.commitment.commitment_hash == numerical.fold_commitment_hash
                assert fold.training_input_binding is not None
                _training, prediction_input = bind_alpha_model_inputs(
                    binding=fold.training_input_binding,
                    ordered_feature_ids=fold.ordered_factor_ids,
                    training_features=fold.training_features,
                    training_targets=fold.training_targets,
                    training_mask=fold.training_model_mask,
                    prediction_features=fold.validation_features,
                    prediction_mask=fold.validation_feature_complete,
                )
                assert prediction_input.training_binding_hash == provenance.training_binding_hash
                reloaded = adapter.predict(estimator=estimator, inputs=prediction_input)
                scored = np.asarray([v for v in scores if v is not None], dtype=np.float64)
                assert scored.shape == reloaded.predictions.shape
                assert np.array_equal(scored, reloaded.predictions)
                unscored = sum(1 for v in scores if v is None)
                assert unscored == int((~fold.validation_feature_complete).sum())


def test_a_lightgbm_policy_without_a_stated_partition_is_refused_before_arrays(
    real_risk_workspace: RealRiskWorkspace,
    alpha_authority: tuple[str, tuple[str, ...], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A LightGBM policy without a stated partition is refused before arrays."""

    handle, factor_ids, evidence_root = alpha_authority
    parameters = {**_lightgbm_parameters(), "training_policy": "L2_EARLY_STOPPING"}
    parameters["fixed_iterations"] = None
    document = _document(
        handle=handle,
        feature_ids=factor_ids[:4],
        model_capability_handle="capability-2",
        model_parameters=parameters,
    )
    workflow = _workflow(
        real_risk_workspace, document, evidence_root=evidence_root, workspace_root=tmp_path
    )
    leased: list[int] = []
    original_lease = AlphaProgramArrayWorkspace.fold_lease

    def _counted(self: AlphaProgramArrayWorkspace, fold_index: int) -> Any:
        leased.append(fold_index)
        return original_lease(self, fold_index)

    monkeypatch.setattr(AlphaProgramArrayWorkspace, "fold_lease", _counted)
    with pytest.raises(AuthoringError, match="development_training_policy_not_installed"):
        workflow.run(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")
    assert leased == []


def test_the_declared_feature_axis_is_the_axis_that_was_fitted(
    real_risk_workspace: RealRiskWorkspace,
    alpha_authority: tuple[str, tuple[str, ...], Path],
    tmp_path: Path,
) -> None:
    """The declared feature axis is the axis that was fitted."""

    handle, selected_ids, evidence_root = alpha_authority
    declared = selected_ids[:2]
    assert len(declared) < len(selected_ids)

    workspace = real_risk_workspace
    resolver = ArtifactResolver(workspace.artifact_root)
    panel_manifest = dict(resolver.load_feature_panel_manifest(workspace.panel_manifest_ref))
    foundation, outcome_ref = build_alpha_development_foundation(
        panel_manifest=panel_manifest,
        panel_snapshot_hash=workspace.panel_snapshot_hash,
        artifact_root=workspace.artifact_root,
        factor_evidence_root=evidence_root,
        factor_evidence_handle=handle,
        source_workspace=workspace.workspace,
        output_workspace=tmp_path / "alpha-out",
    )
    # The parent axis is kept whole -- it records what the selection was made
    # from -- while the selected axis is what a document may draw from.
    assert set(foundation.selected_factor_ids).issubset(set(foundation.ordered_factor_ids))

    plan = prepare_alpha_fold_plan(
        foundation=foundation,
        ordered_listing_ids=tuple(sorted(workspace.sector_by_listing_id)),
        feature_reader=FeaturePanelReader(resolver),
        outcome_reader=CausalExecutionOutcomeDevelopmentReader(workspace.artifact_root),
        feature_panel_manifest_ref=workspace.panel_manifest_ref,
        causal_outcome_manifest_ref=outcome_ref,
        frozen_at=_FROZEN_AT,
        split_policy=load_alpha_split_policy(),
        ordered_base_feature_ids=declared,
    )
    assert plan.base_feature_ids == declared
    assert plan.base_feature_ids != foundation.ordered_factor_ids

    # And the arrays agree: width, estimator axis and training binding all follow
    # the declaration rather than the parent.
    with (
        prepare_alpha_program_array_workspace(plan, train_session_count=None) as arrays,
        arrays.fold_lease(0) as fold,
    ):
        assert fold.ordered_factor_ids == declared
        assert fold.training_features.shape[1] == len(declared)
        assert fold.validation_features.shape[1] == len(declared)
        if fold.training_input_binding is not None:
            assert tuple(fold.training_input_binding.ordered_feature_ids) == declared


def test_an_axis_outside_the_parent_evidence_is_refused() -> None:
    """A declared axis may narrow the parent's, never widen it."""

    from alphalattice.investment.alpha_research.inputs.folds import _assert_admissible_base_axis

    authorized = ("factor.a", "factor.b", "factor.c")
    _assert_admissible_base_axis(("factor.c", "factor.a"), authorized)

    with pytest.raises(AlphaArrayBoundaryError, match="NOT_AUTHORIZED"):
        _assert_admissible_base_axis(("factor.a", "factor.z"), authorized)
    with pytest.raises(AlphaArrayBoundaryError, match="DUPLICATED"):
        _assert_admissible_base_axis(("factor.a", "factor.a"), authorized)
    with pytest.raises(AlphaArrayBoundaryError, match="EMPTY"):
        _assert_admissible_base_axis((), authorized)


def test_a_receipt_cannot_pair_one_recipe_with_another_recipes_values() -> None:
    """A receipt cannot pair one recipe with another recipe's values."""

    methods = installed_alpha_target_methods(
        sector_revision="a" * 64,
        execution_outcome_recipe_id=ONE_SESSION_RECIPE_ID,
    )
    robust = AlphaTargetMethodBinding.create(
        method=methods.resolve(CROSS_STANDARDIZED_DEVELOPMENT_TARGET_RECIPE_ID),
        execution_outcome_recipe_id=ONE_SESSION_RECIPE_ID,
        causal_outcome_snapshot_hash="c" * 64,
        outcome_method_binding_hash="d" * 64,
    )
    rank = AlphaTargetMethodBinding.create(
        method=methods.resolve(DEFAULT_DEVELOPMENT_TARGET_RECIPE_ID),
        execution_outcome_recipe_id=ONE_SESSION_RECIPE_ID,
        causal_outcome_snapshot_hash="c" * 64,
        outcome_method_binding_hash="d" * 64,
    )
    record = AlphaDevelopmentTargetFoldRecord.create(
        fold_index=0,
        fold_commitment_hash="d" * 64,
        training_dtype="float64",
        training_shape=(120,),
        training_value_hash="e" * 64,
        validation_dtype="float64",
        validation_shape=(30,),
        validation_value_hash="f" * 64,
    )
    materialized = AlphaDevelopmentTargetMaterializationBinding.create(
        recipe_binding=rank,
        ordered_base_feature_ids=("factor.a",),
        fold_records=(record,),
        training_row_count=120,
    )

    split_policy = derive_alpha_development_split_policy(
        geometry=load_alpha_split_policy(),
        maturity_lag_sessions=2,
        session_count=2515,
    )

    def _receipt(
        recipe_binding: AlphaTargetMethodBinding,
        materialization: AlphaDevelopmentTargetMaterializationBinding | None = None,
    ) -> object:
        materialized_binding = materialization or materialized
        return AlphaDevelopmentExecutionReceipt.create(
            program_hash="1" * 64,
            desk_program_hash="2" * 64,
            method_binding_hash="3" * 64,
            desk_input_binding_hash="4" * 64,
            target_recipe_binding=recipe_binding,
            target_materialization_binding=materialized_binding,
            ordered_base_feature_ids=("factor.a",),
            research_recipe_hash="5" * 64,
            model_adapter_id="ridge",
            model_adapter_recipe_hash="6" * 64,
            development_program_hash="7" * 64,
            split_policy=split_policy,
            model_method_binding_hash="e" * 64,
            batch_hash="8" * 64,
            batch_result_hash="9" * 64,
            development_surface_hash="a" * 64,
            development_surface_binding_hash="b" * 64,
            fold_surface_hashes=("c" * 64,),
            child_lineage=(
                seal_current_contract(
                    AlphaDevelopmentChildLineage,
                    {
                        "kind": "AlphaDevelopmentChildLineage",
                        "candidate_id": "alpha-candidate-0000000000000000",
                        "fold_index": 0,
                        "fold_commitment_hash": "d" * 64,
                        "numerical_result_hash": "0" * 64,
                        "estimator_state_hash": "1" * 64,
                        "fit_evidence_hash": "2" * 64,
                        "training_input_binding_hash": "3" * 64,
                        "estimator_content_hash": "4" * 64,
                        "fit_provenance_hash": "5" * 64,
                        "score_evidence_hash": "6" * 64,
                        "ordered_factor_ids": ("factor.a",),
                        "target_recipe_binding_hash": recipe_binding.binding_hash,
                        "target_materialization_binding_hash": materialized_binding.binding_hash,
                    },
                    "lineage_hash",
                ),
            ),
        )

    # The consistent pairing is admitted.
    assert _receipt(rank)
    # The mixed one is refused before it can be published, let alone read.
    with pytest.raises(ValueError, match="target_layers_disagree"):
        _receipt(robust)

    def _mutated(**overrides: object) -> AlphaDevelopmentTargetMaterializationBinding:
        """A materialization built by hand and resealed, so its own hash is valid.

        ``create`` copies the standardization from the recipe binding, so the
        mismatch is unreachable through it -- and that is exactly why the receipt
        has to check rather than trust: the contract can be constructed directly.
        """

        payload = materialized.model_dump(mode="json")
        payload.update(overrides)
        payload.pop("binding_hash")
        return AlphaDevelopmentTargetMaterializationBinding.model_validate(
            {**payload, "binding_hash": canonical_hash(payload)}
        )

    # requirement: the two layers must name one standardization. The recipe
    # binding hash agrees here, so nothing else in the chain notices.
    with pytest.raises(ValueError, match="standardization_disagrees"):
        _receipt(rank, _mutated(standardization_id=ROBUST_Z_STANDARDIZATION_ID))

    # requirement: the row count is a consequence of the fold records, not a free
    # field beside them.
    with pytest.raises(ValueError, match="training_rows_disagree"):
        _receipt(rank, _mutated(training_row_count=119))


def test_a_chunk_read_again_is_checked_again_and_changed_bytes_are_refused(
    tmp_path: Path,
) -> None:
    """Requirement: a chunk whose content hash a process derived once is served for the
    same file bytes, so each read still hashes the file: bytes changed after a read are refused,
    and the original bytes read back again."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    from alphalattice.investment.alpha_research.experiments.development_artifacts import (
        AlphaDevelopmentArtifactReadbackError,
        AlphaDevelopmentArtifactStore,
    )

    store = AlphaDevelopmentArtifactStore(tmp_path)
    table = pa.table(
        {
            "row_index": pa.array([0, 1, 2], pa.int64()),
            "score": pa.array([0.1, -0.2, 0.3]),
            "availability": pa.array([True, True, False]),
        }
    )
    reference = store.publish_numerical_score_chunk(
        table,
        execution_binding_hash="a" * 64,
        development_surface_binding_hash="b" * 64,
        candidate_id="C1",
        candidate_card_hash="c" * 64,
        fold_index=0,
        fold_commitment_hash="d" * 64,
    )
    assert store.resolve_numerical_score_chunk(reference).equals(table)
    path = next(tmp_path.rglob(f"{reference.content_hash}.parquet"))
    original = path.read_bytes()
    changed = table.set_column(1, "score", pa.array([0.1, -0.2, 0.4]))
    pq.write_table(
        changed.replace_schema_metadata(pq.read_schema(path).metadata), path, compression="zstd"
    )
    with pytest.raises(AlphaDevelopmentArtifactReadbackError, match="content changed"):
        store.resolve_numerical_score_chunk(reference)
    path.write_bytes(original)
    assert store.resolve_numerical_score_chunk(reference).equals(table)


def test_a_tampered_child_lineage_entry_is_refused_on_readback(
    real_risk_workspace: RealRiskWorkspace,
    alpha_authority: tuple[str, tuple[str, ...], Path],
    tmp_path: Path,
) -> None:
    """A tampered child lineage entry is refused on readback."""

    handle, factor_ids, evidence_root = alpha_authority
    document = _document(handle=handle, feature_ids=factor_ids[:2])
    workflow = _workflow(
        real_risk_workspace,
        document,
        evidence_root=evidence_root,
        workspace_root=tmp_path,
    )
    evidence, _binding = workflow.run(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")

    store_root = _ALPHA_STORE_ROOT(tmp_path)
    reader = AlphaDevelopmentReceiptReader(store_root)
    receipt = reader.load(evidence.artifact_uris[0])
    assert len(receipt.child_lineage) > 1

    # One fold's entry pointing at another fold's estimator state. The fold
    # commitment still belongs to the materialization, so the receipt's own
    # validators are satisfied.
    first, second = receipt.child_lineage[0], receipt.child_lineage[1]
    swapped = seal_current_contract(
        AlphaDevelopmentChildLineage,
        {
            **first.model_dump(exclude={"lineage_hash"}),
            "estimator_state_hash": second.estimator_state_hash,
        },
        "lineage_hash",
    )
    store = AlphaDevelopmentArtifactStore(store_root)

    def _resealed(lineage: tuple[AlphaDevelopmentChildLineage, ...]) -> str:
        forged = _rebuilt_receipt(receipt, child_lineage=lineage)
        assert forged.receipt_hash != receipt.receipt_hash
        store.publish_development_execution_receipt(forged)
        return forged.receipt_hash

    # Refused at the first artifact that disagrees: the numerical fold result
    # names its own estimator state, and that pointer is not the one the entry
    # claims. Nothing in the receipt could have detected this on its own.
    with pytest.raises(AuthoringError, match="child_result_mismatch"):
        reader.load(_resealed((swapped, *receipt.child_lineage[1:])))

    # requirement: the lineage must be *all* the children, not a truthful subset.
    # Every remaining entry here resolves perfectly; the receipt is resealed and
    # valid on its own terms. Only the batch result -- sealed after the run and
    # naming every child it produced -- can say one is missing.
    with pytest.raises(AuthoringError, match="child_set_incomplete"):
        reader.load(_resealed(receipt.child_lineage[:-1]))

    # And order is part of the claim. A reversed lineage drops nothing and
    # resolves every child, so a set comparison would admit it -- which is why
    # the batch result is rebuilt as an ordered tuple rather than a set.
    with pytest.raises(AuthoringError, match="child_set_incomplete"):
        reader.load(_resealed(tuple(reversed(receipt.child_lineage))))

    # requirement: a fold dropped from the batch result *and* the receipt together
    # is still refused.
    #
    # This is the case comparing the two post-run records cannot see: they agree
    # with each other perfectly, because both were resealed. Only the surface
    # manifest -- written before any fit, naming the candidate cards and the whole
    # fold axis -- still says how many children there were supposed to be.
    batch = store.load_batch_result(receipt.batch_result_hash)
    original = batch.candidates[0]
    assert original.fold_surface_hashes is not None
    truncated_candidate = seal_contract(
        AlphaExperimentCandidateResult,
        {
            **original.model_dump(exclude={"result_hash"}),
            "numerical_result_hashes": original.numerical_result_hashes[:-1],
            "estimator_state_hashes": original.estimator_state_hashes[:-1],
            "fold_surface_hashes": original.fold_surface_hashes[:-1],
        },
        "result_hash",
    )
    truncated_batch = seal_contract(
        AlphaExperimentBatchResult,
        {
            **batch.model_dump(exclude={"result_hash", "candidates"}),
            "candidates": (truncated_candidate,),
            "fold_surface_hashes": batch.fold_surface_hashes[:-1],
        },
        "result_hash",
    )
    assert truncated_batch.result_hash != batch.result_hash
    store.publish_batch_result(truncated_batch)

    consistent_forgery = _rebuilt_receipt(
        receipt,
        child_lineage=receipt.child_lineage[:-1],
        batch_result_hash=truncated_batch.result_hash,
        fold_surface_hashes=truncated_batch.fold_surface_hashes,
    )
    store.publish_development_execution_receipt(consistent_forgery)
    with pytest.raises(AuthoringError, match="child_set_incomplete"):
        reader.load(consistent_forgery.receipt_hash)


def _fields(model: Any, identity_field: str) -> dict[str, Any]:
    """A model's field values as attributes, minus its identity, for resealing.

    Attributes rather than a dump: nested contracts stay contracts and tuples
    stay tuples, so the resealed model is built the way the producer built it.
    """

    return {
        name: getattr(model, name) for name in type(model).model_fields if name != identity_field
    }


@contextmanager
def _forgery(label: str) -> Iterator[None]:
    """Name the forgery in the failure message when a refusal does not come."""

    try:
        yield
    except AssertionError as error:  # pragma: no cover - failure reporting
        raise AssertionError(f"{label} forgery was admitted: {error}") from error


def _forged_receipt(
    store: AlphaDevelopmentArtifactStore,
    receipt: AlphaDevelopmentExecutionReceipt,
    *,
    candidate: dict[str, Any] | None = None,
    report: AlphaCandidateDevelopmentReport | None = None,
    inference: AlphaCandidateInferenceEvidence | None = None,
) -> str:
    """Reseal the chain above one candidate's evidence, valid on its own terms.

    A replacement report or inference evidence is published and named by a
    resealed candidate result; the batch result and the receipt are resealed to
    name them in turn. Every artifact along the chain then carries its own
    valid identity -- the only thing that can refuse it is a reader that proves
    what each one binds.
    """

    batch = store.load_batch_result(receipt.batch_result_hash)
    fields = _fields(batch.candidates[0], "result_hash")
    if report is not None:
        store.publish_candidate_report(report)
        fields["development_report_hash"] = report.report_hash
        if inference is None:
            # The inference evidence names the report; rebound so the forgery
            # under test is the report's binding alone.
            inference = seal_current_contract(
                AlphaCandidateInferenceEvidence,
                {
                    **_fields(
                        store.load_candidate_inference_evidence(fields["inference_evidence_hash"]),
                        "evidence_hash",
                    ),
                    "report_hash": report.report_hash,
                },
                "evidence_hash",
            )
    if inference is not None:
        store.publish_candidate_inference_evidence(inference)
        fields["inference_evidence_hash"] = inference.evidence_hash
    fields.update(candidate or {})
    forged_candidate = seal_contract(AlphaExperimentCandidateResult, fields, "result_hash")
    forged_batch = seal_contract(
        AlphaExperimentBatchResult,
        {
            **{k: v for k, v in _fields(batch, "result_hash").items() if k != "candidates"},
            "candidates": (forged_candidate, *batch.candidates[1:]),
        },
        "result_hash",
    )
    store.publish_batch_result(forged_batch)
    forged = _rebuilt_receipt(
        receipt, child_lineage=receipt.child_lineage, batch_result_hash=forged_batch.result_hash
    )
    assert forged.receipt_hash != receipt.receipt_hash
    store.publish_development_execution_receipt(forged)
    return forged.receipt_hash


def test_the_candidate_evidence_a_batch_names_is_verified_on_readback(
    real_risk_workspace: RealRiskWorkspace,
    alpha_authority: tuple[str, tuple[str, ...], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The candidate evidence a batch names is verified on readback."""

    handle, factor_ids, evidence_root = alpha_authority
    document = _document(handle=handle, feature_ids=factor_ids[:2])
    workflow = _workflow(
        real_risk_workspace, document, evidence_root=evidence_root, workspace_root=tmp_path
    )
    evidence, _binding = workflow.run(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")
    # A second run over another axis: another candidate, program and surface in
    # the same store, whose evidence is valid and belongs somewhere else.
    other_document = _document(handle=handle, feature_ids=factor_ids[:1])
    other_workflow = _workflow(
        real_risk_workspace, other_document, evidence_root=evidence_root, workspace_root=tmp_path
    )
    other_evidence, _other = other_workflow.run(
        other_document, actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )

    store_root = _ALPHA_STORE_ROOT(tmp_path)
    reader = AlphaDevelopmentReceiptReader(store_root)
    store = AlphaDevelopmentArtifactStore(store_root)
    graph = reader.read(evidence.artifact_uris[0])
    receipt, batch = graph.receipt, graph.batch
    assert len(batch.candidates) == 1 and len(graph.candidate_reports) == 1
    (report,), (inference,) = graph.candidate_reports, graph.inference_evidence
    assert report.report_hash == batch.candidates[0].development_report_hash
    assert inference.evidence_hash == batch.candidates[0].inference_evidence_hash
    assert len(report.fold_evidence_hashes) == len(receipt.child_lineage) > 1
    projection = reader.projection_of(graph)
    assert projection["candidate_reports"][0]["report_hash"] == report.report_hash
    assert projection["inference_evidence"][0]["evidence_hash"] == inference.evidence_hash
    other_graph = reader.read(other_evidence.artifact_uris[0])
    assert other_graph.batch.program_hash != batch.program_hash

    def read(handle: str) -> None:
        reader.read(handle)

    current = store_root / "alpha-research" / "current"

    # Missing, then damaged, then restored: refused while absent or altered,
    # read back again once the bytes are back -- nothing is remembered between
    # walks in either direction.
    for category, artifact_hash, code in (
        ("candidate-development-reports", report.report_hash, "report_unavailable"),
        ("candidate-inference-evidence", inference.evidence_hash, "inference_unavailable"),
        (
            "candidate-fold-evidence",
            report.fold_evidence_hashes[1],
            "fold_evidence_unavailable",
        ),
    ):
        path = current / category / f"{artifact_hash}.json"
        original = path.read_bytes()
        path.unlink()
        with pytest.raises(AuthoringError, match=code):
            read(receipt.receipt_hash)
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
        with pytest.raises(AuthoringError, match=code):
            read(receipt.receipt_hash)
        path.write_bytes(original)
        read(receipt.receipt_hash)

    # requirement: a fold evidence list with a fold out of order, repeated or
    # taken from another candidate is refused even though every file it names
    # is valid and the report, batch and receipt were all resealed around it.
    report_fields = _fields(report, "report_hash")
    hashes = report.fold_evidence_hashes
    for label, forged_hashes, code in (
        ("reversed", tuple(reversed(hashes)), "fold_evidence_mismatch"),
        ("repeated", (hashes[0], hashes[0], *hashes[2:]), "report_mismatch"),
        (
            "another candidate's",
            (other_graph.candidate_reports[0].fold_evidence_hashes[0], *hashes[1:]),
            "fold_evidence_mismatch",
        ),
    ):
        forged_report = seal_current_contract(
            AlphaCandidateDevelopmentReport,
            {**report_fields, "fold_evidence_hashes": forged_hashes},
            "report_hash",
        )
        with pytest.raises(AuthoringError, match=code), _forgery(label):
            read(_forged_receipt(store, receipt, report=forged_report))

    # requirement: a report of another candidate, or one whose aggregate does
    # not aggregate these folds' metrics, is not this candidate's report.
    with pytest.raises(AuthoringError, match="report_mismatch"):
        read(
            _forged_receipt(
                store,
                receipt,
                candidate={
                    "development_report_hash": (
                        other_graph.batch.candidates[0].development_report_hash
                    )
                },
            )
        )
    swapped_metrics = seal_current_contract(
        AlphaCandidateDevelopmentReport,
        {
            **report_fields,
            "metrics": other_graph.candidate_reports[0].metrics,
        },
        "report_hash",
    )
    with pytest.raises(AuthoringError, match="report_mismatch"):
        read(_forged_receipt(store, receipt, report=swapped_metrics))

    # requirement: inference evidence that names the score chunks in another
    # order, or another candidate's, is refused against the resolved children.
    inference_fields = _fields(inference, "evidence_hash")
    for label, forged_fields, code in (
        (
            "reversed scores",
            {"score_chunk_hashes": tuple(reversed(inference.score_chunk_hashes))},
            "inference_mismatch",
        ),
        (
            "another candidate's",
            {
                "candidate_id": other_graph.inference_evidence[0].candidate_id,
                "report_hash": other_graph.inference_evidence[0].report_hash,
            },
            "inference_mismatch",
        ),
    ):
        forged_inference = seal_current_contract(
            AlphaCandidateInferenceEvidence,
            {**inference_fields, **forged_fields},
            "evidence_hash",
        )
        with pytest.raises(AuthoringError, match=code), _forgery(label):
            read(_forged_receipt(store, receipt, inference=forged_inference))
    with pytest.raises(AuthoringError, match="inference_mismatch"):
        read(
            _forged_receipt(
                store,
                receipt,
                candidate={
                    "inference_evidence_hash": (
                        other_graph.batch.candidates[0].inference_evidence_hash
                    )
                },
            )
        )

    # requirement: a fold evidence in the shape parents published before thin
    # references is verified by the fields that shape has, and by no other.
    # Written directly -- the active publisher only writes the thin contract --
    # under the hash the legacy contract seals for itself.
    entry, numerical = receipt.child_lineage[0], graph.folds[0]
    thin = store.load_candidate_fold_evidence(hashes[0])
    assert isinstance(thin, AlphaCandidateFoldEvidence)
    legacy_fields: dict[str, Any] = {
        **_fields(thin, "fold_evidence_hash"),
        "role": numerical.role,
        "status": numerical.status,
        "score_chunk": numerical.score_chunk,
        "metrics": numerical.metrics,
        "session_rank_ics": numerical.session_rank_ics,
        "session_spreads": numerical.session_spreads,
        "fit_ledger": numerical.fit_ledger,
        "estimator_state_hash": entry.estimator_state_hash,
        "failure": None,
    }

    def _legacy(**overrides: Any) -> str:
        legacy = seal_current_contract(
            LegacyAlphaCandidateFoldEvidence, {**legacy_fields, **overrides}, "fold_evidence_hash"
        )
        path = current / "candidate-fold-evidence" / f"{legacy.fold_evidence_hash}.json"
        path.write_text(json.dumps(legacy.model_dump(mode="json")), encoding="utf-8")
        assert isinstance(
            store.load_candidate_fold_evidence(legacy.fold_evidence_hash),
            LegacyAlphaCandidateFoldEvidence,
        )
        return legacy.fold_evidence_hash

    def _legacy_receipt(**overrides: Any) -> str:
        return _forged_receipt(
            store,
            receipt,
            report=seal_current_contract(
                AlphaCandidateDevelopmentReport,
                {**report_fields, "fold_evidence_hashes": (_legacy(**overrides), *hashes[1:])},
                "report_hash",
            ),
        )

    resolutions: list[int] = []
    original_resolve = AlphaDevelopmentArtifactStore._resolve_development_chunk

    def counted_resolve(self: AlphaDevelopmentArtifactStore, **kwargs: Any) -> Any:
        resolutions.append(1)
        return original_resolve(self, **kwargs)

    monkeypatch.setattr(
        AlphaDevelopmentArtifactStore, "_resolve_development_chunk", counted_resolve
    )
    resolutions.clear()
    read(receipt.receipt_hash)
    thin_resolutions = len(resolutions)
    # Both historical shapes, whole: one naming its numerical result and one
    # predating that reference, each repeating its fold's facts. Comparing the
    # repeated facts adds no chunk resolution beyond the one the legacy loader
    # has always made for the score chunk the evidence itself carries.
    for whole in (_legacy_receipt(), _legacy_receipt(numerical_result_hash=None)):
        resolutions.clear()
        read(whole)
        assert len(resolutions) == thin_resolutions + 1

    # requirement: a legacy evidence that repeats a fold's facts must repeat
    # the numerical result's facts. Each forgery below is valid on its own
    # terms -- the ledger resealed through its own contract, the evidence, the
    # report and the chain resealed around it -- and was admitted when only the
    # role, status and metrics were compared.
    ledger = numerical.fit_ledger
    assert ledger is not None
    other = receipt.child_lineage[1]
    other_result = graph.folds[1]
    assert other_result.metrics != numerical.metrics
    contradicting_ledger = seal_contract(
        AlphaFitLedgerEntry,
        {**_fields(ledger, "ledger_hash"), "training_row_count": ledger.training_row_count + 1},
        "ledger_hash",
    )
    for label, overrides in (
        ("fit ledger one training row off", {"fit_ledger": contradicting_ledger}),
        ("one session rank IC more", {"session_rank_ics": (*numerical.session_rank_ics, 0.0)}),
        ("one session spread more", {"session_spreads": (*numerical.session_spreads, 0.0)}),
        ("another fold's estimator state", {"estimator_state_hash": other.estimator_state_hash}),
        ("estimator state omitted beside a numerical reference", {"estimator_state_hash": None}),
        ("another fold's metrics", {"metrics": other_result.metrics}),
        (
            "no numerical reference, fit ledger one training row off",
            {"numerical_result_hash": None, "fit_ledger": contradicting_ledger},
        ),
        (
            "no numerical reference, another fold's estimator state",
            {"numerical_result_hash": None, "estimator_state_hash": other.estimator_state_hash},
        ),
    ):
        with pytest.raises(AuthoringError, match="fold_evidence_mismatch"), _forgery(label):
            read(_legacy_receipt(**overrides))
    # A score chunk of another fold is refused by the legacy contract itself.
    with pytest.raises(ValueError, match="score chunk binding differs"):
        _legacy(score_chunk=other_result.score_chunk)

    # requirement: without a numerical reference the score chunk is still held
    # to the numerical result, by the chunk's own shape. Every chunk below is
    # sealed by the store's own publisher over the rows this fold scored and
    # resolves under its own identity; none is a damaged reference. A numerical
    # chunk is one execution's payload, so this candidate's and fold's rows
    # under the other run's execution binding, with this run's surface binding
    # or the other run's, are not this result's. The older chunk shape is
    # another payload identity whose content hash is never the numerical
    # chunk's: it is held to the request, surface, candidate, card, fold,
    # commitment and row axis it carries, so the whole pre-reference shape
    # reads back and the same rows sealed under the other run's program and
    # surface do not; beside a numerical reference it is not that result's
    # chunk at all.
    _numerical, rows = store.read_candidate_numerical_fold_result(entry.numerical_result_hash)
    assert rows is not None and numerical.score_chunk is not None
    other_run = other_graph.folds[0]
    fold_identity = {
        "candidate_id": numerical.candidate_id,
        "candidate_card_hash": numerical.candidate_card_hash,
        "fold_index": numerical.fold_index,
        "fold_commitment_hash": numerical.fold_commitment_hash,
    }
    older_shape = store.publish_candidate_score_chunk(
        rows,
        request_hash=batch.program_hash,
        development_surface_hash=str(batch.candidates[0].development_surface_hash),
        **fold_identity,
    )
    assert older_shape.content_hash != numerical.score_chunk.content_hash
    whole_older_shape = _legacy_receipt(numerical_result_hash=None, score_chunk=older_shape)
    resolutions.clear()
    read(whole_older_shape)
    assert len(resolutions) == thin_resolutions + 1
    for label, chunk, overrides in (
        (
            "another execution's numerical chunk, this surface binding",
            store.publish_numerical_score_chunk(
                rows,
                execution_binding_hash=other_run.execution_binding_hash,
                development_surface_binding_hash=numerical.development_surface_binding_hash,
                **fold_identity,
            ),
            {"numerical_result_hash": None},
        ),
        (
            "another execution's numerical chunk and surface binding",
            store.publish_numerical_score_chunk(
                rows,
                execution_binding_hash=other_run.execution_binding_hash,
                development_surface_binding_hash=other_run.development_surface_binding_hash,
                **fold_identity,
            ),
            {"numerical_result_hash": None},
        ),
        (
            "older-shape chunk under the other run's program and surface",
            store.publish_candidate_score_chunk(
                rows,
                request_hash=other_graph.batch.program_hash,
                development_surface_hash=str(
                    other_graph.batch.candidates[0].development_surface_hash
                ),
                **fold_identity,
            ),
            {"numerical_result_hash": None},
        ),
        ("older-shape chunk beside a numerical reference", older_shape, {}),
    ):
        assert chunk.content_hash != numerical.score_chunk.content_hash
        with pytest.raises(AuthoringError, match="fold_evidence_mismatch"), _forgery(label):
            read(_legacy_receipt(score_chunk=chunk, **overrides))

    # The untouched receipt still reads back whole after every refusal above.
    assert reader.read(receipt.receipt_hash).candidate_reports == (report,)


def test_the_declared_standardization_is_the_one_that_executed(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """The declared standardization is the one that executed."""

    recipes = installed_alpha_development_target_recipes(sector_revision="a" * 64)
    recipe = recipes.resolve(CROSS_STANDARDIZED_DEVELOPMENT_TARGET_RECIPE_ID)

    # The lane still says rank-gauss and always will -- it is a property of a
    # closed enum. The recipe says robust-z. The two are genuinely separable.
    assert recipe.policy.standardization_id == RANK_GAUSS_STANDARDIZATION_ID
    assert recipe.standardization_id == ROBUST_Z_STANDARDIZATION_ID

    sessions = [datetime(2024, 5, day, tzinfo=UTC).date() for day in (1, 2, 3)]
    listings = tuple(f"listing-{index:02d}" for index in range(12))
    sectors = {value: ("s-1" if index < 6 else "s-2") for index, value in enumerate(listings)}
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0, 0.02, size=len(sessions) * len(listings))
    import pyarrow as pa

    source = pa.table(
        {
            "formation_session": [session for session in sessions for _ in listings],
            "listing_id": [listing for _ in sessions for listing in listings],
            "fit_target": pa.array(returns, type=pa.float64()),
            "simple_economic_return": pa.array(np.expm1(returns), type=pa.float64()),
        }
    )
    surface = compile_alpha_development_target_surface(
        source_table=source, recipe=recipe, sector_by_listing_id=sectors
    )
    values = np.asarray(
        surface["fit_target"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64
    )
    finite = values[np.isfinite(values)]
    assert finite.size > 0
    # Rank-gauss maps a cross-section onto normal scores, so its per-session
    # values are symmetric about zero with a fixed spread regardless of input
    # scale. Robust-z rescales by the observed MAD and does not. Distinguishing
    # them by output shape is what makes this a real check rather than a
    # restatement of the recipe field.
    assert not np.isclose(float(np.nanstd(finite)), 1.0, atol=0.05)


def test_a_feature_axis_outside_the_factor_evidence_is_refused(
    real_risk_workspace: RealRiskWorkspace,
    alpha_authority: tuple[str, tuple[str, ...], Path],
    tmp_path: Path,
) -> None:
    """The declared axis must be authorized, and is never quietly truncated."""

    handle, factor_ids, evidence_root = alpha_authority
    unauthorized = _document(handle=handle, feature_ids=(*factor_ids[:2], "factor.not.published"))
    workflow = _workflow(
        real_risk_workspace,
        unauthorized,
        evidence_root=evidence_root,
        workspace_root=tmp_path,
    )
    with pytest.raises(AuthoringError, match="feature_axis_not_in_panel"):
        workflow.run(unauthorized, actor_kind=ActorKind.HUMAN, actor_id="researcher")


def test_evidence_root_may_not_overlap_the_workspaces_it_authorizes() -> None:
    """A nested evidence root makes this run's output next run's authority."""

    root = Path("workspaces/alpha")
    assert_distinct_roots(Path("evidence"), Path("source"), root)
    with pytest.raises(AuthoringError, match="evidence_root_nested"):
        assert_distinct_roots(root / "evidence", Path("source"), root)
    with pytest.raises(AuthoringError, match="evidence_root_identical"):
        assert_distinct_roots(root, Path("source"), root)


def test_the_factor_evidence_handle_is_read_by_hash_not_by_recency(
    real_risk_workspace: RealRiskWorkspace,
    alpha_authority: tuple[str, tuple[str, ...], Path],
) -> None:
    """A handle naming no stored checkpoint fails; nothing falls back to newest."""

    from alphalattice.foundation.factor_research.experiments.development_evidence import (
        FactorDevelopmentReceiptReader,
    )

    handle, _factor_ids, evidence_root = alpha_authority
    reader = FactorDevelopmentReceiptReader(evidence_root)
    receipt, child = reader.load(handle)
    assert receipt.receipt_hash == handle
    assert receipt.checkpoint_hash == child.checkpoint_hash

    with pytest.raises(AuthoringError, match="development_receipt_unavailable"):
        reader.load("b" * 64)
    # A handle is content addressing, not a location, and is admitted as exactly
    # 64 lowercase hex rather than merely "no separators".
    for bad in ("../secrets", "not-a-hash", "A" * 64):
        with pytest.raises(AuthoringError, match="receipt_handle_invalid"):
            reader.load(bad)


def test_the_panel_manifest_reader_is_the_one_the_host_uses(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """The Host resolves the axis it validates against, rather than being told it."""

    resolver = ArtifactResolver(real_risk_workspace.artifact_root)
    manifest = resolver.load_feature_panel_manifest(real_risk_workspace.panel_manifest_ref)
    summary = dict(manifest)["safe_summary"]["factor_catalog_summary"]
    assert summary
    # Every entry carries method identity, which is what the Factor Desk binds
    # and therefore what an Alpha axis is checked against.
    assert all(len(entry["methodology_hash"]) == 64 for entry in summary.values())


def _saved_alpha_side(
    real_risk_workspace: RealRiskWorkspace,
    document: dict[str, Any],
    *,
    evidence_root: Path,
    tmp_path: Path,
    reader: AlphaDevelopmentReceiptReader,
    before_read: Callable[[], None] = lambda: None,
) -> tuple[dict[str, Any], str, AlphaDevelopmentVerifiedGraph]:
    """Run one saved Alpha study and hand back what a comparison consumes: the
    readback body, the first candidate and the graph one verification walked.
    ``before_read`` runs between the run and the walk (a counter's reset)."""

    evidence, _binding = _workflow(
        real_risk_workspace,
        document,
        evidence_root=evidence_root,
        workspace_root=tmp_path,
    ).run(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")
    before_read()
    graph = reader.read(evidence.artifact_uris[0])
    projection = reader.projection_of(graph)
    candidate_id = projection["result"]["candidates"][0]["candidate_id"]
    body = {
        "status": "EXPERIMENT_PUBLISHED",
        "task_id": str(uuid4()),
        "program": {"program_hash": projection["receipt"]["program_hash"]},
        "document": document,
        "input_binding_hash": "a" * 64,
        **projection,
    }
    return body, candidate_id, graph


def test_saved_alpha_comparison_reads_two_real_ridge_receipts_without_recomputing_metrics(
    real_risk_workspace: RealRiskWorkspace,
    alpha_authority: tuple[str, tuple[str, ...], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saved alpha comparison reads two real ridge receipts without recomputing metrics."""

    handle, factor_ids, evidence_root = alpha_authority
    reader = AlphaDevelopmentReceiptReader(_ALPHA_STORE_ROOT(tmp_path))
    left_document = _document(
        handle=handle,
        feature_ids=factor_ids[:8],
        model_parameters={"family": "ridge", "alpha": 1.0},
    )
    right_document = _document(
        handle=handle,
        feature_ids=factor_ids[:8],
        model_parameters={"family": "ridge", "alpha": 100.0},
    )
    resolved: list[str] = []
    real_resolve = AlphaDevelopmentArtifactStore._resolve_development_chunk

    def counted(self: AlphaDevelopmentArtifactStore, **kwargs: Any) -> Any:
        resolved.append(f"{kwargs['category']}/{kwargs['reference'].content_hash}")
        return real_resolve(self, **kwargs)

    monkeypatch.setattr(AlphaDevelopmentArtifactStore, "_resolve_development_chunk", counted)
    left, left_candidate_id, left_graph = _saved_alpha_side(
        real_risk_workspace,
        left_document,
        evidence_root=evidence_root,
        tmp_path=tmp_path,
        reader=reader,
        before_read=resolved.clear,
    )
    left_walk = list(resolved)
    right, right_candidate_id, right_graph = _saved_alpha_side(
        real_risk_workspace,
        right_document,
        evidence_root=evidence_root,
        tmp_path=tmp_path,
        reader=reader,
        before_read=resolved.clear,
    )
    right_walk = list(resolved)
    # Each verification walk resolved every chunk of its graph exactly once; the
    # comparison below adds nothing to that count.
    for walk in (left_walk, right_walk):
        assert walk and len(walk) == len(set(walk)), "each chunk proved once, by the walk"
    walked = list(resolved)
    from alphalattice.investment.alpha_research.experiments import score_rows

    def matrix_assembly_unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("matrix assembly is unused")

    monkeypatch.setattr(score_rows, "assemble_score_matrix", matrix_assembly_unexpected)
    comparison = compare_saved_alpha_candidates(
        left=left,
        left_candidate_id=left_candidate_id,
        left_graph=left_graph,
        right=right,
        right_candidate_id=right_candidate_id,
        right_graph=right_graph,
    )
    assert resolved == walked, "the comparison consumed the verified graphs and resolved no chunk"

    assert comparison["status"] == "COMPARABLE"
    assert comparison["disposition"] == "DESCRIPTIVE_ALPHA_COMPARISON_NO_SELECTION"
    assert comparison["declared_parameter_difference"] == {
        "left": {"family": "ridge", "alpha": 1.0},
        "right": {"family": "ridge", "alpha": 100.0},
    }
    assert len(comparison["folds"]) == len(left["fold_results"])
    assert comparison["score_support"]["scored_row_count"] > 0
    assert "values" not in comparison["score_support"]
    assert "formation_sessions" not in comparison["score_support"]
    assert "ordered_listing_ids" not in comparison["score_support"]
    assert "availability" not in comparison["score_support"]

    # The graph is a value of one request: the next request walks again, and a
    # chunk changed in between is refused then, not served from the last walk.
    score_chunk = next(
        value.score_chunk for value in left_graph.folds if value.score_chunk is not None
    )
    target = (
        reader.root
        / "current"
        / "numerical-development-score-chunks"
        / f"{score_chunk.content_hash}.parquet"
    )
    assert target.is_file(), target
    original = target.read_bytes()
    try:
        target.write_bytes(original[: len(original) // 2] + original[len(original) // 2 :][::-1])
        with pytest.raises(AuthoringError):
            reader.read(left["receipt"]["receipt_hash"])
    finally:
        target.write_bytes(original)
    assert (
        reader.read(left["receipt"]["receipt_hash"]).receipt.receipt_hash
        == (left["receipt"]["receipt_hash"])
    )


def test_saved_alpha_comparison_refuses_mismatched_input_binding(
    real_risk_workspace: RealRiskWorkspace,
    alpha_authority: tuple[str, tuple[str, ...], Path],
    tmp_path: Path,
) -> None:
    """Stored metrics never bridge different declared Alpha inputs."""

    handle, factor_ids, evidence_root = alpha_authority
    document = _document(
        handle=handle,
        feature_ids=factor_ids[:8],
        model_parameters={"family": "ridge", "alpha": 1.0},
    )
    reader = AlphaDevelopmentReceiptReader(_ALPHA_STORE_ROOT(tmp_path))
    left, left_candidate_id, graph = _saved_alpha_side(
        real_risk_workspace, document, evidence_root=evidence_root, tmp_path=tmp_path, reader=reader
    )
    # The owner must refuse on the declared binding before any candidate or
    # score support could be treated as shared evidence.  Reuse the verified
    # graph so this is a boundary test, not a second numerical run.
    right = {**left, "task_id": str(uuid4()), "input_binding_hash": "b" * 64}
    with pytest.raises(AuthoringError, match="saved_comparison_input_binding_hash_mismatch"):
        compare_saved_alpha_candidates(
            left=left,
            left_candidate_id=left_candidate_id,
            left_graph=graph,
            right=right,
            right_candidate_id=left_candidate_id,
            right_graph=graph,
        )


def test_a_side_without_an_alpha_graph_is_reported_not_compared(
    real_risk_workspace: RealRiskWorkspace,
    alpha_authority: tuple[str, tuple[str, ...], Path],
    tmp_path: Path,
) -> None:
    """A Task that published no Alpha graph carries no graph: refused by name, never
    read from the store behind the comparison's back."""

    handle, factor_ids, evidence_root = alpha_authority
    document = _document(
        handle=handle,
        feature_ids=factor_ids[:8],
        model_parameters={"family": "ridge", "alpha": 1.0},
    )
    reader = AlphaDevelopmentReceiptReader(_ALPHA_STORE_ROOT(tmp_path))
    left, left_candidate_id, graph = _saved_alpha_side(
        real_risk_workspace, document, evidence_root=evidence_root, tmp_path=tmp_path, reader=reader
    )
    with pytest.raises(AuthoringError, match="saved_comparison_receipt_unavailable"):
        compare_saved_alpha_candidates(
            left=left,
            left_candidate_id=left_candidate_id,
            left_graph=graph,
            right={**left, "task_id": str(uuid4())},
            right_candidate_id=left_candidate_id,
            right_graph=None,
        )
