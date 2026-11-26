"""Execute one compiled Alpha development Program against real inputs.

The Alpha Desk already owned everything expensive: fold array construction from a
published Panel and published causal outcomes, a model catalog and capability
mandate, a fold-at-a-time batch executor, and a development artifact store. What
it did not own was a path from an *authored document* to any of it.

This module is that path. It calls the existing owners in the order production
calls them and adds no second implementation of fitting, scoring or metrics.

Three boundaries are load-bearing:

Development authority is not admission authority. The run is authorized by a
published Panel, published causal outcomes, and one Factor development
checkpoint, bound in ``AlphaDevelopmentFoundationBinding`` -- not by the frozen
``ResearchFoundationBinding``, whose four lineage fields name admission evidence a
development run does not have.

Artifacts are written to the document's own output workspace, never the source
workspace. The source workspace is read-only here and no lease is taken on it.

This module reaches neither the publication owner nor the Goal research package.
Development execution cannot qualify a candidate or move a current pointer, and
the import graph is where that is enforced rather than left to convention -- a
structural guard fails if either name appears anywhere in this package, including
in prose like this sentence, which is why neither is spelled out.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from alphalattice.capabilities.alpha_modeling.catalog import AlphaModelCatalog
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalOutcomeDevelopmentRows,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.investment.alpha_research.experiments.authoring import (
    ALPHA_EXPERIMENT_KIND,
    AlphaExperimentCompiler,
)
from alphalattice.investment.alpha_research.experiments.bindings import (
    build_alpha_development_program,
)
from alphalattice.investment.alpha_research.experiments.contracts import (
    AlphaDevelopmentProgram,
    AlphaDevelopmentSplitPolicy,
    AlphaExperimentBatch,
    AlphaExperimentBatchResult,
    AlphaSplitPolicy,
    seal_contract,
)
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.experiments.development_contracts import (
    AlphaDevelopmentChildLineage,
    AlphaDevelopmentExecutionReceipt,
    seal_current_contract,
)
from alphalattice.investment.alpha_research.experiments.development_evidence import (
    alpha_development_receipt_uri,
)
from alphalattice.investment.alpha_research.experiments.execution import (
    AlphaExperimentProgress,
    execute_alpha_model_batch,
)
from alphalattice.investment.alpha_research.experiments.fit_plan import (
    AlphaFitPlanAuthorityError,
    resolve_alpha_model_fit_protocol,
)
from alphalattice.investment.alpha_research.experiments.mandate import (
    AlphaDevelopmentModelMethodBinding,
    AlphaModelCapabilityAuthority,
    AlphaModelRecipeProposal,
    AlphaResearchModelRecipe,
    bind_admitted_model_to_target,
)
from alphalattice.investment.alpha_research.experiments.policies import (
    derive_alpha_development_split_policy,
)
from alphalattice.investment.alpha_research.inputs.development_foundation import (
    AlphaDevelopmentFoundationBinding,
)
from alphalattice.investment.alpha_research.inputs.folds import (
    AlphaCurrentRefitArrays,
    AlphaFoldArrayPlan,
    AlphaFoldArrays,
    prepare_alpha_fold_plan,
)
from alphalattice.investment.alpha_research.inputs.surfaces import (
    AlphaProgramArrayWorkspace,
    prepare_alpha_program_array_workspace,
)
from alphalattice.investment.alpha_research.targets.authority import (
    AlphaTargetMethod,
    AlphaTargetMethodBinding,
    AlphaTargetMethodCatalog,
    InstalledAlphaTargetMethodBinding,
    TotalReturnTargetMethod,
    WholeUniverseAlphaTargetMethodBinding,
)
from alphalattice.investment.alpha_research.targets.development import (
    AlphaDevelopmentTargetFoldRecord,
    AlphaDevelopmentTargetMaterializationBinding,
)
from alphalattice.kernel.quant.sector_history import SectorHistory
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskExecutionResult,
    NumericalCallRecorder,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)


def _array_identity(values: npt.NDArray[Any]) -> tuple[str, tuple[int, ...], str]:
    """dtype, shape and content hash of one array, as one inseparable statement.

    Bytes alone are ambiguous: the same buffer read as a different dtype or
    reshaped is a different array with the same digest. Carrying all three is
    what makes the hash a claim about values rather than about a buffer.
    """

    contiguous = np.ascontiguousarray(values)
    return (
        str(contiguous.dtype),
        tuple(int(value) for value in contiguous.shape),
        sha256(contiguous.tobytes()).hexdigest(),
    )


class _MaterializationCollector:
    """Verify and describe each fold on the one pass that actually fits it.

    This used to be a pre-pass that leased every fold, and then
    ``execute_alpha_model_batch`` leased them all again. In ``PROGRAM_CACHE`` mode
    the second walk is a cache hit; in ``BOUNDED_FOLD`` mode it re-reads the
    Feature and Outcome surfaces and re-runs the target transformation for every
    fold, so proving the identity cost as much as computing it.

    Decorating the workspace instead of walking it twice keeps the checks in the
    only window where both facts exist -- the declaration is fixed and the values
    are real -- while the arrays are handed to the fit exactly once. It implements
    ``AlphaArrayWorkspaceLike``, so ``execute_alpha_model_batch`` learns nothing
    about robust-z, recipe names or any Desk policy; it is handed something that
    leases folds, which is all it ever wanted.

    The consequence, stated rather than hidden: a mismatch on fold three is now
    raised after folds zero to two have been fitted. Nothing claims authority for
    them -- the run raises and no receipt is published -- and the alternative is
    materializing every fold twice.
    """

    def __init__(
        self,
        workspace: AlphaProgramArrayWorkspace,
        *,
        declared: tuple[str, ...],
    ) -> None:
        self._workspace = workspace
        self._declared = declared
        self._records: dict[int, AlphaDevelopmentTargetFoldRecord] = {}
        self._training_rows: dict[int, int] = {}

    @contextmanager
    def fold_lease(self, fold_index: int) -> Iterator[AlphaFoldArrays]:
        with self._workspace.fold_lease(fold_index) as fold:
            self._observe(fold_index, fold)
            yield fold

    def load_fold(self, fold_index: int) -> AlphaFoldArrays:
        fold = self._workspace.load_fold(fold_index)
        self._observe(fold_index, fold)
        return fold

    def prepare_current_refit(self) -> AlphaCurrentRefitArrays:
        return self._workspace.prepare_current_refit()

    def _observe(self, fold_index: int, fold: AlphaFoldArrays) -> None:
        # The estimator axis, in order. A permuted axis is a different matrix, so
        # equality here is exact rather than set-based.
        if fold.ordered_factor_ids != self._declared:
            raise AuthoringError("alpha_research.development_fold_axis_mismatch")
        if fold.training_features.shape[1] != len(self._declared):
            raise AuthoringError("alpha_research.development_training_width_mismatch")
        if fold.validation_features.shape[1] != len(self._declared):
            raise AuthoringError("alpha_research.development_validation_width_mismatch")
        binding = fold.training_input_binding
        if binding is not None and tuple(binding.ordered_feature_ids) != self._declared:
            raise AuthoringError("alpha_research.development_training_binding_axis_mismatch")
        training_dtype, training_shape, training_hash = _array_identity(fold.training_targets)
        validation_dtype, validation_shape, validation_hash = _array_identity(
            fold.validation_targets
        )
        record = AlphaDevelopmentTargetFoldRecord.create(
            fold_index=fold_index,
            fold_commitment_hash=fold.commitment.commitment_hash,
            training_dtype=training_dtype,
            training_shape=training_shape,
            training_value_hash=training_hash,
            validation_dtype=validation_dtype,
            validation_shape=validation_shape,
            validation_value_hash=validation_hash,
        )
        seen = self._records.get(fold_index)
        if seen is not None:
            # A bounded workspace rebuilds a fold on each lease. Rebuilding it to
            # different values would mean the surface is not a function of the
            # plan, which is a stronger failure than any hash mismatch.
            if seen.record_hash != record.record_hash:
                raise AuthoringError("alpha_research.development_fold_materialization_unstable")
            return
        self._records[fold_index] = record
        self._training_rows[fold_index] = int(fold.training_features.shape[0])

    def seal(
        self,
        *,
        recipe_binding: InstalledAlphaTargetMethodBinding,
        fold_count: int,
    ) -> AlphaDevelopmentTargetMaterializationBinding:
        """Describe what was actually fitted, once every fold has been."""

        if not self._records:
            raise AuthoringError("alpha_research.development_split_empty")
        if sorted(self._records) != list(range(fold_count)):
            # A binding covering some of the folds would describe a run nobody
            # made, and would still validate its own hash.
            raise AuthoringError("alpha_research.development_materialization_incomplete")
        records = tuple(self._records[index] for index in range(fold_count))
        return AlphaDevelopmentTargetMaterializationBinding.create(
            recipe_binding=recipe_binding,
            ordered_base_feature_ids=self._declared,
            fold_records=records,
            training_row_count=sum(self._training_rows.values()),
        )


def _child_lineage(
    *,
    store: AlphaDevelopmentArtifactStore,
    result: AlphaExperimentBatchResult,
    recipe_binding: InstalledAlphaTargetMethodBinding,
    materialization: AlphaDevelopmentTargetMaterializationBinding,
) -> tuple[AlphaDevelopmentChildLineage, ...]:
    """Join each fitted child to the target method that produced it.

    Read back from the store rather than assembled from local variables: the
    receipt's claim is about what was published, and an executor describing its
    own intentions would be exactly the self-consistency this closes.

    One entry per ``(candidate_id, fold_index)``. Parallel tuples of estimator
    and fit hashes would carry the same values and none of the correspondence.
    """

    entries: list[AlphaDevelopmentChildLineage] = []
    for candidate in result.candidates:
        for numerical_result_hash in candidate.numerical_result_hashes:
            numerical = store.load_candidate_numerical_fold_result(numerical_result_hash)
            if numerical.estimator_state_hash is None:
                raise AuthoringError("alpha_research.development_child_estimator_missing")
            state = store.load_development_estimator_state(numerical.estimator_state_hash)
            if state.fit_evidence_hash is None:
                raise AuthoringError("alpha_research.development_child_fit_evidence_missing")
            evidence = store.load_development_fit_evidence(state.fit_evidence_hash)
            entries.append(
                seal_current_contract(
                    AlphaDevelopmentChildLineage,
                    {
                        "kind": "AlphaDevelopmentChildLineage",
                        "candidate_id": numerical.candidate_id,
                        "fold_index": numerical.fold_index,
                        "fold_commitment_hash": numerical.fold_commitment_hash,
                        "numerical_result_hash": numerical.numerical_result_hash,
                        "estimator_state_hash": numerical.estimator_state_hash,
                        "fit_evidence_hash": state.fit_evidence_hash,
                        "training_input_binding_hash": evidence.training_binding_hash,
                        "estimator_content_hash": evidence.estimator_content_hash,
                        "fit_provenance_hash": evidence.fit_provenance_hash,
                        "score_evidence_hash": evidence.score_evidence_hash,
                        "ordered_factor_ids": state.ordered_factor_ids,
                        "target_recipe_binding_hash": recipe_binding.binding_hash,
                        "target_materialization_binding_hash": materialization.binding_hash,
                    },
                    "lineage_hash",
                )
            )
    # Batch order, deliberately not sorted: candidates as the result lists them,
    # each candidate's folds as its ``numerical_result_hashes`` lists them. A
    # reader rebuilds exactly this sequence from the published batch result and
    # compares it tuple to tuple, and re-sorting here would make that comparison
    # a set comparison in disguise -- which loses ordering and collapses repeats.
    return tuple(entries)


class AlphaDevelopmentCancelled(Exception):
    """Cooperative stop after a complete numerical child; no partial result claim."""


@dataclass(frozen=True)
class AlphaDevelopmentPreparation:
    """Call-scoped plan shared by preview and execution, never a durable runtime."""

    fold_plan: AlphaFoldArrayPlan
    method: AlphaTargetMethod
    recipe_binding: InstalledAlphaTargetMethodBinding
    model_recipe: AlphaResearchModelRecipe
    model_method_binding: AlphaDevelopmentModelMethodBinding
    split_policy: AlphaDevelopmentSplitPolicy
    development_program: AlphaDevelopmentProgram
    batch: AlphaExperimentBatch
    admitted_mandate: AlphaModelCapabilityAuthority
    """The mandate narrowed to the model this run admits, which the Program seals."""
    maturity_lag_sessions: int
    fit_protocol: str
    numerical_policy: dict[str, Any]

    @property
    def calls_per_fold(self) -> tuple[int, int, int]:
        """Fits, predictions and metrics one complete fold costs under the protocol.

        A direct plan fits once and predicts the training and validation
        surfaces; a nested plan fits the tuning partition, refits the whole
        training surface, and predicts the tuning validation surface as well.
        Either way one metric evaluation follows.
        """

        if self.fit_protocol == "NESTED_EARLY_STOPPING_REFIT":
            return 2, 3, 1
        return 1, 2, 1

    def describe(self) -> dict[str, Any]:
        windows = self.fold_plan.split_plan.windows
        sessions = [s for w in windows for s in w.validation_sessions]
        fits, predictions, metrics = self.calls_per_fold
        return {
            "model_configuration_count": 1,
            "model_adapter_id": self.model_recipe.recipe.adapter_id,
            "model_recipe_hash": self.model_recipe.recipe.recipe_hash,
            "model_parameters": dict(self.model_recipe.recipe.parameters),
            "fit_protocol": self.fit_protocol,
            "numerical_policy": dict(self.numerical_policy),
            "target_recipe_id": self.method.target_recipe_id,
            "target_method_hash": self.method.target_method_hash,
            "base_feature_ids": list(self.fold_plan.base_feature_ids),
            "additional_feature_ids": list(self.fold_plan.additional_factor_ids),
            "fold_count": len(windows),
            "folds": [
                {
                    "fold_index": i,
                    "train_start": str(w.train_sessions[0]),
                    "train_end": str(w.train_sessions[-1]),
                    "validation_start": str(w.validation_sessions[0]),
                    "validation_end": str(w.validation_sessions[-1]),
                }
                for i, w in enumerate(windows)
            ],
            "statistical_start": str(sessions[0]),
            "statistical_end": str(sessions[-1]),
            "statistical_session_count": len(sessions),
            "split_policy": self.split_policy.model_dump(mode="json"),
            "maturity_lag_sessions": self.maturity_lag_sessions,
            "fit_call_upper_bound": fits * len(windows),
            "predict_call_upper_bound": predictions * len(windows),
            "metric_call_upper_bound": metrics * len(windows),
            "expected_numerical_calls": (fits + predictions + metrics) * len(windows),
            "effective_model_threads": 1,
            "determinism": (
                "DETERMINISTIC_LINEAR_METHOD_NOT_A_SEED_ENSEMBLE"
                if self.model_recipe.recipe.adapter_id == "regularized_linear"
                else "DETERMINISTIC_SINGLE_RECIPE_SINGLE_SEED_NOT_AN_ENSEMBLE"
            ),
            "interval_semantics": (
                "Requested authority interval does not reslice policy-derived folds."
            ),
        }


@dataclass
class _FoldCancellation:
    requested: Callable[[], bool]

    def record_candidate_fold(self, **_facts: Any) -> None:
        if self.requested():
            raise AlphaDevelopmentCancelled("alpha_research.cancelled_at_fold_boundary")


class AlphaExperimentExecutor:
    """Adapt the Alpha development path to the Desk executor Protocol.

    Readers, foundation binding, catalogs and the split policy are constructor
    state supplied by Host composition. Assembling a foundation binding requires
    reading a published Panel, published causal outcomes and a Factor development
    checkpoint from three different roots, which is workspace knowledge; a Desk
    module that acquired it would stop being composable.
    """

    kind = ALPHA_EXPERIMENT_KIND

    def __init__(
        self,
        *,
        foundation: AlphaDevelopmentFoundationBinding,
        target_recipes: AlphaTargetMethodCatalog,
        model_mandate: AlphaModelCapabilityAuthority,
        model_catalog: AlphaModelCatalog,
        feature_reader: FeaturePanelReader,
        outcome_reader: CausalOutcomeDevelopmentRows,
        feature_panel_manifest_ref: str,
        causal_outcome_manifest_ref: str,
        ordered_listing_ids: tuple[str, ...],
        sector_by_listing_id: Mapping[str, str],
        split_policy: AlphaSplitPolicy,
        frozen_at: datetime,
        panel_factor_ids: tuple[str, ...],
    ) -> None:
        self._operational_progress: AlphaExperimentProgress | None = None
        self._foundation = foundation
        self._target_recipes = target_recipes
        self._model_mandate = model_mandate
        self._model_catalog = model_catalog
        self._feature_reader = feature_reader
        self._outcome_reader = outcome_reader
        self._feature_panel_manifest_ref = feature_panel_manifest_ref
        self._causal_outcome_manifest_ref = causal_outcome_manifest_ref
        self._ordered_listing_ids = ordered_listing_ids
        # A history is immutable and read per session; a plain map is copied.
        self._sector_by_listing_id: Mapping[str, str] = (
            sector_by_listing_id
            if isinstance(sector_by_listing_id, SectorHistory)
            else dict(sector_by_listing_id)
        )
        self._split_policy = split_policy
        self._frozen_at = frozen_at
        self._compiler = AlphaExperimentCompiler(
            target_recipes=target_recipes,
            model_mandate=model_mandate,
            model_catalog=model_catalog,
            panel_factor_ids=panel_factor_ids,
            # The *selected* axis, not the full statistical universe. A factor the
            # deterministic program corrected over but this run never reported on
            # has no development standing here, and gating on the wider axis
            # would let a document draw features on the strength of a number
            # computed about a different question.
            factor_evidence_factor_ids=(
                foundation.research_foundation.ordered_factor_ids
                if foundation.research_foundation is not None
                else foundation.selected_factor_ids
            ),
            factor_evidence_checkpoint_hash=foundation.factor_development_checkpoint_hash,
            research_foundation_hash=(
                foundation.research_foundation.foundation_hash
                if foundation.research_foundation is not None
                else None
            ),
        )

    def prepare_execution(
        self,
        *,
        program: SealedResearchProgram,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
    ) -> AlphaDevelopmentPreparation:
        envelope_document = document.get("experiment")
        if not isinstance(envelope_document, dict):
            raise AuthoringError("research_authoring.experiment_section_missing")
        envelope = ResearchExperimentEnvelope.create(**envelope_document)
        compiled = self._compiler.compile_desk_program(
            envelope=envelope,
            document=document,
            authority=authority,
        )
        if compiled.desk_program_hash != program.desk_program_hash:
            # The sealed Program and the document must still agree, or the run
            # produces evidence for identity nobody admitted.
            raise AuthoringError("research_authoring.replay_identity_mismatch")
        if authority.panel_snapshot_hash != self._foundation.feature_panel_snapshot_hash:
            raise AuthoringError("alpha_research.development_panel_not_this_authority")

        section = document["alpha"]
        method = self._target_recipes.resolve(str(section["target_recipe_id"]))
        ordered_feature_ids = tuple(str(value) for value in section["ordered_feature_ids"])
        # The declaration, fixed from the recipe alone before anything is
        # compiled. The lane cannot express it: ``policy.standardization_id`` is
        # derived from a closed enum and would keep saying rank-gauss.
        outcome_seal = self._outcome_reader.resolve_method_seal(
            self._foundation.execution_outcome.snapshot_hash
        )
        if outcome_seal.disposition != "METHOD_BOUND":
            # Development evidence is built on method-bound outcomes or not at
            # all. A legacy snapshot stays readable; it does not become the
            # source of new science.
            raise AuthoringError("alpha_research.development_outcome_method_unbound")
        declared_outcome = section.get("execution_outcome_recipe_id")
        if (
            declared_outcome is not None
            and str(declared_outcome) != outcome_seal.method_bound.recipe_id
        ):
            # A document may state which outcome method it believes it is
            # studying, and the Host refuses a mismatch rather than silently
            # running the other one. It may not *select* the method: which
            # snapshot this run reads is resolved from evidence, so a document
            # that could choose would be choosing what its target transforms.
            raise AuthoringError("alpha_research.development_outcome_method_not_declared")
        if isinstance(method, TotalReturnTargetMethod):
            recipe_binding: InstalledAlphaTargetMethodBinding = (
                WholeUniverseAlphaTargetMethodBinding.create(
                    method=method,
                    listing_set_hash=self._foundation.execution_outcome.listing_set_hash,
                    execution_outcome_recipe_id=outcome_seal.method_bound.recipe_id,
                    causal_outcome_snapshot_hash=(self._foundation.execution_outcome.snapshot_hash),
                    outcome_method_binding_hash=outcome_seal.method_bound.binding_hash,
                    maturity_lag_sessions=(outcome_seal.method_bound.maturity_lag_sessions),
                )
            )
        else:
            recipe_binding = AlphaTargetMethodBinding.create(
                method=method,
                execution_outcome_recipe_id=outcome_seal.method_bound.recipe_id,
                causal_outcome_snapshot_hash=(self._foundation.execution_outcome.snapshot_hash),
                outcome_method_binding_hash=outcome_seal.method_bound.binding_hash,
            )
        if recipe_binding.standardization_id != method.standardization_id:
            raise AuthoringError("alpha_research.development_target_recipe_not_installed")

        # The embargo comes from the method that produced the labels, not from
        # the caller and not from a recipe-id branch. A formation whose outcome
        # is still open where a fold boundary falls would otherwise train on a
        # label that partly describes validation sessions, and the frozen policy
        # says zero because the lanes it was written for are Goal and current.
        maturity_lag_sessions = outcome_seal.method_bound.maturity_lag_sessions
        available_sessions = self._feature_reader.available_sessions(
            self._feature_panel_manifest_ref
        )
        split_policy = derive_alpha_development_split_policy(
            geometry=self._split_policy,
            maturity_lag_sessions=maturity_lag_sessions,
            session_count=len(available_sessions),
        )

        # The plan carries two things the document declared and the numerical
        # path used to ignore: the standardization the recipe *named*, so the
        # surface compiles the selected method rather than the lane's derived
        # default, and the base feature axis, so the arrays are built over the
        # factors the document actually asked for. Both were previously recorded
        # in identity and then not used, which is the shape of defect where every
        # hash agrees and the numbers came from somewhere else.
        fold_plan = prepare_alpha_fold_plan(
            foundation=self._foundation.research_foundation or self._foundation,
            ordered_listing_ids=self._ordered_listing_ids,
            feature_reader=self._feature_reader,
            outcome_reader=self._outcome_reader,
            feature_panel_manifest_ref=self._feature_panel_manifest_ref,
            causal_outcome_manifest_ref=self._causal_outcome_manifest_ref,
            frozen_at=self._frozen_at,
            split_policy=split_policy,
            # The whole resolved recipe, plus the method authority the outcome
            # snapshot actually carries. Passing the recipe's parts separately
            # let the plan describe a method nobody had resolved as a unit.
            target_method=method,
            outcome_method=outcome_seal,
            sector_by_listing_id=self._sector_by_listing_id,
            ordered_base_feature_ids=ordered_feature_ids,
            # The calendar read above for the split policy's session count.
            panel_sessions=available_sessions,
        )
        if not fold_plan.split_plan.windows:
            raise AuthoringError("alpha_research.development_split_empty")
        if fold_plan.base_feature_ids != ordered_feature_ids:
            raise AuthoringError("alpha_research.development_feature_axis_not_installed")

        lanes = method.admitted_model_lanes
        admitted_recipe = self._model_mandate.admit_proposal(
            proposal=AlphaModelRecipeProposal(
                capability_handle=str(section["model_capability_handle"]),
                parameters=dict(section["model_parameters"]),
                target_lane=None if lanes is None else lanes[0],
            ),
            catalog=self._model_catalog,
            admitted_target_lanes=lanes,
        )
        # The model bound to the target it will actually be fitted to, not merely
        # admitted beside it. Without this the two arms of the paired study --
        # same estimator, same parameters, two target compositions with no lane --
        # seal the same ``research_recipe_hash``, and ``candidate_id`` derives
        # from exactly that, so nothing downstream could tell the arms apart.
        model_recipe, model_method_binding = bind_admitted_model_to_target(
            admitted=admitted_recipe,
            target_recipe_id=method.target_recipe_id,
            target_method_hash=method.target_method_hash,
            target_recipe_binding_hash=recipe_binding.binding_hash,
            outcome_method_binding_hash=outcome_seal.method_bound.binding_hash,
        )
        # The Program seals the model it runs, never the other installed ones.
        admitted_mandate = self._model_mandate.admitting(admitted_recipe.search_domain_hash)
        development_program = build_alpha_development_program(
            fold_plan=fold_plan,
            model_mandate=admitted_mandate,
        )
        batch = seal_contract(
            AlphaExperimentBatch,
            {
                "program_hash": development_program.program_hash,
                "batch_index": 1,
                "specs": (model_recipe,),
                "predecessor_batch_hash": None,
            },
            "batch_hash",
        )
        # The admitted adapter states the protocol this recipe fits under, and
        # the protocol states what one complete fold costs. A recipe whose
        # protocol the development split cannot plan -- an early-stopping policy
        # whose domain states no tuning partition -- is refused here, before
        # any array is built. The bound is checked before arrays/fit.
        domain = self._model_mandate.resolve_capability_handle(
            str(section["model_capability_handle"])
        )
        adapter = self._model_catalog.admit_recipe(recipe=model_recipe.recipe, domain=domain)
        try:
            fit_protocol = resolve_alpha_model_fit_protocol(
                adapter=adapter, recipe=model_recipe.recipe, domain=domain
            )
        except AlphaFitPlanAuthorityError as error:
            raise AuthoringError(
                "alpha_research.development_training_policy_not_installed"
            ) from error
        prepared = AlphaDevelopmentPreparation(
            fold_plan,
            method,
            recipe_binding,
            model_recipe,
            model_method_binding,
            split_policy,
            development_program,
            batch,
            admitted_mandate,
            maturity_lag_sessions,
            fit_protocol,
            # JSON-native, as the sealed plan re-reads it after a restart.
            dict(
                adapter.describe_numerical_binding().model_dump(mode="json")["deterministic_policy"]
            ),
        )
        if sum(prepared.calls_per_fold) * fold_plan.fold_count > (
            envelope.budget.maximum_numerical_calls
        ):
            raise AuthoringError("alpha_research.development_work_budget_exceeded")
        return prepared

    def set_cancellation_check(self, requested: Callable[[], bool]) -> None:
        self._operational_progress = _FoldCancellation(requested)

    def execute(
        self,
        *,
        program: SealedResearchProgram,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
        output_workspace: Path,
        recorder: NumericalCallRecorder | None = None,
    ) -> DeskExecutionResult:
        prepared = self.prepare_execution(program=program, document=document, authority=authority)
        fold_plan, method = prepared.fold_plan, prepared.method
        recipe_binding, model_recipe = prepared.recipe_binding, prepared.model_recipe
        model_method_binding = prepared.model_method_binding
        split_policy, development_program, batch = (
            prepared.split_policy,
            prepared.development_program,
            prepared.batch,
        )
        maturity_lag_sessions = prepared.maturity_lag_sessions
        ordered_feature_ids = fold_plan.base_feature_ids
        store = AlphaDevelopmentArtifactStore(Path(output_workspace) / "alpha-development")
        with prepare_alpha_program_array_workspace(
            fold_plan, train_session_count=None
        ) as array_workspace:
            # The batch is handed a decorated workspace, so every fold is checked
            # and described on the single pass that fits it. Verifying it
            # beforehand meant materializing the whole split twice, and under a
            # bounded budget that is a second full read and a second target
            # transformation per fold.
            collector = _MaterializationCollector(
                array_workspace, declared=fold_plan.base_feature_ids
            )
            result = execute_alpha_model_batch(
                program=development_program,
                batch=batch,
                fold_plan=fold_plan,
                store=store,
                array_workspace=collector,
                model_catalog=self._model_catalog,
                model_mandate=prepared.admitted_mandate,
                operational_progress=self._operational_progress,
            )
            materialization = collector.seal(
                recipe_binding=recipe_binding, fold_count=fold_plan.fold_count
            )

        # What this run was actually allowed to read and which method read it.
        # Opaque to the generic layer by design: every Desk seals something
        # describing its inputs, and what belongs in that seal is methodology.
        input_binding_hash = str(
            canonical_hash(
                {
                    "kind": "AlphaDevelopmentInputBinding",
                    "foundation_hash": self._foundation.foundation_hash,
                    "panel_snapshot_hash": authority.panel_snapshot_hash,
                    "panel_manifest_ref": self._feature_panel_manifest_ref,
                    "causal_outcome_manifest_ref": self._causal_outcome_manifest_ref,
                    "factor_development_checkpoint_hash": (
                        self._foundation.factor_development_checkpoint_hash
                    ),
                    "ordered_feature_ids": list(ordered_feature_ids),
                    "target_recipe_id": method.target_recipe_id,
                    "target_method_hash": method.target_method_hash,
                    "standardization_id": method.standardization_id,
                    # The clock the embargo was derived from, stated rather than
                    # left to be inferred from the split geometry. Two methods
                    # with the same geometry and different spans are different
                    # science, and only one of them is what this run read.
                    "outcome_method_binding_hash": model_method_binding.outcome_method_binding_hash,
                    "maturity_lag_sessions": maturity_lag_sessions,
                    "split_policy_hash": split_policy.policy_hash,
                    "model_method_binding_hash": model_method_binding.binding_hash,
                    # Both target layers, so a reader sees the declared recipe
                    # *and* the values it actually produced, rather than a lane
                    # that cannot name either.
                    "target_recipe_binding_hash": recipe_binding.binding_hash,
                    "target_materialization_binding_hash": materialization.binding_hash,
                    "transformed_target_value_hash": (
                        materialization.transformed_target_value_hash
                    ),
                    "research_recipe_hash": model_recipe.research_recipe_hash,
                    "authority_hash": authority.authority_hash,
                }
            )
        )
        # Sealed before the receipt that names it, so ``batch_result_hash`` is a
        # resolvable claim rather than a value nobody can look up. It is also the
        # executed child axis: a reader rebuilds the expected lineage from it and
        # refuses a receipt that quietly dropped one.
        store.publish_batch_result(result)
        # The development authority parent. Both target bindings used to end here
        # as inputs to one opaque hash and then go out of scope, so nothing
        # durable said which target method ran or which children came from it --
        # and the children themselves record only the lane.
        receipt = AlphaDevelopmentExecutionReceipt.create(
            program_hash=program.program_hash,
            desk_program_hash=program.desk_program_hash,
            method_binding_hash=program.method_binding_hash,
            desk_input_binding_hash=input_binding_hash,
            target_recipe_binding=recipe_binding,
            target_materialization_binding=materialization,
            ordered_base_feature_ids=ordered_feature_ids,
            research_recipe_hash=model_recipe.research_recipe_hash,
            model_adapter_id=model_recipe.recipe.adapter_id,
            model_adapter_recipe_hash=model_recipe.recipe.recipe_hash,
            development_program_hash=development_program.program_hash,
            metric_policy_hash=development_program.metric_policy_hash,
            split_policy=split_policy,
            model_method_binding_hash=model_method_binding.binding_hash,
            batch_hash=batch.batch_hash,
            batch_result_hash=result.result_hash,
            development_surface_hash=result.development_surface_hash,
            development_surface_binding_hash=result.development_surface_binding_hash,
            fold_surface_hashes=result.fold_surface_hashes,
            child_lineage=_child_lineage(
                store=store,
                result=result,
                recipe_binding=recipe_binding,
                materialization=materialization,
            ),
        )
        store.publish_development_execution_receipt(receipt)
        formation_sessions = tuple(
            session
            for window in fold_plan.split_plan.windows
            for session in window.validation_sessions
        )
        if recorder is not None:
            recorder.record(capability=development_program.program_hash)
        return DeskExecutionResult(
            disposition="COMPUTED",
            # The receipt, and only the receipt. The batch result was previously
            # named by a hand-built URI no store resolves, and a raw child cannot
            # answer for the target method that produced it.
            artifact_uris=(alpha_development_receipt_uri(receipt.receipt_hash),),
            formation_sessions=formation_sessions,
            # Fit plus predict plus metric, as the batch itself counted them.
            # Reported rather than recomputed: the executor that made the calls
            # is the only owner that can answer how many it made.
            numerical_call_count=(
                result.fit_call_count + result.predict_call_count + result.metric_call_count
            ),
            desk_input_binding_hash=input_binding_hash,
        )

    @property
    def compiler(self) -> AlphaExperimentCompiler:
        """The compiler this executor validates against, for Host installation."""

        return self._compiler

    @property
    def foundation(self) -> AlphaDevelopmentFoundationBinding:
        """The verified call-scoped input binding, shared with preparation readback."""
        return self._foundation


__all__ = ["AlphaExperimentExecutor"]
