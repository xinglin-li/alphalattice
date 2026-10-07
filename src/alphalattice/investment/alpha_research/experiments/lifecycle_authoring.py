"""The Alpha lifecycle method through the existing researcher authoring workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
    DYNAMIC_PANEL_LIGHTGBM_SEEDS,
)
from alphalattice.control.workspace_runtime.content_store import verified_model_read_scope
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.experiments.development_execution import (
    AlphaDevelopmentCancelled,
)
from alphalattice.investment.alpha_research.scores.frozen_inference import ComponentFeatureHistory
from alphalattice.investment.alpha_research.scores.model_renewal import (
    AdmittedRenewingInference,
    AlphaModelLifecycleAdmission,
    AlphaModelSetPublication,
    AlphaPreparedRefit,
    AlphaTrainingObservations,
    copy_training_observations,
    prepare_alpha_refit,
    publish_lifecycle_projection,
    read_lifecycle_projection,
    read_training_prices,
    reuse_refit_child,
    verified_lifecycle_admissions,
    verified_prepared_values,
    verify_model_set,
)
from alphalattice.investment.alpha_research.scores.product_lifecycle import (
    AlphaModelLifecycleRecipe,
    ResolvedAlphaRefitPlan,
    _LifecycleContract,
    lifecycle_implementation_hash,
    resolve_alpha_refit_plan,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskExecutionResult,
    DeskProgramCompilation,
    DeskSection,
    NumericalCallRecorder,
    ResearchExecutionEvidence,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    SealedResearchProgram,
    refuse_unknown_section_keys,
)

METHOD = "MODEL_LIFECYCLE_REPLAY"
CATEGORY = "lifecycle-research-executions"
EARLIER_SCHEME_REFUSAL = "alpha_research.lifecycle_program_scheme_superseded"


def lifecycle_method_binding(component_recipe_hash: str, lifecycle_hash: str) -> str:
    """What a lifecycle Program binds: the component recipe and the lifecycle rule.

    The code that runs them is bound by the study plan's implementation hash, whose moves
    are recorded (binding plan, P), so an edit that leaves the numbers alone moves no
    Program.
    """

    return str(canonical_hash([component_recipe_hash, lifecycle_hash]))


def _earlier_method_binding(
    component_recipe_hash: str, lifecycle_hash: str, implementation_hash: str
) -> str:
    """A Program sealed before P also bound the implementation; it reads back as historical."""

    return str(canonical_hash([component_recipe_hash, lifecycle_hash, implementation_hash]))


class AlphaLifecycleSection(DeskSection):
    """A lifecycle replay's `alpha` section: the installed component and its refit rule."""

    methodology_id: str
    """This method's id, `MODEL_LIFECYCLE_REPLAY`."""
    component_recipe_id: str
    """The installed component the lifecycle refits."""
    lifecycle: dict[str, Any]
    """The refit rule: the fields of `AlphaModelLifecycleRecipe`, each admitted by the method."""


def lifecycle_section(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    section = document.get("alpha")
    return (
        section
        if isinstance(section, Mapping) and section.get("methodology_id") == METHOD
        else None
    )


class AlphaLifecyclePreflight(_LifecycleContract):
    lifecycle: AlphaModelLifecycleRecipe
    required_vintages: tuple[str, ...]
    fit_upper_bound: int
    prediction_call_upper_bound: int


class AlphaLifecycleResearchReceipt(_LifecycleContract):
    program_hash: str
    authority_hash: str
    method_binding_hash: str
    component_recipe_hash: str
    implementation_hash: str
    lifecycle: AlphaModelLifecycleRecipe
    model_sets: tuple[str, ...]
    projections: tuple[str, ...]
    projection_files: tuple[str, ...]
    formation_sessions: tuple[str, ...]
    fit_call_count: int | None = Field(default=None, ge=0, exclude_if=lambda v: v is None)
    prediction_call_count: int | None = Field(default=None, ge=0, exclude_if=lambda v: v is None)


class AlphaLifecycleExperiment:
    """Compile and execute one installed method; component choice is declaration data."""

    kind = "alpha.model-development"

    def __init__(
        self,
        store: AlphaDevelopmentArtifactStore,
        admission: AlphaModelLifecycleAdmission,
        cancelled: Callable[[], bool] = lambda: False,
        *,
        packed_capacity: Callable[[int], None] = lambda _bytes: None,
        sharing_workspace: Path | None = None,
    ):
        self.store, self.admission = store, admission
        self.cancelled = cancelled
        self.packed_capacity = packed_capacity
        self.sharing_workspace = sharing_workspace
        self.preflight: AlphaLifecyclePreflight | None = None

    @property
    def compiler(self) -> AlphaLifecycleExperiment:
        return self

    def validate_declaration(self, document: Mapping[str, Any]) -> AlphaModelLifecycleRecipe:
        section = lifecycle_section(document)
        if section is None:
            raise AuthoringError("alpha_research.lifecycle_declaration_invalid")
        refuse_unknown_section_keys(section, AlphaLifecycleSection, place="alpha")
        if section.get("component_recipe_id") != self.admission.component.component_id:
            raise AuthoringError("alpha_research.lifecycle_component_not_installed")
        selected = section.get("lifecycle", {})
        if not isinstance(selected, Mapping) or set(selected) - (
            set(AlphaModelLifecycleRecipe.model_fields) - {"content_hash"}
        ):
            raise AuthoringError("alpha_research.lifecycle_parameter_not_installed")
        values = {**self.admission.lifecycle.model_dump(exclude={"content_hash"}), **selected}
        for field in ("seeds", "vintage_weights"):
            if not isinstance(values[field], (list, tuple)):
                raise AuthoringError("alpha_research.lifecycle_parameter_not_installed")
            values[field] = tuple(values[field])
        rule = AlphaModelLifecycleRecipe.create(**values)
        if not set(rule.seeds) <= set(DYNAMIC_PANEL_LIGHTGBM_SEEDS):
            raise AuthoringError("alpha_research.lifecycle_seed_not_installed")
        return rule

    def _plans(
        self, rule: AlphaModelLifecycleRecipe, authority: ResolvedResearchAuthority
    ) -> tuple[ResolvedAlphaRefitPlan, ...]:
        observations = self.store._load(
            "lifecycle-training-observations",
            self.admission.observations_hash,
            "content_hash",
            AlphaTrainingObservations,
        )
        source = read_training_prices(self.store, observations.observation_hash)
        if authority.training_snapshot_hash != observations.content_hash:
            raise AuthoringError("alpha_research.lifecycle_source_authority_mismatch")
        if authority.ordered_listing_ids != source.ordered_listing_ids or not set(
            authority.sessions
        ) <= set(source.formation_sessions):
            raise AuthoringError("alpha_research.lifecycle_source_authority_mismatch")
        retained = (
            {p.plan.vintage: p.plan for p in self.admission.prepared}
            if rule == self.admission.lifecycle
            else {}
        )
        return tuple(
            retained[vintage]
            if vintage in retained
            else resolve_alpha_refit_plan(
                lifecycle=rule,
                vintage=vintage,
                sessions=source.formation_sessions,
                component_recipe_hash=self.admission.component.recipe_hash,
                source_binding_hash=observations.content_hash,
                ordered_listing_ids=source.ordered_listing_ids,
                ordered_feature_ids=self.admission.component.ordered_feature_ids,
            )
            for vintage in sorted({v for day in authority.sessions for v in rule.vintages(day)})
        )

    def compile_desk_program(
        self,
        *,
        envelope: ResearchExperimentEnvelope,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
    ) -> DeskProgramCompilation:
        rule = self.validate_declaration(document)
        plans = self._plans(rule, authority)
        initial = (
            {(c.vintage, c.seed) for c in self.admission.initial_children}
            if rule == self.admission.lifecycle
            else set()
        )
        fits = sum((p.vintage, seed) not in initial for p in plans for seed in rule.seeds)
        if any(
            p.vintage not in self.admission.fit_vintages
            for p in plans
            for seed in rule.seeds
            if (p.vintage, seed) not in initial
        ):
            raise AuthoringError("alpha_research.lifecycle_refit_period_not_admitted")
        self.preflight = AlphaLifecyclePreflight.create(
            lifecycle=rule,
            required_vintages=tuple(p.vintage for p in plans),
            fit_upper_bound=fits,
            prediction_call_upper_bound=len(authority.sessions)
            * len(rule.seeds)
            * rule.vintage_count,
        )
        if (
            fits > self.admission.maximum_fit_attempts
            or fits * 2 + len(authority.sessions) * len(rule.seeds) * rule.vintage_count
            > envelope.budget.maximum_numerical_calls
        ):
            raise AuthoringError("alpha_research.lifecycle_research_budget_exceeded")
        method = lifecycle_method_binding(self.admission.component.recipe_hash, rule.content_hash)
        # The envelope is bound once, by the sealed Program itself (B6).
        return DeskProgramCompilation(
            desk_program_hash=canonical_hash(
                [
                    authority.authority_hash,
                    method,
                    self.admission.content_hash,
                    [p.content_hash for p in plans],
                ]
            ),
            catalog_hash=canonical_hash([self.admission.content_hash, METHOD]),
            method_binding_hash=method,
            # The seeds the rule draws, not the installed seeds or the rule's schema: a
            # contract docstring or an installed seed decides none of its numbers (V91).
            parameter_domain_hash=canonical_hash({"seeds": list(rule.seeds)}),
        )

    @verified_model_read_scope(reuse_verified=True)
    @verified_lifecycle_admissions()
    def execute(
        self,
        *,
        program: SealedResearchProgram,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
        output_workspace: Path,
        recorder: NumericalCallRecorder | None = None,
    ) -> DeskExecutionResult:
        envelope = ResearchExperimentEnvelope.create(**document["experiment"])
        compilation = self.compile_desk_program(
            envelope=envelope, document=document, authority=authority
        )
        if (
            compilation.desk_program_hash != program.desk_program_hash
            or program.authority_hash != authority.authority_hash
        ):
            raise AuthoringError("alpha_research.lifecycle_sealed_program_mismatch")
        rule = self.validate_declaration(document)
        store = AlphaDevelopmentArtifactStore(
            output_workspace / "alpha-lifecycle",
            packed_capacity=self.packed_capacity,
            sharing_workspace=self.sharing_workspace,
        )
        observations = self.store._load(
            "lifecycle-training-observations",
            self.admission.observations_hash,
            "content_hash",
            AlphaTrainingObservations,
        )
        original = copy_training_observations(self.store, store, observations)
        copied_sources = {observations.content_hash}
        prepared_values = []
        retained = {item.plan.content_hash: item for item in self.admission.prepared}
        for plan in self._plans(rule, authority):
            if self.cancelled():
                raise AlphaDevelopmentCancelled("alpha_research.lifecycle_cancelled_at_checkpoint")
            period_source = self.store._load(
                "lifecycle-training-observations",
                plan.source_binding_hash,
                "content_hash",
                AlphaTrainingObservations,
            )
            if period_source.content_hash not in copied_sources:
                copy_training_observations(self.store, store, period_source)
                copied_sources.add(period_source.content_hash)
            item = retained.get(plan.content_hash)
            if item is not None:
                if item.plan != plan or item.observations_hash != period_source.content_hash:
                    raise AuthoringError("alpha_research.lifecycle_prepared_input_changed")
                verified_prepared_values(self.store, item)
                store.import_packed(
                    self.store, category="lifecycle-arrays", content_hash=item.array_file_hash
                )
                store._publish("lifecycle-prepared-refits", item, "content_hash")
            else:
                item = prepare_alpha_refit(
                    store,
                    plan=plan,
                    observations=period_source,
                    component=self.admission.component,
                )
            prepared_values.append(item)
        prepared: tuple[AlphaPreparedRefit, ...] = tuple(prepared_values)
        initial = self.admission.initial_children if rule == self.admission.lifecycle else ()
        for item in prepared:
            for seed in rule.seeds:
                reuse_refit_child(
                    self.store, store, prepared=item, component=self.admission.component, seed=seed
                )
        initial = tuple(c for c in initial if c.prepared_hash in {p.content_hash for p in prepared})
        for child in initial:
            store._publish_packed_bytes(
                category="current/imported-models",
                payload=self.store._frozen_payload("imported-models", child.payload_hash),
            )
        admitted = AlphaModelLifecycleAdmission.create(
            component=self.admission.component,
            lifecycle=rule,
            observations_hash=observations.content_hash,
            prepared=prepared,
            initial_children=initial,
            formation_start=authority.sessions[0],
            formation_end=authority.sessions[-1],
            maximum_fit_attempts=self.admission.maximum_fit_attempts,
            environment_hash=self.admission.environment_hash,
            fit_vintages=tuple(
                v for v in self.admission.fit_vintages if v in {p.plan.vintage for p in prepared}
            ),
        )
        model_sets, projections, projection_files = [], [], []
        calls = 0
        fits = predictions = 0
        history = ComponentFeatureHistory(
            original,
            feature_ids=self.admission.component.ordered_feature_ids,
            through=authority.sessions[-1],
        )
        source_positions = {day: i for i, day in enumerate(original.formation_sessions)}
        for day in authority.sessions:
            if self.cancelled():
                raise AlphaDevelopmentCancelled("alpha_research.lifecycle_cancelled_at_checkpoint")
            position = source_positions[day]
            inference = AdmittedRenewingInference(store.root.parent, admitted)
            projection = inference.score(
                formation=day,
                listing_ids=original.ordered_listing_ids,
                eligible=(
                    np.ones(len(original.ordered_listing_ids), dtype=np.bool_)
                    if original.reference_eligible is None
                    else original.reference_eligible[position]
                ),
                surfaces=inference.features(original, day, history=history),
                raw_12_1_momentum=(
                    original.formula_values["mom_252_21"][position]
                    if "mom_252_21" in original.formula_values
                    else None
                ),
            )
            assert inference.model_set_publication_hash is not None
            model_sets.append(inference.model_set_publication_hash)
            projections.append(projection.projection_hash)
            projection_files.append(publish_lifecycle_projection(store, projection))
            calls += inference.prediction_owner.predictions + inference.fit_numerical_calls
            fits += inference.fit_calls
            predictions += inference.prediction_owner.predictions
        for _ in range(calls):
            if recorder is not None:
                recorder.record(capability="alpha_model.lifecycle")
        receipt = AlphaLifecycleResearchReceipt.create(
            program_hash=program.program_hash,
            authority_hash=authority.authority_hash,
            method_binding_hash=program.method_binding_hash,
            component_recipe_hash=self.admission.component.recipe_hash,
            implementation_hash=lifecycle_implementation_hash(),
            lifecycle=rule,
            model_sets=tuple(model_sets),
            projections=tuple(projections),
            projection_files=tuple(projection_files),
            formation_sessions=tuple(d.isoformat() for d in authority.sessions),
            fit_call_count=fits,
            prediction_call_count=predictions,
        )
        uri = store._publish(CATEGORY, receipt, "content_hash")
        return DeskExecutionResult(
            disposition="COMPUTED",
            artifact_uris=(uri,),
            formation_sessions=authority.sessions,
            numerical_call_count=calls,
            desk_input_binding_hash=authority.authority_hash,
        )


def read_lifecycle_research_receipt(
    *,
    program: SealedResearchProgram,
    evidence: ResearchExecutionEvidence,
    output_workspace: Path,
) -> AlphaLifecycleResearchReceipt:
    """Bind saved execution metadata; this alone does not verify its numeric children."""
    store = AlphaDevelopmentArtifactStore(output_workspace / "alpha-lifecycle")
    if len(evidence.artifact_uris) != 1:
        raise AuthoringError("alpha_research.lifecycle_receipt_not_unique")
    identity = store._hash_from_uri(evidence.artifact_uris[0], "current/" + CATEGORY)
    receipt = store._load(CATEGORY, identity, "content_hash", AlphaLifecycleResearchReceipt)
    if (
        receipt.program_hash != program.program_hash
        or receipt.authority_hash != program.authority_hash
        or receipt.method_binding_hash != program.method_binding_hash
        or lifecycle_program_scheme(program, receipt) is None
        or evidence.desk_input_binding_hash != program.authority_hash
        or receipt.formation_sessions != tuple(d.isoformat() for d in evidence.formation_sessions)
        or len(receipt.model_sets) != len(receipt.formation_sessions)
        or len(receipt.projection_files) != len(receipt.model_sets)
        or len(receipt.projections) != len(receipt.model_sets)
    ):
        raise AuthoringError("alpha_research.lifecycle_receipt_binding_invalid")
    return receipt


def lifecycle_program_scheme(
    program: SealedResearchProgram, receipt: AlphaLifecycleResearchReceipt
) -> str | None:
    """`CURRENT`, `EARLIER` (sealed before P, historical), or None when neither binds."""

    recipe, rule = receipt.component_recipe_hash, receipt.lifecycle.content_hash
    if program.method_binding_hash == lifecycle_method_binding(recipe, rule):
        return "CURRENT"
    if program.method_binding_hash == _earlier_method_binding(
        recipe, rule, receipt.implementation_hash
    ):
        return "EARLIER"
    return None


@verified_model_read_scope(reuse_verified=True)
@verified_lifecycle_admissions()
def verify_lifecycle_research(
    *,
    program: SealedResearchProgram,
    evidence: ResearchExecutionEvidence,
    output_workspace: Path,
) -> str:
    """Verify every numerical child; answer the Program's scheme (`CURRENT` or `EARLIER`)."""

    receipt = read_lifecycle_research_receipt(
        program=program, evidence=evidence, output_workspace=output_workspace
    )
    store = AlphaDevelopmentArtifactStore(output_workspace / "alpha-lifecycle")
    for day, identity, projection_hash, array_hash in zip(
        receipt.formation_sessions,
        receipt.model_sets,
        receipt.projections,
        receipt.projection_files,
        strict=True,
    ):
        value = store._load(
            "lifecycle-model-sets", identity, "content_hash", AlphaModelSetPublication
        )
        projection = read_lifecycle_projection(store, array_hash)
        if (
            value.lifecycle != receipt.lifecycle
            or value.component.recipe_hash != receipt.component_recipe_hash
            or projection.formation_session.isoformat() != day
            or projection.projection_hash != projection_hash
            or projection.evidence_manifest_sha256 != value.content_hash
            or projection.live_vintages != value.lifecycle.vintages(value.formation)
            or projection.live_vintages != value.lifecycle.vintages(projection.formation_session)
            or projection.model_identity_hashes != tuple(c.estimator_hash for c in value.children)
        ):
            raise AuthoringError("alpha_research.lifecycle_model_set_binding_invalid")
        verify_model_set(store, value)
    return str(lifecycle_program_scheme(program, receipt))
