"""Host-owned publication of one deterministic Sector development experiment.

The request boundary is the point. A caller may name *what* to run -- an
outcome snapshot, a sector revision, an installed method with its admitted
singleton parameters, and the training and forecast ranges -- and may not state
any identity that gives the result its authority. The catalog hash, the method
and numerical bindings, the target and evidence hashes and the ordered axes are
all derived here from resolved evidence; a caller that could supply them would
be asserting exactly what publication exists to establish.

This service is deliberately narrow: it resolves authority, compiles the clean
target, runs one installed method over the causal boundary, evaluates, publishes
children before parents, and verifies its own publication by reading it back
through the verifier. It owns no schedule, fits no Alpha model, and never
touches a current pointer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.foundation.factor_research.inputs.execution_target import (
    build_factor_target_policy,
    compile_factor_target_surface,
)
from alphalattice.kernel.quant.sector_history import sector_treatment_of
from alphalattice.protocols.actor_execution import ActorKind

from ..contracts import FloatArray, SectorResearchError, grid_to_matrix, sector_array_identity
from ..evaluation.calibration import (
    SectorShrinkCalibrationEvidence,
    calibrate_sector_forecast_surface,
)
from ..evaluation.comparison import (
    SectorMethodComparisonEvidence,
    SectorMethodComparisonRow,
    build_sector_method_comparison,
)
from ..inputs.membership import load_sector_membership
from ..models.catalog import SectorForecastCatalog, build_installed_sector_forecast_catalog
from ..models.contracts import (
    BoundSectorForecastInput,
    SectorForecastAdapter,
    SectorForecastRecipe,
    SectorForecastValues,
)
from ..targets.execution import (
    SectorTargetEvidence,
    build_sector_target_recipe,
    compile_sector_target_surface,
    seal_sector_target_evidence,
)
from .campaign import CONDITIONAL_SECTOR_METHOD_IDS, SectorCampaignRequest
from .campaign_evidence import (
    SectorCampaignDossier,
    SectorCampaignReplayReceipt,
    SectorForecastSelectionReceipt,
    SectorForecastSelectionSubmission,
    admit_sector_forecast_selection,
    verify_sector_campaign_recursive_replay,
)
from .development_artifacts import (
    SECTOR_CAMPAIGN_DECISION_CATEGORY,
    SECTOR_CAMPAIGN_DOSSIER_CATEGORY,
    SECTOR_CAMPAIGN_REPLAY_CATEGORY,
    SECTOR_METHOD_COMPARISON_CATEGORY,
    SECTOR_SHRINK_CALIBRATION_CATEGORY,
    SectorDevelopmentArtifactStore,
    SectorExperimentEvidence,
    SectorForecastEvaluation,
    SectorForecastProgramBinding,
    SectorForecastSurface,
    causal_training_selection,
    evaluate_sector_forecast_surface,
    sector_artifact_uri,
    sector_development_source_closure_hash,
)
from .verification import SectorEvidenceVerifier, SectorExperimentLineage


@dataclass(frozen=True, slots=True)
class SectorExperimentRequest:
    """What a caller is entitled to name: handles and ranges, never conclusions."""

    causal_outcome_snapshot_hash: str
    sector_revision: str
    method_id: str
    training_start: date
    training_end: date
    forecast_start: date
    forecast_end: date
    parameters: Mapping[str, int] | None = None
    """Optional restatement of the admitted singleton. Naming anything else is
    refused by the catalog; omitting it selects the singleton."""


@dataclass(frozen=True, slots=True)
class SectorExperimentPublication:
    """Everything one run sealed, with the URIs it can be re-read from."""

    experiment: SectorExperimentEvidence
    target_evidence: SectorTargetEvidence
    surface: SectorForecastSurface
    evaluation: SectorForecastEvaluation
    target_evidence_uri: str
    surface_uri: str
    evaluation_uri: str
    experiment_uri: str
    formation_sessions: tuple[date, ...]
    forecast_formation_sessions: tuple[date, ...]
    ordered_sectors: tuple[str, ...]
    adapter_call_count: int
    """Distinct refits, not forecast rows: the cadence reuses one refit's values
    across the sessions until the next one, and counting rows would report work
    that did not happen."""


@dataclass(frozen=True, slots=True)
class SectorCampaignPublication:
    """Everything one Campaign sealed, before any decision is submitted."""

    dossier: SectorCampaignDossier
    comparison: SectorMethodComparisonEvidence
    calibrations: tuple[SectorShrinkCalibrationEvidence, ...]
    target_evidence: SectorTargetEvidence
    experiment_hashes: tuple[str, ...]


class SectorResearchDevelopmentService:
    """Resolve, compile, forecast, evaluate, publish and verify one experiment."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        outcome_reader: CausalExecutionOutcomeDevelopmentReader,
        resolver: ArtifactResolver,
        catalog: SectorForecastCatalog | None = None,
    ) -> None:
        self.store = SectorDevelopmentArtifactStore(artifact_root)
        self._outcome_reader = outcome_reader
        self._resolver = resolver
        self._catalog = catalog or build_installed_sector_forecast_catalog()

    # ------------------------------------------------------------------ publish
    def publish(self, request: SectorExperimentRequest) -> SectorExperimentPublication:
        evidence = self._compile_target_evidence(request)
        return self._publish_experiment(
            evidence=evidence,
            sector_revision=request.sector_revision,
            method_id=request.method_id,
            parameters=request.parameters,
            training_start=request.training_start,
            training_end=request.training_end,
            forecast_start=request.forecast_start,
            forecast_end=request.forecast_end,
        )

    def _publish_experiment(
        self,
        *,
        evidence: SectorTargetEvidence,
        sector_revision: str,
        method_id: str,
        parameters: Mapping[str, int] | None,
        training_start: date,
        training_end: date,
        forecast_start: date,
        forecast_end: date,
    ) -> SectorExperimentPublication:
        """Run and seal one method against an already-compiled target.

        Separated from ``publish`` so a Campaign compares methods against the
        *same* compiled evidence rather than recompiling it per method: two
        compilations would be content-identical here, but making that an
        assumption is how a comparison quietly stops being one.
        """

        recipe = self._catalog.seal_recipe(method_id=method_id, parameters=parameters)
        adapter = self._catalog.resolve(recipe)
        numerical_binding = adapter.describe_numerical_binding()

        forecast_sessions = tuple(
            value
            for value in evidence.formation_sessions
            if forecast_start <= value <= forecast_end
        )
        if not forecast_sessions:
            raise SectorResearchError("sector_research.forecast_axis_empty")

        program = SectorForecastProgramBinding.create(
            kind="SectorForecastProgramBinding",
            target_evidence_hash=evidence.evidence_hash,
            catalog_hash=self._catalog.binding.catalog_hash,
            method_id=recipe.method_id,
            recipe_hash=recipe.recipe_hash,
            numerical_binding_hash=numerical_binding.numerical_binding_hash,
            training_start=training_start,
            training_end=training_end,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            forecast_horizon_sessions=evidence.forecast_horizon_sessions,
            maturity_lag_sessions=evidence.maturity_lag_sessions,
            refit_every_sessions=recipe.parameters.get("refit_every_sessions", 0),
            minimum_history_sessions=recipe.parameters.get("minimum_history_sessions", 0),
            forecast_formation_sessions=forecast_sessions,
            development_source_closure_hash=sector_development_source_closure_hash(),
        )

        target_matrix = grid_to_matrix(evidence.sector_target_values)
        values_rows: list[tuple[float | None, ...]] = []
        reason_rows: list[tuple[str | None, ...]] = []
        counts: list[int] = []
        refit_results: dict[int, SectorForecastValues] = {}
        cadence = program.refit_every_sessions
        for row in range(len(forecast_sessions)):
            refit_row = row - (row % cadence) if cadence > 0 else row
            result = refit_results.get(refit_row)
            selection = causal_training_selection(
                evidence=evidence, program=program, forecast_row=row
            )
            if result is None:
                result = self._run_adapter(
                    adapter=adapter,
                    recipe=recipe,
                    evidence=evidence,
                    target_matrix=target_matrix,
                    selection=selection,
                    refit_formation=forecast_sessions[refit_row],
                )
                refit_results[refit_row] = result
            values_rows.append(result.values)
            reason_rows.append(result.unavailable_reasons)
            counts.append(len(selection))

        surface = SectorForecastSurface.create(
            kind="SectorForecastSurface",
            identity_class="DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED",
            target_evidence_hash=evidence.evidence_hash,
            recipe=recipe,
            program_binding=program,
            ordered_sectors=evidence.ordered_sectors,
            forecast_formation_sessions=forecast_sessions,
            values=tuple(values_rows),
            unavailable_reasons=tuple(reason_rows),
            training_row_counts=tuple(counts),
            values_identity=sector_array_identity(grid_to_matrix(tuple(values_rows))),
        )
        evaluation = evaluate_sector_forecast_surface(evidence=evidence, surface=surface)

        # Children before parents, so nothing durable ever names a document
        # that does not yet exist. The experiment root is published last.
        target_evidence_uri = self.store.publish_target_evidence(evidence)
        surface_uri = self.store.publish_forecast_surface(surface)
        evaluation_uri = self.store.publish_evaluation(evaluation)
        experiment = SectorExperimentEvidence.create(
            kind="SectorExperimentEvidence",
            causal_outcome_snapshot_hash=evidence.causal_outcome_snapshot_hash,
            sector_revision=sector_revision,
            method_id=recipe.method_id,
            catalog_hash=self._catalog.binding.catalog_hash,
            target_evidence_hash=evidence.evidence_hash,
            forecast_surface_hash=surface.surface_hash,
            evaluation_hash=evaluation.evaluation_hash,
            ordered_child_uris=(target_evidence_uri, surface_uri, evaluation_uri),
        )
        experiment_uri = self.store.publish_experiment_evidence(experiment)

        # Read back through the verifier rather than trusting the objects still
        # in memory: a publisher that verifies its own variables proves only
        # that it can remember them.
        self.verify(experiment_hash=experiment.experiment_hash)
        return SectorExperimentPublication(
            experiment=experiment,
            target_evidence=evidence,
            surface=surface,
            evaluation=evaluation,
            target_evidence_uri=target_evidence_uri,
            surface_uri=surface_uri,
            evaluation_uri=evaluation_uri,
            experiment_uri=experiment_uri,
            formation_sessions=evidence.formation_sessions,
            forecast_formation_sessions=forecast_sessions,
            ordered_sectors=evidence.ordered_sectors,
            adapter_call_count=len(refit_results),
        )

    # ------------------------------------------------------------------- verify
    def verify(self, *, experiment_hash: str) -> SectorExperimentLineage:
        """Walk a published experiment back to the outcome method that authorized it."""

        return SectorEvidenceVerifier(
            store=self.store,
            outcome_reader=self._outcome_reader,
            catalog_binding=self._catalog.binding,
        ).verify(experiment_hash=experiment_hash)

    def find_published_experiments(self) -> tuple[str, ...]:
        """Experiment hashes already sealed in this workspace, in canonical order."""

        return self.store.list_experiment_evidence()

    # ----------------------------------------------------------------- campaign
    def run_campaign(self, request: SectorCampaignRequest) -> SectorCampaignPublication:
        """Compare the selected installed methods on one compiled target.

        Every method is run against the same target evidence, the same forecast
        axis and the same causal training rule, so the only thing that differs
        across the comparable table is the method. Each earns a cross-fitted
        shrink calibration against the raw economic lane, the frozen rules rank
        them, and the dossier is published last, after every child it names.

        The conditional VAR family is not installed and is not run. Its
        disposition is published either way, so "we did not run these" is a
        recorded fact carrying the trigger that would have changed it.
        """

        evidence = self._compile_target_evidence(request)
        rows: list[SectorMethodComparisonRow] = []
        calibrations: list[SectorShrinkCalibrationEvidence] = []
        experiment_hashes: list[str] = []
        adapter_calls = 0
        metric_calls = 0

        for method_id in request.method_ids:
            published = self._publish_experiment(
                evidence=evidence,
                sector_revision=request.sector_revision,
                method_id=method_id,
                parameters=request.parameters_for(method_id),
                training_start=request.training_start,
                training_end=request.training_end,
                forecast_start=request.forecast_start,
                forecast_end=request.forecast_end,
            )
            adapter_calls += published.adapter_call_count
            metric_calls += 1  # the evaluation the experiment already sealed
            calibration = calibrate_sector_forecast_surface(
                evidence=evidence,
                surface=published.surface,
                fold_count=request.calibration_fold_count,
            )
            metric_calls += 1
            self.store.publish_contract(
                category=SECTOR_SHRINK_CALIBRATION_CATEGORY,
                value=calibration,
                identity_field="calibration_hash",
            )
            calibrations.append(calibration)
            experiment_hashes.append(published.experiment.experiment_hash)
            available = sum(
                1 for row in published.surface.values for value in row if value is not None
            )
            rows.append(
                SectorMethodComparisonRow(
                    method_id=method_id,
                    disposition="EVALUATED",
                    experiment_hash=published.experiment.experiment_hash,
                    surface_hash=published.surface.surface_hash,
                    evaluation_hash=published.evaluation.evaluation_hash,
                    calibration_hash=calibration.calibration_hash,
                    forecast_cell_count=len(published.surface.values)
                    * len(published.surface.ordered_sectors),
                    available_cell_count=available,
                    evaluated_pair_count=calibration.evaluated_pair_count,
                    mean_cross_fitted_slope=calibration.mean_cross_fitted_slope,
                    mean_calibrated_economic_squared_error=(
                        calibration.mean_calibrated_economic_squared_error
                    ),
                    mean_identity_economic_squared_error=(
                        calibration.mean_identity_economic_squared_error
                    ),
                )
            )

        comparison = build_sector_method_comparison(
            target_evidence_hash=evidence.evidence_hash,
            catalog_hash=self._catalog.binding.catalog_hash,
            rows=rows,
        )
        self.store.publish_contract(
            category=SECTOR_METHOD_COMPARISON_CATEGORY,
            value=comparison,
            identity_field="comparison_hash",
        )

        dossier = SectorCampaignDossier.create(
            kind="SectorCampaignDossier",
            identity_class="DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED",
            request_hash=request.request_hash,
            causal_outcome_snapshot_hash=evidence.causal_outcome_snapshot_hash,
            sector_revision=request.sector_revision,
            catalog_hash=self._catalog.binding.catalog_hash,
            target_evidence_hash=evidence.evidence_hash,
            training_start=request.training_start,
            training_end=request.training_end,
            forecast_start=request.forecast_start,
            forecast_end=request.forecast_end,
            calibration_fold_count=request.calibration_fold_count,
            ordered_experiment_hashes=tuple(experiment_hashes),
            ordered_calibration_hashes=tuple(value.calibration_hash for value in calibrations),
            comparison_hash=comparison.comparison_hash,
            conditional_dispositions={
                value: ("EVALUATED" if comparison.joint_dynamics_triggered else "NOT_TRIGGERED")
                for value in CONDITIONAL_SECTOR_METHOD_IDS
            },
            forecast_call_count=adapter_calls,
            metric_call_count=metric_calls,
            limitations=(
                "DEVELOPMENT_ONLY_NO_CURRENT_OR_PRODUCTION_ADMISSION",
                "NO_HOLDOUT_OR_PROSPECTIVE_EVIDENCE",
                "SINGLE_SECTOR_REVISION_AND_SINGLE_OUTCOME_SNAPSHOT",
                "CONDITIONAL_VAR_FAMILY_NOT_INSTALLED",
            ),
            ordered_child_uris=(
                *(
                    sector_artifact_uri(SECTOR_SHRINK_CALIBRATION_CATEGORY, value.calibration_hash)
                    for value in calibrations
                ),
                sector_artifact_uri(SECTOR_METHOD_COMPARISON_CATEGORY, comparison.comparison_hash),
            ),
        )
        self.store.publish_contract(
            category=SECTOR_CAMPAIGN_DOSSIER_CATEGORY,
            value=dossier,
            identity_field="dossier_hash",
        )
        return SectorCampaignPublication(
            dossier=dossier,
            comparison=comparison,
            calibrations=tuple(calibrations),
            target_evidence=evidence,
            experiment_hashes=tuple(experiment_hashes),
        )

    def seal_campaign_decision(
        self,
        *,
        dossier: SectorCampaignDossier,
        comparison: SectorMethodComparisonEvidence,
        submission: SectorForecastSelectionSubmission,
        actor_kind: ActorKind,
        actor_id: str,
    ) -> SectorForecastSelectionReceipt:
        """Validate an actor's proposal against the frozen rules and seal it."""

        receipt = admit_sector_forecast_selection(
            dossier=dossier,
            comparison=comparison,
            submission=submission,
            actor_kind=actor_kind,
            actor_id=actor_id,
        )
        self.store.publish_contract(
            category=SECTOR_CAMPAIGN_DECISION_CATEGORY,
            value=receipt,
            identity_field="receipt_hash",
        )
        return receipt

    def verify_campaign(
        self, *, dossier_hash: str, decision_receipt_hash: str
    ) -> SectorCampaignReplayReceipt:
        """Replay the whole Campaign graph and publish the receipt."""

        replay = verify_sector_campaign_recursive_replay(
            store=self.store,
            outcome_reader=self._outcome_reader,
            catalog_binding=self._catalog.binding,
            dossier_hash=dossier_hash,
            decision_receipt_hash=decision_receipt_hash,
        )
        self.store.publish_contract(
            category=SECTOR_CAMPAIGN_REPLAY_CATEGORY,
            value=replay,
            identity_field="replay_hash",
        )
        return replay

    # ---------------------------------------------------------------- internals
    def _compile_target_evidence(
        self, request: SectorExperimentRequest | SectorCampaignRequest
    ) -> SectorTargetEvidence:
        """Resolve every authority and compile the clean target for one run.

        Takes either request shape because the compilation depends only on the
        handles and the forecast end, which both carry: a Campaign resolving its
        target through a second code path would be a second target owner.
        """

        manifest_ref = self._outcome_reader.manifest_uri(request.causal_outcome_snapshot_hash)
        manifest = self._outcome_reader.load_manifest(request.causal_outcome_snapshot_hash)
        seal = self._outcome_reader.resolve_method_seal(request.causal_outcome_snapshot_hash)
        if seal.disposition != "METHOD_BOUND":
            # A pre-seam snapshot stays readable and never becomes the source of
            # new development evidence.
            raise SectorResearchError("sector_research.outcome_method_unbound")
        method = seal.method_bound
        if method.snapshot_hash != manifest.snapshot_hash:
            raise SectorResearchError("sector_research.outcome_seal_mismatch")

        # The evidence axis is every development formation whose label had
        # matured by the end of the forecast range: a later formation could be
        # neither trained on nor evaluated, so admitting it would put rows in
        # the record that no consumer may touch.
        sessions = self._outcome_reader.available_development_sessions(
            manifest_ref, holding_end_through=request.forecast_end
        )
        if not sessions:
            raise SectorResearchError("sector_research.matured_axis_empty")
        source_rows = self._outcome_reader.read_development_sessions(manifest_ref, sessions)
        if source_rows.num_rows == 0:
            raise SectorResearchError("sector_research.outcome_rows_unavailable")

        # The log/simple lanes come from the Factor target compiler, the one
        # owner of that projection, given the resolved seal rather than a
        # described method.
        factor_surface = compile_factor_target_surface(
            source_table=source_rows,
            source_manifest=manifest,
            source_manifest_ref=manifest_ref,
            policy=build_factor_target_policy(),
            outcome_method=seal,
        )
        sector_by_listing_id = load_sector_membership(
            resolver=self._resolver, sector_revision=request.sector_revision
        )
        listing_ids = tuple(sorted(set(str(value) for value in manifest.listing_ids)))
        if set(listing_ids) - set(sector_by_listing_id):
            # Membership is an authority, not a convenience: a listing the
            # revision does not cover cannot be aggregated into a sector.
            raise SectorResearchError("sector_research.sector_authority_incomplete")

        recipe = build_sector_target_recipe(
            execution_outcome_recipe_id=method.recipe_id,
            sector_revision=request.sector_revision,
            # What its formations read.
            sector_history_treatment=sector_treatment_of(sector_by_listing_id),
        )
        surface = compile_sector_target_surface(
            source_table=factor_surface.table,
            recipe=recipe,
            sector_by_listing_id=sector_by_listing_id,
        )
        if surface.ordered_listing_ids != listing_ids:
            raise SectorResearchError("sector_research.listing_axis_mismatch")
        if surface.formation_sessions != sessions:
            raise SectorResearchError("sector_research.session_axis_mismatch")

        # Every identity below is derived here. Nothing in the request could
        # have supplied one.
        return seal_sector_target_evidence(
            surface=surface,
            causal_outcome_snapshot_hash=manifest.snapshot_hash,
            outcome_method_binding_hash=method.binding_hash,
            maturity_lag_sessions=method.maturity_lag_sessions,
            actual_session_span=method.actual_session_span,
            forecast_horizon_sessions=(method.exit_offset_sessions - method.entry_offset_sessions),
        )

    @staticmethod
    def _run_adapter(
        *,
        adapter: SectorForecastAdapter,
        recipe: SectorForecastRecipe,
        evidence: SectorTargetEvidence,
        target_matrix: FloatArray,
        selection: tuple[int, ...],
        refit_formation: date,
    ) -> SectorForecastValues:
        bound = BoundSectorForecastInput.create(
            target_evidence_hash=evidence.evidence_hash,
            ordered_sectors=evidence.ordered_sectors,
            forecast_formation_at=refit_formation,
            training_formation_sessions=tuple(
                evidence.formation_sessions[index] for index in selection
            ),
            training_target_available_sessions=tuple(
                evidence.target_available_sessions[index] for index in selection
            ),
            training_values=target_matrix[list(selection), :],
        )
        result = adapter.forecast(bound_input=bound, recipe=recipe)
        if (
            result.method_id != recipe.method_id
            or result.ordered_sectors != evidence.ordered_sectors
            or result.forecast_formation_at != refit_formation
        ):
            raise SectorResearchError("sector_research.adapter_result_route_invalid")
        return result


__all__ = [
    "SectorCampaignPublication",
    "SectorExperimentPublication",
    "SectorExperimentRequest",
    "SectorResearchDevelopmentService",
]
