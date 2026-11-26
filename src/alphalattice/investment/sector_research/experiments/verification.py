"""Read one Sector development evidence graph back by hash, and cross-check it.

Every artifact in the graph is content-addressed, so each one *individually*
verifies against its own name no matter which run produced it. A target
evidence from run A and a surface from run B are both perfectly valid
artifacts; what makes the pair wrong is that they do not belong to each other.
So this walks the graph from the root and asserts the lineage: outcome terminal
seal, target and membership recipe, exact ordered axes, the matured training
boundary, method recipe and implementation identity, numerical environment,
forecast value identity, the re-derived evaluation, and child completeness and
order.

The terminal edge is re-derived from the *outcome reader* rather than from
anything this Desk wrote. That last step is what a self-hash check cannot do: a
forger who sealed a target evidence, a surface and an evaluation against each
other would produce a perfectly self-consistent triple, and it fails here
because the outcome snapshot it names does not carry the method binding it
claims.

Nothing here can compute a forecast. This module must not import
``experiments.service``, the model catalog or any model adapter: verification
resolves evidence and must not be able to reach a method through it, and the
import graph is where that is enforced rather than left to convention. The
installed catalog identity arrives as a contract from whoever composed the
verifier.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.kernel.shared_kernel.identity_successors import is_current

from ..contracts import SectorResearchError
from ..models.catalog import SECTOR_CATALOG_ROLE, numerical_binding_role
from ..models.contracts import SectorForecastCatalogBinding
from ..targets.execution import SectorTargetEvidence, build_sector_target_recipe
from .development_artifacts import (
    SectorArtifactReadbackError,
    SectorDevelopmentArtifactStore,
    SectorExperimentEvidence,
    SectorForecastEvaluation,
    SectorForecastSurface,
    causal_training_selection,
    evaluate_sector_forecast_surface,
    sector_development_source_closure_hash,
)


@dataclass(frozen=True, slots=True)
class SectorExperimentLineage:
    """A verified walk from a published experiment hash back to its authority."""

    experiment: SectorExperimentEvidence
    target_evidence: SectorTargetEvidence
    surface: SectorForecastSurface
    evaluation: SectorForecastEvaluation
    outcome_snapshot_hash: str
    outcome_method_binding_hash: str
    maturity_lag_sessions: int


class SectorEvidenceVerifier:
    """Verify the experiment root, its three children, and every edge between them."""

    def __init__(
        self,
        *,
        store: SectorDevelopmentArtifactStore,
        outcome_reader: CausalExecutionOutcomeDevelopmentReader,
        catalog_binding: SectorForecastCatalogBinding,
    ) -> None:
        self._store = store
        self._outcome_reader = outcome_reader
        self._catalog_binding = catalog_binding

    def verify(self, *, experiment_hash: str) -> SectorExperimentLineage:
        experiment = self._load_root(experiment_hash)
        evidence, surface, evaluation = self._load_children(experiment)
        self._verify_child_lineage(experiment, evidence, surface, evaluation)
        self._verify_method_installation(surface)
        self._verify_target_policy(evidence)
        self._verify_training_boundary(evidence, surface)
        self._verify_evaluation(evidence, surface, evaluation)
        self._verify_terminal_seal(evidence)
        return SectorExperimentLineage(
            experiment=experiment,
            target_evidence=evidence,
            surface=surface,
            evaluation=evaluation,
            outcome_snapshot_hash=evidence.causal_outcome_snapshot_hash,
            outcome_method_binding_hash=evidence.outcome_method_binding_hash,
            maturity_lag_sessions=evidence.maturity_lag_sessions,
        )

    # ------------------------------------------------------------------ loading
    def _load_root(self, experiment_hash: str) -> SectorExperimentEvidence:
        try:
            return self._store.load_experiment_evidence(experiment_hash)
        except (SectorArtifactReadbackError, ValueError) as error:
            raise SectorResearchError(
                f"sector_research.evidence_root_unverifiable:{error}"
            ) from error

    def _load_children(
        self, experiment: SectorExperimentEvidence
    ) -> tuple[SectorTargetEvidence, SectorForecastSurface, SectorForecastEvaluation]:
        try:
            evidence = self._store.load_target_evidence(experiment.target_evidence_hash)
            surface = self._store.load_forecast_surface(experiment.forecast_surface_hash)
            evaluation = self._store.load_evaluation(experiment.evaluation_hash)
        except FileNotFoundError as error:
            # A named child that is gone is an incomplete graph, and replay must
            # refuse cleanly rather than let a bare miss escape the walk.
            raise SectorResearchError(f"sector_research.evidence_child_missing:{error}") from error
        except (SectorArtifactReadbackError, ValueError) as error:
            raise SectorResearchError(
                f"sector_research.evidence_child_unverifiable:{error}"
            ) from error
        return evidence, surface, evaluation

    # ------------------------------------------------------------------- edges
    @staticmethod
    def _verify_child_lineage(
        experiment: SectorExperimentEvidence,
        evidence: SectorTargetEvidence,
        surface: SectorForecastSurface,
        evaluation: SectorForecastEvaluation,
    ) -> None:
        """Reject valid children that do not belong to each other.

        Child completeness and order were already re-derived by the root's own
        validator when it parsed; what remains is that every cross-reference
        between the children names the sibling that actually loaded.
        """

        if surface.target_evidence_hash != evidence.evidence_hash:
            raise SectorResearchError("sector_research.evidence_surface_not_this_target")
        if evaluation.surface_hash != surface.surface_hash:
            raise SectorResearchError("sector_research.evidence_evaluation_not_this_surface")
        if evaluation.target_evidence_hash != evidence.evidence_hash:
            raise SectorResearchError("sector_research.evidence_evaluation_not_this_target")
        if experiment.method_id != surface.recipe.method_id:
            raise SectorResearchError("sector_research.evidence_method_route_invalid")
        if experiment.causal_outcome_snapshot_hash != evidence.causal_outcome_snapshot_hash:
            raise SectorResearchError("sector_research.evidence_snapshot_route_invalid")
        if experiment.sector_revision != evidence.recipe.sector_revision:
            raise SectorResearchError("sector_research.evidence_sector_revision_mismatch")
        if surface.ordered_sectors != evidence.ordered_sectors:
            raise SectorResearchError("sector_research.evidence_sector_axis_mismatch")
        missing = set(surface.forecast_formation_sessions) - set(evidence.formation_sessions)
        if missing:
            raise SectorResearchError("sector_research.evidence_forecast_axis_not_in_target")

    def _verify_method_installation(self, surface: SectorForecastSurface) -> None:
        """The implementation, its frozen recipe, and the Host execution path
        that sealed the surface must all still be the installed ones.

        Four separate refusals for four separate drifts: the catalog hash ties
        the run to the installed set as a whole; the per-method numerical
        binding catches a rewritten adapter wearing a stable id; the admitted
        recipe hash catches a re-sealed surface carrying parameters outside the
        frozen singleton -- an internally self-consistent 999-session half-life
        verified before this check existed, because "installed" said nothing
        about parameters; and the source closure catches Host compiler,
        selection, evaluation or service code that is no longer what ran.
        """

        if not is_current(
            SECTOR_CATALOG_ROLE,
            surface.program_binding.catalog_hash,
            self._catalog_binding.catalog_hash,
        ):
            raise SectorResearchError("sector_research.evidence_catalog_not_installed")
        installed = {value.method_id: value for value in self._catalog_binding.ordered_capabilities}
        capability = installed.get(surface.recipe.method_id)
        if capability is None:
            raise SectorResearchError("sector_research.evidence_method_not_installed")
        if capability.recipe_schema_id != surface.recipe.recipe_schema_id:
            raise SectorResearchError("sector_research.evidence_schema_not_installed")
        if not is_current(
            numerical_binding_role(surface.recipe.method_id),
            surface.program_binding.numerical_binding_hash,
            capability.numerical_binding_hash,
        ):
            raise SectorResearchError("sector_research.evidence_implementation_drift")
        if capability.admitted_recipe_hash != surface.recipe.recipe_hash:
            raise SectorResearchError("sector_research.evidence_recipe_not_admitted")
        if (
            surface.program_binding.development_source_closure_hash
            != sector_development_source_closure_hash()
        ):
            raise SectorResearchError("sector_research.evidence_execution_closure_drift")

    @staticmethod
    def _verify_target_policy(evidence: SectorTargetEvidence) -> None:
        """Re-derive the installed clean-target policy; equality is the check.

        The recipe's per-run fields -- which outcome method, which membership
        revision -- legitimately vary and are verified against the terminal
        seal. Everything else is installed policy: the aggregation, the
        minimum sector sample, the sector source and history treatment. A
        surface compiled under a one-listing "sector floor" or a different
        membership source would still parse and re-derive its own hashes, so
        the expected recipe is rebuilt from the installed builder with only the
        per-run fields taken from the evidence, and the whole hash must agree.
        """

        expected = build_sector_target_recipe(
            execution_outcome_recipe_id=evidence.recipe.execution_outcome_recipe_id,
            sector_revision=evidence.recipe.sector_revision,
            sector_history_treatment=evidence.recipe.sector_history_treatment,
        )
        if expected.recipe_hash != evidence.recipe.recipe_hash:
            raise SectorResearchError("sector_research.evidence_target_policy_not_installed")

    @staticmethod
    def _verify_training_boundary(
        evidence: SectorTargetEvidence, surface: SectorForecastSurface
    ) -> None:
        """Re-derive the causal training selection and hold the surface to it.

        The rule is shared with the executor, so this is not a second opinion;
        it is the same derivation applied to the published clocks. A surface
        whose recorded training row counts disagree describes a run that did
        not select by maturity -- or edited its account of having done so.
        """

        for row in range(len(surface.forecast_formation_sessions)):
            selection = causal_training_selection(
                evidence=evidence, program=surface.program_binding, forecast_row=row
            )
            if len(selection) != surface.training_row_counts[row]:
                raise SectorResearchError("sector_research.evidence_training_boundary_violated")

    @staticmethod
    def _verify_evaluation(
        evidence: SectorTargetEvidence,
        surface: SectorForecastSurface,
        evaluation: SectorForecastEvaluation,
    ) -> None:
        """Recompute the evaluation from its two parents; equality is the check."""

        derived = evaluate_sector_forecast_surface(evidence=evidence, surface=surface)
        if derived.evaluation_hash != evaluation.evaluation_hash:
            raise SectorResearchError("sector_research.evidence_evaluation_not_rederivable")

    def _verify_terminal_seal(self, evidence: SectorTargetEvidence) -> None:
        """Terminate at the outcome Desk's reader, not at anything Sector wrote."""

        seal = self._outcome_reader.resolve_method_seal(evidence.causal_outcome_snapshot_hash)
        if seal.disposition != "METHOD_BOUND":
            raise SectorResearchError("sector_research.evidence_outcome_method_unbound")
        method = seal.method_bound
        if method.binding_hash != evidence.outcome_method_binding_hash:
            raise SectorResearchError("sector_research.evidence_outcome_method_mismatch")
        if method.recipe_id != evidence.recipe.execution_outcome_recipe_id:
            raise SectorResearchError("sector_research.evidence_outcome_recipe_mismatch")
        if (
            method.maturity_lag_sessions != evidence.maturity_lag_sessions
            or method.actual_session_span != evidence.actual_session_span
            or method.exit_offset_sessions - method.entry_offset_sessions
            != evidence.forecast_horizon_sessions
        ):
            raise SectorResearchError("sector_research.evidence_outcome_clock_mismatch")


__all__ = ["SectorEvidenceVerifier", "SectorExperimentLineage"]
