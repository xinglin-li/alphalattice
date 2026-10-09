"""Execute one compiled Risk development Program, whichever method it selected.

Nothing here knows which method is running. The executor resolves an adapter
from the catalog, hands it the sealed recipe envelope, and lets the adapter
decode its own recipe. It imports no recipe contract and has no branch on
adapter id, recipe schema or family -- an executor that had to recognise a
schema before running it would be the compiler's old hardcoding relocated one
layer down, and every additional method would need another branch.

Development builds write their own artifacts, shared by every capability, rather
than the published ``HistoricalCovarianceSurface``. That contract types its
recipe field as ``CovarianceRecipe`` and so can only ever describe one schema;
pointing a second method at it would mean writing the wrong recipe into a
current field, and giving the second method a private artifact would put the
branch back.

This module must not import ``risk_research.publication``. Development execution
has no authority to move a current or admitted pointer, and the import graph is
where that is enforced rather than left to convention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from alphalattice.investment.risk_research.contracts import CausalRiskReturnSurface
from alphalattice.investment.risk_research.estimators.capability import RiskRecipeAdmission
from alphalattice.investment.risk_research.estimators.catalog import (
    ESTIMATOR_CATALOG_ROLE,
    RiskEstimatorCatalog,
    build_installed_risk_estimator_catalog,
    numerical_binding_role,
)
from alphalattice.investment.risk_research.experiments.compiler import (
    RISK_EXPERIMENT_KIND,
    RiskExperimentCompiler,
)
from alphalattice.investment.risk_research.experiments.contracts import (
    RiskDevelopmentProgramBinding,
)
from alphalattice.investment.risk_research.experiments.development import (
    RiskDevelopmentBuild,
    build_development_covariance_surface,
)
from alphalattice.investment.risk_research.experiments.development_artifacts import (
    DEVELOPMENT_DIAGNOSTICS_CATEGORY,
    DEVELOPMENT_SURFACE_CATEGORY,
)
from alphalattice.investment.risk_research.experiments.observation import (
    NumericalCallCounter,
    ObservingAdapter,
)
from alphalattice.investment.risk_research.experiments.series import (
    SERIES_CATEGORY,
    RiskDevelopmentSeries,
    RiskSeriesMember,
    combined_diagnostics,
)
from alphalattice.investment.risk_research.experiments.window import (
    RISK_INPUT_BINDING_CATEGORY,
    BoundedCausalReturnReader,
    PublishedReturnSurfaceProvider,
    ReturnSurfaceFreshnessProbe,
    RiskDevelopmentInputBinding,
    resolve_development_input_binding,
    select_return_surface_for_authority,
)
from alphalattice.investment.risk_research.surfaces.artifacts import RiskArtifactStore
from alphalattice.investment.risk_research.surfaces.returns import CausalRiskReturnReader
from alphalattice.kernel.quant.sector_history import SectorHistory, sector_subset
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskExecutionResult,
    NumericalCallRecorder,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)


@dataclass(frozen=True, slots=True)
class RiskDevelopmentExecution:
    """One completed development build and the identity it ran under."""

    binding: RiskDevelopmentProgramBinding
    input_binding: RiskDevelopmentInputBinding
    build: RiskDevelopmentBuild
    formation_sessions: tuple[str, ...]
    estimate_calls: int
    """Real ``adapter.estimate`` invocations, counted as they happened.

    Not ``len(formation_sessions)``: a run that resumed a checkpoint executes
    only the missing suffix, and the difference is the whole reason this is
    observed rather than derived.
    """

    @property
    def surface_hash(self) -> str:
        return str(self.build.surface.surface_hash)

    @property
    def artifact_uris(self) -> tuple[str, ...]:
        """Every terminal artifact this run produced, in a stable order.

        Derived from the execution rather than assembled by each caller, so the
        Desk executor and anything else reporting this run's evidence cannot
        drift into naming different artifact sets.

        Chunks are deliberately absent: the surface references them and the
        verifier descends into them from there. Listing them here as well would
        make the same bytes reachable by two independent paths that could
        disagree about which chunks belong to the surface.
        """

        surface = self.build.surface
        return (
            f"playpen://risk-research/{RISK_INPUT_BINDING_CATEGORY}"
            f"/{self.input_binding.input_binding_hash}",
            f"playpen://risk-research/{DEVELOPMENT_SURFACE_CATEGORY}/{surface.surface_hash}",
            f"playpen://risk-research/{DEVELOPMENT_DIAGNOSTICS_CATEGORY}"
            f"/{surface.diagnostics_hash}",
        )


class RiskDevelopmentExecutor:
    """Run a compiled development Program against a resolved workspace.

    Generic over the method. It receives an adapter and a sealed recipe
    envelope, and passes them straight through to the development writer. It
    never narrows the recipe, never names a schema, and has no branch on adapter
    id or family -- an executor that recognised a schema before running it would
    only be the compiler's old hardcoding relocated one layer down.
    """

    kind = RISK_EXPERIMENT_KIND

    def __init__(self, *, estimators: RiskEstimatorCatalog | None = None) -> None:
        self._estimators = estimators

    def execute(
        self,
        *,
        binding: RiskDevelopmentProgramBinding,
        input_binding: RiskDevelopmentInputBinding,
        admission: RiskRecipeAdmission,
        return_surface: CausalRiskReturnSurface,
        return_reader: CausalRiskReturnReader,
        bounded_sessions: Sequence[date],
        output_workspace: Path,
        sector_by_listing_id: Mapping[str, str],
        counter: NumericalCallCounter | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        reserve_output: Callable[[int], None] | None = None,
    ) -> RiskDevelopmentExecution:
        catalog = self._estimators or build_installed_risk_estimator_catalog()
        if not is_current(
            ESTIMATOR_CATALOG_ROLE, binding.catalog_hash, catalog.binding.catalog_hash
        ):
            # The Program was sealed against a different catalog than the one
            # about to compute. Reject before the first estimate rather than
            # publish A's identity over B's numbers.
            raise AuthoringError("research_authoring.catalog_identity_mismatch")
        # The recipe about to be executed must be the recipe the Program sealed.
        # Checked before anything computes, because the failure this replaces was
        # silent: the builder constructed its own recipe and the authored one was
        # never executed at all.
        if admission.recipe_hash != binding.recipe_hash:
            raise AuthoringError("research_authoring.recipe_identity_mismatch")
        if admission.selected_numerical_binding_hash != binding.selected_numerical_binding_hash:
            raise AuthoringError("research_authoring.selected_numerical_binding_mismatch")

        tally = counter if counter is not None else NumericalCallCounter()
        # Resolved through the catalog rather than carried on the admission, so
        # the adapter that computes is one this catalog installs.
        resolved = catalog.resolve(admission.recipe)
        # And it must be the *same implementation* the Program was sealed against.
        # Resolution routes by adapter id, and an id is not an identity: without
        # this, implementation B could compute under implementation A's numerical
        # binding and every hash on the resulting evidence would be internally
        # consistent while describing a computation that never happened.
        #
        # Compared against both the admission and the Program binding rather than
        # relying on the equality asserted above, because these two are what the
        # evidence will actually claim, and a transitive argument is not what a
        # reader of this line needs to reconstruct.
        resolved_binding_hash = resolved.describe_numerical_binding().numerical_binding_hash
        role = numerical_binding_role(resolved.adapter_id)
        if not is_current(
            role, admission.selected_numerical_binding_hash, resolved_binding_hash
        ) or not is_current(role, binding.selected_numerical_binding_hash, resolved_binding_hash):
            raise AuthoringError("research_authoring.resolved_numerical_binding_mismatch")
        # Counting wraps it here, which is why the count is observed rather than
        # derived.
        adapter = ObservingAdapter(resolved, tally)

        # Persisted before the build, not after: an input binding that only
        # exists once the numbers do is a description of what happened rather
        # than a commitment made in advance. It is content-addressed like every
        # other artifact, so writing it twice is a no-op.
        store = RiskArtifactStore(output_workspace)
        store.publish_json(
            category=RISK_INPUT_BINDING_CATEGORY,
            payload=input_binding.model_dump(mode="json"),
            identity_field="input_binding_hash",
        )
        build = build_development_covariance_surface(
            adapter=adapter,
            recipe_envelope=admission.recipe,
            recipe_identity_hash=admission.recipe_hash,
            capability_handle=admission.capability_handle,
            binding=binding,
            input_binding=input_binding,
            return_surface=return_surface,
            return_reader=BoundedCausalReturnReader(
                reader=return_reader,
                bounded_sessions=bounded_sessions,
                binding=input_binding,
            ),
            bounded_sessions=bounded_sessions,
            artifact_store=store,
            sector_by_listing_id=sector_by_listing_id,
            cancel_requested=cancel_requested,
            reserve_output=reserve_output,
        )
        # What was computed is what was sealed. Asserted rather than trusted,
        # because a silent divergence here is the defect this replaced.
        if tuple(build.surface.formation_sessions) != tuple(input_binding.formation_sessions):
            raise AuthoringError("research_authoring.formation_axis_diverged_from_program")
        if build.surface.recipe_hash != binding.recipe_hash:
            raise AuthoringError("research_authoring.recipe_diverged_from_program")
        return RiskDevelopmentExecution(
            binding=binding,
            input_binding=input_binding,
            build=build,
            formation_sessions=tuple(
                value.isoformat() for value in build.surface.formation_sessions
            ),
            estimate_calls=tally.estimate_calls,
        )


class RiskExperimentExecutor:
    """Adapt the Risk development build to the Desk executor Protocol.

    The workspace inputs -- the published return surface, its reader, and the
    sector map -- are constructor state supplied by Host composition, because
    knowing where a workspace keeps them is exactly the concern this Desk module
    should not acquire.
    """

    kind = RISK_EXPERIMENT_KIND

    def __init__(
        self,
        *,
        surface_provider: PublishedReturnSurfaceProvider,
        return_reader: CausalRiskReturnReader,
        sector_by_listing_id: Mapping[str, str],
        freshness_probe: ReturnSurfaceFreshnessProbe,
        estimators: RiskEstimatorCatalog | None = None,
        reserve_output: Callable[[int], None] | None = None,
        scope_provider: Callable[
            [ResolvedResearchAuthority],
            tuple[tuple[CausalRiskReturnSurface, ...], ReturnSurfaceFreshnessProbe],
        ]
        | None = None,
    ) -> None:
        # A provider, not a surface. Choosing one at composition time meant the
        # choice was made before the authority existed -- so a workspace holding
        # more than one immutable surface either failed outright or silently
        # bound a run to whichever file was read first.
        self._surface_provider = surface_provider
        self._return_reader = return_reader
        # A history is immutable and read per formation; a plain map is copied.
        self._sector_by_listing_id: Mapping[str, str] = (
            sector_by_listing_id
            if isinstance(sector_by_listing_id, SectorHistory)
            else dict(sector_by_listing_id)
        )
        self._freshness_probe = freshness_probe
        # One catalog instance owns both compilation and execution. It used to
        # be possible to pass a compiler built on catalog A alongside estimators
        # from catalog B and get evidence carrying A's identity over B's
        # numbers; there is now no parameter with which to express that.
        self._estimators = estimators or build_installed_risk_estimator_catalog()
        self._compiler = RiskExperimentCompiler(self._estimators)
        self._cancel_requested: Callable[[], bool] | None = None
        self._reserve_output = reserve_output
        self._scope_provider = scope_provider

    def set_cancellation_check(self, requested: Callable[[], bool]) -> None:
        self._cancel_requested = requested

    def execute(
        self,
        *,
        program: SealedResearchProgram,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
        output_workspace: Path,
        recorder: NumericalCallRecorder | None = None,
    ) -> DeskExecutionResult:
        envelope_document = document.get("experiment")
        if not isinstance(envelope_document, dict):
            raise AuthoringError("research_authoring.experiment_section_missing")
        envelope = ResearchExperimentEnvelope.create(**envelope_document)
        compiled = self._compiler.compile_development_program(
            envelope=envelope,
            document=document,
            authority=authority,
        )
        binding = compiled.binding
        if compiled.desk_program_hash != program.desk_program_hash:
            # The sealed Program and the document must still agree, or the run
            # would produce evidence for identity nobody admitted.
            raise AuthoringError("research_authoring.replay_identity_mismatch")

        # The compile above checked the declared budget and threads.
        authorities = tuple(authority.for_scope(scope) for scope in authority.listing_scopes) or (
            authority,
        )
        # Resolve all required sources before any estimate. A missing scope is
        # not permission to omit those formations and publish a shorter run.
        inputs = []
        for selected in authorities:
            surfaces, freshness = (
                self._scope_provider(selected)
                if self._scope_provider is not None
                else (self._surface_provider.published_surfaces(), self._freshness_probe)
            )
            surface = select_return_surface_for_authority(authority=selected, surfaces=surfaces)
            input_binding, bounded_sessions = resolve_development_input_binding(
                authority=selected,
                return_surface=surface,
                return_reader=self._return_reader,
                freshness_probe=freshness,
            )
            inputs.append((input_binding, bounded_sessions, surface))
        if (
            tuple(day for item, _, _ in inputs for day in item.formation_sessions)
            != authority.sessions
        ):
            raise AuthoringError("risk_research.series_formation_coverage_invalid")
        executions = []
        for input_binding, bounded_sessions, surface in inputs:
            execution = RiskDevelopmentExecutor(estimators=self._estimators).execute(
                binding=binding,
                input_binding=input_binding,
                admission=compiled.admission,
                return_surface=surface,
                return_reader=self._return_reader,
                bounded_sessions=bounded_sessions,
                output_workspace=output_workspace,
                sector_by_listing_id=sector_subset(
                    self._sector_by_listing_id, input_binding.ordered_listing_ids
                ),
                counter=NumericalCallCounter(),
                cancel_requested=self._cancel_requested,
                reserve_output=self._reserve_output,
            )
            executions.append(execution)
            if recorder is not None:
                for _call in range(execution.estimate_calls):
                    recorder.record(capability=binding.recipe_hash)
        if authority.listing_scopes:
            members = tuple(
                RiskSeriesMember(
                    input_binding_hash=run.input_binding.input_binding_hash,
                    surface_hash=run.build.surface.surface_hash,
                    diagnostics_hash=run.build.diagnostics.diagnostics_hash,
                )
                for run in executions
            )
            draft = RiskDevelopmentSeries.model_construct(
                program_hash=program.program_hash,
                authority=authority,
                members=members,
                diagnostics_hash="0" * 64,
                series_hash="0" * 64,
            )
            diagnostics = combined_diagnostics(
                input_binding_hash=draft.input_binding_hash,
                children=tuple(run.build.diagnostics for run in executions),
            )
            series = RiskDevelopmentSeries.create(
                program_hash=program.program_hash,
                authority=authority,
                members=members,
                diagnostics_hash=diagnostics.diagnostics_hash,
            )
            store = RiskArtifactStore(output_workspace)
            store.publish_json(
                category=DEVELOPMENT_DIAGNOSTICS_CATEGORY,
                payload=diagnostics.model_dump(mode="json"),
                identity_field="diagnostics_hash",
            )
            store.publish_json(
                category=SERIES_CATEGORY,
                payload=series.model_dump(mode="json"),
                identity_field="series_hash",
            )
            return DeskExecutionResult(
                disposition="COMPUTED",
                artifact_uris=(store.uri(SERIES_CATEGORY, series.series_hash),),
                formation_sessions=authority.sessions,
                numerical_call_count=sum(run.estimate_calls for run in executions),
                desk_input_binding_hash=series.input_binding_hash,
            )
        execution = executions[0]
        return DeskExecutionResult(
            disposition="COMPUTED",
            # Named the surface and its diagnostics only, so the dossier and the
            # input binding could be deleted or edited and replay still reported
            # exact reuse.
            artifact_uris=execution.artifact_uris,
            formation_sessions=tuple(
                date.fromisoformat(value) for value in execution.formation_sessions
            ),
            # Observed at the adapter, so a resumed build reports only the work
            # it actually did rather than the length of the surface it produced.
            numerical_call_count=execution.estimate_calls,
            desk_input_binding_hash=execution.input_binding.input_binding_hash,
        )


__all__ = [
    "RiskDevelopmentExecution",
    "RiskDevelopmentExecutor",
    "RiskExperimentExecutor",
]
