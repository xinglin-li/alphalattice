"""Bind installed Risk research to one sealed local input, never today's workspace."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.research_authoring import (
    build_research_program_workflow,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceExperimentInput,
)
from alphalattice.control.product_host.research_authoring.authority import (
    WorkspaceResearchAuthorityResolver,
)
from alphalattice.control.product_host.research_authoring.execution import (
    WorkspaceReturnSurfaceProvider,
    _sector_by_listing_id,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    PROFILE,
    PreparedInputAuthority,
    factor_input_paths,
    input_experiment_envelope,
    normalize_input_document,
    read_factor_bundle,
)
from alphalattice.control.product_host.storage.inventory import (
    StorageInventoryError,
    require_storage_capacity,
)
from alphalattice.control.research_program.authoring.workflow import ResearchProgramWorkflow
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.contracts import PanelSourceExclusion
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.investment.risk_research.contracts import CausalRiskReturnSurface
from alphalattice.investment.risk_research.estimators.capability import CANONICAL_NO_RANDOMNESS_SEED
from alphalattice.investment.risk_research.estimators.catalog import (
    build_installed_risk_estimator_catalog,
)
from alphalattice.investment.risk_research.estimators.domains import COVARIANCE_RECIPE_SCHEMA_ID
from alphalattice.investment.risk_research.experiments.compiler import (
    RISK_EXPERIMENT_KIND,
    RiskExperimentCompiler,
)
from alphalattice.investment.risk_research.experiments.development import RiskDevelopmentCancelled
from alphalattice.investment.risk_research.experiments.execution import RiskExperimentExecutor
from alphalattice.investment.risk_research.experiments.formation_selection import (
    FormationSelectionPolicy,
    eligible_formation_sessions,
    select_formation_sessions,
)
from alphalattice.investment.risk_research.experiments.policy import enforce_execution_policy
from alphalattice.investment.risk_research.experiments.window import (
    REQUIRED_LOOKBACK_SESSIONS,
    REQUIRED_NEXT_SESSIONS,
    ReturnSurfaceFreshnessProbe,
)
from alphalattice.investment.risk_research.surfaces.returns import (
    CausalRiskReturnReader,
    CausalRiskReturnSurfacePublisher,
)
from alphalattice.kernel.quant.sector_history import reclassification_payload
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.sector_treatment import SECTOR_HISTORY_FORWARD
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExperimentEnvelope,
    ResolvedListingScope,
    ResolvedResearchAuthority,
)

ROOT = resolve_playpen_root(Path(__file__))

_CANCEL_PROBE_EVERY = 32
"""Listings between two cancellation probes while causal returns are prepared.

One probe is one Task-registry read (about ten milliseconds); a 466-listing
input reported 466 of them, which was half of its preparation. A cancellation
is observed at a probe or, after the last probe, at the first chunk boundary
of the estimation build; a return surface published in between is kept.
"""


def _capacity(session: WorkspaceApplicationSession, additional: int) -> None:
    try:
        require_storage_capacity(
            session.workspace,
            additional_bytes=additional,
        )
    except StorageInventoryError as error:
        raise AuthoringError(error.failure_code) from error


@dataclass
class _ReturnInputs:
    session: WorkspaceApplicationSession
    source: Path
    artifact_root: Path
    panel_ref: str
    listing_count: int
    session_count: int
    required_sessions: tuple[date, ...]
    required_listing_ids: tuple[str, ...]
    reader: CausalRiskReturnReader
    cancelled: Callable[[], bool] | None = None

    def published_surfaces(self) -> tuple[CausalRiskReturnSurface, ...]:
        """Called only by admitted execution; PLAN cannot build returns."""
        return self._publish(self.required_sessions, self.required_listing_ids)

    def for_authority(
        self, authority: ResolvedResearchAuthority
    ) -> tuple[tuple[CausalRiskReturnSurface, ...], ReturnSurfaceFreshnessProbe]:
        first = self.required_sessions.index(authority.sessions[0]) - REQUIRED_LOOKBACK_SESSIONS - 1
        stop = self.required_sessions.index(authority.sessions[-1]) + REQUIRED_NEXT_SESSIONS + 1
        if first < 0 or stop > len(self.required_sessions):
            raise AuthoringError("risk_research.return_scope_outside_request")
        required = self.required_sessions[first:stop]
        surfaces = self._publish(required, authority.ordered_listing_ids)
        market = MarketDataRepository(self.source)
        manifest = market.load_universe_manifest_revision(authority.universe_revision_sha256)

        def freshness(*, through: date) -> str:
            return str(
                market.execution_source_watermark(
                    manifest, through=through, listing_ids=authority.ordered_listing_ids
                )["watermark_hash"]
            )

        return surfaces, freshness

    def _publish(
        self, required_sessions: tuple[date, ...], listing_ids: tuple[str, ...]
    ) -> tuple[CausalRiskReturnSurface, ...]:
        provider = WorkspaceReturnSurfaceProvider(self.artifact_root)
        try:
            surfaces = provider.published_surfaces()
        except AuthoringError as error:
            if str(error) != "research_authoring.return_surface_unavailable":
                raise
            surfaces = ()
        surfaces = tuple(
            surface
            for surface in surfaces
            if (
                surface.epoch.ordered_listing_ids == listing_ids
                and surface.first_formation_session == required_sessions[1]
                and surface.last_formation_session == required_sessions[-1]
            )
        )
        if not surfaces:
            _capacity(
                self.session,
                len(listing_ids) * len(required_sessions) * 256,
            )

            probes = 0

            def progress(_message: str) -> None:
                # The publisher reports every listing; the cancellation probe
                # reads the Task registry, so it is consulted on the first
                # listing and every 32nd after it rather than on all of them.
                nonlocal probes
                probes += 1
                if (
                    self.cancelled is not None
                    and (probes == 1 or probes % _CANCEL_PROBE_EVERY == 0)
                    and self.cancelled()
                ):
                    raise RiskDevelopmentCancelled(
                        "risk_research.cancelled_during_input_preparation"
                    )

            surface, _ = CausalRiskReturnSurfacePublisher(
                store=MarketDataRepository(self.source),
                resolver=ArtifactResolver(self.source / "artifacts"),
                artifact_root=self.artifact_root,
                mutation_gate=self.session.mutation_gate,
                progress=progress,
            ).publish(
                panel_manifest_ref=self.panel_ref,
                market_profile_id=PROFILE,
                required_sessions=required_sessions,
                required_listing_ids=listing_ids,
            )
            surfaces = (surface,)
        # Proved through the execution's own reader, so the window reads that
        # follow reuse the proof instead of verifying every chunk again.
        for surface in surfaces:
            self.reader.verify_closure(surface.surface_hash)
        return surfaces


def risk_controls(
    session: WorkspaceApplicationSession, binding: ResearchWorkspaceExperimentInput
) -> dict[str, Any]:
    """Project installed Risk parameter choices and causal lookback/realized-session bounds.

    Args:
        session: Retained workspace task session.
        binding: Exact admitted research input.

    Returns:
        Risk draft/template and installed parameter/date controls with development-only claim
        limits.

    Raises:
        AuthoringError: Input history cannot support the required lookback and next outcome session.
    """
    import yaml  # type: ignore[import-untyped]

    bundle = read_factor_bundle(session.workspace, binding.binding_hash)
    eligible = eligible_formation_sessions(
        available=bundle.sessions[1:],
        lookback_sessions=REQUIRED_LOOKBACK_SESSIONS,
        next_sessions=REQUIRED_NEXT_SESSIONS,
    )
    if not eligible:
        raise AuthoringError("risk_research.insufficient_input_history")
    catalog = build_installed_risk_estimator_catalog()
    domain = catalog.parameter_domain(COVARIANCE_RECIPE_SCHEMA_ID)
    envelope = input_experiment_envelope(
        bundle,
        kind=RISK_EXPERIMENT_KIND,
        maximum_candidates=1,
        seed=CANONICAL_NO_RANDOMNESS_SEED,
    )
    template = {"experiment": envelope}
    envelope["sessions"].update(
        start=str(eligible[max(0, len(eligible) - 21)]), end=str(eligible[-1])
    )
    parameters = {axis.name: axis.default for axis in domain.axes if len(axis.admissible) > 1}
    template["risk"] = {
        "estimator": {"capability": domain.recipe_schema_id, "parameters": parameters}
    }
    controls = [
        {
            "path": ["experiment", "sessions", key],
            "label": label,
            "type": "date",
            "value": envelope["sessions"][key],
            "min": str(eligible[0]),
            "max": str(eligible[-1]),
        }
        for key, label in (
            ("start", "First evaluated formation"),
            ("end", "Last evaluated formation"),
        )
    ]
    controls.extend(
        {
            "path": ["risk", "estimator", "parameters", axis.name],
            "label": axis.name.replace("_", " "),
            "type": "select",
            "options": list(axis.admissible),
            "value": axis.default,
            "help": "Only the installed Risk parameter domain is admitted.",
        }
        for axis in domain.axes
        if len(axis.admissible) > 1
    )
    controls.append(
        {
            "path": ["experiment", "sessions", "as_of", "session"],
            "label": "Diagnostic cutoff (official close)",
            "type": "date",
            "value": envelope["sessions"]["as_of"]["session"],
            "max": str(bundle.sessions[-1]),
            "help": "Must include the realized next session; no later observation is used.",
        }
    )
    return {
        "status": "READY",
        "input_id": binding.input_id,
        "input_binding_hash": binding.binding_hash,
        "method": RISK_EXPERIMENT_KIND,
        "template": template,
        "yaml": yaml.safe_dump(template, sort_keys=False),
        "controls": controls,
        "limits": [
            "DEVELOPMENT_DIAGNOSTICS_ONLY",
            "CURRENT_UNIVERSE_NON_PIT",
            "NO_PORTFOLIO_ALLOCATION_CHANGE",
        ],
    }


def risk_workflow(
    *,
    session: WorkspaceApplicationSession,
    binding: ResearchWorkspaceExperimentInput,
    document: Mapping[str, Any],
    cancelled: Callable[[], bool] | None = None,
) -> tuple[ResearchProgramWorkflow, ResolvedResearchAuthority, dict[str, Any], dict[str, Any]]:
    """Resolve exact historical Risk scope and compose its causal return/estimator workflow.

    Historical membership and Panel exclusions determine per-formation listing scopes. Required raw
    return history ends at the last admitted realized next session; the shared reader verifies the
    surfaces consumed by execution.

    Args:
        session: Retained workspace writer/task session.
        binding: Exact admitted research input.
        document: Explicit authored Risk declaration.
        cancelled: Optional explicit cancellation callback.

    Returns:
        Workflow, exact authority, statistical/source preparation preview and normalized
        declaration.

    Raises:
        AuthoringError: Official-close kind, realized cutoff, universe epoch, listing support or
            execution budget is inadmissible.
    """
    bundle = read_factor_bundle(session.workspace, binding.binding_hash)
    source, _ = factor_input_paths(session.workspace, binding.binding_hash)
    normalized = normalize_input_document(document, binding, bundle, section_name="risk")
    envelope = ResearchExperimentEnvelope.create(**normalized["experiment"])
    if (
        envelope.kind != RISK_EXPERIMENT_KIND
        or envelope.sessions.as_of.phase.name != "OFFICIAL_CLOSE"
    ):
        raise AuthoringError("risk_research.official_close_development_required")
    authority = WorkspaceResearchAuthorityResolver(workspace=source).resolve(envelope)
    available = bundle.sessions[1:]
    formations = select_formation_sessions(
        policy=FormationSelectionPolicy.bound_authority(),
        available=available,
        requested_sessions=authority.sessions,
        lookback_sessions=REQUIRED_LOOKBACK_SESSIONS,
        next_sessions=REQUIRED_NEXT_SESSIONS,
    )
    next_session = available[available.index(formations[-1]) + REQUIRED_NEXT_SESSIONS]
    if next_session > envelope.sessions.as_of.session:
        raise AuthoringError("risk_research.realized_diagnostic_after_cutoff")
    # One preceding raw open is needed for the first lookback return. A
    # historical Risk question must not demand unrelated observations after
    # its final realized session (for example from a later-delisted member).
    first_raw = bundle.sessions.index(formations[0]) - REQUIRED_LOOKBACK_SESSIONS - 1
    required_sessions = bundle.sessions[first_raw : bundle.sessions.index(next_session) + 1]
    market = MarketDataRepository(source)
    manifest = market.load_universe_manifest_revision(authority.universe_revision_sha256)
    if manifest.profile.market_profile_id != PROFILE:
        raise AuthoringError("risk_research.input_universe_epoch_mismatch")
    panel = ArtifactResolver(source / "artifacts").load_feature_panel_manifest(
        str(authority.panel_manifest_ref)
    )
    schedule = market.membership_schedule(
        PROFILE,
        sessions=formations,
        fallback_listing_ids=tuple(item.listing_id for item in manifest.listings),
    )
    exclusions = tuple(
        PanelSourceExclusion.from_payload(item)
        for item in panel.get("safe_summary", {}).get("membership", {}).get("source_exclusions", ())
    )
    scopes: list[ResolvedListingScope] = []
    for day in formations:
        eligible = set(schedule.members(day)) - {
            item.listing_id for item in exclusions if item.first_session <= day <= item.last_session
        }
        listings = tuple(
            listing for listing in authority.ordered_listing_ids if listing in eligible
        )
        if not listings:
            raise AuthoringError("risk_research.estimation_scope_empty")
        if scopes and scopes[-1].ordered_listing_ids == listings:
            prior = scopes.pop()
            scopes.append(
                ResolvedListingScope(
                    first_session=prior.first_session,
                    last_session=day,
                    ordered_listing_ids=listings,
                )
            )
        else:
            scopes.append(
                ResolvedListingScope(
                    first_session=day, last_session=day, ordered_listing_ids=listings
                )
            )
    selected_ids = {listing for scope in scopes for listing in scope.ordered_listing_ids}
    authority = ResolvedResearchAuthority.create(
        **{
            **authority.model_dump(exclude={"authority_hash"}),
            "ordered_listing_ids": tuple(
                v for v in authority.ordered_listing_ids if v in selected_ids
            ),
            "listing_scopes": tuple(scopes) if len(scopes) > 1 else (),
        }
    )
    sector_labels, _coverage_hash = _sector_by_listing_id(
        workspace=source,
        market_profile_id=PROFILE,
        panel_manifest=panel,
        listing_ids=authority.ordered_listing_ids,
    )
    sector_revision = str(panel["safe_summary"]["lineage"]["sector_revision"])
    source_hash = str(
        canonical_hash(
            {
                "input": binding.binding_hash,
                "sector": sector_revision,
                "sector_map": dict(sector_labels),
                **reclassification_payload(sector_labels),
                "return_sessions": (required_sessions[0], required_sessions[-1]),
                "risk_listing_ids": authority.ordered_listing_ids,
                **(
                    {"risk_listing_scopes": [scope.model_dump(mode="json") for scope in scopes]}
                    if len(scopes) > 1
                    else {}
                ),
            }
        )
    )
    authority = ResolvedResearchAuthority.create(
        **{
            **authority.model_dump(exclude={"authority_hash", "execution_input_binding_hash"}),
            "execution_input_binding_hash": source_hash,
        }
    )
    catalog = build_installed_risk_estimator_catalog()
    enforce_execution_policy(
        envelope=envelope,
        planned_numerical_calls=len(formations),
        adapter_bindings=tuple(v.describe_numerical_binding() for v in catalog.adapters),
    )
    artifact_root = session.workspace / "artifacts/research-risk-inputs" / source_hash

    def freshness(*, through: date) -> str:
        return str(
            market.execution_source_watermark(
                manifest, through=through, listing_ids=authority.ordered_listing_ids
            )["watermark_hash"]
        )

    # One reader for this workflow: the surfaces it proves for the Host are
    # the surfaces the executor reads.
    reader = CausalRiskReturnReader(artifact_root)
    return_inputs = _ReturnInputs(
        session,
        source,
        artifact_root,
        str(authority.panel_manifest_ref),
        len(authority.ordered_listing_ids),
        len(bundle.sessions),
        required_sessions,
        authority.ordered_listing_ids,
        reader,
        cancelled,
    )
    executor = RiskExperimentExecutor(
        surface_provider=return_inputs,
        scope_provider=return_inputs.for_authority,
        return_reader=reader,
        sector_by_listing_id=sector_labels,
        freshness_probe=freshness,
        estimators=catalog,
        reserve_output=lambda size: _capacity(session, size),
    )
    if cancelled is not None:
        executor.set_cancellation_check(cancelled)
    workflow = build_research_program_workflow(
        workspace=source,
        workspace_root=session.workspace,
        executors=(executor,),
        compilers=(RiskExperimentCompiler(catalog),),
        authority=PreparedInputAuthority(envelope, authority),
        verifier_kinds=(RISK_EXPERIMENT_KIND,),
    )
    preview = {
        "statistical_start": str(formations[0]),
        "statistical_end": str(formations[-1]),
        "statistical_session_count": len(formations),
        "latest_outcome_session": str(next_session),
        "listing_count": len(authority.ordered_listing_ids),
        "listing_scopes": [scope.model_dump(mode="json") for scope in scopes],
        "lookback_sessions": REQUIRED_LOOKBACK_SESSIONS,
        "sector_revision": sector_revision,
        "source_hash": source_hash,
        "expected_numerical_calls": len(formations),
        "interval_semantics": (
            "Exact evaluated formations with causal lookback and one realized next session."
        ),
        "source_preparation": "SEALED_LOCAL_RETURNS_SHARED_BY_INPUT",
        "limitations": [
            "POST_OBSERVED_DEVELOPMENT",
            "CURRENT_SECTOR_SNAPSHOT_NON_PIT"
            if not sector_labels.reclassifications
            else SECTOR_HISTORY_FORWARD,
        ],
    }
    return workflow, authority, preview, normalized
