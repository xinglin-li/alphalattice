"""Long-lived host for the governed pre-Factor product workspace.

This is application composition, not another workflow framework.  It drives
the already-qualified deterministic owners, persists every data effect in the
workspace DuckDB, and emits operational timing which never participates in a
research-content hash.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from alphalattice.control.data_platform.maintenance.contracts import (
    MaintenanceTrigger,
    WorkspaceMaintenanceRequest,
)
from alphalattice.control.data_platform.preflight import (
    DataTruthPreflightError,
    DataTruthPreflightRequest,
    DataTruthPreflightResult,
    DataTruthScopeUnavailable,
    blocked_preflight_result,
    routine_preflight_formation_scope,
    run_data_truth_preflight,
)
from alphalattice.control.data_platform.readiness import (
    SourceLoader,
    WorkspaceConsentAction,
    WorkspaceReadinessConsent,
    WorkspaceReadinessStatus,
    _latest_common_us_session,
    build_workspace_readiness,
)
from alphalattice.control.product_host.composition.workspace import WorkspaceRuntime
from alphalattice.control.product_host.research_authoring.feature_activations import (
    workspace_feature_catalog,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.contracts import canonical_hash
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.sources.manifest import current_index_profile_id
from alphalattice.foundation.market_data_ops.sources.providers import (
    MarketDataProvider,
    YFinanceMarketDataProvider,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.foundation.research_foundation.runtime.delta import (
    PreResearchObservedDelta,
    PreResearchRevisionDisposition,
    PreResearchUpdatePlan,
)
from alphalattice.interface.local_application.failure_codes import public_failure


@dataclass(frozen=True)
class ProductWorkspacePaths:
    """Name the confined local workspace database, artifacts, reports and source cache."""

    root: Path

    @classmethod
    def at(cls, root: Path) -> ProductWorkspacePaths:
        """Resolve an explicit caller-owned local workspace root.

        Args:
            root: Explicit workspace path.

        Returns:
            Resolved workspace path declaration.
        """
        return cls(Path(root).resolve())

    @property
    def database(self) -> Path:
        """Resolve the market-data.duckdb database beneath this workspace.

        Returns:
            Declared workspace-local path.
        """
        return self.root / "market-data.duckdb"

    @property
    def reports(self) -> Path:
        """Resolve the reports directory beneath this workspace.

        Returns:
            Declared workspace-local path.
        """
        return self.root / "reports"

    @property
    def artifacts(self) -> Path:
        """Resolve the artifacts directory beneath this workspace.

        Returns:
            Declared workspace-local path.
        """
        return self.root / "artifacts"

    @property
    def cache(self) -> Path:
        """Resolve the yfinance cache directory beneath this workspace.

        Returns:
            Declared workspace-local path.
        """
        return self.root / "yfinance-cache"

    def ensure(self) -> None:
        """Create the declared local workspace, report, artifact, cache and staging directories."""
        for path in (
            self.root,
            self.reports,
            self.artifacts,
            self.cache,
            self.root / "staging",
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PreFactorHostOutcome:
    """Retain onboarding/maintenance status, exact report and explicit failure/retry facts."""

    status: str
    phase: str
    run_id: str
    report_path: Path
    failure_code: str | None = None
    retry_after_at: datetime | None = None
    manifest_revision: str | None = None
    snapshot_ref: str | None = None
    data_truth_preflight: DataTruthPreflightResult | None = None
    """The Data-truth verdict this run actually produced.

    Surfaced on the Host's own outcome rather than only inside a case study: an
    advisory nobody can see from the product is not evidence, and a blocking
    verdict must be visible where the decision is made.
    """


class _TimingAccumulator:
    def __init__(self) -> None:
        self.elapsed: dict[str, float] = defaultdict(float)
        self.metrics: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def record(self, stage: str, elapsed: float, metrics: dict[str, object] | Any) -> None:
        self.elapsed[stage] += elapsed
        for key, value in dict(metrics).items():
            if isinstance(value, (int, float)):
                self.metrics[stage][key] += float(value)


@dataclass
class PreFactorWorkspaceHost:
    """Compose explicit source readiness, consent, onboarding and maintenance owners."""

    paths: ProductWorkspacePaths
    profile_path: Path
    diagnose: Any | None = None
    provider: MarketDataProvider | None = None
    source_loader: SourceLoader | None = None

    def run(
        self,
        *,
        initialize_workspace: bool,
        run_until_terminal: bool,
        full_history_symbols: tuple[str, ...] = (),
        observed_at: datetime | None = None,
    ) -> PreFactorHostOutcome:
        """Run admitted source onboarding and maintenance within explicit consent and work bounds.

        A blocked pre-research delta or truth preflight prevents research readiness. The composed
        runtime is closed after maintenance, including on failure.

        Args:
            initialize_workspace: Whether exact initialization/manifest-refresh consent is granted.
            run_until_terminal: Whether to continue deterministic work until terminal outcome.
            full_history_symbols: Explicit full-history source selections.
            observed_at: Optional explicit timezone-aware observation timestamp.

        Returns:
            Consent/refusal/running or completed outcome with exact report, manifest/snapshot and
            truth preflight.

        Raises:
            ValueError: Observation time, frozen membership candidate or exact confirmation scope is
                invalid.
        """
        now = observed_at or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("pre-factor host clock must be timezone-aware")
        self.paths.ensure()
        run_id = canonical_hash(
            {
                "kind": "pre-factor-onboarding-run",
                "workspace": str(self.paths.root),
                "started_at": now.astimezone(UTC).isoformat(),
            }
        )
        market_data = MarketDataRepository(self.paths.root)
        gate = build_workspace_readiness(
            market_data,
            profile_path=self.profile_path,
            provider=self.provider,
            source_loader=self.source_loader,
        )
        feature_state, panel_state, provider = gate.feature_state, gate.panel_state, gate.provider
        timings = _TimingAccumulator()
        total_started = time.perf_counter()
        market_profile_id = current_index_profile_id(self.profile_path)
        local_delta: PreResearchObservedDelta | None = None
        local_plan: PreResearchUpdatePlan | None = None
        readiness_record = market_data.readiness.load(market_profile_id)
        if (
            readiness_record is not None
            and readiness_record.active_manifest_id is not None
            and readiness_record.active_candidate_manifest_document is not None
        ):
            local_delta, local_plan = self._assess_local_pre_research_delta(
                market_data=market_data,
                provider=provider,
                candidate_document=readiness_record.active_candidate_manifest_document,
                target_market_session=_latest_common_us_session(
                    on_or_before=now.date(), observed_at=now
                ),
                observed_at=now,
                timings=timings,
            )
            if local_delta.revision_disposition is PreResearchRevisionDisposition.BLOCKED:
                return self._finish(
                    market_data,
                    feature_state,
                    panel_state,
                    timings,
                    run_id=run_id,
                    total_started=total_started,
                    status="BLOCKED",
                    phase="pre_research_delta_gate",
                    failure_code="pre_research_delta.local_observation_blocked",
                    pre_research_revision=self._delta_summary(local_delta, local_plan),
                )
        discovery_started = time.perf_counter()
        decision = gate.refresh_sources_if_due(observed_at=now)
        timings.record(
            "candidate_discovery",
            time.perf_counter() - discovery_started,
            {"candidate_count": len(decision.manifest.listings) if decision.manifest else 0},
        )

        if decision.status is WorkspaceReadinessStatus.INITIALIZATION_REQUIRED:
            if not initialize_workspace:
                return self._finish(
                    market_data,
                    feature_state,
                    panel_state,
                    timings,
                    run_id=run_id,
                    total_started=total_started,
                    status="CONSENT_REQUIRED",
                    phase="candidate_discovery",
                    failure_code="workspace_readiness.initialization_consent_required",
                )
            discovery_started = time.perf_counter()
            onboarding = gate.start_approved_onboarding(
                WorkspaceReadinessConsent(
                    consent_id=uuid4(),
                    market_profile_id=current_index_profile_id(self.profile_path),
                    action=WorkspaceConsentAction.INITIALIZE,
                    approved_at=now,
                )
            )
            timings.record(
                "candidate_discovery",
                time.perf_counter() - discovery_started,
                {"candidate_count": len(onboarding.runner.acquisition_manifest.listings)},
            )
        elif decision.status is WorkspaceReadinessStatus.MANIFEST_UPDATE_PENDING:
            if not initialize_workspace:
                return self._finish(
                    market_data,
                    feature_state,
                    panel_state,
                    timings,
                    run_id=run_id,
                    total_started=total_started,
                    status="CONSENT_REQUIRED",
                    phase="candidate_discovery",
                    failure_code="workspace_readiness.manifest_refresh_consent_required",
                    pre_research_revision=self._delta_summary(local_delta, local_plan),
                )
            pending = market_data.readiness.load(market_profile_id)
            if (
                pending is None
                or pending.pending_candidate_manifest_document is None
                or pending.active_manifest_id is None
            ):
                raise ValueError("pending membership candidate is not frozen")
            pending_delta, pending_plan = self._assess_local_pre_research_delta(
                market_data=market_data,
                provider=provider,
                candidate_document=pending.pending_candidate_manifest_document,
                target_market_session=_latest_common_us_session(
                    on_or_before=now.date(), observed_at=now
                ),
                observed_at=now,
                timings=timings,
            )
            if not pending_plan.user_confirmation_required:
                raise ValueError("membership transition did not require exact confirmation")
            self._confirm_local_pre_research_plan(
                market_data=market_data,
                provider=provider,
                plan=pending_plan,
                observed_at=now,
                confirmation_token=run_id,
            )
            local_delta, local_plan = pending_delta, pending_plan
            discovery_started = time.perf_counter()
            onboarding = gate.start_approved_onboarding(
                WorkspaceReadinessConsent(
                    consent_id=uuid4(),
                    market_profile_id=current_index_profile_id(self.profile_path),
                    action=WorkspaceConsentAction.REFRESH_MANIFEST,
                    approved_at=now,
                )
            )
            timings.record(
                "candidate_discovery",
                time.perf_counter() - discovery_started,
                {"candidate_count": len(onboarding.runner.acquisition_manifest.listings)},
            )
        elif decision.status is WorkspaceReadinessStatus.ONBOARDING_IN_PROGRESS:
            onboarding = gate.resume_onboarding()
        else:
            onboarding = None

        if onboarding is not None:
            onboarding.runner.timing_sink = timings.record
            raw_started = time.perf_counter()
            outcome = onboarding.runner.run(
                observed_at=now,
                work_budget=None if run_until_terminal else 25,
            )
            timings.record(
                "raw_onboarding_total",
                time.perf_counter() - raw_started,
                {
                    "candidate_count": outcome.candidates,
                    "raw_ready": outcome.raw_ready,
                    "feature_ready": outcome.feature_ready,
                    "failed": outcome.failed,
                },
            )
            if outcome.status.value in {"deferred", "blocked", "running"}:
                return self._finish(
                    market_data,
                    feature_state,
                    panel_state,
                    timings,
                    run_id=run_id,
                    total_started=total_started,
                    status=outcome.status.value.upper(),
                    phase="raw_action_onboarding",
                    failure_code=outcome.failure_code,
                    retry_after_at=outcome.retry_after_at,
                )
            decision = gate.complete_onboarding(onboarding, outcome, observed_at=now)

        if decision.status is WorkspaceReadinessStatus.RESEARCH_READY and (
            local_delta is None
            or local_delta.revision_disposition
            in {
                PreResearchRevisionDisposition.NOOP,
                PreResearchRevisionDisposition.SOURCE_CHANGED_ACTIVE_SET_UNCHANGED,
            }
        ):
            assert decision.manifest is not None
            snapshot = panel_state.feature_panel_snapshot_for_active(
                decision.manifest.profile.market_profile_id
            )
            return self._finish(
                market_data,
                feature_state,
                panel_state,
                timings,
                run_id=run_id,
                total_started=total_started,
                status="NOOP",
                phase="completed",
                manifest_revision=decision.manifest.revision_sha256,
                snapshot_ref=str(snapshot["manifest_uri"]) if snapshot else None,
                pre_research_revision=self._delta_summary(local_delta, local_plan),
            )
        if decision.status not in {
            WorkspaceReadinessStatus.FEATURE_BUILDING,
            WorkspaceReadinessStatus.RESEARCH_READY,
        }:
            return self._finish(
                market_data,
                feature_state,
                panel_state,
                timings,
                run_id=run_id,
                total_started=total_started,
                status="BLOCKED",
                phase="preflight",
                failure_code=decision.failure_code or "workspace_readiness.not_feature_buildable",
            )
        assert decision.manifest is not None

        runtime = WorkspaceRuntime.create(
            workspace=self.paths.root,
            manifest=decision.manifest,
            provider=provider,
            artifact_root=self.paths.artifacts,
            feature_catalog=workspace_feature_catalog(self.paths.root),
            max_live_symbols=1_000,
        )
        try:
            coordinator = runtime.maintenance_coordinator(
                readiness_gate=gate,
                diagnose=self.diagnose,
            )
            listing_ids = self._resolve_full_history_symbols(
                decision.manifest, full_history_symbols
            )
            target_session = _latest_common_us_session(on_or_before=now.date(), observed_at=now)
            request = WorkspaceMaintenanceRequest.create(
                market_profile_id=decision.manifest.profile.market_profile_id,
                target_market_session=target_session,
                knowledge_cutoff_at=now,
                trigger=(
                    MaintenanceTrigger.USER_REQUEST
                    if listing_ids
                    else MaintenanceTrigger.STARTUP
                    if decision.status is WorkspaceReadinessStatus.RESEARCH_READY
                    else MaintenanceTrigger.ONBOARDING
                ),
                membership_revision=decision.manifest.revision_sha256,
                data_policy_hash=canonical_hash(
                    {
                        "rolling_days": 45,
                        "daily_bar_settlement_delay_minutes": 120,
                        "historical_revision_detection": (
                            "EVIDENCE_TRIGGERED_NO_PROVIDER_CHANGE_FEED"
                        ),
                        "transport": "chunk25-workers2-to1-backoff5-15-45",
                    }
                ),
                feature_policy_hash=canonical_hash(
                    {
                        "catalog": runtime.feature_foundation.catalog.binding.catalog_hash,
                        "invalidation": "domain-topology",
                        "snapshot": "annual-content-addressed-zstd",
                    }
                ),
                full_history_listing_ids=listing_ids,
            )
            runtime.prepare_feature_closure()
            feature_started = time.perf_counter()
            maintenance = coordinator.run(request, observed_at=now)
            while run_until_terminal and maintenance.status.value == "running":
                maintenance = coordinator.run(
                    request,
                    observed_at=datetime.now(UTC),
                )
            timings.record(
                "feature_foundation_and_snapshot",
                time.perf_counter() - feature_started,
                {"child_task_count": len(maintenance.child_task_refs)},
            )
            snapshot = runtime.panel_state.feature_panel_snapshot_for_active(
                decision.manifest.profile.market_profile_id
            )
            preflight_started = time.perf_counter()
            preflight = self._run_data_truth_preflight(
                market_profile_id=decision.manifest.profile.market_profile_id,
                as_of_session=target_session,
                now=now,
            )
            timings.record(
                "data_truth_preflight",
                time.perf_counter() - preflight_started,
                {"advisory_count": len(preflight.advisory_codes) if preflight else 0},
            )
            status = maintenance.status.value.upper()
            failure_code = maintenance.failure_code
            if preflight is not None and preflight.blocks_research_ready:
                # A scope the qualified calendar does not admit is not a
                # research-ready workspace, whatever maintenance concluded.
                status = "BLOCKED"
                failure_code = preflight.blocking_failure_code
            return self._finish(
                runtime.market_data,
                runtime.feature_state,
                runtime.panel_state,
                timings,
                run_id=run_id,
                total_started=total_started,
                status=status,
                phase=maintenance.phase.value,
                failure_code=failure_code,
                retry_after_at=maintenance.retry_after_at,
                manifest_revision=decision.manifest.revision_sha256,
                snapshot_ref=str(snapshot["manifest_uri"]) if snapshot else None,
                pre_research_revision=self._delta_summary(local_delta, local_plan),
                data_truth_preflight=preflight,
            )
        finally:
            runtime.close()

    def _run_data_truth_preflight(
        self,
        *,
        market_profile_id: str,
        as_of_session: date,
        now: datetime,
    ) -> DataTruthPreflightResult | None:
        """Run the Data-truth preflight over a small matured formation scope.

        The Host resolves the scope and every authority itself and submits only
        a request; nothing here accepts a caller-computed hash or verdict.

        Two failures that look alike are kept apart on purpose. A workspace that
        cannot yet express a matured formation axis has genuinely nothing to
        compare and yields no verdict. Every other failure -- a missing research
        manifest, a session with no qualified members, an uninstalled policy --
        means the evidence should have existed and its authority could not be
        resolved. Collapsing those into "no verdict" would let a resolution
        failure read exactly like a clean run, so they seal a typed blocking
        result instead.
        """

        try:
            formation_sessions = routine_preflight_formation_scope(
                as_of_session=as_of_session,
                as_of_timestamp=now,
            )
        except DataTruthScopeUnavailable:
            return None
        request = DataTruthPreflightRequest(
            workspace=self.paths.root,
            market_profile_id=market_profile_id,
            formation_sessions=formation_sessions,
        )
        try:
            return run_data_truth_preflight(request, as_of_timestamp=now)
        except DataTruthScopeUnavailable:
            return None
        except DataTruthPreflightError as error:
            return blocked_preflight_result(
                request=request,
                failure_code=public_failure(error, "data_truth.preflight_refused"),
            )

    @staticmethod
    def _resolve_full_history_symbols(manifest, symbols: tuple[str, ...]) -> tuple[str, ...]:
        listing_ids = []
        for symbol in tuple(sorted(set(item.strip().upper() for item in symbols))):
            listing = manifest.listing_for_symbol(symbol)
            if listing is None:
                raise ValueError(f"full-history audit symbol is not active: {symbol}")
            listing_ids.append(listing.listing_id)
        return tuple(listing_ids)

    def _assess_local_pre_research_delta(
        self,
        *,
        market_data: MarketDataRepository,
        provider: YFinanceMarketDataProvider,
        candidate_document: dict[str, object],
        target_market_session: date,
        observed_at: datetime,
        timings: _TimingAccumulator,
    ) -> tuple[PreResearchObservedDelta, PreResearchUpdatePlan]:
        readiness = market_data.readiness.load(current_index_profile_id(self.profile_path))
        if readiness is None or readiness.active_manifest_id is None:
            raise ValueError("active manifest is required for local delta assessment")
        runtime = WorkspaceRuntime.create(
            workspace=self.paths.root,
            manifest=market_data.load_universe_manifest(readiness.active_manifest_id),
            provider=provider,
            artifact_root=self.paths.artifacts,
            feature_catalog=workspace_feature_catalog(self.paths.root),
            max_live_symbols=1_000,
        )
        started = time.perf_counter()
        try:
            service = runtime.pre_research_delta_service(profile_path=self.profile_path)
            service.bootstrap_verified_head(
                market_profile_id=readiness.market_profile_id,
                verified_at=observed_at,
            )
            _candidate, delta, plan = service.assess(
                candidate_document=candidate_document,
                target_market_session=target_market_session,
                observed_at=observed_at,
            )
            return delta, plan
        finally:
            timings.record(
                "pre_research_delta_gate",
                time.perf_counter() - started,
                {"provider_calls": 0},
            )
            runtime.close()

    def _confirm_local_pre_research_plan(
        self,
        *,
        market_data: MarketDataRepository,
        provider: YFinanceMarketDataProvider,
        plan: PreResearchUpdatePlan,
        observed_at: datetime,
        confirmation_token: str,
    ) -> None:
        readiness = market_data.readiness.load(current_index_profile_id(self.profile_path))
        if readiness is None or readiness.active_manifest_id is None:
            raise ValueError("active manifest is required for delta confirmation")
        runtime = WorkspaceRuntime.create(
            workspace=self.paths.root,
            manifest=market_data.load_universe_manifest(readiness.active_manifest_id),
            provider=provider,
            artifact_root=self.paths.artifacts,
            feature_catalog=workspace_feature_catalog(self.paths.root),
            max_live_symbols=1_000,
        )
        try:
            runtime.pre_research_delta_service(profile_path=self.profile_path).confirm_plan(
                plan=plan,
                confirmed_at=observed_at,
                confirmation_token=confirmation_token,
            )
        finally:
            runtime.close()

    @staticmethod
    def _delta_summary(
        delta: PreResearchObservedDelta | None,
        plan: PreResearchUpdatePlan | None,
    ) -> dict[str, object] | None:
        if delta is None or plan is None:
            return None
        return {
            "base_head_hash": delta.base_head_hash,
            "candidate_snapshot_hash": delta.candidate_snapshot_hash,
            "delta_hash": delta.delta_hash,
            "update_plan_hash": plan.update_plan_hash,
            "disposition": delta.revision_disposition.value,
            "additions": list(delta.additions),
            "removals": list(delta.removals),
            "full_history_addition_count": len(plan.full_history_addition_listing_ids),
            "bounded_retained_count": len(plan.bounded_retained_listing_ids),
            "user_confirmation_required": plan.user_confirmation_required,
        }

    def _finish(
        self,
        market_data: MarketDataRepository,
        feature_state: FeatureStateRepository,
        panel_state: PanelStateRepository,
        timings: _TimingAccumulator,
        *,
        run_id: str,
        total_started: float,
        status: str,
        phase: str,
        failure_code: str | None = None,
        retry_after_at: datetime | None = None,
        manifest_revision: str | None = None,
        snapshot_ref: str | None = None,
        pre_research_revision: dict[str, object] | None = None,
        data_truth_preflight: DataTruthPreflightResult | None = None,
    ) -> PreFactorHostOutcome:
        timings.record("total_wall", time.perf_counter() - total_started, {})
        database_bytes = self.paths.database.stat().st_size if self.paths.database.exists() else 0
        artifact_bytes = sum(
            path.stat().st_size for path in self.paths.artifacts.rglob("*") if path.is_file()
        )
        payload = {
            "run_id": run_id,
            "status": status,
            "phase": phase,
            "failure_code": failure_code,
            "retry_after_at": retry_after_at,
            "manifest_revision": manifest_revision,
            "snapshot_ref": snapshot_ref,
            "database_bytes": database_bytes,
            "artifact_bytes": artifact_bytes,
            "historical_revision_detection": "EVIDENCE_TRIGGERED_NO_PROVIDER_CHANGE_FEED",
            "pre_research_revision": pre_research_revision,
            # Written into the durable report, not only returned in memory: an
            # advisory that survives only inside the process is not evidence.
            "data_truth_preflight": (
                {
                    "result_hash": data_truth_preflight.result_hash,
                    "request_scope_hash": data_truth_preflight.request_scope_hash,
                    "session_authority_hash": data_truth_preflight.session_authority_hash,
                    "divergence_policy_hash": data_truth_preflight.divergence_policy_hash,
                    "disposition": data_truth_preflight.disposition,
                    "blocking_failure_code": data_truth_preflight.blocking_failure_code,
                    "advisory_codes": list(data_truth_preflight.advisory_codes),
                    "divergence": (
                        {
                            "classification": data_truth_preflight.divergence.classification,
                            "attribution": data_truth_preflight.divergence.attribution,
                            "session_count": data_truth_preflight.divergence.session_count,
                            "largest_absolute_divergence": (
                                data_truth_preflight.divergence.largest_absolute_divergence
                            ),
                            "diagnostic_hash": (data_truth_preflight.divergence.diagnostic_hash),
                        }
                        if data_truth_preflight.divergence is not None
                        else None
                    ),
                }
                if data_truth_preflight is not None
                else None
            ),
            "workspace_summary": self._workspace_summary(
                market_data,
                feature_state,
                panel_state,
            ),
            "stages": [
                {
                    "stage": stage,
                    "elapsed_seconds": timings.elapsed[stage],
                    "metrics": dict(timings.metrics[stage]),
                }
                for stage in sorted(timings.elapsed)
            ],
        }
        self.paths.reports.mkdir(parents=True, exist_ok=True)
        report = self.paths.reports / f"pre-factor-onboarding-{run_id}.json"
        staged = report.with_suffix(".json.tmp")
        staged.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str),
            encoding="utf-8",
        )
        staged.replace(report)
        return PreFactorHostOutcome(
            status=status,
            phase=phase,
            run_id=run_id,
            report_path=report,
            failure_code=failure_code,
            retry_after_at=retry_after_at,
            manifest_revision=manifest_revision,
            snapshot_ref=snapshot_ref,
            data_truth_preflight=data_truth_preflight,
        )

    def _workspace_summary(
        self,
        market_data: MarketDataRepository,
        feature_state: FeatureStateRepository,
        panel_state: PanelStateRepository,
    ) -> dict[str, object]:
        """Build the path-free, bounded audit projection used by runtime reports."""

        market_profile_id = current_index_profile_id(self.profile_path)
        readiness = market_data.readiness.load(market_profile_id)
        candidate_document = (
            readiness.active_candidate_manifest_document if readiness is not None else None
        )
        candidates = (
            candidate_document.get("candidates", []) if isinstance(candidate_document, dict) else []
        )
        summary: dict[str, object] = {
            # The stored value as recorded, with its time: this report is written as the run
            # ends and keeps no live assessment, which the data update's input readback gives
            # (V107).
            "stored_readiness": readiness.status if readiness is not None else None,
            "stored_readiness_recorded_at": (
                readiness.updated_at.isoformat() if readiness is not None else None
            ),
            "candidate_listing_count": len(candidates) if isinstance(candidates, list) else 0,
            "candidate_manifest_hash": (
                candidate_document.get("content_hash")
                if isinstance(candidate_document, dict)
                else None
            ),
            "initial_onboarding": market_data.latest_current_universe_onboarding_disclosure(
                market_profile_id=market_profile_id
            ),
        }
        if readiness is None or readiness.active_manifest_id is None:
            return summary

        manifest = market_data.load_universe_manifest(readiness.active_manifest_id)
        active_count = len(manifest.listings)
        summary.update(
            {
                "active_manifest_revision": manifest.revision_sha256,
                "active_listing_count": active_count,
                "total_excluded_listing_count": max(
                    0,
                    int(summary["candidate_listing_count"]) - active_count,
                ),
                "universe_policy": {
                    "type": manifest.universe_policy_type,
                    "components": list(manifest.universe_components),
                    "survivorship_bias_warning": manifest.survivorship_bias_warning,
                    "research_use_class": manifest.research_use_class,
                },
                "quality_governance": panel_state.feature_input_quality_disclosure(
                    result_manifest_revision=manifest.revision_sha256
                ),
            }
        )
        sector = feature_state.current_sector_state(manifest)
        if sector is not None:
            summary["sector_revision"] = sector.sector_revision
            summary["sector_distribution"] = sector.sector_distribution

        snapshot = panel_state.feature_panel_snapshot_for_active(market_profile_id)
        if snapshot is not None:
            manifest_uri = str(snapshot["manifest_uri"])
            summary["snapshot_ref"] = manifest_uri
            summary["snapshot_hash"] = str(snapshot["snapshot_hash"])
            summary["panel_content_hash"] = str(snapshot["panel_content_hash"])
            summary["feature_panel"] = ArtifactResolver(
                self.paths.artifacts
            ).inspect_feature_panel_snapshot(manifest_uri)
        return summary
