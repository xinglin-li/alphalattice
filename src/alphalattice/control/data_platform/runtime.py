"""Deterministic owner for a small pre-Factor Research data preparation run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor, ArtifactResolver
from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.foundation.market_data_ops.publication.projection import FeatureAdmissionBlocked
from alphalattice.foundation.market_data_ops.runtime.refresh import normal_refresh_start
from alphalattice.foundation.market_data_ops.runtime.remediation import DataOperationsRequest
from alphalattice.foundation.market_data_ops.sources.contracts import FailureEvidence
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.sources.providers import (
    MarketDataProvider,
    ProviderFetchError,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import (
    CorruptedPayload,
    sanitize_payload,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository

DEFAULT_LIVE_SYMBOL_BUDGET = 5


class PreFactorStatus(StrEnum):
    """Classify Data preparation as ready, blocked or deferred before Factor research."""

    DATA_READY = "data_ready"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class PreFactorDataOutcome:
    """A safe parent-graph projection, not a provider payload or agent transcript."""

    request_id: str
    status: PreFactorStatus
    summary: str
    effective_as_of: date | None
    failure_code: str | None = None
    snapshot_path: Path | None = None
    snapshot_hash: str | None = None
    snapshot_artifact: ArtifactDescriptor | None = None
    attempts: int = 0
    retry_after_at: datetime | None = None


@dataclass
class PreFactorDataOperations:
    """Own one workspace's raw/action refresh and immutable feature-input freeze.

    This class intentionally contains no LLM call.  It owns the normal path and
    bounded retry; a later parent adapter can turn only the returned diagnostic
    outcome into a token-scoped Data Engineer case.
    """

    market_data: MarketDataRepository
    panel_state: PanelStateRepository
    manifest: UniverseManifest
    provider: MarketDataProvider
    snapshot_root: Path
    max_live_symbols: int = DEFAULT_LIVE_SYMBOL_BUDGET
    retry_budget: int = 2
    artifact_resolver: ArtifactResolver | None = None

    def prepare(
        self,
        request: DataOperationsRequest,
        *,
        observed_at: datetime | None = None,
    ) -> PreFactorDataOutcome:
        """Validate and freeze admitted Data inputs before Factor research begins.

        Args:
            request: Manifest-bound Data operation request, optionally carrying a frozen panel
                input.
            observed_at: Optional aware operational clock; omission uses the UTC product clock.

        Returns:
            A ready frozen input or a bounded blocked/deferred preparation outcome.

        Raises:
            ValueError: The supplied observation clock is naive.
        """
        now = observed_at or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        if request.manifest_revision != self.manifest.revision_sha256:
            return self._blocked(request, "data.manifest_revision_mismatch", None, attempts=0)
        if len(request.symbols) > self.max_live_symbols:
            return self._blocked(request, "data.live_symbol_budget_exceeded", None, attempts=0)
        listings = tuple(self.manifest.listing_for_symbol(symbol) for symbol in request.symbols)
        if any(item is None for item in listings):
            return self._blocked(request, "data.not_in_manifest", None, attempts=0)
        resolved_listings = tuple(item for item in listings if item is not None)
        if tuple(item.listing_id for item in resolved_listings) != request.listing_ids:
            return self._blocked(request, "data.listing_identity_mismatch", None, attempts=0)
        if request.panel_input is not None:
            return self._use_frozen_panel(request)
        if self.provider.name != self.manifest.profile.provider:
            return self._blocked(request, "data.provider_not_bound_to_manifest", None, attempts=0)

        self.market_data.bootstrap(self.manifest)
        fetch_start = self._refresh_start(request)
        fetched = self._fetch_with_bounded_retry(request, fetch_start, observed_at=now)
        if isinstance(fetched, PreFactorDataOutcome):
            return fetched
        payload, attempts = fetched
        try:
            batch = sanitize_payload(self.manifest, self.provider.name, payload, request.symbols)
        except CorruptedPayload as exc:
            return self._record_and_block(
                request,
                code=f"data.sanitizer.{exc.code.casefold()}",
                effective_as_of=None,
                attempts=attempts,
                observed_at=now,
            )

        latest_by_listing: dict[str, date] = {}
        for bar in batch.bars:
            latest_by_listing[bar.listing_id] = max(
                latest_by_listing.get(bar.listing_id, bar.session_date), bar.session_date
            )
        if set(latest_by_listing) != set(request.listing_ids):
            return self._record_and_block(
                request,
                code="data.partial_response",
                effective_as_of=None,
                attempts=attempts,
                observed_at=now,
            )
        effective_as_of = min(latest_by_listing.values())
        if effective_as_of < request.requested_range_start:
            return self._record_and_block(
                request,
                code="data.no_coverage_in_requested_range",
                effective_as_of=effective_as_of,
                attempts=attempts,
                observed_at=now,
            )

        # Sanitized daily observations are the bounded staging input for the
        # short raw upsert. A full action audit is fetched only if a receipt
        # cannot be reused after the just-observed raw evidence is persisted.
        self.market_data.apply_validated_batch(
            self.manifest,
            batch,
            ingestion_id=f"{request.request_id}:refresh:{attempts}",
            observed_at=now,
        )
        try:
            for listing in resolved_listings:
                bars = self.market_data.raw_bars(listing.listing_id, through=effective_as_of)
                if not bars:
                    raise RuntimeError("validated listing did not persist any daily bars")
                reusable = self.market_data.reusable_action_audit_receipt(
                    self.manifest,
                    listing_id=listing.listing_id,
                    provider=self.provider.name,
                    requested_as_of=effective_as_of,
                    now=now,
                )
                if reusable is not None:
                    continue
                actions = self.provider.fetch_action_history(
                    listing_id=listing.listing_id,
                    provider_symbol=listing.provider_symbol,
                    start=bars[0].session_date,
                    end=effective_as_of,
                )
                adjusted_closes = self.provider.fetch_adjusted_close_history(
                    listing_id=listing.listing_id,
                    provider_symbol=listing.provider_symbol,
                    start=bars[0].session_date,
                    end=effective_as_of,
                )
                self.market_data.complete_action_audit(
                    self.manifest,
                    listing_id=listing.listing_id,
                    provider=self.provider.name,
                    observed_actions=actions,
                    observed_adjusted_closes=adjusted_closes,
                    history_start=bars[0].session_date,
                    history_end=effective_as_of,
                    requested_as_of=effective_as_of,
                    observed_at=now,
                )
            resolver = self.artifact_resolver or ArtifactResolver(self.snapshot_root)
            staged_snapshot_path = (
                self.snapshot_root / ".staging" / f"{request.request_id}-{uuid4().hex}.parquet"
            )
            snapshot_hash = self.panel_state.freeze_snapshot(
                self.manifest,
                staged_snapshot_path,
                as_of_session=effective_as_of,
                observed_at=now,
                requested_listing_ids=request.listing_ids,
            )
            snapshot_artifact = resolver.publish_feature_input(staged_snapshot_path, snapshot_hash)
            snapshot_path = resolver.resolve_feature_input(snapshot_artifact)
        except ProviderFetchError as exc:
            return self._provider_failure(request, exc, attempts=attempts, observed_at=now)
        except FeatureAdmissionBlocked as exc:
            return self._blocked(
                request,
                f"data.feature_admission.{exc.code.casefold()}",
                effective_as_of,
                attempts=attempts,
            )
        except Exception:
            return self._record_and_block(
                request,
                code="data.postflight_failed",
                effective_as_of=effective_as_of,
                attempts=attempts,
                observed_at=now,
            )
        return PreFactorDataOutcome(
            request_id=str(request.request_id),
            status=PreFactorStatus.DATA_READY,
            summary=(
                "Validated raw/action data was refreshed and frozen as a FeatureInputSnapshot; "
                "Factor Research has not started."
            ),
            effective_as_of=effective_as_of,
            snapshot_path=snapshot_path,
            snapshot_hash=snapshot_hash,
            snapshot_artifact=snapshot_artifact,
            attempts=attempts,
        )

    def _use_frozen_panel(self, request: DataOperationsRequest) -> PreFactorDataOutcome:
        """Validate and project an already published full-universe panel.

        A research handoff is a consumer of the workspace data product.  It
        must never turn back into a ticker-scoped Yahoo refresh once an
        immutable panel has been admitted by the host Gateway.
        """

        panel_input = request.panel_input
        if panel_input is None:  # pragma: no cover - guarded by the caller
            raise RuntimeError("frozen panel input is required")
        panel = self.panel_state.active_feature_panel(self.manifest.profile.market_profile_id)
        snapshot = self.panel_state.feature_panel_snapshot_for_active(
            self.manifest.profile.market_profile_id
        )
        expected_panel = {
            "manifest_revision": request.manifest_revision,
            "sector_revision": panel_input.sector_revision,
            "catalog_hash": panel_input.catalog_hash,
            "spy_revision": panel_input.spy_revision,
            "policy_hash": panel_input.policy_hash,
            "panel_binding_hash": panel_input.panel_binding_hash,
            "panel_content_hash": panel_input.panel_content_hash,
            "temporal_identity_hash": panel_input.temporal_identity_hash,
            "temporal_risk_hash": panel_input.temporal_risk_hash,
        }
        if panel is None or any(panel.get(key) != value for key, value in expected_panel.items()):
            return self._blocked(request, "data.panel_binding_mismatch", None, attempts=0)
        expected_snapshot = {
            "snapshot_hash": panel_input.snapshot_hash,
            "manifest_uri": panel_input.snapshot_ref,
            "metadata_hash": panel_input.metadata_hash,
            "panel_content_hash": panel_input.panel_content_hash,
            "manifest_revision": request.manifest_revision,
            "sector_revision": panel_input.sector_revision,
            "catalog_hash": panel_input.catalog_hash,
            "spy_revision": panel_input.spy_revision,
            "policy_hash": panel_input.policy_hash,
            "panel_binding_hash": panel_input.panel_binding_hash,
            "temporal_identity_hash": panel_input.temporal_identity_hash,
            "temporal_risk_hash": panel_input.temporal_risk_hash,
        }
        if snapshot is None or any(
            snapshot.get(key) != value for key, value in expected_snapshot.items()
        ):
            return self._blocked(request, "data.panel_snapshot_mismatch", None, attempts=0)
        if panel_input.as_of_session < request.as_of_session:
            return self._blocked(request, "data.panel_snapshot_stale", None, attempts=0)
        descriptor = ArtifactDescriptor(
            kind="feature-panel-manifest",
            content_hash=panel_input.snapshot_hash,
            metadata_hash=panel_input.metadata_hash,
            uri=panel_input.snapshot_ref,
        )
        return PreFactorDataOutcome(
            request_id=str(request.request_id),
            status=PreFactorStatus.DATA_READY,
            summary=(
                "The handoff is bound to the workspace's immutable full-universe "
                "FeaturePanelSnapshot; no provider refresh was performed."
            ),
            effective_as_of=panel_input.as_of_session,
            snapshot_hash=panel_input.snapshot_hash,
            snapshot_artifact=descriptor,
            attempts=0,
        )

    def _refresh_start(self, request: DataOperationsRequest) -> date:
        starts: list[date] = []
        for listing_id in request.listing_ids:
            bars = self.market_data.raw_bars(listing_id, through=request.as_of_session)
            latest = bars[-1].session_date if bars else None
            starts.append(
                normal_refresh_start(latest, request.as_of_session) or request.requested_range_start
            )
        return min(starts)

    def _fetch_with_bounded_retry(
        self,
        request: DataOperationsRequest,
        fetch_start: date,
        *,
        observed_at: datetime,
    ) -> tuple[object, int] | PreFactorDataOutcome:
        for attempt in range(1, self.retry_budget + 1):
            try:
                return self.provider.fetch_daily(
                    request.symbols,
                    start=fetch_start,
                    end=request.as_of_session,
                ), attempt
            except ProviderFetchError as exc:
                if exc.retryable and attempt < self.retry_budget:
                    continue
                return self._provider_failure(
                    request, exc, attempts=attempt, observed_at=observed_at
                )
        raise AssertionError("bounded retry loop should return or raise")

    def _provider_failure(
        self,
        request: DataOperationsRequest,
        error: ProviderFetchError,
        *,
        attempts: int,
        observed_at: datetime,
    ) -> PreFactorDataOutcome:
        if error.code == "data.rate_limited":
            return PreFactorDataOutcome(
                request_id=str(request.request_id),
                status=PreFactorStatus.DEFERRED,
                summary=(
                    "The provider rate-limited a bounded refresh; "
                    "Data Operations will not guess or loop."
                ),
                effective_as_of=None,
                failure_code=error.code,
                attempts=attempts,
                retry_after_at=observed_at + timedelta(minutes=5),
            )
        return self._record_and_block(
            request,
            code=error.code,
            effective_as_of=None,
            attempts=attempts,
            observed_at=observed_at,
        )

    def _record_and_block(
        self,
        request: DataOperationsRequest,
        *,
        code: str,
        effective_as_of: date | None,
        attempts: int,
        observed_at: datetime,
    ) -> PreFactorDataOutcome:
        range_end = effective_as_of or request.as_of_session
        self.market_data.record_failures(
            FailureEvidence(
                listing_id,
                self.manifest.profile.market_profile_id,
                code,
                request.requested_range_start,
                range_end,
                observed_at,
            )
            for listing_id in request.listing_ids
        )
        return self._blocked(request, code, effective_as_of, attempts=attempts)

    @staticmethod
    def _blocked(
        request: DataOperationsRequest,
        code: str,
        effective_as_of: date | None,
        *,
        attempts: int,
    ) -> PreFactorDataOutcome:
        return PreFactorDataOutcome(
            request_id=str(request.request_id),
            status=PreFactorStatus.BLOCKED,
            summary="Data Operations did not admit this handoff to a feature snapshot.",
            effective_as_of=effective_as_of,
            failure_code=code,
            attempts=attempts,
        )
