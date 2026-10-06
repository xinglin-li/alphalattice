"""Local startup adapter for the deterministic Pre-Research Delta Engine."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq

from alphalattice.control.observation_runtime.telemetry.progress import (
    WorkProgressUpdate,
    WorkspaceProgressPublisher,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease
from alphalattice.foundation.factor_research.publication.artifacts import (
    FactorResearchArtifactStore,
)
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.contracts import panel_as_of_listing_identity
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.returns.semantic_revisions import (
    AdjustedReturnSemanticRevision,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    build_current_index_acquisition_manifest,
)
from alphalattice.foundation.market_data_ops.sources.universe import (
    bootstrap_from_candidate_manifest_document,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.foundation.research_foundation.contracts import (
    PreResearchDeskSafeProjection,
    ResearchFoundationBinding,
    ResearchFoundationMarker,
)
from alphalattice.foundation.research_foundation.runtime.delta import (
    ConfirmedPreResearchUpdateMandate,
    ListingDataDeltaKind,
    ObservedPreResearchCandidateSnapshot,
    PreResearchHeadStore,
    PreResearchListingObservation,
    PreResearchObservedDelta,
    PreResearchQuarantine,
    PreResearchRevisionMarker,
    PreResearchStagedDelta,
    PreResearchStorageEvidence,
    PreResearchUpdatePlan,
    QuarantineReason,
    VerifiedPreResearchHead,
    build_candidate_snapshot,
    build_verified_head,
    compile_update_plan,
    compute_observed_delta,
    confirm_update_plan,
    stage_qualified_delta,
)
from alphalattice.foundation.research_foundation.storage.repository import (
    ResearchFoundationStateRepository,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

PRE_RESEARCH_DELTA_POLICY_HASHES = (
    canonical_hash(
        {
            "engine": "git-like-pre-research-delta",
            "history_window": "inherit-verified-head",
            "retained_full_history_provider_calls": 0,
            "adjusted_return_epsilon": 1e-12,
            "membership_panel_policy": "full-history-cross-section",
            "activation": "marker-last-authoritative-readback-cas",
        }
    ),
)


class PreResearchDeltaService:
    """Resolve local HEAD/candidate state without Provider or numerical reads."""

    def __init__(
        self,
        *,
        market_data: MarketDataRepository,
        feature_state: FeatureStateRepository,
        panel_state: PanelStateRepository,
        profile_path: Path,
        artifact_root: Path,
        mutation_gate: WorkspaceMutationGate,
        writer_lease: WorkspaceWriterLease,
        progress_publisher: WorkspaceProgressPublisher | None = None,
    ) -> None:
        """Compose local transition evidence readers and the gated revision store.

        Args:
            market_data: Workspace source-membership and data-state repository.
            feature_state: Feature state used to assess numerical maintenance.
            panel_state: Panel state used to verify research-prefix publication.
            profile_path: Local market-profile configuration.
            artifact_root: Root of the admitted workspace artifacts.
            mutation_gate: Gate controlling final HEAD publication.
            writer_lease: Workspace writer ownership required for transition mutations.
            progress_publisher: Optional publisher of bounded transition progress.

        Raises:
            ValueError: The installed Feature catalog cannot be qualified.
        """
        self.market_data = market_data
        self.feature_state = feature_state
        self.panel_state = panel_state
        self.foundation_state = ResearchFoundationStateRepository(market_data.database)
        self.profile_path = Path(profile_path)
        self.artifact_root = Path(artifact_root)
        self.resolver = ArtifactResolver(self.artifact_root)
        self.factor_artifacts = FactorResearchArtifactStore(self.artifact_root)
        self.writer_lease = writer_lease
        self.progress_publisher = progress_publisher
        self.catalog = FeatureCatalog.load()
        self._adjusted_factor_ids = tuple(
            sorted(
                contract.factor_id
                for contract in self.catalog.maintenance_contracts
                if "provider_adjusted_close" in contract.required_fields
            )
        )
        raw_fields = {
            "open_raw",
            "high_raw",
            "low_raw",
            "close_raw",
            "volume_raw",
            "open_split_adjusted",
            "high_split_adjusted",
            "low_split_adjusted",
            "close_split_adjusted",
        }
        self._raw_factor_ids = tuple(
            sorted(
                contract.factor_id
                for contract in self.catalog.maintenance_contracts
                if raw_fields.intersection(contract.required_fields)
            )
        )
        self.heads = PreResearchHeadStore(
            artifact_root=self.artifact_root, mutation_gate=mutation_gate
        )

    def resolve_verified_head(self) -> VerifiedPreResearchHead | None:
        """Read the active research HEAD through validated marker and transition lineage.

        Returns:
            Verified HEAD, or None before an active pointer is installed.

        Raises:
            ValueError: Durable pointer, marker or child lineage is inconsistent.
            FileNotFoundError: An active pointer names an absent child.
        """
        return self.heads.current_head()

    def bootstrap_verified_head(
        self, *, market_profile_id: str, verified_at: datetime
    ) -> VerifiedPreResearchHead:
        """Bind an existing verified Foundation once, without rotating it."""
        if not self.writer_lease.held:
            raise RuntimeError("PRE_RESEARCH_WRITER_LEASE_NOT_HELD")
        current = self.resolve_verified_head()
        if current is not None and current.adjusted_return_revision_cursor is not None:
            pending = self.heads.pending_transition()
            if pending is not None and pending.base_head_hash != current.head_hash:
                self.finalize_pending_transition(published_at=verified_at)
                current = self.resolve_verified_head()
                if current is None:
                    raise ValueError("PRE_RESEARCH_HEAD_RECOVERY_LOST_CURRENT")
            return current
        readiness = self.market_data.readiness.load(market_profile_id)
        if (
            readiness is None
            or readiness.active_manifest_id is None
            or readiness.active_candidate_manifest_document is None
        ):
            raise ValueError("PRE_RESEARCH_HEAD_FOUNDATION_NOT_READY")
        manifest = self.market_data.load_universe_manifest(readiness.active_manifest_id)
        snapshot = self.panel_state.feature_panel_snapshot_for_active(market_profile_id)
        if snapshot is None:
            raise ValueError("PRE_RESEARCH_HEAD_PANEL_NOT_READY")
        panel = self.resolver.load_feature_panel_manifest(str(snapshot["manifest_uri"]))
        if str(panel["snapshot_hash"]) != str(snapshot["snapshot_hash"]):
            raise ValueError("PRE_RESEARCH_HEAD_PANEL_READBACK_MISMATCH")

        projection = PreResearchDeskSafeProjection.model_validate(
            self.factor_artifacts.pre_research_desk_projection()
        )
        marker_ref = FactorResearchArtifactStore.uri(
            "research-desk/foundation-markers", projection.foundation_marker_hash
        )
        marker = ResearchFoundationMarker.model_validate(
            self.factor_artifacts.load_research_foundation_marker(marker_ref)
        )
        foundation = ResearchFoundationBinding.model_validate(
            self.factor_artifacts.load_research_foundation(marker.foundation_ref)
        )
        if (
            marker.foundation_hash != projection.foundation_hash
            or foundation.foundation_hash != projection.foundation_hash
            or foundation.feature_panel_snapshot_hash != panel["snapshot_hash"]
            or foundation.execution_outcome.snapshot_hash
            != projection.execution_outcome_snapshot_hash
        ):
            raise ValueError("PRE_RESEARCH_HEAD_FOUNDATION_READBACK_MISMATCH")
        adjusted_revision = self._load_adjusted_return_revision()
        baseline_cursor, baseline_chain_hash = self._foundation_revision_binding(
            panel_snapshot_hash=str(panel["snapshot_hash"]),
            current=adjusted_revision,
        )

        bootstrap = bootstrap_from_candidate_manifest_document(
            readiness.active_candidate_manifest_document
        )
        acquisition = build_current_index_acquisition_manifest(self.profile_path, bootstrap)
        source_ids = tuple(sorted(value.listing_id for value in acquisition.listings))
        listing_ids = tuple(sorted(value.listing_id for value in manifest.listings))
        admitted_hash, admitted_count = panel_as_of_listing_identity(panel)
        if admitted_hash != canonical_hash(listing_ids) or admitted_count != len(listing_ids):
            raise ValueError("PRE_RESEARCH_HEAD_LISTING_AUTHORITY_MISMATCH")
        lineage = panel.get("safe_summary", {}).get("lineage", {})
        if not isinstance(lineage, dict):
            raise ValueError("PRE_RESEARCH_HEAD_PANEL_LINEAGE_MISSING")
        source_revision = str(
            readiness.active_candidate_manifest_document.get("content_hash")
            or canonical_hash(readiness.active_candidate_manifest_document)
        )
        watermark = self.market_data.execution_source_watermark(
            manifest, through=date.fromisoformat(str(panel["as_of_session"]))
        )
        head = build_verified_head(
            kind="VerifiedPreResearchHead",
            market_profile_id=market_profile_id,
            manifest_revision=manifest.revision_sha256,
            source_revision=source_revision,
            membership_fingerprint=canonical_hash(source_ids),
            ordered_source_candidate_listing_ids=source_ids,
            ordered_listing_ids=listing_ids,
            listing_set_hash=admitted_hash,
            research_history_start=date.fromisoformat(str(panel["history_start"])),
            target_market_session=date.fromisoformat(str(panel["as_of_session"])),
            sector_revision=str(lineage["sector_revision"]),
            catalog_hash=str(lineage["catalog_hash"]),
            panel_snapshot_hash=str(panel["snapshot_hash"]),
            factor_screening_result_hash=projection.factor_screening_result_hash,
            factor_candidate_slate_hash=projection.factor_candidate_slate_hash,
            research_desk_factor_input_hash=projection.research_desk_factor_input_hash,
            causal_execution_outcome_snapshot_hash=(projection.execution_outcome_snapshot_hash),
            research_foundation_hash=foundation.foundation_hash,
            research_foundation_marker_hash=marker.marker_hash,
            ordered_factor_ids=foundation.ordered_factor_ids,
            data_state_hash=canonical_hash(
                {
                    "manifest_revision": manifest.revision_sha256,
                    "panel_content_hash": panel["panel_content_hash"],
                    "source_watermark_hash": watermark["watermark_hash"],
                    "adjusted_return_revision_cursor": baseline_cursor,
                    "adjusted_return_revision_chain_hash": baseline_chain_hash,
                }
            ),
            adjusted_return_revision_cursor=baseline_cursor,
            adjusted_return_revision_chain_hash=baseline_chain_hash,
            validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
            is_point_in_time_historical=False,
            verified_at=verified_at,
        )
        transition_base = current or head
        candidate = self.observe_candidate(
            head=transition_base,
            candidate_document=readiness.active_candidate_manifest_document,
            target_market_session=head.target_market_session,
            observed_at=verified_at,
            adjusted_revision_cursor=baseline_cursor,
            adjusted_revision_chain_hash=baseline_chain_hash,
            suppress_adjusted_revision_delta=True,
        )
        delta = compute_observed_delta(head=transition_base, candidate=candidate)
        if delta.revision_disposition.value != "NOOP":
            raise ValueError(f"PRE_RESEARCH_HEAD_BOOTSTRAP_NOT_NOOP:{delta.revision_disposition}")
        plan = compile_update_plan(delta=delta, policy_hashes=PRE_RESEARCH_DELTA_POLICY_HASHES)
        staged = stage_qualified_delta(
            delta=delta,
            ordered_qualified_listing_ids=head.ordered_listing_ids,
            quarantines=(),
        )
        self.heads.publish_candidate(candidate)
        self.heads.publish_delta(delta)
        self.heads.publish_plan(plan)
        self.heads.publish_staged_delta(staged)
        self.heads.advance_head(
            expected_base_head_hash=current.head_hash if current is not None else None,
            next_head=head,
            candidate_snapshot_hash=candidate.candidate_snapshot_hash,
            delta_hash=delta.delta_hash,
            staged_delta_hash=staged.staged_delta_hash,
            update_plan_hash=plan.update_plan_hash,
            mandate_hash=None,
            child_hashes=(
                head.panel_snapshot_hash,
                head.factor_screening_result_hash,
                head.factor_candidate_slate_hash,
                head.research_desk_factor_input_hash,
                head.causal_execution_outcome_snapshot_hash,
                head.research_foundation_hash,
                head.research_foundation_marker_hash,
            ),
            storage_evidence=self._panel_storage_evidence(panel, prior_head=current),
            published_at=verified_at,
        )
        return self.heads.current_head() or head

    def observe_candidate(
        self,
        *,
        head: VerifiedPreResearchHead,
        candidate_document: Mapping[str, object],
        target_market_session: date,
        observed_at: datetime,
        semantic_delta_kinds: Mapping[str, ListingDataDeltaKind] | None = None,
        adjusted_revision_cursor: int | None = None,
        adjusted_revision_chain_hash: str | None = None,
        suppress_adjusted_revision_delta: bool = False,
    ) -> ObservedPreResearchCandidateSnapshot:
        """Freeze one local, payload-free working tree for deterministic diff."""
        bootstrap = bootstrap_from_candidate_manifest_document(dict(candidate_document))
        acquisition = build_current_index_acquisition_manifest(self.profile_path, bootstrap)
        listing_ids = tuple(sorted(value.listing_id for value in acquisition.listings))
        ranges = self.market_data.listing_raw_ranges(listing_ids, through=target_market_session)
        revision = self._load_adjusted_return_revision()
        frozen_cursor = (
            revision.cursor if adjusted_revision_cursor is None else adjusted_revision_cursor
        )
        frozen_chain_hash = (
            revision.chain_hash
            if adjusted_revision_chain_hash is None
            else adjusted_revision_chain_hash
        )
        if (adjusted_revision_cursor is None) != (adjusted_revision_chain_hash is None):
            raise ValueError("PRE_RESEARCH_ADJUSTED_REVISION_OVERRIDE_INCOMPLETE")
        semantic_impacts = (
            {}
            if suppress_adjusted_revision_delta
            else self._adjusted_revision_impacts(head=head, current=revision)
        )
        for listing_id, kind in (semantic_delta_kinds or {}).items():
            semantic_impacts[listing_id] = (
                kind,
                semantic_impacts.get(listing_id, (kind, ()))[1],
            )
        observations = []
        for listing_id in listing_ids:
            bounds = ranges.get(listing_id)
            kind, affected_sessions = semantic_impacts.get(
                listing_id, (ListingDataDeltaKind.UNCHANGED, ())
            )
            affected_factor_ids = (
                self.catalog.factor_ids
                if kind is ListingDataDeltaKind.TAIL_APPEND_REQUIRED
                else self._adjusted_factor_ids
                if kind is ListingDataDeltaKind.ADJUSTED_RETURN_CORRECTION
                else self._raw_factor_ids
                if kind is ListingDataDeltaKind.RAW_SEMANTIC_CORRECTION
                else ()
            )
            observations.append(
                PreResearchListingObservation(
                    listing_id=listing_id,
                    source_member=True,
                    local_history_start=bounds[0] if bounds else None,
                    local_history_end=bounds[1] if bounds else None,
                    data_state_hash=canonical_hash((listing_id, bounds)) if bounds else None,
                    raw_semantic_delta=kind is ListingDataDeltaKind.RAW_SEMANTIC_CORRECTION,
                    adjusted_return_semantic_delta=(
                        kind is ListingDataDeltaKind.ADJUSTED_RETURN_CORRECTION
                    ),
                    evidence_only_delta=kind is ListingDataDeltaKind.EVIDENCE_ONLY,
                    semantic_delta_kind=(
                        kind if kind is not ListingDataDeltaKind.UNCHANGED else None
                    ),
                    affected_factor_ids=affected_factor_ids,
                    affected_sessions=affected_sessions,
                    session_set_known=kind is not ListingDataDeltaKind.BLOCKED_SOURCE_AMBIGUITY,
                    source_ambiguous=kind is ListingDataDeltaKind.BLOCKED_SOURCE_AMBIGUITY,
                )
            )
        source_revision = str(
            candidate_document.get("content_hash") or canonical_hash(candidate_document)
        )
        readiness = self.market_data.readiness.load(head.market_profile_id)
        if readiness is None or readiness.active_manifest_id is None:
            raise ValueError("PRE_RESEARCH_ACTIVE_MANIFEST_NOT_READY")
        sector = self.feature_state.current_sector_state(
            self.market_data.load_universe_manifest(readiness.active_manifest_id)
        )
        return build_candidate_snapshot(
            kind="ObservedPreResearchCandidateSnapshot",
            market_profile_id=head.market_profile_id,
            base_head_hash=head.head_hash,
            source_revision=source_revision,
            source_membership_fingerprint=canonical_hash(listing_ids),
            target_market_session=target_market_session,
            research_history_start=head.research_history_start,
            ordered_candidate_listing_ids=listing_ids,
            listing_observations=tuple(observations),
            sector_revision=sector.sector_revision if sector else None,
            adjusted_return_revision_cursor=frozen_cursor,
            adjusted_return_revision_chain_hash=frozen_chain_hash,
            policy_hashes=PRE_RESEARCH_DELTA_POLICY_HASHES,
            observed_at=observed_at,
        )

    def _load_adjusted_return_revision(self) -> AdjustedReturnSemanticRevision:
        try:
            payload = self.resolver.load_current_adjusted_return_semantic_revision()
        except FileNotFoundError:
            return AdjustedReturnSemanticRevision(
                cursor=0,
                chain_hash=canonical_hash([]),
                deltas=(),
            )
        return AdjustedReturnSemanticRevision.model_validate(payload)

    def _foundation_revision_binding(
        self,
        *,
        panel_snapshot_hash: str,
        current: AdjustedReturnSemanticRevision,
    ) -> tuple[int, str]:
        outcome_hashes = {
            str(manifest["snapshot_hash"])
            for manifest, _descriptor in self.factor_artifacts.outcome_manifests()
            if manifest.get("parent_panel_snapshot_hash") == panel_snapshot_hash
        }
        receipts = tuple(
            value
            for value in self.factor_artifacts.outcome_reuse_receipts()
            if value.get("outcome_snapshot_hash") in outcome_hashes
        )
        if not receipts:
            if current.cursor == 0:
                return 0, current.chain_hash
            raise ValueError("PRE_RESEARCH_FOUNDATION_ADJUSTED_REVISION_BINDING_MISSING")
        receipt = max(
            receipts,
            key=lambda value: (int(value.get("revision_cursor", -1)), str(value["observed_at"])),
        )
        cursor = int(receipt.get("revision_cursor", -1))
        if cursor < 0 or cursor > current.cursor:
            raise ValueError("PRE_RESEARCH_FOUNDATION_ADJUSTED_REVISION_CURSOR_GAP")
        expected = canonical_hash([value.semantic_hash for value in current.deltas[:cursor]])
        if receipt.get("revision_chain_hash") != expected:
            raise ValueError("PRE_RESEARCH_FOUNDATION_ADJUSTED_REVISION_CHAIN_DIVERGED")
        return cursor, expected

    @staticmethod
    def _adjusted_revision_delta_kinds(
        *,
        head: VerifiedPreResearchHead,
        current: AdjustedReturnSemanticRevision,
    ) -> dict[str, ListingDataDeltaKind]:
        return {
            listing_id: impact[0]
            for listing_id, impact in PreResearchDeltaService._adjusted_revision_impacts(
                head=head, current=current
            ).items()
        }

    @staticmethod
    def _adjusted_revision_impacts(
        *,
        head: VerifiedPreResearchHead,
        current: AdjustedReturnSemanticRevision,
    ) -> dict[str, tuple[ListingDataDeltaKind, tuple[date, ...]]]:
        cursor = head.adjusted_return_revision_cursor
        chain_hash = head.adjusted_return_revision_chain_hash
        if cursor is None or chain_hash is None:
            raise ValueError("PRE_RESEARCH_HEAD_ADJUSTED_REVISION_BINDING_MISSING")
        if cursor > current.cursor:
            raise ValueError("PRE_RESEARCH_ADJUSTED_REVISION_CURSOR_GAP")
        expected = canonical_hash([value.semantic_hash for value in current.deltas[:cursor]])
        if chain_hash != expected:
            raise ValueError("PRE_RESEARCH_ADJUSTED_REVISION_CHAIN_DIVERGED")
        by_listing: dict[str, tuple[ListingDataDeltaKind, set[date]]] = {}
        for delta in current.deltas[cursor:]:
            prior_kind, prior_sessions = by_listing.get(
                delta.listing_id, (ListingDataDeltaKind.UNCHANGED, set())
            )
            sessions = prior_sessions | set(delta.changed_return_sessions)
            if delta.session_set_changed:
                kind = (
                    ListingDataDeltaKind.TAIL_APPEND_REQUIRED
                    if delta.changed_return_sessions
                    and all(
                        session > head.target_market_session
                        for session in delta.changed_return_sessions
                    )
                    else ListingDataDeltaKind.BLOCKED_SOURCE_AMBIGUITY
                )
            elif delta.uniform_rescale:
                if delta.changed_return_sessions:
                    raise ValueError("PRE_RESEARCH_UNIFORM_RESCALE_CHANGED_RETURNS")
                kind = ListingDataDeltaKind.EVIDENCE_ONLY
            elif delta.changed_return_sessions:
                kind = ListingDataDeltaKind.ADJUSTED_RETURN_CORRECTION
            else:
                kind = ListingDataDeltaKind.EVIDENCE_ONLY
            precedence = {
                ListingDataDeltaKind.UNCHANGED: 0,
                ListingDataDeltaKind.EVIDENCE_ONLY: 1,
                ListingDataDeltaKind.TAIL_APPEND_REQUIRED: 2,
                ListingDataDeltaKind.ADJUSTED_RETURN_CORRECTION: 3,
                ListingDataDeltaKind.BLOCKED_SOURCE_AMBIGUITY: 4,
            }
            selected = kind if precedence[kind] >= precedence[prior_kind] else prior_kind
            by_listing[delta.listing_id] = (selected, sessions)
        return {
            listing_id: (kind, tuple(sorted(sessions)))
            for listing_id, (kind, sessions) in by_listing.items()
        }

    def assess(
        self,
        *,
        candidate_document: Mapping[str, object],
        target_market_session: date,
        observed_at: datetime,
        semantic_delta_kinds: Mapping[str, ListingDataDeltaKind] | None = None,
    ) -> tuple[
        ObservedPreResearchCandidateSnapshot,
        PreResearchObservedDelta,
        PreResearchUpdatePlan,
    ]:
        """Freeze local source evidence and compile exact transition scopes under writer ownership.

        Args:
            candidate_document: Declared source-candidate document to observe locally.
            target_market_session: Requested end of the admitted maintenance interval.
            observed_at: Timezone-aware clock for frozen source evidence.
            semantic_delta_kinds: Optional exact per-listing semantic classifications.

        Returns:
            Candidate, observed delta and compiled plan. A confirmed pending tree on
            the same HEAD is reopened unchanged; new actionable work becomes pending.
            No-op or blocked work clears only an unconfirmed proposal on that base.

        Raises:
            RuntimeError: The workspace writer lease is not held.
            ValueError: HEAD is absent/stale or source, policy, or transition evidence is invalid.
        """
        if not self.writer_lease.held:
            raise RuntimeError("PRE_RESEARCH_WRITER_LEASE_NOT_HELD")
        head = self.resolve_verified_head()
        if head is None:
            raise ValueError("PRE_RESEARCH_HEAD_MISSING")
        pending = self.heads.pending_transition()
        if pending is not None and pending.mandate_hash is not None:
            if pending.base_head_hash != head.head_hash:
                raise ValueError("PRE_RESEARCH_CONFIRMED_PENDING_BASE_HEAD_STALE")
            return (
                self.heads.load_candidate(pending.candidate_snapshot_hash),
                self.heads.load_delta(pending.delta_hash),
                self.heads.load_plan(pending.update_plan_hash),
            )
        operation_id = canonical_hash(
            (
                "pre-research-local-delta",
                head.head_hash,
                target_market_session.isoformat(),
                candidate_document.get("content_hash"),
            )
        )
        self._publish_progress(operation_id, "RUNNING", 0, current_item="resolve_verified_head")
        candidate = self.observe_candidate(
            head=head,
            candidate_document=candidate_document,
            target_market_session=target_market_session,
            observed_at=observed_at,
            semantic_delta_kinds=semantic_delta_kinds,
        )
        self._publish_progress(operation_id, "RUNNING", 1, current_item="freeze_candidate")
        delta = compute_observed_delta(head=head, candidate=candidate)
        self._publish_progress(operation_id, "RUNNING", 2, current_item="compute_delta")
        plan = compile_update_plan(delta=delta, policy_hashes=PRE_RESEARCH_DELTA_POLICY_HASHES)
        self.heads.publish_candidate(candidate)
        self.heads.publish_delta(delta)
        self.heads.publish_plan(plan)
        if delta.revision_disposition.value not in {"NOOP", "BLOCKED"}:
            self.heads.set_pending_transition(
                candidate=candidate,
                delta=delta,
                plan=plan,
                frozen_at=observed_at,
            )
        else:
            self.heads.clear_unconfirmed_pending_transition(
                base_head_hash=head.head_hash,
            )
        self._publish_progress(
            operation_id,
            "REUSED_EXACT" if delta.revision_disposition.value == "NOOP" else "SUCCEEDED",
            3,
            current_item="compile_update_plan",
            counters={
                "full_history_additions": len(plan.full_history_addition_listing_ids),
                "bounded_retained": len(plan.bounded_retained_listing_ids),
                "removals": len(plan.removed_listing_ids),
            },
        )
        return candidate, delta, plan

    def confirm_plan(
        self,
        *,
        plan: PreResearchUpdatePlan,
        confirmed_at: datetime,
        confirmation_token: str,
    ) -> ConfirmedPreResearchUpdateMandate:
        """Bind user confirmation to the current frozen plan and retain its pending tree.

        Args:
            plan: Sealed plan requiring confirmation under the installed delta policies.
            confirmed_at: Timezone-aware confirmation clock.
            confirmation_token: User confirmation bound to this exact tree and plan.

        Returns:
            Durable mandate retained with the frozen candidate/delta/plan pointer.

        Raises:
            RuntimeError: The workspace writer lease is not held.
            ValueError: Base HEAD is stale, policies changed, confirmation is unnecessary,
                or the mandate/transition lineage cannot be validated.
        """
        if not self.writer_lease.held:
            raise RuntimeError("PRE_RESEARCH_WRITER_LEASE_NOT_HELD")
        current = self.resolve_verified_head()
        if (current.head_hash if current is not None else None) != plan.base_head_hash:
            raise ValueError("PRE_RESEARCH_CONFIRMATION_BASE_HEAD_STALE")
        if plan.policy_hashes != PRE_RESEARCH_DELTA_POLICY_HASHES:
            raise ValueError("PRE_RESEARCH_CONFIRMATION_POLICY_CHANGED")
        if not plan.user_confirmation_required:
            raise ValueError("PRE_RESEARCH_UPDATE_CONFIRMATION_NOT_REQUIRED")
        mandate = confirm_update_plan(
            plan=plan,
            confirmed_at=confirmed_at,
            confirmation_token=confirmation_token,
        )
        self.heads.publish_mandate(mandate)
        self.heads.set_pending_transition(
            candidate=self.heads.load_candidate(plan.candidate_snapshot_hash),
            delta=self.heads.load_delta(plan.delta_hash),
            plan=plan,
            frozen_at=confirmed_at,
            mandate=mandate,
        )
        return mandate

    def pending_qualified_active_set_changed(self) -> bool | None:
        """Resolve the post-hydration active-set fact without reading feature values."""
        pending = self.heads.pending_transition()
        if pending is None:
            return None
        head = self.resolve_verified_head()
        if head is None or head.head_hash != pending.base_head_hash:
            raise ValueError("PRE_RESEARCH_PENDING_BASE_HEAD_STALE")
        candidate = self.heads.load_candidate(pending.candidate_snapshot_hash)
        readiness = self.market_data.readiness.load(candidate.market_profile_id)
        if readiness is None or readiness.active_manifest_id is None:
            raise ValueError("PRE_RESEARCH_QUALIFIED_ACTIVE_SET_NOT_READY")
        manifest = self.market_data.load_universe_manifest(readiness.active_manifest_id)
        qualified_ids = tuple(sorted(value.listing_id for value in manifest.listings))
        return qualified_ids != head.ordered_listing_ids

    def activate_staged_revision(
        self,
        *,
        candidate: ObservedPreResearchCandidateSnapshot,
        delta: PreResearchObservedDelta,
        plan: PreResearchUpdatePlan,
        staged: PreResearchStagedDelta,
        next_head: VerifiedPreResearchHead,
        storage_evidence: PreResearchStorageEvidence,
        published_at: datetime,
        mandate: ConfirmedPreResearchUpdateMandate | None = None,
        transition_id: str | None = None,
    ) -> PreResearchRevisionMarker:
        """Validate one exact staged tree and CAS-advance the verified HEAD."""
        if not self.writer_lease.held:
            raise RuntimeError("PRE_RESEARCH_WRITER_LEASE_NOT_HELD")
        current = self.resolve_verified_head()
        actual_base = current.head_hash if current is not None else None
        if (
            plan.policy_hashes != PRE_RESEARCH_DELTA_POLICY_HASHES
            or candidate.policy_hashes != PRE_RESEARCH_DELTA_POLICY_HASHES
        ):
            raise ValueError("PRE_RESEARCH_STAGED_POLICY_CHANGED")
        if actual_base != plan.base_head_hash or delta.base_head_hash != plan.base_head_hash:
            raise ValueError("PRE_RESEARCH_STAGED_BASE_HEAD_MISMATCH")
        if (
            candidate.candidate_snapshot_hash != plan.candidate_snapshot_hash
            or delta.candidate_snapshot_hash != candidate.candidate_snapshot_hash
            or plan.delta_hash != delta.delta_hash
            or staged.observed_delta_hash != delta.delta_hash
            or next_head.ordered_listing_ids != staged.ordered_qualified_listing_ids
            or next_head.listing_set_hash != staged.qualified_listing_set_hash
            or next_head.research_history_start != delta.research_history_start
            or next_head.target_market_session != delta.target_market_session
            or next_head.adjusted_return_revision_cursor
            != (
                staged.adjusted_return_revision_cursor
                if staged.adjusted_return_revision_cursor is not None
                else candidate.adjusted_return_revision_cursor
            )
            or next_head.adjusted_return_revision_chain_hash
            != (
                staged.adjusted_return_revision_chain_hash
                if staged.adjusted_return_revision_chain_hash is not None
                else candidate.adjusted_return_revision_chain_hash
            )
        ):
            raise ValueError("PRE_RESEARCH_STAGED_TREE_MISMATCH")
        if plan.user_confirmation_required:
            if mandate is None:
                raise ValueError("PRE_RESEARCH_UPDATE_CONFIRMATION_REQUIRED")
            if (
                mandate.base_head_hash != plan.base_head_hash
                or mandate.candidate_snapshot_hash != plan.candidate_snapshot_hash
                or mandate.delta_hash != plan.delta_hash
                or mandate.update_plan_hash != plan.update_plan_hash
                or mandate.policy_hashes != plan.policy_hashes
            ):
                raise ValueError("PRE_RESEARCH_UPDATE_MANDATE_MISMATCH")
        elif mandate is not None:
            raise ValueError("PRE_RESEARCH_UPDATE_MANDATE_NOT_REQUIRED")
        child_hashes = (
            next_head.panel_snapshot_hash,
            next_head.factor_screening_result_hash,
            next_head.factor_candidate_slate_hash,
            next_head.research_desk_factor_input_hash,
            next_head.causal_execution_outcome_snapshot_hash,
            next_head.research_foundation_hash,
            next_head.research_foundation_marker_hash,
        )
        if staged.staged_child_hashes and staged.staged_child_hashes != child_hashes:
            raise ValueError("PRE_RESEARCH_STAGED_CHILDREN_MISMATCH")
        if staged.active_set_changed:
            if transition_id is None:
                raise ValueError("PRE_RESEARCH_MEMBERSHIP_TRANSITION_ID_REQUIRED")
            self.foundation_state.start_feature_universe_rebuild(
                transition_id, observed_at=published_at
            )
        elif transition_id is not None:
            raise ValueError("PRE_RESEARCH_TRANSITION_ID_WITHOUT_ACTIVE_SET_CHANGE")
        self.heads.publish_candidate(candidate)
        self.heads.publish_delta(delta)
        self.heads.publish_plan(plan)
        self.heads.publish_staged_delta(staged)
        if mandate is not None:
            self.heads.publish_mandate(mandate)

        marker: PreResearchRevisionMarker | None = None
        try:
            marker = self.heads.advance_head(
                expected_base_head_hash=plan.base_head_hash,
                next_head=next_head,
                candidate_snapshot_hash=candidate.candidate_snapshot_hash,
                delta_hash=delta.delta_hash,
                staged_delta_hash=staged.staged_delta_hash,
                update_plan_hash=plan.update_plan_hash,
                mandate_hash=mandate.mandate_hash if mandate is not None else None,
                child_hashes=child_hashes,
                storage_evidence=storage_evidence,
                published_at=published_at,
            )
            if transition_id is not None:
                self._fulfill_feature_universe_rebuild(
                    transition_id=transition_id,
                    next_head=next_head,
                    marker=marker,
                    observed_at=published_at,
                )
            return marker
        except Exception:
            if transition_id is not None and marker is None:
                self.foundation_state.block_feature_universe_rebuild(
                    transition_id,
                    failure_code="PRE_RESEARCH_REVISION_LIFECYCLE_PUBLICATION_FAILED",
                    observed_at=published_at,
                )
            raise

    def _fulfill_feature_universe_rebuild(
        self,
        *,
        transition_id: str,
        next_head: VerifiedPreResearchHead,
        marker: PreResearchRevisionMarker,
        observed_at: datetime,
    ) -> None:
        requirement = self.foundation_state.feature_universe_rebuild_requirement(transition_id)
        if requirement.lifecycle != "FULFILLED":
            self.foundation_state.start_feature_universe_rebuild(
                transition_id, observed_at=observed_at
            )
            requirement = self.foundation_state.fulfill_feature_universe_rebuild(
                transition_id,
                panel_snapshot_hash=next_head.panel_snapshot_hash,
                factor_result_hash=next_head.factor_screening_result_hash,
                factor_slate_hash=next_head.factor_candidate_slate_hash,
                execution_outcome_hash=next_head.causal_execution_outcome_snapshot_hash,
                foundation_hash=next_head.research_foundation_hash,
                revision_marker_hash=marker.marker_hash,
                observed_at=observed_at,
            )
        expected = (
            next_head.panel_snapshot_hash,
            next_head.factor_screening_result_hash,
            next_head.factor_candidate_slate_hash,
            next_head.causal_execution_outcome_snapshot_hash,
            next_head.research_foundation_hash,
            marker.marker_hash,
        )
        actual = (
            requirement.panel_snapshot_hash,
            requirement.factor_result_hash,
            requirement.factor_slate_hash,
            requirement.execution_outcome_hash,
            requirement.foundation_hash,
            requirement.revision_marker_hash,
        )
        if requirement.lifecycle != "FULFILLED" or actual != expected:
            raise ValueError("PRE_RESEARCH_REVISION_LIFECYCLE_READBACK_MISMATCH")

    def finalize_pending_transition(
        self, *, published_at: datetime
    ) -> PreResearchRevisionMarker | None:
        """Bind newly verified deterministic children and CAS-advance the frozen tree."""
        if not self.writer_lease.held:
            raise RuntimeError("PRE_RESEARCH_WRITER_LEASE_NOT_HELD")
        pending = self.heads.pending_transition()
        if pending is None:
            return None
        current = self.resolve_verified_head()
        actual_base = current.head_hash if current is not None else None
        if actual_base != pending.base_head_hash:
            marker = self.heads.current_revision_marker()
            if (
                current is not None
                and marker is not None
                and marker.update_plan_hash == pending.update_plan_hash
                and marker.next_head_hash == current.head_hash
            ):
                staged = self.heads.load_staged_delta(marker.staged_delta_hash)
                if staged.active_set_changed:
                    requirement = self.foundation_state.feature_universe_rebuild_for_manifest(
                        current.manifest_revision
                    )
                    if requirement is None:
                        raise ValueError("PRE_RESEARCH_MEMBERSHIP_REBUILD_REQUIREMENT_MISSING")
                    self._fulfill_feature_universe_rebuild(
                        transition_id=requirement.transition_id,
                        next_head=current,
                        marker=marker,
                        observed_at=published_at,
                    )
                self.heads.clear_pending_transition(expected_pending_hash=pending.pending_hash)
                return marker
            raise ValueError("PRE_RESEARCH_PENDING_BASE_HEAD_STALE")
        candidate = self.heads.load_candidate(pending.candidate_snapshot_hash)
        delta = self.heads.load_delta(pending.delta_hash)
        plan = self.heads.load_plan(pending.update_plan_hash)
        mandate = self.heads.load_mandate(pending.mandate_hash) if pending.mandate_hash else None

        readiness = self.market_data.readiness.load(candidate.market_profile_id)
        if readiness is None or readiness.active_manifest_id is None:
            raise ValueError("PRE_RESEARCH_FINALIZATION_MANIFEST_NOT_READY")
        manifest = self.market_data.load_universe_manifest(readiness.active_manifest_id)
        snapshot = self.panel_state.feature_panel_snapshot_for_active(candidate.market_profile_id)
        if snapshot is None:
            raise ValueError("PRE_RESEARCH_FINALIZATION_PANEL_NOT_READY")
        panel = self.resolver.load_feature_panel_manifest(str(snapshot["manifest_uri"]))
        if str(panel["snapshot_hash"]) != str(snapshot["snapshot_hash"]):
            raise ValueError("PRE_RESEARCH_FINALIZATION_PANEL_READBACK_MISMATCH")
        listing_ids = tuple(sorted(value.listing_id for value in manifest.listings))
        if panel_as_of_listing_identity(panel)[0] != canonical_hash(listing_ids):
            raise ValueError("PRE_RESEARCH_FINALIZATION_LISTING_SET_MISMATCH")

        projection = PreResearchDeskSafeProjection.model_validate(
            self.factor_artifacts.pre_research_desk_projection()
        )
        foundation_marker = ResearchFoundationMarker.model_validate(
            self.factor_artifacts.load_research_foundation_marker(
                FactorResearchArtifactStore.uri(
                    "research-desk/foundation-markers",
                    projection.foundation_marker_hash,
                )
            )
        )
        foundation = ResearchFoundationBinding.model_validate(
            self.factor_artifacts.load_research_foundation(foundation_marker.foundation_ref)
        )
        if (
            foundation.foundation_hash != projection.foundation_hash
            or foundation.feature_panel_snapshot_hash != str(panel["snapshot_hash"])
            or foundation.factor_screening_result_hash != projection.factor_screening_result_hash
            or foundation.factor_candidate_slate_hash != projection.factor_candidate_slate_hash
        ):
            raise ValueError("PRE_RESEARCH_FINALIZATION_FOUNDATION_MISMATCH")

        lineage = panel.get("safe_summary", {}).get("lineage", {})
        if not isinstance(lineage, dict):
            raise ValueError("PRE_RESEARCH_FINALIZATION_PANEL_LINEAGE_MISSING")
        adjusted_revision = self._load_adjusted_return_revision()
        child_hashes = (
            str(panel["snapshot_hash"]),
            projection.factor_screening_result_hash,
            projection.factor_candidate_slate_hash,
            projection.research_desk_factor_input_hash,
            projection.execution_outcome_snapshot_hash,
            foundation.foundation_hash,
            foundation_marker.marker_hash,
        )
        active_set = set(listing_ids)
        delta_by_listing = {value.listing_id: value.kind for value in delta.listing_deltas}
        candidate_ranges = self.market_data.listing_raw_ranges(
            candidate.ordered_candidate_listing_ids,
            through=delta.target_market_session,
        )
        quarantines = tuple(
            PreResearchQuarantine(
                listing_id=listing_id,
                reason=(
                    QuarantineReason.INSUFFICIENT_RESEARCH_HISTORY
                    if (
                        delta_by_listing.get(listing_id)
                        is ListingDataDeltaKind.INSUFFICIENT_RESEARCH_HISTORY
                        or (
                            listing_id in candidate_ranges
                            and candidate_ranges[listing_id][0] > delta.research_history_start
                        )
                    )
                    else QuarantineReason.QUALITY_NOT_QUALIFIED
                ),
                evidence_hash=canonical_hash(
                    ("pre-research-quarantine", delta.delta_hash, listing_id)
                ),
            )
            for listing_id in candidate.ordered_candidate_listing_ids
            if listing_id not in active_set
        )
        staged = stage_qualified_delta(
            delta=delta,
            ordered_qualified_listing_ids=listing_ids,
            quarantines=quarantines,
            staged_child_hashes=child_hashes,
            adjusted_return_revision_cursor=adjusted_revision.cursor,
            adjusted_return_revision_chain_hash=adjusted_revision.chain_hash,
        )
        watermark = self.market_data.execution_source_watermark(
            manifest, through=delta.target_market_session
        )
        next_head = build_verified_head(
            kind="VerifiedPreResearchHead",
            market_profile_id=candidate.market_profile_id,
            manifest_revision=manifest.revision_sha256,
            source_revision=candidate.source_revision,
            membership_fingerprint=candidate.source_membership_fingerprint,
            ordered_source_candidate_listing_ids=candidate.ordered_candidate_listing_ids,
            ordered_listing_ids=listing_ids,
            listing_set_hash=canonical_hash(listing_ids),
            research_history_start=delta.research_history_start,
            target_market_session=delta.target_market_session,
            sector_revision=str(lineage["sector_revision"]),
            catalog_hash=str(lineage["catalog_hash"]),
            panel_snapshot_hash=str(panel["snapshot_hash"]),
            factor_screening_result_hash=projection.factor_screening_result_hash,
            factor_candidate_slate_hash=projection.factor_candidate_slate_hash,
            research_desk_factor_input_hash=projection.research_desk_factor_input_hash,
            causal_execution_outcome_snapshot_hash=projection.execution_outcome_snapshot_hash,
            research_foundation_hash=foundation.foundation_hash,
            research_foundation_marker_hash=foundation_marker.marker_hash,
            ordered_factor_ids=foundation.ordered_factor_ids,
            data_state_hash=canonical_hash(
                {
                    "manifest_revision": manifest.revision_sha256,
                    "panel_content_hash": panel["panel_content_hash"],
                    "source_watermark_hash": watermark["watermark_hash"],
                    "adjusted_return_revision_cursor": adjusted_revision.cursor,
                    "adjusted_return_revision_chain_hash": adjusted_revision.chain_hash,
                }
            ),
            adjusted_return_revision_cursor=adjusted_revision.cursor,
            adjusted_return_revision_chain_hash=adjusted_revision.chain_hash,
            validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
            is_point_in_time_historical=False,
            verified_at=published_at,
        )
        transition_id = None
        if staged.active_set_changed:
            requirement = self.foundation_state.feature_universe_rebuild_for_manifest(
                manifest.revision_sha256
            )
            if requirement is None:
                raise ValueError("PRE_RESEARCH_MEMBERSHIP_REBUILD_REQUIREMENT_MISSING")
            transition_id = requirement.transition_id
        marker = self.activate_staged_revision(
            candidate=candidate,
            delta=delta,
            plan=plan,
            staged=staged,
            next_head=next_head,
            storage_evidence=self._panel_storage_evidence(panel, prior_head=current),
            published_at=published_at,
            mandate=mandate,
            transition_id=transition_id,
        )
        self.heads.clear_pending_transition(expected_pending_hash=pending.pending_hash)
        return marker

    def _publish_progress(
        self,
        operation_id: str,
        status: Literal["RUNNING", "SUCCEEDED", "FAILED", "REUSED_EXACT"],
        completed_units: int,
        *,
        current_item: str,
        counters: Mapping[str, int] | None = None,
    ) -> None:
        if self.progress_publisher is None:
            return
        self.progress_publisher.publish(
            WorkProgressUpdate(
                operation_id=operation_id,
                stage_id="pre_research_delta_gate",
                status=status,
                completed_units=completed_units,
                total_units=3,
                unit_name="transition_steps",
                current_item=current_item,
                counters=dict(counters or {}),
            )
        )

    def _panel_storage_evidence(
        self,
        panel: Mapping[str, object],
        *,
        prior_head: VerifiedPreResearchHead | None,
    ) -> PreResearchStorageEvidence:
        chunk_hashes = tuple(str(value["chunk_hash"]) for value in panel.get("chunks", ()))  # type: ignore[index]
        prior_chunk_hashes: set[str] = set()
        if prior_head is not None:
            prior_panel = self.resolver.load_feature_panel_manifest(
                ArtifactResolver.feature_panel_manifest_uri(prior_head.panel_snapshot_hash)
            )
            prior_chunk_hashes = {
                str(value["chunk_hash"])
                for value in prior_panel.get("chunks", ())  # type: ignore[index]
            }
        compressed = 0
        uncompressed = 0
        for chunk_hash in chunk_hashes:
            path = self.artifact_root / "feature-panel" / "chunks" / f"{chunk_hash}.parquet"
            compressed += path.stat().st_size
            metadata = pq.ParquetFile(path).metadata
            uncompressed += sum(
                metadata.row_group(group).column(column).total_uncompressed_size
                for group in range(metadata.num_row_groups)
                for column in range(metadata.num_columns)
            )
        reachable_hashes: set[str] = set(chunk_hashes)
        manifests_root = self.artifact_root / "feature-panel" / "manifests"
        for manifest_path in manifests_root.glob("*.json"):
            historic = self.resolver.load_feature_panel_manifest(
                ArtifactResolver.feature_panel_manifest_uri(manifest_path.stem)
            )
            reachable_hashes.update(
                str(value["chunk_hash"])
                for value in historic.get("chunks", ())  # type: ignore[index]
            )
        reachable = sum(
            (self.artifact_root / "feature-panel" / "chunks" / f"{chunk_hash}.parquet")
            .stat()
            .st_size
            for chunk_hash in reachable_hashes
        )
        reused = sum(chunk_hash in prior_chunk_hashes for chunk_hash in chunk_hashes)
        return PreResearchStorageEvidence(
            compressed_bytes=compressed,
            uncompressed_bytes=uncompressed,
            chunk_count=len(chunk_hashes),
            newly_written_chunks=len(chunk_hashes) - reused,
            content_reused_chunks=reused,
            cumulative_reachable_panel_bytes=reachable,
            reachability_class="HEAD",
        )


__all__ = ["PRE_RESEARCH_DELTA_POLICY_HASHES", "PreResearchDeltaService"]
