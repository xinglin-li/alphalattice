"""Host-owned readiness gate for the current-US research workspace.

The gate runs before a conversational Front Desk exists. It owns current
universe freshness, user consent, and immutable manifest transitions; agents
receive only its already-activated manifest projection.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from functools import partial
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from alphalattice.control.data_platform.candidate_requalification import (
    CandidateDataRequalification,
)
from alphalattice.control.workspace_runtime.network_access import network_access
from alphalattice.control.workspace_runtime.storage.readiness import WorkspaceReadinessRecord
from alphalattice.foundation.feature_engine.inputs.contracts import (
    FEATURE_QUALIFICATION_DOMAINS,
    FeatureCandidateRecheck,
)
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.runtime.universe_onboarding import (
    CurrentUniverseOnboarding,
    CurrentUniverseOnboardingOutcome,
    _ten_calendar_year_anniversary,
)
from alphalattice.foundation.market_data_ops.sources.contracts import CandidateDataRecheck
from alphalattice.foundation.market_data_ops.sources.manifest import (
    UniverseManifest,
    build_current_index_acquisition_manifest,
    build_quality_filtered_research_manifest,
    current_index_profile_id,
)
from alphalattice.foundation.market_data_ops.sources.membership import (
    UniverseSourceObservation,
    journal_members,
    membership_effective_session,
)
from alphalattice.foundation.market_data_ops.sources.providers import (
    MarketDataProvider,
    YFinanceMarketDataProvider,
)
from alphalattice.foundation.market_data_ops.sources.universe import (
    CurrentUniverseBootstrap,
    bootstrap_from_candidate_manifest_document,
    candidate_manifest_document,
    discover_current_universe_candidates,
    membership_fingerprint,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.foundation.research_foundation.storage.repository import (
    ResearchFoundationStateRepository,
)
from alphalattice.kernel.data.calendar import materialize_calendar_schedule


class WorkspaceReadinessStatus(StrEnum):
    """Describe initialization, source verification, Feature building and transition readiness."""

    INITIALIZATION_REQUIRED = "INITIALIZATION_REQUIRED"
    FEATURE_BUILDING = "FEATURE_BUILDING"
    RESEARCH_READY = "RESEARCH_READY"
    MANIFEST_UPDATE_PENDING = "MANIFEST_UPDATE_PENDING"
    ONBOARDING_IN_PROGRESS = "ONBOARDING_IN_PROGRESS"
    SOURCE_VERIFICATION_BLOCKED = "SOURCE_VERIFICATION_BLOCKED"
    SOURCE_CHECK_REQUIRED = "SOURCE_CHECK_REQUIRED"


class WorkspaceConsentAction(StrEnum):
    """Distinguish consent to initial acquisition from consent to refresh membership."""

    INITIALIZE = "INITIALIZE"
    REFRESH_MANIFEST = "REFRESH_MANIFEST"


@dataclass(frozen=True)
class WorkspaceReadinessConsent:
    """Retain one recorded Human consent action for a workspace market profile.

    Attributes:
        consent_id: Consent record UUID.
        market_profile_id: Market profile within the consent scope.
        action: Initialization or membership-refresh consent.
        approved_at: Recorded approval clock.
    """

    consent_id: UUID
    market_profile_id: str
    action: WorkspaceConsentAction
    approved_at: datetime


@dataclass(frozen=True)
class ManifestRefreshProposal:
    """Describe the exact candidate membership transition requiring admission.

    Attributes:
        transition_id: Proposed transition identity.
        additions: Proposed symbol additions.
        removals: Proposed symbol removals.
        membership_fingerprint: Observed candidate membership commitment.
    """

    transition_id: str
    additions: tuple[str, ...]
    removals: tuple[str, ...]
    membership_fingerprint: str


@dataclass(frozen=True)
class WorkspaceReadinessDecision:
    """Return readiness with its activated manifest, pending proposal or stable failure.

    Attributes:
        status: Workspace readiness disposition.
        manifest: Activated research manifest when available.
        proposal: Pending membership transition when required.
        failure_code: Optional stable source/readiness cause.
    """

    status: WorkspaceReadinessStatus
    manifest: UniverseManifest | None = None
    proposal: ManifestRefreshProposal | None = None
    failure_code: str | None = None


@dataclass(frozen=True)
class ReadinessOnboarding:
    """Bind a readiness transition to the retained onboarding runner and candidate bootstrap.

    Attributes:
        runner: Bounded current-universe onboarding owner.
        bootstrap: Retained current-universe candidate discovery.
        transition_id: Membership transition being prepared.
    """

    runner: CurrentUniverseOnboarding
    bootstrap: CurrentUniverseBootstrap
    transition_id: str


SourceLoader = Callable[..., CurrentUniverseBootstrap]


def guarded_membership_source(
    *, observed_at: datetime, workspace: Path | None = None
) -> CurrentUniverseBootstrap:
    """Discover current membership only when the typed network control admits access.

    Args:
        observed_at: Operational observation clock; callers supply an aware instant.
        workspace: Workspace root owning the local records.

    Returns:
        The official current-universe candidate bootstrap.

    Raises:
        ValueError: Workspace network access is not admitted.
    """
    if not network_access(workspace).allowed:
        raise ValueError("workspace_readiness.source_access_not_admitted")
    return discover_current_universe_candidates(observed_at=observed_at)


def build_workspace_readiness(
    market: MarketDataRepository,
    *,
    profile_path: Path,
    provider: MarketDataProvider | None = None,
    source_loader: SourceLoader | None = None,
) -> WorkspaceReadinessGate:
    """One composition for CLI, initial preparation and subsequent updates."""
    return WorkspaceReadinessGate(
        market_data=market,
        feature_state=FeatureStateRepository(market.database, market_data=market),
        panel_state=PanelStateRepository(market.database, market_data=market),
        profile_path=profile_path,
        provider=provider
        if provider is not None
        else YFinanceMarketDataProvider(market.workspace / "yfinance-cache"),
        source_loader=source_loader
        if source_loader is not None
        else partial(guarded_membership_source, workspace=market.workspace),
    )


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


SOURCE_FAILURE_RETRY_DELAY = timedelta(minutes=5)


def source_verification_failed(
    last_checked_at: datetime | None, failed_at: datetime | None
) -> bool:
    """Compare the latest failed source check against the latest successful check.

    Args:
        last_checked_at: Last successful source verification clock, if any.
        failed_at: Last source verification failure clock, if any.

    Returns:
        Whether a failure exists and is later than any successful verification.
    """
    return failed_at is not None and (
        last_checked_at is None or _utc(failed_at) > _utc(last_checked_at)
    )


def source_check_due(
    last_checked_at: datetime | None,
    observed_at: datetime,
    *,
    reference_session: date,
    failed_at: datetime | None = None,
    operation_started_at: datetime | None = None,
) -> bool:
    """Whether the membership source has yet to be observed for a session.

    This decides whether another explicit check is needed, not when its result
    was knowable. A fresh check during historical catch-up does not establish
    historical membership. Market Data records checks separately from member
    changes and resolves their effective sessions against the exchange clocks.
    """
    # A failed check is never a successful observation. The operation may
    # finish with a disclosed last-known roster, then retry on a bounded clock.
    if source_verification_failed(last_checked_at, failed_at):
        assert failed_at is not None
        if _utc(observed_at) < _utc(failed_at) + SOURCE_FAILURE_RETRY_DELAY or (
            operation_started_at is not None and _utc(failed_at) >= _utc(operation_started_at)
        ):
            return False
    return last_checked_at is None or _utc(last_checked_at).date() < reference_session


def _transition_id(
    *,
    market_profile_id: str,
    prior_manifest_revision: str | None,
    fingerprint: str,
    additions: tuple[str, ...],
    removals: tuple[str, ...],
) -> str:
    payload = "|".join(
        (
            market_profile_id,
            prior_manifest_revision or "",
            fingerprint,
            ",".join(additions),
            ",".join(removals),
            "workspace-readiness-manifest-transition",
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def daily_source_ready_at(session_close: datetime) -> datetime:
    """One operational provider-finality rule for readiness and scheduled checks."""
    if session_close.tzinfo is None or session_close.utcoffset() is None:
        raise ValueError("workspace_readiness.session_close_timestamp_not_timezone_aware")
    return session_close + timedelta(hours=2)


def _latest_common_us_session(*, on_or_before: date, observed_at: datetime) -> date:
    """Return the latest common session whose Yahoo daily bar has settled.

    Exchange close is not provider finality.  Live dogfood showed a mixed
    cohort for roughly the first half hour after close and same-session bar
    revisions on the next pull.  A two-hour operational delay keeps that
    provider instability out of canonical daily data without sleeping or
    asking an Agent to reinterpret partial market truth.
    """

    schedule = materialize_calendar_schedule(
        ("XNAS", "XNYS"),
        start=on_or_before - timedelta(days=14),
        end=on_or_before,
        as_of_timestamp=observed_at,
    )
    calendar_count_by_session: dict[date, int] = {}
    for row in schedule.to_pylist():
        if daily_source_ready_at(row["session_close_timestamp"]) > observed_at:
            continue
        session = row["session_date"]
        calendar_count_by_session[session] = calendar_count_by_session.get(session, 0) + 1
    shared_sessions = [
        session for session, count in calendar_count_by_session.items() if count == 2
    ]
    if not shared_sessions:
        raise ValueError("could not resolve a common XNAS/XNYS as-of session")
    return max(shared_sessions)


@dataclass
class WorkspaceReadinessGate:
    """A concrete pre-Front-Desk gate; it is intentionally not a generic base."""

    market_data: MarketDataRepository
    feature_state: FeatureStateRepository
    panel_state: PanelStateRepository
    profile_path: Path
    provider: MarketDataProvider
    source_loader: SourceLoader = discover_current_universe_candidates
    """The universe membership source, held like ``provider`` as composition state.

    Both are network edges, and only the composition root knows which one a
    deployment should use. Leaving this as a per-call default made the loader
    unreachable from callers that own the gate rather than the call, so a
    scheduled assessment always fell back to the live network fetch.
    """

    def __post_init__(self) -> None:
        """Resolve the installed profile and the candidate Data requalification owner."""
        self.profile_path = Path(self.profile_path)
        self.market_profile_id = self._market_profile_id()
        self.candidate_data = CandidateDataRequalification(
            market_data=self.market_data,
            market_profile_id=self.market_profile_id,
            profile_path=self.profile_path,
            provider=self.provider,
            readiness_record=self._require_record,
            research_history_start=self._inherited_history_start,
        )

    def assess(
        self, *, observed_at: datetime, target_session: date | None = None
    ) -> WorkspaceReadinessDecision:
        """Read local readiness only: no acquisition, proposal or state migration.

        An overdue membership check is work to request, not permission to fetch
        from a page read or to call the prior manifest current. The check is
        due for ``target_session`` when one is being updated, else for the
        latest settled session at ``observed_at``.
        """
        record = self.market_data.readiness.load(self.market_profile_id)
        if record is None:
            return WorkspaceReadinessDecision(WorkspaceReadinessStatus.INITIALIZATION_REQUIRED)
        if record.status == WorkspaceReadinessStatus.ONBOARDING_IN_PROGRESS.value:
            return WorkspaceReadinessDecision(WorkspaceReadinessStatus.ONBOARDING_IN_PROGRESS)
        if record.status == WorkspaceReadinessStatus.MANIFEST_UPDATE_PENDING.value:
            return self._pending_decision(record)
        if self._is_source_check_due(record, _utc(observed_at), target_session=target_session):
            return WorkspaceReadinessDecision(
                WorkspaceReadinessStatus.SOURCE_CHECK_REQUIRED,
                failure_code="workspace_readiness.source_check_due",
            )
        return self._ready_or_blocked(record)

    def refresh_sources_if_due(
        self,
        *,
        observed_at: datetime,
        source_loader: SourceLoader | None = None,
        before_fetch: Callable[[], object] | None = None,
        target_session: date | None = None,
        operation_started_at: datetime | None = None,
    ) -> WorkspaceReadinessDecision:
        """Explicit update command; unchanged consent gates still own activation.

        The source load is this command's network edge; ``before_fetch`` is
        called right before it so a caller can release whatever it retained
        for the rest of its own preflight. It is not called when no check is
        due.
        """
        loader = source_loader if source_loader is not None else self.source_loader
        now = _utc(observed_at)
        record = self.market_data.readiness.load(self.market_profile_id)
        if record is None:
            return WorkspaceReadinessDecision(WorkspaceReadinessStatus.INITIALIZATION_REQUIRED)
        if record.status == WorkspaceReadinessStatus.ONBOARDING_IN_PROGRESS.value:
            return WorkspaceReadinessDecision(WorkspaceReadinessStatus.ONBOARDING_IN_PROGRESS)
        if record.status == WorkspaceReadinessStatus.MANIFEST_UPDATE_PENDING.value:
            return self._pending_decision(record)
        if not self._is_source_check_due(
            record, now, target_session=target_session, operation_started_at=operation_started_at
        ):
            return self._ready_or_blocked(record, persist_transition=True)
        if before_fetch is not None:
            before_fetch()
        try:
            bootstrap = loader(observed_at=now)
        except Exception:
            retained = self._carry_forward_source(record, observed_at=now)
            if retained is not None:
                return retained
            self._save(
                record,
                status=WorkspaceReadinessStatus.SOURCE_VERIFICATION_BLOCKED,
                failure_code="workspace_readiness.source_verification_failed",
                observed_at=now,
            )
            return WorkspaceReadinessDecision(
                WorkspaceReadinessStatus.SOURCE_VERIFICATION_BLOCKED,
                failure_code="workspace_readiness.source_verification_failed",
            )
        fingerprint = membership_fingerprint(bootstrap.source_manifest)
        self.market_data.record_universe_source_observation(
            UniverseSourceObservation(
                market_profile_id=self.market_profile_id,
                observed_at=now,
                previous_observed_at=(
                    _utc(record.last_checked_at) if record.last_checked_at is not None else None
                ),
                candidate_membership_hash=fingerprint,
                source_identity_hash=bootstrap.source_manifest.content_hash,
                first_eligible_session=membership_effective_session(
                    observed_at=now,
                    decided_at=now,
                    not_before=now.date() - timedelta(days=2),
                ),
            )
        )
        if fingerprint == record.active_membership_fingerprint:
            self._save(
                record,
                status=(
                    WorkspaceReadinessStatus.FEATURE_BUILDING
                    if record.status == WorkspaceReadinessStatus.FEATURE_BUILDING.value
                    else WorkspaceReadinessStatus.RESEARCH_READY
                ),
                last_checked_at=now,
                failure_code=None,
                observed_at=now,
            )
            return self._ready_or_blocked(self._require_record(), persist_transition=True)
        return self._propose_manifest_refresh(record, bootstrap=bootstrap, observed_at=now)

    def _carry_forward_source(
        self,
        record: WorkspaceReadinessRecord,
        *,
        observed_at: datetime,
    ) -> WorkspaceReadinessDecision | None:
        """Retain verified approval, never invent a source observation or membership."""
        if record.active_candidate_manifest_document is None or record.active_manifest_id is None:
            return None
        approved = bootstrap_from_candidate_manifest_document(
            record.active_candidate_manifest_document
        )
        manifest = self._active_manifest(record)
        if membership_fingerprint(
            approved.source_manifest
        ) != record.active_membership_fingerprint or (
            {item.symbol for item in manifest.listings} - set(approved.candidate_symbols)
        ):
            return None
        updated = self._save(
            record,
            status=(
                WorkspaceReadinessStatus.FEATURE_BUILDING
                if record.status == "FEATURE_BUILDING"
                else WorkspaceReadinessStatus.RESEARCH_READY
            ),
            observed_at=observed_at,
            source_check_failed_at=observed_at,
        )
        return self._ready_or_blocked(updated, persist_transition=True)

    def plan_feature_candidate_recheck(
        self, *, observed_at: datetime
    ) -> FeatureCandidateRecheck | None:
        """Read due Feature/Sector prerequisites without fetching or admitting a roster."""
        record = self.market_data.readiness.load(self.market_profile_id)
        parent = self.market_data.source_admission_manifest(
            market_profile_id=self.market_profile_id
        )
        if record is None or record.active_manifest_id is None or parent is None:
            return None
        if self.market_data.universe_bootstrap(self.market_profile_id) is None:
            return None
        current = self._active_manifest(record)
        existing = {item.listing_id for item in current.listings}
        raw_qualified = {item.listing_id for item in parent.listings}
        due = tuple(
            sorted(
                {
                    item.listing_id
                    for domain in FEATURE_QUALIFICATION_DOMAINS
                    for item in self.panel_state.active_listing_quarantines(
                        parent.revision_sha256,
                        include_profile_history=True,
                        qualification_domain=domain,
                    )
                    if item.listing_id in raw_qualified - existing
                    and item.recheck_after_at <= _utc(observed_at)
                }
            )
        )
        if not due:
            return None
        # Revisit the source root's pending batch when a Feature prerequisite is due.
        # Keep that existing root (and its admission lineage), not a new working
        # manifest which pretends preparation itself was a qualification.
        scope = FeatureCandidateRecheck(
            parent.revision_sha256, tuple(sorted(raw_qualified - existing))
        )
        self.resolve_feature_candidate_recheck(scope, prior_revision=current.revision_sha256)
        return scope

    def resolve_feature_candidate_recheck(
        self,
        scope: FeatureCandidateRecheck,
        *,
        prior_revision: str,
        data_recheck: CandidateDataRecheck | None = None,
        target_session: date | None = None,
    ) -> UniverseManifest:
        """The exact prepared scope must still belong to the approved source.

        The scope was planned against the source parent of its day. The one
        way that parent may have moved for the same request is the request's
        own raw retry admitting candidates under its recorded transition
        (``data_recheck`` names it; ``CandidateDataRequalification.activated_transition``
        answers it): the scope then resolves across that transition -- its
        planned parent must be the transition's prior, the current parent the
        transition's next, the fingerprint the approved source's, and the
        listings the transition admitted are the retry's own -- and the same
        range rules hold on the successor: the scope names no prior member
        and no admitted candidate, and prior members, the scope and the
        admitted candidates are exactly the parent. Any other move of the
        source refuses by name; nothing here widens a scope.
        """
        record = self._require_record()
        parent = self.market_data.source_admission_manifest(
            market_profile_id=self.market_profile_id
        )
        if parent is None or record.active_candidate_manifest_document is None:
            raise ValueError("workspace_maintenance.candidate_recheck_source_changed")
        parent_ids = {item.listing_id for item in parent.listings}
        admitted_ids: set[str] = set()
        if parent.revision_sha256 != scope.parent_manifest_revision:
            transition = (
                self.candidate_data.activated_transition(
                    data_recheck, target_session=target_session
                )
                if data_recheck is not None and target_session is not None
                else None
            )
            if (
                transition is None
                or data_recheck is None
                or data_recheck.qualified_parent_revision != scope.parent_manifest_revision
                or transition.market_profile_id != self.market_profile_id
                or transition.prior_manifest_revision != scope.parent_manifest_revision
                or transition.next_manifest_revision != parent.revision_sha256
                or transition.membership_fingerprint != record.active_membership_fingerprint
                or transition.removals
            ):
                raise ValueError("workspace_maintenance.candidate_recheck_source_changed")
            admitted_symbols = set(transition.additions)
            admitted_ids = {
                item.listing_id for item in parent.listings if item.symbol in admitted_symbols
            }
            if len(admitted_ids) != len(admitted_symbols) or not admitted_ids <= set(
                data_recheck.listing_ids
            ):
                raise ValueError("workspace_maintenance.candidate_recheck_source_changed")
        approved = bootstrap_from_candidate_manifest_document(
            record.active_candidate_manifest_document
        )
        prior = self.market_data.load_universe_manifest_revision(prior_revision)
        prior_ids = {item.listing_id for item in prior.listings}
        if (
            parent.profile.market_profile_id != self.market_profile_id
            or prior.profile != parent.profile
            or parent.membership_fingerprint != record.active_membership_fingerprint
            or membership_fingerprint(approved.source_manifest)
            != record.active_membership_fingerprint
            or {item.symbol for item in parent.listings} - set(approved.candidate_symbols)
            or set(scope.listing_ids) & prior_ids
        ):
            raise ValueError("workspace_maintenance.candidate_recheck_source_changed")
        if (
            set(scope.listing_ids) & admitted_ids
            or prior_ids | set(scope.listing_ids) | admitted_ids != parent_ids
        ):
            raise ValueError("workspace_maintenance.candidate_recheck_scope_invalid")
        return parent

    def start_approved_onboarding(
        self,
        consent: WorkspaceReadinessConsent,
        *,
        source_loader: SourceLoader | None = None,
        target_session: date | None = None,
    ) -> ReadinessOnboarding:
        """Create a data task only after host-owned approval has been verified."""
        loader = source_loader if source_loader is not None else self.source_loader
        now = _utc(consent.approved_at)
        if consent.market_profile_id != self.market_profile_id:
            raise ValueError("workspace readiness consent targets another market profile")
        record = self.market_data.readiness.load(self.market_profile_id)
        if record is None:
            if consent.action is not WorkspaceConsentAction.INITIALIZE:
                raise ValueError("initial workspace requires initialization consent")
            bootstrap = loader(observed_at=now)
            proposal = self._initial_proposal(bootstrap)
            inherited_listing_ids: tuple[str, ...] = ()
        else:
            if (
                record.status != WorkspaceReadinessStatus.MANIFEST_UPDATE_PENDING.value
                or consent.action is not WorkspaceConsentAction.REFRESH_MANIFEST
                or record.pending_candidate_manifest_document is None
                or record.pending_membership_fingerprint is None
            ):
                raise ValueError("workspace readiness has no matching manifest-refresh approval")
            bootstrap = bootstrap_from_candidate_manifest_document(
                record.pending_candidate_manifest_document
            )
            proposal = self._pending_proposal(record, bootstrap)
            inherited_listing_ids = self._inherited_listing_ids(record, bootstrap)
        latest = _latest_common_us_session(on_or_before=now.date(), observed_at=now)
        if target_session is not None and target_session > latest:
            raise ValueError("workspace_readiness.target_not_available_at_approval")
        qualification_start = (
            max(
                self._inherited_history_start(record),
                _ten_calendar_year_anniversary(target_session or latest),
            )
            if record
            else None
        )
        runner = CurrentUniverseOnboarding(
            store=self.market_data,
            bootstrap=bootstrap,
            profile_path=self.profile_path,
            provider=self.provider,
            as_of_session=target_session or latest,
            history_start=qualification_start,
            retained_listing_ids=inherited_listing_ids,
        )
        self.market_data.bootstrap(runner.acquisition_manifest)
        # Persist the reconstructable Data task before exposing resumable
        # readiness. A crash between these writes re-admits the same task.
        runner.admit(observed_at=now)
        if record is None:
            self.market_data.record_manifest_transition(
                transition_id=proposal.transition_id,
                market_profile_id=self.market_profile_id,
                prior_manifest_revision=None,
                membership_fingerprint=proposal.membership_fingerprint,
                additions=proposal.additions,
                removals=proposal.removals,
                lifecycle="ONBOARDING_IN_PROGRESS",
                approved_at=now,
                created_at=now,
            )
        else:
            self.market_data.approve_manifest_transition(proposal.transition_id, approved_at=now)
        active_document = record.active_candidate_manifest_document if record else None
        active_fingerprint = record.active_membership_fingerprint if record else None
        active_manifest_id = record.active_manifest_id if record else None
        active_manifest_revision = record.active_manifest_revision if record else None
        self.market_data.readiness.save(
            market_profile_id=self.market_profile_id,
            status=WorkspaceReadinessStatus.ONBOARDING_IN_PROGRESS.value,
            active_manifest_id=active_manifest_id,
            active_manifest_revision=active_manifest_revision,
            active_membership_fingerprint=active_fingerprint,
            active_candidate_manifest_document=active_document,
            pending_membership_fingerprint=proposal.membership_fingerprint,
            pending_candidate_manifest_document=candidate_manifest_document(
                bootstrap.source_manifest
            ),
            last_checked_at=now,
            last_changed_at=(record.last_changed_at if record else None),
            failure_code=None,
            observed_at=now,
        )
        return ReadinessOnboarding(
            runner=runner, bootstrap=bootstrap, transition_id=proposal.transition_id
        )

    def resume_onboarding(self) -> ReadinessOnboarding:
        """Reconstruct the same durable onboarding task without a second consent."""
        record = self.market_data.readiness.load(self.market_profile_id)
        if (
            record is None
            or record.status != WorkspaceReadinessStatus.ONBOARDING_IN_PROGRESS.value
            or record.pending_candidate_manifest_document is None
        ):
            raise ValueError("workspace readiness has no resumable onboarding")
        persisted = self.market_data.resumable_current_universe_onboarding_input(
            market_profile_id=self.market_profile_id
        )
        if persisted is None:
            raise ValueError("workspace readiness onboarding has no resumable task input")
        _onboarding_id, document, history_start, as_of_session = persisted
        bootstrap = bootstrap_from_candidate_manifest_document(document)
        proposal = (
            self._initial_proposal(bootstrap)
            if record.active_manifest_id is None
            else self._pending_proposal(record, bootstrap)
        )
        return ReadinessOnboarding(
            runner=CurrentUniverseOnboarding(
                store=self.market_data,
                bootstrap=bootstrap,
                profile_path=self.profile_path,
                provider=self.provider,
                as_of_session=as_of_session,
                history_start=history_start,
                retained_listing_ids=self._inherited_listing_ids(record, bootstrap),
            ),
            bootstrap=bootstrap,
            transition_id=proposal.transition_id,
        )

    def complete_onboarding(
        self,
        onboarding: ReadinessOnboarding,
        outcome: CurrentUniverseOnboardingOutcome,
        *,
        observed_at: datetime,
    ) -> WorkspaceReadinessDecision:
        """Activate qualified candidates while preserving observed nominal incumbents."""
        now = _utc(observed_at)
        if outcome.status.value != "completed" or outcome.research_manifest is None:
            return WorkspaceReadinessDecision(WorkspaceReadinessStatus.ONBOARDING_IN_PROGRESS)
        proposed = outcome.research_manifest
        established = self.market_data.universe_bootstrap(self.market_profile_id)
        if established is not None:
            nominal = set(
                journal_members(
                    established, self.market_data.membership_events(self.market_profile_id)
                )
            )
            acquisition = build_current_index_acquisition_manifest(
                self.profile_path, onboarding.bootstrap
            )
            observed_ids = {item.listing_id for item in acquisition.listings}
            proposed_ids = {item.listing_id for item in proposed.listings}
            retained_ids = (nominal & observed_ids) | proposed_ids
            if retained_ids != proposed_ids:
                # Data failure is not index departure. Reuse Data's identity
                # resolver; do not fabricate bars or newly qualify these old members.
                listings = self.market_data.listing_scope(proposed, listing_ids=tuple(retained_ids))
                proposed = replace(
                    build_quality_filtered_research_manifest(
                        acquisition,
                        eligible_listing_ids=tuple(retained_ids),
                        qualification_obligations=proposed.obligations_for_derivation(),
                    ),
                    listings=tuple(sorted(listings, key=lambda item: item.symbol)),
                )
        fingerprint = membership_fingerprint(onboarding.bootstrap.source_manifest)
        record = self.market_data.readiness.load(self.market_profile_id)
        if record is not None and record.active_manifest_id is not None:
            active = self._active_manifest(record)
            active_ids = tuple(sorted(value.listing_id for value in active.listings))
            qualified_ids = tuple(sorted(value.listing_id for value in proposed.listings))
            if active_ids == qualified_ids:
                self.market_data.record_source_change_without_active_set_change(
                    transition_id=onboarding.transition_id,
                    active_manifest=active,
                    membership_fingerprint=fingerprint,
                    candidate_manifest_document=candidate_manifest_document(
                        onboarding.bootstrap.source_manifest
                    ),
                    checked_at=now,
                    changed_at=now,
                )
                return WorkspaceReadinessDecision(
                    WorkspaceReadinessStatus.FEATURE_BUILDING,
                    manifest=active,
                )
        self.market_data.activate_workspace_manifest(
            transition_id=onboarding.transition_id,
            manifest=proposed,
            membership_fingerprint=fingerprint,
            candidate_manifest_document=candidate_manifest_document(
                onboarding.bootstrap.source_manifest
            ),
            checked_at=now,
            changed_at=now,
            within_activation=ResearchFoundationStateRepository.require_rebuild,
        )
        return WorkspaceReadinessDecision(
            WorkspaceReadinessStatus.FEATURE_BUILDING,
            manifest=proposed,
        )

    def _ready_or_blocked(
        self, record: WorkspaceReadinessRecord, *, persist_transition: bool = False
    ) -> WorkspaceReadinessDecision:
        if record.status == WorkspaceReadinessStatus.FEATURE_BUILDING.value:
            return WorkspaceReadinessDecision(
                WorkspaceReadinessStatus.FEATURE_BUILDING,
                manifest=self._active_manifest(record),
            )
        if record.status != WorkspaceReadinessStatus.RESEARCH_READY.value:
            return WorkspaceReadinessDecision(
                WorkspaceReadinessStatus.SOURCE_VERIFICATION_BLOCKED,
                failure_code=record.failure_code or "workspace_readiness.not_ready",
            )
        try:
            manifest = self._active_manifest(record)
            panel = self.panel_state.active_feature_panel(self.market_profile_id)
            snapshot = self.panel_state.feature_panel_snapshot_for_active(self.market_profile_id)
            spy_range = self.market_data.market_reference_raw_range("SPY")
            spy_reference = self.feature_state.market_reference("SPY")
            if (
                panel is None
                or str(panel["manifest_revision"]) != manifest.revision_sha256
                or snapshot is None
                or str(snapshot["manifest_revision"]) != manifest.revision_sha256
                or spy_reference is None
                or str(panel["spy_revision"]) != str(spy_reference["revision_hash"])
                or spy_range is None
                or spy_range[0] > panel["history_start"]
                or spy_range[1] < panel["as_of_session"]
            ):
                # Deterministic migration for older raw/action-only workspaces:
                # no panel/snapshot means no research-ready admission.
                if persist_transition:
                    self._save(
                        record,
                        status=WorkspaceReadinessStatus.FEATURE_BUILDING,
                        failure_code=None,
                        observed_at=_utc(record.updated_at),
                    )
                return WorkspaceReadinessDecision(
                    WorkspaceReadinessStatus.FEATURE_BUILDING,
                    manifest=manifest,
                )
            return WorkspaceReadinessDecision(
                WorkspaceReadinessStatus.RESEARCH_READY,
                manifest=manifest,
            )
        except ValueError:
            return WorkspaceReadinessDecision(
                WorkspaceReadinessStatus.SOURCE_VERIFICATION_BLOCKED,
                failure_code="workspace_readiness.active_manifest_missing",
            )

    def _pending_decision(self, record: WorkspaceReadinessRecord) -> WorkspaceReadinessDecision:
        if (
            record.pending_candidate_manifest_document is None
            or record.pending_membership_fingerprint is None
        ):
            return WorkspaceReadinessDecision(
                WorkspaceReadinessStatus.SOURCE_VERIFICATION_BLOCKED,
                failure_code="workspace_readiness.pending_manifest_missing",
            )
        bootstrap = bootstrap_from_candidate_manifest_document(
            record.pending_candidate_manifest_document
        )
        return WorkspaceReadinessDecision(
            WorkspaceReadinessStatus.MANIFEST_UPDATE_PENDING,
            proposal=self._pending_proposal(record, bootstrap),
        )

    def _propose_manifest_refresh(
        self,
        record: WorkspaceReadinessRecord,
        *,
        bootstrap: CurrentUniverseBootstrap,
        observed_at: datetime,
    ) -> WorkspaceReadinessDecision:
        proposal = self._pending_proposal(record, bootstrap)
        self.market_data.bootstrap(self._active_manifest(record))
        self.market_data.record_manifest_transition(
            transition_id=proposal.transition_id,
            market_profile_id=self.market_profile_id,
            prior_manifest_revision=record.active_manifest_revision,
            membership_fingerprint=proposal.membership_fingerprint,
            additions=proposal.additions,
            removals=proposal.removals,
            lifecycle="PROPOSED",
            approved_at=None,
            created_at=observed_at,
        )
        self._save(
            record,
            status=WorkspaceReadinessStatus.MANIFEST_UPDATE_PENDING,
            pending_membership_fingerprint=proposal.membership_fingerprint,
            pending_candidate_manifest_document=candidate_manifest_document(
                bootstrap.source_manifest
            ),
            last_checked_at=observed_at,
            failure_code=None,
            observed_at=observed_at,
        )
        return WorkspaceReadinessDecision(
            WorkspaceReadinessStatus.MANIFEST_UPDATE_PENDING,
            proposal=proposal,
        )

    def _initial_proposal(self, bootstrap: CurrentUniverseBootstrap) -> ManifestRefreshProposal:
        additions = bootstrap.candidate_symbols
        fingerprint = membership_fingerprint(bootstrap.source_manifest)
        return ManifestRefreshProposal(
            transition_id=_transition_id(
                market_profile_id=self.market_profile_id,
                prior_manifest_revision=None,
                fingerprint=fingerprint,
                additions=additions,
                removals=(),
            ),
            additions=additions,
            removals=(),
            membership_fingerprint=fingerprint,
        )

    def _pending_proposal(
        self, record: WorkspaceReadinessRecord, bootstrap: CurrentUniverseBootstrap
    ) -> ManifestRefreshProposal:
        if record.active_candidate_manifest_document is None:
            raise ValueError("active candidate manifest is missing")
        previous = bootstrap_from_candidate_manifest_document(
            record.active_candidate_manifest_document
        )
        additions = tuple(
            sorted(set(bootstrap.candidate_symbols) - set(previous.candidate_symbols))
        )
        removals = tuple(sorted(set(previous.candidate_symbols) - set(bootstrap.candidate_symbols)))
        fingerprint = membership_fingerprint(bootstrap.source_manifest)
        return ManifestRefreshProposal(
            transition_id=_transition_id(
                market_profile_id=self.market_profile_id,
                prior_manifest_revision=record.active_manifest_revision,
                fingerprint=fingerprint,
                additions=additions,
                removals=removals,
            ),
            additions=additions,
            removals=removals,
            membership_fingerprint=fingerprint,
        )

    def _active_manifest(self, record: WorkspaceReadinessRecord) -> UniverseManifest:
        if record.active_manifest_id is None:
            raise ValueError("workspace has no active manifest")
        manifest = self.market_data.load_universe_manifest(record.active_manifest_id)
        if manifest.revision_sha256 != record.active_manifest_revision:
            raise ValueError("workspace active manifest revision does not match readiness record")
        return manifest

    def _inherited_listing_ids(
        self, record: WorkspaceReadinessRecord, bootstrap: CurrentUniverseBootstrap
    ) -> tuple[str, ...]:
        if record.active_manifest_id is None:
            return ()
        active = self._active_manifest(record)
        active_symbols = {listing.symbol for listing in active.listings}
        acquisition = build_current_index_acquisition_manifest(self.profile_path, bootstrap)
        return tuple(
            listing.listing_id
            for listing in acquisition.listings
            if listing.symbol in active_symbols
        )

    def _inherited_history_start(self, record: WorkspaceReadinessRecord) -> date:
        """Preserve the verified Research History Window across membership revisions."""

        panel = self.panel_state.active_feature_panel(self.market_profile_id)
        if panel is not None and panel.get("history_start") is not None:
            value = panel["history_start"]
            return value if isinstance(value, date) else date.fromisoformat(str(value))
        if record.active_manifest_id is None:
            raise ValueError("workspace membership revision has no inherited history authority")
        active = self._active_manifest(record)
        raw_range = self.market_data.manifest_raw_range(active)
        if raw_range is None:
            raise ValueError("workspace membership revision has no reusable history")
        start: date = raw_range[0]
        return start

    def _is_source_check_due(
        self,
        record: WorkspaceReadinessRecord,
        now: datetime,
        *,
        target_session: date | None = None,
        operation_started_at: datetime | None = None,
    ) -> bool:
        reference = (
            target_session
            if target_session is not None
            else _latest_common_us_session(on_or_before=now.date(), observed_at=now)
        )
        return source_check_due(
            record.last_checked_at,
            now,
            reference_session=reference,
            failed_at=record.source_check_failed_at,
            operation_started_at=operation_started_at,
        )

    def _save(
        self,
        record: WorkspaceReadinessRecord,
        *,
        status: WorkspaceReadinessStatus,
        observed_at: datetime,
        active_manifest_id: str | None = None,
        active_manifest_revision: str | None = None,
        active_membership_fingerprint: str | None = None,
        active_candidate_manifest_document: dict[str, object] | None = None,
        pending_membership_fingerprint: str | None = None,
        pending_candidate_manifest_document: dict[str, object] | None = None,
        last_checked_at: datetime | None = None,
        last_changed_at: datetime | None = None,
        failure_code: str | None = None,
        source_check_failed_at: datetime | None = None,
    ) -> WorkspaceReadinessRecord:
        return self.market_data.readiness.save(
            market_profile_id=self.market_profile_id,
            status=status.value,
            active_manifest_id=(
                active_manifest_id if active_manifest_id is not None else record.active_manifest_id
            ),
            active_manifest_revision=(
                active_manifest_revision
                if active_manifest_revision is not None
                else record.active_manifest_revision
            ),
            active_membership_fingerprint=(
                active_membership_fingerprint
                if active_membership_fingerprint is not None
                else record.active_membership_fingerprint
            ),
            active_candidate_manifest_document=(
                active_candidate_manifest_document
                if active_candidate_manifest_document is not None
                else record.active_candidate_manifest_document
            ),
            pending_membership_fingerprint=pending_membership_fingerprint,
            pending_candidate_manifest_document=pending_candidate_manifest_document,
            last_checked_at=last_checked_at
            if last_checked_at is not None
            else record.last_checked_at,
            last_changed_at=last_changed_at
            if last_changed_at is not None
            else record.last_changed_at,
            failure_code=failure_code,
            observed_at=observed_at,
            source_check_failed_at=source_check_failed_at,
        )

    def _require_record(self) -> WorkspaceReadinessRecord:
        record = self.market_data.readiness.load(self.market_profile_id)
        if record is None:
            raise AssertionError("workspace readiness write was not durable")
        return record

    def _market_profile_id(self) -> str:
        # The profile's owner reads it, by its contract and the one loader.
        return str(current_index_profile_id(self.profile_path))
