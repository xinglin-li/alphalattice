"""Bounded raw-candidate requalification under an already approved source scope."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

from alphalattice.control.workspace_runtime.storage.readiness import WorkspaceReadinessRecord
from alphalattice.foundation.market_data_ops.runtime.universe_onboarding import (
    CurrentUniverseOnboarding,
    CurrentUniverseOnboardingOutcome,
    _ten_calendar_year_anniversary,
)
from alphalattice.foundation.market_data_ops.sources.contracts import CandidateDataRecheck
from alphalattice.foundation.market_data_ops.sources.manifest import (
    UniverseManifest,
    build_quality_filtered_research_manifest,
)
from alphalattice.foundation.market_data_ops.sources.providers import MarketDataProvider
from alphalattice.foundation.market_data_ops.sources.universe import (
    CurrentUniverseBootstrap,
    bootstrap_from_candidate_manifest_document,
    membership_fingerprint,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    ManifestTransitionRecord,
    MarketDataRepository,
)
from alphalattice.foundation.research_foundation.storage.repository import (
    ResearchFoundationStateRepository,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_CANDIDATE_DATA_RECHECK_DELAY = timedelta(days=1)
_FAILED_CANDIDATE_STATES = frozenset({"RAW_FAILED", "QUALITY_INELIGIBLE", "AUDIT_FAILED"})


@dataclass
class CandidateDataRequalification:
    """Own bounded retries for failed candidates outside the admitted parent universe.

    The readiness record, retained onboarding disclosure and source manifest must
    agree before a retry scope is proposed. Preparing candidates does not itself
    admit them to a research universe.

    Attributes:
        market_data: Workspace market-data and onboarding records.
        market_profile_id: Market profile whose candidates are reconsidered.
        profile_path: Installed market profile configuration.
        provider: Provider used by the retained onboarding runner.
        readiness_record: Callback resolving the current readiness record.
        research_history_start: Callback resolving inherited research history.
    """

    market_data: MarketDataRepository
    market_profile_id: str
    profile_path: Path
    provider: MarketDataProvider
    readiness_record: Callable[[], WorkspaceReadinessRecord]
    research_history_start: Callable[[WorkspaceReadinessRecord], date]
    _runner: tuple[CandidateDataRecheck, date, CurrentUniverseOnboarding] | None = field(
        default=None, init=False, repr=False
    )
    # A completed recheck, remembered with the readiness record it was
    # validated against. The Data-update request carries its recheck scope for
    # every one-listing cycle of the Task; once the bounded retry has completed
    # and its manifest is known, a later cycle re-reads only the readiness
    # record and takes the same answer while that record is unchanged. Any
    # change to it (a source check, a transition, a child activation) sends the
    # cycle back through the full validation, which still refuses by name.
    _completed: (
        tuple[
            CandidateDataRecheck,
            date,
            CurrentUniverseOnboardingOutcome,
            UniverseManifest | None,
            tuple[object, ...],
        ]
        | None
    ) = field(default=None, init=False, repr=False)

    def plan(self, *, observed_at: datetime, target_session: date) -> CandidateDataRecheck | None:
        """Propose due raw-data retries from matching retained candidate evidence.

        Args:
            observed_at: Operational observation clock; callers supply an aware instant.
            target_session: Market session targeted by this operation.

        Returns:
            A bounded candidate recheck scope, or None when prerequisites or due candidates are
            absent.
        """
        record = self.market_data.readiness.load(self.market_profile_id)
        if (
            record is None
            or record.active_candidate_manifest_document is None
            or record.status not in {"RESEARCH_READY", "FEATURE_BUILDING"}
            or self.market_data.universe_bootstrap(self.market_profile_id) is None
        ):
            return None
        prior = self.market_data.latest_current_universe_onboarding_disclosure(
            market_profile_id=self.market_profile_id
        )
        parent = self.market_data.source_admission_manifest(
            market_profile_id=self.market_profile_id
        )
        if prior is None or parent is None:
            return None
        bootstrap = bootstrap_from_candidate_manifest_document(
            record.active_candidate_manifest_document
        )
        run = self.market_data.current_universe_onboarding_run(str(prior["onboarding_id"]))
        if run.candidate_manifest_hash != bootstrap.source_manifest.content_hash:
            return None
        existing = {item.listing_id for item in parent.listings}
        requested = tuple(
            sorted(
                item.listing_id
                for item in self.market_data.current_universe_onboarding_listings(run.onboarding_id)
                if item.state in _FAILED_CANDIDATE_STATES
                and item.listing_id not in existing
                and item.symbol in bootstrap.candidate_symbols
                and item.updated_at.replace(tzinfo=UTC) + _CANDIDATE_DATA_RECHECK_DELAY
                <= observed_at.astimezone(UTC)
            )
        )
        if not requested:
            return None
        scope = CandidateDataRecheck(
            run.onboarding_id,
            run.candidate_manifest_hash,
            parent.revision_sha256,
            requested,
            max(
                self.research_history_start(record),
                _ten_calendar_year_anniversary(target_session),
            ),
        )
        self._context(scope, target_session=target_session)
        return scope

    def _context(
        self, scope: CandidateDataRecheck, *, target_session: date
    ) -> tuple[WorkspaceReadinessRecord, CurrentUniverseBootstrap, UniverseManifest]:
        record = self.readiness_record()
        if (
            record.active_candidate_manifest_document is None
            or record.last_checked_at is None
            or record.pending_membership_fingerprint
        ):
            raise ValueError("workspace_maintenance.candidate_recheck_source_changed")
        bootstrap = bootstrap_from_candidate_manifest_document(
            record.active_candidate_manifest_document
        )
        parent = self.market_data.load_universe_manifest_revision(scope.qualified_parent_revision)
        run = self.market_data.current_universe_onboarding_run(scope.prior_onboarding_id)
        eligible_scope = {
            item.listing_id
            for item in self.market_data.current_universe_onboarding_listings(run.onboarding_id)
            if item.state in _FAILED_CANDIDATE_STATES and item.symbol in bootstrap.candidate_symbols
        }
        if (
            run.lifecycle != "COMPLETED"
            or run.candidate_manifest_hash != scope.candidate_manifest_hash
            or bootstrap.source_manifest.content_hash != scope.candidate_manifest_hash
            or membership_fingerprint(bootstrap.source_manifest)
            != record.active_membership_fingerprint
            or parent.profile.market_profile_id != self.market_profile_id
            or not set(scope.listing_ids) <= eligible_scope
            or set(scope.listing_ids) & {item.listing_id for item in parent.listings}
            or {item.symbol for item in parent.listings} - set(bootstrap.candidate_symbols)
        ):
            raise ValueError("workspace_maintenance.candidate_data_recheck_scope_invalid")
        permitted_roots = {parent.revision_sha256}
        try:
            transition = self.market_data.manifest_transition(
                self._transition(scope, target_session)
            )
        except ValueError as exc:
            if str(exc) != "manifest transition does not exist":
                raise
        else:
            if (
                transition.prior_manifest_revision == parent.revision_sha256
                and transition.market_profile_id == self.market_profile_id
                and transition.membership_fingerprint == record.active_membership_fingerprint
                and transition.lifecycle == "ACTIVATED"
                and transition.next_manifest_revision is not None
            ):
                permitted_roots.add(transition.next_manifest_revision)
        current = self.market_data.source_admission_manifest(
            market_profile_id=self.market_profile_id
        )
        if current is None or current.revision_sha256 not in permitted_roots:
            raise ValueError("workspace_maintenance.candidate_recheck_source_changed")
        return record, bootstrap, parent

    def _onboarding(
        self, scope: CandidateDataRecheck, *, target_session: date
    ) -> CurrentUniverseOnboarding:
        record, bootstrap, _parent = self._context(scope, target_session=target_session)
        if scope.history_start != max(
            self.research_history_start(record), _ten_calendar_year_anniversary(target_session)
        ):
            raise ValueError("workspace_maintenance.candidate_data_recheck_history_invalid")
        cached = self._runner
        if cached is None or cached[:2] != (scope, target_session):
            runner = CurrentUniverseOnboarding(
                store=self.market_data,
                bootstrap=bootstrap,
                profile_path=self.profile_path,
                provider=self.provider,
                as_of_session=target_session,
                history_start=scope.history_start,
                requested_listing_ids=scope.listing_ids,
                retry_of=scope.prior_onboarding_id,
            )
            self._runner = (scope, target_session, runner)
            return runner
        return cached[2]

    @staticmethod
    def _transition(scope: CandidateDataRecheck, target_session: date) -> str:
        return cast(
            str,
            canonical_hash(
                {
                    "kind": "candidate-data-requalification",
                    "scope": asdict(scope),
                    "target": target_session,
                }
            ),
        )

    def advance(
        self,
        scope: CandidateDataRecheck,
        *,
        target_session: date,
        observed_at: datetime,
        work_budget: int | None,
    ) -> tuple[CurrentUniverseOnboardingOutcome, UniverseManifest | None]:
        """Advance the bounded retry; a completed one names the manifest to prepare, if any."""
        remembered = self.completed_if_unchanged(scope, target_session=target_session)
        if remembered is not None:
            return remembered
        runner = self._onboarding(scope, target_session=target_session)
        outcome = runner.run(observed_at=observed_at, work_budget=work_budget)
        if outcome.status.value != "completed":
            return outcome, None
        record, bootstrap, parent = self._context(scope, target_session=target_session)
        if outcome.research_manifest is not None:
            admitted = {item.listing_id for item in outcome.research_manifest.listings}
            if not admitted <= set(scope.listing_ids):
                raise ValueError("workspace_maintenance.candidate_qualified_scope_mismatch")
            merged = build_quality_filtered_research_manifest(
                runner.acquisition_manifest,
                eligible_listing_ids=tuple(
                    admitted | {item.listing_id for item in parent.listings}
                ),
                qualification_obligations=parent.obligations_for_derivation(),
            )
            transition_id = self._transition(scope, target_session)
            # The Data-update request authorizes this bounded retry under its
            # existing source grant. Activation does not grant Universe ENTRY;
            # the subsequent Feature qualification still owns that prerequisite.
            transition = self.market_data.record_manifest_transition(
                transition_id=transition_id,
                market_profile_id=self.market_profile_id,
                prior_manifest_revision=parent.revision_sha256,
                membership_fingerprint=membership_fingerprint(bootstrap.source_manifest),
                additions=tuple(item.symbol for item in outcome.research_manifest.listings),
                removals=(),
                lifecycle="ONBOARDING_IN_PROGRESS",
                approved_at=observed_at,
                created_at=observed_at,
            )
            if (
                transition.market_profile_id != self.market_profile_id
                or transition.prior_manifest_revision != parent.revision_sha256
                or transition.membership_fingerprint
                != membership_fingerprint(bootstrap.source_manifest)
                or set(transition.additions)
                != {item.symbol for item in outcome.research_manifest.listings}
                or transition.removals
            ):
                raise ValueError("workspace_maintenance.candidate_recheck_activation_conflict")
            if transition.lifecycle == "ONBOARDING_IN_PROGRESS":
                assert record.active_candidate_manifest_document is not None
                assert record.last_checked_at is not None
                self.market_data.activate_workspace_manifest(
                    transition_id=transition_id,
                    manifest=merged,
                    membership_fingerprint=membership_fingerprint(bootstrap.source_manifest),
                    candidate_manifest_document=record.active_candidate_manifest_document,
                    checked_at=record.last_checked_at,
                    changed_at=observed_at,
                    within_activation=ResearchFoundationStateRepository.require_rebuild,
                )
            elif (
                transition.lifecycle != "ACTIVATED"
                or transition.next_manifest_revision != merged.revision_sha256
            ):
                raise ValueError("workspace_maintenance.candidate_recheck_activation_conflict")
        manifest = self.completed(scope, target_session=target_session)
        self._completed = (scope, target_session, outcome, manifest, self._readiness_guard())
        return outcome, manifest

    def activated_transition(
        self, scope: CandidateDataRecheck, *, target_session: date
    ) -> ManifestTransitionRecord | None:
        """The transition this recheck activated, when it admitted candidates.

        A Data-update request authorizes its raw retry to merge the candidates
        it admits into the source parent under one recorded transition, whose
        identity is this scope and session. That transition is the only
        authority under which the parent may have moved for the request; a
        Feature/Sector scope planned against the earlier parent is resolved
        across it, never across an unrelated move of the source.
        """
        try:
            transition = self.market_data.manifest_transition(
                self._transition(scope, target_session)
            )
        except ValueError as exc:
            if str(exc) != "manifest transition does not exist":
                raise
            return None
        if transition.lifecycle != "ACTIVATED" or transition.next_manifest_revision is None:
            return None
        return transition

    def completed_if_unchanged(
        self, scope: CandidateDataRecheck, *, target_session: date
    ) -> tuple[CurrentUniverseOnboardingOutcome, UniverseManifest | None] | None:
        """The completed recheck's answer while its validated readiness record stands."""
        remembered = self._completed
        if remembered is None or remembered[:2] != (scope, target_session):
            return None
        if self._readiness_guard() != remembered[4]:
            self._completed = None
            return None
        return remembered[2], remembered[3]

    def _readiness_guard(self) -> tuple[object, ...]:
        record = self.readiness_record()
        return (
            record.status,
            record.active_manifest_id,
            record.active_manifest_revision,
            record.active_membership_fingerprint,
            record.pending_membership_fingerprint,
            record.last_checked_at,
            record.last_changed_at,
            record.updated_at,
        )

    def completed(
        self, scope: CandidateDataRecheck, *, target_session: date
    ) -> UniverseManifest | None:
        """The manifest a completed recheck prepares, or None when no candidate qualified.

        A retry that admitted nothing leaves nothing to prepare: the day keeps
        the current research membership as its working manifest. Binding the
        source parent instead (every candidate the source names, the
        quarantined ones included) restarted the whole day's maintenance under
        a new revision -- every listing fetched again, every receipt
        re-validated on every cycle, the quarantined listings re-evaluated and
        their standing decisions asked for again -- for a membership that did
        not change.
        """
        remembered = self.completed_if_unchanged(scope, target_session=target_session)
        if remembered is not None:
            return remembered[1]
        runner = self._onboarding(scope, target_session=target_session)
        run = self.market_data.current_universe_onboarding_run(runner.onboarding_id)
        if run.lifecycle != "COMPLETED":
            raise ValueError("workspace_maintenance.candidate_data_recheck_incomplete")
        if run.research_manifest_revision is None:
            return None
        transition = self.market_data.manifest_transition(self._transition(scope, target_session))
        if transition.lifecycle != "ACTIVATED" or transition.next_manifest_revision is None:
            raise ValueError("workspace_maintenance.candidate_data_recheck_incomplete")
        return self.market_data.load_universe_manifest_revision(transition.next_manifest_revision)
