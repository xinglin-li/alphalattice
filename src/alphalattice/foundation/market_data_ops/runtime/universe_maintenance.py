"""Resumable normal maintenance for one completed full-US research universe.

Initial ten-year qualification and routine refresh deliberately have different
owners. This module never rebuilds a candidate universe or replays historical
quality gates: it advances a frozen quality-filtered manifest with a bounded
45-day raw overlap and action-admission check per listing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol

from alphalattice.foundation.market_data_ops.publication.projection import action_set_hash
from alphalattice.foundation.market_data_ops.runtime.diagnostics import (
    RestatementObservationReceipt,
    audit_bounded_restatements,
)
from alphalattice.foundation.market_data_ops.runtime.refresh import (
    MAXIMUM_PROVIDER_WINDOW_DAYS,
    RefreshWindow,
    normal_refresh_plan,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    FailureEvidence,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.sources.price_integrity import (
    PriceActionIntegritySentinelReport,
    TradingSessionAuthority,
    evaluate_price_action_integrity_sentinels,
)
from alphalattice.foundation.market_data_ops.sources.providers import (
    HistoricalHydrationProvider,
    HydrationEvidence,
    MarketDataProvider,
    ProviderFetchError,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import (
    CorruptedPayload,
    SanitizedBatch,
    sanitize_payload,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    ActionAuditScopeInsufficient,
    CurrentUniverseMaintenanceListing,
    MarketDataRepository,
)


class CurrentUniverseMaintenanceStatus(StrEnum):
    """Durable outcome of a bounded maintenance run."""

    RUNNING = "running"
    DEFERRED = "deferred"
    COMPLETED = "completed"


class MutationGate(Protocol):
    """Serialize store mutations while allowing bounded retained work."""

    def run(self, operation, /, *args, **kwargs) -> Any:
        """Apply one store mutation under the gate."""
        ...

    def hold(self) -> AbstractContextManager[None]:
        """Retain the mutation gate across a bounded unit."""
        ...


@dataclass(frozen=True)
class CurrentUniverseMaintenanceOutcome:
    """Durable progress and listing changes for one maintenance operation."""

    maintenance_id: str
    status: CurrentUniverseMaintenanceStatus
    listings: int
    updated: int
    failed: int
    failure_code: str | None = None
    listing_changes: tuple[dict[str, object], ...] = ()


CURRENT_UNIVERSE_MAINTENANCE_POLICY_ID = (
    "current-universe-maintenance:raw-action-provider-adjusted:typed-audit-scope"
)
NON_RETRYABLE_MAINTENANCE_FAILURE_CODES = (
    "data.full_history_audit_approval_required",
    "data.incremental_gap_approval_required",
    "data.listing_attempt_budget_exhausted_after_interruption",
)


def current_universe_maintenance_id(
    manifest: UniverseManifest,
    *,
    as_of_session: date,
    authorized_full_history_listing_ids: tuple[str, ...] = (),
) -> str:
    """Derive a stable ID from manifest, as-of session, policy, and authorization."""
    payload = "|".join(
        (
            manifest.revision_sha256,
            as_of_session.isoformat(),
            CURRENT_UNIVERSE_MAINTENANCE_POLICY_ID,
            ",".join(sorted(set(authorized_full_history_listing_ids))),
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()


_Chunk = list[CurrentUniverseMaintenanceListing]
_Job = tuple[CurrentUniverseMaintenanceListing, ManifestListing, tuple[object, ...], date]


@dataclass
class CurrentUniverseMaintenance:
    """Advance one frozen full-universe manifest without publishing snapshots.

    Every listing is an independently durable unit. A crash after a raw upsert
    can repeat a fetch, but the upsert and action revisions remain idempotent;
    a terminal ``UPDATED`` or ``FAILED`` unit is never fetched again for the
    same manifest and as-of session.
    """

    store: MarketDataRepository
    manifest: UniverseManifest
    provider: MarketDataProvider
    as_of_session: date
    retry_budget: int = 2
    full_audit_listing_ids: frozenset[str] = frozenset()
    max_workers: int = 4
    chunk_size: int = 25
    maximum_listing_attempts: int = 2
    full_history_escalation_listing_ids: frozenset[str] = frozenset()
    full_history_required_listing_ids: frozenset[str] = frozenset()
    authorization_identity_listing_ids: frozenset[str] = frozenset()
    retry_grants: Mapping[str, tuple[str, datetime]] = field(default_factory=dict)
    """Elapsed waits granting failed listings one attempt past the attempt budget.

    Listing -> (execution receipt of the confirmed ``wait_for_provider_recovery``
    decision, the wait instant that elapsed). The store records a grant on
    the listing's row when it reopens the unit, holds it across batches and
    restarts, and the attempt consumes it; the same wait never grants
    twice. Not part of the operation's identity.
    """

    trading_session_authority: TradingSessionAuthority | None = None
    """The qualified exchange-session axis, resolved and injected by the Host.

    Deliberately injected rather than materialized here. Resolving a calendar is
    a qualification act, and a refresh runner that built its own axis would be
    quarantining listings against a calendar nobody qualified. Absent, session
    membership is simply not judged; the dividend invariants still run.
    """

    mutation_gate: MutationGate | None = None
    progress_sink: Callable[[object], object] | None = None

    def __post_init__(self) -> None:
        """Validate the frozen manifest, provider, and bounded work policy."""
        if ":research-whitelist:" not in self.manifest.manifest_id:
            raise ValueError("routine maintenance requires a quality-filtered research manifest")
        if self.provider.name != self.manifest.profile.provider:
            raise ValueError("maintenance provider is not bound to research manifest")
        if self.retry_budget < 1 or self.retry_budget > 3:
            raise ValueError("maintenance retry_budget must be between 1 and 3")
        if self.max_workers not in {1, 2, 4}:
            raise ValueError("maintenance workers must follow the qualified 4 -> 2 -> 1 policy")
        if self.chunk_size != 25:
            raise ValueError("maintenance network chunk size is fixed at 25")
        if self.maximum_listing_attempts != 2:
            raise ValueError("maintenance listing attempts are fixed at two")
        manifest_listing_ids = {item.listing_id for item in self.manifest.listings}
        if not self.full_audit_listing_ids <= manifest_listing_ids:
            raise ValueError("full-audit authorization is outside the research manifest")
        if not self.full_history_escalation_listing_ids <= manifest_listing_ids:
            raise ValueError("full-history escalation authorization is outside the manifest")
        if not self.full_history_required_listing_ids <= manifest_listing_ids:
            raise ValueError("full-history requirement is outside the research manifest")
        if not self.authorization_identity_listing_ids <= manifest_listing_ids:
            raise ValueError("full-history authorization identity is outside the manifest")
        if self.full_history_required_listing_ids & self.full_audit_listing_ids:
            raise ValueError("full-history requirement cannot also be authorized")
        if not set(self.retry_grants) <= manifest_listing_ids:
            raise ValueError("retry grant is outside the research manifest")
        authorization_identity = self.authorization_identity_listing_ids or (
            self.full_audit_listing_ids | self.full_history_escalation_listing_ids
        )
        self.maintenance_id = current_universe_maintenance_id(
            self.manifest,
            as_of_session=self.as_of_session,
            authorized_full_history_listing_ids=tuple(sorted(authorization_identity)),
        )
        self._admitted = False

    @contextmanager
    def _retained(self) -> Iterator[None]:
        """One database instance for a bounded unit of this runner's work.

        Holds the single-writer gate for the whole unit and keeps the store's
        database open across the unit's reads and short transactions, so each
        of them costs a cursor rather than an instance open, a metadata read
        and a checkpoint. Every transaction inside still begins and ends on its
        own; the unit is never one transaction. A unit ends before a Provider
        fetch begins and starts again after it, so no connection or gate is
        held across a network wait.
        """
        gate = self.mutation_gate.hold() if self.mutation_gate is not None else nullcontext()
        with gate, self.store.database.retain(read_only=False):
            yield

    def admit(self, *, observed_at: datetime) -> None:
        """Admit the operation and any authorized prefix into durable store state."""
        # Admission is idempotent durable state that a bounded ``run`` cannot
        # change; a runner kept across runs need not repeat the schema bootstrap
        # and the staged listing upsert (about 0.24 s and a checkpoint per run
        # on a 472-name manifest). The marker is the last statement, so an
        # exception or an incomplete prefix seed leaves the next run repeating
        # the whole path; a fresh runner always re-admits from the store.
        if self._admitted:
            return
        self._mutate(
            self.store.admit_current_universe_maintenance,
            self.manifest,
            maintenance_id=self.maintenance_id,
            as_of_session=self.as_of_session,
            observed_at=observed_at,
        )
        authorization_ids = tuple(
            sorted(self.full_audit_listing_ids | self.full_history_escalation_listing_ids)
        )
        if authorization_ids:
            base_maintenance_id = current_universe_maintenance_id(
                self.manifest,
                as_of_session=self.as_of_session,
            )
            self._mutate(
                self.store.seed_current_universe_maintenance_verified_prefix,
                source_maintenance_id=base_maintenance_id,
                target_maintenance_id=self.maintenance_id,
                excluded_listing_ids=authorization_ids,
                observed_at=observed_at,
            )
        self._admitted = True

    def run(
        self,
        *,
        observed_at: datetime | None = None,
        work_budget: int | None = None,
        before_fetch: Callable[[], object] | None = None,
        after_fetch: Callable[[], object] | None = None,
    ) -> CurrentUniverseMaintenanceOutcome:
        """Advance at most ``work_budget`` complete listing units, then return safely.

        The Provider fetch is this run's network edge. ``before_fetch`` is
        called right before the runner reaches it, so a caller can release
        whatever it retained for its own preflight; ``after_fetch`` is called
        once the run's last fetch is back, so the caller can retain again for
        the rest of the run and its own epilogue. Neither is called by a run
        that fetches nothing.
        """
        now = observed_at or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("maintenance observed_at must be timezone-aware")
        if self.as_of_session > now.date():
            raise ValueError("maintenance as_of session cannot be in the future")
        if work_budget is not None and work_budget < 1:
            raise ValueError("maintenance work_budget must be positive")
        with self._retained():
            prepared = self._open_run(now, work_budget=work_budget)
        if not isinstance(prepared, tuple):
            return prepared
        chunks, first_jobs = prepared
        for index, chunk in enumerate(chunks):
            if index == 0:
                jobs = first_jobs
            else:
                with self._retained():
                    jobs = self._prepare_chunk(chunk, observed_at=now)
            if before_fetch is not None:
                before_fetch()
            systemic_failure: str | None = None
            last_chunk = index == len(chunks) - 1
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(jobs) or 1)) as pool:
                futures = {
                    pool.submit(
                        self._fetch_with_retry,
                        listing,
                        audit_start,
                        listing.listing_id not in self.full_audit_listing_ids,
                    ): (
                        item,
                        listing,
                        existing,
                        audit_start,
                    )
                    for item, listing, existing, audit_start in jobs
                }
                # One retained instance serves every completed fetch that is
                # ready to apply and, for the last chunk, the run's close, so a
                # one-listing run pays one open and one checkpoint after its
                # fetch. It is taken on a completion, never before one, and
                # released again whenever the next completion is still in
                # flight: nothing is held while the Provider is awaited.
                pending = set(futures)
                with ExitStack() as retained:
                    held = False
                    for future in as_completed(futures):
                        pending.discard(future)
                        if not held:
                            retained.enter_context(self._retained())
                            held = True
                        item, listing, existing, audit_start = futures[future]
                        try:
                            hydration = future.result()
                        except ProviderFetchError as exc:
                            if exc.code in {
                                "data.rate_limited",
                                "data.provider_session_unstable",
                            }:
                                systemic_failure = systemic_failure or exc.code
                                self._record_deferred_failure(item, code=exc.code, observed_at=now)
                            else:
                                self._record_failure(item, code=exc.code, observed_at=now)
                            self._publish_progress(
                                self._outcome(CurrentUniverseMaintenanceStatus.RUNNING),
                                current_item=listing.symbol,
                            )
                        else:
                            result = self._apply_hydration(
                                item,
                                listing=listing,
                                existing=existing,
                                audit_start=audit_start,
                                hydration=hydration,
                                observed_at=now,
                            )
                            if result.startswith("DEFERRED:"):
                                systemic_failure = systemic_failure or result.partition(":")[2]
                            self._publish_progress(
                                self._outcome(CurrentUniverseMaintenanceStatus.RUNNING),
                                current_item=listing.symbol,
                            )
                        if pending and not any(f.done() for f in pending):
                            retained.close()
                            held = False
                    if after_fetch is not None and (last_chunk or systemic_failure is not None):
                        after_fetch()
                    if not held:
                        retained.enter_context(self._retained())
                    if systemic_failure is not None:
                        self._mutate(
                            self.store.set_current_universe_maintenance_lifecycle,
                            self.maintenance_id,
                            lifecycle="DEFERRED",
                            observed_at=now,
                        )
                        outcome = self._outcome(
                            CurrentUniverseMaintenanceStatus.DEFERRED,
                            failure_code=systemic_failure,
                        )
                        self._publish_progress(
                            outcome,
                            status="FAILED",
                            failure_code=systemic_failure,
                        )
                        return outcome
                    self._publish_progress(self._outcome(CurrentUniverseMaintenanceStatus.RUNNING))
                    if last_chunk:
                        return self._close_run(now)
        with self._retained():
            return self._close_run(now)

    def _open_run(
        self, now: datetime, *, work_budget: int | None
    ) -> CurrentUniverseMaintenanceOutcome | tuple[list[_Chunk], list[_Job]]:
        """Admit, reconcile and stage the run; prepare the first chunk before any fetch."""
        self.admit(observed_at=now)
        prior = self.store.current_universe_maintenance_run(self.maintenance_id)
        required_listing_ids = set(self.full_history_required_listing_ids) - (
            set(self.full_audit_listing_ids) | set(self.full_history_escalation_listing_ids)
        )
        if prior.lifecycle == "COMPLETED":
            reopened = self._mutate(
                self.store.requeue_failed_current_universe_maintenance_listings,
                maintenance_id=self.maintenance_id,
                maximum_attempts=self.maximum_listing_attempts,
                excluded_failure_codes=NON_RETRYABLE_MAINTENANCE_FAILURE_CODES,
                retry_grants=self.retry_grants,
                observed_at=now,
            )
            if not reopened:
                outcome = self._outcome(CurrentUniverseMaintenanceStatus.COMPLETED)
                self._publish_progress(outcome, status="REUSED_EXACT")
                return outcome
        # A due or evidence-triggered full-history audit is not authorization.
        # Keep the listing as a durable, zero-attempt requirement so an
        # ordinary maintenance retry cannot spend a historical fetch budget.
        for listing_id in sorted(required_listing_ids):
            self._mutate(
                self.store.update_current_universe_maintenance_listing,
                maintenance_id=self.maintenance_id,
                listing_id=listing_id,
                state="FAILED",
                failure_code="data.full_history_audit_approval_required",
                observed_at=now,
            )
        listings = self.store.current_universe_maintenance_listings(self.maintenance_id)
        # A pending unit past the budget is exhausted unless its row holds a
        # governed retry grant not yet spent: enqueuing did not spend it, and
        # neither did a batch boundary or a restart in between.
        exhausted = tuple(
            item
            for item in listings
            if item.state == "PENDING"
            and item.attempt_count >= self.maximum_listing_attempts
            and not (item.retry_grant is not None and item.retry_grant.held)
        )
        for item in exhausted:
            self._mutate(
                self.store.update_current_universe_maintenance_listing,
                maintenance_id=self.maintenance_id,
                listing_id=item.listing_id,
                state="FAILED",
                failure_code=("data.listing_attempt_budget_exhausted_after_interruption"),
                observed_at=now,
            )
        self._mutate(
            self.store.set_current_universe_maintenance_lifecycle,
            self.maintenance_id,
            lifecycle="RUNNING",
            observed_at=now,
        )
        if exhausted:
            listings = self.store.current_universe_maintenance_listings(self.maintenance_id)
        pending = [item for item in listings if item.state == "PENDING"]
        self._publish_progress(
            self._outcome(CurrentUniverseMaintenanceStatus.RUNNING, listings=listings)
        )
        if work_budget is not None:
            pending = pending[:work_budget]
        chunks = [
            pending[offset : offset + self.chunk_size]
            for offset in range(0, len(pending), self.chunk_size)
        ]
        first_jobs = self._prepare_chunk(chunks[0], observed_at=now) if chunks else []
        return chunks, first_jobs

    def _prepare_chunk(self, chunk: _Chunk, *, observed_at: datetime) -> list[_Job]:
        jobs: list[_Job] = []
        for item in chunk:
            prepared = self._prepare_listing(item, observed_at=observed_at)
            if prepared is not None:
                listing, existing, audit_start = prepared
                jobs.append((item, listing, existing, audit_start))
        return jobs

    def _close_run(self, now: datetime) -> CurrentUniverseMaintenanceOutcome:
        current = self.store.current_universe_maintenance_listings(self.maintenance_id)
        if any(item.state == "PENDING" for item in current):
            outcome = self._outcome(CurrentUniverseMaintenanceStatus.RUNNING, listings=current)
            self._publish_progress(outcome)
            return outcome
        self._mutate(
            self.store.set_current_universe_maintenance_lifecycle,
            self.maintenance_id,
            lifecycle="COMPLETED",
            observed_at=now,
        )
        outcome = self._outcome(CurrentUniverseMaintenanceStatus.COMPLETED)
        if outcome.failed:
            self._publish_progress(
                outcome,
                status="FAILED",
                failure_code="data.listing_updates_incomplete",
            )
        else:
            self._publish_progress(outcome, status="SUCCEEDED")
        return outcome

    def _refresh_listing(
        self, item: CurrentUniverseMaintenanceListing, *, observed_at: datetime
    ) -> str:
        with self._retained():
            prepared = self._prepare_listing(item, observed_at=observed_at)
        if prepared is None:
            return "FAILED"
        listing, existing, audit_start = prepared
        try:
            hydration = self._fetch_with_retry(
                listing,
                audit_start,
                listing.listing_id not in self.full_audit_listing_ids,
            )
        except ProviderFetchError as exc:
            with self._retained():
                if exc.code in {"data.rate_limited", "data.provider_session_unstable"}:
                    self._record_deferred_failure(item, code=exc.code, observed_at=observed_at)
                    return f"DEFERRED:{exc.code}"
                self._record_failure(item, code=exc.code, observed_at=observed_at)
                return "FAILED"
        with self._retained():
            return self._apply_hydration(
                item,
                listing=listing,
                existing=existing,
                audit_start=audit_start,
                hydration=hydration,
                observed_at=observed_at,
            )

    def _prepare_listing(
        self, item: CurrentUniverseMaintenanceListing, *, observed_at: datetime
    ) -> tuple[ManifestListing, tuple[object, ...], date] | None:
        self._mutate(
            self.store.begin_current_universe_maintenance_listing_attempt,
            maintenance_id=self.maintenance_id,
            listing_id=item.listing_id,
            observed_at=observed_at,
        )
        listing = self._listing(item.listing_id)
        existing = self.store.raw_bars(item.listing_id, through=self.as_of_session)
        if not existing:
            self._record_failure(
                item,
                code="data.maintenance_missing_raw_history",
                observed_at=observed_at,
            )
            return None
        refresh_plan = normal_refresh_plan(existing[-1].session_date, self.as_of_session)
        if refresh_plan.requires_explicit_approval and (
            listing.listing_id not in self.full_audit_listing_ids
        ):
            self._record_failure(
                item,
                code="data.incremental_gap_approval_required",
                observed_at=observed_at,
            )
            return None
        refresh_start = refresh_plan.fetch_start
        if refresh_start is None:
            raise AssertionError("a listing with raw history must have a refresh start")
        audit_start = (
            existing[0].session_date
            if listing.listing_id in self.full_audit_listing_ids
            else refresh_start
        )
        return listing, tuple(existing), audit_start

    def _fetch_with_retry(
        self,
        listing: ManifestListing,
        audit_start: date,
        allow_chunking: bool = True,
    ) -> HydrationEvidence:
        windows = (RefreshWindow(audit_start, self.as_of_session),)
        if allow_chunking:
            planned: list[RefreshWindow] = []
            cursor = audit_start
            while cursor <= self.as_of_session:
                window_end = min(
                    self.as_of_session,
                    cursor + timedelta(days=MAXIMUM_PROVIDER_WINDOW_DAYS - 1),
                )
                planned.append(RefreshWindow(cursor, window_end))
                cursor = window_end + timedelta(days=1)
            windows = tuple(planned)
        parts = tuple(self._fetch_window_with_retry(listing, window) for window in windows)
        if len(parts) == 1:
            return parts[0]
        return HydrationEvidence(
            daily_rows=tuple(row for part in parts for row in part.daily_rows),
            actions=tuple(action for part in parts for action in part.actions),
            adjusted_closes=tuple(point for part in parts for point in part.adjusted_closes),
            repaired_sessions=tuple(
                sorted({session for part in parts for session in part.repaired_sessions})
            ),
            provider_policy_hash=self._combined_provider_policy_hash(parts),
        )

    @staticmethod
    def _combined_provider_policy_hash(parts: tuple[HydrationEvidence, ...]) -> str | None:
        identities = {part.provider_policy_hash for part in parts if part.provider_policy_hash}
        if len(identities) > 1:
            raise ProviderFetchError(
                "data.provider_policy_mismatch",
                "chunked hydration used inconsistent provider price policies",
                retryable=False,
            )
        return next(iter(identities), None)

    def _fetch_window_with_retry(
        self,
        listing: ManifestListing,
        window: RefreshWindow,
    ) -> HydrationEvidence:
        for attempt in range(1, self.retry_budget + 1):
            try:
                return self._fetch_hydration(
                    listing,
                    start=window.start,
                    end=window.end,
                )
            except ProviderFetchError as exc:
                if exc.retryable and attempt < self.retry_budget:
                    continue
                raise
        raise AssertionError("bounded maintenance fetch did not return")

    def _apply_hydration(
        self,
        item: CurrentUniverseMaintenanceListing,
        *,
        listing: ManifestListing,
        existing: tuple[object, ...],
        audit_start: date,
        hydration: HydrationEvidence,
        observed_at: datetime,
    ) -> str:
        try:
            batch = sanitize_payload(
                self.manifest,
                self.provider.name,
                {listing.symbol: hydration.daily_rows},
                (listing.symbol,),
            )
            raw_through = max(bar.session_date for bar in batch.bars)
        except (CorruptedPayload, KeyError, ValueError):
            self._record_failure(
                item,
                code="data.sanitizer.corrupted_payload",
                observed_at=observed_at,
            )
            return "FAILED"
        # Bounded deterministic sentinels over the sanitized batch, before any
        # write. ``trading_sessions`` is None because this runner holds no
        # exchange-calendar authority; membership anchoring stays with the
        # execution-axis owners, and a breach of the checks this runner *can*
        # evaluate quarantines the listing instead of qualifying it.
        sentinel_report = self._price_action_sentinel(batch)
        if sentinel_report.disposition != "ANCHORED":
            self._record_failure(
                item,
                code="data.price_action_sentinel_quarantine",
                observed_at=observed_at,
            )
            return "FAILED"
        before_bars = {bar.session_date: bar for bar in existing}
        before_actions = self.store.actions(item.listing_id)
        before_adjusted = self.store.provider_adjusted_closes(
            item.listing_id,
            through=self.as_of_session,
        )
        before_adjusted_through = (
            max(point.session_date for point in before_adjusted) if before_adjusted else None
        )
        # Pure pre-admission observation: how the candidate batch disagrees with
        # already-qualified history, by content identity, before qualification
        # decides anything. It never writes; ``apply_validated_batch`` below
        # remains the only explicit qualification act.
        candidate_sessions = tuple(bar.session_date for bar in batch.bars)
        qualified_bars = tuple(bar for bar in existing if isinstance(bar, RawDailyBar))
        restatement_observation = audit_bounded_restatements(
            candidate_bars=batch.bars,
            qualified_bars=qualified_bars,
            provider=self.provider.name,
            listing_scope=(item.listing_id,),
            session_scope_start=min(candidate_sessions),
            session_scope_end=max(candidate_sessions),
        )
        # An unauthorized restatement must not be qualified at all. The
        # post-audit escalation check below reached the same verdict, but only
        # after the rows had already been written: the current row changed, no
        # change document was produced, and no dependency impact was published,
        # which is the one sequence a correction may never take. Now that the
        # observation knows the correction before any write, the authorization
        # decision moves ahead of it. Escalation authority for this listing is
        # what makes the rewrite explicit, so the failure code stays the one
        # every downstream remediation affordance already branches on; only its
        # position moved. Action, adjusted-series and audit-scope corrections
        # are not knowable here and keep their existing post-audit check.
        if (
            restatement_observation.corrections
            and audit_start != qualified_bars[0].session_date
            and listing.listing_id not in self.full_history_escalation_listing_ids
        ):
            self._record_failure(
                item,
                code="data.full_history_audit_approval_required",
                observed_at=observed_at,
            )
            return "FAILED"
        write_counts = self._mutate(
            self.store.apply_validated_batch,
            self.manifest,
            batch,
            ingestion_id=f"{self.maintenance_id}:refresh:{listing.listing_id}",
            observed_at=observed_at,
        )
        if raw_through < self.as_of_session:
            self._record_failure(
                item,
                code="data.maintenance_stale_payload",
                observed_at=observed_at,
            )
            return "FAILED"
        return self._audit_and_finish(
            item,
            listing=listing,
            raw_through=raw_through,
            observed_at=observed_at,
            hydration=hydration,
            audit_start=audit_start,
            before_bars=before_bars,
            before_actions=before_actions,
            before_adjusted_through=before_adjusted_through,
            write_counts=write_counts,
            sentinel_report=sentinel_report,
            restatement_observation=restatement_observation,
        )

    def _price_action_sentinel(self, batch: SanitizedBatch) -> PriceActionIntegritySentinelReport:
        """Run the sentinels against the authority the Host qualified.

        Session membership is judged only when the Host injected a resolved
        axis, and only inside the range that axis covers. ``known_actions``
        stays empty: golden event truth is external evidence a person supplies
        with its own provenance, not a constant a refresh may assert, so
        external anchoring remains pending rather than silently claimed.
        """
        return evaluate_price_action_integrity_sentinels(
            bars=batch.bars,
            actions=batch.actions,
            provider=self.provider.name,
            as_of=self.as_of_session,
            trading_sessions=self.trading_session_authority,
        )

    def _audit_and_finish(
        self,
        item: CurrentUniverseMaintenanceListing,
        *,
        listing: ManifestListing,
        raw_through: date,
        observed_at: datetime,
        hydration: HydrationEvidence,
        audit_start: date,
        before_bars: dict[date, object],
        before_actions: tuple[object, ...],
        before_adjusted_through: date | None,
        write_counts: dict[str, int],
        sentinel_report: PriceActionIntegritySentinelReport,
        restatement_observation: RestatementObservationReceipt,
    ) -> str:
        bars = self.store.raw_bars(item.listing_id, through=self.as_of_session)
        force_full_audit = False
        adjusted_return_change_sessions: set[date] = set()
        try:
            receipt, audit_counts = self._mutate(
                self.store.complete_action_audit,
                self.manifest,
                listing_id=item.listing_id,
                provider=self.provider.name,
                observed_actions=hydration.actions,
                observed_adjusted_closes=hydration.adjusted_closes,
                history_start=audit_start,
                history_end=self.as_of_session,
                requested_as_of=self.as_of_session,
                observed_at=observed_at,
            )
            adjusted_revision = self.store.provider_adjusted_revision(receipt.receipt_hash)
            if adjusted_revision is not None:
                adjusted_return_change_sessions.update(adjusted_revision.changed_return_sessions)
        except ProviderFetchError as exc:
            if exc.code in {"data.rate_limited", "data.provider_session_unstable"}:
                self._mutate(
                    self.store.record_failures,
                    (
                        FailureEvidence(
                            item.listing_id,
                            self.manifest.profile.market_profile_id,
                            exc.code,
                            bars[0].session_date,
                            self.as_of_session,
                            observed_at,
                        ),
                    ),
                )
                return f"DEFERRED:{exc.code}"
            self._record_failure(item, code=exc.code, observed_at=observed_at)
            return "FAILED"
        except ActionAuditScopeInsufficient:
            # A rolling payload can be internally valid yet insufficient to
            # reconcile a session-set or same-day action ambiguity.  Escalate
            # exactly once to the already-required full-history hydration;
            # a full payload that remains ambiguous fails closed below.
            if audit_start == bars[0].session_date:
                self._record_failure(
                    item, code="data.action_audit_invalid", observed_at=observed_at
                )
                return "FAILED"
            force_full_audit = True
        except ValueError:
            # Programmer errors, unsupported actions, and invalid diagnostics
            # are not authority to spend a full-history Provider budget.
            self._record_failure(item, code="data.action_audit_invalid", observed_at=observed_at)
            return "FAILED"
        historical_adjusted_change = before_adjusted_through is not None and any(
            session <= before_adjusted_through for session in adjusted_return_change_sessions
        )
        needs_full = force_full_audit or (
            audit_start != bars[0].session_date
            and (
                write_counts["corrected"] > 0
                or write_counts["action_corrected"] > 0
                or audit_counts["corrected"] > 0
                or audit_counts["retracted"] > 0
                or historical_adjusted_change
            )
        )
        if needs_full and item.listing_id not in self.full_history_escalation_listing_ids:
            self._record_failure(
                item,
                code="data.full_history_audit_approval_required",
                observed_at=observed_at,
            )
            return "FAILED"
        if needs_full:
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    hydration = pool.submit(
                        self._fetch_with_retry, listing, bars[0].session_date, False
                    ).result()
                full_batch = sanitize_payload(
                    self.manifest,
                    self.provider.name,
                    {listing.symbol: hydration.daily_rows},
                    (listing.symbol,),
                )
                sentinel_report = self._price_action_sentinel(full_batch)
                if sentinel_report.disposition != "ANCHORED":
                    self._record_failure(
                        item,
                        code="data.price_action_sentinel_quarantine",
                        observed_at=observed_at,
                    )
                    return "FAILED"
                full_counts = self._mutate(
                    self.store.apply_validated_batch,
                    self.manifest,
                    full_batch,
                    ingestion_id=f"{self.maintenance_id}:full:{listing.listing_id}",
                    observed_at=observed_at,
                )
                write_counts = {
                    key: write_counts.get(key, 0) + full_counts.get(key, 0)
                    for key in set(write_counts) | set(full_counts)
                }
                receipt, audit_counts = self._mutate(
                    self.store.complete_action_audit,
                    self.manifest,
                    listing_id=item.listing_id,
                    provider=self.provider.name,
                    observed_actions=hydration.actions,
                    observed_adjusted_closes=hydration.adjusted_closes,
                    history_start=bars[0].session_date,
                    history_end=self.as_of_session,
                    requested_as_of=self.as_of_session,
                    observed_at=observed_at,
                )
                adjusted_revision = self.store.provider_adjusted_revision(receipt.receipt_hash)
                if adjusted_revision is not None:
                    adjusted_return_change_sessions.update(
                        adjusted_revision.changed_return_sessions
                    )
                audit_start = bars[0].session_date
                bars = self.store.raw_bars(item.listing_id, through=self.as_of_session)
            except ProviderFetchError as exc:
                if exc.code in {"data.rate_limited", "data.provider_session_unstable"}:
                    self._record_deferred_failure(item, code=exc.code, observed_at=observed_at)
                    return f"DEFERRED:{exc.code}"
                self._record_failure(
                    item, code="data.full_history_audit_failed", observed_at=observed_at
                )
                return "FAILED"
            except (CorruptedPayload, ValueError):
                self._record_failure(
                    item, code="data.full_history_audit_failed", observed_at=observed_at
                )
                return "FAILED"
        adjusted = self.store.provider_adjusted_closes(
            item.listing_id, through=self.as_of_session, start=bars[0].session_date
        )
        if {point.session_date for point in adjusted} != {bar.session_date for bar in bars}:
            self._record_failure(
                item,
                code="data.provider_adjusted_series_incomplete",
                observed_at=observed_at,
            )
            return "FAILED"
        self._mutate(
            self.store.update_current_universe_maintenance_listing,
            maintenance_id=self.maintenance_id,
            listing_id=item.listing_id,
            state="UPDATED",
            raw_through=raw_through,
            change_document=self._change_document(
                listing_id=item.listing_id,
                before_bars=before_bars,
                after_bars={bar.session_date: bar for bar in bars},
                before_actions=before_actions,
                after_actions=self.store.actions(item.listing_id),
                audit_start=audit_start,
                provider_receipt_hash=receipt.receipt_hash,
                action_set_hash_value=receipt.action_set_hash,
                raw_evidence_hash=receipt.raw_evidence_hash,
                mapping_revision=receipt.mapping_revision,
                adjusted_diagnostic_max_bps=receipt.max_adjusted_close_difference_bps,
                adjusted_return_change_sessions=tuple(sorted(adjusted_return_change_sessions)),
                provider_policy_hash=hydration.provider_policy_hash,
                provider_repaired_sessions=hydration.repaired_sessions,
                sentinel_report=sentinel_report,
                restatement_observation=restatement_observation,
                observed_at=observed_at,
            ),
            observed_at=observed_at,
        )
        return "UPDATED"

    def _fetch_hydration(
        self,
        listing: ManifestListing,
        *,
        start: date,
        end: date,
    ) -> HydrationEvidence:
        if isinstance(self.provider, HistoricalHydrationProvider):
            return self.provider.fetch_hydration(
                listing_id=listing.listing_id,
                provider_symbol=listing.provider_symbol,
                start=start,
                end=end,
            )
        payload = self.provider.fetch_daily((listing.provider_symbol,), start=start, end=end)
        try:
            rows = tuple(payload[listing.provider_symbol])
        except KeyError as exc:
            raise ProviderFetchError(
                "data.symbol_mismatch", "provider omitted the requested symbol", retryable=False
            ) from exc
        return HydrationEvidence(
            daily_rows=rows,
            actions=tuple(
                self.provider.fetch_action_history(
                    listing_id=listing.listing_id,
                    provider_symbol=listing.provider_symbol,
                    start=start,
                    end=end,
                )
            ),
            adjusted_closes=tuple(
                self.provider.fetch_adjusted_close_history(
                    listing_id=listing.listing_id,
                    provider_symbol=listing.provider_symbol,
                    start=start,
                    end=end,
                )
            ),
        )

    @staticmethod
    def _change_document(
        *,
        listing_id: str,
        before_bars: dict[date, object],
        after_bars: dict[date, object],
        before_actions: tuple[object, ...],
        after_actions: tuple[object, ...],
        audit_start: date,
        provider_receipt_hash: str,
        action_set_hash_value: str,
        raw_evidence_hash: str,
        mapping_revision: str,
        adjusted_diagnostic_max_bps: float | None,
        adjusted_return_change_sessions: tuple[date, ...],
        provider_policy_hash: str | None,
        provider_repaired_sessions: tuple[date, ...],
        sentinel_report: PriceActionIntegritySentinelReport,
        restatement_observation: RestatementObservationReceipt,
        observed_at: datetime,
    ) -> dict[str, object]:
        def bar_payload(value: object) -> tuple[object, ...]:
            return tuple(
                getattr(value, name) for name in ("open", "high", "low", "close", "volume")
            )

        new_sessions = sorted(set(after_bars) - set(before_bars))
        corrected = sorted(
            session
            for session in set(before_bars) & set(after_bars)
            if bar_payload(before_bars[session]) != bar_payload(after_bars[session])
        )
        before_action_map = {
            (item.effective_date, item.action_kind): asdict(item) for item in before_actions
        }
        after_action_map = {
            (item.effective_date, item.action_kind): asdict(item) for item in after_actions
        }
        action_dates = sorted(
            key[0]
            for key in set(before_action_map) | set(after_action_map)
            if before_action_map.get(key) != after_action_map.get(key)
        )
        return {
            "listing_id": listing_id,
            "new_session_start": new_sessions[0].isoformat() if new_sessions else None,
            "new_sessions": [item.isoformat() for item in new_sessions],
            "raw_correction_start": corrected[0].isoformat() if corrected else None,
            "raw_correction_sessions": [item.isoformat() for item in corrected],
            "action_correction_start": action_dates[0].isoformat() if action_dates else None,
            "adjusted_return_change_sessions": [
                item.isoformat() for item in adjusted_return_change_sessions
            ],
            "audit_scope": "FULL" if audit_start == min(after_bars) else "ROLLING",
            "history_start": min(after_bars).isoformat(),
            "history_end": max(after_bars).isoformat(),
            "window_start": audit_start.isoformat(),
            "provider_receipt_hash": provider_receipt_hash,
            "window_action_hash": action_set_hash(
                item for item in after_actions if item.effective_date >= audit_start
            ),
            "action_set_hash": action_set_hash_value,
            "raw_evidence_hash": raw_evidence_hash,
            "mapping_revision": mapping_revision,
            "adjusted_diagnostic_max_bps": adjusted_diagnostic_max_bps,
            "provider_policy_hash": provider_policy_hash,
            "provider_repaired_sessions": [item.isoformat() for item in provider_repaired_sessions],
            "price_action_sentinel": {
                "report_hash": sentinel_report.report_hash,
                "disposition": sentinel_report.disposition,
                "policy_hash": sentinel_report.policy_hash,
                "input_identity": sentinel_report.input_identity,
                "finding_codes": sorted({item.code for item in sentinel_report.findings}),
            },
            "restatement_observation": {
                "receipt_hash": restatement_observation.receipt_hash,
                "qualified_scope_identity": restatement_observation.qualified_scope_identity,
                "candidate_scope_identity": restatement_observation.candidate_scope_identity,
                "exact_reuse_count": restatement_observation.exact_reuse_count,
                "new_session_count": restatement_observation.new_session_count,
                "qualified_only_count": restatement_observation.qualified_only_count,
                "corrections": [
                    {
                        "listing_id": item.listing_id,
                        "session_date": item.session_date.isoformat(),
                        "prior_identity": item.prior_identity,
                        "candidate_identity": item.candidate_identity,
                        "classification": item.classification,
                    }
                    for item in restatement_observation.corrections
                ],
            },
            "observed_at": observed_at.astimezone(UTC).isoformat(),
        }

    def _listing(self, listing_id: str) -> ManifestListing:
        listing = next(
            (item for item in self.manifest.listings if item.listing_id == listing_id), None
        )
        if listing is None:
            raise ValueError("maintenance listing is absent from research manifest")
        return listing

    def _record_failure(
        self,
        item: CurrentUniverseMaintenanceListing,
        *,
        code: str,
        observed_at: datetime,
    ) -> None:
        self._mutate(
            self.store.record_failures,
            (
                FailureEvidence(
                    item.listing_id,
                    self.manifest.profile.market_profile_id,
                    code,
                    self.as_of_session,
                    self.as_of_session,
                    observed_at,
                ),
            ),
        )
        self._mutate(
            self.store.update_current_universe_maintenance_listing,
            maintenance_id=self.maintenance_id,
            listing_id=item.listing_id,
            state="FAILED",
            failure_code=code,
            observed_at=observed_at,
        )

    def _record_deferred_failure(
        self,
        item: CurrentUniverseMaintenanceListing,
        *,
        code: str,
        observed_at: datetime,
    ) -> None:
        """Record transport evidence without making the listing terminal."""
        self._mutate(
            self.store.record_failures,
            (
                FailureEvidence(
                    item.listing_id,
                    self.manifest.profile.market_profile_id,
                    code,
                    self.as_of_session,
                    self.as_of_session,
                    observed_at,
                ),
            ),
        )

    def _mutate(self, operation, /, *args, **kwargs):
        if self.mutation_gate is None:
            return operation(*args, **kwargs)
        return self.mutation_gate.run(operation, *args, **kwargs)

    def _outcome(
        self,
        status: CurrentUniverseMaintenanceStatus,
        *,
        failure_code: str | None = None,
        listings: tuple[CurrentUniverseMaintenanceListing, ...] | None = None,
    ) -> CurrentUniverseMaintenanceOutcome:
        if listings is None:
            listings = self.store.current_universe_maintenance_listings(self.maintenance_id)
        return CurrentUniverseMaintenanceOutcome(
            maintenance_id=self.maintenance_id,
            status=status,
            listings=len(listings),
            updated=sum(item.state == "UPDATED" for item in listings),
            failed=sum(item.state == "FAILED" for item in listings),
            failure_code=failure_code,
            listing_changes=tuple(
                item.change_document for item in listings if item.change_document is not None
            ),
        )

    def _publish_progress(
        self,
        outcome: CurrentUniverseMaintenanceOutcome,
        *,
        status: str = "RUNNING",
        failure_code: str | None = None,
        current_item: str | None = None,
    ) -> None:
        if self.progress_sink is None:
            return
        from alphalattice.control.observation_runtime.telemetry.progress import WorkProgressUpdate

        completed = outcome.updated + outcome.failed
        self.progress_sink(
            WorkProgressUpdate(
                operation_id=self.maintenance_id,
                stage_id="market_data_increment",
                status=status,
                completed_units=completed,
                total_units=outcome.listings,
                unit_name="listings",
                current_item=current_item,
                counters={
                    "updated": outcome.updated,
                    "failed": outcome.failed,
                    "pending": max(0, outcome.listings - completed),
                },
                failure_code=failure_code,
            )
        )
