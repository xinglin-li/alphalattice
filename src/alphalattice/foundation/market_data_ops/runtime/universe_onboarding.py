"""Resumable deterministic onboarding for the full current-US candidate universe.

This owner does not create a feature snapshot and does not call an agent. It
hydrates the durable data foundation first, derives a quality-qualified current
research manifest, and leaves user-selected research as a later task.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Literal

from alphalattice.foundation.market_data_ops.sources.contracts import FailureEvidence, RawDailyBar
from alphalattice.foundation.market_data_ops.sources.manifest import (
    UniverseManifest,
    build_current_index_acquisition_manifest,
    build_quality_filtered_research_manifest,
)
from alphalattice.foundation.market_data_ops.sources.providers import (
    HistoricalHydrationProvider,
    HydrationEvidence,
    MarketDataProvider,
    ProviderFetchError,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import (
    CorruptedPayload,
    sanitize_payload,
)
from alphalattice.foundation.market_data_ops.sources.universe import (
    CurrentUniverseBootstrap,
    candidate_manifest_document,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    ActionAuditReceipt,
    CurrentUniverseOnboardingListing,
    HydrationDeferred,
    MarketDataRepository,
)
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from alphalattice.kernel.data.quality import diagnose_daily_table
from alphalattice.kernel.data.table import coerce_daily_records


class CurrentUniverseOnboardingStatus(StrEnum):
    """Durable state of a current-universe onboarding operation."""

    RUNNING = "running"
    DEFERRED = "deferred"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class DataTargetSessionLag(ValueError):
    """The admitted baseline's minimum population has not reached its target."""

    failure_code = "data.target_session_not_covered"

    def __init__(
        self, *, target: date, reaching: int, total: int, minimum: int, latest: date
    ) -> None:
        """Record observed session coverage against the admitted baseline minimum."""
        detail = (
            f"At {target}, {reaching} of {total} listings have a market bar; "
            f"the baseline needs at least {minimum}. Latest common bar: {latest}."
        )
        self.cause = {
            "exception_type": type(self).__name__,
            "detail": detail,
            "step": "prepare_data",
        }
        super().__init__(self.failure_code)


@dataclass(frozen=True)
class CurrentUniverseOnboardingOutcome:
    """Progress counts, deferred retry, and admitted research manifest."""

    onboarding_id: str
    status: CurrentUniverseOnboardingStatus
    candidates: int
    raw_ready: int
    quality_eligible: int
    feature_ready: int
    failed: int
    research_manifest: UniverseManifest | None = None
    failure_code: str | None = None
    deferred_retry_id: str | None = None
    retry_after_at: datetime | None = None
    next_workers: int | None = None
    exclusion_counts: dict[str, int] = field(default_factory=dict)


def listing_outcome_category(
    state: str, reasons: Sequence[str] = (), failure_code: str | None = None
) -> str | None:
    """Classify the recorded acquisition, qualification or audit outcome, not its wording."""
    if state == "RAW_FAILED" or (
        state == "AUDIT_FAILED"
        and failure_code
        in {
            "data.provider_fetch_failed",
            "data.provider_timeout",
            "data.provider_unavailable",
            "data.provider_session_unstable",
            "data.rate_limited",
            "data.empty_payload",
            "data.adjusted_close_unavailable",
        }
    ):
        return "ACQUISITION_FAILURE"
    if state == "QUALITY_INELIGIBLE" and tuple(reasons) == ("insufficient_research_history",):
        return "HISTORY_INELIGIBLE"
    if state in {"QUALITY_INELIGIBLE", "AUDIT_FAILED"}:
        return "QUALITY_REJECTION"
    return None


@dataclass(frozen=True)
class ListingUnitObservation:
    """One recorded unit transition announced to an optional observer.

    The observer hears this after the durable listing row changes. The row is
    authoritative; this contains only what the runner knew at that instant.

    ``origin`` is stated only where the runner knows it: ACQUIRED -- the listing's bars
    were fetched from the Provider in this run and applied; RETAINED -- durable history
    was reused (``tail_acquired`` when a bounded tail was fetched to complete it);
    LOCAL -- an assessment or admission made from the store, which says nothing about a
    download. A failure carries its code and no origin.

    ``run_start_state`` is the listing's durable state when this run loaded it (PENDING,
    RAW_READY or QUALITY_ELIGIBLE): the state the announced transition departs from for
    the run's first announcement of that listing; a listing that moves more than once in
    one run departs its later transitions from what the observer already heard.
    """

    listing_id: str
    symbol: str
    state: str
    observed_at: datetime
    run_start_state: str
    origin: Literal["ACQUIRED", "RETAINED", "LOCAL"] | None = None
    raw_through: date | None = None
    failure_code: str | None = None
    reasons: tuple[str, ...] = ()
    tail_acquired: bool = False


ListingObserver = Callable[[ListingUnitObservation], None]


def _ten_calendar_year_anniversary(value: date) -> date:
    try:
        return value.replace(year=value.year - 10)
    except ValueError:
        return value.replace(year=value.year - 10, day=28)


def _onboarding_id(
    bootstrap: CurrentUniverseBootstrap,
    acquisition_manifest: UniverseManifest,
    *,
    history_start: date,
    as_of_session: date,
    requested_listing_ids: tuple[str, ...] = (),
    retry_of: str | None = None,
) -> str:
    payload = "|".join(
        (
            bootstrap.source_manifest.content_hash,
            acquisition_manifest.revision_sha256,
            history_start.isoformat(),
            as_of_session.isoformat(),
            "current-universe-onboarding:raw-quality-action-audit",
        )
    )
    if requested_listing_ids:
        payload += "|selected-candidates:" + ",".join(requested_listing_ids)
    if retry_of is not None:
        payload += "|retry-of:" + retry_of
    return sha256(payload.encode("utf-8")).hexdigest()


def _listing_calendar(bootstrap: CurrentUniverseBootstrap) -> dict[str, str]:
    """Use the original source convention until exchange identity is separately grounded."""
    return {
        candidate.symbol: (
            "XNAS"
            if any(item.index == "NASDAQ100" for item in candidate.source_memberships)
            else "XNYS"
        )
        for candidate in bootstrap.source_manifest.candidates
    }


@dataclass
class _RunnerHold:
    """One retained store instance between the runner's network edges.

    The maintenance runner's pattern: a unit's store reads and short
    transactions share one instance, taken for them and released before the
    Provider is awaited -- on the hydration path and on the audit's evidence
    fallback alike -- so nothing is held while the process waits on the
    outside world. Each transaction still begins and ends on its own.
    """

    store: MarketDataRepository
    _stack: ExitStack | None = field(default=None, init=False, repr=False)

    @property
    def held(self) -> bool:
        return self._stack is not None

    def take(self) -> None:
        if self._stack is not None:
            return
        stack = ExitStack()
        try:
            stack.enter_context(self.store.database.retain(read_only=False))
        except BaseException:
            stack.close()
            raise
        self._stack = stack

    def release(self) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            stack.close()

    @contextmanager
    def released(self) -> Iterator[None]:
        """Release the retained store across a network edge, then retake it."""
        held = self.held
        self.release()
        try:
            yield
        finally:
            if held:
                self.take()


@dataclass
class CurrentUniverseOnboarding:
    """A bounded-transport, crash-resumable full-universe maintenance task.

    One listing is one bounded unit of work. A provider failure excludes that
    listing from this manifest build but never aborts unrelated listings. A
    rate limit defers the task instead of sleeping or retrying indefinitely.
    """

    store: MarketDataRepository
    bootstrap: CurrentUniverseBootstrap
    profile_path: Path
    provider: MarketDataProvider
    as_of_session: date
    history_start: date | None = None
    retry_budget: int = 2
    retained_listing_ids: tuple[str, ...] = ()
    hydration_chunk_size: int = 25
    hydration_workers: int = 2
    timing_sink: Callable[[str, float, Mapping[str, object]], None] | None = None
    # Optional: told each unit transition after the store recorded it. Observation only;
    # it must not raise, and the runner neither waits for nor depends on what it does.
    listing_observer: ListingObserver | None = None
    requested_listing_ids: tuple[str, ...] = ()
    retry_of: str | None = None

    def __post_init__(self) -> None:
        """Validate bounded work settings and derive the durable onboarding ID."""
        self.profile_path = Path(self.profile_path)
        if self.retry_budget < 1 or self.retry_budget > 3:
            raise ValueError("onboarding retry_budget must be between 1 and 3")
        if self.hydration_chunk_size < 1 or self.hydration_chunk_size > 50:
            raise ValueError("hydration_chunk_size must be between 1 and 50")
        if self.hydration_workers < 1 or self.hydration_workers > 8:
            raise ValueError("hydration_workers must be between 1 and 8")
        if self.history_start is None:
            self.history_start = _ten_calendar_year_anniversary(self.as_of_session)
        if self.history_start > self.as_of_session:
            raise ValueError("onboarding history start is after as_of session")
        self.acquisition_manifest = build_current_index_acquisition_manifest(
            self.profile_path, self.bootstrap
        )
        if self.provider.name != self.acquisition_manifest.profile.provider:
            raise ValueError("onboarding provider is not bound to acquisition profile")
        self._hold = _RunnerHold(self.store)
        self._calendar_by_symbol = _listing_calendar(self.bootstrap)
        known_listing_ids = {listing.listing_id for listing in self.acquisition_manifest.listings}
        if self.requested_listing_ids != tuple(sorted(set(self.requested_listing_ids))) or (
            set(self.requested_listing_ids) - known_listing_ids
        ):
            raise ValueError("data.candidate_recheck_scope_invalid")
        if set(self.requested_listing_ids) == known_listing_ids:
            self.requested_listing_ids = ()
        if self.retry_of is not None and (
            not self.requested_listing_ids or re.fullmatch(r"[0-9a-f]{64}", self.retry_of) is None
        ):
            raise ValueError("data.candidate_recheck_reference_invalid")
        selected = set(self.requested_listing_ids) or known_listing_ids
        self._calendar_by_listing_id = {
            listing.listing_id: self._calendar_by_symbol[listing.symbol]
            for listing in self.acquisition_manifest.listings
            if listing.listing_id in selected
        }
        if not set(self.retained_listing_ids).issubset(selected):
            raise ValueError("retained listing is absent from the acquisition work scope")
        self.onboarding_id = _onboarding_id(
            self.bootstrap,
            self.acquisition_manifest,
            history_start=self.history_start,
            as_of_session=self.as_of_session,
            requested_listing_ids=self.requested_listing_ids,
            retry_of=self.retry_of,
        )
        self._calendar_cache: dict[str, tuple[object, tuple[dict[str, object], ...]]] = {}
        self._admitted = False

    def admit(self, *, observed_at: datetime) -> None:
        """Admit the candidate units and bind available retained receipts."""
        # Admission is idempotent durable state: the onboarding row, its listing
        # units and any retained receipts do not change between the bounded
        # ``run`` calls of one runner. The Host drives a full universe one
        # listing per call, so re-running the schema bootstrap and the staged
        # listing upsert every call cost about 0.13s each on a 518-name build
        # (measured 2026-09-11) for no new information. A fresh runner, which
        # is what a restart constructs, still re-admits from the store.
        self.store.admit_current_universe_onboarding(
            self.acquisition_manifest,
            onboarding_id=self.onboarding_id,
            candidate_manifest_hash=self.bootstrap.source_manifest.content_hash,
            candidate_manifest_document=candidate_manifest_document(self.bootstrap.source_manifest),
            history_start=self.history_start,
            as_of_session=self.as_of_session,
            calendar_by_listing_id=self._calendar_by_listing_id,
            requested_listing_ids=self.requested_listing_ids,
            observed_at=observed_at,
        )
        if self.retained_listing_ids:
            rebound = self.store.bind_available_action_audit_receipts_to_manifest(
                self.acquisition_manifest,
                requested_as_of=self.as_of_session,
                now=observed_at,
            )
            if not set(self.retained_listing_ids).issubset(rebound):
                # The listing worker records the specific typed failure.  This
                # admission step must never fill the gap with a full-history
                # retained Provider request. The admission is not marked
                # complete, so the next run rebinds again as it always did.
                return
        # The success marker is the last statement: an exception or an
        # incomplete rebind above leaves the flag unset and the next run
        # repeats the whole admission path.
        self._admitted = True

    def retained_progress(self) -> CurrentUniverseOnboardingOutcome:
        """Return durable unit counts without advancing the onboarding task.

        Before the first chunk, a fresh admission counts zero of its candidates;
        a resumed one reports its retained units.
        """
        return self._outcome(CurrentUniverseOnboardingStatus.RUNNING)

    def _mark_listing(
        self,
        item: CurrentUniverseOnboardingListing,
        *,
        state: str,
        observed_at: datetime,
        origin: Literal["ACQUIRED", "RETAINED", "LOCAL"] | None,
        raw_through: date | None = None,
        failure_code: str | None = None,
        tail_acquired: bool = False,
    ) -> None:
        """Record one unit transition durably, then announce it to the observer."""
        self.store.update_current_universe_onboarding_listing(
            onboarding_id=self.onboarding_id,
            listing_id=item.listing_id,
            state=state,
            raw_through=raw_through,
            failure_code=failure_code,
            observed_at=observed_at,
        )
        if self.listing_observer is not None:
            self.listing_observer(
                ListingUnitObservation(
                    listing_id=item.listing_id,
                    symbol=item.symbol,
                    state=state,
                    observed_at=observed_at,
                    run_start_state=item.state,
                    origin=origin,
                    raw_through=raw_through,
                    failure_code=failure_code,
                    tail_acquired=tail_acquired,
                )
            )

    def _admit_quality(
        self,
        item: CurrentUniverseOnboardingListing,
        *,
        eligible: bool,
        expected_sessions: int,
        observed_sessions: int,
        missing_sessions: int,
        missing_ratio: float,
        maximum_consecutive_gap: int,
        reasons: Sequence[str],
        observed_at: datetime,
    ) -> None:
        """Record the quality gate's verdict (which moves the unit's state), then announce it."""
        self.store.record_current_universe_quality_admission(
            onboarding_id=self.onboarding_id,
            listing_id=item.listing_id,
            eligible=eligible,
            expected_sessions=expected_sessions,
            observed_sessions=observed_sessions,
            missing_sessions=missing_sessions,
            missing_ratio=missing_ratio,
            maximum_consecutive_gap=maximum_consecutive_gap,
            reasons=reasons,
            observed_at=observed_at,
        )
        if self.listing_observer is not None:
            self.listing_observer(
                ListingUnitObservation(
                    listing_id=item.listing_id,
                    symbol=item.symbol,
                    state="QUALITY_ELIGIBLE" if eligible else "QUALITY_INELIGIBLE",
                    observed_at=observed_at,
                    run_start_state=item.state,
                    origin="LOCAL",
                    reasons=tuple(reasons),
                )
            )

    def run(
        self,
        *,
        observed_at: datetime | None = None,
        work_budget: int | None = None,
        minimum_target_listings: int | None = None,
    ) -> CurrentUniverseOnboardingOutcome:
        """Advance at most ``work_budget`` complete listing units, then return safely."""
        now = observed_at or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("onboarding observed_at must be timezone-aware")
        if self.as_of_session > now.date():
            raise ValueError("onboarding as_of session cannot be in the future")
        if work_budget is not None and work_budget < 1:
            raise ValueError("onboarding work_budget must be positive")
        if minimum_target_listings is not None and minimum_target_listings < 1:
            raise ValueError("onboarding minimum_target_listings must be positive")
        with self._retained():
            opened = self._open_run(now, work_budget=work_budget)
        if not isinstance(opened, tuple):
            return opened
        candidates, effective_workers = opened
        for chunk_start in range(0, len(candidates), self.hydration_chunk_size):
            chunk = candidates[chunk_start : chunk_start + self.hydration_chunk_size]
            provider_wide_failures = self._advance_chunk(
                chunk,
                observed_at=now,
                workers=effective_workers,
            )
            if provider_wide_failures:
                with self._retained():
                    deferred = self._defer_hydration(
                        provider_wide_failures,
                        observed_at=now,
                        workers=effective_workers,
                    )
                    return self._outcome(
                        CurrentUniverseOnboardingStatus.DEFERRED,
                        failure_code=deferred.failure_code,
                        deferred=deferred,
                    )
        with self._retained():
            return self._close_run(now, minimum_target_listings=minimum_target_listings)

    @contextmanager
    def _retained(self) -> Iterator[None]:
        """One database instance for a bounded unit of this runner's work.

        The maintenance runner's contract: every read and short transaction
        of the unit costs a cursor rather than an instance open, a metadata
        read and a checkpoint; each transaction still begins and ends on its
        own. A unit ends before the Provider is awaited and starts again on
        the first hydration that is back, so nothing is held across a fetch;
        the audit's evidence fallback to the Provider releases the hold for
        its own fetch (``_RunnerHold.released``).
        """
        if self._hold.held:
            yield
            return
        self._hold.take()
        try:
            yield
        finally:
            self._hold.release()

    def _open_run(
        self, now: datetime, *, work_budget: int | None
    ) -> (
        CurrentUniverseOnboardingOutcome | tuple[tuple[CurrentUniverseOnboardingListing, ...], int]
    ):
        """Admit, resolve a deferral, and select this run's candidates."""
        if not self._admitted:
            self.admit(observed_at=now)
        completed = self.store.current_universe_onboarding_run(self.onboarding_id)
        if completed.lifecycle == "COMPLETED":
            manifest = (
                self.store.load_universe_manifest(completed.research_manifest_id)
                if completed.research_manifest_id is not None
                else None
            )
            units = self.store.current_universe_onboarding_listings(self.onboarding_id)
            qualified = {item.listing_id for item in units if item.state == "FEATURE_READY"}
            members = {item.listing_id for item in manifest.listings} if manifest else set()
            revision = manifest.revision_sha256 if manifest else None
            if (
                revision != completed.research_manifest_revision
                or members != qualified
                or (manifest is None and not self.requested_listing_ids)
            ):
                raise ValueError("data.completed_onboarding_binding_invalid")
            return self._outcome(
                CurrentUniverseOnboardingStatus.COMPLETED,
                research_manifest=manifest,
            )
        durable_worker_limit = self.store.hydration_worker_limit(self.onboarding_id)
        effective_workers = min(
            self.hydration_workers,
            durable_worker_limit if durable_worker_limit is not None else self.hydration_workers,
        )
        existing_deferred = self.store.current_hydration_deferred(self.onboarding_id)
        if existing_deferred is not None:
            if now.astimezone(UTC) < existing_deferred.retry_after_at:
                return self._outcome(
                    CurrentUniverseOnboardingStatus.DEFERRED,
                    failure_code=existing_deferred.failure_code,
                    deferred=existing_deferred,
                )
            resolved = self.store.resolve_hydration_deferred(
                existing_deferred.deferred_retry_id, observed_at=now
            )
            effective_workers = resolved.next_workers
        self.store.set_current_universe_onboarding_lifecycle(
            self.onboarding_id, lifecycle="RUNNING", observed_at=now
        )
        candidates = tuple(
            item
            for item in self.store.current_universe_onboarding_listings(self.onboarding_id)
            if item.state in {"PENDING", "RAW_READY", "QUALITY_ELIGIBLE"}
        )
        if work_budget is not None:
            candidates = candidates[:work_budget]
        return candidates, effective_workers

    def _close_run(
        self, now: datetime, *, minimum_target_listings: int | None = None
    ) -> CurrentUniverseOnboardingOutcome:
        """Report progress, or complete the onboarding once no unit is left."""
        listings = self.store.current_universe_onboarding_listings(self.onboarding_id)
        if any(item.state in {"PENDING", "RAW_READY", "QUALITY_ELIGIBLE"} for item in listings):
            return self._outcome(CurrentUniverseOnboardingStatus.RUNNING)
        listings = self.store.current_universe_onboarding_listings(self.onboarding_id)
        if any(item.state == "QUALITY_ELIGIBLE" for item in listings):
            return self._outcome(CurrentUniverseOnboardingStatus.RUNNING)

        eligible_listing_ids = tuple(
            item.listing_id for item in listings if item.state == "FEATURE_READY"
        )
        if eligible_listing_ids and minimum_target_listings is not None:
            ranges = self.store.listing_raw_ranges(eligible_listing_ids, through=self.as_of_session)
            reaching = {
                listing_id
                for listing_id, (_, latest) in ranges.items()
                if latest == self.as_of_session
            }
            if len(reaching) < minimum_target_listings and len(reaching) < len(
                eligible_listing_ids
            ):
                # Keep verified target units. Lagging units re-enter this same
                # onboarding's provider path on resume, before any manifest is published.
                for item in listings:
                    if item.state == "FEATURE_READY" and item.listing_id not in reaching:
                        self._mark_listing(
                            item,
                            state="PENDING",
                            observed_at=now,
                            origin=None,
                            raw_through=ranges[item.listing_id][1],
                        )
                self.store.set_current_universe_onboarding_lifecycle(
                    self.onboarding_id, lifecycle="DEFERRED", observed_at=now
                )
                raise DataTargetSessionLag(
                    target=self.as_of_session,
                    reaching=len(reaching),
                    total=len(eligible_listing_ids),
                    minimum=minimum_target_listings,
                    latest=min(latest for _, latest in ranges.values()),
                )
        if not eligible_listing_ids:
            if self.requested_listing_ids:
                self.store.set_current_universe_onboarding_lifecycle(
                    self.onboarding_id, lifecycle="COMPLETED", observed_at=now
                )
                return self._outcome(CurrentUniverseOnboardingStatus.COMPLETED)
            self.store.set_current_universe_onboarding_lifecycle(
                self.onboarding_id, lifecycle="BLOCKED", observed_at=now
            )
            return self._outcome(
                CurrentUniverseOnboardingStatus.BLOCKED,
                failure_code="data.current_universe_no_feature_ready_listings",
            )
        quality_admission_hash = self._quality_admission_hash(listings)
        research_manifest = build_quality_filtered_research_manifest(
            self.acquisition_manifest,
            eligible_listing_ids=eligible_listing_ids,
            quality_admission_hash=quality_admission_hash,
        )
        self.store.bind_action_audit_receipts_to_manifest(
            self.acquisition_manifest,
            research_manifest,
            requested_as_of=self.as_of_session,
            now=now,
        )
        self.store.complete_current_universe_onboarding(
            onboarding_id=self.onboarding_id,
            research_manifest=research_manifest,
            quality_admission_hash=quality_admission_hash,
            observed_at=now,
        )
        return self._outcome(
            CurrentUniverseOnboardingStatus.COMPLETED,
            research_manifest=research_manifest,
        )

    def _advance_chunk(
        self,
        items: Sequence[CurrentUniverseOnboardingListing],
        *,
        observed_at: datetime,
        workers: int,
    ) -> tuple[tuple[CurrentUniverseOnboardingListing, ProviderFetchError], ...]:
        pending = tuple(item for item in items if item.state == "PENDING")
        retained_ids = set(self.retained_listing_ids)
        retained_pending = tuple(item for item in pending if item.listing_id in retained_ids)
        addition_pending = tuple(item for item in pending if item.listing_id not in retained_ids)
        resumed = tuple(item for item in items if item.state != "PENDING")
        provider_wide: list[tuple[CurrentUniverseOnboardingListing, ProviderFetchError]] = []
        if addition_pending and isinstance(self.provider, HistoricalHydrationProvider):
            with ThreadPoolExecutor(max_workers=min(workers, len(addition_pending))) as executor:
                futures: dict[
                    Future[tuple[HydrationEvidence, float]], CurrentUniverseOnboardingListing
                ] = {
                    executor.submit(self._fetch_hydration_timed, item): item
                    for item in addition_pending
                }
                # One retained instance serves every hydration that is back and
                # ready to apply; it is taken on a completion, never before one,
                # and released again whenever the next completion is still in
                # flight, so nothing is held while the Provider is awaited.
                pending = set(futures)
                try:
                    for future in as_completed(futures):
                        pending.discard(future)
                        self._hold.take()
                        item = futures[future]
                        try:
                            evidence, network_elapsed = future.result()
                        except ProviderFetchError as exc:
                            if exc.code in {"data.rate_limited", "data.provider_session_unstable"}:
                                provider_wide.append((item, exc))
                            else:
                                self._record_listing_failure(
                                    item,
                                    code=exc.code,
                                    observed_at=observed_at,
                                    state="RAW_FAILED",
                                )
                        else:
                            self._record_elapsed(
                                "raw_network",
                                network_elapsed,
                                {"listing_count": 1, "provider_calls": 1},
                            )
                            self._process_hydrated_listing(item, evidence, observed_at=observed_at)
                        if pending and not any(f.done() for f in pending):
                            self._hold.release()
                finally:
                    self._hold.release()
        elif addition_pending:
            resumed = (*resumed, *addition_pending)
        resumed = (*resumed, *retained_pending)
        for item in resumed:
            # A unit that already holds its raw history only reads and writes
            # the store; a pending one reaches the Provider first and is not
            # retained across that fetch.
            if item.state == "PENDING":
                result = self._advance_listing_without_evidence(item, observed_at=observed_at)
            else:
                with self._retained():
                    result = self._advance_listing_without_evidence(item, observed_at=observed_at)
            if isinstance(result, ProviderFetchError):
                provider_wide.append((item, result))
        return tuple(provider_wide)

    def _fetch_hydration_with_retry(
        self, item: CurrentUniverseOnboardingListing
    ) -> HydrationEvidence:
        if not isinstance(self.provider, HistoricalHydrationProvider):
            raise TypeError("provider does not implement historical hydration")
        listing = self._listing(item.listing_id)
        for attempt in range(1, self.retry_budget + 1):
            try:
                return self.provider.fetch_hydration(
                    listing_id=listing.listing_id,
                    provider_symbol=listing.provider_symbol,
                    start=self.history_start,
                    end=self.as_of_session,
                )
            except ProviderFetchError as exc:
                if exc.code in {"data.rate_limited", "data.provider_session_unstable"}:
                    raise
                if not exc.retryable or attempt == self.retry_budget:
                    raise
        raise AssertionError("bounded hydration retry did not return")

    def _fetch_hydration_timed(
        self, item: CurrentUniverseOnboardingListing
    ) -> tuple[HydrationEvidence, float]:
        started = time.perf_counter()
        return self._fetch_hydration_with_retry(item), time.perf_counter() - started

    def _process_hydrated_listing(
        self,
        item: CurrentUniverseOnboardingListing,
        evidence: HydrationEvidence,
        *,
        observed_at: datetime,
    ) -> None:
        listing = self._listing(item.listing_id)
        sanitizer_started = time.perf_counter()
        try:
            batch = sanitize_payload(
                self.acquisition_manifest,
                self.provider.name,
                {listing.symbol: evidence.daily_rows},
                (listing.symbol,),
            )
            raw_through = max(bar.session_date for bar in batch.bars)
        except (CorruptedPayload, ValueError):
            self._record_listing_failure(
                item,
                code="data.sanitizer.corrupted_payload",
                observed_at=observed_at,
                state="RAW_FAILED",
            )
            return
        self._record_timing(
            "sanitizer",
            sanitizer_started,
            {"listing_count": 1, "row_count": len(batch.bars)},
        )
        persistence_started = time.perf_counter()
        self.store.apply_validated_batch(
            self.acquisition_manifest,
            batch,
            ingestion_id=f"{self.onboarding_id}:raw:{listing.listing_id}",
            observed_at=observed_at,
        )
        self._record_timing(
            "raw_persistence",
            persistence_started,
            {"listing_count": 1, "row_count": len(batch.bars)},
        )
        self._mark_listing(
            item,
            state="RAW_READY",
            raw_through=raw_through,
            observed_at=observed_at,
            origin="ACQUIRED",
        )
        quality_started = time.perf_counter()
        qualified = self._qualify_raw_history(item, observed_at=observed_at)
        self._record_timing(
            "calendar_quality",
            quality_started,
            {"listing_count": 1, "qualified_count": int(qualified)},
        )
        if qualified:
            audit_started = time.perf_counter()
            self._audit_feature_admission(item, observed_at=observed_at, evidence=evidence)
            self._record_timing(
                "action_audit",
                audit_started,
                {"listing_count": 1},
            )

    def _record_timing(
        self,
        stage: str,
        started: float,
        metrics: Mapping[str, object],
    ) -> None:
        if self.timing_sink is not None:
            self.timing_sink(stage, time.perf_counter() - started, metrics)

    def _record_elapsed(self, stage: str, elapsed: float, metrics: Mapping[str, object]) -> None:
        if self.timing_sink is not None:
            self.timing_sink(stage, elapsed, metrics)

    def _advance_listing_without_evidence(
        self, item: CurrentUniverseOnboardingListing, *, observed_at: datetime
    ) -> ProviderFetchError | None:
        state = item.state
        if state == "PENDING":
            result = (
                self._refresh_retained_raw(item, observed_at=observed_at)
                if item.listing_id in set(self.retained_listing_ids)
                else self._refresh_raw(item, observed_at=observed_at)
            )
            if isinstance(result, ProviderFetchError):
                return result
            if result != "READY":
                return None
            state = "RAW_READY"
        if state == "RAW_READY":
            if not self._qualify_raw_history(item, observed_at=observed_at):
                return None
            state = "QUALITY_ELIGIBLE"
        if state == "QUALITY_ELIGIBLE":
            result = self._audit_feature_admission(item, observed_at=observed_at)
            return result if isinstance(result, ProviderFetchError) else None
        return None

    def _refresh_retained_raw(
        self, item: CurrentUniverseOnboardingListing, *, observed_at: datetime
    ) -> str | ProviderFetchError:
        """Reuse durable history and fetch only a retained listing's missing tail."""
        listing = self._listing(item.listing_id)
        existing = self.store.raw_bars(item.listing_id, through=self.as_of_session)
        if not existing:
            self._record_listing_failure(
                item,
                code="data.retained_history_not_reusable",
                observed_at=observed_at,
                state="RAW_FAILED",
            )
            return "FAILED"
        raw_through = existing[-1].session_date
        tail_acquired = raw_through < self.as_of_session
        if raw_through < self.as_of_session:
            overlap_start = max(self.history_start, raw_through - timedelta(days=45))
            for attempt in range(1, self.retry_budget + 1):
                try:
                    if isinstance(self.provider, HistoricalHydrationProvider):
                        evidence = self.provider.fetch_hydration(
                            listing_id=listing.listing_id,
                            provider_symbol=listing.provider_symbol,
                            start=overlap_start,
                            end=self.as_of_session,
                        )
                    else:
                        payload = self.provider.fetch_daily(
                            (listing.provider_symbol,),
                            start=overlap_start,
                            end=self.as_of_session,
                        )
                        evidence = HydrationEvidence(
                            daily_rows=tuple(payload[listing.provider_symbol]),
                            actions=tuple(
                                self.provider.fetch_action_history(
                                    listing_id=listing.listing_id,
                                    provider_symbol=listing.provider_symbol,
                                    start=overlap_start,
                                    end=self.as_of_session,
                                )
                            ),
                            adjusted_closes=tuple(
                                self.provider.fetch_adjusted_close_history(
                                    listing_id=listing.listing_id,
                                    provider_symbol=listing.provider_symbol,
                                    start=overlap_start,
                                    end=self.as_of_session,
                                )
                            ),
                        )
                    translated = {listing.symbol: evidence.daily_rows}
                except KeyError:
                    self._record_listing_failure(
                        item,
                        code="data.symbol_mismatch",
                        observed_at=observed_at,
                        state="RAW_FAILED",
                    )
                    return "FAILED"
                except ProviderFetchError as exc:
                    if exc.retryable and attempt < self.retry_budget:
                        continue
                    if exc.code in {"data.rate_limited", "data.provider_session_unstable"}:
                        return exc
                    self._record_listing_failure(
                        item, code=exc.code, observed_at=observed_at, state="RAW_FAILED"
                    )
                    return "FAILED"
                try:
                    batch = sanitize_payload(
                        self.acquisition_manifest,
                        self.provider.name,
                        translated,
                        (listing.symbol,),
                    )
                except (CorruptedPayload, ValueError):
                    self._record_listing_failure(
                        item,
                        code="data.sanitizer.corrupted_payload",
                        observed_at=observed_at,
                        state="RAW_FAILED",
                    )
                    return "FAILED"
                self.store.apply_validated_batch(
                    self.acquisition_manifest,
                    batch,
                    ingestion_id=f"{self.onboarding_id}:retained-tail:{listing.listing_id}",
                    observed_at=observed_at,
                )
                if not self._complete_retained_action_audit(
                    item, evidence=evidence, observed_at=observed_at
                ):
                    return "FAILED"
                raw_through = max(raw_through, max(bar.session_date for bar in batch.bars))
                break
        self._mark_listing(
            item,
            state="RAW_READY",
            raw_through=raw_through,
            observed_at=observed_at,
            origin="RETAINED",
            tail_acquired=tail_acquired,
        )
        return "READY"

    def _complete_retained_action_audit(
        self,
        item: CurrentUniverseOnboardingListing,
        *,
        evidence: HydrationEvidence,
        observed_at: datetime,
    ) -> bool:
        """Splice bounded fresh evidence onto durable history before publication."""
        bars = self.store.raw_bars(item.listing_id, through=self.as_of_session)
        action_by_key = {
            (value.effective_date, value.action_kind): value
            for value in self.store.actions(item.listing_id)
        }
        action_by_key.update(
            {(value.effective_date, value.action_kind): value for value in evidence.actions}
        )
        adjusted_by_session = {
            value.session_date: value
            for value in self.store.provider_adjusted_closes(
                item.listing_id, through=self.as_of_session
            )
        }
        adjusted_by_session.update(
            {value.session_date: value for value in evidence.adjusted_closes}
        )
        try:
            self.store.complete_action_audit(
                self.acquisition_manifest,
                listing_id=item.listing_id,
                provider=self.provider.name,
                observed_actions=tuple(action_by_key[key] for key in sorted(action_by_key)),
                observed_adjusted_closes=tuple(
                    adjusted_by_session[key] for key in sorted(adjusted_by_session)
                ),
                history_start=bars[0].session_date,
                history_end=self.as_of_session,
                requested_as_of=self.as_of_session,
                observed_at=observed_at,
            )
        except ValueError:
            self._record_listing_failure(
                item,
                code="data.retained_action_evidence_not_reusable",
                observed_at=observed_at,
                state="AUDIT_FAILED",
            )
            return False
        return True

    def _defer_hydration(
        self,
        failures: Sequence[tuple[CurrentUniverseOnboardingListing, ProviderFetchError]],
        *,
        observed_at: datetime,
        workers: int,
    ) -> HydrationDeferred:
        failure_code = (
            "data.rate_limited"
            if any(exc.code == "data.rate_limited" for _item, exc in failures)
            else "data.provider_session_unstable"
        )
        affected = tuple(
            sorted(
                item.listing_id
                for item in self.store.current_universe_onboarding_listings(self.onboarding_id)
                if item.state in {"PENDING", "RAW_READY", "QUALITY_ELIGIBLE"}
            )
        )
        if not affected:
            affected = tuple(sorted(item.listing_id for item, _exc in failures))
        affected_hash = sha256("|".join(affected).encode("utf-8")).hexdigest()
        defer_count = self.store.hydration_defer_count(self.onboarding_id)
        backoff_seconds = (300, 900, 2700)[min(defer_count, 2)]
        jitter_seconds = (
            int(
                sha256(f"{self.onboarding_id}:{defer_count}".encode()).hexdigest()[:8],
                16,
            )
            % 31
        )
        policy_hash = sha256(
            json.dumps(
                {
                    "chunk_size": self.hydration_chunk_size,
                    "retry_delay_seconds": (300, 900, 2700),
                    "maximum_jitter_seconds": 30,
                    "degradation": "halve_to_one",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        retry_after = observed_at + timedelta(seconds=backoff_seconds + jitter_seconds)
        deferred_id = sha256(
            "|".join(
                (
                    self.onboarding_id,
                    failure_code,
                    observed_at.isoformat(),
                    str(workers),
                    affected_hash,
                    policy_hash,
                )
            ).encode("utf-8")
        ).hexdigest()
        deferred = HydrationDeferred(
            onboarding_id=self.onboarding_id,
            deferred_retry_id=deferred_id,
            failure_code=failure_code,
            retry_after_at=retry_after,
            observed_workers=workers,
            next_workers=max(1, workers // 2),
            affected_listing_set_hash=affected_hash,
            transport_policy_hash=policy_hash,
            created_at=observed_at,
        )
        self.store.record_hydration_deferred(deferred)
        return deferred

    def _refresh_raw(
        self, item: CurrentUniverseOnboardingListing, *, observed_at: datetime
    ) -> str | ProviderFetchError:
        listing = self._listing(item.listing_id)
        for attempt in range(1, self.retry_budget + 1):
            try:
                payload = self.provider.fetch_daily(
                    (listing.provider_symbol,),
                    start=self.history_start,
                    end=self.as_of_session,
                )
                translated = {listing.symbol: payload[listing.provider_symbol]}
            except KeyError:
                self._record_listing_failure(
                    item, code="data.symbol_mismatch", observed_at=observed_at, state="RAW_FAILED"
                )
                return "FAILED"
            except ProviderFetchError as exc:
                if exc.retryable and attempt < self.retry_budget:
                    continue
                if exc.code in {"data.rate_limited", "data.provider_session_unstable"}:
                    return exc
                self._record_listing_failure(
                    item, code=exc.code, observed_at=observed_at, state="RAW_FAILED"
                )
                return "FAILED"
            try:
                batch = sanitize_payload(
                    self.acquisition_manifest,
                    self.provider.name,
                    translated,
                    (listing.symbol,),
                )
                raw_through = max(bar.session_date for bar in batch.bars)
            except (CorruptedPayload, ValueError):
                self._record_listing_failure(
                    item,
                    code="data.sanitizer.corrupted_payload",
                    observed_at=observed_at,
                    state="RAW_FAILED",
                )
                return "FAILED"
            # A write or store failure is an owner failure, not a corrupt
            # provider payload. Propagate it instead of silently excluding a
            # listing under a misleading sanitizer code.
            self.store.apply_validated_batch(
                self.acquisition_manifest,
                batch,
                ingestion_id=f"{self.onboarding_id}:raw:{listing.listing_id}",
                observed_at=observed_at,
            )
            self._mark_listing(
                item,
                state="RAW_READY",
                raw_through=raw_through,
                observed_at=observed_at,
                origin="ACQUIRED",
            )
            return "READY"
        raise AssertionError("bounded raw refresh did not return")

    def _qualify_raw_history(
        self, item: CurrentUniverseOnboardingListing, *, observed_at: datetime
    ) -> bool:
        cached = self._calendar_cache.get(item.calendar_id)
        if cached is None:
            schedule = materialize_calendar_schedule(
                (item.calendar_id,),
                start=self.history_start,
                end=self.as_of_session,
                as_of_timestamp=observed_at,
            )
            schedule_rows = tuple(schedule.to_pylist())
            self._calendar_cache[item.calendar_id] = (schedule, schedule_rows)
        else:
            schedule, schedule_rows = cached
        bars = tuple(
            bar
            for bar in self.store.raw_bars(item.listing_id, through=self.as_of_session)
            if bar.session_date >= self.history_start
        )
        expected_dates = {
            row["session_date"] for row in schedule_rows if row["calendar_id"] == item.calendar_id
        }
        observed_dates = {bar.session_date for bar in bars}
        if not expected_dates or not observed_dates or min(observed_dates) > min(expected_dates):
            missing = expected_dates - observed_dates
            self._admit_quality(
                item,
                eligible=False,
                expected_sessions=len(expected_dates),
                observed_sessions=len(observed_dates & expected_dates),
                missing_sessions=len(missing),
                missing_ratio=len(missing) / len(expected_dates) if expected_dates else 1.0,
                maximum_consecutive_gap=len(missing),
                reasons=("insufficient_research_history",),
                observed_at=observed_at,
            )
            return False
        unexpected_dates = sorted({bar.session_date for bar in bars} - expected_dates)
        if unexpected_dates:
            self._admit_quality(
                item,
                eligible=False,
                expected_sessions=len(expected_dates),
                observed_sessions=len({bar.session_date for bar in bars} & expected_dates),
                missing_sessions=len(expected_dates - {bar.session_date for bar in bars}),
                missing_ratio=(
                    len(expected_dates - {bar.session_date for bar in bars}) / len(expected_dates)
                ),
                maximum_consecutive_gap=0,
                reasons=("provider_returned_non_session_dates",),
                observed_at=observed_at,
            )
            return False
        closes = {row["session_date"]: row["session_close_timestamp"] for row in schedule_rows}
        table = coerce_daily_records(
            self._quality_rows(
                symbol=item.symbol,
                calendar_id=item.calendar_id,
                bars=bars,
                closes=closes,
            )
        )
        report = diagnose_daily_table(
            table,
            symbol_calendars={item.symbol: item.calendar_id},
            schedule=schedule,
            as_of_timestamp=observed_at,
            created_at=observed_at,
        ).symbol_reports[0]
        self._admit_quality(
            item,
            eligible=report.eligible,
            expected_sessions=report.expected_sessions,
            observed_sessions=report.observed_sessions,
            missing_sessions=report.missing_sessions,
            missing_ratio=report.missing_ratio,
            maximum_consecutive_gap=report.maximum_consecutive_gap,
            reasons=report.reasons,
            observed_at=observed_at,
        )
        return bool(report.eligible)

    def _audit_feature_admission(
        self,
        item: CurrentUniverseOnboardingListing,
        *,
        observed_at: datetime,
        evidence: HydrationEvidence | None = None,
    ) -> str | ProviderFetchError:
        listing = self._listing(item.listing_id)
        bars = self.store.raw_bars(item.listing_id, through=self.as_of_session)
        if not bars:
            self._record_listing_failure(
                item, code="data.no_raw_history", observed_at=observed_at, state="AUDIT_FAILED"
            )
            return "FAILED"
        receipt = self.store.reusable_action_audit_receipt(
            self.acquisition_manifest,
            listing_id=item.listing_id,
            provider=self.provider.name,
            requested_as_of=self.as_of_session,
            now=observed_at,
        )
        if receipt is None and item.listing_id in set(self.retained_listing_ids):
            receipt = self._retained_audit_chain(item, observed_at=observed_at)
        if receipt is None and item.listing_id in set(self.retained_listing_ids):
            self._record_listing_failure(
                item,
                code="data.retained_action_evidence_not_reusable",
                observed_at=observed_at,
                state="AUDIT_FAILED",
            )
            return "FAILED"
        try:
            if receipt is None:
                receipt = self._complete_action_audit(
                    item,
                    listing.provider_symbol,
                    bars,
                    observed_at=observed_at,
                    evidence=evidence,
                )
        except ProviderFetchError as exc:
            if exc.code in {"data.rate_limited", "data.provider_session_unstable"}:
                return exc
            self._record_listing_failure(
                item, code=exc.code, observed_at=observed_at, state="AUDIT_FAILED"
            )
            return "FAILED"
        except ValueError:
            self._record_listing_failure(
                item,
                code="data.action_audit_invalid",
                observed_at=observed_at,
                state="AUDIT_FAILED",
            )
            return "FAILED"
        assert receipt is not None
        adjusted = self.store.provider_adjusted_closes(
            item.listing_id, through=self.as_of_session, start=bars[0].session_date
        )
        if {point.session_date for point in adjusted} != {bar.session_date for bar in bars}:
            self._record_listing_failure(
                item,
                code="data.provider_adjusted_series_incomplete",
                observed_at=observed_at,
                state="AUDIT_FAILED",
            )
            return "FAILED"
        self._mark_listing(item, state="FEATURE_READY", observed_at=observed_at, origin="LOCAL")
        return "READY"

    def _retained_audit_chain(
        self, item: CurrentUniverseOnboardingListing, *, observed_at: datetime
    ) -> ActionAuditReceipt | None:
        """Complete retained coverage from its unchanged anchor and current rolling proof.

        A same-session roster change needs no new tail. Its daily audit still
        covers only the rolling range: retain both witnesses as recorded,
        never manufacture a fresh full-history Provider receipt.
        """
        anchors = self.store.latest_action_audit_receipts(
            (item.listing_id,), provider=self.provider.name
        )
        links = self.store.latest_action_audit_receipts(
            (item.listing_id,),
            requested_as_of=self.as_of_session,
            provider=self.provider.name,
        )
        anchor, link = anchors.get(item.listing_id), links.get(item.listing_id)
        if anchor is None or link is None or anchor.mapping_revision != link.mapping_revision:
            return None
        anchor = self.store.verified_action_audit_receipt(
            self.acquisition_manifest,
            receipt_hash=anchor.receipt_hash,
            listing_id=item.listing_id,
            requested_as_of=anchor.requested_as_of,
            now=observed_at,
            allow_historical=True,
            provider=self.provider.name,
        )
        link = self.store.verified_action_audit_receipt(
            self.acquisition_manifest,
            receipt_hash=link.receipt_hash,
            listing_id=item.listing_id,
            requested_as_of=self.as_of_session,
            now=observed_at,
            allow_equivalent_manifest=True,
            provider=self.provider.name,
        )
        if anchor is None or link is None:
            return None
        bars = self.store.raw_bars(item.listing_id, through=self.as_of_session)
        if not bars or any(
            not (
                anchor.history_start <= bar.session_date <= anchor.history_end
                or link.history_start <= bar.session_date <= link.history_end
            )
            for bar in bars
        ):
            return None
        return link

    def _complete_action_audit(
        self,
        item: CurrentUniverseOnboardingListing,
        provider_symbol: str,
        bars: Sequence[RawDailyBar],
        *,
        observed_at: datetime,
        evidence: HydrationEvidence | None,
    ):
        history_start = bars[0].session_date

        def complete(actions, adjusted_closes):
            return self.store.complete_action_audit(
                self.acquisition_manifest,
                listing_id=item.listing_id,
                provider=self.provider.name,
                observed_actions=actions,
                observed_adjusted_closes=adjusted_closes,
                history_start=history_start,
                history_end=self.as_of_session,
                requested_as_of=self.as_of_session,
                observed_at=observed_at,
            )[0]

        if evidence is not None:
            try:
                actions = tuple(
                    action
                    for action in evidence.actions
                    if history_start <= action.effective_date <= self.as_of_session
                )
                adjusted_closes = tuple(
                    point
                    for point in evidence.adjusted_closes
                    if history_start <= point.session_date <= self.as_of_session
                )
                if {point.session_date for point in adjusted_closes} != {
                    bar.session_date for bar in bars
                }:
                    raise ValueError("hydration adjusted-close sessions do not match raw bars")
                return complete(actions, adjusted_closes)
            except ValueError:
                # Ephemeral evidence is an optimization, never an authority.
                # Any scope mismatch falls back to the established provider path.
                pass
        with self._hold.released():
            actions = self.provider.fetch_action_history(
                listing_id=item.listing_id,
                provider_symbol=provider_symbol,
                start=history_start,
                end=self.as_of_session,
            )
            adjusted_closes = self.provider.fetch_adjusted_close_history(
                listing_id=item.listing_id,
                provider_symbol=provider_symbol,
                start=history_start,
                end=self.as_of_session,
            )
        return complete(actions, adjusted_closes)

    def _quality_rows(
        self,
        *,
        symbol: str,
        calendar_id: str,
        bars: Sequence[RawDailyBar],
        closes: dict[date, datetime],
    ) -> Sequence[dict[str, object]]:
        return tuple(
            {
                "symbol": symbol,
                "calendar_id": calendar_id,
                "session_date": bar.session_date,
                "observation_timestamp": closes[bar.session_date],
                "availability_timestamp": closes[bar.session_date],
                "availability_method": "INFERRED_SESSION_CLOSE",
                "open_raw": bar.open,
                "high_raw": bar.high,
                "low_raw": bar.low,
                "close_raw": bar.close,
                "volume_raw": bar.volume,
                "open_split_adjusted": bar.open,
                "high_split_adjusted": bar.high,
                "low_split_adjusted": bar.low,
                "close_split_adjusted": bar.close,
                "close_total_return_adjusted": None,
                "cash_dividend": 0.0,
                "split_ratio": 1.0,
                "provider": self.provider.name,
                "attempt_id": f"quality:{self.onboarding_id}:{bar.listing_id}",
            }
            for bar in bars
        )

    def _listing(self, listing_id: str):
        listing = next(
            (item for item in self.acquisition_manifest.listings if item.listing_id == listing_id),
            None,
        )
        if listing is None:
            raise ValueError("onboarding listing is absent from acquisition manifest")
        return listing

    def _record_listing_failure(
        self,
        item: CurrentUniverseOnboardingListing,
        *,
        code: str,
        observed_at: datetime,
        state: str,
    ) -> None:
        self.store.record_failures(
            (
                FailureEvidence(
                    item.listing_id,
                    self.acquisition_manifest.profile.market_profile_id,
                    code,
                    self.history_start,
                    self.as_of_session,
                    observed_at,
                ),
            )
        )
        self._mark_listing(
            item, state=state, failure_code=code, observed_at=observed_at, origin=None
        )

    def _quality_admission_hash(self, listings: Sequence[CurrentUniverseOnboardingListing]) -> str:
        listing_outcomes = [
            {
                "listing_id": item.listing_id,
                "state": item.state,
                "failure_code": item.failure_code,
                "raw_through": item.raw_through.isoformat() if item.raw_through else None,
            }
            for item in listings
        ]
        quality_admissions = [
            {
                "listing_id": admission.listing_id,
                "eligible": admission.eligible,
                "expected_sessions": admission.expected_sessions,
                "observed_sessions": admission.observed_sessions,
                "missing_sessions": admission.missing_sessions,
                "missing_ratio": admission.missing_ratio,
                "maximum_consecutive_gap": admission.maximum_consecutive_gap,
                "reasons": admission.reasons,
            }
            for admission in self.store.current_universe_quality_admissions(self.onboarding_id)
        ]
        encoded = json.dumps(
            {
                "listing_outcomes": listing_outcomes,
                "quality_admissions": quality_admissions,
                "retained_listing_ids_revalidated": sorted(self.retained_listing_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(encoded.encode("utf-8")).hexdigest()

    def _outcome(
        self,
        status: CurrentUniverseOnboardingStatus,
        *,
        research_manifest: UniverseManifest | None = None,
        failure_code: str | None = None,
        deferred: HydrationDeferred | None = None,
    ) -> CurrentUniverseOnboardingOutcome:
        listings = self.store.current_universe_onboarding_listings(self.onboarding_id)
        quality_reasons = (
            {
                item.listing_id: item.reasons
                for item in self.store.current_universe_quality_admissions(self.onboarding_id)
            }
            if any(item.state == "QUALITY_INELIGIBLE" for item in listings)
            else {}
        )
        exclusion_counts = dict.fromkeys(
            ("acquisition_failure", "history_ineligible", "quality_rejection"), 0
        )
        for item in listings:
            category = listing_outcome_category(
                item.state, quality_reasons.get(item.listing_id, ()), item.failure_code
            )
            if category is not None:
                exclusion_counts[category.lower()] += 1
        return CurrentUniverseOnboardingOutcome(
            onboarding_id=self.onboarding_id,
            status=status,
            candidates=len(listings),
            raw_ready=sum(
                item.state != "PENDING" and item.raw_through is not None for item in listings
            ),
            quality_eligible=sum(
                item.state in {"QUALITY_ELIGIBLE", "FEATURE_READY"} for item in listings
            ),
            feature_ready=sum(item.state == "FEATURE_READY" for item in listings),
            failed=sum(
                item.state in {"RAW_FAILED", "AUDIT_FAILED", "QUALITY_INELIGIBLE"}
                for item in listings
            ),
            research_manifest=research_manifest,
            failure_code=failure_code,
            deferred_retry_id=deferred.deferred_retry_id if deferred else None,
            retry_after_at=deferred.retry_after_at if deferred else None,
            next_workers=deferred.next_workers if deferred else None,
            exclusion_counts=exclusion_counts,
        )
