"""Offline proof of a resumable full-current-universe onboarding task."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from scripts.run_current_universe_onboarding import _latest_common_us_session

from alphalattice.foundation.market_data_ops.runtime import universe_onboarding as onboarding_module
from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
    CurrentUniverseMaintenance,
    CurrentUniverseMaintenanceStatus,
)
from alphalattice.foundation.market_data_ops.runtime.universe_onboarding import (
    CurrentUniverseOnboarding,
    CurrentUniverseOnboardingStatus,
    ListingUnitObservation,
)
from alphalattice.foundation.market_data_ops.sources.contracts import ProviderAdjustedClosePoint
from alphalattice.foundation.market_data_ops.sources.providers import (
    HydrationEvidence,
    ProviderFetchError,
)
from alphalattice.foundation.market_data_ops.sources.universe import (
    ORIGINAL_RESEARCH_WHITELIST_STANDARD,
    CandidateMembershipEvidence,
    CurrentUniverseBootstrap,
    CurrentUniverseCandidate,
    CurrentUniverseCandidateManifest,
    bootstrap_from_candidate_manifest_document,
    candidate_manifest_document,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data.calendar import materialize_calendar_schedule

AS_OF = date(2026, 7, 31)
OBSERVED_AT = datetime(2026, 8, 2, tzinfo=UTC)
PROFILE = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "market-profiles"
    / "us-current-index-research.yaml"
)


def test_bulk_calendar_preserves_scalar_clocks_hashes_and_cutoff(monkeypatch):
    import exchange_calendars
    import pyarrow as pa

    from alphalattice.kernel.data.calendar import (
        calendar_schedule_schema,
        canonicalize_calendar_schedule,
    )
    from alphalattice.kernel.data.errors import DataQualityError

    periods = (
        (date(2016, 8, 1), date(2026, 8, 3), datetime(2026, 8, 3, 23, tzinfo=UTC)),
        (date(2025, 11, 26), date(2025, 11, 28), datetime(2025, 11, 28, 17, 59, tzinfo=UTC)),
        (date(2025, 11, 26), date(2025, 11, 28), datetime(2025, 11, 28, 18, tzinfo=UTC)),
        (date(2026, 3, 6), date(2026, 3, 9), datetime(2026, 3, 9, 20, tzinfo=UTC)),
    )
    calendars = {name: exchange_calendars.get_calendar(name) for name in ("XNAS", "XNYS")}
    references = []
    for start, end, observed in periods:
        rows = []
        for name, calendar in calendars.items():
            for session in calendar.sessions_in_range(start, end):
                opened = calendar.session_open(session).to_pydatetime().astimezone(UTC)
                closed = calendar.session_close(session).to_pydatetime().astimezone(UTC)
                if closed <= observed:
                    rows.append(
                        dict(
                            calendar_id=name,
                            session_date=session.date(),
                            session_open_timestamp=opened,
                            session_close_timestamp=closed,
                        )
                    )
        references.append(
            canonicalize_calendar_schedule(
                pa.Table.from_pylist(rows, schema=calendar_schedule_schema())
            )
        )

    def no_scalar_lookup(*_args, **_kwargs):
        raise AssertionError("calendar clocks were fetched one cell at a time")

    for calendar in calendars.values():
        monkeypatch.setattr(calendar, "session_open", no_scalar_lookup)
        monkeypatch.setattr(calendar, "session_close", no_scalar_lookup)
    for (start, end, observed), expected in zip(periods, references, strict=True):
        actual = materialize_calendar_schedule(
            (" xnys ", "XNAS", "XNYS"), start=start, end=end, as_of_timestamp=observed
        )
        assert actual.equals(expected)
    assert references[1].num_rows + 2 == references[2].num_rows
    for calendars_arg, start, end, observed, code in (
        (("UNKNOWN",), *periods[0], "data.provider_schema_drift"),
        (("XNYS",), periods[0][1], periods[0][0], periods[0][2], "data.provider_schema_drift"),
        (
            ("XNYS",),
            periods[0][0],
            periods[0][1],
            datetime(2026, 8, 3),
            "data.provider_schema_drift",
        ),
        (
            ("XNYS",),
            date(2026, 3, 9),
            date(2026, 3, 9),
            datetime(2026, 3, 9, 19, 59, tzinfo=UTC),
            "data.snapshot_incomplete",
        ),
    ):
        with pytest.raises(DataQualityError) as failure:
            materialize_calendar_schedule(
                calendars_arg, start=start, end=end, as_of_timestamp=observed
            )
        assert failure.value.failure.code == code


class FixtureProvider:
    name = "yfinance"

    def __init__(self, sessions: Sequence[date]) -> None:
        self.sessions = tuple(sessions)
        self.daily_calls: dict[str, int] = {}

    def fetch_daily(self, symbols: Sequence[str], *, start: date, end: date):
        symbol = symbols[0]
        self.daily_calls[symbol] = self.daily_calls.get(symbol, 0) + 1
        if symbol == "FAIL":
            raise ProviderFetchError(
                "data.provider_fetch_failed", "fixture failure", retryable=False
            )
        selected = tuple(session for session in self.sessions if start <= session <= end)
        if symbol == "BAD":
            selected = tuple(
                session for index, session in enumerate(selected) if index < 100 or index >= 160
            )
        return {
            symbol: tuple(
                {
                    "session_date": session.isoformat(),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1_000,
                    "split_ratio": 0.0,
                    "cash_dividend": 0.0,
                    "capital_gain": 0.0,
                }
                for session in selected
            )
        }

    def fetch_action_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ):
        del listing_id, provider_symbol, start, end
        return ()

    def fetch_adjusted_close_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ):
        return tuple(
            ProviderAdjustedClosePoint(
                listing_id=listing_id,
                provider=self.name,
                session_date=session,
                adjusted_close=100.0,
            )
            for session in self.sessions
            if start <= session <= end
        )


class HydrationFixtureProvider(FixtureProvider):
    """Thread-safe one-response fixture with a durable 4→2→1 rate-limit scenario."""

    def __init__(self, sessions: Sequence[date], *, provider_wide_failures: int) -> None:
        super().__init__(sessions)
        self.provider_wide_failures = provider_wide_failures
        self.hydration_calls: dict[str, int] = {}
        self.fallback_adjusted_calls: dict[str, int] = {}
        self.worker_thread_ids: set[int] = set()
        self._lock = threading.Lock()

    def fetch_hydration(
        self,
        *,
        listing_id: str,
        provider_symbol: str,
        start: date,
        end: date,
    ) -> HydrationEvidence:
        with self._lock:
            self.worker_thread_ids.add(threading.get_ident())
            call = self.hydration_calls.get(provider_symbol, 0) + 1
            self.hydration_calls[provider_symbol] = call
        if provider_symbol in {"AAPL", "MSFT"} and call <= self.provider_wide_failures:
            raise ProviderFetchError("data.rate_limited", "fixture 429", retryable=True)
        selected = tuple(session for session in self.sessions if start <= session <= end)
        rows = tuple(
            {
                "session_date": session.isoformat(),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000,
                "split_ratio": 0.0,
                "cash_dividend": 0.0,
                "capital_gain": 0.0,
            }
            for session in selected
        )
        adjusted_sessions = selected[1:] if provider_symbol == "FAIL" else selected
        return HydrationEvidence(
            daily_rows=rows,
            actions=(),
            adjusted_closes=tuple(
                ProviderAdjustedClosePoint(
                    listing_id=listing_id,
                    provider=self.name,
                    session_date=session,
                    adjusted_close=100.0,
                )
                for session in adjusted_sessions
            ),
        )

    def fetch_adjusted_close_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ):
        self.fallback_adjusted_calls[provider_symbol] = (
            self.fallback_adjusted_calls.get(provider_symbol, 0) + 1
        )
        return super().fetch_adjusted_close_history(
            listing_id=listing_id,
            provider_symbol=provider_symbol,
            start=start,
            end=end,
        )


class ShortHistoryHydrationProvider(HydrationFixtureProvider):
    def fetch_hydration(self, **kwargs) -> HydrationEvidence:
        evidence = super().fetch_hydration(**kwargs)
        if kwargs["provider_symbol"] != "SHORT":
            return evidence
        cutoff = date(2021, 1, 4)
        return HydrationEvidence(
            daily_rows=tuple(
                row
                for row in evidence.daily_rows
                if date.fromisoformat(str(row["session_date"])) >= cutoff
            ),
            actions=evidence.actions,
            adjusted_closes=tuple(
                point for point in evidence.adjusted_closes if point.session_date >= cutoff
            ),
        )


def _bootstrap(
    symbols: Sequence[str] = ("AAPL", "BAD", "FAIL", "MSFT"),
) -> CurrentUniverseBootstrap:
    candidates = tuple(
        CurrentUniverseCandidate(
            symbol=symbol,
            provider_symbol=symbol,
            source_memberships=(
                CandidateMembershipEvidence(index="NASDAQ100", company_name=symbol),
            ),
        )
        for symbol in symbols
    )
    provisional = CurrentUniverseCandidateManifest(
        created_at=OBSERVED_AT,
        as_of_timestamp=OBSERVED_AT,
        construction_rule=ORIGINAL_RESEARCH_WHITELIST_STANDARD.construction_rule,
        data_validity_class=ORIGINAL_RESEARCH_WHITELIST_STANDARD.data_validity_class,
        sources=(),
        candidates=candidates,
        content_hash="",
    )
    document = candidate_manifest_document(provisional)
    document.pop("content_hash")
    manifest = CurrentUniverseCandidateManifest(
        created_at=provisional.created_at,
        as_of_timestamp=provisional.as_of_timestamp,
        construction_rule=provisional.construction_rule,
        data_validity_class=provisional.data_validity_class,
        sources=provisional.sources,
        candidates=provisional.candidates,
        content_hash=sha256(
            json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    return CurrentUniverseBootstrap(
        source_manifest=manifest,
        standard=ORIGINAL_RESEARCH_WHITELIST_STANDARD,
    )


def test_full_onboarding_resumes_without_refetching_and_freezes_quality_manifest(
    tmp_path, monkeypatch
) -> None:
    schedule = materialize_calendar_schedule(
        ("XNAS",),
        start=date(2016, 7, 31),
        end=AS_OF,
        as_of_timestamp=OBSERVED_AT,
    )
    sessions = tuple(row["session_date"] for row in schedule.to_pylist())
    provider = FixtureProvider(sessions)
    store = MarketDataRepository(tmp_path / "workspace")
    onboarding = CurrentUniverseOnboarding(
        store=store,
        bootstrap=_bootstrap(),
        profile_path=PROFILE,
        provider=provider,
        as_of_session=AS_OF,
    )
    # The optional observer is told each unit transition after the store recorded it:
    # one announcement per transition, in order, never for a unit that was not advanced.
    announced: list[ListingUnitObservation] = []
    onboarding.listing_observer = announced.append

    first = onboarding.run(observed_at=OBSERVED_AT, work_budget=1)

    assert first.status is CurrentUniverseOnboardingStatus.RUNNING
    assert provider.daily_calls == {"AAPL": 1}
    assert [(v.symbol, v.state, v.origin) for v in announced] == [
        ("AAPL", "RAW_READY", "ACQUIRED"),
        ("AAPL", "QUALITY_ELIGIBLE", "LOCAL"),
        ("AAPL", "FEATURE_READY", "LOCAL"),
    ]
    assert announced[0].raw_through == AS_OF
    assert announced[0].observed_at == OBSERVED_AT
    assert all(v.failure_code is None and not v.tail_acquired for v in announced)
    # One run loads the listing once: every transition it announces departs from that
    # loaded state (PENDING here); the observer follows the later ones itself.
    assert [v.run_start_state for v in announced] == ["PENDING"] * 3
    persisted = store.resumable_current_universe_onboarding_input(
        market_profile_id="us-current-index-research"
    )
    assert persisted is not None
    _onboarding_id, document, history_start, persisted_as_of = persisted

    resumed_runner = CurrentUniverseOnboarding(
        store=store,
        bootstrap=bootstrap_from_candidate_manifest_document(document),
        profile_path=PROFILE,
        provider=provider,
        as_of_session=persisted_as_of,
        history_start=history_start,
    )
    # Before its first chunk a resumed runner reports the retained counts without
    # advancing anything: the denominator and what the earlier run left behind.
    retained = resumed_runner.retained_progress()
    assert (retained.candidates, retained.raw_ready, retained.feature_ready, retained.failed) == (
        4,
        1,
        1,
        0,
    )
    assert provider.daily_calls == {"AAPL": 1}
    announced.clear()
    resumed_runner.listing_observer = announced.append
    resumed = resumed_runner.run(observed_at=OBSERVED_AT)

    assert resumed.status is CurrentUniverseOnboardingStatus.COMPLETED
    assert [(v.symbol, v.state, v.origin) for v in announced] == [
        ("BAD", "RAW_READY", "ACQUIRED"),
        ("BAD", "QUALITY_INELIGIBLE", "LOCAL"),
        ("FAIL", "RAW_FAILED", None),
        ("MSFT", "RAW_READY", "ACQUIRED"),
        ("MSFT", "QUALITY_ELIGIBLE", "LOCAL"),
        ("MSFT", "FEATURE_READY", "LOCAL"),
    ], "AAPL, already admitted, is not announced again"
    assert set(announced[1].reasons) == {
        "maximum_consecutive_gap_exceeds_20:60",
        "missing_ratio_exceeds_0.02:60/2513",
    }
    assert announced[2].failure_code == "data.provider_fetch_failed"
    assert {v.run_start_state for v in announced} == {"PENDING"}
    assert resumed.research_manifest is not None
    assert tuple(item.symbol for item in resumed.research_manifest.listings) == ("AAPL", "MSFT")
    assert (
        store.current_quality_filtered_research_manifest(
            market_profile_id="us-current-index-research"
        )
        == resumed.research_manifest
    )
    assert provider.daily_calls["AAPL"] == 1
    assert provider.daily_calls["MSFT"] == 1
    assert resumed.feature_ready == 2
    assert resumed.failed == 2
    disclosure = store.latest_current_universe_onboarding_disclosure(
        market_profile_id="us-current-index-research"
    )
    assert disclosure is not None
    assert disclosure["candidate_listing_count"] == 4
    assert disclosure["state_counts"] == {
        "FEATURE_READY": 2,
        "QUALITY_INELIGIBLE": 1,
        "RAW_FAILED": 1,
    }
    failure_reasons = disclosure["failure_reason_counts"]
    assert failure_reasons["data.provider_fetch_failed"] == 1
    assert failure_reasons["maximum_consecutive_gap_exceeds_20:60"] == 1
    assert failure_reasons["missing_ratio_exceeds_0.02:60/2513"] == 1
    admissions = store.current_universe_quality_admissions(resumed.onboarding_id)
    symbols_by_listing = {
        item.listing_id: item.symbol for item in onboarding.acquisition_manifest.listings
    }
    assert {
        symbols_by_listing[admission.listing_id]: admission.eligible for admission in admissions
    } == {
        "AAPL": True,
        "BAD": False,
        "MSFT": True,
    }
    aapl = resumed.research_manifest.listing_for_symbol("AAPL")
    assert aapl is not None
    assert (
        store.reusable_action_audit_receipt(
            resumed.research_manifest,
            listing_id=aapl.listing_id,
            provider="yfinance",
            requested_as_of=AS_OF,
            now=OBSERVED_AT,
        )
        is not None
    )

    progress = []
    maintenance = CurrentUniverseMaintenance(
        store=store,
        manifest=resumed.research_manifest,
        provider=provider,
        as_of_session=AS_OF,
        progress_sink=progress.append,
    )
    read_listings = store.current_universe_maintenance_listings
    listing_reads = 0

    def counted_listings(identity):
        nonlocal listing_reads
        listing_reads += 1
        return read_listings(identity)

    with monkeypatch.context() as reads:
        reads.setattr(store, "current_universe_maintenance_listings", counted_listings)
        partial_maintenance = maintenance.run(observed_at=OBSERVED_AT, work_budget=1)
    # Progress counts its units in the engine: whole reads only where an outcome
    # is built, the run's opening report and its close.
    assert listing_reads == 2
    assert [item.completed_units for item in progress] == [0, 1, 1, 1]
    assert partial_maintenance.status is CurrentUniverseMaintenanceStatus.RUNNING
    assert provider.daily_calls["AAPL"] == 2

    completed_maintenance = CurrentUniverseMaintenance(
        store=store,
        manifest=resumed.research_manifest,
        provider=provider,
        as_of_session=AS_OF,
    ).run(observed_at=OBSERVED_AT)
    assert completed_maintenance.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert completed_maintenance.updated == 2
    assert completed_maintenance.failed == 0
    assert provider.daily_calls == {"AAPL": 2, "MSFT": 2, "BAD": 1, "FAIL": 1}
    assert (
        CurrentUniverseMaintenance(
            store=store,
            manifest=resumed.research_manifest,
            provider=provider,
            as_of_session=AS_OF,
        ).run(observed_at=OBSERVED_AT)
        == completed_maintenance
    )

    def no_candidate_verification(*_args, **_kwargs):
        raise AssertionError("receipt validation ran without a rebind candidate")

    with monkeypatch.context() as absent:
        absent.setattr(store, "reusable_action_audit_receipt", no_candidate_verification)
        assert (
            store.bind_available_action_audit_receipts_to_manifest(
                resumed.research_manifest,
                requested_as_of=AS_OF + timedelta(days=1),
                now=OBSERVED_AT,
            )
            == ()
        )

    gateway_manifest = replace(
        resumed.research_manifest,
        manifest_id=f"{resumed.research_manifest.manifest_id}:feature-input-admitted",
        listings=(aapl,),
        revision_sha256="f" * 64,
    )
    store.bootstrap(gateway_manifest)
    connection = store._connect()
    try:
        connection.execute(
            """
            INSERT INTO feature_input_admission VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                "a" * 64,
                resumed.research_manifest.revision_sha256,
                gateway_manifest.manifest_id,
                gateway_manifest.revision_sha256,
                "b" * 64,
                "c" * 64,
                AS_OF,
                OBSERVED_AT.replace(tzinfo=None),
                1,
                1,
                "{}",
                OBSERVED_AT.replace(tzinfo=None),
            ],
        )
    finally:
        connection.close()

    rebound = store.bind_available_action_audit_receipts_to_manifest(
        gateway_manifest,
        requested_as_of=AS_OF,
        now=OBSERVED_AT,
    )
    assert rebound == (aapl.listing_id,)
    assert (
        store.reusable_action_audit_receipt(
            gateway_manifest,
            listing_id=aapl.listing_id,
            provider="yfinance",
            requested_as_of=AS_OF,
            now=OBSERVED_AT,
        )
        is not None
    )
    # The coordinator calls this once per cycle. One call took every receipt
    # of another manifest that proves its own range -- the full audit's and
    # the maintenance run's rolling one alike, each judged over its range --
    # so the child answers the full-history question from the copy, and the
    # calls after it find nothing unheld: no evidence digest, nothing bound.
    reader = store._connect(read_only=True)
    try:
        ranges = {
            (row[0], row[1])
            for row in reader.execute(
                """SELECT history_start, history_end FROM action_audit_receipt
                   WHERE listing_id = ? AND requested_as_of = ? AND manifest_revision = ?""",
                [aapl.listing_id, AS_OF, gateway_manifest.revision_sha256],
            ).fetchall()
        }
    finally:
        reader.close()
    assert len(ranges) == 2 and all(end == AS_OF for _start, end in ranges)
    validated: list[str] = []
    original_evidence = store._action_audit_evidence

    def counting_evidence(connection, **kwargs):
        validated.append(kwargs["listing_id"])
        return original_evidence(connection, **kwargs)

    monkeypatch.setattr(store, "_action_audit_evidence", counting_evidence)
    for _ in range(2):
        assert (
            store.bind_available_action_audit_receipts_to_manifest(
                gateway_manifest, requested_as_of=AS_OF, now=OBSERVED_AT
            )
            == ()
        )
    assert validated == []
    monkeypatch.setattr(store, "_action_audit_evidence", original_evidence)
    assert (
        store.current_quality_filtered_research_manifest(
            market_profile_id="us-current-index-research"
        )
        == gateway_manifest
    )

    terminal_manifest = replace(
        gateway_manifest,
        manifest_id=f"{gateway_manifest.manifest_id}:next-admission",
        revision_sha256="e" * 64,
    )
    store.bootstrap(terminal_manifest)
    connection = store._connect()
    try:
        connection.execute(
            """
            INSERT INTO feature_input_admission VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                "d" * 64,
                gateway_manifest.revision_sha256,
                terminal_manifest.manifest_id,
                terminal_manifest.revision_sha256,
                "b" * 64,
                "c" * 64,
                AS_OF,
                (OBSERVED_AT + timedelta(minutes=1)).replace(tzinfo=None),
                1,
                0,
                "{}",
                (OBSERVED_AT + timedelta(minutes=1)).replace(tzinfo=None),
            ],
        )
    finally:
        connection.close()
    store.readiness.save(
        market_profile_id="us-current-index-research",
        status="FEATURE_BUILDING",
        active_manifest_id=terminal_manifest.manifest_id,
        active_manifest_revision=terminal_manifest.revision_sha256,
        active_membership_fingerprint=terminal_manifest.membership_fingerprint,
        active_candidate_manifest_document=None,
        pending_membership_fingerprint=None,
        pending_candidate_manifest_document=None,
        last_checked_at=OBSERVED_AT,
        last_changed_at=OBSERVED_AT,
        failure_code=None,
        observed_at=OBSERVED_AT + timedelta(minutes=1),
    )
    assert (
        store.current_quality_filtered_research_manifest(
            market_profile_id="us-current-index-research"
        )
        == terminal_manifest
    )


def test_membership_revision_hydrates_only_additions_and_reuses_retained_history(
    tmp_path,
) -> None:
    schedule = materialize_calendar_schedule(
        ("XNAS",),
        start=date(2016, 7, 31),
        end=AS_OF,
        as_of_timestamp=OBSERVED_AT,
    )
    sessions = tuple(row["session_date"] for row in schedule.to_pylist())
    store = MarketDataRepository(tmp_path / "workspace")
    initial_provider = HydrationFixtureProvider(sessions, provider_wide_failures=0)
    initial = CurrentUniverseOnboarding(
        store=store,
        bootstrap=_bootstrap(("AAPL", "REM")),
        profile_path=PROFILE,
        provider=initial_provider,
        as_of_session=AS_OF,
    ).run(observed_at=OBSERVED_AT)
    assert initial.status is CurrentUniverseOnboardingStatus.COMPLETED
    assert initial.research_manifest is not None
    retained = initial.research_manifest.listing_for_symbol("AAPL")
    removed = initial.research_manifest.listing_for_symbol("REM")
    assert retained is not None
    assert removed is not None

    revision_provider = HydrationFixtureProvider(sessions, provider_wide_failures=0)
    announced: list[ListingUnitObservation] = []
    revision = CurrentUniverseOnboarding(
        store=store,
        bootstrap=_bootstrap(("AAPL", "MSFT")),
        profile_path=PROFILE,
        provider=revision_provider,
        as_of_session=AS_OF,
        history_start=date(2016, 7, 31),
        retained_listing_ids=(retained.listing_id,),
    )
    revision.listing_observer = announced.append
    revised = revision.run(observed_at=OBSERVED_AT + timedelta(minutes=1))

    assert revised.status is CurrentUniverseOnboardingStatus.COMPLETED
    # The observer distinguishes what was evidenced: the retained listing reused its
    # durable history (no tail was needed, nothing fetched), the addition was acquired.
    assert [(v.symbol, v.state, v.origin, v.tail_acquired) for v in announced] == [
        ("MSFT", "RAW_READY", "ACQUIRED", False),
        ("MSFT", "QUALITY_ELIGIBLE", "LOCAL", False),
        ("MSFT", "FEATURE_READY", "LOCAL", False),
        ("AAPL", "RAW_READY", "RETAINED", False),
        ("AAPL", "QUALITY_ELIGIBLE", "LOCAL", False),
        ("AAPL", "FEATURE_READY", "LOCAL", False),
    ]
    # retained rows are admitted PENDING, so the retained listing departs from PENDING too
    assert {v.run_start_state for v in announced} == {"PENDING"}
    assert revised.research_manifest is not None
    disclosure = store.latest_current_universe_onboarding_disclosure(
        market_profile_id="us-current-index-research"
    )
    assert disclosure is not None
    assert disclosure["failure_reason_counts"] == {}
    assert tuple(value.symbol for value in revised.research_manifest.listings) == (
        "AAPL",
        "MSFT",
    ), disclosure
    assert revision_provider.hydration_calls == {"MSFT": 1}
    assert revision_provider.daily_calls == {}
    assert "REM" not in revision_provider.hydration_calls


@pytest.mark.parametrize("anchor_state", ["verified", "missing", "historical_bytes_changed"])
def test_same_session_membership_uses_its_unchanged_full_anchor_and_real_rolling_audit(
    tmp_path, anchor_state
):
    """behaviour: retained history needs both recorded audit scopes, without a new full audit."""
    from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload

    old_symbols = tuple(f"OLD{i:02d}" for i in range(10))
    target = date(2026, 8, 3)
    daily_at = datetime(2026, 8, 3, 23, tzinfo=UTC)
    membership_at = daily_at + timedelta(minutes=6)
    history_start = date(2016, 7, 31)
    schedule = materialize_calendar_schedule(
        ("XNAS",), start=history_start, end=target, as_of_timestamp=daily_at
    )
    sessions = tuple(row["session_date"] for row in schedule.to_pylist())
    store = MarketDataRepository(tmp_path / "workspace")
    initial_provider = HydrationFixtureProvider(sessions, provider_wide_failures=0)
    initial = CurrentUniverseOnboarding(
        store=store,
        bootstrap=_bootstrap(old_symbols),
        profile_path=PROFILE,
        provider=initial_provider,
        as_of_session=AS_OF,
        history_start=history_start,
    )
    if anchor_state == "missing":
        # Real ingestion alone supplies no authority to invent a full-history audit.
        manifest = initial.acquisition_manifest
        store.bootstrap(manifest)
        for symbol in old_symbols:
            payload = initial_provider.fetch_daily((symbol,), start=history_start, end=AS_OF)
            store.apply_validated_batch(
                manifest,
                sanitize_payload(manifest, initial_provider.name, payload, (symbol,)),
                ingestion_id=f"unaudited-history:{symbol}",
                observed_at=OBSERVED_AT,
            )
    else:
        prepared = initial.run(observed_at=OBSERVED_AT)
        assert prepared.status is CurrentUniverseOnboardingStatus.COMPLETED
        manifest = prepared.research_manifest
        assert manifest is not None
        assert {
            unit.state for unit in store.current_universe_onboarding_listings(initial.onboarding_id)
        } == {"FEATURE_READY"}
    retained_ids = tuple(listing.listing_id for listing in manifest.listings)
    anchors = store.latest_action_audit_receipts(retained_ids, provider=initial_provider.name)
    assert len(anchors) == (0 if anchor_state == "missing" else 10)
    if anchor_state == "missing":
        # This deliberately unaudited acquisition cannot enter normal maintenance.
        # Its real next-day ingestion and bounded audit still cannot replace the
        # missing historical anchor or admit the entire retained history.
        rolling_start = target - timedelta(days=45)
        for listing in manifest.listings:
            payload = initial_provider.fetch_daily(
                (listing.symbol,), start=AS_OF + timedelta(days=1), end=target
            )
            store.apply_validated_batch(
                manifest,
                sanitize_payload(manifest, initial_provider.name, payload, (listing.symbol,)),
                ingestion_id=f"unaudited-tail:{listing.symbol}",
                observed_at=daily_at,
            )
            evidence = initial_provider.fetch_hydration(
                listing_id=listing.listing_id,
                provider_symbol=listing.provider_symbol,
                start=rolling_start,
                end=target,
            )
            store.complete_action_audit(
                manifest,
                listing_id=listing.listing_id,
                provider=initial_provider.name,
                observed_actions=evidence.actions,
                observed_adjusted_closes=evidence.adjusted_closes,
                history_start=rolling_start,
                history_end=target,
                requested_as_of=target,
                observed_at=daily_at,
            )
    else:
        daily = CurrentUniverseMaintenance(
            store=store,
            manifest=manifest,
            provider=initial_provider,
            as_of_session=target,
        ).run(observed_at=daily_at)
        assert daily.status is CurrentUniverseMaintenanceStatus.COMPLETED
        assert daily.updated == 10 and daily.failed == 0
    rolling = store.latest_action_audit_receipts(
        retained_ids, provider=initial_provider.name, requested_as_of=target
    )
    assert len(rolling) == 10
    assert all(
        receipt.history_start > sessions[0] and receipt.observed_at == daily_at.replace(tzinfo=None)
        for receipt in rolling.values()
    )
    if anchor_state == "historical_bytes_changed":
        listing = manifest.listings[0]
        old_bar = store.raw_bars(listing.listing_id, through=AS_OF)[0]
        row = {key: getattr(old_bar, key) for key in ("open", "high", "low", "close", "volume")}
        row.update(session_date=old_bar.session_date.isoformat(), volume=old_bar.volume + 1)
        counts = store.apply_validated_batch(
            manifest,
            sanitize_payload(
                manifest, initial_provider.name, {listing.symbol: (row,)}, (listing.symbol,)
            ),
            ingestion_id="historical-correction",
            observed_at=daily_at + timedelta(minutes=1),
        )
        assert counts["corrected"] == 1
    provider = HydrationFixtureProvider(sessions, provider_wide_failures=0)
    revised = CurrentUniverseOnboarding(
        store=store,
        bootstrap=_bootstrap((*old_symbols, "NEW")),
        profile_path=PROFILE,
        provider=provider,
        as_of_session=target,
        history_start=history_start,
        retained_listing_ids=retained_ids,
    )
    result = revised.run(observed_at=membership_at)
    assert result.status is CurrentUniverseOnboardingStatus.COMPLETED
    units = {
        unit.symbol: unit
        for unit in store.current_universe_onboarding_listings(revised.onboarding_id)
    }
    rejected = (
        set(old_symbols)
        if anchor_state == "missing"
        else {manifest.listings[0].symbol}
        if anchor_state == "historical_bytes_changed"
        else set()
    )
    assert {symbol for symbol, unit in units.items() if unit.state == "AUDIT_FAILED"} == rejected
    assert all(
        unit.state == "FEATURE_READY" for symbol, unit in units.items() if symbol not in rejected
    )
    assert all(
        units[symbol].failure_code == "data.retained_action_evidence_not_reusable"
        for symbol in rejected
    )
    assert provider.hydration_calls == {"NEW": 1} and provider.daily_calls == {}
    # Successful same-session admission retains the real daily audit, never a
    # fabricated current full receipt: the acquisition-budget answer stays None.
    assert store.latest_action_audit_receipts(retained_ids, provider=provider.name) == anchors
    for listing_id in retained_ids:
        assert (
            store.reusable_action_audit_receipt(
                revised.acquisition_manifest,
                listing_id=listing_id,
                provider=provider.name,
                requested_as_of=target,
                now=membership_at,
            )
            is None
        )


def test_admission_cache_is_set_only_after_complete_receipt_rebinding(tmp_path, monkeypatch):

    class AfterAdmission(RuntimeError):
        pass

    store = MarketDataRepository(tmp_path / "workspace")
    provider = HydrationFixtureProvider((), provider_wide_failures=0)
    options = dict(
        store=store,
        bootstrap=_bootstrap(("AAPL",)),
        profile_path=PROFILE,
        provider=provider,
        as_of_session=AS_OF,
    )
    initial = CurrentUniverseOnboarding(**options)
    retained = (initial.acquisition_manifest.listings[0].listing_id,)
    runner = CurrentUniverseOnboarding(**options, retained_listing_ids=retained)
    calls = []
    replies = iter((OSError("receipt read interrupted"), (), retained))

    def rebind(*args, **kwargs):
        calls.append("rebind")
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def after_admission(*args, **kwargs):
        raise AfterAdmission

    monkeypatch.setattr(
        store, "admit_current_universe_onboarding", lambda *a, **k: calls.append("admit")
    )
    monkeypatch.setattr(store, "bind_available_action_audit_receipts_to_manifest", rebind)
    # Stop the actual run exactly after admission (the run record is the
    # first store read after it); no hydration or source work.
    monkeypatch.setattr(store, "current_universe_onboarding_run", after_admission)
    for failure, expected in (
        (OSError, False),
        (AfterAdmission, False),
        (AfterAdmission, True),
        (AfterAdmission, True),
    ):
        with pytest.raises(failure):
            runner.run(observed_at=OBSERVED_AT, work_budget=1)
        assert runner._admitted is expected
    assert calls == ["admit", "rebind"] * 3
    assert provider.hydration_calls == {}


def test_scoped_candidate_onboarding_preserves_scope_and_does_not_refetch_other_names(
    tmp_path,
) -> None:

    schedule = materialize_calendar_schedule(
        ("XNAS",), start=date(2016, 7, 31), end=AS_OF, as_of_timestamp=OBSERVED_AT
    )
    provider = FixtureProvider(tuple(row["session_date"] for row in schedule.to_pylist()))
    store = MarketDataRepository(tmp_path / "workspace")
    options = dict(
        store=store,
        bootstrap=_bootstrap(),
        profile_path=PROFILE,
        provider=provider,
        as_of_session=AS_OF,
    )
    full = CurrentUniverseOnboarding(**options)
    requested = tuple(
        sorted(
            item.listing_id
            for item in full.acquisition_manifest.listings
            if item.symbol in {"AAPL", "BAD"}
        )
    )
    assert len(requested) == 2
    runner = CurrentUniverseOnboarding(**options, requested_listing_ids=requested)
    assert runner.onboarding_id != full.onboarding_id
    assert (
        runner.run(observed_at=OBSERVED_AT, work_budget=1).status
        is CurrentUniverseOnboardingStatus.RUNNING
    )
    assert provider.daily_calls == {"AAPL": 1}
    assert (
        store.resumable_current_universe_onboarding_input(
            market_profile_id="us-current-index-research"
        )
        is None
    )  # This is not the full-source initialization that reader resumes.
    resumed = CurrentUniverseOnboarding(**options, requested_listing_ids=requested)
    assert resumed.onboarding_id == runner.onboarding_id
    result = resumed.run(observed_at=OBSERVED_AT)
    assert result.status is CurrentUniverseOnboardingStatus.COMPLETED
    assert {item.symbol for item in result.research_manifest.listings} == {"AAPL"}
    assert provider.daily_calls == {"AAPL": 1, "BAD": 1}
    assert {
        item.listing_id for item in store.current_universe_onboarding_listings(runner.onboarding_id)
    } == set(requested)
    assert resumed.run(observed_at=OBSERVED_AT).status is CurrentUniverseOnboardingStatus.COMPLETED
    assert provider.daily_calls == {"AAPL": 1, "BAD": 1}
    with pytest.raises(ValueError, match="different work scope"):
        store.admit_current_universe_onboarding(
            full.acquisition_manifest,
            onboarding_id=runner.onboarding_id,
            candidate_manifest_hash=full.bootstrap.source_manifest.content_hash,
            candidate_manifest_document=candidate_manifest_document(full.bootstrap.source_manifest),
            history_start=runner.history_start,
            as_of_session=AS_OF,
            calendar_by_listing_id=full._calendar_by_listing_id,
            observed_at=OBSERVED_AT,
        )
    short = tuple(
        item.listing_id for item in full.acquisition_manifest.listings if item.symbol == "BAD"
    )
    failed_scope = CurrentUniverseOnboarding(**options, requested_listing_ids=short)
    empty = failed_scope.run(observed_at=OBSERVED_AT + timedelta(minutes=1))
    assert (
        empty.status is CurrentUniverseOnboardingStatus.COMPLETED
        and empty.research_manifest is None
    )
    assert empty.feature_ready == 0 and empty.failed == 1
    disclosure = store.latest_current_universe_onboarding_disclosure(
        market_profile_id="us-current-index-research"
    )
    assert disclosure["work_scope"] == "CANDIDATE_REQUALIFICATION"
    assert disclosure["candidate_listing_count"] == 1 and disclosure["source_candidate_count"] == 4
    retried = CurrentUniverseOnboarding(
        **options, requested_listing_ids=short, retry_of=failed_scope.onboarding_id
    )
    assert retried.onboarding_id != failed_scope.onboarding_id
    before = dict(provider.daily_calls)
    assert (
        retried.run(observed_at=OBSERVED_AT + timedelta(days=1)).status
        is CurrentUniverseOnboardingStatus.COMPLETED
    )
    assert provider.daily_calls["BAD"] == before["BAD"] + 1
    assert provider.daily_calls["AAPL"] == before["AAPL"]
    with store._connect() as connection:
        connection.execute(
            "UPDATE current_universe_onboarding SET research_manifest_revision=? "
            "WHERE onboarding_id=?",
            ["f" * 64, runner.onboarding_id],
        )
    with pytest.raises(ValueError, match="completed_onboarding_binding_invalid"):
        resumed.run(observed_at=OBSERVED_AT)


def test_short_history_addition_is_hydrated_once_then_quarantined(tmp_path) -> None:
    schedule = materialize_calendar_schedule(
        ("XNAS",),
        start=date(2016, 7, 31),
        end=AS_OF,
        as_of_timestamp=OBSERVED_AT,
    )
    sessions = tuple(row["session_date"] for row in schedule.to_pylist())
    store = MarketDataRepository(tmp_path / "workspace")
    initial = CurrentUniverseOnboarding(
        store=store,
        bootstrap=_bootstrap(("AAPL",)),
        profile_path=PROFILE,
        provider=HydrationFixtureProvider(sessions, provider_wide_failures=0),
        as_of_session=AS_OF,
    ).run(observed_at=OBSERVED_AT)
    assert initial.research_manifest is not None
    retained = initial.research_manifest.listing_for_symbol("AAPL")
    assert retained is not None

    provider = ShortHistoryHydrationProvider(sessions, provider_wide_failures=0)
    revised = CurrentUniverseOnboarding(
        store=store,
        bootstrap=_bootstrap(("AAPL", "SHORT")),
        profile_path=PROFILE,
        provider=provider,
        as_of_session=AS_OF,
        history_start=date(2016, 7, 31),
        retained_listing_ids=(retained.listing_id,),
    ).run(observed_at=OBSERVED_AT + timedelta(minutes=1))

    assert revised.research_manifest is not None
    assert tuple(value.symbol for value in revised.research_manifest.listings) == ("AAPL",)
    assert provider.hydration_calls == {"SHORT": 1}
    disclosure = store.latest_current_universe_onboarding_disclosure(
        market_profile_id="us-current-index-research"
    )
    assert disclosure is not None
    assert disclosure["history_start"] == date(2016, 7, 31)
    assert disclosure["failure_reason_counts"]


def test_interactive_calendar_date_resolves_to_common_us_session() -> None:
    assert (
        _latest_common_us_session(on_or_before=date(2026, 8, 2), observed_at=OBSERVED_AT) == AS_OF
    )


def test_provider_wide_deferred_degrades_only_on_resume_and_keeps_successes(
    tmp_path, monkeypatch
) -> None:
    schedule = materialize_calendar_schedule(
        ("XNAS",),
        start=date(2016, 7, 31),
        end=AS_OF,
        as_of_timestamp=OBSERVED_AT,
    )
    sessions = tuple(row["session_date"] for row in schedule.to_pylist())
    provider = HydrationFixtureProvider(sessions, provider_wide_failures=2)
    store = MarketDataRepository(tmp_path / "workspace")
    calendar_calls = 0
    original_calendar = onboarding_module.materialize_calendar_schedule

    def counted_calendar(*args, **kwargs):
        nonlocal calendar_calls
        calendar_calls += 1
        return original_calendar(*args, **kwargs)

    monkeypatch.setattr(onboarding_module, "materialize_calendar_schedule", counted_calendar)
    main_thread_id = threading.get_ident()
    original_connect = store._connect

    def main_thread_connect(*, read_only: bool = False):
        assert threading.get_ident() == main_thread_id
        return original_connect(read_only=read_only)

    store._connect = main_thread_connect  # type: ignore[method-assign]
    onboarding = CurrentUniverseOnboarding(
        store=store,
        bootstrap=_bootstrap(),
        profile_path=PROFILE,
        provider=provider,
        as_of_session=AS_OF,
        hydration_chunk_size=25,
        hydration_workers=4,
    )

    first = onboarding.run(observed_at=OBSERVED_AT)
    assert first.status is CurrentUniverseOnboardingStatus.DEFERRED
    assert first.failure_code == "data.rate_limited"
    assert first.next_workers == 2
    assert provider.hydration_calls == {"AAPL": 1, "BAD": 1, "FAIL": 1, "MSFT": 1}

    early = onboarding.run(observed_at=OBSERVED_AT + timedelta(minutes=4))
    assert early == first
    assert provider.hydration_calls == {"AAPL": 1, "BAD": 1, "FAIL": 1, "MSFT": 1}

    assert first.retry_after_at is not None
    second = onboarding.run(observed_at=first.retry_after_at)
    assert second.status is CurrentUniverseOnboardingStatus.DEFERRED
    assert second.next_workers == 1
    assert provider.hydration_calls == {"AAPL": 2, "BAD": 1, "FAIL": 1, "MSFT": 2}

    assert second.retry_after_at is not None
    assert second.retry_after_at - first.retry_after_at >= timedelta(minutes=15)
    completed = onboarding.run(observed_at=second.retry_after_at)
    assert completed.status is CurrentUniverseOnboardingStatus.COMPLETED
    assert completed.feature_ready == 4
    assert provider.hydration_calls == {"AAPL": 3, "BAD": 1, "FAIL": 1, "MSFT": 3}
    assert provider.fallback_adjusted_calls == {"FAIL": 1}
    assert main_thread_id not in provider.worker_thread_ids
    assert store.hydration_worker_limit(onboarding.onboarding_id) == 1
    assert calendar_calls == 1


def test_audit_fallback_and_resumed_units_hold_nothing_across_the_provider(
    tmp_path, monkeypatch
) -> None:
    """Audit fallback and resumed units hold nothing across the provider."""

    from alphalattice.control.workspace_runtime.database import (
        live_workspace_connections,
        open_workspace_database,
    )

    schedule = materialize_calendar_schedule(
        ("XNAS",), start=date(2016, 7, 31), end=AS_OF, as_of_timestamp=OBSERVED_AT
    )
    sessions = tuple(row["session_date"] for row in schedule.to_pylist())
    store = MarketDataRepository(tmp_path / "workspace")
    blocked_in_fallback = threading.Event()
    checked = threading.Event()
    seen: list[object] = []

    class BlockingFallbackProvider(HydrationFixtureProvider):
        def fetch_adjusted_close_history(self, *, listing_id, provider_symbol, start, end):
            if provider_symbol == "FAIL":
                # The runner's main thread waits here; the checker looks meanwhile.
                blocked_in_fallback.set()
                assert checked.wait(timeout=30.0)
            return super().fetch_adjusted_close_history(
                listing_id=listing_id, provider_symbol=provider_symbol, start=start, end=end
            )

    provider = BlockingFallbackProvider(sessions, provider_wide_failures=0)

    def checker() -> None:
        if not blocked_in_fallback.wait(timeout=60.0):
            seen.append("fallback never reached")
            checked.set()
            return
        try:
            seen.append(("live", live_workspace_connections(store.path)))
            reader = open_workspace_database(store.path, read_only=True, wait_seconds=0.5)
            try:
                seen.append(("reader", live_workspace_connections(store.path)))
                seen.append(
                    (
                        "rows",
                        reader.execute(
                            "SELECT count(*) FROM current_universe_onboarding_listing"
                        ).fetchone()[0],
                    )
                )
            finally:
                reader.close()
            with (
                store.database.retain(read_only=False, wait_seconds=0.5),
                store.database.transaction() as connection,
            ):
                connection.execute("CREATE TABLE IF NOT EXISTS page_write (id INTEGER)")
                connection.execute("INSERT INTO page_write VALUES (1)")
            seen.append(("writer", "wrote"))
        except Exception as error:
            seen.append(("error", f"{type(error).__name__}: {error}"))
        finally:
            checked.set()

    watcher = threading.Thread(target=checker, daemon=True)
    watcher.start()
    onboarding = CurrentUniverseOnboarding(
        store=store,
        bootstrap=_bootstrap(),
        profile_path=PROFILE,
        provider=provider,
        as_of_session=AS_OF,
        hydration_chunk_size=25,
        hydration_workers=2,
    )
    completed = onboarding.run(observed_at=OBSERVED_AT)
    watcher.join(timeout=60.0)
    assert not watcher.is_alive()
    assert completed.status is CurrentUniverseOnboardingStatus.COMPLETED, completed
    assert provider.fallback_adjusted_calls == {"FAIL": 1}
    by_step = dict(item for item in seen if isinstance(item, tuple))
    assert "fallback never reached" not in seen and "error" not in by_step, seen
    # Nothing writable was held while the runner waited on the Provider: the
    # reader took its own read-only instance (not the guard on a writer's),
    # read the live rows, and a writer proceeded within its short wait.
    assert by_step["live"] is None or by_step["live"].read_only, by_step
    assert by_step["reader"].read_only and by_step["rows"] == 4, by_step
    assert by_step["writer"] == "wrote"
    # And nothing is left behind once the run returns.
    assert live_workspace_connections(store.path) is None


def test_resumed_quality_eligible_unit_audits_through_the_provider_holding_nothing(
    tmp_path,
) -> None:
    """A resumed quality-eligible unit releases its store instance while auditing through the
    provider."""

    from alphalattice.control.workspace_runtime.database import (
        live_workspace_connections,
        open_workspace_database,
    )

    schedule = materialize_calendar_schedule(
        ("XNAS",), start=date(2016, 7, 31), end=AS_OF, as_of_timestamp=OBSERVED_AT
    )
    sessions = tuple(row["session_date"] for row in schedule.to_pylist())
    store = MarketDataRepository(tmp_path / "workspace")
    blocked_in_fallback = threading.Event()
    checked = threading.Event()
    seen: dict[str, object] = {}

    class RateLimitedThenBlockingProvider(HydrationFixtureProvider):
        fallback_attempts = 0

        def fetch_adjusted_close_history(self, *, listing_id, provider_symbol, start, end):
            if provider_symbol == "FAIL":
                self.fallback_attempts += 1
                if self.fallback_attempts == 1:
                    raise ProviderFetchError("data.rate_limited", "fixture 429", retryable=True)
                blocked_in_fallback.set()
                assert checked.wait(timeout=30.0)
            return super().fetch_adjusted_close_history(
                listing_id=listing_id, provider_symbol=provider_symbol, start=start, end=end
            )

    provider = RateLimitedThenBlockingProvider(sessions, provider_wide_failures=0)

    def runner() -> CurrentUniverseOnboarding:
        return CurrentUniverseOnboarding(
            store=store,
            bootstrap=_bootstrap(),
            profile_path=PROFILE,
            provider=provider,
            as_of_session=AS_OF,
            hydration_chunk_size=25,
            hydration_workers=2,
        )

    first = runner()
    interrupted = first.run(observed_at=OBSERVED_AT)
    assert interrupted.status is CurrentUniverseOnboardingStatus.RUNNING
    states = {
        item.symbol: item.state
        for item in store.current_universe_onboarding_listings(first.onboarding_id)
    }
    assert states == {
        "AAPL": "FEATURE_READY",
        "BAD": "FEATURE_READY",
        "MSFT": "FEATURE_READY",
        "FAIL": "QUALITY_ELIGIBLE",
    }
    assert provider.hydration_calls == {"AAPL": 1, "BAD": 1, "FAIL": 1, "MSFT": 1}
    assert provider.fallback_attempts == 1 and provider.fallback_adjusted_calls == {}
    assert live_workspace_connections(store.path) is None  # the Provider-failure exit

    def checker() -> None:
        if not blocked_in_fallback.wait(timeout=60.0):
            seen["error"] = "fallback never reached"
            checked.set()
            return
        try:
            seen["live"] = live_workspace_connections(store.path)
            reader = open_workspace_database(store.path, read_only=True, wait_seconds=0.5)
            try:
                seen["reader"] = live_workspace_connections(store.path)
                seen["state"] = reader.execute(
                    "SELECT state FROM current_universe_onboarding_listing WHERE symbol = 'FAIL'"
                ).fetchone()[0]
            finally:
                reader.close()
            with (
                store.database.retain(read_only=False, wait_seconds=0.5),
                store.database.transaction() as connection,
            ):
                connection.execute("CREATE TABLE IF NOT EXISTS page_write (id INTEGER)")
            seen["writer"] = "wrote"
        except Exception as error:
            seen["error"] = f"{type(error).__name__}: {error}"
        finally:
            checked.set()

    watcher = threading.Thread(target=checker, daemon=True)
    watcher.start()
    resumed = runner()  # a fresh instance: nothing of the first run's memory
    assert resumed.onboarding_id == first.onboarding_id
    completed = resumed.run(observed_at=OBSERVED_AT + timedelta(minutes=20))
    watcher.join(timeout=60.0)
    assert not watcher.is_alive() and "error" not in seen, seen
    assert completed.status is CurrentUniverseOnboardingStatus.COMPLETED, completed
    assert completed.feature_ready == 4
    assert provider.hydration_calls == {"AAPL": 1, "BAD": 1, "FAIL": 1, "MSFT": 1}
    assert provider.fallback_attempts == 2 and provider.fallback_adjusted_calls == {"FAIL": 1}
    assert seen["live"] is None or seen["live"].read_only, seen
    assert seen["reader"].read_only and seen["state"] == "QUALITY_ELIGIBLE", seen
    assert seen["writer"] == "wrote"
    assert live_workspace_connections(store.path) is None


def test_rate_limit_stops_before_submitting_the_next_chunk(tmp_path) -> None:
    symbols = tuple(f"S{index:02d}" for index in range(26))

    class AlwaysRateLimited(HydrationFixtureProvider):
        def fetch_hydration(self, **kwargs):
            provider_symbol = str(kwargs["provider_symbol"])
            with self._lock:
                self.hydration_calls[provider_symbol] = (
                    self.hydration_calls.get(provider_symbol, 0) + 1
                )
            raise ProviderFetchError("data.rate_limited", "fixture 429", retryable=True)

    provider = AlwaysRateLimited((), provider_wide_failures=0)
    onboarding = CurrentUniverseOnboarding(
        store=MarketDataRepository(tmp_path / "workspace"),
        bootstrap=_bootstrap(symbols),
        profile_path=PROFILE,
        provider=provider,
        as_of_session=AS_OF,
        hydration_chunk_size=25,
        hydration_workers=4,
    )
    outcome = onboarding.run(observed_at=OBSERVED_AT)
    assert outcome.status is CurrentUniverseOnboardingStatus.DEFERRED
    assert set(provider.hydration_calls) == set(symbols[:25])
    assert symbols[25] not in provider.hydration_calls


def test_a_market_profile_is_read_by_its_typed_contract(tmp_path: Path) -> None:
    """regression (V275, V276): the profile reader took any mapping, so an unknown key or a
    provider written as a callable path built a manifest with its identity; the profile is a
    typed contract read by the one declaration loader, which refuses both and a key twice."""

    from alphalattice.foundation.market_data_ops.sources.manifest import read_market_profile

    root = Path(__file__).resolve().parents[2]
    text = (root / "config/market-profiles/us-current-index-research.yaml").read_text(
        encoding="utf-8"
    )
    good = tmp_path / "good.yaml"
    good.write_text(text, encoding="utf-8")
    assert read_market_profile(good).market_profile_id == "us-current-index-research"
    for bad in (
        text + "resolved_hash: not-a-hash\n",
        text.replace("provider: yfinance", "provider: pkg.module:Factory"),
        text + "market: US\n",
    ):
        path = tmp_path / "bad.yaml"
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match=r"market_data.market_profile_invalid"):
            read_market_profile(path)
