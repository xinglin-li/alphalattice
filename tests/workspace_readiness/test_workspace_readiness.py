"""Offline proof that manifest facts gate Front Desk creation deterministically."""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from alphalattice.control.data_platform.preflight import (
    evaluate_universe_spy_divergence_preflight,
)
from alphalattice.control.data_platform.readiness import (
    WorkspaceConsentAction,
    WorkspaceReadinessConsent,
    WorkspaceReadinessGate,
    WorkspaceReadinessStatus,
    daily_source_ready_at,
    source_verification_failed,
)
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.sources.contracts import ProviderAdjustedClosePoint
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
from alphalattice.foundation.research_foundation.storage.repository import (
    ResearchFoundationStateRepository,
)
from alphalattice.kernel.data.calendar import materialize_calendar_schedule

_SIMPLE_NAMESPACE = SimpleNamespace

PROFILE = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "market-profiles"
    / "us-current-index-research.yaml"
)
STARTED_AT = datetime(2026, 8, 3, 22, tzinfo=UTC)


class FixtureProvider:
    name = "yfinance"

    def __init__(self, sessions: Sequence[date]) -> None:
        self.sessions = tuple(sessions)
        self.daily_calls: list[str] = []

    def fetch_daily(self, symbols: Sequence[str], *, start: date, end: date):
        symbol = symbols[0]
        self.daily_calls.append(symbol)
        selected = tuple(session for session in self.sessions if start <= session <= end)
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


class ShortHistoryAdditionProvider(FixtureProvider):
    def _selected(self, symbol: str, start: date, end: date) -> tuple[date, ...]:
        selected = tuple(session for session in self.sessions if start <= session <= end)
        return selected[-30:] if symbol == "IPO" else selected

    def fetch_daily(self, symbols: Sequence[str], *, start: date, end: date):
        symbol = symbols[0]
        self.daily_calls.append(symbol)
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
                for session in self._selected(symbol, start, end)
            )
        }

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
            for session in self._selected(provider_symbol, start, end)
        )


def _bootstrap(symbols: tuple[str, ...], *, observed_at: datetime) -> CurrentUniverseBootstrap:
    candidates = tuple(
        CurrentUniverseCandidate(
            symbol=symbol,
            provider_symbol=symbol,
            source_memberships=(
                CandidateMembershipEvidence(index="NASDAQ100", company_name=symbol),
            ),
        )
        for symbol in sorted(symbols)
    )
    provisional = CurrentUniverseCandidateManifest(
        created_at=observed_at,
        as_of_timestamp=observed_at,
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


def test_readiness_gates_initialization_and_activates_only_completed_transitions(tmp_path) -> None:
    schedule = materialize_calendar_schedule(
        ("XNAS",),
        start=date(2016, 8, 3),
        end=STARTED_AT.date(),
        as_of_timestamp=STARTED_AT,
    )
    provider = FixtureProvider(tuple(row["session_date"] for row in schedule.to_pylist()))
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    gate = WorkspaceReadinessGate(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        profile_path=PROFILE,
        provider=provider,
    )
    initial = _bootstrap(("AAPL", "MSFT"), observed_at=STARTED_AT)

    assert (
        gate.assess(observed_at=STARTED_AT).status
        is WorkspaceReadinessStatus.INITIALIZATION_REQUIRED
    )
    onboarding = gate.start_approved_onboarding(
        WorkspaceReadinessConsent(
            consent_id=uuid4(),
            market_profile_id="us-current-index-research",
            action=WorkspaceConsentAction.INITIALIZE,
            approved_at=STARTED_AT,
        ),
        source_loader=lambda *, observed_at: initial,
    )
    assert (
        gate.assess(observed_at=STARTED_AT).status
        is WorkspaceReadinessStatus.ONBOARDING_IN_PROGRESS
    )
    completed = onboarding.runner.run(observed_at=STARTED_AT)
    ready = gate.complete_onboarding(onboarding, completed, observed_at=STARTED_AT)
    assert ready.status is WorkspaceReadinessStatus.FEATURE_BUILDING
    assert ready.manifest is not None
    assert ready.manifest.universe_membership_basis == "CURRENT_ACTIVE_SURVIVORS"
    assert not ready.manifest.is_point_in_time_historical
    old_manifest = ready.manifest

    calls = 0

    def unchanged_loader(*, observed_at: datetime) -> CurrentUniverseBootstrap:
        nonlocal calls
        calls += 1
        return _bootstrap(("AAPL", "MSFT"), observed_at=observed_at)

    gate.source_loader = unchanged_loader
    prior = market_data.readiness.load(gate.market_profile_id)
    assert prior is not None
    database_bytes = market_data.path.read_bytes()
    # The source is observed once per session: the next settled session finds
    # the check due, the same session does not, and a bare assessment never
    # reads the source either way.
    assert (
        gate.assess(observed_at=STARTED_AT + timedelta(days=1)).status
        is WorkspaceReadinessStatus.SOURCE_CHECK_REQUIRED
    )
    assert (
        gate.assess(observed_at=STARTED_AT + timedelta(hours=1)).status
        is WorkspaceReadinessStatus.FEATURE_BUILDING
    )
    assert (
        gate.assess(
            observed_at=STARTED_AT + timedelta(hours=1), target_session=date(2026, 8, 4)
        ).status
        is WorkspaceReadinessStatus.SOURCE_CHECK_REQUIRED
    )
    assert calls == 0
    assert market_data.path.read_bytes() == database_bytes
    assert market_data.readiness.load(gate.market_profile_id) == prior

    # An old workspace can claim READY before the required Panel exists.
    # The local check describes the missing work without migrating its state.
    legacy = gate._save(
        prior, status=WorkspaceReadinessStatus.RESEARCH_READY, observed_at=prior.updated_at
    )
    assert (
        gate.assess(observed_at=STARTED_AT + timedelta(hours=1)).status
        is WorkspaceReadinessStatus.FEATURE_BUILDING
    )
    assert market_data.readiness.load(gate.market_profile_id) == legacy
    gate._save(
        prior, status=WorkspaceReadinessStatus.FEATURE_BUILDING, observed_at=prior.updated_at
    )

    # ``before_fetch`` marks the source load, the command's only network edge,
    # so a caller can release what it retained: it is not called when no check
    # is due, and it is called once, before the loader, when one is.
    edges: list[str] = []

    def source_loader(*, observed_at):  # type: ignore[no-untyped-def]
        edges.append("load")
        return unchanged_loader(observed_at=observed_at)

    assert (
        gate.refresh_sources_if_due(
            observed_at=STARTED_AT + timedelta(hours=1),
            source_loader=source_loader,
            before_fetch=lambda: edges.append("before_fetch"),
        ).status
        is WorkspaceReadinessStatus.FEATURE_BUILDING
    )
    assert calls == 0
    assert edges == []
    assert (
        gate.refresh_sources_if_due(
            observed_at=STARTED_AT + timedelta(days=8),
            source_loader=source_loader,
            before_fetch=lambda: edges.append("before_fetch"),
        ).status
        is WorkspaceReadinessStatus.FEATURE_BUILDING
    )
    assert calls == 1
    assert edges == ["before_fetch", "load"]
    observations = market_data.universe_source_observations(gate.market_profile_id)
    assert len(observations) == 1
    assert observations[0].observed_at == STARTED_AT + timedelta(days=8)
    assert observations[0].previous_observed_at == STARTED_AT
    assert observations[0].first_eligible_session == date(2026, 8, 12)

    changed_at = STARTED_AT + timedelta(days=16)
    changed = _bootstrap(("AAPL", "NVDA"), observed_at=changed_at)
    pending = gate.refresh_sources_if_due(
        observed_at=changed_at, source_loader=lambda *, observed_at: changed
    )
    assert pending.status is WorkspaceReadinessStatus.MANIFEST_UPDATE_PENDING
    assert pending.proposal is not None
    assert pending.proposal.additions == ("NVDA",)
    assert pending.proposal.removals == ("MSFT",)
    assert market_data.load_universe_manifest(old_manifest.manifest_id) == old_manifest

    delta = gate.start_approved_onboarding(
        WorkspaceReadinessConsent(
            consent_id=uuid4(),
            market_profile_id="us-current-index-research",
            action=WorkspaceConsentAction.REFRESH_MANIFEST,
            approved_at=changed_at,
        )
    )
    delta_completed = delta.runner.run(observed_at=changed_at)
    new_ready = gate.complete_onboarding(delta, delta_completed, observed_at=changed_at)
    assert new_ready.status is WorkspaceReadinessStatus.FEATURE_BUILDING
    assert new_ready.manifest is not None
    assert tuple(listing.symbol for listing in new_ready.manifest.listings) == ("AAPL", "NVDA")
    assert new_ready.manifest.revision_sha256 != old_manifest.revision_sha256
    # A retained listing reuses unchanged durable rows, but it must establish
    # freshness and action-audit evidence for the new as-of/manifest contract.
    assert provider.daily_calls.count("AAPL") == 2
    assert provider.daily_calls.count("MSFT") == 1
    assert provider.daily_calls.count("NVDA") == 1
    transition = market_data.manifest_transition(pending.proposal.transition_id)
    assert transition.lifecycle == "ACTIVATED"
    assert transition.prior_manifest_revision == old_manifest.revision_sha256
    assert transition.next_manifest_revision == new_ready.manifest.revision_sha256
    foundation_state = ResearchFoundationStateRepository(market_data.database)
    feature_rebuild = foundation_state.feature_universe_rebuild_requirement(
        pending.proposal.transition_id
    )
    assert feature_rebuild.lifecycle == "REQUIRED"
    assert feature_rebuild.prior_manifest_revision == old_manifest.revision_sha256
    assert feature_rebuild.next_manifest_revision == new_ready.manifest.revision_sha256
    running = foundation_state.start_feature_universe_rebuild(
        pending.proposal.transition_id, observed_at=changed_at
    )
    assert running.lifecycle == "RUNNING"
    child_hashes = tuple(f"{index:x}" * 64 for index in range(1, 7))
    fulfilled = foundation_state.fulfill_feature_universe_rebuild(
        pending.proposal.transition_id,
        panel_snapshot_hash=child_hashes[0],
        factor_result_hash=child_hashes[1],
        factor_slate_hash=child_hashes[2],
        execution_outcome_hash=child_hashes[3],
        foundation_hash=child_hashes[4],
        revision_marker_hash=child_hashes[5],
        observed_at=changed_at,
    )
    assert fulfilled.lifecycle == "FULFILLED"
    assert fulfilled.foundation_hash == child_hashes[4]

    source_calls = []

    def failing_loader(*, observed_at: datetime) -> CurrentUniverseBootstrap:
        source_calls.append(observed_at)
        raise RuntimeError("fixture source unavailable")

    before = market_data.readiness.load(gate.market_profile_id)
    observations = market_data.universe_source_observations(gate.market_profile_id)
    failed_at = changed_at + timedelta(days=8)
    carried = gate.refresh_sources_if_due(observed_at=failed_at, source_loader=failing_loader)
    assert carried.status is WorkspaceReadinessStatus.FEATURE_BUILDING
    assert market_data.load_universe_manifest(new_ready.manifest.manifest_id) == new_ready.manifest
    state = market_data.readiness.load(gate.market_profile_id)
    assert state.last_checked_at == before.last_checked_at
    assert state.source_check_failed_at.replace(tzinfo=UTC) == failed_at
    assert source_verification_failed(state.last_checked_at, state.source_check_failed_at)
    assert market_data.universe_source_observations(gate.market_profile_id) == observations
    # A fresh gate uses the durable cooldown. A long-running operation does
    # not re-attempt this source for every listing after the cooldown elapses.
    fresh = replace(gate)
    assert (
        fresh.refresh_sources_if_due(
            observed_at=failed_at + timedelta(minutes=1), source_loader=failing_loader
        ).status
        is WorkspaceReadinessStatus.FEATURE_BUILDING
    )
    assert (
        fresh.refresh_sources_if_due(
            observed_at=failed_at + timedelta(minutes=20),
            source_loader=failing_loader,
            operation_started_at=failed_at,
        ).status
        is WorkspaceReadinessStatus.FEATURE_BUILDING
    )
    assert len(source_calls) == 1
    fresh.refresh_sources_if_due(
        observed_at=failed_at + timedelta(minutes=20), source_loader=failing_loader
    )
    assert len(source_calls) == 2
    approved = bootstrap_from_candidate_manifest_document(state.active_candidate_manifest_document)
    recovered_at = failed_at + timedelta(minutes=30)
    fresh.refresh_sources_if_due(
        observed_at=recovered_at,
        source_loader=lambda **_: _bootstrap(approved.candidate_symbols, observed_at=recovered_at),
    )
    recovered = market_data.readiness.load(gate.market_profile_id)
    assert recovered.last_checked_at.replace(tzinfo=UTC) == recovered_at
    assert not source_verification_failed(
        recovered.last_checked_at, recovered.source_check_failed_at
    )
    assert (
        fresh._carry_forward_source(
            replace(recovered, active_membership_fingerprint="f" * 64),
            observed_at=recovered_at + timedelta(days=1),
        )
        is None
    )
    assert market_data.readiness.load(gate.market_profile_id) == recovered


def test_readiness_failure_clock_reads_legacy_schema_and_survives_other_updates(tmp_path):
    market = MarketDataRepository(tmp_path)
    values = dict(
        market_profile_id="fixture",
        status="RESEARCH_READY",
        active_manifest_id=None,
        active_manifest_revision=None,
        active_membership_fingerprint=None,
        active_candidate_manifest_document=None,
        pending_membership_fingerprint=None,
        pending_candidate_manifest_document=None,
        last_checked_at=STARTED_AT,
        last_changed_at=STARTED_AT,
        failure_code=None,
        observed_at=STARTED_AT,
    )
    saved = market.readiness.save(**values)
    with market._connect() as connection:
        connection.execute("ALTER TABLE workspace_readiness DROP COLUMN source_check_failed_at")
    assert market.readiness.load("fixture") == saved
    with market._connect(read_only=True) as connection:
        assert "source_check_failed_at" not in {
            row[1]
            for row in connection.execute("PRAGMA table_info('workspace_readiness')").fetchall()
        }
    failed = market.readiness.save(**values, source_check_failed_at=STARTED_AT + timedelta(days=1))
    assert failed.source_check_failed_at.replace(tzinfo=UTC) == STARTED_AT + timedelta(days=1)
    assert market.readiness.save(**values).source_check_failed_at == failed.source_check_failed_at


def test_source_change_with_short_history_addition_keeps_active_set_and_avoids_rebuild(
    tmp_path,
) -> None:
    changed_at = STARTED_AT + timedelta(days=8)
    schedule = materialize_calendar_schedule(
        ("XNAS",),
        start=date(2016, 8, 3),
        end=changed_at.date(),
        as_of_timestamp=changed_at,
    )
    provider = ShortHistoryAdditionProvider(
        tuple(row["session_date"] for row in schedule.to_pylist())
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    gate = WorkspaceReadinessGate(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        profile_path=PROFILE,
        provider=provider,
    )
    initial = _bootstrap(("AAPL",), observed_at=STARTED_AT)
    gate.refresh_sources_if_due(
        observed_at=STARTED_AT, source_loader=lambda *, observed_at: initial
    )
    onboarding = gate.start_approved_onboarding(
        WorkspaceReadinessConsent(
            consent_id=uuid4(),
            market_profile_id="us-current-index-research",
            action=WorkspaceConsentAction.INITIALIZE,
            approved_at=STARTED_AT,
        ),
        source_loader=lambda *, observed_at: initial,
    )
    initial_outcome = onboarding.runner.run(observed_at=STARTED_AT)
    original = gate.complete_onboarding(onboarding, initial_outcome, observed_at=STARTED_AT)
    assert original.manifest is not None

    changed = _bootstrap(("AAPL", "IPO"), observed_at=changed_at)
    pending = gate.refresh_sources_if_due(
        observed_at=changed_at, source_loader=lambda *, observed_at: changed
    )
    assert pending.proposal is not None
    revision = gate.start_approved_onboarding(
        WorkspaceReadinessConsent(
            consent_id=uuid4(),
            market_profile_id="us-current-index-research",
            action=WorkspaceConsentAction.REFRESH_MANIFEST,
            approved_at=changed_at,
        )
    )
    outcome = revision.runner.run(observed_at=changed_at)
    for _ in range(5):
        if outcome.status.value != "running":
            break
        outcome = revision.runner.run(observed_at=changed_at)
    assert outcome.status.value == "completed", (
        outcome.failure_code,
        tuple(
            (value.listing_id, value.state, value.failure_code)
            for value in market_data.current_universe_onboarding_listings(
                revision.runner.onboarding_id
            )
        ),
    )
    result = gate.complete_onboarding(revision, outcome, observed_at=changed_at)

    assert result.manifest == original.manifest
    assert tuple(value.symbol for value in result.manifest.listings) == ("AAPL",)
    transition = market_data.manifest_transition(pending.proposal.transition_id)
    assert transition.lifecycle == "SOURCE_CHANGED_ACTIVE_SET_UNCHANGED"
    durable = market_data.readiness.load("us-current-index-research")
    assert durable is not None
    assert durable.status == WorkspaceReadinessStatus.FEATURE_BUILDING.value
    assert durable.pending_candidate_manifest_document is None
    with pytest.raises(ValueError, match="does not exist"):
        ResearchFoundationStateRepository(
            market_data.database
        ).feature_universe_rebuild_requirement(pending.proposal.transition_id)

    # The data-only fixture has no Feature publication; seed just the established
    # cohort contract here to exercise the recheck metadata/permission boundary.
    from alphalattice.foundation.market_data_ops.sources.membership import UniverseBootstrapRecord

    market_data.record_universe_bootstrap(
        UniverseBootstrapRecord(
            market_profile_id=gate.market_profile_id,
            t0_session=STARTED_AT.date(),
            history_start=date(2016, 8, 3),
            cohort_listing_ids=tuple(item.listing_id for item in original.manifest.listings),
            cohort_hash="",
            manifest_revision=original.manifest.revision_sha256,
            candidate_manifest_hash=initial.source_manifest.content_hash,
            qualification_policy_hash="1" * 64,
            feature_input_policy_hash="2" * 64,
            source_observed_at=STARTED_AT,
            admitted_at=STARTED_AT,
            panel_snapshot_hash="3" * 64,
            derivation="FIRST_QUALIFIED_PUBLICATION",
        )
    )
    assert (
        gate.candidate_data.plan(observed_at=changed_at, target_session=changed_at.date()) is None
    )
    due = changed_at + timedelta(days=1)
    scope = gate.candidate_data.plan(observed_at=due, target_session=due.date())
    assert scope is not None and len(scope.listing_ids) == 1
    calls_before = len(provider.daily_calls)
    with pytest.raises(ValueError, match="scope_invalid"):
        gate.candidate_data._context(
            replace(scope, listing_ids=(original.manifest.listings[0].listing_id,)),
            target_session=due.date(),
        )
    with pytest.raises(ValueError, match="history_invalid"):
        gate.candidate_data._onboarding(
            replace(scope, history_start=due.date()), target_session=due.date()
        )
    assert len(provider.daily_calls) == calls_before
    attempted, held = gate.candidate_data.advance(
        scope, target_session=due.date(), observed_at=due, work_budget=None
    )
    assert attempted.status.value == "completed" and attempted.research_manifest is None
    # A retry that admitted nothing prepares nothing: the day keeps its
    # research membership as its working manifest (the source parent is not
    # rebound, so the day's maintenance is not restarted under a new revision).
    assert held is None
    assert set(provider.daily_calls[calls_before:]) == {"IPO"}
    assert gate.candidate_data.plan(observed_at=due, target_session=due.date()) is None
    assert market_data.membership_events(gate.market_profile_id) == ()

    # The Data-update request carries this scope for every cycle of its Task:
    # a completed recheck answers from the owner's memory while the readiness
    # record it was validated against stands, without re-entering the runner
    # or the store's validation reads; a changed record sends the next call
    # back through the full validation.
    calls_after = len(provider.daily_calls)
    validated = 0
    original_context = gate.candidate_data._context

    def counting_context(*args, **kwargs):
        nonlocal validated
        validated += 1
        return original_context(*args, **kwargs)

    gate.candidate_data._context = counting_context  # type: ignore[method-assign]
    remembered = gate.candidate_data.completed_if_unchanged(scope, target_session=due.date())
    assert remembered is not None and remembered[1] == held
    again, held_again = gate.candidate_data.advance(
        scope, target_session=due.date(), observed_at=due, work_budget=None
    )
    assert again is attempted and held_again == held
    assert gate.candidate_data.completed(scope, target_session=due.date()) == held
    assert validated == 0 and len(provider.daily_calls) == calls_after
    durable_record = gate.candidate_data.readiness_record
    gate.candidate_data.readiness_record = lambda: replace(  # type: ignore[method-assign]
        durable_record(), updated_at=due + timedelta(minutes=1)
    )
    assert gate.candidate_data.completed_if_unchanged(scope, target_session=due.date()) is None
    gate.candidate_data.readiness_record = durable_record  # type: ignore[method-assign]
    assert gate.candidate_data.completed(scope, target_session=due.date()) == held
    assert validated == 1
    gate.candidate_data._context = original_context  # type: ignore[method-assign]

    # Acquisition/qualification can use the rolling ten-year window without
    # truncating the older research history already held in this workspace.
    next_year = due.replace(year=due.year + 1)
    later_source = _bootstrap(("AAPL", "IPO", "LATER"), observed_at=next_year)
    proposal = gate.refresh_sources_if_due(
        observed_at=next_year, source_loader=lambda **_: later_source
    )
    assert proposal.proposal is not None
    later = gate.start_approved_onboarding(
        WorkspaceReadinessConsent(
            consent_id=uuid4(),
            market_profile_id=gate.market_profile_id,
            action=WorkspaceConsentAction.REFRESH_MANIFEST,
            approved_at=next_year,
        )
    )
    assert later.runner.history_start.year == later.runner.as_of_session.year - 10
    assert market_data.manifest_raw_range(original.manifest)[0].year == 2016


class RecoveringHistoryProvider(ShortHistoryAdditionProvider):
    """IPO's history is short until the fixture says the provider caught up."""

    full_ipo = False

    def _selected(self, symbol: str, start: date, end: date) -> tuple[date, ...]:
        selected = tuple(session for session in self.sessions if start <= session <= end)
        return selected[-30:] if symbol == "IPO" and not self.full_ipo else selected


def test_feature_recheck_scope_resolves_only_across_the_request_own_raw_transition(tmp_path):
    """Feature recheck scope resolves only across the request's own raw transition."""

    from alphalattice.foundation.feature_engine.inputs.contracts import FeatureCandidateRecheck
    from alphalattice.foundation.market_data_ops.sources.manifest import (
        build_quality_filtered_research_manifest,
    )
    from alphalattice.foundation.market_data_ops.sources.membership import UniverseBootstrapRecord

    changed_at = STARTED_AT + timedelta(days=8)
    schedule = materialize_calendar_schedule(
        ("XNAS",), start=date(2016, 8, 3), end=changed_at.date(), as_of_timestamp=changed_at
    )
    provider = RecoveringHistoryProvider(tuple(row["session_date"] for row in schedule.to_pylist()))
    market_data = MarketDataRepository(tmp_path / "workspace")
    gate = WorkspaceReadinessGate(
        market_data=market_data,
        feature_state=FeatureStateRepository(market_data.database, market_data=market_data),
        panel_state=PanelStateRepository(market_data.database, market_data=market_data),
        profile_path=PROFILE,
        provider=provider,
    )
    initial = _bootstrap(("AAPL",), observed_at=STARTED_AT)
    gate.refresh_sources_if_due(
        observed_at=STARTED_AT, source_loader=lambda *, observed_at: initial
    )
    onboarding = gate.start_approved_onboarding(
        WorkspaceReadinessConsent(
            consent_id=uuid4(),
            market_profile_id="us-current-index-research",
            action=WorkspaceConsentAction.INITIALIZE,
            approved_at=STARTED_AT,
        ),
        source_loader=lambda *, observed_at: initial,
    )
    original = gate.complete_onboarding(
        onboarding, onboarding.runner.run(observed_at=STARTED_AT), observed_at=STARTED_AT
    )
    assert original.manifest is not None

    # The source gains a full-history name (NEWQ) and a short one (IPO): the
    # refresh admits NEWQ into the parent; IPO fails raw and waits a day.
    changed = _bootstrap(("AAPL", "IPO", "NEWQ"), observed_at=changed_at)
    pending = gate.refresh_sources_if_due(
        observed_at=changed_at, source_loader=lambda *, observed_at: changed
    )
    assert pending.proposal is not None
    revision = gate.start_approved_onboarding(
        WorkspaceReadinessConsent(
            consent_id=uuid4(),
            market_profile_id="us-current-index-research",
            action=WorkspaceConsentAction.REFRESH_MANIFEST,
            approved_at=changed_at,
        )
    )
    outcome = revision.runner.run(observed_at=changed_at)
    for _ in range(5):
        if outcome.status.value != "running":
            break
        outcome = revision.runner.run(observed_at=changed_at)
    assert outcome.status.value == "completed", outcome.failure_code
    gate.complete_onboarding(revision, outcome, observed_at=changed_at)
    market_data.record_universe_bootstrap(
        UniverseBootstrapRecord(
            market_profile_id=gate.market_profile_id,
            t0_session=STARTED_AT.date(),
            history_start=date(2016, 8, 3),
            cohort_listing_ids=tuple(item.listing_id for item in original.manifest.listings),
            cohort_hash="",
            manifest_revision=original.manifest.revision_sha256,
            candidate_manifest_hash=initial.source_manifest.content_hash,
            qualification_policy_hash="1" * 64,
            feature_input_policy_hash="2" * 64,
            source_observed_at=STARTED_AT,
            admitted_at=STARTED_AT,
            panel_snapshot_hash="3" * 64,
            derivation="FIRST_QUALIFIED_PUBLICATION",
        )
    )
    planned_parent = market_data.source_admission_manifest(market_profile_id=gate.market_profile_id)
    assert planned_parent is not None
    assert {item.symbol for item in planned_parent.listings} == {"AAPL", "NEWQ"}
    newq = next(item.listing_id for item in planned_parent.listings if item.symbol == "NEWQ")
    aapl = next(item.listing_id for item in planned_parent.listings if item.symbol == "AAPL")
    # The research membership the request binds: the parent's Feature-qualified
    # child holding AAPL alone (this data-only fixture has no Gateway to derive it).
    prior = build_quality_filtered_research_manifest(
        planned_parent,
        eligible_listing_ids=(aapl,),
        qualification_obligations=planned_parent.obligations_for_derivation(),
    )
    market_data.bootstrap(prior)
    scope = FeatureCandidateRecheck(planned_parent.revision_sha256, (newq,))
    # Planned and resolved against the parent of its day.
    assert (
        gate.resolve_feature_candidate_recheck(scope, prior_revision=prior.revision_sha256)
        == planned_parent
    )

    # The raw retry falls due; the provider has caught up, so it admits IPO
    # and the parent moves through the retry's own transition.
    due = changed_at + timedelta(days=1)
    raw_scope = gate.candidate_data.plan(observed_at=due, target_session=due.date())
    assert (
        raw_scope is not None
        and raw_scope.qualified_parent_revision == scope.parent_manifest_revision
    )
    provider.full_ipo = True
    attempted, merged = gate.candidate_data.advance(
        raw_scope, target_session=due.date(), observed_at=due, work_budget=None
    )
    assert attempted.status.value == "completed" and merged is not None
    assert {item.symbol for item in merged.listings} == {"AAPL", "NEWQ", "IPO"}
    assert market_data.source_admission_manifest(market_profile_id=gate.market_profile_id) == merged
    ipo = next(item.listing_id for item in merged.listings if item.symbol == "IPO")
    transition = gate.candidate_data.activated_transition(raw_scope, target_session=due.date())
    assert transition is not None and transition.additions == ("IPO",)
    assert transition.prior_manifest_revision == planned_parent.revision_sha256
    assert transition.next_manifest_revision == merged.revision_sha256

    # 1. The legal scope continues on the merged parent, across that transition.
    assert (
        gate.resolve_feature_candidate_recheck(
            scope,
            prior_revision=prior.revision_sha256,
            data_recheck=raw_scope,
            target_session=due.date(),
        )
        == merged
    )
    # Without the request's retry the parent has simply moved: refused by name.
    with pytest.raises(ValueError, match="candidate_recheck_source_changed"):
        gate.resolve_feature_candidate_recheck(scope, prior_revision=prior.revision_sha256)
    # 2. A wrong or stale planned parent is refused although the scope's
    # listings are a subset of the merged parent.
    for wrong_parent in (prior.revision_sha256, "f" * 64):
        with pytest.raises(ValueError, match="candidate_recheck_source_changed"):
            gate.resolve_feature_candidate_recheck(
                FeatureCandidateRecheck(wrong_parent, (newq,)),
                prior_revision=prior.revision_sha256,
                data_recheck=raw_scope,
                target_session=due.date(),
            )
    # 3. A scope naming a prior member, or the candidate the retry admitted,
    # violates the range rules on the successor as it did on the parent.
    with pytest.raises(ValueError, match="candidate_recheck_source_changed"):
        gate.resolve_feature_candidate_recheck(
            FeatureCandidateRecheck(planned_parent.revision_sha256, tuple(sorted((aapl, newq)))),
            prior_revision=prior.revision_sha256,
            data_recheck=raw_scope,
            target_session=due.date(),
        )
    with pytest.raises(ValueError, match="candidate_recheck_scope_invalid"):
        gate.resolve_feature_candidate_recheck(
            FeatureCandidateRecheck(planned_parent.revision_sha256, tuple(sorted((ipo, newq)))),
            prior_revision=prior.revision_sha256,
            data_recheck=raw_scope,
            target_session=due.date(),
        )
    # A transition this request did not record resolves nothing.
    with pytest.raises(ValueError, match="candidate_recheck_source_changed"):
        gate.resolve_feature_candidate_recheck(
            scope,
            prior_revision=prior.revision_sha256,
            data_recheck=replace(
                raw_scope, history_start=raw_scope.history_start + timedelta(days=1)
            ),
            target_session=due.date(),
        )
    with pytest.raises(ValueError, match="candidate_recheck_source_changed"):
        gate.resolve_feature_candidate_recheck(
            scope,
            prior_revision=prior.revision_sha256,
            data_recheck=raw_scope,
            target_session=due.date() + timedelta(days=1),
        )


def test_activated_source_preserves_existing_unusable_members_but_not_index_leavers(tmp_path):
    """Metadata boundary: raw qualification is not a second nominal membership owner."""
    from alphalattice.foundation.market_data_ops.runtime.universe_onboarding import (
        CurrentUniverseOnboardingStatus,
    )
    from alphalattice.foundation.market_data_ops.sources.manifest import (
        build_current_index_acquisition_manifest,
        build_quality_filtered_research_manifest,
    )
    from alphalattice.foundation.market_data_ops.sources.membership import UniverseBootstrapRecord

    profile = "us-current-index-research"
    market = MarketDataRepository(tmp_path / "membership")
    original = build_current_index_acquisition_manifest(
        PROFILE, _bootstrap(("AAPL", "MSFT"), observed_at=STARTED_AT)
    )
    market.bootstrap(original)
    market.record_universe_bootstrap(
        UniverseBootstrapRecord(
            market_profile_id=profile,
            t0_session=STARTED_AT.date(),
            history_start=date(2016, 8, 3),
            cohort_listing_ids=tuple(item.listing_id for item in original.listings),
            cohort_hash="",
            manifest_revision=original.revision_sha256,
            candidate_manifest_hash="1" * 64,
            qualification_policy_hash="2" * 64,
            feature_input_policy_hash="3" * 64,
            source_observed_at=STARTED_AT,
            admitted_at=STARTED_AT,
            panel_snapshot_hash="4" * 64,
            derivation="FIRST_QUALIFIED_PUBLICATION",
        )
    )
    now = STARTED_AT + timedelta(days=8)
    source = _bootstrap(("AAPL", "NEW"), observed_at=now)
    acquisition = build_current_index_acquisition_manifest(PROFILE, source)
    new = acquisition.listing_for_symbol("NEW")
    raw_qualified = build_quality_filtered_research_manifest(
        acquisition, eligible_listing_ids=(new.listing_id,), qualification_obligations=()
    )
    market.admit_current_universe_onboarding(
        acquisition,
        onboarding_id="source-refresh",
        candidate_manifest_hash=source.source_manifest.content_hash,
        candidate_manifest_document=candidate_manifest_document(source.source_manifest),
        history_start=date(2016, 8, 3),
        as_of_session=now.date(),
        calendar_by_listing_id={item.listing_id: "XNAS" for item in acquisition.listings},
        observed_at=now,
    )
    market.complete_current_universe_onboarding(
        onboarding_id="source-refresh",
        research_manifest=raw_qualified,
        quality_admission_hash="5" * 64,
        observed_at=now,
    )
    market.record_manifest_transition(
        transition_id="approved-refresh",
        market_profile_id=profile,
        prior_manifest_revision=original.revision_sha256,
        membership_fingerprint=acquisition.membership_fingerprint,
        additions=("NEW",),
        removals=("MSFT",),
        lifecycle="ONBOARDING_IN_PROGRESS",
        approved_at=now,
        created_at=now,
    )
    gate = WorkspaceReadinessGate(
        market_data=market,
        feature_state=FeatureStateRepository(market.database, market_data=market),
        panel_state=PanelStateRepository(market.database, market_data=market),
        profile_path=PROFILE,
        provider=FixtureProvider(()),
    )
    result = gate.complete_onboarding(
        SimpleNamespace(bootstrap=source, transition_id="approved-refresh"),
        SimpleNamespace(
            status=CurrentUniverseOnboardingStatus.COMPLETED, research_manifest=raw_qualified
        ),
        observed_at=now,
    )
    assert tuple(item.symbol for item in result.manifest.listings) == ("AAPL", "NEW")
    assert market.membership_events(profile) == ()  # Admission/clock still belongs to the caller.
    assert (
        market.current_quality_filtered_research_manifest(market_profile_id=profile)
        == result.manifest
    )

    # A pointer outside the activated root and its admission lineage still refuses.
    foreign = replace(result.manifest, manifest_id="foreign", revision_sha256="f" * 64)
    market.bootstrap(foreign)
    with market._connect() as connection:
        connection.execute(
            "UPDATE workspace_readiness SET active_manifest_id=?, active_manifest_revision=?",
            [foreign.manifest_id, foreign.revision_sha256],
        )
    with pytest.raises(ValueError, match="outside admission lineage"):
        market.current_quality_filtered_research_manifest(market_profile_id=profile)


def test_universe_spy_divergence_preflight_composes_qualified_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Universe and benchmark divergence preflight combines qualified inputs through the Host
    without changing either lane."""

    import alphalattice.control.data_platform.preflight as readiness

    formations = (date(2026, 1, 5), date(2026, 1, 6))
    axis = (date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8))
    open_paths = {
        "listing-a": (100.0, 100.0, 101.0, 103.02),
        "listing-b": (100.0, 100.0, 103.0, 107.12),
    }

    class StubStore:
        # Preflight holds one read-only database instance across every
        # listing's frame; the stubbed reads need no instance behind it.
        database = SimpleNamespace(retain=lambda *, read_only: nullcontext())

        def current_quality_filtered_research_manifest(self, *, market_profile_id: str):
            assert market_profile_id == "us-current-index-research"
            return SimpleNamespace(
                listings=(
                    SimpleNamespace(listing_id="listing-a"),
                    SimpleNamespace(listing_id="listing-b"),
                ),
                revision_sha256="r" * 64,
            )

        def projected_feature_frame(self, manifest, *, listing_id, through, start):
            del manifest, through, start
            rows = [
                {
                    "session_date": session,
                    "open_split_adjusted": open_paths[listing_id][index],
                    "cash_dividend": 0.0,
                }
                for index, session in enumerate(axis)
            ]
            return rows, f"raw-{listing_id}", f"action-{listing_id}"

    # Two seams, because preflight uses two owners: Market Data answers for the
    # frozen manifest, and the Feature state answers for the projected frame.
    # Substituting only the first left the real Feature repository reaching into
    # the double for private Market Data methods it does not have.
    store = StubStore()
    monkeypatch.setattr(readiness, "MarketDataRepository", lambda _root: store)
    monkeypatch.setattr(
        readiness, "FeatureStateRepository", lambda _root, *, market_data: market_data
    )

    # listing-a: +1% then +2%; listing-b: +3% then +4% -> equal weight 2%, 3%.
    spy_by_scenario = {
        "consistent": (0.018, 0.028),
        "breach": (0.018, -0.03),
    }
    scenario = {"name": "consistent"}

    def _stub_surface(*, workspace, formation_sessions):
        del workspace
        assert formation_sessions == formations
        return SimpleNamespace(
            simple_returns=spy_by_scenario[scenario["name"]],
            surface_hash="5" * 64,
        )

    monkeypatch.setattr(readiness, "build_portfolio_benchmark_surface", _stub_surface)

    consistent = evaluate_universe_spy_divergence_preflight(
        workspace=tmp_path, formation_sessions=formations
    )
    assert consistent.classification == "CONSISTENT"
    assert consistent.attribution == "UNATTRIBUTED"
    assert consistent.spy_source_identity == "5" * 64
    assert consistent.session_count == 2

    scenario["name"] = "breach"
    breached = evaluate_universe_spy_divergence_preflight(
        workspace=tmp_path, formation_sessions=formations
    )
    assert breached.classification == "IMPLAUSIBLE_DIVERGENCE_REVIEW_REQUIRED"
    (flagged,) = breached.flagged_sessions
    assert flagged.session_date == formations[1]
    assert flagged.universe_aggregate_return == pytest.approx(0.03)
    assert flagged.spy_return == pytest.approx(-0.03)
    # The Universe identity binds manifest revision and per-listing lineage, so
    # the same qualified inputs always name the same source.
    assert breached.universe_source_identity == consistent.universe_source_identity


def test_preflight_authority_failure_blocks_instead_of_reporting_no_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preflight authority failure blocks instead of reporting no verdict."""

    SimpleNamespace = _SIMPLE_NAMESPACE

    import alphalattice.control.data_platform.preflight as preflight
    from alphalattice.control.product_host.data_preparation.host import (
        PreFactorWorkspaceHost,
        ProductWorkspacePaths,
    )

    host = object.__new__(PreFactorWorkspaceHost)
    host.paths = ProductWorkspacePaths.at(tmp_path)

    formations = (date(2026, 1, 5), date(2026, 1, 6))
    monkeypatch.setattr(
        preflight, "routine_preflight_formation_scope", lambda **_kwargs: formations
    )

    def _raise_authority_failure(*_args, **_kwargs):
        raise preflight.DataTruthPreflightError("data_truth_preflight.research_manifest_missing")

    monkeypatch.setattr(preflight, "run_data_truth_preflight", _raise_authority_failure)
    import alphalattice.control.product_host.data_preparation.host as host_module

    monkeypatch.setattr(
        host_module, "routine_preflight_formation_scope", lambda **_kwargs: formations
    )
    monkeypatch.setattr(host_module, "run_data_truth_preflight", _raise_authority_failure)

    blocked = host._run_data_truth_preflight(
        market_profile_id="us-current-index-research",
        as_of_session=date(2026, 1, 8),
        now=datetime(2026, 1, 9, tzinfo=UTC),
    )
    assert blocked is not None
    assert blocked.blocks_research_ready
    assert blocked.blocking_failure_code == "data_truth_preflight.research_manifest_missing"

    # A workspace with nothing to compare is the one case that may have no
    # verdict, and it is a *different* exception type so the two cannot be
    # confused by accident.
    def _raise_scope_unavailable(**_kwargs):
        raise preflight.DataTruthScopeUnavailable(
            "data_truth_preflight.matured_session_axis_too_short"
        )

    monkeypatch.setattr(host_module, "routine_preflight_formation_scope", _raise_scope_unavailable)
    assert (
        host._run_data_truth_preflight(
            market_profile_id="us-current-index-research",
            as_of_session=date(2026, 1, 8),
            now=datetime(2026, 1, 9, tzinfo=UTC),
        )
        is None
    )
    del SimpleNamespace


class UnawareZone(tzinfo):
    def utcoffset(self, dt):
        return None


@pytest.mark.parametrize("zone", [None, UnawareZone()])
def test_daily_source_finality_refuses_an_unaware_close_by_name(zone):
    """a missing offset cannot be assigned UTC meaning at the finality boundary."""
    with pytest.raises(
        ValueError, match=r"^workspace_readiness\.session_close_timestamp_not_timezone_aware$"
    ):
        daily_source_ready_at(datetime(2026, 10, 2, 16, tzinfo=zone))


@pytest.mark.parametrize("zone", [UTC, timezone(timedelta(hours=-4))])
def test_daily_source_finality_keeps_an_aware_close(zone):
    """aware UTC and exchange-local instants retain the same two-hour finality rule."""
    close = datetime(2026, 10, 2, 16, tzinfo=zone)
    assert daily_source_ready_at(close) == close + timedelta(hours=2)
