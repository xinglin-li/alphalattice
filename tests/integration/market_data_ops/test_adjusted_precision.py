"""The data rule's precision: a re-sent stored adjusted close within it stands, past it restates."""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.market_data_ops.publication.projection import (
    provider_adjusted_close_evidence_hash,
)
from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
    CurrentUniverseMaintenance,
    CurrentUniverseMaintenanceStatus,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    build_quality_filtered_research_manifest,
)
from alphalattice.foundation.market_data_ops.sources.providers import HydrationEvidence
from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    ActionAuditScopeInsufficient,
    MarketDataRepository,
)
from tests.workspace_maintenance.acquisition_manifest import NOW, acquisition_manifest
from tests.workspace_maintenance.audit_workspace import audit_one

HISTORY = (date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30))
NEW = date(2026, 7, 31)


def _close(day: date) -> float:
    return 100.0 + day.day / 7.0


def _stored_evidence(store, receipt) -> str:  # type: ignore[no-untyped-def]
    return provider_adjusted_close_evidence_hash(
        tuple(
            point
            for point in store.provider_adjusted_closes("listing-aapl", through=receipt.history_end)
            if point.session_date >= receipt.history_start
        )
    )


def _store(tmp_path: Path):  # type: ignore[no-untyped-def]
    manifest = build_quality_filtered_research_manifest(
        acquisition_manifest(), eligible_listing_ids=("listing-aapl",)
    )
    store = MarketDataRepository(tmp_path / "workspace")
    store.bootstrap(manifest)
    bar = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000}

    def day(sessions: tuple[date, ...], adjusted: dict[date, float], offset: int, actions=()):  # type: ignore[no-untyped-def]
        store.apply_validated_batch(
            manifest,
            sanitize_payload(
                manifest,
                "fixture",
                {"AAPL": tuple({"session_date": item.isoformat()} | bar for item in sessions)},
                ("AAPL",),
            ),
            ingestion_id=f"day-{offset}",
            observed_at=NOW + timedelta(days=offset),
        )
        return audit_one(
            store,
            manifest,
            listing_id="listing-aapl",
            provider="fixture",
            actions=actions,
            adjusted_closes=tuple(
                ProviderAdjustedClosePoint("listing-aapl", "fixture", item, value)
                for item, value in adjusted.items()
            ),
            history_start=min(adjusted),
            history_end=max(adjusted),
            requested_as_of=max(adjusted),
            observed_at=NOW + timedelta(days=offset),
            adjusted_tolerance=1e-6,
        )

    day(HISTORY, {item: _close(item) for item in HISTORY}, 0)
    return store, day


@pytest.mark.parametrize(("scale", "stands"), [(1 + 5e-7, True), (1 + 2e-6, False)])
def test_a_resent_stored_close_within_the_precision_stands_and_past_it_restates(
    tmp_path: Path, scale: float, stands: bool
) -> None:
    """requirement: within the rule's precision a re-sent stored close stands; past it, the
    series restates."""
    store, day = _store(tmp_path)
    # The next day re-sends the window re-rounded, as a provider re-adjusting does, and the new
    # session; the raw bars are bit-equal.
    resent = {item: _close(item) * scale for item in HISTORY[1:]} | {NEW: _close(NEW)}
    receipt, counts = day((NEW,), resent, 1)
    revision = store.provider_adjusted_revision(receipt.receipt_hash)
    assert revision is not None
    stored = {
        p.session_date: p.adjusted_close
        for p in store.provider_adjusted_closes("listing-aapl", through=NEW)
    }
    if stands:
        assert receipt.provider_adjusted_close_evidence_hash == _stored_evidence(store, receipt)
        assert counts["adjusted_precision_kept"] == pytest.approx(scale - 1, rel=1e-3)
        assert revision.changed_return_sessions == (NEW,)
        assert stored == {item: _close(item) for item in (*HISTORY, NEW)}
    else:
        assert "adjusted_precision_kept" not in counts
        assert HISTORY[1] in revision.changed_return_sessions
        assert stored[HISTORY[2]] == _close(HISTORY[2]) * scale


ACTIONS = {
    "dividend": ((HISTORY[2], "CASH_DIVIDEND", 0.5),),
    "dividend on the new session": ((NEW, "CASH_DIVIDEND", 0.5),),
    "split on the new session": ((NEW, "SPLIT", 2.0),),
    "two dividends": ((HISTORY[1], "CASH_DIVIDEND", 0.5), (HISTORY[3], "CASH_DIVIDEND", 0.4)),
}


def _factor(day: date, kind: str, value: float) -> float:
    return 1 / value if kind == "SPLIT" else 1 - value / _close(max(h for h in HISTORY if h < day))


@pytest.mark.parametrize("window", [False, True])
@pytest.mark.parametrize("case", ACTIONS)
def test_a_back_adjustment_is_kept_in_the_adjustment_factor_form(tmp_path, case, window) -> None:
    """requirement: a back-adjustment is kept as each segment's stored closes times its factor,
    disclosed and read back; a rolling refresh that sees it reach before its scope escalates."""
    store, day = _store(tmp_path)
    actions, noise = ACTIONS[case], (3e-7, -4e-7, 2e-7, -1e-7)
    events = tuple(
        CorporateActionEvent(
            "listing-aapl",
            "fixture",
            when,
            kind,
            *((value, None) if kind == "SPLIT" else (None, value)),
        )
        for when, kind, value in actions
    )

    def scale(item: date) -> float:
        return math.prod(_factor(*action) for action in actions if action[0] > item)

    resent = {
        item: _close(item) * scale(item) * (1 + wobble)
        for item, wobble in zip(HISTORY, noise, strict=True)
        if not window or item > HISTORY[0]
    } | {NEW: _close(NEW)}
    if window:
        with pytest.raises(ActionAuditScopeInsufficient, match="before the rolling scope"):
            day((NEW,), resent, 1, events)
        closes = store.provider_adjusted_closes("listing-aapl", through=NEW)
        assert {p.session_date: p.adjusted_close for p in closes} == {
            item: _close(item) for item in HISTORY
        }
        return
    receipt, counts = day((NEW,), resent, 1, events)
    revision = store.provider_adjusted_revision(receipt.receipt_hash)
    starts = sorted({when for when, _kind, _value in actions if when in HISTORY} | {HISTORY[0]})
    start = {h: max(first for first in starts if first <= h) for h in HISTORY}
    last = {first: max(h for h in HISTORY if start[h] == first) for first in starts}
    # Each segment before an action is scaled by its ratio beside the action; the last stands.
    factor = {f: resent[last[f]] / _close(last[f]) if scale(last[f]) != 1 else 1.0 for f in starts}
    parts = [[f.isoformat(), last[f].isoformat(), factor[f]] for f in starts]
    assert counts["adjustment_segments"] == parts and 0 < counts["adjusted_precision_kept"] <= 1e-6
    # The revision keeps its disclosure: read back, it is the one written and its hash holds.
    assert revision is not None and revision.disclosed == {
        "adjustment_segments": parts,
        "adjusted_precision_kept": counts["adjusted_precision_kept"],
    }
    assert revision.changed_return_sessions == (*starts[1:], NEW)
    assert revision.hashed() == revision.receipt_hash
    # The receipt binds the closes as stored, so its evidence holds for what reads them later.
    assert receipt.provider_adjusted_close_evidence_hash == _stored_evidence(store, receipt)
    stored = store.provider_adjusted_closes("listing-aapl", through=NEW)
    assert {p.session_date: p.adjusted_close for p in stored} == {
        item: _close(item) * factor[start[item]] for item in HISTORY
    } | {NEW: _close(NEW)}


def test_a_back_adjustment_seen_rolling_leaves_no_receipt_until_its_full_audit(tmp_path) -> None:
    """requirement: a rolling refresh that meets a back-adjustment keeps nothing of it, so the
    listing stays in its membership; its approved full audit keeps the factor form."""
    manifest = build_quality_filtered_research_manifest(
        acquisition_manifest(), eligible_listing_ids=("listing-aapl",)
    )
    store = MarketDataRepository(tmp_path / "workspace")
    store.bootstrap(manifest)
    sessions = [date(2026, 5, 1) + timedelta(n) for n in range(92)]
    ex_date, factor = sessions[-1], 1 - 0.5 / _close(sessions[-2])

    def rows(start: date, end: date) -> tuple[dict[str, object], ...]:
        return tuple(
            {"session_date": d.isoformat(), "open": _close(d), "high": 120.0, "low": 90.0}
            | {"close": _close(d), "volume": 1_000_000}
            for d in sessions
            if start <= d <= end
        )

    def resent(d: date) -> float:
        return _close(d) * (factor * (1 + 3e-7 * (d.day % 3 - 1)) if d < ex_date else 1.0)

    class Provider:
        name = "fixture"

        def fetch_hydration(self, *, listing_id, provider_symbol, start, end):  # type: ignore[no-untyped-def]
            return HydrationEvidence(
                daily_rows=rows(start, end),
                actions=(
                    CorporateActionEvent(
                        listing_id, "fixture", ex_date, "CASH_DIVIDEND", None, 0.5
                    ),
                ),
                adjusted_closes=tuple(
                    ProviderAdjustedClosePoint(listing_id, self.name, d, resent(d))
                    for d in sessions
                    if start <= d <= end
                ),
            )

    store.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", {"AAPL": rows(sessions[0], sessions[-2])}, ("AAPL",)),
        ingestion_id="initial",
        observed_at=NOW,
    )
    audit_one(
        store,
        manifest,
        listing_id="listing-aapl",
        provider="fixture",
        actions=(),
        adjusted_closes=tuple(
            ProviderAdjustedClosePoint("listing-aapl", "fixture", d, _close(d))
            for d in sessions[:-1]
        ),
        history_start=sessions[0],
        history_end=sessions[-2],
        requested_as_of=sessions[-2],
        observed_at=NOW,
    )

    def run(escalate: frozenset[str]):  # type: ignore[no-untyped-def]
        return CurrentUniverseMaintenance(
            store=store,
            manifest=manifest,
            provider=Provider(),
            as_of_session=ex_date,
            max_workers=1,
            mutation_gate=WorkspaceMutationGate(),
            adjusted_tolerance=1e-6,
            full_history_escalation_listing_ids=escalate,
        ).run(observed_at=NOW + timedelta(days=1))

    def state() -> tuple[dict[date, float], dict[str, Any]]:
        points = store.provider_adjusted_closes("listing-aapl", through=ex_date)
        receipts = store.latest_action_audit_receipts(
            ("listing-aapl",), provider="fixture", requested_as_of=ex_date
        )
        return {p.session_date: p.adjusted_close for p in points}, receipts

    prior, _ = state()
    assert run(frozenset()).failed == 1  # its full audit waits for approval
    # Nothing of the rolling attempt stands: no receipt over unscaled closes to quarantine it.
    assert state() == (prior, {})
    outcome = run(frozenset({"listing-aapl"}))
    assert outcome.status is CurrentUniverseMaintenanceStatus.COMPLETED and outcome.failed == 0
    kept, receipts = state()
    assert receipts["listing-aapl"].adjusted_close_mismatch_count == 0
    assert kept == {d: prior[d] * (resent(sessions[-2]) / prior[sessions[-2]]) for d in prior} | {
        ex_date: _close(ex_date)
    }


@pytest.mark.parametrize(
    ("written", "tolerance"),
    [
        ({}, 1e-6),
        ({"corrected": 1}, 0.0),
        ({"action_inserted": 1}, 1e-6),
        ({"action_corrected": 2}, 1e-6),
    ],
)
def test_a_run_that_corrected_a_bar_keeps_no_stored_close(written, tolerance) -> None:
    """requirement: a corrected raw bar written this run keeps nothing; a moved action is
    judged by its returns at the precision, where a dividend's own move restates."""
    runner = SimpleNamespace(provider=SimpleNamespace(name="fixture"), as_of_session=NEW)
    runner.adjusted_tolerance = 1e-6
    hydration = SimpleNamespace(actions=(), adjusted_closes=())
    item = SimpleNamespace(listing_id="listing-aapl")
    observed = CurrentUniverseMaintenance._observation(runner, item, hydration, NEW, NOW, written)  # type: ignore[arg-type]
    assert observed.adjusted_tolerance == tolerance
