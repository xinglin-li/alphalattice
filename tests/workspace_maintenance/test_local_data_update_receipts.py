"""Audit receipts and child bindings across local data updates, offline."""

from __future__ import annotations

import shutil
from dataclasses import asdict, replace
from datetime import date, timedelta

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.maintenance.data_update import bind_existing_data_workspace
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from tests.portfolio_strategy_lab.local_web_support import (
    _resolved,
    _Resolver,
)
from tests.researcher_methodology_surface.real_workspace import (
    AS_OF,
    SYMBOLS,
)
from tests.workspace_maintenance.audit_workspace import build_audit_workspace
from tests.workspace_maintenance.data_update_support import _provider
from tests.workspace_maintenance.local_data_provider import (
    NOW,
    unchanged_membership_source,
)


@pytest.fixture(scope="module")
def audit_source(tmp_path_factory):
    return build_audit_workspace(tmp_path_factory.mktemp("raw-audit") / "workspace")


def test_action_audit_reuse_does_not_order_equal_time_revisions_by_hash(audit_source, tmp_path):

    from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload

    workspace = tmp_path / "workspace"
    shutil.copytree(audit_source[0], workspace)
    market = MarketDataRepository(workspace)
    manifest = audit_source[1]
    listing = manifest.listings[0]
    bars = market.raw_bars(listing.listing_id, through=AS_OF)
    last = bars[-1]
    original = {name: getattr(last, name) for name in ("open", "high", "low", "close", "volume")}
    original["session_date"] = last.session_date.isoformat()

    def write(volume, label):
        batch = sanitize_payload(
            manifest,
            "yfinance",
            {listing.symbol: [{**original, "volume": volume}]},
            [listing.symbol],
        )
        market.apply_validated_batch(manifest, batch, ingestion_id=label, observed_at=NOW)

    def audit():
        return market.complete_action_audit(
            manifest,
            listing_id=listing.listing_id,
            provider="yfinance",
            observed_actions=market.actions(listing.listing_id),
            observed_adjusted_closes=market.provider_adjusted_closes(
                listing.listing_id, through=AS_OF
            ),
            history_start=bars[0].session_date,
            history_end=AS_OF,
            requested_as_of=AS_OF,
            observed_at=NOW,
        )[0]

    first = audit()
    write(last.volume + 1, "same-time-correction")
    second = audit()
    # Two receipts at one observation time. Whichever hash sorts first, the
    # reusable one is the receipt that proves the current bytes -- the second
    # -- and the evidence it proves includes the revision journal, so writing
    # the first version back would not make the first receipt current again.
    assert first.receipt_hash != second.receipt_hash
    actual = market.reusable_action_audit_receipt(
        manifest,
        listing_id=listing.listing_id,
        provider="yfinance",
        requested_as_of=AS_OF,
        now=NOW,
    )
    assert actual == second
    write(last.volume + 2, "neither-receipt-matches")
    assert (
        market.reusable_action_audit_receipt(
            manifest,
            listing_id=listing.listing_id,
            provider="yfinance",
            requested_as_of=AS_OF,
            now=NOW,
        )
        is None
    )

    child_manifest = replace(
        manifest,
        manifest_id=manifest.manifest_id + ":rebind-check",
        revision_sha256="f" * 64,
        listings=(listing,),
    )
    market.bootstrap(child_manifest)
    # A metadata candidate is not authority over changed source evidence.
    # Restoring prices does not undo the revision journal; a new audit is needed.
    assert (
        market.bind_available_action_audit_receipts_to_manifest(
            child_manifest, requested_as_of=AS_OF, now=NOW
        )
        == ()
    )
    write(last.volume + 1, "restore-rebind-source")
    assert (
        market.bind_available_action_audit_receipts_to_manifest(
            child_manifest, requested_as_of=AS_OF, now=NOW
        )
        == ()
    )
    reobserved = audit()
    assert market.bind_available_action_audit_receipts_to_manifest(
        child_manifest, requested_as_of=AS_OF, now=NOW
    ) == (listing.listing_id,)
    child = market.reusable_action_audit_receipt(
        child_manifest,
        listing_id=listing.listing_id,
        provider="yfinance",
        requested_as_of=AS_OF,
        now=NOW,
    )
    assert child is not None and child.manifest_revision == child_manifest.revision_sha256
    assert child.raw_evidence_hash == reobserved.raw_evidence_hash
    assert child.observed_at == reobserved.observed_at


def test_child_rebinding_replaces_a_stale_bound_receipt_and_then_costs_nothing(
    audit_source, tmp_path, monkeypatch
):
    """Child rebinding replaces a stale bound receipt and then costs nothing."""

    from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload

    workspace = tmp_path / "workspace"
    shutil.copytree(audit_source[0], workspace)
    market = MarketDataRepository(workspace)
    parent = audit_source[1]
    listing = parent.listings[0]
    bars = market.raw_bars(listing.listing_id, through=AS_OF)
    last = bars[-1]
    original = {name: getattr(last, name) for name in ("open", "high", "low", "close", "volume")}
    original["session_date"] = last.session_date.isoformat()

    def write(volume, label):
        batch = sanitize_payload(
            parent, "yfinance", {listing.symbol: [{**original, "volume": volume}]}, [listing.symbol]
        )
        market.apply_validated_batch(parent, batch, ingestion_id=label, observed_at=NOW)

    def audit(now):
        return market.complete_action_audit(
            parent,
            listing_id=listing.listing_id,
            provider="yfinance",
            observed_actions=market.actions(listing.listing_id),
            observed_adjusted_closes=market.provider_adjusted_closes(
                listing.listing_id, through=AS_OF
            ),
            history_start=bars[0].session_date,
            history_end=AS_OF,
            requested_as_of=AS_OF,
            observed_at=now,
        )[0]

    child = replace(
        parent,
        manifest_id=parent.manifest_id + ":stale-rebind",
        revision_sha256="e" * 64,
        listings=(listing,),
    )
    market.bootstrap(child)
    # The expensive part of a judgement is the evidence digest over the
    # receipt's range; the byte-free refusals cost none.
    validated: list[str] = []
    evidence = market._action_audit_evidence

    def counting_evidence(connection, **kwargs):
        validated.append(kwargs["listing_id"])
        return evidence(connection, **kwargs)

    monkeypatch.setattr(market, "_action_audit_evidence", counting_evidence)

    def bind(now):
        return market.bind_available_action_audit_receipts_to_manifest(
            child, requested_as_of=AS_OF, now=now
        )

    def reusable(now):
        return market.reusable_action_audit_receipt(
            child,
            listing_id=listing.listing_id,
            provider="yfinance",
            requested_as_of=AS_OF,
            now=now,
        )

    first = audit(NOW)
    assert bind(NOW) == (listing.listing_id,)
    bound = reusable(NOW)
    assert bound is not None and bound.raw_evidence_hash == first.raw_evidence_hash
    # 1. A valid bound receipt and the same source: repeated calls validate nothing.
    validated.clear()
    assert bind(NOW) == () and bind(NOW) == ()
    assert validated == []
    # 2. The source is corrected after the binding: the child's copy no longer
    # proves the bytes, the parent audits again and holds new valid evidence.
    write(last.volume + 1, "correction-after-binding")
    assert reusable(NOW) is None
    later = NOW + timedelta(hours=1)
    second = audit(later)
    assert second.raw_evidence_hash != first.raw_evidence_hash
    validated.clear()
    assert bind(later) == (listing.listing_id,)
    assert validated == [listing.listing_id]
    rebound = reusable(later)
    assert rebound is not None and rebound.raw_evidence_hash == second.raw_evidence_hash
    assert rebound.observed_at == second.observed_at
    # 5. Rebound, the listing is back on the cheap path.
    validated.clear()
    assert bind(later) == ()
    assert validated == []
    # 3. No qualified evidence anywhere: a second correction with no new audit.
    # The parent's evidence is already held, so nothing is re-judged, nothing
    # is bound, and the stale copies are not reused -- the acquisition path
    # must audit again.
    write(last.volume + 2, "correction-without-audit")
    validated.clear()
    assert bind(later) == ()
    assert validated == []
    assert reusable(later) is None
    # 4. Foreign evidence the child does not hold is re-judged by the validator,
    # not taken because it is newest: the parent audits the corrected bytes,
    # the bytes move again before the child's cycle, and the binding refuses.
    third = audit(later + timedelta(minutes=1))
    write(last.volume + 3, "correction-after-third-audit")
    validated.clear()
    assert bind(later + timedelta(minutes=2)) == ()
    assert validated == [listing.listing_id]
    assert reusable(later + timedelta(minutes=2)) is None
    # Once the parent has audited the bytes as they stand, the binding takes it.
    fourth = audit(later + timedelta(minutes=3))
    assert fourth.raw_evidence_hash != third.raw_evidence_hash
    assert bind(later + timedelta(minutes=3)) == (listing.listing_id,)
    current = reusable(later + timedelta(minutes=3))
    assert current is not None and current.raw_evidence_hash == fourth.raw_evidence_hash
    # The refused third receipt is unheld and still fresh, so a repeated call
    # judges it again -- one digest over its range, no binding -- until its
    # freshness ends; a later instant over the same range is not dominance
    # (the bytes may have moved and returned between the two).
    validated.clear()
    assert bind(later + timedelta(minutes=3)) == ()
    assert validated == [listing.listing_id]
    validated.clear()
    assert bind(later + timedelta(hours=25)) == ()
    assert validated == []


def test_older_ancestor_receipt_that_proves_the_current_bytes_is_not_pruned_by_its_instant(
    audit_source, tmp_path, monkeypatch
):
    """Older ancestor receipt that proves the current bytes is not pruned by its instant."""

    workspace = tmp_path / "workspace"
    shutil.copytree(audit_source[0], workspace)
    market = MarketDataRepository(workspace)
    parent = audit_source[1]
    listing = parent.listings[0]
    bars = market.raw_bars(listing.listing_id, through=AS_OF)
    window_start = bars[-45].session_date

    series_a = market.provider_adjusted_closes(listing.listing_id, through=AS_OF)

    def audit(now, *, start, scale):
        points = tuple(
            replace(point, adjusted_close=point.adjusted_close * scale)
            for point in series_a
            if point.session_date >= start
        )
        return market.complete_action_audit(
            parent,
            listing_id=listing.listing_id,
            provider="yfinance",
            observed_actions=market.actions(listing.listing_id),
            observed_adjusted_closes=points,
            history_start=start,
            history_end=AS_OF,
            requested_as_of=AS_OF,
            observed_at=now,
        )[0]

    child = replace(
        parent,
        manifest_id=parent.manifest_id + ":instant-is-not-dominance",
        revision_sha256="d" * 64,
        listings=(listing,),
    )
    market.bootstrap(child)

    def bind(now):
        return market.bind_available_action_audit_receipts_to_manifest(
            child, requested_as_of=AS_OF, now=now
        )

    def full_history_answer(now):
        return market.reusable_action_audit_receipt(
            child,
            listing_id=listing.listing_id,
            provider="yfinance",
            requested_as_of=AS_OF,
            now=now,
        )

    def verifies(receipt, now):
        connection = market._connect(read_only=True)
        try:
            return (
                market._rebindable_action_audit_receipt(
                    connection,
                    listing_id=listing.listing_id,
                    provider="yfinance",
                    requested_as_of=AS_OF,
                    now=now,
                    manifest_revision=parent.revision_sha256,
                    receipt_hash=receipt.receipt_hash,
                )
                is not None
            )
        finally:
            connection.close()

    # 1. Evidence A at t1: the ancestor's full-history receipt; the child holds nothing.
    t1, t2, t3 = NOW, NOW + timedelta(hours=1), NOW + timedelta(hours=2)
    full_a = audit(t1, start=bars[0].session_date, scale=1.0)
    assert full_a.history_start == bars[0].session_date
    # 2. The series moves to B inside the window; the ancestor's rolling audit
    # at t2 seals B, and the child binds that receipt (the full receipt no
    # longer proves the bytes, so it is refused at t2).
    rolling_b = audit(t2, start=window_start, scale=1.001)
    assert bind(t2) == (listing.listing_id,)
    assert not verifies(full_a, t2)
    assert full_history_answer(t2) is None  # a rolling copy answers no full-history question
    # 3. The series returns to A: the ancestor's rolling audit at t3 seals A
    # over the window; no new full audit. The t1 receipt is inside its TTL.
    rolling_a = audit(t3, start=window_start, scale=1.0)
    assert (
        rolling_a.provider_adjusted_close_evidence_hash
        != rolling_b.provider_adjusted_close_evidence_hash
    )
    # 4. The complete validator accepts the ancestor's t1 receipt on its own.
    assert verifies(full_a, t3)
    assert not verifies(rolling_b, t3)
    # 5. The binding must not prune it for t1 < t2: after the child has taken
    # what verifies, the full-history question is answered from the copy of
    # the t1 receipt -- same instant, same range, same evidence -- and once
    # everything is held the calls cost no validation.
    # The expensive part of a judgement is the evidence digest over the
    # receipt's range; the byte-free refusals cost none.
    validated: list[str] = []
    evidence = market._action_audit_evidence

    def counting_evidence(connection, **kwargs):
        validated.append(kwargs["listing_id"])
        return evidence(connection, **kwargs)

    monkeypatch.setattr(market, "_action_audit_evidence", counting_evidence)
    assert bind(t3) == (listing.listing_id,)
    # Two receipts proved their ranges (the t1 full one and the t3 rolling
    # one), one digest each; the superseded t2 rolling receipt cost none.
    assert validated == [listing.listing_id, listing.listing_id]
    answer = full_history_answer(t3)
    assert answer is not None, (
        "the child holds no full-history evidence although the ancestor's t1 receipt verifies"
    )
    assert answer.observed_at == full_a.observed_at
    assert answer.history_start == full_a.history_start
    assert (
        answer.provider_adjusted_close_evidence_hash == full_a.provider_adjusted_close_evidence_hash
    )
    validated.clear()
    assert bind(t3) == () and bind(t3) == ()
    assert validated == []


def test_a_later_full_audit_over_the_same_range_does_not_supersede_an_older_valid_one(
    audit_source, tmp_path, monkeypatch
):
    """A later full audit over the same range does not supersede an older valid one."""

    workspace = tmp_path / "workspace"
    shutil.copytree(audit_source[0], workspace)
    market = MarketDataRepository(workspace)
    parent = audit_source[1]
    listing = parent.listings[0]
    bars = market.raw_bars(listing.listing_id, through=AS_OF)
    window_start = bars[-45].session_date
    series_a = market.provider_adjusted_closes(listing.listing_id, through=AS_OF)

    def audit(now, *, start, tail_scale):
        points = tuple(
            replace(point, adjusted_close=point.adjusted_close * tail_scale)
            if point.session_date >= window_start
            else point
            for point in series_a
            if point.session_date >= start
        )
        return market.complete_action_audit(
            parent,
            listing_id=listing.listing_id,
            provider="yfinance",
            observed_actions=market.actions(listing.listing_id),
            observed_adjusted_closes=points,
            history_start=start,
            history_end=AS_OF,
            requested_as_of=AS_OF,
            observed_at=now,
        )[0]

    child = replace(
        parent,
        manifest_id=parent.manifest_id + ":same-range-is-not-dominance",
        revision_sha256="c" * 64,
        listings=(listing,),
    )
    market.bootstrap(child)

    def bind(now):
        return market.bind_available_action_audit_receipts_to_manifest(
            child, requested_as_of=AS_OF, now=now
        )

    def full_history_answer(now):
        return market.reusable_action_audit_receipt(
            child,
            listing_id=listing.listing_id,
            provider="yfinance",
            requested_as_of=AS_OF,
            now=now,
        )

    def verifies(receipt, now):
        connection = market._connect(read_only=True)
        try:
            return (
                market._rebindable_action_audit_receipt(
                    connection,
                    listing_id=listing.listing_id,
                    provider="yfinance",
                    requested_as_of=AS_OF,
                    now=now,
                    manifest_revision=parent.revision_sha256,
                    receipt_hash=receipt.receipt_hash,
                )
                is not None
            )
        finally:
            connection.close()

    t1, t2, t3 = NOW, NOW + timedelta(hours=1), NOW + timedelta(hours=2)
    full_a = audit(t1, start=bars[0].session_date, tail_scale=1.0)
    full_b = audit(t2, start=bars[0].session_date, tail_scale=1.001)
    assert (full_a.history_start, full_a.history_end) == (full_b.history_start, full_b.history_end)
    assert (
        full_b.provider_adjusted_close_evidence_hash != full_a.provider_adjusted_close_evidence_hash
    )
    assert full_b.raw_evidence_hash == full_a.raw_evidence_hash
    assert full_b.action_evidence_hash == full_a.action_evidence_hash
    assert full_b.mapping_revision == full_a.mapping_revision
    assert bind(t2) == (listing.listing_id,)
    bound_b = full_history_answer(t2)
    assert bound_b is not None and bound_b.observed_at == full_b.observed_at
    rolling_a = audit(t3, start=window_start, tail_scale=1.0)
    assert rolling_a.history_start == window_start
    # Before the binding is asked: the validator on its own.
    assert verifies(full_a, t3)
    assert not verifies(full_b, t3)
    assert verifies(rolling_a, t3)
    assert full_history_answer(t3) is None  # the t2 copy is refused, nothing else answers
    connection = market._connect(read_only=True)
    try:
        held = {
            row[0]
            for row in connection.execute(
                "SELECT receipt_hash FROM action_audit_receipt WHERE manifest_revision = ?",
                [child.revision_sha256],
            ).fetchall()
        }
    finally:
        connection.close()
    assert full_a.receipt_hash not in held
    assert bound_b.receipt_hash in held and len(held) == 1
    # The binding: the t1 receipt is taken, the child answers the full-history
    # question from its copy -- same instant, same range, same evidence.
    validated: list[tuple[str, date, date]] = []
    evidence = market._action_audit_evidence

    def counting_evidence(connection, **kwargs):
        validated.append((kwargs["listing_id"], kwargs["history_start"], kwargs["history_end"]))
        return evidence(connection, **kwargs)

    monkeypatch.setattr(market, "_action_audit_evidence", counting_evidence)
    assert bind(t3) == (listing.listing_id,)
    # One digest per range per call: the full range and the window, not one
    # per receipt.
    assert sorted(validated) == sorted(
        [
            (listing.listing_id, full_a.history_start, AS_OF),
            (listing.listing_id, window_start, AS_OF),
        ]
    )
    answer = full_history_answer(t3)
    assert answer is not None, "the t1 receipt was pruned although it proves the whole history"
    assert answer.observed_at == full_a.observed_at
    assert (answer.history_start, answer.history_end) == (full_a.history_start, full_a.history_end)
    assert (
        answer.provider_adjusted_close_evidence_hash == full_a.provider_adjusted_close_evidence_hash
    )
    # Idempotent: nothing unheld is left, no digest, nothing bound.
    validated.clear()
    assert bind(t3) == () and bind(t3) == ()
    assert validated == []


def test_child_binding_carries_the_parent_receipt_in_its_recorded_scope(audit_source, tmp_path):
    """Child binding carries the parent receipt in its recorded scope."""

    from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload

    workspace = tmp_path / "workspace"
    shutil.copytree(audit_source[0], workspace)
    market = MarketDataRepository(workspace)
    manifest = audit_source[1]
    listing, other = manifest.listings[0], manifest.listings[1]
    bars = market.raw_bars(listing.listing_id, through=AS_OF)
    window_start = bars[-20].session_date
    source = replace(
        manifest, manifest_id=manifest.manifest_id + ":parent", revision_sha256="e" * 64
    )
    market.bootstrap(source)
    rolling = market.complete_action_audit(
        source,
        listing_id=listing.listing_id,
        provider="yfinance",
        observed_actions=tuple(
            item
            for item in market.actions(listing.listing_id)
            if item.effective_date >= window_start
        ),
        observed_adjusted_closes=tuple(
            item
            for item in market.provider_adjusted_closes(listing.listing_id, through=AS_OF)
            if item.session_date >= window_start
        ),
        history_start=window_start,
        history_end=AS_OF,
        requested_as_of=AS_OF,
        observed_at=NOW,
    )[0]
    assert rolling.history_start == window_start > bars[0].session_date
    # Governance that runs after the maintenance reads the audit sealed for
    # the session in the scope it was run; the chain seed still reads the
    # newest full-history audit. Both are the same owner's one scan.
    as_of = market.latest_action_audit_receipts(
        (listing.listing_id,), provider="yfinance", requested_as_of=AS_OF
    )
    assert as_of[listing.listing_id] == rolling
    full = market.latest_action_audit_receipts((listing.listing_id,), provider="yfinance")
    assert full[listing.listing_id].history_start == bars[0].session_date
    child = replace(
        source,
        manifest_id=source.manifest_id + ":child",
        revision_sha256="f" * 64,
        listings=(listing,),
    )
    market.bootstrap(child)

    def receipts(revision: str) -> list[tuple]:
        connection = market._connect(read_only=True)
        try:
            return connection.execute(
                """
                SELECT history_start, history_end, raw_evidence_hash, observed_at
                FROM action_audit_receipt WHERE manifest_revision = ? AND requested_as_of = ?
                ORDER BY listing_id
                """,
                [revision, AS_OF],
            ).fetchall()
        finally:
            connection.close()

    market.bind_action_audit_receipts_to_manifest(source, child, requested_as_of=AS_OF, now=NOW)
    bound = receipts(child.revision_sha256)
    assert bound == [(window_start, AS_OF, rolling.raw_evidence_hash, rolling.observed_at)]
    # The child holds exactly what the parent holds, and the acquisition-budget
    # question is as unanswered for the child as for the parent: a rolling
    # receipt suppresses no full-history audit anywhere.
    for revision in (source, child):
        assert (
            market.reusable_action_audit_receipt(
                revision,
                listing_id=listing.listing_id,
                provider="yfinance",
                requested_as_of=AS_OF,
                now=NOW,
            )
            is None
        )
    # Idempotent: the same binding inserts nothing new.
    market.bind_action_audit_receipts_to_manifest(source, child, requested_as_of=AS_OF, now=NOW)
    assert receipts(child.revision_sha256) == bound

    # Missing evidence: a child listing the parent never audited for this session.
    wider = replace(child, revision_sha256="a" * 64, listings=(listing, other))
    market.bootstrap(wider)
    with pytest.raises(ValueError, match="no reusable action-audit receipt: " + other.listing_id):
        market.bind_action_audit_receipts_to_manifest(source, wider, requested_as_of=AS_OF, now=NOW)
    assert receipts(wider.revision_sha256) == []
    # Expired evidence: the receipt's own observation time is the budget.
    stale = replace(child, revision_sha256="b" * 64)
    market.bootstrap(stale)
    with pytest.raises(ValueError, match="no reusable action-audit receipt"):
        market.bind_action_audit_receipts_to_manifest(
            source, stale, requested_as_of=AS_OF, now=NOW + timedelta(hours=24)
        )
    assert receipts(stale.revision_sha256) == []
    # Drifted evidence inside the audited window: a corrected bar refuses.
    last = bars[-1]
    payload = {name: getattr(last, name) for name in ("open", "high", "low", "close", "volume")}
    payload["session_date"] = last.session_date.isoformat()
    payload["volume"] = last.volume + 1
    market.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "yfinance", {listing.symbol: [payload]}, [listing.symbol]),
        ingestion_id="window-correction",
        observed_at=NOW,
    )
    drifted = replace(child, revision_sha256="c" * 64)
    market.bootstrap(drifted)
    with pytest.raises(ValueError, match="no reusable action-audit receipt"):
        market.bind_action_audit_receipts_to_manifest(
            source, drifted, requested_as_of=AS_OF, now=NOW
        )
    assert receipts(drifted.revision_sha256) == []


def test_admission_diagnostic_covers_the_anchor_and_the_session_link(qualified, tmp_path):
    """Admission diagnostic covers the anchor and the session link."""

    from alphalattice.control.data_platform.maintenance.coordinator import (
        WorkspaceMaintenanceCoordinator,
    )
    from alphalattice.foundation.feature_engine.inputs.quality import (
        FeatureInputQualityEvaluator,
        RawCloseSeries,
    )

    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    market = MarketDataRepository(workspace)
    manifest = market.current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    listing = manifest.listings[0]
    bars = market.raw_bars(listing.listing_id, through=AS_OF)
    adjusted = market.provider_adjusted_closes(listing.listing_id, through=AS_OF)
    old_session = bars[len(bars) // 2].session_date
    # One historical adjusted close ten basis points off the provider's own
    # convention: a full-history audit records the mismatch on its receipt.
    perturbed = tuple(
        replace(item, adjusted_close=item.adjusted_close * 1.001)
        if item.session_date == old_session
        else item
        for item in adjusted
    )
    full = market.complete_action_audit(
        manifest,
        listing_id=listing.listing_id,
        provider="yfinance",
        observed_actions=market.actions(listing.listing_id),
        observed_adjusted_closes=perturbed,
        history_start=bars[0].session_date,
        history_end=AS_OF,
        requested_as_of=AS_OF,
        observed_at=NOW,
    )[0]
    assert full.max_adjusted_close_difference_bps > 5.0 and full.adjusted_close_mismatch_count
    # The next day's rolling link over the last twenty sessions is clean.
    window_start = bars[-20].session_date
    link = market.complete_action_audit(
        manifest,
        listing_id=listing.listing_id,
        provider="yfinance",
        observed_actions=tuple(
            item
            for item in market.actions(listing.listing_id)
            if item.effective_date >= window_start
        ),
        observed_adjusted_closes=tuple(
            item for item in perturbed if item.session_date >= window_start
        ),
        history_start=window_start,
        history_end=AS_OF,
        requested_as_of=AS_OF,
        observed_at=NOW + timedelta(minutes=1),
    )[0]
    assert link.max_adjusted_close_difference_bps < 5.0
    anchors = market.latest_action_audit_receipts((listing.listing_id,), provider="yfinance")
    links = market.latest_action_audit_receipts(
        (listing.listing_id,), provider="yfinance", requested_as_of=AS_OF
    )
    assert anchors[listing.listing_id] == full and links[listing.listing_id] == link
    diagnostic = WorkspaceMaintenanceCoordinator._admission_diagnostic_bps(
        link.max_adjusted_close_difference_bps, full.max_adjusted_close_difference_bps
    )
    assert diagnostic == full.max_adjusted_close_difference_bps
    assert WorkspaceMaintenanceCoordinator._admission_diagnostic_bps(0.0, None) == 0.0
    assert WorkspaceMaintenanceCoordinator._admission_diagnostic_bps(None, 0.25) == 0.25
    assert WorkspaceMaintenanceCoordinator._admission_diagnostic_bps(None, None) is None
    admission = market.latest_quality_admission(listing.listing_id)
    assert admission is not None
    evidence = FeatureInputQualityEvaluator().evaluate(
        admission=admission,
        provider="yfinance",
        range_start=bars[0].session_date,
        range_end=AS_OF,
        raw_closes=RawCloseSeries.of((bar.session_date, bar.close) for bar in bars),
        corporate_action_sessions=frozenset(),
        action_audit_completed=True,
        adjusted_diagnostic_max_bps=diagnostic,
        identity_verified=True,
        retry_exhausted=True,
    )
    assert "ADJUSTED_CLOSE_DIAGNOSTIC_MISMATCH" in evidence.reason_codes


def test_valuation_audit_permission_never_reenters_the_candidate_request(qualified, tmp_path):

    from alphalattice.control.data_platform.maintenance.contracts import (
        WorkspaceDataChange,
        WorkspaceDataUpdatePlan,
        WorkspaceInputStatus,
        WorkspaceMaintenanceRequest,
        WorkspaceValuationGrant,
    )
    from alphalattice.control.data_platform.maintenance.data_changes import WorkspaceDataChanges
    from alphalattice.control.product_host.maintenance.data_update import (
        installed_data_update_binding,
        read_workspace_inputs,
    )
    from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
    from alphalattice.foundation.market_data_ops.sources.manifest import (
        build_quality_filtered_research_manifest,
    )

    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    manifest = bind_existing_data_workspace(workspace)
    market = MarketDataRepository(workspace)
    old = market.current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    removed, *remaining = old.listings
    active = build_quality_filtered_research_manifest(
        old, eligible_listing_ids=tuple(value.listing_id for value in remaining)
    )
    valuation = build_quality_filtered_research_manifest(
        old, eligible_listing_ids=(removed.listing_id,)
    )
    market.bootstrap(active)
    binding = installed_data_update_binding()
    state = read_workspace_inputs(workspace, binding)
    before = WorkspaceInputStatus.seal(
        **{
            **state.model_dump(exclude={"content_hash"}),
            "manifest_revision": active.revision_sha256,
            "panel_manifest_revision": old.revision_sha256,
        }
    )
    with LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=manifest,
        resolver=_Resolver(_resolved()),
        data_provider=_provider(),
        data_source_loader=unchanged_membership_source(SYMBOLS),
        clock=lambda: NOW,
    ) as service:
        service.operations.data_update.plan()
        original = service.operations.data_update.last_plan
    scope = tuple(sorted((removed.listing_id, remaining[0].listing_id)))
    request = WorkspaceMaintenanceRequest.create(
        **{
            **{
                name: value
                for name, value in asdict(original.request).items()
                if name != "request_hash"
            },
            "membership_revision": active.revision_sha256,
            "full_history_listing_ids": scope,
        }
    )
    plan = WorkspaceDataUpdatePlan.seal(
        workspace_id=manifest.workspace_id,
        workspace_manifest_hash=manifest.manifest_hash,
        binding=binding,
        before=before,
        request=request,
        change=WorkspaceDataChange.seal(
            action="FULL_HISTORY_AUDIT", full_history_listing_ids=scope
        ),
        valuation_grants=(
            WorkspaceValuationGrant.seal(
                approved_plan_hash="a" * 64,
                manifest=valuation,
                book_roots=("b" * 64,),
                listing_ids=(removed.listing_id,),
            ),
        ),
    )
    owner = WorkspaceDataChanges(workspace, WorkspaceMutationGate())
    effective = owner.capture_execution(plan, active.revision_sha256)
    assert effective.request.full_history_listing_ids == (remaining[0].listing_id,)
    provider = _provider()
    quote_only = owner._quote_runner(plan, valuation, provider)
    assert quote_only.full_audit_listing_ids == frozenset((removed.listing_id,))
    assert owner.formation_scope(before) == valuation
    caught_up = WorkspaceInputStatus.seal(
        **{
            **before.model_dump(exclude={"content_hash"}),
            "panel_manifest_revision": active.revision_sha256,
        }
    )
    assert owner.formation_scope(caught_up) is None
    assert provider.calls == []
    assert (
        market.current_quality_filtered_research_manifest(
            market_profile_id="us-current-index-research"
        )
        == old
    )
