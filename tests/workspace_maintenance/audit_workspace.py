"""Small product-written raw evidence for receipt mechanics, without a Feature build.

These cases compare a complete range with 20- and 45-session rolling ranges,
correct bytes, and exercise receipt freshness. They do not judge universe or
Feature qualification. Two listings keep the missing-child-evidence case.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

from alphalattice.control.data_platform.maintenance.contracts import (
    ActionAuditChainReceipt,
    ActionAuditScope,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
from alphalattice.foundation.market_data_ops.storage.action_audit import ActionAuditObservation
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    ActionAuditReceipt,
    MarketDataRepository,
)
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from tests.researcher_methodology_surface.real_workspace import (
    AS_OF,
    OBSERVED_AT,
    SeededWalkProvider,
)
from tests.workspace_maintenance.acquisition_manifest import acquisition_manifest


def receipt_context(
    market: MarketDataRepository,
    parent: UniverseManifest,
    child: UniverseManifest,
    listing: ManifestListing,
) -> tuple[
    Callable[[datetime], tuple[str, ...]],
    Callable[[datetime], ActionAuditReceipt | None],
    Callable[[ActionAuditReceipt, datetime], bool],
]:
    market.bootstrap(child)

    def bind(now: datetime) -> tuple[str, ...]:
        bound: tuple[str, ...] = market.bind_available_action_audit_receipts_to_manifest(
            child, requested_as_of=AS_OF, now=now
        )
        return bound

    def full_history_answer(now: datetime) -> ActionAuditReceipt | None:
        return market.reusable_action_audit_receipt(
            child,
            listing_id=listing.listing_id,
            provider="yfinance",
            requested_as_of=AS_OF,
            now=now,
        )

    def verifies(receipt: ActionAuditReceipt, now: datetime) -> bool:
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

    return bind, full_history_answer, verifies


def build_audit_workspace(root: Path) -> tuple[Path, UniverseManifest]:
    start = AS_OF - timedelta(days=180)
    base = acquisition_manifest()
    manifest = replace(
        base,
        manifest_id="receipt-mechanics",
        revision_sha256="3" * 64,
        profile=replace(base.profile, provider="yfinance", manifest_as_of=AS_OF),
    )
    schedule = materialize_calendar_schedule(
        ("XNAS",), start=start, end=AS_OF, as_of_timestamp=OBSERVED_AT
    )
    sessions = tuple(row["session_date"] for row in schedule.to_pylist())
    # The 45-session scope must leave an older part of the full range.
    if len(sessions) < 90:
        raise AssertionError("receipt fixture requires an older range plus its rolling tail")
    symbols = tuple(listing.symbol for listing in manifest.listings)
    provider = SeededWalkProvider(symbols, sessions)
    market = MarketDataRepository(root)
    market.bootstrap(manifest)
    market.apply_validated_batch(
        manifest,
        sanitize_payload(
            manifest, provider.name, provider.fetch_daily(symbols, start=start, end=AS_OF), symbols
        ),
        ingestion_id="receipt-mechanics",
        observed_at=OBSERVED_AT,
    )
    for listing in manifest.listings:
        audit_one(
            market,
            manifest,
            listing_id=listing.listing_id,
            provider=provider.name,
            actions=(),
            adjusted_closes=provider.fetch_adjusted_close_history(
                listing_id=listing.listing_id,
                provider_symbol=listing.provider_symbol,
                start=start,
                end=AS_OF,
            ),
            history_start=sessions[0],
            history_end=AS_OF,
            requested_as_of=AS_OF,
            observed_at=OBSERVED_AT,
        )
    return root, manifest


def audit_one(store, manifest, *, _connection=None, **observation):  # type: ignore[no-untyped-def]
    """One observation's receipt and counts: the store's batch audit of that one alone."""
    audit = ActionAuditObservation(**observation)
    return store.complete_action_audit(manifest, audit, _connection=_connection)[0]


def seed_action_audit_rows(
    market_data: MarketDataRepository,
    manifest: UniverseManifest,
    sessions: tuple[date, ...],
    *,
    observed_at: datetime,
) -> None:
    """Build fresh provider-shaped rows for the caller's action-audit clock."""
    rows = tuple(
        {
            "session_date": session.isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
        }
        for session in sessions
    )
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", {"AAPL": rows}, ("AAPL",)),
        ingestion_id="initial",
        observed_at=observed_at,
    )


def full_action_audit_anchor(*, observed_at: datetime) -> ActionAuditChainReceipt:
    """Build the full-history anchor under the caller's exact clock."""
    return ActionAuditChainReceipt.create(
        listing_id="listing-aapl",
        provider="fixture",
        audit_scope=ActionAuditScope.FULL,
        previous_receipt_hash=None,
        full_anchor_receipt_hash=None,
        history_start=date(2016, 1, 4),
        history_end=date(2026, 7, 31),
        covered_through_session=date(2026, 7, 31),
        window_action_hash="a" * 64,
        action_set_hash="a" * 64,
        raw_evidence_hash="b" * 64,
        mapping_revision="c" * 64,
        data_policy_hash="d" * 64,
        provider_receipt_hash="e" * 64,
        observed_at=observed_at,
    )
