"""Persisted set-scoped quality reads agree with the same store's listing readers."""

from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pytest
from tests.workspace_maintenance.acquisition_manifest import NOW
from tests.workspace_maintenance.audit_workspace import audit_one
from tests.workspace_maintenance.data_update_support import (
    maintenance_workspace as _maintenance_workspace,
)

from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/storage/duckdb.py"


@pytest.mark.proves("market_data_ops.set_quality_inputs_match_listing_readers")
def test_set_scoped_quality_evidence_reads_match_the_per_listing_readers(tmp_path: Path) -> None:
    """Set scoped quality evidence reads match the per listing readers."""

    manifest, market_data, _, _ = _maintenance_workspace(
        tmp_path, eligible_listing_ids=("listing-aapl", "listing-msft")
    )
    sessions = (date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31))

    def rows(selected: tuple[date, ...], *, close: float) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "session_date": session.isoformat(),
                "open": close + index,
                "high": close + index + 1.0,
                "low": close + index - 1.0,
                "close": close + index,
                "volume": 1_000_000,
            }
            for index, session in enumerate(selected)
        )

    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(
            manifest,
            "fixture",
            {"AAPL": rows(sessions, close=100.0), "MSFT": rows(sessions[1:], close=50.0)},
            ("AAPL", "MSFT"),
        ),
        ingestion_id="set-scoped-reads",
        observed_at=NOW - timedelta(days=1),
    )
    for onboarding_id, evaluated_at, eligible in (
        ("onboarding-older", NOW - timedelta(days=2), False),
        ("onboarding-newer", NOW - timedelta(days=1), True),
    ):
        market_data.admit_current_universe_onboarding(
            manifest,
            onboarding_id=onboarding_id,
            candidate_manifest_hash=manifest.revision_sha256,
            candidate_manifest_document=asdict(manifest),
            history_start=sessions[0],
            as_of_session=sessions[-1],
            calendar_by_listing_id={
                listing.listing_id: listing.mic for listing in manifest.listings
            },
            observed_at=evaluated_at,
        )
        for listing_id in ("listing-aapl", "listing-msft"):
            market_data.record_current_universe_quality_admission(
                onboarding_id=onboarding_id,
                listing_id=listing_id,
                eligible=eligible,
                expected_sessions=4,
                observed_sessions=4,
                missing_sessions=0,
                missing_ratio=0.0,
                maximum_consecutive_gap=0,
                reasons=() if eligible else ("QUALITY_FIXTURE",),
                observed_at=evaluated_at,
            )
    audit_one(
        market_data,
        manifest,
        listing_id="listing-aapl",
        provider="fixture",
        actions=(
            CorporateActionEvent(
                listing_id="listing-aapl",
                provider="fixture",
                effective_date=sessions[2],
                action_kind="SPLIT",
                new_shares_per_old_share=2.0,
            ),
            CorporateActionEvent(
                listing_id="listing-aapl",
                provider="fixture",
                effective_date=sessions[1],
                action_kind="CASH_DIVIDEND",
                cash_amount=1.0,
            ),
        ),
        adjusted_closes=tuple(
            ProviderAdjustedClosePoint(
                listing_id="listing-aapl",
                provider="fixture",
                session_date=session,
                adjusted_close=100.0,
            )
            for session in sessions
        ),
        history_start=sessions[0],
        history_end=sessions[-1],
        requested_as_of=sessions[-1],
        observed_at=NOW,
    )

    known = ("listing-msft", "listing-aapl")
    through = sessions[-2]
    inputs = market_data.quality_governance_inputs((*known, "listing-none"), through=through)
    assert inputs.admissions == {
        listing_id: market_data.latest_quality_admission(listing_id) for listing_id in known
    }
    assert all(
        admission.onboarding_id == "onboarding-newer" for admission in inputs.admissions.values()
    )

    series = inputs.raw_close_series
    assert series.column_names == ["listing_id", "session_date", "close"]
    observed = list(
        zip(
            series.column("listing_id").to_pylist(),
            series.column("session_date").to_pylist(),
            series.column("close").to_pylist(),
            strict=True,
        )
    )
    assert observed == [
        (listing_id, bar.session_date, bar.close)
        for listing_id in sorted(known)
        for bar in market_data.raw_bars(listing_id, through=through)
    ]
    assert {session for _listing, session, _close in observed} == set(sessions[:-1])

    assert inputs.actions == {listing_id: market_data.actions(listing_id) for listing_id in known}
    assert [item.action_kind for item in inputs.actions["listing-aapl"]] == [
        "CASH_DIVIDEND",
        "SPLIT",
    ]
    with pytest.raises(ValueError, match="no provider mapping"):
        market_data.actions("listing-none")
