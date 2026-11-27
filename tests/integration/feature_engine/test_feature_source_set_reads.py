"""A Feature build's inputs, read once for every listing, equal each listing's own read."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from alphalattice.capabilities.domain_contracts.bounded_memo import stage_digests
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    projected_feature_rows,
)
from alphalattice.foundation.market_data_ops.publication import projection
from alphalattice.foundation.market_data_ops.sources.contracts import CorporateActionEvent
from alphalattice.foundation.market_data_ops.storage import duckdb as market_storage
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.identity import blake3_digest
from tests._support.identity_fixtures import fixture_digest
from tests._support.researcher_methodology_session_fixtures import (
    real_risk_workspace,
)
from tests._support.researcher_methodology_surface_hooks import offline_execution_environment
from tests.researcher_methodology_surface.session_workspace import copy_workspace

__all__ = ["offline_execution_environment", "real_risk_workspace"]


def _frozen_raw_evidence_hash(connection, listing, provider, start, end):
    """The raw-evidence rule as it stood before the set read: its own SQL and JSON hash."""
    current = connection.execute(
        "SELECT session_date, payload_hash FROM raw_daily_bar_current "
        "WHERE listing_id = ? AND provider = ? AND session_date BETWEEN ? AND ? "
        "ORDER BY session_date",
        [listing, provider, start, end],
    ).fetchall()
    revisions = connection.execute(
        "SELECT session_date, prior_payload_hash, next_payload_hash, observed_at FROM bar_revision "
        "WHERE listing_id = ? AND provider = ? AND session_date BETWEEN ? AND ? "
        "ORDER BY session_date, observed_at",
        [listing, provider, start, end],
    ).fetchall()
    payload = json.dumps(
        {"current": current, "revisions": revisions},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return blake3_digest(payload.encode("utf-8"))


def test_one_read_for_every_listing_equals_each_listings_own_read(
    real_risk_workspace, tmp_path, monkeypatch
):
    """requirement: one read per table, each listing from its own start, equals each
    listing's own read, revisions inside and outside its window included; a listing with no
    bars from its start is left to its own read, which refuses it as before."""
    workspace = copy_workspace(real_risk_workspace.workspace, tmp_path / "inputs")
    market = MarketDataRepository(workspace)
    feature = FeatureStateRepository(market.database, market_data=market)
    manifest = market.current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    assert manifest is not None
    listings = [item.listing_id for item in manifest.listings][:8]
    through = market.raw_bar_sessions(listings[0])[-1]
    starts = {
        listing: (manifest, market.raw_bar_sessions(listing)[-60 - 7 * index])
        for index, listing in enumerate(listings[:-1])
    }
    starts[listings[-1]] = (manifest, through + timedelta(days=1))
    first, second = listings[0], listings[1]
    inside, before = starts[first][1] + timedelta(days=14), starts[second][1] - timedelta(days=3)
    with market.database.connect(read_only=False) as connection:
        for revision, (listing, session) in enumerate(((first, inside), (second, before))):
            connection.execute(
                "INSERT INTO bar_revision SELECT ?, listing_id, provider, ?, ?, ?, ?, 'fixture' "
                "FROM raw_daily_bar_current WHERE listing_id = ? LIMIT 1",
                [
                    f"r{revision}",
                    session,
                    fixture_digest("a"),
                    fixture_digest("b"),
                    datetime(2026, 7, 1, tzinfo=UTC),
                    listing,
                ],
            )
    hash_calls = []
    stage_digest = market_storage.stage_digest

    def counted(slot, inputs, digest):
        def compute(values):
            if slot[1] == "raw":
                hash_calls.append(slot)
            return digest(values)

        return stage_digest(slot, inputs, compute)

    monkeypatch.setattr(market_storage, "stage_digest", counted)
    with stage_digests(), market.database.read_transaction() as connection:
        together = feature.feature_source_inputs_by_listing(
            starts, through=through, _connection=connection
        )
        assert set(together) == set(listings[:-1])
        for listing, (_manifest, start) in starts.items():
            if listing not in together:
                with pytest.raises(ValueError, match="listing has no raw bars"):
                    feature.feature_source_inputs(
                        manifest,
                        listing_id=listing,
                        through=through,
                        start=start,
                        _connection=connection,
                    )
                continue
            alone = feature.feature_source_inputs(
                manifest, listing_id=listing, through=through, start=start, _connection=connection
            )
            held = together[listing]
            assert held.bars.equals(alone.bars)
            assert held.adjusted_close.equals(alone.adjusted_close)
            assert held.actions == alone.actions
            assert (held.raw_hash, held.action_hash) == (alone.raw_hash, alone.action_hash)
            provider = held.bars.column("provider")[0].as_py()
            first_session = held.bars.column("session_date")[0].as_py()
            assert held.raw_hash == _frozen_raw_evidence_hash(
                connection, listing, provider, first_session, through
            )

        assert len(hash_calls) == len(together)
        provider = together[first].bars.column("provider")[0].as_py()
        first_session = together[first].bars.column("session_date")[0].as_py()
        assert (
            MarketDataRepository(workspace)._raw_evidence_hash(
                connection,
                listing_id=first,
                provider=provider,
                history_start=first_session,
                history_end=through,
            )
            == together[first].raw_hash
        )
        assert len(hash_calls) == len(together)

    with stage_digests(), market.database.transaction() as connection:
        before = feature.feature_source_inputs_by_listing(
            starts, through=through, _connection=connection
        )
        assert len(hash_calls) == 2 * len(together)
        connection.execute(
            "UPDATE raw_daily_bar_current SET payload_hash = ? "
            "WHERE listing_id = ? AND provider = ? AND session_date = ?",
            [fixture_digest("c"), first, provider, first_session],
        )
        after = feature.feature_source_inputs_by_listing(
            starts, through=through, _connection=connection
        )
        assert len(hash_calls) == 2 * len(together) + 1
        assert after[first].raw_hash != before[first].raw_hash
        assert after[first].raw_hash == _frozen_raw_evidence_hash(
            connection, first, provider, first_session, through
        )
        assert all(after[key].raw_hash == before[key].raw_hash for key in before if key != first)

    source = together[first]
    in_window = CorporateActionEvent(
        first, provider, first_session, "CASH_DIVIDEND", cash_amount=0.1
    )
    after_window = replace(in_window, effective_date=through + timedelta(days=1))
    digest = projection.action_set_hash
    projection_hashes = []

    def count_projection(actions):
        projection_hashes.append(actions)
        return digest(actions)

    monkeypatch.setattr(projection, "action_set_hash", count_projection)
    for actions, scoped in (
        ((), ()),
        ((in_window,), (in_window,)),
        ((in_window, after_window), (in_window,)),
    ):
        projection_hashes.clear()
        rows = projected_feature_rows(replace(source, actions=actions, action_hash=digest(actions)))
        assert {row["action_set_hash"] for row in rows} == {digest(scoped)}
        assert projection_hashes == ([] if actions == scoped else [scoped])
