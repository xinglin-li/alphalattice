"""A split's rebase of a listing's stored bars is kept as factors, written once and disclosed."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
    CurrentUniverseMaintenance,
    CurrentUniverseMaintenanceStatus,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    build_quality_filtered_research_manifest,
)
from alphalattice.foundation.market_data_ops.sources.providers import HydrationEvidence
from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from tests.workspace_maintenance.acquisition_manifest import NOW, acquisition_manifest
from tests.workspace_maintenance.audit_workspace import audit_one

SESSIONS = tuple(date(2026, 5, 1) + timedelta(n) for n in range(92))


def _bar(day: date, scale: float = 1.0, close: float | None = None) -> RawDailyBar:
    price = (100.0 + day.day / 7.0 if close is None else close) * scale
    return RawDailyBar("L", "p", day, price, price * 1.01, price * 0.99, price, round(1e6 / scale))


def _split(day: date, ratio: float) -> CorporateActionEvent:
    return CorporateActionEvent("L", "p", day, "SPLIT", ratio)


def test_rebases_compose_in_session_order_and_refuse_what_no_split_explains() -> None:
    """requirement: each bar before an ex-date is the stored one times the factors after it, in
    session order, for any ratio; a move no split explains is not a rebase and restates."""
    from alphalattice.foundation.market_data_ops.storage.rebase import recognize

    stored = tuple(_bar(day) for day in SESSIONS[:10])
    first, second = SESSIONS[4], SESSIONS[8]

    def sent(bar: RawDailyBar, ratios: tuple[float, float]) -> RawDailyBar:
        scale = (1 / ratios[0] if bar.session_date < first else 1.0) * (
            1 / ratios[1] if bar.session_date < second else 1.0
        )
        return _bar(bar.session_date, scale) if scale != 1.0 else bar

    splits = (_split(second, 2.0), _split(first, 4.0))
    resent = [sent(b, (4.0, 2.0)) for b in stored]
    agreement = (1e-6, 1e-12)
    bars, rebases, whole = recognize(stored, resent, splits, agreement, NOW)  # type: ignore[misc]
    assert whole and [r.ex_date for r in rebases] == [first, second]
    factors = [0.125] * 4 + [0.5] * 4 + [1.0] * 2  # before both ex-dates, before the second
    assert [bar.close for bar in bars] == [
        b.close * f for b, f in zip(stored, factors, strict=True)
    ]
    gap = resent[:2] + resent[3:]  # a stored session the resend left out
    assert recognize(stored, gap, splits, agreement, NOW)[2] is False  # type: ignore[index]
    for ratios in ((3.0, 2.0), (4.0, 0.5)):  # a 3-for-1, a 1-for-2 reverse: rebased as well
        odd = (_split(second, ratios[1]), _split(first, ratios[0]))
        assert recognize(stored, [sent(b, ratios) for b in stored], odd, agreement, NOW)
    moved = list(resent)
    moved[2] = replace(moved[2], close=moved[2].close * 1.01)
    assert recognize(stored, moved, splits, agreement, NOW) is None
    assert recognize(stored, resent, (), agreement, NOW) is None
    assert recognize(stored, resent, splits, (0.0, 0.0), NOW) is None


@pytest.mark.parametrize(
    ("late", "crash", "ratio"),
    [(False, False, 2.0), (True, False, 2.0), (False, True, 2.0), (False, False, 1.5)],
)
def test_a_split_rebases_the_stored_bars_once_and_discloses_it(
    tmp_path: Path, late: bool, crash: bool, ratio: float
) -> None:
    """requirement: a split on the new session, or reported late on a stored one, keeps the stored
    bars times its factor, writes no correction per session and discloses one rebase, also when
    the process dies after the bars' commit and the cycle runs again."""
    from alphalattice.foundation.market_data_ops.storage.rebase import recorded

    manifest = build_quality_filtered_research_manifest(
        acquisition_manifest(), eligible_listing_ids=("listing-aapl",)
    )
    store = MarketDataRepository(tmp_path / "workspace")
    store.bootstrap(manifest)
    ex_date = SESSIONS[-5] if late else SESSIONS[-1]
    # As traded: the split halves the price from its ex-date; a late one was stored unreported.
    days = (*SESSIONS, SESSIONS[-1] + timedelta(1))  # and the next daily's session
    traded = {
        d: ratio * (100.0 + d.day / 7.0) if d < ex_date else 100.0 + d.day / 7.0 for d in days
    }

    def rows(start: date, end: date, scale: float, noise: float = 0.0) -> tuple[dict, ...]:
        return tuple(
            {"session_date": d.isoformat(), "open": c, "high": c * 1.01, "low": c * 0.99}
            | {"close": c, "volume": round(1e6 / s)}
            for d in days
            if start <= d <= end
            for s in [scale if d < ex_date else 1.0]
            for c in [traded[d] * s * (1 + noise * (d.day % 3 - 1) if d < ex_date else 1.0)]
        )

    store.apply_validated_batch(
        manifest,
        sanitize_payload(
            manifest, "fixture", {"AAPL": rows(SESSIONS[0], SESSIONS[-2], 1.0)}, ("AAPL",)
        ),
        ingestion_id="initial",
        observed_at=NOW,
    )
    audit_one(
        store, manifest, listing_id="listing-aapl", provider="fixture", actions=(),
        adjusted_closes=tuple(
            ProviderAdjustedClosePoint("listing-aapl", "fixture", d, traded[d])
            for d in SESSIONS[:-1]
        ),
        history_start=SESSIONS[0], history_end=SESSIONS[-2], requested_as_of=SESSIONS[-2],
        observed_at=NOW,
    )  # fmt: skip

    class Provider:
        name, noise = "fixture", 0.0

        def fetch_hydration(self, *, listing_id, provider_symbol, start, end):  # type: ignore[no-untyped-def]
            split = CorporateActionEvent(listing_id, "fixture", ex_date, "SPLIT", ratio)
            return HydrationEvidence(
                daily_rows=rows(start, end, 1 / ratio, self.noise),
                actions=(split,),
                adjusted_closes=tuple(
                    ProviderAdjustedClosePoint(
                        listing_id,
                        "fixture",
                        d,
                        traded[d] * ((1 + 3e-7 * (d.day % 3 - 1)) / ratio if d < ex_date else 1.0),
                    )
                    for d in days
                    if start <= d <= end
                ),
            )

    def run(escalate: frozenset[str], day: date = SESSIONS[-1], at: int = 1):  # type: ignore[no-untyped-def]
        return CurrentUniverseMaintenance(
            store=store, manifest=manifest, provider=Provider(), as_of_session=day,
            max_workers=1, mutation_gate=WorkspaceMutationGate(), adjusted_tolerance=1e-6,
            rebase_agreement=(1e-6, 1e-12), full_history_escalation_listing_ids=escalate,
        ).run(observed_at=NOW + timedelta(days=at))  # fmt: skip

    prior = {b.session_date: b for b in store.raw_bars("listing-aapl", through=SESSIONS[-1])}
    # The rebase reaches the full history, by approval; a retry, its new session stored, waits too.
    assert [run(frozenset()).failed for _ in range(2)] == [1, 1]
    if crash:
        apply = store.apply_validated_batch

        def die(*args, **kwargs):  # type: ignore[no-untyped-def]
            counts = apply(*args, **kwargs)
            if kwargs.get("rebases"):
                raise SystemExit("died after the bars' commit")
            return counts

        store.apply_validated_batch = die  # type: ignore[method-assign]
        with pytest.raises(SystemExit):
            run(frozenset({"listing-aapl"}))
        store.apply_validated_batch = apply  # type: ignore[method-assign]
    outcome = run(frozenset({"listing-aapl"}))
    assert outcome.status is CurrentUniverseMaintenanceStatus.COMPLETED and outcome.failed == 0
    stored = {b.session_date: b for b in store.raw_bars("listing-aapl", through=SESSIONS[-1])}
    for day, before in prior.items():
        scale = 1 / ratio if day < ex_date else 1.0
        assert (stored[day].close, stored[day].volume) == (
            before.close * scale,
            round(before.volume / scale),
        )
    assert store.revision_count("listing-aapl") == 0  # no correction a session
    # A build that recovers its changes since a point reads the split from its rebase row.
    (delta,) = store.raw_bar_semantic_deltas_since(manifest, observed_after=NOW, through=days[-2])
    assert delta.sessions == (ex_date,)
    with store.database.read_transaction() as connection:
        (rebase,) = recorded(connection, ("listing-aapl",))
        change = json.loads(
            connection.execute(
                "SELECT change_json FROM current_universe_maintenance_listing"
                " WHERE state = 'UPDATED'"
            ).fetchall()[-1][0]
        )
    assert (rebase.ex_date, rebase.price_factor, rebase.volume_factor) == (
        ex_date,
        1 / ratio,
        ratio,
    )
    assert change["raw_rebases"][0]["receipt_hash"] == rebase.receipt_hash  # the kept head's too
    # A restart finds the ex-date's bar stored by the run that died, as a late split's is.
    assert change["raw_correction_sessions"] == ([ex_date.isoformat()] if late or crash else [])
    # The adjusted closes keep the factor form: a late split's segments end at its ex-date.
    factors = [part[2] for part in change["adjustment_segments"]]
    assert factors[-1] == 1.0 if late else factors == [pytest.approx(1 / ratio, rel=1e-6)]
    Provider.noise = 1e-7  # the next daily's resend re-rounds the rescale it reported
    later = run(frozenset(), days[-1], 2)
    assert (later.status, later.failed) == (CurrentUniverseMaintenanceStatus.COMPLETED, 0)
    assert store.revision_count("listing-aapl") == 0  # the stored bars stand


def test_a_split_rebuilds_exactly_the_feature_cells_its_rescale_moves(tmp_path: Path) -> None:
    """The 3:1 then 1:2 cutoff rebuilds only reached Panel parts, matching cold.
    Two real cold materializations cover quarter locality and cost over ten seconds."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    from alphalattice.control.data_platform.maintenance.contracts import (
        ListingMarketDataChange,
        MarketDataChangeSet,
    )
    from alphalattice.control.data_platform.maintenance.invalidation import (
        FeatureInvalidationTopology,
    )
    from alphalattice.foundation.feature_engine.runtime.rebase_cutoff import rebase_moved
    from tests.feature_engine import panel_workspace as fixture

    build_and_publish, BUILT = fixture.build_and_publish, fixture.NOW
    manifest, as_of, agreement = fixture.fixture_manifest(), fixture.SESSIONS[300], (1e-6, 1e-12)
    listing = manifest.listings[0].listing_id
    build = {"spy": "a" * 64, "as_of": as_of, "observed_at": BUILT, "refresh_sector": False}
    prices = ", ".join(f"{n} = {n} * ?" for n in ("open", "high", "low", "close"))
    exact = manifest.listings[1].listing_id  # a 2-for-1 beside it: a power of two moves nothing

    def stored(space, sql: str, *values: object, name: str = listing) -> None:  # type: ignore[no-untyped-def]
        with space.market_data.database.transaction() as connection:
            connection.execute(sql, [*values, name])

    def split(space, ratio: float = 1.5, name: str = listing) -> None:  # type: ignore[no-untyped-def]
        price = 1 / ratio  # as the store rebases: prices by one over it, volumes by it, rounded
        bars = f"UPDATE raw_daily_bar_current SET {prices}, volume = floor(volume * ? + 0.5)"
        stored(space, bars + " WHERE listing_id = ?", *[price] * 4, ratio, name=name)
        stored(space, "UPDATE provider_adjusted_close_current SET adjusted_close ="
               " adjusted_close * ? WHERE listing_id = ?", price, name=name)  # fmt: skip

    def panel(space, published):  # type: ignore[no-untyped-def]
        chunks = space.market_data.workspace / "artifacts/feature-panel/chunks"
        frames = [pq.read_table(chunks / f"{c['chunk_hash']}.parquet") for c in published["chunks"]]
        frame = pd.concat(item.to_pandas() for item in frames).set_index(
            ["session_date", "listing_id"]
        )
        return frame.sort_index()[sorted(space.service.catalog.factor_ids)]

    spaces = [fixture.panel_workspace(tmp_path / name, manifest, end=as_of, spy="a" * 64)
              for name in ("kept", "cold")]  # fmt: skip
    for space in spaces:  # a thinly traded listing: a 3-for-2's whole-share rounding moves it
        stored(space, "UPDATE raw_daily_bar_current SET volume = CASE WHEN session_date >= ?"
               " THEN 101 + volume % 997 ELSE 2 * floor(volume / 2) END WHERE listing_id = ?",
               fixture.SESSIONS[80])  # fmt: skip
    workspace, cold = spaces
    topology = FeatureInvalidationTopology(workspace.service.catalog)
    _, kept = build_and_publish(workspace, manifest, **build | {"refresh_sector": True})
    for ratios in ((3.0,), (3.0, 0.5)):
        before = kept
        if len(ratios) == 2:
            cold = fixture.panel_workspace(
                tmp_path / "cold-next", manifest, end=as_of, spy="a" * 64
            )
            stored(cold, "UPDATE raw_daily_bar_current SET volume = CASE WHEN session_date >= ?"
                   " THEN 101 + volume % 997 ELSE 2 * floor(volume / 2) END WHERE listing_id = ?",
                   fixture.SESSIONS[80])  # fmt: skip
        split(workspace, ratios[-1])
        for ratio in ratios:
            split(cold, ratio)
        moved = rebase_moved(
            workspace.service, listing, as_of_session=as_of, observed_at=BUILT, agreement=agreement
        )
        if len(ratios) == 2:
            assert "volume_zscore_21" in dict(moved)  # the reverse rounds the odd share volumes
        change = ListingMarketDataChange(listing, rebase_moved=moved)
        cells = topology.plan(
            MarketDataChangeSet((change,)), history_start=fixture.SESSIONS[0], as_of_session=as_of
        ).invalidations  # fmt: skip
        if moved:
            _, kept = build_and_publish(workspace, manifest, **build, invalidations=cells)
        else:
            kept = before
        again = rebase_moved(
            workspace.service, listing, as_of_session=as_of, observed_at=BUILT, agreement=agreement
        )
        assert again == ()  # every stored cell now within the agreement of the cold computation
        _, fresh = build_and_publish(cold, manifest, **build | {"refresh_sector": True})
        held, rebuilt = panel(workspace, kept), panel(cold, fresh)
        spans = [
            (date.fromisoformat(c["first_session"]), date.fromisoformat(c["last_session"]))
            for c in before["chunks"]
            if any(
                c["first_session"] <= day.isoformat() <= c["last_session"]
                for _, days in moved
                for day in days
            )
        ]
        rewritten = [
            any(first <= day <= last for first, last in spans)
            for day in pd.to_datetime(held.index.get_level_values("session_date")).date
        ]
        left, right = held.to_numpy(float), rebuilt.to_numpy(float)
        assert {(c["first_session"], c["last_session"]) for c in before["chunks"]} == {
            (c["first_session"], c["last_session"]) for c in kept["chunks"]
        }
        untouched = [c for c in before["chunks"] if not any(
            first.isoformat() <= c["first_session"] <= last.isoformat() for first, last in spans
        )]  # fmt: skip
        assert all(chunk in kept["chunks"] for chunk in untouched)
        assert before["chunks"][0] in kept["chunks"]  # rounding starts in the following quarter
        assert np.array_equal(np.isnan(left), np.isnan(right))
        np.testing.assert_allclose(left[rewritten], right[rewritten], rtol=0, atol=1e-12)
        # Cells outside the rewritten parts still satisfy the agreement.
        assert np.nanmax(np.abs(left - right) / np.maximum(1.0, np.abs(right))) <= 1e-6
    split(workspace, 2.0, exact)  # isolate the exact-ratio control from the inexact split's Panel
    assert (
        rebase_moved(
            workspace.service, exact, as_of_session=as_of, observed_at=BUILT, agreement=agreement
        )
        == ()
    )


def test_a_kept_feature_gap_cannot_escape_through_a_low_dispersion_panel(tmp_path, monkeypatch):
    """One share of re-rounding is within Feature agreement but amplifies in a narrow Panel.
    Real stored history, cold publication and repair cost over ten seconds."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    from alphalattice.control.data_platform.maintenance import contracts, invalidation, rebase_reach
    from tests.feature_engine import panel_workspace as fixture

    manifest, day = fixture.fixture_manifest(), fixture.SESSIONS[300]
    listing, factor = manifest.listings[0].listing_id, "volume_zscore_21"
    volumes = pd.DataFrame(
        [
            (item.listing_id, session, 10**12 + (j % 31) * 10**10 + i * (j % 11) * 2)
            for i, item in enumerate(manifest.listings)
            for j, session in enumerate(fixture.SESSIONS[:301])
        ],
        columns=["listing_id", "session_date", "volume"],
    )
    spaces = [fixture.panel_workspace(tmp_path / name, manifest, end=day, spy="a" * 64)
              for name in ("kept", "cold")]  # fmt: skip
    for space in spaces:
        with space.market_data.database.transaction() as db:
            db.register("narrow", volumes)
            db.execute(
                "UPDATE raw_daily_bar_current b SET volume = n.volume FROM narrow n"
                " WHERE b.listing_id = n.listing_id AND b.session_date = n.session_date"
            )
            db.unregister("narrow")
    space, cold = spaces
    service = space.service
    build = dict(spy="a" * 64, as_of=day, observed_at=fixture.NOW, refresh_sector=True)
    _, held = fixture.build_and_publish(space, manifest, **build)
    for target in spaces:
        with target.market_data.database.transaction() as db:
            db.execute(
                "UPDATE raw_daily_bar_current SET volume = volume + 1"
                " WHERE listing_id = ? AND session_date = ?",
                [listing, day],
            )
    _, fresh = fixture.build_and_publish(cold, manifest, **build)

    def value(target, published):
        chunk = next(
            c
            for c in published["chunks"]
            if c["first_session"] <= day.isoformat() <= c["last_session"]
        )
        chunks = target.market_data.workspace / "artifacts/feature-panel/chunks"
        table = pq.read_table(
            chunks / f"{chunk['chunk_hash']}.parquet",
            columns=[factor],
            filters=[("listing_id", "=", listing), ("session_date", "=", day)],
        )
        return table.column(factor)[0].as_py(), target.feature_state.feature_rows(
            listing_ids=(listing,), catalog_hash=service.catalog.binding.catalog_hash,
            start=day, end=day, factor_ids=(factor,),
        )[0][factor]  # fmt: skip

    expected, expected_feature = value(cold, fresh)
    assert not np.isclose(value(space, held)[0], expected, rtol=2e-6, atol=1e-12)
    assert (
        rebase_reach.rebase_moved(
            service, listing, as_of_session=day, observed_at=fixture.NOW, agreement=(1e-6, 1e-12)
        )
        == ()
    )  # the Feature-only cutoff lawfully keeps every cell
    reads, saved = [], []
    original = space.feature_state.feature_rows

    def read(**kwargs):
        if len(kwargs["listing_ids"]) > 1:
            reads.append(kwargs["listing_ids"])
        return original(**kwargs)

    monkeypatch.setattr(space.feature_state, "feature_rows", read)
    monkeypatch.setattr(rebase_reach, "ex_date_deltas", lambda *args: [(listing,)])
    diagnose = SimpleNamespace(
        agreement=lambda: (1e-6, 1e-12),
        panel_agreement=lambda: (2e-6, 1e-12),
        panel_kept=lambda *args: saved.append(args),
    )
    owner = SimpleNamespace(
        diagnose=diagnose,
        market_data=space.market_data,
        manifest=manifest,
        feature_foundation=service,
        registry=SimpleNamespace(
            rebase_moved_since=lambda *args, **kwargs: {}, update=lambda *args, **kwargs: None
        ),
    )
    cycle = SimpleNamespace(cycle_id="c", request=SimpleNamespace(target_market_session=day))
    active = space.panel_state.active_feature_panel(manifest.profile.market_profile_id)
    changes = rebase_reach.rebase_reach(
        owner, cycle, contracts.MarketDataChangeSet(), active, fixture.NOW
    )
    assert changes
    assert day in dict(next(c for c in changes if c.listing_id == listing).rebase_moved)[factor]
    assert len(reads) == 1 and saved
    effects = (
        invalidation.FeatureInvalidationTopology(service.catalog)
        .plan(
            contracts.MarketDataChangeSet(changes),
            history_start=fixture.SESSIONS[0],
            as_of_session=day,
        )
        .invalidations
    )
    _, repaired = fixture.build_and_publish(
        space, manifest, **build | {"refresh_sector": False}, invalidations=effects
    )
    assert value(space, repaired) == (expected, expected_feature)


def test_a_splits_moved_cells_are_owed_across_a_crash_until_the_panel_holds_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Real Feature writes repair a split before cancellation; a later request must still
    rebuild the owed Panel cells, then retire them. The two real builds cost over ten seconds."""
    from datetime import UTC
    from types import SimpleNamespace

    from alphalattice.control.data_platform.maintenance import (
        contracts,
        invalidation,
    )  # fmt: skip
    from alphalattice.control.data_platform.maintenance import (
        rebase_reach as reach,
    )
    from alphalattice.control.data_platform.maintenance import (
        registry as maintenance_registry,
    )
    from alphalattice.foundation.feature_engine.contracts import (
        FeatureBuildRequest,
        FeatureBuildStatus,
    )
    from alphalattice.foundation.market_data_ops.storage.rebase import RawBarRebase, record
    from tests.feature_engine import panel_workspace as fixture

    manifest, as_of = fixture.fixture_manifest(), fixture.SESSIONS[300]
    start = fixture.SESSIONS[0]
    listing, agreement = manifest.listings[0].listing_id, (1e-6, 1e-12)
    listings = (listing, manifest.listings[1].listing_id)
    space = fixture.panel_workspace(tmp_path / "real", manifest, end=as_of, spy="a" * 64)
    store, service = space.market_data, space.service
    with store.database.transaction() as db:
        db.execute(
            "UPDATE raw_daily_bar_current SET volume = 101 + volume % 997 "
            "WHERE listing_id IN (?, ?)",
            listings,
        )
    build = dict(spy="a" * 64, as_of=as_of, observed_at=fixture.NOW, refresh_sector=True)
    _, initial = fixture.build_and_publish(space, manifest, **build)
    panel = space.panel_state.active_feature_panel(manifest.profile.market_profile_id)
    at = panel["activated_at"].replace(tzinfo=UTC) + timedelta.resolution
    with store.database.transaction() as db:
        db.execute(
            "UPDATE raw_daily_bar_current SET open = open * ?, high = high * ?, low = low * ?,"
            " close = close * ?, volume = floor(volume * ? + 0.5) WHERE listing_id IN (?, ?)"
            " AND session_date < ?",
            [*[2 / 3] * 4, 1.5, *listings, as_of],
        )
        db.execute(
            "UPDATE provider_adjusted_close_current SET adjusted_close = adjusted_close * ?"
            " WHERE listing_id IN (?, ?) AND session_date < ?",
            [2 / 3, *listings, as_of],
        )
        rebases = [RawBarRebase(
            name, space.provider.name, as_of, start, fixture.SESSIONS[299],
            2 / 3, 1.5, 0.0, at.replace(tzinfo=None),
        ) for name in listings]  # fmt: skip
        record(db, rebases, "fixture")
    registry = maintenance_registry.DuckDbWorkspaceMaintenanceRegistry(store.path, gate=space.gate)
    arguments = dict(
        market_profile_id=manifest.profile.market_profile_id, target_market_session=as_of,
        trigger=contracts.MaintenanceTrigger.STARTUP, membership_revision=manifest.revision_sha256,
        data_policy_hash="d" * 64, feature_policy_hash="e" * 64,
    )  # fmt: skip
    cycle = registry.admit(
        contracts.WorkspaceMaintenanceRequest.create(**arguments, knowledge_cutoff_at=at),
        observed_at=at,
    )
    owner = SimpleNamespace(
        diagnose=SimpleNamespace(agreement=lambda: agreement), market_data=store,
        manifest=manifest, feature_foundation=service, registry=registry,
    )  # fmt: skip
    loads, load = [], space.resolver.load_feature_panel_manifest

    def counted(uri):
        loads.append(uri)
        return load(uri)

    monkeypatch.setattr(space.resolver, "load_feature_panel_manifest", counted)
    reach.rebase_reach(owner, cycle, contracts.MarketDataChangeSet(), panel, at)
    assert len(loads) == 1  # the active descriptor is resolved once for both moved listings
    changes = registry.cycle(cycle.cycle_id).market_data_change_set
    owed = {c.listing_id: c.rebase_moved for c in changes.listing_changes}
    assert set(owed) == set(listings) and all(owed.values())
    cells = changes.listing_changes[0].rebase_moved
    assert cells
    topology = invalidation.FeatureInvalidationTopology(service.catalog)

    def plan(changes):
        return topology.plan(changes, history_start=start, as_of_session=as_of).invalidations

    request = FeatureBuildRequest.create(
        manifest_revision=manifest.revision_sha256, catalog=service.catalog.binding,
        spy_revision="a" * 64, history_start=start, as_of_session=as_of, invalidations=plan(changes)
    )  # fmt: skip
    outcome = service.build(request, observed_at=at, refresh_sector=False)
    assert outcome.status is FeatureBuildStatus.COMPLETED
    assert reach.rebase_moved(
        service, listing, as_of_session=as_of, observed_at=at, agreement=agreement
    ) == ()  # fmt: skip
    registry.cancel(cycle.cycle_id, observed_at=at)
    later_at = at + timedelta.resolution
    later = registry.admit(
        contracts.WorkspaceMaintenanceRequest.create(**arguments, knowledge_cutoff_at=later_at),
        observed_at=later_at,
    )
    assert later.cycle_id != cycle.cycle_id and later.market_data_change_set is None
    addition = contracts.ListingMarketDataChange(listing, new_sessions=(as_of,))
    resumed = reach.rebase_reach(
        owner, later, contracts.MarketDataChangeSet((addition,)), panel, later_at
    )
    assert {c.listing_id: c.rebase_moved for c in resumed} == owed
    change = next(c for c in resumed if c.listing_id == listing)
    assert (change.rebase_moved, change.new_sessions) == (cells, (as_of,))
    _, consumed = fixture.build_and_publish(
        space, manifest, **build | {"observed_at": later_at, "refresh_sector": False},
        invalidations=plan(contracts.MarketDataChangeSet(resumed)),
    )  # fmt: skip
    assert consumed["chunks"] != initial["chunks"]
    active = space.panel_state.active_feature_panel(manifest.profile.market_profile_id)
    assert reach.rebase_reach(owner, later, contracts.MarketDataChangeSet(), active, later_at) == ()
