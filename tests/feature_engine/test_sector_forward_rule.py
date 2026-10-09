"""The Sector's forward rule: each session reads the classification in force on it."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import (
    SectorRevisionEntry,
    SectorRevisionMap,
    durable_payload,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_contracts import (
    SectorRevisionMapActivationReceipt,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    copy_sector_history,
    sector_history_as_of,
)
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.sources.sector_forward import (
    sector_effective_session,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.quant.sector_history import (
    SectorHistory,
    SectorReclassification,
    exposure_sector_ids,
    reclassification_payload,
    sector_exposure,
    sector_slices,
    sector_subset,
    sector_treatment_of,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sector_treatment import (
    SECTOR_HISTORY_BACKFILLED,
    SECTOR_HISTORY_FORWARD,
)
from tests.workspace_maintenance.acquisition_manifest import acquisition_manifest

MONDAY, TUESDAY, WEDNESDAY, THURSDAY = (date(2026, 8, day) for day in (3, 4, 5, 6))


def _history() -> SectorHistory:
    return SectorHistory(
        current_revision="r" * 64,
        current={"a": "Energy", "b": "Technology", "c": "Technology"},
        reclassifications=(SectorReclassification("a", WEDNESDAY, "Technology", "Energy"),),
    )


def test_a_history_reads_each_session_the_map_in_force() -> None:
    """Requirement: a listing reads its first recorded Sector until its reclassification
    and the new one from its effective session; a plain map is one run and binds as it did."""

    history = _history()
    assert dict(history) == {"a": "Energy", "b": "Technology", "c": "Technology"}
    assert history.base["a"] == "Technology" and history.at(TUESDAY)["a"] == "Technology"
    assert history.at(WEDNESDAY)["a"] == "Energy" and history.sectors == ("Energy", "Technology")
    sessions = (MONDAY, TUESDAY, WEDNESDAY, THURSDAY)
    runs = history.slices(sessions)
    assert [(rows.start, rows.stop, mapping["a"]) for rows, mapping in runs] == [
        (0, 2, "Technology"),
        (2, 4, "Energy"),
    ]
    with pytest.raises(ValueError, match=r"sector_history.axis_unordered"):
        history.slices((TUESDAY, MONDAY))
    plain = {"a": "Energy", "b": "Technology"}
    assert sector_slices(plain, sessions) == ((slice(0, 4), plain),)
    assert reclassification_payload(plain) == {} and sector_treatment_of(plain) == (
        SECTOR_HISTORY_BACKFILLED
    )
    assert sector_treatment_of(history) == SECTOR_HISTORY_FORWARD
    assert reclassification_payload(history)["sector_reclassifications"][0]["prior_sector"] == (
        "Technology"
    )
    unmoved = sector_subset(history, ("b", "c"))
    assert isinstance(unmoved, SectorHistory) and unmoved.reclassifications == ()
    assert unmoved.identity == "r" * 64 and history.identity != "r" * 64
    for broken in (
        (SectorReclassification("a", WEDNESDAY, "Technology", "Technology"),),
        (
            SectorReclassification("a", WEDNESDAY, "Health", "Energy"),
            SectorReclassification("a", MONDAY, "Technology", "Health"),
        ),
        (SectorReclassification("z", WEDNESDAY, "Technology", "Energy"),),
    ):
        with pytest.raises(ValueError, match=r"sector_history.invalid"):
            SectorHistory(
                current_revision="r" * 64, current=dict(history), reclassifications=broken
            )


def test_a_history_exposes_one_matrix_per_session_only_where_a_reclassification_falls() -> None:
    """Requirement: the Portfolio's exposure is one matrix over a plain map and one per
    session, over every Sector a session reads, where a reclassification falls inside the axis."""

    listings = ("a", "b", "c")
    sessions = (MONDAY, TUESDAY, WEDNESDAY)
    plain = sector_exposure(
        {"a": "Energy", "b": "Technology", "c": "Technology"},
        sessions,
        listings,
        ("Energy", "Technology"),
    )
    assert plain.shape == (2, 3) and not plain.flags.writeable
    history = _history()
    sectors = exposure_sector_ids(history, sessions, listings)
    assert sectors == ("Energy", "Technology")
    stacked = sector_exposure(history, sessions, listings, sectors)
    assert stacked.shape == (3, 2, 3)
    assert stacked[0].tolist() == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert stacked[2].tolist() == plain.tolist()
    assert sector_exposure(history, (MONDAY, TUESDAY), listings, sectors).shape == (2, 3)


def test_a_reclassification_takes_effect_from_the_first_unpublished_session() -> None:
    """Requirement: the trading day its update observed it, the next session when that
    day has none, and never a session a Panel already published."""

    friday_evening = datetime(2026, 8, 7, 22, tzinfo=UTC)  # 18:00 in New York, a Friday
    assert sector_effective_session(friday_evening, first_unpublished=MONDAY) == date(2026, 8, 7)
    saturday = datetime(2026, 8, 8, 15, tzinfo=UTC)
    assert sector_effective_session(saturday, first_unpublished=MONDAY) == date(2026, 8, 10)
    before_the_horizon = datetime(2026, 8, 3, 15, tzinfo=UTC)
    assert sector_effective_session(before_the_horizon, first_unpublished=THURSDAY) == THURSDAY
    with pytest.raises(ValueError, match=r"sector_history.effective_session_unavailable"):
        sector_effective_session(datetime(2026, 8, 3, 15), first_unpublished=MONDAY)


def _observations(manifest, sectors: dict[str, str]) -> tuple[dict[str, object], ...]:  # type: ignore[no-untyped-def]
    return tuple(
        {
            "listing_id": listing.listing_id,
            "provider": "fixture",
            "provider_symbol": listing.provider_symbol,
            "sector_name": sectors[listing.listing_id],
            "sector_key": None,
            "payload_hash": canonical_hash(
                ["payload", listing.listing_id, sectors[listing.listing_id]]
            ),
            "evidence_hash": canonical_hash(["evidence", listing.listing_id]),
        }
        for listing in manifest.listings
    )


def _publish_through(feature_state: FeatureStateRepository, manifest, session: date) -> None:  # type: ignore[no-untyped-def]
    connection = feature_state._connect()
    try:
        connection.execute(
            """
            INSERT INTO active_feature_panel_binding (
                market_profile_id, manifest_revision, sector_revision, catalog_hash,
                spy_revision, policy_hash, panel_hash, as_of_session, activated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                manifest.profile.market_profile_id,
                manifest.revision_sha256,
                "s" * 64,
                "c" * 64,
                "y" * 64,
                "p" * 64,
                "h" * 64,
                session,
                datetime(2026, 8, 3, tzinfo=UTC),
            ],
        )
    finally:
        connection.close()


def test_a_planted_reclassification_is_read_forward_and_never_moves_a_published_session(
    tmp_path: Path,
) -> None:
    """Requirement: a refresh that reclassifies a listing after a Panel published through
    Wednesday records it from Thursday on; every session through Wednesday reads what it read."""

    manifest = acquisition_manifest()
    market = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market.database, market_data=market)
    PanelStateRepository(market.database, market_data=market)
    market.bootstrap(manifest)
    first = {"listing-aapl": "Technology", "listing-msft": "Technology"}
    revision, changed, _receipt, effective = feature_state.activate_sector_revision(
        manifest=manifest,
        observations=_observations(manifest, first),
        observed_at=datetime(2026, 8, 3, 21, tzinfo=UTC),
    )
    assert effective is None
    before = feature_state.sector_history(manifest)
    assert before is not None and before.reclassifications == ()
    assert before.identity == revision

    _publish_through(feature_state, manifest, WEDNESDAY)
    moved = {"listing-aapl": "Technology", "listing-msft": "Communication Services"}
    later, changed, _receipt, effective = feature_state.activate_sector_revision(
        manifest=manifest,
        observations=_observations(manifest, moved),
        observed_at=datetime(2026, 8, 4, 21, tzinfo=UTC),  # a Tuesday, before the horizon
    )
    assert changed and effective == THURSDAY
    history = feature_state.sector_history(manifest)
    assert history is not None and history.current_revision == later
    assert history.reclassifications == (
        SectorReclassification("listing-msft", THURSDAY, "Technology", "Communication Services"),
    )
    for published in (MONDAY, TUESDAY, WEDNESDAY):
        assert history.at(published) == first
    assert history.at(THURSDAY) == moved
    assert history.identity != later


def test_a_store_sealed_before_the_forward_rule_reads_no_reclassification(
    tmp_path: Path,
) -> None:
    """A store sealed before the forward rule reads no reclassification."""

    import duckdb

    manifest = acquisition_manifest()
    market = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market.database, market_data=market)
    PanelStateRepository(market.database, market_data=market)
    market.bootstrap(manifest)
    first = {"listing-aapl": "Technology", "listing-msft": "Technology"}
    revision, _changed, _receipt, _effective = feature_state.activate_sector_revision(
        manifest=manifest,
        observations=_observations(manifest, first),
        observed_at=datetime(2026, 8, 3, 21, tzinfo=UTC),
    )
    with duckdb.connect(str(market.database.path)) as connection:
        for column in ("effective_session", "sector_name", "prior_sector_name"):
            connection.execute(f"ALTER TABLE sector_classification_revision DROP COLUMN {column}")
    history = feature_state.sector_history(manifest)
    assert history is not None and history.current_revision == revision
    assert history.reclassifications == ()
    assert history.at(MONDAY) == first


def _map(revision: str, sectors: dict[str, str]) -> SectorRevisionMap:
    entries = tuple(
        SectorRevisionEntry(
            listing_id=listing,
            provider="fixture",
            provider_symbol=listing.upper(),
            sector_name=sector,
            sector_key=None,
            payload_hash=canonical_hash(["payload", listing, sector]),
            evidence_hash=canonical_hash(["evidence", listing]),
        )
        for listing, sector in sorted(sectors.items())
    )
    draft = SectorRevisionMap.model_construct(
        manifest_revision="d" * 64, sector_revision=revision, entries=entries, map_hash="0" * 64
    )
    return SectorRevisionMap(
        manifest_revision="d" * 64,
        sector_revision=revision,
        entries=entries,
        map_hash=canonical_hash(draft.model_dump(mode="json", exclude={"map_hash"})),
    )


def _receipt(
    sector_map: SectorRevisionMap, observed: datetime, effective: date | None
) -> SectorRevisionMapActivationReceipt:
    values: dict[str, object] = {
        "manifest_revision": sector_map.manifest_revision,
        "sector_revision": sector_map.sector_revision,
        "sector_map_hash": sector_map.map_hash,
        "store_receipt_hash": canonical_hash(["store", sector_map.map_hash]),
        "changed": True,
        "observed_at": observed,
        "effective_session": effective,
    }
    draft = SectorRevisionMapActivationReceipt.model_construct(**values, receipt_hash="0" * 64)
    return SectorRevisionMapActivationReceipt(
        **values,
        receipt_hash=canonical_hash(draft.model_dump(mode="json", exclude={"receipt_hash"})),
    )


def test_the_ledger_rebuilds_a_history_and_folds_what_came_before_the_rule(
    tmp_path: Path,
) -> None:
    """Requirement: the history as of a revision is its map, then the changes its
    activation receipts recorded with an effective session; a receipt written before the rule
    folds into the backfill; a copy of what it reads answers the same history elsewhere."""

    store = PanelClosureArtifactStore(ArtifactResolver(tmp_path / "artifacts"))
    one = _map("1" * 64, {"a": "Technology", "b": "Technology"})
    two = _map("2" * 64, {"a": "Energy", "b": "Technology"})
    three = _map("3" * 64, {"a": "Energy", "b": "Health"})
    for value in (one, two, three):
        store.publish_json(
            category="sector-maps", content_hash=value.map_hash, payload=durable_payload(value)
        )
    receipts = (
        _receipt(one, datetime(2026, 8, 3, 21, tzinfo=UTC), None),
        _receipt(two, datetime(2026, 8, 4, 21, tzinfo=UTC), None),
        _receipt(three, datetime(2026, 8, 5, 21, tzinfo=UTC), THURSDAY),
    )
    for value in receipts:
        store.publish_json(
            category="sector-activation-receipts",
            content_hash=value.receipt_hash,
            payload=durable_payload(value),
        )
    folded = sector_history_as_of(store, two.sector_revision)
    assert folded.reclassifications == () and dict(folded) == {"a": "Energy", "b": "Technology"}
    forward = sector_history_as_of(store, three.sector_revision)
    assert forward.reclassifications == (
        SectorReclassification("b", THURSDAY, "Technology", "Health"),
    )
    assert forward.at(WEDNESDAY) == {"a": "Energy", "b": "Technology"}
    copy = PanelClosureArtifactStore(ArtifactResolver(tmp_path / "copy"))
    assert copy_sector_history(store, copy, three.sector_revision) == forward
    assert sector_history_as_of(copy, three.sector_revision) == forward


def test_the_forward_treatment_is_stated_only_where_a_reclassification_is_in_force() -> None:
    """Requirement: the Alpha lane policy, the Sector context policy and the Sector
    target recipe keep the identity they had while the backfill is what their sessions read."""

    from alphalattice.investment.alpha_research.targets.execution_outcome import (
        AlphaTargetLane,
        build_alpha_target_policy,
    )
    from alphalattice.investment.sector_research.inputs.surface import (
        build_sector_context_policy,
    )
    from alphalattice.investment.sector_research.targets.execution import (
        build_sector_target_recipe,
    )

    revision = "e" * 64
    for lane in AlphaTargetLane:
        backfilled = build_alpha_target_policy(lane=lane, sector_revision=revision)
        stated = build_alpha_target_policy(
            lane=lane, sector_revision=revision, sector_history_treatment=SECTOR_HISTORY_BACKFILLED
        )
        forward = build_alpha_target_policy(
            lane=lane, sector_revision=revision, sector_history_treatment=SECTOR_HISTORY_FORWARD
        )
        assert stated.policy_hash == backfilled.policy_hash != forward.policy_hash
        assert forward.sector_history_treatment == SECTOR_HISTORY_FORWARD
    assert (
        build_sector_context_policy(sector_revision=revision).policy_hash
        != build_sector_context_policy(
            sector_revision=revision, sector_history_treatment=SECTOR_HISTORY_FORWARD
        ).policy_hash
    )
    recipe = build_sector_target_recipe(
        execution_outcome_recipe_id="outcome",
        sector_revision=revision,
        sector_history_treatment=SECTOR_HISTORY_FORWARD,
    )
    assert recipe.sector_history_treatment == SECTOR_HISTORY_FORWARD


def test_a_published_session_keeps_its_panel_cross_section(tmp_path: Path) -> None:
    """Requirement: a Panel's cross-section identity for each session before the
    reclassification's effective session is the one its published map gave it."""

    from alphalattice.foundation.feature_engine.contracts import (
        PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
        PanelMembership,
        PanelMembershipBasisRange,
        PanelMembershipEpoch,
    )
    from alphalattice.foundation.feature_engine.panels.artifacts import (
        PanelArtifactCompositionOwner,
        PanelCompositionBinding,
    )

    sessions = (MONDAY, TUESDAY, WEDNESDAY, THURSDAY)
    membership = PanelMembership(
        listing_ids=("a", "b", "c"),
        epochs=(PanelMembershipEpoch(MONDAY, THURSDAY, ("a", "b", "c")),),
        basis_ranges=(PanelMembershipBasisRange(MONDAY, THURSDAY, "TEST"),),
    )
    published = SectorHistory(
        current_revision="r" * 64,
        current={"a": "Technology", "b": "Technology", "c": "Technology"},
    )
    moved = SectorHistory(
        current_revision="q" * 64,
        current={"a": "Energy", "b": "Technology", "c": "Technology"},
        reclassifications=(SectorReclassification("a", THURSDAY, "Technology", "Energy"),),
    )

    owner = PanelArtifactCompositionOwner(ArtifactResolver(tmp_path / "artifacts"))
    binding = PanelCompositionBinding(
        manifest_revision="m" * 64,
        sector_revision="s" * 64,
        catalog_hash="c" * 64,
        policy_hash="p" * 64,
        panel_binding_hash="b" * 64,
        history_start=MONDAY,
        as_of_session=THURSDAY,
        factor_ids=("factor_a",),
        row_identity_basis=PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
    )

    def by_session(history: SectorHistory) -> dict[date, str]:
        composition = owner.begin(
            operation_id="v346",
            binding=binding,
            base_manifest=None,
            sessions=sessions,
            listing_ids=membership.listing_ids,
            membership=membership,
            sector_history=history,
        )
        return {
            session: item.cross_section_identity
            for item in composition.all_cross_sections()
            for session in sessions
            if item.first_session <= session <= item.last_session
        }

    before, after = by_session(published), by_session(moved)
    assert [after[session] for session in sessions[:3]] == [
        before[session] for session in sessions[:3]
    ]
    assert after[THURSDAY] != before[THURSDAY]


def test_each_member_reads_its_days_sector_return() -> None:
    """Requirement: `sector_return_log` is the equal-weight one-session log return of
    the day's members of the Sector each reads that day; a non-member and the first session
    have none."""

    from alphalattice.foundation.feature_engine.producers.sector_aggregates import (
        sector_return_log,
    )

    sessions = (MONDAY, TUESDAY, WEDNESDAY, THURSDAY)
    closes = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [11.0, 20.0, 33.0, 40.0],
            [11.0, 22.0, 33.0, 44.0],
            [12.1, 22.0, 36.3, 44.0],
        ]
    )
    history = SectorHistory(
        current_revision="r" * 64,
        current={"a": "X", "b": "X", "c": "Y", "d": "Y"},
        reclassifications=(SectorReclassification("c", WEDNESDAY, "X", "Y"),),
    )
    members = np.ones((4, 4), dtype=np.bool_)
    members[3, 1] = False
    values = sector_return_log(
        closes,
        sessions=sessions,
        listing_ids=("a", "b", "c", "d"),
        sectors=history,
        members=members,
    )
    step = np.log(1.1)
    assert np.isnan(values[0]).all()
    np.testing.assert_allclose(values[1], [2 * step / 3] * 3 + [0.0])
    np.testing.assert_allclose(values[2], [step / 2] * 4)
    np.testing.assert_allclose(values[3, [0, 2, 3]], [step, step / 2, step / 2])
    assert np.isnan(values[3, 1]) and not values.flags.writeable
    with pytest.raises(ValueError, match=r"feature_engine.sector_return_axis_invalid"):
        sector_return_log(
            closes[:3], sessions=sessions, listing_ids=("a", "b", "c", "d"), sectors=history
        )


def test_the_universe_centred_recipe_reads_no_sector() -> None:
    """Requirement: `ROBUST_UNIVERSE_Z` is installed for development overlays and a
    formula's choice, and its values are the same under any classification."""

    from alphalattice.foundation.feature_engine.contracts import FeaturePanelBinding
    from alphalattice.foundation.feature_engine.producers.factors.formula import (
        formula_preprocessing,
    )
    from alphalattice.foundation.feature_engine.producers.preprocessing.catalog import (
        ROBUST_UNIVERSE_Z,
        build_installed_panel_preprocessing_catalog,
    )

    catalog = build_installed_panel_preprocessing_catalog()
    capability = catalog.capability(ROBUST_UNIVERSE_Z)
    assert capability.admitted_for_development_overlay
    assert not capability.admitted_for_active_panel
    assert formula_preprocessing(ROBUST_UNIVERSE_Z) == ROBUST_UNIVERSE_Z
    _recipe, adapter = catalog.resolve_executable(ROBUST_UNIVERSE_Z)
    generator = np.random.default_rng(346)
    listings = tuple(f"l{index:02d}" for index in range(12))
    rows = [
        {"session_date": session, "listing_id": listing, "factor_a": float(generator.normal())}
        for session in (MONDAY, TUESDAY)
        for listing in listings
    ]
    binding = FeaturePanelBinding.create(
        manifest_revision="1" * 64,
        sector_revision="2" * 64,
        catalog_hash="3" * 64,
        spy_revision="4" * 64,
        policy_hash="5" * 64,
    )
    split = {listing: ("X" if index < 6 else "Y") for index, listing in enumerate(listings)}
    one = {listing: "X" for listing in listings}
    first = adapter.materialize(
        feature_rows=rows,
        active_listing_ids=listings,
        sector_by_listing_id=split,
        factor_ids=("factor_a",),
        binding=binding,
    )
    second = adapter.materialize(
        feature_rows=rows,
        active_listing_ids=listings,
        sector_by_listing_id=one,
        factor_ids=("factor_a",),
        binding=binding,
    )
    assert first.transformed_identity == second.transformed_identity
    assert np.isfinite(first.rows["factor_a"].to_numpy(dtype=float)).any()


def test_a_risk_surface_designs_a_formation_by_the_sectors_in_force() -> None:
    """Requirement: a formation where a listing reads an earlier Sector designs its
    factor exposure by that Sector and binds it; one reading only current Sectors is as it was."""

    from alphalattice.investment.risk_research.surfaces.producer import (
        RiskDecompositionInputs,
        RiskSurfaceProducer,
    )

    entries = tuple(
        SectorRevisionEntry(
            listing_id=listing,
            provider="fixture",
            provider_symbol=listing.upper(),
            sector_name=sector,
            sector_key=key,
            payload_hash=canonical_hash(["payload", listing]),
            evidence_hash=canonical_hash(["evidence", listing]),
        )
        for listing, sector, key in (
            ("a", "Energy", "energy"),
            ("b", "Technology", "technology"),
            ("c", "Technology", "technology"),
        )
    )
    draft = SectorRevisionMap.model_construct(
        manifest_revision="d" * 64, sector_revision="e" * 64, entries=entries, map_hash="0" * 64
    )
    classification = SectorRevisionMap(
        manifest_revision="d" * 64,
        sector_revision="e" * 64,
        entries=entries,
        map_hash=canonical_hash(draft.model_dump(mode="json", exclude={"map_hash"})),
    )
    listings = ("a", "b", "c")
    # The installed recipe's warm-up and fit windows: 63 + 504 sessions.
    sessions = tuple(date.fromordinal(730_000 + index) for index in range(567))
    returns = np.random.default_rng(346).normal(0.0, 0.01, (567, len(listings)))

    formation = date.fromordinal(730_568)

    def surface(moves: tuple[SectorReclassification, ...]):  # type: ignore[no-untyped-def]
        current = {"a": "Energy", "b": "Technology", "c": "Technology"}
        history = SectorHistory("e" * 64, current, moves)
        inputs = RiskDecompositionInputs.at(
            formation, sessions, listings, returns, classification, "3" * 64, history
        )
        return RiskSurfaceProducer().produce(inputs).surface

    later = SectorReclassification("a", date.fromordinal(730_569), "Technology", "Energy")
    current, earlier = surface(()), surface((later,))
    assert current.ordered_factor_ids == ("energy", "technology")
    assert current.exposures.tolist() == [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    assert earlier.ordered_factor_ids == ("technology",)
    assert earlier.exposures.tolist() == [[1.0], [1.0], [1.0]]
