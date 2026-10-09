"""The Panel kernel over per-session members: rows for members only, bits by epoch."""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from alphalattice.foundation.feature_engine.contracts import (
    FeaturePanelBinding,
    PanelMembership,
    PanelSourceExclusion,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section import (
    SESSION_LOCAL_POLICY_HASH,
    SOURCE_ELIGIBILITY_POLICY_HASH,
    PanelCrossSectionKernel,
    cross_section_policy_hash,
)

FACTORS = ("factor_a", "factor_b")
SECTORS = ("Energy", "Health", "Tech")


def _axis(count: int) -> tuple[str, ...]:
    # Calculation order deliberately differs from lexical listing order.
    return tuple(f"L{index:02d}" for index in reversed(range(count)))


def _sectors(axis: tuple[str, ...]) -> dict[str, str]:
    return {listing: SECTORS[index % len(SECTORS)] for index, listing in enumerate(axis)}


def _rows(axis: tuple[str, ...], sessions: tuple[date, ...], *, seed: int = 7) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    records = []
    for session in sessions:
        for listing in axis:
            records.append(
                {
                    "session_date": session,
                    "listing_id": listing,
                    "factor_a": float(generator.normal()),
                    "factor_b": float(generator.normal() * 3.0 + 1.0),
                }
            )
    return pd.DataFrame(records)


def _binding(*, current: bool = False) -> FeaturePanelBinding:
    return FeaturePanelBinding.create(
        manifest_revision="1" * 64,
        sector_revision="2" * 64,
        catalog_hash="3" * 64,
        spy_revision="4" * 64,
        policy_hash=cross_section_policy_hash() if current else "5" * 64,
    )


SESSIONS = tuple(date(2026, 1, 5) + timedelta(days=offset) for offset in range(8))


def test_legacy_replay_keeps_single_row_epochs_when_the_retained_slice_merges_them(tmp_path):
    from types import SimpleNamespace

    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
        PanelClosureArtifactStore,
    )
    from alphalattice.foundation.feature_engine.panels.rematerialization import (
        ArtifactOnlyPanelRematerializer,
    )
    from alphalattice.foundation.feature_engine.producers.cross_section import (
        clip_observation_record,
    )

    axis = _axis(60)
    sectors = _sectors(axis)
    sessions = SESSIONS[:3]
    rows = _rows(axis, sessions)
    membership = {sessions[0]: axis, sessions[1]: axis[3:], sessions[2]: axis}
    kernel = PanelCrossSectionKernel()
    original = kernel.materialize(
        feature_rows=rows,
        active_listing_ids=axis,
        sector_by_listing_id=sectors,
        factor_ids=FACTORS,
        binding=_binding(),
        members_by_session=membership,
    )
    retained = (sessions[0], sessions[-1])
    resolver = ArtifactResolver(tmp_path / "artifacts")
    record = clip_observation_record(original)
    resolver.publish_panel_clip_observation(
        payload=record.model_dump(mode="json"), receipt_hash=record.receipt_hash
    )
    # The current closure has gained a later entrant. Its earlier recorded
    # union must reproduce the old batch's axis hash, without guessing a roster.
    future = SESSIONS[3]
    current_members = {**membership, future: (*axis, "later")}
    recipe = SimpleNamespace(
        listing_ids=(*axis, "later"),
        membership=SimpleNamespace(
            epochs=tuple(SimpleNamespace(first_session=day) for day in (*sessions, future))
        ),
        members=current_members.__getitem__,
    )
    attributed = [
        {**row, "materialization_receipt_hash": original.receipt_hash}
        for row in original.availability
        if date.fromisoformat(row["session_date"]) in retained
    ]
    reader = ArtifactOnlyPanelRematerializer(
        resolver=resolver, store=PanelClosureArtifactStore(resolver)
    )
    layout = reader._legacy_replay_layout(recipe, attributed, _binding())
    assert layout == {day: 1 for day in retained}
    replay = kernel.materialize(
        feature_rows=rows.loc[rows.session_date.isin(retained)],
        active_listing_ids=axis,
        sector_by_listing_id=sectors,
        factor_ids=FACTORS,
        binding=_binding(),
        members_by_session={day: axis for day in retained},
        legacy_epoch_session_counts=layout,
    )
    expected = original.rows.loc[
        original.rows.session_date.isin([day.isoformat() for day in retained])
    ]
    np.testing.assert_array_equal(
        expected[list(FACTORS)].to_numpy().view("u8"),
        replay.rows[list(FACTORS)].to_numpy().view("u8"),
    )


def test_dated_quality_exclusion_preserves_nominal_members_and_old_reference_bits() -> None:
    from dataclasses import replace

    axis = _axis(60)
    sectors = _sectors(axis)
    rows = _rows(axis, SESSIONS)
    excluded = axis[0]
    kernel = PanelCrossSectionKernel()
    inputs = dict(
        feature_rows=rows, active_listing_ids=axis, sector_by_listing_id=sectors, factor_ids=FACTORS
    )
    prior = kernel.materialize(
        **inputs,
        binding=FeaturePanelBinding.create(
            manifest_revision="1" * 64,
            sector_revision="2" * 64,
            catalog_hash="3" * 64,
            spy_revision="4" * 64,
            policy_hash=SESSION_LOCAL_POLICY_HASH,
        ),
    )
    normal = kernel.materialize(**inputs, binding=_binding(current=True))
    np.testing.assert_array_equal(
        prior.rows[list(FACTORS)].to_numpy().view("u8"),
        normal.rows[list(FACTORS)].to_numpy().view("u8"),
    )
    member = PanelMembership.uniform(
        axis, first_session=SESSIONS[0], last_session=SESSIONS[-1], basis="FORWARD_AS_OBSERVED"
    )
    qualified = replace(
        member,
        source_exclusions=(
            PanelSourceExclusion(
                excluded,
                SESSIONS[4],
                SESSIONS[5],
                "a" * 64,
                ("SOURCE_QUALITY_UNAVAILABLE",),
            ),
        ),
    )
    masked = kernel.materialize(
        **inputs,
        binding=_binding(current=True),
        members_by_session=qualified.members_by_session(SESSIONS),
        source_exclusions_by_session={
            session: qualified.excluded_sources(session) for session in SESSIONS
        },
    )
    assert len(masked.rows) == len(normal.rows) == 60 * len(SESSIONS)
    selected = masked.rows["session_date"].isin([session.isoformat() for session in SESSIONS[4:6]])
    np.testing.assert_array_equal(
        masked.rows.loc[~selected, list(FACTORS)].to_numpy().view("u8"),
        normal.rows.loc[~selected, list(FACTORS)].to_numpy().view("u8"),
    )
    assert (
        masked.rows.loc[selected & masked.rows["listing_id"].eq(excluded), list(FACTORS)]
        .isna()
        .all()
        .all()
    )
    reference_axis = tuple(item for item in axis if item != excluded)
    reference = kernel.materialize(
        feature_rows=rows.loc[
            rows["session_date"].isin(SESSIONS[4:6]) & rows["listing_id"].ne(excluded)
        ],
        active_listing_ids=reference_axis,
        sector_by_listing_id=sectors,
        factor_ids=FACTORS,
        binding=_binding(current=True),
    )
    np.testing.assert_array_equal(
        masked.rows.loc[selected & masked.rows["listing_id"].ne(excluded), list(FACTORS)]
        .to_numpy()
        .view("u8"),
        reference.rows[list(FACTORS)].to_numpy().view("u8"),
    )
    assert all(
        item["universe_size"] == 60 and item["computed_count"] == 59
        for item in masked.availability
        if item["session_date"] == SESSIONS[4].isoformat()
    )
    assert qualified.members(SESSIONS[4]) == axis
    assert PanelMembership.from_payload(qualified.to_payload()) == qualified
    assert member.cross_sections(SESSIONS[:4], sectors) == qualified.cross_sections(
        SESSIONS[:4], sectors
    )
    assert member.cross_sections(SESSIONS[4:6], sectors) != qualified.cross_sections(
        SESSIONS[4:6], sectors
    )
    relabelled = replace(
        qualified,
        source_exclusions=(replace(qualified.source_exclusions[0], evidence_hash="b" * 64),),
    )
    assert qualified.cross_sections(SESSIONS, sectors) == relabelled.cross_sections(
        SESSIONS, sectors
    )


def test_epoch_rows_are_the_bits_of_a_panel_over_exactly_those_members() -> None:
    """requirement: a session's numbers depend on its members and nothing else.

    A Panel over a 24-name axis where three names leave after the fourth
    session must produce, for every session, exactly the rows a Panel whose
    whole axis was that session's member set produces over the same sessions
    -- bit for bit -- because that is what lets a partition computed before
    the exit be reused after it, and one computed after the exit be reused by
    a build whose axis has since grown again. "Over the same sessions" is
    deliberate: numpy reduces a one-session block and a many-session block in
    different orders, so a session computed alone differs in its last bits
    from the same session inside a batch (a property of the kernel's sector
    mean that predates membership epochs and never enters reuse, which is by
    identity, not by recomputation).
    """

    axis = _axis(24)
    sectors = _sectors(axis)
    leavers = ("L05", "L11", "L17")
    stayers = tuple(listing for listing in axis if listing not in leavers)
    rows = _rows(axis, SESSIONS)
    kernel = PanelCrossSectionKernel()
    members = {session: axis if index < 4 else stayers for index, session in enumerate(SESSIONS)}
    ragged = kernel.materialize(
        feature_rows=rows,
        active_listing_ids=axis,
        sector_by_listing_id=sectors,
        factor_ids=FACTORS,
        binding=_binding(),
        members_by_session=members,
    )
    before = kernel.materialize(
        feature_rows=rows.loc[rows["session_date"].isin(SESSIONS[:4])],
        active_listing_ids=axis,
        sector_by_listing_id=sectors,
        factor_ids=FACTORS,
        binding=_binding(),
    )
    after = kernel.materialize(
        feature_rows=rows.loc[
            rows["session_date"].isin(SESSIONS[4:]) & rows["listing_id"].isin(stayers)
        ],
        active_listing_ids=stayers,
        sector_by_listing_id={listing: sectors[listing] for listing in stayers},
        factor_ids=FACTORS,
        binding=_binding(),
    )
    # Row set: members only, no padded rows for the leavers after the exit.
    assert len(ragged.rows) == 4 * 24 + 4 * 21
    after_exit = ragged.rows.loc[ragged.rows["session_date"] >= SESSIONS[4].isoformat()]
    assert not set(after_exit["listing_id"]).intersection(leavers)
    expected = pd.concat([before.rows, after.rows], ignore_index=True)
    keys = ["session_date", "listing_id"]
    observed = ragged.rows.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, expected, check_exact=True)
    # Availability describes each session over its own members.
    by_key = {(item["session_date"], item["factor_id"]): item for item in ragged.availability}
    assert by_key[(SESSIONS[0].isoformat(), "factor_a")]["universe_size"] == 24
    assert by_key[(SESSIONS[5].isoformat(), "factor_a")]["universe_size"] == 21
    assert all(item["status"] == "available" for item in ragged.availability)
    expected_availability = {
        (item["session_date"], item["factor_id"]): item
        for item in [*before.availability, *after.availability]
    }
    for key, item in by_key.items():
        assert item == expected_availability[key]
    # Uniform membership is the same computation whether or not it is spelled.
    uniform = kernel.materialize(
        feature_rows=rows,
        active_listing_ids=axis,
        sector_by_listing_id=sectors,
        factor_ids=FACTORS,
        binding=_binding(),
        members_by_session={session: axis for session in SESSIONS},
    )
    plain = kernel.materialize(
        feature_rows=rows,
        active_listing_ids=axis,
        sector_by_listing_id=sectors,
        factor_ids=FACTORS,
        binding=_binding(),
    )
    assert uniform.receipt_hash == plain.receipt_hash
    assert uniform.transformed_identity == plain.transformed_identity
    assert uniform.ordered_listing_ids_hash == plain.ordered_listing_ids_hash
    assert ragged.ordered_listing_ids_hash != plain.ordered_listing_ids_hash


def test_not_a_member_is_not_missing_data() -> None:
    """A member without a base row makes its session unavailable; a
    non-member without one is nothing at all, and a finite value never
    makes a non-member a member."""

    axis = _axis(24)
    sectors = _sectors(axis)
    rows = _rows(axis, SESSIONS[:3])
    kernel = PanelCrossSectionKernel()
    members = {SESSIONS[0]: axis, SESSIONS[1]: axis[:-1], SESSIONS[2]: axis[:-1]}
    # L00 (the last on the axis) has a finite value on every session but is a
    # member of the first only; L23 is a member throughout and loses its base
    # row on the third session.
    missing = rows.loc[~((rows["session_date"] == SESSIONS[2]) & (rows["listing_id"] == axis[0]))]
    panel = kernel.materialize(
        feature_rows=missing,
        active_listing_ids=axis,
        sector_by_listing_id=sectors,
        factor_ids=FACTORS,
        binding=_binding(),
        members_by_session=members,
    )
    per_session = panel.rows.groupby("session_date")["listing_id"].apply(set).to_dict()
    assert axis[-1] in per_session[SESSIONS[0].isoformat()]
    assert axis[-1] not in per_session[SESSIONS[1].isoformat()]
    assert axis[0] in per_session[SESSIONS[2].isoformat()]
    by_key = {(item["session_date"], item["factor_id"]): item for item in panel.availability}
    assert by_key[(SESSIONS[1].isoformat(), "factor_a")]["status"] == "available"
    third = by_key[(SESSIONS[2].isoformat(), "factor_a")]
    assert third["status"] == "unavailable"
    assert third["reason"] == "active_manifest_base_row_missing"
    assert third["universe_size"] == 23
    with pytest.raises(ValueError, match="outside the calculation axis"):
        kernel.materialize(
            feature_rows=rows,
            active_listing_ids=axis,
            sector_by_listing_id=sectors,
            factor_ids=FACTORS,
            binding=_binding(),
            members_by_session={session: (*axis, "L99") for session in SESSIONS[:3]},
        )
    with pytest.raises(ValueError, match="has no membership"):
        kernel.materialize(
            feature_rows=rows,
            active_listing_ids=axis,
            sector_by_listing_id=sectors,
            factor_ids=FACTORS,
            binding=_binding(),
            members_by_session={SESSIONS[0]: axis},
        )


def test_current_policy_preserves_a_session_across_batch_shapes_and_future_exits() -> None:
    """A future epoch boundary and partial receipt replay cannot change a prior day."""

    from alphalattice.foundation.feature_engine.panels.rematerialization import (
        ArtifactOnlyPanelRematerializer,
        PanelRematerializationMismatch,
    )

    assert cross_section_policy_hash() == SOURCE_ELIGIBILITY_POLICY_HASH
    axis = _axis(60)  # Sectors larger than eight expose numpy's legacy reduction seam.
    rows = _rows(axis, SESSIONS)
    kernel = PanelCrossSectionKernel()
    arguments = {
        "active_listing_ids": axis,
        "sector_by_listing_id": _sectors(axis),
        "factor_ids": FACTORS,
        "binding": _binding(current=True),
    }
    original = kernel.materialize(feature_rows=rows, **arguments)
    expected = original.rows.loc[original.rows.session_date == SESSIONS[0].isoformat()]
    availability = [
        {
            **item,
            "sector_counts_json": json.dumps(item["sector_counts"]),
            "small_sector_names_json": json.dumps(item["small_sector_names"]),
        }
        for item in original.availability
        if item["session_date"] == SESSIONS[0].isoformat()
    ]
    for height in (1, 2, 3, 7):
        replay = kernel.materialize(
            feature_rows=rows.loc[rows.session_date.isin(SESSIONS[:height])], **arguments
        )
        actual = replay.rows.loc[replay.rows.session_date == SESSIONS[0].isoformat()]
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        ArtifactOnlyPanelRematerializer._verifyavailability_rows(
            expected=availability, actual=replay.availability
        )
    changed = kernel.materialize(
        feature_rows=rows,
        members_by_session={day: axis if day == SESSIONS[0] else axis[:-3] for day in SESSIONS},
        **arguments,
    )
    pd.testing.assert_frame_equal(
        changed.rows.loc[changed.rows.session_date == SESSIONS[0].isoformat()],
        expected,
        check_exact=True,
    )
    tampered = [dict(item) for item in availability]
    tampered[0]["winsor_lower"] = float(tampered[0]["winsor_lower"]) + 1.0
    with pytest.raises(PanelRematerializationMismatch):
        ArtifactOnlyPanelRematerializer._verifyavailability_rows(
            expected=tampered, actual=changed.availability
        )


@pytest.mark.parametrize("missing_count", [1, 2])
def test_current_policy_uses_unchanged_coverage_for_missing_member_rows(missing_count: int) -> None:
    axis = _axis(60)
    rows = _rows(axis, SESSIONS[:2])
    missing = rows.loc[
        ~((rows.session_date == SESSIONS[0]) & rows.listing_id.isin(axis[:missing_count]))
    ]
    arguments = {
        "active_listing_ids": axis,
        "sector_by_listing_id": _sectors(axis),
        "factor_ids": FACTORS,
    }
    kernel = PanelCrossSectionKernel()
    panel = kernel.materialize(feature_rows=missing, binding=_binding(current=True), **arguments)
    items = [x for x in panel.availability if x["session_date"] == SESSIONS[0].isoformat()]
    for item in items:
        assert item["universe_size"] == 60  # Never shrink the denominator to hide missing rows.
        assert item["computed_count"] == 60 - missing_count
        assert item["coverage"] == (60 - missing_count) / 60
        assert item["status"] == ("available" if missing_count == 1 else "unavailable")
        assert item["reason"] == (None if missing_count == 1 else "coverage_below_98_percent")
    day = panel.rows.loc[panel.rows.session_date == SESSIONS[0].isoformat()]
    assert set(day.listing_id) == set(axis)  # Invalid data does not erase membership.
    assert np.isfinite(day.factor_a.to_numpy(dtype=float)).sum() == (
        59 if missing_count == 1 else 0
    )
    legacy = kernel.materialize(feature_rows=missing, binding=_binding(), **arguments)
    assert all(
        item["reason"] == "active_manifest_base_row_missing"
        for item in legacy.availability
        if item["session_date"] == SESSIONS[0].isoformat()
    )


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_a_non_finite_raw_value_is_missing_in_every_installed_method(value: float) -> None:
    """Requirement: a non-finite raw value counts as missing, as NaN does, in every
    session statistic of every installed Panel method -- the median, MAD, winsor and coverage
    alike -- so it never moves another name's value and is never published as a clipped one.
    A stock value and a state child each; the session stays published, so the comparison
    reads the other names rather than a withheld session."""

    from alphalattice.foundation.feature_engine.producers.preprocessing.catalog import (
        build_installed_panel_preprocessing_catalog,
    )

    axis = _axis(60)
    # Past the time-series method's 252-session lookback, so every method publishes the day.
    sessions = tuple(date(2025, 1, 6) + timedelta(days=offset) for offset in range(300))
    rows = _rows(axis, sessions)
    state = dict(zip(sessions, np.sin(np.arange(len(sessions)) / 7.0) + 1.5, strict=True))
    for factor in FACTORS:
        rows[f"{factor}__state"] = rows["session_date"].map(state)
    day, listing = sessions[280], axis[3]
    cell = (rows["session_date"] == day) & (rows["listing_id"] == listing)

    def published(adapter: object, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[object]]:
        panel = adapter.materialize(  # type: ignore[attr-defined]
            feature_rows=frame,
            active_listing_ids=axis,
            sector_by_listing_id=_sectors(axis),
            factor_ids=FACTORS,
            binding=_binding(current=True),
        )
        ordered = panel.rows.sort_values(["session_date", "listing_id"]).reset_index(drop=True)
        return ordered, sorted(json.dumps(item, sort_keys=True) for item in panel.availability)

    catalog = build_installed_panel_preprocessing_catalog()
    for recipe_id in catalog.recipe_ids:
        _recipe, adapter = catalog.resolve_executable(recipe_id)
        for column in ("factor_a", "factor_a__state"):
            missing, raw = rows.copy(), rows.copy()
            missing.loc[cell, column] = np.nan
            raw.loc[cell, column] = value
            expected, expected_availability = published(adapter, missing)
            actual, actual_availability = published(adapter, raw)
            pd.testing.assert_frame_equal(actual, expected, obj=f"{recipe_id} {column}")
            assert actual_availability == expected_availability, (recipe_id, column)
            on_day = actual.loc[actual["session_date"] == day.isoformat()]
            others = on_day.loc[on_day["listing_id"] != listing, "factor_a"]
            assert np.isfinite(others.to_numpy(dtype=float)).any(), (recipe_id, column)
            if column == "factor_a":
                # The name's own non-finite stock value is missing, never a clipped extreme.
                own = on_day.loc[on_day["listing_id"] == listing, "factor_a"]
                assert not np.isfinite(own.to_numpy(dtype=float)).any(), recipe_id
