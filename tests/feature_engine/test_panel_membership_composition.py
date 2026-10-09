"""Partition reuse under the session-cross-section rule.

A row's identity binds its own session's members and their sectors, so a
build whose manifest changed reuses every partition whose cross-sections it
still computes, and a member joining at one session moves only the sessions
from that one on.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.contracts import (
    PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
    PanelMembership,
    PanelMembershipBasisRange,
    PanelMembershipEpoch,
)
from alphalattice.foundation.feature_engine.panels.artifacts import (
    PanelArtifactCompositionOwner,
    PanelCompositionBinding,
    PreparedPanelComposition,
)
from alphalattice.kernel.quant.sector_history import SectorHistory

FACTORS = ("factor_a", "factor_b")
SECTORS = {"L1": "Energy", "L2": "Energy", "L3": "Tech", "L4": "Tech"}


def _history(sectors: dict[str, str]) -> SectorHistory:
    """A map every session reads, as the Feature build hands it."""
    return SectorHistory(current_revision="2" * 64, current=sectors)


YEAR_ONE = (date(2025, 12, 29), date(2025, 12, 30), date(2025, 12, 31))
YEAR_TWO = (date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6))


def _binding(
    *, manifest_revision: str, sector_revision: str, binding_hash: str, as_of: date
) -> PanelCompositionBinding:
    return PanelCompositionBinding(
        manifest_revision=manifest_revision,
        sector_revision=sector_revision,
        catalog_hash="3" * 64,
        policy_hash="4" * 64,
        panel_binding_hash=binding_hash,
        history_start=YEAR_ONE[0],
        as_of_session=as_of,
        factor_ids=FACTORS,
        row_identity_basis=PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
    )


def _membership(
    axis: tuple[str, ...], epochs: tuple[tuple[date, date, tuple[str, ...]], ...]
) -> PanelMembership:
    return PanelMembership(
        listing_ids=axis,
        epochs=tuple(PanelMembershipEpoch(first, last, members) for first, last, members in epochs),
        basis_ranges=(PanelMembershipBasisRange(YEAR_ONE[0], YEAR_TWO[-1], "TEST"),),
    )


def _rows(sessions, members_by_session, *, offset: float = 0.0):
    rows = []
    for session in sessions:
        for index, listing in enumerate(sorted(members_by_session[session])):
            rows.append(
                {
                    "session_date": session,
                    "listing_id": listing,
                    "factor_a": float(index + 1) + offset,
                    "factor_b": float(index + 10) + offset,
                }
            )
    return rows


def _availability(sessions, *, binding_hash: str) -> list[dict[str, object]]:
    return [
        {
            "session_date": session.isoformat(),
            "factor_id": factor_id,
            "availability_hash": hashlib.sha256(
                f"{session.isoformat()}:{factor_id}".encode()
            ).hexdigest(),
            "panel_binding_hash": binding_hash,
        }
        for session in sessions
        for factor_id in FACTORS
    ]


def _manifest(composition: PreparedPanelComposition, *, spy_revision: str) -> dict[str, object]:
    """A snapshot manifest as the publisher records it under the cross-section rule."""

    binding = composition.binding
    return {
        "panel_binding_hash": binding.panel_binding_hash,
        "history_start": binding.history_start.isoformat(),
        "listing_set_hash": "0" * 64,
        "chunks": [
            {
                "year": chunk.year,
                "first_session": chunk.first_session.isoformat(),
                "last_session": chunk.last_session.isoformat(),
                "row_count": chunk.row_count,
                "chunk_hash": chunk.chunk_hash,
                "metadata_hash": chunk.metadata_hash,
                "uri": chunk.uri,
                "origin_binding_hash": chunk.origin_binding_hash,
                "cross_sections": [item.to_payload() for item in chunk.cross_sections],
            }
            for chunk in composition.chunks
        ],
        "safe_summary": {
            "factor_catalog_summary": {factor_id: {} for factor_id in FACTORS},
            "membership": composition.membership.summary_payload()
            if composition.membership is not None
            else None,
            "lineage": {
                "manifest_revision": binding.manifest_revision,
                "sector_revision": binding.sector_revision,
                "catalog_hash": binding.catalog_hash,
                "policy_hash": binding.policy_hash,
                "spy_revision": spy_revision,
                "row_identity_basis": binding.row_identity_basis,
                "partition_origins": {
                    key: {
                        "spy_revision": value.spy_revision,
                        "manifest_revision": value.manifest_revision,
                        "sector_revision": value.sector_revision,
                        "materialization_receipt_hash": None,
                    }
                    for key, value in composition.partition_origins.items()
                },
            },
        },
    }


def test_membership_change_moves_only_the_sessions_from_its_effective_one(
    tmp_path: Path,
) -> None:
    """Membership change moves only the sessions from its effective one."""

    resolver = ArtifactResolver(tmp_path / "artifacts")
    owner = PanelArtifactCompositionOwner(resolver)
    sessions = (*YEAR_ONE, *YEAR_TWO)
    three = ("L1", "L2", "L3")
    day_one_membership = _membership(three, ((YEAR_ONE[0], YEAR_TWO[-1], three),))
    day_one_members = {session: three for session in sessions}
    first = owner.begin(
        operation_id="day-one",
        binding=_binding(
            manifest_revision="1" * 64,
            sector_revision="2" * 64,
            binding_hash="5" * 64,
            as_of=YEAR_TWO[-1],
        ),
        base_manifest=None,
        sessions=sessions,
        listing_ids=three,
        spy_revision="a" * 64,
        membership=day_one_membership,
        sector_history=_history(SECTORS),
    )
    first.stage_patch(
        rows=_rows(sessions, day_one_members),
        factor_ids=FACTORS,
        materialization_receipt_hash="a" * 64,
    )
    day_one = first.finalize(availability=_availability(sessions, binding_hash="5" * 64))
    assert [chunk.row_count for chunk in day_one.chunks] == [9, 9]
    assert len(day_one.cross_sections) == 2  # one range per year, one identity
    identity_one = day_one.chunks[0].cross_sections[0].cross_section_identity
    assert day_one.chunks[1].cross_sections[0].cross_section_identity == identity_one

    # Day two: L4 enters at the second session of year two.
    entry = YEAR_TWO[1]
    four = ("L1", "L2", "L3", "L4")
    day_two_membership = _membership(
        four, ((YEAR_ONE[0], YEAR_TWO[0], three), (entry, YEAR_TWO[-1], four))
    )
    day_two_members = {session: (four if session >= entry else three) for session in sessions}
    second = owner.begin(
        operation_id="day-two",
        binding=_binding(
            manifest_revision="7" * 64,
            sector_revision="8" * 64,
            binding_hash="6" * 64,
            as_of=YEAR_TWO[-1],
        ),
        base_manifest=_manifest(day_one, spy_revision="a" * 64),
        sessions=sessions,
        listing_ids=four,
        spy_revision="b" * 64,
        membership=day_two_membership,
        sector_history=_history(SECTORS),
    )
    assert second.base_compatible
    assert second.reusable_years() == (2025,)
    assert second.unreusable_years() == ()
    assert second.unreusable_sessions() == tuple(YEAR_TWO[1:])
    second.stage_patch(
        rows=_rows(YEAR_TWO[1:], day_two_members, offset=100.0),
        factor_ids=FACTORS,
        materialization_receipt_hash="b" * 64,
    )
    availability = [
        {
            **item,
            "panel_binding_hash": "6" * 64
            if item["session_date"] >= entry.isoformat()
            else "5" * 64,
        }
        for item in _availability(sessions, binding_hash="5" * 64)
    ]
    day_two = second.finalize(availability=availability)
    assert (day_two.reused_chunk_count, day_two.written_chunk_count) == (1, 1)
    assert day_two.chunks[0] == day_one.chunks[0]
    assert day_two.chunks[1].row_count == 3 + 4 + 4
    assert [item.member_count for item in day_two.chunks[1].cross_sections] == [3, 4]
    assert day_two.chunks[1].cross_sections[0].cross_section_identity == identity_one
    # The merged year keeps day one's row identities before the entry.
    year_two_path = resolver.resolve_feature_panel_chunk_ref(
        uri=day_two.chunks[1].uri,
        content_hash=day_two.chunks[1].chunk_hash,
        metadata_hash=day_two.chunks[1].metadata_hash,
    )
    day_one_path = resolver.resolve_feature_panel_chunk_ref(
        uri=day_one.chunks[1].uri,
        content_hash=day_one.chunks[1].chunk_hash,
        metadata_hash=day_one.chunks[1].metadata_hash,
    )
    merged = pq.read_table(year_two_path).to_pandas()
    original = pq.read_table(day_one_path).to_pandas()
    before_entry = merged.loc[merged["session_date"] < entry]
    assert list(before_entry["row_hash"]) == list(
        original.loc[original["session_date"] < entry, "row_hash"]
    )
    assert set(merged.loc[merged["session_date"] >= entry, "listing_id"]) == set(four)
    assert "L4" not in set(before_entry["listing_id"])
    assert day_two.partition_origins["5" * 64].manifest_revision == "1" * 64
    assert day_two.partition_origins["6" * 64].sector_revision == "8" * 64

    # Day three: the manifest changed for governance alone; the membership
    # and sectors are day two's, so every partition is reused as it stands.
    third = owner.begin(
        operation_id="day-three",
        binding=_binding(
            manifest_revision="9" * 64,
            sector_revision="c" * 64,
            binding_hash="d" * 64,
            as_of=YEAR_TWO[-1],
        ),
        base_manifest=_manifest(day_two, spy_revision="b" * 64),
        sessions=sessions,
        listing_ids=four,
        spy_revision="b" * 64,
        membership=day_two_membership,
        sector_history=_history(SECTORS),
    )
    assert third.base_compatible
    assert third.reusable_years() == (2025, 2026)
    assert third.unreusable_sessions() == ()
    day_three = third.finalize(
        availability=[
            {
                **item,
                "panel_binding_hash": "5" * 64
                if item["session_date"] < entry.isoformat()
                else "6" * 64,
            }
            for item in availability
        ]
    )
    assert (day_three.reused_chunk_count, day_three.written_chunk_count) == (2, 0)
    assert [chunk.chunk_hash for chunk in day_three.chunks] == [
        chunk.chunk_hash for chunk in day_two.chunks
    ]
    loaded = owner.load(
        panel_binding_hash="d" * 64, panel_content_hash=day_three.content.panel_content_hash
    )
    assert loaded.membership == day_two_membership
    assert loaded.binding.row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION

    # A sector change for a member reaches every session that member is in.
    resectored = {**SECTORS, "L1": "Health"}
    fourth = owner.begin(
        operation_id="day-four",
        binding=_binding(
            manifest_revision="9" * 64,
            sector_revision="e" * 64,
            binding_hash="f" * 64,
            as_of=YEAR_TWO[-1],
        ),
        base_manifest=_manifest(day_three, spy_revision="b" * 64),
        sessions=sessions,
        listing_ids=four,
        spy_revision="b" * 64,
        membership=day_two_membership,
        sector_history=_history(resectored),
    )
    assert fourth.base_compatible
    assert fourth.reusable_years() == ()
    assert fourth.unreusable_years() == (2025, 2026)

    # A base written under the binding rule (no recorded rule, no ranges)
    # cannot serve a build under the cross-section rule: the transition
    # recomputes every partition once, and every later build reuses.
    legacy = _manifest(day_three, spy_revision="b" * 64)
    legacy_summary = legacy["safe_summary"]
    assert isinstance(legacy_summary, dict)
    lineage = legacy_summary["lineage"]
    assert isinstance(lineage, dict)
    del lineage["row_identity_basis"]
    for chunk in legacy["chunks"]:
        assert isinstance(chunk, dict)
        del chunk["cross_sections"]
    transition = owner.begin(
        operation_id="transition",
        binding=_binding(
            manifest_revision="9" * 64,
            sector_revision="c" * 64,
            binding_hash="e" * 64,
            as_of=YEAR_TWO[-1],
        ),
        base_manifest=legacy,
        sessions=sessions,
        listing_ids=four,
        spy_revision="b" * 64,
        membership=day_two_membership,
        sector_history=_history(SECTORS),
    )
    assert not transition.base_compatible
    assert transition.unreusable_years() == (2025, 2026)


def test_cross_section_rule_refuses_a_composition_without_its_membership(
    tmp_path: Path,
) -> None:
    owner = PanelArtifactCompositionOwner(ArtifactResolver(tmp_path / "artifacts"))
    with pytest.raises(ValueError, match="needs membership and sectors"):
        owner.begin(
            operation_id="x",
            binding=_binding(
                manifest_revision="1" * 64,
                sector_revision="2" * 64,
                binding_hash="5" * 64,
                as_of=YEAR_TWO[-1],
            ),
            base_manifest=None,
            sessions=(*YEAR_ONE, *YEAR_TWO),
            listing_ids=("L1", "L2"),
        )
