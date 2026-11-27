"""Identity-preservation probes for the Panel DuckDB exit."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.artifacts import (
    PanelArtifactCompositionOwner,
    PanelCompositionBinding,
    PreparedPanelComposition,
)
from alphalattice.foundation.feature_engine.panels.identity import (
    panel_chunk_hash,
    panel_schema_hash,
)
from alphalattice.kernel.identity import blake3_from_filename
from tests._support.identity_fixtures import fixture_digest
from tests.feature_engine.panel_workspace import identity_panel_table

_MANIFEST_REVISION = "blake3:" + "1" * 64
_ORIGINAL_BINDING = fixture_digest("original-panel-binding")
_REVISED_BINDING = fixture_digest("revised-panel-binding")
_ORIGINAL_SPY = fixture_digest("original-spy")
_REVISED_SPY = fixture_digest("revised-spy")
_ORIGINAL_RECEIPT = fixture_digest("original-receipt")
_REVISED_RECEIPT = fixture_digest("revised-receipt")

CASE_ROOT = Path(__file__).resolve().parents[3] / "tests/feature_engine"
PLAYPEN_ROOT = CASE_ROOT.parents[1]


def _raw_panel_table() -> pa.Table:
    return pa.table(
        {
            "session_date": pa.array([date(2026, 1, 2), date(2026, 1, 2)], pa.date32()),
            "listing_id": pa.array(["listing-a", "listing-b"]),
            "materialization_receipt_hash": pa.array(["blake3:" + "a" * 64, "blake3:" + "b" * 64]),
            "factor_a": pa.array([1.25, None], pa.float64()),
            "factor_b": pa.array([0.0, -2.5], pa.float64()),
        }
    )


@pytest.mark.proves("feature_engine.panel_chunks_stay_bound_to_their_writer")
def test_ephemeral_duckdb_preserves_row_encoding_under_blake3() -> None:
    """Hash the existing row encoding, including the null, with tagged BLAKE3."""
    table = identity_panel_table(_raw_panel_table())

    assert table.column_names == [
        "manifest_revision",
        "sector_revision",
        "catalog_hash",
        "policy_hash",
        "session_date",
        "listing_id",
        "row_hash",
        "materialization_receipt_hash",
        "factor_a",
        "factor_b",
    ]
    assert table.column("row_hash").to_pylist() == [
        "blake3:0f7b7948f4af840b30a0fbe6c832827c6e167f4fc307c2830b7b5ca164e5e7dc",
        "blake3:4b6a5c5e7f51dfa8c354968ff1c4f0eb6b5881085b806362e9c685ccf1c8105f",
    ]


@pytest.mark.proves("feature_engine.panel_chunks_stay_bound_to_their_writer")
def test_annual_parquet_rewrite_is_byte_stable(tmp_path: Path) -> None:
    """Control: Annual parquet rewrite is byte stable."""
    table = identity_panel_table(_raw_panel_table())
    schema_hash = panel_schema_hash(table.schema)
    chunk_hash = panel_chunk_hash(table, panel_binding_hash=_ORIGINAL_BINDING, year=2026)
    metadata = {
        b"alphalattice.snapshot_kind": b"FeaturePanelChunk",
        b"alphalattice.chunk_hash": chunk_hash.encode(),
        b"alphalattice.panel_binding_hash": _ORIGINAL_BINDING.encode(),
        b"alphalattice.calendar_year": b"2026",
        b"alphalattice.schema_hash": schema_hash.encode(),
    }
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(table.replace_schema_metadata(metadata), first, compression="zstd")
    decoded = pq.read_table(first)
    pq.write_table(decoded, second, compression="zstd")

    assert first.read_bytes() == second.read_bytes()
    # A verifier that reads only the identity columns of a written partition
    # and its schema from the file's metadata names the same chunk identity
    # as one that decodes every factor column; the identity is unchanged.
    projected = pq.read_table(first, columns=["session_date", "listing_id", "row_hash"])
    assert (
        panel_chunk_hash(
            projected,
            panel_binding_hash=_ORIGINAL_BINDING,
            year=2026,
            schema=pq.read_schema(first).remove_metadata(),
        )
        == chunk_hash
    )
    assert (
        panel_chunk_hash(projected, panel_binding_hash=_ORIGINAL_BINDING, year=2026) != chunk_hash
    )
    assert first.read_bytes() == second.read_bytes()


def _binding(*, as_of_session: date) -> PanelCompositionBinding:
    return PanelCompositionBinding(
        manifest_revision=_MANIFEST_REVISION,
        sector_revision="blake3:" + "2" * 64,
        catalog_hash="blake3:" + "3" * 64,
        policy_hash="blake3:" + "4" * 64,
        panel_binding_hash=_ORIGINAL_BINDING,
        history_start=date(2025, 12, 31),
        as_of_session=as_of_session,
        factor_ids=("factor_a", "factor_b"),
    )


def _availability(sessions: tuple[date, ...]) -> list[dict[str, object]]:
    return [
        {
            "session_date": session.isoformat(),
            "factor_id": factor_id,
            "availability_hash": fixture_digest((session, factor_id)),
        }
        for session in sessions
        for factor_id in ("factor_a", "factor_b")
    ]


def _manifest(composition) -> dict[str, object]:
    return {
        "panel_binding_hash": composition.binding.panel_binding_hash,
        "chunks": [
            {
                "year": chunk.year,
                "first_session": chunk.first_session.isoformat(),
                "last_session": chunk.last_session.isoformat(),
                "row_count": chunk.row_count,
                "chunk_hash": chunk.chunk_hash,
                "metadata_hash": chunk.metadata_hash,
                "uri": chunk.uri,
                **({"part": chunk.part} if chunk.part else {}),
            }
            for chunk in composition.chunks
        ],
    }


def test_full_tail_and_sparse_composition_are_bounded_by_year(tmp_path: Path) -> None:
    resolver = ArtifactResolver(tmp_path / "artifacts")
    owner = PanelArtifactCompositionOwner(resolver)
    listings = ("listing-a", "listing-b")
    initial_sessions = (date(2025, 12, 31), date(2026, 1, 2))
    initial = owner.begin(
        operation_id="full",
        binding=_binding(as_of_session=initial_sessions[-1]),
        base_manifest=None,
        sessions=initial_sessions,
        listing_ids=listings,
    )
    initial.stage_patch(
        rows=[
            {
                "session_date": session,
                "listing_id": listing,
                "factor_a": float(index + 1),
                "factor_b": float(index + 10),
            }
            for index, (session, listing) in enumerate(
                (session, listing) for session in initial_sessions for listing in listings
            )
        ],
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash=_ORIGINAL_RECEIPT,
    )
    full = initial.finalize(
        availability=_availability(initial_sessions),
    )
    assert full.written_chunk_count == 2
    assert full.reused_chunk_count == 0

    all_sessions = (*initial_sessions, date(2026, 1, 5))
    tail = owner.begin(
        operation_id="tail",
        binding=_binding(as_of_session=all_sessions[-1]),
        base_manifest=_manifest(full),
        sessions=all_sessions,
        listing_ids=listings,
    )
    tail.stage_patch(
        rows=[
            {
                "session_date": all_sessions[-1],
                "listing_id": listing,
                "factor_a": 100.0 + index,
                "factor_b": 200.0 + index,
            }
            for index, listing in enumerate(listings)
        ],
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash=_REVISED_RECEIPT,
    )
    appended = tail.finalize(
        availability=_availability(all_sessions),
    )
    assert appended.reused_chunk_count == 1
    assert appended.written_chunk_count == 1
    assert appended.chunks[0] == full.chunks[0]

    sparse = owner.begin(
        operation_id="sparse",
        binding=_binding(as_of_session=all_sessions[-1]),
        base_manifest=_manifest(appended),
        sessions=all_sessions,
        listing_ids=listings,
    )
    sparse.stage_patch(
        rows=[
            {
                "session_date": date(2026, 1, 2),
                "listing_id": listing,
                "factor_a": 500.0 + index,
            }
            for index, listing in enumerate(listings)
        ],
        factor_ids=("factor_a",),
        materialization_receipt_hash=fixture_digest("sparse-receipt"),
    )
    corrected = sparse.finalize(
        availability=_availability(all_sessions),
    )
    assert corrected.reused_chunk_count == 1
    assert corrected.written_chunk_count == 1
    assert corrected.chunks[0] == appended.chunks[0]
    loaded = owner.load(
        panel_binding_hash=corrected.binding.panel_binding_hash,
        panel_content_hash=corrected.content.panel_content_hash,
    )
    assert loaded.composition_hash == corrected.composition_hash
    assert loaded.binding == corrected.binding
    assert loaded.content == corrected.content
    assert loaded.schema_hash == corrected.schema_hash
    assert loaded.chunks == corrected.chunks

    prior_path = resolver.resolve_feature_panel_chunk_ref(
        uri=appended.chunks[1].uri,
        content_hash=appended.chunks[1].chunk_hash,
        metadata_hash=appended.chunks[1].metadata_hash,
    )
    corrected_path = resolver.resolve_feature_panel_chunk_ref(
        uri=corrected.chunks[1].uri,
        content_hash=corrected.chunks[1].chunk_hash,
        metadata_hash=corrected.chunks[1].metadata_hash,
    )
    prior = pq.read_table(prior_path).to_pandas().set_index(["session_date", "listing_id"])
    after = pq.read_table(corrected_path).to_pandas().set_index(["session_date", "listing_id"])
    assert after["factor_b"].equals(prior["factor_b"])
    assert tuple(after.loc[(date(2026, 1, 2),), "factor_a"]) == (500.0, 501.0)


def test_a_daily_rewrites_only_its_years_open_quarter(tmp_path: Path) -> None:
    """requirement: a daily on a December-sized year writes only the open
    quarter's part, about a quarter of the year's bytes; closed parts keep their files."""
    resolver = ArtifactResolver(tmp_path / "artifacts")
    owner = PanelArtifactCompositionOwner(resolver)
    listings = tuple(f"listing-{n:02d}" for n in range(40))
    year = [
        date(2026, 1, 1).fromordinal(d)
        for d in range(date(2026, 1, 2).toordinal(), date(2026, 12, 23).toordinal())
    ]
    sessions = (date(2025, 12, 31), *(d for d in year if d.weekday() < 5))
    daily = (*sessions, date(2026, 12, 23))

    def build(axis, base, name):
        session = owner.begin(
            operation_id=name,
            binding=_binding(as_of_session=axis[-1]),
            base_manifest=base,
            sessions=axis,
            listing_ids=listings,
        )
        fresh = axis if base is None else axis[-1:]
        session.stage_patch(
            rows=_rows(fresh, listings, offset=float(len(axis))),
            factor_ids=("factor_a", "factor_b"),
            materialization_receipt_hash=_ORIGINAL_RECEIPT,
        )
        return session.finalize(availability=_availability(axis))

    first = build(sessions, None, "first")
    after = build(daily, _manifest(first), "daily")
    parts = [c for c in after.chunks if c.year == 2026]
    assert (after.reused_chunk_count, after.written_chunk_count) == (4, 1), after.chunks
    assert len(parts) == 4 and parts[:3] == [c for c in first.chunks if c.year == 2026][:3]

    def size(chunk):
        return (
            resolver.resolve_feature_panel_chunk_ref(
                uri=chunk.uri, content_hash=chunk.chunk_hash, metadata_hash=chunk.metadata_hash
            )
            .stat()
            .st_size
        )

    assert size(parts[-1]) <= 0.3 * sum(map(size, parts))  # the open quarter, not the year


def _manifest_with_lineage(
    composition, *, spy_revision: str, manifest_revision: str = _MANIFEST_REVISION
) -> dict[str, object]:
    """A snapshot manifest as the publisher records it: sealed lineage and origins."""

    return {
        **_manifest(composition),
        "history_start": composition.binding.history_start.isoformat(),
        "chunks": [
            {**entry, "origin_binding_hash": chunk.origin_binding_hash}
            for entry, chunk in zip(
                _manifest(composition)["chunks"], composition.chunks, strict=True
            )
        ],
        "safe_summary": {
            "factor_catalog_summary": {
                factor_id: {} for factor_id in composition.binding.factor_ids
            },
            "lineage": {
                "manifest_revision": manifest_revision,
                "sector_revision": composition.binding.sector_revision,
                "catalog_hash": composition.binding.catalog_hash,
                "policy_hash": composition.binding.policy_hash,
                "spy_revision": spy_revision,
                "partition_origins": {
                    binding_hash: {
                        "spy_revision": value.spy_revision,
                        "materialization_receipt_hash": None,
                    }
                    for binding_hash, value in composition.partition_origins.items()
                },
            },
        },
    }


def _rows(sessions, listings, *, offset: float = 0.0):
    return [
        {
            "session_date": session,
            "listing_id": listing,
            "factor_a": float(index + 1) + offset,
            "factor_b": float(index + 10) + offset,
        }
        for index, (session, listing) in enumerate(
            (session, listing) for session in sessions for listing in listings
        )
    ]


def _published_two_year_panel(
    owner: PanelArtifactCompositionOwner, day_one: tuple[date, ...], listings: tuple[str, ...]
) -> PreparedPanelComposition:
    first = owner.begin(
        operation_id="day-one",
        binding=_binding(as_of_session=day_one[-1]),
        base_manifest=None,
        sessions=day_one,
        listing_ids=listings,
        spy_revision=_ORIGINAL_SPY,
    )
    first.stage_patch(
        rows=_rows(day_one, listings),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash=_ORIGINAL_RECEIPT,
    )
    return first.finalize(availability=_availability(day_one))


def test_a_compatible_base_under_a_new_binding_reuses_closed_years_by_origin(
    tmp_path: Path,
) -> None:
    """Control: Compatible base under new binding reuses closed years by origin."""

    resolver = ArtifactResolver(tmp_path / "artifacts")
    owner = PanelArtifactCompositionOwner(resolver)
    listings = ("listing-a", "listing-b")
    day_one = (date(2025, 12, 31), date(2026, 1, 2))
    day_one_panel = _published_two_year_panel(owner, day_one, listings)
    assert {key: value.spy_revision for key, value in day_one_panel.partition_origins.items()} == {
        _ORIGINAL_BINDING: _ORIGINAL_SPY
    }
    assert all(chunk.origin_binding_hash == _ORIGINAL_BINDING for chunk in day_one_panel.chunks)
    base = _manifest_with_lineage(day_one_panel, spy_revision=_ORIGINAL_SPY)

    day_two = (*day_one, date(2026, 1, 5))
    new_binding = PanelCompositionBinding(
        **{**_binding(as_of_session=day_two[-1]).__dict__, "panel_binding_hash": _REVISED_BINDING}
    )
    second = owner.begin(
        operation_id="day-two",
        binding=new_binding,
        base_manifest=base,
        sessions=day_two,
        listing_ids=listings,
        spy_revision=_REVISED_SPY,
    )
    assert second.base_compatible
    assert second.unreusable_years() == ()
    assert second.unreusable_sessions() == (date(2026, 1, 5),)
    second.stage_patch(
        rows=_rows(day_two[-1:], listings, offset=100.0),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash=_REVISED_RECEIPT,
    )
    day_two_availability = [
        {
            **item,
            "panel_binding_hash": _ORIGINAL_BINDING
            if item["session_date"] < "2026-01-05"
            else _REVISED_BINDING,
        }
        for item in _availability(day_two)
    ]
    day_two_panel = second.finalize(availability=day_two_availability)
    assert (day_two_panel.reused_chunk_count, day_two_panel.written_chunk_count) == (1, 1)
    assert day_two_panel.reused_years == (2025,)
    assert day_two_panel.chunks[0] == day_one_panel.chunks[0]
    assert day_two_panel.chunks[0].origin_binding_hash == _ORIGINAL_BINDING
    assert day_two_panel.chunks[1].origin_binding_hash == _REVISED_BINDING
    assert day_two_panel.chunks[1].chunk_hash != day_one_panel.chunks[1].chunk_hash
    assert {key: value.spy_revision for key, value in day_two_panel.partition_origins.items()} == {
        _ORIGINAL_BINDING: _ORIGINAL_SPY,
        _REVISED_BINDING: _REVISED_SPY,
    }
    loaded = owner.load(
        panel_binding_hash=_REVISED_BINDING,
        panel_content_hash=day_two_panel.content.panel_content_hash,
    )
    assert loaded.chunks == day_two_panel.chunks
    assert dict(loaded.partition_origins) == day_two_panel.partition_origins
    # The reused file is the same bytes; the composed year is a new file.
    reused_path = resolver.resolve_feature_panel_chunk_ref(
        uri=day_two_panel.chunks[0].uri,
        content_hash=day_two_panel.chunks[0].chunk_hash,
        metadata_hash=day_two_panel.chunks[0].metadata_hash,
    )
    assert pq.read_table(reused_path).num_rows == 2

    # A cell whose writer no recorded origin explains cannot be sealed.
    third = owner.begin(
        operation_id="day-two-unknown",
        binding=new_binding,
        base_manifest=base,
        sessions=day_two,
        listing_ids=listings,
        spy_revision=_REVISED_SPY,
    )
    third.stage_patch(
        rows=_rows(day_two[-1:], listings, offset=100.0),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash=_REVISED_RECEIPT,
    )
    with pytest.raises(ValueError, match="cannot resolve the origin"):
        third.finalize(
            availability=[
                {**item, "panel_binding_hash": fixture_digest("unrelated-binding")}
                for item in _availability(day_two)
            ]
        )

    # Another membership: not compatible, nothing reusable, every session forced.
    foreign = owner.begin(
        operation_id="day-two-foreign",
        binding=new_binding,
        base_manifest=_manifest_with_lineage(
            day_one_panel,
            spy_revision=_ORIGINAL_SPY,
            manifest_revision=fixture_digest("other-manifest"),
        ),
        sessions=day_two,
        listing_ids=listings,
        spy_revision=_REVISED_SPY,
    )
    assert not foreign.base_compatible
    assert foreign.unreusable_years() == (2025, 2026)
    assert foreign.unreusable_sessions() == day_two

    # A base that lost the year's calendar (a listing added) forces that year.
    grown = owner.begin(
        operation_id="day-two-grown",
        binding=new_binding,
        base_manifest=base,
        sessions=day_two,
        listing_ids=(*listings, "listing-c"),
        spy_revision=_REVISED_SPY,
    )
    assert grown.base_compatible
    assert grown.unreusable_years() == (2025, 2026)


@pytest.mark.parametrize(
    ("held", "requested"),
    [
        pytest.param(
            (date(2025, 12, 26), date(2025, 12, 29), date(2025, 12, 31)),
            (date(2025, 12, 26), date(2025, 12, 30), date(2025, 12, 31)),
            id="full-partition-same-endpoints-and-count",
        ),
        pytest.param(
            (date(2025, 12, 24), date(2025, 12, 29), date(2025, 12, 30)),
            (date(2025, 12, 24), date(2025, 12, 26), date(2025, 12, 30), date(2025, 12, 31)),
            id="prefix-same-first-session-last-covered-and-count",
        ),
    ],
)
def test_a_partition_under_another_calendar_is_recomputed_not_reused(
    tmp_path: Path, held: tuple[date, ...], requested: tuple[date, ...]
) -> None:
    """A partition under another calendar is recomputed not reused."""

    resolver = ArtifactResolver(tmp_path / "artifacts")
    owner = PanelArtifactCompositionOwner(resolver)
    listings = ("listing-a", "listing-b")
    binding = PanelCompositionBinding(
        **{
            **_binding(as_of_session=date(2026, 1, 2)).__dict__,
            "history_start": held[0],
            "as_of_session": held[-1],
        }
    )
    first = owner.begin(
        operation_id="held",
        binding=binding,
        base_manifest=None,
        sessions=held,
        listing_ids=listings,
        spy_revision=_ORIGINAL_SPY,
    )
    first.stage_patch(
        rows=_rows(held, listings),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash=_ORIGINAL_RECEIPT,
    )
    base_panel = first.finalize(availability=_availability(held))
    base = _manifest_with_lineage(base_panel, spy_revision=_ORIGINAL_SPY)
    assert base["chunks"][0]["first_session"] == requested[0].isoformat()
    assert base["chunks"][0]["row_count"] == len(held) * len(listings)

    second = owner.begin(
        operation_id="requested",
        binding=PanelCompositionBinding(
            **{
                **binding.__dict__,
                "as_of_session": requested[-1],
                "panel_binding_hash": _REVISED_BINDING,
            }
        ),
        base_manifest=base,
        sessions=requested,
        listing_ids=listings,
        spy_revision=_REVISED_SPY,
    )
    assert second.base_compatible
    assert second.reusable_years() == ()
    assert second.unreusable_years() == (2025,)
    assert second.unreusable_sessions() == requested
    second.stage_patch(
        rows=_rows(requested, listings, offset=100.0),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash=_REVISED_RECEIPT,
    )
    composed = second.finalize(
        availability=[
            {**item, "panel_binding_hash": _REVISED_BINDING} for item in _availability(requested)
        ]
    )
    assert (composed.reused_chunk_count, composed.written_chunk_count) == (0, 1)
    assert composed.chunks[0].chunk_hash != base_panel.chunks[0].chunk_hash
    path = resolver.resolve_feature_panel_chunk_ref(
        uri=composed.chunks[0].uri,
        content_hash=composed.chunks[0].chunk_hash,
        metadata_hash=composed.chunks[0].metadata_hash,
    )
    sessions = pq.read_table(path, columns=["session_date"]).column("session_date").to_pylist()
    assert tuple(sorted(set(sessions))) == requested


def test_a_reused_partition_that_no_longer_proves_its_identity_is_refused(
    tmp_path: Path,
) -> None:
    """Control: Reused partition no longer proves its identity is refused."""

    resolver = ArtifactResolver(tmp_path / "artifacts")
    owner = PanelArtifactCompositionOwner(resolver)
    listings = ("listing-a", "listing-b")
    day_one = (date(2025, 12, 31), date(2026, 1, 2))
    day_one_panel = _published_two_year_panel(owner, day_one, listings)
    base = _manifest_with_lineage(day_one_panel, spy_revision=_ORIGINAL_SPY)
    # The base manifest claims a different row-content identity for 2025 than
    # the file proves under its origin binding.
    tampered = dict(base)
    tampered["chunks"] = [dict(entry) for entry in base["chunks"]]
    tampered["chunks"][0]["chunk_hash"] = fixture_digest("false-chunk")
    second = owner.begin(
        operation_id="day-two",
        binding=PanelCompositionBinding(
            **{
                **_binding(as_of_session=date(2026, 1, 5)).__dict__,
                "panel_binding_hash": _REVISED_BINDING,
            }
        ),
        base_manifest=tampered,
        sessions=(*day_one, date(2026, 1, 5)),
        listing_ids=listings,
        spy_revision=_REVISED_SPY,
    )
    second.stage_patch(
        rows=_rows((date(2026, 1, 5),), listings, offset=100.0),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash=_REVISED_RECEIPT,
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        second.finalize(availability=_availability((*day_one, date(2026, 1, 5))))


def test_an_interrupted_composition_retried_under_the_same_binding_keeps_its_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted composition retried under the same binding keeps its files."""

    resolver = ArtifactResolver(tmp_path / "artifacts")
    owner = PanelArtifactCompositionOwner(resolver)
    listings = ("listing-a", "listing-b")
    sessions = (date(2025, 12, 31), date(2026, 1, 2))
    original = resolver.publish_feature_panel_chunk
    published_years: list[str] = []

    def dies_after_the_first_year(**kwargs):  # type: ignore[no-untyped-def]
        descriptor = original(**kwargs)
        published_years.append(kwargs["metadata"]["calendar_year"])
        if len(published_years) == 1:
            raise OSError("simulated interruption after the first partition")
        return descriptor

    monkeypatch.setattr(resolver, "publish_feature_panel_chunk", dies_after_the_first_year)
    first = owner.begin(
        operation_id="attempt-one",
        binding=_binding(as_of_session=sessions[-1]),
        base_manifest=None,
        sessions=sessions,
        listing_ids=listings,
        spy_revision=_ORIGINAL_SPY,
    )
    with pytest.raises(OSError, match="simulated interruption"):
        first.stage_patch(
            rows=_rows(sessions, listings),
            factor_ids=("factor_a", "factor_b"),
            materialization_receipt_hash=_ORIGINAL_RECEIPT,
        )
    monkeypatch.setattr(resolver, "publish_feature_panel_chunk", original)
    chunk_dir = tmp_path / "artifacts" / "feature-panel" / "chunks"
    written = sorted(chunk_dir.glob("*.parquet"))
    assert len(written) == 1 and not list(chunk_dir.glob("*.tmp"))
    staging = tmp_path / "artifacts" / "feature-panel" / "composition-staging"
    assert not list(staging.rglob("*.json"))
    first_bytes = written[0].read_bytes()
    first_mtime = written[0].stat().st_mtime_ns

    retry = owner.begin(
        operation_id="attempt-two",
        binding=_binding(as_of_session=sessions[-1]),
        base_manifest=None,
        sessions=sessions,
        listing_ids=listings,
        spy_revision=_ORIGINAL_SPY,
    )
    retry.stage_patch(
        rows=_rows(sessions, listings),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash=_ORIGINAL_RECEIPT,
    )
    composition = retry.finalize(availability=_availability(sessions))
    assert [chunk.year for chunk in composition.chunks] == [2025, 2026]
    assert composition.chunks[0].chunk_hash == blake3_from_filename(written[0].stem)
    assert written[0].read_bytes() == first_bytes
    assert written[0].stat().st_mtime_ns == first_mtime
    assert len(list(chunk_dir.glob("*.parquet"))) == 2
    assert not list(chunk_dir.glob("*.tmp"))
    reachability = resolver.feature_panel_reachability(root_manifest_uris=[])
    assert not reachability.incomplete_publication


@pytest.mark.proves("feature_engine.panel_chunks_stay_bound_to_their_writer")
def test_reader_validates_each_chunk_under_the_binding_that_wrote_it(
    tmp_path: Path, request, monkeypatch
) -> None:
    """Reader validates each chunk under the binding that wrote it."""

    from alphalattice.control.workspace_runtime import verified_facts
    from alphalattice.foundation.feature_engine.panels import reader as owner
    from alphalattice.foundation.feature_engine.panels.reader import (
        FeaturePanelReader,
        FeaturePanelReadRequest,
    )
    from alphalattice.kernel.identity import RACY_SECONDS, file_stamp

    request.addfinalizer(owner.configure_panel_proof_memo)
    owner.configure_panel_proof_memo(capacity=1)

    resolver = ArtifactResolver(tmp_path / "artifacts")
    table = identity_panel_table(_raw_panel_table())
    schema_hash = panel_schema_hash(table.schema)
    chunk_hash = panel_chunk_hash(table, panel_binding_hash=_ORIGINAL_BINDING, year=2026)
    descriptor = resolver.publish_feature_panel_chunk(
        table=table,
        content_hash=chunk_hash,
        metadata={
            "panel_binding_hash": _ORIGINAL_BINDING,
            "calendar_year": "2026",
            "schema_hash": schema_hash,
        },
    )
    chunk_path = resolver.resolve_feature_panel_chunk(descriptor)
    settled_ns = round((RACY_SECONDS + 1) * 1e9)
    monkeypatch.setattr(
        verified_facts, "wall_time_ns", lambda: max(file_stamp(chunk_path)[-2:]) + settled_ns
    )
    entry = {
        "year": 2026,
        "first_session": "2026-01-02",
        "last_session": "2026-01-02",
        "row_count": table.num_rows,
        "chunk_hash": chunk_hash,
        "metadata_hash": descriptor.metadata_hash,
        "uri": descriptor.uri,
    }
    request = FeaturePanelReadRequest(
        manifest_ref=resolver.feature_panel_manifest_uri(fixture_digest("panel-manifest")),
        start_session=date(2026, 1, 2),
        end_session=date(2026, 1, 2),
        factor_columns=("factor_a",),
    )
    reader = FeaturePanelReader(resolver)

    def paths(*, binding: str, origin: str | None) -> list[str]:
        manifest = {
            "panel_binding_hash": binding,
            "schema_hash": schema_hash,
            "chunks": [entry if origin is None else {**entry, "origin_binding_hash": origin}],
        }
        return reader._qualified_chunk_paths(manifest, request)

    assert paths(binding=_ORIGINAL_BINDING, origin=None)
    assert paths(binding=_REVISED_BINDING, origin=_ORIGINAL_BINDING)
    with pytest.raises(ValueError, match="logical content hash mismatch"):
        paths(binding=_REVISED_BINDING, origin=None)

    # A value moved under its kept row hash is refused before use (EV2): the row
    # hashes still bind, so only recomputing them from the values finds it.
    (path,) = paths(binding=_ORIGINAL_BINDING, origin=None)
    original, first = Path(path).read_bytes(), next(iter(owner._VALUES_VERIFIED))
    pq.write_table(pq.read_table(path), path, compression="NONE")
    assert paths(binding=_ORIGINAL_BINDING, origin=None)
    assert len(owner._VALUES_VERIFIED) == 1 and first not in owner._VALUES_VERIFIED
    Path(path).write_bytes(original)
    assert paths(binding=_ORIGINAL_BINDING, origin=None)
    moved = pq.read_table(path)
    moved = moved.set_column(
        moved.schema.get_field_index("factor_b"), "factor_b", pa.array([0.0, 99.0], pa.float64())
    )
    pq.write_table(moved, path)
    with pytest.raises(ValueError, match="values do not match their row hashes"):
        paths(binding=_ORIGINAL_BINDING, origin=None)
