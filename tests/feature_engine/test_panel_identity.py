"""Identity-preservation probes for the Panel DuckDB exit."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.artifacts import (
    PanelArtifactCompositionOwner,
    PanelCompositionBinding,
)
from alphalattice.foundation.feature_engine.panels.identity import (
    hash_panel_rows,
    panel_chunk_hash,
    panel_schema_hash,
)

CASE_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = CASE_ROOT.parents[1]


def _raw_panel_table() -> pa.Table:
    return pa.table(
        {
            "session_date": pa.array([date(2026, 1, 2), date(2026, 1, 2)], pa.date32()),
            "listing_id": pa.array(["listing-a", "listing-b"]),
            "materialization_receipt_hash": pa.array(["a" * 64, "b" * 64]),
            "factor_a": pa.array([1.25, None], pa.float64()),
            "factor_b": pa.array([0.0, -2.5], pa.float64()),
        }
    )


def test_ephemeral_duckdb_preserves_legacy_row_hash_golden() -> None:
    table = hash_panel_rows(
        _raw_panel_table(),
        manifest_revision="1" * 64,
        sector_revision="2" * 64,
        catalog_hash="3" * 64,
        policy_hash="4" * 64,
        factor_ids=("factor_a", "factor_b"),
    )

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
        "e66e60db73f60b1d9a26a989d27c84debed6a0c16a12892ae537b7b46e6c4ecc",
        "1ae867a55a0c80ebb9ef4abc2a0c8c1113d0d100c5169785bbb413bb12e48626",
    ]


def test_annual_parquet_rewrite_is_byte_stable(tmp_path: Path) -> None:
    table = hash_panel_rows(
        _raw_panel_table(),
        manifest_revision="1" * 64,
        sector_revision="2" * 64,
        catalog_hash="3" * 64,
        policy_hash="4" * 64,
        factor_ids=("factor_a", "factor_b"),
    )
    schema_hash = panel_schema_hash(table.schema)
    chunk_hash = panel_chunk_hash(table, panel_binding_hash="5" * 64, year=2026)
    metadata = {
        b"alphalattice.snapshot_kind": b"FeaturePanelChunk",
        b"alphalattice.chunk_hash": chunk_hash.encode(),
        b"alphalattice.panel_binding_hash": b"5" * 64,
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
            panel_binding_hash="5" * 64,
            year=2026,
            schema=pq.read_schema(first).remove_metadata(),
        )
        == chunk_hash
    )
    assert panel_chunk_hash(projected, panel_binding_hash="5" * 64, year=2026) != chunk_hash
    assert (
        hashlib.sha256(first.read_bytes()).hexdigest()
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )


def _binding(*, as_of_session: date) -> PanelCompositionBinding:
    return PanelCompositionBinding(
        manifest_revision="1" * 64,
        sector_revision="2" * 64,
        catalog_hash="3" * 64,
        policy_hash="4" * 64,
        panel_binding_hash="5" * 64,
        history_start=date(2025, 12, 31),
        as_of_session=as_of_session,
        factor_ids=("factor_a", "factor_b"),
    )


def _availability(sessions: tuple[date, ...]) -> list[dict[str, object]]:
    return [
        {
            "session_date": session.isoformat(),
            "factor_id": factor_id,
            "availability_hash": hashlib.sha256(
                f"{session.isoformat()}:{factor_id}".encode()
            ).hexdigest(),
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
        materialization_receipt_hash="a" * 64,
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
        materialization_receipt_hash="b" * 64,
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
        materialization_receipt_hash="c" * 64,
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


def _manifest_with_lineage(
    composition, *, spy_revision: str, manifest_revision: str = "1" * 64
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


def test_a_compatible_base_under_a_new_binding_reuses_closed_years_by_origin(
    tmp_path: Path,
) -> None:
    """A compatible base under a new binding reuses closed years by origin."""

    resolver = ArtifactResolver(tmp_path / "artifacts")
    owner = PanelArtifactCompositionOwner(resolver)
    listings = ("listing-a", "listing-b")
    day_one = (date(2025, 12, 31), date(2026, 1, 2))
    first = owner.begin(
        operation_id="day-one",
        binding=_binding(as_of_session=day_one[-1]),
        base_manifest=None,
        sessions=day_one,
        listing_ids=listings,
        spy_revision="a" * 64,
    )
    first.stage_patch(
        rows=_rows(day_one, listings),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash="a" * 64,
    )
    day_one_panel = first.finalize(availability=_availability(day_one))
    assert {key: value.spy_revision for key, value in day_one_panel.partition_origins.items()} == {
        "5" * 64: "a" * 64
    }
    assert all(chunk.origin_binding_hash == "5" * 64 for chunk in day_one_panel.chunks)
    base = _manifest_with_lineage(day_one_panel, spy_revision="a" * 64)

    day_two = (*day_one, date(2026, 1, 5))
    new_binding = PanelCompositionBinding(
        **{**_binding(as_of_session=day_two[-1]).__dict__, "panel_binding_hash": "6" * 64}
    )
    second = owner.begin(
        operation_id="day-two",
        binding=new_binding,
        base_manifest=base,
        sessions=day_two,
        listing_ids=listings,
        spy_revision="b" * 64,
    )
    assert second.base_compatible
    assert second.unreusable_years() == ()
    assert second.unreusable_sessions() == (date(2026, 1, 5),)
    second.stage_patch(
        rows=_rows(day_two[-1:], listings, offset=100.0),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash="b" * 64,
    )
    day_two_availability = [
        {
            **item,
            "panel_binding_hash": "5" * 64 if item["session_date"] < "2026-01-05" else "6" * 64,
        }
        for item in _availability(day_two)
    ]
    day_two_panel = second.finalize(availability=day_two_availability)
    assert (day_two_panel.reused_chunk_count, day_two_panel.written_chunk_count) == (1, 1)
    assert day_two_panel.reused_years == (2025,)
    assert day_two_panel.chunks[0] == day_one_panel.chunks[0]
    assert day_two_panel.chunks[0].origin_binding_hash == "5" * 64
    assert day_two_panel.chunks[1].origin_binding_hash == "6" * 64
    assert day_two_panel.chunks[1].chunk_hash != day_one_panel.chunks[1].chunk_hash
    assert {key: value.spy_revision for key, value in day_two_panel.partition_origins.items()} == {
        "5" * 64: "a" * 64,
        "6" * 64: "b" * 64,
    }
    loaded = owner.load(
        panel_binding_hash="6" * 64, panel_content_hash=day_two_panel.content.panel_content_hash
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
        spy_revision="b" * 64,
    )
    third.stage_patch(
        rows=_rows(day_two[-1:], listings, offset=100.0),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash="b" * 64,
    )
    with pytest.raises(ValueError, match="cannot resolve the origin"):
        third.finalize(
            availability=[
                {**item, "panel_binding_hash": "7" * 64} for item in _availability(day_two)
            ]
        )

    # Another membership: not compatible, nothing reusable, every session forced.
    foreign = owner.begin(
        operation_id="day-two-foreign",
        binding=new_binding,
        base_manifest=_manifest_with_lineage(
            day_one_panel, spy_revision="a" * 64, manifest_revision="9" * 64
        ),
        sessions=day_two,
        listing_ids=listings,
        spy_revision="b" * 64,
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
        spy_revision="b" * 64,
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
        spy_revision="a" * 64,
    )
    first.stage_patch(
        rows=_rows(held, listings),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash="a" * 64,
    )
    base_panel = first.finalize(availability=_availability(held))
    base = _manifest_with_lineage(base_panel, spy_revision="a" * 64)
    assert base["chunks"][0]["first_session"] == requested[0].isoformat()
    assert base["chunks"][0]["row_count"] == len(held) * len(listings)

    second = owner.begin(
        operation_id="requested",
        binding=PanelCompositionBinding(
            **{**binding.__dict__, "as_of_session": requested[-1], "panel_binding_hash": "6" * 64}
        ),
        base_manifest=base,
        sessions=requested,
        listing_ids=listings,
        spy_revision="b" * 64,
    )
    assert second.base_compatible
    assert second.reusable_years() == ()
    assert second.unreusable_years() == (2025,)
    assert second.unreusable_sessions() == requested
    second.stage_patch(
        rows=_rows(requested, listings, offset=100.0),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash="b" * 64,
    )
    composed = second.finalize(
        availability=[{**item, "panel_binding_hash": "6" * 64} for item in _availability(requested)]
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
    """requirement: reuse names bytes only after they still prove the recorded identity."""

    resolver = ArtifactResolver(tmp_path / "artifacts")
    owner = PanelArtifactCompositionOwner(resolver)
    listings = ("listing-a", "listing-b")
    day_one = (date(2025, 12, 31), date(2026, 1, 2))
    first = owner.begin(
        operation_id="day-one",
        binding=_binding(as_of_session=day_one[-1]),
        base_manifest=None,
        sessions=day_one,
        listing_ids=listings,
        spy_revision="a" * 64,
    )
    first.stage_patch(
        rows=_rows(day_one, listings),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash="a" * 64,
    )
    day_one_panel = first.finalize(availability=_availability(day_one))
    base = _manifest_with_lineage(day_one_panel, spy_revision="a" * 64)
    # The base manifest claims a different row-content identity for 2025 than
    # the file proves under its origin binding.
    tampered = dict(base)
    tampered["chunks"] = [dict(entry) for entry in base["chunks"]]
    tampered["chunks"][0]["chunk_hash"] = "e" * 64
    second = owner.begin(
        operation_id="day-two",
        binding=PanelCompositionBinding(
            **{**_binding(as_of_session=date(2026, 1, 5)).__dict__, "panel_binding_hash": "6" * 64}
        ),
        base_manifest=tampered,
        sessions=(*day_one, date(2026, 1, 5)),
        listing_ids=listings,
        spy_revision="b" * 64,
    )
    second.stage_patch(
        rows=_rows((date(2026, 1, 5),), listings, offset=100.0),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash="b" * 64,
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
        spy_revision="a" * 64,
    )
    with pytest.raises(OSError, match="simulated interruption"):
        first.stage_patch(
            rows=_rows(sessions, listings),
            factor_ids=("factor_a", "factor_b"),
            materialization_receipt_hash="a" * 64,
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
        spy_revision="a" * 64,
    )
    retry.stage_patch(
        rows=_rows(sessions, listings),
        factor_ids=("factor_a", "factor_b"),
        materialization_receipt_hash="a" * 64,
    )
    composition = retry.finalize(availability=_availability(sessions))
    assert [chunk.year for chunk in composition.chunks] == [2025, 2026]
    assert composition.chunks[0].chunk_hash == written[0].stem
    assert written[0].read_bytes() == first_bytes
    assert written[0].stat().st_mtime_ns == first_mtime
    assert len(list(chunk_dir.glob("*.parquet"))) == 2
    assert not list(chunk_dir.glob("*.tmp"))
    reachability = resolver.feature_panel_reachability(root_manifest_uris=[])
    assert not reachability.incomplete_publication


def test_reader_validates_each_chunk_under_the_binding_that_wrote_it(tmp_path: Path) -> None:
    """Reader validates each chunk under the binding that wrote it."""

    from alphalattice.foundation.feature_engine.panels.reader import (
        FeaturePanelReader,
        FeaturePanelReadRequest,
    )

    resolver = ArtifactResolver(tmp_path / "artifacts")
    table = hash_panel_rows(
        _raw_panel_table(),
        manifest_revision="1" * 64,
        sector_revision="2" * 64,
        catalog_hash="3" * 64,
        policy_hash="4" * 64,
        factor_ids=("factor_a", "factor_b"),
    )
    schema_hash = panel_schema_hash(table.schema)
    chunk_hash = panel_chunk_hash(table, panel_binding_hash="5" * 64, year=2026)
    descriptor = resolver.publish_feature_panel_chunk(
        table=table,
        content_hash=chunk_hash,
        metadata={
            "panel_binding_hash": "5" * 64,
            "calendar_year": "2026",
            "schema_hash": schema_hash,
        },
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
        manifest_ref="playpen://feature-panel/manifests/" + "0" * 64,
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

    assert paths(binding="5" * 64, origin=None)
    assert paths(binding="6" * 64, origin="5" * 64)
    with pytest.raises(ValueError, match="logical content hash mismatch"):
        paths(binding="6" * 64, origin=None)

    # A value moved under its kept row hash is refused before use (EV2): the row
    # hashes still bind, so only recomputing them from the values finds it.
    (path,) = paths(binding="5" * 64, origin=None)
    moved = pq.read_table(path)
    moved = moved.set_column(
        moved.schema.get_field_index("factor_b"), "factor_b", pa.array([0.0, 99.0], pa.float64())
    )
    pq.write_table(moved, path)
    with pytest.raises(ValueError, match="values do not match their row hashes"):
        paths(binding="5" * 64, origin=None)
