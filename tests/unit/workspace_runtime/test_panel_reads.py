"""The Panel read switch keys a chunk from its own metadata, refuses what it cannot serve, and
hands the Host the chunks its Panels hold."""

import json
from pathlib import Path
from typing import Any

import pytest

from alphalattice.control.workspace_runtime import panel_reads
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver

TEST_SOURCE = "src/alphalattice/control/workspace_runtime/panel_reads.py"
BINDING = {b"alphalattice.panel_binding_hash": b"b" * 64, b"alphalattice.calendar_year": b"2025"}
A, B, C = "a" * 64, "b" * 64, "c" * 64  # three Panel chunks; only A is stored
KEPT, EVICTED = "1" * 64, "2" * 64  # a snapshot available with A and B; one evicted with C


@pytest.mark.proves("workspace_runtime.panel_reads_key_a_chunk_by_its_binding_year_and_part")
@pytest.mark.parametrize(
    ("setting", "part", "expected"),
    [
        ("stored", b"2", ("b" * 64, "2025/2")),
        ("stored", None, ("b" * 64, "2025")),
        ("derived", b"2", RuntimeError),
        ("neither", b"2", ValueError),
    ],
    ids=["a part", "a whole year", "derived and unserved", "an unknown setting"],
)
def test_a_chunk_is_keyed_by_its_binding_year_and_part_and_unserved_reads_are_refused(
    monkeypatch: pytest.MonkeyPatch, setting: str, part: bytes | None, expected: object
) -> None:
    """A chunk's key is the binding, year and part it records; with no data layer registered a
    derived read is refused by name, a stored one reads nothing, and an unknown setting is
    refused by name."""
    held = panel_reads.registered()
    panel_reads.register(None)
    monkeypatch.setenv(panel_reads.SETTING, setting)
    try:
        if isinstance(expected, type):
            with pytest.raises(expected, match="feature_panel"):
                panel_reads.body("c" * 64)
            return
        assert panel_reads.body("c" * 64) is None
    finally:
        panel_reads.register(*(held or (None,)))
    metadata = BINDING if part is None else BINDING | {b"alphalattice.calendar_part": part}
    assert panel_reads.key(metadata) == expected


@pytest.mark.proves("workspace_runtime.panel_roots_are_the_available_snapshots_chunks")
def test_a_lifecycle_publication_hands_the_available_snapshots_chunks_to_be_rooted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under derived a lifecycle publication hands every available snapshot's chunks and the
    reader's proof of each stored chunk; under stored, or for a malformed list, nothing."""
    resolver = ArtifactResolver(tmp_path)
    (tmp_path / "feature-panel" / "chunks").mkdir(parents=True)
    (tmp_path / "feature-panel" / "chunks" / f"{A}.parquet").write_bytes(b"A")  # only A is stored
    manifests = tmp_path / "feature-panel" / "manifests"
    manifests.mkdir(parents=True)
    for snapshot, chunks in ((KEPT, (A, B)), (EVICTED, (C,))):
        listed = [{"chunk_hash": chunk} for chunk in chunks]
        (manifests / f"{snapshot}.json").write_text(
            json.dumps({"snapshot_hash": snapshot, "chunks": listed})
        )
    projection = [
        {"snapshot_hash": KEPT, "lifecycle": "ACTIVE"},
        {
            "snapshot_hash": EVICTED,
            "lifecycle": "SUPERSEDED",
            "physical_availability": "EVICTED_BY_RETENTION",
            "eviction_plan_hash": "e" * 64,
        },
    ]
    calls: list[tuple[str, list[str], dict[str, Any]]] = []
    held = panel_reads.registered()
    registered = (
        lambda *_: None,
        lambda *_, **__: None,
        lambda table, digests, **lifecycle: calls.append((table, sorted(digests), lifecycle)),
    )
    panel_reads.register(*registered)
    try:
        for setting in ("stored", "derived"):
            monkeypatch.setenv(panel_reads.SETTING, setting)
            resolver.publish_feature_panel_lifecycle_projection(snapshots=projection)
    finally:
        panel_reads.register(*(held or (None,)))
    assert [(table, digests) for table, digests, _ in calls] == [("feature_panel", [A, B])]
    lifecycle = calls[0][2]
    assert lifecycle["unread_for"] == panel_reads.UNREAD
    stored = tmp_path / "feature-panel" / "chunks" / f"{A}.parquet"
    asked: list[tuple[Path, dict, dict]] = []

    def proof(path: Path, manifest: dict, item: dict) -> bool:  # the Panel reader's, faked
        asked.append((path, manifest, item))
        return path == stored

    monkeypatch.setattr(panel_reads, "proven", proof)
    assert [lifecycle["verified"](chunk) for chunk in (A, B)] == [True, False]
    assert [(path.name, item["chunk_hash"]) for path, _, item in asked] == [
        (f"{A}.parquet", A),
        (f"{B}.parquet", B),
    ]  # the proof is asked of each stored chunk against the entry its manifest records
    assert asked[0][1]["snapshot_hash"] == KEPT
    calls.clear()
    (manifests / f"{KEPT}.json").write_text(
        json.dumps({"snapshot_hash": KEPT, "chunks": [{"chunk_hash": A}, "not a chunk"]})
    )
    panel_reads.register(*registered)
    try:
        resolver.publish_feature_panel_lifecycle_projection(snapshots=projection)
    finally:
        panel_reads.register(*(held or (None,)))
    assert calls == []  # a malformed chunk list leaves the roots as they were
