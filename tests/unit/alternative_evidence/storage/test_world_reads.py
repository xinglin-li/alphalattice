"""Which store answers the evidence readers, as the setting names it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from blake3 import blake3

from alphalattice.evidence.alternative_evidence.storage import world_reads
from alphalattice.kernel.data.errors import DataError

TEST_SOURCE = "src/alphalattice/evidence/alternative_evidence/storage/world_reads.py"


@pytest.mark.proves("alternative_evidence.evidence_reads_setting_is_named")
def test_the_reads_setting_defaults_to_world_and_refuses_another_name(monkeypatch):
    """Unset, the setting reads world; it takes workspace; any other value is refused by name."""
    monkeypatch.delenv(world_reads.SETTING, raising=False)
    chosen = [world_reads.chosen()]
    monkeypatch.setenv(world_reads.SETTING, "workspace")
    chosen.append(world_reads.chosen())
    monkeypatch.setenv(world_reads.SETTING, "both")
    with pytest.raises(DataError) as unknown:
        world_reads.chosen()
    assert chosen == ["world", "workspace"]
    assert unknown.value.failure.code == "alternative_evidence.evidence_reads_unknown"


class _Lake:
    """World's catalog as a cursor answering every query with the rows it was given."""

    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows

    def cursor(self) -> _Lake:
        return self

    def execute(self, *_: object) -> _Lake:
        return self

    def fetchall(self) -> list[tuple]:
        return self.rows

    def fetchone(self) -> tuple | None:
        return self.rows[0] if self.rows else None

    def close(self) -> None:
        pass


@pytest.mark.proves("alternative_evidence.world_body_reads_refuse_by_name")
def test_a_world_body_read_returns_its_verified_bytes_or_refuses_by_name(tmp_path):
    """Retrieval reads a version's bytes only when they match world's digest and size, and a
    body world does not know once at the cutoff is refused by name, never guessed."""
    body = b"annual evidence"
    named = blake3(body).hexdigest()
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / named).write_bytes(body)

    def port(rows):
        return SimpleNamespace(lake=_Lake(rows), root=tmp_path)

    assert world_reads.exact(port([(named, len(body))]), "d", "v", 1) == body
    codes = []
    for rows in ([], [(named, len(body) + 1)]):
        with pytest.raises(DataError) as refused:
            world_reads.exact(port(rows), "d", "v", 1)
        codes.append(refused.value.failure.code)
    assert world_reads.identity(port([("d", "v", 7)]), body, 1) == ("d", "v", 7)
    with pytest.raises(DataError) as twice:
        world_reads.identity(port([("d", "v", 7), ("d", "w", 8)]), body, 1)
    codes.append(twice.value.failure.code)
    assert codes == [
        "alternative_evidence.source_object_missing",
        "alternative_evidence.source_object_tampered",
        "alternative_evidence.source_object_missing",
    ]
