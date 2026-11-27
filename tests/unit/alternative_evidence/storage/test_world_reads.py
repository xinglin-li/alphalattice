"""The evidence readers: a world body read is verified by digest and size; identity is unique."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from blake3 import blake3

from alphalattice.capabilities.data.errors import DataError
from alphalattice.evidence.alternative_evidence.storage import world_reads

TEST_SOURCE = "src/alphalattice/evidence/alternative_evidence/storage/world_reads.py"


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


@pytest.mark.proves("alternative_evidence.missing_body_refused_by_name_with_its_restore")
def test_a_missing_body_blob_is_refused_by_name(tmp_path):
    """World's blob for a body is not on disk: the read refuses by its code, and no longer offers
    the removed rollback setting as a way on."""
    missing = tmp_path / "blobs" / ("c" * 64)
    with pytest.raises(DataError) as refused:
        world_reads._blob(missing)
    assert refused.value.failure.code == "alternative_evidence.world_body_missing"
    assert "ALPHALATTICE_EVIDENCE_READS" not in str(refused.value)


@pytest.mark.proves("alternative_evidence.world_versions_read_once_per_batch")
def test_references_find_their_world_versions_in_one_statement():
    """Any number of references find their candidate versions in one statement, each its own rows
    in order, known ones as world sorted them; no reference reads nothing."""

    class Counted(_Lake):
        statements = 0

        def execute(self, *_: object) -> _Lake:
            self.statements += 1
            return self

    def reference(revision: str) -> SimpleNamespace:
        return SimpleNamespace(
            source_name="SEC_EDGAR", source_document_name=None, revision=revision, content_bytes=4
        )

    rows = [(0, "a" * 64, True), (1, "f" * 64, False), (1, "b" * 64, True)]
    lake = Counted(rows)
    port = SimpleNamespace(lake=lake, log=SimpleNamespace(lineage=lambda *_: "provider"))
    found = world_reads.versions(port, (reference("a"), reference("b")), as_of=None)
    assert lake.statements == 1
    assert found == ((("a" * 64, True),), (("f" * 64, False), ("b" * 64, True)))
    assert world_reads.versions(port, (), as_of=None) == () and lake.statements == 1
