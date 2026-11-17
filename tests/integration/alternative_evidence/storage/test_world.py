"""A workspace's evidence documents in world: admitted at its open with 0.2.7's facts, each body
kept once under blobs/ by its BLAKE3, an amendment naming the filing it amends, and a body two
workspaces hold one document."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from blake3 import blake3

from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceClass,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.documents.workspace import (
    AlternativeEvidenceDocumentPublisher,
)
from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    AlternativeEvidenceArtifactStore,
)
from alphalattice.evidence.alternative_evidence.sources.contracts import (
    SOURCE_COMMIT_CATEGORY,
    AcquiredEvidenceDocument,
    AcquiredEvidenceSourceCommit,
    SecFilingInventoryEntry,
)
from alphalattice.evidence.alternative_evidence.sources.sec_edgar import plan_sec_filing_selection
from alphalattice.evidence.alternative_evidence.storage import world
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.shared_kernel.identity import canonical_hash

TEST_SOURCE = "src/alphalattice/evidence/alternative_evidence/storage/world.py"
CIK = "0000320193"
FILED = datetime(2026, 2, 1, tzinfo=UTC)
ACCEPTED = FILED + timedelta(hours=21, minutes=30)
AMENDED = ACCEPTED + timedelta(days=30)
CAPTURED = datetime(2026, 3, 5, 9, tzinfo=UTC)
ORIGINAL, AMENDMENT = "0000320193-26-000001", "0000320193-26-000009"
ROWS = (  # each document's newest version, as a read at 2100 sees it
    "SELECT source_ref, document_type, doc_digest, byte_count, published_precision,"
    " epoch_us(accepted_at), availability_basis, right_of_use, report_period_end,"
    " CAST(doc_id AS VARCHAR), CAST(amends_doc_id AS VARCHAR)"
    " FROM evidence_doc_asof(4102444800000000) ORDER BY source_ref"
)
KNOWN = (  # every version's known time, by document
    "SELECT source_ref, list(epoch_us(known_asof) ORDER BY known_asof)"
    " FROM lake.world.evidence_doc GROUP BY ALL ORDER BY ALL"
)


def source_hash(source: str, revision: str, media_type: str, content: bytes) -> str:
    """The acquisition's own hash of a document's bytes, as 0.2.7 computes it."""
    fields = {"source_name": source, "revision": revision, "media_type": media_type}
    return canonical_hash(fields | {"content_hex": content.hex()})


def micros(at: datetime) -> int:
    return int(at.timestamp()) * 1_000_000


def filing(
    accession: str, form: str, content: bytes, accepted: datetime
) -> AcquiredEvidenceDocument:
    """One SEC filing as 0.2.7 acquired it: published on its date, known at its acceptance."""
    return AcquiredEvidenceDocument(
        semantic_handle=f"DOC-{accession[-6:]}",
        entity_id="AAPL",
        source_name="SEC_EDGAR",
        source_right="SEC_PUBLIC_OFFICIAL_ACCESS",
        evidence_class=AlternativeEvidenceClass.SEC_FILING,
        document_type=form,
        revision=accession,
        title=f"AAPL {form}",
        published_at=FILED,
        published_precision="DATE",
        accepted_at=accepted,
        available_at=accepted,
        report_period_end=date(2025, 12, 27),
        media_type="text/html",
        content=content,
        immutable_source=True,
        source_content_hash=source_hash("SEC_EDGAR", accession, "text/html", content),
        source_cik=CIK,
        source_document_name="aapl.htm",
    )


def recorded(content: bytes) -> AcquiredEvidenceDocument:
    """A document a person recorded, with no acceptance: known from its capture."""
    return AcquiredEvidenceDocument(
        semantic_handle="DOC-NOTE",
        entity_id="AAPL",
        source_name="ISSUER_RECORDED",
        source_right="USER_PROVIDED_FOR_LOCAL_RESEARCH",
        evidence_class=AlternativeEvidenceClass.SEC_FILING,
        document_type="note",
        revision="note-1",
        title="AAPL supplier note",
        available_at=CAPTURED,
        media_type="text/plain",
        content=content,
        immutable_source=True,
        source_content_hash=source_hash("ISSUER_RECORDED", "note-1", "text/plain", content),
    )


def acquire(workspace: Path, *documents: AcquiredEvidenceDocument) -> None:
    """Keep each document as 0.2.7's acquisition does: its bytes in the source-object store and
    its sealed commit record."""
    sources = AlternativeEvidenceDocumentPublisher(workspace / "runtime" / "evidence-knowledge")
    artifacts = AlternativeEvidenceArtifactStore(workspace / "runtime" / "artifacts")
    for document in documents:
        reference = sources.publish_source_object(document)
        commit = seal_contract(
            AcquiredEvidenceSourceCommit, "commit_hash", reference=reference, committed_at=CAPTURED
        )
        artifacts.publish(SOURCE_COMMIT_CATEGORY, commit.commit_hash, commit)


def plan(workspace: Path) -> None:
    """The issuer's sealed selection plan, which links the amendment to its original."""
    entries = tuple(
        SecFilingInventoryEntry(
            entity_id="AAPL",
            cik=CIK,
            accession=accession,
            form=form,
            report_date=date(2025, 12, 27),
            filed_on=accepted.date(),
            accepted_at=accepted,
            primary_document="aapl.htm",
            amendment_of_accession=amends,
            amendment_relation="NOT_APPLICABLE" if amends is None else "LINKED",
        )
        for accession, form, accepted, amends in (
            (ORIGINAL, "10-K", ACCEPTED, None),
            (AMENDMENT, "10-K/A", AMENDED, ORIGINAL),
        )
    )
    sealed = plan_sec_filing_selection(
        entries,
        entity_id="AAPL",
        cik=CIK,
        evidence_as_of=AMENDED + timedelta(days=1),
        event_window_days=90,
        policy_budget=4,
        unit_capacity=4,
    )
    artifacts = AlternativeEvidenceArtifactStore(workspace / "runtime" / "artifacts")
    artifacts.publish("sec-selection-plans", sealed.plan_hash, sealed)


@pytest.fixture
def host(tmp_path: Path) -> Iterator[Path]:
    """A Host history open on a data root of its own; yields the data root."""
    history.open_history("w1", "workspace", tmp_path / "w1", root=tmp_path / "data")
    try:
        yield tmp_path / "data"
    finally:
        history.release_history()


@pytest.mark.proves("alternative_evidence.world_holds_each_document_once")
def test_a_workspace_open_admits_each_document_once_with_its_facts(tmp_path, host):
    """Each document is one world row with 0.2.7's facts and its body under blobs/, and an
    amendment is linked by its next version once its link and original are both known."""
    bodies = {"10-K": b"<html>annual</html>", "10-K/A": b"<html>part iii</html>", "note": b"n"}
    acquire(
        tmp_path / "w1",
        filing(AMENDMENT, "10-K/A", bodies["10-K/A"], AMENDED),
        recorded(bodies["note"]),
    )
    acquire(tmp_path / "w2", filing(ORIGINAL, "10-K", bodies["10-K"], ACCEPTED))
    admitted = [world.admit(tmp_path / "w1", "w1")]
    plan(tmp_path / "w1")  # the link, published after the amendment's admission
    admitted += [world.admit(tmp_path / "w1", "w1"), world.admit(tmp_path / "w1", "w1")]
    admitted.append(world.admit(tmp_path / "w2", "w2"))  # the original, in another workspace
    admitted += [world.admit(tmp_path / "w1", "w1"), world.admit(tmp_path / "w1", "w1")]
    lake = history.store().lake
    rows, known = lake.execute(ROWS).fetchall(), lake.execute(KNOWN).fetchall()
    digest = {kind: blake3(body).hexdigest() for kind, body in bodies.items()}
    sec = "SEC_PUBLIC_OFFICIAL_ACCESS"
    assert [None if a is None else (a["rows"], a["held"], a["waiting"]) for a in admitted] == [
        ({"ISSUER_RECORDED": 1, "SEC_EDGAR": 1}, 0, []),
        ({}, 2, [ORIGINAL]),
        None,  # settled: its original is in no workspace yet
        ({"SEC_EDGAR": 1}, 0, []),
        ({"SEC_EDGAR": 1}, 1, []),  # the amendment's next version, with its link
        None,
    ]
    assert [row[:9] for row in rows] == [
        (f"{ORIGINAL}/aapl.htm", "10-K", digest["10-K"], 19, "DATE", micros(ACCEPTED),
         "OFFICIAL_ACCEPTANCE", sec, date(2025, 12, 27)),
        (f"{AMENDMENT}/aapl.htm", "10-K/A", digest["10-K/A"], 21, "DATE", micros(AMENDED),
         "OFFICIAL_ACCEPTANCE", sec, date(2025, 12, 27)),
        ("note-1", "note", digest["note"], 1, None, None,
         "RECORDED_IMPORT_CAPTURE_ACCEPTANCE_UNKNOWN", "USER_PROVIDED_FOR_LOCAL_RESEARCH", None),
    ]  # fmt: skip
    assert [row[10] for row in rows] == [None, rows[0][9], None]
    assert [(ref, times[0]) for ref, times in known] == [
        (f"{ORIGINAL}/aapl.htm", micros(ACCEPTED)),
        (f"{AMENDMENT}/aapl.htm", micros(AMENDED)),
        ("note-1", micros(CAPTURED)),
    ]
    assert len(known[1][1]) == 2 and known[1][1][1] > micros(CAPTURED)  # linked from admission
    assert {kind: (host / "blobs" / d).read_bytes() for kind, d in digest.items()} == bodies


@pytest.mark.proves("alternative_evidence.world_holds_each_document_once")
def test_one_body_is_one_document_and_other_bytes_are_its_next_version(tmp_path, host):
    """A body another workspace holds adds nothing; the same filing with other bytes is a new
    version of its doc_id; a body the workspace lost is counted and admitted once restored."""
    acquire(tmp_path / "w1", filing(ORIGINAL, "10-K", b"<html>annual</html>", ACCEPTED))
    acquire(tmp_path / "w2", filing(ORIGINAL, "10-K", b"<html>annual</html>", ACCEPTED))
    acquire(tmp_path / "w3", filing(ORIGINAL, "10-K", b"<html>restated</html>", ACCEPTED))
    acquire(tmp_path / "w4", filing(AMENDMENT, "10-K/A", b"<html>lost</html>", AMENDED))
    objects = tmp_path / "w4" / "runtime" / "evidence-knowledge" / ".system" / "source-objects"
    (lost,) = objects.iterdir()
    kept = lost.read_bytes()
    lost.unlink()
    admitted = [world.admit(tmp_path / name, name) for name in ("w1", "w2", "w3", "w4")]
    lost.write_bytes(kept)
    admitted += [world.admit(tmp_path / "w4", "w4"), world.admit(tmp_path / "w4", "w4")]
    versions = (
        history.store()
        .lake.execute(
            "SELECT count(DISTINCT doc_id), count(*),"
            " list(epoch_us(known_asof) ORDER BY known_asof) FROM lake.world.evidence_doc"
            f" WHERE source_ref LIKE '{ORIGINAL}%'"
        )
        .fetchone()
    )
    assert [None if a is None else (a["rows"], a["unreadable"], a["held"]) for a in admitted] == [
        ({"SEC_EDGAR": 1}, 0, 0),
        ({}, 0, 1),
        ({"SEC_EDGAR": 1}, 0, 0),
        ({}, 1, 0),
        ({"SEC_EDGAR": 1}, 0, 0),  # restored, admitted at the next open
        None,
    ]
    assert versions[:2] == (1, 2)  # one document, its second body known from its admission
    assert versions[2][0] == micros(ACCEPTED) < versions[2][1]
