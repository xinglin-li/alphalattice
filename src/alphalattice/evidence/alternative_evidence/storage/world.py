"""A workspace's evidence documents admitted into world once per machine, through StorePort.

0.2.7 keeps each acquired document as a sealed reference (the evidence artifacts'
source-document-commits and source-document-sets) naming its original bytes in the evidence
workspace's source-object store by their SHA-256; a retained inline set carries the bytes. When the
Host opens the workspace, `admit` admits each source document into world.evidence_doc:

- its body is copied to the data root's blobs/, named by its BLAKE3 (doc_digest); the workspace's
  object is only read, never moved or rewritten;
- a body world already holds is that document: nothing more is admitted for it;
- the same source document (its source and its own id: an accession and its primary document)
  with other bytes is a new version of its doc_id; any other document gets a new doc_id;
- an amending SEC filing names the doc_id of the filing it amends, as the issuer's sealed SEC
  selection plans link them;
- every row is a legacy one under lag rule AVAILABILITY: known from its 0.2.7 availability, the
  source's acceptance or a recorded import's capture, never its filing date.

A body the workspace no longer holds, or holds altered, is not admitted; the admission counts it.
The admission records its marks, the identities of the references it read, in its admission event;
equal marks at the next open end it there.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

from blake3 import blake3

from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.log import uuid7
from alphalattice.kernel.data_layer.project import definition_id
from alphalattice.kernel.data_layer.store import as_columns, newest, stage

from ..documents.contracts import AlternativeEvidenceDocumentReference
from ..publication.artifacts import AlternativeEvidenceArtifactStore
from ..sources.contracts import (
    SOURCE_COMMIT_CATEGORY,
    AcquiredEvidenceDocument,
    AcquiredEvidenceDocumentReference,
    AcquiredEvidenceSourceCommit,
    SecFilingInventoryRead,
    SecFilingSelectionPlan,
    source_object_bytes,
)

DATASET: Final = "evidence"
AVAILABILITY: Final = ("evidence-availability", 1)
"""The lag rule of 0.2.7's documents: known at their availability (lag_seconds 0)."""
PROVIDERS: Final = {
    "SEC_EDGAR": "SEC EDGAR, the official filings of US issuers, as 0.2.7 acquired them.",
    "ISSUER_RECORDED": "Documents a person recorded for an issuer, as 0.2.7 imported them.",
}
SETS: Final = "source-document-sets"
LINKS: Final = ("sec-selection-plans", "sec-inventory-reads")
"""The sealed SEC artifacts whose inventory entries link an amending filing to its original."""


@dataclass(frozen=True)
class _Body:
    """One acquired document's provenance with its original bytes, or None when unreadable."""

    reference: AcquiredEvidenceDocumentReference | AcquiredEvidenceDocument
    content: bytes | None


def admit(workspace: Path, workspace_id: str) -> dict[str, Any] | None:
    """Admit the evidence documents of the workspace at `workspace` into world; the rows admitted,
    the bodies it could not read and the documents world already held, or None when nothing was
    read: no Host history, no evidence artifacts, or marks equal to the last admission's when that
    one read every body and world still holds none of the originals its amendments wait for. So a
    body lost then is tried again at each open until it is admitted, and an amendment is linked
    at the first open after any workspace admits its original. Refused by name as StorePort
    refuses."""
    artifacts = AlternativeEvidenceArtifactStore(workspace / "runtime" / "artifacts")
    if not artifacts.root.is_dir():  # no evidence: the Host's StorePort is not opened for it
        return None
    port = history.store()
    if port is None:
        return None
    marks = _marks(artifacts.root)
    last = port.log.read(
        "SELECT json_extract(attrs, '$.marks'), json_extract(attrs, '$.unreadable'),"
        " json_extract(attrs, '$.waiting') FROM event WHERE type = 'admission'"
        " AND subject_id = ? AND json_extract(attrs, '$.dataset') = ? ORDER BY seq DESC LIMIT 1",
        (workspace_id, DATASET),
    )
    settled = last and json.loads(last[0][0]) == marks and not last[0][1]
    if settled and not _arrived(port, json.loads(last[0][2] or "[]")):
        return None
    started = port.log.clock()
    for name, text in PROVIDERS.items():
        history.define("provider", name, text)
    text = "A 0.2.7 evidence document is known at its availability: acceptance or capture."
    history.define("lag_rule", AVAILABILITY[0], text, {"lag_seconds": 0})
    sources = workspace / "runtime" / "evidence-knowledge"
    bodies = list(_bodies(artifacts, sources))
    admitted = _admit(port, bodies, _amended(artifacts))
    history.record(
        "admission",
        ("workspace", workspace_id),
        {"provider": sorted(PROVIDERS), "dataset": DATASET, "marks": marks} | admitted,
        phase="completed",
        started_us=started,
        node="alternative_evidence",
    )
    return admitted


def _marks(root: Path) -> list[str]:
    """The identities of every artifact the admission reads, in order: the references and the SEC
    plans and inventory reads that link an amendment to its original. A sealed artifact's identity
    is its content's hash, so no change leaves them equal."""
    return sorted(
        f"{category}/{identity}"
        for category in (SOURCE_COMMIT_CATEGORY, SETS, *LINKS)
        for identity in _identities(root / category)
    )


def _identities(directory: Path) -> list[str]:
    """Each sealed artifact's identity in `directory`, in order; none when it does not exist, and
    an unreadable folder raises its OSError, never reads as empty."""
    if not directory.is_dir():
        return []
    with os.scandir(directory) as entries:
        return sorted(e.name.removesuffix(".json") for e in entries if e.name.endswith(".json"))


def _bodies(artifacts: AlternativeEvidenceArtifactStore, sources: Path) -> Iterator[_Body]:
    """Each acquired document the workspace's commits and source sets name, once per body."""
    seen: set[tuple[Any, ...]] = set()
    references: list[AcquiredEvidenceDocumentReference | AcquiredEvidenceDocument] = [
        artifacts.load(SOURCE_COMMIT_CATEGORY, identity, AcquiredEvidenceSourceCommit).reference
        for identity in _identities(artifacts.root / SOURCE_COMMIT_CATEGORY)
    ]
    for identity in _identities(artifacts.root / SETS):
        references.extend(artifacts.load_source_set(identity).documents)
    for reference in references:
        once = (*_resource(reference), reference.source_content_hash)
        if once in seen:
            continue
        seen.add(once)
        if isinstance(reference, AcquiredEvidenceDocument):
            yield _Body(reference, reference.content)
            continue
        path = sources / reference.content_object_path
        try:
            content = source_object_bytes(
                path.read_bytes(), reference.content_sha256, reference.content_bytes
            )
        except (OSError, ValueError):  # missing or altered: 0.2.7 refuses to read it too
            content = None
        yield _Body(reference, content)


def _resource(
    reference: AcquiredEvidenceDocumentReference | AcquiredEvidenceDocument,
) -> tuple[str, str, str | None]:
    return (reference.source_name, reference.revision, reference.source_document_name)


def _source_ref(reference: AcquiredEvidenceDocumentReference | AcquiredEvidenceDocument) -> str:
    """The source's own id of a document: its accession (0.2.7's revision) and primary document."""
    name = reference.source_document_name
    return reference.revision if name is None else f"{reference.revision}/{name}"


def _amended(artifacts: AlternativeEvidenceArtifactStore) -> dict[str, str]:
    """Each amending SEC filing's accession, with the accession of the filing it amends, as the
    workspace's sealed selection plans and inventory reads link them."""
    plans = [
        artifacts.load("sec-selection-plans", identity, SecFilingSelectionPlan)
        for identity in _identities(artifacts.root / "sec-selection-plans")
    ]
    plans += [
        artifacts.load("sec-inventory-reads", identity, SecFilingInventoryRead).plan
        for identity in _identities(artifacts.root / "sec-inventory-reads")
    ]
    return {
        entry.accession: entry.amendment_of_accession
        for plan in plans
        for entry in (*plan.required, *plan.events)
        if entry.amendment_of_accession is not None
    }


def _arrived(port: Any, accessions: list[str]) -> bool:
    """Whether world now holds a document of any of these accessions."""
    if not accessions:
        return False
    found = port.lake.execute(
        f"SELECT 1 FROM {newest('evidence_doc')} WHERE NOT retracted"
        " AND split_part(source_ref, '/', 1) IN (SELECT unnest(?)) LIMIT 1",
        [accessions],
    ).fetchall()
    return bool(found)


def _admit(port: Any, bodies: list[_Body], amended: dict[str, str]) -> dict[str, Any]:
    """Each readable body as a world row, in rounds of at most one body per document, so a
    document acquired again with other bytes is admitted in its acquisition order; the rows world
    took, the bodies unreadable, those world already held, and the accessions of the originals
    its amendments wait for (`waiting`). A held amending filing whose original became known since
    is admitted again with its link, its next version."""
    held = port.lake.execute(
        f"SELECT CAST(doc_id AS VARCHAR), provider_id, source_ref, doc_digest,"
        f" CAST(amends_doc_id AS VARCHAR) FROM {newest('evidence_doc')}"
    ).fetchall()
    by_digest = {digest: (doc, linked) for doc, _, _, digest, linked in held}
    by_source = {(provider, ref): doc for doc, provider, ref, _, _ in held}
    unlinked: list[tuple[_Body, str, str]] = []
    unreadable = sum(body.content is None for body in bodies)
    readable = sorted(
        (body for body in bodies if body.content is not None),
        key=lambda body: str(body.reference.retrieved_at or ""),
    )
    rows: dict[str, list[dict[str, Any]]] = {}
    already = 0
    for body in readable:
        assert body.content is not None
        digest = blake3(body.content).hexdigest()
        if digest in by_digest:
            doc, linked = by_digest[digest]
            if linked is None:
                unlinked.append((body, doc, digest))
            else:
                already += 1
            continue
        provider = definition_id("provider", body.reference.source_name)
        source = (provider, _source_ref(body.reference))
        doc_id = by_source.setdefault(source, str(uuid7()))
        by_digest[digest] = (doc_id, None)
        _keep(port.root / "blobs", digest, body.content)
        rows.setdefault(body.reference.source_name, []).append(
            _row(body.reference, doc_id, digest, len(body.content))
        )
    waiting: set[str] = set()
    already += _link(rows, unlinked, amended, by_source, waiting)
    taken: dict[str, int] = {}
    for name, batch in rows.items():
        for part in _rounds(batch):
            with stage(
                port.root, "evidence_doc", definition_id("provider", name), _arrow(part),
                lag=AVAILABILITY,
            ) as staged:  # fmt: skip
                taken[name] = taken.get(name, 0) + port.commit(staged).rows
    return {"rows": taken, "unreadable": unreadable, "held": already, "waiting": sorted(waiting)}


def _link(
    rows: dict[str, list[dict[str, Any]]],
    unlinked: list[tuple[_Body, str, str]],
    amended: dict[str, str],
    by_source: dict[tuple[str, str], str],
    waiting: set[str],
) -> int:
    """Name each new amending filing's original, and add the next version of each held one whose
    original is now known; the accessions still unknown go to `waiting`. Returns the held
    documents that gain no link."""
    accessions = {ref.split("/")[0]: doc for (_, ref), doc in by_source.items()}

    def original(ref: str) -> str | None:
        amends = amended.get(ref.split("/")[0])
        if amends is not None and amends not in accessions:
            waiting.add(amends)
        return None if amends is None else accessions.get(amends)

    for batch in rows.values():
        for row in batch:
            row["amends_doc_id"] = original(row["source_ref"])
    still = 0
    for body, doc, digest in unlinked:
        link = original(_source_ref(body.reference))
        if link is None:
            still += 1
            continue
        assert body.content is not None
        row = _row(body.reference, doc, digest, len(body.content)) | {"amends_doc_id": link}
        rows.setdefault(body.reference.source_name, []).append(row)
    return still


def _rounds(batch: list[dict[str, Any]]) -> Iterable[list[dict[str, Any]]]:
    """The batch as admissions of one version per document each, in its order."""
    rounds: list[list[dict[str, Any]]] = []
    for row in batch:
        place = next((r for r in rounds if all(o["doc_id"] != row["doc_id"] for o in r)), None)
        if place is None:
            rounds.append(place := [])
        place.append(row)
    return rounds


def _row(
    reference: AcquiredEvidenceDocumentReference | AcquiredEvidenceDocument,
    doc_id: str,
    digest: str,
    size: int,
) -> dict[str, Any]:
    """One document's world row: 0.2.7's facts, each its own column."""
    basis = AlternativeEvidenceDocumentReference.availability_basis(
        reference.source_name, reference.accepted_at, reference.available_at
    )
    return {
        "doc_id": doc_id,
        "provider_id": definition_id("provider", reference.source_name),
        "source_ref": _source_ref(reference),
        "doc_digest": digest,
        "byte_count": size,
        "media_type": reference.media_type,
        "title": reference.title,
        "document_type": reference.document_type,
        "entity_id": reference.entity_id,
        "published_at": stamp(reference.published_at),
        "published_precision": reference.published_precision,
        "accepted_at": stamp(reference.accepted_at),
        "available_at": stamp(reference.available_at),
        "availability_basis": basis,
        "report_period_end": stamp(reference.report_period_end),
        "right_of_use": reference.source_right,
        "retrieved_at": stamp(reference.retrieved_at),
    }


def _keep(blobs: Path, digest: str, content: bytes) -> None:
    """Keep `content` under blobs/ at its digest once: written beside it and renamed into place,
    so no reader sees a partial body at a digest."""
    target = blobs / digest
    if target.is_file():
        return
    staged = blobs / f".{digest}.{os.getpid()}.staging"
    staged.write_bytes(content)
    os.replace(staged, target)


def stamp(value: date | None) -> str | None:
    """A time or date as StorePort reads it; None stays None."""
    return None if value is None else value.isoformat()


def _arrow(rows: list[dict[str, Any]]) -> Any:
    import pyarrow as pa  # lazy: the optional data extra, as DuckDB

    return pa.table(as_columns(rows))
