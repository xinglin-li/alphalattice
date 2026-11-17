"""A workspace's evidence documents admitted into world once per machine, through StorePort.

0.2.7 keeps each acquired document as a sealed reference (the evidence artifacts'
source-document-commits and source-document-sets) naming its original bytes in the evidence
workspace's source-object store by their SHA-256; a retained inline set carries the bytes. When the
Host opens the workspace, and when an acquisition ends, `admit` admits each source document into
world.evidence_doc:

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
The admission records its marks in its admission event: the fingerprint of the identities of the
sealed artifacts it read, their count and the latest time one was written. Equal marks at the next
admission end it there; after a clean one, the artifacts written later are read alone when the rest
still carry its fingerprint.
"""

from __future__ import annotations

import os
import struct
import threading
from collections.abc import Iterable, Iterator
from contextlib import closing
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
from .world_reads import source_ref

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
_ADMITTING: Final = threading.Lock()
MARKS_VERSION: Final = 1
"""The version of the marks' fingerprint encoding; an admission recorded under another reads all."""


@dataclass(frozen=True)
class _Body:
    """One acquired document's provenance with its original bytes, or None when unreadable."""

    reference: AcquiredEvidenceDocumentReference | AcquiredEvidenceDocument
    content: bytes | None


def admit(artifact_root: Path, sources: Path, workspace_id: str) -> dict[str, Any] | None:
    """Admit the evidence documents of a workspace, its artifacts under `artifact_root` and its
    source objects under `sources`, into world; the rows admitted, the bodies it could not read and
    the documents world already held, or None when nothing was read: no Host history, no evidence
    artifacts, or marks equal to the last admission's when that one read every body and world
    still holds none of the originals its amendments wait for. So a body lost then is tried again
    at each admission until it is admitted, and an amendment is linked at the first admission
    after any workspace admits its original. After a clean admission, one whose only change is new
    references reads only those, so an acquisition's admission costs its own documents. Refused by
    name as StorePort refuses. One admission runs at a time in a process, so two never mint two
    doc_ids for one document."""
    with _ADMITTING:
        return _admission(artifact_root, sources, workspace_id)


def _admission(artifact_root: Path, sources: Path, workspace_id: str) -> dict[str, Any] | None:
    artifacts = AlternativeEvidenceArtifactStore(artifact_root)
    if not artifacts.root.is_dir():  # no evidence: the Host's StorePort is not opened for it
        return None
    port = history.store()
    if port is None:
        return None
    started = port.log.clock()
    listed = _marks(artifacts.root)
    marked = _marked(listed)
    last = port.log.read(
        "SELECT json_extract(attrs, '$.marks'), json_extract(attrs, '$.unreadable'),"
        " json_extract(attrs, '$.waiting'), json_extract(attrs, '$.written'),"
        " json_extract(attrs, '$.before'), json_extract(attrs, '$.latest') FROM event"
        " WHERE type = 'admission'"
        " AND subject_id = ? AND json_extract(attrs, '$.dataset') = ? ORDER BY seq DESC LIMIT 1",
        (workspace_id, DATASET),
    )
    only: set[str] | None = None  # the artifacts to read; None reads every one
    amended: dict[str, str] | None = None
    if last:
        previous, unreadable, waiting, written, before, latest = last[0]
        if waiting:
            amended = _amended(artifacts)
        if not unreadable and not (amended and _arrived(port, amended)):
            if previous == marked["marks"]:
                return None
            if (
                written is not None
                and _fingerprint(m for m, at in listed.items() if at < written) == before
            ):
                at_last = {m for m, at in listed.items() if at == written}
                only = {m for m, at in listed.items() if at > written}
                if _fingerprint(at_last) != latest:  # written in the last one's instant: read again
                    only |= at_last
    for name, text in PROVIDERS.items():
        history.define("provider", name, text)
    text = "A 0.2.7 evidence document is known at its availability: acceptance or capture."
    history.define("lag_rule", AVAILABILITY[0], text, {"lag_seconds": 0})
    bodies = list(_bodies(artifacts, sources, only))
    admitted = _admit(port, bodies, _amended(artifacts) if amended is None else amended)
    history.record(
        "admission",
        ("workspace", workspace_id),
        {"provider": sorted(PROVIDERS), "dataset": DATASET, **marked}
        | {key: admitted[key] for key in ("rows", "unreadable", "held")}
        | {"waiting": len(admitted["waiting"])},
        phase="completed",
        started_us=started,
        node="alternative_evidence",
    )
    return admitted


def _marks(root: Path) -> dict[str, int]:
    """The identity of every artifact the admission reads, with when it was written, in
    nanoseconds as the file system stamps it: the references and the SEC plans and inventory
    reads that link an amendment to its original. A sealed artifact's identity is its content's
    hash, so no change keeps them."""
    marks: dict[str, int] = {}
    for category in (SOURCE_COMMIT_CATEGORY, SETS, *LINKS):
        if (root / category).is_dir():
            with os.scandir(root / category) as entries:
                for e in entries:
                    if e.name.endswith(".json"):
                        mark = f"{category}/{e.name.removesuffix('.json')}"
                        marks[mark] = e.stat().st_mtime_ns
    return marks


def _marked(listed: dict[str, int]) -> dict[str, Any]:
    """What an admission records of its marks, a fixed size whatever the workspace holds: the
    fingerprint of them all, their count, the latest time one was written, and the fingerprints
    of those written before that time and at it; so the next admission reads only what is newer,
    and again what was written in that same instant when it is not what this one read."""
    written = max(listed.values(), default=0)
    return {
        "marks": _fingerprint(listed),
        "marks_version": MARKS_VERSION,
        "marked": len(listed),
        "written": written,
        "before": _fingerprint(m for m, at in listed.items() if at < written),
        "latest": _fingerprint(m for m, at in listed.items() if at == written),
    }


def _fingerprint(marks: Iterable[str]) -> str:
    """A set of artifact ids' fingerprint (MARKS_VERSION 1): the BLAKE3 of the version as one
    unsigned byte, then each id in sorted order as its UTF-8 bytes, each preceded by their length
    as an unsigned 32-bit big-endian integer. It answers only whether a later admission sees the
    same artifacts; its reader is that admission, and it is never a key."""
    hasher = blake3(struct.pack(">B", MARKS_VERSION))
    for mark in sorted(marks):
        raw = mark.encode("utf-8")
        hasher.update(struct.pack(">I", len(raw)) + raw)
    return hasher.hexdigest()


def _identities(directory: Path) -> list[str]:
    """Each sealed artifact's identity in `directory`, in order; none when it does not exist, and
    an unreadable folder raises its OSError, never reads as empty."""
    if not directory.is_dir():
        return []
    with os.scandir(directory) as entries:
        return sorted(e.name.removesuffix(".json") for e in entries if e.name.endswith(".json"))


def _bodies(
    artifacts: AlternativeEvidenceArtifactStore, sources: Path, only: set[str] | None
) -> Iterator[_Body]:
    """Each acquired document the workspace's commits and source sets name, once per body: of
    every one, or of the artifacts `only` names."""

    def read(category: str) -> list[str]:
        names = _identities(artifacts.root / category)
        return [n for n in names if only is None or f"{category}/{n}" in only]

    seen: set[tuple[Any, ...]] = set()
    references: list[AcquiredEvidenceDocumentReference | AcquiredEvidenceDocument] = [
        artifacts.load(SOURCE_COMMIT_CATEGORY, identity, AcquiredEvidenceSourceCommit).reference
        for identity in read(SOURCE_COMMIT_CATEGORY)
    ]
    for identity in read(SETS):
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


def _arrived(port: Any, amended: dict[str, str]) -> bool:
    """Whether world now holds the original of an amending filing it holds without its link."""
    if not amended:
        return False
    with closing(port.lake.cursor()) as cursor:
        found = cursor.execute(
            "WITH link AS (SELECT unnest(?) AS amendment, unnest(?) AS original)"
            f" SELECT 1 FROM link JOIN {newest('evidence_doc')} a"
            " ON split_part(a.source_ref, '/', 1) = link.amendment AND a.amends_doc_id IS NULL"
            f" JOIN {newest('evidence_doc')} o ON split_part(o.source_ref, '/', 1) = link.original"
            " WHERE NOT a.retracted AND NOT o.retracted LIMIT 1",
            [list(amended), list(amended.values())],
        ).fetchall()
    return bool(found)


def _admit(port: Any, bodies: list[_Body], amended: dict[str, str]) -> dict[str, Any]:
    """Each readable body as a world row, in rounds of at most one body per document, so a
    document acquired again with other bytes is admitted in its acquisition order; the rows world
    took, the bodies unreadable, those world already held, and the accessions of the originals
    its amendments wait for (`waiting`). An amending filing world holds without its link, whose
    link and original are both known now, is admitted again from its own row with the link, its
    next version: no body is read for it."""
    with closing(port.lake.cursor()) as cursor:
        held = cursor.execute(
            f"SELECT CAST(doc_id AS VARCHAR), provider_id, source_ref, doc_digest,"
            f" CAST(amends_doc_id AS VARCHAR) FROM {newest('evidence_doc')}"
        ).fetchall()
    by_digest = {digest: (doc, linked) for doc, _, _, digest, linked in held}
    by_source = {(provider, ref): doc for doc, provider, ref, _, _ in held}
    unreadable = sum(body.content is None for body in bodies)
    readable = sorted(
        (body for body in bodies if body.content is not None),
        key=lambda body: str(body.reference.retrieved_at or ""),
    )
    rows: dict[str, list[dict[str, Any]]] = {}
    already: set[str] = set()
    for body in readable:
        assert body.content is not None
        digest = blake3(body.content).hexdigest()
        if digest in by_digest:
            already.add(by_digest[digest][0])
            continue
        provider = definition_id("provider", body.reference.source_name)
        source = (provider, source_ref(body.reference))
        doc_id = by_source.setdefault(source, str(uuid7()))
        by_digest[digest] = (doc_id, None)
        _keep(port.root / "blobs", digest, body.content)
        rows.setdefault(body.reference.source_name, []).append(
            _row(body.reference, doc_id, digest, len(body.content))
        )
    waiting: set[str] = set()
    already -= _link(port, rows, amended, by_source, waiting)
    taken: dict[str, int] = {}
    for name, batch in rows.items():
        for part in _rounds(batch):
            with stage(
                port.root, "evidence_doc", definition_id("provider", name), _arrow(part),
                lag=AVAILABILITY,
            ) as staged:  # fmt: skip
                taken[name] = taken.get(name, 0) + port.commit(staged).rows
    return {
        "rows": taken,
        "unreadable": unreadable,
        "held": len(already),
        "waiting": sorted(waiting),
    }


def _link(
    port: Any,
    rows: dict[str, list[dict[str, Any]]],
    amended: dict[str, str],
    by_source: dict[tuple[str, str], str],
    waiting: set[str],
) -> set[str]:
    """Name each new amending filing's original, and add the next version of each one world holds
    without its link whose original is now known, from its own row; the accessions still unknown
    go to `waiting`. Returns the documents linked so."""
    accessions = {ref.split("/")[0]: doc for (_, ref), doc in by_source.items()}

    def original(ref: str) -> str | None:
        amends = amended.get(ref.split("/")[0])
        if amends is not None and amends not in accessions:
            waiting.add(amends)
        return None if amends is None else accessions.get(amends)

    for batch in rows.values():
        for row in batch:
            row["amends_doc_id"] = original(row["source_ref"])
    names = {definition_id("provider", name): name for name in PROVIDERS}
    with closing(port.lake.cursor()) as cursor:
        unlinked = cursor.execute(
            f"SELECT CAST(doc_id AS VARCHAR) AS doc_id, {', '.join(FACTS)}"
            f" FROM {newest('evidence_doc')} WHERE amends_doc_id IS NULL AND NOT retracted"
            " AND split_part(source_ref, '/', 1) IN (SELECT unnest(?))",
            [list(amended)],
        )
        columns = [c[0] for c in unlinked.description]
        held = [dict(zip(columns, row, strict=True)) for row in unlinked.fetchall()]
    linked: set[str] = set()
    for row in held:
        link = original(row["source_ref"])
        if link is not None:
            row |= {key: stamp(row[key]) for key in TIMES} | {"amends_doc_id": link}
            rows.setdefault(names[row["provider_id"]], []).append(row)
            linked.add(row["doc_id"])
    return linked


FACTS: Final = (
    "provider_id", "source_ref", "doc_digest", "byte_count", "media_type", "title",
    "document_type", "entity_id", "published_at", "published_precision", "accepted_at",
    "available_at", "availability_basis", "report_period_end", "right_of_use", "retrieved_at",
)  # fmt: skip
"""A document version's own facts, as `_row` writes them: its next version keeps each."""
TIMES: Final = ("published_at", "accepted_at", "available_at", "report_period_end", "retrieved_at")


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
        "source_ref": source_ref(reference),
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
