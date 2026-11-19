"""An evidence document's body read from world at its selection's cutoff, or from the workspace.

The Host's evidence readers take a reference's original bytes through `body`: in a Host whose
StorePort is open, and unless ALPHALATTICE_EVIDENCE_READS is `workspace` (the rollback), the
version of its source document whose body is the reference's own and that world knew by the
cutoff, its blob under the data root's blobs/; a body world knew only after the cutoff is refused
as missing, whichever reference or set carries it (`known`, by the body's digest). A body world
does not hold, never admitted, and every read outside such a Host, the workspace's own
source-object store answers, as 0.2.7 did.
"""

from __future__ import annotations

import os
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Final, NamedTuple

from blake3 import blake3

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.shared_kernel.domain.serialization import sha256_hex

from ..sources.contracts import (
    AcquiredEvidenceDocument,
    AcquiredEvidenceDocumentReference,
)

SETTING: Final = "ALPHALATTICE_EVIDENCE_READS"
CHOICES: Final = ("world", "workspace")


def chosen() -> str:
    """The setting's value; refused by name when it names neither choice."""
    value = os.environ.get(SETTING, CHOICES[0])
    if value not in CHOICES:
        raise DataError(
            f"{SETTING} is {value!r}; set it to world or workspace, or unset it for world",
            code="alternative_evidence.evidence_reads_unknown",
        )
    return value


def serving() -> Any | None:
    """The Host's StorePort when world serves the evidence reads, else None: the setting chooses
    the workspace, or this process holds no Host history."""
    return history.store() if chosen() == "world" else None


class Served(NamedTuple):
    """What world holds of one reference's body: its bytes as known at the cutoff, or None; and
    whether world holds that body at all, known then or later. A body world does not hold was
    never admitted, so the workspace's own store answers for it, as the market's tables do."""

    content: bytes | None
    held: bool


KNOWN: Final = "d.known_asof <= ?::TIMESTAMPTZ"
"""Whether world knew a version of a document by a cutoff (the one parameter). A later version
replaces it for a read of the document's current state, never for a reference that names its
exact bytes."""


def visible(port: Any, cutoff_us: int) -> frozenset[tuple[str, str]]:
    """World document versions visible at the cutoff, before retrieval candidate selection."""
    with closing(port.lake.cursor()) as cursor:
        rows = cursor.execute(
            "SELECT CAST(doc_id AS VARCHAR), CAST(version_id AS VARCHAR) FROM evidence_doc_asof(?)",
            [cutoff_us],
        ).fetchall()
    return frozenset((str(doc_id), str(version_id)) for doc_id, version_id in rows)


def identity(port: Any, content: bytes, cutoff_us: int) -> tuple[str, str, int]:
    """Find the world-minted document and version for admitted bytes visible at the cutoff."""
    body_digest = blake3(content).hexdigest()
    with closing(port.lake.cursor()) as cursor:
        rows = cursor.execute(
            "SELECT CAST(doc_id AS VARCHAR), CAST(version_id AS VARCHAR), epoch_us(known_asof)"
            " FROM evidence_doc_asof(?) WHERE doc_digest = ? AND byte_count = ?",
            [cutoff_us, body_digest, len(content)],
        ).fetchall()
    if len(rows) != 1:
        raise DataError(
            f"world evidence body {body_digest} is not known uniquely at {cutoff_us}",
            code="alternative_evidence.source_object_missing",
        )
    doc_id, version_id, known_us = rows[0]
    return str(doc_id), str(version_id), int(known_us)


def exact(port: Any, doc_id: str, version_id: str, cutoff_us: int) -> bytes:
    """Read the exact world document version visible at the cutoff, with no workspace fallback."""
    with closing(port.lake.cursor()) as cursor:
        row = cursor.execute(
            "SELECT doc_digest, byte_count FROM evidence_doc_asof(?)"
            " WHERE doc_id = ?::UUID AND version_id = ?::UUID",
            [cutoff_us, doc_id, version_id],
        ).fetchone()
    if row is None:
        raise DataError(
            f"world evidence document {doc_id}@{version_id} is not known at {cutoff_us}",
            code="alternative_evidence.source_object_missing",
        )
    digest, byte_count = row
    content = _blob(port.root / "blobs" / digest)
    if len(content) != byte_count or blake3(content).hexdigest() != digest:
        raise DataError(
            f"world evidence body {doc_id}@{version_id} differs from its digest",
            code="alternative_evidence.source_object_tampered",
        )
    return content


def body(port: Any, reference: AcquiredEvidenceDocumentReference, as_of: datetime) -> Served:
    """The reference's original bytes as world knew them at `as_of`: the version of its source
    document whose body is the reference's own (its length and SHA-256), known by then. A
    reference names exact bytes, so a version known later does not hide the one a sealed
    selection read. A document admitted from a reference sealed before its name was recorded, or
    read by one, matches by its accession and its bytes."""
    named = reference.source_document_name is not None
    with closing(port.lake.cursor()) as cursor:  # its own: units read at once
        versions = cursor.execute(
            f"SELECT doc_digest, {KNOWN} AS known"
            " FROM lake.world.evidence_doc d WHERE provider_id = ? AND byte_count = ?"
            " AND (source_ref IN (?, ?) OR starts_with(source_ref, ?))"
            " ORDER BY known DESC, known_asof DESC",
            [
                as_of,
                port.log.lineage("provider", reference.source_name),
                reference.content_bytes,
                source_ref(reference),
                reference.revision,  # admitted from a reference without its document's name
                "\n" if named else f"{reference.revision}/",  # no source_ref starts with a newline
            ],
        ).fetchall()
    held = False
    for digest, known in versions:
        content = _blob(port.root / "blobs" / digest)
        if sha256_hex(content) != reference.content_sha256:
            continue  # another version's body, of the same length
        if known:
            return Served(content, True)
        held = True  # this body, known only after the cutoff
    return Served(None, held)


def known(port: Any, content: bytes, as_of: datetime) -> bool | None:
    """Whether world knew a body at `as_of`, by its digest whichever reference names it: True,
    False when world holds it but learned it only later, None when world does not hold it. A
    body world admitted once under another source is that same document's body (one row per
    digest), so its availability is the one that row carries."""
    with closing(port.lake.cursor()) as cursor:
        held, knew = cursor.execute(
            f"SELECT count(*) > 0, coalesce(bool_or({KNOWN}), false)"
            " FROM lake.world.evidence_doc d WHERE doc_digest = ?",
            [as_of, blake3(content).hexdigest()],
        ).fetchone()
    return bool(knew) if held else None


def _blob(path: Path) -> bytes:
    """A body world holds; refused by name when its blob is not there, as world holds it."""
    try:
        return path.read_bytes()
    except OSError as error:
        raise DataError(
            f"world holds the evidence body {path.name} but it is not under {path.parent} "
            f"({error}); restore the data root from its last backup, or set {SETTING}=workspace "
            "to read the workspace's own store",
            code="alternative_evidence.world_body_missing",
        ) from error


def source_ref(reference: AcquiredEvidenceDocumentReference | AcquiredEvidenceDocument) -> str:
    """The source's own id of a document, as world admits it: its accession (0.2.7's revision) and
    its primary document, when the reference names it."""
    name = reference.source_document_name
    return reference.revision if name is None else f"{reference.revision}/{name}"
