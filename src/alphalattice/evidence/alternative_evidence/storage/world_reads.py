"""An evidence document's body read from world at its selection's cutoff, or from the workspace.

The Host's evidence readers take a reference's original bytes through `body`: in a Host whose
StorePort is open, the version of its source document whose body is the reference's own and
that world knew by the cutoff, its blob under the data root's blobs/; a body world knew only
after the cutoff is refused as missing, whichever reference or set carries it (`known`, by the
body's digest). A body world does not hold, never admitted, and every read outside such a Host,
the workspace's own source-object store answers, as 0.2.7 did.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Final, NamedTuple

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer import history
from alphalattice.foundation.data_layer.root import data_root
from alphalattice.kernel.identity import blake3_digest, identity_blake3, integrity_blake3

from ..sources.contracts import (
    AcquiredEvidenceDocument,
    AcquiredEvidenceDocumentReference,
    source_object_bytes,
)

WORLD_OBJECT: Final = b"alphalattice-source-object world 1\n"
"""A source address pointing at its BLAKE3-addressed body in the machine data root."""


def workspace_body(
    path: Path, reference: AcquiredEvidenceDocumentReference, *, root: Path | None = None
) -> bytes:
    """Read the original object or its world locator at the sealed BLAKE3 address."""
    stored = path.read_bytes()
    if stored.startswith(WORLD_OBJECT) and (
        len(stored) != reference.content_bytes or blake3_digest(stored) != reference.content_digest
    ):
        if root is None:
            port = serving()
            root = data_root() if port is None else port.root
        stored = locator_body(stored, root)
    return source_object_bytes(stored, reference.content_digest, reference.content_bytes)


def locator_body(locator: bytes, root: Path) -> bytes:
    """Read the world body at a locator and verify its content address."""
    key = locator[len(WORLD_OBJECT) :]
    if not locator.startswith(WORLD_OBJECT) or re.fullmatch(rb"[0-9a-f]{64}", key) is None:
        raise ValueError("alternative_evidence.source_object_tampered")
    digest = key.decode("ascii")
    body = _blob(root / "blobs" / digest)
    if integrity_blake3(body).hexdigest() != digest:
        raise ValueError("alternative_evidence.source_object_tampered")
    return body


def serving() -> Any | None:
    """The Host's StorePort when world serves the evidence reads, else None: this process holds
    no Host history."""
    return history.store()


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
    body_digest = identity_blake3(content).hexdigest()
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
    if len(content) != byte_count or integrity_blake3(content).hexdigest() != digest:
        raise DataError(
            f"world evidence body {doc_id}@{version_id} differs from its digest",
            code="alternative_evidence.source_object_tampered",
        )
    return content


Versions = tuple[tuple[str, bool], ...]
"""A reference's candidate bodies in world: each version's body digest and whether world knew it by
the cutoff, known ones first, then the latest known."""


def versions(
    port: Any, references: Sequence[AcquiredEvidenceDocumentReference], as_of: datetime
) -> tuple[Versions, ...]:
    """Each reference's candidate versions, in order, from one statement for them all: a version of
    its source document with the reference's byte count. A document admitted from a reference
    sealed before its name was recorded, or read by one, matches by its accession."""
    if not references:
        return ()
    with closing(port.lake.cursor()) as cursor:  # its own: units read at once
        rows = cursor.execute(
            f"SELECT w.i, d.doc_digest, {KNOWN} AS known FROM lake.world.evidence_doc d JOIN"
            " (SELECT unnest(?::INTEGER[]) AS i, unnest(?::VARCHAR[]) AS provider,"
            " unnest(?::BIGINT[]) AS size, unnest(?::VARCHAR[]) AS ref,"
            " unnest(?::VARCHAR[]) AS revision, unnest(?::VARCHAR[]) AS prefix) w"
            " ON d.provider_id = w.provider AND d.byte_count = w.size"
            " AND (d.source_ref IN (w.ref, w.revision) OR starts_with(d.source_ref, w.prefix))"
            " ORDER BY w.i, known DESC, d.known_asof DESC",
            [
                as_of,
                list(range(len(references))),
                [str(port.log.lineage("provider", value.source_name)) for value in references],
                [value.content_bytes for value in references],
                [source_ref(value) for value in references],
                # admitted from a reference without its document's name
                [value.revision for value in references],
                # no source_ref starts with a newline
                [
                    "\n" if value.source_document_name is not None else f"{value.revision}/"
                    for value in references
                ],
            ],
        ).fetchall()
    found: list[list[tuple[str, bool]]] = [[] for _ in references]
    for index, digest, known in rows:
        found[index].append((digest, known))
    return tuple(tuple(candidates) for candidates in found)


def body(
    port: Any,
    reference: AcquiredEvidenceDocumentReference,
    as_of: datetime,
    found: Versions | None = None,
) -> Served:
    """The reference's original bytes as world knew them at `as_of`: the version of its source
    document whose body is the reference's own, known by then. A reference names exact bytes, so
    a version known later does not hide the one a sealed selection read. `found` is its `versions`
    read with others' in one statement; without it, this read makes its own."""
    held = False
    for digest, known in versions(port, (reference,), as_of)[0] if found is None else found:
        content = _blob(port.root / "blobs" / digest)
        if blake3_digest(content) != reference.content_digest:
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
            [as_of, identity_blake3(content).hexdigest()],
        ).fetchone()
    return bool(knew) if held else None


def _blob(path: Path) -> bytes:
    """A body world holds; refused by name when its blob is not there, as world holds it."""
    try:
        return path.read_bytes()
    except OSError as error:
        raise DataError(
            f"world holds the evidence body {path.name} but it is not under {path.parent} "
            f"({error}); restore the data root from its last backup",
            code="alternative_evidence.world_body_missing",
        ) from error


def source_ref(reference: AcquiredEvidenceDocumentReference | AcquiredEvidenceDocument) -> str:
    """The source's own id of a document, as world admits it: its accession (0.2.7's revision) and
    its primary document, when the reference names it."""
    name = reference.source_document_name
    return reference.revision if name is None else f"{reference.revision}/{name}"
