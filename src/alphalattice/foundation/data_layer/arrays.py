"""The array door: an array with no coordinates, kept once per machine through StorePort.

A lane is named by the digest sealed documents cite (`content_digest`): SHA-256 over its columns'
C-order bytes, in order, as ContentAddressedStore named it. Its body is one Parquet table under
blobs/<artifact_id>, each column flat and plain-encoded (never a dictionary, so every value's bits
stay, NaN payloads and signed zeros included), the shape in its metadata: the form the old store
wrote, so a cited digest holds. A legacy lane known only by its packed bytes is a `packed` array:
one binary value. Its profile is a row of the log's artifact table, type `array`: the kind, shape
and each column's dtype in `shape`, storage blob, the cited digest in `content_digest`. Its
artifact_id is the BLAKE3 of FORM, that profile, the cited digest and the BLAKE3 of the bytes,
the digest computed over the same bytes: equal bytes of another dtype or shape are another
array, and an id names the digest its bytes were kept under. A workspace that keeps a lane
holds it by one live_ref (`artifact.held`).

Write order. The body is written beside its place (blobs/.<artifact_id>.<pid>.staging), then its
profile is recorded (`artifact.produced`, carrying the writing workspace's hold), then, once the
profile has committed, the body is renamed into place (the event's `committed` action). So no body
is in place without a committed profile, and a crash leaves at most a staged body: StorePort's
start moves it into place when its profile was committed and deletes it when not (`settle`).
Until its batch commits, the writing process reads the body where it is staged. A body is read
across processes only by a digest a document cites, and its writer commits the profiles before it
writes the document (`durable=False` defers that commit to it), so no document names a lane a
crash could lose.

A workspace that opens on another data root (a copy, a restored backup) carries its lanes in the
old form (`carry`); the Host that opens it moves them onto its own log (`import_carried`).

Every read checks the profile (kind, shape, dtype) against the body and proves the bytes by the
artifact_id: recomputed over the profile, the digest the read asks for and the body's bytes, it
must equal the profile's. A wrong dtype or shape, a changed body, or a profile that names another
array's body under the asked digest is refused by name. A lane file read in place (`read_file`)
is proved by its SHA-256 against its name.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.chain import canonical
from alphalattice.foundation.data_layer.log import HostLog
from alphalattice.kernel.identity import identity_blake3, integrity_blake3, integrity_sha256

TYPE: Final = "array"
"""The artifact type of every array the door keeps."""
FORM: Final = b"alphalattice.array/1\x00"
"""The versioned start of what an artifact_id hashes: then the profile, the cited digest and the
bytes' BLAKE3."""
PACKED: Final = "packed"
MEANING: Final = "An array with no coordinates: a numerical lane a sealed document cites by digest."
STAGED: Final = ".staging"
"""A body's suffix while it is written beside its place under blobs/."""
CARRIED: Final = Path("artifacts", "portfolio-strategy-lab", "carried")
"""Where a workspace copy carries its lanes, relative to the copy's folder (`carry`)."""

type Read = Callable[[str, tuple[Any, ...]], list[Any]]
"""A query on the log: the Host's `HostLog.read`, or an attached reader's."""

THREADED: Final = 1 << 17
"""From this many bytes BLAKE3 hashes on every core; below it one thread is faster."""

_writing = threading.Lock()
"""One write at a time in this process: a second writer of a lane sees the first's profile."""

_unplaced: dict[Path, Path] = {}
"""This process's bodies whose profiles wait in the log's open batch, by their place: each is
staged until its profile commits."""


def digest(columns: Mapping[str, npt.NDArray[Any]]) -> str:
    """The digest a lane is cited by: SHA-256 over its columns' C-order bytes, in order."""
    return _cited(
        [memoryview(np.ascontiguousarray(values)).cast("B") for values in columns.values()]
    )


def packed_digest(payload: bytes) -> str:
    """The unchanged lane identity over packed bytes, without constructing an array."""
    return _cited([memoryview(payload)])


def _cited(buffers: list[memoryview]) -> str:
    """The one SHA-256 the door computes: the digest documents cite, over the bytes in order."""
    found = integrity_sha256()
    for buffer in buffers:
        found.update(buffer)
    return found.hexdigest()


def keep(
    root: Path,
    log: HostLog,
    columns: Mapping[str, npt.NDArray[Any]],
    *,
    workspace: str | None,
    durable: bool = True,
) -> str:
    """Keep numerical columns of one shape as one array, held by `workspace`; returns its digest.

    The same columns again add only the workspace's hold. Equal bytes already kept with another
    dtype or shape are refused (`data_layer.array_digest_reused`): a digest names one array.
    `durable=False` leaves the profile in the log's open batch: its caller commits it before it
    writes anything that cites the digest.
    """
    return _keep(root, log, describe(columns), workspace, durable)


def keep_packed(
    root: Path,
    log: HostLog,
    payload: bytes,
    *,
    workspace: str | None,
    durable: bool = True,
    capacity: Callable[[int], None] | None = None,
) -> str:
    """Keep a legacy lane known only by its packed bytes (its dtype and shape were never written);
    returns the digest, the SHA-256 of `payload`."""
    return _keep(root, log, describe_packed(payload), workspace, durable, capacity)


@dataclass(frozen=True, slots=True)
class Described:
    """An array as the door names and writes it, before anything is written."""

    profile: dict[str, Any]
    """Its kind, shape and each column's name and dtype."""
    content_digest: str
    artifact_id: str
    body: Callable[[], bytes]
    """Its Parquet body, made when written."""


def describe(columns: Mapping[str, npt.NDArray[Any]]) -> Described:
    """Numerical columns of one shape as one array: its profile, the digest documents cite, its
    artifact_id and its body. Refused by name unless the columns are one or more of one shape."""
    arrays = {name: np.ascontiguousarray(values) for name, values in columns.items()}
    shapes = {array.shape for array in arrays.values()}
    if not arrays or len(shapes) != 1:
        raise DataError(
            "an array's columns are one or more of one shape", code="data_layer.array_invalid"
        )
    described = {
        "kind": "columns",
        "shape": list(shapes.pop()),
        "columns": [[name, array.dtype.str] for name, array in arrays.items()],
    }
    buffers = [memoryview(array).cast("B") for array in arrays.values()]
    return _described(described, buffers, lambda: _columns_body(arrays, described))


def describe_packed(payload: bytes) -> Described:
    """A legacy lane known only by its packed bytes, as `describe` names an array."""
    described = {"kind": PACKED, "shape": [len(payload)], "columns": [[PACKED, "bytes"]]}
    return _described(described, [memoryview(payload)], lambda: _packed_body(payload))


def _described(
    described: dict[str, Any], buffers: list[memoryview], body: Callable[[], bytes]
) -> Described:
    cited = _cited(buffers)
    return Described(described, cited, _named(described, cited, buffers).hexdigest(), body)


def profile(read: Read, content_digest: str) -> dict[str, Any] | None:
    """The array a digest names: its artifact_id, body state and profile; None when none does."""
    rows = read(
        "SELECT artifact_id, body_state, shape FROM artifact WHERE content_digest = ? AND type_id ="
        " (SELECT definition_id FROM definition_name WHERE family = 'artifact_type' AND name = ?)",
        (content_digest, TYPE),
    )
    if not rows:
        return None
    artifact, state, shape = rows[0]
    return {"artifact_id": artifact, "body_state": state, **json.loads(shape)}


def read_columns(root: Path, read: Read, content_digest: str) -> dict[str, npt.NDArray[Any]]:
    """The flat columns of the array a digest names, checked against its profile and proved."""
    known, table = _body(root, read, content_digest)
    if known["kind"] == PACKED:
        raise _mismatch(content_digest, "its kind: it is packed bytes, read as bytes")
    columns = _columns(known, table, content_digest)
    _prove(known, [memoryview(values).cast("B") for values in columns.values()], content_digest)
    return columns


def read_packed(root: Path, read: Read, content_digest: str, *, packed_only: bool = False) -> bytes:
    """The bytes a digest names: the array's columns' C-order bytes, in order, proved."""
    known, table = _body(root, read, content_digest)
    if known["kind"] == PACKED:
        if table.num_rows != 1 or table.column_names != [PACKED]:
            raise _mismatch(content_digest, "its packed form")
        payload = table.column(PACKED)[0].as_buffer()
        if payload is None or len(payload) != known["shape"][0]:
            raise _mismatch(content_digest, "its size")
        _prove(known, [memoryview(payload)], content_digest)
        return payload.to_pybytes()
    if packed_only:
        raise _mismatch(content_digest, "its kind: it is not packed")
    columns = _columns(known, table, content_digest)
    buffers = [memoryview(values).cast("B") for values in columns.values()]
    _prove(known, buffers, content_digest)
    return b"".join(buffers)


def verify(root: Path, read: Read, content_digest: str, artifact_id: str) -> None:
    """Verify a retained body against its own profile, name, and cited column-bytes identity."""
    known = profile(read, content_digest)
    if known is None or known["artifact_id"] != artifact_id:
        raise _tampered(content_digest)
    body = read_packed(root, read, content_digest)
    if _cited([memoryview(body)]) != content_digest:
        raise _tampered(content_digest)


def settle(root: Path, read: Read, collect: Callable[[str], bool]) -> dict[str, list[str]]:
    """Each body a crash left staged under blobs/: moved into place when its profile was committed,
    else handed to `collect` (StorePort's `delete`, True once gone). StorePort's start calls it;
    no write of this process runs meanwhile. Returns the handles placed, collected and waiting."""
    report: dict[str, list[str]] = {"placed": [], "collected": [], "waiting": []}
    blobs = root / "blobs"
    if not blobs.is_dir():
        return report
    with _writing:
        named = sorted(blobs.iterdir())  # an unreadable blobs/ stops the start: never empty
        waiting = set(_unplaced.values())  # this process's own, placed when their batch commits
        staging = (p for p in named if p.name.startswith(".") and p.name.endswith(STAGED))
        for staged in (p for p in staging if p not in waiting):
            name = staged.name.removeprefix(".").removesuffix(STAGED).rsplit(".", 1)[0]
            target = blobs / name
            rows = read("SELECT body_state FROM artifact WHERE artifact_id = ?", (name,))
            if rows and rows[0][0] == "present" and not target.is_file():
                os.replace(staged, target)
                report["placed"].append(f"blobs/{name}")
            else:
                handle = f"blobs/{staged.name}"
                report["collected" if collect(handle) else "waiting"].append(handle)
    return report


def keep_file(
    root: Path, log: HostLog, path: Path, *, workspace: str | None, durable: bool = True
) -> str:
    """Keep a lane file in the old store's form (`<digest>.parquet`, or legacy packed
    `<digest>.bin`) through the door; its digest must be its name, or it is refused by name."""
    found = _decoded(path)
    if isinstance(found, bytes):
        kept = keep_packed(root, log, found, workspace=workspace, durable=durable)
    else:
        kept = keep(root, log, found, workspace=workspace, durable=durable)
    if kept != path.stem:
        raise _tampered(path.stem)
    return kept


def read_file(path: Path) -> dict[str, npt.NDArray[Any]] | bytes:
    """A lane file in the old store's form, read in place and proved: a Parquet table's columns in
    their shape, or a legacy packed lane's bytes, whose digest is the file's name. A file that
    does not decode or does not prove its name is refused by name."""
    found = _decoded(path)
    cited = _cited([memoryview(found)]) if isinstance(found, bytes) else digest(found)
    if cited != path.stem:
        raise _tampered(path.stem)
    return found


def _decoded(path: Path) -> dict[str, npt.NDArray[Any]] | bytes:
    if path.suffix == ".bin":
        return path.read_bytes()
    try:
        table = pq.ParquetFile(path).read()
        shape = json.loads((table.schema.metadata or {}).get(b"shape", b"null"))
        return {
            name: np.ascontiguousarray(table.column(name).to_numpy()).reshape(shape)
            for name in table.column_names
        }
    except (OSError, ValueError, TypeError, pa.ArrowException) as error:
        raise _tampered(path.stem) from error


def carry(read: Read, root: Path, workspace: str, copy: Path) -> int:
    """Write each array `workspace` holds into the workspace copy `copy` in the old store's form,
    under CARRIED: a Parquet body as it is, a packed one as its bytes. A file the copy has is kept:
    each is written whole under another name. Returns the arrays the copy now carries.

    contract: carried lanes retire when a copy carries its own segment of the log, as carried
    Tasks do (`carry_tasks`)."""
    rows = read(
        "SELECT a.artifact_id, a.content_digest, a.shape FROM live_ref l"
        " JOIN artifact a ON a.artifact_id = l.artifact_id"
        " WHERE l.root_kind = 'workspace' AND l.root_id = ? AND a.body_state = 'present'"
        " AND a.type_id = (SELECT definition_id FROM definition_name"
        " WHERE family = 'artifact_type' AND name = ?) ORDER BY a.content_digest",
        (workspace, TYPE),
    )
    folder = copy / CARRIED
    for artifact, content_digest, shape in rows:
        packed = json.loads(shape)["kind"] == PACKED
        target = folder / f"{content_digest}{'.bin' if packed else '.parquet'}"
        if target.is_file():
            continue
        payload = read_packed(root, read, content_digest) if packed else _read(root, artifact)
        folder.mkdir(parents=True, exist_ok=True)
        _place(target, payload)
    return len(rows)


def import_carried(root: Path, log: HostLog, folder: Path, *, workspace: str) -> int:
    """Move the lanes a copy carries (`folder / CARRIED`) onto `log`, held by `workspace`: the
    copy's Host does at its open. A lane `workspace` holds already is not read again. Returns the
    lanes kept."""
    carried = folder / CARRIED
    if not carried.is_dir():
        return 0
    kept = 0
    for path in sorted(carried.iterdir()):
        if path.suffix not in {".parquet", ".bin"} or path.name.startswith("."):
            continue
        known = profile(log.read, path.stem)
        if known is not None and log.read(
            "SELECT 1 FROM live_ref WHERE artifact_id = ? AND root_kind = 'workspace'"
            " AND root_id = ?",
            (known["artifact_id"], workspace),
        ):
            continue
        keep_file(root, log, path, workspace=workspace, durable=False)
        kept += 1
    log.flush()
    return kept


def _keep(
    root: Path,
    log: HostLog,
    array: Described,
    workspace: str | None,
    durable: bool,
    capacity: Callable[[int], None] | None = None,
) -> str:
    described, content_digest, artifact = array.profile, array.content_digest, array.artifact_id
    body = array.body
    target = root / "blobs" / artifact
    with _writing:
        known = profile(log.read, content_digest)
        if known is not None:
            if known["artifact_id"] != artifact:
                raise DataError(
                    f"the digest {content_digest} already names an array of another dtype or shape",
                    code="data_layer.array_digest_reused",
                )
            if known["body_state"] != "present":
                raise DataError(
                    f"the array {content_digest} was collected", code="data_layer.array_collected"
                )
            if target not in _unplaced and not target.is_file():
                payload = body()
                if capacity is not None:
                    capacity(len(payload))
                _place(target, payload)  # its committed profile's body a crash left staged
            _hold(log, artifact, workspace)
        else:
            payload = body()
            if capacity is not None:
                capacity(len(payload))
            staged = _staged(target)
            staged.write_bytes(payload)
            _unplaced[target] = staged
            current = log.read(
                "SELECT version FROM definition_current"
                " WHERE family = 'artifact_type' AND name = ?",
                (TYPE,),
            )
            attrs = {
                "type": TYPE,
                "type_version": int(current[0][0]),  # the registered one: no version defaults
                "identity_kind": "content",
                "storage": "blob",
                "meaning": MEANING,
                "shape": described,
                "size_bytes": len(payload),
                "content_digest": content_digest,
                "held": workspace is not None,
            }
            try:
                log.record(
                    "artifact.produced",
                    ("artifact", artifact),
                    attrs,
                    workspace=workspace,
                    committed=lambda done: _settled(target, staged, committed=done),
                )
            except BaseException:
                del _unplaced[target]  # refused: its staged body goes at the next start
                raise
        if durable:
            log.flush()
    return content_digest


def _hold(log: HostLog, artifact: str, workspace: str | None) -> None:
    held = workspace is None or log.read(
        "SELECT 1 FROM live_ref WHERE artifact_id = ? AND root_kind = 'workspace' AND root_id = ?",
        (artifact, workspace),
    )
    if not held:
        log.record("artifact.held", ("artifact", artifact), {}, workspace=workspace)


def _staged(target: Path) -> Path:
    return target.with_name(f".{target.name}.{os.getpid()}{STAGED}")


def _settled(target: Path, staged: Path, *, committed: bool) -> None:
    """A body into place once its profile has committed. A body whose event rolled back, or whose
    rename failed, is this process's no longer: it stays staged for start's `settle`, which
    collects it without a profile and places it with one, and a later keep places it too."""
    try:
        if committed:
            os.replace(staged, target)
    finally:
        del _unplaced[target]


def _read(root: Path, artifact: str) -> bytes:
    """An array's body: where it is staged while its profile waits to commit, else in place."""
    target = root / "blobs" / artifact
    staged = _unplaced.get(target)
    if staged is None:
        return target.read_bytes()
    try:
        return staged.read_bytes()
    except FileNotFoundError:  # its batch committed meanwhile: placed
        return target.read_bytes()


def _place(target: Path, payload: bytes) -> None:
    staged = _staged(target)
    staged.write_bytes(payload)
    os.replace(staged, target)


def _columns_body(arrays: Mapping[str, npt.NDArray[Any]], described: dict[str, Any]) -> bytes:
    table = pa.table({name: array.reshape(-1) for name, array in arrays.items()})
    table = table.replace_schema_metadata({"shape": json.dumps(described["shape"])})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, use_dictionary=False)
    return sink.getvalue().to_pybytes()


def _packed_body(payload: bytes) -> bytes:
    table = pa.table({PACKED: pa.array([payload], type=pa.large_binary())})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, use_dictionary=False)
    return sink.getvalue().to_pybytes()


def _body(root: Path, read: Read, content_digest: str) -> tuple[dict[str, Any], Any]:
    known = profile(read, content_digest)
    if known is None:
        raise DataError(f"no array is named {content_digest}", code="data_layer.array_missing")
    if known["body_state"] != "present":
        raise DataError(
            f"the array {content_digest} was collected", code="data_layer.array_collected"
        )
    target = root / "blobs" / known["artifact_id"]
    try:
        if target in _unplaced:
            with closing(
                pq.ParquetFile(pa.BufferReader(_read(root, known["artifact_id"])))
            ) as reader:
                table = reader.read()
        else:
            with target.open("rb") as source, closing(pq.ParquetFile(source)) as reader:
                table = reader.read()
    except FileNotFoundError as error:
        raise DataError(
            f"the array {content_digest} has no body under blobs/; restore the data root",
            code="data_layer.array_missing",
        ) from error
    except (OSError, pa.ArrowException) as error:
        raise _tampered(content_digest) from error
    return known, table


def _columns(known: dict[str, Any], table: Any, content_digest: str) -> dict[str, Any]:
    if table.column_names != [name for name, _dtype in known["columns"]]:
        raise _mismatch(content_digest, "its columns")
    stored = json.loads((table.schema.metadata or {}).get(b"shape", b"null"))
    if stored != known["shape"] or table.num_rows != int(np.prod(known["shape"], dtype=np.int64)):
        raise _mismatch(content_digest, "its shape")
    columns: dict[str, npt.NDArray[Any]] = {}
    for name, dtype in known["columns"]:
        values = np.ascontiguousarray(table.column(name).to_numpy())
        if values.dtype.str != dtype:
            raise _mismatch(content_digest, f"the dtype of {name}")
        columns[name] = values
    return columns


def _named(described: Mapping[str, Any], cited: str, buffers: list[memoryview]) -> Any:
    """The artifact_id's hasher: FORM, the profile, the cited digest, then the BLAKE3 of the
    bytes. The bytes are hashed alone, from offset zero: after a prefix that is not a whole number
    of BLAKE3's 1 KiB chunks, its threads run three times slower."""
    size = sum(buffer.nbytes for buffer in buffers)
    body = integrity_blake3(threaded=size >= THREADED)
    for buffer in buffers:
        body.update(buffer)
    named = identity_blake3(FORM)
    named.update(canonical({key: described[key] for key in ("kind", "shape", "columns")}))
    named.update(bytes.fromhex(cited))
    named.update(body.digest())
    return named


def _prove(known: dict[str, Any], buffers: list[memoryview], content_digest: str) -> None:
    if _named(known, content_digest, buffers).hexdigest() != known["artifact_id"]:
        raise _tampered(content_digest)


def _mismatch(content_digest: str, what: str) -> DataError:
    return DataError(
        f"the array {content_digest}'s body differs from its profile in {what}",
        code="data_layer.array_profile_mismatch",
    )


def _tampered(content_digest: str) -> DataError:
    return DataError(
        f"the array {content_digest}'s body does not prove its digest; restore the data root",
        code="data_layer.array_tampered",
    )


__all__ = [
    "CARRIED",
    "MEANING",
    "PACKED",
    "STAGED",
    "TYPE",
    "Described",
    "carry",
    "describe",
    "describe_packed",
    "digest",
    "import_carried",
    "keep",
    "keep_file",
    "keep_packed",
    "profile",
    "read_columns",
    "read_file",
    "read_packed",
    "settle",
]
