"""Content-addressed storage for causal execution outcomes."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor
from alphalattice.control.workspace_runtime.content_store import verified_source_value
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .compile import _SCHEMA_ID, _schema
from .contracts import (
    CausalExecutionOutcomeChunk,
    CausalExecutionOutcomeManifest,
    CausalExecutionOutcomeReceipt,
    DevelopmentOnlyExecutionOutcomeManifest,
    DevelopmentOnlyExecutionOutcomeMarker,
    LocalQAOutcomePreparationFile,
    LocalQAOutcomePreparationHead,
    LocalQAOutcomePreparationMarker,
    LocalQAOutcomePreparationPart,
    LocalQAOutcomePreparationRetentionInventory,
)
from .methods import ExecutionOutcomeMethodBinding, ExecutionOutcomeMethodSealMarker

#: Storage category of the artifact method seal. It lives beside the manifest
#: rather than inside it because the manifest's bytes are frozen, and a seal is
#: a separate assertion about a snapshot rather than a field of one.
METHOD_BINDING_CATEGORY = "method-bindings"

#: Terminal authority for a method-bound snapshot. Separate from the binding
#: because a binding on disk proves only that a well-formed file exists; the
#: marker is what the publisher writes last, so its presence is the claim.
METHOD_SEAL_MARKER_CATEGORY = "method-seal-markers"

#: The development-only successor snapshot and its publication marker. Separate
#: categories rather than shared directories, so a reader of the frozen contract
#: never has to decide which of two schemas a file in "manifests" is -- and a
#: scope that carries no sealed holdout cannot be reached by a consumer looking
#: for one.
DEVELOPMENT_ONLY_MANIFEST_CATEGORY = "development-only-manifests"
DEVELOPMENT_ONLY_MARKER_CATEGORY = "development-only-markers"

LOCAL_QA_PREPARATION_CATEGORY = "local-qa-preparation"


def _local_qa_validity_buffer(array: pa.Array) -> pa.Buffer | None:
    """Rebase validity bits and zero their unused tail; omit all-valid bitmaps."""
    if not array.null_count:
        return None
    bitmap = array.buffers()[0]
    bits_per_byte = pa.uint8().bit_width
    byte_offset, bit_offset = divmod(array.offset, bits_per_byte)
    byte_count = (len(array) + bit_offset + bits_per_byte - 1) // bits_per_byte
    bits = int.from_bytes(bitmap.slice(byte_offset, byte_count).to_pybytes(), "little")
    # Match the builder's offset-zero bitmap, including zero unused tail bits.
    bits = (bits >> bit_offset) & ((1 << len(array)) - 1)
    return pa.py_buffer(bits.to_bytes((len(array) + bits_per_byte - 1) // bits_per_byte, "little"))


def _canonical_local_qa_table(table: pa.Table) -> pa.Table:
    """Restore the full row builder's typed buffers without deriving any row.

    Parquet readers can retain all-valid bitmaps or arbitrary bytes beneath
    nulls. Rebase slices, remove all-valid bitmaps, clear unused validity bits
    and fill null data with zero while preserving every valid value's bits.
    This fixed schema needs no per-cell Python value conversion.
    """
    schema = _schema()
    if not table.schema.equals(schema, check_metadata=True):
        raise ValueError("causal_outcomes.qa_preparation_part_invalid")
    arrays = []
    for field, column in zip(schema, table.columns, strict=True):
        array = column.combine_chunks()
        validity = _local_qa_validity_buffer(array)
        if pa.types.is_string(field.type):
            values = pc.fill_null(array, "") if array.null_count else array
            buffers = values.buffers()
            offsets = pa.Array.from_buffers(
                pa.int32(), len(values) + 1, [None, buffers[1]], offset=values.offset
            )
            first, stop = offsets[0].as_py(), offsets[-1].as_py()
            if first:
                offsets = pc.subtract(offsets, pa.scalar(first, type=pa.int32()))
            offset_width = pa.int32().bit_width // pa.uint8().bit_width
            value_buffers = [
                validity,
                offsets.buffers()[1].slice(
                    offsets.offset * offset_width, (len(values) + 1) * offset_width
                ),
                buffers[2].slice(first, stop - first),
            ]
        else:
            width = field.type.bit_width // pa.uint8().bit_width
            if array.null_count:
                zero = pa.Array.from_buffers(field.type, 1, [None, pa.py_buffer(bytes(width))])[0]
                values = pc.fill_null(array, zero)
            else:
                values = array
            value_buffers = [
                validity,
                values.buffers()[1].slice(values.offset * width, len(values) * width),
            ]
        arrays.append(
            pa.Array.from_buffers(
                field.type, len(array), value_buffers, null_count=array.null_count
            )
        )
    return pa.Table.from_arrays(arrays, schema=schema)


def _local_qa_ipc(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return cast(bytes, sink.getvalue().to_pybytes())


def _local_qa_json(value: LocalQAOutcomePreparationHead | LocalQAOutcomePreparationMarker) -> bytes:
    return json.dumps(
        value.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


class _ExecutionOutcomeArtifactStore:
    _PREFIX = "playpen://data-operations/execution-outcomes/"

    def __init__(
        self, artifact_root: Path, *, preparation_capacity: Callable[[int], None] | None = None
    ) -> None:
        self.root = artifact_root.resolve() / "data-operations" / "execution-outcomes"
        self._preparation_capacity = preparation_capacity

    @staticmethod
    def _require_local_qa_hash(value: str) -> None:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("causal_outcomes.qa_preparation_reference_invalid")

    def _local_qa_path(self, category: str, value: str, suffix: str) -> Path:
        self._require_local_qa_hash(value)
        return self.root / LOCAL_QA_PREPARATION_CATEGORY / category / f"{value}.{suffix}"

    def _local_qa_marker_path(self) -> Path:
        return self.root / LOCAL_QA_PREPARATION_CATEGORY / "markers" / "current.json"

    def _read_local_qa_marker(self) -> LocalQAOutcomePreparationMarker | None:
        path = self._local_qa_marker_path()
        if not path.exists():
            # Uncommitted parts/heads can survive a process death. Only the
            # last-written marker grants durable preparation reuse.
            return None
        try:
            content = path.read_bytes()
            marker = cast(
                LocalQAOutcomePreparationMarker,
                LocalQAOutcomePreparationMarker.model_validate_json(content),
            )
            if content != _local_qa_json(marker):
                raise ValueError("noncanonical marker bytes")
            return marker
        except (OSError, ValueError) as exc:
            raise ValueError("causal_outcomes.qa_preparation_marker_invalid") from exc

    def _read_local_qa_head(self, head_hash: str) -> LocalQAOutcomePreparationHead:
        path = self._local_qa_path("heads", head_hash, "json")
        try:
            content = path.read_bytes()
        except FileNotFoundError as exc:
            raise ValueError("causal_outcomes.qa_preparation_missing") from exc
        except OSError as exc:
            raise ValueError("causal_outcomes.qa_preparation_head_invalid") from exc
        try:
            head = cast(
                LocalQAOutcomePreparationHead,
                LocalQAOutcomePreparationHead.model_validate_json(content),
            )
            if head.content_hash != head_hash or content != _local_qa_json(head):
                raise ValueError("head bytes disagree")
            return head
        except ValueError as exc:
            raise ValueError("causal_outcomes.qa_preparation_head_invalid") from exc

    def _read_local_qa_table(self, head: LocalQAOutcomePreparationHead) -> bytes:
        tables: list[tuple[pa.Table, int]] = []
        for part in head.parts:
            path = self._local_qa_path("parts", part.file_hash, "parquet")
            try:
                content = path.read_bytes()
            except FileNotFoundError as exc:
                raise ValueError("causal_outcomes.qa_preparation_missing") from exc
            except OSError as exc:
                raise ValueError("causal_outcomes.qa_preparation_part_invalid") from exc
            if (
                len(content) != part.byte_count
                or hashlib.sha256(content).hexdigest() != part.file_hash
            ):
                raise ValueError("causal_outcomes.qa_preparation_part_invalid")
            try:
                table = _canonical_local_qa_table(pq.read_table(pa.BufferReader(content)))
            except (ValueError, pa.ArrowException) as exc:
                raise ValueError("causal_outcomes.qa_preparation_part_invalid") from exc
            if table.num_rows != part.row_count:
                raise ValueError("causal_outcomes.qa_preparation_part_invalid")
            if hashlib.sha256(_local_qa_ipc(table)).hexdigest() != part.ipc_hash:
                raise ValueError("causal_outcomes.qa_preparation_ipc_mismatch")
            tables.append((table, part.point_count))
        table = _canonical_local_qa_table(
            pa.concat_tables(
                [
                    table.slice(index * count, count)
                    for index in range(head.listing_count)
                    for table, count in tables
                ]
            ).combine_chunks()
        )
        content = _local_qa_ipc(table)
        if (
            table.num_rows != head.listing_count * head.matured_point_count
            or hashlib.sha256(content).hexdigest() != head.ipc_hash
        ):
            raise ValueError("causal_outcomes.qa_preparation_ipc_mismatch")
        return content

    def load_local_qa_preparation(self) -> tuple[LocalQAOutcomePreparationHead, pa.Table] | None:
        """Read the sealed current QA head; missing or corrupt dependencies refuse."""
        marker = self._read_local_qa_marker()
        if marker is None:
            return None
        head = self._read_local_qa_head(marker.current_head_hash)
        paths = [
            self._local_qa_marker_path(),
            self._local_qa_path("heads", head.content_hash, "json"),
        ]
        commitments = [
            hashlib.sha256(_local_qa_json(marker)).hexdigest(),
            hashlib.sha256(_local_qa_json(head)).hexdigest(),
        ]
        for part in head.parts:
            paths.append(self._local_qa_path("parts", part.file_hash, "parquet"))
            commitments.append(part.file_hash)

        def verified() -> bytes:
            # Verify the small records again after leases were acquired: parsed
            # heads may have been replaced between discovery and admission.
            for path, expected in zip(paths, commitments, strict=True):
                if path.suffix == ".parquet":
                    # _read_local_qa_table verifies these exact bytes once.
                    continue
                try:
                    with path.open("rb") as stream:
                        actual = hashlib.file_digest(stream, "sha256").hexdigest()
                except FileNotFoundError as exc:
                    raise ValueError("causal_outcomes.qa_preparation_missing") from exc
                if actual != expected:
                    raise ValueError("causal_outcomes.qa_preparation_part_invalid")
            return self._read_local_qa_table(head)

        content = verified_source_value(
            identity=("local-qa-outcome-preparation", head.content_hash, *commitments),
            sources=tuple(paths),
            builder=verified,
            nbytes=len,
        )
        return head, pa.ipc.open_stream(pa.py_buffer(content)).read_all()

    def _write_local_qa(self, path: Path, content: bytes, *, replace: bool = False) -> None:
        if path.exists():
            existing = path.read_bytes()
            if existing == content:
                return
            if not replace:
                raise ValueError("causal_outcomes.qa_preparation_part_invalid")
        if self._preparation_capacity is None:
            raise ValueError("causal_outcomes.qa_preparation_capacity_required")
        self._preparation_capacity(len(content))
        self._atomic_write(path, content)

    def publish_local_qa_preparation(
        self,
        *,
        table: pa.Table,
        increment: pa.Table | None,
        previous: LocalQAOutcomePreparationHead | None,
        values: Mapping[str, object],
    ) -> LocalQAOutcomePreparationHead:
        """Write only a base or new rows, their immutable head, and the marker last."""
        if self._preparation_capacity is None:
            raise ValueError("causal_outcomes.qa_preparation_capacity_required")
        ipc_hash = hashlib.sha256(_local_qa_ipc(table)).hexdigest()
        if (
            previous is not None
            and ipc_hash == previous.ipc_hash
            and all(getattr(previous, name) == value for name, value in values.items())
        ):
            return previous
        selected = table if previous is None else increment
        part = None
        if selected is not None:
            selected = _canonical_local_qa_table(selected)
            sink = pa.BufferOutputStream()
            pq.write_table(selected, sink, compression="zstd")
            content = sink.getvalue().to_pybytes()
            file_hash = hashlib.sha256(content).hexdigest()
            part = LocalQAOutcomePreparationPart.create(
                role="BASE" if previous is None else "INCREMENT",
                listing_count=values["listing_count"],
                point_count=selected.num_rows // cast(int, values["listing_count"]),
                row_count=selected.num_rows,
                ipc_hash=hashlib.sha256(_local_qa_ipc(selected)).hexdigest(),
                file_hash=file_hash,
                byte_count=len(content),
            )
            self._write_local_qa(self._local_qa_path("parts", file_hash, "parquet"), content)
        head = LocalQAOutcomePreparationHead.create(
            **dict(values),
            previous_head_hash=None if previous is None else previous.content_hash,
            parts=(() if previous is None else previous.parts) + (() if part is None else (part,)),
            part=part,
            ipc_hash=ipc_hash,
        )
        self._write_local_qa(
            self._local_qa_path("heads", head.content_hash, "json"), _local_qa_json(head)
        )
        old = self._read_local_qa_marker()
        if (
            old is not None
            and self._read_local_qa_head(old.current_head_hash).through > head.through
        ):
            # A historical request can publish its exact immutable head, but
            # cannot replace the latest current/rollback retention roots.
            return head
        marker = LocalQAOutcomePreparationMarker.create(
            current_head_hash=head.content_hash,
            previous_head_hash=None if old is None else old.current_head_hash,
        )
        self._write_local_qa(self._local_qa_marker_path(), _local_qa_json(marker), replace=True)
        return head

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        staged.write_bytes(content)
        os.replace(staged, path)

    def _uri(self, category: str, value: str) -> str:
        return f"{self._PREFIX}{category}/{value}"

    def _json_path(self, category: str, value: str) -> Path:
        return self.root / category / f"{value}.json"

    def _parquet_path(self, split: str, value: str) -> Path:
        return self.root / split.casefold().replace("_", "-") / "chunks" / f"{value}.parquet"

    def publish_json(
        self, *, category: str, payload: Mapping[str, object], identity_field: str
    ) -> ArtifactDescriptor:
        value = str(payload[identity_field])
        content = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        path = self._json_path(category, value)
        if path.exists() and path.read_bytes() != content:
            raise ValueError("causal execution identity was reused with different bytes")
        if not path.exists():
            self._atomic_write(path, content)
        return ArtifactDescriptor(
            kind=f"causal-execution-{category}",
            content_hash=value,
            metadata_hash=hashlib.sha256(content).hexdigest(),
            uri=self._uri(category, value),
        )

    def load_json(self, *, category: str, uri: str) -> dict[str, object]:
        prefix = f"{self._PREFIX}{category}/"
        if not uri.startswith(prefix):
            raise ValueError("causal execution URI is invalid")
        value = uri[len(prefix) :]
        payload = json.loads(self._json_path(category, value).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("causal execution JSON is invalid")
        return payload

    def publish_chunk(self, *, split: str, table: pa.Table) -> CausalExecutionOutcomeChunk:
        ordered = table.combine_chunks().sort_by(
            [("formation_session", "ascending"), ("listing_id", "ascending")]
        )
        # The identity is the canonical JSON of (listing, formation, row hash)
        # per row; a formation session is encoded as ``str(value)`` by the
        # canonical encoder, so the columns are read once each and the
        # session is given as that string -- the same bytes, without a dict
        # per row.
        content_hash = canonical_hash(
            {
                "schema": _SCHEMA_ID,
                "rows": tuple(
                    zip(
                        ordered.column("listing_id").to_pylist(),
                        (str(value) for value in ordered.column("formation_session").to_pylist()),
                        ordered.column("row_hash").to_pylist(),
                        strict=True,
                    )
                ),
            }
        )
        path = self._parquet_path(split, content_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            b"alphalattice.snapshot_kind": b"CausalExecutionOutcomeChunk",
            b"alphalattice.chunk_hash": content_hash.encode(),
            b"alphalattice.split": split.encode(),
        }
        if not path.exists():
            staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            pq.write_table(ordered.replace_schema_metadata(metadata), staged, compression="zstd")
            os.replace(staged, path)
        parquet = pq.ParquetFile(path)
        actual_metadata = parquet.schema_arrow.metadata or {}
        if actual_metadata.get(b"alphalattice.chunk_hash") != content_hash.encode():
            raise ValueError("causal execution chunk metadata is invalid")
        metadata_hash = hashlib.sha256(
            json.dumps(
                {key.decode(): value.decode() for key, value in sorted(actual_metadata.items())},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        dates = ordered.column("formation_session").to_pylist()
        year = dates[0].year
        return CausalExecutionOutcomeChunk(
            split=split,
            year=year,
            row_count=ordered.num_rows,
            first_formation_session=min(dates),
            last_formation_session=max(dates),
            content_hash=content_hash,
            metadata_hash=metadata_hash,
            uri=self._uri(f"{split.casefold().replace('_', '-')}/chunks", content_hash),
        )

    def resolve_chunk(self, chunk: CausalExecutionOutcomeChunk) -> Path:
        path = self._parquet_path(chunk.split, chunk.content_hash)
        metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
        if metadata.get(b"alphalattice.chunk_hash") != chunk.content_hash.encode():
            raise ValueError("causal execution chunk readback is invalid")
        return path

    def manifests(self) -> tuple[CausalExecutionOutcomeManifest, ...]:
        root = self.root / "manifests"
        return (
            tuple(
                CausalExecutionOutcomeManifest.model_validate_json(path.read_text(encoding="utf-8"))
                for path in sorted(root.glob("*.json"))
            )
            if root.is_dir()
            else ()
        )

    def receipts(self) -> tuple[CausalExecutionOutcomeReceipt, ...]:
        root = self.root / "receipts"
        return (
            tuple(
                CausalExecutionOutcomeReceipt.model_validate_json(path.read_text(encoding="utf-8"))
                for path in sorted(root.glob("*.json"))
            )
            if root.is_dir()
            else ()
        )

    def development_only_manifests(self) -> tuple[DevelopmentOnlyExecutionOutcomeManifest, ...]:
        """Every development-only snapshot on disk, validated as it is read."""
        root = self.root / DEVELOPMENT_ONLY_MANIFEST_CATEGORY
        return (
            tuple(
                DevelopmentOnlyExecutionOutcomeManifest.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                for path in sorted(root.glob("*.json"))
            )
            if root.is_dir()
            else ()
        )

    def development_only_markers(self) -> tuple[DevelopmentOnlyExecutionOutcomeMarker, ...]:
        """Every development-only publication marker on disk, validated as it is read."""
        root = self.root / DEVELOPMENT_ONLY_MARKER_CATEGORY
        return (
            tuple(
                DevelopmentOnlyExecutionOutcomeMarker.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                for path in sorted(root.glob("*.json"))
            )
            if root.is_dir()
            else ()
        )

    def method_bindings(self) -> tuple[ExecutionOutcomeMethodBinding, ...]:
        """Every artifact method seal on disk, validated as it is read.

        Absence is normal and means legacy: a snapshot published before the
        method seam existed has no seal, and this returning nothing for it is
        what keeps it legacy rather than letting a reader mint one.
        """
        root = self.root / METHOD_BINDING_CATEGORY
        return (
            tuple(
                ExecutionOutcomeMethodBinding.model_validate_json(path.read_text(encoding="utf-8"))
                for path in sorted(root.glob("*.json"))
            )
            if root.is_dir()
            else ()
        )

    def method_seal_markers(self) -> tuple[ExecutionOutcomeMethodSealMarker, ...]:
        """Every terminal method seal on disk, validated as it is read."""
        root = self.root / METHOD_SEAL_MARKER_CATEGORY
        return (
            tuple(
                ExecutionOutcomeMethodSealMarker.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                for path in sorted(root.glob("*.json"))
            )
            if root.is_dir()
            else ()
        )


def local_qa_preparation_retention_inventory(
    artifact_root: Path, *, referenced_heads: tuple[str, ...] = ()
) -> LocalQAOutcomePreparationRetentionInventory:
    """Inventory only QA preparation bytes, protecting current, previous and explicit heads.

    Args:
        artifact_root: The workspace's existing artifacts root.
        referenced_heads: Exact head hashes protected by external pins or recovery roots.

    Returns:
        Sealed physical file commitments for the existing retention owner to admit.

    Raises:
        ValueError: A root reference or one of its committed physical artifacts is invalid.
    """
    store = _ExecutionOutcomeArtifactStore(artifact_root)
    marker = store._read_local_qa_marker()
    explicit = tuple(sorted(set(referenced_heads)))
    for value in explicit:
        store._require_local_qa_hash(value)
    roots = set(explicit)
    protected: dict[Path, str] = {}
    if marker is not None:
        protected[store._local_qa_marker_path()] = hashlib.sha256(
            _local_qa_json(marker)
        ).hexdigest()
        roots.add(marker.current_head_hash)
        if marker.previous_head_hash is not None:
            roots.add(marker.previous_head_hash)
    for root in sorted(roots):
        head = store._read_local_qa_head(root)
        protected[store._local_qa_path("heads", head.content_hash, "json")] = hashlib.sha256(
            _local_qa_json(head)
        ).hexdigest()
        for part in head.parts:
            protected[store._local_qa_path("parts", part.file_hash, "parquet")] = part.file_hash
    scope = store.root / LOCAL_QA_PREPARATION_CATEGORY
    files = set(path for path in scope.rglob("*") if path.is_file()) if scope.exists() else set()
    if not set(protected) <= files:
        raise ValueError("causal_outcomes.qa_preparation_missing")
    rooted: list[LocalQAOutcomePreparationFile] = []
    candidates: list[LocalQAOutcomePreparationFile] = []
    for path in sorted(files):
        if not path.resolve().is_relative_to(scope.resolve()):
            raise ValueError("causal_outcomes.qa_preparation_reference_invalid")
        content = path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        item = LocalQAOutcomePreparationFile(
            relative_path=path.relative_to(artifact_root.resolve()).as_posix(),
            byte_count=len(content),
            file_hash=file_hash,
        )
        if path in protected:
            if file_hash != protected[path]:
                raise ValueError("causal_outcomes.qa_preparation_part_invalid")
            rooted.append(item)
        else:
            candidates.append(item)
    return LocalQAOutcomePreparationRetentionInventory.create(
        current_head_hash=None if marker is None else marker.current_head_hash,
        previous_head_hash=None if marker is None else marker.previous_head_hash,
        referenced_heads=explicit,
        protected_files=tuple(rooted),
        candidate_files=tuple(candidates),
    )


__all__ = [
    "DEVELOPMENT_ONLY_MANIFEST_CATEGORY",
    "DEVELOPMENT_ONLY_MARKER_CATEGORY",
    "LOCAL_QA_PREPARATION_CATEGORY",
    "METHOD_BINDING_CATEGORY",
    "METHOD_SEAL_MARKER_CATEGORY",
    "_ExecutionOutcomeArtifactStore",
    "local_qa_preparation_retention_inventory",
]
