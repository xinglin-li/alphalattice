"""Trusted, content-addressed artifact publication for one local workspace."""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from alphalattice.capabilities.domain_contracts.bounded_memo import stage_digest
from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.capabilities.domain_contracts.identity_successors import latest
from alphalattice.capabilities.domain_contracts.sealing import scoped_payload_digest
from alphalattice.control.workspace_runtime.paths import (
    resolve_confined,
    resolve_workspace_root,
)
from alphalattice.control.workspace_runtime.verified_facts import (
    record_sealed_file,
    verified_file_digest,
)
from alphalattice.kernel.identity import (
    blake3_filename,
    blake3_from_filename,
    identity_blake3,
    integrity_blake3,
    is_blake3,
    tagged_blake3,
)


def _metadata_hash(source: Path | Mapping[bytes, bytes]) -> str:
    """The hash of a chunk's recorded metadata: read from the file `source`, or already read."""
    metadata = (
        source
        if isinstance(source, Mapping)
        else pq.ParquetFile(source).schema_arrow.metadata or {}
    )
    normalized = {
        key.decode("utf-8"): value.decode("utf-8") for key, value in sorted(metadata.items())
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return tagged_blake3(integrity_blake3(payload.encode("utf-8")))


def _matches_binding_digest(payload: Mapping[str, object], field: str, expected: object) -> bool:
    identity = dict(payload)
    return (
        isinstance(expected, str)
        and identity.pop(field, None) == expected
        and not _breaks_seal(identity, field, expected)
    )


def _breaks_seal(identity: dict[str, object], field: str, expected: str) -> bool:
    """Whether `identity` does not hash to `expected`, checked once per stage for its content: its
    writer's check and its readers' share one hash."""
    return scoped_payload_digest(identity, field, expected, digest=canonical_hash) != expected


ADJUSTED_RETURN_CHAIN = "adjusted-return-chain"
"""The stage slot of an adjusted-return revision chain's identity: its builder, its publisher and
its readers hash the same deltas once per stage."""


def _files(folder: Path, prefix: str = "") -> list[str]:
    """Every regular file under `folder`, relative to it, in name order. The folder itself is
    proved first, a real directory and no link or reparse point; an unreadable folder raises,
    never skipped; a link-like or hard-linked entry, or one neither a regular file nor a
    folder, is refused through the workspace's path confinement."""
    if not prefix:
        resolve_workspace_root(folder)
    found: list[str] = []
    for name in sorted(entry.name for entry in (folder / prefix).iterdir()):
        relative = f"{prefix}{name}"
        mode = resolve_confined(folder, relative).lstat().st_mode
        if stat.S_ISDIR(mode):
            found.extend(_files(folder, f"{relative}/"))
        elif stat.S_ISREG(mode):
            found.append(relative)
        else:
            raise ValueError(f"a published entry is not a regular file: {relative}")
    return found


@dataclass(frozen=True)
class ArtifactDescriptor:
    """A path-free artifact identity which may cross a task boundary."""

    kind: str
    content_hash: str
    metadata_hash: str
    uri: str


@dataclass(frozen=True)
class ArtifactReachabilityReport:
    """Read-only artifact inventory.  It never authorizes deletion."""

    reachable: tuple[str, ...]
    unreferenced: tuple[str, ...]
    incomplete_publication: tuple[str, ...]
    unknown_owner: tuple[str, ...]


class ArtifactResolver:
    """Resolve only known content-addressed artifact kinds inside trusted code."""

    _FEATURE_INPUT_KIND = "feature-input"
    _FEATURE_PANEL_ROOT = "feature-panel"
    _FEATURE_PANEL_LIFECYCLE = "snapshot-lifecycle.json"
    _DATA_OPERATIONS_ROOT = "data-operations"
    _ADJUSTED_RETURN_REVISION_POINTER = "adjusted-return-semantic-revision-current.json"
    _PLUGIN_RESULT_KIND = "plugin-result"

    def __init__(self, root: Path) -> None:
        """Bind trusted artifact resolution to one resolved store root.

        Args:
            root: Workspace or store root used by this owner.
        """
        self._root = root.resolve()

    @property
    def root(self) -> Path:
        """Return the resolved physical root for trusted infrastructure.

        Returns:
            Resolved artifact-store root.
        """
        return self._root

    @staticmethod
    def feature_input_uri(content_hash: str) -> str:
        """Construct a path-free feature input address.

        Args:
            content_hash: Exact content identity used in the artifact address.

        Returns:
            Feature input URI; address construction does not validate the stored artifact.
        """
        return f"playpen://feature-input/{content_hash}"

    def _feature_input_path(self, content_hash: str) -> Path:
        return self._root / self._FEATURE_INPUT_KIND / f"{blake3_filename(content_hash)}.parquet"

    @staticmethod
    def feature_panel_chunk_uri(content_hash: str) -> str:
        """Construct a path-free feature panel chunk address.

        Args:
            content_hash: Exact content identity used in the artifact address.

        Returns:
            Feature panel chunk URI; address construction does not validate the stored artifact.
        """
        return f"playpen://feature-panel/chunks/{content_hash}"

    @staticmethod
    def feature_panel_manifest_uri(content_hash: str) -> str:
        """Construct a path-free feature panel manifest address.

        Args:
            content_hash: Exact content identity used in the artifact address.

        Returns:
            Feature panel manifest URI; address construction does not validate the stored artifact.
        """
        return f"playpen://feature-panel/manifests/{content_hash}"

    def _feature_panel_chunk_path(self, content_hash: str) -> Path:
        self._require_hash(content_hash)
        return (
            self._root
            / self._FEATURE_PANEL_ROOT
            / "chunks"
            / f"{blake3_filename(content_hash)}.parquet"
        )

    def _feature_panel_manifest_path(self, content_hash: str) -> Path:
        self._require_hash(content_hash)
        return (
            self._root
            / self._FEATURE_PANEL_ROOT
            / "manifests"
            / f"{blake3_filename(content_hash)}.json"
        )

    @staticmethod
    def feature_panel_semantic_index_uri(content_hash: str) -> str:
        """Construct a path-free feature panel semantic index address.

        Args:
            content_hash: Exact content identity used in the artifact address.

        Returns:
            Feature panel semantic index URI; address construction does not validate the stored
            artifact.
        """
        return f"playpen://feature-panel/semantic-index/{content_hash}"

    def _feature_panel_semantic_index_path(self, content_hash: str) -> Path:
        self._require_hash(content_hash)
        return (
            self._root
            / self._FEATURE_PANEL_ROOT
            / "semantic-index"
            / f"{blake3_filename(content_hash)}.json"
        )

    def _feature_panel_lifecycle_path(self) -> Path:
        return self._root / self._FEATURE_PANEL_ROOT / self._FEATURE_PANEL_LIFECYCLE

    def _adjusted_return_revision_path(self, chain_hash: str) -> Path:
        self._require_hash(chain_hash)
        return (
            self._root
            / self._DATA_OPERATIONS_ROOT
            / "adjusted-return-semantic-revisions"
            / f"{blake3_filename(chain_hash)}.json"
        )

    def _adjusted_return_revision_pointer_path(self) -> Path:
        return self._root / self._DATA_OPERATIONS_ROOT / self._ADJUSTED_RETURN_REVISION_POINTER

    @staticmethod
    def adjusted_return_semantic_revision_uri(chain_hash: str) -> str:
        """Construct a path-free adjusted-return semantic revision address.

        Args:
            chain_hash: Canonical identity of the ordered semantic-delta hash chain.

        Returns:
            Adjusted-return semantic revision URI; address construction does not validate the stored
            artifact.
        """
        return f"playpen://data-operations/adjusted-return-semantic-revisions/{chain_hash}"

    def publish_feature_input(self, staged_path: Path, content_hash: str) -> ArtifactDescriptor:
        """Verify a staged Parquet and atomically publish it under its content hash."""
        staged = staged_path.resolve()
        metadata = pq.ParquetFile(staged).schema_arrow.metadata or {}
        if metadata.get(b"alphalattice.snapshot_hash", b"").decode("utf-8") != content_hash:
            raise ValueError("staged feature snapshot hash does not match its metadata")
        target = self._feature_input_path(content_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            self._validate_feature_input(target, content_hash)
            staged.unlink(missing_ok=True)
        else:
            os.replace(staged, target)
            self._validate_feature_input(target, content_hash)
        return ArtifactDescriptor(
            kind=self._FEATURE_INPUT_KIND,
            content_hash=content_hash,
            metadata_hash=_metadata_hash(target),
            uri=self.feature_input_uri(content_hash),
        )

    def _plugin_result_path(self, idempotency_key: str) -> Path:
        return self._root / self._PLUGIN_RESULT_KIND / blake3_filename(idempotency_key)

    def publish_plugin_result(
        self,
        staged_dir: Path,
        idempotency_key: str,
        *,
        publication: AbstractContextManager[None] | None = None,
    ) -> ArtifactDescriptor:
        """Atomically publish a plugin run's checked staging folder under its run's key.

        Every staged file is checked and hashed before the key is exposed, so a failure leaves
        it unpublished; each file placed is then sealed in the trust record, and a target whose
        sealing fails is cleared again. A key already published keeps its first result, which
        the same inputs derive, and the staged folder is left to its owner. The descriptor's
        metadata hash names every published file's relative path and digest.
        """
        self._require_hash(idempotency_key)
        target = self._plugin_result_path(idempotency_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = target if target.exists() else staged_dir
        files = {relative: verified_file_digest(source / relative) for relative in _files(source)}
        with publication if publication is not None else nullcontext():
            if source is staged_dir:
                os.replace(staged_dir, target)
        if source is staged_dir:
            try:
                for relative, digest in files.items():
                    record_sealed_file(target / relative, digest)
            except BaseException:
                shutil.rmtree(target)
                raise
        return ArtifactDescriptor(
            kind=self._PLUGIN_RESULT_KIND,
            content_hash=idempotency_key,
            metadata_hash=_metadata_hash(
                {relative.encode(): digest.encode() for relative, digest in files.items()}
            ),
            uri=f"playpen://{self._PLUGIN_RESULT_KIND}/{idempotency_key}",
        )

    def resolve_plugin_result(self, descriptor: ArtifactDescriptor) -> Path:
        """Return a published plugin result's folder only to trusted Host code."""
        if descriptor.kind != self._PLUGIN_RESULT_KIND:
            raise ValueError("unsupported artifact kind")
        return self._plugin_result_path(descriptor.content_hash)

    def resolve_feature_input(self, descriptor: ArtifactDescriptor) -> Path:
        """Return a physical path only to trusted Data Operations code."""
        if descriptor.kind != self._FEATURE_INPUT_KIND:
            raise ValueError("unsupported artifact kind")
        if descriptor.uri != self.feature_input_uri(descriptor.content_hash):
            raise ValueError("feature-input URI does not match its content hash")
        path = self._feature_input_path(descriptor.content_hash)
        self._validate_feature_input(path, descriptor.content_hash)
        if _metadata_hash(path) != descriptor.metadata_hash:
            raise ValueError("feature-input metadata hash does not match descriptor")
        return path

    def publish_feature_panel_chunk(
        self,
        *,
        table: pa.Table,
        content_hash: str,
        metadata: Mapping[str, str],
    ) -> ArtifactDescriptor:
        """Atomically publish one semantic, year-bounded panel chunk."""
        self._require_hash(content_hash)
        target = self._feature_panel_chunk_path(content_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        merged = dict(table.schema.metadata or {})
        merged.update(
            {
                b"alphalattice.snapshot_kind": b"FeaturePanelChunk",
                b"alphalattice.chunk_hash": content_hash.encode("utf-8"),
                **{
                    f"alphalattice.{key}".encode(): value.encode("utf-8")
                    for key, value in metadata.items()
                },
            }
        )
        staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        if target.exists():
            self._validate_feature_panel_chunk(target, content_hash)
        else:
            pq.write_table(table.replace_schema_metadata(merged), staged, compression="zstd")
            os.replace(staged, target)
            self._validate_feature_panel_chunk(target, content_hash)
            verified_file_digest(target, written=True)
        staged.unlink(missing_ok=True)
        return ArtifactDescriptor(
            kind="feature-panel-chunk",
            content_hash=content_hash,
            metadata_hash=_metadata_hash(target),
            uri=self.feature_panel_chunk_uri(content_hash),
        )

    def publish_feature_panel_manifest(
        self, *, payload: Mapping[str, object], snapshot_hash: str
    ) -> ArtifactDescriptor:
        """Publish a path-free JSON manifest after validating its self identity."""
        self._require_hash(snapshot_hash)
        if payload.get("snapshot_hash") != snapshot_hash:
            raise ValueError("feature panel manifest hash does not match its payload")
        target = self._feature_panel_manifest_path(snapshot_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        if target.exists():
            existing = self._load_feature_panel_manifest(target, snapshot_hash)
            if existing != json.loads(serialized):
                raise ValueError("feature panel manifest identity was reused with new content")
        else:
            staged.write_bytes(serialized)
            os.replace(staged, target)
            self._load_feature_panel_manifest(target, snapshot_hash)
        staged.unlink(missing_ok=True)
        return ArtifactDescriptor(
            kind="feature-panel-manifest",
            content_hash=snapshot_hash,
            metadata_hash=tagged_blake3(integrity_blake3(serialized)),
            uri=self.feature_panel_manifest_uri(snapshot_hash),
        )

    def publish_feature_panel_semantic_index(
        self, *, payload: Mapping[str, object], index_hash: str
    ) -> ArtifactDescriptor:
        """Publish the row-hash-only preflight index without changing panel bytes."""
        self._require_hash(index_hash)
        identity = dict(payload)
        if identity.pop("index_hash", None) != index_hash or _breaks_seal(
            identity, "index_hash", index_hash
        ):
            raise ValueError("feature panel semantic index identity is invalid")
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        target = self._feature_panel_semantic_index_path(index_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        if target.exists():
            if target.read_bytes() != serialized:
                raise ValueError("feature panel semantic index identity was reused")
        else:
            staged.write_bytes(serialized)
            os.replace(staged, target)
        staged.unlink(missing_ok=True)
        return ArtifactDescriptor(
            kind="feature-panel-semantic-index",
            content_hash=index_hash,
            metadata_hash=tagged_blake3(integrity_blake3(serialized)),
            uri=self.feature_panel_semantic_index_uri(index_hash),
        )

    @staticmethod
    def panel_preprocessing_binding_uri(binding_hash: str) -> str:
        """Construct a path-free panel preprocessing binding address.

        Args:
            binding_hash: Exact preprocessing method-binding identity.

        Returns:
            Panel preprocessing binding URI; address construction does not validate the stored
            artifact.
        """
        return f"playpen://feature-panel/preprocessing-bindings/{binding_hash}"

    def _panel_preprocessing_binding_path(self, binding_hash: str) -> Path:
        self._require_hash(binding_hash)
        return (
            self.root
            / self._FEATURE_PANEL_ROOT
            / "preprocessing-bindings"
            / f"{blake3_filename(binding_hash)}.json"
        )

    def publish_panel_preprocessing_binding(
        self, *, payload: Mapping[str, object], binding_hash: str
    ) -> ArtifactDescriptor:
        """Publish the method binding the clipping receipt and seal marker both name.

        Without this the binding is a hash two documents quote at each other. Both
        the receipt and the marker carry ``preprocessing_binding_hash``, but they
        copy it from the same in-memory object, so agreeing about it proves only
        that one writer ran twice. Storing the binding is what lets a reader
        re-derive that hash from the method's own fields and find out whether the
        graph the marker describes exists.
        """
        self._require_hash(binding_hash)
        if not _matches_binding_digest(payload, "binding_hash", binding_hash):
            raise ValueError("panel preprocessing binding identity is invalid")
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        target = self._panel_preprocessing_binding_path(binding_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        if target.exists():
            # Content-addressed, so identical bytes are the only lawful rewrite and a
            # true collision is the only way to reach the raise.
            if target.read_bytes() != serialized:  # pragma: no cover - hash collision only
                raise ValueError("panel preprocessing binding identity was reused")
        else:
            staged.write_bytes(serialized)
            os.replace(staged, target)
        staged.unlink(missing_ok=True)
        return ArtifactDescriptor(
            kind="panel-preprocessing-binding",
            content_hash=binding_hash,
            metadata_hash=tagged_blake3(integrity_blake3(serialized)),
            uri=self.panel_preprocessing_binding_uri(binding_hash),
        )

    def load_panel_preprocessing_binding(self, binding_hash: str) -> dict[str, object] | None:
        """Load one method binding by its exact hash.

        ``None`` means the marker named a binding this workspace does not hold.
        That is a broken graph rather than legacy readback -- a pre-seam Panel has
        no marker at all -- so the caller must refuse it, not treat it as absence.
        """
        target = self._panel_preprocessing_binding_path(binding_hash)
        if not target.is_file():
            return None
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("panel preprocessing binding payload is invalid")
        if not _matches_binding_digest(payload, "binding_hash", binding_hash):
            raise ValueError("panel preprocessing binding identity is invalid")
        return payload

    def panel_clipping_evidence_uri(self, evidence_hash: str) -> str:
        """Construct a path-free panel clipping evidence address.

        Args:
            evidence_hash: Exact clipping-evidence identity.

        Returns:
            Panel clipping evidence URI; address construction does not validate the stored artifact.
        """
        return f"playpen://feature-panel/clipping-evidence/{evidence_hash}"

    def _panel_clipping_evidence_path(self, evidence_hash: str) -> Path:
        return (
            self.root
            / self._FEATURE_PANEL_ROOT
            / "clipping-evidence"
            / f"{blake3_filename(evidence_hash)}.json"
        )

    def publish_panel_clipping_evidence(
        self, *, payload: Mapping[str, object], evidence_hash: str
    ) -> ArtifactDescriptor:
        """Publish what one Panel's preprocessing actually changed.

        Lives in the existing Feature artifact store rather than a second one:
        clipping evidence is a fact about a Panel, and a separate store for it
        would be a second place to look for the same Panel's truth.
        """
        self._require_hash(evidence_hash)
        identity = dict(payload)
        if (
            identity.pop("evidence_hash", None) != evidence_hash
            or scoped_payload_digest(
                identity, "evidence_hash", evidence_hash, digest=canonical_hash
            )
            != evidence_hash
        ):
            raise ValueError("panel clipping evidence identity is invalid")
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        target = self._panel_clipping_evidence_path(evidence_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        if target.exists():
            if target.read_bytes() != serialized:
                raise ValueError("panel clipping evidence identity was reused")
        else:
            staged.write_bytes(serialized)
            os.replace(staged, target)
        staged.unlink(missing_ok=True)
        return ArtifactDescriptor(
            kind="panel-clipping-evidence",
            content_hash=evidence_hash,
            metadata_hash=tagged_blake3(integrity_blake3(serialized)),
            uri=self.panel_clipping_evidence_uri(evidence_hash),
        )

    def load_panel_clipping_evidence(self, uri: str) -> dict[str, object]:
        """Read clipping evidence and verify its URI-bound canonical identity.

        Args:
            uri: Artifact URI in the method's declared category.

        Returns:
            Verified clipping-evidence mapping.

        Raises:
            ValueError: The URI category, mapping, cursor or content identity is invalid.
            FileNotFoundError: The addressed artifact is missing.
        """
        prefix = "playpen://feature-panel/clipping-evidence/"
        if not uri.startswith(prefix):
            raise ValueError("unsupported panel clipping evidence URI")
        evidence_hash = uri[len(prefix) :]
        target = self._panel_clipping_evidence_path(evidence_hash)
        if not target.is_file():
            raise FileNotFoundError("panel clipping evidence is missing")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("panel clipping evidence payload is invalid")
        identity = dict(payload)
        if (
            identity.pop("evidence_hash", None) != evidence_hash
            or scoped_payload_digest(
                identity, "evidence_hash", evidence_hash, digest=canonical_hash
            )
            != evidence_hash
        ):
            raise ValueError("panel clipping evidence identity is invalid")
        return payload

    def panel_clip_observation_uri(self, receipt_hash: str) -> str:
        """Construct a path-free panel clip observation address.

        Args:
            receipt_hash: Exact materialization-batch receipt identity.

        Returns:
            Panel clip observation URI; address construction does not validate the stored artifact.
        """
        return f"playpen://feature-panel/clip-observations/{receipt_hash}"

    def _panel_clip_observation_path(self, receipt_hash: str) -> Path:
        self._require_hash(receipt_hash)
        return (
            self.root
            / self._FEATURE_PANEL_ROOT
            / "clip-observations"
            / f"{blake3_filename(receipt_hash)}.json"
        )

    def publish_panel_clip_observation(
        self, *, payload: Mapping[str, object], receipt_hash: str
    ) -> ArtifactDescriptor:
        """Persist one materialization batch's clipping facts under its receipt.

        Keyed by the batch receipt rather than by its own content hash because
        the receipt is the provenance key the Panel's rows and availability
        already carry; the evidence of a Panel that reuses partitions is folded
        from these records by that key. The document still seals its own
        ``record_hash``, so a tampered file is refused on read.
        """
        identity = dict(payload)
        record_hash = identity.pop("record_hash", None)
        if (
            payload.get("receipt_hash") != receipt_hash
            or not isinstance(record_hash, str)
            or scoped_payload_digest(identity, "record_hash", record_hash, digest=canonical_hash)
            != record_hash
        ):
            raise ValueError("panel clip observation identity is invalid")
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        target = self._panel_clip_observation_path(receipt_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        if target.exists():
            # The receipt hashes the binding, the availability and the
            # admission of the batch, and the observations are a deterministic
            # function of the same arrays; a different document under the same
            # receipt is a contradiction, not a rewrite.
            if target.read_bytes() != serialized:
                raise ValueError("panel clip observation receipt was reused")
        else:
            staged.write_bytes(serialized)
            os.replace(staged, target)
        staged.unlink(missing_ok=True)
        return ArtifactDescriptor(
            kind="panel-clip-observation",
            content_hash=record_hash,
            metadata_hash=tagged_blake3(integrity_blake3(serialized)),
            uri=self.panel_clip_observation_uri(receipt_hash),
        )

    def load_panel_clip_observation(self, receipt_hash: str) -> dict[str, object] | None:
        """Load one batch's clipping record; ``None`` when the batch left none.

        Absence is a fact about the build that produced the receipt (every
        build before partition reuse), and the caller decides what it means --
        for reuse it means the year must be recomputed.
        """
        target = self._panel_clip_observation_path(receipt_hash)
        if not target.is_file():
            return None
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("panel clip observation payload is invalid")
        identity = dict(payload)
        record_hash = identity.pop("record_hash", None)
        if (
            payload.get("receipt_hash") != receipt_hash
            or not isinstance(record_hash, str)
            or scoped_payload_digest(identity, "record_hash", record_hash, digest=canonical_hash)
            != record_hash
        ):
            raise ValueError("panel clip observation identity is invalid")
        return payload

    def _panel_preprocessing_marker_path(self, panel_content_hash: str) -> Path:
        return (
            self.root
            / "feature-panel"
            / "preprocessing-markers"
            / f"{blake3_filename(panel_content_hash)}.json"
        )

    @staticmethod
    def _verify_panel_preprocessing_marker(
        payload: object, panel_content_hash: str
    ) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError("panel preprocessing marker payload is invalid")
        marker_hash = str(payload.get("marker_hash", ""))
        if not _matches_binding_digest(payload, "marker_hash", marker_hash):
            raise ValueError("panel preprocessing marker identity is invalid")
        if payload.get("panel_content_hash") != panel_content_hash:
            raise ValueError("panel preprocessing marker does not name this panel")
        return payload

    def _same_panel_preprocessing_method(
        self,
        sealed: Mapping[str, object],
        proposed: Mapping[str, object],
        *,
        panel_content_hash: str,
    ) -> bool:
        exact = ("recipe_id", "recipe_hash", "implementation_id", "panel_binding_hash")
        if any(sealed.get(field) != proposed.get(field) for field in exact):
            return False
        roles = {
            "catalog_hash": "feature_engine.preprocessing_catalog",
            "implementation_binding_hash": (
                f"feature_engine.preprocessing_implementation.{sealed.get('recipe_id')}"
            ),
        }
        rotated = {field for field in roles if sealed.get(field) != proposed.get(field)}
        before_ref = sealed.get("preprocessing_binding_hash")
        after_ref = proposed.get("preprocessing_binding_hash")
        if not rotated:
            return before_ref == after_ref
        for field in rotated:
            before = sealed.get(field)
            if not isinstance(before, str) or latest(roles[field], before) != proposed.get(field):
                return False

        self._verify_panel_preprocessing_marker(sealed, panel_content_hash)
        # The enclosing hashes rotate with their children. Verify both sealed bindings and
        # compare all their other fields, including the policy hash that the marker omits.
        if not isinstance(before_ref, str) or not isinstance(after_ref, str):
            return False
        bindings = []
        for ref, marker in ((before_ref, sealed), (after_ref, proposed)):
            binding = self.load_panel_preprocessing_binding(ref)
            if binding is None or any(
                binding.get(field) != marker.get(field) for field in (*exact, *roles)
            ):
                return False
            if "implementation_binding_hash" in rotated:
                implementation = binding.get("implementation")
                if (
                    not isinstance(implementation, dict)
                    or implementation.get("implementation_id") != marker.get("implementation_id")
                    or not _matches_binding_digest(
                        implementation,
                        "implementation_binding_hash",
                        marker.get("implementation_binding_hash"),
                    )
                ):
                    return False
            bindings.append(binding)
        changed_fields = {"binding_hash", *rotated}
        if "implementation_binding_hash" in rotated:
            changed_fields.add("implementation")
        old, new = bindings
        return {field: value for field, value in old.items() if field not in changed_fields} == {
            field: value for field, value in new.items() if field not in changed_fields
        }

    def publish_panel_preprocessing_marker(
        self, *, payload: Mapping[str, object], panel_content_hash: str
    ) -> ArtifactDescriptor:
        """Publish the terminal marker that makes a Panel's method reachable.

        Keyed by ``panel_content_hash`` -- the Panel that was actually built -- so a
        reader holding a published Panel resolves it directly instead of scanning,
        and a rebuild producing identical content is idempotent. Keying by the
        panel *binding* would collide: one binding can be rebuilt over a different
        session coverage, and a second marker under one key would let write order
        decide which method the Panel is said to have used.
        """
        self._require_hash(panel_content_hash)
        marker_hash = str(payload.get("marker_hash", ""))
        self._require_hash(marker_hash)
        identity = dict(payload)
        if identity.pop("marker_hash", None) != marker_hash or (
            canonical_hash(identity) != marker_hash
        ):
            raise ValueError("panel preprocessing marker identity is invalid")
        target = self._panel_preprocessing_marker_path(panel_content_hash)
        if target.exists():
            # One Panel content can legitimately be produced by more than one
            # build -- a full materialization and a later incremental replay
            # reach the same rows while doing different amounts of work, so
            # their clipping receipts differ. What may *not* differ is which
            # method produced it. The first marker therefore stands, and a
            # rewrite is refused only when it disagrees about the method.
            marker_bytes = target.read_bytes()
            existing = json.loads(marker_bytes)
            if not self._same_panel_preprocessing_method(
                existing, payload, panel_content_hash=panel_content_hash
            ):
                raise ValueError("panel preprocessing marker contradicts the sealed method")
            marker_hash = str(existing["marker_hash"])
        else:
            marker_bytes = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
            staged.write_bytes(marker_bytes)
            os.replace(staged, target)
            staged.unlink(missing_ok=True)
        return ArtifactDescriptor(
            kind="panel-preprocessing-marker",
            content_hash=marker_hash,
            metadata_hash=tagged_blake3(integrity_blake3(marker_bytes)),
            uri=f"playpen://feature-panel/preprocessing-marker/{panel_content_hash}",
        )

    def load_panel_preprocessing_marker(self, panel_content_hash: str) -> dict[str, object] | None:
        """Load a panel preprocessing marker, retaining explicit legacy absence.

        Resolve one Panel's preprocessing marker, or ``None`` for a Panel built
        before the method seam existed. Absence is legacy readback, never a pass.
        """
        target = self._panel_preprocessing_marker_path(panel_content_hash)
        if not target.is_file():
            return None
        return self._verify_panel_preprocessing_marker(
            json.loads(target.read_text(encoding="utf-8")), panel_content_hash
        )

    def load_feature_panel_semantic_index(self, uri: str) -> dict[str, object]:
        """Read a semantic index and verify its URI-bound canonical identity.

        Args:
            uri: Artifact URI in the method's declared category.

        Returns:
            Verified semantic-index mapping.

        Raises:
            ValueError: The URI category, mapping, cursor or content identity is invalid.
            FileNotFoundError: The addressed artifact is missing.
        """
        prefix = "playpen://feature-panel/semantic-index/"
        if not uri.startswith(prefix):
            raise ValueError("unsupported feature panel semantic index URI")
        index_hash = uri[len(prefix) :]
        target = self._feature_panel_semantic_index_path(index_hash)
        if not target.is_file():
            raise FileNotFoundError("feature panel semantic index is missing")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("feature panel semantic index payload is invalid")
        identity = dict(payload)
        if identity.pop("index_hash", None) != index_hash or _breaks_seal(
            identity, "index_hash", index_hash
        ):
            raise ValueError("feature panel semantic index content is invalid")
        return payload

    def find_feature_panel_semantic_index(
        self, *, panel_snapshot_hash: str
    ) -> tuple[dict[str, object], str] | None:
        """Find the unique immutable index for one panel snapshot."""
        self._require_hash(panel_snapshot_hash)
        root = self._root / self._FEATURE_PANEL_ROOT / "semantic-index"
        matches: list[tuple[dict[str, object], str]] = []
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            index_hash = blake3_from_filename(path.stem)
            if index_hash is None:
                continue
            uri = self.feature_panel_semantic_index_uri(index_hash)
            payload = self.load_feature_panel_semantic_index(uri)
            if payload.get("panel_snapshot_hash") == panel_snapshot_hash:
                matches.append((payload, uri))
        if len(matches) > 1:
            raise ValueError("panel snapshot has multiple semantic indices")
        return matches[0] if matches else None

    def publish_adjusted_return_semantic_revision(
        self, *, payload: Mapping[str, object], chain_hash: str
    ) -> ArtifactDescriptor:
        """Publish the Data Operations semantic ledger and move its pointer last."""
        self._require_hash(chain_hash)
        if payload.get("chain_hash") != chain_hash:
            raise ValueError("adjusted-return revision chain hash does not match payload")
        deltas = payload.get("deltas")
        if not isinstance(deltas, Sequence) or isinstance(deltas, (str, bytes)):
            raise ValueError("adjusted-return revision deltas are invalid")
        delta_hashes: list[str] = []
        for value in deltas:
            if not isinstance(value, Mapping):
                raise ValueError("adjusted-return revision delta is invalid")
            identity = dict(value)
            semantic_hash = identity.pop("semantic_hash", None)
            if not isinstance(semantic_hash, str) or _breaks_seal(
                identity, "semantic_hash", semantic_hash
            ):
                raise ValueError("adjusted-return revision delta identity is invalid")
            delta_hashes.append(str(semantic_hash))
        if payload.get("cursor") != len(delta_hashes):
            raise ValueError("adjusted-return revision cursor is invalid")
        chained = stage_digest(
            (ADJUSTED_RETURN_CHAIN, tuple(delta_hashes)), delta_hashes, canonical_hash
        )
        if chained != chain_hash:
            raise ValueError("adjusted-return revision chain is invalid")

        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        target = self._adjusted_return_revision_path(chain_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        if target.exists():
            if target.read_bytes() != serialized:
                raise ValueError("adjusted-return revision identity was reused")
        else:
            staged.write_bytes(serialized)
            os.replace(staged, target)
        staged.unlink(missing_ok=True)

        pointer_payload = {
            "chain_hash": chain_hash,
            "cursor": len(delta_hashes),
            "uri": self.adjusted_return_semantic_revision_uri(chain_hash),
        }
        pointer = self._adjusted_return_revision_pointer_path()
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer_bytes = json.dumps(pointer_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        pointer_staged = pointer.with_name(f".{pointer.name}.{uuid4().hex}.tmp")
        pointer_staged.write_bytes(pointer_bytes)
        os.replace(pointer_staged, pointer)
        pointer_staged.unlink(missing_ok=True)
        self.load_current_adjusted_return_semantic_revision()
        return ArtifactDescriptor(
            kind="adjusted-return-semantic-revision",
            content_hash=chain_hash,
            metadata_hash=tagged_blake3(integrity_blake3(serialized)),
            uri=self.adjusted_return_semantic_revision_uri(chain_hash),
        )

    def load_adjusted_return_semantic_revision(self, uri: str) -> dict[str, object]:
        """Read a semantic revision and verify every delta and the ordered chain.

        Args:
            uri: Artifact URI in the method's declared category.

        Returns:
            Verified ledger mapping, including its cursor and ordered deltas.

        Raises:
            ValueError: The URI category, mapping, cursor or content identity is invalid.
            FileNotFoundError: The addressed artifact is missing.
        """
        prefix = "playpen://data-operations/adjusted-return-semantic-revisions/"
        if not uri.startswith(prefix):
            raise ValueError("unsupported adjusted-return semantic revision URI")
        chain_hash = uri[len(prefix) :]
        target = self._adjusted_return_revision_path(chain_hash)
        if not target.is_file():
            raise FileNotFoundError("adjusted-return semantic revision is missing")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("adjusted-return semantic revision payload is invalid")
        deltas = payload.get("deltas")
        if not isinstance(deltas, list) or payload.get("cursor") != len(deltas):
            raise ValueError("adjusted-return semantic revision cursor is invalid")
        delta_hashes: list[str] = []
        for value in deltas:
            if not isinstance(value, dict):
                raise ValueError("adjusted-return semantic revision delta is invalid")
            identity = dict(value)
            semantic_hash = identity.pop("semantic_hash", None)
            if not isinstance(semantic_hash, str) or _breaks_seal(
                identity, "semantic_hash", semantic_hash
            ):
                raise ValueError("adjusted-return semantic revision delta identity is invalid")
            delta_hashes.append(str(semantic_hash))
        chained = stage_digest(
            (ADJUSTED_RETURN_CHAIN, tuple(delta_hashes)), delta_hashes, canonical_hash
        )
        if payload.get("chain_hash") != chain_hash or chained != chain_hash:
            raise ValueError("adjusted-return semantic revision chain is invalid")
        return payload

    def load_current_adjusted_return_semantic_revision(self) -> dict[str, object]:
        """Resolve only the marker-last revision projection; never query mutable storage."""
        pointer = self._adjusted_return_revision_pointer_path()
        if not pointer.is_file():
            raise FileNotFoundError("adjusted-return semantic revision pointer is missing")
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("adjusted-return semantic revision pointer is invalid")
        chain_hash = str(payload.get("chain_hash", ""))
        uri = str(payload.get("uri", ""))
        if uri != self.adjusted_return_semantic_revision_uri(chain_hash):
            raise ValueError("adjusted-return semantic revision pointer URI is invalid")
        revision = self.load_adjusted_return_semantic_revision(uri)
        if payload.get("cursor") != revision.get("cursor"):
            raise ValueError("adjusted-return semantic revision pointer cursor is invalid")
        return revision

    def publish_feature_panel_lifecycle_projection(
        self, *, snapshots: Sequence[Mapping[str, object]]
    ) -> str:
        """Atomically publish the read-side lifecycle gate maintained by the host.

        The projection contains no matrix data or physical paths.  Factor
        Research may consult it before opening an immutable snapshot without
        gaining access to mutable DuckDB authority.
        """
        normalized: list[dict[str, object]] = []
        seen: set[str] = set()
        for item in snapshots:
            snapshot_hash = str(item.get("snapshot_hash", ""))
            self._require_hash(snapshot_hash)
            if snapshot_hash in seen:
                raise ValueError("feature panel lifecycle projection has duplicate snapshots")
            seen.add(snapshot_hash)
            lifecycle = str(item.get("lifecycle", ""))
            if lifecycle not in {"ACTIVE", "SUPERSEDED", "QUARANTINED"}:
                raise ValueError("feature panel snapshot lifecycle is invalid")
            reason = item.get("reason")
            if reason is not None and (not isinstance(reason, str) or not reason.strip()):
                raise ValueError("feature panel snapshot lifecycle reason is invalid")
            physical_availability = str(item.get("physical_availability", "AVAILABLE"))
            if physical_availability not in {
                "AVAILABLE",
                "EVICTED_BY_RETENTION",
                "MISSING_OR_TAMPERED",
            }:
                raise ValueError("feature panel snapshot physical availability is invalid")
            eviction_plan_hash = item.get("eviction_plan_hash")
            if (physical_availability == "EVICTED_BY_RETENTION") != (
                isinstance(eviction_plan_hash, str) and is_blake3(eviction_plan_hash)
            ):
                raise ValueError("feature panel snapshot eviction evidence is incomplete")
            if lifecycle == "ACTIVE" and physical_availability != "AVAILABLE":
                raise ValueError("active feature panel snapshot must remain physically available")
            normalized.append(
                {
                    "snapshot_hash": snapshot_hash,
                    "lifecycle": lifecycle,
                    "reason": reason,
                    "physical_availability": physical_availability,
                    "eviction_plan_hash": eviction_plan_hash,
                }
            )
        identity = {
            "kind": "FeaturePanelSnapshotLifecycleProjection",
            "snapshots": sorted(normalized, key=lambda item: str(item["snapshot_hash"])),
        }
        serialized_identity = json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        projection_hash = tagged_blake3(identity_blake3(serialized_identity))
        payload = {**identity, "projection_hash": projection_hash}
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        target = self._feature_panel_lifecycle_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        staged.write_bytes(serialized)
        os.replace(staged, target)
        self._load_feature_panel_lifecycle_projection()
        return projection_hash

    def feature_panel_snapshot_lifecycle(self, manifest_uri: str) -> dict[str, object]:
        """Return the verified lifecycle entry for one snapshot URI."""
        snapshot_hash = self._hash_from_uri(manifest_uri, "manifests")
        payload = self._load_feature_panel_lifecycle_projection()
        snapshots = payload.get("snapshots")
        if not isinstance(snapshots, list):
            raise ValueError("feature panel lifecycle projection has no snapshot catalog")
        for item in snapshots:
            if isinstance(item, dict) and item.get("snapshot_hash") == snapshot_hash:
                return dict(item)
        raise ValueError("feature panel snapshot has no lifecycle admission")

    def resolve_feature_panel_chunk(self, descriptor: ArtifactDescriptor) -> Path:
        """Resolve a chunk after verifying kind, URI, content and metadata identities.

        Args:
            descriptor: Path-free artifact kind, URI, content and metadata identities.

        Returns:
            Trusted physical Parquet path.

        Raises:
            ValueError: The descriptor or stored content/metadata identity is inconsistent.
        """
        if descriptor.kind != "feature-panel-chunk":
            raise ValueError("artifact is not a feature panel chunk")
        if descriptor.uri != self.feature_panel_chunk_uri(descriptor.content_hash):
            raise ValueError("feature panel chunk URI does not match its content hash")
        stored = self._feature_panel_chunk_path(descriptor.content_hash)
        self._validate_feature_panel_chunk(stored, descriptor.content_hash)
        if _metadata_hash(stored) != descriptor.metadata_hash:
            raise ValueError("feature panel chunk metadata hash does not match descriptor")
        return stored

    def inspect_feature_panel_snapshot(self, manifest_uri: str) -> dict[str, object]:
        """Return only the safe summary; no path or matrix crosses the boundary."""
        snapshot_hash = self._hash_from_uri(manifest_uri, "manifests")
        payload = self._load_feature_panel_manifest(
            self._feature_panel_manifest_path(snapshot_hash), snapshot_hash
        )
        summary = payload.get("safe_summary")
        if not isinstance(summary, dict):
            raise ValueError("feature panel manifest has no safe summary")
        return dict(summary)

    def load_feature_panel_manifest(self, manifest_uri: str) -> dict[str, object]:
        """Load a verified manifest for trusted read-side infrastructure.

        Agents and the Front Desk must use :meth:`inspect_feature_panel_snapshot`
        instead.  This method exposes chunk references, never physical paths.
        """
        snapshot_hash = self._hash_from_uri(manifest_uri, "manifests")
        payload = self._load_feature_panel_manifest(
            self._feature_panel_manifest_path(snapshot_hash), snapshot_hash
        )
        identity = dict(payload)
        identity.pop("snapshot_hash", None)
        encoded = json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        if tagged_blake3(identity_blake3(encoded)) != snapshot_hash:
            raise ValueError("feature panel manifest payload hash is invalid")
        return payload

    def resolve_feature_panel_chunk_ref(
        self, *, uri: str, content_hash: str, metadata_hash: str
    ) -> Path:
        """Resolve one manifest-owned chunk after URI and metadata validation."""
        descriptor = ArtifactDescriptor(
            kind="feature-panel-chunk",
            content_hash=content_hash,
            metadata_hash=metadata_hash,
            uri=uri,
        )
        return self.resolve_feature_panel_chunk(descriptor)

    def feature_panel_reachability(
        self, *, root_manifest_uris: Sequence[str]
    ) -> ArtifactReachabilityReport:
        """Classify panel artifacts by manifest reachability without deleting anything."""
        panel_root = self._root / self._FEATURE_PANEL_ROOT
        manifests = {
            identity: path
            for path in (panel_root / "manifests").glob("*.json")
            if (identity := blake3_from_filename(path.stem)) is not None
        }
        chunks = {
            identity: path
            for path in (panel_root / "chunks").glob("*.parquet")
            if (identity := blake3_from_filename(path.stem)) is not None
        }
        reachable_manifests: set[str] = set()
        reachable_chunks: set[str] = set()
        unknown: set[str] = set()
        for uri in root_manifest_uris:
            try:
                snapshot_hash = self._hash_from_uri(uri, "manifests")
                payload = self._load_feature_panel_manifest(
                    self._feature_panel_manifest_path(snapshot_hash), snapshot_hash
                )
            except (FileNotFoundError, ValueError):
                unknown.add(uri)
                continue
            reachable_manifests.add(snapshot_hash)
            for item in payload.get("chunks", []):
                if not isinstance(item, dict) or not isinstance(item.get("chunk_hash"), str):
                    unknown.add(uri)
                    continue
                reachable_chunks.add(str(item["chunk_hash"]))
        all_known_paths = set(manifests.values()) | set(chunks.values())
        lifecycle_path = self._feature_panel_lifecycle_path()
        if lifecycle_path.is_file():
            all_known_paths.add(lifecycle_path)
        incomplete = {
            str(path.relative_to(self._root))
            for path in panel_root.rglob("*.tmp")
            if path.is_file()
        }
        unknown.update(
            str(path.relative_to(self._root))
            for path in panel_root.rglob("*")
            if path.is_file() and path not in all_known_paths and not path.name.endswith(".tmp")
        )
        reachable = {self.feature_panel_manifest_uri(value) for value in reachable_manifests} | {
            self.feature_panel_chunk_uri(value) for value in reachable_chunks
        }
        unreferenced = {
            self.feature_panel_manifest_uri(value) for value in set(manifests) - reachable_manifests
        } | {self.feature_panel_chunk_uri(value) for value in set(chunks) - reachable_chunks}
        return ArtifactReachabilityReport(
            reachable=tuple(sorted(reachable)),
            unreferenced=tuple(sorted(unreferenced)),
            incomplete_publication=tuple(sorted(incomplete)),
            unknown_owner=tuple(sorted(unknown)),
        )

    @staticmethod
    def _validate_feature_input(path: Path, content_hash: str) -> None:
        if not path.is_file():
            raise FileNotFoundError("content-addressed feature input is missing")
        metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
        if metadata.get(b"alphalattice.snapshot_kind", b"") != b"FeatureInputSnapshot":
            raise ValueError("artifact is not a FeatureInputSnapshot")
        if metadata.get(b"alphalattice.snapshot_hash", b"").decode("utf-8") != content_hash:
            raise ValueError("artifact snapshot hash does not match requested URI")

    @staticmethod
    def _require_hash(value: str) -> None:
        if not is_blake3(value):
            raise ValueError("artifact content hash must be tagged BLAKE3")

    @staticmethod
    def _validate_feature_panel_chunk(path: Path, content_hash: str) -> Mapping[bytes, bytes]:
        if not path.is_file():
            raise FileNotFoundError("content-addressed feature panel chunk is missing")
        metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
        if metadata.get(b"alphalattice.snapshot_kind") != b"FeaturePanelChunk":
            raise ValueError("artifact is not a FeaturePanelChunk")
        if metadata.get(b"alphalattice.chunk_hash", b"").decode("utf-8") != content_hash:
            raise ValueError("feature panel chunk hash does not match requested URI")
        return metadata

    @staticmethod
    def _load_feature_panel_manifest(path: Path, snapshot_hash: str) -> dict[str, object]:
        if not path.is_file():
            raise FileNotFoundError("content-addressed feature panel manifest is missing")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("snapshot_hash") != snapshot_hash:
            raise ValueError("feature panel manifest content does not match requested URI")
        return payload

    def _load_feature_panel_lifecycle_projection(self) -> dict[str, object]:
        path = self._feature_panel_lifecycle_path()
        if not path.is_file():
            raise FileNotFoundError("feature panel lifecycle projection is missing")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("feature panel lifecycle projection is invalid")
        claimed_hash = payload.get("projection_hash")
        identity = dict(payload)
        identity.pop("projection_hash", None)
        encoded = json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        actual_hash = tagged_blake3(identity_blake3(encoded))
        if claimed_hash != actual_hash:
            raise ValueError("feature panel lifecycle projection hash is invalid")
        return payload

    @staticmethod
    def _hash_from_uri(uri: str, category: str) -> str:
        prefix = f"playpen://feature-panel/{category}/"
        if not uri.startswith(prefix):
            raise ValueError("unsupported feature panel artifact URI")
        content_hash = uri[len(prefix) :]
        ArtifactResolver._require_hash(content_hash)
        return content_hash
