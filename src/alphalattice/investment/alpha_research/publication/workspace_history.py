"""Verified Parquet observation history under Alpha's existing publication root."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from alphalattice.control.workspace_runtime.content_store import (
    ContentAddressedStore,
    ContentAddressedStoreError,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import (
    WorkspaceObservationHistoryHead,
    WorkspaceObservationHistoryMarker,
    WorkspaceObservationHistoryPart,
    seal_current_contract,
)

type _Columns = Mapping[str, npt.NDArray[np.float64]]
type _PartColumns = dict[str, npt.NDArray[Any]]

_BASE = "current/workspace-observation-history"
_PARTS = f"{_BASE}/parts"
_HEADS = f"{_BASE}/heads"
_MARKERS = f"{_BASE}/markers"
_INVALID = "alpha_research.workspace_observation_history_invalid"
_MISSING = "alpha_research.workspace_observation_history_missing"
_REUSE_INVALID = "alpha_research.workspace_observation_history_reuse_invalid"


def _part_category(column_names: tuple[str, ...], shape: tuple[int, int]) -> str:
    layout = canonical_hash({"column_names": column_names, "shape": shape, "dtype": "float64"})
    return f"{_PARTS}/{layout}"


@dataclass(frozen=True, slots=True)
class WorkspaceObservationHistoryFile:
    """Exact file bytes for a retention root or an unrooted cleanup candidate."""

    path: Path
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class WorkspaceObservationHistoryRetention:
    """Deterministic protected files and eligible files; this value deletes nothing."""

    roots: tuple[WorkspaceObservationHistoryFile, ...]
    targets: tuple[WorkspaceObservationHistoryFile, ...]
    head_hashes: tuple[str, ...]


class _WorkspaceObservationHistoryStore:
    """Alpha's marker-last history owner; its caller holds the workspace mutation gate."""

    def __init__(self, root: Path, *, error: type[ValueError]) -> None:
        self.root = root
        self.error = error
        self.content = ContentAddressedStore(root, uri_prefix="artifact://alpha-research")

    def _require_hash(self, value: str) -> None:
        try:
            self.content.require_hash(value)
        except (ContentAddressedStoreError, TypeError) as error:
            raise self.error(_INVALID) from error

    def _path(self, category: str, value: str, extension: str) -> Path:
        self._require_hash(value)
        target = self.root / category / f"{value}.{extension}"
        if not target.resolve().is_relative_to(self.root.resolve()):
            raise self.error(_INVALID)
        return target

    def _marker(self, scope_hash: str) -> WorkspaceObservationHistoryMarker | None:
        target = self._path(_MARKERS, scope_hash, "json")
        try:
            payload = target.read_bytes()
        except FileNotFoundError:
            if target.is_symlink():
                raise self.error(_MISSING) from None
            return None
        except OSError as error:
            raise self.error(_INVALID) from error
        try:
            marker = WorkspaceObservationHistoryMarker.model_validate_json(payload)
        except ValueError as error:
            raise self.error(_INVALID) from error
        if marker.scope_hash != scope_hash:
            raise self.error(_INVALID)
        return marker

    def _head(self, head_hash: str) -> WorkspaceObservationHistoryHead:
        self._path(_HEADS, head_hash, "json")
        try:
            return self.content.load_model(
                category=_HEADS,
                content_hash=head_hash,
                model=WorkspaceObservationHistoryHead,
                identity_field="head_hash",
            )
        except ContentAddressedStoreError as error:
            code = (
                _MISSING if str(error).startswith("content_store.artifact_missing:") else _INVALID
            )
            raise self.error(code) from error

    def _part(
        self,
        head: WorkspaceObservationHistoryHead,
        part: WorkspaceObservationHistoryPart,
        verified: dict[str, _PartColumns],
    ) -> _PartColumns:
        expected_shape = (part.stop - part.start, len(head.ordered_listing_ids))
        category = _part_category(head.column_names, expected_shape)
        target = self._path(category, part.content_hash, "parquet")
        try:
            key = f"{category}/{part.content_hash}"
            if key not in verified:
                verified[key] = self.content.load_columns(
                    category=category, content_hash=part.content_hash
                )
            columns = verified[key]
            metadata = pq.read_metadata(target)
            shape = json.loads((metadata.metadata or {})[b"shape"])
            if (
                shape != list(expected_shape)
                or any(type(value) is not int for value in shape)
                or metadata.num_rows != expected_shape[0] * expected_shape[1]
                or tuple(columns) != head.column_names
                or any(
                    values.dtype != np.dtype("float64")
                    or values.shape != (expected_shape[0] * expected_shape[1],)
                    for values in columns.values()
                )
            ):
                raise self.error(_INVALID)
        except ContentAddressedStoreError as error:
            code = (
                _MISSING if str(error).startswith("content_store.artifact_missing:") else _INVALID
            )
            raise self.error(code) from error
        except (OSError, pa.ArrowException, KeyError, TypeError, json.JSONDecodeError) as error:
            raise self.error(_INVALID) from error
        return columns

    def _columns(
        self, head: WorkspaceObservationHistoryHead, verified: dict[str, _PartColumns]
    ) -> _Columns:
        pieces = [self._part(head, part, verified) for part in head.parts]
        shape = (len(head.formation_sessions), len(head.ordered_listing_ids))
        return MappingProxyType(
            {
                name: np.frombuffer(
                    b"".join(part[name].tobytes(order="C") for part in pieces), dtype=np.float64
                ).reshape(shape)
                for name in head.column_names
            }
        )

    def load(self, scope_hash: str) -> tuple[WorkspaceObservationHistoryHead, _Columns] | None:
        """Absent initial marker is cold; all named current/previous artifacts fail closed."""
        marker = self._marker(scope_hash)
        if marker is None:
            return None
        head = self._head(marker.current_head_hash)
        if head.scope_hash != scope_hash or head.predecessor_head_hash != marker.previous_head_hash:
            raise self.error(_INVALID)
        verified: dict[str, _PartColumns] = {}
        columns = self._columns(head, verified)
        if marker.previous_head_hash is not None:
            previous = self._head(marker.previous_head_hash)
            if previous.scope_hash != scope_hash:
                raise self.error(_INVALID)
            for part in previous.parts:
                self._part(previous, part, verified)
        return head, columns

    def publish(
        self,
        *,
        scope_hash: str,
        selection_hash: str,
        dependency_prefix_hash: str,
        formation_sessions: tuple[date, ...],
        ordered_listing_ids: tuple[str, ...],
        stable_session_count: int,
        columns: Mapping[str, npt.NDArray[Any]],
        reuse: WorkspaceObservationHistoryHead | None,
        capacity: Callable[[int], None],
        loaded: tuple[WorkspaceObservationHistoryHead, _Columns] | None = None,
    ) -> WorkspaceObservationHistoryHead:
        """Admit every staging write and commit a marker after all exact parts and head.

        ``loaded`` is the caller's `load` of this slot; it stands for the current head while
        the marker still names it, so the slot is not read a second time.
        """
        for value in (scope_hash, selection_hash, dependency_prefix_hash):
            self._require_hash(value)
        if (
            not callable(capacity)
            or not formation_sessions
            or any(type(value) is not date for value in formation_sessions)
            or formation_sessions != tuple(sorted(set(formation_sessions)))
            or not ordered_listing_ids
            or any(not isinstance(value, str) or not value for value in ordered_listing_ids)
            or len(set(ordered_listing_ids)) != len(ordered_listing_ids)
            or type(stable_session_count) is not int
            or not 0 <= stable_session_count <= len(formation_sessions)
            or not columns
            or any(not isinstance(name, str) or not name for name in columns)
        ):
            raise self.error(_INVALID)
        shape = (len(formation_sessions), len(ordered_listing_ids))
        if any(
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float64")
            or value.shape != shape
            for value in columns.values()
        ):
            raise self.error(_INVALID)
        held = {
            name: np.frombuffer(columns[name].tobytes(order="C"), dtype=np.float64).reshape(shape)
            for name in sorted(columns)
        }
        marker = self._marker(scope_hash) if loaded is not None else None
        current = (
            loaded
            if loaded is not None
            and marker is not None
            and loaded[0].head_hash == marker.current_head_hash
            else self.load(scope_hash)
        )
        reusable: tuple[WorkspaceObservationHistoryPart, ...] = ()
        if reuse is not None:
            old = (
                current[0]
                if current is not None and current[0].head_hash == reuse.head_hash
                else self._head(reuse.head_hash)
            )
            if old != reuse:
                raise self.error(_REUSE_INVALID)
            old_columns = (
                current[1] if current is not None and old == current[0] else self._columns(old, {})
            )
            if old.scope_hash == scope_hash and old.selection_hash == selection_hash:
                common = min(old.stable_session_count, stable_session_count)
                if (
                    old.ordered_listing_ids != ordered_listing_ids
                    or old.column_names != tuple(held)
                    or old.formation_sessions[:common] != formation_sessions[:common]
                    or any(
                        old_columns[name][:common].tobytes(order="C")
                        != held[name][:common].tobytes(order="C")
                        for name in held
                    )
                ):
                    raise self.error(_REUSE_INVALID)
                reusable = tuple(part for part in old.parts if part.stop <= common)
        if current is not None:
            old, old_columns = current
            if (
                old.selection_hash == selection_hash
                and old.dependency_prefix_hash == dependency_prefix_hash
                and old.formation_sessions == formation_sessions
                and old.ordered_listing_ids == ordered_listing_ids
                and old.column_names == tuple(held)
                and old.stable_session_count == stable_session_count
                and all(
                    old_columns[name].tobytes(order="C") == held[name].tobytes() for name in held
                )
            ):
                return old
        content = ContentAddressedStore(
            self.root, uri_prefix="artifact://alpha-research", capacity=capacity
        )
        parts = list(reusable)
        start = parts[-1].stop if parts else 0
        for stop in (stable_session_count, len(formation_sessions)):
            if stop <= start:
                continue
            try:
                content_hash = content.publish_columns(
                    category=_part_category(tuple(held), (stop - start, len(ordered_listing_ids))),
                    columns={name: value[start:stop] for name, value in held.items()},
                )
            except ContentAddressedStoreError as error:
                raise self.error(_INVALID) from error
            parts.append(
                seal_current_contract(
                    WorkspaceObservationHistoryPart,
                    {"start": start, "stop": stop, "content_hash": content_hash},
                    "part_hash",
                )
            )
            start = stop
        head = seal_current_contract(
            WorkspaceObservationHistoryHead,
            {
                "scope_hash": scope_hash,
                "selection_hash": selection_hash,
                "dependency_prefix_hash": dependency_prefix_hash,
                "formation_sessions": formation_sessions,
                "ordered_listing_ids": ordered_listing_ids,
                "column_names": tuple(held),
                "stable_session_count": stable_session_count,
                "predecessor_head_hash": current[0].head_hash if current is not None else None,
                "parts": tuple(parts),
            },
            "head_hash",
        )
        # The reused parts were verified when this call read them; the new ones are read back.
        verified: dict[str, _PartColumns] = {}
        for part in head.parts[len(reusable) :]:
            self._part(head, part, verified)
        try:
            content.publish_model(category=_HEADS, value=head, identity_field="head_hash")
        except ContentAddressedStoreError as error:
            raise self.error(_INVALID) from error
        self._head(head.head_hash)
        if current is not None and (formation_sessions[-1], len(formation_sessions)) < (
            current[0].formation_sessions[-1],
            len(current[0].formation_sessions),
        ):
            return head
        marker = seal_current_contract(
            WorkspaceObservationHistoryMarker,
            {
                "scope_hash": scope_hash,
                "current_head_hash": head.head_hash,
                "previous_head_hash": head.predecessor_head_hash,
            },
            "marker_hash",
        )
        content.atomic_write(
            self._path(_MARKERS, scope_hash, "json"),
            json.dumps(
                marker.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode(),
        )
        return head

    def retention(
        self,
        *,
        referenced_heads: tuple[str, ...],
        active_scope_hashes: tuple[str, ...] | None,
    ) -> WorkspaceObservationHistoryRetention:
        """Protect current/previous slots and explicit heads; propose only unrooted files."""
        for value in (*referenced_heads, *(active_scope_hashes or ())):
            self._require_hash(value)
        protected: set[Path] = set()
        heads = set(referenced_heads)
        all_heads: set[str] = set()
        files: set[Path] = set()
        for category, extension in ((_MARKERS, "json"), (_HEADS, "json"), (_PARTS, "parquet")):
            directory = self.root / category
            if not directory.exists():
                continue
            if (
                not directory.resolve().is_relative_to(self.root.resolve())
                or not directory.is_dir()
            ):
                raise self.error(_INVALID)
            for path in sorted(directory.rglob("*") if category == _PARTS else directory.iterdir()):
                if path.is_dir() and category == _PARTS and path.parent == directory:
                    self._require_hash(path.name)
                    if not path.resolve().is_relative_to(self.root.resolve()):
                        raise self.error(_INVALID)
                    continue
                if not path.is_file() or not path.resolve().is_relative_to(self.root.resolve()):
                    raise self.error(_INVALID)
                if category == _PARTS and path.parent.parent != directory:
                    raise self.error(_INVALID)
                files.add(path)
                if path.name.startswith(".") and path.name.endswith(".tmp"):
                    continue
                if path.suffix != f".{extension}":
                    raise self.error(_INVALID)
                self._require_hash(path.stem)
                if category == _MARKERS:
                    loaded = self.load(path.stem)
                    if loaded is None:
                        raise self.error(_MISSING)
                    marker = self._marker(path.stem)
                    assert marker is not None
                    if active_scope_hashes is None or path.stem in active_scope_hashes:
                        protected.add(path)
                        heads.add(marker.current_head_hash)
                        if marker.previous_head_hash is not None:
                            heads.add(marker.previous_head_hash)
                elif category == _HEADS:
                    head = self._head(path.stem)
                    all_heads.add(head.head_hash)
                    for part in head.parts:
                        self._part(head, part, {})
                else:
                    try:
                        metadata = pq.read_metadata(path)
                        shape = json.loads((metadata.metadata or {})[b"shape"])
                        values = self.content.load_columns(
                            category=f"{_PARTS}/{path.parent.name}", content_hash=path.stem
                        )
                        if (
                            not isinstance(shape, list)
                            or len(shape) != 2
                            or any(type(v) is not int or v < 1 for v in shape)
                            or metadata.num_rows != shape[0] * shape[1]
                            or tuple(values) != tuple(sorted(values))
                            or not values
                            or any(v.dtype != np.dtype("float64") for v in values.values())
                            or f"{_PARTS}/{path.parent.name}"
                            != _part_category(tuple(values), (shape[0], shape[1]))
                        ):
                            raise self.error(_INVALID)
                    except ContentAddressedStoreError as error:
                        raise self.error(_INVALID) from error
                    except (
                        OSError,
                        pa.ArrowException,
                        KeyError,
                        TypeError,
                        json.JSONDecodeError,
                    ) as error:
                        raise self.error(_INVALID) from error
        for head_hash in sorted(heads):
            head = self._head(head_hash)
            protected.add(self._path(_HEADS, head_hash, "json"))
            for part in head.parts:
                self._part(head, part, {})
                category = _part_category(
                    head.column_names, (part.stop - part.start, len(head.ordered_listing_ids))
                )
                protected.add(self._path(category, part.content_hash, "parquet"))

        def fact(path: Path) -> WorkspaceObservationHistoryFile:
            payload = path.read_bytes()
            return WorkspaceObservationHistoryFile(path, sha256(payload).hexdigest(), len(payload))

        return WorkspaceObservationHistoryRetention(
            roots=tuple(fact(path) for path in sorted(protected)),
            targets=tuple(fact(path) for path in sorted(files - protected)),
            head_hashes=tuple(sorted(all_heads)),
        )
