"""Verified Parquet observation history under Alpha's existing publication root."""

from __future__ import annotations

import json
from bisect import bisect_left
from collections.abc import Callable, Iterable, Mapping, Sequence
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
from alphalattice.foundation.market_data_ops.storage.rebase import PRICES
from alphalattice.foundation.market_data_ops.storage.rebase import apply as apply_rebase
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import (
    HistoryProofParts,
    StoredRebase,
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


HISTORY_PART_SESSIONS = 21
"""Sessions per stable history part: a day rewrites the parts its changed sessions fall in."""


def _same_parts(
    current: tuple[WorkspaceObservationHistoryHead, _Columns] | None,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
    names: tuple[str, ...],
) -> dict[tuple[int, int], WorkspaceObservationHistoryPart]:
    """The current head's parts by session range, while its axes and columns are these."""
    if current is None or current[0].ordered_listing_ids != listings:
        return {}
    head = current[0]
    if head.column_names != names:
        return {}
    return {
        (part.start, part.stop): part
        for part in head.parts
        if head.formation_sessions[: part.stop] == sessions[: part.stop]
    }


def rebased(
    columns: Mapping[str, npt.NDArray[np.float64]],
    listings: Sequence[str],
    rebases: Iterable[StoredRebase],
    prefix: str = "ohlcv::",
) -> dict[str, npt.NDArray[np.float64]]:
    """These columns as read: each stored rebase scales its listing's rows, in the order listed.

    The OHLCV columns are copied when any rebase has rows and held, as every read array is, by
    immutable bytes; every other column is as given.
    """
    rebases = [item for item in rebases if item.rows]
    out = dict(columns)
    if not rebases:
        return out
    view = {name: np.array(out[prefix + name]) for name in (*PRICES, "volume")}
    index = {listing: column for column, listing in enumerate(listings)}
    for item in rebases:
        for start, stop in item.rows:
            apply_rebase(
                view, slice(start, stop), index[item.listing_id], item.price_factor,
                item.volume_factor,
            )  # fmt: skip
    for name, values in view.items():
        out[prefix + name] = _held(values)
    return out


type RebaseKeep = Mapping[str, tuple[npt.NDArray[np.bool_], Mapping[str, npt.NDArray[np.float64]]]]
"""Each rebased listing's kept rows: which rows of the kept history, and its values as stored."""


def _ranges(mask: npt.NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return tuple((int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True))


def rebase_keep(
    history: tuple[WorkspaceObservationHistoryHead, _Columns] | None,
    listings: Sequence[str],
    fresh: Mapping[str, npt.NDArray[np.float64]],
    known: Sequence[Any],
    agreement: tuple[float, float],
    rule_hash: str = "",
) -> tuple[tuple[StoredRebase, ...], RebaseKeep]:
    """The rebases a new head lists, and the rebased listings' rows it keeps as stored.

    `known` are the store's rebases in the order it applied them. A row is kept when they
    give its new OHLCV from the stored one exactly and its other columns, which no rebase
    scales, agree within the data rule's `agreement` (relative, then absolute near zero): a
    part or block that differs only so keeps its bytes, and each rebase lists the rows that
    need it, and the hash of the rule whose agreement kept them.
    """
    head, stored = history if history is not None else (None, {})
    entries = list(head.rebases) if head is not None else []
    sessions = head.formation_sessions if head is not None else ()
    absorbed = {item.receipt_hash for item in entries}
    entries += [
        StoredRebase(
            listing_id=item.listing_id, receipt_hash=item.receipt_hash,
            price_factor=item.price_factor, volume_factor=item.volume_factor,
            rows=((0, bisect_left(sessions, item.ex_date)),),
        )
        for item in known
        if item.receipt_hash not in absorbed
    ]  # fmt: skip
    keep: dict[str, Any] = {}
    within: dict[str, tuple[int, float]] = {}
    if (
        head is not None
        and head.ordered_listing_ids == tuple(listings)
        and set(stored) == set(fresh)
    ):
        size = min(len(sessions), len(next(iter(fresh.values()))))  # a shorter capture bounds it
        for listing in dict.fromkeys(item.listing_id for item in entries if item.rows):
            column = listings.index(listing)
            old = {name: np.array(values[:size, column]) for name, values in stored.items()}
            mine = [item for item in entries if item.listing_id == listing]
            fresh_column = {name: fresh[name][:size, column] for name in old}
            mask, within[listing] = _kept_rows(old, fresh_column, mine, listing, agreement)
            keep[listing] = (mask, old)
    return tuple(
        item.model_copy(
            update={
                "rows": _ranges(
                    keep[item.listing_id][0] & _covered(item.rows, len(keep[item.listing_id][0]))
                )
                if item.listing_id in keep
                else ()
            }
            | (
                {
                    "kept_within": within[item.listing_id][0],
                    "kept_largest": within[item.listing_id][1],
                    "kept_rule_hash": rule_hash,
                }
                if item.receipt_hash not in absorbed and item.listing_id in within
                else {}
            )
        )
        for item in entries
    ), MappingProxyType(keep)


def _kept_rows(
    old: Mapping[str, npt.NDArray[np.float64]],
    fresh: Mapping[str, npt.NDArray[np.float64]],
    rebases: Sequence[StoredRebase],
    listing: str,
    agreement: tuple[float, float],
) -> tuple[npt.NDArray[np.bool_], tuple[int, float]]:
    """One listing's rows kept as stored, and how many of them, with the largest difference,
    differed within the agreement: its OHLCV exactly as rebased, every other column close."""
    view = rebased({n: v.reshape(-1, 1) for n, v in old.items()}, (listing,), rebases)
    size = len(next(iter(old.values())))
    mask, pairs = np.ones(size, dtype=np.bool_), []
    for name, values in view.items():
        a, b = values.reshape(-1), fresh[name]
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        close = np.isclose(a, b, rtol=agreement[0], atol=agreement[1], equal_nan=True)
        mask &= same if name.startswith("ohlcv::") else close
        pairs.append((a, b, same))
    unequal, largest = np.zeros(size, dtype=np.bool_), 0.0
    for a, b, same in pairs:
        moved = mask & ~same
        unequal |= moved
        gap, base = np.abs(a - b)[moved], np.abs(b)[moved]
        gap = np.divide(gap, base, out=gap, where=base > 0)  # a zero keeps its absolute gap
        largest = max(largest, float(gap.max(initial=0.0)))
    return mask, (int(unequal.sum()), largest)


def _covered(rows: Sequence[tuple[int, int]], size: int) -> npt.NDArray[np.bool_]:
    mask = np.zeros(size, dtype=np.bool_)
    for start, stop in rows:
        mask[start:stop] = True
    return mask


def patched(
    values: npt.NDArray[Any],
    name: str,
    start: int,
    keep: RebaseKeep,
    listings: Sequence[str],
) -> npt.NDArray[Any]:
    """Rows of one column from `start`, each rebased listing's kept rows as stored."""
    out = None
    for listing, (mask, old) in keep.items():
        rows = mask[start : start + len(values)]
        if name not in old or listing not in listings or not rows.any():
            continue
        column, kept = listings.index(listing), old[name][start : start + len(rows)][rows]
        if values[: len(rows)][rows, column].tobytes() == kept.tobytes():
            continue  # already as stored: no copy
        if out is None:
            out = np.array(values)
        out[: len(rows)][rows, column] = kept
    return values if out is None else out


def _publishable(
    capacity: object,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
    stable: int,
    columns: Mapping[str, npt.NDArray[Any]],
) -> bool:
    """Whether a publication's arguments are a history's: dated axes, its listings, its columns."""
    shape = (len(sessions), len(listings))
    return not (
        not callable(capacity)
        or not sessions
        or any(type(value) is not date for value in sessions)
        or sessions != tuple(sorted(set(sessions)))
        or not listings
        or any(not isinstance(value, str) or not value for value in listings)
        or len(set(listings)) != len(listings)
        or type(stable) is not int
        or not 0 <= stable <= len(sessions)
        or not columns
        or any(not isinstance(name, str) or not name for name in columns)
        or any(
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float64")
            or value.shape != shape
            for value in columns.values()
        )
    )


def _stored(
    columns: Mapping[str, npt.NDArray[Any]], kept: RebaseKeep, listings: Sequence[str]
) -> dict[str, npt.NDArray[Any]]:
    """Each column as it is stored, a rebased listing's kept rows as they were, sorted by name."""
    return {
        name: _held(patched(columns[name], name, 0, kept, listings)) for name in sorted(columns)
    }


def _held(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """A column as published: as given when immutable bytes back it, as one copy otherwise."""
    root = values
    while isinstance(root.base, np.ndarray):
        root = root.base
    if isinstance(root.base, bytes) and values.flags.c_contiguous and not values.flags.writeable:
        return values
    return np.frombuffer(values.tobytes(order="C"), dtype=np.float64).reshape(values.shape)


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
        self, head: WorkspaceObservationHistoryHead, seen: set[str] | None = None
    ) -> _Columns:
        """Each column whole, every part copied in as it is read and then let go.

        One history and one part or column are held at once; `seen` collects the verified parts'
        keys.
        """
        shape = (len(head.formation_sessions), len(head.ordered_listing_ids))
        columns: dict[str, npt.NDArray[np.float64]] = {
            name: np.empty(shape, dtype=np.float64) for name in head.column_names
        }
        start = 0
        for part in head.parts:
            verified: dict[str, _PartColumns] = {}
            piece = self._part(head, part, verified)
            stop = start + part.stop - part.start
            if stop > shape[0]:
                raise self.error(_INVALID)
            for name in head.column_names:
                columns[name][start:stop] = piece[name].reshape(stop - start, shape[1])
            start = stop
            if seen is not None:
                seen.update(verified)
        if start != shape[0]:
            raise self.error(_INVALID)
        for name in head.column_names:  # immutable bytes back each, one column copied at a time
            columns[name] = np.frombuffer(columns[name].tobytes(), dtype=np.float64).reshape(shape)
        return MappingProxyType(columns)

    def load(self, scope_hash: str) -> tuple[WorkspaceObservationHistoryHead, _Columns] | None:
        """Absent initial marker is cold; all named current/previous artifacts fail closed."""
        marker = self._marker(scope_hash)
        if marker is None:
            return None
        head = self._head(marker.current_head_hash)
        if head.scope_hash != scope_hash or head.predecessor_head_hash != marker.previous_head_hash:
            raise self.error(_INVALID)
        seen: set[str] = set()
        columns = self._columns(head, seen)
        if marker.previous_head_hash is not None:
            previous = self._head(marker.previous_head_hash)
            if previous.scope_hash != scope_hash:
                raise self.error(_INVALID)
            for part in previous.parts:  # a part the current head shares was verified above
                shape = (part.stop - part.start, len(previous.ordered_listing_ids))
                key = f"{_part_category(previous.column_names, shape)}/{part.content_hash}"
                if key not in seen:
                    self._part(previous, part, {})
        return head, columns

    def _parts(
        self,
        content: ContentAddressedStore,
        held: dict[str, npt.NDArray[Any]],
        parts: list[WorkspaceObservationHistoryPart],
        stable_session_count: int,
        current: tuple[WorkspaceObservationHistoryHead, _Columns] | None,
        formation_sessions: tuple[date, ...],
        ordered_listing_ids: tuple[str, ...],
    ) -> list[WorkspaceObservationHistoryPart]:
        """The head's parts after the reused ones.

        The stable sessions in fixed parts, each kept while the current head holds it as it is,
        then the open sessions.
        """
        start = parts[-1].stop if parts else 0
        # Stable sessions in fixed parts: a part whose sessions and values the current head holds
        # as they are is kept, so a listing restated from a recent session rewrites only its parts.
        held_parts = _same_parts(current, formation_sessions, ordered_listing_ids, tuple(held))
        first = (start // HISTORY_PART_SESSIONS + 1) * HISTORY_PART_SESSIONS
        bounds = range(first, stable_session_count, HISTORY_PART_SESSIONS)
        for stop in (*bounds, stable_session_count, len(formation_sessions)):
            if stop <= start:
                continue
            kept = held_parts.get((start, stop))
            if (
                kept is not None
                and current is not None
                and all(
                    current[1][name][start:stop].tobytes() == values[start:stop].tobytes()
                    for name, values in held.items()
                )
            ):
                parts.append(kept)
                start = stop
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
        return parts

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
        proofs: HistoryProofParts | None = None,
        rebased: tuple[tuple[StoredRebase, ...], RebaseKeep] = ((), MappingProxyType({})),
    ) -> WorkspaceObservationHistoryHead:
        """Admit every staging write and commit a marker after all exact parts and head.

        ``loaded`` is the caller's `load` of this slot; it stands for the current head while
        the marker still names it, so the slot is not read a second time. ``rebased`` holds
        the rebases its readers apply and the rows written as stored (`rebase_keep`).
        """
        for value in (scope_hash, selection_hash, dependency_prefix_hash):
            self._require_hash(value)
        if not _publishable(
            capacity, formation_sessions, ordered_listing_ids, stable_session_count, columns
        ):
            raise self.error(_INVALID)
        held = _stored(columns, rebased[1], ordered_listing_ids)
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
                current[1] if current is not None and old == current[0] else self._columns(old)
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
                and old.rebases == rebased[0]
                and all(
                    old_columns[name].tobytes(order="C") == held[name].tobytes() for name in held
                )
            ):
                return old
        content = ContentAddressedStore(
            self.root, uri_prefix="artifact://alpha-research", capacity=capacity
        )
        parts = self._parts(
            content,
            held,
            list(reusable),
            stable_session_count,
            current,
            formation_sessions,
            ordered_listing_ids,
        )
        head = WorkspaceObservationHistoryHead.sealed(
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
                "rebases": rebased[0],
            },
            proofs,
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
