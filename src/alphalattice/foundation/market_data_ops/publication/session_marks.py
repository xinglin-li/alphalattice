"""The intraday move of one session, as a strategy-neutral fact.

A decision taken at ``close(T)`` knows what happened between that session's open
and its close. Nothing published here could tell it: the execution outcome
carries opens, the Risk return surface carries open-to-open, and the tradability
surface carries eligibility and liquidity. So a Portfolio holding a book valued
at ``open(T)`` had no lawful way to re-mark it, and the only reference available
was one trading day stale.

This is the missing half of that pair, and it is deliberately *only* the fact.
It states what a session did between its own two published prices and carries no
schedule, no entry, no exit and no rebalance policy -- one session mark serves a
next-open strategy, a close-auction strategy and an intraday strategy alike.

``project_research_series`` remains the authority for prices and corporate
actions; this reads its output and does not re-derive an adjustment. The two
prices come from one ``ProjectedBar``, so they share a split factor and an action
set by construction, and the ratio cannot straddle an adjustment boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.market_data_ops.publication.projection import (
    ProjectedBar,
    project_research_series,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.shared_kernel.arrow_identity import canonical_row_hashes
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_HASH = r"^[0-9a-f]{64}$"

SESSION_MARK_FORMULA_IDENTITY: Final = (
    "close_split_adjusted / open_split_adjusted - 1, within one session"
)
"""What the number is, spelled out rather than named.

No dividend term, and that is a property of the window rather than an omission:
a cash dividend goes ex at the open, so an open-to-close interval on one session
never spans one. An open-to-*open* interval does, which is why the Risk return
surface carries a period dividend and this does not.
"""

SESSION_MARK_OBSERVED_THROUGH_EVENT: Final = "OFFICIAL_CLOSE"
"""The exchange instant the value is complete at: this session's own close.

A *market fact*, and the only thing here that is one. When the value becomes
usable is a separate question with a separate owner -- the Feature engine's
installed source availability policy -- and the surface below resolves that
policy and seals its hash rather than restating the same string twice. Two equal
strings cannot distinguish a fact from a policy, which is what an earlier version
of this module did.

The value is complete the moment the closing price prints, and this repository's
installed source policy publishes the daily bar at that close. A consumer
deciding at ``close(T)`` may therefore use the mark for ``T`` itself -- which is
the whole point: a book valued at that open can be re-marked to that close by a
consumer who knows which entry session the book was filled at.

That projection belongs to the execution owner and is never inferred here. The
consumer resolves it from the execution snapshot's own ``entry_session`` column
and looks a mark up by that session's *label*; an array indexed beside the
formation axis would apply session T's move to a book filled at open(T+1) -- one
day early, the same class of error this whole remediation exists to remove.
"""


class SessionMarkError(ValueError):
    """Stable fail-closed boundary for the session mark fact."""


class SourceAvailabilityPolicy(Protocol):
    """The publishing schedule an owner asserts for the rows behind a surface.

    A protocol rather than an import. The owner of this schedule is the Feature
    engine's installed source-availability catalog, and that package already
    depends on this one -- so naming its type here would close a cycle the
    structural guard is right to reject.

    Structural is nevertheless stronger than what stood here before, which was
    two independent ``str`` parameters. An id and a hash typed separately can be
    combined freely, so a publisher could seal a real policy's name over a
    different policy's identity and nothing on the route would notice. They now
    travel as one object, and the consumer that admits the surface re-resolves
    that id at the real owner and refuses a hash the owner does not agree with.
    """

    @property
    def policy_id(self) -> str:
        """Return the installed source availability policy identifier."""
        ...

    @property
    def policy_hash(self) -> str:
        """Return the installed policy's content hash."""
        ...


def session_mark_matrix(
    bars: Iterable[ProjectedBar], *, sessions: Sequence[date], listing_ids: Sequence[str]
) -> tuple[tuple[float, ...], ...]:
    """Return open-to-close simple returns on a full ``(session, listing)`` axis.

    Keyed on both axes, because keying on the session alone is not safe: bars for
    two listings, or a duplicated row, would collapse into one entry and the last
    write would win silently. A mark is per listing per session or it is not a
    mark.

    Every cell is required. A missing one is refused rather than filled, because
    a substituted zero asserts that a listing did not move on a day nobody
    observed -- and a quiet day looks exactly the same. A return at or below -1
    is refused too: it would drive a re-marked weight to zero or negative, and no
    split-adjusted close can produce it.
    """
    if not sessions or not listing_ids:
        raise SessionMarkError("market_data_ops.session_mark_axis_empty")
    if len(set(sessions)) != len(sessions) or len(set(listing_ids)) != len(listing_ids):
        raise SessionMarkError("market_data_ops.session_mark_axis_duplicated")

    cells: dict[tuple[date, str], float] = {}
    for bar in bars:
        key = (bar.session_date, bar.listing_id)
        if key in cells:
            raise SessionMarkError("market_data_ops.session_mark_row_duplicated")
        if not bar.open_split_adjusted > 0.0 or not bar.close_split_adjusted > 0.0:
            raise SessionMarkError("market_data_ops.session_mark_price_invalid")
        mark = bar.close_split_adjusted / bar.open_split_adjusted - 1.0
        if not isfinite(mark) or mark <= -1.0:
            raise SessionMarkError("market_data_ops.session_mark_return_invalid")
        cells[key] = mark

    rows: list[tuple[float, ...]] = []
    for session in sessions:
        row: list[float] = []
        for listing in listing_ids:
            value = cells.get((session, listing))
            if value is None:
                raise SessionMarkError("market_data_ops.session_mark_absent")
            row.append(value)
        rows.append(tuple(row))
    return tuple(rows)


_SCHEMA_ID: Final = "session-open-close-mark"
SESSION_MARK_PRICE_BASIS: Final = "split_adjusted"


class SessionMarkEpoch(BaseModel):  # type: ignore[misc]
    """The axis a mark surface is sealed to. Strategy-neutral by construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["SessionMarkEpoch"] = "SessionMarkEpoch"
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    universe_manifest_revision: str = Field(min_length=1, max_length=160)
    market_profile_id: str = Field(min_length=1, max_length=96)
    epoch_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> SessionMarkEpoch:
        """Seal an epoch with its derived content hash."""
        return _seal(cls, "epoch_hash", **values)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_epoch(self) -> SessionMarkEpoch:
        """Require a sorted listing axis and matching epoch hash."""
        if self.ordered_listing_ids != tuple(sorted(set(self.ordered_listing_ids))):
            raise SessionMarkError("market_data_ops.session_mark_listing_axis_invalid")
        if self.epoch_hash != canonical_hash(self.model_dump(mode="json", exclude={"epoch_hash"})):
            raise SessionMarkError("market_data_ops.session_mark_epoch_identity_invalid")
        return self


class SessionMarkChunk(BaseModel):  # type: ignore[misc]
    """One published block of marks, described tightly enough to re-verify it.

    ``content_hash`` covers the *complete* row -- both prices and the corporate
    action set as well as the mark -- through a per-row identity carried in the
    file. An earlier version hashed only ``(session, listing, mark)`` while the
    file also stored the two prices and the action set, so three columns sat in a
    durable artifact with nothing protecting them; anything reading them for
    provenance was reading unauthenticated bytes. They are load-bearing -- the
    reader re-derives the mark from them, which is what makes a tampered mark
    value visible without a second source -- so they are covered rather than
    dropped.

    The counts are carried because a reader needs to know a block is *complete*
    before it can call an absent cell an absence rather than a shorter chunk.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["SessionMarkChunk"] = "SessionMarkChunk"
    first_session: date
    last_session: date
    session_count: int = Field(ge=1)
    listing_count: int = Field(ge=1)
    row_count: int = Field(ge=1)
    content_hash: str = Field(pattern=_HASH)
    metadata_hash: str = Field(pattern=_HASH)
    """The Parquet key-value metadata, hashed as one.

    Separate from ``content_hash`` because they fail differently: a rewritten
    file with the original metadata block is caught by the first, and a file
    whose metadata was re-stamped around unchanged rows is caught by the second.
    """

    uri: str = Field(min_length=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_chunk(self) -> SessionMarkChunk:
        """Require ordered sessions and a complete rectangular chunk."""
        if self.first_session > self.last_session:
            raise SessionMarkError("market_data_ops.session_mark_chunk_axis_invalid")
        if self.row_count != self.session_count * self.listing_count:
            # A rectangular block or it is not a block. A chunk whose rows do not
            # fill its own axes cannot say whether a missing cell was never
            # published or was removed.
            raise SessionMarkError("market_data_ops.session_mark_chunk_not_rectangular")
        return self


class SessionMarkSurface(BaseModel):  # type: ignore[misc]
    """One published intraday-mark surface, and everything it is sealed to.

    Carries what an earlier probe stated as two bare strings: the exchange
    instant it observes through, the *resolved* source availability policy with
    its owner's own hash, the price basis, the corporate-action identity and the
    watermark of the source rows it was projected from. A consumer binds this
    surface hash and can re-derive every one of them at its owner.

    No schedule, entry, exit or rebalance policy appears here, and none may: one
    mark surface serves a next-open strategy, a close-auction strategy and an
    intraday strategy without rotating.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["SessionMarkSurface"] = "SessionMarkSurface"
    epoch: SessionMarkEpoch
    chunks: tuple[SessionMarkChunk, ...] = Field(min_length=1)
    first_session: date
    last_session: date
    session_count: int = Field(ge=1)

    formula_identity: Literal[
        "close_split_adjusted / open_split_adjusted - 1, within one session"
    ] = SESSION_MARK_FORMULA_IDENTITY
    price_basis: Literal["split_adjusted"] = SESSION_MARK_PRICE_BASIS
    corporate_action_identity: str = Field(min_length=1, max_length=128)
    source_watermark_hash: str = Field(pattern=_HASH)

    observed_through_event: Literal["OFFICIAL_CLOSE"] = SESSION_MARK_OBSERVED_THROUGH_EVENT
    observed_through_offset_sessions: Literal[0] = 0
    """The mark for session ``T`` is complete at ``close(T)``. A market fact."""

    availability_policy_id: str = Field(min_length=1, max_length=96)
    availability_policy_hash: str = Field(pattern=_HASH)
    """The Feature owner's installed source policy, resolved and sealed.

    Its own hash, so a consumer checks the policy it was published under rather
    than trusting a repeated event name. This repository is explicit that these
    schedules are installed policy and not measurement, and binding the hash is
    what keeps that assertion attributable.
    """

    limitations: tuple[str, ...] = ()
    surface_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> SessionMarkSurface:
        """Seal a surface with its derived content hash."""
        return _seal(cls, "surface_hash", **values)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_surface(self) -> SessionMarkSurface:
        """Require coherent chunk coverage, distinct blocks and surface hash."""
        if self.first_session > self.last_session:
            raise SessionMarkError("market_data_ops.session_mark_session_axis_invalid")
        for left, right in zip(self.chunks, self.chunks[1:], strict=False):
            if left.last_session >= right.first_session:
                # Ordered *and* disjoint. Two chunks covering one session would
                # let a read take whichever it happened to visit last, which is
                # a silent choice between two answers to the same question.
                raise SessionMarkError("market_data_ops.session_mark_chunk_axis_unordered")
        if (
            self.chunks[0].first_session != self.first_session
            or self.chunks[-1].last_session != self.last_session
            or sum(value.session_count for value in self.chunks) != self.session_count
        ):
            # The manifest and its chunks must agree about what is covered. A
            # surface claiming a wider window than its blocks fill turns a
            # missing session into a lookup that finds nothing, and there is no
            # honest way to tell that apart from a listing that did not trade.
            raise SessionMarkError("market_data_ops.session_mark_chunk_coverage_incomplete")
        if len(set(value.content_hash for value in self.chunks)) != len(self.chunks):
            raise SessionMarkError("market_data_ops.session_mark_chunk_duplicated")
        if self.surface_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"surface_hash"})
        ):
            raise SessionMarkError("market_data_ops.session_mark_surface_identity_invalid")
        return self

    @property
    def manifest_uri(self) -> str:
        """The store-relative name this surface is published under.

        A canonical URI derived from the identity, never a filesystem path: a
        consumer binds this and a replay reopens exactly the surface the binding
        names instead of whatever the directory happens to hold.
        """
        return f"manifests/{self.surface_hash}.json"


def _seal[ContractT: BaseModel](
    model: type[ContractT], field: str, /, **values: object
) -> ContractT:
    unknown = sorted(set(values) - set(model.model_fields))
    if unknown:
        raise SessionMarkError(f"market_data_ops.session_mark_field_unknown:{unknown[0]}")
    draft = model.model_construct(**values, **{field: "0" * 64})
    identity = draft.model_dump(mode="json", exclude={field})
    payload = draft.model_dump(exclude={field})
    return cast(ContractT, model.model_validate({**payload, field: canonical_hash(identity)}))


_ROW_IDENTITY_COLUMNS: Final = (
    "session_date",
    "listing_id",
    "intraday_simple_return",
    "open_split_adjusted",
    "close_split_adjusted",
    "action_set_hash",
)
"""Every column a row carries. The identity covers all of them, not three."""


def _mark_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("session_date", pa.date32(), nullable=False),
            pa.field("listing_id", pa.string(), nullable=False),
            pa.field("intraday_simple_return", pa.float64(), nullable=False),
            pa.field("open_split_adjusted", pa.float64(), nullable=False),
            pa.field("close_split_adjusted", pa.float64(), nullable=False),
            pa.field("action_set_hash", pa.string(), nullable=False),
            pa.field("row_hash", pa.string(), nullable=False),
        ]
    )


def _row_identities(ordered: pa.Table) -> list[str]:
    """Each row's identity -- the canonical hash of its identity columns -- from the columns."""
    return canonical_row_hashes(ordered.select(list(_ROW_IDENTITY_COLUMNS)))


def _chunk_identity(ordered: pa.Table, identities: Sequence[str]) -> str:
    """Hash the chunk's ``(session, listing, row identity)`` triples."""
    sessions = [str(value) for value in ordered.column("session_date").to_pylist()]
    listings = [str(value) for value in ordered.column("listing_id").to_pylist()]
    return str(
        canonical_hash(
            {
                "schema": _SCHEMA_ID,
                "rows": tuple(zip(sessions, listings, identities, strict=True)),
            }
        )
    )


def _metadata_identity(metadata: Mapping[bytes, bytes]) -> str:
    serialized = json.dumps(
        {key.decode(): value.decode() for key, value in sorted(metadata.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


class SessionMarkArtifactStore:
    """Durable home for published mark chunks and manifests.

    Reads verify. Every invariant below was checked at publication and none of
    them was checked again on the way out, which made the whole store worth
    exactly as much as the filesystem it sat on: a rewritten Parquet file, a
    permuted listing axis or a deleted row would all have been read back as
    facts. The Risk return store had settled these invariants already, so this
    follows that shape rather than inventing a second one.
    """

    def __init__(
        self, artifact_root: Path, *, capacity: Callable[[int], None] = lambda _bytes: None
    ) -> None:
        """Place verified mark artifacts below the workspace artifact root."""
        self.root = artifact_root.resolve() / "data-operations" / "session-marks"
        self._capacity = capacity

    def _chunk_path(self, content_hash: str) -> Path:
        return self.root / "chunks" / f"{content_hash}.parquet"

    def _manifest_path(self, surface_hash: str) -> Path:
        return self.root / "manifests" / f"{surface_hash}.json"

    def _publish_bytes(self, target: Path, content: bytes) -> None:
        if target.exists():
            if target.read_bytes() != content:
                raise SessionMarkError("market_data_ops.session_mark_identity_reused")
            return
        self._capacity(len(content))
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        staged.write_bytes(content)
        os.replace(staged, target)

    @staticmethod
    def chunk_uri(content_hash: str) -> str:
        """Return the canonical relative URI for a chunk hash."""
        return f"chunks/{content_hash}.parquet"

    @staticmethod
    def manifest_uri(surface_hash: str) -> str:
        """Return the canonical relative URI for a surface manifest."""
        return f"manifests/{surface_hash}.json"

    def publish_chunk(self, table: pa.Table) -> SessionMarkChunk:
        """Seal a sorted mark block, publish it and verify its stored bytes."""
        ordered = table.combine_chunks().sort_by(
            [("session_date", "ascending"), ("listing_id", "ascending")]
        )
        if ordered.num_rows == 0:
            raise SessionMarkError("market_data_ops.session_mark_chunk_empty")
        identities = _row_identities(ordered)
        sessions = tuple(dict.fromkeys(ordered.column("session_date").to_pylist()))
        listings = tuple(dict.fromkeys(ordered.column("listing_id").to_pylist()))
        content_hash = _chunk_identity(ordered, identities)
        target = self._chunk_path(content_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            b"alphalattice.snapshot_kind": b"SessionMarkChunk",
            b"alphalattice.schema_id": _SCHEMA_ID.encode(),
            b"alphalattice.chunk_hash": content_hash.encode(),
        }
        if not target.exists():
            sealed = pa.Table.from_arrays(
                [
                    *(ordered.column(name) for name in _ROW_IDENTITY_COLUMNS),
                    pa.array(identities, type=pa.string()),
                ],
                # Built against the declared schema rather than by appending a
                # column: an appended one is nullable, and a file written with a
                # nullable column would be refused by the exact-schema check the
                # reader runs -- which is the check, so it must not be relaxed.
                schema=_mark_schema(),
            )
            buffer = pa.BufferOutputStream()
            pq.write_table(sealed.replace_schema_metadata(metadata), buffer, compression="zstd")
            content = buffer.getvalue().to_pybytes()
            self._publish_bytes(target, content)
        chunk = SessionMarkChunk(
            first_session=sessions[0],
            last_session=sessions[-1],
            session_count=len(sessions),
            listing_count=len(listings),
            row_count=ordered.num_rows,
            content_hash=content_hash,
            metadata_hash=_metadata_identity(metadata),
            uri=self.chunk_uri(content_hash),
        )
        # Published, then read back through the same gate a consumer uses. A
        # publisher that certified its own in-memory table would leave the one
        # step that can actually fail -- the write -- unchecked.
        self.resolve_chunk(chunk)
        return chunk

    def resolve_chunk(self, chunk: SessionMarkChunk) -> Path:
        """Re-verify one durable block completely, and return where it lives.

        The whole of what the store promises, in one place: the URI is canonical
        for the identity, the schema is exact, the metadata block hashes to what
        the manifest recorded, every row re-derives its own identity, the rows
        collectively re-derive the chunk identity, the axes are the declared
        ones, and every mark is still the ratio of the two prices beside it.

        That last check is why the price columns are worth keeping. A forger who
        edits a mark in place has to edit the prices to match, and editing the
        prices moves the row identity -- so there is no single-field change to
        this file that a reader accepts.
        """
        if chunk.uri != self.chunk_uri(chunk.content_hash):
            raise SessionMarkError("market_data_ops.session_mark_chunk_uri_invalid")
        target = self._chunk_path(chunk.content_hash)
        try:
            table = pq.read_table(target, columns=_mark_schema().names).combine_chunks()
        except Exception as error:
            raise SessionMarkError("market_data_ops.session_mark_chunk_tampered") from error
        if table.schema.remove_metadata() != _mark_schema():
            raise SessionMarkError("market_data_ops.session_mark_chunk_schema_invalid")
        metadata = table.schema.metadata or {}
        if metadata.get(b"alphalattice.chunk_hash") != chunk.content_hash.encode():
            raise SessionMarkError("market_data_ops.session_mark_chunk_tampered")
        if _metadata_identity(metadata) != chunk.metadata_hash:
            raise SessionMarkError("market_data_ops.session_mark_chunk_metadata_tampered")
        ordered = table.sort_by([("session_date", "ascending"), ("listing_id", "ascending")])
        if ordered.num_rows != chunk.row_count:
            raise SessionMarkError("market_data_ops.session_mark_chunk_row_count_invalid")
        # The same checks as before, each over a column rather than a row at a
        # time: every row re-derives its identity, no (session, listing) is
        # repeated, both prices are positive, and every mark is still the
        # ratio of the two prices beside it -- the division and subtraction
        # are the same two IEEE operations whether the engine or Python runs
        # them, so the equality is exact.
        row_hashes = ordered.column("row_hash").to_pylist()
        if _row_identities(ordered) != row_hashes:
            raise SessionMarkError("market_data_ops.session_mark_chunk_tampered")
        session_column = ordered.column("session_date").to_pylist()
        listing_column = ordered.column("listing_id").to_pylist()
        if len(dict.fromkeys(zip(session_column, listing_column, strict=True))) != len(row_hashes):
            raise SessionMarkError("market_data_ops.session_mark_row_duplicated")
        opened = ordered.column("open_split_adjusted")
        closed = ordered.column("close_split_adjusted")
        if not (
            pc.all(pc.greater(opened, 0.0)).as_py() and pc.all(pc.greater(closed, 0.0)).as_py()
        ):
            raise SessionMarkError("market_data_ops.session_mark_price_invalid")
        if not pc.all(
            pc.equal(
                pc.subtract(pc.divide(closed, opened), 1.0),
                ordered.column("intraday_simple_return"),
            )
        ).as_py():
            raise SessionMarkError("market_data_ops.session_mark_row_not_the_formula")
        sessions = tuple(dict.fromkeys(cast(list[date], session_column)))
        listings = tuple(dict.fromkeys(str(value) for value in listing_column))
        if (
            _chunk_identity(ordered, row_hashes) != chunk.content_hash
            or sessions[0] != chunk.first_session
            or sessions[-1] != chunk.last_session
            or len(sessions) != chunk.session_count
            or len(listings) != chunk.listing_count
        ):
            raise SessionMarkError("market_data_ops.session_mark_chunk_tampered")
        return target

    def publish_manifest(self, surface: SessionMarkSurface) -> str:
        """Publish a sealed surface manifest under its content hash."""
        target = self._manifest_path(surface.surface_hash)
        self._publish_bytes(target, surface.model_dump_json(indent=1).encode("utf-8"))
        return surface.manifest_uri

    def load_manifest(self, surface_hash: str) -> SessionMarkSurface:
        """Load the manifest whose sealed hash matches the requested handle."""
        target = self._manifest_path(surface_hash)
        if not target.is_file():
            raise SessionMarkError("market_data_ops.session_mark_surface_absent")
        surface = cast(
            SessionMarkSurface, SessionMarkSurface.model_validate_json(target.read_bytes())
        )
        if surface.surface_hash != surface_hash:
            # The file answering to one name is sealed under another. Its own
            # validator only proves it is internally consistent, which a forger
            # re-sealing a whole surface satisfies for free.
            raise SessionMarkError("market_data_ops.session_mark_surface_not_this_handle")
        return surface

    def read_marks(
        self,
        surface: SessionMarkSurface,
        *,
        sessions: Sequence[date],
        listing_ids: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """Read marks for a requested axis, refusing anything the surface lacks.

        Positional by construction: the caller states both axes and gets rows in
        exactly that order, so a consumer cannot silently receive a permuted
        listing axis and multiply the wrong weight by the wrong move.

        Every block touched is re-verified through ``resolve_chunk`` first. This
        used to open the Parquet files directly, which meant the durable store
        checked its own bytes exactly once, at publication, and every read after
        that trusted the disk.
        """
        if not sessions or not listing_ids:
            raise SessionMarkError("market_data_ops.session_mark_axis_empty")
        if tuple(sessions) != tuple(sorted(set(sessions))):
            raise SessionMarkError("market_data_ops.session_mark_read_axis_invalid")
        if len(set(listing_ids)) != len(listing_ids):
            raise SessionMarkError("market_data_ops.session_mark_axis_duplicated")
        if not set(listing_ids) <= set(surface.epoch.ordered_listing_ids):
            # A listing this surface was never sealed to. Refused rather than
            # reported absent: the epoch is the axis the marks were published
            # against, and a name outside it is a different universe.
            raise SessionMarkError("market_data_ops.session_mark_listing_not_in_epoch")
        if sessions[0] < surface.first_session or sessions[-1] > surface.last_session:
            raise SessionMarkError("market_data_ops.session_mark_read_outside_surface")
        # Every cell of the requested grid is placed by the position of its
        # session and listing on the requested axes, from the columns of each
        # verified block; a cell no block filled is refused, and a cell two
        # blocks filled is refused as before -- unreachable while the surface
        # validator holds (it refuses overlapping chunk ranges, and
        # ``resolve_chunk`` refuses a repeated ``(session, listing)`` inside
        # one block), kept as the second layer.
        session_positions = {session: index for index, session in enumerate(sessions)}
        listing_positions = {listing: index for index, listing in enumerate(listing_ids)}
        grid: npt.NDArray[np.float64] = np.full(
            (len(sessions), len(listing_ids)), np.nan, dtype=np.float64
        )
        filled: npt.NDArray[np.bool_] = np.zeros(grid.shape, dtype=np.bool_)
        for chunk in surface.chunks:
            if chunk.last_session < sessions[0] or chunk.first_session > sessions[-1]:
                continue
            table = pq.read_table(
                self.resolve_chunk(chunk),
                columns=["session_date", "listing_id", "intraday_simple_return"],
            )
            row_positions: npt.NDArray[np.int64] = np.fromiter(
                (
                    session_positions.get(session, -1)
                    for session in table.column("session_date").to_pylist()
                ),
                dtype=np.int64,
                count=table.num_rows,
            )
            column_positions: npt.NDArray[np.int64] = np.fromiter(
                (
                    listing_positions.get(listing, -1)
                    for listing in table.column("listing_id").to_pylist()
                ),
                dtype=np.int64,
                count=table.num_rows,
            )
            selected = (row_positions >= 0) & (column_positions >= 0)
            rows, columns = row_positions[selected], column_positions[selected]
            if filled[rows, columns].any():
                raise SessionMarkError("market_data_ops.session_mark_read_duplicate")
            grid[rows, columns] = table.column("intraday_simple_return").to_numpy(
                zero_copy_only=False
            )[selected]
            filled[rows, columns] = True
        if not filled.all():
            raise SessionMarkError("market_data_ops.session_mark_absent")
        return tuple(tuple(float(value) for value in row) for row in grid)


def publish_market_session_marks(
    *,
    market: MarketDataRepository,
    store: SessionMarkArtifactStore,
    manifest: UniverseManifest,
    listing_ids: tuple[str, ...],
    sessions: tuple[date, ...],
    source_watermark_hash: str,
    corporate_action_identity: str,
    availability: SourceAvailabilityPolicy,
) -> SessionMarkSurface:
    """Publish the local-bar projection shared by research and Campaign consumers."""
    bars_by_listing: dict[str, tuple[ProjectedBar, ...]] = {}
    with market._connect(read_only=True) as connection:
        for listing in listing_ids:
            raw = market.raw_bars(
                listing, start=sessions[0], through=sessions[-1], _connection=connection
            )
            if not raw:
                raise SessionMarkError("market_data_ops.session_mark_source_absent")
            actions = tuple(
                v
                for v in market.actions(listing, _connection=connection)
                if raw[0].session_date <= v.effective_date <= raw[-1].session_date
            )
            projected = project_research_series(
                raw, actions, daily_price_basis=manifest.profile.daily_price_basis
            )
            by_session = {v.session_date: v for v in projected}
            try:
                bars_by_listing[listing] = tuple(by_session[s] for s in sessions)
            except KeyError as error:
                raise SessionMarkError("market_data_ops.session_mark_source_absent") from error
    return publish_session_mark_surface(
        store=store,
        bars_by_listing=bars_by_listing,
        sessions=sessions,
        universe_manifest_revision=manifest.revision_sha256,
        market_profile_id=manifest.profile.market_profile_id,
        corporate_action_identity=corporate_action_identity,
        source_watermark_hash=source_watermark_hash,
        availability=availability,
    )


def publish_session_mark_surface(
    *,
    store: SessionMarkArtifactStore,
    bars_by_listing: Mapping[str, Sequence[ProjectedBar]],
    sessions: Sequence[date],
    universe_manifest_revision: str,
    market_profile_id: str,
    corporate_action_identity: str,
    source_watermark_hash: str,
    availability: SourceAvailabilityPolicy,
) -> SessionMarkSurface:
    """Publish one mark surface, with its availability resolved at its owner.

    ``availability`` is the owner's own policy object, passed in rather than
    looked up: the Feature engine already depends on this package, so reaching
    into its catalog from here would close a dependency cycle -- the structural
    guard says so, and it is right. Whoever publishes a surface resolves the
    policy at its owner and hands the object across, and the consumer that
    admits the surface re-resolves the same id there and refuses a hash the
    owner does not agree with.
    """
    listing_ids = tuple(sorted(bars_by_listing))
    ordered_sessions = tuple(sorted(set(sessions)))
    if not listing_ids or not ordered_sessions:
        raise SessionMarkError("market_data_ops.session_mark_axis_empty")

    # Listing-major, one column at a time: ``publish_chunk`` sorts the rows on
    # (session, listing) itself, and a dict per cell was most of what
    # assembling the block cost.
    columns: dict[str, list[object]] = {name: [] for name in _ROW_IDENTITY_COLUMNS}
    for listing in listing_ids:
        bars = tuple(bars_by_listing[listing])
        marks = session_mark_matrix(bars, sessions=ordered_sessions, listing_ids=(listing,))
        by_session = {value.session_date: value for value in bars}
        for index, session in enumerate(ordered_sessions):
            bar = by_session[session]
            columns["session_date"].append(session)
            columns["listing_id"].append(listing)
            columns["intraday_simple_return"].append(marks[index][0])
            columns["open_split_adjusted"].append(bar.open_split_adjusted)
            columns["close_split_adjusted"].append(bar.close_split_adjusted)
            columns["action_set_hash"].append(bar.action_set_hash)
    fields = [_mark_schema().field(name) for name in _ROW_IDENTITY_COLUMNS]
    chunk = store.publish_chunk(
        pa.Table.from_arrays(
            [pa.array(columns[field.name], type=field.type) for field in fields],
            schema=pa.schema(fields),
        )
    )
    surface = SessionMarkSurface.create(
        epoch=SessionMarkEpoch.create(
            ordered_listing_ids=listing_ids,
            universe_manifest_revision=universe_manifest_revision,
            market_profile_id=market_profile_id,
        ),
        chunks=(chunk,),
        first_session=ordered_sessions[0],
        last_session=ordered_sessions[-1],
        session_count=len(ordered_sessions),
        corporate_action_identity=corporate_action_identity,
        source_watermark_hash=source_watermark_hash,
        availability_policy_id=availability.policy_id,
        availability_policy_hash=availability.policy_hash,
    )
    store.publish_manifest(surface)
    return surface


__all__ = [
    "SESSION_MARK_FORMULA_IDENTITY",
    "SESSION_MARK_OBSERVED_THROUGH_EVENT",
    "SESSION_MARK_PRICE_BASIS",
    "SessionMarkArtifactStore",
    "SessionMarkChunk",
    "SessionMarkEpoch",
    "SessionMarkError",
    "SessionMarkSurface",
    "SourceAvailabilityPolicy",
    "publish_market_session_marks",
    "publish_session_mark_surface",
    "session_mark_matrix",
]
