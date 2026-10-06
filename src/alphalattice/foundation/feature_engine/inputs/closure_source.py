"""Read-only DuckDB adapter for Feature closure verification."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from alphalattice.control.workspace_runtime.database import open_workspace_database

_CLOSURE_KEY_RELATION = "feature_closure_keys"
_DIGEST_PREFIX = b"FeatureBaseClosureRowHashDigest\0"
_DIGEST_BATCH_ROWS = 131_072
"""Rows framed per step of the streaming digest: about 16 MB of framed bytes at a time."""


def _row_hash_digest(rows: Sequence[tuple[date, str, str]]) -> str:
    """Hash ordered ``(session, listing_id, row_hash)`` closure rows.

    Per row: the session's ISO date, the listing id as a length-prefixed
    UTF-8 string (big-endian 32-bit length), the row hash; then the row count
    as a big-endian 64-bit integer. ``_framed_rows`` produces the same bytes
    from an Arrow batch; both must stay identical to every stored digest.
    """
    digest = hashlib.sha256(_DIGEST_PREFIX)
    for session, listing_id, row_hash in rows:
        listing = str(listing_id).encode("utf-8")
        digest.update(session.isoformat().encode("ascii"))
        digest.update(struct.pack(">I", len(listing)))
        digest.update(listing)
        digest.update(str(row_hash).encode("ascii"))
    digest.update(struct.pack(">Q", len(rows)))
    return digest.hexdigest()


@dataclass(eq=False, frozen=True)
class FeatureRowHashDigestState:
    """A closure digest and the frame state before its open sessions.

    ``boundary`` is the first session of the closure's open tail: every row
    with an earlier session was framed into ``prefix`` (the running hash
    before the tail), ``prefix_rows`` of them. A caller that only ever adds
    or changes rows from ``boundary`` on can re-seal the digest by framing
    that tail again from the store, not the whole closure.
    """

    digest: str
    rows: int
    boundary: date | None
    prefix: hashlib._Hash
    prefix_rows: int


def _sealed(frames: hashlib._Hash, rows: int) -> str:
    sealed = frames.copy()
    sealed.update(struct.pack(">Q", rows))
    return sealed.hexdigest()


def _split_at_boundary(batch: pa.RecordBatch, boundary: date) -> int:
    """Find the first batch row at or after the boundary session."""
    sessions = batch.column(0).to_numpy(zero_copy_only=False)
    return int(np.searchsorted(sessions, np.datetime64(boundary, "D"), side="left"))


def _framed_rows(batch: pa.RecordBatch) -> memoryview:
    """Frame one ordered batch column-wise for the closure digest."""
    session, listing, row_hash = (batch.column(index) for index in range(3))
    if session.null_count or listing.null_count or row_hash.null_count:
        raise ValueError("feature_closure.row_hash_digest_input_invalid")
    # The old date/ASCII conversions refused out-of-domain dates and non-ASCII
    # row hashes. Vectorized valid-row framing must not admit those corrupt rows.
    pc.min_max(session).as_py()
    if not pc.all(pc.string_is_ascii(row_hash)).as_py():
        for value in row_hash.to_pylist():
            value.encode("ascii")
    lengths = pc.binary_length(listing).to_numpy(zero_copy_only=False).astype(">u4")
    prefixes = pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(4), batch.num_rows, [None, pa.py_buffer(lengths.tobytes())]
    )
    joined = pc.binary_join_element_wise(
        pc.cast(session, pa.string()).cast(pa.binary()),
        prefixes.cast(pa.binary()),
        listing.cast(pa.binary()),
        row_hash.cast(pa.binary()),
        b"",
    )
    offsets: np.ndarray = np.frombuffer(joined.buffers()[1], dtype=np.int32)
    first, last = int(offsets[0]), int(offsets[batch.num_rows])
    return memoryview(joined.buffers()[2])[first:last]


def empty_feature_row_hash_digest() -> str:
    """Return the digest of a closure with no rows.

    ``feature_row_hash_digest`` fails closed on an empty store because an
    admitted root always describes rows that already exist. A genesis closure is
    empty by definition, so it needs the same construction evaluated over zero
    rows rather than a relaxed guard on the admission path.
    """
    return _row_hash_digest(())


@contextmanager
def _staged_closure_keys(
    connection: duckdb.DuckDBPyConnection, keys: Sequence[tuple[date, str]]
) -> Iterator[str]:
    """Expose one batch's key relation to DuckDB under a scoped name.

    The keys arrive as a single bounded batch and every reader consumes them
    through an ``ORDER BY`` on the join, so the relation never needs a table of
    its own -- it is registered as an Arrow view for the length of the query.
    Inserting the 10,060 keys a four-listing batch carries one row at a time
    costs 793 ms against 0.23 ms here, and the closure reads this twice per
    batch: three minutes of a full base-closure rebuild spent staging keys.
    """
    staged = pa.table(
        {
            "session_date": pa.array([session for session, _ in keys], pa.date32()),
            "listing_id": pa.array([listing_id for _, listing_id in keys], pa.string()),
        }
    )
    connection.register(_CLOSURE_KEY_RELATION, staged)
    try:
        yield _CLOSURE_KEY_RELATION
    finally:
        connection.unregister(_CLOSURE_KEY_RELATION)


class FeatureClosureSourceRepository:
    """Read exact persisted identities without expanding ``MarketDataRepository``."""

    def __init__(self, database_path: Path) -> None:
        """Open a read-only source over the workspace database."""
        self._database_path = database_path.resolve()

    def feature_row_hash_digest(
        self,
        *,
        catalog_hash: str,
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> str:
        """Return the verified digest of current Feature row hashes."""
        return self.feature_row_hash_digest_state(
            catalog_hash=catalog_hash, connection=connection
        ).digest

    def feature_row_hash_digest_state(
        self,
        *,
        catalog_hash: str,
        boundary: date | None = None,
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> FeatureRowHashDigestState:
        """Return the closure digest and frame state before ``boundary``.

        Streamed in ordered Arrow batches and framed column-wise: the same
        bytes as ``_row_hash_digest`` without a Python object per row or a
        whole-closure buffer. With a ``boundary`` session the running hash
        is copied just before the first row of that session, so a caller that
        keeps writing rows from that session on can re-seal the digest with
        ``reseal_feature_row_hash_digest`` instead of streaming the closure.
        """
        frames = hashlib.sha256(_DIGEST_PREFIX)
        rows = 0
        prefix = frames.copy()
        prefix_rows = 0
        before_boundary = boundary is not None
        with self._connection(connection) as active:
            reader = active.execute(
                """
                SELECT session_date, listing_id, row_hash
                FROM feature_daily_runtime
                WHERE catalog_hash = ?
                ORDER BY session_date, listing_id
                """,
                [catalog_hash],
            ).to_arrow_reader(_DIGEST_BATCH_ROWS)
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                if before_boundary:
                    assert boundary is not None
                    split = _split_at_boundary(batch, boundary)
                    if split < batch.num_rows:
                        if split:
                            head = batch.slice(0, split)
                            frames.update(_framed_rows(head))
                            rows += head.num_rows
                            batch = batch.slice(split)
                        prefix = frames.copy()
                        prefix_rows = rows
                        before_boundary = False
                rows += batch.num_rows
                frames.update(_framed_rows(batch))
        if before_boundary:
            # Every row precedes the boundary: the tail is empty.
            prefix = frames.copy()
            prefix_rows = rows
        if not rows:
            raise ValueError("feature_closure.root_unavailable")
        return FeatureRowHashDigestState(
            digest=_sealed(frames, rows),
            rows=rows,
            boundary=boundary,
            prefix=prefix,
            prefix_rows=prefix_rows,
        )

    def reseal_feature_row_hash_digest(
        self,
        state: FeatureRowHashDigestState,
        *,
        catalog_hash: str,
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> FeatureRowHashDigestState | None:
        """Re-seal a digest state by framing its open tail again from the store.

        The digest is the running hash of the ordered row stream with the row
        count last, so a closure whose rows before ``state.boundary`` are
        unchanged seals to exactly the value re-streaming the whole closure
        would: the remembered prefix, then the rows from the boundary on in
        order. The caller proves every write since the state was taken lies
        on or after the boundary; the store re-proves that the prefix still
        holds ``prefix_rows`` rows. Anything else answers None and the caller
        streams the whole closure.
        """
        if state.boundary is None:
            return None
        frames = state.prefix.copy()
        rows = state.prefix_rows
        with self._connection(connection) as active:
            before = active.execute(
                """
                SELECT count(*) FROM feature_daily_runtime
                WHERE catalog_hash = ? AND session_date < ?
                """,
                [catalog_hash, state.boundary],
            ).fetchone()
            if before is None or int(before[0]) != state.prefix_rows:
                return None
            reader = active.execute(
                """
                SELECT session_date, listing_id, row_hash
                FROM feature_daily_runtime
                WHERE catalog_hash = ? AND session_date >= ?
                ORDER BY session_date, listing_id
                """,
                [catalog_hash, state.boundary],
            ).to_arrow_reader(_DIGEST_BATCH_ROWS)
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                rows += batch.num_rows
                frames.update(_framed_rows(batch))
        if not rows:
            return None
        return FeatureRowHashDigestState(
            digest=_sealed(frames, rows),
            rows=rows,
            boundary=state.boundary,
            prefix=state.prefix,
            prefix_rows=state.prefix_rows,
        )

    def feature_row_count(
        self,
        *,
        catalog_hash: str,
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> int:
        """Count persisted rows so a caller can distinguish empty from missing.

        Feature current storage is installed by the first build, so a workspace
        that has never built has no table at all. That is legitimately zero rows,
        and it is checked explicitly rather than by catching a catalog error,
        which would also swallow a genuinely damaged store.
        """
        with self._connection(connection) as active:
            installed = active.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'feature_daily_runtime'
                """
            ).fetchone()
            if installed is None:
                return 0
            result = active.execute(
                "SELECT COUNT(*) FROM feature_daily_runtime WHERE catalog_hash = ?",
                [catalog_hash],
            ).fetchone()
        return int(result[0]) if result else 0

    def materialized_listing_ids(
        self,
        *,
        catalog_hash: str,
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> tuple[str, ...]:
        """Which listings this catalog actually holds live rows for.

        Completeness is a membership question, and a row count cannot answer it:
        one listing of 466 is a positive count and an unusable closure. The build
        commits one listing's whole history in a single write, so presence per
        listing is the unit the closure actually advances in, and comparing that
        set with the expected membership is the real durable statement.
        """
        with self._connection(connection) as active:
            installed = active.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'feature_daily_runtime'
                """
            ).fetchone()
            if installed is None:
                return ()
            rows = active.execute(
                """
                SELECT DISTINCT listing_id FROM feature_daily_runtime
                WHERE catalog_hash = ?
                ORDER BY listing_id
                """,
                [catalog_hash],
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def read_rows(
        self,
        *,
        catalog_hash: str,
        factor_ids: Sequence[str],
        keys: Sequence[tuple[date, str]],
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> dict[tuple[date, str], dict[str, object]]:
        """Read bounded Feature rows for closure verification.

        A batch none of whose keys the current table holds -- each transition of a first
        build -- is answered from the table's key columns. The runtime view joins the
        table's rows to their cutoff sets, so a key the table does not hold is not in the
        view either; reading the view's factor columns to learn that cost a transition
        45 ms at half a first build, against 7 ms here.
        """
        bounded = tuple(keys)
        if not bounded:
            return {}
        with self._connection(connection) as active, _staged_closure_keys(active, bounded):
            held = active.execute(
                """
                SELECT 1
                FROM feature_closure_keys AS k
                JOIN feature_daily_current AS f
                  ON f.catalog_hash = ?
                 AND f.session_date = k.session_date
                 AND f.listing_id = k.listing_id
                LIMIT 1
                """,
                [catalog_hash],
            ).fetchone()
            if held is None:
                return {}
            projection = ", ".join(f'f."{factor_id}"' for factor_id in factor_ids)
            rows = active.execute(
                f"""
                SELECT f.session_date, f.listing_id, f.input_cutoffs_json, f.row_hash,
                       f.raw_input_hash, f.action_set_hash, f.market_reference_revision,
                       {projection}
                FROM feature_closure_keys AS k
                LEFT JOIN feature_daily_runtime AS f
                  ON f.catalog_hash = ?
                 AND f.session_date = k.session_date
                 AND f.listing_id = k.listing_id
                ORDER BY k.session_date, k.listing_id
                """,
                [catalog_hash],
            ).fetchall()
        result: dict[tuple[date, str], dict[str, object]] = {}
        for row in rows:
            if row[0] is None:
                continue
            key = (row[0], str(row[1]))
            result[key] = {
                "input_cutoffs_json": row[2],
                "row_hash": row[3],
                "raw_input_hash": row[4],
                "action_set_hash": row[5],
                "market_reference_revision": row[6],
                **dict(zip(factor_ids, row[7:], strict=True)),
            }
        return result

    def read_row_hashes(
        self,
        *,
        catalog_hash: str,
        keys: Sequence[tuple[date, str]],
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> dict[tuple[date, str], str]:
        """Read only the persisted row identity for a bounded key batch.

        A transition disposition compares row hashes and nothing else.  Reading
        it through :meth:`read_rows` also materialises every factor column of
        every key as Python objects -- 533,180 values per four-listing batch to
        reach 10,060 hashes.  This asks for what the comparison uses.
        """
        bounded = tuple(keys)
        if not bounded:
            return {}
        with self._connection(connection) as active, _staged_closure_keys(active, bounded):
            rows = active.execute(
                """
                SELECT f.session_date, f.listing_id, f.row_hash
                FROM feature_closure_keys AS k
                LEFT JOIN feature_daily_runtime AS f
                  ON f.catalog_hash = ?
                 AND f.session_date = k.session_date
                 AND f.listing_id = k.listing_id
                ORDER BY k.session_date, k.listing_id
                """,
                [catalog_hash],
            ).fetchall()
        return {(row[0], str(row[1])): str(row[2]) for row in rows if row[0] is not None}

    def receipt_hashes(
        self,
        expected: Sequence[str],
        *,
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> tuple[str, ...]:
        """Return materialization receipt hashes for the requested listings."""
        bounded = tuple(expected)
        if not bounded:
            return ()
        placeholders = ", ".join("?" for _ in bounded)
        with self._connection(connection) as active:
            rows = active.execute(
                f"""
                SELECT receipt_hash
                FROM feature_materialization_receipt
                WHERE receipt_hash IN ({placeholders}) AND work_status = 'COMPLETED'
                ORDER BY receipt_hash
                """,
                list(bounded),
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def revision_ids(
        self,
        *,
        receipt_hashes: Sequence[str],
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> tuple[str, ...]:
        """Return revision identities for the requested Feature rows."""
        bounded = tuple(receipt_hashes)
        if not bounded:
            return ()
        placeholders = ", ".join("?" for _ in bounded)
        with self._connection(connection) as active:
            rows = active.execute(
                f"""
                SELECT revision_id
                FROM feature_daily_revision
                WHERE materialization_receipt_hash IN ({placeholders})
                ORDER BY revision_id
                """,
                list(bounded),
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    @contextmanager
    def _connection(
        self, supplied: duckdb.DuckDBPyConnection | None
    ) -> Iterator[duckdb.DuckDBPyConnection]:
        if supplied is not None:
            yield supplied
            return
        connection = open_workspace_database(self._database_path, read_only=True)
        try:
            yield connection
        finally:
            connection.close()


__all__ = ["FeatureClosureSourceRepository"]
