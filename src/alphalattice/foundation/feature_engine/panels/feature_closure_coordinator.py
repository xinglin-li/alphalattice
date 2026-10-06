"""Prepare immutable Feature recovery evidence around the existing Store write."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime
from operator import itemgetter
from typing import cast

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from alphalattice.foundation.feature_engine.catalog.layer import FeatureCatalogLayer
from alphalattice.foundation.feature_engine.inputs.closure_source import (
    FeatureClosureSourceRepository,
    FeatureRowHashDigestState,
    empty_feature_row_hash_digest,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_contracts import (
    FeatureBaseClosureTransitionReceipt,
    FeatureClosureFirstLoad,
    PreparedFeatureBaseClosureTransition,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    FeatureClosureLedger,
    identified,
)
from alphalattice.foundation.feature_engine.panels.materialization_identity import (
    feature_materialization_coverage,
    feature_materialization_receipt_hash,
    feature_row_content_hash_from_canonical,
    feature_row_is_numerically_identical,
)
from alphalattice.foundation.feature_engine.publication.current_storage import (
    canonical_cutoff_set,
)
from alphalattice.foundation.feature_engine.publication.persistence import (
    FeatureClosureDisposition,
    admitted_writes,
)
from alphalattice.foundation.feature_engine.storage.contracts import (
    FeatureMaterializationWrite,
    FeatureRowIdentity,
)
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_Candidate = tuple[date, str, str, str, int, int]
"""One row of a transition: its session and listing (its key), its serialized cutoffs, its row
hash, and where its cells are -- its write's position in the batch and its position in that
write's frame."""


class FeatureBaseClosureCoordinator:
    """Make every production Feature mutation recoverable before it commits."""

    def __init__(
        self,
        *,
        store: FeatureStateRepository,
        source: FeatureClosureSourceRepository,
        ledger: FeatureClosureLedger,
        factor_ids: Sequence[str],
    ) -> None:
        """Bind the Feature writer, closure source, ledger, and factor axis."""
        self._store = store
        self._source = source
        self._ledger = ledger
        self._factor_ids = tuple(factor_ids)
        if not self._factor_ids or len(self._factor_ids) != len(set(self._factor_ids)):
            raise ValueError("Feature closure factor axis is invalid")
        # Per catalog: the head this coordinator sealed last and the digest
        # state it sealed, so a daily append extends it (see ``_digest_after``).
        self._digest_states: dict[str, tuple[str, FeatureRowHashDigestState]] = {}
        # Per catalog: the first load this coordinator began and has not completed (V92).
        self._first_loads: dict[str, str] = {}

    def persist_batch(
        self,
        writes: Sequence[FeatureMaterializationWrite],
        *,
        connection: duckdb.DuckDBPyConnection,
        timing_sink: Callable[[str, float], None] | None = None,
    ) -> tuple[str, ...]:
        """Persist a Feature batch with recoverable closure receipts."""
        bounded = tuple(
            sorted(
                writes,
                key=lambda write: (
                    write.catalog_hash,
                    write.listing_id,
                    write.idempotency_key,
                ),
            )
        )
        if not bounded:
            raise ValueError("Feature closure persistence batch is empty")
        if len(bounded) > admitted_writes(len(self._factor_ids)):
            raise ValueError("Feature closure persistence batch exceeds the admitted bound")
        catalogs = {write.catalog_hash for write in bounded}
        if len(catalogs) != 1:
            raise ValueError("Feature closure persistence batch crosses catalogs")
        catalog_hash = next(iter(catalogs))
        if catalog_hash in self._first_loads:
            return self._persist_first_load_batch(
                bounded, catalog_hash=catalog_hash, connection=connection, timing_sink=timing_sink
            )
        if self._ledger.pending_first_load(catalog_hash) is not None:
            raise ValueError("feature_closure.recovery_required")
        head = self._ledger.require_head(catalog_hash)
        # A genesis closure is a legitimate parent for the first batch, so this
        # reads the factor axis both roots share rather than demanding a Panel.
        root = self._ledger.closure_root_for_head(head)
        if root.factor_ids != self._factor_ids:
            raise ValueError("feature_closure.root_unavailable")
        batch_key = canonical_hash(tuple(write.idempotency_key for write in bounded))
        expected_receipts = self._expected_receipts(bounded)
        pending = self._ledger.pending_transition(catalog_hash)
        if pending is not None:
            if pending.idempotency_key != batch_key:
                raise ValueError("feature_closure.pending_transition_conflict")
            disposition = self._pending_disposition(pending, connection=connection)
            if disposition == "NEXT_STATE_VERIFIED":
                self._complete(
                    transition=pending,
                    expected_receipts=expected_receipts,
                    connection=connection,
                    observed_at=max(write.observed_at for write in bounded),
                )
                return expected_receipts
            if disposition != "PRIOR_STATE_SAFE_TO_RETRY":
                raise ValueError("feature_closure.recovery_required")

        changed, prior_rows, next_hashes, identified_writes = self._changed_rows(
            bounded, catalog_hash=catalog_hash, connection=connection
        )
        if not changed:
            return cast(
                tuple[str, ...],
                self._store.upsert_feature_materialization_batch(
                    identified_writes, _connection=connection, timing_sink=timing_sink
                ),
            )
        if pending is None:
            patch_table = self._patch_table(bounded, changed)
            sessions = tuple(row[0] for row in changed)
            listing_ids = tuple(row[1] for row in changed)
            patch = self._ledger.publish_patch(
                catalog_hash=catalog_hash,
                base_head_hash=head.head_hash,
                idempotency_key=batch_key,
                factor_ids=self._factor_ids,
                listing_ids=listing_ids,
                first_session=min(sessions),
                last_session=max(sessions),
                table=patch_table,
            )
            ordered_keys = sorted(next_hashes)
            # Each session's text once: a batch's rows share their sessions.
            session_text = {
                session: session.isoformat() for session in {key[0] for key in ordered_keys}
            }
            transition = identified(
                PreparedFeatureBaseClosureTransition,
                {
                    "catalog_hash": catalog_hash,
                    "base_head_hash": head.head_hash,
                    "patch_hash": patch.patch_hash,
                    "idempotency_key": batch_key,
                    "prior_row_hashes": tuple(
                        (
                            session_text[key[0]],
                            key[1],
                            str(prior_rows[key]["row_hash"]) if key in prior_rows else None,
                        )
                        for key in ordered_keys
                    ),
                    "next_row_hashes": tuple(
                        (session_text[key[0]], key[1], next_hashes[key]) for key in ordered_keys
                    ),
                    "expected_receipt_hashes": expected_receipts,
                    "prepared_at": min(write.observed_at for write in bounded),
                },
                "transition_hash",
            )
            self._ledger.prepare(transition)
        else:
            transition = pending
        receipts = cast(
            tuple[str, ...],
            self._store.upsert_feature_materialization_batch(
                identified_writes, _connection=connection, timing_sink=timing_sink
            ),
        )
        if tuple(receipts) != expected_receipts:
            raise ValueError("feature_closure.store_readback_mismatch")
        self._complete(
            transition=transition,
            expected_receipts=expected_receipts,
            connection=connection,
            observed_at=max(write.observed_at for write in bounded),
        )
        return receipts

    def begin_first_loads(
        self,
        catalog_hashes: Sequence[str],
        *,
        idempotency_key: str,
        observed_at: datetime,
        connection: duckdb.DuckDBPyConnection,
    ) -> tuple[str, ...]:
        """Begin the first load of each part whose head is its genesis head and holds no row.

        Such a part has no prior row to compare, revise or recover to: its rows are written as
        the build computes them, with no per-key transition, and ``complete_first_loads``
        advances its head to their digest (V92). Any other part's batches pass per-key
        transitions as before.

        Returns:
            The parts whose first load began.
        """
        begun = []
        for catalog_hash in catalog_hashes:
            head = self._ledger.require_head(catalog_hash)
            if (
                head.transition_cursor != 0
                or not self._ledger.is_genesis_root(head.root_hash)
                or head.feature_row_hash_digest != empty_feature_row_hash_digest()
                or self._ledger.pending_transition(catalog_hash) is not None
                or self._ledger.pending_first_load(catalog_hash) is not None
                or self._source.feature_row_count(catalog_hash=catalog_hash, connection=connection)
            ):
                continue
            first_load = identified(
                FeatureClosureFirstLoad,
                {
                    "catalog_hash": catalog_hash,
                    "base_head_hash": head.head_hash,
                    "idempotency_key": idempotency_key,
                    "prepared_at": observed_at.astimezone(UTC),
                },
                "first_load_hash",
            )
            self._ledger.prepare_first_load(first_load)
            self._first_loads[catalog_hash] = first_load.first_load_hash
            begun.append(catalog_hash)
        return tuple(begun)

    def complete_first_loads(
        self, catalog_hashes: Sequence[str], *, connection: duckdb.DuckDBPyConnection
    ) -> None:
        """Advance each part's head to the digest of every row its first load wrote."""
        for catalog_hash in catalog_hashes:
            first_load = self._ledger.pending_first_load(catalog_hash)
            if first_load is None or (
                self._first_loads.get(catalog_hash) != first_load.first_load_hash
            ):
                raise ValueError("feature_closure.recovery_required")
            state = self._source.feature_row_hash_digest_state(
                catalog_hash=catalog_hash, connection=connection
            )
            head = self._ledger.complete_first_load(
                first_load, feature_row_hash_digest=state.digest
            )
            self._digest_states[catalog_hash] = (head.head_hash, state)
            del self._first_loads[catalog_hash]

    def _persist_first_load_batch(
        self,
        bounded: tuple[FeatureMaterializationWrite, ...],
        *,
        catalog_hash: str,
        connection: duckdb.DuckDBPyConnection,
        timing_sink: Callable[[str, float], None] | None,
    ) -> tuple[str, ...]:
        """A first load's batch: its part held no row when the load began, so no row of the
        batch replaces one, and none is compared or recorded key by key."""
        expected_receipts = self._expected_receipts(bounded)
        cutoff_sets: dict[str, tuple[str, str]] = {}
        writes = []
        for write in bounded:
            if not isinstance(write.rows, pd.DataFrame) or not write.rows_are_canonical:
                raise ValueError("production Feature closure requires canonical DataFrame rows")
            writes.append(
                replace(
                    write,
                    row_identities=(
                        write.row_identities
                        if write.row_identities is not None
                        else feature_row_identities(
                            write.rows,
                            catalog_hash=catalog_hash,
                            factor_ids=self._factor_ids,
                            cutoff_sets=cutoff_sets,
                        )
                    ),
                    existing_rows={},
                )
            )
        receipts = cast(
            tuple[str, ...],
            self._store.upsert_feature_materialization_batch(
                writes, _connection=connection, timing_sink=timing_sink
            ),
        )
        if tuple(receipts) != expected_receipts:
            raise ValueError("feature_closure.store_readback_mismatch")
        return receipts

    def assert_ready(self, catalog_hash: str) -> None:
        """Require an installed closure head without a pending transition or first load."""
        self._ledger.require_head(catalog_hash)
        if (
            self._ledger.pending_transition(catalog_hash) is not None
            or self._ledger.pending_first_load(catalog_hash) is not None
        ):
            raise ValueError("feature_closure.recovery_required")

    def closure_disposition(
        self, catalog_hash: str, *, expected_listing_ids: Sequence[str]
    ) -> FeatureClosureDisposition:
        """Report what the durable closure is, for a caller deciding a recovery.

        ``assert_ready`` answers one bit -- may work proceed -- and a caller that
        turns its exception into a state necessarily flattens four different
        situations into whichever one it named. They are not interchangeable: an
        absent closure has every listing left to compute, a pending transition
        must be reconciled before anything else runs, damaged authority must not
        be reconciled at all, and a complete closure needs no Feature work of any
        kind. This is deliberately the closure owner's answer rather than the
        caller's inference, and it reports rather than repairs.

        Completeness is membership plus integrity, never a row count. Every
        expected listing must hold live rows, and the digest the head attests
        must still be the digest those rows produce -- otherwise the store moved
        under a head that no longer describes it, which is damage rather than
        progress.
        """
        expected = frozenset(expected_listing_ids)
        if not expected:
            raise ValueError("feature_closure.expected_membership_required")
        try:
            head = self._ledger.current_head(catalog_hash)
        except (OSError, ValueError):
            # The pointer exists and what it names will not load: a head that
            # cannot be read is not a head that is absent.
            return "CLOSURE_AUTHORITY_UNAVAILABLE"
        if head is None:
            return "CLOSURE_ABSENT"
        try:
            root = self._ledger.closure_root_for_head(head)
            pending = self._ledger.pending_transition(catalog_hash)
            first_load = self._ledger.pending_first_load(catalog_hash)
        except (OSError, ValueError):
            return "CLOSURE_AUTHORITY_UNAVAILABLE"
        if root.factor_ids != self._factor_ids:
            # The closure was opened for a different factor axis; persisting into
            # it is already refused, and calling it recoverable would be wrong.
            return "CLOSURE_AUTHORITY_UNAVAILABLE"
        if pending is not None or first_load is not None:
            return "TRANSITION_RECOVERY_PENDING"
        materialized = frozenset(self._source.materialized_listing_ids(catalog_hash=catalog_hash))
        if not materialized or not expected <= materialized:
            return "MEMBERSHIP_INCOMPLETE"
        if self._source.feature_row_hash_digest(catalog_hash=catalog_hash) != (
            head.feature_row_hash_digest
        ):
            return "CLOSURE_AUTHORITY_UNAVAILABLE"
        return "MEMBERSHIP_COMPLETE"

    def reconcile_pending(
        self,
        catalog_hash: str,
        *,
        connection: duckdb.DuckDBPyConnection,
    ) -> None:
        """Resolve a verified post-commit crash before any new Feature work."""
        head = self._ledger.require_head(catalog_hash)
        first_load = self._ledger.pending_first_load(catalog_hash)
        # A build reconciles before it begins its own first loads, so a pending one is an
        # earlier build's, this coordinator's own included: complete when its head names it;
        # interrupted otherwise, and its rows are discarded back to the empty part its genesis
        # head attests, to be built again (V92).
        self._first_loads.pop(catalog_hash, None)
        if first_load is not None:
            if head.last_transition_hash != first_load.first_load_hash:
                if head.head_hash != first_load.base_head_hash:
                    raise ValueError("feature_closure.recovery_required")
                self._store.discard_part_rows(catalog_hash, _connection=connection)
            self._ledger.clear_first_load(first_load)
        pending = self._ledger.pending_transition(catalog_hash)
        if pending is None:
            return
        if self._ledger.clear_finalized_pending(pending):
            return
        disposition = self._pending_disposition(pending, connection=connection)
        if disposition == "NEXT_STATE_VERIFIED":
            self._complete(
                transition=pending,
                expected_receipts=pending.expected_receipt_hashes,
                connection=connection,
                observed_at=pending.prepared_at,
            )
            return
        if disposition == "PRIOR_STATE_SAFE_TO_RETRY":
            return
        raise ValueError("feature_closure.recovery_required")

    def _complete(
        self,
        *,
        transition: PreparedFeatureBaseClosureTransition,
        expected_receipts: tuple[str, ...],
        connection: duckdb.DuckDBPyConnection,
        observed_at: datetime,
    ) -> None:
        if self._pending_disposition(transition, connection=connection) != "NEXT_STATE_VERIFIED":
            raise ValueError("feature_closure.store_readback_mismatch")
        present_receipts = self._source.receipt_hashes(expected_receipts, connection=connection)
        if present_receipts != tuple(sorted(expected_receipts)):
            raise ValueError("feature_closure.store_readback_mismatch")
        receipt = identified(
            FeatureBaseClosureTransitionReceipt,
            {
                "transition_hash": transition.transition_hash,
                "patch_hash": transition.patch_hash,
                "store_receipt_hashes": expected_receipts,
                "revision_ids": self._source.revision_ids(
                    receipt_hashes=expected_receipts, connection=connection
                ),
                "readback_disposition": "NEXT_STATE_VERIFIED",
                "observed_at": observed_at.astimezone(UTC),
            },
            "receipt_hash",
        )
        state = self._digest_after(transition, connection=connection)
        next_head, _marker = self._ledger.complete(
            transition=transition,
            receipt=receipt,
            feature_row_hash_digest=state.digest,
        )
        self._digest_states[transition.catalog_hash] = (next_head.head_hash, state)

    def _digest_after(
        self,
        transition: PreparedFeatureBaseClosureTransition,
        *,
        connection: duckdb.DuckDBPyConnection,
    ) -> FeatureRowHashDigestState:
        """The closure digest the transition's next head attests.

        The digest is defined over the whole ordered closure and its value
        does not change here. A build writes rows of its target sessions,
        every one of them later than any session the closure held before the
        build; this coordinator is the closure's only writer. So the first
        transition it seals streams the closure once and keeps the running
        hash just before the build's earliest session; every later transition
        whose base is the head this coordinator sealed last and whose keys all
        lie on or after that session re-seals from that prefix by framing only
        the tail again (a few hundred rows a day), and the store re-proves the
        prefix's row count. A correction of an earlier session, a resumed build
        or a head sealed elsewhere streams the whole closure as before.
        """
        # ISO session texts order as their dates do.
        earliest = (
            date.fromisoformat(min(item[0] for item in transition.next_row_hashes))
            if transition.next_row_hashes
            else None
        )
        remembered = self._digest_states.get(transition.catalog_hash)
        if (
            remembered is not None
            and remembered[0] == transition.base_head_hash
            and earliest is not None
            and remembered[1].boundary is not None
            and earliest >= remembered[1].boundary
        ):
            resealed = self._source.reseal_feature_row_hash_digest(
                remembered[1], catalog_hash=transition.catalog_hash, connection=connection
            )
            if resealed is not None:
                return resealed
        return self._source.feature_row_hash_digest_state(
            catalog_hash=transition.catalog_hash, boundary=earliest, connection=connection
        )

    def _pending_disposition(
        self,
        transition: PreparedFeatureBaseClosureTransition,
        *,
        connection: duckdb.DuckDBPyConnection,
    ) -> str:
        sessions = {
            text: date.fromisoformat(text)
            for text in {item[0] for item in transition.next_row_hashes}
        }
        keys = tuple(
            (sessions[session], listing_id)
            for session, listing_id, _row_hash in transition.next_row_hashes
        )
        rows = self._source.read_row_hashes(
            catalog_hash=transition.catalog_hash,
            keys=keys,
            connection=connection,
        )
        current = tuple(rows.get(key) for key in keys)
        prior = tuple(item[2] for item in transition.prior_row_hashes)
        next_values = tuple(item[2] for item in transition.next_row_hashes)
        if current == next_values:
            return "NEXT_STATE_VERIFIED"
        if current == prior:
            return "PRIOR_STATE_SAFE_TO_RETRY"
        return "MIXED_OR_UNKNOWN"

    def _changed_rows(
        self,
        writes: tuple[FeatureMaterializationWrite, ...],
        *,
        catalog_hash: str,
        connection: duckdb.DuckDBPyConnection,
    ) -> tuple[
        tuple[_Candidate, ...],
        dict[tuple[date, str], dict[str, object]],
        dict[tuple[date, str], str],
        tuple[FeatureMaterializationWrite, ...],
    ]:
        candidates: list[_Candidate] = []
        derived: list[
            tuple[
                FeatureMaterializationWrite,
                tuple[FeatureRowIdentity, ...],
                list[tuple[date, str]],
            ]
        ] = []
        cutoff_sets: dict[str, tuple[str, str]] = {}
        for position, write in enumerate(writes):
            if not isinstance(write.rows, pd.DataFrame) or not write.rows_are_canonical:
                raise ValueError("production Feature closure requires canonical DataFrame rows")
            missing = {"session_date", "listing_id", "input_cutoffs_json", *self._factor_ids} - set(
                write.rows.columns
            )
            if missing:
                raise ValueError("Feature closure row is missing the complete factor axis")
            # A worker that computed the rows derived their identities beside them
            # (W10); otherwise they are derived here, once, and carried to the store
            # with the write.
            identities = write.row_identities
            if identities is None:
                identities = feature_row_identities(
                    write.rows,
                    catalog_hash=catalog_hash,
                    factor_ids=self._factor_ids,
                    cutoff_sets=cutoff_sets,
                )
            elif len(identities) != len(write.rows) or any(
                identity.session_date != _session(raw)
                for identity, raw in zip(identities, write.rows["session_date"], strict=True)
            ):
                raise ValueError("feature row identities do not align with the rows")
            listing_ids = write.rows["listing_id"].tolist()
            cutoff_strings = write.rows["input_cutoffs_json"].tolist()
            write_keys: list[tuple[date, str]] = []
            for index, identity in enumerate(identities):
                listing_id = str(listing_ids[index])
                write_keys.append((identity.session_date, listing_id))
                candidates.append(
                    (
                        identity.session_date,
                        listing_id,
                        str(cutoff_strings[index]),
                        identity.row_hash,
                        position,
                        index,
                    )
                )
            derived.append((write, identities, write_keys))
        candidates.sort(key=itemgetter(0, 1))
        keys: tuple[tuple[date, str], ...] = tuple(
            (candidate[0], candidate[1]) for candidate in candidates
        )
        if len(keys) != len(set(keys)):
            raise ValueError("Feature closure batch contains duplicate row keys")
        existing = self._source.read_rows(
            catalog_hash=catalog_hash,
            factor_ids=self._factor_ids,
            keys=keys,
            connection=connection,
        )
        # The store's rows at each write's keys, read once for the whole transition; the
        # store consumes them for the write rather than reading them again, per listing.
        # Nothing writes in between: the build holds the writer's gate, and the ledger's
        # files are all that is published before the store's write.
        identified = tuple(
            replace(
                write,
                row_identities=identities,
                existing_rows={key[0]: existing[key] for key in write_keys if key in existing},
            )
            for write, identities, write_keys in derived
        )
        # A write's cells column by column, as its frame holds them: a float column's
        # cells are Python floats (NaN kept), any other column's its own objects -- the
        # values a record dict would carry. Taken for a write only when one of its rows
        # is compared by value with the row it would replace.
        cells: dict[int, list[list[object]]] = {}

        def values(position: int, index: int) -> tuple[object, ...]:
            columns = cells.get(position)
            if columns is None:
                frame = cast(pd.DataFrame, writes[position].rows)
                columns = cells[position] = [
                    frame[factor_id].tolist() for factor_id in self._factor_ids
                ]
            return tuple(column[index] for column in columns)

        changed: list[_Candidate] = []
        hashes: dict[tuple[date, str], str] = {}
        for candidate, key in zip(candidates, keys, strict=True):
            prior = existing.get(key)
            if prior is not None and prior["row_hash"] == candidate[3]:
                continue
            if prior is not None and feature_row_is_numerically_identical(
                existing_cutoffs=str(prior["input_cutoffs_json"]),
                existing_values=tuple(prior[factor_id] for factor_id in self._factor_ids),
                next_cutoffs=json.loads(candidate[2]),
                next_values=values(candidate[4], candidate[5]),
            ):
                continue
            changed.append(candidate)
            hashes[key] = candidate[3]
        return tuple(changed), existing, hashes, identified

    def _expected_receipts(
        self, writes: tuple[FeatureMaterializationWrite, ...]
    ) -> tuple[str, ...]:
        receipts: list[str] = []
        for write in writes:
            if not isinstance(write.rows, pd.DataFrame) or write.rows.empty:
                raise ValueError("Feature closure requires non-empty DataFrame writes")
            sessions = tuple(
                value if isinstance(value, date) else date.fromisoformat(str(value))
                for value in write.rows["session_date"]
            )
            ineligible_count = len(write.ineligibility)
            rebuilt = tuple(sorted(set(write.factor_ids or self._factor_ids)))
            receipts.append(
                feature_materialization_receipt_hash(
                    listing_id=write.listing_id,
                    range_start=min(sessions),
                    range_end=max(sessions),
                    catalog_hash=write.catalog_hash,
                    raw_input_hash=write.raw_input_hash,
                    action_set_hash=write.action_set_hash_value,
                    market_reference_revision=write.market_reference_revision,
                    idempotency_key=write.idempotency_key,
                    coverage=feature_materialization_coverage(
                        rows=len(sessions),
                        factor_count=len(self._factor_ids),
                        rebuilt_factor_count=len(rebuilt),
                        ineligible_count=ineligible_count,
                        source_window=write.source_window.to_payload()
                        if write.source_window is not None
                        else None,
                    ),
                )
            )
        return tuple(receipts)

    def _patch_table(
        self, writes: tuple[FeatureMaterializationWrite, ...], rows: tuple[_Candidate, ...]
    ) -> pa.Table:
        payload: dict[str, pa.Array] = {
            "session_date": pa.array([row[0] for row in rows], type=pa.date32()),
            "listing_id": pa.array([row[1] for row in rows], type=pa.string()),
            "row_hash": pa.array([row[3] for row in rows], type=pa.string()),
        }
        # The patch carries raw IEEE-754 bits beside an explicit validity mask so
        # a replay can tell a missing cell from one that merely holds NaN: a
        # missing cell is exactly ``None`` and keeps zero bits, every other cell
        # keeps the little-endian bits of its own float, and a cell is valid when
        # it is neither missing nor NaN. The cells are taken from the writes'
        # frames column by column (``_factor_cells``) and gathered in key order.
        # Turned into a Python tuple per row and back, the change check and the
        # patch cost a first build's single writer 10.9 s over its 60 patches,
        # against 2.1 s this way.
        numeric_blocks: list[np.ndarray] = []
        missing_blocks: list[np.ndarray] = []
        offsets: list[int] = []
        offset = 0
        for write in writes:
            frame = cast(pd.DataFrame, write.rows)
            numeric_block, missing_block = self._factor_cells(frame)
            numeric_blocks.append(numeric_block)
            missing_blocks.append(missing_block)
            offsets.append(offset)
            offset += len(frame)
        order: np.ndarray = np.fromiter(
            (offsets[row[4]] + row[5] for row in rows), dtype=np.intp, count=len(rows)
        )
        numeric = np.concatenate(numeric_blocks)[order]
        missing = np.concatenate(missing_blocks)[order]
        bit_patterns = numeric.view("<u8")
        validity = ~missing & ~np.isnan(numeric)
        for index, factor_id in enumerate(self._factor_ids):
            payload[f"{factor_id}__bits"] = pa.array(bit_patterns[:, index], type=pa.uint64())
            payload[f"{factor_id}__valid"] = pa.array(validity[:, index], type=pa.bool_())
        return pa.table(payload)

    def _factor_cells(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Each factor cell of a frame as a float, and whether it is missing, row by factor.

        A float64 column is taken as it is: no cell of it is missing, and its NaNs keep
        their bits. Any other column's cells are taken one by one as their own objects,
        as a record dict carries them: a missing cell is exactly ``None`` and reads 0.0,
        every other cell the float it converts to.
        """
        numeric: np.ndarray = np.empty((len(frame), len(self._factor_ids)), dtype="<f8")
        missing: np.ndarray = np.zeros((len(frame), len(self._factor_ids)), dtype=bool)
        for index, factor_id in enumerate(self._factor_ids):
            column = frame[factor_id]
            if column.dtype == np.dtype("float64"):
                numeric[:, index] = column.to_numpy()
                continue
            cells = np.asarray(column.tolist(), dtype=object)
            absent = np.equal(cells, None)
            missing[:, index] = absent
            numeric[:, index] = np.where(absent, 0.0, cells).astype("<f8")
        return numeric, missing


class FeatureLayerClosures:
    """The Feature closures of an installed catalog's layer, one coordinator per part (V92).

    An unlayered catalog is its own only part, and this is its coordinator. A layered catalog
    has no closure of its own: its rows are its parts', each batch is written through its own
    part's closure, and the catalog is ready, recovered or complete exactly when every part is.
    """

    def __init__(
        self,
        *,
        store: FeatureStateRepository,
        source: FeatureClosureSourceRepository,
        ledger: FeatureClosureLedger,
        layer: FeatureCatalogLayer,
    ) -> None:
        """Bind one closure coordinator to each part of the layer."""
        self._layer = layer
        self._parts = {
            part.binding.catalog_hash: FeatureBaseClosureCoordinator(
                store=store, source=source, ledger=ledger, factor_ids=part.factor_ids
            )
            for part in layer.parts
        }

    def _closures(self, catalog_hash: str) -> tuple[tuple[str, FeatureBaseClosureCoordinator], ...]:
        if self._layer.layered and catalog_hash == self._layer.catalog.binding.catalog_hash:
            return tuple(self._parts.items())
        coordinator = self._parts.get(catalog_hash)
        if coordinator is None:
            raise ValueError("feature_closure.catalog_root_required")
        return ((catalog_hash, coordinator),)

    def persist_batch(
        self,
        writes: Sequence[FeatureMaterializationWrite],
        *,
        connection: duckdb.DuckDBPyConnection,
        timing_sink: Callable[[str, float], None] | None = None,
    ) -> tuple[str, ...]:
        """Persist a batch through its part's closure."""
        catalogs = {write.catalog_hash for write in writes}
        if len(catalogs) != 1:
            raise ValueError("Feature closure persistence batch crosses catalogs")
        coordinator = self._parts.get(next(iter(catalogs)))
        if coordinator is None:
            raise ValueError("feature_closure.catalog_root_required")
        return coordinator.persist_batch(writes, connection=connection, timing_sink=timing_sink)

    def begin_first_loads(
        self,
        catalog_hashes: Sequence[str],
        *,
        idempotency_key: str,
        observed_at: datetime,
        connection: duckdb.DuckDBPyConnection,
    ) -> tuple[str, ...]:
        """Begin each named part's first load where its own coordinator admits one."""
        begun: list[str] = []
        for catalog_hash in catalog_hashes:
            coordinator = self._parts.get(catalog_hash)
            if coordinator is None:
                raise ValueError("feature_closure.catalog_root_required")
            begun.extend(
                coordinator.begin_first_loads(
                    (catalog_hash,),
                    idempotency_key=idempotency_key,
                    observed_at=observed_at,
                    connection=connection,
                )
            )
        return tuple(begun)

    def complete_first_loads(
        self, catalog_hashes: Sequence[str], *, connection: duckdb.DuckDBPyConnection
    ) -> None:
        """Complete each named part's first load through its own coordinator."""
        for catalog_hash in catalog_hashes:
            coordinator = self._parts.get(catalog_hash)
            if coordinator is None:
                raise ValueError("feature_closure.catalog_root_required")
            coordinator.complete_first_loads((catalog_hash,), connection=connection)

    def assert_ready(self, catalog_hash: str) -> None:
        """Require every part's closure head without a pending transition."""
        for part_hash, coordinator in self._closures(catalog_hash):
            coordinator.assert_ready(part_hash)

    def closure_disposition(
        self, catalog_hash: str, *, expected_listing_ids: Sequence[str]
    ) -> FeatureClosureDisposition:
        """The catalog's disposition: its parts', the gravest first.

        Damaged authority anywhere is damage, a pending transition anywhere is reconciled
        first, and the catalog is complete only when every part is.
        """
        dispositions = {
            coordinator.closure_disposition(part_hash, expected_listing_ids=expected_listing_ids)
            for part_hash, coordinator in self._closures(catalog_hash)
        }
        for grave in ("CLOSURE_AUTHORITY_UNAVAILABLE", "TRANSITION_RECOVERY_PENDING"):
            if grave in dispositions:
                return cast(FeatureClosureDisposition, grave)
        if len(dispositions) == 1:
            return next(iter(dispositions))
        return "MEMBERSHIP_INCOMPLETE"

    def reconcile_pending(
        self,
        catalog_hash: str,
        *,
        connection: duckdb.DuckDBPyConnection,
    ) -> None:
        """Resolve each part's verified post-commit crash before any new Feature work."""
        for part_hash, coordinator in self._closures(catalog_hash):
            coordinator.reconcile_pending(part_hash, connection=connection)


def feature_row_identities(
    rows: pd.DataFrame,
    *,
    catalog_hash: str,
    factor_ids: tuple[str, ...],
    cutoff_sets: dict[str, tuple[str, str]],
) -> tuple[FeatureRowIdentity, ...]:
    """Each canonical Feature row's identity, in row order.

    What a canonical row is: its cutoff set's canonical payload and hash, and its content
    hash over that payload. The one derivation, run by the closure coordinator for a write
    that carries none, and by a Host worker beside the listing's computation (W10). A cutoff
    set belongs to a session, not to a listing, so every listing of a first build offers the
    same ~2,500 sets: `cutoff_sets` keeps each serialized set's answer, a function of the
    string alone, so a batch -- or a kept worker -- derives each once.
    """
    listing_ids = rows["listing_id"].tolist()
    cutoff_strings = rows["input_cutoffs_json"].tolist()
    # Column by column, as the frame holds them: a float column's cells are Python
    # floats (NaN kept), any other column's cells are its own objects.
    columns = [rows[factor_id].tolist() for factor_id in factor_ids]
    identities: list[FeatureRowIdentity] = []
    for index, raw_session in enumerate(rows["session_date"].tolist()):
        session = _session(raw_session)
        cutoff_string = str(cutoff_strings[index])
        derived = cutoff_sets.get(cutoff_string)
        if derived is None:
            derived = cutoff_sets[cutoff_string] = _canonical_cutoffs(cutoff_string, factor_ids)
        canonical_cutoffs, cutoff_set = derived
        identities.append(
            FeatureRowIdentity(
                session_date=session,
                canonical_cutoffs=canonical_cutoffs,
                cutoff_set_hash=cutoff_set,
                row_hash=feature_row_content_hash_from_canonical(
                    listing_id=str(listing_ids[index]),
                    session_date=session,
                    catalog_hash=catalog_hash,
                    canonical_cutoffs=canonical_cutoffs,
                    factor_ids=factor_ids,
                    values=tuple(column[index] for column in columns),
                ),
            )
        )
    return tuple(identities)


def _canonical_cutoffs(cutoff_string: str, factor_ids: tuple[str, ...]) -> tuple[str, str]:
    """One serialized cutoff set's canonical payload and hash, once its scope is the axis."""
    cutoffs = json.loads(cutoff_string)
    if not isinstance(cutoffs, Mapping) or set(cutoffs) != set(factor_ids):
        raise ValueError("Feature closure row has invalid input-cutoff scope")
    return canonical_cutoff_set(cutoffs, factor_ids=factor_ids)


def _session(value: object) -> date:
    """A row's session as a date, as the frame holds it or from its ISO text."""
    return value if isinstance(value, date) else date.fromisoformat(str(value))


__all__ = ["FeatureBaseClosureCoordinator", "feature_row_identities"]
