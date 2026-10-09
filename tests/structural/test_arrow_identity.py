"""The Arrow-native canonical hashes are the canonical encoder's hashes, bit for bit.

``arrow_identity`` is used where an identity is verified on every read of an
artifact, so its only requirement is parity: whatever ``canonical_hash`` says
of the Python rows, the Arrow path must say of the columns, and whatever it
cannot encode exactly it must hand back to the general encoder.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pyarrow as pa
import pytest

from alphalattice.kernel.shared_kernel.arrow_identity import (
    canonical_hash_with_row_lists,
    canonical_hash_with_rows,
    canonical_hash_with_values,
    canonical_row_hashes,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_SPECIAL_FLOATS = (
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.1,
    1e-7,
    5e-324,
    1e16,
    9.999999999999998e15,
    1e21,
    1.7976931348623157e308,
    math.nan,
    math.inf,
    -math.inf,
)
_SPECIAL_STRINGS = (
    "",
    "plain",
    'quote"d',
    "back\\slash",
    "tab\tnew\nline",
    "é中\U0001f600",
    "\x00\x1f\x7f",
)


def _random_table(rng: random.Random, rows: int) -> pa.Table:
    def a_float() -> float | None:
        roll = rng.random()
        if roll < 0.05:
            return None
        if roll < 0.15:
            return rng.choice(_SPECIAL_FLOATS)
        return rng.uniform(-1e6, 1e6) * 10 ** rng.randint(-12, 12)

    def a_string() -> str | None:
        roll = rng.random()
        if roll < 0.05:
            return None
        if roll < 0.15:
            return rng.choice(_SPECIAL_STRINGS)
        return "".join(rng.choice("abcdef0123456789-_") for _ in range(rng.randint(0, 12)))

    def a_date() -> date | None:
        if rng.random() < 0.05:
            return None
        return date(1970, 1, 1) + timedelta(days=rng.randint(-40_000, 40_000))

    def a_bool() -> bool | None:
        return None if rng.random() < 0.05 else rng.random() < 0.5

    def an_int() -> int | None:
        return None if rng.random() < 0.05 else rng.randint(-(2**62), 2**62)

    return pa.table(
        {
            "zeta_float": pa.array([a_float() for _ in range(rows)], pa.float64()),
            "alpha_text": pa.array([a_string() for _ in range(rows)], pa.string()),
            "mid_date": pa.array([a_date() for _ in range(rows)], pa.date32()),
            "flag": pa.array([a_bool() for _ in range(rows)], pa.bool_()),
            "count": pa.array([an_int() for _ in range(rows)], pa.int64()),
            "small": pa.array(
                [None if v is None else v % 100 for v in [an_int() for _ in range(rows)]],
                pa.int16(),
            ),
            "wide_text": pa.array([a_string() for _ in range(rows)], pa.large_string()),
        }
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_row_and_value_hashes_equal_the_canonical_encoder(seed: int) -> None:
    """Row and value hashes equal the canonical encoder."""

    rng = random.Random(seed)
    for rows in (0, 1, 2, 17, 500):
        table = _random_table(rng, rows)
        payload = {"identity": {"kind": "x", "n": rows}, "schema": str(table.schema)}
        assert canonical_hash_with_rows(payload, key="rows", table=table) == canonical_hash(
            {**payload, "rows": table.to_pylist()}
        )
        for name in table.schema.names:
            column = table.column(name)
            assert canonical_hash_with_values(
                payload, key="values", column=column
            ) == canonical_hash({**payload, "values": column.to_pylist()})
        chunked = pa.concat_tables([table, table]).column("zeta_float")
        assert chunked.num_chunks == 2 or rows == 0
        assert canonical_hash_with_values(payload, key="values", column=chunked) == (
            canonical_hash({**payload, "values": chunked.to_pylist()})
        )
        # One identity per row: the hash of that row as the encoder's sorted-key
        # object, including over a table of two chunks.
        doubled = pa.concat_tables([table, table])
        assert canonical_row_hashes(doubled) == [canonical_hash(row) for row in doubled.to_pylist()]
        # Rows as positional lists in column order, the shape a store hashes
        # when its identity is a sequence of per-row tuples.
        assert canonical_hash_with_row_lists(payload, key="rows", table=doubled) == canonical_hash(
            {**payload, "rows": [tuple(row.values()) for row in doubled.to_pylist()]}
        )


def _batched_row_encoder_hash(payload: dict[str, object], table: pa.Table, batch_rows: int) -> str:
    """The UI branch's chunk hash before the merge (2026-09-21): the rows streamed through
    the canonical encoder a batch at a time, under the same three sorted keys."""

    from hashlib import sha256

    from alphalattice.kernel.shared_kernel.identity import canonical_json

    digest = sha256()
    digest.update(f'{{"identity":{canonical_json(payload["identity"])},"rows":['.encode())
    for offset in range(0, table.num_rows, batch_rows):
        rows = canonical_json(table.slice(offset, batch_rows).to_pylist())
        digest.update(("," if offset else "").encode() + rows[1:-1].encode())
    digest.update(f'],"schema":{canonical_json(str(table.schema))}}}'.encode())
    return digest.hexdigest()


@pytest.mark.parametrize("seed", [3, 11])
def test_the_three_chunk_encoders_state_one_identity(seed: int) -> None:
    """The three chunk encoders state one identity."""

    rng = random.Random(seed)
    for rows in (0, 1, 7, 2_050):
        table = _random_table(rng, rows)
        if rows:
            table = pa.concat_tables([table, table.slice(0, 1)])  # two chunks in every column
        payload = {"identity": {"kind": "score", "n": rows}, "schema": str(table.schema)}
        general = canonical_hash({**payload, "rows": table.to_pylist()})
        assert canonical_hash_with_rows(payload, key="rows", table=table) == general
        assert _batched_row_encoder_hash(payload, table, batch_rows=1024) == general
        assert _batched_row_encoder_hash(payload, table, batch_rows=3) == general


def test_unsupported_columns_fall_back_to_the_general_encoder() -> None:
    table = pa.table(
        {
            "stamp": pa.array([1, 2], pa.timestamp("us")),
            "text": pa.array(["x", "y"]),
        }
    )
    payload = {"identity": 1}
    assert canonical_hash_with_rows(payload, key="rows", table=table) == canonical_hash(
        {**payload, "rows": table.to_pylist()}
    )
    assert canonical_hash_with_values(payload, key="values", column=table.column("stamp")) == (
        canonical_hash({**payload, "values": table.column("stamp").to_pylist()})
    )
    duplicated = pa.Table.from_arrays([pa.array([1]), pa.array([2])], names=["same", "same"])
    assert canonical_hash_with_rows(payload, key="rows", table=duplicated) == canonical_hash(
        {**payload, "rows": duplicated.to_pylist()}
    )
    for fallback in (table, duplicated):
        assert canonical_row_hashes(fallback) == [
            canonical_hash(row) for row in fallback.to_pylist()
        ]


def test_the_rows_member_is_the_one_the_caller_names() -> None:
    with pytest.raises(ValueError, match="rows_member_already_present"):
        canonical_hash_with_rows({"rows": 1}, key="rows", table=pa.table({"a": [1]}))
    with pytest.raises(ValueError, match="values_member_already_present"):
        canonical_hash_with_values({"values": 1}, key="values", column=pa.array([1]))
