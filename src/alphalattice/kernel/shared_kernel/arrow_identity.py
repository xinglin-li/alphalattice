"""Canonical content hashes of Arrow columns and tables, bit for bit ``canonical_hash``.

``canonical_hash`` states an identity as the SHA-256 of one canonical JSON
document. When the document's bulk is the rows of an Arrow table or the
values of one column, building those rows as Python objects and encoding them
one value at a time is where a verification spends its time. The functions
here emit the same document -- the same bytes, proved by the parity test --
from the Arrow columns directly, and hand anything they cannot encode exactly
to the general encoder rather than approximating it.

The rules reproduced are those of the canonical encoder
(``json.JSONEncoder(sort_keys=True, separators=(",", ":"), default=str)``
over ``Table.to_pylist()`` values): object keys sorted, no whitespace, ints
as decimal integers, booleans as ``true``/``false``, nulls as ``null``,
floats as ``float.__repr__`` with ``NaN``/``Infinity``/``-Infinity`` for the
non-finite values, strings ASCII-escaped, and dates as ``str(date)``. Any
other column type, and any string the fast path cannot quote verbatim, is
encoded by the general encoder itself.

``canonical_hash_with_rows`` hashes one payload whose one large member is a
table's rows as objects, ``canonical_hash_with_row_lists`` one whose rows are
positional lists in column order; ``canonical_hash_with_values`` one whose
large member is a column's values; ``canonical_row_hashes`` hashes every row
of a table on its own, for a store that seals and re-verifies one identity per
row.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from json.encoder import encode_basestring_ascii

import pyarrow as pa
import pyarrow.compute as pc

from alphalattice.kernel.shared_kernel.identity import _CANONICAL_ENCODER, canonical_hash

_PLAIN_ASCII = r"^[\x20\x21\x23-\x5b\x5d-\x7e]*$"
"""Printable ASCII without the double quote and the backslash: the strings the
canonical encoder emits verbatim between two quotes."""


def _quoted(strings: pa.Array) -> pa.Array:
    """JSON string literals of a string array whose members are all plain ASCII."""

    return pc.binary_join_element_wise('"', strings, '"', "")


def _cell_json(column: pa.ChunkedArray) -> pa.Array | None:
    """The JSON text of every cell of one column, ``null`` for nulls; None if unsupported."""

    array = column.combine_chunks()
    kind = array.type
    if pa.types.is_boolean(kind):
        cells = pc.if_else(array, "true", "false")
    elif pa.types.is_integer(kind):
        cells = pc.cast(array, pa.string())
    elif pa.types.is_floating(kind):
        # ``float.__repr__`` per value is the floor, and no value pays more: a
        # null or non-finite value is written over its placeholder by mask
        # (null stays null, then ``null``) rather than dispatched one by one.
        finite = pc.fill_null(pc.is_finite(array), False)
        if array.null_count == 0 and pc.all(finite).as_py():
            cells = pa.array(list(map(float.__repr__, array.to_pylist())), pa.string())
        else:
            shown = pc.if_else(finite, array, pa.scalar(0.0, kind)).to_pylist()
            special = pc.if_else(
                pc.is_nan(array),
                "NaN",
                pc.if_else(pc.greater(array, 0), "Infinity", "-Infinity"),
            )
            cells = pc.if_else(
                finite, pa.array(list(map(float.__repr__, shown)), pa.string()), special
            )
    elif pa.types.is_string(kind) or pa.types.is_large_string(kind):
        text = pc.cast(array, pa.string()) if pa.types.is_large_string(kind) else array
        plain = pc.match_substring_regex(text, _PLAIN_ASCII)
        if pc.all(pc.fill_null(plain, True)).as_py():
            cells = _quoted(text)
        else:
            cells = pa.array(
                [
                    None if value is None else encode_basestring_ascii(value)
                    for value in text.to_pylist()
                ],
                pa.string(),
            )
    elif pa.types.is_date(kind):
        cells = _quoted(_iso_dates(array))
    else:
        return None
    return pc.fill_null(cells, "null")


def _iso_dates(dates: pa.Array) -> pa.Array:
    """``str(date)`` of every date: ``%04d-%02d-%02d``, from the calendar fields.

    Assembled from ``year``/``month``/``day`` rather than ``strftime``, which
    formats one element at a time and costs more than the rest of the row.
    """

    def padded(values: pa.Array, width: int) -> pa.Array:
        return pc.utf8_lpad(pc.cast(values, pa.string()), width, "0")

    year = padded(pc.year(dates), 4)
    month = padded(pc.month(dates), 2)
    day = padded(pc.day(dates), 2)
    return pc.binary_join_element_wise(year, "-", month, "-", day, "")


def _joined(rows: pa.Array) -> bytes:
    """All rows concatenated with ``,`` between them, as ASCII bytes."""

    if len(rows) == 0:
        return b""
    offsets = pa.array([0, len(rows)], pa.int32())
    whole = pc.binary_join(pa.ListArray.from_arrays(offsets, rows), ",")
    return str(whole[0].as_py()).encode("ascii")


def _chunked(column: pa.Array | pa.ChunkedArray) -> pa.ChunkedArray:
    return column if isinstance(column, pa.ChunkedArray) else pa.chunked_array([column])


def _values_json(column: pa.ChunkedArray) -> bytes | None:
    """``[v,v,...]``: the column's values as one JSON list; None if the type is unsupported."""

    cells = _cell_json(column)
    if cells is None:
        return None
    return b"[" + _joined(cells) + b"]"


def _row_objects(table: pa.Table) -> pa.Array | None:
    """``{...}`` per row: each row as a sorted-key object; None if any column is unsupported."""

    names = list(table.schema.names)
    if not names or len(set(names)) != len(names):
        return None
    parts: list[object] = []
    for position, name in enumerate(sorted(names)):
        cells = _cell_json(table.column(name))
        if cells is None:
            return None
        parts.append(("{" if position == 0 else ",") + encode_basestring_ascii(name) + ":")
        parts.append(cells)
    parts.append("}")
    return pc.binary_join_element_wise(*parts, "")


def _rows_json(table: pa.Table) -> bytes | None:
    """``[{...},{...}]``: every row as a sorted-key object; None if any column is unsupported."""

    rows = _row_objects(table)
    if rows is None:
        return None
    return b"[" + _joined(rows) + b"]"


def _row_lists(table: pa.Table) -> pa.Array | None:
    """``[...]`` per row: each row as a list of its cells in column order; None if unsupported.

    The document the canonical encoder writes for a row given as a tuple or
    list (``["2024-06-25","LISTING","hash"]``) rather than as an object; the
    positional shape a store uses when it seals one identity per row and
    then hashes the sequence of those identities.
    """

    if not table.schema.names:
        return None
    parts: list[object] = []
    for position, name in enumerate(table.schema.names):
        cells = _cell_json(table.column(name))
        if cells is None:
            return None
        parts.append("[" if position == 0 else ",")
        parts.append(cells)
    parts.append("]")
    return pc.binary_join_element_wise(*parts, "")


def _row_lists_json(table: pa.Table) -> bytes | None:
    """``[[...],[...]]``: every row as a positional list; None if any column is unsupported."""

    rows = _row_lists(table)
    if rows is None:
        return None
    return b"[" + _joined(rows) + b"]"


def _document(payload: Mapping[str, object], key: str, encoded: bytes) -> bytes:
    """The canonical object of ``payload`` with ``encoded`` in place of member ``key``."""

    members = {**payload, key: None}
    if any(not isinstance(name, str) for name in members):
        raise ValueError("arrow_identity.member_names_must_be_strings")
    pieces = []
    for name in sorted(members):
        value = encoded if name == key else _CANONICAL_ENCODER.encode(payload[name]).encode("utf-8")
        pieces.append(encode_basestring_ascii(name).encode("utf-8") + b":" + value)
    return b"{" + b",".join(pieces) + b"}"


def canonical_hash_with_rows(payload: Mapping[str, object], *, key: str, table: pa.Table) -> str:
    """``canonical_hash({**payload, key: table.to_pylist()})``, from the columns.

    ``key`` names the member that carries the table's rows (a list of one
    object per row, keys sorted); the other members of ``payload`` are encoded
    by the canonical encoder as always.
    """
    if key in payload:
        raise ValueError("arrow_identity.rows_member_already_present")
    encoded = _rows_json(table)
    if encoded is None:
        general: str = canonical_hash({**payload, key: table.to_pylist()})
        return general
    return hashlib.sha256(_document(payload, key, encoded)).hexdigest()


def canonical_hash_with_row_lists(
    payload: Mapping[str, object], *, key: str, table: pa.Table
) -> str:
    """``canonical_hash({**payload, key: [tuple(row) for row in rows]})``, from the columns.

    ``key`` names the member that carries the table's rows as positional lists
    in column order (the encoder writes a tuple and a list the same way); the
    other members of ``payload`` are encoded by the canonical encoder as
    always.
    """
    if key in payload:
        raise ValueError("arrow_identity.rows_member_already_present")
    encoded = _row_lists_json(table)
    if encoded is None:
        general: str = canonical_hash(
            {**payload, key: [tuple(row.values()) for row in table.to_pylist()]}
        )
        return general
    return hashlib.sha256(_document(payload, key, encoded)).hexdigest()


def canonical_hash_with_values(
    payload: Mapping[str, object], *, key: str, column: pa.Array | pa.ChunkedArray
) -> str:
    """``canonical_hash({**payload, key: column.to_pylist()})``, from the column.

    ``key`` names the member that carries the column's values as a list.
    """
    if key in payload:
        raise ValueError("arrow_identity.values_member_already_present")
    chunked = _chunked(column)
    encoded = _values_json(chunked)
    if encoded is None:
        general: str = canonical_hash({**payload, key: chunked.to_pylist()})
        return general
    return hashlib.sha256(_document(payload, key, encoded)).hexdigest()


def canonical_row_hashes(table: pa.Table) -> list[str]:
    """``[canonical_hash(row) for row in table.to_pylist()]``, from the columns.

    One identity per row, each the hash of that row as a sorted-key object;
    the rows of a table with a column the fast path cannot encode go through
    the general encoder one by one, as before.
    """
    rows = _row_objects(table)
    if rows is None:
        return [canonical_hash(row) for row in table.to_pylist()]
    return [hashlib.sha256(text.encode("ascii")).hexdigest() for text in rows.to_pylist()]


__all__ = [
    "canonical_hash_with_row_lists",
    "canonical_hash_with_rows",
    "canonical_hash_with_values",
    "canonical_row_hashes",
]
