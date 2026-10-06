"""Stable identity helpers for annual Sector-Neutral Panel artifacts.

The DuckDB ``to_json(struct_pack())`` row-hash semantics are preserved under
both identity rules. DuckDB is used here only as an in-memory calculation
engine over a registered Arrow relation; no workspace table is created.

Under ``PANEL_ROW_IDENTITY_BY_BINDING`` a row's hash binds the build's
manifest and sector revisions. Under ``PANEL_ROW_IDENTITY_BY_CROSS_SECTION``
it binds the identity of the row's own session cross-section instead, read
from the ``cross_section_identity`` column the composition stamps per row;
the ``manifest_revision`` and ``sector_revision`` columns are then the
provenance of the build that wrote the file, like the receipt column, and
no longer enter the hash.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import duckdb
import pyarrow as pa

from alphalattice.foundation.feature_engine.contracts import (
    PANEL_ROW_IDENTITY_BY_BINDING,
    PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
)
from alphalattice.kernel.shared_kernel.arrow_identity import canonical_hash_with_values

PANEL_IDENTITY_COLUMNS = (
    "manifest_revision",
    "sector_revision",
    "catalog_hash",
    "policy_hash",
    "cross_section_identity",
    "session_date",
    "listing_id",
    "row_hash",
    "materialization_receipt_hash",
)
"""Every non-factor column a Panel partition may carry, under either rule."""

CROSS_SECTION_IDENTITY_COLUMN = "cross_section_identity"

_HASH_BATCH_ROWS = 8192
"""Rows in each record batch a row-hash query scans. DuckDB scans a registered Arrow table one
batch per thread, and a partition read from its file or composed in memory is one batch, which
hashed every row on one thread (V92); a row's hash and the rows' order are each batch's alone."""


def _panel_row_hash_select_sql(
    *,
    relation_name: str,
    factors: tuple[str, ...],
    output_factors: tuple[str, ...] | None = None,
    receipt_expression: str,
    order_rows: bool,
    identity_basis: str,
) -> str:
    selected_factors = output_factors or factors
    factor_select = ", ".join(f'"{factor_id}"' for factor_id in selected_factors)
    ordering = " ORDER BY session_date, listing_id" if order_rows else ""
    row_hash = _row_hash_sql(identity_basis=identity_basis, factors=factors, stamped=True)
    column = CROSS_SECTION_IDENTITY_COLUMN
    cross_section = (
        f"CAST({column} AS VARCHAR) AS {column},"
        if identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION
        else ""
    )
    return f"""
        SELECT
            ?::VARCHAR AS manifest_revision,
            ?::VARCHAR AS sector_revision,
            ?::VARCHAR AS catalog_hash,
            ?::VARCHAR AS policy_hash,
            {cross_section}
            CAST(session_date AS DATE) AS session_date,
            CAST(listing_id AS VARCHAR) AS listing_id,
            {row_hash} AS row_hash,
            {receipt_expression} AS materialization_receipt_hash,
            {factor_select}
        FROM {relation_name}{ordering}
    """


def _row_hash_sql(*, identity_basis: str, factors: tuple[str, ...], stamped: bool) -> str:
    """The row hash under one rule, the one definition a writer stamps and a reader checks.

    A writer binds the provenance it stamps as parameters; a reader checking a written
    partition reads the same fields from the partition's own columns.
    """
    provenance: tuple[str, ...] = (
        "manifest_revision",
        "sector_revision",
        "catalog_hash",
        "policy_hash",
    )
    if identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
        provenance = ("catalog_hash", "policy_hash")
        packed = [f"cross_section := CAST({CROSS_SECTION_IDENTITY_COLUMN} AS VARCHAR)"]
    elif identity_basis == PANEL_ROW_IDENTITY_BY_BINDING:
        packed = []
    else:
        raise ValueError(f"unknown Panel row identity basis: {identity_basis}")
    packed += [f"{name} := {'?::VARCHAR' if stamped else name}" for name in provenance]
    packed += [
        "session := CAST(session_date AS DATE)",
        "listing := CAST(listing_id AS VARCHAR)",
        "factor_values := struct_pack("
        + ", ".join(f'{factor_id} := "{factor_id}"' for factor_id in factors)
        + ")",
    ]
    return f"sha256(to_json(struct_pack({', '.join(packed)})))"


def panel_rows_match_their_hashes(
    table: pa.Table, *, row_hash_factor_ids: Sequence[str] | None
) -> bool:
    """Whether every row's hash is the one its values give (LAWS.md EV2, V270).

    The factor axis is the recorded one: the snapshot's catalog factors in the partition's
    physical order, as its row hashes packed them when it was written; without a catalog, every
    factor column the partition holds.

    Args:
        table: One written partition, every column.
        row_hash_factor_ids: The factors its row hashes cover, if the snapshot names them.

    Returns:
        False when a value moved under a row hash that was kept.
    """
    covered = (
        set(row_hash_factor_ids)
        if row_hash_factor_ids is not None
        else set(table.column_names).difference(PANEL_IDENTITY_COLUMNS)
    )
    factors = _validated_recorded_factor_ids(
        tuple(name for name in table.column_names if name in covered)
    )
    if set(factors) != covered:
        return False
    identity_basis = (
        PANEL_ROW_IDENTITY_BY_CROSS_SECTION
        if CROSS_SECTION_IDENTITY_COLUMN in table.column_names
        else PANEL_ROW_IDENTITY_BY_BINDING
    )
    provenance = (
        ("catalog_hash", "policy_hash")
        if identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION
        else ("manifest_revision", "sector_revision", "catalog_hash", "policy_hash")
    )
    # A partition without the columns its row identity packs cannot show its values match.
    if not {*provenance, "session_date", "listing_id", "row_hash"} <= set(table.column_names):
        return False
    row_hash = _row_hash_sql(identity_basis=identity_basis, factors=factors, stamped=False)
    connection = duckdb.connect(":memory:")
    try:
        connection.register("panel_partition", _in_batches(table))
        found = connection.execute(
            f"SELECT count(*) FILTER (WHERE {row_hash} IS DISTINCT FROM row_hash) "
            "FROM panel_partition"
        ).fetchone()
    finally:
        connection.close()
    return found is not None and found[0] == 0


def hash_panel_rows(
    table: pa.Table,
    *,
    manifest_revision: str,
    sector_revision: str,
    catalog_hash: str,
    policy_hash: str,
    factor_ids: Sequence[str],
    identity_basis: str = PANEL_ROW_IDENTITY_BY_BINDING,
) -> pa.Table:
    """Add identity columns and exact row hashes to raw Panel rows under one rule."""
    factors = _validated_factor_ids(factor_ids)
    return _hash_panel_rows(
        table,
        manifest_revision=manifest_revision,
        sector_revision=sector_revision,
        catalog_hash=catalog_hash,
        policy_hash=policy_hash,
        factors=factors,
        identity_basis=identity_basis,
    )


def hash_panel_rows_with_recorded_axes(
    table: pa.Table,
    *,
    manifest_revision: str,
    sector_revision: str,
    catalog_hash: str,
    policy_hash: str,
    row_hash_factor_ids: Sequence[str],
    output_factor_ids: Sequence[str],
    identity_basis: str = PANEL_ROW_IDENTITY_BY_BINDING,
) -> pa.Table:
    """Replay distinct historical row-hash and physical output axes.

    Some legacy wide snapshots retained all-null columns from an older catalog
    generation. Those columns remained physical output but did not participate
    in the newer catalog's DuckDB ``struct_pack`` row hash.
    """
    row_hash_factors = _validated_recorded_factor_ids(row_hash_factor_ids)
    output_factors = _validated_recorded_factor_ids(output_factor_ids)
    if not set(row_hash_factors).issubset(output_factors):
        raise ValueError("row-hash factor axis is not contained in physical output axis")
    return _hash_panel_rows(
        table,
        manifest_revision=manifest_revision,
        sector_revision=sector_revision,
        catalog_hash=catalog_hash,
        policy_hash=policy_hash,
        factors=row_hash_factors,
        output_factors=output_factors,
        identity_basis=identity_basis,
    )


def _hash_panel_rows(
    table: pa.Table,
    *,
    manifest_revision: str,
    sector_revision: str,
    catalog_hash: str,
    policy_hash: str,
    factors: tuple[str, ...],
    output_factors: tuple[str, ...] | None = None,
    identity_basis: str,
) -> pa.Table:
    selected_factors = output_factors or factors
    expected = {"session_date", "listing_id", "materialization_receipt_hash", *selected_factors}
    if identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
        expected.add(CROSS_SECTION_IDENTITY_COLUMN)
    if not expected.issubset(table.column_names):
        missing = tuple(sorted(expected.difference(table.column_names)))
        raise ValueError(f"panel row-hash input is missing columns: {missing}")
    relation = "panel_row_hash_input"
    connection = duckdb.connect(":memory:")
    try:
        connection.register(relation, _in_batches(table))
        sql = _panel_row_hash_select_sql(
            relation_name=relation,
            factors=factors,
            output_factors=selected_factors,
            receipt_expression="CAST(materialization_receipt_hash AS VARCHAR)",
            order_rows=True,
            identity_basis=identity_basis,
        )
        parameters = [manifest_revision, sector_revision, catalog_hash, policy_hash]
        if identity_basis == PANEL_ROW_IDENTITY_BY_BINDING:
            parameters.extend([manifest_revision, sector_revision, catalog_hash, policy_hash])
        else:
            parameters.extend([catalog_hash, policy_hash])
        return connection.execute(sql, parameters).to_arrow_table()
    finally:
        connection.close()


def _in_batches(table: pa.Table) -> pa.Table:
    """The same rows in batches DuckDB's threads scan apart."""
    return pa.Table.from_batches(
        table.to_batches(max_chunksize=_HASH_BATCH_ROWS), schema=table.schema
    )


def panel_schema_hash(schema: pa.Schema) -> str:
    """Return the unchanged logical schema identity used by snapshots."""
    return hashlib.sha256(str(schema.remove_metadata()).encode("utf-8")).hexdigest()


def panel_chunk_hash(
    table: pa.Table, *, panel_binding_hash: str, year: int, schema: pa.Schema | None = None
) -> str:
    """Return the unchanged binding-aware annual chunk identity.

    ``schema`` is the partition's whole schema when ``table`` is a projection
    of its identity columns (a reader that verifies a written partition need
    not decode every factor column to do so); the identity itself is unchanged.

    The row hashes are hashed from the Arrow column
    (``canonical_hash_with_values``, byte for byte the canonical encoder's
    document over ``to_pylist()``) rather than from a Python list of every
    row's hash: a reader verifies this identity on every chunk of every read.
    """
    return canonical_hash_with_values(
        {
            "panel_binding_hash": panel_binding_hash,
            "year": year,
            "row_count": table.num_rows,
            "schema_hash": panel_schema_hash(table.schema if schema is None else schema),
        },
        key="row_hashes",
        column=table.column("row_hash"),
    )


def _validated_factor_ids(factor_ids: Sequence[str]) -> tuple[str, ...]:
    factors = tuple(str(value) for value in factor_ids)
    if not factors or factors != tuple(sorted(set(factors))):
        raise ValueError("panel factor IDs must be sorted and unique")
    if any(not factor_id.replace("_", "").isalnum() for factor_id in factors):
        raise ValueError("panel factor ID is not a safe SQL identifier")
    return factors


def _validated_recorded_factor_ids(factor_ids: Sequence[str]) -> tuple[str, ...]:
    factors = tuple(str(value) for value in factor_ids)
    if not factors or len(factors) != len(set(factors)):
        raise ValueError("recorded panel factor IDs must be non-empty and unique")
    if any(not factor_id.replace("_", "").isalnum() for factor_id in factors):
        raise ValueError("panel factor ID is not a safe SQL identifier")
    return factors


__all__ = [
    "CROSS_SECTION_IDENTITY_COLUMN",
    "PANEL_IDENTITY_COLUMNS",
    "hash_panel_rows",
    "hash_panel_rows_with_recorded_axes",
    "panel_chunk_hash",
    "panel_rows_match_their_hashes",
    "panel_schema_hash",
]
