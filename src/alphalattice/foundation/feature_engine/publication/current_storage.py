"""Current-only base Feature storage schema and cutoff-set normalization.

Rows are keyed by listing, session and the catalog that computed them, so rows of two catalogs
never collide. A layered catalog (`catalog/layer.py`, V92) owns no rows of its own: the runtime
view presents its rows composed from its parts' rows, the base's and one column catalog's per
added factor, under its own hash.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from hashlib import sha256
from typing import Literal

import duckdb

from alphalattice.control.workspace_runtime.verified_facts import ensure_year_fact_schema
from alphalattice.kernel.shared_kernel.identity import canonical_hash

FeatureStorageLayout = Literal["CURRENT_BY_ROW_CATALOG", "CURRENT_ONLY", "LEGACY_BY_CATALOG"]
_HEX = re.compile(r"[0-9a-f]{64}")


class FeatureRowCatalogAuthorityError(RuntimeError):
    """A feature row is being treated as current under a catalog that did not compute it."""


def assert_rows_carry_installed_catalog(
    connection: duckdb.DuckDBPyConnection,
    *,
    catalog_hash: str,
    parts: Sequence[str] = (),
) -> None:
    """Refuse before publication if any live row was computed by another catalog.

    Absence is the ordinary state after a catalog rotation: rows under the old
    identity simply stop matching, and the maintenance path recomputes them. What
    must never happen is a row of the right *shape* answering for a catalog that
    did not produce it, so this names the offending identities rather than
    reporting a count. A layered catalog's rows are its parts' (``parts``).
    """
    if feature_storage_layout(connection) != "CURRENT_BY_ROW_CATALOG":
        raise FeatureRowCatalogAuthorityError("feature_storage.row_catalog_identity_absent")
    rows = connection.execute(
        """
        SELECT DISTINCT catalog_hash FROM feature_daily_current
        WHERE NOT list_contains(?::VARCHAR[], catalog_hash)
        ORDER BY catalog_hash
        """,
        [list(parts) or [catalog_hash]],
    ).fetchall()
    if rows:
        raise FeatureRowCatalogAuthorityError("feature_storage.row_catalog_identity_mismatch")


def canonical_cutoff_set(
    value: str | Mapping[str, object], *, factor_ids: Sequence[str]
) -> tuple[str, str]:
    """Return the canonical payload and its content hash for one cutoff set."""
    parsed: object = json.loads(value) if isinstance(value, str) else dict(value)
    if not isinstance(parsed, Mapping):
        raise ValueError("feature input cutoffs must be a mapping")
    normalized = {str(key): parsed[key] for key in sorted(parsed)}
    if tuple(normalized) != tuple(sorted(factor_ids)):
        raise ValueError("feature input-cutoff scope does not match the active catalog")
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return payload, cutoff_set_hash(payload)


def cutoff_set_hash(canonical_payload: str) -> str:
    """Hash one cutoff set from its canonical payload.

    The same bytes ``canonical_hash({"kind": "FeatureInputCutoffSet",
    "cutoffs": normalized})`` encodes -- the canonical encoder sorts the two
    keys and writes the mapping exactly as ``payload`` already spells it --
    without encoding the mapping a second time.
    """
    return sha256(
        f'{{"cutoffs":{canonical_payload},"kind":"FeatureInputCutoffSet"}}'.encode()
    ).hexdigest()


def feature_storage_layout(connection: duckdb.DuckDBPyConnection) -> FeatureStorageLayout:
    """Classify the existing Feature current table by its recorded identity columns."""
    columns = {
        str(row[0])
        for row in connection.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'main' AND table_name = 'feature_daily_current'
            """
        ).fetchall()
    }
    if not columns:
        return "CURRENT_BY_ROW_CATALOG"
    if "cutoff_set_hash" in columns and "catalog_hash" in columns:
        return "CURRENT_BY_ROW_CATALOG"
    if "cutoff_set_hash" in columns and "catalog_hash" not in columns:
        # Rows with no recorded computing catalog. The runtime view used to
        # supply one by joining the current singleton, so rotating the catalog
        # presented every historical value as belonging to a method that never
        # produced it. Nothing can repair that from the rows themselves, so the
        # shape is still recognised -- an explicit reader can still read frozen
        # legacy values -- and refused wherever it would be treated as current.
        return "CURRENT_ONLY"
    if "input_cutoffs_json" in columns and "catalog_hash" in columns:
        return "LEGACY_BY_CATALOG"
    raise ValueError("feature_daily_current has an unsupported storage schema")


def ensure_feature_current_schema(
    connection: duckdb.DuckDBPyConnection,
    *,
    catalog_hash: str,
    factor_ids: Sequence[str],
    activated_at: datetime,
    parts: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> FeatureStorageLayout:
    """Create the current-only schema, or expose a read view over a legacy database.

    ``parts`` lays a layered catalog's rows out (V92): each part's hash and factor axis, the
    base first. Their rows compose the catalog's in the runtime view.
    """
    factors = tuple(factor_ids)
    if not factors or len(factors) != len(set(factors)):
        raise ValueError("current Feature factor axis must be ordered and unique")
    layer = _layer(catalog_hash, factors, parts)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_catalog_current (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
            catalog_hash VARCHAR NOT NULL,
            factor_count INTEGER NOT NULL,
            factor_axis_hash VARCHAR NOT NULL,
            activated_at TIMESTAMP NOT NULL
        )
        """
    )
    # The year seals are year facts now; 0.1.3's own table goes with this schema.
    connection.execute("DROP TABLE IF EXISTS feature_year_seal")
    ensure_year_fact_schema(connection)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_input_cutoff_set (
            cutoff_set_hash VARCHAR PRIMARY KEY,
            input_cutoffs_json VARCHAR NOT NULL,
            factor_count INTEGER NOT NULL
        )
        """
    )
    table_exists = bool(
        connection.execute(
            """
            SELECT count(*) FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'feature_daily_current'
              AND table_type = 'BASE TABLE'
            """
        ).fetchone()[0]
    )
    if not table_exists:
        factor_columns = ",\n                ".join(
            f'"{factor_id}" DOUBLE' for factor_id in factors
        )
        connection.execute(
            f"""
            CREATE TABLE feature_daily_current (
                listing_id VARCHAR NOT NULL,
                session_date DATE NOT NULL,
                catalog_hash VARCHAR NOT NULL,
                raw_input_hash VARCHAR NOT NULL,
                action_set_hash VARCHAR NOT NULL,
                market_reference_revision VARCHAR NOT NULL,
                cutoff_set_hash VARCHAR NOT NULL,
                row_hash VARCHAR NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                {factor_columns},
                PRIMARY KEY (listing_id, session_date, catalog_hash)
            )
            """
        )
    layout = feature_storage_layout(connection)
    if layout == "CURRENT_BY_ROW_CATALOG":
        _key_rows_by_catalog(connection)
    connection.execute(
        "ALTER TABLE feature_daily_current ADD COLUMN IF NOT EXISTS "
        "source_verification_receipt_hash VARCHAR"
    )
    # A catalog that gains a factor (a person's activation) gains its column. A row an earlier
    # catalog computed holds nothing in it and names that catalog, so it answers no reader of
    # this one until a build computes it again.
    present = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info('feature_daily_current')").fetchall()
    }
    for factor_id in factors:
        if factor_id not in present:
            connection.execute(f'ALTER TABLE feature_daily_current ADD COLUMN "{factor_id}" DOUBLE')
    connection.execute(
        """
        INSERT INTO feature_catalog_current
        VALUES (TRUE, ?, ?, ?, ?)
        ON CONFLICT (singleton) DO UPDATE SET
            catalog_hash = excluded.catalog_hash,
            factor_count = excluded.factor_count,
            factor_axis_hash = excluded.factor_axis_hash,
            activated_at = excluded.activated_at
        """,
        [
            catalog_hash,
            len(factors),
            canonical_hash({"kind": "FeatureFactorAxis", "factor_ids": factors}),
            activated_at,
        ],
    )
    connection.execute("DROP VIEW IF EXISTS feature_daily_runtime")
    if layout == "CURRENT_BY_ROW_CATALOG":
        connection.execute(
            f"CREATE VIEW feature_daily_runtime AS {_runtime_view(catalog_hash, factors, layer)}"
        )
    elif layout == "CURRENT_ONLY":
        raise FeatureRowCatalogAuthorityError("feature_storage.row_catalog_identity_absent")
    else:
        connection.execute(
            "CREATE VIEW feature_daily_runtime AS SELECT * FROM feature_daily_current"
        )
    return layout


def runtime_view_hash(connection: duckdb.DuckDBPyConnection) -> str:
    """The runtime view's definition, by hash: a seal of its rows holds under that one only."""
    row = connection.execute(
        "SELECT sql FROM duckdb_views() WHERE view_name = 'feature_daily_runtime' AND NOT internal"
    ).fetchone()
    return sha256(str(row[0] if row is not None else "").encode()).hexdigest()


def layered_row_hash(catalog_hash: str, part_row_hashes: Sequence[str]) -> str:
    """A layered catalog's row identity: its parts' row identities, the base first (V92).

    Each part's row hash binds that part's catalog, cutoffs and values, so the composed row
    changes exactly when one of its parts does. The runtime view derives it in SQL.

    Args:
        catalog_hash: The layered catalog.
        part_row_hashes: Each part's row hash, in the layer's order.

    Returns:
        The composed row's hash.
    """
    return sha256(
        "|".join(("FeatureLayeredRow", catalog_hash, *part_row_hashes)).encode()
    ).hexdigest()


def _layer(
    catalog_hash: str,
    factors: tuple[str, ...],
    parts: Sequence[tuple[str, Sequence[str]]] | None,
) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
    if parts is None:
        return None
    layer = tuple((str(part_hash), tuple(axis)) for part_hash, axis in parts)
    owned = [factor_id for _hash, axis in layer for factor_id in axis]
    hashes = [part_hash for part_hash, _axis in layer]
    if (
        not layer[1:]
        or sorted(owned) != sorted(factors)
        or len(set(hashes)) != len(hashes)
        or catalog_hash in hashes
        or not all(_HEX.fullmatch(value) for value in (catalog_hash, *hashes))
    ):
        raise ValueError("feature.catalog_layer_invalid")
    return layer


def _runtime_view(
    catalog_hash: str,
    factors: tuple[str, ...],
    layer: tuple[tuple[str, tuple[str, ...]], ...] | None,
) -> str:
    """Every row as its own catalog's, and a layered catalog's rows composed.

    `f.catalog_hash`, not the singleton's. The row carries the identity of the method that
    computed it, so rotating the current catalog leaves values that no method recomputed
    matching no reader -- absent rather than relabelled -- instead of silently answering for
    the new one. A layered catalog has a row where every part holds one: each value and cutoff
    its factor's part's, the lineage the base's, the identity ``layered_row_hash``, and no
    single materialization verified it. Rows an earlier build stored under the layered catalog
    itself are not presented beside the composed ones.
    """
    projection = ", ".join(f'f."{factor_id}"' for factor_id in factors)
    rows = f"""
        SELECT f.listing_id, f.session_date, f.catalog_hash,
               f.raw_input_hash, f.action_set_hash,
               f.market_reference_revision, cutoffs.input_cutoffs_json,
               f.row_hash, f.updated_at, f.source_verification_receipt_hash, {projection}
        FROM feature_daily_current AS f
        JOIN feature_input_cutoff_set AS cutoffs USING (cutoff_set_hash)"""
    if layer is None:
        return rows
    owner = {factor_id: index for index, (_hash, axis) in enumerate(layer) for factor_id in axis}
    values = ", ".join(f'p{owner[factor_id]}."{factor_id}"' for factor_id in factors)
    cutoffs = ", ".join(
        f"'{factor_id}', json_extract(c{owner[factor_id]}.input_cutoffs_json, '$.\"{factor_id}\"')"
        for factor_id in sorted(factors)
    )
    identities = " || '|' || ".join(f"p{index}.row_hash" for index in range(len(layer)))
    updated = ", ".join(f"p{index}.updated_at" for index in range(len(layer)))
    joins = "".join(
        f"""
        JOIN feature_daily_current AS p{index}
          ON p{index}.listing_id = p0.listing_id AND p{index}.session_date = p0.session_date
         AND p{index}.catalog_hash = '{part_hash}'
        JOIN feature_input_cutoff_set AS c{index}
          ON c{index}.cutoff_set_hash = p{index}.cutoff_set_hash"""
        for index, (part_hash, _axis) in enumerate(layer)
        if index
    )
    return f"""{rows}
        WHERE f.catalog_hash <> '{catalog_hash}'
        UNION ALL
        SELECT p0.listing_id, p0.session_date, '{catalog_hash}',
               p0.raw_input_hash, p0.action_set_hash,
               p0.market_reference_revision, CAST(json_object({cutoffs}) AS VARCHAR),
               sha256('FeatureLayeredRow|{catalog_hash}|' || {identities}),
               greatest({updated}), CAST(NULL AS VARCHAR), {values}
        FROM feature_daily_current AS p0
        JOIN feature_input_cutoff_set AS c0 ON c0.cutoff_set_hash = p0.cutoff_set_hash{joins}
        WHERE p0.catalog_hash = '{layer[0][0]}'"""


def _key_rows_by_catalog(connection: duckdb.DuckDBPyConnection) -> None:
    """Key a table an earlier release keyed by listing and session alone by catalog too (V92).

    Copied once, every row and column kept in place; a table already keyed is left as it is.
    """
    key = connection.execute(
        """
        SELECT constraint_column_names FROM duckdb_constraints()
        WHERE schema_name = 'main' AND table_name = 'feature_daily_current'
          AND constraint_type = 'PRIMARY KEY'
        """
    ).fetchone()
    if key is None or list(key[0]) == ["listing_id", "session_date", "catalog_hash"]:
        return
    definitions = ", ".join(
        f'"{name}" {kind}{" NOT NULL" if not_null else ""}'
        for _position, name, kind, not_null, _default, _key in connection.execute(
            "PRAGMA table_info('feature_daily_current')"
        ).fetchall()
    )
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            "CREATE TEMPORARY TABLE feature_rows_before_keying AS "
            "SELECT * FROM feature_daily_current"
        )
        connection.execute("DROP TABLE feature_daily_current")
        connection.execute(
            f"CREATE TABLE feature_daily_current ({definitions}, "
            "PRIMARY KEY (listing_id, session_date, catalog_hash))"
        )
        connection.execute(
            "INSERT INTO feature_daily_current SELECT * FROM feature_rows_before_keying"
        )
        connection.execute("DROP TABLE feature_rows_before_keying")
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def assert_current_catalog(connection: duckdb.DuckDBPyConnection, *, catalog_hash: str) -> None:
    """Refuse a storage binding that differs from the required catalog hash."""
    row = connection.execute(
        "SELECT catalog_hash FROM feature_catalog_current WHERE singleton = TRUE"
    ).fetchone()
    if row is None or str(row[0]) != catalog_hash:
        raise ValueError("feature catalog hash does not match current storage binding")


__all__ = [
    "FeatureRowCatalogAuthorityError",
    "FeatureStorageLayout",
    "assert_current_catalog",
    "assert_rows_carry_installed_catalog",
    "canonical_cutoff_set",
    "cutoff_set_hash",
    "ensure_feature_current_schema",
    "feature_storage_layout",
    "layered_row_hash",
]
