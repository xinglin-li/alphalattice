"""Admission and projected reads for sealed historical Risk covariance."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path
from threading import Lock
from typing import Any

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.log import attach, uuid7
from alphalattice.kernel.data_layer.root import data_root

_RISK_FORMATION = "risk_covariance_formation"
_writing = Lock()


def _risk_covariance_profiles(
    read: Callable[..., Any], chunk_hash: str, axis_hash: str
) -> dict[str, tuple[str, str, int, str]]:
    rows = read(
        "SELECT valid_from, artifact_id, body_state, row_count,"
        " json_extract(shape, '$.matrix_hash') FROM artifact"
        " WHERE content_digest = ? AND json_extract(shape, '$.axis_hash') = ?"
        " AND type_id = (SELECT definition_id FROM definition_name"
        " WHERE family = 'artifact_type' AND name = ?)",
        (chunk_hash, axis_hash, _RISK_FORMATION),
    )
    profiles = {
        str(day): (str(identity), str(state), int(count), str(seal))
        for day, identity, state, count, seal in rows
    }
    if len(profiles) != len(rows):
        raise DataError(
            "the Risk chunk repeats a formation profile",
            code="risk_research.covariance_chunk_identity_invalid",
        )
    return profiles


def _coordinate_counts(cursor: Any, identities: Sequence[str]) -> list[tuple[str, int, int]]:
    return cursor.execute(
        """
        SELECT a.artifact_id,
            (SELECT count(*) FROM lake.results.risk_covariance_order o
             WHERE o.artifact_id = a.artifact_id),
            (SELECT count(*) FROM lake.results.risk_covariance_cell c
             WHERE c.artifact_id = a.artifact_id)
        FROM unnest(?::VARCHAR[]) AS a(artifact_id)
        """,
        (list(identities),),
    ).fetchall()


def read_risk_covariance(
    chunk_hash: str,
    axis_hash: str,
    sessions: Sequence[str],
    matrix_hashes: Sequence[str],
    listings: Sequence[str],
    *,
    root: Path | None = None,
    retry_unadmitted: bool = False,
) -> Any | None:
    """Project sealed Risk coordinates, leaving an unadmitted resume for its packed reader."""
    import numpy as np  # lazy: Risk projection needs the optional data extra

    where = data_root() if root is None else root.resolve()
    log = history.holding() if where == data_root() else None
    if log is not None:
        profiles = _risk_covariance_profiles(log.read, chunk_hash, axis_hash)
    elif (where / "machine" / "catalog.sqlite").is_file():
        with closing(attach(where)) as connection:
            profiles = _risk_covariance_profiles(
                lambda sql, args: connection.execute(sql, args).fetchall(), chunk_hash, axis_hash
            )
    else:
        return None
    if not profiles:
        return None

    row, col = np.tril_indices(len(listings))
    width = len(row)
    ids = []
    for day, seal in zip(sessions, matrix_hashes, strict=True):
        profile = profiles.get(day)
        if profile is None or profile[1:] != ("present", width, seal):
            raise DataError(
                f"the held Risk formation at {day} differs from its declaration",
                code="risk_research.covariance_chunk_tampered",
            )
        ids.append(profile[0])

    with history.catalog_session(where) as cursor:
        if cursor is None:
            raise DataError(
                "the catalog holding this Risk chunk is unavailable",
                code="risk_research.covariance_chunk_incomplete",
            )
        result = cursor.sql(
            """
            SELECT slot.ordinality AS formation, i.position AS row, j.position AS col, c.value
            FROM unnest(?::VARCHAR[]) WITH ORDINALITY AS slot(artifact_id, ordinality)
            JOIN lake.results.risk_covariance_cell c USING (artifact_id)
            JOIN lake.results.risk_covariance_order i
              ON i.artifact_id = c.artifact_id AND i.listing_id = c.listing_i
            JOIN lake.results.risk_covariance_order j
              ON j.artifact_id = c.artifact_id AND j.listing_id = c.listing_j
            WHERE i.listing_id = list_extract(?::UUID[], i.position + 1)
              AND j.listing_id = list_extract(?::UUID[], j.position + 1)
            ORDER BY formation, row, col
            """,
            params=(ids, list(listings), list(listings)),
        ).to_arrow_table()
        if (
            retry_unadmitted
            and not len(result)
            and len(profiles) == len(ids)
            and all(order == cells == 0 for _, order, cells in _coordinate_counts(cursor, ids))
        ):
            return None

    if (
        len(result) != len(ids) * width
        or not np.array_equal(
            result["formation"].to_numpy(), np.repeat(np.arange(1, len(ids) + 1), width)
        )
        or not np.array_equal(result["row"].to_numpy(), np.tile(row, len(ids)))
        or not np.array_equal(result["col"].to_numpy(), np.tile(col, len(ids)))
    ):
        raise DataError(
            "the stored Risk coordinates do not cover the declared lower triangles",
            code="risk_research.covariance_chunk_tampered",
        )
    return result["value"].to_numpy().reshape(len(ids), width)


def risk_covariance_held(chunk_hash: str, axis_hash: str, workspace_id: str) -> int:
    """How many formations of this chunk and axis the workspace already holds."""
    log = history.holding()
    if log is None:
        return 0
    rows = log.read(
        "SELECT count(*) FROM artifact a JOIN live_ref r USING (artifact_id)"
        " WHERE a.content_digest = ? AND json_extract(a.shape, '$.axis_hash') = ?"
        " AND a.body_state = 'present' AND r.root_kind = 'workspace' AND r.root_id = ?"
        " AND a.type_id = (SELECT definition_id FROM definition_name"
        " WHERE family = 'artifact_type' AND name = ?)",
        (chunk_hash, axis_hash, workspace_id, _RISK_FORMATION),
    )
    return int(rows[0][0])


def unregistered_risk_listings(listings: Sequence[str]) -> list[str] | None:
    """List missing world listings, or None when there is no catalog to inspect."""
    with history.catalog_session() as cursor:
        if cursor is None:
            return None
        missing = cursor.execute(
            "SELECT unnest(?::UUID[]) EXCEPT SELECT listing_id FROM lake.world.listing",
            (list(listings),),
        ).fetchall()
    return sorted(str(row[0]) for row in missing)


def require_risk_listings(listings: Sequence[str]) -> None:
    """Refuse Risk coordinates whose world listing the catalog has never held."""
    unknown = unregistered_risk_listings(listings)
    if unknown is None:
        raise DataError("the Risk catalog is unavailable", code="data_layer.history_unavailable")
    if unknown:
        raise DataError(
            f"the Risk formation names unregistered listings: {unknown}",
            code="data_layer.lineage_unregistered",
        )


def write_risk_covariance(
    chunk_hash: str,
    axis_hash: str,
    sessions: Sequence[str],
    matrix_hashes: Sequence[str],
    listings: Sequence[str],
    packed: Any,
    *,
    workspace_id: str | None = None,
) -> None:
    """Admit exact lower-triangle cells and their ordered listing axis in one snapshot."""
    import numpy as np  # lazy: Risk admission needs the optional data extra
    import pyarrow as pa  # lazy: only catalog admission builds the Arrow rows

    from alphalattice.kernel.data_layer.store import stage  # lazy: open the data extra at admission

    names = {
        "storage": "table",
        "table": "results.risk_covariance_cell",
        "measures": [],
        "keep": "while_active",
    }
    history.define(
        "artifact_type", _RISK_FORMATION, "One sealed historical Risk covariance.", names
    )
    with _writing:
        port = history.store()
        opened = history.opened_log()
        if port is None or opened is None:
            raise DataError(
                "open the Host's history before admitting Risk history",
                code="data_layer.history_unavailable",
            )
        log, workspace = opened
        workspace = workspace_id or workspace
        require_risk_listings(listings)
        existing = _risk_covariance_profiles(log.read, chunk_hash, axis_hash)
        width = packed.shape[1]
        version = log.read(
            "SELECT version FROM definition_current WHERE family = 'artifact_type' AND name = ?",
            (_RISK_FORMATION,),
        )[0][0]
        identities = []
        with log.unit():
            for day, seal in zip(sessions, matrix_hashes, strict=True):
                profile = existing.get(day)
                if profile is None:
                    identity = uuid7()
                    log.record(
                        "artifact.produced",
                        ("artifact", identity),
                        {
                            "type": _RISK_FORMATION,
                            "type_version": int(version),
                            "identity_kind": "content",
                            "storage": "table",
                            "meaning": "Sealed covariance for one Risk formation.",
                            "content_digest": chunk_hash,
                            "shape": {"axis_hash": axis_hash, "matrix_hash": seal},
                            "valid_from": day,
                            "valid_to": day,
                            "row_count": width,
                            "held": True,
                        },
                        workspace=workspace,
                        node="host",
                    )
                else:
                    identity, state, count, recorded = profile
                    if (state, count, recorded) != ("present", width, seal):
                        raise DataError(
                            f"the stored Risk formation at {day} differs from its declaration",
                            code="risk_research.covariance_chunk_identity_invalid",
                        )
                    if not log.read(
                        "SELECT 1 FROM live_ref WHERE artifact_id = ?"
                        " AND root_kind = 'workspace' AND root_id = ?",
                        (identity, workspace),
                    ):
                        log.record(
                            "artifact.held",
                            ("artifact", identity),
                            {},
                            workspace=workspace,
                            node="host",
                        )
                identities.append(identity)
        log.flush()

        held = set()
        if existing:
            ids = [profile[0] for profile in existing.values()]
            with history.catalog_session() as cursor:
                if cursor is None:
                    raise DataError(
                        "the Risk catalog is unavailable", code="data_layer.history_unavailable"
                    )
                counts = _coordinate_counts(cursor, ids)
            for identity, order_count, cell_count in counts:
                if order_count == cell_count == 0:
                    continue  # the profile survived an interrupted first admission
                if (order_count, cell_count) != (len(listings), width):
                    raise DataError(
                        "the stored Risk formation has incomplete coordinates",
                        code="risk_research.covariance_chunk_identity_invalid",
                    )
                held.add(identity)
        pending = [index for index, identity in enumerate(identities) if identity not in held]
        if not pending:
            return
        names = pa.array(listings, type=pa.string())
        row, col = np.tril_indices(len(listings))
        ids = pa.array([identities[index] for index in pending], type=pa.string())
        groups = np.arange(len(pending), dtype=np.int32)
        values = packed if len(pending) == len(identities) else packed[pending]
        cells = pa.table(
            {
                "artifact_id": ids.take(np.repeat(groups, width)),
                "listing_i": names.take(np.tile(row, len(pending))),
                "listing_j": names.take(np.tile(col, len(pending))),
                "value": values.ravel(),
            }
        )
        order = pa.table(
            {
                "artifact_id": ids.take(np.repeat(groups, len(listings))),
                "position": np.tile(np.arange(len(listings), dtype=np.int32), len(pending)),
                "listing_id": names.take(np.tile(np.arange(len(listings)), len(pending))),
            }
        )
        with stage(
            port.root, {"risk_covariance_cell": cells, "risk_covariance_order": order}
        ) as staged:
            port.commit(staged)
