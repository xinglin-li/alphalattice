"""Record and read sealed Sector development results."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.log import attach, uuid7
from alphalattice.foundation.data_layer.root import (
    data_root,
    schema_text,
    statements,
)

if TYPE_CHECKING:
    from alphalattice.foundation.data_layer.store import StorePort
from alphalattice.foundation.data_layer.history_runtime import (
    _hold_existing_result,
    _once,
    _open,
    _produce_result_artifact,
    _result_writer,
    catalog_session,
)

_SECTOR_RESULT_FIELDS: Final = {
    "series": ("sector_development_series", "name, session_date, value", "name, session_date"),
    "sector_evaluation": (
        "sector_development_evaluation",
        "sector_index, evaluated_count, mean_error, mean_squared_error",
        "sector_index",
    ),
}


def _sector_result_profile(
    read: Callable[..., Any], digest: str, table: str, experiment: str | None = None
) -> Any:
    kind, _, _ = _SECTOR_RESULT_FIELDS[table]
    sql = (
        "SELECT a.artifact_id, a.body_state, a.shape FROM artifact a JOIN definition d"
        " ON d.definition_id = a.type_id AND d.version = a.type_version"
        " WHERE d.family = 'artifact_type' AND d.name = ? AND a.content_digest = ?"
    )
    params = (kind, digest)
    if experiment is not None:
        sql += " AND json_extract(a.shape, '$.experiment') = ?"
        params += (experiment,)
    rows = read(sql + " ORDER BY a.artifact_id", params)
    if len(rows) > 1 and (table == "series" or experiment is not None):
        raise DataError("duplicate Sector result", code="data_layer.sector_result_invalid")
    return rows[0] if rows else None


def _sector_result_rows(connection: Any, table: str, artifact: str) -> Any:
    _, columns, order = _SECTOR_RESULT_FIELDS[table]
    return connection.sql(
        f"SELECT {columns} FROM lake.results.{table} WHERE artifact_id = ? ORDER BY {order}",
        params=(artifact,),
    ).to_arrow_table()


def _ensure_sector_evaluation_table(port: StorePort) -> None:
    """Add the Sector-owned table to a catalog created before this family."""
    if port.lake.execute(
        "SELECT 1 FROM duckdb_tables() WHERE database_name = 'lake'"
        " AND schema_name = 'results' AND table_name = 'sector_evaluation'"
    ).fetchone():
        return
    ddl = next(
        sql
        for sql in statements(schema_text("catalog.sql"))
        if sql.partition("(")[0].strip().endswith("results.sector_evaluation")
    )
    port.lake.execute(
        ddl.replace("results.sector_evaluation", "lake.results.sector_evaluation", 1)
    ).complete()


def write_sector_results(
    results: Iterable[tuple[str, str, Any]],
    *,
    experiment: str | None = None,
    upgrade_owner: tuple[Any, Path] | None = None,
) -> dict[str, str]:
    """Admit an experiment's new grids and evaluation in one StorePort snapshot."""
    import pyarrow as pa  # lazy: the data extra is optional.

    from alphalattice.foundation.data_layer.store import stage  # lazy: data extra

    with _once:
        log, root, port = _result_writer(upgrade_owner)
        artifacts: dict[str, str] = {}
        pending: dict[str, list[Any]] = {}
        for digest, table, rows in results:
            kind, _, _ = _SECTOR_RESULT_FIELDS[table]
            owner = (experiment or digest) if table == "sector_evaluation" else None
            descriptor = (
                {"descriptor": digest}
                if owner is None
                else {"descriptor": digest, "experiment": owner}
            )
            if not log.read(
                "SELECT 1 FROM definition WHERE family = 'artifact_type' AND name = ?", (kind,)
            ):
                if owner is not None:
                    _ensure_sector_evaluation_table(port)
                key = {"family": "artifact_type", "name": kind, "version": 1}
                body = key | {
                    "spec": {
                        "storage": "table",
                        "table": f"results.{table}",
                        "keep": "while_active",
                    },
                    "status": "standard",
                    "owner": "host",
                    "text": "A sealed Sector result.",
                }
                log.record(
                    "definition.registered",
                    ("definition", "artifact_type/" + kind),
                    key,
                    text=json.dumps(body, sort_keys=True),
                    node="host",
                )
            profile = _sector_result_profile(log.read, digest, table, owner)
            artifact = profile[0] if profile else uuid7()
            if profile is None:
                _produce_result_artifact(
                    log,
                    artifact,
                    kind,
                    digest,
                    descriptor,
                    rows.num_rows,
                    "The values of a Sector development result.",
                    workspace=_open["workspace"],
                    held=_open["workspace"] is not None,
                )
            else:
                if profile[1] != "present" or json.loads(profile[2]) != descriptor:
                    raise DataError(
                        "Sector result descriptor differs", code="data_layer.sector_result_invalid"
                    )
                _hold_existing_result(log, artifact)
                held = _sector_result_rows(port.lake, table, artifact).cast(rows.schema)
                if held.num_rows:
                    if not held.equals(rows):
                        raise DataError(
                            "Sector result body differs", code="data_layer.sector_result_invalid"
                        )
                    artifacts[digest] = artifact
                    continue
            artifacts[digest] = artifact
            if rows.num_rows:
                pending.setdefault(table, []).append(
                    rows.append_column("artifact_id", pa.repeat(pa.scalar(artifact), rows.num_rows))
                )
        if pending:
            with stage(
                root, {table: pa.concat_tables(rows) for table, rows in pending.items()}
            ) as admission:
                port.commit(admission)
        return artifacts


def read_sector_result(
    digest: str, table: str = "series", *, experiment: str | None = None
) -> tuple[dict[str, Any], Any]:
    """Read a Sector result descriptor and its catalog rows."""
    log = _open.get("log")
    if log is not None:
        profile = _sector_result_profile(log.read, digest, table, experiment)
    else:
        with closing(attach(_open.get("root") or data_root())) as reader:
            profile = _sector_result_profile(
                lambda sql, args: reader.execute(sql, args).fetchall(), digest, table, experiment
            )
    if profile is None or profile[1] != "present":
        raise DataError("Sector result is missing", code="data_layer.sector_result_missing")
    with catalog_session() as connection:
        if connection is None:
            raise DataError("Sector catalog is unavailable", code="data_layer.history_unavailable")
        rows = _sector_result_rows(connection, table, profile[0])
    return json.loads(profile[2]), rows
