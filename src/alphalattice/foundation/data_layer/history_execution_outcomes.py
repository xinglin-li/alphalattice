"""Record and read catalog-backed execution outcomes."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.history_runtime import (
    _hold_existing_result,
    _lock,
    _once,
    _open,
    _record_result_count,
    catalog_projection,
    recording,
    store,
)
from alphalattice.foundation.data_layer.log import attach, uuid7
from alphalattice.foundation.data_layer.reading import (
    reading_session,
)
from alphalattice.foundation.data_layer.root import (
    data_root,
)


def write_execution_outcome(
    content_hash: str, table: Any, *, upgrade_owner: tuple[Any, Path] | None = None
) -> str:
    """Admit one immutable outcome chunk through the Host's StorePort and log its profile."""
    import pyarrow as pa  # lazy: the optional data extra, as DuckDB

    from alphalattice.foundation.data_layer.store import stage  # lazy: the data extra is optional

    with _once:
        port = store()  # first open takes the log guard before history's lock
        with _lock:
            log, root = _open.get("log"), _open.get("root")
            if log is None or root is None:
                raise DataError(
                    "the Host's data root is unavailable; open its history before publishing",
                    code="data_layer.history_unavailable",
                )
            if upgrade_owner is not None and (
                log is not upgrade_owner[0] or root.resolve() != upgrade_owner[1].resolve()
            ):
                raise DataError(
                    "the upgrade and outcome writer do not own the same data root",
                    code="upgrade.refused",
                )
            if port is None:
                raise DataError(
                    "the Host's StorePort is unavailable", code="data_layer.history_unavailable"
                )
        type_name = "execution_outcome"
        exists = log.read(
            "SELECT 1 FROM definition WHERE family = 'artifact_type' AND name = ?", (type_name,)
        )
        if not exists:
            key = {"family": "artifact_type", "name": type_name, "version": 1}
            spec = {
                "storage": "table",
                "table": "results.execution_outcome",
                "measures": ["n_obs"],
                "keep": "while_active",
            }
            body = key | {
                "spec": spec,
                "status": "standard",
                "owner": "host",
                "text": "A sealed execution-outcome chunk.",
            }
            log.record(
                "definition.registered",
                ("definition", "artifact_type/" + type_name),
                key,
                text=json.dumps(body, sort_keys=True),
                node="host",
            )
        profile = _execution_outcome_profile(log.read, content_hash)
        artifact = profile[0] if profile else uuid7()
        if profile and profile[1] != "present":
            raise DataError("the outcome body was collected", code="data_layer.outcome_collected")
        if not profile:
            current = log.read(
                "SELECT max(version) FROM definition WHERE family = 'artifact_type' AND name = ?",
                (type_name,),
            )
            attrs = {
                "type": type_name,
                "type_version": int(current[0][0]),  # the registered one: no version defaults
                "identity_kind": "content",
                "storage": "table",
                "meaning": "The realized execution outcomes for a sealed chunk.",
                "content_digest": content_hash,
                "row_count": table.num_rows,
                "held": _open["workspace"] is not None,  # its workspace keeps it (governance)
            }
            log.record(
                "artifact.produced",
                ("artifact", artifact),
                attrs,
                workspace=_open["workspace"],
                node="host",
            )
            log.flush()
        else:  # another workspace's equal outcome: this one keeps it too
            _hold_existing_result(log, artifact)
        count = port.lake.execute(
            "SELECT count(*) FROM lake.results.execution_outcome WHERE artifact_id = ?",
            (artifact,),
        ).fetchone()[0]
        if count == 0:
            with stage(
                root,
                "execution_outcome",
                None,
                table.append_column("artifact_id", pa.repeat(pa.scalar(artifact), table.num_rows)),
            ) as staged:
                port.commit(staged)
        elif count != table.num_rows:
            raise DataError("the outcome body is incomplete", code="data_layer.outcome_incomplete")
        else:
            columns = ", ".join('"' + name.replace('"', '""') + '"' for name in table.column_names)
            held = (
                port.lake.execute(
                    f"SELECT {columns} FROM lake.results.execution_outcome WHERE artifact_id = ? "
                    "ORDER BY formation_session, listing_id",
                    (artifact,),
                )
                .to_arrow_table()
                .cast(table.schema)
            )
            if not held.equals(table):
                raise DataError(
                    "the outcome identity is already bound to different values",
                    code="data_layer.outcome_incomplete",
                )
        _record_result_count(log, artifact, table.num_rows)
        return artifact


def _execution_outcome_profile(read: Callable[..., Any], content_hash: str) -> Any:
    """Find the one registered artifact for an old chunk digest, including after a restart."""
    rows = read(
        "SELECT a.artifact_id, a.body_state FROM artifact a JOIN definition d"
        " ON d.definition_id = a.type_id AND d.version = a.type_version"
        " WHERE d.family = 'artifact_type' AND d.name = 'execution_outcome'"
        " AND a.content_digest = ?",
        (content_hash,),
    )
    if len(rows) > 1:
        raise DataError(
            "the outcome digest names multiple artifacts", code="data_layer.outcome_incomplete"
        )
    return rows[0] if rows else None


def execution_outcome_artifact_id(content_hash: str) -> str:
    """Find the registered artifact named by a legacy outcome content digest."""
    log = _open.get("log")
    if log is not None:
        profile = _execution_outcome_profile(log.read, content_hash)
    else:
        with closing(attach(_open.get("root") or data_root())) as log_reader:
            profile = _execution_outcome_profile(
                lambda sql, args: log_reader.execute(sql, args).fetchall(), content_hash
            )
    if profile is None or profile[1] != "present":
        raise DataError("the outcome body is not in the catalog", code="data_layer.outcome_missing")
    return str(profile[0])


def read_execution_outcome(content_hash: str, columns: list[str] | None = None) -> Any:
    """Read the outcome body through the data root's catalog, including from non-Host processes."""
    selected = (
        "* EXCLUDE (artifact_id)"
        if columns is None
        else ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
    )
    query = (
        f"SELECT {selected} FROM lake.results.execution_outcome "
        "WHERE artifact_id = ? ORDER BY formation_session, listing_id"
    )
    in_host = recording()
    connection = (
        _open.get("duckdb")
        if in_host
        else reading_session(_open.get("root") or data_root(), log=False)
    )
    if connection is None:
        raise DataError("the outcome catalog is unavailable", code="data_layer.history_unavailable")
    if in_host:
        artifact = execution_outcome_artifact_id(content_hash)
        with catalog_projection("execution_outcome", query, (artifact,)) as result:
            if result is None:
                raise DataError(
                    "the outcome catalog is unavailable", code="data_layer.history_unavailable"
                )
            table = result.to_arrow_table()
    else:
        with closing(connection):
            artifact = execution_outcome_artifact_id(content_hash)
            table = connection.execute(query, (artifact,)).to_arrow_table()
    if table.num_rows == 0:
        raise DataError("the outcome body is not in the catalog", code="data_layer.outcome_missing")
    return table if columns is None else table.select(columns)
