"""Record and read catalog-backed Portfolio results."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any, Final

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.history_runtime import (
    _array_reader,
    _once,
    _open,
    _produce_result_artifact,
    _record_result_count,
    _result_writer,
    catalog_projection,
    holding,
    store,
)
from alphalattice.foundation.data_layer.log import attach, uuid7
from alphalattice.foundation.data_layer.root import (
    data_root,
)


def _portfolio_result_profile(read: Callable[..., Any], kind: str, digest: str) -> Any:
    rows = read(
        "SELECT a.artifact_id, a.body_state, a.shape FROM artifact a JOIN definition d"
        " ON d.definition_id = a.type_id AND d.version = a.type_version"
        " WHERE d.family = 'artifact_type' AND d.name = ?"
        " AND a.content_digest = ?",
        (kind, digest),
    )
    if len(rows) > 1:
        raise DataError("duplicate benchmark profile", code="data_layer.benchmark_invalid")
    return rows[0] if rows else None


def require_portfolio_result_store() -> None:
    """Refuse Portfolio publication before any artifact when the Host has no results writer."""
    if holding() is None or store() is None:
        raise DataError("Host data root is unavailable", code="data_layer.history_unavailable")


def write_portfolio_result(
    kind: str,
    digest: str,
    metadata: Mapping[str, Any],
    table: Any,
    *,
    result_table: str,
    upgrade_owner: tuple[Any, Path] | None = None,
    upgrade_workspace: str | None = None,
) -> str:
    """Admit one Portfolio artifact's rows and retain its lineage in its profile."""
    import pyarrow as pa  # lazy: only catalog result admission needs the data extra

    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numeric extras
    from alphalattice.foundation.data_layer.store import stage  # lazy: the data extra is optional

    if result_table not in {"series", "holding"}:
        raise ValueError("portfolio result table is invalid")
    selected, order = (
        ("name, session_date, value", "name, session_date")
        if result_table == "series"
        else ("as_of, listing_id, weight, shares", "as_of, listing_id")
    )

    with _once:
        log, root, port = _result_writer(upgrade_owner)
        workspace = upgrade_workspace if upgrade_owner is not None else _open["workspace"]
        if not log.read(
            "SELECT 1 FROM definition WHERE family = 'artifact_type' AND name = ?",
            (kind,),
        ):
            key = {"family": "artifact_type", "name": kind, "version": 1}
            body = key | {
                "spec": {
                    "storage": "table",
                    "table": "results." + result_table,
                    "measures": ["n_obs"],
                    "keep": "while_active",
                },
                "status": "standard",
                "owner": "host",
                "text": "A sealed Portfolio series artifact.",
            }
            log.record(
                "definition.registered",
                ("definition", "artifact_type/" + kind),
                key,
                text=json.dumps(body, sort_keys=True),
                node="host",
            )
        profile = _portfolio_result_profile(log.read, kind, digest)
        artifact = profile[0] if profile else uuid7()
        if profile and (profile[1] != "present" or json.loads(profile[2]) != dict(metadata)):
            raise DataError("benchmark profile differs", code="data_layer.benchmark_invalid")
        if not profile:
            _produce_result_artifact(
                log,
                artifact,
                kind,
                digest,
                metadata,
                table.num_rows,
                "The sealed series of a Portfolio artifact.",
                workspace=workspace,
            )
        count = port.lake.execute(
            f"SELECT count(*) FROM lake.results.{result_table} WHERE artifact_id = ?", (artifact,)
        ).fetchone()[0]
        if count == 0:
            with stage(
                root,
                result_table,
                None,
                table.append_column("artifact_id", pa.repeat(pa.scalar(artifact), table.num_rows)),
            ) as staged:
                port.commit(staged)
        elif count != table.num_rows:
            raise DataError("benchmark body is incomplete", code="data_layer.benchmark_invalid")
        else:
            held = (
                port.lake.execute(
                    f"SELECT {selected} FROM lake.results.{result_table}"
                    f" WHERE artifact_id = ? ORDER BY {order}",
                    (artifact,),
                )
                .to_arrow_table()
                .cast(table.schema)
            )
            if not held.equals(table):
                raise DataError(
                    "benchmark identity has different values", code="data_layer.benchmark_invalid"
                )
        arrays._hold(log, str(artifact), workspace)
        _record_result_count(log, artifact, table.num_rows)
        return str(artifact)


def portfolio_result_held(kind: str, digest: str, workspace: str | None) -> bool:
    """Whether this workspace keeps the catalog artifact with this type and content."""
    if workspace is None:
        return True
    with _array_reader() as (_, read):
        profile = _portfolio_result_profile(read, kind, digest)
        return bool(
            profile
            and profile[1] == "present"
            and read(
                "SELECT 1 FROM live_ref WHERE artifact_id = ? AND root_kind = 'workspace'"
                " AND root_id = ?",
                (profile[0], workspace),
            )
        )


def read_portfolio_result(
    kind: str, digest: str, *, result_table: str
) -> tuple[dict[str, Any], Any]:
    """Read a registered Portfolio profile and its catalog rows."""
    if result_table not in _PORTFOLIO_PROJECTIONS:
        raise ValueError("portfolio result table is invalid")
    profile = _portfolio_read_profile(kind, digest)
    artifact = str(uuid.UUID(str(profile[0])))
    with catalog_projection(
        f"portfolio_{result_table}", _PORTFOLIO_PROJECTIONS[result_table], (artifact,)
    ) as result:
        if result is None:
            raise DataError("portfolio catalog unavailable", code="data_layer.history_unavailable")
        table = result.to_arrow_table()
    return json.loads(profile[2]), table


def _portfolio_read_profile(kind: str, digest: str) -> Any:
    log = _open.get("log")
    root = _open.get("root") or data_root()
    if log is not None:
        profile = _portfolio_result_profile(log.read, kind, digest)
    elif not (root / "machine" / "log.sqlite").is_file():
        profile = None  # a root whose log was never made holds no result
    else:
        with closing(attach(root)) as reader:
            profile = _portfolio_result_profile(
                lambda sql, args: reader.execute(sql, args).fetchall(), kind, digest
            )
    if profile is None or profile[1] != "present":
        raise DataError(
            f"portfolio result {digest} is not in the catalog", code="data_layer.benchmark_missing"
        )
    return profile


_PORTFOLIO_PROJECTIONS: Final = {
    "series": "SELECT name, session_date, value FROM lake.results.series"
    " WHERE artifact_id = ? ORDER BY name, session_date",
    "holding": "SELECT as_of, listing_id, weight, shares FROM lake.results.holding"
    " WHERE artifact_id = ? ORDER BY as_of, listing_id",
}

_PORTFOLIO_PAIR: Final = (
    "SELECT 'series' AS kind, name, session_date, value,"
    " NULL::DATE AS as_of, NULL::VARCHAR AS listing_id, NULL::DOUBLE AS weight,"
    " NULL::DOUBLE AS shares FROM lake.results.series WHERE artifact_id = ? UNION ALL"
    " SELECT 'holding', NULL, NULL, NULL, as_of,"
    " CAST(listing_id AS VARCHAR), weight, shares"
    " FROM lake.results.holding WHERE artifact_id = ?"
)


def read_portfolio_pair(series_digest: str, holding_digest: str) -> tuple[Any, Any]:
    """Read a ledger's series and its report's holdings in one catalog query."""
    import pyarrow.compute as pc  # lazy: only catalog result reads need the data extra

    series = _portfolio_read_profile("portfolio_execution_series", series_digest)
    holding = _portfolio_read_profile("portfolio_report_holding", holding_digest)
    ids = (str(uuid.UUID(str(series[0]))), str(uuid.UUID(str(holding[0]))))
    with catalog_projection("portfolio_pair", _PORTFOLIO_PAIR, ids) as result:
        if result is None:
            raise DataError("portfolio catalog unavailable", code="data_layer.history_unavailable")
        table = result.to_arrow_table()
    first = table.filter(pc.equal(table["kind"], "series")).select(
        ["name", "session_date", "value"]
    )
    second = table.filter(pc.equal(table["kind"], "holding")).select(
        ["as_of", "listing_id", "weight", "shares"]
    )
    return (
        json.loads(series[2]),
        first.sort_by([("name", "ascending"), ("session_date", "ascending")]),
    ), (
        json.loads(holding[2]),
        second.sort_by([("as_of", "ascending"), ("listing_id", "ascending")]),
    )
