"""Record and read Alpha development measures and result chunks."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from contextlib import closing
from typing import Any

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.history_runtime import (
    _once,
    _open,
    _query,
    catalog_session,
    opened_log,
    recording,
    store,
)
from alphalattice.kernel.data_layer.log import HostLog, attach, uuid7
from alphalattice.kernel.data_layer.root import (
    data_root,
    seed_rows,
)


def _register_result_definitions(
    log: HostLog,
    kind: str,
    measures: set[str],
) -> None:
    """Install the seed's result definitions in a data root created before they shipped."""
    needed = {("artifact_type", kind), *(("metric", name) for name in measures)}
    present = log.read(
        "SELECT family, name FROM definition WHERE family IN ('artifact_type', 'metric')"
        " AND version = 1"
    )
    missing = needed.difference((row[0], row[1]) for row in present)
    if not missing:
        return
    for family, name, version, _, _, _, spec, text, *_ in seed_rows():
        if (family, name) in missing:
            key = {"family": family, "name": name, "version": version}
            body = key | {
                "spec": json.loads(spec),
                "status": "standard",
                "owner": "host",
                "text": text,
            }
            log.record(
                "definition.registered",
                ("definition", f"{family}/{name}"),
                key,
                text=json.dumps(body, sort_keys=True),
                node="host",
            )


type AlphaDevelopmentMeasures = list[tuple[str, int, dict[str, str], float | int | None]]


def record_alpha_development_summary(report_hash: str, measures: AlphaDevelopmentMeasures) -> None:
    """Record the measures of a sealed Alpha report under its own durable profile."""
    opened = opened_log()
    if opened is not None:
        log, workspace = opened
        with log.unit():
            _record_alpha_summary(log, workspace, report_hash, measures)
        log.flush()


def _record_alpha_summary(
    log: HostLog,
    workspace: str,
    report_hash: str,
    measures: AlphaDevelopmentMeasures,
) -> None:
    kind = "alpha_development_summary"
    if log.read(
        "SELECT 1 FROM artifact WHERE type_id = ("
        " SELECT definition_id FROM definition_name WHERE family = 'artifact_type' AND name = ?"
        ") AND content_digest = ?",
        (kind, report_hash),
    ):
        return
    _register_result_definitions(log, kind, {name for name, *_ in measures})
    artifact = ("artifact", uuid7())
    log.record(
        "artifact.produced",
        artifact,
        {
            "type": kind,
            "type_version": 1,
            "identity_kind": "content",
            "storage": "none",
            "meaning": "The measures of a sealed Alpha candidate development report.",
            "content_digest": report_hash,
        },
        workspace=workspace,
        node="host",
    )
    log.record(
        "result.recorded", artifact, {"measures": measures}, workspace=workspace, node="host"
    )


_ALPHA_CHUNK = "alpha_development_chunk"

_ALPHA_LANES = ("target", "economic_return", "score")

_ALPHA_CHUNK_PROFILES = (
    "SELECT content_digest, artifact_id, body_state, shape, row_count FROM artifact"
    " WHERE type_id = (SELECT definition_id FROM definition_name"
    " WHERE family = 'artifact_type' AND name = ?) AND content_digest IN ({slots})"
)

_ALPHA_BREAKDOWN = """
    WITH named AS (
        SELECT artifact_id, metric_id, metric_version, value,
               json_transform(dims, '{"row_index":"BIGINT","listing_id":"VARCHAR",
                   "formation_session":"DATE","feature_available":"BOOLEAN",
                   "outcome_available":"BOOLEAN","feature_row_hash":"VARCHAR",
                   "causal_outcome_row_hash":"VARCHAR","availability":"VARCHAR"}') AS d
        FROM lake.results.breakdown
        WHERE artifact_id BETWEEN ? AND ? AND list_contains(?::VARCHAR[], artifact_id)
    ), lane AS (SELECT ? AS target, ? AS economic_return, ? AS score)
    SELECT artifact_id, d.row_index AS row_index,
           max(d.listing_id) AS listing_id,
           max(d.formation_session) AS formation_session,
           bool_or(d.feature_available) AS feature_available,
           bool_or(d.outcome_available) AS outcome_available,
           max(d.feature_row_hash) AS feature_row_hash,
           max(d.causal_outcome_row_hash) AS causal_outcome_row_hash,
           max(d.availability) AS availability,
           max(value) FILTER (WHERE metric_id = lane.target) AS target,
           max(value) FILTER (WHERE metric_id = lane.economic_return)
               AS economic_return,
           max(value) FILTER (WHERE metric_id = lane.score) AS score,
           count(*) AS n_lanes,
           bool_and(metric_version = 1) AS v1,
           bit_or(CASE metric_id
               WHEN lane.target THEN 1
               WHEN lane.economic_return THEN 2
               WHEN lane.score THEN 4
               ELSE 8
           END) AS lanes
    FROM named CROSS JOIN lane
    GROUP BY artifact_id, d.row_index ORDER BY artifact_id, row_index
"""


def _alpha_chunk_profiles(
    read: Callable[..., Any],
    digests: Iterable[str],
) -> dict[str, Any]:
    named = tuple(dict.fromkeys(digests))
    if not named:
        return {}
    slots = ", ".join("?" for _ in named)
    rows = read(
        _ALPHA_CHUNK_PROFILES.format(slots=slots),
        (_ALPHA_CHUNK, *named),
    )
    profiles = {str(row[0]): row for row in rows}
    if len(profiles) != len(rows):
        raise ValueError("alpha_research.development_breakdown_profile_repeated")
    return profiles


def _alpha_lane_ids(read: Callable[..., Any]) -> dict[str, str]:
    names = tuple(f"alpha_development_{lane}" for lane in _ALPHA_LANES)
    registered = dict(
        read(
            "SELECT name, definition_id FROM definition_name"
            " WHERE family = 'metric' AND name IN (?, ?, ?)",
            names,
        )
    )
    return {lane: registered[name] for lane, name in zip(_ALPHA_LANES, names, strict=True)}


def _alpha_breakdown_rows(
    chunks: list[tuple[str, Any]], lane_ids: Mapping[str, str], lake: Any
) -> Any:
    """Encode a stage's dimensions and numeric lanes in one columnar projection."""
    import pyarrow as pa  # lazy: the data extra is optional
    import pyarrow.compute as pc  # lazy: the data extra is optional

    types = {
        "availability": pa.string(),
        "causal_outcome_row_hash": pa.string(),
        "feature_available": pa.bool_(),
        "feature_row_hash": pa.string(),
        "formation_session": pa.date32(),
        "listing_id": pa.string(),
        "outcome_available": pa.bool_(),
        "row_index": pa.int64(),
    }
    batches = []
    for artifact, table in chunks:
        count = table.num_rows
        nulls = {name: pa.nulls(count, type_) for name, type_ in types.items()}
        dims = {
            name: pc.cast(table[name], type_).combine_chunks()
            if name in table.column_names
            else nulls[name]
            for name, type_ in types.items()
        }
        anchor = nulls | {"row_index": dims["row_index"]}
        for name, metric in lane_ids.items():
            if name not in table.column_names:
                continue
            fields = anchor if name == "economic_return" else dims
            batches.append(
                pa.table(
                    {
                        "artifact_id": pa.repeat(pa.scalar(artifact), count),
                        "metric_id": pa.repeat(pa.scalar(metric), count),
                        "metric_version": pa.repeat(pa.scalar(1, pa.int32()), count),
                        "dimensions": pa.StructArray.from_arrays(
                            list(fields.values()), names=list(fields)
                        ),
                        "value": pc.cast(table[name], pa.float64()),
                    }
                )
            )
    with closing(lake.cursor()) as cursor:
        cursor.register("alpha_lane", pa.concat_tables(batches))
        try:
            return cursor.sql(
                "SELECT artifact_id, metric_id, metric_version,"
                " json_merge_patch('{}', to_json(dimensions))::VARCHAR AS dims, value"
                " FROM alpha_lane"
            ).to_arrow_table()
        finally:
            cursor.unregister("alpha_lane")


def write_alpha_development_chunks(
    chunks: Mapping[str, tuple[str, Any]],
    *,
    summaries: Mapping[str, AlphaDevelopmentMeasures] | None = None,
) -> None:
    """Record profiles and measures, then admit the stage's new detail in one snapshot."""
    if not (chunks or summaries) or not recording():
        return
    from alphalattice.kernel.data_layer.store import stage  # lazy: the data extra is optional

    with _once:
        port = store()
        if port is None:
            raise DataError(
                "the Host's data root is unavailable for Alpha development evidence",
                code="data_layer.history_unavailable",
            )
        log, workspace = port.log, _open["workspace"]
        profiles = _alpha_chunk_profiles(log.read, chunks)
        artifacts: dict[str, str] = {}
        with log.unit():
            for report_hash, measures in (summaries or {}).items():
                _record_alpha_summary(log, workspace, report_hash, measures)
            _register_result_definitions(
                log, _ALPHA_CHUNK, {f"alpha_development_{name}" for name in _ALPHA_LANES}
            )
            for digest, (seal, table) in chunks.items():
                shape = {
                    "arrow_schema": table.schema.serialize().to_pybytes().hex(),
                    "metadata_hash": seal,
                }
                profile = profiles.get(digest)
                if profile is not None:
                    if profile[2] != "present" or json.loads(profile[3]) != shape:
                        raise ValueError(
                            f"alpha_research.development_breakdown_profile_invalid: {digest}"
                        )
                    if profile[4] != table.num_rows:
                        raise ValueError(
                            f"alpha_research.development_breakdown_count_changed: {digest}"
                        )
                    artifacts[digest] = str(profile[1])
                    continue
                artifact = artifacts[digest] = uuid7()
                log.record(
                    "artifact.produced",
                    ("artifact", artifact),
                    {
                        "type": _ALPHA_CHUNK,
                        "type_version": 1,
                        "identity_kind": "content",
                        "storage": "table",
                        "meaning": "Sealed Alpha development evidence rows.",
                        "content_digest": digest,
                        "row_count": table.num_rows,
                        "shape": shape,
                        "held": True,
                    },
                    workspace=workspace,
                    node="host",
                )
        log.flush()
        ids = [artifacts[digest] for digest in profiles]
        existing = (
            dict(
                port.lake.execute(
                    "SELECT artifact_id, count(*) FROM lake.results.breakdown"
                    " WHERE list_contains(?::VARCHAR[], artifact_id) GROUP BY artifact_id",
                    (ids,),
                ).fetchall()
            )
            if ids
            else {}
        )
        pending = []
        for digest, (_, table) in chunks.items():
            artifact = artifacts[digest]
            held_rows = existing.get(artifact)
            if held_rows is None:
                pending.append((artifact, table))
            elif held_rows != table.num_rows * (1 + ("economic_return" in table.column_names)):
                raise ValueError(f"alpha_research.development_breakdown_count_changed: {digest}")
        if pending:
            rows = _alpha_breakdown_rows(pending, _alpha_lane_ids(log.read), port.lake)
            with stage(port.root, "breakdown", None, rows) as staged:
                port.commit(staged)


def read_alpha_development_chunks(digests: Iterable[str]) -> dict[str, Any]:
    """Read one projected breakdown for the supplied sealed chunks."""
    import pyarrow as pa  # lazy: the data extra is optional
    import pyarrow.compute as pc  # lazy: the data extra is optional

    named = tuple(dict.fromkeys(digests))
    if not named:
        return {}
    log = _open.get("log")
    if log is None:
        with closing(attach(data_root())) as reader:
            read = _query(reader)
            profiles = _alpha_chunk_profiles(read, named)
            lane_ids = _alpha_lane_ids(read) if profiles else {}
    else:
        profiles = _alpha_chunk_profiles(log.read, named)
        lane_ids = _alpha_lane_ids(log.read) if profiles else {}
    for digest in named:
        if digest not in profiles or profiles[digest][2] != "present":
            raise FileNotFoundError(f"Alpha development breakdown {digest}")
    with catalog_session() as cursor:
        if cursor is None:
            raise DataError(
                "the Alpha evidence catalog is unavailable", code="data_layer.history_unavailable"
            )
        ids = sorted(str(profile[1]) for profile in profiles.values())
        params = [ids[0], ids[-1], ids, *(lane_ids[name] for name in _ALPHA_LANES)]
        rows = cursor.sql(_ALPHA_BREAKDOWN, params=params).to_arrow_table()
    result = {}
    for digest, (_, artifact, _, raw_shape, expected) in profiles.items():
        shape = json.loads(raw_shape)
        schema = pa.ipc.read_schema(pa.BufferReader(bytes.fromhex(shape["arrow_schema"])))
        selected = rows.filter(pc.equal(rows["artifact_id"], artifact))
        has_returns = "economic_return" in schema.names
        lanes = 4 if "score" in schema.names else 3 if has_returns else 1
        if (
            selected.num_rows != expected
            or not pc.all(selected["v1"]).as_py()
            or not pc.all(pc.equal(selected["n_lanes"], 1 + has_returns)).as_py()
            or not pc.all(pc.equal(selected["lanes"], lanes)).as_py()
        ):
            raise ValueError(f"alpha_research.development_breakdown_count_changed: {digest}")
        result[digest] = (
            selected.select(schema.names)
            .cast(schema)
            .replace_schema_metadata(
                {b"alphalattice.metadata_hash": shape["metadata_hash"].encode()}
            )
        )
    return result
