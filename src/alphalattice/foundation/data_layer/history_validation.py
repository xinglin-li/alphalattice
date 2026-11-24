"""Record and read protected-validation claims and receipts through the Host history."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, Final

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.history_runtime import (
    _once,
    _open,
    catalog_session,
    define,
    store,
)
from alphalattice.foundation.data_layer.log import uuid7

_VALIDATION_TYPE: Final = "protected_validation_receipt"

_VALIDATION_CLAIM: Final = "protected_validation.claimed"

_VALIDATION_COLUMNS: Final = (
    "artifact_id",
    "permit_hash",
    "candidate_hash",
    "package_hash",
    "fixture_hash",
    "pre_protected_state_hash",
    "closure",
    "claim_limits",
    "reselection_disposition",
    "sealed_at",
    "sealed_offset_microseconds",
    "receipt_hash",
)


def validation_receipt_table(
    rows: Sequence[Mapping[str, Any]], artifacts: Mapping[str, str]
) -> Any:
    """Represent scalar, validated gate receipts as one typed Arrow batch."""
    import pyarrow as pa  # lazy: only catalog receipt admission needs Arrow

    columns = {name: [] for name in _VALIDATION_COLUMNS}
    for row in rows:  # each gate decision is already an individual sealed contract
        offset = row["sealed_at"].utcoffset()
        if offset is None:
            raise DataError("receipt has no clock offset", code="data_layer.validation_invalid")
        for name in _VALIDATION_COLUMNS:
            columns[name].append(
                artifacts[str(row["receipt_hash"])]
                if name == "artifact_id"
                else offset // timedelta(microseconds=1)
                if name == "sealed_offset_microseconds"
                else row[name]
            )
    columns["sealed_at"] = pa.array(columns["sealed_at"], type=pa.timestamp("us", tz="UTC"))
    columns["claim_limits"] = pa.array(columns["claim_limits"], type=pa.list_(pa.string()))
    return pa.table(columns)


def write_validation_receipts(
    rows: Sequence[Mapping[str, Any]],
    *,
    workspace: str | None = None,
    upgrade_owner: tuple[Any, Path] | None = None,
    owners: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str]:
    """Register receipt profiles, then admit all missing receipt rows in one results snapshot."""
    import pyarrow.compute as pc  # lazy: only catalog admission needs Arrow

    from alphalattice.foundation.data_layer.store import stage  # lazy: Host opens StorePort on use

    if not rows:
        return {}
    with _once:
        port = store()
        log, root = _open.get("log"), _open.get("root")
        owner = workspace or _open.get("workspace")
        if log is None or root is None or port is None or owner is None:
            raise DataError(
                "the Host's validation store is unavailable", code="data_layer.history_unavailable"
            )
        if upgrade_owner is not None and (
            log is not upgrade_owner[0] or Path(root).resolve() != upgrade_owner[1].resolve()
        ):
            raise DataError("upgrade owns another data root", code="upgrade.refused")
        if not log.read(
            "SELECT 1 FROM definition WHERE family = 'artifact_type' AND name = ?",
            (_VALIDATION_TYPE,),
        ):
            key = {"family": "artifact_type", "name": _VALIDATION_TYPE, "version": 1}
            body = key | {
                "spec": {
                    "storage": "table",
                    "table": "results.validation_receipt",
                    "measures": [],
                    "keep": "while_active",
                },
                "status": "standard",
                "owner": "host",
                "text": "A sealed protected validation receipt.",
            }
            log.record(
                "definition.registered",
                ("definition", "artifact_type/" + _VALIDATION_TYPE),
                key,
                text=json.dumps(body, sort_keys=True),
                node="host",
            )
        digests = [str(row["receipt_hash"]) for row in rows]
        if len(set(digests)) != len(digests):
            raise DataError("duplicate receipt identity", code="data_layer.validation_invalid")
        marks = ", ".join("?" for _ in digests)
        existing = log.read(
            "SELECT a.content_digest, a.artifact_id, a.body_state FROM artifact a JOIN definition d"
            " ON d.definition_id = a.type_id AND d.version = a.type_version"
            " WHERE d.family = 'artifact_type' AND d.name = ? AND a.content_digest IN ("
            + marks
            + ")",
            (_VALIDATION_TYPE, *digests),
        )
        found: dict[str, str] = {}
        for digest, artifact, state in existing:
            if digest in found or state != "present":
                raise DataError(
                    "receipt identity is duplicated or collected",
                    code="data_layer.validation_invalid",
                )
            found[str(digest)] = str(artifact)
        existing_ids = list(found.values())
        with log.unit():
            for row in rows:
                digest = str(row["receipt_hash"])
                artifact = found.get(digest)
                held_by = tuple(owners[digest]) if owners is not None else (str(owner),)
                if not held_by:
                    raise DataError(
                        "receipt has no workspace owner", code="data_layer.validation_invalid"
                    )
                new_artifact = artifact is None
                if new_artifact:
                    artifact = found[digest] = uuid7()
                    log.record(
                        "artifact.produced",
                        ("artifact", artifact),
                        {
                            "type": _VALIDATION_TYPE,
                            "type_version": 1,
                            "identity_kind": "content",
                            "storage": "table",
                            "meaning": "A sealed protected validation receipt.",
                            "content_digest": digest,
                            "shape": {"permit_hash": row["permit_hash"]},
                            "row_count": 1,
                            "held": True,
                        },
                        workspace=held_by[0],
                        node="host",
                    )
                for workspace_id in held_by[1:] if new_artifact else held_by:
                    if not log.read(
                        "SELECT 1 FROM live_ref WHERE artifact_id = ?"
                        " AND root_kind = 'workspace' AND root_id = ?",
                        (artifact, workspace_id),
                    ):
                        log.record(
                            "artifact.held",
                            ("artifact", artifact),
                            {},
                            workspace=workspace_id,
                            node="host",
                        )
        log.flush()
        expected = validation_receipt_table(rows, found)
        missing = expected
        if existing_ids:  # a newly minted artifact cannot already have catalog rows
            keys = ", ".join("?" for _ in existing_ids)
            present = port.lake.sql(
                "SELECT " + ", ".join(_VALIDATION_COLUMNS) + " FROM lake.results.validation_receipt"
                " WHERE artifact_id IN (" + keys + ") ORDER BY artifact_id",
                params=existing_ids,
            ).to_arrow_table()
            already = pc.is_in(expected["artifact_id"], value_set=present["artifact_id"])
            if present.num_rows and not expected.filter(already).sort_by("artifact_id").cast(
                present.schema
            ).equals(present):
                raise DataError(
                    "the receipt identity has different values",
                    code="data_layer.validation_invalid",
                )
            missing = expected.filter(pc.invert(already))
        if missing.num_rows:
            with stage(Path(root), {"validation_receipt": missing}) as staged:
                port.commit(staged)
        return found


def read_validation_receipts(
    field: str, value: str | Sequence[str], *, workspace: str | None = None
) -> Any | None:
    """Read a workspace's requested receipt bodies with one filtered Arrow catalog query."""
    if field not in {"permit_hash", "receipt_hash"}:
        raise ValueError("unsupported validation receipt key")
    values = [value] if isinstance(value, str) else list(value)
    if not values:
        return None
    log = _open.get("log")
    owner = workspace or _open.get("workspace")
    if log is None or owner is None:
        raise DataError(
            "the Host's validation history is unavailable", code="data_layer.history_unavailable"
        )
    marks = ", ".join("?" for _ in values)
    lookup = (
        "a.content_digest" if field == "receipt_hash" else "json_extract(a.shape, '$.permit_hash')"
    )
    ids = [
        str(row[0])
        for row in log.read(
            "SELECT a.artifact_id FROM artifact a JOIN definition d"
            " ON d.definition_id = a.type_id AND d.version = a.type_version"
            " JOIN live_ref r ON r.artifact_id = a.artifact_id"
            " WHERE d.family = 'artifact_type' AND d.name = ? AND a.body_state = 'present'"
            " AND r.root_kind = 'workspace' AND r.root_id = ? AND "
            + lookup
            + " IN ("
            + marks
            + ")",
            (_VALIDATION_TYPE, owner, *values),
        )
    ]
    if not ids:
        return None
    artifact_marks = ", ".join("?" for _ in ids)
    with catalog_session() as connection:
        if connection is None:
            raise DataError(
                "the validation catalog is unavailable", code="data_layer.history_unavailable"
            )
        return connection.sql(
            "SELECT " + ", ".join(_VALIDATION_COLUMNS) + " FROM lake.results.validation_receipt"
            " WHERE " + field + " IN (" + marks + ") AND artifact_id IN (" + artifact_marks + ")",
            params=[*values, *ids],
        ).to_arrow_table()


def validation_workspace(folder: Path) -> str:
    """Resolve the registered workspace containing this gate's artifact root."""
    log = _open.get("log")
    if log is None:
        raise DataError(
            "the Host's validation history is unavailable", code="data_layer.history_unavailable"
        )
    rows = log.read(
        "SELECT workspace_id FROM workspace WHERE folder = ? AND status = 'active'",
        (str(folder.resolve()),),
    )
    if len(rows) != 1:
        raise DataError(
            "the validation workspace is not registered",
            code="data_layer.validation_workspace_unknown",
        )
    return str(rows[0][0])


def validation_claims(
    permits: Sequence[str], *, workspace: str | None = None
) -> dict[str, dict[str, Any]]:
    """Read a workspace's requested durable claims, refusing ambiguous permit events."""
    if not permits:
        return {}
    log, owner = _open.get("log"), workspace or _open.get("workspace")
    if log is None or owner is None:
        raise DataError(
            "the Host's validation history is unavailable", code="data_layer.history_unavailable"
        )
    marks = ", ".join("?" for _ in permits)
    rows = log.read(
        "SELECT subject_id, attrs FROM event WHERE type = ? AND workspace_id = ?"
        " AND subject_id IN (" + marks + ") ORDER BY seq",
        (_VALIDATION_CLAIM, owner, *permits),
    )
    found: dict[str, dict[str, Any]] = {}
    for permit, attrs in rows:  # each event is one permit-spending decision
        if permit in found:
            raise DataError("the permit has multiple claims", code="data_layer.validation_invalid")
        found[str(permit)] = json.loads(attrs)
    return found


def validation_claim(permit: str, *, workspace: str | None = None) -> dict[str, Any] | None:
    """Read one permit's durable claim."""
    return validation_claims([permit], workspace=workspace).get(permit)


def record_validation_claims(values: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
    """Record permit claims in one durable log unit, retaining each original claim clock."""
    if not values:
        return
    define(
        "event_type",
        _VALIDATION_CLAIM,
        "A protected validation permit was spent.",
        {
            "work": False,
            "subject": "validation_permit",
            "required": list(values[0][1]),
            "text": "none",
            "durable": True,
        },
    )
    with _once:
        log = _open.get("log")
        if log is None:
            raise DataError(
                "the Host's validation history is unavailable",
                code="data_layer.history_unavailable",
            )
        by_owner: dict[str, list[Mapping[str, Any]]] = {}
        for owner, value in values:
            by_owner.setdefault(owner, []).append(value)
        with log.unit():
            for owner, claims in by_owner.items():
                permits = [str(value["permit_hash"]) for value in claims]
                existing = validation_claims(permits, workspace=owner)
                for value in claims:
                    permit = str(value["permit_hash"])
                    prior = existing.get(permit)
                    if prior is not None and prior != dict(value):
                        raise DataError(
                            "the permit has a different claim", code="data_layer.validation_invalid"
                        )
                    if prior is None:
                        log.record(
                            _VALIDATION_CLAIM,
                            ("validation_permit", permit),
                            dict(value),
                            workspace=owner,
                            node="host",
                        )
                        existing[permit] = dict(value)


def record_validation_claim(value: Mapping[str, Any], *, workspace: str | None = None) -> None:
    """Record one protected claim."""
    owner = workspace or _open.get("workspace")
    if owner is None:
        raise DataError(
            "the Host's validation history is unavailable", code="data_layer.history_unavailable"
        )
    record_validation_claims([(str(owner), value)])
