"""Carry the execution outcomes and Portfolio results a workspace keeps into its copy."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.history_execution_outcomes import (
    read_execution_outcome,
    write_execution_outcome,
)
from alphalattice.foundation.data_layer.history_portfolio_results import (
    read_portfolio_result,
    write_portfolio_result,
)
from alphalattice.foundation.data_layer.history_runtime import (
    _array_reader,
    _open,
    flush,
    holding,
    own_root,
)
from alphalattice.foundation.data_layer.root import data_root

CARRIED_RESULTS: Final = Path("runtime") / "carried-results"
"""Where a workspace's copy carries the catalog results its workspace keeps, one Parquet file per
result under its type's folder, its profile in the file's metadata."""

_CARRIED_TABLES: Final = {
    "execution_outcome": None,
    "portfolio_benchmark": "series",
    "portfolio_execution_series": "series",
    "portfolio_report_holding": "holding",
}
_CARRIED_PROFILE: Final = b"alphalattice.profile"

type CarriedCheck = Callable[[Path, str, Any, Any], object]
"""A result owner's proof that a carried body is the one its digest names, given the copy's
workspace folder, the digest, the carried profile and rows; it raises when it is not."""


def carry_results(workspace: Path, copy: Path, *, root: Path | None = None) -> int:
    """Write the execution outcomes and Portfolio results the catalog (`root`'s when named) keeps
    for `workspace` into its copy `copy` (`CARRIED_RESULTS`); the Host that opens the copy
    (`open_history`) admits them onto its own catalog. Returns the results carried."""
    where = data_root() if root is None else root
    log = holding()
    if log is not None and Path(_open["root"]) == Path(where):
        return _carry_results(workspace, copy)
    if not (where / "machine" / "log.sqlite").is_file():
        return 0
    with own_root(where):
        return _carry_results(workspace, copy)


def _carry_results(workspace: Path, copy: Path) -> int:
    import pyarrow.parquet as pq  # lazy: the data extra

    with _array_reader() as (_root, read):
        known = read(
            "SELECT workspace_id FROM workspace WHERE folder = ? AND status = 'active'",
            (str(workspace.resolve()),),
        )
        rows = (
            read(
                "SELECT d.name, a.content_digest FROM live_ref l"
                " JOIN artifact a ON a.artifact_id = l.artifact_id"
                " JOIN definition_name d ON d.definition_id = a.type_id"
                " WHERE l.root_kind = 'workspace' AND l.root_id = ? AND a.body_state = 'present'"
                " AND d.family = 'artifact_type' AND d.name IN (?, ?, ?, ?)"
                " ORDER BY d.name, a.content_digest",
                (str(known[0][0]), *_CARRIED_TABLES),
            )
            if known
            else []
        )
    for kind, digest in rows:
        result_table = _CARRIED_TABLES[kind]
        if result_table is None:
            profile, table = None, read_execution_outcome(digest)
        else:
            profile, table = read_portfolio_result(kind, digest, result_table=result_table)
        metadata = {_CARRIED_PROFILE: json.dumps(profile, sort_keys=True).encode()}
        target = copy / CARRIED_RESULTS / kind / f"{digest}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        pq.write_table(table.replace_schema_metadata(metadata), staged, compression="zstd")
        os.replace(staged, target)
    return len(rows)


def carried_result_files(copy: Path) -> list[Path]:
    """The result files `copy` carries (`carry_results`), in order, each under its type's folder."""
    carried = copy / CARRIED_RESULTS
    if not carried.is_dir():
        return []
    return sorted(path for kind in carried.iterdir() for path in kind.iterdir())


def _import_carried_results(folder: Path, checks: Mapping[str, CarriedCheck]) -> None:
    """Admit the results a copy carries (`carry_results`) onto this Host's catalog, held by its
    workspace, each once its owner's check proves its rows are the ones its digest names; the
    files go once the log has committed every admission. A file of a type the carry does not
    write, not named by a digest, with no check here, or whose check refuses it, is refused by
    name."""
    files = carried_result_files(folder)
    if not files:
        return
    import pyarrow.parquet as pq  # lazy: the data extra

    for path in files:
        kind, digest = path.parent.name, path.stem
        named = len(digest) == 64 and not set(digest) - set("0123456789abcdef")
        check = checks.get(kind)
        if kind not in _CARRIED_TABLES or path.suffix != ".parquet" or not named or check is None:
            raise DataError(
                f"the copy carries {path.name} under {kind}, which no result here is",
                code="data_layer.carried_result_invalid",
            )
        table = pq.read_table(path)
        profile = json.loads((table.schema.metadata or {}).get(_CARRIED_PROFILE, b"null"))
        body = table.replace_schema_metadata(None)
        try:
            check(folder, digest, profile, body)
        except Exception as error:  # each owner refuses by its own error
            raise DataError(
                f"the copy carries {path.name} under {kind}, whose rows it does not name ({error})",
                code="data_layer.carried_result_invalid",
            ) from error
        result_table = _CARRIED_TABLES[kind]
        if result_table is None:
            write_execution_outcome(digest, body)
        else:
            write_portfolio_result(kind, digest, profile, body, result_table=result_table)
    flush()  # the admissions and holds are durable before their files go, the log's guard alone
    for path in files:
        path.unlink()
