"""Open the Host history and admit the files and parts a workspace carried."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import alphalattice.foundation.data_layer.return_series as return_series
from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.history_alpha_arrays import (
    _import_alpha_parts,
)
from alphalattice.foundation.data_layer.history_carried_results import (
    CarriedCheck,
    _import_carried_results,
)
from alphalattice.foundation.data_layer.history_knowledge import (
    CARRIED_KNOWLEDGE,
    import_carried_knowledge,
)
from alphalattice.foundation.data_layer.history_runtime import (
    _CONDITION_TEXT,
    _EVIDENCE_TEXT,
    _RETURN_SERIES_TEXT,
    EVIDENCE_TYPE,
    FACTS,
    STEPS,
    _import_kept_files,
    _lifecycle,
    _lock,
    _open,
    _remote,
    close_history,
    define,
    held,
    release_history,
    workspace_on_log,
)
from alphalattice.foundation.data_layer.reading import (
    CONDITION_METRICS,
    CONDITION_TYPE,
    CatalogReader,
    LogReader,
    reading_session,
)
from alphalattice.foundation.data_layer.root import (
    data_root,
    duckdb_session,
    enable_catalog_wal,
)


def open_history(
    workspace_id: str,
    name: str,
    folder: Path,
    root: Path | None = None,
    *,
    can_export: Callable[[], bool] | None = None,
    carried_checks: Mapping[str, CarriedCheck] | None = None,
    duckdb_threads: int | None = None,
) -> None:
    """Open this Host's history: the held log, its reading views and its workspace's row.

    Refused by name when the log is held elsewhere or does not open, and
    `data_layer.history_unavailable` when the views do not; either way the data root is released.
    The first Host sets the shared catalog's pool width for its lifetime. Opening an open history
    again records the workspace only. The results a copy carries are admitted once their owners'
    `carried_checks` prove them; a refusal releases this opening.
    """
    with _lifecycle:
        log = held(root)
        if "reading" not in _open:
            where = data_root() if root is None else root
            catalog_reader = None
            try:  # the views' DuckDB, checked here without the log, which only the reader opens
                # All cursors share these pools. Size them before publishing the owner.
                owner = duckdb_session(where, threads=duckdb_threads)
                if owner is not None:
                    with _lock:
                        _open["duckdb"] = owner
                    has_catalog = (where / "machine" / "catalog.sqlite").is_file()
                    if has_catalog:
                        enable_catalog_wal(where)
                    session = reading_session(where, log=False, owner=owner)
                    if has_catalog:
                        catalog_reader = CatalogReader(session, catalog="read_lake")
                    else:
                        session.close()
                reading = None if owner is None else LogReader(where)
            except Exception as error:  # DuckDB raises its own errors too
                close_history()  # the data root's lock goes with it, so the next start can open it
                raise DataError(
                    f"the reading views at {where} did not open ({error})",
                    code="data_layer.history_unavailable",
                ) from error
            from alphalattice.foundation.data_layer.spans import (  # lazy: the SDK, a Host's alone
                ExternalSpanProcessor,
                LogFactExporter,
                LogSpanExporter,
                logger_provider,
                tracer_provider,
            )

            exporter = LogSpanExporter(log)
            remote = _remote(ExternalSpanProcessor, can_export)
            tracer = tracer_provider(exporter, remote=remote)
            facts = LogFactExporter(exporter)
            logger = logger_provider(facts)
            with _lock:
                _open.update(
                    reading=reading, exporter=exporter, tracer=tracer, facts=facts, logger=logger
                )
                if catalog_reader is not None:
                    _open["catalog_reader"] = catalog_reader
                if remote is not None:
                    _open["remote"] = remote
        try:
            workspace = workspace_on_log(folder, name, workspace_id, root)
        except DataError:
            if not _open.get("hosts"):
                close_history()
            raise
        with _lock:
            bound = _open.get("workspace")  # an open Host's, kept if this opening is refused
            _open.update(workspace=workspace, hosts=_open.get("hosts", 0) + 1)
        import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3, Parquet

        # a copy's lanes and kept files move onto this log before anything reads them, its
        # workers and its first start included
        where = Path(_open["root"])
        arrays.import_carried(where, log, folder, workspace=workspace)
        _import_alpha_parts(where, log, folder, workspace)
        _import_kept_files(log, folder, workspace)
        spec: dict[str, Any] = {
            "storage": "none",  # the evidence stays in the workspace's store until row 8a
            "table": None,
            "measures": [],
            "keep": "while_active",
        }
        define("artifact_type", EVIDENCE_TYPE, _EVIDENCE_TEXT, spec)
        kept = {"storage": "inline", "table": None, "measures": [], "keep": "forever"}
        define("artifact_type", return_series.TYPE, _RETURN_SERIES_TEXT, kept)
        for metric, (metric_spec, words) in CONDITION_METRICS.items():
            define("metric", metric, words, metric_spec)
        condition = {"storage": "none", "table": None, "keep": "forever"}
        condition["measures"] = sorted(CONDITION_METRICS)
        define("artifact_type", CONDITION_TYPE, _CONDITION_TEXT, condition)
        for step_name, (required, words, text) in STEPS.items():
            kind = {"work": True, "subject": None, "required": required, "text": words}
            define("event_type", step_name, text, kind | {"durable": False})
        for fact_name, (about, required, words, text) in FACTS.items():
            kind = {"work": False, "subject": about, "required": required, "text": words}
            define("event_type", fact_name, text, kind | {"durable": False})
        define(
            "event_type",
            "experience.read",
            "A read of recorded trial experience.",
            {
                "work": False,
                "subject": None,
                "required": ["count"],
                "text": "none",
                "durable": False,
            },
        )
    try:  # outside the lock: its writers take the StorePort's guard
        _import_carried_results(folder, carried_checks or {})
    except BaseException:
        with _lock:
            if _open.get("hosts", 0) > 1:  # another open Host keeps the history and its workspace
                _open["workspace"] = bound
        release_history()  # this opening's reference alone
        raise


def move_carried(folder: Path, workspace_id: str) -> int:
    """A Host's admission of a workspace copy: the lanes and research inputs it carries move onto
    the log this process holds, held by the workspace, before admission reads them. Nothing when
    it carries none."""
    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet
    import alphalattice.foundation.data_layer.inputs as inputs  # lazy: BLAKE3

    lanes, bindings = (folder / arrays.CARRIED).is_dir(), (folder / inputs.CARRIED).is_dir()
    knowledge = (folder / CARRIED_KNOWLEDGE).exists(follow_symlinks=False)
    if not (lanes or bindings or knowledge):
        return 0
    log = held()
    held_id = workspace_on_log(folder, folder.name, workspace_id)
    root = Path(_open["root"])
    count = 0
    if lanes:
        count += arrays.import_carried(root, log, folder, workspace=held_id)
        _import_alpha_parts(root, log, folder, held_id)
        _import_kept_files(log, folder, held_id)
    if bindings:
        count += inputs.import_carried(root, log, folder, workspace=held_id)
    if knowledge:
        count += import_carried_knowledge(log, folder, held_id)
    return count
