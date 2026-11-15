"""StorePort: the one door to the catalog and its files, the write order, and its one repair.

The data root's files, by who names them:
    staging/<admission_id>/  one admission's Parquet before its catalog commit
    staging/<admission_id>.lock  held by its producer from before the folder exists until its
                             commit returns; staging whose lock no one holds belongs to a write
                             that will never finish, and start collects it
    data/                    DuckLake's files, named by its catalog; any other file is collected
    blobs/<artifact_id>      an artifact's body, kept while a live_ref holds it

The write order (catalog.sql's header). `stage` mints the admission's id, digests its rows and
writes their Parquet under staging. It needs no log, lock or catalog, so it runs beside the
writer, in the process that fetched the rows, and hands the writer only a commit request
(`Staged`): the digests are Python work, and in the Host's process they would hold the
interpreter while an append waits. `StorePort.commit` is the writer: one catalog snapshot inserts
the rows, writes the supersession of every world version they replace and carries in its commit
info the whole envelope of the `admission.committed` it owes; then `repair`
appends that event, durably, and the staging folder is deleted. `start` runs the same `repair`
when the Host opens, so the normal path and crash recovery are one path, and collects what no
snapshot or write owns. Every file leaves through `delete`: one another process holds open waits
in deletion_queue and is retried at each start, and nothing a live_ref holds is ever deleted.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, NamedTuple

from alphalattice.control.workspace_runtime.storage.capacity import (
    FREE_FLOOR_BYTES,
    free_floor_shortfall,
)
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import ENVELOPE, canonical, digest
from alphalattice.kernel.data_layer.log import HostLog, drop, envelope, hold, uuid7
from alphalattice.kernel.data_layer.root import load_extensions, schema_text, statements

KINDS: Final = {"staging": "staging", "blobs": "blob", "data": "file"}
"""The deletion queue's kind for each top folder a deleted file may live in."""
WORLD_KEYS: Final = {
    "listing": ("listing_id",),
    "provider_symbol": ("provider", "symbol", "valid_from"),
    "universe_member": ("universe_id", "listing_id", "valid_from"),
    "sector": ("scheme", "listing_id", "valid_from"),
    "raw_bar": ("provider", "listing_id", "session_date"),
    "corporate_action": ("provider", "listing_id", "effective_date", "kind", "action_id"),
    "session": ("market", "session_date"),
    "evidence_doc": ("doc_digest",),
}
"""Each world table catalog.sql defines, with its key less value_digest. An observed row is known
from its admission's moment: known_at and known_asof are both the log's clock."""
RESULTS_KEYS: Final = {"execution_outcome": ("artifact_id", "listing_id", "formation_session")}
"""The results tables StorePort admits, with their keys. A results row is an artifact's body,
written once: a key already in the catalog is refused."""
LEGACY_VALID: Final = {
    "listing": ("last_session", "first_session"),
    "provider_symbol": ("valid_to", "valid_from"),
    "universe_member": ("valid_to", "valid_from"),
    "sector": ("valid_to", "valid_from"),
    "raw_bar": ("session_date",),
    "corporate_action": ("effective_date",),
}
"""The world tables a legacy batch may fill, each with the date a row's lag rule counts from: the
first of these the row has, so an interval's closing version is known only from its end."""
PARTITION: Final = {"raw_bar": "session_date"}
"""The date column whose year partitions a table, as catalog.sql sets it; any other table is one
part, `all`. An admission's event names the knowledge times it brought to each part."""
METADATA: Final = "__ducklake_metadata_lake"
"""The DuckLake catalog's own tables, as the Host's session names them."""


class Staged(NamedTuple):
    """One admission's rows, digested and written under staging, waiting for its commit."""

    admission_id: str
    table: str
    provider: str
    row_count: int
    lag: tuple[str, int] | None
    """A legacy batch's lag rule, by name and version; None for an observed one."""


class StorePort:
    """The Host's one writer of the catalog; open it with `StorePort.open`."""

    def __init__(self, root: Path, log: HostLog, lake: Any):
        """Hold the data root, its log and the Host's DuckDB session; `open` makes one."""
        self.root, self.log, self.lake = root, log, lake
        self._writer = threading.Lock()
        self.started: dict[str, list[str]] = {}

    @classmethod
    def open(cls, root: Path, log: HostLog) -> StorePort:
        """Attach the catalog, the log read-only and the reading surface, then run `start`."""
        import duckdb  # lazy: DuckDB is the optional data extra; only a Host with data opens this

        lake = duckdb.connect(  # in memory: the catalog and the log are attached to it
            ":memory:",
            config={"autoinstall_known_extensions": False, "autoload_known_extensions": False},
        )
        load_extensions(lake)
        machine, data = (root / "machine").as_posix(), (root / "data").as_posix()
        for sql in (
            f"ATTACH 'ducklake:sqlite:{machine}/catalog.sqlite' AS lake"
            f" (DATA_PATH '{data}/', DATA_INLINING_ROW_LIMIT 1000)",
            f"ATTACH '{machine}/log.sqlite' AS log (TYPE SQLITE, READ_ONLY)",
            "SET TimeZone = 'UTC'",
            *statements(schema_text("views.sql")),
        ):
            lake.execute(sql).complete()
        port = cls(root, log, lake)
        port.started = start(root, log, lake)
        return port

    def commit(self, staged: Staged) -> int:
        """Commit one staged admission as one catalog snapshot, then its owed event; returns the
        snapshot. The event is durable, so it is on disk when this returns. Refused by name: a
        lag rule the log does not hold, legacy rows with no session to count from, and results
        keys already admitted."""
        with self._writer:
            lag = None if staged.lag is None else self._lag_seconds(*staged.lag)
            self._catalog_commit(staged, self.log.clock(), lag)
            repair(self.log, self.lake)
            delete(self.root, self.log, f"staging/{staged.admission_id}")
            found = self.log.read(
                "SELECT catalog_snapshot FROM admission WHERE admission_id = ?",
                (staged.admission_id,),
            )
            return int(found[0][0])

    def close(self) -> None:
        """Close the Host's DuckDB session; the log is the caller's."""
        self.lake.close()

    def _lag_seconds(self, name: str, version: int) -> int:
        """The lag a registered lag_rule version adds to a session's close, in seconds."""
        found = self.log.read(
            "SELECT spec FROM definition WHERE family = 'lag_rule' AND name = ? AND version = ?",
            (name, version),
        )
        seconds = json.loads(found[0][0]).get("lag_seconds") if found else None
        if type(seconds) is not int or seconds < 0:
            raise DataError(
                f"lag_rule {name} v{version} is not registered with whole lag_seconds of 0 or"
                " more; register the rule, then commit the batch",
                code="data_layer.lag_rule_unknown",
            )
        return seconds

    def _catalog_commit(self, staged: Staged, known: int, lag: int | None) -> None:
        """One snapshot: the supersession of each world version the rows replace, then the rows,
        carrying the event it owes."""
        table, admission = staged.table, staged.admission_id
        parquet = self.root / "staging" / admission / f"{table}.parquet"
        source = f"read_parquet('{parquet.as_posix()}')"
        world = table in WORLD_KEYS
        key = WORLD_KEYS[table] if world else RESULTS_KEYS[table]
        same = " AND ".join(f"t.{k} = n.{k}" for k in key)
        rows = self._known(table, source, known, lag) if world else f"SELECT * FROM {source}"
        target = f"lake.{'world' if world else 'results'}.{table}"
        run = self.lake.execute
        run("BEGIN").complete()
        try:
            parts = self._parts(table, rows, staged.row_count) if world else None
            if world:
                self._in_order(table, target, rows, same)
            if not world and run(f"SELECT 1 FROM {target} t, ({rows}) n WHERE {same}").fetchone():
                raise DataError(
                    f"{table} already holds rows of these keys; an artifact's rows are admitted"
                    " once",
                    code="data_layer.result_key_present",
                )
            run(
                "CALL lake.set_commit_message('host', 'admission', extra_info => ?)",
                (canonical(self._owed(staged, known, parts)).decode(),),
            ).complete()
            if world:
                run(
                    f"INSERT INTO lake.world.superseded SELECT '{table}', t.value_digest,"
                    f" t.admission_id, n.known_asof, '{admission}' FROM {target} t, ({rows}) n"
                    f" WHERE {same} AND NOT EXISTS (SELECT 1 FROM lake.world.superseded s"
                    f" WHERE s.table_name = '{table}' AND s.value_digest = t.value_digest"
                    " AND s.admission_id = t.admission_id)"
                ).complete()
                rows = f"SELECT *, false AS retracted, '{admission}' AS admission_id FROM ({rows})"
            run(f"INSERT INTO {target} BY NAME {rows}").complete()
            run("COMMIT").complete()
        except BaseException:
            run("ROLLBACK").complete()
            raise

    def _in_order(self, table: str, target: str, rows: str, same: str) -> None:
        """Refuse rows known before a version of their key already admitted: a key's versions
        are admitted in their known_asof order, so each supersedes the one before it."""
        late = self.lake.execute(
            f"SELECT count(*) FROM {target} t, ({rows}) n WHERE {same}"
            " AND n.known_asof < t.known_asof"
        ).fetchone()[0]
        if late:
            raise DataError(
                f"{late} {table} rows would be known before a version of their key already"
                " admitted; admit a key's versions in their known_asof order",
                code="data_layer.admission_out_of_order",
            )

    def _known(self, table: str, source: str, known: int, lag: int | None) -> str:
        """The staged rows with their known_at and known_asof, as a query. An observed row's are
        both `known`; a legacy row's known_at is empty and its known_asof is its publication time,
        or its lag after the close of its market's first session on or after its valid date. The
        market is the row's own listing's, known only while that listing has had one market."""
        if lag is None:
            at = f"make_timestamptz({known}::BIGINT)"
            return f"SELECT *, {at} AS known_at, {at} AS known_asof FROM {source}"
        described = self.lake.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
        columns = {row[0] for row in described}
        valid = ", ".join(f"n.{c}" for c in LEGACY_VALID[table] if c in columns)
        derived = f"a.close_utc + to_seconds({lag})"
        if "published_at" in columns:
            derived = f"coalesce(CAST(n.published_at AS TIMESTAMPTZ), {derived})"
        market, joined = "n.market", ""
        if table != "listing":
            market = "l.market"
            joined = (  # a listing that changed market has no market for a date before the change
                " LEFT JOIN (SELECT listing_id, any_value(market) AS market FROM lake.world.listing"
                " WHERE NOT retracted GROUP BY listing_id HAVING count(DISTINCT market) = 1) l"
                " ON l.listing_id = CAST(n.listing_id AS UUID)"
            )
        return (
            f"SELECT n.*, NULL::TIMESTAMPTZ AS known_at, {derived} AS known_asof FROM {source} n"
            f"{joined} ASOF LEFT JOIN {_current('session')} a ON a.market = {market}"
            f" AND CAST(coalesce({valid}) AS DATE) <= a.session_date"
        )

    def _parts(self, table: str, rows: str, count: int) -> list[dict[str, Any]]:
        """Each part the rows fall in, with the range of their known_asof there; refused by name
        when a legacy row found no session to count its lag from."""
        column = PARTITION.get(table)
        part = f"CAST(year(CAST({column} AS DATE)) AS VARCHAR)" if column else "'all'"
        found = self.lake.execute(
            f"SELECT {part} AS part, min(epoch_us(known_asof)), max(epoch_us(known_asof)),"
            f" count(known_asof) FROM ({rows}) GROUP BY ALL ORDER BY part"
        ).fetchall()
        unknown = count - sum(row[3] for row in found)
        if unknown:
            raise DataError(
                f"{unknown} legacy {table} rows have no session close to count from: their listing"
                " is not in world.listing, or has had more than one market so its market then is"
                " unknown, or that market has no session on or after their valid date",
                code="data_layer.legacy_anchor_missing",
            )
        return [
            {"part": p, "min_known_asof_us": lo, "max_known_asof_us": hi} for p, lo, hi, _ in found
        ]

    def _owed(
        self, staged: Staged, known: int, parts: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        """The `admission.committed` event the catalog commit carries."""
        whole = [{"part": "all", "min_known_asof_us": known, "max_known_asof_us": known}]
        attrs: dict[str, Any] = {
            "provider": staged.provider,
            "dataset": staged.table,
            "row_count": staged.row_count,
            "known_us": known,
            "parts": parts or whole,
        }
        if staged.lag is not None:
            attrs |= {"lag_rule": staged.lag[0], "lag_rule_version": staged.lag[1]}
        return dict(
            envelope(
                "admission.committed",
                ("admission", staged.admission_id),
                attrs,
                actor=("host", "store"),
                node="store",
                occurred_us=known,
            )
        )


@contextmanager
def stage(
    root: Path,
    table: str,
    provider: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    lag: tuple[str, int] | None = None,
    floor_bytes: int = FREE_FLOOR_BYTES,
) -> Iterator[Staged]:
    """Digest one admission's rows and write them under staging; the commit request it yields is
    all the writer needs. Its producer holds the admission's lock until the block ends, so a Host
    that starts meanwhile leaves it: commit it inside the block. A legacy batch, rows 0.2.7 kept
    with no knowledge time, names its source's lag rule as `lag=(name, version)`. Refused by name
    below the free-space floor, and as `digested` refuses."""
    short = free_floor_shortfall(root, floor_bytes)
    if short:
        raise DataError(
            f"the data root's disk is {short} bytes below its {floor_bytes}-byte free-space floor,"
            " kept for the log, its repair and cleanup; free space on that disk first",
            code="data_layer.below_free_floor",
        )
    admission = uuid7()  # first: the rows and the staging folder carry it
    staged = digested(table, provider, rows, legacy=lag is not None)
    import pyarrow as pa  # lazy: the optional data extra, as DuckDB
    import pyarrow.parquet as pq  # lazy: as pyarrow

    lock = root / "staging" / f"{admission}.lock"
    held = hold(lock)  # before the folder, so no start sees the folder unheld
    if held is None:
        raise DataError(
            "a starting Host collected this admission's staging before its producer held it;"
            " stage the rows again",
            code="data_layer.staging_taken",
        )
    try:
        (root / "staging" / admission).mkdir()
        pq.write_table(
            pa.table(as_columns(staged)), root / "staging" / admission / f"{table}.parquet"
        )
        yield Staged(admission, table, provider, len(staged), lag)
    finally:
        drop(held, lock)


def digested(
    table: str, provider: str, rows: Iterable[Mapping[str, Any]], *, legacy: bool = False
) -> list[dict[str, Any]]:
    """Each world row with its value_digest, the digest of the row with its provider where its
    table keys one; a results row as it is.

    Refused by name: a table StorePort does not admit, a legacy batch for a table with no valid
    date to count a lag from, and two rows of one key, since DuckLake keeps no keys.
    """
    key = WORLD_KEYS.get(table) or RESULTS_KEYS.get(table)
    if key is None:
        raise DataError(
            f"{table} is not a table StorePort admits; it admits"
            f" {', '.join([*WORLD_KEYS, *RESULTS_KEYS])}",
            code="data_layer.world_table_not_admitted",
        )
    if legacy and table not in LEGACY_VALID:
        raise DataError(
            f"{table} rows have no valid date a lag rule counts from; admit them as observed",
            code="data_layer.legacy_table_unanchored",
        )
    staged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for given in rows:
        row = {**given, "provider": provider} if "provider" in key else dict(given)
        values = tuple(row[k] for k in key)
        if values in seen:
            raise DataError(
                f"{table} {dict(zip(key, values, strict=True))} appears twice in one admission;"
                " send one version of a key",
                code="data_layer.admission_key_repeated",
            )
        seen.add(values)
        staged.append(
            row | {"value_digest": digest(canonical(row))} if table in WORLD_KEYS else row
        )
    return staged


def as_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """The rows as columns: every column any row has, empty where a row lacks it. Arrow's own
    `from_pylist` takes its columns from the first row and would drop a later row's field."""
    names = dict.fromkeys(name for row in rows for name in row)
    return {name: [row.get(name) for row in rows] for name in names}


def repair(log: HostLog, lake: Any) -> list[str]:
    """Append each event a catalog snapshot carried and the log lacks; returns their ids.

    An owed event keeps every field its writer made, its id, version, workspace, actor, trace and
    time included; only its recording time is new, after the moment it carries, and its
    attributes name the snapshot. Running it again adds none.
    """
    carried = lake.execute(
        "SELECT snapshot_id, commit_extra_info FROM lake.snapshots()"
        " WHERE commit_extra_info IS NOT NULL ORDER BY snapshot_id"
    ).fetchall()
    added = []
    for snapshot, info in carried:
        owed = json.loads(info)
        if log.read("SELECT 1 FROM event WHERE event_id = ?", (owed["event_id"],)):
            continue
        attrs = json.loads(owed["attrs"])
        row = {column: owed[column] for column in ENVELOPE if column in owed and column != "seq"}
        row["attrs"] = canonical(attrs | {"catalog_snapshot": snapshot}).decode()
        log.clock(at_least=attrs.get("known_us", 0))
        log.append(row)
        added.append(owed["event_id"])
    return added


def delete(root: Path, log: HostLog, handle: str) -> bool:
    """Delete the file or folder at `handle` (a path under the data root); True once it is gone.

    A body a live_ref holds is refused by name. One another process holds open goes to
    deletion_queue with its error, to be retried; False then.
    """
    path, kind = _resolve(root, handle)
    if kind == "blob" and log.read("SELECT 1 FROM live_ref WHERE artifact_id = ?", (path.name,)):
        raise DataError(
            f"{handle} is held by a live reference and is never collected",
            code="data_layer.live_ref_held",
        )
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError as error:
        log.queue_deletion(handle, kind, f"{type(error).__name__}: {error.strerror or error}")
        return False
    log.dequeue_deletion(handle)
    return True


def start(root: Path, log: HostLog, lake: Any) -> dict[str, list[str]]:
    """The Host's start: repair the log from the catalog, then collect. Returns the events it
    repaired, by id, and what it collected and what still waits, by handle.

    It retries the deletion queue, collects the staging of every admission no producer holds,
    holding that admission's lock itself until its staging and lock are gone, so no producer can
    take it meanwhile, and collects every file under data/ the catalog does not name.
    """
    report: dict[str, list[str]] = {"repaired": repair(log, lake), "collected": [], "waiting": []}
    queued = [row[0] for row in log.read("SELECT handle FROM deletion_queue ORDER BY handle")]
    staging, data = root / "staging", root / "data"
    admissions = {Path(handle).name for handle in queued if handle.startswith("staging/")}
    if staging.is_dir():
        admissions |= {child.name.removesuffix(".lock") for child in staging.iterdir()}
    for admission in sorted(admissions):
        lock, handle = staging / f"{admission}.lock", f"staging/{admission}"
        probe = hold(lock)
        if probe is None:
            continue  # its producer is staging it or waiting for its commit
        try:
            if (staging / admission).exists() or handle in queued:
                report["collected" if delete(root, log, handle) else "waiting"].append(handle)
        finally:
            drop(probe, lock)
    named = _named_files(lake)
    handles = [handle for handle in queued if not handle.startswith("staging/")]
    handles += sorted(
        f"data/{(folder / name).relative_to(data).as_posix()}"
        for folder, _, names in data.walk(on_error=_unreadable)
        for name in names
        if name not in named
    )
    for handle in dict.fromkeys(handles):
        report["collected" if delete(root, log, handle) else "waiting"].append(handle)
    log.flush()
    return report


def _current(table: str) -> str:
    """A world table's current versions, as a subquery: those not withdrawn and not replaced."""
    return (
        f"(SELECT * FROM lake.world.{table} x WHERE NOT x.retracted AND NOT EXISTS (SELECT 1"
        f" FROM lake.world.superseded s WHERE s.table_name = '{table}'"
        " AND s.value_digest = x.value_digest AND s.admission_id = x.admission_id))"
    )


def _named_files(lake: Any) -> frozenset[str]:
    """The name of every file the catalog names: data, deletes, and those DuckLake will clean up.
    DuckLake names each file by a fresh UUIDv7, so a name is unique under data/."""
    rows = lake.execute(
        " UNION ALL ".join(
            f"SELECT path FROM {METADATA}.ducklake_{table}"
            for table in ("data_file", "delete_file", "files_scheduled_for_deletion")
        )
    ).fetchall()
    return frozenset(Path(row[0]).name for row in rows)


def _unreadable(error: OSError) -> None:
    """A folder under data/ that cannot be listed stops the start: read as empty, its unnamed
    files would stay unseen."""
    raise DataError(
        f"{error.filename} under the data root's data/ cannot be listed ({error.strerror});"
        " give the Host read access to it and start again",
        code="data_layer.data_folder_unreadable",
    ) from error


def _resolve(root: Path, handle: str) -> tuple[Path, str]:
    """The path a handle names and its deletion kind; refused unless it lies in a known folder."""
    parts = Path(handle).parts
    kind = KINDS.get(parts[0]) if len(parts) > 1 else None
    if kind is None or ".." in parts or Path(handle).is_absolute():
        raise DataError(
            f"{handle} names no file under staging/, data/ or blobs/",
            code="data_layer.handle_outside_root",
        )
    return root.joinpath(*parts), kind
